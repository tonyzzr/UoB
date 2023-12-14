'''
  For helper functions involved in point clouds.
'''
import torch
from torch.nn.functional import avg_pool2d

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

from dataclasses import dataclass

from imageio import imread
from sklearn.neighbors import KernelDensity

from geomloss import SamplesLoss


'''
  Geomloss helper functions.
'''

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

def grid(W):
    x, y = torch.meshgrid([torch.arange(0.0, W).type(dtype) / W] * 2, indexing="xy")
    return torch.stack((x, y), dim=2).view(-1, 2)


def as_measure(img, size):
    __img = cv2.resize(img, (size, size))
    weights = torch.from_numpy(__img).type(dtype)
    sampling = weights.shape[0] // size
    weights = (
        avg_pool2d(weights.unsqueeze(0).unsqueeze(0), 
                   sampling).squeeze(0).squeeze(0)
    )
    weights = weights / weights.sum()

    samples = grid(size)
    return weights.view(-1), samples

def migrate(measure, N, n_epoch):
    x_i = grid(N).view(-1, 2)
    a_i = (torch.ones(N * N) / (N * N)).type_as(x_i)

    x_i.requires_grad = True
    Loss = SamplesLoss("sinkhorn", blur=0.01, scaling=0.9)
    (b_j, y_j) = measure

    __x_i = x_i
    for epoch in range(n_epoch):
        L_ab = Loss(a_i, __x_i, b_j, y_j)
        [g_i] = torch.autograd.grad(L_ab, [__x_i])
        __x_i = __x_i - g_i / a_i.view(-1, 1)

    return __x_i.detach().clone()



'''
  Point cloud helper functions.
'''
@dataclass
class PointCloud:
  coord: np.ndarray
  
  @property
  def shape(self):
    return self.coord.shape

  def __str__(self):
    return f"PointCloud(shape = {self.shape})"

  def __repr__(self):
    return self.__str__()

  def plot(self, ax = None):
    if ax is None:
      fig, ax = plt.subplots(1, 1, figsize=(2, 2))
      
    ax.scatter(self.coord[:, 0], self.coord[:, 1], s=3, alpha=0.2)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.invert_yaxis()
    
  def to_tensor(self, three_d = True, homogeneous = True):
    t_pc = torch.tensor(self.coord)

    if three_d:
      t_pc = torch.cat([t_pc, torch.zeros(t_pc.shape[0], 1)], dim=1)

    if homogeneous:
      return torch.cat([t_pc, torch.ones(t_pc.shape[0], 1)], dim=1)
    else:
      return t_pc

# ---

def img2pc(img:np.ndarray, N = 16) -> PointCloud:
  measure = geo.as_measure(img, N)
  pc = geo.migrate(measure, N, n_epoch=2).cpu().numpy()

  return PointCloud(coord = pc)

def seg2pcs(seg:np.ndarray, N = 16) -> list[PointCloud]:
  classes = np.unique(seg)
  fg_classes = classes[~np.isnan(classes)]
  pcs = []

  for c in fg_classes:
    mask = seg == c
    mask = mask.astype(np.uint8)
    
    measure = geo.as_measure(mask, N)
    pc = geo.migrate(measure, N, n_epoch=2).cpu().numpy()
    pcs.append(PointCloud(coord = pc))

  return pcs

def img_seg2pcs(img:np.ndarray, seg:np.ndarray, N = 16) -> PointCloud:
  classes = np.unique(seg)
  fg_classes = classes[~np.isnan(classes)]
  pcs = []

  for c in fg_classes:
    mask = seg == c # add a gaussian conv here?
    masked_img = mask.astype(np.uint8) * img[..., 0]
    
    measure = geo.as_measure(masked_img, N)
    pc = geo.migrate(measure, N, n_epoch=2).cpu().numpy()
    pcs.append(PointCloud(coord = pc))

  return pcs


# --- 


def create_point_clouds_dataframe(seg_data, N = 32):
  '''
    low freq tx only
  '''
  for i in trange(8):
    pcs_dict = {}

    # img point cloud
    pcs_dict["img_pc"] = [img2pc(seg_data.lf_imgs[i][..., 0], N = N)]
    
    # seg point cloud
    seg_pcs = seg2pcs(seg_data.lf_segs[i], N = N)
    for j in range(len(seg_pcs)):
      pcs_dict[f"seg_pc{j}"] = [seg_pcs[j]]

    # img-seg point cloud
    img_seg_pcs = img_seg2pcs(seg_data.lf_imgs[i], seg_data.lf_segs[i], N = N)
    for j in range(len(img_seg_pcs)):
      pcs_dict[f"img_seg_pc{j}"] = [img_seg_pcs[j]]

    # create and concatenate
    if i == 0:
      pcs_df = pd.DataFrame(pcs_dict)
    else:
      pcs_df = pd.concat([pcs_df, pd.DataFrame(pcs_dict)], ignore_index=True)


  return pcs_df

def plot_point_clouds_dataframe(pcs_df: pd.DataFrame, alpha=0.05):
  '''
    low freq tx only
  '''

  cols = pcs_df.head(0).to_dict().keys()

  fig, ax = plt.subplots(3, 8, figsize=(8, 3))
  for i in range(pcs_df.shape[0]):
    for j, c in enumerate(cols):
      pc = pcs_df.loc[i, c]

      if j in range(1, 4):
        axis = ax[1, i]
      elif j in range(4, 7):
        axis = ax[2, i]
      else:
        axis = ax[0, i]

      axis.plot(pc.coord[:, 0], pc.coord[:, 1], '.', alpha=alpha)
      axis.set_xlim([0, 1])
      axis.set_ylim([0, 1])
      axis.invert_yaxis()
      axis.set_xticks([])
      axis.set_yticks([])

  plt.show()
# ---
