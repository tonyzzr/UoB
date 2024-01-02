import torch
from torch import nn
import numpy as np
import pypose as pp
from tqdm import tqdm

import matplotlib.pyplot as plt
from geomloss import SamplesLoss

from ..data.tmp_seg_data import SegData
from ..data.tmp_pcs import create_point_clouds_dataframe, plot_point_clouds_dataframe
from .rela_pose_est import estimate_relative_pose

# ------ #

class RegNet(nn.Module):

    def __init__(self, angle=0,):
        super().__init__()

        self.theta = torch.nn.Parameter(data = torch.tensor(angle/180*np.pi),
                                        requires_grad=True)

    def forward(self, pc1, trans_pos):
        pose_mat = self.theta2pos_mat(self.theta, trans_pos)
        return pose_mat @ pc1
    
    def criterion(self, pc0, pc1):
        w_dist = SamplesLoss("sinkhorn", p=1, blur=1e-5)
        return w_dist(pc0, pc1)

    def theta2pos_mat(self, theta:torch.Tensor, trans_pos:torch.Tensor) -> torch.Tensor:

        theta_vec = torch.tensor([0., 0., 1.]) * theta

        # the initial pose is composed by two parts
        # 1. translational movement (to align the right edge of array 0, and left edge of array 1)
        # 2. rotation (rotate around the right edge of array 0)


        # 1. translational movement 
        # -- on x-dim, orientation=right, distance=trans_pos[1, 0]-trans_pos[0, 0]
        poses = []

        dist = trans_pos[1, 0] - trans_pos[0, 0]
        # print(dist)
        poses.append(pp.SE3([dist, 0, 0, 0, 0, 0, 1]))


        # 2. rotation
        # -- center = (trans_pos[1, 0], 0, 0), angle=theta

        # decompose to T(x,y)*R(theta)*T(-x, -y) (P), from right to left:
        # T(-x, -y)
        poses.append(pp.SE3([-trans_pos[1, 0], -trans_pos[1, 1], 0, 0, 0, 0, 1])) 

        # R(theta)
        rot_pose = pp.so3(theta_vec).Exp()
        rot_se3_vec = torch.cat([torch.tensor([0, 0, 0,]), rot_pose.tensor()])
        poses.append(pp.SE3(rot_se3_vec)) 

        # T(x, y)
        poses.append(pp.SE3([trans_pos[1, 0], trans_pos[1, 1], 0, 0, 0, 0, 1])) # T(x, y)

        # 3. finally multiply all matrix together
        # firstly is the translation matrix
        # secondly is the rotation matrices - 1. T(-x, -y); 2. R; 3. T(x, y)
        pose_mat = pp.identity_SE3().matrix()
        for pose in poses:
          pose_mat = pose.matrix() @ pose_mat
          # print(pose_mat)

        return pose_mat


def estimate_relative_pose(data, config):
  # configs
  model, optimizer = config['model'], config['optimizer']
  n_epochs = config['n_epochs']
  device = config['device']

  # depackage data
  pc0, pc1 = data['pc0'], data['pc1']
  if data['consider_gap']:
    rela_trans_pos = data['rela_trans_pos']
  else:
    rela_trans_pos = data['rela_trans_pos_gap']
  
  pc0 = pc0.to(device)
  pc1 = pc1.to(device)

  # loop
  log = {
      'loss': [],
      'theta': [],
      'pc1_star': [],
  }
  for epoch in range(n_epochs):
    optimizer.zero_grad()
    pc1_star = model.forward(pc1, rela_trans_pos)

    loss = model.criterion(pc0, pc1_star)
    loss.backward(retain_graph=True)
    optimizer.step()

    log['loss'].append(loss.detach().item()) 
    log['theta'].append(model.theta.detach().item())
    log['pc1_star'].append(pc1_star.detach().clone())

  return log


def plot_overlapped_pc(pc0, pc1, ax=None,
                       rela_trans_pos=None, 
                       rela_trans_pos_gap=None,):
  pc1_star = pc1.detach()

  if ax is None:
    fig, ax = plt.subplots(1,1,figsize=(2, 2))

  ax.plot(pc0[0, :], pc0[1, :], '.', alpha = 0.3, label='pc0')
  ax.plot(pc1_star[0, :], pc1_star[1, :], '.',  alpha = 0.3, label='pc1_star')

  if rela_trans_pos is not None:
    ax.plot(rela_trans_pos[0, 0], rela_trans_pos[0, 1], 'rx', alpha = 1)
    ax.plot(rela_trans_pos[1, 0], rela_trans_pos[1, 1], 'rx', alpha = 1)

  if rela_trans_pos_gap is not None:
    ax.plot(rela_trans_pos_gap[0, 0], rela_trans_pos_gap[0, 1], 'gx', alpha = 1)
    ax.plot(rela_trans_pos_gap[1, 0], rela_trans_pos_gap[1, 1], 'gx', alpha = 1)
  

  ax.set_xlim([-0.2, 1.2])
  ax.set_ylim([-0.2, 1.2])

  ax.invert_yaxis()
  ax.set_xticks([])
  ax.set_yticks([])


def plot_point_cloud_reg_log(log, pc0, r=5, c=5):
  fig, ax = plt.subplots(r, c, figsize=(c*1.2, r*1.2))

  for i in range(r):
    for j in range(c):
      n = i * c + j
      idx = n * len(log['pc1_star']) // (r * c)

      pc1 = log['pc1_star'][idx]
      plot_overlapped_pc(pc0, pc1, ax=ax[i, j])

      ax[i, j].annotate(f'{idx}', xytext=(-0.1, 0.05), xy=(0, 0))
  plt.show()


# ------ #


class RelativePoseEstimation:
  def __init__(self, mvbsegs, config):
    self.mvbsegs = mvbsegs
    self.config = config

    N = self.config['N']

    self.seg_data = self._multi_view_bmode_seg_to_seg_data()
    self.pcs_df = create_point_clouds_dataframe(self.seg_data, N=N)

  def _get_rela_trans_pos(self, ):
    mvbseg = self.mvbsegs['lftx']

    h, w = mvbseg.view_images[0].shape
    x0, y0 = mvbseg.origin
    x1, y1 = x0 + mvbseg.aperture_size, y0

    trans_pos = torch.tensor(
        [[x0/w, y0/h],
         [x1/w, y1/h],]
    )

    return trans_pos

  def _multi_view_bmode_seg_to_seg_data(self, ):
    mvbsegs = self.mvbsegs

    mat_directory = mvbsegs['lftx'].mat_source_file
    frame_index = None

    lf_imgs = [
        mvbsegs['lftx'].view_images.unsqueeze(-1)[i, ...].numpy() for i in range(mvbsegs['lftx'].n_view)
    ]
    lf_segs = [
        mvbsegs['lftx'].seg_masks[i, ...].numpy() for i in range(mvbsegs['lftx'].n_view)
    ]

    hf_imgs = [
        mvbsegs['hftx'].view_images.unsqueeze(-1)[i, ...].numpy() for i in range(mvbsegs['hftx'].n_view)
    ]
    hf_segs = [
        mvbsegs['hftx'].seg_masks[i, ...].numpy() for i in range(mvbsegs['hftx'].n_view)
    ]

    lf_pad = None
    hf_pad = None


    return SegData(
        mat_directory=mat_directory,
        frame_index=frame_index,
        lf_imgs=lf_imgs,
        lf_segs=lf_segs,
        hf_imgs=hf_imgs,
        hf_segs=hf_segs,
        lf_pad=lf_pad,
        hf_pad=hf_pad
    )

  def _plot_point_clouds_dataframe(self, ):
    plot_point_clouds_dataframe(self.pcs_df, alpha=0.005)
    return

  def run_single_pose_estimation(self, view_index=0, pc_tag='img_seg_pc2'):
    assert view_index < self.pcs_df.shape[0] - 1
    assert pc_tag in self.pcs_df.columns

    pcs_df = self.pcs_df

    pc0 = pcs_df.iloc[view_index][pc_tag].to_tensor(three_d=True, homogeneous=True).T
    pc1 = pcs_df.iloc[view_index+1][pc_tag].to_tensor(three_d=True, homogeneous=True).T
    data = {
        'pc0': pc0,
        'pc1': pc1,
        'rela_trans_pos': self._get_rela_trans_pos(),
        'rela_trans_pos_gap': self._get_rela_trans_pos(),
        'consider_gap': False,
    }

    log = estimate_relative_pose(data, self.config)

    theta_star = log['theta'][-1]

    return theta_star

  def run_all_pose_estimation(self, pc_tag='img_seg_pc2',):
    n_view = self.mvbsegs['lftx'].n_view
    thetas = np.zeros((n_view, ))

    for view_index in tqdm(range(0, n_view-1)):
      thetas[view_index+1] = self.run_single_pose_estimation(view_index, pc_tag)

    notes = {
        'pc_tag': pc_tag,
        'mat_source_file': self.mvbsegs['lftx'].mat_source_file,
        'pose_est_config': self.config,
    }

    return RelativePoses(
        thetas = torch.tensor(thetas),
        notes = notes,
      )
