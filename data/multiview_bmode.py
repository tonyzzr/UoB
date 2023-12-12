'''
  This part of code should be placed in 'data/bmode.py'.
'''
import os
import cv2
import glob
import shutil

from dataclasses import dataclass, field

import torch
import torchvision
import torch.nn.functional as F

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from ipywidgets import interactive, IntSlider

from tqdm import tqdm

from .mat import MatData # change to relative path .mat later in github
from .bmode import Bmode


@dataclass
class MultiViewBmode:

  n_view: int
  image_shape: tuple
  origin: tuple
  aperture_size: float

  view_images: torch.Tensor
  view_masks: torch.Tensor

  mat_source_file: str
  bmode_config_file: str

  def zero_pad_2d(self, padding:tuple):
    '''
      padding: (padding_left, padding_right, padding_top, padding_bottom)
    '''
    padding_left, padding_right, padding_top, padding_bottom = padding

    # 1 - update views and view masks (torch.nn.ZeroPad2d)
    zero_pad = torch.nn.ZeroPad2d(padding)
    self.view_images = zero_pad(self.view_images)
    self.view_masks = zero_pad(self.view_masks)

    # 2 - update image_shape and origin
    _, h, w = self.view_images.size()
    self.image_shape = (h, w)

    x0, y0 = self.origin
    self.origin = (x0 + padding_left, 
                   y0 + padding_top)

    return

  def resize(self, size:tuple):
    '''
      size: (h, w)
    '''

    # 1 - update views and view masks (torchvision.transforms.Resize)

    # 2 - update image_shape, origin, aperture_size


    pass



# ------ #
class Bmode2MultiViewBmode:

  def __init__(self, b_mode:Bmode, ):
    self.b_mode = b_mode

  def convert(self, frame_ind = 0, matfile_path=None, bmode_config_path=None):
    n_frame, n_view, h, w = self.b_mode.b_img_seq.shape

    origin_w = self.b_mode.trans_pos[0].left_edge_coord[0, 0]
    origin_h = self.b_mode.trans_pos[0].left_edge_coord[1, 0]

    aperture_size = self.b_mode.trans_pos[0].right_edge_coord[0, 0] - \
                    self.b_mode.trans_pos[0].left_edge_coord[0, 0]

    view_images = torch.tensor(self.b_mode.b_img_seq[frame_ind, :, :, :])
    view_masks = torch.tensor(self.b_mode.mask_seq[0, :, :, :])

    mat_source_file = matfile_path
    bmode_config_file = bmode_config_path

    return MultiViewBmode(
        n_view = n_view,
        image_shape = (h, w),
        origin = (origin_w, origin_h),
        aperture_size = aperture_size,

        view_images = view_images,
        view_masks = view_masks,

        mat_source_file = mat_source_file,
        bmode_config_file = bmode_config_file,
    )


def plot_image_and_transducer_positions(mvbs:dict,):
  for key in mvbs.keys():
    n_view = mvbs[key].n_view
    origin = mvbs[key].origin
    aperture_size = mvbs[key].aperture_size

    # transducer element coordinates
    x0, y0 = origin
    x1, y1 = x0 + aperture_size, y0
    transducer_element_coordinates_x = np.linspace(x0, x1, 32)
    transducer_element_coordinates_y = np.linspace(y0, y1, 32)

    # mvbs[key].transducer_element_coordinates_x = transducer_element_coordinates_x
    # mvbs[key].transducer_element_coordinates_y = transducer_element_coordinates_y

    print(mvbs[key].mat_source_file)
    fig, ax = plt.subplots(1, n_view, figsize=(24, 3))

    for i in range(n_view):
      ax[i].imshow(mvbs[key].view_images[i, ...])
      ax[i].axis('off')
      # ax[i].scatter(x0, y0, c='r')
      # ax[i].scatter(x1, y1, c='r')
      ax[i].scatter(transducer_element_coordinates_x,
                    transducer_element_coordinates_y, c='r', s=1);

    plt.show()

# ------ #
@dataclass
class MultiViewBmodeSeg(MultiViewBmode):
  # new attributes
  n_class: int = field(default=0, init=False)
  seg_masks: torch.Tensor = field(default=None, init=False)
  seg_configs: torch.Tensor = field(default=None, init=False)
  
  def __post_init__(self, ):
    pass

  # add a __str__ method here

  def zero_pad_2d(self, padding:tuple):
    '''
      padding: (padding_left, padding_right, padding_top, padding_bottom)
    '''
    padding_left, padding_right, padding_top, padding_bottom = padding

    # 1 - update views and view masks (torch.nn.ZeroPad2d)
    zero_pad = torch.nn.ZeroPad2d(padding)
    self.view_images = zero_pad(self.view_images)
    self.view_masks = zero_pad(self.view_masks)

    # add padding of segmentation masks -- pad by NaN?
    nan_pad = torch.nn.ConstantPad2d(padding, float('nan'))
    self.seg_masks = nan_pad(self.seg_masks)

    # 2 - update image_shape and origin
    _, h, w = self.view_images.size()
    self.image_shape = (h, w)

    x0, y0 = self.origin
    self.origin = (x0 + padding_left,
                  y0 + padding_top)

    return

  def resize(self, size:tuple):
    '''
      size: (height, width)
    '''
    h, w = size

    raise NotImplementedError

  def save_view_images(self, dir):
    assert dir[-1] == '/'

    for i in range(self.n_view):
      file_path = dir + f'view_{i}.png'

      # rescale data to image
      image_to_save = self.view_images[i, ...]
      image_to_save /= image_to_save.max()

      # save image -- torchvision.utils.save_image can only deal with pixel value within (0, 1)
      torchvision.utils.save_image(image_to_save,
                                   file_path)

    return


def create_cmap(num_parts):
    color_list = ["red", "yellow", "blue", "lime", "darkviolet", "magenta", "cyan", "brown", "yellow"]
    cmap = 'jet' if num_parts > 10 else ListedColormap(color_list[:num_parts])
    return cmap

def plot_image_and_segmentation_masks(mvbsegs:dict,):
  for key in mvbsegs.keys():
    n_view = mvbsegs[key].n_view
    cmap = create_cmap(mvbsegs[key].n_class)

    fig, ax = plt.subplots(1, n_view, figsize=(24, 3))
    for i in range(n_view):
      ax[i].imshow(mvbsegs[key].view_images[i, ...], cmap = 'gray')
      ax[i].imshow(mvbsegs[key].seg_masks[i, ...], cmap = cmap, alpha = 0.5)
      ax[i].axis('off')

    plt.show()



# ------ #
@dataclass
class MultiViewBmodeVideo(MultiViewBmode):
  # new attributes
  n_frame: int = field(default=0, init=False)
  mat_source_dir: str = field(default='', init=False)  
  
  def __post_init__(self, ):
    '''
      self.view_images will be in shape (n_frame, n_view, h, w)
      self.view_masks will be in shape (1, n_view, h, w)
    '''
    pass


  def zero_pad_2d(self, padding:tuple):
    '''
      padding: (padding_left, padding_right, padding_top, padding_bottom)
    '''
    padding_left, padding_right, padding_top, padding_bottom = padding

    # 1 - update views and view masks (torch.nn.ZeroPad2d)
    zero_pad = torch.nn.ZeroPad2d(padding)
    self.view_images = zero_pad(self.view_images)
    self.view_masks = zero_pad(self.view_masks)

    # 2 - update image_shape and origin
    _, _, h, w = self.view_images.size()
    self.image_shape = (h, w)

    x0, y0 = self.origin
    self.origin = (x0 + padding_left, 
                   y0 + padding_top)

    return

# ------ # 
class Bmode2MultiViewBmodeVideo(Bmode2MultiViewBmode):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def convert(self, mat_file_dir = None, bmode_config_path = None):
    n_frame, n_view, h, w = self.b_mode.b_img_seq.shape

    origin_w = self.b_mode.trans_pos[0].left_edge_coord[0, 0]
    origin_h = self.b_mode.trans_pos[0].left_edge_coord[1, 0]

    aperture_size = self.b_mode.trans_pos[0].right_edge_coord[0, 0] - \
                    self.b_mode.trans_pos[0].left_edge_coord[0, 0]
    
    view_images = torch.tensor(self.b_mode.b_img_seq)
    view_masks = torch.tensor(self.b_mode.mask_seq)

    mat_source_dir = mat_file_dir
    bmode_config_file = bmode_config_path

    mvbv =  MultiViewBmodeVideo(
        n_view = n_view,
        image_shape = (h, w),
        origin = (origin_w, origin_h),
        aperture_size = aperture_size,

        view_images = view_images,
        view_masks = view_masks,

        mat_source_file = 'multiple files, refer to .mat_source_dir',
        bmode_config_file = bmode_config_file,
    )

    mvbv.n_frame = n_frame
    mvbv.mat_source_dir = mat_source_dir

    return mvbv

  def __str__(self, ):
    return f'MultiViewBmodeVideo - {self.__dict__.keys()}'


def plot_single_frame_in_multiview_bmode_video(mvbvs:dict, frame_index=0, ax=None):
  if ax is None:
    fig, ax = plt.subplots(2, 8)

  assert ax.shape == (2, 8), 'ax should be 2x8'

  for i, key in enumerate(mvbvs):
    mvbv = mvbvs[key]

    for j in range(mvbv.n_view):
      ax[i, j].imshow(mvbv.view_images[frame_index, j, ...])
      ax[i, j].axis('off')
      
  return ax

class MultiViewBmodeVideoPlayer:
  def __init__(self, 
               mvbvs:dict, 
               plot_func = plot_single_frame_in_multiview_bmode_video):
    
    self.mvbvs = mvbvs
    self.n_frame = mvbvs['lftx'].n_frame
    self.plot_func = plot_func

  def plot_frame(self, frame_index):
    fig, ax = plt.subplots(2, 8)
    self.plot_func(self.mvbvs, 
                   frame_index = frame_index, 
                   ax = ax)
    plt.show()

  def show_player(self):
    frame_slider = IntSlider(min = 0, 
                             max = self.n_frame-1, 
                             step = 1, 
                             value = 0, 
                             description = 'Frame')
    interactive_plot = interactive(self.plot_frame, 
                                   frame_index=frame_slider)
    display(interactive_plot)

  def save_video(self, 
                 tmp_dir = 'tmp', 
                 video_path = 'video.mp4'):
    # Create tmp dir
    if os.path.exists(tmp_dir):
      shutil.rmtree(tmp_dir)
    os.makedirs('tmp', exist_ok=True)

    # Save frames
    print('Saving frames ...')
    for i in tqdm(range(self.n_frame)):
      self.plot_func(self.mvbvs,
                     frame_index = i,
                     ax = None)
      
      plt.savefig(f'{tmp_dir}/{i:03}.jpg')
      plt.close()
    
    images = [img for img in glob.glob(f'{tmp_dir}/*jpg')]
    images.sort()

    # Save video
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_path,
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            15,
                            (width,height))
    print('Writing video ...')
    for image in tqdm(images):
      video.write(cv2.imread(image))

    video.release()

    return 
