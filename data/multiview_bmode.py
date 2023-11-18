'''
  This part of code should be placed in 'data/bmode.py'.
'''
from dataclasses import dataclass

import torch
import numpy as np
import matplotlib.pyplot as plt

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

    origin_h, origin_w = self.origin
    self.origin = (origin_h + padding_top, 
                   origin_w + padding_left)

    return

  def resize(self, size:tuple):
    '''
      size: (h, w)
    '''

    # 1 - update views and view masks (torchvision.transforms.Resize)

    # 2 - update image_shape, origin, aperture_size


    pass




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
