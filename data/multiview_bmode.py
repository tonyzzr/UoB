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


class MultiViewBmodeSeg(MultiViewBmode):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    # initialize
    self.n_class = None
    self.seg_masks = None
    self.seg_configs = None


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
