import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

from ..data.poses import RelativePoses
from ..data.multiview_bmode import MultiViewBmode, plot_image_and_transducer_positions


import .lie as lie
from .rigid_link import RigidLink
from .apply_pose import cv2apply_poses
from .image_fusion import weighted_mean_fuser, max_fuser, mean_fuser

from tqdm import tqdm

def spatial_mapping(mvbs, pose_in_degree, fuser = max_fuser, return_rl = False):
  tissue_maps = {}
  rigid_links = {}
  for key in ['lftx', 'hftx']:
    mvb = mvbs[key]
    n_view, aperture_size, origin = mvb.n_view, mvb.aperture_size, mvb.origin

    # Rigid link (angle -> global_poses_mat)
    rl = RigidLink(n=n_view, length=aperture_size)
    rl.set_origin(x0=origin[0], y0=origin[1])
    rl.set_thetas(pose_in_degree)
    rl.forward_kinematics()
    global_poses_mat = rl.global_poses.matrix()

    # Apply transformations
    registered_imgs = cv2apply_poses(mvb.view_images, global_poses_mat)
    registered_masks = cv2apply_poses(mvb.view_masks, global_poses_mat)

    # Image fusion
    params = dict(registered_imgs = registered_imgs,
                  registered_masks = registered_masks,)
    fused_img = fuser(**params)
    tissue_maps[key] = fused_img
    rigid_links[key] = rl

  if return_rl:
    return tissue_maps, rigid_links
  else:
    return tissue_maps
  
def reneder_tissue_maps(tissue_maps, ax=None):
  if ax is None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
  for i, key in enumerate(['lftx', 'hftx']):
    ax[i].imshow(tissue_maps[key], cmap='gray')
    ax[i].axis('off')




# ------ #
#@title Define Laplacian pyramid variance loss


def gaussian_kernel(size=5, sigma=1.0):
  # Create a 2D Gaussian kernel array.
  kernel1d = torch.from_numpy(np.linspace(-(size // 2), size // 2, size))
  kernel1d = torch.exp(-0.5 * (kernel1d / sigma).pow(2))
  kernel1d = kernel1d / kernel1d.sum()
  kernel2d = torch.outer(kernel1d, kernel1d)
  kernel2d = kernel2d.unsqueeze(0).unsqueeze(0)
  return kernel2d

def apply_gaussian_filter(input, kernel):
  # Assure that the kernel and input is on the same device and dtype
  kernel = kernel.to(input.device).type(input.dtype)
  padding = kernel.shape[-1] // 2
  return F.conv2d(input, kernel, padding=padding)

def gaussian_pyramid(img, max_levels):
  kernel = gaussian_kernel()  # Define a suitable Gaussian filter
  current = img
  pyramids = [current]
  for _ in range(max_levels-1):
    current = apply_gaussian_filter(current, kernel)
    current = F.interpolate(current, scale_factor=0.5, mode='bilinear', align_corners=False)
    pyramids.append(current)
  return pyramids

def laplacian_pyramid(gauss_pyr):
  lap_pyr = [gauss_pyr[-1]]  # the last image in Gaussian is the first in Laplacian
  for i in range(len(gauss_pyr)-1, 0, -1):
    upsampled = F.interpolate(gauss_pyr[i], size=gauss_pyr[i-1].shape[-2:], mode='bilinear', align_corners=False)
    laplacian = gauss_pyr[i-1] - upsampled  # Difference is Laplacian
    lap_pyr.append(laplacian)
  return lap_pyr

def laplacian_variance_loss(img, n_level=5,
                            target_level=1,
                            logrithm = True,):
  gauss_pyr = gaussian_pyramid(img, n_level)
  lap_pyr = laplacian_pyramid(gauss_pyr)

  lap_img = lap_pyr[target_level][0, 0, ...]

  lap_var = torch.var(lap_img)
  if logrithm:
    lap_var = torch.log(lap_var)

  lap_var_loss = -lap_var
  return lap_var_loss

class LaplacianVarianceLoss(nn.Module):
  def __init__(self, n_level=5, target_level=1, logrithm=True):
    super().__init__()

    self.n_level = n_level
    self.target_level = target_level
    self.logrithm = logrithm

  def forward(self, pred, *args, **kwargs):
    return laplacian_variance_loss(pred, n_level=self.n_level,
                                   target_level=self.target_level,
                                   logrithm=self.logrithm)
#@title SpatialMapNet

class SpatialMapNet(nn.Module):
  '''
    A differentiable spatial mapping of tissue architecture using multiview B-mode data.
  '''

  def __init__(self, init_poses_in_radian, fuser):
    super().__init__()
    self.init_poses_in_radian = init_poses_in_radian
    self.rela_poses_in_radian = nn.Parameter(data = self.init_poses_in_radian,
                                                   requires_grad = True)
    self.rela_poses_in_radian.register_hook(self._mask_grad)

    self.fuser = fuser

  def _mask_grad(self, grad):
    grad_mask = torch.ones_like(grad)
    grad_mask[0] = 0
    grad_mask.to(grad.device)

    return grad * grad_mask

  def _calc_stn_theta_matrices(self, normalized_transducer_position):

    transformation_matrices = lie.forward_kinematics(normalized_transducer_position,
                                                      self.rela_poses_in_radian)
    inversed_matrices = torch.linalg.inv(transformation_matrices)
    stn_theta_matrices = inversed_matrices.squeeze()[:, :2, :]

    return stn_theta_matrices

  def _calc_tissue_map(self, image_tensor, mask_tensor, grids):
    '''
      input
        image_tensor: [N, C, H, W]
        mask_tensor: [N, C, H, W]
        grids: [N, H, W, 2]

      return
        tissue_map: [1, C, H, W]
    '''

    registered_imgs = F.grid_sample(image_tensor, grids)
    registered_masks = F.grid_sample(mask_tensor, grids)

    tissue_map = self.fuser(registered_imgs = registered_imgs.squeeze(),
                        registered_masks = registered_masks.squeeze())
    tissue_map = tissue_map.unsqueeze(0).unsqueeze(0)

    return tissue_map


  def forward(self, image_tensor, mask_tensor, normalized_transducer_position):

    stn_theta_matrices = self._calc_stn_theta_matrices(normalized_transducer_position)
    grids = F.affine_grid(stn_theta_matrices, image_tensor.size())
    tissue_map = self._calc_tissue_map(image_tensor, mask_tensor, grids)

    return tissue_map


#@title PixelDomainOptimization
from IPython.display import display, clear_output

class PixelDomainOptimization:
  def __init__(self, mvbs, init_rela_poses, ref_rela_poses=None):

    '''
      In pixel domain optimization, we aim to start with an intial guess of relative poses,
      and then minimize the loss function defined on pixel values with respect to the relative poses.
    '''

    self.mvbs = mvbs
    self.mvbs_pad = self._zero_padding(mvbs)

    self.init_poses = init_rela_poses

    self.optimized_poses = None
    self.ref_poses = ref_rela_poses

    self.config = {}
    self.training_history = {}

  @staticmethod
  def _zero_padding(mvbs):
    mvbs_pad = deepcopy(mvbs)

    for key in ['lftx', 'hftx']:
      mvb = mvbs_pad[key]

      aperture_size = mvb.aperture_size
      h, w = mvb.image_shape

      larger_dim = max(h, w)
      mvb.zero_pad_2d((int(larger_dim * 0.2), int(larger_dim * 1.5)-w-int(larger_dim * 0.2),
                       int(larger_dim * 0.2), int(larger_dim * 1.5)-h-int(larger_dim * 0.2)))

    return mvbs_pad

  def _get_normalized_transducer_positions(self):
    normalized_transducer_positions = {}
    for key in ['lftx', 'hftx']:
      mvb = self.mvbs_pad[key]
      aperture_size, origin, image_shape =  mvb.aperture_size, mvb.origin, mvb.image_shape

      h, w = image_shape
      x0, y0 = origin
      x1, y1 = x0 + aperture_size, y0

      x_l = x0/(w/2) - 1, y0/(h/2) - 1, 1
      x_r = x1/(w/2) - 1, y1/(h/2) - 1, 1

      x_l = np.array(x_l).reshape(3, 1)
      x_r = np.array(x_r).reshape(3, 1)


      normalized_transducer_positions[key] = torch.tensor([[x_l, x_r]] * mvb.n_view)

    return normalized_transducer_positions

  def set_model(self, model_class, fuser):

    init_poses_in_radian = self.init_poses.thetas
    model = model_class(init_poses_in_radian = init_poses_in_radian,
                        fuser = fuser)
    self.config['model'] = model

    return

  def set_optimizer(self, optimizer_class, **params):

    optimizer = optimizer_class(self.config['model'].parameters(), **params)
    self.config['optimizer'] = optimizer

    return

  def set_loss_func(self, loss_func, train_data_key = 'lftx'):

    self.config['loss_func'] = loss_func
    self.config['train_data_key'] = train_data_key
    return


  def set_trainer(self, n_epochs, device, **kwargs):

    self.config['n_epochs'] = n_epochs
    self.config['device'] = device

    for key in kwargs.keys():
      self.config[key] = kwargs[key]

    return

  def set_ref_poses(self, ref_poses):
    self.ref_poses = ref_poses
    return


  def _get_image_and_mask_tensors(self,):
    image_tensors = {}
    mask_tensors = {}
    for key in ['lftx', 'hftx']:
      mvb = self.mvbs_pad[key]
      image_tensors[key] = mvb.view_images.clone().unsqueeze(1)
      mask_tensors[key] = mvb.view_masks.clone().unsqueeze(1)

    return image_tensors, mask_tensors


  def run(self,):

    # unpack configurations
    model = self.config['model']
    optimizer = self.config['optimizer']
    loss_func = self.config['loss_func']

    train_data_key = self.config['train_data_key']
    n_epochs = self.config['n_epochs']
    device = self.config['device']

    # reference poses
    ref_poses = self.ref_poses

    # initialization
    model = model.to(device)
    image_tensors, mask_tensors = self._get_image_and_mask_tensors()
    normalized_transducer_positions = self._get_normalized_transducer_positions()

    fig, ax = plt.subplots(2, 2)

    # run SGD
    for epoch in tqdm(range(n_epochs)):

      tissue_maps = {}
      for key in ['lftx', 'hftx']:
        normalized_transducer_position = normalized_transducer_positions[key].to(device)
        image_tensor = image_tensors[key].to(device)
        mask_tensor = mask_tensors[key].to(device)

        if key == train_data_key:
          model.train()
          tissue_map = model(image_tensor, mask_tensor, normalized_transducer_position)
          loss = loss_func(tissue_map)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

        else:
          model.eval()
          tissue_map = model(image_tensor, mask_tensor, normalized_transducer_position)
          rela_poses_in_radian = model.rela_poses_in_radian

        tissue_maps[key] = tissue_map.detach().to('cpu').clone()

      # log and plot
      clear_output(wait=True)

      self.log_training_history(epoch,
                                loss = loss.detach().cpu().item(),
                                rela_poses_in_radian = rela_poses_in_radian.detach().cpu().numpy(),)
      self.plot_training_status(key='loss', ax=ax[0, 0])
      self.plot_pose_error_history(['se'], ax=ax[0, 1])
      self.plot_tissue_maps(tissue_maps, ax=ax[1, :])

      display(fig)


    return tissue_maps

  def _calc_pose_error_history(self, metric:str):

    pose_history = self.training_history['rela_poses_in_radian']
    pose_history = torch.tensor(pose_history)

    pose_error_history = pose_history - self.ref_poses.thetas
    pose_error_mask = torch.ones_like(pose_history)
    pose_error_mask[:, 0, ...] = 0

    pose_error_history = pose_error_history * pose_error_mask

    if metric == 'se':
      return torch.sum(pose_error_history**2, dim=1)

    return

  def plot_pose_error_history(self, metrics:list = [], ax=None):
    if ax is None:
      fig, ax = plt.subplots()

    ax.clear()
    for metric in metrics:
      ax.semilogy(self._calc_pose_error_history(metric), label=metric)

    ax.legend()
    return

  def log_training_history(self, epoch, **kwargs):
    for key in kwargs.keys():
      if key not in self.training_history.keys():
        self.training_history[key] = []

      self.training_history[key].append(kwargs[key])
    return

  def plot_training_status(self, key, ax=None):
    if ax is None:
      fig, ax = plt.subplots()

    ax.clear()
    ax.plot(self.training_history[key])
    ax.set_title(key)
    return

  def plot_tissue_maps(self, tissue_maps, ax=None):
    if ax is None:
      fig, ax = plt.subplots(1, 2)

    for i, key in enumerate(['lftx', 'hftx']):
      tissue_map = tissue_maps[key][0, 0, ...]

      ax[i].clear()
      ax[i].imshow(tissue_map, cmap='gray')
      ax[i].set_title(key)
      ax[i].axis('off')

    return


  def convert_optimized_pose_tensor_to_pose_class(self, ): # this part needs update
    notes = {
        'pdo': True,
        'init_rela_poses': self.init_rela_poses,
        'pdo_configs': self.configs
    }
    return RelativePoses(
        thetas = self.rela_poses_thetas,
        notes = notes
    )
