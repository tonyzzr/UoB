
import torch.nn.functional as F
import UoB.model.lie as lie

class PixelDomainOptimization:
  def __init__(self, mvbs, init_rela_poses):

    '''
      In pixel domain optimization, we aim to start with an intial guess of relative poses,
      and then minimize the loss function defined on pixel values with respect to the relative poses.
    '''

    self.mvbs = mvbs
    self.mvbs_pad = self._zero_padding(mvbs)

    self.init_rela_poses = init_rela_poses

    self.rela_poses_thetas = init_rela_poses.thetas.clone()
    self.rela_poses_thetas.requires_grad_()

    self.ref_poses = None

    self.configs = {}

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

  def _get_gradient_mask(self):
    grad_mask = torch.ones(self.rela_poses_thetas.size())
    grad_mask[0] = 0.
    return grad_mask


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

  def set_configs(self, configs):
    assert isinstance(configs, dict)

    assert 'n_epochs' in configs
    assert 'lr' in configs
    assert 'fuser' in configs

    assert 'device' in configs
    assert configs['device'] in ['cpu', 'cuda']

    assert 'criterion' in configs
    assert 'key' in configs

    self.configs = configs

    return

  def _get_image_and_mask_tensors(self,):
    image_tensors = {}
    mask_tensors = {}
    for key in ['lftx', 'hftx']:
      mvb = self.mvbs_pad[key]
      image_tensors[key] = mvb.view_images.clone().unsqueeze(1)
      mask_tensors[key] = mvb.view_masks.clone().unsqueeze(1)

    return image_tensors, mask_tensors


  def run(self, ref_poses = None):


    # unpack configurations
    lr = self.configs['lr']
    n_epochs = self.configs['n_epochs']

    device = self.configs['device']
    fuser = self.configs['fuser']
    criterion = self.configs['criterion']


    # initialization
    rela_poses_thetas = self.rela_poses_thetas.to(device)
    grad_mask = self._get_gradient_mask().to(device)

    mvbs_pad = self.mvbs_pad
    n_view = self.mvbs_pad[self.configs['key']].n_view

    ref_poses_thetas = ref_poses.thetas.to(device)

    tissue_maps = {}
    image_tensors, mask_tensors = self._get_image_and_mask_tensors()
    normalized_transducer_positions = self._get_normalized_transducer_positions()

    # run SGD
    for epoch in range(n_epochs):
      for key in ['lftx', 'hftx']:
        normalized_transducer_position = normalized_transducer_positions[key].to(device)
        image_tensor = image_tensors[key].to(device)
        mask_tensor = mask_tensors[key].to(device)

        transformation_matrices = lie.forward_kinematics(normalized_transducer_position,
                                                         rela_poses_thetas)
        inversed_matrices = torch.linalg.inv(transformation_matrices)
        stn_theta_matrices = inversed_matrices.squeeze()[:, :2, :]

        grids = F.affine_grid(stn_theta_matrices, image_tensor.size())

        registered_imgs = F.grid_sample(image_tensor, grids)
        registered_masks = F.grid_sample(mask_tensor, grids)

        tissue_map = fuser(registered_imgs = registered_imgs.squeeze(),
                           registered_masks = registered_masks.squeeze())
        tissue_map = tissue_map.unsqueeze(0)
        tissue_maps[key] = tissue_map.detach().clone()[0, ...]

        if key == self.configs['key']:
          pred = tissue_map.unsqueeze(0).repeat(n_view, 1, 1, 1) * registered_masks
          loss = criterion(pred, registered_imgs)
          [grad] = torch.autograd.grad(loss,[rela_poses_thetas])
          grad *= grad_mask
          rela_poses_thetas.data -= lr * grad

          error = (rela_poses_thetas - ref_poses_thetas) * grad_mask
          lse = torch.sum(error**2)
          print(f'epoch = {epoch}, pixel loss = {loss:.6f}, pose lse = {lse:.6f}')

    self.rela_poses_thetas = rela_poses_thetas.detach().to('cpu').clone()
    return tissue_maps

  def convert_optimized_pose_tensor_to_pose_class(self, ):
    notes = {
        'pdo': True,
        'init_rela_poses': self.init_rela_poses,
        'pdo_configs': self.configs
    }
    return RelativePoses(
        thetas = self.rela_poses_thetas,
        notes = notes
    )

