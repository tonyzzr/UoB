import torch

def mean_image_fusion(registered_imgs, **kwargs):
  assert len(registered_imgs.shape) == 3, "registered_imgs dimension should be (N, H, W)"
  
  n_view = registered_imgs.shape[0]
  fused_img = torch.zeros_like(registered_imgs[0, ...])

  for i in range(n_view):
    fused_img += registered_imgs[i, ...]

  fused_img /= n_view

  return fused_img

#@title Fusers
def max_fuser(registered_imgs, **kwargs):
  assert type(registered_imgs) == torch.Tensor
  assert registered_imgs.ndim == 3

  fused_img, _ = torch.max(registered_imgs, dim=0)
  return fused_img

def weighted_mean_fuser(registered_imgs, registered_masks, eps=0.5, **kwargs):
  assert type(registered_imgs) == torch.Tensor
  assert registered_imgs.ndim == 3
  assert type(registered_masks) == torch.Tensor
  assert registered_masks.ndim == 3
  
  return torch.sum(registered_imgs * registered_masks, dim=0) / (torch.sum(registered_masks, dim=0) + eps)

def mean_fuser(registered_imgs, **kwargs):
  assert type(registered_imgs) == torch.Tensor
  assert registered_imgs.ndim == 3

  return torch.mean(registered_imgs, dim=0)

def red_green_fuser(registered_imgs, **kwargs):
  assert type(registered_imgs) == torch.Tensor
  assert registered_imgs.ndim == 3

  n_view, h, w = registered_imgs.size()

  red_channel_ind = list(range(0, n_view, 2))
  red_channel_imgs = registered_imgs[red_channel_ind, ...]

  green_channel_ind = list(range(1, n_view, 2))
  green_channel_imgs = registered_imgs[green_channel_ind, ...]
                         
  red_channel_fused = mean_fuser(red_channel_imgs)
  green_channel_fused = mean_fuser(green_channel_imgs)

  red_channel_fused /= torch.max(red_channel_fused)
  green_channel_fused /= torch.max(green_channel_fused)

  fused_img = torch.zeros(h, w, 3)
  fused_img[..., 0] = red_channel_fused
  fused_img[..., 1] = green_channel_fused

  return fused_img
