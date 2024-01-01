import torch

def mean_image_fusion(registered_imgs):
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
