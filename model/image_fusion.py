import torch

def mean_image_fusion(registered_imgs):
  n_view = registered_imgs.shape[0]
  fused_img = torch.zeros_like(registered_imgs[0, 0, ...])

  for i in range(n_view):
    fused_img += registered_imgs[i, 0, ...]

  fused_img /= n_view

  return fused_img
