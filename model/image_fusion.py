import torch

def mean_image_fusion(registered_imgs):
  assert len(registered_imgs.shape) == 3, "registered_imgs dimension should be (N, H, W)"
  
  n_view = registered_imgs.shape[0]
  fused_img = torch.zeros_like(registered_imgs[0, ...])

  for i in range(n_view):
    fused_img += registered_imgs[i, ...]

  fused_img /= n_view

  return fused_img
