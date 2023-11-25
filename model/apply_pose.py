import cv2
import torch

def cv2apply_poses(imgs, matrices3d):
  assert len(imgs.shape) == 3, "img dimension should be (N, H, W)"
  
  transformed_imgs = torch.zeros_like(imgs)

  rc_ind = [0, 1, 3]
  matrices2d = matrices3d[:, rc_ind, :][:, :, rc_ind]
  matrices2d = matrices2d[:, 0:2, :]

  n_view = imgs.shape[0]
  for i in range(n_view):
    img = imgs[i, ...].numpy()
    mat = matrices2d[i, ...].numpy()

    # print(mat.shape)

    img = cv2.warpAffine(img, mat, (img.shape[1], img.shape[0]))
    transformed_imgs[i, ...] = torch.tensor(img)

  return transformed_imgs
