import cv2

def cv2apply_poses(imgs, matrices3d):
  transformed_imgs = torch.zeros_like(imgs)

  rc_ind = [0, 1, 3]
  matrices2d = matrices3d[:, rc_ind, :][:, :, rc_ind]
  matrices2d = matrices2d[:, 0:2, :]

  for i in range(imgs.shape[0]):
    img = imgs[i, 0, ...].numpy()
    mat = matrices2d[i, :, :].numpy()

    # print(mat.shape)

    img = cv2.warpAffine(img, mat, (img.shape[1], img.shape[0]))
    transformed_imgs[i, 0, ...] = torch.tensor(img)

  return transformed_imgs
