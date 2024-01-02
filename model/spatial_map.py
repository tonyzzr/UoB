import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

from ..data.poses import RelativePoses
from ..data.multiview_bmode import MultiViewBmode, plot_image_and_transducer_positions


from .rigid_link import RigidLink
from .apply_pose import cv2apply_poses
from .image_fusion import weighted_mean_fuser, max_fuser, mean_fuser


def spatial_mapping(mvbs, pose_in_degree, fuser = max_fuser):
  tissue_maps = {}
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

  return tissue_maps
  
def reneder_tissue_maps(tissue_maps, ax=None):
  if ax is None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
  for i, key in enumerate(['lftx', 'hftx']):
    ax[i].imshow(tissue_maps[key], cmap='gray')
    ax[i].axis('off')
