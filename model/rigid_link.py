import pypose as pp
import torch
import numpy as np

import matplotlib.pyplot as plt

class RigidLink:

  def __init__(self, n=3, length=1):
    self.n = n
    self.length = length

    # initialize
    self.thetas = torch.zeros(n)
    self.rela_poses = pp.identity_SE3(n)
    self.global_poses = pp.identity_SE3(n)
    self.joint_locations = torch.zeros(4).repeat(n, 1) # homogenous coordinates

    self.origin = torch.zeros(4)

  def set_origin(self, x0, y0):
    self.origin = torch.tensor([x0, y0, 0, 1])


  def set_thetas(self, angles):
    thetas = (angles) / 180 * torch.tensor(np.pi)
    # should i accumulate the angles?
    self.thetas = thetas
    return self.thetas

  def calc_rela_poses(self,):
    for i in range(1, self.n):
      M = []
      theta = self.thetas[i]

      # construct rotation in SE3
      theta_vec = torch.tensor([0, 0, 1]) * theta
      R_SO3 = pp.so3(theta_vec).Exp()
      R_SE3 = torch.cat([torch.tensor([0, 0, 0,]), R_SO3.tensor()])

      # move (0, 0) to (L, 0)
      M.append(pp.SE3([self.length, 0, 0, 0, 0, 0, 1]))

      # rotate around (L, 0)
      M.append(pp.SE3([-self.length, 0, 0, 0, 0, 0, 1]))
      M.append(pp.SE3(R_SE3))
      M.append(pp.SE3([self.length, 0, 0, 0, 0, 0, 1]))

      transform_mat = self.rela_poses[i].matrix()
      for m in M:
        transform_mat = m.matrix() @ transform_mat
      self.rela_poses[i] = pp.mat2SE3(transform_mat)

    return self.rela_poses

  def calc_global_poses(self,):
    global_poses = self.global_poses

    # global pose of the first array
    # 1 - rotate around (0, 0) of theta0
    theta0 = self.thetas[0]
    theta_vec = torch.tensor([0, 0, 1]) * theta0
    R_SO3 = pp.so3(theta_vec).Exp()
    R_SE3 = torch.cat([torch.tensor([0, 0, 0,]), R_SO3.tensor()])

    # 2 - move the origin to self.origin (x0, y0)
    x0, y0 = self.origin[0], self.origin[1]
    T_SE3 = pp.SE3([x0, y0, 0, 0, 0, 0, 1])

    transform_mat = T_SE3.matrix() @ pp.SE3(R_SE3).matrix()
    global_poses[0] = pp.mat2SE3(transform_mat)


    # global poses of all other arrays
    for i in range(1, self.n):
      transform_mat =  global_poses[i-1].matrix() @ self.rela_poses[i].matrix() # do relative transformation first
      global_poses[i] = pp.mat2SE3(transform_mat)

    self.global_poses = global_poses
    return self.global_poses


  def calc_joint_locations(self,):
    # joint locates (L, 0)
    joint_homo_coord = torch.tensor([self.length, 0, 0, 1], dtype=torch.float32)
    joint_locations = torch.zeros(4).repeat(self.n, 1)

    for i in range(self.n):
      joint_locations[i, ...] = self.global_poses[i].matrix() @ joint_homo_coord

    self.joint_locations = joint_locations
    return self.joint_locations


  def show_rigid_link(self, ax=None, axlim=(-1, 3)):
    if ax is None:
      fig, ax = plt.subplots(1, 1, figsize=(3, 3))

    data = self.joint_locations.numpy()
    # data = np.concatenate(([[0, 0, 0, 1]], data), axis=0) # origin = (0, 0)
    data = np.concatenate((self.origin.repeat(1, 1), data), axis=0)

    # Extracting x and y coordinates
    x = [point[0] for point in data]
    y = [point[1] for point in data]

    # Plotting lines between the points
    ax.plot(x, y, 'x-')

    ax.set_aspect('equal')
    ax.set_xlim(axlim)
    ax.set_ylim(axlim)

    ax.invert_yaxis()

    return ax

def forward_kinematics(self,):
  self.calc_rela_poses()
  self.calc_global_poses()
  self.calc_joint_locations()
  return
