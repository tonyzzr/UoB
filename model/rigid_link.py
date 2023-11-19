#@title Define rigid link
import pypose as pp
import torch
import numpy as np

import matplotlib.pyplot as plt

class RigidLink:

  def __init__(self, n=3, length=1):
    self.n = n
    self.length = length

    # initialize
    self.n_link_points = 32 # points to visualize links
    self.link_coordinates = torch.zeros(4, self.n_link_points)

    self.thetas = torch.zeros(n)
    self.rela_poses = pp.identity_SE3(n)
    self.global_poses = pp.identity_SE3(n)
    self.joint_locations = torch.zeros(4).repeat(n, 1) # homogenous coordinates

    self.origin = torch.zeros(4)

  def set_origin(self, x0, y0):
    self.origin = torch.tensor([x0, y0, 0, 1])

    x1, y1 = x0 + self.length, y0
    link_x = torch.linspace(x0, x1, self.n_link_points)
    link_y = torch.linspace(y0, y1, self.n_link_points)
    self.link_coordinates = torch.stack([link_x, link_y, 
                             torch.zeros_like(link_x), 
                             torch.ones_like(link_y)], dim=1).T
    


  def set_thetas(self, angles):
    thetas = (angles) / 180 * torch.tensor(np.pi)
    # should i accumulate the angles?
    self.thetas = thetas
    return self.thetas

  def calc_rela_poses(self,):
    '''
      Note the left corner of the transducer to be [x0, y0, 0, 1].T, apertuere size l.
      The right corner is [x1, y1, 0, 1].T == [x0 + l, y0, 0, 1].T
      There are two steps in this rigid transformation:
        1. Move (x0, y0) to (x0 + l, y0)
        2. Rotate around (x0 + l, y0)
      ---
      This is also equal to:
        1. Rotate around (x0, y0)    
        2. Move (x0, y0) to (x0 + l, y0)
    '''
    for i in range(1, self.n):
      M = [] # a sequence of transformations applied to derive relative pose
      theta = self.thetas[i]

      # construct rotation in SE3
      theta_vec = torch.tensor([0, 0, 1]) * theta
      R_SO3 = pp.so3(theta_vec).Exp()
      R_SE3 = torch.cat([torch.tensor([0, 0, 0,]), R_SO3.tensor()])

      # move (0, 0) to (L, 0) == Move (x0, y0) to (x0 + l, y0)
      M.append(pp.SE3([self.length, 0, 0, 0, 0, 0, 1]))

      # rotate around (x0 + l, y0) == (x1, y1)
      x0, y0 = self.origin[0], self.origin[1] 
      x1, y1 = x0 + self.length, y0

      M.append(pp.SE3([-x1, -y1, 0, 0, 0, 0, 1]))
      M.append(pp.SE3(R_SE3))
      M.append(pp.SE3([x1, y1, 0, 0, 0, 0, 1]))

      transform_mat = self.rela_poses[i].matrix()
      for m in M:
        transform_mat = m.matrix() @ transform_mat
      self.rela_poses[i] = pp.mat2SE3(transform_mat)

    return self.rela_poses

  def calc_global_poses(self,):
    global_poses = self.global_poses

    # # global pose of the first array
    # # 1 - rotate around (0, 0) of theta0
    # theta0 = self.thetas[0]
    # theta_vec = torch.tensor([0, 0, 1]) * theta0
    # R_SO3 = pp.so3(theta_vec).Exp()
    # R_SE3 = torch.cat([torch.tensor([0, 0, 0,]), R_SO3.tensor()])

    # # 2 - move the origin to self.origin (x0, y0)
    # x0, y0 = self.origin[0], self.origin[1]
    # T_SE3 = pp.SE3([x0, y0, 0, 0, 0, 0, 1])

    # transform_mat = T_SE3.matrix() @ pp.SE3(R_SE3).matrix()
    # global_poses[0] = pp.mat2SE3(transform_mat)


    # global poses of all other arrays
    for i in range(1, self.n):
      transform_mat =  global_poses[i-1].matrix() @ self.rela_poses[i].matrix() # do relative transformation first
      global_poses[i] = pp.mat2SE3(transform_mat)

    self.global_poses = global_poses
    return self.global_poses


  def calc_joint_locations(self,):
    # first joint locates (x1, y1)
    x0, y0 = self.origin[0], self.origin[1] 
    x1, y1 = x0 + self.length, y0

    joint_homo_coord = torch.tensor([x1, y1, 0, 1], dtype=torch.float32)
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

  def show_relative_poses(self, ):
    transdformed_link_coordinates = self.rela_poses.matrix() @ self.link_coordinates

    for i in range(self.n):
      fig, ax = plt.subplots(1, 1, figsize=(6, 3))

      # plot original elements
      ax.scatter(self.link_coordinates[0, ...], 
                 self.link_coordinates[1, ...], 
                 marker='x', c='k', s=10)
      
      # highlight the first element in the transformed array
      ax.scatter(transdformed_link_coordinates[i, 0, 0], 
                 transdformed_link_coordinates[i, 1, 0], c='b', s=10)
      
      # plot all elements in the transformed array
      ax.scatter(transdformed_link_coordinates[i, 0, ...], 
                 transdformed_link_coordinates[i, 1, ...], c='r', s=1)

      ax.axis('equal')
      ax.invert_yaxis()

      plt.show()

  def show_global_poses(self, ax=None):
    transdformed_link_coordinates = self.global_poses.matrix() @ self.link_coordinates

    if ax is None:
      fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    # plot original elements
    ax.scatter(self.link_coordinates[0, ...], 
               self.link_coordinates[1, ...], 
               marker='x', c='k', s=10)
    
    for i in range(self.n):

      # highlight the first element in the transformed array
      ax.scatter(transdformed_link_coordinates[i, 0, 0], 
                 transdformed_link_coordinates[i, 1, 0], c='b', s=10)
      
      # plot all elements in the transformed array
      ax.scatter(transdformed_link_coordinates[i, 0, ...], 
                 transdformed_link_coordinates[i, 1, ...], c='r', s=1)

    ax.axis('equal')
    ax.invert_yaxis()

    # plt.show()
    return ax

  def __str__(self, ):
    return f'RigidLink(n={self.n}, length={self.length}, origin={self.origin})'
