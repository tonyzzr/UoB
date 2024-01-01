import torch
import torch.nn as nn
import numpy as np


def trans_mat2D(x, y):
  '''
    2D translation matrix works with homogenous coordinates.
  '''
  x = torch.tensor(x).double()
  y = torch.tensor(y).double()

  Gi = np.array([[1, 0, 0],
                 [0, 1, 0],
                 [0, 0, 1],
                ])

  Gx = np.array([[0, 0, 1],
                 [0, 0, 0],
                 [0, 0, 0],
                 ])

  Gy = np.array([[0, 0, 0],
                 [0, 0, 1],
                 [0, 0, 0],
                 ])

  
  Gi_tensor = torch.from_numpy(Gi).double()
  Gx_tensor = torch.from_numpy(Gx).double()
  Gy_tensor = torch.from_numpy(Gy).double()

  trans_mat = Gx_tensor * x + \
              Gy_tensor * y + \
              Gi_tensor

  return trans_mat

def rot_mat2D(rot_angle, center_x = 0, center_y = 0):
  '''
    2D rotation matrix works with homogenous coordinates.
  '''
  center_x = torch.tensor(center_x).double()
  center_y = torch.tensor(center_y).double()
  # rotation matrix around center (0, 0)

  G0 = np.array([[0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 1],
                 ])

  G1 = np.array([[1, 0, 0],
                 [0, 1, 0],
                 [0, 0, 0],
                 ])

  G2 = np.array([[0, -1, 0],
                 [1, 0, 0],
                 [0, 0, 0],
                 ])
  
  G0_tensor = torch.from_numpy(G0).double()
  G1_tensor = torch.from_numpy(G1).double()
  G2_tensor = torch.from_numpy(G2).double()

  rot_mat = G1_tensor * torch.cos(rot_angle) + \
            G2_tensor * torch.sin(rot_angle) + \
            G0_tensor

  # rotation matrix around (center_x, center_y)
  t_positive = trans_mat2D(center_x, center_y)
  t_negative = trans_mat2D(-center_x, -center_y)

  return torch.mm(torch.mm(t_positive, 
                           rot_mat), 
                  t_negative)

def tensorize_trans_pos(trans_pos, img_shape ):
  '''
    output dimensions = (num_arr, num_edge=2, 3, 1)
  '''
  num_arr = len(trans_pos.keys())

  trans_pos_tensor = []
  for i in range(num_arr):
    x_l = trans_pos[i].left_edge_coord.copy() 
    x_r = trans_pos[i].right_edge_coord.copy() 
    # print(x_l, x_r)
    
    x_l[0] = x_l[0] / (img_shape[0]/2) - 1
    x_r[0] = x_r[0] / (img_shape[0]/2) - 1

    x_l[1] = x_l[1] / (img_shape[1]/2) - 1
    x_r[1] = x_r[1] / (img_shape[1]/2) - 1



    trans_pos_tensor.append([x_l,
                             x_r,])
  
  trans_pos_tensor = torch.tensor(trans_pos_tensor)



  return trans_pos_tensor.double()

def forward_kinematics(trans_pos_tensor, 
                       thetas, ):
  # assert trans_pos_tensor.device == thetas.device
  device = thetas.device
  
  num_arr = trans_pos_tensor.size()[0]
  rot_angles = torch.cumsum(thetas, dim=0).unsqueeze(1).unsqueeze(2)
                         
  # print(rot_angles.size())
  x_l = trans_pos_tensor[:, 0, ...]
  x_r = trans_pos_tensor[:, 1, ...]
  # print(x_l.size())

  G0 = torch.from_numpy(np.array([[0, 0, 0],
                                  [0, 0, 0],
                                  [0, 0, 1],
                                  ])
                        ).unsqueeze(0).repeat(num_arr, 1, 1).to(device)

  G1 = torch.from_numpy(np.array([[1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 0],
                                  ])
                        ).unsqueeze(0).repeat(num_arr, 1, 1).to(device)

  G2 = torch.from_numpy(np.array([[0, -1, 0],
                                  [1, 0, 0],
                                  [0, 0, 0],
                                  ])
                        ).unsqueeze(0).repeat(num_arr, 1, 1).to(device)
  
  # print(G1.size())
  R = G1 * torch.cos(rot_angles) + G2 * torch.sin(rot_angles)
  # print(R)

  R_cum = torch.cumsum(R, dim=0)
  R_cum0 = R_cum - R
  R_cum1 = R_cum - R[0, ...]

  t = torch.einsum('bij,bjk->bik', R_cum0, x_r) - \
      torch.einsum('bij,bjk->bik', R_cum1, x_l)
  
  # print(t)
  t_cat = torch.cat((torch.zeros(num_arr, 3, 2), t), dim=2)
  T = R + t_cat + G0

  # print(T)

#   T_inv = torch.linalg.inv(T)

  # for i in range(8):
  #   print(torch.mm(T[i,...], 
  #                T_inv[i, ...]))

  return T
