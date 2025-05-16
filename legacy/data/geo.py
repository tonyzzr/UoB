import cv2
import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from sklearn.neighbors import KernelDensity
from torch.nn.functional import avg_pool2d

import torch
from geomloss import SamplesLoss

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

def grid(W):
    x, y = torch.meshgrid([torch.arange(0.0, W).type(dtype) / W] * 2, indexing="xy")
    return torch.stack((x, y), dim=2).view(-1, 2)


def as_measure(img, size):
    __img = cv2.resize(img, (size, size))
    weights = torch.from_numpy(__img).type(dtype)
    sampling = weights.shape[0] // size
    weights = (
        avg_pool2d(weights.unsqueeze(0).unsqueeze(0), 
                   sampling).squeeze(0).squeeze(0)
    )
    weights = weights / weights.sum()

    samples = grid(size)
    return weights.view(-1), samples

def migrate(measure, N, n_epoch):
    x_i = grid(N).view(-1, 2)
    a_i = (torch.ones(N * N) / (N * N)).type_as(x_i)

    x_i.requires_grad = True
    Loss = SamplesLoss("sinkhorn", blur=0.01, scaling=0.9)
    (b_j, y_j) = measure

    __x_i = x_i
    for epoch in range(n_epoch):
        L_ab = Loss(a_i, __x_i, b_j, y_j)
        [g_i] = torch.autograd.grad(L_ab, [__x_i])
        __x_i = __x_i - g_i / a_i.view(-1, 1)

    return __x_i.detach().clone()
