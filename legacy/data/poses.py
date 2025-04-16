import torch
from dataclasses import dataclass, field

@dataclass
class RelativePoses:
  thetas: torch.Tensor # in radian
  notes: dict = field(default_factory=dict)

  def __len__(self):
    return self.thetas.shape[0]

  def __getitem__(self, idx):
    return self.thetas[idx]

  def pose_in_degree(self, ):
    return torch.rad2deg(self.thetas)
