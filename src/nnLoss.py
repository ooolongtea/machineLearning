import torch
from torch import nn
from torch.nn import L1Loss

inputs = torch.tensor([1., 2., 3.])
targets = torch.tensor([1., 3., 5.])

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

loss = L1Loss()
rel = loss(inputs, targets)
print(rel)
criterion = nn.CrossEntropyLoss()
