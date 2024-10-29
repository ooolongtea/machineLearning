import torch
import torchvision.datasets as datasets
from sympy.functions.elementary.tests.test_trigonometric import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

dataset = datasets.CIFAR10(root="./data", train=False,
                           transform=transforms.ToTensor(),
                           download=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear = nn.Linear(in_features=196608, out_features=10)

    def forward(self, x):
        x = self.linear(x)
        return x


data_loader = DataLoader(dataset, batch_size=64)
for data in data_loader:
    images, targets = data
    print(images.shape)
    output = torch.reshape(images, [1, 1, 1, -1])
    # 展平 torch.flatten(images)
