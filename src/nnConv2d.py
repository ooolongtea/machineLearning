# import torch
# import torchvision
# from torch.utils.data import DataLoader
#
# dataset = torchvision.datasets.CIFAR10("./data", train=False, transform=torchvision.transforms.ToTensor(),
#                                        download=True)
# dataloader = DataLoader(dataset, batch_size=64)
#
#
# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = torch.nn.Conv2d(3, 6, 3, stride=1, padding=0)
#         # 自动生成权重矩阵,并随机初始化这些权重
#         # 每个输出通道有独立的卷积核，每个卷积核的大小是3x3x3
#
#     # 输出x
#     def forward(self, x):
#         x = self.conv1(x)
#         return x
#
#
# net = Net()
#
# for data in dataloader:
#     img, label = data
#     output = net(img)
#     print(img.shape)
#     print(output.shape)
#     # torch.Size([64, 3, 32, 32])
#     # torch.Size([64, 6, 30, 30])
import torch
import torch.nn as nn

# 创建一个 MaxPool2d 池化层，池化窗口大小为 2x2
max_pool = nn.MaxPool2d(kernel_size=2)

# 假设输入是一个大小为 (1, 1, 4, 4) 的 4x4 图像
input_tensor = torch.tensor([[[[1., 2., 3., 4.],
                               [5., 6., 7., 8.],
                               [9., 10., 11., 12.],
                               [13., 14., 15., 16.]]]])

# 应用最大池化
output = max_pool(input_tensor)

print(output)
# 输出是 2x2，每个区域取最大值
# tensor([[[[ 6.,  8.],
#           [14., 16.]]]])
