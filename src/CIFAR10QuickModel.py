import torch
from torch import nn
from torch.ao.nn.qat import Conv2d
from torch.nn import MaxPool2d, Flatten, Linear, Conv2d
from torch.utils.tensorboard import SummaryWriter


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(3, 32, 5, padding=2)  # input 3@32x32 output 32@32x32
        # self.maxpool1 = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(32, 32, 5, padding=2)  # input 32@16x16 output 32@16x16
        # self.maxpool2 = nn.MaxPool2d(2, 2)
        # self.conv3 = nn.Conv2d(32, 64, 5, padding=2)  # input 32@8x8 output 64@8x8
        # self.maxpool3 = nn.MaxPool2d(2, 2)
        # self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(1024, 64)
        # self.fc2 = nn.Linear(64, 10)

        self.model = nn.Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2, 2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2, 2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2, 2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        # x = self.maxpool1(self.conv1(x))
        # x = self.maxpool2(self.conv2(x))
        # x = self.maxpool3(self.conv3(x))
        # x = self.flatten(x)
        # x = self.fc1(x)
        # x = self.fc2(x)
        x = self.model(x)
        return x


if __name__ == '__main__':
    net = Net()
    print(net)
    input = torch.ones(64, 3, 32, 32)
    output = net(input)
    print(output.shape)

    writer = SummaryWriter(log_dir='../logs')
    writer.add_graph(net, input)
    writer.close()
