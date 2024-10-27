import torch
import torchvision
from torch import nn
from torch.nn import Sequential, Flatten
from torch.utils.data import DataLoader

# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载数据集
dataset = torchvision.datasets.CIFAR10("../data", train=True, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model1 = Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


net = Net().to(device)  # 将模型移动到GPU
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        image, labels = data
        image, labels = image.to(device), labels.to(device)  # 将数据移动到GPU
        outputs = net(image)
        rel_loss = loss(outputs, labels)
        optimizer.zero_grad()
        rel_loss.backward()
        optimizer.step()
        running_loss += rel_loss.item()
    print(running_loss)
