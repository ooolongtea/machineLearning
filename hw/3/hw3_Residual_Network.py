import torch
from torch import nn
from torch.backends.cudnn import set_flags
from torch.nn import Conv2d
from torchvision import models


class ResidualNetwork(nn.Module):
    def __init__(self):
        super(ResidualNetwork, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, ),
            nn.BatchNorm2d(64),
        )

        self.layer2 = nn.Sequential(
            Conv2d(64, 64, 3, 1, 1, ),
            nn.BatchNorm2d(64),
        )

        self.layer3 = nn.Sequential(
            Conv2d(64, 128, 3, 2, 1, ),
            nn.BatchNorm2d(128),
        )

        self.layer4 = nn.Sequential(
            Conv2d(128, 128, 3, 1, 1, ),
            nn.BatchNorm2d(128),
        )

        self.layer5 = nn.Sequential(
            Conv2d(128, 256, 3, 2, 1, ),
            nn.BatchNorm2d(256),
        )

        self.layer6 = nn.Sequential(
            Conv2d(256, 256, 3, 1, 1, ),
            nn.BatchNorm2d(256),
        )

        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 32 * 32, 256),
            nn.ReLU(),
            nn.Linear(256, 11),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.layer1(x)
        x1 = self.relu(x1)
        residual = x1
        # 第二层残差
        x2 = self.layer2(x1)
        x2 += residual
        x2 = self.relu(x2)

        x3 = self.layer3(x2)
        x3 = self.relu(x3)
        residual = x3

        x4 = self.layer4(x3)
        x4 += residual
        x4 = self.relu(x4)

        x5 = self.layer5(x4)
        x5 = self.relu(x5)
        residual = x5

        x6 = self.layer6(x5)
        x6 += residual
        x6 = self.relu(x6)

        x6 = self.fc_layer(x6)
        return x6


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        # ResNet18 作为特征提取器
        self.feature_extractor = models.resnet18(pretrained=False)
        # 移除 ResNet 的最后一层（全连接层）
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])

        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(512, 256),  # 调整输入大小
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 11)  # 输出类别数
        )

    def forward(self, x):
        # 输入形状: [batch_size, 3, 128, 128]

        x = self.feature_extractor(x)

        # 展平特征
        x = x.flatten(1)

        # 通过全连接层得到最终输出
        x = self.fc_layers(x)
        return x


# if __name__ == '__main__':
#     torch.cuda.set_device(2)
#     device = "cuda" if torch.cuda.is_available() else "CPU"
#     model = ResidualNetwork().to(device)
#     input = torch.rand(1, 3, 128, 128).to(device)
#     output = model(input)
#     print(output.shape)
