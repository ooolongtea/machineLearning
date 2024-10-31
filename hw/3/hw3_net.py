import torch.nn as nn
import torchvision.models as models


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        # 使用 VGG16 作为特征提取器
        self.feature_extractor = models.vgg16(weights=None).features  # 不使用预训练参数

        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 8 * 8, 256),  # 调整输入大小
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 11)  # 输出类别数
        )

    def forward(self, x):
        # 输入形状: [batch_size, 3, 128, 128]

        # 先通过 VGG16 特征提取器
        x = self.feature_extractor(x)
        print(x.shape)

        # 然后通过原有的 CNN 层
        x = self.cnn_layers(x)

        # 展平特征
        x = x.flatten(1)

        # 通过全连接层得到最终输出
        x = self.fc_layers(x)
        return x
