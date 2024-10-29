# 使用最大池化对图片采样
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

transform = transforms.Compose([transforms.ToTensor()])
image = Image.open("./testdata_kxe/test1.jpg")
image_tenor = transform(image).unsqueeze(0)

max_pool = nn.MaxPool2d(kernel_size=8, stride=8)
pooled_image = max_pool(image_tenor)

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(pooled_image.squeeze(0).permute(1, 2, 0))  # 还原维度，显示图像
plt.title("Pooled Image")

plt.show()
