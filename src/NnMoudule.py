# 卷积层
import torch
import numpy as np
import torch.nn.functional as F

m, n = -8, 8
random_input = torch.tensor(np.random.randint(m, n, size=(5, 5)))
random_input = torch.reshape(random_input, (1, 1, 5, 5))  # 1batch size 1channel
print(random_input)
# 卷积核
kernel = torch.tensor(np.random.randint(0, 2, size=(3, 3)))
kernel = torch.reshape(kernel, (1, 1, 3, 3))
print(kernel)
# conv2d函数(输入，卷积核，偏置，步幅，输入边缘填充的像素数)
output = F.conv2d(random_input, kernel, bias=torch.tensor([2]), stride=1, padding=0)
print(output)
print(random_input.shape)
print(output.shape)
# rel = 0
# for i in range(3):
#     for j in range(3):
#         temp = random_input[0, 0, i, j] * kernel[0, 0, i, j]
#         rel += temp
# print(rel)
