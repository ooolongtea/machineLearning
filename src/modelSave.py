import torch
import torchvision

vgg16 = torchvision.models.vgg16(weights=None)
# 保存模型和参数(自定义后需要from xxx import xxx)
# torch.save(vgg16, 'models/vgg16_model1.pth')
vgg16_model1 = torch.load('../models/vgg16_model1.pth')
print(vgg16_model1)

# 把参数保存为字典
# torch.save(vgg16.state_dict(), 'models/vgg16_model2.pth')
vgg16_model2 = torchvision.models.vgg16(weights=None)
vgg16_model2.load_state_dict(torch.load('../models/vgg16_model2.pth'))
print(vgg16_model2)
