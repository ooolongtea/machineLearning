import torchvision

# vgg16_true = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)  # 模型参数为已训练好的
# vgg16_false = torchvision.models.vgg16(weights=None)  # 模型参数为默认参数
# print(vgg16_true)

train_sets = torchvision.datasets.CIFAR10("./CIFAR10_datasets", train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())
test_sets = torchvision.datasets.CIFAR10("python/CIFAR10_datasets", train=False, download=True,
                                         transform=torchvision.transforms.ToTensor())
# 修改vgg
