import torchvision

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
]
)
train_set = torchvision.datasets.CIFAR10(root='/home/zhangxiaohong/zhouxingyu/demo/python/CIFAR10_datasets',
                                         transform=dataset_transform, train=True, download=True)
test_set = torchvision.datasets.CIFAR10(root='/home/zhangxiaohong/zhouxingyu/demo/python/CIFAR10_datasets',
                                        transform=dataset_transform, train=False, download=True)

# print(test_set[0])
# print(test_set.classes)
# img, target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()
