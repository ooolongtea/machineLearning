import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, transform=dataset_transform, download=True)

# img, target = test_set[0]
# print(test_set[0])  # (<PIL.Image.Image image mode=RGB size=32x32 at 0x18668E6C280>, 3)
# print(test_set.classes)  # ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# print(test_set.classes[target])  # cat
# print(img)  # <PIL.Image.Image image mode=RGB size=32x32 at 0x21B532AC220>
# img.show()

writer = SummaryWriter(log_dir="log")
for i in range(10):
    img, target = test_set[i]
    writer.add_image('img', img, i)
    print(i, " ", img, " ", target)
print("done")
writer.close()
