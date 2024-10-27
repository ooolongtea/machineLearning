import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)
# batch_size：每次从数据集中读取的样本数量，一个batch的大小
# shuffle：是否在每个 epoch 开始时打乱数据集中的样本顺序
# num_workers：使用多少个子进程来并行加载数据
# pin_memory：是否将加载的数据复制到 CUDA 固定内存中
# drop_last：如果true，在数据集不能被整除时，丢弃最后一个不完整的batch
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0)

# for data in test_loader:
#     images, targets = data
#     print(images.shape, targets.shape)
#     # torch.Size([4, 3, 32, 32]) 表示四张图片，每个图片3个通道（RGB），大小是32x32
#     # torch.Size([4]) 表示4个标签
#     print(targets)
writer = SummaryWriter(log_dir='../logs')
for epoch in range(2):
    step = 0
    for data in test_loader:
        images, labels = data
        writer.add_images("Epoch{}".format(epoch), images, step)
        step += 1

writer.close()
# tensorboard --logdir="logs"
