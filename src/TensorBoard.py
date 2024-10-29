# from PIL import Image
# from numpy.distutils.tests.test_npy_pkg_config import simple
# from torch.utils.tensorboard import SummaryWriter
# import numpy as np
#
# writer = SummaryWriter("logs")
# image_path = "/home/zhangxiaohong/zhouxingyu/demo/data/train/ants_image/0013035.jpg"
# img_PIL = Image.open(image_path)
# img_array = np.array(img_PIL)
# # writer.add_image()读取格式:img_tensor (torch.Tensor, numpy.ndarray, or string/blobname): Image data
# writer.add_image("train", img_array, 1, dataformats='HWC')
# # eg.fx=x
# for i in range(100):
#     # 名，y，x
#     writer.add_scalar("fx=x", i, i)
# writer.close()
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)
# batch_size：每次从数据集中# # 读取的样本数量，一个batch的大小
# shuffle：是否在每个 epoch 开始时打乱数据集中的样本顺序
# num_workers：使用多少个子进程来并行加载数据
# pin_memory：是否将加载的数据复制到 CUDA 固定内存中
# drop_last：如果true，在数据集不能被整除时，丢弃最后一个不完整的batch
test_loader = DataLoader(dataset=test_data, batch_size=4, shuffle=True, num_workers=0)

writer = SummaryWriter(log_dir='/home/zhangxiaohong/zhouxingyu/demo/python/logs')
for epoch in range(2):
    step = 0
    for data in test_loader:
        images, labels = data
        writer.add_images("Epoch{}".format(epoch), images, step)
        step += 1

writer.close()
