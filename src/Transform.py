# 常用Transform
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter(log_dir='../logs')
img = Image.open("../dataset/train/bees_image/92663402_37f379e57a.jpg")

# 转为tensor对象
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
# def add_image(
#         self, tag, img_tensor, global_step=None, walltime=None, dataformats="CHW"
#     ):
# img_tensor需要(torch.Tensor, numpy.ndarray, or string/blobname)
writer.add_image("To tensor", img_tensor)
writer.close()
