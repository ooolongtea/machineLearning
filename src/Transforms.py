from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter(log_dir='/home/zhangxiaohong/zhouxingyu/demo/python/logs')
img = Image.open("/home/zhangxiaohong/zhouxingyu/demo/dataset/train/bees/92663402_37f379e57a.jpg")

# 转为tensor对象
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
# add_image需要(torch.Tensor, numpy.ndarray, or string/blobname)
writer.add_image("To tensor", img_tensor)

# 提供三个均值三个标准差
# output[channel] = (input[channel] - mean[channel]) / std[channel]
trans_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalized Image", img_norm)
writer.close()

transforms.RandomCrop
