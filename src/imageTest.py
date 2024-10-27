import torchvision.transforms
from PIL import Image
from nm_model import *

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
image_path = "../img/test7.jpg"
image = Image.open(image_path)
print(image)
image = image.convert("RGB")
transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
     torchvision.transforms.Resize((32, 32)), ]
)
image = transform(image)
image = torch.reshape(image, (1, 3, 32, 32)).cuda()
print(image.shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
net = Net().to(device)
net.load_state_dict(torch.load("../models/net_14.path"))
outputs = net(image).argmax(dim=1)
print(f"图片类型为：{classes[outputs[0]]}")
