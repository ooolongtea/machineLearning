import datetime
import torchvision.models as models
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import DatasetFolder
from tqdm.auto import tqdm
from hw3_Residual_Network import *

def load_image(path):
    return Image.open(path)


def same_seeds(seed):
    torch.backends.cudnn.deterministic = True


# 伪标签生成
def get_pseudo_labels(dataset, model, threshold=0.65):
    model.eval()
    torch.cuda.set_device(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    softmax = nn.Softmax(dim=-1)

    pseudo_images = []
    pseudo_labels = []

    for batch in tqdm(data_loader):
        img, _ = batch

        # Forward the data
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(img.to(device))

        # 计算概率
        probs = softmax(logits)
        # 获取最大概率和标签
        max_probs, preds = probs.max(dim=1)
        mask = max_probs >= threshold
        pseudo_images.append(img[mask])
        pseudo_labels.append(preds[mask])

    if pseudo_images and pseudo_labels:
        pseudo_images = torch.cat(pseudo_images, dim=0)
        pseudo_labels = torch.cat(pseudo_labels, dim=0)

        # 创建一个新的 DatasetFolder 来保存伪标签数据集
        pseudo_dataset = DatasetFolder(root='pseudo_labels',
                                       loader=lambda x: Image.fromarray(x),
                                       extensions=".jpg",
                                       transform=test_tfm)

        model.train()
        return pseudo_dataset

    model.train()
    return None


if __name__ == '__main__':
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=f'../../logs/hw3/{timestamp}')

    train_tfm = transforms.Compose([
        # transforms.Resize((128, 128)),
        transforms.Resize((128, 128)),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        # transforms.RandomHorizontalFlip(p=1),  # 随机水平翻转
        # transforms.RandomVerticalFlip(p=1),  # 随机上下翻转
        # transforms.RandomGrayscale(0.5),  # 随机灰度化
        # transforms.RandomSolarize(threshold=192.0),
        # transforms.ColorJitter(brightness=.5, hue=0.5),  # 改变图像的亮度和饱和度
        # transforms.RandomRotation(degrees=(0, 180)),  # 图像随机旋转
        # transforms.RandomInvert(),  # 改变图像的颜色
        # transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_tfm = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    batch_size = 45

    train_set = DatasetFolder("../../dataset/food-11/training/labeled", loader=load_image, extensions="jpg",
                              transform=train_tfm)
    valid_set = DatasetFolder("../../dataset/food-11/validation", loader=load_image, extensions="jpg",
                              transform=test_tfm)
    unlabeled_set = DatasetFolder("../../dataset/food-11/training/unlabeled", loader=load_image,
                                  extensions="jpg",
                                  transform=train_tfm)
    test_set = DatasetFolder("../../dataset/food-11/testing", loader=load_image, extensions="jpg",
                             transform=test_tfm)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    torch.cuda.set_device(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print(torch.cuda.is_available())
    scaler = torch.cuda.amp.GradScaler()

    # 神经网络设置
    # model = Classifier().to(device)
    model=ResidualNetwork().to(device)
    model.device = device
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)

    # 参数设置
    n_epochs = 80
    best_acc = 0

    do_semi = False

    for epoch in range(n_epochs):
        if do_semi:
            pseudo_set = get_pseudo_labels(unlabeled_set, model)
            if pseudo_set is not None:
                concat_dataset = ConcatDataset([train_set, pseudo_set])
                train_loader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                                          pin_memory=True)

        # 开始训练
        model.train()
        train_loss = []
        train_accs = []
        for batch in tqdm(train_loader):
            # imgs, labels = batch
            # imgs, labels = imgs.to(device), labels.to(device)
            # logits = model(imgs.to(device))
            #
            # loss = criterion(logits, labels.to(device))
            # optimizer.zero_grad()
            # loss.backward()
            # grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            # optimizer.step()
            #
            # acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            # train_loss.append(loss.item())
            # train_accs.append(acc)
            with torch.cuda.amp.autocast():
                img, label = batch
                img, label = img.to(device), label.to(device)
                logits = model(img).to(device)
                loss = criterion(logits, label)

            scaler.scale(loss).backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            scaler.step(optimizer)
            scaler.update()

            acc = (logits.argmax(dim=-1) == label.to(device)).float().mean()
            train_loss.append(loss.item())
            train_accs.append(acc)

        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)

        print(f'epoch: {epoch}------------------------------------------------------------')
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)

        # 验证
        model.eval()
        valid_loss = []
        valid_accs = []

        for batch in tqdm(valid_loader):
            imgs, labels = batch

            with torch.no_grad():
                logits = model(imgs.to(device))

            loss = criterion(logits.to(device), labels.to(device))
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            valid_loss.append(loss.item())
            valid_accs.append(acc)

        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)

        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), '../../models/hw3/model.pth')
            print(f'saving model with acc:{valid_acc:.5f}')

        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
        writer.add_scalar('valid/loss', valid_loss, epoch)
        writer.add_scalar('valid/acc', valid_acc, epoch)

        del img, label, logits, loss
        torch.cuda.empty_cache()

    writer.close()
