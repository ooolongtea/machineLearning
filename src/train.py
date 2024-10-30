import datetime

import torch.optim
import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from nm_model import *

# 添加tensorboard
# 获取当前时间戳
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f'../logs/{timestamp}'
writer = SummaryWriter(log_dir)

# transform = torchvision.transforms.Compose([
#     torchvision.transforms.RandomHorizontalFlip(),
#     torchvision.transforms.RandomCrop(32, padding=4),
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 标准化
# ])

# 数据
train_data = torchvision.datasets.CIFAR10(root='../data', train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10(root='../data', train=False, download=True,
                                         transform=torchvision.transforms.ToTensor())
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f'Train data size: {train_data_size}')
print(f'Test data size: {test_data_size}')

train_dataLoader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataLoader = DataLoader(test_data, batch_size=64, shuffle=True)

# 参数设置
total_train_step = 0
total_test_step = 0
epoch = 30
best_acc = 0
warmup_steps = 10

# 神经网络模型设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
net = Net().to(device)
loss_fn = nn.CrossEntropyLoss().to(device)
learning_rate = 1e-2
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)
# 学习率调度器，lr_lambda返回一个浮动因子，lr=该因子*lr
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                              lr_lambda=lambda step: min(1,
                                                                         step / warmup_steps) if step < warmup_steps else 1)

for i in range(epoch):
    print(f'Epoch {i + 1}---------------------------------------------------------------------')
    train_loss = 0

    net.train()
    for data in train_dataLoader:
        image, labels = data
        image, labels = image.to(device), labels.to(device)
        output = net(image)
        loss = loss_fn(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        total_train_step += 1
        if total_train_step % 100 == 0:
            # print(f'Loss: {loss.item():.4f},total_train_step: {total_train_step}')
            writer.add_scalar('train_loss', loss.item(), total_train_step)
    scheduler.step()

    # 测试
    total_test_loss = 0
    total_accuracy = 0
    net.eval()
    with torch.no_grad():
        for data in test_dataLoader:
            image, labels = data
            image, labels = image.to(device), labels.to(device)
            output = net(image)
            loss = loss_fn(output, labels)
            total_test_loss += loss.item()

            # 正确率
            accuracy = (output.argmax(dim=1) == labels).sum()
            total_accuracy += accuracy.item()

    # print(f'Test Loss: {total_test_loss}')
    print(f'Train Loss: {train_loss:.4f}')
    print(f'Test Loss: {total_test_loss / test_data_size:.4f},Test Accuracy: {total_accuracy / test_data_size:.4f}')
    writer.add_scalar('test_loss', total_test_loss, i)
    writer.add_scalar('test_accuracy', total_accuracy, i)

    # 模型保存
    if total_accuracy / len(test_data) > best_acc:
        best_acc = total_accuracy / len(test_data)
        torch.save(net.state_dict(), f'../models/model.pt')
        print('saving model with acc {:.3f}'.format(best_acc))

writer.close()
