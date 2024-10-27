import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from nm_model import *

# 添加tensorboard
writer = SummaryWriter('../logs')

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

# 神经网络模型设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
net = Net().to(device)
loss_fn = nn.CrossEntropyLoss().to(device)
learning_rate = 1e-2
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

# 参数设置
total_loss = 0
total_train_step = 0
total_test_step = 0
epoch = 20
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
            print(f'Loss: {loss.item():.4f},total_train_step: {total_train_step}')
            writer.add_scalar('train_loss', loss.item(), total_train_step)

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

    print(f'Test Loss: {total_test_loss}')
    print(f'Test Accuracy: {total_accuracy / len(test_data)}')
    writer.add_scalar('test_loss', total_test_loss, i)
    writer.add_scalar('test_accuracy', total_accuracy, i)

    # 模型保存
    torch.save(net.state_dict(), f'models/net_{i}.path')
writer.close()
