import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import datetime

# 下载后解压缩
# gdown --id '1HPkcmQmFGu-3OknddKIa5dNDsR05lIQR' --output data.zip
# unzip data.zip
# ls

class TIMITDataset(Dataset):
    def __init__(self, X, y=None, window_size=11):
        self.window_size=window_size
        self.data = torch.from_numpy(X).float()
        if y is not None:
            y = y.astype(int)
            self.label = torch.LongTensor(y)#改标签
        else:
            self.label = None

        self.windowed_data = self.create_windows(self.data)
        if self.label is not None:
            self.windowed_labels = self.create_window_labels(self.label)

    def create_windows(self, data):
        # 确保数据是二维的
        num_samples = data.shape[0]
        half_window = self.window_size // 2
        windows = []

        for i in range(half_window, num_samples - half_window):
            window = data[i - half_window:i + half_window + 1]  # 包含上下各5个点(左闭右开)
            windows.append(window)

        return torch.stack(windows)

    def create_window_labels(self, labels):
        half_window = self.window_size // 2
        return labels[half_window:-half_window]  # 标签与窗口中心对齐

    def __getitem__(self, idx):
        if self.label is not None:
            return self.windowed_data[idx], self.windowed_labels[idx]
        else:
            return self.windowed_data[idx]

    def __len__(self):
        return len(self.windowed_data)


# 模型
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.layer1 = nn.Linear(11*429, 1024)
        # 添加归一化
        self.bn1 = nn.BatchNorm1d(1024)
        # 添加正则化
        self.dropout1 = nn.Dropout(p=0.5)
        self.layer2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(p=0.5)
        self.layer3 = nn.Linear(512, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(p=0.5)
        self.out = nn.Linear(128, 39)

        self.act_fn = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.act_fn(x)

        x = self.layer2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.act_fn(x)

        x = self.layer3(x)
        x = self.bn3(x)
        x = self.dropout3(x)
        x = self.act_fn(x)

        x = self.out(x)

        return x

#check device
def get_device():
  return 'cuda' if torch.cuda.is_available() else 'cpu'

def same_seeds(seed):
    # 设置pytorch随机数生成器种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
# 获取当前时间戳
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
writer=SummaryWriter(f'../logs/hw2/{timestamp}')

data_root='../dataset/timit_11/'
train = np.load(data_root + 'train_11.npy')
train_label = np.load(data_root + 'train_label_11.npy')
test = np.load(data_root + 'test_11.npy')


VAL_RATIO = 0.2
percent = int(train.shape[0] * (1 - VAL_RATIO))
# 切片 [:percent]是0-percent，区分了训练集和验证集
train_x, train_y, val_x, val_y = train[:percent], train_label[:percent], train[percent:], train_label[percent:]
print('Size of training set: {}'.format(train_x.shape))# Size of training set: (983945, 429)表示983945个数据，每个数据的特征值有429个
print('Size of validation set: {}'.format(val_x.shape))

BATCH_SIZE = 64
train_set = TIMITDataset(train_x, train_y)
val_set = TIMITDataset(val_x, val_y)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True) #only shuffle the training data
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

# fix random seed for reproducibility
same_seeds(0)

# get device
device = get_device()
print(f'DEVICE: {device}')

# 参数设置
num_epoch = 30               # number of training epoch
# 学习率热身
learning_rate = 0.0001       # learning rate
warmup_steps = 8

# the path where checkpoint saved
model_path = 'model_moregood.ckpt'

# create model, define a loss function, and optimizer
model = Classifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# 学习率调度器，lr_lambda返回一个浮动因子，lr=该因子*lr
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
            lr_lambda=lambda step: min(1, step / warmup_steps) if step < warmup_steps else 1)
# 开始训练

best_acc = 0.0
for epoch in range(num_epoch):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    # training
    model.train()  # set the model to training mode
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        batch_loss = criterion(outputs, labels)
        _, train_pred = torch.max(outputs, 1)  # get the index of the class with the highest probability
        batch_loss.backward()
        optimizer.step()
        scheduler.step()

        train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
        train_loss += batch_loss.item()

    # validation
    if len(val_set) > 0:
        model.eval()  # set the model to evaluation mode
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                batch_loss = criterion(outputs, labels)
                # 忽略第一个变量，第一个值是最大值，第二个值是最大值的索引,此时沿着第二个维度进行操作
                _, val_pred = torch.max(outputs, 1)
                val_acc += (
                            val_pred.cpu() == labels.cpu()).sum().item()  # get the index of the class with the highest probability
                val_loss += batch_loss.item()

            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                epoch + 1, num_epoch, train_acc / len(train_set), train_loss / len(train_loader),
                val_acc / len(val_set), val_loss / len(val_loader)
            ))
            writer.add_scalar('Train Acc', train_acc / len(train_set), epoch)
            writer.add_scalar('Val Acc', val_acc / len(val_set), epoch)

            # 模型有改进则保存
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), model_path)
                print('saving model with acc {:.3f}'.format(best_acc / len(val_set)))
    else:
        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
            epoch + 1, num_epoch, train_acc / len(train_set), train_loss / len(train_loader)
        ))

# if not validating, save the last epoch
if len(val_set) == 0:
    torch.save(model.state_dict(), model_path)
    print('saving model at last epoch')

writer.close()
