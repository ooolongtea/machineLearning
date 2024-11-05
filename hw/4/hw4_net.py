import json
import math
import os
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Module, MultiheadAttention, Conv1d, LayerNorm, Dropout
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# v2 dim_feedforward=1024改为512

class myDataset(Dataset):
    def __init__(self, data_dir, segment_len=128):
        self.data_dir = data_dir
        self.segment_len = segment_len

        # 加载发言者-id映射
        mapping_path = Path(data_dir) / "mapping.json"  # Path类使用/拼接
        mapping = json.load(mapping_path.open())
        self.speaker2id = mapping["speaker2id"]

        # 加载元数据
        metadata_path = Path(data_dir) / "metadata.json"
        metadata = json.load(open(metadata_path))["speakers"]

        # 提取每个发言者的语音特征路径和对应id
        self.speaker_num = len(metadata.keys())
        self.data = []
        for speaker in metadata.keys():
            for utterances in metadata[speaker]:
                self.data.append([utterances["feature_path"], self.speaker2id[speaker]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        feat_path, speaker = self.data[index]
        # 加载预处理的梅尔频谱图
        mel = torch.load(os.path.join(self.data_dir, feat_path),weights_only=True)

        # Segmemt长度超过了指定的segment_len
        if len(mel) > self.segment_len:
            # 从[0, len(mel) - self.segment_len]选一个起始点
            start = random.randint(0, len(mel) - self.segment_len)
            # Get a segment with "segment_len" frames.
            mel = torch.FloatTensor(mel[start:start + self.segment_len])
        else:
            mel = torch.FloatTensor(mel)
        # Turn the speaker id into long for computing loss later.
        speaker = torch.FloatTensor([speaker]).long()
        return mel, speaker

    def get_speaker_number(self):
        return self.speaker_num


def collate_batch(batch):
    # Process features within a batch.
    """Collate a batch of data."""
    mel, speaker = zip(*batch)
    # Because we train the model batch by batch, we need to pad the features in the same batch to make their lengths the same.
    mel = pad_sequence(mel, batch_first=True, padding_value=-20)  # pad log 10^(-20) which is very small value.
    # mel: (batch size, length, 40)
    return mel, torch.FloatTensor(speaker).long()


def get_dataloader(data_dir, batch_size, n_workers):
    """Generate dataloader"""
    dataset = myDataset(data_dir)
    speaker_num = dataset.get_speaker_number()
    # Split dataset into training dataset and validation dataset
    trainlen = int(0.9 * len(dataset))
    lengths = [trainlen, len(dataset) - trainlen]  # 训练集和验证集的长度
    trainset, validset = random_split(dataset, lengths)

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )
    valid_loader = DataLoader(
        validset,
        batch_size=batch_size,
        num_workers=n_workers,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_batch,
    )

    return train_loader, valid_loader, speaker_num


# 卷积变压器编码器层
class Smoother(Module):
    """Convolutional Transformer Encoder Layer"""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout=0.1):
        super(Smoother, self).__init__()
        # 多头自注意力层
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        self.conv1 = Conv1d(d_model, dim_feedforward, 5, padding=2)
        self.conv2 = Conv1d(dim_feedforward, d_model, 1, padding=0)  # 卷积核大小为1-点卷积

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(
            self,
            src: Tensor,
            # 用于指示在计算注意力时哪些位置应该被忽略，值为 True 表示相应位置被遮蔽(seq_len, seq_len)
            src_mask: Optional[Tensor] = None,
            # 用于指定输入中填充的部分，也应该被忽略(batch_size, seq_len)
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal:Optional[bool] = False,
    ) -> Tensor:
        # multi-head self attention
        src2 = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]

        # 残差连接
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # shape(seq_len, batch_size, embed_dim)
        # conv1d
        src2 = src.transpose(0, 1).transpose(1, 2)
        # shape(batch_size, embed_dim, seq_len)
        src2 = self.conv2(F.relu(self.conv1(src2)))
        src2 = src2.transpose(1, 2).transpose(0, 1)

        # add & norm
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class Classifier(nn.Module):
    def __init__(self, d_model=128, n_spks=600, dropout=0.1):
        super().__init__()
        # 映射到d_model
        self.prenet = nn.Linear(40, d_model)
        #   Change Transformer to Conformer.
        #   https://arxiv.org/abs/2005.08100
        self.encoder_layer = Smoother(
            d_model=d_model, dim_feedforward=512, nhead=16, dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        # Project the  dimension of features from d_model into speaker nums.
        self.pred_layer = nn.Linear(d_model, n_spks)

    def forward(self, mels):
        """
        args:
          mels: (batch size, length, 40)
        return:
          out: (batch size, n_spks)
        """
        # out: (batch size, length, d_model)
        out = self.prenet(mels)
        # out: (length, batch size, d_model)
        out = out.permute(1, 0, 2)
        # encoder层期望输入维度(length, batch size, d_model).
        out = self.encoder(out)
        # out: (batch size, length, d_model)
        out = out.transpose(0, 1)
        # mean pooling
        stats = out.mean(dim=1)

        # out: (batch, n_spks)
        out = self.pred_layer(stats)
        return out



def get_cosine_schedule_with_warmup(
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        # 学习率衰减的波动次数（这里是学习率减少一次，达到0）
        num_cycles: float = 0.5,
        last_epoch: int = -1,
):


    def lr_lambda(current_step):
        # 学习率热身
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # 余弦退火
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            # 余弦函数的输出范围是[-1.1],调整为[0，1]
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def model_fn(batch, model, criterion, device):
    """Forward a batch through the model."""

    mels, labels = batch
    mels = mels.to(device)
    labels = labels.to(device)

    outs = model(mels)

    loss = criterion(outs, labels)

    # Get the speaker id with the highest probability.
    # 返回的是max的索引
    preds = outs.argmax(1)
    # 计算张量平均值
    accuracy = torch.mean((preds == labels).float())

    return loss, accuracy


def valid(dataloader, model, criterion, device):
    """Validate on validation set."""

    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    pbar = tqdm(total=len(dataloader.dataset), ncols=0, desc="Valid", unit=" uttr")

    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            loss, accuracy = model_fn(batch, model, criterion, device)
            running_loss += loss.item()
            running_accuracy += accuracy.item()

        pbar.update(dataloader.batch_size)
        pbar.set_postfix(
            loss=f"{running_loss / (i + 1):.2f}",
            accuracy=f"{running_accuracy / (i + 1):.2f}",
        )

    pbar.close()
    model.train()

    return running_accuracy / len(dataloader)


def parse_args():
    """arguments"""
    config = {
        "data_dir": "../../dataset/Dataset",
        "save_path": "../../models/hw4/model_v2.ckpt",
        "log_path":"../../logs/hw4/model_v2.log",
        "batch_size": 32,
        "n_workers": 8,
        "valid_steps": 2000,
        "warmup_steps": 1000,
        "save_steps": 10000,
        "total_steps": 70000,
    }

    return config


def main(
        data_dir,
        save_path,
        log_path,
        batch_size,
        n_workers,
        valid_steps,
        warmup_steps,
        total_steps,
        save_steps,
):
    """Main function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info]: Use {device} now!")

    train_loader, valid_loader, speaker_num = get_dataloader(data_dir, batch_size, n_workers)
    train_iterator = iter(train_loader)
    print(f"[Info]: Finish loading data!", flush=True)

    model = Classifier(n_spks=speaker_num).to(device)
    state_dict = torch.load("../../models/hw4/model_v1.ckpt", map_location=device)
    # model.load_state_dict(state_dict)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"[Info]: Finish creating model!", flush=True)

    writer = SummaryWriter(log_dir=log_path)

    best_accuracy = -1.0
    best_state_dict = None

    pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

    for step in range(total_steps):
        # Get data
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        loss, accuracy = model_fn(batch, model, criterion, device)
        batch_loss = loss.item()
        batch_accuracy = accuracy.item()

        # Updata model
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Log
        pbar.update()
        pbar.set_postfix(
            loss=f"{batch_loss:.2f}",
            accuracy=f"{batch_accuracy:.2f}",
            step=step + 1,
        )

        writer.add_scalar("train_loss", batch_loss, step)
        writer.add_scalar("train_accuracy", batch_accuracy, step)


        # Do validation
        if (step + 1) % valid_steps == 0:
            pbar.close()

            valid_accuracy = valid(valid_loader, model, criterion, device)
            writer.add_scalar("valid_accuracy", valid_accuracy, step)

            # keep the best model
            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                best_state_dict = model.state_dict()

            pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

        # Save the best model so far.
        if (step + 1) % save_steps == 0 and best_state_dict is not None:
            torch.save(best_state_dict, save_path)
            pbar.write(f"Step {step + 1}, best model saved. (accuracy={best_accuracy:.4f})")

    pbar.close()


if __name__ == "__main__":
    main(**parse_args())
    # Step 70000, best model saved. (accuracy=0.8654)
