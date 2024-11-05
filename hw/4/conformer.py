import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention, Conv1d, LayerNorm, Dropout


class FeedForwardModule(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super(FeedForwardModule, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

    def forward(self, x):
        return self.dropout(self.linear2(F.relu(self.linear1(x))))


class ConvolutionModule(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(ConvolutionModule, self).__init__()
        self.pointwise_conv1 = Conv1d(d_model, d_model * 2, kernel_size=1)
        self.depthwise_conv = Conv1d(d_model * 2, d_model * 2, kernel_size=3, padding=1, groups=d_model * 2)
        self.batch_norm = nn.BatchNorm1d(d_model * 2)
        self.pointwise_conv2 = Conv1d(d_model * 2, d_model, kernel_size=1)
        self.dropout = Dropout(dropout)

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        x = x.transpose(0, 1).transpose(1, 2)  # to (batch_size, d_model, seq_len)
        x = F.glu(self.pointwise_conv1(x), dim=1)
        x = self.batch_norm(self.depthwise_conv(x))
        x = F.relu(x)
        x = self.dropout(self.pointwise_conv2(x))
        x = x.transpose(1, 2).transpose(0, 1)  # back to (seq_len, batch_size, d_model)
        return x


class ConformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(ConformerBlock, self).__init__()
        self.norm1 = LayerNorm(d_model)
        self.feed_forward1 = FeedForwardModule(d_model, dim_feedforward, dropout)

        self.norm2 = LayerNorm(d_model)
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm3 = LayerNorm(d_model)
        self.conv_module = ConvolutionModule(d_model, dropout)

        self.norm4 = LayerNorm(d_model)
        self.feed_forward2 = FeedForwardModule(d_model, dim_feedforward, dropout)

        self.dropout = Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # First Feed Forward Module with residual connection
        src = src + 0.5 * self.dropout(self.feed_forward1(self.norm1(src)))

        # Multi-Head Self Attention with residual connection
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout(src2)
        src = self.norm2(src)

        # Convolution Module with residual connection
        src = src + self.dropout(self.conv_module(self.norm3(src)))

        # Second Feed Forward Module with residual connection
        src = src + 0.5 * self.dropout(self.feed_forward2(self.norm4(src)))

        return src


class ConformerEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_layers=6, dim_feedforward=512, dropout=0.1):
        super(ConformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            ConformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        for layer in self.layers:
            src = layer(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return self.norm(src)


# Example usage:
d_model = 256
nhead = 4
num_layers = 6
dim_feedforward = 512
dropout = 0.1

encoder = ConformerEncoder(d_model=d_model, nhead=nhead, num_layers=num_layers, dim_feedforward=dim_feedforward,
                           dropout=dropout)
src = torch.rand(50, 8, d_model)  # (seq_len, batch_size, d_model)
output = encoder(src)
print(output.shape)  # Expected output shape: (seq_len, batch_size, d_model)
