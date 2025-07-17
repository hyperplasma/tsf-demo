import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import cast

class PatchEmbedding(nn.Module):
    """
    Patch Embedding for time series (Conv1d方式)。
    """
    def __init__(self, patch_len, stride, d_model, in_chans):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.in_chans = in_chans
        self.proj = nn.Conv1d(in_chans, d_model, kernel_size=patch_len, stride=stride)

    def forward(self, x):
        # x: [B, L, C] -> [B, C, L]
        x = x.transpose(1, 2)
        x = self.proj(x)  # [B, d_model, N_patches]
        x = x.transpose(1, 2)  # [B, N_patches, d_model]
        return x

class SeriesDecomp(nn.Module):
    """
    Series decomposition block for PatchTST (moving average)。
    """
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def moving_avg(self, x):
        # x: [B, L, C]
        x_padded = nn.functional.pad(x, (0, 0, self.padding, self.padding), mode='replicate')
        avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1, padding=0)
        return avg(x_padded.transpose(1, 2)).transpose(1, 2)

    def forward(self, x):
        # x: [B, L, C]
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class PatchTST(nn.Module):
    """
    PatchTST: Transformer-based model for time series forecasting.
    输入: [B, input_length, in_chans]
    输出: [B, output_length]
    """
    def __init__(
        self,
        input_length=336,     # 输入序列长度
        output_length=96,     # 预测序列长度
        patch_len=16,        # Patch长度
        stride=8,            # Patch步长
        d_model=128,         # Transformer隐藏层维度
        n_heads=16,          # 注意力头数
        e_layers=3,          # 编码器层数
        d_ff=256,            # 前馈层维度
        dropout=0.2,         # 丢弃率
        act='gelu',          # 激活函数
        in_chans=21,         # 输入变量数
        individual=False,    # True：每个变量一个预测头，False：所有变量共享一个预测头
        **kwargs
    ):
        super().__init__()
        self.input_length = input_length
        self.output_length = output_length
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.in_chans = in_chans
        self.individual = individual
        self.patch_embed = PatchEmbedding(patch_len, stride, d_model, in_chans)
        self.num_patches = (input_length - patch_len) // stride + 1
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True, activation=act)
        self.encoder = nn.TransformerEncoder(encoder_layer, e_layers)
        self.dropout = nn.Dropout(dropout)
        # projection: flatten所有patch后映射到output_length
        if self.individual:
            self.projection = nn.ModuleList([
                nn.Linear(self.num_patches * d_model, output_length) for _ in range(in_chans)
            ])
        else:
        self.projection = nn.Linear(self.num_patches * d_model, output_length)

    def forward(self, x):
        # x: [B, L, C]
        x = self.patch_embed(x)  # [B, N_patches, d_model]
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.dropout(x)
        x = self.encoder(x)
        x = x.reshape(x.size(0), -1)  # [B, N_patches*d_model]
        if self.individual:
            projection_list = cast(nn.ModuleList, self.projection)
            outs = []
            for proj in projection_list:
                outs.append(proj(x))
            x = torch.stack(outs, dim=-1)  # [B, output_length, in_chans]
        else:
        x = self.projection(x)        # [B, output_length]
        return x

if __name__ == "__main__":
    # 示例：用随机数据测试PatchTST模型
    batch_size = 4
    input_length = 336
    in_chans = 21
    output_length = 96
    x = torch.randn(batch_size, input_length, in_chans)
    # 测试多变量预测分支
    model = PatchTST(input_length=input_length, output_length=output_length, in_chans=in_chans, individual=True)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")  # [4, 96, 21]
