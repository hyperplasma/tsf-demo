import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

class Transformer(nn.Module):
    """
    标准时间序列Transformer模型（无patch embedding），用于与PatchTST公平对比。
    输入: [B, input_length, in_chans]
    输出: [B, output_length]
    参数集与PatchTST完全对齐（去除patch相关参数），部分参数仅为对齐保留。
    """
    def __init__(
        self,
        input_length=336,
        output_length=96,
        d_model=128,
        n_heads=16,
        e_layers=3,
        d_ff=256,
        dropout=0.2,
        act='gelu',
        in_chans=21,
        individual=False,
        **kwargs
    ):
        super().__init__()
        self.input_length = input_length
        self.output_length = output_length
        self.d_model = d_model
        self.in_chans = in_chans
        self.individual = individual
        self.res_attention = res_attention
        self.pre_norm = pre_norm
        self.kernel_size = kernel_size
        self.norm_type = norm_type
        self.pred_type = pred_type
        # 1. 输入投影: [B, L, C] -> [B, L, d_model]
        self.input_proj = nn.Linear(in_chans, d_model)
        # 2. 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, input_length, d_model))
        # 3. Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout, batch_first=True, activation=act
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, e_layers)
        self.dropout = nn.Dropout(dropout)
        # 4. 输出投影
        if not individual:
            # 所有变量共享一个预测头，输出[B, output_length]
            self.projection = nn.Linear(input_length * d_model, output_length)
        else:
            # 每变量独立预测头，输出[B, output_length, in_chans]
            self.projection = nn.ModuleList([
                nn.Linear(input_length * d_model, output_length) for _ in range(in_chans)
            ])

    def forward(self, x):
        # x: [B, L, C]
        x = self.input_proj(x)  # [B, L, d_model]
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.dropout(x)
        x = self.encoder(x)     # [B, L, d_model]
        x = x.reshape(x.size(0), -1)  # [B, L*d_model]
        if not self.individual:
            x = self.projection(x)         # [B, output_length]
        else:
            # 每变量独立预测头，输出[B, output_length, in_chans]
            x = torch.stack([proj(x) for proj in self.projection], dim=-1)
        return x

if __name__ == "__main__":
    # 示例：用随机数据测试Transformer模型
    batch_size = 4
    input_length = 336
    in_chans = 21
    output_length = 96
    x = torch.randn(batch_size, input_length, in_chans)
    model = Transformer(input_length=input_length, output_length=output_length, in_chans=in_chans)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")  # 期望: [4, 96] 或 [4, 96, 21]（individual=True时）
