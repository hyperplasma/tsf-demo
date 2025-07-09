import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    """
    标准多层LSTM时间序列预测模型，接口与PatchTST等兼容。
    支持双向、individual、多步输出、标准化（norm_type）、激活函数等。
    输入: [B, input_length, in_chans]
    输出: [B, output_length] 或 [B, output_length, in_chans]（individual=True）
    """
    def __init__(
        self,
        input_length=336,         # 输入序列长度
        output_length=96,         # 预测序列长度
        d_model=128,             # LSTM隐藏层维度
        n_layers=2,              # LSTM层数
        dropout=0.2,             # 丢弃率
        act='gelu',              # 激活函数
        in_chans=21,             # 输入变量数
        individual=False,        # True：每个变量一个预测头，False：所有变量共享一个预测头
        bidirectional=False,     # 是否双向LSTM
        norm_type='BatchNorm',   # 标准化类型
        pred_type='direct',      # 预测类型（direct：最后hidden，multi：多步输出）
        **kwargs
    ):
        super().__init__()
        self.input_length = input_length
        self.output_length = output_length
        self.d_model = d_model
        self.in_chans = in_chans
        self.individual = individual
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.norm_type = norm_type
        self.pred_type = pred_type
        self.act = getattr(F, act) if hasattr(F, act) else F.gelu
        self.num_directions = 2 if bidirectional else 1

        # 标准化层
        if norm_type == 'BatchNorm':
            self.norm = nn.BatchNorm1d(in_chans)
        elif norm_type == 'LayerNorm':
            self.norm = nn.LayerNorm(in_chans)
        elif norm_type == 'GroupNorm':
            self.norm = nn.GroupNorm(1, in_chans)
        elif norm_type == 'InstanceNorm':
            self.norm = nn.InstanceNorm1d(in_chans)
        else:
            self.norm = nn.Identity()

        self.lstm = nn.LSTM(
            input_size=in_chans,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        # 输出层
        proj_in_dim = d_model * self.num_directions
        if not individual:
            self.projection = nn.Linear(proj_in_dim, output_length)
        else:
            self.projection = nn.ModuleList([
                nn.Linear(proj_in_dim, output_length) for _ in range(in_chans)
            ])

    def forward(self, x):
        # x: [B, L, C]
        # 标准化
        if isinstance(self.norm, nn.BatchNorm1d) or isinstance(self.norm, nn.InstanceNorm1d) or isinstance(self.norm, nn.GroupNorm):
            x = x.transpose(1, 2)  # [B, C, L]
            x = self.norm(x)
            x = x.transpose(1, 2)  # [B, L, C]
        else:
            x = self.norm(x)
        # LSTM
        out, _ = self.lstm(x)  # [B, L, d_model*num_directions]
        # 多步输出方式：可选用最后output_length步，也可用最后一个hidden映射
        if self.pred_type == 'direct':
            # 只用最后一个hidden
            last_hidden = out[:, -1, :]  # [B, d_model*num_directions]
            if not self.individual:
                y = self.projection(last_hidden)  # [B, output_length]
            else:
                y = torch.stack([proj(last_hidden) for proj in self.projection], dim=-1)  # [B, output_length, in_chans]
        elif self.pred_type == 'multi':
            # 用最后output_length步的hidden
            multi_hidden = out[:, -self.output_length:, :]  # [B, output_length, d_model*num_directions]
            if not self.individual:
                # 投影到1维后 squeeze
                y = self.projection(multi_hidden).squeeze(-1)  # [B, output_length]
            else:
                y = torch.stack([proj(multi_hidden) for proj in self.projection], dim=-1)  # [B, output_length, in_chans]
        else:
            raise ValueError(f"Unknown pred_type: {self.pred_type}")
        # 激活
        y = self.act(y)
        return y

if __name__ == "__main__":
    batch_size = 4
    input_length = 336
    in_chans = 21
    output_length = 96
    x = torch.randn(batch_size, input_length, in_chans)
    model = LSTM(input_length=input_length, output_length=output_length, in_chans=in_chans, bidirectional=True, norm_type='LayerNorm', pred_type='direct')
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")  # [4, 96] 或 [4, 96, 21]（individual=True时）
