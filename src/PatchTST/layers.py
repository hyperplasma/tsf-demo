import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    """
    Patch Embedding for time series (官方实现：Conv1d方式)。
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
    Series decomposition block for PatchTST (moving average).
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