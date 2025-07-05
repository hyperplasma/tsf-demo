import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """
    Patch Embedding for time series.
    将输入序列切分为patch，并通过线性层映射到高维空间。
    """
    def __init__(self, patch_len, stride, d_model, in_chans):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.in_chans = in_chans
        self.proj = nn.Linear(patch_len * in_chans, d_model)

    def forward(self, x):
        """
        x: [B, L, C]
        return: [B, N_patches, d_model]
        """
        B, L, C = x.shape
        # unfold: [B, N_patches, patch_len, C]
        num_patches = 1 + (L - self.patch_len) // self.stride
        patches = []
        for i in range(num_patches):
            start = i * self.stride
            end = start + self.patch_len
            patch = x[:, start:end, :]  # [B, patch_len, C]
            patches.append(patch)
        patches = torch.stack(patches, dim=1)  # [B, N_patches, patch_len, C]
        patches = patches.reshape(B, num_patches, self.patch_len * C)
        out = self.proj(patches)  # [B, N_patches, d_model]
        return out

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