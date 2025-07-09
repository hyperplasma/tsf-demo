import torch
import sys
import os
import torch.nn as nn
import torch.nn.functional as F
from models.PatchTST.layers import PatchEmbedding, SeriesDecomp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

class PatchTST(nn.Module):
    """
    PatchTST: Transformer-based model for time series forecasting.
    """
    def __init__(
        self,
        input_length=336,
        output_length=96,
        patch_len=16,
        stride=8,
        d_model=128,
        n_heads=16,
        e_layers=3,
        d_ff=256,
        dropout=0.2,
        act='gelu',
        res_attention=False,
        pre_norm=True,
        attn_dropout=0.0,
        in_chans=21,
        individual=False,
        kernel_size=25,
        norm_type='BatchNorm',
        pred_type='direct',
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
        self.projection = nn.Linear(self.num_patches * d_model, output_length)

    def forward(self, x):
        # x: [B, L, C]
        x = self.patch_embed(x)  # [B, N_patches, d_model]
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.dropout(x)
        x = self.encoder(x)
        x = x.reshape(x.size(0), -1)  # [B, N_patches*d_model]
        x = self.projection(x)        # [B, output_length]
        return x

if __name__ == "__main__":
    # Test model
    model = PatchTST()
    print(model)
    x = torch.randn(32, 336, 7)  # [B, input_length, in_chans]
    output = model(x)
    print(output.shape)  # Should be [32, 96, 7]