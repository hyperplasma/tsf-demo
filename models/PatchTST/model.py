import torch
import sys
import os
import torch.nn as nn
import torch.nn.functional as F
from layers import PatchEmbedding, SeriesDecomp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from models.common.utils import get_positional_encoding

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
        in_chans=7,
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
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.act = act
        self.res_attention = res_attention
        self.pre_norm = pre_norm
        self.attn_dropout = attn_dropout
        self.in_chans = in_chans
        self.individual = individual
        self.kernel_size = kernel_size
        self.norm_type = norm_type
        self.pred_type = pred_type

        # Patch Embedding
        self.patch_embed = PatchEmbedding(
            patch_len=self.patch_len,
            stride=self.stride,
            d_model=self.d_model,
            in_chans=self.in_chans
        )

        # Number of patches
        self.num_patches = 1 + (self.input_length - self.patch_len) // self.stride

        # Positional encoding
        self.pos_embed = nn.Parameter(
            get_positional_encoding(self.num_patches, self.d_model), requires_grad=False
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
            activation=self.act,
            batch_first=True,
            norm_first=self.pre_norm
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.e_layers,
            norm=nn.LayerNorm(self.d_model) if self.pre_norm else None
        )

        # Series Decomposition
        self.decomp = SeriesDecomp(self.kernel_size)

        # Output head
        if self.individual:
            self.head = nn.ModuleList([
                nn.Linear(self.num_patches * self.d_model, self.output_length)
                for _ in range(self.in_chans)
            ])
        else:
            self.head = nn.Linear(self.num_patches * self.d_model, self.output_length)

        # Normalization
        if self.norm_type == 'BatchNorm':
            self.norm = nn.BatchNorm1d(self.in_chans)
        elif self.norm_type == 'LayerNorm':
            self.norm = nn.LayerNorm(self.in_chans)
        else:
            self.norm = None

    def forward(self, x):
        """
        x: [B, input_length, in_chans]
        """
        # Normalization (fix: transpose for BatchNorm1d)
        if self.norm is not None:
            if isinstance(self.norm, nn.BatchNorm1d):
                x = x.transpose(1, 2)  # [B, in_chans, input_length]
                x = self.norm(x)
                x = x.transpose(1, 2)  # [B, input_length, in_chans]
            else:
                x = self.norm(x)

        # Series Decomposition
        seasonal_init, trend_init = self.decomp(x)

        # Patch Embedding
        x_patch = self.patch_embed(seasonal_init)  # [B, N_patches, d_model]

        # Add positional encoding
        x_patch = x_patch + self.pos_embed

        # Transformer Encoder
        x_enc = self.encoder(x_patch)  # [B, N_patches, d_model]

        # Flatten
        x_flat = x_enc.reshape(x_enc.shape[0], -1)  # [B, N_patches*d_model]

        # Output head
        if self.individual:
            outputs = []
            for i in range(self.in_chans):
                out = self.head[i](x_flat).unsqueeze(-1)  # [B, output_length, 1]
                outputs.append(out)
            out = torch.cat(outputs, dim=-1)  # [B, output_length, in_chans]
        else:
            out = self.head(x_flat)  # [B, output_length]
            out = out.unsqueeze(-1).repeat(1, 1, self.in_chans)  # [B, output_length, in_chans]

        # Trend part direct extrapolation
        trend_patch = trend_init[:, -self.output_length:, :]

        # Restore
        output = out + trend_patch

        return output  # [B, output_length, in_chans]

if __name__ == "__main__":
    # Test model
    model = PatchTST()
    print(model)
    x = torch.randn(32, 336, 7)  # [B, input_length, in_chans]
    output = model(x)
    print(output.shape)  # Should be [32, 96, 7]