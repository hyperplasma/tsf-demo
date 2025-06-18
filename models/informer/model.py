import torch
import torch.nn as nn
from .layers import ProbSparseSelfAttention

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = ProbSparseSelfAttention(d_model, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x2 = self.attn(x)
        x = self.norm1(x + x2)
        x2 = self.ff(x)
        x = self.norm2(x + x2)
        return x

class Informer(nn.Module):
    def __init__(self, enc_in, d_model, n_heads, seq_len, pred_len, e_layers=2):
        super().__init__()
        self.embedding = nn.Linear(enc_in, d_model)
        self.encoder = nn.ModuleList([EncoderLayer(d_model, n_heads) for _ in range(e_layers)])
        self.proj = nn.Linear(d_model, enc_in)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.encoder:
            x = layer(x)
        # 只取最后pred_len步
        out = self.proj(x[:, -self.pred_len:, :])
        return out