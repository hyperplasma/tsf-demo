import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B, Lq, D = q.size()
        Lk = k.size(1)
        q = self.q_linear(q).view(B, Lq, self.n_heads, self.d_k).transpose(1,2)  # [B, n_heads, Lq, d_k]
        k = self.k_linear(k).view(B, Lk, self.n_heads, self.d_k).transpose(1,2)
        v = self.v_linear(v).view(B, Lk, self.n_heads, self.d_k).transpose(1,2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B, n_heads, Lq, Lk]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)  # [B, n_heads, Lq, d_k]
        context = context.transpose(1,2).contiguous().view(B, Lq, D)
        return self.out(context)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, activation='gelu'):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        if activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation = F.relu
        else:
            raise ValueError("activation must be 'gelu' or 'relu'")

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, activation='gelu'):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        attn = self.self_attn(x, x, x, src_mask)
        x = x + self.dropout1(attn)
        x = self.norm1(x)
        ff = self.ff(x)
        x = x + self.dropout2(ff)
        x = self.norm2(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, activation='gelu'):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        attn1 = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout1(attn1)
        x = self.norm1(x)
        attn2 = self.cross_attn(x, memory, memory, memory_mask)
        x = x + self.dropout2(attn2)
        x = self.norm2(x)
        ff = self.ff(x)
        x = x + self.dropout3(ff)
        x = self.norm3(x)
        return x

class Encoder(nn.Module):
    def __init__(self, input_dim, d_model, n_layers, n_heads, d_ff, dropout=0.1, max_len=512, activation='gelu'):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout, activation) for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        x = self.input_proj(src)
        x = self.pos_enc(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x

class Decoder(nn.Module):
    def __init__(self, input_dim, d_model, n_layers, n_heads, d_ff, dropout=0.1, max_len=512, activation='gelu'):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout, activation) for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        x = self.input_proj(tgt)
        x = self.pos_enc(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return x

class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        d_model=256,
        n_layers=6,
        n_heads=8,
        d_ff=1024,
        dropout=0.1,
        max_len=512,
        activation='gelu'
    ):
        super().__init__()
        self.encoder = Encoder(input_dim, d_model, n_layers, n_heads, d_ff, dropout, max_len, activation)
        self.decoder = Decoder(output_dim, d_model, n_layers, n_heads, d_ff, dropout, max_len, activation)
        self.out_proj = nn.Linear(d_model, output_dim)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.tril(torch.ones(sz, sz)).bool()
        return mask  # [sz, sz]

    def forward(self, src, tgt):
        # src: [B, src_len, input_dim]
        # tgt: [B, tgt_len, output_dim]
        src_mask = None
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        memory = self.encoder(src, src_mask)
        out = self.decoder(tgt, memory, tgt_mask, None)
        out = self.out_proj(out)
        return out