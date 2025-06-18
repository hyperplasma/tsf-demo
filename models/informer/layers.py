import torch
import torch.nn as nn
import torch.nn.functional as F

class ProbSparseSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.size()
        q = self.q_linear(x).view(B, L, self.n_heads, self.d_k).transpose(1,2)
        k = self.k_linear(x).view(B, L, self.n_heads, self.d_k).transpose(1,2)
        v = self.v_linear(x).view(B, L, self.n_heads, self.d_k).transpose(1,2)
        # 这里只做普通Attention，ProbSparse可后续补充
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1,2).contiguous().view(B, L, D)
        return self.out(out)