import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(
        self,
        input_length=336,
        output_length=96,
        d_model=64,
        n_layers=2,
        dropout=0.1,
        normalize=True
    ):
        super().__init__()
        self.input_length = input_length
        self.output_length = output_length
        self.d_model = d_model
        self.n_layers = n_layers
        self.dropout = dropout
        self.normalize = normalize
        self.gru = None  # lazy init
        self.proj = None  # lazy init

    def forward(self, x):
        # x: (batch, input_length, input_dim)
        if self.normalize:
            mean = x.mean(dim=1, keepdim=True)  # (batch, 1, input_dim)
            std = x.std(dim=1, keepdim=True) + 1e-6
            x = (x - mean) / std
        batch_size, seq_len, input_dim = x.shape
        if self.gru is None:
            self.gru = nn.GRU(
                input_size=input_dim,
                hidden_size=self.d_model,
                num_layers=self.n_layers,
                batch_first=True,
                dropout=self.dropout if self.n_layers > 1 else 0.0
            ).to(x.device)
        if self.proj is None:
            self.proj = nn.Linear(self.d_model, input_dim).to(x.device)
        out, _ = self.gru(x)
        last_hidden = out[:, -1, :]
        repeated = last_hidden.unsqueeze(1).repeat(1, self.output_length, 1)
        pred = self.proj(repeated)
        if self.normalize:
            pred = pred * std[:, 0, :] + mean[:, 0, :]
        return pred

if __name__ == "__main__":
    batch = 4
    input_length = 36
    output_length = 24
    input_dim = 8
    model = GRUModel(input_length, output_length, d_model=32, n_layers=2, dropout=0.1, normalize=True)
    x = torch.randn(batch, input_length, input_dim)
    y = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape) 