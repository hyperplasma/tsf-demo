import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from .model import Informer
from .data import TimeSeriesDataset

def train_model(train_data, val_data, enc_in, d_model, n_heads, seq_len, pred_len, epochs=10, batch_size=32):
    train_set = TimeSeriesDataset(train_data, seq_len, 0, pred_len)
    val_set = TimeSeriesDataset(val_data, seq_len, 0, pred_len)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    model = Informer(enc_in, d_model, n_heads, seq_len, pred_len)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            out = model(x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    return model