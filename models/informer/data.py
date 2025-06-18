import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, label_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len

    def __getitem__(self, idx):
        seq_x = self.data[idx:idx+self.seq_len]
        seq_y = self.data[idx+self.seq_len:idx+self.seq_len+self.pred_len]
        return torch.tensor(seq_x, dtype=torch.float), torch.tensor(seq_y, dtype=torch.float)