import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

def load_and_split_data(csv_path, seq_len=36, pred_len=12, val_ratio=0.1, test_ratio=0.1):
    df = pd.read_csv(csv_path)
    values = df['IPG2211A2N'].values.astype(np.float32)
    total_len = len(values)
    test_size = int(total_len * test_ratio)
    val_size = int(total_len * val_ratio)
    train_size = total_len - val_size - test_size

    train = values[:train_size]
    val = values[train_size:train_size+val_size]
    test = values[train_size+val_size:]

    return train, val, test

class SeqDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+self.seq_len:idx+self.seq_len+self.pred_len]
        return torch.tensor(x, dtype=torch.float32).unsqueeze(-1), torch.tensor(y, dtype=torch.float32).unsqueeze(-1)