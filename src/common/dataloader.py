import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, data, input_length, output_length, target_col_idx):
        self.data = data  # shape: [N, C]
        self.input_length = input_length
        self.output_length = output_length
        self.target_col_idx = target_col_idx
        self.length = len(data) - input_length - output_length + 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.input_length, :]  # [input_length, in_chans]
        y = self.data[idx+self.input_length:idx+self.input_length+self.output_length, self.target_col_idx]  # [output_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def split_data(data, input_length, output_length):
    n = len(data)
    n_train = int(n * 0.7)
    n_val = int(n * 0.1)
    n_test = n - n_train - n_val

    train_data = data[:n_train]
    val_data = data[n_train - input_length : n_train + n_val + output_length - 1]
    test_data = data[n_train + n_val - input_length :]

    return train_data, val_data, test_data

def load_data(data_path, input_length=336, output_length=96, scaler=None, target_col='T (degC)', split='all'):
    df = pd.read_csv(data_path)
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
    feature_cols = list(df.columns)
    target_col_idx = feature_cols.index(target_col)
    data = df.values.astype(np.float32)
    if scaler is None:
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)
    else:
        data = scaler.transform(data)
    train_data, val_data, test_data = split_data(data, input_length, output_length)
    train_set = TimeSeriesDataset(train_data, input_length, output_length, target_col_idx)
    val_set = TimeSeriesDataset(val_data, input_length, output_length, target_col_idx)
    test_set = TimeSeriesDataset(test_data, input_length, output_length, target_col_idx)
    
    if split == 'all':
        return train_set, val_set, test_set, scaler, target_col_idx
    else:
        split_keys = [s.strip() for s in split.lower().split(',')]
        sets = {'train': None, 'val': None, 'test': None}  # type: dict[str, TimeSeriesDataset | None]
        for key in split_keys:
            if key == 'train':
                sets['train'] = train_set
            elif key == 'val':
                sets['val'] = val_set
            elif key == 'test':
                sets['test'] = test_set
        return sets['train'], sets['val'], sets['test'], scaler, target_col_idx