import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class TimeSeriesDataset:
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
        return x, y

def load_data(data_path, input_length=336, output_length=96, val_ratio=0.1, scaler=None):
    df = pd.read_csv(data_path)
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
    feature_cols = list(df.columns)
    target_col = 'T (degC)'
    target_col_idx = feature_cols.index(target_col)
    data = df.values.astype(np.float32)
    if scaler is None:
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)
    else:
        data = scaler.transform(data)
    n = len(data)
    n_val = int(n * val_ratio)
    train_data = data[:-n_val]
    val_data = data[-(n_val+input_length+output_length-1):]
    train_set = TimeSeriesDataset(train_data, input_length, output_length, target_col_idx)
    val_set = TimeSeriesDataset(val_data, input_length, output_length, target_col_idx)
    return train_set, val_set, scaler, target_col_idx

def load_test_data(data_path, input_length=336, output_length=96, scaler=None, target_col_idx=None):
    df = pd.read_csv(data_path)
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
    data = df.values.astype(np.float32)
    if scaler is not None:
        data = scaler.transform(data)
    num_test = input_length + output_length
    test_data = data[-num_test:]
    test_set = TimeSeriesDataset(test_data, input_length, output_length, target_col_idx)
    return test_set