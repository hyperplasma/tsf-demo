import torch
import pandas as pd
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    """
    Time Series Dataset for PyTorch, all located in `dataset/`.
    This dataset is designed to handle time series data for forecasting tasks.
    It takes a sequence of data and splits it into input and output segments.
    """
    def __init__(self, data, input_length, output_length):
        self.data = data
        self.input_length = input_length
        self.output_length = output_length

    def __len__(self):
        return len(self.data) - self.input_length - self.output_length + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.input_length]
        y = self.data[idx+self.input_length:idx+self.input_length+self.output_length]
        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)

def load_data(csv_path, val_ratio=0.1, test_ratio=0.1):
    df = pd.read_csv(csv_path)
    df = df.drop(columns=[col for col in df.columns if col.lower() in ['date', 'datetime', 'time', 'timestamp']])
    data = df.values
    num_samples = len(data)
    num_val = int(num_samples * val_ratio)
    num_test = int(num_samples * test_ratio)
    num_train = num_samples - num_val - num_test

    train_data = data[:num_train]
    val_data = data[num_train - num_val:num_train]
    return train_data, val_data

def load_test_data(csv_path, val_ratio=0.1, test_ratio=0.1):
    df = pd.read_csv(csv_path)
    df = df.drop(columns=[col for col in df.columns if col.lower() in ['date', 'datetime', 'time', 'timestamp']])
    data = df.values
    num_samples = len(data)
    num_val = int(num_samples * val_ratio)
    num_test = int(num_samples * test_ratio)
    num_train = num_samples - num_val - num_test

    test_data = data[num_train:num_train + num_test]
    return test_data