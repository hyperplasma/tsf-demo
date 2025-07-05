import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from models.PatchTST.model import PatchTST
from models.PatchTST.config import get_config
from models.PatchTST.utils import ensure_dir, load_checkpoint, save_log

# ------------------ Dataset Definition ------------------
class TimeSeriesDataset(Dataset):
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

def load_test_data(csv_path, val_ratio=0.1, test_ratio=0.1):
    df = pd.read_csv(csv_path)
    df = df.drop(columns=[col for col in df.columns if col.lower() in ['date', 'datetime', 'time', 'timestamp']])
    data = df.values
    num_samples = len(data)
    num_val = int(num_samples * val_ratio)
    num_test = int(num_samples * test_ratio)
    num_train = num_samples - num_val - num_test

    test_data = data[num_train:num_train+num_test]
    return test_data

def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Testing", leave=False):
            x, y = x.to(device), y.to(device)
            output = model(x)
            preds.append(output.cpu().numpy())
            trues.append(y.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    mse = mean_squared_error(trues.flatten(), preds.flatten())
    mae = mean_absolute_error(trues.flatten(), preds.flatten())
    r2 = r2_score(trues.flatten(), preds.flatten())
    return mse, mae, r2, preds, trues

def main():
    cfg = get_config()
    dataset_name = cfg['dataset']
    print(f"\n========== Testing on dataset: {dataset_name} ==========")
    data_path = os.path.join('dataset', f'{dataset_name}.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    # Load test data
    test_data = load_test_data(data_path)
    in_chans = test_data.shape[1]
    cfg['in_chans'] = in_chans

    test_set = TimeSeriesDataset(test_data, cfg['input_length'], cfg['output_length'])
    test_loader = DataLoader(test_set, batch_size=cfg['batch_size'], shuffle=False)

    # Load model
    model = PatchTST(**cfg).to(cfg['device'])
    output_dir = os.path.join(cfg['output_dir'], dataset_name)
    best_ckpt = os.path.join(output_dir, f'best.pth')
    if not os.path.exists(best_ckpt):
        best_ckpt = os.path.join(output_dir, f'checkpoint_{dataset_name}.pth')
    if not os.path.exists(best_ckpt):
        raise FileNotFoundError(f"Best checkpoint not found: {best_ckpt}")
    model = load_checkpoint(model, best_ckpt, device=cfg['device'])

    # Evaluate
    mse, mae, r2, preds, trues = evaluate(model, test_loader, cfg['device'])
    print(f"Test MSE: {mse:.4f} | Test MAE: {mae:.4f} | Test R2: {r2:.4f}")

    # Save results
    log_path = os.path.join(output_dir, f'test_result_{dataset_name}.csv')
    header = ['mse', 'mae', 'r2']
    save_log(log_path, header, [[mse, mae, r2]])

    # Optionally save predictions and ground truth for further analysis
    np.save(os.path.join(output_dir, f'preds_{dataset_name}.npy'), preds)
    np.save(os.path.join(output_dir, f'trues_{dataset_name}.npy'), trues)
    print(f"Test results and predictions saved in: {output_dir}")

if __name__ == '__main__':
    main()