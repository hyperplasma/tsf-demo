import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from models.PatchTST.model import PatchTST
from models.PatchTST.config import get_config
from models.PatchTST.utils import ensure_dir, save_checkpoint, count_parameters

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

# ------------------ Train/Validation Functions ------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    preds, trues = [], []
    for x, y in tqdm(loader, desc="Training", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        preds.append(output.detach().cpu().numpy())
        trues.append(y.detach().cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    acc = r2_score(trues.flatten(), preds.flatten())
    return total_loss / len(loader.dataset), acc

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    preds, trues = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validating", leave=False):
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            total_loss += loss.item() * x.size(0)
            preds.append(output.cpu().numpy())
            trues.append(y.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    mse = mean_squared_error(trues.flatten(), preds.flatten())
    mae = mean_absolute_error(trues.flatten(), preds.flatten())
    acc = r2_score(trues.flatten(), preds.flatten())
    return total_loss / len(loader.dataset), mse, mae, acc

# ------------------ Main Training Loop ------------------
def main():
    # Config
    cfg = get_config()
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    dataset_name = cfg['dataset']
    print(f"\n========== Dataset: {dataset_name} ==========")
    data_path = os.path.join('dataset', f'{dataset_name}.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    # Data loading
    train_data, val_data = load_data(data_path)
    in_chans = train_data.shape[1]
    cfg['in_chans'] = in_chans

    train_set = TimeSeriesDataset(train_data, cfg['input_length'], cfg['output_length'])
    val_set = TimeSeriesDataset(val_data, cfg['input_length'], cfg['output_length'])

    train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg['batch_size'], shuffle=False)

    # Model
    model = PatchTST(**cfg).to(cfg['device'])
    num_params = count_parameters(model)
    print(f"Number of parameters: {num_params}")

    # Loss, optimizer, scheduler
    criterion = torch.nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=cfg['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5, verbose=True)

    # Logging and checkpoint paths
    output_dir = os.path.join(cfg['output_dir'], dataset_name)
    ensure_dir(output_dir)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f'train_log_weather_{current_time}.txt')

    # Write model and dataset info to log file
    with open(log_path, 'w') as f:
        # Model information
        model_config_str = ', '.join([f"{key}={value}" for key, value in cfg.items()])
        f.write(f"Model parameters: {num_params} parameters\n")
        f.write(f"Model config: {model_config_str}\n\n")
        # Dataset information
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Input channels: {in_chans}\n")
        f.write(f"Train samples: {len(train_data)}, Train batches: {len(train_loader)}\n")
        f.write(f"Val samples: {len(val_data)}, Val batches: {len(val_loader)}\n\n")
        f.write("Epoch,Train Loss,Train R2,Val Loss,Val R2,Val MSE,Val MAE\n")

    best_loss = float('inf')
    best_epoch = 0

    # Training loop
    for epoch in range(1, cfg['epochs'] + 1):
        print(f"\n--- Epoch {epoch} / {cfg['epochs']} ---")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, cfg['device'])
        val_loss, val_mse, val_mae, val_acc = evaluate(model, val_loader, criterion, cfg['device'])

        scheduler.step(val_loss)

        with open(log_path, 'a') as f:
            f.write(f"{epoch},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f},{val_mse:.4f},{val_mae:.4f}\n")

        print(f"Train Loss: {train_loss:.4f} | Train R2: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val R2: {val_acc:.4f} | Val MSE: {val_mse:.4f} | Val MAE: {val_mae:.4f}")

        # Save best model
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            best_epoch = epoch
            save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict()}, is_best, output_dir, filename=f'checkpoint_{dataset_name}.pth')

        # Early stopping
        if epoch - best_epoch > cfg['early_stop_patience']:
            print("Early stopping!")
            break

    print(f"\nTraining finished. Best val loss: {best_loss:.4f} (epoch {best_epoch})")
    print(f"All logs and weights are saved in: {output_dir}")

if __name__ == '__main__':
    main()