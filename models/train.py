import os
import sys
import importlib
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from common.utils import ensure_dir, save_checkpoint, count_parameters
from common.dataloader import load_data, TimeSeriesDataset

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
def main(model_name="PatchTST", **kwargs):
    # Dynamically import model config
    try:
        model_module = importlib.import_module(f'models.{model_name}.model')
        config_module = importlib.import_module(f'models.{model_name}.config')
        ModelClass = getattr(model_module, model_name)
        get_config = getattr(config_module, 'get_config')
    except ImportError as e:
        raise ImportError(f"Failed to import model {model_name}. Ensure the model and config files exist.") from e
    except AttributeError as e:
        raise AttributeError(f"Model {model_name} does not have the required class or config function.") from e
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while importing model {model_name}.") from e
    
    # Config
    cfg = get_config(**kwargs)
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    dataset_name = cfg['dataset']
    print(f"\nTrain dataset: {dataset_name}")
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
    model = ModelClass(**cfg).to(cfg['device'])
    num_params = count_parameters(model)
    print(f"Model: {model_name}")
    print(f"Number of parameters: {num_params}")

    # Loss, optimizer, scheduler
    criterion = torch.nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=cfg['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5, verbose=True)

    # Logging and checkpoint paths
    output_dir = os.path.join(cfg['output_dir'], dataset_name)
    ensure_dir(output_dir)
    log_path = os.path.join(output_dir, f'train_log_weather.txt')
    
    # Write model and dataset info to log file (if not exist, then create its header)
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            f.write(f"Training started at {current_time}\n")
            # Model information
            model_config_str = ', '.join([f"{key}={value}" for key, value in cfg.items()])
            f.write(f"Model: {model_name}\n")
            f.write(f"Number of parameters: {num_params}\n")
            f.write(f"Model config: {model_config_str}\n\n")
            # Dataset information
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Input channels: {in_chans}\n")
            f.write(f"Train samples: {len(train_data)}, Train batches: {len(train_loader)}\n")
            f.write(f"Val samples: {len(val_data)}, Val batches: {len(val_loader)}\n\n")
            f.write("Epoch,Train Loss,Train R2,Val Loss,Val R2,Val MSE,Val MAE\n")

    start_epoch = 1
    best_loss = float('inf')
    best_epoch = 0
    
    # Resume from checkpoint if exists
    checkpoint_path = os.path.join(output_dir, f'checkpoint_{dataset_name}.pth')
    if os.path.exists(checkpoint_path):
        print(f"Checkpoint detected, resuming training from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=cfg['device'])
        model.load_state_dict(checkpoint['state_dict'])
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        if 'best_loss' in checkpoint:
            best_loss = checkpoint['best_loss']
        if 'best_epoch' in checkpoint:
            best_epoch = checkpoint['best_epoch']
        print(f"Resumed at epoch {start_epoch-1}, best_loss={best_loss:.4f}")

    # Training loop
    for epoch in range(start_epoch, cfg['epochs'] + 1):
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
        # Save checkpoint with optimizer and epoch info for resume
        save_checkpoint({'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'best_loss': best_loss,
                         'best_epoch': best_epoch},
                        is_best,
                        output_dir,
                        filename=f'checkpoint_{dataset_name}.pth',
                        best_filename=f'best_{dataset_name}.pth')

        # Early stopping
        if epoch - best_epoch > cfg['early_stop_patience']:
            print("Early stopping!")
            break

    print(f"\nTraining finished. Best val loss: {best_loss:.4f} (epoch {best_epoch})")
    print(f"All logs and weights are saved in: {output_dir}")

if __name__ == '__main__':
    main(model_name="PatchTST", small=True)
