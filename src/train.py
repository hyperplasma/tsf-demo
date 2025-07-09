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
from common.utils import ensure_dir, save_checkpoint, count_parameters, get_model_config_str
from common.dataloader import load_data
from common.config import get_config

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

def evaluate(model, loader, criterion, device, scaler, target_col_idx):
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
    # 合并预测和真实值（归一化尺度）
    preds_norm = np.concatenate(preds, axis=0)
    trues_norm = np.concatenate(trues, axis=0)
    # 计算归一化尺度指标
    mse_norm = mean_squared_error(trues_norm.flatten(), preds_norm.flatten())
    mae_norm = mean_absolute_error(trues_norm.flatten(), preds_norm.flatten())
    acc = r2_score(trues_norm.flatten(), preds_norm.flatten())
    # 反归一化到原始尺度（只对目标列）
    dummy_pred = np.zeros((preds_norm.size, scaler.n_features_in_))
    dummy_true = np.zeros((trues_norm.size, scaler.n_features_in_))
    dummy_pred[:, target_col_idx] = preds_norm.flatten()
    dummy_true[:, target_col_idx] = trues_norm.flatten()
    preds_raw = scaler.inverse_transform(dummy_pred)[:, target_col_idx].reshape(preds_norm.shape)
    trues_raw = scaler.inverse_transform(dummy_true)[:, target_col_idx].reshape(trues_norm.shape)
    # 计算原始尺度指标
    mse_raw = mean_squared_error(trues_raw.flatten(), preds_raw.flatten())
    mae_raw = mean_absolute_error(trues_raw.flatten(), preds_raw.flatten())
    return total_loss / len(loader.dataset), mse_norm, mae_norm, acc, mse_raw, mae_raw

def main(model_name="PatchTST", **kwargs):
    # Config
    cfg = get_config(**kwargs)
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    dataset_name = cfg['dataset']
    print(f"\nTrain dataset: {dataset_name}")
    data_path = os.path.join('dataset', f'{dataset_name}.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    # 动态加载模型类
    model_module = importlib.import_module(f"models.{model_name}")
    ModelClass = getattr(model_module, model_name)
    model = ModelClass(**cfg).to(cfg['device'])
    num_params = count_parameters(model)
    print(f"Model: {model_name}")
    print(f"Number of parameters: {num_params}")

    # 用模型成员变量传递数据参数
    input_length = getattr(model, 'input_length', 336)
    output_length = getattr(model, 'output_length', 96)

    # Data loading
    train_set, val_set, scaler, target_col_idx = load_data(
        data_path,
        input_length=input_length,
        output_length=output_length,
        target_col=cfg['target_col'],
        split='train,val'
    )
    in_chans = train_set.data.shape[1]
    cfg['in_chans'] = in_chans

    train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg['batch_size'], shuffle=False)

    # 如果模型支持in_chans动态调整，可重设
    if hasattr(model, 'in_chans'):
        model.in_chans = in_chans

    # Loss, optimizer, scheduler
    criterion = torch.nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=cfg['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5, verbose=True)

    # Logging and checkpoint paths
    output_dir = os.path.join(cfg['output_dir'], model_name, dataset_name)
    ensure_dir(output_dir)
    log_path = os.path.join(output_dir, f'train_log_{dataset_name}.txt')

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
        val_loss, val_mse_norm, val_mae_norm, val_acc, val_mse_raw, val_mae_raw = evaluate(model, val_loader, criterion, cfg['device'], scaler, target_col_idx)

        scheduler.step(val_loss)

        # Write model and dataset info to the log file header (if not exist)
        if not os.path.exists(log_path):
            with open(log_path, 'w') as f:
                current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                f.write(f"Training started at {current_time}\n")
                # Model information
                model_config_str = get_model_config_str(model, cfg)
                f.write(f"Model: {model_name}\n")
                f.write(f"Number of parameters: {num_params}\n")
                f.write(f"Model config: {model_config_str}\n\n")
                # Dataset information
                f.write(f"Dataset: {dataset_name}\n")
                f.write(f"Input channels: {in_chans}\n")
                f.write(f"Train samples: {len(train_set)}, Train batches: {len(train_loader)}\n")
                f.write(f"Val samples: {len(val_set)}, Val batches: {len(val_loader)}\n\n")
                f.write("Epoch,Train Loss,Train R2,Val Loss,Val R2,Val MSE (norm),Val MAE (norm),Val MSE (raw),Val MAE (raw)\n")

        with open(log_path, 'a') as f:
            f.write(f"{epoch},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f},{val_mse_norm:.4f},{val_mae_norm:.4f},{val_mse_raw:.4f},{val_mae_raw:.4f}\n")

        print(f"Train Loss: {train_loss:.4f} | Train R2: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val R2: {val_acc:.4f}")
        print(f"Val MSE (norm): {val_mse_norm:.4f} | Val MAE (norm): {val_mae_norm:.4f}")
        print(f"Val MSE (raw): {val_mse_raw:.4f} | Val MAE (raw): {val_mae_raw:.4f}")

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
                         'best_epoch': best_epoch,
                         'scaler': scaler},
                        is_best,
                        output_dir,
                        filename=f'checkpoint_{dataset_name}.pth',
                        best_filename=f'best_{dataset_name}.pth')

        # Early stopping
        if epoch - best_epoch > cfg.get('early_stop_patience', 12):
            print("Early stopping!")
            break

    print(f"\nTraining finished. Best val loss: {best_loss:.4f} (epoch {best_epoch})")
    print(f"All logs and weights are saved in: {output_dir}")

if __name__ == '__main__':
    main(model_name="PatchTST")