import os
import sys
import importlib
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from common.utils import count_parameters, get_model_config_str
from common.dataloader import load_data
from common.config import get_config

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
    # 计算归一化尺度指标
    mse_norm = mean_squared_error(trues.flatten(), preds.flatten())
    mae_norm = mean_absolute_error(trues.flatten(), preds.flatten())
    r2_norm = r2_score(trues.flatten(), preds.flatten())
    return mse_norm, mae_norm, r2_norm, preds, trues

def main(model_name="PatchTST", **kwargs):
    # Config
    cfg = get_config(**kwargs)
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    
    dataset_name = cfg['dataset']
    print(f"\nTest dataset: {dataset_name}")
    data_path = os.path.join('dataset', f'{dataset_name}.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    output_dir = os.path.join(cfg['output_dir'], model_name, dataset_name)
    best_ckpt = os.path.join(output_dir, f'best_{dataset_name}.pth')
    if not os.path.exists(best_ckpt):
        best_ckpt = os.path.join(output_dir, f'checkpoint_{dataset_name}.pth')
    if not os.path.exists(best_ckpt):
        raise FileNotFoundError(f"Best checkpoint not found: {best_ckpt}")

    # Load checkpoint and scaler
    checkpoint = torch.load(best_ckpt, map_location=cfg['device'])
    scaler = checkpoint['scaler']

    # 动态加载模型类
    model_module = importlib.import_module(f"models.{model_name}")
    ModelClass = getattr(model_module, model_name)
    model = ModelClass(**cfg, individual=True).to(cfg['device'])
    num_params = count_parameters(model)
    print(f"Model: {model_name}")
    print(f"Number of parameters: {num_params}")
    model.load_state_dict(checkpoint['state_dict'])

    # 用模型成员变量传递数据参数
    input_length = getattr(model, 'input_length', 336)
    output_length = getattr(model, 'output_length', 96)

    # Load test data using new load_data interface
    _, _, test_set, _, target_col_idx = load_data(
        data_path,
        scaler=scaler,
        input_length=input_length,
        output_length=output_length,
        target_col=cfg['target_col'],
        split='test'
    )
    in_chans = test_set.data.shape[1] if test_set is not None else 0
    cfg['in_chans'] = in_chans
    # 如果模型支持in_chans动态调整，可重设
    if hasattr(model, 'in_chans'):
        model.in_chans = in_chans

    test_loader = DataLoader(test_set, batch_size=cfg['batch_size'], shuffle=False) if test_set is not None else []

    # Evaluate（归一化尺度）
    mse_norm, mae_norm, r2_norm, preds, trues = evaluate(model, test_loader, cfg['device'])
    print(f"Test MSE (norm): {mse_norm:.4f} | Test MAE (norm): {mae_norm:.4f} | Test R2 (norm): {r2_norm:.4f}")

    # 反归一化（所有变量）
    preds_inv = scaler.inverse_transform(preds.reshape(-1, preds.shape[-1])).reshape(preds.shape)
    trues_inv = scaler.inverse_transform(trues.reshape(-1, trues.shape[-1])).reshape(trues.shape)

    # 计算原始尺度指标
    mse_raw = mean_squared_error(trues_inv.flatten(), preds_inv.flatten())
    mae_raw = mean_absolute_error(trues_inv.flatten(), preds_inv.flatten())
    r2_raw = r2_score(trues_inv.flatten(), preds_inv.flatten())
    print(f"Test MSE (raw): {mse_raw:.4f} | Test MAE (raw): {mae_raw:.4f} | Test R2 (raw): {r2_raw:.4f}")

    # Save results to txt file
    log_path = os.path.join(output_dir, f'test_result_{dataset_name}.txt')
    with open(log_path, 'w') as f:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        f.write(f"Testing started at {current_time}\n")
        # Model config info
        model_config_str = get_model_config_str(model, cfg)
        f.write(f"Model: {model_name}\n")
        f.write(f"Number of parameters: {num_params}\n")
        f.write(f"Model config: {model_config_str}\n\n")
        # Dataset info
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Input channels: {in_chans}\n")
        f.write(f"Test samples: {len(test_set) if test_set is not None else 0}, Test batches: {len(test_loader) if test_loader else 0}\n\n")
        f.write("Test MSE (norm),Test MAE (norm),Test R2 (norm),Test MSE (raw),Test MAE (raw),Test R2 (raw)\n")
        f.write(f"{mse_norm:.4f},{mae_norm:.4f},{r2_norm:.4f},{mse_raw:.4f},{mae_raw:.4f},{r2_raw:.4f}\n")

    # Save inverse transformed predictions and ground truth
    np.save(os.path.join(output_dir, f'preds_{dataset_name}_inv.npy'), preds_inv)
    np.save(os.path.join(output_dir, f'trues_{dataset_name}_inv.npy'), trues_inv)
    print(f"Test results and predictions saved in: {output_dir}")

if __name__ == '__main__':
    main(model_name="PatchTST")