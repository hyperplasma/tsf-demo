import os
import sys
import importlib
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from common.utils import count_parameters, inverse_transform_predictions
from common.dataloader import load_test_data, TimeSeriesDataset

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
    print(f"\nTest dataset: {dataset_name}")
    data_path = os.path.join('dataset', f'{dataset_name}.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    output_dir = os.path.join(cfg['output_dir'], dataset_name)
    best_ckpt = os.path.join(output_dir, f'best_{dataset_name}.pth')
    if not os.path.exists(best_ckpt):
        best_ckpt = os.path.join(output_dir, f'checkpoint_{dataset_name}.pth')
    if not os.path.exists(best_ckpt):
        raise FileNotFoundError(f"Best checkpoint not found: {best_ckpt}")
    
    # Load test data (already normalized)
    checkpoint = torch.load(best_ckpt, map_location=cfg['device'])
    scaler = checkpoint['scaler']
    test_data = load_test_data(data_path, scaler=scaler)
    in_chans = test_data.shape[1]
    cfg['in_chans'] = in_chans
    
    # Load model
    model = ModelClass(**cfg).to(cfg['device'])
    num_params = count_parameters(model)
    print(f"Model: {model_name}")
    print(f"Number of parameters: {num_params}")
    model.load_state_dict(checkpoint['state_dict'])

    test_set = load_test_data(data_path, scaler=scaler, target_col_idx=2) # target_col_idx=2 for T (degC)
    test_loader = DataLoader(test_set, batch_size=cfg['batch_size'], shuffle=False)

    # Evaluate（归一化尺度）
    mse_norm, mae_norm, r2_norm, preds, trues = evaluate(model, test_loader, cfg['device'])
    print(f"Test MSE (norm): {mse_norm:.4f} | Test MAE (norm): {mae_norm:.4f} | Test R2 (norm): {r2_norm:.4f}")

    # 反归一化（使用utils中的通用函数）
    preds_inv, trues_inv = inverse_transform_predictions(preds, trues, scaler)

    # 计算原始尺度指标
    mse_raw = mean_squared_error(trues_inv.flatten(), preds_inv.flatten())
    mae_raw = mean_absolute_error(trues_inv.flatten(), preds_inv.flatten())
    r2_raw = r2_score(trues_inv.flatten(), preds_inv.flatten())

    # Save results to txt file
    log_path = os.path.join(output_dir, f'test_result_{dataset_name}.txt')
    with open(log_path, 'w') as f:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        f.write(f"Testing started at {current_time}\n")
        # Model config info
        model_config_str = ', '.join([f"{key}={value}" for key, value in cfg.items()])
        f.write(f"Model: {model_name}\n")
        f.write(f"Number of parameters: {num_params}\n")
        f.write(f"Model config: {model_config_str}\n\n")
        # Dataset info
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Input channels: {in_chans}\n")
        f.write(f"Test samples: {len(test_data)}, Test batches: {len(test_loader)}\n\n")
        f.write("Test MSE (norm),Test MAE (norm),Test R2 (norm),Test MSE (raw),Test MAE (raw),Test R2 (raw)\n")
        f.write(f"{mse_norm:.4f},{mae_norm:.4f},{r2_norm:.4f},{mse_raw:.4f},{mae_raw:.4f},{r2_raw:.4f}\n")

    # Save inverse transformed predictions and ground truth
    np.save(os.path.join(output_dir, f'preds_{dataset_name}_inv.npy'), preds_inv)
    np.save(os.path.join(output_dir, f'trues_{dataset_name}_inv.npy'), trues_inv)
    print(f"Test results and predictions saved in: {output_dir}")

if __name__ == '__main__':
    main(model_name="PatchTST")