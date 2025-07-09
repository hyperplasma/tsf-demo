import os
import sys
import importlib
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
from datetime import datetime
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from common.utils import ensure_dir, count_parameters
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
    mse = mean_squared_error(trues.flatten(), preds.flatten())
    mae = mean_absolute_error(trues.flatten(), preds.flatten())
    r2 = r2_score(trues.flatten(), preds.flatten())
    return mse, mae, r2, preds, trues

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

    # Load model
    model = ModelClass(**cfg).to(cfg['device'])
    num_params = count_parameters(model)
    print(f"Model: {model_name}")
    print(f"Number of parameters: {num_params}")
    
    best_ckpt = os.path.join(output_dir, f'best_{dataset_name}.pth')
    if not os.path.exists(best_ckpt):
        best_ckpt = os.path.join(output_dir, f'checkpoint_{dataset_name}.pth')
    if not os.path.exists(best_ckpt):
        raise FileNotFoundError(f"Best checkpoint not found: {best_ckpt}")
    checkpoint = torch.load(best_ckpt, map_location=cfg['device'])
    model.load_state_dict(checkpoint['state_dict'])
    scaler = checkpoint['scaler']

    # Load test data (already normalized)
    test_data = load_test_data(data_path, scaler=scaler)
    in_chans = test_data.shape[1]
    cfg['in_chans'] = in_chans

    test_set = TimeSeriesDataset(test_data, cfg['input_length'], cfg['output_length'])
    test_loader = DataLoader(test_set, batch_size=cfg['batch_size'], shuffle=False)

    # Evaluate
    mse, mae, r2, preds, trues = evaluate(model, test_loader, cfg['device'])
    print(f"Test MSE: {mse:.4f} | Test MAE: {mae:.4f} | Test R2: {r2:.4f}")

    # ---- Inverse transform (反归一化) ----
    # preds, trues shape: (samples, output_length, channels) or (samples, output_length, ...)
    # Need to reshape to 2D for scaler.inverse_transform
    shape = preds.shape
    preds_2d = preds.reshape(-1, shape[-1])
    trues_2d = trues.reshape(-1, shape[-1])
    preds_inv = scaler.inverse_transform(preds_2d).reshape(shape)
    trues_inv = scaler.inverse_transform(trues_2d).reshape(shape)

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
        f.write("Test MSE,Test MAE,Test R2\n")
        f.write(f"{mse:.4f},{mae:.4f},{r2:.4f}\n")

    # Save inverse transformed predictions and ground truth
    np.save(os.path.join(output_dir, f'preds_{dataset_name}_inv.npy'), preds_inv)
    np.save(os.path.join(output_dir, f'trues_{dataset_name}_inv.npy'), trues_inv)
    print(f"Test results and predictions saved in: {output_dir}")

if __name__ == '__main__':
    main(model_name="PatchTST", small=True)