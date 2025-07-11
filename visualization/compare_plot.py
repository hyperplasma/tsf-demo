import os
import numpy as np
import matplotlib.pyplot as plt


def plot_predictions(trues, preds, save_dir, var_indices=None, sample_range=None):
    os.makedirs(save_dir, exist_ok=True)
    # 自动扩展为3维 (样本数, 步长, 变量数)
    if trues.ndim == 2:
        trues = trues[:, :, None]
        preds = preds[:, :, None]
    elif trues.ndim == 1:
        trues = trues[:, None, None]
        preds = preds[:, None, None]
    n_samples, out_len, n_vars = trues.shape
    if var_indices is None:
        var_indices = list(range(n_vars))
    else:
        var_indices = [v for v in var_indices if v < n_vars]
        if not var_indices:
            print(f"[Error] 变量索引超出范围，实际变量数为 {n_vars}")
            return
    # 只画前20个样本
    max_samples = min(20, trues.shape[0])
    trues = trues[:max_samples]
    preds = preds[:max_samples]
    for var in var_indices:
        plt.figure(figsize=(10, 4))
        for i in range(trues.shape[0]):
            plt.plot(trues[i, :, var], color='blue', alpha=0.7, label='True' if i == 0 else "")
            plt.plot(preds[i, :, var], color='red', alpha=0.7, label='Pred' if i == 0 else "")
        plt.title(f'Variable {var} Prediction vs True (First {max_samples} Samples)')
        plt.xlabel('Forecast Step')
        plt.ylabel('Value')
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'var{var}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    preds_path = 'outputs/PatchTST/weather/preds_weather_inv.npy'
    trues_path = 'outputs/PatchTST/weather/trues_weather_inv.npy'
    save_dir = 'outputs/PatchTST/weather/figures'
    var_indices = [0, 1]
    sample_range = [0, 20]

    preds = np.load(preds_path)
    trues = np.load(trues_path)
    if preds.shape != trues.shape:
        print(f"[Error] preds.shape {preds.shape} != trues.shape {trues.shape}")
    else:
        plot_predictions(trues, preds, save_dir, var_indices, sample_range) 