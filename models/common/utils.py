import os
import numpy as np
import torch

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth', best_filename='best.pth'):
    ensure_dir(checkpoint_dir)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_path = os.path.join(checkpoint_dir, best_filename)
        torch.save(state, best_path)

def save_log(log_path, header, rows):
    """
    log_path: 日志文件路径
    header: list, csv表头
    rows: list of list, 每一行数据
    """
    import csv
    ensure_dir(os.path.dirname(log_path))
    write_header = not os.path.exists(log_path)
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerows(rows)

def get_positional_encoding(seq_len, d_model):
    """
    Transformer位置编码（非学习参数）
    """
    pe = np.zeros((seq_len, d_model))
    position = np.arange(0, seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return torch.tensor(pe, dtype=torch.float)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def inverse_transform_predictions(preds, trues, scaler):
    """
    对预测值和真实值进行反归一化（从归一化尺度恢复原始数据尺度）
    Args:
        preds: np.ndarray, 预测值（形状：[N, output_length, channels] 或其他任意维度，最后一维为特征）
        trues: np.ndarray, 真实值（形状需与preds一致）
        scaler: sklearn.preprocessing.Scaler, 训练时使用的归一化器
    Returns:
        preds_inv: np.ndarray, 反归一化后的预测值（与preds形状一致）
        trues_inv: np.ndarray, 反归一化后的真实值（与trues形状一致）
    """
    # 保留原始形状（用于恢复）
    shape = preds.shape
    # 展平为二维（最后一维为特征数）
    preds_2d = preds.reshape(-1, shape[-1])
    trues_2d = trues.reshape(-1, shape[-1])
    # 反归一化
    preds_inv = scaler.inverse_transform(preds_2d).reshape(shape)
    trues_inv = scaler.inverse_transform(trues_2d).reshape(shape)
    return preds_inv, trues_inv