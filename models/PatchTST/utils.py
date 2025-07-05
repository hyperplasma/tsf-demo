import os
import numpy as np
import torch

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth'):
    ensure_dir(checkpoint_dir)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best.pth')
        torch.save(state, best_path)

def load_checkpoint(model, checkpoint_path, device='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    return model

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
    标准Transformer位置编码
    """
    pe = np.zeros((seq_len, d_model))
    position = np.arange(0, seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return torch.tensor(pe, dtype=torch.float)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)