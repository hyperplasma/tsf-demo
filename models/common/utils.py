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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)