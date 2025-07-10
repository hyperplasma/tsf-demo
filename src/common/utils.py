import os
import inspect
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

def get_model_config_str(model, cfg=None):
    """
    返回模型全部参数及当前值，并合并cfg（如有），用于日志打印。
    """
    sig = inspect.signature(model.__init__)
    param_names = [k for k, v in sig.parameters.items()
                   if k != 'self' and v.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)]
    config = {}
    for name in param_names:
        config[name] = getattr(model, name, None)
    if cfg is not None:
        for k, v in cfg.items():
            if k not in config:
                config[k] = v
    return ', '.join([f'{k}={v}' for k, v in config.items()])