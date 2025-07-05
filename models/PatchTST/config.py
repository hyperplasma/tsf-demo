import os

# PatchTST官方默认参数配置
PATCHTST_CONFIG = {
    "input_length": 336,
    "output_length": 96,
    "patch_len": 16,
    "stride": 8,
    "d_model": 128,
    "n_heads": 16,
    "e_layers": 3,
    "d_ff": 256,
    "dropout": 0.2,
    "act": "gelu",
    "res_attention": False,
    "pre_norm": True,
    "attn_dropout": 0.0,
    "in_chans": 7,  # 根据数据集自动调整
    "individual": False,
    "kernel_size": 25,
    "norm_type": "BatchNorm",
    "pred_type": "direct",
    "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
    "learning_rate": 0.0001,
    "batch_size": 32,
    "epochs": 100,
    "early_stop_patience": 10,
    "output_dir": "outputs/PatchTST",
    "log_interval": 10,
    "seed": 2023,
}

def get_config(custom_cfg=None):
    """
    获取PatchTST配置，支持自定义覆盖
    """
    cfg = PATCHTST_CONFIG.copy()
    if custom_cfg:
        cfg.update(custom_cfg)
    return cfg