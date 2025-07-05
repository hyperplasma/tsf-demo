import os

# PatchTST默认参数配置
PATCHTST_CONFIG = {
    "input_length": 336,         # 输入序列长度
    "output_length": 96,         # 预测序列长度
    "patch_len": 16,             # 每个patch的长度
    "stride": 8,                 # patch滑动步长
    "d_model": 128,              # Transformer特征维度
    "n_heads": 16,               # 多头注意力头数
    "e_layers": 3,               # Transformer编码器层数
    "d_ff": 256,                 # 前馈网络隐藏层维度
    "dropout": 0.2,              # dropout概率
    "act": "gelu",               # 激活函数
    "res_attention": False,      # 是否使用残差注意力（一般为False）
    "pre_norm": True,            # 是否使用pre-norm结构
    "attn_dropout": 0.0,         # 注意力dropout概率
    "dataset": "weather",        # 数据集名称（如ETTh1、ETTh2、ETTm1、ETTm2、electricity、exchange_rate、traffic、weather等）
    "individual": False,         # 是否为每个变量单独建头
    "kernel_size": 25,           # 趋势分解的滑动平均窗口大小
    "norm_type": "BatchNorm",    # 归一化类型（BatchNorm或LayerNorm）
    "pred_type": "direct",       # 预测类型（一般为direct）
    "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",  # 设备
    "learning_rate": 0.0001,     # 学习率
    "batch_size": 32,            # 批大小
    "epochs": 100,               # 最大训练轮数
    "early_stop_patience": 10,   # 早停容忍轮数
    "output_dir": "outputs/PatchTST",  # 输出目录
    "log_interval": 10,          # 日志打印间隔（未用，可扩展）
    "seed": 2023,                # 随机种子
}

def get_config(custom_cfg=None):
    """
    获取PatchTST配置，支持自定义覆盖
    """
    cfg = PATCHTST_CONFIG.copy()
    if custom_cfg:
        cfg.update(custom_cfg)
    return cfg