import os

# Transformer config
config = {
    # 模型参数
    'input_length': 336,    # 输入长度
    'output_length': 96,    # 输出长度
    'd_model': 128,         # 模型维度
    'n_heads': 16,          # 注意力头数
    'e_layers': 3,          # 编码层数
    'd_ff': 256,            # 前馈层维度
    'dropout': 0.2,         # 丢弃率
    'act': 'gelu',          # 激活函数
    'attn_dropout': 0.0,    # 注意力丢弃率
    # 数据集相关参数
    'dataset': "weather",        # 数据集名称
    'target_col': 'T (degC)',    # 目标列名称
    # 训练参数
    'device': "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
    'learning_rate': 1e-4,     # 学习率
    'batch_size': 32,            # 批大小
    'epochs': 100,               # 最大训练轮数
    'early_stop_patience': 12,   # 早停容忍轮数
    # 其他
    'output_dir': "outputs/Transformer",  # 输出目录
    'log_interval': 10,          # 日志打印间隔
    'seed': 2023                # 随机种子
}

def get_config(**kwargs):
    """
    获取Transformer配置，支持自定义覆盖
    """
    cfg = config.copy()
    if kwargs:
        cfg.update(kwargs)
    return cfg 