import os

# LSTM config（与PatchTST/Transformer参数风格一致，包含所有公平对比参数）
config = {
    # 模型参数
    'input_length': 336,    # 输入长度
    'output_length': 96,    # 输出长度
    'd_model': 128,         # LSTM隐藏层维度
    'n_layers': 2,          # LSTM层数
    'dropout': 0.2,         # 丢弃率
    'act': 'gelu',          # 激活函数
    'res_attention': False, # 残差注意力（保留接口）
    'pre_norm': True,       # 预归一化（保留接口）
    'attn_dropout': 0.0,    # 注意力丢弃率（保留接口）
    'in_chans': 21,         # 输入变量数
    'individual': False,    # 多变量预测
    'bidirectional': False, # 是否双向LSTM
    'kernel_size': 25,      # 卷积核大小（保留接口）
    'norm_type': 'BatchNorm', # 标准化类型
    'pred_type': 'direct',  # 预测类型（direct:最后hidden，multi:多步输出）
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
    'output_dir': "outputs/LSTM",  # 输出目录
    'log_interval': 10,          # 日志打印间隔
    'seed': 2023                # 随机种子
}

def get_config(**kwargs):
    """
    获取LSTM配置，支持自定义覆盖
    """
    cfg = config.copy()
    if kwargs:
        cfg.update(kwargs)
    return cfg
