import os

# PatchTST config
config = {
    # 模型参数
    'input_length': 336,    # 输入长度
    'output_length': 96,    # 输出长度
    'patch_len': 16,        # Patch长度
    'stride': 8,            # 步长
    'd_model': 128,         # 模型维度
    'n_heads': 16,          # 注意力头数
    'e_layers': 3,          # 编码层数
    'd_ff': 256,            # 前馈层维度
    'dropout': 0.2,         # 丢弃率
    'act': 'gelu',          # 激活函数
    'res_attention': False, # 残差注意力
    'pre_norm': True,       # 预归一化
    'attn_dropout': 0.0,    # 注意力丢弃率
    'individual': False,    # True：每个变量一个预测头，False：所有变量共享一个预测头
    'kernel_size': 25,      # 卷积核大小
    'norm_type': 'BatchNorm', # 归一化类型（BatchNorm、LayerNorm、GroupNorm、InstanceNorm）
    'pred_type': 'direct',  # 预测类型（direct：直接预测，trend：趋势预测）
    # 数据集相关参数
    'dataset': "weather",        # 数据集名称（如ETTh1、ETTh2、ETTm1、ETTm2、electricity、exchange_rate、traffic、weather等）
    'target_col': 'T (degC)',    # 目标列名称（将数据集可视化后，选择一个变量作为目标）
    # 训练参数
    'device': "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
    'learning_rate': 1e-4,     # 学习率
    'batch_size': 32,            # 批大小
    'epochs': 100,               # 最大训练轮数
    'early_stop_patience': 12,   # 早停容忍轮数
    # 其他
    'output_dir': "outputs/PatchTST",  # 输出目录
    'log_interval': 10,          # 日志打印间隔（未用，可扩展）
    'seed': 2023                # 随机种子
}

def get_config(**kwargs):
    """
    获取PatchTST配置，支持自定义覆盖
    """
    cfg = config.copy()
    
    if kwargs:
        cfg.update(kwargs)
        
    return cfg