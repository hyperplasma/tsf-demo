import os

# PatchTST config
config = {
    # 模型参数
    'input_length': 336,
    'output_length': 96,
    'patch_len': 16,
    'stride': 8,
    'd_model': 128,
    'n_heads': 16,
    'e_layers': 3,
    'd_ff': 256,
    'dropout': 0.2,
    'act': 'gelu',
    'res_attention': False,
    'pre_norm': True,
    'attn_dropout': 0.0,
    'individual': False,  # 单变量预测时建议False
    'kernel_size': 25,
    'norm_type': 'BatchNorm',
    'pred_type': 'direct',
    # 数据集相关参数
    'dataset': "weather",        # 数据集名称（如ETTh1、ETTh2、ETTm1、ETTm2、electricity、exchange_rate、traffic、weather等）
    'target_col': 'T (degC)',    # 目标列名称（将数据集可视化后，选择一个变量作为目标）
    'in_chans': 21,  # 多变量输入，单变量输出（T (degC)）
    # 训练参数
    'device': "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",  # 设备
    'learning_rate': 0.0001,     # 学习率
    'batch_size': 32,            # 批大小
    'epochs': 100,               # 最大训练轮数
    'early_stop_patience': 12,   # 早停容忍轮数
    'output_dir': "outputs/PatchTST",  # 输出目录
    'log_interval': 10,          # 日志打印间隔（未用，可扩展）
    'seed': 2023                # 随机种子
}

# 小模型变动参数（适配Apple Silicon）
PATCHTST_SMALL_CONFIG_DIFF = {
    # 序列参数（减少内存压力）
    "input_length": 192,    # 缩短输入序列
    "output_length": 24,   # 缩短预测序列
    "kernel_size": 15,      # 趋势分解的滑动平均窗口大小
    
    # Patch参数（平衡计算量）
    "patch_len": 8,        # 保持与stride的比例
    "stride": 4,           # 保持与patch_len的比例
    
    # 模型结构（提升并行度）
    "d_model": 64,         # 提升特征维度
    "n_heads": 4,           # 增加注意力头利用M1的8核GPU
    "e_layers": 2,          # 增加编码层但保持轻量
    
    # 计算优化
    "d_ff": 128,           # 前馈层维度匹配d_model
    "dropout": 0.2,        # 适度增加防过拟合
    
    # 训练参数（利用Metal性能）
    "learning_rate": 0.001, # 保持稳定学习率
    "batch_size": 24,      # 提升批次利用统一内存
    "epochs": 30,          # 增加总轮次配合早停
    "early_stop_patience": 12 # 延长早停等待
}

def get_config(custom_cfg=None, **kwargs):
    """
    获取PatchTST配置，支持自定义覆盖
    """
    cfg = config.copy()
    
    if kwargs.get("small", False):
        # 如果是小模型，使用小模型的配置差异
        cfg.update(PATCHTST_SMALL_CONFIG_DIFF)
        
    if custom_cfg:
        cfg.update(custom_cfg)
        
    return cfg