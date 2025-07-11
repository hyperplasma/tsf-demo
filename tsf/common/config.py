import os

common_config = {
    'dataset': "weather",   # 数据集（位于`dataset/`）
    'target_col': "T (degC)",   # 单预测变量
    'device': "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
    'learning_rate': 1e-4,     # 学习率
    'batch_size': 32,          # 批大小
    'epochs': 100,             # 最大训练轮数
    'early_stop_patience': 12, # 早停容忍轮数
    'output_dir': "outputs",        # 输出目录（各模型自行指定子目录）
    'log_interval': 10,        # 日志打印间隔
    'seed': 2023               # 随机种子
}

def get_config(**kwargs):
    """
    获取通用参数字典
    """
    cfg = common_config.copy()
    cfg.update(kwargs)
    return cfg