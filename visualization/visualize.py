import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime  # 新增：用于解析真实时间戳

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from models.common.utils import ensure_dir  # 路径确保工具函数

def visualize_predictions(dataset_name="weather", output_dir="outputs/PatchTST", 
                          num_features=3, num_samples=5):  # 新增参数控制展示数量
    """
    Visualize time series predictions vs ground truth with real timestamps and error annotations
    
    Args:
        dataset_name: Dataset name (consistent with predict.py)
        output_dir: Output directory (consistent with training/prediction)
        num_features: Number of features to visualize (default: 3)
        num_samples: Number of samples to visualize per feature (default: 5)
    """
    # 配置路径
    data_dir = os.path.join(output_dir, dataset_name)
    pred_path = os.path.join(data_dir, f'preds_{dataset_name}_inv.npy')
    true_path = os.path.join(data_dir, f'trues_{dataset_name}_inv.npy')
    vis_dir = os.path.join(data_dir, 'visualizations')
    ensure_dir(vis_dir)  # 确保可视化目录存在

    # 加载数据（形状：[num_samples, output_length, num_features]）
    preds = np.load(pred_path)
    trues = np.load(true_path)
    num_total_samples, output_length, num_total_features = preds.shape

    # 限制展示数量（避免图表过多）
    num_features = min(num_features, num_total_features)
    num_samples = min(num_samples, num_total_samples)

    # 加载真实时间戳（假设原始数据包含时间列，需根据实际数据调整）
    # 示例：假设原始数据第一列为时间戳（格式如"2023-01-01 00:00:00"）
    # 需根据你的数据集实际路径和格式修改！
    raw_data_path = os.path.join('dataset', f'{dataset_name}.csv')
    raw_data = np.genfromtxt(raw_data_path, delimiter=',', dtype=str, skip_header=1)
    timestamps = raw_data[:, 0]  # 假设第一列是时间戳

    # 绘制每个特征的对比图
    for feature_idx in range(num_features):
        plt.figure(figsize=(18, 8))  # 增大图表尺寸
        
        # 遍历选中的样本
        for sample_idx in range(num_samples):
            # 计算该样本对应的真实时间范围（需与预测的output_length对齐）
            # 示例：假设每个样本预测的是原始数据中第 [start:start+output_length] 时间点
            # 需根据你的数据划分逻辑调整！
            start_idx = sample_idx * output_length
            end_idx = start_idx + output_length
            sample_timestamps = timestamps[start_idx:end_idx]
            
            # 转换为datetime对象（用于更友好的时间轴显示）
            sample_times = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") for ts in sample_timestamps]
            
            # 绘制真实值与预测值
            plt.plot(sample_times, trues[sample_idx, :, feature_idx], 
                     label='Ground Truth' if sample_idx == 0 else None, 
                     color='blue', linewidth=2, alpha=0.8)
            plt.plot(sample_times, preds[sample_idx, :, feature_idx], 
                     label='Prediction' if sample_idx == 0 else None, 
                     color='orange', linestyle='--', linewidth=2, alpha=0.8)
            
            # 添加误差标注（顶部显示最大绝对误差）
            errors = np.abs(trues[sample_idx, :, feature_idx] - preds[sample_idx, :, feature_idx])
            max_error = np.max(errors)
            plt.text(sample_times[-1],  # 标注在样本末尾位置
                     max(trues[sample_idx, :, feature_idx].max(), preds[sample_idx, :, feature_idx].max()),
                     f'Sample {sample_idx+1}\nMax Error: {max_error:.2f}',
                     ha='right', va='bottom', color='red', fontsize=10)

        # 图表美化
        plt.title(f'Feature {feature_idx+1} Prediction vs Ground Truth ({dataset_name} Dataset)', fontsize=14)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Original Value', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xticks(rotation=45)  # 时间轴标签旋转避免重叠
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(os.path.join(vis_dir, f'feature_{feature_idx+1}_comparison.png'), dpi=300)  # 提高分辨率
        plt.close()

    print(f"Visualization plots saved to: {vis_dir}")

if __name__ == '__main__':
    # 可调整参数控制展示的特征数和样本数（如展示5个特征、10个样本）
    visualize_predictions(dataset_name="weather", output_dir="outputs/PatchTST", num_features=5, num_samples=10)
