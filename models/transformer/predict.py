import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.data_process import load_and_split_data
from .model import TimeSeriesTransformer

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class StandardScaler:
    def fit(self, data):
        self.mean = np.mean(data)
        self.std = np.std(data)
    def transform(self, data):
        return (data - self.mean) / (self.std + 1e-8)
    def inverse_transform(self, data):
        return data * (self.std + 1e-8) + self.mean

def autoregressive_predict(model, x_input, pred_len, device):
    model.eval()
    preds = []
    decoder_input = torch.zeros((1, 1, 1), dtype=torch.float32).to(device)
    with torch.no_grad():
        memory = model.encoder(x_input)
        for _ in range(pred_len):
            out = model.decoder(decoder_input, memory)
            y_pred = model.out_proj(out[:, -1:, :])  # 取最后一个预测
            preds.append(y_pred.cpu().numpy().flatten()[0])
            decoder_input = torch.cat([decoder_input, y_pred], dim=1)
    return np.array(preds)

def predict_future(
    csv_path,
    model_path,
    seq_len=24,
    pred_len=12,
    d_model=64,
    n_heads=4,
    n_layers=2,
    d_ff=128,
    dropout=0.1,
    max_len=512,
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 读取全部数据
    df = pd.read_csv(csv_path)
    values = df['IPG2211A2N'].values.astype(np.float32)
    # 归一化（用训练集均值和方差）
    scaler = StandardScaler()
    scaler.fit(values[:-int(len(values)*0.2)])  # 假设最后20%为val+test
    values_norm = scaler.transform(values)
    # 取最后seq_len个点作为输入
    x_input = values_norm[-seq_len:]
    x_input = torch.tensor(x_input, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    # 加载模型
    model = TimeSeriesTransformer(
        input_dim=1,
        output_dim=1,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
        max_len=max_len
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    # 自回归预测
    pred_norm = autoregressive_predict(model, x_input, pred_len, device)
    pred = scaler.inverse_transform(pred_norm)
    print("未来预测值：", pred)
    return pred

if __name__ == "__main__":
    csv_path = "data/electric_production/Electric_Production.csv"
    model_path = "outputs/transformer/best_transformer.pth"
    seq_len = 24
    pred_len = 12
    pred = predict_future(
        csv_path=csv_path,
        model_path=model_path,
        seq_len=seq_len,
        pred_len=pred_len,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.1,
        max_len=512,
        device=None
    )
    df = pd.read_csv(csv_path)
    plt.figure()
    plt.plot(np.arange(seq_len), df['IPG2211A2N'].values[-seq_len:], label='历史')
    plt.plot(np.arange(seq_len, seq_len+pred_len), pred, label='预测')
    plt.legend()
    plt.title('未来预测')
    plt.savefig("outputs/transformer/future_pred.png")
    plt.show()