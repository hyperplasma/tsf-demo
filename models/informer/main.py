from utils import load_csv_data
from train import train_model

if __name__ == "__main__":
    data = load_csv_data('your_data.csv')  # 替换为你的数据路径
    train_data, val_data = data[:800], data[800:]
    model = train_model(
        train_data, val_data,
        enc_in=data.shape[1],
        d_model=64,
        n_heads=4,
        seq_len=96,
        pred_len=24,
        epochs=5
    )