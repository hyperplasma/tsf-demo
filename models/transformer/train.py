import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from utils.data_process import load_and_split_data, SeqDataset
from .model import TimeSeriesTransformer

# 归一化与反归一化工具
class StandardScaler:
    def fit(self, data):
        self.mean = np.mean(data)
        self.std = np.std(data)
    def transform(self, data):
        return (data - self.mean) / (self.std + 1e-8)
    def inverse_transform(self, data):
        return data * (self.std + 1e-8) + self.mean

def evaluate(model, data_loader, criterion, device, scaler):
    model.eval()
    total_loss = 0
    preds = []
    trues = []
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            # teacher forcing: decoder_input为y的左移一位，首位补零
            decoder_input = torch.zeros_like(y)
            decoder_input[:, 1:, :] = y[:, :-1, :]
            out = model(x, decoder_input)
            loss = criterion(out, y)
            total_loss += loss.item()
            preds.append(out.cpu().numpy())
            trues.append(y.cpu().numpy())
    avg_loss = total_loss / len(data_loader)
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    # 反归一化
    preds_inv = scaler.inverse_transform(preds)
    trues_inv = scaler.inverse_transform(trues)
    rmse = np.sqrt(np.mean((preds_inv - trues_inv) ** 2))
    acc = 1 - rmse / (np.mean(np.abs(trues_inv)) + 1e-8)
    return avg_loss, acc

def train_val_test(
    csv_path,
    seq_len=24,
    pred_len=12,
    batch_size=32,
    epochs=50,
    lr=1e-3,
    val_ratio=0.1,
    test_ratio=0.1,
    d_model=64,
    n_heads=4,
    n_layers=2,
    d_ff=128,
    dropout=0.1,
    max_len=512,
    device=None,
    output_dir="outputs/transformer"
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, "best_transformer.pth")
    loss_curve_path = os.path.join(output_dir, "loss_curve.png")
    acc_curve_path = os.path.join(output_dir, "acc_curve.png")

    # 数据加载与划分
    train_data, val_data, test_data = load_and_split_data(csv_path, seq_len, pred_len, val_ratio, test_ratio)

    # 归一化
    scaler = StandardScaler()
    scaler.fit(train_data)
    train_data_norm = scaler.transform(train_data)
    val_data_norm = scaler.transform(val_data)
    test_data_norm = scaler.transform(test_data)

    train_set = SeqDataset(train_data_norm, seq_len, pred_len)
    val_set = SeqDataset(val_data_norm, seq_len, pred_len)
    test_set = SeqDataset(test_data_norm, seq_len, pred_len)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    print(f"seq_len: {seq_len}, pred_len: {pred_len}")
    for name, arr in zip(['train', 'val', 'test'], [train_data, val_data, test_data]):
        if len(arr) < seq_len + pred_len:
            raise ValueError(f"{name} set too small for seq_len+pred_len ({seq_len+pred_len}), got {len(arr)}")

    # 模型
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
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    best_val_loss = float('inf')

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # 训练与验证
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        preds = []
        trues = []
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for x, y in progress_bar:
            x, y = x.to(device), y.to(device)
            # teacher forcing: decoder_input为y的左移一位，首位补零
            decoder_input = torch.zeros_like(y)
            decoder_input[:, 1:, :] = y[:, :-1, :]
            out = model(x, decoder_input)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            preds.append(out.detach().cpu().numpy())
            trues.append(y.detach().cpu().numpy())
            progress_bar.set_postfix(loss=loss.item())
        avg_train_loss = epoch_loss / len(train_loader)
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        # 反归一化
        preds_inv = scaler.inverse_transform(preds)
        trues_inv = scaler.inverse_transform(trues)
        rmse = np.sqrt(np.mean((preds_inv - trues_inv) ** 2))
        train_acc = 1 - rmse / (np.mean(np.abs(trues_inv)) + 1e-8)
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)

        avg_val_loss, val_acc = evaluate(model, val_loader, criterion, device, scaler)
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f} | Train Acc={train_acc:.4f} | Val Acc={val_acc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(">>> 最优模型已保存")

    print(f"训练结束，最佳验证集Loss: {best_val_loss:.4f}")

    # 保存损失曲线和准确率曲线
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.savefig(loss_curve_path)
    plt.close()

    plt.figure()
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')
    plt.savefig(acc_curve_path)
    plt.close()

    # 测试
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    test_losses = []
    preds = []
    trues = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            decoder_input = torch.zeros_like(y)
            decoder_input[:, 1:, :] = y[:, :-1, :]
            out = model(x, decoder_input)
            loss = criterion(out, y)
            test_losses.append(loss.item())
            preds.append(out.cpu().numpy())
            trues.append(y.cpu().numpy())
    test_loss = np.mean(test_losses)
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    # 反归一化
    preds_inv = scaler.inverse_transform(preds)
    trues_inv = scaler.inverse_transform(trues)
    rmse = np.sqrt(np.mean((preds_inv - trues_inv) ** 2))
    test_acc = 1 - rmse / (np.mean(np.abs(trues_inv)) + 1e-8)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    return preds_inv, trues_inv, scaler

def main():
    preds, trues, scaler = train_val_test(
        csv_path="data/electric_production/Electric_Production.csv",
        seq_len=24,
        pred_len=12,
        batch_size=32,
        epochs=100,
        lr=1e-3,
        val_ratio=0.1,
        test_ratio=0.1,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.1,
        max_len=512,
        device=None,
        output_dir="outputs/transformer"
    )
    plt.figure()
    plt.plot(trues.flatten(), label='True')
    plt.plot(preds.flatten(), label='Pred')
    plt.legend()
    plt.title('Prediction vs True')
    pred_vs_true_path = "outputs/transformer/pred_vs_true.png"
    plt.savefig(pred_vs_true_path)
    print(f"预测对比图已保存至: {pred_vs_true_path}")
    plt.show()
    
if __name__ == "__main__":
    main()