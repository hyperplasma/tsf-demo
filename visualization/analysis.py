import os
import sys
import pandas as pd

def analyze_dataset(dataset_name):
    data_path = os.path.join('dataset', f'{dataset_name}.csv')
    if not os.path.exists(data_path):
        print(f"[Error] Dataset file not found: {data_path}")
        return
    print(f"\nDataset path: {data_path}")

    # 只读取前1000行，防止大文件卡死
    try:
        df = pd.read_csv(data_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(data_path, encoding='gbk')
    except Exception as e:
        print(f"[Error] Failed to read CSV: {e}")
        return

    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print("\nDtypes:")
    print(df.dtypes)
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nDescriptive statistics:")
    print(df.describe(include='all').T)

    print("\nUnique value count per column:")
    for col in df.columns:
        nunique = df[col].nunique(dropna=False)
        print(f"  {col}: {nunique}")

    print("\nFirst 5 rows:")
    print(df.head())
    print("\nLast 5 rows:")
    print(df.tail())

if __name__ == "__main__":
    dataset_name = "weather"
    analyze_dataset(dataset_name)
