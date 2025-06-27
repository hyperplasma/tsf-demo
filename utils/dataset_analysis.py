import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def plot_series(csv_path, column='IPG2211A2N', date_column='DATE', n=None, title='数据时序图', show_xticks=True):
    """
    绘制时间序列曲线
    :param csv_path: CSV文件路径
    :param column: 要绘制的数值列名
    :param date_column: 日期列名
    :param n: 只显示最近n条（为None则显示全部）
    :param title: 图标题
    :param show_xticks: 是否显示x轴刻度（True显示，False隐藏）
    """
    df = pd.read_csv(csv_path)
    if n is not None:
        df = df.tail(n)
    plt.figure(figsize=(10, 5))
    plt.plot(df[date_column], df[column], label=column)
    plt.xlabel('日期')
    plt.ylabel(column)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if not show_xticks:
        plt.xticks([])
    else:
        plt.xticks(rotation=45)
    plt.show()

class CSVAnalyzer:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = pd.read_csv(filepath)

    def print_basic_info(self):
        print(f"文件路径: {self.filepath}")
        print(f"数据行数（含表头）: {self.df.shape[0]}")
        print(f"数据列数: {self.df.shape[1]}")
        print("列名:", list(self.df.columns))

    def print_head(self, n=5):
        print(f"前{n}行数据:")
        print(self.df.head(n))

    def print_tail(self, n=5):
        print(f"后{n}行数据:")
        print(self.df.tail(n))

    def print_dtypes(self):
        print("各列数据类型:")
        print(self.df.dtypes)

    def print_missing_info(self):
        print("各列缺失值数量:")
        print(self.df.isnull().sum())

    def print_describe(self):
        print("数值型数据统计信息:")
        print(self.df.describe())

    def print_structure(self):
        print("数据结构信息:")
        print(self.df.info())


if __name__ == "__main__":
    data_file = "data/electric_production/Electric_Production.csv"
    analyzer = CSVAnalyzer(data_file)
    analyzer.print_basic_info()
    analyzer.print_head()
    analyzer.print_tail()
    analyzer.print_dtypes()
    analyzer.print_missing_info()
    analyzer.print_describe()
    analyzer.print_structure()
    
    plot_series(
        csv_path="data/electric_production/Electric_Production.csv",
        column='IPG2211A2N',
        date_column='DATE',
        title='美国电力生产时序图',
        # show_xticks=False,
    )