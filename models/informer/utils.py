import numpy as np

def load_csv_data(path):
    return np.loadtxt(path, delimiter=',', skiprows=1)