import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

def download_btc_data(start='2018-01-01', end='2025-01-01'):
    """Bitcoin verisini Yahoo Finance'den indirir."""
    btc_data = yf.download('BTC-USD', start=start, end=end)
    btc_data = btc_data[['Close']]
    btc_data = btc_data.dropna()
    return btc_data

def scale_data(data):
    """Veriyi 0-1 arasına ölçekler."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    return scaled_data, scaler

def train_test_split(scaled_data, train_ratio=0.8):
    """Veriyi train ve test olarak ayırır."""
    train_size = int(len(scaled_data) * train_ratio)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]
    return train_data, test_data

def create_dataset(data, time_step=60):
    """LSTM için X ve y oluşturur."""
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)
