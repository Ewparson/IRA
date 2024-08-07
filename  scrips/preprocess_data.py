from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import ta

def preprocess_data(file_path, sequence_length=60):
    data = pd.read_csv(file_path, index_col="Date")
    data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
    data['VWAP'] = ta.volume.VolumeWeightedAveragePrice(data['High'], data['Low'], data['Close'], data['Volume']).volume_weighted_average_price()
    macd = ta.trend.MACD(data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['MACD_Diff'] = macd.macd_diff()
    data['SMA_50'] = ta.trend.SMAIndicator(data['Close'], window=50).sma_indicator()
    data['SMA_200'] = ta.trend.SMAIndicator(data['Close'], window=200).sma_indicator()
    data['EMA_50'] = ta.trend.EMAIndicator(data['Close'], window=50).ema_indicator()
    data['EMA_200'] = ta.trend.EMAIndicator(data['Close'], window=200).ema_indicator()

    data = data[['Open', 'High', 'Low', 'Close', 'RSI', 'VWAP', 'MACD', 'MACD_Signal', 'MACD_Diff', 'SMA_50', 'SMA_200', 'EMA_50', 'EMA_200']]

    # Check for NaNs and handle them
    if data.isna().any().any():
        print("Data contains NaNs. Applying forward fill.")
        data.ffill(inplace=True)  # Forward fill
        data.bfill(inplace=True)  # Backward fill if any NaNs remain

    # Print statistics for debugging
    print("Data statistics before scaling: mean =", data.mean(), ", std =", data.std())

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Check for NaNs in scaled data
    if np.isnan(data_scaled).any():
        print("Scaled data contains NaNs!")

    X, y = [], []
    dates = data.index[sequence_length:]
    for i in range(sequence_length, len(data_scaled)):
        X.append(data_scaled[i-sequence_length:i])
        y.append(data_scaled[i, 3])  # Closing price
    
    return np.array(X), np.array(y), scaler, dates
