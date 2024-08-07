import torch
import torch.nn as nn
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from scripts.preprocess_data import preprocess_data
from scripts.train_model import TemporalTransformer
from sklearn.preprocessing import MinMaxScaler
import ta

class LiveModel:
    def __init__(self, model_path, input_dim, model_dim, num_heads, num_layers, dropout=0.1):
        self.model = TemporalTransformer(input_dim, model_dim, num_heads, num_layers, dropout)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.scaler = None

    def fetch_live_data(self, ticker, start_date, end_date):
        data = yf.download(ticker, start=start_date, end=end_date)
        return data

    def preprocess_live_data(self, data, sequence_length=60):
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
        if data.isna().any().any():
            print("Data contains NaNs. Applying forward fill.")
            data.ffill(inplace=True)  # Forward fill
            data.bfill(inplace=True)  # Backward fill if any NaNs remain

        self.scaler = MinMaxScaler()
        data_scaled = self.scaler.fit_transform(data)

        # Check for NaNs in scaled data
        if np.isnan(data_scaled).any():
            print("Scaled data contains NaNs!")
        
        X = []
        for i in range(sequence_length, len(data_scaled)):
            X.append(data_scaled[i-sequence_length:i])
        return np.array(X)

    def make_prediction(self, X):
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            predictions = self.model(X_tensor).numpy()
        if np.isnan(predictions).any():
            print("Predictions contain NaNs!")
        return predictions

    def simulate_trading(self, ticker, start_date, end_date, wallet, sequence_length=60, rsi_buy_threshold=30, sell_threshold=0.02, stop_loss=0.05, take_profit=0.1):
        data = self.fetch_live_data(ticker, start_date, end_date)
        if len(data) < sequence_length:
            print("Not enough data to make predictions.")
            return

        X = self.preprocess_live_data(data, sequence_length)
        predictions = self.make_prediction(X)
        predictions = self.scaler.inverse_transform(np.repeat(predictions, 13, axis=1))[:, 3]

        current_cash = wallet['cash']
        current_holdings = wallet['holdings']
        journal = []  # Trade journal

        entry_price = None

        for i in range(len(predictions) - 1):
            current_price = data['Close'].iloc[sequence_length + i]
            predicted_price = predictions[i + 1]
            rsi = data['RSI'].iloc[sequence_length + i]
            sma_50 = data['SMA_50'].iloc[sequence_length + i]
            sma_200 = data['SMA_200'].iloc[sequence_length + i]

            print(f"Current price: {current_price}, Predicted price: {predicted_price}, RSI: {rsi}, SMA_50: {sma_50}, SMA_200: {sma_200}")

            # Check for overall bullish trend
            if sma_50 > sma_200:
                # Buy condition: RSI is below the threshold indicating oversold condition
                if rsi < rsi_buy_threshold and current_cash > current_price:
                    current_cash -= current_price
                    current_holdings += 1
                    entry_price = current_price
                    journal.append((data.index[sequence_length + i], "Buy", current_price, current_cash, current_holdings))
                    print(f"Buying at {current_price}, new cash: {current_cash}, holdings: {current_holdings}")

                # Sell condition: Predicted price is lower or hitting the profit target
                elif predicted_price < current_price * (1 - sell_threshold) and current_holdings > 0:
                    current_cash += current_price
                    current_holdings -= 1
                    journal.append((data.index[sequence_length + i], "Sell", current_price, current_cash, current_holdings))
                    print(f"Selling at {current_price}, new cash: {current_cash}, holdings: {current_holdings}")

                # Stop-loss condition
                if entry_price and current_holdings > 0 and current_price < entry_price * (1 - stop_loss):
                    current_cash += current_price
                    current_holdings -= 1
                    journal.append((data.index[sequence_length + i], "Stop Loss", current_price, current_cash, current_holdings))
                    print(f"Stop Loss at {current_price}, new cash: {current_cash}, holdings: {current_holdings}")

                # Take-profit condition
                if entry_price and current_holdings > 0 and current_price > entry_price * (1 + take_profit):
                    current_cash += current_price
                    current_holdings -= 1
                    journal.append((data.index[sequence_length + i], "Take Profit", current_price, current_cash, current_holdings))
                    print(f"Take Profit at {current_price}, new cash: {current_cash}, holdings: {current_holdings}")

        wallet['cash'] = current_cash
        wallet['holdings'] = current_holdings

        # Save trade journal to CSV
        journal_df = pd.DataFrame(journal, columns=["Date", "Action", "Price", "Cash", "Holdings"])
        journal_df.to_csv("trade_journal.csv", index=False)
        print("Trade journal saved to trade_journal.csv")

# Initialize wallet
wallet = {
    'cash': 10000,  # Initial cash amount
    'holdings': 0  # Initial number of shares
}

# Simulate trading with live data
start_date = '2023-01-01'
end_date = datetime.datetime.today().strftime('%Y-%m-%d')

live_model = LiveModel('models/trained_model.pth', input_dim=13, model_dim=64, num_heads=4, num_layers=2, dropout=0.1)
live_model.simulate_trading('AAPL', start_date, end_date, wallet)

print(f"Final wallet state: {wallet}")