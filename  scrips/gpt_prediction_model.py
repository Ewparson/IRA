# main.py
import matplotlib.pyplot as plt
import pandas as pd
from data_collection import collect_data
from preprocess_data import preprocess_data
from gpt_prediction_model import train_model
import torch

def main():
    # Collect data
    ticker = "AAPL"
    period = "5y"
    print("Collecting data...")
    collect_data(ticker, period)
    print("Data collection complete.")
    
    # Preprocess data
    print("Preprocessing data...")
    X, y, dates = preprocess_data(f"{ticker}_historical_data.csv")
    print(f"Preprocessing complete. Shape of X: {X.shape}, Shape of y: {y.shape}")
    
    # Train model
    print("Training model...")
    model = train_model(X, y)
    print("Model training complete.")
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(X, dtype=torch.float32)).numpy()
    
    # Inverse transform predictions
    actual = y  # already in original scale
    predictions = predictions.squeeze()

    # Convert dates from string to datetime
    dates = pd.to_datetime(dates[-len(predictions):])

    # Plot results
    plt.figure(figsize=(14, 7))
    plt.plot(dates, actual, label='Actual Price')
    plt.plot(dates, predictions, label='Predicted Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
