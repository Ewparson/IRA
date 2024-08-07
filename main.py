import matplotlib.pyplot as plt
import pandas as pd
from scripts.data_collection import collect_data
from scripts.preprocess_data import preprocess_data
from scripts.train_model import train_model, TemporalTransformer
import torch
import numpy as np

def main():
    # Collect data
    ticker = "AAPL"
    period = "max"
    print("Collecting data...")
    file_name = collect_data(ticker, period)
    print("Data collection complete. File saved as:", file_name)
    
    # Preprocess data
    print("Preprocessing data...")
    X, y, scaler, dates = preprocess_data(file_name)
    print(f"Preprocessing complete. Shape of X: {X.shape}, Shape of y: {y.shape}")
    
    # Train model
    input_dim = X.shape[2]  # Number of features
    print("Training model...")
    model = train_model(X, y, input_dim)
    print("Model training complete.")
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(X, dtype=torch.float32)).numpy()
    
    # Inverse transform predictions
    predictions = predictions.squeeze()
    predictions = scaler.inverse_transform(np.repeat(predictions[:, np.newaxis], input_dim, axis=1))[:, 3]
    actual = scaler.inverse_transform(np.repeat(y[:, np.newaxis], input_dim, axis=1))[:, 3]

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
