import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from scripts.preprocess_data import preprocess_data
from scripts.data_collection import collect_data  # Import collect_data from data_collection module

class TemporalTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, dropout=0.1):
        super(TemporalTransformer, self).__init__()
        self.input_layer = nn.Linear(input_dim, model_dim)  # Add input transformation layer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(model_dim, 1)
        
    def forward(self, x):
        x = self.input_layer(x)  # Transform input to match model_dim
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Ensure the mean is taken over the correct dimension
        x = self.fc(x)
        return x

def train_model(X, y, epochs=10, learning_rate=0.001):
    input_dim = X.shape[2]
    model_dim = 64
    num_heads = 4
    num_layers = 2
    dropout = 0.1

    model = TemporalTransformer(input_dim, model_dim, num_heads, num_layers, dropout)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    # Save the trained model
    torch.save(model.state_dict(), 'models/trained_model.pth')

    return model

if __name__ == "__main__":
    # Example usage
    ticker = "AAPL"
    period = "5y"
    file_name = f"data/{ticker}_historical_data.csv"
    
    # Collect and preprocess data
    collect_data(ticker, period)  # Collect data
    X, y, _ = preprocess_data(file_name)  # Preprocess data
    
    # Train the model
    train_model(X, y)
