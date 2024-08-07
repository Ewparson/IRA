import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class TemporalTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, dropout):
        super(TemporalTransformer, self).__init__()
        self.input_layer = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(model_dim, 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.transformer(x)
        x = self.output_layer(x[:, -1, :])
        return x

def train_model(X_train, y_train, input_dim):
    model = TemporalTransformer(input_dim=input_dim, model_dim=64, num_heads=4, num_layers=2, dropout=0.1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Reduced learning rate
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            if torch.isnan(loss):
                print("Loss is NaN. Exiting training.")
                return model
            loss.backward()

            # Check for NaNs in gradients
            for name, param in model.named_parameters():
                if torch.isnan(param.grad).any():
                    print(f"NaN detected in gradients of {name}. Exiting training.")
                    return model

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

    torch.save(model.state_dict(), 'models/trained_model.pth')
    print("Model training complete and saved as 'models/trained_model.pth'.")
    return model

if __name__ == "__main__":
    from scripts.data_collection import collect_data
    from scripts.preprocess_data import preprocess_data
    
    ticker = "AAPL"
    period = "max"
    
    # Step 1: Collect data
    file_path = collect_data(ticker, period)
    
    # Step 2: Preprocess data
    X, y, scaler, dates = preprocess_data(file_path)
    
    # Step 3: Train model
    input_dim = X.shape[2]
    model = train_model(X, y, input_dim)
