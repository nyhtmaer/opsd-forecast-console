"""
LSTM Model Building with GPU/CUDA Support
Implements LSTM neural network for comparison with SARIMA models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
# PyTorch + CUDA
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

os.makedirs('results/phase3b_lstm_de_fr_it_results', exist_ok=True)

print("LSTM MODEL BUILDING WITH GPU/CUDA - DE/FR/IT")
# GPU check
print("\n Checking CUDA/GPU availability...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if not torch.cuda.is_available():
    print("CUDA not available - using CPU (will be slower)")

# Load data
print("\n Loading preprocessed data...")

# Load train and validation data
df_train = pd.read_csv('data/preprocessed/train_data.csv', parse_dates=['utc_timestamp'])
df_val = pd.read_csv('data/preprocessed/val_data.csv', parse_dates=['utc_timestamp'])

countries = ['DE', 'FR', 'IT']
load_columns = {
    'DE': 'DE_load_actual_entsoe_transparency',
    'FR': 'FR_load_actual_entsoe_transparency',
    'IT': 'IT_load_actual_entsoe_transparency'
}

# Prepare data splits using preprocessed train/val
data_splits = {}
for country, load_col in load_columns.items():
    train_data = df_train[load_col].dropna()
    val_data = df_val[load_col].dropna()
    
    # Use full train and validation data
    data_splits[country] = {
        'train': train_data.values,
        'dev': val_data.values,
        'test': val_data.values
    }
    
    print(f"{country}: Train={len(train_data)} | Dev={len(val_data)} | Test={len(val_data)} hours")

# Dataset class

class TimeSeriesDataset(Dataset):
    """Time series dataset with sliding window"""
    
    def __init__(self, data, lookback=168, horizon=24):
        """
        Args:
            data: pandas Series or numpy array of time series data
            lookback: lookback period (default: 168 = 1 week)
            horizon: forecast horizon (default: 24)
        """
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            self.data = data.values
        self.lookback = lookback
        self.horizon = horizon
        
        # Normalize data
        self.scaler = StandardScaler()
        self.data_normalized = self.scaler.fit_transform(self.data.reshape(-1, 1)).flatten()
        
    def __len__(self):
        return len(self.data) - self.lookback - self.horizon + 1
    
    def __getitem__(self, idx):
        # Input: lookback window
        x = self.data_normalized[idx:idx + self.lookback]
        # Target: next 24 hours
        y = self.data_normalized[idx + self.lookback:idx + self.lookback + self.horizon]
        
        return torch.FloatTensor(x), torch.FloatTensor(y)

# Model architecture

class LSTMForecaster(nn.Module):
    """LSTM for 24-hour ahead forecasting"""
    
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_size=24, dropout=0.2):
        super(LSTMForecaster, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
    def forward(self, x):
        # x shape: (batch, lookback)
        x = x.unsqueeze(-1)  # (batch, lookback, 1)
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, lookback, hidden)
        
        # Take last output
        last_out = lstm_out[:, -1, :]  # (batch, hidden)
        
        # Fully connected
        out = self.fc1(last_out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)  # (batch, 24)
        
        return out

# Training function

def train_lstm_model(train_dataset, dev_dataset, device, epochs=50, batch_size=32, lr=0.001):
    """Train LSTM model with early stopping"""
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    model = LSTMForecaster(
        input_size=1,
        hidden_size=128,
        num_layers=2,
        output_size=24,
        dropout=0.2
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training history
    train_losses = []
    dev_losses = []
    best_dev_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    print(f"  Training on {device}...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        dev_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in dev_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                dev_loss += loss.item()
        
        dev_loss /= len(dev_loader)
        dev_losses.append(dev_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Dev Loss: {dev_loss:.6f}")
        
        # Early stopping
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model, train_losses, dev_losses

# TRAIN MODELS FOR EACH COUNTRY
print("\n Training LSTM models...")

trained_models = {}

for country in countries:
    print(f"\n{'='*60}")
    print(f"Training LSTM: {country}")
    print(f"{'='*60}")
    
    # Create datasets
    train_dataset = TimeSeriesDataset(
        data_splits[country]['train'],
        lookback=168,  # 1 week
        horizon=24      # 24 hours ahead
    )
    
    dev_dataset = TimeSeriesDataset(
        data_splits[country]['dev'],
        lookback=168,
        horizon=24
    )
    
    # Copy scaler for later use
    scaler = train_dataset.scaler
    
    # Train model
    model, train_losses, dev_losses = train_lstm_model(
        train_dataset, 
        dev_dataset, 
        device,
        epochs=50,
        batch_size=32,
        lr=0.001
    )
    
    trained_models[country] = {
        'model': model,
        'scaler': scaler,
        'train_losses': train_losses,
        'dev_losses': dev_losses
    }
    
    print(f" Training complete for {country}")

# SAVE MODELS
print("\n Saving trained models...")

for country in countries:
    model_path = f'results/phase3b_lstm_results/lstm_model_{country}.pt'
    torch.save({
        'model_state_dict': trained_models[country]['model'].state_dict(),
        'scaler_mean': trained_models[country]['scaler'].mean_[0],
        'scaler_scale': trained_models[country]['scaler'].scale_[0],
    }, model_path)
    print(f" Saved: {model_path}")

# PLOT TRAINING HISTORY
print("\n Plotting training history...")

for country in countries:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    train_losses = trained_models[country]['train_losses']
    dev_losses = trained_models[country]['dev_losses']
    
    ax.plot(train_losses, label='Training Loss', linewidth=2)
    ax.plot(dev_losses, label='Validation Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss', fontsize=12)
    ax.set_title(f'LSTM Training History: {country}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/phase3b_lstm_results/training_history_{country}.png', dpi=300, bbox_inches='tight')
    print(f" Saved: results/phase3b_lstm_results/training_history_{country}.png")
    plt.close()

# MODEL SUMMARY
print("\n Creating model summary...")

model_summary = {
    'architecture': {
        'type': 'LSTM',
        'input_size': 1,
        'hidden_size': 128,
        'num_layers': 2,
        'output_size': 24,
        'dropout': 0.2,
        'lookback_window': 168,
        'forecast_horizon': 24
    },
    'training': {
        'device': str(device),
        'cuda_available': torch.cuda.is_available(),
        'epochs_max': 50,
        'batch_size': 32,
        'learning_rate': 0.001,
        'early_stopping_patience': 10
    },
    'models': {}
}

for country in countries:
    final_train_loss = trained_models[country]['train_losses'][-1]
    final_dev_loss = trained_models[country]['dev_losses'][-1]
    epochs_trained = len(trained_models[country]['train_losses'])
    
    model_summary['models'][country] = {
        'final_train_loss': round(final_train_loss, 6),
        'final_dev_loss': round(final_dev_loss, 6),
        'epochs_trained': epochs_trained,
        'model_file': f'lstm_model_{country}.pt'
    }

with open('results/phase3b_lstm_results/model_summary.json', 'w') as f:
    json.dump(model_summary, f, indent=2)
print(" Saved: results/phase3b_lstm_results/model_summary.json")

# PRINT SUMMARY
print("\n Summary of trained models...")

summary_df = pd.DataFrame({
    country: {
        'Train Loss': f"{trained_models[country]['train_losses'][-1]:.6f}",
        'Dev Loss': f"{trained_models[country]['dev_losses'][-1]:.6f}",
        'Epochs': len(trained_models[country]['train_losses'])
    }
    for country in countries
}).T

print("\n" + "="*80)
print("LSTM MODEL SUMMARY")
print(summary_df.to_string())

print("\n" + "="*80)
print(f"\nDevice Used: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("\nGenerated Files:")
print("  1. results/phase3b_lstm_results/lstm_model_AT.pt")
print("  2. results/phase3b_lstm_results/lstm_model_BE.pt")
print("  3. results/phase3b_lstm_results/lstm_model_BG.pt")
print("  4. results/phase3b_lstm_results/training_history_AT.png")
print("  5. results/phase3b_lstm_results/training_history_BE.png")
print("  6. results/phase3b_lstm_results/training_history_BG.png")
print("  7. results/phase3b_lstm_results/model_summary.json")
print("\nLSTM models trained and ready for Phase 4b forecasting!")
 