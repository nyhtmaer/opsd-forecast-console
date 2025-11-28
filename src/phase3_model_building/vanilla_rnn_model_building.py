"""
Vanilla/Classic RNN Model Building with GPU/CUDA Support
Implements classic RNN neural network for comparison with SARIMA, LSTM, and GRU models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import time
# PyTorch + CUDA
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

os.makedirs('results/phase3e_vanilla_rnn_de_fr_it_results', exist_ok=True)

print("VANILLA RNN MODEL BUILDING WITH GPU/CUDA - DE/FR/IT")
# GPU check
print("\n Checking CUDA/GPU availability...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(" CUDA is available!")
    print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"  CUDA Version: {torch.version.cuda}")
else:
    print("  CUDA not available - using CPU (will be slower)")

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
    
    def __init__(self, data, lookback=168, horizon=24, scaler=None):
        """
        Args:
            data: pandas Series or numpy array of time series data
            lookback: lookback period (default: 168 = 1 week)
            horizon: forecast horizon (default: 24)
            scaler: optional pre-fitted scaler for dev/test sets
        """
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            self.data = data.values
        self.lookback = lookback
        self.horizon = horizon
        
        # Normalize data
        if scaler is None:
            self.scaler = StandardScaler()
            self.data_normalized = self.scaler.fit_transform(self.data.reshape(-1, 1)).flatten()
        else:
            self.scaler = scaler
            self.data_normalized = self.scaler.transform(self.data.reshape(-1, 1)).flatten()
        
    def __len__(self):
        return len(self.data) - self.lookback - self.horizon + 1
    
    def __getitem__(self, idx):
        # Input: lookback window
        x = self.data_normalized[idx:idx + self.lookback]
        # Target: next 24 hours
        y = self.data_normalized[idx + self.lookback:idx + self.lookback + self.horizon]
        
        return torch.FloatTensor(x), torch.FloatTensor(y)

# Model architecture

class VanillaRNNForecaster(nn.Module):
    """
    Vanilla/Classic RNN model for multi-step time series forecasting
    
    Vanilla RNN characteristics:
    - Simplest recurrent architecture
    - Uses tanh activation (or relu)
    - Prone to vanishing/exploding gradients
    - Struggles with long-term dependencies
    - Good baseline for comparison with LSTM/GRU
    
    Differences from LSTM/GRU:
    - No gating mechanisms
    - No cell state (only hidden state)
    - Faster but less powerful for long sequences
    """
    
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_size=24, dropout=0.2):
        super(VanillaRNNForecaster, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Vanilla RNN layers
        # Uses simple recurrent connections with tanh
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            nonlinearity='tanh'  # Classic choice (can also use 'relu')
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
    def forward(self, x):
        # x shape: (batch, lookback)
        x = x.unsqueeze(-1)  # (batch, lookback, 1)
        
        # RNN forward pass
        # h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b)
        rnn_out, _ = self.rnn(x)
        
        # Take last time step's output
        last_hidden = rnn_out[:, -1, :]  # (batch, hidden_size)
        
        # Fully connected layers
        x = self.fc1(last_hidden)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # (batch, output_size)
        
        return x

# Training function

def train_vanilla_rnn_model(model, train_loader, dev_loader, num_epochs=50, lr=0.001, country=""):
    """Train Vanilla RNN model with gradient clipping to prevent exploding gradients"""
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    train_losses = []
    dev_losses = []
    best_dev_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    max_patience = 10
    
    # Gradient clipping value (important for vanilla RNN)
    clip_value = 1.0
    
    start_time = time.time()
    
    print(f"\nTraining Vanilla RNN for {country}...")
    print(f"  Epochs: {num_epochs} | Learning Rate: {lr} | Device: {device}")
    print(f"  Gradient clipping: {clip_value}")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (crucial for vanilla RNN)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        dev_loss = 0.0
        
        with torch.no_grad():
            for x_batch, y_batch in dev_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                dev_loss += loss.item()
        
        dev_loss /= len(dev_loader)
        dev_losses.append(dev_loss)
        
        # Learning rate scheduling
        scheduler.step(dev_loss)
        
        # Save best model
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Dev Loss: {dev_loss:.6f} | "
                  f"Best Dev: {best_dev_loss:.6f}")
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f"\n   Early stopping at epoch {epoch+1}")
            break
    
    training_time = time.time() - start_time
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"   Training completed in {training_time:.1f} seconds")
    print(f"   Best dev loss: {best_dev_loss:.6f}")
    
    return {
        'train_losses': train_losses,
        'dev_losses': dev_losses,
        'best_dev_loss': best_dev_loss,
        'final_epoch': epoch + 1,
        'training_time_seconds': training_time
    }

# TRAIN MODELS FOR ALL COUNTRIES
print("\n Training Vanilla RNN models...")

# Hyperparameters
LOOKBACK = 168  # 7 days
HORIZON = 24    # 1 day ahead
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001

# Store results
training_results = {}

for country in countries:
    print(f"\n{'='*80}")
    print(f"Training Vanilla RNN: {country}")
    print(f"{'='*80}")
    
    # Prepare datasets
    train_dataset = TimeSeriesDataset(
        data_splits[country]['train'],
        lookback=LOOKBACK,
        horizon=HORIZON
    )
    
    dev_dataset = TimeSeriesDataset(
        data_splits[country]['dev'],
        lookback=LOOKBACK,
        horizon=HORIZON,
        scaler=train_dataset.scaler  # Use same scaler as training
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)} | Dev samples: {len(dev_dataset)}")
    
    # Initialize Vanilla RNN model
    model = VanillaRNNForecaster(
        input_size=1,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=HORIZON,
        dropout=DROPOUT
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Train
    history = train_vanilla_rnn_model(
        model, train_loader, dev_loader,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        country=country
    )
    
    # Save model
    model_path = f"results/phase3e_vanilla_rnn_results/{country}_vanilla_rnn_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': train_dataset.scaler,
        'lookback': LOOKBACK,
        'horizon': HORIZON,
        'hidden_size': HIDDEN_SIZE,
        'num_layers': NUM_LAYERS,
        'num_parameters': num_params,
        'training_time': history['training_time_seconds']
    }, model_path)
    
    print(f"   Model saved to {model_path}")
    
    # Store results
    training_results[country] = {
        'num_parameters': num_params,
        'training_time_seconds': history['training_time_seconds'],
        'final_epoch': history['final_epoch'],
        'best_dev_loss': history['best_dev_loss'],
        'train_losses': history['train_losses'],
        'dev_losses': history['dev_losses']
    }

# SAVE TRAINING RESULTS
print("\n Saving training results...")

# Save JSON summary
json_results = {}
for country in training_results:
    json_results[country] = {
        'num_parameters': int(training_results[country]['num_parameters']),
        'training_time_seconds': float(training_results[country]['training_time_seconds']),
        'final_epoch': int(training_results[country]['final_epoch']),
        'best_dev_loss': float(training_results[country]['best_dev_loss'])
    }

with open('results/phase3e_vanilla_rnn_results/training_summary.json', 'w') as f:
    json.dump(json_results, f, indent=2)

print(" Training summary saved")

# PLOT TRAINING CURVES
print("\n Plotting training curves...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Vanilla RNN Training Curves', fontsize=16, fontweight='bold')

for idx, country in enumerate(countries):
    ax = axes[idx]
    
    epochs = range(1, len(training_results[country]['train_losses']) + 1)
    
    ax.plot(epochs, training_results[country]['train_losses'], 
            label='Train Loss', linewidth=2, marker='o', markersize=3)
    ax.plot(epochs, training_results[country]['dev_losses'], 
            label='Dev Loss', linewidth=2, marker='s', markersize=3)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss', fontsize=12)
    ax.set_title(f'{country}\n({training_results[country]["training_time_seconds"]:.1f}s training)', 
                 fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/phase3e_vanilla_rnn_results/training_curves.png', dpi=150, bbox_inches='tight')
plt.close()

print(" Training curves saved")

# TRAINING SUMMARY TABLE
print("\n Creating training summary...")

summary_data = []
for country in countries:
    summary_data.append({
        'Country': country,
        'Parameters': training_results[country]['num_parameters'],
        'Training_Time_s': round(training_results[country]['training_time_seconds'], 1),
        'Final_Epoch': training_results[country]['final_epoch'],
        'Best_Dev_Loss': round(training_results[country]['best_dev_loss'], 6)
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('results/phase3e_vanilla_rnn_results/training_summary.csv', index=False)

print("\n" + "="*80)
print("VANILLA RNN TRAINING SUMMARY")
print(summary_df.to_string(index=False))

# FINAL MESSAGE
print("\n Complete!")
print("\n" + "="*80)
print("\nVanilla RNN characteristics:")
print("  Simplest recurrent architecture")
print("  No gating mechanisms (unlike LSTM/GRU)")
print("  Prone to vanishing/exploding gradients")
print("  Good baseline for comparison")
print("  Gradient clipping used to stabilize training")
print("\nNext step: Run phase4e_vanilla_rnn_forecasting.py for rolling window forecasting")
