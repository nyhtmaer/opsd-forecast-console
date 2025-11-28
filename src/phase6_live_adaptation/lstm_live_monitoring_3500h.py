"""LSTM Live Monitoring - 3,500 hour evaluation with periodic retraining"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import json
import os
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

print(" LSTM LIVE MONITORING - 3,500 HOUR EVALUATION")
print("\nStrategy: Online Retraining (Equalized with SARIMA)")
print("  - Refit frequency: Every 336 hours (2 weeks)")
print("  - Minimum history: 60 days (1,440 hours)")
print("  - Simulation: 3,500 hours")

# Check CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# CONFIGURATION

COUNTRIES = ['DE', 'FR', 'IT']
LOAD_COLUMNS = {
    'DE': 'DE_load_actual_entsoe_transparency',
    'FR': 'FR_load_actual_entsoe_transparency',
    'IT': 'IT_load_actual_entsoe_transparency'
}

REFIT_FREQUENCY = 336
MIN_HISTORY = 1440
LOOKBACK = 168
HORIZON = 24
SIMULATION_HOURS = 3500

# LSTM MODEL

class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_size=24, dropout=0.2):
        super(LSTMForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, 
                           dropout=dropout if num_layers > 1 else 0)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
    
    def forward(self, x):
        x = x.unsqueeze(-1)
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        out = self.fc1(last_out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# DATASET

class TimeSeriesDataset(Dataset):
    def __init__(self, data, scaler, lookback=168, horizon=24):
        self.data_normalized = scaler.transform(data.values.reshape(-1, 1)).flatten()
        self.lookback = lookback
        self.horizon = horizon
    
    def __len__(self):
        return len(self.data_normalized) - self.lookback - self.horizon + 1
    
    def __getitem__(self, idx):
        x = self.data_normalized[idx:idx + self.lookback]
        y = self.data_normalized[idx + self.lookback:idx + self.lookback + self.horizon]
        return torch.FloatTensor(x), torch.FloatTensor(y)

# TRAINING FUNCTION

def train_model(train_data, scaler, device, epochs=25, batch_size=32, lr=0.001):
    """Quick training function for refits"""
    dataset = TimeSeriesDataset(train_data, scaler, LOOKBACK, HORIZON)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = LSTMForecaster().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.6f}")
    
    return model

# FORECAST FUNCTION

def generate_forecast(model, scaler, input_data, device):
    """Generate 24-hour forecast"""
    model.eval()
    with torch.no_grad():
        normalized = scaler.transform(input_data.reshape(-1, 1)).flatten()
        input_tensor = torch.FloatTensor(normalized[-LOOKBACK:]).unsqueeze(0).to(device)
        output = model(input_tensor)
        forecast = scaler.inverse_transform(output.cpu().numpy().reshape(-1, 1)).flatten()
    return forecast

# METRICS

def calculate_mase(actual, forecast, train_data, seasonal_period=24):
    mae_naive = np.mean(np.abs(train_data[seasonal_period:] - train_data[:-seasonal_period]))
    mae = np.mean(np.abs(actual - forecast))
    return mae / mae_naive if mae_naive > 0 else np.nan

def calculate_mape(actual, forecast):
    return 100 * np.mean(np.abs((actual - forecast) / actual))

def calculate_rmse(actual, forecast):
    return np.sqrt(np.mean((actual - forecast) ** 2))

# LOAD DATA

print("\n Loading dataset (FULL RUN)...")

df = pd.read_csv('time_series_60min_singleindex_filtered.csv', index_col=0, parse_dates=True)

simulation_data = {}
for country, col in LOAD_COLUMNS.items():
    data = df[col].dropna()
    # Get last portion for simulation (need MIN_HISTORY + SIMULATION_HOURS)
    total_needed = MIN_HISTORY + SIMULATION_HOURS + LOOKBACK
    data = data.iloc[-total_needed:]
    simulation_data[country] = data
    print(f"{country}: {len(data)} hours available")

# RUN SIMULATION

print("\n Running live monitoring simulation...")

all_results = {}

for country in COUNTRIES:
    print(f"\n{'='*60}")
    print(f"COUNTRY: {country}")
    print(f"{'='*60}")
    
    data = simulation_data[country]
    
    # Results storage
    results = {
        'timestamp': [],
        'actual': [],
        'forecast': [],
        'mase_rolling_24h': [],
        'mape_rolling_24h': [],
        'rmse_rolling_24h': [],
        'refit_flag': [],
        'model_age_hours': []
    }
    
    # Initialize scaler with first MIN_HISTORY hours
    initial_train = data.iloc[:MIN_HISTORY]
    scaler = StandardScaler()
    scaler.fit(initial_train.values.reshape(-1, 1))
    
    # Initial training
    print(f"\n  Initial training (hours 0-{MIN_HISTORY})...")
    model = train_model(initial_train, scaler, device, epochs=25)
    
    last_refit_hour = 0
    refit_count = 0
    
    # Simulate hour by hour
    for hour in range(SIMULATION_HOURS):
        current_idx = MIN_HISTORY + hour
        
        # Check if refit needed
        if (hour > 0) and (hour % REFIT_FREQUENCY == 0):
            print(f"\n  Refit #{refit_count + 1} at hour {hour}...")
            train_window = data.iloc[:current_idx]
            scaler = StandardScaler()
            scaler.fit(train_window.values.reshape(-1, 1))
            model = train_model(train_window, scaler, device, epochs=20)
            last_refit_hour = hour
            refit_count += 1
        
        # Generate forecast
        input_window = data.iloc[current_idx - LOOKBACK:current_idx].values
        forecast_24h = generate_forecast(model, scaler, input_window, device)
        
        # Get actual values for next 24 hours
        actual_24h = data.iloc[current_idx:current_idx + HORIZON].values
        
        # Calculate metrics
        train_for_mase = data.iloc[:current_idx].values
        mase = calculate_mase(actual_24h, forecast_24h, train_for_mase)
        mape = calculate_mape(actual_24h, forecast_24h)
        rmse = calculate_rmse(actual_24h, forecast_24h)
        
        # Store results (store first forecast hour)
        results['timestamp'].append(data.index[current_idx])
        results['actual'].append(actual_24h[0])
        results['forecast'].append(forecast_24h[0])
        results['mase_rolling_24h'].append(mase)
        results['mape_rolling_24h'].append(mape)
        results['rmse_rolling_24h'].append(rmse)
        results['refit_flag'].append(1 if hour % REFIT_FREQUENCY == 0 else 0)
        results['model_age_hours'].append(hour - last_refit_hour)
        
        # Progress update
        if (hour + 1) % 200 == 0:
            print(f"  Progress: {hour + 1}/{SIMULATION_HOURS} hours | MAPE: {mape:.2f}% | Refits: {refit_count}")
    
    all_results[country] = pd.DataFrame(results)
    
    avg_mape = np.mean(results['mape_rolling_24h'])
    avg_mase = np.mean(results['mase_rolling_24h'])
    print(f"\n   Complete: Avg MAPE={avg_mape:.2f}%, Avg MASE={avg_mase:.4f}, Refits={refit_count}")

# SAVE RESULTS

print("\n Saving results...")

os.makedirs('results/phase6_live_adaptation', exist_ok=True)

for country in COUNTRIES:
    filename = f'results/phase6_live_adaptation/{country}_lstm_live_simulation.csv'
    all_results[country].to_csv(filename, index=False)
    print(f" Saved {filename}")

# Save summary
summary = {}
for country in COUNTRIES:
    df = all_results[country]
    summary[country] = {
        'total_hours': int(len(df)),
        'num_refits': int(df['refit_flag'].sum()),
        'avg_mase': float(df['mase_rolling_24h'].mean()),
        'avg_mape': float(df['mape_rolling_24h'].mean()),
        'avg_rmse': float(df['rmse_rolling_24h'].mean())
    }

with open('results/phase6_live_adaptation/lstm_simulation_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(" Saved lstm_simulation_summary.json")

# VISUALIZATIONS

print("\n Creating visualizations...")

# Performance evolution
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

for idx, country in enumerate(COUNTRIES):
    ax = axes[idx]
    df = all_results[country]
    
    ax.plot(df['timestamp'], df['mape_rolling_24h'], label='MAPE (%)', color='steelblue', linewidth=1.5)
    
    # Mark refits
    refits = df[df['refit_flag'] == 1]
    ax.scatter(refits['timestamp'], refits['mape_rolling_24h'], color='red', s=50, 
              marker='v', label='Model Refit', zorder=5)
    
    ax.set_title(f'{country} - LSTM Performance Evolution (Avg MAPE: {summary[country]["avg_mape"]:.2f}%)', 
                fontsize=12, fontweight='bold')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('MAPE (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/phase6_live_adaptation/lstm_performance_evolution.png', dpi=300, bbox_inches='tight')
print(" Saved lstm_performance_evolution.png")
plt.close()

print("\n" + "="*80)
print("LSTM LIVE MONITORING COMPLETE")
for country in COUNTRIES:
    print(f"{country}: MAPE {summary[country]['avg_mape']:.2f}% | MASE {summary[country]['avg_mase']:.4f} | Refits {summary[country]['num_refits']}")
