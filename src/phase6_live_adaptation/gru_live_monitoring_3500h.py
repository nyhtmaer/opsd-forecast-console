"""GRU Live Monitoring - 3,500 hour evaluation with periodic retraining"""

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

print(" GRU LIVE MONITORING - 3,500 HOUR EVALUATION")
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

# GRU MODEL

class GRUForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_size=24, dropout=0.2):
        super(GRUForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, 
                         dropout=dropout if num_layers > 1 else 0)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
    
    def forward(self, x):
        x = x.unsqueeze(-1)
        gru_out, _ = self.gru(x)
        last_out = gru_out[:, -1, :]
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

# TRAINING & FORECASTING

def train_model(model, train_loader, epochs=20, lr=0.001):
    """Train GRU model"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.6f}")
    
    return model

def generate_forecast(model, recent_data, scaler):
    """Generate 24-hour forecast"""
    model.eval()
    with torch.no_grad():
        recent_normalized = scaler.transform(recent_data.values.reshape(-1, 1)).flatten()
        X = torch.FloatTensor(recent_normalized[-LOOKBACK:]).unsqueeze(0).to(device)
        forecast_normalized = model(X).cpu().numpy().flatten()
        forecast = scaler.inverse_transform(forecast_normalized.reshape(-1, 1)).flatten()
    return forecast

# METRICS

def calculate_mase(actual, forecast, train_data):
    """Mean Absolute Scaled Error"""
    mae = np.mean(np.abs(actual - forecast))
    naive_mae = np.mean(np.abs(np.diff(train_data)))
    return mae / naive_mae if naive_mae > 0 else np.inf

def calculate_mape(actual, forecast):
    """Mean Absolute Percentage Error"""
    return np.mean(np.abs((actual - forecast) / actual)) * 100

def calculate_rmse(actual, forecast):
    """Root Mean Squared Error"""
    return np.sqrt(np.mean((actual - forecast) ** 2))

# MAIN SIMULATION

print("\n Loading dataset (FULL RUN)...")
df = pd.read_csv('time_series_60min_singleindex_filtered.csv')
df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
df = df.sort_values('utc_timestamp').reset_index(drop=True)

for country in COUNTRIES:
    print(f"{country}: {df[LOAD_COLUMNS[country]].notna().sum()} hours available")

print("\n Running live monitoring simulation...")

all_results = {}

for country in COUNTRIES:
    print("\n" + "="*60)
    print(f"COUNTRY: {country}")
        # Extract country data
    country_df = df[['utc_timestamp', LOAD_COLUMNS[country]]].copy()
    country_df.columns = ['timestamp', 'load']
    country_df = country_df.dropna().reset_index(drop=True)
    
    # Initialize results storage
    results = []
    refit_count = 0
    
    # Simulation loop
    for t in range(MIN_HISTORY, MIN_HISTORY + SIMULATION_HOURS, HORIZON):
        if t + HORIZON > len(country_df):
            break
        
        # Check if refit is needed
        hours_since_start = t - MIN_HISTORY
        should_refit = (hours_since_start % REFIT_FREQUENCY == 0)
        
        if should_refit or t == MIN_HISTORY:
            refit_count += 1
            if t == MIN_HISTORY:
                print(f"\n  Initial training (hours 0-{MIN_HISTORY})...")
                epochs = 25
            else:
                print(f"\n  Refit #{refit_count-1} at hour {hours_since_start}...")
                epochs = 20
            
            # Prepare training data
            train_data = country_df['load'].iloc[:t]
            scaler = StandardScaler()
            scaler.fit(train_data.values.reshape(-1, 1))
            
            # Create dataset and loader
            train_dataset = TimeSeriesDataset(train_data, scaler, LOOKBACK, HORIZON)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            # Train model
            model = GRUForecaster().to(device)
            model = train_model(model, train_loader, epochs=epochs)
        
        # Generate forecast
        recent_data = country_df['load'].iloc[t-LOOKBACK:t]
        forecast = generate_forecast(model, recent_data, scaler)
        
        # Get actual values
        actual = country_df['load'].iloc[t:t+HORIZON].values
        
        # Calculate metrics
        train_for_mase = country_df['load'].iloc[:t].values
        mase = calculate_mase(actual, forecast, train_for_mase)
        mape = calculate_mape(actual, forecast)
        rmse = calculate_rmse(actual, forecast)
        
        # Store results
        for h in range(HORIZON):
            results.append({
                'timestamp': country_df['timestamp'].iloc[t + h],
                'actual': actual[h],
                'forecast': forecast[h],
                'mase': mase,
                'mape': mape,
                'rmse': rmse,
                'refit_flag': 1 if should_refit and h == 0 else 0
            })
        
        # Progress update
        if (t - MIN_HISTORY) % 200 == 0:
            print(f"  Progress: {t-MIN_HISTORY}/{SIMULATION_HOURS} hours | MAPE: {mape:.2f}% | Refits: {refit_count-1}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    results_df['mase_rolling_24h'] = results_df['mase']
    results_df['mape_rolling_24h'] = results_df['mape']
    results_df['rmse_rolling_24h'] = results_df['rmse']
    
    all_results[country] = results_df
    
    avg_mape = results_df['mape'].mean()
    avg_mase = results_df['mase'].mean()
    print(f"\n   Complete: Avg MAPE={avg_mape:.2f}%, Avg MASE={avg_mase:.4f}, Refits={refit_count-1}")

# SAVE RESULTS

print("\n Saving results...")

os.makedirs('results/phase6_live_adaptation', exist_ok=True)

for country in COUNTRIES:
    filepath = f'results/phase6_live_adaptation/{country}_gru_live_simulation.csv'
    all_results[country].to_csv(filepath, index=False)
    print(f" Saved {country}_gru_live_simulation.csv")

# Summary JSON
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

with open('results/phase6_live_adaptation/gru_simulation_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(" Saved gru_simulation_summary.json")

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
    
    ax.set_title(f'{country} - GRU Performance Evolution (Avg MAPE: {summary[country]["avg_mape"]:.2f}%)', 
                fontsize=12, fontweight='bold')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('MAPE (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/phase6_live_adaptation/gru_performance_evolution.png', dpi=300, bbox_inches='tight')
print(" Saved gru_performance_evolution.png")
plt.close()

print("\n" + "="*80)
print("GRU LIVE MONITORING COMPLETE")
for country in COUNTRIES:
    print(f"{country}: MAPE {summary[country]['avg_mape']:.2f}% | MASE {summary[country]['avg_mase']:.4f} | Refits {summary[country]['num_refits']}")
