"""
Phase 4d: Vanilla RNN Forecasting and Backtesting
Implements 24-step rolling-window forecasting with comprehensive metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

import os
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.makedirs(os.path.join(base_dir, 'results/phase4d_rnn_de_fr_it_results'), exist_ok=True)

device = torch.device('cpu')
print(f"\nUsing device: {device}")

# VANILLA RNN MODEL ARCHITECTURE

class VanillaRNNForecaster(nn.Module):
    """Vanilla RNN for 24-hour ahead forecasting"""
    
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_size=24, dropout=0.2):
        super(VanillaRNNForecaster, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            nonlinearity='tanh'
        )
        
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
    def forward(self, x):
        x = x.unsqueeze(-1)
        rnn_out, _ = self.rnn(x)
        last_out = rnn_out[:, -1, :]
        out = self.fc1(last_out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# LOAD DATA
print("\n Loading dataset (FULL RUN)...")

import os
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
df = pd.read_csv(os.path.join(base_dir, 'time_series_60min_singleindex_filtered.csv'), index_col=0, parse_dates=True)

countries = ['DE', 'FR', 'IT']
load_columns = {
    'DE': 'DE_load_actual_entsoe_transparency',
    'FR': 'FR_load_actual_entsoe_transparency',
    'IT': 'IT_load_actual_entsoe_transparency'
}

data_splits = {}
for country, load_col in load_columns.items():
    data = df[load_col].dropna()
    
    hours_120_days = 120 * 24
    if len(data) > hours_120_days:
        data = data.iloc[-hours_120_days:]
    
    n = len(data)
    train_end = int(0.8 * n)
    dev_end = int(0.9 * n)
    
    train = data.iloc[:train_end]
    dev = data.iloc[train_end:dev_end]
    test = data.iloc[dev_end:]
    
    data_splits[country] = {
        'train': train,
        'dev': dev,
        'test': test,
        'full': data
    }
    
    print(f"{country}: Test set = {len(test)} hours")

# LOAD TRAINED MODELS
print("\n Loading trained Vanilla RNN models...")

trained_models = {}

for country in countries:
    model = VanillaRNNForecaster(
        input_size=1,
        hidden_size=128,
        num_layers=2,
        output_size=24,
        dropout=0.2
    ).to(device)
    
    checkpoint = torch.load(os.path.join(base_dir, f'results/phase3e_vanilla_rnn_results/{country}_vanilla_rnn_model.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    scaler = checkpoint['scaler']
    
    trained_models[country] = {
        'model': model,
        'scaler': scaler
    }
    
    print(f" Loaded model for {country}")

# FORECASTING FUNCTIONS

def create_sequences(data, seq_length=168):
    """Create input sequences for forecasting"""
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
    return np.array(sequences)

def rolling_forecast(model, scaler, data, seq_length=168, horizon=24):
    """Perform rolling-window forecasting"""
    model.eval()
    
    forecasts = []
    actuals = []
    timestamps = []
    
    # Normalize data
    data_scaled = scaler.transform(data.values.reshape(-1, 1)).flatten()
    
    # Rolling forecast
    for i in range(len(data) - seq_length - horizon + 1):
        # Input window
        input_seq = data_scaled[i:i+seq_length]
        
        # Get actual values
        actual = data.values[i+seq_length:i+seq_length+horizon]
        timestamp = data.index[i+seq_length:i+seq_length+horizon]
        
        # Forecast
        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).to(device)
            output = model(input_tensor)
            forecast = scaler.inverse_transform(output.cpu().numpy().reshape(-1, 1)).flatten()
        
        forecasts.extend(forecast)
        actuals.extend(actual)
        timestamps.extend(timestamp)
    
    return np.array(forecasts), np.array(actuals), timestamps

# METRICS

def calculate_mase(y_true, y_pred, y_train, seasonal_period=24):
    """Mean Absolute Scaled Error"""
    n = len(y_train)
    mae_naive = np.mean(np.abs(y_train[seasonal_period:] - y_train[:-seasonal_period]))
    mae = np.mean(np.abs(y_true - y_pred))
    return mae / mae_naive

def calculate_smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def calculate_mape(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    return 100 * np.mean(np.abs((y_true - y_pred) / y_true))

def calculate_mse(y_true, y_pred):
    """Mean Squared Error"""
    return np.mean((y_true - y_pred) ** 2)

def calculate_rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(calculate_mse(y_true, y_pred))

def calculate_all_metrics(y_true, y_pred, y_train):
    """Calculate all metrics"""
    return {
        'MASE': calculate_mase(y_true, y_pred, y_train, seasonal_period=24),
        'sMAPE_%': calculate_smape(y_true, y_pred),
        'MAPE_%': calculate_mape(y_true, y_pred),
        'RMSE_MW': calculate_rmse(y_true, y_pred),
        'MSE': calculate_mse(y_true, y_pred)
    }

# PERFORM FORECASTING
print("\n Performing rolling forecasts...")

all_results = {}
all_metrics = {}

for country in countries:
    print(f"\n  Processing {country}...")
    
    model_info = trained_models[country]
    test_data = data_splits[country]['test']
    train_data = data_splits[country]['train']
    
    forecasts, actuals, timestamps = rolling_forecast(
        model_info['model'],
        model_info['scaler'],
        test_data,
        seq_length=168,
        horizon=24
    )
    
    # Calculate metrics
    metrics = calculate_all_metrics(actuals, forecasts, train_data.values)
    all_metrics[country] = metrics
    
    # Store results
    all_results[country] = {
        'forecasts': forecasts,
        'actuals': actuals,
        'timestamps': timestamps
    }
    
    print(f"   MASE: {metrics['MASE']:.4f} | MAPE: {metrics['MAPE_%']:.2f}%")

# SAVE RESULTS
print("\n Saving forecast results...")

# Save metrics
metrics_df = pd.DataFrame(all_metrics).T
metrics_df.to_csv(os.path.join(base_dir, 'results/phase4d_rnn_de_fr_it_results/metrics_comparison.csv'))
print(f" Saved metrics to results/phase4d_rnn_de_fr_it_results/metrics_comparison.csv")

# Save individual country forecasts
for country in countries:
    result_df = pd.DataFrame({
        'timestamp': all_results[country]['timestamps'],
        'actual': all_results[country]['actuals'],
        'forecast': all_results[country]['forecasts']
    })
    result_df.to_csv(os.path.join(base_dir, f'results/phase4d_rnn_de_fr_it_results/forecast_data_{country}.csv'), index=False)
    print(f" Saved {country} forecasts")

# VISUALIZATIONS
print("\n Creating visualizations...")

# Forecast plots
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

for idx, country in enumerate(countries):
    ax = axes[idx]
    
    df_plot = pd.read_csv(os.path.join(base_dir, f'results/phase4d_rnn_de_fr_it_results/forecast_data_{country}.csv'), parse_dates=['timestamp'])
    df_plot = df_plot.head(240)  # First 10 days
    
    ax.plot(df_plot['timestamp'], df_plot['actual'], 'k-', label='Actual', linewidth=1.5)
    ax.plot(df_plot['timestamp'], df_plot['forecast'], 'r--', label='Vanilla RNN Forecast', linewidth=1.5, alpha=0.7)
    
    ax.set_title(f'{country} - Vanilla RNN 24-hour Forecast (MAPE: {all_metrics[country]["MAPE_%"]:.2f}%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Load (MW)')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'results/phase4d_rnn_de_fr_it_results/forecast_comparison.png'), dpi=300, bbox_inches='tight')
print(" Saved forecast_comparison.png")
plt.close()

# Metrics comparison
fig, ax = plt.subplots(figsize=(10, 6))

metrics_plot = metrics_df[['MASE', 'MAPE_%', 'sMAPE_%']].copy()
metrics_plot.plot(kind='bar', ax=ax, width=0.7)

ax.set_title('Vanilla RNN Model Performance Metrics by Country', fontsize=14, fontweight='bold')
ax.set_xlabel('Country', fontsize=12)
ax.set_ylabel('Metric Value', fontsize=12)
ax.legend(title='Metrics', loc='upper right')
ax.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'results/phase4d_rnn_de_fr_it_results/metrics_bar_chart.png'), dpi=300, bbox_inches='tight')
print(" Saved metrics_bar_chart.png")
plt.close()

# SUMMARY
print("\n" + "="*80)
print("\nPerformance Metrics:")
print(metrics_df.to_string())

print("\n All results saved to results/phase4d_rnn_de_fr_it_results/")
