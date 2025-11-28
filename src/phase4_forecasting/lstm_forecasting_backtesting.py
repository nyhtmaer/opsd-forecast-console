"""
Phase 4b: LSTM Forecasting and Backtesting with GPU/CUDA
Implements 24-step rolling-window forecasting with comprehensive metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
# PyTorch imports
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

os.makedirs('results/phase4b_lstm_de_fr_it_results', exist_ok=True)

# CHECK CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# LSTM MODEL ARCHITECTURE (same as Phase 3b)

class LSTMForecaster(nn.Module):
    """LSTM for 24-hour ahead forecasting"""
    
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_size=24, dropout=0.2):
        super(LSTMForecaster, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
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

# LOAD DATA
print("\n Loading dataset...")

# Load all preprocessed data for rolling window forecasting
df_train = pd.read_csv('data/preprocessed/train_data.csv', parse_dates=['utc_timestamp'])
df_train = df_train.set_index('utc_timestamp')
df_val = pd.read_csv('data/preprocessed/val_data.csv', parse_dates=['utc_timestamp'])
df_val = df_val.set_index('utc_timestamp')
df_test = pd.read_csv('data/preprocessed/test_data.csv', parse_dates=['utc_timestamp'])
df_test = df_test.set_index('utc_timestamp')

countries = ['DE', 'FR', 'IT']
load_columns = {
    'DE': 'DE_load_actual_entsoe_transparency',
    'FR': 'FR_load_actual_entsoe_transparency',
    'IT': 'IT_load_actual_entsoe_transparency'
}

# Prepare data splits using test set
data_splits = {}
for country, load_col in load_columns.items():
    train = df_train[load_col].dropna()
    val = df_val[load_col].dropna()
    test = df_test[load_col].dropna()
    
    # Combine train + val for full history
    full = pd.concat([train, val, test])
    
    data_splits[country] = {
        'train': train,
        'val': val,
        'test': test,
        'full': full
    }
    
    print(f"{country}: Train={len(train)} | Val={len(val)} | Test={len(test)} hours")

# LOAD TRAINED MODELS
print("\n Loading trained LSTM models...")

trained_models = {}

for country in countries:
    # Load model
    model = LSTMForecaster(
        input_size=1,
        hidden_size=128,
        num_layers=2,
        output_size=24,
        dropout=0.2
    ).to(device)
    
    checkpoint = torch.load(f'results/phase3b_lstm_results/lstm_model_{country}.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load scaler parameters
    scaler = StandardScaler()
    scaler.mean_ = np.array([checkpoint['scaler_mean']])
    scaler.scale_ = np.array([checkpoint['scaler_scale']])
    
    trained_models[country] = {
        'model': model,
        'scaler': scaler
    }
    
    print(f" Loaded model for {country}")

# FORECASTING FUNCTIONS

def lstm_forecast(model, scaler, history, device, lookback=168):
    """
    Generate 24-step forecast using LSTM model
    
    Args:
        model: Trained LSTM model
        scaler: Fitted StandardScaler
        history: Historical data (pandas Series)
        device: torch device (cuda/cpu)
        lookback: lookback period
    
    Returns:
        forecast: 24-hour forecast (original scale)
    """
    # Get last lookback hours
    history_values = history.values[-lookback:]
    
    # Normalize
    history_norm = scaler.transform(history_values.reshape(-1, 1)).flatten()
    
    # Convert to tensor
    x = torch.FloatTensor(history_norm).unsqueeze(0).to(device)
    
    # Forecast
    with torch.no_grad():
        forecast_norm = model(x).cpu().numpy().flatten()
    
    # Inverse transform
    forecast = scaler.inverse_transform(forecast_norm.reshape(-1, 1)).flatten()
    
    return forecast

def calculate_mase(actual, forecast, train_data):
    """Calculate Mean Absolute Scaled Error (MASE)"""
    # MAE of forecast
    mae_forecast = np.mean(np.abs(actual - forecast))
    
    # MAE of naive seasonal forecast (24-hour seasonality)
    naive_errors = np.abs(train_data.values[24:] - train_data.values[:-24])
    mae_naive = np.mean(naive_errors)
    
    # MASE
    mase = mae_forecast / mae_naive if mae_naive > 0 else np.inf
    return mase

def calculate_smape(actual, forecast):
    """Calculate Symmetric Mean Absolute Percentage Error (sMAPE)"""
    denominator = (np.abs(actual) + np.abs(forecast)) / 2
    smape = np.mean(np.abs(actual - forecast) / denominator) * 100
    return smape

def calculate_mape(actual, forecast):
    """Calculate Mean Absolute Percentage Error (MAPE)"""
    # Avoid division by zero
    mask = actual != 0
    if not np.any(mask):
        return np.inf
    mape = np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100
    return mape

# ROLLING-WINDOW FORECASTING
print("\n Running 24-step rolling-window forecasting...")

forecast_results = {}
lookback = 168  # 1 week

for country in countries:
    print(f"\n{'='*60}")
    print(f"Forecasting: {country}")
    print(f"{'='*60}")
    
    test_data = data_splits[country]['test']
    train_data = data_splits[country]['train']
    val_data = data_splits[country]['val']
    full_data = data_splits[country]['full']
    
    model = trained_models[country]['model']
    scaler = trained_models[country]['scaler']
    
    # Calculate the dev_end index (end of train + val)
    dev_end = len(train_data) + len(val_data)
    
    # Number of 24-hour windows in test set
    n_windows = len(test_data) // 24
    
    all_forecasts = []
    all_actuals = []
    
    print(f"Generating {n_windows} forecasts of 24 hours each...")
    
    for i in range(n_windows):
        # Forecast origin
        forecast_origin = dev_end + (i * 24)
        
        # Historical data up to forecast origin
        history = full_data.iloc[:forecast_origin]
        
        # Actual values for next 24 hours
        actual = test_data.iloc[i*24:(i+1)*24].values
        
        # Generate forecast
        forecast = lstm_forecast(model, scaler, history, device, lookback)
        
        all_forecasts.append(forecast)
        all_actuals.append(actual)
    
    # Combine all forecasts
    forecasts_array = np.concatenate(all_forecasts)
    actuals_array = np.concatenate(all_actuals)
    
    # Calculate metrics
    mase = calculate_mase(actuals_array, forecasts_array, train_data)
    smape = calculate_smape(actuals_array, forecasts_array)
    mape = calculate_mape(actuals_array, forecasts_array)
    rmse = np.sqrt(np.mean((actuals_array - forecasts_array)**2))
    mse = np.mean((actuals_array - forecasts_array)**2)
    
    forecast_results[country] = {
        'forecasts': forecasts_array,
        'actuals': actuals_array,
        'metrics': {
            'MASE': mase,
            'sMAPE': smape,
            'MAPE': mape,
            'RMSE': rmse,
            'MSE': mse,
            'n_forecasts': n_windows
        }
    }
    
    print(f" Generated {len(forecasts_array)} hourly forecasts")
    print(f"  MASE: {mase:.4f}")
    print(f"  sMAPE: {smape:.2f}%")
    print(f"  RMSE: {rmse:.2f} MW")

# SAVE METRICS
print("\n Saving metrics...")

# Summary metrics
metrics_summary = {
    country: {
        'MASE': round(forecast_results[country]['metrics']['MASE'], 4),
        'sMAPE_%': round(forecast_results[country]['metrics']['sMAPE'], 2),
        'MAPE_%': round(forecast_results[country]['metrics']['MAPE'], 2),
        'RMSE_MW': round(forecast_results[country]['metrics']['RMSE'], 2),
        'MSE': round(forecast_results[country]['metrics']['MSE'], 2),
        'n_forecasts': forecast_results[country]['metrics']['n_forecasts']
    }
    for country in countries
}

with open('results/phase4b_lstm_de_fr_it_results/metrics_summary.json', 'w') as f:
    json.dump(metrics_summary, f, indent=2)
print(" Saved: results/phase4b_lstm_de_fr_it_results/metrics_summary.json")

# Comparison CSV
metrics_df = pd.DataFrame(metrics_summary).T
metrics_df.to_csv('results/phase4b_lstm_de_fr_it_results/metrics_comparison.csv')
print(" Saved: results/phase4b_lstm_de_fr_it_results/metrics_comparison.csv")

# VISUALIZATIONS
print("\n Creating visualizations...")

for country in countries:
    test_data = data_splits[country]['test']
    forecasts = forecast_results[country]['forecasts']
    actuals = forecast_results[country]['actuals']
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Full test set
    ax = axes[0]
    test_index = test_data.index[:len(actuals)]
    ax.plot(test_index, actuals, label='Actual', linewidth=1.5, alpha=0.8)
    ax.plot(test_index, forecasts, label='LSTM Forecast', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Load (MW)', fontsize=11)
    ax.set_title(f'{country}: LSTM 24-Step Forecasting - Full Test Set', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 7-day zoom
    ax = axes[1]
    zoom_hours = 7 * 24
    zoom_index = test_index[:zoom_hours]
    ax.plot(zoom_index, actuals[:zoom_hours], label='Actual', linewidth=2, alpha=0.8, marker='o', markersize=3)
    ax.plot(zoom_index, forecasts[:zoom_hours], label='LSTM Forecast', linewidth=2, alpha=0.8, marker='s', markersize=3)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Load (MW)', fontsize=11)
    ax.set_title(f'{country}: LSTM 24-Step Forecasting - First 7 Days (Detail)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/phase4b_lstm_de_fr_it_results/forecast_{country}.png', dpi=300, bbox_inches='tight')
    print(f" Saved: results/phase4b_lstm_de_fr_it_results/forecast_{country}.png")
    plt.close()

# SAVE FORECAST DATA
print("\n Saving forecast data...")

for country in countries:
    test_data = data_splits[country]['test']
    forecasts = forecast_results[country]['forecasts']
    actuals = forecast_results[country]['actuals']
    
    forecast_df = pd.DataFrame({
        'timestamp': test_data.index[:len(actuals)],
        'actual': actuals,
        'forecast': forecasts,
        'error': actuals - forecasts,
        'abs_error': np.abs(actuals - forecasts),
        'pct_error': ((actuals - forecasts) / actuals) * 100
    })
    
    forecast_df.to_csv(f'results/phase4b_lstm_de_fr_it_results/forecast_data_{country}.csv', index=False)
    print(f" Saved: results/phase4b_lstm_de_fr_it_results/forecast_data_{country}.csv")

# COMPARISON WITH SARIMA
print("\n" + "="*80)
print("\nCOMPARING LSTM vs SARIMA PERFORMANCE")
# Load SARIMA metrics if available
try:
    with open('results/phase4_results/metrics_summary.json', 'r') as f:
        sarima_metrics = json.load(f)
    
    # Create comparison table
    comparison_data = []
    for country in countries:
        lstm = metrics_summary[country]
        sarima = sarima_metrics.get(country, None)
        
        comparison_data.append({
            'Country': country,
            'Model': 'LSTM',
            'MASE': lstm['MASE'],
            'sMAPE_%': lstm['sMAPE_%'],
            'RMSE_MW': lstm['RMSE_MW']
        })
        
        if sarima:
            comparison_data.append({
                'Country': country,
                'Model': 'SARIMA',
                'MASE': sarima['MASE'],
                'sMAPE_%': sarima['sMAPE'],
                'RMSE_MW': sarima['RMSE']
            })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv('results/phase4b_lstm_de_fr_it_results/lstm_vs_sarima_comparison.csv', index=False)
        print("Saved: results/phase4b_lstm_de_fr_it_results/lstm_vs_sarima_comparison.csv")
    else:
        print("[INFO] No SARIMA results available for comparison yet")
except FileNotFoundError:
    print("[INFO] SARIMA results not available yet - comparison skipped")
except Exception as e:
    print(f"[WARNING] Could not compare with SARIMA: {e}")

print("\n" + "="*80)
print("\nLSTM FORECASTING COMPLETED")
print("="*80)
print(f"\nDevice Used: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("\nGenerated Files:")
print("  1. results/phase4b_lstm_de_fr_it_results/metrics_summary.json")
print("  2. results/phase4b_lstm_de_fr_it_results/metrics_comparison.csv")
print("  3. results/phase4b_lstm_de_fr_it_results/forecast_DE.png")
print("  4. results/phase4b_lstm_de_fr_it_results/forecast_FR.png")
print("  5. results/phase4b_lstm_de_fr_it_results/forecast_IT.png")
print("  6. results/phase4b_lstm_de_fr_it_results/forecast_data_DE.csv")
print("  7. results/phase4b_lstm_de_fr_it_results/forecast_data_FR.csv")
print("  8. results/phase4b_lstm_de_fr_it_results/forecast_data_IT.csv")
print("  9. results/phase4b_lstm_de_fr_it_results/lstm_vs_sarima_comparison.csv (if SARIMA available)")

print("\n" + "="*80)
print("LSTM forecasting complete with SARIMA comparison!")
