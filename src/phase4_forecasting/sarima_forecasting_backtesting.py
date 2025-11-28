"""
 Day-Ahead 24-Step Forecasting with Backtesting
Implements rolling-window forecasting and computes evaluation metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
import json
import os
os.makedirs('results/phase4_sarima_de_fr_it_results', exist_ok=True)

print(" DAY-AHEAD 24-STEP FORECASTING WITH BACKTESTING")
# LOAD DATA AND SELECTED MODELS
print("\n Loading data and selected models...")

# Load preprocessed datasets
train_df = pd.read_csv('data/preprocessed/train_data.csv', parse_dates=['utc_timestamp'])
train_df = train_df.set_index('utc_timestamp')

val_df = pd.read_csv('data/preprocessed/val_data.csv', parse_dates=['utc_timestamp'])
val_df = val_df.set_index('utc_timestamp')

test_df = pd.read_csv('data/preprocessed/test_data.csv', parse_dates=['utc_timestamp'])
test_df = test_df.set_index('utc_timestamp')

# Combine for rolling forecast
df = pd.concat([train_df, val_df, test_df])

# Load selected models
with open('phase3_results/model_selection_summary.json', 'r') as f:
    model_summary = json.load(f)

countries = ['DE', 'FR', 'IT']
load_columns = {
    'DE': 'DE_load_actual_entsoe_transparency',
    'FR': 'FR_load_actual_entsoe_transparency',
    'IT': 'IT_load_actual_entsoe_transparency'
}

# Get selected models
selected_models = {}
for country in countries:
    model_info = model_summary['selected_models'][country]['selected_by_BIC']
    selected_models[country] = {
        'order': tuple(model_info['order']),
        'seasonal_order': tuple(model_info['seasonal_order'])
    }
    print(f"{country}: SARIMA{selected_models[country]['order']} x {selected_models[country]['seasonal_order']}")

# DATA PREPARATION (Same 120-day window as Phase 3)
print("\n Preparing data with 120-day window...")

data_splits = {}
for country, load_col in load_columns.items():
    # Get clean data
    data = df[load_col].dropna()
    
    # Use only last 120 days (same as Phase 3)
    hours_120_days = 120 * 24
    if len(data) > hours_120_days:
        data = data.iloc[-hours_120_days:]
    
    # Calculate split indices (80% train, 10% dev, 10% test)
    n = len(data)
    train_end = int(0.8 * n)
    dev_end = int(0.9 * n)
    
    # Split data
    train = data.iloc[:train_end]
    dev = data.iloc[train_end:dev_end]
    test = data.iloc[dev_end:]
    
    data_splits[country] = {
        'train': train,
        'dev': dev,
        'test': test
    }
    
    print(f"{country}: Train={len(train)} | Dev={len(dev)} | Test={len(test)} hours")

# METRIC FUNCTIONS

def calculate_mase(actual, forecast, train_data, seasonal_period=24):
    """
    Calculate Mean Absolute Scaled Error (MASE)
    MASE < 1: forecast is better than naive seasonal forecast
    """
    # Calculate naive forecast error (seasonal naive)
    # Correct: compare values that are seasonal_period apart
    naive_errors = np.abs(train_data.values[seasonal_period:] - train_data.values[:-seasonal_period])
    scale = np.mean(naive_errors)
    
    # Calculate forecast errors
    errors = np.abs(actual - forecast)
    mase = np.mean(errors) / scale
    
    return mase

def calculate_smape(actual, forecast):
    """Calculate Symmetric Mean Absolute Percentage Error"""
    denominator = (np.abs(actual) + np.abs(forecast)) / 2
    smape = np.mean(np.abs(actual - forecast) / denominator) * 100
    return smape

def calculate_mape(actual, forecast):
    """Calculate Mean Absolute Percentage Error"""
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    return mape

def calculate_rmse(actual, forecast):
    """Calculate Root Mean Squared Error"""
    return np.sqrt(np.mean((actual - forecast) ** 2))

def calculate_mse(actual, forecast):
    """Calculate Mean Squared Error"""
    return np.mean((actual - forecast) ** 2)

def calculate_pi_coverage(actual, lower_bound, upper_bound):
    """Calculate prediction interval coverage"""
    within_interval = ((actual >= lower_bound) & (actual <= upper_bound)).sum()
    coverage = within_interval / len(actual) * 100
    return coverage

# ROLLING-WINDOW BACKTESTING
print("\n Performing rolling-window 24-step forecasting on test set...")

forecast_results = {}

for country in countries:
    print(f"\n{'='*60}")
    print(f"Backtesting: {country}")
    print(f"{'='*60}")
    
    train_data = data_splits[country]['train']
    test_data = data_splits[country]['test']
    
    order = selected_models[country]['order']
    seasonal_order = selected_models[country]['seasonal_order']
    
    # Storage for forecasts
    all_forecasts = []
    all_lower = []
    all_upper = []
    all_actual = []
    all_timestamps = []
    
    # Rolling window: forecast 24 steps ahead, then move window forward
    test_length = len(test_data)
    n_forecasts = test_length // 24  # Number of 24-step forecasts
    
    print(f"Generating {n_forecasts} rolling 24-step forecasts...")
    
    for i in range(n_forecasts):
        # Determine forecast origin
        forecast_origin = i * 24
        
        # Prepare training data up to forecast origin
        if forecast_origin == 0:
            fit_data = train_data
        else:
            fit_data = pd.concat([train_data, test_data.iloc[:forecast_origin]])
        
        # Fit model
        try:
            model = SARIMAX(
                fit_data,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            fitted_model = model.fit(disp=False, maxiter=100)
            
            # Generate 24-step forecast with 80% prediction intervals
            forecast_result = fitted_model.get_forecast(steps=24)
            forecast = forecast_result.predicted_mean
            forecast_ci = forecast_result.conf_int(alpha=0.20)  # 80% PI
            
            # Get actual values for this forecast window
            actual_window = test_data.iloc[forecast_origin:forecast_origin+24]
            
            # Store results
            all_forecasts.extend(forecast.values)
            all_lower.extend(forecast_ci.iloc[:, 0].values)
            all_upper.extend(forecast_ci.iloc[:, 1].values)
            all_actual.extend(actual_window.values)
            all_timestamps.extend(actual_window.index)
            
            if (i + 1) % 5 == 0:
                print(f"  Completed {i+1}/{n_forecasts} forecasts...")
                
        except Exception as e:
            print(f"  Warning: Forecast {i+1} failed: {str(e)}")
            # Use previous forecast or skip
            continue
    
    # Convert to arrays
    forecasts = np.array(all_forecasts)
    actual = np.array(all_actual)
    lower_bound = np.array(all_lower)
    upper_bound = np.array(all_upper)
    timestamps = pd.DatetimeIndex(all_timestamps)
    
    # Store results
    forecast_results[country] = {
        'timestamps': timestamps,
        'actual': actual,
        'forecast': forecasts,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'train_data': train_data
    }
    
    print(f" Generated {len(forecasts)} hourly forecasts")

# CALCULATE METRICS
print("\n Computing evaluation metrics...")

metrics_summary = {}

for country in countries:
    actual = forecast_results[country]['actual']
    forecast = forecast_results[country]['forecast']
    lower_bound = forecast_results[country]['lower_bound']
    upper_bound = forecast_results[country]['upper_bound']
    train_data = forecast_results[country]['train_data']
    
    # Calculate all metrics
    mase = calculate_mase(actual, forecast, train_data, seasonal_period=24)
    smape = calculate_smape(actual, forecast)
    mape = calculate_mape(actual, forecast)
    rmse = calculate_rmse(actual, forecast)
    mse = calculate_mse(actual, forecast)
    pi_coverage = calculate_pi_coverage(actual, lower_bound, upper_bound)
    
    metrics_summary[country] = {
        'MASE': round(mase, 4),
        'sMAPE': round(smape, 2),
        'MAPE': round(mape, 2),
        'RMSE': round(rmse, 2),
        'MSE': round(mse, 2),
        'PI_Coverage_80%': round(pi_coverage, 2)
    }
    
    print(f"\n{country} Metrics:")
    print(f"  MASE: {mase:.4f} {'( Better than naive)' if mase < 1 else '(ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â  Worse than naive)'}")
    print(f"  sMAPE: {smape:.2f}%")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MSE: {mse:.2f}")
    print(f"  80% PI Coverage: {pi_coverage:.2f}% {'( Good)' if 75 <= pi_coverage <= 85 else '(ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â )'}")

# VISUALIZE FORECASTS
print("\n Generating forecast visualizations...")

for country in countries:
    timestamps = forecast_results[country]['timestamps']
    actual = forecast_results[country]['actual']
    forecast = forecast_results[country]['forecast']
    lower_bound = forecast_results[country]['lower_bound']
    upper_bound = forecast_results[country]['upper_bound']
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle(f'24-Step Day-Ahead Forecasting: {country}', fontsize=16, fontweight='bold')
    
    # Plot 1: Full test set forecast
    axes[0].plot(timestamps, actual, label='Actual', color='black', linewidth=1.5, alpha=0.8)
    axes[0].plot(timestamps, forecast, label='Forecast', color='red', linewidth=1.2)
    axes[0].fill_between(timestamps, lower_bound, upper_bound, 
                         alpha=0.2, color='red', label='80% Prediction Interval')
    axes[0].set_xlabel('Time', fontsize=11)
    axes[0].set_ylabel('Load (MW)', fontsize=11)
    axes[0].set_title('Full Test Set Forecast', fontsize=12, fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Zoom into first 7 days (168 hours)
    zoom_hours = min(168, len(timestamps))
    axes[1].plot(timestamps[:zoom_hours], actual[:zoom_hours], 
                label='Actual', color='black', linewidth=2, alpha=0.8, marker='o', markersize=3)
    axes[1].plot(timestamps[:zoom_hours], forecast[:zoom_hours], 
                label='Forecast', color='red', linewidth=1.5, marker='x', markersize=3)
    axes[1].fill_between(timestamps[:zoom_hours], lower_bound[:zoom_hours], upper_bound[:zoom_hours],
                         alpha=0.2, color='red', label='80% Prediction Interval')
    axes[1].set_xlabel('Time', fontsize=11)
    axes[1].set_ylabel('Load (MW)', fontsize=11)
    axes[1].set_title('First 7 Days Detail', fontsize=12, fontweight='bold')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/phase4_sarima_de_fr_it_results/forecast_{country}.png', dpi=300, bbox_inches='tight')
    print(f" Saved: results/phase4_sarima_de_fr_it_results/forecast_{country}.png")
    plt.close()

# SAVE RESULTS
print("\n Saving forecast results and metrics...")

# Save metrics summary
with open('results/phase4_sarima_de_fr_it_results/metrics_summary.json', 'w') as f:
    json.dump(metrics_summary, f, indent=2)
print("Saved: results/phase4_sarima_de_fr_it_results/metrics_summary.json")

# Create metrics comparison table
metrics_df = pd.DataFrame(metrics_summary).T
metrics_df.to_csv('results/phase4_sarima_de_fr_it_results/metrics_comparison.csv')
print(" Saved: results/phase4_sarima_de_fr_it_results/metrics_comparison.csv")

# Save forecast data for each country
for country in countries:
    forecast_df = pd.DataFrame({
        'timestamp': forecast_results[country]['timestamps'],
        'actual': forecast_results[country]['actual'],
        'forecast': forecast_results[country]['forecast'],
        'lower_bound_80%': forecast_results[country]['lower_bound'],
        'upper_bound_80%': forecast_results[country]['upper_bound'],
        'error': forecast_results[country]['actual'] - forecast_results[country]['forecast']
    })
    forecast_df.to_csv(f'results/phase4_sarima_de_fr_it_results/forecast_data_{country}.csv', index=False)
    print(f" Saved: results/phase4_sarima_de_fr_it_results/forecast_data_{country}.csv")

# Print final summary
print("\n" + "="*80)
print("METRICS COMPARISON (PRIMARY: MASE)")
print(metrics_df.to_string())

print("\n" + "="*80)
print("\nGenerated Files:")
print("  1. results/phase4_sarima_de_fr_it_results/forecast_AT.png")
print("  2. results/phase4_sarima_de_fr_it_results/forecast_BE.png")
print("  3. results/phase4_sarima_de_fr_it_results/forecast_BG.png")
print("  4. results/phase4_sarima_de_fr_it_results/metrics_summary.json")
print("  5. results/phase4_sarima_de_fr_it_results/metrics_comparison.csv")
print("  6. results/phase4_sarima_de_fr_it_results/forecast_data_AT.csv")
print("  7. results/phase4_sarima_de_fr_it_results/forecast_data_BE.csv")
print("  8. results/phase4_sarima_de_fr_it_results/forecast_data_BG.csv")
print("\nDay-ahead forecasting and backtesting completed successfully!")
