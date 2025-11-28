"""
 Live Monitoring & Online Adaptation Simulation
Simulates a live data feed with rolling SARIMA refit strategy

Assignment Requirements:
- Simulate 2,000 hours of streaming data
- Implement ONE online adaptation strategy (Rolling SARIMA refit chosen)
- Track performance metrics over time
- Detect performance drift
- Log adaptation events

Strategy: Rolling SARIMA Refit
- Refit SARIMA model every 168 hours (1 week)
- Use expanding window (minimum 60 days history)
- Generate 24-step ahead forecasts hour-by-hour
- Monitor MASE and MAPE drift
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
import json
import os
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

os.makedirs('results/phase6_live_adaptation', exist_ok=True)

print(" LIVE MONITORING & ONLINE ADAPTATION SIMULATION")
print("\nStrategy: Rolling SARIMA Refit")
print("  - Refit frequency: Every 168 hours (1 week)")
print("  - Minimum history: 60 days (1,440 hours)")
print("  - Forecast horizon: 24 hours ahead")
print("  - Simulation period: 2,000 hours")

# CONFIGURATION

COUNTRIES = ['DE', 'FR', 'IT']
SELECTED_MODELS = {
    'DE': {'order': (2, 0, 1), 'seasonal_order': (1, 1, 1, 24)},
    'FR': {'order': (2, 0, 1), 'seasonal_order': (1, 1, 1, 24)},
    'IT': {'order': (2, 0, 1), 'seasonal_order': (1, 1, 1, 24)}
}

# Load columns
LOAD_COLUMNS = {
    'DE': 'DE_load_actual_entsoe_transparency',
    'FR': 'FR_load_actual_entsoe_transparency',
    'IT': 'IT_load_actual_entsoe_transparency'
}

# Simulation parameters
REFIT_FREQUENCY = 336  # Refit every 336 hours (2 weeks) - faster execution
MIN_HISTORY = 1440     # Minimum 60 days (1,440 hours) of history
FORECAST_HORIZON = 24  # 24-hour ahead forecasting
SIMULATION_HOURS = 3500  # Simulate 3,500 hours (146 days) - 75% above minimum

# METRIC FUNCTIONS

def calculate_mase(actual, forecast, train_data, seasonal_period=24):
    """Calculate Mean Absolute Scaled Error"""
    if len(train_data) <= seasonal_period:
        return np.nan
    
    naive_errors = np.abs(train_data.values[seasonal_period:] - 
                          train_data.values[:-seasonal_period])
    scale = np.mean(naive_errors)
    
    if scale == 0:
        return np.nan
    
    errors = np.abs(actual - forecast)
    mase = np.mean(errors) / scale
    return mase

def calculate_mape(actual, forecast):
    """Calculate Mean Absolute Percentage Error"""
    return np.mean(np.abs((actual - forecast) / actual)) * 100

def calculate_rmse(actual, forecast):
    """Calculate Root Mean Squared Error"""
    return np.sqrt(np.mean((actual - forecast) ** 2))

# LOAD DATA

print("\n Loading dataset...")

df = pd.read_csv('data/time_series_60min_singleindex.csv', index_col=0, parse_dates=True)

print(f"Dataset: {len(df):,} hours ({df.index[0]} to {df.index[-1]})")

# PREPARE SIMULATION DATA

print("\n Preparing simulation data...")
print(f"  - Simulation period: {SIMULATION_HOURS:,} hours (ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â°ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¥2,000 required)")
print(f"  - Refit frequency: Every {REFIT_FREQUENCY} hours")
print(f"  - Minimum history: {MIN_HISTORY} hours ({MIN_HISTORY/24:.0f} days)")

# Use data BEFORE the 120-day window we used for training
# We used last 2,880 hours (120 days), so use the 2,400 hours before that
# This simulates "live" data that the model hasn't seen

simulation_data = {}

for country, load_col in LOAD_COLUMNS.items():
    # Get clean data
    data = df[load_col].dropna()
    
    # Skip the last 2,880 hours (our training window)
    # Use 2,400 hours before that for live simulation
    end_idx = len(data) - 2880
    start_idx = end_idx - SIMULATION_HOURS
    
    # Ensure we have enough history before simulation start
    history_start_idx = start_idx - MIN_HISTORY
    
    if history_start_idx < 0:
        print(f"ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¯ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¸ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â  Warning: Insufficient history for {country}. Adjusting simulation start.")
        history_start_idx = 0
        start_idx = MIN_HISTORY
        end_idx = start_idx + SIMULATION_HOURS
    
    # Extract simulation period + initial history
    full_data = data.iloc[history_start_idx:end_idx]
    
    simulation_data[country] = {
        'full_data': full_data,
        'simulation_start': start_idx - history_start_idx,
        'simulation_end': end_idx - history_start_idx,
        'initial_history': full_data.iloc[:MIN_HISTORY]
    }
    
    sim_period_start = full_data.index[start_idx - history_start_idx]
    sim_period_end = full_data.index[-1]
    
    print(f"\n{country}:")
    print(f"  Initial history: {MIN_HISTORY} hours ({full_data.index[0]} to {full_data.index[MIN_HISTORY-1]})")
    print(f"  Simulation period: {SIMULATION_HOURS} hours ({sim_period_start} to {sim_period_end})")

# LIVE SIMULATION WITH ROLLING SARIMA REFIT

print("\n Running live simulation with rolling SARIMA refit...")

simulation_results = {}

for country in COUNTRIES:
    print(f"\n{'='*60}")
    print(f"Simulating: {country}")
    print(f"{'='*60}")
    
    order = SELECTED_MODELS[country]['order']
    seasonal_order = SELECTED_MODELS[country]['seasonal_order']
    
    full_data = simulation_data[country]['full_data']
    sim_start = simulation_data[country]['simulation_start']
    sim_end = simulation_data[country]['simulation_end']
    
    # Storage for results
    timestamps = []
    actuals = []
    forecasts_1h = []  # 1-hour ahead
    forecasts_24h = []  # 24-hour ahead
    mase_values = []
    mape_values = []
    rmse_values = []
    refit_flags = []
    model_age = []  # Hours since last refit
    
    # Initial model training
    initial_train_data = full_data.iloc[:sim_start]
    
    print(f"Training initial model on {len(initial_train_data)} hours...")
    
    try:
        current_model = SARIMAX(
            initial_train_data,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fitted_model = current_model.fit(disp=False, maxiter=50, method='nm')  # Faster optimizer
        hours_since_refit = 0
        print(" Initial model trained")
    except Exception as e:
        print(f"ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚ÂÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ Error training initial model: {e}")
        continue
    
    # Simulate hour-by-hour
    print(f"Simulating {SIMULATION_HOURS} hours hour-by-hour...")
    
    for t in range(sim_start, sim_end):
        # Current timestamp
        current_time = full_data.index[t]
        actual_value = full_data.iloc[t]
        
        # Check if we need to refit
        refit_this_hour = (hours_since_refit >= REFIT_FREQUENCY)
        
        if refit_this_hour:
            # Refit model with expanding window
            train_data = full_data.iloc[:t]
            
            try:
                current_model = SARIMAX(
                    train_data,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                fitted_model = current_model.fit(disp=False, maxiter=50, method='nm')
                hours_since_refit = 0
                print(f"  Hour {t-sim_start:4d}/{SIMULATION_HOURS}:  Model refit (training size: {len(train_data)} hours)")
            except:
                # If refit fails, keep using old model
                refit_this_hour = False
                print(f"  Hour {t-sim_start:4d}/{SIMULATION_HOURS}: ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â  Refit failed, using existing model")
        
        # Generate forecasts
        try:
            forecast_result = fitted_model.get_forecast(steps=24)
            forecast_values = forecast_result.predicted_mean
            
            forecast_1h = forecast_values.iloc[0]  # 1-hour ahead
            forecast_24h = forecast_values.iloc[-1] if len(forecast_values) >= 24 else np.nan  # 24-hour ahead
        except Exception as e:
            forecast_1h = np.nan
            forecast_24h = np.nan
        
        # Calculate metrics (using last 24 hours of forecasts vs actuals)
        if len(forecasts_1h) >= 24:
            recent_actuals = np.array(actuals[-24:])
            recent_forecasts = np.array(forecasts_1h[-24:])
            
            # Use larger training window for MASE calculation
            train_for_mase = full_data.iloc[max(0, t-168):t]  # Last week
            
            mase = calculate_mase(recent_actuals, recent_forecasts, train_for_mase)
            mape = calculate_mape(recent_actuals, recent_forecasts)
            rmse = calculate_rmse(recent_actuals, recent_forecasts)
        else:
            mase = np.nan
            mape = np.nan
            rmse = np.nan
        
        # Store results
        timestamps.append(current_time)
        actuals.append(actual_value)
        forecasts_1h.append(forecast_1h)
        forecasts_24h.append(forecast_24h)
        mase_values.append(mase)
        mape_values.append(mape)
        rmse_values.append(rmse)
        refit_flags.append(1 if refit_this_hour else 0)
        model_age.append(hours_since_refit)
        
        hours_since_refit += 1
        
        # Progress update
        if (t - sim_start + 1) % 200 == 0:
            n_refits = sum(refit_flags)
            avg_mape = np.nanmean([m for m in mape_values[-100:] if not np.isnan(m)])
            print(f"  Hour {t-sim_start+1:4d}/{SIMULATION_HOURS}: MAPE={avg_mape:.2f}%, Refits={n_refits}, Model age={hours_since_refit}h")
    
    # Save results
    results_df = pd.DataFrame({
        'timestamp': timestamps,
        'actual': actuals,
        'forecast_1h': forecasts_1h,
        'forecast_24h': forecasts_24h,
        'mase_rolling_24h': mase_values,
        'mape_rolling_24h': mape_values,
        'rmse_rolling_24h': rmse_values,
        'refit_flag': refit_flags,
        'model_age_hours': model_age
    })
    
    simulation_results[country] = results_df
    
    # Calculate overall statistics
    valid_mase = results_df['mase_rolling_24h'].dropna()
    valid_mape = results_df['mape_rolling_24h'].dropna()
    valid_rmse = results_df['rmse_rolling_24h'].dropna()
    n_refits = results_df['refit_flag'].sum()
    
    print(f"\n Simulation complete for {country}")
    print(f"  Total hours: {len(results_df)}")
    print(f"  Model refits: {n_refits}")
    print(f"  Avg MASE: {valid_mase.mean():.4f}")
    print(f"  Avg MAPE: {valid_mape.mean():.2f}%")
    print(f"  Avg RMSE: {valid_rmse.mean():.2f} MW")

# SAVE RESULTS

print("\n Saving simulation results...")

for country, results_df in simulation_results.items():
    output_file = f'results/phase6_live_adaptation/{country}_live_simulation.csv'
    results_df.to_csv(output_file, index=False)
    print(f" Saved: {output_file}")

# Save summary statistics
summary_stats = {}

for country, results_df in simulation_results.items():
    valid_mase = results_df['mase_rolling_24h'].dropna()
    valid_mape = results_df['mape_rolling_24h'].dropna()
    valid_rmse = results_df['rmse_rolling_24h'].dropna()
    
    summary_stats[country] = {
        'total_hours': len(results_df),
        'n_refits': int(results_df['refit_flag'].sum()),
        'refit_frequency_hours': REFIT_FREQUENCY,
        'avg_mase': float(valid_mase.mean()),
        'std_mase': float(valid_mase.std()),
        'avg_mape_%': float(valid_mape.mean()),
        'std_mape_%': float(valid_mape.std()),
        'avg_rmse_MW': float(valid_rmse.mean()),
        'std_rmse_MW': float(valid_rmse.std()),
        'min_mape_%': float(valid_mape.min()),
        'max_mape_%': float(valid_mape.max())
    }

with open('results/phase6_live_adaptation/simulation_summary.json', 'w') as f:
    json.dump(summary_stats, f, indent=2)

print(f" Saved: results/phase6_live_adaptation/simulation_summary.json")

# VISUALIZATIONS

print("\n Creating visualizations...")

for country, results_df in simulation_results.items():
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    fig.suptitle(f'{country} - Live Monitoring Simulation Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Actual vs Forecast
    ax1 = axes[0]
    ax1.plot(results_df['timestamp'], results_df['actual'], 
             label='Actual Load', color='black', alpha=0.7, linewidth=0.8)
    ax1.plot(results_df['timestamp'], results_df['forecast_1h'], 
             label='1-hour Ahead Forecast', color='blue', alpha=0.6, linewidth=0.8)
    
    # Mark refits
    refit_times = results_df[results_df['refit_flag'] == 1]['timestamp']
    if len(refit_times) > 0:
        ax1.scatter(refit_times, 
                   results_df[results_df['refit_flag'] == 1]['actual'],
                   color='red', s=30, marker='v', label='Model Refit', zorder=5)
    
    ax1.set_ylabel('Load (MW)', fontsize=11)
    ax1.set_title('Actual Load vs 1-hour Ahead Forecast', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: MAPE Evolution
    ax2 = axes[1]
    ax2.plot(results_df['timestamp'], results_df['mape_rolling_24h'], 
             color='orangered', linewidth=1, alpha=0.8)
    ax2.axhline(y=results_df['mape_rolling_24h'].mean(), 
               color='red', linestyle='--', linewidth=1, label=f'Mean: {results_df["mape_rolling_24h"].mean():.2f}%')
    ax2.set_ylabel('MAPE (%)', fontsize=11)
    ax2.set_title('Rolling 24-hour MAPE Evolution', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: MASE Evolution
    ax3 = axes[2]
    ax3.plot(results_df['timestamp'], results_df['mase_rolling_24h'], 
             color='steelblue', linewidth=1, alpha=0.8)
    ax3.axhline(y=results_df['mase_rolling_24h'].mean(), 
               color='blue', linestyle='--', linewidth=1, label=f'Mean: {results_df["mase_rolling_24h"].mean():.4f}')
    ax3.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, label='Baseline (MASE=1.0)')
    ax3.set_ylabel('MASE', fontsize=11)
    ax3.set_title('Rolling 24-hour MASE Evolution', fontsize=12)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Model Age
    ax4 = axes[3]
    ax4.fill_between(results_df['timestamp'], 0, results_df['model_age_hours'],
                     color='lightgreen', alpha=0.5)
    ax4.axhline(y=REFIT_FREQUENCY, color='red', linestyle='--', 
               linewidth=1, label=f'Refit Threshold ({REFIT_FREQUENCY}h)')
    ax4.set_ylabel('Hours', fontsize=11)
    ax4.set_xlabel('Timestamp', fontsize=11)
    ax4.set_title('Model Age (Hours Since Last Refit)', fontsize=12)
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/phase6_live_adaptation/{country}_live_monitoring.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" Saved: results/phase6_live_adaptation/{country}_live_monitoring.png")

# Create comparison plot across countries
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
fig.suptitle('Live Monitoring Performance Comparison (All Countries)', 
             fontsize=16, fontweight='bold')

# MAPE comparison
ax1 = axes[0]
for country, results_df in simulation_results.items():
    ax1.plot(results_df['timestamp'], results_df['mape_rolling_24h'], 
             label=country, linewidth=1, alpha=0.8)
ax1.set_ylabel('MAPE (%)', fontsize=11)
ax1.set_title('Rolling 24-hour MAPE Evolution', fontsize=12)
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# MASE comparison
ax2 = axes[1]
for country, results_df in simulation_results.items():
    ax2.plot(results_df['timestamp'], results_df['mase_rolling_24h'], 
             label=country, linewidth=1, alpha=0.8)
ax2.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, label='Baseline (MASE=1.0)')
ax2.set_ylabel('MASE', fontsize=11)
ax2.set_xlabel('Timestamp', fontsize=11)
ax2.set_title('Rolling 24-hour MASE Evolution', fontsize=12)
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/phase6_live_adaptation/comparison_all_countries.png', 
            dpi=300, bbox_inches='tight')
plt.close()

print(f" Saved: results/phase6_live_adaptation/comparison_all_countries.png")

# SUMMARY REPORT

print("\n Generating summary report...")

print("\n" + "="*80)
print("LIVE MONITORING SIMULATION SUMMARY")
print(f"\nConfiguration:")
print(f"  - Simulation period: {SIMULATION_HOURS} hours ({SIMULATION_HOURS/24:.0f} days)")
print(f"  - Refit strategy: Rolling SARIMA with expanding window")
print(f"  - Refit frequency: Every {REFIT_FREQUENCY} hours ({REFIT_FREQUENCY/24:.0f} days)")
print(f"  - Minimum history: {MIN_HISTORY} hours ({MIN_HISTORY/24:.0f} days)")

print(f"\nResults Summary:")
print(f"{'Country':<10} {'Hours':<8} {'Refits':<8} {'Avg MASE':<12} {'Avg MAPE':<12} {'Avg RMSE':<12}")
print("-" * 70)

for country in COUNTRIES:
    stats = summary_stats[country]
    print(f"{country:<10} {stats['total_hours']:<8} {stats['n_refits']:<8} "
          f"{stats['avg_mase']:<12.4f} {stats['avg_mape_%']:<12.2f} {stats['avg_rmse_MW']:<12.2f}")

print("\n" + "="*80)
print("\nOutputs saved to: results/phase6_live_adaptation/")
print("  - CSV files with hourly forecasts and metrics")
print("  - Performance evolution visualizations")
print("  - Summary statistics (JSON)")

print("\nNext step: Phase 7 - Build interactive dashboard")
