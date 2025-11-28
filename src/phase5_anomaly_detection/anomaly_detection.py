"""
 Anomaly Detection
Part 1: Residual z-score + CUSUM (unsupervised)
Part 2: ML-based anomaly classifier (supervised with silver labels)

Assignment Requirements:
- Compute 1-step-ahead residuals from SARIMA forecasts on Test set
- Rolling z-score (window=336h, min_periods=168)
- Flag anomalies if |z| >= 3.0
- Optional: CUSUM with k=0.5, h=5.0
- Save outputs/<CC>_anomalies.csv

NOTE: Uses last 1000 hours for rolling window calculation
      SARIMA forecast residuals used for test period (Sept 19-30)
      Naive 24h forecast used for historical period
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

os.makedirs('outputs', exist_ok=True)

print(" ANOMALY DETECTION (FULL RUN)")
# PART 1: RESIDUAL Z-SCORE + CUSUM

countries = ['DE', 'FR', 'IT']

print("\n[Part 1] Residual z-score based anomaly detection")
for country in countries:
    print(f"\nProcessing {country}...")
    
    # Load dataset to get historical context
    df_full = pd.read_csv('time_series_60min_singleindex_filtered.csv', index_col=0, parse_dates=True)
    
    load_columns = {
        'DE': 'DE_load_actual_entsoe_transparency',
        'FR': 'FR_load_actual_entsoe_transparency',
        'IT': 'IT_load_actual_entsoe_transparency'
    }
    
    # Get last 1000 hours of actual data for rolling window calculation
    data = df_full[load_columns[country]].dropna()
    data_1000h = data.iloc[-1000:].copy()
    
    # Load SARIMA forecast results from Phase 4 (test set)
    df_forecast = pd.read_csv(f'results/phase4_sarima_de_fr_it_results/forecast_data_{country}.csv', parse_dates=['timestamp'])
    
    # Create dataframe with historical data + test set
    # Use naive 24h forecast for historical period, SARIMA forecast for test period
    df_hist = pd.DataFrame({
        'timestamp': data_1000h.index,
        'actual': data_1000h.values
    })
    df_hist['forecast'] = df_hist['actual'].shift(24)  # Naive forecast for historical
    df_hist['residual'] = df_hist['actual'] - df_hist['forecast']
    df_hist = df_hist.dropna()
    
    # Merge with SARIMA test set forecasts (replace test period with SARIMA residuals)
    df_hist['is_test'] = df_hist['timestamp'].isin(df_forecast['timestamp'])
    
    # Replace test period residuals with SARIMA residuals
    for idx, row in df_forecast.iterrows():
        mask = df_hist['timestamp'] == row['timestamp']
        if mask.any():
            df_hist.loc[mask, 'forecast'] = row['forecast']
            df_hist.loc[mask, 'residual'] = row['error']
    
    df = df_hist.copy()
    
    # Rolling z-score (window=336h = 14 days, min_periods=168 = 7 days)
    # NOTE: Test set only has 288 hours, so z-scores will have many NaN values initially
    # This is expected - we're using the rolling window specification from assignment
    window = 336
    min_periods = 168
    
    # Calculate rolling mean and std
    rolling_mean = df['residual'].rolling(window=window, min_periods=min_periods).mean()
    rolling_std = df['residual'].rolling(window=window, min_periods=min_periods).std()
    
    # Calculate z-score
    df['z_resid'] = (df['residual'] - rolling_mean) / rolling_std
    
    # Flag anomalies (|z| >= 3.0)
    df['flag_z'] = (df['z_resid'].abs() >= 3.0).astype(int)
    
    # CUSUM (Cumulative Sum Control Chart)
    
    # CUSUM parameters
    k = 0.5  # Reference value (allowance/slack)
    h = 5.0  # Decision interval (threshold)
    
    # Initialize CUSUM
    n = len(df)
    S_plus = np.zeros(n)   # Upper CUSUM
    S_minus = np.zeros(n)  # Lower CUSUM
    
    # Calculate CUSUM
    for i in range(1, n):
        if not pd.isna(df['z_resid'].iloc[i]):
            z = df['z_resid'].iloc[i]
            S_plus[i] = max(0, S_plus[i-1] + z - k)
            S_minus[i] = max(0, S_minus[i-1] - z - k)
    
    df['cusum_plus'] = S_plus
    df['cusum_minus'] = S_minus
    
    # Flag CUSUM anomalies (either S+ or S- exceeds threshold)
    df['flag_cusum'] = ((S_plus > h) | (S_minus > h)).astype(int)
    
    # Save anomaly results
    
    # Select columns for output
    output_cols = ['timestamp', 'actual', 'forecast', 'residual', 'z_resid', 
                   'flag_z', 'cusum_plus', 'cusum_minus', 'flag_cusum']
    
    # Rename for clarity
    anomaly_df = df[output_cols].copy()
    anomaly_df.rename(columns={'actual': 'y_true', 'forecast': 'yhat'}, inplace=True)
    
    # Save to CSV
    output_file = f'outputs/{country}_anomalies.csv'
    anomaly_df.to_csv(output_file, index=False)
    
    # Print statistics
    
    n_total = len(df)
    n_z_anomalies = df['flag_z'].sum()
    n_cusum_anomalies = df['flag_cusum'].sum()
    
    print(f"  Total points: {n_total}")
    print(f"  Z-score anomalies (|z| >= 3.0): {n_z_anomalies} ({n_z_anomalies/n_total*100:.2f}%)")
    print(f"  CUSUM anomalies: {n_cusum_anomalies} ({n_cusum_anomalies/n_total*100:.2f}%)")
    print(f"   Saved: {output_file}")

# VISUALIZATION: Anomaly Detection Results

print("\n[Visualization] Creating anomaly detection plots...")

for country in countries:
    # Load anomaly results
    df = pd.read_csv(f'outputs/{country}_anomalies.csv', parse_dates=['timestamp'])
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f'{country} - Anomaly Detection Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Actual vs Forecast with anomalies
    ax1 = axes[0]
    ax1.plot(df['timestamp'], df['y_true'], label='Actual Load', color='black', alpha=0.7, linewidth=1)
    ax1.plot(df['timestamp'], df['yhat'], label='Forecast', color='blue', alpha=0.5, linewidth=1)
    
    # Highlight z-score anomalies
    z_anomalies = df[df['flag_z'] == 1]
    if len(z_anomalies) > 0:
        ax1.scatter(z_anomalies['timestamp'], z_anomalies['y_true'], 
                   color='red', s=50, label='Z-score Anomaly', zorder=5, marker='o')
    
    ax1.set_ylabel('Load (MW)', fontsize=11)
    ax1.set_title('Load Forecast with Anomalies Flagged', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Residual z-scores
    ax2 = axes[1]
    ax2.plot(df['timestamp'], df['z_resid'], color='steelblue', linewidth=0.8, alpha=0.8)
    ax2.axhline(y=3.0, color='red', linestyle='--', linewidth=1, label='Threshold (+3ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚ÂÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢)')
    ax2.axhline(y=-3.0, color='red', linestyle='--', linewidth=1, label='Threshold (-3ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚ÂÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢)')
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax2.fill_between(df['timestamp'], -3, 3, alpha=0.1, color='green')
    ax2.set_ylabel('Z-score', fontsize=11)
    ax2.set_title('Rolling Z-score of Residuals (window=336h)', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: CUSUM
    ax3 = axes[2]
    ax3.plot(df['timestamp'], df['cusum_plus'], label='CUSUM (+)', color='orangered', linewidth=1)
    ax3.plot(df['timestamp'], df['cusum_minus'], label='CUSUM (-)', color='dodgerblue', linewidth=1)
    ax3.axhline(y=5.0, color='red', linestyle='--', linewidth=1, label='Threshold (h=5.0)')
    ax3.set_ylabel('CUSUM Value', fontsize=11)
    ax3.set_xlabel('Timestamp', fontsize=11)
    ax3.set_title('CUSUM Control Chart (k=0.5, h=5.0)', fontsize=12)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'outputs/{country}_anomaly_detection.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved: outputs/{country}_anomaly_detection.png")

# SUMMARY STATISTICS

print("\n[Summary] Anomaly Detection Statistics")
summary_stats = {}

for country in countries:
    df = pd.read_csv(f'outputs/{country}_anomalies.csv', parse_dates=['timestamp'])
    
    summary_stats[country] = {
        'total_points': len(df),
        'z_anomalies': int(df['flag_z'].sum()),
        'z_anomaly_rate_%': round(df['flag_z'].mean() * 100, 2),
        'cusum_anomalies': int(df['flag_cusum'].sum()),
        'cusum_anomaly_rate_%': round(df['flag_cusum'].mean() * 100, 2),
        'max_z_score': round(df['z_resid'].abs().max(), 2),
        'mean_abs_z': round(df['z_resid'].abs().mean(), 2)
    }

# Save summary
with open('outputs/anomaly_summary_stats.json', 'w') as f:
    json.dump(summary_stats, f, indent=2)

print("\n" + pd.DataFrame(summary_stats).T.to_string())
print("\n Saved: outputs/anomaly_summary_stats.json")

print("\n" + "="*80)
print("PART 1 COMPLETE: Z-score and CUSUM anomaly detection")
print("\nNext steps:")
print("  - Part 2: ML-based anomaly classifier (silver labels + training)")
print("  - Review anomaly plots in outputs/ folder")
