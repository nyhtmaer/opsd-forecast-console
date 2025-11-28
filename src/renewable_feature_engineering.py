"""Engineer renewable generation features from solar and wind data"""

import pandas as pd
import numpy as np
import os

print("\nRENEWABLE FEATURE ENGINEERING PIPELINE")
print("=" * 80)

# Load cleaned data
print("\nLoading cleaned data...")
df_clean = pd.read_csv('data/preprocessed/cleaned_full_data.csv')
df_clean['utc_timestamp'] = pd.to_datetime(df_clean['utc_timestamp'])
print(f"Loaded: {len(df_clean):,} rows from {df_clean['utc_timestamp'].min()} to {df_clean['utc_timestamp'].max()}")

# Define countries and renewable columns
countries = ['DE', 'FR', 'IT']
renewable_features = {}

# Extract renewable columns
for country in countries:
    solar_col = f'{country}_solar_generation_actual'
    wind_onshore_col = f'{country}_wind_onshore_generation_actual'
    wind_gen_col = f'{country}_wind_generation_actual'
    load_col = f'{country}_load_actual_entsoe_transparency'
    
    renewable_features[country] = {
        'solar': solar_col if solar_col in df_clean.columns else None,
        'wind_onshore': wind_onshore_col if wind_onshore_col in df_clean.columns else None,
        'wind_gen': wind_gen_col if wind_gen_col in df_clean.columns else None,
        'load': load_col if load_col in df_clean.columns else None
    }

print("\nRenewable columns found:")
for country, cols in renewable_features.items():
    print(f"  {country}: {[c for c in cols.values() if c is not None]}")

def engineer_renewable_features(df, country, renewable_cols):
    """
    Engineer renewable features for a given country
    
    Features:
    - Lagged values: t-1, t-24, t-168
    - Rolling statistics: 24h and 168h windows (mean, std, min, max)
    - Renewable share: (solar + wind) / load ratio
    - Variability index: rolling_std / (rolling_mean + 1)
    """
    features_df = pd.DataFrame(index=df.index)
    features_df['utc_timestamp'] = df['utc_timestamp']
    
    solar_col = renewable_cols['solar']
    wind_onshore_col = renewable_cols['wind_onshore']
    wind_gen_col = renewable_cols['wind_gen']
    load_col = renewable_cols['load']
    
    # Combine wind sources (use wind_gen if available, else wind_onshore)
    wind_col_to_use = wind_gen_col if wind_gen_col is not None and df[wind_gen_col].notna().sum() > 0 else wind_onshore_col
    
    prefix = f'{country}_renew_'
    
    # ===== SOLAR FEATURES =====
    if solar_col is not None:
        solar = df[solar_col].fillna(0)
        
        # Lagged solar
        features_df[f'{prefix}solar_lag1'] = solar.shift(1)
        features_df[f'{prefix}solar_lag24'] = solar.shift(24)
        features_df[f'{prefix}solar_lag168'] = solar.shift(168)
        
        # Rolling solar statistics (24h window)
        features_df[f'{prefix}solar_roll24_mean'] = solar.rolling(window=24, min_periods=1).mean()
        features_df[f'{prefix}solar_roll24_std'] = solar.rolling(window=24, min_periods=1).std().fillna(0)
        features_df[f'{prefix}solar_roll24_min'] = solar.rolling(window=24, min_periods=1).min()
        features_df[f'{prefix}solar_roll24_max'] = solar.rolling(window=24, min_periods=1).max()
        
        # Rolling solar statistics (168h window)
        features_df[f'{prefix}solar_roll168_mean'] = solar.rolling(window=168, min_periods=1).mean()
        features_df[f'{prefix}solar_roll168_std'] = solar.rolling(window=168, min_periods=1).std().fillna(0)
        features_df[f'{prefix}solar_roll168_min'] = solar.rolling(window=168, min_periods=1).min()
        features_df[f'{prefix}solar_roll168_max'] = solar.rolling(window=168, min_periods=1).max()
        
        # Solar variability (normalized std)
        features_df[f'{prefix}solar_variability_24h'] = features_df[f'{prefix}solar_roll24_std'] / (features_df[f'{prefix}solar_roll24_mean'] + 1)
        features_df[f'{prefix}solar_variability_168h'] = features_df[f'{prefix}solar_roll168_std'] / (features_df[f'{prefix}solar_roll168_mean'] + 1)
    
    # ===== WIND FEATURES =====
    if wind_col_to_use is not None:
        wind = df[wind_col_to_use].fillna(0)
        
        # Lagged wind
        features_df[f'{prefix}wind_lag1'] = wind.shift(1)
        features_df[f'{prefix}wind_lag24'] = wind.shift(24)
        features_df[f'{prefix}wind_lag168'] = wind.shift(168)
        
        # Rolling wind statistics (24h window)
        features_df[f'{prefix}wind_roll24_mean'] = wind.rolling(window=24, min_periods=1).mean()
        features_df[f'{prefix}wind_roll24_std'] = wind.rolling(window=24, min_periods=1).std().fillna(0)
        features_df[f'{prefix}wind_roll24_min'] = wind.rolling(window=24, min_periods=1).min()
        features_df[f'{prefix}wind_roll24_max'] = wind.rolling(window=24, min_periods=1).max()
        
        # Rolling wind statistics (168h window)
        features_df[f'{prefix}wind_roll168_mean'] = wind.rolling(window=168, min_periods=1).mean()
        features_df[f'{prefix}wind_roll168_std'] = wind.rolling(window=168, min_periods=1).std().fillna(0)
        features_df[f'{prefix}wind_roll168_min'] = wind.rolling(window=168, min_periods=1).min()
        features_df[f'{prefix}wind_roll168_max'] = wind.rolling(window=168, min_periods=1).max()
        
        # Wind variability (normalized std)
        features_df[f'{prefix}wind_variability_24h'] = features_df[f'{prefix}wind_roll24_std'] / (features_df[f'{prefix}wind_roll24_mean'] + 1)
        features_df[f'{prefix}wind_variability_168h'] = features_df[f'{prefix}wind_roll168_std'] / (features_df[f'{prefix}wind_roll168_mean'] + 1)
    
    # ===== RENEWABLE SHARE FEATURES =====
    if solar_col is not None and wind_col_to_use is not None and load_col is not None:
        solar = df[solar_col].fillna(0)
        wind = df[wind_col_to_use].fillna(0)
        load = df[load_col].fillna(1)  # Avoid division by zero
        
        total_renewable = solar + wind
        
        # Renewable share (ratio of renewable to total load)
        features_df[f'{prefix}share_renewable'] = total_renewable / load
        
        # Rolling renewable share
        features_df[f'{prefix}share_roll24_mean'] = (total_renewable / load).rolling(window=24, min_periods=1).mean()
        features_df[f'{prefix}share_roll168_mean'] = (total_renewable / load).rolling(window=168, min_periods=1).mean()
        
        # Renewable generation change rate (useful for volatility analysis)
        features_df[f'{prefix}renewable_change'] = total_renewable.diff().fillna(0)
        features_df[f'{prefix}renewable_change_abs'] = total_renewable.diff().abs().fillna(0)
    
    return features_df

# Engineer features for each country
print("\nEngineering renewable features...")
all_renewable_features = []

for country in countries:
    print(f"  Processing {country}...")
    features = engineer_renewable_features(df_clean, country, renewable_features[country])
    all_renewable_features.append(features)
    print(f"    Generated {len(features.columns) - 1} features")

# Combine all features
print("\nCombining features from all countries...")
combined_features = all_renewable_features[0]
for features in all_renewable_features[1:]:
    # Merge on utc_timestamp, keeping all columns except duplicate timestamp
    combined_features = combined_features.merge(
        features.drop('utc_timestamp', axis=1),
        left_index=True,
        right_index=True,
        how='left'
    )

print(f"  Total renewable features: {len(combined_features.columns) - 1}")
print(f"  Feature columns: {[c for c in combined_features.columns if c != 'utc_timestamp'][:10]}... (showing first 10)")

# Replace NaN values with 0 (for initial periods before rolling windows stabilize)
combined_features = combined_features.fillna(0)

# Create train/val/test splits (same as in data_preprocessing.py)
print("\nCreating train/val/test splits...")
total_hours = len(df_clean)
train_size = int(0.8 * total_hours)
val_size = int(0.1 * total_hours)

train_features = combined_features.iloc[:train_size].copy()
val_features = combined_features.iloc[train_size:train_size + val_size].copy()
test_features = combined_features.iloc[train_size + val_size:].copy()

# Save renewable features
print("\nSaving renewable feature files...")
os.makedirs('data/preprocessed', exist_ok=True)

train_features.to_csv('data/preprocessed/train_renewable_features.csv', index=False)
print(f"  train_renewable_features.csv ({len(train_features):,} rows)")

val_features.to_csv('data/preprocessed/val_renewable_features.csv', index=False)
print(f"  val_renewable_features.csv ({len(val_features):,} rows)")

test_features.to_csv('data/preprocessed/test_renewable_features.csv', index=False)
print(f"  test_renewable_features.csv ({len(test_features):,} rows)")

# Print summary statistics
print("\nRenewable Features Summary Statistics:")
print("-" * 80)

summary_stats = []
for feature_col in combined_features.columns:
    if feature_col != 'utc_timestamp':
        stats = {
            'Feature': feature_col,
            'Mean': combined_features[feature_col].mean(),
            'Std': combined_features[feature_col].std(),
            'Min': combined_features[feature_col].min(),
            'Max': combined_features[feature_col].max(),
            'Non-zero %': (combined_features[feature_col] != 0).sum() / len(combined_features) * 100
        }
        summary_stats.append(stats)

summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv('data/preprocessed/renewable_features_summary.csv', index=False)
print(summary_df.to_string(index=False))

print("\n" + "=" * 80)
print("RENEWABLE FEATURE ENGINEERING COMPLETE")
print(f"\nOutput directory: data/preprocessed/")
print(f"Feature files:")
print(f"  - train_renewable_features.csv")
print(f"  - val_renewable_features.csv")
print(f"  - test_renewable_features.csv")
print(f"  - renewable_features_summary.csv")
print(f"\nThese files can be merged with train_data.csv, val_data.csv, test_data.csv")
print(f"using left merge on 'utc_timestamp' when needed for future model enhancements.")
