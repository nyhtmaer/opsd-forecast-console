"""Load and clean OPSD power data"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
print("\nLoading dataset...")

df = pd.read_csv('time_series_60min_singleindex_filtered.csv')
print(f"Loaded {len(df):,} rows from full dataset")

df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
df = df.sort_values('utc_timestamp').reset_index(drop=True)

print(f"\nDate Range: {df['utc_timestamp'].min()} to {df['utc_timestamp'].max()}")
print(f"Total Columns: {len(df.columns)}")

load_cols = [col for col in df.columns if 'load_actual_entsoe' in col and col.startswith(('DE_', 'FR_', 'IT_'))]
print(f"\nTarget Load Columns: {load_cols}")

# Extract renewable columns (solar and wind)
renewable_cols = []
for country in ['DE_', 'FR_', 'IT_']:
    solar = f'{country}solar_generation_actual'
    wind_onshore = f'{country}wind_onshore_generation_actual'
    wind_offshore = f'{country}wind_offshore_generation_actual'
    wind_gen = f'{country}wind_generation_actual'
    
    # Check which renewable columns exist
    if solar in df.columns:
        renewable_cols.append(solar)
    if wind_onshore in df.columns:
        renewable_cols.append(wind_onshore)
    if wind_offshore in df.columns:
        renewable_cols.append(wind_offshore)
    if wind_gen in df.columns:
        renewable_cols.append(wind_gen)

print(f"\nExtracted Renewable Columns: {renewable_cols}")

print("\nData quality check:")
print("-" * 40)

for col in load_cols:
    country = col.split('_')[0]
    total = len(df)
    missing = df[col].isna().sum()
    available = total - missing
    pct = (available / total) * 100
    
    print(f"{country}: {available:,} values ({pct:.1f}%) | Missing: {missing:,} ({100-pct:.1f}%)")
    
    if available > 0:
        print(f"  Range: {df[col].min():.2f} to {df[col].max():.2f} MW")
        print(f"  Mean: {df[col].mean():.2f} MW | Std: {df[col].std():.2f} MW")

print("\nRenewable data quality check:")
print("-" * 40)

for col in renewable_cols:
    country = col.split('_')[0]
    total = len(df)
    missing = df[col].isna().sum()
    available = total - missing
    pct = (available / total) * 100
    
    renewable_type = 'Solar' if 'solar' in col else 'Wind'
    print(f"{country} {renewable_type}: {available:,} values ({pct:.1f}%) | Missing: {missing:,} ({100-pct:.1f}%)")
    
    if available > 0:
        print(f"  Range: {df[col].min():.2f} to {df[col].max():.2f} MWh")
        print(f"  Mean: {df[col].mean():.2f} MWh | Std: {df[col].std():.2f} MWh")

print("\nBasic statistics:")
print(df[load_cols].describe())

print("\nHandling missing values...")

for col in load_cols + renewable_cols:
    missing_before = df[col].isna().sum()
    if missing_before > 0:
        df[col] = df[col].ffill().bfill()
        missing_after = df[col].isna().sum()
        print(f"  {col}: {missing_before} → {missing_after} missing values")

df_clean = df.dropna(subset=load_cols)
print(f"\n Cleaned dataset: {len(df_clean):,} rows ({len(df_clean)/len(df)*100:.1f}% retained)")

print("\nChecking for outliers in load data...")

for col in load_cols:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    
    outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
    if outliers > 0:
        print(f"  {col}: {outliers} outliers detected (bounds: {lower_bound:.2f} - {upper_bound:.2f})")
        df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)

print("\nChecking for outliers in renewable data...")

for col in renewable_cols:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    
    outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
    if outliers > 0:
        print(f"  {col}: {outliers} outliers detected (bounds: {lower_bound:.2f} - {upper_bound:.2f})")
        df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)

print("\nSaving cleaned data...")

os.makedirs('data/preprocessed', exist_ok=True)
df_clean.to_csv('data/preprocessed/cleaned_full_data.csv', index=False)
print(" Saved to: data/preprocessed/cleaned_full_data.csv")

print("\n[STEP 5] Generating Quality Visualizations...")

os.makedirs('results/preprocessing', exist_ok=True)

fig, axes = plt.subplots(3, 1, figsize=(15, 10))
countries = {'DE': 'Germany', 'FR': 'France', 'IT': 'Italy'}

for idx, (code, name) in enumerate(countries.items()):
    col = f"{code}_load_actual_entsoe_transparency"
    
    axes[idx].plot(df_clean['utc_timestamp'], df_clean[col], linewidth=0.5, alpha=0.7)
    axes[idx].set_title(f"{name} - Electric Load Time Series (Cleaned)", fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('Load (MW)')
    axes[idx].grid(True, alpha=0.3)

axes[2].set_xlabel('Timestamp')
plt.tight_layout()
plt.savefig('results/preprocessing/cleaned_time_series.png', dpi=300, bbox_inches='tight')
print(" cleaned_time_series.png")
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (code, name) in enumerate(countries.items()):
    col = f"{code}_load_actual_entsoe_transparency"
    
    axes[idx].hist(df_clean[col], bins=50, alpha=0.7, edgecolor='black')
    axes[idx].axvline(df_clean[col].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df_clean[col].mean():.0f} MW')
    axes[idx].set_title(f"{name} - Load Distribution", fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Load (MW)')
    axes[idx].set_ylabel('Frequency')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/preprocessing/load_distributions.png', dpi=300, bbox_inches='tight')
print(" load_distributions.png")
plt.close()

summary_stats = pd.DataFrame()
for code in ['DE', 'FR', 'IT']:
    col = f"{code}_load_actual_entsoe_transparency"
    stats = {
        'Country': code,
        'Mean (MW)': df_clean[col].mean(),
        'Std (MW)': df_clean[col].std(),
        'Min (MW)': df_clean[col].min(),
        'Max (MW)': df_clean[col].max(),
        'Total Hours': len(df_clean),
        'Missing (%)': (df[col].isna().sum() / len(df)) * 100
    }
    summary_stats = pd.concat([summary_stats, pd.DataFrame([stats])], ignore_index=True)

summary_stats.to_csv('results/preprocessing/cleaning_summary.csv', index=False)
print(" cleaning_summary.csv")

print("\n" + "=" * 80)
print("DATA CLEANING COMPLETE")
print(f"\n Cleaned data: {len(df_clean):,} hours")
print(" Output: data/preprocessed/cleaned_full_data.csv")
print(" Visualizations: results/preprocessing/")
