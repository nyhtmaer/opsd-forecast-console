"""Split data into train/val/test sets"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
print("OPSD DATA PREPROCESSING PIPELINE")
print("\nLoading cleaned data...")

df_clean = pd.read_csv('data/preprocessed/cleaned_full_data.csv')
df_clean['utc_timestamp'] = pd.to_datetime(df_clean['utc_timestamp'])

load_cols = [col for col in df_clean.columns if 'load_actual_entsoe' in col and col.startswith(('DE_', 'FR_', 'IT_'))]

print(f" Loaded: {len(df_clean):,} hours")
print(f"  Period: {df_clean['utc_timestamp'].min()} to {df_clean['utc_timestamp'].max()}")
print(f"  Load columns: {load_cols}")

print("\nCreating train/val/test splits (80/10/10)...")

total_hours = len(df_clean)
train_size = int(0.8 * total_hours)
val_size = int(0.1 * total_hours)
test_size = total_hours - train_size - val_size

train_data = df_clean.iloc[:train_size].copy()
val_data = df_clean.iloc[train_size:train_size + val_size].copy()
test_data = df_clean.iloc[train_size + val_size:].copy()

print(f"Total hours: {total_hours:,}")
print(f"\nTrain set: {len(train_data):,} hours ({len(train_data)/total_hours*100:.1f}%)")
print(f"  Period: {train_data['utc_timestamp'].min()} to {train_data['utc_timestamp'].max()}")
print(f"\nValidation set: {len(val_data):,} hours ({len(val_data)/total_hours*100:.1f}%)")
print(f"  Period: {val_data['utc_timestamp'].min()} to {val_data['utc_timestamp'].max()}")
print(f"\nTest set: {len(test_data):,} hours ({len(test_data)/total_hours*100:.1f}%)")
print(f"  Period: {test_data['utc_timestamp'].min()} to {test_data['utc_timestamp'].max()}")

print("\nAdding time features...")

for dataset, name in [(train_data, 'train'), (val_data, 'val'), (test_data, 'test')]:
    dataset['hour'] = dataset['utc_timestamp'].dt.hour
    dataset['day_of_week'] = dataset['utc_timestamp'].dt.dayofweek
    dataset['month'] = dataset['utc_timestamp'].dt.month
    dataset['is_weekend'] = (dataset['day_of_week'] >= 5).astype(int)
    print(f"  {name}: hour, day_of_week, month, is_weekend")

print("\nSaving split datasets...")

train_data.to_csv('data/preprocessed/train_data.csv', index=False)
val_data.to_csv('data/preprocessed/val_data.csv', index=False)
test_data.to_csv('data/preprocessed/test_data.csv', index=False)

print(" train_data.csv")
print(" val_data.csv")
print(" test_data.csv")

print("\nGenerating visualizations...")

fig, axes = plt.subplots(3, 1, figsize=(15, 10))
countries = {'DE': 'Germany', 'FR': 'France', 'IT': 'Italy'}

for idx, (code, name) in enumerate(countries.items()):
    col = f"{code}_load_actual_entsoe_transparency"
    
    axes[idx].plot(df_clean['utc_timestamp'], df_clean[col], linewidth=0.5, alpha=0.7)
    axes[idx].set_title(f"{name} - Train/Val/Test Split", fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('Load (MW)')
    axes[idx].grid(True, alpha=0.3)
    
    axes[idx].axvline(train_data['utc_timestamp'].iloc[-1], color='red', linestyle='--', alpha=0.5, label='Train/Val split')
    axes[idx].axvline(val_data['utc_timestamp'].iloc[-1], color='orange', linestyle='--', alpha=0.5, label='Val/Test split')
    axes[idx].legend()

axes[2].set_xlabel('Timestamp')
plt.tight_layout()
plt.savefig('results/preprocessing/data_splits.png', dpi=300, bbox_inches='tight')
print(" data_splits.png")
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (code, name) in enumerate(countries.items()):
    col = f"{code}_load_actual_entsoe_transparency"
    
    hourly_avg = train_data.groupby('hour')[col].mean()
    axes[idx].plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2)
    axes[idx].fill_between(hourly_avg.index, hourly_avg.values, alpha=0.3)
    axes[idx].set_title(f"{name} - Average Hourly Load", fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Hour of Day')
    axes[idx].set_ylabel('Average Load (MW)')
    axes[idx].set_xticks(range(0, 24, 3))
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/preprocessing/hourly_patterns.png', dpi=300, bbox_inches='tight')
print(" hourly_patterns.png")
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

for idx, (code, name) in enumerate(countries.items()):
    col = f"{code}_load_actual_entsoe_transparency"
    
    daily_avg = train_data.groupby('day_of_week')[col].mean()
    axes[idx].bar(range(7), daily_avg.values, alpha=0.7, edgecolor='black')
    axes[idx].set_title(f"{name} - Average Load by Day", fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Day of Week')
    axes[idx].set_ylabel('Average Load (MW)')
    axes[idx].set_xticks(range(7))
    axes[idx].set_xticklabels(days)
    axes[idx].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/preprocessing/weekly_patterns.png', dpi=300, bbox_inches='tight')
print(" weekly_patterns.png")
plt.close()

summary_stats = pd.DataFrame()
for dataset_name, dataset in [('Train', train_data), ('Val', val_data), ('Test', test_data)]:
    for code in ['DE', 'FR', 'IT']:
        col = f"{code}_load_actual_entsoe_transparency"
        stats = {
            'Dataset': dataset_name,
            'Country': code,
            'Hours': len(dataset),
            'Mean (MW)': dataset[col].mean(),
            'Std (MW)': dataset[col].std(),
            'Min (MW)': dataset[col].min(),
            'Max (MW)': dataset[col].max()
        }
        summary_stats = pd.concat([summary_stats, pd.DataFrame([stats])], ignore_index=True)

summary_stats.to_csv('results/preprocessing/split_summary.csv', index=False)
print(" split_summary.csv")

print("\n" + "=" * 80)
print("DATA PREPROCESSING COMPLETE")
print(f"\n Train set: {len(train_data):,} hours")
print(f" Validation set: {len(val_data):,} hours")
print(f" Test set: {len(test_data):,} hours")
print("\n Output directory: data/preprocessed/")
print(" Visualizations: results/preprocessing/")
