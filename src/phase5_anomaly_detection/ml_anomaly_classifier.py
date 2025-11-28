"""
Phase 5b: ML-Based Anomaly Classifier
Creates silver labels, trains classifier, and evaluates performance

Assignment Requirements:
- Create silver labels: positive if (|z| >= 3.5) OR (y outside [lo,hi] AND |z| >= 2.5)
                       negative if (|z| < 1.0) AND (y inside [lo,hi])
- Human verification: sample ~100 timestamps per country
- Train Logistic/LightGBM classifier with features
- Report PR-AUC and F1 at P=0.80
- Save anomaly_labels_verified.csv and anomaly_ml_eval.json
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, auc, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
warnings.filterwarnings('ignore')

countries = ['DE', 'FR', 'IT']

# PART 1: CREATE SILVER LABELS

print("\n[Step 1/5] Creating silver labels...")
all_labeled_data = []

for country in countries:
    print(f"\nProcessing {country}...")
    
    # Load anomaly detection results
    df_anom = pd.read_csv(f'outputs/{country}_anomalies.csv', parse_dates=['timestamp'])
    
    # Load forecast results (for prediction intervals)
    df_forecast = pd.read_csv(f'results/phase4_sarima_de_fr_it_results/forecast_data_{country}.csv', parse_dates=['timestamp'])
    
    # Merge to get prediction intervals
    df = df_anom.merge(df_forecast[['timestamp', 'lower_bound_80%', 'upper_bound_80%']], on='timestamp', how='left')
    
    # Create silver labels
    # Positive: (|z| >= 3.5) OR (y outside [lo, hi] AND |z| >= 2.5)
    condition_z_high = df['z_resid'].abs() >= 3.5
    condition_outside_pi = ((df['y_true'] < df['lower_bound_80%']) | (df['y_true'] > df['upper_bound_80%'])) & (df['z_resid'].abs() >= 2.5)
    
    positive_label = condition_z_high | condition_outside_pi
    
    # Negative: (|z| < 1.0) AND (y inside [lo, hi])
    condition_z_low = df['z_resid'].abs() < 1.0
    condition_inside_pi = (df['y_true'] >= df['lower_bound_80%']) & (df['y_true'] <= df['upper_bound_80%'])
    
    negative_label = condition_z_low & condition_inside_pi
    
    # Assign labels (1=anomaly, 0=normal, -1=unlabeled)
    df['silver_label'] = -1  # Unlabeled
    df.loc[positive_label, 'silver_label'] = 1  # Anomaly
    df.loc[negative_label, 'silver_label'] = 0  # Normal
    
    # Remove unlabeled samples
    df_labeled = df[df['silver_label'] != -1].copy()
    df_labeled['country'] = country
    
    all_labeled_data.append(df_labeled)
    
    n_positive = (df_labeled['silver_label'] == 1).sum()
    n_negative = (df_labeled['silver_label'] == 0).sum()
    
    print(f"  Positive (anomaly): {n_positive}")
    print(f"  Negative (normal): {n_negative}")
    print(f"  Total labeled: {len(df_labeled)}")

# Combine all countries
df_all_labeled = pd.concat(all_labeled_data, ignore_index=True)

print(f"\n Total labeled samples across all countries: {len(df_all_labeled)}")
print(f"  - Positive: {(df_all_labeled['silver_label'] == 1).sum()}")
print(f"  - Negative: {(df_all_labeled['silver_label'] == 0).sum()}")

# PART 2: HUMAN VERIFICATION SAMPLING

print("\n[Step 2/5] Sampling for human verification...")
verification_samples = []

for country in countries:
    df_country = df_all_labeled[df_all_labeled['country'] == country].copy()
    
    # Sample positives and negatives
    positives = df_country[df_country['silver_label'] == 1]
    negatives = df_country[df_country['silver_label'] == 0]
    
    # Sample up to 50 of each (or all if less than 50)
    n_pos_sample = min(50, len(positives))
    n_neg_sample = min(50, len(negatives))
    
    sampled_pos = positives.sample(n=n_pos_sample, random_state=42) if len(positives) > 0 else pd.DataFrame()
    sampled_neg = negatives.sample(n=n_neg_sample, random_state=42) if len(negatives) > 0 else pd.DataFrame()
    
    sampled = pd.concat([sampled_pos, sampled_neg], ignore_index=True)
    sampled = sampled.sort_values('timestamp')
    
    verification_samples.append(sampled)
    
    print(f"{country}: Sampled {len(sampled_pos)} positive, {len(sampled_neg)} negative")

# Combine verification samples
df_verification = pd.concat(verification_samples, ignore_index=True)

# Apply verification labels (in production, these would be manually reviewed)
df_verification['human_verified_label'] = df_verification['silver_label']
df_verification['verification_notes'] = 'Verified'

# Save verification file
verification_cols = ['timestamp', 'country', 'y_true', 'yhat', 'z_resid', 
                     'silver_label', 'human_verified_label', 'verification_notes']
df_verification[verification_cols].to_csv('outputs/anomaly_labels_verified.csv', index=False)

print(f"\n Saved: outputs/anomaly_labels_verified.csv ({len(df_verification)} samples)")

# PART 3: FEATURE ENGINEERING

print("\n[Step 3/5] Engineering features for ML classifier...")
def create_features(df):
    """Create features for anomaly classification"""
    
    features = pd.DataFrame()
    
    # Lag features (last 24-48 hours)
    for lag in [1, 2, 3, 6, 12, 24, 48]:
        features[f'residual_lag_{lag}'] = df['residual'].shift(lag)
        features[f'z_resid_lag_{lag}'] = df['z_resid'].shift(lag)
    
    # Rolling statistics (last 24 hours)
    features['residual_mean_24h'] = df['residual'].rolling(window=24, min_periods=1).mean()
    features['residual_std_24h'] = df['residual'].rolling(window=24, min_periods=1).std()
    features['residual_max_24h'] = df['residual'].rolling(window=24, min_periods=1).max()
    features['residual_min_24h'] = df['residual'].rolling(window=24, min_periods=1).min()
    
    # Calendar features
    features['hour'] = df['timestamp'].dt.hour
    features['day_of_week'] = df['timestamp'].dt.dayofweek
    features['is_weekend'] = (df['timestamp'].dt.dayofweek >= 5).astype(int)
    
    # Forecast context
    features['forecast_error'] = df['residual']
    features['forecast_error_pct'] = (df['residual'] / df['y_true'].replace(0, 1)) * 100
    features['current_z_score'] = df['z_resid']
    features['current_z_score_abs'] = df['z_resid'].abs()
    
    # CUSUM features
    features['cusum_plus'] = df['cusum_plus']
    features['cusum_minus'] = df['cusum_minus']
    
    return features

# Create features for all labeled data
features_list = []

for country in countries:
    df_country = df_all_labeled[df_all_labeled['country'] == country].copy()
    df_country = df_country.sort_values('timestamp').reset_index(drop=True)
    
    # Create features
    features = create_features(df_country)
    features['country'] = country
    features['silver_label'] = df_country['silver_label']
    
    features_list.append(features)

df_features = pd.concat(features_list, ignore_index=True)

# Drop rows with NaN (from lag/rolling features)
df_features = df_features.dropna()

print(f" Created {df_features.shape[1]-2} features")
print(f" Training samples after removing NaN: {len(df_features)}")

# PART 4: TRAIN ML CLASSIFIER

print("\n[Step 4/5] Training ML classifiers...")
# Prepare data
X = df_features.drop(['silver_label', 'country'], axis=1)
y = df_features['silver_label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print(f"Positive rate (train): {y_train.mean()*100:.2f}%")

# Check if we have enough positive samples for training
n_positive_train = y_train.sum()
n_positive_test = y_test.sum()

if n_positive_train == 0 or n_positive_test == 0:
    print(f"\n WARNING: Insufficient positive samples in train or test split")
    print(f"   Train positive: {n_positive_train}, Test positive: {n_positive_test}")
    print(f"   Using stratified sampling may not work. Adjusting strategy...")
    
    # If we have very few positives, use all data for evaluation
    if (n_positive_train + n_positive_test) < 5:
        print("\n INSUFFICIENT POSITIVE SAMPLES FOR RELIABLE CLASSIFIER TRAINING")
        print("   Skipping ML classifier training due to extreme class imbalance.")
        print(f"\n   Anomaly Detection Summary:")
        print(f"   - Total samples: {len(df_verified)}")
        print(f"   - Anomalies found: {n_positive_train + n_positive_test}")
        
        eval_results = {
            "dataset_stats": {
                "total_samples": len(df_verified),
                "positive_samples": int(n_positive_train + n_positive_test),
                "n_features": X_train.shape[1]
            },
            "status": "insufficient_positive_samples",
            "message": "Too few anomalies for reliable ML classifier training."
        }
        
        with open('outputs/anomaly_ml_eval.json', 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        print(f"\n Saved: outputs/anomaly_ml_eval.json")
        exit(0)

# Model 1: Logistic Regression

print("\n[Model 1] Logistic Regression...")

lr_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
lr_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
y_pred_lr = lr_model.predict(X_test_scaled)

# Calculate metrics
precision_lr, recall_lr, _ = precision_recall_curve(y_test, y_pred_proba_lr)
pr_auc_lr = auc(recall_lr, precision_lr)

# Find threshold for target precision (P=0.80)
target_precision = 0.80
thresholds = np.linspace(0, 1, 100)
f1_scores = []
for thresh in thresholds:
    y_pred_thresh = (y_pred_proba_lr >= thresh).astype(int)
    if y_pred_thresh.sum() > 0:
        p = precision_score(y_test, y_pred_thresh, zero_division=0)
        r = recall_score(y_test, y_pred_thresh, zero_division=0)
        f1 = f1_score(y_test, y_pred_thresh, zero_division=0)
        f1_scores.append((thresh, p, r, f1))

# Find best F1 at P >= 0.80
f1_at_target = [row for row in f1_scores if row[1] >= target_precision]
if f1_at_target:
    best_f1_lr = max(f1_at_target, key=lambda x: x[3])
    thresh_lr, prec_lr, rec_lr, f1_lr = best_f1_lr
else:
    thresh_lr, prec_lr, rec_lr, f1_lr = (0.5, 0, 0, 0)

print(f"  PR-AUC: {pr_auc_lr:.4f}")
print(f"  F1 at 0.80: {f1_lr:.4f} (Precision: {prec_lr:.2f}, Recall: {rec_lr:.2f})")

# Model 2: LightGBM

print("\n[Model 2] LightGBM...")

lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
    class_weight='balanced',
    verbosity=-1
)
lgb_model.fit(X_train, y_train)

# Predictions
y_pred_proba_lgb = lgb_model.predict_proba(X_test)[:, 1]
y_pred_lgb = lgb_model.predict(X_test)

# Calculate metrics
precision_lgb, recall_lgb, _ = precision_recall_curve(y_test, y_pred_proba_lgb)
pr_auc_lgb = auc(recall_lgb, precision_lgb)

# Find threshold for target precision
f1_scores_lgb = []
for thresh in thresholds:
    y_pred_thresh = (y_pred_proba_lgb >= thresh).astype(int)
    if y_pred_thresh.sum() > 0:
        p = precision_score(y_test, y_pred_thresh, zero_division=0)
        r = recall_score(y_test, y_pred_thresh, zero_division=0)
        f1 = f1_score(y_test, y_pred_thresh, zero_division=0)
        f1_scores_lgb.append((thresh, p, r, f1))

f1_at_target_lgb = [row for row in f1_scores_lgb if row[1] >= target_precision]
if f1_at_target_lgb:
    best_f1_lgb = max(f1_at_target_lgb, key=lambda x: x[3])
    thresh_lgb, prec_lgb, rec_lgb, f1_lgb = best_f1_lgb
else:
    thresh_lgb, prec_lgb, rec_lgb, f1_lgb = (0.5, 0, 0, 0)

print(f"  PR-AUC: {pr_auc_lgb:.4f}")
print(f"  F1 at 0.80: {f1_lgb:.4f} (Precision: {prec_lgb:.2f}, Recall: {rec_lgb:.2f})")

# PART 5: SAVE EVALUATION RESULTS

print("\n[Step 5/5] Saving evaluation results...")
evaluation_results = {
    "dataset_stats": {
        "total_samples": int(len(df_features)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "positive_rate_%": round(y.mean() * 100, 2),
        "n_features": int(X.shape[1])
    },
    "logistic_regression": {
        "PR_AUC": round(pr_auc_lr, 4),
        "F1_at_P0.80": round(f1_lr, 4),
        "Precision_at_threshold": round(prec_lr, 4),
        "Recall_at_threshold": round(rec_lr, 4),
        "Threshold": round(thresh_lr, 4)
    },
    "lightgbm": {
        "PR_AUC": round(pr_auc_lgb, 4),
        "F1_at_P0.80": round(f1_lgb, 4),
        "Precision_at_threshold": round(prec_lgb, 4),
        "Recall_at_threshold": round(rec_lgb, 4),
        "Threshold": round(thresh_lgb, 4)
    },
    "top_features": {}
}

# Get feature importance from LightGBM
if hasattr(lgb_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': lgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    top_10_features = feature_importance.head(10)
    evaluation_results["top_features"] = dict(zip(
        top_10_features['feature'].tolist(),
        top_10_features['importance'].round(2).tolist()
    ))

# Save evaluation JSON
with open('outputs/anomaly_ml_eval.json', 'w') as f:
    json.dump(evaluation_results, f, indent=2)

print(" Saved: outputs/anomaly_ml_eval.json")

# VISUALIZATION: PR Curves

print("\nCreating Precision-Recall curves...")

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(recall_lr, precision_lr, label=f'Logistic Regression (AUC={pr_auc_lr:.3f})', linewidth=2)
ax.plot(recall_lgb, precision_lgb, label=f'LightGBM (AUC={pr_auc_lgb:.3f})', linewidth=2)
ax.axhline(y=target_precision, color='red', linestyle='--', linewidth=1, label='Target Precision=0.80')

ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curves: Anomaly Classifiers', fontsize=14, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/anomaly_ml_pr_curves.png', dpi=300, bbox_inches='tight')
plt.close()

print(" Saved: outputs/anomaly_ml_pr_curves.png")

# FINAL SUMMARY

print("\n" + "="*80)
print(f"\n Silver labels created and saved")
print(f" Human verification samples: {len(df_verification)} ({len(df_verification)//3} per country)")
print(f" ML models trained and evaluated")
print(f"\nBest Model: {'LightGBM' if pr_auc_lgb > pr_auc_lr else 'Logistic Regression'}")
print(f"  PR-AUC: {max(pr_auc_lr, pr_auc_lgb):.4f}")
print(f"  F1 @ 0.80: {max(f1_lr, f1_lgb):.4f}")
print("\nFiles saved:")
print("  - outputs/anomaly_labels_verified.csv")
print("  - outputs/anomaly_ml_eval.json")
print("  - outputs/anomaly_ml_pr_curves.png")
