"""
Model Building - SARIMA Grid Search with AIC/BIC (Parallel)
Implements optimal SARIMA parameter selection for day-ahead electricity load forecasting
Uses parallel processing for faster grid search
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product
import warnings
import os
import json
import time
from joblib import Parallel, delayed
import gc  # Garbage collector for memory management
warnings.filterwarnings('ignore')

os.makedirs('phase3_results', exist_ok=True)

# Determine number of CPU cores
n_jobs = 4  # Use 2 cores for parallel processing

# Helper function for parallel SARIMA fitting
def fit_sarima_model(train_data, dev_data, p, d, q, P, D, Q, s, country):
    """
    Fit a single SARIMA model and validate on dev set.
    Used for parallel processing with aggressive memory cleanup.
    """
    try:
        model = SARIMAX(
            train_data,
            order=(p, d, q),
            seasonal_order=(P, D, Q, s),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        start_time = time.time()
        fitted_model = model.fit(disp=False, maxiter=100)
        fit_time = time.time() - start_time
        
        # Validate on dev set - forecast and calculate MSE
        try:
            forecast_steps = len(dev_data)
            forecast = fitted_model.forecast(steps=forecast_steps)
            dev_mse = np.mean((dev_data.values - forecast.values) ** 2)
        except Exception:
            dev_mse = float('inf')
        
        # Extract results before deleting objects
        result = {
            'country': country,
            'p': p, 'd': d, 'q': q,
            'P': P, 'D': D, 'Q': Q, 's': s,
            'AIC': float(fitted_model.aic),
            'BIC': float(fitted_model.bic),
            'dev_MSE': float(dev_mse),
            'fit_time': fit_time,
            'success': True
        }
        
        # Explicitly delete large objects and force garbage collection
        del fitted_model
        del model
        gc.collect()
        
        return result
    except Exception as e:
        # Clean up on error too
        gc.collect()
        return {
            'country': country,
            'p': p, 'd': d, 'q': q,
            'P': P, 'D': D, 'Q': Q, 's': s,
            'success': False,
            'error': str(e)
        }

print("MODEL BUILDING - SARIMA GRID SEARCH - DE/FR/IT")
# Load dataset
print("\n Loading dataset...")
df = pd.read_csv('data/preprocessed/train_data.csv', parse_dates=['utc_timestamp'])
df = df.set_index('utc_timestamp')

# Select 3 countries
countries = ['DE', 'FR', 'IT']
print(f"Selected countries: {countries}")

# Get load columns
load_columns = {}
for country in countries:
    cols = [col for col in df.columns if col.startswith(f'{country}_load')]
    if cols:
        load_columns[country] = cols[0]

print(f"Load columns: {load_columns}")

# Load preprocessed train and validation data
print("\n Loading train and validation data...")

# Load validation data for dev set
df_val = pd.read_csv('data/preprocessed/val_data.csv', parse_dates=['utc_timestamp'])
df_val = df_val.set_index('utc_timestamp')

data_splits = {}

for country, load_col in load_columns.items():
    # Get train and dev data
    train = df[load_col].dropna()
    dev = df_val[load_col].dropna()
    
    # Use dev as test for grid search phase
    test = dev
    
    n = len(train) + len(dev) + len(test)
    
    data_splits[country] = {
        'train': train,
        'dev': dev,
        'test': test,
        'train_size': len(train),
        'dev_size': len(dev),
        'test_size': len(test),
        'train_period': f"{train.index[0].strftime('%Y-%m-%d')} to {train.index[-1].strftime('%Y-%m-%d')}",
        'dev_period': f"{dev.index[0].strftime('%Y-%m-%d')} to {dev.index[-1].strftime('%Y-%m-%d')}",
        'test_period': f"{test.index[0].strftime('%Y-%m-%d')} to {test.index[-1].strftime('%Y-%m-%d')}"
    }
    
    print(f"\n{country}:")
    print(f"  Total: {n:,} hours")
    print(f"  Train: {len(train):,} hours ({len(train)/n*100:.1f}%) - {data_splits[country]['train_period']}")
    print(f"  Dev:   {len(dev):,} hours ({len(dev)/n*100:.1f}%) - {data_splits[country]['dev_period']}")
    print(f"  Test:  {len(test):,} hours ({len(test)/n*100:.1f}%) - {data_splits[country]['test_period']}")

# SARIMA GRID SEARCH WITH AIC/BIC + DEV SET VALIDATION
print("\n Performing SARIMA grid search with AIC/BIC + Dev Set Validation...")
print("Grid parameters:")
print("  - (p,q)  {0,1,2}")
print("  - d ‚  {0,1}")
print("  - (P,Q)   {0,1}")
print("  - D   {0,1}")
print("  - s = 24 (daily seasonality)")
print("Note: Models trained on train set, validated on dev set")

# Define parameter grid (as per PDF)
p_values = [0, 1, 2]
d_values = [0, 1]
q_values = [0, 1, 2]
P_values = [0, 1]
D_values = [0, 1]
Q_values = [0, 1]
s = 24

# Store results
grid_search_results = {}

# Partial results file (checkpoint)
partial_csv = 'phase3_results/grid_search_partial.csv'

for country, load_col in load_columns.items():
    print(f"\n{'='*60}")
    print(f"Grid Search: {country}")
    print(f"{'='*60}")
    
    train_data = data_splits[country]['train']
    dev_data = data_splits[country]['dev']
    
    # Prepare all parameter combinations
    combos = list(product(p_values, d_values, q_values, P_values, D_values, Q_values))
    total_combinations = len(combos)
    print(f"Testing {total_combinations} parameter combinations...")

    # Load previously-tested combos if partial CSV exists
    tested_combos = set()
    results = []
    best_aic = float('inf')
    best_bic = float('inf')
    best_model_aic = None
    best_model_bic = None

    if os.path.exists(partial_csv):
        try:
            prev = pd.read_csv(partial_csv)
            # Filter for this country
            prev_country = prev[prev['country'] == country]
            for _, row in prev_country.iterrows():
                combo = (int(row['p']), int(row['d']), int(row['q']), int(row['P']), int(row['D']), int(row['Q']))
                tested_combos.add(combo)
                results.append({
                    'order': (int(row['p']), int(row['d']), int(row['q'])),
                    'seasonal_order': (int(row['P']), int(row['D']), int(row['Q']), s),
                    'AIC': float(row['AIC']),
                    'BIC': float(row['BIC']),
                    'fit_time': float(row.get('fit_time', 0.0))
                })
                # Update best trackers
                if float(row['AIC']) < best_aic:
                    best_aic = float(row['AIC'])
                    best_model_aic = combo
                if float(row['BIC']) < best_bic:
                    best_bic = float(row['BIC'])
                    best_model_bic = combo
            print(f"Loaded {len(prev_country)} previously tested combinations for {country} from {partial_csv}")
        except Exception:
            print(f"Warning: could not read existing partial CSV ({partial_csv}). Starting fresh for {country}.")

    tested = len([c for c in tested_combos if c in combos])
    
    # Filter out already-tested combos
    combos_to_test = [c for c in combos if c not in tested_combos]
    
    if not combos_to_test:
        print(f"All {total_combinations} combinations already tested for {country}!")
    else:
        if n_jobs == 1:
            print(f"Running sequential grid search on {len(combos_to_test)} untested combinations (no parallelization)...")
            print(f"Estimated time: ~{len(combos_to_test) * 7 / 60:.1f} minutes at ~7 sec/model")
        else:
            print(f"Running parallel grid search on {len(combos_to_test)} untested combinations using {n_jobs} CPU cores...")
        
        # Parallel execution with progress tracking
        try:
            # Fit models in parallel
            parallel_results = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(fit_sarima_model)(
                    train_data, dev_data, p, d, q, P, D, Q, s, country
                ) for p, d, q, P, D, Q in combos_to_test
            )
            
            # Process results and save to partial CSV
            for result in parallel_results:
                if result['success']:
                    # Append to partial CSV
                    df_row = pd.DataFrame([{
                        'country': result['country'],
                        'p': result['p'], 'd': result['d'], 'q': result['q'],
                        'P': result['P'], 'D': result['D'], 'Q': result['Q'],
                        's': result['s'],
                        'AIC': result['AIC'],
                        'BIC': result['BIC'],
                        'dev_MSE': result['dev_MSE'],
                        'fit_time': result['fit_time']
                    }])
                    header = not os.path.exists(partial_csv)
                    df_row.to_csv(partial_csv, mode='a', index=False, header=header)
                    
                    # Update in-memory results
                    results.append({
                        'order': (result['p'], result['d'], result['q']),
                        'seasonal_order': (result['P'], result['D'], result['Q'], s),
                        'AIC': result['AIC'],
                        'BIC': result['BIC'],
                        'dev_MSE': result['dev_MSE'],
                        'fit_time': result['fit_time']
                    })
                    
                    # Update best trackers
                    combo = (result['p'], result['d'], result['q'], result['P'], result['D'], result['Q'])
                    if result['AIC'] < best_aic:
                        best_aic = result['AIC']
                        best_model_aic = combo
                    if result['BIC'] < best_bic:
                        best_bic = result['BIC']
                        best_model_bic = combo
                    
                    tested += 1
                else:
                    print(f"  Failed: ({result['p']},{result['d']},{result['q']})({result['P']},{result['D']},{result['Q']})")
            
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt received ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â saving progress and exiting gracefully.")
            # Reload from partial CSV to get latest state
            try:
                prev = pd.read_csv(partial_csv)
                prev_country = prev[prev['country'] == country]
                tested = len(prev_country)
            except Exception:
                pass
            print(f"Saved progress: {tested}/{total_combinations} for {country}. You can resume later.")
            # Persist current in-memory bests
            grid_search_results[country] = {
                'all_results': results,
                'best_aic': None,
                'best_bic': None
            }
            # Stop processing further countries
            break

    print(f"\nCompleted: {tested}/{total_combinations} models tested for {country}")

    # If results are empty, try to load from partial CSV
    if len(results) == 0 and os.path.exists(partial_csv):
        prev = pd.read_csv(partial_csv)
        prev_country = prev[prev['country'] == country]
        for _, row in prev_country.iterrows():
            results.append({
                'order': (int(row['p']), int(row['d']), int(row['q'])),
                'seasonal_order': (int(row['P']), int(row['D']), int(row['Q']), s),
                'AIC': float(row['AIC']),
                'BIC': float(row['BIC']),
                'fit_time': float(row.get('fit_time', 0.0))
            })

    # Derive best models from results if available
    if results:
        # Determine best by AIC/BIC/Dev MSE
        best_aic = min(results, key=lambda r: r['AIC'])['AIC']
        best_bic = min(results, key=lambda r: r['BIC'])['BIC']
        # Handle both dict with 'dev_MSE' key and missing key cases
        results_with_dev = [r for r in results if 'dev_MSE' in r or 'fit_time' in r]
        if results_with_dev:
            best_dev_mse = min(results_with_dev, key=lambda r: r.get('dev_MSE', float('inf')))['dev_MSE'] if any('dev_MSE' in r for r in results_with_dev) else None
        else:
            best_dev_mse = None
        best_model_aic = next((r for r in results if r['AIC'] == best_aic), None)
        best_model_bic = next((r for r in results if r['BIC'] == best_bic), None)
        best_model_dev = next((r for r in results_with_dev if 'dev_MSE' in r and r['dev_MSE'] == best_dev_mse), None) if best_dev_mse else None

        # Normalize best model formats
        if isinstance(best_model_aic, dict):
            best_model_aic_tuple = (
                best_model_aic['order'][0], best_model_aic['order'][1], best_model_aic['order'][2],
                best_model_aic['seasonal_order'][0], best_model_aic['seasonal_order'][1], best_model_aic['seasonal_order'][2]
            )
        else:
            best_model_aic_tuple = best_model_aic

        if isinstance(best_model_bic, dict):
            best_model_bic_tuple = (
                best_model_bic['order'][0], best_model_bic['order'][1], best_model_bic['order'][2],
                best_model_bic['seasonal_order'][0], best_model_bic['seasonal_order'][1], best_model_bic['seasonal_order'][2]
            )
        else:
            best_model_bic_tuple = best_model_bic

        if isinstance(best_model_dev, dict):
            best_model_dev_tuple = (
                best_model_dev['order'][0], best_model_dev['order'][1], best_model_dev['order'][2],
                best_model_dev['seasonal_order'][0], best_model_dev['seasonal_order'][1], best_model_dev['seasonal_order'][2]
            )
        else:
            best_model_dev_tuple = best_model_dev

        print(f"\nBest Model (AIC): {best_model_aic_tuple[:3]} x {(best_model_aic_tuple[3:] + (s,))} - AIC: {best_aic:.2f}")
        print(f"Best Model (BIC): {best_model_bic_tuple[:3]} x {(best_model_bic_tuple[3:] + (s,))} - BIC: {best_bic:.2f}")
        if best_dev_mse and best_model_dev_tuple:
            print(f"Best Model (Dev MSE): {best_model_dev_tuple[:3]} x {(best_model_dev_tuple[3:] + (s,))} - Dev MSE: {best_dev_mse:.2f}")
        print(f"[OK] Selected: BIC-based model (recommended for generalization)")

        # Store results
        grid_search_results[country] = {
            'all_results': results,
            'best_aic': {
                'order': best_model_aic_tuple[:3],
                'seasonal_order': best_model_aic_tuple[3:] + (s,),
                'aic': best_aic,
                'bic': next((r['BIC'] for r in results if r['order'] == best_model_aic_tuple[:3] and r['seasonal_order'][:3] == best_model_aic_tuple[3:]), None)
            },
            'best_bic': {
                'order': best_model_bic_tuple[:3],
                'seasonal_order': best_model_bic_tuple[3:] + (s,),
                'aic': next((r['AIC'] for r in results if r['order'] == best_model_bic_tuple[:3] and r['seasonal_order'][:3] == best_model_bic_tuple[3:]), None),
                'bic': best_bic
            }
        }
    else:
        # No results available
        grid_search_results[country] = {
            'all_results': results,
            'best_aic': None,
            'best_bic': None
        }

# VISUALIZE GRID SEARCH RESULTS
print("\n Generating grid search visualization...")

for country in countries:
    # Skip countries not yet processed
    if country not in grid_search_results or not grid_search_results[country]['all_results']:
        print(f"Skipping visualization for {country} (no results yet)")
        continue
    
    results_df = pd.DataFrame(grid_search_results[country]['all_results'])
    results_df['model'] = results_df.apply(
        lambda x: f"({x['order'][0]},{x['order'][1]},{x['order'][2]})({x['seasonal_order'][0]},{x['seasonal_order'][1]},{x['seasonal_order'][2]})", 
        axis=1
    )
    
    # Sort by BIC
    results_df = results_df.sort_values('BIC').reset_index(drop=True)
    
    # Plot top 20 models
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle(f'SARIMA Grid Search Results: {country}', fontsize=16, fontweight='bold')
    
    # AIC comparison
    top_20_aic = results_df.nsmallest(20, 'AIC')
    axes[0].barh(range(len(top_20_aic)), top_20_aic['AIC'].values, alpha=0.7, color='steelblue')
    axes[0].set_yticks(range(len(top_20_aic)))
    axes[0].set_yticklabels(top_20_aic['model'].values, fontsize=8)
    axes[0].set_xlabel('AIC (lower is better)', fontsize=11)
    axes[0].set_title('Top 20 Models by AIC', fontsize=12, fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Highlight best
    best_idx = top_20_aic['AIC'].idxmin()
    axes[0].barh(list(top_20_aic.index).index(best_idx), top_20_aic.loc[best_idx, 'AIC'], 
                 alpha=0.9, color='red', label='Best AIC')
    axes[0].legend()
    
    # BIC comparison
    top_20_bic = results_df.nsmallest(20, 'BIC')
    axes[1].barh(range(len(top_20_bic)), top_20_bic['BIC'].values, alpha=0.7, color='seagreen')
    axes[1].set_yticks(range(len(top_20_bic)))
    axes[1].set_yticklabels(top_20_bic['model'].values, fontsize=8)
    axes[1].set_xlabel('BIC (lower is better)', fontsize=11)
    axes[1].set_title('Top 20 Models by BIC', fontsize=12, fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3, axis='x')
    
    # Highlight best
    best_idx = top_20_bic['BIC'].idxmin()
    axes[1].barh(list(top_20_bic.index).index(best_idx), top_20_bic.loc[best_idx, 'BIC'], 
                 alpha=0.9, color='red', label='Best BIC')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f'phase3_results/grid_search_{country}.png', dpi=300, bbox_inches='tight')
    print(f" Saved: phase3_results/grid_search_{country}.png")
    plt.close()

# TASK 4: DOCUMENT MODEL ORDERS
print("\n Documenting selected model orders...")

# Create summary document
model_selection_summary = {
    'selection_criterion': 'BIC (Bayesian Information Criterion - preferred for larger datasets)',
    'grid_parameters': {
        'p_values': p_values,
        'd_values': d_values,
        'q_values': q_values,
        'P_values': P_values,
        'D_values': D_values,
        'Q_values': Q_values,
        'seasonal_period': s
    },
    'selected_models': {}
}

for country in countries:
    # Skip countries not yet processed or with no best models
    if country not in grid_search_results or not grid_search_results[country]['best_bic']:
        print(f"Skipping summary for {country} (incomplete results)")
        continue
    
    best_bic = grid_search_results[country]['best_bic']
    best_aic = grid_search_results[country]['best_aic']
    
    model_selection_summary['selected_models'][country] = {
        'selected_by_BIC': {
            'order': best_bic['order'],
            'seasonal_order': best_bic['seasonal_order'],
            'AIC': round(best_bic['aic'], 2),
            'BIC': round(best_bic['bic'], 2),
            'model_notation': f"SARIMA{best_bic['order']} x {best_bic['seasonal_order']}"
        },
        'selected_by_AIC': {
            'order': best_aic['order'],
            'seasonal_order': best_aic['seasonal_order'],
            'AIC': round(best_aic['aic'], 2),
            'BIC': round(best_aic['bic'], 2),
            'model_notation': f"SARIMA{best_aic['order']} x {best_aic['seasonal_order']}"
        },
        'data_split': {
            'train_size': data_splits[country]['train_size'],
            'dev_size': data_splits[country]['dev_size'],
            'test_size': data_splits[country]['test_size'],
            'train_period': data_splits[country]['train_period'],
            'dev_period': data_splits[country]['dev_period'],
            'test_period': data_splits[country]['test_period']
        }
    }

# Save to JSON
with open('phase3_results/model_selection_summary.json', 'w') as f:
    json.dump(model_selection_summary, f, indent=2)

print("\n Saved: phase3_results/model_selection_summary.json")

# Create readable summary table
summary_data = []
for country in countries:
    # Skip countries not yet processed
    if country not in grid_search_results or not grid_search_results[country]['best_bic']:
        continue
    
    best_bic = grid_search_results[country]['best_bic']
    summary_data.append({
        'Country': country,
        'Model': f"SARIMA{best_bic['order']} x {best_bic['seasonal_order']}",
        'p': best_bic['order'][0],
        'd': best_bic['order'][1],
        'q': best_bic['order'][2],
        'P': best_bic['seasonal_order'][0],
        'D': best_bic['seasonal_order'][1],
        'Q': best_bic['seasonal_order'][2],
        's': best_bic['seasonal_order'][3],
        'AIC': round(best_bic['aic'], 2),
        'BIC': round(best_bic['bic'], 2)
    })

if summary_data:
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('phase3_results/selected_models.csv', index=False)
    print(" Saved: phase3_results/selected_models.csv")
else:
    print("ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â  No completed countries; selected_models.csv not created yet")

# Print summary
print("\n" + "="*80)
print("MODEL SELECTION SUMMARY (Based on BIC)")
if summary_data:
    print(summary_df.to_string(index=False))
else:
    print("No results yet ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â grid search incomplete or interrupted.")

print("\n" + "="*80)
print("\nGenerated Files:")
print("  1. phase3_results/grid_search_AT.png")
print("  2. phase3_results/grid_search_BE.png")
print("  3. phase3_results/grid_search_BG.png")
print("  4. phase3_results/model_selection_summary.json")
print("  5. phase3_results/selected_models.csv")
print("\nModel Building Phase completed successfully!")
print("Ready for Day-ahead 24-step forecasting with backtesting")
