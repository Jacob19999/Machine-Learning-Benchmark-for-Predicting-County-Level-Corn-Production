"""
Compare ML model results with Hybrid model results
Calculate R² and RMSE on both log and original scale
Create bar plots for visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import json
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error

# Load ML predictions
ml_df = pd.read_csv('2022_annualized_predictions_by_county.csv')

# Parse hybrid model results from JSON Response.txt
hybrid_file = Path('Hybrid Model/JSON Response/JSON Response.txt')
with open(hybrid_file, 'r') as f:
    content = f.read()

# Extract JSON objects for each county
hybrid_data = {}

# Alternative parsing - extract from text more carefully
hybrid_data = {}
lines = content.split('\n')
current_fips = None
current_county = None

for i, line in enumerate(lines):
    # Look for county header
    if 'County (Rank' in line and 'FIPS' in line:
        match = re.search(r'(\w+\s+County).*FIPS\s+(\d+)', line)
        if match:
            current_county = match.group(1)
            current_fips = int(match.group(2))
    
    # Look for pred_production_bu
    if current_fips and '"pred_production_bu":' in line:
        match = re.search(r'"pred_production_bu":\s*(\d+)', line)
        if match:
            if current_fips not in hybrid_data:
                hybrid_data[current_fips] = {}
            hybrid_data[current_fips]['county_name'] = current_county
            hybrid_data[current_fips]['pred_production_bu'] = int(match.group(1))

print(f"Found {len(hybrid_data)} counties in hybrid model results")
print("Hybrid FIPS codes:", sorted(hybrid_data.keys()))

# Filter ML data to top 10 counties (those in hybrid model)
top_10_fips = list(hybrid_data.keys())
ml_top10 = ml_df[ml_df['FIPS'].isin(top_10_fips)].copy()

# Merge hybrid predictions
ml_top10['Hybrid_Prediction'] = ml_top10['FIPS'].map(
    lambda x: hybrid_data[x]['pred_production_bu'] if x in hybrid_data else np.nan
)

# Get actual values
actual = ml_top10['Actual_Production_2022_Bushels'].values

# Models to compare (using LightGBM as baseline ML model)
models_to_compare = {
    'LightGBM': 'LightGBM_2022_Bushels',
    'Hybrid': 'Hybrid_Prediction'
}

# Calculate metrics for each model
results = []

for model_name, col_name in models_to_compare.items():
    pred = ml_top10[col_name].values
    
    # Remove NaN values
    mask = ~np.isnan(pred) & ~np.isnan(actual) & (actual > 0) & (pred > 0)
    y_true = actual[mask]
    y_pred = pred[mask]
    
    if len(y_true) == 0:
        continue
    
    # Original scale metrics
    r2_orig = r2_score(y_true, y_pred)
    rmse_orig = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Log scale metrics
    y_true_log = np.log(y_true)
    y_pred_log = np.log(y_pred)
    r2_log = r2_score(y_true_log, y_pred_log)
    rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    
    results.append({
        'Model': model_name,
        'R2_Original': r2_orig,
        'RMSE_Original': rmse_orig,
        'R2_Log': r2_log,
        'RMSE_Log': rmse_log
    })
    
    print(f"\n{model_name}:")
    print(f"  R² (Original): {r2_orig:.4f}")
    print(f"  RMSE (Original): {rmse_orig:,.0f} bushels")
    print(f"  R² (Log): {r2_log:.4f}")
    print(f"  RMSE (Log): {rmse_log:.4f}")

results_df = pd.DataFrame(results)

# Create bar plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('ML Model vs Hybrid Model Comparison (Top 10 Counties, 2022)', fontsize=16, fontweight='bold')

# R² Original Scale
ax1 = axes[0, 0]
x_pos = np.arange(len(results_df))
bars1 = ax1.bar(x_pos, results_df['R2_Original'], color=['#3498db', '#e74c3c'], alpha=0.7)
ax1.set_xlabel('Model', fontsize=12)
ax1.set_ylabel('R²', fontsize=12)
ax1.set_title('R² Score (Original Scale)', fontsize=13, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(results_df['Model'])
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim([0, max(results_df['R2_Original']) * 1.2])
for i, (bar, val) in enumerate(zip(bars1, results_df['R2_Original'])):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.4f}', ha='center', va='bottom', fontweight='bold')

# RMSE Original Scale
ax2 = axes[0, 1]
bars2 = ax2.bar(x_pos, results_df['RMSE_Original'], color=['#3498db', '#e74c3c'], alpha=0.7)
ax2.set_xlabel('Model', fontsize=12)
ax2.set_ylabel('RMSE (Bushels)', fontsize=12)
ax2.set_title('RMSE (Original Scale)', fontsize=13, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(results_df['Model'])
ax2.grid(axis='y', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars2, results_df['RMSE_Original'])):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(results_df['RMSE_Original']) * 0.02,
             f'{val:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# R² Log Scale
ax3 = axes[1, 0]
bars3 = ax3.bar(x_pos, results_df['R2_Log'], color=['#3498db', '#e74c3c'], alpha=0.7)
ax3.set_xlabel('Model', fontsize=12)
ax3.set_ylabel('R²', fontsize=12)
ax3.set_title('R² Score (Log Scale)', fontsize=13, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(results_df['Model'])
ax3.grid(axis='y', alpha=0.3)
ax3.set_ylim([0, max(results_df['R2_Log']) * 1.2])
for i, (bar, val) in enumerate(zip(bars3, results_df['R2_Log'])):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.4f}', ha='center', va='bottom', fontweight='bold')

# RMSE Log Scale
ax4 = axes[1, 1]
bars4 = ax4.bar(x_pos, results_df['RMSE_Log'], color=['#3498db', '#e74c3c'], alpha=0.7)
ax4.set_xlabel('Model', fontsize=12)
ax4.set_ylabel('RMSE (Log Scale)', fontsize=12)
ax4.set_title('RMSE (Log Scale)', fontsize=13, fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(results_df['Model'])
ax4.grid(axis='y', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars4, results_df['RMSE_Log'])):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(results_df['RMSE_Log']) * 0.02,
             f'{val:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('ml_vs_hybrid_comparison.png', dpi=300, bbox_inches='tight')
print(f"\nSaved comparison plot to: ml_vs_hybrid_comparison.png")

# Save results to CSV
results_df.to_csv('ml_vs_hybrid_metrics.csv', index=False)
print(f"Saved metrics to: ml_vs_hybrid_metrics.csv")

# Also create a detailed comparison by county
county_comparison = ml_top10[['FIPS', 'County_Name', 'Actual_Production_2022_Bushels', 
                              'LightGBM_2022_Bushels', 'Hybrid_Prediction']].copy()
county_comparison['LightGBM_Error'] = abs(county_comparison['Actual_Production_2022_Bushels'] - 
                                          county_comparison['LightGBM_2022_Bushels'])
county_comparison['Hybrid_Error'] = abs(county_comparison['Actual_Production_2022_Bushels'] - 
                                        county_comparison['Hybrid_Prediction'])
county_comparison['LightGBM_Error_Pct'] = (county_comparison['LightGBM_Error'] / 
                                           county_comparison['Actual_Production_2022_Bushels'] * 100)
county_comparison['Hybrid_Error_Pct'] = (county_comparison['Hybrid_Error'] / 
                                         county_comparison['Actual_Production_2022_Bushels'] * 100)

county_comparison.to_csv('ml_vs_hybrid_by_county.csv', index=False)
print(f"Saved county-by-county comparison to: ml_vs_hybrid_by_county.csv")

