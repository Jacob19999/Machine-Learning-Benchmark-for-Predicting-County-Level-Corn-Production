"""
Compare ML models with Hybrid model on top 10 counties
Calculate R^2 and RMSE (original scale) and create bar plots
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error

# Load JSON Response with hybrid predictions
json_path = Path("Hybrid Model/JSON Response/JSON Response.txt")
with open(json_path, 'r') as f:
    hybrid_data = json.load(f)

# Get top 10 counties from JSON (FIPS codes)
top_10_fips = [int(fips) for fips in hybrid_data.keys()]
print(f"Top 10 counties (FIPS): {top_10_fips}")

# Load Excel file with ML model predictions
excel_path = "model_predictions_by_county.xlsx"
df = pd.read_excel(excel_path)
print(f"\nLoaded Excel file: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Columns: {df.columns.tolist()}")

# Filter for top 10 counties
df_top10 = df[df['fips'].isin(top_10_fips)].copy()
print(f"\nFiltered to top 10 counties: {df_top10.shape[0]} rows")

# Check if we have acres planted data in Excel
if 'corn_acres_planted' not in df_top10.columns:
    # Try to load from consolidated data
    print("\nAcres planted not in Excel, checking consolidated data...")
    consolidated_files = [
        'consolidated_data_phase3.csv',
        'consolidated_all_data.csv',
        'consolidated_data_phase3_preprocessed.csv'
    ]
    
    acres_data = None
    for file in consolidated_files:
        if Path(file).exists():
            try:
                df_consolidated = pd.read_csv(file)
                if 'corn_acres_planted' in df_consolidated.columns and 'fips' in df_consolidated.columns:
                    # Get most recent year's acres planted for each county
                    if 'year' in df_consolidated.columns:
                        acres_data = df_consolidated[df_consolidated['fips'].isin(top_10_fips)][
                            ['fips', 'year', 'corn_acres_planted']
                        ].dropna().sort_values('year', ascending=False)
                        # Get most recent year for each county
                        acres_data = acres_data.groupby('fips').first().reset_index()[['fips', 'corn_acres_planted']]
                    else:
                        acres_data = df_consolidated[df_consolidated['fips'].isin(top_10_fips)][
                            ['fips', 'corn_acres_planted']
                        ].dropna()
                    print(f"  Found acres data in {file}: {len(acres_data)} counties")
                    break
            except Exception as e:
                print(f"  Error reading {file}: {e}")
                continue
    
    if acres_data is not None and len(acres_data) > 0:
        # Merge acres data
        df_top10 = df_top10.merge(acres_data, on='fips', how='left')
        print(f"  Merged acres data: {df_top10['corn_acres_planted'].notna().sum()} counties have acres data")
    else:
        # Estimate acres from actual production / typical yield (180 bu/ac)
        print("  No acres data found, estimating from actual production...")
        df_top10['corn_acres_planted'] = df_top10['Actual_Corn_Production_Bushels'] / 180.0
        print("  Estimated acres using 180 bu/ac average yield")

# Aggregate by county and year (if there are multiple months)
if 'month' in df_top10.columns:
    df_agg = df_top10.groupby(['fips', 'county_name', 'year']).agg({
        'Actual_Corn_Production_Bushels': 'first',
        'Polynomial_Regression_Prediction': 'first',
        'SVM_Prediction': 'first',
        'Random_Forest_Prediction': 'first',
        'XGBoost_Prediction': 'first',
        'LightGBM_Prediction': 'first',
        'TabNet_Prediction': 'first',
        'Temporal_NN_LSTM_Prediction': 'first',
        'TCN_Prediction': 'first',
        'corn_acres_planted': 'first'
    }).reset_index()
else:
    df_agg = df_top10.copy()

print(f"\nAfter aggregation: {df_agg.shape[0]} county-year combinations")

# Get actual values
y_actual = df_agg['Actual_Corn_Production_Bushels'].values

# Prepare model predictions dictionary
model_predictions = {}

# ML models from Excel
ml_models = {
    'Polynomial Regression': 'Polynomial_Regression_Prediction',
    'SVM': 'SVM_Prediction',
    'Random Forest': 'Random_Forest_Prediction',
    'XGBoost': 'XGBoost_Prediction',
    'LightGBM': 'LightGBM_Prediction',
    'TabNet': 'TabNet_Prediction',
    'LSTM': 'Temporal_NN_LSTM_Prediction',
    'TCN': 'TCN_Prediction'
}

for model_name, col_name in ml_models.items():
    if col_name in df_agg.columns:
        pred = df_agg[col_name].values
        # Remove NaN values for calculation
        mask = ~np.isnan(pred) & ~np.isnan(y_actual)
        if mask.sum() > 0:
            model_predictions[model_name] = {
                'predictions': pred[mask],
                'actual': y_actual[mask]
            }

# Add Hybrid model predictions
# Convert from bu/ac to total bushels using acres planted for each year
hybrid_preds = []
hybrid_actuals = []

for fips in top_10_fips:
    fips_str = str(fips)
    if fips_str in hybrid_data:
        # Get hybrid prediction in bu/ac
        hybrid_yield_bu_ac = hybrid_data[fips_str]['pred_yield_bu_ac']
        
        # Get data for this county (all years)
        county_data = df_agg[df_agg['fips'] == fips]
        if len(county_data) > 0:
            # For each year, use that year's acres planted
            for idx, row in county_data.iterrows():
                # Try to get acres planted for this specific year
                if 'corn_acres_planted' in row and pd.notna(row['corn_acres_planted']) and row['corn_acres_planted'] > 0:
                    acres = row['corn_acres_planted']
                else:
                    # Estimate from actual production for this year
                    actual_prod = row['Actual_Corn_Production_Bushels']
                    if actual_prod > 0:
                        # Estimate acres from actual production (assuming ~180 bu/ac)
                        acres = actual_prod / 180.0
                    else:
                        # Use average from other years for this county
                        avg_acres = county_data['corn_acres_planted'].mean() if 'corn_acres_planted' in county_data.columns else None
                        if avg_acres is None or np.isnan(avg_acres) or avg_acres <= 0:
                            avg_acres = county_data['Actual_Corn_Production_Bushels'].mean() / 180.0
                        acres = avg_acres
                
                # Convert to total bushels
                hybrid_total_bu = hybrid_yield_bu_ac * acres
                
                hybrid_preds.append(hybrid_total_bu)
                hybrid_actuals.append(row['Actual_Corn_Production_Bushels'])

if len(hybrid_preds) > 0:
    hybrid_preds = np.array(hybrid_preds)
    hybrid_actuals = np.array(hybrid_actuals)
    mask = ~np.isnan(hybrid_preds) & ~np.isnan(hybrid_actuals)
    if mask.sum() > 0:
        model_predictions['Hybrid'] = {
            'predictions': hybrid_preds[mask],
            'actual': hybrid_actuals[mask]
        }

# Calculate R^2 and RMSE for each model
results = []
for model_name, data in model_predictions.items():
    y_pred = data['predictions']
    y_true = data['actual']
    
    # Calculate metrics on original scale
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    results.append({
        'Model': model_name,
        'R2': r2,
        'RMSE': rmse,
        'n_samples': len(y_true)
    })
    
    print(f"\n{model_name}:")
    print(f"  R² = {r2:.4f}")
    print(f"  RMSE = {rmse:,.0f} bushels")
    print(f"  n = {len(y_true)}")

# Create results DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('R2', ascending=False)

print(f"\n{'='*80}")
print("RESULTS SUMMARY")
print(f"{'='*80}")
print(results_df.to_string(index=False))

# Create bar plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# R² bar plot
models = results_df['Model'].values
r2_values = results_df['R2'].values
# Use different colors - highlight Hybrid model
colors = ['#2E86AB' if m == 'Hybrid' else plt.cm.viridis(i/len(models)) 
          for i, m in enumerate(models)]

bars1 = ax1.bar(range(len(models)), r2_values, color=colors, edgecolor='black', linewidth=1.2)
ax1.set_xlabel('Model', fontsize=13, fontweight='bold')
ax1.set_ylabel('R² Score', fontsize=13, fontweight='bold')
ax1.set_title('R² Comparison: ML Models vs Hybrid Model', fontsize=15, fontweight='bold', pad=20)
ax1.set_xticks(range(len(models)))
ax1.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.axhline(y=0, color='black', linewidth=0.8)

# Set y-axis limits to show negative values if needed
y_min = min(r2_values) * 1.1 if min(r2_values) < 0 else 0
y_max = max(r2_values) * 1.15 if max(r2_values) > 0 else 0.1
ax1.set_ylim([y_min, y_max])

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars1, r2_values)):
    height = bar.get_height()
    y_pos = height + (y_max - y_min) * 0.02 if height >= 0 else height - (y_max - y_min) * 0.02
    ax1.text(bar.get_x() + bar.get_width()/2., y_pos,
             f'{val:.3f}',
             ha='center', va='bottom' if height >= 0 else 'top', fontsize=9, fontweight='bold')

# RMSE bar plot
rmse_values = results_df['RMSE'].values
bars2 = ax2.bar(range(len(models)), rmse_values, color=colors, edgecolor='black', linewidth=1.2)
ax2.set_xlabel('Model', fontsize=13, fontweight='bold')
ax2.set_ylabel('RMSE (bushels)', fontsize=13, fontweight='bold')
ax2.set_title('RMSE Comparison: ML Models vs Hybrid Model (Original Scale)', fontsize=15, fontweight='bold', pad=20)
ax2.set_xticks(range(len(models)))
ax2.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars (rotated for readability)
for i, (bar, val) in enumerate(zip(bars2, rmse_values)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{val/1e6:.2f}M',
             ha='center', va='bottom', fontsize=9, fontweight='bold', rotation=0)

plt.tight_layout(pad=3.0)
plt.savefig('model_comparison_r2_rmse.png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nSaved plot to: model_comparison_r2_rmse.png")

# Save results to CSV
results_df.to_csv('model_comparison_results.csv', index=False)
print(f"Saved results to: model_comparison_results.csv")

plt.show()

