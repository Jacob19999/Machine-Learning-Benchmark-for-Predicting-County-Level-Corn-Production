"""
Compare Hybrid Model (LLM-adjusted) vs LightGBM/ML baseline for top 10 counties in 2023
Serializes JSON responses and calculates RMSE
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
import sys
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Get project root (parent of Hybrid Model folder)
script_dir = Path(__file__).parent
project_root = script_dir.parent

print("="*80)
print("HYBRID MODEL vs ML BASELINE COMPARISON - 2023")
print("="*80)

# Load consolidated data (prefer preprocessed if available)
data_files = [
    project_root / 'consolidated_data_phase3_preprocessed.csv',  # Prefer preprocessed
    project_root / 'consolidated_data_phase3.csv',
    project_root / 'consolidated_all_data.csv'
]
df = None
is_preprocessed = False

for file in data_files:
    if file.exists():
        df = pd.read_csv(file)
        is_preprocessed = 'preprocessed' in file.name.lower()
        print(f"\nLoaded data from {file}")
        if is_preprocessed:
            print("  (Using preprocessed data with engineered features)")
        break

if df is None:
    raise FileNotFoundError("No consolidated data file found")

# Aggregate by county to find top 10
if 'month' in df.columns:
    county_yearly = df.groupby(['fips', 'county_name', 'year'])['corn_production_bu'].sum().reset_index()
else:
    county_yearly = df[['fips', 'county_name', 'year', 'corn_production_bu']].copy()

# Find top 10 counties by total production
all_counties = county_yearly.groupby(['fips', 'county_name'])['corn_production_bu'].sum().reset_index()
top_counties = all_counties.sort_values('corn_production_bu', ascending=False).head(10)

print(f"\nTop 10 Counties:")
print("-" * 80)
for idx, row in top_counties.iterrows():
    rank = idx + 1
    print(f"  {rank}. {row['county_name']} (FIPS: {int(row['fips'])})")
print("-" * 80)

# Get 2023 data for top 10 counties
target_year = 2023
top_10_fips = top_counties['fips'].tolist()

# Filter for 2023 data
data_2023 = county_yearly[
    (county_yearly['year'] == target_year) & 
    (county_yearly['fips'].isin(top_10_fips))
].copy()

if len(data_2023) == 0:
    print(f"\n⚠ Warning: No data found for year {target_year}")
    print("Available years:", sorted(county_yearly['year'].unique()))
    # Use most recent year available
    latest_year = county_yearly['year'].max()
    print(f"Using latest available year: {latest_year}")
    target_year = latest_year
    data_2023 = county_yearly[
        (county_yearly['year'] == target_year) & 
        (county_yearly['fips'].isin(top_10_fips))
    ].copy()

print(f"\nUsing year: {target_year}")
print(f"Found {len(data_2023)} counties with data for {target_year}")

# Get actual values
actual_values = data_2023.set_index('fips')['corn_production_bu'].to_dict()

# Load ML model (try LightGBM first, then XGBoost)
ml_model = None
model_name = None
model_paths = [
    project_root / 'lightgbm_best_model.pkl',
    project_root / 'xgboost_best_model.pkl',
    project_root / 'best_model.pkl'
]

for model_path in model_paths:
    if model_path.exists():
        print(f"\nLoading ML model from {model_path}...")
        with open(model_path, 'rb') as f:
            ml_model = pickle.load(f)
        model_name = model_path.stem.replace('_best_model', '').replace('_', ' ').title()
        print(f"  Model loaded: {model_name}")
        break

if ml_model is None:
    print("\n⚠ Warning: No ML model found. Will need to load predictions manually.")
    print("Searched for:")
    for p in model_paths:
        print(f"  - {p}")

# Load scaler if exists
scaler = None
scaler_path = project_root / 'robust_scaler_phase3.pkl'
if scaler_path.exists():
    print(f"\nLoading scaler from {scaler_path}...")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print("  Scaler loaded")

# Prepare features for ML prediction
# This is a simplified version - you may need to adjust based on your feature engineering
print("\n" + "="*80)
print("PREPARING FEATURES FOR ML PREDICTION")
print("="*80)

# Get feature columns (exclude target and metadata)
exclude_cols = ['corn_production_bu', 'fips', 'county_name', 'year', 'month']
feature_cols = [col for col in df.columns if col not in exclude_cols]

print(f"Found {len(feature_cols)} feature columns")

# Filter data for 2023 and top 10 counties
if 'month' in df.columns:
    # If monthly data, aggregate to yearly
    df_2023 = df[(df['year'] == target_year) & (df['fips'].isin(top_10_fips))].copy()
    # Aggregate by county (take mean or sum depending on feature)
    numeric_cols = df_2023.select_dtypes(include=[np.number]).columns.tolist()
    # Build aggregation dict, excluding groupby columns
    agg_dict = {}
    for col in numeric_cols:
        if col not in ['fips', 'county_name', 'year']:
            if col == 'corn_production_bu':
                agg_dict[col] = 'sum'  # Sum production
            elif col in feature_cols:
                agg_dict[col] = 'mean'
            else:
                agg_dict[col] = 'first'
    # Use as_index=False to keep fips and county_name as columns
    df_2023_agg = df_2023.groupby(['fips', 'county_name'], as_index=False).agg(agg_dict)
    # Add back year column
    df_2023_agg['year'] = target_year
else:
    df_2023_agg = df[(df['year'] == target_year) & (df['fips'].isin(top_10_fips))].copy()

# Ensure we have all top 10 counties
missing_fips = set(top_10_fips) - set(df_2023_agg['fips'].unique())
if missing_fips:
    print(f"\n⚠ Warning: Missing data for FIPS: {missing_fips}")
    # Use most recent available year for missing counties
    for fips in missing_fips:
        county_data = df[df['fips'] == fips].copy()
        if len(county_data) > 0:
            latest_year = county_data['year'].max()
            latest_data = county_data[county_data['year'] == latest_year]
            if 'month' in latest_data.columns:
                latest_data = latest_data.groupby(['fips', 'county_name'], as_index=False).agg(agg_dict)
                latest_data['year'] = latest_year
            df_2023_agg = pd.concat([df_2023_agg, latest_data], ignore_index=True)
            print(f"  Using {latest_year} data for FIPS {fips}")

# Sort by FIPS to match top_counties order
df_2023_agg = df_2023_agg.sort_values('fips').reset_index(drop=True)

# Get expected feature names from scaler if available
expected_features = None
if scaler is not None:
    try:
        # Try to get feature names from scaler
        if hasattr(scaler, 'feature_names_in_'):
            expected_features = list(scaler.feature_names_in_)
            print(f"  Scaler expects {len(expected_features)} features")
    except:
        pass

# Prepare feature matrix
if expected_features is not None and not is_preprocessed:
    # Use expected features from scaler (only if not using preprocessed data)
    print("  Aligning features with scaler expectations...")
    X_2023 = pd.DataFrame(index=df_2023_agg.index)
    
    # Add features that exist in data
    for feat in expected_features:
        if feat in df_2023_agg.columns:
            X_2023[feat] = df_2023_agg[feat]
        else:
            # Feature needs to be engineered
            X_2023[feat] = 0.0  # Placeholder, will be filled below
    
    # Engineer missing features
    START_YEAR = df['year'].min() if 'year' in df.columns else 2000
    
    # Temporal features
    if 'year_trend' in expected_features and 'year' in df_2023_agg.columns:
        X_2023['year_trend'] = df_2023_agg['year'] - START_YEAR
    if 'month_sin' in expected_features:
        if 'month' in df_2023_agg.columns:
            X_2023['month_sin'] = np.sin(2 * np.pi * df_2023_agg['month'] / 12)
        else:
            X_2023['month_sin'] = 0.0  # Default for yearly data
    if 'month_cos' in expected_features:
        if 'month' in df_2023_agg.columns:
            X_2023['month_cos'] = np.cos(2 * np.pi * df_2023_agg['month'] / 12)
        else:
            X_2023['month_cos'] = 1.0  # Default for yearly data
    
    # Soil moisture features
    soil_moisture_cols = [col for col in df_2023_agg.columns if 'SoilMoi' in col]
    if 'soil_moisture_avg' in expected_features and len(soil_moisture_cols) > 0:
        X_2023['soil_moisture_avg'] = df_2023_agg[soil_moisture_cols].mean(axis=1)
    if 'soil_moisture_gradient' in expected_features:
        if 'SoilMoi0_10cm_inst' in df_2023_agg.columns and 'SoilMoi100_200cm_inst' in df_2023_agg.columns:
            X_2023['soil_moisture_gradient'] = (
                df_2023_agg['SoilMoi100_200cm_inst'] - df_2023_agg['SoilMoi0_10cm_inst']
            )
        else:
            X_2023['soil_moisture_gradient'] = 0.0
    
    # Temperature features
    temp_cols = [col for col in df_2023_agg.columns if 'Temp' in col or 'Tair' in col or 'Tveg' in col or 'tmean' in col.lower() or 'tmin' in col.lower() or 'tmax' in col.lower()]
    if 'temperature_avg' in expected_features and len(temp_cols) > 0:
        X_2023['temperature_avg'] = df_2023_agg[temp_cols].mean(axis=1)
    if 'temperature_range' in expected_features:
        if 'prism_tmax_degf' in df_2023_agg.columns and 'prism_tmin_degf' in df_2023_agg.columns:
            X_2023['temperature_range'] = df_2023_agg['prism_tmax_degf'] - df_2023_agg['prism_tmin_degf']
        else:
            X_2023['temperature_range'] = 0.0
    
    # Water balance features
    if 'precipitation_evap_balance' in expected_features:
        if 'prism_ppt_in' in df_2023_agg.columns and 'Evap_tavg' in df_2023_agg.columns:
            X_2023['precipitation_evap_balance'] = df_2023_agg['prism_ppt_in'] - df_2023_agg['Evap_tavg']
        else:
            X_2023['precipitation_evap_balance'] = 0.0
    if 'precipitation_efficiency' in expected_features:
        min_eps = 1e-8
        if 'prism_ppt_in' in df_2023_agg.columns and 'SoilMoi0_10cm_inst' in df_2023_agg.columns:
            X_2023['precipitation_efficiency'] = df_2023_agg['SoilMoi0_10cm_inst'] / (df_2023_agg['prism_ppt_in'] + min_eps)
        else:
            X_2023['precipitation_efficiency'] = 0.0
    
    # Agricultural context features
    if 'yield_per_acre' in expected_features:
        min_eps = 1e-8
        if 'corn_acres_planted' in df_2023_agg.columns and 'corn_production_bu' in df_2023_agg.columns:
            X_2023['yield_per_acre'] = df_2023_agg['corn_production_bu'] / (df_2023_agg['corn_acres_planted'] + min_eps)
        else:
            X_2023['yield_per_acre'] = 0.0
    if 'fuel_cost_proxy' in expected_features:
        if 'diesel_usd_gal' in df_2023_agg.columns and 'corn_acres_planted' in df_2023_agg.columns:
            X_2023['fuel_cost_proxy'] = df_2023_agg['diesel_usd_gal'] * df_2023_agg['corn_acres_planted']
        else:
            X_2023['fuel_cost_proxy'] = 0.0
    
    # Economic features
    if 'total_revenue_sources' in expected_features:
        if 'income_farmrelated_receipts_total_usd' in df_2023_agg.columns and 'govt_programs_federal_receipts_usd' in df_2023_agg.columns:
            X_2023['total_revenue_sources'] = (
                df_2023_agg['income_farmrelated_receipts_total_usd'] + 
                df_2023_agg['govt_programs_federal_receipts_usd']
            )
        else:
            X_2023['total_revenue_sources'] = 0.0
    if 'revenue_per_bushel' in expected_features:
        min_eps = 1e-8
        if 'income_farmrelated_receipts_total_usd' in df_2023_agg.columns and 'corn_production_bu' in df_2023_agg.columns:
            X_2023['revenue_per_bushel'] = df_2023_agg['income_farmrelated_receipts_total_usd'] / (df_2023_agg['corn_production_bu'] + min_eps)
        else:
            X_2023['revenue_per_bushel'] = 0.0
    
    # Ensure all expected features are present
    for feat in expected_features:
        if feat not in X_2023.columns:
            X_2023[feat] = 0.0
    
    # Reorder to match scaler expectations
    X_2023 = X_2023[expected_features]
elif expected_features is not None and is_preprocessed:
    # Preprocessed data - just align columns with expected features
    print("  Aligning preprocessed features with scaler expectations...")
    X_2023 = pd.DataFrame(index=df_2023_agg.index)
    for feat in expected_features:
        if feat in df_2023_agg.columns:
            X_2023[feat] = df_2023_agg[feat]
        else:
            X_2023[feat] = 0.0  # Missing feature
    X_2023 = X_2023[expected_features]
else:
    # Use available feature columns
    X_2023 = df_2023_agg[feature_cols].copy()

# Handle missing values
X_2023 = X_2023.fillna(X_2023.median())

# Scale if scaler available
if scaler is not None:
    print("  Scaling features...")
    try:
        X_2023_scaled = scaler.transform(X_2023)
        X_2023 = pd.DataFrame(X_2023_scaled, columns=X_2023.columns, index=X_2023.index)
    except Exception as e:
        print(f"  ⚠ Warning: Scaling failed: {e}")
        print("  Continuing without scaling...")

# Get ML predictions
ml_predictions = {}
if ml_model is not None:
    print("\nGenerating ML predictions...")
    try:
        y_pred_log = ml_model.predict(X_2023)
        # Check if predictions are in log scale
        if np.all(y_pred_log < 20):  # Likely log scale
            y_pred = np.expm1(y_pred_log)
        else:
            y_pred = y_pred_log
        
        for idx, row in df_2023_agg.iterrows():
            fips = int(row['fips'])
            ml_predictions[fips] = float(y_pred[idx])
        
        print(f"  Generated {len(ml_predictions)} ML predictions")
    except Exception as e:
        print(f"  ⚠ Error generating ML predictions: {e}")
        print("  Will use placeholder values")
        ml_predictions = {int(fips): 0.0 for fips in top_10_fips}

# Load JSON responses from Grok (if they exist)
print("\n" + "="*80)
print("LOADING JSON RESPONSES FROM GROK")
print("="*80)

json_responses = {}
json_dir = script_dir / "grok_responses"
json_dir.mkdir(exist_ok=True)

# Look for JSON files
json_files = list(json_dir.glob("*.json")) + list(script_dir.glob("grok_response_*.json"))

if json_files:
    print(f"Found {len(json_files)} JSON file(s)")
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Handle both single county and batch responses
                if isinstance(data, dict):
                    if 'fips' in data or '27001' in str(data):  # Single county or batch
                        if 'fips' in data:
                            # Single county response
                            fips = int(data.get('fips', 0))
                            if fips in top_10_fips:
                                json_responses[fips] = data
                        else:
                            # Batch response with FIPS keys
                            for fips_str, county_data in data.items():
                                try:
                                    fips = int(fips_str)
                                    if fips in top_10_fips:
                                        json_responses[fips] = county_data
                                except ValueError:
                                    continue
            print(f"  Loaded: {json_file.name}")
        except Exception as e:
            print(f"  ⚠ Error loading {json_file.name}: {e}")
else:
    print("  No JSON response files found")
    print(f"  Looking in: {json_dir}")
    print("  Please add JSON responses from Grok to compare")

# Extract hybrid predictions from JSON
hybrid_predictions = {}
for fips in top_10_fips:
    if fips in json_responses:
        resp = json_responses[fips]
        # Try different possible keys
        pred = resp.get('pred_yield_bu_ac', resp.get('pred_yield_total_bu', resp.get('y_hyb', None)))
        if pred is None:
            # Try to calculate from adjustment
            ml_baseline = ml_predictions.get(fips, 0)
            adj = resp.get('adj', resp.get('adjustment', 0))
            if ml_baseline > 0:
                pred = ml_baseline * (1 + adj)
        if pred is not None:
            hybrid_predictions[fips] = float(pred)
    else:
        # No JSON response - use ML baseline
        hybrid_predictions[fips] = ml_predictions.get(fips, 0.0)

# Serialize JSON responses to a single file
output_json_path = script_dir / f"grok_responses_2023_top10.json"
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(json_responses, f, indent=2, ensure_ascii=False)
print(f"\nSerialized JSON responses to: {output_json_path}")

# Prepare comparison DataFrame
comparison_data = []
for fips in top_10_fips:
    county_name = top_counties[top_counties['fips'] == fips]['county_name'].iloc[0]
    actual = actual_values.get(fips, np.nan)
    ml_pred = ml_predictions.get(fips, np.nan)
    hybrid_pred = hybrid_predictions.get(fips, np.nan)
    
    comparison_data.append({
        'fips': fips,
        'county_name': county_name,
        'actual': actual,
        'ml_pred': ml_pred,
        'hybrid_pred': hybrid_pred,
        'ml_error': actual - ml_pred if not np.isnan(actual) and not np.isnan(ml_pred) else np.nan,
        'hybrid_error': actual - hybrid_pred if not np.isnan(actual) and not np.isnan(hybrid_pred) else np.nan,
        'ml_abs_error': abs(actual - ml_pred) if not np.isnan(actual) and not np.isnan(ml_pred) else np.nan,
        'hybrid_abs_error': abs(actual - hybrid_pred) if not np.isnan(actual) and not np.isnan(hybrid_pred) else np.nan,
    })

comparison_df = pd.DataFrame(comparison_data)

# Calculate RMSE
print("\n" + "="*80)
print("RMSE COMPARISON")
print("="*80)

# Filter out NaN values for RMSE calculation
ml_valid = comparison_df[['actual', 'ml_pred']].dropna()
hybrid_valid = comparison_df[['actual', 'hybrid_pred']].dropna()

if len(ml_valid) > 0:
    ml_rmse = np.sqrt(mean_squared_error(ml_valid['actual'], ml_valid['ml_pred']))
    print(f"\n{model_name or 'ML'} RMSE: {ml_rmse:,.0f} bushels")
    print(f"  (Based on {len(ml_valid)} counties)")
else:
    ml_rmse = np.nan
    print(f"\n{model_name or 'ML'} RMSE: N/A (no valid predictions)")

if len(hybrid_valid) > 0:
    hybrid_rmse = np.sqrt(mean_squared_error(hybrid_valid['actual'], hybrid_valid['hybrid_pred']))
    print(f"Hybrid Model RMSE: {hybrid_rmse:,.0f} bushels")
    print(f"  (Based on {len(hybrid_valid)} counties)")
else:
    hybrid_rmse = np.nan
    print("Hybrid Model RMSE: N/A (no valid predictions)")

if not np.isnan(ml_rmse) and not np.isnan(hybrid_rmse):
    improvement = ((ml_rmse - hybrid_rmse) / ml_rmse) * 100
    print(f"\nImprovement: {improvement:+.2f}%")

# Display detailed comparison
print("\n" + "="*80)
print("DETAILED COMPARISON BY COUNTY")
print("="*80)
print(comparison_df.to_string(index=False))

# Save comparison to CSV
output_csv_path = script_dir / f"comparison_2023_top10.csv"
comparison_df.to_csv(output_csv_path, index=False)
print(f"\nSaved comparison to: {output_csv_path}")

# Save summary
summary = {
    'year': target_year,
    'model_name': model_name or 'Unknown',
    'ml_rmse': float(ml_rmse) if not np.isnan(ml_rmse) else None,
    'hybrid_rmse': float(hybrid_rmse) if not np.isnan(hybrid_rmse) else None,
    'improvement_pct': float(improvement) if not np.isnan(ml_rmse) and not np.isnan(hybrid_rmse) else None,
    'n_counties': len(comparison_df),
    'n_valid_ml': len(ml_valid),
    'n_valid_hybrid': len(hybrid_valid)
}

summary_path = script_dir / f"summary_2023_top10.json"
with open(summary_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)
print(f"Saved summary to: {summary_path}")

print("\n" + "="*80)
print("COMPARISON COMPLETE")
print("="*80)

