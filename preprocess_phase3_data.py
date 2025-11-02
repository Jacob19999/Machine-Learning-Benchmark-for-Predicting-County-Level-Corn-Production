# ============================================================================
# PHASE 3 DATA PREPROCESSING AND FEATURE ENGINEERING
# ============================================================================
# This script:
# 1. Analyzes the consolidated Phase 3 dataset
# 2. Performs data cleaning and quality checks
# 3. Implements scaling strategies
# 4. Creates feature interactions
# 5. Outputs a training-ready dataset
# ============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PHASE 3 DATA PREPROCESSING AND FEATURE ENGINEERING")
print("="*80)

# ============================================================================
# 1. LOAD AND INITIAL ANALYSIS
# ============================================================================
print("\n[1/7] Loading consolidated Phase 3 dataset...")
df = pd.read_csv("consolidated_data_phase3.csv")
print(f"  [OK] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

print("\n" + "="*80)
print("INITIAL DATA ANALYSIS")
print("="*80)
print(f"Shape: {df.shape}")
print(f"Columns: {len(df.columns)}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\n--- Missing Data Summary ---")
missing_summary = df.isnull().sum()
missing_pct = (missing_summary / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'Column': missing_summary.index,
    'Missing_Count': missing_summary.values,
    'Missing_Percent': missing_pct.values
})
missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
print(missing_df.to_string(index=False))

print("\n--- Target Variable Analysis ---")
if 'corn_production_bu' in df.columns:
    target = df['corn_production_bu']
    print(f"Target: corn_production_bu")
    print(f"  Non-null: {target.notna().sum()} ({100*target.notna().sum()/len(df):.1f}%)")
    print(f"  Range: {target.min():,.0f} - {target.max():,.0f} bushels")
    print(f"  Mean: {target.mean():,.0f}")
    print(f"  Median: {target.median():,.0f}")
    print(f"  Std: {target.std():,.0f}")

print("\n--- Data Types ---")
print(df.dtypes.value_counts())

print("\n--- Temporal Coverage ---")
if 'year' in df.columns:
    print(f"Year range: {df['year'].min()} - {df['year'].max()}")
    print(f"Unique years: {sorted(df['year'].unique())}")
    if 'month' in df.columns:
        print(f"Has monthly data: Yes")
        print(f"Unique months: {sorted(df['month'].dropna().unique())}")

# ============================================================================
# 2. DATA CLEANING
# ============================================================================
print("\n[2/7] Data cleaning...")

df_clean = df.copy()

print("  [OK] Removing rows with missing target variable...")
if 'corn_production_bu' in df_clean.columns:
    before = len(df_clean)
    df_clean = df_clean.dropna(subset=['corn_production_bu'])
    after = len(df_clean)
    print(f"    Removed {before - after} rows ({100*(before-after)/before:.1f}%)")

print("  [OK] Removing rows with zero production...")
if 'corn_production_bu' in df_clean.columns:
    before = len(df_clean)
    df_clean = df_clean[df_clean['corn_production_bu'] > 0]
    after = len(df_clean)
    print(f"    Removed {before - after} rows ({100*(before-after)/before:.1f}%)")

print("  [OK] Handling infinite values...")
inf_cols = []
for col in df_clean.select_dtypes(include=[np.number]).columns:
    if np.isinf(df_clean[col]).any():
        inf_cols.append(col)
        df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
if inf_cols:
    print(f"    Fixed infinite values in: {inf_cols}")
else:
    print("    No infinite values found")

print("  [OK] Identifying feature columns...")
# Separate ID, target, and feature columns
ID_COLS = ['fips', 'county_name', 'year', 'month']
TARGET_COL = 'corn_production_bu'

# All other numeric columns are features
feature_cols = [col for col in df_clean.columns 
                if col not in ID_COLS + [TARGET_COL] 
                and df_clean[col].dtype in [np.float64, np.int64]]

print(f"    Feature columns: {len(feature_cols)}")
print(f"    ID columns: {ID_COLS}")
print(f"    Target column: {TARGET_COL}")

# ============================================================================
# 3. HANDLE MISSING VALUES IN FEATURES
# ============================================================================
print("\n[3/7] Handling missing values in features...")

# Check which features have missing values
missing_features = [col for col in feature_cols if df_clean[col].isnull().any()]

if missing_features:
    print(f"  Features with missing values: {len(missing_features)}")
    
    for col in missing_features:
        missing_pct = 100 * df_clean[col].isnull().sum() / len(df_clean)
        
        if missing_pct > 50:
            print(f"    {col}: {missing_pct:.1f}% missing - dropping column")
            feature_cols.remove(col)
            df_clean = df_clean.drop(columns=[col])
        elif missing_pct > 20:
            print(f"    {col}: {missing_pct:.1f}% missing - filling with median")
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
        else:
            print(f"    {col}: {missing_pct:.1f}% missing - using forward/backward fill")
            df_clean[col] = df_clean[col].ffill().bfill()
            if df_clean[col].isnull().any():
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
else:
    print("  [OK] No missing values in features")

print(f"  Final feature count: {len(feature_cols)}")

# ============================================================================
# 4. OUTLIER DETECTION AND HANDLING
# ============================================================================
print("\n[4/7] Outlier detection and handling...")

outlier_stats = {}
for col in feature_cols:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    
    outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
    outlier_pct = 100 * outliers / len(df_clean)
    
    if outliers > 0:
        outlier_stats[col] = {
            'count': outliers,
            'pct': outlier_pct,
            'lower': lower_bound,
            'upper': upper_bound
        }

if outlier_stats:
    print(f"  Features with outliers (>3 IQR): {len(outlier_stats)}")
    for col, stats in sorted(outlier_stats.items(), key=lambda x: x[1]['count'], reverse=True)[:10]:
        print(f"    {col}: {stats['count']} outliers ({stats['pct']:.1f}%)")
    print("  Note: Using RobustScaler which handles outliers well")
else:
    print("  [OK] No significant outliers detected")

# ============================================================================
# 5. FEATURE ENGINEERING - INTERACTIONS AND TRANSFORMATIONS
# ============================================================================
print("\n[5/7] Feature engineering...")

df_features = df_clean.copy()

print("  [OK] Creating temporal features...")
if 'year' in df_features.columns:
    START_YEAR = df_features['year'].min()
    df_features['year_trend'] = df_features['year'] - START_YEAR
    feature_cols.append('year_trend')

if 'month' in df_features.columns:
    df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
    feature_cols.extend(['month_sin', 'month_cos'])

print("  [OK] Creating soil moisture interaction features...")
soil_moisture_cols = [col for col in feature_cols if 'SoilMoi' in col]
if len(soil_moisture_cols) >= 2:
    # Average soil moisture across depths
    df_features['soil_moisture_avg'] = df_features[soil_moisture_cols].mean(axis=1)
    feature_cols.append('soil_moisture_avg')
    
    # Soil moisture gradient (deep vs shallow)
    if 'SoilMoi0_10cm_inst' in feature_cols and 'SoilMoi100_200cm_inst' in feature_cols:
        df_features['soil_moisture_gradient'] = (
            df_features['SoilMoi100_200cm_inst'] - df_features['SoilMoi0_10cm_inst']
        )
        feature_cols.append('soil_moisture_gradient')

print("  [OK] Creating temperature interaction features...")
temp_cols = [col for col in feature_cols if 'Temp' in col or 'Tair' in col or 'Tveg' in col or 'tmean' in col.lower() or 'tmin' in col.lower() or 'tmax' in col.lower()]
if len(temp_cols) >= 2:
    # Average temperature
    df_features['temperature_avg'] = df_features[temp_cols].mean(axis=1)
    feature_cols.append('temperature_avg')
    
    # Temperature range (max - min)
    if 'prism_tmax_degf' in feature_cols and 'prism_tmin_degf' in feature_cols:
        df_features['temperature_range'] = (
            df_features['prism_tmax_degf'] - df_features['prism_tmin_degf']
        )
        feature_cols.append('temperature_range')

print("  [OK] Creating water balance features...")
if 'prism_ppt_in' in feature_cols and 'Evap_tavg' in feature_cols:
    # Precipitation minus evaporation
    df_features['precipitation_evap_balance'] = (
        df_features['prism_ppt_in'] - df_features['Evap_tavg']
    )
    feature_cols.append('precipitation_evap_balance')

if 'prism_ppt_in' in feature_cols and 'SoilMoi0_10cm_inst' in feature_cols:
    # Precipitation efficiency (moisture gain per unit precipitation)
    min_eps = 1e-8
    df_features['precipitation_efficiency'] = (
        df_features['SoilMoi0_10cm_inst'] / (df_features['prism_ppt_in'] + min_eps)
    )
    feature_cols.append('precipitation_efficiency')

print("  [OK] Creating agricultural context features...")
if 'corn_acres_planted' in feature_cols and 'corn_production_bu' in df_features.columns:
    # Yield per acre
    min_eps = 1e-8
    df_features['yield_per_acre'] = (
        df_features['corn_production_bu'] / (df_features['corn_acres_planted'] + min_eps)
    )
    feature_cols.append('yield_per_acre')

if 'diesel_usd_gal' in feature_cols:
    # Create interaction with planted acres (fuel cost proxy)
    if 'corn_acres_planted' in feature_cols:
        df_features['fuel_cost_proxy'] = (
            df_features['diesel_usd_gal'] * df_features['corn_acres_planted']
        )
        feature_cols.append('fuel_cost_proxy')

print("  [OK] Creating economic interaction features...")
if 'income_farmrelated_receipts_total_usd' in feature_cols and 'govt_programs_federal_receipts_usd' in feature_cols:
    # Total revenue sources
    df_features['total_revenue_sources'] = (
        df_features['income_farmrelated_receipts_total_usd'] + 
        df_features['govt_programs_federal_receipts_usd']
    )
    feature_cols.append('total_revenue_sources')

if 'income_farmrelated_receipts_total_usd' in feature_cols and 'corn_production_bu' in df_features.columns:
    # Revenue per bushel
    min_eps = 1e-8
    df_features['revenue_per_bushel'] = (
        df_features['income_farmrelated_receipts_total_usd'] / 
        (df_features['corn_production_bu'] + min_eps)
    )
    feature_cols.append('revenue_per_bushel')

print("  [OK] Creating spatial features...")
if 'dist_km_ethanol' in feature_cols:
    # Distance categories (discretize for potential interactions)
    df_features['ethanol_dist_category'] = pd.cut(
        df_features['dist_km_ethanol'],
        bins=[0, 25, 50, 100, float('inf')],
        labels=['Very Close', 'Close', 'Medium', 'Far']
    )
    # One-hot encode (but keep original too)
    dist_dummies = pd.get_dummies(df_features['ethanol_dist_category'], prefix='ethanol_dist')
    df_features = pd.concat([df_features, dist_dummies], axis=1)
    feature_cols.extend(dist_dummies.columns.tolist())

print(f"  [OK] Total features after engineering: {len(feature_cols)}")

# ============================================================================
# 6. SCALING STRATEGIES
# ============================================================================
print("\n[6/7] Scaling features...")

# Separate numeric features for scaling
numeric_features = [col for col in feature_cols if df_features[col].dtype in [np.float64, np.int64]]

print(f"  Features to scale: {len(numeric_features)}")

# Use RobustScaler (handles outliers better than StandardScaler)
scaler = RobustScaler()

print("  [OK] Fitting RobustScaler (robust to outliers)...")
scaler.fit(df_features[numeric_features])

print("  [OK] Transforming features...")
df_features[numeric_features] = scaler.transform(df_features[numeric_features])

print("  [OK] Scaling complete")

# Save scaler for later use
import pickle
with open('robust_scaler_phase3.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("  [OK] Saved scaler to robust_scaler_phase3.pkl")

# ============================================================================
# 7. FINAL DATASET PREPARATION
# ============================================================================
print("\n[7/7] Preparing final training dataset...")

# Select final columns
final_cols = ID_COLS + [TARGET_COL] + feature_cols
# Remove any columns that don't exist
final_cols = [col for col in final_cols if col in df_features.columns]

df_final = df_features[final_cols].copy()

print(f"  Final dataset shape: {df_final.shape}")
print(f"  Features: {len(feature_cols)}")
print(f"  Samples: {len(df_final)}")

# Verify no missing values
missing_final = df_final[feature_cols].isnull().sum().sum()
if missing_final > 0:
    print(f"  [WARNING] {missing_final} missing values remain - filling with 0")
    df_final[feature_cols] = df_final[feature_cols].fillna(0)
else:
    print("  [OK] No missing values in final dataset")

# Verify no infinite values
inf_final = np.isinf(df_final[feature_cols].select_dtypes(include=[np.number])).sum().sum()
if inf_final > 0:
    print(f"  [WARNING] {inf_final} infinite values found - replacing with 0")
    df_final[feature_cols] = df_final[feature_cols].replace([np.inf, -np.inf], 0)
else:
    print("  [OK] No infinite values in final dataset")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

output_file = "consolidated_data_phase3_preprocessed.csv"
df_final.to_csv(output_file, index=False)
print(f"[OK] Saved preprocessed dataset to: {output_file}")
print(f"  File size: {Path(output_file).stat().st_size / (1024*1024):.2f} MB")

# Save feature list
with open('phase3_feature_list.txt', 'w') as f:
    f.write("PHASE 3 PREPROCESSED DATASET - FEATURE LIST\n")
    f.write("="*80 + "\n\n")
    f.write(f"Total Features: {len(feature_cols)}\n")
    f.write(f"Total Samples: {len(df_final)}\n\n")
    f.write("Features:\n")
    for i, feat in enumerate(feature_cols, 1):
        f.write(f"  {i:3d}. {feat}\n")
    f.write("\n" + "="*80 + "\n")
    f.write("ID Columns:\n")
    for col in ID_COLS:
        f.write(f"  - {col}\n")
    f.write(f"\nTarget Column: {TARGET_COL}\n")

print("[OK] Saved feature list to: phase3_feature_list.txt")

# Summary statistics
print("\n" + "="*80)
print("PREPROCESSING SUMMARY")
print("="*80)
print(f"Original dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Final dataset: {df_final.shape[0]} rows, {df_final.shape[1]} columns")
print(f"Features created: {len(feature_cols)}")
print(f"Rows removed: {df.shape[0] - df_final.shape[0]} ({100*(df.shape[0]-df_final.shape[0])/df.shape[0]:.1f}%)")
print(f"Target variable range: {df_final[TARGET_COL].min():,.0f} - {df_final[TARGET_COL].max():,.0f} bushels")

print("\n" + "="*80)
print("PREPROCESSING COMPLETE!")
print("="*80)

