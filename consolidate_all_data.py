import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CONSOLIDATING ALL CSV FILES INTO ONE DATASET")
print("="*80)

base_path = Path(".")

df_combined = pd.DataFrame()

print("\n1. Loading base dataset: combined_gldas_corn_data.csv")
if (base_path / "combined_gldas_corn_data.csv").exists():
    df_combined = pd.read_csv("combined_gldas_corn_data.csv")
    print(f"   Loaded: {df_combined.shape[0]} rows, {df_combined.shape[1]} columns")
    print(f"   Key columns: {list(df_combined.columns[:5])}...")
else:
    print("   WARNING: Base file not found! Starting with empty dataset.")
    print("   Attempting to build from component files...")
    if (base_path / "Data" / "gldas_all_bands_data.csv").exists():
        df_combined = pd.read_csv("Data/gldas_all_bands_data.csv")
        print(f"   Loaded GLDAS data: {df_combined.shape}")

print("\n2. Adding PRISM weather data...")
try:
    if (base_path / "Data" / "PRISM_ppt_tmin_tmean_tmax_tdmean_vpdmin_vpdmax_stable_4km_201001_202408.csv").exists():
        df_prism = pd.read_csv("Data/PRISM_ppt_tmin_tmean_tmax_tdmean_vpdmin_vpdmax_stable_4km_201001_202408.csv")
        
        if 'Name' in df_prism.columns:
            county_mapping = {
                'MilleLacs': '27095', 'Sherburne': '27141', 'Stearns': '27145',
                'Aitkin': '27001', 'Anoka': '27003', 'Becker': '27005'
            }
            
            df_prism['date'] = pd.to_datetime(df_prism['date'], errors='coerce')
            df_prism = df_prism.dropna(subset=['date'])
            df_prism['year'] = df_prism['date'].dt.year
            df_prism['month'] = df_prism['date'].dt.month
            
            county_to_fips = pd.read_csv("combined_gldas_corn_data.csv")[['county_name', 'fips']].drop_duplicates()
            county_to_fips_dict = dict(zip(county_to_fips['county_name'].str.upper(), county_to_fips['fips'].astype(str)))
            
            df_prism['fips'] = df_prism['Name'].map(county_mapping)
            
            if df_prism['fips'].notna().any():
                df_prism_agg = df_prism.groupby(['fips', 'year', 'month']).agg({
                    'ppt (inches)': 'mean',
                    'tmin (degrees F)': 'mean',
                    'tmean (degrees F)': 'mean',
                    'tmax (degrees F)': 'mean',
                    'tdmean (degrees F)': 'mean',
                    'vpdmin (hPa)': 'mean',
                    'vpdmax (hPa)': 'mean'
                }).reset_index()
                
                df_prism_agg.columns = ['fips', 'year', 'month', 'prism_ppt_inches', 'prism_tmin_f', 
                                       'prism_tmean_f', 'prism_tmax_f', 'prism_tdmean_f', 
                                       'prism_vpdmin_hpa', 'prism_vpdmax_hpa']
                
                if not df_combined.empty:
                    df_combined = df_combined.merge(df_prism_agg, on=['fips', 'year', 'month'], how='left')
                    print(f"   Merged PRISM data: Added {len(df_prism_agg.columns) - 3} columns")
except Exception as e:
    print(f"   WARNING: Could not merge PRISM data: {e}")

print("\n3. Adding corn production data (wide format)...")
try:
    if (base_path / "Data" / "mn_county_corn_production_2000_2022.csv").exists():
        df_prod = pd.read_csv("Data/mn_county_corn_production_2000_2022.csv")
        
        years_cols = [str(year) for year in range(2000, 2025) if str(year) in df_prod.columns]
        if not years_cols:
            years_cols = [col for col in df_prod.columns if col.isdigit() and 2000 <= int(col) <= 2025]
        
        if years_cols:
            df_prod_melted = df_prod.melt(
                id_vars=['fips', 'county_name'],
                value_vars=years_cols,
                var_name='year',
                value_name='corn_production_from_prod_file'
            )
            df_prod_melted['year'] = df_prod_melted['year'].astype(int)
            
            if not df_combined.empty:
                df_combined = df_combined.merge(
                    df_prod_melted[['fips', 'year', 'corn_production_from_prod_file']],
                    on=['fips', 'year'],
                    how='left'
                )
                print(f"   Merged production data: Added 1 column")
except Exception as e:
    print(f"   WARNING: Could not merge production data: {e}")

print("\n4. Adding yield and harvested acres data...")
try:
    if (base_path / "yield and harvested acres.csv").exists():
        df_yield = pd.read_csv("yield and harvested acres.csv")
        
        if 'fips' in df_yield.columns and 'year' in df_yield.columns:
            merge_cols = ['fips', 'year']
            if 'month' in df_yield.columns and 'month' in df_combined.columns:
                merge_cols.append('month')
            
            df_combined = df_combined.merge(df_yield, on=merge_cols, how='left', suffixes=('', '_yield'))
            print(f"   Merged yield data: Added columns")
except Exception as e:
    print(f"   WARNING: Could not merge yield data: {e}")

print("\n5. Adding NDVI data...")
try:
    ndvi_files = [
        "Data/mn_county_ndvi_2022_2022.csv",
        "Data/mn_county_ndvi_sample.csv"
    ]
    
    for ndvi_file in ndvi_files:
        if (base_path / ndvi_file).exists():
            df_ndvi = pd.read_csv(ndvi_file, header=[0, 1])
            
            if len(df_ndvi.columns.levels) == 2:
                df_ndvi.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in df_ndvi.columns]
            
            df_ndvi.columns = [str(col).strip() for col in df_ndvi.columns]
            
            if 'fips' in df_ndvi.columns:
                df_ndvi_melted = pd.melt(
                    df_ndvi,
                    id_vars=['fips', 'county_name'] if 'county_name' in df_ndvi.columns else ['fips'],
                    var_name='temp_col',
                    value_name='ndvi_value'
                )
                
                year_month_pattern = r'(\d{4})|month|year'
                if df_ndvi.columns[2].isdigit():
                    df_ndvi_long = df_ndvi.set_index(['fips', 'county_name' if 'county_name' in df_ndvi.columns else 'fips'])
                    
                    years = [int(col) for col in df_ndvi.columns[2:] if str(col).isdigit()]
                    months = list(range(1, 13))
                    
                    df_ndvi_list = []
                    for idx, row in df_ndvi.iterrows():
                        fips_val = row['fips']
                        county_val = row.get('county_name', '')
                        
                        for year in years:
                            for month_idx, month in enumerate([2,3,4,5,6,7,8]):
                                col_idx = 2 + years.index(year) * 7 + month_idx
                                if col_idx < len(df_ndvi.columns):
                                    val = row.iloc[col_idx]
                                    if pd.notna(val):
                                        df_ndvi_list.append({
                                            'fips': fips_val,
                                            'year': year,
                                            'month': month,
                                            'ndvi': val
                                        })
                    
                    if df_ndvi_list:
                        df_ndvi_clean = pd.DataFrame(df_ndvi_list)
                        if not df_combined.empty:
                            df_combined = df_combined.merge(
                                df_ndvi_clean,
                                on=['fips', 'year', 'month'],
                                how='left'
                            )
                            print(f"   Merged NDVI data from {ndvi_file}")
                            break
except Exception as e:
    print(f"   WARNING: Could not merge NDVI data: {e}")

print("\n6. Adding SSM (Soil Surface Moisture) data...")
try:
    ssm_files = [
        "Data/mn_county_ssm_2010_2022.csv",
        "Data/mn_county_ssm_2022_2022.csv"
    ]
    
    for ssm_file in ssm_files:
        if (base_path / ssm_file).exists():
            df_ssm = pd.read_csv(ssm_file, header=[0, 1])
            
            if len(df_ssm.columns.levels) == 2:
                df_ssm.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in df_ssm.columns]
            
            df_ssm.columns = [str(col).strip() for col in df_ssm.columns]
            
            if 'fips' in df_ssm.columns:
                ssm_list = []
                for idx, row in df_ssm.iterrows():
                    fips_val = row['fips']
                    county_val = row.get('county_name', '')
                    
                    year_row = None
                    month_row = None
                    for col in df_ssm.columns:
                        if 'year' in str(col).lower():
                            year_row = row[col] if pd.notna(row[col]) else year_row
                        if 'month' in str(col).lower() and col != 'fips':
                            month_row = row[col] if pd.notna(row[col]) else month_row
                    
                    numeric_cols = [col for col in df_ssm.columns[2:] if str(col).replace('.','').isdigit() or '.' in str(col)]
                    
                    for i, col in enumerate(df_ssm.columns[2:], start=0):
                        if i < len(df_ssm.columns) - 2:
                            year_idx = (i // 7) if len(df_ssm.columns) > 9 else 0
                            month_idx = i % 7
                            
                            years_in_data = sorted([int(c) for c in df_ssm.columns[2:] if str(c).isdigit()], reverse=True)
                            
                            if years_in_data:
                                year = years_in_data[0] + (i // 7) if i // 7 < len(years_in_data) else years_in_data[0]
                                month = [2,3,4,5,6,7,8][i % 7] if i % 7 < 7 else None
                                
                                if month and pd.notna(row[col]):
                                    ssm_list.append({
                                        'fips': fips_val,
                                        'year': year,
                                        'month': month,
                                        'ssm': row[col]
                                    })
                
                if ssm_list:
                    df_ssm_clean = pd.DataFrame(ssm_list)
                    df_ssm_clean = df_ssm_clean.drop_duplicates(subset=['fips', 'year', 'month'])
                    
                    if not df_combined.empty:
                        df_combined = df_combined.merge(
                            df_ssm_clean,
                            on=['fips', 'year', 'month'],
                            how='left'
                        )
                        print(f"   Merged SSM data from {ssm_file}")
                        break
except Exception as e:
    print(f"   WARNING: Could not merge SSM data: {e}")

print("\n7. Adding ethanol plant distance data...")
try:
    if (base_path / "ethanol_dist.csv").exists():
        df_ethanol = pd.read_csv("ethanol_dist.csv")
        
        if 'fips' in df_ethanol.columns:
            if not df_combined.empty:
                df_combined = df_combined.merge(df_ethanol, on='fips', how='left', suffixes=('', '_ethanol'))
                print(f"   Merged ethanol distance data: Added {len(df_ethanol.columns) - 1} columns")
except Exception as e:
    print(f"   WARNING: Could not merge ethanol data: {e}")

print("\n8. Adding diesel price data...")
try:
    if (base_path / "diesel_dist.csv").exists():
        df_diesel = pd.read_csv("diesel_dist.csv")
        
        if 'year' in df_diesel.columns and 'month' in df_diesel.columns:
            if not df_combined.empty:
                df_combined = df_combined.merge(df_diesel, on=['year', 'month'], how='left', suffixes=('', '_diesel'))
                print(f"   Merged diesel price data: Added 1 column")
except Exception as e:
    print(f"   WARNING: Could not merge diesel data: {e}")

print("\n9. Handling duplicate column names...")
if not df_combined.empty:
    cols_to_keep = []
    seen = set()
    for col in df_combined.columns:
        if col not in seen:
            cols_to_keep.append(col)
            seen.add(col)
        else:
            print(f"   Removed duplicate column: {col}")
    
    df_combined = df_combined[cols_to_keep]

print("\n10. Final dataset summary:")
if not df_combined.empty:
    print(f"   Total rows: {len(df_combined):,}")
    print(f"   Total columns: {len(df_combined.columns)}")
    print(f"   Memory usage: {df_combined.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\n   Column names ({len(df_combined.columns)} total):")
    for i, col in enumerate(df_combined.columns, 1):
        print(f"   {i:3d}. {col}")
    
    print(f"\n   Missing values per column:")
    missing = df_combined.isnull().sum()
    missing_pct = (missing / len(df_combined) * 100).round(2)
    for col in df_combined.columns:
        if missing[col] > 0:
            print(f"   {col}: {missing[col]:,} ({missing_pct[col]}%)")
    
    print(f"\n   Saving consolidated dataset...")
    output_file = "consolidated_all_data.csv"
    df_combined.to_csv(output_file, index=False)
    print(f"   ✓ Saved to: {output_file}")
    
    print(f"\n   Sample of first 3 rows:")
    print(df_combined.head(3).to_string())
    
else:
    print("   ERROR: No data could be consolidated!")

print("\n" + "="*80)
print("CONSOLIDATION COMPLETE!")
print("="*80)


