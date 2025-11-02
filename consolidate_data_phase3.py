# ============================================================================
# DATA CONSOLIDATION SCRIPT - Phase 3 Methodology
# ============================================================================
# This script consolidates multiple data sources into a single dataset:
# 1. Corn harvest/planted data (NASA Quick Stats) - acres planted as feature
# 2. Diesel price data - monthly pricing
# 3. Economy MN data - economic indicators (handles missing years)
# 4. Ethanol distance - nearest ethanol processing facility distance
# 5. PRISM precipitation data - monthly precipitation and temperature
# ============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("DATA CONSOLIDATION - PHASE 3 METHODOLOGY")
print("="*80)

base_path = Path(".")
data_path = base_path / "Data"

# ============================================================================
# 1. LOAD BASE DATASET (GLDAS Corn Data)
# ============================================================================
print("\n[1/6] Loading base dataset...")
try:
    df_base = pd.read_csv("combined_gldas_corn_data.csv")
    print(f"  ✓ Loaded base dataset: {df_base.shape[0]} rows, {df_base.shape[1]} columns")
    print(f"  ✓ Year range: {df_base['year'].min()} - {df_base['year'].max()}")
    print(f"  ✓ Counties: {df_base['fips'].nunique()}")
except FileNotFoundError:
    print("  ⚠ WARNING: Base dataset not found. Creating from scratch...")
    df_base = pd.DataFrame()
except Exception as e:
    print(f"  ✗ ERROR loading base dataset: {e}")
    df_base = pd.DataFrame()

# ============================================================================
# 2. PROCESS CORN HARVEST/PLANTED DATA (NASA Quick Stats)
# ============================================================================
print("\n[2/6] Processing corn harvest/planted data...")
try:
    df_corn = pd.read_csv(data_path / "corn_harvest_planted_2000-2023.csv")
    print(f"  ✓ Loaded corn data: {df_corn.shape[0]} rows")
    
    # Filter for planted acres data
    df_planted = df_corn[df_corn['Data Item'].str.contains('ACRES PLANTED', case=False, na=False)].copy()
    
    # Clean value column - remove commas and convert to numeric
    df_planted['Value'] = df_planted['Value'].astype(str).str.replace(',', '').replace('(D)', np.nan)
    df_planted['Value'] = pd.to_numeric(df_planted['Value'], errors='coerce')
    
    # Extract FIPS code: Combine State ANSI (27 for Minnesota) + County ANSI
    # Minnesota FIPS codes are 27001-27171 where 27 = state, 001-171 = county
    if 'State ANSI' in df_planted.columns and 'County ANSI' in df_planted.columns:
        # Convert County ANSI to numeric first (handles floats like 9.0), then to int, then string
        # This ensures 9.0 -> 9 -> "9" -> "009" -> "27009"
        state_code = df_planted['State ANSI'].astype(str).str.strip()
        county_code_numeric = pd.to_numeric(df_planted['County ANSI'], errors='coerce')
        county_code = county_code_numeric.fillna(0).astype(int).astype(str).str.zfill(3)
        df_planted['fips'] = (state_code + county_code).astype(str)
        df_planted['fips'] = pd.to_numeric(df_planted['fips'], errors='coerce')
        # Filter out invalid FIPS (where county was NaN or 0)
        df_planted = df_planted[df_planted['fips'] >= 27001]
    else:
        # Fallback: use County ANSI only (old method)
        df_planted['fips'] = df_planted['County ANSI'].astype(str).str.zfill(5)
        df_planted['fips'] = pd.to_numeric(df_planted['fips'], errors='coerce')
    
    # Filter out rows without valid FIPS or values
    df_planted = df_planted.dropna(subset=['fips', 'Value', 'Year'])
    
    # Aggregate by FIPS and Year (sum if multiple records)
    df_planted_agg = df_planted.groupby(['fips', 'Year'])['Value'].sum().reset_index()
    df_planted_agg.columns = ['fips', 'year', 'corn_acres_planted']
    
    print(f"  ✓ Processed planted acres: {df_planted_agg.shape[0]} county-year combinations")
    print(f"  ✓ Year range: {df_planted_agg['year'].min()} - {df_planted_agg['year'].max()}")
    print(f"  ✓ Unique FIPS: {df_planted_agg['fips'].nunique()}")
    print(f"  ✓ Sample FIPS: {sorted(df_planted_agg['fips'].unique())[:5]}")
    
    # Merge with base dataset
    if not df_base.empty:
        before_merge = len(df_base)
        df_base = df_base.merge(df_planted_agg, on=['fips', 'year'], how='left')
        after_merge = df_base['corn_acres_planted'].notna().sum()
        print(f"  ✓ Merged planted acres data: {after_merge}/{before_merge} rows matched ({100*after_merge/before_merge:.1f}%)")
    else:
        df_base = df_planted_agg.copy()
        
except FileNotFoundError:
    print("  ⚠ WARNING: corn_harvest_planted_2000-2023.csv not found")
except Exception as e:
    print(f"  ⚠ WARNING: Error processing corn data: {e}")

# ============================================================================
# 3. PROCESS DIESEL PRICE DATA
# ============================================================================
print("\n[3/6] Processing diesel price data...")
try:
    df_diesel = pd.read_csv(data_path / "diesel_price.csv")
    print(f"  ✓ Loaded diesel data: {df_diesel.shape[0]} rows")
    
    # Ensure columns exist
    if 'year' in df_diesel.columns and 'month' in df_diesel.columns:
        df_diesel_clean = df_diesel[['year', 'month', 'diesel_usd_gal']].copy()
        df_diesel_clean = df_diesel_clean.dropna()
        
        print(f"  ✓ Clean diesel data: {df_diesel_clean.shape[0]} rows")
        print(f"  ✓ Year range: {df_diesel_clean['year'].min()} - {df_diesel_clean['year'].max()}")
        
        # Merge with base dataset (monthly merge)
        if not df_base.empty and 'month' in df_base.columns:
            df_base = df_base.merge(df_diesel_clean, on=['year', 'month'], how='left')
            print(f"  ✓ Merged diesel price data (monthly)")
        elif not df_base.empty:
            # If no month column, aggregate to yearly average
            df_diesel_yearly = df_diesel_clean.groupby('year')['diesel_usd_gal'].mean().reset_index()
            df_diesel_yearly.columns = ['year', 'diesel_price_avg_usd_gal']
            df_base = df_base.merge(df_diesel_yearly, on=['year'], how='left')
            print(f"  ✓ Merged diesel price data (yearly average)")
    else:
        print("  ⚠ WARNING: Expected columns not found in diesel data")
        
except FileNotFoundError:
    print("  ⚠ WARNING: diesel_price.csv not found")
except Exception as e:
    print(f"  ⚠ WARNING: Error processing diesel data: {e}")

# ============================================================================
# 4. PROCESS ECONOMY MN DATA (Handle Missing Years)
# ============================================================================
print("\n[4/6] Processing economy MN data...")
try:
    df_economy = pd.read_csv(data_path / "enonomy_mn.csv")
    print(f"  ✓ Loaded economy data: {df_economy.shape[0]} rows")
    
    # Available years
    years_available = sorted(df_economy['Year'].unique())
    print(f"  ⚠ Available years: {years_available} (inconsistent - missing years)")
    
    # Clean value column
    df_economy['Value'] = df_economy['Value'].astype(str).str.replace(',', '').replace('(D)', np.nan)
    df_economy['Value'] = pd.to_numeric(df_economy['Value'], errors='coerce')
    
    # Extract FIPS code: Combine State ANSI + County ANSI (same as corn data)
    if 'State ANSI' in df_economy.columns and 'County ANSI' in df_economy.columns:
        # Convert County ANSI to numeric first (handles floats like 9.0), then to int, then string
        # This ensures 9.0 -> 9 -> "9" -> "009" -> "27009"
        state_code = df_economy['State ANSI'].astype(str).str.strip()
        county_code_numeric = pd.to_numeric(df_economy['County ANSI'], errors='coerce')
        county_code = county_code_numeric.fillna(0).astype(int).astype(str).str.zfill(3)
        df_economy['fips'] = (state_code + county_code).astype(str)
        df_economy['fips'] = pd.to_numeric(df_economy['fips'], errors='coerce')
        # Filter out invalid FIPS (where county was NaN or 0)
        df_economy = df_economy[df_economy['fips'] >= 27001]
    else:
        # Fallback: use County ANSI only (old method)
        df_economy['fips'] = df_economy['County ANSI'].astype(str).str.zfill(5)
        df_economy['fips'] = pd.to_numeric(df_economy['fips'], errors='coerce')
    
    # Filter valid records
    df_economy = df_economy.dropna(subset=['fips', 'Value', 'Year'])
    
    # Select key economic indicators
    # Focus on total income, government payments, and farm-related income
    key_indicators = [
        'INCOME, FARM-RELATED - RECEIPTS, MEASURED IN $',
        'GOVT PROGRAMS, FEDERAL - RECEIPTS, MEASURED IN $',
        'INCOME, FARM-RELATED - RECEIPTS, MEASURED IN $ / OPERATION'
    ]
    
    # Pivot to create columns for each indicator
    economy_indicators = []
    for indicator in key_indicators:
        df_ind = df_economy[df_economy['Data Item'] == indicator].copy()
        if not df_ind.empty:
            # Create better column names based on indicator type
            if 'INCOME, FARM-RELATED - RECEIPTS, MEASURED IN $' in indicator and '/ OPERATION' not in indicator:
                col_name = 'income_farmrelated_receipts_total_usd'
            elif 'INCOME, FARM-RELATED - RECEIPTS, MEASURED IN $ / OPERATION' in indicator:
                col_name = 'income_farmrelated_receipts_per_operation_usd'
            elif 'GOVT PROGRAMS, FEDERAL - RECEIPTS, MEASURED IN $' in indicator:
                col_name = 'govt_programs_federal_receipts_usd'
            else:
                # Fallback: clean up the indicator name
                col_name = indicator.lower().replace(',', '').replace(' ', '_').replace('-', '_').replace('/', '_')
                col_name = ''.join(c for c in col_name if c.isalnum() or c == '_')[:50]
            
            df_ind_pivot = df_ind[['fips', 'Year', 'Value']].copy()
            df_ind_pivot.columns = ['fips', 'year', col_name]
            economy_indicators.append(df_ind_pivot)
            print(f"    ✓ Processed indicator: {indicator[:60]}... -> {col_name}")
    
    if economy_indicators:
        df_economy_merged = economy_indicators[0]
        for df_ind in economy_indicators[1:]:
            df_economy_merged = df_economy_merged.merge(df_ind, on=['fips', 'year'], how='outer')
        
        print(f"  ✓ Processed {len(economy_indicators)} economic indicators")
        print(f"  ✓ Economy data shape: {df_economy_merged.shape[0]} rows, {len(economy_indicators)} columns")
        print(f"  ✓ Year range: {df_economy_merged['year'].min()} - {df_economy_merged['year'].max()}")
        print(f"  ✓ Unique FIPS: {df_economy_merged['fips'].nunique()}")
        
        # Merge with base dataset
        if not df_base.empty:
            before_merge = len(df_base)
            df_base = df_base.merge(df_economy_merged, on=['fips', 'year'], how='left')
            # Count matches for each economy column
            economy_cols = [col for col in df_economy_merged.columns if col not in ['fips', 'year']]
            matches_before = {col: df_base[col].notna().sum() for col in economy_cols}
            print(f"  ✓ Merged economy data (before imputation): {matches_before}")
            
            print(f"  ✓ Applying multi-strategy imputation for economy data...")
            
            # Sort by fips and year to ensure proper temporal order for interpolation
            df_base = df_base.sort_values(['fips', 'year']).reset_index(drop=True)
            
            # Multi-strategy imputation for economy data
            for col in economy_cols:
                initial_missing = df_base[col].isna().sum()
                
                # Strategy 1: Forward fill (last known value forward)
                df_base[col] = df_base.groupby('fips')[col].ffill()
                after_ffill = df_base[col].isna().sum()
                
                # Strategy 2: Backward fill (first known value backward)
                df_base[col] = df_base.groupby('fips')[col].bfill()
                after_bfill = df_base[col].isna().sum()
                
                # Strategy 3: Linear interpolation within each county (time-based)
                # Groupby preserves order, so interpolation works correctly
                df_base[col] = df_base.groupby('fips', group_keys=False)[col].apply(
                    lambda x: x.interpolate(method='linear', limit_direction='both')
                )
                after_interp = df_base[col].isna().sum()
                
                # Strategy 4: Fill with county median (for counties with some data but gaps)
                county_medians = df_base.groupby('fips')[col].median()
                for fips in df_base[df_base[col].isna()]['fips'].unique():
                    if fips in county_medians.index and pd.notna(county_medians[fips]):
                        mask = (df_base['fips'] == fips) & (df_base[col].isna())
                        df_base.loc[mask, col] = county_medians[fips]
                after_median = df_base[col].isna().sum()
                
                # Strategy 5: Fill with year-specific median across all counties (for counties with no data)
                year_medians = df_base.groupby('year')[col].median()
                for year in df_base[df_base[col].isna()]['year'].unique():
                    if year in year_medians.index and pd.notna(year_medians[year]):
                        mask = (df_base['year'] == year) & (df_base[col].isna())
                        df_base.loc[mask, col] = year_medians[year]
                after_year_median = df_base[col].isna().sum()
                
                # Strategy 6: Fill with overall median (last resort)
                overall_median = df_base[col].median()
                if pd.notna(overall_median):
                    df_base[col] = df_base[col].fillna(overall_median)
                
                final_missing = df_base[col].isna().sum()
                filled = initial_missing - final_missing
                
                print(f"    {col}:")
                print(f"      Initial missing: {initial_missing} ({100*initial_missing/len(df_base):.1f}%)")
                print(f"      After forward-fill: {after_ffill} missing (filled {initial_missing - after_ffill})")
                print(f"      After backward-fill: {after_bfill} missing (filled {after_ffill - after_bfill})")
                print(f"      After interpolation: {after_interp} missing (filled {after_bfill - after_interp})")
                print(f"      After county median: {after_median} missing (filled {after_interp - after_median})")
                print(f"      After year median: {after_year_median} missing (filled {after_median - after_year_median})")
                print(f"      Final missing: {final_missing} ({100*final_missing/len(df_base):.1f}%)")
                print(f"      Total filled: {filled} values ({100*filled/initial_missing:.1f}% of missing)")
        else:
            df_base = df_economy_merged.copy()
    else:
        print("  ⚠ WARNING: No key economic indicators found")
        
except FileNotFoundError:
    print("  ⚠ WARNING: enonomy_mn.csv not found")
except Exception as e:
    print(f"  ⚠ WARNING: Error processing economy data: {e}")

# ============================================================================
# 5. PROCESS ETHANOL DISTANCE DATA
# ============================================================================
print("\n[5/6] Processing ethanol distance data...")
try:
    df_ethanol = pd.read_csv(data_path / "ethanol_dist.csv")
    print(f"  ✓ Loaded ethanol data: {df_ethanol.shape[0]} rows")
    
    # Ensure FIPS is properly formatted
    df_ethanol['fips'] = df_ethanol['fips'].astype(int)
    
    # Select relevant columns
    df_ethanol_clean = df_ethanol[['fips', 'dist_km_ethanol']].copy()
    
    print(f"  ✓ Processed ethanol distance: {df_ethanol_clean.shape[0]} counties")
    
    # Merge with base dataset (static feature - same for all years)
    if not df_base.empty:
        df_base = df_base.merge(df_ethanol_clean, on=['fips'], how='left')
        print(f"  ✓ Merged ethanol distance data")
    else:
        print("  ⚠ WARNING: Base dataset empty, cannot merge")
        
except FileNotFoundError:
    print("  ⚠ WARNING: ethanol_dist.csv not found")
except Exception as e:
    print(f"  ⚠ WARNING: Error processing ethanol data: {e}")

# ============================================================================
# 6. PROCESS PRISM PRECIPITATION DATA
# ============================================================================
print("\n[6/6] Processing PRISM precipitation data...")
try:
    df_prism = pd.read_csv(data_path / "PRISM_percipitation_data.csv")
    print(f"  ✓ Loaded PRISM data: {df_prism.shape[0]} rows, {df_prism.shape[1]} columns")
    print(f"  ✓ Available columns: {list(df_prism.columns)}")
    
    # Parse date column - handle both 'Date' (capital) and 'date' (lowercase)
    # Date format is 'YYYY-MM' (e.g., '2000-01')
    if 'Date' in df_prism.columns:
        # Parse YYYY-MM format
        df_prism['date_parsed'] = pd.to_datetime(df_prism['Date'], format='%Y-%m', errors='coerce')
        df_prism = df_prism.dropna(subset=['date_parsed'])
        df_prism['year'] = df_prism['date_parsed'].dt.year
        df_prism['month'] = df_prism['date_parsed'].dt.month
        df_prism = df_prism.drop(columns=['date_parsed'])
        print(f"  ✓ Parsed Date column (format: YYYY-MM)")
    elif 'date' in df_prism.columns:
        # Fallback for lowercase date column
        df_prism['date'] = pd.to_datetime(df_prism['date'], errors='coerce')
        df_prism = df_prism.dropna(subset=['date'])
        df_prism['year'] = df_prism['date'].dt.year
        df_prism['month'] = df_prism['date'].dt.month
    elif 'Date_old' in df_prism.columns:
        # Legacy format support
        df_prism['date'] = pd.to_datetime(df_prism['Date_old'], format='%Y-%m', errors='coerce')
        df_prism = df_prism.dropna(subset=['date'])
        df_prism['year'] = df_prism['date'].dt.year
        df_prism['month'] = df_prism['date'].dt.month
    else:
        raise ValueError("No recognized date column found (expected 'Date', 'date', or 'Date_old')")
    
    print(f"  ✓ Extracted year and month from date")
    print(f"  ✓ Year range: {df_prism['year'].min()} - {df_prism['year'].max()}")
    print(f"  ✓ Unique months: {sorted(df_prism['month'].unique())}")
    
    # Create FIPS mapping from Name/Location
    # Try to match county names from base dataset first
    county_fips_dict = {}
    if not df_base.empty and 'county_name' in df_base.columns:
        county_fips_map = df_base[['fips', 'county_name']].drop_duplicates()
        county_fips_dict = dict(zip(county_fips_map['county_name'].str.upper().str.strip(), 
                                   county_fips_map['fips']))
        print(f"  ✓ Created county name to FIPS mapping from base dataset ({len(county_fips_dict)} counties)")
    
    # Try matching PRISM names directly
    if 'Name' in df_prism.columns:
        df_prism['Name_upper'] = df_prism['Name'].str.upper().str.strip()
        df_prism['fips'] = df_prism['Name_upper'].map(county_fips_dict)
        
        unmatched_count = df_prism['fips'].isna().sum()
        if unmatched_count > 0:
            print(f"  ⚠ {unmatched_count} PRISM records could not be matched to FIPS initially")
            print(f"     Attempting alternative matching strategies...")
            
            # Try matching by removing spaces and special characters, partial matching
            if not df_base.empty and 'county_name' in df_base.columns:
                base_counties = df_base[['fips', 'county_name']].drop_duplicates()
                base_counties['name_clean'] = base_counties['county_name'].str.upper().str.strip().str.replace(' ', '').str.replace('-', '')
                
                # Also try reverse matching (PRISM name in base county name)
                for idx in df_prism[df_prism['fips'].isna()].index:
                    prism_name_clean = df_prism.loc[idx, 'Name'].upper().strip().replace(' ', '').replace('-', '')
                    
                    # Try exact match first
                    match = base_counties[base_counties['name_clean'] == prism_name_clean]
                    if match.empty:
                        # Try contains match (PRISM name contains base name or vice versa)
                        match = base_counties[base_counties['name_clean'].str.contains(prism_name_clean, na=False, case=False)]
                        if match.empty:
                            match = base_counties[base_counties['county_name'].str.upper().str.contains(df_prism.loc[idx, 'Name'], na=False, case=False)]
                    
                    if not match.empty:
                        df_prism.loc[idx, 'fips'] = match.iloc[0]['fips']
                
                remaining_unmatched = df_prism['fips'].isna().sum()
                if remaining_unmatched > 0:
                    print(f"  ⚠ {remaining_unmatched} records still unmatched - will be filtered out")
                else:
                    print(f"  ✓ Successfully matched all PRISM records to FIPS")
        
        df_prism = df_prism.drop(columns=['Name_upper'] if 'Name_upper' in df_prism.columns else [])
    else:
        print("  ⚠ WARNING: 'Name' column not found in PRISM data - cannot map to FIPS")
        df_prism['fips'] = np.nan
    
    # Filter valid records (must have FIPS, year, and month)
    initial_count = len(df_prism)
    df_prism = df_prism.dropna(subset=['fips', 'year', 'month'])
    filtered_count = len(df_prism)
    if initial_count != filtered_count:
        print(f"  ✓ Filtered to {filtered_count} valid records (removed {initial_count - filtered_count} invalid)")
    
    # Aggregate by fips, year, month (take mean if multiple records for same combination)
    prism_cols = ['ppt (inches)', 'tmin (degrees F)', 'tmean (degrees F)', 
                  'tmax (degrees F)', 'tdmean (degrees F)', 'vpdmin (hPa)', 'vpdmax (hPa)']
    available_cols = [col for col in prism_cols if col in df_prism.columns]
    
    if available_cols:
        print(f"  ✓ Found {len(available_cols)} PRISM data columns: {available_cols}")
        df_prism_agg = df_prism.groupby(['fips', 'year', 'month'])[available_cols].mean().reset_index()
        
        # Rename columns with prism prefix and clean up formatting
        rename_dict = {}
        for col in available_cols:
            new_name = f'prism_{col.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("degrees_f", "degf").replace("inches", "in")}'
            # Clean up any double underscores or trailing underscores
            new_name = new_name.replace('__', '_').strip('_')
            rename_dict[col] = new_name
        df_prism_agg = df_prism_agg.rename(columns=rename_dict)
        
        print(f"  ✓ Processed PRISM data: {df_prism_agg.shape[0]} county-year-month combinations")
        print(f"  ✓ Year range: {df_prism_agg['year'].min()} - {df_prism_agg['year'].max()}")
        print(f"  ✓ Unique counties: {df_prism_agg['fips'].nunique()}")
        
        # Merge with base dataset
        if not df_base.empty:
            merge_keys = ['fips', 'year']
            if 'month' in df_base.columns:
                merge_keys.append('month')
                df_base = df_base.merge(df_prism_agg, on=merge_keys, how='left')
                print(f"  ✓ Merged PRISM data (monthly level)")
            else:
                # Aggregate to yearly if no month in base
                df_prism_yearly = df_prism_agg.groupby(['fips', 'year'])[[col for col in df_prism_agg.columns if col not in ['fips', 'year', 'month']]].mean().reset_index()
                df_base = df_base.merge(df_prism_yearly, on=['fips', 'year'], how='left')
                print(f"  ✓ Merged PRISM data (yearly aggregated)")
        else:
            df_base = df_prism_agg.copy()
            print(f"  ✓ Created base dataset from PRISM data")
    else:
        print("  ⚠ WARNING: No expected PRISM columns found")
        print(f"     Available columns: {list(df_prism.columns)}")
        
except FileNotFoundError:
    print("  ⚠ WARNING: PRISM_percipitation_data.csv not found in Data directory")
except Exception as e:
    print(f"  ⚠ WARNING: Error processing PRISM data: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# FINAL PROCESSING AND SAVING
# ============================================================================
print("\n" + "="*80)
print("FINAL PROCESSING")
print("="*80)

if df_base.empty:
    print("\n❌ ERROR: Consolidated dataset is empty!")
    print("   Please check that at least one data source loaded successfully.")
else:
    print(f"\n✓ Final dataset shape: {df_base.shape[0]} rows, {df_base.shape[1]} columns")
    
    # Summary statistics
    print(f"\nDataset Summary:")
    print(f"  - Unique counties (FIPS): {df_base['fips'].nunique()}")
    if 'year' in df_base.columns:
        print(f"  - Year range: {df_base['year'].min()} - {df_base['year'].max()}")
        print(f"  - Unique years: {sorted(df_base['year'].unique())}")
    if 'month' in df_base.columns:
        print(f"  - Has monthly data: Yes")
    
    # Missing data summary
    print(f"\nMissing Data Summary:")
    missing_counts = df_base.isnull().sum()
    missing_pct = (missing_counts / len(df_base) * 100).round(2)
    missing_df = pd.DataFrame({
        'Column': missing_counts.index,
        'Missing Count': missing_counts.values,
        'Missing %': missing_pct.values
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    if not missing_df.empty:
        print(missing_df.head(10).to_string(index=False))
        print(f"  ... and {len(missing_df) - 10} more columns with missing data")
    else:
        print("  ✓ No missing data!")
    
    # Save consolidated dataset
    output_file = "consolidated_data_phase3.csv"
    df_base.to_csv(output_file, index=False)
    print(f"\n✓ Saved consolidated dataset to: {output_file}")
    print(f"  File size: {Path(output_file).stat().st_size / (1024*1024):.2f} MB")
    
    # Save summary report
    summary_file = "consolidation_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("DATA CONSOLIDATION SUMMARY - PHASE 3\n")
        f.write("="*80 + "\n\n")
        f.write(f"Dataset Shape: {df_base.shape[0]} rows x {df_base.shape[1]} columns\n")
        f.write(f"Unique Counties: {df_base['fips'].nunique()}\n")
        if 'year' in df_base.columns:
            f.write(f"Year Range: {df_base['year'].min()} - {df_base['year'].max()}\n")
        f.write("\nColumns:\n")
        for i, col in enumerate(df_base.columns, 1):
            f.write(f"  {i:3d}. {col}\n")
        f.write("\n" + "="*80 + "\n")
        f.write("Missing Data:\n")
        f.write(missing_df.to_string(index=False))
    
    print(f"✓ Saved summary report to: {summary_file}")

print("\n" + "="*80)
print("CONSOLIDATION COMPLETE!")
print("="*80)

