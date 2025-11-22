"""
Predict for top corn production county using hybrid model
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Find top corn production county
print("="*80)
print("FINDING TOP CORN PRODUCTION COUNTY")
print("="*80)

# Get project root (parent of Hybrid Model folder)
script_dir = Path(__file__).parent
project_root = script_dir.parent

# Data files are in project root
data_files = [
    project_root / 'consolidated_data_phase3.csv',
    project_root / 'consolidated_all_data.csv',
    project_root / 'consolidated_data_phase3_preprocessed.csv'
]
df = None

for file in data_files:
    if file.exists():
        df = pd.read_csv(file)
        print(f"Loaded data from {file}")
        break

if df is None:
    print(f"Checked project root: {project_root}")
    print(f"Available files in project root:")
    for f in sorted(project_root.glob('*.csv')):
        print(f"  - {f.name}")
    raise FileNotFoundError("No consolidated data file found in project root")

# Aggregate by county to find top producer
if 'month' in df.columns:
    county_yearly = df.groupby(['fips', 'county_name', 'year'])['corn_production_bu'].sum().reset_index()
else:
    county_yearly = df[['fips', 'county_name', 'year', 'corn_production_bu']].copy()

# Find top 10 counties by total production
all_counties = county_yearly.groupby(['fips', 'county_name'])['corn_production_bu'].sum().reset_index()
top_counties = all_counties.sort_values('corn_production_bu', ascending=False).head(10)

print(f"\nTop 10 Corn Production Counties:")
print("-" * 80)
for idx, row in top_counties.iterrows():
    rank = idx + 1
    print(f"  {rank}. {row['county_name']} (FIPS: {int(row['fips'])}) - {row['corn_production_bu']:,.0f} bushels")
print("-" * 80)

# Note: API key not needed - we're generating prompt files for manual upload
print(f"\nNote: LLM API call is disabled. Generating prompt files for manual Grok upload.")
print(f"\n{'='*80}")
print("GENERATING PROMPT FILES FOR TOP 10 COUNTIES")
print("="*80)

# System prompt (same for all counties)
system_prompt = """You are a conservative adjustment module for county-level corn yield.

Your ONLY inputs are a structured "factsheet" with exogenous signals that the ML model does not capture well.

You must ignore agronomic variables already modeled by ML (soils, ethanol distance, historical yields, GLDAS/PRISM windowed indices, USDM, NDVI, etc.).

Goal: Propose a small, bounded multiplicative adjustment to the provided ML baseline prediction based solely on the allowed facts.

Hard constraints

Use only these fields if present:

economy_population, economy_unemployment_rate_pct, economy_available_workers,

economy_required_hourly_wage_single_usd, economy_median_hourly_wage_region_usd,

economy_median_household_income_usd, selected industry wage/size metrics.

market_basis_anomaly_usd, elevator_outage_flag, rail_disruption_flag.

prevented_planting_flag, pp_share_pct, insurance_disaster_flag.

hail_event_flag, derecho_event_flag, localized_flood_flag.

nass_planting_progress_pct, nass_harvest_progress_pct (weekly progress summaries).

ML meta: ml_pred (bu/ac), ml_band_width (U−L), train_rmse.

Forbidden: any soils, ethanol distance, recent yields, or windowed agronomic/drought/canopy variables. If such fields appear, ignore them.

Compute a raw signed adjustment adj_raw in [-1, +1] reflecting only exogenous stress/support.

Negative signals (examples): very high unemployment, severe labor shortages, high required_hourly_wage vs regional median, active logistics disruptions, hail/derecho flags, widespread prevented planting.

Positive signals (examples): strong labor availability with low wage pressure and no disruptions.

The final applied adjustment is bounded and damped by ML confidence:

alpha = 0.15 (max magnitude).

kappa = exp( - ml_band_width / train_rmse ).

adj = clip(adj_raw, -alpha, +alpha) * (1 - kappa).

Calibrated caution: If evidence is mixed or weak, set adj_raw = 0.0. Never exceed the bounds.

Output JSON only with this schema (numbers only, no explanations in prose):

For a SINGLE county:
{
  "adj_raw": float,                  // in [-1, 1], before damping
  "adj": float,                      // after bounds & damping rule above
  "pred_yield_bu_ac": float,         // = ml_pred * (1 + adj)
  "low_80": float,                   // = (ml_pred - 0.5*ml_band_width) * (1 + 0.7*adj)
  "high_80": float,                  // = (ml_pred + 0.5*ml_band_width) * (1 + 0.7*adj)
  "drivers": [ "short phrase", ... ] // 3–5 concise drivers referencing ONLY allowed fields
}

For MULTIPLE counties (differentiated by FIPS):
{
  "<fips_code>": {
    "adj_raw": float,
    "adj": float,
    "pred_yield_bu_ac": float,
    "low_80": float,
    "high_80": float,
    "drivers": [ "short phrase", ... ]
  },
  "<fips_code>": { ... },
  ...
}

No chain-of-thought. Do not explain your reasoning; just fill the JSON.

If required fields are missing (e.g., ml_pred, ml_band_width, train_rmse), return:
{"adj_raw": 0.0, "adj": 0.0, "pred_yield_bu_ac": null, "low_80": null, "high_80": null, "drivers": ["insufficient_fields"]}

FIPS codes should be strings (e.g., "27129" not 27129)."""

import PyPDF2
generated_files = []
missing_factsheets = []

# Process top 10 counties
for rank, (idx, county_row) in enumerate(top_counties.iterrows(), 1):
    fips = int(county_row['fips'])
    county_name = county_row['county_name']
    total_production = county_row['corn_production_bu']
    
    print(f"\n[{rank}/10] Processing {county_name} (FIPS: {fips})...")
    
    # Get most recent year for this county
    recent_data = county_yearly[county_yearly['fips'] == fips].sort_values('year', ascending=False)
    if len(recent_data) > 0:
        latest_year = int(recent_data.iloc[0]['year'])
        predict_year = latest_year + 1
    else:
        predict_year = 2023
    
    # Estimate ML baseline
    if len(recent_data) > 0:
        recent_avg = recent_data.head(3)['corn_production_bu'].mean()
        ml_baseline = float(recent_avg)
    else:
        ml_baseline = float(total_production / 20)
    
    p10_ml = ml_baseline * 0.8
    p90_ml = ml_baseline * 1.2
    
    # Check if fact sheet exists
    fact_sheet_path = Path(__file__).parent / "Fact Sheet" / f"{fips}.pdf"
    if not fact_sheet_path.exists():
        print(f"  [WARN] Warning: Fact sheet not found at {fact_sheet_path}")
        missing_factsheets.append((county_name, fips))
        continue
    
    try:
        # Read PDF fact sheet
        with open(fact_sheet_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            factsheet_text = ""
            for page in pdf_reader.pages:
                factsheet_text += page.extract_text() + "\n"
        factsheet_text = factsheet_text.strip()
        print(f"  [OK] Extracted {len(factsheet_text)} characters from PDF")
        
        # Estimate ML metadata (these would come from actual ML model in production)
        # For now, estimate based on historical data
        if len(recent_data) > 0:
            # Estimate bushels per acre (rough estimate: assume ~150-200 bu/ac typical)
            # Convert total bushels to bu/ac using acres planted if available
            acres_planted = recent_data.iloc[0].get('corn_acres_planted', None) if 'corn_acres_planted' in recent_data.columns else None
            if acres_planted and acres_planted > 0:
                ml_pred_bu_ac = ml_baseline / acres_planted
            else:
                # Rough estimate: assume 180 bu/ac average
                ml_pred_bu_ac = 180.0
        else:
            ml_pred_bu_ac = 180.0
        
        # Estimate band width and RMSE (these would come from ML model in production)
        # Rough estimates: band width ~20% of prediction, RMSE ~10% of prediction
        ml_band_width = ml_pred_bu_ac * 0.20  # 20% uncertainty band
        train_rmse = ml_pred_bu_ac * 0.10     # 10% RMSE estimate
        
        # Build user prompt (single county format)
        user_prompt = f"""Analyze this corn production factsheet and provide a bounded adjustment to the ML baseline prediction.

County FIPS: {fips}

ML Meta:
- ml_pred (bu/ac): {ml_pred_bu_ac:.2f}
- ml_band_width (U−L): {ml_band_width:.2f}
- train_rmse: {train_rmse:.2f}

Fact Sheet Information:
{factsheet_text}

Provide your analysis as JSON following the required schema. Use ONLY the allowed fields from the fact sheet (economy, market, insurance, weather flags, NASS progress). Ignore any agronomic variables already modeled by ML.

Return ONLY the JSON object, no other text."""
        
        # Create output file
        output_file = Path(__file__).parent / f"grok_prompt_{county_name.replace(' ', '_')}_{fips}_{predict_year}.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("GROK PROMPT FOR MANUAL UPLOAD\n")
            f.write("="*80 + "\n\n")
            
            f.write("COUNTY INFORMATION\n")
            f.write("-"*80 + "\n")
            f.write(f"Rank: {rank} (Top 10 Corn Production Counties)\n")
            f.write(f"County: {county_name}\n")
            f.write(f"FIPS Code: {fips}\n")
            f.write(f"Year: {predict_year}\n")
            f.write(f"Total Production (all years): {total_production:,.0f} bushels\n")
            f.write(f"\n")
            
            f.write("ML BASELINE PREDICTION\n")
            f.write("-"*80 + "\n")
            f.write(f"ML Baseline (total): {ml_baseline:,.0f} bushels\n")
            f.write(f"ML Pred (bu/ac): {ml_pred_bu_ac:.2f}\n")
            f.write(f"ML Band Width: {ml_band_width:.2f}\n")
            f.write(f"Train RMSE: {train_rmse:.2f}\n")
            f.write(f"P10 (10th percentile): {p10_ml:,.0f} bushels\n")
            f.write(f"P90 (90th percentile): {p90_ml:,.0f} bushels\n")
            f.write(f"Uncertainty Interval: [{p10_ml:,.0f}, {p90_ml:,.0f}] bushels\n")
            f.write(f"\n")
            
            if len(recent_data) > 0:
                f.write("RECENT PRODUCTION HISTORY\n")
                f.write("-"*80 + "\n")
                for idx, row in recent_data.head(5).iterrows():
                    f.write(f"  {int(row['year'])}: {row['corn_production_bu']:,.0f} bushels\n")
                f.write(f"\n")
            
            f.write("="*80 + "\n")
            f.write("SYSTEM PROMPT (for Grok)\n")
            f.write("="*80 + "\n\n")
            f.write(system_prompt)
            f.write("\n\n")
            
            f.write("="*80 + "\n")
            f.write("USER PROMPT (for Grok)\n")
            f.write("="*80 + "\n\n")
            f.write(user_prompt)
            f.write("\n\n")
            
            f.write("="*80 + "\n")
            f.write("FACT SHEET CONTENT (from PDF)\n")
            f.write("="*80 + "\n\n")
            f.write(factsheet_text)
            f.write("\n\n")
            
            f.write("="*80 + "\n")
            f.write("INSTRUCTIONS\n")
            f.write("="*80 + "\n\n")
            f.write("1. Copy the SYSTEM PROMPT above and paste it as the system message in Grok\n")
            f.write("2. Copy the USER PROMPT above and paste it as the user message in Grok\n")
            f.write("3. Grok should return a JSON response with the required schema:\n")
            f.write("   - adj_raw: float in [-1, 1]\n")
            f.write("   - adj: float (after bounds & damping)\n")
            f.write("   - pred_yield_bu_ac: float (= ml_pred * (1 + adj))\n")
            f.write("   - low_80: float\n")
            f.write("   - high_80: float\n")
            f.write("   - drivers: array of 3-5 short phrases\n")
            f.write("4. The adjustment uses alpha=0.15 and damping based on ML confidence\n")
            f.write("5. Only use allowed fields from fact sheet (economy, market, insurance, weather flags, NASS progress)\n")
        
        generated_files.append((rank, county_name, fips, output_file, ml_baseline))
        print(f"  [OK] Generated: {output_file.name}")
        
    except Exception as e:
        print(f"  [ERROR] Error processing {county_name}: {e}")
        import traceback
        traceback.print_exc()

# Summary
print(f"\n{'='*80}")
print("SUMMARY")
print("="*80)
print(f"Successfully generated {len(generated_files)} prompt files:")
for rank, county_name, fips, output_file, ml_baseline in generated_files:
    print(f"  {rank}. {county_name} (FIPS: {fips}) - {output_file.name}")

if missing_factsheets:
    print(f"\n[WARN] Missing fact sheets ({len(missing_factsheets)}):")
    for county_name, fips in missing_factsheets:
        print(f"  - {county_name} (FIPS: {fips})")

print(f"\n{'='*80}")
print("All prompt files are ready for manual upload to Grok!")
print("="*80)

# Generate batch prompt file for all counties together
if len(generated_files) > 0:
    print(f"\n{'='*80}")
    print("GENERATING BATCH PROMPT FILE (ALL COUNTIES)")
    print("="*80)
    print("Creating a single prompt file with all counties for batch processing...")
    
    # Collect all county data
    batch_counties_data = []
    for rank, county_name, fips, output_file, ml_baseline in generated_files:
        # Get recent data for this county
        recent_data = county_yearly[county_yearly['fips'] == fips].sort_values('year', ascending=False)
        
        # Read fact sheet
        fact_sheet_path = Path(__file__).parent / "Fact Sheet" / f"{fips}.pdf"
        if fact_sheet_path.exists():
            with open(fact_sheet_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                factsheet_text = ""
                for page in pdf_reader.pages:
                    factsheet_text += page.extract_text() + "\n"
            factsheet_text = factsheet_text.strip()
            
            batch_counties_data.append({
                'rank': rank,
                'county_name': county_name,
                'fips': fips,
                'ml_baseline': ml_baseline,
                'factsheet_text': factsheet_text,
                'recent_data': recent_data
            })
    
    # Build batch user prompt
    batch_user_prompt = f"""Analyze the following corn production factsheets for multiple counties and provide bounded adjustments to the ML baseline predictions.

You will receive {len(batch_counties_data)} counties. For each county, analyze its fact sheet and provide an adjustment.

COUNTIES TO ANALYZE:
"""
    
    for county_data in batch_counties_data:
        # Calculate ML metadata for this county
        recent_data = county_data['recent_data']
        ml_baseline = county_data['ml_baseline']
        
        if len(recent_data) > 0:
            acres_planted = recent_data.iloc[0].get('corn_acres_planted', None) if 'corn_acres_planted' in recent_data.columns else None
            if acres_planted and acres_planted > 0:
                ml_pred_bu_ac = ml_baseline / acres_planted
            else:
                ml_pred_bu_ac = 180.0
        else:
            ml_pred_bu_ac = 180.0
        
        ml_band_width = ml_pred_bu_ac * 0.20
        train_rmse = ml_pred_bu_ac * 0.10
        
        batch_user_prompt += f"""
--- COUNTY {county_data['rank']}: {county_data['county_name']} (FIPS: {county_data['fips']}) ---
ML Meta:
- ml_pred (bu/ac): {ml_pred_bu_ac:.2f}
- ml_band_width (U−L): {ml_band_width:.2f}
- train_rmse: {train_rmse:.2f}

Fact Sheet Information:
{county_data['factsheet_text']}

"""
    
    # Set batch output file name after collecting data
    batch_output_file = Path(__file__).parent / f"grok_prompt_BATCH_all_{len(batch_counties_data)}_counties_{predict_year}.txt"
    
    batch_user_prompt += f"""
Provide your analysis as a SINGLE JSON object with FIPS codes as keys. Each county should have its own complete response following the required schema.

Example format (for multiple counties):
{{
  "{batch_counties_data[0]['fips']}": {{
    "adj_raw": -0.08,
    "adj": -0.06,
    "pred_yield_bu_ac": 169.2,
    "low_80": 150.5,
    "high_80": 188.0,
    "drivers": [
      "High unemployment rate indicating labor stress",
      "Elevated required wage vs regional median",
      "Rail disruption flag active"
    ]
  }},
  "{batch_counties_data[1]['fips'] if len(batch_counties_data) > 1 else batch_counties_data[0]['fips']}": {{
    "adj_raw": 0.05,
    "adj": 0.04,
    "pred_yield_bu_ac": 187.2,
    "low_80": 166.3,
    "high_80": 208.1,
    "drivers": [
      "Strong labor availability",
      "No logistics disruptions",
      "Favorable planting progress"
    ]
  }}
}}

IMPORTANT:
- Return ONE JSON object with all counties
- Use FIPS codes as keys (as strings, e.g., "{batch_counties_data[0]['fips']}")
- Each county must have the complete schema: adj_raw, adj, pred_yield_bu_ac, low_80, high_80, drivers
- Base each county's analysis ONLY on its own fact sheet data
- Use ONLY allowed fields (economy, market, insurance, weather flags, NASS progress)
- Ignore agronomic variables already modeled by ML

Return ONLY the JSON object, no other text."""
    
    # Write batch file
    with open(batch_output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"GROK BATCH PROMPT FOR MANUAL UPLOAD (ALL {len(batch_counties_data)} COUNTIES)\n")
        f.write("="*80 + "\n\n")
        
        f.write("BATCH INFORMATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Number of Counties: {len(batch_counties_data)}\n")
        f.write(f"Year: {predict_year}\n")
        f.write(f"\nCounties included:\n")
        for county_data in batch_counties_data:
            f.write(f"  {county_data['rank']}. {county_data['county_name']} (FIPS: {county_data['fips']}) - ML Baseline: {county_data['ml_baseline']:,.0f} bushels\n")
        f.write(f"\n")
        
        f.write("="*80 + "\n")
        f.write("SYSTEM PROMPT (for Grok)\n")
        f.write("="*80 + "\n\n")
        f.write(system_prompt)
        f.write("\n\n")
        
        f.write("="*80 + "\n")
        f.write("USER PROMPT (for Grok) - BATCH MODE\n")
        f.write("="*80 + "\n\n")
        f.write(batch_user_prompt)
        f.write("\n\n")
        
        f.write("="*80 + "\n")
        f.write("INSTRUCTIONS\n")
        f.write("="*80 + "\n\n")
        f.write("1. Copy the SYSTEM PROMPT above and paste it as the system message in Grok\n")
        f.write("2. Copy the USER PROMPT above and paste it as the user message in Grok\n")
        f.write("3. Grok should return a SINGLE JSON object with all counties\n")
        f.write("4. The JSON format should be: {<fips>: {adj_raw, adj, pred_yield_bu_ac, low_80, high_80, drivers}, ...}\n")
        f.write("5. Each county must have the complete schema with all required fields\n")
        f.write("6. Only use allowed fields from fact sheets (economy, market, insurance, weather flags, NASS progress)\n")
        f.write(f"\nExample response format:\n")
        f.write("{\n")
        for i, county_data in enumerate(batch_counties_data[:2]):  # Show first 2 as example
            f.write(f'  "{county_data["fips"]}": {{\n')
            f.write(f'    "adj_raw": -0.08,\n')
            f.write(f'    "adj": -0.06,\n')
            f.write(f'    "pred_yield_bu_ac": 169.2,\n')
            f.write(f'    "low_80": 150.5,\n')
            f.write(f'    "high_80": 188.0,\n')
            f.write(f'    "drivers": ["Driver 1", "Driver 2", "Driver 3"]\n')
            f.write(f'  }}{"," if i < 1 else ""}\n')
        if len(batch_counties_data) > 2:
            f.write(f'  ... (and {len(batch_counties_data) - 2} more counties)\n')
        f.write("}\n")
    
    print(f"\n[OK] Batch prompt file generated: {batch_output_file.name}")
    print(f"  This file contains all {len(batch_counties_data)} counties in a single prompt")
    print(f"  Grok will return one JSON object with all counties keyed by FIPS code")
    print(f"\n{'='*80}")

