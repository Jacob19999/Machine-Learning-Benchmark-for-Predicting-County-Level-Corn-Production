"""
Example usage of Hybrid Model
"""

import pandas as pd
import numpy as np
from pathlib import Path
from hybrid_model import HybridModel

# Configuration
API_KEY_PATH = Path(__file__).parent / "API Key.txt"
ML_MODEL_PATH = '../xgboost_best_model.pkl'  # Optional
FACT_SHEET_DIR = "Fact Sheet"  # Directory containing PDF fact sheets

# Read API key
with open(API_KEY_PATH, 'r') as f:
    API_KEY = f.read().strip()

# Initialize hybrid model
print("Initializing hybrid model...")
model = HybridModel(
    api_key=API_KEY,
    ml_model_path=ML_MODEL_PATH if Path(ML_MODEL_PATH).exists() else None,
    fact_sheet_dir=FACT_SHEET_DIR,
    method='adjust_only'
)

# Example 1: Single prediction
print("\n" + "="*80)
print("EXAMPLE 1: Single Prediction")
print("="*80)

result = model.predict_single(
    fips=27001,  # Aitkin County
    year=2020,
    y_ml=15000000.0,  # ML baseline: 15M bushels
    p10_ml=12000000.0,
    p90_ml=18000000.0
)

print(f"\nFIPS: {result['fips']}, Year: {result['year']}")
print(f"ML Baseline: {result['y_ml']:,.0f} bushels")
print(f"LLM Adjustment: {result['adjustment']:.3f}")
print(f"Hybrid Prediction: {result['y_hyb']:,.0f} bushels")
print(f"\nLLM Drivers:")
if result['drivers']:
    for driver in result['drivers']:
        print(f"  - {driver}")
else:
    print("  (No drivers - fallback used)")

# Example 2: Batch prediction (limited samples for demo)
print("\n" + "="*80)
print("EXAMPLE 2: Batch Prediction")
print("="*80)

# Create sample test data (FIPS codes and years)
test_data = pd.DataFrame({
    'fips': [27001, 27003, 27005, 27007, 27009],  # Sample FIPS codes
    'year': [2020, 2020, 2020, 2020, 2020]
})

# Get ML predictions (if model loaded)
if model.ml_model is not None:
    # This would require feature preparation - simplified for example
    print("Note: ML predictions would be computed from features")
    print("For demo, using dummy ML predictions")
    y_ml_dummy = np.random.uniform(10000000, 20000000, len(test_data))
else:
    print("Using dummy ML predictions (no model loaded)")
    y_ml_dummy = np.random.uniform(10000000, 20000000, len(test_data))

# Compute uncertainty intervals
p10_ml = y_ml_dummy * 0.8
p90_ml = y_ml_dummy * 1.2

# Batch predict (limit to 5 samples for demo)
print(f"\nProcessing {min(5, len(test_data))} samples...")
results_df = model.predict_batch(
    test_data=test_data[['fips', 'year', 'county_name']].head(5),
    y_ml_predictions=y_ml_dummy[:5],
    p10_ml=p10_ml[:5],
    p90_ml=p90_ml[:5],
    use_fallback=True,
    max_samples=5
)

print("\nResults:")
print(results_df[['fips', 'year', 'y_ml', 'adjustment', 'y_hyb']].to_string(index=False))

# Example 3: Evaluation (if true values available)
print("\n" + "="*80)
print("EXAMPLE 3: Evaluation")
print("="*80)

if 'corn_production_bu' in test_data.columns:
    y_true = test_data['corn_production_bu'].head(5).values
    
    metrics = model.evaluate(results_df, y_true)
    
    print("\nEvaluation Metrics:")
    print(f"  R² Score: {metrics['r2']:.4f}")
    print(f"  RMSE: {metrics['rmse']:,.0f} bushels")
    print(f"  MAE: {metrics['mae']:,.0f} bushels")
    print(f"  MAPE: {metrics['mape']:.2f}%")
else:
    print("True values not available for evaluation")

print("\n" + "="*80)
print("Examples complete!")
print("="*80)

