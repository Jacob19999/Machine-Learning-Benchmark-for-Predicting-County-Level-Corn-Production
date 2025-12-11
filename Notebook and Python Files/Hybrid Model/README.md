# Hybrid Model - LLM Track with Bounded Reasoning

This module implements the hybrid model combining ML predictions with LLM adjustments using Grok API.

## Overview

The hybrid model consists of two main components:

1. **LLM Track**: Calls Grok API for bounded reasoning (adjustment a ∈ [-0.20, 0.20]) using PDF fact sheets
2. **Hybrid Aggregation**: Combines ML baseline with LLM adjustment

## Architecture

### LLM Track (Bounded Reasoning)

For each (county, year), the LLM analyzes a PDF fact sheet for the county (named by FIPS code, e.g., `27001.pdf` for Aitkin County). The fact sheet contains comprehensive county-level agricultural information.

The LLM returns strict JSON with:
- `adjustment`: float in [-0.20, 0.20]
- `drivers`: array of 2-4 short bullet points

Agronomic rules are embedded in the system prompt (e.g., if moisture deficit and VPD high → negative adj).

### 6. Hybrid Aggregation

#### 6.1 Adjust-Only Hybrid (default)

```
y^_hyb = y^_ML * (1 + clip(a, -0.2, 0.2))
P10_hyb = P10_ML * (1 + 0.7*a)
P90_hyb = P90_ML * (1 + 0.7*a)
```

#### 6.2 Stacked Hybrid (optional)

Forms z_c,t = [y^_EN, y^_EBM, y^_XGB, y^_LLM] using OOF predictions, and fits:

```
y^ = argmin_{w≥0, 1^T w=1} ||y - Zw||² + λ||w||²
```

We keep Adjust-Only as the operational default for stability and cost.

## Files

- `llm_track.py`: Grok API integration for LLM reasoning (reads PDF fact sheets)
- `hybrid_aggregation.py`: Hybrid aggregation methods
- `hybrid_model.py`: Main orchestration script
- `Fact Sheet/`: Directory containing PDF fact sheets named by FIPS code (e.g., `27001.pdf`)
- `API Key.txt`: Grok API key (keep secure)
- `Grok API Request format.txt`: Example API request format

## Usage

### Basic Usage

```python
from hybrid_model import HybridModel

# Initialize
model = HybridModel(
    api_key='your-api-key',
    ml_model_path='../xgboost_best_model.pkl',
    fact_sheet_dir='Fact Sheet',  # Directory with PDF fact sheets
    method='adjust_only'
)

# Single prediction
result = model.predict_single(
    fips=27001,  # County FIPS code (PDF will be loaded: 27001.pdf)
    year=2020,
    y_ml=15000000.0,  # ML baseline prediction
    p10_ml=12000000.0,
    p90_ml=18000000.0
)

print(f"ML: {result['y_ml']:,.0f}")
print(f"Adjustment: {result['adjustment']:.3f}")
print(f"Hybrid: {result['y_hyb']:,.0f}")
print(f"Drivers: {result['drivers']}")

# Batch prediction
results_df = model.predict_batch(
    test_data=test_df[['fips', 'year']],
    y_ml_predictions=y_pred_ml,
    p10_ml=p10_ml,
    p90_ml=p90_ml,
    max_samples=100  # Limit for testing
)

# Evaluate
metrics = model.evaluate(results_df, y_true)
print(f"R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:,.0f}")
```

### Command Line

```bash
python hybrid_model.py \
    --ml-model ../xgboost_best_model.pkl \
    --fact-sheet-dir "Fact Sheet" \
    --method adjust_only \
    --max-samples 10
```

## Requirements

```
pandas >= 1.5.0
numpy >= 1.23.0
scikit-learn >= 1.2.0
requests >= 2.28.0
scipy >= 1.9.0
PyPDF2 >= 3.0.0
```

## Notes

1. **PDF Fact Sheets**: Fact sheets must be named by FIPS code (e.g., `27001.pdf` for Aitkin County). The LLM reads the PDF content directly.

2. **API Costs**: Each prediction requires one API call. Consider batching and caching for production.

3. **Fallback**: If LLM parsing fails or PDF not found, the model falls back to ML baseline (adjustment = 0.0).

4. **Validation**: The model clips adjustments to [-0.20, 0.20] and validates JSON parsing.

5. **Uncertainty Bands**: P10/P90 are scaled conservatively by 0.7 * adjustment to avoid overconfidence.

## Future Improvements

- Cache PDF text extraction for efficiency
- Batch API calls if supported
- Add ensemble of multiple LLM calls for robustness
- Support alternative PDF libraries for better text extraction

