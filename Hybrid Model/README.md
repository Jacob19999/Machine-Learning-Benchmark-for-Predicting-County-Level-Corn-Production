# Hybrid Model - LLM Track with Bounded Reasoning

This module implements the hybrid model combining ML predictions with LLM adjustments using Grok API.

## Overview

The hybrid model consists of three main components:

1. **Factsheet Builder**: Constructs numeric factsheet per (county, year)
2. **LLM Track**: Calls Grok API for bounded reasoning (adjustment a ∈ [-0.20, 0.20])
3. **Hybrid Aggregation**: Combines ML baseline with LLM adjustment

## Architecture

### 5.2 LLM Track (Bounded Reasoning)

For each (county, year), we construct a numeric factsheet containing:

- **Soils**: AWC, OM, drainage (approximated from available data)
- **CDL fraction**: Placeholder (would need CDL data)
- **Ethanol distance**: From county to nearest ethanol plant
- **Pre-season price signals**: Dec futures avg/vol, basis, prior-harvest cash (placeholder)
- **Windowed indices**: MB_w, VPD, thermal, radiation, root moisture, wind
- **Drought**: %D2+_w (placeholder, would need drought monitor data)
- **Recent yields**: y_c,t-1, y_c,t-2, y_c,t-3
- **ML baseline**: y^_ML (from best ML model)

The LLM returns strict JSON with:
- `adjustment`: float in [-0.20, 0.20]
- `drivers`: array of 2-4 short bullet points

Agronomic rules are embedded in the system prompt (e.g., if MB_w << 0 and VPD high → negative adj).

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

- `factsheet_builder.py`: Constructs factsheets per (county, year)
- `llm_track.py`: Grok API integration for LLM reasoning
- `hybrid_aggregation.py`: Hybrid aggregation methods
- `hybrid_model.py`: Main orchestration script
- `API Key.txt`: Grok API key (keep secure)
- `Grok API Request format.txt`: Example API request format

## Usage

### Basic Usage

```python
from hybrid_model import HybridModel

# Initialize
model = HybridModel(
    data_path='../consolidated_data_phase3.csv',
    api_key='your-api-key',
    ml_model_path='../xgboost_best_model.pkl',
    method='adjust_only'
)

# Single prediction
result = model.predict_single(
    fips=27001,
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
    --data ../consolidated_data_phase3.csv \
    --ml-model ../xgboost_best_model.pkl \
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
```

## Notes

1. **Missing Features**: Some factsheet features (AWC, OM, drainage, CDL, futures prices, drought) are placeholders. In production, these would need actual data sources.

2. **API Costs**: Each prediction requires one API call. Consider batching and caching for production.

3. **Fallback**: If LLM parsing fails, the model falls back to ML baseline (adjustment = 0.0).

4. **Validation**: The model clips adjustments to [-0.20, 0.20] and validates JSON parsing.

5. **Uncertainty Bands**: P10/P90 are scaled conservatively by 0.7 * adjustment to avoid overconfidence.

## Future Improvements

- Add actual SSURGO data for soil properties
- Integrate CDL data for crop fraction
- Add commodity futures API integration
- Integrate US Drought Monitor data
- Cache factsheets for efficiency
- Batch API calls if supported
- Add ensemble of multiple LLM calls for robustness

