# Corn Yield Prediction Using Satellite and Environmental Data

A comprehensive machine learning project for predicting county-level corn production in Minnesota using satellite-derived environmental variables, economic indicators, and agricultural data.

## 📋 Project Overview

This project integrates multiple data sources to build predictive models for corn production. The methodology combines:
- **Satellite Data**: GLDAS (Global Land Data Assimilation System) environmental variables
- **Climate Data**: PRISM precipitation and temperature data
- **Agricultural Data**: Corn acres planted (NASA Quick Stats)
- **Economic Data**: Farm income, government payments, diesel prices
- **Spatial Data**: Distance to nearest ethanol processing facilities

**Target Variable**: County-level annual corn production (bushels)

**Time Period**: 2000-2022 (23 years)
**Geographic Scope**: Minnesota counties

## 🎯 Objectives

1. Integrate heterogeneous data sources into a unified dataset
2. Compare multiple machine learning models (simple to complex)
3. Identify optimal hyperparameters through systematic tuning
4. Evaluate model performance using comprehensive metrics
5. Analyze feature importance and relationships

## 📊 Data Sources

### Primary Datasets
- **GLDAS Data** (`Data/gldas_all_bands_data.csv`)
  - Satellite-derived environmental variables: albedo, surface temperature, soil moisture, evapotranspiration, radiation, etc.
  
- **Corn Production Data** (`Data/corn_harvest_planted_2000-2023.csv`)
  - NASA Quick Stats data
  - Includes: acres planted, acres harvested, production (bushels)
  - **Feature Used**: Acres planted as predictor variable

- **PRISM Climate Data** (`Data/PRISM_percipitation_data.csv`)
  - Monthly precipitation, temperature (min/mean/max), vapor pressure
  - Gridded climate data at 4km resolution

- **Economic Data** (`Data/enonomy_mn.csv`)
  - Farm-related income
  - Government program receipts
  - Available for years: 2002, 2007, 2012, 2017, 2022 (inconsistent)
  - **Handling**: Multi-strategy imputation (forward-fill, backward-fill, interpolation, median imputation)

- **Diesel Price Data** (`Data/diesel_price.csv`)
  - Monthly diesel fuel prices (USD/gallon)

- **Ethanol Distance** (`Data/ethanol_dist.csv`)
  - Distance to nearest ethanol processing facility (km)
  - Static feature (same for all years per county)

## 🤖 Models Implemented

### Simple Models
- **Polynomial Regression**: Degree-optimized with Ridge regularization
- **Random Forest**: Ensemble of decision trees with hyperparameter tuning

### Medium Complexity Models
- **Support Vector Machine (SVM)**: RBF and linear kernels with parameter optimization

### Complex Models
- **XGBoost**: Gradient boosting with comprehensive hyperparameter fine-tuning
- **TCN (Temporal Convolutional Network)**: Deep learning for temporal patterns (PyTorch-based)
- **MLP (Multi-Layer Perceptron)**: Neural network with transformer-inspired architecture

## 📁 Project Structure

```
cis631_research_project/
│
├── Data/                          # Raw data files
│   ├── corn_harvest_planted_2000-2023.csv
│   ├── diesel_price.csv
│   ├── enonomy_mn.csv
│   ├── ethanol_dist.csv
│   ├── gldas_all_bands_data.csv
│   └── PRISM_percipitation_data.csv
│
├── Paper/                        # Research documentation
│   └── Phase 3.docx             # Methodology document
│
├── Visuals/                      # Generated visualizations
│   ├── model_comparison_*.png
│   ├── xgboost_finetuning_analysis.png
│   └── EDA_*.png
│
├── consolidate_data_phase3.py   # Data consolidation script
├── preprocess_phase3_data.py    # Data preprocessing script
├── model_benchmarking.py        # Model comparison script
│
├── *.ipynb                       # Jupyter notebooks
│   ├── EDA_Phase3_Comprehensive.ipynb
│   ├── Model_Comparison.ipynb
│   └── SatelliteDataNew.ipynb
│
├── consolidated_data_phase3.csv  # Consolidated dataset (output)
├── X_train.csv, X_test.csv      # Preprocessed features
├── y_train.csv, y_test.csv      # Preprocessed targets
│
└── model outputs/
    ├── xgboost_best_model.pkl
    ├── model_comparison_results.csv
    └── xgboost_feature_importances.csv
```

## 🚀 Getting Started

### Prerequisites

```python
pandas >= 1.5.0
numpy >= 1.23.0
scikit-learn >= 1.2.0
xgboost >= 2.0.0
matplotlib >= 3.6.0
seaborn >= 0.12.0
torch >= 2.0.0  # Optional, for TCN model
```

### Installation

1. **Clone the repository** (or download the project files)

2. **Install required packages**:
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn
   ```
   
   For deep learning models (TCN, MLP):
   ```bash
   pip install torch
   ```

3. **Verify data files are present**:
   - All CSV files should be in the `Data/` directory
   - Base dataset: `combined_gldas_corn_data.csv` should exist

### Usage

#### Step 1: Data Consolidation
Consolidate all data sources into a unified dataset:
```bash
python consolidate_data_phase3.py
```
This will create:
- `consolidated_data_phase3.csv` - Combined dataset
- `consolidation_summary.txt` - Summary report

**Features:**
- Automatically checks for missing libraries
- Handles missing years in economic data with multi-strategy imputation
- Maps county names to FIPS codes
- Validates data integrity

#### Step 2: Data Preprocessing (Optional)
Preprocess the consolidated data:
```bash
python preprocess_phase3_data.py
```

#### Step 3: Model Benchmarking
Compare all models with hyperparameter tuning:
```bash
python model_benchmarking.py
```

**Features:**
- Automatically checks and installs required libraries
- Trains 6 different models with hyperparameter tuning
- Generates comprehensive performance metrics
- Creates visualization-ready results dictionary

#### Step 4: XGBoost Fine-Tuning (Optional)
Run comprehensive XGBoost hyperparameter optimization:
```python
# Run in Jupyter notebook or Python script
# See SatelliteDataNew.ipynb for details
```

The fine-tuning process includes:
- **Phase 1**: Baseline model with default parameters
- **Phase 2**: Coarse grid search (broad parameter space)
- **Phase 3**: Fine-tuning (narrowed parameter space, ~400 combinations)
- **Phase 4**: Final evaluation and feature importance analysis

**Estimated Runtime**: 
- Phase 2: ~1-2 hours
- Phase 3: ~30-60 minutes (on AMD 9900X with 12 cores)

## 📈 Model Performance

Models are evaluated using:
- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Squared Error (in bushels)
- **MAE**: Mean Absolute Error
- **Training Time**: Computational efficiency

Performance is reported on:
- **Log scale**: Models trained on log-transformed target
- **Original scale**: Metrics converted back to bushels

## 🔍 Key Features

### Data Handling
- **Missing Value Imputation**: Median imputation for numeric, forward/backward fill for categorical
- **Infinite Value Handling**: Detection and replacement
- **Temporal Consistency**: Handles inconsistent year coverage in economic data
- **FIPS Code Standardization**: Proper 5-digit FIPS code formatting (27001-27171 for Minnesota)

### Model Tuning
- **RandomizedSearchCV**: Efficient hyperparameter search
- **GridSearchCV**: Exhaustive search for fine-tuning
- **Cross-Validation**: 3-5 fold CV depending on model
- **Early Stopping**: For deep learning models

### Evaluation
- **Comprehensive Metrics**: Multiple performance measures
- **Visualization**: Model comparison plots, feature importance, residuals analysis
- **Feature Importance**: XGBoost and Random Forest provide interpretability

## 📝 Methodology

### Data Integration Approach
1. **Base Dataset**: Start with GLDAS corn production data (fips, year, month, features)
2. **Feature Addition**: Merge additional features by FIPS and year/month
3. **Temporal Aggregation**: Monthly to yearly (if needed)
4. **Missing Data**: Multi-strategy imputation based on data type and temporal patterns

### Model Selection Strategy
1. **Baseline**: Simple models establish performance floor
2. **Comparison**: Medium complexity models test nonlinear relationships
3. **Optimization**: Complex models fine-tuned for maximum performance

### Hyperparameter Tuning
- **Coarse Search**: Broad parameter ranges to identify optimal regions
- **Fine-Tuning**: Narrow ranges around best parameters
- **Reduced Search Space**: Optimized to complete in ~1 hour for Phase 3

## 🛡️ Data Leakage Prevention

### What is Data Leakage?

Data leakage occurs when information from the future or test set inadvertently influences model training, leading to overly optimistic performance estimates that don't generalize to new data. This is a critical issue that can invalidate research findings.

### Types of Data Leakage Prevented

#### 1. **Temporal Data Leakage**
- **Risk**: Using future information to predict past values
- **Prevention**: 
  - **Temporal Train-Test Split**: Data split by time period (2000-2019 for training, 2020-2022 for testing)
  - No future data is used to predict past years
  - Economic data imputation only uses historical values (forward-fill, not backward-fill from future)

#### 2. **Preprocessing Data Leakage**
- **Risk**: Computing statistics (mean, median, std) on entire dataset before splitting
- **Prevention**:
  - **Scaler Fitting**: `StandardScaler` is fit exclusively on training data
    ```python
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit on train only
    X_test_scaled = scaler.transform(X_test)        # Transform test using train stats
    ```
  - **Imputation**: `SimpleImputer` fit on training data only
    ```python
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)  # Learn median from train
    X_test_imputed = imputer.transform(X_test)        # Use train median on test
    ```

#### 3. **Target Leakage**
- **Risk**: Using target-derived features that wouldn't be available at prediction time
- **Prevention**:
  - **Feature Selection**: Only use predictive features, not production-derived metrics
  - **Acres Planted**: Used as feature (available at planting time), not acres harvested
  - **No Future Production Data**: Production data from future years never influences training

#### 4. **Cross-Validation Data Leakage**
- **Risk**: Information leaking across CV folds
- **Prevention**:
  - **Scikit-learn Pipeline**: Ensures preprocessing is refit for each fold
  - **Temporal CV**: When using time-based CV, ensures no future data in training folds
  - **GridSearchCV**: Automatically handles proper splitting within each fold

#### 5. **Feature Engineering Leakage**
- **Risk**: Creating features using test set information
- **Prevention**:
  - All feature engineering (polynomial features, interactions) done on training set
  - Transformers fit on training, applied to test:
    ```python
    poly_features = PolynomialFeatures(degree=degree)
    X_train_poly = poly_features.fit_transform(X_train_scaled)  # Fit on train
    X_test_poly = poly_features.transform(X_test_scaled)        # Apply to test
    ```

### Specific Prevention Measures in This Project

#### Data Consolidation Phase (`consolidate_data_phase3.py`)
- ✅ **No Target Information**: Only uses predictive features (acres planted, not production)
- ✅ **Temporal Ordering**: Economic data imputation respects temporal order (forward-fill)
- ✅ **Separate Processing**: Each data source processed independently before merging

#### Preprocessing Phase (`preprocess_phase3_data.py`)
- ✅ **Train-Test Split First**: Data split before any preprocessing
- ✅ **Separate Scalers**: Each preprocessing step fit on training only
- ✅ **No Test Statistics**: Test set never used to compute normalization parameters

#### Model Training Phase (`model_benchmarking.py`, `SatelliteDataNew.ipynb`)
- ✅ **Strict Temporal Split**: 
  - Training: Years 2000-2019
  - Testing: Years 2020-2022
- ✅ **Independent Scaling**: Test data scaled using training statistics only
- ✅ **Cross-Validation**: CV folds respect temporal order (no time shuffling)
- ✅ **Hyperparameter Tuning**: GridSearchCV ensures no leakage across CV folds

#### Feature Selection
- ✅ **Only Predictive Features**: Features available at prediction time
  - ✅ Acres planted (known at planting)
  - ✅ Environmental variables (historical/current)
  - ✅ Economic indicators (historical, forward-filled)
  - ❌ Production data (target variable)
  - ❌ Harvest data (occurs after planting)

### Validation Protocol

1. **Temporal Split Verification**:
   - Training set: Years 2000-2019 (historical data)
   - Test set: Years 2020-2022 (future data)
   - No overlap or cross-contamination

2. **Preprocessing Verification**:
   - All scalers/imputers fit on training data
   - Test data transformed using training parameters
   - No statistics computed from test set

3. **Feature Verification**:
   - All features available at prediction time
   - No target-derived features
   - No future information leakage

### Code Examples of Prevention

```python
# ✅ CORRECT: Fit scaler on training data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Learn from train
X_test_scaled = scaler.transform(X_test)        # Apply to test

# ❌ WRONG: Would cause leakage
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(pd.concat([X_train, X_test]))  # DON'T DO THIS

# ✅ CORRECT: Temporal split before preprocessing
train_mask = df['year'] <= 2019
X_train = df[train_mask].drop(columns=['corn_production_bu'])
X_test = df[~train_mask].drop(columns=['corn_production_bu'])

# ✅ CORRECT: Imputation fit on training only
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
```

### Impact on Model Evaluation

Proper data leakage prevention ensures:
- **Realistic Performance Estimates**: Test scores reflect true generalization ability
- **Reliable Model Comparison**: Models compared fairly without artificial inflation
- **Production Readiness**: Models that perform well on test set will perform similarly in production
- **Scientific Validity**: Research findings are credible and reproducible

### Warning Signs to Watch For

If you see these, data leakage may be present:
- ⚠️ Test performance suspiciously high (R² > 0.99) without obvious reason
- ⚠️ Test performance significantly better than cross-validation scores
- ⚠️ Preprocessing fit on entire dataset before splitting
- ⚠️ Features derived from target variable
- ⚠️ Future data used to predict past values
- ⚠️ Same scaler/transformer fit on train+test combined

## 📊 Outputs

### Generated Files
- `consolidated_data_phase3.csv` - Final consolidated dataset
- `model_comparison_results.csv` - Model performance comparison
- `xgboost_best_model.pkl` - Trained XGBoost model
- `xgboost_feature_importances.csv` - Feature importance rankings
- `model_benchmark_comparison.png` - Comprehensive visualization (if visualization code is run)

### Visualizations
- Model performance comparisons (bar charts, scatter plots)
- Feature importance plots
- Residual analysis
- Prediction vs actual scatter plots
- Training time comparisons

## 🔬 Research Context

This project is part of Phase 3 research methodology focusing on:
- Integration of heterogeneous data sources
- Comprehensive model comparison (simple vs complex)
- Hyperparameter optimization
- Feature engineering and selection

## ⚠️ Important Notes

### Data Quality
- Economic data has missing years (only available for 2002, 2007, 2012, 2017, 2022)
- Missing values are imputed using multi-strategy approach
- Some models may skip if dependencies are unavailable (TCN requires PyTorch)

### Performance Considerations
- Model training can be computationally intensive
- Use `n_jobs=-1` for parallel processing
- XGBoost fine-tuning Phase 3: ~400 combinations × 5 folds = ~2000 fits
- Estimated runtime: 30-60 minutes on AMD 9900X (12 cores)

### Data Requirements
- Ensure all CSV files are in the `Data/` directory
- Base dataset (`combined_gldas_corn_data.csv`) should be present
- Scripts check for file existence and provide helpful error messages

## 🤝 Contributing

This is a research project. For questions or issues:
1. Check the consolidation summary report
2. Review the methodology document (Paper/Phase 3.docx)
3. Examine notebook outputs for detailed error messages

## 📄 License

Research project - See paper documentation for details.

## 📚 References

- GLDAS: Global Land Data Assimilation System
- PRISM: Parameter-elevation Regressions on Independent Slopes Model
- NASA Quick Stats: Agricultural Census Data
- XGBoost: Chen & Guestrin (2016)
- TCN: Temporal Convolutional Networks for sequence modeling

## 📧 Contact

For questions about methodology or data sources, refer to:
- `Research_Paper.md` - Detailed research documentation
- `Research_Insights.md` - Analysis insights
- `Paper/Phase 3.docx` - Methodology document

---

**Last Updated**: 2024
**Project Status**: Active Research
