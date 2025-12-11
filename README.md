# Machine Learning Benchmark for Predicting County-Level Corn Production in Minnesota

A comprehensive machine learning research project for predicting county-level corn production in Minnesota using multi-source satellite, environmental, and economic data. This project systematically compares eight machine learning algorithms across different complexity levels and achieves state-of-the-art predictive performance.

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

Eight machine learning algorithms were systematically evaluated across different complexity levels:

### Simple Models
- **Polynomial Regression**: Degree-2 polynomial features with Ridge regularization (α = 100.0)
- **Random Forest**: Ensemble of 200 decision trees with bootstrap aggregation

### Medium Complexity Models
- **Support Vector Machine (SVM)**: RBF kernel with epsilon-SVR algorithm

### Complex Models
- **XGBoost**: Gradient boosting with tree learners, L1/L2 regularization (R² = 0.9910)
- **LightGBM**: Leaf-wise gradient boosting with histogram-based algorithm (**Best Model: R² = 0.9930**)
- **TabNet**: Deep learning architecture with attention mechanism for tabular data
- **LSTM (Long Short-Term Memory)**: Temporal neural network for sequential patterns
- **TCN (Temporal Convolutional Network)**: Temporal convolutional network with progressive capacity reduction

## 📁 Project Structure

```
cis631_research_project/
│
├── Data/                          # Raw data files
│   ├── 0_consolidated_all_data.csv
│   ├── 1_diesel_price.csv
│   ├── 2_enonomy_mn.csv
│   ├── 3_ethanol_dist.csv
│   ├── 4_gldas_all_bands_data.csv
│   ├── 5_corn_harvest_planted_2000-2023.csv
│   ├── 6_prism_.csv
│   ├── 7_mn labour.csv
│   └── Misc/                      # Processed data and outputs
│       ├── consolidated_data_phase3.csv
│       ├── consolidated_data_phase3_preprocessed.csv
│       ├── X_train.csv, X_test.csv
│       ├── y_train.csv, y_test.csv
│       ├── model_comparison_results.csv
│       ├── xgboost_feature_importances.csv
│       └── [other processed data files]
│
├── Notebook and Python Files/     # Code and notebooks
│   ├── Python Scripts/            # Main Python scripts
│   │   ├── consolidate_data_phase3.py
│   │   ├── preprocess_phase3_data.py
│   │   ├── model_benchmarking.py
│   │   ├── feature_importance_analysis.py
│   │   ├── compare_ml_vs_hybrid.py
│   │   └── [other utility scripts]
│   │
│   ├── Notebook/                  # Jupyter notebooks
│   │   ├── EDA_Phase3_Comprehensive.ipynb
│   │   ├── Model_Comparison.ipynb
│   │   ├── Model_Comparison_NoAcres.ipynb
│   │   └── SatelliteData.ipynb
│   │
│   ├── Hybrid Model/              # Hybrid ML-LLM model implementation
│   │   ├── hybrid_model.py
│   │   ├── hybrid_aggregation.py
│   │   ├── llm_track.py
│   │   ├── predict_top_county.py
│   │   ├── example_usage.py
│   │   ├── requirements.txt
│   │   ├── README.md
│   │   ├── Fact Sheet/            # County fact sheets (PDFs)
│   │   ├── Prompt/                # LLM prompt templates
│   │   └── [JSON responses and outputs]
│   │
│   ├── Model Files/               # Trained model artifacts
│   │   ├── xgboost_best_model.pkl
│   │   ├── robust_scaler_phase3.pkl
│   │   ├── cnn_corn_yield_model.keras
│   │   └── tcn_corn_yield_model.keras
│   │
│   └── Minnesota Shp Files/       # Geographic shapefiles
│       └── [Minnesota county boundary files]
│
├── Research Paper/                # Complete research paper
│   ├── Complete_Research_Paper.txt
│   ├── Complete_Research_Paper_Words.txt
│   ├── Research paper Phase 3 Final.pdf
│   ├── Phase 3 Data Methdology and model results only.pdf
│   ├── Phase 3 Intro Section business case .pdf
│   ├── best practice Latex.md
│   └── Visuals/                   # Research paper visualizations
│       ├── EDA_*.png              # Exploratory data analysis
│       ├── model_comparison_*.png
│       ├── Feature Importance*.png
│       └── xgboost_finetuning_analysis.png
│
├── Conference Paper/              # Conference paper version
│   ├── Final Paper/               # Final conference paper
│   │   ├── From Satellite To Silos Final.tex
│   │   ├── From Satellite To Silos Final.pdf
│   │   └── [other formats]
│   ├── From Satellite To Silos.docx
│   ├── From Satellite to Silos.pdf
│   ├── best practice Latex.md
│   ├── diagram_flow.png
│   ├── Overview diagram.svg
│   └── Visuals/                   # Conference paper visuals
│       ├── model_comparison_*.png
│       ├── model_scatter_*.png
│       └── lightgbm_top15_feature_importance.png
│
├── Outputs/                       # Generated visualizations and outputs
│   ├── conference paper visuals/  # Conference-specific visuals
│   ├── model_comparison_*.png     # Model performance comparisons
│   ├── model_scatter_*.png        # Prediction scatter plots
│   ├── feature_importance_comparison.png
│   ├── ml_vs_hybrid_comparison.png
│   └── visualizations.pdf
│
├── Research_Paper.md              # Research documentation
├── Research_Insights.md           # Analysis insights
├── consolidation_summary.txt      # Data consolidation report
├── Model Desc.txt                 # Model descriptions
└── README.md                      # This file
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
   - Base dataset: `Data/0_consolidated_all_data.csv` should exist

### Usage

#### Step 1: Data Consolidation
Consolidate all data sources into a unified dataset:
```bash
cd "Notebook and Python Files/Python Scripts"
python consolidate_data_phase3.py
```
This will create:
- `Data/Misc/consolidated_data_phase3.csv` - Combined dataset
- `consolidation_summary.txt` - Summary report (in project root)

**Features:**
- Automatically checks for missing libraries
- Handles missing years in economic data with multi-strategy imputation
- Maps county names to FIPS codes
- Validates data integrity

#### Step 2: Data Preprocessing (Optional)
Preprocess the consolidated data:
```bash
cd "Notebook and Python Files/Python Scripts"
python preprocess_phase3_data.py
```

#### Step 3: Model Benchmarking
Compare all models with hyperparameter tuning:
```bash
cd "Notebook and Python Files/Python Scripts"
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
# Run in Jupyter notebook
# See Notebook and Python Files/Notebook/SatelliteData.ipynb for details
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

### Best Performing Model: LightGBM
- **R² Score**: 0.9930 (99.30% variance explained)
- **RMSE**: 1,137,001 bushels
- **MAE**: 526,195 bushels
- **MAPE**: 29.44%

### Top 3 Models
1. **LightGBM**: R² = 0.9930, RMSE = 1,137,001 bushels
2. **XGBoost**: R² = 0.9910, RMSE = 1,288,613 bushels
3. **TabNet**: R² = 0.9599, RMSE = 2,719,162 bushels

### Evaluation Metrics
Models are evaluated using:
- **R² Score**: Coefficient of determination (primary metric)
- **RMSE**: Root Mean Squared Error (in bushels)
- **MAE**: Mean Absolute Error (in bushels)
- **MAPE**: Mean Absolute Percentage Error (%)

### Performance Configuration
- **Training Set**: Years 2000-2019 (10,400 samples, 86.5%)
- **Test Set**: Years 2020-2022 (1,620 samples, 13.5%)
- **Temporal Split**: Prevents data leakage by maintaining temporal separation
- **Evaluation**: Metrics computed on original scale (after inverse log transformation)

### Dual-Configuration Analysis
Models evaluated with:
1. **Full feature set**: 66 features including `corn_acres_planted`
2. **Reduced feature set**: 65 features excluding `corn_acres_planted`

LightGBM maintains strong performance even without primary feature (R² = 0.9780), demonstrating robust compensatory mechanisms.

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
1. **Base Dataset**: GLDAS corn production data (fips, year, month, features)
2. **Feature Addition**: Merge additional features by FIPS and year/month using left joins
3. **Data Sources**: Five primary sources integrated:
   - GLDAS satellite-derived environmental variables
   - USDA corn harvest and planted acres data
   - Economic indicators from USDA Economic Research Service
   - Diesel fuel prices (monthly)
   - PRISM precipitation and temperature data
4. **Temporal Aggregation**: Monthly to yearly aggregation where appropriate
5. **Missing Data**: Multi-strategy imputation (forward-fill, backward-fill, interpolation, median imputation)

### Feature Engineering
Created **66 informative features** across multiple categories:
- **Temporal**: Year trend, cyclical month encoding (sine/cosine)
- **Soil**: Average moisture, moisture gradient
- **Temperature**: Average, range, PCA components
- **Water Balance**: Precipitation-evaporation balance, efficiency metrics
- **Agricultural Context**: Yield per acre, fuel cost proxy
- **Economic**: Revenue sources, revenue per bushel
- **Spatial**: Ethanol plant distance categories

### Model Selection Strategy
1. **Simple Models**: Polynomial Regression, Random Forest (baseline performance)
2. **Medium Complexity**: SVM (nonlinear pattern capture)
3. **Complex Models**: XGBoost, LightGBM, TabNet, LSTM, TCN (state-of-the-art approaches)

### Hyperparameter Tuning
- **RandomizedSearchCV**: Efficient sampling of parameter space (15-25 iterations)
- **GridSearchCV**: Exhaustive search for fine-tuning (reduced space for speed)
- **Cross-Validation**: 3-fold CV for most models
- **Early Stopping**: Applied for gradient boosting and deep learning models

### Data Preprocessing
- **Target Transformation**: Log transformation (log1p) to handle right-skewed distribution
- **Feature Scaling**: RobustScaler (median and IQR-based, robust to outliers)
- **Missing Value Handling**: Multi-strategy imputation with temporal and spatial context
- **Outlier Handling**: RobustScaler inherently handles outliers while preserving information

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

#### Model Training Phase (`model_benchmarking.py`, `SatelliteData.ipynb`)
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

## 📊 Outputs

### Generated Files
- `Data/Misc/consolidated_data_phase3.csv` - Final consolidated dataset
- `Data/Misc/model_comparison_results.csv` - Model performance comparison
- `Notebook and Python Files/Model Files/xgboost_best_model.pkl` - Trained XGBoost model
- `Data/Misc/xgboost_feature_importances.csv` - Feature importance rankings
- `Outputs/` - All visualization outputs

### Visualizations
All visualizations are saved in the `Outputs/` directory:
- Model performance comparisons (bar charts, scatter plots)
- Feature importance plots
- Residual analysis
- Prediction vs actual scatter plots
- Training time comparisons
- Conference paper specific visuals in `Outputs/conference paper visuals/`

## 🔬 Research Context

This project represents a comprehensive benchmark study comparing eight machine learning algorithms for agricultural yield prediction. The research contributes to the agricultural yield prediction literature by:

### Key Research Contributions
1. **Comprehensive Benchmark**: Systematic comparison of 8 algorithms across complexity levels
2. **Multi-Source Integration**: Integration of 5 heterogeneous data sources (satellite, economic, agricultural, climate, infrastructure)
3. **Feature Engineering**: Creation of 66 informative features through comprehensive engineering
4. **Dual-Configuration Analysis**: Evaluation with and without primary feature to assess redundancy
5. **Methodological Rigor**: Temporal splitting, multi-strategy imputation, robust scaling

### Key Findings
- **Gradient Boosting Dominance**: LightGBM and XGBoost achieve R² > 0.99, significantly outperforming all other approaches
- **Feature Importance Hierarchy**: Agricultural context (48.2% importance) > Economic indicators > Environmental variables
- **Compensatory Mechanisms**: Models maintain strong performance (R² = 0.9780) even without primary feature
- **Optimal Complexity**: Medium-complexity gradient boosting provides best balance of performance and efficiency

### Research Paper
A complete research paper is available in `Research Paper/` following IMRaD structure:
- Introduction and Literature Review
- Data Sources and Methodology
- Results and Model Comparison
- Discussion and Conclusions
- Complete Bibliography (APA style)
- **Files**: `Complete_Research_Paper.txt`, `Research paper Phase 3 Final.pdf`

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
- Base dataset (`Data/0_consolidated_all_data.csv`) should be present
- Scripts check for file existence and provide helpful error messages

## 🤝 Contributing

This is a research project. For questions or issues:
1. Check the consolidation summary report (`consolidation_summary.txt`)
2. Review the research paper (`Research Paper/Research paper Phase 3 Final.pdf`)
3. Examine notebook outputs for detailed error messages

## 📄 License

Research project - See paper documentation for details.

## 📚 Key References

### Data Sources
- **GLDAS**: Global Land Data Assimilation System (Rodell et al., 2004)
- **PRISM**: Parameter-elevation Regressions on Independent Slopes Model
- **USDA Quick Stats**: Agricultural Census Data
- **USDA Economic Research Service**: Economic indicators

### Algorithm References
- **XGBoost**: Chen & Guestrin (2016) - Scalable Tree Boosting System
- **LightGBM**: Ke et al. (2017) - Highly Efficient Gradient Boosting Decision Tree
- **TabNet**: Arik & Pfister (2021) - Attentive Interpretable Tabular Learning
- **TCN**: Bai et al. (2018) - Empirical Evaluation of Generic Convolutional and Recurrent Networks

### Complete Bibliography
See `Research Paper/Complete_Research_Paper.txt` for full reference list (17 references, APA style).

## 👥 Authors

- Tang Zi Jian (Jacob)
- Emma Wele
- Victor C
- Onishi Rei

## 📧 Contact

For questions about methodology or data sources, refer to:
- `Research Paper/Research paper Phase 3 Final.pdf` - Complete research paper
- `Research_Paper.md` - Detailed research documentation
- `Research_Insights.md` - Analysis insights
- `Conference Paper/Final Paper/` - Conference paper version

---

**Last Updated**: 2024
**Project Status**: Research Complete - Paper Ready for Submission
