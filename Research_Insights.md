# Corn Yield Prediction Research - Comprehensive Insights Document

## 1. Data Gathering

### Data Sources
- **Primary Dataset**: Combined GLDAS (Global Land Data Assimilation System) corn data
- **File**: `combined_gldas_corn_data.csv`
- **Data Type**: Satellite-derived environmental variables combined with county-level corn production data
- **Temporal Coverage**: Years 2000-2022 (23 years of data)
- **Spatial Coverage**: Minnesota counties (FIPS codes)

### Data Collection Process
- Monthly aggregated environmental data from GLDAS system
- County-level corn production data (bushels) from USDA NASS
- Data aggregated from monthly to yearly level by county
- Initial dataset contained 2,001 observations across multiple counties

### Key Variables Collected
- **Environmental Variables**: Albedo, Average Surface Temperature, Canopy Interception, Evapotranspiration components (ESoil, ECanop, Evap), Land Surface Temperature, Longwave/Shortwave Radiation, Precipitation, Soil Moisture (multiple depths), Soil Temperature (multiple depths), Wind Speed, Vegetation Temperature
- **Target Variable**: `corn_production_bu` - County-level corn production in bushels

---

## 2. Data Discovery

### Initial Data Inspection
- **Total Observations**: 2,001 yearly county-level records
- **Features**: ~40 environmental and derived features
- **Zero Production Records**: 283 records with zero corn production (filtered out)
- **Final Dataset**: 1,718 observations with non-zero production

### Data Quality Findings
- **Missing Values**: Minimal missing values after aggregation
- **Data Range**: 
  - Production: 5,900 to 56,800,000 bushels
  - Mean production: ~16.1 million bushels
  - Standard deviation: ~12.3 million bushels
- **Temporal Distribution**: 
  - Years 2000-2019: Training data (1,109 observations)
  - Years 2020-2022: Test data (609 observations)
- **County Distribution**: Multiple counties across Minnesota represented

### Key Data Characteristics
- **Temporal Trend**: Overall increasing production trend from 2000-2019 (2.84% per year), followed by slight decline in 2020-2022 (-0.86% per year)
- **Peak Production Year**: 2016 (20,173,618 bushels average)
- **Lowest Production Year**: 2001 (10,201,820 bushels average)
- **Right-Skewed Distribution**: Production data shows positive skewness, indicating some high-producing counties/years

---

## 3. Exploratory Data Analysis

### 3.1 Temporal Analysis

#### Yearly Production Trends
- **Overall Growth**: Steady increase in average corn production from 2000-2019
- **Production Volatility**: High year-to-year variability (standard deviation ~1.2-1.4 million bushels per year)
- **Recent Decline**: Post-2020 shows slight decreasing trend, potentially related to weather patterns or agricultural practices
- **County Count Variation**: Number of reporting counties varies by year (71-83 counties)

#### Year-over-Year Changes
- Significant variability in year-over-year production changes
- Both positive and negative growth years observed
- Suggests high sensitivity to environmental conditions

### 3.2 Correlation Analysis

#### Top Positively Correlated Features (with Production)
1. **ESoil_tavg** (0.782) - Bare soil evaporation average temperature - Strongest predictor
2. **SoilMoi100_200cm_inst** (0.601) - Deep soil moisture (100-200cm depth)
3. **LWdown_f_tavg** (0.511) - Longwave downward radiation
4. **SoilTMP100_200cm_inst** (0.454) - Deep soil temperature
5. **Tair_f_inst** (0.448) - Air temperature
6. **Wind_f_inst** (0.440) - Wind speed
7. **Evap_tavg** (0.434) - Total evaporation

#### Top Negatively Correlated Features
1. **Albedo_inst** (-0.262) - Surface albedo (reflectance)
2. **SnowDepth_inst** (-0.246) - Snow depth
3. **Qs_acc** (-0.219) - Accumulated surface runoff
4. **SWE_inst** (-0.215) - Snow water equivalent

#### Key Insights from Correlations
- **Soil-related features dominate**: Soil moisture and temperature at deeper levels show strong positive correlations
- **Evaporation important**: Both soil evaporation and total evaporation highly correlated with yield
- **Winter conditions matter**: Snow-related features negatively correlated (less snow = better yields, likely due to warmer winters)
- **Albedo negative correlation**: Higher reflectance (possibly from barren fields or snow) associated with lower production

### 3.3 Distribution Analysis

#### Production Distribution
- **Mean**: 16.1 million bushels
- **Median**: ~12.5 million bushels (right-skewed)
- **Standard Deviation**: 12.3 million bushels (high variability)
- **Interquartile Range**: Q1 ~7.3M, Q3 ~25.1M bushels

#### Environmental Variable Distributions
- **Temperature Variables**: Relatively narrow distributions (IQR ~2-4K), consistent with Minnesota's temperate climate
- **Soil Moisture**: Varied by depth, showing seasonal and regional patterns
- **Precipitation/Evaporation**: Show seasonal cycles and year-to-year variability

---

## 4. Imputation Steps

### Data Preprocessing Pipeline

#### Zero Production Handling
- **Filtering Approach**: Removed 283 records with zero production
- **Rationale**: Zero production likely represents non-corn-growing years or counties, not missing data
- **Impact**: Reduced dataset from 2,001 to 1,718 observations

#### Missing Value Strategy
- **Initial Assessment**: Minimal missing values after aggregation (yearly level)
- **No Explicit Imputation Needed**: The temporal aggregation to yearly level likely filled gaps through averaging
- **Feature Engineering**: Some engineered features include safeguards (MIN_EPSILON = 1e-8) to prevent division by zero

#### Data Leakage Prevention
- **Temporal Split First**: Train/test split performed BEFORE any encoding or scaling
- **Split Point**: Year 2020 (years < 2020 = training, years >= 2020 = testing)
- **Encoders Fit on Training Only**: TargetEncoder and StandardScaler fit exclusively on training data, then applied to test data

---

## 5. Feature Engineering Step

### 5.1 Categorical Encoding
- **County Name Encoding**: Used TargetEncoder (fit on training data only)
  - Encodes county names using target variable statistics
  - Prevents data leakage by fitting only on training set
  - Creates `county_name_encoded` feature
  - Original categorical column dropped after encoding

### 5.2 Temporal Features
- **Year Trend**: Linear trend feature calculated as `(year - START_YEAR)` to capture long-term temporal patterns
- **Rationale**: Captures improvements in agricultural practices, technology, and gradual climate changes

### 5.3 Interaction Features
Created multiplicative interaction terms between related variables:

1. **evap_moist_interact**: Evaporation × Soil Moisture interaction
   - Captures how moisture availability affects evapotranspiration
   - Important for understanding water stress

2. **wind_moist_interact**: Wind Speed × Soil Moisture interaction
   - Represents wind-driven drying effects
   - Significant for predicting water loss

### 5.4 Ratio Features
- **soil_evap_ratio**: Soil Moisture / Evaporation ratio
   - Prevents division by zero using MIN_EPSILON (1e-8)
   - Represents water availability relative to water loss
   - Validated for infinite/NaN values

### 5.5 Dimensionality Reduction
- **PCA on Temperature Features**: Applied Principal Component Analysis to temperature-related features
- **Components**: 2 principal components (`PCA_COMPONENTS = 2`)
- **Purpose**: Reduce multicollinearity among highly correlated temperature measurements
- **Features Included**: Multiple soil temperature measurements at different depths

### 5.6 Target Transformation
- **Log Transformation**: Applied `log1p()` (log(1+x)) transformation to target variable
- **Rationale**: 
  - Production data is right-skewed
  - Log transformation normalizes distribution
  - Models trained on log scale, predictions converted back using `expm1()`

### 5.7 Feature Scaling
- **StandardScaler**: Applied to numeric features (after encoding)
- **Fit on Training Only**: Scaler fit on training data, then applied to both train and test
- **Columns Scaled**: All numeric features including engineered interactions and ratios
- **Purpose**: Normalize feature scales for model training

---

## 6. Model Selection Justification and Reason

### Models Selected

#### 1. **XGBoost (eXtreme Gradient Boosting)**
- **Justification**: 
  - State-of-the-art gradient boosting algorithm
  - Excellent for tabular data with mixed feature types
  - Handles non-linear relationships and feature interactions
  - Robust to outliers and missing values
  - Provides built-in feature importance
- **Why Selected**: Industry standard for regression tasks with complex feature interactions

#### 2. **LightGBM (Light Gradient Boosting Machine)**
- **Justification**:
  - Faster training than XGBoost
  - Better memory efficiency
  - Handles categorical features natively
  - Often performs comparably to XGBoost
- **Why Selected**: Provides alternative gradient boosting approach, good for comparison

#### 3. **Random Forest**
- **Justification**:
  - Ensemble of decision trees
  - Provides excellent interpretability through feature importance
  - Less prone to overfitting than single decision trees
  - Handles non-linear relationships
  - No feature scaling required
- **Why Selected**: Baseline ensemble method, good for understanding feature importance patterns

#### 4. **TabNet**
- **Justification**:
  - Deep learning architecture specifically designed for tabular data
  - Provides attention mechanism for feature selection
  - Can capture complex non-linear patterns
  - Maintains some interpretability
- **Why Selected**: Modern deep learning approach to compare against tree-based methods

#### 5. **Temporal CNN (Temporal Convolutional Network)**
- **Justification**:
  - Designed for sequential/temporal patterns
  - Can capture long-term dependencies
  - Alternative deep learning approach
- **Why Selected**: Explore temporal patterns in agricultural data
- **Note**: Encountered numerical instability (NaN predictions) in initial runs, likely due to:
  - Learning rate too high
  - Architecture complexity
  - Limited training data for deep learning model

### Model Selection Criteria
1. **Diversity**: Mix of ensemble (XGBoost, LightGBM, RF) and deep learning (TabNet, TCN) methods
2. **Proven Performance**: All models have strong track records in regression tasks
3. **Interpretability Balance**: Tree-based models offer feature importance, while deep learning captures complex patterns
4. **Computational Efficiency**: Range from fast (LightGBM) to more intensive (TCN)

---

## 7. Running of Model and Analysis of Results

### 7.1 Model Training Process

#### Hyperparameter Tuning
- **XGBoost**: GridSearchCV with 3-fold cross-validation
  - Parameters tuned: `learning_rate`, `max_depth`, `n_estimators`, `subsample`
  - Best parameters: `{'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 200, 'subsample': 0.8}`
  - Best CV R²: 0.7632

#### Training Approach
- **Temporal Split**: Train on 2000-2019, test on 2020-2022
- **Target Scale**: Models trained on log-transformed target
- **Evaluation**: Metrics calculated on both log and original scales

### 7.2 Model Performance Results

#### Overall Rankings (by R² Score)

1. **XGBoost** - R²: 0.9486, RMSE: 3,080,126 bushels
   - Best overall performance
   - Excellent generalization to test set
   - Strong feature interaction capture

2. **LightGBM** - R²: 0.9418, RMSE: 3,277,371 bushels
   - Nearly as good as XGBoost
   - Slightly faster training
   - Consistent performance

3. **TabNet** - R²: 0.9045, RMSE: 4,198,016 bushels
   - Strong deep learning performance
   - Captures complex non-linear patterns
   - May benefit from more tuning

4. **Random Forest** - R²: 0.7640, RMSE: 6,599,428 bushels
   - Good baseline performance
   - Excellent interpretability
   - Lower performance than gradient boosting methods

5. **Temporal CNN** - Performance Issues
   - Encountered NaN predictions (numerical instability)
   - Would require architecture/learning rate adjustments
   - Not included in final comparison

### 7.3 Feature Importance Analysis

#### Top Features Across Models

**XGBoost Top Features:**
1. `county_name_encoded` (29.6%) - Geographic location most important
2. `RootMoist_inst` (13.0%) - Root zone moisture critical
3. `evap_moist_interact` (8.6%) - Interaction term highly valued
4. `Tveg_tavg` (6.2%) - Vegetation temperature important
5. `year_trend` (3.2%) - Temporal trend captured

**LightGBM Top Features:**
1. `county_name_encoded` (627 importance score) - Dominant feature
2. `RootMoist_inst` (295) - Root zone moisture
3. `year_trend` (226) - Strong temporal component
4. `Psurf_f_inst` (209) - Surface pressure
5. `SWdown_f_tavg` (188) - Solar radiation

**Random Forest Top Features:**
1. `county_name_encoded` (21.6%) - Location critical
2. `evap_moist_interact` (11.3%) - Engineered interaction valuable
3. `ESoil_tavg` (8.1%) - Soil evaporation important
4. `wind_moist_interact` (6.8%) - Another interaction term

### 7.4 Key Performance Insights

#### Strengths
1. **Excellent Predictive Power**: All top models achieve R² > 0.90, explaining >90% of variance
2. **Gradient Boosting Dominance**: XGBoost and LightGBM outperform other approaches
3. **Feature Engineering Success**: Interaction terms rank highly across models
4. **Geographic Patterns**: County encoding consistently most important feature

#### Weaknesses
1. **RMSE Scale**: Even best model has RMSE of ~3 million bushels (though this is <20% of mean production)
2. **Deep Learning Challenges**: TCN failed, TabNet underperformed gradient boosting
3. **Temporal Limitations**: Recent years (2020-2022) show different patterns than training period

#### Error Analysis
- **RMSE Relative to Mean**: Best model RMSE is ~19% of mean production, which is acceptable for agricultural prediction
- **Scale Dependency**: Errors are in absolute terms; for large-production counties, percentage error may be lower

---

## 8. Conclusion

### Summary of Findings

This research successfully developed and evaluated multiple machine learning models for predicting county-level corn production in Minnesota using satellite-derived environmental data. Key achievements include:

1. **Model Performance**: Achieved excellent predictive accuracy with XGBoost (R² = 0.9486), demonstrating that machine learning can effectively predict agricultural yields from environmental variables.

2. **Feature Importance Discoveries**:
   - **Geographic location** (county) is the strongest predictor, indicating regional soil quality, climate zones, and agricultural practices matter most
   - **Soil moisture** (especially root zone) is critical for yield prediction
   - **Engineered interaction features** (evaporation × moisture, wind × moisture) provide valuable predictive power
   - **Temporal trends** capture improvements in farming practices over time

3. **Data Engineering Success**: 
   - Careful temporal splitting prevented data leakage
   - Feature engineering (interactions, ratios, temporal trends) significantly improved model performance
   - Log transformation of target variable improved model training

4. **Model Comparison**:
   - Gradient boosting methods (XGBoost, LightGBM) clearly outperform traditional ensemble (Random Forest) and deep learning (TabNet, TCN) approaches for this tabular dataset
   - Simpler tree-based models are more effective than complex deep learning architectures for this problem

### Practical Implications

1. **For Agricultural Decision-Making**: 
   - Models can inform planting decisions, resource allocation, and yield expectations
   - Soil moisture monitoring (especially root zone) should be prioritized
   - Regional factors (county-level) must be considered

2. **For Data Collection**:
   - Continue collecting GLDAS environmental variables
   - Monitor soil moisture at multiple depths
   - Track evaporation and evapotranspiration metrics

3. **For Model Deployment**:
   - Use XGBoost model for production predictions
   - Consider ensemble approach (averaging top models) for robustness
   - Monitor performance as new data arrives, retrain periodically

---

## 9. Future Improvements to the Research Topic

### 9.1 Data Enhancements

1. **Expanded Temporal Coverage**
   - Include more recent years (2023-2024) as they become available
   - Extend historical data backward if possible
   - Address the post-2020 production decline pattern

2. **Additional Environmental Variables**
   - Incorporate precipitation timing (early vs. late season)
   - Add extreme weather indicators (drought indices, heat stress days)
   - Include satellite-derived NDVI (vegetation health) data
   - Consider satellite soil moisture products (SMAP, SMOS)

3. **Spatial Features**
   - Add latitude/longitude coordinates explicitly
   - Include soil type classification
   - Incorporate elevation data
   - Add county-level agricultural statistics (irrigation rates, crop varieties)

4. **Management Variables**
   - Fertilizer application rates
   - Planting dates
   - Crop rotation patterns
   - Technology adoption (precision agriculture)

### 9.2 Feature Engineering Improvements

1. **Advanced Interaction Terms**
   - Multi-way interactions (temperature × moisture × precipitation)
   - Polynomial features for key variables
   - Seasonal indicators (growing degree days, chill hours)

2. **Temporal Feature Engineering**
   - Moving averages of key variables (3-month, 6-month windows)
   - Lag features (previous year's production, environmental conditions)
   - Seasonal decomposition (trend, seasonal, residual components)

3. **Spatial Feature Engineering**
   - Distance to major agricultural centers
   - County clustering based on similar environmental conditions
   - Regional trend features (state-wide averages, neighboring county features)

### 9.3 Model Improvements

1. **Ensemble Methods**
   - Stack multiple models (XGBoost + LightGBM + TabNet)
   - Use voting or weighted averaging
   - Implement meta-learning approaches

2. **Time Series Models**
   - ARIMA/SARIMA for temporal patterns
   - LSTM/GRU networks for sequential data
   - Prophet for trend and seasonality decomposition

3. **Spatial-Temporal Models**
   - Graph Neural Networks for county relationships
   - Spatial autoregressive models
   - Convolutional networks on spatial grids

4. **Uncertainty Quantification**
   - Implement prediction intervals (not just point estimates)
   - Quantile regression for distribution predictions
   - Bayesian approaches for uncertainty estimates

### 9.4 Model Robustness

1. **Cross-Validation Improvements**
   - Time-series cross-validation (proper temporal splits)
   - Spatial cross-validation (leave-one-county-out)
   - Combined spatial-temporal validation

2. **Generalization**
   - Test on out-of-state data (Iowa, Illinois) for generalization assessment
   - Validate on different crops (soybeans, wheat)
   - Test robustness to climate change scenarios

3. **Explainability**
   - Implement SHAP (SHapley Additive exPlanations) values
   - Generate partial dependence plots
   - Create feature interaction plots
   - Develop county-specific prediction explanations

### 9.5 Technical Improvements

1. **Deep Learning Fixes**
   - Resolve TCN numerical instability issues
   - Experiment with different architectures and learning rates
   - Implement batch normalization and dropout more effectively

2. **Hyperparameter Optimization**
   - Use Bayesian optimization (Optuna, Hyperopt)
   - Automated hyperparameter search with wider parameter spaces
   - Multi-objective optimization (accuracy + speed + interpretability)

3. **Model Monitoring**
   - Implement model drift detection
   - Automated retraining pipelines
   - Performance monitoring dashboards
   - A/B testing frameworks

### 9.6 Research Extensions

1. **Causal Inference**
   - Identify causal relationships (not just correlations)
   - Understand which interventions actually improve yield
   - Estimate treatment effects of management practices

2. **Climate Change Adaptation**
   - Predict yields under future climate scenarios
   - Assess vulnerability to extreme events
   - Recommend adaptation strategies

3. **Economic Integration**
   - Incorporate commodity prices
   - Optimize planting decisions based on predicted yields
   - Economic optimization models combining yield and price predictions

4. **Real-Time Prediction**
   - Develop in-season yield forecasting
   - Update predictions as growing season progresses
   - Early warning systems for yield shortfalls

5. **Multi-Scale Modeling**
   - Field-level predictions (higher resolution)
   - Regional/state-level aggregation
   - National-scale forecasting

### 9.7 Validation and Reproducibility

1. **External Validation**
   - Test on completely independent datasets
   - Validate in different geographic regions
   - Cross-validate with other research groups' data

2. **Reproducibility**
   - Document all preprocessing steps clearly
   - Version control for data and code
   - Publish detailed methodology
   - Share code and data (if possible)

3. **Uncertainty Communication**
   - Quantify prediction uncertainty
   - Communicate confidence intervals to stakeholders
   - Develop risk assessment frameworks

---

## Appendix: Key Statistics Summary

### Dataset Statistics
- **Training Set**: 1,109 observations (2000-2019)
- **Test Set**: 609 observations (2020-2022)
- **Total Features**: ~40 (including engineered features)
- **Counties**: Multiple Minnesota counties
- **Time Period**: 23 years (2000-2022)

### Model Performance Summary
| Model | R² Score | RMSE (bushels) | RMSE % of Mean |
|-------|----------|----------------|----------------|
| XGBoost | 0.9486 | 3,080,126 | ~19% |
| LightGBM | 0.9418 | 3,277,371 | ~20% |
| TabNet | 0.9045 | 4,198,016 | ~26% |
| Random Forest | 0.7640 | 6,599,428 | ~41% |

### Top Correlated Features
1. ESoil_tavg: 0.782
2. SoilMoi100_200cm_inst: 0.601
3. LWdown_f_tavg: 0.511
4. SoilTMP100_200cm_inst: 0.454
5. Tair_f_inst: 0.448

---

**Document Generated**: Based on analysis of SatelliteDataNew.ipynb  
**Date**: 2024  
**Research Focus**: County-level Corn Production Prediction using Satellite Environmental Data

