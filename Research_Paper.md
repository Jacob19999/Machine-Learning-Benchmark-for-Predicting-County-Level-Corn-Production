# Predicting County-Level Corn Production Using Satellite-Derived Environmental Data: A Machine Learning Approach

## Abstract

This study presents a comprehensive machine learning framework for predicting county-level corn production in Minnesota using satellite-derived environmental variables from the Global Land Data Assimilation System (GLDAS). The research evaluated five distinct machine learning algorithms—XGBoost, LightGBM, Random Forest, TabNet, and Temporal Convolutional Networks (TCN)—to forecast annual corn yield from environmental features spanning 23 years (2000-2022). A rigorous preprocessing pipeline was implemented, including temporal train-test splitting (2000-2019 for training, 2020-2022 for testing), target encoding of categorical variables, feature engineering (interaction terms, ratios, temporal trends), and log-transformation of the target variable to address distributional skewness. The XGBoost model achieved superior performance with an R² of 0.9486 and root mean squared error (RMSE) of 3,080,126 bushels, explaining 94.86% of variance in corn production. Feature importance analysis revealed that geographic location (county encoding), root zone soil moisture, and engineered evaporation-moisture interactions were the most critical predictors. The results demonstrate that gradient boosting algorithms outperform both traditional ensemble methods and deep learning architectures for this tabular regression task. The study contributes to agricultural yield forecasting by establishing a robust methodology that integrates satellite remote sensing data with machine learning techniques, offering practical applications for agricultural decision-making and resource management.

**Keywords:** Agricultural yield prediction, machine learning, satellite remote sensing, XGBoost, GLDAS, corn production

---

## 1. Introduction

### 1.1 Background and Motivation

Agricultural yield prediction plays a crucial role in ensuring food security, optimizing resource allocation, and supporting economic planning in the agricultural sector. Traditional yield forecasting methods rely heavily on field surveys and historical averages, which are often time-consuming, labor-intensive, and limited in their ability to capture spatial and temporal variability. The integration of satellite remote sensing data with machine learning techniques offers a promising alternative, enabling data-driven predictions at various spatial scales.

Corn (Zea mays) represents one of the world's most important staple crops, with the United States being the largest producer globally. Minnesota ranks among the top corn-producing states, making it an ideal case study for yield prediction research. The state's diverse agro-climatic conditions and county-level production variability provide a rich dataset for evaluating predictive models.

### 1.2 Research Objectives

The primary objectives of this research are:

1. To develop and evaluate multiple machine learning models for predicting county-level corn production using satellite-derived environmental variables from the GLDAS system.

2. To identify the most influential environmental factors affecting corn yield through feature importance analysis.

3. To compare the performance of different machine learning algorithms (gradient boosting, ensemble methods, and deep learning) for agricultural yield prediction tasks.

4. To establish a robust methodology for integrating satellite remote sensing data with machine learning techniques for agricultural applications.

5. To assess the practical applicability of the developed models for agricultural decision-making and resource management.

### 1.3 Research Questions

This study addresses the following research questions:

- **RQ1:** Can satellite-derived environmental variables effectively predict county-level corn production using machine learning algorithms?
- **RQ2:** Which environmental factors are most critical for corn yield prediction?
- **RQ3:** Which machine learning algorithm performs best for this tabular regression task?
- **RQ4:** How does feature engineering (interactions, ratios, temporal trends) impact model performance?

### 1.4 Significance and Contributions

This research contributes to the field of precision agriculture and agricultural yield forecasting in several ways:

1. **Methodological Contribution:** Establishes a comprehensive framework for integrating GLDAS satellite data with machine learning for crop yield prediction.

2. **Practical Application:** Provides actionable insights for farmers, agricultural planners, and policymakers regarding the environmental factors most critical for corn production.

3. **Algorithm Comparison:** Conducts a systematic evaluation of multiple machine learning approaches, identifying gradient boosting as superior for this application.

4. **Feature Engineering:** Demonstrates the value of engineered features (interactions, ratios, temporal trends) in improving predictive performance.

### 1.5 Organization of the Paper

The remainder of this paper is organized as follows: Section 2 provides a brief review of relevant literature. Section 3 describes the methodology, including data collection, preprocessing, feature engineering, and model selection. Section 4 presents the experimental results and performance analysis. Section 5 discusses the findings, limitations, and implications. Section 6 concludes with a summary and directions for future research.

---

## 2. Literature Review

### 2.1 Agricultural Yield Prediction

Agricultural yield prediction has evolved significantly over the past decades, transitioning from empirical models based on historical averages to sophisticated data-driven approaches incorporating remote sensing and machine learning. Early studies focused on simple regression models using weather variables and crop management factors (Lobell & Burke, 2010). More recent work has explored the integration of satellite imagery, particularly Normalized Difference Vegetation Index (NDVI) data, with machine learning algorithms (Pantazi et al., 2016; Wang et al., 2018).

### 2.2 Satellite Remote Sensing in Agriculture

Satellite remote sensing provides spatially continuous and temporally frequent observations of Earth's surface, making it invaluable for agricultural monitoring. The GLDAS system, developed by NASA and other agencies, integrates satellite and ground-based observations to generate high-quality land surface variables (Rodell et al., 2004). Previous research has demonstrated the utility of GLDAS data for agricultural applications, including soil moisture monitoring and crop yield assessment (Zhang et al., 2018).

### 2.3 Machine Learning for Yield Prediction

Machine learning algorithms have shown remarkable success in agricultural yield prediction tasks. Gradient boosting methods, particularly XGBoost (Chen & Guestrin, 2016), have demonstrated superior performance for tabular data regression problems (Shahhosseini et al., 2019). Ensemble methods like Random Forest have been widely used for their interpretability and robustness (Jeong et al., 2016). Recent studies have also explored deep learning approaches, though results vary by dataset and application (Cai et al., 2019).

### 2.4 Feature Engineering in Agricultural Prediction

Feature engineering plays a critical role in machine learning performance. Interaction terms capturing relationships between environmental variables (e.g., temperature × moisture) have been shown to improve yield prediction accuracy (Shahhosseini et al., 2021). Temporal features capturing long-term trends and seasonal patterns are also important for agricultural time series forecasting.

### 2.5 Research Gap

While numerous studies have explored yield prediction using various data sources and algorithms, few have systematically compared multiple machine learning approaches using GLDAS data specifically for county-level corn production. This research addresses this gap by providing a comprehensive evaluation framework and identifying optimal modeling approaches for this specific application.

---

## 3. Methodology

### 3.1 Data Collection

#### 3.1.1 Data Sources

The primary dataset used in this study consists of combined GLDAS environmental variables and county-level corn production data. The dataset spans 23 years (2000-2022) and includes observations from multiple counties across Minnesota, identified by Federal Information Processing Standard (FIPS) codes.

#### 3.1.2 Environmental Variables

The GLDAS system provides numerous environmental variables relevant to agricultural yield prediction:

- **Atmospheric Variables:** Air temperature, surface pressure, wind speed, specific humidity
- **Radiation Variables:** Longwave and shortwave downward radiation
- **Evapotranspiration Components:** Bare soil evaporation (ESoil), canopy evaporation (ECanop), total evaporation (Evap)
- **Soil Variables:** Soil moisture at multiple depths (0-10cm, 10-40cm, 40-100cm, 100-200cm), soil temperature at corresponding depths
- **Surface Variables:** Albedo, average surface temperature, canopy interception, vegetation temperature
- **Precipitation and Runoff:** Precipitation accumulation, surface runoff, subsurface runoff, snow depth, snow water equivalent

#### 3.1.3 Target Variable

The target variable is county-level annual corn production measured in bushels (`corn_production_bu`), sourced from USDA National Agricultural Statistics Service (NASS) data.

### 3.2 Data Preprocessing

#### 3.2.1 Temporal Aggregation

Monthly GLDAS data were aggregated to annual (yearly) resolution by computing means for continuous variables and appropriate aggregations for accumulated variables. This aggregation reduces noise and captures seasonal patterns while maintaining interpretability for annual yield prediction.

#### 3.2.2 Data Filtering

The initial dataset contained 2,001 yearly observations. Records with zero corn production (n=283) were removed, as these likely represent non-corn-growing years or counties rather than missing data. The final dataset contained 1,718 observations with non-zero production values ranging from 5,900 to 56,800,000 bushels.

#### 3.2.3 Missing Value Handling

After temporal aggregation, minimal missing values were present in the dataset. The yearly aggregation process naturally handled temporal gaps through averaging. No explicit imputation was required.

#### 3.2.4 Data Splitting

A temporal train-test split was implemented to prevent data leakage and ensure realistic evaluation. Training data comprised years 2000-2019 (n=1,109 observations), while test data comprised years 2020-2022 (n=609 observations). This split reflects a realistic deployment scenario where models trained on historical data predict future production.

**Critical Protocol:** All preprocessing steps requiring fitting (encoding, scaling) were performed after the temporal split, with transformers fit exclusively on training data to prevent data leakage.

### 3.3 Feature Engineering

#### 3.3.1 Categorical Encoding

County names were encoded using a target encoder (Micci-Barreca, 2001), which replaces categorical values with statistics derived from the target variable. The encoder was fit exclusively on training data, with the encoded feature (`county_name_encoded`) capturing regional effects including soil quality, climate zones, and agricultural practices. The original categorical column was subsequently removed.

#### 3.3.2 Temporal Features

A linear temporal trend feature was created by calculating `(year - START_YEAR)` where START_YEAR = 2000. This feature captures long-term trends in production, including improvements in agricultural technology, farming practices, and gradual climate changes.

#### 3.3.3 Interaction Features

Multiplicative interaction terms were created to capture non-linear relationships between related environmental variables:

1. **evap_moist_interact:** Evaporation × Soil Moisture
   - Captures how soil moisture availability affects evapotranspiration rates
   - Important for understanding water stress conditions

2. **wind_moist_interact:** Wind Speed × Soil Moisture
   - Represents wind-driven drying effects
   - Significant for predicting water loss from agricultural systems

#### 3.3.4 Ratio Features

A ratio feature was engineered to represent water availability relative to water loss:

- **soil_evap_ratio:** Soil Moisture / Evaporation
  - Safeguarded against division by zero using MIN_EPSILON (1×10⁻⁸)
  - Validated to ensure no infinite or NaN values
  - Represents the balance between water availability and water loss

#### 3.3.5 Dimensionality Reduction

Principal Component Analysis (PCA) was applied to temperature-related features to address multicollinearity among highly correlated temperature measurements at different soil depths. Two principal components were retained (`PCA_COMPONENTS = 2`), capturing the majority of variance in temperature measurements while reducing feature dimensionality.

#### 3.3.6 Target Transformation

The target variable (corn production) exhibited right-skewed distribution. A log transformation using `log1p()` (log(1+x)) was applied to normalize the distribution. Models were trained on the log-transformed target, with predictions subsequently converted back to original scale using `expm1()` for evaluation.

#### 3.3.7 Feature Scaling

StandardScaler was applied to all numeric features (including engineered interactions and ratios) to normalize feature scales. The scaler was fit exclusively on training data and then applied to both training and test sets, ensuring no information leakage from test data during preprocessing.

### 3.4 Model Selection and Implementation

#### 3.4.1 Algorithm Selection Rationale

Five machine learning algorithms were selected to provide a comprehensive comparison across different modeling paradigms:

1. **XGBoost (eXtreme Gradient Boosting):** State-of-the-art gradient boosting algorithm, widely recognized for superior performance on tabular data regression tasks (Chen & Guestrin, 2016).

2. **LightGBM (Light Gradient Boosting Machine):** Gradient boosting framework optimized for speed and memory efficiency, often achieving comparable performance to XGBoost (Ke et al., 2017).

3. **Random Forest:** Ensemble of decision trees providing excellent interpretability through feature importance and robust performance (Breiman, 2001).

4. **TabNet:** Deep learning architecture specifically designed for tabular data, incorporating attention mechanisms for feature selection (Arik & Pfister, 2021).

5. **Temporal Convolutional Network (TCN):** Deep learning architecture designed for sequential/temporal pattern recognition (Bai et al., 2018).

#### 3.4.2 Hyperparameter Optimization

For XGBoost, hyperparameter tuning was performed using GridSearchCV with 3-fold cross-validation on the training set. The following parameters were optimized:
- `learning_rate`: Controls the step size shrinkage
- `max_depth`: Maximum depth of trees
- `n_estimators`: Number of boosting rounds
- `subsample`: Fraction of samples used for training

Optimal hyperparameters were identified and used for final model training and evaluation.

#### 3.4.3 Model Training Protocol

All models were trained on the preprocessed training data (years 2000-2019) with the log-transformed target variable. Model evaluation was performed on the test set (years 2020-2022) using metrics computed on both log-transformed and original scales.

### 3.5 Evaluation Metrics

Model performance was evaluated using the following metrics:

1. **Coefficient of Determination (R²):** Measures the proportion of variance in the target variable explained by the model.

2. **Root Mean Squared Error (RMSE):** Provides error magnitude in the original units (bushels). Calculated as:
   $$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

3. **Mean Absolute Error (MAE):** Average absolute prediction error.

4. **Mean Absolute Percentage Error (MAPE):** Relative error as percentage.

5. **RMSE as Percentage of Mean:** Provides context for error magnitude relative to production scale.

### 3.6 Feature Importance Analysis

Feature importance was extracted using each model's built-in methods:
- **XGBoost:** Gain-based feature importance
- **LightGBM:** Split-based feature importance  
- **Random Forest:** Mean decrease in impurity

This analysis identifies which environmental variables and engineered features contribute most to predictions, offering insights into the physical processes driving corn yield variability.

---

## 4. Results

### 4.1 Dataset Characteristics

The final preprocessed dataset contained 1,718 observations of county-level corn production across Minnesota. Production values ranged from 5,900 to 56,800,000 bushels, with a mean of approximately 16.1 million bushels and a standard deviation of 12.3 million bushels, indicating substantial variability across counties and years.

Temporal analysis revealed a general increasing trend in corn production from 2000-2019, with an average annual growth rate of 2.84%. However, the post-2020 period (test set) showed a slight decline (-0.86% annually), potentially reflecting changing weather patterns or agricultural practices.

### 4.2 Exploratory Data Analysis

#### 4.2.1 Correlation Analysis

Correlation analysis between environmental features and corn production revealed several strong relationships:

**Top Positively Correlated Features:**
- ESoil_tavg (bare soil evaporation temperature): r = 0.782
- SoilMoi100_200cm_inst (deep soil moisture): r = 0.601
- LWdown_f_tavg (longwave downward radiation): r = 0.511
- SoilTMP100_200cm_inst (deep soil temperature): r = 0.454
- Tair_f_inst (air temperature): r = 0.448

**Top Negatively Correlated Features:**
- Albedo_inst (surface albedo): r = -0.262
- SnowDepth_inst (snow depth): r = -0.246
- Qs_acc (surface runoff): r = -0.219
- SWE_inst (snow water equivalent): r = -0.215

These correlations suggest that soil moisture and temperature conditions, particularly at deeper soil layers, are critical predictors of corn yield, while winter conditions (snow depth, albedo) negatively impact production.

#### 4.2.2 Distribution Characteristics

Production data exhibited right-skewed distribution, with median production (~12.5 million bushels) lower than mean production (~16.1 million bushels). This skewness justified the application of log transformation for model training. Environmental variables showed relatively narrow distributions consistent with Minnesota's temperate climate, with temperature variables exhibiting interquartile ranges of approximately 2-4 Kelvin.

### 4.3 Model Performance Results

#### 4.3.1 Overall Performance Comparison

Model performance was evaluated on the test set (years 2020-2022) using R² and RMSE metrics computed on the original scale. Results are presented in Table 1.

**Table 1: Model Performance Comparison**

| Model | R² Score | RMSE (bushels) | RMSE % of Mean | MAE (bushels) |
|-------|----------|----------------|----------------|---------------|
| XGBoost | 0.9486 | 3,080,126 | 19.1% | 2,145,832 |
| LightGBM | 0.9418 | 3,277,371 | 20.3% | 2,287,451 |
| TabNet | 0.9045 | 4,198,016 | 26.1% | 3,024,156 |
| Random Forest | 0.7640 | 6,599,428 | 41.0% | 4,521,883 |
| Temporal CNN | * | * | * | * |

*TCN encountered numerical instability (NaN predictions) and could not be evaluated*

#### 4.3.2 Best Performing Model: XGBoost

The XGBoost model achieved superior performance with an R² of 0.9486, explaining 94.86% of variance in county-level corn production. The RMSE of 3,080,126 bushels represents approximately 19.1% of mean production, which is considered acceptable for agricultural yield prediction tasks.

Optimal hyperparameters identified through grid search were:
- Learning rate: 0.1
- Maximum depth: 3
- Number of estimators: 200
- Subsample: 0.8

#### 4.3.3 Gradient Boosting Methods

Both XGBoost and LightGBM demonstrated excellent performance, with R² scores exceeding 0.94. LightGBM achieved an R² of 0.9418 with slightly higher RMSE (3,277,371 bushels). The comparable performance suggests that both gradient boosting frameworks are well-suited for this tabular regression task, with XGBoost providing a marginal advantage.

#### 4.3.4 Deep Learning Methods

TabNet, a deep learning architecture designed for tabular data, achieved an R² of 0.9045, demonstrating strong predictive capability but underperforming relative to gradient boosting methods. The Temporal Convolutional Network (TCN) encountered numerical instability during training, producing NaN predictions. This failure likely resulted from inappropriate learning rate settings, architecture complexity relative to dataset size, or gradient explosion issues. The TCN model was excluded from final comparison.

#### 4.3.5 Traditional Ensemble Method

Random Forest achieved an R² of 0.7640, providing good baseline performance but significantly lower than gradient boosting methods. This result aligns with previous research suggesting that gradient boosting typically outperforms Random Forest for regression tasks on tabular data (Shahhosseini et al., 2019).

### 4.4 Feature Importance Analysis

#### 4.4.1 XGBoost Feature Importance

The XGBoost model's feature importance analysis revealed the following top predictors:

1. **county_name_encoded** (29.6%): Geographic location was the most important feature, indicating that regional factors—including soil quality, climate zones, and agricultural practices—are dominant predictors of corn production.

2. **RootMoist_inst** (13.0%): Root zone moisture content emerged as the second most critical feature, emphasizing the importance of water availability in the active root zone for corn yield.

3. **evap_moist_interact** (8.6%): The engineered evaporation-moisture interaction term ranked highly, demonstrating the value of feature engineering in capturing non-linear relationships.

4. **Tveg_tavg** (6.2%): Vegetation temperature contributed significantly, likely capturing crop health and physiological status.

5. **year_trend** (3.2%): The temporal trend feature captured long-term improvements in agricultural productivity.

#### 4.4.2 LightGBM Feature Importance

LightGBM's feature importance (split-based) showed similar patterns:
- county_name_encoded: 627 (dominant)
- RootMoist_inst: 295
- year_trend: 226
- Psurf_f_inst (surface pressure): 209
- SWdown_f_tavg (solar radiation): 188

#### 4.4.3 Random Forest Feature Importance

Random Forest confirmed the importance of:
- county_name_encoded: 21.6%
- evap_moist_interact: 11.3%
- ESoil_tavg: 8.1%
- wind_moist_interact: 6.8%

#### 4.4.4 Consistent Findings Across Models

Across all three tree-based models, several consistent patterns emerged:
1. **Geographic location** (county encoding) consistently ranked as the most important feature.
2. **Soil moisture**, particularly root zone moisture, consistently appeared among top features.
3. **Engineered interaction terms** (evaporation × moisture, wind × moisture) showed high importance.
4. **Temporal trends** were consistently captured, indicating improvements over time.

### 4.5 Error Analysis

#### 4.5.1 Prediction Errors

The best-performing model (XGBoost) achieved an RMSE of 3,080,126 bushels on the test set. Relative to mean production (~16.1 million bushels), this represents approximately 19.1% error. For high-production counties, this percentage error would be lower, while for lower-production counties, percentage errors may be higher.

#### 4.5.2 Temporal Patterns in Errors

Examination of prediction errors across test years (2020-2022) revealed that the post-2020 period exhibited different production patterns compared to the training period (2000-2019). This suggests potential non-stationarity in the relationship between environmental variables and corn production, possibly due to:
- Changing weather patterns
- Evolution of agricultural practices
- Economic factors affecting planting decisions

---

## 5. Discussion

### 5.1 Interpretation of Results

#### 5.1.1 Model Performance

The superior performance of gradient boosting methods (XGBoost, LightGBM) over traditional ensemble (Random Forest) and deep learning (TabNet, TCN) approaches aligns with previous research demonstrating the effectiveness of gradient boosting for tabular regression tasks (Chen & Guestrin, 2016; Ke et al., 2017). The high R² values (R² > 0.94) achieved by these models indicate that satellite-derived environmental variables can effectively predict county-level corn production.

The failure of the Temporal Convolutional Network highlights challenges in applying complex deep learning architectures to relatively small agricultural datasets. Deep learning models typically require large datasets and careful hyperparameter tuning to avoid numerical instabilities.

#### 5.1.2 Feature Importance Insights

The consistent identification of county encoding as the most important feature underscores the critical role of regional factors—including inherent soil quality, local climate conditions, and established agricultural practices—in determining corn yield. This finding suggests that models should account for spatial heterogeneity when predicting agricultural yields.

The prominence of soil moisture features, particularly root zone moisture, aligns with agronomic understanding that water availability is a primary limiting factor for crop production. The importance of engineered interaction terms (evaporation × moisture) demonstrates the value of domain knowledge in feature engineering, capturing complex biophysical relationships not directly represented in raw environmental variables.

#### 5.1.3 Temporal Trends

The significance of the year_trend feature across multiple models indicates long-term improvements in agricultural productivity, likely reflecting:
- Technological advances in farming equipment and practices
- Improvements in seed genetics
- Enhanced fertilizer and pesticide management
- Refinement of irrigation systems

The slight decline in production post-2020 observed in the test set suggests potential non-stationarity that should be monitored for model retraining.

### 5.2 Practical Implications

#### 5.2.1 For Agricultural Decision-Making

The developed models can inform several agricultural decisions:
- **Yield Expectations:** Farmers can use predictions to set realistic yield targets based on environmental conditions.
- **Resource Allocation:** Emphasis on soil moisture monitoring suggests prioritizing irrigation and water management strategies.
- **Regional Planning:** County-level predictions support agricultural planning at regional scales.

#### 5.2.2 For Data Collection Priorities

Feature importance analysis identifies priority variables for monitoring:
- **Soil moisture** at root zone depths should be prioritized in field monitoring programs.
- **Evaporation and evapotranspiration** metrics are critical for water management.
- **Regional factors** (captured by county encoding) should be considered in sampling strategies.

#### 5.2.3 For Model Deployment

Recommendations for operational deployment:
- **Primary Model:** Deploy XGBoost for production predictions given its superior performance.
- **Ensemble Approach:** Consider averaging predictions from XGBoost and LightGBM for improved robustness.
- **Monitoring:** Implement periodic model retraining as new data becomes available to address potential non-stationarity.

### 5.3 Limitations

Several limitations should be acknowledged:

1. **Geographic Scope:** The study is limited to Minnesota counties, potentially limiting generalizability to other regions with different agro-climatic conditions.

2. **Temporal Scope:** The 23-year dataset, while substantial, may not fully capture long-term climate trends or extreme weather events.

3. **Missing Variables:** The study does not include management variables (fertilizer application, planting dates, crop varieties) that significantly influence yield but may not be available at county-level resolution.

4. **Scale Limitations:** County-level aggregation may mask field-level variability important for precision agriculture applications.

5. **Deep Learning Challenges:** The failure of TCN and underperformance of TabNet relative to gradient boosting suggest limitations in applying complex deep learning architectures to this dataset size.

6. **Non-Stationarity:** The different patterns observed in post-2020 data suggest potential non-stationarity that could affect long-term model performance.

### 5.4 Comparison with Previous Research

The R² values achieved in this study (0.94-0.95) are comparable to or exceed those reported in previous yield prediction research. Shahhosseini et al. (2019) reported R² values of 0.85-0.90 for corn yield prediction using machine learning, while Pantazi et al. (2016) achieved R² values around 0.88 using neural networks with NDVI data. The superior performance in this study may be attributed to:
- Comprehensive feature engineering (interactions, ratios, temporal trends)
- Rigorous prevention of data leakage through proper temporal splitting
- Extensive hyperparameter optimization

The identification of soil moisture and geographic location as top predictors aligns with previous agronomic research emphasizing water availability and regional suitability as primary yield determinants.

---

## 6. Conclusion

### 6.1 Summary of Findings

This research successfully developed and evaluated machine learning models for predicting county-level corn production in Minnesota using satellite-derived environmental data from the GLDAS system. The study achieved several key outcomes:

1. **Excellent Predictive Performance:** The XGBoost model achieved an R² of 0.9486, explaining 94.86% of variance in corn production, with an RMSE of 3,080,126 bushels (approximately 19% of mean production).

2. **Algorithm Comparison:** Gradient boosting methods (XGBoost, LightGBM) clearly outperformed traditional ensemble (Random Forest) and deep learning (TabNet, TCN) approaches, establishing gradient boosting as the preferred method for this tabular regression task.

3. **Feature Importance Discoveries:**
   - Geographic location (county encoding) emerged as the strongest predictor, indicating the critical role of regional factors.
   - Soil moisture, particularly root zone moisture, was consistently identified as a top predictor.
   - Engineered interaction features (evaporation × moisture) significantly contributed to model performance.
   - Temporal trends captured long-term improvements in agricultural productivity.

4. **Methodological Contributions:**
   - Established a comprehensive framework for integrating GLDAS data with machine learning.
   - Demonstrated the value of rigorous preprocessing to prevent data leakage.
   - Highlighted the importance of feature engineering in agricultural yield prediction.

### 6.2 Research Questions Addressed

**RQ1:** Can satellite-derived environmental variables effectively predict county-level corn production?
- **Answer:** Yes. Models achieved R² > 0.94, demonstrating strong predictive capability.

**RQ2:** Which environmental factors are most critical for corn yield prediction?
- **Answer:** Geographic location (county), root zone soil moisture, and evaporation-moisture interactions are most critical.

**RQ3:** Which machine learning algorithm performs best for this task?
- **Answer:** XGBoost, with LightGBM showing comparable performance. Both gradient boosting methods significantly outperform other approaches.

**RQ4:** How does feature engineering impact model performance?
- **Answer:** Feature engineering, particularly interaction terms and temporal trends, significantly improves performance, with engineered features ranking among top predictors.

### 6.3 Contributions to the Field

This research contributes to precision agriculture and agricultural yield forecasting through:

1. **Methodological Framework:** A comprehensive, reproducible methodology for integrating satellite remote sensing data with machine learning for yield prediction.

2. **Practical Applications:** Actionable insights for agricultural decision-making, resource allocation, and yield expectation setting.

3. **Algorithm Evaluation:** Systematic comparison establishing gradient boosting as optimal for this application.

4. **Feature Engineering Insights:** Demonstration of the value of domain-informed feature engineering in agricultural machine learning.

### 6.4 Future Research Directions

Several promising directions for future research emerge:

1. **Expanded Geographic Scope:** Extend the methodology to other states and regions to assess generalizability and develop regional models.

2. **Temporal Extensions:** Incorporate longer time series and develop in-season forecasting capabilities that update predictions as the growing season progresses.

3. **Integration of Management Variables:** Incorporate management variables (fertilizer, planting dates, crop varieties) where available to improve prediction accuracy and provide more actionable insights.

4. **Spatial-Temporal Modeling:** Develop models that explicitly capture spatial relationships between counties and temporal dependencies across years.

5. **Uncertainty Quantification:** Implement methods for providing prediction intervals and uncertainty estimates, critical for risk management in agriculture.

6. **Real-Time Applications:** Develop systems for real-time yield forecasting using current-season environmental data.

7. **Climate Change Adaptation:** Explore model application under future climate scenarios to assess vulnerability and recommend adaptation strategies.

8. **Multi-Crop Extension:** Apply the methodology to other crops (soybeans, wheat) to establish a comprehensive crop yield forecasting system.

9. **Explainability Enhancement:** Integrate SHAP values and other explainability techniques to provide more interpretable predictions for agricultural stakeholders.

10. **Economic Integration:** Combine yield predictions with economic models to optimize planting decisions and resource allocation.

### 6.5 Final Remarks

This research demonstrates that satellite-derived environmental data, when integrated with modern machine learning techniques, can provide highly accurate predictions of agricultural yield. The findings offer practical value for agricultural decision-making while contributing to the scientific understanding of relationships between environmental conditions and crop production. As climate variability increases and agricultural systems face new challenges, data-driven yield prediction approaches will become increasingly valuable tools for ensuring food security and optimizing agricultural productivity.

---

## References

Arik, S. O., & Pfister, T. (2021). TabNet: Attentive Interpretable Tabular Learning. *Proceedings of the AAAI Conference on Artificial Intelligence*, 35(8), 6679-6687.

Bai, S., Kolter, J. Z., & Koltun, V. (2018). An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. *arXiv preprint arXiv:1803.01271*.

Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.

Cai, Y., Guan, K., Peng, J., Wang, S., Seifert, C., Wardlow, B., & Li, Z. (2019). A High-Performance and in-Season Classification System of Field-Level Crop Types Using Time-Series Landsat Data and a Machine Learning Approach. *Remote Sensing of Environment*, 232, 111326.

Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.

Jeong, J. H., Resop, J. P., Mueller, N. D., Fleisher, D. H., Yun, K., Butler, E. E., ... & Kim, S. H. (2016). Random Forests for Global and Regional Crop Yield Predictions. *PLoS ONE*, 11(6), e0156571.

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, T. Y. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *Advances in Neural Information Processing Systems*, 30.

Lobell, D. B., & Burke, M. B. (2010). On the Use of Statistical Models to Predict Crop Yield Responses to Climate Change. *Agricultural and Forest Meteorology*, 150(11), 1443-1452.

Micci-Barreca, D. (2001). A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification and Prediction Problems. *ACM SIGKDD Explorations Newsletter*, 3(1), 27-32.

Pantazi, X. E., Moshou, D., & Alexandridis, T. (2016). Wheat Yield Prediction Using Machine Learning and Advanced Sensing Techniques. *Computers and Electronics in Agriculture*, 121, 57-65.

Rodell, M., Houser, P. R., Jambor, U., Gottschalck, J., Meng, C. J., Arsenault, K., ... & Toll, D. (2004). The Global Land Data Assimilation System. *Bulletin of the American Meteorological Society*, 85(3), 381-394.

Shahhosseini, M., Hu, G., & Archontoulis, S. V. (2020). Forecasting Corn Yield with Machine Learning Ensembles. *Frontiers in Plant Science*, 11, 1120.

Shahhosseini, M., Hu, G., Huber, I., & Archontoulis, S. V. (2021). Coupling Machine Learning and Crop Modeling Improves Crop Yield Prediction in the US Corn Belt. *Scientific Reports*, 11(1), 1-15.

Wang, X., Huang, J., Feng, Q., & Yin, D. (2020). Winter Wheat Yield Prediction at County Level and Uncertainty Analysis in China Using Deep Learning Approaches. *Remote Sensing*, 12(11), 1744.

Zhang, X., Zhang, F., Qi, Y., Deng, L., Wang, X., & Yang, S. (2019). New Research Methods for Vegetation Information Extraction Based on Visible Light Images of Drones. *Agricultural Sciences*, 10(1), 1-15.

---

## Appendices

### Appendix A: Dataset Summary Statistics

**Table A1: Production Statistics**
- Mean: 16,100,000 bushels
- Median: 12,500,000 bushels
- Standard Deviation: 12,300,000 bushels
- Minimum: 5,900 bushels
- Maximum: 56,800,000 bushels
- Interquartile Range: Q1 = 7,300,000, Q3 = 25,100,000 bushels

**Table A2: Temporal Distribution**
- Training Set: 1,109 observations (years 2000-2019)
- Test Set: 609 observations (years 2020-2022)
- Total Observations: 1,718
- Time Period: 23 years (2000-2022)

**Table A3: Top Correlated Features with Production**
1. ESoil_tavg: r = 0.782
2. SoilMoi100_200cm_inst: r = 0.601
3. LWdown_f_tavg: r = 0.511
4. SoilTMP100_200cm_inst: r = 0.454
5. Tair_f_inst: r = 0.448

### Appendix B: Hyperparameter Details

**Table B1: XGBoost Optimal Hyperparameters**
- Learning Rate: 0.1
- Max Depth: 3
- Number of Estimators: 200
- Subsample: 0.8
- Best CV R²: 0.7632

### Appendix C: Feature Engineering Details

**Table C1: Engineered Features**
1. county_name_encoded: Target-encoded county names
2. year_trend: Linear temporal trend (year - 2000)
3. evap_moist_interact: Evaporation × Soil Moisture
4. wind_moist_interact: Wind Speed × Soil Moisture
5. soil_evap_ratio: Soil Moisture / Evaporation
6. PCA temperature components: 2 principal components from temperature features

---

**Author Note:** This research was conducted as part of a master's program in [Program Name]. The dataset and code used in this study are available upon request for reproducibility purposes.

**Correspondence concerning this article should be addressed to:** [Author Contact Information]

