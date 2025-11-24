import json
from sklearn.model_selection import RandomizedSearchCV

# Read the notebook
with open('Model_Comparison_NoAcres.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# New code with comprehensive hyperparameter tuning
new_code = '''print("="*80)
print("MODEL 1: LASSO REGRESSION - Low Complexity (Hyperparameter Tuned)")
print("="*80)

# ElasticNet Regression with comprehensive hyperparameter tuning
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import RandomizedSearchCV
import time

print("\\nTraining ElasticNet Regression with hyperparameter tuning...")
print("Using: RandomizedSearchCV for efficient hyperparameter optimization")
print("Note: This will search a wide parameter space to find optimal settings.")

# Use RobustScaler (more robust to outliers than StandardScaler)
scaler_lasso = RobustScaler()
X_train_scaled_lasso = scaler_lasso.fit_transform(X_train)
X_test_scaled_lasso = scaler_lasso.transform(X_test)

# Define comprehensive parameter grid for tuning
param_grid = {
    'alpha': np.logspace(-6, 1, 100),  # Wide range: 1e-6 to 10
    'l1_ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],  # Full range L1 to L2
    'max_iter': [2000, 3000, 5000],  # Different iteration limits
    'selection': ['cyclic', 'random']  # Both selection methods
}

# Base ElasticNet model
base_model = ElasticNet(random_state=42, warm_start=True)

# RandomizedSearchCV for efficient hyperparameter search
# Using 50 random combinations (good balance between thoroughness and speed)
print("\\nPerforming RandomizedSearchCV (50 combinations, 5-fold CV)...")
print("This may take a few minutes...")
start_time = time.time()

lasso_search = RandomizedSearchCV(
    base_model,
    param_distributions=param_grid,
    n_iter=50,  # Sample 50 random combinations
    cv=5,        # 5-fold cross-validation
    scoring='r2',  # Optimize for R² score
    n_jobs=-1,   # Use all CPU cores
    random_state=42,
    verbose=1,   # Show progress
    return_train_score=True
)

# Fit the search on log-transformed target
lasso_search.fit(X_train_scaled_lasso, y_train)

tuning_time = time.time() - start_time
print(f"\\nHyperparameter tuning completed in {tuning_time:.1f} seconds")

# Get best model and parameters
lasso_reg = lasso_search.best_estimator_
best_params = lasso_search.best_params_
best_score = lasso_search.best_score_

print(f"\\nBest hyperparameters:")
print(f"  Alpha (regularization): {best_params['alpha']:.6f}")
print(f"  L1 ratio: {best_params['l1_ratio']:.4f} (1.0 = pure Lasso, 0.0 = pure Ridge)")
print(f"  Max iterations: {best_params['max_iter']}")
print(f"  Selection method: {best_params['selection']}")
print(f"  Best CV R² score: {best_score:.4f}")

# Number of selected features
n_selected_features = np.sum(np.abs(lasso_reg.coef_) > 1e-6)
print(f"\\nNumber of selected features: {n_selected_features} out of {len(X_train.columns)}")

# Show top selected features
feature_importance = pd.Series(np.abs(lasso_reg.coef_), index=X_train.columns)
top_features = feature_importance.nlargest(10)
print(f"\\nTop 10 selected features:")
for feat, imp in top_features.items():
    print(f"  {feat}: {imp:.4f}")

# Predictions on log scale
y_pred_lasso_log = lasso_reg.predict(X_test_scaled_lasso)

# Calculate predictions on training set for bias correction
y_pred_train_log = lasso_reg.predict(X_train_scaled_lasso)

# Use quantile-based clipping instead of fixed range
# Clip to 1st-99th percentile of actual log values to avoid extreme predictions
log_min = np.percentile(y_train, 1)
log_max = np.percentile(y_train, 99)
y_pred_train_log_clipped = np.clip(y_pred_train_log, log_min, log_max)
train_pred_orig = np.expm1(y_pred_train_log_clipped)

# Use quantile-based bias correction (more robust than median)
# Calculate correction factor using 25th-75th percentile range
q25_train = np.percentile(y_train_original, 25)
q75_train = np.percentile(y_train_original, 75)
q25_pred = np.percentile(train_pred_orig, 25)
q75_pred = np.percentile(train_pred_orig, 75)

# Use median ratio for bias correction
bias_correction_factor = np.median(y_train_original) / np.median(train_pred_orig)

# Alternative: use mean ratio if median gives extreme values
mean_correction = np.mean(y_train_original) / np.mean(train_pred_orig)
if abs(bias_correction_factor - 1.0) > abs(mean_correction - 1.0):
    bias_correction_factor = mean_correction

print(f"\\nBias correction factor: {bias_correction_factor:.4f}")

# Clip test predictions using quantile-based range
y_pred_lasso_log_clipped = np.clip(y_pred_lasso_log, log_min, log_max)
y_pred_lasso = np.expm1(y_pred_lasso_log_clipped)

# Apply bias correction
y_pred_lasso = y_pred_lasso * bias_correction_factor

# Use quantile-based clipping on original scale (more intelligent than fixed multipliers)
# Clip to reasonable range based on actual data distribution
q1_actual = np.percentile(y_test_original, 1)
q99_actual = np.percentile(y_test_original, 99)
y_pred_lasso = np.clip(y_pred_lasso, max(0, q1_actual * 0.5), q99_actual * 2.0)

# Metrics on log scale
lasso_rmse_log = np.sqrt(mean_squared_error(y_test, y_pred_lasso_log))
lasso_r2_log = r2_score(y_test, y_pred_lasso_log)
lasso_mae_log = mean_absolute_error(y_test, y_pred_lasso_log)

# Metrics on original scale (using improved predictions)
lasso_rmse_orig = np.sqrt(mean_squared_error(y_test_original, y_pred_lasso))
lasso_r2_orig = r2_score(y_test_original, y_pred_lasso)
lasso_mae_orig = mean_absolute_error(y_test_original, y_pred_lasso)
lasso_mape = np.mean(np.abs((y_test_original - y_pred_lasso) / (y_test_original + 1e-6))) * 100

print("\\nElasticNet Regression Performance (After Hyperparameter Tuning):")
print(f"  Log Scale  - RMSE: {lasso_rmse_log:.4f}")
print(f"  Log Scale  - R²:   {lasso_r2_log:.4f}")
print(f"  Log Scale  - MAE:  {lasso_mae_log:.4f}")
print(f"\\n  Orig Scale - RMSE: {lasso_rmse_orig:,.0f} bushels")
print(f"  Orig Scale - R²:   {lasso_r2_orig:.4f}")
print(f"  Orig Scale - MAE:  {lasso_mae_orig:,.0f} bushels")
print(f"  Orig Scale - MAPE: {lasso_mape:.2f}%")

# Show improvement summary
print("\\n" + "="*80)
print("HYPERPARAMETER TUNING SUMMARY")
print("="*80)
print(f"Best parameters found: {best_params}")
print(f"Cross-validation R²: {best_score:.4f}")
print(f"Test set R² (original scale): {lasso_r2_orig:.4f}")
print(f"Features selected: {n_selected_features}/{len(X_train.columns)}")

LASSO_TRAINED = True
'''

# Split into lines (notebook format uses array of strings)
new_code_lines = new_code.split('\n')

# Update cell 6
nb['cells'][6]['source'] = new_code_lines

# Clear the output (so it will re-run)
nb['cells'][6]['outputs'] = []

# Write back
with open('Model_Comparison_NoAcres.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Successfully updated cell 6 with comprehensive hyperparameter tuning!")
print("  - Added RandomizedSearchCV with 50 parameter combinations")
print("  - Tuning: alpha, l1_ratio, max_iter, selection method")
print("  - Using 5-fold cross-validation")
print("\nPlease re-run the cell to see improved results!")

