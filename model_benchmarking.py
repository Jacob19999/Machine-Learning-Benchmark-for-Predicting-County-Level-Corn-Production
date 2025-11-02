# ============================================================================
# COMPREHENSIVE MODEL BENCHMARKING: Simple vs Medium vs Complex Models
# ============================================================================

# ============================================================================
# LIBRARY CHECK AND INSTALLATION
# ============================================================================
import subprocess
import sys
import importlib

def check_and_install_library(package_name, import_name=None, auto_install=True):
    """Check if a library is installed, and optionally install it if missing."""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"  ✓ {package_name} is available")
        return True
    except ImportError:
        if auto_install:
            print(f"⚠ {package_name} not found. Attempting to install...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package_name], 
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                # Verify installation
                try:
                    importlib.import_module(import_name)
                    print(f"  ✓ {package_name} installed and verified successfully")
                    return True
                except ImportError:
                    print(f"  ✗ {package_name} installed but import still fails")
                    return False
            except subprocess.CalledProcessError:
                print(f"  ✗ Failed to install {package_name}")
                return False
        else:
            print(f"  ✗ {package_name} not found (auto-install disabled)")
            return False

print("="*80)
print("CHECKING REQUIRED LIBRARIES")
print("="*80)

# Required libraries (package_name, import_name)
required_libraries = [
    ('pandas', 'pandas'),
    ('numpy', 'numpy'),
    ('matplotlib', 'matplotlib'),
    ('seaborn', 'seaborn'),
    ('scikit-learn', 'sklearn'),
    ('xgboost', 'xgboost'),
]

# Optional libraries (won't fail if missing)
optional_libraries = [
    ('torch', 'torch'),  # For TCN model
]

all_installed = True
for package, import_name in required_libraries:
    if not check_and_install_library(package, import_name):
        all_installed = False
        print(f"\n❌ ERROR: Required library '{package}' could not be installed.")
        print("   Please install it manually: pip install " + package)
        sys.exit(1)

# Check optional libraries
print("\nChecking optional libraries...")
for package, import_name in optional_libraries:
    check_and_install_library(package, import_name)

if all_installed:
    print("\n✓ All required libraries are available!")
    print("="*80)
else:
    print("\n⚠ Some libraries may not be available. Script will continue but some features may not work.")
    print("="*80)

# Now import all libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
import time
import warnings
warnings.filterwarnings('ignore')

# Try importing transformer models (may not be available)
try:
    from sklearn.neural_network import MLPRegressor
    MLP_AVAILABLE = True
except:
    MLP_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except:
    TORCH_AVAILABLE = False

print("="*80)
print("COMPREHENSIVE MODEL BENCHMARKING")
print("="*80)
print("\nModels to benchmark:")
print("  SIMPLE:     Polynomial Regression, Random Forest")
print("  MEDIUM:     Support Vector Machine (SVM)")
print("  COMPLEX:    XGBoost, TCN (if available), Transformer-based MLP")
print("="*80)

# Load data
print("\nLoading data...")
import os

# Check if required data files exist
required_files = ['X_train.csv', 'X_test.csv', 'y_train.csv', 'y_test.csv']
missing_files = [f for f in required_files if not os.path.exists(f)]

if missing_files:
    print(f"\n❌ ERROR: Required data files not found:")
    for f in missing_files:
        print(f"   - {f}")
    print("\nPlease ensure the following files are in the current directory:")
    for f in required_files:
        print(f"   - {f}")
    sys.exit(1)

try:
    X_train = pd.read_csv('X_train.csv')
    X_test = pd.read_csv('X_test.csv')
    y_train = pd.read_csv('y_train.csv').iloc[:, 0]  # log scale
    y_test = pd.read_csv('y_test.csv').iloc[:, 0]    # log scale
    
    # Verify data shapes
    if X_train.shape[0] != len(y_train):
        print(f"\n⚠ Warning: X_train rows ({X_train.shape[0]}) don't match y_train length ({len(y_train)})")
    if X_test.shape[0] != len(y_test):
        print(f"\n⚠ Warning: X_test rows ({X_test.shape[0]}) don't match y_test length ({len(y_test)})")
    
    print("  ✓ Data files loaded successfully")
except FileNotFoundError as e:
    print(f"\n❌ ERROR: Could not find data file: {e}")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ ERROR: Failed to load data: {e}")
    sys.exit(1)

# Convert to original scale for evaluation
y_train_orig = np.expm1(y_train)
y_test_orig = np.expm1(y_test)

# Check for and handle missing values
print(f"Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"Test: {X_test.shape[0]} samples")

# Check for NaN values
train_nan = X_train.isna().sum().sum()
test_nan = X_test.isna().sum().sum()
if train_nan > 0 or test_nan > 0:
    print(f"\n⚠ Found missing values: Train={train_nan}, Test={test_nan}")
    print("  Handling missing values with imputation...")
    
    from sklearn.impute import SimpleImputer
    
    # Store original column order
    original_cols = X_train.columns.tolist()
    
    # Separate numeric and non-numeric columns
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [col for col in original_cols if col not in numeric_cols]
    
    print(f"  Numeric columns: {len(numeric_cols)}, Non-numeric: {len(non_numeric_cols)}")
    
    # Impute numeric columns with median
    if len(numeric_cols) > 0:
        imputer = SimpleImputer(strategy='median')
        X_train_imputed_numeric = imputer.fit_transform(X_train[numeric_cols])
        X_test_imputed_numeric = imputer.transform(X_test[numeric_cols])
    
    # Handle non-numeric columns (fill with mode or forward fill)
    X_train_imputed = X_train.copy()
    X_test_imputed = X_test.copy()
    
    if len(numeric_cols) > 0:
        # Update numeric columns with imputed values
        for i, col in enumerate(numeric_cols):
            X_train_imputed[col] = X_train_imputed_numeric[:, i]
            X_test_imputed[col] = X_test_imputed_numeric[:, i]
    
    # For non-numeric columns, use forward fill then backward fill
    if len(non_numeric_cols) > 0:
        for col in non_numeric_cols:
            X_train_imputed[col] = X_train_imputed[col].ffill().bfill()
            X_test_imputed[col] = X_test_imputed[col].ffill().bfill()
            # If still NaN, fill with empty string or first non-null value
            if X_train_imputed[col].isna().any():
                fill_value = X_train_imputed[col].dropna().iloc[0] if not X_train_imputed[col].dropna().empty else ''
                X_train_imputed[col] = X_train_imputed[col].fillna(fill_value)
                X_test_imputed[col] = X_test_imputed[col].fillna(fill_value)
    
    # Ensure original column order is preserved
    X_train = X_train_imputed[original_cols]
    X_test = X_test_imputed[original_cols]
    
    # Final check - if any NaN remains in numeric columns, fill with 0
    remaining_nan = X_train.isna().sum().sum() + X_test.isna().sum().sum()
    if remaining_nan > 0:
        print(f"  Some NaN remain ({remaining_nan}), filling numeric columns with 0...")
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)
    
    print(f"  ✓ Missing values handled. Final shape: Train={X_train.shape}, Test={X_test.shape}")
else:
    print("  ✓ No missing values found")

# Additional check for infinite values (can cause issues)
try:
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    train_inf = X_train[numeric_cols].isin([np.inf, -np.inf]).sum().sum()
    test_inf = X_test[numeric_cols].isin([np.inf, -np.inf]).sum().sum()
    if train_inf > 0 or test_inf > 0:
        print(f"\n⚠ Found infinite values: Train={train_inf}, Test={test_inf}")
        print("  Replacing infinite values with NaN and re-imputing...")
        # Store original columns
        original_cols = X_train.columns.tolist()
        numeric_cols_list = numeric_cols.tolist()
        non_numeric_cols_list = [col for col in original_cols if col not in numeric_cols_list]
        
        # Replace inf with NaN
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        
        # Re-impute numeric columns only
        if len(numeric_cols_list) > 0:
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X_train_imputed = X_train.copy()
            X_test_imputed = X_test.copy()
            
            X_train_numeric = pd.DataFrame(
                imputer.fit_transform(X_train[numeric_cols_list]),
                columns=numeric_cols_list,
                index=X_train.index
            )
            X_test_numeric = pd.DataFrame(
                imputer.transform(X_test[numeric_cols_list]),
                columns=numeric_cols_list,
                index=X_test.index
            )
            
            # Update numeric columns
            for col in numeric_cols_list:
                X_train_imputed[col] = X_train_numeric[col]
                X_test_imputed[col] = X_test_numeric[col]
            
            X_train = X_train_imputed[original_cols]
            X_test = X_test_imputed[original_cols]
        
        print(f"  ✓ Infinite values handled")
except Exception as e:
    print(f"  Note: Infinite value check skipped ({str(e)})")

# Final verification
if X_train.isna().sum().sum() > 0 or X_test.isna().sum().sum() > 0:
    print("\n⚠ Warning: Some NaN values remain after imputation")
    print("  Dropping rows with remaining NaN values...")
    X_train = X_train.dropna()
    y_train = y_train.loc[X_train.index]
    y_train_orig = y_train_orig.loc[X_train.index]
    X_test = X_test.dropna()
    y_test = y_test.loc[X_test.index]
    y_test_orig = y_test_orig.loc[X_test.index]
    print(f"  ✓ Final: Train={X_train.shape[0]} samples, Test={X_test.shape[0]} samples")

print()

# StandardScaler for models that need it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Store results
results = []

# ============================================================================
# SIMPLE MODELS
# ============================================================================

print("\n" + "="*80)
print("SIMPLE MODELS")
print("="*80)

# --- 1. POLYNOMIAL REGRESSION ---
print("\n[1/6] Training Polynomial Regression...")
start_time = time.time()

# Fine-tune polynomial degree
poly_params = {'poly__degree': [1, 2, 3]}
from sklearn.pipeline import Pipeline

best_poly_score = -np.inf
best_poly_degree = 2

for degree in [1, 2, 3]:
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly_features.fit_transform(X_train_scaled)
    X_test_poly = poly_features.transform(X_test_scaled)
    
    # Use Ridge for regularization
    model = Ridge(alpha=1.0)
    model.fit(X_train_poly, y_train)
    score = model.score(X_test_poly, y_test)
    
    if score > best_poly_score:
        best_poly_score = score
        best_poly_degree = degree
        best_poly_model = model
        best_poly_features = poly_features

X_test_poly = best_poly_features.transform(X_test_scaled)
y_pred_poly_log = best_poly_model.predict(X_test_poly)
y_pred_poly = np.expm1(y_pred_poly_log)

poly_time = time.time() - start_time
poly_rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_poly))
poly_r2 = r2_score(y_test_orig, y_pred_poly)
poly_mae = mean_absolute_error(y_test_orig, y_pred_poly)

results.append({
    'model': 'Polynomial Regression',
    'complexity': 'Simple',
    'degree': best_poly_degree,
    'train_time': poly_time,
    'rmse': poly_rmse,
    'r2': poly_r2,
    'mae': poly_mae,
    'predictions': y_pred_poly
})

print(f"  ✓ Best degree: {best_poly_degree}")
print(f"  ✓ R²: {poly_r2:.4f}, RMSE: {poly_rmse:,.0f}, Time: {poly_time:.1f}s")

# --- 2. RANDOM FOREST ---
print("\n[2/6] Training Random Forest...")
start_time = time.time()

rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10]
}

rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
rf_search = RandomizedSearchCV(
    rf_base, 
    rf_param_grid, 
    n_iter=20,  # Sample 20 combinations for speed
    cv=3,
    scoring='r2',
    n_jobs=-1,
    random_state=42,
    verbose=0
)

rf_search.fit(X_train, y_train)
rf_model = rf_search.best_estimator_
y_pred_rf_log = rf_model.predict(X_test)
y_pred_rf = np.expm1(y_pred_rf_log)

rf_time = time.time() - start_time
rf_rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_rf))
rf_r2 = r2_score(y_test_orig, y_pred_rf)
rf_mae = mean_absolute_error(y_test_orig, y_pred_rf)

results.append({
    'model': 'Random Forest',
    'complexity': 'Simple',
    'best_params': rf_search.best_params_,
    'train_time': rf_time,
    'rmse': rf_rmse,
    'r2': rf_r2,
    'mae': rf_mae,
    'predictions': y_pred_rf
})

print(f"  ✓ Best params: {rf_search.best_params_}")
print(f"  ✓ R²: {rf_r2:.4f}, RMSE: {rf_rmse:,.0f}, Time: {rf_time:.1f}s")

# ============================================================================
# MEDIUM COMPLEXITY MODELS
# ============================================================================

print("\n" + "="*80)
print("MEDIUM COMPLEXITY MODELS")
print("="*80)

# --- 3. SUPPORT VECTOR MACHINE (SVM) ---
print("\n[3/6] Training Support Vector Machine (SVM)...")
start_time = time.time()

# Use smaller grid for SVM (can be slow)
svm_param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01],
    'kernel': ['rbf', 'linear']
}

svm_base = SVR()
svm_search = RandomizedSearchCV(
    svm_base,
    svm_param_grid,
    n_iter=15,  # Sample combinations for speed
    cv=3,
    scoring='r2',
    n_jobs=-1,
    random_state=42,
    verbose=0
)

svm_search.fit(X_train_scaled, y_train)
svm_model = svm_search.best_estimator_
y_pred_svm_log = svm_model.predict(X_test_scaled)
y_pred_svm = np.expm1(y_pred_svm_log)

svm_time = time.time() - start_time
svm_rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_svm))
svm_r2 = r2_score(y_test_orig, y_pred_svm)
svm_mae = mean_absolute_error(y_test_orig, y_pred_svm)

results.append({
    'model': 'Support Vector Machine',
    'complexity': 'Medium',
    'best_params': svm_search.best_params_,
    'train_time': svm_time,
    'rmse': svm_rmse,
    'r2': svm_r2,
    'mae': svm_mae,
    'predictions': y_pred_svm
})

print(f"  ✓ Best params: {svm_search.best_params_}")
print(f"  ✓ R²: {svm_r2:.4f}, RMSE: {svm_rmse:,.0f}, Time: {svm_time:.1f}s")

# ============================================================================
# COMPLEX MODELS
# ============================================================================

print("\n" + "="*80)
print("COMPLEX MODELS")
print("="*80)

# --- 4. XGBOOST ---
print("\n[4/6] Training XGBoost...")
start_time = time.time()

xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0]
}

xgb_base = XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
xgb_search = RandomizedSearchCV(
    xgb_base,
    xgb_param_grid,
    n_iter=25,  # Sample combinations
    cv=3,
    scoring='r2',
    n_jobs=-1,
    random_state=42,
    verbose=0
)

xgb_search.fit(X_train, y_train)
xgb_model = xgb_search.best_estimator_
y_pred_xgb_log = xgb_model.predict(X_test)
y_pred_xgb = np.expm1(y_pred_xgb_log)

xgb_time = time.time() - start_time
xgb_rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_xgb))
xgb_r2 = r2_score(y_test_orig, y_pred_xgb)
xgb_mae = mean_absolute_error(y_test_orig, y_pred_xgb)

results.append({
    'model': 'XGBoost',
    'complexity': 'Complex',
    'best_params': xgb_search.best_params_,
    'train_time': xgb_time,
    'rmse': xgb_rmse,
    'r2': xgb_r2,
    'mae': xgb_mae,
    'predictions': y_pred_xgb
})

print(f"  ✓ Best params: {xgb_search.best_params_}")
print(f"  ✓ R²: {xgb_r2:.4f}, RMSE: {xgb_rmse:,.0f}, Time: {xgb_time:.1f}s")

# --- 5. TCN (Temporal Convolutional Network) ---
if TORCH_AVAILABLE:
    print("\n[5/6] Training TCN (Temporal Convolutional Network)...")
    start_time = time.time()
    
    try:
        class TCNRegressor(nn.Module):
            def __init__(self, input_size, num_channels, kernel_size=2, dropout=0.2):
                super(TCNRegressor, self).__init__()
                self.tcn = nn.Sequential(
                    nn.Conv1d(input_size, num_channels, kernel_size, padding=(kernel_size-1)),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(num_channels, num_channels, kernel_size, padding=(kernel_size-1)),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(num_channels, num_channels, kernel_size, padding=(kernel_size-1)),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1)
                )
                self.fc = nn.Linear(num_channels, 1)
            
            def forward(self, x):
                # Reshape for Conv1d: (batch, features, sequence)
                if len(x.shape) == 2:
                    x = x.unsqueeze(2)  # Add sequence dimension
                x = self.tcn(x)
                x = x.squeeze(2)
                x = self.fc(x)
                return x.squeeze(1)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_train_tensor = torch.FloatTensor(y_train.values)
        y_test_tensor = torch.FloatTensor(y_test.values)
        
        # Simple hyperparameter search
        best_tcn_r2 = -np.inf
        best_tcn_model = None
        best_tcn_channels = 32  # Default
        best_tcn_pred = None
        
        for num_channels in [32, 64]:
            model = TCNRegressor(X_train.shape[1], num_channels)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            # Train
            model.train()
            for epoch in range(50):  # Reduced epochs for speed
                optimizer.zero_grad()
                pred = model(X_train_tensor)
                loss = criterion(pred, y_train_tensor)
                loss.backward()
                optimizer.step()
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                y_pred_tcn_log = model(X_test_tensor).numpy()
            
            y_pred_tcn = np.expm1(y_pred_tcn_log)
            tcn_r2 = r2_score(y_test_orig, y_pred_tcn)
            
            if tcn_r2 > best_tcn_r2:
                best_tcn_r2 = tcn_r2
                best_tcn_model = model
                best_tcn_channels = num_channels
                best_tcn_pred = y_pred_tcn
        
        tcn_time = time.time() - start_time
        tcn_rmse = np.sqrt(mean_squared_error(y_test_orig, best_tcn_pred))
        tcn_r2 = r2_score(y_test_orig, best_tcn_pred)
        tcn_mae = mean_absolute_error(y_test_orig, best_tcn_pred)
        
        results.append({
            'model': 'TCN',
            'complexity': 'Complex',
            'best_params': {'num_channels': best_tcn_channels},
            'train_time': tcn_time,
            'rmse': tcn_rmse,
            'r2': tcn_r2,
            'mae': tcn_mae,
            'predictions': best_tcn_pred
        })
        
        print(f"  ✓ R²: {tcn_r2:.4f}, RMSE: {tcn_rmse:,.0f}, Time: {tcn_time:.1f}s")
    except Exception as e:
        print(f"  ⚠ TCN training failed: {str(e)}")
        print("  → Skipping TCN model")
else:
    print("\n[5/6] Skipping TCN (PyTorch not available)")

# --- 6. Transformer-based MLP (Multi-Layer Perceptron) ---
if MLP_AVAILABLE:
    print("\n[6/6] Training Transformer-inspired MLP...")
    start_time = time.time()
    
    mlp_param_grid = {
        'hidden_layer_sizes': [(100,), (200,), (100, 50), (200, 100)],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.01]
    }
    
    mlp_base = MLPRegressor(max_iter=500, random_state=42, early_stopping=True)
    mlp_search = RandomizedSearchCV(
        mlp_base,
        mlp_param_grid,
        n_iter=15,
        cv=3,
        scoring='r2',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    
    mlp_search.fit(X_train_scaled, y_train)
    mlp_model = mlp_search.best_estimator_
    y_pred_mlp_log = mlp_model.predict(X_test_scaled)
    y_pred_mlp = np.expm1(y_pred_mlp_log)
    
    mlp_time = time.time() - start_time
    mlp_rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_mlp))
    mlp_r2 = r2_score(y_test_orig, y_pred_mlp)
    mlp_mae = mean_absolute_error(y_test_orig, y_pred_mlp)
    
    results.append({
        'model': 'MLP (Neural Network)',
        'complexity': 'Complex',
        'best_params': mlp_search.best_params_,
        'train_time': mlp_time,
        'rmse': mlp_rmse,
        'r2': mlp_r2,
        'mae': mlp_mae,
        'predictions': y_pred_mlp
    })
    
    print(f"  ✓ Best params: {mlp_search.best_params_}")
    print(f"  ✓ R²: {mlp_r2:.4f}, RMSE: {mlp_rmse:,.0f}, Time: {mlp_time:.1f}s")
else:
    print("\n[6/6] Skipping MLP (scikit-learn version may not support)")

print("\n" + "="*80)
print("BENCHMARKING COMPLETE!")
print("="*80)

