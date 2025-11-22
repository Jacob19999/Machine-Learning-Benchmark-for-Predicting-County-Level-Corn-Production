"""
Hybrid Aggregation Module
Implements Adjust-Only Hybrid (default) and Stacked Hybrid
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from sklearn.linear_model import Ridge
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class HybridAggregation:
    """
    Hybrid aggregation combining ML predictions with LLM adjustments
    """
    
    def __init__(self, method: str = "adjust_only"):
        """
        Initialize hybrid aggregation
        
        Args:
            method: "adjust_only" (default) or "stacked"
        """
        if method not in ["adjust_only", "stacked"]:
            raise ValueError(f"Method must be 'adjust_only' or 'stacked', got {method}")
        self.method = method
        self.stacked_weights = None
        self.lambda_reg = 0.1  # Ridge regularization for stacked
    
    def clip_adjustment(self, a: float, bounds: Tuple[float, float] = (-0.20, 0.20)) -> float:
        """
        Clip adjustment to bounds
        
        Args:
            a: Adjustment factor
            bounds: (min, max) bounds
            
        Returns:
            Clipped adjustment
        """
        return np.clip(a, bounds[0], bounds[1])
    
    def adjust_only_hybrid(self, y_ml: float, adjustment: float, 
                          p10_ml: Optional[float] = None, 
                          p90_ml: Optional[float] = None) -> Dict[str, float]:
        """
        Adjust-Only Hybrid (default)
        
        y^_hyb = y^_ML * (1 + clip(a, -0.2, 0.2))
        P10_hyb = P10_ML * (1 + 0.7*a)
        P90_hyb = P90_ML * (1 + 0.7*a)
        
        Args:
            y_ml: ML baseline prediction
            adjustment: LLM adjustment factor a
            p10_ml: ML 10th percentile (optional)
            p90_ml: ML 90th percentile (optional)
            
        Returns:
            Dict with y_hyb, p10_hyb, p90_hyb
        """
        a_clipped = self.clip_adjustment(adjustment)
        
        y_hyb = y_ml * (1 + a_clipped)
        
        result = {'y_hyb': y_hyb}
        
        if p10_ml is not None:
            result['p10_hyb'] = p10_ml * (1 + 0.7 * a_clipped)
        
        if p90_ml is not None:
            result['p90_hyb'] = p90_ml * (1 + 0.7 * a_clipped)
        
        return result
    
    def _stacked_objective(self, w: np.ndarray, Z: np.ndarray, y: np.ndarray, 
                          lambda_reg: float) -> float:
        """
        Objective function for stacked hybrid:
        min ||y - Zw||^2 + lambda * ||w||^2
        subject to w >= 0, sum(w) = 1
        
        Args:
            w: Weight vector
            Z: Stacked predictions matrix (n_samples, n_models)
            y: True targets
            lambda_reg: Ridge regularization parameter
            
        Returns:
            Objective value
        """
        residual = y - Z @ w
        mse = np.mean(residual ** 2)
        ridge_penalty = lambda_reg * np.sum(w ** 2)
        return mse + ridge_penalty
    
    def _stacked_constraints(self, n_models: int) -> list:
        """
        Constraints for stacked hybrid:
        - w >= 0 (non-negative)
        - sum(w) = 1 (sum-to-one)
        
        Args:
            n_models: Number of models
            
        Returns:
            List of constraint dictionaries
        """
        constraints = [
            # Sum-to-one constraint
            {
                'type': 'eq',
                'fun': lambda w: np.sum(w) - 1.0
            }
        ]
        
        # Non-negativity constraints (handled via bounds in minimize)
        return constraints
    
    def fit_stacked(self, Z: np.ndarray, y: np.ndarray, lambda_reg: float = 0.1):
        """
        Fit stacked hybrid model using OOF predictions
        
        y^ = argmin_{w>=0, sum(w)=1} ||y - Zw||^2 + lambda * ||w||^2
        
        Args:
            Z: Stacked predictions matrix (n_samples, n_models)
                Columns: [y^_EN, y^_EBM, y^_XGB, y^_LLM]
            y: True targets (n_samples,)
            lambda_reg: Ridge regularization parameter
        """
        n_samples, n_models = Z.shape
        
        if n_samples != len(y):
            raise ValueError(f"Z shape {Z.shape} incompatible with y length {len(y)}")
        
        # Initial weights: uniform
        w0 = np.ones(n_models) / n_models
        
        # Bounds: non-negative
        bounds = [(0, None) for _ in range(n_models)]
        
        # Constraints: sum-to-one
        constraints = self._stacked_constraints(n_models)
        
        # Optimize
        result = minimize(
            fun=self._stacked_objective,
            x0=w0,
            args=(Z, y, lambda_reg),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            print(f"Warning: Stacked optimization did not converge: {result.message}")
            print(f"Using uniform weights as fallback")
            self.stacked_weights = w0
        else:
            self.stacked_weights = result.x
        
        self.lambda_reg = lambda_reg
        
        print(f"Stacked weights: {dict(zip(['EN', 'EBM', 'XGB', 'LLM'], self.stacked_weights))}")
    
    def predict_stacked(self, Z: np.ndarray) -> np.ndarray:
        """
        Predict using stacked hybrid
        
        Args:
            Z: Stacked predictions matrix (n_samples, n_models)
            
        Returns:
            Hybrid predictions (n_samples,)
        """
        if self.stacked_weights is None:
            raise ValueError("Must call fit_stacked() before predict_stacked()")
        
        return Z @ self.stacked_weights
    
    def predict(self, y_ml: float, adjustment: float, 
                p10_ml: Optional[float] = None, 
                p90_ml: Optional[float] = None,
                y_llm: Optional[float] = None,
                Z_stacked: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Predict using hybrid method
        
        Args:
            y_ml: ML baseline prediction
            adjustment: LLM adjustment factor
            p10_ml: ML 10th percentile (optional)
            p90_ml: ML 90th percentile (optional)
            y_llm: LLM direct prediction (for stacked, optional)
            Z_stacked: Stacked predictions array [EN, EBM, XGB, LLM] (for stacked, optional)
            
        Returns:
            Dict with predictions and uncertainty bands
        """
        if self.method == "adjust_only":
            return self.adjust_only_hybrid(y_ml, adjustment, p10_ml, p90_ml)
        
        elif self.method == "stacked":
            if Z_stacked is None:
                # Fallback to adjust_only if stacked inputs not available
                print("Warning: Stacked inputs not provided, falling back to adjust_only")
                return self.adjust_only_hybrid(y_ml, adjustment, p10_ml, p90_ml)
            
            if self.stacked_weights is None:
                raise ValueError("Must call fit_stacked() before using stacked method")
            
            y_stacked = self.predict_stacked(Z_stacked.reshape(1, -1))[0]
            
            result = {'y_hyb': y_stacked}
            
            # For uncertainty, use weighted combination of ML intervals
            if p10_ml is not None and p90_ml is not None:
                # Approximate: scale by adjustment factor similar to adjust_only
                a_clipped = self.clip_adjustment(adjustment)
                result['p10_hyb'] = p10_ml * (1 + 0.7 * a_clipped)
                result['p90_hyb'] = p90_ml * (1 + 0.7 * a_clipped)
            
            return result
        
        else:
            raise ValueError(f"Unknown method: {self.method}")


def compute_uncertainty_intervals(y_pred: np.ndarray, y_true: np.ndarray, 
                                 method: str = "quantile") -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute uncertainty intervals (P10, P90) from predictions
    
    Args:
        y_pred: Predictions
        y_true: True values
        method: "quantile" (residual-based) or "constant" (constant fraction)
        
    Returns:
        Tuple of (p10, p90) arrays
    """
    if method == "quantile":
        residuals = y_true - y_pred
        p10_residual = np.percentile(residuals, 10)
        p90_residual = np.percentile(residuals, 90)
        
        p10 = y_pred + p10_residual
        p90 = y_pred + p90_residual
    
    elif method == "constant":
        # Constant 20% uncertainty bands
        p10 = y_pred * 0.8
        p90 = y_pred * 1.2
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return p10, p90

