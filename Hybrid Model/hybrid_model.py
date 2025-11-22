"""
Main Hybrid Model Script
Orchestrates factsheet building, LLM track, and hybrid aggregation
"""

import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from factsheet_builder import FactsheetBuilder
from llm_track import LLMTrack
from hybrid_aggregation import HybridAggregation, compute_uncertainty_intervals


class HybridModel:
    """
    Complete hybrid model pipeline
    """
    
    def __init__(self, data_path: str, api_key: str, 
                 ml_model_path: Optional[str] = None,
                 method: str = "adjust_only"):
        """
        Initialize hybrid model
        
        Args:
            data_path: Path to consolidated data CSV
            api_key: Grok API key
            ml_model_path: Path to trained ML model pickle (optional)
            method: "adjust_only" (default) or "stacked"
        """
        print("="*80)
        print("INITIALIZING HYBRID MODEL")
        print("="*80)
        
        # Load data
        print(f"\n[1/4] Loading data from {data_path}...")
        self.data = pd.read_csv(data_path)
        print(f"  Loaded {len(self.data)} rows, {len(self.data.columns)} columns")
        
        # Initialize components
        print("\n[2/4] Initializing components...")
        self.factsheet_builder = FactsheetBuilder(self.data)
        self.llm_track = LLMTrack(api_key=api_key)
        self.hybrid_agg = HybridAggregation(method=method)
        
        # Load ML model if provided
        self.ml_model = None
        if ml_model_path and os.path.exists(ml_model_path):
            print(f"\n[3/4] Loading ML model from {ml_model_path}...")
            with open(ml_model_path, 'rb') as f:
                self.ml_model = pickle.load(f)
            print("  ML model loaded")
        else:
            print("\n[3/4] No ML model provided - will need ML predictions as input")
        
        print("\n[4/4] Hybrid model initialized")
        print("="*80)
    
    def get_ml_prediction(self, X: pd.DataFrame, use_log_transform: bool = True) -> np.ndarray:
        """
        Get ML baseline predictions
        
        Args:
            X: Feature matrix
            use_log_transform: Whether to apply log transform (default: True)
            
        Returns:
            Predictions (in original scale)
        """
        if self.ml_model is None:
            raise ValueError("ML model not loaded. Provide ml_model_path or predictions directly.")
        
        y_pred_log = self.ml_model.predict(X)
        
        if use_log_transform:
            y_pred = np.expm1(y_pred_log)
        else:
            y_pred = y_pred_log
        
        return y_pred
    
    def predict_single(self, fips: int, year: int, 
                      y_ml: Optional[float] = None,
                      X_features: Optional[pd.DataFrame] = None,
                      p10_ml: Optional[float] = None,
                      p90_ml: Optional[float] = None,
                      cdl_fraction: Optional[float] = None,
                      use_fallback: bool = True) -> Dict[str, any]:
        """
        Predict for single (county, year) using hybrid model
        
        Args:
            fips: County FIPS code
            year: Target year
            y_ml: ML baseline prediction (if None, will compute from X_features)
            X_features: Feature matrix for ML prediction (if y_ml not provided)
            p10_ml: ML 10th percentile (optional)
            p90_ml: ML 90th percentile (optional)
            cdl_fraction: CDL fraction (optional)
            use_fallback: Use ML baseline if LLM fails (default: True)
            
        Returns:
            Dict with predictions, adjustment, drivers, and metadata
        """
        # Get ML baseline
        if y_ml is None:
            if X_features is None:
                raise ValueError("Must provide either y_ml or X_features")
            y_ml = self.get_ml_prediction(X_features)[0]
        
        # Build factsheet
        factsheet = self.factsheet_builder.build_factsheet(
            fips=fips,
            year=year,
            ml_baseline=y_ml,
            cdl_fraction=cdl_fraction
        )
        
        # Format for LLM
        factsheet_text = self.factsheet_builder.format_factsheet_for_llm(factsheet)
        
        # Get LLM adjustment
        if use_fallback:
            adjustment, drivers = self.llm_track.get_adjustment_with_fallback(
                factsheet_text,
                fallback_adjustment=0.0
            )
        else:
            adjustment, drivers, success = self.llm_track.get_adjustment(factsheet_text)
            if not success:
                raise ValueError("LLM adjustment failed and use_fallback=False")
        
        # Hybrid aggregation
        hybrid_result = self.hybrid_agg.predict(
            y_ml=y_ml,
            adjustment=adjustment,
            p10_ml=p10_ml,
            p90_ml=p90_ml
        )
        
        return {
            'fips': fips,
            'year': year,
            'county_name': factsheet.get('county_name', 'Unknown'),
            'y_ml': y_ml,
            'adjustment': adjustment,
            'drivers': drivers,
            'y_hyb': hybrid_result['y_hyb'],
            'p10_hyb': hybrid_result.get('p10_hyb'),
            'p90_hyb': hybrid_result.get('p90_hyb'),
            'factsheet': factsheet
        }
    
    def predict_batch(self, test_data: pd.DataFrame,
                     y_ml_predictions: Optional[np.ndarray] = None,
                     X_test: Optional[pd.DataFrame] = None,
                     p10_ml: Optional[np.ndarray] = None,
                     p90_ml: Optional[np.ndarray] = None,
                     use_fallback: bool = True,
                     max_samples: Optional[int] = None) -> pd.DataFrame:
        """
        Predict for batch of (county, year) pairs
        
        Args:
            test_data: DataFrame with fips, year columns
            y_ml_predictions: ML predictions array (if None, will compute from X_test)
            X_test: Feature matrix for ML predictions (if y_ml_predictions not provided)
            p10_ml: ML 10th percentile array (optional)
            p90_ml: ML 90th percentile array (optional)
            use_fallback: Use ML baseline if LLM fails (default: True)
            max_samples: Maximum number of samples to process (for testing, default: None = all)
            
        Returns:
            DataFrame with predictions and metadata
        """
        print("\n" + "="*80)
        print("BATCH PREDICTION")
        print("="*80)
        
        # Get ML predictions if not provided
        if y_ml_predictions is None:
            if X_test is None:
                raise ValueError("Must provide either y_ml_predictions or X_test")
            print("\nComputing ML baseline predictions...")
            y_ml_predictions = self.get_ml_prediction(X_test)
            print(f"  Computed {len(y_ml_predictions)} ML predictions")
        
        # Limit samples if specified
        n_samples = len(test_data)
        if max_samples is not None:
            n_samples = min(n_samples, max_samples)
            test_data = test_data.head(n_samples)
            y_ml_predictions = y_ml_predictions[:n_samples]
            if p10_ml is not None:
                p10_ml = p10_ml[:n_samples]
            if p90_ml is not None:
                p90_ml = p90_ml[:n_samples]
        
        print(f"\nProcessing {n_samples} samples...")
        
        results = []
        for idx, row in test_data.iterrows():
            fips = int(row['fips'])
            year = int(row['year'])
            y_ml = float(y_ml_predictions[idx])
            p10 = float(p10_ml[idx]) if p10_ml is not None else None
            p90 = float(p90_ml[idx]) if p90_ml is not None else None
            
            try:
                result = self.predict_single(
                    fips=fips,
                    year=year,
                    y_ml=y_ml,
                    p10_ml=p10,
                    p90_ml=p90,
                    use_fallback=use_fallback
                )
                results.append(result)
                
                if (idx + 1) % 10 == 0:
                    print(f"  Processed {idx + 1}/{n_samples} samples...")
            
            except Exception as e:
                print(f"  Error processing FIPS {fips}, Year {year}: {e}")
                if use_fallback:
                    # Fallback to ML baseline
                    results.append({
                        'fips': fips,
                        'year': year,
                        'county_name': row.get('county_name', 'Unknown'),
                        'y_ml': y_ml,
                        'adjustment': 0.0,
                        'drivers': None,
                        'y_hyb': y_ml,
                        'p10_hyb': p10,
                        'p90_hyb': p90,
                        'factsheet': None
                    })
                else:
                    raise
        
        print(f"\nCompleted {len(results)} predictions")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        return results_df
    
    def evaluate(self, predictions_df: pd.DataFrame, y_true: pd.Series) -> Dict[str, float]:
        """
        Evaluate hybrid model predictions
        
        Args:
            predictions_df: DataFrame with y_hyb column
            y_true: True target values
            
        Returns:
            Dict with evaluation metrics
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        y_pred = predictions_df['y_hyb'].values
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Hybrid Model for Corn Yield Prediction')
    parser.add_argument('--data', type=str, default='../consolidated_data_phase3.csv',
                       help='Path to consolidated data CSV')
    parser.add_argument('--api-key', type=str, default=None,
                       help='Grok API key (or read from API Key.txt)')
    parser.add_argument('--ml-model', type=str, default=None,
                       help='Path to trained ML model pickle')
    parser.add_argument('--method', type=str, default='adjust_only',
                       choices=['adjust_only', 'stacked'],
                       help='Hybrid method')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum samples to process (for testing)')
    
    args = parser.parse_args()
    
    # Read API key
    if args.api_key is None:
        api_key_path = Path(__file__).parent / "API Key.txt"
        if api_key_path.exists():
            with open(api_key_path, 'r') as f:
                api_key = f.read().strip()
        else:
            raise ValueError("API key not provided and API Key.txt not found")
    else:
        api_key = args.api_key
    
    # Initialize model
    model = HybridModel(
        data_path=args.data,
        api_key=api_key,
        ml_model_path=args.ml_model,
        method=args.method
    )
    
    print("\nHybrid model ready for predictions")
    print("Use model.predict_single() or model.predict_batch() methods")


if __name__ == "__main__":
    main()

