"""Evaluation utilities for model performance assessment."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics for regression.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary with evaluation metrics
    """
    return {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'n_samples': len(y_true),
        'mean_true': np.mean(y_true),
        'std_true': np.std(y_true),
        'mean_pred': np.mean(y_pred),
        'std_pred': np.std(y_pred)
    }


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                  model_name: str = "Model") -> Dict[str, float]:
    """
    Evaluate a trained model.
    
    Args:
        model: Trained model with predict method
        X_test: Test features
        y_test: Test target
        model_name: Name of the model for logging
        
    Returns:
        Dictionary with evaluation metrics
    """
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)
    
    print(f"[{model_name}] RMSE: {metrics['rmse']:.3f}, "
          f"R²: {metrics['r2']:.3f}, MAE: {metrics['mae']:.3f}")
    
    return metrics


def compare_models(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Compare multiple model results.
    
    Args:
        results: Dictionary mapping model names to their metrics
        
    Returns:
        DataFrame with comparison results
    """
    comparison_df = pd.DataFrame(results).T
    
    # Sort by RMSE (lower is better)
    comparison_df = comparison_df.sort_values('rmse')
    
    return comparison_df


def print_model_comparison(results: Dict[str, Dict[str, float]]) -> None:
    """
    Print formatted model comparison table.
    
    Args:
        results: Dictionary mapping model names to their metrics
    """
    comparison_df = compare_models(results)
    
    print("\n" + "="*80)
    print("모델 성능 비교 (RMSE 기준 정렬)")
    print("="*80)
    print(f"{'모델명':<20} {'RMSE':<10} {'R²':<10} {'MAE':<10} {'샘플수':<10}")
    print("-"*80)
    
    for model_name, row in comparison_df.iterrows():
        print(f"{model_name:<20} {row['rmse']:<10.3f} {row['r2']:<10.3f} "
              f"{row['mae']:<10.3f} {int(row['n_samples']):<10}")
    
    print("="*80)


def calculate_residual_stats(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate residual statistics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary with residual statistics
    """
    residuals = y_true - y_pred
    
    return {
        'residual_mean': np.mean(residuals),
        'residual_std': np.std(residuals),
        'residual_min': np.min(residuals),
        'residual_max': np.max(residuals),
        'residual_q25': np.percentile(residuals, 25),
        'residual_q50': np.percentile(residuals, 50),
        'residual_q75': np.percentile(residuals, 75),
        'abs_residual_mean': np.mean(np.abs(residuals)),
        'abs_residual_std': np.std(np.abs(residuals))
    }


def evaluate_by_groups(y_true: pd.Series, y_pred: np.ndarray, 
                      groups: pd.Series, group_name: str = "Group") -> pd.DataFrame:
    """
    Evaluate model performance by groups.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        groups: Group labels
        group_name: Name of the grouping variable
        
    Returns:
        DataFrame with group-wise evaluation results
    """
    results = []
    
    for group in groups.unique():
        if pd.notna(group):
            mask = groups == group
            group_true = y_true[mask]
            group_pred = y_pred[mask]
            
            if len(group_true) > 0:
                metrics = calculate_metrics(group_true, group_pred)
                metrics[group_name] = group
                results.append(metrics)
    
    return pd.DataFrame(results)


def calculate_percentage_errors(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate percentage-based error metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary with percentage error metrics
    """
    # Avoid division by zero
    mask = y_true != 0
    
    if not mask.any():
        return {
            'mape': np.nan,
            'wmape': np.nan,
            'smape': np.nan
        }
    
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100
    
    # Weighted Mean Absolute Percentage Error
    wmape = np.sum(np.abs(y_true_filtered - y_pred_filtered)) / np.sum(y_true_filtered) * 100
    
    # Symmetric Mean Absolute Percentage Error
    smape = np.mean(2 * np.abs(y_true_filtered - y_pred_filtered) / 
                   (np.abs(y_true_filtered) + np.abs(y_pred_filtered))) * 100
    
    return {
        'mape': mape,
        'wmape': wmape,
        'smape': smape
    }


def create_performance_summary(y_true: pd.Series, y_pred: np.ndarray,
                             model_name: str = "Model") -> Dict[str, Any]:
    """
    Create comprehensive performance summary.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of the model
        
    Returns:
        Dictionary with comprehensive performance metrics
    """
    basic_metrics = calculate_metrics(y_true, y_pred)
    residual_stats = calculate_residual_stats(y_true, y_pred)
    percentage_errors = calculate_percentage_errors(y_true, y_pred)
    
    summary = {
        'model_name': model_name,
        **basic_metrics,
        **residual_stats,
        **percentage_errors
    }
    
    return summary


def save_evaluation_results(results: Dict[str, Dict[str, float]], 
                           filename: str = "evaluation_results.csv") -> None:
    """
    Save evaluation results to CSV file.
    
    Args:
        results: Dictionary mapping model names to their metrics
        filename: Output filename
    """
    comparison_df = compare_models(results)
    comparison_df.to_csv(filename, encoding='utf-8')
    print(f"평가 결과를 {filename}에 저장했습니다.")


def check_prediction_bounds(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Check if predictions are within reasonable bounds.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary with bound check results
    """
    true_min, true_max = y_true.min(), y_true.max()
    pred_min, pred_max = y_pred.min(), y_pred.max()
    
    # Check for negative predictions (if true values are all positive)
    negative_preds = (y_pred < 0).sum() if y_true.min() >= 0 else 0
    
    # Check for extreme predictions
    extreme_high = (y_pred > true_max * 2).sum()
    extreme_low = (y_pred < true_min * 0.5).sum() if true_min > 0 else 0
    
    return {
        'true_range': (true_min, true_max),
        'pred_range': (pred_min, pred_max),
        'negative_predictions': negative_preds,
        'extreme_high_predictions': extreme_high,
        'extreme_low_predictions': extreme_low,
        'total_predictions': len(y_pred)
    }
