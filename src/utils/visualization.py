"""Visualization utilities for model analysis."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
from typing import List, Optional, Any, Dict
import warnings
warnings.filterwarnings('ignore')


def safe_feature_names(feature_names: List[str]) -> List[str]:
    """
    Convert feature names to ASCII-safe format for SHAP plots.
    
    Args:
        feature_names: Original feature names
        
    Returns:
        ASCII-safe feature names
    """
    return [str(name).encode('ascii', 'ignore').decode('ascii').replace(' ', '_') 
            for name in feature_names]


def plot_shap_summary(shap_values: np.ndarray, X_test: pd.DataFrame,
                     filename: str = "shap_summary.png",
                     title: str = "SHAP Feature Importance") -> None:
    """
    Plot and save SHAP summary plot.
    
    Args:
        shap_values: SHAP values array
        X_test: Test features dataframe
        filename: Output filename
        title: Plot title
    """
    try:
        import shap
        
        # Convert feature names to ASCII-safe format
        X_test_safe = X_test.copy()
        X_test_safe.columns = safe_feature_names(X_test.columns)
        
        # Create SHAP summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test_safe, show=False)
        plt.title(title, fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"SHAP summary plot이 {filename}에 저장되었습니다.")
        
    except ImportError:
        print("SHAP가 설치되지 않았습니다. pip install shap을 실행하세요.")
    except Exception as e:
        print(f"SHAP summary plot 저장 중 오류: {e}")


def save_feature_importance(importance_series: pd.Series, 
                           filename: str = "feature_importance.png",
                           title: str = "Feature Importance",
                           top_n: int = 20) -> None:
    """
    Save feature importance plot.
    
    Args:
        importance_series: Series with feature importance values
        filename: Output filename
        title: Plot title
        top_n: Number of top features to show
    """
    try:
        # Get top N features
        top_features = importance_series.head(top_n)
        
        # Create plot
        plt.figure(figsize=(10, max(6, len(top_features) * 0.3)))
        bars = plt.barh(range(len(top_features)), top_features.values)
        plt.yticks(range(len(top_features)), top_features.index)
        plt.xlabel('Importance Score')
        plt.title(title, fontsize=14, pad=20)
        plt.gca().invert_yaxis()
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, top_features.values)):
            plt.text(value + max(top_features.values) * 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{value:.4f}', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"피처 중요도 플롯이 {filename}에 저장되었습니다.")
        
    except Exception as e:
        print(f"피처 중요도 플롯 저장 중 오류: {e}")


def plot_predictions_vs_actual(y_true: pd.Series, y_pred: np.ndarray,
                              filename: str = "predictions_vs_actual.png",
                              title: str = "Predictions vs Actual Values") -> None:
    """
    Plot predictions vs actual values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        filename: Output filename
        title: Plot title
    """
    try:
        plt.figure(figsize=(10, 8))
        
        # Scatter plot
        plt.scatter(y_true, y_pred, alpha=0.5, s=20)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(title, fontsize=14, pad=20)
        
        # Add R² score
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
                fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"예측값 vs 실제값 플롯이 {filename}에 저장되었습니다.")
        
    except Exception as e:
        print(f"예측값 vs 실제값 플롯 저장 중 오류: {e}")


def plot_residuals(y_true: pd.Series, y_pred: np.ndarray,
                  filename: str = "residuals.png",
                  title: str = "Residual Plot") -> None:
    """
    Plot residual analysis.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        filename: Output filename
        title: Plot title
    """
    try:
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.5, s=20)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.8)
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted')
        
        # Histogram of residuals
        axes[0, 1].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(x=0, color='r', linestyle='--', alpha=0.8)
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Residuals')
        
        # Q-Q plot (approximate)
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        
        # Absolute residuals vs predicted
        axes[1, 1].scatter(y_pred, np.abs(residuals), alpha=0.5, s=20)
        axes[1, 1].set_xlabel('Predicted Values')
        axes[1, 1].set_ylabel('Absolute Residuals')
        axes[1, 1].set_title('Absolute Residuals vs Predicted')
        
        plt.suptitle(title, fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"잔차 분석 플롯이 {filename}에 저장되었습니다.")
        
    except Exception as e:
        print(f"잔차 분석 플롯 저장 중 오류: {e}")


def plot_model_comparison(results: Dict[str, Dict[str, float]],
                         filename: str = "model_comparison.png",
                         metrics: List[str] = ['rmse', 'r2', 'mae']) -> None:
    """
    Plot model comparison across different metrics.
    
    Args:
        results: Dictionary mapping model names to their metrics
        filename: Output filename
        metrics: List of metrics to plot
    """
    try:
        comparison_df = pd.DataFrame(results).T
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 6))
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            if metric in comparison_df.columns:
                # Sort by metric (ascending for RMSE/MAE, descending for R²)
                ascending = metric.lower() in ['rmse', 'mae', 'mse']
                sorted_df = comparison_df.sort_values(metric, ascending=ascending)
                
                bars = axes[i].bar(range(len(sorted_df)), sorted_df[metric])
                axes[i].set_xticks(range(len(sorted_df)))
                axes[i].set_xticklabels(sorted_df.index, rotation=45, ha='right')
                axes[i].set_ylabel(metric.upper())
                axes[i].set_title(f'Model Comparison - {metric.upper()}')
                
                # Add value labels on bars
                for bar, value in zip(bars, sorted_df[metric]):
                    height = bar.get_height()
                    axes[i].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                               f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"모델 비교 플롯이 {filename}에 저장되었습니다.")
        
    except Exception as e:
        print(f"모델 비교 플롯 저장 중 오류: {e}")


def plot_time_series_predictions(dates: pd.Series, y_true: pd.Series, y_pred: np.ndarray,
                                filename: str = "time_series_predictions.png",
                                title: str = "Time Series Predictions",
                                sample_size: int = 1000) -> None:
    """
    Plot time series predictions vs actual values.
    
    Args:
        dates: Date series
        y_true: True values
        y_pred: Predicted values
        filename: Output filename
        title: Plot title
        sample_size: Number of points to sample for visualization
    """
    try:
        # Sample data if too large
        if len(dates) > sample_size:
            indices = np.random.choice(len(dates), sample_size, replace=False)
            indices = np.sort(indices)
            dates_sample = dates.iloc[indices]
            y_true_sample = y_true.iloc[indices]
            y_pred_sample = y_pred[indices]
        else:
            dates_sample = dates
            y_true_sample = y_true
            y_pred_sample = y_pred
        
        plt.figure(figsize=(15, 8))
        
        # Sort by date for proper line plot
        sort_idx = np.argsort(dates_sample)
        dates_sorted = dates_sample.iloc[sort_idx]
        y_true_sorted = y_true_sample.iloc[sort_idx]
        y_pred_sorted = y_pred_sample[sort_idx]
        
        plt.plot(dates_sorted, y_true_sorted, label='Actual', alpha=0.7, linewidth=1)
        plt.plot(dates_sorted, y_pred_sorted, label='Predicted', alpha=0.7, linewidth=1)
        
        plt.xlabel('Date')
        plt.ylabel('Call Count')
        plt.title(title, fontsize=14, pad=20)
        plt.legend()
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"시계열 예측 플롯이 {filename}에 저장되었습니다.")
        
    except Exception as e:
        print(f"시계열 예측 플롯 저장 중 오류: {e}")


def create_model_report_plots(y_true: pd.Series, y_pred: np.ndarray,
                             model_name: str, output_dir: str = "plots") -> None:
    """
    Create comprehensive model evaluation plots.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of the model
        output_dir: Directory to save plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean model name for filename
    clean_name = model_name.replace(" ", "_").replace("/", "_")
    
    # Create plots
    plot_predictions_vs_actual(
        y_true, y_pred, 
        filename=os.path.join(output_dir, f"{clean_name}_predictions_vs_actual.png"),
        title=f"{model_name} - Predictions vs Actual"
    )
    
    plot_residuals(
        y_true, y_pred,
        filename=os.path.join(output_dir, f"{clean_name}_residuals.png"),
        title=f"{model_name} - Residual Analysis"
    )
    
    print(f"{model_name} 모델의 평가 플롯들이 {output_dir} 디렉토리에 저장되었습니다.")
