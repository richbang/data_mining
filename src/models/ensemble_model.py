"""Ensemble model for combining multiple predictions."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


class EnsembleModel:
    """
    Ensemble model for combining predictions from multiple models.
    Supports simple averaging and weighted averaging.
    """
    
    def __init__(self):
        """Initialize EnsembleModel."""
        self.predictions: Dict[str, np.ndarray] = {}
        self.weights: Optional[Dict[str, float]] = None
        self.ensemble_prediction: Optional[np.ndarray] = None

    def add_predictions(self, model_name: str, predictions: np.ndarray) -> None:
        """
        Add predictions from a model.
        
        Args:
            model_name: Name of the model
            predictions: Prediction array
        """
        self.predictions[model_name] = predictions

    def set_weights(self, weights: Dict[str, float]) -> None:
        """
        Set weights for weighted ensemble.
        
        Args:
            weights: Dictionary mapping model names to weights
        """
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        self.weights = {k: v/total_weight for k, v in weights.items()}

    def predict_simple_average(self) -> np.ndarray:
        """
        Create ensemble prediction using simple averaging.
        
        Returns:
            Ensemble predictions
        """
        if not self.predictions:
            raise ValueError("예측값이 없습니다. add_predictions()를 먼저 호출하세요.")
        
        # Stack all predictions
        pred_matrix = np.column_stack(list(self.predictions.values()))
        
        # Simple average
        self.ensemble_prediction = np.mean(pred_matrix, axis=1)
        return self.ensemble_prediction

    def predict_weighted_average(self, weights: Dict[str, float] = None) -> np.ndarray:
        """
        Create ensemble prediction using weighted averaging.
        
        Args:
            weights: Optional weights for each model
            
        Returns:
            Ensemble predictions
        """
        if not self.predictions:
            raise ValueError("예측값이 없습니다. add_predictions()를 먼저 호출하세요.")
        
        if weights is not None:
            self.set_weights(weights)
        
        if self.weights is None:
            # If no weights specified, use simple average
            return self.predict_simple_average()
        
        # Weighted average
        weighted_sum = np.zeros(len(list(self.predictions.values())[0]))
        
        for model_name, predictions in self.predictions.items():
            if model_name in self.weights:
                weighted_sum += self.weights[model_name] * predictions
        
        self.ensemble_prediction = weighted_sum
        return self.ensemble_prediction

    def evaluate_ensemble(self, y_true: pd.Series, y_pred: np.ndarray = None) -> Dict[str, float]:
        """
        Evaluate ensemble performance.
        
        Args:
            y_true: True values
            y_pred: Ensemble predictions (if None, use stored predictions)
            
        Returns:
            Dictionary with evaluation metrics
        """
        if y_pred is None:
            if self.ensemble_prediction is None:
                raise ValueError("앙상블 예측값이 없습니다. predict_*() 메소드를 먼저 호출하세요.")
            y_pred = self.ensemble_prediction
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'n_samples': len(y_true)
        }
        
        return metrics

    def evaluate_individual_models(self, y_true: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Evaluate individual model performances.
        
        Args:
            y_true: True values
            
        Returns:
            Dictionary with metrics for each model
        """
        results = {}
        
        for model_name, predictions in self.predictions.items():
            metrics = {
                'rmse': np.sqrt(mean_squared_error(y_true, predictions)),
                'r2': r2_score(y_true, predictions),
                'mae': mean_absolute_error(y_true, predictions),
                'n_samples': len(y_true)
            }
            results[model_name] = metrics
        
        return results

    def get_best_weights_by_performance(self, y_true: pd.Series, 
                                       metric: str = 'rmse') -> Dict[str, float]:
        """
        Calculate weights based on individual model performance.
        
        Args:
            y_true: True values for calculating performance
            metric: Metric to use for weighting ('rmse', 'r2', 'mae')
            
        Returns:
            Dictionary with calculated weights
        """
        individual_metrics = self.evaluate_individual_models(y_true)
        
        if metric == 'rmse' or metric == 'mae':
            # Lower is better - use inverse
            scores = {name: 1.0 / metrics[metric] for name, metrics in individual_metrics.items()}
        elif metric == 'r2':
            # Higher is better - use as is
            scores = {name: metrics[metric] for name, metrics in individual_metrics.items()}
        else:
            raise ValueError(f"지원하지 않는 메트릭입니다: {metric}")
        
        # Normalize to sum to 1
        total_score = sum(scores.values())
        weights = {name: score / total_score for name, score in scores.items()}
        
        return weights

    def compare_ensemble_methods(self, y_true: pd.Series) -> pd.DataFrame:
        """
        Compare different ensemble methods.
        
        Args:
            y_true: True values
            
        Returns:
            DataFrame comparing ensemble methods
        """
        results = []
        
        # Simple average
        simple_pred = self.predict_simple_average()
        simple_metrics = self.evaluate_ensemble(y_true, simple_pred)
        simple_metrics['method'] = 'Simple Average'
        results.append(simple_metrics)
        
        # Weighted by RMSE
        rmse_weights = self.get_best_weights_by_performance(y_true, 'rmse')
        rmse_pred = self.predict_weighted_average(rmse_weights)
        rmse_metrics = self.evaluate_ensemble(y_true, rmse_pred)
        rmse_metrics['method'] = 'RMSE Weighted'
        results.append(rmse_metrics)
        
        # Weighted by R²
        r2_weights = self.get_best_weights_by_performance(y_true, 'r2')
        r2_pred = self.predict_weighted_average(r2_weights)
        r2_metrics = self.evaluate_ensemble(y_true, r2_pred)
        r2_metrics['method'] = 'R² Weighted'
        results.append(r2_metrics)
        
        return pd.DataFrame(results)

    def print_ensemble_summary(self, y_true: pd.Series) -> None:
        """
        Print summary of ensemble results.
        
        Args:
            y_true: True values for evaluation
        """
        print("\n" + "="*80)
        print("앙상블 모델 성능 비교")
        print("="*80)
        
        # Individual model performance
        individual_results = self.evaluate_individual_models(y_true)
        print("\n개별 모델 성능:")
        print(f"{'모델명':<15} {'RMSE':<10} {'R²':<10} {'MAE':<10}")
        print("-"*50)
        for model_name, metrics in individual_results.items():
            print(f"{model_name:<15} {metrics['rmse']:<10.3f} {metrics['r2']:<10.3f} {metrics['mae']:<10.3f}")
        
        # Ensemble comparison
        ensemble_comparison = self.compare_ensemble_methods(y_true)
        print("\n앙상블 방법 비교:")
        print(f"{'방법':<15} {'RMSE':<10} {'R²':<10} {'MAE':<10}")
        print("-"*50)
        for _, row in ensemble_comparison.iterrows():
            print(f"{row['method']:<15} {row['rmse']:<10.3f} {row['r2']:<10.3f} {row['mae']:<10.3f}")
        
        print("="*80)
