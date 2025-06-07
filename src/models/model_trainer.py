"""Model training and evaluation module."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


class ModelTrainer:
    """
    Model trainer for 119 call prediction.
    Supports RandomForest, LightGBM, XGBoost, and CatBoost.
    """
    
    def __init__(self):
        """Initialize ModelTrainer."""
        self.models: Dict[str, Any] = {}
        self.predictions: Dict[str, np.ndarray] = {}
        self.metrics: Dict[str, Dict[str, float]] = {}

    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series,
                           params: Dict[str, Any] = None,
                           model_name: str = "RandomForest") -> RandomForestRegressor:
        """
        Train RandomForest model.
        
        Args:
            X_train: Training features
            y_train: Training target
            params: Model parameters
            model_name: Name for the model
            
        Returns:
            Trained RandomForest model
        """
        if params is None:
            params = {
                'n_estimators': 200,
                'max_depth': 10,
                'random_state': 42,
                'n_jobs': -1
            }
        
        print(f"[{model_name}] 모델 훈련 시작")
        start_time = time.time()
        
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        
        elapsed = time.time() - start_time
        print(f"[{model_name}] 모델 훈련 완료. 소요 시간: {elapsed:.2f}초")
        
        self.models[model_name] = model
        return model

    def train_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series,
                      params: Dict[str, Any] = None,
                      model_name: str = "LightGBM") -> LGBMRegressor:
        """
        Train LightGBM model.
        
        Args:
            X_train: Training features
            y_train: Training target
            params: Model parameters
            model_name: Name for the model
            
        Returns:
            Trained LightGBM model
        """
        if params is None:
            params = {
                'n_estimators': 200,
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': -1
            }
        
        print(f"[{model_name}] 모델 훈련 시작")
        start_time = time.time()
        
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)
        
        elapsed = time.time() - start_time
        print(f"[{model_name}] 모델 훈련 완료. 소요 시간: {elapsed:.2f}초")
        
        self.models[model_name] = model
        return model

    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                     params: Dict[str, Any] = None,
                     model_name: str = "XGBoost") -> XGBRegressor:
        """
        Train XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training target
            params: Model parameters
            model_name: Name for the model
            
        Returns:
            Trained XGBoost model
        """
        if params is None:
            params = {
                'n_estimators': 200,
                'max_depth': 10,
                'random_state': 42,
                'n_jobs': -1,
                'tree_method': 'hist'
            }
        
        print(f"[{model_name}] 모델 훈련 시작")
        start_time = time.time()
        
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        elapsed = time.time() - start_time
        print(f"[{model_name}] 모델 훈련 완료. 소요 시간: {elapsed:.2f}초")
        
        self.models[model_name] = model
        return model

    def train_catboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                      params: Dict[str, Any] = None,
                      model_name: str = "CatBoost") -> CatBoostRegressor:
        """
        Train CatBoost model.
        
        Args:
            X_train: Training features
            y_train: Training target
            params: Model parameters
            model_name: Name for the model
            
        Returns:
            Trained CatBoost model
        """
        if params is None:
            params = {
                'iterations': 200,
                'depth': 10,
                'random_seed': 42,
                'verbose': 0
            }
        
        print(f"[{model_name}] 모델 훈련 시작")
        start_time = time.time()
        
        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train)
        
        elapsed = time.time() - start_time
        print(f"[{model_name}] 모델 훈련 완료. 소요 시간: {elapsed:.2f}초")
        
        self.models[model_name] = model
        return model

    def predict(self, model_name: str, X_test: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using trained model.
        
        Args:
            model_name: Name of the model to use
            X_test: Test features
            
        Returns:
            Predictions array
        """
        if model_name not in self.models:
            raise ValueError(f"모델 '{model_name}'이 훈련되지 않았습니다.")
        
        predictions = self.models[model_name].predict(X_test)
        self.predictions[model_name] = predictions
        
        return predictions

    def evaluate_model(self, model_name: str, y_true: pd.Series, y_pred: np.ndarray = None) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            model_name: Name of the model
            y_true: True values
            y_pred: Predicted values (if None, use stored predictions)
            
        Returns:
            Dictionary with evaluation metrics
        """
        if y_pred is None:
            if model_name not in self.predictions:
                raise ValueError(f"모델 '{model_name}'의 예측값이 없습니다.")
            y_pred = self.predictions[model_name]
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'n_samples': len(y_true)
        }
        
        self.metrics[model_name] = metrics
        return metrics

    def get_feature_importance(self, model_name: str, feature_names: List[str] = None) -> pd.Series:
        """
        Get feature importance from trained model.
        
        Args:
            model_name: Name of the model
            feature_names: List of feature names
            
        Returns:
            Series with feature importance
        """
        if model_name not in self.models:
            raise ValueError(f"모델 '{model_name}'이 훈련되지 않았습니다.")
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            raise ValueError(f"모델 '{model_name}'은 feature importance를 지원하지 않습니다.")
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        return pd.Series(importances, index=feature_names).sort_values(ascending=False)

    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        rf_params: Dict[str, Any] = None,
                        lgbm_params: Dict[str, Any] = None,
                        xgb_params: Dict[str, Any] = None,
                        cat_params: Dict[str, Any] = None,
                        model_suffix: str = "") -> Dict[str, Any]:
        """
        Train all supported models.
        
        Args:
            X_train: Training features
            y_train: Training target
            rf_params: RandomForest parameters
            lgbm_params: LightGBM parameters
            xgb_params: XGBoost parameters
            cat_params: CatBoost parameters
            model_suffix: Suffix to add to model names
            
        Returns:
            Dictionary of trained models
        """
        models = {}
        
        # Train RandomForest
        try:
            rf_name = f"RandomForest{model_suffix}"
            models[rf_name] = self.train_random_forest(X_train, y_train, rf_params, rf_name)
        except Exception as e:
            print(f"RandomForest 훈련 중 오류: {e}")
        
        # Train LightGBM
        try:
            lgbm_name = f"LightGBM{model_suffix}"
            models[lgbm_name] = self.train_lightgbm(X_train, y_train, lgbm_params, lgbm_name)
        except Exception as e:
            print(f"LightGBM 훈련 중 오류: {e}")
        
        # Train XGBoost
        try:
            xgb_name = f"XGBoost{model_suffix}"
            models[xgb_name] = self.train_xgboost(X_train, y_train, xgb_params, xgb_name)
        except Exception as e:
            print(f"XGBoost 훈련 중 오류: {e}")
        
        # Train CatBoost
        try:
            cat_name = f"CatBoost{model_suffix}"
            models[cat_name] = self.train_catboost(X_train, y_train, cat_params, cat_name)
        except Exception as e:
            print(f"CatBoost 훈련 중 오류: {e}")
        
        return models

    def evaluate_all_models(self, X_test: pd.DataFrame, y_test: pd.Series,
                           model_suffix: str = "") -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models.
        
        Args:
            X_test: Test features
            y_test: Test target
            model_suffix: Suffix used in model names
            
        Returns:
            Dictionary with evaluation metrics for all models
        """
        all_metrics = {}
        model_names = [f"RandomForest{model_suffix}", f"LightGBM{model_suffix}", 
                      f"XGBoost{model_suffix}", f"CatBoost{model_suffix}"]
        
        for model_name in model_names:
            if model_name in self.models:
                try:
                    y_pred = self.predict(model_name, X_test)
                    metrics = self.evaluate_model(model_name, y_test, y_pred)
                    all_metrics[model_name] = metrics
                    
                    print(f"[{model_name}] RMSE: {metrics['rmse']:.3f}, "
                          f"R²: {metrics['r2']:.3f}, MAE: {metrics['mae']:.3f}, "
                          f"N: {metrics['n_samples']}")
                except Exception as e:
                    print(f"{model_name} 평가 중 오류: {e}")
        
        return all_metrics

    def save_feature_importance(self, model_name: str, feature_names: List[str],
                               filename: str, top_n: int = 30) -> None:
        """
        Save feature importance to CSV file.
        
        Args:
            model_name: Name of the model
            feature_names: List of feature names
            filename: Output filename
            top_n: Number of top features to save
        """
        try:
            importance = self.get_feature_importance(model_name, feature_names)
            importance.head(top_n).to_csv(filename, encoding='utf-8')
            print(f"{model_name} 피처 중요도 Top {top_n}을 {filename}에 저장했습니다.")
        except Exception as e:
            print(f"피처 중요도 저장 중 오류: {e}")

    def print_summary(self) -> None:
        """Print summary of all trained models and their performance."""
        print("\n" + "="*80)
        print("모델 성능 요약")
        print("="*80)
        print(f"{'모델명':<20} {'RMSE':<10} {'R²':<10} {'MAE':<10} {'샘플수':<10}")
        print("-"*80)
        
        for model_name, metrics in self.metrics.items():
            print(f"{model_name:<20} {metrics['rmse']:<10.3f} {metrics['r2']:<10.3f} "
                  f"{metrics['mae']:<10.3f} {metrics['n_samples']:<10}")
        
        print("="*80) 