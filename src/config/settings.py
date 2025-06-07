"""Configuration settings for 119 Call Prediction project."""

import os
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class Config:
    """Configuration class for the 119 call prediction project."""
    
    # Data settings
    DATA_FILE: str = 'human_combined_full_data_utf8.csv'
    ENCODING: str = 'utf-8'
    
    # Train/Test split
    TRAIN_YEARS: List[int] = None
    TEST_YEARS: List[int] = None
    
    # City coordinates (Busan)
    CITY_COORDINATES: tuple = (35.1795543, 129.0756416)
    COAST_LAT: float = 34.8902691
    
    # Feature engineering settings
    LAG_DAYS: List[int] = None
    ROLLING_WINDOWS: List[int] = None
    WEATHER_FEATURES: List[str] = None
    
    # Model settings
    RANDOM_STATE: int = 42
    N_JOBS: int = -1
    
    # PCA settings (팀원 논의 결과: PCA 사용 여부 선택 가능)
    USE_PCA: bool = False # True가 성능이 더 낮음
    PCA_COMPONENTS: int = 3
    
    # Feature selection settings
    CORRELATION_THRESHOLD: float = 0.95  # 스레쉬홀드 선택
    USE_FEATURE_IMPORTANCE_FILTERING: bool = False  # 팀원 우려에 따라 기본값 False
    
    # Model parameters
    RF_PARAMS: Dict[str, Any] = None
    LGBM_PARAMS: Dict[str, Any] = None
    XGB_PARAMS: Dict[str, Any] = None
    CATBOOST_PARAMS: Dict[str, Any] = None
    
    # Output settings
    OUTPUT_DIR: str = 'outputs'
    SAVE_PREDICTIONS: bool = True
    SAVE_FEATURE_IMPORTANCE: bool = True
    SAVE_SHAP_PLOTS: bool = True
    
    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.TRAIN_YEARS is None:
            self.TRAIN_YEARS = [2020, 2021, 2022]
        
        if self.TEST_YEARS is None:
            self.TEST_YEARS = [2023]
        
        if self.LAG_DAYS is None:
            self.LAG_DAYS = [1, 3, 7]
        
        if self.ROLLING_WINDOWS is None:
            self.ROLLING_WINDOWS = [3, 7, 14]
        
        if self.WEATHER_FEATURES is None:
            self.WEATHER_FEATURES = ['ta_max', 'ta_min', 'hm_max', 'hm_min', 'ws_max', 'rn_day']
        
        if self.RF_PARAMS is None:
            self.RF_PARAMS = {
                'n_estimators': 200,
                'max_depth': 10,
                'random_state': self.RANDOM_STATE,
                'n_jobs': self.N_JOBS
            }
        
        if self.LGBM_PARAMS is None:
            self.LGBM_PARAMS = {
                'n_estimators': 200,
                'random_state': self.RANDOM_STATE,
                'n_jobs': self.N_JOBS
            }
        
        if self.XGB_PARAMS is None:
            self.XGB_PARAMS = {
                'n_estimators': 200,
                'max_depth': 10,
                'random_state': self.RANDOM_STATE,
                'n_jobs': self.N_JOBS,
                'tree_method': 'hist'
            }
        
        if self.CATBOOST_PARAMS is None:
            self.CATBOOST_PARAMS = {
                'iterations': 200,
                'depth': 10,
                'random_seed': self.RANDOM_STATE,
                'verbose': 0
            }
        
        # Create output directory if it doesn't exist
        os.makedirs(self.OUTPUT_DIR, exist_ok=True) 