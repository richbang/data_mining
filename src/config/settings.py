"""
119 응급신고 예측 프로젝트 설정 파일

핵심 설정값들:
- CORRELATION_THRESHOLD: 0.85 → 0.90 → 0.95로 실험 후 결정
- USE_PCA: True → False로 변경
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class Config:
    """
    119 응급신고 예측 프로젝트 통합 설정 클래스
    
    이 파일 하나로 모든 실험 설정 관리:
    - 데이터 파일 경로 및 인코딩
    - 피처 엔지니어링 파라미터
    - 모델 하이퍼파라미터
    - 평가 및 출력 설정
    
    수정하고 싶은 설정이 있으면 여기서만 바꾸면 됨!
    """
    
    # 📁 데이터 관련 설정
    DATA_FILE: str = 'human_combined_full_data_utf8.csv'  # UTF-8로 변환된 데이터 파일
    ENCODING: str = 'utf-8'  # UTF-8로 변환해서 사용
    
    # 📅 Train/Test 분할 (시계열 특성 고려)
    TRAIN_YEARS: List[int] = None  # [2020, 2021, 2022] - 3년간 훈련
    TEST_YEARS: List[int] = None   # [2023]
    
    # 🗺️ 부산시 지리 정보 (거리 계산용)
    CITY_COORDINATES: tuple = (35.1795543, 129.0756416)  # 부산 중심 좌표
    COAST_LAT: float = 34.8902691
    
    # 🔧 피처 엔지니어링 설정
    LAG_DAYS: List[int] = None        # [1, 3, 7] - 지연 피처 생성 일수
    ROLLING_WINDOWS: List[int] = None # [3, 7, 14] - 롤링 윈도우 크기들
    WEATHER_FEATURES: List[str] = None # 상호작용 피처 생성용 기상 변수들
    
    # ⚙️ 모델 공통 설정
    RANDOM_STATE: int = 42  # 재현성을 위한 고정 시드
    N_JOBS: int = -1        # 모든 CPU 코어 사용
    
    # 🎯 PCA 설정
    USE_PCA: bool = True #False   # PCA 사용 여부
    PCA_COMPONENTS: int = 3 # PCA 사용 시 주성분 개수
    
    # 🔍 피처 선택 설정 (핵심!)
    CORRELATION_THRESHOLD: float = 0.95  # 실험 후 결정
    USE_FEATURE_IMPORTANCE_FILTERING: bool = False  # 자동 필터링 비활성화
    
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