"""Main pipeline for 119 Call Prediction project."""

import time
import pandas as pd
import numpy as np
from typing import Tuple

# Import project modules
from src.config.settings import Config
from src.data.data_loader import DataLoader
from src.features.feature_engineer import FeatureEngineer
from src.features.stats_holder import StatsHolder
from src.features.pca_holder import PCAHolder
from src.models.model_trainer import ModelTrainer
from src.utils.evaluation import print_model_comparison, save_evaluation_results


def create_features(df, is_train=True, stats_holder=None, pca_holder=None, config=None):
    """Create features using feature engineering pipeline."""
    print(f"피처 엔지니어링 시작 ({'훈련' if is_train else '테스트'} 데이터)")
    
    start_time = time.time()
    
    # Feature engineering
    fe = FeatureEngineer()
    df_featured = fe.engineer_all_features(
        df, 
        lag_days=config.LAG_DAYS,
        windows=config.ROLLING_WINDOWS,
        weather_cols=config.WEATHER_FEATURES
    )
    
    # Statistical features
    if is_train:
        stats_holder = StatsHolder(
            city_coords=config.CITY_COORDINATES,
            coast_lat=config.COAST_LAT
        )
        stats_holder.fit(df_featured)
    
    df_featured = stats_holder.transform(df_featured)
    
    # PCA features (팀원 논의 결과: 조건부 사용)
    if config.USE_PCA:
        numeric_cols = df_featured.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ['call_count']]
        
        if is_train:
            pca_holder = PCAHolder(n_components=config.PCA_COMPONENTS)
            pca_holder.fit(df_featured, numeric_cols)
        
        df_featured = pca_holder.transform(df_featured)
        print(f"PCA 피처 {config.PCA_COMPONENTS}개 추가됨")
    else:
        print("PCA 피처 사용 안함")
    
    elapsed = time.time() - start_time
    print(f"피처 엔지니어링 완료. 소요 시간: {elapsed:.2f}초")
    
    return df_featured, stats_holder, pca_holder


def main():
    """Main pipeline execution."""
    print("="*80)
    print("119 Call Prediction Pipeline 시작")
    print("="*80)
    
    total_start_time = time.time()
    
    # Initialize configuration
    config = Config()
    
    # Data Loading
    print("\n1. 데이터 로딩")
    data_loader = DataLoader(config.DATA_FILE, config.ENCODING)
    data = data_loader.load_data()
    
    # Train/Test Split
    print("\n2. Train/Test 분할")
    train_df, test_df = data_loader.split_train_test(config.TRAIN_YEARS, config.TEST_YEARS)
    
    # Feature Engineering
    print("\n3. 피처 엔지니어링")
    train_featured, stats_holder, pca_holder = create_features(train_df, is_train=True, config=config)
    test_featured, _, _ = create_features(test_df, is_train=False, 
                                        stats_holder=stats_holder, 
                                        pca_holder=pca_holder, 
                                        config=config)
    
    # Feature Selection (팀원 논의 반영: 도메인 지식 기반 + threshold 완화)
    print("\n4. 피처 선택")
    print(f"상관관계 기준: {config.CORRELATION_THRESHOLD} (팀원 요청에 따라 완화)")
    train_cleaned, removed_features = data_loader.remove_highly_correlated_features(
        train_featured, 
        threshold=config.CORRELATION_THRESHOLD,
        save_analysis=True  # 투명성 확보
    )
    keep_features = [col for col in train_cleaned.columns if col in test_featured.columns]
    test_cleaned = test_featured[keep_features].copy()
    print(f"도메인 지식 기반으로 {len(removed_features)}개 피처 제거됨")
    
    # Prepare datasets
    feature_cols = data_loader.get_feature_columns(train_cleaned)
    X_train = data_loader.clean_features(train_cleaned[feature_cols])
    y_train = train_cleaned['call_count']
    X_test = data_loader.clean_features(test_cleaned[feature_cols])
    y_test = test_cleaned['call_count']
    
    print(f"최종 피처 수: {len(feature_cols)}")
    
    # Model Training
    print("\n5. 모델 훈련 및 평가")
    trainer = ModelTrainer()
    trainer.train_all_models(X_train, y_train, 
                           rf_params=config.RF_PARAMS,
                           lgbm_params=config.LGBM_PARAMS,
                           xgb_params=config.XGB_PARAMS,
                           cat_params=config.CATBOOST_PARAMS)
    
    # Evaluate models
    results = trainer.evaluate_all_models(X_test, y_test)
    
    # Print results
    print("\n최종 결과 요약")
    print_model_comparison(results)
    
    # Save results
    save_evaluation_results(results, f"{config.OUTPUT_DIR}/evaluation_results.csv")
    
    # Total execution time
    total_elapsed = time.time() - total_start_time
    print(f"\n전체 파이프라인 완료! 총 소요 시간: {total_elapsed:.2f}초")


if __name__ == "__main__":
    main()
