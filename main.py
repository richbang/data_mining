"""
119 응급신고 예측 시스템 메인 파이프라인
"""

import time
import pandas as pd
import numpy as np
from typing import Tuple

# 프로젝트 모듈들 import
# 처음에는 다 한 파일에 있었는데 유지보수 어려워서 분리함
from src.config.settings import Config
from src.data.data_loader import DataLoader  # 데이터 로딩 및 전처리 담당
from src.features.feature_engineer import FeatureEngineer  # 피처 생성 (원본 데이터 weekday 오류 해결 포함)
from src.features.stats_holder import StatsHolder  # 통계 피처들 (거리, 클러스터 등)
from src.features.pca_holder import PCAHolder  # PCA 처리 (결국 안 쓰기로 했지만 혹시 몰라서 남겨둠)
from src.features.smart_feature_selector import SmartFeatureSelector  # 승정님 제안: Elastic Net 자동 변수선택
from src.models.model_trainer import ModelTrainer  # 모델 훈련 및 관리
from src.models.ensemble_model import EnsembleModel  # 앙상블 모델 (여러 모델 결합)
from src.utils.evaluation import print_model_comparison, save_evaluation_results  # 결과 출력 및 저장
from src.utils.variable_tracker import VariableTracker  # 투명성 확보를 위한 변수 추적
from src.utils.visualization import (  # 시각화 유틸리티들
    create_model_report_plots, plot_model_comparison, plot_predictions_vs_actual,
    save_feature_importance
)


def create_features(df, is_train=True, stats_holder=None, pca_holder=None, config=None):
    """
    피처 엔지니어링 파이프라인
    
    처음에는 모든 피처를 한번에 만들려고 했는데, 메모리 부족으로 단계별로 나눔
    train/test에서 동일한 전처리 적용되도록 stats_holder와 pca_holder 재사용
    """
    print(f"피처 엔지니어링 시작 ({'훈련' if is_train else '테스트'} 데이터)")
    
    start_time = time.time()
    
    # 기본 피처 엔지니어링 (시간, 날씨, lag/rolling 등)
    # 박민혜님 지적사항: 원본 데이터 weekday 오류 문제 해결됨
    fe = FeatureEngineer()
    df_featured = fe.engineer_all_features(
        df, 
        lag_days=config.LAG_DAYS,      # [1, 3, 7] - 1일, 3일, 7일 전 데이터
        windows=config.ROLLING_WINDOWS, # [3, 7, 14] - 롤링 윈도우 크기들
        weather_cols=config.WEATHER_FEATURES  # 날씨 상호작용 피처용 컬럼들
    )
    
    # 통계적 피처들 (거리, 클러스터, 네트워크 등)
    # 이 부분이 시간이 제일 오래 걸림 (특히 네트워크 피처)
    if is_train:
        stats_holder = StatsHolder(
            city_coords=config.CITY_COORDINATES,  # 부산 중심 좌표
            coast_lat=config.COAST_LAT  # 위키에서 찾은 좌표 (거리 계산용)
        )
        stats_holder.fit(df_featured)  # train 데이터로 통계 학습
    
    # test에서는 train에서 학습한 통계 적용
    df_featured = stats_holder.transform(df_featured)
    
    # PCA 피처 (효과 없어서 기본값 False)
    # 하지만 혹시 나중에 실험해볼 수 있게 코드는 남겨둠
    if config.USE_PCA:
        # call_count는 target이므로 PCA에서 제외
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
    """
    메인 파이프라인 실행 함수
    """
    print("="*80)
    print("119 Call Prediction Pipeline 시작")
    print("="*80)
    
    total_start_time = time.time()
    
    # 설정 불러오기 (settings.py에서 모든 하이퍼파라미터 관리)
    config = Config()
    
    # 투명성 확보를 위한 변수 추적기 초기화
    tracker = VariableTracker()
    print("✅ 변수 추적 시스템 초기화 완료")
    
    # 출력 폴더들 생성 확인
    import os
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{config.OUTPUT_DIR}/plots", exist_ok=True)
    print(f"📁 출력 폴더 준비 완료: {config.OUTPUT_DIR}/")
    
    # 1. 데이터 로딩
    # UTF-8 인코딩 문제로 한참 고생했었음 (원본이 EUC-KR이었음)
    print("\n1. 데이터 로딩")
    data_loader = DataLoader(config.DATA_FILE, config.ENCODING)
    data = data_loader.load_data()
    
    # 2. 시계열 기반 train/test 분할
    # 2020-2022년: train, 2023년: test (미래 예측이므로)
    print("\n2. Train/Test 분할")
    train_df, test_df = data_loader.split_train_test(config.TRAIN_YEARS, config.TEST_YEARS)
    
    # 3. 피처 엔지니어링 (가장 시간 오래 걸리는 부분)
    # train과 test에 동일한 전처리 적용되도록 주의
    print("\n3. 피처 엔지니어링")
    train_featured, stats_holder, pca_holder = create_features(train_df, is_train=True, config=config)
    test_featured, _, _ = create_features(test_df, is_train=False, 
                                        stats_holder=stats_holder,  # train에서 학습된 통계 재사용
                                        pca_holder=pca_holder,      # train에서 학습된 PCA 재사용
                                        config=config)
    
    # 피처 엔지니어링 후 변수 목록 추적 저장
    tracker.save_initial_variables(train_featured, "after_feature_engineering")
    print(f"📊 피처 엔지니어링 결과: {len(train_featured.columns)}개 변수 생성")
    
    # 4. 피처 선택
    # 기존: 상관관계 임계값만 사용 → 53개 고상관 변수쌍으로 다중공선성 심각
    # 개선: Elastic Net으로 자동 변수선택 → 다중공선성 근본 해결
    print("\n4. 피처 선택")
    print("높은 상관관계 변수쌍이 너무 많아서 Elastic Net 먼저 적용")
    
    # SmartFeatureSelector로 Elastic Net 기반 변수선택
    selector = SmartFeatureSelector(correlation_threshold=config.CORRELATION_THRESHOLD)
    
    # Elastic Net + 상관관계 정제 결합
    final_features, selection_results = selector.combined_selection(
        train_featured, 
        target_col='call_count',
        use_elastic_net=True,    # 엘라스틱 넷 사용 여부
        use_correlation=False    # Elastic Net이면 충분함 (중복 제거 방지)
    )
    
    # 결과 저장 (투명성 확보)
    if selection_results['elastic_net_results']:
        selector.save_analysis_results(
            selection_results['elastic_net_results'], 
            output_prefix="elastic_net_selection"
        )
    
    # 상관관계 분석 결과를 tracker에 저장 (투명성 강화)
    if hasattr(selector, 'correlation_matrix') and selector.correlation_matrix is not None:
        removed_vars = [col for col in train_featured.select_dtypes(include=[np.number]).columns 
                       if col not in final_features and col != 'call_count']
        tracker.save_correlation_analysis(
            selector.correlation_matrix, 
            config.CORRELATION_THRESHOLD,
            removed_vars, 
            final_features
        )
    
    # train/test에 동일한 피처 적용
    final_features_with_target = final_features + ['call_count']
    train_cleaned = train_featured[final_features_with_target].copy()
    test_cleaned = test_featured[final_features_with_target].copy()
    
    print(f"   Elastic Net 기반 최종 선택: {len(final_features)}개 변수")
    
    # 최종 선택된 변수 목록 저장
    tracker.save_initial_variables(train_cleaned, "after_feature_selection")
    
    # 5. 모델링용 데이터 준비
    feature_cols = final_features  # Elastic Net으로 선택된 피처들
    X_train = data_loader.clean_features(train_cleaned[feature_cols])  # inf, nan 처리
    y_train = train_cleaned['call_count']
    X_test = data_loader.clean_features(test_cleaned[feature_cols])
    y_test = test_cleaned['call_count']
    
    print(f"최종 피처 수: {len(feature_cols)} (Elastic Net 선택)")
    
    # 6. 모델 훈련 (4개 모델 동시 훈련)
    # 하이퍼파라미터는 이전 실험들을 통해 튜닝된 값들
    print("\n5. 모델 훈련 및 평가")
    trainer = ModelTrainer()
    trainer.train_all_models(X_train, y_train, 
                           rf_params=config.RF_PARAMS,       # RandomForest 파라미터
                           lgbm_params=config.LGBM_PARAMS,   # LightGBM 파라미터  
                           xgb_params=config.XGB_PARAMS,     # XGBoost 파라미터
                           cat_params=config.CATBOOST_PARAMS) # CatBoost 파라미터
    
    # 7. 모델 평가 및 변수 중요도 분석
    results = trainer.evaluate_all_models(X_test, y_test)
    
    # 각 모델의 변수 중요도를 tracker에 저장 (투명성 강화)
    print("\n📈 모델별 변수 중요도 분석 및 저장")
    model_predictions = {}
    
    for model_name in ['RandomForest', 'LightGBM', 'XGBoost', 'CatBoost']:
        if model_name in trainer.models:
            model = trainer.models[model_name]
            predictions = model.predict(X_test)
            model_predictions[model_name] = predictions
            
            # 변수 중요도 저장
            tracker.save_model_importance(
                model_name, model, feature_cols, X_test, y_test, predictions
            )
            
            # 개별 모델 시각화 생성
            create_model_report_plots(y_test, predictions, model_name, f"{config.OUTPUT_DIR}/plots")
            
            # 피처 중요도 시각화 (visualization.py 활용)
            if hasattr(model, 'feature_importances_'):
                importance_series = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
                save_feature_importance(importance_series, f"{config.OUTPUT_DIR}/{model_name}_feature_importance.png", 
                                       f"{model_name} Feature Importance", top_n=20)
    
    # 8. 앙상블 모델 생성 및 평가
    print("\n🔀 앙상블 모델 생성 및 평가")
    ensemble = EnsembleModel()
    
    # 각 모델의 예측값을 앙상블에 추가
    for model_name, predictions in model_predictions.items():
        ensemble.add_predictions(model_name, predictions)
    
    # 여러 앙상블 방법 비교
    ensemble_comparison = ensemble.compare_ensemble_methods(y_test)
    print("\n앙상블 성능 비교:")
    print(ensemble_comparison)
    
    # 최고 성능 앙상블 방법 선택
    best_ensemble_method = ensemble_comparison.loc[ensemble_comparison['rmse'].idxmin(), 'method']
    print(f"\n🏆 최고 성능 앙상블 방법: {best_ensemble_method}")
    
    # 최적 앙상블 예측 생성
    if best_ensemble_method == 'Simple Average':
        best_ensemble_pred = ensemble.predict_simple_average()
    elif best_ensemble_method == 'RMSE Weighted':
        best_weights = ensemble.get_best_weights_by_performance(y_test, 'rmse')
        best_ensemble_pred = ensemble.predict_weighted_average(best_weights)
    else:  # R² Weighted
        best_weights = ensemble.get_best_weights_by_performance(y_test, 'r2')
        best_ensemble_pred = ensemble.predict_weighted_average(best_weights)
    
    # 앙상블 결과를 results에 추가
    ensemble_metrics = ensemble.evaluate_ensemble(y_test, best_ensemble_pred)
    results[f'Ensemble_{best_ensemble_method.replace(" ", "_")}'] = ensemble_metrics
    
    # 앙상블 시각화
    create_model_report_plots(y_test, best_ensemble_pred, f'Ensemble_{best_ensemble_method}', f"{config.OUTPUT_DIR}/plots")
    
    # 9. 결과 출력 (RMSE 기준으로 정렬해서 보여줌)
    print("\n📊 최종 결과 요약 (앙상블 포함)")
    print_model_comparison(results)
    
    # 모델 비교 차트 생성
    plot_model_comparison(results, f"{config.OUTPUT_DIR}/model_comparison.png")
    
    # 10. 결과 저장 (CSV 파일로 저장해서 나중에 분석 가능)
    save_evaluation_results(results, f"{config.OUTPUT_DIR}/evaluation_results.csv")
    
    # 11. 변수 추적 종합 요약 생성
    print("\n📋 변수 선택 과정 종합 요약 생성")
    tracker.save_comprehensive_summary()
    
    # 전체 실행 시간 출력
    total_elapsed = time.time() - total_start_time
    print(f"\n🎉 전체 파이프라인 완료! 총 소요 시간: {total_elapsed:.2f}초")
    print(f"📁 모든 결과 파일이 {config.OUTPUT_DIR}/ 폴더에 저장되었습니다:")
    print(f"   - 평가 결과: evaluation_results.csv")
    print(f"   - 변수 분석: comprehensive_feature_analysis.csv")
    print(f"   - 모델 시각화: plots/ 폴더")
    print(f"   - 피처 중요도: *_feature_importance.png")
    
    # 최종 베스트 모델 출력
    if results:
        best_model = min(results.items(), key=lambda x: x[1]['rmse'])
        print(f"\n🏆 최고 성능 모델: {best_model[0]} (RMSE: {best_model[1]['rmse']:.3f}, R²: {best_model[1]['r2']:.3f})")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
