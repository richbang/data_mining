"""PCA effect comparison experiment module for team discussion support."""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# 기존 PCAHolder 사용 (중복 방지)
from src.features.pca_holder import PCAHolder


class PCAExperiment:
    """
    PCA 사용/미사용 효과를 비교하는 실험 클래스.

    **실험 내용:**
    - PCA 미사용 vs 사용 성능 비교
    - 변수 개수 변화 분석
    - 모델별 효과 차이 확인
    """
    
    def __init__(self):
        self.results = {}
        self.recommendation = ""
    
    def compare_with_without_pca(self, train_feat: pd.DataFrame, test_feat: pd.DataFrame, 
                                final_features: List[str]) -> Dict[str, Any]:
        """PCA 사용/미사용 성능 비교 실험"""
        print("\n" + "="*60)
        print("🔬 PCA 효과 분석 실험 (팀원 요청사항)")
        print("="*60)
        
        results = {}
        
        # PCA 적용할 기상 변수들
        pca_features = ['ta_max', 'ta_min', 'hm_max', 'hm_min', 'ws_max', 'rn_day']
        available_pca_features = [f for f in pca_features if f in train_feat.columns]
        
        if len(available_pca_features) < 3:
            print("❌ PCA 적용 가능한 기상 변수가 부족합니다.")
            return None
        
        print(f"📊 PCA 적용 대상: {available_pca_features}")
        
        # 1. PCA 없는 버전
        print("\n1️⃣ PCA 미사용 버전")
        results['without_pca'] = self._run_models_without_pca(
            train_feat, test_feat, final_features
        )
        
        # 2. PCA 있는 버전  
        print("\n2️⃣ PCA 사용 버전")
        results['with_pca'] = self._run_models_with_pca(
            train_feat, test_feat, final_features, available_pca_features
        )
        
        # 3. 결과 비교
        self._compare_results(results)
        
        # 4. 결과 저장
        self._save_comparison_results(results)
        
        return results
    
    def _run_models_without_pca(self, train_feat: pd.DataFrame, test_feat: pd.DataFrame, 
                               final_features: List[str]) -> Dict[str, Any]:
        """PCA 없이 모델 실행"""
        # PCA 변수들 제거
        features_no_pca = [f for f in final_features if not f.startswith('pca_')]
        
        X_train = train_feat[features_no_pca].copy()
        X_test = test_feat[features_no_pca].copy()
        y_train = train_feat['call_count'].copy()
        y_test = test_feat['call_count'].copy()
        
        # 안전한 결측치 처리
        X_train = X_train.fillna(X_train.mean())
        X_test = X_test.fillna(X_train.mean())  # train의 평균 사용
        
        print(f"   📋 변수 개수: {len(features_no_pca)}개")
        
        results = {}
        
        # LightGBM
        lgbm_model = LGBMRegressor(n_estimators=200, random_state=42, verbose=-1)
        lgbm_model.fit(X_train, y_train)
        lgbm_pred = lgbm_model.predict(X_test)
        
        results['LightGBM'] = {
            'RMSE': np.sqrt(mean_squared_error(y_test, lgbm_pred)),
            'R2': r2_score(y_test, lgbm_pred),
            'feature_count': len(features_no_pca)
        }
        
        # RandomForest
        rf_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        
        results['RandomForest'] = {
            'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
            'R2': r2_score(y_test, rf_pred),
            'feature_count': len(features_no_pca)
        }
        
        print(f"   ✅ LightGBM - RMSE: {results['LightGBM']['RMSE']:.4f}, R²: {results['LightGBM']['R2']:.4f}")
        print(f"   ✅ RandomForest - RMSE: {results['RandomForest']['RMSE']:.4f}, R²: {results['RandomForest']['R2']:.4f}")
        
        return results
    
    def _run_models_with_pca(self, train_feat: pd.DataFrame, test_feat: pd.DataFrame, 
                            final_features: List[str], pca_features: List[str]) -> Dict[str, Any]:
        """PCA 포함하여 모델 실행"""
        # PCA 적용 (기존 PCAHolder 방식)
        pca_holder = PCAHolder(n_components=3)
        train_with_pca = train_feat.copy()
        test_with_pca = test_feat.copy()
        
        # 기존 방식: fit 후 transform
        pca_holder.fit(train_with_pca, pca_features)
        train_with_pca = pca_holder.transform(train_with_pca)
        test_with_pca = pca_holder.transform(test_with_pca)
        
        # PCA 변수 추가된 피처 리스트
        pca_columns = ['pca_1', 'pca_2', 'pca_3']
        features_with_pca = final_features + pca_columns
        
        # PCA 변수가 이미 포함된 경우 중복 제거
        features_with_pca = list(set(features_with_pca))
        
        X_train = train_with_pca[features_with_pca].copy()
        X_test = test_with_pca[features_with_pca].copy()
        y_train = train_with_pca['call_count'].copy()
        y_test = test_with_pca['call_count'].copy()
        
        # 안전한 결측치 처리
        X_train = X_train.fillna(X_train.mean())
        X_test = X_test.fillna(X_train.mean())  # train의 평균 사용
        
        print(f"   📋 변수 개수: {len(features_with_pca)}개 (PCA +3개)")
        
        results = {}
        
        # LightGBM
        lgbm_model = LGBMRegressor(n_estimators=200, random_state=42, verbose=-1)
        lgbm_model.fit(X_train, y_train)
        lgbm_pred = lgbm_model.predict(X_test)
        
        results['LightGBM'] = {
            'RMSE': np.sqrt(mean_squared_error(y_test, lgbm_pred)),
            'R2': r2_score(y_test, lgbm_pred),
            'feature_count': len(features_with_pca)
        }
        
        # RandomForest
        rf_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        
        results['RandomForest'] = {
            'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
            'R2': r2_score(y_test, rf_pred),
            'feature_count': len(features_with_pca)
        }
        
        print(f"   ✅ LightGBM - RMSE: {results['LightGBM']['RMSE']:.4f}, R²: {results['LightGBM']['R2']:.4f}")
        print(f"   ✅ RandomForest - RMSE: {results['RandomForest']['RMSE']:.4f}, R²: {results['RandomForest']['R2']:.4f}")
        
        return results
    
    def _compare_results(self, results: Dict[str, Any]):
        """결과 비교 분석"""
        print("\n" + "="*60)
        print("📈 PCA 효과 분석 결과")
        print("="*60)
        
        comparison_df = []
        
        for model in ['LightGBM', 'RandomForest']:
            without_pca = results['without_pca'][model]
            with_pca = results['with_pca'][model]
            
            rmse_diff = with_pca['RMSE'] - without_pca['RMSE']
            r2_diff = with_pca['R2'] - without_pca['R2']
            
            comparison_df.append({
                'Model': model,
                'RMSE_without_PCA': without_pca['RMSE'],
                'RMSE_with_PCA': with_pca['RMSE'],
                'RMSE_차이': rmse_diff,
                'RMSE_개선율(%)': -rmse_diff/without_pca['RMSE']*100,
                'R2_without_PCA': without_pca['R2'],
                'R2_with_PCA': with_pca['R2'],
                'R2_차이': r2_diff,
                'Variables_without_PCA': without_pca['feature_count'],
                'Variables_with_PCA': with_pca['feature_count']
            })
        
        comp_df = pd.DataFrame(comparison_df)
        print(comp_df.round(4))
        
        # 권장사항 출력
        print("\n🎯 **팀원 논의 사항에 대한 분석 결과**")
        
        lgbm_improved = comp_df.loc[0, 'RMSE_개선율(%)'] > 0
        rf_improved = comp_df.loc[1, 'RMSE_개선율(%)'] > 0
        
        if lgbm_improved and rf_improved:
            print("✅ **PCA 사용 권장**: 두 모델 모두 성능 향상")
            recommendation = "PCA 사용"
        elif lgbm_improved or rf_improved:
            print("⚠️ **혼재된 결과**: 한 모델만 성능 향상")
            recommendation = "모델별 선택적 사용"
        else:
            print("❌ **PCA 미사용 권장**: 두 모델 모두 성능 저하")
            print("   🔍 양다현 팀원 의견 지지: '변수가 많으니 PCA 변수까지 추가 안 해도 괜찮다'")
            recommendation = "PCA 미사용"
        
        # 변수 개수 관점에서 분석
        var_increase = comp_df.loc[0, 'Variables_with_PCA'] - comp_df.loc[0, 'Variables_without_PCA']
        print(f"\n📊 **변수 개수 분석**")
        print(f"   • PCA 미사용: {comp_df.loc[0, 'Variables_without_PCA']}개 변수")
        print(f"   • PCA 사용: {comp_df.loc[0, 'Variables_with_PCA']}개 변수 (+{var_increase}개)")
        
        self.recommendation = recommendation
        
    def _save_comparison_results(self, results: Dict[str, Any]):
        """비교 결과 저장"""
        # 상세 결과 저장
        detailed_results = []
        for pca_status in ['without_pca', 'with_pca']:
            for model in ['LightGBM', 'RandomForest']:
                result = results[pca_status][model]
                detailed_results.append({
                    'PCA_Status': pca_status,
                    'Model': model,
                    'RMSE': result['RMSE'],
                    'R2': result['R2'],
                    'Feature_Count': result['feature_count']
                })
        
        detailed_df = pd.DataFrame(detailed_results)
        detailed_df.to_csv('pca_comparison_detailed.csv', index=False, encoding='utf-8')
        
        # 요약 결과 저장
        summary = {
            'Recommendation': self.recommendation,
            'Analysis_Date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        pd.DataFrame([summary]).to_csv('pca_analysis_summary.csv', index=False, encoding='utf-8')
        
        print(f"\n💾 결과 저장 완료:")
        print(f"   • pca_comparison_detailed.csv")
        print(f"   • pca_analysis_summary.csv")
        print(f"   • 최종 권장사항: {self.recommendation}")
    
    def get_recommendation(self) -> str:
        """팀 논의를 위한 최종 권장사항 반환"""
        return self.recommendation 