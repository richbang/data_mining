"""Smart feature selection module with domain knowledge integration."""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class SmartFeatureSelector:
    """
    도메인 지식을 활용한 스마트 변수 선택 클래스.
    
    **주요 기능:**
    - 상관관계 기반 변수 제거 시 도메인 지식 적용
    - 기상 변수 우선 보존 정책
    - 투명한 선택 과정 추적
    - 팀원 논의 사항 반영
    """
    
    def __init__(self, correlation_threshold: float = 0.85):
        self.correlation_threshold = correlation_threshold
        self.domain_priorities = {
            'weather_original': ['ta_max', 'ta_min', 'ws_max', 'hm_max', 'hm_min', 'rn_day'],
            'target': ['call_count'],
            'temporal_basic': ['year', 'month', 'day', 'weekday'],
            'population': ['총인구수', '세대수', '남자.인구수', '여자.인구수'],
            'location': ['xcoord', 'ycoord'],
            'derived_important': ['temp_avg', 'humidity_avg', 'is_rainy', 'is_weekend']
        }
        self.removal_log = []
        self.selection_log = []
    
    def select_better_variable(self, var1: str, var2: str, correlation_val: float) -> str:
        """
        두 변수 중 제거할 변수를 도메인 지식 기반으로 선택.
        
        **우선순위 규칙:**
        1. 기상 원본 변수 > 파생 변수
        2. 짧은 이름 > 긴 이름 (단순성)
        3. 기본 변수 > 복잡한 상호작용 변수
        
        Args:
            var1: 첫 번째 변수명
            var2: 두 번째 변수명  
            correlation_val: 상관관계 값
            
        Returns:
            제거할 변수명
        """
        
        # 1. 기상 원본 변수 우선 보존
        for category, variables in self.domain_priorities.items():
            if var1 in variables and var2 not in variables:
                self._log_selection(var1, var2, correlation_val, f"도메인 우선순위: {category}")
                return var2
            elif var2 in variables and var1 not in variables:
                self._log_selection(var2, var1, correlation_val, f"도메인 우선순위: {category}")
                return var1
        
        # 2. 둘 다 같은 카테고리에 속하거나 둘 다 속하지 않는 경우
        
        # py_weekday vs weekday 특별 처리
        if {var1, var2} == {'weekday', 'py_weekday'}:
            self._log_selection('weekday', 'py_weekday', correlation_val, "중복 변수: weekday 우선")
            return 'py_weekday'
        
        # 복잡한 파생 변수보다 단순한 변수 선호
        complex_patterns = ['interaction', 'poly', 'roll', 'lag', 'cluster', 'network']
        
        var1_complex = any(pattern in var1.lower() for pattern in complex_patterns)
        var2_complex = any(pattern in var2.lower() for pattern in complex_patterns)
        
        if var1_complex and not var2_complex:
            self._log_selection(var2, var1, correlation_val, "단순성 우선: 복잡한 파생변수 제거")
            return var1
        elif var2_complex and not var1_complex:
            self._log_selection(var1, var2, correlation_val, "단순성 우선: 복잡한 파생변수 제거")
            return var2
        
        # 3. 이름 길이 기준 (단순성)
        if len(var1) != len(var2):
            if len(var1) < len(var2):
                self._log_selection(var1, var2, correlation_val, "단순성: 짧은 변수명 선호")
                return var2
            else:
                self._log_selection(var2, var1, correlation_val, "단순성: 짧은 변수명 선호")
                return var1
        
        # 4. 기본값: 첫 번째 변수 제거
        self._log_selection("무작위", f"{var1}>{var2}", correlation_val, "기본 규칙")
        return var1
    
    def remove_correlated_features(self, df: pd.DataFrame, 
                                 exclude_columns: List[str] = None) -> Tuple[List[str], Dict[str, Any]]:
        """
        상관관계 기반 변수 제거 (도메인 지식 적용).
        
        Args:
            df: 입력 데이터프레임
            exclude_columns: 분석에서 제외할 컬럼들 (예: target)
            
        Returns:
            (최종_선택된_변수_리스트, 상세_분석_결과)
        """
        
        print(f"\n🔍 도메인 지식 기반 상관관계 분석 (임계값: {self.correlation_threshold})")
        
        # 수치형 변수만 선택
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if exclude_columns:
            numeric_cols = [col for col in numeric_cols if col not in exclude_columns]
        
        print(f"   📊 분석 대상 변수: {len(numeric_cols)}개")
        
        # 상관관계 행렬 계산
        corr_matrix = df[numeric_cols].corr().abs()
        
        # 상삼각행렬에서 높은 상관관계 쌍 찾기
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_remove = set()
        high_corr_pairs = []
        
        for i in range(len(upper.columns)):
            for j in range(len(upper.columns)):
                if pd.notna(upper.iloc[i, j]) and upper.iloc[i, j] > self.correlation_threshold:
                    var1, var2 = upper.columns[i], upper.columns[j]
                    corr_val = upper.iloc[i, j]
                    
                    high_corr_pairs.append({
                        'var1': var1,
                        'var2': var2, 
                        'correlation': corr_val
                    })
                    
                    # 도메인 지식으로 제거할 변수 선택
                    remove_var = self.select_better_variable(var1, var2, corr_val)
                    to_remove.add(remove_var)
        
        # 최종 선택된 변수들
        final_features = [col for col in numeric_cols if col not in to_remove]
        
        analysis_result = {
            'initial_count': len(numeric_cols),
            'removed_count': len(to_remove),
            'final_count': len(final_features),
            'high_corr_pairs': high_corr_pairs,
            'removed_variables': list(to_remove),
            'final_features': final_features,
            'selection_log': self.selection_log,
            'corr_matrix': corr_matrix
        }
        
        print(f"   ✅ 제거된 변수: {len(to_remove)}개")
        print(f"   ✅ 최종 변수: {len(final_features)}개")
        print(f"   📋 높은 상관관계 쌍: {len(high_corr_pairs)}개")
        
        return final_features, analysis_result
    
    def get_feature_importance_summary(self, analysis_result: Dict[str, Any]) -> pd.DataFrame:
        """변수 선택 과정 요약 정보 생성"""
        
        selection_summary = []
        
        # 선택된 변수들을 카테고리별로 분류
        for feature in analysis_result['final_features']:
            category = self._classify_feature(feature)
            
            selection_summary.append({
                'feature_name': feature,
                'category': category,
                'selection_reason': 'passed_correlation_filter',
                'domain_priority': self._get_domain_priority(feature),
                'is_original_weather': feature in self.domain_priorities['weather_original'],
                'is_derived': self._is_derived_feature(feature)
            })
        
        return pd.DataFrame(selection_summary)
    
    def _log_selection(self, kept_var: str, removed_var: str, 
                      correlation: float, reason: str):
        """변수 선택 과정 로깅"""
        self.selection_log.append({
            'kept_variable': kept_var,
            'removed_variable': removed_var,
            'correlation': correlation,
            'selection_reason': reason,
            'timestamp': pd.Timestamp.now()
        })
    
    def _classify_feature(self, feature: str) -> str:
        """변수를 카테고리별로 분류"""
        for category, variables in self.domain_priorities.items():
            if feature in variables:
                return category
        
        # 패턴 기반 분류
        if any(pattern in feature.lower() for pattern in ['interaction', 'poly']):
            return 'interaction_features'
        elif any(pattern in feature.lower() for pattern in ['roll', 'lag']):
            return 'temporal_features'
        elif any(pattern in feature.lower() for pattern in ['cluster', 'network']):
            return 'spatial_features'
        elif feature.startswith('pca_'):
            return 'pca_features'
        else:
            return 'other_features'
    
    def _get_domain_priority(self, feature: str) -> int:
        """변수의 도메인 우선순위 반환 (1=최고)"""
        priority_order = [
            'weather_original', 'target', 'temporal_basic', 
            'derived_important', 'population', 'location'
        ]
        
        for i, category in enumerate(priority_order):
            if feature in self.domain_priorities.get(category, []):
                return i + 1
        
        return 10  # 기타
    
    def _is_derived_feature(self, feature: str) -> bool:
        """파생 변수 여부 확인"""
        derived_patterns = [
            'avg', 'range', 'interaction', 'poly', 'roll', 'lag', 
            'cluster', 'network', 'pca_', 'sin', 'cos'
        ]
        return any(pattern in feature.lower() for pattern in derived_patterns)
    
    def save_analysis_results(self, analysis_result: Dict[str, Any], 
                            output_prefix: str = "smart_selection"):
        """분석 결과를 CSV 파일로 저장"""
        
        # 1. 높은 상관관계 쌍
        if analysis_result['high_corr_pairs']:
            corr_df = pd.DataFrame(analysis_result['high_corr_pairs'])
            corr_df = corr_df.sort_values('correlation', ascending=False)
            corr_df.to_csv(f'{output_prefix}_high_correlations.csv', 
                          index=False, encoding='utf-8')
        
        # 2. 제거된 변수 상세 정보
        if self.selection_log:
            selection_df = pd.DataFrame(self.selection_log)
            selection_df.to_csv(f'{output_prefix}_selection_log.csv', 
                               index=False, encoding='utf-8')
        
        # 3. 최종 선택된 변수 요약
        feature_summary = self.get_feature_importance_summary(analysis_result)
        feature_summary.to_csv(f'{output_prefix}_final_features.csv', 
                              index=False, encoding='utf-8')
        
        # 4. 상관관계 행렬
        analysis_result['corr_matrix'].to_csv(f'{output_prefix}_correlation_matrix.csv', 
                                             encoding='utf-8')
        
        print(f"\n💾 도메인 지식 기반 분석 결과 저장:")
        print(f"   - {output_prefix}_high_correlations.csv")
        print(f"   - {output_prefix}_selection_log.csv")  
        print(f"   - {output_prefix}_final_features.csv")
        print(f"   - {output_prefix}_correlation_matrix.csv")
        
        return feature_summary
    
    def validate_selection(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                          final_features: List[str], target_col: str = 'call_count') -> Dict[str, float]:
        """선택된 변수들로 간단한 성능 검증"""
        
        from sklearn.ensemble import RandomForestRegressor
        from lightgbm import LGBMRegressor
        
        X_train = train_df[final_features].fillna(0)
        X_test = test_df[final_features].fillna(0)
        y_train = train_df[target_col]
        y_test = test_df[target_col]
        
        results = {}
        
        # RandomForest로 간단 검증
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        
        results['RandomForest'] = {
            'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
            'R2': r2_score(y_test, rf_pred)
        }
        
        # LightGBM으로 간단 검증
        lgbm_model = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        lgbm_model.fit(X_train, y_train)
        lgbm_pred = lgbm_model.predict(X_test)
        
        results['LightGBM'] = {
            'RMSE': np.sqrt(mean_squared_error(y_test, lgbm_pred)),
            'R2': r2_score(y_test, lgbm_pred)
        }
        
        print(f"\n🎯 도메인 지식 기반 선택 변수 성능 검증:")
        print(f"   RF:  RMSE {results['RandomForest']['RMSE']:.4f}, R² {results['RandomForest']['R2']:.4f}")
        print(f"   LGBM: RMSE {results['LightGBM']['RMSE']:.4f}, R² {results['LightGBM']['R2']:.4f}")
        
        return results 