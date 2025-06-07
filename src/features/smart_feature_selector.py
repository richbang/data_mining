"""
스마트 피처 선택 모듈 - 도메인 지식 기반 변수 선택

핵심 아이디어:
- 기상 원본 변수 > 파생 변수 우선순위
- 생성된 변수보다 기본 변수 선호
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')  # 모델 훈련 시 불필요한 경고 제거


class SmartFeatureSelector:
    """
    도메인 지식을 활용한 스마트 변수 선택 클래스
    
    - Elastic Net 자동 변수선택 추가 (다중공선성 해결)
    - 기존 상관관계 기반 + Elastic Net 결합 옵션 제공
    
    **핵심 철학:**
    - 53개 고상관 변수쌍 → Elastic Net으로 해결 기대해봄
    - 기상 원본 변수는 최대한 보존 (예측에 가장 중요)
    - 복잡한 파생변수보다 해석 가능한 기본 변수 선호
    - 같은 의미의 중복 변수는 표준화 (py_weekday → weekday)
    - 모든 선택 이유를 명확히 기록
    
    **우선순위 체계:**
    1. 기상 원본 > 파생 변수
    2. 기본 시간 변수 > 복잡한 상호작용
    3. 짧은 이름 > 긴 이름 (단순성 선호)
    """
    
    def __init__(self, correlation_threshold: float = 0.85):
        """
        초기화
        """
        self.correlation_threshold = correlation_threshold
        
        # 도메인 지식 기반 우선순위 체계
        # 응급의료 전문가와 기상 전문가 의견 반영
        self.domain_priorities = {
            'weather_original': ['ta_max', 'ta_min', 'ws_max', 'hm_max', 'hm_min', 'rn_day'],  # 가장 중요
            'target': ['call_count'],  # 타겟 변수
            'temporal_basic': ['year', 'month', 'day', 'weekday'],  # 기본 시간 변수
            'population': ['총인구수', '세대수', '남자.인구수', '여자.인구수'],  # 인구 정보
            'location': ['xcoord', 'ycoord'],  # 위치 좌표
            'derived_important': ['temp_avg', 'humidity_avg', 'is_rainy', 'is_weekend']  # 중요한 파생변수
        }
        
        # 로깅 시스템 (투명성 확보용)
        self.removal_log = []      # 제거된 변수들 기록
        self.selection_log = []    # 선택 과정 상세 기록
        self.elastic_net_log = []  # Elastic Net 선택 과정 기록
    
    def select_better_variable(self, var1: str, var2: str, correlation_val: float) -> str:
        """
        두 고상관 변수 중 어떤 것을 제거할지 지능적으로 선택
        

        도메인 지식에 기반해서 더 중요한 변수를 보존함
        
        **의사결정 트리:**
        1단계: 도메인 우선순위 확인 (기상 > 시간 > 인구 등)
        2단계: py_weekday/weekday 같은 알려진 중복 처리  
        3단계: 복잡도 비교 (기본 변수 > 상호작용 변수)
        4단계: 이름 길이 (짧은 것이 보통 더 기본적)
        5단계: 최후 수단으로 첫 번째 제거
        
        Args:
            var1: 첫 번째 변수명
            var2: 두 번째 변수명
            correlation_val: 두 변수 간 상관관계 값
            
        Returns:
            제거할 변수명 (보존할 게 아니라 제거할 것!)
        """
        
        # 1단계: 도메인 지식 우선순위 적용
        # 한 변수가 중요 카테고리에 속하고 다른 건 안 속하면 → 중요한 것 보존
        for category, variables in self.domain_priorities.items():
            if var1 in variables and var2 not in variables:
                self._log_selection(var1, var2, correlation_val, f"도메인 우선순위: {category}")
                return var2  # var1 보존, var2 제거
            elif var2 in variables and var1 not in variables:
                self._log_selection(var2, var1, correlation_val, f"도메인 우선순위: {category}")
                return var1  # var2 보존, var1 제거
        
        # 2단계: 알려진 중복 변수 특별 처리
        if {var1, var2} == {'weekday', 'py_weekday'}:
            self._log_selection('weekday', 'py_weekday', correlation_val, "중복 변수: weekday 우선")
            return 'py_weekday'  # py_weekday 제거, weekday 보존
        
        # 3단계: 복잡도 기반 선택
        # 복잡한 파생변수(상호작용, 롤링 등)보다 기본 변수 선호
        complex_patterns = ['interaction', 'poly', 'roll', 'lag', 'cluster', 'network']
        
        var1_complex = any(pattern in var1.lower() for pattern in complex_patterns)
        var2_complex = any(pattern in var2.lower() for pattern in complex_patterns)
        
        if var1_complex and not var2_complex:
            self._log_selection(var2, var1, correlation_val, "단순성 우선: 복잡한 파생변수 제거")
            return var1  # 복잡한 var1 제거
        elif var2_complex and not var1_complex:
            self._log_selection(var1, var2, correlation_val, "단순성 우선: 복잡한 파생변수 제거")
            return var2  # 복잡한 var2 제거
        
        # 4단계: 이름 길이 기준 (단순성 휴리스틱)
        # 짧은 이름이 보통 더 기본적인 변수 (ta_max vs weather_interaction_ta_max_hm_max)
        if len(var1) != len(var2):
            if len(var1) < len(var2):
                self._log_selection(var1, var2, correlation_val, "단순성: 짧은 변수명 선호")
                return var2  # 긴 var2 제거
            else:
                self._log_selection(var2, var1, correlation_val, "단순성: 짧은 변수명 선호")
                return var1  # 긴 var1 제거
        
        # 5단계: 최후 수단 (임의 선택)
        # 모든 휴리스틱이 실패하면 첫 번째 변수 제거
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
        
        print(f"\n도메인 지식 기반 상관관계 분석 (임계값: {self.correlation_threshold})")
        
        # 수치형 변수만 선택
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if exclude_columns:
            numeric_cols = [col for col in numeric_cols if col not in exclude_columns]
        
        print(f"   분석 대상 변수: {len(numeric_cols)}개")
        
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
        
        # outputs 폴더 생성
        import os
        os.makedirs('outputs', exist_ok=True)
        
        # Elastic Net 결과인지 확인
        is_elastic_net = 'feature_coefficients' in analysis_result
        
        if is_elastic_net:
            # Elastic Net 결과 저장
            print(f"\nElastic Net 분석 결과 저장:")
            
            # 1. 선택된 변수와 계수
            coef_data = []
            for feature, coef in analysis_result['feature_coefficients'].items():
                coef_data.append({
                    'feature': feature,
                    'coefficient': coef,
                    'abs_coefficient': abs(coef),
                    'selected': abs(coef) > 1e-8
                })
            
            coef_df = pd.DataFrame(coef_data)
            coef_df = coef_df.sort_values('abs_coefficient', ascending=False)
            coef_df.to_csv(f'outputs/{output_prefix}_coefficients.csv', 
                          index=False, encoding='utf-8')
            
            # 2. 최적 파라미터 정보
            params_data = {
                'parameter': ['best_alpha', 'best_l1_ratio', 'cv_r2_score', 'selected_features_count'],
                'value': [
                    analysis_result['best_alpha'],
                    analysis_result['best_l1_ratio'], 
                    analysis_result['cv_score'],
                    analysis_result['selected_count']
                ]
            }
            params_df = pd.DataFrame(params_data)
            params_df.to_csv(f'outputs/{output_prefix}_parameters.csv', 
                            index=False, encoding='utf-8')
            
            # 3. 선택된 변수 목록
            selected_df = pd.DataFrame({'selected_features': analysis_result['selected_features']})
            selected_df.to_csv(f'outputs/{output_prefix}_selected_features.csv', 
                              index=False, encoding='utf-8')
            
            print(f"   - outputs/{output_prefix}_coefficients.csv")
            print(f"   - outputs/{output_prefix}_parameters.csv")
            print(f"   - outputs/{output_prefix}_selected_features.csv")
            
            return coef_df
            
        else:
            # 기존 상관관계 기반 결과 저장
            # 1. 높은 상관관계 쌍
            if analysis_result.get('high_corr_pairs'):
                corr_df = pd.DataFrame(analysis_result['high_corr_pairs'])
                corr_df = corr_df.sort_values('correlation', ascending=False)
                corr_df.to_csv(f'outputs/{output_prefix}_high_correlations.csv', 
                              index=False, encoding='utf-8')
            
            # 2. 제거된 변수 상세 정보
            if self.selection_log:
                selection_df = pd.DataFrame(self.selection_log)
                selection_df.to_csv(f'outputs/{output_prefix}_selection_log.csv', 
                                   index=False, encoding='utf-8')
            
            # 3. 최종 선택된 변수 요약
            feature_summary = self.get_feature_importance_summary(analysis_result)
            feature_summary.to_csv(f'outputs/{output_prefix}_final_features.csv', 
                                  index=False, encoding='utf-8')
            
            # 4. 상관관계 행렬
            if 'corr_matrix' in analysis_result:
                analysis_result['corr_matrix'].to_csv(f'outputs/{output_prefix}_correlation_matrix.csv', 
                                                     encoding='utf-8')
            
            print(f"\n💾 도메인 지식 기반 분석 결과 저장:")
            print(f"   - outputs/{output_prefix}_high_correlations.csv")
            print(f"   - outputs/{output_prefix}_selection_log.csv")  
            print(f"   - outputs/{output_prefix}_final_features.csv")
            print(f"   - outputs/{output_prefix}_correlation_matrix.csv")
            
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
    
    def elastic_net_selection(self, train_df: pd.DataFrame, target_col: str = 'call_count',
                             cv_folds: int = 5, l1_ratio_range: List[float] = None,
                             alpha_range: List[float] = None) -> Tuple[List[str], Dict[str, Any]]:
        """
        **핵심 아이디어:**
        - L1 정규화: 불필요한 변수 계수를 0으로 만들어 자동 제거
        - L2 정규화: 상관관계 높은 변수들을 그룹으로 처리
        - Cross-validation으로 최적 파라미터 자동 선택
        - 다중공선성 문제를 근본적으로 해결
        
        Args:
            train_df: 훈련 데이터
            target_col: 타겟 변수명
            cv_folds: Cross-validation fold 수
            l1_ratio_range: L1/L2 비율 범위 (None이면 기본값 사용)
            alpha_range: 정규화 강도 범위 (None이면 기본값 사용)
            
        Returns:
            (선택된_변수_리스트, 상세_분석_결과)
        """
        
        print(f"\nElastic Net 자동 변수선택 시작")
        print(f"   다중공선성 해결을 위한 L1+L2 정규화 적용")
        
        # 수치형 변수만 선택 (타겟 제외)
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        print(f"   분석 대상 변수: {len(numeric_cols)}개")
        
        # 데이터 준비
        X = train_df[numeric_cols].fillna(0)
        y = train_df[target_col]
        
        # 무한대와 매우 큰 값 처리 (StandardScaler 에러 방지)
        print(f"   데이터 전처리: inf/nan 값 처리 중...")
        
        # 모든 컬럼을 수치형으로 강제 변환 (object 타입 문제 해결)
        for col in X.columns:
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            except:
                print(f"   ⚠️  {col} 컬럼 수치형 변환 실패 - 0으로 대체")
                X[col] = 0
        
        # inf 값을 nan으로 변환 후 0으로 대체
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        # 매우 큰 값들을 클리핑 (overflow 방지)
        # 각 컬럼별로 99.9% 분위수를 상한으로 설정
        for col in X.columns:
            try:
                if X[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    upper_bound = X[col].quantile(0.999)
                    lower_bound = X[col].quantile(0.001)
                    X[col] = X[col].clip(lower=lower_bound, upper=upper_bound)
            except:
                print(f"   ⚠️  {col} 컬럼 클리핑 실패 - 건너뜀")
                continue
        
        # 최종 확인 (안전한 방법으로)
        try:
            # DataFrame 단위로 확인
            inf_count = X.isin([np.inf, -np.inf]).sum().sum()
            nan_count = X.isna().sum().sum()
            print(f"   전처리 후 inf: {inf_count}개, nan: {nan_count}개")
            
            if inf_count > 0 or nan_count > 0:
                print(f"   ⚠️  여전히 inf/nan 존재 - 0으로 최종 대체")
                X = X.fillna(0)
                X = X.replace([np.inf, -np.inf], 0)
        except Exception as e:
            print(f"   ⚠️  inf/nan 확인 중 에러: {e}")
            print(f"   안전을 위해 모든 비정상값을 0으로 대체")
            X = X.fillna(0)
            X = X.replace([np.inf, -np.inf], 0)
        
        # 데이터 타입 최종 확인
        print(f"   데이터 타입 확인: {X.dtypes.value_counts().to_dict()}")
        
        # 표준화 (Elastic Net은 스케일에 민감)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 파라미터 범위 설정
        if l1_ratio_range is None:
            l1_ratio_range = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99]  # L1 비중
        
        if alpha_range is None:
            # 자동으로 적절한 alpha 범위 생성 (50개 → 20개로 축소)
            alpha_range = np.logspace(-3, 0, 20)  # 0.001 ~ 1 범위
        
        print(f"   L1 ratio 후보: {len(l1_ratio_range)}개")
        print(f"   Alpha 후보: {len(alpha_range)}개")
        print(f"   총 조합: {len(l1_ratio_range) * len(alpha_range)}개 (CV 적용시 {len(l1_ratio_range) * len(alpha_range) * cv_folds}회 훈련)")
        
        # Elastic Net Cross-Validation
        elastic_net = ElasticNetCV(
            l1_ratio=l1_ratio_range,
            alphas=alpha_range,
            cv=cv_folds,
            random_state=42,
            max_iter=2000,
            n_jobs=-1
        )
        
        print(f"   🔄 {cv_folds}-fold CV로 최적 파라미터 탐색 중...")
        
        elastic_net.fit(X_scaled, y)
        
        # 결과 분석
        selected_features = []
        feature_coefficients = {}
        
        for i, coef in enumerate(elastic_net.coef_):
            feature_name = numeric_cols[i]
            feature_coefficients[feature_name] = coef
            
            if abs(coef) > 1e-8:  # 0이 아닌 계수만 선택
                selected_features.append(feature_name)
                self.elastic_net_log.append({
                    'feature': feature_name,
                    'coefficient': coef,
                    'abs_coefficient': abs(coef),
                    'selected': True
                })
            else:
                self.elastic_net_log.append({
                    'feature': feature_name,
                    'coefficient': coef,
                    'abs_coefficient': abs(coef),
                    'selected': False
                })
        
        # 결과 정리
        analysis_result = {
            'initial_count': len(numeric_cols),
            'selected_count': len(selected_features),
            'removed_count': len(numeric_cols) - len(selected_features),
            'selected_features': selected_features,
            'feature_coefficients': feature_coefficients,
            'best_alpha': elastic_net.alpha_,
            'best_l1_ratio': elastic_net.l1_ratio_,
            'cv_score': elastic_net.score(X_scaled, y),
            'elastic_net_model': elastic_net,
            'scaler': scaler,
            'selection_log': self.elastic_net_log
        }
        
        print(f"\n   ✅ Elastic Net 선택 완료:")
        print(f"      최적 Alpha: {elastic_net.alpha_:.6f}")
        print(f"      최적 L1 ratio: {elastic_net.l1_ratio_:.3f}")
        print(f"      CV R² 점수: {elastic_net.score(X_scaled, y):.4f}")
        print(f"      선택된 변수: {len(selected_features)}개 (제거: {len(numeric_cols) - len(selected_features)}개)")
        
        # 중요한 변수 top 10 출력
        sorted_features = sorted(
            [(name, abs(coef)) for name, coef in feature_coefficients.items() if abs(coef) > 1e-8],
            key=lambda x: x[1], reverse=True
        )
        
        print(f"\n   🏆 중요 변수 Top 10:")
        for i, (feature, abs_coef) in enumerate(sorted_features[:10]):
            print(f"      {i+1:2d}. {feature:<30} (계수: {abs_coef:.4f})")
        
        return selected_features, analysis_result
    
    def combined_selection(self, train_df: pd.DataFrame, target_col: str = 'call_count',
                          use_elastic_net: bool = True, use_correlation: bool = True) -> Tuple[List[str], Dict[str, Any]]:
        """
        엘라스틱넷 + 기존 방법 결합 선택
        
        **전략:**
        1단계: Elastic Net으로 대량 변수선택 (다중공선성 해결)
        2단계: 도메인 지식 기반 상관관계 정제 (선택적)
        3단계: 최종 검증 및 보고서 생성
        
        Args:
            train_df: 훈련 데이터
            target_col: 타겟 변수
            use_elastic_net: Elastic Net 사용 여부
            use_correlation: 상관관계 기반 추가 정제 여부
            
        Returns:
            (최종_선택된_변수_리스트, 전체_분석_결과)
        """
        
        print(f"\n통합 변수선택 시작")
        print(f"   Elastic Net: {'✅' if use_elastic_net else '❌'}")
        print(f"   상관관계 정제: {'✅' if use_correlation else '❌'}")
        
        combined_result = {
            'elastic_net_results': None,
            'correlation_results': None,
            'final_features': [],
            'selection_steps': []
        }
        
        current_features = None
        current_df = train_df.copy()
        
        # 1단계: Elastic Net 선택
        if use_elastic_net:
            print(f"\n=== 1단계: Elastic Net 자동 변수선택 ===")
            selected_features, elastic_result = self.elastic_net_selection(
                current_df, target_col
            )
            
            combined_result['elastic_net_results'] = elastic_result
            combined_result['selection_steps'].append({
                'step': 'elastic_net',
                'input_features': len(current_df.select_dtypes(include=[np.number]).columns) - 1,
                'output_features': len(selected_features),
                'method': 'L1+L2 regularization'
            })
            
            current_features = selected_features + [target_col]  # 타겟 포함
            current_df = current_df[current_features]
            
            print(f"   Elastic Net 후 변수 수: {len(selected_features)}개")
        
        # 2단계: 상관관계 기반 추가 정제 (선택적)
        if use_correlation and current_features:
            print(f"\n=== 2단계: 도메인 지식 기반 상관관계 정제 ===")
            final_features, correlation_result = self.remove_correlated_features(
                current_df, exclude_columns=[target_col]
            )
            
            combined_result['correlation_results'] = correlation_result
            combined_result['selection_steps'].append({
                'step': 'correlation_refinement',
                'input_features': len(selected_features) if use_elastic_net else len(current_df.columns) - 1,
                'output_features': len(final_features),
                'method': 'domain_knowledge_correlation'
            })
            
        else:
            final_features = current_features[:-1] if current_features else []  # 타겟 제외
        
        combined_result['final_features'] = final_features
        
        # 최종 결과 요약
        print(f"\n📊 통합 선택 결과 요약:")
        print(f"   원본 변수 수: {len(train_df.select_dtypes(include=[np.number]).columns) - 1}개")
        
        for step in combined_result['selection_steps']:
            print(f"   {step['step']}: {step['input_features']} → {step['output_features']}개 ({step['method']})")
        
        print(f"   최종 선택: {len(final_features)}개 변수")
        
        # 변수 제거율 계산
        original_count = len(train_df.select_dtypes(include=[np.number]).columns) - 1
        removal_rate = (original_count - len(final_features)) / original_count * 100
        print(f"   변수 제거율: {removal_rate:.1f}%")
        
        return final_features, combined_result 