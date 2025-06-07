"""Variable selection tracking module for transparent feature selection process."""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, List, Any


class VariableTracker:
    """
    변수 선택 과정을 추적하고 분석하는 클래스.
    
    **주요 기능:**
    - 단계별 변수 목록 저장
    - 상관관계 분석 결과 저장  
    - 모델별 변수 중요도 저장
    - 종합 분석 결과 생성
    """
    
    def __init__(self):
        self.initial_variables = {}
        self.correlation_analysis = {}
        self.removed_variables = {}
        self.final_variables = {}
        self.model_importance = {}
        self.selection_summary = {}
    
    def save_initial_variables(self, df: pd.DataFrame, stage: str = "initial"):
        """초기 생성된 모든 변수 저장"""
        all_cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        self.initial_variables[stage] = {
            'stage': stage,
            'all_columns': all_cols,
            'numeric_columns': numeric_cols,
            'total_count': len(all_cols),
            'numeric_count': len(numeric_cols),
            'categorical_columns': [col for col in all_cols if col not in numeric_cols],
            'timestamp': pd.Timestamp.now()
        }
        
        # CSV로 저장
        pd.DataFrame({
            'variable_name': all_cols,
            'data_type': [str(df[col].dtype) for col in all_cols],
            'is_numeric': [col in numeric_cols for col in all_cols],
            'null_count': [df[col].isnull().sum() for col in all_cols],
            'null_percentage': [df[col].isnull().sum() / len(df) * 100 for col in all_cols]
        }).to_csv(f'variables_{stage}_all.csv', index=False, encoding='utf-8')
        
        print(f"✅ {stage} 단계 변수 목록 저장: variables_{stage}_all.csv")
        print(f"   총 {len(all_cols)}개 변수 (수치형: {len(numeric_cols)}개)")
        
    def save_correlation_analysis(self, corr_matrix: pd.DataFrame, threshold: float, 
                                removed_vars: List[str], kept_vars: List[str]):
        """상관관계 분석 결과 저장"""
        
        # 1. 상관관계 행렬 저장
        corr_matrix.to_csv('correlation_matrix_full.csv', encoding='utf-8')
        
        # 2. 높은 상관관계 변수 쌍 찾기 및 저장
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_pairs = []
        
        for i in range(len(upper.columns)):
            for j in range(len(upper.columns)):
                if pd.notna(upper.iloc[i, j]) and upper.iloc[i, j] > threshold:
                    high_corr_pairs.append({
                        'variable_1': upper.columns[i],
                        'variable_2': upper.columns[j],
                        'correlation': upper.iloc[i, j],
                        'above_threshold': True,
                        'threshold_used': threshold
                    })
        
        if high_corr_pairs:
            high_corr_df = pd.DataFrame(high_corr_pairs)
            high_corr_df = high_corr_df.sort_values('correlation', ascending=False)
            high_corr_df.to_csv('high_correlation_pairs.csv', index=False, encoding='utf-8')
        
        # 3. 제거된 변수와 사유 저장
        removal_reasons = []
        for var in removed_vars:
            if high_corr_pairs:  # high_corr_pairs가 비어있지 않을 때만
                # 해당 변수가 제거된 이유 찾기
                related_pairs = [p for p in high_corr_pairs 
                               if p['variable_1'] == var or p['variable_2'] == var]
                
                if related_pairs:
                    max_corr_pair = max(related_pairs, key=lambda x: x['correlation'])
                    other_var = (max_corr_pair['variable_2'] if max_corr_pair['variable_1'] == var 
                               else max_corr_pair['variable_1'])
                    removal_reasons.append({
                        'removed_variable': var,
                        'reason': 'high_correlation',
                        'correlation_with': other_var,
                        'correlation_value': max_corr_pair['correlation'],
                        'kept_variable': other_var if other_var in kept_vars else 'unknown',
                        'threshold': threshold
                    })
        
        if removal_reasons:
            pd.DataFrame(removal_reasons).to_csv('removed_variables_correlation.csv', index=False, encoding='utf-8')
        
        # 4. 최종 선택된 변수 저장
        pd.DataFrame({
            'selected_variable': kept_vars,
            'selection_stage': 'after_correlation_filtering',
            'selection_reason': 'passed_correlation_threshold'
        }).to_csv('selected_variables_after_correlation.csv', index=False, encoding='utf-8')
        
        self.correlation_analysis = {
            'threshold': threshold,
            'total_pairs_above_threshold': len(high_corr_pairs) if high_corr_pairs else 0,
            'removed_count': len(removed_vars),
            'kept_count': len(kept_vars),
            'high_corr_pairs': high_corr_pairs
        }
        
        print(f"✅ 상관관계 분석 결과 저장 완료")
        print(f"   - 전체 상관관계 행렬: correlation_matrix_full.csv")
        print(f"   - 높은 상관관계 쌍: high_correlation_pairs.csv ({len(high_corr_pairs) if high_corr_pairs else 0}개)")
        print(f"   - 제거된 변수: removed_variables_correlation.csv ({len(removed_vars)}개)")
        print(f"   - 선택된 변수: selected_variables_after_correlation.csv ({len(kept_vars)}개)")
    
    def save_model_importance(self, model_name: str, model: Any, feature_names: List[str], 
                            X_test: pd.DataFrame, y_test: pd.Series, predictions: np.ndarray):
        """모델별 변수 중요도와 성능 저장"""
        
        # 1. 기본 성능 지표
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        mae = np.mean(np.abs(y_test - predictions))
        
        # 2. Feature Importance 추출
        if hasattr(model, 'feature_importances_'):
            importance_values = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance_values = np.abs(model.coef_)
        else:
            importance_values = np.zeros(len(feature_names))
        
        # 3. 변수별 상세 정보 생성
        feature_analysis = []
        for i, (feature, importance) in enumerate(zip(feature_names, importance_values)):
            feature_analysis.append({
                'model_name': model_name,
                'feature_name': feature,
                'importance_score': importance,
                'importance_rank': i + 1,
                'importance_percentage': importance / importance_values.sum() * 100 if importance_values.sum() > 0 else 0,
                'cumulative_importance': importance_values[:i+1].sum() / importance_values.sum() * 100 if importance_values.sum() > 0 else 0,
                'is_top_10': i < 10,
                'is_top_20': i < 20
            })
        
        # 중요도 순으로 정렬
        feature_analysis = sorted(feature_analysis, key=lambda x: x['importance_score'], reverse=True)
        
        # 순위 재조정
        for i, item in enumerate(feature_analysis):
            item['importance_rank'] = i + 1
            item['cumulative_importance'] = sum([x['importance_score'] for x in feature_analysis[:i+1]]) / sum([x['importance_score'] for x in feature_analysis]) * 100
        
        # 4. 저장
        feature_df = pd.DataFrame(feature_analysis)
        feature_df.to_csv(f'feature_importance_{model_name.lower()}.csv', index=False, encoding='utf-8')
        
        # 5. 모델 성능 저장
        performance = {
            'model_name': model_name,
            'rmse': rmse,
            'r2_score': r2,
            'mae': mae,
            'n_features': len(feature_names),
            'n_samples': len(y_test),
            'top_feature': feature_analysis[0]['feature_name'] if feature_analysis else 'none',
            'top_importance': feature_analysis[0]['importance_score'] if feature_analysis else 0
        }
        
        self.model_importance[model_name] = {
            'performance': performance,
            'feature_importance': feature_analysis
        }
        
        print(f"✅ {model_name} 모델 분석 저장: feature_importance_{model_name.lower()}.csv")
        print(f"   RMSE: {rmse:.4f}, R²: {r2:.4f}, Top feature: {performance['top_feature']}")
        
        return feature_df
    
    def save_comprehensive_summary(self):
        """전체 변수 선택 과정 종합 요약 저장"""
        
        # 1. 모든 모델 성능 비교
        model_comparison = []
        for model_name, info in self.model_importance.items():
            model_comparison.append(info['performance'])
        
        if model_comparison:
            pd.DataFrame(model_comparison).to_csv('model_performance_comparison.csv', index=False, encoding='utf-8')
        
        # 2. 변수별 종합 중요도 (모든 모델 평균)
        all_features = set()
        for model_name, info in self.model_importance.items():
            for feat_info in info['feature_importance']:
                all_features.add(feat_info['feature_name'])
        
        comprehensive_features = []
        for feature in all_features:
            feature_scores = []
            feature_ranks = []
            appeared_models = []
            
            for model_name, info in self.model_importance.items():
                for feat_info in info['feature_importance']:
                    if feat_info['feature_name'] == feature:
                        feature_scores.append(feat_info['importance_score'])
                        feature_ranks.append(feat_info['importance_rank'])
                        appeared_models.append(model_name)
                        break
            
            if feature_scores:
                comprehensive_features.append({
                    'feature_name': feature,
                    'avg_importance': np.mean(feature_scores),
                    'std_importance': np.std(feature_scores),
                    'avg_rank': np.mean(feature_ranks),
                    'std_rank': np.std(feature_ranks),
                    'appeared_in_models': len(appeared_models),
                    'model_list': ', '.join(appeared_models),
                    'max_importance': np.max(feature_scores),
                    'min_importance': np.min(feature_scores),
                    'is_consistent_top10': all(rank <= 10 for rank in feature_ranks),
                    'is_consistent_top20': all(rank <= 20 for rank in feature_ranks)
                })
        
        # 평균 중요도로 정렬
        comprehensive_features = sorted(comprehensive_features, key=lambda x: x['avg_importance'], reverse=True)
        
        if comprehensive_features:
            pd.DataFrame(comprehensive_features).to_csv('comprehensive_feature_analysis.csv', index=False, encoding='utf-8')
        
        # 3. 전체 요약 통계
        initial_count = 0
        for stage, info in self.initial_variables.items():
            if stage == "after_feature_engineering":
                initial_count = info.get('numeric_count', 0)
                break
        
        summary_stats = {
            'total_initial_variables': initial_count,
            'total_numeric_variables': initial_count,
            'variables_removed_by_correlation': self.correlation_analysis.get('removed_count', 0),
            'final_variables_count': len(all_features),
            'correlation_threshold_used': self.correlation_analysis.get('threshold', 'unknown'),
            'best_model': max(model_comparison, key=lambda x: x['r2_score'])['model_name'] if model_comparison else 'unknown',
            'best_rmse': min(model_comparison, key=lambda x: x['rmse'])['rmse'] if model_comparison else 'unknown',
            'best_r2': max(model_comparison, key=lambda x: x['r2_score'])['r2_score'] if model_comparison else 'unknown'
        }
        
        pd.DataFrame([summary_stats]).to_csv('variable_selection_summary.csv', index=False, encoding='utf-8')
        
        print(f"\n🎯 종합 분석 결과 저장 완료:")
        print(f"   - 모델 성능 비교: model_performance_comparison.csv")
        print(f"   - 종합 변수 분석: comprehensive_feature_analysis.csv") 
        print(f"   - 전체 요약: variable_selection_summary.csv")
        print(f"   - 최고 성능 모델: {summary_stats['best_model']} (R²: {summary_stats['best_r2']:.3f})") 