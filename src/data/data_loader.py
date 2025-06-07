"""Data loading and preprocessing module."""

import pandas as pd
import numpy as np
from typing import Tuple, List
import time


class DataLoader:
    """
    Data loader for 119 call prediction project.
    Handles data loading, train/test splitting, and basic preprocessing.
    """
    
    def __init__(self, data_file: str, encoding: str = 'utf-8'):
        """
        Initialize DataLoader.
        
        Args:
            data_file: Path to the data file
            encoding: File encoding
        """
        self.data_file = data_file
        self.encoding = encoding
        self.data: pd.DataFrame = None

    def load_data(self, verbose: bool = True) -> pd.DataFrame:
        """
        Load data from file.
        
        Args:
            verbose: Whether to print loading information
            
        Returns:
            Loaded dataframe
        """
        if verbose:
            print(f"데이터 로딩 시작: {self.data_file}")
            start_time = time.time()
        
        try:
            self.data = pd.read_csv(self.data_file, encoding=self.encoding)
            
            if verbose:
                elapsed = time.time() - start_time
                print(f"데이터 로딩 완료. 소요 시간: {elapsed:.2f}초")
                print(f"데이터 크기: {self.data.shape}")
                
        except Exception as e:
            print(f"데이터 로딩 중 오류 발생: {e}")
            raise
        
        return self.data

    def split_train_test(self, train_years: List[int], test_years: List[int],
                        verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets based on years.
        
        Args:
            train_years: List of years for training
            test_years: List of years for testing
            verbose: Whether to print split information
            
        Returns:
            Tuple of (train_df, test_df)
        """
        if self.data is None:
            raise ValueError("데이터를 먼저 로드해야 합니다. load_data()를 호출하세요.")
        
        if verbose:
            start_time = time.time()
            print(f"train/test 분할 시작 (train: {train_years}, test: {test_years})")
        
        # Ensure year column exists
        if 'year' not in self.data.columns:
            if 'TM' in self.data.columns:
                self.data['date'] = pd.to_datetime(self.data['TM'].astype(str), errors='coerce')
                self.data['year'] = self.data['date'].dt.year
            else:
                raise ValueError("year 컬럼 또는 TM 컬럼이 데이터에 없습니다.")
        
        # Split data
        train_df = self.data[self.data['year'].isin(train_years)].copy()
        test_df = self.data[self.data['year'].isin(test_years)].copy()
        
        if verbose:
            elapsed = time.time() - start_time
            print(f"train/test 분할 완료. 소요 시간: {elapsed:.2f}초")
            print(f"Train 크기: {train_df.shape}, Test 크기: {test_df.shape}")
        
        return train_df, test_df

    def split_by_typhoon(self, df: pd.DataFrame, 
                        verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split dataframe by typhoon status.
        
        Args:
            df: Input dataframe
            verbose: Whether to print split information
            
        Returns:
            Tuple of (normal_df, typhoon_df)
        """
        if 'typhoon' not in df.columns:
            raise ValueError("typhoon 컬럼이 데이터에 없습니다.")
        
        if verbose:
            start_time = time.time()
            print("태풍/비태풍 분할 시작")
        
        normal_df = df[df['typhoon'] == 0].copy()
        typhoon_df = df[df['typhoon'] == 1].copy()
        
        if verbose:
            elapsed = time.time() - start_time
            print(f"태풍/비태풍 분할 완료. 소요 시간: {elapsed:.2f}초")
            print(f"비태풍 크기: {normal_df.shape}, 태풍 크기: {typhoon_df.shape}")
        
        return normal_df, typhoon_df

    def remove_highly_correlated_features(self, df: pd.DataFrame, 
                                        threshold: float = 0.85,
                                        target_col: str = 'call_count',
                                        verbose: bool = True,
                                        save_analysis: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove highly correlated features with domain knowledge consideration.
        
        Args:
            df: Input dataframe
            threshold: Correlation threshold for feature removal (lowered to 0.85)
            target_col: Target column to exclude from correlation analysis
            verbose: Whether to print information
            save_analysis: Whether to save correlation analysis results
            
        Returns:
            Tuple of (cleaned_df, removed_features)
        """
        if verbose:
            start_time = time.time()
            print(f"상관관계 기반 피처 제거 시작 (threshold: {threshold})")
        
        # Select numeric columns excluding target
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr().abs()
        
        # Find highly correlated pairs with transparency
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Get high correlation pairs for transparency
        high_corr_pairs = []
        for i in range(len(upper_triangle.columns)):
            for j in range(len(upper_triangle.columns)):
                if pd.notna(upper_triangle.iloc[i, j]) and upper_triangle.iloc[i, j] > threshold:
                    high_corr_pairs.append({
                        'feature_1': upper_triangle.columns[i],
                        'feature_2': upper_triangle.columns[j],
                        'correlation': upper_triangle.iloc[i, j]
                    })
        
        # Sort by correlation for transparency
        high_corr_pairs = sorted(high_corr_pairs, key=lambda x: x['correlation'], reverse=True)
        
        if verbose and high_corr_pairs:
            print(f"\n높은 상관관계 변수 쌍들 (상관계수 > {threshold}):")
            for i, pair in enumerate(high_corr_pairs[:15]):  # Show top 15
                print(f"  {pair['feature_1']} - {pair['feature_2']}: {pair['correlation']:.3f}")
            if len(high_corr_pairs) > 15:
                print(f"  ... 총 {len(high_corr_pairs)}개 쌍")
        
        # Domain knowledge based selection
        def select_better_feature(feat1: str, feat2: str, corr_val: float) -> str:
            """Select which feature to keep based on domain knowledge."""
            
            # Priority 1: Keep original weather variables over derived ones
            original_weather = ['ta_max', 'ta_min', 'ws_max', 'hm_max', 'hm_min', 'rn_day']
            if feat1 in original_weather and feat2 not in original_weather:
                return feat2  # Remove derived feature
            elif feat2 in original_weather and feat1 not in original_weather:
                return feat1  # Remove derived feature
            
            # Priority 2: Keep simpler variable names (usually original)
            if feat1 == 'weekday' and 'py_weekday' in feat2:
                return feat2  # Remove py_weekday
            elif feat2 == 'weekday' and 'py_weekday' in feat1:
                return feat1  # Remove py_weekday
            
            # Priority 3: Keep call_count related features
            call_related = ['call_count_lag', 'call_count_roll', 'nearby_avg_calls']
            feat1_is_call = any(term in feat1 for term in call_related)
            feat2_is_call = any(term in feat2 for term in call_related)
            
            if feat1_is_call and not feat2_is_call:
                return feat2
            elif feat2_is_call and not feat1_is_call:
                return feat1
            
            # Priority 4: Prefer shorter, simpler names
            if len(feat1) < len(feat2):
                return feat2
            elif len(feat2) < len(feat1):
                return feat1
            
            # Priority 5: Keep first alphabetically as tiebreaker
            return feat2 if feat1 < feat2 else feat1
        
        # Apply domain knowledge selection
        to_remove = set()
        removal_reasons = []
        
        for pair in high_corr_pairs:
            feat1, feat2, corr_val = pair['feature_1'], pair['feature_2'], pair['correlation']
            
            # Skip if one already marked for removal
            if feat1 in to_remove or feat2 in to_remove:
                continue
                
            remove_feat = select_better_feature(feat1, feat2, corr_val)
            to_remove.add(remove_feat)
            keep_feat = feat1 if remove_feat == feat2 else feat2
            
            removal_reasons.append({
                'removed_feature': remove_feat,
                'kept_feature': keep_feat,
                'correlation': corr_val,
                'reason': 'domain_knowledge_selection'
            })
        
        # Create cleaned dataframe
        features_to_keep = [col for col in numeric_cols if col not in to_remove]
        if target_col in df.columns:
            features_to_keep.append(target_col)
        
        cleaned_df = df[features_to_keep].copy()
        
        # Save analysis results if requested
        if save_analysis:
            self._save_correlation_analysis(
                high_corr_pairs, removal_reasons, threshold, 
                list(to_remove), features_to_keep, corr_matrix
            )
        
        if verbose:
            elapsed = time.time() - start_time
            print(f"상관관계 기반 피처 제거 완료. 소요 시간: {elapsed:.2f}초")
            print(f"제거된 피처 수: {len(to_remove)}")
            print(f"남은 피처 수: {len(features_to_keep) - (1 if target_col in df.columns else 0)}")
            print(f"투명성 확보: 높은 상관관계 쌍 {len(high_corr_pairs)}개 식별됨")
        
        return cleaned_df, list(to_remove)
    
    def _save_correlation_analysis(self, high_corr_pairs: List[dict], 
                                 removal_reasons: List[dict], threshold: float,
                                 removed_features: List[str], kept_features: List[str],
                                 corr_matrix: pd.DataFrame) -> None:
        """Save correlation analysis results for transparency."""
        try:
            import os
            os.makedirs('outputs', exist_ok=True)
            
            # Save high correlation pairs
            if high_corr_pairs:
                pairs_df = pd.DataFrame(high_corr_pairs)
                pairs_df.to_csv('outputs/high_correlation_pairs.csv', 
                              index=False, encoding='utf-8')
            
            # Save removal reasons
            if removal_reasons:
                reasons_df = pd.DataFrame(removal_reasons)
                reasons_df.to_csv('outputs/feature_removal_reasons.csv', 
                                index=False, encoding='utf-8')
            
            # Save full correlation matrix
            corr_matrix.to_csv('outputs/correlation_matrix_full.csv', encoding='utf-8')
            
            # Save selected features
            selected_df = pd.DataFrame({
                'selected_features': kept_features[:-1] if kept_features[-1] == 'call_count' else kept_features,
                'selection_reason': 'passed_correlation_filtering'
            })
            selected_df.to_csv('outputs/selected_features.csv', 
                             index=False, encoding='utf-8')
            
            print(f"✅ 상관관계 분석 결과를 outputs/ 디렉토리에 저장했습니다.")
            
        except Exception as e:
            print(f"상관관계 분석 결과 저장 중 오류: {e}")

    def get_feature_columns(self, df: pd.DataFrame, 
                           exclude_cols: List[str] = None) -> List[str]:
        """
        Get list of feature columns (numeric columns excluding specified columns).
        
        Args:
            df: Input dataframe
            exclude_cols: Columns to exclude from features
            
        Returns:
            List of feature column names
        """
        if exclude_cols is None:
            exclude_cols = ['call_count', 'TM', 'date']
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        return feature_cols

    def clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean feature values (replace inf, fill NaN).
        
        Args:
            df: Input dataframe
            
        Returns:
            Cleaned dataframe
        """
        cleaned_df = df.copy()
        
        # Replace infinite values with 0
        cleaned_df = cleaned_df.replace([np.inf, -np.inf], 0)
        
        # Fill NaN values with 0 for numeric columns
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(0)
        
        return cleaned_df

    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """
        Get summary statistics of the dataset.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist()
        }
        
        if 'call_count' in df.columns:
            summary['call_count_stats'] = df['call_count'].describe().to_dict()
        
        return summary 