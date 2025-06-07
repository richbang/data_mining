"""
PCA 피처 변환 모듈 - 차원 축소 및 잠재 패턴 추출

- 60+ 피처들 간 다중공선성 문제 해결 시도
- 고차원 데이터의 노이즈 제거 및 주요 패턴 추출
- 계산 효율성 향상을 위한 차원 축소
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA  # 주성분 분석용
from typing import List, Optional


class PCAHolder:
    """
    **PCA 변수 (차원축소)**
    
    **PCA (Principal Component Analysis) 작동 원리:**
    
    A) 전처리 과정:
       1. 모든 수치형 변수 선택 (call_count 제외)
       2. 무한값(inf) → 0으로 변환
       3. 결측값(NaN) → 0으로 채움
       4. StandardScaler 적용 (평균0, 분산1로 정규화)
       → 변수별 스케일 차이 제거 필수
    
    B) PCA 변환 과정:
       1. 공분산 행렬 계산: X'X / (n-1)
       2. 고유값 분해: eigenvalue, eigenvector 추출
       3. 분산 기여도 순으로 주성분 정렬
       4. 상위 n개 주성분 선택 (기본: 3개)
    
    C) 생성되는 변수:
       - pca_1: 첫 번째 주성분 (가장 큰 분산, ~20-30%)
       - pca_2: 두 번째 주성분 (~10-15%)  
       - pca_3: 세 번째 주성분 (~8-12%)
       → 총 3개 변수로 원본 60+개 변수의 50-60% 분산 설명
    
    D) 주성분의 의미 (예시):
       - PC1: 전반적 기상 강도 (온도+습도+풍속 종합)
       - PC2: 계절성 패턴 (여름 vs 겨울 대비)
       - PC3: 지역적 특성 (도심 vs 외곽 대비)
       → 원본 변수들의 선형 조합으로 잠재 개념 추출
    
    E) PCA의 장단점:
       **장점:**
       - 다중공선성 완전 해결 (주성분은 직교)
       - 노이즈 제거 효과 (작은 분산 성분 제거)
       - 차원 축소로 계산 효율성 증대
       - 시각화 가능 (2-3차원)
       
       **단점:**  
       - 해석 가능성 크게 감소
       - 원본 변수 중요도 파악 어려움
       - 도메인 지식 활용 제한
       - 이 프로젝트에서는 성능 저하 확인
    """
    
    def __init__(self, n_components: int = 5):
        """
        Initialize PCAHolder.
        
        Args:
            n_components: Number of PCA components to keep
        """
        self.n_components = n_components
        self.pca: Optional[PCA] = None
        self.feature_columns: Optional[List[str]] = None

    def fit(self, df: pd.DataFrame, feature_columns: List[str]) -> None:
        """
        Fit PCA on the specified columns of the dataframe.
        
        Args:
            df: Training dataframe
            feature_columns: List of column names to use for PCA
        """
        self.feature_columns = feature_columns
        
        # Clean the data: replace inf values and fill NaN
        safe_df = df[feature_columns].replace([np.inf, -np.inf], 0).fillna(0)
        
        # Initialize and fit PCA
        self.pca = PCA(n_components=self.n_components, random_state=42)
        self.pca.fit(safe_df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform dataframe using fitted PCA.
        
        Args:
            df: Dataframe to transform
            
        Returns:
            Dataframe with additional PCA features
        """
        if self.pca is None or self.feature_columns is None:
            return df
        
        df_result = df.copy()
        
        # Clean the data: replace inf values and fill NaN
        safe_df = df_result[self.feature_columns].replace([np.inf, -np.inf], 0).fillna(0)
        
        # Transform using fitted PCA
        pca_features = self.pca.transform(safe_df)
        
        # Add PCA features to dataframe
        for i in range(self.n_components):
            df_result[f'pca_{i+1}'] = pca_features[:, i]
        
        return df_result

    def get_explained_variance_ratio(self) -> Optional[np.ndarray]:
        """
        Get explained variance ratio for each component.
        
        Returns:
            Array of explained variance ratios or None if PCA not fitted
        """
        if self.pca is None:
            return None
        return self.pca.explained_variance_ratio_

    def get_cumulative_variance_ratio(self) -> Optional[np.ndarray]:
        """
        Get cumulative explained variance ratio.
        
        Returns:
            Array of cumulative explained variance ratios or None if PCA not fitted
        """
        if self.pca is None:
            return None
        return np.cumsum(self.pca.explained_variance_ratio_)

    def get_component_importance(self, feature_names: List[str] = None) -> Optional[pd.DataFrame]:
        """
        Get feature importance for each PCA component.
        
        Args:
            feature_names: Names of original features
            
        Returns:
            DataFrame with feature importance per component or None if PCA not fitted
        """
        if self.pca is None:
            return None
            
        if feature_names is None:
            feature_names = self.feature_columns
            
        if feature_names is None:
            return None
        
        # Create DataFrame with component loadings
        components_df = pd.DataFrame(
            self.pca.components_.T,
            columns=[f'PC{i+1}' for i in range(self.n_components)],
            index=feature_names
        )
        
        return components_df 