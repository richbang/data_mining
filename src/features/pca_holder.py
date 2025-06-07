"""PCA feature transformation module."""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from typing import List, Optional


class PCAHolder:
    """
    PCA feature transformation holder.
    Fits PCA on training data and transforms both train and test data.
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