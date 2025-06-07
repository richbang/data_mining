"""Statistical feature holder for groupby statistics, outlier detection, and clustering."""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from typing import Dict, Any, Optional


class StatsHolder:
    """
    Holds and manages statistical features including:
    - Group statistics by district
    - Outlier detection thresholds
    - Spatial and temporal clustering
    - Network features (nearby regions)
    """
    
    def __init__(self, city_coords: tuple = (35.1795543, 129.0756416), 
                 coast_lat: float = 34.8902691):
        """
        Initialize StatsHolder.
        
        Args:
            city_coords: Latitude and longitude of city center (Busan)
            coast_lat: Latitude of coast line
        """
        self.gu_stats: Optional[pd.DataFrame] = None
        self.q: Dict[str, Dict[str, float]] = {}
        self.city = city_coords
        self.coast_lat = coast_lat
        self.temporal_stats: Optional[pd.DataFrame] = None
        self.spatial_clusters: Dict[str, KMeans] = {}
        self.temporal_clusters: Optional[pd.DataFrame] = None
        self.nearby_map: Optional[Dict[str, list]] = None

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit statistical features on training data.
        
        Args:
            df: Training dataframe with features
        """
        self._fit_group_stats(df)
        self._fit_quantiles(df)
        self._fit_temporal_clusters(df)
        self._fit_spatial_clusters(df)
        self._fit_nearby_map(df)

    def _fit_group_stats(self, df: pd.DataFrame) -> None:
        """Fit group statistics by district."""
        if 'address_gu' in df.columns and 'call_count' in df.columns:
            self.gu_stats = (
                df.groupby('address_gu')['call_count']
                .agg(['sum', 'mean'])
                .rename(columns={'sum': 'gu_total_calls', 'mean': 'gu_avg_calls'})
            )

    def _fit_quantiles(self, df: pd.DataFrame) -> None:
        """Fit quantiles for outlier/extreme detection."""
        for c in ['ta_max', 'ws_max', 'rn_day']:
            if c in df.columns:
                self.q[c] = {
                    'q05': df[c].quantile(0.05),
                    'q75': df[c].quantile(0.75),
                    'q95': df[c].quantile(0.95)
                }

    def _fit_temporal_clusters(self, df: pd.DataFrame) -> None:
        """Fit temporal clusters by sub_address."""
        if 'sub_address' not in df.columns or 'call_count' not in df.columns:
            return
            
        temporal_features = df.groupby('sub_address').agg({
            'call_count': ['mean', 'std', 'min', 'max'],
            'month': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
            'weekday': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
        }).reset_index()
        
        temporal_features.columns = [
            'sub_address', 'call_mean', 'call_std', 'call_min', 
            'call_max', 'peak_month', 'peak_weekday'
        ]
        
        if len(temporal_features) > 3:
            kmeans = KMeans(
                n_clusters=min(5, len(temporal_features)//5), 
                random_state=42, 
                n_init=10
            )
            cluster_features = temporal_features[
                ['call_mean', 'call_std', 'peak_month', 'peak_weekday']
            ].fillna(0)
            temporal_features['temporal_cluster'] = kmeans.fit_predict(cluster_features)
            self.temporal_clusters = temporal_features[['sub_address', 'temporal_cluster']]

    def _fit_spatial_clusters(self, df: pd.DataFrame) -> None:
        """Fit spatial clusters by season."""
        if not {'season', 'xcoord', 'ycoord'}.issubset(df.columns):
            return
            
        for season in df['season'].unique():
            if pd.notna(season):
                season_data = df[df['season'] == season]
                coords = season_data[['xcoord', 'ycoord']].values
                
                if len(coords) > 5:
                    kmeans = KMeans(
                        n_clusters=min(5, len(coords)//10), 
                        random_state=42,
                        n_init=10
                    )
                    kmeans.fit(coords)
                    self.spatial_clusters[season] = kmeans

    def _fit_nearby_map(self, df: pd.DataFrame) -> None:
        """Fit nearby region map for network features."""
        if not {'sub_address', 'xcoord', 'ycoord'}.issubset(df.columns):
            return
            
        unique_regions = df[['sub_address', 'xcoord', 'ycoord']].drop_duplicates()
        coords = unique_regions[['xcoord', 'ycoord']].values
        
        if len(coords) < 2:
            return
            
        distances = squareform(pdist(coords))
        nearby_map = {}
        
        for i, region in enumerate(unique_regions['sub_address']):
            nearest_indices = np.argsort(distances[i])[1:4]  # Top 3 nearest (excluding self)
            nearest_regions = unique_regions.iloc[nearest_indices]['sub_address'].tolist()
            nearby_map[region] = nearest_regions
            
        self.nearby_map = nearby_map

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform dataframe with fitted statistical features.
        
        Args:
            df: Dataframe to transform
            
        Returns:
            Transformed dataframe with additional features
        """
        res = df.copy()
        
        res = self._add_group_stats(res)
        res = self._add_extreme_features(res)
        res = self._add_interaction_features(res)
        res = self._add_distance_features(res)
        res = self._add_cluster_features(res)
        res = self._add_network_features(res)
        
        return res

    def _add_group_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add group statistics features."""
        if self.gu_stats is not None and 'address_gu' in df.columns:
            df = df.merge(self.gu_stats, left_on='address_gu', right_index=True, how='left')
        return df

    def _add_extreme_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add extreme value features based on quantiles."""
        for c, q in self.q.items():
            if c in df.columns:
                df[f'{c}_is_extreme_q95'] = (df[c] > q['q95']).astype(int)
                df[f'{c}_is_extreme_q05'] = (df[c] < q['q05']).astype(int)
        return df

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add weather interaction features."""
        # Hot and rainy
        if {'ta_max', 'is_rainy'}.issubset(df.columns) and 'ta_max' in self.q:
            df['hot_and_rainy'] = (
                (df['ta_max'] > self.q['ta_max']['q75']) & 
                (df['is_rainy'] == 1)
            ).astype(int)
        
        # Windy and rainy
        if {'ws_max', 'is_rainy'}.issubset(df.columns) and 'ws_max' in self.q:
            df['windy_and_rainy'] = (
                (df['ws_max'] > self.q['ws_max']['q75']) & 
                (df['is_rainy'] == 1)
            ).astype(int)
            
        return df

    def _add_distance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add distance-based features."""
        if {'xcoord', 'ycoord'}.issubset(df.columns):
            df['distance_from_center'] = np.sqrt(
                (df['xcoord'] - self.city[1]) ** 2 + 
                (df['ycoord'] - self.city[0]) ** 2
            )
            df['distance_from_coast'] = df['ycoord'] - self.coast_lat
        return df

    def _add_cluster_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cluster-based features."""
        # Temporal cluster
        if self.temporal_clusters is not None and 'sub_address' in df.columns:
            df = df.merge(self.temporal_clusters, on='sub_address', how='left')
        
        # Spatial cluster
        if self.spatial_clusters and {'season', 'xcoord', 'ycoord'}.issubset(df.columns):
            for season, kmeans in self.spatial_clusters.items():
                mask = df['season'] == season
                coords = df.loc[mask, ['xcoord', 'ycoord']].values
                
                if len(coords) > 0:
                    clusters = kmeans.predict(coords)
                    df.loc[mask, f'{season}_spatial_cluster'] = clusters
                    
        return df

    def _add_network_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add network features based on nearby regions."""
        if (self.nearby_map is not None and 
            {'sub_address', 'TM', 'call_count'}.issubset(df.columns)):
            
            df['nearby_avg_calls'] = np.nan
            
            for idx, row in df.iterrows():
                region = row['sub_address']
                tm = row['TM']
                
                if region in self.nearby_map:
                    nearby = self.nearby_map[region]
                    mask = (df['TM'] == tm) & (df['sub_address'].isin(nearby))
                    val = df.loc[mask, 'call_count'].mean()
                    df.at[idx, 'nearby_avg_calls'] = val
                    
        return df 