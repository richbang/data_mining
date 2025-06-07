"""
통계적 피처 홀더 - 그룹 통계, 이상치 탐지, 클러스터링, 네트워크 피처

- 단순 개별 변수로는 포착 못하는 공간적/시간적 패턴 필요
- 지역별 특성, 이상기후, 공간 클러스터, 네트워크 효과 모델링
- train 데이터로 학습한 통계를 test에 일관되게 적용 필요
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans  # 공간/시간 클러스터링용
from scipy.spatial.distance import pdist, squareform  # 거리 계산용
from typing import Dict, Any, Optional


class StatsHolder:
    """
    통계 피처 생성 및 관리 클래스
    
    **상호작용/이상치/클러스터/네트워크 변수 상세:**
    
    A) 그룹 통계 피처:
       - gu_total_calls: 구별 총 호출수 (지역 규모)
       - gu_avg_calls: 구별 평균 호출수 (지역 위험도)
       → 해운대구는 관광지라 호출수 많음, 중구는 상업지역 등
    
    B) 이상치/극값 탐지 피처:
       - ta_max_is_extreme_q95: 최고온도 상위 5% 여부 (폭염)
       - ta_max_is_extreme_q05: 최고온도 하위 5% 여부 (한파)
       - ws_max_is_extreme_q95: 풍속 상위 5% 여부 (강풍/태풍)
       - rn_day_is_extreme_q95: 강수량 상위 5% 여부 (폭우)
       → 기상청 특보 기준과 유사한 극값 임계값 학습
    
    C) 상호작용 복합 피처:
       - hot_and_rainy: 고온(상위25%) + 비 동시 발생 (0/1)
       - windy_and_rainy: 강풍(상위25%) + 비 동시 발생 (폭풍, 0/1)
       → 단일 기상요소보다 복합 기상재해가 더 위험
    
    D) 거리 기반 피처:
       - distance_from_center: 부산 중심부에서 거리
         계산식: sqrt((x-129.0756416)² + (y-35.1795543)²)
       - distance_from_coast: 해안선에서 거리 (y좌표 - 34.8902691)
       → 도심/해안 접근성이 응급상황 발생/대응에 영향
    
    E) 시간적 클러스터링:
       - temporal_cluster: 지역별 시간 패턴 기반 클러스터링
         특징: call_mean, call_std, peak_month, peak_weekday
         예: [주거지역 패턴, 상업지역 패턴, 관광지 패턴, 공업지역 패턴, 기타]
       → 유사한 시간적 호출 패턴을 가진 지역들을 그룹화
    
    F) 공간적 클러스터링 (계절별):
       - spring_spatial_cluster: 봄철 공간 클러스터 (0~4)
       - summer_spatial_cluster: 여름철 공간 클러스터
       - autumn_spatial_cluster: 가을철 공간 클러스터  
       - winter_spatial_cluster: 겨울철 공간 클러스터
       → 계절별로 유사한 공간적 특성을 가진 지역 그룹화
       → 여름엔 해수욕장 주변이 한 클러스터, 겨울엔 산악지역이 한 클러스터
    
    G) 네트워크 피처:
       - nearby_avg_calls: 인근 3개 지역 평균 호출수
         알고리즘: 유클리드 거리 기반 최근접 3개 지역 선택
         계산: 동일 날짜 인근 지역들의 호출수 평균
       → 공간적 상관관계 (한 지역 사고 많으면 인근도 많음)
       → 지역간 응급상황 전파 효과 모델링
    
    **학습/적용 과정:**
    1. fit(): train 데이터로 모든 통계/클러스터 학습
    2. transform(): 학습된 통계를 train/test에 일관 적용
    → Data leakage 방지: test 정보가 train 학습에 사용 안됨
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
        """
        네트워크 피처 생성 - 공간적 상관관계 모델링
        
        **네트워크 피처 생성 알고리즘:**
        
        1. 거리 기반 인근 지역 매핑:
           - 모든 지역 간 유클리드 거리 계산
           - 각 지역별로 가장 가까운 3개 지역 선택
           - 예: 해운대구 → [수영구, 동래구, 기장군]
        
        2. 동일 시점 인근 지역 호출수 집계:
           - 같은 날짜(TM)의 인근 3개 지역 호출수 평균 계산
           - 예: 2023-01-01 해운대 → 같은 날 수영+동래+기장 평균
        
        3. 생성되는 피처:
           - nearby_avg_calls: 인근 지역 평균 호출수
        
        **공간적 상관관계의 의미:**
        - 대형 사고/재해는 인근 지역에도 영향
        - 교통체증, 기상재해 등의 공간적 확산
        - 응급실 포화 시 인근 병원으로 이송
        - 지역 축제/행사의 주변 지역 파급효과
        
        **계산 예시:**
        부산역(중구) 인근: [동구, 서구, 영도구]
        2023-07-15 데이터:
        - 동구: 15건, 서구: 12건, 영도구: 18건
        - nearby_avg_calls = (15+12+18)/3 = 15.0
        → 부산역 지역의 네트워크 피처 = 15.0
        
        **Data Leakage 방지:**
        - 인근 지역 매핑은 train 데이터로만 학습
        - test 적용 시에도 동일한 매핑 관계 사용
        - 미래 정보 사용 안함 (같은 시점 데이터만)
        """
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