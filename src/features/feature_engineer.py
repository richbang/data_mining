"""Feature engineering module for creating temporal, weather, and interaction features."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from typing import List


class FeatureEngineer:
    """
    Feature engineering class for creating various features:
    
    **5. 파생/시계열 변수:**
    - 기본 시간 피처: 연도, 월, 일, 요일, 주차
    - 계절성 피처: 계절, 월 구간, 주말 여부
    - 주기적 피처: sin/cos 변환으로 순환성 표현
    - 지연 피처: 1, 3, 7일 전 데이터
    - 롤링 피처: 3, 7, 14일 이동 평균/표준편차/최대값
    
    **6. 상호작용/이상치/클러스터/네트워크 변수:**
    - 날씨 상호작용: 온도×습도, 바람×비 등 조합
    - 체감온도: 여름(열지수), 겨울(체감온도) 계산
    - 불쾌지수: 온도와 습도 기반 계산
    - 극값 탐지: 95%, 5% 분위수 기반 이상치 표시
    - 공간 거리: 도심/해안 거리 계산
    - 네트워크 피처: 인근 지역 평균 신고수
    
    **7. PCA 변수 (선택적):**
    - 수치형 피처들의 주성분 분석
    - 차원 축소 및 잠재 패턴 추출
    - 팀원 논의에 따라 사용/미사용 선택 가능
    """
    
    def __init__(self):
        """Initialize FeatureEngineer."""
        self.created_features = set()  # 중복 생성 방지용

    def add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic temporal features from TM column.
        
        Args:
            df: Input dataframe with TM column
            
        Returns:
            Dataframe with additional temporal features
        """
        df = df.copy()
        
        # Parse date from TM column (YYYY-MM-DD format)
        if 'date' not in df.columns:
            df['date'] = pd.to_datetime(df['TM'].astype(str), errors='coerce')
        
        # Basic date features (중복 생성 방지)
        if 'year' not in df.columns:
            df['year'] = df['date'].dt.year
        if 'month' not in df.columns:
            df['month'] = df['date'].dt.month
        if 'day' not in df.columns:
            df['day'] = df['date'].dt.day
        if 'day_of_year' not in df.columns:
            df['day_of_year'] = df['date'].dt.dayofyear
        if 'week_of_year' not in df.columns:
            df['week_of_year'] = df['date'].dt.isocalendar().week
            
        # weekday: 중복 방지 및 표준화 처리
        if 'weekday' in df.columns and 'py_weekday' in df.columns:
            # 둘 다 있는 경우: py_weekday 제거하고 표준 weekday 사용
            print("⚠️  중복 변수 발견: weekday와 py_weekday 모두 존재")
            print(f"   - weekday 샘플: {df['weekday'].iloc[0]} (사용)")
            print(f"   - py_weekday 샘플: {df['py_weekday'].iloc[0]} (제거)")
            df = df.drop(columns=['py_weekday'])
            print("✅ py_weekday 제거 완료 - weekday로 통일")
        elif 'weekday' not in df.columns and 'py_weekday' not in df.columns:
            # 둘 다 없으면 새로 생성
            df['weekday'] = df['date'].dt.weekday
            print("✅ 표준 weekday 생성 (월=0, 일=6)")
        elif 'py_weekday' in df.columns and 'weekday' not in df.columns:
            # py_weekday만 있으면 weekday로 이름 변경
            df['weekday'] = df['py_weekday']
            df = df.drop(columns=['py_weekday'])
            print("✅ py_weekday를 weekday로 이름 변경")
        
        if 'is_weekend' not in df.columns:
            weekday_col = 'weekday' if 'weekday' in df.columns else 'py_weekday'
            if weekday_col in df.columns:
                df['is_weekend'] = df[weekday_col].isin([5, 6]).astype(int)
        
        # Season mapping
        df['season'] = df['month'].map({
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'autumn', 10: 'autumn', 11: 'autumn',
            12: 'winter', 1: 'winter', 2: 'winter'
        })
        
        # Month period (early, mid, late)
        df['month_period'] = pd.cut(
            df['day'], 
            bins=[0, 10, 20, 31], 
            labels=['early', 'mid', 'late']
        )
        
        # Cyclic features for capturing seasonality
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        return df

    def add_apparent_temperature(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate apparent temperature based on humidity and wind.
        Uses different formulas for summer and winter.
        
        Args:
            df: Dataframe with temperature, humidity, and wind features
            
        Returns:
            Dataframe with apparent temperature feature
        """
        df = df.copy()
        df['apparent_temp'] = np.nan
        
        # Summer apparent temperature (heat index)
        summer = df['month'].isin([5, 6, 7, 8, 9])
        if summer.any() and all(col in df.columns for col in ['hm_min', 'hm_max', 'ta_max']):
            rh = (df.loc[summer, 'hm_min'] + df.loc[summer, 'hm_max']) / 2
            ta = df.loc[summer, 'ta_max']
            
            # Wet bulb temperature calculation
            tw = (ta * np.arctan(0.151977 * (rh + 8.313659) ** 0.5) + 
                  np.arctan(ta + rh) - np.arctan(rh - 1.67633) + 
                  0.00391838 * (rh ** 1.5) * np.arctan(0.023101 * rh) - 4.686035)
            
            # Apparent temperature formula
            df.loc[summer, 'apparent_temp'] = (
                -0.2442 + 0.55399 * tw + 0.45535 * ta - 
                0.0022 * tw ** 2 + 0.00278 * tw * ta + 3.0
            )
        
        # Winter apparent temperature (wind chill)
        winter = ~summer
        if winter.any() and all(col in df.columns for col in ['ws_max', 'ta_max']):
            ta = df.loc[winter, 'ta_max']
            v = df.loc[winter, 'ws_max'] * 3.6  # Convert to km/h
            
            df.loc[winter, 'apparent_temp'] = (
                13.12 + 0.6215 * ta - 11.37 * v ** 0.16 + 
                0.3965 * ta * v ** 0.16
            )
        
        return df

    def add_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived weather features.
        
        Args:
            df: Dataframe with weather columns
            
        Returns:
            Dataframe with additional weather features
        """
        df = df.copy()
        
        # Temperature features
        if all(col in df.columns for col in ['ta_max', 'ta_min']):
            df['temp_range'] = df['ta_max'] - df['ta_min']
            df['temp_avg'] = (df['ta_max'] + df['ta_min']) / 2
        
        # Humidity features
        if all(col in df.columns for col in ['hm_max', 'hm_min']):
            df['humidity_range'] = df['hm_max'] - df['hm_min']
            df['humidity_avg'] = (df['hm_max'] + df['hm_min']) / 2
        
        # Discomfort index
        if all(col in df.columns for col in ['temp_avg', 'humidity_avg']):
            df['discomfort_index'] = (
                1.8 * df['temp_avg'] - 
                0.55 * (1 - df['humidity_avg']/100) * 
                (1.8 * df['temp_avg'] - 26) + 32
            )
        
        # Rain indicator
        if 'rn_day' in df.columns:
            df['is_rainy'] = (df['rn_day'] > 0).astype(int)
        
        # Typhoon indicator (strong wind)
        if 'ws_max' in df.columns:
            df['is_typhoon'] = (df['ws_max'] >= 17).astype(int)
        
        return df

    def add_lag_rolling_features(self, df: pd.DataFrame, 
                                lag_days: List[int] = [1, 3, 7],
                                windows: List[int] = [3, 7, 14]) -> pd.DataFrame:
        """
        Add lag and rolling window features.
        
        Args:
            df: Input dataframe
            lag_days: List of lag days to create
            windows: List of rolling window sizes
            
        Returns:
            Dataframe with lag and rolling features
        """
        df = df.copy()
        df = df.sort_values(['sub_address', 'TM']).reset_index(drop=True)
        
        # Lag features for call_count
        if 'call_count' in df.columns:
            for lag in lag_days:
                df[f'call_count_lag_{lag}'] = (
                    df.groupby('sub_address')['call_count'].shift(lag)
                )
        
        # Lag features for weather variables
        weather_cols = ['ta_max', 'ta_min', 'rn_day', 'ws_max', 'humidity_avg']
        for col in weather_cols:
            if col in df.columns:
                for lag in [1, 3]:
                    df[f'{col}_lag_{lag}'] = (
                        df.groupby('sub_address')[col].shift(lag)
                    )
        
        # Rolling features for call_count
        if 'call_count' in df.columns:
            for window in windows:
                rolling = df.groupby('sub_address')['call_count'].rolling(
                    window=window, min_periods=1
                )
                
                df[f'call_count_roll_mean_{window}'] = (
                    rolling.mean().reset_index(0, drop=True)
                )
                df[f'call_count_roll_std_{window}'] = (
                    rolling.std().reset_index(0, drop=True)
                )
                df[f'call_count_roll_max_{window}'] = (
                    rolling.max().reset_index(0, drop=True)
                )
        
        return df

    def add_weather_interactions(self, df: pd.DataFrame, 
                                cols: List[str] = None) -> pd.DataFrame:
        """
        Add polynomial interaction features for weather variables.
        
        Args:
            df: Input dataframe
            cols: List of columns to create interactions for
            
        Returns:
            Dataframe with interaction features
        """
        if cols is None:
            cols = ['ta_max', 'ta_min', 'hm_max', 'hm_min', 'ws_max', 'rn_day']
        
        # Filter to existing columns
        use_cols = [c for c in cols if c in df.columns]
        if len(use_cols) < 2:
            return df
        
        df = df.copy()
        
        # Create polynomial features (interactions only, no bias)
        poly = PolynomialFeatures(
            degree=2, 
            interaction_only=True, 
            include_bias=False
        )
        
        # Fit and transform the selected columns
        interaction_matrix = poly.fit_transform(df[use_cols].fillna(0.0))
        feature_names = poly.get_feature_names_out(use_cols)
        
        # Add only the interaction terms (skip original features)
        for i, name in enumerate(feature_names[len(use_cols):], start=len(use_cols)):
            df[f'weather_interaction_{name}'] = interaction_matrix[:, i]
        
        return df

    def add_rolling_correlation(self, df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
        """
        Add rolling correlation between call_count and weather variables.
        
        Args:
            df: Input dataframe
            window: Rolling window size
            
        Returns:
            Dataframe with rolling correlation features
        """
        if not all(col in df.columns for col in ['call_count', 'rn_day']):
            return df
            
        df = df.copy()
        df = df.sort_values(['sub_address', 'TM']).reset_index(drop=True)
        df['rolling_corr_call_rain'] = np.nan
        
        for region in df['sub_address'].unique():
            mask = df['sub_address'] == region
            region_data = df[mask]
            
            if len(region_data) >= window:
                corr_values = (
                    region_data['call_count']
                    .rolling(window, min_periods=window)
                    .corr(region_data['rn_day'])
                )
                df.loc[mask, 'rolling_corr_call_rain'] = corr_values.values
        
        return df

    def engineer_all_features(self, df: pd.DataFrame, 
                             lag_days: List[int] = [1, 3, 7],
                             windows: List[int] = [3, 7, 14],
                             weather_cols: List[str] = None) -> pd.DataFrame:
        """
        Apply all feature engineering steps in sequence.
        
        Args:
            df: Input dataframe
            lag_days: Lag days for lag features
            windows: Window sizes for rolling features
            weather_cols: Weather columns for interactions
            
        Returns:
            Fully engineered dataframe
        """
        df = self.add_basic_features(df)
        df = self.add_apparent_temperature(df)
        df = self.add_weather_features(df)
        df = self.add_lag_rolling_features(df, lag_days, windows)
        df = self.add_weather_interactions(df, weather_cols)
        df = self.add_rolling_correlation(df)
        
        return df 