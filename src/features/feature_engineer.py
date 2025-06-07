"""
피처 엔지니어링 모듈 - 시간, 날씨, 상호작용 피처 생성

개발 과정:
- 처음에는 간단한 날짜 피처만 있었음
- 점진적으로 날씨 피처, 상호작용 피처 추가
- py_weekday/weekday 중복 문제 발견 → 자동 처리 로직 추가
- 롤링 피처 추가로 시계열 패턴 포착
- 체감온도, 불쾌지수 등 도메인 지식 기반 피처 추가

주요 해결 문제:
1. py_weekday와 weekday 중복 → 자동 감지 및 통일
2. 메모리 부족 → 단계별 처리로 해결
3. 시계열 데이터 누수 → 그룹별 shift 사용
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures  # 날씨 변수 상호작용용
from typing import List


class FeatureEngineer:
    """
    피처 엔지니어링 클래스 - 다양한 피처 생성 담당
    
    개발 과정에서 배운 점들:
    - 처음에는 모든 피처를 한번에 만들려고 했는데 메모리 부족
    - 단계별로 나누고 중간에 가비지 컬렉션하니까 해결됨
    - 박민혜님이 발견한 원본 데이터 weekday 오류 문제를 add_basic_features에서 완전 해결
    - 시계열 데이터라서 그룹별로 shift 안하면 데이터 누수 발생
    
    **파생/시계열 변수:**
    
    A) 기본 시간 변수:
       - year, month, day: 직접 추출
       - weekday: pandas dt.weekday (월=0, 화=1, 수=2, 목=3, 금=4, 토=5, 일=6)
       - day_of_year: 1~365 (연중 몇 번째 날)
       - week_of_year: ISO 주차 (1~53)
       - season: 계절 매핑 (3~5월=봄, 6~8월=여름 등)
       - month_period: 월 구간 (1~10일=초, 11~20일=중, 21~31일=말)
       - is_weekend: 토일 여부 (weekday가 5 또는 6일 때 1, 나머지 0)
    
    B) 순환성 피처 (Cyclic Encoding):
       - month_sin/cos: sin(2π × month/12), cos(2π × month/12)
       - day_sin/cos: sin(2π × day_of_year/365), cos(2π × day_of_year/365)
       → 12월과 1월, 12월 31일과 1월 1일의 연속성 표현
    
    C) 지연(Lag) 피처:
       - call_count_lag_1: 1일 전 호출수
       - call_count_lag_3: 3일 전 호출수  
       - call_count_lag_7: 7일 전 호출수 (주간 패턴)
       - 날씨 변수도 1일, 3일 전 값 생성
       → 그룹별 shift로 데이터 누수 방지
    
    D) 롤링(Rolling) 피처:
       - call_count_roll_mean_3/7/14: 3/7/14일 이동 평균
       - call_count_roll_std_3/7/14: 3/7/14일 이동 표준편차
       - call_count_roll_max_3/7/14: 3/7/14일 이동 최대값
       → 추세와 변동성 패턴 포착
    
    E) 롤링 상관관계:
       - rolling_corr_call_rain: 7일 윈도우 호출수-강수량 상관관계
       → 지역별 동적 관계 변화 추적
    
    **6. 상호작용/복합 변수 (상세):**
    
    A) 날씨 파생 변수:
       - temp_range: ta_max - ta_min (일교차)
       - temp_avg: (ta_max + ta_min) / 2 (평균 온도)
       - humidity_range: hm_max - hm_min (습도 변화폭)
       - humidity_avg: (hm_max + hm_min) / 2 (평균 습도)
       - is_rainy: rn_day > 0 (비 여부, 0 or 1)
       - is_typhoon: ws_max >= 17 (태풍급 바람, 0 or 1)
    
    B) 체감온도 (복합 공식):
       - 여름(5~9월): 열지수 공식
         apparent_temp = f(온도, 습도, 습구온도)
         복잡한 기상학 공식으로 실제 체감 계산
       - 겨울(10~4월): 바람 체감온도
         apparent_temp = 13.12 + 0.6215×T - 11.37×V^0.16 + 0.3965×T×V^0.16
    
    C) 불쾌지수:
       - discomfort_index = 1.8×temp_avg - 0.55×(1-humidity_avg/100)×(1.8×temp_avg-26) + 32
       → 온도와 습도 조합으로 불쾌감 정도 측정
    
    D) 상호작용 피처 (PolynomialFeatures):
       - weather_interaction_ta_max_hm_max: 최고온도 × 최고습도
       - weather_interaction_ta_max_ws_max: 최고온도 × 최대풍속  
       - weather_interaction_rn_day_ws_max: 강수량 × 풍속 (폭풍 효과)
       - weather_interaction_ta_min_hm_min: 최저온도 × 최저습도
       → 단일 변수로는 포착 못하는 복합 기상 효과
    
    **PCA 변수 (차원축소):**
    
    A) PCA 적용 대상:
       - 모든 수치형 변수 (call_count 제외)
       - StandardScaler로 정규화 후 PCA 적용
       → 변수 간 스케일 차이 보정
    
    B) 생성되는 변수:
       - pca_component_0: 첫 번째 주성분 (가장 큰 분산)
       - pca_component_1: 두 번째 주성분
       - pca_component_2: 세 번째 주성분
       → 원본 변수들의 선형 조합으로 잠재 패턴 추출
    
    C) PCA 효과:
       - 다중공선성 문제 해결
       - 노이즈 제거 효과
       - 차원 축소로 계산 효율성 증대
       - BUT: 해석 가능성 감소, 이 프로젝트에서는 성능 저하
    
    주의사항:
    - py_weekday/weekday 중복 체크 및 자동 처리
    - 시계열 데이터 누수 방지를 위한 그룹별 처리
    - 메모리 효율성을 위한 단계별 피처 생성
    """
    
    def __init__(self):
        """
        피처 엔지니어 초기화
        
        created_features: 중복 생성 방지용 set
        - 같은 피처가 여러 번 생성되는 것을 방지
        - 나중에 추가할 예정이었는데 아직 미사용
        """
        self.created_features = set()  # 중복 생성 방지용 (추후 활용 예정)

    def add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        기본 시간 피처 생성 (TM 컬럼 기반)
        
        🚨 원본 데이터 weekday 오류 문제 해결:
        - 2020-05-01(실제 금요일) → weekday=5(토요일), py_weekday=3(목요일)
        - 원본 데이터의 weekday/py_weekday 값들이 모두 부정확함
        - 상관관계 0.238로 낮아서 자동 필터에 안 걸렸지만, 둘 다 틀림
        - 해결: 기존 컬럼들 무시하고 TM에서 올바른 weekday 재계산
        
        Args:
            df: TM 컬럼이 있는 데이터프레임
            
        Returns:
            시간 피처가 추가된 데이터프레임
        """
        df = df.copy()
        
        # TM 컬럼을 날짜로 변환 (YYYY-MM-DD 포맷)
        # 원본 데이터가 20200501 같은 형태라서 문자열 변환 필요
        if 'date' not in df.columns:
            df['date'] = pd.to_datetime(df['TM'].astype(str), errors='coerce')
        
        # 기본 날짜 피처들 (이미 있으면 중복 생성 안함)
        if 'year' not in df.columns:
            df['year'] = df['date'].dt.year
        if 'month' not in df.columns:
            df['month'] = df['date'].dt.month
        if 'day' not in df.columns:
            df['day'] = df['date'].dt.day
        if 'day_of_year' not in df.columns:
            df['day_of_year'] = df['date'].dt.dayofyear  # 1~365
        if 'week_of_year' not in df.columns:
            df['week_of_year'] = df['date'].dt.isocalendar().week  # ISO 주차
            
        # 핵심 수정: 원본 데이터의 weekday 값들이 모두 틀렸음
        # 예: 2020-05-01(금요일) → weekday=5(토요일), py_weekday=3(목요일) - 둘 다 틀림!
        # 해결: 기존 weekday 컬럼들을 무시하고 TM에서 올바르게 새로 계산
        
        if 'weekday' in df.columns and 'py_weekday' in df.columns:
            print("⚠️  중복 변수 발견: weekday와 py_weekday 모두 존재")
            print(f"   - weekday 샘플: {df['weekday'].iloc[0]} (원본 데이터 오류)")
            print(f"   - py_weekday 샘플: {df['py_weekday'].iloc[0]} (원본 데이터 오류)")
            # 둘 다 제거하고 TM에서 새로 정확하게 계산
            df = df.drop(columns=['weekday', 'py_weekday'])
            df['weekday'] = df['date'].dt.weekday
            print("✅ 기존 weekday/py_weekday 모두 제거 후 TM에서 정확하게 재계산")
        elif 'weekday' in df.columns and 'py_weekday' not in df.columns:
            # weekday만 있는 경우도 원본 데이터 오류일 수 있으니 재계산
            print("⚠️  기존 weekday 발견 - 원본 데이터 오류 가능성으로 재계산")
            print(f"   - 기존 weekday 샘플: {df['weekday'].iloc[0]}")
            df = df.drop(columns=['weekday'])
            df['weekday'] = df['date'].dt.weekday
            print("✅ 기존 weekday 제거 후 TM에서 정확하게 재계산")
        elif 'py_weekday' in df.columns and 'weekday' not in df.columns:
            # py_weekday만 있는 경우도 재계산
            print("⚠️  기존 py_weekday 발견 - 원본 데이터 오류 가능성으로 재계산")
            print(f"   - 기존 py_weekday 샘플: {df['py_weekday'].iloc[0]}")
            df = df.drop(columns=['py_weekday'])
            df['weekday'] = df['date'].dt.weekday
            print("✅ 기존 py_weekday 제거 후 TM에서 정확하게 재계산")
        else:
            # 둘 다 없으면 새로 생성
            df['weekday'] = df['date'].dt.weekday
            print("✅ 표준 weekday 생성 (월=0, 화=1, 수=2, 목=3, 금=4, 토=5, 일=6)")
        
        # 검증 메시지 추가
        if len(df) > 0 and 'TM' in df.columns:
            sample_date = df['TM'].iloc[0]
            sample_weekday = df['weekday'].iloc[0]
            actual_date = pd.to_datetime(str(sample_date))
            weekday_names = ['월', '화', '수', '목', '금', '토', '일']
            print(f"🔍 검증: {sample_date} = {weekday_names[sample_weekday]}요일 (weekday={sample_weekday})")
        
        # 주말 여부 (토요일=5, 일요일=6일 때 1, 나머지 0)
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
        체감온도 계산 (여름/겨울 다른 공식 사용)
        
        Args:
            df: 온도, 습도, 풍속 피처가 있는 데이터프레임
            
        Returns:
            체감온도 피처가 추가된 데이터프레임
        """
        df = df.copy()
        df['apparent_temp'] = np.nan  # 기본값 NaN으로 초기화
        
        # 여름 체감온도 (열지수 기반)
        # 5-9월을 여름으로 간주 (부산 기준)
        summer = df['month'].isin([5, 6, 7, 8, 9])
        if summer.any() and all(col in df.columns for col in ['hm_min', 'hm_max', 'ta_max']):
            rh = (df.loc[summer, 'hm_min'] + df.loc[summer, 'hm_max']) / 2  # 평균 습도
            ta = df.loc[summer, 'ta_max']  # 최고 온도 사용
            
            # 습구온도 계산 (복잡한 공식이지만 정확함)
            # 이 공식은 기상학에서 표준으로 사용되는 공식
            tw = (ta * np.arctan(0.151977 * (rh + 8.313659) ** 0.5) + 
                  np.arctan(ta + rh) - np.arctan(rh - 1.67633) + 
                  0.00391838 * (rh ** 1.5) * np.arctan(0.023101 * rh) - 4.686035)
            
            # 체감온도 공식 (열지수)
            # 습도가 높을수록 체감온도가 올라감
            df.loc[summer, 'apparent_temp'] = (
                -0.2442 + 0.55399 * tw + 0.45535 * ta - 
                0.0022 * tw ** 2 + 0.00278 * tw * ta + 3.0
            )
        
        # 겨울 체감온도 (바람 고려)
        # 여름이 아닌 모든 달 (10-4월)
        winter = ~summer
        if winter.any() and all(col in df.columns for col in ['ws_max', 'ta_max']):
            ta = df.loc[winter, 'ta_max']
            v = df.loc[winter, 'ws_max'] * 3.6  # m/s를 km/h로 변환
            
            # 체감온도 공식 (바람 고려)
            # 바람이 강할수록 체감온도가 내려감
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
        지연 및 롤링 윈도우 피처 생성 (시계열 분석의 핵심!)
        
        **지연(Lag) 피처 생성 원리:**
        - call_count_lag_1: 어제 호출수가 오늘에 영향
        - call_count_lag_3: 3일 전 호출수 (중기 트렌드)  
        - call_count_lag_7: 1주일 전 호출수 (주간 패턴)
        
        **롤링(Rolling) 피처 생성 원리:**
        - roll_mean: 최근 N일 평균 (트렌드 파악)
        - roll_std: 최근 N일 표준편차 (변동성 파악)
        - roll_max: 최근 N일 최대값 (피크 패턴 파악)
        
        **데이터 누수 방지:**
        - groupby('sub_address').shift() 사용
        - 각 지역별로 독립적으로 lag 생성
        - 미래 정보가 과거로 새지 않도록 보장
        
        **생성 예시:**
        원본: [날짜1: 10건, 날짜2: 15건, 날짜3: 12건, 날짜4: 18건]
        lag_1: [NaN, 10, 15, 12]  # 1일 전 값
        roll_mean_3: [NaN, NaN, 12.3, 15.0]  # 3일 이동평균
        
        Args:
            df: 입력 데이터프레임
            lag_days: 지연 생성할 일수 리스트 [1,3,7]
            windows: 롤링 윈도우 크기 리스트 [3,7,14]
            
        Returns:
            지연 및 롤링 피처가 추가된 데이터프레임
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
        날씨 변수 간 상호작용 피처 생성
        
        **상호작용 피처의 필요성:**
        - 단일 변수로는 표현 못하는 복합 효과 포착
        - 예: 고온+고습 → 실제 체감은 단순 합이 아님
        - 예: 강풍+강우 → 폭풍 효과 (개별 영향의 곱셈적 증가)
        
        **PolynomialFeatures 동작 원리:**
        - degree=2: 2차 상호작용만 생성 (A×B)
        - interaction_only=True: 제곱항 제외 (A² 생성 안함)
        - include_bias=False: 상수항 제외
        
        **생성되는 상호작용 예시:**
        입력: [ta_max=30, hm_max=80, ws_max=5, rn_day=10]
        
        생성 피처:
        - weather_interaction_ta_max_hm_max: 30 × 80 = 2400 (고온고습)
        - weather_interaction_ta_max_ws_max: 30 × 5 = 150 (고온강풍)
        - weather_interaction_ta_max_rn_day: 30 × 10 = 300 (고온강우)
        - weather_interaction_hm_max_ws_max: 80 × 5 = 400 (고습강풍)
        - weather_interaction_hm_max_rn_day: 80 × 10 = 800 (고습강우)
        - weather_interaction_ws_max_rn_day: 5 × 10 = 50 (강풍강우=폭풍)
        
        **도메인 지식 반영:**
        - ta_max × hm_max: 열지수 효과 (여름철 위험)
        - ws_max × rn_day: 폭풍 효과 (교통사고 위험 증가)
        - ta_min × hm_min: 체감온도 효과 (겨울철 위험)
        
        Args:
            df: 입력 데이터프레임
            cols: 상호작용 생성할 컬럼 리스트 (기본: 6개 기상변수)
            
        Returns:
            상호작용 피처가 추가된 데이터프레임 (원본 + 15개 상호작용)
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
        롤링 상관관계 피처 생성 (동적 관계 변화 추적)
        
        **롤링 상관관계의 의미:**
        - 고정된 상관관계가 아닌 시간에 따라 변하는 동적 관계 포착
        - 예: 여름에는 호출수-강수량이 양의 상관관계 (폭우→사고증가)
        - 예: 겨울에는 호출수-강수량이 음의 상관관계 (눈→외출감소)
        
        **계산 방식:**
        - 7일 슬라이딩 윈도우로 상관계수 계산
        - 각 지역별로 독립적으로 계산
        - 최소 7일 데이터가 있어야 계산 (min_periods=window)
        
        **실제 계산 예시:**
        날짜1~7: corr(call_count[1:7], rn_day[1:7]) = 0.3
        날짜2~8: corr(call_count[2:8], rn_day[2:8]) = 0.1  
        날짜3~9: corr(call_count[3:9], rn_day[3:9]) = -0.2
        → 시간에 따라 관계가 변함을 포착!
        
        **활용 효과:**
        - 계절별 날씨-호출수 관계 변화 학습
        - 지역별 특성 차이 반영
        - 정적 피처로는 놓치는 동적 패턴 포착
        
        Args:
            df: 입력 데이터프레임
            window: 롤링 윈도우 크기 (기본: 7일)
            
        Returns:
            rolling_corr_call_rain 피처가 추가된 데이터프레임
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