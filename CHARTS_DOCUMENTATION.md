# 📊 119 응급신고 예측 시스템 - 차트 및 시각화 문서

이 문서는 119 응급신고 예측 시스템에서 자동 생성된 모든 차트와 시각화 자료에 대한 상세한 설명을 제공합니다.

## 📁 파일 구조

```
outputs/
├── 📊 모델 성능 비교
│   └── model_comparison.png
├── 📈 피처 중요도 차트 (4개)
│   ├── RandomForest_feature_importance.png
│   ├── LightGBM_feature_importance.png
│   ├── XGBoost_feature_importance.png
│   └── CatBoost_feature_importance.png
└── plots/ (모델별 상세 분석)
    ├── 🎯 예측 vs 실제값 (5개)
    │   ├── RandomForest_predictions_vs_actual.png
    │   ├── LightGBM_predictions_vs_actual.png
    │   ├── XGBoost_predictions_vs_actual.png
    │   ├── CatBoost_predictions_vs_actual.png
    │   └── Ensemble_RMSE_Weighted_predictions_vs_actual.png
    └── 📉 잔차 분석 (5개)
        ├── RandomForest_residuals.png
        ├── LightGBM_residuals.png
        ├── XGBoost_residuals.png
        ├── CatBoost_residuals.png
        └── Ensemble_RMSE_Weighted_residuals.png
```

---

## 🏆 1. 모델 성능 비교 차트

### 📄 `model_comparison.png`
**목적**: 전체 모델(4개 개별 모델 + 1개 앙상블)의 성능을 한눈에 비교

**포함 지표**:
- **RMSE** (Root Mean Square Error) - 낮을수록 좋음
- **R²** (결정계수) - 높을수록 좋음  
- **MAE** (Mean Absolute Error) - 낮을수록 좋음

**해석 방법**:
- 막대 높이로 각 모델의 성능 비교
- RMSE/MAE: 짧은 막대 = 더 좋은 성능
- R²: 긴 막대 = 더 좋은 성능
- **결과**: LightGBM이 가장 우수한 성능 (RMSE: 0.734, R²: 0.725)

---

## 📈 2. 피처 중요도 차트 (Feature Importance)

각 모델이 예측에 사용한 변수들의 중요도를 시각화합니다.

### 📄 `RandomForest_feature_importance.png`
- **알고리즘**: 트리 기반 importance (불순도 감소량)
- **특징**: 안정적이고 해석하기 쉬움
- **Top 변수**: call_count_roll_mean_3 (3일 이동평균이 가장 중요)

### 📄 `LightGBM_feature_importance.png`  
- **알고리즘**: Gradient Boosting 기반 gain importance
- **특징**: 빠르고 정확한 중요도 계산
- **Top 변수**: call_count_roll_std_3 (3일 이동 표준편차가 중요)

### 📄 `XGBoost_feature_importance.png`
- **알고리즘**: 정교한 Gradient Boosting importance  
- **특징**: 복잡한 상호작용 포착
- **Top 변수**: typhoon (태풍 지시자가 중요)

### 📄 `CatBoost_feature_importance.png`
- **알고리즘**: 범주형 변수 특화 importance
- **특징**: 오버피팅에 강함
- **Top 변수**: call_count_roll_mean_3 (RandomForest와 동일)

**공통 패턴**:
- 모든 모델에서 **시계열 피처**가 상위권 (lag, rolling)
- **날씨 상호작용 피처**들이 중요하게 평가됨
- **계절성 피처** (month, day 등)가 일관되게 중요

---

## 🎯 3. 예측 vs 실제값 차트 (Predictions vs Actual)

모델의 예측 정확도를 직관적으로 확인할 수 있는 산점도입니다.

### 📊 차트 구성 요소
- **X축**: 실제 응급신고 건수 (y_true)
- **Y축**: 모델 예측 건수 (y_pred)  
- **빨간 대각선**: 완벽한 예측선 (y=x)
- **R² 점수**: 우상단에 표시된 결정계수

### 📄 개별 모델 차트들

#### `RandomForest_predictions_vs_actual.png`
- **R² ≈ 0.698**: 69.8%의 변동성 설명
- **패턴**: 대부분의 점들이 대각선 근처에 분포
- **특징**: 안정적이지만 약간의 과소예측 경향

#### `LightGBM_predictions_vs_actual.png` 🏆
- **R² ≈ 0.725**: 72.5%의 변동성 설명 (최고 성능)
- **패턴**: 가장 대각선에 가까운 분포
- **특징**: 극값에서도 비교적 정확한 예측

#### `XGBoost_predictions_vs_actual.png`  
- **R² ≈ 0.666**: 66.6%의 변동성 설명
- **패턴**: 다른 모델보다 산포가 큼
- **특징**: 일부 극값에서 예측 오차 발생

#### `CatBoost_predictions_vs_actual.png`
- **R² ≈ 0.703**: 70.3%의 변동성 설명  
- **패턴**: RandomForest와 유사한 분포
- **특징**: 중간값 예측에 강점

#### `Ensemble_RMSE_Weighted_predictions_vs_actual.png`
- **R² ≈ 0.721**: 72.1%의 변동성 설명
- **패턴**: 개별 모델의 약점을 보완한 안정적 분포
- **특징**: 극값에서의 오차가 줄어듦

---

## 📉 4. 잔차 분석 차트 (Residual Analysis)

모델의 예측 오차 패턴을 상세히 분석하는 4가지 서브플롯으로 구성됩니다.

### 📊 차트 구성 (2×2 레이아웃)

#### **좌상단**: Residuals vs Predicted
- **목적**: 예측값에 따른 오차 패턴 확인
- **이상적**: 빨간 수평선(y=0) 주위에 무작위 분포
- **문제 신호**: 특정 패턴이나 곡선 형태

#### **우상단**: Distribution of Residuals  
- **목적**: 오차의 분포가 정규분포인지 확인
- **이상적**: 0을 중심으로 한 종 모양 분포
- **문제 신호**: 비대칭이나 다중 봉우리

#### **좌하단**: Q-Q Plot
- **목적**: 잔차가 정규분포를 따르는지 통계적 검증
- **이상적**: 대각선 위에 점들이 일직선으로 배열
- **문제 신호**: S자 곡선이나 극값에서 이탈

#### **우하단**: Absolute Residuals vs Predicted
- **목적**: 예측값에 따른 오차 크기(분산) 변화 확인  
- **이상적**: 수평선 형태 (등분산성)
- **문제 신호**: 우상향/우하향 기울기 (이분산성)

### 📄 개별 모델 잔차 분석

#### `RandomForest_residuals.png`
- **특징**: 안정적인 잔차 분포, 등분산성 양호
- **장점**: 극값에서의 큰 오차 없음
- **단점**: 약간의 우편향 (과소예측 경향)

#### `LightGBM_residuals.png` 🏆
- **특징**: 가장 이상적인 잔차 패턴
- **장점**: 정규분포에 가까움, 등분산성 우수
- **성능**: 다른 모델 대비 가장 작은 잔차 분산

#### `XGBoost_residuals.png`
- **특징**: 약간의 이분산성 존재
- **단점**: 높은 예측값에서 오차 증가 경향
- **패턴**: 일부 극값에서 큰 잔차 발생

#### `CatBoost_residuals.png`
- **특징**: RandomForest와 유사한 패턴
- **장점**: 대체로 안정적인 분포
- **단점**: 일부 구간에서 체계적 편향

#### `Ensemble_RMSE_Weighted_residuals.png`
- **특징**: 개별 모델의 장점을 결합한 최적화된 패턴
- **장점**: 극값에서의 오차 감소, 더 나은 등분산성
- **성과**: 개별 모델 대비 더 안정적인 예측

---

## 🔍 5. 차트 해석 가이드

### ✅ **좋은 모델의 신호**
1. **Predictions vs Actual**: 점들이 대각선에 가깝게 분포
2. **높은 R² 값**: 0.7 이상이면 양호한 성능
3. **Residuals**: 0 주위에 무작위 분포, 패턴 없음
4. **정규분포**: 잔차가 종 모양 분포를 따름
5. **등분산성**: 예측값에 관계없이 일정한 오차 크기

### ⚠️ **주의할 점들**
1. **체계적 편향**: 잔차가 일정한 방향으로 치우침
2. **이분산성**: 예측값이 커질수록 오차도 커짐
3. **비정규분포**: 잔차 분포가 심하게 비대칭
4. **극값 문제**: 매우 높거나 낮은 값에서 큰 오차

### 🏆 **이번 결과의 우수성**
- **LightGBM**: 모든 지표에서 가장 우수한 성능
- **앙상블**: 개별 모델의 약점을 효과적으로 보완
- **전체적**: R² 0.7 이상으로 실용적 수준의 예측 정확도
- **안정성**: 잔차 분석에서 큰 문제점 발견되지 않음