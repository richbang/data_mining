# 119 Call Prediction Project

부산시 119 신고 건수 예측을 위한 머신러닝 프로젝트입니다.

## 프로젝트 구조

```
119_call_prediction/
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py          # 프로젝트 설정
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_loader.py       # 데이터 로딩 및 전처리
│   ├── features/
│   │   ├── __init__.py
│   │   ├── feature_engineer.py  # 피처 엔지니어링 (weekday 오류 해결 포함)
│   │   ├── stats_holder.py      # 통계적 피처 생성
│   │   ├── pca_holder.py        # PCA 차원 축소
│   │   └── smart_feature_selector.py  # 스마트 변수선택 (Elastic Net 포함)
│   ├── models/
│   │   ├── __init__.py
│   │   └── model_trainer.py     # 모델 훈련 및 평가
│   └── utils/
│       ├── __init__.py
│       └── evaluation.py        # 평가 유틸리티
├── outputs/                     # 분석 결과 저장 폴더
├── main.py                      # 메인 파이프라인
├── requirements.txt             # 의존성 패키지
└── README.md                    # 프로젝트 설명서
```

## 주요 기능

### 1. 데이터 처리
- CSV 파일 로딩 및 인코딩 처리
- 연도별 train/test 분할 (2020-2022 train, 2023 test)
- **원본 데이터 품질 검증**: weekday 오류 발견 및 자동 수정
- 태풍/비태풍 데이터 분리 분석

### 2. 피처 엔지니어링 (95개 변수 생성)
- **시간적 피처**: 날짜, 계절, 요일, 주기적 피처 (sin/cos)
- **날씨 피처**: 체감온도, 불쾌지수, 날씨 상호작용
- **지연/롤링 피처**: 과거 데이터 기반 lag 및 rolling 통계
- **통계적 피처**: 구별 통계, 극값 탐지, 클러스터링
- **공간적 피처**: 도심/해안 거리, 인근 지역 네트워크 피처
- **PCA 피처**: 차원 축소를 통한 잠재 피처 (선택적)

### 3. 스마트 변수선택
**박승정님 제안 반영**: 다중공선성 근본 해결

#### **문제점 발견**
- 기존 상관관계 기반 제거: **53개 고상관 변수쌍** 발견
- 단순 threshold로는 다중공선성 해결 한계

#### **Elastic Net 자동 변수선택** 🎯
- **L1 정규화**: 불필요한 변수 계수를 0으로 만들어 자동 제거
- **L2 정규화**: 상관관계 높은 변수들을 그룹으로 지능적 처리
- **Cross-validation**: 최적 파라미터 자동 탐색
- **도메인 지식 결합**: 기상 변수 우선순위 등 전문 지식 반영

#### **통합 변수선택 옵션**
```python
# 1. Elastic Net만 사용 (권장)
use_elastic_net=True, use_correlation=False

# 2. 상관관계 기반만 사용 (기존 방식)
use_elastic_net=False, use_correlation=True

# 3. 두 방법 결합 (실험적)
use_elastic_net=True, use_correlation=True
```

### 4. 머신러닝 모델
- **RandomForest**: 안정적인 베이스라인 모델
- **LightGBM**: 빠르고 효율적인 그래디언트 부스팅 **최고 성능**
- **XGBoost**: 높은 성능의 그래디언트 부스팅
- **CatBoost**: 카테고리 변수 처리에 특화된 부스팅
- **앙상블**: 다중 모델 결합을 통한 성능 향상

### 5. 평가 및 분석
- 다양한 평가 지표 (RMSE, R², MAE, MAPE 등)
- SHAP 기반 피처 중요도 분석
- 잔차 분석 및 시각화
- 모델 간 성능 비교

## 설치 및 실행

### 1. 환경 설정
```bash
# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate     # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. 데이터 준비
- `human_combined_full_data_utf8.csv` 파일을 프로젝트 루트 디렉토리에 위치시킵니다.

### 3. 실행
```bash
python main.py
```

## 설정 변경

`src/config/settings.py` 파일에서 다음 설정들을 변경할 수 있습니다:

- **데이터 설정**: 파일 경로, 인코딩
- **분할 설정**: 훈련/테스트 연도
- **피처 엔지니어링**: lag 기간, 롤링 윈도우 크기
- **변수 선택 설정**: 
  - `CORRELATION_THRESHOLD`: 상관관계 기준 (기본값 0.95, Elastic Net 사용시 무관)
  - `USE_PCA`: PCA 사용 여부 (기본값 True, 실험 결과 반영)
  - `USE_ELASTIC_NET`: Elastic Net 변수선택 사용 여부 (기본값 True)
- **모델 파라미터**: 각 모델별 하이퍼파라미터
- **출력 설정**: 결과 저장 여부 및 경로

## 팀원 논의 반영사항

### **박승정님 제안**: 다중공선성 해결
**해결책 구현:**
- **Elastic Net 자동 변수선택** 도입
- **53개 고상관 변수쌍** → 알고리즘이 지능적으로 처리
- **상관관계 기반 제거 대체**: 더 정교한 변수선택
- **다중공선성 근본 해결**: L1+L2 정규화 활용

### 원본 데이터 품질 검증
- **weekday 오류 발견**: 2020-05-01이 금요일인데 원본에서 잘못된 값
- **자동 수정 시스템**: TM 컬럼에서 정확한 weekday 재계산
- **검증 로깅**: 주요 날짜들의 올바른 요일 확인 메시지

### 변수 선택 방식 개선
- **도메인 지식 반영**: 원본 기상변수 우선, 단순한 변수명 선호
- **투명성 확보**: 제거되는 변수 쌍과 사유를 CSV 파일로 저장
- **중복 변수 방지**: weekday/py_weekday 같은 중복 변수 자동 통합

### PCA 사용 논의 결과
- **기본값 True**: 실험적으로 3개 컴포넌트 추가
- **성능 모니터링**: PCA 효과 지속적 검증
- **해석력 vs 성능**: 트레이드오프 고려

## 출력 파일

실행 완료 후 `outputs/` 디렉토리에 다음 파일들이 생성됩니다:

### 모델 결과
- `evaluation_results.csv`: 모델별 평가 결과 (RMSE, R², MAE 등)

### **Elastic Net 변수선택 결과**
- `elastic_net_selection_coefficients.csv`: 변수별 Elastic Net 계수 (중요도 순 정렬)
- `elastic_net_selection_parameters.csv`: 최적 alpha, l1_ratio, CV 점수
- `elastic_net_selection_selected_features.csv`: 최종 선택된 84개 변수 목록

### **실제 성능 향상 결과**
- **변수 선택**: 95개 → 84개 (11.6% 감소)
- **다중공선성 해결**: L1+L2 정규화로 자동 처리
- **최고 성능**: LightGBM RMSE 0.734, R² 0.725

## 주요 클래스 및 함수

### FeatureEngineer
시간적, 날씨적, 상호작용 피처들을 생성하고 데이터 품질을 검증하는 클래스입니다.

```python
fe = FeatureEngineer()
df_featured = fe.engineer_all_features(df)
# 자동으로 weekday 오류 검출 및 수정
```

### SmartFeatureSelector
Elastic Net과 도메인 지식을 결합한 변수선택 클래스입니다.

```python
selector = SmartFeatureSelector()

# Elastic Net 자동 변수선택
features, results = selector.elastic_net_selection(train_df)

# 통합 변수선택 (권장)
final_features, combined_results = selector.combined_selection(
    train_df, use_elastic_net=True, use_correlation=False
)
```

### ModelTrainer
다양한 머신러닝 모델을 훈련하고 평가하는 클래스입니다.

```python
trainer = ModelTrainer()
trainer.train_all_models(X_train, y_train)
results = trainer.evaluate_all_models(X_test, y_test)
```

### StatsHolder
통계적 피처와 클러스터링 피처를 관리하는 클래스입니다.

```python
stats = StatsHolder()
stats.fit(train_df)
transformed_df = stats.transform(test_df)
```

## 최신 성능 결과

**최종 모델 성능** (Elastic Net 변수선택 적용):
- **LightGBM**: RMSE 0.734, R² 0.725 **최고 성능**
- **CatBoost**: RMSE 0.764, R² 0.703
- **RandomForest**: RMSE 0.770, R² 0.698  
- **XGBoost**: RMSE 0.810, R² 0.666

**주요 개선사항**:
- 박승정님 제안으로 다중공선성 해결
- 원본 데이터 weekday 오류 수정
- 95개 변수 → Elastic Net 자동 최적화