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
│   │   ├── feature_engineer.py  # 피처 엔지니어링
│   │   ├── stats_holder.py      # 통계적 피처 생성
│   │   └── pca_holder.py        # PCA 차원 축소
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model_trainer.py     # 모델 훈련 및 평가
│   │   └── ensemble_model.py    # 앙상블 모델
│   └── utils/
│       ├── __init__.py
│       ├── evaluation.py        # 평가 유틸리티
│       └── visualization.py     # 시각화 유틸리티
├── main.py                      # 메인 파이프라인
├── requirements.txt             # 의존성 패키지
└── README.md                    # 프로젝트 설명서
```

## 주요 기능

### 1. 데이터 처리
- CSV 파일 로딩 및 인코딩 처리
- 연도별 train/test 분할
- 태풍/비태풍 데이터 분리 분석
- 고상관도 피처 제거

### 2. 피처 엔지니어링
- **시간적 피처**: 날짜, 계절, 요일, 주기적 피처 (sin/cos)
- **날씨 피처**: 체감온도, 불쾌지수, 날씨 상호작용
- **지연/롤링 피처**: 과거 데이터 기반 lag 및 rolling 통계
- **통계적 피처**: 구별 통계, 극값 탐지, 클러스터링
- **공간적 피처**: 도심/해안 거리, 인근 지역 네트워크 피처
- **PCA 피처**: 차원 축소를 통한 잠재 피처

### 3. 머신러닝 모델
- **RandomForest**: 안정적인 베이스라인 모델
- **LightGBM**: 빠르고 효율적인 그래디언트 부스팅
- **XGBoost**: 높은 성능의 그래디언트 부스팅
- **CatBoost**: 카테고리 변수 처리에 특화된 부스팅
- **앙상블**: 다중 모델 결합을 통한 성능 향상

### 4. 평가 및 분석
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
  - `CORRELATION_THRESHOLD`: 상관관계 기준 (기본값 0.85, 팀원 요청에 따라 완화)
  - `USE_PCA`: PCA 사용 여부 (기본값 False, 팀원 합의 반영)
  - `USE_FEATURE_IMPORTANCE_FILTERING`: 변수 중요도 기반 제거 여부 (기본값 False)
- **모델 파라미터**: 각 모델별 하이퍼파라미터
- **출력 설정**: 결과 저장 여부 및 경로

## 팀원 논의 반영사항

### 변수 선택 방식 개선
- **상관관계 기준 완화**: 0.95 → 0.85로 조정하여 유용한 변수 보존
- **도메인 지식 반영**: 원본 기상변수 우선, 단순한 변수명 선호
- **투명성 확보**: 제거되는 변수 쌍과 사유를 CSV 파일로 저장
- **중복 변수 방지**: weekday/py_weekday 같은 중복 변수 자동 통합

### PCA 사용 논의 결과
- **기본값 False**: 차원 축소 후 변수 추가의 모순 해결
- **선택적 사용**: 필요시 설정에서 활성화 가능
- **해석력 향상**: PCA 제거로 모델 해석력 개선

### 변수 중요도 기반 제거 논의
- **기본값 False**: 모델별 중요도 차이 및 자의적 기준 문제 해결
- **예측 목적 중심**: 상관관계 기반 제거로 충분
- **선택적 사용**: 필요시 설정에서 활성화 가능

## 출력 파일

실행 완료 후 `outputs/` 디렉토리에 다음 파일들이 생성됩니다:

### 모델 결과
- `evaluation_results.csv`: 모델별 평가 결과
- `test_predictions.csv`: 테스트 데이터 예측 결과
- `feature_importance_*.csv`: 모델별 피처 중요도
- `shap_summary_*.png`: SHAP 분석 결과 (선택사항)

### 변수 선택 투명성 (팀원 요청 반영)
- `high_correlation_pairs.csv`: 높은 상관관계 변수 쌍 목록
- `feature_removal_reasons.csv`: 변수 제거 사유 및 보존 변수
- `correlation_matrix_full.csv`: 전체 상관관계 행렬
- `selected_features.csv`: 최종 선택된 변수 목록

## 주요 클래스 및 함수

### FeatureEngineer
시간적, 날씨적, 상호작용 피처들을 생성하는 클래스입니다.

```python
fe = FeatureEngineer()
df_featured = fe.engineer_all_features(df)
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

## 기여 방법

1. 이 저장소를 포크합니다.
2. 새로운 기능 브랜치를 생성합니다 (`git checkout -b feature/새기능`)
3. 변경사항을 커밋합니다 (`git commit -am '새기능 추가'`)
4. 브랜치에 푸시합니다 (`git push origin feature/새기능`)
5. Pull Request를 생성합니다.

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 문의사항

프로젝트 관련 문의사항이 있으시면 이슈를 생성해주세요.
