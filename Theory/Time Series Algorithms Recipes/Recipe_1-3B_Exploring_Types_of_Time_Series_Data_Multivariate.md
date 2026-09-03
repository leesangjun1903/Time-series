# Recipe 1-3B. Exploring Types of Time Series Data — Multivariate

> 학습 수준: **대학/대학원 → 연구자 발전형**  
> 원문 위치: **PDF 24–28 / book pp.9–13**  
> 기준 문헌: *Time Series Algorithms Recipes* (Apress, 2023)

## 1. Executive Summary — 10문장 이내

1. 이 절의 핵심은 시계열을 단순한 표가 아니라 시간 순서가 보존되어야 하는 데이터 구조로 다루는 것입니다.
2. 모델링 전에 timestamp의 타입, 정렬, 중복, 샘플링 간격과 결측을 확인하는 과정이 필요합니다.
3. 시간 정보가 잘못 처리되면 이후 어떤 고급 모델을 사용해도 평가가 왜곡될 수 있습니다.
4. 연구 단계에서는 각 변수의 실제 이용 가능 시점을 기록하여 미래 정보 누수를 차단해야 합니다.
5. 최신 forecasting 연구에서도 preprocessing leakage와 평가 프로토콜이 재현성의 핵심 이슈로 강조됩니다.
6. 본 학습자료는 Recipe 1-3B “Exploring Types of Time Series Data — Multivariate”의 원문을 출발점으로 하되, 대학원 수준의 수학적 정의와 2020년 이후 연구를 연결합니다.

## 2. 목적과 필요성

이 Recipe를 공부하는 목적은 Python API를 외우는 것이 아니라, 해당 절이 가정하는 데이터 생성 구조와 평가 조건을 이해하는 것입니다. 시계열 연구에서는 동일한 알고리즘이라도 시간축 처리, validation 방식, feature availability에 따라 성능이 크게 달라집니다. 특히 연구자로 발전하려면 “코드가 실행된다”와 “통계적으로 타당한 예측 실험이다”를 구분해야 합니다.

> **용어 설명 — data provenance**  
> 데이터가 언제, 어디서, 어떤 전처리를 거쳐 만들어졌는지 추적 가능한 기록입니다.

> **용어 설명 — schema**  
> 열 이름, dtype, 단위, 허용 범위 같은 데이터 구조 규칙입니다.

## 3. 이론 강의


### 3.1 다변량 시계열의 핵심 구조

다변량 시계열은 시점 $t$에서 벡터

$$
\mathbf{Y}_t=[Y_{1t},Y_{2t},\ldots,Y_{pt}]^\top
$$

를 관찰하는 문제입니다. 중요한 것은 단순히 열이 여러 개라는 사실이 아니라, **변수 간 동시 상관관계와 시간 지연 상관관계가 함께 존재한다는 점**입니다.

교차공분산은

$$
\Gamma_{ij}(k)=\text{Cov}(Y_{i,t},Y_{j,t-k})
$$

로 정의할 수 있습니다. $\Gamma_{ij}(k)$가 크다면 $j$번째 변수의 $k$시점 과거가 $i$번째 변수와 선형적으로 관련될 가능성이 있습니다. 다만 상관은 인과를 의미하지 않으므로, 공정·경제·물리적 선후관계를 함께 검토해야 합니다.

> **용어 설명 — cross-covariance**  
> 서로 다른 두 변수의 시간 지연 관계를 공분산으로 측정한 값입니다. 같은 변수의 지연 관계를 보는 autocovariance의 다변량 확장입니다.

예측은 일반적으로

$$
\hat{\mathbf{Y}}_{t+h\mid t}=f(\mathbf{Y}_{t},\ldots,\mathbf{Y}_{t-L+1},\mathbf{X}_{t+h}^{\text{known}})
$$

처럼 표현합니다. 여기서 $\mathbf{X}_{t+h}^{\text{known}}$는 예측 시점에도 실제로 알 수 있는 달력·계획값 같은 미래 공변량입니다. 미래에 알 수 없는 센서값을 입력에 넣으면 target leakage가 됩니다.




### 수식 기호 상세 해설

- $\mathbf{Y}_t$: 시점 $t$의 여러 변수를 한 번에 묶은 $p$차원 벡터입니다.
- $Y_{it}$: 시점 $t$의 $i$번째 변수입니다.
- $p$: 동시에 모델링하는 변수의 수입니다.
- $\top$: 벡터/행렬의 전치(transpose)를 뜻합니다.
- $\Gamma_{ij}(k)$: 변수 $j$의 $k$시점 과거와 현재 변수 $i$ 사이의 cross-covariance입니다.
- $i,j$: 변수 인덱스, $k$: lag 인덱스입니다.
- $\mathbf{X}_{t+h}^{\text{known}}$: 예측 시점에서 실제로 미리 알 수 있는 미래 공변량 벡터입니다.
- $f(\cdot)$: 과거 여러 변수와 알려진 미래 정보를 결합해 미래 벡터를 예측하는 함수입니다.

### 3.3 무엇을 학습하거나 추정하는가

이 절의 핵심 추정 대상은 데이터의 시간 구조를 설명하는 **파라미터 또는 상태**입니다. 통계모델이라면 계수와 오차분산을 추정하고, tree/deep model이라면 예측손실을 최소화하는 함수 $f_\theta$를 학습합니다. 공통적인 연구 목표는 새로운 미래 구간에서의 기대 손실

$$
R_{\text{future}}(f)=\text{E}\left[L(Y_{t+h},f(\mathcal{I}_t))\right]
$$

을 낮추는 것입니다. 예측 위험식의 기호는 다음과 같습니다. $R_{\text{future}}(f)$는 아직 보지 못한 미래 분포에서의 기대 예측손실, $\text{E}[\cdot]$는 그 분포에 대한 기대값, $Y_{t+h}$는 $h$단계 미래의 실제값, $f$는 예측함수, $\mathcal{I}\_t$는 시점 $t$까지 이용 가능한 정보, $L(\cdot,\cdot)$은 실제값과 예측값의 오차를 수치화하는 손실함수입니다. $f=f_\theta$로 쓸 때 $\theta$는 데이터에서 학습하는 모델 파라미터 전체를 뜻합니다.

> **용어 설명 — information set $\mathcal{I}_t$**  
> 예측 순간에 현실적으로 알고 있는 모든 정보의 집합입니다. 미래 target이나 미래에만 측정되는 sensor 값을 포함하면 데이터 누수입니다.

## 4. 원문이 직접 보고한 내용

저자는 Beijing pollution 데이터를 이용해 PM2.5와 기상 변수를 함께 다루는 multivariate time series를 소개하고 여러 변수를 시간축에 따라 시각화합니다.

이 문단은 **저자의 보고를 요약한 것**이며, 아래의 비판적 해석과 구분해야 합니다.

## 5. 비판적 해석 및 연구자 관점

다변량 데이터는 정보량이 많지만 leakage 경로도 많아집니다. 각 변수의 허용 가능 시간열을 명시해야 합니다.

또한 이 절의 결과를 다른 Recipe와 비교할 때는 데이터셋, target scale, forecast horizon, split, metric이 동일한지 확인해야 합니다. 조건이 다르면 숫자의 크기만으로 모델 우열을 말할 수 없습니다.

## 6. 통계적으로 취약한 부분과 비교 불가능한 수치

PM2.5 결측을 0으로 채우는 처리는 물리적으로 실제 0 농도와 결측을 혼동할 수 있습니다. 결측 처리 전략이 미래 정보를 사용하지 않는지도 명시되어 있지 않습니다.

**비교 가능성 판정:** 이 Recipe의 숫자는 동일한 데이터·동일한 forecast origin·동일한 horizon·동일한 metric을 사용한 실험끼리만 직접 비교해야 합니다. 다른 target이나 단위가 다른 RMSE는 직접 크기 비교를 하지 않습니다.

## 7. 문서가 답하지 않는 질문과 답변

**질문 1. 왜 timestamp를 단순 문자열로 두면 안 되나요?**  
정렬이 사전식으로 처리되거나 시간 간격 계산이 불가능해질 수 있기 때문입니다. datetime 또는 정수 time index로 명확히 변환해야 합니다.

**질문 2. 결측값을 0으로 채워도 되나요?**  
0이 실제 가능한 물리값이면 결측과 실제 0을 혼동합니다. 결측 indicator, forward-only imputation, model-based imputation 등을 train-only 조건으로 비교해야 합니다.

**질문 3. 파일 포맷이 모델 성능에 영향을 주나요?**  
직접적인 알고리즘 성능보다 dtype 손실, timestamp precision, category encoding 변화가 간접적으로 재현성과 성능을 바꿀 수 있습니다.

## 8. 2020년 이후 관련 최신 연구 비교 분석

2020년 이후 연구에서는 모델 구조보다 **평가 프로토콜과 데이터 누수 방지**가 재현성의 핵심이라는 점이 강조됩니다. 2022년 forecast evaluation 연구는 fixed-origin과 rolling-origin을 구분하고, smoothing·decomposition·normalization 단계에서도 미래 정보가 들어갈 수 있음을 명시합니다. 2024~2026 foundation model 연구는 대규모 사전학습을 사용하지만, 새로운 도메인에서 zero-shot 성능이 항상 최적이라는 보장은 없으며 train/test contamination과 domain shift 검증이 더 중요해졌습니다.

| 연구 | 핵심 기술 변화 | 이 Recipe에 주는 의미 |
|---|---|---|
| Hewamalage et al., 2022 | rolling-origin 평가와 preprocessing leakage | 전처리도 fold 내부에서 수행해야 한다는 원칙 |
| TimesFM, 2024 | 100B time-points pretraining, zero-shot | 데이터 규모가 커져도 target-domain 검증은 필요 |
| TimesFM-3, 2026 | multivariate zero-shot + covariates | 다변량 데이터 schema와 covariate availability가 더 중요해짐 |

### 최신 연구 결과를 읽을 때의 주의점

논문에서 “SOTA”라고 보고된 수치는 특정 benchmark, split, horizon, preprocessing, tuning budget에 종속됩니다. 특히 foundation model은 pretraining corpus가 benchmark와 얼마나 겹치는지, zero-shot인지 fine-tuned인지에 따라 비교 조건이 달라집니다. 따라서 **서로 다른 논문의 숫자를 한 표에 놓고 단순 순위화하지 않습니다.**

## 9. 실제 파이프라인 적용 방향

1. timestamp parser와 timezone을 명시합니다.  
2. 시간 정렬, 중복, 누락 interval, frequency를 검사합니다.  
3. 각 변수의 측정 시점과 실제 이용 가능 시점을 데이터 사전에 기록합니다.  
4. 원본 immutable copy와 전처리 버전을 분리합니다.  
5. 저장 시 dtype/schema/version hash를 함께 관리합니다.

### 일반화 성능을 높이기 위한 연구 방향

- 모델 복잡도를 먼저 키우기보다 **시간순 교차검증과 leakage 제거**로 평가분산을 줄입니다.
- 예측 horizon별로 필요한 구조가 다르므로 $h=1$과 long-horizon을 분리하여 튜닝합니다.
- 여러 seed 또는 여러 rolling origin에서 평균과 표준편차를 보고 selection noise를 줄입니다.
- 작은 데이터에서는 low-capacity statistical/tree baseline을 유지하고, deep/foundation model이 실제로 유의한 개선을 주는지 검증합니다.
- domain shift가 예상되면 최근 구간 가중, calibration, covariate shift 진단, fine-tuning을 별도 실험합니다.

## 10. 후속 연구 계획

이 Recipe를 단독 실습으로 끝내지 않고, 동일 데이터에 대해 **고전 baseline → leakage-safe validation → 최신 모델**의 순서로 비교하는 것이 좋습니다. 후속 연구에서는 (1) rolling-origin 반복평가, (2) residual diagnostic, (3) 모델 복잡도 대비 성능, (4) domain shift 구간 성능, (5) uncertainty calibration을 공통 실험 프로토콜로 고정합니다. 이렇게 해야 모델 이름이 바뀌어도 동일한 연구 질문 아래에서 비교할 수 있습니다.

## 11. 참고자료 및 출처

- **Time Series Algorithms Recipes: Implement Machine Learning and Deep Learning Techniques with Python** — Akshay R. Kulkarni, Adarsha Shivananda, Anoosh Kulkarni, V. Adithya Krishnan, Apress, 2023. https://doi.org/10.1007/978-1-4842-8978-5
- **Forecast evaluation for data scientists: common pitfalls and best practices** — Hewamalage, Ackermann, Bergmeir, published online 2022; Data Mining and Knowledge Discovery 37 (2023). https://doi.org/10.1007/s10618-022-00894-5
- **Deep Time Series Models: A Comprehensive Survey and Benchmark** — Wang et al., IEEE TPAMI, 2026 (arXiv versions 2024–2026). https://doi.org/10.1109/TPAMI.2026.3690845
- **A decoder-only foundation model for time-series forecasting** — Das, Kong, Sen, Zhou, ICML 2024. https://research.google/pubs/a-decoder-only-foundation-model-for-time-series-forecasting/
- **Chronos: Learning the Language of Time Series** — Ansari et al., Transactions on Machine Learning Research, 2024. https://arxiv.org/abs/2403.07815
- **TimesFM-3: A zero-shot foundation model for multivariate forecasting** — Google Research, 2026-08-31. https://www.research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/

---

### 학습 체크포인트

이 절을 공부한 뒤에는 **정의 → 수식 → 가정 → 추정 대상 → 평가 설계 → 누수 가능성 → 최신 연구와의 차이**를 자신의 말로 설명할 수 있어야 합니다. 코드를 외우는 것보다 “어떤 조건에서 이 방법이 맞고, 어떤 조건에서 실패하는가”를 설명하는 능력이 연구 단계의 목표입니다.
