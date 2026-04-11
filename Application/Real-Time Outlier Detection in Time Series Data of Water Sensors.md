# Real-Time Outlier Detection in Time Series Data of Water Sensors

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

본 논문(van de Wiel, van Es & Feelders, 2020)은 네덜란드 수자원 당국 "Aa en Maas"의 수위 센서 시계열 데이터에서 **실시간 이상값(outlier) 탐지**를 위해 단변량(univariate) 및 다변량(multivariate) 방법들을 체계적으로 비교하고, **다변량 분위수 회귀 다층 퍼셉트론(QR-MLP)**이 가장 우수한 성능을 보임을 주장합니다.

### 주요 기여

| 기여 항목 | 설명 |
|---|---|
| **체계적 알고리즘 비교** | 10개 이상의 단/다변량 모델을 동일 조건에서 비교 |
| **합성 이상값 평가 프레임워크** | 도메인 전문가 협력 하에 Spike/Jump/Drift 이상값 시뮬레이션 |
| **QR-MLP의 우수성 입증** | 레이블 없는 데이터에서 비지도 방식으로 다양한 이상값 탐지 가능 |
| **실용적 실시간 적용 가능성** | 월 1회 수준 탐지 성능을 연 1회 수준에서 월 1회 미만으로 향상 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

- **문제 1: 레이블 없는 원시 데이터에서의 이상값 탐지**
  - 도메인 전문가가 이상값 여부를 표시하지 않은 비지도(unlabelled) 환경
- **문제 2: 실시간 탐지 요구**
  - 센서 오작동이나 환경 변화를 즉각 감지해야 함
- **문제 3: 단변량 모델의 Drift 탐지 한계**
  - 단변량 모델은 과거 이상값(drift)을 미래 예측에 반영하여 이상값 탐지에 실패할 수 있음

이상값의 유형은 세 가지로 정의됩니다:
- **Jump**: 일정 기간 동안 상수값만큼 증가/감소 후 원래 값으로 복귀
- **Extreme value (Spike)**: 고립된 단일 포인트의 급격한 이상
- **Linear drift**: 서서히 선형적으로 변화하는 추세

### 2.2 제안하는 방법 및 수식

#### (1) 센서 선택 및 결측값 대체

- 결측값 비율 10% 초과 센서 제외
- MICE(Multivariate Imputation by Chained Equations) 기반 선형 회귀 대체법 사용

#### (2) 피처 엔지니어링

- **다변량**: rolling lag, min, max, mean (15분, 30분, 1h, 2h, 4h, 8h, 16h 윈도우)
- **단변량**: 평균값 (64h, 128h, ..., 1048h 윈도우)
- 모든 피처는 zero mean, unit variance로 표준화

#### (3) 선형 회귀의 분위수 추정 (Lasso)

선형 회귀 기반 이상값 탐지를 위한 분위수 계산:

$$q_i = \hat{y} \pm \frac{i}{2}\hat{\sigma}(y_{train}) \quad | \quad i \in \{1, 2, 3\}$$

여기서 $\hat{y}$는 예측값, $\hat{\sigma}(y_{train})$은 훈련 데이터의 표준편차입니다.

> **단점**: 전체 데이터에 걸쳐 분위수 폭이 고정(fixed quantile width)되어 예측 불확실성이 구간별로 다를 경우 부적합합니다.

Lasso 페널티($\lambda$)는 검증 세트로 튜닝하며, 최종적으로 $\lambda = 0.03$ 사용:

```math
\hat{\beta} = \arg\min_{\beta} \left\{ \sum_{i=1}^{n}(y_i - \mathbf{x}_i^T\beta)^2 + \lambda \|\beta\|_1 \right\}
```

#### (4) QR-MLP (핵심 제안 모델)

**Pinball Loss Function** (분위수 손실 함수):

$$L_\tau(\hat{q}_\tau, y) = \begin{cases} \tau \cdot (y - \hat{q}_\tau) & \text{if } y \geq \hat{q}_\tau \\ (1 - \tau) \cdot (\hat{q}_\tau - y) & \text{if } y < \hat{q}_\tau \end{cases}$$

여기서 $\tau \in (0, 1)$은 분위수, $\hat{q}_\tau$는 모델의 $\tau$ 분위수 예측값입니다.

전체 손실은 여러 분위수를 동시에 고려합니다:

$$\mathcal{L} = \sum_{\tau} L_\tau(\hat{q}_\tau, y)$$

#### (5) 이상값 분류: Western Electric 규칙 적용

- **Rule 1**: 단일 포인트가 $3\sigma$ 한계를 벗어나면 이상값으로 판정
- **Rule 2 (변형)**: 드리프트 탐지를 위한 다운샘플링 기반 알고리즘 (Algorithm 1)

$$\text{outliers daily} = (y_{\text{daily}} > q2_{\text{upper}}) \cup (y_{\text{daily}} < q2_{\text{lower}})$$

**Algorithm 1** (Drift Detection by Downsampling):
```
DRIFT_DETECTION(y_in, q2_upper, q2_lower):
  y_daily ← DOWNSAMPLE_TO_DAY(y_in)
  q2_upper ← DOWNSAMPLE_TO_DAY(q2_upper)
  q2_lower ← DOWNSAMPLE_TO_DAY(q2_lower)
  outliers_daily ← (y_daily > q2_upper) ∪ (y_daily < q2_lower)
  return UPSAMPLE_TO_15MIN(outliers_daily)
```

#### (6) 평가 지표: $F_\beta$-score ($\beta = 2$)

$$F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \times \text{Recall}}{(\beta^2 \times \text{Precision}) + \text{Recall}}$$

$\beta = 2$로 설정하여 **재현율(Recall)에 더 높은 가중치** 부여 (이상값을 놓치는 것이 더 큰 문제이기 때문).

### 2.3 모델 구조

#### QR-MLP 아키텍처 (센서별)

| 센서 | Dropout | Learning Rate | 레이어 수 | 뉴런/레이어 | 검증 손실 |
|---|---|---|---|---|---|
| 104OYE | 0.4 | 0.0005 | 1 | 128 | 0.1820 |
| 103HOE | 0.0 | 0.005 | 1 | 256 | 0.8790 |
| 201D | 0.4 | 0.005 | 2 | 128 | 0.9580 |
| 102BFS | 0.4 | 0.00005 | 8 | 128 | 1.5330 |

- **하이퍼파라미터 튜닝**: Hyperband 알고리즘(270 trials) 사용
- **앙상블**: 다변량 실험에서 10개 신경망의 예측 평균 사용
- **데이터 분할**: Train 60% / Validation 20% / Test 20%

#### 비교 모델 목록

| 유형 | 모델 |
|---|---|
| **다변량** | LR (Lasso), QR-MLP, QR-Perceptron, QRF, QR-RNN (LSTM/GRU) |
| **단변량** | LR, QR-MLP, QR-Perceptron, AR, Isolation Forest |

### 2.4 성능 향상

- **다변량 > 단변량**: Friedman Aligned Ranks 검정에서 유의미한 차이 확인 ($p < 0.05$)
- **QR-MLP**: Drift 및 Jump에서 전반적으로 최고 성능
- **Drift 탐지**: 0.05m 드리프트를 평균 1개월 이내에 탐지 (기존 연 1회 수동 검사 대비 대폭 향상)
- **실제 이상값 탐지**: 도메인 전문가가 표시한 실제 이상값(108HOL 센서)도 탐지 성공

### 2.5 한계

1. **극단값(Extreme) 탐지 한계**: 다변량 모델이 단변량 모델(특히 AR)보다 낮은 성능 (합성 평가의 아티팩트 가능성)
2. **고정 분위수 폭 문제**: 선형 회귀 기반 방법에서 분위수 폭이 고정됨
3. **센서 상관관계 의존성**: 102BFS처럼 상관 센서가 부족하면 다변량 방법 적용 불가
4. **센서별 개별 아키텍처**: 일반화된 단일 아키텍처를 찾지 못함
5. **합성 평가의 현실성**: 실제 이상값 패턴이 합성 시나리오와 다를 수 있음
6. **전파 오류 미처리**: 하나의 센서 오작동이 다른 센서 예측에 미치는 영향 미분석

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재 일반화 한계

논문은 다음과 같은 일반화 한계를 명시적으로 인정합니다:

> *"It was not possible to find one general network architecture that works in all cases."*

즉, 센서마다 별도의 아키텍처가 필요하며, 이는 **새로운 센서나 도메인에 직접 적용 시 재튜닝 비용**이 발생함을 의미합니다.

### 3.2 일반화 성능 향상 가능성 (논문 내 언급 사항)

#### (A) 모델 앙상블 전략

다변량 실험에서 10개 QR-MLP의 앙상블 평균을 사용함으로써 랜덤 가중치 초기화의 분산을 줄이고 예측 안정성을 향상시켰습니다:

$$\hat{q}_\tau^{\text{ensemble}} = \frac{1}{M}\sum_{m=1}^{M} \hat{q}_\tau^{(m)}$$

여기서 $M$은 앙상블 크기(다변량: 10, 단변량: 5)입니다.

#### (B) 상관 센서 기반 피처 선택

가장 상관관계가 높은 4개 센서를 자동으로 선택함으로써 과적합 위험을 줄이고 일반화 가능성을 높였습니다.

#### (C) 혼합 모델 전략 제안

논문은 일반화 성능 향상을 위해 **서로 다른 이상값 유형에 특화된 모델을 결합**하는 앙상블 전략을 미래 연구 방향으로 제안합니다:

> *"It may be a fruitful idea to use different models to detect different outlier categories. For example, combining the results of an AR model and a multivariate QR-MLP model could work to detect extreme values, jumps, and drifts."*

이를 수식으로 표현하면:

$$\text{Outlier}(t) = f_{\text{AR}}(t) \vee f_{\text{QR-MLP}}(t)$$

#### (D) 피처 엔지니어링의 일반화 기여

다양한 시간 윈도우 기반 통계적 피처(rolling mean, min, max, lag)를 사용함으로써 특정 데이터셋에 과적합되지 않고 다양한 시계열 패턴에 대응 가능합니다.

### 3.3 일반화 성능 향상을 위한 추가 제언 (논문 기반)

- **더 현실적인 합성 시나리오**: gradual extreme value 등 복합 이상값 유형 추가
- **파일럿 프로그램 운영**: 실제 환경에서의 성능 검증
- **이상값 원인 분석**: 이상값 유형별 근본 원인을 파악하여 모델 특화 가능

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4.1 향후 연구에 미치는 영향

#### (A) 합성 평가 프레임워크의 확산

레이블이 없는 환경에서 합성 이상값을 활용한 평가 방법론은 향후 수질 데이터, 기상 데이터, IoT 센서 데이터 등 다양한 분야로 확장될 수 있습니다.

#### (B) 분위수 회귀 기반 이상값 탐지의 활성화

QR-MLP가 단일 점 예측(point prediction)이 아닌 **예측 구간(prediction interval)**을 제공함으로써 불확실성을 정량화하는 접근법은 이후 연구에서 Conformal Prediction, Bayesian Deep Learning 등과 결합될 수 있습니다.

#### (C) 비지도 실시간 이상값 탐지의 실용화

도메인 지식(Western Electric 규칙)과 머신러닝(QR-MLP)을 결합한 하이브리드 접근법은 산업 IoT, 스마트 그리드 등 다양한 실시간 모니터링 분야에 영향을 줍니다.

### 4.2 앞으로 연구 시 고려할 점

#### (A) 모델 일반화 측면

- **Transfer Learning 적용**: 센서 A에서 학습한 모델을 센서 B에 전이하여 재튜닝 비용 절감
- **Meta-learning**: 소수 데이터(few-shot)로도 새로운 센서에 적응할 수 있는 MAML 등 적용 고려
- **Domain Adaptation**: 서로 다른 지역/기후 조건의 센서 데이터 간 도메인 이동 문제 해결

#### (B) 이상값 탐지 고도화

- **복합 이상값 시나리오**: Jump + Drift가 동시에 발생하는 복합 케이스 대응
- **점진적(Gradual) Extreme Value**: 논문이 언급한 것처럼 몇 번의 타임스텝에 걸친 급격 변화 시나리오 추가
- **온라인 학습(Online Learning)**: 데이터 분포 변화(concept drift)에 적응하는 점진적 학습 메커니즘

#### (C) 센서 오류 전파 문제

- 한 센서의 오작동이 다변량 모델의 입력 피처로 사용될 경우 다른 센서 예측에도 영향 → **결함 있는 피처 감지 모듈** 별도 개발 필요

#### (D) 설명 가능성(Explainability)

- QR-MLP는 블랙박스 모델이므로 도메인 전문가의 신뢰 확보를 위해 SHAP, LIME 등 XAI 기법 통합 권장

#### (E) 계산 효율성

- 하이퍼파라미터 튜닝(270 trials)의 계산 비용이 높음 → Neural Architecture Search(NAS)나 베이지안 최적화로 효율화 가능

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

> **⚠️ 주의**: 아래 비교는 본 논문의 내용과 AI 학습 데이터 기반의 일반적 지식을 결합한 것입니다. 각 최신 논문의 구체적 수치는 해당 논문을 직접 확인하시기 바랍니다.

### 5.1 주요 연구 비교

| 연구 | 방법론 | 본 논문 대비 차이점 |
|---|---|---|
| **Audibert et al. (2020)** *USAD* | Autoencoder 기반 비지도 이상값 탐지 | 재구성 오차 기반 / 분위수 예측 구간 미제공 |
| **Tuli et al. (2022)** *TranAD* | Transformer 기반 이상값 탐지 | Self-attention으로 장거리 의존성 포착 / 계산 비용 높음 |
| **Shen et al. (2020)** *Timeseries Anomaly Detection* | 딥러닝 + 통계적 방법 혼합 | 복잡도 높음, 실시간 적용 제한 |
| **Schmidl et al. (2022)** *Anomaly Detection Benchmark* | 71개 탐지 알고리즘 벤치마크 | 수자원 특화 아님 / 일반 도메인 비교 |

### 5.2 주요 트렌드와 본 논문의 위치

```
2020: QR-MLP (본 논문) → 분위수 회귀 + 도메인 규칙 하이브리드
2021: Transformer 기반 이상값 탐지 연구 급증
2022: Foundation Model (대형 사전 학습 모델)의 시계열 적용 시도
2023: LLM 기반 이상값 탐지 (예: GPT-4 기반 zero-shot 탐지)
```

**본 논문의 차별성**:
- 수자원 도메인 특화 합성 평가 프레임워크
- 계산 효율적인 QR-MLP (Transformer 대비)
- 도메인 전문가 규칙(Western Electric)과 ML의 결합
- 실시간 적용 가능성 검증

**본 논문의 한계 (최신 연구 대비)**:
- Attention 메커니즘 미활용 (장거리 시간 의존성 포착 제한)
- Conformal Prediction 등 최신 불확실성 정량화 기법 미적용
- 사전 학습 모델(Pre-trained model) 미활용으로 일반화 한계

---

## 참고 자료

**본 논문 (주요 출처)**:
- van de Wiel, L., van Es, D., & Feelders, A.J. (2020). *Real-Time Outlier Detection in Time Series Data of Water Sensors*. In: Advanced Analytics and Learning on Temporal Data (AALTD 2020), LNAI 12588, pp. 155–170. Springer. https://doi.org/10.1007/978-3-030-65742-0_11
- Utrecht University Repository: https://dspace.library.uu.nl/handle/1874/481634

**논문 내 인용 문헌**:
- Koenker, R., Hallock, K.F. (2001). Quantile regression. *Journal of Economic Perspectives*, 15(4), 143–156.
- Meinshausen, N. (2006). Quantile regression forests. *Journal of Machine Learning Research*, 7, 983–999.
- Leigh, C., et al. (2019). A framework for automated anomaly detection in high frequency water-quality data from in situ sensors. *Science of the Total Environment*, 664, 885–898.
- Liu, F.T., Ting, K.M., Zhou, Z.H. (2008). Isolation forest. *2008 Eighth IEEE International Conference on Data Mining*, pp. 413–422.
- Li, L., et al. (2017). Hyperband: a novel bandit-based approach to hyperparameter optimization. *Journal of Machine Learning Research*, 18(1), 6765–6816.
- Van Buuren, S., Groothuis-Oudshoorn, K. (2011). MICE: multivariate imputation by chained equations in R. *Journal of Statistical Software*, 45(3), 1–68.
- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society*, Series B, 58(1), 267–288.
- Rodrigues, F., Pereira, F.C. (2018). Beyond expectation: deep joint mean and quantile regression for spatio-temporal problems. arXiv:1808.08798.
- Western Electric Company (1956). *Statistical Quality Control Handbook*.
