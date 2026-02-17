<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

## WaveBound: 시계열 예측의 안정적 학습을 위한 동적 오차 한계

WaveBound는 시계열 예측 모델의 과적합 문제를 해결하기 위해 훈련 손실에 동적 오차 한계를 도입한 정규화 기법입니다. 이 방법은 각 시간 단계와 특징마다 개별적으로 오차 한계를 조정하여 모델이 예측 불가능한 패턴에 과도하게 적합되는 것을 방지합니다.[^1_1]

### 핵심 주장과 주요 기여

WaveBound의 주요 기여는 다음 세 가지로 요약됩니다:[^1_1]

1. **동적 오차 한계 제공**: 시계열 예측에서 훈련 손실의 오차 한계를 동적으로 제공하는 간단하면서도 효과적인 정규화 방법 제안
2. **일관된 성능 향상**: 6개의 실제 벤치마크에서 최신 시계열 예측 모델을 포함한 기존 모델들의 성능을 지속적으로 개선
3. **오차 한계 조정의 중요성 검증**: 각 시간 단계, 특징, 패턴에 대한 오차 한계 조정이 과적합 문제 해결에 중요함을 광범위한 실험을 통해 입증

## 해결하고자 하는 문제

### 시계열 예측의 과적합 문제

시계열 데이터는 일관성 없는 패턴과 예측 불가능한 행동을 포함하며, 이로 인해 딥러닝 모델이 불안정한 학습과 과적합에 시달립니다. 실제 데이터에 나타나는 비일관적 패턴은 모델이 특정 패턴에 편향되도록 만들어 일반화 능력을 제한합니다.[^1_1]

기존의 flooding 정규화 방법은 다음 두 가지 이유로 시계열 예측에 적용할 수 없습니다:[^1_1]

1. **평균 손실만 고려**: 이미지 분류와 달리 시계열 예측은 예측 길이와 특징 수의 곱만큼의 벡터 출력이 필요한데, 원래 flooding은 각 시간 단계와 특징을 개별적으로 다루지 않고 평균 훈련 손실만 고려합니다
2. **고정된 오차 한계**: 시계열 데이터에서는 서로 다른 패턴에 대해 오차 한계가 동적으로 변경되어야 하는데, 상수 flood level로는 이를 반영할 수 없습니다

## 제안하는 방법: WaveBound

### 수학적 정의

시계열 예측기 $g : \mathbb{R}^{L \times K} \rightarrow \mathbb{R}^{M \times K}$는 과거 시리즈 $x_t = \{z_{t-L+1}, z_{t-L+2}, ..., z_t : z_i \in \mathbb{R}^K\}$가 주어졌을 때 미래 시리즈 $y_t = \{z_{t+1}, z_{t+2}, ..., z_{t+M} : z_i \in \mathbb{R}^K\}$를 예측합니다. 여기서 $K$는 특징 차원, $L$과 $M$은 입력 길이와 출력 길이입니다.[^1_1]

경험적 위험은 각 예측 단계와 특징에 대한 위험의 합으로 표현됩니다:[^1_1]

$$
R(g) = \frac{1}{MK} \sum_{j,k} R_{jk}(g), \quad \hat{R}(g) = \frac{1}{MK} \sum_{j,k} \hat{R}_{jk}(g)
$$

여기서 $R_{jk}(g) := \mathbb{E}_{(u,v) \sim p(u,v)} [||g_{jk}(u) - v_{jk}||^2]$이고, $\hat{R}_{jk}(g) := \frac{1}{N} \sum_{i=1}^N ||g_{jk}(x_i) - (y_i)_{jk}||^2$입니다[^1_1].

### Wave Empirical Risk

WaveBound는 소스 네트워크 $g_\theta$와 타겟 네트워크 $g_\tau$를 사용하며, 타겟 네트워크는 소스 네트워크의 지수 이동 평균(EMA)으로 업데이트됩니다:[^1_1]

$$
\tau \leftarrow \alpha \tau + (1 - \alpha) \theta
$$

여기서 $\alpha \in [0, 1]$은 타겟 감쇠율입니다.[^1_1]

Wave empirical risk는 다음과 같이 정의됩니다:[^1_1]

$$
\hat{R}^{wb}(g_\theta) = \frac{1}{MK} \sum_{j,k} \hat{R}^{wb}_{jk}(g_\theta)
$$

$$
\hat{R}^{wb}_{jk}(g_\theta) = |\hat{R}_{jk}(g_\theta) - (\hat{R}_{jk}(g_\tau) - \epsilon)| + (\hat{R}_{jk}(g_\tau) - \epsilon)
$$

여기서 $\epsilon$은 소스 네트워크의 오차 한계가 타겟 네트워크의 오차로부터 얼마나 떨어질 수 있는지를 나타내는 하이퍼파라미터입니다.[^1_1]

### 모델 구조 및 작동 원리

WaveBound의 핵심 메커니즘은 다음과 같습니다:[^1_1]

1. **이중 네트워크 시스템**:
    - 소스 네트워크 $g_\theta$: 일반적인 그래디언트 하강으로 학습
    - 타겟 네트워크 $g_\tau$: 소스 네트워크의 EMA로 업데이트
2. **동적 오차 한계 추정**: 타겟 네트워크가 각 시간 단계와 특징에 대한 적절한 오차 하한을 추정하여, 소스 네트워크가 해당 한계 이하로 손실을 낮추려 할 때 그래디언트 상승을 수행하도록 합니다[^1_1]
3. **미니배치 최적화**: Jensen 부등식에 의해 다음이 성립합니다:[^1_1]

$$
\hat{R}^{wb}_{jk}(g) \leq \frac{1}{T} \sum_{t=1}^T (|\hat{R}_t)_{jk}(g) - (\hat{R}_t)_{jk}(g^*) + \epsilon| + (\hat{R}_t)_{jk}(g^*) - \epsilon)
$$

## 이론적 보장: MSE 감소

**Theorem 1**: 다음 두 조건이 만족될 때 $\text{MSE}(\hat{R}(g)) \geq \text{MSE}(\hat{R}^{wb}(g))$가 성립합니다:[^1_1]

(a) 모든 $(i,j), (k,l) \in I$에 대해 $(i,j) \neq (k,l)$일 때, $\hat{R}_{ij}(g) - \hat{R}_{ij}(g^*) \perp \hat{R}_{kl}(g)$

(b) $J(X)$의 모든 $(i,j)$에 대해 $\hat{R}_{ij}(g^*) < R_{ij}(g) + \epsilon$ (거의 확실하게)

이 정리는 네트워크 $g^*$가 충분한 표현력을 가지고 각 출력 변수에서 $g$와 $g^*$ 간의 손실 차이가 다른 출력 변수의 손실과 무관할 때, wave empirical risk 추정기의 MSE가 경험적 위험 추정기의 MSE보다 작을 수 있음을 보여줍니다.[^1_1]

## 성능 향상

### 다변량 설정 결과

6개의 실제 벤치마크(ETT, Electricity, Exchange, Traffic, Weather, ILI)에서 다양한 예측 모델(Autoformer, Pyraformer, Informer, LSTNet)에 WaveBound를 적용했을 때 일관된 성능 향상을 보였습니다.[^1_1]

주목할 만한 성과:

- **ETTm2 데이터셋 (M=96)**: Autoformer의 MSE를 22.13% 개선 (0.262 → 0.204), MAE를 12.57% 개선 (0.326 → 0.285)[^1_1]
- **LSTNet**: MSE 41.10% 개선 (0.455 → 0.268), MAE 27.98% 개선 (0.511 → 0.368)[^1_1]
- **장기 예측 (M=720)**: Autoformer의 MSE를 7.39% 개선 (0.446 → 0.413)[^1_1]


### 단변량 설정 결과

단변량 설정에서도 우수한 결과를 보였으며, 특히 단변량 시계열 예측을 위해 특별히 설계된 N-BEATS의 경우:[^1_1]

- **ETTm2 (M=96)**: MSE 8.22% 개선 (0.073 → 0.067), MAE 5.05% 개선 (0.198 → 0.188)[^1_1]
- **ECL (M=720)**: Informer의 MSE 40.10% 개선 (0.631 → 0.378), MAE 24.35% 개선 (0.612 → 0.463)[^1_1]


### 손실 경관 평탄화

WaveBound는 손실 경관(loss landscape)을 평탄화하여 모델의 견고성과 일반화를 개선합니다. 실험 결과 WaveBound를 사용한 모델이 원본 모델에 비해 더 부드러운 손실 경관을 보였으며, 이는 훈련의 안정성을 향상시킵니다.[^1_1]

## 모델의 일반화 성능 향상

### 일반화 갭 감소

WaveBound의 가장 중요한 기여는 일반화 갭(generalization gap) 감소입니다. 훈련 곡선 분석 결과:[^1_1]

- **WaveBound 없이**: 훈련 손실은 감소하지만 테스트 손실이 증가하여 높은 일반화 갭을 보임[^1_1]
- **WaveBound 사용**: 테스트 손실이 더 많은 에폭 동안 계속 감소하여, WaveBound가 시계열 예측에서 과적합을 성공적으로 해결함을 나타냄[^1_1]


### 동적 조정의 중요성

상수 flood level을 사용하는 원래 flooding 및 constant flooding과 비교했을 때, WaveBound(개별)가 모든 시간 단계에서 일관되게 낮은 테스트 오차를 보였습니다. 이는 시계열 예측에서 오차 한계를 조정하는 것의 중요성을 구체적으로 보여줍니다.[^1_1]

실험 결과(ECL 데이터셋):[^1_1]


| 방법 | Autoformer (M=96) | Autoformer (M=336) |
| :-- | :-- | :-- |
| Base model | MSE: 0.202, MAE: 0.317 | MSE: 0.247, MAE: 0.351 |
| Flooding | MSE: 0.194, MAE: 0.309 | MSE: 0.247, MAE: 0.351 |
| Constant flooding | MSE: 0.198, MAE: 0.314 | MSE: 0.247, MAE: 0.351 |
| WaveBound (Avg.) | MSE: 0.194, MAE: 0.309 | MSE: 0.221, MAE: 0.331 |
| **WaveBound (Indiv.)** | **MSE: 0.176, MAE: 0.288** | **MSE: 0.217, MAE: 0.327** |

### 특징 및 시간 단계별 개별 오차 한계의 필요성

논문은 각 특징과 시간 단계가 서로 다른 훈련 동역학을 보이며, 이는 예측 난이도가 특징에 따라 달라짐을 보여줍니다. 조기 종료(early-stopping)와 같은 기존 정규화 방법은 집계된 값만 고려하여 각 오차를 개별적으로 처리하지 않기 때문에 과적합 문제를 적절히 처리할 수 없습니다.[^1_1]

WaveBound는 각 특징과 시간 단계에 대해 훈련 오차가 특정 수준 이하로 떨어지는 것을 방지함으로써 테스트 손실을 낮은 수준으로 유지하며, 이는 시계열 예측에서 전반적인 성능 향상으로 이어집니다.[^1_1]

## 한계점

논문에서 명시적으로 언급된 한계는 다음과 같습니다:[^1_1]

1. **하이퍼파라미터 탐색**: $\epsilon$ 값을 {0.01, 0.001}에서 탐색해야 하며, 각 설정에 적절한 $\epsilon$을 선택해야 성능이 향상됩니다[^1_1]
2. **EMA 감쇠율**: $\alpha = 0.99$로 고정되어 있으며, 다른 데이터셋이나 모델에 대한 최적 값은 다를 수 있습니다[^1_1]
3. **추가 네트워크 필요**: 타겟 네트워크를 별도로 유지해야 하므로 메모리 사용량이 증가합니다[^1_1]

## 향후 연구에 미치는 영향

### 시계열 예측을 위한 전용 정규화의 중요성

WaveBound의 유의미한 성능 향상은 시계열 예측을 위해 정규화가 특별히 설계되어야 함을 나타냅니다. 이는 향후 연구의 중요한 방향을 제시합니다:[^1_1]

1. **도메인 특화 정규화**: 다른 시계열 작업(분류, 이상 탐지 등)을 위한 특화된 정규화 기법 개발
2. **적응형 오차 한계**: 데이터셋과 모델에 따라 자동으로 $\epsilon$ 값을 조정하는 메커니즘 연구
3. **다양한 아키텍처 적용**: Transformer, CNN, RNN 등 다양한 아키텍처에 대한 최적화된 적용 방법 연구

### 다른 도메인으로의 확장 가능성

WaveBound는 공간-시간 시계열 예측에서도 효과적임이 입증되었습니다. Graph WaveNet에 적용했을 때 METR-LA 트래픽 데이터셋에서 모든 메트릭(MAE, RMSE, MAPE)에서 성능을 향상시켰습니다.[^1_1]

### RevIN과의 결합

Reversible Instance Normalization (RevIN)과 결합했을 때 WaveBound는 더욱 큰 성능 향상을 보였습니다. 이는 WaveBound가 다른 정규화 및 정규화 기법과 상호 보완적임을 시사합니다.[^1_1]

## 향후 연구 시 고려할 점

### 1. 이론적 확장

- **더 강력한 이론적 보장**: Theorem 1의 가정을 완화하고 더 일반적인 조건에서의 수렴 보장 연구
- **다양한 손실 함수**: MSE 외에 MAE, Huber loss 등 다른 손실 함수에 대한 이론적 분석
- **수렴 속도 분석**: WaveBound를 사용한 최적화의 수렴 속도에 대한 이론적 분석


### 2. 실용적 개선

- **자동 하이퍼파라미터 튜닝**: $\epsilon$과 $\alpha$ 값을 자동으로 선택하는 메커니즘 개발
- **계산 효율성**: 타겟 네트워크 유지로 인한 메모리 및 계산 오버헤드 감소 방법
- **적응형 EMA**: 훈련 단계에 따라 $\alpha$를 동적으로 조정하는 전략


### 3. 새로운 응용 분야

- **Foundation 모델**: 대규모 사전 훈련된 시계열 foundation 모델에 WaveBound 적용
- **Few-shot 학습**: 제한된 데이터 상황에서 WaveBound의 효과 분석
- **멀티모달 시계열**: 다양한 소스의 시계열 데이터를 결합하는 멀티모달 설정에서의 적용


### 4. 해석 가능성

- **오차 한계 분석**: 학습된 오차 한계가 시계열의 어떤 특성(주기성, 트렌드, 노이즈 등)을 반영하는지 분석
- **특징 중요도**: WaveBound가 각 특징과 시간 단계에 부여하는 중요도 시각화 및 해석


## 2020년 이후 관련 최신 연구 비교 분석

### 정규화 및 일반화 기법

**1. Consistency Regularization (2025)**[^1_2]

- **접근법**: 시간-주파수 마이닝과 일관성 정규화를 결합하여 few-shot 다변량 시계열 예측
- **WaveBound와의 차이**: Consistency regularization은 약한 섭동 기법을 통해 데이터를 증강하고 입력 데이터 변화에 대한 안정적 예측을 보장하지만, WaveBound는 훈련 손실 자체에 동적 한계를 설정
- **장점**: 제한된 데이터 상황에서 효과적
- **한계**: 증강 데이터 생성에 추가 계산 필요

**2. Attention Logic Regularization (Attn-L-Reg, 2025)**[^1_3][^1_4]

- **접근법**: 논리적 관점에서 attention map을 희소하게 만들어 효과적인 토큰 의존성 학습
- **핵심 아이디어**: 모든 토큰 의존성을 동등하게 다루지 않고, attention map의 희소성을 통해 더 적지만 효과적인 의존성 사용
- **WaveBound와의 관계**: 둘 다 모델이 모든 정보에 동등하게 적합되는 것을 방지하지만, Attn-L-Reg는 attention 메커니즘에 집중하고 WaveBound는 손실 함수에 집중
- **플러그인 가능**: Transformer 기반 모델에 쉽게 적용 가능

**3. Robust Time-series Forecasting with Regularized INR Basis (2025)**[^1_5]

- **접근법**: 시간 인덱스 기반 모델(DeepTime)에 정규화 항을 추가하여 강건성 향상
- **정규화 목표**: 시간 인덱스 기반 요소들이 더 단위 표준화되고 상호 상관이 적도록 유도
- **장점**: 테스트 시간에 룩백 윈도우에 결측값이 있을 때 더 탄력적이고, 더 높은 주파수 데이터에 적용 시 예측 정확도 향상
- **WaveBound와의 차이**: INR 기반 모델에 특화되어 있으며, 기저 함수의 특성에 집중

**4. Root Purge and Rank Reduction (2025)**[^1_6]

- **접근법**: 선형 시계열 예측 모델의 특성근(characteristic roots) 분석을 통한 정규화
- **핵심 발견**: 노이즈가 있는 상황에서 모델이 허위 근(spurious roots)을 생성하는 경향
- **제안 방법**:
    - Rank reduction 기법 (Reduced-Rank Regression, Direct Weight Rank Reduction)
    - Root Purge: 훈련 중 노이즈를 억제하는 null space 학습을 장려하는 적응형 방법
- **WaveBound와의 차이**: 선형 모델에 특화되어 있으며, 모델 가중치의 내재적 구조에 집중


### Transformer 기반 모델의 과적합 문제

**5. Benign Overfitting in Transformers (2024)**[^1_7][^1_8]

- **핵심 발견**: 선형 변환기가 레이블 플리핑 노이즈가 있는 컨텍스트 내 예제를 모두 기억하면서도 깨끗한 테스트 예제에 대해 거의 최적으로 일반화할 수 있음
- **조건**: 특징이 고차원 공간에 있고 SNR이 상대적으로 작을 때 발생
- **시사점**: 과적합이 항상 나쁜 것은 아니며, 특정 조건에서는 "benign overfitting"이 가능함을 보여줌
- **WaveBound와의 관계**: WaveBound는 해로운 과적합을 방지하는 데 집중하지만, 이 연구는 과적합의 양성 형태를 탐구


### 도메인 일반화 및 비정상성 처리

**6. Domain Generalization (2024)**[^1_9]

- **접근법**: 도메인 공유 및 도메인 특화 잠재 요인을 분리하여 시계열 예측의 도메인 일반화 개선
- **핵심 기법**: 디코더 조건부 설계와 도메인 정규화를 통해 도메인 정보 학습 강화
- **적용 범위**: 웹 트래픽, 전자상거래, 금융, 전력 소비 데이터에서 검증
- **WaveBound와의 보완성**: WaveBound는 단일 도메인 내에서 과적합을 방지하고, 이 방법은 여러 도메인 간 일반화에 집중

**7. FredNormer (2024)**[^1_10]

- **접근법**: 주파수 도메인 정규화를 통해 비정상 시계열 예측 처리
- **핵심 아이디어**: 시간 도메인에서 작동하는 정규화 방법은 주파수 도메인에서 더 명확한 동적 패턴을 완전히 포착하지 못할 수 있음
- **WaveBound와의 차이**: WaveBound는 시간 도메인의 훈련 손실에 집중하지만, FredNormer는 주파수 도메인의 특성을 활용


### 데이터 증강 및 잔차 학습

**8. Data Augmentation Policy Search (TSAA, 2025)**[^1_11]

- **접근법**: 장기 예측을 위한 시계열 자동 증강 방법
- **효율성**: 효율적이고 구현하기 쉬우며, 이단계 프로세스를 통해 이중 레벨 최적화 문제 해결
- **WaveBound와의 결합 가능성**: TSAA는 데이터 레벨에서, WaveBound는 훈련 프로세스 레벨에서 작동하므로 상호 보완 가능

**9. Minusformer (2024)**[^1_12]

- **접근법**: 잔차를 점진적으로 학습하는 이중 스트림 및 감산 메커니즘 도입
- **핵심**: 시계열의 본질적 가치를 미래 구간에 점진적으로 복원하는 탈중복화 접근법
- **관계**: Deep Boosting 앙상블 학습과 유사한 철학
- **WaveBound와의 차이**: 아키텍처 레벨의 변경이며, WaveBound는 정규화 기법


### Foundation 모델 및 Zero-shot 학습

**10. TimesFM (2024)**[^1_13]

- **접근법**: 디코더 전용 foundation 모델로 시계열 예측
- **특징**: 다양한 공개 데이터셋에서 즉시 사용 가능한 zero-shot 성능이 각 개별 데이터셋에 대한 최신 지도 학습 모델의 정확도에 근접
- **패치 디코더 스타일 attention 모델**: 대규모 시계열 코퍼스에 사전 훈련
- **WaveBound의 적용 가능성**: Foundation 모델의 미세 조정 단계에서 WaveBound 적용 가능

**11. FlexTSF (2024)**[^1_14]

- **접근법**: 규칙성이 다양한 시계열을 위한 범용 예측 모델
- **특징**: 정규 및 불규칙 시계열 모두에 적용 가능
- **자기 지도 사전 훈련**: zero-shot 및 few-shot 설정에서 뛰어난 성능
- **WaveBound와의 관계**: WaveBound는 few-shot 설정에서도 효과적임이 입증되어 이러한 foundation 모델과 결합 가능


### 경량화 및 희소성

**12. SparseTSF (2025)**[^1_15]

- **접근법**: 교차 주기 희소 예측 기법을 통한 초경량 장기 시계열 예측
- **핵심**: 원본 시퀀스를 다운샘플링하여 교차 주기 트렌드 예측에 집중
- **장점**: 1,000개 미만의 파라미터로 최신 방법과 경쟁력 있는 성능
- **암묵적 정규화**: 다운샘플링 자체가 암묵적 정규화 메커니즘으로 작동하여 모델의 강건성 향상
- **WaveBound와의 차이**: SparseTSF는 모델 복잡도를 줄이는 반면, WaveBound는 훈련 프로세스를 정규화


### 실용적 적용 연구

**13. Elastic Net with Gaussian Decay (2025)**[^1_16]

- **적용 분야**: S\&P 500, Dow Jones, Nasdaq 지수 예측 (2020-2025)
- **개선점**: 가우시안 가중치 감쇠를 도입하여 마지막 역사적 관측값과 첫 예측 간의 급격한 "점프" 완화
- **결과**: S\&P 500과 Nasdaq에서 가장 낮은 RMSE 달성

**14. LSTM with Time Series Cross-Validation (2025)**[^1_17]

- **적용 분야**: 인도네시아 은행 주식 예측 (2020-2025)
- **핵심**: 5-fold TSCV를 사용하여 데이터 누수 방지 및 시간적 무결성 유지
- **정규화**: 드롭아웃 정규화를 사용한 이중 LSTM 레이어
- **결과**: BBCA R² > 0.95, MAPE 2.34%


## 종합 비교 및 WaveBound의 위치

WaveBound는 2020년 이후 시계열 예측 정규화 연구에서 다음과 같은 독특한 위치를 차지합니다:

### 차별화 요소

1. **동적 오차 한계**: 대부분의 정규화 방법이 고정된 정규화 항을 사용하는 반면, WaveBound는 EMA 타겟 네트워크를 통해 동적으로 오차 한계를 조정[^1_1]
2. **시간 단계 및 특징별 개별화**: Attn-L-Reg가 attention에 집중하고, Root Purge가 가중치 구조에 집중하는 반면, WaveBound는 각 시간 단계와 특징에 대해 개별적으로 오차 한계를 설정[^1_1]
3. **모델 불가지론적**: SparseTSF나 Minusformer가 특정 아키텍처 변경을 요구하는 반면, WaveBound는 다양한 시계열 예측 모델에 플러그인 방식으로 적용 가능[^1_1]
4. **이론적 보장**: MSE 감소에 대한 명확한 이론적 보장을 제공하며, Theorem 1을 통해 wave empirical risk 추정기가 경험적 위험 추정기보다 나은 조건을 명시[^1_1]

### 상호 보완적 접근법

WaveBound는 다음 방법들과 결합하여 더 나은 성능을 달성할 수 있습니다:

- **RevIN + WaveBound**: 분포 이동과 과적합을 동시에 해결[^1_1]
- **Attn-L-Reg + WaveBound**: attention 메커니즘과 훈련 손실을 동시에 정규화
- **Data Augmentation + WaveBound**: 데이터 레벨과 훈련 프로세스 레벨에서 동시에 작동
- **Foundation Models + WaveBound**: 대규모 사전 훈련 후 미세 조정 단계에서 과적합 방지


### 미래 연구 방향

WaveBound의 성공은 시계열 예측을 위한 특화된 정규화의 중요성을 입증하며, 다음과 같은 연구 방향을 제시합니다:

1. **적응형 메타 학습**: 데이터셋 특성에 따라 자동으로 $\epsilon$과 $\alpha$를 학습하는 메타 학습 프레임워크
2. **주파수 도메인 WaveBound**: FredNormer의 아이디어를 결합하여 주파수 도메인에서도 동적 오차 한계 적용
3. **멀티태스크 WaveBound**: 여러 예측 작업을 동시에 수행할 때 각 작업별로 다른 오차 한계 적용
4. **설명 가능한 WaveBound**: 학습된 오차 한계를 통해 시계열의 예측 가능성과 불확실성을 정량화하고 해석

WaveBound는 시계열 예측의 과적합 문제를 해결하는 강력하고 유연한 프레임워크를 제공하며, 향후 연구에서 다른 최신 기법들과 결합하여 더욱 강건하고 정확한 시계열 예측 시스템을 구축하는 데 기여할 것으로 기대됩니다.[^1_2][^1_3][^1_6][^1_1]
<span style="display:none">[^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36]</span>

<div align="center">⁂</div>

[^1_1]: 2210.14303v2.pdf

[^1_2]: https://www.nature.com/articles/s41598-025-99339-4

[^1_3]: https://arxiv.org/abs/2503.06867

[^1_4]: https://arxiv.org/pdf/2503.06867.pdf

[^1_5]: https://openreview.net/forum?id=uDRzORdPT7

[^1_6]: https://arxiv.org/abs/2509.23597

[^1_7]: https://arxiv.org/html/2410.01774v2

[^1_8]: https://arxiv.org/abs/2410.01774

[^1_9]: http://arxiv.org/pdf/2412.11171.pdf

[^1_10]: http://arxiv.org/pdf/2410.01860.pdf

[^1_11]: http://arxiv.org/pdf/2405.00319.pdf

[^1_12]: http://arxiv.org/pdf/2402.02332.pdf

[^1_13]: https://arxiv.org/pdf/2310.10688.pdf

[^1_14]: http://arxiv.org/pdf/2410.23160.pdf

[^1_15]: https://ieeexplore.ieee.org/document/11141354/

[^1_16]: https://oeipt.vntu.edu.ua/index.php/oeipt/article/view/771

[^1_17]: https://jurnal.polibatam.ac.id/index.php/JAIC/article/view/11314

[^1_18]: https://arxiv.org/pdf/2601.15514.pdf

[^1_19]: https://arxiv.org/pdf/2510.06466.pdf

[^1_20]: https://arxiv.org/html/2506.14831v2

[^1_21]: https://arxiv.org/html/2601.05975v1

[^1_22]: https://pdfs.semanticscholar.org/75cf/b5426bdb701f29a14693ae8bb55eddab76ea.pdf

[^1_23]: https://ar5iv.labs.arxiv.org/html/2210.14303

[^1_24]: https://arxiv.org/html/2512.10913v1

[^1_25]: https://arxiv.org/abs/2210.14303

[^1_26]: https://iaj.aktuaris.or.id/index.php/iaj/article/view/28

[^1_27]: https://linkinghub.elsevier.com/retrieve/pii/S1568494625010920

[^1_28]: https://ieeexplore.ieee.org/document/10992348/

[^1_29]: https://dl.acm.org/doi/10.1145/3785706.3785755

[^1_30]: https://pmc.ncbi.nlm.nih.gov/articles/PMC7660543/

[^1_31]: https://arxiv.org/pdf/2312.17100.pdf

[^1_32]: https://www.sciencedirect.com/science/article/abs/pii/S1568494625010920

[^1_33]: https://dl.acm.org/doi/10.1145/3637528.3671969

[^1_34]: https://openreview.net/pdf/c73ec02e25c23af95e8142e6808b6eb35a6f9334.pdf

[^1_35]: https://proceedings.neurips.cc/paper_files/paper/2022/file/7b99e3c648898b9e4923dea0aeb4afa1-Paper-Conference.pdf

[^1_36]: https://www.reddit.com/r/AIMadeSimple/comments/16v580q/why_do_transformers_suck_at_time_series/

