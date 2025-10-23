# Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting

## 논문의 핵심 주장과 주요 기여[1]

본 논문은 시계열 예측에서 **과도한 정상화(Over-stationarization)** 문제를 처음으로 규명하고 이를 해결하기 위한 혁신적인 프레임워크를 제시합니다. 기존 방법들이 시계열의 비정상성을 완전히 제거하여 예측력을 해치는 반면, Non-stationary Transformers는 정상화의 이점과 원본 데이터의 비정상 정보를 동시에 활용합니다. 주요 기여는 세 가지입니다: (1) 과도한 정상화 문제 규명, (2) Series Stationarization과 De-stationary Attention으로 구성된 일반적 프레임워크 제시, (3) 여러 Transformer 변형에 적용 가능함을 입증.[1]

## 해결 대상 문제 및 제안 방법

### 문제 정의[1]

실세계 시계열은 **비정상성(Non-stationarity)**을 특징으로 하며, 이는 평균과 표준편차가 시간에 따라 변한다는 의미입니다. 기존 Transformer 기반 모델들이 이러한 데이터에서 성능이 저하되는 이유는 두 가지입니다:[1]

1. **분포 변화로 인한 일반화 어려움**: 딥 러닝 모델이 변하는 분포에서 잘 학습되지 않음
2. **과도한 정상화로 인한 정보 손실**: 기존 정상화 방법들이 비정상 정보를 완전히 제거하여 모델이 구별되는 시간 의존성을 학습하지 못함

Figure 1의 시각화에서 보듯이, 정상화된 시계열로 학습한 Transformer는 서로 다른 시계열에 대해 동일한 시간 주의(temporal attention)를 생성하게 되어 예측력이 심각하게 저하됩니다.[1]

### Series Stationarization[1]

정상화 모듈은 다음 수식으로 표현됩니다:

$$
\mu_x = \frac{1}{S}\sum_{i=1}^{S}x_i, \quad \sigma_x^2 = \frac{1}{S}\sum_{i=1}^{S}(x_i - \mu_x)^2, \quad x'_i = \frac{1}{\sigma_x} \odot (x_i - \mu_x)
$$

여기서 $$\mu_x, \sigma_x \in \mathbb{R}^{C \times 1}$$이고, $$\odot$$는 원소별(element-wise) 곱입니다. 역정규화 모듈은:[1]

$$
\hat{y}_i = \sigma_x \odot y'_i + \mu_x
$$

이 설계는 추가 학습 가능한 매개변수가 없으면서도 RevIN만큼 효과적입니다.[1]

### De-stationary Attention - 핵심 메커니즘[1]

이것이 논문의 가장 중요한 혁신입니다. 표준 Self-Attention은:

$$
\text{Attn}(Q, K, V) = \text{Softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

정상화된 입력 $$x' = (x - 1\mu_x^\top)/\sigma_x$$로부터 도출되는 정규화된 쿼리는 $$Q' = (Q - 1\mu_Q^\top)/\sigma_x$$입니다. 논문은 정상화되지 않은 데이터에서의 주의를 다음과 같이 근사할 수 있음을 증명합니다:[1]

$$
\text{Softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) = \text{Softmax}\left(\frac{\sigma_x^2 Q'K'^\top + 1\mu_Q^\top K^\top}{\sqrt{d_k}}\right)
$$

이를 통해 **비정상 인자(de-stationary factors)** 두 가지를 정의합니다:[1]

- **스케일링 인자**: $$\tau = \sigma_x^2 \in \mathbb{R}^+$$
- **시프팅 인자**: $$\Delta = K\mu_Q \in \mathbb{R}^{S \times 1}$$

최종적으로 De-stationary Attention은:

$$
\log \tau = \text{MLP}(\sigma_x, x), \quad \Delta = \text{MLP}(\mu_x, x)
$$

$$
\text{Attn}(Q', K', V', \tau, \Delta) = \text{Softmax}\left(\frac{\tau Q'K'^\top + 1\Delta^\top}{\sqrt{d_k}}\right)V'
$$

이 두 인자는 모든 주의 계층에서 공유됩니다.[1]

## 모델 구조[1]

표준 Encoder-Decoder 구조를 채택하며, 다음과 같은 특징을 가집니다:

- **인코더**: 과거 관측값으로부터 정보 추출 (2개 층)
- **디코더**: 과거 정보 통합 및 예측 정제 (1개 층)
- **Series Stationarization**: 입력과 출력 모두에 적용
- **De-stationary Attention**: 모든 Self-Attention 메커니즘을 대체

알고리즘 4에서 보듯이, 전체 파이프라인은 정규화 → 특징 추출 → 역정규화 구조로 진행됩니다.[1]

## 성능 향상[1]

6개 실세계 벤치마크에서 검증한 결과:

| 모델 | 평균 MSE 감소율 |
|------|----------------|
| Transformer | 49.43% |
| Informer | 47.34% |
| Reformer | 46.89% |
| Autoformer | 10.57% |

특히 비정상성이 높은 데이터셋에서 놀라운 개선을 보입니다:[1]

- **Exchange**: 17% MSE 감소 (0.509 → 0.421)
- **ILI**: 25% MSE 감소 (2.669 → 2.010)
- **ETTm2**: 30% MSE 감소 (0.598 → 0.417)

유니변량 예측에서도 consistently 우수한 성능을 유지합니다.[1]

## 일반화 성능 향상[1]

### Over-stationarization 문제 해결

Figure 4에서 볼 수 있듯이, 기존 정상화 방법들은 예측값의 정상성 정도가 실제 값보다 비정상적으로 높습니다 (즉, 너무 "부드러운" 예측을 생성). Non-stationary Transformers는 예측 정상성을 실제 값과 매우 유사하게 유지하며 (상대 정상성: 97%-103%), 이는 더 현실적이고 정확한 예측을 의미합니다.[1]

### 구별 가능한 주의 학습

Figure 1의 시각화 비교에서:
- **(a) 기본 Transformer**: 원본 데이터로부터 구별되는 주의 학습 가능
- **(b) Stationarization 포함**: 정상화로 인해 유사한 주의 생성 (문제)
- **(c) Non-stationary Transformer**: 비정상 정보를 복구하여 구별되는 주의 재구성 (해결)[1]

### 일반화 메커니즘

논문의 핵심 통찰은 **분포 변화에 강한 모델 학습**입니다:[1]

1. 정상화는 학습 분포의 안정성을 보장하여 일반화 성능 향상
2. De-stationary Attention은 원본 데이터의 특정 특성을 유지하여 비정상 이벤트 예측 능력 보존

이를 통해 모델은 "학습하기 쉬운" 정상화 데이터로부터 학습하면서도 "예측해야 할" 원본 데이터의 특성을 모두 활용할 수 있습니다.[1]

### 적응형 표현 학습

De-stationary 인자를 MLP로 학습함으로써 모델은 각 시계열의 고유한 비정상 패턴을 동적으로 학습합니다. 이는 다양한 통계적 특성을 가진 시계열들 간의 도메인 격차를 효과적으로 처리합니다.[1]

## 모델의 한계[1]

1. **주의 메커니즘 의존성**: De-stationary Attention은 vanilla Self-Attention 분석으로부터 유도되었으므로, KL-divergence 기반 주의(Informer) 같은 고급 메커니즘에는 최적이 아닐 수 있음

2. **제한적 적용 범위**: 현재 Transformer 기반 모델에만 적용 가능하며, RNN이나 CNN 기반 모델로의 확장 미흡

3. **정상화 방법의 제약**: 차분(differencing)이나 분위수(quantile) 기반 정상화 방법과의 상호작용 미검토

4. **계산 복잡성**: 추가 MLP 계산이 필요하지만, 논문에서는 총 매개변수 증가가 미미하다고 주장 (부록 C.5 참조)[1]

## 향후 연구에 미치는 영향[1]

### 학술적 기여

1. **정상화 문제의 재정의**: 정상화의 "더 많을수록 좋다"는 가정을 의문 제기하고 **과도한 정상화** 개념 도입

2. **분포 변화 일반화 연구**: 시계열 예측에서 분포 이동 문제의 중요성 부각

3. **딥 모델의 내재 메커니즘 분석**: 주의 메커니즘과 정상화의 상호작용을 수식적으로 규명

### 실무 적용

- **기상 예측, 에너지 소비, 금융 위험 평가** 등 실세계 응용에서 Transformers의 실용성 향상
- 기존 모델들의 성능 개선을 위한 경량 플러그-인 솔루션 제공[1]

## 향후 연구 시 고려할 점[1]

1. **모델 무관 솔루션 개발**: Transformers를 넘어 다양한 아키텍처에서 과도한 정상화 문제 해결 방법 모색

2. **다양한 정상화 기법 통합**: 차분, 로그 변환, 계절성 분해 등 여러 정상화 전략과의 통합 연구

3. **비정상 정보의 다른 형태 활용**: De-stationary Attention 외에 feed-forward 계층이나 다른 모듈에 비정상 정보를 재통합하는 방법 탐색

4. **동적 정상화 수준 결정**: 데이터셋 특성에 따라 정상화 강도를 적응적으로 조절하는 메커니즘 개발

5. **시계열 분해와의 결합**: Autoformer처럼 추세-계절 분해와 De-stationary Attention을 더 깊이 있게 통합

6. **장기 예측 안정성**: 예측 길이가 매우 긴 경우 비정상 정보의 동적 변화를 추적하는 방법 연구[1]

이 논문은 시계열 예측에서 정상화라는 기본적인 전처리 기법을 새로운 관점에서 재조명하고, 이론과 실험을 통해 그 한계를 극복하는 우아한 해결책을 제시함으로써 **시계열 예측 분야에 중요한 패러다임 전환**을 가져올 것으로 예상됩니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b4390add-bf7f-4fbc-b375-f3ad266286ab/2205.14415v4.pdf)
