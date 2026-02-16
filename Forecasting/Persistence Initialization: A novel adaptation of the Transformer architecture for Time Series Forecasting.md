# Persistence Initialization: A novel adaptation of the Transformer architecture for Time Series Forecasting

## 핵심 주장과 주요 기여

Persistence Initialization은 Transformer를 시계열 예측에 적합하게 만드는 혁신적인 초기화 방법을 제안합니다. 이 논문의 핵심은 단일 Transformer 모델이 복잡한 앙상블 방법들과 경쟁할 수 있는 성능을 달성하면서도, naive persistence model(단순 지속성 모델)로 초기화하여 학습을 가속화하고 일반화 성능을 향상시킨다는 것입니다.[^1_1]

주요 기여는 다음과 같습니다:

- **Persistence Initialization 기법**: 영으로 초기화된 곱셈 게이팅 메커니즘과 잔차 연결을 결합하여 모델을 naive random walk model로 초기화[^1_1]
- **단일 모델의 우수한 성능**: M4 데이터셋에서 단일 Transformer가 대규모 앙상블과 경쟁 가능한 성능 달성(OWA 0.815)[^1_1]
- **구조적 최적화 검증**: Rotary 인코딩과 ReZero 정규화의 중요성을 실증적으로 입증[^1_1]


## 해결하고자 하는 문제

### 문제 정의

기존 시계열 예측에서는 앙상블 기반 딥러닝 모델이 높은 정확도를 보였지만, 실제 환경에서는 배포 복잡성과 모델 크기로 인해 비실용적입니다. 특히 M4 competition 우승 모델은 6-9개의 복잡한 계층적 앙상블을 사용했으며, N-BEATS는 180개의 MLP 앙상블을 활용했습니다. Transformer는 다양한 도메인에서 성공했지만, 시계열 예측에는 상대적으로 덜 연구되었습니다.[^1_1]

### 주요 도전 과제

- 단일 모델의 낮은 표현력
- 깊은 Transformer의 학습 불안정성
- 시계열 데이터의 국소적 패턴 포착의 어려움
- 모델 크기 증가에 따른 성능 저하


## 제안하는 방법

### Persistence Initialization

모델의 핵심은 Equation 3으로 정의됩니다:[^1_1]

$h(z) = z + \alpha \cdot g(z)$

여기서:

- $z$는 정규화된 입력 시계열
- $g(z)$는 Transformer 변환
- $\alpha$는 0으로 초기화된 학습 가능한 스칼라 게이팅 파라미터


### 입력 정규화

입력 정규화는 다음과 같이 수행됩니다:[^1_1]

$z = f(x) = \ln\left(\frac{x}{\mu_H(x)}\right)$

여기서 $\mu_H(x)$는 가장 최근 $H$개 값의 평균이며, 로그 변환은 양수 출력을 보장합니다.[^1_1]

### Transformer 아키텍처

시계열을 특징 벡터로 변환하는 과정:[^1_1]

$g(z) = \text{Transformer}(zW_{in})W_{out}$

여기서 $W_{in} \in \mathbb{R}^{1 \times d_{model}}$, $W_{out} \in \mathbb{R}^{d_{model} \times 1}$입니다.[^1_1]

### ReZero와 Rotary 인코딩

각 레이어의 정의:[^1_1]

$X_l = \text{FF}(\text{SA}(X_{l-1}))$

$\text{SA}(X) = X + \alpha_l \cdot \text{SelfAttention}(X)$

$\text{FF}(X) = X + \alpha_l \cdot \text{FeedForward}(X)$

Self-Attention with Rotary encoding:[^1_1]

$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{\tilde{Q}\tilde{K}^T}{\sqrt{d_{qk}}}\right)V$

여기서 $\tilde{Q}$와 $\tilde{K}$는 Rotary positional encoding이 적용된 행렬입니다.[^1_1]

### 손실 함수: MASE

모델은 MASE(Mean Absolute Scaled Error)를 손실 함수로 사용합니다:[^1_1]

```math
\text{mase} = \frac{1}{N}\sum_{i=1}^{N}\frac{\frac{1}{H}\sum_{j=1}^{H}|y_j^{(i)} - \hat{y}_j^{(i)}|}{\frac{1}{T^{(i)}-S^{(i)}}\sum_{j=S^{(i)}+1}^{T^{(i)}}|x_j^{(i)} - x_{j-S^{(i)}}^{(i)}|}
```

여기서 $S^{(i)}$는 시계열 $i$의 계절성입니다.[^1_1]

## 모델 구조

### 전체 아키텍처

모델은 decoder-only Transformer 구조를 사용하며, 다음과 같은 구성 요소로 이루어집니다:[^1_1]

- **4개의 Transformer 레이어**: 각 레이어는 causal self-attention과 feedforward 네트워크로 구성
- **4개의 attention heads**: multi-head attention 메커니즘
- **$d_{model} = 512$**: 모델 차원
- **$d_{ff} = 2048$**: feedforward 네트워크 차원


### 핵심 설계 특징

**Channel-independent 처리**: 각 채널을 독립적으로 처리하여 계산 효율성 향상[^1_1]

**Windowing 전략**: 입력 윈도우 크기는 $n \times H$로 설정되며, $n=3$ (Yearly, Quarterly, Monthly, Daily) 또는 $n=4$ (Weekly, Hourly)[^1_1]

**Autoregressive 예측**: 훈련 시에는 teacher forcing을 사용하고, 테스트 시에는 autoregressive 방식으로 예측 생성[^1_1]

## 성능 향상

### M4 Dataset 결과

논문은 M4 데이터셋의 100,000개 시계열에서 다음과 같은 결과를 달성했습니다:[^1_1]


| 모델 | Total OWA | 앙상블 크기 |
| :-- | :-- | :-- |
| M4 Rank 1 (Smyl) | 0.821 | 6-9 |
| M4 Rank 2 (FFORMA) | 0.838 | 9 |
| N-BEATS-180 | 0.795 | 180 |
| PI-Transformer (단일) | **0.815** | 1 |
| PI-Transformer (앙상블) | **0.800** | 9 |

단일 PI-Transformer 모델이 M4 competition 우승자를 능가했으며, 9개 모델의 평균 앙상블은 180개 모델의 N-BEATS와 유사한 성능을 보였습니다.[^1_1]

### Ablation Study 결과

**Skip connection과 Gating의 효과**:[^1_1]

- Skip connection과 gating을 모두 사용할 때만 모델 크기 증가에 따라 성능이 향상
- Gating 없이는 성능이 정체되고, skip connection 없이는 큰 모델에서 성능 저하

**정규화와 위치 인코딩의 영향**:[^1_1]

- Rotary encoding이 sinusoidal encoding보다 모든 모델 크기에서 우수
- ReZero normalization만이 모델 크기 증가 시 성능 향상을 보임
- Layer Normalization(Pre/Post)은 큰 모델에서 성능 저하


### 다른 Transformer 모델과의 비교

Hourly 데이터에서의 성능 비교:[^1_1]


| 모델 | OWA | R0.5 |
| :-- | :-- | :-- |
| LogSparse Transformer | - | 0.067 |
| Informer | 0.670 | 0.056 |
| Autoformer | 1.033 | 0.078 |
| **PI-Transformer** | **0.525** | **0.046** |

## 모델의 한계

### 명시된 한계점

**단일 주파수 최적화 부족**: 하이퍼파라미터는 주로 Monthly 데이터에서 수동으로 탐색되었으며, Weekly와 Hourly는 윈도우 크기 조정만 수행[^1_1]

**계산 복잡도**: Transformer의 quadratic attention complexity는 여전히 매우 긴 시퀀스에서 제한적[^1_1]

**단순 앙상블 전략**: 평균 예측만 사용했으며, 더 정교한 앙상블 기법 미탐색[^1_1]

### 암묵적 한계

**One-shot forecasting 미지원**: Informer와 Autoformer처럼 전체 horizon을 한 번에 예측하지 않아 추론 속도가 느릴 수 있음[^1_1]

**제한된 데이터셋 평가**: M4에서만 주로 평가되었으며, 다른 도메인(전력, 교통 등)에서의 일반화 검증 부족[^1_1]

**해석 가능성 부족**: Attention map의 의미론적 해석이나 학습된 패턴에 대한 심층 분석 부재[^1_1]

## 모델의 일반화 성능 향상 가능성

### 이론적 근거

**Inductive bias 제공**: Persistence initialization은 시계열의 기본적인 귀납적 편향(시간적 연속성)을 모델에 주입합니다. 초기 모델이 $\hat{x}_{t+1} = x_t$를 예측하도록 설정되어, 모델은 이 baseline에서 시작하여 점진적으로 복잡한 패턴을 학습합니다.[^1_1]

**학습 안정성**: 그림 3의 결과에 따르면, Persistence Initialization을 사용한 모델은 훈련 손실의 표준편차가 현저히 작아 더 안정적인 학습을 보입니다. 안정적인 학습은 overfitting을 줄이고 일반화를 향상시킵니다.[^1_1]

**모델 용량 활용**: Ablation study는 Persistence Initialization이 있을 때만 모델 크기 증가($d_{model}$: 32→512)가 성능 향상으로 이어짐을 보여줍니다. 이는 더 큰 모델 용량이 일반화에 효과적으로 활용됨을 의미합니다.[^1_1]

### 실증적 증거

**Cross-frequency 일반화**: 동일한 하이퍼파라미터 설정이 6개의 서로 다른 주파수(Yearly, Quarterly, Monthly, Weekly, Daily, Hourly)에서 모두 경쟁력 있는 성능을 달성했습니다. 특히:[^1_1]

- Hourly: OWA 0.431 (최고 성능)
- Weekly: OWA 0.733
- Daily: OWA 0.987

**다양한 시계열 길이**: M4 데이터셋은 최소 19개부터 최대 9,933개의 관측치를 가진 시계열을 포함하며, 단일 모델이 이 모든 범위에서 작동합니다.[^1_1]

**Zero-shot 전이 가능성**: Channel-independent 설계는 다른 변수 수를 가진 데이터셋으로의 전이를 용이하게 합니다. 각 변수가 동일한 가중치를 공유하므로, 새로운 도메인에 대한 재학습 없이 적용 가능성이 있습니다.[^1_1]

### 일반화 메커니즘

**Residual learning**: Skip connection은 모델이 identity function으로부터의 잔차를 학습하게 하여, 작은 변화에 집중하고 과도한 변환을 방지합니다.[^1_1]

$h(z) = z + \alpha \cdot g(z)$

여기서 초기 $\alpha = 0$이므로 모델은 점진적으로 복잡도를 증가시킵니다.[^1_1]

**Rotary encoding의 상대적 위치 정보**: 절대 위치 대신 상대적 위치를 인코딩하여, 훈련 중 보지 못한 위치에서도 일반화가 가능합니다. 이는 windowed time series에서 특히 중요합니다.[^1_1]

**ReZero의 deep network 안정화**: ReZero normalization은 100개 이상의 레이어를 가진 Transformer를 안정적으로 학습시킬 수 있으며, 이는 더 깊은 모델로의 확장 가능성을 시사합니다.[^1_1]

## 앞으로의 연구에 미치는 영향

### 방법론적 영향

**초기화 전략의 재평가**: 이 논문은 도메인 지식을 활용한 초기화가 복잡한 아키텍처 수정보다 효과적일 수 있음을 보여줍니다. 향후 연구는 다른 시계열 작업(분류, 이상 탐지)에도 유사한 초기화 전략을 탐색할 수 있습니다.[^1_1]

**앙상블 의존도 감소**: 단일 모델로 대규모 앙상블과 경쟁 가능한 성능을 달성함으로써, 실용적인 배포를 위한 새로운 방향을 제시합니다. 이는 산업 응용에서 중요한 의미를 갖습니다.[^1_1]

**Component-wise 설계 중요성**: Rotary encoding과 ReZero normalization의 조합이 핵심임을 보여주어, 향후 시계열 Transformer 설계 시 이러한 구성 요소를 표준으로 고려해야 합니다.[^1_1]

### 이론적 영향

**Inductive bias와 일반화**: Persistence initialization은 적절한 inductive bias가 일반화에 미치는 영향을 실증적으로 보여줍니다. 이는 시계열의 시간적 연속성이라는 기본 원리를 모델 아키텍처에 통합하는 중요성을 강조합니다.[^1_1]

**학습 역학 이해**: 그림 3에서 보듯이 초기화 방법이 학습 안정성과 수렴 속도에 미치는 영향을 명확히 제시합니다. 이는 deep learning 최적화 연구에 기여합니다.[^1_1]

### 실용적 영향

**계산 효율성**: 단일 모델 사용으로 추론 비용과 메모리 사용량을 크게 줄일 수 있어, edge device나 실시간 예측 시스템에 적용 가능성이 높습니다.[^1_1]

**하이퍼파라미터 강건성**: 다양한 주파수에서 동일한 설정이 작동하여, 실무자들이 광범위한 튜닝 없이 모델을 적용할 수 있습니다.[^1_1]

## 앞으로 연구 시 고려할 점

### 아키텍처 개선 방향

**Sparse attention 통합**: LogSparse Transformer의 아이디어를 Persistence Initialization과 결합하여 더 긴 컨텍스트를 효율적으로 처리할 수 있습니다. 특히 $O(N \log N)^2$ 복잡도로 개선 가능합니다.[^1_1]

**Adaptive gating**: 현재 단일 스칼라 $\alpha$를 사용하지만, 시간 단계별 또는 채널별로 적응적인 gating을 도입하여 더 세밀한 제어가 가능합니다.

**Hierarchical structure**: 다중 스케일 패턴 포착을 위해 다양한 윈도우 크기를 가진 계층적 구조 탐색이 필요합니다.

### 학습 전략 개선

**Pre-training**: Zerveas et al.의 masked autoencoder 접근법을 Persistence Initialization과 결합하여 대규모 unlabeled 시계열 데이터로부터 일반적인 표현 학습.[^1_2][^1_1]

**Curriculum learning**: 짧은 horizon에서 시작하여 점진적으로 긴 horizon으로 확장하는 커리큘럼 학습 전략이 수렴을 더욱 가속화할 수 있습니다.

**Multi-task learning**: 예측, 분류, 이상 탐지를 동시에 학습하여 더 강건한 표현 획득 가능성 탐색.

### 평가 및 벤치마킹

**다양한 도메인 평가**: 전력 소비(Electricity), 교통(Traffic), 날씨(Weather), 금융 시계열 등 다양한 도메인에서의 성능 검증 필요.[^1_1]

**실시간 성능**: Latency, throughput, 메모리 사용량 등 실시간 배포 시나리오에서의 성능 평가 필요.

**불확실성 정량화**: 점 예측뿐만 아니라 예측 구간(prediction interval)이나 분포 예측 능력 평가.

### 해석 가능성 연구

**Attention pattern 분석**: Rotary encoding을 사용한 attention map이 어떤 시간적 패턴을 포착하는지 시각화 및 해석.

**Gating parameter 진화**: 학습 중 $\alpha$ 값의 변화를 추적하여 모델이 persistence에서 벗어나는 과정 이해.

**Feature importance**: 다변량 시계열에서 각 변수의 기여도를 정량화하는 방법 개발.

### 확장성 연구

**Very long sequence**: 수천~수만 time step을 가진 초장기 시계열에서의 성능과 안정성 검증.

**Foundation model**: 대규모 시계열 corpus로 pre-train된 foundation model 개발 가능성 탐색.

**Transfer learning**: 한 도메인에서 학습한 모델을 다른 도메인으로 전이하는 효과적인 방법 연구.

## 2020년 이후 관련 최신 연구 비교 분석

### PatchTST (2023)

**핵심 아이디어**: 시계열을 subseries-level patch로 분할하고 channel-independent 방식으로 처리합니다.[^1_3][^1_4]

**PI-Transformer와의 비교**:

- **유사점**: 둘 다 channel-independent 접근법 사용하여 계산 효율성 향상[^1_3][^1_1]
- **차이점**: PatchTST는 patch 기반 입력을 사용하지만, PI-Transformer는 point-wise 입력에 persistence initialization 적용[^1_3][^1_1]
- **장점**: PatchTST의 patching은 더 긴 lookback window를 효율적으로 처리하며, 지역적 semantic 정보를 보존[^1_4]
- **성능**: PatchTST는 multiple benchmark에서 SOTA 달성, PI-Transformer는 M4에서 우수[^1_3][^1_1]


### TimesNet (2023)

**핵심 아이디어**: Fast Fourier Transform(FFT)을 사용하여 다중 주기성을 발견하고, 1D 시계열을 2D 텐서로 변환하여 CNN 기반 Inception block으로 처리합니다.[^1_5][^1_6]

**PI-Transformer와의 비교**:

- **아키텍처**: TimesNet은 CNN 기반, PI-Transformer는 Transformer 기반[^1_5][^1_1]
- **주기성 처리**: TimesNet은 명시적으로 주기를 탐지하고 활용하지만, PI-Transformer는 implicit하게 학습[^1_6][^1_1]
- **일반성**: TimesNet은 forecasting, imputation, classification, anomaly detection 등 5개 task에서 SOTA 달성[^1_5]
- **M4 성능**: TimesNet도 M4에서 우수한 성능을 보이며, 특히 diverse source의 시계열에서 강건함[^1_5]

**통합 가능성**: TimesNet의 multi-periodicity detection을 PI-Transformer에 통합하면 계절성 패턴 포착 개선 가능.

### iTransformer (2023)

**핵심 아이디어**: Transformer의 차원을 반전(invert)하여, 시간 포인트를 variate token으로 임베딩하고 attention으로 다변량 상관관계 포착.[^1_7][^1_8]

**PI-Transformer와의 비교**:

- **차원 처리**: iTransformer는 변수 간 관계에 attention 적용, PI-Transformer는 시간 차원에 attention 적용[^1_8][^1_1]
- **목표**: iTransformer는 다변량 상관관계 명시적 모델링, PI-Transformer는 시간적 의존성에 집중[^1_8][^1_1]
- **성능**: iTransformer는 challenging real-world dataset에서 SOTA, 특히 더 긴 lookback window에서 우수[^1_8]
- **일반화**: iTransformer는 다른 변수 수에 대한 일반화 능력 향상[^1_8]

**보완 관계**: 두 접근법은 상호 보완적이며, 시간과 변수 차원 모두를 효과적으로 모델링하는 hybrid 아키텍처 가능.

### Informer \& Autoformer (2021)

**Informer**: ProbSparse attention으로 $O(L \log L)$ 복잡도 달성, 48+ horizon의 장기 예측에 초점.[^1_9][^1_1]

**Autoformer**: Dot-product attention을 auto-correlation 메커니즘으로 대체.[^1_1]

**PI-Transformer와의 비교**:

- **성능**: PI-Transformer가 M4-Hourly에서 Informer(OWA 0.670)와 Autoformer(OWA 1.033)를 크게 능가(OWA 0.525)[^1_1]
- **접근법**: Informer/Autoformer는 attention 효율성에 집중, PI-Transformer는 초기화 전략에 집중[^1_1]
- **예측 방식**: Informer/Autoformer는 one-shot forecasting, PI-Transformer는 autoregressive[^1_1]


### Recent Trends (2024-2025)

**LSEAttention (2024)**: Entropy collapse와 training instability 문제 해결에 초점. PI-Transformer의 ReZero와 유사한 동기를 가지며, 안정적인 학습을 위한 attention 메커니즘 개선.[^1_10]

**CT-PatchTST (2025)**: PatchTST를 확장하여 channel과 temporal 정보를 동시 통합. PI-Transformer의 channel-independent 접근과 대조적.[^1_11][^1_2]

**LATST (2025)**: Transformer의 entropy collapse 완화에 집중. PI-Transformer의 persistence initialization과 결합하면 더 강건한 학습 가능.[^1_10]

**PSformer (2025)**: Parameter sharing과 Spatial-Temporal Segment Attention으로 parameter 효율성 개선. PI-Transformer에 적용하면 모델 크기 감소 가능.[^1_12]

### 종합 비교표

| 모델 | 연도 | 핵심 기법 | 주요 강점 | M4 성능 |
| :-- | :-- | :-- | :-- | :-- |
| PI-Transformer | 2022 | Persistence Init | 단일 모델 우수 성능, 앙상블 대체 | OWA 0.815 |
| PatchTST | 2023 | Patch-based | 효율적 긴 시퀀스 처리 | 보고 안 됨 |
| TimesNet | 2023 | FFT + 2D CNN | Multi-task SOTA | 우수 |
| iTransformer | 2023 | Inverted dims | 다변량 상관관계 | 보고 안 됨 |
| Informer | 2021 | Sparse attention | 장기 예측 효율성 | OWA 0.670 (hourly) |
| Autoformer | 2021 | Auto-correlation | 주기성 포착 | OWA 1.033 (hourly) |

### 향후 통합 방향

**Hybrid architecture**: PI-Transformer의 initialization + PatchTST의 patching + TimesNet의 periodicity detection을 결합한 통합 모델이 각 접근법의 장점을 최대화할 수 있습니다.

**Foundation model**: 2024-2025 연구들은 대규모 pre-training으로 이동하고 있으며, PI-Transformer의 안정적인 학습 특성은 대규모 모델 학습에 유리합니다.[^1_13]

**Efficiency vs. Accuracy**: 최신 연구들은 parameter efficiency와 성능의 균형을 추구하며, PI-Transformer는 단일 모델로 경쟁력 있는 성능을 보여 이 방향과 일치합니다.[^1_12]
<span style="display:none">[^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27]</span>

<div align="center">⁂</div>

[^1_1]: 2208.14236v1.pdf

[^1_2]: https://arxiv.org/html/2501.08620v1

[^1_3]: http://arxiv.org/pdf/2211.14730v2.pdf

[^1_4]: https://huggingface.co/blog/patchtst

[^1_5]: https://ar5iv.labs.arxiv.org/html/2210.02186

[^1_6]: https://www.datasciencewithmarco.com/blog/timesnet-the-latest-advance-in-time-series-forecasting

[^1_7]: https://arxiv.org/html/2510.07084v1

[^1_8]: https://arxiv.org/abs/2310.06625

[^1_9]: https://towardsdatascience.com/influential-time-series-forecasting-papers-of-2023-2024-part-1-1b3d2e10a5b3/

[^1_10]: http://arxiv.org/pdf/2410.23749.pdf

[^1_11]: https://arxiv.org/html/2501.08620v4

[^1_12]: https://arxiv.org/html/2411.01419v1

[^1_13]: https://arxiv.org/html/2507.02907v1

[^1_14]: https://arxiv.org/abs/2207.05397

[^1_15]: http://arxiv.org/pdf/2408.09723.pdf

[^1_16]: https://arxiv.org/pdf/2401.13968.pdf

[^1_17]: https://arxiv.org/pdf/2502.13721.pdf

[^1_18]: https://arxiv.org/html/2411.01623

[^1_19]: https://arxiv.org/abs/2311.04147

[^1_20]: https://arxiv.org/html/2508.16641v1

[^1_21]: https://peerj.com/articles/cs-3001/

[^1_22]: https://www.semanticscholar.org/paper/Make-Transformer-Great-Again-for-Time-Series-Robust-Wang-Zhou/156b823384f49770c170319e0a1e6751130c3919

[^1_23]: https://arxiv.org/html/2502.09683v1

[^1_24]: https://openreview.net/forum?id=kHEVCfES4Q\&noteId=mrNbq9EkQa

[^1_25]: https://www.sciencedirect.com/science/article/pii/S1574013725001595

[^1_26]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12453695/

[^1_27]: https://ise.thss.tsinghua.edu.cn/~mlong/doc/TimesNet-iclr23.pdf

