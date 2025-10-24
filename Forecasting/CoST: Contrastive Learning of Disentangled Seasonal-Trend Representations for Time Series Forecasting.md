# CoST: Contrastive Learning of Disentangled Seasonal-Trend Representations for Time Series Forecasting

## 1. 핵심 주장과 주요 기여 요약

CoST는 시계열 예측을 위한 새로운 대조 학습(contrastive learning) 프레임워크로, **분리된 계절성-추세 표현(disentangled seasonal-trend representations)**을 학습합니다. 논문의 핵심 주장은 end-to-end 학습 방식보다 **표현 학습과 예측 과제를 분리**하는 것이 더 효과적이며, 이를 인과적 관점(causal perspective)에서 정당화합니다.[1]

**주요 기여:**
- 인과적 관점에서 분리된 계절성-추세 표현 학습의 필요성 제시[1]
- 시간 도메인과 주파수 도메인 대조 손실을 결합한 CoST 프레임워크 제안[1]
- 다변량 벤치마크에서 MSE 21.3% 개선으로 SOTA 달성[1]
- 다양한 백본 인코더 및 회귀기에 대한 강건성 입증[1]

## 2. 문제 정의 및 제안 방법

### 해결하고자 하는 문제

**1) 얽힌 표현의 문제점**

기존 end-to-end 딥러닝 방법은 관측 데이터에 포함된 예측 불가능한 노이즈의 허위 상관관계(spurious correlations)를 학습할 수 있습니다. 특히 **얽힌 표현(entangled representations)**—하나의 특징 차원이 데이터 생성 프로세스의 여러 독립적 모듈 정보를 인코딩—을 학습하면, 지역적 분포 변화(distribution shift)에 취약합니다.[1]

예를 들어, 계절 모듈에만 분포 변화가 발생해도 얽힌 표현으로는 불변하는 추세 모듈을 활용한 예측이 어렵습니다. 이는 **비정상 환경(non-stationary environment)**에서 일반화 성능 저하를 야기합니다.[1]

### 제안 방법

CoST는 구조적 시계열 모델의 아이디어를 활용하여 시계열을 $$X = T + S + E$$로 분해합니다:[1]
- $$T$$: 추세(Trend)
- $$S$$: 계절성(Seasonality)  
- $$E$$: 오차(Error)

**인과적 정당화:**

오차 변수 $$E$$에 대한 개입(intervention)이 조건부 분포 $$P(X^*|T,S)$$에 영향을 주지 않습니다[1]:

```math
P^{do(E=e_i)}(X^*|T,S) = P^{do(E=e_j)}(X^*|T,S)
```

즉, $$S$$와 $$T$$는 $$E$$의 변화에 불변(invariant)하므로, 이들을 학습하면 다양한 오차 유형에 안정적인 예측을 할 수 있습니다.[1]

### 모델 구조

**전체 프레임워크:**

$$ V = [V^{(T)}; V^{(S)}] \in \mathbb{R}^{h \times d} $$

여기서 $$V^{(T)} \in \mathbb{R}^{h \times d_T}$$는 추세 표현, $$V^{(S)} \in \mathbb{R}^{h \times d_S}$$는 계절성 표현이며, $$d = d_T + d_S$$입니다.[1]

**1) 백본 인코더 (Backbone Encoder)**

TCN(Temporal Convolutional Network)을 사용하여 관측값을 잠재 공간으로 매핑합니다:[1]

$$ \tilde{V} = f_b(X), \quad X \in \mathbb{R}^{h \times m}, \tilde{V} \in \mathbb{R}^{h \times d} $$

**2) 추세 특징 분리기 (Trend Feature Disentangler, TFD)**

자기회귀 전문가의 혼합(mixture of auto-regressive experts)을 사용하여 lookback window 선택 문제를 완화합니다.[1]

$$L+1$$개의 전문가로 구성되며, 각 전문가는 커널 크기 $$2^i$$의 1D causal convolution입니다:[1]

$$ \tilde{V}^{(T,i)} = \text{CausalConv}(\tilde{V}, 2^i) $$

최종 추세 표현은 평균 풀링으로 계산됩니다:[1]

$$ V^{(T)} = \text{AvePool}(\tilde{V}^{(T,0)}, \tilde{V}^{(T,1)}, \ldots, \tilde{V}^{(T,L)}) = \frac{1}{L+1}\sum_{i=0}^{L}\tilde{V}^{(T,i)} $$

**시간 도메인 대조 손실:**

MoCo 방식을 사용하여 추세 표현을 학습합니다:[1]

$$ \mathcal{L}_{\text{time}} = \sum_{i=1}^{N} -\log \frac{\exp(q_i \cdot k_i/\tau)}{\exp(q_i \cdot k_i/\tau) + \sum_{j=1}^{K}\exp(q_i \cdot k_j/\tau)} $$

**3) 계절성 특징 분리기 (Seasonal Feature Disentangler, SFD)**

주파수 도메인에서 계절성 표현을 학습하기 위해 학습 가능한 푸리에 층(learnable Fourier layer)을 사용합니다.[1]

DFT로 주파수 도메인으로 변환 후, 각 주파수에 고유한 복소수 파라미터를 적용합니다:[1]

$$ \mathcal{F}(\tilde{V}) \in \mathbb{C}^{F \times d}, \quad F = \lfloor h/2 \rfloor + 1 $$

학습 가능한 푸리에 층의 $$(i,k)$$번째 출력:[1]

$$ V^{(S)}_{i,k} = \mathcal{F}^{-1}\left(\sum_{j=1}^{d}A_{i,j,k}\mathcal{F}(\tilde{V})_{i,j} + B_{i,k}\right) $$

여기서 $$A \in \mathbb{C}^{F \times d \times d_S}$$, $$B \in \mathbb{C}^{F \times d_S}$$는 학습 가능한 파라미터입니다.[1]

**주파수 도메인 대조 손실:**

진폭과 위상에 대한 별도 손실을 사용합니다:[1]

$$ \mathcal{L}_{\text{amp}} = \frac{1}{FN}\sum_{i=1}^{F}\sum_{j=1}^{N} -\log \frac{\exp(|\mathcal{F}^{(j)}_{i,:}| \cdot |(\mathcal{F}^{(j)}_{i,:})'|)}{\exp(|\mathcal{F}^{(j)}_{i,:}| \cdot |(\mathcal{F}^{(j)}_{i,:})'|) + \sum_{k \neq j}\exp(|\mathcal{F}^{(j)}_{i,:}| \cdot |\mathcal{F}^{(k)}_{i,:}|)} $$

$$ \mathcal{L}_{\text{phase}} = \frac{1}{FN}\sum_{i=1}^{F}\sum_{j=1}^{N} -\log \frac{\exp(\phi(\mathcal{F}^{(j)}_{i,:}) \cdot \phi((\mathcal{F}^{(j)}_{i,:})'))}{\exp(\phi(\mathcal{F}^{(j)}_{i,:}) \cdot \phi((\mathcal{F}^{(j)}_{i,:})')) + \sum_{k \neq j}\exp(\phi(\mathcal{F}^{(j)}_{i,:}) \cdot \phi(\mathcal{F}^{(k)}_{i,:}))} $$

**전체 손실 함수:**

$$ \mathcal{L} = \mathcal{L}_{\text{time}} + \frac{\alpha}{2}(\mathcal{L}_{\text{amp}} + \mathcal{L}_{\text{phase}}) $$

여기서 $$\alpha$$는 추세와 계절성 요소 간 균형을 조절하는 하이퍼파라미터입니다.[1]

### 성능 향상

**벤치마크 결과:**

- **다변량 설정**: 최고 성능 end-to-end 방법 대비 MSE 39.3% 개선, 최고 feature-based 방법 대비 21.3% 개선[1]
- **단변량 설정**: 최고 성능 end-to-end 방법 대비 MSE 18.22% 개선, 최고 feature-based 방법 대비 4.71% 개선[1]
- ETT, Electricity, Weather 등 5개 실제 데이터셋에서 일관되게 SOTA 달성[1]

**Ablation Study:**
- 추세와 계절성 구성요소 모두 베이스라인(SimCLR, MoCo) 대비 성능 향상[1]
- 두 구성요소의 결합이 최적 성능 달성[1]
- TCN, LSTM, Transformer 등 다양한 백본 인코더에서 강건한 성능[1]
- Ridge, Linear, Kernel Ridge 등 다양한 회귀기에서 강건한 성능[1]

### 한계

논문에서 명시적으로 언급된 한계는 제한적이지만, 다음을 추론할 수 있습니다:

1. **계산 비용**: TS2Vec 대비 학습 시간이 약 3배 더 소요됩니다(ETTm1 기준: CoST 262.78초 vs TS2Vec 91.9초). 이는 자기회귀 전문가 혼합의 순차 계산 때문이며, 병렬 처리 방법으로 가속화 가능합니다.[1]

2. **하이퍼파라미터 민감성**: $$\alpha$$ 파라미터가 데이터셋에 따라 최적값이 다를 수 있습니다. 대부분의 경우 작은 값($$5 \times 10^{-4}$$)이 효과적이지만, 특정 경우(ETTh2 168, 336 예측 구간)에는 더 큰 값이 선호됩니다.[1]

3. **예측 작업 한정**: 논문은 예측에 초점을 맞추고 있으며, 다른 시계열 과제(분류, 이상 탐지 등)로의 확장은 향후 연구 과제입니다.[1]

## 3. 일반화 성능 향상

**일반화 성능 향상의 핵심 메커니즘:**

### 인과적 불변성 (Causal Invariance)

분리된 계절성-추세 표현은 오차 변수의 개입에 불변하므로, **비정상 환경에서의 일반화 성능이 우수**합니다. 독립 메커니즘 가정(independent mechanisms assumption)에 따라, 계절성과 추세 모듈은 서로 영향을 주지 않습니다. 따라서 한 메커니즘이 분포 변화를 겪어도 다른 메커니즘은 변하지 않아 더 나은 전이(transfer) 및 일반화를 제공합니다.[1]

### 데이터 증강을 통한 개입 시뮬레이션

CoST는 데이터 증강(scaling, shifting, jittering)을 오차 변수에 대한 개입으로 해석하여 불변 표현을 학습합니다. 이는 다양한 오차 유형에 대한 강건성을 제공합니다.[1]

### 허위 상관관계 방지

End-to-end 방법과 달리, CoST는 예측 불가능한 노이즈의 허위 상관관계를 학습하는 문제를 완화합니다. 표현 학습과 예측을 분리함으로써 노이즈에 덜 민감한 특징을 학습합니다.[1]

### 실증적 증거

**Case Study**: 합성 데이터에서 CoST는 다양한 계절성 및 추세 패턴을 명확히 구분하는 반면, TS2Vec은 계절성 패턴 구분에 실패합니다. T-SNE 시각화에서 CoST의 표현은 높은 클러스터성을 보이며, TFD 표현은 추세를 잘 분리하고 SFD 표현은 계절성을 잘 분리합니다.[1]

**강건성 검증**: 
- 다양한 백본 인코더(TCN, LSTM, Transformer)에서 일관된 성능[1]
- 다양한 회귀기(Ridge, Linear, Kernel Ridge)에서 일관된 성능[1]
- 이는 학습된 표현이 **다양한 하류 작업 및 아키텍처에 일반화**됨을 시사합니다[1]

## 4. 향후 연구에 미치는 영향 및 고려사항

### 향후 연구에 미치는 영향

**1) 패러다임 전환**

CoST는 시계열 예측에서 **표현 학습 우선(representation learning first)** 패러다임의 효과를 입증했습니다. 이는 컴퓨터 비전과 NLP의 성공적인 접근법을 시계열에 적용한 사례로, 향후 연구 방향을 제시합니다.[1]

**2) 인과 기반 표현 학습**

인과적 관점에서 분리된 표현 학습의 필요성을 이론적으로 정당화했습니다. 이는 다른 시계열 과제(분류, 이상 탐지 등)에도 적용 가능한 프레임워크를 제공합니다.[1]

**3) 주파수 도메인 대조 학습**

주파수 도메인 대조 손실의 새로운 활용법을 제시했습니다. 이는 주기 정보를 사전에 지정하지 않고도 계절성 패턴을 학습할 수 있는 방법으로, 다양한 응용이 가능합니다.[1]

**4) 모듈러 아키텍처**

추세와 계절성을 독립적으로 학습하고 재사용할 수 있는 모듈러 설계는 **전이 학습 및 도메인 적응**에 유용합니다. 예를 들어, 특정 도메인에서 학습한 추세 모듈을 다른 도메인의 계절성 모듈과 결합할 수 있습니다.[1]

### 향후 연구 시 고려할 점

**1) 다른 시계열 과제로의 확장**

논문의 결론에서 언급했듯이, CoST 프레임워크를 분류, 이상 탐지 등 다른 시계열 인텔리전스 과제로 확장하는 것이 중요한 향후 연구 방향입니다.[1]

**2) 계산 효율성 개선**

자기회귀 전문가 혼합의 병렬화를 통한 학습 시간 단축이 필요합니다. FastMoE와 같은 병렬 처리 기법을 활용할 수 있습니다.[1]

**3) 적응적 $$\alpha$$ 선택**

데이터셋별로 최적의 $$\alpha$$ 값이 다르므로, 자동으로 최적값을 선택하는 메커니즘 개발이 유용할 것입니다. 예를 들어, 메타 학습이나 validation 성능 기반 적응적 조정을 고려할 수 있습니다.[1]

**4) 더 복잡한 시계열 구조**

현재 CoST는 추세와 계절성 두 가지 요소를 분리합니다. 그러나 실제 시계열은 주기(cycle), 불규칙 변동(irregular fluctuations), 구조적 변화점(structural breaks) 등 더 복잡한 구조를 가질 수 있습니다. 이러한 추가 요소를 모델링하는 확장이 필요합니다.[1]

**5) 다변량 의존성 모델링**

현재 프레임워크는 각 시간 단계의 다변량 표현을 학습하지만, 변수 간 복잡한 의존성(예: Granger causality, 공적분)을 명시적으로 모델링하지 않습니다. 변수 간 인과 구조를 학습하는 메커니즘 추가가 유용할 것입니다.[1]

**6) 극단값 및 이상치 처리**

데이터 증강(jittering, scaling, shifting)이 극단값이나 이상치에 민감할 수 있습니다. 강건한 데이터 증강 전략 개발이 필요합니다.[1]

**7) 해석 가능성 향상**

학습된 추세 및 계절성 표현의 해석 가능성을 높이는 연구가 필요합니다. 예를 들어, attention 메커니즘이나 saliency map을 통해 모델이 어떤 시간 구간이나 주파수에 집중하는지 시각화할 수 있습니다.

**8) 실시간 및 온라인 학습**

현재 CoST는 오프라인 배치 학습을 가정합니다. 실시간 스트리밍 데이터나 개념 변화(concept drift)가 있는 온라인 시나리오로 확장하는 것이 중요한 과제입니다.[1]

**9) 다중 해상도 표현**

현재 단일 시간 해상도에서 작동하지만, 웨이블릿 변환이나 다중 해상도 분석을 통해 다양한 시간 스케일의 패턴을 포착하는 확장이 가능합니다.

**10) 불확실성 정량화**

예측의 불확실성을 정량화하는 메커니즘(예: 베이지안 접근법, 앙상블 방법)을 통합하면, 실제 응용에서 더 신뢰할 수 있는 의사결정을 지원할 수 있습니다.

CoST는 시계열 예측에서 표현 학습의 중요성을 입증하고, 인과적 관점에서 분리된 표현의 이점을 보여준 중요한 연구입니다. 이 프레임워크는 향후 시계열 연구에 폭넓은 영향을 미칠 것이며, 위에서 언급한 다양한 방향으로 확장될 수 있습니다.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d83434d7-bac1-4716-ad76-baca4acca630/2202.01575v3.pdf)
