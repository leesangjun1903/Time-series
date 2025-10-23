# Learning Latent Seasonal-Trend Representations for Time Series Forecasting

## 1. 핵심 주장과 주요 기여

본 논문은 **LaST(Learning Latent Seasonal-Trend Representations)** 프레임워크를 제안하여 시계열 예측 문제를 해결합니다. 핵심 주장은 시계열 데이터를 **계절성(seasonal)과 추세(trend) 성분으로 분리된(disentangled) 잠재 표현**으로 학습함으로써, 기존 단일 표현 방식의 한계를 극복할 수 있다는 것입니다.[1]

**주요 기여:**

- **변분 추론(Variational Inference)과 정보 이론**을 활용한 계절성-추세 표현 학습 및 분리 메커니즘 설계[1]
- 계절성과 추세를 **별도로 재구성**하여 혼란을 방지하는 실용적 접근법 제공[1]
- **상호 정보(Mutual Information) 하한(lower bound)과 상한(upper bound)**을 도입하여 MINE 방법의 편향된 그래디언트 문제를 개선하고 정보성 있는 표현을 보장[1]
- 7개 실제 데이터셋에서 **state-of-the-art 성능** 달성 (MSE 기준 CoST 대비 25.6%, Autoformer 대비 22.0% 향상)[1]

## 2. 문제, 제안 방법, 모델 구조, 성능 및 한계

### 해결하고자 하는 문제

기존 딥러닝 시계열 예측 모델은 다음과 같은 한계를 가집니다:[1]

1. **복잡한 시간적 패턴 추출의 어려움**: 계절성, 추세, 수준 등의 명확한 정보를 추출하기 어려움
2. **표현의 얽힘(Entanglement)**: 단일 고차원 표현은 신경망의 얽힌 특성으로 인해 정보 활용도와 설명 가능성이 낮음
3. **과적합 위험**: 표현의 얽힘으로 인한 과적합 및 성능 저하

### 제안하는 방법 (수식 포함)

#### 기본 프레임워크

LaST는 변분 추론 기반 인코더-디코더 구조를 사용하며, 시계열을 계절성과 추세의 합으로 분해합니다:[1]

$$X = X_s + X_t$$

잠재 표현 $$Z$$는 독립적인 계절성 표현 $$Z_s$$와 추세 표현 $$Z_t$$로 분해됩니다:[1]

$$P(Z) = P(Z_s)P(Z_t)$$

#### ELBO 분해

Evidence Lower Bound(ELBO)는 다음과 같이 분해됩니다:[1]

$$
\mathcal{L}_{\text{ELBO}} = \log \int_{Z_s} \int_{Z_t} P_\psi(Y|Z_s, Z_t)Q_{\phi_s,\phi_t}(Z_s, Z_t|X)dZ_sdZ_t
$$

```math
+ \mathbb{E}_{Q_{\phi_s}(Z_s|X)}[\log P_{\theta_s}(X_s|Z_s)] + \mathbb{E}_{Q_{\phi_t}(Z_t|X)}[\log P_{\theta_t}(X_t|Z_t)]
```

```math
- \text{KL}(Q_{\phi_s}(Z_s|X)||P(Z_s)) - \text{KL}(Q_{\phi_t}(Z_t|X)||P(Z_t))
```

#### 재구성 손실

가우시안 분포 가정 하에서 재구성 손실은 다음과 같이 추정됩니다:[1]

$$
\mathcal{L}_{\text{rec}} = -\sum_{\kappa=1}^{T-1} ||A_{XX}(\kappa) - A_{\hat{X}_s\hat{X}_s}(\kappa)||^2 + \text{CORT}(X, \hat{X}_t) - ||\hat{X}_t + \hat{X}_s - X||^2
$$

여기서:
- $$A_{XX}(\kappa) = \sum_{i=1}^{T-\kappa}(X_t - \bar{X})(X_{t+\kappa} - \bar{X})$$: 자기상관 계수
- $$\text{CORT}(X, \hat{X}\_t) = \frac{\sum_{i=1}^{T-1}\Delta X_i^t \Delta \hat{X}\_i^t}{\sqrt{\sum_{i=1}^{T-1}(\Delta X_i^t)^2}\sqrt{\sum_{i=1}^{T-1}(\Delta \hat{X}_i^t)^2}}$$: 시간적 상관 계수

#### 상호 정보 최적화

최종 목적 함수는:[1]

$$
\mathcal{L}_{\text{LaST}} = \mathcal{L}_{\text{ELBO}} + I(X, Z_s) + I(X, Z_t) - I(Z_s, Z_t)
$$

**개선된 하한(Lower Bound)** - MINE의 편향된 그래디언트 문제를 해결:[1]

$$
I(X, Z) \geq \mathbb{E}_{Q_\phi(X,Z)}[\gamma_\alpha(X, Z)] - \frac{1}{\eta}\mathbb{E}_{Q(x)Q_\phi(z)}[e^{\gamma_\alpha(X,Z)}]
$$

여기서 $$\eta = \mathbb{E}\_{Q(x)Q_\phi(z)}[e^{\gamma_\alpha(X,Z)}]$$로 설정하여 접선점과 독립 변수 간 거리를 최소화합니다.

**상한(Upper Bound) - STUB** (Seasonal-Trend Upper Bound):[1]

$$
I(Z_s, Z_t) \leq \mathbb{E}_{Q_{\phi_s,\phi_t}(Z_s,Z_t)}[\gamma_\beta(Z_s, Z_t)] - \mathbb{E}_{Q_{\phi_s}(Z_s)Q_{\phi_t}(Z_t)}[\gamma_\beta(Z_s, Z_t)]
$$

### 모델 구조

**인코더**: 
- 계절성 인코더 $$Q_{\phi_s}$$와 추세 인코더 $$Q_{\phi_t}$$가 독립적으로 잠재 표현 학습[1]
- FFN(Feed Forward Network) 기반 단일 레이어 완전 연결 네트워크 사용[1]

**예측기(Predictor)**:[1]
- **계절성 예측**: DFT(Discrete Fourier Transform)로 주파수 도메인에서 패턴 감지 후 iDFT로 미래 시점 확장
  - $$Z_s^F = \text{DFT}(Z_s) \in \mathbb{C}^{F \times d}$$
  - $$\tilde{Z}_s = \text{iDFT}(Z_s^F) \in \mathbb{R}^{\tau \times d}$$
- **추세 예측**: FFN을 사용하여 $$\tilde{Z}_t$$ 생성
- 최종 예측: $$Y = Y^s + Y^t$$

**디코더**:
- 계절성 디코더 $$P_{\theta_s}$$와 추세 디코더 $$P_{\theta_t}$$가 별도로 재구성[1]
- 2-layer MLP로 상호 정보 추정을 위한 critic $$\gamma$$ 구현[1]

### 성능 향상

**단변량(Univariate) 예측**:[1]
- CoST 대비 MSE 25.6%, MAE 22.1% 향상
- Autoformer 대비 MSE 22.0%, MAE 18.9% 향상
- ETTh1, ETTm1, Electricity 등 5개 데이터셋에서 최고 성능

**다변량(Multivariate) 예측**:[1]
- 평균 MSE 0.330, MAE 0.347로 모든 baseline 능가
- Exchange, Weather 데이터셋에서 특히 큰 성능 차이

**Ablation Study 결과**:[1]
- 추세 성분 제거 시 성능이 크게 저하 (합성 데이터에서 MSE 2.870으로 급증)
- MINE 대신 제안된 하한 사용 시 일관된 성능 향상
- 자기상관 및 CORT 계수 제거 시 성능 감소

### 한계

**논문에서 언급된 한계**:[1]
1. **Autoformer의 장기 예측 우위**: 시간별 ETT 데이터셋의 장기 예측에서 Autoformer가 더 나은 성능
   - Transformer 기반 모델의 장거리 의존성 포착 능력
   - 고정 커널 크기의 단순 분해가 강한 주기성 데이터에 유리

2. **이론적 한계 미언급**: 논문 체크리스트에서 연구의 한계에 대한 설명이 부족하다고 명시[1]

**추론 가능한 한계**:
- 계절성 예측이 iDFT 알고리즘에 의존하여 본질적으로 과거 관측의 주기적 반복
- 2개 이상의 성분 분해로 확장 가능하다고 언급되었으나 구체적 검증 부족[1]
- 비정상(non-stationary) 시계열에 대한 성능 검증 미흡

## 3. 모델의 일반화 성능 향상 가능성

### 일반화 성능 향상 메커니즘

**1. 분리된 표현 학습(Disentangled Representation)**[1]

LaST의 핵심 일반화 전략은 계절성과 추세를 독립적으로 모델링하는 것입니다. t-SNE 시각화 결과, LaST의 표현은 동일 색상의 점들이 명확하게 클러스터링되는 반면, Autoformer는 혼합되어 나타납니다. 이는 다음을 의미합니다:[1]

- **특징 얽힘 방지**: 단일 표현의 과적합 위험 감소
- **패턴 특화 학습**: 각 성분이 특정 시간적 패턴에 집중
- **해석 가능성 향상**: 모델 결정의 투명성 증가

**2. 상호 정보 제약**[1]

$$I(X, Z_s)$$와 $$I(X, Z_t)$$ 최대화:
- KL divergence로 인한 posterior와 prior 간 거리 축소 문제 완화
- 입력 데이터에 대한 정보성 있는 표현 보장
- 과적합 방지를 위한 정보 보존

$$I(Z_s, Z_t)$$ 최소화:
- 계절성과 추세 표현 간 중복 감소
- 독립적 특징 학습으로 일반화 능력 향상

**3. 재구성 기반 감독(Reconstruction-based Supervision)**[1]

자기상관 계수와 시간적 상관 계수를 통한 분리된 재구성:
- $$X_s$$와 $$X_t$$를 직접 사용하지 않고도 재구성 손실 추정 가능
- 디코더가 모든 표현에서 복잡한 시계열을 재구성하려는 혼란 방지
- 암묵적 정규화 효과로 일반화 성능 향상

**4. 입력 길이 민감도**[1]

Table 4에서 긴 look-back window가 특히 장기 예측에서 성능을 향상시킴을 보여줍니다:
- 입력 길이 201에서 ETTm1 평균 MSE 0.397 달성
- 과거 정보를 효과적으로 활용하여 패턴 이해 및 예측 수행
- 다양한 시간 스케일의 패턴 포착 가능

### 일반화 성능의 실증적 증거

**1. 다양한 데이터셋에서 일관된 성능**[1]

7개 데이터셋(ETTh1, ETTh2, ETTm1, ETTm2, Electricity, Exchange, Weather)에서 모두 우수한 성능:
- 다양한 도메인(전력, 금융, 기상)
- 다양한 샘플링 주기(15분, 시간, 일)
- 다양한 변수 수(단변량/다변량)

**2. Case Study 시각화**[1]

Figure 3에서 LaST가 실제 데이터셋에서 계절성 패턴을 포착함을 보여줍니다:
- ETTh1/ETTm1: 강한 일일 주기 감지
- Exchange: 명확한 주기가 없어도 장기 패턴 제공
- 추세와 계절성 성분이 공동으로 원본 시퀀스를 정확히 복원

**3. Ablation Study를 통한 메커니즘 검증**[1]

- "w/o seasonal" 및 "w/o trend": 각 성분의 필요성 확인
- "w/o coe": 자기상관 및 CORT 계수의 중요성 입증
- "w/o lower" 및 "w/o upper": MI 제약의 기여도 확인
- "with MINE": 제안된 unbiased bound의 우수성 검증

### 일반화 성능 한계 및 개선 방향

**현재 한계**:
- 강한 주기성을 가진 시간별 데이터의 장기 예측에서 Transformer 기반 모델에 뒤짐[1]
- 비정상 시계열이나 개념 변동(concept drift)에 대한 강건성 미검증

**개선 가능성**:
- Transformer의 self-attention 메커니즘과 결합하여 장거리 의존성 포착 능력 강화
- 확률적 요인(stochastic factors)을 명시적으로 모델링하여 불확실성 처리 개선[1]
- 2개 이상의 성분 분해로 확장하여 더 세밀한 패턴 포착[1]

## 4. 앞으로의 연구에 미치는 영향 및 고려 사항

### 연구에 미치는 영향

**1. 시계열 분야의 패러다임 전환**

LaST는 **분리된 표현 학습(disentangled representation learning)**을 시계열 예측에 성공적으로 적용한 사례입니다. 이는 다음을 가능하게 합니다:[1]

- 컴퓨터 비전의 disentangled VAE 성공을 시계열 도메인으로 확장
- 고전적 시계열 분해(STL 등)와 딥러닝의 효과적 결합
- 해석 가능성과 성능을 동시에 달성하는 새로운 방향 제시

**2. 변분 추론의 개선**

- **Unbiased MI lower bound**: MINE의 편향된 그래디언트 문제 해결[1]
- **Traceable upper bound (STUB)**: 에너지 기반 변분 계열을 사용한 새로운 상한 제안[1]
- 다른 변분 추론 기반 모델에도 적용 가능한 범용적 기법

**3. 다운스트림 태스크로의 확장**[1]

논문에서 언급된 향후 연구 방향:
- **시계열 생성(generation)**: 분리된 표현을 활용한 조건부 생성
- **결측치 보간(imputation)**: 계절성과 추세를 독립적으로 복원
- **이상 탐지(anomaly detection)**: 성분별 이상 패턴 감지

### 앞으로 연구 시 고려할 점

**1. 모델 설계 측면**

**성분 수의 확장**:
- LaST는 계절성과 추세 2개 성분으로 제한되어 있지만, 프레임워크는 더 많은 성분으로 확장 가능하다고 언급[1]
- 고려 사항: 수준(level), 주기성(cyclicality), 노이즈 등 추가 성분의 명시적 모델링
- 성분 수 증가 시 상호 정보 제약의 계산 복잡도 증가

**장거리 의존성 포착**:
- Transformer 기반 모델과의 통합 필요[1]
- 고려 사항: DFT 기반 계절성 예측의 한계를 attention 메커니즘으로 보완
- 계산 효율성과 성능 간의 트레이드오프

**확률적 요인 모델링**:
- 논문에서 향후 연구로 언급된 stochastic factors의 명시적 모델링[1]
- 고려 사항: 불확실성 정량화, 확률적 예측, 분포 shift 대응

**2. 데이터 및 평가 측면**

**비정상 시계열 처리**:
- 현재 검증은 주로 정상(stationary) 또는 약 정상 데이터에 국한
- 고려 사항: concept drift, regime change, structural break에 대한 강건성 검증

**다양한 시간 스케일**:
- 현재 고정된 입력 길이(T=201) 사용[1]
- 고려 사항: 적응형 입력 길이, 다중 스케일 패턴 동시 포착

**평가 메트릭 확장**:
- 현재 MSE, MAE에 국한[1]
- 고려 사항: 분포 기반 메트릭(CRPS 등), 불확실성 캘리브레이션, 해석 가능성 메트릭

**3. 실용성 및 확장성 측면**

**계산 효율성**:
- MI 추정을 위한 critic network의 추가 계산 비용
- 고려 사항: 경량화 버전 개발, 효율적인 MI 추정 방법 탐구

**하이퍼파라미터 민감도**:
- 표현 차원(32/128), 학습률 decay(0.95), MI 가중치 등 다수의 하이퍼파라미터[1]
- 고려 사항: AutoML 기법 적용, 강건한 기본 설정 제공

**실시간 예측**:
- 온라인 학습 및 점진적 업데이트 가능성
- 고려 사항: 스트리밍 데이터에 대한 적응형 분해, 계산 효율성

**4. 이론적 기반 강화**

**수렴 보장**:
- STUB의 음수 값 문제에 대한 페널티 항 추가[1]
- 고려 사항: 이론적 수렴 조건 분석, 최적화 안정성 증명

**일반화 한계 분석**:
- 샘플 복잡도, PAC learning 프레임워크 적용
- 고려 사항: 분리 품질과 예측 성능 간의 이론적 관계 규명

**인과 관계 분석**:
- 계절성과 추세의 독립성 가정에 대한 인과적 정당화
- 고려 사항: 인과 추론 프레임워크와의 통합, 반사실적(counterfactual) 예측

***

LaST는 시계열 예측 분야에 **분리된 표현 학습**, **개선된 변분 추론**, **해석 가능한 성분 분해**라는 세 가지 핵심 기여를 제공합니다. 향후 연구는 이러한 기반 위에 장거리 의존성 포착, 확률적 요인 모델링, 다운스트림 태스크 확장, 실용성 향상 등의 방향으로 발전할 것으로 기대됩니다. 특히 일반화 성능 측면에서 분리된 표현이 과적합을 방지하고 다양한 데이터셋에서 일관된 성능을 보인다는 점이 중요한 통찰입니다.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c0e52905-a9e1-460a-b06b-b6d70ac69995/NeurIPS-2022-learning-latent-seasonal-trend-representations-for-time-series-forecasting-Paper-Conference.pdf)
