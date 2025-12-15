# CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 핵심 주장

**CSDI(Conditional Score-based Diffusion models for Imputation)**는 시계열 데이터의 결측값 대체(Imputation) 문제를 해결하기 위해 조건부(Conditional) Score 기반 확산 모델(Diffusion Models)을 최초로 명시적으로 설계한 방법입니다. 기존의 비조건부 확산 모델을 조건부로 확장하여, 관찰된 데이터(observed values)를 직접 입력으로 받아 결측값의 조건부 분포를 정확히 학습합니다.

### 1.2 주요 기여

1. **조건부 확산 모델의 설계**: 기존 확산 모델을 조건부 분포 학습에 최적화하여, 관찰된 값들의 정보를 직접 활용할 수 있는 구조 제안

2. **자기지도 학습(Self-Supervised Learning) 전략**: 실제 결측값을 알 수 없는 상황에서, Masked Language Modeling 영감의 자기지도 학습으로 훈련 데이터 생성 가능

3. **우수한 성능 달성**:
   - 확률적 대체(Probabilistic Imputation): CRPS 기준 기존 방법 대비 **40-65% 개선**
   - 결정적 대체(Deterministic Imputation): MAE 기준 기존 방법 대비 **5-20% 개선**
   - 시계열 보간(Interpolation) 및 확률적 예측(Forecasting)에도 적용 가능

---

## 2. 문제 정의, 제안 방법, 모델 구조 상세 설명

### 2.1 해결하고자 하는 문제

#### 문제 설정

$$X = \{x_{1:K, 1:L}\} \in \mathbb{R}^{K \times L}$$

여기서 $K$는 특성(feature) 개수, $L$은 시계열 길이입니다.

관찰 마스크: 
$$M = \{m_{1:K, 1:L}\} \in \{0,1\}^{K \times L}$$
- $m_{k,l} = 0$ (결측)
- $m_{k,l} = 1$ (관찰됨)

**확률적 시계열 대체(Probabilistic Time Series Imputation)**: 관찰된 값들 $x^{co}_0$을 이용하여 결측값 $x^{ta}_0$의 조건부 분포를 추정하는 것
$$q(x^{ta}_0 | x^{co}_0)$$

#### 기존 방법의 한계

기존 연구들은 조건부 확산 모델을 다음과 같이 근사화했습니다:

$$p_\theta(x^{ta}_{t-1} | x^{ta}_t, x^{co}_0) \approx p_\theta(x^{ta}_{t-1} | x^{ta}_t)$$

이 근사화는:
- 조건부 정보를 제대로 활용하지 못함
- 관찰된 값에 노이즈를 추가하여 유용한 정보 손상
- 정확한 조건부 분포 학습 불가능

### 2.2 제안하는 방법: CSDI

#### 2.2.1 Forward Process (확산 과정)

표준 DDPM의 forward process:
$$q(x_{1:T} | x_0) := \prod^T_{t=1} q(x_t | x_{t-1})$$
$$q(x_t | x_{t-1}) := \mathcal{N}(\sqrt{1-\beta_t}x_{t-1}, \beta_t I)$$

Closed-form:
$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\alpha_t}x_0, (1-\alpha_t)I)$$
$$x_t = \sqrt{\alpha_t}x_0 + \sqrt{1-\alpha_t}\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

여기서:
- $\hat{\alpha}_t := 1 - \beta_t$
- $\alpha_t := \prod^t_{i=1} \hat{\alpha}_i$

#### 2.2.2 Reverse Process (역확산 과정)

**표준 DDPM 역과정**:

$$p_\theta(x_{0:T}) := p(x_T) \prod^T_{t=1} p_\theta(x_{t-1} | x_t)$$

$$p_\theta(x_{t-1} | x_t) := \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_\theta(x_t, t)I)$$

DDPM의 매개변수화:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\hat{\alpha}_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\alpha_t}}\epsilon_\theta(x_t, t) \right)$$

$$\sigma_\theta(x_t, t) = \tilde{\beta}^{1/2}_t \quad \text{where} \quad \tilde{\beta}_t = \begin{cases} \frac{1-\alpha_{t-1}}{1-\alpha_t}\beta_t & t > 1 \\ \beta_1 & t = 1 \end{cases}$$

**CSDI의 조건부 역과정** (핵심):

$$p_\theta(x^{ta}_{0:T} | x^{co}_0) := p(x^{ta}_T) \prod^T_{t=1} p_\theta(x^{ta}_{t-1} | x^{ta}_t, x^{co}_0)$$

$$p_\theta(x^{ta}_{t-1} | x^{ta}_t, x^{co}_0) := \mathcal{N}(x^{ta}_{t-1}; \mu_\theta(x^{ta}_t, t | x^{co}_0), \sigma_\theta(x^{ta}_t, t | x^{co}_0)I)$$

#### 2.2.3 조건부 매개변수화

조건부 디노이징 함수 $\epsilon_\theta: (X^{ta} \times \mathbb{R} | X^{co}) \rightarrow X^{ta}$를 정의:

$$\mu_\theta(x^{ta}_t, t | x^{co}_0) = \mu_{DDPM}(x^{ta}_t, t, \epsilon_\theta(x^{ta}_t, t | x^{co}_0))$$

$$\sigma_\theta(x^{ta}_t, t | x^{co}_0) = \sigma_{DDPM}(x^{ta}_t, t)$$

핵심 차이점: 표준 매개변수화는 유지하되, **디노이징 함수가 조건부 정보를 입력으로 받음**

#### 2.2.4 훈련 목적 함수

$$\min_\theta L(\theta) := \min_\theta \mathbb{E}_{x_0 \sim q(x_0), \epsilon \sim \mathcal{N}(0,I), t} || \epsilon - \epsilon_\theta(x^{ta}_t, t | x^{co}_0)||^2_2$$

여기서:
$$x^{ta}_t = \sqrt{\alpha_t} x^{ta}_0 + \sqrt{1-\alpha_t} \epsilon$$

### 2.3 자기지도 학습 전략 (Self-Supervised Learning)

#### 문제점
실무에서는 실제 결측값을 알 수 없으므로, 직접적인 $(x^{co}_0, x^{ta}_0)$ 쌍을 구성할 수 없음.

#### 해결책: 마스크된 학습 (Masked Learning)

Masked Language Modeling 영감의 접근:

1. **훈련 시**: 관찰된 값들을 두 부분으로 분할
   - 조건부 관찰 정보: $x^{co}_0$
   - 대체 목표: $x^{ta}_0$ (관찰된 값의 부분집합)

2. **샘플링 시**: 모든 관찰된 값을 조건부 정보로 사용

#### 대체 목표 선택 전략 (Target Choice Strategy)

**전략 1: Random Strategy** (무작위 전략)
- 관찰된 값 중 임의의 백분율(0%-100%) 선택
- 다양한 결측 비율에 적응 가능
- 결측 패턴을 모를 때 사용

**전략 2: Historical Strategy** (역사 기반 전략)
- 훈련 데이터의 결측 패턴 활용
- 훈련 데이터에서 다른 샘플 $\tilde{x}_0$ 선택
- 대체 목표: $x^{ta}_0 = x^{co}_0(\text{in } x_0) \cap x^{ta}_0(\text{in } \tilde{x}_0)$
- 구조화된 결측 패턴이 있을 때 효과적

**전략 3: Mix Strategy** (혼합 전략)
- Random과 Historical 전략의 조합
- 일반화와 구조화된 패턴 모두 활용

**전략 4: Test Pattern Strategy** (테스트 패턴 전략)
- 테스트 데이터의 결측 패턴을 사전에 알 때 사용
- 시계열 예측(Forecasting) 작업에서 활용

### 2.4 모델 구조 (Architecture)

#### 2.4.1 시계열 입력 처리

조건부 디노이징 함수를 고정 크기 입력으로 처리:

$$\epsilon_\theta(x^{ta}_t, t | x^{co}_0, m^{co}) : (\mathbb{R}^{K \times L} \times \mathbb{R} | \mathbb{R}^{K \times L} \times \{0,1\}^{K \times L}) \rightarrow \mathbb{R}^{K \times L}$$

조건부 마스크 $m^{co} \in \{0,1\}^{K \times L}$:
- 조건부 관찰 인덱스에 대해 1
- 패딩된 영역에 대해 0

입력 표현:
$$x^{co}_0 = m^{co} \odot X$$

여기서 $\odot$는 원소별 곱셈(element-wise product)

#### 2.4.2 기본 구조: DiffWave 기반

기본 아키텍처: DiffWave (Diff Wave, 2021)
- 다중 Residual Layers (Residual Channel C)
- Diffusion Step T = 50

#### 2.4.3 핵심 개선: 2D Attention Mechanism

**시간적 의존성 (Temporal Dependencies) 학습**:
$$\text{Temporal Transformer Layer: } (1, L, C) \text{ shape 입력}$$

**특성 의존성 (Feature Dependencies) 학습**:
$$\text{Feature Transformer Layer: } (K, 1, C) \text{ shape 입력}$$

2D Attention 메커니즘의 이점:
1. **시간적 패턴**: 각 특성 내 시간 축 의존성 포착
2. **특성 간 관계**: 각 시점에서 특성 간 상호작용 포착
3. **계산 효율**: 분리된 주의 메커니즘으로 복잡도 감소

#### 2.4.4 보조 정보 (Side Information)

**시간 임베딩**:
$$\text{Temporal Embedding: } s = \{s_1, s_L\} \in \mathbb{R}^{L \times 128}$$
- 시점 간 거리 정보 활용
- Positional Encoding 기법 적용

**특성 범주 임베딩**:
$$\text{Categorical Feature Embedding: } \text{Dimension} = 16$$
- 각 특성의 의미 정보 인코딩

---

## 3. 성능 향상 및 한계

### 3.1 성능 향상

#### 3.1.1 확률적 대체 성능 (Probabilistic Imputation)

**지표**: Continuous Ranked Probability Score (CRPS) - 낮을수록 좋음

**의료 데이터 (PhysioNet Challenge 2012)**:
| Missing Ratio | Multitask GP | GP-VAE | V-RIN | CSDI |
|---|---|---|---|---|
| 10% | 0.489 | 0.574 | 0.808 | **0.238** |
| 50% | 0.581 | 0.774 | 0.831 | **0.330** |
| 90% | 0.942 | 0.998 | 0.922 | **0.522** |

**개선율**: 40-65% (기존 최고 기법 대비)

**환경 데이터 (공기질, 베이징)**:
- Multitask GP: 0.301
- **CSDI: 0.108** (64% 개선)

#### 3.1.2 결정적 대체 성능 (Deterministic Imputation)

**지표**: Mean Absolute Error (MAE)

**의료 데이터**:
| Missing Ratio | GLIMA | BRITS | CSDI |
|---|---|---|---|
| 10% | 0.265 | 0.284 | **0.217** |
| 50% | - | 0.368 | **0.301** |
| 90% | - | 0.517 | **0.481** |

**개선율**: 5-20% (기존 SOTA 대비)

**특성**: 결측 비율이 낮을수록 성능 이득이 큼
- 관찰된 값이 많을수록 조건부 정보 활용이 효과적

#### 3.1.3 보간(Interpolation) 성능

**비정규 샘플링 시계열 보간**

| Method | 10% Missing | 50% Missing | 90% Missing |
|---|---|---|---|
| Latent ODE | 0.700 | 0.676 | 0.761 |
| mTANs | 0.526 | 0.567 | 0.689 |
| **CSDI** | **0.380** | **0.418** | **0.556** |

개선율: 28-45%

#### 3.1.4 예측(Forecasting) 성능

**확률적 시계열 예측 (CRPS-sum 기준)**

| Dataset | GP-copula | TransMAF | TimeGrad | CSDI |
|---|---|---|---|---|
| Electricity | 0.024 | 0.021 | 0.021 | **0.017** |
| Traffic | 0.078 | 0.056 | 0.044 | **0.020** |

관찰: 예측 작업에서는 확률적 대체만큼 큰 이점이 없음
- 이유: 예측 데이터는 결측이 적고 기존 인코더에 적합

### 3.2 한계점 (Limitations)

#### 3.2.1 계산 효율성

**샘플링 속도**:
- 확산 모델은 본질적으로 반복적인 역과정 필요 (T=50 단계)
- 다른 생성 모델(GAN 등)보다 느림
- 실시간 응용에 제한

**해결 방안**:
- ODE 솔버 기반 고속화 (DPM-Solver, DDIM 등)
- 더 효율적인 샘플링 알고리즘 통합

#### 3.2.2 조건부 마스크의 제한

**고정 크기 입력 패딩**:
- 다양한 길이의 시계열을 고정 $(K \times L)$ 크기로 조정
- 영 패딩은 정보 손실 유발 가능
- 극도로 짧은 시계열에서 비효율

#### 3.2.3 훈련 데이터 요구사항

**완전 데이터 또는 자기지도 학습 필요**:
- 실제 결측 데이터로 직접 훈련 불가 (Ground-truth 부재)
- 자기지도 학습은 훈련/테스트 분포 차이 가능성

#### 3.2.4 영역 특화(Domain-Specific) 패턴 활용

**일반화 제한**:
- 특정 영역의 물리적 제약이나 도메인 지식 미통합
- 순수 데이터 기반 학습에만 의존

#### 3.2.5 예측 작업에서의 성능

- 보간/대체 작업에 특화
- 예측 성능은 기존 방법과 유사 (큰 이점 없음)

---

## 4. 모델의 일반화 성능 향상 가능성

### 4.1 현재 일반화 성능 분석

#### 4.1.1 데이터셋 간 성능 일관성

**의료 데이터 (PhysioNet)**:
- 완전히 관찰된 데이터에서 인공적으로 결측 생성
- 결측 비율 변화(10%, 50%, 90%)에 모두 강건
- 표준 편차 범위: 0.001-0.007 (낮은 분산)

**환경 데이터 (공기질)**:
- 자연 결측 패턴 (13% 결측)
- 구조화된 결측 패턴에도 효과적
- CSDI: 0.108 (매우 낮은 오류)

**관찰**: 서로 다른 도메인과 결측 패턴에 일관되게 효과적

#### 4.1.2 조건부 정보 활용의 우수성

**표: 무조건 vs 조건 확산 모델**

| Setting | 10% | 50% | 90% |
|---|---|---|---|
| Unconditional | 0.360 | 0.458 | 0.671 |
| **CSDI** | **0.238** | **0.330** | **0.522** |
| 개선 | **34%** | **28%** | **22%** |

결론: **조건부 모델이 명시적으로 더 나은 성능** 제공

### 4.2 일반화 성능 향상 가능성

#### 4.2.1 전이 학습(Transfer Learning) 잠재력

**현재 상태**:
- CSDI는 각 데이터셋에서 독립적으로 훈련
- 도메인 간 지식 전이 미탐색

**개선 방향**:
1. **사전훈련(Pre-training) + 미세조정(Fine-tuning)**
   - 대규모 다중 도메인 데이터로 사전훈련
   - 특정 도메인에서 미세조정으로 빠른 적응
   
2. **메타 학습(Meta-Learning)**
   - 다양한 결측 패턴에 빠르게 적응하는 능력 학습
   - Few-shot imputation 가능성

#### 4.2.2 도메인 적응(Domain Adaptation)

**최근 연구 (2024-2025)**:

**CD2-TSI (Cross-Domain Conditional Diffusion Models)**:
- 소스와 타겟 도메인 간 지식 전이
- 도메인 공유 표현 + 도메인 특화 표현 분리
- 도메인 불일치 해결

**성능**: 기존 CSDI 대비 추가 1.34-1.92% 개선

#### 4.2.3 자기지도 학습의 한계와 개선

**현재 자기지도 전략의 한계**:
- 훈련/테스트 분포 불일치 가능
- Random Strategy: 무작위 결측 가정 (현실과 다를 수 있음)
- Historical Strategy: 구조화된 패턴만 활용

**개선 방향**:
1. **적응형 전략(Adaptive Strategy)**
   - 훈련 과정에서 최적의 마스킹 비율 동적 학습
   
2. **결측 패턴 모델링**
   - 결측 메커니즘 명시적 학습
   - Missing Not At Random (MNAR) 처리
   
3. **다중 뷰 학습(Multi-view Learning)**
   - 다양한 마스킹 관점에서 일관된 표현 학습

#### 4.2.4 아키텍처 개선을 통한 일반화 강화

**최근 개선 사항 (2023-2025)**:

| 방법 | 개선 내용 | 성능 향상 |
|---|---|---|
| **SSSD** (2023) | Structured State Space Model 통합 | 기존 CSDI 대비 +50% MAE 감소 |
| **tBN-CSDI** (2025) | Time-varying Blue Noise 적용 | +30% 희소 데이터 오류 감소 |
| **MTSCI** (2024) | 일관성 제약(Consistency Constraint) | +17.88% MAE, +15.09% RMSE |
| **STDiff** (2025) | 상태 전이 모델링 + 인과성 편향 | 산업 시계열 특화, 긴 간격 처리 |

**핵심 개선 방향**:
1. 아키텍처: RNN → State Space Model → Transformer
2. 신호 처리: 백색 노이즈 → 블루 노이즈 → 주파수 기반
3. 제약: 무제약 → 일관성 제약 → 동역학 제약

#### 4.2.5 특성 간 의존성 모델링 개선

**현재**: 2D Attention (시간 + 특성 축 분리)

**개선 가능성**:
1. **상호 의존성(Cross-correlation) 학습**
   - 특성 간 시간 변화 의존성 명시적 모델
   - 예: Correlated Attention Mechanism

2. **다중 스케일 특성 추출**
   - 여러 시간 척도의 패턴 동시 학습
   - 계절성, 추세, 노이즈 분리

3. **물리 정보 통합(Physics-Informed)**
   - 도메인 제약 조건 명시적 통합
   - 예: 에너지 보존, 인과성 등

### 4.3 데이터 효율성 개선

**현재 상황**:
- 각 도메인마다 상당한 훈련 데이터 필요
- 소규모 데이터셋에서 성능 검증 부재

**개선 전략**:
1. **데이터 증강(Data Augmentation)**
   - Mixup, CutMix 적용
   - 기하학적 변환 기반 증강

2. **학습 효율 개선**
   - 더 적은 Diffusion Step (T < 50)으로 성능 유지
   - 증류(Distillation) 기반 고속화

3. **준지도 학습(Semi-supervised Learning)**
   - 라벨 없는 데이터 활용 확대

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 시계열 대체 관련 주요 연구 시계

#### Phase 1: 초기 확산 모델 적용 (2020-2021)
- **DDPM** (2020): 기본 확산 모델 제시
- **CSDI** (2021): **최초 시계열 대체 특화 조건부 확산 모델**
- **TimeGrad** (2021): 확률적 예측에 확산 모델 적용

#### Phase 2: 아키텍처 개선 (2022-2023)
- **SSSD** (2023): State Space Model 통합
- **TSDiff** (2023): 비조건 확산 + 자기 지도

#### Phase 3: 제약 조건 통합 (2024)
- **MTSCI** (2024): 일관성 제약
- **Score-CDM** (2024): 점수 가중 CNN 확산
- **ImDiffusion** (2023): 이상 탐지 응용

#### Phase 4: 고급 일반화 (2025)
- **tBN-CSDI** (2025): 주파수 적응 노이즈 스케줄
- **CD2-TSI** (2025): 도메인 적응 확산
- **STDiff** (2025): 상태 전이 프레임워크
- **LSCD** (2025): Lomb-Scargle 주파수 조건화

### 5.2 주요 경쟁 방법 비교

#### 5.2.1 SSSD (Diffusion-based Time Series Imputation with Structured State Space Models)

**발표**: 2022년 8월 (arXiv), 2023년 정식 출판

**핵심 기여**:
- **State Space Model (S4)** 아키텍처 통합
- CSDI의 Transformer 대신 S4 사용

**장점**:
- 장기 의존성(Long-term Dependencies) 포착 뛰어남
- 계산 복잡도 감소
- 블랙아웃 시나리오에서 매우 강함

**성능 비교**:
| Dataset | CSDI (MAE) | SSSD (MAE) | 개선 |
|---|---|---|---|
| Healthcare | 0.217 | 0.167 | **23.5%** |
| Electricity | 9.60 | 7.2 | **25%** |

**한계**:
- S4의 선형성이 일부 복잡한 의존성 놓칠 수 있음
- 특성 간 복잡한 상호작용 모델링에 제한

**CSDI와의 관계**:
- SSSD는 CSDI의 명확한 진화
- CSDI의 아이디어(조건부 확산)는 유지
- 내부 아키텍처만 개선

#### 5.2.2 tBN-CSDI (Time-varying Blue Noise-based CSDI)

**발표**: 2025년 9월

**핵심 기여**:
- CSDI의 **백색 노이즈 → 시간 가변 블루 노이즈** 변경
- 블루 노이즈: 고주파 성분 강조, 시간 구조 보존

**혁신점**:
$$\text{Blue Noise Schedule: } q_\text{blue}(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\alpha_t} x_0, (1-\alpha_t) \text{BlueNoise})$$

**성능**:
- 희소 데이터 (high sparsity): **30% 오류 감소**
- 일반화 능력: 다양한 희소성 수준에서 일관되게 개선

**장점**:
- 주파수 의존적 상관 관계 포착
- 세밀한 시간 패턴 보존

**한계**:
- 계산 오버헤드 증가
- 충분한 훈련 데이터 필요

#### 5.2.3 MTSCI (Multivariate Time Series Consistent Imputation)

**발표**: 2024년 8월

**핵심 기여**:
- 일관성 제약 명시적 도입
  - 대체-관찰 일관성 (Intra-consistency)
  - 인접 윈도우 일관성 (Inter-consistency)

**수식**:
- **보완적 마스크 대조**: 쌍을 이루는 샘플 생성
$$\mathcal{L}_\text{intra} = -\log \frac{\exp(\text{sim}(v_i, v_i^+) / \tau)}{\sum_j \exp(\text{sim}(v_i, v_j) / \tau)}$$

- **Mixup 메커니즘**: 인접 윈도우 정보 통합

**성능**:
- 의료 데이터: **17.88% MAE 감소** (CSDI 대비)
- RMSE: **15.09% 개선**
- MAPE: **13.64% 개선**

**장점**:
- 시간적 일관성 명시적 보장
- 인접 샘플과의 부드러운 연결

#### 5.2.4 STDiff (State Transition Diffusion Framework)

**발표**: 2025년 8월

**핵심 기여**:
- **동역학 기반 임퓨테이션**: 고정 윈도우 대신 상태 전이 모델
- 산업 시스템의 제어 입력 통합
- 인과성 편향(Causal Bias) 도입

**주요 특성**:
- 제어 이론 기반 설계
- 비정상성 시계열에 강함
- 긴 결측 구간 처리 우수

**성능**:
- 폐수 처리 데이터: 장 간격에서 **SOTA 달성**
- 산업 데이터: 현실적 궤적 생성 (기존 방법은 평탄화)

**한계**:
- 도메인 특화 (산업 시스템)
- 제어 입력 정보 필요

#### 5.2.5 CD2-TSI (Cross-Domain Conditional Diffusion Models)

**발표**: 2025년 6월

**핵심 기여**:
- **도메인 적응**: 소스 도메인 지식을 타겟 도메인에 전이
- 도메인 공유 표현 + 도메인 특화 표현 분리
- **교차 도메인 일관성 정렬** (Cross-Domain Consistency Alignment)

**주요 기법**:
1. **데이터**: 주파수 기반 보간 (공유 스펙트럼 활용)
2. **모델**: 이중 디노이징 네트워크 (도메인별)
3. **알고리즘**: 출력 수준 도메인 불일치 조정

**성능**:
- 단일 도메인 CSDI 대비: **1.34-1.92% RMSE 개선**
- 도메인 이동 시나리오에서 특히 우수

**장점**:
- 새로운 도메인에 빠른 적응
- 제한된 타겟 데이터 활용

#### 5.2.6 최신 아키텍처 개선: NuwaTS (2024)

**발표**: 2024년 10월

**혁신**:
- **사전훈련된 언어 모델(PLM)** 재목적화
- 다중 도메인 범용 임퓨테이션 모델

**특성**:
- Zero-shot 도메인 전이 가능
- 플러그 앤 플레이 미세조정

### 5.3 종합 비교 표

| 방법 | 년도 | 핵심 기술 | 강점 | 약점 |
|---|---|---|---|---|
| **CSDI** | 2021 | 조건부 확산 + 2D Attention | 우수한 성능, 다목적성 | 느린 샘플링, 낮은 계산 효율 |
| **SSSD** | 2023 | State Space Model (S4) | 장기 의존성, 블랙아웃 강건 | 특성 간 상호작용 제한 |
| **MTSCI** | 2024 | 일관성 제약 + Mixup | 시간적 일관성 보장 | 추가 훈련 복잡성 |
| **tBN-CSDI** | 2025 | 주파수 적응 노이즈 | 희소 데이터 성능 | 계산 오버헤드 증가 |
| **CD2-TSI** | 2025 | 도메인 적응 | 도메인 전이 효과 | 도메인 수준 설정 필요 |
| **STDiff** | 2025 | 상태 전이 + 인과성 | 산업 시스템 특화 | 도메인 의존성 높음 |
| **NuwaTS** | 2024 | PLM 기반 | 범용성, 영점 학습 | 모델 크기 큼 |

### 5.4 최신 연구의 핵심 추세

#### 추세 1: 아키텍처 진화
**경로**: Transformer (CSDI) → State Space Model (SSSD) → 하이브리드 (2024-2025)

CSDI의 2D Attention에서 시작하여:
- S4의 선형성과 Transformer의 표현력 결합
- 예: Mamba 기반 구조 (2024-2025)

#### 추세 2: 신호 처리 고도화
**경로**: 시간 영역만 (CSDI) → 주파수 영역 통합 (tBN-CSDI, LSCD)

- 주파수 기반 보간
- Lomb-Scargle 변환 (비정규 샘플링)
- 멀티 스케일 분해

#### 추세 3: 제약 조건 통합
**경로**: 무제약 생성 (CSDI) → 일관성 제약 (MTSCI) → 동역학 제약 (STDiff)

#### 추세 4: 도메인 일반화
**경로**: 단일 도메인 (CSDI) → 도메인 적응 (CD2-TSI) → 범용 모델 (NuwaTS)

---

## 6. 앞으로의 연구에 미치는 영향과 고려사항

### 6.1 학문적 영향

#### 6.1.1 확산 모델의 새로운 응용 분야 개척

**CSDI의 기여**:
- 확산 모델을 **조건부 생성 문제**에 명시적으로 설계
- 이미지/음성 외 **시계열에 성공적 적용**
- 불완전한 데이터 처리의 새로운 패러다임 제시

**후속 연구 영향**:
1. **이상 탐지**: ImDiffusion (2023)
2. **이상 치 처리**: 확산 모델 기반 이상치 제거
3. **생성 모델**: 조건부 시계열 생성
4. **시공간 데이터**: 위도 전이 (Geospatial) 응용

#### 6.1.2 자기지도 학습 전략의 확대

**CSDI의 마스킹 전략 → 시계열 분야 표준화**:
- TimeGPT, Moment 등 기초 모델에서 채택
- 결측 패턴 학습의 핵심 기법으로 정립

#### 6.1.3 이론적 기초 제공

**조건부 확산 모델의 이론**:
- Score matching under conditioning
- 확산 모델의 일반화 한계 분석
- Bayesian 해석 제시

### 6.2 실무 응용 확대

#### 6.2.1 헬스케어 응용

**의료 센서 데이터**:
- ICU 모니터링에서 **40-65% 오류 감소**
- 의료 진단의 신뢰도 향상
- 규제 준수 (Data Quality) 개선

**확장 가능성**:
- 영상의학 (의료 이미지) 시계열
- 심전도(ECG) 등 생리신호
- 환자 예후 예측 정확도 개선

#### 6.2.2 환경 모니터링

**공기질, 수질, 기후 데이터**:
- 센서 고장으로 인한 결측 해결
- 환경 정책 수립의 데이터 신뢰도 향상

**확장 가능성**:
- 실시간 환경 오염 예측
- 기후 시나리오 시뮬레이션

#### 6.2.3 금융 응용

**금융 시계열**:
- 거래 데이터의 결측 처리
- 포트폴리오 리스크 평가 개선
- 고주파 거래 데이터 재구성

#### 6.2.4 산업 IoT

**STDiff 기반 응용**:
- 제조업 장비 센서 데이터
- 예측 유지보수(Predictive Maintenance)
- 공정 최적화

### 6.3 기술적 고려사항

#### 6.3.1 계산 효율성 개선의 필요성

**현재 문제**:
- T=50 단계의 반복 역과정
- 대규모 데이터셋에서 훈련 비용 높음
- 실시간 응용 어려움

**해결 방향**:
1. **고속 ODE 솔버**:
   - DPM-Solver (20-30단계로 감소)
   - 샘플링 속도 5-10배 향상
   
2. **증류(Distillation)**:
   - 가벼운 학생 모델로 지식 이전
   - 배포 시 지연시간 감소
   
3. **혼합 정밀도(Mixed Precision)**:
   - FP16 + FP32 조합으로 메모리 절감

**최근 진전**: CoSTI (2025) - Consistency Model 기반 고속화
- 단일 스텝 샘플링 가능
- CSDI 대비 166배 빠름

#### 6.3.2 확장성(Scalability) 개선

**고차원 시계열**:
- 현재: K(특성)과 L(길이) 각각 ~50-100
- 필요: K=1000+, L=10000+

**개선 방안**:
1. **특성 분해(Feature Decomposition)**
   - 높은 상관성 특성 그룹화
   - 부분별 임퓨테이션
   
2. **계층적 처리(Hierarchical Processing)**
   - 특성 트리 구조로 계산 단계 감소

#### 6.3.3 도메인 전이 강화

**기술적 과제**:
- 소스/타겟 도메인 분포 차이 극복
- 최소한의 타겟 데이터로 빠른 적응

**최신 접근법**:
- **CD2-TSI**: 도메인 일관성 정렬
- **Meta-learning**: MAML 기반 빠른 적응
- **Zero-shot**: 사전훈련 모델 직접 응용

#### 6.3.4 불확실성 정량화

**현재**: 확률적 분포 제공 (CRPS 지표)

**향상 방향**:
1. **보정된 신뢰도 구간**
   - 예측 신뢰도 영역의 정확도 개선
   
2. **베이지안 해석**
   - 사후 분포의 신뢰도 평가
   - 결정 이론 기반 활용

### 6.4 데이터 관점의 고려사항

#### 6.4.1 결측 메커니즘

**현재 가정**: Missing Completely At Random (MCAR)

**현실**:
- Missing At Random (MAR): 관찰된 값에 의존
- Missing Not At Random (MNAR): 결측값 자체에 의존

**필요한 개선**:
- 결측 메커니즘 명시적 모델
- MNAR 처리 방법론

#### 6.4.2 이상치(Outliers) 처리

**현재 문제**:
- 확산 모델은 이상치에 민감할 수 있음
- 로버스트성 평가 부재

**개선 방향**:
- 이상치 탐지 + 임퓨테이션 통합
- 강건한 손실 함수(Robust Loss)

#### 6.4.3 다중 대체(Multiple Imputation)

**CSDI의 장점**:
- 확률적 분포에서 다중 표본 생성 가능
- 불확실성 정량화

**활용 가능성**:
- 통계적 추론의 정확성 향상
- 하류 작업(Downstream Task)의 불확실성 전파

---

## 7. 결론

### 7.1 CSDI의 위치와 의의

CSDI는 **확산 모델이 시계열 임퓨테이션에 최적화될 수 있음을 최초 입증**한 groundbreaking work입니다. 단순한 성능 개선을 넘어:

1. **조건부 생성의 새로운 패러다임** 제시
2. **자기지도 학습의 실용적 전략** 제공
3. **다목적 시계열 모델** 가능성 시연

### 7.2 2020-2025 연구 동향 요약

| 시기 | 특성 | 대표 방법 |
|---|---|---|
| **2020-2021** | 기초 확립 | DDPM, CSDI, TimeGrad |
| **2022-2023** | 아키텍처 개선 | SSSD, TSDiff |
| **2024** | 제약 조건 통합 | MTSCI, Score-CDM |
| **2025** | 고급 일반화 | tBN-CSDI, CD2-TSI, STDiff |

### 7.3 앞으로의 주요 연구 방향

1. **계산 효율**: ODE 솔버, 증류 기반 10-100배 고속화
2. **일반화**: 도메인 전이, 영점 학습, 범용 모델
3. **이론**: 수렴 보장, 불확실성 정량화
4. **응용**: 헬스케어, 금융, 환경, 산업 IoT 확대

### 7.4 최종 평가

CSDI는:
- ✅ **강력한 성능** (40-65% 개선)
- ✅ **일반적 적용성** (보간, 예측, 생성 등)
- ✅ **이론적 기초** (조건부 확산)
- ⚠️ 계산 효율 개선 필요
- ⚠️ 초고차원 데이터 확장성 개선 필요

**평가**: AI 시계열 분석 분야의 **중추적 기여 논문**, 5년 이상의 영향력 예상

---

## 참고 자료

### 주요 논문
- Tashiro et al. (2021). CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation. NeurIPS 2021
- Alcaraz & Strodthoff (2023). SSSD: Diffusion-based Time Series Imputation with Structured State Space Models. TMLR 2023
- Zhou et al. (2024). MTSCI: Conditional Diffusion Model for Multivariate Time Series Consistent Imputation. CIKM 2024
- Zhang et al. (2025). CD2-TSI: Cross-Domain Conditional Diffusion Models for Time Series Imputation. arXiv:2506.12412
- 기타 2025년 신규 논문: tBN-CSDI, STDiff, LSCD, WaveStitch 등

### 관련 기초 이론
- Ho et al. (2020). Denoising Diffusion Probabilistic Models
- Song et al. (2021). Score-Based Generative Modeling through SDE
- Vaswani et al. (2017). Attention Is All You Need
- Gu et al. (2022). Efficiently Modeling Long Sequences with Structured State Spaces

---

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e6a56ffa-af8b-46ea-93b6-f80e60730104/NeurIPS-2021-csdi-conditional-score-based-diffusion-models-for-probabilistic-time-series-imputation-Paper.pdf)
[2](https://arxiv.org/abs/2506.12412)
[3](https://ieeexplore.ieee.org/document/11150127/)
[4](https://academic.oup.com/bioinformaticsadvances/article/doi/10.1093/bioadv/vbaf225/8261367)
[5](https://arxiv.org/abs/2508.19011)
[6](https://arxiv.org/abs/2508.11528)
[7](https://arxiv.org/abs/2503.06231)
[8](https://arxiv.org/abs/2506.17039)
[9](https://arxiv.org/abs/2503.01737)
[10](https://ieeexplore.ieee.org/document/11033512/)
[11](https://academic.oup.com/bib/article/26/Supplement_1/i26/8378030)
[12](https://arxiv.org/pdf/2307.00754.pdf)
[13](https://arxiv.org/pdf/2307.11494.pdf)
[14](https://arxiv.org/abs/2208.09399)
[15](https://arxiv.org/pdf/2501.06585.pdf)
[16](https://arxiv.org/html/2410.13338)
[17](http://arxiv.org/pdf/2501.19364.pdf)
[18](http://arxiv.org/pdf/2405.13075.pdf)
[19](http://arxiv.org/pdf/2107.03502v2.pdf)
[20](https://openreview.net/pdf/d69ac66c869e57b7ec656b862b72481633f1c35b.pdf)
[21](https://openreview.net/pdf?id=ZL5wlFMg0Y)
[22](https://www.sciencedirect.com/science/article/abs/pii/S0378779621001978)
[23](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5219053)
[24](https://arxiv.org/abs/2505.23309)
[25](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5156677)
[26](https://www.sciencedirect.com/science/article/abs/pii/S095070512401551X)
[27](https://yang-song.net/blog/2021/score/)
[28](https://kdd-milets.github.io/milets2019/papers/milets19_paper_2.pdf)
[29](https://arxiv.org/pdf/2406.08627.pdf)
[30](https://arxiv.org/abs/1704.04110)
[31](https://arxiv.org/html/2510.10807v2)
[32](https://arxiv.org/abs/2111.13606)
[33](https://arxiv.org/pdf/2302.02597.pdf)
[34](https://arxiv.org/pdf/2508.02621.pdf)
[35](https://arxiv.org/abs/2011.13456)
[36](https://arxiv.org/abs/2508.18921)
[37](https://arxiv.org/html/2510.07793v1)
[38](https://www.semanticscholar.org/paper/dcb7ca9c44181ee9516713160d5f42aa4488a12e)
[39](https://arxiv.org/abs/2503.01157)
[40](https://arxiv.org/abs/2510.24028)
[41](http://www.proceedings.com/079017-1496.html)
[42](https://www.tandfonline.com/doi/full/10.1080/01431161.2025.2527988)
[43](https://ieeexplore.ieee.org/document/10843102/)
[44](https://www.mdpi.com/2079-9292/13/19/3898)
[45](https://arxiv.org/abs/2412.00772)
[46](https://ieeexplore.ieee.org/document/10822707/)
[47](https://ieeexplore.ieee.org/document/10559898/)
[48](https://dl.acm.org/doi/pdf/10.1145/3643035)
[49](http://arxiv.org/pdf/2405.17478.pdf)
[50](https://arxiv.org/html/2405.15317)
[51](https://arxiv.org/pdf/2412.00772.pdf)
[52](http://arxiv.org/pdf/2402.05960.pdf)
[53](https://arxiv.org/pdf/2110.09410.pdf)
[54](https://arxiv.org/pdf/2404.11269.pdf)
[55](https://pmc.ncbi.nlm.nih.gov/articles/PMC10457853/)
[56](https://academic.oup.com/bioinformaticsadvances/article/5/1/vbaf225/8261367)
[57](https://openreview.net/pdf?id=9nXgWT12tb)
[58](https://www.ijcai.org/proceedings/2024/0424.pdf)
[59](https://openreview.net/pdf?id=hHiIbk7ApW)
[60](https://arxiv.org/abs/2506.02694)
[61](https://arxiv.org/html/2412.03068v2)
[62](https://arxiv.org/html/2506.02694v1)
[63](https://arxiv.org/html/2506.12412v1)
[64](https://arxiv.org/pdf/2506.12412.pdf)
[65](https://www.arxiv.org/pdf/2408.05740.pdf)
[66](https://arxiv.org/html/2511.19497v1)
[67](https://arxiv.org/html/2404.18886v5)
[68](https://arxiv.org/abs/2308.12874)
[69](https://arxiv.org/html/2408.05740v1)
[70](https://arxiv.org/abs/2510.06680)
[71](https://arxiv.org/html/2509.22295v1)
[72](https://www.sciencedirect.com/science/article/abs/pii/S003132032400904X)
[73](https://pure.korea.ac.kr/en/publications/transformer-based-multivariate-time-series-anomaly-detection-usin)
[74](https://www.vldb.org/pvldb/vol17/p359-zhang.pdf)
[75](https://openreview.net/forum?id=VVJ6Ck9JBl)
