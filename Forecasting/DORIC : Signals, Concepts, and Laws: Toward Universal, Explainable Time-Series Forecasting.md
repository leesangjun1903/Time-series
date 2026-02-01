
# DORIC: Signals, Concepts, and Laws: Toward Universal, Explainable Time-Series Forecasting

## 1. 핵심 주장 및 기여 요약

"Signals, Concepts, and Laws: Toward Universal, Explainable Time-Series Forecasting"는 다변량 시간 시리즈 예측에서 **정확도(accuracy), 해석 가능성(explainability), 물리적 신뢰성(physical credibility)의 동시 달성**을 목표로 한다. Sydney 대학의 Ma, Gao, Tran이 제시한 DORIC (Domain-Universal, ODE-Regularized, Interpretable-Concept Transformer)은 세 가지 핵심 주장으로 요약된다.

**주요 주장:**
첫째, 기존 Transformer 기반 시간 시리즈 예측 모델은 세 가지 구조적 격차를 지닌다. (1) 순수 데이터 중심 Transformer는 에너지 보존이나 질량 균형 같은 물리 법칙을 위반할 수 있어 안전 중심 도메인에서 배포 불가능하다. (2) 다변량 설정에서 사후 해석(post-hoc explanation) 방법의 신뢰도가 저하된다. (3) 대부분의 특화 아키텍처는 특정 도메인에 맞춰져 있어 표본 속도, 노이즈 특성, 계절성이 다양한 데이터에 일반화하지 못한다.

둘째, 이를 해결하기 위해 DORIC은 **5차원 개념 보틀넥** (concept bottleneck)과 **물리 정보 기반 ODE 헤드**를 결합한다. 이 설계는 모든 정보가 5개의 해석 가능한 개념(수준, 성장률, 순간 전력, 주기 진폭, 국소 변동성)을 통과하도록 강제하고, 이 개념들을 첫 원리 제약(first-principles constraints)에 따라 예측과 결합한다.

셋째, DORIC은 **도메인 보편성**을 실현한다. 6개의 이질적 데이터셋(전력, 교통, 기후, 질병, 환율, 에너지 변압기)에서 고정된 하이퍼파라미터를 사용하여 12개 MSE/MAE 메트릭 중 8개에서 최저 오류를 달성한다.

## 2. 해결 문제, 제안 방법, 모델 구조

### 2.1 문제 정의

실시간 시간 시리즈 예측은 전력망 관제, 도시 이동성 제어, 역학 감시, 고빈도 거래 등의 중요 분야에서 필수적이다. 기존 접근법의 한계는 다음과 같다.

- **ARIMA 같은 고전적 방법**: 통계적 엄밀성은 제공하나 비정상 데이터나 잠재 물리 동역학에 약함
- **일반적 Transformer 아키텍처**: 장거리 의존성을 잘 포착하지만 물리 제약 없이 부정확한 궤적 생성 가능
- **효율 중심 Transformer** (Informer, FEDformer): 계산 효율은 개선했으나 여전히 도메인 특화 또는 해석 불가
- **분해 기반 방법** (Autoformer): 추세-계절성 분해 도입했으나 다변량 설정에서 해석 한계

### 2.2 제안 방법론

DORIC은 인코더-디코더 구조를 기본으로 하며 세 가지 핵심 모듈을 추가한다.

#### (1) 입력 처리 및 인코더

시간 윈도우 $y_{t-L}^{t-1} \in \mathbb{R}^L$에 대해 위치 임베딩을 더한다:

$$H^0 = y_{t-L}^{t-1} W_e + P_{position}$$

여기서 $W_e \in \mathbb{R}^{1 \times d}$는 임베딩 행렬, 위치 인코딩은:
$$P_{i,2k} = \sin\left(\frac{i}{10^{4k/d}}\right), \quad P_{i,2k+1} = \cos\left(\frac{i}{10^{4k/d}}\right)$$

#### (2) 마스크된 다중 헤드 주의

$N_L$개의 Transformer 블록을 스택하며, 각 블록은 사전 정규화(pre-norm) + 마스크된 자기 주의(masked self-attention) + 잔차 연결:

$$H^i = \text{LN}(H^{i-1} + \text{MHA}(H^{i-1}))$$

$$\text{MHA}(H) = \sum_{h=1}^{H} \text{head}_h V_h^o W_o$$

$$\text{head}_h = \text{softmax}\left(\frac{Q_h K_h^T}{\sqrt{d}} \odot M\right)$$

마스킹 행렬 $M$은 인과성(causality)을 보장한다.

마지막 위치의 토큰을 추출하여 잠재 임베딩을 획득:
$$z_t = H^{N_L}_L \in \mathbb{R}^d$$

#### (3) 개념 보틀넥 계층

5차원 개념 벡터로 압축하여 해석 가능한 인터페이스를 만든다:

$$c_t = g(z_t) = \sigma(W_2 \sigma(W_1 z_t + b_1) + b_2), \quad c_t \in \mathbb{R}^5$$

여기서 $\sigma$는 ReLU, $W_1 \in \mathbb{R}^{d \times d'}, W_2 \in \mathbb{R}^{5 \times d'}$.

**개념의 물리적 의미:**
- $c_{1,t}$ (수준): 길이 $\tau$인 슬라이딩 평균
- $c_{2,t}$ (성장): 국소 속도 (1계 유한 차분)
- $c_{3,t}$ (순간 전력): $y_{t-1} \times c_{2,t}$ (에너지 개념)
- $c_{4,t}$ (주기 진폭): 주파수 영역의 첫 번째 조화 성분의 크기
- $c_{5,t}$ (국소 변동성): 제거된 신호의 표준편차

#### (4) 분석적 소프트 타겟 (Soft Targets)

개념을 인간이 해석 가능한 통계량으로 감독한다:

$$\bar{c}_{1,t} = \frac{1}{\tau} \sum_{s=t-\tau}^{t-1} y_s \quad \text{(sliding mean)}$$

$$\bar{c}_{2,t} = y_{t-1} - y_{t-2} \quad \text{(velocity)}$$

$$\bar{c}_{3,t} = y_{t-1} \bar{c}_{2,t} \quad \text{(instantaneous power)}$$

$$[a_1, b_1] = \text{DFT}_1(y_t^{t-1} - \bar{c}_{1,t}), \quad \bar{c}_{4,t} = 2\sqrt{a_1^2 + b_1^2}$$

$$\bar{c}_{5,t} = \frac{1}{\tau} \sum_{s=t-\tau}^{t-1}(y_s - \bar{c}_{1,t})^2 \quad \text{(local volatility)}$$

모든 계산이 인과적이므로 (미래 데이터 미사용) 온라인 배포 가능.

#### (5) 물리 정보 기반 ODE 헤드

개념을 예측과 연결하는 1차 ODE:

$$\frac{dy}{dt} = \beta_0 + \sum_{k=1}^5 \beta_k c_k(t) - \alpha(y - c_{1,t})$$

**해석:**
- $\beta_0$: 상수 기저 드리프트
- $\beta_k$: 각 개념의 결합 가중치
- $\alpha$: 평균 회귀 속도 (수준 $c_{1,t}$으로의 이완)

이 ODE는 정확한 데이터 생성 메커니즘이 아니라, **고수준 동역학 템플릿**으로 작용한다.

#### (6) 물리 잔차

ODE와 개념 정의로부터 5가지 대수적 잔차를 유도한다:

$$R_{1,t} = \Delta_t c_{1,t} - c_{2,t} \quad \text{(level integrates velocity)}$$

$$R_{2,t} = \Delta_t c_{2,t} - \frac{c_{3,t}}{y_{t-1} + \epsilon} \quad \text{(acceleration-power link)}$$

$$R_{3,t} = c_{3,t} - y_{t-1} c_{2,t} \quad \text{(definition of power)}$$

$$R_{4,t} = \Delta_t c_{5,t}^2 - 2(y_{t-1} - c_{1,t})c_{2,t} \quad \text{(variance kinematics, Itô differential)}$$

$$R_{y,t} = \Delta_t y_{t-1} - F_t \quad \text{(ODE compliance)}, \quad F_t = \beta_0 + \sum_k \beta_k c_{k,t} - \alpha(y_{t-1} - c_{1,t})$$

여기서 $\Delta_t u_t = u_t - u_{t-1}$.

#### (7) 결합 손실 함수

$$L = L_{data\_fit} + \lambda_{phys}(t) L_{phys} + \lambda_{con} L_{concept} + \lambda_{reg}||\Theta||_2^2$$

$$L_{data\_fit} = \frac{1}{N}\sum_{t=L}^T ||y_t - \hat{y}_t||^2$$

$$L_{phys} = \frac{1}{|S|}\sum_{t \in S}(R_{1,t}^2 + R_{2,t}^2 + R_{3,t}^2 + R_{4,t}^2 + R_{y,t}^2)$$

$$L_{concept} = \frac{1}{N}\sum_{i=1}^N\sum_{t=L}^T ||c_{i,t} - \bar{c}_{i,t}||_2^2$$

**물리 ramp-up 일정:**
$$\lambda_{phys}(t) = \lambda_0 \log(1 + t), \quad 0 < \lambda_0 < 1$$

물리 손실의 가중치가 초기에 0에서 시작하여 점진적으로 증가하면서, SGD 스텝 크기는 감소하여 수렴성을 보장한다.

### 2.3 이론적 분석

**Theorem 1 (보편 표현 가능성):**
$f$가 컴팩트 집합 $K \subseteq \mathbb{R}^L$에서 연속이고, 그 잠재 동역학이 ODE (식 13)를 만족한다면, 모든 $\epsilon > 0$에 대해 매개변수 $\Theta$와 임베딩 너비 $d$가 존재하여:
$$\sup_{x \in K} ||f(x) - f_\Theta(x)|| \leq \epsilon$$

**증명 스케치:** 
1. 마스크된 자기 주의 인코더 $f$는 컴팩트 영역에서 연속 특성 맵 근사 가능 (universal approximation)
2. 2층 MLP 개념 보틀넥 $g$는 $z$로부터 개념 근사 가능
3. 2층 MLP 물리 헤드 $h$는 ODE의 1단계 풀이 근사 가능
4. 합성 오차 경계로부터 전체 근사 가능성 도출

**Theorem 2 (물리 ramp-up을 포함한 SGD):**
Assumption 1 (L-Lipschitz 기울기, 유계 분산, 적절한 스텝 크기) 하에서:
$$\lim_{t \to \infty} \mathbb{E}[L] = 0, \quad \lim_{t \to \infty} \mathbb{E}[L_{phys}] = 0$$

**의미:** 
- 훈련 손실과 물리 위반이 모두 0으로 수렴
- 물리 제약이 최종적으로 준수됨
- 물리 제약이 데이터 맞춤과 경합하지 않음 (Feasibility-First-Then-Refinement)

**훈련 동역학 (Figure 6):**
- **Phase 1**: 물리 잔차 빠른 붕괴 (초기 에포크에서 0에 가까움) → 물리적 가능성 달성
- **Phase 2**: 개념 정렬 점진적 개선 → 식별 가능한 기하학적 구조
- **Phase 3**: 데이터 맞춤 개선 → 가능 매니폴드 내에서 최적화

## 3. 성능 향상 및 일반화

### 3.1 정량적 성능 평가

**벤치마크 결과 (Table 1):**

| 데이터셋 | DORIC MSE | TimeMixer MSE | 개선율 | 최저값 |
|---------|-----------|--------------|--------|-------|
| Electricity | 0.138 | 0.129 | -6.9% | ✗ |
| Traffic | 0.313 | 0.360 | **13.1%** | ✓ |
| Weather | 0.007 | 0.147 | **95.2%** | ✓ |
| Illness | 0.869 | 0.877 | **0.9%** | ✓ |
| Exchange | 0.051 | 0.117 | **56.4%** | ✓ |
| ETT | 0.111 | 0.164 | **32.3%** | ✓ |

DORIC은 6개 중 5개 데이터셋에서 TimeMixer를 능가하며, Weather 데이터셋에서 특히 탁월하다 (95.2% 개선).

### 3.2 절제 연구 (Ablation Studies)

각 컴포넌트의 중요성을 정량화한다 (Table 2):

| 절제 변경 | 평균 MSE | 증가율 |
|----------|---------|-------|
| DORIC (전체) | 0.328 | - |
| $\lambda_{phys} = 0$ (물리 제거) | 0.547 | +63% |
| $\lambda_{con} = 0$ (개념 정렬 제거) | 0.698 | +127% |
| 공유 인코더 → 5개 분리 헤드 | 0.382 | +76% |

**분석:**
- **물리 손실 제거 (+63%)**: 트래픽(Traffic)과 환율(FX)같은 스파이키(spiky) 도메인에서 특히 큼. 물리 제약이 물리적으로 불가능한 가속도 억제.
- **개념 정렬 제거 (+127%)**: 가장 파괴적. 5개 개념이 단순히 보기 좋은 것이 아니라 예측의 필수 구조.
- **공유 인코더 제거 (+76%)**: 글로벌 컨텍스트가 개념 간 상호작용에 필수적임. 분리 헤드는 독립적 최적화로 정보 손실.

### 3.3 노이즈 강건성

30% 가우시안 노이즈 주입 시 (Table 9,  참조):

$$\text{MSE Ratio} = \frac{\text{MSE}_{noisy}}{\text{MSE}_{clean}}$$

| 모델 | MSE Ratio |
|------|-----------|
| DORIC | 1.120 (±0.060) |
| FEDformer | 1.470 (±0.080) |
| LogTrans | 1.510 (±0.090) |

DORIC의 개념 분포 변화는 KL-divergence < 0.04 nats로 매우 안정적. **물리 제약이 노이즈로 증폭된 가속도 억제.**

### 3.4 해석 가능성 분석

#### (1) 개념-타겟 정렬

학습된 개념 $c_{k,t}$와 분석적 타겟 $\bar{c}_{k,t}$ 간 결정 계수 $R^2$ (Table 5):

| 데이터셋 | 수준 | 성장 | 전력 | 주기 | 변동성 | 평균 |
|---------|------|------|------|------|--------|------|
| Electricity | 0.94 | 0.83 | 0.81 | 0.90 | 0.88 | **0.87** |
| Traffic | 0.92 | 0.79 | 0.78 | 0.87 | 0.84 | **0.84** |
| Weather | 0.91 | 0.74 | 0.72 | 0.86 | 0.81 | **0.81** |
| ETT | 0.93 | 0.77 | 0.75 | 0.88 | 0.83 | **0.83** |
| Illness | 0.89 | 0.82 | 0.80 | 0.69 | 0.76 | **0.79** |
| Exchange | 0.88 | 0.78 | 0.77 | 0.62 | 0.90 | **0.79** |

**해석:** 수준(level) 개념이 가장 강한 정렬(0.88-0.94), 주기 진폭이 환율 같은 비주기 데이터에서 낮음(0.62). 전체적으로 높은 개념-통계 일관성.

#### (2) 정규화된 물리 잔차 (Table 6)

신호 크기 대비 물리 위반 정도:

| 잔차 | 평균 | 표준편차 |
|------|------|---------|
| $R_1$ (수준) | 0.028 | 0.015 |
| $R_2$ (성장) | 0.033 | 0.019 |
| $R_3$ (전력) | 0.041 | 0.024 |
| $R_4$ (주기) | 0.037 | 0.021 |
| $R_y$ (ODE) | 0.026 | 0.017 |

모든 값이 신호의 표준편차 대비 5% 미만. **학습된 동역학이 명시된 물리 관계를 대부분 준수.**

#### (3) 학습된 ODE 계수 (Table 7)

각 개념의 정규화된 결합 가중치 $\beta_k$ (단위 분산):

| 데이터셋 | $\alpha$ | $\beta_1$(수준) | $\beta_2$(성장) | $\beta_3$(전력) | $\beta_4$(주기) | $\beta_5$(변동) |
|---------|---------|---------|---------|---------|---------|---------|
| Electricity | 0.52 | 0.91 | 0.28 | 0.07 | 0.63 | 0.19 |
| Traffic | 0.49 | 0.88 | 0.31 | 0.05 | 0.58 | 0.22 |
| Weather | 0.47 | 0.84 | 0.26 | 0.09 | 0.41 | 0.27 |
| Exchange | 0.58 | 0.69 | 0.42 | 0.18 | 0.09 | **0.57** |

**패턴:**
- 모든 도메인에서 $\alpha$ (이완 속도)는 안정적 양수 → 평균 회귀 동역학
- 계절성 강한 도메인 (전력, 교통): $\beta_4$ (주기 진폭) 크고, 변동성 $\beta_5$ 작음
- 금융 도메인 (환율): $\beta_5$ (변동성) 우세 (0.57), $\beta_4$ (주기) 미미 (0.09) → 도메인 지식과 일치

## 4. 모델의 일반화 성능 향상 메커니즘

### 4.1 개념 기반 정규화

**저차원 보틀넥의 효과:**

5차원 개념 벡터로 강제 압축하면:
1. **정보 기하학적 단순화**: $d$ 차원 잠재 공간 → 5차원 개념 공간
2. **특성 추출의 자동화**: 신경망이 수동으로 특성을 설계할 필요 없음
3. **과적합 감소**: 작은 모델 용량이 학습 가능한 파라미터 제한

**예:** Traffic 데이터셋에서 개념 정렬 제거 시 +127% 성능 저하는 이 저차원 구조가 본질적 정보를 포착함을 증명.

### 4.2 물리 제약을 통한 매니폴드 정규화

$$L_{phys} = \frac{1}{|S|} \sum_{t} (R_1^2 + R_2^2 + R_3^2 + R_4^2 + R_y^2)$$

**역할:**
1. **가능성 먼저(Feasibility-First)**: Phase 1에서 물리적으로 가능한 궤적 영역으로 제한
2. **구조화된 최적화**: 가능 매니폴드 내에서만 데이터 맞춤 최적화
3. **외삽 안정성**: 훈련 데이터 밖의 영역에서도 물리 법칙이 예측 안정화

**정량적 증거:** 노이즈 강건성에서 DORIC이 FEDformer 대비 30% 우수 (MSE ratio 1.12 vs 1.47). 물리 제약이 노이즈로 유도된 불안정성 흡수.

### 4.3 도메인 불변 개념

5개 개념이 모든 도메인에 적용되는 이유:

1. **신호 처리 기본**: 수준, 속도, 전력은 모든 시간 시리즈의 본질적 특성
2. **푸리에 분석**: 주기 진폭은 신호의 고주파/저주파 분해
3. **통계적 기초**: 변동성은 데이터 산포의 보편적 척도

**반증:** 모든 6개 데이터셋에서 개념-타겟 $R^2 > 0.79$. 도메인 특화 개념이 아니어도 충분한 예측력.

### 4.4 공유 인코더의 교차 개념 상호작용

절제 연구에서 공유 인코더 제거 시 +76% 성능 저하:

$$\text{MSE}_{5heads} = 0.382, \quad \text{MSE}_{shared} = 0.328$$

**메커니즘:**
- 공유 Transformer 인코더가 각 개념에 필요한 정보를 통합적으로 처리
- 예: 변동성 $c_5$를 계산하려면 수준 $c_1$ 정보 필요 (잔차 계산)
- 분리 헤드는 이러한 상호작용 표현 불가

### 4.5 훈련 동역학의 3단계

**Figure 6 분석 (모든 데이터셋에서 재현):**

| Phase | 목표 | 관찰 | 이론적 근거 |
|-------|------|------|-----------|
| **1. 물리 붕괴** | 가능성 달성 | 초기 에포크에서 $L_{phys}$ 급격히 감소 | Theorem 2: 물리 손실이 먼저 0으로 |
| **2. 개념 정렬** | 잠재 기하 | $L_{concept}$ 점진적 감소 | 개념 감독이 안정적으로 작동 |
| **3. 데이터 맞춤** | 예측 정확도 | 총 MSE 단조 개선 | 가능 매니폴드 내에서 최적화 |

**임계점:** 물리와 데이터 손실의 기울기 정렬 (Figure 5):

$$\alpha_t = \frac{\langle \nabla L_{phys}(t), \nabla L_{data}(t) \rangle}{||\nabla L_{phys}||\_2 \cdot ||\nabla L_{data}||_2}$$

후기 훈련에서 $\alpha_t > 0$이 되어 물리와 데이터 목표가 **협력적**(non-adversarial).

## 5. 한계 및 실제 적용 고려사항

### 5.1 개념 설계의 제한성

**현재 고정된 5개 개념의 문제:**

1. **도메인 특수 현상 미포착**: 물류 데이터의 주간 패턴, 금융의 마이크로구조(microstructure) 등
2. **개념 윈도우 민감도**: 슬라이딩 평균 윈도우 $\tau=50$ (데이터 의존적)
3. **개념 외삽성**: 자동화된 도메인 특화 개념 발견 메커니즘 부재

### 5.2 물리 모델의 경직성

**ODE의 한계:**
- Weather 데이터셋의 습도(Figure 3d): "Soft law" 변수로, 보존 법칙이 아닌 범위 제약
- DORIC은 진폭을 과도하게 축소 (20-60 스텝) → $\lambda_{phys}$ 스케줄 조정 필요
- 고도로 비선형 시스템에서는 1차 ODE 부족

**완화책:** 
- Per-channel $\lambda_{phys,j}$ 도입 (Table 12에서 시도)
- Robust loss (Huber, Quantile)로 극단값 처리

### 5.3 계산 효율

**Table 8 런타임 비교:**

| 모델 | 훈련 시간/에포크 | 추론 처리량 |
|------|---------|----------|
| Informer | 132s | 9.1k seq/s |
| FEDformer | 165s | 7.4k seq/s |
| PatchTST | 118s | 10.3k seq/s |
| TimeMixer | 124s | 9.8k seq/s |
| **DORIC** | **139s** | **9.0k seq/s** |

DORIC은 기준 대비 10-15% 느림. 개념 MLP, ODE 헤드, 물리 잔차 계산 오버헤드.

### 5.4 극단값 처리

**전력망 데이터의 이상치:** Figure 3a에서 첫 20 스텝 약간의 과대추정
- 표준 MSE는 이상치에 민감
- MAE는 더 안정적 (0.214 vs 0.224)
- Quantile loss 사용 시 RMSE 개선

### 5.5 도메인 적응 실패 사례

**ETT 데이터셋:** 
- DORIC MSE 0.111 vs AR (고전 모형) MSE 0.082
- AR 오라클이 더 우수한 드문 경우
- 원인: 극히 규칙적 시간 시리즈로 선형 모형 충분

## 6. 2020년 이후 최신 연구와의 비교 분석

### 6.1 물리 정보 기반 신경망 (PINNs) 계열

**PINNs (Raissi et al., 2019 이후 활발)**

- **핵심:** 미분 방정식을 손실에 포함하여 신경망 정규화
- **강점:** 물리 법칙 명시적 준수, 데이터 부족 시 유리
- **한계:** 
  - 도메인 특화 (예: CFD용 Navier-Stokes)
  - 해석 가능성 부족 (검은 상자)
  - 계산 비용 높음 (자동 미분)

**PINT (2025):** 조화 진동자 방정식으로 주기 동역학 제한
- **vs DORIC:** DORIC은 5개 개념의 보편성 추구, PINT는 특정 방정식 가정

### 6.2 Concept Bottleneck Models (CBM) 계열

**Koh et al. (2020) 원본**
- **설계:** 입력 → 개념 예측 → 레이블 (2단계)
- **주로 이미지 분류에 사용**
- **시간 시리즈 확장:** Minimal (대부분 시각 도메인)

**DORIC의 혁신:**
- 시간 시리즈에 **최초 적용**
- 개념에 물리 의미 결합
- ODE를 통한 동역학 모델링

**최신 CBM 발전:**

| 논문 | 년도 | 핵심 아이디어 | DORIC과의 차이 |
|------|------|-----------|-----------|
| Stochastic CBM | 2024 | 확률적 개념 의존성 | DORIC은 결정적 ODE |
| MCBM | 2025 | 정보 병목으로 정보 누수 제어 | DORIC은 물리로 정규화 |
| CB-LLM | 2025 | LLM으로 개념 생성 | DORIC은 고정 개념 (단순성) |

### 6.3 최신 Transformer 예측 모델 (2023-2025)

#### TimeMixer (Wang et al., 2024)
- **혁신:** 다중 스케일 혼합, 분해-혼합 아키텍처
- **성능:** 많은 벤치마크에서 SOTA
- **약점:** 해석 불가
- **DORIC vs TimeMixer:** 4개 데이터셋에서 DORIC 우수, 추가로 해석 가능성 + 물리 신뢰성

#### FreEformer (2025)
- **혁신:** DFT 기반 주파수 향상, 저랭크 주의 개선
- **성능:** 18개 벤치마크 SOTA 달성
- **vs DORIC:** 정확도는 비슷하거나 우수하나, 해석 불가, 물리 준수 안 함

#### Timer-XL / TimeFound (2025)
- **특징:** 초거대 모델 (200M-710M 파라미터)
- **장점:** 영점 학습(zero-shot) 강력
- **단점:** 복잡성, 해석 불가, 배포 비용
- **DORIC:** 소형 모델, 해석 중심, 경량

### 6.4 해석 가능 시간 시리즈 (2024-2025)

#### iTFKAN (Interpretable Time Forecasting with KAN)
- **기초:** Kolmogorov-Arnold Network (기호화)
- **장점:** 고해석성, 대수 공식으로 변환 가능
- **vs DORIC:** DORIC은 물리 동역학 명시, iTFKAN은 KAN 기하 기반

#### WEITS (Wavelet-Enhanced Residual Framework)
- **접근:** 웨이블릿 분해 + 주파수 인식
- **vs DORIC:** 주파수 도메인 분석 vs 시간-개념 도메인

#### TFT (Temporal Fusion Transformer, 여전히 활발)
- **특징:** 변수 선택, 게이팅, 주의 해석
- **한계:** 개념 수준 해석 부족, 물리 제약 없음
- **DORIC:** 더 명시적인 개념 + 물리

### 6.5 ODE 기반 시간 시리즈

**Neural ODE (Chen et al., 2018, 지속적 영향)**
- **원리:** 연속 시간 동역학, 적응형 계산
- **응용:** 불규칙 샘플링 데이터, 정규화 흐름(normalizing flows)
- **vs DORIC:** DORIC은 ODE를 물리 정규화로 사용, NODE는 일반적 동역학

**AdaNODEs (2025)**
- **목표:** 테스트 시간 적응
- **vs DORIC:** 적응은 DORIC의 물리 안정성에 이미 포함

### 6.6 벤치마킹 및 평가 프레임워크 (2024-2025)

**TFB (Time Series Forecasting Benchmark)**
- **규모:** 8,068 시계열, 통계/ML/DL 방법 25개
- **발견:** 통계 방법(VAR, 선형 회귀)이 때로 우수 (특히 강력한 정상성)
- **시사:** DORIC 같은 신경 방법의 가치는 복잡도에서 비롯

**Fidel-TS (High-Fidelity Benchmark)**
- **기여:** 인과 건전성, 누수 제거
- **발견:** 기존 벤치마크 편향 노출
- **DORIC의 강점:** 6개 이질 도메인에서 고정 하이퍼파라미터 = 진정한 일반화

### 6.7 종합 비교 매트릭스

| 측면 | DORIC | TimeMixer | FreEformer | PINNs | iTFKAN | CB-LLM |
|------|-------|-----------|-----------|-------|--------|--------|
| **정확도** | 우수 | 최고 | 최고 | 낮음 | 우수 | 낮음 |
| **해석 가능성** | 최고* | 낮음 | 낮음 | 매우낮음 | 높음** | 높음*** |
| **물리 신뢰성** | 명시적 | 없음 | 없음 | 명시적 | 없음 | 없음 |
| **도메인 보편성** | 높음 | 낮음 | 낮음 | 낮음 | 낮음 | 중간 |
| **계산 효율** | 중간 | 높음 | 높음 | 낮음 | 높음 | 낮음 |
| **이론적 기초** | 최고 | 약함 | 약함 | 강함 | 중간 | 약함 |
| **모델 크기** | 소형 | 소형 | 소형 | 중형 | 소형 | 거대 |

*: 개념 기반 + 물리 잔차 시각화
**: KAN 기호화
***: LLM 개념 제너레이션

## 7. 향후 연구에 미치는 영향 및 고려 사항

### 7.1 이 연구의 학문적 영향

1. **해석 가능성-정확도 통합의 모델 사례**
   - "해석성과 정확도는 트레이드오프" 신화 타파
   - 절제 연구: 개념 정렬 제거 시 +127% 성능 저하 → 해석성이 예측 구조의 본질

2. **시간 시리즈에 Concept Bottleneck 적용**
   - CBM이 단순히 이미지 분류 기법이 아님 입증
   - 시간 동역학과 개념의 수학적 결합 체계 제시

3. **물리 정보 학습의 새 패러다임**
   - PINNs처럼 모든 방정식을 알 필요 없음
   - 고수준 동역학 템플릿(ODE) + 소프트 감독(soft targets)으로 충분

4. **도메인 불변 설계의 중요성**
   - "한 모델, 모든 도메인" 실현의 첫 사례
   - 고정 개념이 아니라 도메인 특화 개념보다 나을 수 있음

### 7.2 향후 연구 방향

#### (1) 적응형 개념 발견
**문제:** 현재 5개 개념이 고정
**해결책:**
- 메타 학습으로 도메인별 최적 개념 자동 발견
- 다항 기저 함수 또는 주성분 분석(PCA)으로 동적 개념 생성
- LLM을 이용한 도메인 특화 개념 제안

**기대 효과:** 극히 비표준 도메인(예: 암호화폐, IoT 센서)에서 성능 향상

#### (2) 고차 또는 비선형 ODE
**현재:** 1차 선형 ODE만 사용
**확장:**
- 2차 ODE로 가속도 명시 모델링
- 비선형 항 추가 (예: $\frac{dy}{dt} = f(y, c) + g(y)^2$)
- 신경 ODE 네트워크 (learned $f$ 함수)

**기대 효과:** 극도로 복잡한 동역학(난류, 금융 위기) 포착

#### (3) 개념 개입(Intervention) 및 반사실(Counterfactual)
**아이디어:** CBM처럼 테스트 시 개념 값 수정
- 물리학자: "이 전력망에서 변동성을 25% 줄이면?"
- 모델: 개념 $c_5$를 조정하고 예측 업데이트

**도전:** 물리 제약 하에서 개입의 일관성 보장

#### (4) 다중 스케일 또는 계층적 개념
**다중 해상도:** 
- 짧은 윈도우(분 단위) 개념 vs 긴 윈도우(일 단위) 개념
- 계층적 관계: 저수준 개념 → 고수준 개념

**예:** 전력 예측에서 (5분 변동성, 일일 트렌드, 계절성)의 3계층

#### (5) 도메인 간 전이 학습
**가설:** 5개 개념이 도메인 불변이면 전이 학습 가능
- 전력망 모델 → 교통 도메인 전이
- 최소한의 파인튜닝

**실험:** 몇 개 도메인은 사전 훈련, 새 도메인에 신속 적응

#### (6) 불확실성 정량화
**현재:** 점 예측만
**추가:**
- 확률적 개념: $c_{k,t} \sim p(\cdot; \mu, \sigma)$
- 예측 불확실성: $\hat{y}_t \pm \text{CI}$
- 물리 제약 하에서의 신뢰 구간

#### (7) 그래프 신경망과의 결합
**다변량 시간 시리즈에서:**
- 변수 간 의존성을 그래프로 모델링
- GNN 엔코더 + DORIC 개념 헤드
- 공간-시간 복합 예측

### 7.3 실무 적용 시 주의사항

#### (1) 개념 윈도우 선택
- 데이터 주기에 맞춰 $\tau$ 설정 필수
- 자동 선택 알고리즘 필요 (예: 자기상관 기반)

#### (2) 물리 손실 가중치 스케줄
- Per-channel $\lambda_{phys,j}$ 권장 (Table 12)
- Robust loss 사용 (Huber) 극단값이 많을 때

#### (3) 온라인 학습 시 안정성
- 개념 정의가 인과적이므로 온라인 배포 가능
- 개념 통계의 드리프트 모니터링 필요

#### (4) 해석 검증
- 학습된 ODE 계수가 도메인 지식과 일치하는지 확인
- 예: 계절성 강한 데이터에서 $\beta_4$ (주기) 커야 함

#### (5) 다른 방법과의 앙상블
- DORIC의 물리 신뢰성 + TimeMixer의 정확도
- 앙상블 가중치를 데이터 특성에 따라 조정

## 결론

**DORIC은 시간 시리즈 예측에서 정확도, 해석 가능성, 물리적 신뢰성을 동시에 달성하는 획기적 아키텍처**이다. 5차원 개념 보틀넥으로 강제 압축하고, 물리 정보 기반 ODE로 개념-예측을 연결하며, 도메인 불변 설계로 보편성을 실현한다.

**핵심 성과:**
- 6개 도메인에서 고정 하이퍼파라미터로 경쟁력 있는 성능 (12개 메트릭 중 8개 최저)
- 절제 연구로 각 컴포넌트의 필수성 입증 (물리 +63%, 개념 +127%, 공유 +76%)
- 강한 해석 가능성 (개념 정렬 $R^2$ > 0.79, 물리 잔차 < 5%)
- 이론적 보증 (Universal expressiveness, SGD convergence)

**한계:**
- 극단값 처리 미흡 (Huber loss 필요)
- 계산 효율 10-15% 낮음
- 고정 개념이 모든 도메인에 최적은 아님 (ETT에서 AR에 뒤짐)

**향후 가능성:**
적응형 개념, 고차 ODE, 개념 개입, 전이 학습 등으로 더욱 강력하고 유연한 시간 시리즈 예측 시스템으로 발전할 것으로 예상된다. 특히 안전 중심 도메인(전력망, 의료)에서 즉각적 적용 가능성이 높다.

<span style="display:none">[^1_1][^1_10][^1_11][^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_2][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_3][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_4][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_5][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_6][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_7][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_8][^1_80][^1_81][^1_82][^1_83][^1_84][^1_85][^1_86][^1_87][^1_88][^1_89][^1_9][^1_90][^1_91][^1_92]</span>

<div align="center">⁂</div>

[^1_1]: 2508.01407v3.pdf

[^1_2]: https://dl.acm.org/doi/10.1145/3757749.3757774

[^1_3]: https://www.mdpi.com/2073-4433/16/3/292

[^1_4]: https://ieeexplore.ieee.org/document/11345094/

[^1_5]: https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13802/3067869/A-transformer-based-approach-for-multivariate-time-series-forecasting-of/10.1117/12.3067869.full

[^1_6]: https://www.mdpi.com/2413-4155/7/1/7

[^1_7]: https://www.mdpi.com/1424-8220/25/3/652

[^1_8]: https://arxiv.org/abs/2501.13989

[^1_9]: https://www.mdpi.com/2227-7390/13/5/814

[^1_10]: https://dl.acm.org/doi/10.1145/3637528.3671928

[^1_11]: https://ieeexplore.ieee.org/document/10926918/

[^1_12]: https://arxiv.org/html/2411.01419v1

[^1_13]: http://arxiv.org/pdf/2211.14730v2.pdf

[^1_14]: https://arxiv.org/pdf/2502.13721.pdf

[^1_15]: https://arxiv.org/pdf/2202.01381.pdf

[^1_16]: http://arxiv.org/pdf/2410.04803.pdf

[^1_17]: http://arxiv.org/pdf/2503.17658.pdf

[^1_18]: https://arxiv.org/pdf/2209.03945.pdf

[^1_19]: https://arxiv.org/pdf/2503.04118.pdf

[^1_20]: https://arxiv.org/html/2510.07084v1

[^1_21]: https://arxiv.org/pdf/2502.04018.pdf

[^1_22]: https://www.arxiv.org/abs/2508.03269

[^1_23]: https://arxiv.org/html/2508.16641v1

[^1_24]: https://arxiv.org/abs/2405.08111

[^1_25]: https://arxiv.org/html/2504.16432v2

[^1_26]: https://arxiv.org/pdf/2601.18837.pdf

[^1_27]: https://pubmed.ncbi.nlm.nih.gov/39876937/

[^1_28]: https://pubmed.ncbi.nlm.nih.gov/34325021/

[^1_29]: https://arxiv.org/html/2507.02907v1

[^1_30]: https://arxiv.org/html/2502.04018v1

[^1_31]: https://arxiv.org/pdf/2305.14582.pdf

[^1_32]: https://arxiv.org/abs/2411.05793

[^1_33]: https://arxiv.org/abs/2506.03897

[^1_34]: https://arxiv.org/abs/2405.10877

[^1_35]: https://peerj.com/articles/cs-3001/

[^1_36]: https://www.youtube.com/watch?v=-zrY7P2dVC4

[^1_37]: https://research.google/blog/interpretable-deep-learning-for-time-series-forecasting/

[^1_38]: https://proceedings.mlr.press/v238/zhang24l.html

[^1_39]: https://www.mathworks.com/discovery/physics-informed-neural-networks.html

[^1_40]: https://papers.neurips.cc/paper_files/paper/2020/file/47a3893cc405396a5c30d91320572d6d-Paper.pdf

[^1_41]: https://www.nature.com/articles/s41746-023-00853-4

[^1_42]: https://icml.cc/virtual/2025/poster/44262

[^1_43]: https://www.sciencedirect.com/science/article/pii/S0378383924002345

[^1_44]: https://www.sciencedirect.com/science/article/pii/S1532046421002057

[^1_45]: https://openreview.net/forum?id=kHEVCfES4Q\&noteId=mrNbq9EkQa

[^1_46]: https://arxiv.org/abs/2501.09298

[^1_47]: https://arxiv.org/abs/2305.14582

[^1_48]: https://www.semanticscholar.org/paper/9be1a1ad73aa42bef615507bc49617893cbb4346

[^1_49]: https://arxiv.org/abs/2408.01432

[^1_50]: https://ojs.aaai.org/index.php/AAAI/article/view/30109

[^1_51]: https://arxiv.org/abs/2401.01259

[^1_52]: https://arxiv.org/abs/2405.17575

[^1_53]: https://arxiv.org/abs/2410.15555

[^1_54]: https://www.semanticscholar.org/paper/2f62a1821bbf6b7b8dbc2a11fd3d900e5ebd5fe9

[^1_55]: https://arxiv.org/abs/2402.00912

[^1_56]: https://arxiv.org/abs/2410.06352

[^1_57]: https://www.semanticscholar.org/paper/8c8c15fa087cc54f3704f8f45b52703f5cc16d9f

[^1_58]: https://arxiv.org/abs/2407.04307

[^1_59]: https://arxiv.org/pdf/2209.09056.pdf

[^1_60]: https://arxiv.org/html/2412.07992

[^1_61]: http://arxiv.org/pdf/2311.05014.pdf

[^1_62]: https://arxiv.org/html/2408.02265v1

[^1_63]: https://arxiv.org/html/2310.02116

[^1_64]: https://arxiv.org/pdf/2501.19271.pdf

[^1_65]: https://arxiv.org/pdf/2502.13632.pdf

[^1_66]: https://arxiv.org/html/2506.04877v3

[^1_67]: https://pubmed.ncbi.nlm.nih.gov/40640232/

[^1_68]: https://arxiv.org/html/2509.24789v1

[^1_69]: https://arxiv.org/abs/2310.19660

[^1_70]: https://arxiv.org/html/2601.12893v1

[^1_71]: https://arxiv.org/pdf/2509.26468.pdf

[^1_72]: https://arxiv.org/html/2505.24492v2

[^1_73]: https://arxiv.org/html/2502.09885v1

[^1_74]: https://arxiv.org/abs/2510.18037

[^1_75]: https://openaccess.thecvf.com/content/CVPR2024/papers/Shang_Incremental_Residual_Concept_Bottleneck_Models_CVPR_2024_paper.pdf

[^1_76]: https://arxiv.org/html/2305.00338v3

[^1_77]: https://arxiv.org/html/2403.20150v1

[^1_78]: https://openaccess.thecvf.com/content/CVPR2025/papers/Yu_Language_Guided_Concept_Bottleneck_Models_for_Interpretable_Continual_Learning_CVPR_2025_paper.pdf

[^1_79]: https://arxiv.org/abs/2306.01674

[^1_80]: https://arxiv.org/html/2509.05215v1

[^1_81]: https://dl.acm.org/doi/10.5555/3524938.3525433

[^1_82]: https://www.datadoghq.com/blog/datadog-time-series-foundation-model/

[^1_83]: https://neurips.cc/virtual/2024/poster/95698

[^1_84]: https://emptymalei.github.io/deep-learning/time-series-deep-learning/timeseries.neural-ode/

[^1_85]: https://www.vldb.org/pvldb/vol17/p2363-hu.pdf

[^1_86]: https://sebastiancallh.github.io/post/neural-ode-weather-forecast/

[^1_87]: https://research.spec.org/icpe_proceedings/2021/proceedings/p189.pdf

[^1_88]: https://www.sciencedirect.com/science/article/pii/S1566253525005007

[^1_89]: https://ekimetrics.github.io/blog/2022/07/11/neural_ode/

[^1_90]: https://arxiv.org/html/2505.16705v1

[^1_91]: https://www.nature.com/articles/s41598-025-05958-2

[^1_92]: https://www.lgresearch.ai/blog/view?seq=424
