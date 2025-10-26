# Probabilistic Transformer for Time Series Analysis

## 1. 핵심 주장 및 주요 기여

**ProTran(Probabilistic Transformer)**은 다변량 시계열 분석을 위해 **상태공간모델(State-Space Models, SSMs)과 트랜스포머 아키텍처를 결합**한 확률적 생성 모형입니다. 논문의 핵심 주장은 다음과 같습니다:[1]

기존의 SSM 기반 접근법들(특히 RNN 기반)은 **장거리 시간 의존성을 포착하지 못하고**, 트랜스포머 기반 방법들은 **확률적 불확실성을 명시적으로 모델링하지 못한다**는 문제점을 지적합니다. ProTran은 이 두 접근법의 장점을 결합하여 다음 세 가지를 달성합니다:[1]

- **트랜스포머 기반 SSM 제안**: 잠재공간에서 비마르코프(non-Markovian) 동역학을 모델링하기 위해 어텐션 메커니즘을 사용하며, RNN을 완전히 제거
- **계층적 확률 모델 확장**: 표현력을 높이기 위해 여러 층의 확률 잠재변수를 계층 구조로 조직화
- **포괄적인 실험 검증**: 시계열 예측(forecasting)과 인간 동작 예측에서 경쟁 기준 모델을 크게 능가

## 2. 문제 정의 및 제안 방법

### 2.1 문제 설정

시계열 예측에서 다루는 문제는 **조건부 분포 예측**입니다:[1]

$$\text{Given: } x_{1:C} \text{ (context/과거 관측)}, \quad \text{Predict: } p(x_{C+1:T} | x_{1:C})$$

여기서 $$x_t \in \mathbb{R}^N$$은 시간 t에서의 N차원 다변량 관측값입니다. 기존 모델들의 제한:

- **Markovian SSM** (LDS 등): 선형 가정, 장거리 의존성 포착 불가
- **RNN 기반 SSM**: 정보 압축으로 인한 손실, 재귀적 처리로 인한 최적화 어려움
- **트랜스포머**: 확률적 불확실성 미모델링, 추론 메커니즘 부재

### 2.2 확률 모델 구조

**변분 추론(Variational Inference)** 프레임워크를 기반으로 하는 확률 모델:[1]

$$p_\theta(x_{1:T} | x_{1:C}) = \int p_\theta(x_{1:T} | z_{1:T}) p_\theta(z_{1:T} | x_{1:C}) dz_{1:T}$$

여기서 $$z_{1:T}$$는 잠재변수 수열입니다.

**생성 모델(Generative Model)**의 분해:

$$p_\theta(z_{1:T} | x_{1:C}) = \prod_{t=1}^{T} p_\theta(z_t | z_{1:t-1}, x_{1:C})$$

$$p_\theta(x_{1:T} | z_{1:T}) = \prod_{t=1}^{T} p_\theta(x_t | z_t)$$

**핵심 차별성**: 전통적인 마르코프 SSM과 달리, $$z_{t+1}$$은 $$z_t$$뿐 아니라 모든 선행 잠재변수 $$z_{1:t-1}$$에 의존합니다.[1]

**변분 하한(Variational Lower Bound)**:

$$\log p_\theta(x_{1:T}|x_{1:C}) \geq \sum_{t=1}^{T} \left( \mathbb{E}_q[\log p_\theta(x_t|z_t)] - \text{KL}(q_\phi(z_t|z_{1:t-1}, x_{1:T}) \| p_\theta(z_t|z_{1:t-1}, x_{1:C})) \right)$$

손실함수로는 **라플라스 분포(Laplace distribution)**를 가정하여 **L1 재구성 손실**을 사용합니다.[1]

### 2.3 멀티헤드 어텐션(Multi-Head Attention)

트랜스포머의 핵심 메커니즘:[1]

$$O_h = \text{Attention}(Q_h, K_h, V_h) = \text{Softmax}\left(\frac{Q_h K_h^T}{\sqrt{d}}\right) V_h$$

여기서 $$Q_h = QW_h^Q, K_h = KW_h^K, V_h = VW_h^V$$는 투영된 쿼리, 키, 값입니다.

**위치 임베딩(Positional Embedding)**:[1]

$$\text{Position}(t) = [p_t^{(1)}, \ldots, p_t^{(d)}], \quad p_t^{(i)} = \begin{cases} \sin(t \cdot c^{i/d}) & \text{even } i \\ \cos(t \cdot c^{i/d}) & \text{odd } i \end{cases}$$

## 3. 모델 아키텍처

### 3.1 단층 Probabilistic Transformer

**생성 프로세스** (Generative Model):

$$h_t = \text{LayerNorm}(\text{MLP}(x_t) + \text{Position}(t))$$

비마르코프 동역학을 모델링하기 위해 두 단계의 어텐션과 결정론적 성분을 결합한 **숨겨진 표현** $$w_t$$를 사용합니다:[1]

$$\bar{w}_t = \text{LayerNorm}(w_{t-1} + \text{Attention}(w_{t-1}, w_{1:t-1}, w_{1:t-1}))$$

$$\hat{w}_t = \text{LayerNorm}(\bar{w}_t + \text{Attention}(\bar{w}_t, h_{1:C}, h_{1:C}))$$

$$z_t \sim \mathcal{N}(z_t; \text{MLP}(\hat{w}_t), \text{Softplus}(\text{MLP}(\hat{w}_t)))$$

$$w_t = \text{LayerNorm}(\hat{w}_t + \text{MLP}(z_t) + \text{Position}(t))$$

마지막으로 $$w_{1:T}$$을 다층 퍼셉트론으로 매핑하여 $$x_{1:T}$$ 생성.[1]

**추론 모델** (Inference Model):

생성 모델과 매개변수를 공유하며, 추가적으로 모든 관측값을 고려한 자기 어텐션 적용:[1]

$$k_t = \text{Attention}(h_{1:T}, h_{1:T}, h_{1:T})$$

$$z_t \sim \mathcal{N}(\text{MLP}([\hat{w}_t, k_t]), \text{Softplus}(\text{MLP}([\hat{w}_t, k_t])))$$

이는 기존 RNN 기반 필터링(filtering)과 달리 **평활화(smoothing)** 과정을 모방합니다.[1]

### 3.2 다층 확장 (Multi-Layered Extension)

계층적 구조로 L개의 잠재변수 층을 도입:[1]

$$p_\theta(x_{1:T}, z_{1:T}^{(1:L)} | x_{1:C}) = \prod_{t=1}^{T} p_\theta(x_t | z_t^{(L)}) \prod_{\ell=1}^{L} \prod_{t=1}^{T} p_\theta(z_t^{(\ell)} | z_{1:t-1}^{(\ell)}, z_{1:T}^{(\ell-1)}, x_{1:C})$$

각 층 $$\ell$$에서의 어텐션 연산:[1]

$$\tilde{w}_t^{(\ell)} = \text{LayerNorm}(w_{t-1}^{(\ell)} + \text{Attention}(w_{t-1}^{(\ell)}, w_{1:T}^{(\ell-1)}, w_{1:T}^{(\ell-1)}))$$

$$\bar{w}_t^{(\ell)} = \text{LayerNorm}(\tilde{w}_t^{(\ell)} + \text{Attention}(\tilde{w}_t^{(\ell)}, w_{1:t-1}^{(\ell)}, w_{1:t-1}^{(\ell)}))$$

$$\hat{w}_t^{(\ell)} = \text{LayerNorm}(\bar{w}_t^{(\ell)} + \text{Attention}(\bar{w}_t^{(\ell)}, h_{1:C}, h_{1:C}))$$

**계산 복잡도**: 단층은 $$O(T^2d)$$, L층은 $$O(LT^2d)$$ (시간 및 메모리).[1]

## 4. 성능 향상 분석

### 4.1 시계열 예측 성능

**연속 순위 확률 점수(Continuous Ranked Probability Score, CRPS)** 지표 사용:[1]

ProTran이 5개 데이터셋 모두에서 최고 성능 달성:

| 데이터셋 | ProTran | TimeGrad | Transformer-MAF | NKF |
|---------|---------|----------|-----------------|-----|
| SOLAR | 0.194 | 0.287 | 0.301 | 0.320 |
| ELECTRICITY | 0.016 | 0.021 | 0.021 | 0.016 |
| TRAFFIC | 0.028 | 0.044 | 0.056 | 0.100 |
| TAXI | 0.084 | 0.114 | 0.179 | - |
| WIKIPEDIA | 0.047 | 0.049 | 0.063 | 0.071 |

특히 SOLAR, TRAFFIC, TAXI 데이터셋에서 **상당한 성능 향상** 달성.[1]

### 4.2 인간 동작 예측 성능

ADE(Average Displacement Error)와 FDE(Final Displacement Error) 지표:[1]

| 데이터셋 | ProTran | DLow | DSP | GMVAE |
|---------|---------|------|-----|-------|
| Human3.6M ADE | 0.381 | 0.425 | 0.493 | 0.461 |
| Human3.6M FDE | 0.491 | 0.518 | 0.592 | 0.555 |
| HumanEva-I ADE | 0.258 | 0.251 | 0.273 | 0.305 |
| HumanEva-I FDE | 0.255 | 0.268 | 0.290 | 0.345 |

특히 Human3.6M에서 **최고 성능 모델 DLow를 명확히 능가**.[1]

### 4.3 제거 연구(Ablation Study)

TRAFFIC 데이터셋에서 주요 컴포넌트의 영향 분석:[1]

- **2개 층 제거**: CRPS 0.028 → 0.031 (성능 저하)
- **문맥 어텐션 제거**: CRPS 0.031 → 0.033
- **결정론적 성분 제거** (순수 확률적 전환): CRPS 0.041 (가장 큰 영향)

**결정론적 성분 $$w_t$$가 장거리 의존성 유지에 결정적 역할**.[1]

## 5. 일반화 성능 분석

### 5.1 일반화 향상 메커니즘

ProTran이 우수한 일반화 성능을 달성하는 이유:

**1) 비마르코프 잠재 동역학**:[1]
- 각 $$z_t$$이 모든 선행 잠재변수 $$z_{1:t-1}$$에 의존하므로, 복잡한 장거리 의존성을 직접 모델링
- RNN 기반 모델의 정보 병목 현상(bottleneck) 제거

**2) 어텐션의 장거리 포착 능력**:[1]
- 어텐션은 $$O(1)$$ 거리에서 장거리 정보 접근 가능 (RNN은 $$O(T)$$)
- 그래디언트 흐름 개선으로 최적화 용이

**3) 추론-생성 분리**:[1]
- 추론 모델: 모든 관측 $$x_{1:T}$$ 활용 (평활화)
- 생성 모델: 조건 $$x_{1:C}$$만 사용 (필터링과 유사)
- 이는 기존 필터링 접근법보다 향상된 잠재 표현 학습 가능

**4) 계층적 구조의 표현 유연성**:[1]
- 다층 잠재변수는 서로 다른 시간 척도의 의존성을 모델링
- 상위층: 느린 변화, 하위층: 빠른 변화

### 5.2 데이터 효율성

고차원 데이터셋에서도 안정적 성능:[1]
- **WIKIPEDIA**: 고차원(1,140개 시계열) → 지표값 0.047 달성
- 잠재공간 모델링으로 관측공간 차원의 저주 회피

### 5.3 불확실성 정량화

분포 예측을 통한 예측 구간 제공:[1]
- 큰 크기나 미래 관측값: 더 높은 분산 추정
- 정보 이론적으로 건전한 확률 모델

## 6. 모델의 한계

논문에서 명시한 한계점:[1]

**1) 이차 계산 복잡도**:
- 시간 복잡도: $$O(LT^2d)$$
- 메모리: $$O(T^2d)$$
- 긴 수열(T가 매우 큼)에 부적합
- 예: 음악 생성, 언어 모델링 등 장기 시퀀스 작업에 제한

**2) 재귀적 잠재 동역학**:
- 잠재 업데이트가 순차적이므로 완전 병렬화 불가
- 트랜스포머의 병렬 처리 장점 일부 상실

**3) 추가 가능한 개선**:
- **희소 어텐션(Sparse Attention)** 활용으로 계산 복잡도 개선 가능
- **정규화 흐름(Normalizing Flows)** 또는 **코플러(Copula)** 적용으로 확률 모델 유연성 증진

## 7. 연구 영향 및 향후 고려사항

### 7.1 학계에 미치는 영향

**확률적 시계열 모델링의 패러다임 전환**:[1]
- 트랜스포머 기반 SSM의 실현 가능성 입증
- RNN 완전 제거의 타당성 증명
- 계층적 확률 모델의 시계열 적용 확대

**일반화 성능의 중요성 강조**:
- 단순 포인트 예측에서 분포 예측 중심으로 전환
- 불확실성 정량화의 중요성 재인식

### 7.2 향후 연구 고려사항

**1) 계산 효율성 개선**:[1]
- 희소 트랜스포머 도입 검토
- 선형 어텐션 메커니즘 탐색
- 회소 구조화(sparse factorization) 적용

**2) 더 복잡한 도메인 적용**:
- 초장기(very long-term) 예측 작업
- 다중 모드 시계열(multimodal time series)

**3) 의료 응용 확대**:
- 질병 진행 모니터링
- 임상 사건 예측
- 개인화된 건강 예측

**4) 모델 표현 능력 향상**:
- 구조적 출력(structured outputs) 통합
- 정규화 흐름 또는 코플러 기반 확률 모델 확장
- 적응형 아키텍처 설계

**5) 도메인 특화 변수 통합**:
- 외부 공변량(exogenous covariates)의 효과적 활용
- 도메인 지식 기반 제약 조건 추가

### 7.3 기술적 진화 방향

**다중 작업 학습(Multi-task Learning)**:
- 다양한 시계열 데이터셋 간 전이 학습
- 메타 학습(Meta-learning) 적용

**자기 지도 학습(Self-Supervised Learning)**:
- 레이블이 없는 대규모 시계열 데이터 활용
- 마스킹 기반 프리트레이닝 전략

**신경 미분 방정식(Neural ODEs) 결합**:
- 연속 시간 동역학 모델링
- 비정규 시계열 처리

이 논문은 **확률적 시계열 모델링 분야에 중대한 기여**를 하였으며, 트랜스포머 기반 접근법의 **실용성과 효과성**을 충분히 입증했습니다. 특히 일반화 성능 향상, 장거리 의존성 포착, 불확실성 정량화 측면에서 기존 방법들의 한계를 극복했다는 점이 주목할 가치가 있습니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/7aa03d46-b68c-4826-8603-5587801358c1/NeurIPS-2021-probabilistic-transformer-for-time-series-analysis-Paper.pdf)
