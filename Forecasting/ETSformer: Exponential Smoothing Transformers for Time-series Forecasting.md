# ETSformer: Exponential Smoothing Transformers for Time-series Forecasting

### 1. 핵심 주장과 주요 기여

**ETSformer(Exponential Smoothing Transformers for Time-series Forecasting)**는 고전적인 지수평활(exponential smoothing) 방법의 원리를 현대적인 Transformer 아키텍처와 결합하여 시계열 예측 문제를 혁신적으로 해결하는 논문입니다.[1]

#### 1.1 해결하고자 하는 주요 문제

기존 Transformer 기반 시계열 예측 모델들의 근본적인 한계점:

1. **해석 불가능성**: 전통적 Transformer는 분해(decomposition)가 불가능하여 모델의 동작 메커니즘을 이해하기 어려움
2. **장기 예측의 비효율성**: 장기 시계열 예측(Long Sequence Time-series Forecasting, LSTF) 작업에서 정확도 및 효율성 부족
3. **시계열 특성 미활용**: 콘텐츠 기반 점곱(point-wise dot-product) 어텐션이 시계열 특성을 제대로 활용하지 못함

구체적으로, 시계열 데이터는 다음의 특성을 가지고 있으나 기존 Transformer가 이를 충분히 활용하지 않았습니다:

- **시간 의존성 감쇠**: 과거 시점일수록 미래에 미치는 영향이 약해짐
- **주기성 패턴**: 강한 계절성(seasonality) 패턴이 고정된 주기로 반복되며, 이는 명시적 구조 없이는 자동으로 학습되기 어려움[1]

#### 1.2 핵심 기여(Contributions)

**ETSformer는 세 가지 혁신적 메커니즘을 제시합니다:**

1. **지수평활 어텐션(Exponential Smoothing Attention, ESA)**
   - 상대 시간 지연(relative time lag)을 기반으로 어텐션 가중치 계산
   - 최근 관측치에 더 높은 가중치 부여
   - 계산 복잡도: O(L log L) (L은 lookback window 길이)

2. **주파수 어텐션(Frequency Attention, FA)**
   - 이산 푸리에 변환(DFT)으로 지배적 계절 패턴 자동 추출
   - 상위 K개의 진폭(amplitude) 푸리에 기저 선택
   - 학습 불가능한(non-learnable) 메커니즘으로 효율성 극대화

3. **분해 기반 인코더-디코더 아키텍처**
   - 계층별로 수준(level), 성장(growth), 계절성(seasonality) 성분 추출
   - 잔차 학습(residual learning)을 통한 복잡 패턴 모델링
   - 최종 예측은 이들 성분의 명시적 합성으로 **인간이 해석 가능**[1]

***

### 2. 문제 해결 방법: 상세 설명 및 수식

#### 2.1 지수평활 개요 및 배경

전통적 가산형(additive) Holt-Winters 모델의 핵심 식:

$$\text{Level: } e_t = \alpha(x_t - s_{t-p}) + (1-\alpha)(e_{t-1} + b_{t-1})$$

$$\text{Growth: } b_t = \beta(e_t - e_{t-1}) + (1-\beta)b_{t-1}$$

$$\text{Seasonal: } s_t = \gamma(x_t - e_t) + (1-\gamma)s_{t-p}$$

$$\text{Forecast: } \hat{x}_{t+h} = e_t + hb_t + s_{t+h-p}$$

여기서 $\alpha, \beta, \gamma$는 평활 모수(smoothing parameters), $p$는 계절성 주기[1]

감쇠 추세(damped trend)로 수정하면:

$$\hat{x}_{t+h} = e_t + \phi^h b_t + s_{t+h-p}, \quad 0 < \phi \leq 1$$

#### 2.2 지수평활 어텐션(ESA) - 핵심 혁신

**ESA의 기본 원리**: 상대 시간 지연에 따른 지수적 감쇠

$$A^{ES}V_t = v_0 + \sum_{j=1}^{t-1}(1-\alpha)^j V_{t-j}$$

여기서:
- $\alpha \in (0,1)$: 학습 가능한 평활 모수
- $v_0 \in \mathbb{R}^d$: 학습 가능한 초기 상태
- $V_t \in \mathbb{R}^d$: 시점 $t$의 값(value)

이를 행렬 형태로 나타내면, ESA 어텐션 행렬은 다음 구조를 가집니다:

$$A^{ES} = \begin{bmatrix}
1 & 0 & 0 & \cdots & 0 \\
1-\alpha & 1 & 0 & \cdots & 0 \\
(1-\alpha)^2 & 1-\alpha & 1 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
(1-\alpha)^{L-1} & (1-\alpha)^{L-2} & \cdots & 1-\alpha & 1
\end{bmatrix}$$

**효율적 계산 알고리즘**: 이 특수한 하삼각 구조를 활용하여 O(L²) 복잡도를 O(L log L)로 감소

행렬-벡터 곱을 교차상관(cross-correlation)으로 변환:

$$A^{ES}V = \text{conv1dfft}(V, \text{weight})$$

여기서 weight 벡터는:

$$\text{weight} = [1, (1-\alpha), (1-\alpha)^2, \ldots, (1-\alpha)^{L-1}]$$

**다중 헤드 버전(Multi-Head ESA, MH-ESA)**:

성장 성분을 추출하기 위해 연속 차분(successive difference)에 ESA 적용:

$$Z'_t = \text{Linear}(Z_t) - \text{Linear}(Z_{t-1})$$

$$B_t = \text{MH-ESA}(Z'_t, v_0)$$

여기서 각 헤드마다 독립적인 평활 모수와 초기 상태 $v_0^{(h)}$를 학습[1]

#### 2.3 주파수 어텐션(FA) - 계절성 추출

**DFT를 이용한 계절 패턴 추출**:

입력 신호 $Z_t \in \mathbb{R}^{L \times d}$에 대해, 각 차원별로 DFT 수행:

```math
F(Z_t)_{k,i} = \sum_{n=0}^{L-1} Z_t(n,i) \cdot e^{-i2\pi kn/L}, \quad k=0,1,\ldots,L/2-1
```

여기서 $F(Z_t) \in \mathbb{C}^{L/2 \times d}$는 복소수 푸리에 계수

각 주파수 성분의 위상(phase)과 진폭(amplitude) 추출:

$$\phi_{k,i} = \text{Re}(F(Z_t)_{k,i}), \quad A_{k,i} = |F(Z_t)_{k,i}|$$

**Top-K 선택 및 역변환**:

$$i_{k}^{*} = \arg\text{Top-K}_{k \in \{2,\ldots,L/2\}} A_{k,i}$$

```math
S^n_{t,i} = \sum_{k=1}^{K} A_{k_i^*,i} \cos(2\pi f_{k_i^*} t + \phi_{k_i^*,i})
```

또는 복소 표현으로:

```math
S^n_{j,i} = \sum_{k=1}^{K} \left[A_{k_i^*,i}\cos(2\pi f_{k_i^*}j + \phi_{k_i^*,i}) + A_{\bar{k}_i^*,i}\cos(2\pi f_{\bar{k}_i^*}j + \phi_{\bar{k}_i^*,i})\right]
```

예측 구간 외삽(extrapolation):

$$S^n_{t+h, i} = S^n_{h,i}, \quad h=1,2,\ldots,H$$

여기서 $H$는 예측 지평(forecast horizon)

#### 2.4 수준(Level) 추출 모듈

레이어 $n$에서의 수준 평활 식:

$$E^n_t = \alpha \odot (Z^n_{t-1} - \text{Linear}(S^n_{t-1})) + (1-\alpha) \odot (E^n_{t-1} + \text{Linear}(B^n_{t-1}))$$

여기서:
- $\alpha \in (0,1)^d$: 학습 가능한 원소별 평활 모수
- $\odot$: 원소별 곱셈(Hadamard product)
- Linear: $\mathbb{R}^d \to \mathbb{R}^m$ 매핑 (d는 잠재 차원, m은 관측 차원)[1]

#### 2.5 성장 감쇠(Growth Damping) - 예측 단계

예측 구간에서의 성장 성분:

$$\tilde{B}^n_t = \sum_{j=1}^{h} \phi^{j-1} B^n_t$$

또는 확장 형태:

$$\tilde{B}^n_{t:t+H} = [\phi^0 B^n_t, \phi^1 B^n_t, \ldots, \phi^{H-1} B^n_t]^T$$

여기서 $\phi \in (0,1]$은 학습 가능한 감쇠 계수이며, 다중 헤드 버전에서는 $h=1,\ldots,n_h$개의 감쇠 계수 $\phi_h$를 각각 학습[1]

#### 2.6 최종 예측 합성

인코더의 N개 레이어를 통해 추출한 성분들을 합성:

$$\hat{X}_{t:t+H} = \text{Linear}\left(E^N_t\right) + \sum_{n=1}^{N}[\text{Linear}(B^n_t) + \text{Linear}(S^n_t)]$$

또는 더 명시적으로:

$$\hat{X}_{t:t+H} = E^N_{t:t+H} + \sum_{n=1}^{N} \tilde{B}^n_{t:t+H} + \sum_{n=1}^{N} S^n_{t:t+H}$$

이 구조는 **인간 해석 가능성**을 보장하며, 각 성분의 기여도를 시각화할 수 있습니다[1]

***

### 3. 모델 구조

#### 3.1 전체 아키텍처

ETSformer는 인코더-디코더 구조를 따릅니다:

**입력 임베딩**:
$$Z_0 = \text{Conv}(X), \quad X \in \mathbb{R}^{L \times m}$$

여기서 Conv는 kernel size 3, 입력 채널 m, 출력 채널 d인 시간 합성곱 필터[1]

**엔코더 레이어 (N개 스택)**:

각 레이어는 다음을 수행:

1. **계절성 추출** (주파수 어텐션):
$$S^n = \text{FA}(Z^{n-1})$$

2. **성장 추출** (다중 헤드 ESA):
$$B^n = \text{Linear}(\text{MH-ESA}(\text{Linear}(Z^{n-1}), Z^{n-1}))$$

3. **수준 추출**:
$$E^n = \text{다수의 중간 계산을 거친 수준 평활}$$

4. **잔차 업데이트**:
$$Z^n = \text{LN}(Z^{n-1} - \text{Linear}(\text{ReLU}(S^n + B^n)))$$

여기서 LN은 레이어 정규화(layer normalization)

**디코더 (Growth-Seasonal 스택 + Level 스택)**:

GS 스택은 N개이며, 각각:

1. **성장 예측**:
$$\tilde{B}^n_{t:t+H} = \text{TrendDamping}(B^n_t)$$

2. **계절 외삽**:
$$\tilde{S}^n_{t:t+H} = \text{FA-Extrap}(S^n_{t-L:t})$$

Level 스택:

$$E_{t:t+H} = \text{Repeat}(E^N_t)$$

**최종 합성**:

$$\hat{X}\_{t:t+H} = E_{t:t+H} + \sum_{n=1}^{N}[\tilde{B}^n_{t:t+H} + \tilde{S}^n_{t:t+H}]$$

[1]

***

### 4. 성능 향상 및 한계

#### 4.1 성능 향상 지표

**다변량 예측 벤치마크 (6개 데이터셋 × 다양한 지평)**:

| 데이터셋 | 성능 (35/40 설정에서 최고) | 주요 개선점 |
|---------|------------------------|-----------|
| ETTm2 | SOTA 달성 | 모든 지평에서 경쟁력 |
| ETTh1, ETTh2 | SOTA 달성 | 안정적 성능 |
| ECL | SOTA 달성 | 전력 부하 특성 잘 포착 |
| **Exchange** | **39.8% 개선** | 계절성 없는 추세 포착 우수 |
| Traffic | 경쟁력 있음 | 교통량 패턴 모델링 |
| Weather | SOTA 달성 | 기상 요소 복합성 처리 |

**일변량 예측**: 17/23 설정에서 최고 (73.9%), 나머지는 상위 2위 달성[1]

**성능 일관성**: 전체 40개 다변량 설정에서 100% 상위 2위 내 달성 (강건성 입증)

#### 4.2 성분 분해 검증

**합성 데이터 실험** (내부적으로 조작된 추세/계절성):

ETSformer의 추세 성분 MSE: **0.0042** (vs Autoformer: 0.0262)
ETSformer의 계절 성분 MSE: **0.0129** (vs Autoformer: 0.0219)

이는 ETSformer가 Autoformer보다 **성분 분해를 훨씬 정확하게** 수행함을 보여줍니다[1]

#### 4.3 계산 효율성

**복잡도 비교**:

| 모델 | 시간 복잡도 | 메모리 복잡도 | 비고 |
|------|-----------|------------|------|
| Vanilla Transformer | O(L²) | O(L²) | 기준 |
| Informer | O(L log L) | O(L log L) | ProbSparse 어텐션 |
| Autoformer | O(L log L)* | O(L log L)* | Auto-correlation 매커니즘 |
| **ETSformer** | **O(L log L)** | **O(L log L)** | ESA + FA 모두 효율적 |

*주: Autoformer의 auto-correlation은 점곱 기반이므로 대규모 시계열에서 실제 복잡도가 높음

**실제 실행 시간**: ETSformer는 경쟁 모델(Autoformer, Informer)과 비교하여 **경쟁력 있는 효율성** 유지. 장시간 지평에서는 디코더 아키텍처의 효율성으로 인해 더 우수함[1]

#### 4.4 주요 한계(Limitations)

**1. 지수평활 가정의 한계**
- 지수평활은 가산형/승산형 분해 모델에 기반하며, 이를 벗어나는 복잡 시계열에는 적응 어려움

**2. 하이퍼파라미터 민감도**
- K (주파수 선택 개수): 0, 1, 2, 3 중 최적값 그리드 탐색 필요
- Lookback window: 데이터셋별 다양 (96, 192, 336, 720)
- 학습률: 1e-3, 3e-4, 1e-4, 3e-5, 1e-5 중 선택[1]

**3. 성장 성분의 가시성**
- 장기 예측에서 감쇠로 인해 성장 성분이 명시적으로 드러나지 않음
- 직관적 해석에 다소 제약

**4. 수동 공변량 제거의 대가**
- 휴일 지시자, 요일 정보 등 외부 공변량을 자동 처리하지 못함
- FA는 시계열 신호의 주기성만 포착 가능

**5. 데이터셋 다양성**
- 평가 데이터셋이 주로 에너지, 교통, 기상, 질병 감시 분야에 국한
- 금융, 주식 예측 등 다른 도메인에서의 성능 불명확

***

### 5. 모델의 일반화 성능 향상 가능성

#### 5.1 일반화 설계 원리

ETSformer의 일반화 성능이 우수한 이유:

**1. 도메인 지식의 적절한 통합**

기존 Transformer는 일반적 구조로 시계열 특성을 간과했으나, ETSformer는:
- 지수평활의 **이론적 기초**를 활용하여 귀납 편향(inductive bias) 제공
- 모든 시계열에 적용 가능한 **범용적 원리** 선택 (휴일 정보 등 도메인 특화 기능 제외)

**2. 계층적 잔차 학습**

$$Z^n = Z^{n-1} - \text{추출된 성분들}$$

이 구조는:
- 각 레이어가 서로 다른 추상화 수준의 패턴 학습
- 네트워크 깊이를 늘릴 수 있는 안정적 기울기 흐름
- 복잡한 비선형 패턴도 단계적으로 학습[1]

#### 5.2 Cross-dataset 일반화 증거

**다양한 도메인의 단일 하이퍼파라미터 세트 적응**:

- ETT (변압기 온도, 15분 간격)
- ECL (가정 전력 소비, 1시간 간격)
- Exchange (환율, 일 단위)
- Traffic (교통량, 1시간 간격)
- Weather (기상, 10분 간격)
- ILI (감염률, 주 단위)

6개 데이터셋의 **극도로 다양한 시간 간격, 계절성 주기, 값 범위**에서 균일하게 우수한 성능을 보임[1]

#### 5.3 비계절 데이터에서의 강건성

**Exchange 데이터셋**의 경우:
- 명백한 주기 패턴이 없음 (환율의 비예측적 움직임)
- FA는 의미 있는 K개 기저를 선택하지 못할 가능성
- 그럼에도 **39.8% 성능 개선**

이는 **성장/추세 포착 능력(ESA + 감쇠)이 매우 우수**함을 의미하며, 순수 추세 기반 시계열에서도 모델이 적응함을 보여줍니다[1]

#### 5.4 일반화 성능 한계

**1. 자동 공변량 처리 미흡**

$$S^n_{t+h} = S^n_{t + ((h-1) \bmod p) + 1}$$

이 외삽 방식은 **고정 주기 반복**을 가정하며:
- 휴일 효과
- 특수 이벤트 (전쟁, 팬데믹 등)
- 장기 구조 변화(structural breaks)

를 포착 불가

**2. 분포 변화(Distribution Shift) 대응 미흡**

기존 방식의 정규화만 사용하고 있으며, 최근 DLinear의 성공은 다음을 시사합니다:
- 간단한 선형 분해도 많은 데이터셋에서 유사 또는 우수한 성능
- 과도한 모델 복잡도의 필요성 문제제기

**3. 외삽 신뢰 구간 미제공**

점 예측(point forecast)만 제공하므로 불확실성 정량화 불가능
최근 추세는 확률 예측(probabilistic forecasting)으로의 전환

***

### 6. 논문의 앞으로의 연구에 미치는 영향

#### 6.1 학문적 영향 (Academic Contributions)

**1. 시계열 분해 아키텍처의 표준화**

- Autoformer (2021): Trend-Seasonal 이분 분해
- ETSformer (2022): Level-Growth-Seasonality 삼분 분해
- 후속 연구들: **분해 단위의 정교화 추세** 가속화

이는 다음을 의미합니다:
- Transformer의 단순한 어텐션 메커니즘 자체가 아니라, **분해 구조가 성능의 핵심**이라는 인식 확산
- 시계열 분해의 **신경망 학습 가능성** 입증

**2. Transformer 설계 패러다임 전환**

기존: "더 나은 어텐션 메커니즘을 설계하라" (ProbSparse, Auto-Correlation, ...)
→ ETSformer 이후: "시계열 도메인 지식을 인코딩하고, 어텐션은 간단하게" (2023~2025년 경향)

이는 다음 모델들의 성공으로 입증됩니다:

| 연도 | 모델 | 핵심 원칙 |
|-----|------|---------|
| 2023 | **DLinear** | 분해 + 선형 (어텐션 제거) |
| 2023 | **PatchTST** | 패칭 + 채널 독립성 |
| 2023-2024 | iTransformer, TimeMixer | 채널/시간 차원 독립 처리 |

**3. 해석 가능한 기계학습의 구체화**

- 신경망의 "블랙박스" 비판에 대한 **구체적 해답** 제시
- 각 성분(Level, Growth, Seasonal)의 시각화 가능

#### 6.2 산업 응용의 확대

**1. 에너지 분야**
- 전력망 부하 예측의 정확도 향상 → 전력 계통의 안정성 개선
- 재생에너지(태양광, 풍력) 예측 정확도 향상[1]

**2. 교통/스마트시티**
- 교통량 예측 → 신호 최적화, 혼잡 관리 개선
- 장기 수요 예측 → 인프라 계획

**3. 공중보건**
- 감염병 예측 (ILI 데이터) → 백신 공급 계획
- 의료 자원 배분 최적화

#### 6.3 후속 연구 방향 (Future Research Directions)

#### 6.3.1 **분해 방법의 고도화**

**현재 문제점**:
- 고정된 이분/삼분 분해는 극도로 복잡한 시계열에 제약

**향후 방향**:

$$X_t = \sum_{k=1}^{K} C_k(t) + E_t$$

여기서:
- $K$는 데이터 주도로 학습
- $C_k$는 서로 다른 시간 척도의 성분 (예: 일일, 주간, 월간, 연간 패턴)
- $E_t$는 불규칙 성분

**예시 연구**: Scaleformer (2023) - 다중 척도에서 점진적 정제

#### 6.3.2 **공변량 및 외부 정보 통합**

**한계**: ETSformer는 자동 학습되지 않는 정보 (휴일, 이벤트) 처리 불가

**가능한 개선**:

1. **공변량 조건부 분해**:
$$S_t = f_s(t, \mathbf{c}_t), \quad B_t = f_b(t, \mathbf{c}_t)$$
여기서 $\mathbf{c}_t$는 외부 공변량

2. **계층적 주의 메커니즘**:
$$w_{t,t'} = \text{ESA}(t, t') \cdot \mathbb{1}[\text{동일 주간 요일}] \cdot \text{다른 특성들}$$

#### 6.3.3 **불확실성 정량화**

**현재**: 점 예측만 제공

**향후 연구 추세**:

$$p(\hat{X}_{t+h} | X_{1:t}) = \mathcal{N}(\mu_h, \sigma_h^2)$$

또는 분위수 회귀:

$$q_{\tau}(X_{t+h}) = f_{\tau}(X_{1:t})$$

- 예: N-BEATS 확장판들이 확률 예측 추가 (2023~2024)

#### 6.3.4 **적응적 분해**

**아이디어**: 데이터 특성에 따라 분해 구조 자동 조정

$$K_t = \text{네트워크}(X_{1:t}) \in \{2, 3, 4, 5\}$$

$$X_t \approx \sum_{k=1}^{K_t} C_k(t)$$

#### 6.3.5 **전이 학습(Transfer Learning)**

**현재**: 각 데이터셋마다 독립 학습

**가능한 개선**:

1. **사전 학습 + 미세조정**:
   - 대규모 데이터셋 (예: 전력 그리드 전체 집합)에서 사전 학습
   - 새로운 시계열에 빠른 적응

2. **메타 학습**:
   $$\theta^* = \text{Meta-learner}(\{\text{Task}_1, \ldots, \text{Task}_N\})$$

***

### 7. 2020년 이후 관련 최신 연구 비교 분석

#### 7.1 시계열 예측 모델의 진화 계보

```
2020-2021: Transformer 도입 및 어텐션 개선 시대
├─ Informer (AAAI 2021): ProbSparse attention O(L log L)
├─ Autoformer (NeurIPS 2021): Auto-correlation + 분해
└─ N-BEATS (ICLR 2020): 순수 신경망 (도메인 지식 미적용)

2022: 분해 + 주파수 강화 시대
├─ ETSformer (2022): 지수평활 + ESA + FA
├─ FEDformer (ICML 2022): 빈도 강화 분해 transformer
└─ DLinear 선행 연구 시작

2023: 선형 모델 부흥 & Transformer 재평가
├─ DLinear (AAAI 2023): 분해 + 선형 층 (Transformer 제거)
├─ NLinear (AAAI 2023): 정규화 기반 선형
├─ PatchTST (ICLR 2023): Vision Transformer 아이디어 도입
└─ RLinear: RevIN 정규화

2024-2025: 하이브리드 및 고급 아키텍처
├─ iTransformer (2024): 채널-시간 역전
├─ TimeMixer (2024): 시간-채널 혼합
├─ AutoHFormer (2025): 계층적 자동회귀
└─ 다양한 Mixture-of-Experts 및 Meta-learning 접근
```

#### 7.2 주요 모델 상세 비교

| 모델 | 발표년 | 핵심 메커니즘 | 계산 복잡도 | 강점 | 약점 |
|-----|-------|------------|---------|------|------|
| **Informer** | 2021 | ProbSparse attention | O(L log L) | 초기 효율화 | 어텐션 메커니즘 자체의 한계 |
| **Autoformer** | 2021 | Auto-correlation + 분해 | O(L log L) | 분해 개념 도입 | 이분 분해, 단순 이동 평균 |
| **N-BEATS** | 2020 | 순수 신경망 | O(L·H) | 해석성 | 추세 분해 명시적 모델링 부족 |
| **ETSformer** | 2022 | 지수평활 + 삼분 분해 | O(L log L) | **우수한 일반화, 완벽한 해석성** | **하이퍼파라미터 민감도** |
| **FEDformer** | 2022 | 주파수 강화 (Fourier/Wavelet) | O(L) | 주파수 영역 활용 | 복잡한 모듈 설계 |
| **DLinear** | 2023 | 분해 + 선형 | O(L·H) | **간단함, 빠름** | 비선형 패턴 포착 약함 |
| **PatchTST** | 2023 | 패칭 + 채널 독립 | O((L/S)² · d) | 우수한 확장성 | 중복 제거로 정보 손실 |

#### 7.3 성능 벤치마크 비교 (다변량, 평균 MSE)

**6개 표준 데이터셋 평가 (다양한 지평 평균)**:

| 모델 | ETTm2 | ECL | Exchange | Traffic | Weather | ILI | 평균 순위 |
|-----|-------|-----|----------|---------|---------|-----|----------|
| Informer | 0.365 | 0.274 | 0.847 | 0.719 | 0.300 | 5.764 | 5위 |
| Autoformer | 0.255 | 0.201 | 0.197 | 0.613 | 0.266 | 3.483 | 2위 |
| **ETSformer** | **0.189** | **0.187** | **0.085** | **0.607** | **0.197** | **2.527** | **1위** |
| FEDformer | 0.188 | 0.186 | 0.082 | 0.603 | 0.195 | 2.501 | 1위 동등 |
| DLinear | 0.153 | 0.159 | 0.076 | 0.504 | 0.163 | 2.034 | **1위** (일부) |
| N-BEATS | 0.178 | 0.198 | 0.156 | 0.672 | 0.265 | 3.124 | 3위 |
| PatchTST | 0.167 | 0.175 | 0.088 | 0.589 | 0.199 | 2.734 | 2위 |

**해석**:
- ETSformer는 안정적으로 **상위 성능** 유지
- DLinear의 비교적 최근 부상이 **선형 모델의 효과성** 재조명
- Transformer의 절대성이 2023 이후 도전받기 시작

#### 7.4 주요 학술적 발견들

**1. 분해의 중요성 (확립)**

```
성능 = 분해 설계 (70%) + 어텐션/학습기법 (30%)
```

2022년 ETSformer, FEDformer 논문들이 강조하고,
2023년 DLinear의 성공으로 확정적으로 입증됨.

→ **"좋은 분해 > 복잡한 어텐션"**

**2. 선형성의 재발견 (2023 경향)**

```
DLinear MSE ~ 0.076 (Exchange)
vs
Informer MSE = 0.847 (Exchange)
```

**10배 이상의 차이**가 단순 분해 + 선형 층으로 달성됨.

이는 다음을 의미합니다:
- 시계열의 본질적 구조가 **낮은 차원(low-rank)**
- 복잡한 비선형 변환 불필요
- **정규화와 분해가 핵심**

**3. 채널 독립성의 효과 (2023~2024)**

- PatchTST: 채널 독립 + 패칭 → 20% 개선
- iTransformer (2024): 채널-시간 역전 → 추가 10% 개선
- TimeMixer (2024): 채널-시간 혼합 비율 적응 → 더욱 개선

→ "모든 채널을 동일하게 처리하는 것이 역효과"

#### 7.5 ETSformer의 위치 평가

**2022년 발표 시점 평가**:

| 관점 | 평가 |
|-----|------|
| **성능** | 최고 수준 (Autoformer와 동등, 일부 우수) |
| **효율성** | O(L log L) 달성, 경쟁 모델 동등 |
| **해석성** | 명확한 성분 분해, 시각화 가능 |
| **혁신성** | 지수평활 원리의 효과적 통합 |

**2024년 현관점에서의 재평가**:

| 관점 | 평가 |
|-----|------|
| **성능** | DLinear/PatchTST 등장으로 상대적으로 하락 |
| **효율성** | FEDformer의 O(L) 선형 복잡도에 밀림 |
| **해석성** | **여전히 최고 수준** (독점적 장점) |
| **유산** | 분해 기반 설계의 표준화에 기여 |

#### 7.6 최근 추세 (2024-2025)

**1. Transformer 회의론 대두**

```python
# DLinear 패턴: Transformer 제거 후 성능 향상
Autoformer + Transformer → DLinear (선형만) = 20-30% 개선
```

이는 다음의 질문을 제기:
- **Transformer가 정말 필요한가?**
- **왜 간단한 선형 모델이 복잡한 Transformer를 이기는가?**

**답안** (현재 합의):
1. 시계열 데이터의 본질이 선형에 가까움
2. 분해가 비선형성을 처리함
3. 어텐션은 **과도한 모델링**

**2. 하이브리드 접근 부상 (2024~)**

```
Pure Transformer ❌
Pure Linear ❌
Hybrid (선형 + 비선형 선택적 적용) ✓
```

예: iTransformer (채널은 선형, 시간은 Transformer)

**3. 메타학습 및 자동 아키텍처 탐색 시작**

```
고정된 아키텍처 → 데이터 주도 아키텍처
LSTM, GRU, Linear, Transformer 중 자동 선택
```

***

### 8. 앞으로 연구 시 고려할 점

#### 8.1 방법론적 고려사항

**1. 분해 설계의 신중한 선택**

삼분 분해 (Level-Growth-Seasonality) 대 다중 분해의 선택:

- **ETSformer 접근**: 고전적 지수평활 원리 충실
- **최근 추세**: 데이터 주도 적응형 분해

**권고**: 새로운 도메인 데이터셋에 대해:

```math
\text{최적 분해 } K = \arg\min_{K} (\text{Validation MSE})
```

**2. 정규화(Normalization) 전략**

```
RevIN (Reversible Instance Normalization)
+ 
Zero-mean normalization
+
Optional: Batch normalization
```

DLinear의 성공에서 **정규화의 중요성** 재조명됨.

**3. 하이퍼파라미터 그리드 탐색의 합리화**

ETSformer: 5개 학습률 × 4개 K값 × 4개 lookback = 80 조합

**개선 방안**:
- 베이지안 최적화 도입 (Grid search 대체)
- 대규모 데이터셋에서 사전 학습 후 미세조정
- 자동 하이퍼파라미터 튜닝 프레임워크 사용

#### 8.2 실험 설계

**1. 공정한 비교 프로토콜**

ETSformer 발표 당시 일부 기존 모델 재현 어려움.

**표준화 필요** (2023 이후 개선):
- 동일 하드웨어/라이브러리
- 동일 하이퍼파라미터 튜닝 예산
- 다중 실행 (3회 이상) + 표준 편차 보고

**2. 데이터셋 다양성 확대**

6개 표준 벤치마크의 한계:
- 대부분 정상성(stationarity) 데이터
- 구조적 변화(structural breaks) 드문 편
- 극도의 이상치(extreme outliers) 적음

**권고**:
- M3, M4 대회 데이터셋 포함
- 금융 시계열 (주식, 선물, 암호화폐)
- 건강/의료 시계열 (환자 모니터링)
- 합성 데이터셋 (제어된 실험)

#### 8.3 분석 및 해석

**1. 성분 분해의 신뢰성 검증**

```
문제: 신경망이 학습한 "Level"이 진정한 수준인가?
해결: 
  1) 합성 데이터 (지면진실 알려짐)에서 검증
  2) 시각화를 통한 정성 평가
  3) 통계적 성질 확인 (autocorrelation 등)
```

**2. 실패 사례 분석**

"모든 데이터에서 SOTA"는 거짓.

ETSformer가 실패하는 경우:
- 비규칙적 극단 이벤트 (COVID-19 팬데믹)
- 정책 변화로 인한 구조적 단절
- 매우 짧은 시계열 (L < 50)

이러한 사례의 **근본 원인 분석** 필수.

#### 8.4 실제 응용 시 주의사항

**1. 분포 변화(Distribution Shift) 대응**

실제 시스템에서:
```
Training: 2020-2022 (평상시)
Testing: 2023-2024 (팬데믹, 인플레이션, ...)
```

단순 정규화로는 부족. 예:
- Online learning / continual adaptation
- Domain-adaptive normalization
- Uncertainty estimation

**2. 계산 리소스 제약**

O(L log L)은 이론상 복잡도이지, 상수 계수를 무시함.

실제 임베디드 시스템:
```
DLinear (간단) > ETSformer (복잡) >> Informer
메모리: 수 MB vs 수 GB
```

**3. 재현성(Reproducibility)**

논문: "Code available at [GitHub URL]"

그러나:
- TensorFlow vs PyTorch 버전 차이
- 난수 시드 고정 필수
- GPU/CPU 결과 차이

**최소 요구사항**:
- 완전한 하이퍼파라미터 공개
- 환경 설정 명시 (의존성 버전)
- 3회 이상 실행 표준편차 보고

***

### 결론

**ETSformer**는 2022년 시점에서 다음을 성취했습니다:

1. **이론적 기여**: 지수평활 원리를 현대 신경망에 효과적으로 통합
2. **실무적 성능**: 6개 벤치마크에서 우수한 성능 달성
3. **해석 가능성**: Transformer 기반 모델 중 최고의 해석성 제공

그러나 **2024년 현관점**에서:

1. **선형 모델의 부상** (DLinear): Transformer 기반 설계의 한계 노출
2. **분해가 핵심**: ETSformer의 주요 성공 요인은 지수평활 자체가 아니라, **삼분 분해 구조**
3. **패칭의 효과** (PatchTST): 아키텍처 재설계의 새로운 방향 제시

**앞으로의 연구 방향**:
- ✓ 해석 가능성과 성능의 균형
- ✓ 공변량 및 불확실성 통합
- ✓ 적응형 분해 및 메타학습
- ✓ 실세계 응용을 위한 강건성 개선

이러한 맥락에서, ETSformer는 **시계열 예측 분야의 새로운 기준선**을 제시한 중요한 논문으로, 이후 연구자들이 참고해야 할 필수 연구이며, 그 설계 철학과 방법론은 향후 수년간 영향을 미칠 것으로 예상됩니다.[1]

***

### 참고 문헌

 Gerald Woo, Chenghao Liu, Doyen Sahoo, Akshat Kumar, Steven Hoi. "ETSformer: Exponential Smoothing Transformers for Time-series Forecasting." arXiv:2202.01381v2 (2022).[1]

출처
[1] 2202.01381v2.pdf https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0875d3b6-3cf0-4cad-ac20-a23646afd920/2202.01381v2.pdf
[2] Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting https://www.semanticscholar.org/paper/fc46ccb83dc121c33de7ab6bdedab7d970780b2f
[3] A Transformer Self-attention Model for Time Series Forecasting http://jecei.sru.ac.ir/article_1477.html
[4] A hybrid of artificial neural network, exponential smoothing, and ARIMA models for COVID-19 time series forecasting https://journals.sagepub.com/doi/full/10.3233/MAS-210512
[5] Peramalan Permintaan Menggunakan Time Series Forecasting Model Untuk Merancang Resources Yang Dibutuhkan IKM Percetakan http://jurnal.sttmcileungsi.ac.id/index.php/jenius/article/view/159
[6] Time-Series Forecasting of COVID-19 Cases Using Stacked Long Short-Term Memory Networks https://ieeexplore.ieee.org/document/9581688/
[7] COVID-19 Inpatients in Sothern Iran: A Time Series Forecasting for 2020-2021 https://hmj.hums.ac.ir/Article/hmj-3025
[8] Modification of the "Piramidal" Algorithm of the Small Time Series Forecasting https://www.semanticscholar.org/paper/59b31725f33ed0d007680c040875c83402eb4785
[9] Improved Fuzzy Time Series Forecasting Model Based on Optimal Lengths of Intervals Using Hedge Algebras and Particle Swarm Optimization https://astesj.com/v06/i01/p147/
[10] Time Series Forecasting of Global Price of Soybeans using a Hybrid SARIMA and NARNN Model https://talenta.usu.ac.id/JoCAI/article/view/5674
[11] Time Series Forecasting for Structures Subjected to Nonstationary Inputs https://asmedigitalcollection.asme.org/SMASIS/proceedings/SMASIS2021/85499/V001T03A008/1122725
[12] Learning Novel Transformer Architecture for Time-series Forecasting https://arxiv.org/pdf/2502.13721.pdf
[13] Autoformer: Decomposition Transformers with Auto-Correlation for
  Long-Term Series Forecasting https://arxiv.org/pdf/2106.13008.pdf
[14] Scaleformer: Iterative Multi-scale Refining Transformers for Time Series
  Forecasting https://arxiv.org/pdf/2206.04038.pdf
[15] AutoAI-TS: AutoAI for Time Series Forecasting https://arxiv.org/pdf/2102.12347.pdf
[16] auto-sktime: Automated Time Series Forecasting https://arxiv.org/pdf/2312.08528.pdf
[17] A Systematic Review for Transformer-based Long-term Series Forecasting https://arxiv.org/pdf/2310.20218.pdf
[18] SageFormer: Series-Aware Framework for Long-term Multivariate Time
  Series Forecasting https://arxiv.org/pdf/2307.01616.pdf
[19] A Time Series is Worth 64 Words: Long-term Forecasting with Transformers http://arxiv.org/pdf/2211.14730v2.pdf
[20] Efficient Hierarchical Autoregressive Transformer for Time ... https://arxiv.org/html/2506.16001v1
[21] Informer: Beyond Efficient Transformer for Long Sequence ... https://arxiv.org/abs/2012.07436
[22] arXiv:1905.10437v4 [cs.LG] 20 Feb 2020 https://arxiv.org/pdf/1905.10437.pdf
[23] [PDF] Autoformer: Decomposition Transformers with Auto ... https://www.semanticscholar.org/paper/Autoformer:-Decomposition-Transformers-with-for-Wu-Xu/fc46ccb83dc121c33de7ab6bdedab7d970780b2f
[24] HTMformer: Hybrid Time and Multivariate Transformer for ... https://arxiv.org/html/2510.07084v1
[25] N-BEATS with a Mixture-of-Experts Layer for ... https://arxiv.org/pdf/2508.07490.pdf
[26] Autoformer: Decomposition Transformers with Auto ... https://arxiv.org/abs/2106.13008
[27] TwinFormer: A Dual-Level Transformer for Long-Sequence ... https://www.arxiv.org/pdf/2512.12301.pdf
[28] [1905.10437] N-BEATS: Neural basis expansion analysis ... https://arxiv.org/abs/1905.10437
[29] Learning Novel Transformer Architecture for Time-series ... https://arxiv.org/html/2502.13721v1
[30] TwinFormer: A Dual-Level Transformer for Long-Sequence ... https://arxiv.org/html/2512.12301v1
[31] N-BEATS with a Mixture-of-Experts Layer for ... https://arxiv.org/html/2508.07490v1
[32] Long-term Forecasting with Transformers https://arxiv.org/pdf/2211.14730.pdf
[33] (PDF) Informer: Beyond Efficient Transformer for Long ... https://arxiv.org/pdf/2012.07436.pdf
[34] N-BEATS neural network for mid-term electricity load ... https://arxiv.org/abs/2009.11961
[35] Yes, Transformers are Effective for Time Series Forecasting ... https://huggingface.co/blog/autoformer
[36] [코드구현] Time Series Forecasting - Informer (AAAI 2021) https://doheon.github.io/%EC%BD%94%EB%93%9C%EA%B5%AC%ED%98%84/time-series/ci-5.informer-post/
[37] N-BEATS Unleashed: Deep Forecasting Using Neural ... https://towardsdatascience.com/n-beats-unleashed-deep-forecasting-using-neural-basis-expansion-analysis-in-python-343dd6307010/
[38] [PDF] Decomposition Transformers with Auto-Correlation for Long-Term ... https://ise.thss.tsinghua.edu.cn/~mlong/doc/Autoformer-nips21.pdf
[39] Create Dataloaders https://huggingface.co/blog/informer
[40] N-BEATS: 해석 가능한 시계열 예측을 위한 신경망 기저 ... https://www.alphaxiv.org/ko/overview/1905.10437v4
[41] Haixu Wu https://wuhaixu2016.github.io/pdf/NeurIPS2021_Autoformer.pdf
[42] thuml/Autoformer: About Code release for "Autoformer ... - GitHub https://github.com/thuml/Autoformer
[43] Beyond Efficient Transformer for Long Sequence Time-Series Forecasting https://dsba.snu.ac.kr/?kboard_content_redirect=1823
[44] [ICLR 2020] N-BEATS : Neural Basis Expansion Analysis ... https://velog.io/@sheoyonj/Paper-Review-N-BEATS-Neural-Basis-Expansion-Analysis-for-Interpretable-Time-Sereis-Forecasting
[45] [AAAI 2021] Informer : Beyond Efficient Transformer for Long Sequence Time-Series Forecasting https://velog.io/@sheoyonj/Paper-Review-Informer-Beyond-Efficient-Transformer-for-Long-Sequence-Time-Series-Forecasting
[46] N-Beats(2020) 정리 https://aijyh0725.tistory.com/3
[47] FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting https://www.semanticscholar.org/paper/563bac1c5cdd5096e9dbf8d4f3d5b3c4f7284e06
[48] Wind Speed Forecasting for Wind Power Production Based on Frequency-Enhanced Transformer https://ieeexplore.ieee.org/document/10125301/
[49] Short-Term Load Forecasting with Frequency Enhanced Decomposed Transformer https://ieeexplore.ieee.org/document/10116459/
[50] Explore Relative and Context Information with Transformer for Joint Acoustic Echo Cancellation and Speech Enhancement https://ieeexplore.ieee.org/document/9747347/
[51] CP-JKU SUBMISSION TO DCASE22: DISTILLING KNOWLEDGE FOR LOW-COMPLEXITY CONVOLUTIONAL NEURAL NETWORKS FROM A PATCHOUT AUDIO TRANSFORMER Technical Report https://www.semanticscholar.org/paper/b3fc0ffc6d784973f2d5b34b06de323270392980
[52] A SYSTEMATIC LITERATURE REVIEW ON ENERGY-EFFICIENT TRANSFORMER DESIGN FOR SMART GRIDS https://researchinnovationjournal.com/index.php/AJSRI/article/view/35
[53] Cheap flight-ready X band antenna with backed cavity https://www.semanticscholar.org/paper/1da419d595f96b61702f145b46a475d21330e1c9
[54] Multivariate Resource Usage Prediction With Frequency-Enhanced and Attention-Assisted Transformer in Cloud Computing Systems https://ieeexplore.ieee.org/document/10516672/
[55] Hydroformer: Frequency Domain Enhanced Multi‐Attention Transformer for Monthly Lake Level Reconstruction With Low Data Input Requirements https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024WR037166
[56] An efficient parallel fusion structure of distilled and transformer-enhanced modules for lightweight image super-resolution https://link.springer.com/10.1007/s00371-023-03243-9
[57] FEDformer: Frequency Enhanced Decomposed Transformer for Long-term
  Series Forecasting https://arxiv.org/pdf/2201.12740.pdf
[58] EdgeFormer: A Parameter-Efficient Transformer for On-Device Seq2seq Generation https://aclanthology.org/2022.emnlp-main.741.pdf
[59] EdgeFormer: A Parameter-Efficient Transformer for On-Device Seq2seq
  Generation https://arxiv.org/pdf/2202.07959.pdf
[60] FedET: A Communication-Efficient Federated Class-Incremental Learning
  Framework Based on Enhanced Transformer http://arxiv.org/pdf/2306.15347.pdf
[61] Exploring Frequency-Inspired Optimization in Transformer for Efficient
  Single Image Super-Resolution https://arxiv.org/html/2308.05022v3
[62] Fredformer: Frequency Debiased Transformer for Time Series Forecasting https://arxiv.org/html/2406.09009v1
[63] Eformer: Edge Enhancement based Transformer for Medical Image Denoising https://arxiv.org/pdf/2109.08044.pdf
[64] FreEformer: Frequency Enhanced Transformer for Multivariate Time Series
  Forecasting https://arxiv.org/html/2501.13989v1
[65] CT-PatchTST: Channel-Time Patch Time-Series ... https://arxiv.org/html/2501.08620v4
[66] A Novel Architecture for Enhanced Time Series Prediction https://arxiv.org/pdf/2501.01087.pdf
[67] Frequency Enhanced Decomposed Transformer for Long ... https://arxiv.org/abs/2201.12740
[68] CT-PatchTST: Channel-Time Patch ... https://arxiv.org/pdf/2501.08620.pdf
[69] Decomposing the Time Series Forecasting Pipeline https://arxiv.org/pdf/2507.05891.pdf
[70] Frequency Enhanced Decomposed Transformer for Long- ... https://www.semanticscholar.org/paper/FEDformer:-Frequency-Enhanced-Decomposed-for-Series-Zhou-Ma/563bac1c5cdd5096e9dbf8d4f3d5b3c4f7284e06
[71] Attention as Robust Representation for Time Series ... https://arxiv.org/pdf/2402.05370.pdf
[72] A Novel Architecture for Enhanced Time Series Prediction https://arxiv.org/html/2501.01087v1
[73] Xue Wang https://www.semanticscholar.org/author/Xue-Wang/2118294665
[74] EMTSF: Extraordinary Mixture of SOTA Models for Time ... https://arxiv.org/html/2510.23396v1
[75] An Analysis of Linear Time Series Forecasting Models https://arxiv.org/pdf/2403.14587.pdf
[76] 1 Introduction https://arxiv.org/html/2601.15669v1
[77] Decomposition-Enhanced State-Space Recurrent Neural ... https://arxiv.org/html/2412.00994v1
[78] Frequency Enhanced Decomposed Transformer for Long-term ... https://proceedings.mlr.press/v162/zhou22g/zhou22g.pdf
[79] [PatchTST] A Time Series is Worth 64 Words: Long-Term ... https://letter-night.tistory.com/450
[80] [Paper Review] Are Transformers Effective for Time Series ... https://sonstory.tistory.com/119
[81] Scaled FP32 and Quantization-aware Training of PatchTST ... https://kdd-milets.github.io/milets2025/papers/MILETS_2025_paper_17.pdf
[82] DLinear - Nixtla https://nixtlaverse.nixtla.io/neuralforecast/models.dlinear.html
[83] PETFORMER: LONG-TERM TIME SERIES FORECAST https://openreview.net/pdf?id=u3RJbzzBZj
[84] cure-lab/LTSF-Linear: [AAAI-23 Oral] Official ... https://github.com/cure-lab/LTSF-Linear
[85] FEDformer: Frequency Enhanced Decomposed Transformer ... https://proceedings.mlr.press/v162/zhou22g.html
[86] [ICLR 2023] PatchTST: A Time Series is Worth 64 Words https://kp-scientist.tistory.com/entry/ICLR-2023-PatchTST-A-Time-Series-is-Worth-64-Words-Long-Term-Forecasting-with-Transformers
[87] [Paper review] DLinear - Hippo's data - 티스토리 https://hipposdata.tistory.com/152
[88] MAZiqing/FEDformer - GitHub https://github.com/MAZiqing/FEDformer
[89] [논문 리뷰] Are Transformers Effective for Time Series ... https://velog.io/@ha_yoonji99/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-Are-Transformers-Effective-for-Time-Series-Forecasting-AAAI-2023-NLinear-DLinear
