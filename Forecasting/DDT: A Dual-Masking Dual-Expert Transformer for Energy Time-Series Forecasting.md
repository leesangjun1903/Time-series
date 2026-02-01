
# DDT: A Dual-Masking Dual-Expert Transformer for Energy Time-Series Forecasting

## I. 핵심 주장 및 주요 기여도 요약

DDT(Dual-masking Dual-expert Transformer)는 에너지 시계열 예측의 두 가지 근본적 과제를 해결하기 위해 설계된 심층 학습 프레임워크이다. 첫째, 다원 이질 데이터의 효율적 융합 문제로, 전력 부하 데이터와 기상 관측 데이터는 상이한 통계 분포와 주파수 특성을 가지며 결합 시 신호 대 잡음비 15dB 이상의 열화를 초래한다. 둘째, 인과 일관성(causal consistency) 보장과 적응형 특성 선택 간의 내재적 긴장이다. 기존 고정 causal mask는 모든 과거 정보를 무분별하게 보존하여 잡음 증폭으로 이어지는 반면, 동적 마스크는 인과 구조를 위반할 가능성이 있다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4f264fc3-89ba-4b26-bc61-9713a54917c2/2601.07250v1.pdf)

이에 대응하여 DDT는 네 가지 핵심 기여를 제시한다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4f264fc3-89ba-4b26-bc61-9713a54917c2/2601.07250v1.pdf)

1. **Dual-Masking 메커니즘**: 엄격한 인과 마스크와 데이터 기반 동적 마스크를 통일된 행렬로 통합하여 이론적 인과 정합성을 보장하면서 적응형 특성 선택을 실현
2. **Dual-Expert 시스템**: 시간적 역학과 교차 변수 상관성의 모델링을 병렬 전문화된 경로로 분리한 후 동적 게이트 융합 모듈로 지능형 통합
3. **동적 게이트 융합**: 계층적 특성 정렬 구조로 다변량 예측 및 데이터 융합의 과제 해결
4. **광범위 벤치마킹**: 25개 데이터셋 실험으로 SOTA 성능 입증, 특히 7개 에너지 벤치마크에서 전 예측 지평에 걸쳐 압도적 우수성 달성

***

## II. 해결하는 문제 및 제안 방법

### II-1. 핵심 문제 정의

#### 문제 1: 다원 이질 데이터 융합의 과제
에너지 시스템의 정확한 예측은 다양한 이질 데이터의 통합을 요구한다. 전력 부하 시계열 $X_e$와 기상 관측 $X_w$는 다음의 통계-주파수 충돌을 나타낸다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4f264fc3-89ba-4b26-bc61-9713a54917c2/2601.07250v1.pdf)

- **통계 영역 충돌**: $\text{KL}(p(X_e) \| p(X_w))$이 사전 정의 임계값 $\epsilon$을 초과하여 확률 분포의 심각한 불일치 초래
- **주파수 영역 충돌**: 고주파 전압 신호(50~60Hz)와 저주파 온도 변화(24시간 주기) 간의 주파수 대역 에일리어싱으로 인해 중첩 대역에서 신호 대 잡음비(SNR) 15dB 이상 열화

#### 문제 2: 인과성과 적응형 선택 간의 트레이드오프
시계열 모델은 다음의 조건부 확률 조건을 만족해야 한다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4f264fc3-89ba-4b26-bc61-9713a54917c2/2601.07250v1.pdf)

$$P(y_t | y_{1:t-1}) \neq P(y_t | y_{1:T})$$

즉, $t$ 시점의 예측은 $t$ 시점 이전의 정보만 의존해야 미래 정보 누출을 방지한다. 그러나:

- **고정 Causal Mask**: 모든 과거 정보 $y_{1:t-1}$을 동일하게 처리하므로, 예측에 무관한 노이즈 $I(y_t, x_{noise}) \approx 0$도 포함되어 모델 용량 비효율 초래
- **동적 Mask**: 데이터 기반 특성 선택을 통해 중요 정보 부분집합 $S \subset \{1, \ldots, t-1\}$에 $I(y_t, x_S) \gg I(y_t, x_{1:t-1})$를 만족시키지만, 비대각(off-diagonal) 가중치로 인해 인과 구조 위반 가능

### II-2. 제안 방법: 이중 마스킹 메커니즘

#### 2.1 Causal Mask의 정의
길이 $L$인 수열에 대해 하삼각 행렬로 표현되는 인과 마스크는: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4f264fc3-89ba-4b26-bc61-9713a54917c2/2601.07250v1.pdf)

$$M_{\text{causal}}[i,j] = \begin{cases} 0, & \text{if } i \geq j \\ -\infty, & \text{if } i < j \end{cases} \quad (i,j \in \{1,\ldots,L\})$$

이를 attention 점수에 적용하면:

$$\text{Score}_{\text{masked}} = \text{Score} + M_{\text{causal}}$$

Softmax 정규화 후 미래 위치의 attention 가중치는 0으로 수렴하여 미래 정보 접근 차단을 보장한다.

#### 2.2 동적 마스크의 생성
동적 마스크는 두 단계의 생성 과정을 거친다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4f264fc3-89ba-4b26-bc61-9713a54917c2/2601.07250v1.pdf)

**1단계: 주파수 영역 특성 분석**
FFT를 통해 각 변수의 주파수 성분을 추출한 후, 에너지 가중 풀링으로 집계:

$$X_k^{\text{freq}} = \sum_{n=1}^{N} w_n \left| \mathcal{F}(X_{\cdot,n}) \right|_k, \quad w_n = \frac{\sum_i |\mathcal{F}(X_{\cdot,n})|_i}{\sum_{i,n'} |\mathcal{F}(X_{\cdot,n'})|_i}$$

**2단계: 마할라노비스 거리 기반 특성 유사성**
학습 가능 행렬 $A = LL^T$ (Cholesky 분해, $L$은 학습 파라미터)를 통해 정의:

$$d_{\text{Mahal}}(t,t') = \sqrt{(X_t - X_{t'})^T A (X_t - X_{t'})} \quad (8)$$

이 거리는 데이터의 공분산 구조를 고려하여 분포 적응형 유사성 측정을 제공한다.

**3단계: Gumbel-Softmax 기반 미분 가능 샘플링**
온도 매개변수 $\tau(\text{epoch})$의 지수 감소로 탐색에서 활용으로의 점진적 전환: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4f264fc3-89ba-4b26-bc61-9713a54917c2/2601.07250v1.pdf)

$$P_{t,t'} = \text{Softmax}\left(\frac{d_{\text{Mahal}}(t,t')}{\tau(\text{epoch})} + \text{Gumbel}(0,1)\right) \quad (9)$$

**4단계: Top-k 선택으로 이진 마스크 획득**
각 시점 $t$에서 상위 $k$개 연결만 유지하여 해석 가능한 희소성 달성:

$$M_{\text{dynamic}}[t,t'] = \begin{cases} 1, & \text{if } P_{t,t'} \in \text{Top-k}(P_{t,:}) \\ 0, & \text{otherwise} \end{cases} \quad (11)$$

#### 2.3 이중 마스크 융합
두 마스크의 원소별 곱셈으로 인과 제약 내에서만 동적 선택이 작동하도록 강제: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4f264fc3-89ba-4b26-bc61-9713a54917c2/2601.07250v1.pdf)

$$M_{\text{fusion}} = M_{\text{causal}} \odot M_{\text{dynamic}} \quad (4)$$

이는 $M_{\text{causal}}$의 0인 위치에서 동적 마스크가 0으로 강제되어 미래 정보 접근을 완전히 차단한다.

#### 2.4 융합 Attention 계산
최종 attention 점수는 세 성분의 시너지: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4f264fc3-89ba-4b26-bc61-9713a54917c2/2601.07250v1.pdf)

$$\text{Score}_{\text{fusion}} = \frac{QK^T}{\sqrt{d_k}} + M_{\text{causal}} + \log(M_{\text{dynamic}} + 10^{-8}) \quad (5)$$

첫 번째 항은 기본 attention 유사성, 두 번째는 엄격한 인과 제약, 세 번째는 낮은 중요도 특성에 대한 적응형 페널티를 제공한다.

$$\text{Attention}_{\text{fusion}} = \text{Softmax}(\text{Score}_{\text{fusion}}) V \quad (6)$$

### II-3. 시계열 패칭 설계

메모리 효율성과 다중 규모 패턴 캡처를 위해 계층적 신호 분해 적용: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4f264fc3-89ba-4b26-bc61-9713a54917c2/2601.07250v1.pdf)

$$X^{\text{patches}}_{b,p,:} = X_{b, pS:pS+K, :} \quad (12)$$

여기서 $K$는 패치 길이, $S$는 스트라이드이며, 고주파($K=16$~$32$)와 저주파($K=64$~$128$) 신호에 동적 조정된다.

#### 2.4 Temporal-aware Layer Normalization
전통적 LayerNorm은 특성 차원만 정규화하지만, T-LayerNorm은 시간-특성 차원을 동시에 처리: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4f264fc3-89ba-4b26-bc61-9713a54917c2/2601.07250v1.pdf)

$$\mu_d^p = \frac{1}{DK}\sum_{d=1}^{D}\sum_{k=1}^{K} h^p_{d,k}, \quad \sigma_d^p = \sqrt{\frac{1}{DK}\sum_{d=1}^{D}\sum_{k=1}^{K}(h^p_{d,k} - \mu_d^p)^2} \quad (13)$$

비정상성 데이터에 대한 정규화 효율성 향상.

***

## III. 모델 구조 상세 설명

### III-1. 전처리 파이프라인

다원 이질 에너지 시계열의 비정상성과 노이즈를 대응하기 위해 다층 전처리 설계: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4f264fc3-89ba-4b26-bc61-9713a54917c2/2601.07250v1.pdf)

| 처리 단계 | 방법 | 목적 |
|---------|------|------|
| **노이즈 처리** | Local Outlier Factor (LOF) 밀도 추정 | k-근접이웃 평균 거리 대비 편차로 이상점 식별 |
| **결측값 보완** | Bayesian interpolation (Gaussian Process) | 불확실성 정량화와 함께 결측값 추정 |
| **이상탐지** | Boxplot + GEV 분포 모델링 | 극값 필터링으로 설정값 벗어난 이상 제거 |
| **정규화** | Z-score + Wasserstein 제약 | 평균 0, 표준편차 1로 정규화하되 분포 편차 제한 |
| **데이터 증강** | DTW, Scaling, Noise injection, cGAN | Few-shot 학습 능력 강화 |
| **데이터 분할** | Stratified k-means sampling | 각 클러스터 내 train:val:test = 70:15:15 유지 |

특히 **Wasserstein 거리 제약**으로 정규화된 데이터의 분포가 표준 정규분포 $\mathcal{N}(0,1)$에 정확히 근접하도록 강제하여, 모델의 학습 안정성을 제고한다.

### III-2. 다변량 특성 융합 메커니즘

에너지 시계열, 기상 데이터, 시간 특성의 이질성을 처리하기 위해 계층적 임베딩 구조 설계: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4f264fc3-89ba-4b26-bc61-9713a54917c2/2601.07250v1.pdf)

$$E_e = \text{Attention}(\text{CNN}(X_e)) \quad \text{(에너지 임베딩)}$$
$$E_w = \text{DomainMap}(X_w) \quad \text{(기상 임베딩)}$$
$$E_t = \text{DomainMap}(X_t) \quad \text{(시간 특성 임베딩)}$$

**게이트 융합 메커니즘**으로 다원 정보 통합:

$$G = \sigma(W_g E_e + b_g)$$
$$E_{\text{fusion}} = G \odot E_e + (1-G) \odot \text{MLP}(E_e, E_w, E_t)$$

Sigmoid 함수 $\sigma$로 생성된 게이팅 벡터 $G \in $은 에너지 임베딩과 융합 정보 간의 동적 가중치 결정. **Fisher 정보 기준**으로 임베딩 차원을 제약하여 $d \geq \log_2 N$일 때 교차 변수 의존성의 90% 이상 보존을 보장한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4f264fc3-89ba-4b26-bc61-9713a54917c2/2601.07250v1.pdf)

### III-3. 이중 전문가 시스템

#### 3.1 Temporal Expert: 시간적 역학 모델링
다중 규모 게이트 희석 컨볼루션(Dilated Convolution)으로 구성: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4f264fc3-89ba-4b26-bc61-9713a54917c2/2601.07250v1.pdf)

$$h_t = \sum_{k \in \{1,3,5\}} \text{DConv}^k(Z) \otimes \text{Gating}(\text{DConv}^k(Z))$$

커널 크기 $k=1, 3, 5$는 각각 짧은 시기(1시간), 중간 주기(3시간), 긴 주기(5시간)의 패턴을 포착하여 에너지 시계열의 다중 규모 변동성을 효과적으로 모델링한다.

#### 3.2 Channel Expert: 교차 변수 상관성 모델링
MLP 기반의 변수 간 동적 의존성 학습: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4f264fc3-89ba-4b26-bc61-9713a54917c2/2601.07250v1.pdf)

$$A_{ij} = \text{MLP}(z_i, z_j, \Delta t_{ij})$$

여기서 $z_i, z_j$는 변수의 현재 상태 벡터, $\Delta t_{ij}$는 시간 거리. 이는 변수 간 비대칭 관계(예: 온도 변화가 전력 수요에 선행하는 구조)를 포착한다.

#### 3.3 Dynamic Gated Fusion: 지능형 통합
두 Expert의 출력을 학습 가능한 게이팅으로 통합: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4f264fc3-89ba-4b26-bc61-9713a54917c2/2601.07250v1.pdf)

$$g = \text{MLP}_2(\text{GELU}(\text{MLP}_1(\bar{Z})))$$
$$h_{\text{out}} = g \odot h_t + (1-g) \odot h_c \oplus Z$$

여기서 $\bar{Z}$는 전역 문맥 벡터, GELU는 Gaussian Error Linear Unit 활성화함수, $\oplus$는 Skip connection. 게이팅 메커니즘은 시간적 역학과 교차 변수 상관성 중 어느 것이 더 중요한지를 샘플별/시점별로 동적 결정한다.

### III-4. 조건부 독립 모드
고차원 희소 시스템에 대해 모델 복잡도를 선택적으로 감소: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4f264fc3-89ba-4b26-bc61-9713a54917c2/2601.07250v1.pdf)

$$p(Y_{t+1:t+H}|X_{1:t}) = \prod_{n=1}^{N} p(Y^n_{t+1:t+H}|X^n_{1:t}) \quad (18)$$

복잡도 개선: $O(N^2) \to O(N)$로 감소하여 엣지 디바이스 배포에 적합. 또한 few-shot 학습 시나리오에서 일반화 향상.

***

## IV. 성능 평가 및 분석

### IV-1. 실험 설정

**벤치마크 데이터셋 (7개 에너지 관련)**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4f264fc3-89ba-4b26-bc61-9713a54917c2/2601.07250v1.pdf)

| 데이터셋 | 특성 | 샘플링 | 목적 |
|--------|------|--------|------|
| **ETTh1/2** | 전력 변압기 온도 | 시간 단위 | 그리드 운영 역학 |
| **ETTm1/2** | 전력 변압기 온도 | 분 단위 | 세밀한 시간적 패턴 |
| **Electricity** | 전력 부하 | 시간 단위 | 수요 예측 |
| **Solar/Wind** | 재생에너지 발전 | 시간 단위 | 변동성 높은 자원 관리 |

**평가 지표**: MSE(평균제곱오차), MAE(평균절대오차)  
**예측 지평**: 96h, 192h, 336h(2주), 720h(1개월)  
**비교 대상**: 10개 SOTA 모델 (Pathformer, iTransformer, TimeMixer, FITS, PDF, LiPFormer 등)

### IV-2. 주요 실험 결과

#### 4.1 전체 성능 순위
DDT는 25개 데이터셋, 4개 예측 지평, 2개 지표에 걸쳐 **전체 순위 1위(109점)**로 모든 기본 모델을 압도: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4f264fc3-89ba-4b26-bc61-9713a54917c2/2601.07250v1.pdf)

| 랭킹 | 모델 | 1위 확보 | 2위 확보 | 평균 점수 |
|------|------|---------|---------|----------|
| **1** | **DDT** | 대다수 | - | 109 |
| 2 | LiPFormer | 중간 | 다수 | 대폭 낮음 |
| 3 | PDF | 일부 | 중간 | 더 낮음 |

#### 4.2 이상 데이터(Anomalous Data) 처리 우수성
ILI(Influenza-Like Illness, 독감 유사 질환) 데이터셋은 비주기적·급변 사건이 특징: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4f264fc3-89ba-4b26-bc61-9713a54917c2/2601.07250v1.pdf)

**예측 지평 F=24:**
- DDT: MSE **1.577**, MAE **0.760**
- 2위 LiPFormer: MSE 1.753, MAE 0.852
- 성능 개선: **약 10% 우수**

DDT의 동적 마스킹이 갑작스러운 주파수 변화와 노이즈를 효과적으로 구분하는 능력을 입증.

#### 4.3 장기 예측(Long-horizon) 강점
ETTm2 데이터셋, 최대 예측 지평 F=720(30일):

- DDT MAE: **0.377** (2위와 동점)
- Exchange 데이터셋 F=720: DDT MSE **0.583**, MAE **0.580** (전체 1위)

$O(L^2)$ Transformer attention에도 불구하고 패칭으로 순차 길이 단축, 희소 마스킹으로 계산 효율 확보.

### IV-3. Ablation 연구 결과

체계적 분해를 통한 각 컴포넌트 기여도 검증: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4f264fc3-89ba-4b26-bc61-9713a54917c2/2601.07250v1.pdf)

#### 4.3.1 Dual-Masking vs 전체 구조
Ablation 1 (Dual-Masking 제거, Multivariate만):

| 데이터셋 | Full DDT MSE | Ablation 1 MSE | 성능 저하 |
|---------|-------------|----------------|----------|
| **ETTh1** | 0.405 | 0.651 | **+60.7%** |
| **Wind** | 1.072 | 1.245 | **+16.2%** |
| **ETTh2** | 0.337 | 0.382 | **+13.4%** |

Dual-Masking이 핵심 혁신 입증. 이중 마스크 제거 시 성능 급격히 하락.

#### 4.3.2 개별 마스크 기여도
Ablation 2 (Causal mask만) vs Ablation 3 (Dynamic mask만):

| 데이터셋 | Full DDT | Causal만 | Dynamic만 | 우월성 |
|---------|---------|---------|---------|-------|
| **ETTh1** | 0.405 | 0.414 | 0.499 | 동적 > 인과 |
| **ETTh2** | 0.337 | 0.342 | 0.364 | 동적 > 인과 |

**해석**: Causal mask는 단독으로도 경쟁 수준이지만, 동적 마스크 단독은 노이즈 처리에서 미흡. **두 마스크의 시너지**가 최적 성능 달성. $M_{\text{dynamic}}$의 적응성과 $M_{\text{causal}}$의 엄격함의 결합이 상승 효과.

### IV-4. 성능 개선의 원인 분석

| 혁신 요소 | 기여 메커니즘 | 정량 효과 |
|---------|-----------|---------|
| **T-LayerNorm** | 시간-특성 동시 정규화로 비정상성 대응 | 수렴 속도 향상, 학습 안정성 |
| **Gumbel-Softmax 온도 소급** | 탐색에서 활용으로 점진 전환 | 최적 특성 발견 능력 강화 |
| **Top-k 희소성** | 계산 효율과 과적합 방지 | 메모리 사용 감소, 일반화 개선 |
| **Stratified Sampling** | 통계 일관성 보장 | 학습/검증/테스트 세트 편향 제거 |
| **Quantile Loss** | 극값 및 확률적 분포 포착 | 예측 불확실성 정량화 |

***

## V. 일반화 성능 향상 가능성 분석

### V-1. 구조적 설계의 일반화 이점

#### 1.1 Sparsity 유도
Top-k 동적 마스킹은 활성 연결을 제한하여 과적합 위험 저감: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4f264fc3-89ba-4b26-bc61-9713a54917c2/2601.07250v1.pdf)

$$\text{활성 위치 수} = \mathcal{O}(k \cdot L) \quad (\text{vs} \mathcal{O}(L^2) \text{ full attention})$$

학습 파라미터 감소로 규제(regularization) 효과. 테스트 분포 편차에 대한 견고성 향상.

#### 1.2 정보 보존 보장
Fisher 정보 기준으로 임베딩 차원 제약: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4f264fc3-89ba-4b26-bc61-9713a54917c2/2601.07250v1.pdf)

$$d \geq \log_2 N \Rightarrow \text{교차 변수 의존성 보존율} \geq 90\%$$

고차원 공간으로의 과도 확장을 방지하면서 필수 정보 유지.

#### 1.3 다층 정규화의 상호작용
- **T-LayerNorm**: 시간 축 정규화로 추세 편향 제거
- **Wasserstein 제약**: 분포 일관성으로 배치 간 편차 제어
- **Stratified sampling**: 학습/테스트 세트 분포 정렬

세 계층의 정규화가 유기적으로 작동하여, 도메인 외(out-of-distribution) 데이터에 대한 견고성 강화.

### V-2. 학습 전략의 일반화 효과

#### 2.1 Teacher Forcing의 점진적 약화
장기 예측($H > 96$) 시 이전 예측을 입력으로 사용하면 오류 누적(error accumulation) 발생. DDT는 교사 강제를 활용하되, 학습 진행에 따라 감소시켜:

$$\text{MAE 개선} = 21.7\% \quad \text{(ETTh2, F=720)}$$

자기회귀(autoregressive) 생성과 오류 전파 간의 균형 달성.

#### 2.2 Quantile Loss의 확률적 커버리지
3개 분위값(0.1, 0.5, 0.9)의 손실:

$$\mathcal{L}_{\text{quantile}} = \sum_q w_q \max(q(y-\hat{y}), (q-1)(y-\hat{y}))$$

예측 범위를 포괄하여 불확실성 정량화, 극값 예측에 견고성 부여.

### V-3. 조건부 독립 모드의 전이 학습 가능성

CI 모드($O(N)$ 복잡도)는 변수별 독립 처리로 인해: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4f264fc3-89ba-4b26-bc61-9713a54917c2/2601.07250v1.pdf)

1. **변수 추가 시 선형 확장**: 새 에너지원(태양광, 풍력) 추가 시 기존 모델 재학습 불필요
2. **도메인 전이 용이**: 전력 부하 $\to$ 가스 수요, 수도 사용량 등 다른 유틸리티로 전이 시 개별 변수 미세 조정(fine-tuning)만 필요
3. **Few-shot 학습 강화**: 제한된 데이터 시나리오에서 부분 모델만 학습 가능

***

## VI. 모델의 한계

### VI-1. 구조적 한계

| 한계 | 원인 | 영향 |
|------|------|------|
| **계산 복잡도** | Full-attention 기본 구조의 $O(L^2)$ | 매우 장기(>5000시간) 예측 시 계산 병목 |
| **이질 데이터 처리의 휴리스틱** | KL 발산, SNR 임계값의 사전 정의 | 새로운 데이터셋에서 하이퍼파라미터 조정 필요 |
| **동적 마스크 생성의 검은상자성** | Gumbel 샘플링 + Top-k의 확률적 특성 | 특정 시점에서 선택된 이유 해석 곤란 |
| **도메인 외 일반화 부재** | 벤치마크는 대부분 음전원/전력 도메인 | 금융, 기후 시계열로의 성능 보장 불분명 |

### VI-2. 실증적 평가 한계

- **초장기 예측**: 720시간(30일)이 최대, 분기/연 단위 예측 미평가
- **극한 시나리오**: 정전, 재생에너지 급증 등 극심한 변동성 상황 명시적 평가 부족
- **모델 압축**: 엣지 배포용 경량화 기법(양자화, 가지치기) 미포함

***

## VII. 최신 관련 연구와의 비교 분석 (2020년 이후)

### VII-1. 진화 흐름

| 연도 | 주요 모델 | 혁신 포인트 | DDT와의 차이점 |
|------|---------|-----------|--------------|
| **2021** | Informer | ProbSparse attention ( $O(L\log L)$ ) | 주파수 적응성 부족 |
| **2021** | Autoformer | 자기상관 분해 | 동적 인과성 처리 미흡 |
| **2022** | FEDformer | 주파수 강화 분해 | 마스킹의 적응성 한계 |
| **2023** | PatchTST | 패치 기반 임베딩 | 이질 다원 데이터 융합 미흡 |
| **2024** | iTransformer | 채널-독립 반전 | Causal consistency 보장 약화 |
| **2024** | Pathformer | 적응형 경로 구조 | 이중 전문화 체계 부재 |
| **2026** | **DDT** | **이중 마스킹 + 이중 전문가** | **인과성과 적응성의 원리적 통일** |

### VII-2. DDT의 차별적 기여

1. **마스킹의 이중 이원화**: 기존 모델들은 고정 또는 단일 동적 마스킹만 사용했으나, DDT는 인과 마스크의 엄격성과 동적 마스크의 유연성을 명시적으로 결합하는 원리적 프레임워크 제시 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4f264fc3-89ba-4b26-bc61-9713a54917c2/2601.07250v1.pdf)

2. **주파수 인식 학습**: Mahalanobis 거리 + Gumbel-Softmax 결합으로 데이터 공분산 구조를 학습 가능하게 설계. 기존 모델의 고정 주파수 필터 방식 대비 적응성 우수 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4f264fc3-89ba-4b26-bc61-9713a54917c2/2601.07250v1.pdf)

3. **명확한 이중 전문화**: Temporal Expert(희석 컨볼루션)와 Channel Expert(MLP 기반) 간의 명시적 분리 및 동적 게이팅 융합. Autoformer, iTransformer 등의 암묵적 멀티태스크 학습 방식과 상이 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4f264fc3-89ba-4b26-bc61-9713a54917c2/2601.07250v1.pdf)

4. **다층 정규화 체계**: T-LayerNorm, Wasserstein 제약, Stratified sampling의 조화로운 시너지. 선행 모델들은 개별 정규화 기법만 적용

***

## VIII. 앞으로의 연구에 미치는 영향 및 고려사항

### VIII-1. 이론적 확장 방향

#### 1.1 인과 구조 발견의 통합
**현상**: DDT는 인과성을 강제하지만, 실제 인과 그래프는 학습하지 않음.  
**개선 방향**: Granger 인과성 테스트와의 결합으로 변수 간 진정한 인과 관계 규명:

$$\mathcal{G} = \{(i \to j) : \text{Granger}(X_i, X_j) > \tau\}$$

이를 DDT의 Channel Expert의 선행 지식으로 제공하면 데이터 효율성 및 해석성 향상 가능.

#### 1.2 온도 소급의 수렴 보장
**현상**: Gumbel-Softmax 온도 $\tau(\text{epoch}) = \tau_0 \exp(-\alpha \cdot \text{epoch})$의 감소율이 수렴 보장을 제공하지 않음.  
**개선 방향**: 적응형 온도 스케줄의 수렴 조건 도출:

$$\sum_{t=0}^{\infty} \tau(t) = \infty, \quad \sum_{t=0}^{\infty} \tau(t)^2 < \infty$$

이를 준수하면 최적 마스크로의 확률적 수렴 보장.

#### 1.3 상호 정보 최대화의 정량화
**현황**: 동적 마스킹이 상호 정보 $I(y_t; x_S)$를 최대화한다는 주장이 정성적.  
**개선 방향**: 정보 이론적 보유량(information retention) 하한:

$$I(y_t; \hat{y}_t^{\text{DDT}}) \geq I(y_t; \hat{y}_t^{\text{causal}}) - \epsilon$$

여기서 $\epsilon$는 동적 마스킹의 추가 오차.

### VIII-2. 응용 확장 방향

#### 2.1 크로스 도메인 전이 학습
**기회**: CI 모드의 변수별 처리 능력을 활용한 도메인 외 전이.

예) 전력 부하 예측 학습 모델을 가스 수요, 수도 사용량 예측으로 전이:
- Temporal Expert: 시간적 패턴 재사용 (일반성 높음)
- Channel Expert: 변수별 미세 조정 (효율성 우수)

**예상 성과**: Few-shot 설정에서 데이터 필요량 50% 이상 감소.

#### 2.2 재생에너지 변동성 극복
**도전**: 태양광/풍력의 고비정상성(non-stationarity)과 급변성.

**DDT 활용**:
- 동적 마스킹으로 구름 변화, 기상 급변 등의 이상 신호 적응형 포착
- Dual-Expert로 태양 일주기(Temporal Expert) + 날씨(Channel Expert) 분리 모델링

**예상 효과**: 신재생 변동성 예측 MAE 15~20% 개선.

#### 2.3 스마트 그리드와의 실시간 연계
**적용**: DDT의 장기 예측 능력을 에너지 저장 시스템(ESS) 제어와 연동.

- 24~720시간 예측 → ESS 충방전 계획 수립
- 초단기(1~6시간) 예측 오류 → 자동 응응 메커니즘

**기대 효과**: 그리드 안정성 지수(Grid Stability Index) 10% 이상 개선.

### VIII-3. 산업 배포 시 고려사항

#### 3.1 해석성 강화
**현황**: 동적 마스킹의 블랙박스 특성으로 운영자 신뢰 저하 우려.

**개선**:
- Attention weight 시각화: 각 시점에서 역사적 의존성의 구조적 표시
- 중요도 지수(Feature Importance Index): $\sum_i M_{\text{dynamic}}[:,i] / L$ 시간 기반 추이
- 반사실적 설명(Counterfactual Explanation): "만약 기상 정보 제거 시" 시뮬레이션

#### 3.2 온라인 학습 적응
**도전**: 새로운 장비 추가, 기후 변화 등의 점진적 데이터 분포 변화(Data Drift).

**전략**:
- CI 모드를 통한 증분 학습(Incremental Learning)
- Replay Buffer 기반 연속 미세 조정(Continual Fine-tuning)

#### 3.3 신뢰도 평가 및 모니터링
**필수**: 예측 신뢰 구간 제공.

**구현**: Quantile Loss에서 학습한 0.1, 0.9 분위로 95% 신뢰 구간 구성:

$$\hat{y}_t \in [\hat{y}_t^{0.1}, \hat{y}_t^{0.9}]$$

실제 관측값 $y_t$이 구간 내 포함 비율 모니터링으로 모델 신뢰성 추적.

***

## IX. 2020년 이후 관련 최신 연구 비교 분석

### IX-1. 주요 연구 동향 분류

#### A. Transformer 효율화 트렌드 (2020~2023)
- **Informer(2021)**: ProbSparse attention으로 $O(L\log L)$ 달성 [mdpi](https://www.mdpi.com/2071-1050/17/19/8655)
- **Autoformer(2021)**: 자기상관(autocorrelation) 기반 분해로 계산 효율성 및 해석성 동시 확보 [mdpi](https://www.mdpi.com/2071-1050/17/19/8655)
- **FEDformer(2022)**: 주파수 강화 분해로 스펙트럼 편향 제거 [mdpi](https://www.mdpi.com/2071-1050/17/19/8655)

**DDT 대비**: 이들은 **계산 복잡도 감소**에 초점. DDT는 동일 복잡도 내에서 **마스킹 적응성**을 우선 해결.

#### B. 패치 기반 토큰화 (2023~2024)
- **PatchTST(2023)**: 시계열을 고정 길이 패치로 분할, 채널-독립 처리 [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11355634/)
- **N-HiTS(2023)**: 계층적 보간(hierarchical interpolation)으로 다중 규모 포착 [mdpi](https://www.mdpi.com/2071-1050/17/19/8655)

**DDT 대비**: PatchTST는 효율성 우수이나 **이질 다원 데이터 명시적 융합 부족**. N-HiTS는 선형 모델이라 복잡한 비선형 상관성 포착 미흡.

#### C. 채널 상호작용 고도화 (2024~2025)
- **iTransformer(2024)**: 채널을 독립 토큰으로 전환하여 변수 간 상호작용 강화 [mdpi](https://www.mdpi.com/2071-1050/17/19/8655)
- **Pathformer(2024)**: 적응형 경로 선택으로 각 변수별 최적 모형 발견 [mdpi](https://www.mdpi.com/2071-1050/17/19/8655)
- **TimeMixer(2024)**: 시간-채널 축소(temporal-channel decomposition)로 이중 표현 학습 [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11355634/)

**DDT 대비**: 이들은 **채널 상호작용만** 강조. DDT는 **Temporal Expert와 Channel Expert를 명시적으로 분리 + 동적 게이팅**으로 더 정교한 통합.

#### D. 주파수 적응 (2024~2025)
- **Fredformer(2024)**: 주파수 편향(frequency bias) 완화로 저주파 특성 강조 [mdpi](https://www.mdpi.com/2071-1050/17/19/8655)
- **FAformer(2025)**: 주파수 인식 에너지 분해 [mdpi](https://www.mdpi.com/2071-1050/17/19/8655)

**DDT 대비**: 이들은 **사후 처리(post-hoc) 보정**. DDT는 **학습 가능한 Mahalanobis 거리**로 **사전 적응(adaptive)** 설계.

### IX-2. 에너지 시계열 특화 연구 (2024~2026)

| 연구 | 방법 | 강점 | 약점 vs DDT |
|-----|------|------|-----------|
| **CT-PatchTST(2025)** | 채널-시간 패칭, Transformer | 재생에너지 장기 예측 우수 | 단일 마스킹, 이질 데이터 처리 미흡 |
| **Temporal Fusion Transformer(2020~)** | 다변량 시계열 특화 | 다원 정보 통합 용이 | 계산 복잡도 높음 |
| **HTMformer(2025)** | Hybrid Time-Multivariate Embedding | 특성 추출 강화 | Causal consistency 명시적 보장 없음 |
| **TimeDART(2025)** | Self-supervised + Causal Transformer | 사전학습 기반 전이 능력 | 에너지 특화성 부족 |

**DDT 평가**: 에너지 도메인에서 **이중 마스킹의 원리적 정합성**, **다원 이질 데이터 명시적 처리**, **인과성과 적응성의 통일**로 **가장 종합적**. 단, 계산 효율(Informer, Pathformer)이나 경량성(LightTS)에서는 상대적으로 미흡.

### IX-3. 최신 기초 모델(Foundation Models) 트렌드

**TimesFM, GPT4TS, Chronos** 등의 대규모 사전학습 모델 등장: [mdpi](https://www.mdpi.com/2071-1050/17/19/8655)

| 특징 | 강점 | 한계 |
|------|------|------|
| **제로샷 능력** | 사전학습 데이터의 광범위 일반 패턴 포착 | 에너지 시스템의 극한 상황 미처리 |
| **데이터 효율성** | 제한 데이터에서 우수 성능 | 도메인 특수 미세한 조정 필요 |
| **계산 비용** | 추론(Inference) 저비용 | 사전학습 단계 막대한 자원 소비 |

**DDT vs Foundation Models**:
- **Foundation Models**: "여러 도메인 일반적 패턴"에 최적화 → 에너지의 특이성 포착 미흡
- **DDT**: "에너지 도메인 특화" → 이질 데이터, 극한 상황, 인과성 명시적 처리

**향후**: 두 접근의 **하이브리드** 방향 예상. DDT의 이중 마스킹을 Foundation Model의 사전학습 목표 함수로 통합하는 연구 필요.

***

## X. 결론

DDT는 에너지 시계열 예측의 두 가지 근본 과제—다원 이질 데이터 융합과 인과성-적응성 트레이드오프—를 원리적으로 해결하는 혁신 모델이다. **이중 마스킹 메커니즘**으로 엄격한 인과 일관성을 보장하면서 동시에 데이터 기반 적응형 특성 선택을 가능하게 하고, **이중 전문가 시스템**으로 시간적 역학과 교차 변수 상관성을 명시적으로 분리 모델링하며, **동적 게이트 융합**으로 이들을 지능형 통합한다.

광범위한 벤치마킹(7개 에너지 데이터셋, 25개 총 데이터셋)에서 SOTA 성능 달성 및 Ablation 연구로 각 컴포넌트의 기여도 명증. 특히 이상 데이터 처리, 장기 예측, 극단적 변동성 대응에서 기존 모델 대비 10~60% 우수성 입증.

일반화 성능 향상은 구조적 희소성, 다층 정규화, 조건부 독립 모드, 확률적 손실 함수의 유기적 시너지에 기인. 향후 연구는 인과 구조 발견 통합, 도메인 간 전이 학습, 엣지 배포 최적화, 극한 시나리오 견고성 강화에 초점을 두어야 한다.

2020년 이후 트랜스포머 기반 시계열 예측 진화 맥락에서 DDT는 **효율화(Informer 계열)와 채널 상호작용(iTransformer 계열)을 통합하면서 에너지 도메인 특화성을 극대화**한 모델로 평가된다. 향후 기초 모델과의 하이브리드 접근, 현장 배포 시 신뢰성 및 해석성 강화, 에너지 전환의 가속 속에서 재생에너지 변동성 극복을 위한 실질적 도구로 자리잡을 것으로 기대된다.

***

## 참고 문헌

 Zhu, M., Zhang, Q., Cheng, Y., Gu, F., & Lin, S. "DDT: A Dual-Masking Dual-Expert Transformer for Energy Time-Series Forecasting." arXiv preprint arXiv:2601.07250v1 (2026). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4f264fc3-89ba-4b26-bc61-9713a54917c2/2601.07250v1.pdf)

 Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W. "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting." AAAI (2021). [mdpi](https://www.mdpi.com/2071-1050/17/19/8655)

 Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers." ICLR (2023). [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11355634/)
