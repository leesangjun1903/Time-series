# TimeGrad: Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting

### 1. 핵심 주장 및 주요 기여

**TimeGrad**는 다변량 시계열 데이터의 확률적 예측을 위해 **에너지 기반 모델(Energy-Based Model, EBM)과 자기회귀(autoregressive) 구조를 결합한 새로운 생성 모델**입니다. 본 논문의 핵심 주장은 다음과 같습니다.[1]

**주요 기여:**

1. **확산 확률 모델(Diffusion Probabilistic Models)의 적용**: 경사 추정을 통해 각 시간 단계에서 데이터 분포를 샘플링하는 방식을 제안하여, 기존의 제한적인 분포 가정(예: 가우시안)을 벗어날 수 있게 함[1]

2. **자기회귀-EBM 결합의 장점**: 자기회귀 모델의 외삽(extrapolation) 성능과 EBM의 고차원 분포 모델링 유연성을 동시에 확보하면서도 계산 가능성 유지[1]

3. **상태-of-더-아트(SOTA) 성능**: 6개의 실제 데이터셋(수천 개의 상관된 차원)에서 기존의 모든 다변량 확률적 예측 방법들을 능가[1]

4. **규제되지 않은 함수 근사기**: 정규화 흐름(normalizing flows)과 달리 야코비안 행렬식에 대한 제약이 없어 더 자유로운 구조 설계 가능[1]

---

### 2. 문제 정의, 제안 방법 및 모델 구조

#### 2.1 해결하고자 하는 문제

기존의 다변량 시계열 예측 방법들은 다음과 같은 한계를 가지고 있습니다.[1]

- **분포 클래스 제약**: 추적 가능(tractable)한 분포 클래스에만 의존하거나 저차원 근사(low-rank approximation)를 사용하여 진정한 데이터 분포를 학습할 수 없음
- **공분산 행렬의 계산 복잡성**: 전체 다변량 가우시안을 모델링하려면 매개변수가 $$O(D^2)$$으로 증가하고 손실 계산이 $$O(D^3)$$이 되어 실용적이지 않음[1]
- **통계적 의존성 표현의 한계**: 저차원 근사(Vec-LSTM 등)는 2차 효과(second-order effects)만 포착 가능
- **생성 모델의 문제**: 정규화 흐름은 연속 변환으로 인해 비연결 모드(disconnected modes)를 모델링하기 어렵고, VAE는 연속 공간을 비연결 공간으로 매핑하기 어려움[1]

#### 2.2 제안 방법: TimeGrad의 구조

**기본 아이디어**: 다변량 시계열의 조건부 분포를 다음과 같이 인수분해합니다.[1]

$$q_X(x^0_{t_0:T}|x^0_{1:t_0-1}, c_{1:T}) = \prod_{t=t_0}^T q_X(x^0_t|x^0_{1:t-1}, c_{1:T})$$

여기서 각 인수는 조건부 소거 확산 모델(conditional denoising diffusion model)로 학습됩니다.[1]

**확산 확률 모델의 핵심 수식:**

확산 모델은 순방향 과정(forward process)과 역방향 과정(reverse process)으로 구성됩니다.[1]

순방향 과정(고정된 마르코프 연쇄):

$$q(x_n|x_{n-1}) = \mathcal{N}(x_n; \sqrt{1-\beta_n}x_{n-1}, \beta_n I)$$

여기서 $$\beta_1, ..., \beta_N$$은 증가하는 분산 스케줄입니다.[1]

임의의 노이즈 레벨에서의 샘플링:

$$q(x_n|x_0) = \mathcal{N}(x_n; \sqrt{\bar{\alpha}_n}x_0, (1-\bar{\alpha}_n)I)$$

여기서 $$\alpha_n = 1 - \beta_n$$, $$\bar{\alpha}\_n = \prod_{i=1}^n \alpha_i$$[1]

역방향 과정(학습 가능한 가우시안 전이):
$$p(x_{n-1}|x_n) = \mathcal{N}(x_{n-1}; \mu_\theta(x_n, n), \sigma_n I)$$

변분하한(variational lower bound):

$$L = \mathbb{E}\_{q(x_0,x_{1:N})} \left[\log p(x_N) + \sum_{n=1}^N \log \frac{p(x_{n-1}|x_n)}{q(x_n|x_{n-1})}\right]$$

이를 KL 발산으로 재표현하면:[1]

$$\mathcal{L} = \mathbb{E}_{x_0,\epsilon,n} \left[\left\|\epsilon - \epsilon_\theta(x_n, n, h_{t-1}, n)\right\|^2\right]$$

여기서 $$\epsilon$$는 표준 가우시안 노이즈, $$h_{t-1}$$은 RNN의 은닉 상태입니다.[1]

**시간 역학 모델링:**

RNN(LSTM 또는 GRU)을 사용하여 시간 동역학을 인코딩합니다.[1]

$$h_t = \text{RNN}(\text{concat}(x^0_t, c_t), h_{t-1})$$

최종 모델:
$$\prod_{t=t_0}^T p(x^0_t|h_{t-1})$$

#### 2.3 모델 아키텍처

**신경망 구조:**[1]

- **RNN 성분**: 2층 LSTM, 은닉 상태 차원 $$h_t \in \mathbb{R}^{40}$$
- **노이즈 인덱스 인코딩**: Transformer 기반 Fourier 위치 임베딩 ($$N_{\max}=500$$)을 $$\mathbb{R}^{32}$$으로 변환
- **이푼(denoising) 네트워크 $$\epsilon_\theta$$**:
  - 조건부 1D 팽창 합성곱(dilated convolution, WaveNet 및 DiffWave 개조)
  - 잔여 연결(residual connections)
  - 8개 블록 ($$i=0,...,7$$)
  - 각 블록의 팽창: $$\text{dilation}=2^{\lfloor i/2 \rfloor}$$
  - 채널: 8개 (마지막 제외)
  - 게이트 활성화 단위(gated activation unit): $$\tanh$$
  - 모든 8개 블록의 스킵 연결 출력 합산

**학습 설정:**[1]

- 최적화: Adam, 학습률 $$1 \times 10^{-3}$$
- 확산 단계: $$N=100$$
- 분산 스케줄: 선형, $$\beta_1 = 1 \times 10^{-4}$$부터 $$\beta_N = 0.1$$까지
- 배치 크기: 64
- 초기 멈춤(early stopping): 검증 세트 사용

#### 2.4 훈련 및 추론 알고리즘

**훈련 (Algorithm 1):**[1]

각 시간 단계 $$t \in [t_0, T]$$에 대해:
1. 노이즈 인덱스 $$n$$ 균등 분포로 샘플링
2. 노이즈 $$\epsilon \sim \mathcal{N}(0, I)$$ 샘플링
3. 손실 $$\|\epsilon - \epsilon_\theta(x_n^t, n, h_{t-1})\|^2$$에 대해 경사 하강

**추론 (Algorithm 2) - 어닐링 Langevin 역학:**[1]

$$n=N$$에서 $$n=1$$까지:
$$x_t^{n-1} = \frac{1}{\sqrt{\alpha_n}}x_t^n - \frac{1-\alpha_n}{\sqrt{\alpha_n(1-\bar{\alpha}_n)}} \epsilon_\theta(x_t^n, h_{t-1}, n) + \sigma_n z$$

여기서 $$z \sim \mathcal{N}(0, I)$$ (단, $$n=1$$일 때 $$z=0$$)[1]

#### 2.5 스케일링 및 공변수

**스케일 정규화:**[1]

각 시계열 개체를 $$\text{context window mean}$$ 또는 1로 정규화하여 모델의 학습을 단순화합니다.

**공변수:**[1]

- 시간 종속: 요일, 시간 등
- 시간 독립: 범주형 특성 임베딩
- 지연 특성(lag features)

***

### 3. 성능 향상 및 일반화

#### 3.1 실험 설정 및 평가 지표

**평가 지표: CRPS (Continuous Ranked Probability Score)**[1]

$$\text{CRPS}(F, x) = \int_{-\infty}^{\infty} (F(z) - \mathbb{I}_{x \leq z})^2 dz$$

여기서 $$F$$는 누적분포함수(CDF), $$\mathbb{I}$$는 지시함수(indicator function)입니다.[1]

경험적 CDF를 사용하여 샘플에서 직접 계산 가능:
$$F(z) = \frac{1}{S}\sum_{s=1}^S \mathbb{I}_{X_s \leq z}$$

**CRPSsum**: 모든 $$D$$개 시계열 차원의 합에 대한 CRPS[1]

**데이터셋:**[1]

| 데이터셋 | 차원 | 도메인 | 빈도 | 시간 단계 | 예측 길이 |
|---------|------|--------|------|----------|----------|
| Exchange | 8 | 실수 | 일일 | 6,071 | 30 |
| Solar | 137 | 실수 | 시간 | 7,009 | 24 |
| Electricity | 370 | 실수 | 시간 | 5,833 | 24 |
| Traffic | 963 | [1] | 시간 | 4,001 | 24 |
| Taxi | 1,214 | 카운트 | 30분 | 1,488 | 24 |
| Wikipedia | 2,000 | 카운트 | 일일 | 792 | 30 |

#### 3.2 성능 비교 결과

**Table 2: 테스트 세트 CRPSsum 비교 (낮을수록 좋음)**[1]

TimeGrad는 Wikipedia를 제외한 모든 데이터셋에서 최고 성능을 달성했습니다:[1]

| 방법 | Exchange | Solar | Electricity | Traffic | Taxi | Wikipedia |
|------|----------|-------|-------------|---------|------|-----------|
| VAR | 0.005 | 0.83 | 0.039 | 0.29 | - | - |
| Vec-LSTM ind-scaling | 0.008 | 0.391 | 0.025 | 0.087 | 0.506 | 0.133 |
| Vec-LSTM lowrank-Copula | 0.007 | 0.319 | 0.064 | 0.103 | 0.326 | 0.241 |
| GP Copula | 0.007 | 0.337 | 0.02450 | 0.078 | 0.208 | 0.086 |
| Transformer MAF | 0.005 | 0.301 | 0.02070 | 0.056 | 0.179 | 0.063 |
| **TimeGrad** | **0.006** | **0.287** | **0.0206** | **0.044** | **0.114** | **0.0485** |

특히 고차원 데이터셋(Traffic: 963차원, Taxi: 1,214차원)에서 현저한 성능 향상을 보였습니다.[1]

#### 3.3 성능 향상의 메커니즘

**1. EBM의 유연성:** 정규화 흐름과 VAE는 연속 변환 또는 연속-이산 매핑의 제약이 있지만, EBM은 이러한 제약이 없어 다양한 분포 형태를 학습할 수 있습니다.[1]

**2. 다중 노이즈 스케일:** 확산 모델은 여러 노이즈 스케일에서 학습함으로써 거시적(coarse) 및 미시적(fine-grained) 데이터 특성을 모두 포착합니다.[1]

**3. 조건부 학습:** RNN을 통한 컨텍스트 정보의 효율적인 인코딩으로 시간 역학을 정확히 모델링합니다.

**4. 스케일 정규화:** 데이터의 크기 변동이 큰 실제 데이터에서 각 개체를 정규화함으로써 학습 안정성 개선.[1]

#### 3.4 하이퍼파라미터 분석 (Ablation Study)

**확산 단계 수 $$N$$의 영향:**[1]

논문에서는 Electricity 데이터셋을 대상으로 $$N \in \{2, 4, 8, ..., 256\}$$에 대해 검증했습니다.[1]

- $$N=10$$에서 이미 좋은 성능 ($$\sim 0.021$$ CRPS)
- **$$N=100$$에서 최적 성능** ($$\sim 0.0206$$ CRPS)
- $$N > 100$$은 성능 개선이 미미하거나 악화

**해석:** 더 많은 단계는 역방향 과정을 더 가우시안에 가깝게 근사하지만, 학습 안정성과 계산 비용의 트레이드오프를 고려하면 $$N=100$$이 최적입니다.[1]

***

### 4. 모델의 한계

#### 4.1 주요 한계

**1. 샘플링 속도:**[1]

- 추론 시 역방향 과정을 $$N$$번(논문에서는 $$N=100$$) 순회해야 함
- 정규화 흐름의 여러 이중 변환(bijection stacks) 순회와는 다르지만, 여전히 계산 비용이 높음
- $$S=100$$개 샘플을 생성하려면 총 10,000번의 네트워크 평가 필요

**2. 이산 데이터 모델링:**[1]

- 현재 구현은 실수 값 데이터에 최적화
- 이산 데이터(카운트 등)의 경우 정규화 흐름처럼 dequantization(균일 노이즈 추가) 필요
- 논문에서는 "dequantization 불필요"라고 주장하지만, 실제 구현에서는 이 문제가 제약

**3. 장시간 시계열:**[1]

- RNN 구조의 한계로 인한 장기 의존성 문제
- 매우 긴 컨텍스트 윈도우에서 성능 저하 가능성

**4. 메모리 효율성:**[1]

- 네트워크 아키텍처가 복잡하여 GPU 메모리 사용량이 높을 수 있음
- 실험에서는 V100 GPU(16GB)로 충분했지만, 초대형 데이터셋에는 확장성 문제 가능

***

### 5. 일반화 성능과 향상 가능성

#### 5.1 현재 일반화 성능의 강점

**1. 다양한 데이터셋에서의 일관된 성능:**[1]

- Exchange(8차원) → Wikipedia(2,000차원): 전 범위에서 경쟁력 있는 성능
- 카운트 데이터(Taxi, Wikipedia)와 연속 데이터(Solar, Electricity) 모두 잘 처리
- 데이터 크기(792 ~ 7,009 시간 단계)에 관계없이 안정적 성능

**2. 구조적 일반화:**[1]

- EBM의 제약 없는 함수 근사기로 인해 다양한 분포 형태 학습 가능
- 자동 특성 추출로 수동 특성 엔지니어링 불필요

**3. 불확실성 정량화:**[1]

- $$S=100$$개 샘플로부터 경험적 분위수 계산
- 예측 구간의 신뢰도 높음

#### 5.2 일반화 성능 향상을 위한 제안

**1. 아키텍처 개선:**[1]

**Transformer 적용**: 장시간 시계열에서 주의 메커니즘으로 장기 의존성 포착
```
한계: "For long time sequences, one could replace the RNN with a 
Transformer architecture Rasul et al., 2021 to provide better 
conditioning for the EBM emission head."[1]
```

**2. 그래프 신경망(GNN) 통합:**[1]

엔티티(시계열 차원) 간의 관계가 알려진 경우, GNN으로 구조적 편향(inductive bias) 인코딩
```
"incorporating architectural choices that best encode the inductive 
bias of the problem being tackled, for example with graph neural 
networks Niu et al., 2020 when the relationships between entities 
are known."[1]
```

**3. 빠른 샘플링 기법:**[1]

- **개선된 분산 스케줄 + L1 손실**: 더 적은 단계로 샘플링 가능 (예: Chen et al., 2021)
- **비-마르코프 확산 과정**: Song et al., 2021의 일반화된 접근으로 더 빠른 샘플링

**4. 이산 데이터 모델링:**[1]

정규화 흐름의 dequantization 대신 EBM의 직접적 이산 분포 모델링 가능성

**5. 특이치 탐지(Anomaly Detection):**[1]

EBM의 우수한 out-of-distribution(OOD) 탐지 능력 활용
```
"EBMs exhibit better out-of-distribution OOD detection than other 
likelihood models. Such a task requires models to have a high 
likelihood on the data manifold and low at all other locations."[1]
```

***

### 6. 향후 연구 영향 및 고려사항

#### 6.1 학계 및 산업에 미치는 영향

**1. 다변량 시계열 예측의 새로운 패러다임:**

확산 모델과 EBM의 결합이 시계열 분야에서의 실용적 적용 가능성을 입증함으로써, 확산 모델 활용의 새로운 방향 제시

**2. 확률적 예측의 표준화:**

CRPS를 평가 지표로 사용하면서 단순 점 예측(point forecast) 대신 불확실성 정량화를 강조하는 추세 강화

**3. 생성 모델의 유연성 증명:**

정규화 흐름, VAE의 구조적 제약 없이도 높은 차원의 복잡한 분포를 모델링할 수 있음을 보여줌

#### 6.2 향후 연구 시 고려할 점

**1. 계산 효율성:**

샘플링 속도 개선이 실제 배포를 위한 핵심 과제. 분산 스케줄 최적화, 적응적 단계 수 선택, 지식 증류(knowledge distillation) 등 검토 필요

**2. 설명 가능성:**

EBM과 확산 모델의 "블랙박스" 특성으로 인해 예측의 해석성 낮음. 특성 중요도, 주의 가시화 등 필요

**3. 데이터 효율성:**

작은 데이터셋(예: Wikipedia는 792 시간 단계)에서 성능이 하락하는 경향. 사전 학습(pre-training), 전이 학습(transfer learning), 데이터 증강 기법 검토

**4. 실시간 예측:**

많은 실제 응용에서 온라인 예측이 필요하지만, 현재 방법은 배치 처리 중심. 온라인 학습 또는 슬라이딩 윈도우 적응 필요

**5. 멀티태스크 학습:**

여러 관련 시계열 데이터셋에서 동시에 학습하여 일반화 성능 향상 가능성 탐색

**6. 외부 변수(Exogenous Variables):**

기후, 경제 지표 등 외부 정보의 효율적 통합 메커니즘 개발

**7. 이상 탐지와의 통합:**

예측 분포와 실제 값의 괴리를 바탕으로 한 이상 탐지 시스템 구축

***

### 결론

**TimeGrad**는 확산 확률 모델과 자기회귀 구조를 결합하여 다변량 시계열의 전체 확률 분포를 정확히 학습하는 획기적인 접근법입니다. 에너지 기반 모델의 유연성으로 기존 방법들의 구조적 제약을 극복하면서도 상태-of-더-아트 성능을 달성했습니다.[1]

다만 **샘플링 속도, 장시간 시계열 처리, 작은 데이터셋 성능** 등이 개선 과제이며, 향후 **Transformer 기반 아키텍처, GNN 통합, 고속 샘플링 기법, 이상 탐지 적용** 등의 방향으로 연구가 진행될 것으로 예상됩니다. 특히 산업 응용 관점에서는 **계산 효율성 개선과 해석 가능성 강화**가 실제 배포의 관건이 될 것입니다.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/14991573-a4b5-4ec1-9371-77fb28da7419/2101.12072v2.pdf)
