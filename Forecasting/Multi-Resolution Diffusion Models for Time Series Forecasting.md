# Multi-Resolution Diffusion Models for Time Series Forecasting

### 1. 핵심 주장과 주요 기여

**Multi-Resolution Diffusion Models (mr-Diff)** 논문의 핵심 주장은 다음과 같습니다.[1]

시계열 데이터가 **다중 시간 규모(multiple temporal scales)에서 서로 다른 패턴**을 보인다는 점을 활용하면, 기존의 직접적 역노이징(direct denoising) 방식보다 훨씬 효과적인 시계열 예측이 가능하다는 것입니다. 기존 확산 모델 기반 시계열 방법들(TimeGrad, CSDI, TimeDiff 등)은 무작위 벡터에서 직접 역노이징을 수행하지만, mr-Diff는 **계절-추세 분해(seasonal-trend decomposition)**를 활용하여 역노이징 목표를 여러 개의 하위 목표로 분해합니다.[1]

**주요 기여:**

1. 시계열 확산 모델에 계절-추세 분해 기반 다중 해상도 분석을 **최초로 통합**했습니다.[1]
2. **쉬운-어려운(easy-to-hard) 방식의 점진적 역노이징**을 수행하여, 먼저 대략적인 추세를 생성한 후 단계적으로 세부 사항을 추가합니다.[1]
3. 9개의 실제 시계열 데이터셋에서 기존의 최첨단 확산 모델을 능가하고, 다양한 고급 시계열 예측 모델과 비교해도 경쟁력 있는 성능을 보입니다.[1]

***

### 2. 해결하고자 하는 문제, 제안 방법, 모델 구조

#### 2.1 문제 정의

**기존 시계열 확산 모델의 한계:**[1]

- 비정상(non-stationary)이고 잡음이 많은 실제 시계열에서 무작위 벡터로부터 직접 역노이징하는 것이 어렵습니다.
- 시계열의 고유한 **다중 규모 시간 구조**를 충분히 활용하지 못합니다.
- 계절 성분과 추세 성분이 섞여있어 효율적인 학습이 제약됩니다.

#### 2.2 제안 방법 (수식 포함)

**기본 확산 모델 기초:**[1]

표준 확산 모델에서는 전진 과정(forward process)이 다음과 같이 정의됩니다:

$$q(x_k|x_{k-1}) = \mathcal{N}(x_k; \sqrt{1-\beta_k}x_{k-1}, \beta_k\mathbb{I})$$

이는 다음과 같이 재정의할 수 있습니다:

$$x_k = \sqrt{\bar{\alpha}_k}x_0 + \sqrt{1-\bar{\alpha}_k}\epsilon, \quad k = 1, \ldots, K$$

여기서 $\bar{\alpha}\_k = \prod_{s=1}^{k}\alpha_s$, $\alpha_k = 1 - \beta_k$, $\epsilon \sim \mathcal{N}(0, \mathbb{I})$입니다.[1]

역노이징 과정(backward denoising)에서는 데이터 예측 전략을 사용합니다:

$$\mu_{\theta_s}(Y_s^k, k|c_s) = \frac{\sqrt{\alpha_k}(1-\bar{\alpha}_{k-1})}{1-\bar{\alpha}_k}Y_s^k + \frac{\sqrt{\bar{\alpha}_{k-1}}\beta_k}{1-\bar{\alpha}_k}Y_{\theta_s}(Y_s^k, k|c_s)$$

손실 함수는 다음과 같이 정의됩니다:

$$\min_{\theta_s} L_s(\theta_s) = \min_{\theta_s} \mathbb{E}_{Y_s^0 \sim q(Y_s), \epsilon \sim \mathcal{N}(0,\mathbb{I}), k}\left[\|Y_s^0 - Y_{\theta_s}(Y_s^k, k|c_s)\|^2\right]$$

**추세 추출(Trend Extraction):**[1]

다중 해상도 구조를 구성하기 위해 평균 풀링(average pooling)을 사용하여 순차적으로 추세를 추출합니다:

$$X_s = \text{AvgPool}(\text{Padding}(X_{s-1}), \tau_s), \quad s = 1, \ldots, S-1$$

여기서 $X_0 = X$이고, $\tau_s$는 평활화 커널 크기입니다. $s$가 증가하면서 추세가 더 대략적이 됩니다.[1]

**조건화 네트워크(Conditioning Network):**[1]

훈련 과정에서 미래 혼합(future mixup)을 사용합니다:

$$z_{\text{mix}} = m \odot z_{\text{history}} + (1-m) \odot Y_s^0$$

여기서 $\odot$는 Hadamard 곱(원소별 곱)이고, $m \in [0,1)^{d \times H}$는 균등 분포에서 표본화된 혼합 행렬입니다.[1]

조건 $c_s$는 다음과 같이 구성됩니다:

- $s < S-1$일 때: $z_{\text{mix}}$ 와 $\(Y_{ss+1}^{0}\)$ (더 대략적인 추세)을 채널 방향으로 연결
- $s = S-1$일 때: $c_s = z_{\text{mix}}$

**역노이징 스텝 임베딩(Diffusion Step Embedding):**[1]

확산 스텝 $k$의 임베딩은 정현파 위치 임베딩(sinusoidal positional embedding)을 통해 얻어집니다:

$$p_k = \text{SiLU}(\text{FC}(\text{SiLU}(\text{FC}(k_{\text{embedding}}))))$$

#### 2.3 모델 구조

**계층적 구조 (S=5 단계):**[1]

mr-Diff는 **5개 단계의 계단식 구조**로 구성됩니다:

1. **Stage 1**: 가장 세밀한(fine-grained) 추세 $Y_0$ 복원
2. **Stage 2-4**: 중간 해상도 추세 $Y_1, Y_2, Y_3$ 생성  
3. **Stage 5**: 가장 대략적인(coarse-grained) 추세 $Y_4$ 생성

각 단계에서는 독립적인 역노이징 네트워크를 가지고 있으며, 이전 단계의 출력이 다음 단계의 조건 입력으로 사용됩니다.[1]

**역노이징 네트워크 구조:**[1]

1. **입력 투영 블록**: $Y_s^k$를 임베딩 $\bar{z}^k \in \mathbb{R}^{d' \times H}$로 매핑
2. **인코더**: $\bar{z}^k$와 확산 스텝 임베딩 $p_k \in \mathbb{R}^{d' \times 1}$를 입력으로 받아 표현 $z^k \in \mathbb{R}^{d'' \times H}$ 생성
3. **연결(Concatenation)**: $z^k$와 조건 $c_s$를 변수 차원으로 연결하여 크기 $(2d + d'') \times H$의 텐서 생성
4. **디코더**: 이 텐서를 처리하여 $Y_{\theta_s}(Y_s^k, k|c_s)$ 출력

***

### 3. 성능 향상 및 한계

#### 3.1 성능 향상

**단변량 시계열 예측 (Table 1):**[1]

- 9개 데이터셋 중 **5개에서 최고 성능** 달성
- 나머지 4개 데이터셋에서는 3개가 2등, 1개가 후순위
- **평균 순위: 1.7** (모든 기준 중 최고)
- 다른 확산 모델들 (TimeDiff: 4.0, TimeGrad: 16.6, CSDI: 15.1) 대비 우수

특히 복잡한 추세 정보를 가진 데이터셋(Exchange, ETTh1)에서 두드러진 개선을 보였습니다.[1]

**다변량 시계열 예측 (Table 2):**[1]

- 9개 데이터셋 중 **5개에서 상위 2위 달성**
- **평균 순위: 2.6** (전체 최고)
- N-Hits (5.1), FiLM (6.1), SCINet (5.8) 등 다중 해상도 정보를 활용하는 강력한 기준 모델들도 능가

**추론 효율성 (Table 3):**[1]

- 학습 가능한 매개변수: **0.9M-2.3M** (TimeDiff: 1.7M, CSDI: 10M, SSSD: 32M)
- 추론 시간 (H=168): 
  - mr-Diff (S=3): **14.3ms** 
  - TimeDiff: 17.3ms
  - TimeGrad: 1,620.9ms
  - CSDI: 128.3ms
  - SSSD: 590.2ms

#### 3.2 한계

**제한 사항:**[1]

1. **복잡한 추세 정보 부재 시**: Caiso 데이터셋처럼 명확한 추세 구조가 없는 경우 성능 개선이 제한적입니다. mr-Diff는 추세 구조를 활용하는 방식이므로, 이러한 구조가 약한 데이터셋에서는 효과가 감소합니다.

2. **단계 수(S) 설정**: 적절한 단계 수는 데이터셋에 따라 다르며, 그리드 서치가 필요합니다. 과도한 단계는 계산 비용을 증가시킵니다.

3. **일반화 성능 제약**: 
   - 특정 데이터셋(예: Weather)에서 DLinear나 NLinear 같은 단순 모델에 비해 성능이 낮을 수 있습니다.
   - 학습 데이터의 특성에 따라 성능 변동이 있습니다.

4. **선형 예측 기준선 강화**: 최근 NLinear(3.3), DLinear(9.3) 등 단순 선형 기준선들이 강해져서, 이들을 능가하기 위해서는 추가 개선이 필요합니다.

***

### 4. 모델 일반화 성능 향상 가능성

#### 4.1 일반화 성능의 강점

**다중 스케일 구조의 이점:**

다중 해상도 분석을 통한 일반화 성능 향상 메커니즘:[2][3]

1. **특성 추상화(Feature Abstraction)**: 먼저 대략적 추세를 학습하여 스케일 불변 표현을 획득한 후, 세부 사항을 추가합니다. 이는 **도메인 간 일반화**에 유리합니다.[3]

2. **점진적 정교화(Progressive Refinement)**: 쉬운 작업부터 어려운 작업으로 진행하는 커리큘럼 학습(curriculum learning) 효과가 있어 **과적합 감소**에 도움됩니다.[1]

3. **노이즈에 대한 견고성**: 계절-추세 분해는 시계열의 주요 구조를 강조하므로, 입력 노이즈에 덜 민감합니다.

**실험 증거:**

- Table 7: 5회 반복 실행에서 **최대 표준편차 0.0042**로 매우 안정적입니다.
- 다양한 데이터셋(주기성, 비정상성, 샘플링 레이트)에서 일관된 성능을 보입니다.

#### 4.2 전이 학습(Transfer Learning) 가능성

**미충분 연구 영역 - 향후 기회:**[4][2]

1. **도메인 적응**: 기존 도메인 적응 시계열 예측 연구(예: DAF, DATSING)에서처럼, mr-Diff의 다중 해상도 표현을 도메인 불변 특성으로 활용하면 **교차 도메인 일반화**를 개선할 수 있습니다.[5][6]

2. **사전 학습(Pre-training)**: 최근 연구들(UTSD, TimeDiT, GPD)은 확산 모델 기반 시계열 기초 모델 개발을 시도하고 있습니다. mr-Diff의 구조는 **영역 전반의 사전 학습**에 적합할 수 있습니다.[7][8][9][4]

3. **데이터 부족 시나리오**: 시계열 데이터가 제한적인 상황에서 합성 데이터 생성을 통한 데이터 증강이 가능합니다.

#### 4.3 일반화 성능 향상의 잠재 제약

1. **데이터 분포 편이(Distribution Shift)**: 시계열의 통계적 특성(평균, 분산)이 크게 변하면, 학습된 추세 분해 패턴이 적용되기 어렵습니다. 인스턴스 정규화를 사용하지만 근본적 해결은 미흡합니다.

2. **이질적 데이터 유형**: 다양한 주기성, 추세 강도를 가진 데이터셋들이 혼재되면 단일 모델로 일반화가 어렵습니다.

***

### 5. 최근 연구 동향 (2020년 이후)

#### 5.1 다중 해상도/계층적 접근

**Scaleformer (2023):** 트랜스포머 기반으로 대략적 수준에서 세밀한 수준으로 점진적으로 예측을 생성합니다. 그러나 모든 해상도에서 단일 모듈을 사용하므로 mr-Diff의 단계별 독립 네트워크보다 유연성이 떨어집니다.[2]

**MG-TSD (2024):** 다중 입도(multi-granularity) 확산 모델로 중간 확산 스텝에 다양한 입도의 목표를 설정하여 학습을 안내합니다. mr-Diff와 유사한 개념이지만 구현 방식이 다릅니다.[3]

**Coherent Probabilistic Forecasting (2023):** 시간 계층 구조에서 일관된 확률적 예측을 수행합니다. mr-Diff보다 엄격한 일관성 요구사항이 있어 유연성이 감소합니다.[10]

#### 5.2 조건화 메커니즘 개선

**TimeDiff (2023):** 미래 혼합(future mixup)과 자기회귀 초기화를 도입했으며, mr-Diff가 이를 계승합니다.[1]

**RATD (2024):** 검색 증강(retrieval-augmented) 방식으로 데이터베이스에서 관련 샘플을 검색하여 조건으로 사용합니다. 불안정한 확산 모델 성능을 개선하는 데 중점을 두고 있습니다.[11]

**Diffusion-TS (2024):** 분해 기법과 트랜스포머를 결합하여 해석 가능성을 강화합니다.[12]

#### 5.3 기초 모델 및 사전 학습 패러다임

**Generative Pre-Trained Diffusion (GPD, 2024):** 무조건 확산 모델을 사전 학습하여 영역 간 일반화를 개선하려는 시도입니다.[4]

**TimeDiT (2024-2025):** 디퓨전 트랜스포머로 기초 모델의 역할을 목표로 하며, 영역 특화 작업에 미세 조정할 수 있습니다.[7]

**UTSD (2024):** 통합 확산 모델로 다양한 데이터 영역에서 사전 학습되어 영역별 모델을 능가합니다.[9]

#### 5.4 연속 확산 및 흐름 기반 방법

**Flow Matching with Gaussian Process Priors (TSFlow, 2024):** 전통적 고정 사전 분포 대신 가우스 프로세스 기반 조건부 사전을 사용하여 데이터 분포와 더 잘 정렬됩니다.[13]

**Auto-Regressive Moving Diffusion (ARMD, 2024):** 시계열의 연속적 순차 특성을 더 잘 반영하기 위해 ARMA 이론에 영감을 받아 설계되었습니다.[14]

**Series-to-Series Diffusion Bridge Model (S²DBM, 2024):** 브라운 브릿지 프로세스를 활용하여 역추정에서 무작위성을 감소시킵니다.[15]

#### 5.5 구조 시계열 모델링

**Diffusion with Temporal Correlation (2025):** 시계열의 내재적 시간 의존성을 더 잘 보존하는 분해 프레임워크를 제안합니다.[16]

**Hierarchical Time Series Forecasting (2025):** 계층적 시계열에서 대략적 수준과 세밀한 수준에 일관되게 예측하는 구조를 제안합니다.[17]

**LLM-Enhanced Time Series Forecasting:** 대규모 언어 모델과 확산을 결합하여 구조적 정보를 더 효과적으로 활용합니다.[18]

***

### 6. 향후 연구에 미치는 영향 및 고려 사항

#### 6.1 학술적 영향

**확산 모델 패러다임 확장:**

mr-Diff는 확산 모델을 시계열에 적용할 때 단순 직접 역노이징이 아닌 **구조화된 계층적 접근**이 효과적임을 입증했습니다. 이는 다른 구조화된 데이터(그래프 시계열, 공간-시간 데이터 등)에도 적용할 수 있는 원칙을 제시합니다.[19][20][1]

**다중 해상도 분석의 부활:**

NBeats, Autoformer 등에서 성공한 다중 해상도 접근이 생성 모델에도 효과적임을 보여, **구성적 접근(compositional approaches)**의 가치를 재확인했습니다.[1]

#### 6.2 실무적 응용 가능성

1. **재무 시계열 예측**: Diffolio 등 재무 응용 분야에서 활용 가능합니다.

2. **에너지 수요 예측**: 전력 부하 예측에서 요일, 계절성, 장기 추세를 구분하여 더 정확한 예측이 가능합니다.

3. **교통 흐름 예측**: 시간별, 일별, 주별 패턴을 계층적으로 모델링하여 개선된 예측을 제공합니다.

#### 6.3 향후 연구 시 고려할 주요 점

**1. 도메인 적응 및 전이 학습**[2][4][7]

- 기존 도메인 적응 기법(Domain Adversarial Training)과 mr-Diff의 다중 해상도 표현을 결합하여 **교차 도메인 일반화** 능력을 강화할 필요가 있습니다.
- 소규모 데이터셋에 대한 사전 학습 모델 활용 방안을 체계적으로 연구해야 합니다.

**2. 이질적 데이터 처리**[21][22]

- 서로 다른 주기성, 추세 강도, 노이즈 수준을 가진 데이터들에 대한 **적응적 단계 수 선택 메커니즘** 개발이 필요합니다.
- 메타 러닝(meta-learning) 기법으로 데이터셋 특성에 맞는 모델 구성을 자동화할 수 있습니다.

**3. 이론적 이해 강화**[23]

- 다중 해상도 확산이 왜 일반화 성능을 개선하는지에 대한 **이론적 분석**이 부족합니다.
- 근사 능력(approximation capability), 일반화 경계(generalization bounds)에 대한 정형적 분석이 필요합니다.

**4. 불확실성 정량화**[24]

- 확산 모델의 강점인 **확률적 예측 능력**을 더 잘 활용하여 예측 구간(prediction intervals)의 정확성을 개선해야 합니다.
- 편향된 불확실성 추정 문제를 해결해야 합니다.

**5. 장기 예측 개선**[14][16]

- 현재 mr-Diff는 중기 예측(168시간 이내)에서 우수하지만, 매우 장기 예측(예: 1년)에서의 성능은 미흡합니다.
- 경향 분해와 장기 의존성 모델링의 균형을 맞춰야 합니다.

**6. 계산 효율성 최적화**

- S=5일 때 추론 시간이 43.6ms에 달해 실시간 응용에 제약이 있습니다.
- 가속화 기법(예: DPM-Solver 확장) 연구가 필요합니다.

**7. 해석 가능성 강화**[12]

- 각 단계에서 생성된 추세의 의미를 시각화하여 모델 결정 과정을 이해하기 쉽게 할 필요가 있습니다.
- 실무진이 신뢰할 수 있는 설명 가능한 예측이 중요합니다.

#### 6.4 기술적 혁신 방향

**계층적 사전 학습:**

영역별 데이터로 사전 학습된 mr-Diff 기초 모델을 개발하면, 새로운 응용 분야에서 빠른 미세 조정으로 효과적인 모델을 얻을 수 있습니다. 이는 최근 타이밍 트렌드와도 맞아떨어집니다.[9][4][7]

**적응적 해상도 선택:**

데이터의 통계적 특성(자기상관, 주파수 성분)으로부터 자동으로 최적 단계 수를 결정하는 메커니즘 개발이 필요합니다.

**하이브리드 접근:**

mr-Diff와 시간 상관성 보존 모델, 또는 대형 언어 모델 기반 방법을 결합하여 더 강력한 시계열 예측 시스템을 구축할 수 있습니다.[18][16]

***

### 결론

**Multi-Resolution Diffusion Models (mr-Diff)** 논문은 확산 모델을 시계열 예측에 적용할 때 **단순한 구조가 아닌 시계열의 고유 특성(다중 규모 패턴)을 활용하는 정교한 설계**의 중요성을 보여줍니다. 계절-추세 분해를 통한 계층적 역노이징은 이론적으로도 직관적이며, 실험적으로도 우수한 성능을 입증했습니다.

특히 **일반화 성능 측면**에서 mr-Diff의 다중 해상도 구조는 도메인 간 전이, 데이터 부족 상황, 불균형 데이터 처리 등에서 잠재적 우월성을 가지고 있습니다. 다만 이 잠재성을 완전히 활용하려면 도메인 적응, 이론적 분석, 해석 가능성 강화 등 향후 연구가 필수적입니다.

시계열 예측 분야가 **기초 모델 시대로 진입**하고 있는 만큼, mr-Diff의 구조가 대규모 사전 학습 기초 모델의 설계 원칙으로 채택될 가능성은 높습니다. 따라서 향후 연구는 이 모델 구조의 일반화 능력을 최대한 활용하는 방향으로 진행될 것으로 예상됩니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f8abe600-8bea-4f1b-b91d-98f506bd2155/4109_Multi_Resolution_Diffusio.pdf)
[2](https://arxiv.org/abs/2401.03006)
[3](https://arxiv.org/abs/2403.05751)
[4](https://arxiv.org/abs/2406.02212)
[5](https://dl.acm.org/doi/10.1145/3340531.3412155)
[6](https://arxiv.org/pdf/2102.06828.pdf)
[7](https://arxiv.org/pdf/2409.02322v1.pdf)
[8](https://arxiv.org/html/2503.01157v1)
[9](https://arxiv.org/html/2412.03068)
[10](https://proceedings.mlr.press/v206/rangapuram23a/rangapuram23a.pdf)
[11](https://proceedings.neurips.cc/paper_files/paper/2024/file/053ee34c0971568bfa5c773015c10502-Paper-Conference.pdf)
[12](https://arxiv.org/abs/2403.01742)
[13](https://arxiv.org/abs/2410.03024)
[14](https://arxiv.org/abs/2412.09328)
[15](http://arxiv.org/pdf/2411.04491.pdf)
[16](https://www.ijcai.org/proceedings/2025/0749.pdf)
[17](https://www.arxiv.org/pdf/2506.19633.pdf)
[18](https://aclanthology.org/2025.findings-emnlp.58.pdf)
[19](https://www.nature.com/articles/s41598-025-28592-4)
[20](https://arxiv.org/html/2512.01572v1)
[21](https://www.semanticscholar.org/paper/463a808105cbbcf2eb0395ddc03a61dfbbc593fd)
[22](https://arxiv.org/abs/2405.00946)
[23](https://arxiv.org/pdf/2503.14076.pdf)
[24](https://papers.nips.cc/paper_files/paper/2022/file/91a85f3fb8f570e6be52b333b5ab017a-Paper-Conference.pdf)
[25](https://www.semanticscholar.org/paper/0cb94863249f65c45e2f0129aa1bb574eedf1f5e)
[26](https://www.semanticscholar.org/paper/f5cc95fae2ff9ea1f1a2d30be26acccf3e448803)
[27](https://arxiv.org/abs/2410.18712)
[28](https://ieeexplore.ieee.org/document/10733342/)
[29](https://ieeexplore.ieee.org/document/10808510/)
[30](https://ieeexplore.ieee.org/document/10899271/)
[31](https://arxiv.org/pdf/2307.11494.pdf)
[32](http://arxiv.org/pdf/2406.02827.pdf)
[33](http://arxiv.org/pdf/2406.02212.pdf)
[34](http://arxiv.org/pdf/2410.18712.pdf)
[35](https://arxiv.org/pdf/2305.00624.pdf)
[36](https://arxiv.org/pdf/2412.09328.pdf)
[37](https://arxiv.org/abs/2503.00951)
[38](https://www.sciencedirect.com/science/article/abs/pii/S0957417425035596)
[39](https://iclr.cc/media/iclr-2024/Slides/17883_mrXtGgm.pdf)
[40](https://dmqa.korea.ac.kr/uploads/seminar/Feature-centric%20Diffusion%20Models%20for%20Time%20Series%20Forecasting.pdf)
[41](https://www.sciencedirect.com/science/article/abs/pii/S0957417423017049)
[42](https://arxiv.org/html/2507.14507)
[43](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2025.1540912/full)
[44](https://arxiv.org/abs/2307.11494)
[45](http://arxiv.org/pdf/2410.03024.pdf)
[46](https://www.nature.com/articles/s41467-020-20398-4)
[47](https://openreview.net/pdf/1b5cacee607a2ff11b7c8092001614d82f68ee33.pdf)
[48](https://www.arxiv.org/abs/2512.00293)
[49](https://arxiv.org/pdf/2307.00754.pdf)
[50](http://arxiv.org/pdf/2501.00910.pdf)
[51](https://arxiv.org/pdf/2408.14408.pdf)
[52](http://arxiv.org/pdf/2409.05399.pdf)
[53](https://arxiv.org/abs/2511.07014)
[54](https://www.vldb.org/pvldb/vol17/p359-zhang.pdf)
[55](https://seunghan96.github.io/ts/da/ts20/)
[56](https://proceedings.iclr.cc/paper_files/paper/2024/file/d64740dd69bcc90ba225a182984b81ba-Paper-Conference.pdf)
[57](https://proceedings.iclr.cc/paper_files/paper/2024/file/516a9317af9d89e9f2251bd7fde49b8f-Paper-Conference.pdf)
[58](https://openreview.net/forum?id=mmjnr0G8ZY)
