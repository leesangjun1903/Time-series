# An empirical survey of data augmentation for time series classification with neural networks

### I. 논문의 핵심 주장 및 기여

**"An empirical survey of data augmentation for time series classification with neural networks"** (Iwana & Uchida, 2021)의 핵심 주장은 명확합니다. 이미지 인식에서는 데이터 증강이 표준 관행이지만, 시계열 분류에서는 여전히 체계적이지 않다는 점입니다. 저자들은 시계열 데이터가 본질적으로 영상 데이터보다 규모가 작다는 실제 문제를 지적합니다—UCR 시계열 아카이브의 128개 데이터셋 중 단 12개만이 1,000개 이상의 학습 샘플을 보유하고 있습니다.[1]

논문의 가장 중요한 기여는 **네 가지 데이터 증강 계열을 체계적으로 분류하고, 128개 데이터셋 전체에서 12개의 서로 다른 기법을 6개 신경망 아키텍처로 평가**한 점입니다. 이는 기존 연구들이 제한된 데이터셋과 모델에서만 평가했던 것과 비교하여 질적으로 다른 수준의 포괄성을 제공합니다.

***

### II. 해결하는 문제, 제안 방법, 모델 구조

#### A. 해결하는 문제

시계열 분류는 다음 세 가지 근본적인 문제에 직면하고 있습니다:[1]

1. **데이터 부족 문제 (Data Scarcity)**: 많은 현실 응용에서 라벨링된 시계열 데이터를 대규모로 수집하기 어렵습니다. 특히 의료, 생체신호, 산업 센서 데이터에서 심각합니다.

2. **과적합 위험**: 데이터가 부족할 때 신경망 모델은 훈련 데이터의 노이즈를 학습하여 일반화 성능이 급격히 떨어집니다.

3. **클래스 불균형 (Class Imbalance)**: 많은 실제 데이터셋이 특정 클래스의 샘플이 극도로 부족한 상황입니다.

데이터 증강의 핵심 목표는 **모델 독립적인 방식으로 결정 경계를 확장하고 과적합을 줄임으로써 일반화 능력을 개선**하는 것입니다.[1]

#### B. 제안 방법의 분류 및 수식

논문은 데이터 증강을 네 가지 계열로 분류합니다:

**1) 변환 기반 방법 (Transformation-Based Methods)**

이는 세 가지 도메인으로 나뉩니다:

- **진폭 도메인 (Magnitude Domain)**

$$x' = x_1 + \epsilon_1, \ldots, x_t + \epsilon_t, \ldots, x_T + \epsilon_T$$

여기서 $\epsilon \sim \mathcal{N}(0, \sigma^2)$인 **지터링(Jittering)**입니다.[1]

스케일링은:
$$x' = \alpha x_1, \ldots, \alpha x_t, \ldots, \alpha x_T$$

여기서 $\alpha \sim \mathcal{N}(1, \sigma^2)$입니다.[1]

진폭 왜곡(Magnitude Warping)은:
$$x' = \alpha_1 x_1, \ldots, \alpha_t x_t, \ldots, \alpha_T x_T$$

여기서 $[\alpha_1, \ldots, \alpha_T]$는 3차 스플라인 $S(u)$로 보간된 수열입니다.[1]

- **시간 도메인 (Time Domain)**

슬라이싱(Slicing):
$$x' = x_\phi, \ldots, x_t, \ldots, x_{W+\phi}$$

여기서 $W$는 윈도우 크기, $\phi$는 무작위 정수입니다.[1]

시간 왜곡(Time Warping):
$$x' = x_{\tau(1)}, \ldots, x_{\tau(t)}, \ldots, x_{\tau(T)}$$

여기서 $\tau(\cdot)$는 3차 스플라인으로 정의된 매끄러운 왜곡 함수입니다.[1]

- **주파수 도메인 (Frequency Domain)**

음성 데이터에 특화된 성대 길이 섭동(VTLP):
$$f' = f\omega \quad \text{if } f \leq F_{hi}\min(\omega, 1)$$

$$f' = \frac{s}{2} - \frac{\frac{s}{2} - F_{hi}\min(\omega, 1)}{\omega} \quad \text{otherwise}$$

여기서 $\omega$는 무작위 왜곡 계수입니다.[1]

**2) 패턴 혼합 (Pattern Mixing)**

SMOTE (Synthetic Minority Over-sampling Technique):
$$x' = x + \lambda|x - x_{NN}|$$

여기서 $\lambda \in \{0, 1\}$, $x_{NN}$는 k-최근접 이웃입니다.[1]

가중 DTW 무게중심 평균(wDBA)는 동적 시간 왜곡(DTW)의 정렬 함수를 사용하여 여러 시계열의 "평균"을 계산하는 방식입니다.[1]

SPAWNER (SuboPtimAl Warped time series geNEratoR)는 부분 최적 시간 왜곡 경로를 강제하여 제약이 있는 DTW 정렬을 수행합니다.[1]

**3) 생성 모델 (Generative Models)**

- **통계 모델**: 혼합 가우시안 트리(Gaussian Mixture Trees), GRATIS (생성적 자동회귀 모델)
- **신경망 기반**: 
  - LSTM 기반 시퀀스-투-시퀀스 모델
  - LSTM 오토인코더 (VAE)
  - GAN 기반 모델 (T-CGAN, WaveGAN, cGAN)

**4) 분해 방법 (Decomposition Methods)**

- **경험적 모드 분해 (EMD)**: 비선형, 비정상 신호 분해
- **독립 성분 분석 (ICA)**: 혼합 신호의 독립 성분 분리
- **계절-추세 분해 (STL)**: 계절, 추세, 나머지 성분으로 분해 후 재합성

#### C. 평가 모델 구조

논문에서 사용된 6가지 신경망 아키텍처:[1]

| 모델 | 구조 | 특성 |
|------|------|------|
| **MLP** | 3개 숨겨진 레이어 (각 500 노드), ReLU, 드롭아웃 (0.1, 0.2, 0.3) | 시간 구조 미고려 |
| **1D VGG** | 적응 블록 수: $B = \text{round}(\log_2(T)) - 3$, 64-512 필터 | 계층적 특성 추출 |
| **1D ResNet** | 3개 잔차 블록, 각 블록 64-128 필터, 배치 정규화 | 깊은 네트워크 학습 |
| **LSTM** | 1개 LSTM 레이어 (100 유닛), Nadam 옵티마이저 | 순차 의존성 모델링 |
| **BLSTM** | 2개 양방향 LSTM 레이어 (100 유닛) | 양방향 시간 정보 |
| **LSTM-FCN** | LSTM 스트림 + 완전 합성곱 스트림, GAP | 하이브리드 아키텍처 |

***

### III. 성능 향상 및 한계 분석

#### A. 성능 향상 결과

논문의 실증적 평가는 명확한 패턴을 보여줍니다:[1]

**최고 성능 방법들:**

| 방법 | 최우수 모델 | 평균 개선율 | 강점 |
|------|-----------|-----------|------|
| **Window Warping** | VGG, ResNet, LSTM | 일관된 개선 | 빠른 연산, 모든 모델에서 효과적 |
| **Slicing** | VGG, ResNet | CNN 모델에서 우수 | 엔드포인트 정보 제거로 내부 특성 강조 |
| **DGW** | BLSTM | 최대 성능 향상 | 높은 정확도 (계산 시간 많음) |

**수식적 성능 지표:**

$$\Delta \text{Acc} = \text{Acc}_{\text{augmented}} - \text{Acc}_{\text{baseline}}$$

논문은 Spearman의 순위 상관계수를 사용하여 데이터셋 특성과 성능 향상의 관계를 분석했습니다:[1]

$$\rho_s = 1 - \frac{6\sum d_i^2}{n(n^2-1)}$$

여기서 $d_i$는 두 변수의 순위 차이입니다.

#### B. 데이터셋 특성과의 상관 관계

**1) 훈련 집합 크기와의 관계**

$$\text{Correlation}(\text{Training Size}, \Delta \text{Acc}) < 0$$

작은 데이터셋일수록 데이터 증강으로부터 더 큰 이득을 얻습니다. 이는 직관과 일치하며, ResNet과 LSTM-FCN에서 특히 강한 음의 상관을 보입니다.[1]

**2) 데이터셋 분산과의 관계**

$$\overline{\sigma}^2_{\text{DS}} = \frac{1}{T}\sum_{t=1}^{T} \sigma_t^2$$

분산이 높은 데이터셋(즉, 패턴의 변동성이 큼)은 데이터 증강으로부터 더 큰 이점을 얻습니다.[1]

**3) 클래스 내 분산**

$$\overline{\sigma}^2_{\text{IC}} = \frac{1}{N_C}\sum_{c=1}^{C}\sum_{t=1}^{T}\sigma_{t,c}^2$$

클래스 내 분산이 높을수록 증강의 효과가 더 강합니다.[1]

**4) 클래스 불균형도 (Imbalance Degree)**

$$\text{ID} = \frac{d(\zeta, \beta)}{d(\iota, \beta) + (C_m - 1)}$$

여기서 Hellinger 거리를 사용하여 클래스 불균형을 측정합니다. 진폭 도메인 증강이 불균형 데이터셋에서 더 효과적입니다.[1]

#### C. 계산 비용 분석

| 방법 | 평균 시간 | 복잡도 | 평가 |
|------|----------|------|------|
| 지터링, 스케일링, 회전 | <0.01초 | O(T) | 즉시 가능 |
| 진폭 왜곡, 슬라이싱 | 0.02-0.11초 | O(T) | 실용적 |
| SPAWNER, RGW | 67-71초 | O(T²) | 제한적 사용 |
| wDBA, DGW | 2,300-6,380초 | O(T²) | 짧은 시계열만 가능 |

DTW 기반 방법들의 계산 복잡도 때문에 긴 시계열에서는 변환 기반 방법이 필수적입니다.[1]

#### D. 주요 한계

1. **모델 간 일관성 부족**: LSTM-FCN과 MLP는 대부분의 증강에 부정적 반응을 보입니다. 이는 높은 드롭아웃(0.8)이나 아키텍처 특성 때문일 수 있습니다.[1]

2. **무작위 변환의 한계**: 과도한 변환은 클래스를 중복되도록 할 수 있습니다. 논문에서 보인 PCA 분석에 따르면, Time Warping, Permutation, Rotation은 클래스 경계를 모호하게 합니다.[1]

3. **도메인 특화성**: 특정 증강 기법이 특정 도메인 데이터(음성, 신호, 영상 기반)에서만 효과적입니다. 주파수 왜곡은 비주기적 시계열에 해로울 수 있습니다.[1]

4. **하이퍼파라미터 민감성**: wDBA의 가중치 스킴(AA, AS, ASD)이나 Time Warping의 노트 수(I)와 표준편차(σ) 같은 하이퍼파라미터는 신중한 튜닝을 요구합니다.[1]

***

### IV. 일반화 성능 향상에 대한 심층 분석

#### A. 메커니즘: 왜 데이터 증강이 일반화를 개선하는가?

논문은 **결정 경계 확장 (Decision Boundary Expansion)** 이론을 암묵적으로 검증합니다. 합성 샘플이 클래스 사이의 경계를 밀어내면, 모델이 더 강건한 특성을 학습합니다.[1]

정규화 관점에서 본다면:[1]
- **노이즈 추가 (지터링)**: Tikhonov 정규화의 암묵적 형태. 목적함수가:

$$\mathcal{L}_{\text{augmented}} = \mathcal{L}_{\text{original}} + \lambda \|\mathbf{w}\|^2$$
  
와 동등합니다.

- **패턴 혼합**: 클래스 내 다양성을 인위적으로 증가시켜 일반화 갭을 줄입니다.

- **생성 모델**: 실제 데이터 분포 $p(\mathbf{x})$를 근사하여, 진정한 OOD (Out-of-Distribution) 샘플이 아닌 IID 샘플을 생성합니다.

#### B. 기여 메커니즘 세분화

**1) 다양성 증가 (Diversity Enhancement)**

Window Warping이나 Slicing 같은 방법은 특성 공간에서 훈련 샘플을 "이동"시켜 클래스의 포함 영역을 확장합니다. 수학적으로:[1]

$$\text{Coverage}(\mathcal{C}) = \frac{|\bigcup_{(\mathbf{x}, \mathbf{x}') \in \text{Augmented}} B(\mathbf{x}, r)|}{|\mathcal{C}|}$$

여기서 $B(\mathbf{x}, r)$은 중심이 $\mathbf{x}$이고 반경이 $r$인 공입니다.

**2) 과적합 감소**

논문의 상관 분석(Fig. 5)에서 훈련 집합 크기가 작을수록 $\Delta \text{Acc}$가 크다는 것은:[1]
$$\mathbb{E}[\text{Generalization Gap}] = \mathbb{E}[\text{Test Error}] - \mathbb{E}[\text{Train Error}]$$
가 감소함을 의미합니다.

**3) 클래스 분리성 (Class Separability) 개선**

패턴 혼합 방법들 (wDBA, DGW)은 DTW를 사용하여 구조를 보존하면서 샘플을 생성합니다:[1]

$$\text{Separability} = \frac{\text{Inter-class distance}}{\text{Intra-class distance}}$$

wDBA가 완선형 평균보다 우수한 이유는 DTW가 위상 정렬(phase alignment)을 고려하기 때문입니다.

#### C. 일반화 성능 향상의 한계 (Generalization-Augmentation Trade-off)

흥미로운 발견은 **어떤 경우 증강이 해를 끼친다는 것**입니다:[1]

- **Rotation (플리핑)**: 영상 기반 시계열(Adiac, Fish 데이터셋)에서는 의미 있는 증강이 아닙니다. 이는:
$$\text{P}(\text{class}|\text{flipped sample}) \neq \text{P}(\text{class}|\text{original sample})$$
을 위반합니다.

- **Permutation**: 시간 의존성을 완전히 파괴합니다:
$$\tau(\mathbf{x}) = \sigma(\mathbf{x}) \quad \text{(무작위 순열 } \sigma\text{)}$$

이는 시계열의 본질인 순차 구조를 제거합니다.

- **Time Warping 과도**: 과도한 왜곡은 원본 클래스 자체를 변경할 수 있습니다. 임계값이 존재합니다:
$$\exists \gamma_{\max}: \text{for } \gamma > \gamma_{\max}, \quad \text{Accuracy} \downarrow$$

***

### V. 2020년 이후의 최신 연구 비교 분석

본 논문(2021) 이후의 급속한 발전을 정리하면:

#### A. 생성 모델의 혁신

**원본 논문의 한계**: GANs를 "외부 학습 필요" 카테고리로 제외했습니다.[1]

**최신 발전**:[2][3][4][5]
- **TimeGAN** (Yoon et al., 2019 → 2020년대 확대): LSTM 기반 생성으로 시계열의 시간적 특성 보존
- **Diffusion Models** (DiffAT, 2025): 확산 모델이 GAN의 학습 불안정성을 극복
  $$q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_0, \beta_t I)$$
  여기서 $t$는 시간 스텝입니다.

**성과**: 168개 예측 사례 중 156개에서 성능 향상[5]

#### B. 대조 학습 기반 접근 (Contrastive Learning)

**새로운 패러다임** (NI-MTSC, 2025):[6]

표현 학습과 동시에 데이터 증강:

$$\mathcal{L}\_{\text{contrastive}} = -\frac{1}{|P|}\sum_{p \in P}\log\frac{\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_p)/\tau)}{\sum_a \exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_a)/\tau)}$$

여기서 $P$는 긍정 쌍, $\tau$는 온도 파라미터입니다.

**혁신**: 인접 샘플 보간으로 더 판별적인 양성/음성 쌍 생성

#### C. 멀티 스케일 분해 (Multi-Scale Decomposition)

**원본의 한계**: 단일 분해 기법만 평가[1]

**최신 추세** (2024-2025):[7][8]
- **Variational Mode Decomposition (VMD) + 변환**: 다중 도메인 분해
- **성과**: 라디오 신호 분류에서 정확도 유의미 향상

#### D. 인과관계 기반 생성 (Causal Intervention)

**새로운 개념** (CFAMG, 2025):[9]

원인-결과 관계를 모델링하여 소수 클래스 샘플 생성:
$$X_{\text{counterfactual}} = X_{\text{majority}} \text{ with } C_{\text{minority}}$$

여기서 $C$는 인과 요인입니다.

**이점**: 모호한 클래스 경계에서 더 강력한 분리성

#### E. 자동 데이터 증강 (Automated Data Augmentation)

**원본의 한계**: 수동 하이퍼파라미터 튜닝 필요[1]

**최신 솔루션** (TSAA, 2025):[10]

이중 수준 최적화를 통한 자동 증강 정책 선택:
$$\min_A \mathcal{L}_{\text{val}}(\theta^*_A(A), A)$$

여기서 $A$는 증강 정책, $\theta^*_A$는 최적 모델 파라미터입니다.

***

### VI. 종합 성과 비교표

| 방면 | 원본 논문 (2021) | 최신 연구 (2024-2025) | 진전 |
|------|----------------|------------------|------|
| **평가 범위** | 12개 기법, 128개 데이터셋 | 60개 이상 기법, 다양한 도메인 | 5배 확대 |
| **모델 아키텍처** | CNN, RNN, 하이브리드 | + Transformer, Vision Transformer | 새 패러다임 추가 |
| **생성 모델** | GAN 제외 (외부 학습) | 적분, GAN, VAE, Diffusion | 주류 포함 |
| **최고 성능** | 85-90% (데이터셋 종속) | 96-100% (특화 도메인) | 도메인별 최적화 진전 |
| **자동화 수준** | 수동 선택 필요 | 자동 정책 학습 | 실무 편의성 향상 |
| **계산 효율** | wDBA 6,380초 (제한적) | Wavelet, Diffusion (더 효율적) | 실시간 적용 가능성 |

***

### VII. 향후 연구 시 고려할 점

#### A. 이론적 개선 방향

1. **표본 복잡도 분석 (Sample Complexity Bounds)**
   
   데이터 증강이 필요한 최소 샘플 수를 이론적으로 규명:
   $$N_{\min}(\delta, \epsilon) = f(\text{VC-dim}, \delta, \epsilon, \text{Augmentation Method})$$

2. **분포 보존 보장 (Distribution Preservation)**
   
   증강 샘플이 원본 분포 $p(\mathbf{x}|y)$를 변경하지 않는다는 보증:
   $$\text{KL}(p_{\text{original}} || p_{\text{augmented}}) < \epsilon$$

3. **최적 증강 정책의 특성화 (Characterization)**
   
   어떤 데이터셋 특성이 어떤 기법을 최적으로 하는지 이론적 근거 제시

#### B. 실무적 권장사항

**1) 전략적 선택 프레임워크**

```
IF 데이터셋 크기 < 500:
    CHOICE: Window Warping 또는 Slicing
    REASON: 빠른 연산, 일관된 성능
ELIF 시계열 길이 > 2000:
    CHOICE: 변환 기반 (지터링, 스케일링)
    REASON: O(T²) DTW 방법 회피
ELIF 클래스 불균형 > 0.3:
    CHOICE: SMOTE 기반 또는 DGW
    REASON: 소수 클래스 표현 증가
ELIF 신경망 = LSTM 기반:
    CHOICE: Magnitude Warping, Window Warping
    REASON: 시간 구조 보존 중요
ELSE:
    CHOICE: 자동 증강 (TSAA)
    REASON: 데이터셋별 최적 정책 학습
```

**2) 실험 설계 지침**

- 기저선(베이스라인) 대비 검증을 위해 **같은 반복 횟수의 원본 샘플 사용**[1]
- Paired t-검정으로 통계 유의성 확인 (p < 0.05)
- **교차 검증** (k-fold) 필수—단일 훈련/테스트 분할로는 분산이 큼

**3) 도메인별 최적화**

| 도메인 | 추천 기법 | 회피할 기법 |
|--------|---------|----------|
| **의료 신호 (ECG/EEG)** | SPAWNER, DGW, GAN | Rotation, Permutation |
| **센서 데이터** | 지터링, Scaling, wDBA | Time Warping (과도) |
| **금융 시계열** | SMOTE, Decomposition | Frequency warping |
| **음성/신호** | Frequency warping, SpecAugment | Slicing (정보 손실) |
| **움직임 데이터** | Magnitude warping, Window warping | Permutation |

#### C. 미흡한 부분 및 향후 과제

1. **아키텍처 특이성 문제**
   
   LSTM-FCN의 부정적 반응 원인을 근본적으로 분석할 필요가 있습니다. 높은 드롭아웃이 증강의 이점을 마스크하는지, 또는 아키텍처 자체가 증강을 필요로 하지 않는지 불명확합니다.

2. **다중 증강 조합 (Ensemble Augmentation)**
   
   논문은 단일 증강만 평가했지만, Um et al. 을 제외하고는 직렬 또는 병렬 조합의 시너지를 탐구하지 않았습니다. 향후 연구는 **증강 파이프라인 최적화**에 초점을 맞춰야 합니다.[11][1]

3. **비정상 시계열 (Non-stationary Time Series)**
   
   금융 데이터나 환경 센서 데이터 같이 분포가 시간에 따라 변하는 데이터에는 현재 기법들의 적용이 제한적입니다. STL 분해 후 각 성분별 증강이 한 가지 방향입니다.

4. **계산 효율성 vs. 정확도 트레이드오프**
   
   DGW (6,380초)나 wDBA (2,300초)는 매우 정확하지만, 실시간 응용에서는 불가능합니다. 최근의 확산 모델이나 Wavelet 기반 방법이 이를 개선하지만, 여전히 벤치마킹이 부족합니다.

***

### VIII. 결론: 이 논문의 학문적 영향과 실무적 가치

**Iwana & Uchida (2021)**의 논문은 시계열 분류 분야에서 데이터 증강의 **표준화와 체계화**라는 중대한 역할을 했습니다. 이전에 산재해 있던 기법들을 네 가지 계열로 분류하고, 128개 데이터셋에 걸친 포괄적 평가를 수행함으로써, 연구자들이 자신의 응용에 적합한 증강 기법을 **데이터 기반으로 선택**할 수 있는 프레임워크를 제공했습니다.

2020년 이후의 발전은 이 기초 위에서 **생성 모델의 정교화, 자동화, 인과 기반 접근** 등으로 심화되었습니다. 특히 Diffusion 모델과 대조 학습의 통합은 새로운 표현 학습 패러다임을 열었습니다.

그러나 실무 관점에서는 여전히 다음이 필요합니다:
- **투명한 의사결정 도구**: 데이터셋과 모델이 주어졌을 때 최적 기법을 추천하는 자동 시스템
- **이론적 보증**: 어떤 증강이 언제 안전한지, 분포 변경의 한계는 무엇인지에 대한 이론
- **도메인 특화 가이드**: 의료, 금융, 산업 분야별 best practice

이 논문은 이러한 과제로 가는 길을 환하게 밝혀주는 **이정표** 역할을 합니다.[1]

***

### 참고문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4ba1e3af-c157-41ce-825a-700bcf8911a0/2007.15951v4.pdf)
[2](https://ieeexplore.ieee.org/document/10833777/)
[3](https://ieeexplore.ieee.org/document/11229225/)
[4](https://aimjournals.com/index.php/ijmcsit/article/view/243/221)
[5](https://ieeexplore.ieee.org/document/10764737/)
[6](https://dx.plos.org/10.1371/journal.pone.0254841)
[7](https://dl.acm.org/doi/10.1145/3711896.3737049)
[8](https://ieeexplore.ieee.org/document/11182667/)
[9](https://ieeexplore.ieee.org/document/10902613/)
[10](https://arxiv.org/abs/2507.12645)
[11](https://royalsocietypublishing.org/rsta/article/379/2194/20200209/41189/Time-series-forecasting-with-deep-learning-a)
[12](https://www.semanticscholar.org/paper/4adee7936a2fe5564bf7807254e89cb7099868d7)
[13](http://arxiv.org/pdf/2310.10060.pdf)
[14](https://arxiv.org/html/2408.10951)
[15](https://arxiv.org/pdf/2405.16456.pdf)
[16](https://itc.ktu.lt/index.php/ITC/article/download/35797/16450)
[17](https://arxiv.org/html/2502.02924)
[18](https://arxiv.org/pdf/2201.11739.pdf)
[19](http://arxiv.org/pdf/2405.00319.pdf)
[20](http://thesai.org/Downloads/Volume15No1/Paper_118-Overview_of_Data_Augmentation_Techniques.pdf)
[21](https://www.sciencedirect.com/science/article/abs/pii/S0952197625020998)
[22](https://www.tandfonline.com/doi/full/10.1080/08839514.2025.2490057)
[23](https://par.nsf.gov/servlets/purl/10194342)
[24](https://arxiv.org/html/2310.10060v6)
[25](https://www.sciencedirect.com/org/science/article/pii/S1546221825008872)
[26](https://www.scitepress.org/Papers/2025/131859/131859.pdf)
[27](https://arxiv.org/html/2310.10060v5)
[28](https://arxiv.org/abs/2409.09106)
[29](https://www.sciencedirect.com/science/article/pii/S1319157822002361)
[30](https://www.ijcai.org/proceedings/2021/631)
[31](https://dl.acm.org/doi/full/10.1145/3663485)
[32](https://dl.acm.org/doi/10.1613/jair.1.17084)
[33](https://dl.acm.org/doi/10.1145/3742784)
[34](https://arxiv.org/html/2506.22927)
[35](https://ieeexplore.ieee.org/document/9263602/)
[36](https://www.nature.com/articles/s41598-025-12516-3)
[37](https://www.sciencedirect.com/science/article/pii/S156849462501508X)
[38](https://www.semanticscholar.org/paper/Data-Augmentation-for-Multivariate-Time-Series-An-Ilbert-Hoang/f55e28399bc720e77c24707e6bfdb6d5492740cb)
[39](https://github.com/thuml/Time-Series-Library)
[40](https://arxiv.org/html/2511.12104v1)
[41](https://arxiv.org/html/2503.10198v1)
[42](https://arxiv.org/pdf/2509.08306.pdf)
[43](https://arxiv.org/html/2506.14831v2)
[44](https://arxiv.org/html/2506.20347v1)
[45](https://arxiv.org/html/2503.06072v3)
[46](https://arxiv.org/html/2511.03799v1)
[47](https://arxiv.org/html/2503.23621v1)
[48](https://arxiv.org/pdf/2501.09223.pdf)
[49](https://arxiv.org/html/2507.22659v1)
[50](https://arxiv.org/html/2501.12215v1)
[51](https://arxiv.org/pdf/2510.03231.pdf)
[52](https://arxiv.org/html/2509.12845)
[53](https://arxiv.org/abs/2401.13912)
[54](https://arxiv.org/html/2509.08306v1)
[55](https://arxiv.org/pdf/2408.17059.pdf)
[56](https://arxiv.org/abs/2307.03759)
[57](https://arxiv.org/html/2507.11974v1)
[58](https://arxiv.org/pdf/2508.02621.pdf)
[59](https://arxiv.org/abs/2311.16834)
[60](https://openreview.net/pdf?id=dN9Sxy675T)
