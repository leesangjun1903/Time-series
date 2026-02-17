# T-SMOTE: Temporal-oriented Synthetic Minority Oversampling Technique for Imbalanced Time Series Classification

### 1. 핵심 주장 및 기여도 요약

T-SMOTE는 Microsoft Research에서 발표한 논문으로, 시계열 데이터의 클래스 불균형 문제를 해결하기 위한 혁신적인 오버샘플링 방법을 제안합니다. 이 방법의 핵심 가치 제안(value proposition)은 다음과 같습니다:[1]

**핵심 주장:**
시계열 데이터는 점진적이고 연속적으로 변하는 특성을 지니고 있으므로, 기존의 표준 SMOTE나 Borderline-SMOTE 같은 공간 기반 오버샘플링 기법은 시계열의 시간적 구조를 충분히 활용하지 못합니다. T-SMOTE는 (1) 시간적 특성을 고려한 클래스 경계 근처 샘플 생성, (2) 시간적 이웃을 활용한 합성 샘플 생성, (3) 가중 샘플링을 통한 노이즈 감소 등 세 가지 혁신을 통해 이를 극복합니다.

**주요 기여도:**
1. **시간-지향적 오버샘플링**: 기존의 K-최근접 이웃(K-NN) 방식을 대체하여, Leading time이라는 시간적 특성을 활용한 이웃 결정 방식 도입
2. **단변량/다변량 모두 지원**: SPO와 INOS는 단변량만 지원, MBO는 순차 구조 손실 문제가 있었던 것에 비해 T-SMOTE는 양쪽 모두 효과적으로 처리
3. **조기 예측(Early Prediction) 시나리오 우수성**: 산업 현장에서 중요한 미래 시점의 조기 예측에서 기존 방법 대비 현저하게 우수한 성능 입증[1]

***

### 2. 해결하고자 하는 문제

**근본적 문제**: 현실 세계의 많은 시계열 분류 과제에서 관찰되는 심각한 클래스 불균형(class imbalance) 문제
- 예시: 클라우드 플랫폼의 하드웨어 장애 예측(건강 상태:장애=19:1)
- 영향: 대부분의 머신러닝 알고리즘은 다수 클래스에 편향되어 소수 클래스의 분류 성능을 크게 떨어뜨림

**기존 방법의 한계**:
- **일반 오버샘플링** (SMOTE, Borderline-SMOTE, ADASYN): 시계열의 시간적 정보를 무시하고 공간적 거리만 고려
- **시계열 특화 방법** (SPO, INOS, MBO): 단변량만 지원하거나, 생성된 샘플의 순차 구조가 손상되는 문제 발생

**T-SMOTE가 해결하는 구체적 문제**:
시계열 데이터에서 소수 클래스의 대표성 부족으로 인한 분류 성능 저하 → 시간적 특성을 보존하면서 고품질 합성 샘플 생성 → 조기 예측 시나리오에서도 강건한 성능 달성

***

### 3. 제안하는 방법: 수식 포함 상세 설명

#### 3.1 기본 개념 및 표기법

원본 양성(양성=소수 클래스) 샘플 집합을 $P = \{X_1, \ldots, X_n\}$이라 하고, 음성 샘플 집합을 $N$이라 합니다. 각 양성 샘플 $X_i$는 2차원 행렬로 표현됩니다:

$$X_i = \begin{pmatrix} x^1_i \\ x^2_i \\ \vdots \\ x^T_i \end{pmatrix} \in \mathbb{R}^{T \times d}$$

여기서 $T$는 타임스탬프(수열 길이)이고, $d$는 특성(feature) 차원입니다. 각 $x^j_i \in \mathbb{R}^d$는 $j$번째 시점의 특성 벡터입니다.[1]

**Leading time** $l$이라는 핵심 개념: 각 원본 샘플 $X_i$로부터 길이 $w$인 부분수열을 생성합니다:

$$X^l_i = (x^{T-l-w+1}_i, \ldots, x^{T-l}_i)$$

Leading time $l$이 클수록 해당 샘플은 클래스 경계에 더 가깝습니다(양성 클래스에서 음성 클래스로 점진적으로 이동하는 구간을 포착).

#### 3.2 세 가지 핵심 단계

**단계 1: 클래스 경계 근처 샘플 생성**

최대 leading time $L$을 결정하기 위해 **Spy-based method**를 사용합니다:[1]

1. 음성 샘플 중 15%를 무작위로 선택하여 스파이(spy) $P_{spy}$로 지정
2. 음성 샘플을 재정의: $N' = N \setminus P_{spy}$
3. $P'= P \cup P_{spy}$로 재균형하고, 분류기 $f$를 훈련
4. 스파이 샘플의 예측 점수를 기반으로 임계값 $h = \max_{\text{spy}} s(\text{spy})$ 설정
5. Leading time $l$을 반복 증가시키다가, 평균 예측 점수가 임계값 $h$ 아래로 떨어지면 중단

이 과정을 통해 $L+1$개의 양성 샘플 쌍 $(X^l_i, s^l_i)$을 생성하며, $s^l_i = f(X^l_i)$는 양성 클래스일 확률입니다.[1]

**단계 2: 합성 샘플 생성 (Beta 분포 활용)**

기존 SMOTE는 가우스 분포를 사용했지만, T-SMOTE는 베타 분포를 활용하여 사전 정보를 더 잘 반영합니다.

각 생성된 샘플 $X^l_i$에 대해, 시간적 이웃 $X^{l+1}_i$를 선택하고 다음과 같이 합성 샘플을 생성합니다:

$$X^{\text{new}} = X^l_i + \beta(X^{l+1}_i - X^l_i)$$

여기서 $\beta \sim \text{Beta}(s^l_i, s^{l+1}_i)$이며, 베타 분포는 이항 분포의 켤레 사전분포로서 두 확률 매개변수를 반영합니다.[1]

새로운 합성 샘플의 예측 점수는:

$$s^{\text{new}} = s^l_i + (1 - s^l_i) \cdot \beta$$

**중요한 개선점**: 범주형 특성의 경우, T-SMOTE는 확률 $\beta$로 $X^l_i$의 값을, 확률 $(1-\beta)$로 $X^{l+1}_i$의 값을 선택합니다.

**합성 샘플 생성 수량 결정** (안전 레벨 기반):

$$m^l_i = \left\lfloor r \cdot \frac{s^l_i}{\sum_{j=1}^n \sum_{k=1}^L s^k_j} \right\rfloor$$

여기서 $r = |N|/|P|$는 불균형 비율입니다. 이 공식은 예측 점수가 높을수록(안전 레벨이 높을수록) 더 많은 합성 샘플을 생성하도록 설계되었습니다.[1]

**단계 3: 가중 샘플링을 통한 노이즈 감소**

클래스 경계 근처 샘플은 모호할 수 있으므로, 가중 샘플링을 적용합니다:

$$w = \max(0, s - h)$$

여기서 $s$는 예측 점수, $h$는 앞서 결정한 스파이 기반 임계값입니다.[1]

최종 양성 데이터셋 $P^*$는 이 가중치에 비례하는 확률로 모든 생성된 샘플(경계 근처 + 합성) 중에서 $|N|$개를 무작위로 선택하여 구성됩니다.

***

### 4. 모델 구조: 3단계 통합 프레임워크

T-SMOTE 알고리즘은 순차적 3단계 구조로 설계되었습니다:

```
단계 1: 부분수열 생성 및 클래스 경계 식별
  ├─ Leading time l을 0부터 시작하여 증가
  ├─ 각 l에 대해 부분수열 X^l 생성
  ├─ Spy-based method로 L값 결정
  └─ 결과: 각 샘플당 L+1개의 (X^l_i, s^l_i) 쌍

단계 2: 시간적 이웃을 활용한 합성 샘플 생성
  ├─ 각 X^l_i에 대해 시간적 이웃 X^{l+1}_i 결정
  ├─ Beta 분포로부터 보간 계수 β 샘플링
  ├─ 합성 샘플 생성: X_new = X^l_i + β(X^{l+1}_i - X^l_i)
  ├─ 안전 레벨에 따라 m^l_i개의 샘플 생성
  └─ 결과: 고품질의 시간적 구조를 보존한 합성 샘플들

단계 3: 가중 샘플링으로 최종 양성 데이터셋 구성
  ├─ 각 샘플에 가중치 w = max(0, s - h) 할당
  ├─ 가중치에 비례하여 샘플 선택
  └─ 결과: 균형잡힌 훈련 데이터셋
```

**설계 철학**: 시간적 정보를 최대한 보존하면서도, 클래스 경계 근처의 모호한 샘플로 인한 노이즈를 최소화합니다.

***

### 5. 성능 향상 및 실증적 결과

#### 5.1 종합 성능 비교

T-SMOTE는 10개 데이터셋(단변량 7개, 다변량 3개)에서 8개의 경쟁 방법과 비교됨:[1]

| 데이터셋 | 타입 | 불균형비율 | T-SMOTE AUC | MBO AUC | INOS AUC | SMOTE AUC |
|---------|------|----------|-----------|---------|---------|----------|
| Adiac | 단변량 | 37.70 | **0.9926** | 0.9803 | 0.9815 | 0.9500 |
| FaceAll | 단변량 | 13.00 | **0.9858** | 0.9677 | 0.9684 | 0.9510 |
| SLeaf | 단변량 | 18.96 | **0.9987** | 0.9823 | 0.9767 | 0.9584 |
| Wafer | 단변량 | 18.06 | **0.9992** | 0.9888 | 0.9890 | 0.9782 |
| AUS | 다변량 | 94.00 | **0.9991** | 0.9887 | 0.9889 | 0.9723 |
| **평균(모든 데이터셋)** | - | - | **0.9476** | 0.9353 | 0.9316 | 0.9121 |

**F1 점수 평가**:
- T-SMOTE 평균: 0.7821
- MBO 평균: 0.7656
- INOS 평균: 0.7555
- SMOTE 평균: 0.7127

T-SMOTE는 MBO 대비 **AUC 3.10% 향상**, **F1 점수 2.15% 향상**을 달성했습니다.[1]

#### 5.2 조기 예측(Early Prediction) 시나리오에서의 우수성

조기 예측은 산업 현장에서 매우 중요한 응용입니다. Leading time을 3, 5, 10으로 설정하고 테스트했을 때:[1]

**Leading time = 3**:
- T-SMOTE AUPRC: 0.8290 | MBO: 0.7961 | **개선도: 3.29%**

**Leading time = 5**:
- T-SMOTE AUPRC: 0.7644 | MBO: 0.7058 | **개선도: 5.86%**

**Leading time = 10**:
- T-SMOTE AUPRC: 0.5588 | MBO: 0.4534 | **개선도: 10.54%**

**핵심 발견**: Leading time이 증가할수록(더 멀리 미래를 예측할수록) T-SMOTE의 성능 우위가 더욱 두드러집니다. 이는 T-SMOTE가 생성한 다양한 leading time의 샘플들이 분류기가 더 정보 풍부한 패턴을 학습하게 만들기 때문입니다.

#### 5.3 시간적 정보 활용의 효과

시간적 특성을 고려하는 방법(T-SMOTE, MBO, INOS)과 고려하지 않는 방법들을 비교:[1]

| 메트릭 | 시간 고려 방법 | 기타 방법 | 개선도 |
|--------|--------------|---------|-------|
| AUC | 0.9353 (T-SMOTE) | 0.9121 (SMOTE) | 2.5% |
| F1 | 0.7821 (T-SMOTE) | 0.7127 (SMOTE) | 9.7% |
| AUPRC | 0.8495 (T-SMOTE) | 0.8009 (SMOTE) | 6.1% |

***

### 6. 모델의 일반화 성능 향상 가능성

#### 6.1 일반화 성능의 세 가지 메커니즘

**첫째, 시간적 구조 보존**:
T-SMOTE는 베타 분포를 이용한 보간으로 부드러운 전환을 보장하여, 생성된 샘플이 원본 데이터의 확률 분포 $p(X)$와 더 잘 일치합니다. 반면 기존 SMOTE는 고차원 공간에서 선형 보간을 수행하면서 분포 왜곡을 초래합니다.[2]

**둘째, 클래스 경계 집중**:
Spy-based method로 결정된 leading time 범위는 클래스 경계 근처의 가장 정보 풍부한 영역에 집중합니다. 이는 분류기가 의사결정 경계를 명확히 학습하게 도와, 테스트 데이터에 대한 일반화 능력을 향상시킵니다.[1]

**셋째, 노이즈 감소를 통한 안정성**:
가중 샘플링 단계에서 $w = \max(0, s - h)$로 모호한 샘플의 영향을 감소시켜, 합성 샘플로 인한 추가 노이즈를 효과적으로 제어합니다.

#### 6.2 일반화 성능의 정량적 증거

1. **다양한 데이터셋에서의 일관된 성능**:
   - 10개의 서로 다른 특성을 가진 데이터셋에서 모두 최고 또는 동등한 성능 달성
   - 불균형 비율이 극단적(94:1)인 AUS 데이터셋에서도 0.9991 AUC 달성[1]

2. **교차검증 안정성**:
   - 동등한 구조의 GAN 기반 방법들(IB-GAN)과 비교하면, 표준편차가 ±0.032~0.039 범위로 비교적 안정적입니다.[3]

3. **소수 클래스 성능**:
   - AUPRC 평균 0.8495는 소수 클래스 감지에 특화된 메트릭으로, 일반화 성능을 직접 반영합니다.[1]

#### 6.3 관련 최신 연구와의 비교를 통한 통찰

**IB-GAN (Imputation Balanced GAN, 2021)**:[3]
- 방식: GAN 기반 생성으로 마스킹된 벡터로부터 합성 샘플 생성 ($p_{miss}$로 노이즘 조절)
- 장점: 신경망 기반으로 더 복잡한 패턴 학습 가능
- 단점: 두 단계 훈련(생성 후 분류) 필요, 계산 복잡도 증가
- T-SMOTE 대비: T-SMOTE는 직접적이고 해석 가능성 높음

**APC (Autoregressive Predictive Coding, 2022)**:[4]
- 방식: 자가지도학습으로 시계열의 $n$-step 미래 예측 학습
- 장점: 결측값과 불균형을 동시에 해결
- 단점: 사전학습 단계 필요, 훈련 시간 길음
- T-SMOTE 대비: T-SMOTE는 직접적인 오버샘플링으로 더 빠른 적용 가능

**Diffusion Models with Multi-Domain Augmentation (2025)**:[5]
- 방식: 시간-주파수 도메인 동시 증강
- 장점: 매우 최신의 생성 모델 활용
- 단점: 계산 복잡도 매우 높음, 새로운 아키텍처 필요
- T-SMOTE 대비: T-SMOTE는 기존 분류기와 호환 가능, 더 실용적

**결론**: T-SMOTE는 **단순성, 효율성, 해석성** 면에서 우수하면서도, 성능에서는 최신 deep learning 기반 방법들과 경쟁할 수 있는 최적의 균형점을 제공합니다.

***

### 7. 한계(Limitations)

#### 7.1 방법론적 한계

1. **Spy-based threshold 결정의 불안정성**:
   - Leading time $L$ 결정이 스파이 샘플의 15% 선택에 민감할 수 있음
   - 서로 다른 무작위 시드에서 $L$이 변할 가능성

2. **베타 분포 모수의 민감성**:
   - $\beta \sim \text{Beta}(s^l_i, s^{l+1}_i)$에서 예측 점수 $s$에 크게 의존
   - 초기 훈련 시 분류기의 예측이 부정확하면 악영향 가능

3. **부분수열 길이 $w$ 선택의 임의성**:
   - 논문에서 $w$값을 설정하는 명확한 기준 제시 부재
   - 데이터셋별로 최적 $w$를 찾기 위한 추가 실험 필요

#### 7.2 실험적 한계

1. **제한된 데이터셋**:
   - 10개 데이터셋 대부분이 비교적 작은 규모 (수백~수천 샘플)
   - 대규모 시계열(예: 금융 시계열, 센서 네트워크) 데이터에서의 성능 미검증

2. **분류기 아키텍처의 고정성**:
   - 실험에서 LSTM만 사용하여, CNN, Transformer 등 다른 아키텍처에서의 성능 불명확

3. **조기 예측 시나리오의 제한성**:
   - 고정된 leading time (3, 5, 10)에서만 평가
   - 동적으로 변하는 시스템에서의 적응성 미검증

#### 7.3 확장성 한계

1. **다중 클래스 분류의 부재**:
   - 실험이 모두 이진 분류 문제
   - 3개 이상의 클래스를 가진 다중 클래스 불균형 문제는 미다룸

2. **결측값 처리 미흡**:
   - 시계열 데이터에서 흔한 결측값에 대한 명시적 처리 전략 부재
   - APC에서 다루는 이 문제가 T-SMOTE에서는 간과됨[4]

***

### 8. 논문이 앞으로의 연구에 미치는 영향 및 고려사항

#### 8.1 학술적 영향

**패러다임 전환**: T-SMOTE는 "시계열 특화 오버샘플링"이 단순한 휴리스틱이 아니라 체계적으로 설계할 수 있음을 보여주었습니다. 이후 연구들이 시간 축의 특성을 명시적으로 활용하도록 영감을 주었습니다.

**평가 메트릭 중요성 강조**: AUC, F1, AUPRC 세 메트릭을 동시에 보고함으로써, 불균형 데이터의 평가에서 단일 메트릭의 함정을 명확히 했습니다.[1]

#### 8.2 산업 응용의 가능성

Microsoft Azure와 365에 배포되어 실제 성과를 입증:[1]
- **디스크 장애 예측**: 예측 정확도 향상으로 사전 조치 시간 확보
- **고 지연 시간(High Latency) 예측**: 사용자 경험 저하 사전 방지

이는 이론적 기여를 넘어 실무적 가치를 입증했습니다.

#### 8.3 향후 연구시 고려할 점

1. **하이브리드 접근의 필요성**:
   - 데이터 레벨 (T-SMOTE) + 알고리즘 레벨 (cost-sensitive learning) 결합
   - 앙상블 방법과의 통합 (예: T-SMOTE + XGBoost)[6]

2. **도메인 적응(Domain Adaptation)**:
   - 소스 도메인에서 학습한 모델이 타겟 도메인으로 일반화되도록 하는 연구
   - 예: 한 병원의 환자 모니터링 데이터로 다른 병원에서 사용하는 모델 개발

3. **설명 가능성(Explainability) 통합**:
   - 생성된 합성 샘플이 왜 특정 특성을 가지는지 설명
   - SHAP, LIME 같은 해석 가능성 도구와의 결합

4. **적응형 파라미터 학습**:
   - 스파이 비율(15%), 베타 분포 모수 등을 데이터로부터 자동 학습
   - 메타 학습(Meta-learning) 활용

5. **결측값과 이상치 처리**:
   - T-SMOTE + GRU-D (결측값 처리) 같은 결합[4]
   - Robust SMOTE 개발로 이상치의 영향 최소화

#### 8.4 평가 메트릭의 진화

최근 연구(2023-2025)는 더 미묘한 평가 메트릭을 제안하고 있습니다:[7][8]
- **Minimum Recall Loss**: 모든 클래스에 동등한 중요도 부여
- **PATE (Proximity-Aware Time series anomaly Evaluation)**: 시계열 이상 탐지 맥락에서의 공간-시간적 근접성 고려

이러한 지표들이 표준화되면, T-SMOTE의 효과를 더욱 정교하게 평가할 수 있을 것입니다.

***

### 9. 2020년 이후 관련 최신 연구 비교 분석

| 연도 | 방법 | 핵심 특징 | 장점 | 단점 | T-SMOTE와의 관계 |
|------|------|---------|------|------|-----------------|
| **2020** | **APC**(Wever et al.)[4] | 자가지도학습 기반, GRU-D 활용 | 결측값과 불균형 동시 처리 | 사전학습 필요, 느린 수렴 | 상호보완: APC는 특성 학습, T-SMOTE는 데이터 균형화 |
| **2021** | **IB-GAN**(Deng et al.)[3] | GAN + 마스킹 기반 생성 | 고품질 합성 샘플, 모델 무관성 | 계산 복잡, 두 단계 훈련 | 대안 방법: IB-GAN은 더 복잡하지만 더 표현력 높음 |
| **2022** | **OHIT**(Tao et al.)[9] | 클러스터링 + 공분산 추정 | 다중모드 소수 클래스 처리 | DRSNN 클러스터링 필요 | 경쟁 방법: OHIT는 고차원 공간 특화 |
| **2023** | **Minimum Recall Loss**(Li et al.)[7] | 신규 손실 함수 설계 | 모든 클래스 동등 중요도 | 미분 불가능 함수 근사 필요 | 보완적: T-SMOTE에 새 손실함수 적용 가능 |
| **2023** | **Deep Imbalanced TSC**(Park et al.)[10] | LSTM + 로컬 이상도 | 예측 기반 이상 탐지 | 시계열 예측 능력 필요 | 대안 영역: 조기 예측에서 경쟁 |
| **2024** | **ShapeFormer**(Wu et al.)[11] | Shapelet + Transformer | 클래스 특화 특성 추출 | 새로운 아키텍처 필요 | 상호보완: 분류기 선택 문제로 결합 가능 |
| **2024** | **PATE**(Papadimitriou et al.)[8] | 새로운 평가 메트릭 | 공간-시간적 근접성 고려 | 메트릭일 뿐 방법론 아님 | T-SMOTE 평가 향상에 활용 가능 |
| **2025** | **MDCA**(Chuang et al.)[5] | 다중 도메인 diffusion | 시간-주파수 동시 증강 | 매우 높은 계산비용 | 차세대 기술: T-SMOTE 아이디어를 diffusion으로 확장 |
| **2025** | **AxelSMOTE**(Jovanović et al.)[12] | 에이전트 기반 오버샘플링 | 강화학습으로 적응형 생성 | 복잡한 훈련 프로세스 | 미래 방향: T-SMOTE를 강화학습으로 개선 가능 |

**핵심 통찰**:
- **단기(2020-2022)**: 신경망 기반 생성 방법(IB-GAN, APC) 등장으로 더 정교한 패턴 학습 시도
- **중기(2023)**: 평가 메트릭과 손실함수의 정제로 "어떻게 평가하는가"의 중요성 대두
- **최신(2024-2025)**: 최신 생성 모델(Diffusion, 강화학습)과 Transformer 기반 아키텍처 통합 추세

***

### 결론

T-SMOTE는 **시간적 정보의 체계적 활용**이라는 핵심 통찰과 **spy-based threshold**, **베타 분포**, **가중 샘플링** 등의 정교한 기법 조합으로, 시계열 불균형 분류 문제에서 강력한 성능을 입증했습니다. 

특히 **조기 예측 시나리오(early prediction)**에서의 우수성과 **산업 배포 성공 사례**는 이론과 실무의 간극을 메웠다는 점에서 의미가 있습니다. 

향후 연구는 결측값, 다중 클래스, 새로운 생성 모델과의 통합, 설명 가능성 강화 등의 방향으로 진행될 것으로 예상되며, T-SMOTE의 기본 철학(시간적 특성 보존)은 이러한 모든 발전의 기초가 될 것입니다.

***

### 참고문헌 색인

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/99e6dcf8-4618-4840-8191-1aba40c94441/0334.pdf)
[2](https://www.sciencedirect.com/science/article/abs/pii/S0020025519306838)
[3](https://arxiv.org/pdf/2110.07460.pdf)
[4](https://arxiv.org/pdf/2106.15577.pdf)
[5](https://ieeexplore.ieee.org/document/11065686/)
[6](https://www.nature.com/articles/s41598-025-09506-w)
[7](https://ieeexplore.ieee.org/document/10106021/)
[8](https://arxiv.org/html/2405.12096v1)
[9](https://www.sciencedirect.com/science/article/abs/pii/S0950705122003586)
[10](https://arxiv.org/abs/2302.13563)
[11](https://dl.acm.org/doi/10.1145/3637528.3671862)
[12](https://arxiv.org/html/2509.06875v1)
[13](https://dl.acm.org/doi/10.1145/3340531.3412710)
[14](https://ieeexplore.ieee.org/document/9044218/)
[15](https://www.semanticscholar.org/paper/b41cd8904eb6c54e880f38479dee323832c97e64)
[16](https://www.semanticscholar.org/paper/c3b926fb90b04834131dd803a1ff67fce5b39ac2)
[17](https://iopscience.iop.org/article/10.3847/1538-4365/aba8ff)
[18](http://pm-research.com/lookup/doi/10.3905/jfds.2020.1.040)
[19](https://www.mdpi.com/2072-4292/12/20/3301)
[20](https://www.mdpi.com/1660-4601/17/14/4979)
[21](https://ieeexplore.ieee.org/document/9313442/)
[22](https://ieeexplore.ieee.org/document/9311041/)
[23](http://arxiv.org/pdf/2405.12122.pdf)
[24](https://arxiv.org/pdf/2502.10381.pdf)
[25](https://arxiv.org/abs/2110.04748)
[26](https://arxiv.org/pdf/2502.06878.pdf)
[27](https://arxiv.org/pdf/1801.04396.pdf)
[28](https://arxiv.org/pdf/2204.03719.pdf)
[29](https://arxiv.org/pdf/2201.01212.pdf)
[30](https://www.ijcai.org/proceedings/2022/0334.pdf)
[31](https://dl.acm.org/doi/10.1145/3711896.3737049)
[32](https://smote-variants.readthedocs.io/en/latest/oversamplers.html)
[33](https://arxiv.org/html/2404.18537v1)
[34](https://www.sciencedirect.com/science/article/abs/pii/S0957417425032075)
[35](https://github.com/danielgy/Paper-list-on-Imbalanced-Time-series-Classification-with-Deep-Learning)
[36](https://github.com/ZhiningLiu1998/awesome-imbalanced-learning)
[37](https://towardsdatascience.com/smote-synthetic-data-augmentation-for-tabular-data-1ce28090debc/)
[38](https://em12.tistory.com/6)
[39](https://www.semanticscholar.org/paper/Class-imbalanced-time-series-anomaly-detection-on-Wang-Zhang/d9108dab513194c025621b0079fa612466d76d43)
[40](https://housekdk.gitbook.io/ml/ml/tabular/imbalanced-learning/oversampling-basic-smote-variants)
[41](https://pure.kaist.ac.kr/en/publications/deep-imbalanced-time-series-forecasting-via-local-discrepancy-den/)
[42](https://github.com/analyticalmindsltd/smote_variants)
[43](https://dl.acm.org/doi/10.1145/3637528.3671581)
[44](https://pmc.ncbi.nlm.nih.gov/articles/PMC10789107/)
[45](https://pypi.org/project/smote-variants/)
[46](https://pdfs.semanticscholar.org/f03d/764e82960195ca05d24b20540f94d4c22261.pdf)
[47](https://pdfs.semanticscholar.org/7116/100dbb52880cfabbbef20425fbe4490386dd.pdf)
[48](https://arxiv.org/pdf/2302.04018.pdf)
[49](https://www.biorxiv.org/content/10.1101/2021.01.07.425672v1.full-text)
[50](https://arxiv.org/pdf/2411.07013.pdf)
[51](https://arxiv.org/html/2407.17877v1)
[52](https://pdfs.semanticscholar.org/23e1/3352d88636b28210980225a1baa24b6d716d.pdf)
[53](https://pdfs.semanticscholar.org/c25f/e5884206a5cc83e245074815a182f78100e2.pdf)
[54](https://arxiv.org/pdf/2301.00496.pdf)
[55](https://pdfs.semanticscholar.org/f931/50675391907d152c4ba8a426698d762826bd.pdf)
[56](https://pdfs.semanticscholar.org/f7ff/74974f0297cfb2a5a84d78d58842b91f7780.pdf)
[57](https://pdfs.semanticscholar.org/8085/6938e9f0ec71a46a2648a43ee3c4d7004b3d.pdf)
[58](https://www.biorxiv.org/content/10.1101/2025.06.17.660271v3.full.pdf)
[59](https://pdfs.semanticscholar.org/79d2/586d8bc9ba549a4866997ae22e4f3e3982af.pdf)
[60](https://pdfs.semanticscholar.org/4e86/acb3af9d209ef4784cfa764d1008e0f3a0f8.pdf)
[61](https://pdfs.semanticscholar.org/f779/f5f17fe702c2bf880f791476808c8606c447.pdf)
[62](https://pdfs.semanticscholar.org/f4d1/41dfaf85d9a31ca0914af9ac609c59a9d46f.pdf)
[63](https://link.springer.com/10.1007/s00521-023-08636-4)
[64](https://ieeexplore.ieee.org/document/11182667/)
[65](https://ieeexplore.ieee.org/document/10233975/)
[66](https://ieeexplore.ieee.org/document/10239215/)
[67](https://www.sciendo.com/article/10.2478/amns-2024-1695)
[68](https://iopscience.iop.org/article/10.1088/1742-6596/1576/1/012045)
[69](https://ieeexplore.ieee.org/document/10456589/)
[70](https://arxiv.org/pdf/2010.05995.pdf)
[71](https://arxiv.org/pdf/2309.04732.pdf)
[72](https://pmc.ncbi.nlm.nih.gov/articles/PMC10947150/)
[73](https://www.nrso.ntua.gr/geyannis/wp-content/uploads/geyannis-pc327.pdf)
[74](https://neptune.ai/blog/anomaly-detection-in-time-series)
[75](https://blog.jetbrains.com/pycharm/2025/01/anomaly-detection-in-time-series/)
[76](https://arxiv.org/abs/2404.18537)
[77](https://pmc.ncbi.nlm.nih.gov/articles/PMC12217700/)
[78](https://arxiv.org/abs/2410.12206)
[79](https://www.nature.com/articles/s41598-025-26583-z)
[80](https://www.sciencedirect.com/science/article/pii/S2666651022000110)
[81](https://www.sciencedirect.com/science/article/pii/S259012302401658X)
[82](https://www.sciencedirect.com/science/article/pii/S0010482525001805)
[83](https://dl.acm.org/doi/full/10.1145/3691338)
[84](https://www.diva-portal.org/smash/get/diva2:1438865/FULLTEXT01.pdf)
[85](https://arxiv.org/html/2509.02592v1)
[86](https://papers.phmsociety.org/index.php/ijphm/article/view/3853)
[87](https://dl.acm.org/doi/abs/10.1109/TKDE.2007.190623)
[88](https://arxiv.org/html/2207.05295v2)
[89](https://arxiv.org/html/2412.20512v1)
[90](https://arxiv.org/html/2502.08960v3)
[91](https://arxiv.org/html/2211.05244v3)
[92](https://arxiv.org/html/2509.07605v1)
[93](https://arxiv.org/html/2509.19856v1)
[94](https://arxiv.org/abs/2010.11595)
[95](https://arxiv.org/html/2509.11511)
[96](https://arxiv.org/html/2412.19286v1)
[97](https://arxiv.org/html/2406.06518v1)
[98](https://arxiv.org/pdf/2408.03526.pdf)
[99](https://www.semanticscholar.org/paper/Early-Time-Series-Anomaly-Prediction-With-Chao-Huang/e3bed35a5c186cdd145990df778be9c3c62d91db)
[100](https://arxiv.org/html/2405.21070v1)
[101](https://arxiv.org/html/2404.00897v3)
