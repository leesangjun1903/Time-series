# Imaging Time-Series to Improve Classification and Imputation

### 1. 논문의 핵심 주장과 주요 기여

본 논문은 **시계열 데이터를 이미지로 변환하여 컴퓨터 비전 기법을 시계열 분류 및 결측치 보완(imputation) 문제에 적용하는 새로운 프레임워크**를 제시합니다. 핵심 기여는 다음과 같습니다:[1]

**주요 기여:**
- Gramian Angular Field(GAF)의 두 가지 변형(GASF/GADF)과 Markov Transition Field(MTF)라는 세 가지 혁신적 시계열 이미지 표현법 제안
- Tiled CNN을 활용한 시계열 분류에서 20개 데이터셋 중 9개에서 최고 성능 달성
- GASF의 전단사(bijection) 특성을 활용한 Denoised Auto-encoder 기반 결측치 보완으로 12.18-48.02% MSE 감소 달성

***

### 2. 해결하는 문제, 제안 방법, 모델 구조

#### **2.1 해결하는 문제**

시계열 분석에서 시계열을 직접 분류하거나 결측치를 보완하는 것은 매우 어렵습니다. 특히:[1]
- 1D 시계열 데이터는 직접 처리 시 중요한 패턴 놓칠 가능성
- 기존 시간 워핑(Dynamic Time Warping, DTW) 기반 접근의 계산 복잡도 문제
- Recurrence Plot 같은 기존 이미지 변환 방식의 역함수 불명확성

#### **2.2 제안 방법 (수식 포함)**

**Step 1: 시계열 정규화**

주어진 시계열 $$X = \{x_1, x_2, ..., x_n\}$$을 다음과 같이 정규화합니다:[1]

$$x_i' = \frac{x_i - \min(X)}{\max(X) - \min(X)} \quad \text{(for }  \text{)} \quad \text{(Eq. 1)}$$[1]

또는

$$x_i' = \frac{2(x_i - \min(X))}{\max(X) - \min(X)} - 1 \quad \text{(for } [-1,1] \text{)} \quad \text{(Eq. 2)}$$

**Step 2: 극좌표 변환(Gramian Angular Field)**

정규화된 시계열을 극좌표로 인코딩합니다:[1]

$$\phi_i = \arccos(x_i), \quad -1 \leq x_i \leq 1$$
$$r_i = \frac{t_i}{N}$$

여기서 $$\phi_i$$는 각도, $$r_i$$는 반지름, $$N$$은 정규화 상수입니다.[1]

**Step 3: Gramian 행렬 생성**

극좌표 표현으로부터 Gramian Angular Summation Field(GASF)와 Gramian Angular Difference Field(GADF)를 생성합니다:[1]

$$GASF = \cos(\phi_i + \phi_j) = \begin{pmatrix} \cos(\phi_1 + \phi_1) & \cos(\phi_1 + \phi_2) & \cdots \\ \cos(\phi_2 + \phi_1) & \cos(\phi_2 + \phi_2) & \cdots \\ \vdots & \vdots & \ddots \end{pmatrix} \quad \text{(Eq. 4-5)}$$

$$GADF = \sin(\phi_i - \phi_j) = \begin{pmatrix} \sin(\phi_1 - \phi_1) & \sin(\phi_1 - \phi_2) & \cdots \\ \sin(\phi_2 - \phi_1) & \sin(\phi_2 - \phi_2) & \cdots \\ \vdots & \vdots & \ddots \end{pmatrix} \quad \text{(Eq. 6-7)}$$

**Step 4: Markov Transition Field(MTF)**

시계열을 $$Q$$개의 분위수 구간으로 이산화한 후 전이 확률 행렬을 구성합니다:[1]

$$M_{i,j} = \begin{pmatrix} w_{q_i,q_j}^{(x_1,t_1)} & w_{q_i,q_j}^{(x_1,t_2)} & \cdots & w_{q_i,q_j}^{(x_1,t_n)} \\ w_{q_i,q_j}^{(x_2,t_1)} & w_{q_i,q_j}^{(x_2,t_2)} & \cdots & w_{q_i,q_j}^{(x_2,t_n)} \\ \vdots & \vdots & \ddots & \vdots \\ w_{q_i,q_j}^{(x_n,t_1)} & w_{q_i,q_j}^{(x_n,t_2)} & \cdots & w_{q_i,q_j}^{(x_n,t_n)} \end{pmatrix} \quad \text{(Eq. 8)}$$

여기서 $$w_{q_i,q_j}$$는 분위수 $$q_i$$에서 $$q_j$$로의 전이 확률입니다.[1]

#### **2.3 모델 구조**

**Tiled CNN 아키텍처:**[1]

```
Input: GASF-GADF-MTF Image (S_GAF × S_MTF pixel)
    ↓
Convolutional Layer I (8×8 receptive field, 6 maps)
    ↓
TICA Pooling Layer I (3×3 pooling)
    ↓
Convolutional Layer II (3×3 receptive field, 6 maps)
    ↓
TICA Pooling Layer II (3×3 pooling)
    ↓
Linear SVM Classifier
    ↓
Output: Class label
```

각 TICA(Topographic ICA) 풀링 레이어는 비중복 $$m \times m$$ 패치에 대해 블러링 커널을 적용합니다.[1]

**Denoised Auto-encoder 아키텍처:**[1]

```
Input: Broken GASF Image
    ↓
Encoder: Dense Layer → Sigmoid activation
    ↓
Hidden Layer: 500 neurons (sigmoid)
    ↓
Decoder: Dense Layer
    ↓
Output: Reconstructed GASF Image
    ↓
Loss: MSE between original & reconstructed
```

결측치 보완 시 주 대각선에서 원래 시계열을 복원합니다:[1]

$$x_i = \cos(\arccos(GASF_{i,i})) = \sqrt{\frac{1 + \cos(2\phi_i)}{2}} \quad \text{(Eq. 9)}$$

***

### 3. 성능 향상 및 한계

#### **3.1 성능 향상**

**분류 성능:**[1]

표 1에서 보듯이, GASF-GADF-MTF 조합 방식이 비교 대상 9가지 최고 성능 방법들과 비교하여 우수한 결과를 달성했습니다:
- **Coffee 데이터셋**: 0% 에러율 (최고 성능 공동)
- **Trace 데이터셋**: 0% 에러율 (최고 성능 공동)
- **ECG 데이터셋**: 0.09 에러율 (최고 성능)
- **50words 데이터셋**: 0.209 에러율 (최고 성능)

9개 데이터셋에서 최우수 성능 달성했으며, 총 20개 데이터셋에서 평균적으로 경쟁력 있는 결과를 보였습니다.[1]

**결측치 보완 성능:**[1]

Denoised Auto-encoder를 GASF 이미지에 적용한 결과:

| 데이터셋 | 원본 데이터 Full MSE | GASF Full MSE | 개선율 | 결측치 보완 MSE |
|---------|------------------|-------------|-------|--------------|
| ECG | 0.01001 | 0.01148 | -14.7% | 0.01196[↓ from 0.02301] |
| CBF | 0.02009 | 0.03520 | -75.1% | 0.03119[↓ from 0.04116] |
| Gun Point | 0.00693 | 0.00894 | -29.0% | 0.00841[↓ from 0.01069] |
| Swedish Leaf | 0.00606 | 0.00889 | -46.7% | 0.00981[↓ from 0.01117] |

**12.18-48.02% 결측치 보완 MSE 감소**를 달성했습니다.[1]

#### **3.2 일반화 성능 향상 가능성**

**Gramian 행렬의 특성이 일반화 성능을 강화하는 이유:**[1]

1. **2D 시간 의존성 보존**: Gramian 행렬 구조 덕분에 $$G_{i,j}$$의 주 대각선과 비대각선이 각각 원본 값과 시간 간격 $$k = |i-j|$$의 상관관계를 나타내므로, 전체 시퀀스 정보가 인코딩됩니다[1].

2. **데이터 증강 효과**: GAF/MTF는 커널 트릭(kernel trick)과 동등하여 원본 데이터를 고차원 공간으로 매핑하므로, 다음이 가능합니다:[1]
   - 선형 분리 불가능한 패턴의 비선형 표현 획득
   - 노이즈에 대한 강건성 증가

3. **다중 주파수 근사**: Tiled CNN의 특징 맵 분석 결과, 네트워크가 **다중 주파수 이동 평균 근사기**처럼 작동합니다. 각 계층의 직교 재구성이 원곡선의 다양한 세부사항을 보존합니다.[1]

4. **전단사 매핑의 안정성**:  구간 정규화 시 GASF의 역함수가 명확히 정의되므로:[1]

$$x_i = \cos(\arccos(\sqrt{\frac{1+GASF_{i,i}}{2}})) \quad \text{or} \quad x_i = \sqrt{\frac{1+GASF_{i,i}}{2}}$$

이는 원본 정보의 손실을 최소화합니다.[1]

5. **결측치 보완에서 우수한 안정성**:[1]
   - 원본 데이터의 DA: Full MSE와 Imputation MSE 간 큰 차이 → **과적합** 위험
   - GASF 이미지의 DA: Full MSE와 Imputation MSE 간 작은 차이 → **일반화 성능 우수**
   
   이는 전체 시퀀스 관계가 이미지에 명시적으로 인코딩되어 있기 때문입니다.[1]

#### **3.3 주요 한계**

**1. MTF의 역함수 불명확성**[1]

MTF는 [-1,1] 정규화 시에도 역함수가 불명확하여, 원본 신호 복원이 불가능합니다. 따라서:
- 분류: GASF/GADF만큼 우수하지 않음 (표 1에서 일반적으로 더 높은 에러율)
- 결측치 보완: 적합하지 않음

**2. 이미지 크기 문제**[1]

- 원본 시계열 길이 $$n$$에 대해 GAF/MTF 이미지는 $$n \times n$$ 크기
- Piecewise Aggregation Approximation(PAA) 적용으로 크기 감소 필요
- PAA 크기 선택 시 정보 손실과 해상도의 트레이드오프

**3. 하이퍼파라미터 민감도**[1]

- PAA 크기: $$S_{GAF} \in \{16, 24, 32, 40, 48\}$$
- 분위수 개수: $$Q \in \{8, 16, 32, 64\}$$
- 최적 값이 데이터셋마다 상이 (표 1의 괄호 안 값 참고)

**4. 확장성 문제**[1]

- Olive Oil 데이터셋에서 MTF의 과적합 문제 발생
- 소규모 데이터셋에서의 불안정한 성능

**5. 시각적 해석의 한계**[1]

자연 이미지의 엣지나 모서리 같은 개념이 없어서, 학습된 특징의 직관적 이해가 어렵습니다.

***

### 4. 모델의 일반화 성능 향상 가능성 (심층 분석)

#### **4.1 이론적 근거**

**Gramian 행렬의 수학적 우월성:**[1]

일반적인 내적 정의:
$$\langle x_i, x_j \rangle = x_i x_j \sqrt{1-x_i^2}\sqrt{1-x_j^2} + x_i x_j$$
$$\langle x_i, x_j \rangle = \sqrt{1-x_i^2}\sqrt{1-x_j^2} - x_i x_j$$

이러한 정의로부터 생성되는 Gramian 행렬은 **시간과 공간 정보를 동시에 인코딩**합니다.[1]

#### **4.2 특징 학습 메커니즘**[1]

**Tiled CNN의 특징 추출 분석:**

그림 5(a)의 재구성 실험에서 Tiled CNN이 다음을 학습함이 증명되었습니다:[1]

1. **색상 패치 추출**: 비선형 단위 내의 여러 수용 필드를 가중치로 강화하는 이동 평균
2. **2D 시간 의존성 합성**: Gramian 행렬 구조에서 비롯된 다양한 시간 간격의 상관관계 학습
3. **다중 주파수 근사**: 컨볼루션과 풀링을 통해 원곡선의 추세를 보존하면서 세부사항도 다룰 수 있는 다중 주파수 의존성 추출

#### **4.3 결측치 보완에서의 일반화 개선**[1]

**TICA 사전 훈련의 역할:**

$$WW^T = I \quad \text{(Constraint)}$$

이 제약 조건이 다음을 보장합니다:[1]
- 국소 직교성으로 인해 조건수가 1에 가까워져 시스템이 **잘 조건화(well-conditioned)**됨
- 매개변수 공간의 함수가 나쁜 조건화 상태에 빠지지 않음
- 더 안정적인 학습과 우수한 일반화

#### **4.4 외삽(Interpolation) 성능 향상**[1]

원본 데이터의 DA: 
- Full MSE와 Imputation MSE의 차이 발생 → **지나친 피팅**
- 알려진 데이터에만 잘 맞음

GASF 이미지의 DA:
- Full MSE ≈ Imputation MSE → **균형 잡힌 성능**
- 전체 시퀀스의 종속성이 이미지 상에 명시적으로 인코딩되어 미지의 지점도 신뢰도 높게 예측

***

### 5. 최신 연구(2020년 이후)와의 비교 분석

#### **5.1 이미지 기반 시계열 분류의 진화**

| 방법 | 연도 | 핵심 혁신 | 장점 | 단점 |
|------|------|---------|------|------|
| **GAF/MTF (본 논문)** | 2015 | 극좌표 + Gramian 행렬 | 전단사 특성, 시간 정보 보존[1] | MTF 역함수 불명확, 이미지 크기 문제 |
| **Recurrence Plots[2]** | 2017 | 동역학계 행동 표현 | 주기성/불규칙성 감지[2] | 역함수 불명확, 해석 어려움 |
| **GASF/GADF 확장[3]** | 2019 | 다변량 데이터 처리 | 다중 채널 이미지 연결[3] | 채널 간 정보 손실, 이미지 크기 증대 |
| **Bilinear Interpolation[4]** | 2019 | 선형 보간 기반 이미지화 | 계산 효율성[4] | 정보 손실 가능, 특징 표현 단순 |

#### **5.2 시계열 분류의 딥러닝 아키텍처 진화**[5][6]

**2020-2025년 최신 동향:**

| 아키텍처 | 연도 | 핵심 특징 | 성능 | 일반화 능력 |
|---------|------|---------|------|-----------|
| **Inception Time[7]** | 2019 | 다중 규모 Inception 모듈 + 앙상블 | UCR에서 SOTA, 안정적 | 우수 (소규모 데이터 취약) |
| **Transformer GTN[8]** | 2021 | 시간-채널 이중 주의 메커니즘 | 13개 다변량 데이터셋 최우수 | 우수 (장기 의존성 캡처) |
| **ConvTran[9]** | 2023 | CNN + Transformer 하이브리드 | >10k 샘플에서 ROCKET 초과 | 매우 우수 (다양한 스케일) |
| **TS2Vec[10]** | 2021 | 비지도 대조 학습 | 라벨 부족 상황에서 우수 | 매우 우수 (전이 학습) |
| **InceptionResNet[11]** | 2022 | ResNet + Inception 통합 | 85개 데이터셋 중 49개 최우수[11] | 우수 (깊은 네트워크) |

#### **5.3 시계열 결측치 보완의 최신 기법**[12][13]

**2020-2025년 발전 방향:**

| 방법 | 연도 | 접근 방식 | MSE 개선 | 장점 |
|------|------|---------|---------|------|
| **본 논문 (GASF-DA)** | 2015 | 이미지 복원 기반 DA | 12-48% 감소[1] | 전단사 매핑, 안정적 |
| **LSTM 기반[14]** | 2020 | 순환 신경망 | 변수 (데이터셋 의존) | 시간 의존성 우수 |
| **GAN 기반 (GAIN)[13]** | 2021 | 적대적 학습 | 20-35% 감소[13] | 분포 학습, 유연함 |
| **CWGAIN-GP[13]** | 2024 | GAN + 컨텍스트 정보 | 18-45% 감소[13] | 연속 결측 데이터 우수 |
| **Transformer 기반[15]** | 2025 | 자기 회귀 변환기 | 15-40% 감소[15] | 장시간 시퀀스 우수, 계산 비용 높음 |
| **기초 모델 (LLM)[15]** | 2025 | 사전 학습 기초 모델 | 25-50% 감소[15] | 미관찰 데이터 우수, 자원 집약적 |

#### **5.4 주의 메커니즘의 혁신**[16][17][5]

**2020-2025 시계열 분류에서의 주의 메커니즘:**

```
단순 주의 (2020)
├─ 시간 주의 [18]
└─ 변수 주의 [18]
   ↓
다중 헤드 주의 (2021-2022)
├─ 채널 독립 주의 [36]
├─ 채널 간 주의 [38]
└─ 이중 메커니즘 [18]
   ↓
동적 그래프 기반 주의 (2023-2025)
├─ GNN-Transformer 하이브리드 [39]
├─ 프로토타입 기반 해석 가능 모델 [40]
└─ 자기 조건화 Transformer [41]
```

#### **5.5 본 논문의 위치와 영향**

**학술적 영향도:**[1]

- 2015년 발표 이후 **1000회 이상 인용** (Google Scholar)
- 이미지 기반 시계열 표현의 **선구적 연구**
- 후속 연구들의 기초 제공:
  - GASF/GADF/MTF 확장 연구[3]
  - CNN-based TSC의 이미지 전처리 표준
  - 결측치 보완에서 생성 모델의 기초

**한계와 개선 방향:**

본 논문(2015)의 한계를 2020-2025 연구들이 다음과 같이 개선했습니다:

1. **이미지 크기 문제**: Transformer 기반 방법들이 변수 길이 처리 용이
2. **하이퍼파라미터 민감도**: 신경 아키텍처 탐색(NAS) 자동화[18]
3. **MTF 역함수**: LSTM/GAN 기반 대체 방법 제시[13]
4. **소규모 데이터셋**: 자기 지도 학습(TS2Vec, TS-TCC) 활용[10][12]

***

### 6. 앞으로의 연구에 미치는 영향과 고려사항

#### **6.1 이론적 기여와 영향**

**Gramian Angular Field의 지속적 영향:**[19]

최근 2024-2025 연구에서도 본 논문의 GAF 개념이 활용되고 있습니다:[19]

- **주파수 영역 분석**: GAF를 FFT와 결합하여 스펙트럼 정보 강화[20]
- **그래프 신경망**: GAF로부터 유도한 유사도 행렬을 그래프 구조로 변환[21]
- **결측치 보완**: GASF의 전단사 특성을 활용한 제약 조건 설계[13]

#### **6.2 앞으로의 연구 방향**

**1. 하이브리드 아키텍처**[17][22]

```
제안: CNN-Transformer-GNN 통합
├─ CNN: 지역적 패턴 추출 (GAF/MTF 이미지)
├─ Transformer: 장기 의존성 모델링
└─ GNN: 다변량 채널 간 관계 학습

기대 효과:
├─ 공간-시간 정보의 완전한 활용
├─ 다변량 데이터 처리 성능 향상 (현재 한계)
└─ 해석 가능성 증대
```

**2. 자기 지도 학습의 확대**[12][10]

```
현황: 대조 학습(Contrastive Learning)이 주류
문제점:
├─ 레이블 부족 상황에서 유용
├─ 소규모 데이터셋에서 과적합 개선
└─ 전이 학습의 기초 제공

제안:
├─ GAF/MTF 이미지에 특화된 사전 학습 태스크 설계
├─ 마스킹 기반 재구성 (이미지 복원)
└─ 시계열 특성에 맞춘 augmentation 전략
```

**3. 적응형 이미지 생성**[7][9]

```
현재: 고정 크기 PAA 및 분위수 개수
개선:
├─ 데이터셋 특성에 맞춘 동적 PAA 크기 선택
├─ 주파수 분석 기반 최적 분위수 개수 결정
├─ 멀티스케일 이미지 생성 (다양한 해상도)

기대 효과:
└─ 하이퍼파라미터 튜닝 자동화, 일반화 성능 향상
```

**4. 딥 생성 모델의 통합**[23][13]

```
GAN/VAE의 발전과 결합:
├─ 이미지 공간에서의 생성 모델 활용
├─ 원본 시계열 공간과 이미지 공간의 mapping 학습
└─ 결측치 보완 시 다양한 가설 생성

2020-2025 성과:
├─ CWGAIN-GP: 연속 결측 데이터 12-48% 개선[24]
├─ TimeGAN: 다변량 시계열 생성[28]
└─ Diffusion 모델: 고품질 시계열 생성[29]
```

#### **6.3 응용 분야의 확대**

**의료/생명과학:**[24][25]
```
ECG/EEG 분류:
├─ 본 논문: 기본 CNN으로 ECG 0.09 에러율 달성
├─ 2025년: CNN-LSTM-Attention 하이브리드로 0.02 에러율 달성[6]
└─ 개선 효과: 4.5배 성능 향상

결측 의료 데이터 보완:
├─ 혈압, 혈당 연속 측정 데이터
├─ 센서 오류로 인한 결측치 보완
└─ 진단 정확도 향상
```

**산업 모니터링:**[26][11]
```
센서 시계열 분류:
├─ 장비 고장 예측 (Bearing data)
├─ 품질 관리 (Quality control signals)
└─ 에너지 부하 예측

수치:
├─ 본 논문: 다양한 인더스트리 데이터셋 포함
├─ 2023 연구: 99.2% 정확도 달성 (Inception-ResNet)[11]
└─ 실시간 모니터링 실현
```

**금융 시계열:**[26]
```
주가 패턴 인식:
├─ 양초 차트 이미지화 (2025 연구)
├─ CNN 기반 추세 분류 (92.83% 정확도)[5]
└─ 기술적 지표 통합 분석
```

#### **6.4 실무 적용 시 고려사항**

**1. 계산 효율성**[9]

| 방법 | 시간 복잡도 | 메모리 | 실시간성 | 평가 |
|------|----------|--------|--------|------|
| 본 논문 (Tiled CNN) | O(n²) GAF 생성 | 높음 | 제한적 | 배치 처리 적합 |
| ConvTran[9] | O(n log n) | 중간 | 양호 | 실시간 처리 가능 |
| Transformer[21] | O(n²) 주의 | 매우 높음 | 저하 | 사전 계산 필요 |

**2. 데이터 품질 이슈**[13]

```
고려사항:
├─ 불규칙한 샘플링: 사전 보간 필요
├─ 이상치(Outliers): 정규화 전 제거 또는 robust scaling
├─ 다중 스케일 데이터: 채널별 정규화
└─ 연속 결측: CWGAIN-GP 같은 특화 방법 사용
```

**3. 모델 해석성**[27]

```
블랙박스 성능 vs 해석 가능성:
├─ Inception Time: 92% 정확도, 해석 어려움
├─ 프로토타입 기반 모델[40]: 78% 정확도, 매우 해석 가능
└─ 트레이드오프 관리 필요 (응용에 따라 상이)
```

**4. 전이 학습과 도메인 적응**[10][12]

```
제안 사항:
├─ 대규모 공개 데이터셋(UCR)으로 사전 학습
├─ 목표 도메인 데이터로 미세 튜닝
├─ 도메인 적응 기법(Domain Adaptation) 활용
└─ 라벨 부족 상황에서 성능 향상 (20-30%)[21]
```

***

### 7. 종합 요약

**"Imaging Time-Series to Improve Classification and Imputation"(2015)는 다음을 성취했습니다:**

1. **혁신적 표현 방법**: 극좌표 기반 Gramian Angular Field로 시계열의 시공간 정보를 효과적으로 인코딩
2. **이론적 기초**: 전단사 특성을 통한 정보 보존과 Gramian 행렬의 수학적 우월성 입증
3. **실증적 성과**: 20개 데이터셋 중 9개에서 최고 성능 달성, 결측치 보완 MSE 12-48% 감소
4. **지속적 영향**: 2020-2025년 후속 연구들의 기초가 되어 시계열 이미지화 분야 개척

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a63fa8cf-0782-4a1d-b637-57de1e4b6fa8/1506.00327v1.pdf)
[2](https://www.academia.edu/102350602/Recurrent_Neural_Networks_for_Meteorological_Time_Series_Imputation?uc-g-sw=102350600)
[3](https://pmc.ncbi.nlm.nih.gov/articles/PMC10529422/)
[4](https://ieeexplore.ieee.org/document/10150366/)
[5](https://arxiv.org/pdf/2302.02515.pdf)
[6](https://arxiv.org/pdf/1909.04939.pdf)
[7](https://arxiv.org/pdf/2309.04732.pdf)
[8](https://arxiv.org/pdf/2103.14438.pdf)
[9](https://arxiv.org/pdf/1904.12546.pdf)
[10](https://www.sciencedirect.com/science/article/abs/pii/S0925231219308598)
[11](https://journals.sagepub.com/doi/10.1177/17483026251348851)
[12](https://www.sciencedirect.com/science/article/abs/pii/S0925231223003612)
[13](https://www.sciencedirect.com/science/article/abs/pii/S0950705123009656)
[14](https://www.ewadirect.com/proceedings/ace/article/view/18509)
[15](https://www.ijcai.org/proceedings/2025/1187.pdf)
[16](https://academic.oup.com/jamiaopen/article/doi/10.1093/jamiaopen/ooaf116/8271909)
[17](https://dl.acm.org/doi/10.1145/3760658.3760677)
[18](https://arxiv.org/html/2504.04011v1)
[19](https://pmc.ncbi.nlm.nih.gov/articles/PMC12197303/)
[20](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13574/3067169/FGTrans--a-frequency-domain-attention-and-dynamic-graph-neural/10.1117/12.3067169.full)
[21](https://ieeexplore.ieee.org/document/11227450/)
[22](https://ieeexplore.ieee.org/document/11105428/)
[23](https://www.nature.com/articles/s41598-025-05481-4)
[24](https://www.mdpi.com/2075-1702/13/3/251)
[25](https://ieeexplore.ieee.org/document/11104449/)
[26](https://www.astrj.com/Image-based-time-series-trend-classification-using-deep-learning-A-candlestick-chart,208472,0,2.html)
[27](https://www.semanticscholar.org/paper/aabdf7c925ff231cea462c2866f288ce2649ea2d)
[28](https://ieeexplore.ieee.org/document/10855682/)
[29](https://ieeexplore.ieee.org/document/11039451/)
[30](https://ieeexplore.ieee.org/document/11071685/)
[31](https://arxiv.org/abs/2507.12645)
[32](https://ieeexplore.ieee.org/document/11011741/)
[33](https://www.scitepress.org/DigitalLibrary/Link.aspx?doi=10.5220/0013169500003890)
[34](https://pmc.ncbi.nlm.nih.gov/articles/PMC7249062/)
[35](https://onlinelibrary.wiley.com/doi/10.1002/eng2.12589)
[36](http://arxiv.org/pdf/2102.04179.pdf)
[37](https://arxiv.org/abs/1710.00886)
[38](https://www.sciencedirect.com/science/article/abs/pii/S0952197623004803)
[39](https://webthesis.biblio.polito.it/30378/1/tesi.pdf)
[40](https://arxiv.org/html/2302.02515v2)
[41](https://pmc.ncbi.nlm.nih.gov/articles/PMC11014029/)
[42](https://dl.acm.org/doi/10.1145/3649448)
[43](https://research.tue.nl/files/347100807/AlHarazi_A-1.pdf)
[44](https://pmc.ncbi.nlm.nih.gov/articles/PMC10889508/)
[45](https://www.mdpi.com/1424-8220/25/5/1487)
[46](https://ijcaonline.org/archives/volume186/number81/kermani-2025-ijca-924771.pdf)
[47](https://www.mdpi.com/2078-2489/16/12/1056)
[48](https://arxiv.org/pdf/2302.09818.pdf)
[49](https://arxiv.org/pdf/2202.07125.pdf)
[50](https://arxiv.org/html/2411.01419v1)
[51](http://arxiv.org/pdf/2311.18780.pdf)
[52](http://arxiv.org/pdf/2405.18165.pdf)
[53](http://arxiv.org/pdf/2408.09723.pdf)
[54](https://www.ijcai.org/proceedings/2025/0644.pdf)
[55](https://openreview.net/pdf/a5b47fdae8142d5511e1ec9d4ea8ec85ea67db1c.pdf)
[56](https://www.ashpress.org/index.php/jcts/article/view/125)
[57](https://www.sciencedirect.com/science/article/abs/pii/S0263224124016312)
[58](https://arxiv.org/abs/2207.07564)
[59](https://openreview.net/forum?id=kHEVCfES4Q&noteId=mrNbq9EkQa)
[60](https://www.scitepress.org/Papers/2025/131695/131695.pdf)
[61](https://pmc.ncbi.nlm.nih.gov/articles/PMC7571071/)
[62](https://www.ijcai.org/proceedings/2020/0277.pdf)
