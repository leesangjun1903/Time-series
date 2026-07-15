# Explainable Deep Neural Networks for Multivariate Time Series Predictions
## 종합 분석 보고서

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장
본 논문(Assaf & Schumann, IJCAI-19)의 핵심 주장은 다음과 같습니다:

> **CNN 기반 딥러닝 모델이 다변량 시계열 데이터에 대해 예측(prediction)뿐만 아니라, 그 예측에 대한 설명(explanation)도 동시에 제공할 수 있다.**

즉, 기존에 "블랙박스(black box)"로 여겨지던 딥러닝 모델을 **설명 가능한(Explainable AI, XAI)** 방식으로 설계하면서도 예측 정확도를 희생하지 않을 수 있음을 보입니다.

### 주요 기여
| 기여 항목 | 설명 |
|---|---|
| **2단계 CNN 아키텍처 설계** | 시간 차원과 특징(feature) 차원을 모두 보존하는 구조 제안 |
| **Grad-CAM의 시계열 적용** | 이미지 분야의 Grad-CAM을 다변량 시계열에 맞게 확장 적용 |
| **이중 Saliency Map 생성** | 시간 구간별 + 특징별 기여도를 동시에 시각화 |
| **정확도-설명성 트레이드오프 완화** | 설명 가능성 확보에도 불구하고 경쟁력 있는 분류 정확도 유지 |
| **실제 응용 사례 검증** | 태양광(PV) 발전소 에너지 생산량 예측에 실제 적용 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**문제 정의:**
- 다변량 시계열 데이터(Multivariate Time Series, MTS)를 분류하는 딥러닝 모델은 높은 성능을 보이지만, **왜 그런 예측을 했는지 설명하지 못하는 블랙박스 문제**가 존재함
- 에너지, 의료, 금융 등 **의사결정이 중요한 도메인**에서는 예측의 신뢰성(confidence) 확보가 필수적
- 기존 CNN 모델들은 시간 차원과 특징 차원의 중요도를 **동시에** 설명하는 메커니즘이 부재함

**구체적 응용 문제:**
- 태양광(PV) 발전소의 향후 4일간 평균 에너지 생산량을 6개 클래스(0~300 kW)로 분류
  - 데이터: 시간 단위 집계, 19 time steps/day, 9개 특징(기상 7개 + 센서 2개)
  - 입력: 80 time steps (4일) × 9 features

---

### 2.2 제안하는 방법 및 수식

#### 2.2.1 2단계 CNN 아키텍처

**Stage 1: 2D Convolution (특징별 패턴 학습)**

$$\text{Filter size: } k \times 1$$

- $k$개의 time step을 한 번에, 특징은 1개씩 독립적으로 처리
- 각 특징에서 발생하는 **개별 패턴(temporal pattern per feature)** 학습
- 이후 $1 \times 1$ Convolution으로 feature map 수를 1로 축소

$$\text{1×1 Conv: } \mathbb{R}^{T \times F \times f_{2d}} \rightarrow \mathbb{R}^{T \times F \times 1}$$

**Stage 2: 1D Convolution (교차-특징 패턴 학습)**

$$\text{Filter size: } k \times n \quad (n = \text{총 특징 수})$$

- 모든 특징(n개)을 동시에 고려하여 **특징 간 교차 패턴(cross-feature pattern)** 학습

**전체 구조 흐름:**
$$\text{Input}(T \times n) \xrightarrow{\text{2D Conv}(k\times1)} \xrightarrow{1\times1\text{ Conv}} \xrightarrow{\text{1D Conv}(k\times n)} \xrightarrow{\text{FC}} \text{Class } c$$

---

#### 2.2.2 Grad-CAM 기반 Saliency Map 생성

**Step 1: 중요도 가중치 계산**

클래스 $c$에 대한 출력 스코어 $y^c$를 feature map $A$의 각 activation unit $u$에 대해 편미분하고, 전체 평균(global average pooling)을 취합니다:

$$w^c = \frac{1}{Z} \sum_{u} \frac{\delta y^c}{\delta A_u} \tag{1}$$

- $Z$: feature map $A$ 내 총 unit 수
- 2D의 경우 $u$는 좌표 $(i, j)$를 가짐
- $w^c$: 클래스 $c$ 예측에 대한 feature map의 중요도 가중치

**Step 2: 가중 조합 및 ReLU 적용**

$$L^c_{1/2D} = \text{ReLU}\left(\sum_{f_{maps}} w^c A\right) \tag{2}$$

- **양의 기여만** 추출하기 위해 ReLU 적용
- $L^c_{2D}$: **각 time interval × feature**의 기여도 (2D Saliency Map)
- $L^c_{1D}$: **전체 특징의 공동 기여도** (1D Time Attention Map)

**두 Saliency Map의 의미:**

| Map | 의미 | 시각화 정보 |
|---|---|---|
| $L^c_{2D}$ | 특정 time step에서 특정 feature의 기여 | 어떤 특징이 언제 중요했는가 |
| $L^c_{1D}$ | 전체 feature의 시간별 공동 기여 | 어느 시간 구간이 가장 중요했는가 |

---

### 2.3 모델 구조 상세

```
입력층: [T=80, n=9] (다변량 시계열)
    ↓
[Stage 1]
2D Conv Layer (filter: k×1) → 특징별 시간 패턴 추출
    ↓
1×1 Conv Layer → Feature map 차원 압축
    ↓
[Stage 2]
1D Conv Layer (filter: k×n) → 교차-특징 시간 패턴 추출
    ↓
Fully Connected Layer
    ↓
출력층: 6개 클래스 (Softmax)

[설명 생성]
Grad-CAM → Stage 1 마지막 layer → L^c_{2D} (Feature-Time Saliency)
Grad-CAM → Stage 2 마지막 layer → L^c_{1D} (Time Attention)
```

---

### 2.4 성능 비교

**Table 1: 분류 정확도 비교 (태양광 에너지 생산 예측)**

| Model | Validation | Testing |
|---|---|---|
| **Proposed Net** | **87%** | **86%** |
| 1D CNN | 88% | 87% |
| 2D CNN | 84% | 83% |
| MLP | 72% | 67% |

**성능 분석:**
- 제안 모델은 1D CNN 대비 약 1%p 낮지만, **설명 가능성을 추가로 제공**하면서도 2D CNN, MLP 대비 우수한 성능
- 논문 저자들이 강조하는 핵심: *"정확도를 희생하지 않고 설명 가능성 달성"*
- 단, 1D CNN이 더 높은 정확도를 보임 → **설명 가능성을 위한 소폭의 정확도 트레이드오프 존재 가능성**

**한계점:**
- 단일 도메인(태양광 발전)에 한정된 실험 검증
- 소규모 실험: 벤치마크 데이터셋 수가 제한적
- 정량적 설명 품질 평가 지표 부재 (사람의 시각적 해석에 의존)
- 하이퍼파라미터 $k$의 최적값 선정 기준 불명확
- 회귀(regression) 문제가 아닌 분류(classification)로만 제한

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재 논문에서의 일반화 관련 내용

논문 자체에서 일반화 성능을 명시적으로 분석하지는 않으나, 구조적으로 일반화에 기여하는 요소들이 있습니다:

**① 2단계 구조의 귀납적 편향(Inductive Bias)**

$$\text{Stage 1: } f_{\text{per-feature}}(x_i) \quad \text{Stage 2: } f_{\text{cross-feature}}(x_1, ..., x_n)$$

- 특징별 패턴과 교차 특징 패턴을 분리하여 학습함으로써, **각 특징의 독립적 패턴과 상호작용 패턴을 분리 모델링**
- 이는 도메인 전환 시에도 비교적 안정적인 특징 표현 학습 가능

**② $1 \times 1$ Convolution을 통한 차원 축소**

$$\mathbb{R}^{T \times F \times f_{2d}} \rightarrow \mathbb{R}^{T \times F \times 1}$$

- 불필요한 파라미터 수를 줄여 **과적합(overfitting) 방지**에 기여
- Lin et al.(2013)의 Network in Network에서 검증된 기법

**③ Saliency Map의 정규화 효과**

- 모델이 **실제로 중요한 특징과 시간 구간에 집중**하도록 유도
- 노이즈 특징에 대한 과도한 의존 감소 가능

### 3.2 일반화 성능 향상을 위한 추가 가능성 (논문 외 분석)

**데이터 증강(Data Augmentation) 결합:**
$$\tilde{x} = \text{Aug}(x) = x + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$
- Window slicing, jittering, magnitude warping 등을 결합하면 일반화 성능 향상 기대

**Attention 메커니즘과의 결합:**

$$\alpha_t = \frac{\exp(e_t)}{\sum_{t'} \exp(e_{t'})}, \quad e_t = \text{score}(h_t)$$

- Self-attention과 결합 시 장기 의존성(long-term dependency) 포착 능력 향상

**도메인 적응(Domain Adaptation) 가능성:**
- Saliency Map이 어떤 특징이 도메인 불변(domain-invariant)인지 식별하는 데 활용 가능
- Transfer Learning 시 중요 특징 선택의 가이드로 활용

---

## 4. 앞으로의 연구에 미치는 영향 및 고려사항

### 4.1 연구에 미치는 영향

**① XAI + 시계열 연구의 방향 제시**
- 이미지 분야의 XAI 기법(Grad-CAM)을 시계열에 성공적으로 이식함으로써, **다른 이미지 XAI 기법들의 시계열 적용 가능성**을 열어줌
- LIME, SHAP, Integrated Gradients 등의 시계열 적용 연구 촉진

**② 산업 응용 가능성 확장**
- 에너지, 의료(ECG, EEG), 금융, 제조업 예측/이상탐지 등에서 **설명 가능한 예측 시스템** 구축의 기반

**③ 이중 Saliency Map 패러다임**
- 시간 + 특징 두 축에서의 설명 생성이라는 **이중 설명 프레임워크**는 후속 연구에서 표준 평가 방식으로 채택 가능

### 4.2 앞으로 연구 시 고려할 점

**기술적 고려사항:**

| 고려사항 | 설명 |
|---|---|
| **설명 품질의 정량 평가** | 인간 평가에만 의존하지 않는 자동화된 설명 품질 지표 개발 필요 |
| **더 다양한 데이터셋 검증** | 단일 도메인(PV 발전)에서 벗어나 UCR/UEA Archive 등 다양한 벤치마크 검증 |
| **Transformer와의 비교** | Attention 기반 모델(Transformer)과의 성능 및 설명 품질 비교 필요 |
| **회귀 문제 확장** | 현재 분류에만 적용; 연속값 예측(regression)으로 확장 필요 |
| **온라인/스트리밍 적용** | 실시간 데이터 스트림에 대한 설명 생성 속도 최적화 |
| **인과관계 vs. 상관관계** | Saliency Map은 상관관계를 보여주지만 인과관계와 혼동될 위험 존재 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 아래 비교는 제가 학습한 지식을 바탕으로 하며, 논문의 정확한 수치나 세부 내용에 대해서는 원문 확인이 필요합니다. 확인이 어려운 세부 수치는 의도적으로 기재하지 않겠습니다.

### 5.1 XAI for Time Series 분야 주요 후속 연구

**① TIMESHAP (Bento et al., 2021)**
- SHAP을 시계열에 맞게 확장한 Tree-based/model-agnostic 방법
- **차별점**: 모델 구조에 무관한 사후 설명(post-hoc explanation) 제공
- **비교**: Assaf & Schumann의 방법은 특정 CNN 아키텍처에 종속적이나, TIMESHAP은 모델 비종속적

$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!}[f(S \cup \{i\}) - f(S)]$$

**② Transformer for Time Series + Explainability**
- **Temporal Fusion Transformer (Lim et al., 2021, IJF)**: 어텐션 가중치 자체가 설명으로 활용
- **비교**: CNN 기반 Grad-CAM 설명보다 **장기 의존성 포착**과 **더 직관적인 어텐션 시각화** 제공

**③ TSInterpret (Schlegel et al., 2023)**
- 시계열 모델 설명을 위한 통합 라이브러리
- Gradient-based, perturbation-based, example-based 방법들을 통합
- Assaf & Schumann의 접근이 이 라이브러리의 gradient-based 카테고리의 선구적 사례로 위치

### 5.2 비교 요약표

| 논문 | 방법 | 설명 유형 | 모델 종속성 | 시간/특징 이중 설명 |
|---|---|---|---|---|
| **Assaf & Schumann (2019)** | Grad-CAM | 그래디언트 기반 | CNN 종속 | ✅ |
| TIMESHAP (2021) | SHAP 확장 | 게임이론 기반 | 모델 무관 | 부분적 |
| TFT (Lim et al., 2021) | Attention | 어텐션 가중치 | Transformer 종속 | ✅ |
| TSInterpret (2023) | 복합 | 복합 | 모델 무관 | ✅ |

---

## 참고 자료

### 본 논문 출처
- **Assaf, R., & Schumann, A. (2019)**. "Explainable Deep Neural Networks for Multivariate Time Series Predictions." *Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence (IJCAI-19)*, pp. 6488–6490.

### 논문 내 참조 문헌
- Selvaraju, R. R., et al. (2017). "Grad-cam: Visual explanations from deep networks via gradient-based localization." *ICCV 2017*, pp. 618–626.
- Adebayo, J., et al. (2018). "Sanity checks for saliency maps." *NeurIPS 2018*, pp. 9505–9515.
- Gilpin, L. H., et al. (2018). "Explaining explanations: An approach to evaluating interpretability of machine learning." *arXiv:1806.00069*.
- Lin, M., Chen, Q., & Yan, S. (2013). "Network in network." *arXiv:1312.4400*.
- Szegedy, C., et al. (2015). "Going deeper with convolutions." *CVPR 2015*, pp. 1–9.
- Fawaz, H. I., et al. (2018). "Deep learning for time series classification: a review." *arXiv:1809.04356*.
- Zheng, Y., et al. (2014). "Time series classification using multi-channels deep convolutional neural networks." *WAIM 2014*, pp. 298–310.
- Zheng, Y., et al. (2016). "Exploiting multi-channels deep convolutional neural networks for multivariate time series classification." *Frontiers of Computer Science*, 10(1):96–112.

### 비교 분석에 활용한 추가 문헌 (일반 지식 기반, 원문 확인 권장)
- Bento, J., et al. (2021). "TimeSHAP: Explaining Recurrent Explainable AI one timestamp at a time." *KDD 2021*.
- Lim, B., et al. (2021). "Temporal Fusion Transformers for interpretable multi-horizon time series forecasting." *International Journal of Forecasting*, 37(4):1748–1764.
- Schlegel, U., & Keim, D. A. (2023). "TSInterpret: A Python Package for the Interpretability of Time Series Classification." *arXiv:2208.05280*.
