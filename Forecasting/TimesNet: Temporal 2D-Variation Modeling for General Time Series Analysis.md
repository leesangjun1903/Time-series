# TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis

## 1. 핵심 주장과 주요 기여 (간단 요약)

- **핵심 주장**: 시계열은 다중 주기성(multi‑periodicity)을 가지며, 한 시점의 변화는 “한 주기 안(intraperiod)”과 “서로 다른 주기 간(interperiod)” 변동이 동시에 얽혀 있다. 기존 1D 시계열 상에서 이를 직접 모델링하는 것은 표현력이 부족하므로, 1D를 여러 주기에 기반한 2D 텐서로 변환해 2D 커널(비전 백본)을 사용해 **temporal 2D‑variation**을 모델링하면 더 일반적이고 강력한 시계열 백본을 만들 수 있다는 주장이다.[^1_2]
- **주요 기여**[^1_2]

1. 시계열의 다중 주기성을 기반으로, 1D 시계열을 다수의 주기에 기반한 2D 텐서 집합으로 변환하여 intra/inter‑period 변동을 동시에 표현하는 **Temporal 2D‑variation** 프레임워크 제안.
2. 주기 추출(FFT), 1D→2D 리쉐이프, 2D 비전 백본(Inception 블록)과 주파수‑가중 합산을 결합한 **TimesBlock** 및 이를 쌓은 **TimesNet**을 제안.
3. Long/short‑term forecasting, imputation, classification, anomaly detection 등 5가지 주요 시계열 과제에서 SOTA 또는 그에 준하는 성능을 보이며, **task‑general backbone (foundation‑style model)**로서의 가능성을 보여줌.
4. 2D 비전 백본(ResNet, ResNeXt, ConvNeXt, Swin 등)을 대체 모듈로 쓸 수 있음을 보여, 시계열과 비전 커뮤니티를 연결하는 설계임을 강조.[^1_2]

***

## 2. 해결 문제, 방법(수식 포함), 구조, 성능, 한계 (자세한 설명)

### 2.1 해결하고자 하는 문제

1. **복잡한 시계열 변동(temporal variation) 분해의 어려움**
    - 실제 시계열은 연속성과 추세, 다중 주기(일/주/월/연), 잡음, 급격한 변동이 섞여 있어,
상승/하락/진동 등의 패턴이 서로 겹치며 나타난다.[^1_2]
    - 기존 RNN, TCN, Transformer는 대부분 1D 시계열 상에서 인접 시점 또는 시점‑쌍의 의존성만을 직접 학습한다.
2. **1D 표현의 한계**
    - 한 시점의 값은 스칼라(혹은 저차원 벡터)라, 그 시점이 “어떤 주기 내에서 어느 phase에 있는지”와 “이 phase가 다른 주기들에서 어떻게 반복되는지”를 동시에 드러내기 어렵다.[^1_2]
    - attention 기반 모델도 “퍼져 있는 시점들” 사이에서 유의미한 의존성을 찾는 것이 어려워질 수 있다.[^1_2]
3. **여러 시계열 과제를 하나의 백본으로 다루기 어려움**
    - 기존 모델은 보통 forecasting 전용, anomaly 전용 등 task‑specific 설계가 많아, 하나의 통합 백본으로 다양한 과제를 잘 풀기 어렵다.[^1_2]

> 이 논문은 “다중 주기성에 기반한 2D 구조화”를 통해 위 문제를 동시에 완화하는 **task‑general 시계열 백본**을 제안한다.[^1_2]

***

### 2.2 제안 방법: 1D → 2D 변환과 주기 기반 모듈화

#### 2.2.1 주기 추출: FFT로 유의미한 주파수 선택

입력 시계열을 길이 $T$, 변수 수 $C$라 하면

$$
X_{\text{1D}} \in \mathbb{R}^{T \times C}.
$$

먼저 FFT로 주파수별 진폭을 계산한다.[^1_2]

$$
A = \text{Avg}\left( \text{Amp}\left( \text{FFT}(X_{\text{1D}}) \right) \right)
\in \mathbb{R}^{T},
$$

여기서 $\text{Amp}(\cdot)$는 복소 FFT 결과의 크기, $\text{Avg}(\cdot)$는 채널 방향 평균.[^1_2]

상위 $k$개의 큰 진폭을 갖는 주파수들을 선택한다.

$$
\{ f_1, \dots, f_k \}
= \text{Top}_k\big( \{ A_f \}_{f=1}^{\lfloor T/2 \rfloor} \big),
$$

각 주파수 $f_i$에 대응되는 “대략적인 주기 길이”는

$$
p_i = \left\lfloor \frac{T}{f_i} \right\rfloor, \quad i = 1,\dots,k.
$$

논문에서는 이를 요약해

$$
A, \{ f_i\}, \{ p_i \} = \text{Period}(X_{\text{1D}})
$$

로 표기한다.[^1_2]

#### 2.2.2 1D 시계열을 주기별 2D 텐서로 리쉐이프

각 주파수/주기쌍 $(f_i, p_i)$에 대해, $X_{\text{1D}}$를 zero‑padding 해서 $p_i \times f_i$ 크기가 되도록 만든 뒤 2D 텐서로 변환한다.[^1_2]

1. 패딩:

$$
\tilde{X}_{\text{1D}} = \text{Padding}(X_{\text{1D}}) \in \mathbb{R}^{(p_i f_i) \times C}.
$$

2. 리쉐이프:

$$
X^{(i)}_{\text{2D}} = \text{Reshape}_{p_i, f_i}\left( \tilde{X}_{\text{1D}} \right) 
\in \mathbb{R}^{p_i \times f_i \times C}.
$$

여기서[^1_2]

- **열(column)**: 하나의 주기 내에서 시간 순서에 따른 intra‑period 변동.
- **행(row)**: 서로 다른 주기들에서 같은 phase에 해당하는 시점들 간의 inter‑period 변동.

결과적으로, 각 $X^{(i)}_{\text{2D}}$ 는 특정 주기 $p_i$에 대한 **temporal 2D‑variation**을 담는다.[^1_2]

#### 2.2.3 TimesBlock 내부 연산 (수식 포함)

모델은 여러 층 $l=1,\dots,L$의 TimesBlock을 residual 형태로 쌓는다. 입력 길이‑ $T$ 특징 시계열을

$$
X^{(l-1)}_{\text{1D}} \in \mathbb{R}^{T \times d_{\text{model}}}
$$

라고 하면, TimesBlock의 출력은

```math
X^{(l)}_{\text{1D}} 
= \text{TimesBlock}\big( X^{(l-1)}_{\text{1D}} \big)
+ X^{(l-1)}_{\text{1D}}.
```

여기서 TimesBlock은 다음 두 단계로 구성된다.[^1_2]

##### (1) 깊은 특징에 대한 주기 추정 및 1D→2D 변환

먼저 깊은 특징에 대해 다시 Period 연산을 수행한다.

$$
A^{(l-1)}, \{ f_i \}, \{ p_i \} 
= \text{Period}\big( X^{(l-1)}_{\text{1D}} \big),
$$

이후 각 $i$에 대해

$$
X^{(l,i)}_{\text{2D}} 
= \text{Reshape}_{p_i, f_i}\left(\text{Padding}(X^{(l-1)}_{\text{1D}})\right)
\in \mathbb{R}^{p_i \times f_i \times d_{\text{model}}}.
$$

이 텐서에 대해 **Inception 기반 2D 블록**을 적용한다.[^1_2]

$$
\tilde{X}^{(l,i)}_{\text{2D}} = \text{Inception}\big( X^{(l,i)}_{\text{2D}} \big),
$$

- Inception 블록은 $1 \times 1$, $3 \times 3$, $5 \times 5$, $7 \times 7$ 등 여러 크기 2D 커널을 병렬로 적용하는 구조로, 서로 다른 scale의 intra/inter‑period 변동을 동시에 포착한다.[^1_2]
- 주의할 점: $k$개의 주기마다 **동일한 파라미터를 공유**하는 Inception 모듈을 사용해 파라미터 효율을 확보한다.[^1_2]

다시 1D로 되돌릴 때는, 패딩을 잘라(truncate) 원래 길이 $T$로 맞춘다.

$$
\tilde{X}^{(l,i)}_{\text{1D}} 
= \text{Trunc}\Big(
\text{Reshape}_{1,\, p_i f_i}\big( \tilde{X}^{(l,i)}_{\text{2D}} \big)
\Big)
\in \mathbb{R}^{T \times d_{\text{model}}}.
$$

##### (2) 주파수 진폭 기반 adaptive aggregation

각 주기 $p_i$의 상대적 중요도는 진폭 $A^{(l-1)}_{f_i}$로 반영한다.[^1_2]

정규화 weight:

$$
\tilde{A}^{(l-1)}_{f_i} 
= \frac{\exp(A^{(l-1)}_{f_i})}{\sum_{j=1}^{k} \exp(A^{(l-1)}_{f_j})},
\quad i = 1,\dots,k.
$$

마지막으로 $k$개의 1D 표현을 가중합:

$$
X^{(l)}_{\text{1D}} 
= \sum_{i=1}^{k} \tilde{A}^{(l-1)}_{f_i}\, \tilde{X}^{(l,i)}_{\text{1D}}.
$$

이렇게 얻은 $X^{(l)}_{\text{1D}}$가 해당 층 TimesBlock의 출력이며, residual 연결을 포함한다.[^1_2]

#### 2.2.4 전체 모델 구조 (TimesNet)

- **입력 임베딩**:

$$
X^{(0)}_{\text{1D}} = \text{Embed}(X_{\text{1D}}) \in \mathbb{R}^{T \times d_{\text{model}}}.
$$
- **Stacked TimesBlocks**:
TimesBlock을 $L$개 쌓아

$$
X^{(L)}_{\text{1D}} \in \mathbb{R}^{T \times d_{\text{model}}}
$$

를 획득.[^1_2]
- **Task‑specific 헤드**:
    - Forecasting: 마지막 시점(또는 윈도우)에서 MLP를 통해 미래 구간을 예측.
    - Imputation: 입력에서 마스킹된 위치에 대해 재구성 값을 예측.
    - Classification: 전체 시퀀스 풀링(global average 등) 후 softmax classifier.
    - Anomaly detection: 재구성 기반 autoencoder 방식으로, 재구성 오차를 anomaly score로 사용.[^1_2]

***

### 2.3 성능 향상 (다양한 태스크에서의 결과)

논문은 5가지 주요 태스크에서 광범위한 비교를 수행한다.[^1_2]

1. **Long‑term forecasting** (ETT, Electricity, Traffic, Weather, Exchange, ILI 등)
    - Autoformer, FEDformer, ETSformer, Non‑stationary Transformer, Informer, Reformer, DLinear, LightTS 등과 비교.
    - 대부분의 설정(데이터셋 × 예측 길이)에서 MSE/MAE 기준 **약 80% 이상 케이스에서 SOTA**를 달성.[^1_2]
2. **Short‑term forecasting (M4)**
    - N‑BEATS, N‑HiTS, ETSformer, Transformer 계열, MLP 계열과 비교.
    - SMAPE, MASE, OWA 기준으로 **가장 낮은 OWA(0.851)**를 기록해, 복잡한 대규모 마케팅 시계열(M4)에서 최고 성능을 보인다.[^1_2]
3. **Imputation**
    - ETT, Electricity, Weather에서 랜덤 마스킹 비율 12.5–50%.
    - 모든 마스킹 비율에 대해 MSE/MAE가 다른 Transformer/MLP 계열보다 크게 낮으며, 특히 MLP 기반 DLinear/LightTS가 크게 열화되는 상황에서도 TimesNet은 안정적이다.[^1_2]
4. **Classification (UEA Multivariate Archive 일부 10개 데이터셋)**
    - Rocket, Flowformer, 다양한 Transformer 및 MLP, RNN, TCN 등과 비교.
    - 평균 정확도 73.6%로, Rocket(72.5%)과 Flowformer(73.0%)를 약간 상회하며, 단순 MLP 구조(DLinear)는 67.5%로 크게 낮다.[^1_2]
5. **Anomaly detection** (SMD, MSL, SMAP, SWaT, PSM)
    - reconstruction 기반 설정에서 F1 점수가 Autoformer, FEDformer, ETSformer, Anomaly Transformer 등보다 높고, 평균 F1 86%대(TimesNet + ResNeXt 구성) SOTA를 달성.[^1_2]

> 결론적으로, 하나의 백본으로 5개 태스크를 모두 커버하면서, 기존 task‑specific 모델들보다 전반적으로 높은 성능을 보이는 것이 TimesNet의 가장 큰 경험적 메시지다.[^1_2]

***

### 2.4 한계와 비판적 관점

논문에서 직접적으로 “한계”를 길게 적지는 않지만, 설계와 실험을 보면 다음과 같은 제약과 비판점을 추론할 수 있다.[^1_2]

1. **명시적 주기 가정과 FFT 기반 주기 추출 의존**
    - 시계열에 “상당히 뚜렷한” 주기성이 있다는 가정을 암묵적으로 깔고 있다.
    - 일부 비주기적이거나 강한 non‑stationarity를 가진 데이터에서는 FFT 기반 토프‑k 주파수 선택이 최선이 아닐 수 있다. 비주기적일 때는 사실상 intraperiod가 “거의 원래 1D와 같은 구조”가 되어 모델 이점이 줄어들 가능성이 있다.[^1_2]
2. **조금 복잡한 구조와 모듈 수**
    - Period 연산, 여러 2D 리쉐이프, Inception, 다시 1D로 복원, 주파수‑가중 합산까지 한 Block에서 상당수 연산을 수행한다.
    - MLP 기반 간단 모델(DLinear, Temporal Linear Net 계열 등)에 비해 구조가 복잡하며, 일부 설정에서는 단순 모델이 경쟁력을 유지한다는 후속 연구도 보고되고 있다.[^1_5][^1_9]
3. **2D 비전 백본 의존의 양날의 검**
    - 2D 커널이 intra/inter‑period locality를 잘 포착하는 장점이 있지만,
    - 시계열 도메인 구조에 꼭 맞는 inductive bias인지, 혹은 2D 시각 구조를 무리하게 이식한 것인지에 대한 이론적 분석은 제한적이다.
4. **대규모 self‑supervised pretraining은 아직 초기**
    - 논문 후반에서 “향후 대규모 pre‑training backbone으로 확장”을 제안하지만, 실제로 기계 번역/언어 모델 수준의 거대 pretraining은 수행되어 있지 않다.[^1_2]

***

## 3. 모델의 일반화 성능 향상 가능성 (중점)

TimesNet의 **generalization 관점**에서 중요한 포인트는 다음 세 가지다.

### 3.1 2D 변환에 따른 표현력 증가와 task‑general inductive bias

1. **분리된 intra/inter‑period 구조**
    - 1D 상에서 긴 시퀀스를 직접 attention/conv로 처리하는 대신,
주기 $p_i$마다 2D 텐서 $(\text{period index}, \text{phase})$ 구조를 만들기 때문에
“한 주기 내부의 로컬 패턴(intraperiod)”과 “주기 간의 추세(interperiod)”를 자연스럽게 분리해 본다.[^1_2]
    - 이는 forecasting, reconstruction, classification 등 여러 과제에 공통적으로 유효한 **도메인 디펜던트 inductive bias**로 작용한다.
2. **멀티스케일 2D 커널**
    - Inception은 서로 다른 kernel size를 병렬로 사용해, 짧은 변동과 긴 변동을 모두 잡는다.
    - 이는 “다양한 horizon에서의 일반화”에 도움을 준다. 예: 96→720 step long‑term forecasting에서도 성능이 안정적이다.[^1_2]
3. **공유 Inception 모듈**
    - $k$개 주기에 대해 같은 2D 블록을 공유함으로써 파라미터 수가 과도하게 증가하지 않고,
    - 서로 다른 주기에서 학습된 패턴이 공유되므로 **데이터 효율과 generalization**에 유리하다.[^1_2]

### 3.2 CKA 분석: 과제별로 “적절한 수준의 계층 표현”

논문은 layer 간 표현 유사도를 CKA(Centered Kernel Alignment)로 측정해, 각 과제별로 “좋은 generalization”이 요구하는 표현 특성이 다름을 보인다.[^1_2]

- Forecasting, anomaly detection: 잘 generalize하는 모델일수록 **layer 간 표현 유사도가 높고**, 상대적으로 low‑level 변동을 유지하는 경향. TimesNet은 여기서 높은 CKA와 좋은 성능을 동시에 보인다.[^1_2]
- Imputation, classification: 오히려 **layer 간 표현이 많이 달라지는 (CKA 낮음) hierarchical representation**이 성능과 상관성이 높다. TimesNet은 이 과제들에서 FEDformer보다 더 “깊이 구분되는 표현”을 형성한다.[^1_2]

이 분석은 TimesNet이 하나의 구조로 서로 다른 태스크에서 “필요한 표현 구조”를 자연스럽게 학습할 수 있음을 보여주는 간접적인 증거로 볼 수 있다.

### 3.3 Mixed dataset 실험: 다른 주기/도메인을 함께 학습했을 때

논문 부록에서는 ETTh1/2, ETTm1/2 네 데이터셋(서로 다른 sampling 주기)들을 섞어서 **하나의 모델을 joint training**하고, 각 데이터셋 성능을 비교한다.[^1_2]

- Autoformer/FEDformer는 일부 데이터셋에서 unified training 시 성능이 오히려 떨어지는 반면,
- TimesNet은 네 데이터셋 모두에서 “개별 학습 대비 동등 혹은 향상된 성능”을 보이는 경향이 있다.[^1_2]

이는 서로 다른 주기 구조를 가진 데이터 간에도 TimesNet의 2D 모듈화 구조가 **공통 패턴을 공유하면서도 dataset‑specific 패턴도 유지**할 수 있음을 시사한다.
→ “foundation‑model‑style time‑series backbone”으로 확장했을 때의 generalization 가능성이 크다는 근거가 된다.[^1_8][^1_2]

### 3.4 2020년 이후 관련 최신 연구와의 비교 속에서 본 일반화

2020년 이후 시계열 연구의 몇 가지 큰 흐름과 비교하면 TimesNet의 generalization 특징은 다음처럼 위치 지을 수 있다.

1. **Transformer 계열 (Autoformer, FEDformer, ETSformer 등)**[^1_9][^1_2]
    - Autoformer/FEDformer: period‑based decomposition, frequency‑domain sparse attention 등으로 긴 horizon forecasting 성능을 올렸으나, classification/imputation/anomaly처럼 고차 표현이 필요한 과제에서는 제한적인 경우가 많다.[^1_2]
    - TimesNet은 **frequency를 사용하면서도 attention 대신 2D conv‑style 구조**를 씀으로써, 다양한 태스크에서 안정적 generalization을 확보했다는 장점이 있다.
2. **MLP / Linear 계열 (DLinear, LightTS, 최근 Temporal Linear Net 등)**[^1_5][^1_2]
    - 이들은 놀랄 정도로 간단한 선형/MLP 구조로 많은 forecasting 벤치마크에서 강한 성능과 좋은 generalization을 보여준다.
    - 그러나 imputation, classification, anomaly 같은 high‑level representation tasks에서는 TimesNet이 크게 앞선다.[^1_2]
    - TimesNet은 구조는 더 복잡하지만, “다양한 태스크와 도메인에서의 일반화”라는 측면에서 foundation model 후보 역할을 하며, 단순 모델과의 상보성을 형성한다.
3. **Large time‑series foundation models (예: MOMENT, STEP 등)**[^1_8][^1_9]
    - MOMENT(2024) 등은 대규모 시계열 데이터로 pretraining을 수행하는 “시계열 전용 LLM/ViT”에 가까운 프레임워크를 제안한다.
    - TimesNet은 그 이전 세대의 연구로, 아직 massive pretraining은 하지 않았지만, **구조 자체가 vision backbone을 바로 끼워 넣을 수 있고, mixed dataset에서도 잘 generalize**한다는 점에서 이런 foundation 모델의 백본으로 사용하기 좋은 형태이다.[^1_8][^1_2]

요약하면, TimesNet은 “한 태스크에서의 극한 성능”보다는 “여러 태스크·도메인에 걸친 robust generalization”에 초점을 맞춘 설계로 볼 수 있고, 최근의 foundation‑style·pretraining 흐름과 잘 결합될 여지가 크다.

***

## 4. 앞으로의 연구 영향과, 향후 연구 시 고려할 점

### 4.1 연구적 영향

1. **시계열 ↔ 비전 백본의 연결**
    - 1D 시계열을 2D 텐서로 구조화하여 ResNet, ResNeXt, ConvNeXt, Swin 등 임의의 2D 백본을 사용할 수 있게 함으로써,
    - 비전에서 축적된 설계·pretraining 결과를 시계열에 재활용할 수 있는 길을 열었다.[^1_2]
2. **“Temporal 2D‑variation”이라는 새로운 관점**
    - 시계열 변동을 “주기 내/주기 간” 행렬 구조로 명시적으로 표현하는 아이디어는 forecasting 외에 representation learning, contrastive/self‑supervised pretraining, generative modeling(예: diffusion‑style)에도 응용 가능하다.
3. **Task‑general backbone에 대한 실증**
    - 하나의 구조로 forecasting, imputation, classification, anomaly detection을 모두 커버하면서 SOTA에 근접/초과하는 결과를 보여,
    - “시계열을 위한 foundation backbone” 연구의 필요성과 가능성을 뒷받침한다.[^1_8][^1_2]

### 4.2 향후 연구 시 고려할 점 및 구체적 아이디어

1. **대규모 self‑supervised pretraining으로 확장**
    - TimesNet 구조를 유지하되, MOMENT/STEP류의 대규모 시계열 코퍼스 위에서 masked reconstruction, contrastive, forecasting‑as‑pretext 등으로 pretraining을 수행할 수 있다.[^1_9][^1_8]
    - 특히 2D 텐서 상에서의 self‑supervised objective (patch masking, jigsaw, contrastive patch discrimination 등 비전에서 쓰인 기법)를 시계열에 직접 이식해볼 수 있다.
2. **주기 추출과 2D 매핑의 학습형(learnable) 확장**
    - 현재는 FFT + top‑k로 주기를 구하지만,
    - (a) 주기 탐색을 neural spectral layer나 differentiable periodogram으로 대체,
    - (b) 주기 수 $k$ 및 $(f_i, p_i)$ 자체를 end‑to‑end로 학습시키는 구조를 탐색할 수 있다.
    - distribution shift를 다루는 최근 연구(예: intra/inter‑series shift modeling)와 결합해, time‑varying 주기 구조를 동적으로 추적하는 방향도 가능하다.[^1_4]
3. **비주기적·heavy‑tailed 데이터에 대한 robust 변형**
    - 금융, 이벤트 로그 등 명확한 주기성이 약한 데이터에서는 FFT 기반 top‑k가 노이즈에 민감할 수 있다.
    - wavelet 변환, multi‑resolution analysis, 또는 time‑varying frequency 추정법을 도입해, “pseudo‑period” 혹은 local pattern 길이를 adaptively 결정하는 방법이 필요하다.
4. **단순 모델과의 하이브리드**
    - DLinear, Temporal Linear Net(TLN) 등 linear 계열 모델은 구조는 단순하지만 forecasting generalization이 매우 뛰어나므로,[^1_5]
    - TimesNet의 2D 모듈을 “잔차(residual) 비선형 모듈”로 쓰고, base trend/seasonality는 선형 계열이 담당하는 hybrid 구조가 유망하다.
5. **Uncertainty, causal modeling, anomaly‑aware representation**
    - 현재 TimesNet은 주로 point estimation 및 reconstruction에 초점을 둔다.
    - Bayesian/variational 확장을 통해 불확실성 추정, causal structure 탐색(예: time‑varying VAR, Granger networks),[^1_7]
    - 그리고 anomaly detection 시 “정상‑패턴 manifold”와의 거리 기반 metric을 2D 공간에서 직접 정의하는 방법을 탐구할 수 있다.
6. **실제 응용에서의 효율/배치 전략**
    - 시간축 길이가 매우 긴 산업/IoT 로그에서는 FFT와 2D 변환 비용이 커질 수 있다.
    - sliding‑window FFT, online 업데이트, 또는 low‑rank FFT approximation을 이용한 경량 버전 TimesNet 설계가 필요하다.

***

### 정리

- TimesNet은 “multi‑periodicity → 1D→2D 변환 → 2D 커널로 temporal 2D‑variation 모델링”이라는 뚜렷한 아이디어를 바탕으로, 다양한 시계열 과제에서 좋은 generalization을 보여주는 **task‑general backbone**이다.[^1_2]
- 2020년 이후 Transformer, MLP, foundation‑style 모델 흐름 속에서, TimesNet은 비전 백본을 시계열에 직접 이식하고 주기 기반 2D 구조를 명시화한 점에서 독특한 위치를 차지하며, 향후 대규모 self‑supervised pretraining과 hybrid 설계의 핵심 빌딩 블록이 될 가능성이 크다.[^1_5][^1_8][^1_2]
<span style="display:none">[^1_1][^1_3][^1_6]</span>

<div align="center">⁂</div>

[^1_1]: 2210.02186v3.pdf

[^1_2]: https://arxiv.org/pdf/2210.02186.pdf

[^1_3]: https://arxiv.org/pdf/2405.19647.pdf

[^1_4]: https://arxiv.org/html/2407.13194v1

[^1_5]: https://arxiv.org/html/2410.21448v1

[^1_6]: https://arxiv.org/html/2412.10859v1

[^1_7]: https://arxiv.org/pdf/2302.02476.pdf

[^1_8]: https://arxiv.org/pdf/2402.03885.pdf

[^1_9]: https://arxiv.org/pdf/2206.09113.pdf

