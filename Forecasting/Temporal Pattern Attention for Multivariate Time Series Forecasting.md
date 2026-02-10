# Temporal Pattern Attention for Multivariate Time Series Forecasting
## 1. 핵심 주장과 주요 기여 (간결 요약)

이 논문은 다변량 시계열(multivariate time series) 예측에서, 기존 “시점 기반” 어텐션이 여러 변수 간 상호작용과 장기 주기 패턴을 잘 잡지 못한다는 점을 지적하고, **변수(feature)별 주기 패턴**에 주목하는 새로운 어텐션 메커니즘인 Temporal Pattern Attention(TPA)을 제안합니다.[^1_1][^1_2]
RNN(LSTM)의 은닉 상태 시퀀스에 1D CNN 필터를 적용해 “주파수 영역”에 가까운 시간-불변 패턴 표현을 만든 뒤, 이들 패턴을 대상으로 **변수별 가중치를 학습하는 어텐션**을 도입하여, 전력·교통·환율·다성 음원 등 여러 실제 데이터에서 LSTNet 등 기존 SOTA보다 더 낮은 에러와 더 높은 상관계수를 달성합니다.[^1_3][^1_1]

주요 기여는 다음 네 가지입니다.[^1_1]

- 시점이 아니라 **변수(시계열 축)를 대상으로 하는 어텐션 개념**을 도입해, 어느 변수들이 예측에 중요한지 선택하도록 설계.
- CNN 필터로 각 변수의 장·단기 주기 패턴을 추출하여, 통상적인 어텐션이 잘 못 잡는 다중 스텝 패턴을 활용.
- 주기적/비주기적, 선형/비선형 MTS, 이산적인 폴리포닉 음악까지 다양한 데이터셋에서 SOTA 또는 그에 근접한 성능을 보임.
- 학습된 CNN 필터가 DFT 기저와 유사한 주파수 특성을 보여, **해석 가능한 주파수 기반 표현**으로 이해 가능함.[^1_1]

***

## 2. 문제 설정, 제안 방법(수식), 모델 구조, 성능 및 한계

### 2.1 논문이 다루는 문제

- 입력: 길이 $L$의 다변량 시계열

$$
X = \{x_1, x_2, \dots, x_{t-1}\}, \quad x_i \in \mathbb{R}^n
$$

여기서 $n$은 변수(시계열)의 개수입니다.[^1_1]
- 예측 과제: 윈도우 크기 $w$와 예측 지평 $\Delta$가 주어졌을 때,

$$
\{x_{t-w}, \dots, x_{t-1}\} \mapsto x_{t-1+\Delta}
$$

를 예측하는 회귀 문제로 정의합니다.[^1_1]
- 기존 한계:
    - 일반적인 RNN은 장기 의존성(예: 연 단위 주기)을 학습하기 어렵고, 기울기 소실·불안정 문제가 존재.[^1_1]
    - 기존 “시점 기반” 어텐션은 각 hidden state $h_i$ (시점 $i$)만을 가중 합하여,
        - 다변량에서 변수 간 상호의존성(“어떤 시계열이 중요한가?”)을 직접적으로 모델링하지 못하고,
        - 여러 시점에 걸친 주기 패턴(예: 매일 반복되는 24시간 주기)을 명시적으로 포착하기 어렵다는 한계를 지닙니다.[^1_1]


### 2.2 RNN(LSTM) 기반 베이스라인

은닉 상태는 일반적인 RNN 또는 LSTM으로 계산됩니다.[^1_1]

RNN 일반형:

$$
h_t = F(h_{t-1}, x_t), \quad h_t \in \mathbb{R}^m
$$

LSTM의 경우:

$$
\begin{aligned}
i_t &= \sigma(W_{xi} x_t + W_{hi} h_{t-1}) \\
f_t &= \sigma(W_{xf} x_t + W_{hf} h_{t-1}) \\
o_t &= \sigma(W_{xo} x_t + W_{ho} h_{t-1}) \\
\tilde{c}_t &= \tanh(W_{xg} x_t + W_{hg} h_{t-1}) \\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \\
h_t &= o_t \odot \tanh(c_t)
\end{aligned}
$$

여기서 $\sigma$는 시그모이드, $\odot$는 element-wise 곱입니다.[^1_1]

기존(“typical”) 어텐션은 과거 은닉 상태 집합

$$
H = [h_1, \dots, h_{t-1}] \in \mathbb{R}^{m \times (t-1)}
$$

에 대해, 각 시점별 스칼라 가중치 $\alpha_i$를 계산해 컨텍스트 벡터 $v_t$를 만듭니다.[^1_1]

$$
\alpha_i = \frac{\exp(f(h_i, h_t))}{\sum_{j=1}^{t-1} \exp(f(h_j, h_t))}, \quad
v_t = \sum_{i=1}^{t-1} \alpha_i h_i
$$

이는 “어느 시점이 중요한가?”에만 답하며, “어느 변수(시계열)가 중요한가?”에 대한 직접적인 구조는 아닙니다.[^1_1]

### 2.3 제안 방식: Temporal Pattern Attention

핵심 아이디어는 다음 두 단계입니다.[^1_1]

1. **CNN으로 각 변수의 시간 패턴(주기성·주파수 성분)을 추출**
2. 이 CNN 출력(“주파수 영역 표현”)에 대해 **변수 단위 어텐션(feature-wise attention)**을 수행

#### (1) CNN 기반 Temporal Pattern 추출

- RNN 은닉 상태 행렬:

$$
H \in \mathbb{R}^{n \times w}
$$

논문에서는 “이전 hidden state들의 행(row)이 변수 축을 따른다”는 관점으로 기술합니다.[^1_1]
- 1D CNN 필터 $k$개:

$$
C_j \in \mathbb{R}^{1 \times T}, \quad j = 1, \dots, k
$$

여기서 $T$는 필터 길이(기본적으로 $T = w$)입니다.[^1_1]

각 변수 $i$에 대해, 행 벡터 $H_i$ (길이 $w$)와 필터 $C_j$를 1D convolution:

$$
HC_{i,j} = \sum_{l=1}^{w} H_{i, (t-w-1+l)} \cdot C_{j, T-w+l}
$$

이를 통해

$$
HC \in \mathbb{R}^{n \times k}
$$

을 얻습니다. $HC_i \in \mathbb{R}^k$는 변수 $i$의 CNN 기반 temporal 패턴 표현(“frequency-like feature”)입니다.[^1_1]

#### (2) 변수(feature) 단위 어텐션

현재 시점의 RNN 은닉 상태 $h_t \in \mathbb{R}^m$과 각 변수의 CNN 표현 $HC_i$를 비교하는 scoring 함수:

$$
f(HC_i, h_t) = HC_i^\top W_a h_t,\quad W_a \in \mathbb{R}^{k \times m}
$$

어텐션 가중치는 softmax 대신 **sigmoid**를 사용해 각 변수를 독립적으로 선택·비선택 가능하도록 합니다.[^1_1]

$$
\alpha_i = \sigma(f(HC_i, h_t))
$$

컨텍스트 벡터는 변수 축에 대한 가중 합:

$$
v_t = \sum_{i=1}^{n} \alpha_i HC_i \in \mathbb{R}^k
$$

마지막으로, RNN 은닉 상태와 CNN 기반 컨텍스트를 선형 결합하여 최종 hidden representation:

$$
h'_t = W_h h_t + W_v v_t
$$

$$
y_{t-1+\Delta} = W_{h'} h'_t
$$

여기서 $W_h \in \mathbb{R}^{m \times m}, W_v \in \mathbb{R}^{m \times k}, W_{h'} \in \mathbb{R}^{n \times m}$입니다.[^1_1]

정리하면:

- RNN: 시간 축의 비선형 동역학/단기 의존성 모델링
- CNN: 윈도우 내 주기 패턴(“주파수 영역”) 추출
- 어텐션: **어떤 변수의 어떤 패턴이 현재 예측에 중요한지**를 선택


### 2.4 전체 모델 구조

구조적 관점에서 보면, TPA-LSTM은 다음 블록들로 구성됩니다.[^1_1]

1. **입력 및 전처리**
    - 고정 길이 윈도우 $\{x_{t-w}, \dots, x_{t-1}\}$ 사용
    - 각 시계열을 자체 최대값으로 정규화하거나, 전체 데이터 최대값으로 정규화하는 두 가지 스킴 중 선택.[^1_1]
2. **LSTM 인코더**
    - 각 시점 $x_\tau$에 대해 LSTM을 적용해 $h_\tau$ 계산
    - 최근 $w$개의 은닉 상태로 행렬 $H$ 구성
3. **CNN/Temporal Pattern Attention 모듈**
    - $H$의 행(변수 축)에 1D CNN 필터 $C_j$를 적용 → $HC$
    - $h_t$와 $HC$를 이용해 변수별 가중치 $\alpha_i$ 산출
    - 컨텍스트 $v_t = \sum_i \alpha_i HC_i$, 최종 표현 $h'_t$ 생성
4. **출력층 + (선형) AR 컴포넌트**
    - 예측값:

$$
y_{t-1+\Delta} = W_{h'} h'_t + \text{(선형 AR 성분)}
$$

- LSTNet과 유사하게, 전통적 AR(autoregression)을 추가해 neural net 스케일 민감도 문제를 완화합니다.[^1_1]
5. **손실 함수 및 학습**
    - 연속값 MTS: 절대 오차 기반 손실(RAE, RSE 등) + Adam 최적화.[^1_1]
    - 다성 음악: 이진(0/1) note on/off를 위해 cross-entropy(negative log-likelihood) 사용.[^1_1]

***

## 3. 성능 향상과 한계, 일반화 성능 관점

### 3.1 실험 결과 요약 (성능 향상)

#### (1) 일반 MTS 데이터 (Solar, Traffic, Electricity, Exchange Rate)

- 비교 대상:
    - 전통: AR, VAR(LRidge, LSVR), GP, SETAR
    - 딥러닝: LSTNet-Skip, LSTNet-Attn 등.[^1_1]
- 지표: RAE, RSE(정규화된 MAE/RMSE), CORR.[^1_1]

핵심 결과:[^1_1]

- 네 개 데이터셋(특히 **Traffic, Electricity**)에서 대부분의 horizon(3,6,12,24 step)에 대해 TPA-LSTM이 **LSTNet 계열과 전통 기법 모두보다 낮은 RSE·RAE, 높은 CORR**을 기록.
- 주기성이 거의 없는 Exchange Rate에서도 전통 기법보다 전반적으로 우수하며, LSTNet-Skip/Attn보다 안정적인 성능을 보임. 이는 제안 메커니즘이 **주기적 데이터에만 특화된 것이 아니라 보다 일반적인 비선형 패턴에도 적용 가능**함을 시사합니다.[^1_1]

예: Electricity, 24-step horizon, RSE 기준에서 TPA-LSTM이 LSTNet-Skip, LSTNet-Attn, SETAR보다 더 낮은 값을 보입니다.[^1_1]

#### (2) 폴리포닉 음악 데이터 (MuseData, LPD-5-Cleansed)

- 비교 대상: LSTM, LSTM+Luong 어텐션.[^1_1]
- 지표: Precision, Recall, F1.

결과:[^1_1]

- MuseData: TPA-LSTM이 F1에서 LSTM(0.7495)과 Luong 어텐션(0.6207)을 모두 상회하는 0.7633을 기록.
- LPD-5-Cleansed: TPA-LSTM F1=0.7897로, LSTM(0.7805), Luong(0.7756)보다 우수.
- Validation loss 곡선에서도 동일 파라미터 규모에서 **학습 수렴 속도와 최종 손실 모두에서 우위**를 보입니다.[^1_1]

이는 **연속값 MTS뿐 아니라 고차원 이진 시계열(음표 on/off)**에도 제안 메커니즘이 잘 일반화된다는 증거입니다.

### 3.2 일반화 성능 향상 메커니즘 (이론·실험적 근거)

논문은 “일반화”를 이론적으로 분석하지는 않지만, 몇 가지 설계 요소와 실험이 **일반화 가능성**을 뒷받침합니다.[^1_1]

1. **주파수(패턴) 기반 표현: CNN 필터 vs DFT**
    - Traffic 데이터에 대해, 학습된 CNN 필터들을 평균한 후 DFT(평균 Discrete Fourier Transform, avg-DFT)를 취하면, 원 데이터(avg-DFT)의 주요 피크(24, 12, 8, 6시간 주기 등)와 CNN 필터들의 주파수 응답 피크가 거의 일치.[^1_1]
    - 이는 CNN 필터가 **DFT 기저와 유사한 역할**을 하며, 주기성과 같은 전역 패턴을 압축 표현함을 의미. 전역 패턴을 명시적으로 포착하면 **노이즈와 단기 변동에 덜 민감**해져 일반화에 유리합니다.
2. **변수(feature) 단위 어텐션 → 불필요한 변수 억제**
    - Luong 어텐션은 모든 hidden state를 같은 차원에서 평균하면서, 예측에 도움이 안 되는 변수의 정보까지 함께 섞게 됩니다.[^1_1]
    - 제안 방식은 CNN 출력 $HC_i$에 대해 변수별 스칼라 $\alpha_i$를 곱하므로, **유의미한 변수만 강조하고 노이즈 변수는 억제**할 수 있습니다.
    - 토이 실험에서, 변수 수 $D$가 증가할수록 Luong 어텐션 LSTM은 LSTM(어텐션 없음)보다도 성능이 나빠지지만, TPA-LSTM은 높은 차원에서도 안정적인 loss를 유지합니다.[^1_1]
→ 변수 수 증가에 따른 차원의 저주에 대해 더 견고한 구조임을 시사.
3. **시그모이드 기반 multi-hot 선택 vs softmax 기반 single-hot 선택**
    - $\alpha_i = \sigma(f(HC_i, h_t))$를 사용함으로써, 동시에 여러 변수의 패턴을 활용할 수 있습니다.[^1_1]
    - 어블레이션(softmax vs sigmoid, CNN on/off)에서, 특히 MuseData(복잡/비주기적)에서 **sigmoid+position-attention 구조가 가장 낮은 NLL**을 보여, 보다 일반적인 상황에서 우수함을 보입니다.[^1_1]
    - softmax는 “한두 개 변수만” 강하게 선택하는 경향이 있어, 복잡한 상호작용이 있는 다변량 데이터에서 일반화가 떨어질 수 있습니다.
4. **토이 예제에서의 interdependency 활용**
    - 독립된 사인파 vs 상호 섞인 사인파 실험에서,
        - Luong 어텐션 및 LSTM은 interdependency를 추가해도 loss가 거의 개선되지 않음.
        - TPA-LSTM은 interdependency가 있는 경우 **loss가 더 크게 감소**하는데, 이는 변수 간 종속성을 실제로 활용하고 있음을 의미합니다.[^1_1]
    - 변수 간 상호작용을 잘 활용할수록, 새로운 환경/테스크로의 전이(일반화) 가능성이 커집니다.
5. **여러 유형의 데이터에 대한 일관된 우수성**
    - 크기(수백 KB ~ 수백 MB), 주기성(강/약/없음), 값의 타입(연속/이산) 등이 다양한 데이터셋에서 **단일 구조로 일관되게 상위 성능**을 보임.[^1_1]
    - 이는 특정 도메인(예: 전력 수요) 특화가 아니라, 구조 자체가 일반적인 MTS에 잘 맞는다는 경험적 근거입니다.

### 3.3 한계 및 비판적 논의

논문 자체 및 그 이후의 문헌을 바탕으로 보면, 다음과 같은 한계가 있습니다.

1. **RNN 기반 구조의 한계 유지**
    - LSTM 기반이라 긴 시퀀스(수천 스텝 이상)에서 Transformer류에 비해 효율과 gradient 흐름에서 불리할 수 있습니다.
    - CNN+어텐션이 장기 패턴을 어느 정도 보완하지만, 최근의 Long-horizon Transformer(FEDformer, TimesNet 등)와 직접 비교 실험은 없습니다.[^1_4][^1_5]
2. **CNN 필터 길이 및 윈도우 사이즈에 대한 의존성**
    - 주기 패턴을 포착하기 위해 충분히 긴 윈도우 $w$와 필터 길이 $T$가 필요하며, grid search로 수동 튜닝합니다.[^1_1]
    - 데이터 주기가 변화하거나 다중 스케일(시간 척도)이 섞인 경우, 고정 filter 길이로는 충분히 유연하지 않을 수 있습니다. 이후 Multi-scale CNN/patch 기반 Transformer 등이 이 점을 개선합니다.[^1_6]
3. **이론적 일반화 보장은 부족**
    - CNN 필터와 DFT 기저의 유사성 등 흥미로운 경험적 분석은 있지만, attention+CNN 구조가 왜 generalization bound를 개선하는지에 대한 이론은 제공되지 않습니다.[^1_1]
    - 최근 Transformer/MLP 기반 단순 모델(Zeng et al., 2023 등)이 복잡한 attention 구조 없이도 강력한 성능을 내는 결과와 비교해, 구조 복잡성 대비 이득을 더 면밀히 분석할 필요가 있습니다.[^1_7]
4. **최근 Transformer 계열과의 직접 비교 부재**
    - 논문 시점(2019) 기준으로는 합리적인 비교지만, 이후 DSANet, FEDformer, TimesNet, iTransformer, PatchTST, HTMformer, MultiPatchFormer 등 다양한 MTS용 Transformer가 등장했습니다.[^1_5][^1_8][^1_4][^1_6]
    - 이들과의 직접 비교 없이 “state-of-the-art”를 주장하는 것은 2020년 이후 관점에서는 한계로 볼 수 있습니다.

***

## 4. 2020년 이후 관련 최신 연구와의 비교 분석

여기서는 **Temporal Pattern Attention(TPA)**가 이후 연구 흐름에서 어떤 위치를 차지하는지, 특히 “주파수/패턴 기반 attention”과 “변수-시점 분리 attention” 측면에서 살펴봅니다.

### 4.1 TPA와 직접 후속/파생 연구

1. **TPA-LSTM을 응용한 연구 (예: 주가 예측)**
    - Hindawi의 한 연구는 TPA-LSTM을 사용해 **주가 지수 예측**에서 RNN, CNN, LSTNet 등 전통·딥러닝 기반 방법보다 더 낮은 예측 오차를 얻었다고 보고합니다.[^1_9]
    - 이는 TPA 구조가 금융 시계열처럼 **약한 주기성 + 장·단기 혼합 정보**를 가진 도메인에도 유효함을 보여줍니다.
2. **패턴 지향 어텐션/패턴 기반 프로젝터**
    - 최근 PRformer (Pattern-oriented Attention)와 같은 모델은 multivariate time series에서 **패턴 단위 representation을 학습하고, 패턴 기반 projector와 attention을 결합**합니다.[^1_10]
    - 개념적으로, TPA가 “CNN 필터를 통해 패턴 공간으로 투영한 후 feature-wise attention을 하는 구조”라면, PRformer류는 이를 Transformer 맥락으로 일반화한 것으로 볼 수 있습니다.
3. **해석 가능한 CNN+어텐션 구조**
    - Pantiskas et al. (2020)은 Temporal Attention CNN을 사용해 multivariate time series forecasting에서 **어떤 시점이 중요한지를 시각화 가능한 어텐션 맵으로 제공**하면서, TCN 기반 성능은 유지합니다.[^1_11][^1_12]
    - 이들은 주로 “시점 기반” 어텐션에 초점을 두지만, CNN으로 패턴을 추출한 후 주의를 두는 점에서 TPA와 유사한 철학을 공유합니다.

### 4.2 주파수/스펙트럼 기반 Attention으로의 확장

TPA의 “CNN→frequency-domain-like representation→attention” 아이디어는 이후 **명시적인 주파수 변환 + attention**으로 확장됩니다.

1. **Frequency Spectrum Attention (FSatten), Scaled Orthogonal Attention (SOatten)**
    - Wu \& Zhenjiang (2024)의 “Revisiting Attention for MTSF”는 **Fourier Transform(FFT)**로 시계열을 주파수 영역으로 옮긴 뒤, 그 위에서 Multi-head Spectrum Scaling(MSS)을 수행하는 FSatten을 제안합니다.[^1_13][^1_14]
    - 이는 TPA와 유사하게 **주파수 영역에서 attention을 수행해 주기성을 포착**한다는 점에서 철학적 연속선 상에 있습니다.
    - SOatten은 주파수 기반이 아닌 orthogonal embedding과 head-coupling convolution을 통해 더 일반적인 의존성을 학습하는데, 이는 TPA의 CNN+attention이 특정 주파수 기반 구조에 묶여 있는 것과 대조됩니다.[^1_13]
2. **FEDformer, FreqTSF, TimesNet 등 주파수 도메인 모델**
    - FEDformer(Zhou et al., 2022): FFT를 활용해 **self-attention을 주파수 도메인에서 수행**하여 복잡도를 줄이고 주기 패턴을 잘 포착.[^1_4]
    - FreqTSF(2024): Frequency Cross Attention을 도입해 주파수 도메인에서 attention을 수행하면서, intra-/inter-variable variation을 더 잘 포착하도록 설계합니다.[^1_7]
    - TimesNet(2023): 시계열을 여러 주파수 도메인으로 분할하고, 각 주파수 성분에서 temporal dynamics를 학습.[^1_5]
    - 이들 모두 **“frequency domain이 전역 의존성, 주기성, 노이즈 억제에 유리하다”는 가정**을 공유하며, 이는 TPA가 CNN 필터와 DFT 유사성을 보여주며 시사한 방향과 정확히 맞닿아 있습니다.[^1_1]

정리하면, TPA는 **주파수/패턴 기반 attention으로 가는 초창기 CNN-기반 접근**으로 볼 수 있고, 이후 연구들은 이를 **명시적인 FFT/DFT 기반 attention, Transformer 아키텍처**로 확장·정교화했다고 볼 수 있습니다.

### 4.3 변수/채널 축과 시간 축을 분리하는 최근 모델들

TPA는 **변수(feature) 축에 attention을 두는 것**이 핵심인데, 이후 많은 Transformer 계열 모델이 이 아이디어를 변형/확장합니다.

1. **채널 및 시간 축을 분리한 Transformer (예: HTMformer, Sentinel, MultiPatchFormer)**
    - HTMformer: 시계열 대부분의 정보가 시간 축에 있으므로, temporal feature extraction은 비-attention 모듈에 맡기고, attention은 **채널(변수) 간 상관관계**를 모델링하도록 설계합니다.[^1_15]
    - Sentinel, MultiPatchFormer 등도 temporal encoder와 channel-wise encoder를 분리해, 시간 및 변수 의존성을 각각 처리합니다.[^1_16][^1_6]
    - 이는 “시간 축은 CNN/patch/MLP로, 변수 축은 attention으로 처리”라는 TPA의 구조적 분해와 매우 흡사합니다.
2. **Gateformer, Crossformer, Scalable Transformers**
    - Gateformer, Crossformer 등은 temporal-wise attention과 variate-wise attention을 분리하거나 결합하여, multivariate 상호작용을 더 명시적으로 모델링합니다.[^1_17][^1_8][^1_5]
    - TPA는 RNN+CNN 기반이지만, “시계열 축과 변수 축을 분리하여 attention을 설계”한 초기 예시로 이해할 수 있습니다.
3. **Self-attention을 재해석/단순화한 연구**
    - 최근 연구는 attention 전체를 MLP로 근사할 수 있고, QKV projection과 score computation을 제거해도 성능 손실이 크지 않다고 보고합니다.[^1_18][^1_19]
    - 이런 맥락에서 TPA의 구조(“CNN+feature-wise gate-like attention”)는 더욱 단순화된 attention 설계로 이어질 수 있는 중간 단계로 볼 수 있습니다.

***

## 5. 앞으로의 연구에 대한 영향과 향후 고려 사항

### 5.1 향후 연구에 미치는 영향 (연구 방향)

1. **주파수/패턴 기반 representation의 중요성 부각**
    - TPA는 “time-domain hidden state 위에서만 attention을 돌린다”는 기존 관행에서 벗어나, **CNN을 통한 frequency-like 패턴 공간**을 attention의 입력으로 사용하는 설계를 보였습니다.[^1_1]
    - 이후 FEDformer, FSatten, FreqTSF, TimesNet 같은 모델들이 **명시적인 주파수 변환+attention**으로 확장된 점을 보면, TPA는 “주파수 영역에서 attention을 하는 것이 효과적”이라는 방향성을 경험적으로 제시한 사례로 평가할 수 있습니다.[^1_4][^1_7][^1_13]
2. **변수(feature)-wise attention의 유용성 입증**
    - TPA는 “어떤 시점이 중요한가?”(time-wise) 대신 “어떤 변수의 어떤 패턴이 중요한가?”(feature-wise)를 직접 모델링했습니다.[^1_1]
    - 이는 이후 channel-wise attention, variate-wise attention, graph-based variable modeling 등으로 일반화되었고, 최신 Transformer 기반 MTS 모델들에서 사실상 표준 설계 패턴이 되었습니다.[^1_8][^1_16][^1_15]
3. **해석 가능성(interpretability)에 대한 기여**
    - CNN 필터를 DFT 기저처럼 해석하고, 각 변수의 attention weight를 통해 “어떤 변수의 어떤 주파수 패턴이 중요했는지”를 어느 정도 설명할 수 있습니다.[^1_1]
    - 이는 이후 **interactable/visualizable attention map**을 강조하는 연구들(Temporally attentive CNN, attention-based interpretability in industry MTS 등)에 영향을 미쳤습니다.[^1_20][^1_11]

### 5.2 앞으로 연구 시 고려할 점 및 연구 아이디어

연구자로서 TPA를 기반으로 후속 연구를 진행할 때 고려할 수 있는 포인트를 정리하면 다음과 같습니다.

1. **RNN 대신 Transformer/MLP와 결합한 TPA 변형**
    - TPA의 CNN+feature-wise attention 아이디어를 **Transformer backbone**(예: PatchTST, iTransformer, HTMformer) 위에 얹어,
        - Temporal encoder: patch-based CNN/TCN/MLP
        - Variate encoder: TPA-style CNN+feature-attention
구조를 만드는 것이 자연스러운 확장입니다.[^1_15][^1_8][^1_5]
    - 이렇게 하면 긴 시퀀스에서도 효율적이면서, 주파수 기반 패턴과 변수 간 상호작용을 동시에 잘 모델링할 수 있습니다.
2. **명시적 주파수 변환(FFT) vs CNN 기반 암묵적 주파수 학습의 비교**
    - TPA는 CNN 필터가 DFT 기저를 “학습적으로 근사”하는 구조인데, FEDformer/FSatten은 **명시적으로 FFT를 사용**합니다.[^1_13][^1_4][^1_1]
    - 연구 과제:
        - CNN 기반 implicit frequency representation vs FFT-based explicit representation 간 **표현력·일반화·연산량** 비교
        - 두 방식을 하이브리드로 결합한 구조(예: frequency-aware CNN initialisation, frequency-regularized filters 등)
3. **다중 스케일 및 비정상(non-stationary) 패턴에 대한 확장**
    - 현재 TPA는 고정 윈도우 길이 $w$와 고정 길이 필터 $T$에 의존합니다.[^1_1]
    - 향후:
        - 여러 필터 길이(멀티스케일 CNN) 또는 wavelet-like filter 구조 도입
        - non-stationary 환경에서 필터를 동적으로 조정하는 gating/adapter 설계
        - 주파수 도메인에서 time-varying spectrum을 직접 모델링하는 구조(예: STFT 기반 attention)
4. **이론적 분석: generalization 및 robustness**
    - CNN+feature-wise attention 구조가 왜 **노이즈, missing data, covariate shift**에 견고한지에 대한 이론적 설명이 필요합니다.
    - 예를 들어,
        - frequency-domain representation이 high-frequency noise를 필터링하여 Rademacher complexity를 줄인다는 식의 분석,
        - feature-wise gating이 irrelevant variable에 대한 sensitivity를 줄이는 formal argument 등이 연구 주제가 될 수 있습니다.
    - 최신 survey/이론 논문들은 multivariate time series forecasting에서 CNN, MLP, Transformer 각각의 장단점을 정리하고 있어, 그 맥락에서 TPA류 구조를 재해석할 수 있습니다.[^1_21]
5. **비전형 도메인에의 적용 및 전이 학습**
    - 논문은 전력/교통/환율/음악에 적용했지만, 그 외에
        - 의료·웨어러블 센서 데이터(불규칙 샘플링, missingness 많음)
        - 산업/공정 데이터(변수 수 수백–수천, 그래프 구조 포함)
에서 TPA-style attention이 얼마나 확장 가능한지 체계적인 평가가 필요합니다.
    - 그래프 기반 MTS (graph neural network + attention)와의 결합도 자연스러운 연구 방향입니다.[^1_22][^1_23]

***

### 6. 요약적으로, 연구자로서의 take-away

- TPA 논문은 **“주파수/패턴 기반 표현 + 변수(feature)-wise attention”**라는 두 가지 중요한 아이디어를 RNN+CNN 구조로 깔끔히 구현한 초기 작업입니다.[^1_1]
- 이후 Transformer 및 명시적 주파수 도메인 attention으로 발전한 최근 문헌 흐름을 고려하면, TPA는 **frequency-domain attention과 variate-wise modeling 패러다임의 선행 사례**로 보는 것이 타당합니다.[^1_5][^1_4][^1_13]
- 향후 연구에서는
    - 더 강력한 백본(Transformer/MLP)과 결합,
    - 명시적 FFT/멀티스케일 주파수 표현과의 통합,
    - 이론적 일반화 분석과 고차원·비정상 데이터로의 확장
을 중점적으로 탐구하는 것이 의미 있을 것입니다.
<span style="display:none">[^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46]</span>

<div align="center">⁂</div>

[^1_1]: 1809.04206v3.pdf

[^1_2]: https://arxiv.org/pdf/1809.04206.pdf

[^1_3]: https://arxiv.org/abs/1809.04206

[^1_4]: https://arxiv.org/html/2505.04158v1

[^1_5]: https://arxiv.org/html/2408.04245v1

[^1_6]: https://www.nature.com/articles/s41598-024-82417-4

[^1_7]: https://arxiv.org/html/2407.21275v1

[^1_8]: https://arxiv.org/html/2503.17658v1

[^1_9]: https://www.hindawi.com/journals/am/2020/8831893/

[^1_10]: https://dl.acm.org/doi/abs/10.1145/3712606

[^1_11]: https://www.semanticscholar.org/paper/Interpretable-Multivariate-Time-Series-Forecasting-Pantiskas-Verstoep/d0567c2f1c491079fdd4e0414033a06ee16c6f4f

[^1_12]: https://research.vu.nl/ws/portalfiles/portal/121454922/tacn_vu.pdf

[^1_13]: https://arxiv.org/abs/2407.13806

[^1_14]: https://www.themoonlight.io/en/review/revisiting-attention-for-multivariate-time-series-forecasting

[^1_15]: https://arxiv.org/html/2510.07084v1

[^1_16]: http://arxiv.org/pdf/2503.17658.pdf

[^1_17]: https://arxiv.org/html/2505.00307v2

[^1_18]: http://arxiv.org/pdf/2410.24023.pdf

[^1_19]: https://arxiv.org/html/2511.03190v1

[^1_20]: https://www.semanticscholar.org/paper/Attention-Mechanism-for-Multivariate-Time-Series-to-Schockaert-Leperlier/3186b7e1fdb0589971f4062e53763ffffd6d389c

[^1_21]: https://arxiv.org/html/2502.10721v3

[^1_22]: https://ieeexplore.ieee.org/document/9219721/

[^1_23]: https://www.semanticscholar.org/paper/Multivariate-Time-Series-Forecasting-With-Dynamic-Jin-Zheng/b02a5aebd9598ddbb2b4021b5576cedb5519d436

[^1_24]: https://www.hindawi.com/journals/complexity/2020/8846608/

[^1_25]: https://ieeexplore.ieee.org/document/9206751/

[^1_26]: https://ieeexplore.ieee.org/document/9378408/

[^1_27]: https://ieeexplore.ieee.org/document/9129692/

[^1_28]: https://academic.oup.com/ije/article/49/6/1909/5923437

[^1_29]: https://osf.io/j57pk_v1

[^1_30]: https://iopscience.iop.org/article/10.1088/1741-2552/ab965b

[^1_31]: http://medrxiv.org/lookup/doi/10.1101/2020.11.16.20232868

[^1_32]: https://arxiv.org/pdf/2402.05370.pdf

[^1_33]: https://arxiv.org/html/2407.13806v1

[^1_34]: https://arxiv.org/pdf/2302.06683.pdf

[^1_35]: https://arxiv.org/pdf/1806.08523.pdf

[^1_36]: https://arxiv.org/pdf/2306.07114.pdf

[^1_37]: https://www.semanticscholar.org/paper/Temporal-pattern-attention-for-multivariate-time-Shih-Sun/5d1d53c671b20db116ba8c91c6446bb4757614da

[^1_38]: https://www.semanticscholar.org/paper/e91f8469e7f53dc678ece25a1b37ffe3560ab1fe

[^1_39]: https://arxiv.org/html/2505.00302v1

[^1_40]: https://arxiv.org/html/2511.11817v2

[^1_41]: https://www.sciencedirect.com/science/article/abs/pii/S0952197625020822

[^1_42]: https://dl.acm.org/doi/10.1007/s10994-019-05815-0

[^1_43]: https://www.scribd.com/document/961217771/s10994-019-05815-0

[^1_44]: http://www.arxiv.org/abs/1809.04206

[^1_45]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10007534/

[^1_46]: https://www.themoonlight.io/es/review/revisiting-attention-for-multivariate-time-series-forecasting

