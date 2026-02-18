# TKAN: Temporal Kolmogorov-Arnold Networks

TKAN 논문의 핵심 주장은 “KAN의 표현력·해석력 + LSTM의 장기 메모리”를 결합한 새로운 순환 구조로, 특히 장기 multi-step 시계열 예측에서 GRU/LSTM보다 더 높은 정확도와 더 안정적인 일반화(작은 분산, 과적합 감소)를 달성한다는 것입니다.[^1_1][^1_2]

***

## 1. 논문의 핵심 주장과 주요 기여

- Kolmogorov–Arnold Networks(KAN)의 “엣지 단위 1차원 B-스플라인 함수” 구조를 RNN에 접목한 Recurrent KAN(RKAN)을 정의하고, 이를 LSTM 게이트와 결합한 Temporal KAN(TKAN) 레이어를 제안합니다.[^1_3][^1_1]
- Binance 실거래(시가총액 상위 19개 코인 시가총액 기준 notional 거래량, 시간당 데이터)를 이용해 BTC multi-step(최대 15 스텝 ahead) 예측에서, 동일 구성의 GRU/LSTM보다 TKAN이 긴 예측 시점에서 더 높은 평균 $R^2$와 훨씬 작은 분산(훈련 안정성)을 보인다고 보고합니다.[^1_1]
- 실험적으로 TKAN이 짧은 구간(1–3 step)에서는 GRU와 비슷하지만, 중·장기(6 step 이상)에서는 GRU·LSTM·naive persistence를 모두 꾸준히 상회하며, 학습·검증 손실 곡선도 과적합 없이 안정적으로 수렴하는 것을 보여 “일반화 관점에서 유리한 구조”라는 점을 강조합니다.[^1_3][^1_1]

***

## 2. 해결하고자 하는 문제

### 2.1 문제 설정

논문이 겨냥하는 문제는 “현실 금융 시계열(암호화폐 거래 notional)에서 여러 스텝 앞을 한꺼번에 예측하는 multi-step time series forecasting”입니다.[^1_2][^1_1]
전통적인 ARMA/ARIMA 계열은 비선형·복잡 상호작용을 잘 잡지 못하고, MLP는 시퀀스 구조를 전혀 반영하지 못하며, vanilla RNN은 gradient 소실·폭주로 긴 의존성을 학습하기 어렵다는 한계를 갖습니다.[^1_1]

LSTM/GRU는 게이팅 덕분에 장기 의존성을 어느 정도 해결했지만,

- 시퀀스가 길어질수록 과적합·불안정성이 커지고,
- 선형 가중치 + 고정 활성함수 구조라 표현력이 한정적이며,
- multi-step horizon이 길어지면 예측 성능이 급격히 저하되는 문제가 있습니다.[^1_4][^1_1]


### 2.2 데이터·태스크 구체 설정

- 데이터: Binance에서 2020-01-01 ~ 2022-12-31 사이 시간당 notional 거래량(각 시점 19개 자산; BTC, ETH, ADA, XMR, EOS, MATIC 등).[^1_1]
- 입력: 과거 여러 시점의 19차원 시계열 윈도우.
- 출력: BTC notional의 미래 1, 3, 6, 9, 12, 15 스텝 ahead 값(멀티 아웃풋 회귀).
- 평가: RMSE를 loss로 최적화하고, 성능 평가는 $R^2$ 지표 사용.[^1_1]

이 태스크는 장기 패턴(계절성, 자가상관)과 급변(볼륨 급등락)이 함께 존재해, 일반적인 RNN이 과적합·불안정에 빠지기 쉬운 “일반화가 어려운” 문제 설정입니다.[^1_1]

***

## 3. 제안 방법: TKAN 수식 중심 설명

### 3.1 KAN 기본 구조

KAN은 Kolmogorov–Arnold 표현 정리에 기반해, 다변수 연속함수 $f(x_1,\dots,x_n)$를 1차원 함수 합성으로 표현합니다.[^1_3][^1_1]
KAN 레이어에서는 노드 간의 “엣지”마다 1차원 활성함수 $\phi_{l,j,i}(\cdot)$를 두고, 이 함수들을 B-스플라인 계수로 파라미터화합니다.[^1_5][^1_1]

하나의 KAN 레이어 $l$에 대해, 입력 벡터 $\mathbf{x}^{(l)} \in \mathbb{R}^{n_l}$, 출력 $\mathbf{x}^{(l+1)} \in \mathbb{R}^{n_{l+1}}$라 하면

$$
x^{(l+1)}_j = \sum_{i=1}^{n_l} \phi_{l,j,i}\!\big(x^{(l)}_i\big), \quad j=1,\dots,n_{l+1} 
$$

이고, 행렬 형태로는

$$
\mathbf{x}^{(l+1)} = \Phi_l(\mathbf{x}^{(l)}), 
$$

여기서 $\Phi_l$은 각 엣지의 B‑스플라인 함수 집합으로 정의된 “함수 행렬”입니다.[^1_3][^1_1]

이 구조 덕분에,

- 각 엣지는 고정 활성함수가 아니라 학습 가능한 1D 함수이며,
- 복잡한 비선형을 상대적으로 적은 파라미터로 표현할 수 있고,
- 함수가 B‑스플라인이므로 매끄럽고 Lipschitz 상수가 작아 과적합 완화에 유리하다는 이점이 있습니다.[^1_6][^1_7]


### 3.2 Recurrent KAN(RKAN) – 시계열용 순환 커널

시계열을 처리하려면 “과거 은닉 상태”를 현재 입력과 결합해야 하므로, 저자들은 KAN 레이어 앞에 RNN 스타일의 순환 커널을 도입해 RKAN 레이어를 정의합니다.[^1_1]

레이어 $l$에서 시점 $t$의 입력을 $x_t \in \mathbb{R}^d$, 레이어 별 서브메모리 벡터를 $h_{l,t} \in \mathbb{R}^{\text{KANout}}$이라 두면,

$$
s_{l,t} = W_{l,x} x_t + W_{l,h} h_{l,t-1}, 
$$

여기서

- $W_{l,x} \in \mathbb{R}^{\text{KANin} \times d}$: 현재 입력에 대한 가중,
- $W_{l,h} \in \mathbb{R}^{\text{KANin} \times \text{KANout}}$: 직전 서브메모리에 대한 가중입니다.[^1_1]

이전 단계에서 정의한 KAN 레이어 $\mathcal{K}_l$를 적용해

$$
o_{l,t} = \mathcal{K}_l(s_{l,t}), 
$$

그리고 레이어 내부 메모리 업데이트를

$$
h_{l,t} = W_{hh} h_{l,t-1} + W_{hz} o_{l,t}, 
$$

형태로 정의해, 과거 은닉 상태와 현재 KAN 출력을 선형 결합하는 구조를 취합니다.[^1_1]

핵심은 “순환 커널은 여전히 선형 + 합성 구조를 가지되, 비선형은 엣지 단위 B‑스플라인으로 이동”시켜,

- RNN의 장기 의존성 표현 능력,
- KAN의 고해상도 비선형 근사 및 해석력
을 동시에 확보하는 것입니다.[^1_3][^1_1]


### 3.3 TKAN: RKAN + LSTM 게이팅 결합

다음으로, 여러 RKAN 레이어의 출력을 하나의 벡터로 이어붙이고, 이를 LSTM 스타일의 게이팅 모듈로 감싸서 “Temporal KAN(TKAN) 레이어”를 만듭니다.[^1_1]

먼저 $L$개의 RKAN 레이어 출력 $s_{1,t},\dots,s_{L,t}$를

$$
r_t = \text{Concat}\big( s_{1,t},\dots,s_{L,t} \big) 
$$

로 결합합니다.[^1_1]

LSTM 게이트는 전형적인 형태를 따릅니다(논문에서는 활성함수 기호를 생략하고 “activation vector”로 지칭):[^1_1]

```math
\begin{aligned}
f_t &= W_f x_t + U_f h_{t-1} + b_f, \\
i_t &= W_i x_t + U_i h_{t-1} + b_i, \\
o_t &= W_o r_t + b_o, 
\end{aligned}
```

여기서 $f_t, i_t, o_t$는 각각 forget, input, output 게이트 활성 벡터입니다.[^1_1]

셀 상태와 최종 은닉 상태는

```math
\begin{aligned}
\tilde c_t &= W_c x_t + U_c h_{t-1} + b_c,  \\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde c_t,  \\
h_t &= o_t \odot \tanh(c_t),
\end{aligned}
```

로 업데이트됩니다.[^1_1]

요약하면,

- 시간 축 비선형 표현은 RKAN(KAN 기반)에서 담당,
- “어떤 정보를 얼마나 오래 유지·망각할지”는 LSTM 게이트에서 관리,
- 두 모듈을 한 블록(TKAN 레이어)로 묶어, 복잡한 비선형 시계열을 장기까지 안정적으로 모형화하도록 설계한 구조입니다.[^1_2][^1_1]

***

## 4. TKAN 전체 모델 구조

시계열 예측 모델로 사용할 때, 저자들은 TKAN·GRU·LSTM에 대해 동일한 3층 구조를 사용해 공정 비교를 수행합니다.[^1_1]

- 1층: 100 유닛의 순환 레이어(각각 TKAN/GRU/LSTM), `return_sequences=True`.
- 2층: 100 유닛의 순환 레이어, 마지막 hidden state만 반환.
- 출력층: 선형 dense 레이어, 유닛 수 = 예측할 time step 수(예: 15-step ahead이면 15).
- TKAN의 내부 KAN 서브레이어는 0–4차 B‑스플라인 5개를 사용.[^1_1]

학습 설정은 다음과 같습니다.[^1_1]

- 전처리:
    - 2주 이동 중앙값으로 나눈 후, 자산별 MinMax 스케일링(0–1 구간).
    - train:test = 80:20 (약 21k:5k 포인트).
- 최적화: Adam, RMSE loss, early stopping(검증 성능 6 epoch 정체 시 종료), plateau LR decay(3 epoch 정체 시 LR 1/2).[^1_1]

이렇게 구성된 모델은 1, 3, 6, 9, 12, 15 스텝 ahead multi-step 예측을 동시에 수행합니다.[^1_1]

***

## 5. 성능 향상과 한계

### 5.1 R² 성능 요약

논문에서 5회 반복 실험의 평균 $R^2$는 다음과 같습니다.[^1_1]

**Binance notional 예측에서 평균 $R^2$**


| 예측 horizon (step) | TKAN | GRU | LSTM | Naive(Last value) |
| :-- | :-- | :-- | :-- | :-- |
| 1 | 0.33736 [^1_1] | 0.36514 [^1_1] | 0.35553 [^1_1] | 0.29217 [^1_1] |
| 3 | 0.21227 [^1_1] | 0.20067 [^1_1] | 0.06122 [^1_1] | −0.06281 [^1_1] |
| 6 | 0.13784 [^1_1] | 0.08250 [^1_1] | −0.22584 [^1_1] | −0.33135 [^1_1] |
| 9 | 0.09803 [^1_1] | 0.08716 [^1_1] | −0.29058 [^1_1] | −0.45772 [^1_1] |
| 12 | 0.10401 [^1_1] | 0.01786 [^1_1] | −0.47322 [^1_1] | −0.51825 [^1_1] |
| 15 | 0.09512 [^1_1] | 0.03342 [^1_1] | −0.40443 [^1_1] | −0.55563 [^1_1] |

관찰할 수 있는 점은:

- 1-step에서는 GRU가 TKAN보다 아주 약간 우수하나, 차이는 작습니다.[^1_1]
- 3-step부터 TKAN ≥ GRU ≫ LSTM/naive이며, 특히 6-step 이후에는 TKAN의 $R^2$가 GRU보다 25% 이상 상대적 우위를 보입니다(예: 12-step에서 TKAN 0.104 vs GRU 0.018).[^1_1]
- LSTM은 6-step 이상에서 $R^2 < 0$로, 단순 “마지막 값 유지”보다도 나쁜 예측을 합니다.[^1_1]


### 5.2 안정성(표준편차) 및 학습 동역학

5회 반복 실험의 $R^2$ 표준편차는 TKAN이 GRU/LSTM보다 작아, 초기화·batch 샘플링에 따른 변동성이 가장 낮습니다.[^1_1]
또한 epoch에 따른 train/validation loss 곡선을 보면, TKAN은 두 곡선이 근접해 과적합이 억제되는 반면, GRU/LSTM은 epoch가 진행될수록 train loss는 계속 감소하지만 validation loss는 되려 증가하는 패턴을 보입니다.[^1_3][^1_1]

정리하면,

- TKAN은 “긴 horizon에서 평균 예측 성능이 가장 높고”,
- “실험 간 분산과 과적합 정도가 가장 작다”는 점에서, 장기 예측 태스크에서 더 나은 일반화 특성을 보이는 것으로 해석할 수 있습니다.[^1_1]


### 5.3 한계

- **데이터·도메인 한정**: Binance 암호화폐 notional이라는 단일 도메인만 사용해, 다양한 표준 벤치마크(ETT, M4, Weather 등)에 대한 일반성 검증은 부족합니다.[^1_8][^1_1]
- **비교 기준 제한**: GRU/LSTM/naive만 비교 대상이며, Temporal Fusion Transformer(TFT), 최근의 TS-Mixer, state-space 모델 등 최신 시계열 모델과의 비교는 없습니다.[^1_9][^1_1]
- **복잡도·효율성 분석 부재**: B‑스플라인 기반 KAN은 일반 MLP보다 계산량이 크고 메모리 사용량도 증가하는데, FLOPs/latency 측면 분석은 제공되지 않습니다.[^1_10][^1_1]
- **이론적 일반화 분석 없음**: KAN 일반화 경계에 대한 최근 이론 연구와 연결해 TKAN의 복잡도·일반화를 정량적으로 논의하지는 않습니다.[^1_11][^1_7]

***

## 6. TKAN의 일반화 성능 향상 가능성

### 6.1 실험 관찰에 기반한 일반화 개선

논문은 두 가지 근거로 TKAN의 일반화 우수성을 주장합니다.[^1_1]

1. **장기 horizon에서의 성능 격차**
    - training 데이터에 더 가까운 단기(1–3 step)에서는 GRU/TKAN의 성능이 비슷하지만, test 시 장기(6–15 step)로 갈수록 GRU/LSTM의 성능이 급락하는 반면 TKAN은 점진적인 감소에 그칩니다.[^1_1]
    - 이는 TKAN이 train horizon을 넘어서는 예측 범위에서도 보다 안정적으로 외삽(extrapolation)한다는 간접 증거입니다.[^1_1]
2. **실험 간 분산 및 loss 곡선**
    - 동일 데이터·설정에서 TKAN의 $R^2$ 표준편차가 가장 작고, train/val 곡선 괴리가 작습니다.[^1_1]
    - 이는 최적화 지형이 완만하고, local minima에 대한 민감성이 적어 “재현성·일반화” 측면에서 유리함을 시사합니다.[^1_3][^1_1]

### 6.2 구조적 관점: 왜 일반화에 유리한가?

KAN 및 TKAN 구조가 일반화에 기여하는 메커니즘은 다음과 같이 해석할 수 있습니다.

- **엣지 기반 1D B‑스플라인으로 인한 매끄러운 근사**
    - 각 엣지는 로컬 B‑스플라인의 선형 결합이므로, 함수는 구간별 다항식 형태를 갖고 매끄럽게 연결됩니다.[^1_5][^1_1]
    - 최근 연구들은 KAN이 MLP에 비해 낮은 Lipschitz 상수와 더 나은 adversarial·노이즈 강건성을 보인다고 보고하며, 이는 일반화 오류 상계에도 직접적인 영향을 줍니다.[^1_7][^1_6]
- **메모리와 비선형의 분리**
    - 시간의존성(메모리)은 순환 커널·LSTM 게이트에서, 비선형 근사는 KAN에서 담당하도록 역할을 분리함으로써, vanilla LSTM보다 파라미터 사용 구조가 체계적입니다.[^1_1]
    - 이는 “같은 expressive power를 더 구조화된 방식으로 표현”하게 만들어, 효과적인 파라미터 정규화 역할을 할 수 있습니다.[^1_10]
- **multi-step joint training**
    - 여러 horizon을 동시에 예측하는 구조는, 미래 값들 간의 공통 구조(트렌드, 계절성)를 학습하게 해, 단일-step을 autoregressive하게 반복하는 방식보다 error accumulation과 overfitting을 줄일 수 있습니다.[^1_8][^1_1]

최근 KAN 일반화 이론 연구는 KAN의 Rademacher 복잡도와 generalization bound를 분석해, 충분한 너비에서 SGD가 전역 수렴하며 합리적인 일반화 성능을 보장할 수 있음을 보이고 있어, TKAN 같은 KAN 기반 순환 구조에도 이와 유사한 이론적 근거가 확장될 여지가 있습니다.[^1_11][^1_7]

***

## 7. 2020년 이후 관련 최신 연구 비교 분석 (시계열·시퀀스용 KAN 계열)

아래는 TKAN과 밀접하게 관련된, 2020년 이후 KAN/시계열 관련 주요 오픈 액세스 연구들입니다. (요청하신 형식: 제목, 저자, 출처/링크, 1–2문장 요약)

### 7.1 KAN 기반 시계열 회귀·예측

- **Kolmogorov-Arnold Networks (KANs) for Time Series Analysis**
    - Authors: Cristian J. Vaca-Rubio et al.[^1_4]
    - Source/Link: arXiv:2405.08790, IEEE Globecom Workshops 2024.[^1_12][^1_4]
    - Summary: 위성 트래픽 예측에 vanilla KAN(비순환 feed-forward)을 적용해 MLP 대비 더 적은 파라미터로 더 나은 예측 성능을 보이며, KAN 하이퍼파라미터(그리드, 차수 등)가 성능에 미치는 영향을 실증 분석합니다.[^1_4]
- **Kolmogorov-Arnold Networks for Time Series: Bridging Predictive Power and Interpretability (T‑KAN, MT‑KAN)**
    - Authors: Kunpeng Xu, Lifei Chen, Shengrui Wang.[^1_13][^1_5]
    - Source/Link: arXiv:2406.02496.[^1_14][^1_5]
    - Summary: 개념 드리프트 감지·해석을 위한 단변량 T‑KAN과 다변량 예측 성능 향상을 위한 MT‑KAN을 제안하고, KAN이 시계열 예측에서 정확도·해석력을 동시에 높일 수 있음을 다양한 데이터셋으로 검증합니다.[^1_13][^1_5]
- **KAN4TSF: Reversible Mixture of KAN Experts (RMoK)**
    - Authors: (익명; KAN4TSF 프로젝트 팀).[^1_15]
    - Source/Link: KAN4TSF: Are KAN and KAN-based models Effective for Time Series Forecasting?, arXiv/AXI.[^1_15]
    - Summary: 시계열 예측용 mixture-of-experts 구조 RMoK를 제안해 KAN 전문가들을 조합하고, 여러 시계열 벤치마크에서 기존 딥러닝 기반 모델들을 상회하는 성능과 해석력을 보여 TKAN과 유사하게 KAN 기반 구조의 효과성을 입증합니다.[^1_15]
- **A multivariate time series prediction model based on the KAN network (KANMTS)**
    - Authors: Yunji Long, Xue Qin.[^1_16][^1_17]
    - Source/Link: Scientific Reports 15, 23621 (2025), 오픈 액세스.[^1_17][^1_16]
    - Summary: KAN과 MLP를 결합한 KANMTS 구조를 제안하고, 변동성이 큰 multivariate 시계열에서 기존 RNN·Transformer 대비 더 나은 예측과 자원 효율, 해석력을 달성함을 보여 TKAN과 유사한 “KAN 기반 시계열 모델의 일반화 장점”을 뒷받침합니다.[^1_16]
- **Recurrent Fourier-Kolmogorov Arnold Networks (RFKAN) for photovoltaic power forecasting**
    - Authors: Desheng Rong et al.[^1_18]
    - Source/Link: Scientific Reports 15, 4684 (2025), 오픈 액세스.[^1_18]
    - Summary: Fourier series 기반 주기성 추출 + KAN + 순환 구조를 결합한 RFKAN을 제안해 PV 발전량 day-ahead 예측에서 SOTA 대비 RMSE·MAE 최소 5% 개선, CORR 2% 향상, 학습 시간 24% 단축을 보고하며, TKAN과 같이 “KAN+순환 구조”가 장기 패턴 일반화에 유리함을 보여줍니다.[^1_18]
- **A Temporal Kolmogorov-Arnold Transformer (TKAT) for Time Series Forecasting**
    - Authors: (TKAN 저자 포함).[^1_19][^1_20]
    - Source/Link: arXiv:2406.02486.[^1_20][^1_19]
    - Summary: 본 TKAN 레이어를 Transformer 인코더-디코더 구조에 삽입한 TKAT를 제안하여, Temporal Fusion Transformer와 유사한 multi-horizon 예측 셋업에서 TKAN 기반 어텐션 구조가 긴 범위 의존성·해석력 측면에서 장점을 가짐을 보입니다.[^1_19][^1_20]
- **Zero Shot Time Series Forecasting Using Kolmogorov Arnold Networks**
    - Authors: (에너지 가격 예측 연구팀).[^1_21]
    - Source/Link: arXiv:2412.17853.[^1_21]
    - Summary: N-BEATS 기반 구조의 코어를 KAN으로 대체한 모델을 제안해, 전력 시장 간 cross-domain zero-shot 예측에서 기존 모델 대비 개선된 성능을 보이며, KAN의 표현력이 도메인 불일치 상황의 일반화에도 유리함을 시사합니다.[^1_21]


### 7.2 시계열 분류·강건성·이론

- **Kolmogorov-Arnold Networks (KAN) for Time Series Classification and Robust Analysis**
    - Authors: Chang Dong et al.[^1_22][^1_6]
    - Source/Link: arXiv:2408.07314.[^1_6][^1_22]
    - Summary: 128개 UCR 시계열 데이터셋에서 KAN vs MLP를 비교해 KAN이 유사 혹은 더 나은 정확도를 유지하면서도 Lipschitz 상수가 낮아 적대적 교란에 더 강건함을 보이며, KAN 계열 모델(예: TKAN)의 일반화·강건성 이점을 실증적으로 뒷받침합니다.[^1_6]
- **Exploring Kolmogorov-Arnold Networks for Interpretable Time Series Classification**
    - Authors: Irina Barašin et al.[^1_23]
    - Source/Link: arXiv:2411.14904.[^1_23]
    - Summary: 117개 UCR 데이터셋에서 Efficient KAN 구조를 분석해, MLP 대비 더 빠른 학습과 높은 안정성, SOTA 분류기와 경쟁력 있는 정확도를 달성함을 보여 KAN이 시계열 분류에서도 좋은 일반화-복잡도 트레이드오프를 제공함을 보고합니다.[^1_23]
- **A Comprehensive Survey on Kolmogorov-Arnold Networks (KAN)**
    - Authors: (Survey 팀).[^1_24][^1_10]
    - Source/Link: arXiv:2411.06078 (기간 2024-04~10 KAN 연구 정리).[^1_10]
    - Summary: 스플라인 기반 활성 개선, 시퀀스 데이터용 메모리 강화, 그래프 구조 적응 등 TKAN을 포함한 다양한 KAN 변형들을 연대기적으로 정리하며, KAN의 일반화·해석력·응용 분야를 폭넓게 조망합니다.[^1_10]
- **Generalization Bounds and Model Complexity for Kolmogorov-Arnold Networks**
    - Authors: (이론 연구팀).[^1_7]
    - Source/Link: arXiv:2410.08026.[^1_7]
    - Summary: KAN의 일반화 경계와 모델 복잡도를 분석해, KAN이 적절한 너비에서 좋은 수렴 특성과 일반화 보장을 가질 수 있음을 보이며, TKAN 같은 KAN 기반 순환 구조의 이론적 뒷받침을 제공합니다.[^1_7]
- **On the Convergence of (Stochastic) Gradient Descent for Kolmogorov-Arnold Networks**
    - Authors: (이론 연구팀).[^1_11]
    - Source/Link: arXiv:2410.08041.[^1_11]
    - Summary: 2-layer KAN에서 GD/SGD의 전역 수렴을 이론적으로 증명해, TKAN이 실험적으로 보인 안정적 학습 동역학을 이해하는 데 중요한 이론적 배경을 제공합니다.[^1_11]


### 7.3 TKAN과의 비교 관점 정리

- 구조적으로 TKAN은 **“KAN + 명시적 순환(LSTM)”**을 결합한 반면, T‑KAN/MT‑KAN, KANMTS, RMoK, RFKAN 등은 대체로 feed-forward KAN 블록(혹은 KAN+MLP)을 시계열 피쳐 인코더로 사용하는 방식이 많습니다.[^1_5][^1_16][^1_15][^1_18][^1_1]
- 장기 horizon 예측 측면에서 RFKAN, KANMTS, KAN4TSF, TKAN 모두 vanilla RNN/MLP보다 좋은 성능과 안정성을 보고하여, “KAN 기반 비선형 근사 + 별도의 시계열 구조(순환, patch, mixture 등)” 조합이 일반화에 효과적이라는 공통 결론을 보여줍니다.[^1_16][^1_15][^1_18]
- TKAT, Zero-shot KAN 등은 TKAN/KAN을 Transformer·N-BEATS 등 상위 아키텍처에 통합해, TKAN이 “레이어/블록” 수준 building block으로서 확장 가능함을 보여줍니다.[^1_20][^1_19][^1_21]

***

## 8. 앞으로의 연구에 대한 영향과 고려할 점

### 8.1 향후 연구에 미치는 영향

TKAN은 “KAN을 시계열/순환 구조에 자연스럽게 넣는 구체적 설계 예시”라는 점에서, 이후 연구에 다음과 같은 방향성을 제시합니다.

- **KAN 기반 RNN·Transformer 설계 템플릿**
    - RKAN + LSTM 게이트라는 설계는, 다른 RNN 셀(GRU, state-space, spline-RNN 등)이나 Transformer 블록으로도 일반화할 수 있는 빌딩 블록 아이디어를 제공합니다.[^1_19][^1_1]
- **일반화·안정성을 중시하는 시계열 모델링**
    - TKAN, KANMTS, RFKAN 등이 공통적으로 “성능 + 안정성(분산, 과적합 억제)”을 강조함에 따라, 향후 시계열 연구에서 단순 점 성능뿐 아니라 run-to-run variance, Lipschitz 상수, adversarial·domain shift 강건성 등이 핵심 지표로 부각될 가능성이 큽니다.[^1_6][^1_16][^1_18][^1_1]
- **해석 가능한 시계열 모델**
    - T‑KAN/MT‑KAN, KANMTS, KAN 분류 연구 등과 함께, TKAN도 KAN의 B‑스플라인 구조 덕분에 “시간축에 따른 기여도 함수”를 직접 시각화·해석할 수 있는 잠재력이 있어, 금융·의료 등 설명 가능성이 중요한 도메인에 직접적인 영향을 줄 수 있습니다.[^1_5][^1_23][^1_16]


### 8.2 앞으로 연구 시 고려할 구체적 포인트

연구자로서 TKAN·KAN 계열을 활용/발전시키려 할 때 고려할 만한 점은 다음과 같습니다.

1. **표준 벤치마크 확장 및 공정 비교**
    - ETT, M4, Weather, Traffic 등 널리 쓰이는 시계열 벤치마크에서 TKAN과 T‑KAN/MT‑KAN, RFKAN, KANMTS, Transformer 계열(TFT, TKAT 등)을 동일 설정으로 비교하는 것이 필요합니다.[^1_8][^1_19][^1_16][^1_18]
2. **복잡도–성능–해석력 트레이드오프**
    - B‑스플라인 그리드 수, 차수, RKAN 레이어 수, LSTM 차원 등을 변화시키며,
        - 파라미터 수/연산량,
        - 일반화 성능,
        - 해석력(예: symbolic regression, sensitivity plots)
사이의 정량적 관계를 분석하는 것이 중요합니다.[^1_16][^1_10][^1_5][^1_1]
3. **이론과의 정합성**
    - KAN 일반화 경계·수렴 이론을 TKAN의 순환 구조까지 확장해, horizon 길이, 시퀀스 길이, KAN 그리드 크기 등에 따른 generalization bound를 유도하면, TKAN의 설계 선택(레이어 수, 너비 등)에 이론적 가이드를 줄 수 있습니다.[^1_7][^1_11]
4. **도메인 전이·제로샷·도메인 적응**
    - Zero-shot KAN, cross-domain 전력 가격 예측 연구처럼, TKAN을 여러 시장·도메인(주식·암호화폐·전력·기상 등)에 동시에 학습시켜 도메인 불변 표현을 학습하는 방향이 유망합니다.[^1_9][^1_21]
5. **다른 KAN 변형과의 결합**
    - rKAN, SpectralKAN, SigKAN, Conv-KAN 등 다양한 KAN 변형들을 순환/Transformer 구조와 결합해,
        - 주파수 영역(예: RFKAN, Time-Frequency KAN),
        - 서명(signature) 기반 경로 특성(SigKAN),
        - 합리 함수 기반 근사(rKAN)
등이 시계열 일반화에 어떤 차이를 만드는지 비교하는 연구가 필요합니다.[^1_25][^1_26][^1_18][^1_10]
6. **실무적 관점 – 구현·최적화**
    - B‑스플라인 평가·보간은 GPU 최적화가 까다로울 수 있으므로, 효율적인 커널 구현과 mixed-precision, sparsity 활용 등이 실무 적용에 중요합니다.[^1_12][^1_10]
    - TKAN을 포함한 KAN 계열 레이어를 표준 딥러닝 프레임워크(PyTorch, JAX 등)에 안정적으로 통합하는 라이브러리 수준의 연구·개발도 병행될 필요가 있습니다.[^1_12][^1_1]

요약하면, TKAN은 “KAN을 기반으로 한 시계열용 순환 아키텍처”의 첫 번째 구체적 사례로서, 장기 multi-step 예측에서의 성능·안정성 향상을 실증적으로 보여주며, 앞으로의 연구에서는 (1) 더 광범위한 벤치마크 검증, (2) 이론–실험의 연계, (3) 다양한 KAN 변형 및 상위 아키텍처와의 조합을 통해 그 일반화 능력을 보다 체계적으로 이해·확장하는 것이 주요 과제가 될 것입니다.[^1_19][^1_18][^1_10][^1_16][^1_1]
<span style="display:none">[^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43]</span>

<div align="center">⁂</div>

[^1_1]: 2405.07344v4.pdf

[^1_2]: https://arxiv.org/abs/2405.07344

[^1_3]: https://arxiv.org/html/2405.07344v1

[^1_4]: https://arxiv.org/abs/2405.08790

[^1_5]: https://arxiv.org/abs/2406.02496

[^1_6]: https://arxiv.org/abs/2408.07314

[^1_7]: http://arxiv.org/pdf/2410.08026.pdf

[^1_8]: https://arxiv.org/abs/2412.15373

[^1_9]: https://arxiv.org/abs/2407.15236

[^1_10]: https://arxiv.org/pdf/2411.06078.pdf

[^1_11]: https://arxiv.org/abs/2410.08041

[^1_12]: http://arxiv.org/pdf/2405.08790.pdf

[^1_13]: https://www.semanticscholar.org/paper/Kolmogorov-Arnold-Networks-for-Time-Series:-Power-Xu-Chen/10145b2238569436754c4d9be3f9c7db501cc65c

[^1_14]: http://arxiv.org/pdf/2406.02496.pdf

[^1_15]: https://axi.lims.ac.uk/paper/2408.11306

[^1_16]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12222760/

[^1_17]: https://www.nature.com/articles/s41598-025-07654-7

[^1_18]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11805904/

[^1_19]: https://arxiv.org/abs/2406.02486

[^1_20]: https://arxiv.org/pdf/2406.02486.pdf

[^1_21]: https://arxiv.org/abs/2412.17853

[^1_22]: https://arxiv.org/html/2408.07314v3

[^1_23]: https://arxiv.org/abs/2411.14904

[^1_24]: https://arxiv.org/html/2407.11075v5

[^1_25]: https://arxiv.org/abs/2406.17890

[^1_26]: http://arxiv.org/pdf/2406.17890.pdf

[^1_27]: https://arxiv.org/pdf/2601.18837.pdf

[^1_28]: https://arxiv.org/pdf/2601.02310.pdf

[^1_29]: https://arxiv.org/html/2602.11190v1

[^1_30]: https://arxiv.org/html/2407.11075v8

[^1_31]: https://arxiv.org/html/2506.12696v1

[^1_32]: https://arxiv.org/pdf/2504.16432.pdf

[^1_33]: https://www.arxiv.org/pdf/2511.18613.pdf

[^1_34]: https://arxiv.org/html/2510.04622v1

[^1_35]: https://arxiv.org/abs/2412.01224

[^1_36]: https://arxiv.org/abs/2410.10041

[^1_37]: https://arxiv.org/abs/2410.02033

[^1_38]: http://arxiv.org/pdf/2405.07344.pdf

[^1_39]: https://arxiv.org/pdf/2501.17411.pdf

[^1_40]: https://axi.lims.ac.uk/paper/2406.02496

[^1_41]: https://linnk.ai/insight/machine-learning/kolmogorov-arnold-networks-kan-for-time-series-classification-and-adversarial-robustness-analysis-8WbqRNrv/

[^1_42]: https://arxiv.org/abs/2410.05500v2

[^1_43]: https://www.sciencedirect.com/science/article/pii/S2352484724008539

