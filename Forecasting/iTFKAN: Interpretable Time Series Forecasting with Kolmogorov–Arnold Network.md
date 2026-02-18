# iTFKAN: Interpretable Time Series Forecasting with Kolmogorov–Arnold Network

iTFKAN은 **KAN(Kolmogorov–Arnold Network)을 시계열 예측에 적용한 최초의 완전 해석 가능한 프레임워크**로, 예측 성능과 해석 가능성을 동시에 달성하는 것이 핵심 목표입니다. 기존 딥러닝 예측 모델(MLP, Transformer 계열)이 블랙박스 구조로 인해 신뢰성이 낮다는 문제를 해결하기 위해, 모델 구조 자체에서 상징적 수식(symbolic formula)을 통해 의사결정 근거를 제공합니다.[^1_1]

**주요 기여 3가지:**

- KAN의 상징화(symbolization) 능력을 시계열에 적용한 해석 가능한 예측 프레임워크 제안
- 사전 지식 주입(Prior Knowledge Injection) 전략: 트렌드·계절성 패턴을 수식으로 KAN 구조 학습에 유도
- 시간-주파수 시너지 학습(Time-Frequency Synergy Learning): 두 도메인의 보완적 정보를 손실 없이 결합[^1_1]

***

## 해결하고자 하는 문제

현행 딥러닝 예측 모델은 두 가지 핵심 한계를 갖습니다:[^1_1]

1. **해석 불가능성**: RNN, CNN, Transformer 기반 모델은 복잡한 네트워크 구조로 인해 모델 결정 근거를 추적하기 어려우며, 자율주행·의료·금융 등 안전-critical 분야에서 신뢰성 문제 야기
2. **복잡 패턴 학습의 어려움**: 시계열에 내재된 트렌드·계절성·무작위 변동이 뒤엉켜 있어, KAN이 이상적인 모델 구조를 자동으로 학습하기 어렵고 국소 최적해에 빠지는 경향 존재

***

## 제안 방법 및 수식

### 1. KAN 이론적 기반

Kolmogorov-Arnold 표현 정리(Kolmogorov-Arnold Representation Theorem)에 따르면, 임의의 연속 다변수 함수는 단변수 함수들의 유한 합성으로 표현됩니다:[^1_1]

$f(x_1, \ldots, x_n) = \sum_{q=1}^{2n+1} \Phi_q \left( \sum_{p=1}^{n} \phi_{q,p}(x_p) \right) \tag{1}$

KAN의 구조는 이를 확장하여 레이어를 쌓은 형태로 정의됩니다:[^1_1]

$\text{KAN}(x) = (\Phi_D \circ \cdots \circ \Phi_2 \circ \Phi_1)(x), \quad \Phi_d = \{\phi^d_{i,j} \mid 1 \leq i \leq I,\ 1 \leq j \leq J\} \tag{2}$

### 2. Trend-Seasonal Decomposition

입력 $\mathbf{X} \in \mathbb{R}^{N \times L}$를 선형 레이어로 고차원 표현 $\mathbf{X} \in \mathbb{R}^{N \times L \times d}$로 매핑 후, Moving Average 기반으로 분해합니다:[^1_1]

$\text{Trend:}\ \mathbf{T} = \text{AvgPool}(\text{Padding}(\mathbf{X})), \quad \text{Seasonal:}\ \mathbf{S} = \mathbf{X} - \mathbf{T} \tag{3}$

### 3. Prior-Guided TaylorKAN

#### Taylor 급수 기반 활성화 함수 (효율성 향상)

KAN의 Spline 파라미터화를 Taylor 급수로 대체하여 계산 복잡도를 낮춥니다:[^1_1]

$\phi(x) = w\left(b(x) + \sum_{o=0}^{O} a_o x^o\right), \quad O=2 \tag{4}$

#### TrendInject (다항식 사전 지식 주입)

트렌드의 단조성을 소차 다항식으로 모델링합니다:[^1_1]

$\text{TrendInject}(x) = m_p x^p + m_{p-1}x^{p-1} + \cdots + m_1 x + m_0 \tag{5}$

#### SeasonalInject (주기성 Fourier 급수 주입)

계절성을 Fourier 급수로 모델링하여, 상위 K개 주파수 성분을 주입합니다:[^1_1]

$\text{SeasonalInject}(x) = \frac{a_0}{2} + \sum_{k=1}^{K}\left[a_k \cos(f_k \pi x) + b_k \sin(f_k \pi x)\right] \tag{6}$

이를 통해 TrendKAN의 첫 번째 레이어 활성화는 $\Phi^1_{\text{Trend}} = \{x^j \mid 1 \leq j \leq p\}$, SeasonalKAN의 활성화는 아래와 같이 정의됩니다:[^1_1]

$\Phi^1_{\text{Season}} = \{\text{Four}(f_j \pi x) \mid j \in I_K\}, \quad \text{Four}(f_k \pi x) = \cos(f_k \pi x) + \sin(f_k \pi x) \tag{7,8}$

#### TrendKAN / SeasonalKAN 출력

$h_T = \text{TrendKAN}(\mathbf{T}), \quad h_S = \text{SeasonalKAN}(\mathbf{S}), \quad h_T, h_S \in \mathbb{R}^{N \times L \times d} \tag{9}$

### 4. Time-Frequency Synergy Learning (TFKAN)

계절 성분 $\mathbf{S}$를 패치(patch)로 분할하고 이산 푸리에 변환(DFT)을 적용합니다:[^1_1]

$F_k = \sum_{p=1}^{P} P_p \cdot e^{-j \frac{2\pi}{P} kp} = A_k \cdot e^{j\phi_k} \tag{10}$

이후 역푸리에 변환(IDFT)으로 2차원 시간-주파수 관계 $TF \in \mathbb{R}^{N \times K \times P \times d}$를 구성합니다:[^1_1]

$TF_{p,k} = A_k \cdot e^{j\left(\phi_k + \frac{2\pi}{P}kp\right)} \tag{11}$

각 패치에 대해 KAN을 적용하여 시간-주파수 보완 표현을 학습합니다:[^1_1]

$h^p_{tf} = \text{TFKAN}(TF_{p,k}) = \frac{1}{K}\sum_{k=1}^{K} \text{KAN}\_p(TF_{p,k}) \tag{12}$

### 5. 손실 함수 (Sparsification Loss 포함)

해석 가능성을 위한 희소화(sparsification) 패널티를 총 손실에 포함합니다:[^1_1]

$\ell_{reg} = \sum_{d=1}^{D}\sum_{j=1}^{J}\sum_{i=1}^{I} \|\phi^d_{i,j}\|\_2, \quad \|\phi\|\_2 = \frac{1}{O}\sum_{o=1}^{O} a_o^2 \tag{13}$

$\ell_{total} = \ell_{pred} + \lambda \sum_{\text{kan} \in \text{KANs}} \ell^{\text{kan}}_{reg} \tag{14}$

***

## 모델 구조

iTFKAN은 세 모듈이 직렬 연결된 파이프라인 구조입니다:[^1_1]


| 모듈 | 입력 | 역할 |
| :-- | :-- | :-- |
| Trend-Seasonal Decomposition | $\mathbf{X}$ | Moving Average로 트렌드/계절 분리 |
| Prior-Guided TaylorKAN | $\mathbf{T}$, $\mathbf{S}$ | 사전 지식 주입 후 각 성분의 비선형 표현 학습 |
| Time-Frequency Synergy Learning | $\mathbf{S}$ | 패치 분할 + DFT/IDFT로 2D 시간-주파수 의존성 학습 |
| Linear Projector | $h_T, h_S, h_{TF}$ | 세 표현을 결합하여 최종 예측값 생성 |

해석 가능성은 다음 3단계로 달성됩니다:[^1_1]

1. **희소화(Sparsification)**: L2-norm 기반 패널티로 불필요한 엣지 억제
2. **가지치기(Pruning)**: 임계값 $\tau$ 이하의 활성화 함수 제거 (예: ETTh1에서 TrendKAN 1층 가지치기율 98.74%)
3. **상징화(Symbolification)**: 활성화 함수를 $R^2$ 계수로 최적 상징 수식에 대응, $y \approx c \cdot f_s(ax+b)+d$ 형태로 출력

***

## 성능 향상

장기 예측 8개 벤치마크와 M4 단기 예측 6개 카테고리에서 9개 최신 베이스라인 대비 전반적으로 최상위 또는 2위 성능을 달성하였습니다:[^1_1]


| 데이터셋 | iTFKAN (MSE) | TimeMixer (MSE) | iTransformer (MSE) | PatchTST (MSE) |
| :-- | :-- | :-- | :-- | :-- |
| ETTh1 | **0.434** | 0.448 | 0.458 | 0.446 |
| ETTh2 | **0.360** | 0.372 | 0.382 | 0.379 |
| ETTm1 | **0.380** | 0.384 | 0.408 | 0.386 |
| ETTm2 | **0.274** | 0.276 | 0.291 | 0.290 |
| Weather | **0.244** | 0.244 | 0.260 | 0.257 |
| Exchange | **0.355** | 0.369 | 0.368 | 0.371 |
| M4-Avg (OWA) | **0.85** | 0.87 | 0.90 | — |

단, **Electricity·Traffic** 데이터셋에서는 채널 의존성이 매우 강해 채널 독립(channel-independent) 구조를 채택한 iTFKAN이 iTransformer·TimeMixer 대비 소폭 열위를 보였습니다.[^1_1]

***

## 모델의 일반화 성능 향상 가능성

iTFKAN의 일반화 성능은 여러 메커니즘에 의해 뒷받침됩니다:[^1_1]

1. **사전 지식 주입으로 데이터 의존성 감소**: KAN의 데이터 중심 최적화 의존도를 줄여 로컬 최적해 탈출 가능. 특히 단기 예측(M4 monthly·weekly)처럼 주기성이 명확하지 않은 데이터에서 SeasonalInject의 효과가 두드러짐
2. **Sparsification + Pruning**: L2-norm 기반 정규화 손실($\ell_{reg}$)이 과적합을 억제하며 간결한 모델 구조 유도 (ETTh1에서 TrendKAN 94.99% 가지치기 후에도 성능 유지)
3. **채널 독립(channel-independence) 설계**: 각 변수의 독립적 시간적 진화 패턴 학습으로 일부 다변량 구조 환경에서도 성능 전이 가능
4. **시간-주파수 2D 표현**: 단순 주파수 통합 방식(FreqKAN)보다 정보 손실이 적어 다양한 데이터셋 특성에 강인함. ablation에서 TFKAN 제거 시 ETTh1 MSE가 0.434→0.455로 크게 하락[^1_1]

***

## 한계

- **고차원 다변량 데이터 취약**: Electricity(321 변수), Traffic(862 변수)처럼 변수 간 채널 의존성이 강한 데이터에서는 채널 독립 가정으로 인해 iTransformer 등에 뒤처짐[^1_1]
- **TaylorKAN 2차 근사의 한계**: 2차 Taylor 급수가 국소 근사에서는 우수하나, 급격한 비선형 패턴(노이즈, 급변)에 대한 표현력이 제한적일 수 있음
- **하이퍼파라미터 민감성**: 패치 길이($P$), 스트라이드($S$), 주파수 수($K$) 등을 데이터셋별 grid search로 결정하므로 새로운 도메인에 대한 즉시 적용이 어려움[^1_1]
- **확률적 예측(probabilistic forecasting) 미지원**: 현재 점 예측(point forecast)에 한정되어 불확실성 정량화 불가

***

## 2020년 이후 관련 최신 연구 비교 분석

| 모델 | 연도 | 핵심 접근 | 해석 가능성 | KAN 사용 | 주요 차별점 |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Autoformer | 2021 | 자기상관 + 분해 Transformer | 없음 | ✗ | 장기 의존성 학습 |
| PatchTST | 2023 | 패치 기반 Transformer | 낮음 | ✗ | 채널 독립 patch attention |
| iTransformer | 2024 | 역전된 어텐션 (변수 축) | 낮음 | ✗ | 다변량 변수 관계 포착 |
| TimeMixer | 2024 | 다중 스케일 MLP 분해 | 낮음 | ✗ | 분해 기반 멀티스케일 믹싱 |
| T-KAN / MT-KAN | 2024 | KAN + 기본 시계열 구조 | 중간 | ✓ | 개념 드리프트 감지 [^1_2] |
| TKAN | 2024 | 순환 구조 + KAN | 중간 | ✓ | RNN 대체 KAN 셀 [^1_3] |
| TimeKAN | 2025 | 계단식 주파수 분해 + KAN | 중간 | ✓ | 주파수 밴드별 M-KAN 표현 [^1_4] |
| AR-KAN | 2025 | AR 모듈 + KAN 결합 | 중간 | ✓ | ARIMA 수준 정확도 [^1_5] |
| **iTFKAN** | **2025** | **KAN + 사전 지식 + TF 시너지** | **높음** | **✓** | **완전 심볼릭 해석 + 사전지식 주입** |

iTFKAN은 기존 KAN 기반 예측 모델(T-KAN, TimeKAN 등)과 달리, **모델 내부를 수식으로 완전히 해석 가능한 상징화 파이프라인**을 제공하는 점에서 독보적입니다.[^1_5][^1_4][^1_1]

***

## 앞으로의 연구에 미치는 영향과 고려 사항

### 연구에 미치는 영향

- **해석 가능한 AI의 새 표준 제시**: 기존 예측 모델의 블랙박스 문제를 KAN 상징화로 해결하는 패러다임을 확립하여, 의료·금융·에너지 등 고신뢰 분야에서의 딥러닝 도입을 가속화할 수 있음
- **사전 지식 통합 프레임워크의 확장 가능성**: TrendInject·SeasonalInject 방식은 도메인 전문 지식을 다양한 형태(물리 방정식, 경제 모델)로 주입하는 연구로 확장될 수 있음[^1_1]
- **시간-주파수 2D 표현 학습의 가능성**: KAN과 Fourier 변환의 결합 방식은 음성·생체신호 등 다른 시계열 도메인에도 응용 가능하며, TFKAN 구조에서 영감을 받은 후속 연구가 이미 등장하고 있음[^1_6]


### 앞으로 연구 시 고려할 점

1. **채널 의존성 통합**: 채널 독립 가정을 완화하거나, iTransformer처럼 변수 간 의존성을 학습하는 채널 의존적 TFKAN 변형 연구가 필요함[^1_1]
2. **불확실성 정량화**: P-KAN 처럼 확률적 예측으로 확장하여, 안전-critical 분야에서의 신뢰구간 및 리스크 추정 지원 필요[^1_7]
3. **도메인 일반화 및 제로샷 전이**: 데이터셋별 grid search에 의존하지 않고 다양한 도메인에 즉시 적용 가능한 메타러닝 또는 프롬프트 기반 사전지식 주입 전략 연구
4. **LLM과의 결합 가능성**: Time-LLM, SE-LLM 등 대형 언어 모델 기반 시계열 예측과 KAN의 해석 가능성을 결합하는 하이브리드 접근이 유망한 연구 방향[^1_8]
5. **비정상(non-stationary) 시계열 대응**: 개념 드리프트(concept drift)와 분포 이동(distribution shift)에 강인한 적응형 KAN 구조 개발 (ShifTS, T-KAN  방식 참고)[^1_9][^1_10]
6. **계산 효율화**: TaylorKAN이 스플라인 대비 효율적이나, 대규모 변수 환경에서의 병렬화·경량화 연구가 추가적으로 필요함[^1_11]
<span style="display:none">[^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46]</span>

<div align="center">⁂</div>

[^1_1]: 2504.16432v2.pdf

[^1_2]: https://arxiv.org/abs/2406.02496

[^1_3]: https://arxiv.org/abs/2405.07344

[^1_4]: https://arxiv.org/abs/2502.06910

[^1_5]: https://arxiv.org/abs/2509.02967

[^1_6]: https://arxiv.org/html/2506.12696v1

[^1_7]: https://arxiv.org/abs/2510.16940

[^1_8]: https://arxiv.org/html/2508.07697v3

[^1_9]: https://arxiv.org/html/2510.14814v1

[^1_10]: https://axi.lims.ac.uk/paper/2406.02496

[^1_11]: https://dl.acm.org/doi/10.1145/3743128

[^1_12]: https://pubs.acs.org/doi/10.1021/acs.est.4c11113

[^1_13]: https://www.mdpi.com/2227-7390/12/19/3022

[^1_14]: https://www.mdpi.com/1424-8220/25/23/7287

[^1_15]: https://ieeexplore.ieee.org/document/11221577/

[^1_16]: https://dl.acm.org/doi/10.1145/3746252.3760836

[^1_17]: https://dx.plos.org/10.1371/journal.pone.0337793

[^1_18]: http://arxiv.org/pdf/2405.08790.pdf

[^1_19]: http://arxiv.org/pdf/2502.06910.pdf

[^1_20]: http://arxiv.org/pdf/2406.02496.pdf

[^1_21]: http://arxiv.org/pdf/2408.11306.pdf

[^1_22]: http://arxiv.org/pdf/2406.17890.pdf

[^1_23]: https://arxiv.org/pdf/2502.18410.pdf

[^1_24]: http://arxiv.org/pdf/2411.00278.pdf

[^1_25]: https://arxiv.org/pdf/2502.00980.pdf

[^1_26]: https://arxiv.org/pdf/2601.18837.pdf

[^1_27]: https://arxiv.org/abs/2411.00278

[^1_28]: https://arxiv.org/html/2412.17853v2

[^1_29]: https://peerj.com/articles/cs-3001/

[^1_30]: https://www.semanticscholar.org/paper/Kolmogorov-Arnold-networks-for-time-series-a-review-Yamak-Li/44e39edc11dc6ab7ac768f1d01dfc362a137f319

[^1_31]: https://arxiv.org/html/2503.10198v1

[^1_32]: https://arxiv.org/html/2510.16940v1

[^1_33]: https://arxiv.org/pdf/2412.04532.pdf

[^1_34]: https://arxiv.org/html/2509.18483v1

[^1_35]: https://www.datasciencewithmarco.com/blog/kolmogorov-arnold-networks-kans-for-time-series-forecasting

[^1_36]: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5220118

[^1_37]: https://research.google/blog/interpretable-deep-learning-for-time-series-forecasting/

[^1_38]: https://openreview.net/forum?id=LWQ4zu9SdQ

[^1_39]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11605417/

[^1_40]: https://www.semanticscholar.org/paper/Kolmogorov-Arnold-Networks-(KANs)-for-Time-Series-Vaca-Rubio-Blanco/081eb8781725e560f597b01c624fe65618c3c0f8

[^1_41]: https://www.sciencedirect.com/science/article/abs/pii/S0360835224005333

[^1_42]: https://icml.cc/virtual/2025/poster/45584

[^1_43]: https://neurips.cc/virtual/2024/poster/93522

[^1_44]: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4825654

[^1_45]: https://ieeexplore.ieee.org/iel8/6287639/10380310/10583885.pdf

[^1_46]: https://5g-stardust.eu/wp-content/uploads/sites/97/2025/01/GLOBECOM_WS_KANs-1.pdf

