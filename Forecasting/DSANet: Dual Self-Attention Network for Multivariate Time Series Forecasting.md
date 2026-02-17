# DSANet: Dual Self-Attention Network for Multivariate Time Series Forecasting

## 1. 핵심 주장과 주요 기여

DSANet(Dual Self-Attention Network)은 다변량 시계열 예측을 위한 혁신적인 딥러닝 프레임워크로, 특히 동적 주기(dynamic-period) 패턴이나 비주기적(nonperiodic) 패턴을 가진 시계열 데이터에 효과적입니다. 이 논문의 핵심 주장은 RNN 기반 구조를 완전히 배제하고 이중 합성곱 구조와 자기 주의(self-attention) 메커니즘을 결합하여 복잡한 시간적 의존성과 변수 간 의존성을 동시에 포착할 수 있다는 것입니다.[^1_1]

주요 기여는 다음과 같습니다:[^1_1]

- 시계열 예측에 자기 주의 메커니즘을 적용한 최초의 연구
- Global과 Local 시간적 패턴을 동시에 모델링하는 이중 분기 아키텍처 설계
- 비선형 신경망과 선형 자기회귀 모델의 병렬 통합을 통한 강건성 향상


## 2. 해결하고자 하는 문제

### 기존 방법의 한계

전통적인 통계 기반 방법(VAR, GP 등)은 시계열의 분포나 함수 형태에 대한 특정 가정을 필요로 하며, 복잡한 비선형 관계를 포착하지 못합니다. 또한 다변량 시계열에서 변수 간 의존성을 무시하는 경우가 많아 예측 정확도가 저하됩니다.[^1_1]

LSTM/GRU 기반 RNN 모델들은 장기 의존성 모델링에서 개선을 보였으나, 동적 주기 패턴이나 비주기적 패턴을 가진 데이터에서는 성능이 크게 저하됩니다. 특히 LSTNet-S, LSTNet-A, TPA 같은 복잡한 구조들도 이러한 데이터 특성에는 적합하지 않습니다.[^1_1]

### 핵심 도전 과제

다변량 시계열 예측의 주요 도전 과제는 시간 단계 간 동적 의존성과 다중 변수 간 의존성을 동시에 포착하는 것입니다. 이러한 의존성은 시간에 따라 동적으로 변화하여 분석의 난이도를 크게 증가시킵니다.[^1_1]

## 3. 제안하는 방법 (DSANet)

### 모델 구조

DSANet은 세 가지 주요 구성 요소로 이루어져 있습니다:[^1_1]

**1) 이중 시간적 합성곱 (Dual Temporal Convolution)**

**Global Temporal Convolution:** 전체 시간 단계에 걸친 시불변(time-invariant) 패턴을 추출하기 위해 $T \times 1$ 크기의 필터를 사용합니다. 입력 행렬 $X$에 대해 ReLU 활성화 함수를 적용하여 크기 $D \times n_G$의 출력 행렬 $H^G$를 생성합니다. 여기서 $n_G$는 전역 시간적 합성곱의 필터 수입니다.[^1_1]

**Local Temporal Convolution:** 상대적 거리가 짧은 시간 단계들이 서로 더 큰 영향을 미친다는 점을 고려하여, 길이 $l$ (여기서 $l < T$)의 필터를 사용합니다. 각 필터는 시간 차원을 따라 슬라이딩하여 행렬 $M_k^L$을 생성하고, 1-D max-pooling을 적용하여 가장 대표적인 특징을 포착합니다. 최종적으로 크기 $D \times n_L$의 출력 행렬 $H^L$을 얻습니다.[^1_1]

**2) 자기 주의 모듈 (Self-Attention Module)**

Transformer에서 영감을 받은 자기 주의 모듈은 서로 다른 시계열 간의 의존성을 학습합니다. 모듈은 $N$개의 동일한 레이어로 구성되며, 각 레이어는 자기 주의 계층과 위치별 피드포워드 계층을 포함합니다.[^1_2][^1_1]

**Scaled Dot-Product Self-Attention:**

$$
Z^G = \text{softmax}\left(\frac{Q^G(K^G)^T}{\sqrt{d_k}}\right)V^G
$$

여기서 $Q^G$, $K^G$, $V^G$는 입력 $H^G$에 투영을 적용하여 얻은 쿼리(query), 키(key), 값(value) 행렬이며, $d_k$는 키의 차원입니다.[^1_1]

Multi-head attention을 활용하여 모델이 서로 다른 위치에서 다양한 표현 부공간의 정보를 공동으로 처리할 수 있도록 합니다. 결과적인 가중 표현들은 연결되어 최종 표현 $Z_O^G$를 생성합니다.[^1_1]

**Position-wise Feed-Forward Layer:**

$$
F^G = \text{ReLU}(Z_O^G W_1 + b_1)W_2 + b_2
$$

여기서 $W_1$, $W_2$는 가중치 행렬이고 $b_1$, $b_2$는 편향(bias)입니다. Layer normalization과 residual connection을 통해 학습이 용이해지고 일반화 성능이 향상됩니다.[^1_1]

Local temporal convolution 이후의 자기 주의 모듈도 유사한 절차를 거쳐 $H^L$을 입력받아 최종 출력 $F^L$을 생성합니다.[^1_1]

**3) 자기회귀 구성 요소 (Autoregressive Component)**

합성곱과 자기 주의 구성 요소의 비선형성으로 인해 신경망 출력의 스케일이 입력 스케일에 민감하지 않은 문제를 해결하기 위해, 고전적인 AR 모델을 선형 구성 요소로 통합합니다. AR 구성 요소의 예측은 $\hat{X}_{T+h}^L \in \mathbb{R}^D$로 표현됩니다.[^1_1]

**최종 예측 생성:**

먼저 dense layer를 사용하여 두 자기 주의 모듈의 출력을 결합하여 자기 주의 기반 예측 $\hat{X}\_{T+h}^D \in \mathbb{R}^D$를 얻습니다. DSANet의 최종 예측 $\hat{X}\_{T+h}$는 자기 주의 기반 예측 $\hat{X}\_{T+h}^D$와 AR 예측 $\hat{X}_{T+h}^L$을 합산하여 얻습니다.[^1_1]

### 학습 방법

모든 신경망 모델은 Adam optimizer를 사용한 미니 배치 확률적 경사 하강법(SGD)으로 최적화되며, 손실은 평균 제곱 오차(MSE)로 계산됩니다. Dropout 비율은 0.1로 설정하여 과적합을 방지합니다.[^1_1]

## 4. 성능 향상 및 한계

### 실험 결과

DSANet은 가스 스테이션 일일 수익 데이터셋(2015-2018, 5개 스테이션)에서 평가되었습니다. 데이터는 훈련(60%), 검증(20%), 테스트(20%)로 분할되었습니다.[^1_1]

**정량적 성능:**

Window=32 설정에서 DSANet은 모든 horizon(3, 6, 12, 24일)에서 일관되게 최고 성능을 달성했습니다:[^1_1]


| Horizon | RRSE | MAE |
| :-- | :-- | :-- |
| 32-3 | 0.7817 | 0.4074 |
| 32-6 | 0.7713 | 0.4102 |
| 32-12 | 0.8297 | 0.4367 |
| 32-24 | 0.9277 | 0.4422 |

DSANet은 VAR, LRidge, LSVR, GP, GRU, LSTNet-S, LSTNet-A, TPA를 포함한 8개 베이스라인 모델보다 우수한 성능을 보였습니다.[^1_1]

**Ablation Study:**

각 구성 요소의 효과를 검증하기 위해 Global branch(DSAwoGlobal), Local branch(DSAwoLocal), AR component(DSAwoAR)를 각각 제거한 실험을 수행했습니다. 결과는 다음을 보여줍니다:[^1_1]

1. 완전한 DSANet이 모든 window-horizon 쌍에서 최고 성능을 달성하여 모든 구성 요소가 모델의 효과성과 강건성에 기여함을 입증[^1_1]
2. DSAwoAR의 성능이 가장 크게 저하되어 AR 구성 요소가 중요한 역할을 함을 확인[^1_1]
3. DSAwoGlobal과 DSAwoLocal도 성능 손실을 겪지만 AR 제거보다는 적음[^1_1]

### 모델의 일반화 성능 향상 가능성

**구조적 강점:**

1. **Residual Connections \& Layer Normalization:** 자기 주의 모듈의 각 하위 계층 주변에 잔차 연결(residual connection)과 layer normalization을 적용하여 학습을 용이하게 하고 일반화 성능을 향상시킵니다.[^1_1]
2. **하이브리드 아키텍처:** 비선형 신경망과 선형 AR 모델을 병렬로 통합하여 다양한 데이터 패턴에 대한 적응력을 높입니다. 이는 특히 데이터 스케일 변화에 대한 강건성을 제공합니다.[^1_1]
3. **이중 분기 설계:** Global과 Local 시간적 패턴을 동시에 모델링함으로써 다양한 시간 스케일의 패턴을 포착하여 일반화 성능이 향상됩니다.[^1_1]
4. **RNN 구조 배제:** RNN의 순차적 특성을 제거하고 병렬 처리가 가능한 합성곱 구조를 사용하여 긴 시퀀스 모델링에서 효율성과 성능을 개선합니다.[^1_1]

### 한계점

**1) 데이터셋 제한:**

- 실험이 단일 도메인(가스 스테이션 수익)에서만 수행되어 다양한 도메인에서의 일반화 능력이 검증되지 않음[^1_1]
- 5개 변수만 포함된 상대적으로 작은 규모의 다변량 시계열에서만 평가됨[^1_1]

**2) 계산 복잡도:**

- 자기 주의 메커니즘의 $\mathcal{O}(N^2)$ 복잡도로 인해 변수 수가 많은 경우 계산 비용이 증가[^1_3]
- 두 개의 병렬 분기와 자기 주의 모듈의 스택으로 인한 메모리 요구사항[^1_1]

**3) 하이퍼파라미터 민감도:**

- Local temporal convolution의 필터 길이 $l$, 필터 수 $n_G$, $n_L$, 스택 레이어 수 $N$ 등 여러 하이퍼파라미터에 대한 그리드 서치가 필요[^1_1]

**4) 해석가능성:**

- 이중 분기 구조와 다층 자기 주의가 어떻게 특정 패턴을 학습하는지에 대한 명확한 설명이 부족[^1_1]


## 5. 향후 연구에 미치는 영향 및 고려사항

### 영향

**1) 아키텍처 혁신의 촉진:**

DSANet은 RNN을 배제하고 합성곱과 자기 주의를 결합한 최초의 시계열 예측 모델로서, 이후 많은 연구들이 유사한 접근 방식을 탐구하게 되었습니다. 특히 Transformer 기반 모델들이 시계열 예측 분야에서 주류가 되는 계기를 마련했습니다.[^1_4][^1_2][^1_1]

**2) 이중 경로 설계 패러다임:**

Global과 Local 정보를 동시에 모델링하는 이중 분기 아키텍처는 후속 연구에서 널리 채택되었습니다. 예를 들어, AltTS(2026)는 autoregression과 cross-relation을 분리하는 이중 경로 프레임워크를 제안하여 최적화 충돌을 방지합니다.[^1_5]

**3) 하이브리드 모델링 접근:**

비선형 딥러닝과 선형 통계 모델의 결합은 강건성 향상에 효과적임이 입증되어, VARMAformer(2025)와 같은 모델이 VARMA 통계적 통찰력을 Transformer에 통합하는 연구로 이어졌습니다.[^1_6]

### 앞으로 연구 시 고려할 점

**1) 효율성 개선:**

자기 주의 메커니즘의 $\mathcal{O}(N^2)$ 계산 복잡도는 대규모 다변량 시계열에서 병목이 됩니다. 최근 연구들은 이를 해결하기 위한 다양한 접근을 제시합니다:

- **선형 복잡도 주의:** vLinear(2026)는 vecTrans 모듈을 통해 $\mathcal{O}(N)$ 복잡도로 다변량 상관관계를 모델링합니다[^1_3]
- **Sparse Attention:** 동적 희소화(dynamic sparsification)를 통해 노이즈가 많은 환경에서 효과적으로 작동합니다[^1_7]
- **MLP 기반 대안:** TSMixer(2023)는 주의 메커니즘 없이 MLP만으로 경쟁력 있는 성능을 달성합니다[^1_8]

**2) 일반화 및 전이 학습:**

DSANet의 한계인 단일 도메인 검증을 넘어서기 위해:

- **대규모 사전 학습:** Moirai, TimeGPT 같은 범용 시계열 모델들은 다양한 데이터셋에서 사전 학습되어 전이 학습 능력을 향상시킵니다[^1_2]
- **메타 학습:** MMformer(2025)는 자기 주의와 메타 학습을 결합하여 다양한 데이터 분포에 대한 일반화 능력과 적응성을 향상시킵니다[^1_9]
- **도메인 적응:** 다양한 벤치마크 데이터셋에서 일관된 성능을 보이는 모델 설계가 중요합니다[^1_2]

**3) 장기 예측 성능:**

DSANet의 실험은 상대적으로 짧은 horizon(최대 24일)에서만 수행되었습니다. 장기 예측을 위해서는:

- **분해 기반 접근:** 트렌드, 계절성, 잔차를 분리하여 모델링하는 방법이 효과적입니다[^1_10]
- **Patch 기반 토크나이제이션:** 긴 시퀀스를 패치로 분할하여 처리하는 PatchTST 같은 방법이 장기 의존성 포착에 유리합니다[^1_11]
- **교차 최적화:** AltTS는 alternating optimization을 통해 장기 예측에서 가장 두드러진 개선을 보입니다[^1_5]

**4) 해석가능성 향상:**

모델의 결정 과정을 이해하는 것은 실무 적용에 중요합니다:

- **주의 가중치 분석:** 학습된 주의 패턴이 의미 있고 설명 가능한 구조를 드러내는지 분석해야 합니다[^1_7]
- **시각화 도구:** Graph Attention Network(2020)는 이상 탐지에서 좋은 해석가능성을 보여주며 이상 진단에 유용합니다[^1_12]

**5) 다양한 아키텍처 탐색:**

- **주의 메커니즘의 재고:** 최근 연구는 주의 행렬이 정말로 자기 주의의 핵심인지 의문을 제기하며, 일부 경우 전체 주의 메커니즘을 MLP로 축소할 수 있음을 보입니다[^1_13]
- **Fourier 기반 접근:** F-Net(2026)은 주의 메커니즘을 Fourier 기반 표현 학습으로 대체하여 파라미터를 1000배 이상 감소시켰습니다[^1_14]
- **하이브리드 Convolutional-Attention:** EffiCANet(2024)은 대형 커널 분해 합성곱과 전역 시간-변수 주의를 결합하여 효율성을 개선합니다[^1_15]


## 6. 2020년 이후 관련 최신 연구 비교 분석

### 주요 발전 동향

**2020-2021:**

1. **Graph Attention Network (2020):** 다변량 시계열 이상 탐지에 그래프 주의를 적용하여 시간적 차원과 특징 차원 모두에서 복잡한 의존성을 학습합니다. DSANet과 달리 그래프 구조를 명시적으로 모델링합니다.[^1_12]
2. **Multi-Attention RNN (2020):** Input-attention, self-attention, temporal-convolution-attention을 결합한 RNN 기반 접근으로, DSANet과 유사하게 합성곱과 주의를 결합하지만 여전히 RNN에 의존합니다.[^1_16]

**2022-2023:**

3. **First De-Trend then Attend (2022):** 시계열을 먼저 de-trend한 후 주의를 적용하는 방법으로, 계절적 패턴을 더 잘 포착합니다. 이는 DSANet이 다루지 않은 전처리 전략을 제시합니다.[^1_10]
4. **TSMixer (2023):** 주의 메커니즘 없이 순수 MLP 기반 아키텍처로 경쟁력 있는 성능을 달성하여, DSANet의 복잡한 주의 메커니즘이 항상 필요하지 않을 수 있음을 시사합니다.[^1_17][^1_8]
5. **Spatiotemporal Self-Attention LSTNet (2023):** 공간적 및 시간적 자기 주의를 LSTNet에 통합하여 DSANet의 이중 분기 개념을 발전시킵니다.[^1_18]

**2024:**

6. **Attention as Robust Representation (2024):** 주의 가중치를 시계열의 주요 표현으로 사용하여 노이즈와 비정상성에 대한 강건성을 향상시켜 MSE를 3.6% 감소시킵니다.[^1_19]
7. **Scaling Transformers (2024):** 사전 학습된 대규모 Transformer 모델(Timer_XL, LLM4TS_FS)이 작은 task-specific 모델보다 우수한 성능을 보이며, 특히 주의 메커니즘을 통한 시간적 패턴 인식과 전이 학습이 효과적임을 입증합니다.[^1_2]

**2025-2026:**

8. **VARMAformer (2025):** VARMA 통계 모델의 통찰력을 Transformer에 융합하여 전역 장기 의존성과 지역 통계 구조를 모두 포착합니다. DSANet의 하이브리드 접근을 더 정교하게 발전시킵니다.[^1_6]
9. **MMformer (2025):** 자기 주의와 메타 학습을 결합한 적응형 예측 프레임워크로, 다양한 데이터 분포에 대한 일반화 능력을 크게 향상시킵니다. 이는 DSANet의 일반화 한계를 해결합니다.[^1_9]
10. **AltTS (2026):** Autoregression과 cross-relation 모델링을 분리하는 이중 경로 프레임워크로 alternating optimization을 사용합니다. DSANet의 이중 분기 개념을 최적화 관점에서 재해석합니다.[^1_5]
11. **vLinear (2026):** vecTrans 모듈로 $\mathcal{O}(N)$ 복잡도를 달성하고 WFMLoss를 도입하여 22개 벤치마크에서 SOTA 성능을 달성합니다. DSANet의 계산 복잡도 문제를 효과적으로 해결합니다.[^1_3]
12. **F-Net (2026):** 주의 메커니즘을 Fourier 기반 표현 학습으로 대체하여 파라미터를 $\sim 10^3$개로 줄이고 SOTA 대비 28.18% 성능 향상을 달성합니다.[^1_14]

### 비교 요약

| 측면 | DSANet (2019) | 최신 연구 (2020-2026) |
| :-- | :-- | :-- |
| **아키텍처** | Dual Conv + Self-Attention | Transformer 변형, MLP-only, Hybrid[^1_2][^1_8][^1_15] |
| **계산 복잡도** | $\mathcal{O}(N^2)$ | $\mathcal{O}(N)$ 달성 (vLinear, F-Net)[^1_3][^1_14] |
| **일반화** | 단일 도메인 | 대규모 사전 학습, 메타 학습[^1_2][^1_9] |
| **장기 예측** | 최대 24 horizon | 720+ horizon 지원[^1_2] |
| **해석가능성** | 제한적 | 주의 패턴 분석, 시각화 도구[^1_7][^1_12] |

DSANet은 시계열 예측에서 RNN 대신 합성곱과 자기 주의를 결합한 선구적 연구로, 이후 Transformer 기반 모델들의 발전을 촉진했습니다. 그러나 최신 연구들은 계산 효율성, 일반화 성능, 장기 예측 능력 측면에서 DSANet의 한계를 극복하고 있습니다.[^1_4][^1_14][^1_3][^1_2][^1_1]
<span style="display:none">[^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39]</span>

<div align="center">⁂</div>

[^1_1]: 3357384.3358132.pdf

[^1_2]: https://arxiv.org/html/2507.02907v1

[^1_3]: https://www.semanticscholar.org/paper/6586515c642a7d1ad7bc3edf1e3e261ce3397db1

[^1_4]: https://ieeexplore.ieee.org/document/11352790/

[^1_5]: https://www.semanticscholar.org/paper/8877218316156926916da0cc8e7b3b3e4f941f38

[^1_6]: https://arxiv.org/abs/2509.04782

[^1_7]: https://www.semanticscholar.org/paper/7535430803f65326977156b894bb4d4ff75c140d

[^1_8]: https://arxiv.org/pdf/2306.09364.pdf

[^1_9]: https://www.sciencedirect.com/science/article/abs/pii/S1568494625014036

[^1_10]: https://arxiv.org/pdf/2212.08151.pdf

[^1_11]: https://www.semanticscholar.org/paper/58461d3e54d54bf026d4488489a3f9091a4fca05

[^1_12]: https://ieeexplore.ieee.org/document/9338317/

[^1_13]: http://arxiv.org/pdf/2410.24023.pdf

[^1_14]: https://ieeexplore.ieee.org/document/11060930/

[^1_15]: https://arxiv.org/html/2411.04669v1

[^1_16]: https://ieeexplore.ieee.org/document/9219721/

[^1_17]: https://arxiv.org/pdf/2303.06053.pdf

[^1_18]: https://onlinelibrary.wiley.com/doi/10.1155/2023/9523230

[^1_19]: https://arxiv.org/pdf/2402.05370.pdf

[^1_20]: https://arxiv.org/pdf/2307.05909.pdf

[^1_21]: https://ar5iv.labs.arxiv.org/html/2007.06028

[^1_22]: https://arxiv.org/html/2510.08202v1

[^1_23]: https://www.biorxiv.org/content/10.64898/2026.01.07.698166v1.full.pdf

[^1_24]: https://arxiv.org/pdf/2508.08101.pdf

[^1_25]: https://pdfs.semanticscholar.org/d0d4/2fa6fb4ab0650854f8f8080f7b7c8a4dd88a.pdf

[^1_26]: https://arxiv.org/pdf/2307.09543.pdf

[^1_27]: https://www.biorxiv.org/content/10.1101/2024.09.13.612861v1.full.pdf

[^1_28]: https://linkinghub.elsevier.com/retrieve/pii/S147403462501105X

[^1_29]: https://link.springer.com/10.1007/s11227-026-08295-x

[^1_30]: http://arxiv.org/pdf/1809.02105.pdf

[^1_31]: https://arxiv.org/html/2407.13806v1

[^1_32]: http://arxiv.org/pdf/2408.09723.pdf

[^1_33]: https://dl.acm.org/doi/10.1145/3663976.3664241

[^1_34]: https://www.techscience.com/cmc/online/detail/25861/pdf

[^1_35]: https://openreview.net/forum?id=Mu18gwLAnk

[^1_36]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12453695/

[^1_37]: https://www.sciencedirect.com/science/article/abs/pii/S0925231222007330

[^1_38]: https://openreview.net/forum?id=hHg7sc02R6

[^1_39]: https://yanglin1997.github.io/files/TCAN.pdf

