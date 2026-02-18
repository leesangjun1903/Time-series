# Kolmogorov-Arnold Networks (KANs) for Time Series Analysis

## 1. 핵심 주장과 주요 기여 요약

이 논문은 **Kolmogorov-Arnold Networks(KANs)를 시계열 예측에 최초로 적용**한 선구적 연구로, KANs가 실세계 위성 트래픽 예측 과제에서 기존 MLP를 능가하면서도 훨씬 적은 학습 파라미터를 사용함을 실증적으로 증명합니다. 주요 기여는 다음과 같습니다:[^1_1]

- **KANs의 시계열 적용 최초 검증**: 학술적으로 탐구되지 않았던 영역을 개척[^1_1]
- **MLP 대비 우월한 파라미터 효율성**: KAN(4-depth)은 109k 파라미터로 MLP(4-depth) 329k 파라미터 대비 3배 효율적[^1_1]
- **적응형 활성화 함수의 예측 우위 입증**: 고정된 ReLU 대신 학습 가능한 B-spline 함수 활용[^1_1]
- **Ablation study 제공**: 노드 수(n)와 그리드 크기(G)가 성능에 미치는 영향 체계적 분석[^1_1]

***

## 2. 해결 문제, 제안 방법, 모델 구조, 성능 및 한계

### 해결하고자 하는 문제

기존 MLP 기반 딥러닝 모델(LSTM, CNN 등)은 **파라미터 비효율적인 스케일링 법칙**을 갖고 있으며, 레이어 수 증가 시 파라미터가 비선형적으로 증가합니다. 또한 블랙박스 특성으로 인해 **해석 가능성이 낮다**는 한계가 있습니다. 이 논문은 위성 통신 시스템의 실제 트래픽 예측을 대상으로 이 문제에 접근합니다.[^1_1]

### 수학적 기반 및 제안 방법

**Kolmogorov-Arnold 표현 정리**에 따르면, $^n$의 유계 도메인 위의 임의의 연속 다변수 함수 $f(\mathbf{x})$는 단변수 함수의 유한 합성으로 표현 가능합니다:[^1_1]

$f(\mathbf{x}) = \sum_{i=1}^{2n+1} \Phi_i\left(\sum_{j=1}^{n} \phi_{i,j}(x_j)\right) \tag{1}$

여기서 $\Phi_i : \mathbb{R} \to \mathbb{R}$는 외부 함수, $\phi_{i,j} :  \to \mathbb{R}$는 내부 함수입니다.[^1_1]

**KAN 레이어**는 훈련 가능한 B-spline 단변수 함수 $\{\phi_{i,j}(\cdot)\}$로 이루어진 행렬 $\Phi$로 정의되며, 일반화된 L-layer KAN은 다음과 같이 표현됩니다:

$\mathbf{y} = \text{KAN}(\mathbf{x}) = (\Phi_L \circ \Phi_{L-1} \circ \cdots \circ \Phi_1)\mathbf{x} \tag{2}$

B-spline은 다항식 차수 $k=3$, 그리드 인터벌 수 $G$로 정의되며 모든 연산이 미분 가능하여 역전파 학습이 가능합니다.[^1_1]

**시계열 예측 프레임워크**는 컨텍스트 길이 $c$의 과거 입력을 기반으로 예측 길이 $T$의 미래 값을 추정합니다:

$\hat{y}\_{t_0:T} \approx f(x_{t_0-c:t_0-1}) = (\Phi_2 \circ \Phi_1)\mathbf{x} \tag{3}$

손실 함수는 예측 구간의 **MAE(Mean Absolute Error)** 최소화를 목표로 하며, Adam optimizer(lr=0.001), 500 epochs로 훈련합니다.[^1_1]

### 모델 구조

논문에서 비교한 구성은 다음과 같습니다:


| 모델 | 구조 | 스플라인 | 활성화 함수 | 파라미터 수 |
| :-- | :-- | :-- | :-- | :-- |
| MLP (3-depth) | [^1_2] | N/A | ReLU (고정) | 238k |
| MLP (4-depth) | [^1_2] | N/A | ReLU (고정) | 329k |
| KAN (3-depth) | [^1_3][^1_3][^1_2] | B-spline, k=3, G=5 | 학습 가능 | 93k |
| KAN (4-depth) | [^1_3][^1_3][^1_3][^1_2] | B-spline, k=3, G=5 | 학습 가능 | 109k |

컨텍스트/예측 길이는 168시간/24시간(1주일 → 1일 예측)이며, 6개 위성 빔 영역의 실제 GEO 위성 데이터로 평가하였습니다.[^1_1]

### 성능 결과

| 모델 | MSE (×10⁻³) | RMSE (×10⁻²) | MAE (×10⁻²) | MAPE |
| :-- | :-- | :-- | :-- | :-- |
| MLP (3-depth) | 6.34 | 7.96 | 5.41 | 0.64 |
| MLP (4-depth) | 6.12 | 7.82 | 5.55 | 1.05 |
| KAN (3-depth) | 5.99 | 7.73 | 5.51 | 0.62 |
| **KAN (4-depth)** | **5.08** | **7.12** | **5.06** | **0.52** |

KAN(4-depth)은 MLP(4-depth) 대비 **MSE 17% 감소, 파라미터 67% 절감**을 달성합니다.[^1_1]

### 한계

- **단일 도메인 검증**: 위성 트래픽 데이터 하나에만 국한되어 범용성 확인 부족[^1_1]
- **LSTM/CNN 미비교**: KANs가 초기 단계임을 인정하며 복잡한 아키텍처와의 비교는 향후 과제로 남겨둠[^1_1]
- **지속적 학습(Continual Learning) 미평가**: 원 KAN 논문에서 언급된 가능성이지만 본 연구에서는 다루지 않음[^1_1]
- **계산 비용**: 노드 수 $n$과 그리드 $G$가 커질수록 학습 시간과 연산량이 크게 증가[^1_1]

***

## 3. 모델의 일반화 성능 향상 가능성

논문은 KANs의 일반화 성능에 대해 다음과 같은 긍정적 단서를 제시합니다:[^1_1]

1. **다양한 트래픽 조건에서의 강건성**: 낮은 트래픽(beam 2), 높고 가변적인 트래픽(beam 3) 모두에서 KAN이 MLP보다 안정적으로 실제값을 추종하며, 이는 **다양한 스케일/강도의 데이터에 대한 일반화 잠재력**을 시사합니다.
2. **2가지 자유도(Degrees of Freedom)**: KAN은 MLP처럼 특성(feature)을 학습하는 **외부 자유도**와 spline처럼 단변수 함수를 최적화하는 **내부 자유도**를 동시에 가집니다. 이 이중 구조가 새로운 분포에 대한 적응력을 높이는 요인으로 작용합니다.
3. **파라미터 효율성과 과적합 위험 감소**: KAN(4-depth)은 MLP(4-depth) 대비 $\frac{1}{3}$ 수준의 파라미터로 더 나은 성능을 보이므로, 제한된 데이터 환경에서도 과적합 없이 일반화 성능을 유지할 가능성이 높습니다.
4. **Ablation study의 시사점**: $n=20, G=20$ 조합이 가장 좋은 성능을 보이지만, 노드 수가 적을 때($n=5$) 그리드를 키우면 오히려 역효과가 발생합니다. 이는 **적절한 용량 설계가 일반화에 중요**함을 보여주고, 과적합과 과소적합 사이의 균형을 spline 파라미터로 세밀하게 조정할 수 있음을 의미합니다.

후속 연구들도 이 가능성을 확인합니다. 대형 호수 클로로필-a 농도 예측에서 KAN은 LSTM, GRU, MLP 등 모든 비교 모델을 능가하며 **미관측 데이터(forecast phase)에서도 우수한 일반화**를 보였습니다. 또한 WormKAN은 KAN 기반 모델이 금융·의료 등 개념 변화(concept drift)가 있는 시계열에서도 강건하게 패턴을 세분화할 수 있음을 보입니다.[^1_4][^1_2]

***

## 4. 연구 영향 및 향후 고려사항

### 향후 연구에 미치는 영향

이 논문은 KAN의 시계열 분야 적용 가능성을 최초로 열어, 불과 수개월 만에 다양한 후속 아키텍처를 촉발했습니다:[^1_5]

- **T-KAN / MT-KAN** (2024): 개념 변화 감지 및 다변량 시계열 해석 가능성 강화[^1_6][^1_7]
- **TKAN** (2024): LSTM + KAN 결합으로 다단계 시계열 예측 정확도 향상[^1_8]
- **TimeKAN** (2025): 주파수 분해 + KAN으로 복잡한 다중 주파수 패턴 처리[^1_9][^1_10]
- **HaKAN** (2025): Hahn 다항식 기반 KAN으로 글로벌/로컬 시간 패턴 동시 포착[^1_11][^1_12]
- **TSKANMixer** (2025): TSMixer에 KAN 레이어 통합, 다중 데이터셋에서 성능 향상[^1_13]
- **KANMixer** (2025): 28개 실험 중 16개에서 최고 성능 달성[^1_14]
- **AR-KAN**: ARIMA의 자기회귀 기억력 + KAN 비선형성 결합[^1_15]
- **DiffKANformer**: 예측, 이상 탐지, 분류 등 4가지 시계열 태스크 통합 모델[^1_16]


### 향후 연구 시 고려할 점

1. **다양한 벤치마크 데이터셋 검증**: 단일 위성 도메인을 넘어 ETT, Weather, Exchange 등 표준 시계열 벤치마크에서의 비교 실험이 필요합니다. 실제로 LSTM 대비 KAN의 열위를 보고한 연구도 존재하여, 도메인별 적합성 분석이 필수입니다.[^1_17]
2. **하이브리드 아키텍처 설계**: 단독 KAN보다 Transformer, CNN, LSTM과의 결합이 더 강력한 성능을 보이는 경우가 많습니다. KAN을 기존 아키텍처의 **선형 레이어 대체 모듈**로 활용하는 설계 전략이 유망합니다.[^1_18][^1_19]
3. **주파수 도메인 통합**: 시간 도메인 위주의 현재 KAN 연구에서 벗어나, 주파수 도메인 분석을 접목하면 주기성이 강한 데이터에서 추가적인 성능 향상이 기대됩니다.[^1_20]
4. **Continual Learning 및 Concept Drift 대응**: 이 논문이 명시적으로 다루지 않은 지속적 학습 능력과 개념 변화 대응은 실시간 트래픽 관리, 금융 등 실세계 응용에 핵심적입니다.[^1_2]
5. **스플라인 파라미터 자동 최적화**: $k$(다항식 차수), $G$(그리드 크기), $n$(노드 수)의 수동 튜닝 부담을 줄이기 위한 AutoML 또는 Neural Architecture Search(NAS) 접목이 실용화의 관건입니다.
6. **계산 비용 및 추론 속도 최적화**: KAN의 spline 연산은 MLP보다 느릴 수 있어, 실시간 시스템 배포를 위한 FastKAN, EfficientKAN 등의 경량화 변형 연구가 중요합니다.[^1_21]
7. **이론적 일반화 보장**: 현재는 실험적 증거에 의존하므로, PAC-learning 관점에서의 샘플 복잡도 분석이나 VC 차원 이론 적용 등 **이론적 일반화 보장**을 위한 연구가 필요합니다.
<span style="display:none">[^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46]</span>

<div align="center">⁂</div>

[^1_1]: 2405.08790v2.pdf

[^1_2]: https://www.semanticscholar.org/paper/10f394afbb5356c235a8b221f2bae0a88b1d3254

[^1_3]: https://www.nature.com/articles/s41598-025-07654-7

[^1_4]: https://pubs.acs.org/doi/10.1021/acs.est.4c11113

[^1_5]: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5220118

[^1_6]: https://arxiv.org/abs/2406.02496

[^1_7]: http://arxiv.org/pdf/2406.02496.pdf

[^1_8]: https://arxiv.org/abs/2405.07344

[^1_9]: https://arxiv.org/abs/2502.06910

[^1_10]: http://arxiv.org/pdf/2502.06910.pdf

[^1_11]: https://arxiv.org/pdf/2601.18837.pdf

[^1_12]: https://arxiv.org/html/2601.18837v1

[^1_13]: https://arxiv.org/abs/2502.18410

[^1_14]: https://arxiv.org/abs/2508.01575

[^1_15]: https://arxiv.org/html/2509.02967v2

[^1_16]: https://openreview.net/forum?id=v9CIPqun2Z

[^1_17]: https://arxiv.org/abs/2511.18613

[^1_18]: https://arxiv.org/html/2602.11190v1

[^1_19]: https://ieeexplore.ieee.org/document/10925058/

[^1_20]: https://arxiv.org/html/2506.12696v1

[^1_21]: https://arxiv.org/html/2411.14904v1

[^1_22]: https://arxiv.org/pdf/2502.18410.pdf

[^1_23]: https://arxiv.org/html/2601.22690v1

[^1_24]: https://arxiv.org/html/2509.25826v1

[^1_25]: https://arxiv.org/html/2601.02310v1

[^1_26]: https://arxiv.org/html/2410.10393v2

[^1_27]: https://www.arxiv.org/pdf/2511.18613.pdf

[^1_28]: https://www.mdpi.com/2227-7390/12/19/3022

[^1_29]: https://arxiv.org/abs/2407.15236

[^1_30]: https://ieeexplore.ieee.org/document/10924997/

[^1_31]: https://www.frontiersin.org/articles/10.3389/frwa.2025.1638839/full

[^1_32]: http://arxiv.org/pdf/2405.08790.pdf

[^1_33]: http://arxiv.org/pdf/2406.17890.pdf

[^1_34]: https://arxiv.org/pdf/2410.14939.pdf

[^1_35]: http://arxiv.org/pdf/2408.11306.pdf

[^1_36]: http://arxiv.org/pdf/2411.00278.pdf

[^1_37]: http://arxiv.org/abs/2405.08790

[^1_38]: https://www.semanticscholar.org/paper/Kolmogorov-Arnold-Networks-(KANs)-for-Time-Series-Vaca-Rubio-Blanco/081eb8781725e560f597b01c624fe65618c3c0f8

[^1_39]: https://axi.lims.ac.uk/paper/2406.02496

[^1_40]: https://www.datasciencewithmarco.com/blog/kolmogorov-arnold-networks-kans-for-time-series-forecasting

[^1_41]: https://www.semanticscholar.org/paper/Kolmogorov-Arnold-Networks-for-Time-Series:-Power-Xu-Chen/10145b2238569436754c4d9be3f9c7db501cc65c

[^1_42]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12222760/

[^1_43]: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4825654

[^1_44]: https://axi.lims.ac.uk/paper/2408.11306

[^1_45]: https://arxiv.org/html/2408.11306v1

[^1_46]: https://openreview.net/forum?id=LWQ4zu9SdQ

