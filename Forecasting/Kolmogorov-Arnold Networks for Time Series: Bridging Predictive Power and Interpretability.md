# Kolmogorov-Arnold Networks for Time Series: Bridging Predictive Power and Interpretability

## 1. 핵심 주장 및 주요 기여 요약

이 논문(Xu et al., 2024)은 MIT 팀이 제안한 **Kolmogorov-Arnold Networks(KAN)**을 시계열 예측에 최초로 체계적으로 적용하고, **T-KAN**과 **MT-KAN** 두 가지 변형 모델을 제안합니다. MLP 기반 모델의 고질적 문제인 해석 불가능성(black-box)과 스케일링 비효율을 동시에 해결하는 것이 핵심 주장이며, 적은 파라미터로도 경쟁력 있는 예측 정확도와 높은 해석 가능성을 모두 달성할 수 있음을 실험으로 검증합니다.[^1_1]

***

## 2. 해결하고자 하는 문제

기존 MLP 기반 딥러닝 모델은 두 가지 근본적 한계를 가집니다:[^1_1]

- **해석 불가능성**: 금융·의료·기상 등 의사결정 분야에서 모델이 왜 그런 예측을 했는지 설명할 수 없음
- **개념 드리프트(Concept Drift) 미감지**: 시계열 데이터의 통계적 특성이 시간에 따라 변화할 때 기존 모델은 이를 탐지하거나 적응하지 못함
- **파라미터 비효율**: 높은 성능을 위해 과도한 파라미터가 필요하며 스케일링 법칙이 불량함

***

## 3. 이론적 기반 및 제안 방법 (수식 포함)

### KAN의 이론적 기반

**Kolmogorov-Arnold 표현 정리**에 의하면, 임의의 다변량 연속 함수는 단변량 연속 함수의 합성으로 분해 가능합니다:[^1_1]

$f(x_1, \ldots, x_n) = \sum_{q=1}^{2n+1} \Phi_q \left( \sum_{p=1}^{n} \phi_{q,p}(x_p) \right)$

여기서 $\phi_{q,p}$는 각 입력 변수 $x_p$에 대한 단변량 함수이며, $\Phi_q$는 연속 함수입니다.

### KAN 레이어 및 심층 구조

KAN 레이어 $\Phi = \{\phi_{q,p}\}$를 여러 층으로 쌓은 심층 KAN은 다음과 같이 정의됩니다:[^1_1]

$\text{KAN}(x) = (\Phi_{L-1} \circ \Phi_{L-2} \circ \cdots \circ \Phi_0)(x) \tag{2}$

기존 MLP가 **노드(node)**에 고정 활성화 함수를 배치하는 것과 달리, KAN은 **엣지(edge)**에 학습 가능한 B-스플라인(B-spline) 곡선을 배치합니다. 이로써 그리드를 세밀화할수록 임의의 목표 함수에 무한히 가까워질 수 있습니다.[^1_1]

***

## 4. 모델 구조

### T-KAN (Temporal KAN) — 단변량 시계열용

슬라이딩 윈도우로 입력-출력 쌍을 구성하며, 시각 $t$에서의 예측값 $\hat{S}_{t+T}$는 다음과 같습니다:[^1_1]

$\hat{S}\_{t+T} = \sum_{q=1}^{2n+1} \Phi_q \left( \sum_{p=1}^{h} \phi_{q,p}(S_{t-h+p}) \right)$

- 2-layer, 5 hidden neuron 구조 `[84, 5, 21]`
- **Symbolic Regression**을 통해 학습된 활성화 함수를 수학 표현식(예: $\hat{S}\_{t+1} = e^{S_t^2} + \sin(3.14 \cdot S_{t-1})$ )으로 변환, 인간이 읽을 수 있는 해석 제공[^1_1]
- 서로 다른 개념(concept)에서 학습된 KAN들의 활성화 패턴 변화를 비교하여 **개념 드리프트 탐지** 수행


### MT-KAN (Multivariate Temporal KAN) — 다변량 시계열용

$k$번째 변수의 과거 관측값 $S_{t-h+p,k}$를 활용하는 MT-KAN의 예측 수식:[^1_1]

$\hat{S}\_{t+T} = \sum_{q=1}^{2n+1} \Phi_q \left( \sum_{p=1}^{h} \sum_{k=1}^{m} \phi_{q,p,k}(S_{t-h+p,k}) \right) $

- 모든 변수의 히스토리를 flatten하여 입력으로 활용: `[84×5, 5, 21×5]`
- 교차 변수 상호작용(cross-variable interaction)을 명시적으로 모델링
- 각 노드의 출력이 모든 변수와 모든 시점과의 관계로 역추적 가능

***

## 5. 성능 향상 결과

실험은 2012~2022년 Nasdaq 금융 시계열(OHCLV 데이터, 503개 종목)에서 수행되었습니다:[^1_1]


| 모델 | 구성 | MSE | MAE | RMSE | 파라미터 수 |
| :-- | :-- | :-- | :-- | :-- | :-- |
| MLP | [^1_2] | 8.92e-5 | 0.0072 | 0.0088 | 21,221 |
| RNN | [^1_2] | 8.03e-5 | 0.0069 | 0.0083 | 44,821 |
| LSTM | [^1_2] | 6.69e-5 | 0.0066 | 0.0078 | 11,671 |
| **T-KAN** | **[^1_3][^1_2]** | **6.91e-5** | **0.0069** | **0.0078** | **193** |
| **MT-KAN** | **[84×5,5,21×5]** | **6.37e-5** | **0.0062** | **0.0075** | **2,132** |

T-KAN은 단 **193개의 파라미터**로 11,671개 파라미터의 LSTM과 동등한 성능을 달성했으며, MT-KAN은 최고 성능을 기록했습니다. 이는 KAN의 **파라미터 효율성**을 명확히 보여줍니다.[^1_1]

***

## 6. 일반화 성능 향상 가능성

이 논문에서 일반화 성능 향상과 직결되는 요소는 다음과 같습니다:[^1_1]

- **그리드 정제(Grid Refinement)**: B-스플라인의 그리드를 세밀하게 조정할수록 목표 함수에 임의로 근접 가능하여 과적합 없이 표현력 향상 가능
- **가지치기(Pruning)**: 임계값($5 \times 10^{-2}$) 기반 프루닝 후 재학습 파이프라인이 모델을 간결하게 유지하여 일반화에 기여
- **Symbolic Regression 정규화 효과**: 학습된 함수를 수식으로 근사화하는 과정이 일종의 정규화 역할을 하여 노이즈에 강건
- **적은 파라미터 수**: MLP 대비 수십~수백 배 적은 파라미터는 과적합 위험을 근본적으로 낮춤
- **개념 드리프트 적응**: 시계열 분포 변화에 동적으로 대응하는 T-KAN 구조 자체가 OOD(out-of-distribution) 일반화에 유리

다만, 논문은 금융 데이터셋에 한정된 실험을 수행하였으므로, **다양한 도메인 및 데이터셋에 대한 일반화 검증은 부족**하다는 한계가 있습니다.[^1_1]

***

## 7. 한계점

- **학습 속도**: 동일 파라미터 수의 MLP 대비 약 **10배 느린 학습 속도** - 배치 처리(batch processing)를 충분히 활용하지 못하는 다양한 활성화 함수 구조 때문[^1_1]
- **실험 범위 협소**: 금융 시계열 단일 도메인에만 실험 한정
- **최신 딥러닝 모델 비교 부재**: Transformer 계열 (PatchTST, iTransformer 등)과의 비교 없이 MLP, RNN, LSTM 등 고전 모델과만 비교[^1_1]
- **고차원 데이터 확장성**: 변수가 매우 많은 시계열에서의 확장성 미검증

***

## 8. 향후 연구에 미치는 영향과 고려 사항

### 연구 파급 효과

이 논문은 KAN의 시계열 적용 가능성을 선구적으로 제시하여 이후 수십 편의 후속 연구를 촉발했습니다. 주요 후속 연구로는 다음이 있습니다:

- **TSKANMixer** (2025): TSMixer의 MLP를 KAN으로 대체하여 다중 데이터셋에서 예측 정확도 향상[^1_4]
- **TimeKAN** (2025): 다중 주파수 성분 분해에 KAN을 적용하는 경량 아키텍처[^1_5]
- **TFKAN** (2025): 시간 도메인과 주파수 도메인을 동시에 처리하는 이중 브랜치 KAN[^1_6]
- **KAN-AD** (2024): 시계열 이상 탐지에 KAN의 해석 가능성을 활용[^1_7]
- **C-KAN** (2024): CNN과 KAN을 결합하여 비정상 시계열 예측 성능 강화[^1_8]
- **KANMixer** (2025): 멀티스케일 믹싱과 KAN을 결합, 28개 실험 중 16개에서 SOTA 달성[^1_9]


### 2020년 이후 관련 연구 비교 분석

| 연구 | 연도 | 핵심 기여 | KAN 활용 방식 |
| :-- | :-- | :-- | :-- |
| **T-KAN / MT-KAN** (본 논문) | 2024 | 개념 드리프트 탐지 + 해석 가능성 | 순수 KAN, Symbolic Regression |
| **KANs for TS Analysis** [Vaca-Rubio et al.] | 2024 | 위성 트래픽 예측, MLP 대비 우위 | 단일 KAN 레이어 [^1_10] |
| **TKAN** [Genet \& Inzirillo] | 2024 | 시계열 전용 KAN 구조 제안 | 순환 구조와 KAN 결합 [^1_7] |
| **C-KAN** | 2024 | 비정상 시계열 + DILATE 손실 | CNN + KAN 하이브리드 [^1_8] |
| **TSKANMixer** | 2025 | MLP-Mixer 구조 강화 | MLP 대체 [^1_4] |
| **TimeKAN** | 2025 | 주파수 분해 기반 경량 예측 | 주파수 밴드별 KAN [^1_5] |
| **TFKAN** | 2025 | 시간+주파수 이중 도메인 | 이중 브랜치 KAN [^1_6] |
| **KANMixer** | 2025 | 멀티스케일 믹싱 SOTA | 적응적 기저 함수 KAN [^1_9] |

후속 연구들은 본 논문이 제시한 **"KAN = 해석 가능성 + 예측력"** 프레임워크를 계승하면서도, 주파수 분해, Transformer 통합, 이상 탐지 등 다양한 방향으로 확장하고 있습니다.[^1_11]

### 향후 연구 시 고려할 점

1. **학습 속도 병목 해결**: 활성화 함수를 "heads"로 그룹화하거나 병렬 처리를 통해 배치 연산 효율화 필요[^1_1]
2. **비교 베이스라인 확장**: PatchTST, iTransformer, TimesNet 등 최신 Transformer 기반 모델과의 공정한 비교 필요[^1_12]
3. **다도메인 일반화 검증**: 금융 외 의료·에너지·교통 등 도메인에서 일반화 성능 체계적 평가 필요
4. **RNN/LSTM/Transformer와의 하이브리드**: KAN의 해석 가능성을 유지하면서 시퀀스 모델링 능력 결합 — 이미 TKAN, TFKAN 등에서 시도[^1_6][^1_1]
5. **적응적 시퀀스 분할(Adaptive Segmentation)**: 비정상(non-stationary) 데이터에서 시계열을 동적으로 분할하여 KAN의 개념 드리프트 대응력 강화
6. **정규화 및 스플라인 최적화**: 고차원 데이터에서의 과적합 방지를 위한 메타학습(meta-learned) 기반 스플라인 초기화 전략 연구[^1_13]
7. **공정한 파라미터 비교**: 동일 파라미터 수 또는 FLOPs 기준 비교 시 KAN이 MLP 대비 항상 우위에 있지 않다는 점을 인식하고 실험 설계  — KAN의 강점은 **기호 회귀 및 해석 가능성 요구 태스크**에 집중됨[^1_14]
<span style="display:none">[^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45]</span>

<div align="center">⁂</div>

[^1_1]: 2406.02496v1.pdf

[^1_2]: https://arxiv.org/html/2506.12696v1

[^1_3]: https://arxiv.org/abs/2406.02496

[^1_4]: https://arxiv.org/abs/2502.18410

[^1_5]: https://arxiv.org/abs/2502.06910

[^1_6]: https://arxiv.org/abs/2506.12696

[^1_7]: https://dl.acm.org/doi/10.1145/3743128

[^1_8]: https://www.mdpi.com/2227-7390/12/19/3022

[^1_9]: https://arxiv.org/abs/2508.01575

[^1_10]: http://arxiv.org/abs/2405.08790

[^1_11]: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5220118

[^1_12]: https://arxiv.org/abs/2511.18613

[^1_13]: https://ir.bjut.edu.cn/irpui/item/ir/435535

[^1_14]: https://arxiv.org/html/2407.16674v1

[^1_15]: https://pubs.acs.org/doi/10.1021/acs.est.4c11113

[^1_16]: https://iwaponline.com/jh/article/27/3/560/107135/GKASA-DDPM-a-novel-flood-forecasting-model-based

[^1_17]: http://arxiv.org/pdf/2405.08790.pdf

[^1_18]: http://arxiv.org/pdf/2406.02496.pdf

[^1_19]: http://arxiv.org/pdf/2502.06910.pdf

[^1_20]: http://arxiv.org/pdf/2406.17890.pdf

[^1_21]: http://arxiv.org/pdf/2408.11306.pdf

[^1_22]: https://arxiv.org/pdf/2502.18410.pdf

[^1_23]: http://arxiv.org/pdf/2411.00278.pdf

[^1_24]: https://arxiv.org/pdf/2502.00980.pdf

[^1_25]: https://arxiv.org/pdf/2601.18837.pdf

[^1_26]: https://arxiv.org/html/2412.17853v2

[^1_27]: https://arxiv.org/html/2504.13593v2

[^1_28]: https://arxiv.org/pdf/2504.16432.pdf

[^1_29]: https://arxiv.org/html/2504.13593v4

[^1_30]: https://arxiv.org/html/2601.02310v1

[^1_31]: https://arxiv.org/html/2504.13593v1

[^1_32]: https://arxiv.org/html/2509.18483v1

[^1_33]: https://arxiv.org/html/2410.03027v1

[^1_34]: https://www.semanticscholar.org/paper/Kolmogorov-Arnold-networks-for-time-series-a-review-Yamak-Li/44e39edc11dc6ab7ac768f1d01dfc362a137f319

[^1_35]: https://arxiv.org/html/2409.10594v1

[^1_36]: https://arxiv.org/html/2602.11190v1

[^1_37]: https://icml.cc/virtual/2025/poster/45584

[^1_38]: https://axi.lims.ac.uk/paper/2406.02496

[^1_39]: https://www.semanticscholar.org/paper/Kolmogorov-Arnold-Networks-for-Time-Series:-Power-Xu-Chen/10145b2238569436754c4d9be3f9c7db501cc65c

[^1_40]: https://openreview.net/pdf?id=1x0eJ8uUx6

[^1_41]: https://www.semanticscholar.org/paper/Kolmogorov-Arnold-Networks-(KANs)-for-Time-Series-Vaca-Rubio-Blanco/081eb8781725e560f597b01c624fe65618c3c0f8

[^1_42]: https://www.sciencedirect.com/science/article/pii/S1110866525000593

[^1_43]: https://openreview.net/forum?id=LWQ4zu9SdQ

[^1_44]: https://ieeexplore.ieee.org/iel8/6287639/10820123/11131148.pdf

[^1_45]: https://5g-stardust.eu/wp-content/uploads/sites/97/2025/01/GLOBECOM_WS_KANs-1.pdf

