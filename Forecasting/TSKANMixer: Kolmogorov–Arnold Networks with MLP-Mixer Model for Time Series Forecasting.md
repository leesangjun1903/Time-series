# TSKANMixer: Kolmogorov–Arnold Networks with MLP-Mixer Model for Time Series Forecasting

## 1. 핵심 주장 및 주요 기여 요약

TSKANMixer는 Amazon 연구팀(Hong, Xiao, Chen)이 2025년 AAAI에 제출한 논문으로, **KAN(Kolmogorov-Arnold Networks)을 TSMixer(MLP-Mixer 기반 시계열 모델)에 통합**하는 것이 시계열 예측 성능을 향상시킬 수 있음을 실험적으로 입증합니다. 핵심 주장은 "고정 활성화 함수를 사용하는 MLP를 학습 가능한 스플라인 기반 활성화 함수를 쓰는 KAN 레이어로 대체하거나 보완하면, 시계열의 복잡한 비선형 패턴 포착 능력이 향상된다"는 것입니다.[^1_1][^1_2]

**주요 기여:**

- TSMixer에 KAN 레이어를 적용한 두 가지 아키텍처(v01, v02)를 최초로 제안[^1_1]
- ETT, NN5, CIF-2016, Hospital, Exchange, FRED-MD 등 10개 벤치마크 데이터셋에 걸친 포괄적 평가[^1_1]
- KAN이 시계열 분야에서 MLP의 유망한 대안임을 실증적으로 증명[^1_3]
- 과적합 없이 일반화된 패턴 학습이 가능함을 손실 곡선으로 시각화[^1_1]

***

## 2. 해결하고자 하는 문제, 제안 방법(수식), 모델 구조

### 해결하고자 하는 문제

Transformer 기반 모델들은 긴 시계열에서 과적합 문제가 심각하여 단순 선형 모델보다 성능이 낮은 역설이 지적되어 왔습니다. TSMixer가 MLP 기반으로 이를 어느 정도 해결했지만, 고정된 활성화 함수라는 MLP의 구조적 한계는 여전히 복잡한 비선형 시간 패턴 포착을 제한합니다. TSKANMixer는 이 문제를 KAN의 **학습 가능한 활성화 함수**로 해결하고자 합니다.[^1_4][^1_1]

### 수식: 이론적 기반 — 콜모고로프-아놀드 표현 정리

임의의 다변수 연속 함수 $f(\mathbf{x})$는 다음과 같이 단변수 함수의 유한 합성으로 표현 가능합니다:[^1_1]

$f(x_1, \ldots, x_n) = \sum_{j=1}^{2n+1} \Phi_j \left( \sum_{i=1}^{n} \phi_{j,i}(x_i) \right) \tag{1}$

여기서 $\Phi_j: \mathbb{R} \to \mathbb{R}$ 은 외부 함수(outer function), $\phi_{j,i}:  \to \mathbb{R}$ 은 내부 함수(inner function)입니다.[^1_1]

**KAN 레이어 정의** — 입력 $n_{in}$개, 출력 $n_{out}$개인 KAN 레이어는 단변수 함수의 행렬로 정의됩니다:[^1_1]

$\mathbf{\Phi} = \{ \phi_{j,i} \}, \quad i = 1, \ldots, n_{in}, \; j = 1, \ldots, n_{out} \tag{2}$

**다층 KAN 표현** — $L$개의 KAN 레이어를 쌓으면:[^1_1]

$y = \text{KAN}(\mathbf{x}) = (\mathbf{\Phi}_{L-1} \circ \cdots \circ \mathbf{\Phi}_1 \circ \mathbf{\Phi}_0)(\mathbf{x}) \tag{3}$

각 $\phi_{j,i}$는 **B-스플라인**으로 파라미터화되며, 스플라인 차수 $k$(다항식 차수)와 격자 간격 수 $G$로 제어됩니다. MLP와 비교하면 KAN은 $k \times G$배 더 많은 파라미터를 갖습니다.[^1_5][^1_1]

### 모델 구조: TSKANMixer v01 \& v02

**TSKANMixer v01** — Temporal Projection의 Fully-Connected(FC) 레이어를 KAN으로 **대체**:[^1_1]

> 입력 길이 $L$에서 예측 지평선 $H$로의 매핑을 KAN이 담당하며, 과거 입력과 미래 예측 간의 복잡한 비선형 관계를 학습

**TSKANMixer v02** — Mixer 레이어와 Temporal Projection 사이에 KAN 기반 Time-Mixing 레이어를 **추가**:[^1_1]

> 기존 FC Temporal Projection은 유지하고, 추가된 KAN Time-Mixing이 시간 영역의 패턴 탐색 능력을 강화

두 버전 모두 **2-깊이(2-depth) KAN 레이어**를 사용하며, 이는 식 (1)의 콜모고로프-아놀드 정리의 $[n, 2n+1, 1]$ 구조에 대응합니다.[^1_1]

***

## 3. 성능 향상 및 한계

### 성능 향상

8개 데이터셋에서 TSMixer 대비 성능이 향상되었으며, 가장 두드러진 결과는 ETTh2에서 TSKANMixer v02의 MSE **18.97% 감소**, MAE **9.41% 감소**입니다. 두 버전 모두 10개 데이터셋 중 7회씩 상위 3위 안에 들었습니다.[^1_3][^1_1]


| 데이터셋 | TSKANMixer v01 MSE | TSKANMixer v02 MSE | TSMixer MSE |
| :-- | :-- | :-- | :-- |
| ETTh1 | 0.285 (+33.57%) | 0.398 (+2.69%) | 0.429 |
| ETTh2 | 0.199 (-2.05%) | **0.158** (+18.97%) | 0.195 |
| ETTm1 | **0.190** (+34.26%) | 0.281 (+2.77%) | 0.289 |
| ETTm2 | 0.131 (+9.66%) | **0.109** (+24.83%) | 0.145 |
| NN5 daily | 0.521 (-1.36%) | 0.506 (+1.56%) | 0.514 |
| Exchange | 0.017 (+5.56%) | **0.016** (+11.11%) | 0.018 |

*(괄호 안은 TSMixer 대비 MSE 개선율; 양수가 개선을 의미)*[^1_1]

### 한계

- **학습 속도**: KAN은 같은 파라미터 수 대비 MLP보다 약 10배 느리며, TSKANMixer는 TSMixer 대비 최대 **50배 느린 학습 시간**을 기록[^1_1]
- **하이퍼파라미터 탐색 어려움**: B-스플라인 격자 수, 차수, KAN 히든 크기 등 KAN 고유 파라미터가 추가되어 그리드 서치가 계산적으로 비실용적[^1_1]
- **일부 데이터셋 한계**: CIF 2016 및 FRED-MD 데이터셋에서는 TSMixer 대비 성능이 하락[^1_1]

***

## 4. 일반화 성능 향상 가능성

TSKANMixer의 가장 주목할 만한 특성 중 하나는 **과적합에 대한 내성**입니다. ETTh2 학습 곡선 분석에 따르면, TSMixer는 50 에폭 이전에 최적값에 도달하지만 이후 검증 손실이 빠르게 증가(과적합)하는 반면, TSKANMixer는 초기 손실이 높더라도 검증 손실이 지속적으로 감소하며 **로컬 최적값에 빠지지 않고 일반화된 패턴**을 학습합니다.[^1_1]

이는 KAN의 스플라인 기반 학습 가능 활성화 함수가 데이터의 근본적인 구조를 유연하게 적합화하기 때문으로, 다음과 같은 정규화 기법과 결합하면 일반화 성능이 더욱 향상됩니다:[^1_1]

- **드롭아웃(Dropout)**: KAN 훈련 수렴 가속 및 과적합 방지
- **Weight Decay**: L2 정규화로 스플라인 파라미터 복잡도 제어
- **배치 정규화(Batch Normalization)**: 레이어 간 분포 안정화
- **베이지안 최적화(Bayesian Optimization)**: 파라미터 탐색 공간 효율화

또한 Vaca-Rubio et al.의 연구는 KAN이 MLP보다 **더 적은 파라미터로 더 낮은 오차**를 달성함을 보여, 파라미터 효율성이 일반화에도 기여함을 시사합니다.[^1_5]

***

## 5. 관련 최신 연구 비교 분석 (2020년 이후)

| 모델 | 연도 | 핵심 아이디어 | KAN 활용 | 주요 한계 |
| :-- | :-- | :-- | :-- | :-- |
| **TSMixer** (Chen et al.) | 2023 | MLP Mixer로 시·공간 패턴 동시 포착 [^1_6] | ✗ | 고정 활성화 함수 |
| **TKAN** (Genet \& Inzirillo) | 2024 | KAN+LSTM 결합, RKAN 메모리 레이어 [^1_7] | ✓ | 순환 구조의 학습 속도 |
| **T-KAN / MT-KAN** (Xu et al.) | 2024 | Concept Drift 탐지 + 심볼릭 회귀 해석 [^1_8] | ✓ | 다변량 복잡도 처리 |
| **KANs for TS** (Vaca-Rubio et al.) | 2024 | 위성 트래픽에서 MLP 대비 파라미터 효율성 증명 [^1_5] | ✓ | 특정 도메인에 국한 |
| **C-KAN** | 2024 | CNN+KAN 통합, DILATE 손실 [^1_9] | ✓ | 비정상 시계열 한정 |
| **TSKANMixer** (본 논문) | 2025 | MLP-Mixer에 KAN 통합, 두 가지 아키텍처 제안 [^1_1] | ✓ | 학습 속도 50배 느림 |
| **TimeKAN** (Huang et al.) | 2025 | 주파수 분해 + KAN, 다중 주파수 성분 처리 [^1_10] | ✓ | 계산 복잡도 |
| **HaKAN** | 2025 | Hahn 다항식 KAN, inter/intra-patch 레이어 [^1_11] | ✓ | 실험 범위 제한 |

TSKANMixer는 기존 KAN 연구들이 KAN을 **독립형 모델**로 사용한 것과 달리, 검증된 MLP-Mixer 프레임워크에 **통합 부품으로 KAN을 삽입**한 점이 차별화됩니다.[^1_12][^1_1]

***

## 6. 향후 연구에 미치는 영향 및 고려할 점

### 연구에 미치는 영향

TSKANMixer는 **"기존 MLP 기반 시계열 모델에서 MLP를 KAN으로 교체하는 접근법"의 실현 가능성과 타당성**을 공개 벤치마크로 입증함으로써, 이후 연구들이 동일한 패러다임을 Transformer, GNN, TCN 등 다양한 아키텍처에 적용할 근거를 마련합니다. 특히 심볼릭 회귀를 통한 **KAN의 해석 가능성(interpretability)** 활용이 향후 시계열 예측의 불투명성 문제를 해결할 유망한 방향으로 제시됩니다.[^1_13][^1_3][^1_1]

### 앞으로 연구 시 고려할 점

1. **학습 효율화**: 현재 최대 50배의 학습 속도 차이는 실용적 장벽이므로, 더 효율적인 KAN 구현(예: FastKAN, Chebyshev-KAN 등)이나 지식 증류(Knowledge Distillation) 기법 적용이 시급합니다[^1_1]
2. **더 깊고 넓은 KAN 탐색**: 본 논문은 2-depth KAN에 한정되어 있으며, 더 깊거나 넓은 KAN이 성능을 더 향상시킬 가능성이 남아 있습니다[^1_1]
3. **자동화된 하이퍼파라미터 최적화**: 스플라인 차수, 격자 수 등 KAN 고유 파라미터의 최적 조합을 찾기 위한 NAS(Neural Architecture Search)나 베이지안 최적화 체계화가 필요합니다[^1_1]
4. **비정상성(Non-stationarity) 처리**: CIF 2016 실패 사례는 TSKANMixer가 특정 비정상 시계열에 취약함을 보여주므로, 시계열 분해(decomposition)나 가역 정규화(RevIN)와의 결합을 고려해야 합니다[^1_1]
5. **심볼릭 회귀 기반 해석 가능성 연구**: KAN의 학습된 활성화 함수를 심볼릭 수식으로 변환하면 예측 메커니즘을 설명할 수 있어 금융·의료 분야의 신뢰성 있는 예측에 기여할 수 있습니다[^1_8][^1_14]
6. **확률론적 예측(Probabilistic Forecasting)으로 확장**: P-KAN(Probabilistic KAN)처럼  예측 불확실성을 정량화하는 TSKANMixer의 확장은 실제 의사결정 지원 시스템에서 중요한 연구 방향입니다[^1_15]
7. **그래프 신경망(GNN)과의 결합**: FRED-MD와 같이 거시경제 변수 간 복잡한 상호의존성을 가진 데이터셋에서의 성능 한계를 극복하기 위해 KAN과 GNN의 결합이 유망합니다[^1_16]
<span style="display:none">[^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44]</span>

<div align="center">⁂</div>

[^1_1]: 2502.18410v2.pdf

[^1_2]: https://arxiv.org/abs/2502.18410

[^1_3]: https://arxiv.org/html/2502.18410v1

[^1_4]: https://www.semanticscholar.org/paper/TSMixer:-An-all-MLP-Architecture-for-Time-Series-Chen-Li/59694c8dce4f13db2f486eb8102459a3f7c23da6

[^1_5]: https://ieeexplore.ieee.org/document/11100692/

[^1_6]: https://arxiv.org/abs/2303.06053

[^1_7]: https://arxiv.org/abs/2405.07344

[^1_8]: http://arxiv.org/pdf/2406.02496.pdf

[^1_9]: https://www.mdpi.com/2227-7390/12/19/3022

[^1_10]: http://arxiv.org/pdf/2502.06910.pdf

[^1_11]: https://arxiv.org/pdf/2601.18837.pdf

[^1_12]: http://arxiv.org/pdf/2408.11306.pdf

[^1_13]: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5220118

[^1_14]: https://axi.lims.ac.uk/paper/2406.02496

[^1_15]: https://arxiv.org/abs/2510.16940

[^1_16]: https://ir.bjut.edu.cn/irpui/item/ir/435535

[^1_17]: https://arxiv.org/abs/2406.02496

[^1_18]: https://arxiv.org/pdf/2502.18410.pdf

[^1_19]: https://arxiv.org/html/2506.12696v1

[^1_20]: https://www.semanticscholar.org/paper/Kolmogorov-Arnold-networks-for-time-series-a-review-Yamak-Li/44e39edc11dc6ab7ac768f1d01dfc362a137f319

[^1_21]: https://arxiv.org/html/2505.08199v1

[^1_22]: https://arxiv.org/pdf/2404.19756.pdf

[^1_23]: https://arxiv.org/abs/2411.00278

[^1_24]: https://arxiv.org/html/2410.16032v1

[^1_25]: https://www.arxiv.org/pdf/2510.16940.pdf

[^1_26]: https://arxiv.org/pdf/2412.17176.pdf

[^1_27]: https://www.semanticscholar.org/paper/Kolmogorov-Arnold-Networks-(KAN)-for-Time-Series-Dong-Zheng/b0992e50f0360fe60fe1436d6a748acf4c259345

[^1_28]: https://pubs.acs.org/doi/10.1021/acs.est.4c11113

[^1_29]: https://link.springer.com/10.1007/s10586-025-05574-9

[^1_30]: https://arxiv.org/abs/2509.02967

[^1_31]: https://www.ssrn.com/abstract=4825654

[^1_32]: http://arxiv.org/pdf/2405.08790.pdf

[^1_33]: http://arxiv.org/pdf/2406.17890.pdf

[^1_34]: http://arxiv.org/pdf/2411.00278.pdf

[^1_35]: https://arxiv.org/pdf/2406.02486.pdf

[^1_36]: https://www.datasciencewithmarco.com/blog/kolmogorov-arnold-networks-kans-for-time-series-forecasting

[^1_37]: https://openreview.net/forum?id=wbpxTuXgm0

[^1_38]: https://www.semanticscholar.org/paper/Kolmogorov-Arnold-Networks-(KANs)-for-Time-Series-Vaca-Rubio-Blanco/081eb8781725e560f597b01c624fe65618c3c0f8

[^1_39]: https://research.google/blog/tsmixer-an-all-mlp-architecture-for-time-series-forecasting/

[^1_40]: https://openreview.net/forum?id=LWQ4zu9SdQ

[^1_41]: https://storage.prod.researchhub.com/uploads/papers/2024/04/09/3580305.3599533.pdf

[^1_42]: https://icml.cc/virtual/2025/poster/45584

[^1_43]: https://arxiv.org/abs/2306.09364

[^1_44]: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4825654

