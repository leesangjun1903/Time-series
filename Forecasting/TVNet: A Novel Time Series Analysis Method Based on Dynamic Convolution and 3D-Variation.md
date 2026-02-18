# TVNet: A Novel Time Series Analysis Method Based on Dynamic Convolution and 3D-Variation

## 1. 핵심 주장과 주요 기여 요약

TVNet(ICLR 2025)은 기존 Transformer 및 MLP 기반 모델이 지배하던 시계열 분석 분야에서, CNN의 잠재력을 재발굴한 연구입니다. **1D 시계열 데이터를 3D 텐서로 변환**하여 intra-patch, inter-patch, cross-variable 세 가지 차원의 의존성을 동시에 포착한다는 것이 핵심 주장입니다.[^1_1][^1_2]

주요 기여는 다음 세 가지입니다:[^1_1]

- **3D-Embedding 기법**: 1D 시계열을 3D 형태로 재구성하는 새로운 임베딩 방법 제안
- **동적 합성곱(Dynamic Convolution) 기반 TVNet 아키텍처**: 비디오 처리 기법에서 영감을 받은 시간 적응형 가중치 메커니즘 도입
- **5가지 태스크 SOTA 달성**: 장·단기 예측, 결측값 보간, 분류, 이상 탐지에서 Transformer/MLP 모델 대비 우수한 성능·효율 균형

***

## 2. 해결하고자 하는 문제

### 기존 방법론의 한계

| 방법론 | 장점 | 한계 |
| :-- | :-- | :-- |
| Transformer (iTransformer, PatchTST 등) | 장거리 의존성 포착 | 이차 복잡도 $O(L^2)$, 메모리 과다 소비 [^1_1] |
| MLP (DLinear, RLinear 등) | 경량, 고효율 | 다변수 간 복잡한 의존성 포착 어려움 [^1_1] |
| CNN (TimesNet, MICN 등) | 효율적 | 단일 시간 창 내 특징 분석에 집중, 전역·지역·변수 간 통합 부족 [^1_1] |

### TVNet이 주목한 아이디어

시계열 데이터는 **비디오 프레임**과 유사하게 연속적 시간 변화를 담고 있습니다. 이 통찰을 바탕으로, 영상 처리에서 효과적으로 활용되던 동적 합성곱(dynamic convolution)을 시계열에 도입하여 **모드 드리프트(mode drift)** 현상에 적응적으로 대응합니다.[^1_1]

***

## 3. 제안 방법과 수식

### 3.1 3D-Embedding

입력 시계열 $X_{in} \in \mathbb{R}^{L \times C}$에 대해 먼저 특징 차원을 임베딩 차원 $C_m$으로 확장합니다. 이후 Conv1D (kernel size = $P$)로 $N = L/P$개의 패치로 분할하고, 각 패치를 홀수(odd)/짝수(even) 인덱스로 분리·스택하여 3D 텐서를 생성합니다:[^1_1]

$X_{emb} = \text{3D-Embedding}(X_{in}) \in \mathbb{R}^{C_m \times N \times 2 \times (P/2)} \tag{1}$

이 구조가 intra-patch(패치 내), inter-patch(패치 간), cross-variable(변수 간) 세 차원의 상호작용을 자연스럽게 구현합니다.[^1_1]

### 3.2 3D-Block: 동적 합성곱

$i$번째 패치의 출력은 동적 가중치 $\alpha_i$와 기저 가중치 $W_b$의 곱으로 구성됩니다:[^1_1]

$\tilde{x}_i = W_i \cdot x_i = (\alpha_i \cdot W_b) \cdot x_i \tag{2}$

시간 변화 가중치 $\alpha_i$는 전체 패치 임베딩 $X_{emb}$로부터 다음과 같이 생성됩니다:[^1_1]

$\alpha_i = G(X_{emb}) = 1 + F(v_{inter}) + F(v_{intra}) \tag{3}$

- **Intra-patch 특징** — 3D Adaptive Average Pooling으로 패치 내부 기술 벡터 추출:

$v_{intra} = \text{AdaptiveAvgPool3d}(X_{emb}) \in \mathbb{R}^{C_m \times N} \tag{4}$

$F_{intra}(v_{intra}) = \delta(\text{BN}(\text{Conv1D}\_{C \to C}(v_{intra}))) \tag{5}$

- **Inter-patch 특징** — 1D Adaptive Pooling으로 패치 간 정보 집약:

$v_{inter} = \text{AdaptiveAvgPool1d}(v_{intra}) \in \mathbb{R}^{C_m \times 1} \tag{6}$

$F_{inter}(v_{inter}) = \delta(\text{Conv1D}\_{C \to C}(v_{inter})) \tag{7}$

여기서 $\delta$는 ReLU 활성화 함수입니다.[^1_1]

### 3.3 전체 구조 (잔차 연결)

3D-block은 잔차(residual) 방식으로 스택됩니다:[^1_1]

$X_{i+1}^{3D} = \text{3D-block}(X_i^{3D}) + X_i^{3D} \tag{8}$

최종적으로 $X^{3D} \in \mathbb{R}^{C_m \times N \times 2 \times (P/2)}$를 $(NP) \times C_m$으로 reshape한 후 태스크별 선형 헤드(task-linear head)로 출력을 생성합니다.[^1_1]

### 3.4 복잡도 분석

$\text{FLOPs} = O(L C_m^2), \quad \text{Space} = O(C_m^2 + LC_m) \tag{9}$

Transformer의 $O(L^2)$ 공간 복잡도와 달리, TVNet의 공간 복잡도는 시퀀스 길이 $L$에 독립적입니다.[^1_1]

***

## 4. 성능 향상 결과

### 5개 태스크 성능 요약

| 태스크 | TVNet 성능 | 주요 경쟁 모델 대비 |
| :-- | :-- | :-- |
| 장기 예측 (9개 데이터셋) | ETTm1 MSE 0.348, ETTh2 MSE 0.324 | PatchTST, iTransformer, ModernTCN 대부분 초과 [^1_1] |
| 단기 예측 (M4) | SMAPE 11.671, OWA 0.832 | PatchTST(OWA 0.851), ModernTCN(OWA 0.838) 상회 [^1_1] |
| 결측값 보간 | ETTm1 MSE 0.018, Weather MSE 0.024 | Cross-variable 포착으로 PatchTST·DLinear 압도 [^1_1] |
| 분류 | 평균 정확도 **74.6%** | ModernTCN 74.2%, PatchTST 72.5% 상회 [^1_1] |
| 이상 탐지 | 평균 F1 **86.8%** | PatchTST 86.6%, FEDformer 85.0% 초과 [^1_1] |

### 효율성

ETTm2 데이터셋 기준, TVNet은 **2.1GB / 25.2s/epoch**으로 TimesNet(5.7GB / 89.3s), FEDformer(7.2GB / 133.5s)에 비해 월등히 효율적입니다.[^1_1]

### Ablation 결과

동적 합성곱 제거 시 Weather MSE가 0.147 → 0.251로 급등하여, 동적 가중치 메커니즘이 핵심 성능 원천임을 확인했습니다.[^1_1]

***

## 5. 일반화 성능 향상 가능성

TVNet의 일반화 능력은 다음 세 측면에서 두드러집니다:[^1_3][^1_1]

### 전이학습(Transfer Learning)

ETTh1 → ETTh2, ETTm2 전이 실험에서 TVNet의 Direct Prediction 및 Full-Tuning 모두 PatchTST, FEDformer를 능가했습니다.  
예를 들어 ETTh2 전이 시, TVNet Full-Tuning MAE는 0.327 ~ 0.422 범위로 PatchTST(0.337~0.450)보다 일관되게 낮았습니다.  
이는 **동적 가중치 메커니즘이 다양한 데이터 분포에 적응적으로 반응**하기 때문입니다.[^1_1]

### 하이퍼파라미터 견고성

패치 길이 $P \in \{8, 12, 24, 32\}$ 변화에 따른 성능 차이가 미미하여(ETTm1 MAE 0.379±0.004), **하이퍼파라미터에 대한 강건성**이 검증되었습니다.[^1_1]

### 이론적 근거 (동적 가중치 우위 정리)

고정 가중치 모델의 총 오차 $E_f = \sum_{i=1}^{N}(W_f \cdot x_i - y_i^\*)^2$에 비해, 동적 가중치 모델은 $\alpha_i = y_i^* / (W_b \cdot x_i)$ 조건에서 최적화되어 $E_d < E_f$가 항상 성립합니다. 이는 일반화 성능 향상의 수학적 토대를 제공합니다.[^1_1]

### 한계

- 임베딩 차원 $C_m$ 과 패치 길이 $P$의 조합에서 **극단값(너무 크거나 작은 경우)은 성능 저하** 유발[^1_1]
- **대규모 사전학습(pre-training) 실험 부재**: 저자들도 향후 과제로 명시[^1_1]
- LLM 기반 모델(S2IP-LLM 등)과의 비교에서 **일부 데이터셋(Electricity 96 step)**에서 소폭 뒤처지는 케이스 존재[^1_1]

***

## 6. 관련 최신 연구 비교 분석 (2020년 이후)

| 모델 | 연도 | 방법론 유형 | 핵심 아이디어 | TVNet과의 차이 |
| :-- | :-- | :-- | :-- | :-- |
| **Informer** | 2021 | Transformer | ProbSparse self-attention, $O(L \log L)$ 복잡도 [^1_1] | TVNet이 복잡도 및 성능 측면 우위 |
| **Autoformer** | 2021 | Transformer | 자동상관 분해, 계절-추세 분리 [^1_1] | TVNet이 장기 예측 MSE에서 전반 우위 |
| **TimesNet** | 2022 | CNN | 1D → **2D** 변환 후 2D Conv 적용 [^1_1] | TVNet은 **3D** 변환 + 동적 가중치로 표현력 강화 |
| **PatchTST** | 2022 | Transformer | 패치(patch) 단위 Transformer, 지역 특징 포착 [^1_1] | TVNet이 전이학습 및 보간 태스크에서 우위 |
| **iTransformer** | 2023 | Transformer | 역전(inverted) 어텐션으로 변수 간 의존성 모델링 [^1_1] | TVNet이 효율성 측면 우위 |
| **ModernTCN** | 2024 | CNN | 대형 합성곱 커널로 글로벌 특징 포착 [^1_1] | TVNet이 동적 가중치로 분포 변화 대응 강화 |
| **TSLANet** | 2024 | CNN+Freq | 스펙트럼 어댑티브 블록 + CNN 경량화 [^1_4] | TVNet은 주파수 분해 없이 3D 구조로 유사 효과 |
| **WaveTuner** | 2025 | Wavelet | 웨이블릿 서브밴드 전체 스펙트럼 튜닝 [^1_5] | TVNet은 도메인 변환 없이 원시 데이터 처리 |
| **TimeDistill** | 2025 | KD+MLP | Teacher(CNN/Transformer)에서 MLP로 지식 증류 [^1_6] | TVNet은 단일 모델로 효율-성능 균형 달성 |


***

## 7. 미래 연구에 미치는 영향과 고려 사항

### 연구 영향

**CNN 재평가 패러다임 전환**: TVNet은 "시계열 분야에서 CNN은 Transformer보다 열등하다"는 통념을 뒤집으며, CNN을 1등 시민으로 재정립했습니다. 특히 3D 텐서 재구성이라는 아이디어는 이후 WaveTuner, PatchMLP 등의 후속 연구에서 patch 기반 설계의 중요성으로 이어지고 있습니다.[^1_5][^1_3][^1_1]

**대규모 사전학습 기반 마련**: 저자들은 TVNet을 기반으로 한 **시계열 파운데이션 모델(Foundation Model)** 연구와 **멀티스케일 패치** 설계를 미래 연구로 명시했으며, 이는 Time-LLM, GPT4TS 등 LLM 기반 시계열 연구와의 융합 가능성을 시사합니다.[^1_1]

### 향후 연구 시 고려 사항

1. **대규모 사전학습 적용**: TVNet의 동적 가중치 구조는 Masked Autoencoder(MAE) 방식의 사전학습과 결합할 경우, cross-domain 전이학습 성능이 크게 향상될 가능성이 있습니다.[^1_1]
2. **멀티스케일 패치 설계**: 현재 단일 패치 길이 $P$를 사용하나, 계층적 또는 가변 패치(multi-resolution patch)를 도입하면 단·장기 패턴을 동시에 더 잘 포착할 수 있습니다.
3. **외생 변수(exogenous variable) 통합**: TimeXer(2024) 논문에서 제기한 외생 변수 예측에서 TVNet이 경쟁력을 보였으나, 이를 명시적으로 모델링하는 확장이 실용 응용에 필수적입니다.[^1_1]
4. **이론적 분석 심화**: 현재 Theorem B.1은 단순 quadratic loss 기준의 증명으로, **비선형 활성화 함수 하에서의 일반화 오차 경계(generalization bound)** 도출이 필요합니다.
5. **희소·비정형 시계열 대응**: 결측률이 50% 이상이거나 불규칙 샘플링 데이터에 대한 검증이 부족합니다. 패치 분할 방식이 불규칙 구간에서 어떻게 작동하는지 추가 연구가 요구됩니다.
6. **LLM과의 융합**: S2IP-LLM, TimeLLM 대비 일부 태스크에서 열위가 있는 만큼, TVNet의 3D-Embedding을 LLM의 입력 인코더로 활용하는 하이브리드 아키텍처 탐색이 유망합니다.[^1_5][^1_1]
<span style="display:none">[^1_10][^1_11][^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_7][^1_8][^1_9]</span>

<div align="center">⁂</div>

[^1_1]: 2503.07674v1.pdf

[^1_2]: https://iclr.cc/virtual/2025/poster/29927

[^1_3]: https://arxiv.org/abs/2503.07674

[^1_4]: https://arxiv.org/html/2404.08472v1

[^1_5]: https://arxiv.org/pdf/2511.18846.pdf

[^1_6]: https://arxiv.org/pdf/2502.15016.pdf

[^1_7]: https://www.arxiv.org/list/cs.LG/2025-03?skip=25\&show=1000

[^1_8]: https://arxiv.org/html/2503.03262v2

[^1_9]: https://arxiv.org/html/2409.14737v3

[^1_10]: http://arxiv.org/list/cs/2022-01?skip=95\&show=2000

[^1_11]: https://arxiv.org/html/2502.15016v2

[^1_12]: https://arxiv.org/html/2504.04011v1

[^1_13]: https://arxiv.org/html/2505.08199v2

[^1_14]: https://arxiv.org/html/2502.10721v1

[^1_15]: https://arxiv.org/html/2506.08977v1

[^1_16]: https://arxiv.org/html/2503.07674v1

[^1_17]: https://arxiv.org/html/2601.02694v1

[^1_18]: https://arxiv.org/html/2502.14045v1

[^1_19]: https://arxiv.org/pdf/2507.10098.pdf

[^1_20]: https://arxiv.org/pdf/2404.08472.pdf

[^1_21]: https://arxiv.org/html/2403.01493v1

[^1_22]: http://arxiv.org/pdf/2405.12038.pdf

[^1_23]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11402102/

[^1_24]: http://arxiv.org/pdf/2405.05499.pdf

[^1_25]: https://arxiv.org/pdf/1711.08200.pdf

[^1_26]: https://www.mdpi.com/1424-8220/21/2/603/pdf

[^1_27]: https://arxiv.org/pdf/1708.05038.pdf

[^1_28]: https://proceedings.iclr.cc/paper_files/paper/2025/file/58b9a640af6d69781e90969d936e87ce-Paper-Conference.pdf

[^1_29]: https://iclr.cc/virtual/2025/papers.html

[^1_30]: https://www.themoonlight.io/en/review/tvnet-a-novel-time-series-analysis-method-based-on-dynamic-convolution-and-3d-variation

[^1_31]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12453695/

[^1_32]: https://proceedings.iclr.cc/paper_files/paper/2025/hash/58b9a640af6d69781e90969d936e87ce-Abstract-Conference.html

[^1_33]: https://www.sciencedirect.com/science/article/pii/S1574013725001595

[^1_34]: https://openreview.net/pdf/25169aec774382829c964a95b14d00f74a7b67a0.pdf

[^1_35]: https://proceedings.iclr.cc/paper_files/paper/2025

[^1_36]: https://towardsdatascience.com/influential-time-series-forecasting-papers-of-2023-2024-part-1-1b3d2e10a5b3/

[^1_37]: https://openreview.net/forum?id=MZDdTzN6Cy

