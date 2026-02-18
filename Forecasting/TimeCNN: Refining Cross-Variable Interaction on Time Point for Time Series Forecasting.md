# TimeCNN: Refining Cross-Variable Interaction on Time Point for Time Series Forecasting

## 1. 핵심 주장 및 주요 기여

**TimeCNN**은 기존 Transformer 기반 모델들이 다변량 시계열의 **동적이고 다면적인(양·음 상관관계)** 교차 변수 상관관계를 효과적으로 포착하지 못한다는 문제를 해결하기 위해 제안된 모델입니다.[^1_1]

### 핵심 주장

- **iTransformer의 한계**: 전체 시계열을 단일 변수 토큰으로 임베딩하여 시간에 따라 변하는 변수 관계(동적 상관관계)를 무시하며, softmax 기반 어텐션이 음의 상관관계를 사실상 소멸시킨다[^1_1]
- **TimeCNN의 혁신**: "시간점 독립(Timepoint-Independent)" 전략으로, **각 타임포인트마다 독립적인 합성곱 커널**을 할당해 변수 간 관계를 시점별로 독립적으로 모델링한다[^1_2][^1_1]


### 주요 기여 3가지

- 기존 Transformer 계열이 복잡한 동적 다변량 상관관계를 제대로 포착하지 못해 일반화 능력이 저하됨을 실증적으로 관찰[^1_1]
- **CrossCNN** 블록을 핵심으로 하는 TimeCNN 아키텍처 제안 — 양·음 상관관계 모두 포착 가능[^1_1]
- 12개의 실제 데이터셋 실험에서 SOTA 달성, iTransformer 대비 연산량 **60.46% 감소**, 파라미터 **57.50% 감소**, 추론 속도 **3~4배 향상**[^1_3][^1_1]

***

## 2. 해결하고자 하는 문제

### 문제 정의

다변량 시계열 $X = [x^{(1)}, x^{(2)}, ···, x^{(L)}] \in \mathbb{R}^{L \times N}$ 이 주어졌을 때, 미래 시퀀스 $Y = [x^{(L+1)}, ···, x^{(L+T)}] \in \mathbb{R}^{T \times N}$ 를 예측하는 과제에서:[^1_1]

1. **동적 상관관계 미포착**: 변수 간 상관관계는 시간에 따라 변화하지만(Figure 1의 Pearson 상관계수 시각화), iTransformer는 전체 시계열을 단일 토큰으로 인코딩하므로 이 동적 변화를 반영하지 못함[^1_1]
2. **음의 상관관계 소멸**: iTransformer의 self-attention은 pre-Softmax 단계에서 음의 상관관계를 인식하지만, Softmax를 거치면 **매우 작은 유사도 값**으로 변환되어 음의 상관관계 정보가 손실됨[^1_1]

***

## 3. 제안 방법 및 수식

### 전체 파이프라인

$X_{\text{crosscnn}} = \text{CrossCNN}(X) \tag{1}$
$X_{\text{trans}} = \text{Transpose}(X_{\text{crosscnn}}) \tag{2}$
$X_{\text{emb}} = \text{Embedding}(X_{\text{trans}}) \tag{3}$
$X_{\text{FFN}} = \text{FFN}(X_{\text{emb}}) \tag{4}$
$\hat{Y} = \text{Projection}(X_{\text{FFN}}) \tag{5}$

여기서 $X_{\text{crosscnn}} \in \mathbb{R}^{L \times N}$, $X_{\text{emb}}, X_{\text{FFN}} \in \mathbb{R}^{N \times D}$, $\hat{Y} \in \mathbb{R}^{T \times N}$[^1_1]

### CrossCNN 핵심 수식

$i$번째 타임포인트 입력 $x^{(i)} \in \mathbb{R}^{1 \times N}$에 대해, 패딩 후 합성곱 커널 $w^{(i)} = [w^{(i)}_1, w^{(i)}_2, \ldots, w^{(i)}_N]$을 적용하여 $j$번째 변수의 출력을 계산:

$c^{(i)}\_j = \sum_{k=1}^{N} w^{(i)}\_k \cdot \hat{x}^{(i)}_{(j+k)}, \quad j=1,2,\ldots,N, \quad i=1,2,\ldots,L$

> **핵심 통찰**: 합성곱 커널 파라미터의 **부호(sign)**가 변수 간 양·음 상관관계를 직접 표현하며, 이는 Softmax가 소멸시키던 음의 상관관계를 명시적으로 학습 가능하게 한다[^1_1]

### FFN 수식

$X_h = \text{Dropout}(\text{GELU}(\text{Dense}(\text{LayerNorm}(X_{m-1})))) \tag{7}$
$X_m = \text{Dropout}(\text{Dense}(X_h)) + X_{m-1} \tag{8}$

여기서 $X_{m-1}, X_m \in \mathbb{R}^{N \times D}$, $X_h \in \mathbb{R}^{N \times H}$ (스킵 커넥션 포함)[^1_1]

### 손실 함수 (MSE)

$\mathcal{L} = \frac{1}{T} \sum_{i=1}^{T} \left\| \hat{x}^{(L+i)} - x^{(L+i)} \right\|_2^2 \tag{9}$

***

## 4. 모델 구조

TimeCNN은 아래 4개의 주요 컴포넌트로 구성됩니다:[^1_1]


| 컴포넌트 | 역할 | 핵심 특징 |
| :-- | :-- | :-- |
| **CrossCNN** | 각 타임포인트의 변수 간 상호작용 포착 | 타임포인트별 독립 합성곱 커널, 커널 크기 = $N$ |
| **Transpose** | 시간 차원 ↔ 변수 차원 전환 | $\mathbb{R}^{L \times N} \to \mathbb{R}^{N \times L}$ |
| **FFN (M층)** | 각 변수의 시계열 표현 학습 | 공유 임베딩, GELU 활성화, 스킵 커넥션 |
| **Projection** | 최종 예측값 출력 | 선형 레이어, $\mathbb{R}^{N \times D} \to \mathbb{R}^{T \times N}$ |

또한 **Instance Norm**을 통해 훈련/테스트 셋 간의 분포 이동(distribution shift) 문제를 완화합니다.[^1_1]

***

## 5. 성능 향상 및 한계

### 성능 향상

12개의 실제 데이터셋 실험에서 TimeCNN은 ECL, Traffic, Weather, Solar-Energy, PEMS 등 전반에 걸쳐 SOTA를 달성했습니다. 특히 변수가 많은 데이터셋(Traffic: 862개, PEMS07: 883개)에서 iTransformer, PatchTST 대비 두드러진 성능 향상을 보입니다:[^1_1]


| 데이터셋 | TimeCNN MSE | iTransformer MSE | 개선율(T=96) |
| :-- | :-- | :-- | :-- |
| ECL (T=96) | **0.140** | 0.148 | ~5.4% ↓ |
| Traffic (T=96) | **0.377** | 0.395 | ~4.6% ↓ |
| PEMS03 (T=12) | **0.062** | 0.071 | ~12.7% ↓ |
| PEMS07 (T=12) | **0.055** | 0.067 | ~17.9% ↓ |

### 한계

- **ETT 계열 (소수 변수, 7개)**: 변수 수가 적은 데이터셋에서는 iTransformer 대비 우위가 크지 않으며, 교차 변수 학습의 이점이 제한적[^1_1]
- **전역 시간 의존성 모델링 부재**: CrossCNN은 타임포인트 단위로 작동하므로, 시간 축 방향의 장기 전역 패턴은 FFN에 의존하며 Transformer 어텐션의 전역 수용 영역이 없음[^1_1]
- **룩백 창 = 96 고정 실험 기준**: 기본 실험은 $L=96$으로 고정, 더 긴 창에서의 성능 향상은 별도 실험에서 확인됨[^1_1]

***

## 6. 일반화 성능 향상 가능성

TimeCNN의 일반화 향상은 구조적·전략적 근거 위에 세워져 있습니다:[^1_1]

- **Dropout 이중 적용**: CrossCNN 출력과 FFN의 선형 매핑 양측에 dropout을 적용해 과적합을 억제하고 일반화를 촉진[^1_1]
- **파라미터 효율성**: CrossCNN은 각 타임포인트당 파라미터가 $L \times N$으로 선형 증가하는 반면, CrossLinear는 $N^2$으로 이차 증가하여 변수 수가 많아질수록 과적합 위험이 큼 — CrossCNN이 이 문제를 해결[^1_1]
- **노이즈 강건성 실험**: PEMS04에서 가우시안 노이즈($\sigma$ = Noise_strength)를 증가시키는 실험에서, PatchTST는 MSE가 급격히 증가하고, iTransformer는 불안정해지는 반면, **TimeCNN은 안정적인 성능을 유지** — 변수 간 동적 상호작용이 노이즈의 영향을 완충[^1_1]
- **랜덤 시드 강건성**: 5개의 랜덤 시드(2021~2025)로 반복 실험 시, TimeCNN은 매우 낮은 표준편차를 유지하여 초기화 변동에 강건함을 확인[^1_1]
- **룩백 창 확장 시 성능 향상**: $L \in \{48, 96, 192, 336, 720\}$ 민감도 분석에서, 창이 길어질수록 예측 정확도가 지속 향상 — FFN/CrossCNN 모두 긴 창 정보를 효과적으로 활용[^1_1]
- **모듈형 적용 가능성**: CrossCNN 블록을 RMLP, PatchTST, iTransformer에 플러그인으로 삽입하면 기존 모델의 성능이 일관되게 향상(PEMS07에서 iTransformer +14.85%, PatchTST +33.33%)되어 다양한 모델에 대한 일반화 능력을 검증[^1_1]

***

## 7. 관련 최신 연구 비교 분석 (2020년 이후)

| 모델 | 연도 | 전략 | Cross-Variable | 음의 상관관계 | 동적 관계 | 계산 복잡도 |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| **Informer** | 2021 | 시간 토큰 | 암묵적 | ✗ | ✗ | $O(L \log L)$ |
| **Autoformer** | 2021 | Auto-Correlation | 암묵적 | ✗ | ✗ | $O(L \log L)$ |
| **PatchTST** | 2023 | 채널 독립 + 패치 | ✗ | ✗ | ✗ | $O(L^2/P^2)$ |
| **DLinear** | 2023 | 선형 분해 | ✗ | ✗ | ✗ | $O(L)$ |
| **iTransformer** | 2024 | 변수 토큰 + 어텐션 | ✓ | ✗ (Softmax 소멸) | ✗ | $O(N^2)$ |
| **ModernTCN** | 2024 | ConvFFN1+2 | 부분적 | 부분적 | ✗ | 고비용 |
| **VCformer** | 2024 | Variable Correlation Attention | ✓ | ✗ | 제한적 | $O(N^2)$ [^1_4] |
| **CDAM** | 2024 | 상호정보 기반 채널 혼합 | ✓ | 부분적 | ✗ | 복잡 [^1_5] |
| **TimeCNN** | 2024 | 시간점 독립 CNN | ✓ | **✓** | **✓** | $O(LN)$ |

**주요 차별점**: TimeCNN은 각 타임포인트에 독립 합성곱 커널을 할당함으로써 기존 모델들이 공통으로 가지는 **"정적 변수 관계 가정"**을 근본적으로 해결하는 첫 번째 모델로 위치합니다.[^1_6][^1_1]

***

## 8. 앞으로의 연구에 미치는 영향 및 고려 사항

### 연구적 영향

- **Timepoint-Independent 패러다임 확산**: 시계열의 변수 관계를 전역 통계가 아닌 **타임포인트 단위로 모델링**해야 한다는 새 관점을 제시하며, 후속 연구에서 이 전략을 채택하거나 확장할 가능성이 높음[^1_3][^1_1]
- **CrossCNN의 모듈화**: 기존 시계열 모델에 플러그인 방식으로 통합될 수 있는 구조 덕분에, 다양한 백본 모델의 성능을 즉시 향상할 수 있는 **범용 보조 모듈**로 활용될 수 있음[^1_1]
- **CNN vs Transformer 논쟁 재점화**: 단순 구조의 CNN이 Transformer보다 더 효율적·효과적일 수 있음을 12개 데이터셋에서 보여주며, "Are Transformers Effective for Time Series?" 논쟁의 후속 증거로 기여[^1_1]


### 앞으로 연구 시 고려할 점

1. **시간 축 의존성의 보완**: TimeCNN의 FFN은 각 변수를 독립적으로 처리하므로 **교차 시간(cross-time) 전역 패턴**에 대한 명시적 모델링이 부족 — 이를 보완하는 하이브리드 구조(e.g. CrossCNN + Temporal Attention) 연구가 필요[^1_1]
2. **소수 변수 데이터셋에서의 성능 개선**: ETT(7변수)에서 이점이 제한적이므로, 변수 수에 무관하게 일관된 성능 향상을 달성하는 적응형 타임포인트 커널 전략이 필요[^1_1]
3. **분포 이동(Distribution Shift) 심층 연구**: Instance Norm만으로는 도메인 변화에 충분히 대응하기 어려우며, Out-of-Distribution 일반화를 위한 불변 특성 학습(invariant feature learning)과의 결합이 유망[^1_7]
4. **대형 언어 모델(LLM)과의 융합**: CVTN, TimeCMA 등 최신 연구는 LLM의 사전학습 지식과 시계열 예측을 결합하는 방향을 탐구하며, CrossCNN의 경량 구조는 LLM 기반 프레임워크의 전처리 모듈로 통합할 가능성이 높음[^1_8][^1_9]
5. **스트리밍/온라인 학습 적용**: 합성곱 커널이 타임포인트별로 독립적이므로, 실시간 스트리밍 데이터에서 증분 학습(incremental learning)으로 확장하는 연구도 유망한 방향임[^1_1]
<span style="display:none">[^1_10][^1_11][^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46]</span>

<div align="center">⁂</div>

[^1_1]: 2410.04853v1.pdf

[^1_2]: https://arxiv.org/abs/2410.04853

[^1_3]: https://arxiv.org/html/2410.04853v1

[^1_4]: https://arxiv.org/abs/2405.11470

[^1_5]: https://arxiv.org/abs/2403.00869

[^1_6]: https://arxiv.org/abs/2310.06625

[^1_7]: https://arxiv.org/html/2406.09130v1

[^1_8]: https://arxiv.org/abs/2404.18730

[^1_9]: https://ojs.aaai.org/index.php/AAAI/article/view/34067

[^1_10]: https://link.springer.com/10.1007/s10489-024-05764-9

[^1_11]: https://ieeexplore.ieee.org/document/10831173/

[^1_12]: https://www.semanticscholar.org/paper/1e065aa771fc3a11e39ec5ef8567663feb0b18b8

[^1_13]: https://arxiv.org/abs/2406.01638

[^1_14]: https://ieeexplore.ieee.org/document/10688325/

[^1_15]: http://arxiv.org/pdf/2410.05726.pdf

[^1_16]: http://arxiv.org/pdf/2310.06625.pdf

[^1_17]: http://arxiv.org/pdf/2211.14730v2.pdf

[^1_18]: http://arxiv.org/pdf/2404.18730.pdf

[^1_19]: https://arxiv.org/pdf/2402.02592.pdf

[^1_20]: https://arxiv.org/pdf/2307.01616.pdf

[^1_21]: https://arxiv.org/abs/2207.05397

[^1_22]: http://arxiv.org/pdf/2405.14982.pdf

[^1_23]: https://arxiv.org/html/2505.12761v1

[^1_24]: https://arxiv.org/html/2404.18730v1

[^1_25]: https://arxiv.org/html/2508.16641v1

[^1_26]: https://arxiv.org/html/2503.01157v1

[^1_27]: https://arxiv.org/html/2505.00307v2

[^1_28]: https://arxiv.org/html/2501.08620v1

[^1_29]: https://arxiv.org/html/2507.11439v1

[^1_30]: https://arxiv.org/html/2501.08620v3

[^1_31]: https://arxiv.org/html/2507.05891v1

[^1_32]: https://arxiv.org/html/2508.12235v1

[^1_33]: https://arxiv.org/html/2412.10859v1

[^1_34]: https://arxiv.org/pdf/2507.05891.pdf

[^1_35]: https://arxiv.org/pdf/2404.18730.pdf

[^1_36]: https://github.com/thuml/iTransformer

[^1_37]: https://arxiv.org/html/2310.06625v4

[^1_38]: https://openreview.net/pdf?id=JePfAI8fah

[^1_39]: https://www.youtube.com/watch?v=SJZKNIZbhbk

[^1_40]: https://13akstjq.github.io/TIL/post/2024-07-09-iTransformerTheLatestBreakthroughinTimeSeriesForecasting

[^1_41]: https://www.datasciencewithmarco.com/blog/patchtst-a-breakthrough-in-time-series-forecasting

[^1_42]: https://www.datasciencewithmarco.com/blog/itransformer-the-latest-breakthrough-in-time-series-forecasting

[^1_43]: https://github.com/PatchTST/PatchTST

[^1_44]: https://blog.csdn.net/weixin_43145427/article/details/143720806

[^1_45]: https://iclr.cc/virtual/2024/poster/18933

[^1_46]: https://github.com/yuqinie98/PatchTST

