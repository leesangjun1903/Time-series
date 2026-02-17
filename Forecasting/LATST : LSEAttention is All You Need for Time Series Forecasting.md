# LATST : LSEAttention is All You Need for Time Series Forecasting

## 1. 핵심 주장과 주요 기여

이 논문은 Transformer 기반 시계열 예측 모델이 단순한 선형 모델보다 성능이 낮은 근본적인 원인을 **엔트로피 붕괴(entropy collapse)** 문제로 규명하고, 이를 해결하는 새로운 어텐션 메커니즘인 **LSEAttention**을 제안합니다. LATST(LSEAttention Time-Series Transformer)는 단일 레이어 구조로도 기존 Transformer 대비 19.75%, SAMformer 대비 4% 성능 향상을 달성했습니다.[^1_1]

주요 기여는 다음과 같습니다:[^1_1]

- 엔트로피 붕괴와 수치적 안정성을 연결하는 새로운 이론적 프레임워크 제시
- Log-Sum-Exp(LSE) 트릭과 GELU 활성화를 결합한 LSEAttention 메커니즘 개발
- 8개 실제 데이터셋에서 32개 시나리오 중 21개에서 최저 MSE 달성


## 2. 해결하려는 문제와 제안 방법

### 문제점: 엔트로피 붕괴의 근본 원인

전통적인 Transformer의 softmax 함수는 시계열 데이터의 급격한 변화나 노이즈에 과도하게 반응하여, 어텐션 분포가 소수의 요소에 집중되는 엔트로피 붕괴 현상을 일으킵니다. 기존 어텐션 메커니즘은 다음과 같이 정의됩니다:[^1_1]

$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$

여기서 softmax 함수는:

$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}$

이 과정에서 큰 어텐션 스코어는 지수 오버플로우(overflow)를, 작은 스코어는 언더플로우(underflow)를 발생시켜 수치적 불안정성을 초래합니다. 결과적으로 가장 큰 스코어가 계산을 지배하고, 작은 스코어들은 무시되어 모델의 표현력이 제한됩니다.[^1_1]

### 제안 방법: LSEAttention

논문은 Log-Sum-Exp(LSE) 트릭을 활용한 새로운 어텐션 메커니즘을 제안합니다. LSE 함수는 다음과 같이 정의됩니다:[^1_1]

$\text{LSE}(x) = \log\left(\sum_{i=1}^n e^{x_i}\right)$

LSE 트릭은 최댓값 $a$를 사용하여 수치적 안정성을 확보합니다:

$y = \log\left(\sum_{i=1}^n e^{x_i}\right) = \log\left(e^a \sum_{i=1}^n e^{x_i - a}\right)$

이를 단순화하면:

$y = a + \log\left(\sum_{i=1}^n e^{x_i - a}\right)$

GELU 활성화 함수를 적용하여 비선형성을 추가합니다:

$y = \text{GELU}\left(a + \log\left(\sum_{i=1}^n e^{x_i - a}\right)\right)$

최종적으로 안정화된 어텐션 가중치는:

$g_i = \exp(x_i - y) = \frac{e^{x_i}}{e^y}$

이는 원래 softmax와 동일한 형태이지만 수치적으로 안정적입니다:

$\frac{e^{x_i}}{e^{\log(\sum_{i=1}^n e^{x_i})}} = \frac{e^{x_i}}{\sum_{i=1}^n e^{x_i}} = \text{softmax}(x_i)$

최종 어텐션 가중치는 다시 softmax를 적용하여 0과 1 사이로 정규화됩니다:

$\text{softmax}(g_i)$

### 모델 구조: LATST

LATST는 다음 주요 구성요소를 포함합니다:[^1_1]

1. **Reversible Instance Normalization (RevIN)**: 학습과 테스트 데이터 분포 차이를 완화
2. **LSEAttention 모듈**: 기존 temporal self-attention을 대체
3. **Parametric ReLU (PReLU)**: Feed-Forward Network에서 dying ReLU 문제 해결
4. **Fractional Temporal Block Attention Encoder**: 64 타임스텝 블록으로 시계열을 분할하여 명시적 시간 관계 모델링

문제 정식화는 다음과 같습니다. $C$개 채널의 길이 $L$ 시계열 $X \in \mathbb{R}^{C \times L}$로부터 미래 $P$ 시점의 값 $Y \in \mathbb{R}^{C \times P}$를 예측하는 모델 $f_\omega: \mathbb{R}^{C \times L} \rightarrow \mathbb{R}^{C \times P}$를 학습합니다:[^1_1]

$\text{MSE} = \frac{1}{C}\sum_{c=1}^C \|Y_c - f_\omega(X_c)\|^2$

## 3. 모델의 일반화 성능 향상 가능성

### 일반화 성능 향상 메커니즘

LATST는 여러 측면에서 일반화 성능을 향상시킵니다:[^1_1]

**수치적 안정성 개선**: LSE 트릭은 모든 지수 항을 0과 1 사이로 제한하여 정규화 효과를 제공합니다. 이는 모델이 극단적인 값에 과적합되는 것을 방지하고, 훈련 데이터에 없던 패턴에도 안정적으로 반응할 수 있게 합니다.[^1_1]

**다양성 있는 어텐션 분포**: 기존 softmax가 소수 요소에 집중하는 것과 달리, LSEAttention은 더 균등한 어텐션 분포를 유지합니다. 이는 모델이 트렌드와 주기성을 포함한 다양한 시간적 패턴을 고려하도록 하여 일반화 능력을 높입니다.[^1_1]

**매개변수 효율성**: LATST는 단일 레이어로 구성되어 평균 271.4K 매개변수를 사용하며, 이는 일부 선형 모델(DLinear: 139.7K)보다 약간 많지만 대부분의 Transformer 모델보다 적습니다. 적은 매개변수는 과적합 위험을 줄이고 일반화 성능을 향상시킵니다.[^1_1]

### 실증적 증거

논문의 실험 결과는 강력한 일반화 능력을 보여줍니다:[^1_1]

- **데이터셋 간 일관성**: 8개의 다양한 데이터셋(ETT, Electricity, Weather, Traffic, Exchange)에서 안정적인 성능을 보임
- **예측 길이 확장성**: 96, 192, 336, 720의 다양한 예측 길이에서 일관된 성능 유지
- **변수 수 적응성**: 낮은 변수(7개)부터 초고변수(862개)까지 다양한 설정에서 효과적

특히 Traffic 데이터셋에서 LATST는 긴 예측 시점에서도 낮은 MSE를 유지하여 과적합에 대한 저항성과 훈련 조건을 넘어선 일반화 능력을 입증했습니다.[^1_1]

## 4. 성능 향상 및 한계

### 성능 향상

LATST는 여러 벤치마크에서 우수한 성능을 달성했습니다:[^1_1]


| 데이터셋 | 예측 길이 | LATST MSE | SAMformer MSE | 개선율 |
| :-- | :-- | :-- | :-- | :-- |
| ECL | 720 | 0.205 | 0.219 | 6.4% |
| Traffic | 720 | 0.433 | 0.456 | 5.0% |
| Weather | 96 | 0.146 | 0.197 | 25.9% |
| ETTh2 | 336 | 0.384 | 0.350 | -9.7% |

- **8개 데이터셋 중 6개에서 SAMformer 초과**[^1_1]
- **전체 Transformer 대비 19.75% 향상**[^1_1]
- **높은 변동성 데이터셋(Electricity, Traffic, Weather)에서 특히 효과적**[^1_1]

Ablation study 결과:[^1_1]

- GeLU 제거 시 성능 저하 (예: ETTh1 192 길이에서 0.415 → 0.433)
- PReLU가 ReLU보다 Exchange Rate에서 11.5% 더 우수 (0.935 vs 1.043)


### 한계점

논문에서 명시적으로 언급된 한계:[^1_1]

- **엔트로피 붕괴의 모든 원인을 탐구하지 못함**: 논문은 수치적 불안정성에 초점을 맞췄으나 다른 기여 요인들이 존재할 수 있음
- **일부 데이터셋에서 선형 모델에 미치지 못함**: Exchange Rate 데이터셋의 720 길이 예측에서 DLinear(0.603)이 LATST(0.935)보다 우수
- **단일 레이어 한계**: 깊은 네트워크의 표현력을 충분히 활용하지 못할 가능성

추가 관찰된 한계:

- ETTh2 데이터셋에서 일부 예측 길이(336)에서 SAMformer보다 낮은 성능
- 초장기 예측(720)에서 일부 데이터셋의 성능 격차 확대


## 5. 향후 연구에 미치는 영향과 고려사항

### 연구에 미치는 영향

**어텐션 메커니즘 재설계**: LSEAttention은 도메인별 특성을 고려한 어텐션 메커니즘 설계의 중요성을 보여줍니다. 이는 다른 시퀀스 모델링 작업(음성 인식, 동영상 분석 등)에도 적용 가능합니다.[^1_1]

**수치적 안정성의 중요성**: 논문은 수치적 안정성이 단순한 엔지니어링 이슈가 아니라 모델 성능에 직접적으로 영향을 미치는 핵심 요소임을 입증했습니다. 이는 향후 딥러닝 모델 설계 시 수치적 안정성을 우선 고려 사항으로 만들 것입니다.[^1_1]

**효율성과 성능의 균형**: 단일 레이어로 SOTA 성능에 근접한 결과는 "더 깊은 모델 = 더 좋은 성능"이라는 통념에 도전합니다. 이는 효율적인 아키텍처 설계의 새로운 방향을 제시합니다.[^1_1]

### 향후 연구 시 고려사항

**1. 다층 구조 확장 연구**
현재 LATST는 단일 레이어 구조입니다. 다층 LSEAttention의 효과와 깊이에 따른 엔트로피 붕괴 양상을 연구해야 합니다. 최근 연구에서 깊이가 증가할수록 어텐션 랭크가 지수적으로 감소함이 밝혀졌습니다.[^1_2][^1_1]

**2. 하이브리드 아키텍처 탐구**
LSEAttention을 다른 시계열 기법(분해, 주파수 분석 등)과 결합하는 연구가 필요합니다. HTMformer와 FreEformer는 각각 시간-다변량 분리와 주파수 향상으로 35.8%와 상당한 성능 개선을 달성했습니다.[^1_3][^1_4]

**3. 기초 모델(Foundation Model) 적용**
대규모 사전 학습된 시계열 모델에 LSEAttention을 적용하는 연구가 필요합니다. Timer와 Moment 같은 최신 기초 모델들이 전이 학습에서 우수한 성능을 보이고 있습니다.[^1_5]

**4. 주파수 편향(Frequency Bias) 문제**
Fredformer 연구에 따르면 Transformer는 고주파 특징을 무시하는 경향이 있습니다. LSEAttention이 이 문제를 완화하는지 검증이 필요합니다.[^1_6]

**5. 계산 효율성 최적화**
FlashAttention이나 선형 복잡도 어텐션과 LSE 트릭의 결합을 연구해야 합니다. PatchTST는 패칭으로 복잡도를 O(N²)에서 O((L/S)²)로 감소시켰습니다.[^1_7]

**6. 설명 가능성(Explainability) 향상**
LSEAttention의 어텐션 패턴이 어떻게 해석 가능한지 연구가 필요합니다. 최근 연구들은 Transformer 기반 모델의 블랙박스 문제를 지적하고 있습니다.[^1_8]

## 6. 2020년 이후 관련 최신 연구 비교 분석

### 주요 연구 동향

**패치 기반 접근 (2022-2023)**

- **PatchTST (ICLR 2023)**: 시계열을 서브시리즈 패치로 분할하여 입력 토큰으로 사용. 채널 독립성(channel-independence) 도입으로 각 채널을 개별 단변량 시계열로 처리. 복잡도를 22배 감소시켰으며 LATST의 Fractional Temporal Block과 유사한 접근.[^1_9][^1_7]
- **CT-PatchTST (2025)**: PatchTST를 확장하여 채널과 시간 정보를 동시 통합. 복잡한 다변량 시계열 데이터 처리 능력 향상.[^1_10][^1_11]

**어텐션 메커니즘 개선 (2023-2025)**

- **SAMformer (ICML 2024)**: Sharpness-Aware Minimization(SAM)과 RevIN을 통합하여 엔트로피 붕괴 완화. LATST의 직접적인 비교 대상이며, LATST가 4% 성능 개선.[^1_1]
- **FreEformer (2025)**: 주파수 도메인에서 Transformer 적용. 바닐라 어텐션의 저랭크 특성을 해결하기 위해 학습 가능한 행렬을 추가하고 L1 정규화 적용. 18개 벤치마크에서 SOTA 달성.[^1_4]
- **Fredformer (KDD 2024)**: 주파수 편향(frequency bias) 문제 해결. 모델이 고에너지 저주파 특징에 과도하게 집중하는 것을 방지하여 고주파 특징도 균등하게 학습.[^1_6]

**하이브리드 아키텍처 (2024-2025)**

- **HTMformer (2025)**: Hybrid Time and Multivariate Transformer로 시간적 특징 추출과 다변량 특징 추출을 분리. HTME 전략으로 기존 Transformer 대비 평균 35.8% 성능 향상.[^1_3]
- **iTransformer (ICLR 2024)**: 각 단변량 시계열을 개별 토큰으로 처리하는 역전된(inverted) 접근. 변수 간 의존성 모델링에 효과적이며 LATST와 비교 벤치마크로 사용됨.[^1_8][^1_1]

**대규모 사전 학습 모델 (2024-2025)**

- **Timer \& Moment (2024)**: 대규모 다양한 데이터셋으로 사전 학습된 기초 모델. Timer_XL은 전이 학습에서 작은 태스크별 모델을 크게 초과하는 성능을 보임.[^1_5]
- **TimePFN (2025)**: Prior-data Fitted Networks 개념을 시계열에 적용. 합성 데이터로 학습하여 few-shot 학습 능력 강화.[^1_12]

**특수 목적 개선 (2023-2025)**

- **PSformer (2025)**: Parameter Sharing과 Segment Attention으로 매개변수 효율성 개선. LATST의 효율성 목표와 유사.[^1_13]
- **Sentinel (2025)**: Multi-patch attention으로 시간적/채널 간 어텐션 통합. Multi-head splitting을 패칭 프로세스로 대체.[^1_14]


### LATST와의 비교

| 연구 | 핵심 기여 | LATST와의 차이점 | 성능 비교 |
| :-- | :-- | :-- | :-- |
| **PatchTST** | 패치 기반 처리, 채널 독립성 | LSEAttention 미사용, 표준 softmax | LATST가 일부 데이터셋에서 우수 |
| **SAMformer** | SAM 최적화, RevIN 정규화 | 수치적 안정성 근본 해결 미흡 | LATST가 4% 개선[^1_1] |
| **FreEformer** | 주파수 도메인, 학습 가능 행렬 | 시간 도메인에서 작동 | 18개 벤치마크 SOTA |
| **HTMformer** | 시간-다변량 분리 전략 | 단일 통합 어텐션 사용 | 35.8% Transformer 개선[^1_3] |
| **iTransformer** | 역전된 시계열 임베딩 | 패치 기반 처리 사용 | 데이터셋별로 혼재 |

### 통합적 관찰

**1. 엔트로피 붕괴는 핵심 문제**: LATST의 엔트로피 붕괴 분석은 최근 여러 연구에서 검증되었습니다. 특히 VGGT 연구는 정규화된 어텐션 엔트로피와 재구성 정확도 간 선형 상관관계(엔트로피 > 0.8 유지 필요)를 발견했습니다.[^1_2]

**2. 수치적 안정성의 재조명**: LATST는 수치적 안정성을 성능 개선의 핵심으로 제시했으며, 이는 최근 entropy-stable attention 연구에서도 강조되고 있습니다.[^1_15]

**3. 효율성 vs 성능 트레이드오프**: LATST의 단일 레이어 접근은 효율성을 강조하지만, HTMformer와 FreEformer 같은 다층 구조가 더 높은 성능을 보입니다. 향후 연구는 이 균형점을 찾아야 합니다.[^1_4][^1_3]

**4. 주파수 도메인의 잠재력**: FreEformer와 Fredformer의 성공은 시간 도메인만의 접근이 한계가 있음을 시사합니다. LATST를 주파수 도메인과 결합하면 추가 개선 가능성이 있습니다.

**5. 기초 모델로의 전환**: Timer와 Moment의 등장은 시계열 예측이 태스크별 모델에서 대규모 사전 학습 모델로 패러다임 전환하고 있음을 보여줍니다. LATST의 LSEAttention을 이러한 기초 모델에 통합하는 것이 미래 연구 방향입니다.[^1_5]
<span style="display:none">[^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37]</span>

<div align="center">⁂</div>

[^1_1]: 2410.23749v9.pdf

[^1_2]: https://arxiv.org/html/2512.21691v1

[^1_3]: https://arxiv.org/html/2510.07084v1

[^1_4]: https://arxiv.org/abs/2501.13989

[^1_5]: https://arxiv.org/html/2507.02907v1

[^1_6]: https://dl.acm.org/doi/10.1145/3637528.3671928

[^1_7]: https://github.com/yuqinie98/PatchTST

[^1_8]: https://arxiv.org/abs/2411.05793

[^1_9]: http://arxiv.org/pdf/2211.14730v2.pdf

[^1_10]: https://arxiv.org/html/2501.08620v1

[^1_11]: https://arxiv.org/pdf/2501.08620.pdf

[^1_12]: https://arxiv.org/pdf/2502.16294.pdf

[^1_13]: https://arxiv.org/html/2411.01419v1

[^1_14]: http://arxiv.org/pdf/2503.17658.pdf

[^1_15]: https://openreview.net/forum?id=Qe4go2UH1H

[^1_16]: https://arxiv.org/html/2508.16641v1

[^1_17]: https://arxiv.org/html/2509.23145v1

[^1_18]: https://arxiv.org/pdf/2502.13721.pdf

[^1_19]: https://arxiv.org/html/2410.23749v3

[^1_20]: https://dl.acm.org/doi/10.1145/3757749.3757774

[^1_21]: https://www.mdpi.com/2073-4433/16/3/292

[^1_22]: https://ieeexplore.ieee.org/document/11345094/

[^1_23]: https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13802/3067869/A-transformer-based-approach-for-multivariate-time-series-forecasting-of/10.1117/12.3067869.full

[^1_24]: https://www.mdpi.com/2413-4155/7/1/7

[^1_25]: https://www.mdpi.com/1424-8220/25/3/652

[^1_26]: https://www.mdpi.com/2227-7390/13/5/814

[^1_27]: https://link.springer.com/10.1007/s10844-025-00937-5

[^1_28]: https://arxiv.org/pdf/2202.01381.pdf

[^1_29]: https://arxiv.org/pdf/2209.03945.pdf

[^1_30]: https://arxiv.org/abs/2207.05397

[^1_31]: https://icml.cc/virtual/2025/poster/44262

[^1_32]: https://openreview.net/forum?id=kHEVCfES4Q\&noteId=mrNbq9EkQa

[^1_33]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12453695/

[^1_34]: https://www.sciencedirect.com/science/article/pii/S2665963824001040

[^1_35]: https://peerj.com/articles/cs-3001/

[^1_36]: https://aihorizonforecast.substack.com/p/influential-time-series-forecasting-8c3

[^1_37]: https://kp-scientist.tistory.com/entry/ICLR-2023-PatchTST-A-Time-Series-is-Worth-64-Words-Long-Term-Forecasting-with-Transformers

