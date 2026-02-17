# PatchMixer: A Patch-Mixing Architecture for Long-Term Time Series Forecasting

## 1. 핵심 주장과 주요 기여

PatchMixer는 최근 패치 기반 Transformer(PatchTST)가 우수한 성능을 보이는 이유가 Transformer 아키텍처 자체보다는 **패치 기반 입력 표현(patch-based representation)**에 기인한다는 가설을 검증하기 위해 제안된 CNN 기반 모델입니다. 이 논문의 핵심 기여는 다음과 같습니다:[^1_1]

- SOTA Transformer 대비 MSE 3.9%, MLP 대비 11.6%, CNN 대비 21.2% 성능 향상을 달성했으며, 추론 속도는 3배, 학습 속도는 2배 향상되었습니다[^1_1]
- 패치 임베딩 파라미터와 손실 함수 최적화를 통해 패치 기반 방법의 일반화 성능을 개선했습니다[^1_1]
- 제한된 수용 영역을 가진 CNN으로도 패치 믹싱 아키텍처를 통해 우수한 예측 성능을 달성할 수 있음을 입증했습니다[^1_1]


## 2. 해결하고자 하는 문제와 제안 방법

### 2.1 문제 정의

장기 시계열 예측(LTSF)에서 다변량 시계열 $X^{M \times L} = (x_1^M, ..., x_L^M)$이 주어졌을 때, 향후 $T$ 시점의 예측값 $\hat{X}^{M \times T} = (x_{L+1}^M, ..., x_{L+T}^M)$을 생성하는 것입니다.[^1_1]

### 2.2 패치 믹싱 설계

논문은 Normalized Mutual Information (NMI)을 사용하여 채널 간 의존성과 패치 간 의존성을 분석했습니다:[^1_1]

$$
\text{NMI}(X; Y) = \frac{2I(X; Y)}{H(X) + H(Y)}
$$

여기서 $I(X; Y)$는 상호 정보, $H(X)$와 $H(Y)$는 각각의 엔트로피입니다. 실험 결과, 변수 내 시간 패턴이 채널 간 상관관계보다 훨씬 강한 상호 정보를 갖는 것으로 나타났습니다.[^1_1]

### 2.3 PatchMixer Block

**Depthwise Convolution**: 커널 크기 $K=8$을 사용하며, $l$번째 레이어에서의 처리는 다음과 같습니다:[^1_1]

$$
x_l^{N \times D} = \text{BN}\left(\sigma(\text{Conv}_{N \to N}(x_{l-1}^{N \times D}), \text{kernel} = \text{step} = K)\right)
$$

**Pointwise Convolution**: 패치 간 특징 상관관계를 포착하며 잔차 연결로 강화됩니다:[^1_1]

$$
x_{l+1}^{N \times D} = \text{BN}\left(\sigma(\text{Conv}_{N \to N}(x_l^{N \times D}), \text{kernel} = \text{step} = 1)\right)
$$

여기서 $\sigma$는 GELU 활성화 함수, BN은 BatchNorm 연산, $N$은 패치 수, $D$는 임베딩 차원입니다.[^1_1]

### 2.4 모델 구조

PatchMixer는 다음 구성 요소로 이루어져 있습니다:[^1_1]

1. **Instance Normalization**: 패치 생성 전 적용되며, 예측 후 원래의 평균과 표준편차를 재적용합니다[^1_1]
2. **Patch Embedding**: 슬라이딩 윈도우(크기 $P$, 스트라이드 $S$)를 사용하여 $N = \lfloor(L-P)/S\rfloor + 2$개의 패치를 생성합니다[^1_1]
3. **PatchMixer Block**: Depthwise Separable Convolution 기반[^1_1]
4. **Dual Forecasting Heads**: 선형 트렌드와 비선형 동역학을 동시에 포착합니다[^1_1]

## 3. 성능 향상 및 한계

### 3.1 성능 향상

7개 벤치마크 데이터셋(Weather, Traffic, Electricity, ETTh1, ETTh2, ETTm1, ETTm2)에서 평가한 결과:[^1_1]


| 데이터셋 | 변수 수 | 시간 단계 | PatchMixer 우위 |
| :-- | :-- | :-- | :-- |
| Traffic | 862 | 17,544 | 모든 예측 길이에서 최고 |
| Electricity | 321 | 26,304 | 모든 예측 길이에서 최고 |
| Weather | 21 | 52,696 | 모든 예측 길이에서 최고 |

**계산 복잡도 비교**:[^1_1]

- Attention 메커니즘: $\mathcal{O}(N \cdot D^2 + N^2 \cdot D)$ (293.63M MACs)
- Standard Convolution: $\mathcal{O}(N^2 \cdot D \cdot K)$ (175.57M MACs)
- PatchMixer Block: $\mathcal{O}(N^2 \cdot D + N \cdot D \cdot K)$ (66.32M MACs)


### 3.2 한계

논문에서 명시적으로 언급된 한계는 제한적이지만, 다음을 추론할 수 있습니다:

- CNN의 제한된 수용 영역으로 인한 장거리 의존성 포착의 잠재적 제약
- 패치 길이와 스트라이드 선택이 데이터셋에 따라 다르게 최적화되어야 함[^1_1]


## 4. 모델의 일반화 성능 향상

### 4.1 패치 임베딩 최적화

**패치 길이(Patch Length) 실험**: 패치 길이를 $P = [1, 2, 4, 8, 12, 16, 24, 32, 40]$으로 변경하며 실험한 결과, 패치 기반 방법은 일반적으로 패치가 클수록 손실이 감소하거나 약간의 변동만 보였습니다. PatchMixer는 더 큰 패치 크기에서 일관되게 더 큰 성능 향상을 달성했습니다.[^1_1]

**패치 스트라이드(Patch Stride)**: 스트라이드 증가에 따라 MSE는 ±0.002 범위 내에서 불규칙하게 진동했으며, 최저 손실 함수 지점이 데이터셋마다 다르게 나타났습니다. 이는 패치 스트라이드 파라미터가 최적화에 크게 기여하지 않음을 시사합니다.[^1_1]

### 4.2 손실 함수 최적화

다양한 손실 함수(MSE, MAE, SmoothL1Loss, MSE+MAE)를 비교한 결과, **MSE와 MAE를 1:1 비율로 결합한 손실 함수**가 가장 우수한 성능을 보였습니다. 이는 다음과 같은 이점을 제공합니다:[^1_1]

- MSE는 큰 오차에 민감하여 이상치에 강건하지 않지만, MAE는 모든 오차를 동등하게 처리합니다
- 두 손실 함수의 결합은 균형 잡힌 최적화를 가능하게 하여 일반화 성능을 향상시킵니다[^1_1]


### 4.3 Look-back Window 변화에 대한 강건성

입력 길이를 $L = [24, 48, 96, 192, 336, 720]$으로 변경하며 실험한 결과, 전통적인 CNN 기반 방법은 입력 길이 증가에 따라 성능이 불규칙하게 변동한 반면, PatchMixer는 손실이 꾸준히 감소하여 더 긴 look-back 윈도우를 효과적으로 활용했습니다. 이는 **일반화 성능과 확장성**이 우수함을 입증합니다.[^1_1]

## 5. 앞으로의 연구에 미치는 영향

### 5.1 패러다임 전환

PatchMixer는 시계열 예측에서 **아키텍처의 복잡성보다 데이터 전처리 기법(패칭)이 더 중요**할 수 있음을 시사합니다. 이는 다음과 같은 연구 방향을 제시합니다:[^1_1]

- 효율적인 입력 표현 방법에 대한 연구 강화
- 계산 효율성과 성능의 균형을 추구하는 경량 모델 개발
- 패치 기반 접근법의 다른 시계열 분석 태스크로의 확장


### 5.2 향후 연구 시 고려사항

**1. 적응형 패칭 전략**: 데이터셋마다 최적의 패치 길이가 다르므로, 데이터 특성에 따라 자동으로 패치 파라미터를 조정하는 메커니즘 개발이 필요합니다.[^1_1]

**2. 장거리 의존성 모델링**: CNN의 제한된 수용 영역 문제를 해결하기 위해 dilated convolution이나 hybrid 아키텍처 고려가 필요합니다.

**3. 멀티스케일 분석**: 다양한 시간 스케일의 패턴을 동시에 포착할 수 있는 멀티스케일 패치 임베딩 연구가 필요합니다.

**4. 사전 학습 및 전이 학습**: PatchTST가 보여준 자기 지도 학습 방식을 CNN 기반 패치 모델에도 적용하여 일반화 성능을 더욱 향상시킬 수 있습니다.[^1_2][^1_3]

## 6. 2020년 이후 관련 최신 연구 비교 분석

### 6.1 Transformer 기반 모델

**Informer (2021)**: ProbSparse attention 메커니즘을 도입하여 계산 복잡도를 $\mathcal{O}(L \log L)$로 감소시켰습니다. 그러나 장기 예측에서 선형 모델에 비해 성능이 떨어지는 경우가 많았습니다.[^1_4][^1_5][^1_6]

**Autoformer (2021)**: 시계열 분해를 모델의 내부 블록으로 통합하고 Auto-Correlation 메커니즘을 제안하여 6개 벤치마크에서 38% 성능 향상을 달성했습니다. 그러나 여전히 $\mathcal{O}(L \log L)$ 복잡도를 가집니다.[^1_5][^1_7]

**FEDformer (2022)**: 주파수 도메인에서 작동하여 계절-추세 분해를 수행하며, 다변량 시계열에서 14.8%, 단변량에서 22.6%의 오차 감소를 달성했습니다. Informer보다 정확하지만 추론 시간이 더 깁니다.[^1_8][^1_9]

**PatchTST (2023)**: 채널 독립성과 패칭을 결합하여 Transformer의 새로운 가능성을 제시했습니다. 패칭을 통해 계산 및 메모리 사용량을 이차적으로 감소시켰으며, 자기 지도 학습도 지원합니다.[^1_3][^1_10][^1_2]

### 6.2 MLP 기반 모델

**DLinear (2023)**: 단순한 선형 레이어를 사용하여 많은 Transformer 기반 모델을 능가했으며, Transformer가 시계열 예측에 효과적인지에 대한 의문을 제기했습니다. 분해 레이어(DecompositionLayer)를 사용하여 추세와 계절성을 분리합니다.[^1_11][^1_4]

### 6.3 CNN 기반 모델

**TimesNet (2023)**: 1D 시계열을 2D 텐서로 변환하여 기간 내(intraperiod) 및 기간 간(interperiod) 변동을 동시에 모델링합니다. Inception 스타일의 병렬 합성곱 구조를 사용하여 5개 주요 시계열 분석 태스크에서 SOTA를 달성했습니다.[^1_12][^1_13][^1_14]

**MICN (2023)**: 멀티스케일 하이브리드 분해와 등거리 합성곱을 도입했습니다. 데이터 누락이 있는 풍력 예측에서 가장 우수한 성능을 보였으며, TimesNet은 가장 높은 강건성을 나타냈습니다.[^1_15][^1_1]

### 6.4 비교 종합

| 모델 | 년도 | 핵심 기술 | 복잡도 | 장점 | 단점 |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Informer | 2021 | ProbSparse Attention | $\mathcal{O}(L \log L)$ | 효율적 긴 시퀀스 처리 | 단순 모델 대비 성능 미흡[^1_6] |
| Autoformer | 2021 | Auto-Correlation, 분해 | $\mathcal{O}(L \log L)$ | 38% 성능 향상[^1_7] | 여전히 높은 계산 비용 |
| FEDformer | 2022 | 주파수 도메인 분해 | Linear | 높은 정확도[^1_8] | 긴 추론 시간[^1_9] |
| DLinear | 2023 | 선형 레이어 + 분해 | Linear | 단순하고 효과적[^1_4] | 복잡한 패턴 포착 한계 |
| TimesNet | 2023 | 2D 변환, Inception | - | 5개 태스크 SOTA[^1_12] | 높은 복잡도 |
| PatchTST | 2023 | 채널 독립 + 패칭 | $\mathcal{O}((L/S)^2)$ | 자기 지도 학습, 높은 정확도[^1_2] | 계산 비용 여전히 높음 |
| **PatchMixer** | 2024 | 패칭 + DWConv | **$\mathcal{O}(N^2 D + NDK)$** | **최고 효율성 + 정확도**[^1_1] | 제한된 수용 영역 |

### 6.5 PatchMixer의 차별점

1. **효율성**: PatchTST 대비 3배 빠른 추론, 2배 빠른 학습[^1_1]
2. **단순성**: 1개의 PatchMixer Block만으로 PatchTST의 3개 Transformer 인코더보다 우수한 성능[^1_1]
3. **검증**: 패치 기반 방법의 효과가 Transformer 아키텍처가 아닌 패칭 자체에서 비롯됨을 입증[^1_1]
4. **일반화**: 패치 파라미터와 손실 함수 최적화를 통한 일반화 성능 향상[^1_1]

## 7. 결론

PatchMixer는 단순한 CNN 아키텍처로도 패치 기반 표현과 효율적인 convolution을 통해 최첨단 성능을 달성할 수 있음을 증명했습니다. 이는 시계열 예측 연구가 복잡한 아키텍처 설계보다 **데이터 전처리와 표현 학습**에 더 집중해야 함을 시사합니다. 향후 연구는 적응형 패칭 전략, 멀티스케일 분석, 그리고 효율적인 사전 학습 방법론 개발에 초점을 맞춰야 할 것입니다.[^1_1]
<span style="display:none">[^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71]</span>

<div align="center">⁂</div>

[^1_1]: 2310.00655v2.pdf

[^1_2]: https://arxiv.org/abs/2211.14730

[^1_3]: http://arxiv.org/pdf/2211.14730v2.pdf

[^1_4]: https://seunghan96.github.io/ts/(paper)DLinear/

[^1_5]: https://arxiv.org/abs/2106.13008

[^1_6]: https://ar5iv.labs.arxiv.org/html/2211.14730

[^1_7]: https://www.semanticscholar.org/paper/fc46ccb83dc121c33de7ab6bdedab7d970780b2f

[^1_8]: https://arxiv.org/abs/2201.12740

[^1_9]: https://arxiv.org/html/2307.00493v1

[^1_10]: https://www.datasciencewithmarco.com/blog/patchtst-a-breakthrough-in-time-series-forecasting

[^1_11]: https://huggingface.co/blog/autoformer

[^1_12]: https://ar5iv.labs.arxiv.org/html/2210.02186

[^1_13]: https://ise.thss.tsinghua.edu.cn/~mlong/doc/TimesNet-iclr23.pdf

[^1_14]: https://github.com/thuml/TimesNet

[^1_15]: https://ieeexplore.ieee.org/document/10489528/

[^1_16]: https://arxiv.org/pdf/2211.14730.pdf

[^1_17]: https://arxiv.org/html/2501.08620v1

[^1_18]: https://arxiv.org/html/2502.09683v1

[^1_19]: https://arxiv.org/html/2503.00877v1

[^1_20]: https://arxiv.org/abs/2501.01087

[^1_21]: https://www.semanticscholar.org/paper/A-Time-Series-is-Worth-64-Words:-Long-term-with-Nie-Nguyen/dad15404d372a23b4b3bf9a63b3124693df3c85e

[^1_22]: https://arxiv.org/abs/2210.02186

[^1_23]: https://arxiv.org/html/2408.04245v1

[^1_24]: https://arxiv.org/abs/2309.15946

[^1_25]: https://ieeexplore.ieee.org/document/10361388/

[^1_26]: https://arxiv.org/abs/2312.06786

[^1_27]: https://ieeexplore.ieee.org/document/10207020/

[^1_28]: http://journal.ummat.ac.id/index.php/jtam/article/view/16783

[^1_29]: https://arxiv.org/abs/2305.04800

[^1_30]: https://jtiik.ub.ac.id/index.php/jtiik/article/view/7505

[^1_31]: https://ieeexplore.ieee.org/document/10394693/

[^1_32]: https://www.semanticscholar.org/paper/4e91dea1249d6a108fab802538f088d6711a85f7

[^1_33]: http://arxiv.org/pdf/2310.00655.pdf

[^1_34]: https://arxiv.org/pdf/2310.10688.pdf

[^1_35]: https://arxiv.org/abs/2405.13575

[^1_36]: https://www.mdpi.com/2673-4591/39/1/101/pdf?version=1695101394

[^1_37]: https://arxiv.org/html/2410.09836v1

[^1_38]: http://arxiv.org/pdf/2408.15997v1.pdf

[^1_39]: http://arxiv.org/pdf/2311.18780.pdf

[^1_40]: https://github.com/yuqinie98/PatchTST

[^1_41]: https://github.com/PatchTST/PatchTST

[^1_42]: https://huggingface.co/blog/patchtst

[^1_43]: https://arxiv.org/html/2506.16001v2

[^1_44]: https://arxiv.org/html/2502.13721v1

[^1_45]: https://arxiv.org/html/2506.16001v1

[^1_46]: https://arxiv.org/pdf/2201.12740.pdf

[^1_47]: https://arxiv.org/html/2507.13043v1

[^1_48]: https://www.semanticscholar.org/paper/d7d723909b11d71252c995f5fff1a889e974aba3

[^1_49]: https://ieeexplore.ieee.org/document/9484769/

[^1_50]: https://link.springer.com/10.1007/s11524-021-00566-7

[^1_51]: https://zenodo.org/record/5512064

[^1_52]: https://linkinghub.elsevier.com/retrieve/pii/S0048969721077123

[^1_53]: https://onlinelibrary.wiley.com/doi/10.1111/itor.13222

[^1_54]: http://jecei.sru.ac.ir/article_1477.html

[^1_55]: https://dl.acm.org/doi/10.1145/3531326

[^1_56]: https://jech.bmj.com/lookup/doi/10.1136/jech-2021-216732

[^1_57]: https://arxiv.org/pdf/2502.13721.pdf

[^1_58]: https://arxiv.org/pdf/2106.13008.pdf

[^1_59]: https://arxiv.org/pdf/2206.04038.pdf

[^1_60]: https://arxiv.org/pdf/2312.11714.pdf

[^1_61]: https://arxiv.org/pdf/2310.20218.pdf

[^1_62]: https://arxiv.org/html/2410.08421v1

[^1_63]: https://arxiv.org/pdf/2102.12347.pdf

[^1_64]: https://arxiv.org/pdf/2502.03383.pdf

[^1_65]: https://github.com/thuml/Autoformer

[^1_66]: https://velog.io/@barley_15/%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-Autoformer-Decomposition-Transformers-withAuto-Correlation-for-Long-Term-Series-Forecasting

[^1_67]: https://openreview.net/forum?id=J4gRj6d5Qm

[^1_68]: https://www.mql5.com/en/articles/14858

[^1_69]: https://huggingface.co/blog/informer

[^1_70]: https://ise.thss.tsinghua.edu.cn/~mlong/doc/Autoformer-nips21.pdf

[^1_71]: https://www.sciencedirect.com/science/article/abs/pii/S0957417425035596

