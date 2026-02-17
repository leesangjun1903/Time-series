# Neural Decomposition of Time-Series Data for Effective Generalization

## 1. 핵심 주장과 주요 기여

Neural Decomposition (ND)는 시계열 데이터를 주기적 성분과 비주기적 성분으로 분해하여 효과적인 외삽(extrapolation)을 가능하게 하는 신경망 기법입니다. 이 논문의 핵심 주장은 효과적인 일반화를 위해서는 (1) 주기적 및 비주기적 성분을 결합하고, (2) 이러한 성분들과 이를 결합하는 가중치를 모두 학습할 수 있어야 한다는 것입니다.[^1_1]

### 주요 기여

- Fourier 변환이 일반화를 위한 부적절한 초기화 지점임을 실증적으로 입증하고, 시그널을 구성 요소로 적절히 분해하기 위해 신경망 가중치를 어떻게 조정해야 하는지 제시[^1_1]
- Fourier 및 Fourier-like 신경망에서 증강 함수(augmentation function)의 필요성을 입증하고, 학습 과정에서 성분들이 조정 가능해야 함을 증명[^1_1]
- 실세계 데이터셋(미국 실업률, 항공 승객 수, 오존 농도, 산소 동위원소 측정치)에서 LSTM, ESN, ARIMA, SARIMA, SVR보다 우수한 성능을 달성[^1_1]


## 2. 상세 분석

### 해결하고자 하는 문제

전통적인 Fourier 변환은 시계열 예측에 직접 적용할 수 없는 근본적인 한계가 있습니다. Fourier 변환은 미리 정해진 주파수 집합을 사용하며, 학습 데이터에 실제로 존재하는 주파수를 학습하지 못합니다. 또한 주기적 성분만 사용하므로 선형 추세나 비선형 이상치 같은 비주기적 성분을 정확히 모델링할 수 없습니다.[^1_1]

### 제안하는 방법

ND 모델은 다음과 같이 정의됩니다:[^1_1]

$$
x(t) = \sum_{k=1}^{N} \left(a_k \cdot \sin(w_k t + \phi_k)\right) + g(t)
$$

여기서:

- $a_k$: 진폭(amplitude)
- $w_k$: 주파수(frequency)
- $\phi_k$: 위상 편이(phase shift)
- $g(t)$: 비주기 성분을 표현하는 증강 함수(augmentation function)

**초기화 전략**:

- 주파수는 $w_k = 2\pi\lfloor k/2 \rfloor$로 초기화[^1_1]
- 위상 편이는 짝수 k에 대해 $\phi_k = \pi/2$ (sin을 cos로 변환), 홀수 k에 대해 $\phi_k = \pi$ (sin을 -sin으로 변환)[^1_1]
- 진폭(출력층 가중치)은 Fourier 변환 값 대신 작은 랜덤 값으로 초기화하여 local optima 회피[^1_1]

**정규화**:

L1 정규화를 출력층에 적용하여 희소성(sparsity)을 촉진하고, 불필요한 단위의 진폭을 0으로 수렴시킵니다. 정규화 항은 $10^{-2}$로 설정되었습니다.[^1_1]

### 모델 구조

- 단일 은닉층을 가진 feedforward 신경망
- 은닉층: N개의 sinusoid 활성화 함수 유닛 + $|g(t)|$개의 비주기 함수 유닛
- 출력층: 단일 선형 유닛으로 은닉층의 선형 결합 계산[^1_1]
- 학습률 $10^{-3}$의 확률적 경사하강법(SGD) 사용[^1_1]

입력 전처리:

1. 시간 축: 학습 데이터가 0(포함)과 1(불포함) 사이에 위치하도록 정규화
2. 값 축: 모든 학습 데이터가 0과 10 사이에 위치하도록 정규화[^1_1]

### 성능 향상

**실험 결과 (MAPE 기준)**:[^1_1]

- 실업률: ND 10.89% vs. LSTM 14.63%, ESN 15.73%, SARIMA 29.69%
- 항공 승객: ND 9.52% vs. ESN 12.05%, ARIMA 12.34%, LSTM 18.95%
- 산소 동위원소(불규칙 샘플링): ND 1.89% vs. SVR 8.50%


### 한계

- 일부 혼돈계(chaotic system)에서는 효과적이지 않음: Mackey-Glass 시리즈에서는 일반적인 형태를 포착했으나, Lorenz-63 모델에서는 역학을 효과적으로 모델링하지 못함[^1_1]
- 오존 농도 문제에서는 LSTM과 ESN이 ND보다 약간 더 나은 성능 (ND: 21.59% vs. LSTM: 16.52%, ESN: 16.15%)[^1_1]
- 모든 혼돈계의 미묘한 변화를 예측할 수 없음[^1_1]


## 3. 모델의 일반화 성능 향상 가능성

### 일반화 메커니즘

**희소 표현(Sparse Representation)**: L1 정규화를 통해 모델은 입력 샘플을 설명하는 데 필요한 최소한의 유닛만 사용하는 간단한 모델을 생성합니다. 이는 과적합을 방지하고 일반화 성능을 향상시킵니다.[^1_1]

**적응적 주파수 학습**: 고정된 Fourier 주파수 대신, 역전파를 통해 주파수와 위상을 조정하여 데이터의 실제 주기를 학습합니다. 이를 통해 주기가 정확히 N이 아닌 시계열도 효과적으로 모델링할 수 있습니다.[^1_1]

**이질적 기저 함수(Heterogeneous Basis Functions)**: 주기적 성분(sinusoid)과 비주기적 성분(linear, softplus, sigmoidal)을 결합하여 복잡한 시계열의 다양한 패턴을 포착합니다.[^1_1]

### 실증적 증거

Toy problem 분석에서 ND는 학습 과정 동안 먼저 주파수를 조정한 후 해당 진폭을 조정하는 것으로 나타났습니다. 대부분의 진폭은 0으로 수렴하여 희소 표현을 형성하며, 2500 에포크 이후 추가 학습에도 가중치가 변하지 않아 과적합에 대한 견고성을 보여줍니다.[^1_1]

불규칙 샘플링 시계열(산소 동위원소 측정)에서 ND는 SVR보다 77.8% 낮은 오류율을 달성하여, 연속적인 예측이 가능한 회귀 기반 외삽의 장점을 입증했습니다.[^1_1]

## 4. 미래 연구에 미치는 영향과 고려사항

### 영향

**분해 기반 접근법의 토대**: ND는 시계열을 명시적으로 주기 및 비주기 성분으로 분해하는 접근법의 선구적 연구로, 이후 많은 분해 기반 딥러닝 모델의 영감이 되었습니다.[^1_1]

**초기화 전략의 중요성**: Fourier 변환을 가중치 초기화에 직접 사용하는 것이 비효과적임을 입증하여, 신경망 기반 시계열 분석에서 초기화 전략에 대한 재고를 촉진했습니다.[^1_1]

**불규칙 샘플링 처리**: 고정된 시간 간격을 가정하는 RNN 기반 모델과 달리, ND는 불규칙 샘플링된 시계열을 직접 처리할 수 있는 능력을 보여줍니다.[^1_1]

### 고려사항

**증강 함수 설계**: ND는 10개의 linear, 10개의 softplus, 10개의 sigmoidal 유닛을 사용했으나, 실험 결과 주로 single linear unit만 활용되었습니다. 향후 연구는 문제 도메인에 따른 적절한 증강 함수 설계 방법을 탐구해야 합니다.[^1_1]

**하이퍼파라미터 민감도**: 정규화 항($10^{-2}$)이 매우 중요하며, 너무 크면 학습이 방해되고 너무 작으면 local optima에 빠집니다. 자동화된 하이퍼파라미터 조정 메커니즘이 필요합니다.[^1_1]

**혼돈계 한계**: Lorenz-63 모델과 같은 일부 혼돈계에서는 효과적이지 않아, 비선형 역학이 강한 시스템에 대한 개선이 필요합니다.[^1_1]

**계산 효율성**: 각 시계열에 대해 별도의 모델을 학습해야 하므로, 대규모 다변량 시계열이나 여러 시계열에 대한 확장성 개선이 필요합니다.

## 5. 2020년 이후 관련 최신 연구 비교 분석

### Transformer 기반 분해 모델

**Autoformer (2022)**: 시계열 분해를 전처리가 아닌 딥 모델의 기본 내부 블록으로 혁신했습니다. Auto-Correlation 메커니즘을 통해 하위 시계열 수준에서 의존성 발견 및 표현 집계를 수행합니다. ND와 달리 progressive decomposition capacity를 가지며, self-attention보다 우수한 성능을 보입니다.[^1_2]

**TFDNet (2023)**: 시간-주파수 도메인에서 장기 패턴과 시간적 주기성을 모두 포착하는 multi-scale time-frequency enhanced encoder를 제안했습니다. ND가 단순 Fourier-like 분해를 사용하는 것과 달리, TFDNet은 시간과 주파수 도메인을 통합적으로 처리합니다.[^1_3]

**LiNo (2025)**: 선형 및 비선형 패턴의 재귀적 잔차 분해(recursive residual decomposition)를 제안했습니다. ND가 주기/비주기 이분법을 사용하는 것과 달리, LiNo는 선형 모드와 다양한 비선형 패턴을 명시적으로 분리합니다.[^1_4]

### 분해 + 딥러닝 하이브리드

**EEMD-BO-LSTM (2020)**: Ensemble Empirical Mode Decomposition과 Bayesian Optimization으로 조정된 LSTM을 결합했습니다. ND가 Fourier-like 분해를 사용하는 것과 달리, EEMD는 데이터 적응적 분해를 제공하여 비정상성 처리에 더 효과적입니다.[^1_5]

**DSSRNN (2024)**: Decomposition State-Space Recurrent Neural Network은 분해 분석을 state-space 모델 및 물리 기반 방정식과 결합했습니다. ND보다 계산 비용이 낮으면서도 장단기 예측 모두에 효과적입니다.[^1_6]

**TSDNet (2025)**: Two-stage decomposition-based hybrid neural network으로, smooth trend components는 single linear layer로, complex seasonal components는 convolutional module로 예측합니다. ND가 모든 성분에 동일한 아키텍처를 사용하는 것과 달리, TSDNet은 각 분해된 성분에 적합한 네트워크 유형을 활용합니다.[^1_7]

### 외삽 및 일반화 개선

**EXIT (2022)**: Neural Controlled Differential Equations에 interpolation과 extrapolation을 통합하여 불규칙 시계열 처리를 개선했습니다. ND와 유사하게 불규칙 샘플링을 처리할 수 있지만, encoder-decoder 아키텍처를 통해 neural network-based interpolation을 제공합니다.[^1_8][^1_9]

**TI-DeepONet (2025)**: 적응형 수치 시간 단계 기법을 neural operator와 통합하여 장기 외삽에서 autoregressive DeepONet 대비 81% 오류 감소를 달성했습니다. ND가 feedforward 신경망을 사용하는 것과 달리, TI-DeepONet은 operator learning framework를 활용합니다.[^1_10]

### 사전학습 및 전이학습

**LLM4TS_FS (2024)**: 대규모 사전학습 시계열 모델(Moirai, TimeGPT)이 처음부터 학습하는 transformer보다 다변량 장기 예측에서 우수한 일반화 성능을 보임을 입증했습니다. ND는 각 시계열에 대해 개별 학습이 필요한 반면, 사전학습 모델은 전이학습을 통해 계산 효율성을 크게 향상시킵니다.[^1_11]

### 비교 요약

| 측면 | ND (2017) | 최신 연구 (2020-2026) |
| :-- | :-- | :-- |
| 분해 방식 | Fourier-like (고정 초기화) | 적응적 분해 (EEMD), 재귀적 분해 (LiNo), progressive 분해 (Autoformer) [^1_5][^1_4][^1_2] |
| 아키텍처 | Feedforward 단일 은닉층 | Transformer, CNN, State-Space, Operator learning [^1_2][^1_7][^1_6][^1_10] |
| 성분별 처리 | 동일한 출력층에서 결합 | 각 성분에 특화된 모듈 (TSDNet) [^1_7] |
| 일반화 메커니즘 | L1 정규화 + 주파수 학습 | 사전학습 + 전이학습 (LLM4TS) [^1_11] |
| 불규칙 샘플링 | 지원 (회귀 기반) | 지원 (EXIT, NCDEs) [^1_8] |
| 계산 효율성 | 개별 모델 학습 필요 | 사전학습 모델로 효율성 대폭 향상 [^1_11] |
| 장기 예측 | 제한적 | TI-DeepONet: 81% 오류 감소 [^1_10] |

### 결론

ND는 시계열 분해와 신경망을 결합한 선구적 연구로, 명시적 분해의 중요성과 학습 가능한 주파수의 필요성을 입증했습니다. 그러나 2020년 이후 연구들은 데이터 적응적 분해, 성분별 특화 모듈, transformer 아키텍처, 그리고 사전학습 기반 전이학습을 통해 ND의 한계를 극복하고 있습니다. 특히 LiNo의 재귀적 분해와 TSDNet의 하이브리드 접근은 ND의 단순 주기/비주기 이분법을 넘어선 더 정교한 분해 방식을 제시합니다. 향후 연구는 ND의 핵심 통찰(명시적 분해, 학습 가능한 성분)을 유지하면서 현대적 아키텍처와 결합하는 방향으로 발전할 것으로 예상됩니다.[^1_11][^1_5][^1_2][^1_4][^1_7][^1_1]
<span style="display:none">[^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38]</span>

<div align="center">⁂</div>

[^1_1]: 1705.09137v2.pdf

[^1_2]: https://arxiv.org/pdf/2106.13008.pdf

[^1_3]: https://arxiv.org/pdf/2308.13386.pdf

[^1_4]: https://arxiv.org/html/2410.17159v2

[^1_5]: https://asmedigitalcollection.asme.org/mechanicaldesign/article/doi/10.1115/1.4048414/1086969/Toward-a-Digital-Twin-Time-Series-Prediction-Based

[^1_6]: https://arxiv.org/pdf/2412.00994.pdf

[^1_7]: https://journals.sagepub.com/doi/10.1177/1088467X241308796

[^1_8]: https://arxiv.org/abs/2204.08771

[^1_9]: https://pure.kaist.ac.kr/en/publications/exit-extrapolation-and-interpolation-based-neural-controlled-diff

[^1_10]: https://arxiv.org/html/2505.17341v1

[^1_11]: https://arxiv.org/html/2507.02907v1

[^1_12]: https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0295803

[^1_13]: https://journals.plos.org/plosone/article/file?type=printable\&id=10.1371%2Fjournal.pone.0295803

[^1_14]: https://www.biorxiv.org/content/biorxiv/early/2025/10/12/2025.10.10.681651.full.pdf

[^1_15]: https://pdfs.semanticscholar.org/000c/efcc0a17a6252c7fe9d977d252bf712354a5.pdf

[^1_16]: https://pdfs.semanticscholar.org/ae0a/6a2b344e2e11b0d3ea50e05c51a755d4036e.pdf

[^1_17]: https://arxiv.org/pdf/2304.11213.pdf

[^1_18]: https://peerj.com/articles/cs-3001/

[^1_19]: https://www.semanticscholar.org/paper/6c1eeab447252f5897209ca50abc863ad7e83f9e

[^1_20]: https://ieeexplore.ieee.org/document/9099274/

[^1_21]: https://dx.plos.org/10.1371/journal.pone.0254841

[^1_22]: https://www.worldscientific.com/doi/abs/10.1142/S0129065720500392

[^1_23]: https://dl.acm.org/doi/10.1145/3396851.3397683

[^1_24]: https://linkinghub.elsevier.com/retrieve/pii/S1568494619307446

[^1_25]: https://ieeexplore.ieee.org/document/9313442/

[^1_26]: https://link.springer.com/10.1007/s00521-020-04948-x

[^1_27]: https://link.springer.com/10.1007/s13131-020-1569-1

[^1_28]: https://arxiv.org/pdf/2403.17814.pdf

[^1_29]: https://arxiv.org/pdf/2303.06394.pdf

[^1_30]: https://arxiv.org/html/2401.11929v3

[^1_31]: https://arxiv.org/pdf/2104.00164.pdf

[^1_32]: https://www.sciencedirect.com/science/article/abs/pii/S0045790625007165

[^1_33]: https://dl.acm.org/doi/10.1145/3677404.3677444

[^1_34]: https://www.nature.com/articles/s41598-023-42815-6

[^1_35]: https://arxiv.org/pdf/1705.09137.pdf

[^1_36]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12453695/

[^1_37]: https://arxiv.org/html/2405.10563v1

[^1_38]: https://deepai.org/publication/neural-decomposition-of-time-series-data-for-effective-generalization

