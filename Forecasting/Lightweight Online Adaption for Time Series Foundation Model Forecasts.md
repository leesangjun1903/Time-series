# Lightweight Online Adaption for Time Series Foundation Model Forecasts

## 핵심 주장과 주요 기여

이 논문은 시계열 Foundation Model(FM)이 배포 단계에서 온라인 피드백을 효율적으로 활용하여 예측 성능을 향상시킬 수 있는 **ELF(Ensembling with online Linear Forecaster)** 방법을 제안합니다. ELF는 FM의 파라미터를 변경하지 않고 예측 결과만을 적응시키는 경량 메커니즘으로, 단일 CPU에서도 실행 가능할 정도로 계산 효율적입니다.[^1_1]

주요 기여는 세 가지로 요약됩니다:[^1_1]

- 배포 단계에서 사용 가능한 온라인 피드백을 FM 성능 향상에 활용하는 효율적인 방법론 제시
- Fourier 도메인에서 작동하는 경량 선형 예측기(ELF-Forecaster)를 Woodbury matrix identity를 통해 온라인으로 효율적으로 학습
- 빠른 적응 컴포넌트와 느린 적응 컴포넌트를 결합한 동적 가중치 메커니즘(ELF-Weighter) 설계


## 해결하고자 하는 문제

### 문제 정의

시계열 Foundation Model은 강력한 zero-shot 예측 성능을 제공하지만, 배포 단계에서 고정된 상태로 유지되어 데이터 분포 변화에 적응하지 못합니다. 실제 배포 환경에서는 새로운 데이터가 지속적으로 도착하여 예측 정확도와 시계열 동작 변화에 대한 피드백을 제공하지만, FM을 온라인으로 학습하는 것은 계산 비용이 너무 높아 실용적이지 않습니다.[^1_1]

### 제안하는 방법: ELF 아키텍처

ELF는 두 가지 핵심 컴포넌트로 구성됩니다:[^1_1]

#### 1. ELF-Forecaster (경량 선형 예측기)

컨텍스트 벡터 $x \in \mathbb{R}^L$에 대해 다음과 같이 예측을 생성합니다:[^1_1]

**이산 푸리에 변환(DFT)**:

```math
\text{DFT}(x)_k := \frac{1}{\sqrt{L}} \sum_{n=0}^{L-1} x_n e^{-2i\pi kn/L}
```

**고주파 성분 제거**: 주파수 보존 비율 $\alpha \in [0,1]$에 대해, $k$가 $\lfloor \alpha/2 \rfloor < k/L < \lfloor 1-\alpha/2 \rfloor$를 만족하는 DFT 성분을 제거하여 차원을 $L$에서 $\alpha L$로 축소합니다.[^1_1]

**선형 변환**: 복소 가중치 행렬 $W \in \mathbb{C}^{\alpha L \times \alpha H/2}$를 적용합니다.[^1_1]

**Ordinary Least Squares (OLS) 해**: 업데이트 스텝 $\tau$에서 가중치 행렬은 다음과 같이 계산됩니다:[^1_1]

```math
W := (\tilde{X}^*\tilde{X} + \lambda I)^{-1} \tilde{X}^*\tilde{Y}
```

여기서 $\tilde{X}, \tilde{Y}$는 각각 컨텍스트와 타겟의 푸리에 도메인 표현이며, $\lambda$는 정규화 계수입니다.

**Woodbury Matrix Identity 업데이트**: 새로운 $M$개의 데이터 포인트가 도착할 때, 다음과 같이 효율적으로 역행렬을 업데이트합니다:[^1_1]

$$
A^{-1} \rightarrow A^{-1} - B
$$

여기서 $A^{-1} := (\tilde{X}^*\tilde{X} + \lambda I)^{-1}$이고,

$$
B := A^{-1}\tilde{X}_\tau^*(I + \tilde{X}_\tau A^{-1}\tilde{X}_\tau^*)^{-1}\tilde{X}_\tau A^{-1}
$$

이 방법은 $L \times L$ 행렬의 역행렬을 계산하는 대신 $M \times M$ 행렬의 역행렬만 계산하면 되므로 ($M < L$일 때) 훨씬 효율적입니다.[^1_1]

#### 2. ELF-Weighter (동적 가중치 메커니즘)

FM과 ELF-Forecaster의 예측을 결합합니다:[^1_1]

$$
\hat{y}_{t,\text{ELF}} = w_\tau \hat{y}_{t,FM} + (1-w_\tau)\hat{y}_{t,EF}
$$

가중치 $w_\tau$는 세 가지 exponential weighter로 구성됩니다:[^1_1]

**Slow Weighter** (전체 이력 기반):

$$
w_\tau^{\text{slow}} = \frac{w_{\tau-1}^{\text{slow}} e^{-\eta \text{Loss}_{\tau,1}}}{w_{\tau-1}^{\text{slow}} e^{-\eta \text{Loss}_{\tau,1}} + (1-w_{\tau-1}^{\text{slow}})e^{-\eta \text{Loss}_{\tau,2}}}
$$

**Fast Weighter** (최근 $B$개 업데이트만 사용):

$$
w_\tau^{\text{fast}} = \frac{e^{-\eta \sum_{\tau'=\tau-B}^\tau \text{Loss}_{\tau',1}}}{\sum_{k'=1}^2 e^{-\eta \sum_{\tau'=\tau-B}^\tau \text{Loss}_{\tau',k'}}}
$$

**Merge Weighter** (fast와 slow 결합):

$$
\beta_\tau^{\text{merge}} = \frac{\beta_{\tau-1}^{\text{merge}} e^{-\eta \text{Loss}_{\tau,f}}}{\beta_{\tau-1}^{\text{merge}} e^{-\eta \text{Loss}_{\tau,f}} + (1-\beta_{\tau-1}^{\text{merge}})e^{-\eta \text{Loss}_{\tau,s}}}
$$

최종 가중치:

$$
w_\tau = \beta_\tau^{\text{merge}} w_\tau^{\text{fast}} + (1-\beta_\tau^{\text{merge}})w_\tau^{\text{slow}}
$$

이 구조는 Complementary Learning System 이론에서 영감을 받아 빠른 적응과 장기 기억을 동시에 달성합니다.[^1_1]

## 성능 향상

실험 결과는 모든 테스트 데이터셋과 FM에서 일관된 성능 향상을 보여줍니다:[^1_1]

- TTM, TimesFM, VisionTS, Chronos, Moirai 등 5개의 최신 FM에 적용
- ETTh1, ETTh2, ETTm1, ETTm2, Weather, Traffic, ECL, Solar, US Weather 등 9개 표준 데이터셋에서 평가
- MASE(Mean Absolute Scaled Error) 기준으로 모든 경우에서 성능 향상
- VisionTS의 경우 평균 10% 이상의 성능 향상
- 계산 시간: CPU 환경에서 업데이트당 0.38초로, TAFAS 대비 10배, OneNet 대비 81배, continual finetuning 대비 2506배 빠름[^1_1]


## 한계점

논문에서 명시적으로 언급된 한계점은 다음과 같습니다:

1. **채널 독립성**: ELF는 각 시계열 채널을 독립적으로 예측하므로, 채널 간 상호작용을 활용하지 못합니다[^1_1]
2. **초기화 기간**: ELF-Forecaster가 처음 학습되기 위해서는 최소 $L+H$ 타임스텝의 데이터가 필요하며, 이 기간 동안은 naive seasonal forecaster를 사용합니다[^1_1]
3. **하이퍼파라미터 고정**: 모든 실험에서 동일한 하이퍼파라미터($\lambda=20, \alpha=0.9, \eta=0.5, B=5, M=200$)를 사용하여, 데이터셋별 최적화가 이루어지지 않았습니다[^1_1]

## 모델의 일반화 성능 향상 가능성

### ELF가 일반화 성능을 향상시키는 메커니즘

ELF는 두 가지 방식으로 일반화 성능을 향상시킵니다:[^1_1]

**1. 분포 변화 감지 및 적응**: ELF-Weighter는 데이터 분포 변화를 빠르게 감지하여 FM과 ELF-Forecaster 간 가중치를 동적으로 조정합니다. 실험에서 특정 채널의 FM 가중치가 분포 변화 구간(시간 8,000-10,000)에서 급격히 변화하는 것을 확인했습니다.[^1_1]

**2. 데이터셋 특화 학습**: ELF-Forecaster는 배포된 특정 시계열 데이터에 대해 지속적으로 학습하므로, 시간이 지남에 따라 해당 데이터의 고유한 패턴을 더 잘 포착합니다. Figure 3에서 대부분의 채널에서 시간이 지남에 따라 FM 가중치가 감소하는 것을 확인할 수 있는데, 이는 ELF-Forecaster가 더 많은 데이터를 관찰할수록 성능이 향상됨을 의미합니다.[^1_1]

**3. 이론적 보장**: Exponential weighting 방법은 다음과 같은 이론적 보장을 제공합니다(Theorem 4.1):[^1_1]

학습률 $\eta = \frac{1}{L_{\max}} \sqrt{\frac{8\ln K}{T}}$일 때, regret 상한은 다음과 같습니다:

$$
R_T \leq L_{\max} \sqrt{\frac{T}{2}\ln K}
$$

이는 ELF가 최선의 단일 예측기와의 성능 차이를 $\sqrt{T}$ 수준으로 제한할 수 있음을 보장합니다.[^1_1]

## 앞으로의 연구에 미치는 영향과 고려사항

### 연구에 미치는 영향

**1. 패러다임 전환**: ELF는 FM의 파라미터를 고정한 채 예측만 적응시키는 새로운 패러다임을 제시합니다. 이는 catastrophic forgetting과 plasticity loss 문제를 우회하면서도 효과적인 온라인 적응을 달성합니다.[^1_1]

**2. 실용성 강조**: CPU 환경에서 실시간 처리가 가능한 수준의 계산 효율성을 달성하여, 실제 배포 환경에서의 적용 가능성을 크게 높였습니다.[^1_1]

**3. FM 불가지론적 접근**: ELF는 어떤 FM과도 결합 가능한 일반적인 프레임워크로, FM 생태계의 확장성을 높입니다.[^1_1]

### 향후 연구 고려사항

**1. 채널 간 상호작용 모델링**: 현재 채널 독립적인 접근을 넘어, 채널 간 의존성을 효율적으로 활용하는 방법 연구가 필요합니다.[^1_1]

**2. 적응형 하이퍼파라미터**: 데이터셋과 시간에 따라 하이퍼파라미터($M, B, \eta, \alpha$)를 동적으로 조정하는 메커니즘 개발이 필요합니다.[^1_1]

**3. FM과의 공동 학습**: 부록 C.5에서 TTM을 finetuning하면서 ELF를 사용한 실험 결과, 일부 경우에는 FM을 고정하는 것이 더 나은 성능을 보였지만, 계산 비용과 continual learning 문제가 덜 중요한 경우에는 FM도 함께 학습하는 것이 유익할 수 있음을 발견했습니다.[^1_1]

**4. 장기 의존성 처리**: 현재 rolling window 접근 방식의 한계를 넘어, 더 긴 시간 범위의 패턴을 포착하는 방법 연구가 필요합니다.

**5. 다변량 시계열 확장**: 단변량 시계열에 최적화된 현재 접근을 진정한 다변량 설정으로 확장하는 연구가 필요합니다.

## 2020년 이후 관련 최신 연구 비교 분석

### Time Series Foundation Models (2023-2025)

**Chronos (2024)**: T5 아키텍처 기반 LLM형 transformer를 사용하며 수십억 개의 학습 데이터로 사전 학습되었습니다. ELF를 적용했을 때 평균 5-10% 성능 향상을 보였습니다.[^1_2][^1_1]

**TimesFM (2024)**: Google에서 개발한 decoder-only foundation model로, 패치 기반 접근을 사용하며 약 2억 개의 파라미터를 가집니다. 1000억 개의 시계열 데이터 포인트로 학습되었으며, ELF와 결합 시 일관된 성능 향상을 보였습니다.[^1_3][^1_4][^1_1]

**Moirai (2024)**: 다양한 크기의 모델을 제공하며, ELF 실험에서는 small 버전이 사용되었습니다. ELF 적용 시 평균 10-15% 성능 향상을 달성했습니다.[^1_5][^1_1]

**TTM (Tiny Time Mixers, 2024)**: TSMixer 아키텍처를 백본으로 사용하여 LLM 기반 모델보다 훨씬 작은 크기를 가집니다. ELF와 결합했을 때 가장 안정적인 성능 향상을 보였으며, 특히 계산 효율성이 뛰어났습니다.[^1_4][^1_1]

**VisionTS (2024)**: ImageNet 데이터셋으로 사전 학습된 masked autoencoder를 백본으로 사용하는 독특한 접근입니다. ELF를 적용했을 때 평균 10% 이상의 큰 성능 향상을 보여 크로스 도메인 학습의 가능성을 시사합니다.[^1_1]

**Sundial (2025)**: Flow-matching 기반 TimeFlow Loss를 제안하여 discrete tokenization 없이 Transformer를 사전 학습합니다. 1조 개의 시계열 데이터 포인트로 학습된 TimeBench를 활용하며, 밀리초 단위의 zero-shot 예측 속도를 달성했습니다.[^1_6][^1_3]

**Time-MoE (2025)**: 최초로 24억 개 파라미터로 확장된 Mixture-of-Experts 기반 시계열 foundation model입니다. Scaling laws를 검증하고 동일한 활성화 파라미터를 가진 dense 모델을 큰 차이로 능가합니다.[^1_7]

### Online Learning 및 Continual Learning 방법 (2020-2025)

**TAFAS (2025)**: FM의 예측을 적응시키는 방법으로 ELF와 유사하지만, 컨텍스트도 조정하고 gating 메커니즘을 사용합니다. ELF 대비 10배 느리며 평균 성능도 낮았습니다.[^1_1]

**OneNet (2023)**: 온라인으로 학습되는 두 예측기의 앙상블로, 하나는 채널 간 정보를 활용하고 다른 하나는 시간 정보를 활용합니다. ELF 대비 113배 느리고 성능도 낮았습니다.[^1_8][^1_1]

**FSNet (2023)**: 온라인 학습을 위해 개선된 temporal convolutional network입니다. ELF 대비 62배 느리고 대부분의 데이터셋에서 성능이 낮았습니다.[^1_8][^1_1]

**NatSR (Natural Score-driven Replay, 2025)**: Natural gradient descent와 score-driven models의 연결을 이론적으로 증명하고, Student's t likelihood를 사용하여 bounded update를 달성합니다. Replay buffer와 dynamic scale heuristic을 결합하여 regime drift에서 빠른 적응을 개선합니다.[^1_9][^1_10]

**TS-ACL (2024)**: Gradient-free 분석적 학습을 사용하여 catastrophic forgetting 문제를 근본적으로 완화합니다. 사전 학습된 feature encoder를 고정하고 분석적 분류기만 재귀적으로 업데이트합니다.[^1_11]

**Proactive Model Adaptation (2024)**: 시계열 예측의 내재적 피드백 지연 문제를 강조하고, concept drift가 더 긴 예측 구간에서 더 심각함을 실증적으로 보여줍니다.[^1_12]

### 평가 및 벤치마킹 연구 (2024-2025)

**FoundTS (2024)**: TSF foundation model의 포괄적이고 공정한 평가를 위한 새로운 벤치마크로, zero-shot, few-shot, full-shot 등 다양한 예측 전략을 지원합니다.[^1_13]

**Time Series Foundation Models: Benchmarking Challenges (2024)**: TSFM 평가의 여러 과제를 식별하고, 데이터 contamination, 정보 유출, 글로벌 패턴 암기 등의 위험을 지적합니다. 진정한 out-of-sample 미래 데이터에 대한 평가 등 새로운 원칙적 접근을 요구합니다.[^1_14][^1_15]

**Performance of Zero-Shot TSFMs on Cloud Data (2025)**: 많은 유명 FM들이 클라우드 데이터에서 의미 있는 zero-shot 예측을 생성하지 못하고 단순 선형 기준선에 일관되게 패배함을 보여줍니다.[^1_16]

### 주요 차별점

ELF는 다음과 같은 점에서 기존 연구들과 차별화됩니다:

1. **계산 효율성**: 기존 continual learning 방법들보다 10-2500배 빠릅니다[^1_1]
2. **FM 불가지성**: 어떤 FM과도 즉시 결합 가능합니다[^1_1]
3. **Catastrophic forgetting 회피**: FM 파라미터를 고정하고 예측만 적응시켜 forgetting 문제를 우회합니다[^1_1]
4. **이론적 보장**: Exponential weighting의 이론적 regret bound를 제공합니다[^1_1]
5. **실용성**: CPU만으로도 실시간 온라인 적응이 가능합니다[^1_1]

이러한 특성들은 ELF를 실제 배포 환경에서 즉시 적용 가능한 실용적인 솔루션으로 만들며, 향후 시계열 FM의 온라인 적응 연구에 중요한 기준점이 될 것으로 예상됩니다.
<span style="display:none">[^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38]</span>

<div align="center">⁂</div>

[^1_1]: 2502.12920v3.pdf

[^1_2]: https://arxiv.org/pdf/2403.14735.pdf

[^1_3]: https://arxiv.org/abs/2502.00816

[^1_4]: https://www.pricepedia.it/en/magazine/article/2025/11/07/the-arrival-of-foundation-models-in-time-series-forecasting/

[^1_5]: https://www.semanticscholar.org/paper/3f8f68f87b04306b45e7a56687e0d6361d2fbb65

[^1_6]: https://arxiv.org/html/2502.00816v1

[^1_7]: https://openreview.net/forum?id=e1wDDFmlVu

[^1_8]: https://www.arxiv.org/pdf/2510.04660.pdf

[^1_9]: https://arxiv.org/html/2601.12931v1

[^1_10]: https://openreview.net/forum?id=l4qsixUiir

[^1_11]: https://arxiv.org/abs/2410.15954v1

[^1_12]: https://arxiv.org/html/2412.08435v5

[^1_13]: http://arxiv.org/pdf/2410.11802.pdf

[^1_14]: https://arxiv.org/abs/2510.13654

[^1_15]: https://arxiv.org/html/2510.13654v1

[^1_16]: https://arxiv.org/abs/2502.12944

[^1_17]: https://www.arxiv.org/pdf/2602.03981.pdf

[^1_18]: https://arxiv.org/html/2602.03981v1

[^1_19]: https://arxiv.org/pdf/2406.08627.pdf

[^1_20]: https://arxiv.org/html/2601.15170v1

[^1_21]: https://arxiv.org/pdf/2506.11412.pdf

[^1_22]: https://arxiv.org/html/2511.03799v1

[^1_23]: https://arxiv.org/html/2601.10143v1

[^1_24]: https://arxiv.org/abs/2505.14766

[^1_25]: https://arxiv.org/abs/2503.07649

[^1_26]: https://ieeexplore.ieee.org/document/11180477/

[^1_27]: https://arxiv.org/abs/2502.06037

[^1_28]: https://arxiv.org/abs/2510.03519

[^1_29]: https://arxiv.org/abs/2508.02879

[^1_30]: https://arxiv.org/pdf/2402.03885.pdf

[^1_31]: http://arxiv.org/pdf/2310.20496.pdf

[^1_32]: https://arxiv.org/pdf/2502.15637.pdf

[^1_33]: https://arxiv.org/pdf/2310.08278.pdf

[^1_34]: https://arxiv.org/html/2504.04011v1

[^1_35]: https://neurips.cc/virtual/2025/workshop/109585

[^1_36]: https://icml.cc/virtual/2025/poster/43707

[^1_37]: https://www.nature.com/articles/s41467-025-63786-4

[^1_38]: https://openreview.net/pdf/cdbc017543a68938c60b30486172e644325c2c19.pdf

