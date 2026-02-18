# xPatch: Dual-Stream Time Series Forecasting with Exponential Seasonal-Trend Decomposition

***

## 1. 핵심 주장 및 주요 기여 요약

xPatch(eXponential Patch)는 KAIST의 Artyom Stitsyuk, Jaesik Choi가 제안한 **비트랜스포머 기반 장기 시계열 예측(LTSF) 모델**로, AAAI 2025에 발표되었습니다.[^1_1][^1_2]

**핵심 주장:** Transformer의 순열 불변(permutation-invariant) self-attention이 시간 순서 정보를 온전히 보존하지 못한다는 근본적 한계를 지적하며, CNN+MLP 이중 스트림 아키텍처와 EMA 기반 분해만으로도 트랜스포머를 능가할 수 있음을 입증합니다.[^1_1]

**4가지 주요 기여:**

- EMA(Exponential Moving Average) 기반 계절-추세 분해 모듈 신설
- MLP 선형 스트림 + CNN 비선형 스트림의 이중 흐름 네트워크 제안
- 아크탄젠트(Arctangent) 손실 함수 설계로 과적합 억제
- 시그모이드 학습률 조정 스킴 도입으로 훈련 안정성 강화

***

## 2. 해결하려는 문제

### 기존 방법의 한계

기존 Transformer 기반 LTSF 모델(Autoformer, FEDformer, PatchTST, CARD)은 두 가지 구조적 문제를 안고 있었습니다.[^1_1]

1. **Transformer의 순열 불변성 문제:** Self-attention은 순서에 독립적이어서, positional encoding을 보완해도 연속적인 시계열에서 시간 순서 정보가 손실될 수 있습니다.
2. **SMA 분해의 편향 문제:** Autoformer, FEDformer가 사용하는 Simple Moving Average(SMA) 방식은 시퀀스 양 끝에 패딩을 삽입해야 하므로 초기/최종 값 쪽으로 편향이 발생하고, 스파이크 패턴에서 트렌드 특성을 제대로 추출하지 못합니다.

***

## 3. 제안 방법 및 수식

### 3.1 EMA 기반 계절-추세 분해

SMA 대신 EMA를 사용하여 패딩 없이 지수적으로 감쇠하는 가중치로 트렌드를 추출합니다.[^1_1]

$s_0 = x_0$

$s_t = \alpha x_t + (1 - \alpha)s_{t-1}, \quad t > 0$

$X_T = \text{EMA}(X), \quad X_S = X - X_T \tag{2}$

여기서 $\alpha \in (0,1)$은 스무딩 계수이며, $X_T$는 트렌드, $X_S$는 계절성 성분입니다. 논문은 $\alpha = 0.3$이 가장 안정적임을 실험으로 확인했고, 학습 가능한 파라미터로 설정해도 성능 향상이 미미하면서 훈련 시간만 증가한다고 보고합니다.[^1_1]

EMA의 전개식을 벡터 내적으로 변환하여 O(n) → **O(1) 시간 복잡도**로 최적화합니다:[^1_1]

$\hat{w} = [(1-\alpha)^t,\ (1-\alpha)^{t-1}\alpha,\ \ldots,\ (1-\alpha)\alpha,\ \alpha]$

$s_t = \hat{w} \cdot x \tag{28}$

### 3.2 이중 스트림 네트워크

채널 독립(Channel-Independence)으로 각 변수를 독립 유니버리어트 시리즈 $x^{(i)} \in \mathbb{R}^L$로 처리한 뒤, EMA 분해를 거쳐 두 스트림에 병렬 투입합니다.[^1_1]

**선형 스트림 (MLP, 트렌드 처리):**

$x^{(i)} = \text{LayerNorm}(\text{AvgPool}(\text{Linear}(x^{(i)}),\ k=2)) \tag{3}$
$\hat{x}^{(i)}_{\text{lin}} = \text{Linear}(x^{(i)}) \tag{4}$

활성화 함수를 의도적으로 제거하여 선형 트렌드 특성만을 보존합니다.

**비선형 스트림 (CNN, 계절성 처리):**

패치 $x^{(i)}_p \in \mathbb{R}^{N \times P}$ (패치 수 $N = \lfloor \frac{L-P}{S} \rfloor + 2$, P=16, S=8) 생성 후:

$x^{N \times P^2}_p = \text{BatchNorm}(\sigma(\text{Embed}(x^{(i)}_p))) \tag{5}$

$x^{N \times P}\_p = \text{Conv}_{N \to N}(x^{N \times P^2}_p,\ k=P,\ s=P,\ g=N) \tag{6}$

$x^{N \times P}_p = \text{DepthwiseConv}(x^{N \times P^2}_p) + x^{N \times P^2}_p \tag{8}$

$x^{N \times P}\_p = \text{Conv}_{N \to N}(x^{N \times P}_p,\ k=1,\ s=1,\ g=1) \tag{9}$

$\hat{x}^{(i)}_{\text{nonlin}} = \text{Linear}(\sigma(\text{Linear}(\text{Flatten}(x^{N \times P}_p)))) \tag{11}$

**최종 출력 (두 스트림 결합):**

```math
\hat{x}^{(i)} = \text{Linear}(\text{concat}(\hat{x}^{(i)}_{\text{lin}},\ \hat{x}^{(i)}_{\text{nonlin}})) 
```

### 3.3 아크탄젠트 손실 함수

기존 CARD 모델의 신호 감쇠 손실 $\rho_{\text{CARD}}(i) = i^{-\frac{1}{2}}$는 감쇠가 너무 빠르다는 문제가 있어, 보다 완만한 아크탄젠트 스케일링 계수를 도입합니다:[^1_1]

$\mathcal{L}\_{\text{arctan}} = \frac{1}{T} \sum_{i=1}^{T} \rho_{\text{arctan}}(i) \| \hat{x}^{(i)}\_{1:T} - x^{(i)}_{1:T} \| \tag{16}$

$\rho_{\text{arctan}}(i) = -\arctan(i) + \frac{\pi}{4} + 1 \tag{17}$

예측 지평 $i = 720$에서 $\rho_{\text{CARD}}(720) \approx 0.037$인 반면 $\rho_{\text{arctan}}(720) \approx 0.428$로, 원거리 예측에 더 균형 잡힌 가중치를 부여합니다.[^1_1]

### 3.4 시그모이드 학습률 조정

$\alpha_t = \frac{\alpha_0}{1 + e^{-k(t-w)}} - \frac{\alpha_0}{1 + e^{-\frac{k}{s}(t-sw)}} \tag{23}$

$k$는 로지스틱 성장률, $s$는 감쇠 곡선 스무딩 계수, $w$는 워밍업 계수입니다. 기존 스텝 감쇠 또는 코사인 감쇠보다 더 부드러운 학습률 전환을 구현합니다.[^1_1]

***

## 4. 모델 구조 요약

| 구성 요소 | 세부 내용 |
| :-- | :-- |
| 전처리 | RevIN (Reversible Instance Normalization) + EMA 분해 |
| 선형 스트림 | MLP + AvgPool(k=2) + LayerNorm; 활성화 함수 없음 |
| 비선형 스트림 | Patching (P=16, S=8) + Depthwise Separable CNN + GELU + BatchNorm |
| 출력 통합 | 두 스트림 concat → 최종 Linear 레이어 |
| 손실 | Arctangent 스케일링 MAE |
| 학습률 | 시그모이드 기반 warm-up 스킴 |


***

## 5. 성능 향상 및 실험 결과

### 통합 설정(Unified Settings) 결과

9개 데이터셋(ETTh1/h2/m1/m2, Weather, Traffic, Electricity, Exchange, ILI, Solar)에서 MSE 기준 **60%**, MAE 기준 **70%**의 데이터셋에서 1위를 달성했습니다.[^1_1]

- vs CARD: MSE 2.46%↓, MAE 2.34%↓
- vs TimeMixer: MSE 3.34%↓, MAE 6.34%↓
- vs PatchTST: MSE 4.76%↓, MAE 6.20%↓


### 하이퍼파라미터 탐색 설정 결과

MSE 기준 **70%**, MAE 기준 **90%** 데이터셋에서 1위를 달성했습니다.[^1_1]

- vs CARD: MSE 5.29%↓, MAE 3.81%↓
- vs PatchTST: MSE 7.87%↓, MAE 8.59%↓


### 계산 효율성

xPatch의 평균 학습 시간은 **3.099 ms/step**으로, CARD(14.877 ms), TimeMixer(13.174 ms), PatchTST(6.618 ms) 대비 월등히 빠릅니다.[^1_1]

***

## 6. 일반화 성능 향상 가능성

xPatch의 일반화 성능 향상에 기여하는 핵심 메커니즘은 다음 세 가지입니다.[^1_1]

### EMA 분해의 도메인 적응성

EMA는 $\alpha$ 값 조정만으로 다양한 데이터 패턴(정상/비정상, 단주기/장주기)에 유연하게 적응합니다. Augmented Dickey-Fuller(ADF) 검정 결과, EMA로 분해된 계절성 성분은 모든 데이터셋에서 안정적인 정상(stationary) 시리즈가 되었고($p < 0.001$), 트렌드는 비정상(non-stationary)으로 분리됩니다. 이는 두 성분 각각에 최적화된 학습이 가능하게 하여, 분포가 다른 데이터셋에서도 일관된 성능을 냅니다.[^1_1]

### 채널 독립성(Channel-Independence)

각 변수를 독립적으로 처리하는 방식은 변수 간 분포 차이로 인한 간섭을 차단합니다. 이는 차원이 매우 다른 Traffic(862개 변수), Weather(21개), ILI(7개) 데이터셋 모두에서 우수한 성능을 보임으로써 검증됩니다.[^1_1]

### RevIN + Arctangent Loss의 결합

RevIN은 데이터 분포 이동(distribution shift)에 대한 강건성을 제공하고, Arctangent 손실은 먼 미래 예측에서 과도한 가중치 감쇠를 방지합니다. 논문에서 EMA 분해를 DLinear, PatchTST, Autoformer, FEDformer에 적용했을 때도 최대 4.96% MSE 개선이 관찰되었는데, 이는 EMA 분해 자체의 **모델 무관(model-agnostic) 일반화 성능**을 시사합니다.[^1_1]

***

## 7. 한계점

논문이 명시하거나 구조적으로 내포하는 한계는 다음과 같습니다.[^1_3][^1_1]

- **단변량 처리의 한계:** 채널 독립 방식은 변수 간 상호의존성(cross-variable correlation)을 모델링하지 않아, 다중 센서 간 강한 상관이 존재하는 도메인(예: 물리 시스템, 금융 포트폴리오)에서는 iTransformer 같은 채널 의존 모델에 뒤처질 수 있습니다.
- **Traffic 데이터셋 약점:** 채널 수 862개의 Traffic 데이터셋에서는 CARD, iTransformer에 MSE 기준 뒤처지며, 고차원 다변량 데이터에서의 한계를 노출합니다.
- **고정된 $\alpha$ 의존성:** EMA의 $\alpha = 0.3$ 고정은 단순하지만, 시간에 따라 동적으로 변화하는 트렌드 특성을 가진 데이터에 최적이 아닐 수 있습니다.
- **단기 예측 벤치마크 부재:** 실험이 LTSF(예측 길이 ≥96)에만 집중되어 있어, 단기 예측에서의 일반화 성능은 검증되지 않습니다.

***

## 8. 관련 최신 연구 비교 분석 (2020년 이후)

| 모델 | 연도 | 핵심 기법 | 분해 방식 | 아키텍처 | xPatch 대비 |
| :-- | :-- | :-- | :-- | :-- | :-- |
| **Autoformer** | 2021 | Auto-Correlation | SMA (내부 블록) | Transformer | MSE 크게 뒤처짐 |
| **DLinear** | 2023 | 1-Layer MLP | SMA (전처리) | Linear | 단순하지만 성능 낮음 |
| **PatchTST** | 2023 | Patch + CI | SMA (선택적) | Transformer | xPatch에 7.87% 뒤처짐 |
| **iTransformer** | 2024 | 역전된 Attention | 없음 | Transformer | 채널 상관 강점 |
| **TimeMixer** | 2024 | 멀티스케일 믹싱 | SMA | MLP | xPatch에 7.45% 뒤처짐 |
| **CARD** | 2024 | 채널 정렬 Transformer | 지수 스무딩(토큰) | Transformer | xPatch에 5.29% 뒤처짐 |
| **xPatch** | 2025 | EMA 분해 + 이중 스트림 | **EMA** | CNN+MLP | 기준 모델 |

주목할 점은 ShifTS(2025)  같은 분포 이동 대응 프레임워크나 MoHETS(2025)  같은 혼합 전문가(MoE) 기반 방법이 xPatch 이후에 등장하여 일반화 성능을 추가로 개선하는 방향으로 연구가 진행되고 있다는 것입니다.[^1_4][^1_5]

***

## 9. 향후 연구에 미치는 영향과 고려 사항

### 연구에 미치는 영향

xPatch는 **"강력한 시계열 예측 = 트랜스포머"라는 통념에 대한 체계적인 반증**을 제공합니다. 특히 EMA 분해 모듈이 다른 모델에 이식해도 성능 향상을 가져온다는 실험 결과는 향후 연구에서 분해 방식 선택의 중요성을 재부각시킵니다.  또한 패칭(patching)과 채널 독립성의 효과가 트랜스포머 아키텍처 자체가 아닌 그 기법에서 비롯됨을 보임으로써, 비트랜스포머 모델 연구의 타당성을 강화합니다.[^1_1]

### 향후 연구 시 고려할 사항

1. **동적 $\alpha$ 최적화:** 현재 고정된 $\alpha = 0.3$ 대신 메타러닝 또는 베이지안 최적화로 데이터 특성에 맞게 $\alpha$를 자동 설정하는 연구가 필요합니다.
2. **채널 의존성과의 융합:** 채널 독립 방식의 한계를 극복하기 위해 그래프 신경망(GNN)이나 Attention을 제한적으로 결합하는 하이브리드 접근이 유망합니다.
3. **분포 이동 대응:** RevIN 외에 ShifTS 처럼 개념 드리프트(concept drift)와 시간적 분포 이동을 동시에 처리하는 프레임워크와의 결합이 일반화 성능을 추가로 높일 수 있습니다.[^1_4]
4. **파운데이션 모델과의 통합:** Time-LLM, TimesFM  같은 LLM 기반 시계열 파운데이션 모델에 EMA 분해를 전처리로 결합하면, 제로샷 일반화 성능의 향상을 기대할 수 있습니다.[^1_6][^1_7]
5. **벤치마크 다각화:** TFB (2024)  같은 포괄적 벤치마크를 활용하여 다양한 도메인과 예측 길이에서의 강건성을 검증하는 것이 중요합니다. 최근 "There are no Champions in LTSF"  논문이 지적하듯, 단일 지표·데이터셋 기반 주장에는 주의가 필요합니다.[^1_3][^1_1]
<span style="display:none">[^1_10][^1_11][^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_8][^1_9]</span>

<div align="center">⁂</div>

[^1_1]: 2412.17323v3.pdf

[^1_2]: https://github.com/stitsyuk/xPatch

[^1_3]: https://arxiv.org/html/2502.14045v1

[^1_4]: https://arxiv.org/html/2510.14814v1

[^1_5]: https://arxiv.org/html/2601.21866v1

[^1_6]: https://arxiv.org/pdf/2310.10688.pdf

[^1_7]: https://arxiv.org/pdf/2310.01728.pdf

[^1_8]: https://arxiv.org/abs/2412.17323

[^1_9]: https://arxiv.org/html/2412.17323v3

[^1_10]: https://arxiv.org/pdf/2412.17323.pdf

[^1_11]: https://www.arxiv.org/list/cs.LG/2024-12?skip=700\&show=2000

[^1_12]: https://www.arxiv.org/list/cs/2024-12?skip=6850\&show=2000

[^1_13]: https://www.arxiv.org/list/cs.LG/2024-12?skip=250\&show=2000

[^1_14]: https://arxiv.org/html/2508.07697v3

[^1_15]: https://arxiv.org/html/2405.17478v3

[^1_16]: https://www.arxiv.org/pdf/2502.14045.pdf

[^1_17]: https://arxiv.org/html/2509.23145v1

[^1_18]: https://arxiv.org/pdf/2502.15016.pdf

[^1_19]: https://arxiv.org/html/2511.08229v6

[^1_20]: http://arxiv.org/pdf/2310.00655.pdf

[^1_21]: http://arxiv.org/pdf/2211.14730v2.pdf

[^1_22]: http://arxiv.org/pdf/2311.18780.pdf

[^1_23]: https://arxiv.org/abs/2405.13575

[^1_24]: http://arxiv.org/pdf/2408.15997v1.pdf

[^1_25]: https://arxiv.org/html/2412.17323v1

[^1_26]: https://www.arxiv.org/abs/2412.17323

[^1_27]: https://liner.com/ko/review/xpatch-dualstream-time-series-forecasting-with-exponential-seasonaltrend-decomposition

[^1_28]: https://www.youtube.com/watch?v=nKUVJntCtuM

[^1_29]: https://www.themoonlight.io/en/review/xpatch-dual-stream-time-series-forecasting-with-exponential-seasonal-trend-decomposition

[^1_30]: https://www.nature.com/articles/s41467-025-63786-4

[^1_31]: https://cloud.tencent.com/developer/article/2536964

[^1_32]: https://arxiv.org/html/2411.05793v1

[^1_33]: https://hyper.ai/kr/papers/2412.17323

[^1_34]: https://www.sciencedirect.com/science/article/abs/pii/S0016003225000845

[^1_35]: https://www.sciencedirect.com/science/article/abs/pii/S0950705124006208

[^1_36]: https://www.themoonlight.io/ko/review/xpatch-dual-stream-time-series-forecasting-with-exponential-seasonal-trend-decomposition

[^1_37]: http://www.arxiv.org/pdf/2501.01087v2.pdf

[^1_38]: https://liner.com/ko/review/forecasting-time-series-with-llms-via-patchbased-prompting-and-decomposition

