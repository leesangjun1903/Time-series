# First De-Trend then Attend: Rethinking Attention for Time-Series Forecasting

## 1. 핵심 주장과 주요 기여

TDformer(Trend Decomposition Transformer)는 시계열 예측에서 시간 도메인, 푸리에 도메인, 웨이블릿 도메인 attention 메커니즘 간의 관계를 이론적·실증적으로 분석한 연구입니다. 핵심 주장은 다음과 같습니다:[^1_1]

- **이론적 등가성**: 선형 조건에서 시간, 푸리에, 웨이블릿 도메인의 attention 모델은 동일한 표현력을 가집니다[^1_1]
- **Softmax의 역할**: 실제로는 softmax 비선형성으로 인해 각 도메인의 attention이 데이터 특성(계절성, 추세, 노이즈)에 따라 다른 성능을 보입니다[^1_1]
- **분리 모델링**: 추세(trend)와 계절성(seasonality)을 분리하여 MLP로 추세를, Fourier attention으로 계절성을 각각 모델링하는 것이 효과적입니다[^1_1]


## 2. 문제 정의 및 제안 방법

### 해결하고자 하는 문제

기존 Transformer 기반 시계열 예측 모델들은 다음의 한계를 보입니다:[^1_1]

1. **도메인 선택의 불명확성**: 어떤 도메인(시간/푸리에/웨이블릿)에서 attention을 학습해야 하는지에 대한 명확한 지침이 없음[^1_1]
2. **추세 데이터의 일반화 실패**: Attention 메커니즘이 본질적으로 컨텍스트를 보간(interpolate)하므로 추세 데이터를 외삽(extrapolate)하는데 약함[^1_1]
3. **분해의 비효율성**: 기존 Autoformer, FEDformer 등도 분해를 사용하지만, 추세 예측에 여전히 attention을 사용하여 최적이 아님[^1_1]

### 제안 방법: TDformer

#### Attention 메커니즘의 수학적 정의

**시간 도메인 Attention**:

$$
o(q, k, v) = \sigma\left(\frac{qk^T}{\sqrt{d_q}}\right)v
$$

[^1_1]

여기서 $q \in \mathbb{R}^{L \times D}$, $k \in \mathbb{R}^{L \times D}$, $v \in \mathbb{R}^{L \times D}$는 입력으로부터 선형 변환된 쿼리, 키, 값이며, $\sigma(\cdot)$는 활성화 함수입니다.[^1_1]

**푸리에 도메인 Attention**:

```math
o(q, k, v) = \mathcal{F}^{-1}\left\{\sigma\left(\frac{\mathcal{F}(q)\mathcal{F}(k)^T}{\sqrt{d_q}}\right)\mathcal{F}(v)\right\}
```

[^1_1]

여기서 $\mathcal{F}(\cdot)$와 $\mathcal{F}^{-1}(\cdot)$는 각각 푸리에 변환과 역변환입니다.[^1_1]

**웨이블릿 도메인 Attention**:

```math
o(q, k, v) = \mathcal{W}^{-1}\left\{\sigma\left(\frac{\mathcal{W}(q)\mathcal{W}(k^T)}{\sqrt{d_q}}\right)\mathcal{W}(v)\right\}
```

[^1_1]

#### 선형 등가성 증명

푸리에 행렬을 $W = (\omega^{jk}/\sqrt{L}) \in \mathbb{C}^{L \times L}$, $\omega = e^{-2\pi j/L}$로 정의하면, 다음 성질을 만족합니다:[^1_1]

$$
W^{-1} = W^H, \quad W^T = W
$$

[^1_1]

여기서 $W^H$는 Hermitian(켤레 전치)입니다. 선형 attention( $\sigma(\cdot) = \text{Id}(\cdot)$ )의 경우:

$$
o(q, k, v) = W^H[(Wq)(Wk)^T(Wv)] = qk^Tv
$$

[^1_1]

따라서 푸리에 도메인과 시간 도메인 선형 attention은 수학적으로 동등합니다. 웨이블릿의 경우도 직교성($W^TW = I$)을 이용하여 동일하게 증명됩니다.[^1_1]

#### 계절성-추세 분해

TDformer는 먼저 시계열을 추세와 계절성 성분으로 분해합니다:[^1_1]

$$
x_{\text{trend}} = \sigma(w(x)) * f(x), \quad x_{\text{seasonal}} = x - x_{\text{trend}}
$$

[^1_1]

여기서 $\sigma$는 softmax, $w(x)$는 데이터 의존적 가중치, $f(x)$는 평균 필터입니다.[^1_1]

#### 모델 구조

**추세 예측**: RevIN(Reversible Instance Normalization)과 3층 MLP 사용:[^1_1]

$$
X_{\text{trend}} = \text{RevIN}(\text{MLP}(\text{RevIN}(x_{\text{trend}})))
$$

[^1_1]

**계절성 예측**: $N$개 인코더 레이어:[^1_1]

$$
X_{en}^{l,1} = \text{Norm}(\text{FA}(X_{en}^{l-1}) + X_{en}^{l-1})
$$

[^1_1]

$$
X_{en}^{l,2} = \text{Norm}(\text{FF}(X_{en}^{l,1}) + X_{en}^{l,1})
$$

[^1_1]

여기서 FA는 Fourier Attention:

$$
o(q, k, v) = \mathcal{F}^{-1}\{\text{softmax}(\mathcal{F}\{q\}\mathcal{F}\{k\}^T)\mathcal{F}\{v\}\}
$$

[^1_1]

**최종 예측**:

$$
X_{\text{final}} = X_{\text{trend}} + X_{de}^M
$$

[^1_1]

## 3. 일반화 성능 향상 가능성

TDformer는 다음의 메커니즘을 통해 일반화 성능을 향상시킵니다:[^1_1]

### Softmax의 편극화 효과 활용

강한 계절성 데이터의 경우, 푸리에 공간에서 지배적인 주파수 모드가 존재합니다. Softmax의 지수 연산은 이러한 지배 모드를 더욱 증폭시켜 샘플 효율성을 높입니다. 실험 결과, 고정된 계절성 데이터(sin(x))에서 푸리에 attention이 5-20개 샘플만으로도 시간 도메인 attention보다 낮은 MSE(0.0-0.2 vs 0.4-0.8)를 달성했습니다.[^1_1]

### 추세 외삽 능력

Attention 메커니즘은 본질적으로 컨텍스트 보간에 기반하므로 추세 외삽에 약합니다. TDformer는 MLP를 사용하여 선형 추세를 완벽하게 예측합니다(MSE ≈ 0). 반면 푸리에 attention은 저주파 모드에 잘못 집중하여 추세 데이터에서 가장 큰 오차(MSE 8.567 ± 0.487)를 보였습니다.[^1_1]

### 노이즈 강건성

시간 도메인의 스파이크 노이즈는 푸리에 공간에서 작은 진폭의 고주파 성분으로 변환됩니다. Softmax는 이러한 노이즈 성분의 attention을 필터링하여, 푸리에 attention이 스파이크 데이터에서 시간 attention(MSE 0.303)보다 훨씬 낮은 오차(MSE 0.019)를 달성합니다.[^1_1]

### RevIN을 통한 비정상성 처리

RevIN 레이어는 비정상적 정보를 제거하고 복원하여 추세에 주로 존재하는 비정상성을 효과적으로 처리합니다. Ablation study에서 RevIN 제거 시 성능이 저하되었습니다(Traffic 데이터 720-step 예측: MSE 0.606 → 0.636).[^1_1]

## 4. 성능 향상 및 한계

### 성능 향상

TDformer는 5개 벤치마크 데이터셋에서 최첨단 성능을 달성했습니다:[^1_1]

- **Electricity**: Non-stationary TF 대비 평균 MSE 5.6% 감소, FEDformer 대비 14.0% 감소[^1_1]
- **Exchange**: 720-step 예측에서 Non-stationary TF(1.092) 대비 14.7% 개선(0.932)[^1_1]
- **Traffic**: 모든 예측 구간에서 일관된 최고 성능 (96-step: MSE 0.545 vs FEDformer 0.587)[^1_1]
- **Weather**: 720-step 예측에서 Non-stationary TF(0.414) 대비 11.1% 개선(0.368)[^1_1]

전체적으로 TDformer는 Non-stationary Transformer 대비 평균 9.14%, FEDformer 대비 14.69% MSE 감소를 달성했습니다.[^1_1]

### 한계점

논문에서 명시적으로 언급된 한계는:[^1_1]

1. **단순한 추세 모델**: 현재 MLP를 사용하지만, autoregressive 모델 등 더 복잡한 추세 모델 탐색 필요[^1_1]
2. **분해 방법의 다양성**: 다른 계절성-추세 분해 방법에 대한 탐색 부족[^1_1]
3. **변동 계절성**: 변동하는 계절성 데이터의 경우 웨이블릿 attention이 더 효과적이지만, TDformer는 푸리에 attention만 사용[^1_1]

## 5. 연구 영향 및 향후 고려사항

### 이론적 영향

1. **도메인 선택 가이드라인**: 데이터 특성(계절성/추세/노이즈)에 따른 attention 도메인 선택 이론적 기반 제공[^1_1]
2. **분해의 재해석**: 단순히 성능 향상 기법이 아닌, attention의 본질적 한계를 극복하기 위한 필수 요소임을 증명[^1_1]
3. **선형 등가성**: 다양한 도메인의 attention이 선형 조건에서 동등함을 수학적으로 증명[^1_1]

### 실무적 영향

1. **효율성**: FEDformer보다 더 간단한 구조로 더 나은 성능 달성 - 추세 예측에 attention 불필요[^1_1]
2. **해석가능성**: 추세와 계절성의 명확한 분리로 예측의 해석 가능성 향상[^1_1]
3. **샘플 효율성**: 계절성 데이터에서 푸리에 attention의 높은 샘플 효율성 입증[^1_1]

### 향후 연구 방향

1. **고급 추세 모델링**: Polynomial regression, ARIMA 등 통계적 방법과의 결합[^1_1]
2. **적응형 분해**: 데이터 특성에 따라 동적으로 분해 방법 선택[^1_1]
3. **하이브리드 attention**: 고정/변동 계절성을 동시에 처리하기 위한 푸리에-웨이블릿 하이브리드[^1_1]
4. **다변량 관계**: 현재는 channel-independent, 다변량 의존성 모델링 필요[^1_1]

## 6. 2020년 이후 최신 연구 비교

### 주요 연구 계보

**2021-2022: 초기 Transformer 변형**

- **Informer (2021)**: ProbSparse attention으로 계산 효율성 개선[^1_1]
- **Autoformer (2021)**: Auto-correlation 메커니즘, 내부 분해 블록 도입[^1_1]
- **FEDformer (2022)**: 푸리에/웨이블릿 enhanced blocks, TDformer의 직접 baseline[^1_2][^1_1]

**2023: 분해 및 도메인 전환 심화**

- **Non-stationary Transformer (2022)**: Series Stationarization과 De-stationary Attention[^1_1]
- **PatchTST (2023)**: Patching 전략으로 시계열을 서브시리즈 레벨 패치로 분할[^1_3]
- **MTST (2023)**: Multi-resolution 다중 브랜치 아키텍처[^1_4]
- **Fredformer (2024)**: 주파수 편향(frequency bias) 문제 해결 - 고에너지 주파수에 과도하게 집중하는 문제[^1_5]

**2024-2025: 선형 모델 도전 및 하이브리드 접근**

- **DLinear (2023)**: "Are Transformers Effective?" - 단순 선형 모델이 복잡한 Transformer 능가 가능 주장[^1_6][^1_7]
- **FAITH (2024)**: Frequency-domain Attention In Two Horizons - 채널 간 관계와 시간적 전역 정보 동시 포착[^1_8]
- **HTMformer (2025)**: Hybrid Time and Multivariate - 시간 및 다변량 특징 추출 분리, HTME 전략으로 Transformer 성능 35.8% 향상[^1_9]
- **FRWKV (2025)**: Frequency-domain linear attention, $\mathcal{O}(T)$ 복잡도로 장기 예측[^1_10][^1_11]
- **MAT (2024)**: Mamba와 Transformer 통합 - Mamba의 장거리, Transformer의 단거리 의존성 결합[^1_12]


### TDformer와의 차별점

| 연구 | 핵심 접근 | TDformer와의 차이 |
| :-- | :-- | :-- |
| FEDformer[^1_1] | 푸리에/웨이블릿 블록 | 추세 예측에도 attention 사용, TDformer는 MLP로 분리 |
| PatchTST[^1_3] | Patching + channel independence | 분해 없이 패치 단위 처리, TDformer는 추세/계절성 명시적 분해 |
| Fredformer[^1_5] | Frequency debiasing | 주파수 편향 완화 집중, TDformer는 도메인별 이론적 분석 |
| HTMformer[^1_9] | Hybrid temporal-multivariate | 다변량 관계 강조, TDformer는 단변량 시간적 분해 |
| FRWKV[^1_10] | Linear attention + frequency | 선형 복잡도 최적화, TDformer는 분해 기반 정확도 |
| DLinear[^1_6] | Pure linear baseline | Transformer 불필요 주장, TDformer는 계절성에 Fourier attention 유효성 증명 |

### 향후 고려사항

1. **선형 vs Transformer 논쟁**: DLinear 연구는 많은 경우 단순 선형이 효과적임을 보였으나, TDformer는 계절성 데이터에서 푸리에 attention의 필요성 입증 - 데이터 특성별 선택적 적용 필요[^1_6][^1_1]
2. **Foundation Model 방향**: TEMPO(2023), Timer(2024) 등 사전학습 기반 대규모 모델 등장 - TDformer의 분해 전략을 foundation model에 통합 가능성[^1_13][^1_14]
3. **하이브리드 메커니즘**: MAT, HTMformer 등 다중 메커니즘 결합 추세 - TDformer의 MLP+Fourier 조합을 Mamba 등 새로운 아키텍처와 결합[^1_9][^1_12]
4. **계산 효율성**: FRWKV의 $\mathcal{O}(T)$ 복잡도 달성 - TDformer도 푸리에 변환 최적화로 효율성 개선 여지[^1_10]
5. **다변량 모델링**: 현재 TDformer는 channel-independent - iTransformer, Crossformer처럼 채널 간 의존성 모델링 확장 필요[^1_15][^1_1]
6. **생성 모델 통합**: Latent Diffusion Transformer(2024) 등 확률적 예측 - TDformer의 결정론적 예측에 불확실성 정량화 추가[^1_16]

TDformer(2022)는 attention 메커니즘의 이론적 이해를 심화시키고 분해 기반 접근의 필요성을 입증하여, 이후 연구들이 도메인 특화, 효율성, 하이브리드 전략으로 발전하는 기반을 제공했습니다.[^1_6][^1_1]
<span style="display:none">[^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38]</span>

<div align="center">⁂</div>

[^1_1]: 2212.08151v1.pdf

[^1_2]: https://arxiv.org/pdf/2202.01381.pdf

[^1_3]: http://arxiv.org/pdf/2211.14730v2.pdf

[^1_4]: https://arxiv.org/abs/2311.04147

[^1_5]: https://dl.acm.org/doi/10.1145/3637528.3671928

[^1_6]: https://towardsdatascience.com/influential-time-series-forecasting-papers-of-2023-2024-part-1-1b3d2e10a5b3/

[^1_7]: https://github.com/cure-lab/LTSF-Linear

[^1_8]: https://www.arxiv.org/abs/2405.13300

[^1_9]: https://arxiv.org/html/2510.07084v1

[^1_10]: https://arxiv.org/html/2512.07539v1

[^1_11]: https://arxiv.org/abs/2512.07539

[^1_12]: https://ieeexplore.ieee.org/document/10823516/

[^1_13]: https://arxiv.org/abs/2310.04948

[^1_14]: https://arxiv.org/html/2507.02907v1

[^1_15]: https://www.semanticscholar.org/paper/fb45d31cc89207aec392dbac8908cc24db2df871

[^1_16]: https://ojs.aaai.org/index.php/AAAI/article/view/29085

[^1_17]: https://arxiv.org/html/2508.16641v1

[^1_18]: https://arxiv.org/html/2510.06680v1

[^1_19]: https://arxiv.org/html/2602.00589v1

[^1_20]: https://www.arxiv.org/pdf/2511.19497.pdf

[^1_21]: https://www.scitepress.org/DigitalLibrary/Link.aspx?doi=10.5220/0013395500003890

[^1_22]: https://www.ssrn.com/abstract=4718033

[^1_23]: https://ieeexplore.ieee.org/document/10352988/

[^1_24]: https://dl.acm.org/doi/10.1145/3637528.3671855

[^1_25]: https://arxiv.org/abs/2406.02486

[^1_26]: https://www.ijcai.org/proceedings/2024/608

[^1_27]: https://arxiv.org/html/2411.01419v1

[^1_28]: https://arxiv.org/pdf/2502.13721.pdf

[^1_29]: http://arxiv.org/pdf/2503.17658.pdf

[^1_30]: http://arxiv.org/pdf/2410.12184.pdf

[^1_31]: http://arxiv.org/pdf/2410.23992.pdf

[^1_32]: https://arxiv.org/pdf/2206.05495.pdf

[^1_33]: https://openreview.net/forum?id=kHEVCfES4Q\&noteId=mrNbq9EkQa

[^1_34]: https://www.sciencedirect.com/science/article/pii/S1574013725001595

[^1_35]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12453695/

[^1_36]: https://www.nature.com/articles/s41598-024-80018-9

[^1_37]: https://www.sciencedirect.com/science/article/abs/pii/S0020025523000968

[^1_38]: https://arxiv.org/html/2405.13300v1

