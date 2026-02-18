# A Temporal Kolmogorov-Arnold Transformer for Time Series Forecasting

## 1. 핵심 주장 및 주요 기여

TKAT(Temporal Kolmogorov-Arnold Transformer)는 **Kolmogorov-Arnold 표현 정리의 이론적 기반과 Transformer의 장거리 의존성 포착 능력을 결합**한 새로운 멀티변량 시계열 예측 아키텍처입니다. 이 논문의 핵심 주장은 LSTM 기반의 Temporal Fusion Transformer(TFT)에서 순환 레이어를 TKAN으로 교체하면 특히 **다단계(multi-step) 예측에서 성능이 향상**된다는 것입니다.[^1_1][^1_2]

주요 기여는 다음과 같습니다:

- **TKAN 레이어의 Transformer 통합**: LSTM 대신 TKAN을 인코더·디코더에 적용한 최초의 어텐션 기반 아키텍처 제안[^1_1]
- **Fully Aware Layer 도입**: Multi-Head Attention 출력을 플래튼(flatten)하여 과거 전체 값과 완전 연결, 지속적인 메모리 유지[^1_1]
- **관측 입력 우선 아키텍처**: 사전 알려진 입력(known inputs)보다 관측된 과거 입력이 많은 태스크(예: 금융)에 특화된 설계[^1_1]
- **해석 가능성 향상**: KAN의 학습 가능한 단변량 함수(spline)로 복잡한 비선형 의존성을 더 해석 가능하게 표현[^1_1]

***

## 2. 해결하고자 하는 문제

### 기존 문제

멀티변량 시계열 예측에서 기존 Transformer(TFT)는 다음 구조적 한계를 가집니다:[^1_1]

- **정적 공변량 의존성**: TFT는 알려진(known) 미래 입력이 많을 때 최적화되어, 금융·거래량 같은 관측 입력 중심 태스크에 부적합
- **LSTM의 표현력 한계**: 고정된 활성화 함수를 가진 MLP 기반 게이팅 메커니즘
- **장거리 다단계 예측 성능 저하**: GRU/LSTM 단독 사용 시 예측 스텝이 증가할수록 급격한 성능 하락

***

## 3. 제안 방법 및 핵심 수식

### 3.1 TKAN 레이어 (시간 의존 KAN)

각 변환 함수 $\phi_{l,j,i}$를 시간 의존적으로 만들어 시계열 처리:

$x_{l+1,j}(t) = \sum_{i=1}^{n_l} \tilde{x}\_{l,j,i}(t) = \sum_{i=1}^{n_l} \phi_{l,j,i,t}(x_{l,i}(t),\, h_{l,i}(t))$

여기서 메모리 함수 $h_{l,i}(t)$는:

$h_{l,i}(t) = W_{hh}\, h_{l,i}(t-1) + W_{hz}\, x_{l,i}(t)$

RKAN 레이어가 포함된 전체 시간 의존 KAN:

$\text{KAN}(x, t) = (\Phi_{L-1,t} \circ \Phi_{L-2,t} \circ \cdots \circ \Phi_{0,t})(x, t)$

### 3.2 LSTM 스타일 게이팅 메커니즘

**망각 게이트(Forget gate)**:
$f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)$

**입력 게이트(Input gate)**:
$i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)$

**출력 게이트(Output gate) — KAN 활용**:
$o_t = \sigma(\text{KAN}(\vec{x},\, t))$

**셀 상태(Cell state) 업데이트**:

$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}\_t, \quad \tilde{c}\_t = \sigma(W_c x_t + U_c h_{t-1} + b_c)$

**은닉 상태(Hidden state) 출력**:
$h_t = o_t \odot \tanh(c_t)$

### 3.3 Gated Residual Network (GRN)

$\text{GRN}\_\omega(x) = \text{LayerNorm}\bigl(x + \text{GLU}_\omega(\eta_1)\bigr)$

$\eta_1 = W_{1,\omega}\,\eta_2 + b_{1,\omega}, \quad \eta_2 = \text{ELU}(W_{2,\omega}\,x + b_{2,\omega})$

$\text{GLU}\_\omega(\gamma) = \sigma(W_{4,\omega}\gamma + b_{4,\omega}) \odot (W_{5,\omega}\gamma + b_{5,\omega})$

### 3.4 Variable Selection Network (VSN)

$v_{\chi t} = \text{Softmax}\bigl(\text{GRN}_{v\chi}(\Xi_t)\bigr)$

$\tilde{\xi}\_t^{(j)} = \text{GRN}\_{\tilde{\xi}^{(j)}}\bigl(\xi_t^{(j)}\bigr), \quad \tilde{\xi}\_t = \sum_{j=1}^{m_\chi} v_{\chi t}^{(j)}\, \tilde{\xi}_t^{(j)}$

### 3.5 Multi-Head Self-Attention

$\text{Attention}(Q, K, V) = A(Q,K)\,V, \quad A(Q,K) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_{\text{attn}}}}\right)$

$\text{MultiHead}(Q,K,V) = [H_1, \ldots, H_{m_H}]\,W_H, \quad H_h = \text{Attention}(QW_Q^{(h)},\, KW_K^{(h)},\, VW_V^{(h)})$

### 3.6 최종 출력 (Fully Aware Layer)

어텐션 출력 $[\tilde{H}\_1, \ldots, \tilde{H}_{m_H}]$를 플래튼 후 선형 투영:

$\hat{y}\_{t:t+\tau} = W_{\hat{y}}\,\tilde{H}\_{\text{flat}} + b_{\hat{y}}$

### 3.7 손실 함수

$\text{MSE} = \frac{1}{N}\sum_{i=1}^{N}\bigl(\hat{X}^{(i)}\_{t+1} - X^{(i)}_{t+1}\bigr)^2$

$R^2 = 1 - \frac{\sum\_{i=1}^{N}(\hat{X}^{(i)}\_{t+1} - X^{(i)}\_{t+1})^2}{\sum_{i=1}^{N}(X^{(i)}\_{t+1} - \bar{X}_{t+1})^2}$

***

## 4. 모델 구조

TKAT는 다음 5대 구성 요소로 이루어집니다:[^1_1]


| 구성 요소 | 역할 |
| :-- | :-- |
| **Variable Selection Network (VSN)** | 가장 중요한 공변량만 선택 (전체 파라미터의 ~65% 차지) |
| **TKAN 인코더** | 과거 관측 입력을 처리; 단기 메모리(RKAN)와 셀 상태로 장기 의존성 포착 |
| **TKAN 디코더** | 인코더의 최종 셀 상태를 초기 상태로 받아 미래 시퀀스 생성 |
| **Gated Residual Network (GRN)** | 비선형 관계 모델링 및 불필요한 레이어 우회(skip) 가능 |
| **Temporal Decoder + Fully Aware Layer** | Multi-Head Attention 후 flatten으로 과거 전체 값과 완전 연결 |


***

## 5. 성능 향상 및 한계

### 성능 결과 ($R^2$ 평균, Binance BTC 예측)

| 예측 스텝 | TKAT | TKAN | GRU | LSTM |
| :--: | :--: | :--: | :--: | :--: |
| 1 | 0.305 | 0.351 | 0.365 | 0.356 |
| 3 | **0.218** | 0.199 | 0.201 | 0.061 |
| 6 | **0.180** | 0.141 | 0.083 | −0.226 |
| 9 | **0.165** | 0.117 | 0.087 | −0.291 |
| 12 | **0.149** | 0.105 | 0.018 | −0.473 |
| 15 | **0.145** | 0.086 | 0.033 | −0.404 |

3스텝 이상부터 TKAT가 모든 베이스라인을 일관되게 상회하며, TKAN 대비 $R^2$가 약 **50% 이상 향상**됩니다. 또한 TKAN 레이어를 LSTM으로 대체한 TKATN 대비 평균 **약 5% 성능 향상**이 관찰됩니다.[^1_1]

### 한계점

- **단일 스텝(1-step) 예측 성능 저하**: 과도한 모델 복잡성으로 단순 GRU/TKAN보다 낮은 1-step 성능[^1_1]
- **대규모 파라미터 수**: ~102만 개로 TKAN(~10만 개)의 약 10배; 연산 비용 높음[^1_1]
- **정적 공변량(static covariate) 미지원**: 표준 TFT와 달리 정적 메타데이터 통합 불가[^1_1]
- **단일 도메인 검증**: 암호화폐 거래량 데이터(Binance)에만 실험; 일반화 가능성 미검증[^1_1]
- **드롭아웃 미사용**: 과적합 방지 정규화 없음[^1_1]

***

## 6. 모델의 일반화 성능 향상 가능성

TKAT의 일반화 성능은 여러 측면에서 긍정적 가능성을 보입니다.

**구조적 일반화 강점**:

- **KAN의 로컬 가소성(local plasticity)**: B-spline 기반 파라미터화는 새로운 입력에 적응하면서 기존 표현을 보존 → 재난적 망각(catastrophic forgetting) 완화[^1_3]
- **GLU의 선택적 레이어 우회**: 특정 데이터셋에 불필요한 서브모듈을 억제($\text{GLU} \approx 0$)하여 자동적으로 모델을 단순화, 과적합 방지[^1_1]
- **VSN의 변수 선택**: 도메인에 따라 중요 공변량만 동적으로 선택하므로 잡음 변수에 강인[^1_1]

**실험적 근거**: TKAT 대 TFT-유사 변형(TKAT-A, TKAT-B) 비교에서, TKAT가 **낮은 평균 $R^2$와 분산**을 모두 달성해 태스크 특화 설계가 일반화에 기여함을 시사합니다. 또한 5회 반복 실험에서 표준편차가 일관적으로 낮아(**예: 15-step에서 σ = 0.010**) 재현성과 안정성을 확인합니다.[^1_1]

**개선 여지**: 현재 드롭아웃·가중치 감쇠(weight decay) 미적용, 단일 데이터셋 실험이라는 한계 때문에 ETT, Weather, Traffic 등 표준 벤치마크에서의 검증이 필요합니다. 제로샷 KAN 연구(N-BEATS + KAN)는 KAN이 미지 도메인에서도 **N-BEATS 대비 13%, 표준 KAN 대비 24% 향상**을 보임으로써 일반화 잠재력을 간접적으로 지지합니다.[^1_4]

***

## 7. 관련 최신 연구 비교 (2020년 이후)

| 모델 | 연도 | 핵심 기법 | 강점 | 한계 |
| :-- | :-- | :-- | :-- | :-- |
| **TFT** (Lim et al.) | 2021 | LSTM + Multi-Head Attention + GRN | 해석 가능, 정적 공변량 지원 | 관측 입력 중심 태스크에 약함 |
| **Informer** (Zhou et al.) | 2021 | ProbSparse Attention ( $O(L\log L)$ ) | 장기 시계열 효율 | 복잡한 다변량 패턴 포착 어려움 |
| **TKAN** (Genet \& Inzirillo) | 2024 | LSTM + KAN 게이팅 | 다단계 예측 우수, 경량 | 단독 사용 시 아키텍처 제한 |
| **TKAT** (본 논문) | 2024 | TKAN + Transformer + VSN + Fully Aware Layer | 다단계 예측 SOTA, 안정성 | 파라미터 많음, 단일 도메인 검증 |
| **C-KAN** (Koukaras et al.) | 2024 | Conv + KAN + DILATE loss | 비정상 시계열 강인 | 장기 의존성 취약 |
| **T-KAN / MT-KAN** | 2024 | KAN + symbolic regression | 개념 드리프트 감지, 해석성 | 계산 비용 높음 |
| **TimeKAN** | 2025 | KAN + FFT 주파수 분해 | 다중 주파수 패턴 포착 | 시간 도메인 KAN만 적용 |
| **TFKAN** | 2025 | 시간+주파수 이중 브랜치 KAN | 장기 예측 SOTA, 7개 데이터셋 검증 | 복잡도 높음 |
| **Time-TK** | 2025 | Transformer + KAN (Multi-offset) | MMK 대비 MSE 6.69%↓ | 실증 데이터셋 제한 |

[^1_5][^1_6][^1_7][^1_8][^1_3]

***

## 8. 향후 연구 영향 및 고려 사항

### 앞으로의 연구에 미치는 영향

TKAT는 **"KAN을 Transformer에 통합하는 패러다임"** 의 선구적 사례로서 다음 방향을 촉발했습니다:[^1_8][^1_9]

1. **KAN-Transformer 하이브리드 연구 가속화**: Time-TK, TFKAN 등 후속 모델들이 TKAT의 설계 철학을 확장[^1_8]
2. **도메인 특화 아키텍처 설계 필요성 입증**: 범용 TFT보다 태스크 맞춤 설계가 우월함을 실험적으로 증명[^1_1]
3. **KAN의 Transformer 내 역할 재정의**: KAN이 단독 예측기보다 Transformer 내 서브모듈로 더 효과적임을 시사[^1_10]

### 향후 연구 시 고려할 점

- **표준 벤치마크 검증 필수**: ETTh1/2, ETTm1/2, Weather, Electricity, Traffic 등 표준 데이터셋으로 일반화 성능 확인[^1_11]
- **정규화 전략 도입**: 드롭아웃, 레이어 드롭(Layer Drop), 가중치 감쇠를 통해 ~100만 파라미터 모델의 과적합 방지[^1_1]
- **정적 공변량 통합**: 금융 외 도메인(에너지, 기상)에서 정적 메타데이터를 지원하도록 정적 공변량 인코더 재도입[^1_1]
- **계산 효율 개선**: KAN의 B-spline 연산은 동일 파라미터 수 MLP보다 훈련 시간이 길므로, FastKAN(RBF 기반) 또는 WavKAN(웨이블렛) 등 효율적 KAN 변형 검토[^1_9][^1_8]
- **확률적 예측(Probabilistic Forecasting) 확장**: P-KAN 등 불확실성 정량화 기법을 결합해 예측 신뢰 구간 제공[^1_12]
- **비정상 시계열 대응**: 이동 중위수 정규화만으로는 급격한 레짐 변화(regime shift)에 취약하므로, 적응적 분해 또는 DILATE 손실 함수 활용 고려[^1_5]
- **KAN 효과성 논쟁 대응**: KAN이 시계열에서 항상 유효하지 않다는 최근 연구(MMK, 2025) 를 고려, 어떤 조건에서 KAN이 유익한지에 대한 체계적 절삭(ablation) 연구 필요[^1_11]
<span style="display:none">[^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45]</span>

<div align="center">⁂</div>

[^1_1]: 2406.02486v2.pdf

[^1_2]: https://arxiv.org/abs/2406.02486

[^1_3]: https://arxiv.org/html/2506.12696v1

[^1_4]: https://arxiv.org/abs/2412.17853

[^1_5]: https://www.mdpi.com/2227-7390/12/19/3022

[^1_6]: https://arxiv.org/abs/2406.02496

[^1_7]: http://arxiv.org/pdf/2502.06910.pdf

[^1_8]: https://arxiv.org/html/2602.11190v1

[^1_9]: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5220118

[^1_10]: https://www.datasciencewithmarco.com/blog/kolmogorov-arnold-networks-kans-for-time-series-forecasting

[^1_11]: http://arxiv.org/pdf/2408.11306.pdf

[^1_12]: https://arxiv.org/html/2510.16940v1

[^1_13]: https://ieeexplore.ieee.org/document/11100692/

[^1_14]: https://arxiv.org/abs/2406.17890

[^1_15]: https://arxiv.org/abs/2408.07314

[^1_16]: https://arxiv.org/abs/2411.00278

[^1_17]: https://www.ssrn.com/abstract=4825654

[^1_18]: https://ieeexplore.ieee.org/document/10804420/

[^1_19]: http://arxiv.org/pdf/2406.02496.pdf

[^1_20]: http://arxiv.org/pdf/2405.08790.pdf

[^1_21]: http://arxiv.org/pdf/2406.17890.pdf

[^1_22]: https://arxiv.org/pdf/2502.18410.pdf

[^1_23]: https://arxiv.org/pdf/2502.14045.pdf

[^1_24]: https://arxiv.org/html/2408.07314v3

[^1_25]: https://arxiv.org/pdf/2511.18613.pdf

[^1_26]: https://arxiv.org/pdf/2504.16432.pdf

[^1_27]: https://ar5iv.labs.arxiv.org/html/2412.17853

[^1_28]: https://arxiv.org/pdf/2506.12696.pdf

[^1_29]: https://arxiv.org/html/2408.11306v1

[^1_30]: https://arxiv.org/abs/2411.14904

[^1_31]: https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0320368

[^1_32]: https://arxiv.org/pdf/2601.02310.pdf

[^1_33]: https://arxiv.org/html/2511.08229v1

[^1_34]: https://ar5iv.labs.arxiv.org/html/2502.00980

[^1_35]: https://arxiv.org/html/2506.20935

[^1_36]: https://www.techrxiv.org/doi/full/10.36227/techrxiv.177130591.14244418/v1

[^1_37]: https://dl.acm.org/doi/abs/10.1007/s10586-025-05574-9

[^1_38]: https://openreview.net/forum?id=LWQ4zu9SdQ

[^1_39]: https://www.arxiv.org/abs/2406.02496

[^1_40]: https://deeplearn.org/arxiv/493757/a-temporal-kolmogorov-arnold-transformer-for-time-series-forecasting

[^1_41]: https://www.sciencedirect.com/science/article/pii/S2666546825000618

[^1_42]: https://www.diva-portal.org/smash/get/diva2:1987981/FULLTEXT01.pdf

[^1_43]: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4825654

[^1_44]: https://www.arxiv.org/pdf/2508.04048.pdf

[^1_45]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10611135/

