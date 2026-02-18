# HiPPO-KAN: Efficient KAN Model for Time Series Analysis

***

## 1. 핵심 주장과 주요 기여 요약

**HiPPO-KAN**은 HiPPO(High-order Polynomial Projection Operator) 이론과 Kolmogorov-Arnold Network(KAN)을 결합하여, **파라미터 수를 고정한 채로 장기 시계열 예측 성능을 극적으로 향상**시킨 모델입니다. 핵심 아이디어는 가변 길이의 시계열을 고정 차원 계수 벡터로 압축한 뒤, KAN으로 그 벡터의 동역학을 학습하는 것입니다.[^1_1]

주요 기여는 세 가지입니다:[^1_1]

- **파라미터 효율성 및 확장성**: 윈도우 크기가 커져도 파라미터 수가 4,384개로 고정됨 (KAN은 윈도우 크기에 선형 비례하여 증가)
- **장기 예측 성능 향상**: 윈도우 1,200 기준 MSE $3.26 \times 10^{-7}$ 달성 (KAN 대비 약 12배 향상)
- **Lagging 문제 해결**: HiPPO 계수 공간에서 직접 MSE를 계산하는 수정 손실 함수 도입

***

## 2. 해결하고자 하는 문제

### 문제 정의

기존 시계열 예측 모델(RNN, LSTM, vanilla KAN)은 두 가지 핵심 문제를 가집니다:[^1_1]

1. **장기 의존성 포착 실패**: 윈도우 크기 증가 시 KAN의 파라미터가 선형 증가하고, LSTM/RNN은 장기 의존성을 충분히 포착하지 못함
2. **Lagging 문제**: 예측값이 실제 데이터의 급격한 변화를 즉각 반영하지 못하고 지연되는 현상[^1_1]

***

## 3. 제안하는 방법과 수식

### 3.1 State Space Model 기반

기본 State Space Model(SSM)은 다음과 같이 정의됩니다:[^1_1]

$\frac{d}{dt}x(t) = Ax(t) + Bu(t) \tag{1}$

$y(t) = Cx(t) + Du(t) \tag{2}$

### 3.2 HiPPO 이론 — 온라인 함수 근사

단변량 시계열 $u(s)$를 시간 $t$까지 직교 기저 함수 $p_n(t,s)$로 투영합니다:[^1_1]

$x_n(t) = \int_0^t ds\, \omega(t,s)\, p_n(t,s)\, u(s) = \langle u, p_n(t) \rangle_\omega \tag{3}$

완비성 가정 하에 함수 근사는 다음과 같습니다:[^1_1]

$u(s) \approx \sum_{n=1}^{N} c_n\, p_n(L, s) \tag{4}$

여기서 $c_n$은 계수, $N$은 상태 공간 차원으로 시퀀스 길이 $L$과 무관합니다.

### 3.3 KAN의 수식 — 활성화 함수 학습

KAN은 Kolmogorov-Arnold 표현 정리를 기반으로 다변량 함수를 단변량 함수의 합성으로 분해합니다:[^1_1]

$f(x_1, \ldots, x_n) = \sum_{q=1}^{2n+1} \Phi_q\left(\sum_{p=1}^{n} \phi_{q,p}(x_p)\right) \tag{5}$

각 노드의 잔차 활성화 함수는:[^1_1]

$\phi(x) = w_b\, b(x) + w_s\, \text{spline}(x) \tag{6}$

### 3.4 HiPPO-KAN 전체 파이프라인

모델은 세 단계로 구성됩니다:[^1_1]

$\text{HiPPO-KAN} \equiv \text{hippo}^{-1}_{L+1} \circ \text{KAN} \circ \text{hippo}_L \tag{7}$

- **인코딩**: $\text{hippo}\_L: \mathbb{R}^L \to \mathbb{R}^N, \quad u_{1:L} \mapsto c^{(L)}$
- **변환 (KAN)**: $\text{KAN}: \mathbb{R}^N \to \mathbb{R}^N, \quad c^{(L)} \mapsto c^{(L+1)}$
- **디코딩**: $\text{hippo}^{-1}\_{L+1}: \mathbb{R}^N \to \mathbb{R}^{L+1}, \quad c^{(L+1)} \mapsto u'_{1:L+1}$

KAN 레이어의 계수 업데이트는 다음과 같습니다:[^1_1]

$c'\_n = \sum_{m=1}^{N} \Phi_{nm}(c_m) \tag{8}$

Leg-S 기저 함수($p_n(L+1,\, L+1) = \sqrt{2n+1}$)를 적용하면 최종 예측값은:[^1_1]

$\hat{u}\_{L+1} = \sum_{n=1}^{N} \sqrt{2n+1} \left(\sum_{m=1}^{N} \Phi_{nm}(c_m) + B\, u_L\right) \tag{9}$

### 3.5 수정된 손실 함수 (Lagging 문제 해결)

기존 시간 도메인 MSE 대신, HiPPO 계수 공간에서 직접 손실을 계산합니다:[^1_1]

$\mathcal{L}(\theta) = \frac{1}{D} \sum_{i=1}^{D} \left\| c^{(L+1)(i)}_{\text{true}} - \hat{c}^{(L+1)(i)} \right\|^2 \tag{10}$

여기서 $c^{(L+1)(i)}\_{\text{true}} = \text{hippo}\_{L+1}(u^{(i)}_{1:L+1})$, $\hat{c}^{(L+1)(i)} = \text{KAN}(c^{(L)(i)})$입니다.[^1_1]

***

## 4. 모델 구조

HiPPO-KAN의 구조는 오토인코더(Auto-Encoder) 패러다임을 따릅니다:[^1_1]

```
입력 시계열 u_{1:L}
      ↓ HiPPO 인코딩 (고정 행렬 A, B)
계수 벡터 c^(L) ∈ ℝ^N   ← 시퀀스 길이 무관, 차원 고정
      ↓ KAN 변환 (학습 가능한 스플라인 함수)
예측 계수 c^(L+1) ∈ ℝ^N
      ↓ HiPPO 역변환
확장 시계열 u'_{1:L+1}
```

실험 구성:[^1_1]

- 기본 아키텍처: `[16, 16]` (4,384 파라미터)
- 병목 아키텍처: `[16, 2, 16]` (1,344 파라미터) — 정보 병목 이론 적용

***

## 5. 성능 향상 및 한계

### 성능 결과 (윈도우 크기 × MSE)

| 모델 | 윈도우 120 | 윈도우 500 | 윈도우 1200 | 파라미터 |
| :-- | :-- | :-- | :-- | :-- |
| **HiPPO-KAN** | $3.40\times10^{-7}$ | $3.34\times10^{-7}$ | **$3.26\times10^{-7}$** | **4,384 (고정)** |
| HiPPO-MLP | $2.33\times10^{-6}$ | $2.68\times10^{-5}$ | $5.87\times10^{-6}$ | 9,792 |
| KAN | $8.9\times10^{-7}$ | $1.66\times10^{-6}$ | $4.03\times10^{-6}$ | 1,680 → 16,800 |
| LSTM | $4.69\times10^{-7}$ | $6.50\times10^{-7}$ | $9.21\times10^{-7}$ | 4,513 (고정) |
| RNN | $1.14\times10^{-6}$ | $1.09\times10^{-6}$ | $1.18\times10^{-6}$ | 12,673 |

[^1_1]

윈도우를 4,000까지 확대해도 HiPPO-KAN의 MSE는 $3.33\times10^{-7}$에서 $4.38\times10^{-7}$로 소폭 증가(약 1.3배)에 그쳐, 33배 윈도우 확장에도 예외적인 확장성을 보입니다.[^1_1]

### 주요 한계

- **단변량(univariate) 한정**: 실험이 BTC-USDT 1분봉 단일 데이터에만 한정되어 다변량 시계열 일반화 검증 부재[^1_1]
- **단기(1-step) 예측 한정**: 다중 스텝(multi-horizon) 예측 성능 미검증[^1_1]
- **데이터셋 다양성 부족**: 금융 데이터 1종만 사용, ETTh/Weather 등 표준 벤치마크 미포함
- **HiPPO 차수 N의 선택 민감성**: 작은 N은 정보 손실, 큰 N은 계산 오버헤드 유발

***

## 6. 일반화 성능 향상 가능성

이 논문에서 일반화 성능 향상과 관련된 핵심 메커니즘은 다음 세 가지입니다:[^1_1]

**① 윈도우별 정규화(Window-specific Normalization)**
각 윈도우마다 $(u_t - \mu)/\mu$ 정규화를 독립 적용하여 비정상성(non-stationarity) 문제를 완화합니다. 이는 데이터 스케일 변화에 무관한 패턴 학습을 가능하게 하여 여러 시계열 구간에 걸친 일반화를 향상시킵니다.[^1_1]

**② 정보 병목 이론(Information Bottleneck) 적용**
`[16, 2, 16]` 병목 아키텍처가 파라미터를 69%(4,384 → 1,344)나 줄임에도 불구하고 더 나은 성능을 달성합니다. 이는 Tishby \& Zaslavsky(2015)의 정보 병목 이론과 일치하며, 압축된 표현이 과적합을 억제하고 강인한 특징 추출을 유도한다는 증거입니다.[^1_1]

**③ 배치 평균화의 정규화 효과**
식 (10)의 배치 학습은 개별 샘플의 이상 변동을 평균화하여 모델이 공통 동역학(common dynamics)으로 수렴하도록 유도합니다:[^1_1]

$\mathcal{L}(\theta) = \frac{1}{D}\sum_{i=1}^{D}\left\|c^{(L+1)(i)}_{\text{true}} - \hat{c}^{(L+1)(i)}\right\|^2$

이 메커니즘은 잡음에 강인한 계수 표현을 학습하게 하여 미지의 데이터 구간에 대한 일반화를 자연스럽게 향상시킵니다.

***

## 7. 2020년 이후 관련 연구 비교 분석

| 논문 | 연도 | 핵심 방법 | 강점 | HiPPO-KAN 대비 차이 |
| :-- | :-- | :-- | :-- | :-- |
| **HiPPO / S4** (Gu et al.) | 2020–2022 | SSM + 직교 기저 투영 | 장기 의존성 이론 기반 | HiPPO-KAN의 이론적 토대 [^1_2] |
| **KAN** (Liu et al.) | 2024 | B-스플라인 활성화 학습 | MLP 대비 파라미터 효율 | 파라미터 수가 윈도우 크기에 비례 [^1_1] |
| **TKAN** (Genet \& Inzirillo) | 2024 | KAN + LSTM 융합 | 멀티스텝 예측 | 복잡한 구조, 높은 계산량 [^1_3] |
| **TKAT** | 2024 | KAN + Transformer | 멀티변량, 어텐션 기반 | 복잡도 증가, 해석성 일부 저하 [^1_4] |
| **T-KAN / MT-KAN** (Xu et al.) | 2024 | 개념 드리프트 탐지 + KAN | 해석성, 비정상 시계열 | 멀티변량 지원하나 구조 복잡 [^1_5] |
| **KAN-AD** | 2024 | 푸리에 기반 KAN | 이상 탐지, 경량화(< 1,000 파라미터) | 이상탐지 특화, 예측은 미검증 [^1_6] |
| **TimeKAN** | 2025 | 주파수 분해 + KAN | 장기 예측, 다중 주파수 분리 | 더 복잡한 다변량 벤치마크 검증 [^1_7] |
| **HaKAN** | 2026 | Hahn 다항식 KAN + 채널 독립 | SOTA 수준, 다변량 | 더 광범위한 벤치마크, 최신 [^1_8] |


***

## 8. 향후 연구에 미치는 영향과 고려 사항

### 연구에 미치는 영향

논문은 **HiPPO 기반 계수 공간에서의 학습**이라는 새로운 패러다임을 제시합니다. 이는 SSM 계열 모델(S4, Mamba 등)과 KAN을 연결하는 가교 역할을 하며, 특히 길이-불변(length-invariant) 표현 학습의 가능성을 열었습니다. 병목 구조의 정보 압축이 성능을 향상시킨다는 실험 결과는 다른 경량 시계열 모델 설계에도 중요한 시사점을 줍니다.[^1_1]

### 앞으로 연구 시 고려할 점

논문 자체가 제안한 미래 방향으로, **Graph Neural Network(GNN)와의 통합**을 통해 각 변수를 그래프 노드로 표현하고 HiPPO 계수를 게이지 벡터처럼 활용하는 다변량 확장이 핵심 과제입니다. 이 외에도 다음이 중요합니다:[^1_1]

- **표준 벤치마크 검증**: ETTh1/h2, Weather, Exchange 등 공개 데이터셋 적용으로 비교 우위 확인 필요
- **다중 스텝 예측 확장**: 현재 1-step ahead에 한정된 실험을 horizon=96/192/336으로 확장
- **HiPPO 차수 $N$ 자동 선택**: Bayesian 최적화나 NAS(신경망 아키텍처 탐색)를 통한 자동화
- **비금융 도메인 적용**: 기상, 의료, 에너지 소비 데이터에서의 일반화 검증
- **KAN 2.0과의 통합**: Liu et al. (2024)의 KAN 2.0 이 제공하는 심볼릭 회귀 기능을 HiPPO 계수 해석에 활용하면 모델의 물리적 해석 가능성을 크게 높일 수 있음[^1_1]
- **비정상 시계열 처리**: 현재의 윈도우별 정규화를 적응형 정규화(adaptive normalization)나 RevIN으로 교체하면 더 강건한 일반화 기대 가능
<span style="display:none">[^1_10][^1_11][^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_9]</span>

<div align="center">⁂</div>

[^1_1]: 2410.14939v1.pdf

[^1_2]: https://arxiv.org/abs/2206.12037

[^1_3]: https://www.ssrn.com/abstract=4825654

[^1_4]: https://arxiv.org/pdf/2406.02486.pdf

[^1_5]: https://arxiv.org/abs/2406.02496

[^1_6]: https://arxiv.org/abs/2411.00278

[^1_7]: http://arxiv.org/pdf/2502.06910.pdf

[^1_8]: https://papers.cool/arxiv/2601.18837

[^1_9]: https://arxiv.org/abs/2410.10041

[^1_10]: https://www.mdpi.com/2227-7390/12/19/3022

[^1_11]: https://ieeexplore.ieee.org/document/11100692/

[^1_12]: https://arxiv.org/abs/2406.17890

[^1_13]: https://arxiv.org/abs/2410.08041

[^1_14]: https://arxiv.org/abs/2412.01224

[^1_15]: https://ieeexplore.ieee.org/document/10804420/

[^1_16]: http://arxiv.org/pdf/2405.08790.pdf

[^1_17]: http://arxiv.org/pdf/2406.02496.pdf

[^1_18]: http://arxiv.org/pdf/2406.17890.pdf

[^1_19]: http://arxiv.org/pdf/2408.11306.pdf

[^1_20]: https://arxiv.org/pdf/2502.18410.pdf

[^1_21]: http://arxiv.org/pdf/2411.00278.pdf

[^1_22]: https://arxiv.org/abs/2405.08790

[^1_23]: https://arxiv.org/html/2405.08790v2

[^1_24]: https://arxiv.org/pdf/2601.02310.pdf

[^1_25]: https://ar5iv.labs.arxiv.org/html/2412.17853

[^1_26]: https://arxiv.org/html/2404.16112v1

[^1_27]: https://arxiv.org/html/2506.12696v1

[^1_28]: https://arxiv.org/abs/2408.11306

[^1_29]: https://arxiv.org/html/2411.19455v1

[^1_30]: https://arxiv.org/html/2406.02486v2

[^1_31]: https://arxiv.org/pdf/2504.16432.pdf

[^1_32]: https://arxiv.org/pdf/2506.05065.pdf

[^1_33]: https://arxiv.org/html/2406.02496v1

[^1_34]: https://www.arxiv.org/pdf/2509.02967.pdf

[^1_35]: https://arxiv.org/pdf/2403.17445.pdf

[^1_36]: https://www.arxiv.org/abs/2406.02496

[^1_37]: https://www.semanticscholar.org/paper/Kolmogorov-Arnold-Networks-(KANs)-for-Time-Series-Vaca-Rubio-Blanco/081eb8781725e560f597b01c624fe65618c3c0f8

[^1_38]: https://www.datasciencewithmarco.com/blog/kolmogorov-arnold-networks-kans-for-time-series-forecasting

[^1_39]: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5220118

[^1_40]: https://www.emergentmind.com/topics/structured-state-space-sequence-model-s4

[^1_41]: https://www.themoonlight.io/en/review/a-temporal-kolmogorov-arnold-transformer-for-time-series-forecasting

[^1_42]: https://neptune.ai/blog/state-space-models-as-natural-language-models

[^1_43]: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4825654

[^1_44]: https://huggingface.co/blog/lbourdois/ssm-2022

[^1_45]: https://arxiv.org/html/2405.07344v1

[^1_46]: https://www.semanticscholar.org/paper/Kolmogorov-Arnold-Networks-for-Time-Series:-Power-Xu-Chen/10145b2238569436754c4d9be3f9c7db501cc65c

