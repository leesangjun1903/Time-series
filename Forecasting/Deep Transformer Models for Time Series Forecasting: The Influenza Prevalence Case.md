# Deep Transformer Models for Time Series Forecasting: The Influenza Prevalence Case

# 1. 핵심 주장과 주요 기여

이 논문은 시계열 예측에 Transformer 아키텍처를 적용한 초기 연구로, RNN/LSTM의 한계를 극복하고자 합니다. 주요 기여는 (1) 시계열 예측을 위한 범용 Transformer 기반 모델 개발, (2) state space 모델과의 상호보완성 입증, (3) 인플루엔자 유사 질병(ILI) 예측에서 당시 state-of-the-art 달성입니다.[^1_1]

# 2. 상세 분석

## 해결하고자 하는 문제

전통적인 sequence-aligned 모델들(RNN, LSTM, CNN)은 시계열 데이터 모델링에서 다음과 같은 한계를 보입니다:[^1_1]

- **Gradient vanishing/exploding 문제**: RNN/LSTM은 장기 의존성 학습에 어려움
- **Convolutional filter의 제약**: CNN은 receptive field가 제한적
- **순차 처리의 비효율성**: 시퀀스를 순차적으로 처리하여 병렬화 곤란


## 제안하는 방법 (수식 포함)

### Encoder 구조

입력 시계열 $x_1, x_2, \ldots, x_{10}$을 $d_{model}$ 차원 벡터로 매핑합니다:[^1_1]

**Positional Encoding**:

$$
PE_{(pos, 2i)} = \sin(pos/10000^{2i/d_{model}})
$$

$$
PE_{(pos, 2i+1)} = \cos(pos/10000^{2i/d_{model}})
$$

**Self-Attention Mechanism**:

$$
Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

여기서 $Q, K, V$는 각각 Query, Key, Value 행렬이며, $d_k$는 key 차원입니다.[^1_1]

### Decoder 구조

Decoder는 마지막 encoder 입력 $x_{10}$부터 시작하여 $x_{11}, \ldots, x_{14}$를 예측합니다. Look-ahead masking을 적용하여 미래 정보 누출을 방지합니다.[^1_1]

### 학습 절차

**Adam Optimizer**:

$$
lrate = d_{model}^{-0.5} \cdot \min(step\_num^{-0.5}, step\_num \cdot warmup\_steps^{-1.5})
$$

여기서 $warmup\_steps = 5000$이며, $\beta_1 = 0.9$, $\beta_2 = 0.98$, $\epsilon = 10^{-9}$입니다.[^1_1]

**Regularization**: 모든 sub-layer에 dropout rate 0.2 적용[^1_1]

## 모델 구조

### 주요 컴포넌트

1. **Input Layer**: 시계열을 $d_{model}$ 차원으로 변환
2. **4개의 Encoder Layer**: 각각 self-attention과 feed-forward sub-layer 포함
3. **4개의 Decoder Layer**: encoder-decoder attention 추가
4. **Output Layer**: 최종 예측 시계열 생성[^1_1]

### Time Delay Embedding (TDE) 활용

Time Delay Embedding은 다음과 같이 정의됩니다:[^1_1]

$$
TDE_{d,\tau}(x_t) = (x_t, x_{t-\tau}, \ldots, x_{t-(d-1)\tau})
$$

Takens' theorem에 따르면, 적절한 $(d, \tau)$ 선택 시 TDE는 원래 dynamical system의 phase space를 복원할 수 있습니다. 실험 결과 **차원 8에서 최적 성능**(RMSE = 0.605)을 달성했습니다.[^1_1]

## 성능 향상 및 한계

### 성능 지표

| 모델 | Pearson Correlation | RMSE | 상대 개선 |
| :-- | :-- | :-- | :-- |
| ARIMA | 0.769 | 1.020 | Baseline |
| LSTM | 0.924 (+19.9%) | 0.807 | -20.9% RMSE |
| Seq2Seq+Attn | 0.920 (+19.5%) | 0.642 | -37.1% RMSE |
| **Transformer** | **0.928 (+20.7%)** | **0.588** | **-42.4% RMSE** |

[^1_1]

**ARGONet 대비 성능**:

- Mean Correlation: 0.931 (ARGONet: 0.912)[^1_1]
- Mean RMSE: 0.593 (ARGONet: 0.550)[^1_1]


### 한계점

1. **제한된 데이터셋**: ILI 데이터만으로 일반화 성능 검증 부족[^1_1]
2. **계산 복잡도**: Transformer의 quadratic complexity 문제 미해결
3. **긴 시퀀스 처리**: 매우 긴 예측 horizon에서의 성능 미검증[^1_1]
4. **해석 가능성**: Attention weight 분석 부재

# 3. 모델의 일반화 성능 향상 가능성

## 핵심 강점

### Self-Attention의 장점

- **병렬 처리**: 전체 시퀀스를 동시에 처리하여 학습 효율성 증대[^1_1]
- **장기 의존성 포착**: 임의의 거리에 있는 time step 간 직접 연결[^1_1]
- **가변 길이 입력**: 다양한 길이의 시계열에 유연하게 적용 가능[^1_1]


### State Space Model과의 상호보완성

논문은 Transformer가 다음 방식으로 일반화될 수 있음을 시사합니다:[^1_1]

1. **관찰 데이터 모델링**: 직접적인 시계열 값 예측
2. **상태 변수 모델링**: TDE를 통한 phase space 복원
3. **다변량 확장**: 최소한의 수정으로 multivariate 시계열 처리

### Channel-Independence의 잠재력

논문의 global model은 모든 state 데이터로 학습되어 US-level 예측에서 최고 성능(correlation = 0.984)을 달성했습니다. 이는 **transfer learning 가능성**을 시사합니다.[^1_1]

## 개선 방향

1. **효율성**: Sparse attention mechanism 도입 필요
2. **도메인 지식 통합**: Static covariates 활용 확대
3. **불확실성 정량화**: Confidence interval 예측 기능 추가

# 4. 앞으로의 연구에 미치는 영향과 고려사항

## 역사적 영향력

이 논문은 **시계열 예측에 Transformer를 적용한 선구적 연구**로, 이후 다수의 후속 연구에 영감을 제공했습니다.[^1_1]

## 고려사항

### 1. 계산 효율성 문제

- **메모리 복잡도**: $O(L^2)$ (L은 시퀀스 길이)
- **권장사항**: Sparse attention 또는 linear attention 활용


### 2. 모델 용량과 데이터 크기 균형

- **과적합 위험**: 작은 데이터셋에서 dropout과 regularization 필수[^1_1]
- **권장사항**: Pre-training + fine-tuning 전략 고려


### 3. 도메인 특화 설계

- **시계열 특성**: Seasonality, trend 명시적 모델링 부족
- **권장사항**: Decomposition 기법과 결합


# 5. 2020년 이후 관련 최신 연구 비교 분석

## 주요 발전 방향

### (1) Efficient Attention Mechanisms

**Informer (2021)**[^1_2]

- **핵심 개선**: ProbSparse Self-Attention으로 복잡도 $O(L \log L)$로 감소
- **성능**: Long Sequence Time-Series Forecasting (LSTF)에서 우수
- **방법론**:

$$
Q' = \text{Top-k}\left[KL(q_i \parallel p)\right]
$$

상위 k개 query만 사용하여 계산량 대폭 감소

**Autoformer (2021)**[^1_3][^1_4]

- **핵심 개선**: Auto-Correlation mechanism으로 주기성 포착

$$
AutoCorr(\tau) = \frac{1}{L}\sum_{t=1}^{L} x_t x_{t-\tau}
$$
- **특징**: Series decomposition (trend + seasonality) 통합
- **성능**: ETT, Weather 데이터셋에서 SOTA


### (2) Patching Strategy

**PatchTST (2023)**[^1_5][^1_6][^1_7][^1_3]

- **혁신**: 시계열을 패치로 분할, "A Time Series is Worth 64 Words"
- **Channel-Independence**: 각 채널을 독립적으로 처리
- **성능**:
    - ECL: MSE 0.205, MAE 0.290[^1_3]
    - Traffic: MSE 0.428, MAE 0.282[^1_3]
    - PEMS: MSE 0.119, MAE 0.218[^1_3]
- **장점**:
    - 계산 효율 대폭 향상
    - Local semantic 정보 보존
    - Self-supervised learning 지원[^1_6]


### (3) Inverted Architecture

**iTransformer (2023)**[^1_8][^1_3]

- **핵심 아이디어**: 시간축과 변수축 전치 (inverted dimensions)
    - 각 time series를 token으로 처리
    - Self-attention으로 multivariate correlation 포착
- **성능 우위**:
    - ECL: MSE **0.178**, MAE **0.270**[^1_3]
    - Traffic: MSE **0.428**, MAE **0.282**[^1_3]
    - 전체 평균 **6-22% 개선**[^1_3]
- **특징**: PatchTST의 "spatial indistinguishability" 문제 해결[^1_9]


### (4) Interpretable Forecasting

**Temporal Fusion Transformer (TFT) (2019-2021)**[^1_10][^1_11][^1_12][^1_13]

- **핵심 설계**:

1. **Variable Selection Network**: 중요 feature 자동 선택
2. **Gated Residual Network (GRN)**: 비선형 처리
3. **Multi-head Attention**: 장기 의존성
4. **Quantile Forecasting**: 불확실성 정량화
- **입력 유형**:[^1_13][^1_10]
    - Static covariates (시불변)
    - Known future inputs (미래 알려진 값)
    - Observed inputs (과거만 알려진 값)
- **성능**: Electricity, Traffic, Volatility 예측에서 우수[^1_4][^1_14]


## 비교표: Wu et al. (2020) vs. 최신 모델들

| 모델 | 발표연도 | 핵심 혁신 | 복잡도 | ETT 성능 (MSE) | 해석가능성 |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Wu et al. | 2020 | Vanilla Transformer for TS | $O(L^2)$ | N/A | 낮음 |
| **Informer** | 2021 | ProbSparse Attention[^1_2] | $O(L \log L)$ | ~0.45 | 중간 |
| **Autoformer** | 2021 | Auto-Correlation[^1_3] | $O(L \log L)$ | ~0.46 | 중간 |
| **TFT** | 2021 | Variable Selection + GRN[^1_13] | $O(L^2)$ | N/A | **높음** |
| **PatchTST** | 2023 | Patching + CI[^1_3][^1_6] | $O((L/P)^2)$ | **0.381** | 낮음 |
| **iTransformer** | 2023 | Inverted Dimensions[^1_3] | $O(M^2)$ | **0.383** | 중간 |

여기서 L = 시퀀스 길이, P = 패치 크기, M = 변수 개수

## 최신 트렌드 (2024-2026)

### Linear vs. Transformer 논쟁

**주요 발견**:[^1_15][^1_16]

- 단순 선형 모델이 많은 경우 Transformer를 능가
- Transformer의 permutation-invariance가 시계열 예측에 불리
- **해결책**:
    - Positional encoding 강화
    - Patch-based processing[^1_6]
    - Channel-independence 전략[^1_3]


### Foundation Models \& LLMs

**TimeLLM, TimesFM (2024-2025)**:[^1_17][^1_18]

- LLM의 semantic 정보와 Transformer의 temporal dynamics 결합
- **과제**: 계산 비효율, "pattern deterioration"[^1_18]
- **방향**: Few-shot learning, zero-shot forecasting


### Specialized Architectures

**PeriodNet (2025)**:[^1_19][^1_20]

- Period attention mechanism으로 temporal similarity 포착
- 720-step 예측에서 **22% 개선**[^1_20]

**LSEAttention (2025)**:[^1_21]

- Entropy collapse 문제 해결
- 적은 파라미터로 경쟁력 있는 성능

**Mamba-based Models (2024-2025)**:[^1_22]

- State Space Model과 Transformer 결합
- 24-hour periodic dependency 명시적 활용[^1_22]


## 실무 응용 사례

### 에너지 예측

- **Wind Power** (2025): Transformer가 LSTM/GRU 대비 RMSE 370 MW 달성 (vs. 395+ MW)[^1_23]
- **Solar PV** (2025): PatchTST + Adaptive Conformal Inference로 R² 0.9696[^1_24][^1_22]


### 의료/역학 예측

- **COVID-19** (2025): SEIR-Informer hybrid로 mechanistic + data-driven 결합[^1_25]
- **ILI 예측**: Wu et al. (2020) 이후 attention 메커니즘 표준화[^1_1]


### 금융 예측

- **Volatility Forecasting** (2024): Informer, Autoformer, PatchTST가 HAR 모델 능가[^1_26][^1_27]
- **주의사항**: 제한된 데이터에서 second-generation Transformer가 효과적[^1_27]


## 연구 시 핵심 고려사항

### 1. 모델 선택 가이드라인

**긴 시퀀스 (>512)**:

- Informer 또는 Sparse attention 변형[^1_2]
- 또는 PatchTST로 패치 크기 조정[^1_6]

**다변량 (>50 변수)**:

- iTransformer로 변수 간 관계 모델링[^1_3]
- 또는 Channel-independent 전략[^1_6]

**해석가능성 중요**:

- TFT 또는 Attention weight 분석 도구 활용[^1_13]


### 2. 데이터 특성 고려

**주기성 강함**:

- Autoformer의 auto-correlation[^1_3]
- 또는 PeriodNet[^1_20]

**노이즈 많음**:

- TDE로 phase space 복원[^1_1]
- 또는 Robust normalization[^1_16]

**분포 변화 (distribution shift)**:

- Adaptive normalization[^1_28]
- 또는 Few-shot adaptation[^1_29]


### 3. 실무 배치 고려사항

**엣지 디바이스**:

- Knowledge Distillation으로 모델 압축[^1_22]
- 예: Transformer 23.5% 압축 후 오히려 MAE 감소[^1_22]

**실시간 예측**:

- 추론 복잡도 우선 고려
- PatchTST의 6.06× 메모리 절감 활용[^1_30]


### 4. 이론적 고려사항

**Attention의 역할**:[^1_15][^1_16]

- 시계열에서 attention은 "robust kernel representation"으로 작용[^1_31]
- 단순히 temporal dependency보다는 noise robustness 제공

**Generalization bound**:

- 변수 수 M vs. 시퀀스 길이 L trade-off
- Channel-independence가 일반화에 유리할 수 있음[^1_3]


## 결론

Wu et al. (2020)의 연구는 시계열 예측에 Transformer를 적용한 **초석**을 마련했으며, 이후 5년간 다음과 같은 발전을 이끌었습니다:[^1_1]

1. **효율성**: $O(L^2)$ → $O(L \log L)$ 또는 $O((L/P)^2)$
2. **정확도**: ILI 예측 RMSE 0.588 → ETT 예측 MSE 0.38[^1_3][^1_1]
3. **적용 범위**: 단변량 → 고차원 다변량 + 해석가능성
4. **실용성**: 연구 → 산업 응용 (에너지, 금융, 의료)

앞으로의 연구는 (1) Foundation model 활용, (2) Few-shot/Zero-shot learning, (3) 인과관계 모델링, (4) 불확실성 정량화에 집중해야 합니다.[^1_28][^1_16]
<span style="display:none">[^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72]</span>

<div align="center">⁂</div>

[^1_1]: 2001.08317v1.pdf

[^1_2]: http://arxiv.org/pdf/2012.07436.pdf

[^1_3]: https://arxiv.org/html/2310.06625v4

[^1_4]: https://www.mdpi.com/2227-7390/12/17/2728

[^1_5]: https://arxiv.org/pdf/2211.14730.pdf

[^1_6]: https://github.com/PatchTST/PatchTST

[^1_7]: https://github.com/yuqinie98/PatchTST

[^1_8]: https://arxiv.org/abs/2310.06625

[^1_9]: https://onlinelibrary.wiley.com/doi/10.1002/for.70105

[^1_10]: https://www.mathworks.com/help/deeplearning/ug/time-series-forecasting-using-temporal-fusion-transformer.html

[^1_11]: https://arxiv.org/html/2508.04048v1

[^1_12]: https://catalog.ngc.nvidia.com/orgs/nvidia/resources/tft_for_pytorch

[^1_13]: https://arxiv.org/abs/1912.09363

[^1_14]: https://ieeexplore.ieee.org/document/11155736/

[^1_15]: https://openreview.net/pdf?id=eBCk0nXz17

[^1_16]: https://openreview.net/forum?id=kHEVCfES4Q\&noteId=mrNbq9EkQa

[^1_17]: https://arxiv.org/html/2507.10098v1

[^1_18]: https://www.mdpi.com/2076-2615/15/21/3180

[^1_19]: https://arxiv.org/html/2511.19497v1

[^1_20]: https://arxiv.org/abs/2511.19497

[^1_21]: http://arxiv.org/pdf/2410.23749.pdf

[^1_22]: https://arxiv.org/abs/2512.23898

[^1_23]: https://www.mdpi.com/2071-1050/17/19/8655

[^1_24]: https://www.mdpi.com/1996-1073/18/18/5000

[^1_25]: https://ieeexplore.ieee.org/document/11154197/

[^1_26]: https://wires.onlinelibrary.wiley.com/doi/10.1002/widm.1519

[^1_27]: https://www.ssrn.com/abstract=4718033

[^1_28]: https://www.semanticscholar.org/paper/41742e4ae0f3c4afcf5a90eefbd685d63134d911

[^1_29]: https://arxiv.org/pdf/2502.16294.pdf

[^1_30]: https://arxiv.org/html/2506.16001v2

[^1_31]: https://arxiv.org/html/2402.05370v1

[^1_32]: https://ar5iv.labs.arxiv.org/html/2007.06028

[^1_33]: https://arxiv.org/pdf/2307.05909.pdf

[^1_34]: https://arxiv.org/html/2407.17877v1

[^1_35]: https://arxiv.org/pdf/2504.16548.pdf

[^1_36]: https://pdfs.semanticscholar.org/6a01/e3921ec169c08be1feecf097544b2a648ab4.pdf

[^1_37]: https://arxiv.org/html/2505.18442v1

[^1_38]: http://arxiv.org/list/physics/2023-10?skip=680\&show=2000

[^1_39]: https://journal.unpacti.ac.id/index.php/JSCE/article/view/2373

[^1_40]: https://journals.scholarpublishing.org/index.php/AIVP/article/view/19906

[^1_41]: https://link.springer.com/10.1007/s11027-025-10279-w

[^1_42]: https://www.mdpi.com/1999-4907/14/8/1596

[^1_43]: https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12791/3005110/Analysis-and-forecast-of-supercomputing-power-load-based-on-time/10.1117/12.3005110.full

[^1_44]: http://arxiv.org/pdf/2211.14730v2.pdf

[^1_45]: https://arxiv.org/abs/2207.05397

[^1_46]: http://arxiv.org/pdf/2408.09723.pdf

[^1_47]: https://arxiv.org/pdf/2502.13721.pdf

[^1_48]: https://arxiv.org/pdf/2304.08424.pdf

[^1_49]: http://arxiv.org/pdf/2405.14982.pdf

[^1_50]: https://arxiv.org/html/2410.16881v1

[^1_51]: https://www.sciencedirect.com/science/article/pii/S2665963824001040

[^1_52]: https://github.com/ddz16/TSFpaper

[^1_53]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12453695/

[^1_54]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11686916/

[^1_55]: https://pubmed.ncbi.nlm.nih.gov/37896601/

[^1_56]: https://arxiv.org/abs/2505.20048

[^1_57]: https://arxiv.org/html/2507.13043v1

[^1_58]: https://arxiv.org/abs/2411.11046

[^1_59]: https://arxiv.org/html/2510.23396v1

[^1_60]: https://arxiv.org/abs/2508.04048

[^1_61]: https://arxiv.org/abs/2508.18130

[^1_62]: https://www.semanticscholar.org/paper/8a8328e1b97b95925d9f83dce321ac01c4b86933

[^1_63]: http://arxiv.org/pdf/2410.05726.pdf

[^1_64]: https://arxiv.org/pdf/2206.04038.pdf

[^1_65]: https://arxiv.org/pdf/2404.10458.pdf

[^1_66]: http://arxiv.org/pdf/2311.18780.pdf

[^1_67]: https://arxiv.org/pdf/2501.10448.pdf

[^1_68]: http://arxiv.org/pdf/2310.00655.pdf

[^1_69]: https://data-newbie.tistory.com/943

[^1_70]: https://velog.io/@haemin_jang/PatchTST-%EB%85%BC%EB%AC%B8-%EC%A0%95%EB%A6%AC

[^1_71]: https://hasaero2.tistory.com/30

[^1_72]: https://openreview.net/pdf?id=JePfAI8fah

