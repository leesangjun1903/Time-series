
# Adversarial Sparse Transformer for Time Series Forecasting 

## 1. 핵심 주장과 주요 기여

이 논문(NeurIPS 2020)은 시계열 예측을 위한 Adversarial Sparse Transformer(AST)를 제안하며, Generative Adversarial Networks(GAN)와 Sparse Transformer를 결합한 새로운 접근법입니다.[^1_1]

**핵심 주장:**

- 기존 시계열 예측 모델들은 단일 목적함수(MSE, likelihood loss 등)만 최적화하여 실제 시계열의 확률적 특성을 충분히 포착하지 못함[^1_1]
- Auto-regressive 모델의 teacher forcing 전략은 훈련과 추론 간 불일치를 야기하여 오류 누적(error accumulation) 문제 발생[^1_1]
- Sparse attention mechanism은 시계열의 중요한 과거 시점에만 집중하여 효율성과 정확도를 향상[^1_1]

**주요 기여:**

1. GAN 기반 adversarial training을 시계열 예측에 최초로 도입하여 sequence-level 정규화 제공[^1_1]
2. α-entmax를 활용한 sparse attention으로 관련 없는 시점에 정확히 0의 가중치 할당[^1_1]
3. 실험을 통해 adversarial training이 모델의 robustness와 generalization을 향상시킴을 입증[^1_1]

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**문제 1: 단일 목적함수의 한계**
기존 모델들은 step-level의 정확한 예측값만 추구하여 실제 시계열 데이터의 stochasticity를 모델링하지 못합니다. 이는 유연성 부족으로 이어집니다.[^1_1]

**문제 2: 오류 누적**
Auto-regressive 모델은 훈련 시 ground-truth를 사용하지만, 추론 시에는 이전 예측값을 사용하여 불일치가 발생하고, 이는 장기 예측에서 오류가 누적됩니다.[^1_1]

**문제 3: Dense Attention의 비효율성**
Vanilla Transformer의 softmax attention은 모든 시점에 0이 아닌 가중치를 할당하여, 관련 없는 시점에도 주의를 분산시켜 성능을 저하시킵니다.[^1_1]

### 2.2 제안하는 방법 (수식 포함)

**Sparse Attention Mechanism:**
기존 softmax를 α-entmax로 대체합니다:

$\alpha\text{-entmax}(\mathbf{h}) = \left[(\alpha - 1)\mathbf{h} - \tau\mathbf{1}\right]_+^{1/(\alpha-1)}$

여기서 $[\cdot]_+$는 ReLU 함수이고, $\tau$는 라그랑주 승수입니다. $\alpha = 1.5$를 사용하여 sparse하면서도 적절한 attention을 제공합니다.[^1_1]

**Attention Computation:**
각 attention head의 출력은:

$\mathbf{O}_m = \alpha_m \mathbf{V}_m = \alpha\text{-entmax}\left(\frac{\mathbf{Q}_m\mathbf{K}_m^T}{\sqrt{d_k}}\right)\mathbf{V}_m$

여기서 $\mathbf{Q}_m = \mathbf{h}\mathbf{W}_m^Q$, $\mathbf{K}_m = \mathbf{h}\mathbf{W}_m^K$, $\mathbf{V}_m = \mathbf{h}\mathbf{W}_m^V$입니다.[^1_1]

**Adversarial Training Framework:**
Generator $G$와 Discriminator $D$ 간의 min-max 최적화:

$\arg\min_G \max_D \lambda \mathcal{L}\_{adv}(\Theta_G, \Theta_D) + \mathcal{L}_\rho(\Theta_G)$

여기서 adversarial loss는:

```math
\mathcal{L}_{adv}(\Theta_G, \Theta_D) = \mathbb{E}[\log(D(\mathbf{Y}_{real}))] + \mathbb{E}[\log(1 - D(\mathbf{Y}_{fake}))]
```

Quantile loss는:

$\mathcal{L}\_\rho(\Theta_G) = \frac{2}{S}\sum_{i=0}\sum_{t=t_0+1}^{t_0+\tau} P_\rho(y_{i,t}, \hat{y}_{i,t})$

```math
P_\rho(y_{i,t}, \hat{y}_{i,t}) = \Delta y_{i,t}(\rho \mathbb{I}_{\hat{y}_{i,t} > y_{i,t}} - (1-\rho)\mathbb{I}_{\hat{y}_{i,t} \leq y_{i,t}})
```

여기서 $\Delta y_{i,t} = \hat{y}\_{i,t} - y_{i,t}$이고, $\mathbf{Y}\_{fake} = \mathbf{Y}\_{1:t_0} \circ \hat{\mathbf{Y}}\_{t_0+1:t_0+\tau}$, $\mathbf{Y}\_{real} = \mathbf{Y}_{1:t_0+\tau}$입니다.[^1_1]

### 2.3 모델 구조

**전체 아키텍처:**

1. **Encoder:** Sparse self-attention layers (N개)와 feed-forward networks로 구성[^1_1]
2. **Decoder:** Sparse self-attention, encoder-decoder attention, feed-forward networks로 구성[^1_1]
3. **Discriminator:** 3개의 fully connected layers와 LeakyReLU activation으로 구성[^1_1]

**입력 형식:**

- Conditioning range: $\mathbf{Y}_{1:t_0}$ (과거 관측값)
- Covariates: $\mathbf{X}_{1:t_0+\tau}$ (시간 의존적/독립적 특징)
- Prediction range: $\hat{\mathbf{Y}}_{t_0+1:t_0+\tau}$ (예측 구간)[^1_1]


### 2.4 성능 향상

**단기 예측 (24시간):**

- Electricity dataset: Q50에서 ConvTrans 대비 28.8% 개선 (0.059 → 0.042)[^1_1]
- Traffic dataset: Q50에서 23.8% 개선 (0.122 → 0.093)[^1_1]

**장기 예측 (7일):**

- Electricity dataset: Q50에서 18.6% 개선 (0.070 → 0.057)[^1_1]
- Traffic dataset: Q50에서 10.1% 개선 (0.139 → 0.125)[^1_1]

**전반적 성과:**

- 평균적으로 ConvTrans 대비 Q50에서 26.4%, Q90에서 15.6% 개선[^1_1]
- 모든 benchmark 데이터셋에서 state-of-the-art 달성[^1_1]


### 2.5 한계

논문에서 명시적으로 언급된 한계는 다음과 같습니다:

1. **GAN 훈련의 불안정성:** Adversarial training의 일반적인 문제인 훈련 불안정성 가능성 (논문에서는 명시하지 않았지만 GAN의 고질적 문제)
2. **하이퍼파라미터 민감도:** $\alpha$ 값과 $\lambda$ (trade-off parameter) 선택이 성능에 영향[^1_1]
3. **계산 비용:** Discriminator 추가로 인한 훈련 시간 증가
4. **Non-autoregressive 구조의 부재:** 여전히 auto-regressive decoder를 사용하여 일부 오류 누적 가능성 존재

## 3. 일반화 성능 향상 가능성

### 3.1 Adversarial Training의 역할

**Sequence-level Regularization:**
Discriminator는 전체 시퀀스를 평가하여 global perspective에서 모델을 정규화합니다. 이는 다음과 같은 효과를 제공합니다:[^1_1]

1. **분포 매칭:** Generator가 생성한 시퀀스 분포가 실제 데이터 분포와 유사하도록 강제[^1_1]
2. **강건성 향상:** Step-level loss만으로는 학습하기 어려운 패턴을 adversarial loss가 보완[^1_1]
3. **오류 누적 완화:** Sequence 전체를 평가하여 누적 오류에 대한 내성 증가[^1_1]

**실험적 증거:**

- DeepAR에 adversarial training 적용 시 electricity dataset에서 Q50이 0.075→0.067로 개선[^1_1]
- 다양한 granularity의 데이터셋(wind, solar, M4-Hourly)에서 일관된 성능 향상[^1_1]


### 3.2 Sparse Attention의 효과

**Attention Weight Density 분석:**
논문의 Figure 4는 α-entmax가 softmax 대비 더 sparse한 attention map을 생성함을 보여줍니다. 이는:[^1_1]

1. **관련 정보에 집중:** 중요한 과거 시점에만 높은 가중치 할당
2. **과적합 감소:** 불필요한 시점에 대한 의존성 제거
3. **해석가능성 향상:** 어떤 과거 시점이 예측에 중요한지 명확히 식별

**α 값에 따른 성능:**

- $\alpha = 1.0$ (softmax): 모든 시점에 분산된 attention
- $\alpha = 1.5$: 최적 성능 (electricity와 traffic 모두에서)[^1_1]
- $\alpha = 2.0$ (sparsemax): 과도하게 sparse하여 성능 저하[^1_1]


### 3.3 일반화 메커니즘

**다양한 데이터셋에서의 성능:**
5개의 서로 다른 특성을 가진 데이터셋에서 일관된 성능 향상은 모델의 높은 일반화 능력을 시사합니다:

- Electricity (370 변수, hourly)[^1_1]
- Traffic (963 변수, hourly)[^1_1]
- Wind (28 변수, daily)[^1_1]
- Solar (137 변수, hourly)[^1_1]
- M4-Hourly (414 시계열)[^1_1]


## 4. 향후 연구에 미치는 영향과 고려사항

### 4.1 학술적 영향

**새로운 연구 방향 제시:**

1. **GAN을 시계열 예측에 적용:** 이전에는 주로 생성 태스크(timeGAN 등)에 사용되었으나, 예측 태스크에도 효과적임을 증명[^1_1]
2. **Sparse attention의 중요성:** 시계열의 sparse dependency 특성을 명시적으로 모델링[^1_1]
3. **Multi-objective learning:** Quantile loss와 adversarial loss의 조합이 상호 보완적임을 입증[^1_1]

### 4.2 후속 연구 시 고려사항

**1. 훈련 안정성:**

- GAN 훈련의 mode collapse 방지 전략 필요
- Generator와 discriminator의 균형 유지 중요
- 적절한 learning rate scheduling 필요

**2. 하이퍼파라미터 튜닝:**

- $\alpha$ 값: 데이터셋 특성에 따라 최적값이 다를 수 있음
- $\lambda$: Adversarial loss의 가중치 조정 필요
- Discriminator update frequency ($k$ steps)[^1_1]

**3. 확장 가능성:**

- Non-autoregressive 구조로의 확장
- Multi-horizon prediction 개선
- Probabilistic forecasting 강화

**4. 해석가능성:**

- Sparse attention map의 시각화 및 분석
- Discriminator의 판별 기준 이해
- Feature importance 분석


## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 Non-Autoregressive 접근법

**NAST (2021):**
Non-Autoregressive Spatial-Temporal Transformer는 AST의 오류 누적 문제를 더욱 근본적으로 해결하고자 제안되었습니다. NAST는:[^1_2]

- 완전히 non-autoregressive 구조로 모든 시점을 동시에 예측[^1_2]
- Spatial-temporal attention mechanism으로 시공간 의존성을 통합적으로 처리[^1_2]
- AST 대비 추론 속도 향상, 하지만 positional information 처리에 한계[^1_2]


### 5.2 분해 기반 Transformer

**Autoformer (2021):**
Auto-Correlation mechanism을 도입하여 AST와는 다른 방향에서 문제를 해결합니다:[^1_3]

- Series decomposition을 모델 내부 블록으로 통합[^1_3]
- Periodicity 기반의 auto-correlation으로 sub-series level에서 의존성 발견[^1_3]
- 6개 benchmark에서 38% 상대적 개선으로 AST를 능가[^1_3]
- **AST와의 차이점:** Sparse attention 대신 주기성 기반 접근, adversarial training 미사용

**ETSformer (2022):**
Exponential smoothing 원리를 Transformer에 통합:[^1_4]

- Exponential Smoothing Attention(ESA)으로 softmax 대체[^1_4]
- Frequency attention으로 효율성과 정확도 개선[^1_4]
- AST의 sparse attention과 유사한 목표이나 방법론은 상이


### 5.3 효율성 개선 연구

**PatchTST \& iTransformer (2023-2024):**

- Patch-based 처리로 입력/출력 정보 bottleneck 해결[^1_5]
- AST의 sparse attention 대신 patching으로 효율성 달성
- Transformer 기반 모델 중 최근 state-of-the-art 달성[^1_6]

**F-Net (2026):**
Fourier-driven representation learning으로 attention 완전히 대체:[^1_7]

- 파라미터 수를 $10^3$ 수준으로 극도로 축소[^1_7]
- AST 대비 1000배 이상 경량화하면서도 경쟁력 있는 성능[^1_7]
- 28.18% 평균 개선으로 최신 benchmark 갱신[^1_7]


### 5.4 Adversarial Learning의 발전

**Adversarial Learning for Irregular Time-Series (2024):**
AST의 adversarial training 아이디어를 irregular time-series로 확장:[^1_8][^1_9]

- Global distribution과 transition dynamics의 균형 강조[^1_9][^1_8]
- Adversarial component의 설계가 시계열 특성에 맞아야 함을 강조[^1_9]
- AST는 regular time-series에 초점, 이 연구는 irregular data로 확장


### 5.5 Sparse Attention의 진화

**Dozer Attention (2024):**
AST의 sparse attention을 더욱 발전시킨 형태:[^1_10]

- Local, Stride, Vary 세 가지 sparse component 도입[^1_10]
- Forecasting horizon에 따라 동적으로 과거 시점 활용[^1_10]
- AST의 고정된 α-entmax 대신 adaptive한 접근

**SEAT (2024):**
Frequency domain sparsification으로 attention 개선:[^1_11]

- 시계열을 주파수 영역으로 변환하여 inherent sparsity 유도[^1_11]
- Model-agnostic하고 plug-and-play 가능[^1_11]
- AST의 α-entmax와 complementary한 접근

**PeriodNet (2025):**
Period attention mechanism으로 temporal redundancy 감소:[^1_12]

- 인접 period 간 유사성 query로 sparse attention 구현[^1_12]
- AST의 시점 단위 sparsity 대신 period 단위 접근[^1_12]


### 5.6 종합 비교표

| 모델 | 연도 | 핵심 기법 | AST 대비 장점 | AST 대비 단점 |
| :-- | :-- | :-- | :-- | :-- |
| AST | 2020 | Sparse attention + Adversarial training | - | Autoregressive 구조 |
| NAST | 2021 | Non-autoregressive | 추론 속도 빠름 | Positional info 처리 약함 |
| Autoformer | 2021 | Auto-correlation + Decomposition | 38% 성능 개선 | Adversarial 미사용 |
| ETSformer | 2022 | Exponential smoothing attention | 해석가능성 높음 | Sparse attention 미사용 |
| PatchTST | 2023 | Patch-based processing | 효율적인 정보 처리 | Adversarial 미사용 |
| Dozer Attention | 2024 | Adaptive sparse attention | Horizon-adaptive | 복잡도 증가 |
| F-Net | 2026 | Fourier-driven learning | 1000배 경량화 | Sparse structure 손실 |

### 5.7 연구 트렌드 분석

**2020-2022: 기초 확립 단계**

- AST를 포함한 초기 Transformer 기반 모델들이 등장
- Sparse attention과 adversarial training의 효과 입증

**2023-2024: 효율성 및 특화 단계**

- Patch-based, frequency-based 등 다양한 효율화 기법 등장
- Irregular time-series, domain-specific 응용 연구 증가[^1_13][^1_14]

**2025-2026: 최적화 및 통합 단계**

- 극도로 경량화된 모델(F-Net) 등장[^1_7]
- Multi-patch, multi-scale 등 복합적 접근[^1_15]
- Adversarial learning의 심화 연구[^1_16]


### 5.8 AST의 위상과 영향력

**지속적 영향:**

1. **Sparse attention의 표준화:** 이후 많은 연구가 sparse attention 개념 채택
2. **Adversarial training 도입:** 시계열 예측에 GAN 적용의 선구적 연구
3. **Sequence-level optimization:** Step-level을 넘어선 전역 최적화 중요성 강조

**한계 극복 방향:**

- Non-autoregressive 구조로의 발전 (NAST 등)
- 더 효율적인 attention mechanism (Dozer, SEAT 등)
- Domain-specific adaptation (Aliformer, ExoTST 등)


## 결론

Adversarial Sparse Transformer는 2020년 당시 시계열 예측 분야에 두 가지 중요한 혁신을 가져왔습니다: (1) Sparse attention을 통한 효율적이고 정확한 temporal dependency 모델링, (2) Adversarial training을 통한 sequence-level 정규화와 일반화 성능 향상. 이후 연구들은 AST의 핵심 아이디어를 계승하면서도 non-autoregressive 구조, decomposition, patch-based processing 등 다양한 방향으로 발전했습니다. 특히 2024-2026년의 최신 연구들은 adaptive sparse attention, frequency-domain sparsification, 그리고 극도의 경량화로 AST의 한계를 극복하고 있습니다. 향후 연구는 AST의 강건성과 최신 기법들의 효율성을 결합하는 방향으로 진행될 것으로 전망됩니다.[^1_5][^1_10][^1_11][^1_2][^1_3][^1_7][^1_1]
<span style="display:none">[^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39]</span>

<div align="center">⁂</div>

[^1_1]: NeurIPS-2020-adversarial-sparse-transformer-for-time-series-forecasting-Paper.pdf

[^1_2]: https://www.semanticscholar.org/paper/7fa6c0b5fd534ecf214b634f68a85a60d3b3191f

[^1_3]: https://www.semanticscholar.org/paper/fc46ccb83dc121c33de7ab6bdedab7d970780b2f

[^1_4]: https://arxiv.org/pdf/2202.01381.pdf

[^1_5]: https://arxiv.org/abs/2207.05397

[^1_6]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12453695/

[^1_7]: https://ieeexplore.ieee.org/document/11060930/

[^1_8]: https://arxiv.org/abs/2411.19341

[^1_9]: https://arxiv.org/html/2411.19341

[^1_10]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11237001/

[^1_11]: https://openreview.net/forum?id=5r6zvadRUD

[^1_12]: https://arxiv.org/html/2511.19497v1

[^1_13]: https://www.semanticscholar.org/paper/76574eff9e451ea6eaca5a7d0636889a99c0dcba

[^1_14]: http://arxiv.org/pdf/2410.12184.pdf

[^1_15]: http://arxiv.org/pdf/2503.17658.pdf

[^1_16]: https://arxiv.org/abs/2602.11940

[^1_17]: http://jecei.sru.ac.ir/article_1477.html

[^1_18]: https://linkinghub.elsevier.com/retrieve/pii/S0925231222000571

[^1_19]: https://link.springer.com/10.1007/978-3-031-26422-1_3

[^1_20]: https://iopscience.iop.org/article/10.1088/1742-6596/2026/1/012036

[^1_21]: https://dl.acm.org/doi/10.1145/3459637.3482054

[^1_22]: https://link.springer.com/10.1007/978-3-030-85713-4_11

[^1_23]: https://arxiv.org/pdf/2502.13721.pdf

[^1_24]: http://arxiv.org/pdf/2410.23992.pdf

[^1_25]: https://arxiv.org/pdf/2206.05495.pdf

[^1_26]: https://arxiv.org/pdf/2209.03945.pdf

[^1_27]: https://openaccess.thecvf.com/content/WACV2025/papers/Pegeot_Temporal_Dynamics_in_Visual_Data_Analyzing_the_Impact_of_Time_WACV_2025_paper.pdf

[^1_28]: https://pdfs.semanticscholar.org/a44e/a69a5b6c3ac5819ac9a707853a1b77f99e86.pdf

[^1_29]: https://openaccess.thecvf.com/content/ICCV2023/papers/Zhu_Universal_Domain_Adaptation_via_Compressive_Attention_Matching_ICCV_2023_paper.pdf

[^1_30]: https://arxiv.org/pdf/2411.06272.pdf

[^1_31]: https://arxiv.org/html/2408.07511v2

[^1_32]: https://arxiv.org/pdf/2407.01872.pdf

[^1_33]: https://arxiv.org/html/2411.13264v1

[^1_34]: https://www.sciencedirect.com/science/article/pii/S1574013725001595

[^1_35]: https://openreview.net/forum?id=kHEVCfES4Q\&noteId=mrNbq9EkQa

[^1_36]: https://peerj.com/articles/cs-3001/

[^1_37]: https://github.com/ddz16/TSFpaper

[^1_38]: https://neurips.cc/virtual/2024/100223

[^1_39]: https://icml.cc/virtual/2025/poster/44262

