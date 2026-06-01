# A K-variate Time Series Is Worth K Words: Evolution of the Vanilla Transformer Architecture for Long-term Multivariate Time Series Forecasting

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

기존 다변량 시계열 예측(MTSF) Transformer들이 채택한 **Time Point based Tokenization(TPT)** 전략은 Transformer의 **토큰 균일성 귀납적 편향(token uniformity inductive bias)**을 무시하여 **과도한 평활화(over-smoothing)** 문제를 야기한다. 이를 해결하기 위해 **Time Variable based Tokenization(TVT)** 전략으로의 전환과 Transformer 디코더를 단순 선형 레이어로 대체하는 것만으로도 당시 최고 성능의 Transformer 모델들을 능가할 수 있음을 보인다.

### 주요 기여

1. **이론적·실험적 문제 제기**: 기존 TPT 전략이 Transformer의 토큰 균일성 편향을 무시하여 over-smoothing 예측을 초래함을 이론·실험적으로 입증

2. **새로운 TVT 패러다임 제안**: $K$-변량 시계열을 $K$개의 단어(토큰)로 다루는 새로운 Transformer 패러다임 제시 (디코더 재설계, 임베딩 전략 변경 포함)

3. **성능 우위 실증**: 단순화된 구조(TVT + Linear Decoder)만으로 FEDformer, Autoformer, Informer 등 복잡하게 설계된 SOTA 모델 대부분을 능가

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

#### 문제의 이론적 배경: Token Uniformity Inductive Bias

Dong et al. (2021)에 의하면, skip connection이 없는 순수 self-attention 네트워크(SAN)의 경우 깊이가 깊어질수록 출력 행렬의 rank가 지수적으로 붕괴된다. 이를 수식으로 표현하면:

$$\|\text{RES}(\text{SAN}(\mathbf{X}))\|_{1,\infty} \leq \left(\frac{4\beta H}{\sqrt{d_{qk}}}\right)^{\frac{3L-1}{2}} \|\text{RES}(\mathbf{X})\|_{1,\infty}^{3^L}$$

여기서:
- $\mathbf{X} \in \mathbb{R}^{N_{token} \times D_{model}}$: self-attention 네트워크의 입력
- $\text{RES}(\mathbf{X}) = \mathbf{X} - \mathbf{1}\mathbf{x}^\top$: $\mathbf{X}$의 잔차 (rank-1 행렬 제거)
- $\|\mathbf{X}\|\_{1,\infty} = \sqrt{\|\mathbf{X}\|\_1 \|\mathbf{X}\|\_\infty}$: $L_1, L_\infty$-복합 노름
- $\beta$: 가중치 행렬 $\|\mathbf{W}^l_{QK,h}\|\_1 \|\mathbf{W}^l_h\|_{1,\infty}$의 상한
- $L$: 네트워크 깊이, $H$: 헤드 수, $d_{qk}$: query/key 차원

이 수식은 출력 잔차가 입력 잔차보다 지수적으로 작아짐을, 즉 출력 토큰들이 점점 균일해지는 **rank collapse**를 의미한다.

#### 문제의 실험적 검증: Over-smoothing 확인

TPT 기반 예측 결과 $\mathbf{Y}^t$에 대해 유클리드 거리 기반 유사도 행렬을 정의:

$$\mathbf{E}^t_{tokens} = \begin{pmatrix} \|\mathbf{y}^t_{t+1} - \mathbf{y}^t_{t+1}\|_2 & \cdots & \|\mathbf{y}^t_{t+1} - \mathbf{y}^t_{t+H}\|_2 \\ \|\mathbf{y}^t_{t+2} - \mathbf{y}^t_{t+1}\|_2 & \cdots & \|\mathbf{y}^t_{t+2} - \mathbf{y}^t_{t+H}\|_2 \\ \vdots & \ddots & \vdots \\ \|\mathbf{y}^t_{t+H} - \mathbf{y}^t_{t+1}\|_2 & \cdots & \|\mathbf{y}^t_{t+H} - \mathbf{y}^t_{t+H}\|_2 \end{pmatrix} \in \mathbb{R}^{H \times H}$$

유사도 측정 함수:

$$\text{TokenSim}(\mathbf{Y}^t) = -\text{softmax}(\mathbf{E}^t_{tokens}) \in \mathbb{R}^{H \times H}$$

음수 부호는 값이 클수록(붉은색) 두 토큰 쌍의 유사도가 높음을 의미한다. 실험 결과, TPT 기반 예측의 유사도 맵은 실제 ground truth보다 훨씬 균일하여 over-smoothing이 명확히 나타났다.

---

### 2.2 제안하는 방법 (수식 포함)

#### Step 1: Time Variable Tokenization (TVT)

기존 TPT 전략:

$$\mathbf{X}^t = \mathbf{X}^t_{TPT} = \{\mathbf{x}^t_{t-L+1}, \ldots, \mathbf{x}^t_t \mid \mathbf{x}^t_{t-L+i} \in \mathbb{R}^K\} \in \mathbb{R}^{K \times L}$$

TPT는 같은 시점의 $K$개 변수 값들을 하나의 토큰으로 묶어 $L$개의 토큰을 생성하므로, token uniformity bias가 **시간 차원**에 작용하여 예측이 시간적으로 균일해진다.

**제안하는 TVT 전략**:

$$\mathbf{X}^t = \mathbf{X}^t_{TVT} = \{\mathbf{x}^t_1, \mathbf{x}^t_2, \ldots, \mathbf{x}^t_K \mid \mathbf{x}^t_i \in \mathbb{R}^L\}$$

각 변수의 전체 시간 시퀀스(길이 $L$)를 하나의 토큰으로 취급하여 $K$개의 토큰을 생성한다. 이 경우 token uniformity bias는 **변수 차원**에 작용하여, 오히려 변수 간 상관관계 학습을 강화한다.

#### Step 2: Linear Projection Embedding

학습 가능한 선형 투영 $\mathbf{E}_{pre} \in \mathbb{R}^{L \times D}$를 사용하여:

$$\mathbf{Z}^t_{enc\_in} = \mathbf{X}^t_{TVT} \mathbf{E}_{pre} = \{\mathbf{x}^t_1 \mathbf{E}_{pre}, \mathbf{x}^t_2 \mathbf{E}_{pre}, \ldots, \mathbf{x}^t_K \mathbf{E}_{pre}\}$$

TVT 방식에서는 토큰 간의 "위치" 관계가 변수의 순서(채널 순서)에 의해 고정되므로, 기존의 sinusoidal/cosinusoidal 위치 임베딩이 불필요해진다.

#### Step 3: Transformer 디코더 제거 및 Linear Decoder 도입

TVT 기반 Transformer에서 디코더의 placeholder 토큰은 타임스탬프 정보만 포함하므로, self-attention/cross-attention이 의미 있는 정보를 학습하지 못한다. 따라서 복잡한 디코더를 제거하고 단순 선형 레이어 $\mathbf{E}_{fc} \in \mathbb{R}^{D \times H}$로 대체:

$$\mathbf{Y}^t = \mathbf{Z}^t_{enc\_out} \mathbf{E}_{fc} = \{\mathbf{z}^t_1 \mathbf{E}_{fc}, \mathbf{z}^t_2 \mathbf{E}_{fc}, \ldots, \mathbf{z}^t_K \mathbf{E}_{fc}\}$$

#### Token Uniformity 측정 지표

$$\text{TU}(\mathbf{Y}^t) = \|\text{RES}(\mathbf{Y}^t)\|_{1,\infty} \, / \, \|\mathbf{Y}^t\|_{1,\infty}$$

값이 낮을수록 token uniformity가 높음을 의미한다.

---

### 2.3 모델 구조

```
[입력: X^t ∈ R^{K×L}]
        ↓
[TVT: K개의 변수 토큰으로 분리 (각 토큰 ∈ R^L)]
        ↓
[선형 투영 E_pre ∈ R^{L×D}: K개의 D-차원 임베딩 생성]
        ↓
[Transformer Encoder (N_enc 레이어)]
  → Multi-head Self-Attention (변수 차원에서 작동)
  → Feed-Forward Network
  → Layer Normalization + Residual Connection
        ↓
[Linear Decoder E_fc ∈ R^{D×H}]
        ↓
[출력: Y^t ∈ R^{K×H}]
```

세 가지 아키텍처 비교:

| 모델 | 토크나이제이션 | 디코더 | 임베딩 |
|------|------|------|------|
| Vanilla MTSF Transformer | TPT (시점 기반) | Transformer 디코더 (one-forward) | Conv + PE + SE |
| TVT + Transformer Decoder | TVT (변수 기반) | Transformer 디코더 (one-forward) | 선형 투영 (±SE) |
| **TVT + Linear Decoder (최종)** | **TVT (변수 기반)** | **단순 선형 레이어** | **선형 투영 only** |

---

### 2.4 성능 향상

5개 벤치마크(ETTm2, Electricity, Exchange, Traffic, Weather), 예측 길이 $H \in \{96, 192, 336, 720\}$에서 평가:

| 비교 대상 | MSE 상대 감소 | MAE 상대 감소 |
|------|------|------|
| vs. FEDformer | **17.5%** (ETTm2 제외 시 21.2%) | **11.7%** (ETTm2 제외 시 15.3%) |
| vs. Autoformer | 더 큰 폭의 개선 | 더 큰 폭의 개선 |

**20/20 케이스 중 19개**에서 최저 MSE, **18/20 케이스**에서 최저 MAE 달성

**효율성 측면**:
- 파라미터 수: 약 **0.126M** (FEDformer의 약 5.28~8.35M 대비 극히 작음)
- 추론 시간: 평균 **0.903ms** (TPT one-forward의 2.558ms 대비 약 64.7% 빠름)

---

### 2.5 한계

논문 자체에서 명시한 한계:

1. **시간 차원 연결의 단순성**: 시간 차원에서 서로 다른 시점들을 연결하는 데 MLP 레이어를 사용하는 것은 다소 단순하며, Autoformer의 자기상관(Auto-Correlation) 메커니즘이나 FEDformer의 주파수 강화 방법 등을 통합하면 추가적인 성능 향상 가능성이 있음

2. **단순 선형 디코더의 표현력 한계**: 단순 선형 레이어 디코더를 N-BEATS와 같은 강력한 MLP 기반 MTSF 모델로 교체하면 더 나은 성능을 기대할 수 있음

3. **고정된 입력 윈도우 크기**: 실험에서 $L=96$으로 고정 — 다양한 입력 길이에 대한 일반화 성능 검증이 제한적

4. **데이터셋 다양성 제한**: 전기, 교통, 기상 등 특정 도메인에 집중되어 있어 다른 도메인(금융, 의료 등)에서의 일반화 성능이 충분히 검증되지 않음

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 TVT가 일반화에 기여하는 메커니즘

TVT 전략은 **과적합(over-fitting)을 억제**하는 방식으로 일반화 성능을 향상시킨다. 이는 Token Uniformity 분석에서 명확히 드러난다:

$$\text{TU}(\mathbf{Y}^t) = \|\text{RES}(\mathbf{Y}^t)\|_{1,\infty} / \|\mathbf{Y}^t\|_{1,\infty}$$

**실험적 관찰 (Fig. 5 분석)**:

1. **Ground Truth의 가변성**: 실제 데이터의 $\text{TU}(\mathbf{Y}^t_{gt})$는 학습 세트와 테스트 세트 간에 $10 \sim 20\%$ 차이가 나는데, 이는 시간에 따른 token uniformity의 자연스러운 변동을 의미한다.

2. **TPT의 과적합**: 학습 에포크 증가에 따라 $\text{TU}(\mathbf{Y}^t_{TPT})$가 ground truth보다 낮아지며(과도한 token uniformity 학습), 학습/테스트 세트 간 값이 수렴하여 **데이터 분포 변화에 취약한 일반화 실패**를 보임

3. **TVT의 우수한 일반화**: $\text{TU}(\hat{\mathbf{Y}}^t_{TVT})$는 학습/테스트 세트 간 차이를 유지하며 ground truth 곡선을 더 잘 추종하므로, **도메인 쉬프트나 시간적 분포 변화에 상대적으로 강인**함

### 3.2 일반화 향상의 구체적 메커니즘

**① Inductive Bias의 방향 전환**

TPT에서 token uniformity bias → 시간 차원 균일화 (해로움)  
TVT에서 token uniformity bias → 변수 차원 균일화 (유익: 변수 간 상관관계 학습 강화)

이는 모델이 학습 데이터에 특화된 시간적 패턴을 암기하는 대신, **변수 간의 일반적인 상관 구조를 학습**하도록 유도하여 새로운 데이터에 대한 일반화를 강화한다.

**② 과도한 모델 복잡도 억제**

복잡한 Transformer 디코더를 제거하고 단순 선형 레이어를 사용함으로써:
- 파라미터 수: ~0.126M (기존 대비 최대 90% 감소)
- **과적합 위험 감소** (Occam's Razor 원칙 적용)

**③ 선형 투영의 정규화 효과**

학습 가능한 $\mathbf{E}_{pre} \in \mathbb{R}^{L \times D}$는 데이터 $\ell_2$ 정규화 효과를 내재하며, 시계열의 전역적 통계 특성(평균, 분산 등)을 자연스럽게 인코딩한다.

### 3.3 일반화 성능의 잠재적 개선 방향

| 방향 | 구체적 방법 | 기대 효과 |
|------|------|------|
| **채널 믹싱 강화** | Cross-variable attention 정교화 | 변수 간 비선형 의존성 포착 |
| **시간 표현 개선** | Auto-correlation, 주파수 도메인 특징 통합 | 시간적 주기성 더 정확히 학습 |
| **사전 학습 활용** | 대규모 시계열 사전학습 + TVT 파인튜닝 | 도메인 간 전이 학습 용이 |
| **정규화 강화** | Dropout, LayerNorm 튜닝 | 작은 데이터셋 과적합 방지 |
| **채널 독립성** | 채널 독립 예측과 TVT의 결합 | 변수 수에 무관한 일반화 |

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

| 모델 | 연도 | 핵심 아이디어 | 토크나이제이션 | 복잡도 | TVT 논문과의 관계 |
|------|------|------|------|------|------|
| **Informer** | 2021 | ProbSparse Attention, one-forward 디코더 | TPT | $O(L \log L)$ | TVT가 성능 능가 |
| **Autoformer** | 2021 | Auto-Correlation, 시계열 분해 | TPT | $O(L \log L)$ | TVT가 성능 능가; 자기상관은 TVT에 통합 가능 |
| **FEDformer** | 2022 | 주파수 도메인 self-attention | TPT | $O(L)$ | TVT가 대부분의 경우 능가 |
| **Pyraformer** | 2022 | 피라미드 계층적 attention | TPT | $O(L)$ | TVT가 큰 폭으로 능가 |
| **PatchTST** | 2023 | 패치(patch) 단위 토크나이제이션 + 채널 독립성 | Patch 기반 (채널 독립) | $O(L/P)$ | TVT와 유사 방향; 채널 독립 vs. 채널 의존 |
| **iTransformer** | 2024 | 변수 전체를 토큰으로 (TVT와 유사), 역전된 attention | TVT 유사 | $O(K^2)$ | TVT 아이디어를 발전·재발견 |
| **DLinear** | 2023 | 단순 선형 분해 모델 | - | $O(1)$ | TVT의 "단순성" 철학과 일치 |
| **TimesNet** | 2023 | 1D→2D 변환으로 시간적 패턴 추출 | 2D Patch | $O(L \log L)$ | 다차원 표현으로 TVT 개념 확장 |
| **Crossformer** | 2023 | 시간 및 변수 차원 이중 attention | Patch + Cross-dim | $O(L/P \cdot K)$ | TVT의 변수 간 관계 학습 아이디어와 유사 |

### 주요 비교 논점

**TVT vs. PatchTST (2023)**:
- PatchTST는 시계열을 패치(patch) 단위로 분할하여 국소적 시간 패턴 포착 + 채널 독립 예측
- TVT는 변수 전체를 하나의 토큰으로 처리하여 변수 간 의존성에 집중
- 두 접근법 모두 TPT의 over-smoothing을 다른 방식으로 회피하며, 상호 보완적

**TVT vs. iTransformer (2024)**:
- iTransformer는 TVT와 거의 동일한 아이디어(변수 단위 토크나이제이션)를 독자적으로 발전시켜 더 체계적인 분석과 더 다양한 실험을 제시
- TVT 논문(2022)이 선도적으로 이 방향을 제시했다는 점에서 선구적 의미가 있음

**TVT vs. DLinear (2023)**:
- Are Transformers Effective for Time Series Forecasting? (Zeng et al., 2023)에서 DLinear가 단순 선형 모델로도 많은 Transformer를 능가함을 보임
- TVT의 "단순 선형 디코더"와 맥을 같이 하나, TVT는 self-attention의 변수 간 관계 학습을 여전히 중요하게 활용

---

## 5. 향후 연구에 미치는 영향 및 고려사항

### 5.1 미치는 영향

**① 시계열 Transformer 설계의 패러다임 전환**

이 논문은 "무엇을 토큰으로 볼 것인가"라는 근본적 질문을 제기함으로써, 이후 연구들이 토크나이제이션 전략을 핵심 설계 요소로 재검토하는 흐름을 이끌었다. iTransformer(2024), PatchTST(2023) 등이 이 방향을 계승·발전시켰다.

**② "단순한 것이 강하다" 패러다임 강화**

복잡한 Transformer 구조보다 올바른 귀납적 편향을 갖는 단순한 구조가 더 효과적일 수 있음을 보여줌으로써, DLinear, NLinear 등 단순 모델의 재조명에 기여했다.

**③ Over-smoothing 문제 의식 확산**

Transformer의 token uniformity bias로 인한 over-smoothing을 시계열 예측의 주요 실패 원인으로 식별함으로써, 이후 연구들이 예측 다양성(diversity) 유지를 명시적 목표로 삼게 되는 계기 제공

**④ 채널 믹싱 방식에 대한 새로운 관점**

기존에는 FFN이 채널 믹싱을 담당했으나, self-attention을 채널 차원에 적용하는 것이 더 효과적임을 실증함으로써 채널 의존성 모델링의 새로운 방향 제시

### 5.2 향후 연구 시 고려할 점

**① 채널 독립성(Channel Independence) vs. 채널 의존성(Channel Dependence) 트레이드오프**

TVT는 변수 간 의존성을 명시적으로 모델링하나, 데이터셋에 따라서는 채널 독립 가정이 더 효과적일 수 있다(PatchTST). 이 두 접근법의 적합성을 데이터 특성(변수 간 상관관계 강도)에 따라 적응적으로 선택하는 메커니즘 연구가 필요하다.

**② 가변적 입력 길이 대응**

현재 TVT의 선형 투영 $\mathbf{E}_{pre} \in \mathbb{R}^{L \times D}$는 고정된 입력 길이 $L$을 가정한다. 가변 길이 입력이나 결측치가 있는 실제 환경에 적용하기 위한 일반화 방안(예: 적응형 위치 인코딩, 마스킹 전략)을 연구해야 한다.

**③ 비정상성(Non-stationarity) 처리**

TVT는 시계열의 정상성을 암묵적으로 가정하나, 실제 시계열은 분포 이동(distribution shift)이 빈번하다. Non-stationary Transformer나 Reversible Instance Normalization(RevIN) 등의 정규화 기법과의 결합이 중요한 연구 과제이다.

**④ 계산 복잡도와 변수 수의 관계**

TVT의 self-attention 복잡도는 변수 수 $K$에 대해 $O(K^2)$이다. 수백~수천 개의 변수를 가진 대규모 다변량 시계열에서 확장성 확보를 위한 희소 attention이나 linear attention 도입이 필요하다.

**⑤ 사전 학습(Pre-training) 및 전이 학습**

TVT의 변수 기반 토크나이제이션은 대규모 시계열 사전학습 모델(예: Time-Series Foundation Models)에서 도메인 간 전이를 용이하게 할 잠재력이 있다. TVT + 대규모 사전학습의 조합 연구가 기대된다.

**⑥ 해석 가능성(Interpretability) 강화**

TVT에서의 self-attention 맵은 변수 간 상관관계를 반영하므로, 이를 명시적으로 시각화하고 해석하는 연구가 도메인 전문가와의 협력에 유용할 것이다.

**⑦ 주파수 도메인 특징과의 결합**

저자들이 직접 언급했듯, FEDformer의 주파수 강화나 Autoformer의 자기상관 메커니즘을 TVT 인코더 내에 통합하면 시간적 패턴 학습을 더욱 강화할 수 있다.

---

## 참고 자료

- **본 논문**: Zhou, Z., Zhong, R., Yang, C., Wang, Y., Yang, X., & Shen, W. (2022). "A K-variate Time Series Is Worth K Words: Evolution of the Vanilla Transformer Architecture for Long-term Multivariate Time Series Forecasting." arXiv:2212.02789
- Vaswani, A., et al. (2017). "Attention is all you need." NeurIPS.
- Dong, Y., Cordonnier, J.-B., & Loukas, A. (2021). "Attention is not all you need: Pure attention loses rank doubly exponentially with depth." ICML.
- Zhou, H., et al. (2021). "Informer: Beyond efficient transformer for long sequence time-series forecasting." AAAI.
- Wu, H., et al. (2021). "Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting." NeurIPS.
- Zhou, T., et al. (2022). "FEDformer: Frequency enhanced decomposed transformer for long-term series forecasting." arXiv:2201.12740.
- Liu, S., et al. (2021). "Pyraformer: Low-complexity pyramidal attention for long-range time series modeling and forecasting." ICLR.
- Nie, Y., et al. (2023). "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers." (PatchTST) ICLR.
- Liu, Y., et al. (2024). "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting." ICLR.
- Zeng, A., et al. (2023). "Are Transformers Effective for Time Series Forecasting?" (DLinear) AAAI.
- Lai, G., et al. (2018). "Modeling long-and short-term temporal patterns with deep neural networks." (LSTNet) ACM SIGIR.
- Oreshkin, B. N., et al. (2019). "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting." arXiv:1905.10437.
- Tolstikhin, I. O., et al. (2021). "MLP-Mixer: An all-MLP Architecture for Vision." NeurIPS.
