
# Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting
## 1. 핵심 주장과 주요 기여 (간결 요약)

이 논문이 주장하는 바를 한 줄로 요약하면 다음과 같다.

> “정적/과거/미래 공변량이 뒤섞인 복잡한 멀티-호라이즌 시계열에서, **고성능 예측과 해석 가능성**을 동시에 달성하는 전용 딥러닝 아키텍처(TFT)를 제안한다.”[^1_1][^1_2][^1_3][^1_4]

구체적인 주요 기여는 다음 네 가지다.

1. **문제 정의 및 세팅**
정적(static) 공변량, 과거에만 관측되는 시변 공변량(past-observed), 미래까지 미리 아는 시변 공변량(known-future)을 모두 포함하는 **일반적인 multi-horizon forecasting 세팅을 명시적으로 정의**하고, 이를 기준으로 모델과 실험을 설계한다.[^1_1]
2. **TFT 아키텍처 제안**[^1_2][^1_3][^1_4][^1_1]
    - LSTM 기반 **sequence-to-sequence 레이어**로 **국소(local) 패턴** 처리
    - **Interpretable multi-head self-attention**으로 **장기(long-term) 의존성** 학습
    - **Gated Residual Network(GRN) + GLU**로 불필요한 비선형 처리를 억제하는 **component gating**
    - **Instance-wise variable selection network**로 샘플별(feature-wise) 중요도 추정
    - **Static covariate encoder**로 정적 메타데이터를 전역 컨텍스트로 주입
    - **Quantile regression**으로 전체 horizon에 대한 예측 구간(불확실성)을 함께 제공
3. **성능 향상**
전력(Electricity), 교통(Traffic), 리테일(Retail), 금융 변동성(Volatility) 등 실제 대규모 데이터셋에서, ARIMA/ETS/TRMF, DeepAR/DSSM, ConvTrans, MQRNN 등 다양한 SOTA 베이스라인 대비 **P50/P90 quantile loss 기준 3–26% 개선**을 보인다.[^1_1]
4. **해석 가능성(interpretability) 유스케이스 제시**[^1_1]
아키텍처 내부 구성요소를 활용해
    - (i) 전역적으로 중요한 변수(global variable importance),
    - (ii) 지속적인 시계 패턴(예: seasonality, lag),
    - (iii) 레짐 전환(regime shift)와 중요한 이벤트
를 분석하는 방법을 제안한다.

***

## 2. TFT가 푸는 문제, 수식 기반 방법론, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제: 일반적인 멀티-호라이즌 시계열 예측

데이터셋에는 여러 개체(entity) $i = 1,\dots, I$가 있고, 각 개체마다:

- 정적 공변량:

$s_i \in \mathbb{R}^{m_s} \$

- 시각 $t$에서 시변 공변량:

$\chi_{i,t} \in \mathbb{R}^{m_\chi} \$

- 스칼라 타깃:

$y_{i,t} \in \mathbb{R} \$

시변 공변량은

- 과거에만 관측되는 입력(예: 과거 기상, 과거 수요)

$z_{i,t} \in \mathbb{R}^{m_z} \$

- 미래까지 미리 아는 입력(예: 캘린더, 프로모션 계획)

$x_{i,t} \in \mathbb{R}^{m_x} \$

로 나뉘어

$$
\chi_{i,t} = 
\begin{bmatrix}
z_{i,t} \\
x_{i,t}
\end{bmatrix}
\in \mathbb{R}^{m_z + m_x}
$$

가 된다.[^1_1]

**Multi-horizon quantile forecasting** 문제는, 시간 $t$에서 과거 $k$ 시점까지와 미래 $\tau_{\max}$ 시점까지의 known-future 입력을 바탕으로, 여러 quantile $q$에 대한 $\tau = 1,\dots,\tau_{\max}$ step ahead 예측을 동시에 출력하는 것:

$$
\hat y_{i}^{(q)}(t,\tau)
= f_q\bigl(
\tau,\; y_{i,t-k:t},\; z_{i,t-k:t},\; x_{i,t-k:t+\tau},\; s_i
\bigr),
\quad \tau = 1,\dots,\tau_{\max}
$$

여기서 $f_q(\cdot)$가 바로 TFT 모델이다.[^1_1]

이 세팅의 난점은 다음과 같다.

- 정적/시변, 과거/미래 공변량이 **이질적**이고 상호작용 구조를 알기 어렵다.
- **하나의 글로벌 모델**로 다수의 entity를 학습해야 일반화가 가능한데, 이는 용량 관리와 정규화가 어렵다.
- 기존 딥러닝 모델은 “블랙박스”라 **변수 중요도, 시계 패턴, 레짐 전환**을 해석하기 어렵다.

TFT는 이 문제에 특화된 inductive bias(정적 인코더, 변수 선택, 시계 self-attention, gating 등)를 설계해 **성능+해석 가능성**을 동시에 노린다.[^1_4][^1_2][^1_1]

***

### 2.2 제안 방법: 수식 중심 구성요소 설명

#### 2.2.1 Gated Residual Network (GRN)와 GLU

GRN은 기본 입력 $a$와 선택적 컨텍스트 $c$를 받아, 필요할 때만 비선형 변환을 적용하고, 필요 없으면 거의 선형/아이덴티티로 동작하도록 설계된 잔차 블록이다.[^1_1]

$$
\mathrm{GRN}_\omega(a, c)
= \mathrm{LayerNorm}\bigl(a + \mathrm{GLU}_\omega(\eta_1)\bigr)
$$

$$
\eta_1 = W_{1,\omega}\,\eta_2 + b_{1,\omega}
$$

$$
\eta_2 = \mathrm{ELU}\bigl( W_{2,\omega} a + W_{3,\omega} c + b_{2,\omega} \bigr)
$$

여기서 $\omega$는 파라미터 셰어링 인덱스, $\mathrm{ELU}$는 exponential linear unit, LayerNorm은 레이어 정규화다.[^1_1]

GLU(gated linear unit)는 특정 블록의 출력을 얼마나 통과시킬지 제어하는 gating 레이어다.

$$
\mathrm{GLU}_\omega(\gamma) 
= \sigma(W_{4,\omega} \gamma + b_{4,\omega}) 
\odot (W_{5,\omega} \gamma + b_{5,\omega})
$$

$\sigma$는 sigmoid, $\odot$는 Hadamard 곱이다. 게이트가 거의 0이면 비선형 블록을 사실상 “스킵”하는 효과가 있다.[^1_1]

**해석**: GRN + GLU는 “필요한 곳에만 비선형성/깊이”를 쓰는 **적응형 용량 제어(adaptive capacity control)** 역할을 하며, 이는 특히 작은/노이즈 많은 데이터셋(Volatility)에서 **과적합을 줄이는 방향**으로 일반화에 기여한다는 것이 ablation으로 확인된다.[^1_1]

#### 2.2.2 변수 선택 네트워크 (Variable Selection Network)

각 시점 $t$에서 $m_\chi$개의 시변 변수(또는 정적 변수)에 대해 **샘플별 가중치**를 산출하고, 그 가중치로 비선형 변환된 특징을 가중합한다.[^1_1]

각 변수 $j$의 임베딩(또는 선형 변환) 출력 $\xi_t^{(j)} \in \mathbb{R}^{d_{\text{model}}}$을 모아

$$
\Xi_t = 
\begin{bmatrix}
(\xi_t^{(1)})^\top, \dots, (\xi_t^{(m_\chi)})^\top
\end{bmatrix}^\top
$$

정적 컨텍스트 $c_s$와 함께 GRN에 통과시켜 소프트맥스로 정규화:

$$
v_t^\chi = \mathrm{Softmax}\bigl( \mathrm{GRN}_{v_\chi}(\Xi_t, c_s) \bigr)
\in \mathbb{R}^{m_\chi}
$$

각 변수는 자신의 GRN으로 비선형 변환된다.

$$
\tilde\xi_t^{(j)} = \mathrm{GRN}_{\tilde \xi^{(j)}}(\xi_t^{(j)})
$$

최종적으로 시점 $t$의 통합 표현은

$$
\tilde\xi_t = \sum_{j=1}^{m_\chi} v_t^{(j)} \tilde\xi_t^{(j)}
$$

이 된다.[^1_1]

**해석**: 이 구조는

- (i) **샘플별(instance-wise) 변수 중요도**를 직접 제공하고,
- (ii) 예측에 덜 중요한 변수는 학습 초기에 자연스럽게 작은 가중치를 받으므로 **노이즈 특성을 억제**해 일반화에 도움을 준다.
Retail/Electricity 등에서 중요한 변수만 높은 가중치를 받는 것이 실증적으로 확인된다.[^1_1]


#### 2.2.3 Static Covariate Encoder

정적 공변량 $s$는 하나의 벡터로 끝나는 것이 아니라, 여러 GRN을 통해 네 가지 컨텍스트 벡터로 변환된다.[^1_1]

- $c_s$: 시변 변수 선택의 컨텍스트
- $c_c, c_h$: LSTM 인코더/디코더 초기 cell/hidden state 초기화용
- $c_e$: static enrichment 레이어에서 시변 표현과 결합

예:

$$
c_s = \mathrm{GRN}_{c_s}(\zeta), \quad 
\zeta = \text{static variable selection output}
$$

이를 통해 하나의 글로벌 모델이 개체별 특성을 잘 반영하면서도, **정적 특성에 따른 시계 패턴의 차이**를 자연스럽게 학습한다.[^1_1]

#### 2.2.4 Interpretable Multi-Head Self-Attention

기본 self-attention은 다음과 같다.[^1_5]

$$
\mathrm{Attention}(Q,K,V)
= \mathrm{Softmax}\left(\frac{QK^\top}{\sqrt{d_{\text{attn}}}}\right)V
$$

Multi-head의 일반형은 head마다 $Q,K,V$를 달리 쓰는데, 이 경우 각 head의 attention weight 자체가 **변수 중요도**로 해석되기 어렵다.

TFT는 해석 가능성을 위해

- **value projection $W_V$를 head 간 공유**하고,
- head 출력을 **평균(additive) 집계**

하여, 실질적으로 하나의 통합 attention 행렬 $\tilde A(Q,K)$에 기반한 구조를 만든다:[^1_1]

$$
\tilde H = \frac{1}{m_H}\sum_{h=1}^{m_H}
\mathrm{Attention}(QW_Q^{(h)}, KW_K^{(h)}, VW_V)
$$

$$
\mathrm{InterpretableMultiHead}(Q,K,V)
= \tilde H W_H
$$

즉, head마다 **다른 시간 패턴**을 보지만, 모두 동일한 value 집합을 쓰므로, 통합 attention $\tilde A$를 사용해 “어느 시점이 어느 시점에 얼마나 기여했는지”를 비교적 안정적으로 해석할 수 있다.[^1_1]

#### 2.2.5 Temporal Fusion Decoder

TFT의 핵심 “fusion”은 디코더에서 일어난다.[^1_1]

1. **Sequence-to-sequence 레이어 (locality enhancement)**
LSTM encoder–decoder로 과거 $\tilde\xi_{t-k:t}$, 미래 $\tilde\xi_{t+1:t+\tau_{\max}}$를 처리해 위치 인덱스 $n \in [-k,\tau_{\max}]$별 시계 특징 $\phi(t,n)$를 얻는다. 이후 gated residual:

$$
\tilde\phi(t,n) = \mathrm{LayerNorm}\bigl(
\tilde\xi_{t+n} + \mathrm{GLU}_{\tilde\phi}(\phi(t,n))
\bigr)
$$

이 레이어는 convolutional positional encoding 대신 **시계열 전용 위치 인코딩 역할**을 하며, local 패턴(이상치, 변화점 등) 학습에 특화돼 있음이 ablation으로 확인된다.[^1_1]
2. **Static enrichment**

$$
\theta(t,n) = \mathrm{GRN}_\theta\bigl( \tilde\phi(t,n), c_e \bigr)
$$

정적 특성이 각 시점의 표현에 직접 주입된다.
3. **Temporal self-attention**

모든 시점의 static-enriched 벡터를 묶어
$\Theta(t) = [\theta(t,-k),\dots,\theta(t,\tau_{\max})]^\top$라 하면,

$$
B(t) = \mathrm{InterpretableMultiHead}\bigl(\Theta(t),\Theta(t),\Theta(t)\bigr)
$$

로부터 $\beta(t,n)$를 얻고, gated residual 적용:

$$
\delta(t,n) = \mathrm{LayerNorm}\bigl(
\theta(t,n) + \mathrm{GLU}_\delta(\beta(t,n))
\bigr)
$$

Decoder masking을 사용해 인과성을 보장한다.[^1_1]
4. **Position-wise feed-forward + transformer 블록 스킵**

$$
\psi(t,n) = \mathrm{GRN}_\psi(\delta(t,n))
$$

$$
\tilde\psi(t,n) = \mathrm{LayerNorm}\bigl(
\tilde\phi(t,n) + \mathrm{GLU}_{\tilde\psi}(\psi(t,n))
\bigr)
$$

이로써 **“필요할 때만 transformer 블록을 사용하고, 그렇지 않으면 LSTM seq2seq 결과를 그대로 쓰는”** 얕은 경로를 확보한다.

#### 2.2.6 Quantile 출력과 학습 손실

각 horizon $\tau$, quantile $q$에 대해 선형 레이어로 출력을 생성한다.

$$
\hat y^{(q)}(t,\tau) = W_q \tilde\psi(t,\tau) + b_q
$$

Quantile loss는

$$
QL(y,\hat y,q)
= q(y-\hat y)_+ + (1-q)(\hat y - y)_+
$$

여기서 $(\cdot)_+ = \max(0,\cdot)$이다. 전체 학습 손실은 학습 샘플 집합 $\Omega$와 quantile 집합 $Q$에 대해

$$
\mathcal{L}(\Theta)
= \sum_{t \in \Omega} \sum_{q \in Q}
\sum_{\tau=1}^{\tau_{\max}}
\frac{1}{M\tau_{\max}}\,
QL\bigl(y_{t+\tau}, \hat y^{(q)}(t,\tau), q\bigr)
$$

으로 정의된다.[^1_1]

테스트 시에는 normalized quantile risk(q-Risk)를 사용해 평가한다.[^1_1]

***

### 2.3 모델 구조 요약

간단히 말하면, TFT는 다음 모듈을 **순차 + 잔차적으로 조합**한 구조다.[^1_6][^1_7][^1_1]

1. 각 입력 타입별 임베딩/선형 투영 (정적, 과거, 미래)
2. 정적 변수 선택 + static encoders → $c_s,c_e,c_c,c_h$
3. 시변 변수 선택 네트워크 (과거/미래 각각)
4. LSTM seq2seq (cell/hidden 초기 상태에 $c_c,c_h$ 사용)
5. static enrichment (GRN with $c_e$)
6. interpretable multi-head self-attention
7. position-wise GRN + transformer block gating
8. quantile output layer

이 전체가 하나의 **글로벌 multi-horizon 예측기**로 학습된다.

***

### 2.4 성능 향상 및 한계

#### 2.4.1 성능 요약

네 가지 주요 데이터셋에서 baselines와 비교했을 때:[^1_1]

- **단변량 + 단순 입력 (Electricity, Traffic)**[^1_1]
    - DeepAR, DSSM, ConvTrans, MQRNN 등 대비
    - P50, P90 q-Risk에서 일관된 개선 (예: Electricity P90 기준 ConvTrans 대비 약 26% 향상).[^1_1]
- **복잡한 입력 (Retail, Volatility)**[^1_1]
    - 정적 메타데이터 + 관측 시변 입력 + known-future 입력을 모두 쓰는 환경
    - DeepAR/ConvTrans/MQRNN/Seq2Seq 대비 Retail P50에서 7–62%, P90에서 3–56%까지 개선.[^1_1]
    - Volatility와 같이 작고 노이즈가 큰 금융 데이터에서도, 가장 좋은 대안 모델 대비 P50/P90 기준 수 % 이상 개선.

**중요한 점**:
단순히 Transformer만 쓴 ConvTrans나, LSTM 기반의 DeepAR/DSSM보다, **입력 타입별 전용 모듈 + gating + variable selection**을 가진 TFT가 더 잘 일반화한다는 점을 보인다.[^1_8][^1_9][^1_10][^1_1]

#### 2.4.2 Ablation에서 드러난 구성요소의 역할

각 구성요소를 제거한 ablation 실험 결과:[^1_1]

- **지역 처리(seq2seq)와 self-attention 제거**
    - Temporal component 제거 시 P90 q-Risk가 데이터셋에 따라 **6–20% 이상 악화**, Electricity의 경우 self-attention이 일차적으로 중요하고, 다른 데이터셋에서는 seq2seq가 중요함을 보여줌.
- **정적 인코더 제거**
    - 정적 정보를 단순 concat으로 대체하면 P90 손실이 평균 2.6% 이상 증가.[^1_1]
    - 개체별 시계 패턴 차이를 충분히 포착하지 못해 성능 하락.
- **Instance-wise variable selection 제거**
    - 변수 선택 가중치를 학습 가능한 global coefficient로 고정하면 P90 손실이 평균 4.1% 이상 증가.[^1_1]
    - 샘플별 동적인 feature 조절이 일반화에 유리함을 시사.
- **Gating(GLU) 제거**
    - 단순 선형+ELU로 대체하면 평균 P90 손실이 1.9% 증가, 특히 Volatility처럼 **작고 노이즈가 큰 데이터**에서 영향이 크다(4% 이상).[^1_1]
    - 이는 gating이 복잡한 아키텍처를 “데이터가 허용하는 수준”으로 얕게 만들어 **과적합을 완화**함을 의미.


#### 2.4.3 한계

1. **복잡도와 자원 요구**
    - Electricity 기준 최적 TFT를 V100 1장으로 학습할 때, epoch당 50분+, 전체 6시간 이상 소요.[^1_1]
    - LSTM + attention + 여러 GRN/GLU로 인해 파라미터 수와 연산량이 크다.
2. **아키텍처 튜닝 비용**
    - hidden size, dropout, head 수 등 하이퍼파라미터 공간이 크고, 데이터셋 별로 최적 조합이 다르다.[^1_1]
3. **아주 장기(long-term) 예측에 대한 특화 설계 부족**
    - Autoformer/FEDformer, PatchTST 등의 후속 연구는 수백~수천 step horizon에 대해 더 특화된 구조(시계열 분해, 주기 기반 attention, patching 등)로 추가 개선을 이룬다.[^1_11][^1_12][^1_13][^1_14][^1_15]
    - TFT는 일반 multi-horizon 문제에는 강하지만, “극단적인 long-term forecasting(LTSF)” 기준으로는 최신 모델들에 비해 효율성이 떨어질 수 있다.
4. **해석 가능성의 한계(신뢰도)**
    - 변수 선택과 attention 패턴은 **유용한 통계적 시그널**을 제공하지만, **인과적 중요도**를 보장하지 않으며, 후속 연구에서는 bottleneck/개입(activation patching) 기반의 더 엄밀한 해석 방법이 제안되고 있다.[^1_16][^1_17]
5. **데이터 분포 변동(distribution shift)에 대한 전용 대책 부족**
    - RevIN, NLinear/DLinear, vLinear 등에서 지적되듯, 단순한 normalization/선형 구조가 분포 변화에 더 견고한 경우가 있다.[^1_18][^1_19][^1_20][^1_21][^1_22]
    - TFT는 이 부분을 명시적으로 다루지 않는다.

***

## 3. TFT와 “일반화 성능” – 구조적 강점과 개선 가능성

### 3.1 구조적으로 일반화에 유리한 요소

1. **Variable Selection + Gating = 구조적 정규화(structural regularization)**
    - 변수 선택 네트워크는 샘플별로 소수의 변수에 높은 가중치를 주며, 나머지는 억제한다. 이는 **feature-wise sparsity**에 가까운 효과를 가지며, 고차원 입력에서 일반화를 돕는다.[^1_1]
    - GRN/GLU gating은 복잡한 블록을 데이터가 필요로 하지 않을 때 “거의 선형”으로 만들며, 이는 깊은 네트워크의 overfitting을 줄이는 쪽으로 작용한다.[^1_1]
2. **글로벌 multi-entity 학습 + 정적 인코더**
    - 하나의 TFT가 수백~수만 개의 entity(매장, 상품, 계정 등)를 동시에 학습하면서, static encoder가 entity별 차이를 요약한다.[^1_1]
    - 이는 “개체 간 공유 통계”(global pattern)를 학습하면서도, static context로 개체별 특성을 반영해 **데이터 효율 좋은 generalization**을 가능하게 한다.
    - 후속 연구에서도 streamflow, 의료, 하이드로, 교통 등 다양한 도메인에 동일 구조가 재사용되며 LSTM/Transformer 대비 일관된 성능 우위를 보인다.[^1_23][^1_24][^1_25][^1_26][^1_27][^1_28]
3. **Quantile loss 기반 예측 구간 학습**
    - Quantile loss는 L1 성격을 가져 **outlier에 덜 민감**하고, 전체 분포의 형태(치우침, fat tail)를 부분적으로 반영한다.[^1_1]
    - 금융 변동성, 극한 강수 예측 등 heavy-tail 환경에서 TFT의 quantile 예측이 baseline보다 견고한 성능을 보였다는 응용 사례가 보고된다.[^1_29][^1_25][^1_26]
4. **Direct multi-horizon 학습**
    - DeepAR류의 **iterated one-step ahead** 방식은 error accumulation에 취약하지만, TFT는 direct 방식으로 각 horizon을 동시에 학습해 장기 horizon에서의 누적 오차를 줄인다.[^1_10][^1_1]
5. **해석 가능성 → 모델 점검 및 디버깅을 통한 간접적 일반화 향상**
    - 전역 variable importance, attention 기반 seasonal/lag 패턴, regime shift 탐지 등을 통해 **도메인 전문가가 모델을 검증/수정**할 수 있다.[^1_1]
    - 예컨대, Retail에서 “National Holiday”나 “On-promotion”이 높은 가중치를 받는 것이 도메인 지식과 부합하는지 확인할 수 있고, 부합하지 않으면 feature engineering이나 입력 선택을 다시 설계할 수 있다.[^1_1]
    - 이런 human-in-the-loop 검증 과정 자체가 실제 운영 환경에서의 **실질적 generalization**을 높인다.

### 3.2 일반화 성능을 더 끌어올릴 여지와 전략

1. **분해 기반 구조와의 결합 (Autoformer/FEDformer 계열)**
    - Autoformer는 시계열을 trend + seasonal로 분해하고, auto-correlation 메커니즘으로 주기성을 직접 모델링해 long-term에서 대규모 개선(38% 상대 개선)을 보인다.[^1_13][^1_30][^1_31][^1_32]
    - FEDformer는 seasonal-trend decomposition + 주파수 도메인(Fourier/Wavelet) attention으로 Transformer의 bias를 보완하고, 다변량/univariate 모두에서 14.8–22.6% MSE 감소를 보고한다.[^1_14][^1_15][^1_33][^1_34]
    - TFT에 이러한 decomposition 모듈을 통합한다면,
        - 추세/계절 요소는 **단순 모델(심지어 선형)**에 맡기고,
        - TFT는 잔차/비선형적 단기 패턴과 heterogeneous input interaction에 집중
시킬 수 있어, 파라미터 효율과 일반화를 동시에 개선할 수 있다.
2. **선형 모델과의 하이브리드**
    - “Are Transformers Effective for Time Series Forecasting?”는 DLinear/NLinear 같은 단순 선형 모델이 다수의 LTSF Transformer보다 일관되게 우수함을 보인다. 이는[^1_35][^1_19][^1_21][^1_22][^1_18]
> “긴 horizon에서는 복잡한 non-linear modeling보다는 trend/seasonality를 잘 잡는 선형 구조가 더 강하다”
는 점을 시사한다.
    - TFT 앞/뒤에 DLinear나 vLinear 같은 선형 계층을 붙여,
        - 장기적인 선형/주기 패턴은 선형 계층이 담당하고,
        - 잔차의 비선형 단기 패턴과 복잡한 공변량 상호작용만 TFT가 담당
하게 하면, **bias–variance trade-off 관점**에서 일반화가 좋아질 가능성이 크다.[^1_20][^1_36]
3. **분포 변동에 대한 전용 정규화/로버스트화**
    - RevIN, RLinear, vLinear 등은 입력 분포의 shift를 reversible normalization으로 보정해 Transformer/linear 모델의 성능을 크게 끌어올렸다고 보고한다.[^1_21][^1_20]
    - TFT에도
        - entity별 / feature별 정규화 (RevIN 스타일)
        - temporal shifting(시간대 변동)에 대한 augmentation
을 추가하면, COVID-19 같은 구조적 변화 상황에서 더 안정적인 일반화를 기대할 수 있다. 실제로 COVID 기간 수요 예측에서 TFT가 기존 모델보다 나은 회복력을 보였다는 전력/수요 관련 사례도 등장했다.[^1_37][^1_38]
4. **해석 결과의 “신뢰도”를 정량화하는 프레임워크와 결합**
    - 최근에는 Autoformer/FEDformer 계열에 **설명 가능한 bottleneck**을 두어, 특정 개념(예: hour-of-day)을 강제로 특정 서브공간에 통과시키고, activation patching으로 해석의 faithfulness를 평가하는 일이 시도되고 있다.[^1_16]
    - TFT의 variable selection/attention도 이런 프레임워크와 결합하면,
        - “이 변수/시점이 중요하다”는 해석이
        - 실제로 예측에 필수적인지(개입 후 성능 하락 여부)
를 검증할 수 있고, 그 과정에서 overfitting된 패턴(허위 상관)을 제거해 일반화를 개선할 수 있다.[^1_17][^1_16]
5. **경량화 및 구조 단순화**
    - TimesNet, TSLANet, ConvTimeNet, vLinear 등은 attention 대신 convolution/linear 구조로 SOTA에 근접하거나 능가하면서도, 계산 복잡도를 크게 줄이고 있다.[^1_39][^1_40][^1_41][^1_42][^1_43][^1_20]
    - TFT의 핵심인
        - static encoder,
        - variable selection,
        - gating 메커니즘
은 그대로 두되,
        - LSTM seq2seq를 1D/2D convolution이나 state space model로 교체,
        - self-attention을 patching(PatchTST)이나 sparse attention으로 교체
하는 식의 경량 TFT 변종을 설계하면, **과적합과 학습 불안정성을 줄여 일반화를 강화**할 수 있다.[^1_41][^1_42][^1_12][^1_11]

요약하면, TFT는 이미 구조적으로 일반화에 유리한 설계를 많이 담고 있지만, **(i) 분해/선형 모듈과의 결합, (ii) 분포 변동 대응, (iii) 해석 신뢰도 검증, (iv) 경량화**를 통해 generalization frontier를 더 밀어올릴 수 있는 여지가 크다.

***

## 4. 2020년 이후 관련 최신 연구 비교 분석

멀티-호라이즌/장기 시계열 예측 및 해석 가능성 관점에서 TFT 이후 중요한 흐름은 크게 다섯 가지로 볼 수 있다.

### 4.1 장기 시계열 전용 Transformer: Autoformer, FEDformer 등

- **Autoformer (NeurIPS 2021)**[^1_30][^1_31][^1_44][^1_45][^1_13]
    - 목표: **장기(long-term) forecasting 특화**
    - 핵심 아이디어
        - 시계열 분해를 아키텍처 내부에 삽입:

$x_t = Trend_t + Seasonal_t \$

를 progressive하게 분해하면서 예측
        - self-attention 대신 **Auto-Correlation** 메커니즘을 도입해, FFT 기반으로 모든 lag에 대한 상관을 계산 후 주기성 높은 lag에 집중
    - 성능
        - 6개 벤치마크(에너지, 교통, 경제, 날씨, 질병)에서 기존 Transformer류 대비 평균 **38% MSE 개선**.[^1_31][^1_13]
    - TFT와 비교
        - 입력 타입(정적/known-future/observed)에 특화된 설계는 거의 없고, 주로 **장기 주기성**에 초점을 둔다.
        - 해석 가능성은 주기/seasonality 수준의 직관은 있지만, TFT처럼 variable selection을 통한 feature-level 해석은 제공하지 않는다.
- **FEDformer (ICML 2022)**[^1_15][^1_33][^1_34][^1_14]
    - 목표: 장기 예측에서 전역 추세와 주파수 특성을 더 잘 반영
    - 핵심 아이디어
        - seasonal-trend decomposition + frequency-enhanced attention (Fourier/Wavelet 블록)
        - 선택적인 주파수 성분만 남겨서 선형 복잡도(시퀀스 길이에 선형) 달성
    - 성능
        - Autoformer 대비 multivariate에서 14.8%, univariate에서 22.6% MSE 감소.[^1_34][^1_14][^1_15]
    - TFT와 비교
        - TFT는 “heterogeneous inputs + interpretability”에 최적화, FEDformer는 “long sequence + global trend/periodicity”에 최적화.
        - 일반화 측면에서, **아주 긴 horizon**에서는 FEDformer류가 우위, **복잡한 입력/도메인 해석**에서는 TFT가 여전히 강점을 가진다.


### 4.2 Patch 기반 Transformer: PatchTST 계열

- **PatchTST (ICLR 2023)**[^1_46][^1_47][^1_48][^1_49][^1_12][^1_50][^1_11]
    - 아이디어
        - 시계열을 길이 $P$의 patch로 나누고, 각 patch를 Transformer token으로 사용
        - 각 채널(변수)은 독립적인 univariate 시계열로 보고 embedding/Transformer weight를 공유하는 **channel-independence** 전략
    - 장점
        - patch 단위로 시퀀스 길이를 줄여 **시간/메모리 복잡도 감소**
        - 더 긴 look-back window 사용 가능 → long-term 예측 성능 향상
        - self-supervised masked patch pretraining을 통해 **transfer learning 및 zero-shot** 능력 확보.[^1_12][^1_11]
    - TFT와 비교
        - PatchTST는 **입력 타입 구분/정적 공변량 통합/변수 선택** 같은 설계가 없고, 순수 시계열 값에 초점을 맞춘다.
        - 해석 가능성 측면에서, attention map은 제공되지만 변수/타입 별 구조적 해석은 TFT보다 약하다.


### 4.3 Task-general 백본: TimesNet 등

- **TimesNet (ICLR 2023)**[^1_40][^1_51][^1_52][^1_53][^1_43][^1_39]
    - 목표: forecasting, imputation, classification, anomaly detection 등 **범용(time-series foundation) 모델**
    - 핵심 아이디어
        - 1D 시계열을 다중 period 기반으로 2D 텐서(행: inter-period, 열: intra-period)로 변환
        - 2D CNN(Inception-style)으로 temporal 2D-variation을 학습
    - 성능
        - 5가지 주요 태스크에서 SOTA에 가까운 성능 보고.[^1_51][^1_52][^1_53][^1_39]
    - TFT와 비교
        - TimesNet은 **입력 타입 구분보다는 시계 패턴 자체의 복잡성**에 집중.
        - 해석 가능성은 2D 필터 시각화/period 분석 정도이며, TFT의 변수 선택/attention 유스케이스보다 덜 구조적이다.


### 4.4 선형 모델 르네상스: DLinear, NLinear, vLinear

- **LTSF-Linear, DLinear, NLinear (AAAI 2023)**[^1_19][^1_36][^1_22][^1_18][^1_35][^1_21]
    - 주장
        - 여러 LTSF Transformer(Informer, Autoformer, FEDformer, 등)와 비교했을 때,
            - 단일 선형 계층 또는 trend/seasonality 분해 + 선형 계층(DLinear) 기반 모델이 **20–50% 수준 MSE 개선**을 보이는 경우가 많다.
    - 통찰
        - LTSF에서는 **look-back window 내의 복잡한 temporal dynamics**보다,
**trend와 주기성을 적절히 캡처하는 것이 더 중요**하며, 이것은 상대적으로 단순한 선형 맵으로도 충분히 가능할 수 있다.[^1_35][^1_19]
- **vLinear (2025)**[^1_20]
    - vecTrans 모듈(다변량 상관을 선형 복잡도로 모델링) + WFMLoss(trajectory-aware flow matching loss)로
        - 22개 벤치마크, 124개 세팅에서 Transformer/TimesNet 등을 포함한 13개 SOTA 대비 SOTA 또는 근접 성능을 보이면서, **FLOPs와 메모리를 크게 절감**한다.[^1_20]

**TFT와의 관계**:
이 선형 계열 연구는 “모든 문제에 복잡한 Transformer/TFT가 꼭 필요하지는 않다”는 것을 보여주며, TFT 같은 복잡한 아키텍처가 **어떤 상황(복잡한 covariate 구조, 비선형 상호작용)**에서만 진가를 발휘하는지 명확히 구분할 필요가 있음을 시사한다.

### 4.5 해석 가능한 Transformer를 향한 후속 연구

- **Concept bottleneck + activation patching 기반 해석 가능성 강화**[^1_17][^1_16]
    - Autoformer, FEDformer 등 일반 Transformer에 대해,
        - 특정 “해석 가능한 개념”(예: 시간대)을 bottleneck으로 강제하고,
        - residual connection을 조절해 모든 정보가 bottleneck을 거치게 한 뒤,
        - activation patching으로 해석의 faithfulness를 검증하는 프레임워크를 제안.
    - TFT의 variable selection/gating/attention은 이런 프레임워크와 결합하기 좋은 구조로 평가되며, 여러 리뷰 논문에서 **해석 가능한 시계열 Transformer의 시초**로 인용된다.[^1_45][^1_54][^1_17]


### 4.6 TFT 기반 응용과 변형

- 의료: **중환자 vital sign trajectory 예측**, 수술 중 혈압 drop 조기 예측 등[^1_23]
- 에너지/유량: **streamflow 예측(2,610 유역)**, district heating/cooling thermal load 예측[^1_25][^1_26][^1_55]
- 교통/수송: 지하철 승객 흐름, 도로 교통량, 전력 수요[^1_56][^1_57][^1_37][^1_25]
- 금융/암호화폐: 주가/비트코인 예측에서 LSTM, CNN, Transformer 대비 우수 성능 및 해석 가능성 보고[^1_58][^1_59][^1_60]
- 환경/생태: 극한 강수량 quantile 예측, GPP(upscaling) 등[^1_61][^1_29][^1_25]

이들 연구는 공통적으로, LSTM/기본 Transformer 대비 **정적/시변 공변량과 불완전한 입력을 잘 처리하면서도 quantile 예측과 해석 가능성을 제공한다는 점** 때문에 TFT를 채택하고 있다.[^1_26][^1_62][^1_63][^1_27][^1_58][^1_29][^1_25][^1_23]

***

### 4.7 요약 비교 표

| 모델 | 연도/목표 | 핵심 아이디어 | TFT 대비 특징 |
| :-- | :-- | :-- | :-- |
| **TFT**[^1_1][^1_4][^1_2][^1_3] | 2019–2021, 일반 multi-horizon + 해석 가능성 | LSTM seq2seq + interpretable self-attention, variable selection, static encoder, gating, quantile loss | 이질적 입력 타입과 해석 가능성에 특화, 장기 horizon 전용은 아님 |
| **Autoformer**[^1_13][^1_30][^1_31] | 2021, LTSF | 내장 seasonal-trend 분해 + Auto-Correlation로 주기성 모델링 | 입력 타입 구분/해석은 약하지만 매우 긴 horizon에서 강함 |
| **FEDformer**[^1_14][^1_15][^1_33][^1_34] | 2022, LTSF | seasonal-trend 분해 + Fourier/Wavelet 기반 frequency attention | 전역 추세/주파수 패턴에 강함, 해석은 주로 주파수/분해 수준 |
| **PatchTST**[^1_11][^1_12][^1_47][^1_48] | 2022–2023, multivariate LTSF + self-supervised | patch 단위 token + channel-independence, masked patch pretraining | 매우 긴 시퀀스에 효율적, feature 타입/정적 정보 구조화는 없음 |
| **TimesNet**[^1_39][^1_51][^1_52][^1_53] | 2022–2023, foundation model | 1D→2D 다중 period 변환 + 2D CNN(Inception) | 다양한 태스크에 좋은 백본, 입력 타입별 구조/해석은 TFT보다 덜 명시적 |
| **DLinear/NLinear**[^1_19][^1_18][^1_35][^1_21][^1_22] | 2022–2023, LTSF 단순 baseline | trend-seasonality 분해 + 1-layer linear, RevIN | 구조 매우 단순, 많은 LTSF Transformer를 능가; 복잡한 covariate 구조에는 한계 |
| **vLinear**[^1_20] | 2025, multivariate linear | vecTrans(다변량 상관의 선형 복잡도) + WFMLoss | SOTA급 성능 + 효율, TFT 같은 복잡한 구조의 필요성을 재평가하게 함 |


***

## 5. TFT가 앞으로의 연구에 미치는 영향과 향후 연구 시 고려할 점

### 5.1 연구/실무에 미친 영향

1. **“멀티-호라이즌 + 이질적 입력 + 해석 가능성”을 위한 레퍼런스 아키텍처 정립**
    - TFT는 PyTorch Forecasting, Nixtla, NVIDIA NGC 모델 등 여러 라이브러리에서 **표준 multi-horizon 모델**로 구현되어 있다.[^1_64][^1_7][^1_6]
    - 실무자 입장에서는 “복잡한 정적/시변 공변량이 있을 때 먼저 시도해볼 수 있는 강력한 베이스라인”이 되었고, 연구자 입장에서는 “heterogeneous covariate 처리 + 해석 가능 Transformer 설계의 교과서적인 예”가 되었다.[^1_65][^1_7][^1_6]
2. **해석 가능한 deep time-series 모델의 방향 제시**
    - variable selection network, gating, interpretable attention, regime detection 등은 이후 많은 논문에서 인용되며,
        - Autoformer/FEDformer 해석 연구,[^1_16]
        - “Universal, Explainable Time-Series Forecasting” 같은 개념-물리(physics)-기반 프레임워크,[^1_17]
        - 도메인 응용(수문학, 의료, 금융)에서의 신뢰 가능한 예측 필요성
을 자극하였다.
3. **Hybrid 아키텍처(RNN + Transformer)의 효용을 재조명**
    - streamflow 연구 등에서 LSTM만/Transformer만보다 TFT(LSTM+attention hybrid)가 일관되게 우수하다는 결과가 보고되며,[^1_27][^1_25][^1_26]
    - “RNN vs Transformer” 구도가 아니라 **local RNN + global attention** 조합의 중요성이 강조되었다.
4. **Quantile-based multi-horizon forecasting의 실무적 채택 촉진**
    - TFT는 전체 horizon에 대한 quantile band를 한 번에 예측하는 구조를 정교하게 구현하여, 리테일 수요, 에너지, 금융 리스크 관리 등에서 “경로 전반의 best/worst-case”를 활용하는 문화를 확산시켰다.[^1_24][^1_55][^1_62][^1_25][^1_1]

### 5.2 앞으로 연구 시 고려할 점 (연구자 관점 제안)

1. **문제 유형에 따른 모델 선택/결합 전략**
    - **장기 horizon + 단순 공변량**(ETT, Weather 등): Autoformer/FEDformer, PatchTST, DLinear/vLinear 계열이 유리할 때가 많다.[^1_11][^1_19][^1_12][^1_14][^1_15][^1_20]
    - **복잡한 정적/시변 공변량 + 해석 필요**(의료, 에너지 수요, 리테일 등): TFT 또는 TFT-inspired 구조(정적 인코더, variable selection, gating)를 우선 고려하는 것이 합리적이다.[^1_55][^1_62][^1_24][^1_25][^1_23][^1_1]
    - 향후 모델 설계 시, **“어떤 문제에서 TFT 스타일 복잡도가 정말 필요한가?”**를 명시적으로 분석하고, 필요 없다면 선형/경량 Transformer와의 하이브리드로 단순화하는 것이 중요하다.
2. **분포 변동과 out-of-distribution 일반화**
    - COVID-19, 구조적 수요 변화, 규제 변경 등으로 인해 시계열 분포가 급변하는 상황에서 TFT가 얼마나 견고한지 정량적으로 평가하고,
    - RevIN, 데이터 증강, adversarial training, meta-learning 등을 통해 **TFT의 distribution shift 대응 능력**을 체계적으로 강화하는 연구가 필요하다.[^1_38][^1_19][^1_21][^1_37][^1_20]
3. **해석 가능성의 “faithfulness” 검증**
    - TFT의 variable importance와 attention 패턴이 실제 예측에 얼마나 필수적인지를
        - activation patching,
        - feature ablation,
        - counterfactual simulation
으로 측정하는 연구가 요구된다.[^1_16][^1_17]
    - 이는 “설명 가능한” 모델이 실제로 **신뢰 가능한(explainable \& faithful)** 모델인지 확인하는 데 핵심이다.
4. **멀티모달/스파티오-템포럴로의 확장**
    - ODformer, xMTrans 등 교통/OD 행렬 예측에서 Transformer가 공간–시간 정보를 동시에 처리하는 예가 늘고 있다.[^1_66][^1_57]
    - TFT의 **입력 타입 구분 + static encoder + variable selection** 개념을,
        - 이미지(위성/레이더),
        - 그래프(도로망/수문망),
        - 텍스트/이벤트 로그
등으로 확장한 **멀티모달 TFT**는 매우 유망한 연구 방향이다.
5. **구조 검색(AutoML)과의 결합**
    - AutoFormer-TS, auto-sktime 등은 Transformer/시계열 아키텍처를 자동으로 탐색하는 프레임워크를 제안한다.[^1_67][^1_8]
    - TFT의 복잡한 블록(정적 인코더, seq2seq, self-attention, gating)을
        - 데이터셋별로 어느 조합/깊이가 최적인지
를 AutoML로 탐색하는 연구는,
        - 실무자가 “TFT 계열을 쓸지, 어떤 변형을 쓸지”를 자동으로 결정할 수 있게 해 줄 것이다.
6. **경량/엣지 환경과 양자/하드웨어 특화 TFT**
    - IoT/엣지 환경에서 TFT를 구동하기 위한 pruning, distillation, low-rank factorization 연구와,
    - Quantum Temporal Fusion Transformer(QTFT)처럼 양자–고전 하이브리드 구조로 확장하는 시도도 시작되었다.[^1_68][^1_69][^1_70]
    - **리소스 제약 환경에서의 해석 가능한 멀티-호라이즌 예측**이라는 새로운 응용 지평을 열 수 있다.

요약하면, TFT는 “복잡한 멀티-호라이즌 시계열 + 해석 가능성”이라는 중요한 영역에서 **사실상 표준(reference)** 역할을 하고 있으며, 이후 Transformer/선형 계열 연구들은 **길이/복잡도/효율/해석 가능성/분포 변동**이라는 다섯 축에서 TFT가 열어놓은 공간을 더욱 구조화하고 확장하고 있다. 앞으로의 연구에서는, TFT의 강점을 계승하면서도 **필요 이상의 복잡성을 줄이고, 분포 변동과 해석 신뢰도 문제를 정면으로 다루는 방향**이 가장 높은 연구·실무 가치를 가질 것이다.
<span style="display:none">[^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_80][^1_81][^1_82][^1_83][^1_84][^1_85][^1_86][^1_87][^1_88][^1_89][^1_90][^1_91][^1_92][^1_93]</span>

<div align="center">⁂</div>

[^1_1]: 1912.09363v3.pdf

[^1_2]: https://arxiv.org/pdf/1912.09363.pdf

[^1_3]: https://research.google/pubs/temporal-fusion-transformers-for-interpretable-multi-horizon-time-series-forecasting/

[^1_4]: https://arxiv.org/abs/1912.09363

[^1_5]: https://arxiv.org/pdf/2209.03945.pdf

[^1_6]: https://catalog.ngc.nvidia.com/orgs/nvidia/resources/tft_for_pytorch

[^1_7]: https://nixtlaverse.nixtla.io/neuralforecast/docs/tutorials/forecasting_tft.html

[^1_8]: https://arxiv.org/pdf/2502.13721.pdf

[^1_9]: https://link.springer.com/10.1007/978-3-031-26422-1_3

[^1_10]: https://link.springer.com/10.1007/978-3-030-85713-4_11

[^1_11]: https://arxiv.org/abs/2211.14730

[^1_12]: https://arxiv.org/pdf/2211.14730.pdf

[^1_13]: https://www.semanticscholar.org/paper/fc46ccb83dc121c33de7ab6bdedab7d970780b2f

[^1_14]: http://arxiv.org/abs/2201.12740v2

[^1_15]: https://proceedings.mlr.press/v162/zhou22g/zhou22g.pdf

[^1_16]: https://openreview.net/pdf?id=6IYJgYc6Ie

[^1_17]: https://www.arxiv.org/pdf/2508.01407.pdf

[^1_18]: https://nixtlaverse.nixtla.io/neuralforecast/models.dlinear.html

[^1_19]: https://arxiv.org/abs/2205.13504

[^1_20]: https://arxiv.org/html/2601.13768v1

[^1_21]: https://arxiv.org/pdf/2305.10721.pdf

[^1_22]: https://arxiv.org/pdf/2403.14587.pdf

[^1_23]: https://ieeexplore.ieee.org/document/9745215/

[^1_24]: https://arxiv.org/pdf/2207.00610.pdf

[^1_25]: https://www.sciencedirect.com/science/article/abs/pii/S0022169424006966

[^1_26]: https://arxiv.org/pdf/2305.12335.pdf

[^1_27]: https://arxiv.org/html/2506.20831

[^1_28]: https://pdfs.semanticscholar.org/7849/fa9d9f1c1f9896bf6a3a785f1ef227b23279.pdf

[^1_29]: http://arxiv.org/pdf/2107.06846.pdf

[^1_30]: https://arxiv.org/pdf/2106.13008.pdf

[^1_31]: https://github.com/SQY2021/Autoformer_NeurIPS-2021

[^1_32]: https://arxiv.org/abs/2106.13008

[^1_33]: https://arxiv.org/abs/2201.12740

[^1_34]: https://arxiv.org/pdf/2201.12740.pdf

[^1_35]: https://sonstory.tistory.com/119

[^1_36]: https://arxiv.org/html/2403.14587v2

[^1_37]: https://annals-csis.org/Volume_39/drp/2959.html

[^1_38]: https://www.arxiv.org/pdf/2411.11350.pdf

[^1_39]: https://arxiv.org/abs/2210.02186

[^1_40]: https://arxiv.org/pdf/2210.02186.pdf

[^1_41]: https://arxiv.org/html/2403.01493v1

[^1_42]: https://arxiv.org/pdf/2404.08472.pdf

[^1_43]: https://ar5iv.labs.arxiv.org/html/2210.02186

[^1_44]: https://ise.thss.tsinghua.edu.cn/~mlong/doc/Autoformer-nips21.pdf

[^1_45]: https://www.semanticscholar.org/paper/Autoformer:-Decomposition-Transformers-with-for-Wu-Xu/fc46ccb83dc121c33de7ab6bdedab7d970780b2f

[^1_46]: https://huggingface.co/blog/patchtst

[^1_47]: https://huggingface.co/docs/transformers/model_doc/patchtst

[^1_48]: https://kp-scientist.tistory.com/entry/ICLR-2023-PatchTST-A-Time-Series-is-Worth-64-Words-Long-Term-Forecasting-with-Transformers

[^1_49]: https://data-newbie.tistory.com/945

[^1_50]: https://www.semanticscholar.org/paper/A-Time-Series-is-Worth-64-Words:-Long-term-with-Nie-Nguyen/dad15404d372a23b4b3bf9a63b3124693df3c85e

[^1_51]: https://www.arxiv.org/abs/2210.02186

[^1_52]: https://ise.thss.tsinghua.edu.cn/~mlong/doc/TimesNet-iclr23.pdf

[^1_53]: https://www.semanticscholar.org/paper/TimesNet:-Temporal-2D-Variation-Modeling-for-Time-Wu-Hu/47696145b3f88c4cc3f3c22035286b5d7ebce09d

[^1_54]: https://peerj.com/articles/cs-3001/

[^1_55]: https://pdfs.semanticscholar.org/033e/592b28c5aa5b9d1dd2d22a98af0ccd6bcd6b.pdf

[^1_56]: https://ieeexplore.ieee.org/document/9551442/

[^1_57]: http://arxiv.org/pdf/2405.04841.pdf

[^1_58]: https://ieeexplore.ieee.org/document/9731073/

[^1_59]: https://arxiv.org/pdf/2509.10542.pdf

[^1_60]: https://www.sciencedirect.com/science/article/pii/S2405844024161737

[^1_61]: https://arxiv.org/pdf/2306.13815.pdf

[^1_62]: https://peerj.com/articles/cs-2713/

[^1_63]: https://pubmed.ncbi.nlm.nih.gov/41339653/

[^1_64]: https://pytorch-forecasting.readthedocs.io/en/v1.4.0/tutorials/stallion.html

[^1_65]: https://deepfa.ir/en/blog/temporal-fusion-transformers-time-series-forecasting

[^1_66]: https://arxiv.org/abs/2208.08218

[^1_67]: https://arxiv.org/pdf/2312.08528.pdf

[^1_68]: https://www.arxiv.org/pdf/2508.04048.pdf

[^1_69]: https://arxiv.org/html/2508.04048v1

[^1_70]: https://www.arxiv.org/pdf/2508.04048v1.pdf

[^1_71]: https://www.mdpi.com/2072-4292/14/3/733

[^1_72]: https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210X.13871

[^1_73]: https://ieeexplore.ieee.org/document/9797076/

[^1_74]: https://www.mdpi.com/2072-4292/14/21/5333

[^1_75]: http://www.cabidigitallibrary.org/doi/10.31220/agriRxiv.2022.00155

[^1_76]: http://biorxiv.org/lookup/doi/10.1101/2022.01.04.474883

[^1_77]: https://onlinelibrary.wiley.com/doi/10.1111/gcbb.13023

[^1_78]: https://www.semanticscholar.org/paper/ef5a2e0e3887d76154a41cf799cbd3cbc216edf4

[^1_79]: https://arxiv.org/abs/1706.08838

[^1_80]: http://arxiv.org/pdf/2411.17382.pdf

[^1_81]: https://arxiv.org/pdf/2405.19647.pdf

[^1_82]: https://arxiv.org/html/2406.03710v1

[^1_83]: https://arxiv.org/pdf/2202.08408.pdf

[^1_84]: https://github.com/cure-lab/LTSF-Linear

[^1_85]: https://liner.com/review/timesnet-temporal-2dvariation-modeling-for-general-time-series-analysis

[^1_86]: https://velog.io/@ha_yoonji99/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-Are-Transformers-Effective-for-Time-Series-Forecasting-AAAI-2023-NLinear-DLinear

[^1_87]: https://arxiv.org/html/2501.08620v1

[^1_88]: https://www.arxiv.org/pdf/2501.01087v3.pdf

[^1_89]: https://arxiv.org/html/2501.08620v4

[^1_90]: https://arxiv.org/pdf/2504.00118.pdf

[^1_91]: https://link.springer.com/10.1007/s00506-021-00770-4

[^1_92]: https://ieeexplore.ieee.org/document/9741878/

[^1_93]: https://www.mdpi.com/1424-8220/21/14/4764

