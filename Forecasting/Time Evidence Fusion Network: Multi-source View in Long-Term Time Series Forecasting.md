# Time Evidence Fusion Network: Multi-source View in Long-Term Time Series Forecasting

아래 내용은 ** `Time Evidence Fusion Network: Multi-source View in Long-Term Time Series Forecasting`을 1차 근거**로 삼고, 2020년 이후 관련 연구는 **공식 학회/논문 페이지와 arXiv를 추가 확인**하여 작성한 연구자 관점의 분석입니다. Tianxiang Zhan 등 6인이 제안한 TEFN 논문이며, 핵심 목적은 시계열 예측에서 **정확도뿐 아니라 계산 효율성·안정성·해석가능성을 동시에 확보하는 것**입니다. 

중요하게, 아래에서는 **[저자 보고]**와 **[해석/비판]**을 명시적으로 구분합니다. 특히 논문의 “SOTA”, “robust”, “uncertainty modeling”, “generalization” 표현은 실험이 실제로 입증한 범위보다 강하게 읽힐 수 있으므로 별도로 검토했습니다.

---

# 1. Executive Summary — 10문장 이내

1. TEFN(Time Evidence Fusion Network)은 다변량 시계열의 **시간 축(time dimension)**과 **채널 축(channel dimension)**을 서로 다른 정보원(source)으로 간주하고, Dempster–Shafer Evidence Theory의 **BPA(Basic Probability Assignment)** 개념을 신경망에 도입한 장기 시계열 예측 모델입니다. 
2. 모델은 입력 정규화 → 시간축 선형 projection → 시간/채널 BPA → expectation-based fusion → 역정규화라는 매우 단순한 구조를 사용합니다. 
3. BPA는 각 시계열 값을 하나의 feature로 압축하기보다 여러 가능한 “event”에 대한 membership representation으로 확장하여 불확실성과 모호성을 표현하려는 것이 핵심 아이디어입니다. 
4. 저자들은 고전적인 Dempster 결합 규칙 대신 계산량과 수치 안정성을 고려하여 **expectation fusion**을 사용하고, 시간 BPA와 채널 BPA의 출력을 최종적으로 더합니다. 
5. 8개 benchmark series와 4개 forecast horizon에서 TEFN은 특히 MAE 기준으로 강한 결과를 보이지만, MSE 기준에서는 iTransformer와 PatchTST가 더 많은 1위를 기록하므로 “모든 조건에서 SOTA”라고 해석하는 것은 어렵습니다. 
6. TEFN의 중요한 장점은 Transformer 계열보다 훨씬 단순한 연산 구조를 유지하면서 경쟁력 있는 정확도를 노린다는 점이며, 저자는 고정된 채널 수와 sample-space 크기를 가정하면 시간 복잡도를 $\mathcal{O}(L)$로 설명합니다. 
7. 하지만 실제 복잡도에는 $C$와 $2^{|S|}$가 포함되므로, 채널 수가 크거나 $|S|$가 커질 경우 “선형 복잡도”라는 표현은 조건부입니다.
8. 논문의 가장 중요한 이론적 약점은 Eq. (11)의 선형 membership 값이 실제 Dempster–Shafer mass가 되기 위한 **비음수성·합이 1이라는 제약이 수식상 명확히 제시되지 않았다는 점**입니다. 
9. 또한 random seed 반복, confidence interval, 유의성 검정, calibrated uncertainty 및 cross-dataset zero-shot 검증이 없기 때문에 논문의 “robustness/generalizability”는 제한된 의미로 해석해야 합니다.
10. 연구적으로 TEFN의 가장 큰 잠재력은 현재의 “선형 BPA + 단순 합산”을 **정규화된 evidential representation, adaptive nonlinear BPA, distribution-shift adaptation, pretraining/zero-shot generalization**으로 발전시키는 데 있습니다.

---

# 1-1. 연구 목적과 필요성

## 연구가 해결하려는 핵심 문제

### [저자 보고]

저자는 실용적인 시계열 시스템에서는 예측 정확도만 높다고 충분하지 않으며,

* prediction accuracy,
* training/inference efficiency,
* parameter stability,
* interpretability

를 동시에 만족해야 한다고 문제를 정의합니다. 기존 Transformer는 강력하지만 계산량과 메모리 요구량이 크고, 단순 linear model은 효율적이지만 복잡한 불확실성과 채널 관계를 충분히 설명하지 못한다는 문제의식입니다. 

여기에 TEFN은 다음 질문을 던집니다.

> “다변량 시계열의 시간 정보와 변수 정보를 서로 다른 evidence source로 간주하고, 이를 evidence theory 방식으로 표현·융합하면 단순한 모델로도 충분한 forecasting 성능을 얻을 수 있는가?”

Fig. 1에서 이 사고방식이 직접 표현됩니다. 같은 종류의 정보원을 먼저 합친 다음, 시간 정보원과 채널 정보원을 다시 융합해 예측을 생성합니다. 

### 용어 설명 — **Evidence Theory / Dempster–Shafer Theory**

일반적인 확률론에서는 특정 사건 $A$ 의 확률 $P(A)$를 직접 할당합니다. Evidence Theory에서는 하나의 사건뿐 아니라

$$
\{A,B\}
$$

처럼 “ $A$인지 $B$인지 아직 구분할 수 없다”는 **집합 자체에도 belief mass를 할당할 수 있습니다.**

즉,

$$
m(A)=0.7,\qquad m(\{A,B\})=0.3
$$

와 같이 **불확실성을 억지로 하나의 확률값으로 분해하지 않고 남겨둘 수 있다는 것**이 핵심입니다.

---

# 2. 핵심 주장과 근거

| 핵심 주장                                   | 저자가 제시한 근거                                       | 위치                      | 연구자 관점 판단                                                                             |                          |                                       |
| --------------------------------------- | ------------------------------------------------ | ----------------------- | ------------------------------------------------------------------------------------- | ------------------------ | ------------------------------------- |
| 시간축과 채널축을 서로 다른 evidence source로 볼 수 있다 | Time BPA와 Channel BPA를 병렬 구성                     | p.2 Fig.1, p.4 Fig.4    | **합리적인 inductive bias**이나 유일한 분해 방법은 아님                                               |                          |                                       |
| BPA가 시계열의 uncertainty/ambiguity를 표현한다   | fuzzy membership을 $2^{ \mid S \mid }$ event dimension으로 확장                                                               | p.4–5, Eq.11, Fig.5      | **아이디어는 흥미롭지만 mass-function 제약이 불명확** |
| 두 BPA를 같이 사용하면 한 축만 사용하는 것보다 일반적으로 좋다   | ablation의 w/o T, w/o C 비교                        | p.10–11, Table V, Eq.17 | 대체로 지지되나 저자 자신도 예외 존재 인정                                                              |                          |                                       |
| expectation fusion이 DSR보다 효율적이다         | multiplication을 피하고 추가 fusion parameter를 사용하지 않음 | p.5–6, Eq.12–13         | 효율성 측면 타당하나 **D-S evidence fusion으로서의 엄밀한 의미는 약해짐**                                   |                          |                                       |
| TEFN이 SOTA급 성능을 낸다                      | Table II, ETT/Exchange 등의 결과                     | p.7, Table II           | **부분적으로만 맞음**. MSE 전체에서는 PatchTST/iTransformer 우세 영역이 많음                              |                          |                                       |
| TEFN은 효율적이다                             | Electricity-96의 model-size/time/MSE bubble plot  | p.8, Fig.7              | 유력한 장점이나 단일 task 평가이며 file size가 parameter count를 완전히 대변하지 않음                         |                          |                                       |
| hyperparameter에 안정적이다                   | learning rate와 $\mid S \mid$ grid 탐색, error variance                                                             | p.9–10, Fig.8–9/Table IV | **random-seed stability와는 다른 개념**     |
| noise에 robust하다                         | Gaussian noise 추가 실험                             | p.11–12, Table VI       | 결과가 비정상적으로 좋아지는 경우가 있어 실험 protocol 추가 설명 필요                                           |                          |                                       |
| 모델이 interpretable하다                     | BPA membership function 시각화                      | p.12 Fig.10             | parameter visibility는 존재하나 **faithful explanation 또는 uncertainty calibration 검증은 없음** |                          |                                       |
| 선형 TEFN은 nonlinear series에 한계가 있다       | nonlinearity proxy와 error의 높은 상관                 | p.9–10, Table III/Fig.9 | 저자가 명확히 인정한 핵심 한계                                                                     |                          |                                       |

TEFN의 실험 데이터는 Electricity, 4개 ETT subset, Exchange, Traffic, Weather를 포함하며 채널 수가 7에서 862까지 다양합니다. 

---

# 2-1. 해결 문제 → 방법 → 수식 → 모델 구조 → 성능 → 한계

# A. 기본 문제 정의

다변량 시계열을

$$
X\in\mathbb{R}^{L_{\text{in}}\times C}
$$

라고 두겠습니다.

여기서

* $L_{\text{in}}$: 입력으로 관찰한 과거 시점의 개수입니다.
* $C$: 변수 또는 channel의 수입니다.
* $L_{\text{pred}}$: 앞으로 예측할 시점의 수입니다.
* $\hat{Y}\in\mathbb{R}^{L_{\text{pred}}\times C}$: 모델이 출력해야 하는 미래 시계열입니다.

TEFN은 이 $X$를 단순히 하나의 feature tensor로 보지 않고,

$$
\text{Time evidence}
\qquad\text{and}\qquad
\text{Channel evidence}
$$

로 분리해 해석합니다.

### 용어 설명 — **Channel**

다변량 시계열에서 channel은 서로 다른 측정 변수를 의미합니다. 예를 들어 온도, 압력, 유량이 동시에 측정된다면 $C=3$입니다.

---

# B. Step 1 — 입력 정규화

논문의 Eq. (6)–(9)는 다음 구조입니다. 

```math
\mu
=
\frac{1}{|x|}
\sum_{x_i\in x}x_i
```

```math
\sigma^2
=
\frac{1}{|x|}
\sum_{x_i\in x}(x_i-\mu)^2
```

```math
x_{\text{Norm}}
=
\frac{x-\mu}{\sigma}
```

예측 후에는

```math
\hat y
=
\sigma\hat y_{\text{Norm}}+\mu
```

로 원래 scale로 되돌립니다.

### 기호 설명

* $x$: 한 입력 시계열 segment입니다.
* $|x|$: segment에 포함된 관측값 수입니다.
* $x_i$: $i$번째 관측값입니다.
* $\mu$: 입력 segment 평균입니다.
* $\sigma^2$: 입력 segment의 분산입니다.
* $\sigma$: 표준편차입니다.
* $x_{\text{Norm}}$: 정규화된 입력입니다.
* $\hat y_{\text{Norm}}$: 정규화 공간에서의 예측입니다.
* $\hat y$: 실제 단위로 복원한 최종 예측입니다.

### [저자 보고]

정규화가 데이터 scale 차이와 outlier 영향을 줄이고 convergence와 stability를 향상시킨다고 설명합니다. 

### [해석]

이 부분은 Non-stationary Transformer 계열의 stationarization 아이디어와 유사합니다. 하지만 Table V에서는 **Exchange처럼 normalization을 제거했을 때 오히려 좋아지는 데이터도 확인됩니다.** 저자 역시 역정규화 시 미래의 평균·분산 변화까지 설명하지 못한다는 문제를 인정합니다. 

즉 일반화 성능 관점에서 normalization은 항상 유리한 것이 아니라,

$$
P_{\text{train}}(X)\neq P_{\text{test}}(X)
$$

인 **distribution shift**가 존재하면 오히려 문제가 될 수 있습니다.

### 용어 설명 — **Distribution Shift**

학습 데이터와 실제 미래 데이터의 통계적 분포가 달라지는 현상입니다. 예를 들어 학습 시 평균 온도가 $20^\circ$ C였지만 미래 운전조건에서 $28^\circ$ C가 된다면 평균과 분산이 이동할 수 있습니다.

---

# C. Step 2 — Time Dimension Projection

논문의 Eq. (10)은

```math
x'
=
\text{project}(x)
=
xW_p+b
```

입니다. 

입력 길이 $L_{\text{in}}$을

$$
L_{\text{in}}+L_{\text{pred}}
$$

차원으로 직접 확장합니다.

### 기호 설명

* $W_p$: 학습되는 projection weight입니다.
* $b$: bias입니다.
* $x'$: forecast horizon까지 시간축이 확장된 표현입니다.

### [해석]

이 구조가 TEFN의 효율성을 크게 높입니다.

autoregressive 모델처럼

$$
\hat y_{t+1}\rightarrow\hat y_{t+2}\rightarrow\cdots
$$

순서로 반복 예측하지 않고 미래 horizon 전체를 한 번에 생성할 수 있기 때문입니다.

따라서 error accumulation도 줄일 가능성이 있습니다.

---

# D. Step 3 — BPA: 논문의 핵심

Evidence Theory의 일반적인 mass function은

$$
m:2^\Omega\rightarrow[0,1]
$$

처럼 정의하며 이상적으로는

$$
m(\emptyset)=0,
\qquad
\sum_{A\subseteq\Omega}m(A)=1
$$

을 만족합니다.

### 용어 설명 — **Frame of Discernment**

$\Omega$는 가능한 기본 상태들의 집합입니다.

예를 들어

$$
\Omega=\{\text{Low},\text{High}\}
$$

라면 power set은

$$
2^\Omega=
\{
\emptyset,
\{\text{Low}\},
\{\text{High}\},
\{\text{Low},\text{High}\}
\}
$$

가 됩니다.

### 용어 설명 — **Power Set**

집합 $S$의 가능한 모든 부분집합들의 집합이며 원소 수는

$$
|2^S|=2^{|S|}
$$

입니다.

그래서 $|S|$가 커질수록 BPA dimension은 지수적으로 증가합니다.

---

TEFN의 실제 BPA는 Eq. (11)입니다.

```math
m_{D,i,j,k}
=
\mu_k(x_{\text{Norm},i,j})
=
w_{D,j,k}x_{\text{Norm},i,j}
+
b_{D,j,k}
```



### 기호 설명

* $D$: 적용하는 차원이며 $D\in{T,C}$입니다.
* $T$: time dimension입니다.
* $C$: channel dimension입니다.
* $i$: sample 또는 반대 축 index입니다.
* $j$: 현재 BPA가 적용되는 time/channel index입니다.
* $k$: event-space index입니다.
* $k=1,\dots,2^{|S|}$입니다.
* $w_{D,j,k}$: membership function의 학습 가능한 기울기입니다.
* $b_{D,j,k}$: 학습 가능한 절편입니다.
* $\mu_k(\cdot)$: $k$번째 fuzzy membership function입니다.
* $m_{D,i,j,k}$: 저자가 mass representation으로 해석하는 값입니다.

Fig. 5는 하나의 값을 **압축하지 않고 $2^{|S|}$개의 가능한 event 방향으로 확장**시키는 것을 보여줍니다. 

### 용어 설명 — **Fuzzy Membership**

“이 관측값이 어떤 개념에 얼마나 속하는가”를 연속적인 값으로 표현합니다.

고전적인 set은

$$
x\in A\quad\text{or}\quad x\notin A
$$

뿐이지만 fuzzy set에서는

$$
\mu_A(x)=0.2,\;0.7,\;0.95
$$

같은 정도를 사용할 수 있습니다.

---

## 여기서 중요한 이론적 문제

논문의 Section II에서는 여러 membership 값을 mass로 바꾸려면 normalization이 필요하다고 설명합니다. 

하지만 실제 TEFN Eq. (11)은

$$
wx+b
$$

만 보여줍니다.

따라서 논문 수식만 놓고 보면

$$
m_{D,i,j,k}<0
$$

가 될 수도 있고,

$$
\sum_km_{D,i,j,k}\neq1
$$

일 수도 있습니다.

즉 **Eq. (11)의 output이 수학적으로 엄밀한 BPA의 모든 조건을 어떻게 만족하는지 논문에서 충분히 명시하지 않았습니다.**

이것은 TEFN을 Evidence Theory 모델로 해석할 때 상당히 중요한 unresolved issue입니다.

---

# E. Step 4 — Expectation Fusion

시간축에서 얻은 mass representation을

$$
m_T
$$

채널축에서 얻은 것을

$$
m_C
$$

라고 합니다.

논문의 Eq. (12)는 이를 fusion parameter와 함께 합산하는 expectation transformation을 사용합니다. 

논문 표기를 최대한 보존하면 개념적으로

```math
\hat y_{\text{Norm},D}
=
\sum_j\sum_k
y_{j,k}m'_{i,j,k}
```

의 형태입니다.

### 기호 설명

* $D$: time 또는 channel source입니다.
* $m'_{i,j,k}$: fusion 전에 변환된 mass representation입니다.
* $y_{j,k}$: 저자가 fusion parameter라고 정의한 학습 가능한 값입니다.
* $j$: time/channel dimension의 위치입니다.
* $k$: evidence-event index입니다.
* $\hat y_{\text{Norm},D}$: 해당 source에서 생성된 normalized prediction입니다.

**주의:** 원문의 Eq. (12)는 $m$, $m'$ 및 $y_{j,k}$ 사이의 표기 관계가 다소 불명확합니다. 따라서 식을 논문보다 더 구체적인 확률적 expectation으로 재해석하는 것은 피하는 것이 안전합니다.

---

# F. Dempster Rule을 왜 직접 사용하지 않았는가?

논문에서 소개하는 Eq. (1)은

```math
m(A)
=
\sum_{B\cap C=A}m_1(B)m_2(C)
```

형태입니다. 

### 기호 설명

* $m_1$, $m_2$: 두 정보원이 제공하는 mass function입니다.
* $B$, $C$: 각각의 focal element입니다.
* $A$: 두 evidence를 합친 뒤 관심 있는 event입니다.

다만 표준적인 **normalized Dempster rule**은 conflict $K$가 존재한다면 일반적으로

```math
m(A)
=
\frac{
\displaystyle\sum_{B\cap C=A}m_1(B)m_2(C)
}{
1-K
}
```

와 같이 normalization을 포함합니다.

따라서 논문 Eq. (1)은 conflict normalization을 생략한 단순화된 표현으로 읽는 것이 적절합니다.

TEFN 자체는 이 DSR을 사용하지 않습니다.

저자들은 DSR이

1. 고차원에서 계산량이 커지고,
2. mass 간 multiplication이 필요하며,
3. extreme mass distribution에 민감할 수 있다는 이유로

expectation fusion으로 대체합니다. 

---

# G. Step 5 — 두 source 결과 통합

Eq. (13)은 매우 단순합니다.

```math
\hat y_{\text{Norm}}
=
\sum_{D\in\{T,C\}}
\hat y_{\text{Norm},D}
=
\hat y_{\text{Norm},T}
+
\hat y_{\text{Norm},C}
```



즉

$$
\boxed{
\text{Time evidence prediction}
+
\text{Channel evidence prediction}
}
$$

입니다.

이후

```math
\hat y
=
\sigma\hat y_{\text{Norm}}+\mu
```

로 복원합니다.

---

# 3. 전체 모델 구조

논문의 Fig. 4를 수학적으로 정리하면 다음과 같습니다.

$$
X
\rightarrow
X_{\text{Norm}}
\rightarrow
X'_{\text{Norm}}
\rightarrow
\begin{cases}
\text{Time BPA}\\
\text{Channel BPA}
\end{cases}
\rightarrow
\begin{cases}
\hat Y_T\\
\hat Y_C
\end{cases}
\rightarrow
\hat Y_T+\hat Y_C
\rightarrow
\text{DeNorm}
$$

Fig. 4는 Norm → Time Projection → parallel BPA → fusion → De-Norm 구조를 직접 보여줍니다. 

이 모델에서 Transformer의 self-attention이나 recurrent state는 없습니다.

따라서 TEFN의 핵심은 “복잡한 sequence model”이 아니라

> **Evidence representation을 이용해 매우 단순한 linear projection을 더 풍부하게 만드는 것**

이라고 보는 편이 정확합니다.

---

# 4. 연구 주제·방법·결과: 저자 주장과 해석 분리

## 연구 주제

**[저자 보고]**
Evidence Theory와 multi-source information fusion을 장기 다변량 시계열 forecasting backbone에 도입하는 것입니다. 

**[해석]**
실질적으로는 “evidentially motivated linear/MLP forecasting model”에 가깝습니다. 기존의 D-S rule 자체를 neural forecasting engine으로 사용하는 것이 아니라 **D-S/BPA 개념에서 영감을 받은 representation expansion**을 사용합니다.

---

## 방법

**[저자 보고]**
time/channel을 별도 source로 보고, 각각 $2^{|S|}$ BPA dimension으로 확장한 뒤 expectation으로 합칩니다.

**[해석]**
TEFN의 진정한 inductive bias는 “Evidence Theory”라는 명칭 자체보다

$$
\text{axis-specific expansion}
+
\text{parameter sharing}
+
\text{additive fusion}
$$

에 있을 가능성도 있습니다.

### 용어 설명 — **Inductive Bias**

모델이 학습 전부터 갖고 있는 구조적 가정입니다.

TEFN의 경우

> “시간 방향 정보와 변수 방향 정보는 별도로 학습한 후 합치는 것이 좋다.”

라는 가정 자체가 inductive bias입니다.

---

# 5. 성능 결과를 정확하게 읽기

Table II는 8개 series × 4 horizon에 대해 MSE와 MAE를 비교합니다. 

## 저자가 강조한 성공 사례

ETTh1, $L_{\text{pred}}=720$에서:

$$
\text{TEFN MSE}=0.475
$$

$$
\text{iTransformer MSE}=0.503
$$

$$
\text{PatchTST MSE}=0.500
$$

따라서 TEFN은 각각

$$
0.028,\qquad0.025
$$

낮습니다. 

Exchange-96에서도

$$
\text{TEFN MSE}=0.082
$$

로 iTransformer의 $0.086$, PatchTST의 $0.088$보다 좋습니다. 

---

## 하지만 전체 결과는 훨씬 복합적입니다

예를 들어 Electricity 평균 MSE:

$$
\text{TEFN}=0.215
$$

$$
\text{iTransformer}=0.178
$$

즉 TEFN이 상당히 뒤집니다. 

Traffic 평균은 차이가 더욱 큽니다.

$$
\text{TEFN}=0.623
$$

$$
\text{iTransformer}=0.428
$$

$$
\text{PatchTST}=0.481
$$



---

## 1위 횟수를 보면 논문의 성격이 더 명확합니다

Table II의 first-count를 기준으로 32개 forecasting condition에서:

| Model        |              MSE 1위 |              MAE 1위 |
| ------------ | ------------------: | ------------------: |
| TEFN         |  4 / 32 = **12.5%** | 15 / 32 = **46.9%** |
| iTransformer | 10 / 32 = **31.3%** | 15 / 32 = **46.9%** |
| PatchTST     | 12 / 32 = **37.5%** |  5 / 32 = **15.6%** |

원자료는 Table II의 first-count입니다. 

### [해석]

따라서 TEFN은

> **“MSE에서 압도적인 SOTA”**

라기보다

> **“특히 MAE 관점에서 매우 경쟁력 있으며, 단순성·효율성까지 고려하면 매력적인 Pareto solution”**

이라고 평가하는 것이 더 정확합니다.

---

# 6. 계산 복잡도

저자들은 BPA complexity를

$$
\mathcal{O}\left(n2^{|S|}\right)
$$

TEFN 전체를

$$
\mathcal{O}
\left(
CL2^{|S|}
\right)
$$

라고 제시합니다. 

이후

$$
C,\ |S|\ll L
$$

이고 고정된 상수라고 보고

$$
\mathcal{O}(L)
$$

이라고 단순화합니다.

### 기호 설명

* $L$: sequence length입니다.
* $C$: channel 수입니다.
* $|S|$: sample-space 크기입니다.
* $2^{|S|}$: BPA event-space 크기입니다.

### [중요한 해석]

이 주장은 **조건부로만 정확합니다.**

예를 들어 $|S|=3$이면

$$
2^{|S|}=8
$$

이지만 $|S|=10$이면

$$
2^{|S|}=1024
$$

입니다.

또 Traffic dataset은 $C=862$ 입니다. 

따라서 실제 scaling을 평가할 때는

$$
\boxed{
\mathcal{O}
\left(
CL2^{|S|}
\right)
}
$$

를 기억해야 하며 단순히 “ $\mathcal O(L)$ model”이라고 부르는 것은 불완전합니다.

---

# 7. 가장 중요한 그림 5개

## Figure 4 — 전체 TEFN architecture

**p.4, Fig.4**

가장 중요한 그림입니다. 입력을 정규화하고 time projection한 뒤 **Time BPA와 Channel BPA로 갈라졌다가 다시 합쳐지는 과정**을 한 번에 보여줍니다. 

핵심은 깊은 hierarchical network가 아니라

$$
\text{normalize}
\rightarrow
\text{expand}
\rightarrow
\text{two-view evidence}
\rightarrow
\text{sum}
$$

이라는 매우 얕은 구조입니다.

따라서 TEFN의 성능이 좋다면 “더 깊은 network”가 아니라 **representation design 자체가 효과가 있었다는 의미**가 됩니다.

---

## Figure 5 — BPA Module

**p.5, Fig.5**

TEFN의 이론적 novelty를 가장 잘 보여주는 그림입니다. 하나의 값이 여러 event representation으로 확장됩니다. 

Convolution이

$$
\text{many values}\rightarrow\text{compressed feature}
$$

를 수행한다면 TEFN BPA는 반대로

$$
\text{one representation}
\rightarrow
\text{multiple possible evidential representations}
$$

를 구성하려 합니다.

다만 바로 이 그림에서 생성된 값이 **정말 D-S mass의 normalization 조건을 만족하는가**가 명확하지 않다는 것이 핵심 비판점입니다.

---

## Figure 7 — Accuracy–Efficiency trade-off

**p.8, Fig.7**

Electricity-96 한 task에서 model file size, iteration time, MSE의 trade-off를 보여줍니다. 저자는 TEFN이 낮은 error를 유지하면서 작은 model size와 빠른 iteration을 갖는다고 주장합니다. 

### [해석]

TEFN의 가장 설득력 있는 결과 중 하나입니다.

다만 figure 하나만으로

> “모든 데이터에서 Transformer보다 효율적이다.”

라고 일반화해서는 안 됩니다.

또한 **binary model-file size**는 실제 parameter count, FLOPs, peak GPU memory와 완전히 동일한 지표가 아닙니다.

---

## Figure 9 — Hyperparameter/nonlinearity correlation

**p.10, Fig.9**

TEFN error가 learning rate나 $|S|$보다 저자들이 정의한 dataset nonlinearity와 더 높은 상관을 보인다고 분석합니다. 

저자의 핵심 결론은

$$
\text{nonlinearity}\uparrow
\quad\Rightarrow\quad
\text{TEFN error}\uparrow
$$

입니다.

이는 TEFN의 본질적인 linear-model ceiling을 잘 드러냅니다.

단, 논문에서 사용하는 “nonlinearity”는 수학적 비선형성 척도가 아니라 **단순 linear predictor의 forecasting error**로 정의된 proxy입니다. 

따라서 noise, predictability, horizon difficulty까지 함께 섞일 수 있습니다.

---

## Figure 10 — BPA interpretability

**p.12, Fig.10**

Time BPA와 Channel BPA에서 학습된 fuzzy membership function을 시각화합니다. 

저자는 이것을 통해 특정 time/channel이 어떤 fuzzy event에 높은 support를 주는지 이해할 수 있다고 주장합니다. 

### [해석]

이것은 black-box Transformer보다 “parameter-level visibility”가 좋다는 의미에서는 타당합니다.

하지만

$$
\text{visibility}\neq\text{causal interpretability}
$$

이고,

$$
\text{membership magnitude}\neq\text{calibrated probability}
$$

입니다.

즉 SHAP faithfulness test, perturbation test, probability calibration 등의 실험이 추가되어야 “높은 interpretability”라는 주장이 더 강해집니다.

---

# 8. 통계적으로 취약한 부분과 비교 불가능한 수치

| 항목                | 문제                                                       | 판단                                             |
| ----------------- | -------------------------------------------------------- | ---------------------------------------------- |
| Table II SOTA 비교  | mean ± std가 없음                                           | ⚠ 통계적 유의성 판단 불가                                |
| random seed       | 여러 initialization 반복 여부 불명확                              | ⚠ stochastic stability 판단 불가                   |
| significance test | paired test, bootstrap CI 등이 없음                          | ⚠ $0.001$ 수준 차이를 우열이라고 보기 어려움                  |
| Table IV variance | seed variance가 아니라 hyperparameter traversal 결과의 variance | ⚠ “모델 자체가 안정적”이라는 주장과 동일하지 않음                  |
| Fig.7 efficiency  | Electricity-96 단일 task                                   | ⚠ 전체 scaling으로 일반화 제한                          |
| model size        | binary save file size                                    | ⚠ parameter/FLOPs와 직접 비교 불가능                   |
| nonlinearity      | linear forecasting error를 nonlinearity proxy로 사용         | ⚠ noise/difficulty와 비선형성이 혼재                   |
| robustness        | 단일 Gaussian-noise 조건 중심                                  | ⚠ 다양한 SNR/OOD noise 미검증                        |
| uncertainty       | MSE/MAE만 평가                                              | ⚠ evidence uncertainty calibration이 검증되지 않음    |
| generalization    | 동일 benchmark 내부 train/test                               | ⚠ zero-shot/domain-transfer generalization이 아님 |

특히 Table IV의 variance가 대체로 $10^{-4}$ 수준이라는 것이 저자의 stability 근거입니다. 

그러나 이는

$$
\text{Var}_{\text{hyperparameter}}
$$

이며,

$$
\text{Var}_{\text{random seed}}
$$

또는

$$
\text{Var}_{\text{rolling test period}}
$$

가 아닙니다.

이 셋은 전혀 다른 종류의 robustness입니다.

---

# 9. Noise robustness 결과에서 특별히 주의할 점

저자는

```math
x'_{ij}
=
x_{ij}
+
\text{std}(x_i)\epsilon,
\qquad
\epsilon\sim\mathcal N(0,1)
```

을 사용합니다. 

### 기호 설명

* $x_{ij}$: channel $i$, time $j$의 값입니다.
* $\text{std}(x_i)$: 해당 channel의 표준편차입니다.
* $\epsilon$: 평균 0, 분산 1인 Gaussian noise입니다.

그런데 Table VI에서는 예를 들어 Exchange-720이

$$
\text{MSE}: 0.861\rightarrow0.092
$$

로 **noise를 넣었는데 약 89% 가까이 감소**합니다. 

ETTm2-720도

$$
0.407\rightarrow0.199
$$

입니다. 

이것은 단순한 “noise robustness”로 설명하기에는 매우 강한 변화입니다.

가능한 설명은 noise augmentation이 regularization처럼 작동했거나, noise가 train/test 어느 단계에 어떻게 들어갔는지가 성능에 크게 영향을 준 경우입니다.

따라서 이 결과는 **재현 전에는 강한 robustness evidence로 해석하면 안 됩니다.**

---

# 10. 문서가 답하지 않는 질문

1. Eq. (11)의 $wx+b$가 어떻게 항상 non-negative, normalized BPA가 되는가?
2. 연속값 regression에서 finite sample space $S$의 각 event는 구체적으로 무엇을 의미하는가?
3. 왜 최적 $|S|$가 특정 데이터에서 그 값을 가져야 하는지 이론적 선택 기준이 있는가?
4. Expectation fusion이 belief-theoretic 관점에서 어떤 optimality criterion을 만족하는가?
5. $m_T$와 $m_C$가 독립 evidence라는 가정은 타당한가?
6. 다른 random seed에서 결과의 분산은 얼마인가?
7. rolling-origin evaluation에서도 결과가 유지되는가?
8. 갑작스러운 regime shift와 concept drift에서는 어떻게 되는가?
9. missing value와 irregular sampling에 강한가?
10. 실제 predictive uncertainty interval을 출력할 수 있는가?
11. BPA uncertainty가 실제 forecast error와 calibration되어 있는가?
12. unseen dataset으로 zero-shot transfer가 가능한가?
13. channel 수가 수천~수만 개일 때에도 lightweight한가?
14. $L_{\text{in}}$을 크게 증가시켰을 때 fully connected projection의 parameter growth는 어떻게 되는가?
15. attention-based adaptive fusion을 동일 parameter budget에서 비교하면 어떻게 되는가?

---

# 11. 저자가 직접 인정한 한계

논문은 의외로 제한점을 비교적 명확하게 적습니다.

저자들은 TEFN이 “naive and unmodified model”이라고 표현하며,

* BPA 생성 방식이 아직 engineering적으로 충분히 개발되지 않았고,
* fuzzy logic이 최적의 구현이라고 보장할 수 없으며,
* linear TEFN이 nonlinear time series에 약하고,
* expectation fusion이 최적이라고 보장할 수 없으며,
* 긴 sequence와 많은 channel에서는 fully connected 방식의 parameter가 빠르게 증가할 수 있다고 인정합니다. 

이는 제가 판단하기에도 이 논문의 가장 중요한 limitation입니다.

---

# 12. 모델 일반화 성능을 중점적으로 평가하면

여기서는 **generalization이라는 용어를 세 가지로 분리해야 합니다.**

## 12.1 동일 dataset 내 미래 구간 일반화

$$
\text{Train history}
\rightarrow
\text{future test interval}
$$

TEFN은 이것을 Table II에서 검증했습니다.

→ **검증됨.**

---

## 12.2 다른 구조의 dataset에 대한 architecture generality

Electricity, Traffic, Weather, Exchange, ETT처럼 다른 특성을 가진 dataset에서 동일 backbone이 작동합니다.

→ **어느 정도 검증됨.**

저자들은 이를 generalizability로 해석합니다. 

---

## 12.3 완전히 unseen domain에 대한 zero-shot generalization

$$
D_{\text{train}}
\neq
D_{\text{test}}
$$

인데 새로운 dataset에서 parameter 재학습 없이 예측하는 문제입니다.

→ **TEFN에서는 검증하지 않았습니다.**

### 용어 설명 — **Zero-shot Forecasting**

새로운 dataset의 target 값을 이용해 다시 훈련하지 않고, 사전학습 모델을 그대로 적용하는 방식입니다.

따라서 Chronos, Moirai, TimesFM 등의 “generalization”과 TEFN의 “generalization”은 **직접 비교할 수 있는 동일한 개념이 아닙니다.**

---

# 13. 2020년 이후 관련 최신 연구 비교

시계열 장기 예측 연구는 대략 다음 방향으로 발전했습니다.

| 연구                                    | 핵심 아이디어                                                   | Generalization 관점                       | TEFN과의 관계                                                                                               |
| ------------------------------------- | --------------------------------------------------------- | --------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| **Informer (2021)**                   | ProbSparse attention으로 긴 sequence 비용 절감                   | dataset-specific                        | TEFN보다 복잡한 attention 계열 ([AAAI Publications][1])                                                        |
| **Autoformer (2021)**                 | decomposition + auto-correlation                          | dataset-specific                        | 명시적 trend/seasonality 사용, TEFN은 decomposition을 회피 ([NeurIPS Proceedings][2])                            |
| **Non-stationary Transformer (2022)** | stationarization + de-stationary attention                | distribution 변화 대응 개선                   | TEFN normalization 개선에 직접 참고 가능 ([NeurIPS Proceedings][3])                                              |
| **FEDformer (2022)**                  | frequency decomposition + Transformer                     | 장기 dependency 개선                        | TEFN에 frequency evidence source 추가 가능 ([Proceedings of Machine Learning Research][4])                   |
| **DLinear (2023)**                    | 단순 linear decomposition model                             | benchmark generality                    | “simple models can beat complex Transformers”라는 TEFN과 동일 흐름 ([AAAI Publications][5])                    |
| **PatchTST (2023)**                   | patch token + channel independence                        | transfer pretraining도 검증                | TEFN보다 cross-dataset representation learning 근거가 강함 ([ML Anthology][6])                                 |
| **TimesNet (2023)**                   | 1-D series를 multi-period 2-D representation으로 변환          | 여러 TS task 공통 backbone                  | TEFN보다 nonlinear pattern 표현력이 큼 ([ML Anthology][7])                                                     |
| **TSMixer (2023)**                    | time/feature MLP mixing                                   | 단순 구조로 multivariate 관계 학습               | TEFN의 time/channel two-view와 구조적으로 가까움 ([arXiv][8])                                                     |
| **iTransformer (2024)**               | variate 자체를 token으로 만들어 cross-variable attention          | variable generalization 강조              | TEFN의 channel evidence보다 훨씬 강한 nonlinear interaction ([ICLR Proceedings][9])                            |
| **TimeMixer (2024)**                  | multi-scale trend/seasonal mixing                         | scale 변화에 강한 representation             | TEFN을 multi-scale evidence로 발전시킬 좋은 비교 대상 ([ICLR Proceedings][10])                                      |
| **ModernTCN (2024)**                  | large receptive-field convolution                         | 여러 TS task에서 범용성                        | lightweight-general backbone이라는 TEFN 목표와 경쟁 ([ICLR Proceedings][11])                                    |
| **Time-LLM (2024)**                   | frozen LLM을 time-series representation으로 reprogramming    | few-shot/zero-shot                      | TEFN보다 일반화 범위가 넓지만 훨씬 무거움 ([ICLR Proceedings][12])                                                      |
| **TimesFM (2024)**                    | 대규모 pretraining + patched decoder                         | unseen dataset zero-shot                | 일반화 측면에서 TEFN보다 훨씬 강한 실험 설정 ([Proceedings of Machine Learning Research][13])                            |
| **Chronos (2024)**                    | time value tokenization + T5 계열 pretraining               | zero-shot forecasting                   | 정확도보다 portability/generalization 패러다임을 변화시킴 ([arXiv][14])                                               |
| **Moirai (2024)**                     | 27B observations 기반 universal forecasting Transformer     | cross-domain zero-shot                  | heterogeneous dataset generalization을 직접 다룸 ([Proceedings of Machine Learning Research][15])            |
| **Timer (2024)**                      | GPT-style generative TS pretraining                       | few-shot/general task transfer          | data-scarce generalization을 직접 목표로 함 ([Proceedings of Machine Learning Research][16])                   |
| **MOMENT (2024)**                     | general-purpose TS foundation model                       | limited-supervision transfer            | forecasting 외 task generalization까지 다룸 ([Proceedings of Machine Learning Research][17])                 |
| **TimeMixer++ (2025)**                | multi-scale × multi-resolution universal pattern modeling | 8 task 범용 분석                            | TEFN의 two-source representation을 훨씬 다양한 view로 확장한 방향 ([ICLR Proceedings][18])                           |
| **Moirai-MoE (2025)**                 | token-level sparse experts                                | heterogeneous pattern 자동 specialization | TEFN의 고정 BPA 대신 adaptive evidence expert로 발전시킬 수 있는 방향 ([Proceedings of Machine Learning Research][19]) |

---

# 14. 이 최신 연구 흐름에서 TEFN의 위치

2018~2022의 시계열 연구는 대체로

$$
\text{better attention}
+
\text{decomposition}
+
\text{frequency modeling}
$$

을 중심으로 발전했습니다.

2023년 DLinear 이후 중요한 변화가 생겼습니다. “Transformer가 항상 필요한가?”라는 질문이 본격적으로 제기됐고, 단순 linear structure가 일부 benchmark에서 복잡한 Transformer를 이길 수 있음이 제시됐습니다. ([AAAI Publications][5])

TEFN은 이 두 번째 흐름과 매우 잘 맞습니다.

즉,

$$
\boxed{
\text{모델을 크게 만들기보다 좋은 inductive bias를 설계하자}
}
$$

는 접근입니다.

하지만 2024~2025 이후에는 또 다른 축이 등장했습니다.

$$
\boxed{
\text{dataset-specific accuracy}
\rightarrow
\text{pretrained universal generalization}
}
$$

입니다.

TimesFM, Chronos, Moirai, Timer, MOMENT, Moirai-MoE가 대표적입니다. ([Proceedings of Machine Learning Research][13])

이 관점에서는 TEFN의 현재 결과만으로 “general-purpose forecasting model”이라고 부르기에는 근거가 부족합니다.

---

# 15. TEFN이 미래 연구에 미칠 수 있는 영향

TEFN의 가장 중요한 연구적 의미는 Transformer를 대체했다는 것이 아니라,

> **시계열 representation 자체를 probability가 아니라 evidential/fuzzy representation으로 만들 수 있다는 방향을 제안했다는 것**

입니다.

이는 기존 모델과 상당히 orthogonal한 아이디어입니다.

예를 들어 앞으로

$$
\text{PatchTST}
+
\text{BPA}
$$

$$
\text{iTransformer}
+
\text{BPA channel evidence}
$$

$$
\text{TimeMixer}
+
\text{multi-scale BPA}
$$

$$
\text{Moirai-MoE}
+
\text{evidential expert routing}
$$

같은 형태의 연구가 가능합니다.

즉 TEFN은 반드시 standalone forecasting backbone으로 남을 필요가 없고 **evidential representation/fusion layer**로 다른 architecture에 이식되는 쪽이 오히려 더 영향력이 클 수 있습니다.

---

# 16. 일반화 성능을 실제로 향상시키기 위한 후속 연구

## 16.1 가장 먼저 수정해야 할 것 — Valid BPA 보장

현재 Eq. (11)의

$$
m_k=wx+b
$$

대신 최소한

$$
z_k=w_kx+b_k
$$

```math
m_k
=
\frac{\exp(z_k)}
{\sum_{r=1}^{2^{|S|}}\exp(z_r)}
```

처럼 만들면

$$
m_k\ge0,
\qquad
\sum_km_k=1
$$

을 명시적으로 만족시킬 수 있습니다.

단, 이렇게 하면 일반 probability distribution과 BPA의 구분을 다시 엄밀하게 설계해야 합니다.

더 좋은 방법은 singleton뿐 아니라 composite event mass를 명시적으로 parameterize하는 것입니다.

---

# 16.2 Adaptive Nonlinear BPA

저자도 future work로 adaptive nonlinear membership function을 언급합니다. 

예를 들어

```math
m_k(x)
=
a_k\tanh(w_kx+b_k)+c_k
```

또는 spline membership을 사용할 수 있습니다.

단순 ReLU보다는 smooth nonlinear function이 더 적절할 가능성이 있으며 실제 ablation에서도 Tanh가 많은 조건에서 ReLU보다 낫다고 보고합니다. 

---

# 16.3 Fixed Sum Fusion → Adaptive Fusion

현재는

```math
\hat y
=
\hat y_T+\hat y_C
```

입니다.

이를

```math
\hat y
=
\alpha(X)\hat y_T
+
\left(1-\alpha(X)\right)\hat y_C
```

로 만들 수 있습니다.

여기서

$$
0\le\alpha(X)\le1
$$

는 입력 상태에 따라 달라지는 gating function입니다.

그러면 어떤 regime에서는 time evidence를, 다른 regime에서는 channel evidence를 더 신뢰할 수 있습니다.

이는 **domain shift에서 특히 중요한 개선 방향**입니다.

---

# 16.4 Multi-source를 실제로 확장

TEFN은 사실 source가 두 개뿐입니다.

$$
\{T,C\}
$$

입니다.

이를

$$
\{
T,\;
C,\;
F,\;
S,\;
R
\}
$$

로 확장할 수 있습니다.

* $T$: temporal evidence
* $C$: channel evidence
* $F$: frequency evidence
* $S$: multi-scale evidence
* $R$: regime/context evidence

TimeMixer++의 multi-scale/multi-resolution 철학과 결합하면 특히 강력할 가능성이 있습니다. ([ICLR Proceedings][18])

---

# 16.5 Distribution Shift 대응

TEFN normalization만으로는

$$
P_{\text{train}}(X,Y)
\neq
P_{\text{test}}(X,Y)
$$

문제를 충분히 다루지 못합니다.

Non-stationary Transformer가 보여준 것처럼 단순 stationarization 이후에도 원래의 non-stationary information을 다시 예측 과정에 넣는 방법이 필요합니다. ([NeurIPS Proceedings][3])

TEFN에서는 이를

$$
m^{\text{source}}
+
m^{\text{shift}}
$$

처럼 **shift 자체를 별도의 evidence source로 모델링하는 방법**으로 발전시킬 수 있습니다.

---

# 16.6 Pretrained TEFN

가장 중요한 일반화 연구 방향 중 하나입니다.

현재는 dataset별

```math
\theta_D
=
\text{Train}(D)
```

방식입니다.

향후에는 여러 dataset을 묶어

```math
\theta^*
=
\arg\min_\theta
\sum_{d=1}^{M}
\mathcal L(D_d;\theta)
```

로 pretraining한 뒤 새로운 dataset $D_{\text{new}}$에

```math
\hat Y_{\text{new}}
=
f_{\theta^*}(X_{\text{new}})
```

를 적용하는 **zero-shot TEFN**을 검증해야 합니다.

TimesFM, Chronos, Moirai가 이미 이 평가 기준을 상당히 높여 놓았습니다. ([Proceedings of Machine Learning Research][13])

---

# 17. 불확실성 모델이라면 반드시 추가되어야 할 평가

현재 TEFN은 MSE와 MAE 중심입니다. 논문의 loss/metric 정의도 Eq. (14)–(15)의 MSE/MAE입니다. 

하지만 모델이 정말 “uncertainty”를 모델링한다고 주장하려면 최소한 다음이 필요합니다.

예측구간 coverage:

```math
\text{PICP}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf 1
\left(
y_i\in[L_i,U_i]
\right)
```

그리고 probabilistic forecasting이라면 CRPS, NLL, interval width 등을 같이 평가하는 것이 좋습니다.

즉

$$
\text{BPA representation}
\neq
\text{validated predictive uncertainty}
$$

입니다.

현재 논문은 첫 번째는 제안했지만 두 번째까지 입증하지는 않았습니다.

---

# 18. 저자들의 결론과 후속 연구 계획

## [저자 결론]

저자들은 TEFN이 time/channel을 서로 다른 정보원으로 다루고, evidence theory의 BPA를 사용해 정보를 선택·융합함으로써 대규모 시계열에서도 효율적인 forecasting이 가능하다고 결론 내립니다. 또한 parameter efficiency, robustness, interpretability를 주요 실용적 장점으로 제시합니다. 

## [저자가 논문 안에서 제시한 후속 방향]

저자의 limitation/future discussion을 종합하면:

* adaptive nonlinear membership function,
* expectation 이외의 fusion,
* concat/attention fusion,
* sampling/kernel 방식의 계산 최적화,
* 더 효과적인 parameter initialization

이 주요 방향입니다. 

---

# 19. 최종 연구자 평가

TEFN의 가장 강한 부분은 **“복잡한 모델이 아니어도 representation을 바꾸면 충분히 경쟁력 있는 forecasting이 가능하다”**는 점입니다.

반면 가장 약한 부분은 이름에서 강조되는 **Evidence Theory의 엄밀한 수학적 구현과 실제 uncertainty validation**입니다.

현재 TEFN은

$$
\boxed{
\text{Evidence-inspired lightweight forecaster}
}
$$

라고 부르는 것이

$$
\boxed{
\text{fully validated evidential uncertainty model}
}
$$

이라고 부르는 것보다 정확합니다.

성능 또한 “전 benchmark에서 압도적 SOTA”가 아니라,

> **MAE에서는 매우 강하고, 일부 ETT/Exchange 조건에서 우수하며, 전체적으로는 낮은 계산비용과 예측성능 사이의 trade-off가 우수한 모델**

이라는 해석이 Table II와 가장 잘 일치합니다.

그리고 일반화 관점에서 TEFN의 다음 단계는 단순히 BPA layer를 더 깊게 만드는 것이 아니라,

$$
\boxed{
\text{valid evidential formulation}
+
\text{adaptive nonlinear BPA}
+
\text{distribution-shift adaptation}
+
\text{cross-dataset pretraining}
+
\text{zero-shot evaluation}
}
$$

을 함께 검증하는 것입니다.

이 단계까지 성공한다면 TEFN은 단순한 lightweight LTSF model을 넘어 **interpretable time-series foundation architecture의 한 구성요소**로 발전할 가능성이 있습니다.

---

# 20. 참고한 자료 및 출처 제목

### 분석 대상 원문

* **Tianxiang Zhan et al., “Time Evidence Fusion Network: Multi-source View in Long-Term Time Series Forecasting.”** 첨부 PDF / arXiv:2405.06419.  ([arXiv][20])

### 2020년 이후 비교 연구

1. **“Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting”**, AAAI 2021. ([AAAI Publications][1])
2. **“Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting”**, NeurIPS 2021. ([NeurIPS Proceedings][2])
3. **“Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting”**, NeurIPS 2022. ([NeurIPS Proceedings][3])
4. **“FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting”**, ICML 2022. ([Proceedings of Machine Learning Research][4])
5. **“Are Transformers Effective for Time Series Forecasting?”**, AAAI 2023 — DLinear/LTSF-Linear. ([AAAI Publications][5])
6. **“A Time Series Is Worth 64 Words: Long-Term Forecasting with Transformers”**, ICLR 2023 — PatchTST. ([ML Anthology][6])
7. **“TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis”**, ICLR 2023. ([ML Anthology][7])
8. **“TSMixer: An All-MLP Architecture for Time Series Forecasting”**, TMLR 2023. ([arXiv][8])
9. **“iTransformer: Inverted Transformers Are Effective for Time Series Forecasting”**, ICLR 2024. ([ICLR Proceedings][9])
10. **“TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting”**, ICLR 2024. ([ICLR Proceedings][10])
11. **“ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis”**, ICLR 2024. ([ICLR Proceedings][11])
12. **“Time-LLM: Time Series Forecasting by Reprogramming Large Language Models”**, ICLR 2024. ([ICLR Proceedings][12])
13. **“A Decoder-Only Foundation Model for Time-Series Forecasting”**, ICML 2024 — TimesFM. ([Proceedings of Machine Learning Research][13])
14. **“Chronos: Learning the Language of Time Series”**, 2024. ([arXiv][14])
15. **“Unified Training of Universal Time Series Forecasting Transformers”**, ICML 2024 — Moirai. ([Proceedings of Machine Learning Research][15])
16. **“Timer: Generative Pre-trained Transformers Are Large Time Series Models”**, ICML 2024. ([Proceedings of Machine Learning Research][16])
17. **“MOMENT: A Family of Open Time-series Foundation Models”**, ICML 2024. ([Proceedings of Machine Learning Research][17])
18. **“TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis”**, ICLR 2025. ([ICLR Proceedings][18])
19. **“Moirai-MoE: Empowering Time Series Foundation Models with Sparse Mixture of Experts”**, ICML 2025. ([Proceedings of Machine Learning Research][19])

연구 재현 관점의 다음 단계로는 **① Eq. (11)의 BPA normalization을 실제 코드에서 확인하고, ② random seed 10회 이상 반복, ③ rolling-origin/OOD split, ④ uncertainty calibration을 추가한 뒤 TEFN의 “generalization” 주장을 다시 검증하는 것**이 가장 가치가 큽니다.

[1]: https://ojs.aaai.org/index.php/AAAI/article/view/17325?utm_source=chatgpt.com "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting | Proceedings of the AAAI Conference on Artificial Intelligence"
[2]: https://proceedings.neurips.cc/paper/2021/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html?utm_source=chatgpt.com "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting"
[3]: https://proceedings.neurips.cc/paper_files/paper/2022/hash/4054556fcaa934b0bf76da52cf4f92cb-Abstract-Conference.html?utm_source=chatgpt.com "Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting"
[4]: https://proceedings.mlr.press/v162/zhou22g.html?utm_source=chatgpt.com "FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting"
[5]: https://ojs.aaai.org/index.php/AAAI/article/view/26317?utm_source=chatgpt.com "Are Transformers Effective for Time Series Forecasting? | Proceedings of the AAAI Conference on Artificial Intelligence"
[6]: https://mlanthology.org/iclr/2023/nie2023iclr-time/?utm_source=chatgpt.com "A Time Series Is Worth 64 Words: Long-Term Forecasting with Transformers | ML Anthology"
[7]: https://mlanthology.org/iclr/2023/wu2023iclr-timesnet/?utm_source=chatgpt.com "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis | ML Anthology"
[8]: https://arxiv.org/abs/2303.06053?utm_source=chatgpt.com "TSMixer: An All-MLP Architecture for Time Series Forecasting"
[9]: https://proceedings.iclr.cc/paper_files/paper/2024/hash/2ea18fdc667e0ef2ad82b2b4d65147ad-Abstract-Conference.html?utm_source=chatgpt.com "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting"
[10]: https://proceedings.iclr.cc/paper_files/paper/2024/hash/a7ac8a21e5a27e7ab31a5f42a0117bdb-Abstract-Conference.html?utm_source=chatgpt.com "TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting"
[11]: https://proceedings.iclr.cc/paper_files/paper/2024/hash/86b1437c1e4c3b3c4debff98234a67e7-Abstract-Conference.html?utm_source=chatgpt.com "ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis"
[12]: https://proceedings.iclr.cc/paper_files/paper/2024/hash/680b2a8135b9c71278a09cafb605869e-Abstract-Conference.html?utm_source=chatgpt.com "Time-LLM: Time Series Forecasting by Reprogramming Large Language Models"
[13]: https://proceedings.mlr.press/v235/das24c.html?utm_source=chatgpt.com "A decoder-only foundation model for time-series forecasting"
[14]: https://arxiv.org/abs/2403.07815?utm_source=chatgpt.com "Chronos: Learning the Language of Time Series"
[15]: https://proceedings.mlr.press/v235/woo24a.html?utm_source=chatgpt.com "Unified Training of Universal Time Series Forecasting Transformers"
[16]: https://proceedings.mlr.press/v235/liu24cb.html?utm_source=chatgpt.com "Timer: Generative Pre-trained Transformers Are Large Time Series Models"
[17]: https://proceedings.mlr.press/v235/goswami24a.html?utm_source=chatgpt.com "MOMENT: A Family of Open Time-series Foundation Models"
[18]: https://proceedings.iclr.cc/paper_files/paper/2025/hash/2b187165e28fdfdc0ffb34d1bfff2b0c-Abstract-Conference.html?utm_source=chatgpt.com "TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis"
[19]: https://proceedings.mlr.press/v267/liu25an.html?utm_source=chatgpt.com "Moirai-MoE: Empowering Time Series Foundation Models with Sparse Mixture of Experts"
[20]: https://arxiv.org/abs/2405.06419 "Time Evidence Fusion Network: Multi-source View in Long-Term Time Series Forecasting"
