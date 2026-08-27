# HPMixer: Hierarchical Patching for Multivariate Time Series Forecasting

아래 분석은 **첨부된 arXiv v2 PDF의 본문·표·그림을 우선 근거**로 하고, 2020년 이후 비교 연구는 학회/출판사/arXiv의 원문을 별도로 확인했습니다. 페이지 표시는 별도 언급이 없으면 **첨부 PDF의 1–18쪽 기준**입니다. 참고로 이 연구는 첨부본에서는 arXiv 원고이지만, 현재는 **PAKDD 2026 Proceedings Part III, pp. 42–53, DOI 10.1007/978-981-92-1465-5_4**로 출판된 것이 Springer와 DBLP에서 확인됩니다. ([Springer][1])

아래에서는 **[저자 보고]**와 **[해석]**을 의도적으로 분리했습니다. 논문이 직접 입증하지 않은 부분은 가능한 한 “가능성”, “해석”, “미확인”으로 표시합니다.

---

# 1. Executive Summary — 10문장 이내

1. **HPMixer는 장기 다변량 시계열을 하나의 복잡한 신호로 직접 예측하기보다, 주기성(periodicity)과 잔차(residual dynamics)를 분리하여 각각 다른 구조로 학습한 뒤 합치는 모델**입니다. [p.1–4, Fig.1] 
2. 주기성은 CycleNet의 learnable recurrent cycle을 기반으로 하되 **MLP를 추가하여 단순한 주기 반복보다 표현력을 높이는 것**이 핵심입니다. [p.4–6, Fig.2–3, Eq.(1)] 
3. 잔차는 **Learnable Stationary Wavelet Transform(LSWT)**으로 다중 주파수 성분으로 분해해 시간 이동에 민감한 DWT의 약점을 줄이고, 데이터별 필터를 학습하도록 설계합니다. [p.3, p.6–7, Eq.(2)–(3)]  
4. LSWT 성분에 대해 **coarse patch → fine patch의 2단계 비중첩 hierarchical patching**을 적용하여 장기 구조와 국소 변동을 함께 포착합니다. [p.7–8, Eq.(4)–(6), Fig.3] 
5. 또한 coarse patch 사이에서 **channel-mixing encoder**를 사용하여 변수 간 의존성을 명시적으로 학습한다는 점이 channel-independent 모델과 다릅니다. [p.7, Fig.3(b)] 
6. 7개 benchmark × 4개 horizon에서 HPMixer는 **MSE 23회, MAE 21회 1위**를 기록하며 특히 ETTm1, ETTm2, Weather에서 강한 결과를 보고합니다. [p.9–10, Table 1] 
7. 그러나 ETTh1·ETTh2에서는 SimpleTM 등과 우열이 혼재하고, Traffic에서는 iTransformer 계열이 더 강하여 **“모든 다변량 데이터에서 우월하다”는 결론은 지지되지 않습니다.** [p.9–10, Table 1] 
8. Ablation은 cycle module, trainable SWT, hierarchical patching 각각이 성능에 기여한다고 제시하지만, **표준편차·신뢰구간·유의성 검정이 없고 baseline 결과 일부를 다른 논문에서 가져왔기 때문에 통계적 우월성의 강도는 제한적으로 해석해야 합니다.** [p.9–11, Table 1–2]  
9. 특히 모델의 진짜 다음 과제는 저자들도 인정하듯 **고정 cycle length를 적응적으로 바꾸고, Traffic 같은 초고차원 데이터에서 channel interaction을 더 확장성 있게 만드는 것**이며, 여기에 domain shift·zero-shot·cross-dataset 평가를 추가해야 일반화 성능을 강하게 주장할 수 있습니다. [p.11, Conclusion] 

---

# 1-1. 연구 목적과 필요성

## 해결하려는 근본 문제

장기 다변량 시계열은 보통 다음과 같은 성분이 섞여 있습니다.

```math
X_t
=
X_t^{\text{periodic}}
+
X_t^{\text{trend}}
+
X_t^{\text{irregular}}
+
\epsilon_t.
```

이 식은 제가 논문을 이해하기 쉽게 나타낸 **개념적 표현**이며 논문에 동일한 형태로 제시된 공식은 아닙니다.

HPMixer의 핵심 문제의식은 다음과 같습니다.

**[저자 보고]** 장기예측에서는 안정적인 주기성이 매우 유용하지만, 실제 데이터에는 장기 trend, irregular fluctuation, structural change 등 주기성으로 설명되지 않는 잔차가 존재합니다. 따라서 주기성만 잘 모델링해서는 장기 예측 오류가 누적됩니다. [p.1–2] 

**용어 — Periodicity(주기성)**
일정 시간 간격마다 비슷한 패턴이 반복되는 성질입니다. 예를 들어 15분 단위 전력 데이터에서 96 step은 $96\times15$분 = 24시간이므로 하루 주기에 해당합니다.

**용어 — Residual(잔차)**
HPMixer에서 residual은 단순한 “백색잡음”과 동일하지 않습니다. 주기성으로 설명하고 남은 trend, irregular fluctuation, structural variation 등 **예측 가능한 정보까지 포함한 잔여 성분**입니다. 저자들도 residual을 단순 noise로 취급하면 안 된다고 명시합니다. [p.1–2] 

따라서 논문의 설계 철학은

```math
\boxed{
\text{Forecast}
=
\text{Periodic Forecast}
+
\text{Structured Residual Forecast}
}
```

라고 요약할 수 있습니다.

이것이 HPMixer의 가장 중요한 아이디어입니다.

---

# 2. 핵심 주장과 근거

| 핵심 주장                                      | 저자가 제시한 근거                                  | 위치                  | 제 판단                                                     |
| ------------------------------------------ | ------------------------------------------- | ------------------- | -------------------------------------------------------- |
| 주기성과 residual을 함께 모델링해야 한다                 | Cycle branch + LSWT residual branch 결합      | p.1–4, Fig.1        | **강하게 설득력 있음.** decomposition inductive bias가 분명함        |
| 단일 주기 모델만으로 채널별 상이한 주기성을 충분히 설명하기 어렵다      | ETTm2의 채널별 ACF 차이                           | p.5–6, Fig.2        | **부분적으로 지지.** ACF 차이는 보여주지만 MLP가 이를 해결한다는 직접적 인과 증거는 제한적 |
| nonlinear MLP가 cycle modeling을 개선한다        | w/o MLP ablation                            | p.10–11, Table 2    | 대체로 지지되나 유의성 검정 없음                                       |
| LSWT가 raw time-domain 또는 fixed SWT보다 효과적이다 | w/o SWT / w/o trainable SWT                 | p.10–11, Table 2    | 대체로 지지. 다만 ECL에서는 trainable SWT가 확실히 우월하지 않음             |
| two-level patching이 one-level보다 좋다         | one-level ablation                          | p.10–11, Table 2    | benchmark상 지지됨                                           |
| HPMixer가 SOTA/competitive이다                | 7개 benchmark, 4 horizons, MSE 23/MAE 21 win | p.9–10, Table 1     | **일부 benchmark에서 강함. 전체 지배는 아님**                         |
| non-overlapping patching은 효율적이다            | SimpleTM 대비 iteration speed                 | p.14–16, Table 5    | 실행시간 결과는 흥미롭지만 hardware 정보 부족                            |
| cycle/patch hyperparameter에 robust하다       | ECL/ETTm1 sensitivity analysis              | p.15–16, Fig.4      | **“robust”라는 표현에는 주의가 필요함**                              |
| 초고차원 channel dependence가 한계다               | Traffic에서 상대적 약세                            | p.10–11, Conclusion | 저자 스스로 인정하며 결과와 일치                                       |

Table 1의 전체 benchmark와 Win Count는 논문에서 5개 random seed에 대한 HPMixer 결과를 사용하며, baseline 수치는 기존 연구에서 가져왔다고 명시되어 있습니다. 

---

# 2-1. 문제, 방법, 수식, 모델 구조 상세 분석

## 2-1-1. 예측 문제의 수학적 정의

논문은 $C$개의 변수가 관찰되는 multivariate time series를

$$
\mathbf{X}
\in
\mathbb{R}^{C\times L}
$$

로 정의합니다.

목표는

$$
\hat f:
\mathbb{R}^{C\times L}
\rightarrow
\mathbb{R}^{C\times H}
$$

를 학습하여

```math
\hat{\mathbf Y}
=
\hat f(\mathbf X)
\in
\mathbb{R}^{C\times H}
```

를 생성하는 것입니다. [p.3–4] 

기호는 다음과 같습니다.

* $C$: channel 또는 variable 개수입니다.
* $L$: look-back window, 즉 과거 몇 step을 입력으로 볼 것인지입니다.
* $H$: forecasting horizon, 미래 몇 step까지 예측할 것인지입니다.
* $\mathbf X$: 입력 multivariate time series입니다.
* $\hat{\mathbf Y}$: 모델이 예측한 미래 시계열입니다.
* $\mathbb R^{C\times L}$: 실수값으로 이루어진 $C\times L$ 크기의 행렬 공간입니다.

**용어 — Look-back window**
예를 들어 $L=96$이라면 가장 최근 96개 관측값을 가지고 미래를 예측합니다.

**용어 — Forecasting horizon**
$H=720$이면 미래 720 step을 한 번에 예측하는 장기 예측 문제입니다.

---

# 2-1-2. 전체 HPMixer 구조

Fig.1의 구조는 개념적으로 다음과 같습니다.

$$
\mathbf X
\longrightarrow
\begin{cases}
\text{Cycle branch} \\
\text{Residual branch}
\end{cases}
\longrightarrow
\hat{\mathbf Y}_{\text{period}}
+
\hat{\mathbf Y}_{\text{residual}}
\longrightarrow
\hat{\mathbf Y}.
$$

[p.4, Fig.1] 

Residual branch는 다시

$$
\mathbf X_{\text{res}}
\xrightarrow{\text{LSWT}}
\{A_k,D_1,\ldots,D_k\}
\xrightarrow{\text{Hierarchical Patching}}
\xrightarrow{\text{Channel Mixing}}
\xrightarrow{\text{Coarse-Fine Mixer}}
\xrightarrow{\text{ISWT}}
\hat{\mathbf X}_{\text{recon}}
$$

순서로 진행됩니다.

Fig.1에는 입력 직후 **RevIN normalization**과 마지막의 inverse RevIN도 표시됩니다.

**용어 — RevIN(Reversible Instance Normalization)**
각 시계열 샘플의 평균과 분산을 이용하여 입력을 정규화하고, 예측 후 원래 scale로 되돌리는 방법입니다. 분포의 level/scale 변화에 대한 민감도를 줄이기 위한 기법입니다. 다만 HPMixer 본문은 Fig.1에 RevIN을 표시하면서도 이에 대한 방법론적 설명은 거의 제공하지 않습니다. 이 부분은 재현성 측면에서 아쉽습니다.

---

# 2-1-3. Learnable Mixing Cycle Module

CycleNet에서 가져온 핵심 parameter는

$$
\mathbf Q
\in
\mathbb{R}^{W\times C}
$$

입니다. [p.6] 

여기에서

* $W$: cycle length
* $C$: channel 수
* $\mathbf Q[:,c]$: $c$번째 channel의 학습 가능한 주기 template

입니다.

$W$ 자체는 학습하지 않고 **training data의 ACF에서 dominant peak를 찾아 미리 정합니다.** [p.6] 

### ACF의 의미

lag $\tau$에서의 autocorrelation은 개념적으로

```math
\rho(\tau)
=
\frac{
\sum_t
(x_t-\bar x)
(x_{t-\tau}-\bar x)
}{
\sum_t
(x_t-\bar x)^2
}
```

와 같이 생각할 수 있습니다.

이 식은 ACF에 대한 표준적 정의를 설명하기 위한 식이며 HPMixer 논문에 공식으로 제시되지는 않습니다.

$\rho(96)$가 크면

> 현재 값과 96 step 전 값이 비슷한 움직임을 반복한다

는 뜻입니다.

Fig.2에서 ETTm2[0]은 96 step마다 비교적 뚜렷한 peak를 보이지만 다른 channel에서는 그 구조가 약해집니다. [p.5–6, Fig.2]  

---

## MLP를 통한 cycle refinement — Eq.(1)

논문 Eq.(1)은

```math
\hat{\mathbf X}_{\text{period}}
=
\text{LayerNorm}
\left(
\text{MLP}
\left(
\mathbf X_{\text{period}}
\right)
\right)
\in
\mathbb R^{W\times C}
```

입니다. [p.6, Eq.(1)] 

* $\mathbf X_{\text{period}}$: recurrent cycle module이 만든 초기 주기 표현
* $\hat{\mathbf X}_{\text{period}}$: MLP로 정제된 주기 표현
* $\text{MLP}$: 여러 fully-connected layer와 nonlinear activation으로 구성된 함수
* $\text{LayerNorm}$: feature representation의 scale을 정규화하는 layer normalization
* $W$: cycle length
* $C$: channel 개수

**설계 의도**는 단순히 저장한 cycle을 반복하는 것보다 nonlinear mapping을 이용해 더 유연한 periodic representation을 만드는 것입니다.

### 중요한 해석상의 주의점

논문은 이를 “channel-wise MLP”라고 표현합니다.

따라서 이 MLP가 **명시적인 cross-channel interaction을 담당한다고 해석하면 과도합니다.**

Fig.3에서 명시적인 channel dependency를 학습하는 부분은 별도의 **Channel-Mixing Encoder**입니다.

즉,

$$
\boxed{
\text{Cycle MLP}
\neq
\text{main cross-channel dependency module}
}
$$

로 이해하는 것이 더 안전합니다.

---

# 2-1-4. Learnable Stationary Wavelet Transform

일반 DWT는 downsampling을 수행하기 때문에 input을 몇 step 이동시키면 wavelet coefficient가 크게 달라질 수 있습니다.

LSWT는 downsampling을 제거하여 모든 scale에서 temporal resolution을 유지합니다. [p.3] 

## Eq.(2): Wavelet decomposition

논문은

```math
(A_j,D_j)
=
\mathcal W_j(\mathbf X)
=
\left(
\mathbf X*h_0^{(j)},
\mathbf X*h_1^{(j)}
\right),
\qquad
j=1,\ldots,k
```

로 나타냅니다. [p.6, Eq.(2)] 

여기에서

* $A_j$: $j$번째 level의 approximation coefficient
* $D_j$: $j$번째 level의 detail coefficient
* $h_0^{(j)}$: $j$번째 scale의 low-pass analysis filter
* $h_1^{(j)}$: $j$번째 scale의 high-pass analysis filter
* $*$: convolution
* $k$: wavelet decomposition level

입니다.

### Approximation coefficient

저주파 성분을 강조하므로 대체로

* baseline
* 장기 변화
* 부드러운 trend

에 대응합니다.

### Detail coefficient

고주파 성분을 강조하므로

* 국소 fluctuation
* 급격한 변화
* 빠른 oscillation

에 대응할 가능성이 큽니다.

다만 **고주파 = noise라는 등식은 성립하지 않습니다.**

고주파에도 예측 가능한 공정/물리 정보가 들어 있을 수 있습니다.

---

# Eq.(3): Reconstruction

```math
\mathbf X
=
\mathcal W^{-1}
(A_k,D_1,\ldots,D_k)
=
A_k*g_0^{(k)}
+
\sum_{j=1}^{k}
D_j*g_1^{(j)}.
```

[p.6, Eq.(3)] 

여기에서

* $g_0^{(j)}$: synthesis low-pass filter
* $g_1^{(j)}$: synthesis high-pass filter
* $\mathcal W^{-1}$: inverse wavelet transform
* $A_k$: 최종 scale의 approximation
* $D_1,\ldots,D_k$: 각 scale detail

입니다.

HPMixer의 중요한 차이는 $h$와 관련 filter들을 fixed wavelet로만 두지 않고 **학습 가능하게 만든다는 점**입니다. 저자는 이것이 dataset-specific spectral structure를 더 잘 분리한다고 주장합니다. [p.7, p.11]  

**용어 — Shift invariance**
input이 한두 step 옮겨졌다고 feature representation이 완전히 바뀌지 않는 성질입니다. 실제 센서 신호에서는 event의 시작 시점이 정확히 grid에 맞지 않기 때문에 유용합니다.

---

# 2-1-5. Hierarchical Non-Overlapping Patching

LSWT의 각 component

$$
\mathbf X_d\in\mathbb R^{C\times L}
$$

에 대해 먼저 coarse patching을 합니다.

```math
\mathbf X_{\text{coarse}}
=
\text{Patch}_{\text{co}}(\mathbf X_d),
```

$$
\mathbf X_{\text{coarse}}
\in
\mathbb R^{C\times P_{\text{co}}\times N_{\text{co}}},
$$

```math
N_{\text{co}}
=
\left\lfloor
\frac{L}{P_{\text{co}}}
\right\rfloor .
```

[p.7] 

* $P_{\text{co}}$: coarse patch 하나의 길이
* $N_{\text{co}}$: coarse patch 개수

그 안을 다시 fine patch로 나눕니다.

```math
\mathbf X_{\text{fine},n}
=
\text{Patch}_{\text{fi}}
(\mathbf X_{\text{coarse},n}),
```

$$
\mathbf X_{\text{fine},n}
\in
\mathbb R^{C\times P_{\text{fi}}\times N_{\text{fi}}},
$$

```math
N_{\text{fi}}
=
\left\lfloor
\frac{P_{\text{co}}}{P_{\text{fi}}}
\right\rfloor .
```

[p.7] 

### 왜 두 단계인가?

Fine patch:

$$
\text{short-range/local dynamics}
$$

Coarse patch:

$$
\text{long-range/context dynamics}
$$

를 담당하도록 구조적 prior를 주는 것입니다.

**용어 — Inductive bias**
모델에게 데이터를 보기 전부터 특정 구조가 유용할 것이라는 가정을 넣는 것입니다. HPMixer에서는 “시간 신호에는 여러 scale의 구조가 존재한다”는 가정이 inductive bias입니다.

---

# 2-1-6. Channel-Mixing Encoder

coarse patch를 만든 직후 Transformer 계열 encoder를 사용하여 channel 간 관계를 학습합니다. [p.7, Fig.3(b)] 

논문은 attention의 상세식을 직접 적지는 않지만, Transformer attention의 표준 형태는

```math
\text{Attention}(Q,K,V)
=
\text{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
```

입니다.

이 식은 **HPMixer 논문에 새로 제시된 식이 아니라**, Fig.3(b)의 multi-head attention을 이해하기 위한 표준 Transformer 식입니다.

* $Q$: Query
* $K$: Key
* $V$: Value
* $d_k$: key vector dimension
* $QK^\top$: 두 token 간 similarity
* $\text{softmax}$: similarity를 가중치로 변환

HPMixer에서는 시간점 자체보다 **channel 간 정보교환을 수행한다는 점**이 핵심입니다.

---

# 2-1-7. Fine-Patch Mixer — Eq.(4)

```math
\hat{\mathbf X}_{\text{fine}}
=
\text{MLP}_{\text{fi}}
(\mathbf X_{\text{fine}})
+
\mathbf X_{\text{fine}}
```

$$
\hat{\mathbf X}_{\text{fine}}
\in
\mathbb R^{
C\times
N_{\text{co}}\times
N_{\text{fi}}\times
P_{\text{fi}}
}.
$$

[p.8, Eq.(4)] 

여기에서 $+\mathbf X_{\text{fine}}$은 **residual connection**입니다.

**용어 — Residual connection**
MLP가 입력 전체를 새로 만들게 하지 않고

$$
\text{output}=\text{input}+\text{correction}
$$

형태로 학습하는 것입니다. 깊은 network의 optimization을 안정화하는 역할을 합니다.

---

# 2-1-8. Coarse-Patch Mixer — Eq.(5)–(6)

먼저 fine 정보를 펼쳐 coarse scale 전체에서 mixing합니다.

```math
\tilde{\mathbf X}_{\text{coarse}}
=
\text{MLP}_{\text{flat}}
(\hat{\mathbf X}_{\text{fine}})
+
\hat{\mathbf X}_{\text{fine}}.
```

[p.8, Eq.(5)] 

그 후 다시 coarse structure를 정제합니다.

```math
\hat{\mathbf X}_{\text{coarse}}
=
\text{MLP}_{\text{patch}}
(\tilde{\mathbf X}_{\text{coarse}})
+
\tilde{\mathbf X}_{\text{coarse}},
```

$$
\hat{\mathbf X}_{\text{coarse}}
\in
\mathbb R^{C\times N_{\text{co}}\times P_{\text{co}}}.
$$

[p.8, Eq.(6)] 

즉 HPMixer는

$$
\boxed{
\text{local mixing}
\rightarrow
\text{global/coarse mixing}
}
$$

순서를 사용합니다.

---

# 2-1-9. 최종 prediction — Eq.(7)

ISWT로 시간영역으로 돌아온 representation을

$$
\hat{\mathbf X}_{\text{recon}}
\in
\mathbb R^{C\times L}
$$

라 하면 최종 prediction은

```math
\hat{\mathbf Y}
=
\text{Linear}
\left(
\text{MLP}_{\text{res}}
(\hat{\mathbf X}_{\text{recon}})
+
\hat{\mathbf X}_{\text{recon}}
\right)
+
\hat{\mathbf Y}_{\text{period}}.
```

[p.8, Eq.(7)] 

즉

```math
\boxed{
\hat{\mathbf Y}
=
\hat{\mathbf Y}_{\text{residual}}
+
\hat{\mathbf Y}_{\text{period}}
}
```

로 이해하면 됩니다.

---

# Training objective

논문은 MSE를 training objective로 사용한다고 명시합니다. [p.8] 

논문에 MSE 공식 자체는 인쇄되어 있지 않으므로 표준 형태를 쓰면

```math
\mathcal L_{\text{MSE}}
=
\frac{1}{NCH}
\sum_{n=1}^{N}
\sum_{c=1}^{C}
\sum_{h=1}^{H}
\left(
Y_{n,c,h}
-
\hat Y_{n,c,h}
\right)^2.
```

* $N$: training sample 수
* $C$: channel 수
* $H$: forecasting horizon
* $Y_{n,c,h}$: 실제 값
* $\hat Y_{n,c,h}$: 예측 값

입니다.

---

# 3. 모델 전체 데이터 흐름

Fig.1과 Fig.3을 결합하면 다음과 같이 이해하는 것이 가장 쉽습니다.

$$
\boxed{
\begin{aligned}
\mathbf X
&\rightarrow \text{RevIN}\\
&\rightarrow
\begin{cases}
\text{Cycle extraction}
\rightarrow
\text{MLP refinement}
\rightarrow
\hat{\mathbf Y}_{period}
\\[2mm]
\text{Residual}
\rightarrow
\text{LSWT}
\rightarrow
\text{Coarse patches}
\\
\qquad\rightarrow
\text{Channel Mixing}
\rightarrow
\text{Fine patches}
\\
\qquad\rightarrow
\text{Fine/Coarse MLP mixing}
\rightarrow
\text{ISWT}
\\
\qquad\rightarrow
\text{Residual MLP}
\rightarrow
\hat{\mathbf Y}_{res}
\end{cases}
\\
&\rightarrow
\hat{\mathbf Y}_{period}
+
\hat{\mathbf Y}_{res}
\\
&\rightarrow
\text{Inverse RevIN}.
\end{aligned}
}
$$

[p.4–8, Fig.1, Fig.3]  

---

# 4. 성능 — 저자 보고 결과

실험에는 ETTh1, ETTh2, ETTm1, ETTm2, Weather, ECL, Traffic의 7개 dataset과 $H\in{96,192,336,720}$을 사용합니다. 시간 순서에 따라 train/validation/test를 분할합니다. [p.8, p.13]  

## Dataset-average MSE

| Dataset |   HPMixer |      가장 강한 비교 baseline | 판단                    |
| ------- | --------: | ---------------------: | --------------------- |
| ETTm1   | **0.364** |         CycleNet 0.379 | HPMixer 우세            |
| ETTm2   | **0.261** |         CycleNet 0.266 | HPMixer 우세            |
| ETTh1   |     0.423 |     **SimpleTM 0.422** | 사실상 동률, HPMixer 미세 열세 |
| ETTh2   |     0.367 |     **SimpleTM 0.353** | HPMixer 열세            |
| ECL     | **0.163** |         SimpleTM 0.166 | HPMixer 우세            |
| Traffic |     0.460 | **iTransformer 0.428** | 뚜렷한 열세                |
| Weather | **0.238** |          TimeXer 0.241 | HPMixer 우세            |

모든 수치는 Table 1에서 직접 가져왔습니다. [p.9, Table 1] 

저자들이 특히 언급한 ETTm1에서

$$
\frac{0.381-0.364}{0.381}\times100
\approx 4.46\%
$$

로 SimpleTM 대비 평균 MSE가 4.46% 낮고,

$$
\frac{0.379-0.364}{0.379}\times100
\approx3.96\%
$$

로 CycleNet 대비 3.96% 낮습니다. 이 수치는 저자 본문에서도 동일하게 보고됩니다. [p.10] 

---

# 5. Ablation에서 실제로 무엇을 알 수 있는가

Table 2에서 full HPMixer와 component 제거 모델을 비교합니다. [p.10–11, Table 2] 

### ETTm1 MSE

* HPMixer: **0.364**
* w/o MLP in cycle: 0.365
* w/o entire cycle module: 0.406
* w/o trainable SWT: 0.417
* w/o SWT: 0.439
* one-level patching: 0.390

여기서 중요한 것은 **cycle MLP 제거의 손실은 매우 작지만 전체 cycle module 제거의 손실은 큽니다.**

따라서 제 해석은

$$
\boxed{
\text{cycle 자체의 가치} > \text{cycle 내부 MLP 추가의 가치}
}
$$

입니다.

저자는 MLP도 중요하다고 기술하지만, 적어도 ETTm1 MSE 기준으로는 증가폭이

$$
0.365-0.364=0.001
$$

에 불과합니다.

반면 SWT 전체 제거는

$$
0.439-0.364=0.075
$$

로 훨씬 큽니다.

따라서 ETTm1에서는 **frequency decomposition + hierarchical residual modeling이 더 큰 성능원인일 가능성**이 있습니다.

이 부분은 제 해석이며 저자의 직접적인 결론과 구분해야 합니다.

---

# 6. 저자가 직접 보고한 결과 vs 제 해석

| 항목                    | 저자 보고                                                    | 제 해석                                                                           |
| --------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------ |
| 연구 주제                 | periodicity와 residual의 complementary modeling            | **decomposition-based inductive bias**가 핵심                                     |
| Cycle module          | MLP가 expressive periodic modeling 향상                     | cycle 자체의 효과가 MLP 추가 효과보다 더 커 보이는 dataset이 있음                                  |
| LSWT                  | trend extraction, denoising, frequency representation 향상 | 가장 중요한 성능원인 중 하나로 보이지만 perfect-reconstruction 조건 설명이 부족                        |
| Hierarchical patching | local + long-range 정보를 동시에 학습                            | Swin 계열 multi-scale 아이디어를 시계열 residual에 적용한 것으로 해석 가능                          |
| Channel mixing        | channel dependency를 명시적으로 처리                             | 저차원에는 효과적이나 Traffic $C=862$에서 한계가 나타남                                          |
| Overall accuracy      | competitive/SOTA                                         | 정확히는 **dataset-dependent SOTA**                                                |
| Robustness            | patch/cycle 변화에도 구조적 robustness                          | Fig.4는 오히려 일부 parameter에 대한 **sensitivity**도 보여줌                               |
| Generalization        | test benchmark에서 좋은 예측                                   | 현재 실험은 **within-dataset generalization**이지 OOD/zero-shot generalization 증명은 아님 |

Traffic에는 862개 channel이 존재합니다. [p.13, Table 3] 

---

# 7. 통계적으로 취약하거나 직접 비교가 곤란한 부분

## ① 표준편차·신뢰구간이 없다 — **중요**

HPMixer 결과는 5개 seed

$$
\{3000,3001,3002,3003,3004\}
$$

의 평균이라고 명시합니다. [p.9, Table 1] 

그러나

$$
\text{mean}\pm\text{std}
$$

또는 confidence interval이 없습니다.

따라서 예를 들어

$$
0.423\quad\text{vs.}\quad0.422
$$

와 같은 ETTh1 차이는 통계적으로 실질적인 차이인지 판단할 수 없습니다.

**통계적 유의성 검정도 없습니다.**

---

## ② baseline과 동일한 random seed 조건이 아니다 — **직접 비교 주의**

Table 1은 HPMixer의 결과는 5 seed 평균이지만, baseline MSE/MAE는 [3], [8], [7], [15]에서 **imported**되었다고 명시합니다. [p.9] 

따라서

* code version
* seed
* hardware
* early stopping
* preprocessing
* hyperparameter search budget

이 완전히 동일한 controlled experiment라고 보기 어렵습니다.

즉 Table 1은 **benchmark comparison**으로 유용하지만 엄격한 paired statistical experiment는 아닙니다.

---

## ③ Optuna search budget이 본문에 명시되지 않는다

저자들은 horizon별로 독립적으로 Optuna/TPE tuning을 수행합니다. [p.14] 

검색 범위에는

* learning rate $10^{-4}\sim10^{-2}$
* $d_{\text{model}}\in{32,\ldots,1024}$
* $d_{\text{ff}}\in{32,\ldots,2048}$
* dropout $0.4\sim0.9$
* encoder layer 1–5
* wavelet level 1–5
* patch size 4–48

등 상당히 넓은 공간이 포함됩니다.

그러나 PDF에는 **몇 회의 Optuna trial을 사용했는지 명시되어 있지 않습니다.**

따라서 baseline들이 동등한 tuning budget을 받았는지도 알 수 없습니다.

---

## ④ validation-set overfitting 가능성을 정량적으로 검증하지 않음

dataset × horizon별로 hyperparameter를 독립 최적화하므로

$$
7\times4=28
$$

개의 configuration selection이 수행됩니다.

이 방식 자체가 잘못된 것은 아니지만 반복적인 validation optimization은

$$
\text{best validation configuration}
$$

이 validation noise에 우연히 맞춰질 위험을 증가시킵니다.

논문은 nested validation이나 selection-bias correction을 제공하지 않습니다.

---

## ⑤ Ablation마다 hyperparameter를 다시 최적화했다

Table 2에는

> Hyperparameters were independently optimized for each ablation setting

이라고 명시되어 있습니다. [p.10, Table 2] 

장점은 각 구조를 최선의 상태에서 비교한다는 것입니다.

반대로

$$
\text{architecture effect}
$$

와

$$
\text{retuning effect}
$$

를 완전히 분리하기 어렵습니다.

따라서 **순수한 component causal contribution**을 측정하려면 동일 hyperparameter에서의 controlled ablation도 함께 필요합니다.

---

## ⑥ Table 2의 ETTh1 MAE에 내부 불일치 가능성 — **중요**

Table 1의 ETTh1 HPMixer 값은 각 horizon MAE가

$$
0.395,\;0.423,\;0.445,\;0.460
$$

이고 평균이

$$
0.430
$$

입니다. [p.9, Table 1] 

그런데 Table 2의 HPMixer ETTh1은

$$
\text{MSE}=0.423,\qquad
\text{MAE}=0.367
$$

로 적혀 있습니다. [p.10, Table 2] 

MSE 0.423은 Table 1 average와 동일하지만 MAE 0.367은 Table 1 average 0.430과 다릅니다.

더구나 Table 1의 **모든 ETTh1 horizon MAE가 0.367보다 큽니다.**

따라서 이 값은 제가 확인한 PDF만 놓고 보면

> **오탈자 또는 Table 1과 다른 실험 설정일 가능성이 높지만, 논문에서는 설명되지 않습니다.**

정확한 원인은 코드 또는 저자 확인 없이는 단정할 수 없습니다.

---

## ⑦ 가장 가까운 경쟁 방법 WPMixer가 Table 1에 없다

HPMixer는 WPMixer를 related work에서 인용합니다. WPMixer 역시

$$
\text{Wavelet}
+
\text{Patching}
+
\text{MLP Mixing}
$$

을 사용합니다. 

WPMixer는 AAAI 2025 논문이며 multi-resolution wavelet decomposition, patching, MLP mixing을 결합합니다. ([AAAI Publications][2])

그런데 HPMixer Table 1의 quantitative baseline에는 WPMixer가 포함되지 않았습니다.

이것은 중요한 비교 공백입니다.

왜냐하면 가장 알고 싶은 질문은 사실

$$
\boxed{
\text{HPMixer}
\overset{?}{>}
\text{WPMixer}
}
$$

이고, 그 차이가

* learnable SWT 때문인지,
* cycle branch 때문인지,
* hierarchical non-overlap 때문인지,
* channel attention 때문인지

직접 비교되어야 하기 때문입니다.

---

## ⑧ 효율성 주장은 제한된 조건에서만 검증

Table 5에서는 Weather와 ETTm1, $L=96$, $H=96$만 SimpleTM과 비교합니다. [p.14–16, Table 5]  

Weather에서는

* HPMixer: 172,172 parameters
* SimpleTM: 14,880

이므로 HPMixer가 약 **11.6배 많은 parameter**를 갖습니다.

GFLOPs도 약 **17.6배**입니다.

그런데 실제 seconds/iteration은

$$
0.1364 \text{ vs. }0.2424
$$

로 HPMixer가 약 **1.78배 빠릅니다.**

이 결과는 병렬화 효율 측면에서 매우 흥미롭습니다.

하지만 PDF에는 제가 확인한 범위에서

* GPU 종류
* CUDA/PyTorch 설정
* batch size
* precision
* compiler

등 핵심 runtime condition이 충분히 제시되지 않습니다.

따라서 **theoretical efficiency와 practical speed를 일반화하기 어렵습니다.**

---

# 8. 논문이 답하지 않는 중요한 질문

1. **Cycle length가 시간에 따라 변화한다면 어떻게 되는가?**
   현재 $W$는 training 전 ACF로 고정됩니다.

2. **채널별 cycle length가 서로 다른 경우에도 하나의 $W$가 최선인가?**
   Fig.2 자체가 heterogeneous periodicity를 보여주는데 $W$는 data-level preset입니다.

3. **주기가 drift하는 nonstationary process에서는 어떻게 대응하는가?**

4. **Learnable SWT filter가 perfect reconstruction을 유지하도록 어떤 constraint가 적용되는가?**
   PDF 본문은 orthogonality/biorthogonality 또는 reconstruction constraint를 구체적으로 설명하지 않습니다.

5. $L$이 $P_{\text{co}}$로 정확히 나누어지지 않으면 남는 timestamp를 **drop/pad** 중 무엇으로 처리하는가?

6. $P_{\text{co}}$가 $P_{\text{fi}}$로 나누어지지 않을 때는 어떻게 하는가?

7. RevIN을 제거하면 성능이 얼마나 감소하는가?

8. Channel-Mixing Encoder가 $C=1000$, $5000$, $10000$으로 증가하면 computational complexity가 어떻게 scaling되는가?

9. missing data 또는 irregular sampling에서는 성능이 유지되는가?

10. sensor noise, outlier, timestamp shift에 대한 robustness는 어떠한가?

11. train/test의 계절성 자체가 바뀌는 **concept drift**에서는 어떻게 되는가?

12. 다른 domain으로 옮겼을 때 zero-shot 또는 few-shot generalization은 가능한가?

13. WPMixer와 동일 benchmark·동일 code protocol에서 직접 비교하면 우위가 유지되는가?

14. 예측 uncertainty 또는 prediction interval을 제공할 수 있는가?

15. learned cycle이 실제 physical periodicity와 대응하는지 정량적으로 검증할 수 있는가?

이 중 저자들이 future work에서 직접 답하려고 한 것은 주로 **adaptive cycle length**와 **scalable cross-channel modeling**입니다. [p.11] 

---

# 9. 가장 중요한 그림 5개 해석

논문은 총 5개의 Figure를 갖고 있으므로 사실상 **Fig.1–5 모두가 핵심 그림**입니다.

## Figure 1 — HPMixer 전체 architecture

**위치:** p.4, Fig.1. 

가장 중요한 포인트는 network가 하나의 backbone으로 구성되지 않았다는 것입니다.

$$
\text{Input}
\rightarrow
\boxed{\text{Cycle branch}}
+
\boxed{\text{Residual LSWT branch}}
$$

라는 **dual-path architecture**입니다.

Residual path에서는 LSWT가 여러 coefficient branch를 만든 다음 각각 Channel-Mixing Encoder와 Coarse-Fine Mixer를 통과시킵니다.

### 제 해석

HPMixer의 진짜 novelty는 단순 patching만이 아니라

$$
\boxed{
\text{signal decomposition}
+
\text{frequency decomposition}
+
\text{temporal hierarchy}
+
\text{channel mixing}
}
$$

의 조합입니다.

다만 구성요소가 많아짐에 따라 어느 구성요소가 성능을 만드는가를 분리하는 실험이 더 중요해집니다.

---

# Figure 2 — ETTm2 channel별 ACF

**위치:** p.5, Fig.2. 

저자의 설명은

* ETTm2[0]: 96-step periodicity 강함
* ETTm2[3]: 약한/불안정한 periodicity
* ETTm2[5]: 96-step cyclic structure 매우 약함

입니다.

### 이것이 의미하는 것

$$
C_1,C_2,C_3
$$

가 모두 같은 multivariate signal에 속해도

$$
\rho_{C_1}(96)
\neq
\rho_{C_2}(96)
\neq
\rho_{C_3}(96)
$$

가 될 수 있습니다.

따라서 channel별 periodicity strength가 다릅니다.

### 중요한 한계

이 그림만으로

> “단일 $W$가 반드시 부적절하다”

는 것을 완전히 증명하지는 못합니다.

같은 96-step frequency를 공유하면서 amplitude와 waveform만 다를 수도 있기 때문입니다.

따라서 **channel-specific learnable $W_c$**와의 직접 ablation이 있었다면 주장이 훨씬 강해졌을 것입니다.

---

# Figure 3 — 세 핵심 모듈

**위치:** p.5, Fig.3. 

### Fig.3(a) Learnable Mixing Cycle Module

Recurrent Cycle Module

$$
\rightarrow
\text{Linear}
\rightarrow
\text{GELU}
\rightarrow
\text{Dropout}
\rightarrow
\text{Linear}
\rightarrow
\text{LayerNorm}
$$

으로 cycle representation을 강화합니다.

**GELU**는 smooth nonlinear activation입니다.

### Fig.3(b) Channel-Mixing Encoder

coarse patch에서

$$
Q,K,V
$$

attention을 이용해 channel 간 correlation을 모델링합니다.

### Fig.3(c) Coarse-Fine Mixer

fine patch에서 local structure를 학습한 후 coarse scale로 다시 통합합니다.

### 제 해석

Figure 3은 HPMixer가 “Mixer”라는 이름을 쓴 이유를 가장 잘 보여줍니다.

attention은 전부를 처리하지 않고 **channel interaction에 집중**시키고, temporal interaction은 MLP patch mixing으로 많이 처리합니다.

이것이 full temporal Transformer보다 계산 구조를 병렬화하기 쉬운 이유 중 하나일 가능성이 있습니다.

---

# Figure 4 — Patch size / cycle length sensitivity

**위치:** p.15, Fig.4. 

저자는 이를 robustness analysis라고 부릅니다.

ECL에서는

* optimal patch size = 16
* optimal cycle length = 168

ETTm1에서는

* patch size = 24
* cycle length = 96

가 좋은 MSE를 보입니다. [p.16] 

### 제 해석: “robustness”보다는 “sensitivity landscape”에 가깝다

특히 ECL cycle length에서는 selected optimum 부근과 다른 값의 차이가 꽤 큽니다.

따라서

> “parameter를 바꿔도 성능이 거의 변하지 않는다”

라는 의미의 robustness와는 다릅니다.

오히려

$$
\boxed{
\text{cycle length selection이 꽤 중요하다}
}
$$

는 사실을 보여주는 것으로 해석하는 편이 더 안전합니다.

그리고 원래 configuration 자체가 validation optimization으로 선택되었으므로 그 값이 가장 좋다는 결과는 어느 정도 예상됩니다.

---

# Figure 5 — Periodic / residual component decoupling

**위치:** p.17–18, Fig.5. 

Electricity와 ETTm1에서

$$
\text{Raw signal}
\rightarrow
\begin{cases}
\text{Periodic component}\\
\text{Residual component}
\end{cases}
$$

을 시각화합니다.

저자는 residual이 broader structural trend와 stochastic deviation을 포착하고, 두 성분을 합치는 것이 정확한 forecast의 핵심이라고 설명합니다. [p.18] 

### 제 해석

이 그림은 qualitative evidence로는 좋습니다.

하지만 이것만으로

```math
\text{learned periodic component}
=
\text{true physical seasonality}
```

라고 결론내릴 수는 없습니다.

더 강한 증거를 위해서는 예를 들어

* PSD 비교
* dominant frequency coherence
* spectral entropy
* reconstruction error
* periodic component autocorrelation
* residual whiteness

등의 정량 분석이 필요합니다.

---

# 10. 모델의 일반화 성능을 어떻게 평가해야 하는가

여기서 가장 중요한 구분이 있습니다.

## 논문이 검증한 일반화

$$
\boxed{
\text{train on dataset A}
\rightarrow
\text{future test period of dataset A}
}
$$

입니다.

즉 **within-domain temporal generalization**입니다.

Chronological split을 사용하므로 random shuffle leakage는 방지합니다. ETT는 6:2:2, Weather/ECL/Traffic은 7:1:2로 시간순 분할합니다. [p.13] 

이것은 올바른 평가입니다.

하지만 다음과는 다릅니다.

## 검증하지 않은 일반화

### Domain generalization

$$
\text{train on A}
\rightarrow
\text{test on B}
$$

### Distribution shift

$$
P_{\text{train}}(X,Y)
\neq
P_{\text{test}}(X,Y)
$$

### Zero-shot forecasting

새 dataset에서 parameter update 없이 forecasting.

### Few-shot adaptation

새 domain의 소량 data만으로 adaptation.

### Variable generalization

training에서 보지 못한 channel/variable 구조로 확장.

따라서 HPMixer의 현재 결과만으로는

> “HPMixer가 다른 산업·다른 센서·다른 sampling frequency에도 일반화된다”

고 말할 수 없습니다.

---

# 11. 저자가 인정한 일반화 한계

Conclusion에서 저자들은 두 가지를 명시합니다.

### ① Predetermined cycle length

고정된 $W$는

* irregular periodicity
* drifting periodicity

에 적응하기 어렵습니다.

### ② High-dimensional channel mixing

Traffic처럼 channel 수가 매우 많은 환경에서는 current channel mixer가 복잡한 inter-channel dependency를 충분히 포착하지 못할 수 있습니다.

그래서 future work로

$$
\boxed{
\text{fully adaptive learnable cycle length}
}
$$

과

$$
\boxed{
\text{scalable cross-channel modeling}
}
$$

을 제시합니다. [p.11] 

이 두 방향은 실제 Table 1의 약점과도 일치하므로 **future-work 제안의 논리적 근거가 비교적 명확합니다.**

---

# 12. 2020년 이후 관련 최신 연구 비교

HPMixer를 이해하려면 시간축으로 보면 훨씬 명확합니다.

| 연도      | 모델           | 핵심 방향                                              | HPMixer와 관계                        | 직접 수치 비교 가능성        |
| ------- | ------------ | -------------------------------------------------- | ---------------------------------- | ------------------- |
| 2021    | Informer     | sparse attention로 장기 sequence 효율화                  | 초기 LTSF Transformer 계보             | 낮음                  |
| 2021    | Autoformer   | trend/seasonal decomposition + autocorrelation     | decomposition 철학 선행                | 제한적                 |
| 2022    | FEDformer    | decomposition + Fourier frequency modeling         | HPMixer frequency branch의 중요한 선행계열 | 제한적                 |
| 2022/23 | DLinear      | 단순 linear가 복잡 Transformer를 이길 수 있음을 제기             | 모델 복잡도의 필요성 검증 기준                  | 제한적                 |
| 2023    | PatchTST     | temporal patch + channel independence              | HPMixer patching의 핵심 비교 대상         | **Table 1 직접 비교**   |
| 2023    | Crossformer  | cross-time + cross-dimension attention             | HPMixer channel dependence와 밀접     | 논문 Table 1에는 없음     |
| 2024    | iTransformer | variate token attention                            | HPMixer channel mixing과 강하게 관련     | **직접 비교**           |
| 2024    | TimeMixer    | multiscale decomposition + MLP mixing              | HPMixer coarse/fine philosophy와 밀접 | **직접 비교**           |
| 2024    | CycleNet     | explicit learnable cycle + residual forecast       | HPMixer periodic branch의 직접 기반     | **직접 비교**           |
| 2024    | TimeXer      | endogenous/exogenous interaction                   | 변수 관계 modeling 비교                  | **직접 비교**           |
| 2025    | WPMixer      | wavelet + patch + MLP                              | **HPMixer와 가장 가까운 구조 중 하나**        | HPMixer Table 1에 없음 |
| 2025    | SimpleTM     | lightweight multivariate specialized model         | 효율/accuracy baseline               | **직접 비교**           |
| 2026    | HPMixer      | cycle + LSWT + hierarchical patch + channel mixing | 통합 architecture                    | 본 논문                |

---

## Informer — 2021

Informer는 긴 sequence에서 Transformer의 $O(L^2)$ attention 문제를 해결하기 위해 ProbSparse attention을 사용하여 대략

$$
O(L\log L)
$$

복잡도를 목표로 했습니다. ([AAAI Publications][3])

HPMixer와의 차이는 HPMixer가 단순히 attention을 sparsify하는 것이 아니라 **signal structure 자체를 분해**한다는 것입니다.

---

## Autoformer — 2021

Autoformer는 decomposition을 network 내부 구조로 넣고 autocorrelation을 이용해 periodic dependency를 추출합니다. ([NeurIPS Proceedings][4])

HPMixer에 이어지는 중요한 철학은

$$
\boxed{
\text{complex signal을 먼저 구조적으로 분해한다}
}
$$

는 것입니다.

---

# FEDformer — 2022

FEDformer는 seasonal-trend decomposition에 frequency representation을 결합하고 Fourier-domain sparse representation을 사용합니다. 논문은 multivariate forecasting에서 기존 SOTA 대비 14.8% error reduction을 보고했습니다. ([Proceedings of Machine Learning Research][5])

HPMixer와의 차이는

* FEDformer: Fourier + Transformer
* HPMixer: **learnable stationary wavelet + patch mixer**

입니다.

Fourier는 global periodicity에 자연스럽고 wavelet은 **time-frequency localization**을 제공한다는 차이가 있습니다.

---

# DLinear — 2022/2023

Zeng 등은 매우 단순한 linear forecasting model이 당시 복잡한 Transformer들을 여러 benchmark에서 능가할 수 있음을 보여주면서 LTSF architecture complexity 자체를 문제 삼았습니다. ([arXiv][6])

이 연구 이후 중요한 연구 기준은

> “복잡한 architecture가 정말 단순 baseline보다 의미 있게 좋은가?”

가 되었습니다.

따라서 HPMixer가 SimpleTM 같은 강력한 lightweight baseline을 포함한 것은 적절합니다.

---

# PatchTST — 2023

PatchTST는 연속 timestamp를 개별 token으로 쓰는 대신 subseries patch를 token으로 사용합니다. 이를 통해 local semantics를 유지하면서 attention token 수를 줄입니다. ([OpenReview][7])

차이:

$$
\text{PatchTST}
:
\text{overlapping patch}
+
\text{channel independent}
$$

$$
\text{HPMixer}
:
\text{non-overlapping hierarchical patch}
+
\text{explicit channel mixing}
$$

입니다.

HPMixer는 PatchTST를 Table 1에서 직접 비교합니다.

---

# Crossformer — 2023

Crossformer는 multivariate forecasting에서 **cross-time dependency와 cross-dimension dependency를 동시에 다뤄야 한다**고 주장하며 Two-Stage Attention을 제안했습니다. ([OpenReview][8])

HPMixer의 Channel-Mixing Encoder는 같은 문제를 더 제한적이고 구조화된 형태로 해결하려는 접근이라고 볼 수 있습니다.

다만 HPMixer Table 1에는 Crossformer 결과가 없습니다.

---

# iTransformer — 2024

iTransformer는 timestamp가 아니라 **각 variable을 하나의 token으로 뒤집어(invert)** attention이 channel correlation을 직접 보게 합니다. ([OpenReview][9])

이 접근은 HPMixer의 약점과 매우 관련 있습니다.

Traffic에서 iTransformer average MSE는

$$
0.428
$$

이고 HPMixer는

$$
0.460
$$

입니다. [p.9, Table 1] 

즉 $C=862$ 같은 고차원 환경에서는 **channel-focused architecture가 HPMixer보다 유리할 가능성**이 있습니다.

---

# TimeMixer — 2024

TimeMixer는 여러 sampling scale에서 microscopic와 macroscopic temporal pattern을 분리한 뒤 MLP로 mixing합니다. ([OpenReview][10])

HPMixer와 매우 가까운 관점은

$$
\boxed{
\text{fine scale}
+
\text{coarse scale}
}
$$

입니다.

차이는 HPMixer가 이를 **LSWT residual coefficient 내부의 hierarchical patch** 구조로 구현한다는 것입니다.

---

# CycleNet — 2024

CycleNet은 long-horizon forecast의 안정적 주기성을 명시적으로 학습하고, cycle을 제거한 residual을 예측하는 Residual Cycle Forecasting을 제안합니다. ([NeurIPS Proceedings][11])

HPMixer의 periodic branch는 직접적으로 이 연구를 확장합니다.

따라서 계보상

$$
\text{CycleNet}
\rightarrow
\text{HPMixer Cycle Branch}
$$

라고 볼 수 있습니다.

---

# TimeXer — 2024

TimeXer는 endogenous target과 exogenous variables를 구분하고 patch-wise self-attention과 variate-wise cross-attention을 결합합니다. ([NeurIPS Proceedings][12])

HPMixer는 모든 변수를 기본적으로 MTS channel로 다룬다는 점에서 목적이 다릅니다.

따라서 산업 시계열처럼

> 일부 변수는 예측 대상이고 나머지는 설명 변수

라는 구조가 명확하면 TimeXer의 역할 구분은 HPMixer가 참고할 만한 방향입니다.

---

# WPMixer — 2025

WPMixer는

$$
\boxed{
\text{Wavelet decomposition}
+
\text{Patching}
+
\text{MLP mixing}
}
$$

을 결합합니다. ([AAAI Publications][2])

따라서 구조적으로 HPMixer와 특히 가깝습니다.

HPMixer의 추가 요소는 크게

* stationary/learnable wavelet
* explicit cycle branch
* two-level non-overlapping hierarchy
* channel-mixing encoder

입니다.

앞서 언급했듯 **직접 head-to-head comparison이 없는 것은 HPMixer 논문의 중요한 공백**입니다.

---

# SimpleTM — 2025

SimpleTM은 큰 general-purpose Transformer가 아닌 **가벼운 MTS 전용 architecture**로도 강한 결과를 낼 수 있음을 보여줍니다. ([OpenReview][13])

HPMixer는 대부분 dataset에서 경쟁적이지만 ETTh1/ETTh2에서는 SimpleTM이 여전히 매우 강합니다.

이는 모든 문제에서 architecture complexity가 이득을 주는 것은 아니라는 점을 다시 보여줍니다.

---

# 13. Foundation Model 연구와 비교 — 일반화 관점에서 특히 중요

이 부분은 **HPMixer와 평가 protocol 자체가 다르므로 수치를 직접 비교해서는 안 됩니다.**

하지만 “일반화 성능”이라는 질문에서는 매우 중요합니다.

## TimesFM — ICML 2024

TimesFM은 대규모 시계열 corpus를 pretraining하여 **unseen dataset에 zero-shot forecasting**을 수행합니다. Google은 약 100B real-world time points를 사용한 200M parameter 모델이 여러 unseen benchmark에서 supervised model에 근접하거나 능가하는 결과를 보고했습니다. ([Google Research][14])

HPMixer:

$$
\text{dataset-specific training}
$$

TimesFM:

$$
\text{large-scale pretraining}
\rightarrow
\text{zero-shot transfer}
$$

이므로 일반화의 정의가 완전히 다릅니다.

---

# Moirai — ICML 2024

Moirai는 LOTSA의 27B observations, 9개 domain을 사용하여 cross-frequency, arbitrary variate count, heterogeneous distributions를 다루는 universal forecasting model을 제안합니다. ([Proceedings of Machine Learning Research][15])

HPMixer가 앞으로 일반화를 강화하려면 이 아이디어가 매우 중요합니다.

특히 현재 HPMixer가 dataset마다 별도 $W$, patch size, model dimension을 선택하는 대신

$$
f_\theta
(
X;
\text{frequency},
\text{channel structure},
\text{scale}
)
$$

처럼 다양한 sampling scale과 channel 수를 하나의 parameterization으로 처리하는 방향을 생각할 수 있습니다.

---

# Timer — ICML 2024

Timer는 최대 10억 time points 규모 pretraining과 GPT-style generative training을 통해 forecasting, imputation, anomaly detection을 하나의 framework로 통합합니다. ([Proceedings of Machine Learning Research][16])

특히 저자들은 작은 dataset-specific 모델의 **data-scarce performance bottleneck**을 직접 문제로 삼습니다.

이 점은 HPMixer의 후속 연구에서 매우 중요합니다.

---

# Chronos — 2024

Chronos는 값 자체를 scaling/quantization하여 token으로 만들고 T5 architecture로 pretrain하며, Gaussian-process synthetic data까지 사용해 generalization을 강화합니다. ([arXiv][17])

이는 HPMixer의 deterministic supervised training과 완전히 다른 방향입니다.

---

# 14. 앞으로 HPMixer가 일반화 성능을 높이려면

제가 연구를 이어간다면 우선순위를 다음과 같이 잡겠습니다.

## 1순위 — Fixed cycle length를 adaptive cycle mixture로 변경

현재:

$$
W=W^\*
$$

하나를 사용합니다.

개선안은

```math
\hat Y_{\text{period}}
=
\sum_{m=1}^{M}
\alpha_m(X)
\hat Y_{\text{period}}^{(W_m)}
```

입니다.

여기에서

* $W_m$: 서로 다른 candidate cycle
* $\alpha_m(X)$: 현재 input에 따라 정해지는 cycle weight
* $\sum_m\alpha_m=1$

로 둡니다.

즉 24시간, 12시간, 7일 주기가 동시에 존재할 수 있게 합니다.

이렇게 하면 **period drift와 multi-periodicity**에 훨씬 강해질 가능성이 있습니다.

---

# 2순위 — channel attention을 sparse/low-rank/graph 구조로 변경

Traffic에서 발생한 문제를 직접 겨냥합니다.

완전 channel attention이

$$
O(C^2)
$$

수준의 interaction을 요구한다면,

$$
A
\approx
UV^\top,
\qquad
U,V\in\mathbb R^{C\times r},
\qquad
r\ll C
$$

인 low-rank channel interaction을 사용할 수 있습니다.

또는 실제 physical connectivity가 있다면 graph prior를 넣는 것이 더 합리적일 수 있습니다.

---

# 3순위 — Learnable SWT에 명시적인 reconstruction/stability constraint

예를 들어

```math
\mathcal L
=
\mathcal L_{\text{forecast}}
+
\lambda_{\text{recon}}
\left\|
X-\mathcal W^{-1}(\mathcal W(X))
\right\|_2^2
```

와 같이 만들 수 있습니다.

이 식은 제가 제안하는 후속 연구식이며 HPMixer 원 논문의 식이 아닙니다.

이렇게 하면 trainable filter가 prediction loss만 따라가다가 wavelet-like decomposition 의미를 잃는 것을 억제할 수 있습니다.

---

# 4순위 — scale-invariant hierarchical patching

현재 patch size

$$
P_{\text{co}},P_{\text{fi}}
$$

를 dataset/horizon별로 tuning합니다.

더 일반화하려면 여러 patch size를 병렬로 사용하고 gating하면 됩니다.

```math
Z
=
\sum_{s=1}^{S}
\beta_s(X)
Z^{(P_s)}.
```

이렇게 하면 dataset마다 patch size를 강하게 재튜닝해야 하는 문제를 줄일 수 있습니다.

---

# 5순위 — Pretraining + HPMixer 구조 결합

가장 흥미로운 장기 방향입니다.

HPMixer의 좋은 inductive bias는 유지하면서

$$
\text{large heterogeneous pretraining corpus}
$$

에서

* LSWT filter
* patch encoder
* channel representation

을 pretrain합니다.

새 dataset에서는

$$
\text{zero-shot}
\quad\text{or}\quad
\text{few-shot}
$$

으로 사용합니다.

이 방향은 TimesFM, Moirai, Timer, Chronos가 보여주는 최근 연구 흐름과 HPMixer를 연결합니다. ([Google Research][14])

---

# 15. 후속 연구 실험에서 반드시 추가할 평가

모델 일반화 연구라면 단순한 Table 1 확대보다 다음 실험이 더 중요합니다.

### A. Cross-dataset transfer

$$
\text{ETTm1 train}
\rightarrow
\text{ETTm2 test}
$$

### B. Temporal distribution shift

초기 계절에서 학습하고 이후 주기/mean/variance가 달라진 구간에서 평가합니다.

### C. Channel-count extrapolation

$$
C_{\text{train}}<C_{\text{test}}
$$

조건을 만듭니다.

### D. Noise robustness

$$
X'=X+\epsilon,
\qquad
\epsilon\sim\mathcal N(0,\sigma^2)
$$

로 $\sigma$ 증가에 따른 성능곡선을 봅니다.

### E. Missing channel / missing timestamp

산업 센서에서는 매우 중요합니다.

### F. Cycle drift

$$
W_t=W_0+\Delta W_t
$$

형태로 주기 자체가 바뀌는 synthetic benchmark가 필요합니다.

### G. Statistical comparison

각 configuration에 대해

$$
\bar e\pm s_e
$$

를 보고하고 paired bootstrap 또는 적절한 forecast-comparison test를 수행하는 것이 좋습니다.

---

# 16. 종합 결론

**[저자의 결론]** HPMixer는 MLP-enhanced cycle module, LSWT, two-level hierarchical patching을 결합함으로써 periodicity와 residual dynamics를 동시에 처리하고, 표준 장기 MTS benchmark에서 competitive한 성능을 달성합니다. 가장 큰 한계로 fixed cycle length와 초고차원 channel mixing을 지적하며 adaptive cycle과 scalable cross-channel architecture를 미래 연구로 제안합니다. [p.11] 

**[제 결론]** 이 논문의 가장 중요한 기여는 “새로운 MLP 하나”가 아니라,

$$
\boxed{
\text{explicit periodicity}
+
\text{learnable time-frequency decomposition}
+
\text{hierarchical temporal resolution}
+
\text{cross-channel interaction}
}
$$

을 하나의 구조 안에서 결합한 것입니다.

성능도 실제로 ETTm1, ETTm2, ECL, Weather에서 매우 강하며 28개 dataset–horizon 조합에서 많은 1위 결과를 냅니다. 그러나 **ETTh1/ETTh2/Traffic에서는 일관된 우위가 없으며**, 표준편차와 유의성 검정 부재, 외부 논문에서 가져온 baseline, HPO budget 불명확성, WPMixer 직접 비교 누락 때문에 “통계적으로 확실한 보편적 SOTA”라고 표현하는 것은 과도합니다.

특히 **일반화 성능** 측면에서 현재 HPMixer는

$$
\text{in-domain chronological generalization}
$$

을 잘 보여준 연구이지,

$$
\text{cross-domain/zero-shot/OOD generalization}
$$

을 입증한 연구는 아닙니다.

향후 가장 가치 있는 확장은

$$
\boxed{
\text{adaptive multi-cycle}
+
\text{scalable channel modeling}
+
\text{constrained LSWT}
+
\text{multi-scale adaptive patching}
+
\text{large-scale pretraining}
}
$$

이라고 판단합니다.

이 방향으로 발전시키면 HPMixer는 단순한 benchmark-specific forecaster를 넘어 **주기 변화·센서 수 변화·domain shift를 견딜 수 있는 generalized multivariate forecasting architecture**로 확장될 가능성이 있습니다.

---

# 참고한 자료 및 사이트

**원 논문**

1. **Choi, Jung Min, Vijaya Krishna Yalavarthi, Lars Schmidt-Thieme, “HPMixer: Hierarchical Patching for Multivariate Time Series Forecasting.”** 첨부 arXiv v2 PDF. 
2. **Springer Nature — Advances in Knowledge Discovery and Data Mining, PAKDD 2026 Proceedings Part III**, HPMixer pp.42–53. ([Springer][1])
3. **DBLP — “HPMixer: Hierarchical Patching for Multivariate Time Series Forecasting.”**, PAKDD 2026, DOI 10.1007/978-981-92-1465-5_4. ([DBLP][18])
4. **arXiv — “HPMixer: Hierarchical Patching for Multivariate Time Series Forecasting.”** ([arXiv][19])

**2020년 이후 관련 연구**

5. **Zhou et al., “Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting,” AAAI 2021.** ([AAAI Publications][3])
6. **Wu et al., “Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting,” NeurIPS 2021.** ([NeurIPS Proceedings][4])
7. **Zhou et al., “FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting,” ICML 2022.** ([Proceedings of Machine Learning Research][5])
8. **Zeng et al., “Are Transformers Effective for Time Series Forecasting?”** ([arXiv][6])
9. **Nie et al., “A Time Series is Worth 64 Words: Long-Term Forecasting with Transformers,” ICLR 2023 — PatchTST.** ([OpenReview][7])
10. **Zhang & Yan, “Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting,” ICLR 2023.** ([OpenReview][8])
11. **Liu et al., “iTransformer: Inverted Transformers Are Effective for Time Series Forecasting,” ICLR 2024.** ([OpenReview][9])
12. **Wang et al., “TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting,” ICLR 2024.** ([OpenReview][10])
13. **Lin et al., “CycleNet: Enhancing Time Series Forecasting through Modeling Periodic Patterns,” NeurIPS 2024.** ([NeurIPS Proceedings][11])
14. **Wang et al., “TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables,” NeurIPS 2024.** ([NeurIPS Proceedings][12])
15. **Murad et al., “WPMixer: Efficient Multi-Resolution Mixing for Long-Term Time Series Forecasting,” AAAI 2025.** ([AAAI Publications][2])
16. **Chen et al., “SimpleTM: A Simple Baseline for Multivariate Time Series Forecasting,” ICLR 2025.** ([OpenReview][13])

**일반화·Foundation Model 관련**

17. **Das et al., “A Decoder-Only Foundation Model for Time-Series Forecasting” — TimesFM, ICML 2024 / Google Research.** ([Google Research][14])
18. **Woo et al., “Unified Training of Universal Time Series Forecasting Transformers” — Moirai, ICML 2024.** ([Proceedings of Machine Learning Research][15])
19. **Liu et al., “Timer: Generative Pre-trained Transformers Are Large Time Series Models,” ICML 2024.** ([Proceedings of Machine Learning Research][16])
20. **Ansari et al., “Chronos: Learning the Language of Time Series,” 2024.** ([arXiv][17])

다음 단계로는 이 분석을 바탕으로 **HPMixer를 실제로 구현할 수 있는 수준의 “모듈별 tensor shape → 수식 → PyTorch 연산 → 학습 순서” 구현 해설**을 만드는 것이 가장 유용합니다. 특히 원 논문에서 설명이 부족한 **LSWT 학습필터, hierarchical patch tensor reshape, channel-mixing attention, cycle extraction**을 코드 수준에서 검증하면 논문의 재현성과 한계를 훨씬 명확하게 판단할 수 있습니다.

[1]: https://link.springer.com/book/10.1007/978-981-92-1465-5?utm_source=chatgpt.com "Advances in Knowledge Discovery and Data Mining: 30th Pacific-Asia Conference on Knowledge Discovery and Data Mining, PAKDD 2026, Hong Kong, China, June 9–12, 2026, Proceedings, Part III | Springer Nature Link"
[2]: https://ojs.aaai.org/index.php/AAAI/article/view/34156?utm_source=chatgpt.com "WPMixer: Efficient Multi-Resolution Mixing for Long-Term Time Series Forecasting | Proceedings of the AAAI Conference on Artificial Intelligence"
[3]: https://ojs.aaai.org/index.php/AAAI/article/view/17325?utm_source=chatgpt.com "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting | Proceedings of the AAAI Conference on Artificial Intelligence"
[4]: https://proceedings.neurips.cc/paper/2021/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html?utm_source=chatgpt.com "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting"
[5]: https://proceedings.mlr.press/v162/zhou22g.html?utm_source=chatgpt.com "FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting"
[6]: https://arxiv.org/abs/2205.13504?utm_source=chatgpt.com "Are Transformers Effective for Time Series Forecasting?"
[7]: https://openreview.net/pdf?id=Jbdc0vTOcol&utm_source=chatgpt.com "Published as a conference paper at ICLR 2023"
[8]: https://openreview.net/pdf?id=vSVLM2j9eie&utm_source=chatgpt.com "Published as a conference paper at ICLR 2023"
[9]: https://openreview.net/pdf?id=JePfAI8fah&utm_source=chatgpt.com "Published as a conference paper at ICLR 2024"
[10]: https://openreview.net/pdf?id=7oLshfEIC2&utm_source=chatgpt.com "Published as a conference paper at ICLR 2024"
[11]: https://proceedings.neurips.cc/paper_files/paper/2024/hash/bfe7998398779dde03cad7a73b1f81b6-Abstract-Conference.html?utm_source=chatgpt.com "CycleNet: Enhancing Time Series Forecasting through Modeling Periodic Patterns"
[12]: https://proceedings.neurips.cc/paper_files/paper/2024/hash/0113ef4642264adc2e6924a3cbbdf532-Abstract-Conference.html?utm_source=chatgpt.com "TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables"
[13]: https://openreview.net/pdf/e7ccd35be296940a46939659136fb42d3cdc1fb1.pdf?utm_source=chatgpt.com "Published as a conference paper at ICLR 2025"
[14]: https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/?utm_source=chatgpt.com "A decoder-only foundation model for time-series forecasting"
[15]: https://proceedings.mlr.press/v235/woo24a.html?utm_source=chatgpt.com "Unified Training of Universal Time Series Forecasting Transformers"
[16]: https://proceedings.mlr.press/v235/liu24cb.html?utm_source=chatgpt.com "Timer: Generative Pre-trained Transformers Are Large Time Series Models"
[17]: https://arxiv.org/abs/2403.07815?utm_source=chatgpt.com "Chronos: Learning the Language of Time Series"
[18]: https://dblp.uni-trier.de/rec/conf/pakdd/ChoiYS26.html?utm_source=chatgpt.com "dblp: HPMixer: Hierarchical Patching for Multivariate Time Series Forecasting."
[19]: https://arxiv.org/abs/2602.16468?utm_source=chatgpt.com "HPMixer: Hierarchical Patching for Multivariate Time Series Forecasting"
