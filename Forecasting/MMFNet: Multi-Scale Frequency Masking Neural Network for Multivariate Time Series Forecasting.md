# MMFNet: Multi-Scale Frequency Masking Neural Network for Multivariate Time Series Forecasting

**Aitian Ma, Dongsheng Luo, Mo Sha, “MMFNet: Multi-Scale Frequency Masking Neural Network for Multivariate Time Series Forecasting”**

먼저 분석 기준을 분명히 하겠습니다. **모델 구조·수식·실험 수치는 사용자가 첨부한 2024년 10월 2일 arXiv v1, 16페이지 “Under Review” 버전을 기준**으로 검토했습니다. 논문의 제목과 저자는 첨부본에서 직접 확인됩니다.  현재 공식 저장소는 MMFNet이 **SAC 2026에 채택되었다고 명시**하고 있으므로 연구의 현재 상태는 preprint보다 진전되었지만, 저장소의 scale 설정 등 일부 내용은 첨부 v1과 달라졌습니다. 따라서 아래에서 **“논문”은 첨부 v1**, “현재 저장소”는 별도로 표시합니다. ([arXiv][1])

---

# 1. Executive Summary — 10문장 이내

1. **[저자 보고]** MMFNet은 장기 다변량 시계열 예측에서 기존 단일-scale 주파수 모델이 전체 구간을 하나의 spectrum으로 처리하고 고주파 성분을 일률적으로 제거함으로써 비정상성 및 단기 변동을 놓치는 문제를 해결하기 위해 제안되었습니다(p.1–2). 
2. **[저자 보고]** 핵심 방법인 MMFT는 시계열을 fine/intermediate/coarse scale로 나누고 각 segment에 DCT를 수행한 뒤, 학습 가능한 frequency mask로 성분별 중요도를 조절합니다(Figure 1, p.3–5). 
3. 그 후 masked spectrum을 선형층으로 미래 spectrum에 사상하고 iDCT로 시간영역에 복원한 뒤 세 scale을 결합하는 비교적 단순한 구조입니다(p.4–5, Algorithm 1 p.13). 
4. **[저자 보고]** Table 1에서 MMFNet은 ETTh1·ETTh2·ETTm2에서는 네 forecasting horizon 모두 우수하며, 최대 개선치는 ETTm2의 horizon 720에서 두 번째 우수 모델 대비 **MSE 6.0% 감소**입니다(p.6–7). 
5. **[내 계산]** 그러나 Table 1의 7개 데이터셋 × 4개 horizon = 28조건을 직접 세면 MMFNet이 명시적으로 최저 MSE인 경우는 **18/28개**이므로 “모든 조건에서 SOTA”로 해석해서는 안 됩니다. 
6. **[저자 보고]** MMFT-vs-SFT ablation은 여러 scale을 사용하는 것이 단일 global frequency transformation보다 유리하며(Table 2), masking 또한 표에 제시된 대부분의 조건에서 MSE를 낮추는 것으로 보고됩니다(Table 3, p.8). 
7. **[내 검증]** 다만 Table 2 본문, Table 3 데이터셋 표기, Table 4 본문에 서로 일치하지 않는 숫자·데이터셋명이 있으며, 반복 실험의 표준편차·신뢰구간·유의성 검정이 없어 0.001–0.006 수준의 작은 차이를 통계적으로 유의한 성능 향상이라고 확정할 수 없습니다.
8. **[내 해석]** MMFNet의 가장 중요한 아이디어는 “고주파를 제거한다”가 아니라 **시간 위치가 다른 여러 scale에서 주파수 표현을 만들고 데이터에 맞추어 어떤 주파수를 살릴지 학습한다**는 점이며, 이는 FITS의 고정 low-pass filtering보다 일반화 가능성이 높은 inductive bias입니다. ([ICLR Proceedings][2])
9. 반면 Algorithm 1의 v1 구현은 channel-independent shared parameters에 의존하므로 명시적인 변수 간 상호작용을 제한하며, 이는 Traffic 같은 862-channel 데이터에서 PatchTST/TimeMixer보다 밀리는 결과와 연결해서 연구할 가치가 있습니다(p.7, p.13).  
10. **[내 판단]** 향후 일반화 성능의 핵심 개선축은 **고정 scale → 학습 가능한 scale**, **channel independence → 저랭크 cross-variable interaction**, **global DCT segmentation → 국소화된 time-frequency representation**, 그리고 **단일 test split → distribution-shift/rolling-origin 검증**입니다.

> **용어 — LTSF(Long-Term Time Series Forecasting)**
> 과거의 비교적 긴 관측 구간을 입력으로 사용하여 수십~수천 step 뒤까지 동시에 예측하는 장기 시계열 예측 문제입니다.

> **비정상성(non-stationarity)**
> 시간에 따라 평균, 분산, 주기, 변수 사이의 관계 등이 변하는 성질입니다. 예를 들어 오전과 야간의 공정 상태나 여름과 겨울의 전력 사용량이 서로 다른 통계적 구조를 갖는 경우입니다.

---

# 1-1. 연구 목적과 필요성

논문의 출발점은 다음 대조입니다.

기존 Transformer 계열은 긴 거리의 의존성을 모델링할 수 있지만 계산량이 크고, 반대로 FITS와 같은 frequency-domain linear model은 매우 가볍지만 **하나의 global spectrum + 고정 low-pass filtering**에 의존한다는 것입니다. 저자들은 이런 방식이 다음 두 종류의 정보가 동시에 존재하는 현실 시계열에 불리하다고 주장합니다.

* 천천히 변화하는 **장기 trend / low-frequency component**
* 순간적으로 변하는 **local fluctuation / high-frequency component**

FITS는 약 10k parameter만으로 frequency interpolation을 수행하는 매우 가벼운 모델이지만, 기본 아이디어에는 중요하지 않은 고주파를 버리는 과정이 포함됩니다. ([ICLR Proceedings][2]) MMFNet 저자들은 일부 실제 데이터에서는 이 고주파 부분이 단순 noise가 아니라 **예측에 필요한 급격한 변화**일 수 있다고 문제를 제기합니다(p.1–2). 

따라서 논문의 핵심 질문은 사실상 다음과 같습니다.

> **“어떤 주파수가 noise인지 사람이 고정하지 않고, 서로 다른 시간 scale에서 모델이 직접 학습하게 만들 수 있는가?”**

이 질문에 대한 답이 **Multi-scale Masked Frequency Transformation(MMFT)** 입니다.

> **주파수 영역(frequency domain)**
> 원 신호를 “시간 $t$에서 값이 얼마인가?”가 아니라 “느린 변화, 빠른 변화 등 각 진동 성분이 얼마나 포함되어 있는가?”라는 관점으로 표현하는 공간입니다.

> **Inductive bias(귀납 편향)**
> 모델이 데이터를 보기 전부터 갖는 구조적 가정입니다. MMFNet에서는 “시계열은 여러 시간 scale의 주파수 패턴으로 설명할 수 있다”는 가정 자체가 inductive bias입니다.

---

# 2. 핵심 주장과 근거

| 핵심 주장                                                                     | 저자가 제시한 근거                                       | 위치                       | 검토 결과                                                       |
| ------------------------------------------------------------------------- | ------------------------------------------------ | ------------------------ | ----------------------------------------------------------- |
| 단일-scale frequency transformation은 non-stationarity와 local variation을 놓친다 | SFT/FITS의 global transformation과 fixed cutoff 비판 | p.2–3                    | **이론적으로 타당한 문제 제기**, 하지만 controlled non-stationarity 실험은 없음 |
| Multi-scale decomposition이 local+global pattern을 함께 포착한다                  | SFT/MFT/MMFT ablation                            | **Table 2, p.8**         | MMFT가 SFT보다 모든 제시 ETT 조건에서 낮은 MSE                           |
| Learnable mask가 fixed filter보다 유연하다                                       | mask vs no-mask ablation                         | **Table 3, p.8**         | 수치는 대부분 개선 방향이나 **dataset label 오류 가능성 큼**                  |
| 장기 horizon에서 강하다                                                          | horizon 720에서 ETT 중심 향상                          | **Table 1, p.6–7**       | ETTm2 H=720 최대 6.0% 감소                                      |
| Ultra-long forecasting에서도 강하다                                             | H=960–1680 실험                                    | **Table 4, p.8–9**       | 대부분 우수하지만 일부 조건에서는 패배; full SOTA 비교 아님                      |
| 고차원 multivariate에도 적용 가능하다                                                | shared channel-independent parameterization      | **Algorithm 1, p.13**    | 확장성 장점은 있지만 **cross-variable 정보를 명시적으로 학습하지 않는 한계**         |
| Noise를 adaptive하게 제거한다                                                    | learned mask visualization                       | **Figures 2–3, p.15–16** | 정성적 evidence는 있지만 mask 값의 수학적 해석이 Eq.6과 완전히 명료하지 않음         |
| 다양한 LTSF task에 robust하다                                                   | 7 benchmark dataset                              | **Table 1**              | 여러 domain에서 검증했으나 통계적 uncertainty 보고 없음                     |

---

# 2-1. 해결 문제 → 수식 → 모델 구조 → 성능 → 한계

## A. 문제 정의

논문의 forecasting 문제는 Eq. (1)로 정의됩니다(p.2).

```math
\hat{\mathbf{x}}_{t+1:t+H}
=
f\left(
\mathbf{x}_{t-L+1:t}
\right)
```

여기서

* $\mathbf{x}_{t-L+1:t}\in\mathbb{R}^{L\times C}$: 과거 관측값입니다.
* $L$: look-back window, 즉 모델에 넣는 과거 길이입니다.
* $C$: 변수 또는 channel 수입니다.
* $H$: 미래 예측 길이인 forecast horizon입니다.
* $\hat{\mathbf{x}}_{t+1:t+H}\in\mathbb{R}^{H\times C}$: 예측된 미래 다변량 시계열입니다.
* $f$: MMFNet이 학습하려는 forecasting mapping입니다. 

$H$가 커질수록 가까운 미래와 먼 미래를 모두 설명해야 하므로 단기 변동과 장기 추세를 동시에 학습하기 어려워집니다.

---

## B. 기존 SFT의 한계

한 주파수 성분은 논문 Eq. (2)에서

```math
X_k
=
|X_k|e^{j\phi_k}
```

로 표현됩니다.

* $X_k$: $k$번째 frequency component입니다.
* $|X_k|$: amplitude, 즉 해당 주파수의 크기입니다.
* $\phi_k$: phase, 즉 진동의 위상입니다.
* $j$: $j^2=-1$을 만족하는 허수 단위입니다.

SFT는 전체 window에 한 번의 Fourier transform을 적용하고,

```math
\tilde{\mathbf{x}}_{t+1:t+H}
=
g\left(
\mathcal F(
\mathbf{x}_{t-L+1:t}
)
\right)
```

와 같이 filtering을 수행합니다(p.2–3). 

문제는 같은 주파수라 하더라도 **시계열 앞부분과 뒷부분에서 중요도가 다를 수 있다는 점**입니다.

예를 들어,

```math
x_t
=
\text{slow trend}
+
\text{local oscillation}
+
\epsilon_t
```

일 때 전체 Fourier spectrum만 보면 local oscillation이 **언제** 나타났는지가 크게 희석됩니다.

---

# C. 핵심 제안: MMFT

논문의 Eq. (4)는 MMFT를 다음과 같이 추상화합니다(p.3).

```math
\tilde{\mathbf{x}}_{t+1:t+H}
=
h\left(
\left\{
\mathcal F_s
\left(
\mathbf{x}_{t-L+1:t}
\right)
\right\}_{s=1}^{S}
\right)
```

* $s$: temporal scale의 index입니다.
* $S$: 사용하는 전체 scale 수입니다.
* $\mathcal F_s$: $s$번째 scale에서 수행하는 frequency transformation입니다.
* $h$: 서로 다른 scale에서 얻은 frequency representation을 filtering·결합하는 mapping입니다.
* $\tilde{\mathbf{x}}$: frequency processing을 거친 prediction representation입니다. 

MMFNet v1은 실질적으로

$$
S=3
$$

이며,

$$
\{\text{fine},\text{intermediate},\text{coarse}\}
$$

세 scale을 사용합니다.

---

# D. 1단계 — Multi-scale Fragmentation

Figure 1의 첫 단계입니다.

입력은 normalization 후 서로 다른 segment 길이로 reshape됩니다.

```math
X^{(s)}
=
\text{Reshape}
\left(
x^{(d)},\,n_s,s_s
\right)
```

여기서 논문 Algorithm 1의 표기 기준으로

* $s_s$: 해당 scale의 segment length입니다.
* $n_s$: 전체 window 안의 segment 개수입니다.
* 대체로 $L=n_s s_s$가 되도록 구성됩니다.
* $x^{(d)}$: RIN 이후의 normalized sequence입니다. 

**fine scale**은 짧은 segment로 빠른 국소 변화를, **coarse scale**은 긴 segment로 전역 추세를 표현하도록 설계됩니다(p.4). 

> **Scale**
> 여기서 scale은 값의 크기가 아니라 “한 번에 얼마만큼 긴 시간 구간을 분석하는가”를 뜻합니다.

### 중요한 해석

짧은 segment에서 DCT를 하는 것은 단순히 “더 높은 주파수를 본다”기보다,

**시간 위치를 더 잘 국소화하는 대신 frequency resolution은 거칠게 하는 trade-off**가 있습니다.

따라서 MMFNet은 엄밀한 의미의 STFT는 아니지만,

$$
\text{time localization}
\leftrightarrow
\text{frequency resolution}
$$

사이의 multi-resolution trade-off를 활용하는 구조로 해석할 수 있습니다.

이 부분은 **제 해석**이며 논문이 이렇게 수학적으로 표현하지는 않습니다.

---

# E. 2단계 — DCT

각 segment에 대해 Eq. (5)를 사용합니다.

```math
X_k
=
\sum_{n=0}^{N-1}
x_n
\cos
\left[
\frac{\pi}{N}
\left(
n+\frac{1}{2}
\right)k
\right]
```

(p.4) 

기호는 다음과 같습니다.

* $x_n$: 한 segment 안의 $n$번째 시간값입니다.
* $n=0,\ldots,N-1$: segment 내부 시간 index입니다.
* $N$: segment 길이입니다.
* $k$: frequency index입니다.
* $X_k$: $k$번째 cosine basis에 투영된 coefficient입니다.
* $\cos(\cdot)$: 해당 frequency basis입니다.

즉 원래 벡터

$$
[x_0,x_1,\ldots,x_{N-1}]
$$

를

$$
[X_0,X_1,\ldots,X_{N-1}]
$$

로 바꿉니다.

> **DCT(Discrete Cosine Transform)**
> 데이터를 여러 개의 cosine 파형의 합으로 표현하는 변환입니다. FFT와 달리 이 표현에서는 실수 coefficient만으로 표현할 수 있어 구조가 비교적 단순합니다.

---

# F. 3단계 — Learnable Frequency Mask

가장 핵심적인 Eq. (6)입니다(p.5).

```math
X_{\text{mask,DCT}}
=
X_{\text{DCT}}
\odot M
```



* $X_{\text{DCT}}$: DCT coefficient입니다.
* $M$: 학습 가능한 frequency mask입니다.
* $\odot$: element-wise multiplication입니다.
* $X_{\text{mask,DCT}}$: mask 적용 후 spectrum입니다.

예를 들어

```math
X_{\text{DCT}}
=
[10,\;4,\;2,\;0.5]
```

이고

```math
M
=
[1,\;0.8,\;0.2,\;0]
```

라면

```math
X_{\text{mask,DCT}}
=
[10,\;3.2,\;0.4,\;0]
```

가 됩니다.

따라서 모델이

* $M_k\approx1$: 성분 유지
* $M_k\approx0$: 성분 억제

와 같은 형태로 학습한다면 adaptive spectral filtering으로 해석할 수 있습니다.

### 그런데 여기에는 중요한 불명확성이 있습니다.

논문 Figure 2–3 해설은 **“larger mask values indicate more aggressive attenuation”**, 즉 mask 값이 클수록 더 강한 attenuation이라고 기술합니다(p.16). 

그러나 Eq. (6)이 실제 mask coefficient 자체라면

$$
|M_k|\uparrow
\quad\Rightarrow\quad
|X_kM_k|\uparrow
$$

이므로 일반적으로는 **attenuation이 약해지는 방향**입니다.

더구나 Figure 2–3의 mask 값은 시각적으로 1을 넘는 경우도 보입니다.

따라서 첨부 v1만으로는 다음이 분명하지 않습니다.

```math
M
=
\text{multiplicative gain}
```

인지,

```math
M
=
\text{attenuation score}
```

를 별도로 시각화한 것인지 알 수 없습니다.

**이것은 논문에서 수학적으로 추가 설명이 필요한 핵심 지점입니다.**

> **Attenuation**
> 특정 주파수 성분의 amplitude를 줄이는 것입니다.

---

# G. 4단계 — Frequency-domain Linear Interpolation

Eq. (7)은

```math
X_{\text{pred,DCT}}
=
W
X_{\text{mask,DCT}}
+b
```

입니다(p.5). 

* $W$: trainable weight matrix입니다.
* $b$: bias vector입니다.
* $X_{\text{mask,DCT}}$: mask를 통과한 입력 spectrum입니다.
* $X_{\text{pred,DCT}}$: 예측 horizon에 대응되는 spectrum입니다.

중요한 점은 **복잡한 Transformer가 아니라 linear mapping이 실제 prediction head 역할을 한다는 것**입니다.

MMFNet의 성능 향상은 따라서 “큰 neural network”보다는,

$$
\boxed{
\text{좋은 표현공간}
+
\text{좋은 filtering}
+
\text{간단한 predictor}
}
$$

라는 설계에서 나옵니다.

---

# H. 5단계 — iDCT

논문의 Eq. (8)은 p.5에서 다음 형태로 제시됩니다.

```math
x_n
=
\frac{1}{2}x_0
+
\sum_{k=1}^{N-1}
X_k
\cos
\left[
\frac{\pi}{N}
\left(
n+\frac12
\right)k
\right]
```



다만 여기서 첫 항이 논문에는 $x_0$로 인쇄되어 있는데, 같은 식의 다른 항이 $X_k$이므로 **DCT coefficient인 $X_0$을 의미하는지 표기가 모호합니다**. 또한 DCT 종류에 따른 normalization coefficient도 상세히 명시되지 않았습니다.

따라서 저는 이를 임의로 “수정된 공식”으로 바꾸어 제시하지 않겠습니다.

---

# I. 전체 모델 구조

Figure 1과 Algorithm 1을 연결하면 다음과 같습니다.

$$
X
\rightarrow
\text{RIN}
\rightarrow
\begin{cases}
\text{Fine fragmentation}\\
\text{Intermediate fragmentation}\\
\text{Coarse fragmentation}
\end{cases}
$$

각 branch마다

$$
X^{(s)}
\rightarrow
\text{DCT}
\rightarrow
\odot M^{(s)}
\rightarrow
\text{Linear}
\rightarrow
\text{iDCT}
\rightarrow
\hat{x}^{(s)}
$$

이고 최종적으로

```math
x_M
=
\hat{x}^{(\text{fine})}
+
\hat{x}^{(\text{inter})}
+
\hat{x}^{(\text{coarse})}
+
e_t
```

를 거쳐 inverse normalization을 수행합니다(Algorithm 1, p.13). 

Figure 1에서는 이 전체 과정이 **① Fragmentation & Decomposition → ② Masking & Interpolation → ③ Spectral Inversion**으로 구성됩니다. 

---

## Channel-independent multivariate modeling

Algorithm 1은 사실 하나의 univariate historical window에 대해 동작하고, 여러 channel에는 **같은 parameter를 공유**하여 적용합니다(p.13). 

즉 v1의 핵심은

```math
\hat{x}^{(c)}
=
f_{\theta}(x^{(c)})
```

이며,

$$
\theta_1=\theta_2=\cdots=\theta_C=\theta
$$

인 형태에 가깝습니다.

PatchTST에서도 이 channel-independence 아이디어가 사용되며, parameter sharing과 긴 look-back 활용에 유리합니다. ([OpenReview][3])

> **Channel Independence(CI)**
> 온도, 압력, 유량 등의 각 변수에 같은 예측기를 독립적으로 적용하되 파라미터는 공유하는 방식입니다. parameter 수와 과적합 위험을 줄일 수 있지만, “온도가 증가하면 압력이 어떻게 반응하는가”와 같은 직접적인 변수 간 상호작용은 약해질 수 있습니다.

---

# 3. 성능 결과와 주장별 위치

## Table 1 — 일반 LTSF benchmark

7개 dataset × 4 horizon에 대한 MMFNet MSE는 다음 패턴입니다.

| Dataset     | MMFNet이 최저 MSE인 horizon 수 | 핵심 관찰                        |
| ----------- | ------------------------: | ---------------------------- |
| ETTh1       |                       4/4 | 전 horizon 1위                 |
| ETTh2       |                       4/4 | 전 horizon 1위                 |
| ETTm1       |                       2/4 | 336, 720에서 우세                |
| ETTm2       |                       4/4 | 전 horizon 1위                 |
| Weather     |                       1/4 | 720만 최저                      |
| Electricity |                       3/4 | 192, 336, 720 우세             |
| Traffic     |                       0/4 | PatchTST/TimeMixer가 더 낮은 MSE |
| **합계**      |                 **18/28** | **내 계산**                     |

원수치는 Table 1, p.6에 있습니다. 

### 저자가 강조한 숫자

**[저자 보고]**

* ETTh1, $H=336$: $0.427\rightarrow0.409$, 약 **4.2% MSE 감소**
* ETTh2, $H=336$: $0.354\rightarrow0.336$, 약 **5.1% 감소**
* ETTm1, $H=720$: second-best 대비 **4.6% 감소**
* ETTm2, $H=720$: $0.348\rightarrow0.327$, **6.0% 감소** 

### 내 해석

가장 설득력 있는 결과는 **ETT 계열, 특히 긴 horizon**입니다.

반면 Traffic에서는

$$
C=862
$$

로 channel 수가 매우 많고 MMFNet이 어느 horizon에서도 최고가 아닙니다. 논문 자체도 Traffic에서 PatchTST가 더 우수함을 인정합니다(p.7). 

이는 MMFNet이 temporal spectrum에는 강하지만,

$$
\text{explicit cross-channel structure}
$$

를 충분히 활용하지 못할 가능성을 제기합니다.

이는 **원인으로 입증된 사실이 아니라 제 연구 가설**입니다.

---

# 4. 저자가 직접 보고한 내용 vs 제 해석

| 주제               | **저자 직접 보고**                                     | **제 해석**                                                           |
| ---------------- | ------------------------------------------------ | ------------------------------------------------------------------ |
| 핵심 문제            | SFT가 non-stationarity/local pattern에 약함          | global spectrum의 time-localization 부족이 핵심                          |
| MMFT             | 여러 scale을 사용하면 local/global frequency pattern 포착 | 일종의 간단한 multi-resolution time-frequency representation             |
| Mask             | irrelevant frequency/noise adaptive filtering    | fixed low-pass보다 유연하지만 mask parameterization 설명 부족                 |
| 성능               | 최대 6.0% MSE 감소                                   | 최대값은 좋은 결과이나 평균적인 improvement와 동일시하면 안 됨                           |
| ETT 결과           | 매우 일관된 우수성                                       | MMFNet의 가장 강한 evidence                                             |
| Traffic          | PatchTST가 우수                                     | high-dimensional cross-variable modeling이 약점일 가능성                  |
| robustness       | 다양한 sampling rate/channel에서 잘 작동                 | distribution shift 자체를 실험한 것이 아니므로 “OOD robustness”로 확대 해석 불가      |
| non-stationarity | multi-scale이 적응에 유리                              | stationarity-breaking synthetic/control experiment가 없어 인과적 증거는 제한적 |
| noise filtering  | mask가 noise 억제                                   | 무엇이 실제 noise인지 ground truth가 없음                                    |
| generalization   | 여러 benchmark에서 양호                                | cross-dataset transfer/zero-shot generalization은 측정하지 않음           |

---

# 5. 통계적으로 취약하거나 비교 불가능한 부분

이 부분은 논문을 실제 연구에 적용하려면 상당히 중요합니다.

## 5-1. 반복 실험의 변동성이 보고되지 않음

Table 1–4에는

$$
\text{mean}\pm\text{std}
$$

또는 confidence interval이 없습니다.

예를 들어

$$
0.376 \quad \text{vs.}\quad 0.377
$$

처럼 차이가

$$
\Delta\text{MSE}=0.001
$$

인 조건도 있습니다.

그런데 seed에 따른 표준편차가 예를 들어

$$
\sigma_{\text{MSE}}=0.003
$$

이라면 이 차이는 모델 우열이라고 보기 어렵습니다.

첨부 v1 전체에서 exact search 기준으로 `"random seed"`가 발견되지 않았고, `"learning rate"`도 발견되지 않았습니다.  

따라서 **0.001–0.006 수준 차이는 통계적으로 유의하다고 말할 근거가 없습니다.**

---

## 5-2. “최대 6.0%”는 best-case improvement

논문 abstract의

> up to 6.0%

은 **평균 6%가 아니라 가장 좋은 특정 조건에서 6%**라는 의미입니다(p.1). 

따라서

$$
\boxed{\text{MMFNet은 평균적으로 SOTA보다 6% 좋다}}
$$

라고 표현하면 잘못입니다.

---

# 5-3. Table 2 본문과 실제 표가 일치하지 않음

p.7 본문은 ETTh1에서 MFT $(N_{\text{seg}}=360)$의 결과를

$$
0.160,\;0.212,\;0.259,\;0.327
$$

이라고 기술합니다. 

그런데 실제 Table 2의 ETTh1 MFT $(N_{\text{seg}}=360)$은

$$
0.366,\;0.403,\;0.418,\;0.425
$$

입니다. 

앞의 $0.160,0.212,0.259,0.327$은 오히려 **Table 1의 ETTm2 MMFNet 값과 동일**합니다.

따라서 **명백한 text/table inconsistency**가 있습니다.

---

# 5-4. Table 3은 dataset header와 값이 충돌함

Table 3의 header는

* ETTh1
* ETTh2
* Electricity
* Traffic

이라고 되어 있습니다.

그런데 mask row의 세 번째 숫자열은

$$
0.307,\;0.334,\;0.358,\;0.396
$$

이고 네 번째는

$$
0.160,\;0.212,\;0.259,\;0.327
$$

입니다. 

이 두 sequence는 Table 1에서 각각 정확히

* **ETTm1**
* **ETTm2**

MMFNet 결과와 일치합니다. 

따라서 첨부 v1만으로는

> Table 3의 dataset label이 잘못된 것인지, 값이 잘못 복사된 것인지

확정할 수 없습니다.

그 결과 **Electricity/Traffic에 대한 mask ablation 주장은 그대로 신뢰하기 어렵습니다.**

---

# 5-5. Table 4에도 비교 문제와 텍스트 오류가 있음

Ultra-long experiment는 Transformer 계열의 FEDformer, TimesNet, TimeMixer, PatchTST가 GPU memory limit 때문에 실행되지 않아

* DLinear
* FITS
* SparseTSF
* MMFNet

만 비교합니다(p.9). 

따라서

$$
\text{MMFNet > all SOTA}
$$

라고 결론 내릴 수 없고,

$$
\text{MMFNet > tested lightweight baselines}
$$

수준으로 해석해야 합니다.

또 p.9 본문은 $0.411,0.419,0.423,0.424$를 **ETTh1**이라고 부르지만 Table 4의 해당 column은 **ETTm1**입니다. 

게다가 Electricity $H=1440$에서는

$$
\text{DLinear}=0.277 < \text{MMFNet}=0.280
$$

인데 Table의 `Imp.`가 $+0.001$로 표기되어 있어 “best baseline 대비 improvement”라는 해석과도 완전히 맞지 않습니다.

---

# 5-6. Weather sampling rate 표기 불일치

Table 5에서는 Weather가 **10-minute sampling**으로 기록되어 있습니다(p.13). 

그런데 Appendix A.2 설명에서는 Weather를 **1-hour interval**이라고 기술합니다(p.14). 

본문 p.7 역시 Weather를 10-minute data로 설명하므로, Appendix 설명이 오기일 가능성이 높지만 **첨부본 자체에는 불일치가 존재합니다.**

---

# 5-7. ultra-long과 일반 LTSF 수치의 직접 비교 불가

$H=720$ 결과와 $H=1680$ 결과를 단순히

$$
0.327 \quad\text{vs.}\quad0.349
$$

식으로 비교해 “성능이 0.022만 떨어졌다”고 해석해서는 안 됩니다.

평가 horizon, prediction target sequence, baseline set이 모두 다르기 때문입니다.

---

# 5-8. 서로 다른 논문의 % improvement는 비교 불가능

예를 들어

* Autoformer: 자체 benchmark에서 특정 상대 향상 보고
* FEDformer: 자체 protocol에서 상대 error 감소 보고
* FiLM: 자체 baseline 대비 improvement 보고
* MMFNet: 자체 Table 1의 second-best 대비 최대 6%

는 모두 평가 프로토콜이 다릅니다. ([NeurIPS Proceedings][4])

따라서

> “FiLM 19.2% > MMFNet 6%, 따라서 FiLM이 무조건 더 좋다”

같은 비교는 **통계적으로 의미가 없습니다.**

---

# 6. 이 문서가 답하지 않는 질문

1. **Mask는 실제로 어떤 함수로 parameterize되는가?**
   $M=\sigma(A)$인지, unconstrained parameter인지, $M\in[0,1]$인지 명확하지 않습니다.

2. **왜 Figure 2–3에서 큰 mask가 더 큰 attenuation을 의미하는가?**
   Eq. (6)의 단순 곱셈과 직관적으로 충돌합니다.

3. **Fine/intermediate/coarse segment length는 어떻게 선택하는가?**
   validation search인지, domain knowledge인지 명확한 selection protocol이 없습니다.

4. **Scale를 바꾸었을 때 sensitivity curve는 어떻게 되는가?**
   Table 2는 일부 single-scale MFT만 보여 줍니다.

5. **왜 DCT가 FFT/STFT/wavelet보다 좋은가?**
   직접 비교가 없습니다.

6. **Mask에 sparsity나 smoothness regularization이 필요한가?**

7. **명시적 cross-channel dependency를 넣으면 Traffic 성능이 개선되는가?**

8. **실제 distribution shift에서 MMFNet이 더 robust한가?**
   예: 이전 연도 train → 새로운 계절 test.

9. **Unseen sampling rate에서도 scale이 transfer되는가?**

10. **Cross-dataset zero-shot 또는 transfer learning은 가능한가?**

11. **missing value 또는 irregular sampling에서는 어떻게 동작하는가?**

12. **모델의 parameter 수·FLOPs·latency·memory는 정확히 얼마인가?**

13. **여러 random seed의 평균과 표준편차는 얼마인가?**

14. **prediction interval/uncertainty는 제공할 수 있는가?**

15. **mask가 정말 noise를 제거했는지 ground-truth frequency component로 확인할 수 있는가?**

16. **각 scale의 예측을 단순 합산하는 것이 최적인가?**
    scale별 learnable gate가 더 나을 수 있습니다.

---

# 7. 가장 중요한 “그림” 5개의 해석

정확성을 위해 한 가지를 먼저 지적해야 합니다. **첨부 v1에는 번호가 붙은 실제 Figure가 Figure 1–3까지 3개뿐**입니다. 따라서 존재하지 않는 Figure 4·5를 만들어내지 않겠습니다.

대신 논문의 핵심 시각적 evidence 5개를
**Figure 1, Table 1, Table 2, Table 3, Figures 2–3의 paired diagnostic**으로 해석하겠습니다.

---

## ① Figure 1 — MMFNet Architecture, p.3

Figure 1은 논문 전체에서 가장 중요합니다. 

왼쪽에서 오른쪽으로 보면

$$
X
\rightarrow
\text{RIN}
\rightarrow
\text{multi-scale segmentation}
\rightarrow
\text{DCT}
\rightarrow
\text{Mask}
\rightarrow
\text{Linear}
\rightarrow
\text{iDCT}
\rightarrow
\text{aggregation}
$$

입니다.

### 핵심 해석

MMFNet은 사실 세 종류의 복잡성을 분리합니다.

**시간적 범위**

$$
\text{fine/intermediate/coarse}
$$

**frequency selection**

$$
M^{(s)}
$$

**prediction**

$$
W^{(s)}X+b
$$

입니다.

즉 “어떤 시간 범위를 볼지”와 “그 범위의 어떤 주파수를 볼지”를 분리했다는 점이 설계상 장점입니다.

---

# ② Table 1 — Main Benchmark, p.6

Table 1에서 가장 눈에 띄는 것은 단순히 “평균적으로 좋다”가 아니라 **dataset별 성격 차이**입니다. 

### 강한 영역

ETTh1 / ETTh2 / ETTm2:

$$
\text{MMFNet wins almost uniformly}
$$

### 약한 영역

Traffic:

$$
\text{MMFNet never ranks first in 4 horizons}
$$

따라서 모델의 inductive bias는 **moderate-channel periodic/temporal structure**에는 매우 잘 맞지만, 매우 높은 차원의 cross-sensor relationship에서는 충분하지 않을 수 있습니다.

---

# ③ Table 2 — SFT → MFT → MMFT, p.8

ETTh1 horizon 336 예를 보면

$$
\text{SFT}=0.427
$$

$$
\text{MFT}_{24}=0.412
$$

$$
\text{MMFT}=0.409
$$

입니다. 

즉 성능 개선을 두 단계로 볼 수 있습니다.

$$
\text{global spectrum}
\rightarrow
\text{localized spectrum}
\rightarrow
\text{multi-resolution localized spectrum}
$$

이것은 MMFNet의 가장 설득력 있는 ablation입니다.

단, 앞서 설명한 **p.7 본문 숫자 오류** 때문에 표 자체를 기준으로 해석하는 것이 안전합니다.

---

# ④ Table 3 — Mask Ablation, p.8

ETTh1 H=96에서는

$$
0.372
\rightarrow
0.359
$$

로 감소합니다.

반면 H=336/720에서는

$$
0.410\rightarrow0.409
$$

$$
0.420\rightarrow0.419
$$

로 개선 폭이 매우 작습니다. 

### 핵심 해석

mask가 항상 큰 성능 차이를 만드는 것은 아닙니다.

따라서 MMFNet의 성능을

$$
\boxed{\text{MMFT decomposition}}
$$

과

$$
\boxed{\text{mask}}
$$

중 어디에서 더 많이 얻는가를 보면, 적어도 일부 ETT 설정에서는 **multi-scale decomposition의 기여가 더 큰 것처럼 보입니다.**

단, Table 3의 dataset label 오류 때문에 전체 dataset으로 일반화하면 안 됩니다.

---

# ⑤ Figures 2–3 — Learned Mask, p.15–16

ETTh1과 ETTh2의 세 scale mask를 시각화합니다.

첨부 v1의 Figure caption에서는 segment length가

$$
2,\;24,\;720
$$

입니다. 

저자는 scale 및 temporal position에 따라 high-frequency attenuation 강도가 달라진다고 해석합니다(p.16). 

### 제가 중요하게 보는 점

그림은 mask가 상수가 아니라 상당히 복잡한 frequency-dependent curve를 학습한다는 점을 보여 줍니다.

따라서 MMFNet은 사실상

```math
M
=
M(s,k,\text{segment})
```

에 가까운 구조를 학습하고 있다고 볼 수 있습니다.

하지만 Eq. (6)만 보면 이 mask의 정확한 constraint 및 “값이 클수록 attenuation”이라는 설명을 재현하기 어렵습니다.

즉 Figure 2–3은 **흥미로운 interpretability evidence이면서 동시에 수학적 정의가 보강되어야 할 evidence**입니다.

---

# 8. 결론 — 저자가 제시한 시사점과 후속 연구

## 저자가 실제로 결론에서 주장하는 것

Conclusion(p.10)은 다음 세 가지를 강조합니다.

1. MMFT가 long-term multivariate forecasting을 개선한다.
2. multi-scale decomposition으로 다양한 시간 패턴을 포착한다.
3. learnable mask로 noise/irrelevant component를 adaptive하게 줄인다. 

### 중요한 점

첨부 v1의 Conclusion에는 명시적인

> “Future work에서 우리는 X를 연구할 것이다”

라는 구체적 연구계획이 없습니다.

따라서 저자의 후속 연구 계획을 제가 임의로 만들어낼 수는 없습니다.

현재 공식 repository는 연구가 **SAC 2026에 채택**되었다고 업데이트했습니다. ([GitHub][5])

---

# 8-1. 일반화 성능을 어떻게 더 높일 수 있는가

여기부터는 **제 연구 제안**입니다.

## 제안 1. 고정 세 scale을 학습 가능한 scale로 변경

첨부 v1의 Figure 2–3에서는

$$
\{2,24,720\}
$$

이 사용됩니다. 그런데 현재 공식 repository 설명은

$$
\{2,360,1440\}
$$

을 fine/intermediate/coarse 예시로 제시합니다. ([GitHub][5])

이 **버전 간 차이 자체가 scale이 데이터 의존적이라는 사실**을 보여 줍니다.

더 일반적인 구조는

```math
\hat y
=
\sum_{s=1}^{S}
\alpha_s(X)
f_s(X)
```

입니다.

여기서

$$
\alpha_s(X)\ge0,\qquad
\sum_s\alpha_s(X)=1
$$

가 되도록 하면 데이터마다 적절한 scale의 가중치를 자동 선택할 수 있습니다.

예:

```math
\alpha_s
=
\frac{\exp(a_s)}
{\sum_j\exp(a_j)}
```

이렇게 하면 특정 dataset에 맞춘 고정 segment length 의존도를 줄일 수 있습니다.

---

## 제안 2. mask의 수학적 제약을 명확하게 하기

예를 들어

```math
M
=
\sigma(A)
```

로 정의하면

$$
0 < M_k <1
$$

이므로 mask가 확실한 attenuation coefficient가 됩니다.

그리고

```math
\mathcal L
=
\mathcal L_{\text{forecast}}
+
\lambda_1\|M\|_1
+
\lambda_2
\sum_k
|M_{k+1}-M_k|
```

같은 regularization을 사용할 수 있습니다.

* $\mathcal L_{\text{forecast}}$: 예측 오차
* $|M|_1$: 불필요한 frequency를 sparse하게 만드는 항
* 두 번째 항: 인접 frequency의 mask가 지나치게 요동하지 않도록 하는 smoothness penalty
* $\lambda_1,\lambda_2$: 규제 강도

이렇게 하면 mask가 더 안정적이고 해석 가능해질 수 있습니다.

---

# 제안 3. Channel Independence + Cross-Variable branch 병렬화

Traffic 결과를 고려하면 가장 중요한 개선 방향 중 하나입니다.

MMFNet의 CI branch를 유지하면서

```math
Z_{\text{CI}}
=
f_{\text{MMF}}(X)
```

추가로 저랭크 cross-variable branch를

```math
Z_{\text{CV}}
=
U
\left(
V^\top Z_{\text{CI}}
\right)
```

로 두는 것입니다.

최종적으로

```math
Z
=
Z_{\text{CI}}
+
\gamma Z_{\text{CV}}
```

로 구성하면 됩니다.

$U,V$의 rank를 작게 유지하면

$$
O(C^2)
$$

full channel attention 대신 훨씬 적은 parameter로 변수 간 관계를 학습할 수 있습니다.

이 방향은 2025년 **FilterTS**의 Dynamic Cross-Variable Filtering 및 **FreEformer**의 cross-variate frequency attention과도 연결됩니다. ([AAAI Publications][6])

---

# 제안 4. DCT segmentation → 진짜 time-frequency localization

non-stationarity를 더 직접 다루려면

* STFT
* wavelet transform
* learnable filter bank

를 고려할 수 있습니다.

예를 들어

```math
X(\tau,\omega)
=
\sum_t
x_t
w(t-\tau)
e^{-j\omega t}
```

와 같이 **시간 위치 $\tau$와 주파수 $\omega$를 동시에** 표현하면,

> “어느 시점에서 어떤 frequency가 발생했는가?”

를 더 직접적으로 모델링할 수 있습니다.

MMFNet의 fragmentation은 이 방향의 단순하고 계산 효율적인 근사로 해석할 수 있습니다.

---

# 제안 5. Sampling-rate invariant scale

현재 공식 repository도 실제 데이터 적용 시 **consistent sampling rate**와 적절한 `scale_factors` 선택을 요구합니다. ([GitHub][5])

이것은 일반화 측면에서 중요한 제약입니다.

segment length $s$ 자체 대신 물리적 기간

```math
T_s
=
s\Delta t
```

를 기준으로 scale을 정의하는 편이 더 좋습니다.

예를 들어

$$
T_s
\in
\{1\text{ hour},1\text{ day},1\text{ week}\}
$$

로 설정하고 dataset의 sampling interval $\Delta t$에 따라

```math
s
=
\frac{T_s}{\Delta t}
```

를 계산합니다.

이렇게 하면 10분 데이터와 1시간 데이터 간 transfer가 더 자연스럽습니다.

---

# 제안 6. Generalization 평가 자체를 강화

현재의 단일 benchmark split보다 다음 실험이 필요합니다.

### Temporal rolling-origin

$$
\text{Train}_1
\rightarrow
\text{Test}_1
$$

$$
\text{Train}_2
\rightarrow
\text{Test}_2
$$

$$
\cdots
$$

여러 미래 구간에서 반복 평가합니다.

### Distribution shift

$$
P_{\text{train}}(X,Y)
\neq
P_{\text{test}}(X,Y)
$$

가 되도록 계절·regime·sampling-rate shift를 의도적으로 만듭니다.

### Multi-seed confidence interval

$$
\bar{m}
\pm
1.96
\frac{s_m}{\sqrt{K}}
$$

* $\bar m$: 여러 seed의 평균 MSE
* $s_m$: seed 간 표준편차
* $K$: 반복 횟수

를 보고해야 합니다.

### Paired test

같은 timestamp의 error에 대해 bootstrap 또는 Diebold–Mariano 계열 검정을 적용해야 합니다.

---

# 제안 7. RevIN을 명확하게 사용하여 distribution shift 대응

첨부 MMFNet은 “Reversible Instance-wise Normalization”을 사용한다고 하지만 Lai et al. (2021)을 인용합니다. 반면 **RevIN으로 널리 알려진 방법의 원 논문은 Kim et al., ICLR 2022의 “Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift”**입니다. ([OpenReview][7])

따라서 재현 연구에서는 normalization implementation이 실제 RevIN과 동일한지 코드를 반드시 확인해야 합니다.

RevIN의 일반적인 개념은 instance별로

```math
z_t
=
\frac{x_t-\mu_x}
{\sigma_x+\epsilon}
```

를 적용하고 prediction 후

```math
\hat{x}_t
=
\hat z_t
(\sigma_x+\epsilon)
+
\mu_x
```

로 되돌리는 것입니다.

이는 평균·scale drift에 대한 일반화에 특히 유용합니다.

---

# 8-2. 2020년 이후 관련 최신 연구 비교

여기서는 **논문별 발표된 % 향상치를 서로 직접 비교하지 않습니다.** 평가 protocol이 달라 숫자를 한 ranking에 올리는 것은 부적절합니다.

| 연도      | 연구                      | 핵심 방법                                                                           | MMFNet과의 관계 / 일반화 관점                                                                                            |
| ------- | ----------------------- | ------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| 2021    | **Informer**            | ProbSparse attention, $O(L\log L)$                                              | 긴 horizon 계산 효율 문제를 Transformer 관점에서 해결 ([AAAI Publications][8])                                                |
| 2021    | **Autoformer**          | progressive decomposition + Auto-Correlation                                    | trend/seasonality 구조적 분해를 deep block 안에 도입 ([NeurIPS Proceedings][4])                                           |
| 2022    | **FEDformer**           | seasonal-trend decomposition + sparse Fourier representation                    | frequency-domain LTSF의 중요한 선행 연구; linear complexity ([Proceedings of Machine Learning Research][9])             |
| 2022    | **FiLM**                | Legendre memory + Fourier projection + low-rank                                 | 역사정보 보존과 frequency noise removal에 초점 ([NeurIPS Proceedings][10])                                                |
| 2022    | **RevIN**               | reversible instance normalization                                               | distribution shift 대응이라는 일반화 축을 명시적으로 다룸 ([OpenReview][7])                                                      |
| 2023    | **DLinear/LTSF-Linear** | one-layer linear forecasting                                                    | 복잡한 모델 없이도 강한 baseline이 가능함을 입증 ([AAAI Publications][11])                                                       |
| 2023    | **PatchTST**            | patching + channel independence                                                 | local patch + CI라는 구조가 MMFNet과 일부 철학적으로 유사하며 transfer/pretraining도 가능 ([OpenReview][3])                         |
| 2023    | **FreTS**               | frequency-domain MLP, inter/intra-series modeling                               | frequency global view뿐 아니라 channel-wise dependency도 학습 ([NeurIPS Proceedings][12])                              |
| 2024    | **FITS**                | complex frequency interpolation, 약 10k params                                   | MMFNet이 직접 문제 삼는 주요 lightweight baseline; global frequency filtering ([ICLR Proceedings][2])                    |
| 2024    | **TimeMixer**           | decomposable multiscale mixing                                                  | 시간영역에서 fine/coarse scale을 계층적으로 혼합; MMFNet과 multi-scale 철학 공유 ([OpenReview][13])                                |
| 2024    | **SparseTSF**           | Cross-Period Sparse Forecasting, <1k params                                     | small/noisy data 및 계산 효율·generalization 측면의 매우 강한 비교대상 ([Proceedings of Machine Learning Research][14])         |
| 2024    | **Fredformer**          | frequency debiasing                                                             | low-energy high-frequency를 Transformer가 무시하는 현상을 명시적으로 분석; MMFNet의 high-frequency 보존 주장과 매우 관련 ([arXiv][15])    |
| 2024    | **JTFT**                | joint time-frequency + learnable frequencies + low-rank cross-channel attention | MMFNet보다 명시적으로 local time information과 channel relationship을 결합 ([ScienceDirect][16])                           |
| 2024    | **MMFNet**              | multi-scale DCT + learnable masking                                             | lightweight adaptive multi-resolution frequency filtering이 핵심 ([arXiv][1])                                      |
| 2025    | **TimeMixer++**         | multi-scale time + multi-resolution frequency representations                   | MMFNet의 “multi-scale”을 더 일반적인 time-frequency multi-resolution learning으로 확장한 연구 흐름과 일치 ([ICLR Proceedings][17]) |
| 2025    | **FilterTS**            | Dynamic Cross-Variable + Static Global frequency filtering                      | MMFNet v1의 CI 한계를 보완하는 방향으로 특히 중요 ([AAAI Publications][6])                                                      |
| 2025    | **FreEformer**          | frequency Transformer + cross-variate attention                                 | frequency representation 위에서 변수 간 관계를 명시적으로 학습                                                                  |
| 2025/26 | **AFMT**                | adaptive frequency decomposition + multi-scale patch Transformer                | 고정 decomposition 대신 adaptive spectral filter와 multi-scale modeling을 통합                                          |
| 2026    | **MMFNet SAC version**  | MMFNet의 학술대회 버전                                                                 | 공식 저장소는 SAC 2026 채택을 명시함                                                                                        |

---

# 관련 연구의 발전 방향을 한 줄 흐름으로 보면

### 2021–2022

$$
\text{Long dependency}
\rightarrow
\text{decomposition}
\rightarrow
\text{frequency representation}
$$

Informer → Autoformer → FEDformer/FiLM

### 2023

$$
\text{복잡한 Transformer가 꼭 필요한가?}
$$

DLinear, PatchTST, FreTS가 각각

* 단순성
* patching
* frequency-domain learning

을 강조합니다.

### 2024

$$
\text{lightweight}
+
\text{multi-scale}
+
\text{frequency}
$$

FITS → TimeMixer → SparseTSF → Fredformer/JTFT → MMFNet으로 연구축이 세분화됩니다.

### 2025 이후

$$
\boxed{
\text{adaptive frequency}
+
\text{multi-resolution}
+
\text{cross-variable learning}
}
$$

방향이 더 강해집니다.

TimeMixer++, FilterTS, FreEformer, AFMT가 그 예입니다.

---

# MMFNet이 앞으로의 연구에 미치는 의미

직접적인 **citation-based causal influence**, 즉 “2025년의 특정 연구가 MMFNet 때문에 만들어졌다”는 주장은 제가 확인한 자료만으로는 확정할 수 없습니다.

따라서 그 표현은 사용하지 않겠습니다.

다만 **연구 방향의 관점에서** MMFNet은 중요한 위치에 있습니다.

기존 FITS 계열의 아이디어는 대략

$$
\text{global frequency}
+
\text{fixed filtering}
$$

이었다면,

MMFNet은

$$
\text{multiple local scales}
+
\text{learnable filtering}
$$

으로 옮깁니다.

그리고 더 최근 연구들은 이를 다시

$$
\text{multiple scales}
+
\text{adaptive frequency}
+
\text{cross-variable}
+
\text{time-frequency fusion}
$$

방향으로 확장하고 있습니다.

따라서 MMFNet의 중요한 연구적 의미는 **“주파수 기반 lightweight forecasting에서도 scale과 filter 자체를 학습 대상으로 만들 수 있다”**는 설계 철학에 있다고 보는 것이 타당합니다.

---

# 최종 연구자 관점 평가

## 장점

**1. 구조가 단순하면서 inductive bias가 강합니다.**
Transformer처럼 매우 큰 representation learner 없이도

$$
\text{segmentation}
+
\text{DCT}
+
\text{mask}
+
\text{linear}
$$

만으로 경쟁력 있는 성능을 냅니다.

**2. FITS의 가장 자연스러운 한계를 잘 겨냥합니다.**

$$
\text{fixed global low-pass}
\rightarrow
\text{adaptive multi-scale filtering}
$$

이라는 개선 논리가 명확합니다.

**3. ETT 계열의 실험 evidence가 강합니다.**

**4. 긴 forecasting horizon에서 성능이 상대적으로 유지되는 경향이 있습니다.**

**5. 해석 가능한 frequency mask라는 장점이 있습니다.**

---

## 가장 중요한 한계

### 1. 통계 검증 부족

$$
\text{single reported score}
\neq
\text{statistically significant superiority}
$$

입니다.

### 2. v1 내부 표·본문 inconsistency가 여러 곳 있습니다.

특히 Table 2, Table 3, Table 4는 재현 전에 원 코드를 확인해야 합니다.

### 3. Mask의 수학적 정의와 visualization semantics가 완전히 일치하지 않습니다.

### 4. 고정 scale 선택이 dataset-specific hyperparameter가 될 위험이 있습니다.

현재 repository 자체도 사용자에게 sampling rate에 맞는 scale 선택을 요구합니다.

### 5. v1의 channel-independent formulation은 high-dimensional multivariate 관계를 충분히 활용하지 못할 가능성이 있습니다.

### 6. “non-stationarity에 robust”라는 주장은 직접적인 distribution-shift benchmark로 검증된 것은 아닙니다.

### 7. 효율성을 강조하지만 첨부 v1에는 comprehensive parameter/FLOPs/latency Pareto table이 없습니다.

---

# 일반화 성능을 최우선으로 한다면 제가 제안하는 MMFNet 후속 모델

가장 연구 가치가 큰 형태는 다음입니다.

$$
\boxed{
\text{RevIN}
\rightarrow
\text{Adaptive Multi-Scale Time-Frequency Decomposition}
\rightarrow
\text{Bounded Sparse Mask}
\rightarrow
\text{CI Spectral Predictor}
+
\text{Low-Rank Cross-Variable Mixer}
\rightarrow
\text{Scale Gating}
}
$$

수식으로는

```math
Z_s
=
\mathcal F_s(X)
```

```math
M_s
=
\sigma(A_s)
```

```math
\tilde Z_s
=
Z_s\odot M_s
```

```math
H_s
=
W_s\tilde Z_s+b_s
```

```math
H_s^{*}
=
H_s
+
U_sV_s^\top H_s
```

```math
\alpha_s
=
\frac{\exp(g_s(X))}
{\sum_j\exp(g_j(X))}
```

```math
\hat Y
=
\text{iRIN}
\left[
\sum_s
\alpha_s
\mathcal F_s^{-1}
(H_s^{*})
\right]
```

로 설계할 수 있습니다.

이 구조는 원 MMFNet의 장점인

* lightweight
* multi-scale
* frequency masking

을 유지하면서

* dataset-specific scale 문제
* mask 불안정성
* cross-channel interaction 부족
* distribution shift

을 동시에 보완하는 방향입니다.

특히 **일반화**를 연구목표로 한다면 단순히 benchmark MSE를 더 낮추는 것보다,

$$
R_{\text{future}}(f)
$$

가 distribution shift에서도 안정적인지를 측정해야 합니다.

---

# 참고한 자료 및 사이트 — 실제 사용한 자료 제목

**주 분석 대상**

1. **Ma, Aitian; Luo, Dongsheng; Sha, Mo — “MMFNet: Multi-Scale Frequency Masking Neural Network for Multivariate Time Series Forecasting”**, arXiv:2410.02070v1, 2024. 첨부 PDF 및 arXiv.
2. **Official MMFNet GitHub Repository — aitianma/MMFNet**, 현재 SAC 2026 acceptance 및 구현 설명 확인에 사용.

**2020년 이후 비교 연구**

3. **“Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting”**, AAAI 2021.
4. **“Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting”**, NeurIPS 2021.
5. **“Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift”**, ICLR 2022.
6. **“FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting”**, ICML 2022.
7. **“FiLM: Frequency improved Legendre Memory Model for Long-term Time Series Forecasting”**, NeurIPS 2022.
8. **“Are Transformers Effective for Time Series Forecasting?”**, AAAI 2023 — LTSF-Linear/DLinear.
9. **“A Time Series is Worth 64 Words: Long-term Forecasting with Transformers”**, PatchTST, ICLR 2023.
10. **“Frequency-domain MLPs are More Effective Learners in Time Series Forecasting”**, FreTS, NeurIPS 2023.
11. **“FITS: Modeling Time Series with 10k Parameters”**, ICLR 2024.
12. **“TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting”**, ICLR 2024.
13. **“SparseTSF: Modeling Long-term Time Series Forecasting with 1k Parameters”**, ICML 2024.
14. **“Fredformer: Frequency Debiased Transformer for Time Series Forecasting”**, 2024.
15. **“A Joint Time-Frequency Domain Transformer for Multivariate Time Series Forecasting”**, Neural Networks 2024.
16. **“TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis”**, ICLR 2025.
17. **“FilterTS: Comprehensive Frequency Filtering for Multivariate Time Series Forecasting”**, AAAI 2025.
18. **“FreEformer: Frequency Enhanced Transformer for Multivariate Time Series Forecasting”**, IJCAI 2025.
19. **“AFMT: Adaptive Frequency Decomposition and Multi-Scale Transformer for Time Series Forecasting”**, Information Sciences.

---

### 최종 판단

**MMFNet의 연구 아이디어 자체는 상당히 명확합니다.** 핵심 기여는 DCT가 아니라,

$$
\boxed{
\text{single/global/fixed frequency filtering}
\quad\longrightarrow\quad
\text{multi-scale/local/adaptive frequency filtering}
}
$$

이라는 전환입니다.

그러나 첨부한 **arXiv v1의 숫자를 그대로 “통계적으로 검증된 SOTA”로 받아들이기는 어렵습니다.** 내부 표기 오류가 여러 곳 존재하고, seed variance나 confidence interval이 없으며, 일부 성능차가 $10^{-3}$ 수준이고, ultra-long 실험의 baseline set도 제한적입니다. 반대로 Table 2의 MMFT ablation과 ETT 계열의 반복적인 개선은 “multi-scale frequency decomposition이 실제로 유용할 가능성”을 충분히 보여 주는 evidence입니다.

**일반화 성능 관점에서 가장 중요한 후속 연구는 MMFNet을 더 깊게 만드는 것이 아니라, fixed scale을 adaptive하게 만들고, bounded mask와 저랭크 cross-variable branch를 추가한 뒤, rolling-origin + distribution-shift + multi-seed 검증으로 그 효과를 입증하는 것**이라고 판단합니다.

다음 연구 단계로는 **첨부 arXiv v1과 현재 SAC 2026 코드/버전을 직접 대조하여 `2/24/720 → 2/360/1440` 같은 구조 변경, Table 2–4 오류 수정 여부, 실제 mask 구현식과 학습 hyperparameter를 코드 수준에서 검증하는 작업**이 가장 가치가 큽니다.

[1]: https://arxiv.org/abs/2410.02070 "https://arxiv.org/abs/2410.02070"
[2]: https://proceedings.iclr.cc/paper_files/paper/2024/hash/701251e1db4a2e4dd2ef23f5265d5936-Abstract-Conference.html "https://proceedings.iclr.cc/paper_files/paper/2024/hash/701251e1db4a2e4dd2ef23f5265d5936-Abstract-Conference.html"
[3]: https://openreview.net/pdf?id=Jbdc0vTOcol "https://openreview.net/pdf?id=Jbdc0vTOcol"
[4]: https://proceedings.neurips.cc/paper_files/paper/2021/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html "https://proceedings.neurips.cc/paper_files/paper/2021/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html"
[5]: https://github.com/aitianma/MMFNet "https://github.com/aitianma/MMFNet"
[6]: https://ojs.aaai.org/index.php/AAAI/article/view/35438 "https://ojs.aaai.org/index.php/AAAI/article/view/35438"
[7]: https://openreview.net/pdf?id=cGDAkQo1C0p "https://openreview.net/pdf?id=cGDAkQo1C0p"
[8]: https://ojs.aaai.org/index.php/AAAI/article/view/17325 "https://ojs.aaai.org/index.php/AAAI/article/view/17325"
[9]: https://proceedings.mlr.press/v162/zhou22g.html "https://proceedings.mlr.press/v162/zhou22g.html"
[10]: https://proceedings.neurips.cc/paper_files/paper/2022/hash/524ef58c2bd075775861234266e5e020-Abstract.html "https://proceedings.neurips.cc/paper_files/paper/2022/hash/524ef58c2bd075775861234266e5e020-Abstract.html"
[11]: https://ojs.aaai.org/index.php/AAAI/article/view/26317 "https://ojs.aaai.org/index.php/AAAI/article/view/26317"
[12]: https://proceedings.neurips.cc/paper_files/paper/2023/hash/f1d16af76939f476b5f040fd1398c0a3-Abstract-Conference.html "https://proceedings.neurips.cc/paper_files/paper/2023/hash/f1d16af76939f476b5f040fd1398c0a3-Abstract-Conference.html"
[13]: https://openreview.net/pdf?id=7oLshfEIC2 "https://openreview.net/pdf?id=7oLshfEIC2"
[14]: https://proceedings.mlr.press/v235/lin24n.html "https://proceedings.mlr.press/v235/lin24n.html"
[15]: https://arxiv.org/abs/2406.09009 "https://arxiv.org/abs/2406.09009"
[16]: https://www.sciencedirect.com/science/article/pii/S0893608024002582 "https://www.sciencedirect.com/science/article/pii/S0893608024002582"
[17]: https://proceedings.iclr.cc/paper_files/paper/2025/hash/2b187165e28fdfdc0ffb34d1bfff2b0c-Abstract-Conference.html "https://proceedings.iclr.cc/paper_files/paper/2025/hash/2b187165e28fdfdc0ffb34d1bfff2b0c-Abstract-Conference.html"
