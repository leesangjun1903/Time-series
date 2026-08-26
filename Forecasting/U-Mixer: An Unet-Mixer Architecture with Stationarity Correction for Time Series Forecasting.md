# U-Mixer: An Unet-Mixer Architecture with Stationarity Correction for Time Series Forecasting

아래 분석은 **첨부된 arXiv v1 PDF**를 1차 근거로 삼고, 동일 논문의 **AAAI 2024 공식 출판본**을 대조한 뒤, 2020년 이후의 관련 연구를 ICLR·NeurIPS·AAAI·ICML 공식 자료 중심으로 추가 조사한 결과입니다. 첨부본은 arXiv:2401.02236v1이고, 최종 논문은 AAAI 2024, Vol. 38(13), pp. 14255–14262에 출판되었습니다.  ([AAAI Publications][1])

# 0. Executive Summary — 10문장 이내

1. **U-Mixer**는 시계열의 비정상성(non-stationarity) 때문에 깊은 네트워크에서 특징 분포와 시간 의존성이 불안정해지는 문제를 해결하기 위해 **Patch + Mixer + U-Net + Stationarity Correction**을 결합한 AAAI 2024 예측 모델입니다. ([AAAI Publications][1])
2. 모델은 시계열을 작은 patch로 나누고, **시간축 상호작용과 채널축 상호작용을 분리한 MLP Mixer**로 처리함으로써 서로 다른 변수의 분포 차이가 시간 패턴 학습을 방해하는 것을 줄이려 합니다.
3. 여기에 U-Net식 encoder-decoder와 skip connection을 사용하여 얕은 층의 국소 정보와 깊은 층의 추상 정보를 결합합니다. 
4. 가장 독창적인 부분은 입력 embedding과 network output 사이의 **평균 및 자기상관 구조를 맞추려는 stationarity correction**이며, 저자들은 이를 통해 정규화 과정에서 사라진 비정상 정보를 되살릴 수 있다고 주장합니다.
5. 저자 보고 기준으로 장기예측 8개 데이터셋·64개 metric setting 중 56개에서 1위를 기록했으며, 본문은 기존 최고 결과 대비 MSE/MAE가 각각 **14.5%/7.7% 개선**되었다고 보고합니다. ([AAAI Publications][2])
6. 다만 Table 1에는 첫 번째 improvement가 **14.8%**로 나타나 Abstract/본문의 14.5%와 미세한 불일치가 있고, U-Mixer에만 $\pm$ 값이 제공되는 반면 baseline에는 불확실성이 제공되지 않아 통계적 우월성을 엄밀히 검증하기 어렵습니다. ([AAAI Publications][2])
7. 더 중요한 이론적 문제는 논문이 $R$을 정규화된 autocorrelation으로 정의했음에도 $R(\alpha Y)\approx\alpha^2R(Y)$로 전개한다는 점으로, **일반적인 correlation은 양의 스케일 변화에 불변**이므로 식 (5)→(6)의 논리가 논문 표기만으로는 성립하지 않습니다.
8. 따라서 U-Mixer의 가장 가치 있는 아이디어는 “비정상성을 단순히 제거하지 말고 예측에 필요한 분포·시간구조를 복원하자”는 설계 철학이며, 향후에는 FAN·DDN·MoF·DTAF처럼 **주파수, 국소 분포, fat-tail, test-time shift까지 명시적으로 모델링하는 방향**으로 발전시키는 것이 일반화 성능 측면에서 더 타당합니다. ([NeurIPS Papers][3])

---

# 1-1. 연구의 목적과 필요성

## 연구가 해결하려는 근본 문제

저자들은 실제 시계열의 평균, 분산, 추세, 계절성 등이 시간에 따라 변화하는 **비정상성(non-stationarity)**을 핵심 문제로 봅니다. 특히 Mixer 계열 모델에서는 다음 세 가지 문제가 생긴다고 주장합니다.

첫째, 네트워크가 깊어질수록 얕은 계층의 특징이 여러 변환을 통과하면서 원래의 저수준 정보가 불안정하게 전달됩니다. 둘째, 여러 채널을 한꺼번에 섞으면 물리적 의미와 분포가 서로 다른 변수들이 동일한 feature space에서 상호작용하기 때문에 시간 패턴 학습 자체가 왜곡될 수 있습니다. 셋째, 입력을 정규화하면 학습은 쉬워지지만 미래의 level·trend·scale과 같은 **예측에 필요한 비정상 정보까지 제거될 수 있습니다.** 이러한 문제 정의는 논문 서론과 Related Work의 핵심 논리입니다. 

**용어 설명 — 비정상성(non-stationarity)**
시계열의 통계적 성질이 시간에 따라 변한다는 뜻입니다. 예를 들어 센서의 평균값이 장비 노후화에 따라 서서히 상승하거나, 시간에 따라 진폭이 달라지는 경우입니다.

**용어 설명 — Distribution shift**
학습 시점과 실제 예측 시점의 데이터 분포가 달라지는 현상입니다. 단순히 값이 커지는 것뿐 아니라 평균, 분산, 주파수 성분, 변수 사이 상관관계가 변하는 경우도 포함할 수 있습니다.

이 문제는 U-Mixer 이전에도 RevIN, Non-stationary Transformer 등에서 중요하게 다뤄졌습니다. RevIN은 각 instance의 통계량을 제거한 후 예측 결과에 다시 복원하는 접근이고, Non-stationary Transformer는 지나친 stationarization이 중요한 비정상 정보를 제거할 수 있다는 **over-stationarization** 문제를 직접 제기했습니다. ([ML Anthology][4])

U-Mixer의 차별점은 평균·분산만 되돌리는 것에서 한 걸음 더 나아가 **시간적 correlation structure까지 되돌리려 시도했다는 점**입니다.

---

# 2. 핵심 주장과 근거

| 핵심 주장                                                 | 저자가 제시한 근거                                                                                      | 위치                              | 평가                             |
| ----------------------------------------------------- | ----------------------------------------------------------------------------------------------- | ------------------------------- | ------------------------------ |
| 비정상성이 Mixer의 안정적 학습을 방해한다                             | 깊은 layer의 feature propagation, channel distribution 차이, prediction distribution shift를 세 문제로 제시 | PDF p.1–2 / AAAI pp.14255–14256 | 문제 설정은 타당하나 각 원인별 독립적인 실험은 없음  |
| Patch가 국소 temporal pattern 학습에 유리하다                   | 길이 $P$의 patch로 분할 후 embedding                                                                   | Figure 1, PDF p.3 / p.14257     | PatchTST 계열 연구와도 방향이 일치        |
| 시간축/채널축을 분리해야 한다                                      | Temporal MLP → transpose → Channel MLP                                                          | Figure 3, PDF p.4 / p.14258     | 합리적인 inductive bias            |
| U-Net 구조가 저수준·고수준 feature를 보존한다                       | 동일 level encoder와 decoder 사이 skip/merge                                                         | Figure 2, Eqs. (1)–(4)          | Ablation으로 일정 부분 지지            |
| Stationarity Correction이 temporal dependency 복원에 유효하다 | 평균과 autocorrelation을 입력과 출력에서 맞추도록 설계                                                           | Eqs. (5)–(8), PDF p.4           | **아이디어는 중요하지만 수학적 표기에 문제가 있음** |
| 완전한 U-Mixer가 component 제거 모델보다 좋다                     | ETTh2, Weather, M4 ablation                                                                     | Table 3, pp.14260–14261         | 방향성은 확인되나 통계검정 없음              |
| U-Mixer가 SOTA보다 우수하다                                  | 64개의 장기 forecasting metric 중 56개 최고                                                             | Table 1                         | 저자 보고상 강한 결과                   |
| $M=3$, $P=16$이 합리적이다                                  | ETTm2/Exchange sensitivity analysis                                                             | Figure 5, p.14261               | 두 dataset만 사용 → 일반화 결론은 제한적    |

저자들은 장기예측에서 56/64개의 metric cell에서 최고를 기록했다고 보고합니다. ([AAAI Publications][2])

---

# 2-1. 해결 문제 → 제안 방법 → 수식 → 모델 구조 → 성능 → 한계

## A. 문제의 수학적 정의

입력 시계열을

$$
X\in\mathbb{R}^{C\times L}
$$

미래 실제값을

$$
Y\in\mathbb{R}^{C\times H}
$$

라고 정의합니다.

모델 $M$은

$$
\hat{Y}=M(X)
$$

를 학습합니다.

여기서:

* $C$: 변수 또는 channel 수입니다.
* $L$: look-back, 즉 입력으로 사용하는 과거 시점 수입니다.
* $H$: forecasting horizon, 즉 앞으로 예측해야 하는 시점 수입니다.
* $X$: 과거 관측 시계열입니다.
* $Y$: 미래 실제값입니다.
* $\hat Y$: 모델이 예측한 미래입니다.

논문 정의는 PDF p.2, AAAI p.14256에 있습니다. ([AAAI Publications][2])

---

## B. 1단계 — Instance Normalization

논문 표기는

$$
\tilde X=
\frac{X-\mu_{\text{in}}}{\sigma_{\text{in}}}
$$

입니다.

$\mu_{\text{in}}$은 입력 channel별 평균이고, $\sigma_{\text{in}}$은 scale 정보를 나타냅니다.

이를 직관적으로 보면 원래

$$
[4300,4350,4400,4500,\ldots]
$$

처럼 절대 level이 큰 센서값을

$$
[-1.1,-0.6,-0.1,0.9,\ldots]
$$

와 같이 상대적인 변화 중심으로 바꾸어 모델이 패턴을 학습하기 쉽게 만드는 것입니다.

**중요한 표기 문제:** 논문 본문은 $\sigma_{\text{in}}$을 “variance”라고 서술하면서 직접 나눗셈에 사용합니다. 일반적인 표준화라면 denominator는 variance가 아니라 **standard deviation**입니다. 최종 denormalization 역시 $\times\sigma_{\text{in}}+\mu_{\text{in}}$이므로, 문서상 “variance”라는 단어가 잘못 쓰였을 가능성이 있지만 **논문 자체만으로 확정할 수는 없습니다.** PDF p.3을 참조해야 합니다. 

**용어 설명 — Instance Normalization**
전체 train set의 평균이 아니라 현재 입력 sample 자체의 평균과 scale을 기준으로 정규화하는 방식입니다. 시간에 따라 level이 변하는 시계열의 distribution shift를 줄이는 데 유리합니다.

---

## C. 2단계 — Patching

정규화된 시계열을 길이 $P$의 작은 조각으로 나눕니다.

Patch 수는 논문에서

$$
N=
\left\lfloor
\frac{L-P}{S}
\right\rfloor+2
$$

로 정의합니다.

여기서:

* $P$: 한 patch 안의 time point 수입니다.
* $S$: 인접 patch 사이 stride입니다.
* $N$: 생성되는 patch 개수입니다.
* $\lfloor\cdot\rfloor$: floor, 즉 내림 연산입니다.

논문은 마지막 시점을 $S$번 반복해서 padding하기 때문에 일반적인 patch 공식의 $+1$ 대신 $+2$가 됩니다.

생성된 patch tensor는

$$
X_p\in\mathbb{R}^{(C N)\times P}
$$

이고 이를 선형 projection합니다.

$$
X_d=X_pW_{\text{val}}+W_{\text{pos}}
$$

$$
W_{\text{val}}\in\mathbb{R}^{P\times D}
$$

$$
X_d\in\mathbb{R}^{(CN)\times D}
$$

입니다.

여기서 $D$는 latent embedding dimension이고 $W_{\text{pos}}$는 patch 위치 정보를 전달합니다. ([AAAI Publications][2])

**용어 설명 — Patch**
연속된 여러 time point를 하나의 토큰처럼 묶는 방법입니다. 한 점씩 보는 것보다 짧은 구간의 기울기·진동·peak·local pattern을 한 번에 표현할 수 있습니다.

---

# D. 3단계 — U-Net Encoder–Decoder

첫 encoder 입력은

```math
X_{\text{in},i}
=
\begin{cases}
X_d,&i=1\\
X_{\text{out},i-1},&1 < i\le M
\end{cases}
```

이고,

```math
X_{\text{out},i}
=
M_{\text{en},i}(X_{\text{in},i})
```

입니다.

여기서:

* $i$: U-Net level입니다.
* $M$: 총 level 수입니다.
* $M_{\text{en},i}$: $i$번째 encoder MLP block입니다.

Decoder에서는

```math
Y_{\text{in},i}
=
\begin{cases}
Y_{\text{out},i+1},&1\le i < M\\
X_{\text{out},i},&i=M
\end{cases}
```

를 사용하고, 논문 식 (4)는

```math
Y_{\text{out},i}
=
\begin{cases}
W_y
\left(
M_{\text{de},i}(Y_{\text{out},i+1})
+
M_{\text{de},i}(Y_{\text{in},i})
\right),
&i < M\\
M_{\text{de},i}(X_{\text{out},M}),
&i=M
\end{cases}
```

로 제시합니다. ([AAAI Publications][2])

여기서 $W_y$는 동일 level의 encoder feature와 decoder feature를 merge하는 linear layer입니다.

### 설계 의도

깊은 encoder는 보다 추상적인 장기 구조를 포착하고, 얕은 encoder는 세부적인 local pattern을 상대적으로 많이 보존합니다.

따라서

$$
\text{local feature}
+
\text{deep feature}
$$

를 결합하면 단순한 깊은 MLP보다 정보 손실을 줄일 수 있다는 것입니다.

**용어 설명 — Skip connection**
중간층 정보를 여러 layer를 거치지 않고 뒤쪽 layer에 직접 전달하는 연결입니다. 정보 손실과 gradient 문제를 완화합니다.

---

# E. 4단계 — Temporal Mixing과 Channel Mixing의 분리

Figure 3의 핵심은

$$
\text{Temporal Mixing}
\rightarrow
\text{LayerNorm}
\rightarrow
\text{Transpose}
\rightarrow
\text{Channel Mixing}
$$

입니다.

시간방향 MLP는 각 channel을 독립적으로 처리합니다. 이후 tensor를 transpose해서 채널방향 MLP를 수행합니다. ([AAAI Publications][2])

예를 들어 온도, RF power, pressure라는 세 변수의 분포가 매우 다르다면 처음부터 모든 값을 섞는 대신

$$
T:\quad t_1,t_2,\ldots,t_L
$$

$$
RF:\quad r_1,r_2,\ldots,r_L
$$

$$
P:\quad p_1,p_2,\ldots,p_L
$$

각각에서 시간 패턴을 먼저 추출한 다음

$$
[T_{\text{feat}},RF_{\text{feat}},P_{\text{feat}}]
$$

사이의 channel relation을 학습하는 방식입니다.

이는 **channel distribution 차이가 temporal modeling을 오염시키는 것을 줄이자**는 inductive bias입니다.

---

# F. 5단계 — Stationarity Correction

이 논문의 핵심이자 동시에 가장 주의해서 읽어야 하는 부분입니다.

저자는 embedding 입력 $X_d$와 U-Net 출력 $Y_d$ 사이에서 평균뿐 아니라 temporal dependency도 복원하려 합니다.

Autocorrelation을

```math
R_{i,j}(X_d)
=
\frac{
\text{Cov}(X_d^i,X_d^j)
}{
\sqrt{\sigma_x^i\sigma_x^j}
}
```

로 정의합니다.

여기서:

* $X_d^i$: $i$ lag에 대응하는 sequence입니다.
* $\text{Cov}(\cdot,\cdot)$: covariance입니다.
* $\sigma_x^i$: 논문 표현상 $X_d^i$의 variance입니다.
* $R_{i,j}$: 두 lagged series 사이 normalized dependence입니다.

저자의 목표 함수는

```math
M_\alpha
=
\left\|
R(X_d)-R(\alpha Y_d)
\right\|_F^2
```

입니다.

$|\cdot|_F$는 Frobenius norm입니다.

**용어 설명 — Frobenius norm**
행렬의 모든 원소를 제곱해서 더한 뒤 제곱근을 취한 값입니다. 두 행렬이 전체적으로 얼마나 다른지를 하나의 숫자로 측정합니다.

---

## 논문의 식 (6)

저자는 이를

```math
M_\alpha
=
\left\|
R(X_d)-\alpha^2R(Y_d)
\right\|_F^2
```

로 바꿉니다. ([AAAI Publications][2])

그리고

```math
\alpha_i
=
\sqrt{
\frac{
\sum_{j=1}^{L}
R_{i,j}(X_d)R_{i,j}(Y_d)
}{
\sum_{j=1}^{L}R_{i,j}^{2}(X_d)
}
}
```

를 제시합니다.

---

# 매우 중요한 수학적 검토

여기부터는 **저자의 주장이 아니라 제 검토**입니다.

### 문제 1 — Correlation은 일반적으로 scale invariant입니다

논문이 정의한

```math
R_{i,j}
=
\frac{\text{Cov}(X_i,X_j)}
{\sqrt{\text{Var}(X_i)\text{Var}(X_j)}}
```

는 실질적으로 correlation입니다.

양의 상수 $a$에 대해

```math
\text{Corr}(aX_i,aX_j)
=
\frac{a^2\text{Cov}(X_i,X_j)}
{\sqrt{a^2\text{Var}(X_i)a^2\text{Var}(X_j)}}
=
\text{Corr}(X_i,X_j)
```

입니다.

즉 일반적으로

$$
R(aY)\neq a^2R(Y)
$$

이고 오히려

$$
R(aY)=R(Y)
$$

가 됩니다.

따라서 논문의 $R$ 정의를 그대로 사용한다면 **식 (5)에서 식 (6)이 직접 도출되지 않습니다.**

만약 저자들이 실제 의도한 것이 normalized autocorrelation이 아니라 **autocovariance**였다면

```math
\text{Cov}(aY_i,aY_j)
=
a^2\text{Cov}(Y_i,Y_j)
```

이므로 식 (6)의 논리가 훨씬 자연스러워집니다.

따라서 이 부분은 논문의 중요한 **정의-유도 간 불일치 가능성**입니다.

---

## 문제 2 — 식 (7)의 denominator

식 (6)을 그대로 최소화한다고 해보겠습니다.

$t_i=\alpha_i^2$라고 두면

$$
\min_{t_i}
\sum_j
\left[
R_{ij}(X)-t_iR_{ij}(Y)
\right]^2
$$

입니다.

이를 $t_i$에 대해 미분하면

$$
-2\sum_jR_{ij}(Y)
\left[
R_{ij}(X)-t_iR_{ij}(Y)
\right]=0
$$

이고 따라서 일반적인 least-squares 해는

```math
t_i
=
\frac{
\sum_jR_{ij}(X)R_{ij}(Y)
}{
\sum_jR_{ij}^{2}(Y)
}
```

가 됩니다.

즉

$$
\alpha_i=
\sqrt{
\frac{
\sum_jR_{ij}(X)R_{ij}(Y)
}{
\sum_jR_{ij}^{2}(Y)
}
}.
$$

그런데 논문의 식 (7) denominator는

$$
\sum_jR_{ij}^{2}(X)
$$

입니다. ([AAAI Publications][2])

따라서 **논문 식 (6)을 그대로 최소화해 식 (7)을 얻었다고 보기 어렵습니다.**

이것은 단순한 해석 차이라기보다 논문 표기상 확인할 필요가 있는 부분입니다.

단, **실제 공개 코드가 어느 식을 구현했는지를 이 논문 본문만으로 단정해서는 안 됩니다.**

---

## 문제 3 — square root의 정의역

논문의

$$
\alpha_i=
\sqrt{\frac{\sum_jR_XR_Y}{\sum_jR_X^2}}
$$

에서

$$
\sum_jR_XR_Y<0
$$

이면 square root 내부가 음수가 될 수 있습니다.

논문은 이에 대한

$$
\max(\epsilon,\cdot)
$$

clipping이나 absolute value 등의 안정화 처리를 수식에서 명시하지 않습니다.

따라서 이것도 구현 세부사항이 필요한 부분입니다.

---

# G. FFT와 Wiener–Khinchin

계산 속도를 높이기 위해 저자들은 Wiener–Khinchin theorem을 이용합니다.

개념적으로 autocorrelation은 Fourier transform을 사용해 빠르게 계산할 수 있습니다.

일반적인 형태는

```math
r_{xx}
=
\mathcal{F}^{-1}
\left[
\mathcal{F}(x)
\overline{\mathcal{F}(x)}
\right]
```

입니다.

여기서:

* $\mathcal F$: Fourier transform입니다.
* $\mathcal F^{-1}$: inverse Fourier transform입니다.
* $\overline{\mathcal F(x)}$: complex conjugate입니다.

논문 식 (8)은 FFT 기반 계산을 제시하며, 계산 전에

$$
\bar X_d^i=X_d^i-\mu_x^i,
$$

$$
\bar Y_d^i=Y_d^i-\mu_y^i
$$

로 zero-centering합니다. ([AAAI Publications][2])

**주의:** 논문 식 (8) 표기에는 일반적인 correlation theorem에서 나타나는 complex conjugate가 명시적으로 보이지 않습니다. 이것이 notation simplification인지 실제 구현 방식인지 논문 본문만으로 판단하기 어렵습니다.

---

# H. 평균 복원

Stationarity correction 이후에는

```math
\hat Y_d
=
\alpha Y_d+\Delta\mu
```

```math
\Delta\mu
=
\mu_x-\mu_y
```

로 평균 차이를 보정합니다.

즉 전체 논리는

$$
\boxed{
\text{temporal dependence correction}
\rightarrow
\text{mean correction}
}
$$

순서입니다.

저자들은 평균 이동은 covariance에 영향을 주지 않지만 covariance 변화는 distribution에 영향을 미칠 수 있으므로 correlation/covariance를 먼저 맞춘다고 설명합니다. ([AAAI Publications][2])

---

# I. Forecasting head와 denormalization

최종 latent representation을 linear layer로 펼쳐

$$
\hat Y_p\in\mathbb R^{C\times(L+H)}
$$

로 만든 후 미래 영역

$$
\tau=L:L+H
$$

만 선택합니다.

최종 출력은

```math
\hat Y
=
\hat Y_p[:,\tau]\sigma_{\text{in}}
+
\mu_{\text{in}}
```

입니다. ([AAAI Publications][2])

즉 처음 제거했던 level과 scale을 마지막에 복원합니다.

---

# J. Loss function

논문은 MSE 대신 $L_1$ loss를 사용합니다.

```math
\mathcal L_{\text{U-Mixer}}
=
\frac1C
\sum_{i=1}^{C}
\left|
Y[i,:]-\hat Y[i,:]
\right|
```

입니다.

저자는 $L_1$이 MSE보다 outlier에 덜 민감하기 때문에 robustness에 유리하다고 설명합니다. ([AAAI Publications][2])

**표기상 주의:** $Y[i,:]$는 horizon 방향 vector인데 식 (9)에는 $H$ 방향 reduction이 명시적으로 적혀 있지 않습니다. 실제 loss는 elementwise 평균 또는 합을 수행해야 scalar가 되므로, 이것도 수식 표기의 생략으로 보는 것이 합리적입니다.

---

# 3. 주장별 Page / Figure / Table 위치

첨부 PDF 페이지와 AAAI 인쇄 페이지의 대응은 다음과 같습니다.

| 내용                            | PDF | AAAI page | Figure / Table / Eq.   |
| ----------------------------- | --: | --------: | ---------------------- |
| 문제 정의·세 가지 비정상성 문제            |   1 |     14255 | Introduction           |
| Related Work·stationarization |   2 |     14256 | —                      |
| 전체 architecture               |   3 |     14257 | **Figure 1**           |
| Patch embedding               |   3 |     14257 | —                      |
| U-Net encoder-decoder         |   3 |     14257 | **Figure 2, Eqs. 1–4** |
| Temporal/channel Mixer        |   4 |     14258 | **Figure 3**           |
| Stationarity correction       |   4 |     14258 | **Eqs. 5–8**           |
| 장기 forecasting 비교             |   5 |     14259 | **Table 1**            |
| Loss                          |   5 |     14259 | **Eq. 9**              |
| M4 단기 forecasting             |   6 |     14260 | **Table 2**            |
| Ablation                      |   6 |     14260 | **Table 3**            |
| 정성 예측 결과                      |   7 |     14261 | **Figure 4**           |
| $M,P$ 민감도                     |   7 |     14261 | **Figure 5**           |
| Conclusion                    |   7 |     14261 | —                      |
| References                    |   8 |     14262 | —                      |

AAAI 공식 출판 범위가 pp.14255–14262임은 공식 proceedings에서도 확인됩니다. ([AAAI Publications][1])

---

# 4. 저자가 직접 보고한 내용과 제 해석의 분리

| 주제             | **저자가 직접 보고한 내용**                                              | **제 해석 / 검토**                                                                   |
| -------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| 연구 문제          | 비정상성이 stable feature propagation과 distribution modeling을 어렵게 함 | 현실 시계열 forecasting의 핵심 문제를 적절히 겨냥함                                              |
| Patching       | local temporal details를 포착                                     | PatchTST 이후 검증된 좋은 inductive bias와 일치                                           |
| Channel 분리     | 채널별 distribution 차이가 temporal interaction을 방해하므로 분리            | 다변량 센서에서 특히 타당하나 channel interaction을 늦추면 유용한 instantaneous relation이 약해질 수도 있음 |
| U-Net          | low/high-level feature를 결합                                     | 작은 데이터에서는 skip connection이 정보 보존에 도움이 될 가능성 있음                                  |
| SC             | autocorrelation과 평균을 보정하여 non-stationary information 복원        | **의도는 매우 흥미롭지만 Eq.5–7은 수학적으로 재검증 필요**                                           |
| 장기예측           | 64개 metric 중 56개 1위                                            | 강한 benchmark result이나 통계적 significance는 입증되지 않음                                 |
| Robustness     | 여러 dataset에서 좋은 성능 → robust                                    | 동일 benchmark family 내 robustness이지 OOD generalization을 입증한 것은 아님                |
| Ablation       | UE와 SC 제거 시 모두 악화                                              | 두 component의 기여 방향은 확인됨                                                         |
| Sensitivity    | $M=3$, $P=16$이 적절                                              | ETTm2/Exchange 두 dataset만으로 universal optimum이라고 볼 수 없음                         |
| Generalization | 여러 dataset에서 우수                                                | cross-domain transfer, unseen regime, test-time drift는 시험하지 않음                  |

---

# 5. 통계적으로 취약한 부분과 비교 불가능한 수치

## 5.1 $\pm$ 값의 의미가 충분히 설명되지 않음

Table 1에서 U-Mixer는

$$
0.317\pm2\times10^{-3}
$$

처럼 uncertainty를 제공합니다.

그러나 baseline은 대부분 단일값만 있습니다. 또한 본문에는 random seed를 고정했다고 기술되어 있지만, $\pm$가 정확히

* 몇 회 반복의 standard deviation인지,
* standard error인지,
* 다른 seed 결과인지

명확히 설명되지 않습니다. ([AAAI Publications][2])

따라서

$$
\text{U-Mixer}=0.317\pm0.002
$$

와

$$
\text{Baseline}=0.338
$$

을 엄밀한 statistical test처럼 읽으면 안 됩니다.

---

## 5.2 “significantly worse”라는 표현

Ablation에서 저자들은 component 제거 결과가 “significantly worsed”라고 표현합니다. 그러나 Table 3에는 p-value, confidence interval, paired test, bootstrap 결과가 없습니다. ([AAAI Publications][2])

따라서 여기서 “significantly”는 **통계적 유의성**이 아니라 사실상 “수치상 명확하게 나빠졌다”는 서술적 표현으로 읽는 것이 안전합니다.

---

## 5.3 Abstract 14.5% vs Table 1 14.8%

Abstract와 본문은

$$
14.5\%\quad\text{MSE improvement}
$$

라고 적습니다. ([AAAI Publications][2])

그러나 Table 1의 Improvement row에서는 TimesNet 대비 첫 수치가 **14.8%**입니다. ([AAAI Publications][2])

따라서 정확하게 보고할 때는

> “저자 본문·초록은 14.5%라고 보고하지만 Table 1에는 14.8%가 표시되어 있다.”

라고 해야 합니다.

제가 임의로 어느 것이 맞다고 정정하지 않겠습니다.

---

## 5.4 U-Mixer가 모든 setting에서 최고는 아님

예를 들어 ETTm2 horizon 336에서:

$$
\text{U-Mixer MSE}=0.331
$$

$$
\text{TimesNet}=0.321
$$

$$
\text{ETSformer}=0.314
$$

이고 horizon 720에서도 U-Mixer보다 TimesNet의 MSE가 낮습니다. Exchange horizon 96도 ETSformer가 약간 우수합니다. ([AAAI Publications][2])

즉 올바른 결론은

> **“대부분의 setting에서 우수하다”**

이지

> **“모든 데이터에서 가장 좋다”**

가 아닙니다.

---

## 5.5 이후 논문과 개선율을 직접 비교하면 안 됨

예를 들어:

* U-Mixer: 14.5%/7.7%
* FAN: MSE 7.76–37.90% 개선
* MoF: 특정 setup에서 평균 6.3%
* Dish-TS: 20% 이상

이라는 숫자들을 나란히 놓고 FAN > Dish-TS > U-Mixer라고 결론내리는 것은 **비교 불가능**합니다.

데이터셋, backbone, baseline, forecasting horizon, normalization, improvement denominator가 서로 다르기 때문입니다. ([NeurIPS Papers][3])

---

# 6. 이 문서가 답하지 않는 중요한 질문

1. 식 (5)의 normalized correlation에서 식 (6)의 $\alpha^2$ scaling이 정확히 어떻게 유도되는가?
2. 식 (7)의 denominator가 왜 $R(Y)^2$가 아니라 $R(X)^2$인가?
3. $\alpha_i$ square root 내부가 음수가 되는 경우 어떻게 처리하는가?
4. 식 (8)의 Fourier correlation 계산에서 complex conjugate가 생략된 이유는 무엇인가?
5. $X_d\in\mathbb R^{(CN)\times D}$라고 정의하면서 왜 $R(X_d)\in\mathbb R^{L\times L}$인가? 어느 dimension에 대해 correlation을 계산하는가?
6. U-Net이라고 부르지만 각 level의 정확한 dimensionality, downsampling/upsampling은 어떻게 구성되는가?
7. Table 1의 $\pm$는 몇 회 반복 실험의 어떤 통계량인가?
8. baseline도 동일한 seed와 동일 환경에서 다시 학습했는가, 아니면 원 논문의 수치를 가져왔는가?
9. Stationarity Correction이 실제 **stationarity**를 개선했음을 ADF/KPSS 같은 통계검정으로 확인했는가?
10. Test distribution이 training/validation보다 훨씬 달라지는 명시적 OOD regime에서도 SC가 작동하는가?
11. 갑작스러운 regime change를 input window에서 관측하지 못했을 때 horizon distribution을 어떻게 예측하는가?
12. SC 추가에 따른 연산량·메모리·latency overhead는 얼마인가?
13. 매우 작은 데이터셋에서 $M=3$ U-Net Mixer가 단순 linear/PLS/Ridge보다 안정적인가?
14. 변수 수 $C$가 수백~수천으로 증가할 때 channel Mixer가 안정적으로 일반화하는가?

이 질문들은 실제 산업 적용에서 benchmark MSE보다 훨씬 중요할 수 있습니다.

---

# 7. 가장 중요한 그림 5개 해석

## Figure 1 — 전체 U-Mixer architecture

$$
X
\rightarrow
\text{Normalization/Patching}
\rightarrow
\text{Embedding}
\rightarrow
\text{U-Net Mixer}
\rightarrow
\text{SC}
\rightarrow
\text{Forecast Head}
\rightarrow
\hat Y
$$

의 전체 흐름을 보여줍니다.

가장 중요한 점은 **Stationarity Correction이 입력 preprocessing이 아니라 deep representation 이후에 위치한다는 것**입니다.

따라서 U-Mixer는

> “처음부터 비정상 정보를 보존한다.”

라기보다

> “예측하기 쉽게 정규화한 뒤, 네트워크 후단에서 일부 통계·시간 구조를 다시 주입한다.”

는 구조입니다. ([AAAI Publications][2])

---

## Figure 2 — U-Net Encoder–Decoder

동일 level encoder와 decoder 사이에 skip pathway가 있습니다.

이 구조의 핵심은

$$
\text{fine/local representation}
+
\text{deep/global representation}
$$

입니다.

다만 그림과 식에서 일반적인 이미지 U-Net처럼 명확한 spatial downsampling/upsampling 연산은 충분히 설명되지 않습니다.

따라서 “U-Net의 topology를 빌린 multi-level MLP architecture”라고 이해하는 편이 정확합니다. ([AAAI Publications][2])

---

## Figure 3 — MLP Block

이 그림이 실제 U-Mixer의 forecasting backbone을 가장 잘 설명합니다.

먼저

$$
\text{time mixing}
$$

을 channel별로 따로 수행한 다음 transpose하여

$$
\text{channel mixing}
$$

을 수행합니다.

이 방식은 이후 PatchTST, iTransformer 등의 연구에서 반복해서 나타난 **time modeling과 variable modeling의 역할 분리**라는 큰 흐름과 연결됩니다. PatchTST는 channel independence를 강조했고, iTransformer는 반대로 variable token을 통해 cross-variate dependency를 명확히 모델링합니다. ([OpenReview][5])

---

## Figure 4 — 실제 prediction trajectory

ETTh2와 Traffic은 주기성이 뚜렷하고 U-Mixer가 반복 pattern을 잘 따라갑니다.

반면 ETTh1과 Electricity처럼 패턴이 불명확한 경우는 trend/shape 추적이 상대적으로 어렵습니다. 저자 역시 이 차이를 언급합니다. ([AAAI Publications][2])

중요한 점은 **Figure 4가 일반화 성능의 통계적 증거는 아니라는 점**입니다.

몇 개의 예시 trajectory를 잘 맞췄다는 사실만으로 전체 test distribution에 대한 robustness를 입증할 수는 없습니다.

---

## Figure 5 — $M$과 $P$ sensitivity

$M$은 U-Net depth이고 $P$는 patch length입니다.

저자 결과에서는:

$$
M=3
$$

부근이 전반적으로 좋고,

$$
P=16
$$

이 성능과 연산 시간의 타협점으로 선택됩니다. ([AAAI Publications][2])

여기서 중요한 해석은 $M$보다 $P$에 더 민감할 수 있다는 점입니다.

Patch length는 사실상 모델이 어떤 시간 scale을 “하나의 local pattern”으로 간주할지를 결정하기 때문입니다.

따라서 새로운 산업 데이터에서는 $P=16$을 그대로 사용하는 것보다 실제 공정 주기나 autocorrelation length에 맞춰 결정하는 것이 더 합리적입니다.

---

# 8. 결론과 저자가 제시한 시사점

## 저자의 결론

저자들은 U-Mixer의 기여를 세 축으로 정리합니다.

$$
\boxed{
\text{Patch/Mixer}
+
\text{U-Net multi-level fusion}
+
\text{Stationarity Correction}
}
$$

입니다.

저자들은 이 구조가 local temporal dependency, channel interaction, low/high-level feature를 동시에 처리하면서 비정상성 때문에 발생하는 distribution shift까지 줄일 수 있다고 결론내립니다. ([AAAI Publications][2])

논문 자체에는 구체적인 후속 연구 계획이 길게 제시되어 있지는 않습니다. 따라서 그 이상은 **저자의 계획이 아니라 제가 제안하는 연구 방향**으로 구분해야 합니다.

---

# 8-1. 모델의 일반화 성능 향상 가능성

U-Mixer가 일반화에 유리할 수 있는 요소는 분명히 있습니다.

### 첫째, instance-wise normalization

각 window의 level/scale 차이를 줄이면

$$
P_{\text{train}}(X)
\neq
P_{\text{test}}(X)
$$

일 때 생기는 단순 mean/variance shift에 덜 민감해질 수 있습니다.

### 둘째, patch representation

개별 point의 절대값보다

$$
\text{local shape}
$$

를 학습하게 만들어 scale 변화에 상대적으로 강해질 가능성이 있습니다.

### 셋째, channel/time separation

새로운 regime에서 특정 channel의 scale만 달라질 경우 다른 channel의 temporal representation이 함께 흔들리는 것을 줄일 가능성이 있습니다.

### 넷째, skip connection

깊은 layer만 사용했을 때 생기는 representation drift를 완화할 수 있습니다.

---

## 그러나 현재 논문만으로는 “미래 분포에 대한 진정한 일반화”가 증명되지 않았습니다

논문의 split은 ETT에서 6:2:2, 기타 dataset에서 7:1:2로 chronological order를 사용합니다. 이는 random shuffle보다 적절합니다. 

그러나 한 번의 chronological split은

$$
\text{temporal generalization}
$$

을 일부 확인할 뿐,

$$
\text{OOD generalization}
$$

이나

$$
\text{regime-shift robustness}
$$

를 직접 검증하지 않습니다.

---

# 제가 제안하는 U-Mixer 후속 일반화 연구

가장 유망한 구조는 단순히 SC를 더 복잡하게 만드는 것이 아니라 **shift를 종류별로 분리하는 것**입니다.

예를 들어

```math
z_{\text{shift}}
=
[
\Delta\mu,\,
\Delta\sigma,\,
\Delta R,\,
\Delta f,\,
\Delta q
]
```

라고 정의할 수 있습니다.

여기서:

* $\Delta\mu$: mean shift
* $\Delta\sigma$: variance/scale shift
* $\Delta R$: temporal correlation shift
* $\Delta f$: frequency/seasonality shift
* $\Delta q$: tail/quantile distribution shift

입니다.

그 뒤 forecasting을

```math
\hat Y
=
g_\theta
\left(
\text{U-Mixer}(X),
z_{\text{shift}}
\right)
```

로 수행하도록 확장하는 것입니다.

이 방식은 “비정상성 = 평균·분산·correlation 하나의 문제”로 보지 않고 여러 종류의 shift로 분해합니다.

2024~2026 연구의 발전 방향도 실제로 이쪽으로 이동하고 있습니다.

---

# 8-2. 2020년 이후 최신 관련 연구 비교

아래 성능 퍼센트들은 **각 논문의 자체 실험 기준이므로 서로 직접 숫자 비교를 하면 안 됩니다.**

| 연도   | 연구                             | 핵심 아이디어                                            | U-Mixer와의 관계                                     | 일반화 관점                          |
| ---- | ------------------------------ | -------------------------------------------------- | ------------------------------------------------ | ------------------------------- |
| 2022 | **RevIN**                      | instance별 normalize → denormalize                  | U-Mixer의 normalization 배경                        | mean/scale shift 대응             |
| 2022 | **Non-stationary Transformer** | stationarization + de-stationary attention         | “비정상 정보를 복원해야 한다”는 직접 선행연구                       | over-stationarization 해결        |
| 2023 | **DLinear**                    | 단순 linear decomposition                            | 복잡한 architecture가 항상 필요하지 않음을 보여줌                | small-data baseline 필수          |
| 2023 | **PatchTST**                   | patch + channel independence                       | U-Mixer patch/time separation과 강하게 연결            | transfer/self-supervised 가능     |
| 2023 | **Dish-TS**                    | input/output distribution을 각각 학습                   | U-Mixer보다 horizon shift를 직접 모델링                  | inter-space shift 대응            |
| 2023 | **SAN**                        | instance 전체가 아니라 temporal slice별 normalize         | global normalization보다 세밀                        | 빠른 local drift 대응               |
| 2023 | **TimesNet**                   | periodicity를 2D representation으로 변환                | U-Mixer의 periodic pattern modeling 경쟁 방식         | multi-period structure          |
| 2024 | **U-Mixer**                    | U-Net Mixer + autocorrelation correction           | 기준 논문                                            | mean + temporal structure       |
| 2024 | **iTransformer**               | variable 자체를 token화                                | channel modeling의 반대 관점                          | cross-variate generalization 강화 |
| 2024 | **TimeMixer**                  | multi-scale trend/seasonality mixing               | U-Mixer multi-level 철학을 scale decomposition으로 발전 | multi-scale robustness          |
| 2024 | **ModernTCN**                  | 큰 receptive field의 modern convolution              | Mixer/Transformer가 필수라는 주장에 반례                   | architecture choice 확대          |
| 2024 | **DDN**                        | time + wavelet frequency dual-domain normalization | U-Mixer SC보다 명시적인 dual-domain shift 처리           | time-frequency drift            |
| 2024 | **FAN**                        | dominant frequency를 제거·복원·예측                       | correlation보다 seasonality shift에 집중              | evolving periodicity            |
| 2025 | **MoF**                        | fat-tail normalization + test-time adaptation      | U-Mixer가 다루지 않은 tail/drift 문제                    | 배포 단계 적응                        |
| 2025 | **TSSA**                       | 어떤 shift가 error를 만들었는지 attribution                 | 모델보다 먼저 shift를 진단                                | generalization evaluation 강화    |
| 2026 | **DTAF**                       | temporal stabilization + frequency differencing    | U-Mixer 이후 non-stationarity 연구의 확장               | temporal + spectral shift 동시 대응 |

---

## RevIN — 2022

RevIN은 time-series distribution shift에 대해

$$
x'=
\frac{x-\mu_x}{\sigma_x}
$$

로 정규화한 뒤 예측 결과에 통계량을 역변환하는 매우 단순하고 model-agnostic한 접근입니다. ([ML Anthology][4])

U-Mixer는 기본 normalization 철학에서 RevIN과 유사하지만, 단순 mean/scale 복원에 더해 temporal dependency까지 복원하려고 했다는 차이가 있습니다.

---

## Non-stationary Transformer — NeurIPS 2022

이 연구는 중요한 개념인 **over-stationarization**을 제안했습니다.

입력을 너무 stationary하게 만들면 burst, level change 같은 예측에 필요한 정보까지 사라질 수 있다는 것입니다.

그래서

$$
\text{Series Stationarization}
+
\text{De-stationary Attention}
$$

구조를 사용합니다. ([NeurIPS Papers][6])

U-Mixer의 Stationarity Correction은 이 연구 문제와 매우 직접적으로 연결됩니다.

---

## Dish-TS — AAAI 2023

Dish-TS는 distribution shift를

$$
\text{intra-space shift}
$$

와

$$
\text{inter-space shift}
$$

로 나눕니다.

특히 **입력 window의 분포와 미래 horizon의 분포가 같다는 보장이 없다**고 지적합니다. ([AAAI Publications][7])

이것은 U-Mixer의 일반화 한계와 직결됩니다.

U-Mixer는 주로 입력 statistics를 활용해 prediction을 복원하지만, Dish-TS처럼 **future distribution 자체를 별도로 예측**하는 확장이 더 강한 regime shift 대응책이 될 수 있습니다.

---

## SAN — NeurIPS 2023

SAN은 하나의 instance 전체에 평균·분산 하나만 적용하는 것을 너무 거칠다고 봅니다.

대신 짧은 temporal slice마다 statistics를 추정합니다. ([NeurIPS Proceedings][8])

즉

$$
(\mu,\sigma)_{\text{whole window}}
$$

대신

$$
(\mu_t,\sigma_t)_{\text{local slice}}
$$

를 사용하는 방향입니다.

빠르게 aging되는 sensor나 공정 drift에서는 U-Mixer의 global instance normalization보다 이 방식이 더 적절할 가능성이 있습니다.

---

## PatchTST — ICLR 2023

PatchTST는

1. Patching
2. Channel independence

두 가지를 핵심으로 사용합니다. ([OpenReview][5])

U-Mixer와 매우 유사한 철학이 있지만 PatchTST는 Transformer를 사용하고, self-supervised pretraining과 transfer에서도 좋은 결과를 보고했습니다.

일반화 측면에서는 **U-Mixer backbone을 self-supervised patch pretraining과 결합**하는 것이 흥미로운 후속 연구가 됩니다.

---

## iTransformer — ICLR 2024

iTransformer는 반대로 각 variable 전체 history를 하나의 token으로 만듭니다.

즉

```math
\text{token}_c
=
[x_{c,1},x_{c,2},\ldots,x_{c,L}]
```

로 구성한 뒤 attention으로 변수 관계를 학습합니다. ([ICLR Proceedings][9])

U-Mixer의 “temporal first → channel later” 설계와 비교하면 매우 흥미로운 대조군입니다.

---

## TimeMixer — ICLR 2024

TimeMixer는 여러 sampling scale에서

$$
\text{fine scale}
\leftrightarrow
\text{seasonality}
$$

$$
\text{coarse scale}
\leftrightarrow
\text{trend}
$$

를 분리해서 mixing합니다. ([ICLR Proceedings][10])

U-Mixer의 U-Net multi-level feature fusion보다 **시계열적 의미가 더 명시적인 multi-scale decomposition**이라고 볼 수 있습니다.

따라서 U-Mixer 후속 연구에서는 U-Net level을 단순 network depth가 아니라 실제 공정 time scale과 연결하는 것이 좋습니다.

---

## DDN — NeurIPS 2024

DDN은 non-stationarity를 time domain 하나에서만 처리하지 않고 wavelet transform으로 low/high frequency를 분리합니다.

$$
x
\xrightarrow{\text{DWT}}
(x_{\text{low}},x_{\text{high}})
$$

를 만든 뒤 각 component에서 sliding normalization을 수행합니다. 

이는 U-Mixer의 autocorrelation correction보다 훨씬 명시적으로

$$
\text{time drift}
+
\text{frequency drift}
$$

를 분리합니다.

---

## FAN — NeurIPS 2024

FAN은 mean/variance 기반 normalization이 **변화하는 seasonality를 충분히 표현하지 못한다**고 지적합니다.

Fourier transform으로 dominant frequency component를 찾아 제거하고, 입력과 출력 사이의 frequency shift를 별도의 MLP로 예측합니다. ([NeurIPS Papers][3])

저자 보고 기준으로 여러 backbone과 8개 benchmark에서 MSE가 7.76%–37.90% 개선됐습니다.

다만 앞서 설명했듯이 이것을 U-Mixer의 14.5%와 직접 비교해서는 안 됩니다.

---

## MoF — ICML 2025

MoF는 한 단계 더 나아가 stationarization 이후에도 남아 있는 **fat-tailed distribution**과 지속적 distribution shift를 다룹니다.

Spline-based transform과 test-time training을 결합합니다. ([Proceedings of Machine Learning Research][11])

이는 산업 sensor에서 매우 중요한 방향입니다.

평균과 covariance가 같더라도 extreme event probability가 다르면

$$
P(|X-\mu|>k\sigma)
$$

가 크게 달라질 수 있기 때문입니다.

U-Mixer는 이런 higher-order/tail distribution을 직접 모델링하지 않습니다.

---

## TSSA — ICLR 2025

TSSA의 메시지는 더욱 근본적입니다.

> 모든 distribution shift를 동일한 방식으로 처리하기 전에 **실제로 무엇이 모델 성능을 떨어뜨렸는지 측정하라**는 것입니다.

시간 의존성, multivariate interaction, trend 변화 등이 error에 미치는 영향을 attribution합니다. ([ICLR Proceedings][12])

U-Mixer의 후속 연구에서도 SC를 무조건 사용하는 대신

$$
\text{shift diagnosis}
\rightarrow
\text{appropriate correction}
$$

순으로 진행하는 것이 더 과학적입니다.

---

## DTAF — AAAI 2026

2026년 AAAI의 DTAF는 비정상성을

$$
\text{temporal distribution shift}
+
\text{spectral variability}
$$

라는 dual-domain 문제로 직접 정의합니다.

Temporal Stabilizing Fusion은 non-stationary mixture-of-experts filter를 사용하고, Frequency Wave Modeling은 frequency differencing으로 spectral shift를 강조합니다. ([AAAI Publications][13])

U-Mixer 이후 연구의 방향이 **단일 평균/상관 구조 복원 → 시간·주파수별 적응적 shift modeling**으로 발전하고 있다는 점을 보여주는 좋은 사례입니다.

---

# 최종 연구자 관점의 평가

U-Mixer의 가장 중요한 기여는 “MLP Mixer가 강하다” 자체가 아닙니다.

더 중요한 아이디어는

$$
\boxed{
\text{Predictability를 위해 stationarize하되,
forecasting에 필요한 non-stationarity를 다시 복원하자}
}
$$

입니다.

이 아이디어는 RevIN → Non-stationary Transformer → Dish-TS/SAN → U-Mixer → FAN/DDN/MoF/DTAF로 이어지는 연구 흐름에서 분명한 위치를 갖습니다.

다만 U-Mixer의 **Stationarity Correction 수식은 그대로 받아들이기보다는 재검증이 필요합니다.** 특히

$$
R(\alpha Y)=\alpha^2R(Y)
$$

라는 전개와 식 (7)의 least-squares 해는 논문이 정의한 normalized correlation과 일관되지 않습니다.

따라서 후속 연구에서는 U-Mixer의 아이디어를 그대로 복제하기보다 다음과 같이 재설계하는 것이 더 타당합니다.

$$
\boxed{
\text{Patch representation}
+
\text{multi-scale feature fusion}
+
\text{time/channel selective mixing}
+
\text{local distribution prediction}
+
\text{frequency-aware correction}
+
\text{shift-aware validation}
}
$$

특히 **일반화 성능**이 목적이라면 단일 train/valid/test split의 평균 MSE를 최적화하기보다 여러 시간 regime에 대한 rolling-origin evaluation, multiple-seed uncertainty, explicit distribution-shift stress test, unseen-device/channel test까지 포함해야 합니다. U-Mixer의 architecture는 좋은 출발점이지만, 2024–2026 연구를 반영하면 “autocorrelation 하나를 복원하는 SC”보다 **어떤 종류의 shift가 발생했는지 식별한 뒤 해당 shift만 보정하는 adaptive correction**으로 발전시키는 것이 가장 유망합니다.

---

# 참고한 자료 및 사이트 제목

이번 답변에서 직접 참고한 주요 자료는 다음과 같습니다.

1. **Ma et al., “U-Mixer: An Unet-Mixer Architecture with Stationarity Correction for Time Series Forecasting”**, AAAI 2024 — 첨부 arXiv v1 및 AAAI 공식 proceedings.  ([AAAI Publications][1])
2. **Kim et al., “Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift”**, ICLR 2022. ([ML Anthology][4])
3. **Liu et al., “Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting”**, NeurIPS 2022. ([NeurIPS Papers][6])
4. **Zeng et al., “Are Transformers Effective for Time Series Forecasting?”**, AAAI 2023. ([AAAI Publications][14])
5. **Fan et al., “Dish-TS: A General Paradigm for Alleviating Distribution Shift in Time Series Forecasting”**, AAAI 2023. ([AAAI Publications][7])
6. **Liu et al., “Adaptive Normalization for Non-stationary Time Series Forecasting: A Temporal Slice Perspective”**, NeurIPS 2023. ([NeurIPS Proceedings][8])
7. **Nie et al., “A Time Series is Worth 64 Words: Long-term Forecasting with Transformers”**, ICLR 2023. ([OpenReview][5])
8. **Wu et al., “TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis”**, ICLR 2023. ([OpenReview][15])
9. **Liu et al., “iTransformer: Inverted Transformers Are Effective for Time Series Forecasting”**, ICLR 2024. ([ICLR Proceedings][9])
10. **Wang et al., “TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting”**, ICLR 2024. ([ICLR Proceedings][10])
11. **Luo & Wang, “ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis”**, ICLR 2024. ([ICLR Proceedings][16])
12. **Dai et al., “DDN: Dual-domain Dynamic Normalization for Non-stationary Time Series Forecasting”**, NeurIPS 2024. 
13. **Ye et al., “Frequency Adaptive Normalization For Non-stationary Time Series Forecasting”**, NeurIPS 2024. ([NeurIPS Papers][3])
14. **Liu et al., “Slimming the Fat-Tail: Morphing-Flow for Adaptive Time Series Modeling”**, ICML 2025. ([Proceedings of Machine Learning Research][11])
15. **Liu et al., “Going Beyond Static: Understanding Shifts with Time-Series Attribution”**, ICLR 2025. ([ICLR Proceedings][12])
16. **Lu et al., “Towards Non-Stationary Time Series Forecasting with Temporal Stabilization and Frequency Differencing”**, AAAI 2026. ([AAAI Publications][13])
17. **XiangMa-Shaun/U-Mixer — official U-Mixer code repository**, 연구 코드의 존재와 공식 연결 여부 확인용. ([github.com][17])

[1]: https://ojs.aaai.org/index.php/AAAI/article/view/29337?utm_source=chatgpt.com "U-Mixer: An Unet-Mixer Architecture with Stationarity Correction for Time Series Forecasting | Proceedings of the AAAI Conference on Artificial Intelligence"
[2]: https://ojs.aaai.org/index.php/AAAI/article/download/29337/30523 "U-Mixer: An Unet-Mixer Architecture with Stationarity Correction for Time Series Forecasting"
[3]: https://papers.nips.cc/paper_files/paper/2024/hash/37c6d0bc4d2917dcbea693b18504bd87-Abstract-Conference.html "Frequency Adaptive Normalization For Non-stationary Time Series Forecasting"
[4]: https://mlanthology.org/iclr/2022/kim2022iclr-reversible/?utm_source=chatgpt.com "Reversible Instance Normalization for Accurate Time-Series Forecasting Against Distribution Shift | ML Anthology"
[5]: https://openreview.net/pdf?id=Jbdc0vTOcol&utm_source=chatgpt.com "Published as a conference paper at ICLR 2023"
[6]: https://papers.neurips.cc/paper_files/paper/2022/hash/4054556fcaa934b0bf76da52cf4f92cb-Abstract-Conference.html?utm_source=chatgpt.com "Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting"
[7]: https://ojs.aaai.org/index.php/AAAI/article/view/25914 "Dish-TS: A General Paradigm for Alleviating Distribution Shift in Time Series Forecasting | Proceedings of the AAAI Conference on Artificial Intelligence"
[8]: https://proceedings.neurips.cc/paper_files/paper/2023/hash/2e19dab94882bc95ed094c4399cfda02-Abstract-Conference.html "Adaptive Normalization for Non-stationary Time Series Forecasting: A Temporal Slice Perspective"
[9]: https://proceedings.iclr.cc/paper_files/paper/2024/hash/2ea18fdc667e0ef2ad82b2b4d65147ad-Abstract-Conference.html?utm_source=chatgpt.com "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting"
[10]: https://proceedings.iclr.cc/paper_files/paper/2024/hash/a7ac8a21e5a27e7ab31a5f42a0117bdb-Abstract-Conference.html?utm_source=chatgpt.com "TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting"
[11]: https://proceedings.mlr.press/v267/liu25bq.html "Slimming the Fat-Tail: Morphing-Flow for Adaptive Time Series Modeling"
[12]: https://proceedings.iclr.cc/paper_files/paper/2025/hash/50ea4cf434c84a50203d7145afc7b853-Abstract-Conference.html "Going Beyond Static: Understanding Shifts with Time-Series Attribution"
[13]: https://ojs.aaai.org/index.php/AAAI/article/view/39585 "Towards Non-Stationary Time Series Forecasting with Temporal Stabilization and Frequency Differencing | Proceedings of the AAAI Conference on Artificial Intelligence"
[14]: https://ojs.aaai.org/index.php/AAAI/article/download/26317/26089?utm_source=chatgpt.com "Are Transformers Effective for Time Series Forecasting?"
[15]: https://openreview.net/pdf/98c0a5bad8225b6d1baf5c74047c4d04bacfcfa1.pdf?utm_source=chatgpt.com "Published as a conference paper at ICLR 2023"
[16]: https://proceedings.iclr.cc/paper_files/paper/2024/file/86b1437c1e4c3b3c4debff98234a67e7-Paper-Conference.pdf?utm_source=chatgpt.com "Published as a conference paper at ICLR 2024"
[17]: https://github.com/XiangMa-Shaun/U-Mixer?utm_source=chatgpt.com "GitHub - XiangMa-Shaun/U-Mixer · GitHub"
