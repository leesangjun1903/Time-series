# Not All Frequencies Are Created Equal: Towards a Dynamic Fusion of Frequencies in Time-Series Forecasting

분석 기준은 **사용자가 첨부한 2024년 ACM MM 논문 PDF**를 1차 근거로 하고, 본문에 빠진 Appendix/구현 세부사항은 arXiv v3 및 저자 공식 GitHub를 교차검증했습니다. 논문의 핵심 아이디어와 저자가 보고한 결과는 그대로 구분해서 적고, 그 아래에 별도로 **검토 해석/통계적 취약점**을 표시하겠습니다. 이 논문은 MM 2024의 Zhang et al. 논문이며, **ICLR 2025의 동명 약어 FreDF, “FreDF: Learning to Forecast in the Frequency Domain”과는 다른 연구**입니다.  ([GitHub][1]) ([GitHub][2])

---

## 1. Executive Summary — 10문장 이내

1. 이 논문의 출발점은 시계열 예측에서 흔히 쓰이는 **“고주파수 = 잡음이므로 제거해야 한다”**는 가정이 데이터와 예측 구간에 따라 성립하지 않을 수 있다는 관찰입니다. 
2. 저자들은 ETT, Exchange-rate, Weather 등의 주파수 대역을 제거하는 실험을 통해 저·중·고주파 성분의 유용성이 상황별로 달라진다고 주장합니다.
3. 이를 바탕으로 시계열 예측을 시간영역의 직접 매핑이 아니라 **각 Fourier 성분에 대한 transfer function 학습 문제**로 재정식화합니다. 
4. 제안 모델 FreDF는 각 주파수를 분리한 뒤 주파수별 복소수 선형 변환 $H^{l,m}$을 학습하고, 각 성분의 예측을 다시 시간영역으로 변환하여 학습 가능한 가중치로 융합합니다. 
5. 저자들은 Rademacher complexity를 이용해 일반화 오차 상계를 제시하고, 특정 조건에서 dynamic fusion의 상계가 static fusion보다 작거나 같다고 주장합니다. 
6. Table 1에서는 FreDF가 여러 장기 다변량 예측 benchmark에서 iTransformer, PatchTST, FEDformer 등과 경쟁하거나 우세한 MSE/MAE를 보였으며, Table 4의 설정에서는 151.4K parameters로 비교 모델보다 훨씬 작습니다.  
7. 다만 저자가 말하는 **dynamic**은 공개 구현상 입력마다 달라지는 instance-conditioned weight가 아니라 학습 과정에서 최적화되는 하나의 전역 frequency-weight vector입니다. ([GitHub][3])
8. 따라서 이론의 핵심인 $\text{Cov}(w^h,l^h)<0$ 효과와 공개 구현 사이에는 해석상 간극이 있으며, 실제 모델의 일반화 우수성을 이 정리만으로 증명했다고 보기는 어렵습니다.
9. 또한 반복 실험 평균은 보고하지만 분산·신뢰구간·유의성 검정이 없고, 일부 Table 수치와 “70/80” 집계에는 논문 내부에서 재현하기 어려운 불일치가 존재합니다. ([arXiv][4])
10. 그럼에도 **“주파수를 일률적으로 버리지 말고 데이터·상황에 따라 적응적으로 취급해야 한다”**는 문제 제기는 이후 FilterTS, FreDN, FreqCycle, DTAF 같은 2025–2026년 연구 방향과 상당히 잘 맞으며, 이 부분이 이 논문의 가장 지속적인 연구적 가치라고 판단합니다. ([AAAI Publications][5]) ([AAAI Publications][6]) ([AAAI Publications][7]) ([AAAI Publications][8])

---

# 1-1. 연구 목적과 필요성

기존 Fourier 기반 forecasting에는 크게 두 가지 경향이 있었습니다. 하나는 FiLM이나 FITS처럼 고주파를 noise 가능성이 높은 영역으로 보고 제거·축소하는 방식이고, 다른 하나는 FreTS처럼 주파수 공간 자체에서 representation을 학습하는 방식입니다. FreDF의 문제 제기는 여기서 한 단계 더 나아갑니다. **어떤 주파수가 noise인지 signal인지는 데이터셋과 예측 시점에 따라 달라질 수 있으므로 frequency selection을 사전에 고정해선 안 된다**는 것입니다. 논문의 Figure 1이 이를 위한 motivation experiment입니다. 

> **Fourier component(푸리에 성분)**
> 복잡한 시계열을 서로 다른 주파수의 sine/cosine 성분으로 나누었을 때 각 주파수에 해당하는 성분입니다. 낮은 주파수는 장기적인 변화·trend를, 높은 주파수는 빠른 변동을 표현하는 경우가 많지만, **높은 주파수라고 해서 수학적으로 곧 noise라는 뜻은 아닙니다.**

Figure 1에서는 FFT 후 spectrum의 첫 $1/3$, 중간 $1/3$, 마지막 $1/3$을 각각 low/middle/high frequency로 정의하고 한 영역을 제거한 뒤 Transformer를 다시 학습합니다. 그 결과 ETTm1에서는 high-frequency 제거가 유리하지만, 다른 데이터에서는 같은 제거가 손해가 되고, Weather에서는 prediction horizon에 따라 유리한 대역 자체가 달라집니다. 

따라서 연구 목적은 단순한 “더 좋은 Fourier model”이 아니라 다음 문제로 정리됩니다.

$$
\boxed{
\text{각 frequency가 미래 예측에 기여하는 정도를 별도로 학습하고,
이를 적절히 fusion하자.}
}
$$

이 아이디어의 필요성은 특히 **non-stationary time series**에서 큽니다.

> **Non-stationarity(비정상성)**
> 시간에 따라 평균, 분산, 주기, 변수 간 관계 또는 spectrum이 달라지는 현상입니다. 실제 산업·기상·금융 데이터에서는 매우 흔하며, “과거에 중요했던 주파수가 미래에도 동일하게 중요하다”는 가정이 깨지는 원인입니다.

---

# 2. 논문의 핵심 주장과 근거

| 핵심 주장                                                             | 저자의 근거                                                 | 위치                               | 검토 해석                                                                             |
| ----------------------------------------------------------------- | ------------------------------------------------------ | -------------------------------- | --------------------------------------------------------------------------------- |
| 모든 frequency의 가치가 동일하지 않으며 high-frequency도 유용할 수 있다               | frequency band 제거 시 성능 변화가 dataset마다 다름                | p.3, **Figure 1**                | 핵심 문제 제기는 설득력 있음. 다만 frequency thirds라는 임의적 분할이고 한두 prediction curve는 통계적 증거로 부족함 |
| forecasting을 Fourier-domain transfer-function learning으로 표현할 수 있다 | LTI convolution + convolution theorem으로 Eq. (1)–(8) 유도 | p.4–5, Eq. (1)–(8)               | 수학적 동기는 명확하지만 **LTI assumption은 실제 비정상 시계열에 강한 가정**                               |
| frequency별 predictor를 독립적으로 두는 것이 유리하다                            | 각 frequency에 $H^{l,m}$ 적용                              | p.5, Eq. (11)–(13), **Figure 2** | frequency-specific transformation은 합리적이나 parameter 수가 $K$에 따라 증가                  |
| dynamic fusion이 static fusion보다 좋다                                | FreDF vs FreSF ablation                                | p.8, **Table 3**                 | 대부분 개선되지만 일부 숫자에 내부 불일치 존재                                                        |
| dynamic fusion의 generalization bound가 더 낮다                        | Rademacher bound와 covariance term, Theorem 4.2–4.3     | p.7, Eq. (17)–(22)               | **조건부 정리**이며 핵심 조건 $r(w^h,l^h)\le0$이 실제 모델에서 검증되지 않음                              |
| FreDF가 주요 baseline보다 정확하다                                         | 장기예측 Table 1                                           | p.6, **Table 1**                 | 전반적으로 강한 결과이나 uncertainty/significance가 없어 작은 차이의 의미는 불확실                         |
| parameter efficiency도 우수하다                                        | 151.4K vs 3.1M–14M                                     | p.8, **Table 4**                 | 비교된 모델에 대해서는 사실. 하지만 FITS(<10K)가 빠져 있어 “일반적으로 가장 작다”는 결론은 불가                      |

논문 자체는 이 네 가지를 주요 contribution으로 명시합니다. 

---

# 2-1. 해결하려는 문제 → 수식 → 모델 구조 → 결과 → 한계

## 2-1-1. 시간영역 forecasting을 transfer-function 문제로 바꾸기

저자는 먼저 시계열 생성 dynamics가 **Linear Time-Invariant system, LTI**로 근사될 수 있다고 가정합니다. p.4에서는 다음 식에서 시작합니다. 

```math
y(t)
=
\int_0^\infty
h(t-\tau)x(\tau)\,d\tau.
```

기호는 다음 의미입니다.

* $x(t)$: 관측된 과거 입력 signal입니다.
* $y(t)$: 예측하고 싶은 출력 signal입니다.
* $t$: 현재 시간입니다.
* $\tau$: 과거 입력을 적분할 때 사용하는 lag 변수입니다.
* $h(t-\tau)$: 과거 입력이 현재 출력에 얼마나 영향을 주는지 나타내는 impulse-response kernel입니다.

> **LTI — Linear Time-Invariant system**
> 입력을 두 배 하면 출력 효과도 두 배가 되는 **선형성**, 그리고 같은 입력 패턴이 언제 들어오더라도 동일한 방식으로 반응한다는 **시간 불변성**을 동시에 가정하는 시스템입니다.

여기서 중요한 점은 실제 시계열이 반드시 LTI라는 것이 아닙니다. 저자가 mathematical reformulation을 위해 이를 전제로 삼은 것입니다. 논문에서는 이를 “without loss of generality”에 가깝게 기술하지만, **비정상 forecasting 전체에 일반적으로 손실 없는 가정이라고 보기는 어렵습니다.**

이산시간에서는 convolution으로 바뀝니다.

```math
Y[n]
=
h[n]*X
=
\sum_{m=0}^{\infty}
h[n-m]X[m].
```

* $n$: discrete time index입니다.
* $m$: 과거 시점 index입니다.
* $*$: convolution 연산입니다.
* $h[n-m]$: $m$번째 입력이 $n$번째 출력에 영향을 주는 coefficient입니다.

---

## 2-1-2. Fourier domain으로 이동

DFT는 다음과 같습니다. 

```math
\mathcal{X}[k]
=
\mathcal{F}(X)[k]
=
\sum_{n=0}^{N-1}
X[n]
e^{-j\frac{2\pi}{N}kn}.
```

* $\mathcal{F}$: Discrete Fourier Transform입니다.
* $\mathcal{X}[k]$: $k$번째 Fourier component입니다.
* $k$: frequency index입니다.
* $N$: sequence length입니다.
* $j=\sqrt{-1}$: imaginary unit입니다.
* $e^{-j2\pi kn/N}$: $k$번째 complex Fourier basis입니다.

> **DFT와 FFT의 차이**
> DFT는 “어떤 수학적 변환을 하는가”를 정의하는 식이고, FFT는 같은 DFT를 빠르게 계산하는 알고리즘입니다.

Convolution theorem에 의해

```math
\mathcal{F}(h*X)[k]
=
\mathcal{F}(h)[k]\mathcal{F}(X)[k].
```

따라서

```math
\mathcal{Y}[k]
=
\mathcal{F}(h)[k]\mathcal{X}[k].
```

가 됩니다.

이것이 FreDF의 수학적 핵심입니다.

**시간영역에서 긴 convolution을 직접 학습하는 대신, Fourier domain에서는 각 frequency를 곱셈으로 변환할 수 있습니다.**

---

## 2-1-3. 주파수별 transfer function을 학습

알 수 없는 $\mathcal{F}(h)$ 대신 학습 가능한 행렬 $H_\theta$를 둡니다.

```math
\hat{\mathcal{Y}}[k]
=
H_\theta \mathcal{X}[k].
```

* $H_\theta$: 학습할 frequency-domain transfer matrix입니다.
* $\theta$: 해당 행렬을 결정하는 model parameters입니다.
* $\hat{\mathcal{Y}}[k]$: 예측된 $k$번째 output Fourier coefficient입니다.

> **Transfer function / frequency response**
> 입력의 특정 주파수가 시스템을 통과했을 때 amplitude와 phase가 어떻게 바뀌는지를 나타내는 함수입니다.

다시 inverse transform하면

```math
\hat{Y}[n]
=
\mathcal{F}^{-1}(\hat{\mathcal{Y}})
=
\frac{1}{N}
\sum_{k=1}^{K}
\hat{\mathcal{Y}}[k]
e^{j\frac{2\pi}{N}kn}.
```

그리고 MSE를 최소화합니다.

$$
\min_\theta
\frac{1}{N}
\sum_{n=0}^{N-1}
\left(Y[n]-\hat{Y}[n]\right)^2.
$$

p.4–5에서 저자들이 이 과정을 통해 **“time-domain forecasting $\rightarrow$ Fourier-domain transfer-function learning”**으로 재정식화합니다.  

---

# 2-1-4. 실제 FreDF 구조

Figure 2의 전체 흐름은

$$
X
\rightarrow
\text{Embedding}
\rightarrow
\text{FDBlock}_1
\rightarrow \cdots \rightarrow
\text{FDBlock}_L
\rightarrow
\text{Projection}
\rightarrow
\hat Y
$$

입니다. 

입력은

$$
X\in\mathbb{R}^{T\times C}
$$

입니다.

* $T$: lookback length입니다.
* $C$: 변수 개수입니다.
* $S$: prediction horizon입니다.
* $D$: latent embedding dimension입니다.
* $L$: FDBlock 개수입니다.

먼저 미래 $S$개 지점을 0으로 padding하여 길이를 $T+S$로 만듭니다.

### (a) Embedding

```math
M^1[n]
=
f(X[n]),
\qquad
f:\mathbb{R}^{C}\rightarrow\mathbb{R}^{D}.
```

중요한 특징은 **time axis를 embedding하는 것이 아니라 feature dimension $C$를 $D$로 변환한다는 것**입니다. 그래서 이후 FFT가 시간축의 구조를 그대로 다룰 수 있습니다. 

---

### (b) FDBlock: Fourier decomposition

$l$번째 block에서

```math
\mathcal{M}^{l}[k]
=
\mathcal{F}(M^l)[k].
```

실수 signal에 대해 논문의 알고리즘은

```math
K
=
\frac{T+S}{2}+1
```

개의 non-redundant frequency coefficient를 사용합니다. 공개 구현도 `torch.fft.rfft`를 사용합니다.  ([GitHub][3])

---

### (c) 주파수 하나씩 분리

$m$번째 frequency만 남긴 copy를 만듭니다.

```math
\mathcal{M}_{\text{in}}^{l,m}(k)
=
\begin{cases}
0, & k\neq m,\\[4pt]
\mathcal{M}^{l}(k), & k=m.
\end{cases}
```

즉,

$$
\text{frequency }1,
\text{ frequency }2,
\dots,
\text{ frequency }K
$$

를 독립적으로 예측할 수 있는 구조를 만드는 것입니다. 

---

### (d) frequency-specific transfer function

각 frequency에는 서로 다른 복소수 행렬

$$
H^{l,m}\in\mathbb{C}^{D\times D}
$$

를 적용합니다.

```math
\mathcal{M}^{l,m}_{\text{out}}
=
\mathcal{M}^{l,m}_{\text{in}}
H^{l,m}.
```

* $m$: frequency index입니다.
* $l$: FDBlock index입니다.
* $H^{l,m}$: $l$번째 block에서 $m$번째 주파수의 dynamics를 변환하는 행렬입니다.
* $\mathbb{C}$: complex-number space입니다.

공개 코드에서도 frequency 개수만큼 `complex64`의 $D\rightarrow D$ Linear layer를 생성합니다.  ([GitHub][3])

---

### (e) 각 frequency를 다시 시간영역으로

```math
Z^{l,m}[n]
=
\mathcal{F}^{-1}
\left(
\mathcal{M}^{l,m}_{\text{out}}
\right)[n].
```

$Z^{l,m}$은 **“frequency $m$ 하나만을 이용해 계산한 time-domain prediction component”**로 볼 수 있습니다.

---

### (f) Frequency Dynamic Fusion

이들을 다음처럼 합칩니다.

```math
\hat M^l[n]
=
\sum_{m=0}^{K}
W_m Z^{l,m}[n].
```

* $W_m$: $m$번째 frequency에 부여된 fusion weight입니다.
* 큰 $W_m$: 현재 학습된 모델이 해당 frequency prediction을 상대적으로 중요하게 사용한다는 뜻입니다.

> **Dynamic fusion**
> 이 논문에서 “dynamic”은 기본적으로 **미리 고정한 weight가 아니라 학습 가능한 weight**라는 의미입니다. 입력 sample마다 매번 새로운 $W_m$을 계산한다는 의미로 읽으면 안 됩니다.

실제로 공식 코드는

* 길이 $K$의 `self.weights`를 하나 정의하고,
* 이를 softmax 형태로 정규화한 뒤,
* 모든 sample에 동일한 vector를 적용합니다. ([GitHub][3])

따라서 더 정확한 표현은

```math
W_m
=
\text{learned global frequency importance}
```

이지,

$$
W_m=W_m(X)
$$

와 같은 **input-conditioned adaptive gating**은 아닙니다.

이 점은 이후 generalization 논의를 해석할 때 매우 중요합니다.

---

### (g) Projection

마지막으로

```math
\hat Y[n]
=
g(\hat M^L[n])[T:T+S,:],
```

$$
g:\mathbb{R}^{D}\rightarrow\mathbb{R}^{C}
$$

를 이용해 원래 변수 공간으로 projection하고 미래 $S$개 지점만 추출합니다. 

---

# 2-1-5. 일반화 이론

이 부분은 이 논문에서 가장 흥미로운 동시에 가장 주의해서 읽어야 할 부분입니다.

저자는 frequency $h$별 predictor를 $f^h$라 두고

```math
f(x)
=
\sum_{h=1}^{H}
w^h f^h(x^h)
```

로 나타냅니다. 일반화 오차는

```math
\text{GError}
=
\mathbb{E}_{(x,y)\sim\mathcal D}
\left[
l(f(x),y)
\right].
```

* $\mathcal D$: 알려지지 않은 실제 data distribution입니다.
* $l$: loss function이며 논문에서는 MSE를 사용합니다.
* $f^h$: frequency $h$ 전용 predictor입니다.
* $w^h$: 해당 frequency의 fusion weight입니다.



---

## Theorem 4.2

논문에서 주어진 generalization upper bound는 다음 형태입니다.

$$
\begin{aligned}
\text{GError}
\le&
\sum_{h=1}^{H}
\mathbb E(w^h)\hat E(f^h)
\\
&+
\sum_{h=1}^{H}
\mathbb E(w^h)
\mathfrak R_h(f^h)
\\
&+
\sum_{h=1}^{H}
\text{Cov}(w^h,l^h)
\\
&+
M
\sqrt{
\frac{\ln(1/\delta)}{2H}
}.
\end{aligned}
\tag{17}
$$



각 항은 다음 뜻입니다.

**첫 번째 항**

$$
\sum_h \mathbb E(w^h)\hat E(f^h)
$$

은 training data에서 frequency별 predictor가 얼마나 틀리는지를 나타냅니다.

**두 번째 항**

$$
\sum_h
\mathbb E(w^h)\mathfrak R_h(f^h)
$$

은 model complexity를 반영합니다.

> **Rademacher complexity $\mathfrak R$**
> 모델이 실제 구조가 없는 무작위 $\pm1$ label까지 얼마나 잘 맞출 수 있는지를 이용하는 함수복잡도 척도입니다. 대체로 값이 클수록 hypothesis class가 더 복잡해서 training data를 과도하게 맞출 가능성이 높고 일반화 bound가 느슨해집니다.

**세 번째 항**

$$
\sum_h
\text{Cov}(w^h,l^h)
$$

이 이 논문의 핵심입니다.

> **Covariance**
> 두 변수가 같이 증가하거나 감소하는 정도입니다.
> $\text{Cov}(w^h,l^h)<0$라면 loss가 큰 frequency에는 작은 weight를, loss가 작은 frequency에는 큰 weight를 주는 방향이라는 의미입니다.

즉 이상적으로는

$$
l^h\uparrow
\quad\Rightarrow\quad
w^h\downarrow
$$

가 되어야 합니다.

---

## Static fusion

static weight는 constant이므로

$$
\text{Cov}(w^h_{\text{static}},l^h)=0
$$

이고 bound에서 covariance term이 사라집니다. 

---

## Theorem 4.3

저자는 다음 두 조건이 성립하면

$$
\overline{\text{GError}}
(f_{\text{dynamic}})
\le
\overline{\text{GError}}
(f_{\text{static}})
\tag{19}
$$

라고 제시합니다.

첫 번째 조건은

```math
\mathbb E
\left[
w^h_{\text{dynamic}}
\right]
=
w^h_{\text{static}},
```

두 번째 조건은

$$
r
\left(
w^h_{\text{dynamic}},l^h
\right)
\le 0.
\tag{21}
$$

여기서 $r$은 Pearson correlation coefficient입니다. 

> **중요한 해석**
> 이 정리는 “dynamic fusion이면 항상 일반화가 좋다”는 무조건적인 정리가 아닙니다.
> 특히 $r(w^h,l^h)\le0$라는 조건이 있어야 합니다.

---

# 3. 저자 보고 결과와 검토 해석의 분리

## 연구 주제

**[저자 보고]**
고주파를 일괄적으로 noise로 취급하는 것은 부적절하며, frequency의 역할은 scenario별로 달라지기 때문에 각각 예측하고 동적으로 융합해야 한다는 것입니다. 

**[검토 해석]**
이 문제 설정은 타당합니다. 실제로 이후 2025–2026 연구에서도 “stable vs variable frequency”, “spectral shift”, “mid/high-frequency preservation”, “spectral entanglement”가 주요 문제로 등장합니다. 따라서 **논문의 문제 제기 자체는 후속 연구 흐름에 의해 상당히 강화되었습니다.** ([AAAI Publications][5]) ([AAAI Publications][6]) ([AAAI Publications][7])

---

## 방법

**[저자 보고]**
Fourier-domain transfer function과 dynamic fusion을 결합하고 Rademacher bound로 일반화 능력을 설명합니다.

**[검토 해석]**
frequency-specific transfer matrix는 frequency마다 서로 다른 dynamics를 허용한다는 점에서 의미가 있습니다. 그러나 실제 fusion weight는 $w_h(x)$가 아니라 global $w_h$이므로 **새로운 operating condition이 입력되었을 때 frequency importance가 즉시 바뀌는 구조는 아닙니다.** 공식 구현에서는 하나의 parameter vector를 모든 batch/sample에 사용합니다. ([GitHub][3])

---

## 결과

**[저자 보고]**
저자는 Table 1을 근거로 FreDF가 “70 out of 80 benchmarks”에서 최적이며, FEDformer 대비 MSE/MAE 평균 약 13% 개선, Exchange-rate에서는 최대 33% 개선이라고 보고합니다. 

**[검토 해석]**
표의 dataset별 Avg 행을 제가 단순 평균하면 FreDF와 FEDformer는 대략

$$
\text{MSE}: 0.3095 \text{ vs. } 0.3704,
$$

$$
\text{MAE}: 0.3430 \text{ vs. } 0.4009
$$

가 되어, 각각 약 **16.4%와 14.4%**의 상대 감소입니다. 저자의 “13%”는 다른 aggregation 방식일 가능성이 있지만, 본문만으로 그 계산법이 명확하게 재현되지는 않습니다. 따라서 여기서는 **저자의 13%를 공식 보고값**, 위 16.4/14.4%는 **Table 1 Avg를 이용한 별도 재계산값**으로 구분해야 합니다. 

---

# 4. 성능 향상은 실제로 어느 정도인가?

대표적인 평균값을 보면:

| Dataset      |   FreDF MSE / MAE |  iTransformer | 검토                       |
| ------------ | ----------------: | ------------: | ------------------------ |
| ETTm1 Avg    | **0.384 / 0.398** | 0.407 / 0.410 | FreDF 우세                 |
| ETTm2 Avg    | **0.281 / 0.323** | 0.288 / 0.332 | MSE는 PatchTST와 사실상 동률 수준 |
| ETTh1 Avg    | **0.435 / 0.431** | 0.454 / 0.447 | FreDF 우세                 |
| ETTh2 Avg    | **0.376 / 0.399** | 0.383 / 0.407 | 소폭 우세                    |
| Exchange Avg | **0.351 / 0.396** | 0.360 / 0.403 | 소폭 우세                    |
| Weather Avg  | **0.241 / 0.270** | 0.258 / 0.279 | 우세                       |
| ECL Avg      | **0.176 / 0.268** | 0.178 / 0.270 | 매우 작은 차이                 |
| Solar Avg    | **0.232 / 0.259** | 0.233 / 0.262 | 매우 작은 차이                 |

수치는 Table 1입니다. 

여기서 **ECL 0.176 vs 0.178**, Solar MSE **0.232 vs 0.233** 같은 차이를 단순히 “우월하다”고 표현하기는 어렵습니다. variance가 제시되지 않았기 때문입니다.

---

# 5. 통계적으로 취약한 부분 및 비교 불가능한 수치

이 부분은 논문 평가에서 특히 중요합니다.

### ① 세 번 반복했지만 분산을 보고하지 않음 — `[통계 취약]`

arXiv Appendix B를 확인하면 실험은 **3회 반복하고 mean을 사용**했다고 명시되어 있습니다. Adam, learning rate ${10^{-3},10^{-4}}$, batch size 4, 최대 10 epochs 등의 조건도 기록되어 있습니다. 그러나 standard deviation, confidence interval 또는 statistical significance test는 보고하지 않습니다. ([arXiv][4])

따라서

$$
0.176 \quad\text{vs.}\quad 0.178
$$

정도의 차이가 run-to-run variation보다 실제로 큰지는 확인할 수 없습니다.

---

### ② Table 1의 “70 out of 80”은 표시된 표만으로 재구성되지 않음 — `[내부 불일치]`

첨부 PDF의 Table 1에는 8개 dataset,

$$
8\times4=32
$$

개의 dataset-horizon 조합이 보입니다. MSE와 MAE를 각각 세면 총

$$
32\times2=64
$$

metric entries입니다.

그런데 표의 `1st Count`는 FreDF에서 **34 MSE + 36 MAE = 70**, 전체 model의 first counts도 각 metric별 40개로 합산됩니다. 즉 **표시된 8 datasets만으로는 이 count를 만들 수 없습니다.**

따라서 “70/80”은 아마 40개 task를 전제로 한 계산으로 보이지만, 현재 공개된 표와 일관되게 재구성할 수 없습니다. 이는 성능 자체가 틀렸다는 뜻은 아니지만 **reporting consistency 문제**입니다. 

---

### ③ Table 3 Weather-336 MAE가 Table 1과 충돌 — `[내부 불일치]`

Table 1:

$$
\text{Weather-336 FreDF MAE}=0.287.
$$



그런데 Table 3에는 동일한 Weather-336 FreDF의 MAE가

$$
0.587
$$

로 표시되어 있습니다. 

PDF page image도 확인했으며 실제 표에 **0.587**로 인쇄되어 있습니다. 따라서 OCR 문제가 아니라 **논문 자체의 수치 불일치 또는 typo일 가능성이 높습니다.**

---

### ④ Transfer function ablation도 모든 경우 개선되는 것은 아님

Table 2에서 대부분 transfer function이 좋지만 Exchange-rate, horizon 336에서는:

$$
\text{with transfer}: 0.316/0.405,
$$

$$
\text{without transfer}: 0.254/0.312.
$$

즉 이 경우에는 transfer function 제거가 더 좋습니다. 따라서 “transfer function이 언제나 중요한 개선 요인”이라는 설명보다 **대부분의 조건에서 효과가 있었다**고 표현하는 것이 정확합니다. 

---

### ⑤ Parameter comparison은 제한된 baseline에 대해서만 유효 — `[비교 주의]`

Table 4는

$$
\text{FreDF}=151.4K,
$$

$$
\text{iTransformer}=3.1M,
\quad
\text{PatchTST}=3.5M,
$$

$$
\text{FEDformer}=14.0M,
\quad
\text{FiLM}=12.0M
$$

을 보고합니다. 

따라서 대략 iTransformer보다 $20.5\times$, PatchTST보다 $23.1\times$, FEDformer보다 $92.5\times$, FiLM보다 $79.3\times$ 적은 parameter입니다.

그러나 같은 시기의 FITS는 ICLR 2024에서 대개 **5K–10K 수준**을 보고합니다. 따라서 “FreDF가 lightweight하다”는 결론은 가능하지만 **“frequency forecasting 모델 중 가장 작은 모델”이라는 결론은 불가능**합니다. ([OpenReview][9])

---

### ⑥ ModernTCN 수치는 저자 스스로 직접 비교 불가능하다고 명시

arXiv Appendix J에서 저자들은 ModernTCN은 lookback window가 96보다 길고, FreDF는 모든 main experiment에서 96을 사용하기 때문에 **direct comparison is not feasible**하다고 명시합니다. ([arXiv][4])

따라서 서로 다른 논문의 headline MSE를 그대로 나열하여 ranking하는 방식은 적절하지 않습니다.

---

### ⑦ 다른 논문의 “38%, 22.6%, 19.2% 개선”과 FreDF의 “13%”는 직접 비교 금지

Autoformer는 6 benchmarks에서 38% improvement, FEDformer는 multivariate/univariate 14.8%/22.6%, FiLM은 19.2%/22.6%를 각각 자신의 baseline 정의하에서 보고했습니다. 이는 **dataset, metric aggregation, baseline set, forecast setting이 서로 다르므로 숫자 크기 자체로 어느 논문이 더 우수한지를 판단하면 안 됩니다.** ([NeurIPS Proceedings][10]) ([Proceedings of Machine Learning Research][11]) ([NeurIPS Proceedings][12])

---

# 6. 일반화 이론에서 특히 주의할 부분

## 6-1. 가장 중요한 이론–구현 간극

Theorem 4.2의 핵심은

$$
\text{Cov}(w^h,l^h)
$$

입니다.

그런데 공개 구현은 하나의 전역 `self.weights` vector를 학습하고 모든 sample에 동일하게 적용합니다. ([GitHub][3])

훈련 완료 후 고정된 $w^h$를 data distribution $\mathcal D$에 대해 보면

$$
w^h=\text{constant}
$$

이므로 원칙적으로

```math
\text{Cov}_{(x,y)\sim\mathcal D}
(w^h,l^h(x,y))
=
0
```

입니다.

즉 이론에서 강조하는

$$
\text{Cov}(w^h,l^h)<0
$$

를 실제 공개 구현이 어떻게 만들어내는지는 명확하지 않습니다.

이것은 **FreDF의 실험 성능이 틀렸다는 뜻이 아닙니다.** 하지만

> “negative weight-loss covariance가 일반화 향상의 원인이다”

라는 이론적 설명은 공개 implementation과 바로 연결되지 않습니다.

진정한 instance-conditioned dynamic fusion이라면 예를 들어

```math
w^h(x)
=
\frac{
\exp(a_h(x))
}{
\sum_{j=1}^{H}\exp(a_j(x))
}
```

와 같이 입력 $x$에 따라 weight가 달라져야 합니다.

---

## 6-2. 정리의 조건이 실험으로 검증되지 않음

Theorem 4.3은

$$
r(w^h_{\text{dynamic}},l^h)\le0
$$

를 요구합니다.

그러나 본 논문의 main experiments에는 실제 각 frequency에서

```math
r_h
=
r(w^h,l^h)
```

를 측정한 table이나 plot이 없습니다.

따라서 **“정리가 요구하는 sufficient condition이 실제 학습된 FreDF에서 만족되었다”는 empirical verification은 없습니다.**

---

## 6-3. LTI assumption

실제 기상·전력·환율 시계열은 regime change와 non-stationarity를 포함합니다.

FreDF의 Fourier derivation은

```math
Y[k]
=
H[k]X[k]
```

라는 LTI 관계에서 가장 자연스럽습니다.

하지만 실제 시스템에서는 더 일반적으로

$$
H=H(k,t,x,\text{regime})
$$

일 수 있습니다.

즉 frequency response 자체가 시간과 operating condition에 따라 달라질 수 있습니다.

이것이 2025–2026 연구들이 **time-varying spectrum**, **spectral shift**, **dynamic filters** 쪽으로 발전한 핵심 이유 중 하나입니다. ([AAAI Publications][5]) ([AAAI Publications][8])

---

## 6-4. Rademacher bound의 concentration term 해석

논문의 식에는

$$
M\sqrt{
\frac{\ln(1/\delta)}{2H}
}
$$

가 들어갑니다. 여기서 $H$는 frequency component 수로 정의됩니다. 

일반적인 statistical learning generalization bound에서는 concentration rate를 결정하는 항에 **independent sample count $n$**이 나타나는 것이 자연스럽습니다. Fourier bins는 동일한 time series에서 만들어지며 서로 완전히 독립된 observation이라고 간주하기 어렵습니다.

따라서 이 논문에서는

> 왜 training sample 수가 아니라 frequency count $H$가 이 concentration term을 결정하는가?

에 대한 확률론적 정당화가 더 필요합니다.

이를 곧바로 “틀린 식”이라고 단정할 근거까지는 없지만, **bound의 statistical interpretation에서 가장 재검토가 필요한 지점 중 하나**입니다.

---

## 6-5. 같은 hypothesis class라고 같은 empirical risk가 되는가?

Appendix A proof에서는 static/dynamic frequency predictor가 같은 function class를 사용한다는 이유로 empirical risk가 동일하다는 관계를 사용합니다. ([arXiv][4])

하지만 실제 neural-network optimization에서는

```math
\mathcal F_{\text{static}}
=
\mathcal F_{\text{dynamic}}
```

혹은 동일한 component architecture라는 사실만으로

```math
\hat E(f^h_{\text{static}})
=
\hat E(f^h_{\text{dynamic}})
```

가 자동으로 성립하지 않습니다.

공동 optimization, local minima, fusion과 component predictor 사이의 gradient coupling 등에 의해 달라질 수 있습니다.

따라서 이 부분도 **이상적 최적화 조건하의 분석**으로 보는 편이 안전합니다.

---

## 6-6. Rademacher complexity와 parameter count를 동일시해서는 안 됨

논문은 Eq. (22)의 complexity 비교와 Table 4의 작은 parameter 수를 연결합니다. 

하지만

$$
\mathfrak R(\mathcal F)
$$

와 단순 parameter count는 같은 개념이 아닙니다.

parameter 수가 작아도 norm이 매우 크거나 hypothesis class가 유연하면 complexity가 클 수 있고, parameter가 많아도 strong regularization에 의해 effective complexity가 낮을 수 있습니다.

따라서

$$
\text{fewer parameters}
\not\Rightarrow
\text{automatically tighter Rademacher bound}.
$$

Table 4는 **효율성의 empirical evidence**이지, Eq. (22)의 직접적인 수학적 증명은 아닙니다.

---

# 7. 가장 중요한 그림 5개 해석

첨부한 10-page PDF에는 번호가 붙은 Figure가 **Figure 1–2 두 개**이고, arXiv full version Appendix D에 **Figure 3**이 추가됩니다. 따라서 존재하지 않는 Figure 4–5를 만들어내지 않고, 요청한 “5개”를 충족하기 위해 **Figure 1의 핵심 sub-panel을 각각 독립적인 시각적 증거로 해석**하겠습니다.

## ① Figure 1(a) — ETTm1, p.3

high-frequency를 제거한 prediction이 all-frequency보다 ground truth에 가까워집니다. 반면 low/mid frequency를 제거하면 악화됩니다. 

**저자 해석:** ETTm1에서는 high frequency가 noise처럼 작용할 수 있습니다.

**제 해석:** “high frequency 제거가 항상 잘못”이라는 이야기가 아니라, 오히려 **low-pass assumption이 일부 dataset에서는 유효함**을 보여줍니다. FreDF의 논리는 “high-frequency를 반드시 보존한다”가 아니라 “삭제 여부를 고정하지 말자”입니다.

---

## ② Figure 1(b) — ETTm2, p.3

ETTm2에서는 low-frequency 제거가 유리하다고 보고합니다. 

이는 직관적으로 상당히 중요합니다. trend·저주파가 항상 signal이라고 생각하기 쉽지만, 현재 forecasting window에서는 오래된 trend가 오히려 미래를 잘못 끌고 갈 수 있음을 암시합니다.

즉,

$$
\text{low frequency}
\neq
\text{always useful trend}.
$$

---

## ③ Figure 1(c) — ETTh1, p.3

ETTm2와 반대로 ETTh1에서는 low-frequency를 제거했을 때 성능이 나빠집니다. 

Figure 1(b)와 1(c)를 나란히 보는 것이 이 논문의 가장 강한 motivation입니다.

$$
\boxed{
\text{동일한 frequency band에 대한 최적 정책도 dataset에 따라 반대가 될 수 있다.}
}
$$

다만 이 figure는 대표 forecast trajectory이지 statistical aggregate가 아니므로 이 현상을 정량적으로 입증하는 데는 Appendix G의 전체 band-ablation table이 더 중요합니다. arXiv Appendix G에서도 band 제거의 효과가 dataset별로 다르다고 보고합니다. ([arXiv][4])

---

## ④ Figure 2 — FreDF architecture, p.4

Figure 2는 논문의 핵심 구조를 가장 잘 보여줍니다. 

핵심 흐름은

$$
M^l
\xrightarrow{\text{FFT}}
\{
\mathcal M^{l,1},
\dots,
\mathcal M^{l,K}
\}
$$

$$
\xrightarrow{H^{l,m}}
\{
\mathcal M_{\text{out}}^{l,1},
\dots,
\mathcal M_{\text{out}}^{l,K}
\}
$$

$$
\xrightarrow{\text{iFFT}}
\{
Z^{l,1},
\dots,
Z^{l,K}
\}
$$

$$
\xrightarrow{\text{weighted fusion}}
\hat M^l
$$

입니다.

일반적인 spectral network가 전체 spectrum을 하나의 tensor로 처리하는 것과 달리, FreDF는 **“각 Fourier bin을 하나의 작은 forecasting expert처럼 취급한다”**고 이해하면 가장 쉽습니다.

---

## ⑤ Figure 3 — FreSF vs FreDF, arXiv Appendix D

Appendix D의 Figure 3은 ETTh1, ETTh2, ETTm1, ETTm2, Weather, Exchange-rate에서 static fusion과 learned fusion의 forecast curve를 비교합니다. 저자들은 dynamic fusion이 trend, extreme point, ground-truth alignment에서 개선된다고 설명합니다. ([arXiv][4])

그러나 여기서도 “dynamic”을 정확히 해석해야 합니다.

Figure 3이 증명하는 것은

$$
\text{learned global frequency weights} > \text{predefined fixed weights}
$$

라는 empirical observation에 가깝습니다.

그것이 곧

$$
w_h=w_h(x_t)
$$

와 같은 실시간 adaptive frequency selector의 유효성을 증명하는 것은 아닙니다.

---

# 8. 이 논문이 답하지 않는 질문

1. **새로운 sample이나 regime이 들어왔을 때 frequency weights가 실제로 바뀌어야 하지 않는가?** 현재 공개 구현에서는 바뀌지 않습니다.

2. **Theorem 4.3의 핵심 조건**

$$r(w^h,l^h)\le0$$

   **가 실제 실험에서 만족되는가?** 논문은 이를 측정하지 않습니다.

3. frequency importance가 dataset 단위가 아니라 **timestamp, window, variable, regime 단위**로 얼마나 달라지는가?

4. low/mid/high를 단순히 spectrum의 $1/3$씩 자르는 것이 물리적으로 의미 있는 분할인가?

5. **Spectral leakage** 때문에 한 실제 periodic component가 여러 Fourier bin으로 퍼지는 문제는 어떻게 처리할 것인가?

> **Spectral leakage**
> 관찰 window와 실제 signal 주기가 정확히 맞지 않을 때 하나의 주파수 에너지가 여러 Fourier bin으로 퍼지는 현상입니다.

6. Fourier amplitude만 중요한가, 아니면 phase 변화가 더 중요한 상황도 있는가?

7. irregularly sampled time series에서는 FFT 기반 구조를 어떻게 확장할 것인가?

8. 예측 uncertainty는 어떻게 정량화할 것인가?

9. train/test 사이에서 spectrum이 크게 달라지는 **domain shift/OOD setting**에서도 FreDF가 우세한가?

10. $K$개 frequency 각각에 $D\times D$ complex matrix를 둘 때 아주 긴 lookback에서는 compute·memory가 어떻게 증가하는가?

공개 구현을 기준으로 transfer-function parameter는 대략

$$
O(LKD^2)
$$

로 증가하므로, Table 4의 151.4K라는 절대 숫자가 **long context에서도 그대로 유지된다고 볼 수는 없습니다.** ([GitHub][3])

---

# 9. 2020년 이후 관련 최신 연구 비교

| 연구                                      |   연도 | 핵심 frequency 관점                                      | FreDF와의 관계                                      | 일반화 관점                                       |
| --------------------------------------- | ---: | ---------------------------------------------------- | ----------------------------------------------- | -------------------------------------------- |
| **Adaptive Temporal-Frequency Network** | 2020 | phase/frequency/amplitude adaptation                 | FreDF 이전부터 adaptive spectrum 아이디어 존재            | 시간에 따라 달라지는 periodicity 모델링                  |
| **From Fourier to Koopman**             | 2021 | Fourier를 nonlinear Koopman basis로 확장                 | 고정 LTI보다 더 일반적 dynamics                         | nonlinear dynamics·irregular sampling까지 고려   |
| **Autoformer**                          | 2021 | periodic sub-series + decomposition                  | explicit Fourier fusion은 아님                     | trend/seasonality 분리로 long horizon 안정화       |
| **FEDformer**                           | 2022 | sparse Fourier basis + decomposition                 | FreDF 주요 baseline                               | global frequency structure 활용                |
| **FiLM**                                | 2022 | Fourier projection으로 noise 제거                        | FreDF가 문제 삼는 low-pass 계열의 대표 예                  | noise suppression 중심                         |
| **FreTS**                               | 2023 | frequency-domain MLP                                 | spectrum을 직접 학습                                 | global dependency와 energy compaction         |
| **FITS**                                | 2024 | low-pass + complex interpolation, 5–10K params       | FreDF보다 훨씬 경량                                   | simple spectral inductive bias               |
| **FreDF (본 논문)**                        | 2024 | frequency-wise transfer + learned fusion             | 중심 논문                                           | conditional generalization-bound 제시          |
| **FilterTS**                            | 2025 | dynamic cross-variable filter + static global filter | FreDF의 “frequency 역할은 상황별로 다름”을 더 구체화           | stable/variable component 분리                 |
| **FBM**                                 | 2025 | explicit Fourier basis/time-frequency mapping        | Fourier bin의 시간정보 손실 문제 보완                      | starting-cycle/series-length 변화 대응           |
| **FreDN**                               | 2026 | learnable spectral disentanglement                   | FreDF보다 spectral leakage/nonstationarity를 직접 다룸 | spectrum entanglement 완화                     |
| **DTAF**                                | 2026 | temporal stabilization + frequency differencing      | frequency 자체의 shift를 직접 모델링                     | non-stationary/OOD 관점 강화                     |
| **FreqCycle / MFreqCycle**              | 2026 | low뿐 아니라 mid/high frequency를 adaptive weighting      | FreDF 핵심 주장과 매우 직접적으로 연결                        | multi-scale periodic regime 적응               |
| **Sonnet**                              | 2026 | learnable wavelet + Koopman + spectral coherence     | fixed Fourier basis보다 광범위                       | multivariable dependency와 nonlinear dynamics |

관련 근거는 각 연구의 공식/1차 자료입니다. Autoformer는 progressive decomposition과 Auto-Correlation을 제안했고, FEDformer는 sparse Fourier basis를 Transformer와 결합했으며, FiLM은 Fourier projection으로 noise를 줄이는 방향입니다. ([NeurIPS Proceedings][10]) ([Proceedings of Machine Learning Research][11]) ([NeurIPS Proceedings][12])

FreTS는 frequency-domain MLP가 global view와 energy compaction을 제공한다고 보고했고, FITS는 low-pass 기반의 매우 작은 complex-valued architecture를 제안했습니다. ([NeurIPS Proceedings][13]) ([OpenReview][9])

### 2025–2026년 연구에서 나타난 중요한 변화

**FilterTS (AAAI 2025)**는 frequency를 **stable component와 variable component**로 구분하고, variable component에는 cross-variable dynamic filter를 적용합니다. 즉 FreDF의 “모든 frequency가 같지 않다”를 한 단계 더 발전시켜 **어떤 성분은 global/static, 어떤 성분은 dynamic**이라는 구조를 명시합니다. ([AAAI Publications][5])

**FreDN (AAAI 2026)**은 non-stationarity와 spectral leakage 때문에 trend·periodicity·noise가 spectrum에서 서로 섞이는 **spectral entanglement** 문제를 정면으로 다룹니다. learnable Frequency Disentangler를 사용하며, 저자들은 7개 장기예측 benchmark에서 최대 10% 개선과 complex architecture 대비 최소 50%의 parameter/compute 절감을 보고합니다. ([AAAI Publications][6])

**DTAF (AAAI 2026)**는 temporal distribution shift뿐 아니라 **spectral variability 자체**를 모델링하고, frequency differencing을 통해 spectrum이 크게 변한 성분을 강조합니다. 이는 FreDF의 global $w_h$보다 OOD/generalization 관점에서 더 직접적인 구조입니다. ([AAAI Publications][8])

**FreqCycle (AAAI 2026)**은 특히 의미가 큽니다. 저자들은 기존 연구가 low-frequency에 집중하면서 **mid-to-high frequency를 놓치는 것이 성능 향상을 제한한다**고 명시하고, 해당 대역을 adaptive weighting합니다. 이는 FreDF가 2024년에 제기한 핵심 문제와 거의 같은 방향의 문제의식을 더욱 구체적으로 발전시킨 것입니다. ([AAAI Publications][7])

**Sonnet (AAAI 2026)**은 learnable wavelet과 Koopman operator, spectral coherence를 이용해 multivariable 관계를 모델링하며 47 forecasting tasks 중 34개에서 최고 성능, 가장 경쟁력 있는 baseline 대비 평균 MAE 2.2% 감소를 보고합니다. 이는 단일 Fourier basis뿐 아니라 **다중 해상도 + nonlinear dynamical basis + cross-variable spectral relation**으로 연구가 확장되고 있음을 보여줍니다. ([AAAI Publications][14])

---

# 10. 그렇다면 FreDF가 이후 연구에 미친 의미는 무엇인가?

제가 보는 핵심 영향은 **“frequency selection 문제의 재정의”**입니다.

과거 접근을 단순화하면

$$
\text{high frequency}
\rightarrow
\text{noise}
\rightarrow
\text{remove}
$$

에 가까웠습니다.

FreDF는 이를

$$
\text{frequency}
\rightarrow
\text{independent prediction}
\rightarrow
\text{learn its importance}
\rightarrow
\text{fusion}
$$

으로 바꿨습니다.

그리고 2025–2026년 연구는 다시

```math
\text{frequency importance}
=
f(
\text{time},
\text{sample},
\text{variable},
\text{regime},
\text{scale}
)
```

방향으로 발전하고 있습니다.

즉 FreDF의 장기적인 의미는 특정 architecture 자체보다도

> **frequency를 noise filtering의 대상이 아니라 “상황에 따라 예측력이 달라지는 latent predictive components”로 보는 관점**

에 있다고 판단합니다.

---

# 11. 일반화 성능 향상 가능성 — 가장 중요한 후속 연구 방향

## 11-1. Global weight를 truly dynamic weight로 확장

현재는

$$
w_h=\text{global parameter}
$$

입니다.

이를

$$
w_h(x_t,c_t,r_t)
$$

로 확장하는 것이 가장 직접적입니다.

예를 들어

```math
a_h
=
q_\phi
\left(
X_{t-T+1:t},
\mathcal X_h,
c_t
\right),
```

```math
w_h
=
\frac{e^{a_h}}
{\sum_j e^{a_j}}.
```

여기서

* $X_{t-T+1:t}$: 현재 input window,
* $\mathcal X_h$: 현재 frequency $h$의 spectrum,
* $c_t$: 현재 operating context,
* $q_\phi$: gating network

입니다.

이렇게 하면 특정 regime에서

$$
w_{20}(x_A)\gg w_{20}(x_B)
$$

처럼 실제로 frequency importance가 sample마다 달라질 수 있습니다.

이것이 FreDF가 주장한 **“scenario-dependent frequency importance”를 architecture 수준에서 가장 직접적으로 구현하는 방법**입니다.

---

## 11-2. 이론도 input-conditioned weight에 맞춰 재작성

새로운 모델에서는

$$
w_h=w_h(X)
$$

이므로 covariance가 실제 의미를 갖습니다.

훈련 목표 자체에

$$
\text{Cov}(w_h(X),l_h(X))
$$

를 제어하는 term을 넣는 방법도 연구할 수 있습니다.

예를 들어 이상적인 방향은

$$
\min_\theta
\mathcal L_{\text{forecast}}
+
\lambda
\sum_h
\max
\left(
0,
\text{Cov}(w_h,l_h)
\right).
$$

즉 **오차가 큰 frequency를 많이 신뢰하는 경우를 벌점**으로 주는 것입니다.

다만 실제 loss를 gating input으로 직접 사용하면 target leakage가 생길 수 있으므로, inference 시 사용할 수 있는 uncertainty proxy나 train-only auxiliary estimator가 필요합니다.

---

## 11-3. 일반화 bound 자체를 시계열에 맞게 확장

독립 표본을 전제로 한 일반적인 Rademacher reasoning보다 시계열에서는

* temporal dependence,
* autocorrelation,
* distribution shift,
* mixing process

를 직접 고려해야 합니다.

따라서 후속 이론에서는 ordinary Rademacher complexity뿐 아니라 **sequential Rademacher complexity, $\beta$-mixing bound, martingale concentration** 등 시간 의존성을 반영하는 framework가 더 적절합니다.

> **$\beta$-mixing**
> 멀리 떨어진 두 시간 구간이 얼마나 독립에 가까워지는지를 측정하는 확률적 의존성 개념입니다.

이렇게 해야

$$
\text{Train}\rightarrow\text{Validation}\rightarrow\text{future Test}
$$

에서 실제로 관심 있는 **temporal generalization**을 bound할 수 있습니다.

---

## 11-4. FFT 하나만 사용하지 말고 local time-frequency representation으로 확장

global FFT는

$$
\mathcal X[k]
$$

를 구하지만 “그 frequency가 **언제** 발생했는가”를 직접 표현하지 못합니다.

따라서

$$
\mathcal X[k,t]
$$

를 얻는 STFT, wavelet, learnable filter bank 등이 유리할 수 있습니다.

2026년 Sonnet과 FreqCycle의 방향도 이와 연결됩니다. ([AAAI Publications][14]) ([AAAI Publications][7])

---

## 11-5. Spectral shift를 명시적인 OOD signal로 활용

training spectrum을

$$
P_{\text{train}}(f)
$$

현재 window spectrum을

$$
P_t(f)
$$

라 하고,

```math
D_t
=
D
\left(
P_t(f),
P_{\text{train}}(f)
\right)
```

를 spectral domain-shift score로 사용할 수 있습니다.

$D_t$가 커지면

* frequency gating을 재조정하거나,
* robust predictor로 전환하거나,
* uncertainty를 높이거나,
* online adaptation을 시작

할 수 있습니다.

이 방향은 2026 DTAF가 다루는 spectral variability와 매우 잘 연결됩니다. ([AAAI Publications][8])

---

## 11-6. Frequency마다 독립 $D\times D$ matrix를 두는 대신 partial sharing

현재는 대략

$$
H^{l,1},H^{l,2},\dots,H^{l,K}
$$

가 별도로 존재합니다.

이를

```math
H^{l,m}
=
\sum_{r=1}^{R}
\alpha_{m,r}B^{l,r},
\qquad
R\ll K
```

로 만들면 됩니다.

* $B^{l,r}$: 공유하는 basis transfer matrix,
* $\alpha_{m,r}$: frequency $m$의 mixture coefficients입니다.

그러면 parameter complexity가 대략

$$
O(LKD^2)
$$

에서

$$
O(LRD^2+LKR)
$$

방향으로 줄어듭니다.

소표본과 domain shift에서 frequency별 matrix의 overfitting을 줄이는 장점도 기대할 수 있습니다.

---

# 12. 논문의 결론과 저자의 후속 연구 계획

**저자 결론**은 세 가지입니다.

첫째, frequency의 역할은 scenario마다 다릅니다.
둘째, Fourier-domain transfer function + frequency-wise prediction + dynamic fusion이 효과적입니다.
셋째, 저자들이 제안한 generalization-bound 관점에서 dynamic fusion이 더 유리할 수 있으며 실험도 이를 뒷받침한다고 주장합니다. 

중요하게도 **첨부 논문의 Conclusion에는 구체적인 “향후에는 X를 연구하겠다”는 명시적 future-work plan이 없습니다.** 따라서 저자의 향후 계획을 제가 임의로 만들어서는 안 됩니다. arXiv version은 추가 ablation과 dataset/implementation detail을 제공하지만, 그것 역시 명확한 future-work roadmap을 제시하는 형태는 아닙니다. ([arXiv][4])

---

# 13. 최종 연구 평가

이 논문에서 가장 강한 부분은 **“고주파를 버릴지 말지”라는 이분법을 버리고 frequency importance 자체를 학습 대상으로 전환한 것**입니다. 이 문제의식은 2026년 FreqCycle이 “mid-to-high frequency를 무시하는 것이 추가 성능 향상을 막는다”고 다시 지적하고, FreDN이 spectral entanglement를, DTAF가 spectral variability를 명시적으로 다루면서 더욱 설득력을 얻었습니다. ([AAAI Publications][7]) ([AAAI Publications][6]) ([AAAI Publications][8])

반면 **“FreDF가 이론적으로 더 좋은 일반화 성능을 보장한다”는 문장은 훨씬 제한적으로 해석해야 합니다.** Theorem 4.3은 명시적인 조건부 결과이고, 핵심인 weight-loss negative correlation이 실험으로 검증되지 않았으며, 공개 implementation의 frequency weight는 input-conditioned random variable가 아니라 전역 parameter입니다. 따라서 현재 증거가 가장 강하게 지지하는 결론은

$$
\boxed{
\text{frequency-wise modeling + learned fusion이
여러 benchmark에서 효과적이다}
}
$$

이지,

$$
\boxed{
\text{모든 unseen domain에서
FreDF의 generalization superiority가 증명되었다}
}
$$

까지는 아닙니다.

향후 가장 가치 있는 연구는 **FreDF의 global frequency weights를 input/time/regime-conditioned weights로 발전시키고, spectral shift를 명시적으로 측정하며, non-stationary/OOD split에서 이를 검증하고, 시간 의존성을 반영한 generalization theory를 다시 세우는 것**이라고 판단합니다. 그렇게 되면 FreDF의 최초 문제 제기였던 “Not All Frequencies Are Created Equal”을 더 정확히

$$
\boxed{
\text{Not All Frequencies Are Equally Useful,
and Their Usefulness Changes with Context}
}
$$

라는 형태로 발전시킬 수 있습니다.

---

# 참고한 자료 및 사이트 — 전체 제목

1. **Zhang et al. (2024), “Not All Frequencies Are Created Equal: Towards a Dynamic Fusion of Frequencies in Time-Series Forecasting”**, ACM Multimedia 2024 / 첨부 PDF. 
2. **arXiv: “Not All Frequencies Are Created Equal: Towards a Dynamic Fusion of Frequencies in Time-Series Forecasting”**, v3 및 Appendix A–J. ([arXiv][15]) ([arXiv][4])
3. **Zh-XY22/FreDF — official GitHub implementation**, 특히 `models/FreDF.py`. ([GitHub][1]) ([GitHub][3])
4. **“Adaptive Temporal-Frequency Network for Time-Series Forecasting”**, IEEE TKDE, 2020. ([ResearchGate][16])
5. **“From Fourier to Koopman: Spectral Methods for Long-term Time Series Prediction”**, JMLR, 2021. ([JMLR][17])
6. **“Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting”**, NeurIPS 2021. ([NeurIPS Proceedings][10])
7. **“FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting”**, ICML 2022. ([Proceedings of Machine Learning Research][11])
8. **“FiLM: Frequency improved Legendre Memory Model for Long-term Time Series Forecasting”**, NeurIPS 2022. ([NeurIPS Proceedings][12])
9. **“Frequency-domain MLPs are More Effective Learners in Time Series Forecasting (FreTS)”**, NeurIPS 2023. ([NeurIPS Proceedings][13])
10. **“iTransformer: Inverted Transformers Are Effective for Time Series Forecasting”**, ICLR 2024. ([OpenReview][18])
11. **“FITS: Modeling Time Series with $10k$ Parameters”**, ICLR 2024 Spotlight. ([OpenReview][9])
12. **“FilterTS: Comprehensive Frequency Filtering for Multivariate Time Series Forecasting”**, AAAI 2025. ([AAAI Publications][5])
13. **“Fourier Basis Mapping: A Time-Frequency Learning Framework for Time Series Forecasting”**, 2025 arXiv preprint. ([arXiv][19])
14. **“A Unified Frequency Domain Decomposition Framework for Interpretable and Robust Time Series Forecasting (FIRE)”**, 2025 arXiv preprint. ([arXiv][20])
15. **“FreDN: Spectral Disentanglement for Time Series Forecasting via Learnable Frequency Decomposition”**, AAAI 2026. ([AAAI Publications][6])
16. **“Towards Non-Stationary Time Series Forecasting with Temporal Stabilization and Frequency Differencing”**, AAAI 2026. ([AAAI Publications][8])
17. **“FreqCycle: A Multi-Scale Time-Frequency Analysis Method for Time Series Forecasting”**, AAAI 2026. ([AAAI Publications][7])
18. **“Sonnet: Spectral Operator Neural Network for Multivariable Time Series Forecasting”**, AAAI 2026. ([AAAI Publications][14])
19. **“M2FMoE: Multi-Resolution Multi-View Frequency Mixture-of-Experts for Extreme-Adaptive Time Series Forecasting”**, AAAI 2026. ([AAAI Publications][21])

특히 **일반화 성능만을 기준으로 후속 연구를 설계한다면**, 다음 핵심 축은 `FreDF → input-conditioned frequency gating → spectral-shift-aware domain generalization → sequential generalization bound` 순으로 잡는 것이 이 논문의 이론을 가장 자연스럽게 발전시키는 방향입니다.

[1]: https://github.com/Zh-XY22/FreDF "GitHub - Zh-XY22/FreDF · GitHub"
[2]: https://github.com/Master-PLC/FreDF?utm_source=chatgpt.com "GitHub - Master-PLC/FreDF: Repository for \"FreDF: Learning to Forecast in the Transformed Domain\" · GitHub"
[3]: https://github.com/Zh-XY22/FreDF/blob/main/models/FreDF.py "FreDF/models/FreDF.py at main · Zh-XY22/FreDF · GitHub"
[4]: https://arxiv.org/abs/2407.12415 "Not All Frequencies Are Created Equal:Towards a Dynamic Fusion of Frequencies in Time-Series Forecasting"
[5]: https://ojs.aaai.org/index.php/AAAI/article/view/35438?utm_source=chatgpt.com "FilterTS: Comprehensive Frequency Filtering for Multivariate Time Series Forecasting | Proceedings of the AAAI Conference on Artificial Intelligence"
[6]: https://ojs.aaai.org/index.php/AAAI/article/view/39042?utm_source=chatgpt.com "FreDN: Spectral Disentanglement for Time Series Forecasting via Learnable Frequency Decomposition | Proceedings of the AAAI Conference on Artificial Intelligence"
[7]: https://ojs.aaai.org/index.php/AAAI/article/view/40042?utm_source=chatgpt.com "FreqCycle: A Multi-Scale Time-Frequency Analysis Method for Time Series Forecasting | Proceedings of the AAAI Conference on Artificial Intelligence"
[8]: https://ojs.aaai.org/index.php/AAAI/article/view/39585?utm_source=chatgpt.com "Towards Non-Stationary Time Series Forecasting with Temporal Stabilization and Frequency Differencing | Proceedings of the AAAI Conference on Artificial Intelligence"
[9]: https://openreview.net/pdf?id=bWcnvZ3qMb&utm_source=chatgpt.com "Published as a conference paper at ICLR 2024"
[10]: https://proceedings.neurips.cc/paper/2021/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html?utm_source=chatgpt.com "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting"
[11]: https://proceedings.mlr.press/v162/zhou22g.html?utm_source=chatgpt.com "FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting"
[12]: https://proceedings.neurips.cc/paper_files/paper/2022/hash/524ef58c2bd075775861234266e5e020-Abstract.html?utm_source=chatgpt.com "FiLM: Frequency improved Legendre Memory Model for Long-term Time Series Forecasting"
[13]: https://proceedings.neurips.cc/paper_files/paper/2023/hash/f1d16af76939f476b5f040fd1398c0a3-Abstract-Conference.html?utm_source=chatgpt.com "Frequency-domain MLPs are More Effective Learners in Time Series Forecasting"
[14]: https://ojs.aaai.org/index.php/AAAI/article/view/39736?utm_source=chatgpt.com "Sonnet: Spectral Operator Neural Network for Multivariable Time Series Forecasting | Proceedings of the AAAI Conference on Artificial Intelligence"
[15]: https://arxiv.org/abs/2407.12415?utm_source=chatgpt.com "Not All Frequencies Are Created Equal:Towards a Dynamic Fusion of Frequencies in Time-Series Forecasting"
[16]: https://www.researchgate.net/publication/342296603_Adaptive_Temporal-Frequency_Network_for_Time-Series_Forecasting?utm_source=chatgpt.com "Adaptive Temporal-Frequency Network for Time-Series Forecasting"
[17]: https://www.jmlr.org/beta/papers/v22/20-406.html?utm_source=chatgpt.com "From Fourier to Koopman: Spectral Methods for Long-term Time Series Prediction"
[18]: https://openreview.net/pdf?id=JePfAI8fah&utm_source=chatgpt.com "Published as a conference paper at ICLR 2024"
[19]: https://arxiv.org/abs/2507.09445?utm_source=chatgpt.com "Fourier Basis Mapping: A Time-Frequency Learning Framework for Time Series Forecasting"
[20]: https://arxiv.org/abs/2510.10145?utm_source=chatgpt.com "A Unified Frequency Domain Decomposition Framework for Interpretable and Robust Time Series Forecasting"
[21]: https://ojs.aaai.org/index.php/AAAI/article/view/39362?utm_source=chatgpt.com "M2FMoE: Multi-Resolution Multi-View Frequency Mixture-of-Experts for Extreme-Adaptive Time Series Forecasting | Proceedings of the AAAI Conference on Artificial Intelligence"
