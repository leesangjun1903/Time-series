# TimeSieve: Extracting Temporal Dynamics through Information Bottlenecks

**대상 논문:** *TimeSieve: Extracting Temporal Dynamics through Information Bottlenecks* — Ninghui Feng, Songning Lai, Jiayu Yang, Fobao Zhou, Zhenxiao Yin, Hang Zhao.
분석은 **arXiv:2406.05036v3 PDF**를 1차 근거로 하고, 공식 arXiv 페이지·저자 GitHub 코드와 2020년 이후 관련 논문을 교차검증했습니다. 업로드된 v3의 초록은 TimeSieve가 **wavelet transform으로 다중 시간척도 정보를 분리하고 Information Bottleneck으로 중복 정보를 제거하여 예측 일반화를 개선**하는 것을 핵심 목표로 제시합니다.  공식 arXiv 역시 같은 논문과 저자 정보를 확인할 수 있습니다. ([arXiv][1])

> **먼저 중요한 주의사항:** 아래에서 **[저자 보고]**는 논문이 실제 주장하거나 실험으로 보고한 내용이고, **[검토/해석]**은 제가 PDF·공식 구현·후속 연구를 대조해 판단한 내용입니다. 특히 “generalization”, “significant”, “70%”와 같은 표현은 논문의 표현과 통계적으로 입증된 의미를 구분해서 보겠습니다.

---

# 1. Executive Summary — 10문장 이내

1. TimeSieve는 시계열의 장·단기 패턴을 단일 표현에서 직접 학습하지 않고, **Wavelet Decomposition Block(WDB)** 으로 저주파 추세와 고주파 세부 성분을 먼저 분리하는 forecasting 모델입니다. 
2. 저자들은 기존 장기 시계열 예측 모델이 다중 시간척도 구조를 처리하기 위해 추가 파라미터나 데이터별 hyperparameter 조정을 요구하며, 특히 강한 계절성에서 유용한 신호와 중복 정보를 충분히 구별하지 못한다고 문제를 정의합니다. 
3. 각 wavelet 계수에 **Information Filtering and Compression Block(IFCB)** 을 적용하여 입력 정보를 압축하면서 예측에 필요한 정보를 보존하는 Information Bottleneck(IB) 원리를 사용합니다. 
4. 핵심 이론은 latent representation $Z$가 원 입력 계수 $\pi_i$와 공유하는 정보 $I(\pi_i;Z)$는 작게 하고, 복원에 필요한 정보 $I(Z;\hat{\pi}_i)$는 크게 만드는 것입니다. 
5. 필터링된 저주파·고주파 계수는 **Wavelet Reconstruction Block(WRB)** 으로 시간영역으로 복원되고 MLP가 최종 미래 값을 출력합니다. 
6. 저자 실험에서는 ETT, Exchange, Electricity, Weather의 7개 benchmark와 $H\in{48,96,144,192}$를 사용하며, 특히 ETTh1과 Exchange에서 강한 결과를 보였습니다. 
7. Table 1을 제가 개별 **dataset × horizon × metric** 셀 기준으로 다시 세면 TimeSieve가 최저 오차인 경우가 약 $39/56=69.6%$로, 공식 저장소의 “70%” 주장과 거의 일치하지만 이를 “70%의 데이터셋에서 승리”라고 읽는 것은 부정확합니다. ([GitHub][2])
8. Wavelet과 IFCB ablation은 두 구성요소가 효과가 있음을 지지하지만, 반복 실험의 표준편차·confidence interval·통계적 유의성 검정이 보고되지 않아 성능 우위의 **통계적 안정성**까지 입증되었다고 보기는 어렵습니다. 
9. 또한 논문의 “manual hyperparameter tuning 감소”라는 동기와 달리 IB loss weight와 wavelet basis에 성능이 상당히 민감하며, 논문 수식과 공개 코드의 IB loss 구현에도 중요한 차이가 있습니다.  ([GitHub][3])
10. 따라서 TimeSieve의 가장 중요한 기여는 단순한 SOTA 수치보다 **“시간-주파수 구조화 → 정보 압축 → 복원”이라는 inductive bias**이며, 향후에는 OOD/domain-shift 평가, conditional IB, cross-variable modeling, 불확실성 추정까지 확장해야 일반화 주장을 훨씬 강하게 만들 수 있습니다.

---

# 1-1. 연구의 목적과 필요성

## 저자가 해결하려는 문제

일반적인 multivariate forecasting은

$$
X=\{x_1,x_2,\ldots,x_T\}
\quad\longrightarrow\quad
Y=\{y_{T+1},y_{T+2},\ldots,y_{T+H}\}=F(X)
$$

로 정의할 수 있습니다.

여기서

* $T$: **lookback window**, 즉 예측에 사용하는 과거 구간 길이입니다.
* $H$: **forecast horizon**, 즉 앞으로 몇 시점을 예측할지 나타냅니다.
* $C$: 다변량 시계열의 변수 개수입니다.
* $F$: 과거 시계열을 미래 시계열로 변환하는 예측 모델입니다.

TimeSieve가 보는 핵심 문제는 $X$ 안에 있는 모든 변화가 동일한 가치의 “signal”은 아니라는 것입니다. 어떤 변화는 장기 trend, 어떤 변화는 반복적 seasonality, 어떤 변화는 단기 fluctuation이며, 그중 일부는 미래 예측과 거의 관계없는 **redundant information**일 수 있습니다. 저자들은 기존 모델이 이러한 성분을 한꺼번에 처리하면 불필요한 정보까지 표현 공간에 남아 학습을 방해한다고 주장합니다. 

**용어 — Redundant information(중복 정보):** 입력에는 존재하지만 목표를 예측하는 데 추가적인 정보가 거의 없는 특징입니다. 단순히 “noise”와 완전히 같은 개념은 아닙니다. 실제 신호여도 예측 목표와 무관하면 information bottleneck 관점에서는 제거 대상이 될 수 있습니다.

TimeSieve는 그래서 두 질문을 분리합니다.

**첫째**, 시계열을 장기적·국소적 변화로 어떻게 구조적으로 분해할 것인가? → **Wavelet**을 사용합니다.
**둘째**, 분해된 성분 중 무엇을 남길 것인가? → **Information Bottleneck**을 사용합니다.

즉 TimeSieve의 철학은

$$
\boxed{
\text{Raw time series}
\rightarrow
\text{structured frequency components}
\rightarrow
\text{information filtering}
\rightarrow
\text{reconstruction}
\rightarrow
\text{forecast}
}
$$

입니다.

---

# 2. 핵심 주장과 근거

| 핵심 주장                                                    | 저자가 제시한 근거                                                        | 위치                              | 검토 결과                                                                                           |
| -------------------------------------------------------- | ----------------------------------------------------------------- | ------------------------------- | ----------------------------------------------------------------------------------------------- |
| Wavelet으로 multi-scale temporal structure를 효과적으로 포착할 수 있다 | WDB/WRB 도입, wavelet 제거 ablation에서 대부분 MAE/MSE 악화                  | p.3–5, **Fig.1**, **Table 2**   | **지지됨.** 다만 공개 구현은 `pywt.dwt` 1단계라 엄밀한 의미의 깊은 multi-resolution hierarchy는 제한적입니다.               |
| IFCB가 중복 정보를 제거하면서 중요한 정보를 유지한다                          | $I(\pi_i;Z)$ 감소, $I(Z;\hat\pi_i)$ 증가를 목표로 하고 Fig.4에서 해당 방향의 변화 관찰 | p.5, 10–11, **Eq.8**, **Fig.4** | **부분 지지.** 한 데이터셋·한 horizon의 시각화이므로 일반적 인과 증거는 아닙니다.                                            |
| Approximation과 Detail 양쪽 모두에 IFCB를 적용하는 것이 좋다            | OnlyA/OnlyD 대비 전체 TimeSieve 우위                                    | p.9, **Table 3**                | 해당 두 benchmark에서는 대체로 지지됩니다.                                                                    |
| TimeSieve가 대부분의 setting에서 SOTA를 달성한다                     | 7 datasets × 4 horizons × MAE/MSE                                 | p.7, **Table 1**                | 제가 표를 재계산하면 약 **39/56=69.6%** 셀에서 최저값입니다. 다만 “70% of datasets”보다는 “약 70%의 평가 셀”이 정확합니다.         |
| Long horizon에 특히 효과적이다                                   | ETTh1, ETTm1 등 $H=144,192$에서 개선 보고                                | p.8, **Table 1**                | 대체로 관찰되지만 모든 dataset/metric에 성립하지 않습니다.                                                         |
| Hyperparameter tuning 부담을 줄인다                            | wavelet decomposition에는 learnable parameter가 필요 없다는 논리            | p.1–2                           | **강하게 표현하기 어렵습니다.** Appendix에서 IB weight와 wavelet basis의 선택에 따라 성능이 달라집니다.                      |
| 모델이 더 잘 generalize한다                                     | 여러 benchmark에서 좋은 평균적 성능                                          | Abstract, Conclusion            | **in-distribution benchmark breadth**에 대한 근거는 있으나 **domain shift/OOD generalization** 실험은 없습니다. |
| 모델이 효율적이다                                                | wavelet 자체가 additional learnable parameter를 요구하지 않는다는 주장          | p.2                             | FLOPs, parameter count, latency, memory 비교가 없어 **실증적 efficiency 우위는 입증되지 않았습니다.**               |

---

# 2-1. 해결 문제 → 제안 방법 → 수식 → 모델 구조

## A. Information Bottleneck의 출발점 — Eq. (1)

논문의 기본 아이디어는 다음과 같습니다.

$$
\max I(y;z),
\qquad
\min I(x;z)
$$



### 기호

* $x$: 원 입력입니다.
* $y$: 예측해야 하는 목표입니다.
* $z$: 압축된 latent representation입니다.
* $I(A;B)$: **mutual information(상호정보량)** 입니다.

**Mutual information:** 두 확률변수가 서로에 대해 얼마나 많은 정보를 제공하는지를 측정합니다. $I(A;B)=0$이면 이상적인 확률론적 의미에서 서로 독립입니다.

직관은 명확합니다.

$$
\text{필요 없는 }x\text{ 정보}
\quad\cancel{\longrightarrow}\quad Z
$$

이지만

$$
\text{미래 }y\text{를 설명하는 정보}
\quad\longrightarrow\quad Z
$$

는 유지해야 합니다.

따라서 IB를 단순히 “압축”이라고 이해하면 안 됩니다. **예측에 불필요한 정보를 선택적으로 압축하는 것**이 목적입니다.

---

## B. Wavelet Decomposition — Eq. (2)

```math
[\pi_a,\pi_d]
=
\text{WDB}(x(t))
```



### 기호

* $x(t)\in\mathbb{R}^{T\times C}$: 입력 multivariate time series입니다.
* $T$: 과거 sequence 길이입니다.
* $C$: 변수 수입니다.
* $\pi_a$: **approximation coefficient**, 저주파 성분입니다.
* $\pi_d$: **detail coefficient**, 고주파 성분입니다.
* $\text{WDB}$: Wavelet Decomposition Block입니다.

**Approximation coefficient:** 천천히 변하는 전체 수준, 장기 추세 등입니다.
**Detail coefficient:** 빠르게 변하는 국소 fluctuation, edge, 단기 진동 등의 성분입니다.

중요한 점은 Fourier transform과 달리 wavelet은 **시간 위치와 주파수 성격을 동시에 어느 정도 보존**한다는 것입니다. 따라서 “언제 발생한 고주파 변화인가”를 다루기에 유리합니다.

---

## C. 실제 wavelet 계수 — Eq. (5)

PDF의 Eq. (5)는 표기가 다소 압축되어 있습니다. 공식 README는 approximation/detail을 다음처럼 더 명확히 표현합니다. ([GitHub][2])

```math
\pi_a
=
\int x(t)\phi(t)\,dt
```

```math
\pi_d
=
\int x(t)\psi(t)\,dt
```

### 기호

* $\phi(t)$: **scaling function**, 저주파/평활 성분을 추출합니다.
* $\psi(t)$: **wavelet function**, 고주파/세부 성분을 추출합니다.
* $dt$: 시간축에 대한 적분입니다.

공개 코드는 실제로 **PyWavelets의 `dwt`와 `db1` wavelet**을 사용하여 각 variable을 따로 분해합니다. ([GitHub][3])

---

## D. Scaling/Wavelet basis — Eq. (6)

```math
\phi(u)
=
\sum_{k=0}^{N-1}
a_k\phi(2u-k)
```

```math
\psi(u)
=
\sum_{k=0}^{M-1}
b_k\phi(2u-k)
```



### 기호

* $a_k$: low-pass scaling filter의 계수입니다.
* $b_k$: wavelet/high-pass filter의 계수입니다.
* $k$: filter tap index입니다.
* $N,M$: 각 filter의 계수 개수입니다.
* $u$: 함수가 정의되는 시간/scale 변수입니다.

여기서 **basis**는 데이터를 바라보는 기본 모양이라고 이해하면 됩니다. Haar/DB1, SYM2 등이 서로 다른 필터 형태를 사용하기 때문에 동일 데이터라도 분해 결과가 달라집니다.

실제로 Appendix Table 7에서 basis에 따라 결과가 달라집니다. 예를 들어 Exchange $H=192$에서 MSE는 Haar $0.2840$, SYM2 $0.1804$, DB1 $0.1803$입니다. 

따라서 **“wavelet은 tuning이 필요 없다”는 주장은 제한적으로 해석해야 합니다.** learnable weight는 없지만 **wavelet family 자체의 선택은 hyperparameter**입니다.

---

## E. IFCB — Eq. (3)

논문 본문의 표기는

```math
\hat{\pi}_a
=
\text{IFCB}(\pi_a),
\qquad
\hat{\pi}_d
=
\text{IFCB}(\pi_d)
```

입니다. 

그런데 공식 README는 residual connection을 명시하여

```math
\hat{\pi}_a
=
\text{IFCB}(\pi_a)+\pi_a
```

```math
\hat{\pi}_d
=
\text{IFCB}(\pi_d)+\pi_d
```

라고 씁니다. ([GitHub][2])

**Residual connection:** 변환 결과에 원 입력을 다시 더해 정보가 지나치게 손실되는 것을 막는 연결입니다.

### [검토]

여기에는 **논문 수식과 README 사이의 명시적 차이**가 있습니다. 논문 텍스트에서는 residual connection을 사용한다고 설명하지만 Eq. (3)에는 $+\pi_i$가 없습니다. 따라서 재현 시에는 공식 코드까지 확인해야 합니다.

---

## F. Mutual Information — Eq. (7)

```math
I(\pi_i;\hat{\pi}_i)
=
D_{\mathrm{KL}}
\left[
p(i,\hat i)
\Vert
p(i)p(\hat i)
\right]
```

```math
=
H(\pi_i)-H(\pi_i|\hat{\pi}_i)
```



### 기호

* $i\in{a,d}$: approximation 또는 detail branch입니다.
* $\pi_i$: filtering 전 wavelet coefficient입니다.
* $\hat{\pi}_i$: filtering 후 coefficient입니다.
* $p(i,\hat i)$: 두 변수의 joint probability distribution입니다.
* $p(i)$, $p(\hat i)$: marginal distribution입니다.
* $D_{\mathrm{KL}}(P\Vert Q)$: **Kullback–Leibler divergence**입니다.
* $H(P)$: entropy입니다.
* $H(P|Q)$: conditional entropy입니다.

**KL divergence:** 두 확률분포가 얼마나 다른지를 측정합니다. 일반적인 거리와 달리 대칭이 아니므로 $D_{\mathrm{KL}}(P\Vert Q)\neq D_{\mathrm{KL}}(Q\Vert P)$일 수 있습니다.

**Entropy:** 확률변수의 불확실성 또는 정보량을 수치화합니다.

---

## G. Markov chain과 핵심 IFCB 목적함수 — Eq. (8)

논문은

$$
\pi_i
\rightarrow Z
\rightarrow
\hat{\pi}_i
$$

를 가정합니다.

**Markov chain:** $Z$가 주어졌을 때 뒤쪽 변수는 앞쪽 변수로부터 추가 정보를 직접 받지 않는 정보 흐름 구조입니다.

Data Processing Inequality에 따라

$$
I(\pi_i;Z)
\ge
I(\pi_i;\hat{\pi}_i)
$$

가 됩니다.

IFCB의 목적은

```math
\boxed{
\min
\left[
I(\pi_i;Z)
-
\beta I(Z;\hat{\pi}_i)
\right]
}
```

입니다. 

### 기호

* $Z$: stochastic latent representation입니다.
* $I(\pi_i;Z)$: 원 coefficient 정보가 $Z$에 얼마나 남아 있는지 나타냅니다.
* $I(Z;\hat{\pi}_i)$: $Z$가 복원해야 하는 coefficient에 얼마나 유용한지를 나타냅니다.
* $\beta$: 압축과 예측·복원 보존 사이의 trade-off입니다.

따라서 첫 항을 낮추면 입력 정보를 압축하고, 두 번째 항이 커지도록 하면 유용한 정보를 보존합니다.

---

# 핵심적인 일반화 해석

IB의 일반화 논리는 대략 다음과 같습니다.

```math
X
=
\underbrace{X_{\mathrm{stable}}}_{\text{예측에 필요한 구조}}
+
\underbrace{X_{\mathrm{nuisance}}}_{\text{우연/중복 정보}}
```

에서

$$
Z
\approx
f(X_{\mathrm{stable}})
$$

이 되도록 학습하면 훈련 데이터에만 존재하는 nuisance pattern을 memorization할 가능성이 줄어듭니다.

다만 **이것은 이론적 동기이지 TimeSieve 논문이 OOD 실험으로 입증한 결과는 아닙니다.**

---

## H. Gaussian latent representation — Eq. (9)

```math
p(z|i)
=
\mathcal{N}
\left(
\mu(i;\theta_\mu),
\Sigma(i;\theta_\Sigma)
\right)
=
\mathcal{N}(\mu_z,\Sigma_z)
```



### 기호

* $\mu_z$: latent Gaussian의 평균입니다.
* $\Sigma_z$: 논문 표기상 Gaussian dispersion parameter입니다.
* $\theta_\mu$: 평균을 추정하는 network parameters입니다.
* $\theta_\Sigma$: scale을 추정하는 network parameters입니다.

### 중요한 구현상의 해석

일반적인 Gaussian 표기에서 $\Sigma$는 covariance matrix를 뜻하는 경우가 많습니다. 그러나 공식 구현은

```text
std = softplus(...)
logvar = log(std²)
```

와 같은 구조를 사용합니다. 즉 TimeSieve 코드에서 $\Sigma_z$는 사실상 **standard deviation/scale vector** 역할에 더 가깝습니다. ([GitHub][3])

---

## I. Reparameterization trick — Eq. (10)

```math
z
=
\mu_z
+
\Sigma_z\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,I)
```



### 기호

* $\epsilon$: standard Gaussian random noise입니다.
* $I$: identity covariance입니다.
* $\mu_z$: 학습된 평균입니다.
* $\Sigma_z$: 학습된 scale입니다.

**Reparameterization trick:** 무작위 sampling을

$$
\text{learnable deterministic parameters}
+
\text{parameter-independent random noise}
$$

형태로 바꿔 backpropagation이 가능하게 하는 VAE 계열의 대표적 방법입니다.

### [검토]

공개 코드에서는 `torch.randn_like(std)`가 forward 때마다 실행됩니다. 즉 별도의 deterministic inference 처리를 하지 않는 한 **평가 시에도 stochastic prediction이 발생할 가능성**이 있습니다. ([GitHub][3])

따라서 repeated-seed mean/std를 보고하는 것이 특히 중요하지만, PDF에는 그러한 통계가 제시되어 있지 않습니다.

---

## J. Decoder probability — Eq. (11)

논문은

```math
q(\hat i|z)
=
e^{-\|\hat i-D(z;\theta_c)\|}
+
C
```

를 제시합니다. 

### 기호

* $q(\hat i|z)$: $Z$로부터 filtered coefficient를 복원할 조건부 분포입니다.
* $D(z;\theta_c)$: decoder입니다.
* $\theta_c$: decoder parameters입니다.
* $|\cdot|$: prediction/reconstruction discrepancy의 norm입니다.
* $C$: 논문이 도입한 상수입니다.

### [검토 — 수학적으로 모호한 부분]

$+C$가 지수 밖에 있는 이 식은 그대로는 **정규화된 probability density임이 보장되지 않습니다.** 논문은 normalization constant와 정확한 확률분포 형태를 추가로 유도하지 않습니다.

따라서 Eq. (11)은 엄밀한 probabilistic decoder 정의라기보다 **“복원 오차가 작을수록 likelihood가 높다”는 의도**를 나타내는 식에 가깝게 읽는 것이 안전합니다.

---

## K. WRB — Eq. (12)

```math
\hat{x}(t)
=
\sum \hat{\pi}_a\phi(t)
+
\sum \hat{\pi}_d\psi(t)
```



### 기호

* $\hat{x}(t)$: filtering 후 복원된 time-domain signal입니다.
* $\hat{\pi}_a$: filtered low-frequency coefficient입니다.
* $\hat{\pi}_d$: filtered high-frequency coefficient입니다.
* $\phi(t)$, $\psi(t)$: scaling/wavelet basis입니다.

그 뒤

```math
\hat{Y}
=
\text{MLP}
\left(
\text{WRB}(\hat{\pi}_a,\hat{\pi}_d)
\right)
```

로 최종 forecasting을 수행합니다.

---

# L. 논문의 최종 Loss — Eq. (13)

논문은

```math
\mathcal{L}
=
\mathcal{L}_o
+
\mathcal{L}_{IB}
```

```math
=
\mathcal{L}_o
+
D_{\mathrm{KL}}
\left[
\mathcal{N}(\mu_z,\Sigma_z)
\Vert
\mathcal{N}(0,I)
\right]
+
D_{\mathrm{KL}}
\left[
p(z)
\Vert
p(z|i)
\right]
```

로 적습니다. 

### 기호

* $\mathcal{L}_o$: 원래 forecasting loss입니다.
* $\mathcal{L}_{IB}$: IFCB regularization loss입니다.
* $\mathcal{N}(0,I)$: standard Gaussian prior입니다.
* $p(z|i)$: input coefficient를 본 뒤 latent distribution입니다.
* $D_{\mathrm{KL}}$: KL divergence입니다.

---

# 매우 중요한 논문–코드 차이

공개 코드는 실제로 훨씬 VAE에 가까운

```math
\mathcal{L}^{\text{code}}_{IB}
=
\mathcal{L}_{recon}
+
\beta
D_{\mathrm{KL}}
\left(
q(z|x)
\Vert
\mathcal{N}(0,I)
\right)
```

형태를 계산합니다.

그리고 IB module 반환 시 이 loss에 다시 $10^{-4}$를 곱합니다. 공식 코드에는

* reconstruction MSE,
* Gaussian KL,
* default $\beta=10^{-3}$,
* 최종 `loss * 0.0001`

가 명시되어 있습니다. ([GitHub][3])

즉 **논문의 Eq. (13)과 공개 코드의 실제 최적화 식은 완전히 동일하지 않습니다.**

이 차이는 단순한 표기 문제가 아닙니다. Information Bottleneck의 이론적 정당성을 평가하거나 논문을 재구현하려면 반드시 확인해야 할 부분입니다.

---

# 전체 모델 구조

$$
X\in\mathbb{R}^{T\times C}
$$

$$
\downarrow
$$

```math
[\pi_a,\pi_d]
=
\text{DWT}(X)
```

$$
\downarrow
$$

$$
\begin{cases}
\pi_a\rightarrow\text{IFCB}_a\rightarrow\hat{\pi}_a\\
\pi_d\rightarrow\text{IFCB}_d\rightarrow\hat{\pi}_d
\end{cases}
$$

$$
\downarrow
$$

```math
\hat X
=
\text{IDWT}(\hat{\pi}_a,\hat{\pi}_d)
```

$$
\downarrow
$$

$$
\hat Y=\text{MLP}(\hat X)
$$

Figure 1이 바로 이 전체 pipeline을 보여 줍니다. 

공식 코드를 보면 DWT가 **각 variable별로 독립 적용**되고, 이후 IB의 Linear layer도 temporal dimension을 처리하는 형태입니다. ([GitHub][3])

### [검토 — 중요한 구조적 특징]

이 구현에는 명시적인 **cross-channel attention/channel mixing**이 거의 없습니다. 따라서 multivariate forecasting이지만 강한 변수 간 dependency를 직접 modeling하는 구조는 아닙니다.

이는 두 가지 상반된 효과를 가질 수 있습니다.

$$
\text{Channel independence}
\Rightarrow
\begin{cases}
\text{parameter sharing / overfitting 감소 가능}\\
\text{cross-variable causal signal 손실 가능}
\end{cases}
$$

특히 Electricity처럼 $C=321$인 고차원 데이터에서 TimeSieve가 PatchTST 등에 압도적이지 않은 점은 이 관점에서 후속 검증할 가치가 있습니다. Electricity 데이터의 변수 수는 Table 4에서 321로 보고됩니다. 

---

# 3. 주장–페이지/Figure/Table 대응표

| 내용                             | 근거 위치                    |
| ------------------------------ | ------------------------ |
| 기존 문제 정의                       | p.1–2                    |
| IB 기본 원리                       | p.3, Eq. (1)             |
| 전체 TimeSieve architecture      | **p.3, Figure 1**        |
| WDB                            | p.3–5, Eq. (2), (5), (6) |
| IFCB architecture              | **p.4, Figure 2**        |
| Mutual information             | p.5, Eq. (7)             |
| IFCB objective                 | p.5, Eq. (8)             |
| Gaussian latent                | p.5, Eq. (9)             |
| Reparameterization             | p.6, Eq. (10)            |
| Decoder formulation            | p.6, Eq. (11)            |
| WRB                            | p.6, Eq. (12)            |
| 최종 training loss               | p.6–7, Eq. (13)          |
| 전체 benchmark                   | **p.7, Table 1**         |
| Wavelet ablation               | **p.8, Table 2**         |
| IFCB ablation                  | **p.9, Table 3**         |
| $\mu$–STD convergence          | **p.9, Figure 3**        |
| IB mutual-information dynamics | **p.10, Figure 4**       |
| Dataset 규모                     | p.15, Table 4            |
| 구현 설정                          | p.15, Table 5            |
| IB weight sensitivity          | p.15–16, Table 6         |
| Wavelet-basis sensitivity      | p.16, Table 7            |
| 결론/후속 연구                       | p.11                     |

---

# 4. 저자가 직접 보고한 결과와 제 해석의 분리

## 4.1 성능

### [저자 보고]

ETTh1, $H=48$:

$$
\text{TimeSieve MAE}=0.361,\quad
\text{MSE}=0.341
$$

Koopa:

$$
0.385,\quad0.364
$$

저자는 각각 약 **6.2%, 6.3% 개선**이라고 설명합니다. 

Exchange, $H=48$:

$$
0.139/0.045
$$

로 Koopa의

$$
0.149/0.046
$$

보다 낮습니다. 저자는 각각 약 **6.7%, 2.2% 개선**으로 보고합니다. 

---

### [검토/해석]

TimeSieve가 항상 우수한 것은 아닙니다.

예를 들어 ETTh2 $H=192$에서는

$$
\text{TimeSieve MSE}=0.377
$$

이지만

$$
\text{Koopa MSE}=0.353
$$

입니다.

ETTm2의 여러 horizon에서는 DLinear/PatchTST가 MSE에서 더 좋고, Electricity에서도 PatchTST 등이 여러 MSE 결과에서 우위입니다. 전체 수치는 Table 1에서 확인됩니다. 

따라서 정확한 표현은

> **“TimeSieve가 다수의 dataset–horizon–metric 조합에서 최고 성능을 얻었다.”**

이지

> **“어떤 데이터에서도 기존 모델보다 항상 우수하다.”**

가 아닙니다.

---

## 4.2 “70%”에 대한 검토

공식 arXiv/GitHub는 “outperforms existing SOTA methods on **70% of the datasets**”라고 표현합니다. ([arXiv][1])

그러나 Table 1은

```math
7\text{ datasets}
\times
4\text{ horizons}
\times
2\text{ metrics}
=
56
```

개의 비교 셀입니다.

표를 그대로 다시 세면 제가 확인한 최저 오차 셀은 약

```math
39/56
=
69.64\%
```

입니다.

따라서 **70%는 “dataset 자체의 70%”라기보다 평가 셀의 약 70%를 의미했을 가능성이 높습니다.**

이 부분은 논문의 표현이 더 정확했어야 합니다.

---

# 5. 통계적으로 취약한 부분과 비교 불가능한 수치

## 5.1 반복 실험 통계가 없습니다

Table 1은 단일 MAE/MSE 값만 제공합니다. PDF에서 제가 확인한 범위에는 다음이 보고되지 않습니다.

* independent random seeds별 결과
* mean $\pm$ standard deviation
* confidence interval
* paired statistical test
* bootstrap interval

따라서 논문에서 사용하는 “significant improvement”는 **통계적 유의성 $p<0.05$를 의미한다고 볼 근거가 없습니다.**

특히 IFCB는 stochastic sampling을 사용하므로 반복 실험 분산이 더 중요합니다.

---

## 5.2 MAE와 MSE의 숫자 자체를 직접 비교해서는 안 됩니다

논문은 MSE가 MAE보다 outlier에 더 민감하다고 설명하는데, 이 이론적 설명 자체는 타당합니다. 

하지만

```math
\text{MAE}
=
\frac{1}{n}\sum |y_i-\hat y_i|
```

와

```math
\text{MSE}
=
\frac{1}{n}\sum(y_i-\hat y_i)^2
```

는 **단위 자체가 다릅니다.** 논문의 정의도 Appendix에 명시되어 있습니다. 

따라서 “MSE가 MAE보다 숫자가 크므로 성능이 나쁘다”와 같은 비교는 성립하지 않습니다.

---

## 5.3 “No manual tuning” 주장에는 중요한 단서가 있습니다

Table 6의 IB weight 후보는

$$
10^{-6},10^{-5},10^{-3},10^{-2},10^{-1}
$$

이며, 최적값이 horizon별로 달라집니다. 

예를 들어 Exchange에서:

| Horizon | 최저 MAE weight | 최저 MSE weight |
| ------: | ------------: | ------------: |
|      48 |     $10^{-2}$ |     $10^{-1}$ |
|      96 |     $10^{-3}$ |     $10^{-5}$ |
|     144 |     $10^{-1}$ |     $10^{-3}$ |
|     192 |     $10^{-2}$ |     $10^{-2}$ |

즉 **IB hyperparameter는 성능에 민감합니다.**

Wavelet basis 또한 Table 7에서 결과에 영향을 미칩니다. 

따라서 “wavelet 자체에는 학습 parameter가 없다”와 “모델 전체에 dataset-specific tuning이 필요 없다”는 서로 다른 주장입니다.

---

## 5.4 $T=2H$ 때문에 타 논문의 published number와 직접 비교하기 어렵습니다

TimeSieve는 모든 실험에서

$$
T=2H
$$

로 lookback을 설정합니다. 

따라서 예를 들어 다른 논문이 모든 horizon에서 $T=96$을 사용했다면 그 논문의 원 논문 숫자와 TimeSieve의 숫자를 직접 나란히 두고 “몇 % 향상”이라고 하는 것은 공정하지 않습니다.

**같은 split, 같은 lookback, 같은 normalization, 같은 horizon, 같은 metric**에서 재실험한 경우에만 직접 비교해야 합니다.

---

## 5.5 다른 논문의 “38%”, “14.8%”와 TimeSieve의 “70%”는 비교 불가능합니다

예를 들어 Autoformer는 원 논문에서 여섯 benchmark에 대한 **38% relative improvement**를 주장합니다. ([NeurIPS Proceedings][4])
FEDformer는 multivariate/univariate prediction error가 각각 **14.8%, 22.6% 감소**했다고 보고합니다. ([Proceedings of Machine Learning Research][5])

반면 TimeSieve의 70%는 앞서 설명했듯 **win-count 비율에 가까운 값**입니다.

따라서

$$
70\% > 38\%
$$

이라고 하여 TimeSieve가 Autoformer보다 2배 좋다는 식의 비교는 **완전히 잘못된 비교**입니다.

---

# 5.6 논문 내부의 정합성 문제

몇 가지는 재현 전에 반드시 확인해야 합니다.

**첫째**, Table 3은 Exchange와 **ETTh2** 결과인데 본문은 해당 $0.333/0.302$ 값을 **ETTh1**이라고 기술합니다. 

**둘째**, Table 2 caption은 Exchange와 ETTh2 비교라고 쓰지만 실제 표에는 **ETTm1도 포함**되어 있습니다. 

**셋째**, Appendix에서 LightTS를 설명하는 참고문헌은 실제로 **time-series classification** 논문으로 기재되어 있는데 Table 1에서는 forecasting baseline으로 사용합니다. baseline의 정확한 구현·reference identity를 추가 확인할 필요가 있습니다. 

**넷째**, 가장 중요하게는 앞서 설명했듯 **논문의 Eq. (13)과 GitHub loss implementation이 동일하지 않습니다.** ([GitHub][3])

---

# 6. 이 문서가 답하지 않는 질문

1. **IB weight는 validation set만으로 선정했는가?** 아니면 각 test horizon을 본 뒤 선택했는가? 이를 명확히 해야 selection bias를 배제할 수 있습니다.
2. 각 결과는 **몇 개 random seed의 평균인가?** 공개된 표만으로는 알 수 없습니다.
3. stochastic latent sampling을 inference에서도 사용한다면 **Monte Carlo 평균을 사용하는가, single sample인가?**
4. Figure 4의 $I(X;T)$와 $I(T;Y)$는 **실제로 어떤 estimator로 계산했는가?** 계산 방법이 충분히 상세하지 않습니다.
5. IB가 제거하는 성분이 정말 “redundancy/noise”인지, 미래의 rare event에 필요한 고주파 정보가 아닌지 어떻게 판정하는가?
6. 한 번도 보지 못한 dataset, device, regime에 대해 성능이 유지되는가?
7. 평균·분산·계절성 등이 변하는 **distribution shift** 상황에서도 IFCB가 도움이 되는가?
8. $T=2H$가 아닌 arbitrary lookback에서도 동일하게 작동하는가?
9. multi-level DWT 또는 wavelet packet을 사용하면 한 단계 DWT보다 나은가?
10. 321-channel Electricity처럼 변수 수가 큰 상황에서 **cross-variable interaction**을 명시적으로 모델링하면 성능이 개선되는가?
11. Wavelet+IFCB의 실제 parameter count, FLOPs, GPU latency 및 memory overhead는 얼마인가?
12. extreme value가 중요한 예측 문제에서도 IB가 유용한가?
13. $\beta$ 및 wavelet basis를 dataset마다 바꾸지 않고 **하나의 global configuration**으로 고정해도 결과가 유지되는가?

이 질문들은 특히 “generalization”을 강하게 주장하려면 답해야 합니다.

---

# 7. 가장 중요한 그림 5개의 해석

중요한 사실이 하나 있습니다. **업로드된 v3에는 번호가 부여된 Figure가 Figure 1–4까지만 있습니다. Figure 5는 검색되지 않습니다.** 
따라서 존재하지 않는 Figure 5를 만들어내지 않고, **Figure 1–4 + 가장 중요한 결과표 Table 1**을 다섯 번째 핵심 시각적 증거로 해석하겠습니다.

## Figure 1 — 전체 TimeSieve Architecture, p.3

가장 중요한 그림입니다.

$$
X
\rightarrow
\text{WDB}
\rightarrow
(\pi_a,\pi_d)
\rightarrow
\text{IFCB}
\rightarrow
(\hat\pi_a,\hat\pi_d)
\rightarrow
\text{WRB}
\rightarrow
\text{MLP}
\rightarrow
\hat Y
$$

의 데이터 흐름을 한 번에 보여 줍니다. 

### 해석

TimeSieve의 핵심은 “더 복잡한 predictor”를 만드는 것이 아닙니다.

$$
\boxed{
\text{예측하기 쉬운 표현으로 입력을 바꾸고 나서 단순한 MLP가 예측하도록 한다}
}
$$

는 구조입니다.

이는 매우 중요한 설계 철학입니다. representation이 좋아지면 forecasting head를 무조건 거대하게 만들 필요가 없습니다.

---

## Figure 2 — IFCB, p.4

Figure 2는

$$
\pi_i
\rightarrow
(\mu_z,\Sigma_z)
\rightarrow
z=\mu_z+\Sigma_z\epsilon
\rightarrow
D(z)
\rightarrow
\hat\pi_i
$$

과정을 보여 줍니다. 

### 해석

이 구조는 본질적으로 **VAE-style stochastic bottleneck**입니다.

단순히 feature를 dropout하는 것이 아니라 입력 계수를 **확률분포의 파라미터로 바꾼 뒤 latent sample을 생성**합니다.

즉 같은 입력에서도

$$
z^{(1)}\neq z^{(2)}
$$

가 될 수 있으므로 representation에 regularizing effect가 생깁니다.

하지만 그만큼 evaluation variance도 반드시 측정해야 합니다.

---

## Figure 3 — Latent $\mu$–STD trajectory, p.9

그림에서 training batch가 진행될수록 STD는 대체로 큰 값에서 작은 값으로 이동하고 $\mu$도 좁은 영역으로 수렴하는 경향을 보입니다. 저자들은 이를 latent representation이 보다 보편적인 영역으로 수렴하고 prediction certainty가 증가하는 현상으로 해석합니다.  

### 제 해석

이 그림은 **latent distribution의 compression이 실제로 일어나고 있다는 정성적 증거**로는 의미가 있습니다.

그러나

$$
\text{STD 감소}
\not\Rightarrow
\text{generalization 개선이 자동으로 증명됨}
$$

입니다.

STD가 너무 줄어들면 오히려 latent가 정보를 지나치게 잃는 **posterior collapse에 가까운 상태**가 될 수도 있기 때문입니다.

따라서 OOD error와 latent compression을 함께 그렸다면 훨씬 강한 근거가 되었을 것입니다.

---

## Figure 4 — Information Plane + MSE, p.10

저자에 따르면 iteration이 진행되면서

$$
I(X;T)\downarrow,
\qquad
I(T;Y)\uparrow,
\qquad
MSE\downarrow
$$

방향이 관찰됩니다. 

그리고 이를 “입력 중 불필요한 정보를 버리면서 출력에 필요한 정보는 더 많이 보존한다”는 IB 원리의 실증적 증거로 해석합니다. 

### 제 해석

논문의 **이론과 가장 직접적으로 연결된 그림**입니다.

다만 다음 이유로 “IB가 성능 향상의 원인임을 증명한다”고까지 말하기는 어렵습니다.

* Exchange dataset 하나입니다.
* $H=48$ 하나입니다.
* MI estimator가 충분히 상세하게 기술되지 않았습니다.
* training iteration과 MSE가 동시에 변하기 때문에 인과관계가 분리되지 않습니다.
* 독립 반복 실험의 confidence band가 없습니다.

즉 **mechanism-consistent evidence**이지 **causal proof**는 아닙니다.

---

## 다섯 번째 핵심 시각적 증거 — Table 1, p.7

Table 1은 7개 데이터셋, 4개 forecast horizon에 대한 전체 성능을 제공합니다. 

가장 중요한 메시지는 “항상 승리”가 아니라 **dataset dependence**입니다.

$$
\text{강함: ETTh1,\ Exchange}
$$

$$
\text{혼합: ETTh2,\ ETTm1,\ Weather}
$$

$$
\text{상대적으로 약함: Electricity의 여러 setting,\ ETTm2의 MSE}
$$

입니다.

따라서 Table 1은 동시에 TimeSieve의 강점과 **일반화 주장의 한계**를 보여 주는 표입니다.

---

# 8. 결론과 시사점

## 저자가 제시한 결론

저자들은 TimeSieve가

* wavelet decomposition/reconstruction,
* information bottleneck filtering,
* MLP prediction

을 결합하여 여러 종류의 시계열과 forecast length에 대응할 수 있으며 finance·climate 등의 영역으로 적용 가능하다고 결론 내립니다. 향후에는 **dataset에 대한 robustness 강화** 및 **multimodal time series로의 확장**을 제시합니다. 

---

# 8-1. 모델의 일반화 성능 향상 가능성

TimeSieve의 일반화를 발전시키려면 제가 보기에는 다음 방향이 우선순위가 높습니다.

### ① Conditional Information Bottleneck

기존 IB:

$$
\min
I(X;Z)-\beta I(Z;Y)
$$

에서 temporal context 자체까지 압축해 버릴 위험이 있습니다.

ICLR 2024의 **Conditional Information Bottleneck Approach for Time Series Imputation**은 conventional IB를 시계열에 직접 적용하면 temporal dependency가 손실될 수 있음을 지적하고, temporal context에 조건부로 redundancy를 제거합니다. ([OpenReview][6])

TimeSieve에도 이를 적용하면 예를 들어

```math
\min
I(\pi_i;Z\mid C_t)
-
\beta I(Z;\hat\pi_i\mid C_t)
```

와 같은 방향을 고려할 수 있습니다.

여기서 $C_t$는 주변 temporal context 또는 regime representation입니다.

**핵심 장점:** “변화한다”는 이유만으로 정보를 제거하지 않고, **현재 시간적 문맥에서 불필요한 변화만 제거**할 수 있습니다.

---

### ② Wavelet scale을 하나가 아니라 계층적으로 만들기

공개 코드는 한 단계 `pywt.dwt(..., wavelet='db1')`를 사용합니다. ([GitHub][3])

향후에는

$$
X
\rightarrow
\{\pi_{a}^{(1)},\pi_d^{(1)},\pi_d^{(2)},\ldots,\pi_d^{(L)}\}
$$

처럼 multi-level wavelet/wavelet packet을 만든 뒤 **scale별 bottleneck strength $\beta_l$**를 학습할 수 있습니다.

그러면

* 장기 trend는 약하게 압축,
* noisy high-frequency band는 강하게 압축,
* 중요한 중간주기는 보존

하는 것이 가능합니다.

TimeMixer와 TimeMixer++가 다양한 temporal scales/resolutions을 동시에 활용하는 방향으로 발전한 것도 이 접근의 타당성을 뒷받침합니다. ([OpenReview][7])

---

### ③ Rare/extreme-event 보존

TimeSieve 저자 자신도 MSE가 일부 dataset에서 상대적으로 불리한 이유 중 하나로 큰 오차/extreme values 문제를 언급합니다. 

따라서

```math
\mathcal L
=
\mathcal L_{\text{forecast}}
+
\lambda_{IB}\mathcal L_{IB}
+
\lambda_{tail}\mathcal L_{tail}
```

같이 **tail-preserving loss**를 별도로 두는 방식을 추천할 수 있습니다.

즉 “고주파 = 제거할 noise”라는 암묵적 판단을 피해야 합니다.

---

### ④ Cross-variable dependency 추가

현재 구현은 channel-independent 성격이 강합니다.

따라서

$$
Z_{\text{temporal}}
\rightarrow
\text{Channel Mixer}
\rightarrow
Z_{\text{joint}}
$$

를 넣거나 iTransformer처럼 **variable 자체를 token으로 만들어 attention**을 수행하는 hybrid가 가능합니다.

iTransformer는 variate token을 사용하여 multivariate correlation을 직접 modeling하고 arbitrary lookback 및 variate generalization을 강조합니다. ([OpenReview][8])

이는 고차원 센서 시계열에서 상당히 중요한 확장입니다.

---

### ⑤ 진짜 generalization test를 도입

현재 benchmark만으로는

$$
P_{\text{train}}(X,Y)
\approx
P_{\text{test}}(X,Y)
$$

인 **in-distribution generalization**을 주로 측정합니다.

향후에는 명시적으로

$$
P_{\text{train}}(X,Y)
\neq
P_{\text{test}}(X,Y)
$$

인 조건을 만들어야 합니다.

예를 들면:

* unseen domain
* unseen device
* altered seasonality
* amplitude scaling
* missing sensor
* frequency drift
* noise variance shift
* unseen forecast horizon

등입니다.

이 실험에서 TimeSieve가 일반 모델보다 잘 버틴다면 “information bottleneck이 진짜 generalization을 높인다”는 주장이 훨씬 강해집니다.

---

# 8-2. 2020년 이후 최신 관련 연구와 비교

아래 수치는 **각 논문의 자체 실험**이므로 TimeSieve와 직접적인 숫자 비교는 하지 않습니다.

| 연구                 |   연도 | 핵심 방법                                                       | TimeSieve와 관계                                                     | 일반화 관점                                            |
| ------------------ | ---: | ----------------------------------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------- |
| **Autoformer**     | 2021 | progressive decomposition + Auto-Correlation                | TimeSieve가 문제의식을 비교하는 대표적 decomposition 모델                        | 명시적 periodic structure를 활용하지만 IB 없음               |
| **FEDformer**      | 2022 | trend/seasonal decomposition + Fourier frequency modeling   | TimeSieve보다 앞선 frequency-domain forecasting 계열                    | global frequency structure에 강점                    |
| **PatchTST**       | 2023 | patching + channel independence                             | TimeSieve와 마찬가지로 단순한 inductive bias의 중요성을 보여 줌                    | 긴 lookback 활용 및 self-supervised representation 장점 |
| **TimesNet**       | 2023 | multi-periodicity를 2D representation으로 변환                   | Wavelet과 다른 방식의 frequency/period structure 추출                     | forecasting뿐 아니라 5개 분석 task에서 task-general        |
| **iTransformer**   | 2024 | variate를 token으로 뒤집어 attention                              | TimeSieve가 약한 cross-variable modeling의 보완책                        | arbitrary lookback/variates generalization 강조     |
| **TimeMixer**      | 2024 | decomposable multiscale mixing                              | TimeSieve의 multi-scale 목표와 가장 가까운 경쟁 방향                           | fixed wavelet보다 learned multi-scale fusion이 유연    |
| **Conditional IB** | 2024 | temporal context를 조건으로 IB 적용                                | TimeSieve의 IB를 직접 개선할 수 있는 이론적 방향                                 | temporal dependency 손실 방지                         |
| **TimeX++**        | 2024 | IB objective를 수정하여 trivial solution과 distribution shift를 완화 | forecasting은 아니지만 TimeSieve의 IB 설계에 중요한 경고                        | IB 목적함수 자체가 distribution shift를 만들어낼 수 있음을 강조     |
| **ModernTCN**      | 2024 | large effective receptive field의 pure convolution           | 복잡한 IB가 항상 필요한 것은 아니라는 강한 baseline                                | 여러 task에서 efficiency/general capability 강조        |
| **CycleNet**       | 2024 | periodic cycle을 명시적으로 학습하고 residual을 예측                     | TimeSieve는 redundancy를 제거, CycleNet은 periodic structure를 명시적으로 보존 | 안정적 periodicity를 nuisance로 오판하지 않는 대안             |
| **TimeMixer++**    | 2025 | time-domain multi-scale + frequency-domain multi-resolution | TimeSieve의 wavelet 2-band보다 훨씬 더 계층적                              | 8개 시계열 task를 대상으로 universal pattern modeling 지향   |

Autoformer는 decomposition을 모델 내부의 기본 block으로 만들고 Auto-Correlation으로 periodic dependency를 처리했습니다. ([NeurIPS Proceedings][4]) FEDformer는 seasonal-trend decomposition과 Fourier representation을 결합하고 sequence length에 대해 linear complexity를 제안했습니다. ([Proceedings of Machine Learning Research][5])

PatchTST는 시계열을 patch token으로 만들고 channel-independent processing을 사용해 긴 history를 효율적으로 이용합니다. ([OpenReview][9]) TimesNet은 여러 주기를 찾아 1D time series를 2D variation으로 변환하며 forecasting, imputation, classification, anomaly detection 등으로 범용화했습니다. ([OpenReview][10])

2024년 이후에는 단순한 decomposition을 넘어 **어떤 representation이 domain과 task를 넘어 유지되는가**가 더 중요한 방향이 됩니다. iTransformer는 variable-centric representation, TimeMixer는 multiscale mixing, ModernTCN은 large receptive-field convolution을 각각 사용합니다. ([OpenReview][8])

특히 TimeSieve와 직접적으로 중요한 것은 CIB와 TimeX++입니다. CIB는 conventional IB가 temporal context를 지나치게 압축할 수 있다고 지적하며, TimeX++는 IB 기반 objective 자체가 trivial solution이나 distribution-shift 문제를 만들 수 있어 objective를 수정해야 한다고 주장합니다. 둘 다 task는 TimeSieve와 다르므로 성능 수치를 직접 비교해서는 안 되지만, **TimeSieve 후속 연구의 이론적 설계에는 매우 중요한 연구**입니다. ([OpenReview][6])

CycleNet은 조금 다른 철학을 취합니다. “강한 주기성을 redundancy로 걸러낼 것인가?”가 아니라 **안정적인 periodic cycle 자체를 명시적으로 모델링하고 residual만 예측**합니다. 또한 기존 PatchTST/iTransformer에 plug-in할 수 있고 90% 이상의 parameter reduction도 보고합니다. ([NeurIPS Proceedings][11])

2025년 TimeMixer++는 더 나아가 time-domain의 여러 scale과 frequency-domain의 여러 resolution을 동시에 사용하고 forecasting뿐 아니라 8개 time-series task를 하나의 pattern-machine 관점으로 다룹니다. 이는 TimeSieve가 향후 “DB1 1-level wavelet + 단일 forecasting”에서 **adaptive multi-resolution representation + task/domain-general bottleneck**으로 발전할 수 있음을 시사합니다. ([OpenReview][12])

---

# TimeSieve가 앞으로 연구에 미치는 영향

TimeSieve의 지속적인 가치는 “wavelet을 새롭게 만들었다”거나 “IB를 처음 만들었다”는 데 있지 않습니다. 두 고전적 원리를 다음과 같이 연결했다는 점에 있습니다.

$$
\boxed{
\text{Temporal decomposition}
+
\text{task-aware information compression}
}
$$

이는 향후 시계열 연구에서 매우 재사용하기 좋은 설계 패턴입니다.

특히 다음 질문을 남깁니다.

$$
\text{무엇을 예측할 것인가?}
$$

뿐 아니라

$$
\boxed{
\text{예측하기 전에 어떤 정보를 버려야 하는가?}
}
$$

를 forecasting architecture의 핵심 설계 문제로 끌어왔다는 점입니다.

다만 후속 연구는 여기서 한 단계 더 나아가

$$
\boxed{
\text{“얼마나 버릴 것인가”}
\rightarrow
\text{“어떤 domain/scale/context에서 무엇을 버릴 것인가”}
}
$$

를 다뤄야 합니다.

---

# 최종 평가

**방법론적 독창성:** 높음 — wavelet decomposition과 stochastic IB를 forecasting pipeline으로 통합한 구조는 명확하고 직관적입니다.
**실험적 성능:** 좋음 — Table 1의 상당수 setting에서 경쟁력이 있고, 특히 ETTh1·Exchange에서는 일관되게 강합니다.
**일반화 증거:** 중간 — 여러 benchmark에 대한 breadth는 있지만 OOD/domain-shift generalization은 시험하지 않았습니다.
**통계적 엄밀성:** 부족 — random-seed variance, CI, significance test가 없습니다.
**재현성:** 주의 필요 — 논문 Eq. (13)과 공개 코드 loss, residual 표현 사이에 차이가 있습니다.
**후속 연구 가치:** 매우 높음 — Conditional IB, adaptive multi-resolution decomposition, channel mixing, OOD 평가 및 uncertainty calibration과 결합할 여지가 큽니다.

제가 이 논문에서 **가장 중요하게 받아들일 부분**은 “SOTA 70%”라는 숫자가 아니라 다음 원칙입니다.

$$
\boxed{
\text{복잡한 시계열을 먼저 구조적으로 분해한 후,}
\quad
\text{예측에 필요한 정보만 통과시키자.}
}
$$

그리고 향후 연구의 핵심은 이를

$$
\boxed{
\text{fixed wavelet + unconditional IB}
}
$$

에서

$$
\boxed{
\text{adaptive multi-resolution decomposition}
+
\text{context/domain-conditioned IB}
+
\text{cross-variable modeling}
}
$$

로 발전시키는 것이라고 판단합니다.

---

## 참고한 자료 및 사이트

| 자료 제목                                                                                             | 출처                                                                |
| ------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| **TimeSieve: Extracting Temporal Dynamics through Information Bottlenecks**                       | 사용자 제공 arXiv v3 PDF / arXiv ([arXiv][1])                          |
| **TimeSieve — Official GitHub Repository**                                                        | GitHub, xll0328/TimeSieve ([GitHub][2])                           |
| **TimeSieve.py — Official Implementation**                                                        | GitHub, xll0328/TimeSieve ([GitHub][3])                           |
| **Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting** | NeurIPS 2021 ([NeurIPS Proceedings][4])                           |
| **FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting**         | ICML 2022 / PMLR ([Proceedings of Machine Learning Research][5])  |
| **A Time Series is Worth 64 Words: Long-term Forecasting with Transformers (PatchTST)**           | ICLR 2023 / OpenReview ([OpenReview][9])                          |
| **TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis**                     | ICLR 2023 / OpenReview ([OpenReview][10])                         |
| **iTransformer: Inverted Transformers Are Effective for Time Series Forecasting**                 | ICLR 2024 / OpenReview ([OpenReview][8])                          |
| **TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting**                         | ICLR 2024 / OpenReview ([OpenReview][7])                          |
| **Conditional Information Bottleneck Approach for Time Series Imputation**                        | ICLR 2024 / OpenReview ([OpenReview][6])                          |
| **TimeX++: Learning Time-Series Explanations with Information Bottleneck**                        | ICML 2024 / PMLR ([Proceedings of Machine Learning Research][13]) |
| **ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis**               | ICLR 2024 / OpenReview ([OpenReview][14])                         |
| **CycleNet: Enhancing Time Series Forecasting through Modeling Periodic Patterns**                | NeurIPS 2024 ([NeurIPS Proceedings][11])                          |
| **TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis**          | ICLR 2025 / OpenReview ([OpenReview][12])                         |

**2026년 연구는 검색 과정에서 확인했지만, 아직 under-review 상태이거나 이번 분석에 필요한 1차 근거가 충분하지 않은 항목은 정량 비교에서 의도적으로 제외했습니다.** 이는 최신성을 위해 불확실한 결과까지 끌어와 비교하는 것보다, peer-reviewed/공식 공개 근거가 명확한 연구를 중심으로 분석하는 것이 정확하기 때문입니다.

[1]: https://arxiv.org/abs/2406.05036?utm_source=chatgpt.com "TimeSieve: Extracting Temporal Dynamics through Information Bottlenecks"
[2]: https://github.com/xll0328/TimeSieve "GitHub - xll0328/TimeSieve · GitHub"
[3]: https://github.com/xll0328/TimeSieve/blob/main/models/TimeSieve.py "TimeSieve/models/TimeSieve.py at main · xll0328/TimeSieve · GitHub"
[4]: https://proceedings.neurips.cc/paper_files/paper/2021/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html?utm_source=chatgpt.com "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting"
[5]: https://proceedings.mlr.press/v162/zhou22g.html?utm_source=chatgpt.com "FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting"
[6]: https://openreview.net/pdf?id=K1mcPiDdOJ&utm_source=chatgpt.com "Published as a conference paper at ICLR 2024"
[7]: https://openreview.net/pdf?id=7oLshfEIC2&utm_source=chatgpt.com "Published as a conference paper at ICLR 2024"
[8]: https://openreview.net/pdf?id=JePfAI8fah&utm_source=chatgpt.com "Published as a conference paper at ICLR 2024"
[9]: https://openreview.net/pdf?id=Jbdc0vTOcol&utm_source=chatgpt.com "Published as a conference paper at ICLR 2023"
[10]: https://openreview.net/pdf/98c0a5bad8225b6d1baf5c74047c4d04bacfcfa1.pdf?utm_source=chatgpt.com "Published as a conference paper at ICLR 2023"
[11]: https://proceedings.neurips.cc/paper_files/paper/2024/hash/bfe7998398779dde03cad7a73b1f81b6-Abstract-Conference.html?utm_source=chatgpt.com "CycleNet: Enhancing Time Series Forecasting through Modeling Periodic Patterns"
[12]: https://openreview.net/pdf/6de8153acf0c7175730f8022e49df3a7183b6900.pdf?utm_source=chatgpt.com "Published as a conference paper at ICLR 2025"
[13]: https://proceedings.mlr.press/v235/liu24bl.html?utm_source=chatgpt.com "TimeX++: Learning Time-Series Explanations with Information Bottleneck"
[14]: https://openreview.net/pdf/c0de77eed380b4b2736dfe855ed3cf0d62f7d8c1.pdf?utm_source=chatgpt.com "Published as a conference paper at ICLR 2024"
