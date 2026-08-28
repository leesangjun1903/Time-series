# Learning Pattern-Specific Experts for Time Series Forecasting Under Patch-level Distribution Shift

분석 기준은 사용자가 첨부한 **arXiv v2 논문 전체 35쪽**과 NeurIPS 2025 공식 공개본입니다. 첨부본의 논문 제목과 저자는 *Learning Pattern-Specific Experts for Time Series Forecasting Under Patch-level Distribution Shift*, Yanru Sun, Zongxia Xie, Emadeldeen Eldele, Dongyue Chen, Qinghua Hu, Min Wu입니다.  이 논문은 NeurIPS 2025 정식 논문으로 확인됩니다. ([NeurIPS Proceedings][1])

아래에서는 **저자가 직접 주장하거나 측정한 결과**와 **제가 논문을 검토해 내린 해석·비판**을 명시적으로 구분하겠습니다.

---

# 1. Executive Summary — 10문장 이내

1. 이 논문은 하나의 시계열 안에서도 서로 다른 patch가 계절성, 추세, 운전 상태 변화 등에 의해 서로 다른 분포와 패턴을 가질 수 있는데, 기존 모델이 이들을 하나의 동일한 예측 함수로 처리하는 것이 일반화 성능을 제한한다고 문제를 정의합니다. 
2. 저자들은 이를 **patch-level distribution shift** 문제로 보고, 각 패턴별로 서로 다른 예측기를 학습시키는 **TFPS(Time-Frequency Pattern-Specific)** 모델을 제안합니다. ([NeurIPS Proceedings][1])
3. TFPS는 시간 영역 Transformer와 주파수 영역 Fourier encoder를 병렬로 사용하는 **Dual-Domain Encoder(DDE)**, patch를 잠재 subspace로 분류하는 **Pattern Identifier(PI)**, 각 패턴을 별도 함수로 모델링하는 **Mixture of Pattern Experts(MoPE)**의 세 부분으로 구성됩니다. 
4. PI는 학습 가능한 여러 subspace와 patch embedding 사이의 affinity를 계산하고, 이를 self-supervised clustering loss로 날카롭게 만든 뒤 이 정보를 expert routing에 사용합니다. 
5. 따라서 TFPS의 핵심은 **“비정상성을 제거해서 모든 데이터를 같게 만들기”보다 “서로 다른 regime을 구별하고 각각 별도의 예측 함수를 학습하기”**에 있습니다.
6. 저자 보고 기준 Main Table 1에서 TFPS는 72개 metric-setting 중 57개에서 1위를 기록하며, 특히 distribution shift가 큰 데이터에서 강한 성능을 보였다고 보고합니다. 
7. 그러나 모든 경우에 우수한 것은 아니며 Exchange의 $H=720$에서는 큰 성능 저하가 있고, foundation model 비교에서는 Traffic에서 TFPS가 네 horizon 모두 열세입니다. 
8. 또한 논문에는 반복 실험에 대한 error bar·confidence interval·통계적 유의성 검정이 보고되지 않아, 작은 MSE 차이가 실제로 안정적인 개선인지 판단하기 어렵습니다. 
9. 가장 중요한 일반화 한계는 expert 수와 patch 길이가 사실상 고정되어 있으며, 학습 중 존재하지 않았던 **새로운 regime**이 등장했을 때 expert를 생성하거나 모델을 온라인으로 확장하는 구조가 아니라는 점입니다. 
10. 따라서 TFPS의 가장 중요한 연구적 기여는 단순히 또 하나의 forecasting architecture라기보다, **distribution shift를 제거해야 할 잡음이 아니라 서로 다른 predictive function을 요구하는 구조적 이질성으로 취급했다는 점**에 있다고 평가할 수 있습니다.

---

# 1-1. 연구의 목적과 필요성

## 문제의 출발점

일반적인 시계열 forecasting 문제는 과거

$$
X=[x_{t-L+1},\ldots,x_t]\in\mathbb{R}^{L\times C}
$$

로부터 미래

$$
Y=[x_{t+1},\ldots,x_{t+H}]\in\mathbb{R}^{H\times C}
$$

를 예측하는 것입니다.

여기서

* $L$: look-back window 길이
* $H$: forecast horizon
* $C$: 변수(channel, variate)의 수
* $x_t$: 시점 $t$의 다변량 관측값입니다.

논문의 문제 제기는 **같은 $X$ 안에서도 모든 부분 구간이 같은 데이터 생성 메커니즘을 따르지 않는다**는 것입니다. 예를 들어 앞부분은 정상 운전, 중간은 급격한 regime 변화, 후반은 점진적인 trend drift일 수 있습니다. 논문은 이를 Figure 1의 ETTh1 예제로 보여줍니다. 

**용어 설명 — Patch:** 긴 시계열을 일정 길이의 작은 구간으로 잘라 만든 일종의 “시계열 토큰”입니다.

---

## 기존 방법의 문제

논문은 기존 PatchTST류 모델의 암묵적인 전략을 **Uniform Distribution Modeling, UDM**이라고 부릅니다.

직관적으로는

$$
f_1(X)\approx f_2(X)\approx\cdots\approx f(X)
$$

처럼 모든 patch에 사실상 같은 예측 규칙 $f$를 적용하는 것입니다.

그런데 실제로 regime마다

$$
P(Y\mid X,Z=k)
$$

가 다르다면 더 적절한 모델은

$$
\hat Y=f_k(X),\qquad Z=k
$$

입니다.

즉 숨겨진 pattern/regime $Z$를 먼저 추론하고, 그에 맞는 predictive function을 사용해야 한다는 논리입니다.

**용어 설명 — Regime:** 데이터가 생성되는 상태 또는 운전 모드입니다. 예를 들어 정상 운전, 장비 열화, 계절 변화, 이상 운전 등이 서로 다른 regime일 수 있습니다.

---

## 매우 중요한 개념적 주의점

여기서 논문의 표현 중 하나는 엄밀하게 구분할 필요가 있습니다.

논문은 Figure 1에서 **Wasserstein distance**로 patch 간 분포 차이를 관찰하고 이를 concept drift 문제와 연결합니다. 그러나

$$
P(X)_t\neq P(X)_{t+1}
$$

이라는 사실만으로

$$
P(Y\mid X)_t\neq P(Y\mid X)_{t+1}
$$

까지 증명되는 것은 아닙니다.

전자는 주로 **covariate/distribution shift**, 후자는 엄밀한 의미의 **concept drift**에 가깝습니다.

따라서 **Figure 1은 patch-level distribution shift의 존재를 보여주는 좋은 근거이지만, predictive relationship 자체가 바뀌었다는 것을 직접 검정한 증거는 아닙니다.**

이 구분은 이 논문의 가장 중요한 이론적 주의점 중 하나입니다.

---

# 2. 핵심 주장과 근거

| 핵심 주장                                           | 저자의 근거                                                                  | 위치                            | 제 평가                                                    |
| ----------------------------------------------- | ----------------------------------------------------------------------- | ----------------------------- | ------------------------------------------------------- |
| 서로 다른 patch에 상당한 distribution shift가 존재한다.      | ETTh1 patch 간 Wasserstein distance를 시간·주파수 영역에서 시각화                     | p.2, **Figure 1**             | **강한 정성적 근거**, 단 한 데이터의 사례이고 concept drift 자체를 증명하지는 않음 |
| 시간+주파수 정보를 함께 보는 것이 유리하다.                       | DDE ablation에서 두 branch를 모두 사용할 때 성능 우수                                 | p.4, Fig.2; p.8, **Table 2**  | 비교적 설득력 있음                                              |
| PI가 patch를 의미 있는 pattern으로 분리한다.                | PI 제거/Linear 대체 시 MSE 증가, Wasserstein intra/inter-cluster separation 개선 | p.8 Table 2, p.30 **Fig.8–9** | 구조적 근거는 강함. 다만 주로 ETTh1/ETTh2                           |
| pattern별 expert가 uniform predictor보다 낫다.        | MoPE ablation 및 alternative predictor 비교                                | p.33, **Table 16**            | 설득력 있으나 2개 데이터 중심                                       |
| TFPS가 SOTA forecasting 성능을 달성한다.                | Main Table 1에서 57/72 top-1                                              | p.8, **Table 1**              | 전반적으로 강하지만 예외 존재                                        |
| shift가 심할수록 pattern-specific modeling의 이점이 커진다. | Weather와 ETTh1의 Wasserstein 및 expert-number 분석                          | p.23 Table 5, p.29 **Fig.7**  | **흥미로운 가설**, 통계적 상관분석은 없음                               |
| 많은 expert가 항상 좋은 것은 아니다.                        | $K_t,K_f$ 변화 실험                                                         | p.10 Fig.6 / p.29 Fig.7       | 지원됨                                                     |
| foundation model보다도 경쟁력이 있다.                    | AutoTimes, MOMENT, Timer 비교                                             | p.9 Table 3 / p.26 Table 7    | 일부 dataset에서 강함. **Traffic에서는 반대**                      |
| TFPS가 실용적 효율성을 가진다.                             | GPU memory / inference time                                             | p.31 **Table 12**             | **주의 필요: 표와 본문 숫자가 서로 불일치**                             |
| 새로운 pattern에 대해서도 현재 구조가 충분하다.                  | 그런 주장은 하지 않음                                                            | p.35 Limitations              | 오히려 저자 스스로 **미해결 문제라고 명시**                              |

---

# 2-1. 해결하려는 문제 → 수식 → 모델 구조

## 2-1-1. Patch embedding

입력

$$
X\in\mathbb{R}^{L\times C}
$$

를 길이 $P$인 patch들로 자릅니다.

논문 표기상 patch 수는

$$
N=
\left\lfloor
\frac{L-P}{S}+2
\right\rfloor
$$

입니다. 

여기서

* $P$: patch length
* $S$: stride
* $N$: 만들어진 patch 수
* $C$: 변수 수
* $P_i\in\mathbb{R}^{C\times P}$: $i$번째 patch입니다.

각 patch는 선형 projection으로

$$
P_i\rightarrow P'_i\in\mathbb{R}^{C\times D}
$$

가 되고 위치 정보를

$$
X_{PE,i}=P_i'+E_i
$$

로 더합니다.

* $D$: embedding dimension
* $E_i$: 학습 가능한 positional embedding
* $X_{PE}$: 위치 정보가 포함된 patch representation입니다.

**설계 의도:** patching 과정에서 “이 patch가 원래 몇 번째 구간인지”가 사라질 수 있으므로 $E_i$가 시간 순서를 복원합니다.

---

# 2-1-2. Dual-Domain Encoder

## A. Time-domain branch

시간 branch는 Transformer self-attention을 사용합니다.

```math
O_t
=
\text{Softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
```

$$
Q=X_{PE}W_Q,\qquad
K=X_{PE}W_K,\qquad
V=X_{PE}W_V
$$

입니다. 

기호는 다음과 같습니다.

* $Q$: Query, 현재 patch가 “무엇을 찾을 것인가”를 표현한 벡터
* $K$: Key, 각 patch가 어떤 정보를 가지고 있는지를 표현
* $V$: Value, 실제 전달될 정보
* $W_Q,W_K,W_V$: 학습 가능한 projection matrix
* $d_k$: Key vector 차원
* $QK^T$: patch들 사이의 유사성
* $\sqrt{d_k}$: 내적 크기가 지나치게 커지는 것을 억제하는 scaling term입니다.

결과적으로

$$
z_t\in\mathbb{R}^{C\times N\times D}
$$

라는 시간 영역 embedding을 얻습니다.

---

## B. Frequency-domain branch

주파수 branch에서는 self-attention을 Fourier operation으로 대체합니다.

```math
O_f
=
\mathcal{F}_{\text{patch}}
\left(
\mathcal{F}_{h}(X_{PE})
\right)
```

입니다. 

* $\mathcal{F}_{h}$: hidden/embedding 축 Fourier transform
* $\mathcal{F}_{\text{patch}}$: patch 축 Fourier transform
* $O_f$: frequency-domain representation입니다.

논문은 최종적으로 Fourier 결과의 **real part만 사용**합니다. 추가 실험 Table 13에서는 real part만 쓴 경우와 real+imaginary를 사용한 경우의 결과가 대부분 유사했습니다. 

**용어 설명 — Fourier transform:** 시간에 따라 변화하는 신호를 “어떤 주파수의 진동이 얼마나 포함되어 있는가”로 바꾸어 보는 표현입니다. 계절성이나 반복 주기를 파악하기 좋습니다.

---

# 2-1-3. Pattern Identifier: 이 논문의 핵심

TFPS가 기존 dual-domain 모델과 가장 크게 달라지는 부분입니다.

embedding

$$
z_i
$$

가 어느 latent pattern에 가까운지를 찾기 위해 $K$개의 subspace를

$$
\mathbf D=
[
D^{(1)},D^{(2)},\ldots,D^{(K)}
]
$$

로 만듭니다.

각

$$
D^{(j)}\in\mathbb{R}^{q\times d}
$$

가 $j$번째 pattern의 subspace basis입니다.

Algorithm 1에서는

$$
q=C D_h,\qquad
d=\frac{q}{K}
$$

로 둡니다. 논문에서는 embedding dimension과 subspace matrix를 모두 $D$라고 표기해 다소 혼동 가능성이 있으므로 여기에서는 embedding dimension을 $D_h$, subspace 전체를 $\mathbf D$로 구분했습니다. 실제 구현 기본값은 feature dimension $512$, encoder layer 수 $2$로 기술됩니다. 

**용어 설명 — Subspace clustering:** 데이터가 하나의 복잡한 공간에 모두 섞여 있다고 보기보다, 서로 다른 낮은 차원의 공간 여러 개에 나뉘어 있다고 가정하여 군집을 찾는 방법입니다.

---

## 2-1-4. Subspace 내부 basis 정규화: $R_1$

논문의 Eq. (3)은

```math
R_1
=
\frac12
\left\|
\mathbf D^T\mathbf D\odot I-I
\right\|_F^2
```

입니다. 

여기서

* $\mathbf D$: 모든 subspace basis를 모은 행렬
* $I$: identity matrix
* $\odot$: Hadamard product, 원소별 곱셈
* $|\cdot|_F$: Frobenius norm입니다.

직관적으로는 각 basis vector의 크기가 임의로 폭주하지 않게 하고 안정적인 subspace를 만들기 위한 regularization입니다.

---

# 2-1-5. 서로 다른 subspace를 분리: $R_2$

Eq. (4)는

```math
R_2
=
\frac12
\left\|
D^{(j)T}D^{(l)}
\right\|_F^2,
\qquad j\neq l
```

이며 전체 행렬 형태로는

```math
R_2
=
\frac12
\left\|
\mathbf D^T\mathbf D\odot O
\right\|_F^2
```

입니다. 

* $D^{(j)}$, $D^{(l)}$: 서로 다른 두 pattern subspace
* $O$: diagonal block은 $0$, off-diagonal block은 $1$인 mask matrix입니다.

$R_2$를 작게 만드는 것은

$$
D^{(j)T}D^{(l)}\approx 0
$$

을 유도하므로 서로 다른 subspace가 가능한 한 겹치지 않게 합니다.

두 regularizer는

$$
R=\alpha(R_1+R_2)
$$

로 결합되고, 논문에서는

$$
\alpha=10^{-3}
$$

를 사용합니다. 

---

# 2-1-6. Patch가 어떤 pattern에 속하는가: affinity

patch embedding $z_i$와 $j$번째 subspace의 친화도는

```math
s_{ij}
=
\frac{
\left\|z_i^TD^{(j)}\right\|_F^2+\eta d
}{
\sum_{\ell=1}^{K}
\left(
\left\|z_i^TD^{(\ell)}\right\|_F^2+\eta d
\right)
}
```

로 정의할 수 있습니다. 논문의 Eq. (6)을 denominator index가 혼동되지 않도록 다시 쓴 것입니다. 

* $s_{ij}$: patch $i$가 pattern/subspace $j$에 속할 affinity
* $z_i$: patch $i$의 embedding
* $D^{(j)}$: $j$번째 subspace basis
* $K$: subspace/expert 수
* $d$: subspace dimension
* $\eta$: smoothing parameter입니다.

$s_{ij}$가 크다면 $z_i$가 $D^{(j)}$가 span하는 방향에 강하게 투영된다는 뜻입니다.

---

# 2-1-7. Affinity sharpening

저자들은 원래 affinity $S$를 그대로 쓰지 않고 고신뢰 assignment를 강화합니다.

먼저

$$
f_j=\sum_r s_{rj}
$$

라 두면 Eq. (7)은

```math
\hat s_{ij}
=
\frac{
s_{ij}^2/f_j
}{
\displaystyle
\sum_{\ell=1}^{K}s_{i\ell}^2/f_\ell
}
```

로 이해할 수 있습니다. 

$s_{ij}$를 제곱하기 때문에

* 큰 affinity는 상대적으로 더 커지고,
* 작은 affinity는 더 작아집니다.

예를 들어

$$
S_i=[0.6,0.3,0.1]
$$

이었다면 sharpening 후에는 첫 번째 cluster의 상대적 중요성이 더 커집니다.

즉 **애매한 cluster assignment를 더 명확하게 만드는 self-training 방식**입니다.

---

# 2-1-8. Subspace clustering loss

Refined distribution $\hat S$가 원래 assignment $S$를 학습 목표로 끌어갑니다.

```math
L_{\text{sub}}
=
\text{KL}(\hat S\|S)
=
\sum_i\sum_j
\hat s_{ij}
\log
\frac{\hat s_{ij}}{s_{ij}}
```

입니다. 

* $\text{KL}$: Kullback-Leibler divergence
* $\hat s_{ij}$: sharpened target affinity
* $s_{ij}$: 현재 모델의 affinity입니다.

**용어 설명 — KL divergence:** 두 확률분포가 얼마나 다른지 측정하는 양입니다. $0$에 가까울수록 두 분포가 비슷합니다.

최종 PI loss는

```math
L_{\text{PI}}
=
R+\beta L_{\text{sub}}
```

입니다.

* $\alpha$: subspace geometry regularization 강도
* $\beta$: clustering loss의 강도입니다.

Appendix의 Figure 11에서는 너무 작은 $\beta$도, 너무 큰 $\beta$도 나쁘며 약 $0.1$ 부근을 권장합니다. 

---

# 2-1-9. Mixture of Pattern Experts

PI가 계산한 $s$를 router로 넘깁니다.

```math
G(s)
=
\text{Softmax}
\left(
\text{TopK}(s)
\right)
```

입니다. 

여기서

* $\text{TopK}(s)$: 가장 높은 affinity를 가진 $k$개 expert만 활성화
* $G_k(s)$: $k$번째 expert에 줄 normalized weight
* $E_k(\cdot)$: $k$번째 expert network입니다.

각 expert는 논문에서 **두 개의 linear layer + ReLU**로 구성된 MLP입니다.

최종 patch representation은 더 정확히 쓰면

```math
h
=
\sum_{k=1}^{K}
G_k(s)E_k(z)
```

가 됩니다. 

**용어 설명 — Mixture of Experts(MoE):** 하나의 큰 예측기만 사용하는 대신 여러 전문 예측기를 두고, 입력마다 router가 어떤 expert를 사용할지 선택하는 구조입니다.

이 구조의 의미가 중요합니다.

기존 모델:

$$
\hat Y=f(X)
$$

TFPS가 지향하는 모델:

```math
\hat Y
=
\sum_k
p(Z=k\mid X)
f_k(X)
```

입니다.

즉 TFPS는 실질적으로 **latent regime에 조건부인 mixture-of-regressions 형태**로 이해할 수 있습니다.

---

# 2-1-10. Time/Frequency 결합

두 branch에서

$$
h_t,\qquad h_f
$$

를 얻고, frequency branch를 inverse FFT 처리한 뒤

$$
h=
\text{concat}(h_t,h_f)
\in
\mathbb{R}^{C\times N\times 2D}
$$

로 결합합니다.

그리고

```math
\hat Y
=
\text{Linear}(h)
\in
\mathbb{R}^{H\times C}
```

로 최종 prediction을 만듭니다. 

---

# 2-1-11. 전체 목적함수

Forecasting loss는 MSE이며 PI loss까지 합쳐

```math
L
=
L_{\text{MSE}}
+
L_{\text{PI},t}
+
L_{\text{PI},f}
```

입니다. 

보다 명시적으로 쓰면

```math
L_{\text{MSE}}
=
\frac{1}{H}
\sum_{\tau=1}^{H}
\left(
\hat Y_{t+\tau}-Y_{t+\tau}
\right)^2
```

입니다.

따라서 optimization은 동시에 두 가지를 수행합니다.

$$
\boxed{
\text{예측 오차 감소}
+
\text{pattern cluster 구조 형성}
}
$$

즉 단순히 “좋은 feature”를 학습하는 것이 아니라 **forecasting에 유리하도록 pattern partition 자체를 end-to-end로 만드는 것**입니다.

---

# 3. 전체 모델 구조를 한 문장으로 표현하면

$$
X
\rightarrow
\text{Patch Embedding}
\rightarrow
\begin{cases}
\text{Time Encoder}
\rightarrow
\text{PI}_t
\rightarrow
\text{MoPE}_t\\[2mm]
\text{Frequency Encoder}
\rightarrow
\text{PI}_f
\rightarrow
\text{MoPE}_f
\end{cases}
\rightarrow
\text{Concat}
\rightarrow
\text{Linear Head}
\rightarrow
\hat Y
$$

입니다.

Figure 2가 바로 이 전체 구조를 보여줍니다. 

핵심은 **encoder보다 PI→MoPE 연결**입니다. DDE 자체는 Transformer/Fourier 계열의 기존 아이디어와 연결되지만, TFPS는 patch별 latent pattern을 알아내고 해당 pattern expert로 보내는 과정을 prediction pipeline 내부에 직접 넣습니다.

---

# 4. 저자가 직접 보고한 내용과 제 해석의 분리

| 항목                  | **저자가 직접 보고한 내용**                                           | **제 해석**                                                                                               |
| ------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| 연구 주제               | Patch 간 distribution shift 때문에 uniform modeling의 일반화가 나빠짐   | forecasting을 사실상 **latent regime identification + conditional regression** 문제로 다시 정의한 연구               |
| Figure 1            | time/frequency domain 모두 patch distribution discrepancy를 보임 | distribution shift는 입증하지만 $P(Y\mid X)$ 변화인 **concept drift 자체는 직접 입증하지 않음**                            |
| DDE                 | 두 domain이 complementary temporal information을 제공            | DDE만으로 novel하다고 보기보다 PI/MoPE가 핵심 novelty                                                               |
| PI                  | subspace clustering으로 evolving pattern을 식별                  | 단순 softmax router보다 **explicit geometry를 가진 router**라는 점이 중요한 차별점                                      |
| MoPE                | pattern마다 expert를 할당해 prediction accuracy 개선                | global model의 bias를 줄이는 대신 expert별 sample 수 감소에 따른 variance 위험이 생김                                     |
| 성능                  | Table 1에서 57/72 top-1                                       | benchmark상 강하지만 “항상 SOTA”는 아님                                                                          |
| 일반화                 | distribution-shift가 심한 데이터에서 더 강함                           | **in-distribution chronological generalization**에는 강한 근거가 있지만 unseen-regime/OOD generalization은 아직 미검증 |
| Foundation model 비교 | 대부분 TFPS가 AutoTimes/MOMENT/Timer보다 좋음                       | pretraining 규모와 adaptation protocol이 다른 모델 간 비교이므로 architecture superiority로 해석하면 안 됨                  |
| 실용성                 | accuracy와 efficiency의 균형이 좋음                                | Table 12와 본문의 inference-time 숫자가 충돌하므로 runtime superiority 수치는 재검증 필요                                  |
| 미래 연구               | 새로운 pattern 및 adaptive patch length를 연구                     | TFPS의 다음 단계는 **open-set/continual MoE**가 되어야 함                                                         |

---

# 5. 실제 성능 향상은 어느 정도인가?

## 5-1. Main Table 1

Main Table 1은 ILI의 경우

$$
H\in\{24,36,48,60\}
$$

나머지는

$$
H\in\{96,192,336,720\}
$$

에서 비교합니다. 

저자 집계로 TFPS는 **72개 setting 중 57개에서 1위**입니다.

또 저자는 전체 평균 기준으로

* time-domain 모델 대비 MSE 약 **9.5%**, MAE 약 **6.4%**
* frequency-domain 모델 대비 MSE 약 **16.9%**, MAE 약 **12.4%**
* time-frequency 모델 대비 MSE 약 **5.2%**, MAE 약 **2.2%**

개선을 보고합니다. 

다만 여기서 매우 중요한 점이 있습니다.

논문의 IMP는

```math
\text{IMP}
=
\frac{
\text{Avg MSE of baselines}
-
\text{MSE}_{\text{TFPS}}
}{
\text{Avg MSE of baselines}
}
\times100\%
```

입니다. 

즉 **best baseline 대비 개선율이 아니라 baseline들의 평균에 대한 개선율**입니다.

따라서

$$
\text{IMP}>0
$$

라고 해서 반드시 TFPS가 가장 좋은 모델이라는 뜻은 아닙니다.

이 부분은 성능표를 읽을 때 반드시 구분해야 합니다.

---

## 5-2. TFPS가 실제로 지는 사례

예를 들어 Main Table 1에서 ETTh1, $H=96$:

* TFPS MSE = $0.398$
* TSLANet = $0.387$
* iTransformer = $0.387$
* FEDformer = $0.385$

이므로 해당 setting에서는 TFPS가 최선이 아닙니다.

또 Exchange의 $H=720$에서 TFPS MSE는 **1.011**로, 다수 baseline의 약 $0.83\sim0.85$보다 상당히 나쁩니다.

즉 “shift-aware model이면 모든 장기 horizon에 강하다”는 결론은 성립하지 않습니다.

---

# 5-3. Foundation model 비교

Appendix의 더 상세한 hyperparameter-search 비교에서 TFPS는 48개 metric-setting 중 30개에서 1위를 기록합니다. 그러나 Traffic에서는 모든 horizon에서 TFPS의 IMP가 음수입니다.

예:

$$
H=96:\quad -5.5\%
$$

$$
H=192:\quad -5.3\%
$$

$$
H=336:\quad -3.1\%
$$

$$
H=720:\quad -1.1\%
$$

입니다. 

저자들은 Traffic의 distribution shift가 비교적 작기 때문에 pattern-specific modeling의 이점이 작다고 해석합니다. 

제가 보기에도 이 결과는 오히려 논문의 가설과 잘 맞는 흥미로운 반례입니다.

즉

$$
\text{heterogeneity가 작으면}
\quad
\text{expert specialization benefit} < \text{routing/estimation overhead}
$$

일 가능성이 있습니다.

---

# 5-4. Ablation은 꽤 설득력 있다

Table 2에서 full TFPS:

$$
\text{Time Encoder + PI + MoPE}
+
\text{Frequency Encoder + PI + MoPE}
$$

가 ETTh1/ETTh2 모든 horizon에서 가장 좋은 MSE를 냅니다. 

추가 Appendix에서도

* Time/Frequency PI 둘 다 사용: **Table 14**
* $R_1+R_2$ 둘 다 사용: **Table 15**
* MoPE 대신 multi-output predictor/attention 사용: **Table 16**

일 때 full TFPS가 대체로 가장 좋습니다.  

따라서 최소한 **“성능이 단순히 Transformer branch를 하나 더 붙였기 때문에 나온 것인가?”**라는 질문에는 어느 정도 대응하고 있습니다.

다만 대부분의 심층 ablation이 ETTh1/ETTh2에 집중되어 있으므로 모든 데이터셋에 동일하게 일반화된다고까지 말하기는 어렵습니다.

---

# 6. 통계적으로 취약한 부분과 비교 불가능한 수치

이 부분은 연구 검증 관점에서 중요합니다.

| 문제                           | 왜 중요한가                                                                | 판단                                |
| ---------------------------- | --------------------------------------------------------------------- | --------------------------------- |
| **Error bar / CI 없음**        | $0.398$ vs $0.401$ 같은 차이가 seed variance보다 작은지 판단 불가                   | **주요 취약점**                        |
| 반복 seed 결과 불명확               | initialization에 따른 MoE routing 변동을 알 수 없음                             | 취약                                |
| 유의성 검정 없음                    | “significant improvement”가 통계적 의미의 significant인지 확인 불가                | 취약                                |
| IMP가 평균 baseline 기준          | best baseline 대비 improvement로 오해하기 쉬움                                 | **비교 주의**                         |
| Main Table 1과 Table 6 조건 차이  | unified setting과 full hyperparameter search를 직접 같은 숫자로 비교 불가          | **비교 불가능**                        |
| Foundation model 비교          | pretrained foundation model과 task-specific TFPS는 학습 regime이 근본적으로 다름  | architecture-only 비교 불가           |
| Wasserstein 절대값              | 데이터 단위와 scaling 영향을 받기 때문에 서로 다른 dataset의 $0.011$과 $520$을 그대로 비교하기 위험 | **정규화 검증 필요**                     |
| shift severity ↔ expert 수 관계 | ETTh1/Weather 사례 위주이며 formal correlation test 없음                      | 가설 수준                             |
| t-SNE                        | visualization 결과가 seed/perplexity 등에 민감                               | 정성적 근거                            |
| expert 사례                    | Expert-0/4 몇 사례만 시각화                                                  | 전체 expert specialization 증명으로는 부족 |
| ablation 범위                  | 많은 추가 ablation이 ETTh1/ETTh2 중심                                        | 외적 타당성 제한                         |
| single chronological split   | 여러 시점의 반복 OOD/generalization 측정이 아님                                   | 실환경 drift 일반화 검증 부족               |

실제로 논문 자체의 NeurIPS checklist에서 “statistical significance/error bars” 질문이 **[NA]**로 표시되어 있으며, 본문에서도 error bar나 confidence interval을 제시하지 않습니다. 

여기서 checklist의 “NA” 자체도 다소 이상합니다. checklist 안내문은 **NA는 실험이 없는 경우**라고 설명하지만 이 논문에는 대규모 실험이 존재합니다. 따라서 이것 역시 문서상의 reporting inconsistency로 볼 수 있습니다.

---

## 또 하나의 명확한 숫자 불일치: Table 12

Table 12에는 TFPS의 평균 inference time이

$$
6.114\text{ ms}
$$

로 적혀 있습니다. PatchTST는 $4.861$ ms, TimesNet은 $12.306$ ms입니다. 

그런데 바로 아래 본문에는 TFPS가

$$
6.457\text{ ms}
$$

이고 PatchTST $17.851$ ms, TimesNet $72.196$ ms라고 서술합니다. 

두 숫자 세트는 상당히 다릅니다.

따라서 본문의

> TFPS가 PatchTST보다 2.8배 빠르다

같은 efficiency 주장은 **현재 공개 PDF만으로는 확정적으로 받아들이면 안 됩니다.**

모델의 forecasting 성능과 별개로, runtime comparison은 재실험이 필요합니다.

---

# 7. 논문이 답하지 않는 질문

### 1. 완전히 새로운 pattern이 들어오면 어떻게 되는가?

현재 구조는 기존 $K$개의 subspace 중 하나로 routing합니다.

하지만

$$
z_{\text{new}}
\notin
\bigcup_{k=1}^{K}\mathcal S_k
$$

인 새로운 regime이 나타나면 어떻게 할 것인지 명시적인 open-set mechanism이 없습니다.

저자도 이를 limitation으로 인정합니다. 

---

### 2. Expert를 online으로 새로 만들 수 있는가?

현재 공개 구조에서는 **아닙니다**.

$K_t,K_f$는 grid search 후보

$$
\{1,2,4,8\}
$$

중 선택합니다. 

따라서 continual learning 관점의 expert birth/merge/delete mechanism은 없습니다.

---

### 3. Top- $k$의 $k$는 어떻게 결정하는가?

Eq. (10)은

$$
G(s)=\text{Softmax}(\text{TopK}(s))
$$

를 명확하게 정의하지만, 공개 PDF의 방법 설명에서 **실제 활성 expert 수 $k$를 선택하는 원칙과 충분한 sensitivity analysis는 명확하지 않습니다.** 

---

### 4. Expert collapse는 발생하지 않는가?

MoE에서는 자주

$$
P(E_1)\gg P(E_2),P(E_3),\ldots
$$

처럼 한두 expert에 routing이 몰릴 수 있습니다.

Figure 10은 allocation distribution을 보여주지만, Switch Transformer류의 explicit load-balancing loss와 같은 장치가 TFPS 핵심 loss에는 없습니다.

즉 **expert utilization imbalance가 장기 학습에서 어느 정도 발생하는지**가 남은 질문입니다.

---

### 5. Wasserstein distance가 정말 concept drift를 측정하는가?

아닙니다. 최소한 Figure 1만으로는 그렇다고 할 수 없습니다.

Wasserstein은

$$
W(P_t(X),P_{t+1}(X))
$$

의 차이는 측정할 수 있지만,

$$
P_t(Y\mid X)
\neq
P_{t+1}(Y\mid X)
$$

인지를 직접 판별하지 않습니다.

따라서 향후에는 **covariate shift와 conditional/concept shift를 분리한 실험**이 필요합니다.

---

### 6. Cross-domain zero-shot generalization은 가능한가?

논문의 주요 forecasting benchmark는 dataset 내부의 chronological train/validation/test split입니다. ETT는 $6:2:2$, 기타는 $7:1:2$로 나누고 validation loss로 최적 hyperparameter를 선택합니다. 

따라서

> “A 데이터셋에서 학습한 TFPS를 B라는 새로운 공정/지역/장비에 그대로 적용했을 때도 잘 되는가?”

는 이 논문에서 입증되지 않았습니다.

---

# 8. 가장 중요한 그림 5개 해석

## Figure 1 — Patch-level distribution shift

p.2의 Figure 1은 이 논문 전체의 문제 정의입니다.

Sudden drift에서는 patch 9–10 부근이 다른 patch들과 크게 달라지고, gradual drift에서는 patch 0–5와 6–11 사이에 점진적인 변화가 관찰됩니다. 시간 영역과 frequency 영역의 heatmap도 완전히 같지 않아 두 표현이 complementary하다는 것이 저자 논리입니다. 

**제 해석:** TFPS가 필요한 이유를 매우 직관적으로 보여줍니다. 다만 이것은 **“분포가 다르다”는 증거이지 “예측 함수가 다르다”는 직접적인 증거는 아닙니다.**

---

## Figure 2 — 전체 TFPS 구조

p.4의 Figure 2가 가장 중요한 architecture figure입니다.

핵심 경로는

$$
\text{Patch}
\rightarrow
\begin{cases}
\text{Time Encoder}\rightarrow PI\rightarrow MoPE\\
\text{Frequency Encoder}\rightarrow PI\rightarrow MoPE
\end{cases}
\rightarrow
\text{Fusion}
$$

입니다. 

**제 해석:** 단순히 time-frequency feature를 concat하는 모델이 아니라, **두 domain 각각에서 별도로 pattern identification과 expert specialization을 수행**한다는 것이 중요합니다.

---

## Figure 3 — Pattern Identifier + MoPE

Figure 3은 TFPS의 실제 novelty를 가장 잘 설명합니다.

$$
z
\rightarrow
\mathbf D
\rightarrow
S
\rightarrow
\hat S
\rightarrow
\text{Top-K expert routing}
$$

순서입니다. 

**제 해석:** 일반 MoE의 router가 단순한 learned linear/MLP gate인 것과 달리, TFPS는 “이 patch가 어느 learned subspace에 속하는가?”라는 **geometrically structured router**를 사용합니다.

이 부분이 MoLE 등의 기존 MoE와 비교할 때 가장 중요한 차별점입니다.

---

## Figure 4 — Sudden/Gradual drift 예측

p.10에서 DLinear와 TFPS를 ETTh1 $H=192$에 비교합니다.

저자 시각화에서는 sudden drift와 gradual drift 모두에서 TFPS가 실제 trajectory를 DLinear보다 더 잘 따라갑니다. 

**제 해석:** “왜 평균 MSE가 개선됐는가?”를 이해하는 좋은 예시입니다. TFPS는 global linear trend 하나를 계속 연장하기보다 pattern별 expert를 사용하기 때문에 regime transition에서 다른 local function을 선택할 수 있습니다.

하지만 **선별된 몇 개의 시각화는 통계적 robustness의 증거는 아닙니다.**

---

## Figure 8 — PI가 실제로 clustering을 잘하는가?

개인적으로는 architecture figure 다음으로 중요한 검증 그림입니다.

PI를 Linear layer로 대체했을 때와 비교하여, PI에서는

$$
W_{\text{intra}}
\downarrow
$$

즉 같은 cluster 내부의 Wasserstein distance가 작아지고,

$$
W_{\text{inter}}
\uparrow
$$

즉 서로 다른 cluster 사이의 Wasserstein distance가 커집니다. 

**용어 설명 — intra-cluster:** 같은 cluster 내부 차이입니다. 작을수록 내부 구성원이 서로 비슷합니다.

**용어 설명 — inter-cluster:** 서로 다른 cluster 사이의 차이입니다. 클수록 cluster들이 잘 분리된 것입니다.

따라서

$$
\boxed{
W_{\text{intra}}\downarrow,\quad
W_{\text{inter}}\uparrow
}
$$

는 PI가 단순히 arbitrary routing을 한 것이 아니라 distributionally coherent한 그룹을 만들었다는 근거입니다.

다만 ETTh1 중심 분석이고 반복 실험의 uncertainty가 없으므로 **모든 데이터에서 안정적으로 이런 구조가 만들어지는가**는 아직 열려 있습니다.

---

# 9. 일반화 성능을 어떻게 평가해야 하는가?

이 부분이 이 논문을 읽을 때 가장 중요합니다.

## TFPS가 증명한 일반화

논문이 비교적 잘 증명한 것은

$$
\boxed{
\text{과거 training period}
\rightarrow
\text{같은 dataset의 미래 test period}
}
$$

에서 distribution drift가 있더라도 다양한 experts를 활용하면 prediction degradation을 줄일 수 있다는 것입니다.

특히 ETTm2, Weather 등 heterogeneity가 큰 데이터에서 상당한 개선이 나타나는 것은 이 주장과 일관적입니다.

---

## TFPS가 아직 증명하지 않은 일반화

하지만 다음은 별개의 문제입니다.

$$
\boxed{
\text{Training에 존재하지 않은 새로운 regime}
\rightarrow
\text{정확한 prediction}
}
$$

현재 TFPS는 이를 직접 해결하지 못합니다.

왜냐하면 새 patch도 결국 기존

$$
\{\mathcal S_1,\ldots,\mathcal S_K\}
$$

중 하나로 projection하고,

$$
\{E_1,\ldots,E_K\}
$$

중 기존 expert를 사용하기 때문입니다.

즉 이것은 **closed-set regime adaptation**에 가깝습니다.

**용어 설명 — Closed-set:** test에서 나타나는 pattern이 기본적으로 training에서 배운 pattern 공간 안에 있다고 가정합니다.

**용어 설명 — Open-set/OOD:** training에서 보지 못한 새로운 pattern이 들어오는 상황입니다.

---

# 10. 2020년 이후 관련 최신 연구 비교

아래 비교는 단순 연대 나열보다 **distribution shift에 어떻게 대응했는가**를 기준으로 정리했습니다.

| 연구                                                                                                    |   연도 | 핵심 전략                                                        | TFPS와의 차이                                                        | 일반화 관점                                                    |
| ----------------------------------------------------------------------------------------------------- | ---: | ------------------------------------------------------------ | ---------------------------------------------------------------- | --------------------------------------------------------- |
| **Reversible Instance Normalization for Accurate Time-Series Forecasting Against Distribution Shift** | 2022 | instance별 normalize→forecast→denormalize                     | shift를 제거/보정                                                     | 간단하고 model-agnostic, 복잡한 regime 분화는 못함                    |
| **Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting**                | 2022 | stationarization + de-stationary attention                   | 제거한 nonstationarity를 attention에 복원                               | normalization 과잉 문제를 인식                                   |
| **Dish-TS: A General Paradigm for Alleviating Distribution Shift in Time Series Forecasting**         | 2023 | input/output distribution 각각 추정                              | input↔horizon shift까지 보정                                         | inter-/intra-space shift 대응                               |
| **Adaptive Normalization for Non-stationary Time Series Forecasting: A Temporal Slice Perspective**   | 2023 | slice-level normalization                                    | TFPS의 patch 관점과 유사하지만 normalization 기반                           | local distribution 변화 대응                                  |
| **OneNet: Enhancing Time Series Forecasting Models under Concept Drift by Online Ensembling**         | 2023 | online ensemble weight adaptation                            | TFPS는 기본적으로 offline routing                                      | **진짜 streaming drift**에 더 직접적                             |
| **Mixture-of-Linear-Experts for Long-Term Time Series Forecasting**                                   | 2024 | 여러 linear experts + router                                   | TFPS와 가장 직접적인 MoE 전신 중 하나                                        | temporal pattern별 expert specialization                   |
| **Calibration of Time-Series Forecasting: Detecting and Adapting Context-Driven Distribution Shift**  | 2024 | Reconditionor + SOLID test-time adaptation                   | TFPS는 latent pattern routing, SOLID는 test-context 기반 fine-tuning | 새로운 context에 online adaptation 가능                         |
| **Pathformer: Multi-scale Transformers with Adaptive Pathways for Time Series Forecasting**           | 2024 | 여러 patch scale + adaptive pathway                            | TFPS의 fixed patch weakness를 직접 보완할 아이디어                          | transfer generalization 명시적 평가                            |
| **TSLANet: Rethinking Transformers for Time Series Representation Learning**                          | 2024 | adaptive spectral block                                      | TFPS frequency encoder와 연관                                       | noise·frequency robustness                                |
| **MOMENT: A Family of Open Time-series Foundation Models**                                            | 2024 | 대규모 multi-dataset pretraining                                | TFPS는 dataset-specific pattern specialization                    | few/limited supervision generalization                    |
| **Timer: Generative Pre-trained Transformers Are Large Time Series Models**                           | 2024 | 최대 약 10억 time points pretraining                             | TFPS와 목표 자체가 다름                                                  | broad/few-shot generalization                             |
| **DUET: Dual Clustering Enhanced Multivariate Time Series Forecasting**                               | 2025 | temporal clustering + channel soft clustering                | TFPS와 매우 가까운 heterogeneity modeling 흐름                           | temporal + channel heterogeneity                          |
| **Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts**                     | 2025 | sparse MoE foundation model                                  | TFPS: pattern semantics / Time-MoE: scaling·capacity             | broad zero/few-shot 방향                                    |
| **TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis**              | 2025 | multi-scale time + multi-resolution frequency pattern mixing | TFPS보다 scale 다양성을 더 직접적으로 모델링                                    | universal predictive analysis                             |
| **TFPS**                                                                                              | 2025 | dual-domain + subspace PI + patch-level MoPE                 | shift를 **분리된 predictive functions**로 직접 모델링                      | strong within-dataset future generalization, open-set 미해결 |

RevIN은 instance별 통계 제거와 복원을 통해 distribution shift를 완화하는 접근입니다. ([ML Anthology][2]) Non-stationary Transformer는 단순 stationarization이 중요한 temporal information까지 지우는 **over-stationarization**을 문제 삼고 de-stationary attention을 제안했습니다. ([NeurIPS Proceedings][3])

Dish-TS는 input window 내부의 **intra-space shift**뿐 아니라 look-back과 prediction horizon 사이의 **inter-space shift**를 명시적으로 모델링했습니다. ([AAAI Publications][4]) SAN은 더 나아가 하나의 instance 전체가 아니라 local temporal slice 단위로 normalization을 적용합니다. ([NeurIPS Papers][5])

**용어 설명 — Over-stationarization:** 데이터를 너무 강하게 정상화하여 실제 forecasting에 필요한 regime·trend·burst 정보까지 없애버리는 현상입니다.

이 흐름에서 TFPS의 철학은 다릅니다.

$$
\text{RevIN/SAN/Dish-TS:}
\quad
\text{shift를 보정}
$$

이라면

$$
\text{TFPS:}
\quad
\text{shift를 pattern identity로 활용}
$$

한다고 볼 수 있습니다.

---

# 10-1. MoE 계열과의 비교

2024년 **Mixture-of-Linear-Experts for Long-Term Time Series Forecasting**은 여러 linear-centric expert가 서로 다른 temporal pattern에 specialization하고 router가 이를 혼합한다는 아이디어를 제안했습니다. ([Proceedings of Machine Learning Research][6])

TFPS와 개념적으로 매우 가깝지만 차이는 router에 있습니다.

MoLE를 개념적으로

```math
\pi_k(X)
=
\text{Router}_k(X)
```

라고 한다면 TFPS는

$$
\pi_k(X)
\approx
\text{subspace-affinity}_k(z)
$$

를 사용합니다.

즉 TFPS는 **pattern assignment에 명시적인 clustering geometry를 부여했다는 점**이 차별적입니다.

---

# 10-2. Online drift 연구와의 비교

OneNet은 streaming data가 계속 들어올 때 두 forecasting 모델의 가중치를 온라인으로 변경합니다. ([NeurIPS Proceedings][7])

SOLID는 residual과 context 사이의 관계를 측정하는 **Reconditionor**로 context-driven shift를 감지하고, 필요한 test sample마다 contextually similar data를 이용해 prediction layer를 짧게 fine-tuning합니다. ([arXiv][8])

이들과 비교하면 TFPS는

```math
\text{TFPS}
=
\text{offline learned regime routing}
```

이고,

```math
\text{OneNet/SOLID}
=
\text{online/test-time adaptation}
```

에 더 가깝습니다.

따라서 “새로운 공정 조건이나 장비 상태가 계속 생기는 실제 운영환경”에서는 TFPS와 OneNet/SOLID식 adaptation을 결합하는 연구 가치가 높습니다.

---

# 10-3. Multi-scale 연구와의 비교

Pathformer는 여러 크기의 patch를 동시에 사용하고 input dynamics에 따라 pathway를 선택합니다. 특히 논문은 transfer scenario의 generalization도 평가합니다. ([ICLR Proceedings][9])

이는 TFPS 저자 스스로 인정한

> patch length가 heuristic이고 multi-period characteristics를 잘 처리하지 못한다

는 한계와 직접 연결됩니다. 

따라서 미래의 자연스러운 결합은

$$
P\in\{P_1,P_2,\ldots,P_M\}
$$

에 대해

$$
p(m,k\mid X)
$$

즉 **scale $m$과 expert $k$를 동시에 routing**하는 방식입니다.

---

# 10-4. 2025년 DUET과의 관계

**DUET: Dual Clustering Enhanced Multivariate Time Series Forecasting**은 heterogeneous temporal pattern을 temporal clustering으로 분리하면서, 동시에 channel 관계를 frequency-domain metric learning과 soft clustering으로 처리합니다. KDD 2025 공식 논문으로 확인됩니다. ([kdd.org][10])

TFPS와 매우 가까운 철학을 갖습니다.

TFPS:

$$
\boxed{
\text{Time/Frequency pattern}
\rightarrow
\text{Pattern Experts}
}
$$

DUET:

$$
\boxed{
\text{Temporal heterogeneity}
+
\text{Channel heterogeneity}
}
$$

따라서 앞으로는 **“patch pattern routing”만으로 충분한가, 아니면 channel/device/group 구조까지 동시에 routing해야 하는가**가 중요한 연구 문제가 됩니다.

특히 다변량 산업 공정에서는 후자가 중요할 가능성이 높습니다.

---

# 10-5. Foundation model과의 관계

MOMENT는 대규모 multi-dataset pretraining을 통한 general-purpose representation을 지향합니다. ([Proceedings of Machine Learning Research][11]) Timer는 최대 약 10억 time points 규모의 pretraining을 통해 few-shot/general-purpose time-series model을 추구합니다. ([Proceedings of Machine Learning Research][12])

2025년 Time-MoE는 sparse Mixture-of-Experts를 foundation-model scaling에 사용합니다. ([ICLR Proceedings][13])

여기에서 TFPS와 Time-MoE는 같은 “MoE”를 쓰지만 목적이 다릅니다.

### Time-MoE

$$
\text{MoE}
\Rightarrow
\text{large capacity with sparse computation}
$$

### TFPS

$$
\text{MoE}
\Rightarrow
\text{different pattern}
\mapsto
\text{different prediction function}
$$

입니다.

이 둘을 결합한다면 매우 흥미로운 연구가 가능합니다.

$$
\boxed{
\text{Pretrained universal representation}
+
\text{local pattern-specific adapters/experts}
}
$$

이 구조가 TFPS의 일반화 한계를 해결할 가장 유망한 방향 중 하나라고 봅니다.

---

# 11. TFPS가 앞으로 연구에 미치는 영향

제가 보기에 이 논문의 가장 중요한 메시지는 다음 관점 변화입니다.

기존 비정상 시계열 연구는 대체로

$$
X
\xrightarrow{\text{normalize}}
\tilde X
\xrightarrow{f}
\hat Y
$$

처럼 distribution variation을 **제거해야 할 nuisance**로 보았습니다.

TFPS는

$$
X
\xrightarrow{\text{identify pattern}}
Z
\xrightarrow{\text{specialized expert}}
f_Z(X)
$$

로 바꿉니다.

즉

> **“데이터가 서로 다르기 때문에 예측이 어렵다”**

를

> **“데이터가 서로 다르므로 서로 다른 함수를 사용해야 한다”**

로 바꾼 것입니다.

이것은 mixture-of-regimes, local experts, conditional computation, continual learning과 시계열 forecasting을 연결할 수 있는 중요한 방향입니다.

---

# 12. 일반화 성능을 더 끌어올리기 위한 후속 연구

## 12-1. Open-set Pattern Identifier

현재는

$$
\sum_{k=1}^{K}s_k=1
$$

이므로 새로운 patch도 반드시 기존 expert에 배정됩니다.

향후에는

$$
\max_k s_k<\tau
$$

또는

$$
d(z,\mathcal S_k)>\tau
\quad\forall k
$$

이면

$$
Z=\text{unknown}
$$

으로 판단하는 OOD detector가 필요합니다.

그 후

$$
E_{K+1}
$$

을 동적으로 생성할 수 있어야 합니다.

이것이 **신규 공정 regime·신규 질병 유행·새로운 시장 구조**처럼 training에 없던 pattern에 대한 일반화를 크게 개선할 수 있습니다.

---

# 12-2. Expert Birth–Merge–Delete

고정 $K$ 대신

$$
K=K(t)
$$

로 만들 수 있습니다.

* 새로운 cluster 등장 → **expert birth**
* 두 expert의 representation이 수렴 → **merge**
* 장기간 사용되지 않는 expert → **delete**

방식입니다.

이렇게 해야 TFPS가 단순 offline architecture에서 **continual forecasting model**로 발전합니다.

---

# 12-3. Adaptive Multi-scale Patching

고정 $P$ 대신

```math
\mathcal P
=
\{P_1,P_2,\ldots,P_M\}
```

을 두고

$$
p(P_m\mid X)
$$

를 학습하는 것이 필요합니다.

Pathformer가 보여준 multi-scale adaptive pathway와 TFPS의 pattern router를 결합하면 저자가 직접 명시한 fixed-patch limitation을 해결할 수 있습니다. ([ICLR Proceedings][9])

---

# 12-4. Expert 수를 drift severity에서 자동 추론

현재

$$
K_t,K_f\in\{1,2,4,8\}
$$

을 grid search합니다.

더 논리적인 방법은 distribution heterogeneity를

```math
\mathcal D
=
\mathbb E_{i,j}
W(P_i,P_j)
```

등으로 측정하고

```math
K^*
=
g(\mathcal D,\;n,\;\text{complexity})
```

를 학습하는 것입니다.

다만 Wasserstein은 scale-dependent할 수 있으므로 먼저

```math
\tilde W
=
\frac{
W(P_i,P_j)
}{
\sigma_X+\epsilon
}
```

같은 standardized drift score를 사용하는 편이 더 합리적입니다.

---

# 12-5. Expert diversity와 load balancing

MoE의 중요한 위험은 expert collapse입니다.

예를 들어 routing probability를

```math
p_k
=
\frac{1}{N}
\sum_{i=1}^{N}
G_k(s_i)
```

라 할 때 entropy

```math
H(p)
=
-\sum_{k=1}^{K}p_k\log p_k
```

가 지나치게 작으면 일부 expert만 계속 사용되고 있다는 의미입니다.

따라서

```math
L
=
L_{\text{TFPS}}
+
\lambda_{\text{bal}}L_{\text{balance}}
+
\lambda_{\text{div}}L_{\text{diversity}}
```

같은 regularization을 추가해보는 것이 타당합니다.

---

# 12-6. Generalization 실험 프로토콜 자체를 강화해야 함

현재의 한 번의 chronological split보다 다음과 같은 rolling-origin evaluation이 더 중요합니다.

$$
\begin{aligned}
&\text{Train}_{1:t_1}\rightarrow \text{Test}_{t_1+1:t_2}\\
&\text{Train}_{1:t_2}\rightarrow \text{Test}_{t_2+1:t_3}\\
&\cdots
\end{aligned}
$$

각 drift 단계별로

$$
\Delta R^2,\quad
\Delta\text{MSE},\quad
\Delta\text{MAE}
$$

를 기록하면

> “shift가 커질수록 TFPS의 relative advantage가 커진다”

는 핵심 가설을 실제 통계적으로 검증할 수 있습니다.

---

# 12-7. 반드시 추가해야 할 통계 검증

각 모델을 여러 seed로 $R$회 반복하여

```math
\bar e
=
\frac1R
\sum_{r=1}^{R}e_r
```

와

```math
s_e
=
\sqrt{
\frac{
\sum_{r=1}^{R}(e_r-\bar e)^2
}{
R-1
}
}
```

를 보고해야 합니다.

그리고 최소한

```math
\text{MSE}
=
\bar e\pm s_e
```

형태의 분산 또는 confidence interval을 제시해야 합니다.

forecasting model 간 paired comparison에서는 sample별 forecast error가 대응되어 있으므로 block bootstrap이나 Diebold-Mariano 계열 검정을 병행하면 훨씬 강한 근거가 됩니다.

그래야

$$
0.398\quad\text{vs.}\quad0.401
$$

같은 차이가 실질적 improvement인지 noise인지 구분할 수 있습니다.

---

# 13. 논문 자체의 한계 — 저자가 직접 인정한 내용

저자들은 Appendix N에서 두 가지를 명확하게 인정합니다.

첫째, patch length가 heuristic하게 선택되며 indivisible length나 multi-period characteristics를 다루기 어렵습니다.

둘째, 현실에서는 시간이 지남에 따라 새로운 pattern이 계속 나타나므로 기존 모델만으로는 충분하지 않고, 향후 자동 patch length selection과 evolving distribution shift에 대응하는 extensible architecture를 연구하겠다고 명시합니다. 

따라서 논문 스스로도 TFPS를 **완성된 universal distribution-shift solution**이라고 보지는 않습니다.

---

# 14. 추가로 발견되는 문서 내부 불일치

세부 검증 과정에서 몇 가지 reporting inconsistency가 있습니다.

**첫째**, Section 4.1 본문에서는 ETT 4종, Exchange, Weather, Electricity, Traffic, ILI로 총 **9개 dataset**을 명시합니다.  반면 Appendix A와 conclusion 일부에서는 “eight datasets”라고 표현합니다. 실제 나열된 데이터는 9개이므로 문서 편집상의 불일치로 보입니다.

**둘째**, main Figure 6 caption은 ETTh1과 ETTh2로 읽히지만, 바로 아래 설명은 ETTh1과 **Weather** 실험이라고 기술하며, Appendix Figure 7 역시 ETTh1과 Weather를 사용합니다. 

**셋째**, 앞서 설명한 것처럼 Table 12와 efficiency-analysis 본문의 inference time 수치가 일치하지 않습니다.

이들은 forecasting 핵심 결과를 무효화할 정도의 오류라고 보기는 어렵지만, **정밀 재현 연구에서는 반드시 원 코드로 확인해야 하는 부분**입니다.

---

# 15. 최종 결론

## 저자들이 제시하는 시사점

저자들의 핵심 결론은 **non-stationary time series의 서로 다른 patch를 하나의 uniform model로 처리하지 말고, latent pattern을 찾아 각 pattern에 특화된 expert를 사용하자**는 것입니다. 실제 benchmark에서 DDE + PI + MoPE 전체 구조가 강한 성능을 보이고, cluster 시각화와 ablation도 이 설계를 지지합니다. 특히 강한 distribution shift가 존재하는 데이터에서 상대적인 장점이 커진다는 결과는 연구 방향 자체의 타당성을 뒷받침합니다. ([NeurIPS Proceedings][1])

## 제가 보는 가장 중요한 결론

다만 **TFPS가 개선한 “generalization”을 정확히 정의해야 합니다.**

현재 논문이 상당히 잘 보여준 것은

$$
\boxed{
\text{known heterogeneous patterns 안에서의 미래구간 일반화}
}
$$

입니다.

반면 아직 충분히 보여주지 못한 것은

$$
\boxed{
\text{training에 없었던 새로운 pattern에 대한 일반화}
}
$$

입니다.

즉 TFPS는 강력한 **pattern-aware forecasting model**이지만 아직 완전한 **continual/open-world forecasting model**은 아닙니다.

---

## 앞으로 가장 유망한 발전 방향

제가 이 논문의 후속 연구를 설계한다면 다음 조합을 우선합니다.

$$
\boxed{
\begin{aligned}
&\text{Foundation/pretrained representation}\\
+&\text{multi-scale adaptive patching}\\
+&\text{subspace Pattern Identifier}\\
+&\text{pattern-specific sparse experts}\\
+&\text{OOD detector}\\
+&\text{dynamic expert birth/merge}\\
+&\text{online adaptation}
\end{aligned}}
$$

즉 **TFPS + Pathformer식 multi-scale + Time-MoE/foundation representation + OneNet/SOLID식 continual adaptation**입니다.

이 방향으로 발전시키면 TFPS의 현재 가장 큰 약점인

$$
\text{fixed patch}
+
\text{fixed experts}
+
\text{closed-set pattern assumption}
$$

을 동시에 완화할 수 있습니다.

특히 산업 공정, 장비 열화, 센서 drift처럼 **미래 데이터의 분포가 training과 달라지는 것이 정상인 분야**에서는 단순히 더 큰 backbone을 만드는 것보다 이러한 **dynamic pattern-specific generalization**이 훨씬 중요한 연구 문제가 될 가능성이 높습니다.

---

# 참고 자료 및 웹 검색 출처

이번 분석에서 직접 확인한 주요 자료는 다음과 같습니다.

1. **Learning Pattern-Specific Experts for Time Series Forecasting Under Patch-level Distribution Shift**, Sun et al., NeurIPS 2025 — 첨부된 arXiv v2 원문 및 NeurIPS 공식 Proceedings. ([NeurIPS Proceedings][1])
2. **Reversible Instance Normalization for Accurate Time-Series Forecasting Against Distribution Shift**, Kim et al., ICLR 2022. ([ML Anthology][2])
3. **Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting**, Liu et al., NeurIPS 2022. ([NeurIPS Proceedings][3])
4. **Dish-TS: A General Paradigm for Alleviating Distribution Shift in Time Series Forecasting**, Fan et al., AAAI 2023. ([AAAI Publications][4])
5. **Adaptive Normalization for Non-stationary Time Series Forecasting: A Temporal Slice Perspective**, Liu et al., NeurIPS 2023. ([NeurIPS Papers][5])
6. **OneNet: Enhancing Time Series Forecasting Models under Concept Drift by Online Ensembling**, Zhang et al., NeurIPS 2023. ([NeurIPS Proceedings][7])
7. **Mixture-of-Linear-Experts for Long-Term Time Series Forecasting**, Ni et al., AISTATS 2024. ([Proceedings of Machine Learning Research][6])
8. **Calibration of Time-Series Forecasting: Detecting and Adapting Context-Driven Distribution Shift**, Chen et al., KDD 2024. ([kdd.org][14])
9. **Pathformer: Multi-scale Transformers with Adaptive Pathways for Time Series Forecasting**, Chen et al., ICLR 2024. ([ICLR Proceedings][9])
10. **TSLANet: Rethinking Transformers for Time Series Representation Learning**, Eldele et al., ICML 2024. ([ML Anthology][15])
11. **iTransformer: Inverted Transformers Are Effective for Time Series Forecasting**, Liu et al., ICLR 2024. ([ICLR Proceedings][16])
12. **MOMENT: A Family of Open Time-series Foundation Models**, Goswami et al., ICML 2024. ([Proceedings of Machine Learning Research][11])
13. **Timer: Generative Pre-trained Transformers Are Large Time Series Models**, Liu et al., ICML 2024. ([Proceedings of Machine Learning Research][12])
14. **DUET: Dual Clustering Enhanced Multivariate Time Series Forecasting**, Qiu et al., KDD 2025. ([kdd.org][10])
15. **Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts**, Shi et al., ICLR 2025. ([ICLR Proceedings][13])
16. **TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis**, Wang et al., ICLR 2025. ([ICLR Proceedings][17])

다음 단계로는 이 분석을 기반으로 **“TFPS를 실제 다변량 소표본 공정 시계열에 적용할 때 누수 없이 구현하는 파이프라인”**을 설계하면서, 원 논문의 PI·MoPE 수식을 실제 tensor shape까지 대입해 구현 수준으로 내려가 보는 것이 가장 유용합니다.

[1]: https://proceedings.neurips.cc/paper_files/paper/2025/hash/8491a7fcc218946b471b600a915c8b02-Abstract-Conference.html "https://proceedings.neurips.cc/paper_files/paper/2025/hash/8491a7fcc218946b471b600a915c8b02-Abstract-Conference.html"
[2]: https://mlanthology.org/iclr/2022/kim2022iclr-reversible/ "https://mlanthology.org/iclr/2022/kim2022iclr-reversible/"
[3]: https://proceedings.neurips.cc/paper_files/paper/2022/hash/4054556fcaa934b0bf76da52cf4f92cb-Abstract-Conference.html "https://proceedings.neurips.cc/paper_files/paper/2022/hash/4054556fcaa934b0bf76da52cf4f92cb-Abstract-Conference.html"
[4]: https://ojs.aaai.org/index.php/AAAI/article/view/25914 "https://ojs.aaai.org/index.php/AAAI/article/view/25914"
[5]: https://papers.nips.cc/paper_files/paper/2023/hash/2e19dab94882bc95ed094c4399cfda02-Abstract-Conference.html "https://papers.nips.cc/paper_files/paper/2023/hash/2e19dab94882bc95ed094c4399cfda02-Abstract-Conference.html"
[6]: https://proceedings.mlr.press/v238/ni24a.html "https://proceedings.mlr.press/v238/ni24a.html"
[7]: https://proceedings.neurips.cc/paper_files/paper/2023/hash/dd6a47bc0aad6f34aa5e77706d90cdc4-Abstract-Conference.html "https://proceedings.neurips.cc/paper_files/paper/2023/hash/dd6a47bc0aad6f34aa5e77706d90cdc4-Abstract-Conference.html"
[8]: https://arxiv.org/abs/2310.14838 "https://arxiv.org/abs/2310.14838"
[9]: https://proceedings.iclr.cc/paper_files/paper/2024/hash/2be6705de7412adf107900add727a795-Abstract-Conference.html "https://proceedings.iclr.cc/paper_files/paper/2024/hash/2be6705de7412adf107900add727a795-Abstract-Conference.html"
[10]: https://www.kdd.org/kdd2025/research-track-papers-2/ "https://www.kdd.org/kdd2025/research-track-papers-2/"
[11]: https://proceedings.mlr.press/v235/goswami24a.html "https://proceedings.mlr.press/v235/goswami24a.html"
[12]: https://proceedings.mlr.press/v235/liu24cb.html "https://proceedings.mlr.press/v235/liu24cb.html"
[13]: https://proceedings.iclr.cc/paper_files/paper/2025/hash/558d48c1f08675daa636e09bfe94a89e-Abstract-Conference.html "https://proceedings.iclr.cc/paper_files/paper/2025/hash/558d48c1f08675daa636e09bfe94a89e-Abstract-Conference.html"
[14]: https://www.kdd.org/kdd2024/research-track-papers/ "https://www.kdd.org/kdd2024/research-track-papers/"
[15]: https://mlanthology.org/icml/2024/eldele2024icml-tslanet/ "https://mlanthology.org/icml/2024/eldele2024icml-tslanet/"
[16]: https://proceedings.iclr.cc/paper_files/paper/2024/hash/2ea18fdc667e0ef2ad82b2b4d65147ad-Abstract-Conference.html "https://proceedings.iclr.cc/paper_files/paper/2024/hash/2ea18fdc667e0ef2ad82b2b4d65147ad-Abstract-Conference.html"
[17]: https://proceedings.iclr.cc/paper_files/paper/2025/hash/2b187165e28fdfdc0ffb34d1bfff2b0c-Abstract-Conference.html "https://proceedings.iclr.cc/paper_files/paper/2025/hash/2b187165e28fdfdc0ffb34d1bfff2b0c-Abstract-Conference.html"
