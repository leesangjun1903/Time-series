# LiNo: Advancing Recursive Residual Decomposition of Linear and Nonlinear Patterns for Robust Time Series Forecasting

**대상 논문:** *LiNo: Advancing Recursive Residual Decomposition of Linear and Nonlinear Patterns for Robust Time Series Forecasting* — Guoqi Yu, Yaoming Li, Xiaoyu Guo, Dayu Wang, Zirui Liu, Shujun Wang, Tong Yang. 첨부 문서는 **arXiv:2410.17159v4, 2025-02-25 버전**이며, arXiv 최초 공개일은 2024-10-22입니다. 이 답변은 **첨부된 v4 원문을 1차 근거**로 삼고, 최신 연구 비교 부분만 별도의 웹 검색으로 보완했습니다.  ([arXiv][1])

중요한 점부터 밝히면, 논문의 성능 수치는 상당히 강하지만 **“linear/nonlinear component가 통계적으로 독립적으로 분리되었다”, “residual이 0으로 수렴한다”, “generalization ability가 증명되었다”까지 실험이 입증한 것은 아닙니다.** 이 세 부분은 저자의 주장과 실제 실험적 증거를 구분해서 읽어야 합니다.

---

## 1. Executive Summary — 10문장 이내

1. LiNo는 실제 시계열을 단순한 **trend + seasonal** 두 부분이 아니라 여러 단계의 **linear pattern + nonlinear pattern + noise**가 중첩된 신호로 보고, 이를 반복적으로 분리하는 **Recursive Residual Decomposition(RRD)**을 제안합니다. 
2. 각 단계에서 **Li block**이 선형 패턴을 먼저 추출하고, 남은 residual에서 **No block**이 nonlinear pattern을 추출한 뒤 다시 residual을 다음 단계로 넘깁니다. 
3. Li block은 기존 moving average나 고정 convolution 대신 **full receptive field를 갖는 learnable autoregressive 계열의 linear extractor**를 사용합니다. 
4. No block은 **time-domain variation, frequency-domain information, inter-series/channel dependency**를 함께 모델링하도록 설계되어 있습니다.  
5. 각 단계가 별도의 linear/nonlinear forecast를 만들고 이를 모두 합산하기 때문에, 최종 예측은 여러 decomposition level의 예측을 결합하는 구조입니다. 
6. 저자는 multivariate benchmark 10종에서 MSE 기준 9/10 데이터셋 1위, 기존 iTransformer 대비 전체적으로 약 3.41% MSE 감소를 보고합니다. 
7. univariate forecasting에서는 6개 데이터셋 모두 최상위 결과를 보고하며 MICN 대비 평균 MSE 19.37%, MAE 10.28% 감소를 보고합니다. 
8. ablation에서는 Li block, No block, channel mixing을 제거할 때 성능이 악화되고, Gaussian-noise 실험에서도 Raw/Mu 설계보다 LiNo가 안정적으로 낮은 오류를 보입니다.  
9. 그러나 baseline별 반복실험 분산과 통계적 유의성 검정이 부족하고, 저자가 주장하는 **component independence와 residual convergence는 수학적으로 보장되지 않았다는 점**이 가장 중요한 이론적 약점입니다.
10. 따라서 LiNo의 핵심 가치는 “특정 SOTA 숫자”보다는 **서로 다른 함수군이 residual을 반복적으로 나누어 담당하도록 하는 구조적 inductive bias**에 있으며, 향후 distribution shift, adaptive decomposition depth, probabilistic forecasting, foundation-model pretraining을 결합하면 일반화 성능을 더 체계적으로 연구할 가치가 큽니다.

**용어 설명 — Inductive bias:** 모델이 데이터를 보기 전부터 구조적으로 가지는 가정입니다. LiNo에서는 “시계열에는 분리 가능한 linear/nonlinear pattern이 반복적으로 존재한다”는 가정 자체가 inductive bias입니다.

---

# 1-1. 연구 목적과 왜 필요한가

기존 seasonal-trend decomposition(STD)은 일반적으로

$$
X = \text{Trend} + \text{Seasonal}
$$

과 같이 생각하고 moving average(MOV), exponential smoothing(ESF), learnable convolution(LD) 등으로 trend를 한 번 추출한 뒤 나머지를 seasonal residual로 취급합니다. 저자는 이 방식에 세 가지 근본적 문제가 있다고 봅니다.  

첫째, **linear = trend**가 아닙니다. 실제 시계열에는 trend뿐 아니라 autoregressive dependence, cyclic linear component 등 다양한 선형 구조가 존재하기 때문에 fixed-window moving average 하나로 모든 선형 구조를 뽑아내기 어렵습니다.

**용어 설명 — Autoregressive dependence:** 현재 값이 이전 값들의 선형 조합과 관계가 있다는 의미입니다. 예를 들어 $x_t\approx0.8x_{t-1}-0.2x_{t-2}$와 같은 관계입니다.

둘째, trend를 빼고 남은 residual을 곧바로 seasonal/nonlinear component라고 부르는 것은 지나치게 단순합니다. 실제로는

```math
\text{Residual}
=
\text{미추출 linear}
+
\text{nonlinear}
+
\text{noise}
```

일 수 있기 때문입니다.

셋째, 실세계 시계열은 여러 시간 규모와 여러 주기의 패턴이 중첩되어 있으므로 **한 번의 decomposition보다 반복적이고 계층적인 decomposition**이 더 자연스럽다는 것이 저자들의 문제의식입니다. Figure 1이 바로 이 주장을 시각화합니다. 

따라서 LiNo가 해결하려는 문제는 단순히 “더 강한 Transformer를 만들자”가 아니라,

> **하나의 복잡한 함수가 모든 구조를 동시에 학습하도록 하지 말고, linear와 nonlinear pattern을 서로 다른 함수군에 배정한 뒤 residual을 반복적으로 다시 분해하면 더 안정적이고 일반적인 forecasting representation을 얻을 수 있는가?**

라고 정리할 수 있습니다.

---

# 2. 핵심 주장과 근거

| 핵심 주장                                              | 저자가 제시한 근거                                 | 위치                              | 검토 결과                                                          |
| -------------------------------------------------- | ------------------------------------------ | ------------------------------- | -------------------------------------------------------------- |
| 기존 trend-seasonal 분해는 지나치게 얕다                      | 실제 신호에 여러 linear/nonlinear level이 존재한다고 주장 | p.1–2, **Figure 1**             | **개념적으로 타당**, 그러나 Figure 1의 decomposition은 ground truth 분해가 아님 |
| Linear와 nonlinear을 명시적으로 반복 분리해야 한다                | RRD 식 (1), Raw/Mu/LiNo 비교                  | p.3, Eq.(1); p.8 **Table 6(b)** | 실험은 지지하지만 “독립성 보장”까지는 증명하지 못함                                  |
| Learnable AR Li block이 기존 linear extractor보다 일반적이다 | full receptive field AR formulation        | p.4, Eq.(2),(3), **Figure 2**   | 합리적이나 latent embedding 이후의 연산이므로 전통적 raw-time AR와 완전히 같지는 않음   |
| No block이 time/frequency/channel 정보를 함께 추출한다       | TF extraction + channel mixing             | p.4–5, **Figure 2**             | ablation이 각 요소의 기여를 대체로 지지                                     |
| 깊은 RRD가 필요하다                                       | $N=1,2,3,4$ 비교                             | p.7, **Table 4**                | $N>1$이 대체로 유리하지만 **더 깊을수록 계속 좋아지는 것은 아님**                      |
| No block이 중요하다                                     | 제거 시 평균 MSE 악화                             | p.7, **Table 3**                | 강한 ablation evidence                                           |
| Multivariate SOTA                                  | MSE 9/10 benchmark 1위                      | p.5–6, **Table 1**              | 강한 benchmark 결과이나 일부 baseline은 논문에서 가져온 숫자                     |
| Univariate SOTA                                    | 6/6 최상위                                    | p.6, **Table 2**                | 강한 결과이나 역시 external baseline results 사용                        |
| Noise robustness가 개선된다                             | Gaussian perturbation에서 LiNo < Mu < Raw 경향 | p.8, **Figure 3**               | robustness evidence는 있으나 OOD generalization과 동일하지 않음           |
| 더 긴 lookback을 잘 이용한다                               | $T=48\sim720$                              | p.15, **Figure 4**              | context utilization에는 긍정적                                      |
| seed에 안정적이다                                        | 5 random seeds의 mean±std                   | p.14, **Table 8**               | training stability 증거이지 cross-domain generalization 증거는 아님     |
| 계산 효율도 양호하다                                        | RTX4090 inference time                     | p.15, **Table 9**               | iTransformer보다 parameter는 조금 많지만 inference는 빠름                 |

---

# 2-1. 문제, 수식, 모델 구조, 성능, 한계 상세 분석

## 2-1-1. Forecasting 문제 정의

논문은 multivariate time series를

$$
X\in\mathbb{R}^{C\times T}
$$

로 정의하고 미래

$$
\hat{Y}\in\mathbb{R}^{C\times F}
$$

를 예측합니다. 

여기서:

* $C$: channel 또는 variate의 개수입니다.
* $T$: 과거 관측 구간, 즉 **look-back window** 길이입니다.
* $F$: 미래 prediction horizon 길이입니다.
* $X$: 모델에 입력되는 과거 시계열입니다.
* $Y$: 실제 미래값입니다.
* $\hat Y$: 모델이 예측한 미래값입니다.

**용어 설명 — Look-back window:** 미래를 예측하기 위해 모델에게 보여 주는 과거 구간입니다. $T=96$이면 직전 96개 시점을 이용합니다.

---

## 2-1-2. LiNo가 가정하는 시계열 생성 구조

저자는 설명을 위해 univariate signal을

```math
X
=
L_1+N_1+L_2+N_2+\cdots+L_S+N_S+\epsilon
```

이라고 가정합니다. 

여기서:

* $L_i$: $i$번째 linear pattern입니다.
* $N_i$: $i$번째 nonlinear pattern입니다.
* $S$: 존재한다고 가정한 decomposition level 수입니다.
* $\epsilon$: white noise입니다.
* $X$: 관측된 원래 시계열입니다.

중요한 것은 이 식이 **관측 데이터에서 유일하게 증명된 decomposition이 아니라 모델링 가정**이라는 점입니다.

**용어 설명 — Identifiability(식별 가능성):** $X=L+N$이라고 했을 때 데이터만 보고 $L$과 $N$을 유일하게 결정할 수 있는지를 뜻합니다. 별도 제약이 없다면 하나의 신호를 여러 방식으로 $L+N$으로 나눌 수 있으므로 decomposition은 일반적으로 유일하지 않습니다.

---

## 2-1-3. Recursive Residual Decomposition

논문의 Eq.(1)은 다음과 같습니다.

$$
R_1^L=X-\hat L_1,
$$

$$
R_1^N=R_1^L-\hat N_1,
$$

그리고 다음 단계에서는

$$
R_i^L=R_{i-1}^N-\hat L_i,
$$

$$
R_i^N=R_i^L-\hat N_i.
$$

즉 한 level의 흐름은

$$
\boxed{
\text{Input}
\rightarrow
\text{Li extraction}
\rightarrow
R_i^L
\rightarrow
\text{No extraction}
\rightarrow
R_i^N
}
$$

이고, $R_i^N$이 다시 다음 level의 입력이 됩니다. 

여기서:

* $\hat L_i$: Li block이 추출한 $i$번째 선형 성분입니다.
* $\hat N_i$: No block이 추출한 $i$번째 비선형 성분입니다.
* $R_i^L$: linear component를 제거한 뒤의 residual입니다.
* $R_i^N$: 이어서 nonlinear component까지 제거한 residual입니다.
* $i$: decomposition level입니다.

**용어 설명 — Residual:** 현재 단계 모델이 아직 설명하지 못한 나머지입니다. “오차”와 비슷하지만, LiNo에서는 다음 block이 추가적으로 학습할 유용한 정보도 residual에 포함될 수 있습니다.

저자는 또한

$$
\lim_{i\rightarrow\infty}R_i^N=0
$$

을 제시하고, 충분히 깊게 decomposition하면 모든 유용한 정보가 추출될 수 있다고 설명합니다. 

### 여기에는 중요한 이론적 주의점이 있습니다

이 논문에는

$$
\lim_{i\rightarrow\infty}R_i^N=0
$$

가 성립하는 조건에 대한 convergence theorem이나 proof가 제시되어 있지 않습니다.

실제로 실험에서 사용한 depth도 무한대가 아니라

$$
N\in\{1,2,3,4\}
$$

입니다. 

따라서 이 식은 **LiNo의 동기를 설명하기 위한 이상적 조건에 가깝고, 학습된 finite neural network의 보장된 성질이라고 읽으면 안 됩니다.**

또한 저자는 residual subtraction이 linear/nonlinear pattern의 “independence”를 보장한다고 서술합니다. 

그러나 일반적으로

$$
R=X-\hat L
$$

이라는 subtraction만으로

$$
L\perp N
$$

또는

$$
p(L,N)=p(L)p(N)
$$

이라는 **statistical independence**가 성립하지 않습니다.

심지어

$$
\text{Cov}(L,N)=0
$$

이라는 decorrelation조차 별도의 constraint 없이 보장되지 않습니다.

**용어 설명 — Statistical independence:** 한 성분을 알아도 다른 성분의 확률분포가 변하지 않는 강한 조건입니다. 단순히 두 신호를 뺀다고 자동으로 성립하지 않습니다.

이 부분은 이 논문에서 가장 분명하게 **“저자의 해석이 수학적 증명보다 앞서 있는 부분”**입니다.

---

# 2-1-4. Whole-series embedding

Figure 2에서 원시 입력은 먼저

$$
X\in\mathbb R^{C\times T}
\longrightarrow
X_{\text{embed}}\in\mathbb R^{C\times D}
$$

로 linear projection됩니다. 저자는 iTransformer와 유사하게 **각 variate의 전체 history를 하나의 representation으로 만드는 설계**라고 설명합니다. 

Appendix에서는 이를 개념적으로

$$
H_0=X_{\text{embed}}=XW+b
$$

라고 씁니다. 

여기서:

* $D$: latent/embedding dimension입니다.
* $H_i$: $i$번째 LiNo level의 입력 representation입니다.
* $W,b$: embedding projection의 학습 파라미터입니다.

**용어 설명 — Embedding:** 원래 데이터 자체 대신 모델이 처리하기 좋은 새로운 좌표 공간으로 데이터를 변환하는 것입니다.

---

# 2-1-5. Li block — Linear pattern extractor

Li block은 다음과 같이 표현됩니다.

```math
\hat L_i[c,d]
=
\phi_i[c,1]H_i[c,1]
+\phi_i[c,2]H_i[c,2]
+\cdots
+\phi_i[c,d]H_i[c,d]
+\beta_i[c],
```

$$
L_i=\text{Dropout}(\hat L_i).
$$

논문의 Eq.(2)입니다. 

여기서:

* $c$: channel index입니다.
* $d$: 현재 latent position입니다.
* $H_i[c,k]$: $i$번째 level에서 channel $c$의 $k$번째 입력 representation입니다.
* $\phi_i[c,k]$: 해당 위치의 learnable AR coefficient입니다.
* $\beta_i[c]$: channel별 bias입니다.
* $\hat L_i$: dropout 이전의 추출된 linear representation입니다.
* $L_i$: dropout 이후 representation입니다.

padding은 Eq.(3)으로

```math
\hat H_i[:,t]
=
\begin{cases}
H_i[:,t-D], & t\ge D,\\
0, & t < D
\end{cases}
```

로 표현됩니다. 

저자는 이를 convolution 방식으로 구현하여 **full receptive field**를 갖는 learnable AR extractor로 설명합니다.

**용어 설명 — Receptive field:** 하나의 출력값을 계산할 때 참조할 수 있는 입력 영역입니다. full receptive field라면 매우 제한된 최근 window가 아니라 이용 가능한 전체 범위를 볼 수 있다는 뜻입니다.

### 중요한 해석상의 주의

여기서 Li block이 동작하는 것은 raw $X_t$ 자체가 아니라 이미

$$
X\rightarrow X_{\text{embed}}
$$

를 거친 representation입니다.

그러므로 고전적인

$$
x_t=\sum_{k=1}^p\phi_kx_{t-k}+\epsilon_t
$$

AR 모델과 완전히 동일하다고 보기보다는,

> **latent representation에 적용되는 learnable AR-like linear/convolutional extractor**

라고 부르는 것이 더 엄밀합니다.

---

# 2-1-6. No block — Nonlinear pattern extractor

No block은 세 종류의 정보를 동시에 모델링하려고 합니다.

### A. Temporal variation

잔차

$$
R_i^L=H_i-L_i
$$

를 time-domain linear projection에 넣어

$$
N_i^T
$$

를 생성합니다. 

### B. Frequency information

동일한 residual을 FFT로 frequency domain에 보낸 뒤 projection하고 IFFT로 다시 time-domain으로 반환하여

$$
N_i^F
$$

를 얻습니다.

이를 결합해 논문은

```math
N_i^{TF}
=
\tanh\left(N_i^T+N_i^F\right)
```

를 사용합니다. 

여기서:

* $N_i^T$: temporal variation representation입니다.
* $N_i^F$: frequency representation입니다.
* $N_i^{TF}$: time-frequency fused representation입니다.
* $\tanh$: 비선형 activation입니다.

**용어 설명 — FFT:** 시계열을 “시간에 따라 값이 어떻게 변하는가”에서 “어떤 주파수 성분이 얼마나 존재하는가”로 바꾸는 변환입니다.

**용어 설명 — IFFT:** FFT의 역변환으로 frequency representation을 다시 time-domain으로 돌립니다.

**용어 설명 — Tanh:** 입력을 $(-1,1)$ 사이의 비선형 값으로 변환하는 함수이며 양수와 음수를 대칭적으로 표현할 수 있습니다.

Table 5에서 Tanh 버전이 ReLU보다 대체로 우수했고 저자는 smooth, symmetric nonlinearity가 time/frequency 결합에 적합하다고 해석합니다. 

---

## C. Inter-series dependency / Channel Mixing

논문은 $N_i^{TF}$에 channel-direction softmax를 적용하고 weighted global representation을 만든 뒤, 이를 각 channel representation과 concatenate하고 MLP에 넣습니다. 

논문의 서술을 **수학적으로 재표현**하면 다음과 같이 이해할 수 있습니다. 아래 식은 저자의 번호식이 아니라 본문의 algorithm description을 수식화한 것입니다.

```math
a_{i,c,d}
=
\frac{
\exp(N_{i,c,d}^{TF})
}{
\sum_{c'=1}^{C}\exp(N_{i,c',d}^{TF})
},
```

```math
g_{i,d}
=
\sum_{c=1}^{C}
a_{i,c,d}N_{i,c,d}^{TF}.
```

여기서:

* $a_{i,c,d}$: channel $c$가 global representation에 얼마나 기여할지를 나타내는 softmax weight입니다.
* $g_{i,d}$: 여러 channel을 weighted sum한 global feature입니다.
* $C$: channel 수입니다.

그다음 개념적으로

```math
\hat N_i^C
=
\text{Concat}
\left(
N_i^{TF},
\text{Repeat}(g_i)
\right),
```

```math
N_i^C
=
\text{MLP}(\hat N_i^C)
```

가 됩니다.

그리고 Add & Norm을 통해

```math
Z_i
=
\text{LayerNorm}
\left(
N_i^{TF}+N_i^C
\right),
```

```math
N_i
=
\text{LayerNorm}
\left(
Z_i+\text{MLP}(Z_i)
\right)
```

와 같은 흐름으로 최종 nonlinear representation을 생성하는 것으로 해석할 수 있습니다. 정확한 흐름은 Figure 2와 본문 설명에 근거한 재표현입니다. 

**용어 설명 — Channel dependency:** 여러 센서나 변수 사이의 관계입니다. 예를 들어 온도 변화가 압력 및 유량 변화와 함께 나타나는 구조를 뜻합니다.

**용어 설명 — Softmax weighted aggregation:** 여러 channel을 단순 평균하지 않고 학습된 중요도에 따라 가중 평균하는 방식입니다.

---

# 2-1-7. 한 LiNo level의 전체 흐름

Appendix의 Eq.(8)이 가장 명료합니다.

$$
L_i=\text{LiBlock}(H_i),
$$

$$
P_i^{li}=\text{FC}(L_i),
$$

$$
R_i^L=H_i-L_i,
$$

$$
N_i=\text{NoBlock}(R_i^L),
$$

$$
P_i^{no}=\text{FC}(N_i),
$$

$$
R_i^N=R_i^L-N_i,
$$

$$
H_{i+1}=R_i^N,
$$

$$
P_i=P_i^{li}+P_i^{no},
$$

$$
\boxed{
\hat Y=\sum_{i=1}^{N}P_i
}
$$

입니다. 

여기서:

* $N$: LiNo block/decomposition level 수입니다.
* $P_i^{li}$: $i$번째 level의 linear forecast입니다.
* $P_i^{no}$: $i$번째 level의 nonlinear forecast입니다.
* $\text{FC}$: representation을 미래 $F$개 값으로 mapping하는 fully-connected prediction head입니다.
* $R_i^N$: 현재 level에서 두 pattern을 제거하고 다음 level로 넘길 residual입니다.

이 구조는 단순히 마지막 representation 하나만 forecasting하는 모델과 다릅니다.

```math
\hat Y
=
\underbrace{P_1^{li}+P_1^{no}}_{\text{1단계}}
+
\underbrace{P_2^{li}+P_2^{no}}_{\text{2단계}}
+\cdots
```

이므로 **각 decomposition level이 최종 forecast에 직접 기여**합니다.

이것이 N-BEATS-style residual refinement와 가장 큰 차이입니다. N-BEATS도 residual architecture를 사용하지만, LiNo는 residual을 **명시적으로 linear와 nonlinear hypothesis class로 다시 분리**합니다. 논문은 Appendix D에서 이를 직접 비교합니다.  

---

# 2-1-8. RevIN

논문은 distribution variation을 완화하기 위해 RevIN을 사용합니다. 

RevIN은 개념적으로 각 sample의 평균과 표준편차로

```math
x_t'
=
\frac{x_t-\mu_x}{\sigma_x+\epsilon}
```

처럼 정규화한 뒤 forecasting 결과를 다시 원래 scale로 돌립니다.

**용어 설명 — RevIN(Reversible Instance Normalization):** 서로 다른 시점/샘플 사이에서 평균이나 분산이 변하는 문제를 완화하는 normalization 방식입니다. “reversible”이라는 이름처럼 예측 후 원래 단위로 복원할 수 있습니다.

다만 이것이 **모든 종류의 distribution shift를 해결한다는 의미는 아닙니다.** 이후 연구들은 concept drift, frequency distribution 변화, input-output conditional shift 등 RevIN만으로 해결되지 않는 shift를 별도 문제로 다루고 있습니다. ([arXiv][2])

---

# 2-1-9. Loss와 평가 지표

논문은 MSE를 학습 loss로 사용하고 Adam optimizer와 early stopping patience 6을 사용했습니다. 

논문 표기상

```math
\text{MSE}
=
\frac1F
\sum_{i=1}^{F}
(Y_i-\hat Y_i)^2,
```

```math
\text{MAE}
=
\frac1F
\sum_{i=1}^{F}
|Y_i-\hat Y_i|
```

입니다. 

여기서 $Y_i,\hat Y_i$는 $i$번째 미래 시점의 ground truth와 prediction입니다.

다만 논문은 $Y,\hat Y\in\mathbb R^{F\times C}$라고 동시에 정의하므로, 이 수식에서 **channel $C$에 대한 scalar aggregation을 정확히 어떻게 표기하는지는 생략되어 있습니다.** 실제 benchmark 구현에서는 일반적으로 모든 prediction element를 평균하지만, 표시된 식 자체만 보면 이 부분이 다소 모호합니다.

---

# 2-1-10. 모델 구조를 한 문장으로 압축하면

$$
\boxed{
X
\rightarrow
\text{RevIN/Embedding}
\rightarrow
[\text{Li}\rightarrow\text{Residual}\rightarrow
\text{No}\rightarrow\text{Residual}]^{N}
\rightarrow
\sum_i(P_i^{li}+P_i^{no})
\rightarrow
\hat Y
}
$$

입니다.

따라서 LiNo의 본질은 **“decomposition을 preprocessing으로 한 번 수행하는 모델”이 아니라 “decomposition 자체를 deep forecasting network의 반복 연산으로 만든 모델”**입니다. Figure 2가 이 논문의 핵심 그림입니다. 

---

# 2-1-11. 성능 향상

## Multivariate

Table 1 기준 LiNo는 MSE에서 10개 benchmark 중 9개에서 1위를 기록합니다. 주요 값은 다음과 같습니다. 

| Dataset  |  LiNo MSE | iTransformer MSE | 관찰          |
| -------- | --------: | ---------------: | ----------- |
| ETT Avg  |     0.368 |            0.383 | LiNo 우세     |
| ECL      | **0.164** |            0.178 | 약 7.9% 감소   |
| Exchange | **0.350** |            0.360 | 소폭 우세       |
| Traffic  |     0.465 |        **0.428** | **LiNo 열세** |
| Weather  | **0.241** |            0.258 | 우세          |
| Solar    | **0.230** |            0.233 | 매우 작은 차이    |
| PEMS03   | **0.096** |            0.113 | 큰 차이        |
| PEMS04   | **0.098** |            0.111 | 우세          |
| PEMS07   | **0.088** |            0.101 | 우세          |
| PEMS08   | **0.138** |            0.150 | 우세          |

저자는 전체 평균에서 iTransformer 대비 **MSE 3.41% 감소**, PEMS 계열에서 평균 **11.89% 감소**라고 보고합니다. 

그러나 Traffic에서는 LiNo가 iTransformer보다 명확히 좋지 않습니다. 즉 “항상 SOTA”보다는 **“대부분의 benchmark에서 우수”**가 정확한 표현입니다.

---

## Univariate

Table 2에서는 6개 데이터셋 모두 LiNo가 가장 낮은 MSE를 보고합니다. 저자가 계산한 MICN 대비 평균 개선은

$$
\Delta\text{MSE}\approx -19.37\%,
$$

$$
\Delta\text{MAE}\approx -10.28\%
$$

입니다. 

특히 저자 보고 기준:

* Weather MSE: 약 **47.11% 감소**
* ETTh2: 약 **28.64% 감소**
* Traffic: 약 **12.97% 감소**

입니다.

---

# 2-1-12. Ablation이 말해주는 것

Table 3의 평균 변화에서 완전한 LiNo 대비:

$$
\text{w/o Li Block}: \quad \text{MSE}\uparrow 10.00\%,
$$

$$
\text{w/o No Block}: \quad \text{MSE}\uparrow 71.82\%,
$$

$$
\text{w/o Channel Dependency}: \quad \text{MSE}\uparrow 15.91\%
$$

를 보고합니다. 

이 결과는 **No block의 중요성이 특히 크고, channel interaction도 상당한 성능 기여를 한다**는 저자의 주장을 지지합니다.

다만 ETTh2 F=720처럼 일부 개별 조건에서는 component를 제거한 모델이 근소하게 더 좋은 경우도 보입니다. 따라서 모든 dataset/horizon에서 각 component가 항상 양의 효과를 내는 것은 아닙니다.

---

# 3. 주장별 Page / Figure / Table 위치 요약

| 내용                              | 위치                            |
| ------------------------------- | ----------------------------- |
| 기존 STD의 세 가지 한계                 | p.1–2                         |
| multi-level linear/nonlinear 예시 | **Figure 1, p.2**             |
| RRD 정의                          | **Eq.(1), p.3**               |
| 전체 LiNo 구조                      | **Figure 2, p.4**             |
| Li block AR                     | **Eq.(2), p.4**               |
| Padding                         | **Eq.(3), p.4**               |
| No block time/frequency/channel | **Figure 2, p.4–5**           |
| 최종 forecast 합산                  | **Eq.(4), p.5**, Eq.(8), p.15 |
| Multivariate benchmark          | **Table 1, p.6**              |
| Univariate benchmark            | **Table 2, p.6**              |
| component ablation              | **Table 3, p.7**              |
| decomposition depth $N$         | **Table 4, p.7**              |
| time/frequency/Tanh ablation    | **Table 5, p.7**              |
| No block 및 Raw/Mu/LiNo 비교       | **Table 6, p.8**              |
| Gaussian noise robustness       | **Figure 3, p.8**             |
| dataset split                   | **Table 7, p.14**             |
| 5-seed error bar                | **Table 8, p.14**             |
| efficiency                      | **Table 9, p.15**             |
| longer lookback                 | **Figure 4, p.15**            |
| N-BEATS-style 구조 비교             | **Table 10–11, p.15–16**      |
| SOFTS 비교                        | **Table 12, p.17**            |
| TimeMixer/SegRNN 추가 비교          | **Table 13, p.17**            |
| 실제 prediction decomposition     | **Figure 5–17, p.19–22**      |
| learned weight visualization    | **Figure 18–30, p.22 이후**     |

---

# 4. “저자가 직접 보고한 것”과 “분석적 해석” 분리

| 항목             | **저자 직접 보고**                                                              | **분석적 해석**                                                                 |
| -------------- | ------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| 연구 주제          | real-world TS는 multi-level linear/nonlinear pattern의 합으로 볼 수 있음           | 매우 유용한 inductive bias지만 decomposition의 유일성은 보장되지 않음                        |
| RRD            | residual을 반복 분리하면 이전 feature가 이후 추출에 방해하지 않고 independent pattern을 얻는다고 주장 | subtraction은 statistical independence를 수학적으로 보장하지 않음                       |
| Residual       | 충분히 깊으면 $R_i^N\to0$이라고 서술                                                 | convergence 조건과 proof가 없음                                                  |
| Li block       | general learnable AR, full receptive field                                | raw-time AR보다는 embedded-space AR-like linear extractor로 보는 것이 더 정확         |
| No block       | time/frequency/inter-series dependency를 동시에 모델링                           | 설계상 합리적이며 ablation으로 일부 입증됨                                                |
| 성능             | multivariate MSE 9/10 1위, univariate 6/6 1위                               | benchmark 수준의 evidence는 강하지만 모든 baseline의 동일-run 공정성은 완전하지 않음              |
| Robustness     | Gaussian noise 아래 Raw/Mu보다 강함                                             | synthetic Gaussian perturbation robustness이며 실제 regime/OOD shift까지 증명하지 않음 |
| Generalization | seed error bar가 작아 generalization ability가 높다고 해석                         | seed stability는 **optimization stability**에 더 가까움                          |
| Long context   | lookback 증가 시 성능 향상                                                       | 장기 context를 이용할 능력의 evidence는 있음                                           |
| Deeper RRD     | $N>1$이 대체로 좋음                                                             | 최적 $N$은 dataset마다 다르고 monotonic improvement는 아님                            |

---

# 5. 통계적으로 취약한 부분과 직접 비교가 어려운 수치

## 5-1. `[통계 취약]` baseline 간 유의성 검정이 없다

Table 8은 LiNo 자체에 대해서는 **5 random seeds의 mean ± std**를 제공합니다. 예를 들어 ETTh2, ETTm2, Weather, PEMS04, PEMS08 결과의 표준편차는 대체로 작습니다. 

하지만 동일 조건에서 iTransformer, TimeMixer, PatchTST 등의

$$
\mu_{\text{LiNo}}-\mu_{\text{baseline}}
$$

에 대한 confidence interval이나 paired test는 없습니다.

따라서 예를 들어

$$
0.395\quad\text{vs.}\quad0.396
$$

처럼 차이가 $0.001$ 수준인 결과를 **“통계적으로 우수하다”**고 해석할 수는 없습니다. Table 13에는 이런 작은 차이가 실제로 존재합니다. 

논문에서 사용한 “significant”라는 표현은 상당 부분 **effect가 크다는 일상적 의미**이지 statistical significance test를 의미하지 않습니다.

---

## 5-2. `[비교 주의]` baseline 숫자가 모두 동일 실험에서 재측정된 것은 아니다

Multivariate Table 1에서 저자들은 TSMixer를 Time-Series-Library로 재현했지만 **다른 상당수 baseline result는 iTransformer 논문에서 가져왔다고 명시**합니다. 

Univariate Table 2는 MICN 연구의 benchmark 숫자를 사용합니다. 

Table 11의 N-BEATS/N-HiTS 역시 기존 논문에서 가져왔으며, Table 12의 SOFTS도 원 논문 결과를 직접 참조합니다.  

따라서:

$$
\text{same dataset}+\text{same horizon}
\neq
\text{완전히 동일한 tuning / seed / code path}
$$

입니다.

이를 **조건부 비교 가능**으로 보는 것이 안전합니다.

---

## 5-3. `[문서 내부 불일치]` PEMS prediction horizon

Table 1 caption은 PEMS prediction lengths를

$$
\{12,24,36,48\}
$$

로 씁니다. 

그러나 Methods와 Appendix dataset description은

$$
\boxed{\{12,24,48,96\}}
$$

이라고 명시합니다.  

따라서 Table 1 caption의 “36”은 **오타일 가능성이 매우 높지만**, 원문이 서로 모순되므로 임의로 확정해서는 안 됩니다.

---

## 5-4. `[통계 취약]` PEMS 일부 Test set이 매우 작다

Table 7에서:

* PEMS04 test = **281 points**
* PEMS07 test = **468**
* PEMS08 test = **265**

입니다. 

시계열 forecast window는 서로 겹치므로 nominal sample 개수와 **effective independent sample size**가 같지 않습니다.

**용어 설명 — Effective sample size:** 데이터 행이 300개라고 해서 독립적인 정보가 300개라는 뜻은 아닙니다. 서로 강하게 autocorrelated한 시계열이면 실질적인 독립 정보량은 더 작습니다.

따라서 작은 차이의 benchmark ranking은 일반적인 IID test set보다 더 조심해서 봐야 합니다.

---

## 5-5. `[통계 취약]` Gaussian noise의 “25%, 50%, 100%”가 SNR로 정의되지 않는다

논문은

$$
\hat X=X+\alpha\cdot\epsilon,
$$

$$
\epsilon\sim\mathcal N(0,1),
$$

$$
\alpha\in\{0,0.25,0.50,0.75,1.0\}
$$

을 사용합니다. 

하지만 $\alpha=100%$가

* signal standard deviation의 100%인지,
* normalized unit variance 기준인지,
* 특정 SNR인지

명확한 정의가 충분하지 않습니다.

따라서 **“100% noise에서도 robust하다”**를 실제 물리 signal에서 그대로 적용해서는 안 됩니다.

---

## 5-6. `[일반화 해석 주의]` Table 8은 OOD generalization 실험이 아니다

5 random seeds에 대해

$$
\text{mean}\pm\text{std}
$$

가 작다는 것은 주로 **random initialization과 optimization에 안정적**이라는 의미입니다. 

진정한 OOD generalization이라면 예를 들어

$$
P_{\text{train}}(X,Y)\ne P_{\text{test}}(X,Y)
$$

인 새로운 설비, 새로운 지역, 새로운 운전 regime, unseen channel 구성 등에서 평가해야 합니다.

그 실험은 LiNo 논문에 없습니다.

**용어 설명 — OOD(Out-of-Distribution):** 학습 데이터와 분포가 다른 환경에서 모델을 평가하는 것입니다.

---

## 5-7. `[구조적 약점]` 더 깊은 decomposition이 항상 좋아지지는 않는다

Table 4에서 최적 $N$은 dataset에 따라 2, 3, 4로 바뀝니다. 

예를 들어 어떤 dataset에서는

$$
N=3 < N=4\quad \text{in MSE},
$$

다른 dataset에서는 반대입니다.

즉,

$$
\text{deeper RRD}\Rightarrow\text{always better}
$$

가 아닙니다.

따라서 $N$은 실질적으로 **dataset-specific hyperparameter**입니다.

---

# 6. 이 논문이 답하지 않는 질문

1. **Linear와 nonlinear component가 실제로 서로 독립적인가?** HSIC, mutual information, cross-correlation 등의 측정이 없습니다.

2. **동일한 $X$에 대해 decomposition이 유일한가?** 즉 identifiability에 대한 이론이 없습니다.

3. **왜 $R_i^N\rightarrow0$이어야 하는가?** contraction condition이나 approximation theorem이 없습니다.

4. **언제 decomposition을 멈춰야 하는가?** 현재는 $N\in{1,2,3,4}$를 validation으로 선택합니다.

5. **새로운 domain으로 이동했을 때도 Li/No 역할이 유지되는가?** cross-domain transfer가 없습니다.

6. **Train/Test distribution이 크게 바뀌면 RevIN만으로 충분한가?** concept drift 실험이 없습니다.

7. **Irregular sampling과 missing sensor가 있으면 어떻게 되는가?** 실험하지 않습니다.

8. **Exogenous variables가 있을 때 구조를 어떻게 확장해야 하는가?** 명시적 연구가 없습니다.

9. **Point forecasting뿐 아니라 predictive uncertainty는 어떠한가?** quantile loss, CRPS, NLL, prediction interval은 보고하지 않습니다.

10. **No block의 channel softmax가 channel 수가 수천 개로 증가해도 안정적인가?** large- $C$ scaling의 체계적인 연구는 없습니다.

11. **Gaussian random noise가 아니라 sensor drift, bias drift, regime change, outlier burst에서도 robust한가?** 알 수 없습니다.

12. **전체 모델이 작은 데이터에서도 효과적인가?** 13개 표준 benchmark는 제공되지만 small-sample learning curve가 없습니다.

---

# 7. 가장 중요한 그림 5개 해석

## Figure 1 — p.2: “왜 LiNo가 필요한가?”

Figure 1은 ETTh2 한 시계열을 Raw, Linear 1, Linear 2, Nonlinear 1, Nonlinear 2로 분해하고 각 신호의 autocorrelation을 함께 보여줍니다. 저자의 핵심 메시지는

$$
X
\approx
L_1+N_1+L_2+N_2
$$

처럼 하나의 trend/seasonality만으로 표현하기 어려운 여러 수준의 패턴이 있다는 것입니다. 

**제 해석:** 이 그림은 “여러 level로 나누는 것이 가능하고 시각적으로 서로 다른 특성을 보인다”는 좋은 illustration입니다. 그러나 이것이 **실제 ground-truth linear/nonlinear decomposition임을 증명하지는 않습니다.** 동일 신호에 다른 decomposition도 가능하기 때문입니다.

**용어 설명 — Autocorrelation:** 현재 신호와 일정 시간만큼 이동시킨 자기 자신이 얼마나 비슷한지를 나타냅니다. 반복 주기가 있으면 특정 lag에서 높은 값이 반복될 수 있습니다.

---

## Figure 2 — p.4: 논문의 가장 중요한 architecture figure

Figure 2는 전체 LiNo, Li block, No block을 한 번에 보여줍니다. 

핵심 흐름은

$$
H_i
\xrightarrow{\text{Li}}
L_i
$$

$$
R_i^L=H_i-L_i
$$

$$
R_i^L
\xrightarrow{\text{No}}
N_i
$$

$$
R_i^N=R_i^L-N_i
$$

$$
H_{i+1}=R_i^N
$$

입니다.

No block 내부에서는

$$
\text{time-domain}
+
\text{frequency-domain}
+
\text{channel interaction}
$$

을 결합합니다.

**제 해석:** LiNo에서 성능을 만드는 핵심은 특정 Transformer layer 하나가 아니라 **residual routing 구조**입니다. 각 block에 “너는 linear 쪽을 먼저 설명하라”, “그 후 남은 것에서 nonlinear을 설명하라”는 역할 분담을 architecture로 강제한 것입니다.

---

## Figure 3 — p.8: Gaussian noise robustness

Figure 3은 ECL, ETTm2, Weather에서 Raw, Mu, LiNo를 여러 noise level로 비교합니다. LiNo curve가 전체적으로 가장 낮은 MSE를 유지합니다. 

**저자의 해석:** linear/nonlinear을 분리하면 noisy representation에 덜 민감해진다는 것입니다.

**제 해석:** 이는 꽤 의미 있는 결과입니다. 복잡한 nonlinear network가 raw mixture 전체를 한꺼번에 fitting하지 않고,

$$
\text{structured component}
\rightarrow
\text{residual}
\rightarrow
\text{next component}
$$

방식으로 설명하기 때문에 noise에 대한 unnecessary fitting이 줄어들 가능성이 있습니다.

그러나 이 실험은 **training-time Gaussian perturbation에 대한 robustness**이며, factory drift나 economic regime change 같은 자연적인 distribution shift와는 다른 문제입니다.

---

## Figure 4 — p.15: Longer lookback을 실제로 활용하는가?

$F=720$을 고정하고

$$
T\in\{48,96,192,336,720\}
$$

으로 늘렸을 때 LiNo의 MSE가 ECL과 Weather에서 안정적으로 감소합니다. 

**제 해석:** Li block의 full receptive field와 No block의 time/frequency modeling이 긴 history를 단순히 입력으로 받는 데서 끝나는 것이 아니라 **실제로 정보로 활용할 가능성**을 지지합니다.

그러나 이것도

> 긴 context 활용 능력 ≠ 새로운 domain으로의 generalization

입니다.

---

## Figure 18 — p.22: Li와 No가 실제로 다른 표현을 배우는가?

Figure 18은 ECL에서 3개 level의 Li/No weight heatmap을 보여줍니다. 

Li 쪽 heatmap과 No 쪽 heatmap의 구조가 분명히 다르고, nonlinear 쪽은 더 광범위한 band/structured dependence를 나타냅니다.

**제 해석:** 이 그림은 “서로 다른 block들이 정확히 동일한 mapping을 복제하는 것만은 아니다”라는 정성적 증거로는 유용합니다.

하지만 **weight pattern이 다르다 = statistical independence가 성립한다**는 뜻은 아닙니다.

또한 Appendix에서는 trained mapping을

$$
f(\mathbf x)=A\mathbf x+\mathbf b
$$

형태로 해석하기 위해 zero vector와 basis vector를 입력하는 visualization 방법을 설명합니다. 

No block 전체는 Tanh와 MLP를 포함하는 nonlinear mapping이므로, Figure 18의 “weight”가 전체 No mapping에 대한 하나의 global affine matrix인지 아니면 forecast head의 effective mapping인지는 본문 설명만으로 완전히 명료하지 않습니다. 따라서 **causal attribution map으로 해석해서는 안 됩니다.**

---

# 8. 결론: 저자 시사점과 후속 연구

저자들의 결론은 명확합니다. 기존 seasonal/trend decomposition은 linear extractor가 지나치게 단순하고 residual의 nonlinear component를 명시적으로 분리하지 않으며 decomposition depth도 얕다는 것입니다. LiNo는 general linear extractor와 nonlinear extractor를 RRD에 함께 넣어 더 정교한 multi-level decomposition을 만들었고, 그것이 accuracy와 noise robustness 향상으로 이어졌다고 결론냅니다. 

다만 **논문 Conclusion에 구체적으로 정의된 future-work agenda는 없습니다.** “더 robust하고 precise한 forecasting model 설계에 insight를 줄 수 있다”는 수준입니다.

공식 GitHub에는 benchmark별 학습 script, logs, MSE/MAE 결과 및 prediction/weight visualization이 공개되어 있어 재현성 측면에서는 긍정적입니다. ([GitHub][3])

흥미롭게도 같은 연구진의 후속 연구인 **ReFocus(2025)** 는 frequency decomposition을 더 세밀하게 발전시켜 mid-frequency gap과 shared key-frequency를 명시적으로 다룹니다. 이는 LiNo 논문이 공식적으로 “다음 연구는 ReFocus다”라고 선언한 것은 아니지만, 연구진의 후속 흐름으로 보면 **LiNo의 time-frequency extraction을 주파수 구조 관점에서 더 세분화하는 방향**으로 해석할 수 있습니다. ([arXiv][4])

---

# 8-1. 모델 일반화 성능 향상 가능성

여기가 LiNo에서 앞으로 가장 중요하게 연구할 부분이라고 판단합니다.

## A. 현재 LiNo가 갖는 일반화에 유리한 요소

첫째, linear/nonlinear 역할 분리는 하나의 네트워크가 모든 패턴을 동시에 외우는 것을 줄이는 **structural regularization** 역할을 할 수 있습니다.

둘째, residual learning은 각 단계가

```math
H_i
=
\text{이미 설명된 부분}
+
\text{아직 설명하지 못한 부분}
```

중 후자에 집중하도록 만듭니다.

셋째, RevIN은 mean/variance level의 temporal distribution variation을 완화합니다.

넷째, dropout과 multilevel prediction aggregation도 overfitting 완화에 기여할 가능성이 있습니다.

하지만 이 논문이 직접 확인한 것은 **benchmark-level generality와 seed stability**이지 진정한 domain generalization이 아닙니다.

---

## B. 1순위 후속 연구 — “독립적 decomposition”을 실제 loss로 만든다

현재는 architecture가 역할을 분리할 뿐입니다.

이를 objective에도 명시하여

```math
\mathcal L
=
\mathcal L_{\text{forecast}}
+
\lambda_{\text{ind}}
\mathcal L_{\text{ind}}
+
\lambda_R
\mathcal L_{\text{residual}}
```

로 확장할 수 있습니다.

예를 들어 단순한 decorrelation version은

```math
\mathcal L_{\text{ind}}
=
\sum_i
\left\|
L_i^\top N_i
\right\|_F^2
```

로 둘 수 있습니다.

더 엄밀한 nonlinear dependence를 억제하려면

```math
\mathcal L_{\text{ind}}
=
\sum_i
\text{HSIC}(L_i,N_i)
```

같은 방법을 사용할 수 있습니다.

**용어 설명 — HSIC:** 두 representation 사이에 선형뿐 아니라 nonlinear dependence가 남아 있는지를 kernel을 이용해 측정하는 통계량입니다.

이렇게 해야 “linear/nonlinear independence”라는 논문의 철학이 **서술이 아니라 실제 optimization target**이 됩니다.

---

## C. 2순위 — Adaptive decomposition depth

Table 4가 보여주듯 optimal $N$은 dataset마다 다릅니다.

따라서 고정

$$
N=1,2,3,4
$$

중 하나를 고르는 대신 residual energy를 사용해

```math
\rho_i
=
\frac{\|R_i^N\|_2^2}
{\|H_i\|_2^2}
```

를 계산하고

$$
\rho_i < \tau
$$

이면 decomposition을 멈추는 방법이 더 합리적입니다.

더 나아가

$$
g_i=\sigma(f_\theta(H_i))
$$

라는 learnable gate를 두어

```math
H_{i+1}
=
g_iR_i^N
```

처럼 **데이터별 adaptive depth**를 학습할 수 있습니다.

이는 unnecessary decomposition과 overfitting을 동시에 줄일 가능성이 있습니다.

---

## D. 3순위 — Distribution shift를 직접 학습한다

RevIN만으로는

$$
P_{\text{train}}(Y|X)
\neq
P_{\text{test}}(Y|X)
$$

같은 concept drift를 해결하기 어렵습니다.

2023 Dish-TS는 lookback/horizon 사이 distribution shift를 별도로 모델링했고, 2024의 robust MTS work와 2025 ShifTS는 temporal shift뿐 아니라 concept drift를 명시적인 연구 대상으로 삼고 있습니다. ([arXiv][2])

**용어 설명 — Concept drift:** 동일한 $X$가 주어져도 시간이 지나면서 $Y$와의 관계 자체가 변하는 현상입니다. 단순 평균/분산 normalization만으로는 해결되지 않을 수 있습니다.

LiNo에서는 각 level을

$$
\text{stable pattern}
+
\text{drifting pattern}
$$

로 다시 분리하거나, Li/No block에 regime-conditioned parameter를 도입하는 방법이 연구 가치가 큽니다.

---

## E. 4순위 — LiNo + Time-Series Foundation Model

Chronos는 대규모 heterogeneous time-series corpus와 synthetic data를 활용하여 unseen dataset에서 zero-shot prediction을 목표로 하고, TimesFM 역시 다양한 context/horizon/granularity를 다루는 pretrained decoder를 제안합니다. ([arXiv][5])

**용어 설명 — Foundation model:** 특정 하나의 데이터셋만 학습하는 대신 매우 큰 다양한 데이터로 pretrain하고 여러 downstream task에 재사용하는 모델입니다.

**용어 설명 — Zero-shot forecasting:** 새로운 데이터셋에 대해 그 데이터로 별도의 supervised training을 거의 하지 않고 예측하는 것입니다.

따라서

$$
\text{LiNo Decomposition Front-end}
+
\text{Pretrained Forecasting Backbone}
$$

이라는 구조가 매우 흥미롭습니다.

Li/No가 domain-specific nuisance structure를 먼저 분리하고 backbone은 여러 domain에서 공통으로 사용되는 representation을 담당하도록 할 수 있습니다.

---

## F. 5순위 — Point accuracy에서 uncertainty-aware forecasting으로

현재 LiNo는 주로 MSE/MAE입니다.

실제 deployment라면

$$
\hat y_t
$$

하나뿐 아니라

$$
p(y_t\mid X)
$$

또는

$$
[\hat y_t^{L},\hat y_t^{U}]
$$

가 필요합니다.

즉 Li/No별로 uncertainty를 별도 추정해

$$
\sigma_{\text{total}}^2
\approx
\sigma_L^2+\sigma_N^2+\sigma_\epsilon^2
$$

와 같은 probabilistic decomposition 연구도 가능합니다.

이는 “nonlinear residual을 다 설명했는가, 아니면 irreducible noise인가?”를 구분하는 데 특히 중요합니다.

---

# 8-2. 2020년 이후 관련 최신 연구 비교

아래 비교는 **2026-08-28 현재 확인 가능한 공개 논문**을 기준으로 합니다.

| 연구                                            | 핵심 방법                                                                        | LiNo와의 관계                                            | 일반화 관점                                                   |
| --------------------------------------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------- | -------------------------------------------------------- |
| **N-BEATS, 2020** ([OpenReview][6])           | backward/forward residual stack                                              | RRD의 직접적 선행 개념                                       | 여러 univariate domain에 동일 구조 적용                           |
| **Autoformer, 2021** ([arXiv][7])             | progressive trend-seasonal decomposition + autocorrelation                   | “decomposition을 deep block 내부로 넣는다”는 직접적 선행 흐름       | long-term forecasting                                    |
| **DLinear/LTSF-Linear, 2022/23** ([arXiv][8]) | 극단적으로 단순한 linear forecaster                                                  | LiNo가 linear component를 무시하면 안 된다는 근거                | 단순 모델도 strong baseline임을 증명                              |
| **PatchTST, 2023** ([OpenReview][9])          | patch token + channel independence                                           | decomposition보다 representation granularity를 개선       | longer context, transfer/self-supervised 활용              |
| **TimesNet, 2023** ([OpenReview][10])         | FFT로 period 탐색 후 1D→2D variation modeling                                    | LiNo No-block frequency branch와 문제의식 공유              | task-general representation                              |
| **iTransformer, 2024** ([OpenReview][11])     | channel을 token으로 invert해 cross-variate attention                             | LiNo embedding/channel modeling의 주요 비교 대상            | arbitrary lookback와 variate generalization 강조            |
| **TimeMixer, 2024** ([OpenReview][12])        | multiscale trend/seasonal decomposition                                      | LiNo와 가장 가까운 decomposition 계열 중 하나                   | multi-scale adaptation                                   |
| **SOFTS, 2024** ([arXiv][13])                 | STAR aggregate–redistribute global channel core                              | LiNo channel mixing과 매우 가까운 철학                       | distribution drift에서 channel-independent + dependency 절충 |
| **Chronos, 2024** ([arXiv][5])                | 대규모 pretrained probabilistic TS model                                        | decomposition이 아니라 pretraining paradigm              | unseen dataset zero-shot이 핵심                             |
| **DUET, 2024/25** ([arXiv][14])               | temporal clustering + channel soft clustering                                | 하나의 extractor 대신 pattern-specific distribution 대응    | distribution heterogeneity에 LiNo보다 직접적                   |
| **TFPS, 2024/25** ([arXiv][15])               | patch-level shift + pattern-specific experts                                 | 고정 Li/No보다 adaptive expert specialization            | pattern drift 일반화 중점                                     |
| **FreqMoE, 2025** ([arXiv][16])               | frequency-band decomposition + MoE + residual refinement                     | LiNo의 frequency branch를 더 세분화하는 방향                   | 주파수별 전문화로 heterogeneity 대응                               |
| **ReFocus, 2025** ([arXiv][4])                | mid-frequency enhancement + shared key-frequency                             | LiNo 연구진의 후속 frequency-focused 흐름                    | channel-shared spectral feature 강화                       |
| **Timer-XL, 2025** ([OpenReview][17])         | long-context causal Transformer, unified multivariate next-token forecasting | LiNo의 fixed task-specific model과 다른 scaling 방향       | pretrained zero-shot 및 varying context                   |
| **ShifTS, 2025/26** ([arXiv][18])             | temporal shift + concept drift                                               | LiNo가 거의 다루지 않은 실제 generalization 문제를 직접 다룸          | 미래분포 변화 자체가 주 연구 대상                                      |
| **DistDF, ICLR 2026** ([OpenReview][19])      | joint-distribution Wasserstein alignment                                     | MSE 중심 point fitting보다 distribution alignment를 전면에 둠 | shift-aware forecasting 평가 방향을 제시                        |

---

## 이 흐름에서 LiNo가 차지하는 위치

### 2020–2021

N-BEATS와 Autoformer가

$$
\boxed{\text{Residual/Decomposition 자체를 network architecture로 만든다}}
$$

는 방향을 확립했습니다. ([OpenReview][6])

### 2022–2024

DLinear는 “linear structure를 무시하면 안 된다”고 보여주었고, PatchTST/iTransformer/TimesNet은 각각 patch, channel, frequency라는 representation axis를 정교화했습니다. ([arXiv][8])

### LiNo의 차별점

LiNo는 이들을 하나의 관점으로 통합하여

$$
\boxed{
\text{Linear hypothesis class}
\rightarrow
\text{Residual}
\rightarrow
\text{Nonlinear hypothesis class}
\rightarrow
\text{Residual}
}
$$

을 **반복**했다는 데 가장 큰 의의가 있습니다.

즉 Autoformer처럼 trend-seasonal을 단순 반복하는 것도 아니고, N-BEATS처럼 generic residual block을 반복하는 것도 아니며, **residual을 서로 다른 함수군이 번갈아 설명하도록 한 것**입니다.

---

# 향후 연구에 LiNo가 미칠 수 있는 영향

LiNo의 가장 재사용 가치가 높은 아이디어는 Li block이나 No block 자체가 아니라 **“Recursive Heterogeneous Decomposition”**이라고 볼 수 있습니다.

현재는

$$
\text{Linear}\leftrightarrow\text{Nonlinear}
$$

이지만 이를 앞으로

$$
\text{Stable}\leftrightarrow\text{Drifting},
$$

$$
\text{Low-frequency}\leftrightarrow\text{High-frequency},
$$

$$
\text{Shared-channel}\leftrightarrow\text{Channel-specific},
$$

$$
\text{Predictable}\leftrightarrow\text{Irreducible noise}
$$

등으로 확장할 수 있습니다.

이것이 LiNo가 향후 연구에 줄 수 있는 가장 큰 설계적 영향입니다.

---

# 앞으로 연구할 때 특히 고려해야 할 점

### 1. SOTA 숫자보다 동일 조건 재현이 우선입니다

2025년 *ModernTCN Revisited* 연구는 time-series SOTA 평가가 **data loading, validation, evaluation protocol**에 민감할 수 있음을 실제 재평가를 통해 지적하고, 여러 run의 표준편차와 통계검정을 강화했습니다. ([OpenReview][20])

LiNo 후속 연구에서도 모든 baseline을 동일 framework에서 다시 학습하고 최소한

$$
\text{mean}\pm\text{std}
$$

와 paired statistical test를 제공하는 편이 바람직합니다.

---

### 2. “Generalization” 평가를 seed stability와 분리해야 합니다

권장 실험은 예를 들어

$$
\text{Train domain A}
\rightarrow
\text{Test domain B},
$$

$$
\text{Train regime }1
\rightarrow
\text{Test regime }2,
$$

$$
C_{\text{train}}
\neq
C_{\text{test}}
$$

처럼 구성해야 합니다.

특히 2024–2026의 연구 흐름은 단순 benchmark MSE 경쟁에서 **distribution shift, concept drift, zero-shot, long-context, joint-distribution alignment** 쪽으로 이동하고 있습니다. ([arXiv][21])

---

### 3. decomposition을 설명가능성으로 주장하려면 identifiability가 필요합니다

현재

$$
X=L_1+N_1+\cdots
$$

이라는 decomposition은 forecasting에는 유용할 수 있지만, 이것이 “실제로 존재하는 물리적 linear/nonlinear source를 발견했다”는 뜻은 아닙니다.

향후에는 orthogonality, independence, sparsity, frequency-band constraint 또는 domain knowledge를 추가해 decomposition의 의미를 고정해야 합니다.

---

### 4. Adaptive LiNo가 고정 LiNo보다 일반화에 유리할 가능성이 큽니다

현재는 모든 dataset에

$$
\text{Li}\rightarrow\text{No}\rightarrow
\text{Li}\rightarrow\text{No}
$$

라는 동일한 순서를 사용합니다.

하지만 실제 data마다 필요한 것은 다를 수 있습니다.

따라서

```math
g_i^{L},g_i^{N}
=
\text{Gate}(H_i)
```

를 두고

```math
P_i
=
g_i^{L}P_i^{li}
+
g_i^{N}P_i^{no}
```

처럼 adaptive routing을 하는 것이 자연스러운 후속 방향입니다.

FreqMoE나 TFPS 같은 최신 연구가 보여주는 **expert specialization**과 LiNo의 residual decomposition을 결합하는 방향이 특히 유망합니다. ([arXiv][16])

---

# 최종 평가

제 판단으로 LiNo의 논문 가치는 다음 두 층을 분리해서 평가해야 합니다.

**모델 성능 측면에서는 강한 논문입니다.** 13개 real-world benchmark, multivariate/univariate 양쪽, component ablation, architecture comparison, noise experiment, longer-lookback study, 5-seed stability, inference-time analysis까지 비교적 폭넓게 수행했습니다. 특히 No block 제거 시 성능 악화와 Raw/Mu/LiNo 비교는 제안 architecture의 필요성을 뒷받침합니다. 

반면 **이론적 decomposition 주장에는 과장이 일부 있습니다.** residual subtraction이 statistical independence를 보장하지 않으며,

$$
\lim_{i\to\infty}R_i^N=0
$$

도 증명되지 않았습니다. 따라서 LiNo를 “linear/nonlinear signal을 수학적으로 정확히 식별하는 decomposition method”로 보는 것보다,

> **linear와 nonlinear hypothesis class에 서로 다른 역할을 부여하고 residual을 재귀적으로 전달하는 forecasting architecture**

라고 이해하는 것이 가장 정확합니다.

그리고 **일반화 성능 관점에서 현재 LiNo는 완성형이라기보다 좋은 출발점입니다.** 실제 다음 단계의 핵심은 더 높은 in-domain benchmark MSE 경쟁보다는 **distribution shift, concept drift, adaptive decomposition depth, statistically identifiable separation, probabilistic uncertainty, cross-domain/zero-shot evaluation**을 결합하여 “왜 decomposition이 unseen future에서도 살아남는가”를 검증하는 것입니다.

---

## 참고한 원문 및 웹 자료

1. **LiNo: Advancing Recursive Residual Decomposition of Linear and Nonlinear Patterns for Robust Time Series Forecasting** — 첨부 arXiv v4 및 arXiv 공개본.  ([arXiv][1])
2. **Official implementation of LiNo — Levi-Ackman/LiNo**. ([GitHub][3])
3. **N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting** — ICLR 2020. ([OpenReview][6])
4. **Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting**. ([arXiv][7])
5. **Are Transformers Effective for Time Series Forecasting?** — LTSF-Linear/DLinear. ([arXiv][8])
6. **A Time Series is Worth 64 Words: Long-term Forecasting with Transformers** — PatchTST. ([OpenReview][9])
7. **TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis**. ([OpenReview][10])
8. **iTransformer: Inverted Transformers Are Effective for Time Series Forecasting**. ([OpenReview][11])
9. **TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting**. ([OpenReview][12])
10. **SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion**. ([arXiv][13])
11. **Chronos: Learning the Language of Time Series**. ([arXiv][5])
12. **A decoder-only foundation model for time-series forecasting** — TimesFM. ([arXiv][22])
13. **DUET: Dual Clustering Enhanced Multivariate Time Series Forecasting**. ([arXiv][14])
14. **Learning Pattern-Specific Experts for Time Series Forecasting Under Patch-level Distribution Shift**. ([arXiv][15])
15. **FreqMoE: Enhancing Time Series Forecasting through Frequency Decomposition Mixture of Experts**. ([arXiv][16])
16. **ReFocus: Reinforcing Mid-Frequency and Key-Frequency Modeling for Multivariate Time Series Forecasting**. ([arXiv][4])
17. **Timer-XL: Long-Context Transformers for Unified Time Series Forecasting** — ICLR 2025. ([OpenReview][17])
18. **Dish-TS: A General Paradigm for Alleviating Distribution Shift in Time Series Forecasting**. ([arXiv][2])
19. **Robust Multivariate Time Series Forecasting against Intra- and Inter-Series Transitional Shift**. ([arXiv][21])
20. **Tackling Time-Series Forecasting Generalization via Mitigating Concept Drift** — ShifTS. ([arXiv][18])
21. **DistDF: Time-Series Forecasting Needs Joint-Distribution Wasserstein Alignment** — ICLR 2026. ([OpenReview][19])
22. **ModernTCN Revisited: A Critical Look at the Experimental Setup in General Time Series Analysis** — time-series benchmark의 validation/evaluation 및 statistical significance 문제를 검토하기 위해 참고했습니다. ([OpenReview][20])

[1]: https://arxiv.org/abs/2410.17159?utm_source=chatgpt.com "LiNo: Advancing Recursive Residual Decomposition of Linear and Nonlinear Patterns for Robust Time Series Forecasting"
[2]: https://arxiv.org/abs/2302.14829?utm_source=chatgpt.com "Dish-TS: A General Paradigm for Alleviating Distribution Shift in Time Series Forecasting"
[3]: https://github.com/levi-ackman/lino?utm_source=chatgpt.com "GitHub - Levi-Ackman/LiNo: Official implementation of paper: LiNo: Advancing Recursive Residual Decomposition of Linear and Nonlinear Patterns for Robust Time Series Forecasting. · GitHub"
[4]: https://arxiv.org/abs/2502.16890?utm_source=chatgpt.com "ReFocus: Reinforcing Mid-Frequency and Key-Frequency Modeling for Multivariate Time Series Forecasting"
[5]: https://arxiv.org/abs/2403.07815?utm_source=chatgpt.com "Chronos: Learning the Language of Time Series"
[6]: https://openreview.net/pdf?id=r1ecqn4YwB&utm_source=chatgpt.com "Published as a conference paper at ICLR 2020"
[7]: https://arxiv.org/abs/2106.13008?utm_source=chatgpt.com "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting"
[8]: https://arxiv.org/abs/2205.13504?utm_source=chatgpt.com "Are Transformers Effective for Time Series Forecasting?"
[9]: https://openreview.net/pdf?id=Jbdc0vTOcol&utm_source=chatgpt.com "Published as a conference paper at ICLR 2023"
[10]: https://openreview.net/references/pdf?id=Mq6sBdvkM1&utm_source=chatgpt.com "Under review as a conference paper at ICLR 2023"
[11]: https://openreview.net/pdf/2dff92321d132ee1a70ba42c0046c99ee9bcd972.pdf?utm_source=chatgpt.com "Published as a conference paper at ICLR 2024"
[12]: https://openreview.net/pdf?id=7oLshfEIC2&utm_source=chatgpt.com "Published as a conference paper at ICLR 2024"
[13]: https://arxiv.org/abs/2404.14197?utm_source=chatgpt.com "SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion"
[14]: https://arxiv.org/abs/2412.10859?utm_source=chatgpt.com "DUET: Dual Clustering Enhanced Multivariate Time Series Forecasting"
[15]: https://arxiv.org/abs/2410.09836?utm_source=chatgpt.com "Learning Pattern-Specific Experts for Time Series Forecasting Under Patch-level Distribution Shift"
[16]: https://arxiv.org/abs/2501.15125?utm_source=chatgpt.com "FreqMoE: Enhancing Time Series Forecasting through Frequency Decomposition Mixture of Experts"
[17]: https://openreview.net/pdf?id=KMCJXjlDDr&utm_source=chatgpt.com "Published as a conference paper at ICLR 2025"
[18]: https://arxiv.org/abs/2510.14814?utm_source=chatgpt.com "Tackling Time-Series Forecasting Generalization via Mitigating Concept Drift"
[19]: https://openreview.net/pdf/aab06f2eeb523320ae04338a644c41b159bdfcab.pdf?utm_source=chatgpt.com "Published as a conference paper at ICLR 2026"
[20]: https://openreview.net/forum?id=R20kKdWmVZ&utm_source=chatgpt.com "ModernTCN Revisited: A Critical Look at the Experimental Setup in General Time Series Analysis | OpenReview"
[21]: https://arxiv.org/abs/2407.13194?utm_source=chatgpt.com "Robust Multivariate Time Series Forecasting against Intra- and Inter-Series Transitional Shift"
[22]: https://arxiv.org/abs/2310.10688?utm_source=chatgpt.com "A decoder-only foundation model for time-series forecasting"
