# When and How: Learning Identifiable Latent States for Nonstationary Time Series Forecasting

**분석 기준:** 사용자가 첨부한 arXiv:2402.12767v2, 2024-04-13 버전을 1차 원문으로 사용하고, 2020년 이후 비교 연구는 NeurIPS·ICLR 공식 논문 페이지를 중심으로 웹 검색하여 교차 검토했습니다. 웹의 arXiv 레코드 역시 논문의 핵심 문제를 “분포 변화가 언제 발생하는지 알 수 없다”는 문제와 stationary/nonstationary latent state 식별 문제로 정의합니다.  ([arXiv][1])

---

# 1. Executive Summary — 10문장 이내

1. 이 논문은 **비정상 시계열에서 분포 변화가 언제 발생하는지 모르는 상황**에서 기존의 “균일하게 변화한다” 또는 “한 instance 내부는 stationary하다”는 가정을 완화하는 것을 목적으로 합니다. 
2. 저자들은 관측값 $x_t$의 잠재상태를 환경과 무관한 정상 상태 $z_t^s$와 환경 $e_t$에 의해 달라지는 비정상 상태 $z_t^e$로 분해하고, $e_t$를 Markov chain으로 모델링합니다. 
3. 핵심 이론은 먼저 $z_t^s$와 $z_t^e$를 **block-wise identifiable**하게 분리하고, 이어 환경 $e_t$를 label swapping까지 식별하며, 마지막으로 각 latent component를 component-wise하게 식별할 수 있다는 것입니다. 
4. 이를 실제 모델로 구현한 **IDEA**는 sequential variational inference, autoregressive HMM, stationary/nonstationary modular prior network, latent predictor와 future decoder를 결합합니다. 
5. 합성 데이터에서 IDEA의 latent-state MCC 평균은 **95.1**로 NCTRL의 80.4보다 크게 높았지만, environment estimation accuracy 자체는 IDEA 91.9/85.8%, NCTRL 91.1/85.0%, HMMICA 91.5/85.7%로 차이가 훨씬 작습니다.  
6. 8개 실세계 benchmark에서 저자들은 경쟁 방법 대비 대부분의 forecasting task에서 개선을 보고했으며, ILI·Weather·ECL처럼 비정상성이 큰 데이터에서 특히 강한 결과를 보입니다. 
7. 그러나 모든 설정에서 최고는 아니며, 예를 들어 Exchange 장기 예측에서는 저자 스스로 부정확한 future-environment estimation을 원인으로 지적했고 Traffic 일부 설정에서도 Koopa가 더 낮은 MSE를 냅니다. 
8. Ablation은 HMM 기반 환경 추정과 stationary/nonstationary prior를 제거하면 성능이 저하됨을 보여 주어 “**when을 찾는 것 + how를 분리하는 것**”이 둘 다 필요하다는 주장을 지지합니다. 
9. 다만 이론에는 알려진 환경 개수 $E$, full-rank transition matrix, 충분한 환경 간 변화, 조건부 독립성, invertible mixing 등의 강한 가정이 있고, 실험은 3개 random seed뿐이며 별도의 통계적 유의성 검정을 제시하지 않습니다.  
10. 따라서 IDEA의 가장 중요한 기여는 단순 normalization보다 **“변화 시점의 잠재적 regime을 찾고, 변하는 원인과 변하지 않는 원인을 식별 가능한 표현으로 분리한다”**는 방향이며, 이것이 실제 unseen-regime 일반화까지 보장된다고 보려면 추가적인 OOD·online·open-set 검증이 필요합니다.

---

# 1-1. 연구의 목적과 필요성

## 문제의 출발점

일반적인 forecasting model은 암묵적으로

$$
p_{\text{train}}(x_{t+1:T}\mid x_{1:t})
\approx
p_{\text{test}}(x_{t+1:T}\mid x_{1:t})
$$

와 같이 과거에서 학습한 관계가 미래에도 어느 정도 유지된다고 기대합니다.

하지만 현실에서는 평균, 분산뿐 아니라 **변수 간 관계 또는 잠재적인 생성 메커니즘 자체가 시간에 따라 달라질 수 있습니다.** 논문은 이를 temporal distribution shift의 핵심 문제로 보고 있으며, 이러한 변화가 forecasting model의 일반화를 방해한다고 명시합니다. 

**용어 메모 — Temporal distribution shift:**
시간이 흐름에 따라 $p(x)$ 또는 $p(y\mid x)$ 같은 데이터 생성 분포가 달라지는 현상입니다. 단순히 평균이 10에서 20으로 바뀌는 경우뿐 아니라, “어떤 입력이 미래를 결정하는가” 자체가 달라지는 경우도 포함합니다.

기존 접근은 크게 두 종류로 설명됩니다.

첫째, RevIN 같은 방법은 각 입력 instance가 내부적으로 비교적 stationary하다고 보고 평균·분산을 제거한 뒤 예측 후 다시 복원합니다. 둘째, 일부 방법은 시간에 따른 shift가 일정하거나 균일한 형태로 나타난다고 보고 stationarization 또는 고정된 구간 분할을 적용합니다. 논문은 실제 shift boundary를 알 수 없기 때문에 이러한 가정이 실패할 수 있다고 주장합니다. 

Figure 1이 바로 이 문제를 직관적으로 보여 줍니다. 실제 amplitude가 특정 시점에 바뀌는데 데이터를 임의의 equal-size segment로 자르면 하나의 segment 안에 두 regime이 섞일 수 있고, 설령 변화 시점을 맞혀도 amplitude와 vertical shift 같은 latent factor가 섞여 있으면 잘못된 미래를 생성할 수 있습니다. 

따라서 이 논문의 질문은 두 단계입니다.

> **When:** 분포가 **언제** 바뀌었는가?
> **How:** 무엇이 변했고 무엇은 변하지 않았는가?

이 둘을 동시에 해결하려는 것이 IDEA의 연구 목적입니다.

---

# 2. 핵심 주장과 근거

| 핵심 주장                                               | 저자가 제시한 근거                                                               | 위치                         | 평가                                     |
| --------------------------------------------------- | ------------------------------------------------------------------------ | -------------------------- | -------------------------------------- |
| 균일한 distribution shift 가정은 현실적으로 지나치게 강하다           | equal-size segmentation이 실제 regime boundary와 일치하지 않는 toy example         | **p.2, Fig.1(a)**          | 문제 제기는 설득력이 높음                         |
| shift timing만 찾는 것도 충분하지 않다                         | 환경을 맞혀도 latent amplitude/vertical shift가 entangled되면 편향 발생               | **p.2, Fig.1(b)**          | IDEA의 disentanglement 필요성을 설명          |
| stationary와 nonstationary latent subspace를 분리할 수 있다 | 조건부 시간 의존성의 비대칭성과 A1–A3를 사용한 Theorem 3.1                                 | **p.3, Thm.3.1**           | 이론적 기여이나 가정 충족 여부가 핵심                  |
| latent environment도 식별 가능하다                         | known $E$, full-rank $A$, linearly independent emission을 이용한 Theorem 3.2 | **p.4, Thm.3.2**           | label swapping까지 식별                    |
| 각 latent component까지 식별할 수 있다                       | nonlinear ICA 조건을 사용한 Thm.3.3–3.4                                        | **p.4, Eq.7–8**            | 강한 variability/independence 조건 필요      |
| 실제 모델에서 environment estimation이 forecasting에 중요하다   | HMM을 제거한 IDEA-H의 ILI 성능 저하                                               | **p.24, Fig.4**            | ablation으로 직접 뒷받침                      |
| stationary/nonstationary prior 모두 필요하다              | IDEA-S, IDEA-E가 full IDEA보다 악화                                           | **p.24, Fig.4**            | 구조적 prior가 단순 장식은 아님                   |
| latent identification 성능이 높다                        | synthetic MCC: IDEA 95.1, NCTRL 80.4                                     | **p.7, Table 1**           | 가장 강한 실험적 증거 중 하나                      |
| 실세계 forecasting이 개선된다                               | ETTh1/2, Exchange, ILI, Weather, Traffic, ECL, M4                        | **pp.8, 24, Tables 2 & 5** | 다수 task에서 우수하지만 전부 최고는 아님              |
| 계산 효율도 경쟁력이 있다                                      | training time/memory 비교                                                  | **p.28, Fig.7**            | ⚠️ Weather에서는 IDEA 메모리가 10.64 GB로 매우 큼 |
| 장기 예측 시 환경 추정 오차가 문제가 될 수 있다                        | Exchange 장기 forecast의 second-best 결과를 환경 추정 오차로 설명                       | **p.8**                    | 핵심 한계를 저자가 직접 인정                       |

---

# 2-1. 해결하고자 하는 문제와 제안 방법

## 2-1-1. 관측값 자체를 바로 stationary하게 만들지 않고 “원인”을 분리합니다

논문은 다음 생성 모델에서 시작합니다.

$$
x_t=g(z_t), \qquad
z_t=\{z_t^s,z_t^e\}.
$$



여기서

* $x_t$: 시간 $t$의 관측 시계열 값,
* $z_t$: 직접 관측되지 않는 latent state,
* $z_t^s\in\mathbb R^{n_s}$: **stationary latent state**,
* $z_t^e\in\mathbb R^{n_e}$: 환경에 따라 달라지는 **nonstationary latent state**,
* $n_s,n_e$: 두 latent block의 차원,
* $g(\cdot)$: latent state를 관측값으로 섞는 invertible nonlinear mixing function입니다.

**용어 메모 — Latent state:**
실제로 측정되지는 않지만 관측 데이터를 만들어 내는 내부 상태입니다. 예를 들어 장비 출력만 관측할 때 실제 장비 degradation 상태, 운전 regime 등이 latent state가 될 수 있습니다.

**용어 메모 — Identifiability:**
관측 데이터만 보고 latent variable을 학습했을 때 서로 완전히 다른 여러 해가 똑같이 가능하지 않고, 허용된 단순한 모호성 안에서 “진짜 latent factor”를 복원할 수 있다는 성질입니다. 좋은 예측 성능과는 별개의 개념입니다.

---

## 2-1-2. Stationary latent state

논문의 Equation (2)은 stationary state가 과거의 stationary state와 독립 noise로 생성된다고 설정합니다.

```math
z_{t,i}^{s}
=
f_i^s
\left(
\left\{
z_{t-\tau,k}^{s}
\mid
z_{t-\tau,k}^{s}\in
\text{Pa}(z_{t,i}^{s})
\right\},
\epsilon_{t,i}^{s}
\right),
\qquad
\epsilon_{t,i}^{s}\sim p_{\epsilon_i^s}.
```

기호는 다음과 같습니다.

* $i$: stationary latent component 번호,
* $\tau$: time lag,
* $f_i^s$: $i$번째 stationary component의 비선형 transition function,
* $\text{Pa}(z_{t,i}^{s})$: $z_{t,i}^{s}$에 직접 영향을 주는 과거 latent parent들의 집합,
* $\epsilon_{t,i}^{s}$: 시간·공간적으로 독립이라고 가정되는 disturbance/noise,
* $p_{\epsilon_i^s}$: 해당 noise의 확률분포입니다.

여기서 “stationary”는 값이 일정하다는 의미가 아닙니다. $z_t^s$는 시간에 따라 계속 움직일 수 있지만 **environment $e_t$가 바뀌어도 그 transition mechanism 자체는 바뀌지 않는다**는 의미에 가깝습니다.

---

## 2-1-3. Nonstationary latent state와 latent environment

반면 nonstationary latent state는

$$
e_1,e_2,\ldots,e_T
\sim
\text{Markov Chain}(A),
$$

```math
z_{t,j}^{e}
=
f_j^e(e_t,\epsilon_{t,j}^{e}),
\qquad
\epsilon_{t,j}^{e}\sim p_{\epsilon_j^e}.
```

로 구성됩니다. 

여기서

* $e_t\in{1,\ldots,E}$: 시간 $t$의 latent environment 또는 regime,
* $E$: 가능한 environment 개수,
* $A\in\mathbb R^{E\times E}$: environment transition matrix,
* $A_{kl}=P(e_t=l\mid e_{t-1}=k)$,
* $f_j^e$: environment와 noise로 $j$번째 nonstationary latent state를 만드는 bijection,
* $\epsilon_{t,j}^{e}$: mutually independent noise입니다.

**용어 메모 — Markov chain:**
현재 상태 $e_t$가 주어졌다면 다음 상태 $e_{t+1}$을 결정할 때 더 오래된 과거 전체를 직접 기억하지 않는 모델입니다.

```math
P(e_{t+1}\mid e_t,e_{t-1},\ldots)
=
P(e_{t+1}\mid e_t).
```

IDEA에서는 이 discrete state가 “지금 어느 distribution regime에 속해 있는가”를 표현합니다.

---

# 2-1-4. 왜 이 분리가 forecasting 문제를 해결하는가

전체 관측 확률분포는 Equation (4)에서

$$
\begin{aligned}
p(x)
&=
\sum_e
\int
\int
p(x,e,z^e,z^s)\,dz^e\,dz^s \\[2mm]
&=
\sum_e
\int
\int
p(x\mid z^e,z^s)
p(z^e\mid e)
p(e)
p(z^s)
\,dz^e\,dz^s .
\end{aligned}
$$

로 분해됩니다. 

즉 IDEA가 해결해야 할 대상은 네 가지입니다.

$$
\boxed{
p(x\mid z^e,z^s),\quad
p(z^s),\quad
p(z^e\mid e),\quad
p(e)
}
$$

관측 생성, 정상 dynamics, environment-dependent dynamics, environment transition을 따로 학습합니다.

이 구조가 중요한 이유는 미래 예측이 단순히

$$
x_{\text{past}}\rightarrow x_{\text{future}}
$$

라는 하나의 black-box mapping이 아니라,

$$
x_{\text{past}}
\rightarrow
(z^s,z^e,e)
\rightarrow
(z_{\text{future}}^s,z_{\text{future}}^e,e_{\text{future}})
\rightarrow
x_{\text{future}}
$$

로 분해되기 때문입니다.

---

# 2-1-5. 식별가능성 이론의 핵심

## A. Block-wise identifiability — “어느 latent가 stationary인가?”

Theorem 3.1의 핵심 논리는 조건부 시간 의존성의 비대칭입니다.

논문의 proof sketch를 직관적으로 요약하면,

$$
z_t^s
\perp
z_{t-2}
\mid z_{t-1},
$$

인 반면 nonstationary component에는 environment의 시간 구조 때문에 더 오래된 정보와의 의존성이 남습니다. 이를 충분한 variability 조건과 결합하면 추정 latent representation 안에서 stationary block과 nonstationary block이 서로 임의로 섞일 수 없음을 보입니다. 

**용어 메모 — Block-wise identifiability:**
각 개별 변수의 정확한 이름까지 찾기 전에, 적어도 “이 변수들은 stationary group, 저 변수들은 nonstationary group”이라고 두 subspace를 분리할 수 있다는 의미입니다.

---

## B. Environment identifiability — “언제 regime이 바뀌었는가?”

Theorem 3.2에서는 다음 핵심 가정을 둡니다.

$$
E \text{ is known},
$$

$$
\text{rank}(A)=E,
$$

그리고 서로 다른 environment가 만드는

$$
\mu_e=p(z_t^e\mid e_t=e)
$$

가 충분히 linearly independent해야 합니다. 이때

$$
p(e_1,\ldots,e_t)
$$

를 **label swapping까지 식별할 수 있다**고 증명합니다. 

**용어 메모 — Label swapping:**
실제 regime이 ${A,B,C}$인데 모델이 이를 ${2,3,1}$이라고 부르는 것은 문제가 아닙니다. 이름만 바뀌었을 뿐 구조는 같습니다. 따라서 “up to label swapping”은 이런 순열 모호성만 남는다는 의미입니다.

---

## C. Component-wise identifiability — “그 안에서 각각 무엇이 변하는가?”

Theorem 3.3과 3.4는 nonlinear ICA의 조건을 사용하여 $z_t^s$ 및 $z_t^e$ 내부의 각 component까지 분리합니다. 

**용어 메모 — Nonlinear ICA:**
여러 독립 또는 조건부 독립 원인이 비선형적으로 섞여 관측될 때, 그 원래 원인을 다시 분리하려는 방법론입니다. 일반 nonlinear ICA는 추가 가정 없이는 식별 불가능하기 때문에 시간 구조, environment, auxiliary variable 등의 정보가 필요합니다.

여기서 특히 주의해야 할 것이 A12입니다. 논문은 nonstationary component의 component-wise identification을 위해 충분히 다양한 environment 조건을 요구하며, 식에서 $2n_e+1$개의 환경 값이 등장합니다. 

반면 실제 sensitivity experiment에서는 환경 개수 $E$를 hyperparameter로 두고 실험하며 실무 설정에서 $E=4$를 사용한다고 밝힙니다. 

**중요한 해석:** 논문이 실제 benchmark에서 선택한 $n_e$와 $E=4$가 모든 이론 조건을 실제로 만족하는지를 별도로 입증하지는 않습니다. 따라서 **“이론적 identifiability theorem이 실제 benchmark 학습에서도 그대로 적용된다”는 연결은 실험적으로 완전히 증명되지 않았습니다.**

---

# 2-1-6. 실제 IDEA 모델의 목적함수

IDEA는 VAE 계열 sequential variational inference를 사용합니다.

핵심 ELBO는 논문의 Equation (9)입니다.

```math
\mathcal L_{\text{ELBO}}
=
\mathcal L_{\text{pre}}
+
\alpha
\mathbb E_{q(z^e_{1:t}\mid x_{1:t})}
\mathbb E_{q(z^s_{1:t}\mid x_{1:t})}
\mathcal L_{\text{rec}}
-
\beta\mathcal L_{\text{KLD}}^s
-
\gamma\mathcal L_{\text{KLD}}^e.
```

기호는

* $\mathcal L_{\text{pre}}$: 미래 forecasting log-likelihood 항,
* $\mathcal L_{\text{rec}}$: 과거 reconstruction 항,
* $\mathcal L_{\text{KLD}}^s$: stationary posterior와 stationary prior 사이의 KL divergence,
* $\mathcal L_{\text{KLD}}^e$: nonstationary posterior와 environment-conditioned prior 사이의 KL divergence,
* $\alpha,\beta,\gamma$: 각 목적항의 중요도를 조절하는 hyperparameter입니다.

**용어 메모 — ELBO:**
직접 계산하기 어려운 likelihood $\log p(x)$를 최적화하기 위해 사용하는 계산 가능한 lower bound입니다. VAE에서 reconstruction과 latent regularization을 동시에 학습시키는 기본 목적함수입니다.

**용어 메모 — KL divergence:**

$$
D_{\text{KL}}(q\|p)
$$

는 두 확률분포 $q$와 $p$가 얼마나 다른지를 측정하는 비대칭적인 거리 개념입니다. 여기서는 encoder가 만든 posterior latent distribution을 IDEA가 원하는 structured prior에 맞추는 역할을 합니다.

---

## Reconstruction과 prediction

Equation (10)은 두 가지 데이터 적합 항을 분리합니다.

```math
\mathcal L_{\text{rec}}
=
\mathbb E_{q(z^e_{1:t}\mid x_{1:t})}
\mathbb E_{q(z^s_{1:t}\mid x_{1:t})}
\log p(x_{1:t}\mid z^e_{1:t},z^s_{1:t}),
```

```math
\mathcal L_{\text{pre}}
=
\mathbb E_{q(z^e_{t:T}\mid z^e_{1:t})}
\mathbb E_{q(z^s_{t:T}\mid z^s_{1:t})}
\log p(x_{t+1:T}\mid
z^e_{t+1:T},z^s_{t+1:T}).
```



첫 번째 항은 “latent representation으로 과거를 제대로 복원할 수 있는가?”, 두 번째는 “그 latent dynamics를 미래로 진행시켰을 때 실제 미래를 잘 맞히는가?”를 동시에 요구합니다.

---

## Structured KL regularization

stationary latent에 대해서는

$$
\begin{aligned}
\mathcal L_{\text{KLD}}^s
=&\;
D_{\text{KL}}
\left(
q(z^s_{1:t}\mid x_{1:t})
\|
p(z^s_{1:t})
\right) \\
&+
\mathbb E_{q(z^s_{1:t}\mid x_{1:t})}
\left[
D_{\text{KL}}
\left(
q(z^s_{t+1:T}\mid z^s_{1:t})
\|
p(z^s_{t+1:T}\mid z^s_{1:t})
\right)
\right],
\end{aligned}
$$

nonstationary latent에는

$$
\begin{aligned}
\mathcal L_{\text{KLD}}^e
=&\;
D_{\text{KL}}
\left(
q(z^e_{1:t}\mid x_{1:t})
\|
p(z^e_{1:t})
\right) \\
&+
\mathbb E_{q(z^e_{1:t}\mid x_{1:t})}
\left[
D_{\text{KL}}
\left(
q(z^e_{t+1:T}\mid z^e_{1:t})
\|
p(z^e_{t+1:T}\mid z^e_{1:t})
\right)
\right].
\end{aligned}
$$

을 사용합니다. 

즉 단순 VAE의 isotropic Gaussian prior가 아니라, **시간 동역학과 environment를 반영한 prior**로 latent를 정렬하려는 것입니다.

---

# 2-1-7. Modular prior가 중요한 이유

저자들은 단순 Gaussian prior가 실제 temporal dynamics를 표현하기 어렵다고 보고 inverse transition network $r_i^s,r_i^e$를 학습합니다. 

stationary side에서는

```math
\hat\epsilon^s_{t,i}
=
r_i^s
\left(
\hat z^s_{t,i},
\hat z^s_{t-1}
\right)
```

를 구성합니다.

변수 변환 공식에 따라

```math
\log p(\hat z^s_{t-1},\hat z^s_t)
=
\log p(\hat z^s_{t-1},\hat\epsilon^s_t)
+
\log |\det J_{\phi^s}|.
```



nonstationary side에서는

```math
\hat\epsilon^e_{t,i}
=
r_i^e(\hat e_t,\hat z^e_{t,i})
```

이고,

```math
\log p(\hat z_t^e\mid\hat e_t)
=
\log p(\hat\epsilon_t^e)
+
\sum_{i=1}^{n_e}
\log
\left|
\frac{\partial r_i^e}
{\partial\hat z_{t,i}^e}
\right|.
```

으로 environment-specific prior를 만듭니다. 

**용어 메모 — Jacobian determinant:**
변수 $z$를 $\epsilon$으로 비선형 변환할 때 확률밀도가 얼마나 압축·팽창되는지를 보정합니다. Normalizing flow의 change-of-variables와 같은 원리입니다.

---

# 2-1-8. 전체 모델 구조

Figure 3은 IDEA를 다음 흐름으로 정리합니다. 

$$
x_{1:t}
\overset{\psi_s,\psi_e}{\longrightarrow}
(\hat z^s_{1:t},\hat z^e_{1:t})
$$

$$
\hat z^s_{1:t}
\overset{T_s}{\longrightarrow}
\hat z^s_{t+1:T},
\qquad
\hat z^e_{1:t}
\overset{T_e}{\longrightarrow}
\hat z^e_{t+1:T},
$$

$$
(\hat z^s_{t+1:T},\hat z^e_{t+1:T})
\overset{F_y}{\longrightarrow}
\hat x_{t+1:T}.
$$

이를 Equation (12)로 쓰면

```math
z^e_{1:t}
=
\psi_e(x_{1:t};\theta_{\psi_e}),
\qquad
z^s_{1:t}
=
\psi_s(x_{1:t};\theta_{\psi_s}),
```

```math
z^e_{t+1:T}
=
T_e(z^e_{1:t};\theta_{T_e}),
\qquad
z^s_{t+1:T}
=
T_s(z^s_{1:t};\theta_{T_s}),
```

```math
\hat x_{1:t}
=
F_x(z^e_{1:t},z^s_{1:t};\theta_x),
```

```math
\hat x_{t+1:T}
=
F_y(z^e_{t+1:T},z^s_{t+1:T};\theta_y).
```



$\psi_s,\psi_e,T_s,T_e,F_x,F_y$는 모두 MLP 계열로 구현됩니다.

Appendix Table 9를 보면 stationary encoder는 384-neuron dense layer를, nonstationary encoder는 384→128→384 계층을 사용하며, modular prior는 128-unit layer 여러 개와 Jacobian computation으로 구성됩니다. 

---

## Environment branch

별도의 autoregressive HMM이

$$
x_{1:t}
\rightarrow
\hat e_{1:t}
$$

를 추정하고 Viterbi algorithm을 이용해 latent environment index를 결정합니다. 학습된 history로 transition matrix $\hat A$를 추정한 뒤 test에서는 $\hat A$를 사용하여 미래 environment $\hat e_{t+1:T}$를 생성합니다. 

**용어 메모 — Viterbi algorithm:**
HMM에서 관측 sequence가 주어졌을 때 가장 가능성이 높은 hidden-state sequence를 동적계획법으로 찾는 알고리즘입니다.

훈련은 **two-phase**입니다.

$$
\text{Phase 1: train AR-HMM}
\quad\rightarrow\quad
\text{Phase 2: freeze HMM and train variational model}.
$$

이 설계는 안정적인 optimization에는 유리하지만, 뒤에서 설명하듯 일반화 관점에서는 joint adaptation을 제한할 수 있습니다.

---

# 3. 연구 주제·방법·결과: 저자 보고와 해석 분리

| 항목                    | **저자가 직접 보고한 내용**                                                                                     | **제 해석**                                                                           |
| --------------------- | ----------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| 연구 주제                 | temporal distribution shift가 언제 발생하는지 알 수 없으므로 environment와 stationary/nonstationary latent state를 식별 | “분포를 stationary하게 만든다”보다 “분포를 만드는 latent mechanism을 찾는다”는 causal representation 접근 |
| When                  | AR-HMM으로 latent $e_t$를 추정                                                                             | discrete change-point/regime inference 문제로 볼 수 있음                                  |
| How                   | $z_t^s$와 $z_t^e$를 분리하고 각각 structured prior 사용                                                         | invariant mechanism과 variant mechanism을 분리하여 미래 regime 변화에 더 강한 predictor를 만들려는 설계 |
| 이론                    | Thm.3.1–3.4에서 block-wise, environment, component-wise identifiability 증명                              | 매우 중요한 강점이나 가정 충족 범위에서만 성립                                                         |
| Synthetic             | MCC 평균 IDEA 95.1, NCTRL 80.4                                                                          | environment detection보다 latent disentanglement 측면의 이점이 더 뚜렷함                       |
| Environment detection | IDEA accuracy A/B = 91.9/85.8%                                                                        | HMMICA 91.5/85.7%, NCTRL 91.1/85.0%라서 environment accuracy 자체는 압도적이라고 보기 어려움       |
| 실세계 forecasting       | 대부분의 task에서 경쟁 방법보다 낮은 MSE/MAE                                                                        | 비정상성이 강한 ILI·Weather·ECL에서 특히 의미가 있으나 모든 task에서 SOTA는 아님                           |
| Generalization        | distribution shifts 대응 및 real-world forecasting 개선을 주장                                                | **explicit unseen-environment generalization experiment는 아님**                      |
| 효율성                   | 상대적으로 좋은 performance/efficiency trade-off 주장                                                          | 데이터에 따라 memory cost가 커지므로 “항상 가볍다”고 해석하면 안 됨                                       |

Synthetic MCC 수치는 p.7 Table 1에, environment accuracy/MSE는 p.23 Table 4에 있습니다.  

---

# 4. 성능 향상 결과를 정확하게 읽기

## Synthetic latent recovery

Table 1:

| Method   | Dataset A MCC | Dataset B MCC |  Average |
| -------- | ------------: | ------------: | -------: |
| BetaVAE  |          64.2 |          63.2 |     63.7 |
| i-VAE    |          76.9 |          73.0 |     74.9 |
| HMNLICA  |          83.2 |          74.5 |     78.8 |
| TDRL     |          78.5 |          78.8 |     78.6 |
| NCTRL    |          81.4 |          79.4 |     80.4 |
| **IDEA** |      **97.5** |      **92.7** | **95.1** |



**[저자 보고]** IDEA가 알려지지 않은 temporal shift에서도 latent variable을 잘 복원한다고 평가합니다.

**[해석]** IDEA의 가장 강한 evidence는 forecasting MSE보다는 오히려 이 결과입니다. NCTRL 대비 평균 MCC가 **14.7 percentage points** 높기 때문에 latent factor identification이라는 논문의 중심 주장과 직접 연결됩니다.

---

## Environment inference는 조금 다른 이야기입니다

Table 4:

| Model  | Dataset A Accuracy | A transition MSE | Dataset B Accuracy | B transition MSE |
| ------ | -----------------: | ---------------: | -----------------: | ---------------: |
| IDEA   |           **91.9** |       **0.0103** |           **85.8** |           0.0163 |
| NCTRL  |               91.1 |           0.0115 |               85.0 |       **0.0160** |
| HMMICA |               91.5 |       **0.0103** |               85.7 |           0.0167 |



따라서

> “IDEA가 environment까지 압도적으로 더 잘 찾는다”

라고 쓰는 것은 원문 수치보다 강한 주장입니다.

Dataset B transition MSE에서는 **NCTRL 0.0160 < IDEA 0.0163**입니다.

IDEA의 진짜 advantage는 단순 environment detection보다 **environment를 이용해 stationary/nonstationary representation을 분리하는 전체 시스템**에서 나타납니다.

---

# 5. 실세계 benchmark 결과

논문은 ETT, Exchange, ILI, Weather, ECL, Traffic, M4를 포함한 8개 real-world benchmark를 사용합니다. 일반 데이터에서는 lookback을 $3H$로 설정하고 여러 forecast horizon을 평가했으며 실험을 **3개 random seed**에서 반복하여 평균을 사용했습니다. 

대표적인 MSE를 보면:

| Task            |      IDEA |            강한 비교 모델 | 해석          |
| --------------- | --------: | ------------------: | ----------- |
| ETTh1 36→12     |     0.291 |          MICN 0.292 | 거의 동률       |
| ETTh2 216→72    | **0.262** |         Koopa 0.283 | IDEA 우위     |
| ILI 36→12       | **1.218** | N-Transformer 1.491 | 큰 개선        |
| Weather 36→12   | **0.072** |    Koopa/MICN 0.076 | 개선          |
| ECL 36→12       | **0.114** |      TimesNet 0.128 | 개선          |
| Traffic 72→24   |     0.458 |     **Koopa 0.450** | IDEA가 최고 아님 |
| Exchange 216→72 |     0.065 |      **MICN 0.064** | IDEA가 최고 아님 |



ILI 36→12에서는 당시 다음으로 좋은 N-Transformer 대비 MSE 감소율이 대략

$$
\frac{1.491-1.218}{1.491}
\approx 18.3\%
$$

입니다.

반면 ETTh1 36→12의 MICN 대비 차이는

$$
\frac{0.292-0.291}{0.292}
\approx0.34\%
$$

뿐입니다.

따라서 **dataset에 따라 IDEA 효과의 크기가 상당히 다릅니다.**

---

# 5-1. 통계적으로 취약한 부분

이 부분은 논문 해석에서 매우 중요합니다.

### ⚠️ 1. Random seed가 3개뿐입니다

저자들은 실세계 실험을 3개 random seed로 반복했다고 명시합니다. 

Appendix Table 8에는 mean과 variance가 보고되어 있지만 **formal hypothesis test, confidence interval, Diebold–Mariano test 또는 multiple-comparison correction은 없습니다.** 

따라서 본문에서 사용하는 “significantly outperforms”는 **통계적 유의성 검정으로 확인된 significant라고 해석하면 안 됩니다.**

---

### ⚠️ 2. 아주 작은 성능 차이가 있습니다

예를 들어 Table 8의 ETTh1 12→36 MSE는

$$
\text{IDEA}=0.2913\;(0.0012),
$$

$$
\text{MICN}=0.2916\;(0.0070)
$$

입니다. 

평균 차이는 0.0003에 불과합니다.

Table 8 caption은 괄호 값을 **variance**라고 명시하기 때문에 이를 표준편차나 confidence interval로 직접 읽어서는 안 됩니다. 3 seed만으로 어느 모델이 통계적으로 확실히 우수한지 결론내리기도 어렵습니다.

---

### ⚠️ 3. 많은 dataset × horizon × metric을 비교하지만 multiplicity가 고려되지 않습니다

수십 개 MSE/MAE 값을 비교하면 우연히 일부 setting에서 가장 좋은 값이 나올 가능성도 증가합니다. 그러나 family-wise test나 false discovery rate와 같은 보정은 보고되지 않습니다.

---

### ⚠️ 4. Environment evaluation은 최적 permutation을 사후 선택합니다

Environment label 자체는 permutation-invariant하므로 이 방식은 clustering 평가에서는 합리적입니다. 다만 저자들은 가능한 permutation을 모두 검토하여 가장 좋은 assignment를 선택했다고 명시합니다. 

따라서 이 accuracy는 raw label accuracy와 같은 의미는 아닙니다.

---

### ⚠️ 5. $3H$ lookback 설계가 IDEA에 유리할 가능성

저자들은 lookback을 $3H$로 설정한 이유를

> 각 environment type이 lookback에 포함되도록 하기 위해

라고 설명합니다. 

이것은 모델 설계상 이해되는 선택이지만 실제 배포에서는 새 regime이 과거 window에 전혀 나타나지 않을 수 있습니다.

즉 논문 실험은 **open-set regime generalization**을 직접 시험하지 않습니다.

---

# 5-2. 서로 직접 비교하면 안 되는 수치

| 숫자                         | 의미                              | 직접 비교 가능 여부                    |
| -------------------------- | ------------------------------- | ------------------------------ |
| MCC 95.1                   | synthetic latent recovery       | 다른 MCC끼리만 비교                   |
| Environment Accuracy 91.9% | regime label recovery           | MCC와 비교 불가                     |
| Transition MSE 0.0103      | $\hat A$와 true $A$ 차이           | forecasting MSE와 비교 불가         |
| Forecast MSE 0.291         | 관측 시계열 예측 오차                    | 같은 dataset/normalization에서만 비교 |
| MAE 0.345                  | 절대오차                            | MSE와 숫자 크기 직접 비교 불가            |
| M4 sMAPE 11.838            | scale-normalized percentage 계열  | ETT MSE와 비교 불가                 |
| M4 MASE 1.483              | naïve forecast 대비 scaled error  | MSE와 비교 불가                     |
| M4 OWA 0.849               | M4 competition composite metric | 다른 metric과 직접 비교 불가            |

M4에서 IDEA의 평균은 sMAPE 11.838, MASE 1.483, OWA 0.849로 우수하지만, 이것을 Table 2의 MSE 개선률과 한 숫자로 합치는 것은 부적절합니다. 

---

# 6. 모델의 한계

## 6-1. 알려진 환경 개수 $E$

Theorem 3.2의 A4는

$$
E\quad\text{is known}
$$

을 요구합니다. 

실제로 sensitivity analysis에서도 $E$를 hyperparameter로 취급하며 선택 값에 따라 성능이 달라집니다. 

현실에서는 “공정 regime이 정확히 몇 개인가?” 자체가 알려지지 않는 경우가 많기 때문에 중요한 한계입니다.

---

## 6-2. First-order Markov assumption

```math
P(e_{t+1}\mid e_{1:t})
=
P(e_{t+1}\mid e_t)
```

라고 보기 어려운 장기 cycle, duration dependency, gradual degradation가 있으면 HMM representation이 잘못될 수 있습니다.

예를 들어 어떤 regime이 최소 200 step 유지되는 물리적 의미가 있다면 **Hidden Semi-Markov Model**처럼 duration을 명시하는 모델이 더 적절할 수 있습니다.

---

## 6-3. 완전히 새로운 future environment

Test 시 future environment는 학습한

$$
\hat A
$$

에서 생성됩니다. 

그러므로 training에 없던 environment $e_{\text{new}}$가 갑자기 등장하면 현재 구조에는 이를 생성할 state 자체가 없습니다.

즉 IDEA는

> **known-regime transition generalization**

에는 적합하지만,

> **unseen-regime discovery/generalization**

까지 해결했다고 볼 수 없습니다.

---

## 6-4. Long-horizon environment uncertainty

저자 스스로 Exchange의 긴 horizon에서 environment estimation의 부정확성 때문에 second-best 결과가 나타났을 가능성을 언급합니다. 

이것은 구조적으로 자연스럽습니다.

$$
\hat e_{t+1}
\rightarrow
\hat e_{t+2}
\rightarrow
\cdots
\rightarrow
\hat e_{t+H}
$$

처럼 horizon이 길어질수록 environment uncertainty가 누적될 수 있기 때문입니다.

---

## 6-5. 이론적 identifiability $\neq$ forecasting generalization guarantee

이 점이 가장 중요합니다.

Identifiability theorem은

> 충분한 조건 아래 latent generating factors를 복원할 수 있는가?

에 대한 정리입니다.

하지만 일반화 성능은

```math
R_{\text{future}}
=
\mathbb E_{p_{\text{future}}}
[
L(Y,\hat Y)
]
```

이 얼마나 작으냐의 문제입니다.

논문에는

$$
\text{identifiability}
\Rightarrow
R_{\text{OOD}}\text{ 감소}
$$

라는 직접적인 forecasting-risk theorem이 없습니다.

후속 연구인 **TOT(2025)**는 오히려 이 빈 부분을 직접 다룹니다. TOT는 latent variable을 forecaster에 제공할 때 Bayes risk가 더 tight해지고, latent state의 identifiability가 좋아질수록 이점이 커진다는 이론을 제시합니다. 이는 IDEA가 제시한 연구 방향을 forecasting generalization theory 쪽으로 더 발전시킨 것으로 해석할 수 있습니다. ([NeurIPS Proceedings][2])

---

# 7. 가장 중요한 그림 5개 해석

## Figure 1 — 논문의 전체 문제를 한 장으로 설명합니다

**위치: p.2**

(a)는 equal-size segmentation이 실제 change point와 일치하지 않으면 서로 다른 regime이 한 segment에 섞인다는 것을 보여 줍니다.

(b)는 environment boundary를 정확히 찾더라도 amplitude와 vertical shift가 서로 entangled되면 엉뚱한 latent factor까지 변화한다고 판단해 bias가 생기는 것을 보여 줍니다.

(c)는

$$
\boxed{\text{Correct When}+\text{Correct How}}
$$

가 동시에 필요하다는 논문의 중심 메시지입니다. 

**제 해석:** IDEA를 이해할 때 가장 중요한 그림입니다. 이 논문은 단순 change-point detection 논문도 아니고 단순 representation disentanglement 논문도 아닙니다.

---

## Figure 2 — IDEA가 믿는 세계의 인과 구조입니다

**위치: p.2**

환경

$$
e_t
$$

는 nonstationary latent state

$$
z_t^e
$$

에 영향을 주고, stationary latent

$$
z_t^s
$$

는 자체 시간적 dynamics를 유지하며, 두 latent block이 관측 $x_t$를 생성합니다. 

**제 해석:** IDEA의 성능 이전에 이 그래프가 실제 응용 분야에 적합한지를 검토해야 합니다. 실제 문제에서 환경이 stationary factor에도 영향을 준다면 모델의 구조적 assumption이 깨집니다.

---

## Figure 3 — 이론을 신경망으로 구현한 지도입니다

**위치: p.5**

왼쪽의 Environment Estimator가 “When”을 담당하고,

stationary/nonstationary encoder와 prior가 “How”를 담당합니다.

그 이후 latent prediction module과 future predictor가 실제 forecasting을 수행합니다. 

**제 해석:** IDEA의 핵심은 forecasting backbone 자체가 특별히 복잡해서가 아닙니다. 실제 network는 상당 부분 MLP이고, 성능상의 novelty는 **latent-space structural constraints**에 있습니다.

---

## Figure 4 — “각 구성요소가 정말 필요한가?”

**위치: p.24**

ILI에서

* IDEA-H: HMM 제거,
* IDEA-S: stationary prior 제거,
* IDEA-E: nonstationary prior 제거

를 비교합니다.

Full IDEA가 전반적으로 더 좋으며, 저자들은 정확한 environment estimation과 두 prior 모두 forecasting에 중요하다고 결론 내립니다. 

**제 해석:** 가장 중요한 empirical causal check입니다. 단순히 parameter 수를 늘린 효과보다는 구조적 구성요소가 성능에 영향을 주고 있음을 지지합니다. 다만 ILI 한 데이터셋 중심의 ablation이므로 모든 데이터에 일반화한다고 단정할 수는 없습니다.

---

## Figure 7 — 성능과 계산량 사이의 실제 trade-off

**위치: p.28**

Exchange에서는 IDEA가 약 2.2 GB, 8.94 s로 비교적 좋은 trade-off를 보입니다.

반면 Weather에서는 표기된 IDEA memory footprint가 **10.64 GB**, training time 126.92 s로, DLinear의 1.36 GB/47.86 s보다 훨씬 무겁습니다. 

본문도 IDEA가 MLP 기반이라 경쟁력이 있지만 MICN·DLinear 대비 효율이 떨어질 수 있으며 이는 latent-variable-wise prior 때문이라고 설명합니다. 

**제 해석:** “IDEA가 높은 성능과 높은 효율을 모두 갖는다”는 문구는 데이터별로 나눠 읽어야 합니다. High-dimensional multivariate application에서는 modular prior의 메모리 확장이 주요 engineering bottleneck이 될 가능성이 있습니다.

---

### 참고할 만한 Figure 5

p.26 Figure 5에서는 IDEA와 여러 baseline의 예측곡선을 비교하고, 저자들은 IDEA가 changing amplitude를 더 잘 추적한다고 설명합니다. 

이 그림은 qualitative evidence이므로 Table 2·8의 정량적 결과보다 증거 강도는 낮게 보는 것이 적절합니다.

---

# 8. 모델의 일반화 성능 향상 가능성

이 논문에서 가장 연구 가치가 높은 부분입니다.

## 현재 IDEA가 일반화에 유리할 수 있는 이유

일반적인 모델이

$$
\hat y=f_\theta(x)
$$

하나를 학습한다고 합시다.

분포가 바뀌어 실제 mapping이

$$
f_{\theta_1}
\rightarrow
f_{\theta_2}
$$

로 변하면 하나의 global model은 평균적인 관계를 학습하기 쉽습니다.

IDEA에서는 이를

$$
x_t
\rightarrow
(z_t^s,z_t^e,e_t)
$$

로 바꿉니다.

따라서 안정적인 mechanism은

$$
z_t^s
$$

에 보존하고, 환경에 따라 변해야 하는 부분만

$$
z_t^e\mid e_t
$$

로 보내는 것이 목표입니다.

이 분리가 정확하다면 training regime의 우연한 correlation을 전부 하나의 predictor에 집어넣는 것보다 미래 shift에 강할 가능성이 있습니다.

Identifiable representation이 설명력 및 일반화와 관련해 중요한 역할을 한다는 점도 논문의 related-work 부분에서 강조됩니다. 

다만 이것은 **가능성에 대한 구조적 이유**이지 이 논문이 unseen-domain risk를 직접 증명했다는 뜻은 아닙니다.

---

# 8-1. 일반화 성능을 실제로 더 올리려면

제가 후속 연구를 설계한다면 IDEA를 다음 방향으로 확장하는 것이 가장 중요하다고 봅니다.

### ① Fixed- $E$ HMM → open-ended regime discovery

현재:

$$
e_t\in\{1,\ldots,E\},
\qquad E\text{ fixed}.
$$

확장:

$$
e_t\in\{1,2,\ldots\},
$$

처럼 환경 수를 data-driven하게 결정하는 HDP-HMM 또는 nonparametric switching model을 고려할 수 있습니다.

그러면 training에 없던 새로운 regime을 별도의 state로 생성할 가능성이 생깁니다.

---

### ② Point environment prediction → distributional environment prediction

현재는

$$
\hat e_{t+h}
$$

를 결정해 latent prior에 넣는 구조에 가깝습니다.

더 안정적인 방법은

$$
p(e_{t+h}\mid x_{1:t})
$$

전체를 보존하여

```math
p(x_{t+h}\mid x_{1:t})
=
\sum_e
p(x_{t+h}\mid e,x_{1:t})
p(e\mid x_{1:t})
```

로 marginalization하는 것입니다.

장기 horizon에서 environment uncertainty가 커질수록 이것이 중요해집니다.

---

### ③ Fixed transition $\hat A$ → time-varying transition

현실에서는

$$
A_t\neq A_{t+1}
$$

일 수 있습니다.

예를 들어 장비 노화, 계절 변화, 시장 구조 변화가 environment 전환확률 자체를 바꿉니다.

따라서

$$
P(e_{t+1}\mid e_t,c_t)
$$

처럼 context-conditioned transition을 도입할 가치가 있습니다.

---

### ④ Two-phase freeze → joint 또는 alternating optimization

현재 HMM을 먼저 학습하고 freeze합니다.

이 경우 forecasting loss가

> “forecast에 정말 필요한 environment란 무엇인가?”

라는 신호를 environment estimator에 직접 되돌려 주기 어렵습니다.

따라서

```math
\mathcal L
=
\mathcal L_{\text{forecast}}
+
\lambda_{\text{id}}
\mathcal L_{\text{identification}}
+
\lambda_{\text{env}}
\mathcal L_{\text{environment}}
```

와 같은 alternating/joint optimization을 연구할 가치가 있습니다.

다만 joint optimization이 latent environment를 forecasting shortcut으로 붕괴시키지 않도록 identifiability regularization이 반드시 필요합니다.

---

### ⑤ 진짜 OOD benchmark를 만들어야 합니다

현재 benchmark보다 강한 평가 방식은 예를 들어

$$
\text{Train}: e\in\{1,2,3\},
$$

$$
\text{Validation}: e\in\{1,2,3\},
$$

$$
\text{Test}: e=4
$$

처럼 **미관측 environment만 test에 배치**하는 것입니다.

또는 transition matrix 자체를

$$
A_{\text{train}}\neq A_{\text{test}}
$$

로 바꿔야 합니다.

이 실험에서 IDEA 계열 모델이 RevIN, FAN, DDN, Koopa보다 강하다면 “일반화” 주장이 훨씬 설득력을 얻습니다.

---

# 8-2. 2020년 이후 관련 최신 연구와 비교

비정상 시계열 연구는 크게 **Normalization 계열**, **dynamic decomposition 계열**, **identifiable latent-state 계열**, 그리고 최근의 **local-expert/online adaptation 계열**로 발전해 왔습니다.

| 연구                                            | 핵심 아이디어                                                                  | Shift granularity   | Identifiability | IDEA와의 관계                                                |
| --------------------------------------------- | ------------------------------------------------------------------------ | ------------------- | --------------- | -------------------------------------------------------- |
| **RevIN (ICLR 2022)**                         | instance별 mean/variance 제거 후 복원                                          | instance            | 없음              | 가장 단순하고 범용적인 통계적 보정                                      |
| **Non-stationary Transformer (NeurIPS 2022)** | Series Stationarization + De-stationary Attention                        | instance/sequence   | 없음              | 비정상성 제거와 정보 복원의 균형                                       |
| **TDRL (NeurIPS 2022)**                       | 시간적 causal latent process의 disentanglement                               | latent dynamics     | 있음              | IDEA의 이론적 선행선                                            |
| **NCTRL (NeurIPS 2023)**                      | unknown nonstationarity 아래 latent causal components 식별                   | latent regime       | 있음              | IDEA와 가장 가까운 representation 계열 선행 연구                     |
| **SAN (NeurIPS 2023)**                        | local temporal slice별 adaptive normalization                             | slice               | 없음              | global instance assumption 완화                            |
| **Koopa (NeurIPS 2023)**                      | Fourier로 variant/invariant dynamics 분리 + Koopman predictor               | dynamic component   | 없음              | IDEA처럼 variant/invariant 분리하지만 causal identification은 아님 |
| **FAN (NeurIPS 2024)**                        | dominant frequency로 trend+seasonality 비정상성 추출                            | frequency/instance  | 없음              | plug-in 방식으로 구현이 매우 간단                                   |
| **DDN (NeurIPS 2024)**                        | time+wavelet frequency domain dynamic normalization                      | sliding window      | 없음              | continuous/local shift에 강함                               |
| **IDEA (2024)**                               | hidden environment + stationary/nonstationary identifiable latent states | latent regime       | **있음**          | “When + How” 결합                                          |
| **TFPS (NeurIPS 2025)**                       | patch pattern clustering + pattern-specific experts                      | patch               | 없음              | discrete/local regime adaptation을 prediction level에서 수행  |
| **TOT (NeurIPS 2025)**                        | online latent shift identification + Bayes-risk theory                   | online latent shift | **있음**          | IDEA 방향을 generalization theory와 online setting으로 확장      |

---

## RevIN — 2022

**Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift**는 instance의 평균·분산을 제거하고 prediction 후 다시 복원하는 model-agnostic 기법입니다. ([OpenReview][3])

IDEA와 비교하면 RevIN은 매우 단순하지만,

$$
\mu,\sigma
$$

수준을 넘어 “왜 distribution이 변했는가?”를 식별하지 않습니다.

장점은 적은 데이터와 낮은 계산량에서 매우 실용적이라는 점입니다.

---

## Non-stationary Transformer — 2022

**Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting**는 stationarization만 지나치게 수행하면 burst/event 정보까지 없애는 **over-stationarization** 문제가 생긴다고 지적하고 Series Stationarization과 De-stationary Attention을 결합합니다. ([NeurIPS Proceedings][4])

IDEA와의 차이는 명확합니다.

Non-stationary Transformer:

$$
\text{statistics correction}
+
\text{attention compensation}
$$

IDEA:

$$
\text{latent regime inference}
+
\text{causally structured disentanglement}.
$$

---

## TDRL — 2022

**Temporally Disentangled Representation Learning**은 nonparametric latent causal process와 distribution changes를 이용해 time-delayed latent causal variables를 식별하는 이론을 제공합니다. ([NeurIPS Proceedings][5])

IDEA는 이 causal-representation 계열을 **forecasting의 unknown environment detection 문제**로 확장했다고 보는 것이 가장 정확합니다.

---

## NCTRL — 2023

**Temporally Disentangled Representation Learning under Unknown Nonstationarity**는 observed auxiliary variable 없이 Markov 구조를 이용해 latent causal components를 식별합니다. ([NeurIPS Proceedings][6])

따라서 IDEA의 직접적인 이론적 경쟁축입니다.

IDEA가 한 단계 추가하는 것은

$$
z_t=(z_t^s,z_t^e)
$$

라는 stationary/nonstationary **partitioned subspace를 forecasting 목적에 직접 연결한 것**입니다.

Synthetic Table 1에서 NCTRL MCC 80.4에 비해 IDEA 95.1인 결과가 이 차이를 실험적으로 지지합니다. 

---

## SAN — 2023

**Adaptive Normalization for Non-stationary Time Series Forecasting: A Temporal Slice Perspective**는 하나의 전체 instance가 같은 통계를 공유한다고 보지 않고 local temporal slice별로 normalization을 수행하고 미래 statistics까지 예측합니다. ([NeurIPS Proceedings][7])

IDEA와 비교하면 SAN은 **continuous/local statistical shift** 처리에는 더 자연스러울 수 있지만, latent causal factor를 식별하지는 않습니다.

---

## Koopa — 2023

**Koopa: Learning Non-stationary Time Series Dynamics with Koopman Predictors**는 Fourier filter를 사용해 time-variant와 time-invariant component를 나누고 각각 Koopman dynamics로 미래로 진행시킵니다. ([NeurIPS Proceedings][8])

**용어 메모 — Koopman operator:**
비선형 dynamical system을 적절한 feature space로 옮긴 뒤 선형 operator로 dynamics를 표현하려는 수학적 방법입니다.

IDEA와 Koopa 모두

$$
\text{variant}+\text{invariant}
$$

분리를 시도하지만,

Koopa는 **dynamical forecasting representation**, IDEA는 **causally identifiable latent representation**에 더 가깝습니다.

---

## FAN — 2024

**Frequency Adaptive Normalization For Non-stationary Time Series Forecasting**은 Fourier domain에서 dominant frequency component를 찾아 trend와 seasonality를 동시에 다루는 model-agnostic normalization입니다. 저자들은 여러 backbone과 8개 benchmark에서 평균 MSE 개선을 보고합니다. ([NeurIPS Proceedings][9])

여기서 보고된 7.76–37.90%와 IDEA의 “1.7–12%” 같은 숫자는 **서로 실험 설정, backbone, baseline 정의가 다르므로 직접 비교해서는 안 됩니다.**

---

## DDN — 2024

**DDN: Dual-domain Dynamic Normalization for Non-stationary Time Series Forecasting**은 sliding window에서 time domain뿐 아니라 wavelet 기반 frequency domain까지 동적으로 normalization합니다. ([NeurIPS Proceedings][10])

이 접근은 IDEA의 discrete $e_t$ 가정보다 gradual/continuous shift에 자연스러울 가능성이 있습니다.

반대로 “어떤 latent physical factor가 변했는가”라는 해석 가능성은 IDEA 쪽이 더 강합니다.

---

## TFPS — 2025

**Learning Pattern-Specific Experts for Time Series Forecasting Under Patch-level Distribution Shift**는 각 patch의 패턴이 달라지는 문제를 명시적으로 다룹니다. Time/frequency dual-domain encoder → subspace clustering → pattern-specific expert 구조를 사용합니다. ([NeurIPS Proceedings][11])

이 연구는 IDEA가 “environment”라고 부른 것을 보다 local한 **pattern cluster**로 옮긴 것으로 볼 수 있습니다.

일반화 관점에서 매우 중요한 흐름입니다.

$$
\text{one global predictor}
\rightarrow
\text{regime/pattern-specific experts}.
$$

---

## TOT — 2025: IDEA 이후 가장 중요한 연결점

**Online Time Series Forecasting with Theoretical Guarantees**는 IDEA 저자 일부가 참여한 후속 연구이며, unknown distribution shift를 latent variable 관점에서 다룹니다. ([NeurIPS Proceedings][2])

특히 IDEA보다 한 단계 더 나아가,

> latent variable 정보를 forecasting에 공급하면 Bayes risk가 더 tight해지고, latent variable의 identifiability가 정확할수록 그 이점이 커진다

는 방향의 이론을 제시합니다.

이는 IDEA의 가장 중요한 미해결 질문,

$$
\boxed{
\text{왜 identifiable latent state가 forecasting generalization을 개선하는가?}
}
$$

를 직접 다루는 후속 방향입니다.

따라서 **IDEA → TOT**의 흐름은 이 분야의 중요한 발전 경로로 볼 수 있습니다.

---

# 9. 이 논문이 답하지 않는 질문

1. **진짜 environment 개수 $E$를 모르면 어떻게 하는가?**
   Sensitivity analysis는 있지만 automatic model selection이나 nonparametric inference는 없습니다.

2. **Training에서 한 번도 나오지 않은 새로운 environment는 어떻게 처리하는가?**
   현재 $\hat A$로는 기존 state 간 transition만 모델링합니다.

3. **Gradual shift가 존재할 때 discrete HMM environment가 적절한가?**
   연속 latent context와 직접 비교하지 않습니다.

4. **실제 benchmark에서 identifiability theorem의 모든 가정이 충족되는가?**
   이를 empirical diagnostic으로 검증하지 않습니다.

5. **예측 성능 향상 중 얼마가 identifiability 때문이고 얼마가 더 많은 parameter 때문인가?**
   IDEA-sh 분석은 일부 정보를 주지만 완전한 parameter-matched control은 아닙니다. 

6. **환경 추정 uncertainty가 forecast uncertainty에 어떻게 전파되는가?**
   probabilistic calibration 관점의 결과가 없습니다.

7. **Structural break가 transition matrix 자체를 바꾼다면?**
   $A_t$가 변하는 경우의 이론이 없습니다.

8. **고차원 multivariate series에서 latent dimension 증가 시 modular prior의 계산량과 memory가 어떻게 scale되는가?**
   이론적으로 $O(nL)$라고 하지만 실제 GPU memory는 dataset에 따라 상당히 큽니다. 

9. **Identifiable latent representation이 실제 unseen-domain test error를 얼마나 감소시키는가?**
   가장 중요한 generalization 질문이지만 현재 논문에는 전용 실험이 없습니다.

10. **환경 state가 실제 물리적 regime과 대응하는가?**
    Synthetic에서는 ground truth가 있지만 실세계 benchmark에서는 latent environment의 semantic validation이 없습니다.

---

# 10. 연구자 관점에서 평가한 핵심 강점과 약점

논문의 가장 강한 부분은 **forecasting trick과 identifiability theory를 연결했다는 것**입니다. RevIN·SAN·FAN·DDN 등이 “관측된 distribution statistics를 어떻게 보정할 것인가”를 묻는다면 IDEA는 한 단계 아래로 내려가 “관측 분포를 바꾸는 latent mechanism이 무엇인가”를 묻습니다.

특히

$$
\boxed{
\text{When}
\rightarrow e_t
}
$$

와

$$
\boxed{
\text{How}
\rightarrow (z_t^s,z_t^e)
}
$$

를 구분한 conceptual decomposition은 상당히 유용합니다.

반면 가장 큰 약점은 이론에서 얻은 identifiability와 실제 forecasting generalization 사이에 **한 단계 논리적 간격이 남아 있다는 것**입니다.

즉

$$
\text{Identifiable latent representation}
$$

이라는 성질과

$$
\text{lower future/OOD forecasting risk}
$$

가 현재 논문에서는 하나의 정리로 직접 연결되어 있지 않습니다.

2025년 TOT 연구가 바로 이 부분을 Bayes-risk 관점에서 보완하려 한다는 점이 중요합니다. ([NeurIPS Proceedings][2])

---

# 11. 결론 및 후속 연구 방향

## 저자들이 실제로 결론에서 제시한 시사점

저자들은 결론에서 IDEA가 기존의 **uniform temporal distribution shift assumption을 완화**하고, stationary/nonstationary latent variables를 식별 가능한 방식으로 모델링함으로써 현실적인 nonstationary forecasting과 causal representation learning을 연결했다고 평가합니다. 

다만 **논문 결론에는 구체적인 future-work roadmap이 별도로 제시되어 있지 않습니다.** 따라서 아래 후속 연구 방향은 저자가 명시한 계획이 아니라, 논문의 결과와 2024–2025 후속 연구를 근거로 한 제안입니다.

### 제가 우선순위를 둔다면 다음 연구가 가장 중요합니다.

첫째, **unknown $E$ + unseen environment discovery**입니다. Fixed-state HMM 대신 nonparametric 또는 continuous switching latent model을 도입해야 합니다.

둘째, **environment uncertainty propagation**입니다. $\hat e_t$ 하나를 고르는 대신 $p(e_t\mid x)$ 전체를 forecasting distribution에 반영해야 장기 horizon에서 robust해질 가능성이 높습니다.

셋째, **time-varying transition matrix $A_t$**를 도입해야 합니다. 실제 nonstationarity에서는 값뿐 아니라 regime 전환 규칙도 drift합니다.

넷째, **identifiability와 OOD forecasting risk를 하나의 이론으로 연결**해야 합니다. 이 방향에서는 2025 TOT가 중요한 출발점입니다. ([NeurIPS Proceedings][2])

다섯째, **IDEA + local-pattern expert**의 결합이 유망합니다. IDEA가 global latent regime을 식별하고 TFPS류 expert가 각 local pattern을 담당하도록 하면 discrete global shift와 patch-level shift를 함께 모델링할 수 있습니다. ([NeurIPS Proceedings][11])

여섯째, 평가 프로토콜도 바뀌어야 합니다. 단순 chronological split을 넘어

$$
\text{unseen regime},
\quad
A_{\text{train}}\neq A_{\text{test}},
\quad
\text{abrupt shift},
\quad
\text{gradual drift},
\quad
\text{recurring regime}
$$

을 별도로 시험하고, 3 seeds가 아니라 더 많은 반복과 block bootstrap 또는 forecast-error dependence를 고려한 Diebold–Mariano 계열 검정을 병행하는 편이 타당합니다.

---

# 최종 연구적 판단

**IDEA의 가장 중요한 의미는 “비정상성을 제거하는 방법”을 하나 더 만든 것이 아닙니다.**

핵심은

$$
\boxed{
\text{Nonstationarity}
\rightarrow
\text{latent environment identification}
+
\text{stationary/nonstationary mechanism disentanglement}
}
$$

로 문제 자체를 다시 정의했다는 데 있습니다.

2022–2024년의 RevIN → SAN → FAN/DDN 계열이 통계량과 frequency 구조를 더 정교하게 보정하는 방향이라면, TDRL → NCTRL → IDEA는 **식별 가능한 latent causal representation**을 강화하는 방향입니다. 그리고 2025년 TFPS와 TOT에서는 다시 **local pattern adaptation**과 **online latent-shift generalization theory**로 확장되고 있습니다. ([OpenReview][3])

따라서 **일반화 성능**만 놓고 보면 IDEA의 현재 성과보다 더 중요한 것은 향후

$$
\boxed{
\text{identifiable latent state}
+
\text{online adaptation}
+
\text{open-set regime discovery}
+
\text{uncertainty-aware forecasting}
}
$$

의 결합입니다.

이 방향이 성공하면 기존의 “미래도 훈련 데이터의 여러 regime 중 하나일 것”이라는 제한을 넘어, **훈련에서 보지 못한 미래 공정·환경·운영 regime에 대한 일반화**로 발전할 수 있습니다.

---

# 참고한 자료 및 사이트 제목

| 구분       | 참고자료 제목                                                                                                | 출처                                              |
| -------- | ------------------------------------------------------------------------------------------------------ | ----------------------------------------------- |
| 주 논문     | **When and How: Learning Identifiable Latent States for Nonstationary Time Series Forecasting**        | 첨부 PDF, arXiv:2402.12767v2 / arXiv ([arXiv][1]) |
| 비교 연구    | **Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift**  | ICLR 2022 / OpenReview ([OpenReview][3])        |
| 비교 연구    | **Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting**                 | NeurIPS 2022 ([NeurIPS Proceedings][4])         |
| 비교 연구    | **Temporally Disentangled Representation Learning**                                                    | NeurIPS 2022 ([NeurIPS Proceedings][5])         |
| 비교 연구    | **Temporally Disentangled Representation Learning under Unknown Nonstationarity**                      | NeurIPS 2023 ([NeurIPS Proceedings][6])         |
| 비교 연구    | **Adaptive Normalization for Non-stationary Time Series Forecasting: A Temporal Slice Perspective**    | NeurIPS 2023 ([NeurIPS Proceedings][7])         |
| 비교 연구    | **Koopa: Learning Non-stationary Time Series Dynamics with Koopman Predictors**                        | NeurIPS 2023 ([NeurIPS Proceedings][8])         |
| 비교 연구    | **Frequency Adaptive Normalization For Non-stationary Time Series Forecasting**                        | NeurIPS 2024 ([NeurIPS Proceedings][9])         |
| 비교 연구    | **DDN: Dual-domain Dynamic Normalization for Non-stationary Time Series Forecasting**                  | NeurIPS 2024 ([NeurIPS Proceedings][10])        |
| 최신 비교 연구 | **Learning Pattern-Specific Experts for Time Series Forecasting Under Patch-level Distribution Shift** | NeurIPS 2025 ([NeurIPS Proceedings][11])        |
| 최신 후속 방향 | **Online Time Series Forecasting with Theoretical Guarantees**                                         | NeurIPS 2025 ([NeurIPS Proceedings][2])         |

다음 단계로는 이 분석을 바탕으로 **IDEA의 Equation (1)–(18)을 하나씩 데이터가 실제로 어떻게 흘러가는지 수치 예제까지 대입하여 설명하고, IDEA를 실제 다변량 공정 시계열에 구현하기 위한 PyTorch 파이프라인·누수 방지 검증·일반화 실험 설계까지 연결하는 것**이 가장 유용합니다.

[1]: https://arxiv.org/abs/2402.12767?utm_source=chatgpt.com "When and How: Learning Identifiable Latent States for Nonstationary Time Series Forecasting"
[2]: https://proceedings.neurips.cc/paper_files/paper/2025/hash/c3010e98dc44b6f76df7cf82b5e12c77-Abstract-Conference.html?utm_source=chatgpt.com "Online Time Series Forecasting with Theoretical Guarantees"
[3]: https://openreview.net/pdf?id=cGDAkQo1C0p&utm_source=chatgpt.com "Published as a conference paper at ICLR 2022"
[4]: https://proceedings.neurips.cc/paper_files/paper/2022/hash/4054556fcaa934b0bf76da52cf4f92cb-Abstract-Conference.html?utm_source=chatgpt.com "Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting"
[5]: https://proceedings.neurips.cc/paper_files/paper/2022/hash/a938292feb86b94ebe3e6200ff7786ef-Abstract-Conference.html?utm_source=chatgpt.com "Temporally Disentangled Representation Learning"
[6]: https://proceedings.neurips.cc/paper_files/paper/2023/hash/19a567abaec3990cb40d7a013556fecd-Abstract-Conference.html?utm_source=chatgpt.com "Temporally Disentangled Representation Learning under Unknown Nonstationarity"
[7]: https://proceedings.neurips.cc/paper_files/paper/2023/hash/2e19dab94882bc95ed094c4399cfda02-Abstract-Conference.html?utm_source=chatgpt.com "Adaptive Normalization for Non-stationary Time Series Forecasting: A Temporal Slice Perspective"
[8]: https://proceedings.neurips.cc/paper_files/paper/2023/hash/28b3dc0970fa4624a63278a4268de997-Abstract-Conference.html?utm_source=chatgpt.com "Koopa: Learning Non-stationary Time Series Dynamics with Koopman Predictors"
[9]: https://proceedings.neurips.cc/paper_files/paper/2024/hash/37c6d0bc4d2917dcbea693b18504bd87-Abstract-Conference.html?utm_source=chatgpt.com "Frequency Adaptive Normalization For Non-stationary Time Series Forecasting"
[10]: https://proceedings.neurips.cc/paper_files/paper/2024/hash/c44c4afd77d5ee760e7f4bed0c50f878-Abstract-Conference.html?utm_source=chatgpt.com "DDN: Dual-domain Dynamic Normalization for Non-stationary Time Series Forecasting"
[11]: https://proceedings.neurips.cc/paper_files/paper/2025/hash/8491a7fcc218946b471b600a915c8b02-Abstract-Conference.html?utm_source=chatgpt.com "Learning Pattern-Specific Experts for Time Series Forecasting Under Patch-level Distribution Shift"
