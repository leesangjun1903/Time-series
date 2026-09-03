# AdaNODEs: Test Time Adaptation for Time Series Forecasting Using Neural ODEs

# 1. Executive Summary — 10문장 이내

1. **AdaNODEs**는 학습 데이터에 다시 접근하지 않고, 테스트 시점의 비라벨 시계열만을 이용하여 예측 모델을 새로운 분포에 적응시키는 **source-free Test-Time Adaptation(TTA)** 방법입니다. 
2. 핵심 문제의식은 기존 TTA가 주로 독립적인 이미지 분류 문제를 대상으로 설계되어 **시간적 의존성이 존재하는 회귀·시계열 예측에는 직접 적용하기 어렵다**는 것입니다. 
3. AdaNODEs는 Encoder–Latent Neural ODE–Decoder 구조를 유지하면서 테스트 시 기존 모델 파라미터를 동결하고, latent ODE 내부의 **scaling parameter $\alpha$와 shifting parameter $\gamma$만 학습**합니다. 
4. 저자들은 $\alpha$가 잠재 동역학의 amplitude/frequency 변화를, $\gamma$가 temporal/phase shift를 흡수하도록 설계했다고 설명합니다. 
5. 라벨이 없는 forecasting TTA를 위해 Gaussian predictive distribution에 대한 **NLL과 context–forecast posterior 사이 KL divergence를 결합한 새로운 적응 손실**을 제안합니다. 
6. 합성 1차원 linear/sinusoidal/damped-oscillator 데이터에서는 분포 이동 강도가 커질수록 source model 대비 개선 효과가 커지는 경향을 보고합니다. 
7. Rotating MNIST에서도 AdaNODEs가 source model보다 MSE, CC, CCC를 개선했지만, severe shift에서는 절대적인 correlation 성능이 여전히 낮거나 음수인 경우가 존재합니다. 
8. 따라서 이 논문의 가장 중요한 공헌은 절대적인 SOTA forecasting 성능보다 **“라벨 없이, source data 없이, 매우 적은 적응 파라미터만으로 시간 동역학 자체를 test time에 수정한다”는 연구 방향을 제시한 것**으로 보는 것이 더 정확합니다.

---

# 1-1. 연구의 목적과 필요성

일반적인 시계열 예측에서는 학습 시점과 실제 운영 시점의 확률분포가 같다고 암묵적으로 가정합니다.

$$
(X_{\text{train}},Y_{\text{train}})\sim P_{\text{train}},
\qquad
(X_{\text{test}},Y_{\text{test}})\sim P_{\text{test}}
$$

통상적인 supervised learning은

$$
P_{\text{train}}\approx P_{\text{test}}
$$

일 때 가장 잘 작동하지만, 현실의 시계열에서는 시간 경과에 따라

$$
P_{\text{train}}\neq P_{\text{test}}
$$

가 되기 쉽습니다. 논문 역시 학습/테스트 분포 차이가 forecasting 성능을 크게 저하시킨다는 점을 연구 출발점으로 삼습니다. 

*용어 메모 — **Distribution Shift**: 학습 때 관찰한 입력·출력의 통계적 성질과 실제 운영 중 관찰되는 통계적 성질이 달라지는 현상입니다. 평균 변화, 분산 변화, 주기 변화, 시간 지연, 센서 특성 변화 등이 모두 포함될 수 있습니다.*

기존 해결책인 Domain Adaptation은 흔히 source/target 데이터를 함께 사용하거나 target label 일부를 요구합니다. 반면 실제 운영 환경에서는 개인정보·저장비용·보안 문제 때문에 source data를 다시 가져올 수 없고, 미래의 정답 $Y$도 forecasting 시점에는 존재하지 않습니다. AdaNODEs는 바로 이 조건을 겨냥합니다. 

여기서 **source-free**는 “source model조차 필요 없다”는 뜻이 아니라,

$$
\boxed{\text{pre-trained source model은 존재하지만 source training data는 사용하지 않는다}}
$$

는 의미입니다.

*용어 메모 — **Test-Time Adaptation(TTA)**: 학습을 완전히 끝낸 모델을 배포한 이후 테스트 데이터가 들어올 때 모델 또는 일부 파라미터를 즉석에서 수정하는 방식입니다.*

---

# 2. 핵심 주장과 근거

| 핵심 주장                                                 | 저자의 근거                                                | 위치                 | 제 평가                                                     |
| ----------------------------------------------------- | ----------------------------------------------------- | ------------------ | -------------------------------------------------------- |
| 기존 TTA는 forecasting과 temporal dependency를 충분히 다루지 못한다 | 대부분 classification 중심이며 entropy/pseudo-label 기반이라고 논의 | p.1–2, Sec. 1, 2.1 | **타당하지만 2025–2026 연구를 고려하면 현재는 경쟁자가 빠르게 늘어난 상태**         |
| NODE에 $\alpha,\gamma$만 추가해 temporal shift에 적응할 수 있다   | Eq. (1), Fig. 1(b)                                    | p.2                | **좋은 inductive bias지만 일반적인 수학적 보장은 아님**                  |
| label-free forecasting용 NLL+KL loss가 가능하다             | Eq. (3)–(5), Fig. 1(c)                                | p.3                | **아이디어는 흥미롭지만 NLL의 self-confidence collapse 가능성 검증이 부족** |
| 큰 distribution shift에서 상대적으로 강하다                      | Fig. 2, Fig. 3                                        | p.3–4              | **합성 shift에서는 지지됨**                                      |
| Rotating MNIST에서도 개선된다                                | Table 1, Fig. 4                                       | p.4                | **수치는 개선되지만 severe shift 복구는 제한적**                       |
| 파라미터·메모리 효율적이다                                        | $\alpha,\gamma$만 업데이트한다는 구조적 주장                       | Fig. 1, Sec. 3.2   | **파라미터 수 측면은 설득력 있지만 실제 memory/runtime benchmark가 없음**   |
| regression 전반으로 확장 가능하다                               | Conclusion                                            | p.4                | **현 실험만으로는 다소 강한 일반화 주장**                                |

논문의 전체적인 주장은 Abstract와 Conclusion에 명확히 나타나며, 저자는 1차원 및 고차원 실험을 통해 강한 shift에서의 robustness를 강조합니다.  

---

# 2-1. 해결 문제, 제안 방법, 수학적 원리, 모델 구조, 성능과 한계

## 2-1-1. 문제의 수학적 정의

과거 관측값을

$$
y_p(t)
$$

미래 예측값을

$$
y_f(t)
$$

라고 하면 source domain에서는

$$
y_p(t),y_f(t)\sim D_{\text{train}}
$$

을 이용해 forecasting model을 학습합니다.

테스트 시에는

$$
y'_p(t),y'_f(t)\sim D_{\text{test}},
\qquad
D_{\text{test}}\neq D_{\text{train}}
$$

가 됩니다. 논문의 핵심은 $D_{\text{test}}$에서 **정답 $y'_f(t)$를 아직 관측하지 않은 상태에서도 모델을 바꾸는 것**입니다. 

---

## 2-1-2. 전체 모델 구조

Figure 1(b)의 구조는

$$
\text{Encoder}
\rightarrow
\text{Latent NODE}
\rightarrow
\text{Decoder}
$$

입니다. 

### ① Encoder

관측된 과거 시계열로부터 초기 latent state를 추론합니다.

$$
y_p
\xrightarrow{f_{\text{enc}}}
z_{t_0}
$$

여기서

* $y_p$: 관측된 과거 시계열
* $f_{\text{enc}}$: encoder
* $z_{t_0}$: ODE가 시작되는 latent state
* $t_0$: latent trajectory의 초기 시점입니다.

*용어 메모 — **Latent state**: 원래 시계열을 그대로 사용하는 대신 모델 내부에서 압축하여 표현한 숨겨진 상태입니다.*

---

### ② Neural ODE

일반적인 Neural ODE는

```math
\frac{dz(t)}{dt}
=
f_{\text{node}}(z(t);\theta)
```

로 표현할 수 있습니다.

여기서

* $z(t)$: 시각 $t$의 latent state
* $\frac{dz(t)}{dt}$: latent state가 시간에 따라 얼마나 빠르게 변하는지
* $f_{\text{node}}$: neural network로 표현된 동역학 함수
* $\theta$: source domain에서 학습한 neural ODE parameter입니다.

*용어 메모 — **Neural ODE**: 일반적인 신경망의 layer-to-layer 변화를 연속시간 미분방정식으로 표현하는 모델입니다.*

---

## 2-1-3. AdaNODEs의 가장 중요한 수식

저자가 제안한 적응식은 다음과 같습니다.

```math
\boxed{
\frac{dz(t)}{dt}
=
f_{\text{node}}
\left(
\alpha z(t)+\gamma;\theta
\right)
}
```

여기서

* $\alpha$: scaling parameter
* $\gamma$: shifting parameter
* $\theta$: 기존 source NODE 파라미터이며 TTA 중에는 고정
* $z(t)$: latent state입니다.

즉 기존의

$$
f_{\text{node}}(z(t);\theta)
$$

대신

$$
f_{\text{node}}(\alpha z(t)+\gamma;\theta)
$$

를 사용합니다.

핵심적으로

$$
\boxed{
\theta \text{는 고정},\qquad
\alpha,\gamma \text{만 업데이트}
}
$$

입니다.

이것이 AdaNODEs의 parameter efficiency를 만들어냅니다.

---

### $\alpha$는 무엇을 하는가?

저자는 $\alpha$가 latent trajectory의 amplitude 및 frequency를 조절한다고 설명합니다.

$$
z(t)\longrightarrow\alpha z(t)
$$

이면 NODE가 보게 되는 latent 위치 자체가 변화합니다.

또한 local Jacobian을 생각하면

```math
J_{\text{adapt}}(z)
=
\left.
\frac{\partial f_{\text{node}}(u;\theta)}
{\partial u}
\right|_{u=\alpha z+\gamma}
\alpha
```

가 됩니다.

여기서

* $J_{\text{adapt}}$: 적응 후 local dynamics를 설명하는 Jacobian
* $u=\alpha z+\gamma$: 변환된 latent coordinate입니다.

따라서 $\alpha$가 변하면 latent dynamics의 변화율까지 변할 수 있습니다.

다만 중요한 주의점이 있습니다.

**일반적인 nonlinear $f_{\text{node}}$에 대해**

$$
\alpha\uparrow
\quad\Rightarrow\quad
\text{frequency}\uparrow
$$

가 항상 성립하는 수학적 정리는 아닙니다.

논문의 “큰 $\alpha$가 더 빠른 변화·더 큰 amplitude를 만든다”는 설명은 **모델 설계에 대한 직관적 해석**에 가깝습니다. 

예를 들어 단순 선형계가

$$
f_{\text{node}}(u)=Wu
$$

라면

```math
\frac{dz}{dt}
=
W(\alpha z+\gamma)
=
\alpha Wz+W\gamma
```

가 되어 $\alpha$가 시스템의 eigenvalue와 시간상수를 변화시킬 수 있지만, 이를 곧바로 모든 nonlinear system의 “frequency parameter”라고 부를 수는 없습니다.

이 점은 논문의 이론적 한계 중 하나입니다.

---

### $\gamma$는 무엇을 하는가?

$$
z(t)\longrightarrow z(t)+\gamma
$$

라는 latent translation을 만들기 때문에 trajectory가 NODE vector field의 다른 영역을 지나게 됩니다.

저자는 이를 phase/time shift와 연결하여 설명합니다. 

그러나 역시 임의의 nonlinear NODE에서

$$
\gamma > 0
\Rightarrow
\text{time delay}
$$

가 일반적으로 보장되는 것은 아닙니다.

즉 $\alpha,\gamma$의 해석 가능성은 **실험적·구조적 inductive bias**로 보는 편이 정확합니다.

*용어 메모 — **Inductive bias**: 모델이 어떤 종류의 패턴을 더 쉽게 학습하도록 미리 넣어 둔 구조적 가정입니다.*

---

## 2-1-4. ODE trajectory 생성

논문의 Eq. (2)는

```math
z(t)
=
\text{ODESolve}
\left(
f,\theta,z_{t_0},t_n,\alpha,\gamma
\right),
\qquad
t_n\in[t_0,t_N]
```

입니다. 

즉 초기 상태 $z_{t_0}$부터 ODE solver가 연속시간 latent trajectory를 계산합니다.

실험에서는 **RK45 adaptive solver**와 **adjoint method**를 사용했습니다. 

*용어 메모 — **RK45**: Runge–Kutta 4차와 5차 근사를 동시에 이용하여 필요한 시간 간격을 자동 조절하는 ODE 수치해석 기법입니다.*

*용어 메모 — **Adjoint method**: 모든 ODE 중간 상태를 저장하지 않고 역방향 미분방정식을 풀어 gradient를 계산하여 메모리를 절약하는 방법입니다.*

---

## 2-1-5. Decoder의 확률적 예측

Decoder는 하나의 값만 출력하지 않고

$$
p(y_t\mid z_{t_0},\theta,\alpha,\gamma)
$$

라는 predictive distribution을 출력합니다.

논문에서는 계산 편의를 위해

$$
p(y_t\mid z_{t_0},\theta,\alpha,\gamma)
\sim
\mathcal N(\mu_t,\sigma_t)
$$

를 사용합니다. 

여기서

* $\mu_t$: 예측 평균
* $\sigma_t$: 예측 불확실성의 scale
* $\mathcal N$: Gaussian distribution입니다.

---

# 2-1-6. 핵심 TTA Loss

전체 test-time objective는

```math
\boxed{
\min_{\alpha,\gamma}
\mathcal L(\theta,\alpha,\gamma)
=
\lambda
\sum_t
\mathcal L_t^{\text{NLL}}
+
(1-\lambda)
\mathcal L^{\text{KL}}
}
```

입니다. 

여기서

* $\lambda\in[0,1]$: NLL과 KL의 상대적인 중요도를 조절
* $\mathcal L_t^{\text{NLL}}$: Negative Log-Likelihood
* $\mathcal L^{\text{KL}}$: KL divergence입니다.

---

### Negative Log-Likelihood

논문의 Eq. (4)는

```math
\mathcal L_t^{\text{NLL}}
=
\mathbb E_{
z_{t_0}\sim
q_\phi(z_{t_0}\mid t_C,y_C)
}
\left[
-\log
p(
y_T
\mid
z_{t_0},t,\alpha,\gamma,\theta
)
\right]
```

입니다.

여기서

* $q_\phi$: encoder가 만드는 approximate posterior
* $\phi$: encoder parameter
* $(t_C,y_C)$: context, 즉 실제로 관측한 과거 시계열
* $y_T$: future/target-side sequence
* $\mathbb E$: latent posterior에 대한 평균입니다.

*용어 메모 — **Negative Log-Likelihood(NLL)**: 모델이 자신이 예측한 확률분포 안에서 관측값을 얼마나 자연스럽게 설명하는지를 측정하는 손실입니다. 값이 작을수록 일반적으로 더 높은 likelihood를 의미합니다.*

그런데 AdaNODEs에는 실제 미래 label이 없습니다. 논문은 이를 **predicted mean을 predicted distribution에 fitting하는 방식으로 계산한다**고 설명합니다. 

이 부분은 매우 중요합니다.

논문의 설명을 Gaussian case에서 문자 그대로 해석하면

$$
y_T=\mu_t
$$

를 사용하게 되고,

```math
-\log
\mathcal N
(\mu_t;\mu_t,\sigma_t^2)
=
\frac12
\log
(2\pi\sigma_t^2)
```

가 됩니다.

따라서 다른 제약이 없다면

$$
\sigma_t\rightarrow0
$$

으로 보내는 것이 NLL을 감소시키는 방향이 됩니다.

즉 이것은 **예측 평균을 실제 정답 쪽으로 움직인다기보다 모델의 uncertainty를 줄이고 confidence를 높이는 효과**를 가질 수 있습니다.

저자 역시 이를 classification TTA의 entropy minimization과 유사하다고 설명합니다. 

따라서 장점과 위험이 동시에 존재합니다.

$$
\text{uncertainty reduction}
\quad\text{vs.}\quad
\text{over-confidence}
$$

입니다.

논문에는 $\sigma$ collapse가 실제로 발생하지 않는지를 보여주는 calibration experiment가 없습니다.

---

## KL divergence

두 번째 손실은

```math
\boxed{
\mathcal L^{\text{KL}}
=
D_{\text{KL}}
\left(
q_\phi(z_{t_0}\mid t_C,y_C)
\,
\Vert
\,
q_\phi(z_{t_0}\mid t_T,y_T)
\right)
}
```

입니다. 

여기서

* $D_{\text{KL}}(P\Vert Q)$: 두 확률분포가 얼마나 다른지를 나타내는 KL divergence
* $(t_C,y_C)$: 실제 관측된 context
* $(t_T,y_T)$: 미래 prediction 기반 target representation입니다.

목적은

$$
q_\phi(z\mid\text{observed context})
\approx
q_\phi(z\mid\text{forecasted sequence})
$$

가 되도록 만드는 것입니다.

즉

> “과거에서 추론한 latent dynamics와 모델이 예측한 미래가 서로 모순되지 않게 만들자.”

라는 consistency regularization입니다.

\*용어 메모 — **KL divergence**: 두 확률분포 $P$와 $Q$의 차이를 측정하는 비대칭적 거리와 유사한 양입니다. 정확한 metric은 아니지만 분포 정렬에 널리 사용합니다.*

다만 $y_T$ 자체가 model prediction이라면 잘못된 prediction을 다시 encoder에 넣어 정합성을 강화할 위험, 즉 **confirmation bias**가 존재합니다.

논문은 이 문제를 별도로 분석하지 않았습니다.

---

# 2-1-7. 학습 및 TTA 과정

논문의 Figure 1을 실제 알고리즘 흐름으로 풀면 다음과 같습니다. 

### Source training

$$
y_C
\rightarrow
f_{\text{enc}}
\rightarrow
z_{t_0}
\rightarrow
f_{\text{node}}(\cdot;\theta)
\rightarrow
z(t)
\rightarrow
f_{\text{dec}}
\rightarrow
\hat y(t)
$$

이때 Encoder, NODE, Decoder를 source domain에서 학습합니다.

### Test-time adaptation

기존 network parameter는 고정하고

$$
\theta_{\text{enc}},
\theta_{\text{node}},
\theta_{\text{dec}}
\quad\text{Frozen}
$$

새로운 두 적응 parameter만 추가합니다.

$$
\alpha,\gamma
\quad\text{Adaptive}
$$

그리고

```math
(\alpha^*,\gamma^*)
=
\arg\min_{\alpha,\gamma}
\mathcal L(\theta,\alpha,\gamma)
```

를 수행합니다.

전체 수백만 parameter를 fine-tuning하는 것보다 훨씬 작은 adaptation space입니다.

---

# 3. 주장별 Page / Figure / Table 위치

핵심 주장들의 근거 위치를 다시 압축하면 다음과 같습니다.

* **TTA가 forecasting에 부족하다는 문제 정의**: p.1, Introduction; p.2, Related Work.
* **Encoder–NODE–Decoder 구조**: p.2, **Figure 1(b)**.
* **$\alpha,\gamma$ test-time adaptation**: p.2, **Eq. (1)** 및 Figure 1(b).
* **ODE trajectory 생성**: p.3, **Eq. (2)**.
* **NLL+KL loss**: p.3, **Eq. (3)–(5), Figure 1(c)**.
* **1-D shift robustness**: p.3–4, **Figure 2, Figure 3**.
* **Rotating MNIST 정량 성능**: p.4, **Table 1**.
* **Rotating MNIST qualitative example**: p.4, **Figure 4**.
* **일반적인 regression TTA로의 확장 가능성 주장**: p.4, Conclusion. 

---

# 4. 저자가 직접 보고한 내용과 제 해석의 분리

| 항목                 | 저자가 직접 보고한 내용                                                    | 제 해석                                                                                                                  |
| ------------------ | ---------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **연구 주제**          | label-free/source-free TTA를 time-series forecasting에 적용          | 정확히는 **source-data-free forecasting adaptation**이 가장 핵심이며, 기존 TTA를 단순 적용한 것이 아니라 latent dynamics를 수정한다는 점이 중요         |
| **방법**             | NODE에 $\alpha,\gamma$를 추가해 amplitude/frequency 및 phase shift에 적응 | $\alpha,\gamma$는 매우 저차원 adaptation bottleneck이며 regularization 효과가 기대되지만, frequency/phase와의 1:1 대응은 수학적으로 보장되지 않음     |
| **Loss**           | NLL로 uncertainty를 감소시키고 KL로 observed/predicted distribution을 정렬  | label-free라는 장점이 있으나 self-generated target을 이용하기 때문에 confidence collapse 및 confirmation bias 검증이 필요                   |
| **1-D 결과**         | amplitude/frequency에서 0.53–18%, delay에서 2.5–21.8% source 대비 향상   | shift severity 증가 시 이득이 커지는 것은 AdaNODEs의 inductive bias와 잘 맞지만 synthetic shift 설계 자체가 $\alpha,\gamma$ 구조에 유리할 가능성이 있음 |
| **Rotating MNIST** | MSE 9.6% 감소, CC 28.4%, CCC 28.3% 증가                              | 표시된 Table 1 값을 이용해 확인하면 이 수치는 **severity별 상대개선율의 평균**과 거의 일치하며, aggregated metric 계산 방식은 본문에서 명시되지 않음                 |
| **효율성**            | $\alpha,\gamma$만 업데이트하여 memory 사용을 줄임                            | parameter efficiency는 분명하지만 wall-clock time, GPU memory, ODE solve 횟수 등의 직접 측정이 없어 실제 deployment 효율성은 아직 미검증          |
| **일반화**            | time-series 및 regression TTA에 새로운 경로를 제시                         | “가능성을 제시했다”는 표현은 타당하지만 실험 범위로 regression 전반의 일반화를 입증했다고 보기는 어려움                                                       |

저자의 1-D 결과 보고는 Figure 2–3 및 본문의 수치에 기반합니다.  Rotating MNIST 결과는 Table 1과 본문에 보고되어 있습니다. 

---

# 5. 통계적으로 취약한 부분과 직접 비교하기 어려운 수치

## 5-1. 가장 큰 문제: 실제 real-world forecasting benchmark가 없다

실험 데이터는

* synthetic linear signal,
* synthetic sinusoidal signal,
* damped oscillator,
* Rotating MNIST

입니다. 

즉 ETTh/ETTm, Electricity, Traffic, Weather, Exchange 같은 실제 forecasting benchmark가 없습니다.

특히 논문에서 “high-dimensional data”라고 부르는 것은 **multivariate industrial time series가 아니라 image sequence인 Rotating MNIST**입니다.

따라서

$$
\text{synthetic temporal shift 성능}
\not\Rightarrow
\text{real-world multivariate forecasting 성능}
$$

입니다.

---

## 5-2. 실험 shift가 AdaNODEs 구조와 지나치게 잘 맞는다

논문이 생성한 shift는 주로

$$
\text{amplitude change},
\quad
\text{frequency change},
\quad
\text{time delay}
$$

입니다. 

그런데 AdaNODEs가 추가한 파라미터 역시

$$
\alpha=\text{scale},
\qquad
\gamma=\text{shift}
$$

입니다.

따라서 benchmark와 모델의 inductive bias 사이에 구조적 정합성이 매우 높습니다.

실제 시계열에서는

$$
\text{variance drift},
\text{cross-variable dependency drift},
\text{regime change},
\text{concept shift},
\text{sensor degradation}
$$

등이 동시에 발생할 수 있으므로 추가 검증이 필요합니다.

---

## 5-3. 실험 반복 횟수와 통계적 검정이 명확하지 않다

Table 1은

$$
\text{mean}\pm\text{quantity}
$$

형식의 결과를 보여주지만, 본문에는 제가 확인한 범위에서

* 반복 실험 횟수 $n$,
* $\pm$가 standard deviation인지 standard error인지,
* confidence interval,
* paired statistical test

가 명확히 설명되어 있지 않습니다. 

따라서

$$
0.073\pm0.003
\quad\text{vs.}\quad
0.071\pm0.003
$$

처럼 차이가 작은 경우 실제 유의한 차이인지 판단하기 어렵습니다.

---

## 5-4. “9.6%, 28.4%, 28.3%” 계산 방식이 명시되어 있지 않다

Table 1의 amplitude/frequency 조건에서 표시된 값을 제가 다시 계산하면,

각 severity별 상대 개선율을 먼저 계산한 후 평균한 경우 MSE는 약

$$
9.64\%
$$

CC는

$$
28.45\%
$$

CCC는

$$
28.38\%
$$

가 되어 저자가 보고한 9.6%, 28.4%, 28.3%와 거의 정확히 일치합니다.

반면 severity 전체의 값을 먼저 평균한 뒤 상대 개선율을 계산하면 약

$$
\text{MSE}:8.25\%,
$$

$$
\text{CC}:24.80\%,
$$

$$
\text{CCC}:24.88\%
$$

입니다.

즉 저자 수치가 잘못되었다고 볼 수는 없지만,

$$
\text{mean of relative improvements}
\neq
\text{relative improvement of means}
$$

이므로 **집계 방식이 명시되어야 합니다.**

---

## 5-5. CC/CCC에 상대 % 향상을 사용하는 것은 특히 위험하다

Table 1의 severity 5 time-delay에서는

$$
CC_{\text{Src}}=-0.118,
\qquad
CC_{\text{AdaNODEs}}=-0.099
$$

이며

$$
CCC_{\text{Src}}=-0.096,
\qquad
CCC_{\text{AdaNODEs}}=-0.083
$$

입니다. 

수치는 개선되었지만 correlation 자체는 여전히 음수입니다.

따라서

> “상대적으로 좋아졌다”

와

> “좋은 예측 성능을 얻었다”

는 완전히 다른 주장입니다.

특히 0 근처 또는 음수인 correlation에 percentage improvement를 적용하면 해석이 불안정합니다.

---

## 5-6. 직접 비교하기 어려운 baseline이 섞여 있다

논문은 Source, DAF, TTT를 주요 비교 대상으로 사용합니다. 

하지만 이들은 정보 조건 자체가 다릅니다.

### DAF

DAF는 source와 target domain을 공동으로 활용하는 domain adaptation 방법입니다. ICML 2022 논문은 attention sharing 및 domain discriminator를 사용합니다. ([Proceedings of Machine Learning Research][2])

따라서

$$
\text{DAF: source+target data}
$$

와

$$
\text{AdaNODEs: source data 없음}
$$

은 동일 조건이 아닙니다.

---

### TTT

TTT는 원래 self-supervised auxiliary task를 source training 단계부터 설계해야 하는 방식입니다. ([Proceedings of Machine Learning Research][3])

AdaNODEs 논문은 이를 forecasting에 맞추기 위해 scaling auxiliary task로 변형했습니다. 

따라서 TTT의 낮은 성능이

$$
\text{TTT라는 방법 자체의 한계}
$$

인지

$$
\text{해당 auxiliary task 설계의 한계}
$$

인지 분리하기 어렵습니다.

---

## 5-7. 더 직접적인 2025년 TSF-TTA baseline들이 실험에서 빠져 있다

논문 Related Work에는

* TAFAS,
* PETSA,
* DynaTTA

를 직접 언급하지만, 실제 주요 실험 비교에서는 DAF와 TTT를 사용합니다. 

TAFAS는 AAAI 2025의 forecasting-specific TTA이고, PETSA와 DynaTTA도 2025년의 직접적인 TSF-TTA 연구입니다. ([AAAI Publications][4])

따라서 “SOTA baseline보다 우수하다”는 표현은 **동일한 protocol의 최신 직접 경쟁 방법과 대규모 비교가 부족하다는 점에서 보수적으로 해석해야 합니다.**

---

## 5-8. 메모리 효율성 주장을 직접 측정하지 않았다

$\alpha,\gamma$만 업데이트하므로 trainable parameter 수는 분명 매우 적습니다.

하지만 실제 inference cost는

$$
\text{gradient calculation}
+
\text{ODE integration}
+
\text{adaptive RK45 evaluations}
$$

에 의해 결정됩니다.

논문에는 제가 확인한 범위에서

* peak GPU memory,
* inference latency,
* number of function evaluations,
* adaptation time per batch

이 보고되지 않습니다.

따라서

$$
\text{few trainable parameters}
\not\equiv
\text{low wall-clock cost}
$$

입니다.

---

## 5-9. NLL의 uncertainty collapse 가능성이 분석되지 않았다

앞서 설명했듯 prediction mean을 자기 predictive Gaussian에 넣는다면

$$
\mathcal L_{\text{NLL}}
\propto
\log\sigma
$$

가 될 가능성이 있습니다.

따라서 adaptation이 실제 mean forecasting을 개선하는 대신

$$
\sigma\downarrow
$$

만 유도하는 것은 아닌지 calibration 평가가 필요합니다.

ECE와 같은 classification calibration metric 대신 forecasting에서는 예를 들어

* NLL,
* CRPS,
* prediction interval coverage,
* calibration curve

등이 필요하지만 제시되지 않았습니다.

---

## 5-10. Figure 4에는 내부 숫자 불일치가 있다

Figure 4(b)의 그림에는 severity level 5가

$$
dt=0.25
$$

로 표시되어 있습니다. 

그런데 바로 아래 본문에서는 severity level 5를

$$
dt=0.15
$$

라고 설명합니다. 

이는 첨부된 v1 기준으로 명백한 내부 표기 불일치이며, **Figure의 $dt=0.25$와 본문의 $dt=0.15$ 중 어느 것이 정확한지 논문만으로 확정할 수 없습니다.**

---

# 6. 이 논문이 답하지 않는 중요한 질문

1. $\alpha$와 $\gamma$는 정확히 scalar인가, latent dimension별 vector인가, broadcasting되는 parameter인가?
2. 왜 $\alpha$가 frequency/amplitude, $\gamma$가 phase를 안정적으로 의미한다고 볼 수 있는가?
3. $\alpha,\gamma$의 initialization은 무엇이며 adaptation 중 허용 범위나 regularization이 있는가?
4. $\alpha < 0$ 또는 지나치게 큰 $\alpha$가 발생하면 ODE 안정성은 어떻게 보장되는가?
5. $\lambda$는 어떻게 선택하며 test label 없이 optimal $\lambda$를 어떻게 결정하는가?
6. NLL minimization이 $\sigma\rightarrow0$의 over-confidence를 일으키지 않는다는 근거가 있는가?
7. KL loss에서 $y_T$가 정확히 어떤 prediction이며 gradient가 어떤 경로로 $\alpha,\gamma$에 전달되는가?
8. $\alpha$만 사용하거나 $\gamma$만 사용할 때의 ablation 결과는 무엇인가?
9. NLL-only, KL-only, NLL+KL의 ablation은 어떠한가?
10. abrupt regime shift, variance shift, cross-variable dependence shift에서도 효과적인가?
11. 실제 industrial/financial/weather/traffic 데이터에서도 동일한 개선이 나타나는가?
12. ODE solver의 tolerance와 step size가 adaptation 성능에 얼마나 영향을 주는가?
13. shift가 없을 때 불필요한 TTA가 source model을 손상시키지는 않는가?
14. 지속적 online adaptation에서 catastrophic drift가 누적되지는 않는가?
15. TTA hyperparameter는 source validation만으로 정했는가, 아니면 target test 성능을 보면서 선택했는가?

특히 마지막 문제는 중요합니다. 대규모 TTA benchmark 연구인 **On Pitfalls of Test-Time Adaptation**은 test-time hyperparameter/model selection 자체가 TTA 평가에서 매우 어려운 문제라고 지적합니다. ([ICML][5])

---

# 7. 가장 중요한 그림 5개 해석

이 논문에는 **Figure가 4개뿐**이므로 존재하지 않는 Figure 5를 만들어내지 않고, **Figure 1–4와 Table 1을 다섯 번째 핵심 도표**로 해석하겠습니다.

## 7-1. Figure 1 — 논문 전체를 이해하는 가장 중요한 그림

Figure 1은 세 부분입니다. 

### Figure 1(a)

Training에서는 상대적으로 느린 oscillation을 봤는데 test에서는 frequency가 크게 변한 사례를 보여줍니다.

즉

$$
D_{\text{train}}\neq D_{\text{test}}
$$

때문에 source forecaster의 미래 trajectory가 틀어지는 상황입니다.

### Figure 1(b)

논문의 핵심입니다.

$$
y_p
\rightarrow
\text{Encoder}
\rightarrow
z_{t_0}
\rightarrow
\boxed{\alpha z+\gamma}
\rightarrow
\text{NODE}
\rightarrow
\text{Decoder}
\rightarrow
\hat y
$$

를 보여줍니다.

특히 Encoder/NODE/Decoder 전체를 다시 학습하는 것이 아니라 **NODE 앞의 scaling/shift만 적응한다**는 것이 시각적으로 드러납니다.

### Figure 1(c)

forecast를 point estimate가 아닌 probability distribution으로 만들고

$$
\mathcal L^{\text{NLL}}
+
\mathcal L^{\text{KL}}
$$

로 적응하는 구조입니다.

**핵심 의미:** AdaNODEs의 새로움은 backbone 그 자체보다 **“어디를 적응할 것인가”와 “label 없이 무엇을 loss로 사용할 것인가”**에 있습니다.

---

# 7-2. Figure 2 — 다양한 shift에서 CCC 비교

Figure 2는 Linear, Sinusoidal, Oscillator에 대해 severity $L1$ – $L5$를 radar plot으로 비교합니다. 

Amplitude/frequency shift에서는 AdaNODEs가 대체로 가장 바깥쪽에 위치하여 높은 CCC를 보입니다.

Time-delay에서도 전반적으로 강하지만 저자 스스로

> sinusoidal severity 3과 4에서는 TTT가 더 우수했다

고 보고합니다. 

따라서 Figure 2가 말하는 것은

$$
\boxed{\text{AdaNODEs가 항상 이긴다}}
$$

가 아니라

$$
\boxed{
\text{설계한 temporal shift 전체에서 비교적 안정적인 성능을 보인다}
}
$$

에 가깝습니다.

다만 radar chart에는 정확한 수치와 uncertainty가 나타나지 않아 통계 검증에는 적합하지 않습니다.

---

# 7-3. Figure 3 — shift가 심해질수록 adaptation 가치가 커진다

Figure 3은 source model 대비 AdaNODEs의 relative improvement를 severity별로 나타냅니다. 

저자는 amplitude/frequency shift에서

$$
0.53\%\sim18\%
$$

time delay에서

$$
2.5\%\sim21.8\%
$$

의 향상을 보고합니다. 

가장 흥미로운 점은

$$
\text{shift severity}\uparrow
\Rightarrow
\text{adaptation benefit}\uparrow
$$

경향입니다.

이는 source model이 충분히 잘 작동하는 상황에서는 TTA의 이득이 작고,

$$
P_{\text{test}}
$$

가 source distribution에서 크게 벗어날수록 adaptation parameter의 가치가 커진다는 것을 의미합니다.

다만 error bar가 없으므로 severity에 따른 상승 추세의 통계적 안정성은 확인할 수 없습니다.

---

# 7-4. Figure 4 — latent dynamics 수정이 실제 trajectory를 바꾼다는 정성적 증거

Rotating MNIST에서 source model은 source rotation speed를 계속 유지하는 반면 AdaNODEs는 target의 느려진 회전 속도에 맞추어 trajectory를 늦춥니다. 

즉 AdaNODEs가 단순히 image appearance를 조정하는 것이 아니라

$$
\frac{dz}{dt}
$$

를 변경함으로써 **시간의 진행 속도 자체를 수정한다**는 아이디어를 직관적으로 보여줍니다.

severity 5에서는 예측 horizon을 더 길게 유지하는 모습도 보입니다. 

다만 앞서 언급했듯 그림에는 $dt=0.25$, 본문에는 $dt=0.15$가 기록되어 있습니다.

---

# 7-5. Table 1 — 가장 중요한 정량적 결과

Table 1에서 in-distribution 성능은 예를 들어

$$
MSE=0.023,\quad CC=0.728,\quad CCC=0.654
$$

입니다. 

severity가 높아지면서 source와 AdaNODEs 모두 상당한 성능 저하가 발생합니다.

예를 들어 amplitude/frequency severity 5에서

$$
CC_{\text{Src}}=0.146
$$

인데

$$
CC_{\text{Ada}}=0.196
$$

으로 개선됩니다.

그러나 in-distribution의

$$
0.728
$$

과 비교하면 여전히 큰 차이가 있습니다. 

따라서 Table 1의 정확한 결론은

> **AdaNODEs가 severe shift를 제거한다**

가 아니라

> **severe shift로 인해 발생한 성능 손실의 일부를 회복한다**

입니다.

이 차이가 매우 중요합니다.

---

# 8. 논문의 결론과 연구적 시사점

저자들은 AdaNODEs가 NODE를 이용하여 시간적 distribution shift의 특성을 직접 다루고, 새로운 label-free loss를 이용해 runtime adaptation을 수행함으로써 time-series forecasting뿐 아니라 regression TTA에도 새로운 가능성을 제공한다고 결론짓습니다. 

저자가 명시적인 장기 후속 연구 계획을 상세하게 나열하지는 않습니다. 따라서 “후속 연구 계획”이라는 표현으로 새로운 내용을 만들어내는 것은 부정확하며, **논문이 실제로 제시한 것은 future pathway에 가까운 일반적 방향성**입니다.

제가 보기에 이 연구의 가장 큰 학술적 의미는

$$
\boxed{
\text{데이터 분포를 직접 정규화하지 않고 dynamics 자체를 adaptation 대상으로 삼았다}
}
$$

는 것입니다.

---

# 8-1. 모델의 일반화 성능 향상 가능성

AdaNODEs가 일반화에 유리할 가능성이 있는 가장 중요한 이유는 **적응 자유도가 매우 낮기 때문**입니다.

전체 neural network parameter를

$$
\theta\in\mathbb R^P
$$

라고 하고 $P$가 수백만이라고 합시다.

AdaNODEs는 이를 고정하고 사실상

$$
(\alpha,\gamma)
$$

만 최적화합니다.

따라서 adaptation hypothesis space를 개념적으로

```math
\mathcal H_{\text{Ada}}
=
\{
f_{\theta,\alpha,\gamma}
:
\theta=\theta_{\text{source}},
\alpha,\gamma\text{ adaptive}
\}
```

로 제한합니다.

이러한 제한은 test sample이 적을 때 full fine-tuning보다 **과적합 위험을 줄이는 regularizer 역할**을 할 가능성이 큽니다.

단순한 통계적 직관으로는 추정해야 할 적응 자유도를 $d_{\text{adapt}}$라고 할 때 estimation uncertainty가 흔히

$$
O
\left(
\sqrt{
\frac{d_{\text{adapt}}}
{n_{\text{test}}}
}
\right)
$$

와 같은 방향으로 증가한다고 생각할 수 있습니다.

이 식은 AdaNODEs 논문에서 증명한 bound가 아니라 **왜 저차원 adaptation이 small-sample test stream에 유리할 수 있는지를 설명하기 위한 일반적인 통계적 직관**입니다.

---

## 그러나 $\alpha,\gamma$ 두 개만으로 모든 shift를 설명할 수는 없다

현재 형태는 사실상 latent state에 대한 affine transformation입니다.

$$
z
\rightarrow
\alpha z+\gamma
$$

따라서 다음과 같은 복잡한 shift에는 부족할 가능성이 있습니다.

$$
\text{variable-specific drift},
$$

$$
\text{cross-variable covariance drift},
$$

$$
\text{nonlinear regime transition},
$$

$$
P(Y\mid X)\text{ 자체의 변화}.
$$

마지막 경우를 특히 **concept shift**라고 합니다.

\*용어 메모 — **Concept shift**: 입력의 분포만 바뀌는 것이 아니라 동일한 입력 $X$에 대해 출력 $Y$를 만드는 관계 $P(Y\mid X)$ 자체가 바뀌는 현상입니다.*

---

## 제가 가장 권장하는 AdaNODEs 후속 구조

일반화 성능을 높이려면 단순히 $\alpha,\gamma$를 큰 neural adapter로 바꾸는 것보다 **저차원 구조를 유지하면서 표현력을 조금씩 늘리는 방향**이 적합합니다.

예를 들어 scalar $\alpha$ 대신

$$
\alpha
\rightarrow
\boldsymbol{\alpha}\in\mathbb R^d
$$

를 사용하되 shrinkage를 걸 수 있습니다.

```math
\mathcal L
=
\mathcal L_{\text{AdaNODE}}
+
\beta
\Vert
\boldsymbol{\alpha}-\mathbf 1
\Vert_2^2
+
\beta_\gamma
\Vert
\boldsymbol{\gamma}
\Vert_2^2
```

여기서

* $\mathbf1$: adaptation이 없는 상태
* $\beta,\beta_\gamma$: source dynamics에서 너무 멀리 이동하지 않게 만드는 regularization strength입니다.

더 발전시키면

$$
A=I+UV^\top
$$

인 low-rank adaptation을 이용하여

```math
\frac{dz}{dt}
=
f_{\text{node}}
\left(
Az+\gamma;\theta
\right)
```

로 만들 수 있습니다.

여기서

$$
U,V\in\mathbb R^{d\times r},
\qquad r\ll d
$$

로 두면 full $d\times d$ adaptation보다 훨씬 적은 파라미터로 변수 간 interaction까지 조절할 수 있습니다.

이 방식이 **현재 AdaNODEs의 parameter efficiency와 더 높은 shift 표현력을 동시에 유지할 가능성이 있는 가장 자연스러운 확장**이라고 판단합니다.

---

## 더욱 중요한 개선: “언제 적응하지 않을 것인가”

TTA에서

$$
\text{항상 adaptation}
$$

은 위험합니다.

shift가 작다면

$$
R_{\text{test}}(f_{\text{adapt}}) > R_{\text{test}}(f_{\text{source}})
$$

가 될 수도 있습니다.

따라서 shift score

```math
s_t
=
D
\left(
P_t(X),
P_{\text{source}}(X)
\right)
```

를 계산하고

```math
g_t
=
\sigma(a(s_t-\tau))
```

같은 gate를 도입한 뒤

```math
\alpha_t
=
1+g_t\Delta\alpha_t,
```

```math
\gamma_t
=
g_t\Delta\gamma_t
```

로 만들 수 있습니다.

shift가 없으면

$$
g_t\approx0
$$

이어서 source model로 돌아가고, shift가 커지면

$$
g_t\rightarrow1
$$

로 adaptation을 활성화합니다.

이 구조는 DynaTTA와 COSA에서 나타나는 **shift-aware/gated adaptation 철학**과도 연결됩니다. ([ICML][6])

---

# 8-2. 2020년 이후 관련 최신 연구와 비교

## 연구 흐름

| 연도   | 연구                                                                                           | Test-time 정보                | 핵심 adaptation                                                  | AdaNODEs와의 관계                                                                           |
| ---- | -------------------------------------------------------------------------------------------- | --------------------------- | -------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| 2020 | **Test-Time Training with Self-Supervision for Generalization under Distribution Shifts**    | unlabeled test sample       | auxiliary self-supervised task로 model update                   | TTA의 대표적 출발점, forecasting 전용 아님 ([Proceedings of Machine Learning Research][3])         |
| 2021 | **Tent: Fully Test-Time Adaptation by Entropy Minimization**                                 | unlabeled target            | entropy minimization, normalization parameter update           | AdaNODEs NLL confidence minimization의 사상적 선행 연구지만 classification 전용 ([ML Anthology][7]) |
| 2022 | **Domain Adaptation for Time Series Forecasting via Attention Sharing (DAF)**                | source+target               | attention latent alignment                                     | forecasting DA지만 source data 필요 ([Proceedings of Machine Learning Research][2])         |
| 2022 | **Self-Adaptive Forecasting for Improved Deep Learning on Non-Stationary Time-Series (SAF)** | test input/self-supervision | backcasting 기반 representation adaptation                       | TS non-stationarity 대응의 초기 흐름, 별도 training stage 필요 ([arXiv][8])                        |
| 2022 | **Continual Test-Time Domain Adaptation (CoTTA)**                                            | unlabeled stream            | teacher averaging + stochastic restoration                     | continual drift와 catastrophic forgetting 문제를 강조, vision 중심 ([Open Access CVF][9])       |
| 2023 | **On Pitfalls of Test-Time Adaptation**                                                      | 다양한 protocol                | TTAB benchmark                                                 | TTA의 hyperparameter selection과 평가 protocol 위험을 체계화 ([ICML][5])                          |
| 2025 | **TAFAS**                                                                                    | partially observed GT       | gated calibration + periodicity-aware adaptation               | forecasting-specific TTA의 직접 경쟁 축 ([AAAI Publications][4])                              |
| 2025 | **DynaTTA / TTFBench**                                                                       | online shifting stream      | shift severity estimation + dynamic adaptation rate/gating     | “shift의 크기에 따라 적응량을 바꾼다”는 발전 방향 ([ICML][6])                                             |
| 2025 | **PETSA**                                                                                    | partial + delayed full GT   | lightweight input/output adapters + Huber/frequency/patch loss | parameter-efficient TTA라는 점에서 AdaNODEs와 매우 직접적으로 비교 가능 ([OpenReview][10])               |
| 2026 | **AdaNODEs**                                                                                 | label-free target           | latent NODE의 $\alpha,\gamma$                                   | **미래 GT 없이 dynamics 자체를 수정한다는 것이 핵심 차별점** ([arXiv][11])                                 |
| 2026 | **COSA**                                                                                     | 최근 관측된 GT                   | output-space residual + context + gating                       | backbone을 건드리지 않고 output correction, ICLR 2026 ([Proceedings ICLR][12])                 |
| 2026 | **Towards Principled TTA / FAC**                                                             | matured GT only             | frequency-domain calibration                                   | TTA protocol 자체를 엄밀하게 다시 정의하고 spectral correction을 직접 parameterize ([arXiv][13])        |

---

## AdaNODEs와 TAFAS

TAFAS는 **Battling the Non-stationarity in Time Series Forecasting via Test-time Adaptation**, AAAI 2025입니다. TAFAS는 부분적으로 관측된 실제값을 활용하고 gated calibration을 사용하여 pre-trained forecaster를 적응시킵니다. ([AAAI Publications][4])

즉

$$
\text{TAFAS}
:
\text{revealed target information}
\rightarrow
\text{adaptation}
$$

인 반면 AdaNODEs는

$$
\text{AdaNODEs}
:
\text{prediction distribution/context}
\rightarrow
\text{adaptation}
$$

을 지향합니다.

따라서 **즉시 label이 없는 환경에서는 AdaNODEs가 더 엄격한 adaptation 조건**을 다룬다고 볼 수 있습니다.

하지만 실제 streaming forecasting에서는 시간이 지나면 정답이 결국 도착한다는 점에서 TAFAS 계열 역시 현실적입니다.

---

## AdaNODEs와 PETSA

PETSA는 input/output에 작은 calibration module을 추가하고, partial/full delayed ground truth를 이용합니다. PETSA의 objective에는 Huber, frequency-domain, patch-wise structural loss가 포함됩니다. ([OpenReview][10])

개념적으로

```math
\text{AdaNODEs}
=
\text{latent dynamics adaptation}
```

```math
\text{PETSA}
=
\text{input/output calibration}
```

이라는 차이가 있습니다.

AdaNODEs가 “왜 dynamics가 변했는지”를 모델 내부 시간축에서 흡수하는 데 장점이 있다면, PETSA는 architecture-agnostic deployment에 더 유리합니다.

---

## AdaNODEs와 DynaTTA

DynaTTA는 shift severity를 추정하여 adaptation rate와 gating을 바꿉니다. 또한 TTFBench라는 forecasting TTA benchmark를 제안했습니다. ([ICML][6])

AdaNODEs에는 현재

$$
\text{adapt or not?}
$$

을 판단하는 명시적 shift detector가 없습니다.

따라서 DynaTTA의

$$
\text{shift detection}
+
\text{AdaNODE latent dynamics}
$$

결합은 상당히 자연스러운 후속 연구가 될 수 있습니다.

---

## AdaNODEs와 COSA — 2026년 기준 매우 중요한 비교

ICLR 2026의 COSA는 frozen source model의 출력을 직접 수정합니다.

```math
\hat Y_t
=
Y_t^{(0)}
+
\tanh(g)H_t
```

이며 최근 ground truth의 통계 정보를 context로 사용합니다. 공식 ICLR 자료에서는 6개 dataset과 6개 architecture에서 non-TTA 대비 13.91–17.03%, 기존 TTA 대비 10.48–13.05% 개선을 보고합니다. ([Proceedings ICLR][12])

COSA의 강점은

$$
\text{간단함}
+
\text{architecture agnostic}
+
\text{실제 forecasting benchmarks}
$$

입니다.

반대로 AdaNODEs의 강점은

$$
\text{미래 실제값 없이 바로 adaptation 가능}
+
\text{continuous dynamics를 직접 조정}
$$

입니다.

---

# 2026년의 중요한 변화: FAC가 TTA protocol 자체를 문제 삼는다

**Towards Principled Test-Time Adaptation for Time Series Forecasting**은 2026년 5월 공개되었으며, 기존 TSF-TTA들이 사용하는 target information protocol이 서로 달라 비교가 깨끗하지 않다는 문제를 제기합니다. 그리고 **오직 matured ground truth만 사용하도록 protocol을 정리한 뒤 Frequency-Aware Calibration(FAC)**을 제안합니다. ([arXiv][13])

이 논문은 AdaNODEs를 평가할 때도 매우 중요합니다.

TTA 성능은 단순히

$$
\text{MSE}
$$

만 비교해서는 안 되고,

$$
\boxed{
\text{adaptation 시점에 어떤 정보를 사용할 수 있었는가}
}
$$

가 동일해야 합니다.

즉 앞으로는 다음 세 protocol을 분리해서 비교하는 것이 좋습니다.

$$
\text{A. strictly label-free TTA}
$$

$$
\text{B. partial/revealed GT TTA}
$$

$$
\text{C. matured-GT online adaptation}
$$

AdaNODEs의 가장 명확한 위치는 **A**입니다.

---

# AdaNODEs가 앞으로의 연구에 미치는 영향

AdaNODEs의 가장 중요한 영향은 TTA를

$$
\text{normalization}
$$

이나

$$
\text{output calibration}
$$

만의 문제로 보지 않고

$$
\boxed{
\text{temporal dynamics adaptation}
}
$$

문제로 재정의했다는 점입니다.

이는 특히 다음 유형의 시계열에서 유용할 가능성이 있습니다.

$$
\text{oscillatory process},
$$

$$
\text{physical dynamics},
$$

$$
\text{irregular temporal dynamics},
$$

$$
\text{changing characteristic time scale}.
$$

Neural ODE는 시간 변화율을 직접 모델링하기 때문입니다.

---

# 제가 제안하는 가장 유망한 후속 연구 구조

2026년 현재 연구들을 종합하면 AdaNODEs를 단독 확장하기보다 **두 단계 adaptation**으로 발전시키는 것이 가장 타당해 보입니다.

### Stage A — 정답이 아직 없을 때

AdaNODEs를 이용합니다.

```math
(\alpha_t,\gamma_t)
=
\arg\min
\mathcal L_{\text{self}}
```

그리고

```math
\hat y_t^{A}
=
f_{\text{dec}}
\left(
z_{\alpha_t,\gamma_t}(t)
\right)
```

를 얻습니다.

### Stage B — 과거 prediction의 정답이 충분히 mature된 후

COSA/FAC류의 residual correction을 추가합니다.

```math
\hat y_t
=
\hat y_t^{A}
+
g_t
\Delta_\omega
\left(
\hat y_t^{A},
c_t
\right)
```

여기서

* $c_t$: 최근 실제값 통계 또는 spectral context
* $\Delta_\omega$: 작은 output adapter
* $g_t$: shift-dependent gate입니다.

전체 objective는 예를 들어

```math
\mathcal L_t
=
\mathcal L_{\text{Ada}}
+
\eta
I_t^{\text{mature}}
\mathcal L_{\text{sup}}
+
\beta
\Vert\alpha_t-1\Vert_2^2
+
\beta_\gamma
\Vert\gamma_t\Vert_2^2
```

처럼 구성할 수 있습니다.

여기서

```math
I_t^{\text{mature}}
=
\begin{cases}
1,&\text{과거 forecast의 실제값이 완전히 도착함}\\
0,&\text{아직 정답을 사용할 수 없음}
\end{cases}
```

입니다.

이 구조의 장점은

$$
\boxed{
\text{즉시 대응}
+
\text{지연된 실제값을 이용한 오류 수정}
+
\text{source model 보존}
}
$$

을 동시에 얻을 수 있다는 점입니다.

---

# 앞으로 연구할 때 반드시 고려해야 할 항목

AdaNODEs 후속 연구에서 단순히 Test MSE가 좋아졌다는 것만으로는 충분하지 않습니다. 최소한 다음을 함께 검증해야 합니다.

* **real-world multivariate datasets**와 synthetic controlled shifts를 모두 사용하고,
* no-shift / mild-shift / severe-shift를 분리하며,
* $\alpha$ only, $\gamma$ only, NLL only, KL only의 ablation을 수행하고,
* trainable parameter 수뿐 아니라 **peak memory, latency, ODE NFE**를 측정하며,
* test label을 이용한 hyperparameter tuning을 금지하고,
* adaptation failure 시 source model로 돌아가는 rollback mechanism을 넣고,
* MSE/MAE뿐 아니라 distributional calibration을 평가하며,
* mixed shift 및 abrupt regime change를 실험하고,
* TAFAS/PETSA/DynaTTA/COSA/FAC와 **동일한 target-information protocol**에서 비교해야 합니다.

TTA 평가에서 hyperparameter selection과 protocol 차이가 성능을 크게 왜곡할 수 있다는 문제는 TTAB 연구에서도 명확히 지적되었습니다. ([ICML][5])

---

# 최종 평가

**AdaNODEs는 매우 흥미로운 논문이지만, 현재 증거 수준에서는 “새로운 forecasting SOTA”라기보다 “Neural ODE의 latent dynamics를 매우 소수의 parameter로 test-time에 조정하는 새로운 TTA 메커니즘”으로 평가하는 것이 가장 정확합니다.**

특히

$$
\boxed{
\theta\text{ 전체를 바꾸지 않고 }
\alpha,\gamma
\text{만 바꾼다}
}
$$

는 설계는 small-sample adaptation과 catastrophic forgetting 억제 측면에서 매우 매력적입니다.

반면 일반화 성능을 강하게 주장하려면 현재의 synthetic/Rotating-MNIST 실험에서 벗어나

$$
\boxed{
\text{real-world multivariate TS}
+
\text{protocol-clean comparison}
+
\text{adaptation stability}
+
\text{uncertainty calibration}
}
$$

검증이 반드시 필요합니다.

2026년의 COSA와 FAC까지 고려하면 향후 가장 유망한 방향은 **“AdaNODEs의 즉각적인 label-free latent-dynamics adaptation + matured ground truth가 도착한 이후의 lightweight output/frequency calibration”**을 결합하는 것입니다. COSA는 output-space correction의 실용성과 실제 benchmark 확장성을 보여주었고, FAC는 TTA 비교에서 supervision protocol 자체를 명확하게 통제해야 한다는 방향을 제시합니다. ([Proceedings ICLR][12])

---

# 참고자료 및 확인한 사이트

1. **Ting Dang et al., “AdaNODEs: Test Time Adaptation for Time Series Forecasting Using Neural ODEs”**, 첨부 arXiv:2601.12893v1 및 arXiv.  ([arXiv][11])
2. **“AdaNODEs: Test Time Adaptation for Time Series Forecasting Using Neural ODEs” — ICASSP 2026 official program**, IEEE ICASSP 2026. ([CMS Workshops][1])
3. **Jin et al., “Domain Adaptation for Time Series Forecasting via Attention Sharing”**, ICML 2022 / PMLR. ([Proceedings of Machine Learning Research][2])
4. **Sun et al., “Test-Time Training with Self-Supervision for Generalization under Distribution Shifts”**, ICML 2020 / PMLR. ([Proceedings of Machine Learning Research][3])
5. **Wang et al., “Tent: Fully Test-Time Adaptation by Entropy Minimization”**, ICLR 2021. ([ML Anthology][7])
6. **Arik et al., “Self-Adaptive Forecasting for Improved Deep Learning on Non-Stationary Time-Series”**, arXiv:2202.02403. ([arXiv][8])
7. **Wang et al., “Continual Test-Time Domain Adaptation”**, CVPR 2022. ([Open Access CVF][9])
8. **Zhao et al., “On Pitfalls of Test-Time Adaptation”**, ICML 2023 / TTAB. ([ICML][5])
9. **Kim et al., “Battling the Non-stationarity in Time Series Forecasting via Test-time Adaptation”**, AAAI 2025 — TAFAS. ([AAAI Publications][4])
10. **Medeiros et al., “Accurate Parameter-Efficient Test-Time Adaptation for Time Series Forecasting”**, ICML 2025 PUT Workshop — PETSA. ([OpenReview][10])
11. **Grover & Etemad, “Shift-Aware Test Time Adaptation and Benchmarking for Time-Series Forecasting”**, ICML 2025 PUT Workshop — DynaTTA/TTFBench. ([ICML][6])
12. **Im & Kwon, “COSA: Context-aware Output-Space Adapter for Test-Time Adaptation in Time Series Forecasting”**, ICLR 2026. ([Proceedings ICLR][12])
13. **Wang et al., “Towards Principled Test-Time Adaptation for Time Series Forecasting”**, arXiv:2605.17250, 2026 — FAC. ([arXiv][13])

원하시면 이 분야의 **AdaNODEs·COSA·FAC 이후 후속 논문을 주기적으로 추적**할 수도 있습니다.

[1]: https://cmsworkshops.com/ICASSP2026/view_paper.php?PaperNum=4334 "https://cmsworkshops.com/ICASSP2026/view_paper.php?PaperNum=4334"
[2]: https://proceedings.mlr.press/v162/jin22d.html "https://proceedings.mlr.press/v162/jin22d.html"
[3]: https://proceedings.mlr.press/v119/sun20b.html "https://proceedings.mlr.press/v119/sun20b.html"
[4]: https://ojs.aaai.org/index.php/AAAI/article/view/33965 "https://ojs.aaai.org/index.php/AAAI/article/view/33965"
[5]: https://icml.cc/virtual/2023/poster/23602 "https://icml.cc/virtual/2023/poster/23602"
[6]: https://icml.cc/virtual/2025/48077 "https://icml.cc/virtual/2025/48077"
[7]: https://mlanthology.org/iclr/2021/wang2021iclr-tent/ "https://mlanthology.org/iclr/2021/wang2021iclr-tent/"
[8]: https://arxiv.org/abs/2202.02403 "https://arxiv.org/abs/2202.02403"
[9]: https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Continual_Test-Time_Domain_Adaptation_CVPR_2022_paper.html "https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Continual_Test-Time_Domain_Adaptation_CVPR_2022_paper.html"
[10]: https://openreview.net/pdf?id=uFj4EL4GTB "https://openreview.net/pdf?id=uFj4EL4GTB"
[11]: https://arxiv.org/abs/2601.12893 "https://arxiv.org/abs/2601.12893"
[12]: https://proceedings.iclr.cc/paper_files/paper/2026/hash/2a8ce71baac4c89bf9ff479d8240c7d9-Abstract-Conference.html "https://proceedings.iclr.cc/paper_files/paper/2026/hash/2a8ce71baac4c89bf9ff479d8240c7d9-Abstract-Conference.html"
[13]: https://arxiv.org/abs/2605.17250 "https://arxiv.org/abs/2605.17250"
