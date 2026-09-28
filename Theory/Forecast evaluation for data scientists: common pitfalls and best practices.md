# Forecast evaluation for data scientists: common pitfalls and best practices

논문은 Hansika Hewamalage, Klaus Ackermann, Christoph Bergmeir의 *Forecast evaluation for data scientists: common pitfalls and best practices*, *Data Mining and Knowledge Discovery* 37, 788–832이며 2022년 12월 온라인 공개, 2023년 권호에 수록되었습니다. 

페이지 표기는 **학술지 인쇄 페이지 788–832 기준**입니다. 중요한 점은 이 논문이 새로운 forecasting model을 제안하는 논문이 아니라, **시계열 예측 모델의 성능과 일반화 성능을 어떻게 올바르게 검증할 것인가를 다루는 tutorial/review + 실증적 반례 논문**이라는 점입니다. 저자 역시 연구 범위를 주로 **point forecast evaluation**에 한정합니다. 

---

# 1. Executive Summary — 10문장 이내

1. 이 논문의 핵심 목적은 ML/DL 연구자들이 일반적인 회귀 문제의 검증 방법을 시계열 예측에 그대로 적용하면서 발생시키는 **잘못된 성능 평가와 허위 우수성(spurious superiority)** 문제를 바로잡는 것입니다. :chatgpt-content-reference{index="3"}  
2. 시계열은 관측치가 서로 독립적이지 않고 추세, 계절성, 단위근, 이분산성, 구조적 변화와 같은 **비정상성(non-stationarity)**을 가지므로 일반적인 i.i.d. 데이터와 동일한 평가법을 사용할 수 없습니다. :chatgpt-content-reference{index="4"}  
3. 저자들은 복잡한 Transformer나 Neural Network가 단순한 **naïve forecast**조차 이기지 못하는 사례를 보여 주면서, 올바른 benchmark 선택이 새로운 모델 구조만큼 중요하다고 주장합니다. :chatgpt-content-reference{index="5"}  
4. 실제로 random-walk 모의실험에서는 naïve 모델의 RMSE가 0.96으로 RF 1.01, SVM 1.00, NN 0.98보다 좋았고, 환율 데이터에서도 naïve 모델이 Autoformer를 모든 예측 horizon에서 크게 앞섰습니다. :chatgpt-content-reference{index="6"} :chatgpt-content-reference{index="7"}  
5. 평가 데이터 분할에서는 충분한 데이터가 있다면 **rolling-origin/time-series cross-validation(tsCV)**을 우선 권고하되, 순수 autoregressive 문제이고 잔차 자기상관이 남지 않는 경우에는 짧은 시계열에서 k-fold CV도 유효할 수 있다고 설명합니다. :chatgpt-content-reference{index="8"}  
6. RMSE, MAE, MAPE, MASE 등 어떤 단일 평가 지표도 모든 시계열 특성에 안전하지 않으며, 모델의 최적화 loss와 최종 평가 metric의 통계적 의미도 맞춰야 합니다. :chatgpt-content-reference{index="9"}  
7. 특히 normalization, decomposition, smoothing 등을 전체 데이터에 먼저 적용하면 미래 정보가 학습 과정으로 유입되어 실제보다 성능이 좋아 보이는 **data leakage**가 발생할 수 있음을 실험적으로 보여 줍니다. :chatgpt-content-reference{index="10"}  
8. 따라서 논문의 실질적인 메시지는 “더 복잡한 모델을 만드는 것”보다 **적절한 baseline, 시간 순서에 맞는 validation, leakage-free preprocessing, 데이터 특성에 맞는 metric, 통계적 유의성 검정**을 함께 사용해야 진짜 일반화 성능을 판별할 수 있다는 것입니다. :chatgpt-content-reference{index="11"}  
9. 2024–2026년 TFB, GIFT-Eval, ProbTS와 foundation-model benchmark 연구들은 이 문제의식을 더 큰 데이터·zero-shot·pretraining contamination 문제로 확장하고 있어, 이 논문의 문제제기는 현재 시계열 foundation model 평가에도 직접적으로 유효합니다. :chatgpt-content-reference{index="12"}

**용어 — Point forecast**  
예측분포 전체를 출력하는 것이 아니라 미래값 하나, 예를 들어 조건부 평균 또는 중앙값을 $\hat y_{t+h}$ 형태로 예측하는 방식입니다.

**용어 — Spurious result**  
겉으로는 모델 A가 모델 B보다 좋아 보이지만, 실제 예측정보 때문이 아니라 잘못된 benchmark, leakage, 작은 표본, 우연한 seed 등에 의해 생긴 허위 성능 차이를 의미합니다.

---

# 1-1. 연구 목적과 필요성

## 연구 목적

이 논문은 다음의 지식 격차를 해결하려고 합니다.

전통적인 forecasting 분야는 통계학·계량경제학을 중심으로 오랫동안 발전했습니다. 반면 최근 시계열 분야에 유입된 ML/DL 연구자들은 neural network나 Transformer architecture에는 익숙하지만, forecast origin, rolling-origin evaluation, naïve benchmark, unit root, scale-free metric과 같은 전통적 forecasting evaluation 원칙에는 상대적으로 익숙하지 않을 수 있습니다. 저자들은 그 결과 실제로 경쟁력이 없는 모델이 “state of the art”처럼 보일 수 있다고 지적합니다. :chatgpt-content-reference{index="13"}

즉 이 논문의 연구 질문은

> **“우리가 보고 있는 성능 향상이 정말 모델의 예측능력 향상인가, 아니면 평가 프로토콜 때문에 만들어진 착시인가?”**

라고 요약할 수 있습니다.

---

## 왜 필요한가?

일반 회귀에서는 흔히

$$
D=D_{\text{train}}\cup D_{\text{valid}}\cup D_{\text{test}}
$$

를 구성하고, i.i.d.를 전제로 random split이나 k-fold CV를 사용합니다.

하지만 시계열에서는

$$
P(Y_t)\neq P(Y_{t+\Delta})
$$

가 될 수 있습니다.

즉 시간이 변함에 따라 데이터 생성분포 자체가 달라질 수 있습니다.

논문은 강정상성(strong stationarity)을 Eq. (1)로 다음과 같이 표현합니다.

```math
F_Y
\left(
y_{t+\tau},
y_{t+1+\tau},
\dots,
y_{t+n+\tau}
\right)
=
F_Y
\left(
y_t,
y_{t+1},
\dots,
y_{t+n}
\right)
```

for all

$$
\tau\in\mathbb Z,\qquad n\in\mathbb N.
$$

논문 p.793, Eq. (1). :chatgpt-content-reference{index="14"}

### 기호 설명

- $Y_t$: 시간 $t$에서의 확률변수입니다.
- $y_t$: 실제 관측값입니다.
- $F_Y$: 여러 시점 값을 동시에 고려한 **joint cumulative distribution function**, 즉 결합누적분포함수입니다.
- $\tau$: 관측 창을 시간축에서 얼마나 이동했는지를 나타냅니다.
- $n$: 한 번에 보는 시계열 구간의 길이입니다.
- $\mathbb Z$: 정수 집합입니다.
- $\mathbb N$: 자연수 집합입니다.

위 관계가 모든 $\tau$에 대해 유지되지 않으면 시간에 따라 분포가 변하는 것이므로 이 논문에서는 이를 **non-stationarity**라고 부릅니다.

**용어 — Non-stationarity**  
시간이 지나면서 평균, 분산, 계절 패턴 또는 전체 확률분포가 변하는 현상입니다. 단순히 “그래프가 흔들린다”는 의미가 아닙니다.

---

# 2. 핵심 주장과 근거

| 핵심 주장 | 저자의 근거 | 정량적/논리적 결과 | 위치 |
|---|---|---|---|
| 복잡한 모델은 반드시 단순 baseline보다 좋지 않다 | random walk에서 RF/SVM/NN과 naïve 비교 | RMSE: RF 1.01, SVM 1.00, NN 0.98, **Naïve 0.96** | p.795–796, Eq.2–3, Fig.5, Table 1 :chatgpt-content-reference{index="15"} |
| 금융·random-walk형 데이터에서는 naïve benchmark가 필수다 | Autoformer 환율 재평가 | horizon 720 MSE: naïve 0.817 vs Autoformer rerun 1.552 | p.797–798, Table 3 :chatgpt-content-reference{index="16"} :chatgpt-content-reference{index="17"} |
| 논문의 원래 benchmark가 약하면 후속 연구 전체가 잘못된 비교를 이어받을 수 있다 | FiLM 재실험 및 Autoformer 사례 | FiLM 원 논문 결과 재현 실패; 5개 trial 모두 naïve보다 열세 | p.797–798, Table 2 | :chatgpt-content-reference{index="18"} :chatgpt-content-reference{index="19"} |
| 계절 데이터에는 계절 구조를 처리하는 benchmark가 필요하다 | Informer와 DHR-ARIMA 비교 | ETTh1 MSE 0.269→**0.140**, ECL 0.582→**0.433** | p.799–800, Table 4 :chatgpt-content-reference{index="20"} |
| 작은 수의 시계열만으로 SOTA를 주장하면 일반화 근거가 약하다 | 여러 ML 연구 사례 검토 | Zhang et al.은 단 3개 series 사용 사례로 지적됨 | p.801 :chatgpt-content-reference{index="21"} |
| 평가 metric은 데이터 특성에 따라 달라져야 한다 | R², MAPE, MSE/MAE 등의 실패 사례 | random walk에서 높은 $R^2$도 오해 가능, $y_t\simeq0$에서는 MAPE 발산 | p.801–802; Table 8–9 :chatgpt-content-reference{index="22"} |
| Forecast plot만 보고 모델을 판단하면 안 된다 | naïve/ETS plot 비교 | 시각적으로 naïve가 좋아 보여도 ETS RMSE=29.93로 더 우수 | p.802–804, Fig.6–7, Tables 5–6 :chatgpt-content-reference{index="23"} |
| preprocessing도 train/test split 이후 각각 적용해야 한다 | EMD leakage 실험 | RMSE: leakage 3.12, no leakage 5.65, naïve 3.46 | p.804–806, Fig.8/Table 7 :chatgpt-content-reference{index="24"} |
| 충분한 데이터에서는 tsCV가 기본 선택이다 | rolling origin 분석 | 미래 시점에 더 가까운 다수 origin에서 반복 검증 가능 | p.807–812, Fig.9–11 :chatgpt-content-reference{index="25"} |
| k-fold CV가 시계열에서 절대 금지되는 것은 아니다 | pure AR + 잔차 독립 조건 | Ljung–Box 등을 이용해 잔차 자기상관이 없음을 확인해야 함 | p.809–810 :chatgpt-content-reference{index="26"} |
| 어떤 validation도 미래의 전례 없는 structural break를 완벽히 예측하지 못한다 | non-stationary validation 논의 | tsCV도 실제 generalization error를 심하게 과소평가할 수 있음 | p.810–811 :chatgpt-content-reference{index="27"} |
| 통계적 유의성까지 확인해야 한다 | DM/Wilcoxon/Friedman/post-hoc/CD diagram | 단순 평균 metric ranking만으로 우월성을 확정하면 안 됨 | p.821–826, Fig.13–14 :chatgpt-content-reference{index="28"} :chatgpt-content-reference{index="29"} |

---

# 2-1. 해결 문제, 제안 방법, 수식, 구조, 성능 향상, 한계

## 2-1-1. 해결하고자 하는 문제

이 논문에서 문제는 forecast model 자체가 아닙니다.

논문이 해결하려는 것은 다음 관계입니다.

$$
\text{Observed benchmark score}
\neq
\text{True future generalization performance}
$$

즉 validation/test score가 낮다고 반드시 모델이 나쁜 것도 아니고, 높다고 반드시 모델이 좋은 것도 아닙니다.

성능 측정값은 사실상

```math
\hat R
=
f(
\text{split},
\text{benchmark},
\text{metric},
\text{preprocessing},
\text{horizon},
\text{random seed},
\text{data regime}
)
```

와 같이 평가 설계에 의존합니다.

이 식은 **논문에 직접 제시된 공식이 아니라 논문의 논리를 제가 수학적으로 요약한 표현**입니다.

**용어 — Generalization performance**  
학습·검증에 이미 사용한 데이터가 아니라, 앞으로 도착하는 진짜 새로운 데이터에서 모델이 얼마나 잘 작동하는지를 의미합니다.

---

# 2-1-2. “제안 모델 구조”가 아니라 “평가 파이프라인 구조”

이 논문에는 새로운 Transformer block이나 새로운 loss network가 없습니다.

논문의 Fig.11, Fig.12, Fig.14를 종합하면 구조는 다음과 같습니다.

$$
\boxed{
\text{Time-series characterization}
}
$$

$$
\downarrow
$$

$$
\boxed{
\text{Data partition selection}
}
$$

$$
\downarrow
$$

$$
\boxed{
\text{Forecast model + proper baseline}
}
$$

$$
\downarrow
$$

$$
\boxed{
\text{Forecast error}
}
$$

$$
\downarrow
$$

$$
\boxed{
\text{Appropriate error measure}
}
$$

$$
\downarrow
$$

$$
\boxed{
\text{Statistical significance test}
}
$$

이 다섯 단계가 논문의 실제 “architecture”에 해당합니다. 논문도 평가 과정을 Data partitioning → Forecasting → Error calculation → Error-measure calculation → optional significance test 순으로 정리합니다. :chatgpt-content-reference{index="30"}

---

## 2-1-3. Naïve benchmark와 random walk

논문의 Eq. (2)는 random walk를

$$
y_{t+1}=y_t+\epsilon_t
$$

로 나타냅니다.

그리고 Eq. (3)의 naïve forecast는

$$
\hat y_{t+h}=y_t
$$

입니다. :chatgpt-content-reference{index="31"}

### 기호 설명

- $y_t$: 현재 시점에서 마지막으로 알고 있는 실제값입니다.
- $y_{t+1}$: 다음 시점의 실제값입니다.
- $\epsilon_t$: 평균이 0인 예측 불가능한 innovation 또는 white-noise 성분입니다.
- $\hat y_{t+h}$: 현재 $t$에서 $h$ step 미래를 예측한 값입니다.
- $h$: forecast horizon입니다.

만약

$$
E[\epsilon_t]=0
$$

이고 $\epsilon_t$에 미래를 예측할 수 있는 구조가 없다면

$$
E[y_{t+h}\mid y_t]=y_t
$$

이므로 마지막 값을 그대로 사용하는 naïve forecast가 조건부 평균 관점에서 최적입니다.

따라서 복잡한 NN이 random walk를 학습해서 naïve보다 조금 좋아 보인다면 먼저 생각해야 할 것은

> “신경망이 숨은 패턴을 발견했는가?”

가 아니라

> “평가 변동, leakage 또는 우연한 표본 효과인가?”

입니다.

**용어 — Random walk**  
현재값에 예측할 수 없는 충격이 순차적으로 누적되는 과정입니다. 수준(level)은 크게 움직일 수 있지만 다음 충격 자체는 예측할 수 없을 수 있습니다.

**용어 — Innovation**  
과거 정보로 설명되지 않는 새롭게 들어온 충격입니다. 모델링 관점에서는 예측오차의 “새 정보” 부분입니다.

---

# 2-1-4. Forecast error와 bias

논문 Eq. (5):

$$
e_t=y_t-\hat y_t
$$

여기서

- $e_t$: 시점 $t$의 forecast error
- $y_t$: 실제값
- $\hat y_t$: 예측값

입니다. :chatgpt-content-reference{index="32"}

Bias를 확인하는 Mean Error는 Eq. (4):

```math
\text{ME}
=
\frac{1}{n}
\sum_{t=1}^{n}
(y_t-\hat y_t)
```

입니다. :chatgpt-content-reference{index="33"}

- $n$: 평가에 사용되는 전체 forecast error 개수입니다.
- $\text{ME} > 0$: 정의상 실제값이 예측보다 평균적으로 큽니다. 즉 과소예측 경향이 있습니다.
- $\text{ME} < 0$: 평균적으로 과대예측하는 경향입니다.

RMSE나 MAE가 작더라도 ME가 한 방향으로 치우쳐 있으면 실제 공정·재고·수요 운영에서는 문제가 될 수 있다는 것이 저자들의 지적입니다.

---

# 2-1-5. 대표 평가 지표

## RMSE

```math
\text{RMSE}
=
\sqrt{
\frac{1}{n}
\sum_{t=1}^{n}
e_t^2
}
```

큰 오차에 제곱 패널티를 주기 때문에 대형 오차를 강하게 싫어하는 상황에 유용합니다.

### 단점

서로 단위나 scale이 다른 여러 series를 단순 통합하면 큰 scale을 가진 series가 결과를 지배합니다. Table 8에서 scale-dependent measure로 분류됩니다. :chatgpt-content-reference{index="34"}

---

## MAE

```math
\text{MAE}
=
\frac{1}{n}
\sum_{t=1}^{n}|e_t|
```

RMSE보다 outlier의 영향이 약합니다.

또한 논문이 강조하듯 squared-error loss는 조건부 평균과, absolute-error loss는 조건부 중앙값과 연결됩니다. 따라서 최종 평가가 MAE인데 학습은 MSE만 최적화하는 경우 “무엇을 최적으로 예측하고 싶은가”가 일치하는지 점검해야 합니다. :chatgpt-content-reference{index="35"}

---

## MAPE

논문의 percentage error:

```math
p_t
=
100\frac{e_t}{y_t}
```

따라서

```math
\text{MAPE}
=
\frac{1}{n}
\sum_{t=1}^{n}
\left|
100\frac{e_t}{y_t}
\right|
```

입니다. :chatgpt-content-reference{index="36"} :chatgpt-content-reference{index="37"}

문제는

$$
y_t\rightarrow0
$$

일 때

$$
\frac{|e_t|}{|y_t|}\rightarrow\infty
$$

가 될 수 있다는 것입니다.

따라서 값이 0에 가깝거나 0을 통과하는 시계열에서는 MAPE가 부적절합니다. 논문은 실제로 $[-1,1]$ 수준의 값에 MAPE를 사용하는 사례들을 문제로 지적합니다. :chatgpt-content-reference{index="38"}

---

## sMAPE

```math
\text{sMAPE}
=
\frac{1}{n}
\sum_{t=1}^{n}
\frac{200|e_t|}
{|y_t|+|\hat y_t|}
```

Table 8에 제시됩니다. :chatgpt-content-reference{index="39"}

MAPE의 비대칭 문제를 어느 정도 줄이지만, denominator가 작아지는 문제를 완전히 제거하는 만능 metric은 아닙니다.

---

## Relative error

benchmark $b$의 오차를 $e_t^b$라고 하면 논문 Eq. (9)는

$$
r_t=\frac{e_t}{e_t^b}
$$

입니다. :chatgpt-content-reference{index="40"}

예를 들어

```math
\text{RelMAE}
=
\frac{\text{MAE}}
{\text{MAE}_b}
```

로 쓰면,

- $\text{RelMAE} < 1$: 제안 모델이 benchmark보다 우수
- $\text{RelMAE}=1$: benchmark와 동일
- $\text{RelMAE} > 1$: benchmark보다 열세

입니다. Table 8에 같은 형태가 정리되어 있습니다. :chatgpt-content-reference{index="41"}

이 방식의 장점은 모델 자체의 숫자보다

> “naïve 대비 실제로 얼마만큼 개선되었는가?”

를 바로 평가할 수 있다는 점입니다.

---

## MASE

논문의 Eq. (10)과 Table 8이 의도하는 non-seasonal naïve scaling은 통상 다음과 같이 정리할 수 있습니다.

먼저 학습구간 naïve MAE scale을

```math
s_{\text{MAE}}
=
\frac{1}{T-1}
\sum_{t=2}^{T}
|y_t-y_{t-1}|
```

로 둡니다.

그 다음

```math
\text{MASE}
=
\frac{1}{n}
\sum_{i=1}^{n}
\frac{|e_i|}
{s_{\text{MAE}}}
```

입니다.

- $T$: training-region 길이
- $n$: 평가 오차 개수
- $s_{\text{MAE}}$: training data에서 naïve forecast가 만들어내는 평균 절대오차 scale입니다.

논문의 parsed text에서는 Eq. (10)의 numerator 부호 표현이 다소 불명확하게 추출되어 있으므로, 위 식은 Table 8이 참조하는 Hyndman–Koehler의 통상적인 absolute scaled error 형태로 명확하게 적었습니다. 논문 본문은 Eq. (10)과 Table 8에서 benchmark 기반 scaled error/MASE를 설명합니다. :chatgpt-content-reference{index="42"} :chatgpt-content-reference{index="43"}

**용어 — Scale-free metric**  
원래 단위 자체보다 기준값이나 baseline error로 나누어 서로 다른 규모의 시계열을 비교할 수 있게 만든 지표입니다.

---

# 2-1-6. Data partitioning

## Fixed origin

한 번의 마지막 training point만 사용합니다.

$$
\underbrace{y_1,\ldots,y_T}_{\text{train}}
\quad\Big|\quad
\underbrace{y_{T+1},\ldots,y_{T+h}}_{\text{test}}
$$

구현은 빠르지만 forecast horizon의 특정 한 구간에 너무 의존하기 때문에 일반화 오차 추정의 variance가 커질 수 있습니다. :chatgpt-content-reference{index="44"}

---

## Rolling origin / tsCV

예를 들어 horizon $h=2$라면

$$
D_1:
(y_1,\ldots,y_T)
\rightarrow
(y_{T+1},y_{T+2})
$$

$$
D_2:
(y_1,\ldots,y_{T+1})
\rightarrow
(y_{T+2},y_{T+3})
$$

$$
D_3:
(y_1,\ldots,y_{T+2})
\rightarrow
(y_{T+3},y_{T+4})
$$

처럼 forecast origin을 앞으로 이동합니다.

Fig.2가 fixed origin과 rolling origin을 시각적으로 명확하게 구분합니다. :chatgpt-content-reference{index="45"}

**용어 — Forecast origin**  
예측을 시작하는 마지막 관측 시점입니다.

**용어 — tsCV**  
Time-series cross-validation입니다. 미래를 과거보다 먼저 학습하지 않으면서 forecast origin을 여러 번 이동시켜 일반화 오차를 반복 측정합니다.

---

## Expanding window와 Rolling window

Expanding:

$$
\{1,\ldots,T\}
\rightarrow
\{1,\ldots,T+1\}
\rightarrow
\{1,\ldots,T+2\}
$$

Rolling:

$$
\{1,\ldots,T\}
\rightarrow
\{2,\ldots,T+1\}
\rightarrow
\{3,\ldots,T+2\}
$$

논문은 작은 데이터에서는 과거 관측치를 버리지 않는 expanding window가 유리할 수 있으며, 오래된 regime의 정보가 현재와 맞지 않는다면 rolling window가 유리할 수 있다고 설명합니다. :chatgpt-content-reference{index="46"}

이는 **일반화 성능** 측면에서 매우 중요합니다.

전체 과거가 항상 동일하게 유효하다는 가정은

```math
P_{\text{past}}(X,Y)
=
P_{\text{future}}(X,Y)
```

를 암묵적으로 요구합니다.

concept drift가 존재한다면 이 가정이 깨집니다.

**용어 — Concept drift**  
시간이 지나면서 입력 $X$, 목표 $Y$, 또는 $P(Y|X)$의 관계가 변하는 현상입니다.

---

# 2-1-7. k-fold CV는 “시계열에서는 무조건 금지”인가?

이 논문에서 흥미로운 부분입니다.

저자들은 **순수 AR 모델**이고 잔차에 남은 serial correlation이 없으면 standard k-fold CV도 유효할 수 있다고 설명합니다. 특히 시계열이 너무 짧아 tsCV의 초기 fold가 지나치게 작을 때 데이터 효율 측면에서 장점이 있습니다. :chatgpt-content-reference{index="47"}

조건은 모델이 충분히 시계열 구조를 설명했는지 확인하는 것입니다.

예를 들어 Ljung–Box test의 기본 귀무가설은

$$
H_0:
\rho_1=\rho_2=\cdots=\rho_m=0
$$

입니다.

여기서 $\rho_k$는 residual의 lag- $k$ autocorrelation입니다.

$H_0$가 기각된다면 잔차에 아직 예측 가능한 시간구조가 남아 있다는 뜻이므로,

$$
\hat e_t
\not\!\perp
\hat e_{t-k}
$$

일 가능성이 있습니다.

논문은 이런 경우 random CV가 실제 generalization error를 과소평가할 수 있다고 경고합니다. :chatgpt-content-reference{index="48"}

**용어 — Autocorrelation**  
현재 오차가 과거 오차와 상관되는 정도입니다. 잔차에 자기상관이 남아 있다는 것은 모델이 아직 설명하지 못한 시간 패턴이 남았음을 의미할 수 있습니다.

---

# 2-1-8. Data leakage

forecasting에서 leakage가 특히 위험한 이유는 시간 방향 때문입니다.

정상적인 변환은

```math
\theta_{\text{prep}}
=
g(D_{\text{train}})
```

이고

```math
X_{\text{train}}'
=
f(X_{\text{train}};\theta_{\text{prep}})
```

```math
X_{\text{test}}'
=
f(X_{\text{test}};\theta_{\text{prep}})
```

이어야 합니다.

하지만 전체 데이터로 normalization parameter를 계산하면

```math
\theta_{\text{prep}}
=
g(D_{\text{train}}\cup D_{\text{test}})
```

가 되어 미래 정보가 학습 시점으로 역류합니다.

논문은 smoothing, decomposition, normalization, tsfeatures/catch22 extraction까지 train 영역만 이용해야 한다고 강조합니다. :chatgpt-content-reference{index="49"}

### EMD leakage 실험

Table 7:

| 방법 | RMSE | naïve 대비 검정 p-value |
|---|---:|---:|
| Naïve | 3.46 | — |
| **Leakage model** | **3.12** | 0.067 |
| No-leakage model | 5.65 | $1.85\times10^{-6}$ |

:chatgpt-content-reference{index="50"}

즉 미래를 보게 하면

$$
5.65
\rightarrow
3.12
$$

로 약 44.8% RMSE 감소가 나타납니다.

하지만 이것은 **모델이 더 잘 일반화한 것이 아닙니다. 미래를 미리 사용한 결과입니다.**

이 실험은 이 논문의 가장 강력한 메시지 중 하나입니다.

---

# 2-1-9. 통계적 유의성 검정

두 forecast를 비교할 때 논문은 Diebold–Mariano, Wilcoxon 계열, Giacomini–White를 논의하고, 여러 모델에서는 Friedman test 및 post-hoc test를 설명합니다. :chatgpt-content-reference{index="51"}

여러 모델 $k=1,\ldots,K$에 대해 dataset별 순위를 $r_{ik}$라 하면 평균 rank는

```math
\bar r_k
=
\frac{1}{N}
\sum_{i=1}^{N}r_{ik}
```

로 생각할 수 있습니다.

Friedman test에서

$$
H_0:
\text{all methods have equivalent performance ranks}
$$

를 검정하고, 기각되면 Nemenyi, Holm, Hochberg 등 post-hoc procedure로 어떤 모델들이 실제로 다른지를 살핍니다. Fig.13의 CD diagram은 유의하게 다르지 않은 모델들을 같은 선으로 연결합니다. :chatgpt-content-reference{index="52"}

**용어 — Post-hoc test**  
전체적으로 모델 차이가 존재함을 확인한 뒤 “A와 B 중 어느 쌍이 실제로 다른가?”를 추가 분석하는 다중비교 검정입니다.

**용어 — Critical Distance(CD)**  
두 모델의 평균 rank 차이가 통계적으로 구분되려면 넘어야 하는 기준 거리입니다.

---

# 2-1-10. 성능 향상은 무엇을 의미하는가?

이 논문은 **새 모델을 개발해서 R²를 몇 % 개선했다는 논문이 아닙니다.**

대신 실제 “성능 향상”의 의미를 다음처럼 바꿉니다.

$$
\boxed{
\text{높은 validation score}
}
\quad\not\Rightarrow\quad
\boxed{
\text{높은 real-world generalization}
}
$$

그리고

```math
\boxed{
\text{reliable generalization estimate}
}
=
\text{proper split}
+
\text{proper baseline}
+
\text{no leakage}
+
\text{proper metric}
+
\text{statistical verification}
```

라는 연구 철학을 제시합니다.

이 두 번째 식은 논문의 핵심 내용을 제가 구조화한 식입니다.

---

# 3. 주장별 Page / Figure / Table 위치 정리

| 주제 | 논문 위치 |
|---|---|
| Strong stationarity 정의 | p.793, Eq. (1), Fig.2 주변 :chatgpt-content-reference{index="53"} |
| Random walk / naïve forecast | p.795, Eq. (2)–(3) :chatgpt-content-reference{index="54"} |
| Naïve vs RF/SVM/NN | p.796, Fig.5, Table 1 :chatgpt-content-reference{index="55"} |
| Autoformer/FiLM reproducibility | p.797–798, Tables 2–3 :chatgpt-content-reference{index="56"} |
| Informer vs DHR-ARIMA | p.800, Table 4 :chatgpt-content-reference{index="57"} |
| Dataset 부족 문제 | p.801, §3.2 :chatgpt-content-reference{index="58"} |
| R²/MAPE 등 metric 문제 | p.801–802, §3.3 :chatgpt-content-reference{index="59"} |
| Forecast plot 착시 | p.802–804, Fig.6–7, Tables 5–6 :chatgpt-content-reference{index="60"} |
| Leakage | p.804–806, §3.5, Fig.8, Table 7 :chatgpt-content-reference{index="61"} |
| Rolling / Expanding window | p.807–808, Fig.9 :chatgpt-content-reference{index="62"} |
| Randomized k-fold 조건 | p.809–810, Fig.10 :chatgpt-content-reference{index="63"} |
| Non-stationarity와 validation | p.810–811, §4.1.4 :chatgpt-content-reference{index="64"} |
| Data-partition 선택 flow | p.811–812, Fig.11 :chatgpt-content-reference{index="65"} |
| Error definitions | p.812–820, Eq.4–14, Table 8 :chatgpt-content-reference{index="66"} :chatgpt-content-reference{index="67"} |
| Error-measure checklist | p.822–823, Table 9 / Fig.12 :chatgpt-content-reference{index="68"} |
| Statistical comparison | p.821–826, Fig.13–14 :chatgpt-content-reference{index="69"} |
| 최종 권고·향후 연구 | p.826–828, §5 :chatgpt-content-reference{index="70"} |

---

# 4. 연구 주제·방법·결과: 저자 보고와 제 해석 분리

| 항목 | 저자가 직접 보고한 내용 | 제 해석 |
|---|---|---|
| **연구 주제** | ML 연구자가 forecasting evaluation에서 반복하는 오류를 정리하고 best practices를 제공 | 모델 architecture 연구라기보다 **model-selection validity와 generalization estimation 연구**로 보는 것이 정확함 |
| **핵심 위험** | non-stationarity, non-normality, 잘못된 benchmark, 부적절 metric, leakage, 부족한 dataset | 실제 SOTA 경쟁에서 “architecture gain”보다 “evaluation-protocol gain”이 더 큰 경우가 있음을 경고 |
| **baseline** | naïve/seasonal-naïve 같은 단순 benchmark를 반드시 비교 | 새로운 모델은 단순 baseline을 통계적으로 의미 있게 이긴 뒤에야 복잡성의 가치가 있음 |
| **validation** | 데이터가 충분하면 tsCV 권장; 일부 pure AR에서는 k-fold도 가능 | “시계열이므로 shuffle은 항상 금지”라는 단순 규칙보다 **DGP와 residual dependency를 먼저 이해해야 한다**는 주장 |
| **metric** | 모든 시계열 특성에 안전한 하나의 metric은 없음 | metric 자체도 모델 선택 함수의 일부이므로, metric을 바꾸면 “best model”이 바뀔 수 있음 |
| **Autoformer** | 환율 데이터에서 naïve가 Autoformer보다 모든 horizon에서 우수 | sophisticated model의 낮은 error가 반드시 predictive structure 학습의 증거는 아님 |
| **FiLM** | 원 논문의 환율 결과가 rerun에서 재현되지 않음 | seed sensitivity와 benchmark instability 자체가 연구결론의 robustness 문제 |
| **Informer** | seasonal benchmark인 DHR-ARIMA가 Informer보다 우수 | 약한 ARIMA만 baseline으로 선택하는 것은 deep model에 구조적으로 유리한 비교가 될 수 있음 |
| **Leakage** | 전체-series EMD가 실제보다 현저히 좋은 결과를 만들 수 있음 | preprocessing을 pipeline 밖에서 수행하면 아무리 test target을 직접 학습하지 않아도 **간접적 target leakage**가 발생 가능 |
| **후속 연구** | MAE와 RMSE의 장점을 결합하는 Huber-like combination metric 제안 | 이후에는 metric 혼합뿐 아니라 **distribution shift와 benchmark contamination을 동시에 평가하는 framework**로 발전할 필요가 있음 |

저자 최종 권고는 충분한 데이터셋, 단순하지만 적절한 benchmark, forecast plot 의존 금지, leakage 회피, tsCV의 적절한 사용, 데이터 특성별 metric 선택, 적절한 통계검정을 포함합니다. :chatgpt-content-reference{index="71"}

---

# 5. 통계적으로 취약한 부분과 비교 불가능한 수치

이 부분은 **저자 주장과 별도로 제가 비판적으로 검토한 내용**입니다.

## 5-1. Table 7의 $p=0.067$

논문은 유의수준 $0.05$를 언급하면서 leakage model의

$$
p=0.067
$$

을 “nearly significant”로 표현합니다. :chatgpt-content-reference{index="72"}

그러나 엄밀히

$$
0.067>0.05
$$

이므로 5% 유의수준에서는 귀무가설을 기각할 수 없습니다.

따라서 말할 수 있는 것은

> leakage model의 **표본 RMSE 3.12가 naïve 3.46보다 작았다**

까지입니다.

“통계적으로 naïve보다 우수하다”는 결론은 이 $p$값만으로는 뒷받침되지 않습니다.

---

## 5-2. FiLM 재현 실패 = 즉시 “원 결과가 우연”이라고 확정할 수는 없음

Table 2에서는 5개의 trial이 제시되고 각 trial 안에서 5 seed 평균을 사용합니다. 모두 naïve보다 좋지 않았습니다. :chatgpt-content-reference{index="73"}

이는 **재현성에 심각한 의문을 제기하는 강한 증거**입니다.

그러나 다음은 없습니다.

- 원 결과와 rerun 결과의 confidence interval
- 효과크기에 대한 검정
- seed distribution 전체 분석
- GPU/라이브러리/version 차이 분해
- hyperparameter implementation discrepancy 분석

따라서

$$
\text{failed reproduction}
\Rightarrow
\text{original result definitely random}
$$

이라고 완전히 동일시하는 것은 다소 강합니다.

더 정확한 표현은

> “공개 코드 기준으로 결과가 안정적으로 재현되지 않아 원래 우월성 주장의 robustness가 약하다.”

입니다.

---

## 5-3. 논문 p.825의 정규분포/RMSE 설명은 수학적으로 주의가 필요함

논문은 충분히 큰 표본에서 MSE/MAE의 평균 분포에 CLT를 적용할 수 있다는 문맥에서 RMSE에 대해서는 “정규분포 변수의 제곱근이 chi-square distribution을 따른다”는 취지의 설명을 사용합니다. :chatgpt-content-reference{index="74"}

이 문장은 수학적으로 일반적으로 정확하지 않습니다.

카이제곱 변수는 대표적으로

```math
Q
=
\sum_{i=1}^{k}Z_i^2,
\qquad
Z_i\sim N(0,1)
```

에서 발생합니다.

즉 **정규확률변수의 단순한 제곱근이 카이제곱분포가 되는 것은 아닙니다.**

또

$$
n\ge30
$$

도 CLT의 보편적 충분조건이 아니라 경험적 heuristic에 가깝습니다. heavy tail이나 강한 dependence가 존재하면 필요한 sample size는 훨씬 클 수 있습니다.

이 부분은 논문의 핵심 결론을 무너뜨리지는 않지만, 통계검정 설명 중 하나는 수정해서 읽는 편이 안전합니다.

---

## 5-4. Wilcoxon 명칭도 구분해야 함

통계검정 section에서는 **Wilcoxon rank-sum / Mann–Whitney**를 설명하지만, leakage 실험에서는 **Wilcoxon signed-rank**라고 기록합니다. :chatgpt-content-reference{index="75"} :chatgpt-content-reference{index="76"}

두 검정은 다릅니다.

- rank-sum / Mann–Whitney: 독립 두 표본
- signed-rank: 같은 forecast origin에서 나온 **paired sample**

forecast comparison에서는 대개 동일 시점에 대한 두 모델 오차가 짝을 이루므로 paired test가 논리적으로 자연스럽습니다. 논문을 구현할 때 둘을 혼용해서는 안 됩니다.

---

## 5-5. “통계적으로 유의”와 “실무적으로 중요한 차이”는 다름

저자도 데이터셋이 아주 많으면 CD가 작아져 아주 작은 차이도 통계적으로 유의해질 수 있음을 지적합니다. :chatgpt-content-reference{index="77"}

따라서 연구에서는

$$
p\text{-value}
$$

뿐 아니라

$$
\text{effect size},
\quad
\text{relative improvement},
\quad
\text{confidence interval}
$$

도 보고해야 합니다.

예를 들어 0.01% RMSE 향상이 $p<10^{-6}$라고 해도 모델이 100배 비싸다면 산업적 가치는 거의 없을 수 있습니다.

---

# 5-6. 서로 직접 비교하면 안 되는 수치

| 수치 | 직접 비교 가능 여부 | 이유 |
|---|---|---|
| Table 1 random-walk RMSE 0.96 vs Table 3 환율 MSE 0.817 | **불가능** | dataset, 단위, metric, horizon이 다름 |
| MSE 0.817 vs MAE 0.694 | **크기 자체 비교 불가능** | MSE는 제곱단위, MAE는 원단위 |
| horizon 96 MSE vs horizon 720 MSE | 제한적 | 예측 난이도와 평가 sample 구조가 달라짐 |
| Autoformer original vs rerun | 비교 가능하지만 주의 | 동일 benchmark를 재현하려 하지만 seed/version 효과 존재 |
| Informer vs DHR-ARIMA Table 4 | 비교적 타당 | 같은 dataset/horizon이지만 서로 다른 inductive bias와 구현 방식 |
| TFB vs GIFT-Eval leaderboard score | 그대로 비교 불가능 | dataset/split/horizon/metric/model version이 다름 |
| GIFT-Eval dataset count 23 vs 28 | **version pinning 필요** | arXiv abstract는 23 datasets, Salesforce의 후속 소개는 28 datasets로 표시 :chatgpt-content-reference{index="78"} |

마지막 사례는 현대 benchmark 연구에서 **버전 관리 자체가 평가 재현성의 일부**가 되었다는 좋은 예입니다.

---

# 6. 이 논문이 답하지 않는 질문

1. “충분히 많은 dataset”은 정확히 몇 개인가? 10개, 100개, 1000개 중 어디부터 우월성 주장이 신뢰할 만한지 정량 기준을 제공하지 않습니다.

2. 미래에 한 번도 나타나지 않은 structural break에 대해 실제 generalization error를 어떻게 추정해야 하는가? 논문은 심지어 tsCV도 이를 과소평가할 수 있음을 인정하지만 완전한 해결책은 제시하지 않습니다. :chatgpt-content-reference{index="79"}

3. concept drift가 존재할 때 expanding window와 rolling window의 최적 길이 $W$를 어떻게 선택해야 하는가?

4. retraining interval을 어떻게 결정해야 하는가? 예를 들어 매 1 lot, 1 day, 100 sample 중 무엇이 최적인지에 대한 공식은 없습니다.

5. 여러 metric이 서로 다른 모델을 선택할 때 어떤 decision rule을 사용해야 하는가?

6. statistical significance와 business utility를 어떻게 하나의 criterion으로 결합해야 하는가?

7. probabilistic forecasting에서 CRPS, pinball loss, calibration, interval coverage를 어떻게 통합 평가해야 하는가? 논문은 point forecast가 주된 범위입니다. :chatgpt-content-reference{index="80"}

8. multivariate global model에서 한 series의 미래 정보가 다른 series로 새는 **cross-series leakage**를 자동으로 탐지하는 일반 알고리즘은 무엇인가?

9. Foundation model처럼 pretraining corpus가 수십억 시점일 때 test dataset이 pretraining에 포함되었는지를 어떻게 검증할 것인가?

10. 계산량·GPU 비용·latency를 포함했을 때 “통계적으로 좋은 모델”과 “실제 운영에서 좋은 모델”을 어떻게 구분할 것인가?

이 중 8–9번은 2024년 이후 foundation-model 연구에서 특히 중요한 문제로 커졌으며, 2025–2026년에는 benchmark contamination 자체를 별도의 연구 문제로 다루기 시작했습니다. :chatgpt-content-reference{index="81"}

---

# 7. 가장 중요한 그림 5개 해석

## 7-1. Figure 2 — Fixed origin vs Rolling origin, p.793

Fig.2에서 왼쪽은 하나의 forecast origin만 사용하지만 오른쪽은 origin이 반복적으로 앞으로 이동합니다. :chatgpt-content-reference{index="82"}

핵심은

```math
\hat R_{\text{fixed}}
=
L(y_{T+1:T+h},\hat y_{T+1:T+h})
```

가 사실상 특정 미래구간 하나에서 계산되는 반면,

```math
\hat R_{\text{rolling}}
=
\frac{1}{K}
\sum_{k=1}^{K}
L_k
```

는 여러 미래 시점에서 모델을 검증한다는 것입니다.

따라서 rolling origin은 **시간에 따라 성능이 어떻게 변하는가**를 측정할 수 있다는 점에서 generalization 평가에 더 유리합니다.

---

## 7-2. Figure 5 — Random walk에서 복잡한 모델과 naïve, p.796

Figure 5는 이 논문의 철학을 가장 직관적으로 보여 줍니다.

그래프만 보면 RF, NN, SVM도 실제값을 꽤 잘 따라갑니다. 하지만 Table 1에서는

```math
\text{RMSE}_{\text{naïve}}
=
0.96
```

로 가장 작습니다. :chatgpt-content-reference{index="83"}

즉 **그래프가 그럴듯하다는 사실은 predictive information을 발견했다는 증거가 아닙니다.**

이 그림이 주는 연구 원칙은:

$$
\text{New model} > \text{Simple benchmark}
$$

가 검증되기 전에는 architecture의 복잡성을 성능 개선으로 해석해서는 안 된다는 것입니다.

---

## 7-3. Figure 8 — Leakage가 성능을 “만드는” 방법, p.806

Figure 8은 leakage model과 no-leakage model의 prediction trajectory를 비교합니다. :chatgpt-content-reference{index="84"}

Table 7에서

```math
\text{RMSE}_{\text{leak}}
=3.12
<
3.46
=
\text{RMSE}_{\text{naïve}}
```

이므로 얼핏 보면 복잡한 EMD+RF+ARIMA model이 theoretical naïve benchmark를 이긴 것처럼 보입니다.

그러나 decomposition에 미래가 포함되어 있었습니다.

이 그림은 실제 연구에서 가장 위험한 오류를 표현합니다.

$$
\text{future information}
\rightarrow
\text{feature/preprocessing}
\rightarrow
\text{model}
$$

의 경로가 존재한다면 test target을 feature에 직접 넣지 않았더라도 leakage입니다.

---

## 7-4. Figure 11 — Data partitioning decision flow, p.812

Fig.11은 논문의 “모델 구조”에 가장 가까운 그림입니다. :chatgpt-content-reference{index="85"}

논리는 대략 다음과 같습니다.

$$
\text{series length}
\rightarrow
\text{model temporal dependence}
\rightarrow
\text{stationarity}
\rightarrow
\text{residual autocorrelation}
\rightarrow
\text{CV choice}
$$

즉 “시계열이면 항상 A를 사용한다”가 아니라,

> 데이터와 모델의 통계적 구조가 validation strategy를 결정해야 한다

는 것입니다.

연구자의 관점에서는 이 그림이 가장 실용적입니다.

---

## 7-5. Figure 14 — Statistical-test selection, p.826

두 방법인가?

$$
K=2
$$

인지,

여러 방법인가?

$$
K > 2
$$

인지부터 구분하고, parametric assumption의 적합 여부와 comparison 목적에 따라 검정을 결정합니다. :chatgpt-content-reference{index="86"}

여러 모델을 비교할 때 단순히

$$
\min_k\text{RMSE}_k
$$

인 모델 하나를 골라 “best”라고 하는 대신

$$
H_0:
\text{model performances are statistically indistinguishable}
$$

를 먼저 검증하는 구조입니다.

즉 0.601과 0.600이라는 두 점수의 차이가 실제 의미 있는 차이인지 판단하기 위한 단계입니다.

---

# 8. 결론 — 저자들의 시사점과 후속 연구

## 저자가 직접 제시한 결론

저자들의 최종 메시지는 명확합니다.

- 충분히 많은 dataset으로 평가할 것
- naïve / seasonal-naïve 같은 **단순하지만 강한 baseline**을 반드시 포함할 것
- forecast plot의 시각적 인상만으로 결론 내리지 말 것
- preprocessing 및 rolling-origin 과정의 leakage를 방지할 것
- 데이터가 충분하다면 tsCV를 선호할 것
- pure AR + 충분히 모델링된 residual 조건에서는 짧은 series에 k-fold도 고려할 것
- 모든 데이터에 통하는 하나의 universal metric이 없음을 인정할 것
- 모델 성능 차이는 통계적 검정까지 수행할 것

입니다. :chatgpt-content-reference{index="87"}

저자들이 명시적으로 제안한 한 가지 향후 연구 방향은 Huber loss가 MAE와 squared loss의 장점을 조합하듯 **여러 evaluation metric의 장점을 결합하는 hybrid evaluation measure**입니다. :chatgpt-content-reference{index="88"}

---

# 8-1. 모델의 일반화 성능 향상 가능성

여기서는 **저자 내용과 제 연구 제안을 구분**하는 것이 중요합니다.

## A. 논문에서 직접 얻을 수 있는 일반화 원칙

### ① 모델보다 먼저 validation bias를 줄여야 함

우리가 원하는 것은 실제 risk

```math
R_{\text{future}}(f)
=
E_{(X,Y)\sim P_{\text{future}}}
\left[
L(Y,f(X))
\right]
```

입니다.

하지만 실제로 관측하는 것은

$$
\hat R_{\text{validation}}(f)
$$

입니다.

좋은 evaluation protocol의 핵심 목적은

$$
\hat R_{\text{validation}}(f)
\approx
R_{\text{future}}(f)
$$

가 되도록 하는 것입니다.

non-stationarity가 강하면 이 근사가 깨질 수 있습니다.

---

### ② 잔차의 predictability를 먼저 제거

잔차에

$$
\text{Corr}(e_t,e_{t-k})\neq0
$$

가 남아 있다면 모델이 사용 가능한 정보를 모두 학습하지 못했을 수 있습니다.

따라서 단순 validation-score tuning보다

$$
\text{point accuracy}
+
\text{residual whiteness}
$$

를 동시에 확인하는 것이 미래 일반화에 더 직접적입니다.

---

### ③ 전처리는 fold 내부에서만 학습

예를 들어 standardization:

```math
\mu_{\text{train}}
=
\frac1{N_{\text{train}}}
\sum_{i\in\text{train}}x_i
```

```math
\sigma_{\text{train}}
=
\sqrt{
\frac1{N_{\text{train}}}
\sum_{i\in\text{train}}
(x_i-\mu_{\text{train}})^2
}
```

```math
z_i
=
\frac{x_i-\mu_{\text{train}}}
{\sigma_{\text{train}}}
```

여야 합니다.

Validation/Test의 평균이나 분산을 scale estimation에 섞어서는 안 됩니다.

---

## B. 논문을 발전시킨 제안: “Generalization-first forecasting”

현대 연구에서는 단순 평균 Test R² 하나보다 다음을 함께 평가하는 것이 더 타당합니다.

### 1. Temporal OOD split

$$
D_{\text{train}} < D_{\text{validation}} < D_{\text{test}}
$$

를 시간 순으로 두고,

test에는 가능하면 이전보다 다른 regime이 들어가도록 합니다.

**용어 — OOD(Out-of-Distribution)**  
학습 때 경험한 분포와 다른 조건에서 들어오는 새로운 데이터입니다.

---

### 2. Regime별 성능

전체 평균만 보지 말고

```math
R_k
=
E[
L(Y,\hat Y)
\mid
G=k
]
```

를 계산합니다.

$G$는 예를 들어

- 계절
- 장비
- chamber
- 생산 recipe
- volatility regime
- 온도 regime
- 시간 구간

등이 될 수 있습니다.

그리고

```math
R_{\text{worst}}
=
\max_k R_k
```

를 같이 봅니다.

평균은 좋아졌지만 특정 regime에서 붕괴하는 모델을 걸러낼 수 있습니다.

---

### 3. Drift-aware window 선택

전체 과거를 동일하게 학습하는 대신

```math
D_t^{(W)}
=
\{t-W+1,\ldots,t\}
```

에서 window $W$를 validation으로 결정할 수 있습니다.

너무 큰 $W$:

$$
\text{low variance}
+
\text{high stale-data bias}
$$

너무 작은 $W$:

$$
\text{low drift bias}
+
\text{high estimation variance}
$$

의 trade-off가 있습니다.

이것은 특히 제조공정이나 장비 drift 데이터에서 중요합니다.

---

### 4. 단일 metric이 아니라 robustness envelope

예:

```math
\mathcal M
=
\{
\text{RMSE},
\text{MAE},
\text{MASE},
\text{Bias},
\text{Worst-regime error}
\}
```

그리고 모델 $f$가 여러 metric에서 일관되게 개선되는지 확인합니다.

이는 논문이 언급한 multi-measure sanity checking 및 combination-metric 방향과 잘 맞습니다. :chatgpt-content-reference{index="89"}

---

# 8-2. 2020년 이후 최신 관련 연구 비교

중요하게, **아래 연구들이 이 논문을 직접 인용해 영향을 받았는지에 대한 citation-network 분석까지 이번 검색에서 확인한 것은 아닙니다.** 따라서 “영향”은 직접 인용관계가 아니라 **평가 방법론의 발전 방향과 내용적 연결성**을 의미합니다.

| 연구 | 핵심 진전 | 본 논문과의 관계 | 일반화 관점 |
|---|---|---|---|
| **Cerqueira et al., 2020 — Evaluating time series forecasting models** | CV/OOS/prequential performance estimation을 실증 비교. 174 real-world series 사례에서 non-stationary 상황에는 temporal OOS가 강함 | Hewamalage 논문의 직접 선행 연구 중 하나 | “검증 score가 미래 error를 얼마나 잘 추정하는가”를 실증적으로 다룸 :chatgpt-content-reference{index="90"} |
| **Petropoulos et al., 2022 — Forecasting: theory and practice** | forecasting 전반의 대규모 theory/practice review | 현재 논문보다 범위가 넓고 probabilistic/실무 관점까지 연결 | 모델 정확도보다 의사결정·불확실성까지 평가 범위를 확장 :chatgpt-content-reference{index="91"} |
| **Hewamalage et al., 2023 — 본 논문** | benchmark, leakage, metric, split, significance를 통합한 tutorial framework | 기준점 | ML 연구의 잘못된 평가 관행 자체를 문제화 :chatgpt-content-reference{index="92"} |
| **Zeng et al., 2023 — Are Transformers Effective for Time Series Forecasting?** | 단순 one-layer LTSF-Linear가 9개 real-world dataset에서 Transformer 계열을 능가했다고 보고 | “복잡한 모델을 약한 benchmark와 비교하지 말라”는 본 논문의 주장과 매우 강하게 정합 | architecture보다 올바른 direct multi-step baseline이 중요함 :chatgpt-content-reference{index="93"} |
| **TFB, Qiu et al., 2024** | 10 domains, 8,068 univariate series, 25 multivariate datasets; 통계/ML/DL을 동일 pipeline에서 비교 | 본 논문의 best-practice 철학을 대규모 자동 benchmark로 operationalize | dataset coverage와 pipeline 표준화로 benchmark overfitting 감소 :chatgpt-content-reference{index="94"} |
| **ProbTS, 2024** | point + distributional forecast, short/long horizon을 unified benchmark에서 비교 | 본 논문의 point forecast 중심 한계를 확장 | 한 horizon/한 output type에서 강한 모델이 universal하게 강하다는 착각을 방지 :chatgpt-content-reference{index="95"} |
| **Chronos, 2024** | T5 기반 tokenized forecasting; 20M–710M parameter, 42 datasets, synthetic GP data로 generalization 강화 | 평가 대상이 dataset-specific model에서 pretrained FM으로 변화 | unseen dataset zero-shot generalization이라는 새 평가축을 도입 :chatgpt-content-reference{index="96"} |
| **TimesFM, 2024** | decoder-only patched model, 대규모 real/synthetic corpus, unseen domain/horizon/granularity zero-shot 평가 | 일반화 범위가 “미래 구간”에서 “새로운 domain”까지 확대 | zero-shot 성능을 task-specific model과 비교 :chatgpt-content-reference{index="97"} |
| **Moirai, 2024** | LOTSA 27B observations, 9 domains; cross-frequency/arbitrary variates를 처리 | 다변량·다주기 general forecasting으로 범위 확대 | zero-shot 모델이 full-shot model과 경쟁 가능하다고 보고 :chatgpt-content-reference{index="98"} |
| **GIFT-Eval, 2024/25** | zero-shot FM용 diverse benchmark + **non-leaking pretraining dataset** | 본 논문의 leakage 원칙이 foundation-model pretraining 문제로 직접 확대된 형태 | “test가 pretraining corpus에 없었는가?”가 새 generalization 조건 :chatgpt-content-reference{index="99"} |
| **Meyer et al., 2025 — Time Series Foundation Models: Benchmarking Challenges and Requirements** | overlap, obscure pretraining datasets, external-shock memorization, spatiotemporal benchmark 문제 지적 | Hewamalage의 leakage/nonstationarity 문제를 foundation model 수준으로 확장 | truly OOS **future data** 평가 필요성을 명시적으로 주장 :chatgpt-content-reference{index="100"} |
| **TSFMAudit, 2026** | 6 TSFM, 187 datasets, 10 baselines에서 pretraining contamination auditing 제안 | “누수가 있었는지 조심하라”에서 “누수를 탐지하는 알고리즘”으로 발전 | adaptation dynamics를 이용해 contaminated benchmark를 식별하려 함; 현재 arXiv preprint :chatgpt-content-reference{index="101"} |

---

## 최신 연구에서 보이는 중요한 변화

### 2020–2023

질문은 주로

$$
\text{“어떤 validation split이 미래 error를 잘 추정하는가?”}
$$

였습니다.

---

### 2023–2024

질문이

$$
\text{“Transformer의 성능 향상이 진짜인가,
아니면 weak baseline 때문인가?”}
$$

로 이동합니다.

Zeng et al.의 LTSF-Linear 결과가 대표적입니다. 9개 실세계 데이터에서 단순 linear 모델이 Transformer 계열을 강하게 이겼다는 결과는 Hewamalage et al.의 핵심 문제의식을 매우 직접적으로 뒷받침합니다. :chatgpt-content-reference{index="102"}

---

### 2024 이후

foundation model 등장으로 질문이 더 어려워집니다.

$$
\text{“이 모델이 test dataset을 정말 처음 보는가?”}
$$

입니다.

**용어 — Foundation model**  
다수의 데이터셋에 대규모 사전학습한 후 새로운 데이터셋에서도 별도 학습 없이 또는 적은 fine-tuning으로 사용하는 범용 모델입니다.

**용어 — Zero-shot forecasting**  
해당 target dataset으로 parameter를 학습하지 않고 pretrained model을 그대로 적용해 예측하는 설정입니다.

---

## GIFT-Eval이 특히 중요한 이유

GIFT-Eval은 단순히 test set을 나누는 수준을 넘어 **non-leaking pretraining dataset**을 별도로 제공합니다. GitHub 설명에서도 statistical / DL / foundation models, univariate/multivariate, short/long horizon, probabilistic forecasting을 함께 다루는 구조를 명시합니다. :chatgpt-content-reference{index="103"}

이는 Hewamalage 논문의

$$
D_{\text{test}}
\not\rightarrow
\text{training information}
$$

원칙을 foundation model에 맞게

```math
D_{\text{evaluation}}
\cap
D_{\text{pretraining}}
=
\varnothing
```

에 가깝게 확장한 것입니다.

실제로 arXiv 초기 버전은 23 datasets를, Salesforce의 이후 소개 자료는 28 datasets를 기술하므로 benchmark 결과를 재현할 때는 **논문/데이터/leaderboard 버전을 함께 고정**해야 합니다. :chatgpt-content-reference{index="104"}

---

# 앞으로 연구 시 반드시 고려해야 할 점

## 1. Validation generalization과 model generalization을 분리

다음 두 질문은 다릅니다.

$$
Q_1:
\quad
\hat R_{\text{valid}}
\text{가 }
R_{\text{test}}
\text{를 잘 추정하는가?}
$$

$$
Q_2:
\quad
f_{\theta}
\text{가 distribution shift에서도 잘 작동하는가?}
$$

좋은 tsCV가 $Q_1$은 개선하지만 구조적 break에 강한 모델을 자동으로 만들어주는 것은 아닙니다.

---

## 2. 최소 세 종류의 일반화를 따로 평가

### Temporal generalization

$$
P_t
\rightarrow
P_{t+\Delta}
$$

같은 domain에서 미래 시간으로 이동합니다.

### Cross-regime generalization

$$
P_{\text{regime A}}
\rightarrow
P_{\text{regime B}}
$$

예: 장비/계절/공정 조건이 달라집니다.

### Cross-domain generalization

$$
P_{\text{domain A}}
\rightarrow
P_{\text{domain B}}
$$

foundation model의 zero-shot 능력에 해당합니다.

이 세 종류를 하나의 “Test RMSE”로 합치면 어떤 일반화에 성공했는지 알 수 없습니다.

---

## 3. Benchmark contamination을 독립 검증 항목으로 만들 것

2025–2026 연구가 보여주듯 foundation model에서는

$$
D_{\text{test}}
\subseteq
D_{\text{pretrain}}
$$

가능성을 무시할 수 없습니다. :chatgpt-content-reference{index="105"}

따라서 앞으로는 논문에 최소한

$$
\text{Pretraining corpus provenance}
$$

$$
\text{training cutoff date}
$$

$$
\text{benchmark release date}
$$

$$
\text{overlap audit}
$$

를 함께 보고하는 것이 바람직합니다.

---

## 4. Average score만으로 SOTA를 선언하지 말 것

추천하는 보고 형태는

$$
\text{Mean performance}
+
\text{Median performance}
+
\text{Worst-regime performance}
+
\text{CI}
+
\text{baseline-relative score}
$$

입니다.

예를 들어

```math
\Delta_{\text{baseline}}
=
\frac{
E_{\text{baseline}}
-
E_{\text{model}}
}{
E_{\text{baseline}}
}
```

를 보고하면 “RMSE=3.1”이라는 절대 수치보다 모델의 추가 예측가치를 알기 쉽습니다.

---

# 이 논문이 앞으로 연구에 미치는 의미

이 논문의 가장 중요한 영향은 특정 metric이나 CV 방법 하나를 추천한 데 있지 않습니다.

보다 근본적으로

$$
\boxed{
\text{model innovation}
\neq
\text{scientific progress}
}
$$

이며,

```math
\boxed{
\text{model innovation}
+
\text{valid evaluation}
=
\text{credible progress}
}
```

라는 기준을 강조했다는 데 있습니다.

2023년의 simple-linear-vs-Transformer 논쟁, 2024년의 TFB/ProbTS/GIFT-Eval, 그리고 2025–2026년 foundation-model contamination 연구는 모두 이 방향을 더욱 강화하고 있습니다. :chatgpt-content-reference{index="106"}

특히 **모델의 일반화 성능을 실제로 향상시키려는 연구자**에게 이 논문이 주는 가장 중요한 원칙은 다음입니다.

$$
\boxed{
\text{Train 성능 향상} < \text{Validation 안정성} < \text{Temporal OOS 재현성} < \text{Distribution-shift robustness}
}
$$

이 식은 논문에 직접 제시된 것이 아니라 연구 방향을 제가 요약한 것입니다. 마지막 단계로 갈수록 “현재 데이터를 얼마나 잘 맞추는가”보다 “새로운 미래 환경에서도 성능이 유지되는가”가 더 중요합니다.

---

# 참고한 원문 및 웹 자료

**1. Hewamalage, Ackermann & Bergmeir — “Forecast evaluation for data scientists: common pitfalls and best practices.”**  
*Data Mining and Knowledge Discovery*, 37, 788–832, Springer Nature. 첨부 PDF를 1차 원문으로 사용했으며 Springer 공식 페이지로 서지정보를 교차확인했습니다. :chatgpt-content-reference{index="107"}

**2. Cerqueira, Torgo & Mozetič — “Evaluating Time Series Forecasting Models: An Empirical Study on Performance Estimation Methods.”**  
*Machine Learning*, 2020. :chatgpt-content-reference{index="108"}

**3. Petropoulos et al. — “Forecasting: theory and practice.”**  
*International Journal of Forecasting*, 2022. :chatgpt-content-reference{index="109"}

**4. Zeng, Chen, Zhang & Xu — “Are Transformers Effective for Time Series Forecasting?”**  
AAAI 2023. :chatgpt-content-reference{index="110"}

**5. Qiu et al. — “TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods.”**  
*Proceedings of the VLDB Endowment*, 2024. :chatgpt-content-reference{index="111"}

**6. Zhang et al. — “ProbTS: Benchmarking Point and Distributional Forecasting across Diverse Prediction Horizons.”**  
NeurIPS 2024 Datasets and Benchmarks Track. :chatgpt-content-reference{index="112"}

**7. Ansari et al. — “Chronos: Learning the Language of Time Series.”**  
*Transactions on Machine Learning Research*, 2024. :chatgpt-content-reference{index="113"}

**8. Das et al. — “A Decoder-Only Foundation Model for Time-Series Forecasting.”**  
ICML 2024 / PMLR. :chatgpt-content-reference{index="114"}

**9. Woo et al. — “Unified Training of Universal Time Series Forecasting Transformers.”**  
Moirai, ICML 2024 / PMLR. :chatgpt-content-reference{index="115"}

**10. Aksu et al. — “GIFT-Eval: A Benchmark For General Time Series Forecasting Model Evaluation.”**  
2024 arXiv / ICLR 2025 submission 자료 및 Salesforce 공개 benchmark 자료. :chatgpt-content-reference{index="116"}

**11. Salesforce — “GIFT-Eval: A Benchmark for General Time Series Forecasting Model Evaluation.”**  
2024 공식 연구 소개. 논문 초기 버전과 최신 dataset count 차이를 확인하는 데 사용했습니다. :chatgpt-content-reference{index="117"}

**12. Meyer et al. — “Time Series Foundation Models: Benchmarking Challenges and Requirements.”**  
2025 arXiv preprint. peer-reviewed publication으로 확인한 것이 아니므로 **preprint 근거**로만 사용했습니다. :chatgpt-content-reference{index="118"}

**13. Li et al. — “TSFMAudit: Data Contamination Auditing in Forecasting Time Series Foundation Models.”**  
2026 arXiv preprint. 현재 답변에서는 **preprint 결과**로 명시적으로 구분했습니다. :chatgpt-content-reference{index="119"}

---

## 최종 평가

이 논문을 한 문장으로 압축하면 다음과 같습니다.

$$
\boxed{
\text{좋은 forecasting 연구는 높은 Test score를 만드는 연구가 아니라,
그 Test score가 미래 성능을 정직하게 나타내도록 만드는 연구이다.}
}
$$

특히 일반화 성능을 연구한다면 **시간순 분할만 지키는 것으로는 충분하지 않습니다.** preprocessing의 train-only fitting, naïve/seasonal-naïve와의 상대 성능, 잔차 자기상관, regime별 성능, horizon별 성능, statistical uncertainty, 그리고 최근 foundation model에서는 pretraining contamination까지 검증해야 비로소 “미래 데이터에서 일반화된다”는 주장을 강하게 뒷받침할 수 있습니다.

다음 단계로는 제가 이 논문의 원칙을 **실제 연구에 바로 사용할 수 있는 ‘누수 없는 시계열 모델 평가 프로토콜’로 변환하여, Train/Validation/Test 설계·Ljung–Box·baseline-relative metric·regime별 일반화 검증까지 포함한 연구용 체크리스트와 Python 실험 설계**로 만들어드릴 수 있습니다.
