# A Multi-Task Learning Approach to Linear Multivariate Forecasting

**논문:** Liran Nochumsohn, Hedi Zisling, Omri Azencot, *A Multi-Task Learning Approach to Linear Multivariate Forecasting*, AISTATS 2025, PMLR Vol. 258, pp. 2638–2646. 첨부하신 PDF는 arXiv:2502.03571v2이며, 2025년 3월 수정본입니다. 논문 초록 자체가 핵심 문제를 “최근 다변량 예측 모델들이 변수 간 관계를 충분히 사용하지 않고 각 변수를 독립적으로 처리한다”는 점으로 설정하고 있습니다.  공식 PMLR 등재 정보도 이를 확인합니다. ([Proceedings of Machine Learning Research][1])

> 아래의 **페이지는 첨부 PDF의 PDF 페이지 번호 기준**입니다.
> 또한 **[저자 주장]**과 **[내 해석/검증]**을 의도적으로 분리했습니다. 특히 논문의 gradient 해석에는 수학적으로 주의해야 할 부분이 있습니다.

---

## 1. Executive Summary — 10문장 이내

1. 이 논문은 **다변량 시계열 예측(Multivariate Time-Series Forecasting, MTSF)을 Multi-Task Learning(MTL) 문제로 재해석**하여, 각 변수를 독립적으로 예측하는 기존 channel-independent 접근이 버리는 변수 간 관계를 활용하려는 연구입니다. 
2. 저자들은 선형 예측기의 gradient를 분석하여, 한 변수의 gradient가 입력 시계열 $x_i$와 예측 오차에 의해 결정된다는 점을 MTLinear 설계의 출발점으로 삼습니다. 
3. 변수 간 절대 Pearson correlation이 높으면 optimization 과정도 유사할 것이라고 보고, **절대 상관계수 기반 hierarchical clustering**으로 변수들을 여러 task group으로 나눕니다. 
4. 각 cluster에는 독립된 Linear/DLinear/NLinear 계열 head를 할당하여, 모든 변수를 하나의 모델로 공유하는 경우와 변수별 모델을 완전히 분리하는 경우의 중간 구조를 만듭니다. 
5. 동시에 예측오차가 큰 변수나 horizon의 gradient가 학습을 지배하지 않도록, 변수별·horizon별 평균오차에 역비례하는 loss weight를 적용합니다. 
6. Table 1에서 MTNLinear의 평균 MSE/MAE는 **0.550/0.423**으로 PatchTST의 **0.584/0.431**, iTransformer의 **0.610/0.436**보다 낮으며, MTLinear 계열이 단순 선형 구조임에도 강한 결과를 보입니다. 
7. 하지만 모든 데이터셋에서 개선되는 것은 아니며, 특히 Exchange와 Traffic에서는 iTransformer/PatchTST 또는 기존 선형모델이 더 강한 경우가 분명히 존재합니다. 
8. longer-lookback 실험과 제한적인 cross-dataset linear-probing 실험은 일반화 가능성을 보여주지만, 실험 범위가 좁고 비교 모델의 표준편차가 없어 **통계적으로 일반화 우월성을 확정할 정도의 증거는 아닙니다**.  
9. 특히 수학적으로는 gradient 방향이 correlation만으로 정해진다는 설명이 정확히는 불완전하며, **residual의 부호와 입력 norm도 gradient conflict 및 magnitude를 결정**한다는 점이 향후 개선의 중요한 단서입니다.
10. 따라서 이 논문의 가장 큰 장기적 가치는 MTLinear 자체의 SOTA 수치라기보다 **“variable relationship → task grouping → task-specific sharing/balancing”이라는 설계 원리를 다변량 예측에 명시적으로 도입한 것**에 있으며, 이를 dynamic clustering, nonlinear/lagged dependency, partial sharing, foundation-model decoder로 확장하는 것이 일반화 성능 측면에서 가장 유망합니다.

---

# 1-1. 연구 목적과 필요성

## 연구가 해결하려는 근본적인 질문

다변량 시계열을

$$
X=
[x_1,x_2,\ldots,x_k]
$$

라고 하겠습니다.

여기서 $x_i$는 단순한 feature 하나가 아니라 **시간축을 가진 하나의 변수 시계열**입니다. 예를 들어 반도체 공정이라면 압력, RF power, 온도, 유량 등이 각각 $x_i$가 될 수 있습니다.

최근의 강력한 모델 중 일부는 다음처럼 생각합니다.

$$
x_i
\rightarrow
\text{동일한 predictor}
\rightarrow
\hat y_i
$$

즉, 모델 parameter는 공유해도 **각 변수의 시계열 자체는 독립적인 channel처럼 처리**합니다. 논문은 이때 생기는 질문을 제기합니다.

* 서로 매우 비슷한 두 변수는 정말 독립적인 문제로 볼 필요가 있는가?
* 전혀 다른 패턴을 가진 두 변수를 같은 parameter로 학습시키는 것이 타당한가?
* 매우 예측하기 어려운 한 변수가 큰 gradient를 발생시켜 다른 변수의 학습을 방해하지 않는가?

저자들은 이것을 **Multi-Task Learning 문제**로 바꿔 해석합니다. 

### 용어 메모 — Channel-independent

**Channel-independent**는 다변량 시계열의 각 변수를 별도의 시계열로 취급하여 시간방향 pattern은 학습하되, 다른 channel의 값을 직접 섞지 않거나 제한적으로만 이용하는 설계입니다. PatchTST가 대표적인 연구 흐름이며, 이 구조는 distribution shift에 견고할 수 있지만 변수 간 유용한 상호정보를 버릴 가능성이 있습니다. PatchTST는 patch 기반 channel-independent Transformer와 self-supervised transfer를 결합한 모델입니다. ([ML Anthology][2])

---

# 2. 핵심 주장과 근거

| 핵심 주장                                              | 논문의 근거                                                       | 위치                    | 평가                                           |                |                           |
| -------------------------------------------------- | ------------------------------------------------------------ | --------------------- | -------------------------------------------- | -------------- | ------------------------- |
| MTSF를 MTL로 볼 수 있다                                  | 변수별 loss가 동일 parameter의 gradient에 함께 기여                      | Eq. (5), (6), PDF p.3 | **이론적으로 타당한 유용한 관점**                         |                |                           |
| 상관성이 높은 변수끼리 묶으면 gradient conflict를 줄일 수 있다        | Fig. 1에서 높은 $\mid \rho \mid$의 변수쌍이 대체로 conflict가 적음                     | Fig. 1, pp.3–4 | **경험적 지지는 있으나 수학적으로 조건부** |
| 큰 prediction error가 학습을 지배한다                       | Fig. 5에서 error와 gradient magnitude가 양의 관계                    | Eq. (6), Fig. 5, p.8  | 방향은 맞지만 ** $\mid x_i \mid$도 빠져서는 안 됨**                |                |                           |
| correlation clustering + error balancing 조합이 효과적이다 | component ablation                                           | Table 2, p.6          | 대체로 지지되나 **모든 dataset에서 일관되지는 않음**           |                |                           |
| MTLinear은 강력한 Transformer들과 경쟁 가능하다                | MTNLinear avg. MSE 0.550, PatchTST 0.584, iTransformer 0.610 | Table 1, p.5          | benchmark상 강력하나 통계적 우월성은 미확정                 |                |                           |
| 기존 gradient manipulation보다 효율적이다                   | penalty 자체는 별도 task별 backward가 필요하지 않음                       | Eq. (7),(8), Table 3  | penalty 계산에는 장점. **전체 모델이 $O(1)$이라는 의미는 아님** |                |                           |
| 긴 lookback에도 성능이 좋아질 수 있다                          | 192–720 lookback 실험                                          | Fig. 3, p.6           | 유망하나 3개 dataset·특정 horizon이라 제한적             |                |                           |
| transfer에도 활용 가능하다                                 | MTLinear/MTLinear probing cross-dataset 실험                   | Table 5, p.8          | 가능성은 보이나 실험 규모가 작음                           |                |                           |
| 완전히 무관한 변수의 유용한 정보는 놓칠 수 있다                        | 저자 스스로 limitation으로 명시                                       | Conclusion pp.8–9     | **핵심 한계**                                    |                |                           |
| 향후 cluster 자체를 학습할 수 있다                            | future work                                                  | p.9                   | 일반화 개선에 가장 중요한 방향 중 하나                       |                |                           |

Ablation을 보면 저자들이 주장하는 두 모듈의 상호보완성이 상당 부분 지지됩니다. 그러나 예를 들어 MTDLinear의 Exchange에서는 penalty-only MSE가 0.278인데 full model은 0.289이며, MTNLinear Exchange에서는 baseline 0.378, penalty-only 0.366에 비해 full model이 0.410입니다. 따라서 **“두 모듈을 합치면 항상 좋아진다”는 강한 명제는 데이터가 지지하지 않습니다.** 

---

# 2-1. 문제, 방법, 수식, 구조, 성능 및 한계

## 2-1-1. MTL formulation

일반적인 $k$-task MTL 문제를 논문은 다음과 같이 정의합니다.

```math
\theta^*
=
\arg\min_{\theta}
L(\theta)
=
\arg\min_{\theta}
\frac{1}{k}
\sum_{i=1}^{k} L_i(\theta)
```

그리고 전체 gradient는

```math
\nabla L(\theta)
=
\frac{1}{k}
\sum_{i=1}^{k} g_i
=
\frac{1}{k}
\sum_{i=1}^{k}
\nabla L_i(\theta)
```

입니다. 

### 기호 설명

* $k$: task의 수. MTLinear에서는 사실상 변수 또는 변수 group과 연결됩니다.
* $i$: task index입니다.
* $\theta$: 여러 task가 공유하는 model parameter입니다.
* $L_i(\theta)$: $i$번째 task의 loss입니다.
* $g_i=\nabla L_i(\theta)$: $i$번째 task가 parameter에 요구하는 update 방향입니다.
* $\nabla$: parameter에 대한 미분, 즉 gradient입니다.

---

## Tragic Triad

MTL에서는 다음 세 문제가 함께 나타날 수 있습니다.

### ① Conflicting gradient

$$
g_i^\top g_j < 0
$$

이면 두 gradient 사이 각도가 $90^\circ$보다 큽니다.

즉 task $i$를 개선하는 방향으로 parameter를 움직였을 때 task $j$의 loss는 오히려 증가할 수 있습니다.

### ② Gradient magnitude imbalance

$$
\|g_i\|_2 \gg \|g_j\|_2
$$

이면 평균 gradient에서 $g_i$가 지나치게 큰 영향력을 갖습니다.

### ③ 높은 loss-surface curvature

parameter 공간에서 loss가 심하게 휘어 있으면 optimization이 불안정해집니다.

논문은 PCGrad 논문의 이른바 “tragic triad” 관점을 가져옵니다.  PCGrad 자체는 2020년 NeurIPS 연구로, conflict gradient를 다른 task gradient의 normal plane에 projection하는 방법입니다. ([NeurIPS Proceedings][3])

### 중요한 비판

MTLinear은 실제로 이 세 문제 중 주로

* **conflict**
* **magnitude imbalance**

두 가지를 다룹니다.

**loss curvature 자체를 직접 완화하는 장치는 없습니다.**

따라서 “tragic triad 전체를 해결한다”라고 표현하면 과도합니다.

---

# 2-1-2. 다변량 선형 forecasting formulation

입력을

$$
X=[x_1,x_2,\ldots,x_k]
\in
\mathbb{R}^{\ell\times k}
$$

라고 합니다.

표준 linear forecasting은

```math
\tilde Y^\top
=
X^\top\Theta+b
```

입니다. 

### 기호

* $\ell$: lookback length, 즉 과거 몇 시점을 입력으로 사용할지입니다.
* $k$: 변수 수입니다.
* $h$: forecast horizon입니다.
* $x_i\in\mathbb R^\ell$: $i$번째 변수의 과거 $\ell$개 값입니다.
* $X\in\mathbb R^{\ell\times k}$: 모든 변수의 입력입니다.
* $\Theta\in\mathbb R^{\ell\times h}$: 시간축 linear mapping의 weight입니다.
* $\theta_j$: $j$번째 미래 시점을 예측하는 weight vector입니다.
* $b\in\mathbb R^h$: bias입니다.
* $\tilde Y$: 예측값입니다.
* $Y$: 실제 미래값입니다.

중요한 점은 $\Theta$가 **변수축이 아니라 시간축에 적용**된다는 것입니다.

---

# 2-1-3. MTLinear 설계의 핵심이 되는 gradient

MSE는

```math
F(\Theta)
=
\frac{1}{kh}
\sum_{j=1}^{h}
\sum_{i=1}^{k}
\left(
x_i^\top\theta_j-y_{j,i}
\right)^2
```

입니다.

그리고 특정 horizon $j$에 대해

```math
\nabla_{\theta_j}F(\theta_j)
=
\frac{1}{k}
\sum_{i=1}^{k}
2x_i
\left(
x_i^\top\theta_j-y_{j,i}
\right)
```

가 됩니다. 

개별 변수 $i$의 gradient contribution을 분리하면

```math
g_{i,j}
=
2x_i r_{i,j}
```

이며,

```math
r_{i,j}
=
x_i^\top\theta_j-y_{j,i}
```

입니다.

* $r_{i,j}$: $i$번째 변수, $j$번째 horizon의 signed residual입니다.
* $g_{i,j}$: 그 observation/task가 $\theta_j$에 주는 gradient입니다.

이 식이 논문의 거의 모든 설계 논리를 만들어냅니다.

---

# 매우 중요한 수학적 검증: 논문 설명을 그대로 받아들이면 안 되는 부분

논문은 대략적으로 **gradient direction은 $x_i$가 결정하고, gradient magnitude는 prediction error가 결정한다**고 설명합니다. 

그런데 식을 그대로 전개하면 더 정확한 관계가 나옵니다.

두 변수 $a,b$의 gradient inner product는

```math
g_{a,j}^\top g_{b,j}
=
4r_{a,j}r_{b,j}
x_a^\top x_b
```

입니다.

따라서 gradient cosine은

```math
\cos(g_{a,j},g_{b,j})
=
\text{sign}(r_{a,j}r_{b,j})
\frac{x_a^\top x_b}
{\|x_a\|_2\|x_b\|_2}
```

즉,

```math
\boxed{
\cos(g_a,g_b)
=
\text{sign}(r_a r_b)
\cos(x_a,x_b)
}
```

입니다.

### 이것이 의미하는 것

gradient conflict는 **변수 correlation 하나만으로 결정되지 않습니다.**

예를 들어

$$
x_a^\top x_b > 0
$$

이어도

$$
r_a r_b < 0
$$

이면

$$
g_a^\top g_b < 0
$$

가 되어 gradient는 conflict합니다.

반대로 $x_a,x_b$가 강한 음의 correlation을 가지더라도 residual signs가 반대라면 gradient가 같은 방향으로 갈 수도 있습니다.

### 따라서

논문의

> “task gradient 사이 angle을 variable 사이 angle로 볼 수 있다”

는 설명은 엄밀하게는 **residual scalar의 부호를 무시한 단순화**입니다.

Fig. 1과 Fig. 6이 경험적으로 Pearson correlation과 conflict의 연관성을 보여주는 것은 맞지만, 그것은 일반적인 수학적 동치가 아닙니다. Fig. 6에서도 DLinear/PatchTST에는 유사성이 강하지만 Autoformer에서는 그 관계가 약합니다. 

이 부분은 이 논문의 이론에서 가장 주의해서 읽어야 할 부분입니다.

---

# Gradient magnitude도 error만으로 결정되지 않습니다

동일하게

$$
g_{i,j}=2x_i r_{i,j}
$$

이므로

```math
\boxed{
\|g_{i,j}\|_2
=
2|r_{i,j}|\|x_i\|_2
}
```

입니다.

따라서 gradient magnitude를 결정하는 것은

1. residual magnitude $|r_{i,j}|$
2. input norm $|x_i|_2$

**둘 다**입니다.

Fig. 5가 prediction error와 gradient magnitude의 강한 양의 관계를 보여주는 것은 맞지만, 논문은 해당 관계에 대한 Pearson $r$, Spearman $\rho$, confidence interval 등을 보고하지 않습니다. 

### 용어 메모 — Residual

Residual은

$$
r=y_{\text{pred}}-y_{\text{true}}
$$

형태의 **부호가 있는 예측오차**입니다. MAE에 사용되는 $|r|$와 달리 $r$ 자체에는 양·음의 방향 정보가 남아 있습니다.

---

# 2-1-4. Variable grouping

중심화된 두 벡터에서는 Pearson correlation과 cosine similarity가 연결됩니다.

```math
\rho(x_a,x_b)
=
\frac{
(x_a-\bar x_a)^\top(x_b-\bar x_b)
}{
\|x_a-\bar x_a\|_2
\|x_b-\bar x_b\|_2
}
```

따라서 논문은 절대상관 행렬을

```math
R_X[a,b]
=
|\rho(x_a,x_b)|
```

로 구성하고 유사한 변수들을 agglomerative hierarchical clustering으로 묶습니다. 

Appendix에서는 **complete linkage**를 사용한다고 명시하며,

```math
d_{\bar\alpha}
=
1-\cos(\bar\alpha)
```

를 cluster cut 기준으로 사용합니다. 

### 기호

* $\rho(x_a,x_b)$: 두 변수의 Pearson correlation입니다.
* $\bar x_a$: $x_a$의 평균입니다.
* $\bar\alpha$: 허용할 최대 angle과 연결된 clustering hyperparameter입니다.
* $d_{\bar\alpha}$: correlation-based cluster distance입니다.
* complete linkage: 두 cluster를 비교할 때 가장 멀리 떨어진 원소쌍을 기준으로 cluster distance를 정하는 방식입니다.

### 왜 절댓값을 쓰는가?

$$
\rho\approx+1
$$

뿐 아니라

$$
\rho\approx-1
$$

도 강한 선형관계이기 때문입니다.

논문에서는 강한 음의 correlation을 가진 변수도 gradient conflict가 적은 경향을 관찰했습니다. 

---

# 2-1-5. Gradient balancing

각 변수/horizon residual magnitude를

```math
e_{i,j}
=
|x_i^\top\theta_j-y_{j,i}|
```

라고 합니다.

동일 horizon에서 변수들의 평균 error는

```math
K_j
=
\frac{1}{k}
\sum_{i=1}^{k}
e_{i,j}
```

이고, 동일 변수에서 horizon들의 평균 error는

```math
H_i
=
\frac{1}{h}
\sum_{j=1}^{h}
e_{i,j}
```

입니다.

weight는

```math
w_{i,j}^{(a)}
=
\frac{1}
{(K_jH_i)^a}
```

로 정합니다.

최종 weighted objective는

```math
F_W(\Theta)
=
\sum_{j=1}^{h}
\sum_{i=1}^{k}
w_{i,j}^{(a)}
\left(
x_i^\top\theta_j-y_{i,j}
\right)^2
```

입니다. 

### $a$의 의미

$a$는 balancing strength입니다.

* $a=0$: weighting 없음.
* $a=1$: inverse error-product weighting.
* $a=2$: 큰 error group을 훨씬 더 강하게 억제합니다.

Appendix 실험에서는 $a\in{0,1,2}$를 비교하고, main experiment의 hyperparameter search에서는 $a=2$가 자주 선택되었습니다. 

---

## 실제 gradient는 어떻게 변하는가?

$w$를 backpropagation graph에서 detach하여 상수로 취급하므로

```math
\nabla_{\theta_j}F_W
=
\sum_i
2w_{i,j}^{(a)}
x_i r_{i,j}
```

가 됩니다.

즉 큰 error를 가진 task의 gradient를 삭제하는 것이 아니라 **그 task가 전체 update를 지나치게 지배하지 못하게 재조정**합니다.

저자들은 이 weighting 연산 자체가 task별 gradient를 각각 계산하는 PCGrad/GradNorm보다 계산상 저렴하다고 설명합니다. 

### 주의할 점

논문에서 말하는 $O(1)$은 **이 gradient-weighting procedure에 대한 비교적 제한된 주장**으로 읽는 것이 정확합니다.

전체 MTLinear memory/parameter cost는 cluster 수 $c$에 따라 증가합니다.

예를 들어 Table 6에서 $\bar\alpha=\pi/6$일 때:

* ECL: 321 → **175 groups**
* Traffic: 862 → **862 groups**
* Weather: 21 → 13 groups
* ILI: 7 → 3 groups

입니다. 

Traffic에서는 해당 threshold에서 사실상 channel-independent model의 head 수가 그대로 유지됩니다.

---

# 2-1-6. MTLinear의 실제 모델 구조

Fig. 2의 전체 pipeline은 매우 단순합니다.

$$
X_{\text{train}}
\rightarrow
R_X
\rightarrow
\text{Hierarchical Clustering}
\rightarrow
\{G_1,\ldots,G_c\}
\rightarrow
\{f_1,\ldots,f_c\}
\rightarrow
\hat Y
$$

Fig. 2는 정확히 **Correlation estimation → Clustering → Linear module assignment**의 3단계를 보여줍니다. 

각 group $G_m$은 하나의 linear head $f_m$을 갖습니다.

$$
\hat Y_{G_m}=f_m(X_{G_m})
$$

그리고 모든 group prediction을 다시 합쳐

```math
\hat Y
=
\text{Concat}
\left(
\hat Y_{G_1},
\dots,
\hat Y_{G_c}
\right)
```

로 만듭니다.

---

## MTDLinear

DLinear은 입력을 trend와 remainder로 분해합니다.

개념적으로는

```math
X
=
T+S
```

$$
T=\text{MovingAverage}(X),
\qquad
S=X-T
$$

후

```math
\hat Y
=
W_TT+W_SS
```

입니다.

MTDLinear에서는 이것을 cluster별로 따로 수행합니다.

> 위 식은 **논문의 문장 설명을 이해하기 쉽게 수식화한 것**이며 논문의 별도 numbered equation은 아닙니다. 저자들은 DLinear가 average pooling으로 trend를 추출하고 trend/remainder에 별도 weight를 사용한다고 설명합니다. 

---

## MTNLinear

NLinear의 핵심은 lookback 마지막 값을 제거한 뒤 학습하고 다시 더하는 것입니다.

```math
X_i'
=
X_i-x_{i,\ell}
```

```math
\hat Y_i'
=
W X_i'
```

```math
\hat Y_i
=
\hat Y_i'+x_{i,\ell}
```

즉 level shift에 대한 부담을 줄입니다.

### 용어 메모 — Level shift

시계열의 형태 자체는 비슷하지만 전체 평균 수준이 위아래로 이동하는 현상입니다. 마지막 값을 빼면 모델은 absolute level보다 **변화 패턴**을 중심으로 학습할 수 있습니다.

---

# 2-1-7. 성능 향상

## Table 1 평균 성능

| Model         |  평균 MSE ↓ |  평균 MAE ↓ |
| ------------- | --------: | --------: |
| **MTNLinear** | **0.550** | **0.423** |
| MTDLinear     |     0.575 |     0.440 |
| PatchTST      |     0.584 |     0.431 |
| iTransformer  |     0.610 |     0.436 |
| DLinear       |     0.649 |     0.475 |
| FEDformer     |     0.680 |     0.483 |
| Autoformer    |     0.741 |     0.515 |
| Crossformer   |     0.902 |     0.579 |



단순 산술 비교를 하면 MTNLinear의 평균 MSE는

$$
\frac{0.584-0.550}{0.584}\times100
\approx 5.82\%
$$

PatchTST보다 낮고,

$$
\frac{0.610-0.550}{0.610}\times100
\approx9.84\%
$$

iTransformer보다 낮습니다.

하지만 **이 숫자를 “MTLinear이 PatchTST보다 통계적으로 5.82% 우월하다”라고 해석하면 안 됩니다.** 뒤의 통계 검증 부분에서 이유를 설명하겠습니다.

---

# 3. 주장별 Page / Figure / Table 위치

| 내용                                 | 논문 위치                                      |
| ---------------------------------- | ------------------------------------------ |
| MTSF를 MTL로 해석                      | PDF pp.1–3, Eq. (1)–(6)                    |
| Tragic Triad                       | PDF p.2                                    |
| correlation과 gradient conflict 관계  | **Fig. 1, PDF p.4**                        |
| linear gradient derivation         | Eq. (5), Eq. (6), PDF p.3; Appendix A p.13 |
| Pearson/cosine 기반 grouping         | Sec. 4.2, pp.4–5                           |
| gradient weighting                 | Eq. (7), Eq. (8), pp.4–5                   |
| 전체 MTLinear architecture           | **Fig. 2, p.5**                            |
| main benchmark                     | **Table 1, p.5**, full Table 7 p.18        |
| component ablation                 | **Table 2, p.6**                           |
| long lookback                      | **Fig. 3, p.6**                            |
| clustering sensitivity             | **Fig. 4, p.7**                            |
| gradient method 비교                 | **Table 3, p.7**                           |
| Linear/RLinear/DLinear/NLinear 적용성 | **Table 4, p.7**                           |
| error ↔ gradient magnitude         | **Fig. 5, p.8**                            |
| cross-dataset probing              | **Table 5, p.8**                           |
| limitations/future work            | pp.8–9                                     |
| correlation/conflict matrix        | **Fig. 6, Appendix p.14**                  |
| dendrogram/group count             | **Fig. 7/Table 6, p.15**                   |
| sensitivity $a$                    | **Fig. 8, p.17**                           |
| 3-seed standard deviation          | **Table 9, p.20**                          |

---

# 4. 저자가 보고한 내용과 내 해석을 분리

| 항목          | **저자가 직접 보고한 내용**                                                            | **내 해석 / 검증**                                                             |                            |                                                                                   |
| ----------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------- | -------------------------- | --------------------------------------------------------------------------------- |
| 연구 주제       | Multivariate forecasting을 MTL 관점으로 재정의하여 variable relationship을 활용           | 핵심 공헌은 새로운 linear layer 자체보다 **어떤 변수가 parameter를 공유해야 하는지 명시적으로 설계했다는 점** |                            |                                                                                   |
| gradient 방향 | gradient direction은 $x_i$에 의해 결정되고 variable correlation이 task alignment를 나타냄 | 정확히는 $g_{i,j}=2r_{i,j}x_i$이므로 **residual의 부호까지 포함해야 방향이 정해짐**             |                            |                                                                                   |
| gradient 크기 | prediction error가 큰 variable의 gradient가 크게 되어 optimization을 지배               | 정확히는 $\mid g_{i,j} \mid =2 \mid r_{i,j} \mid \mid x_i \mid$이므로 **input norm도 중요**                                                      |
| grouping    | $\mid \text{PCC} \mid$가 높은 variable을 clustering | zero-lag linear dependence에 최적화된 heuristic이며 lagged/nonlinear dependency는 놓칠 수 있음 |
| weighting   | $w=(K_jH_i)^{-a}$로 dominant gradient 완화                                      | 효과적인 heuristic이지만 scale/unit에 민감할 수 있음                                    |                            |                                                                                   |
| 결과          | MTNLinear global MSE/MAE가 0.550/0.423으로 전체 평균에서 우수                           | 평균 성능은 강력하지만 dataset별 winner는 다르며 우월성이 universal하지 않음                     |                            |                                                                                   |
| 일반화         | longer lookback 및 transfer experiment에서 유망                                   | 일반화 **가능성**은 지지하지만 domain shift/generalization을 정식으로 검증한 규모는 아님           |                            |                                                                                   |
| 한계          | non-correlated variate의 useful information을 활용하지 못함                          | 이 한계는 매우 중요하며, nonlinear/conditional/lagged interaction으로 확장할 이유가 됨       |                            |                                                                                   |
| 향후          | MTLinear을 SOTA decoder로 사용, cluster를 training 중 학습                           | 이것이 실제로 논문 아이디어의 가장 자연스러운 차세대 연구 방향                                       |                            |                                                                                   |

저자들이 밝힌 limitation과 future plan은 PDF pp.8–9에 명시되어 있습니다. 

---

# 5. 통계적으로 취약한 부분 및 비교 불가능한 수치

이 부분은 논문 평가에서 상당히 중요합니다.

| 문제                       | 구체적 내용                                                     | 영향                                                   |
| ------------------------ | ---------------------------------------------------------- | ---------------------------------------------------- |
| **seed 수가 3개**           | 각 experiment 평균은 3 seeds                                   | variance 추정이 다소 불안정                                  |
| **baseline SD 없음**       | Table 9는 MTLinear SD만 보고, baseline SD는 다른 논문에서 가져와 제공하지 않음 | MTLinear vs baseline significance 계산 불가              |
| **동일한 재실험 조건이 아님**       | Table 1 baseline 대부분은 iTransformer 논문 및 각 원 논문에서 가져온 값     | 완전히 controlled comparison이 아님                        |
| p-value/CI 없음            | paired test, confidence interval 없음                        | “statistically significant”를 말할 근거 부족                |
| hyperparameter selection | seed별 validation grid search 후 best setting 선택             | nested evaluation이 아니므로 selection variance가 추가될 수 있음 |
| Fig.1 정량 검정 없음           | correlation vs conflict의 대표적인 pair curve                   | 전체 pair에 대한 회귀/correlation 통계가 없음                    |
| Fig.5 정량 검정 없음           | error-gradient magnitude가 시각적으로 양의 관계                      | $r$, $\rho$, slope CI 등이 없음                          |
| global average           | 서로 다른 dataset의 MSE를 산술평균                                   | dataset scale/난이도 차이가 aggregate를 왜곡할 수 있음            |
| transfer 실험 좁음           | 2 source configurations + 소수 target                        | domain-generalization 결론에는 불충분                       |
| long-lookback 범위 제한      | Fig.3은 주로 3 datasets, horizon 96                           | 모든 형태의 long-context generalization을 보장하지 않음          |

논문 Appendix는 “모든 experiment에서 3개의 seed 평균을 사용하고, 각 seed에서 validation score가 가장 좋은 grid-search setting을 선택했다”고 명시합니다. 

또한 Table 9에서 저자 스스로 **다른 baseline들의 standard deviation은 대부분 다른 논문에서 가져왔기 때문에 제공하지 않는다**고 명시합니다. 

따라서 Table 1에서

$$
0.550 < 0.584
$$

라는 것은 명백하지만,

```math
H_0:
\mu_{\text{MTLinear}}
=
\mu_{\text{PatchTST}}
```

를 통계적으로 기각했다고 말할 수는 없습니다.

### 특히 “significant improvement” 표현에 주의

Table 5의 설명에는 “significant improvements”와 유사한 표현이 있지만, **통계검정상의 statistical significance라는 의미로 읽어서는 안 됩니다.** p-value나 confidence interval이 제시되지 않았기 때문입니다. 

---

## 비교 불가능하거나 조심해서 비교해야 할 수치

### ① MTLinear vs 외부 최신 연구의 raw MSE

예를 들어 MTLinear의

$$
\text{MSE}=0.550
$$

과 Time-MoE나 Moirai 논문에 보고된 MSE 하나를 직접 비교해서는 안 됩니다.

이유는 다음이 다를 수 있기 때문입니다.

$$
\text{dataset split},
\quad
\ell,
\quad
h,
\quad
\text{normalization},
\quad
\text{training regime},
\quad
\text{pretraining data}
$$

따라서 8-2의 최신 연구 비교는 **architecture와 generalization protocol 중심으로 비교**하겠습니다.

---

## 논문 내부에서 발견되는 소규모 reporting 이슈

Main Table 1에서는 MTNLinear 평균 MAE가 **0.423**, extended Table 7에서는 반올림 결과가 **0.422**로 기재됩니다.  

이것은 결론을 바꿀 정도의 문제는 아니며 **rounding/reporting discrepancy**로 보는 것이 적절합니다.

---

# 6. 이 논문이 답하지 않는 질문

1. **Correlation clustering은 반드시 training split에서만 계산되는가?**
   논문 본문과 Appendix만으로는 모든 구현 경로에서 이 점이 명시적으로 충분히 드러나지 않습니다. 이것은 실제 적용 시 데이터 누수 방지를 위해 반드시 검증해야 합니다.

2. **변수의 correlation structure가 시간에 따라 바뀌면 어떻게 되는가?**

$$
\rho_{ij}^{\text{train}}
\neq
\rho_{ij}^{\text{test}}
$$

이면 fixed cluster가 잘못된 task-sharing 구조가 될 수 있습니다.

3. 왜 zero-lag Pearson correlation이어야 하는가?

예를 들어

$$
x_b(t)\approx x_a(t-5)
$$

이면 causal/lagged dependence가 매우 강해도

$$
\rho(x_a(t),x_b(t))
$$

가 작을 수 있습니다.

4. nonlinear dependence는 어떻게 처리하는가?

```math
x_b(t)
=
x_a(t)^2+\epsilon_t
```

같은 관계는 Pearson correlation이 0에 가까울 수도 있습니다.

5. correlation이 낮지만 다른 channel의 정보가 prediction에 중요하다면 어떻게 하는가?

저자들도 이를 limitation으로 인정합니다. 

6. variable마다 측정 단위가 다를 때 weighting은 scale invariant한가?

```math
w_{i,j}^{(a)}
=
(K_jH_i)^{-a}
```

이므로 scaling convention의 영향을 받을 수 있습니다.

7. 왜 cluster를 hard assignment해야 하는가?

현실적으로 한 변수는 여러 latent process와 동시에 관련될 수 있습니다.

8. cluster 개수가 매우 많을 때 small-sample problem은 어떻게 되는가?

9. point forecasting 이외에 uncertainty calibration은 가능한가?

10. 새로운 variable이 deployment 중 추가되는 **new-channel generalization**은 가능한가?

11. missingness, anomaly, noise, irregular sampling에서 grouping은 얼마나 안정적인가?

12. Table 1의 작은 차이가 실제 deployment에서 의미 있는지 paired significance test를 하면 유지되는가?

---

# 7. 가장 중요한 Figure 5개 해석

## Figure 1 — Correlation과 gradient conflict

**PDF p.4**

DLinear/PatchTST에서 여러 variable pair의 training epoch별 conflict 횟수를 그리고, legend에 해당 variable pair의 correlation을 표시합니다. 높은 $|\rho|$를 가진 pair에서 conflict가 적게 나타나는 경향이 Fig. 1의 핵심입니다. 

### 의미

이 그림이 없으면 Pearson clustering은 단순 heuristic에 가깝습니다. Fig. 1은

$$
\text{similar variables}
\Rightarrow
\text{similar optimization behavior}
$$

라는 설계 가설을 경험적으로 지지합니다.

### 단, 해석 한계

앞서 유도했듯

```math
g_a^\top g_b
=
4r_ar_bx_a^\top x_b
```

이므로 correlation은 conflict의 **충분조건도 필요조건도 아닙니다**.

즉 Fig. 1은 correlation이 useful proxy라는 evidence이지 mathematical equivalence의 증명은 아닙니다.

---

# Figure 2 — MTLinear 전체 구조

**PDF p.5**

그림은 가장 직관적인 논문의 핵심입니다.

$$
\boxed{
\text{Correlation}
\rightarrow
\text{Clustering}
\rightarrow
\text{Linear Heads}
}
$$



기존 두 극단은

$$
\text{모든 channel 공유}
$$

와

$$
\text{channel마다 완전히 독립}
$$

입니다.

MTLinear은 그 사이를

$$
\text{cluster별 parameter sharing}
$$

으로 만듭니다.

이것이 이 연구의 가장 중요한 inductive bias입니다.

### 용어 메모 — Inductive bias

모델이 학습하기 전에 “세상은 대략 이런 구조일 것이다”라고 넣어 주는 가정입니다. MTLinear의 경우 **상관성이 높은 변수는 같은 forecasting rule을 어느 정도 공유할 것이다**라는 가정입니다.

---

# Figure 3 — Lookback 증가에 대한 일반화

**PDF p.6**

lookback을

$$
192,\;336,\;512,\;720
$$

으로 바꾸면서 horizon $96$에서 Electricity, ETTm2, Weather의 MSE를 비교합니다. 저자들은 MTLinear이 긴 lookback에서 지속적으로 좋아지는 경향을 강조하며, iTransformer보다 좋은 경우가 많다고 보고합니다. Electricity의 일부 lookback에서는 PatchTST가 더 좋습니다. 

### 일반화 관점에서 중요한 이유

훈련 input 길이가 달라졌을 때 성능이 즉시 붕괴하지 않는다는 것은

$$
\text{model robustness to context length}
$$

에 대한 긍정적 신호입니다.

하지만 이것은

$$
P_{\text{train}}(X,Y)
\neq
P_{\text{test}}(X,Y)
$$

상황을 평가한 **domain generalization experiment는 아닙니다.**

즉 “context-length robustness”와 “distribution-shift robustness”를 구분해야 합니다.

---

# Figure 4 — 하나의 optimal clustering threshold는 없다

**PDF p.7**

$\bar\alpha$를 변화시키면 cluster sharing 강도가 달라집니다. 

극단적으로

$$
\bar\alpha=0
$$

이면 거의 변수별 개별 모델이며,

$$
\bar\alpha=\frac{\pi}{2}
$$

이면 하나의 shared model에 가까워집니다.

Fig. 4의 가장 중요한 메시지는 **최적 $\bar\alpha$가 dataset마다 다르다**는 점입니다.

즉,

$$
\boxed{
\text{optimal parameter sharing is data-dependent}
}
$$

입니다.

이 결과 자체가 fixed clustering보다 **learnable clustering**이 필요하다는 강력한 근거가 됩니다.

---

# Figure 5 — Prediction error와 gradient magnitude

**PDF p.8**

Weather, ETTm2, ILI에서 error와 gradient magnitude를 비교하고, DLinear/NLinear 모두 양의 관계가 보입니다. 

이는

$$
|r|\uparrow
\Rightarrow
\|g\|\uparrow
$$

라는 penalty 설계의 경험적 근거입니다.

다만 정확한 식은

```math
\|g\|
=
2|r|\|x\|
```

이므로 다음 후속 그림이 있었다면 훨씬 강한 연구가 되었을 것입니다.

$$
\frac{\|g_{i,j}\|}
{2\|x_i\|}
\quad\text{vs.}\quad
|r_{i,j}|
$$

이론상 두 값은 거의 일치해야 하기 때문입니다.

---

## 보너스: 사실 이론 검증에는 Figure 6도 매우 중요

Appendix Fig. 6은 DLinear, PatchTST, Autoformer의 conflict matrix와 Pearson correlation matrix를 나란히 보여줍니다.

DLinear/PatchTST는 상당히 유사하지만 Autoformer는 유사성이 약합니다. 

따라서 MTLinear의 correlation-gradient 이론은 **모든 forecasting architecture에 보편적인 법칙이라기보다 temporal linear-weight structure와 특히 잘 맞는 설명**일 가능성이 큽니다.

---

# 8. 결론: 저자들의 시사점과 후속 연구

저자들은 결론에서 세 가지를 분명히 합니다.

첫째, 다변량 forecasting에서는 variable inter-relation을 무조건 무시할 이유가 없습니다. 둘째, correlation 기반 grouping과 error balancing만으로도 매우 단순한 linear forecaster가 강한 성능을 얻을 수 있습니다. 셋째, 현재 MTLinear은 **상관되지 않았지만 유용한 cross-variable information을 활용하지 못하고**, DLinear/NLinear 같은 특정 linear layer에 의존합니다. 

저자들이 직접 제시한 future work는 다음 방향입니다.

1. **MTLinear을 SOTA forecasting architecture의 decoder로 삽입**
2. preprocessing에서 cluster를 고정하지 않고 **training 중 cluster를 학습**
3. non-correlated cross-variate information과 linear-layer dependency 등의 한계 해결

이는 PDF p.9에서 명시되어 있습니다. 

---

# 8-1. 모델의 일반화 성능 향상 가능성

이 부분에서는 논문의 아이디어를 한 단계 더 발전시켜 보겠습니다.

## 우선순위 1 — Fixed hard clustering → Dynamic soft clustering

현재는

```math
G_i
=
\text{Cluster}(|R_X|)
```

를 학습 전에 한 번 계산합니다.

이 방식의 가장 큰 일반화 위험은 correlation drift입니다.

$$
R_{\text{train}}
\neq
R_{\text{future}}
$$

이면 잘못된 parameter-sharing 구조가 고정됩니다.

더 좋은 방식은 변수 $i$가 expert $m$에 속할 확률을 학습하는 것입니다.

```math
p_{i,m}(t)
=
\frac{
\exp(s_\phi(z_i(t),c_m))
}{
\sum_{q=1}^{M}
\exp(s_\phi(z_i(t),c_q))
}
```

그리고

```math
\hat y_i
=
\sum_{m=1}^{M}
p_{i,m}(t)
f_m(x_i)
```

로 예측합니다.

### 기호

* $z_i(t)$: 현재 variable $i$의 representation입니다.
* $c_m$: $m$번째 cluster/expert representation입니다.
* $s_\phi$: learned similarity function입니다.
* $p_{i,m}$: variable $i$가 expert $m$을 사용할 soft probability입니다.
* $f_m$: expert $m$입니다.

이렇게 하면 regime이 바뀔 때 cluster 관계도 움직일 수 있습니다.

2025년 KDD의 DUET는 바로 이 방향과 상당히 가깝게, **temporal clustering과 channel soft clustering을 동시에 사용하고 frequency-domain metric learning으로 channel 관계를 학습**합니다. ([KDD][4])

따라서 MTLinear 후속연구에서 가장 우선순위가 높습니다.

---

# 우선순위 2 — Pearson → lagged/nonlinear similarity

현재

```math
S_{ab}
=
|\rho(x_a(t),x_b(t))|
```

대신 적어도

```math
S_{ab}^{\text{lag}}
=
\max_{|\tau|\le\tau_{\max}}
\left|
\rho(x_a(t),x_b(t-\tau))
\right|
```

를 사용할 수 있습니다.

이를 더 발전시키면

* cross-correlation,
* coherence,
* mutual information,
* learned metric,
* Granger-type predictive dependence

등을 사용할 수 있습니다.

### 용어 메모 — Coherence

주파수별로 두 신호가 얼마나 같이 움직이는지를 측정하는 값입니다. 단순 Pearson correlation이 놓치는 **주기적 동조 관계**를 발견할 수 있습니다.

Crossformer는 2023년에 이미 cross-time과 cross-dimension dependency를 동시에 명시적으로 모델링했습니다. ([ICLR][5]) TSMixer 역시 time dimension과 feature dimension을 모두 mixing하면서 cross-variate information의 중요성을 보여주었습니다. ([ML Anthology][6])

---

# 우선순위 3 — Gradient balancing을 scale invariant하게

현재

```math
w_{i,j}
=
(K_jH_i)^{-a}
```

인데, 변수 scale이 다르면 absolute error 자체가 공정하게 비교되지 않을 수 있습니다.

이를

```math
\tilde e_{i,j}
=
\frac{
|r_{i,j}|
}{
\hat\sigma_i+\epsilon
}
```

로 표준화할 수 있습니다.

그리고

```math
\tilde K_j
=
\frac1k\sum_i\tilde e_{i,j},
\qquad
\tilde H_i
=
\frac1h\sum_j\tilde e_{i,j}
```

```math
\tilde w_{i,j}
=
\frac{1}
{
(\tilde K_j\tilde H_i+\epsilon)^a
}
```

로 바꾸는 것입니다.

* $\hat\sigma_i$: training data에서 계산한 variable $i$의 scale.
* $\epsilon$: 0으로 나누는 것을 방지하는 작은 양수입니다.

더 직접적으로는

$$
\|g_i\|
$$

자체를 EMA로 추정하여 balance할 수도 있습니다.

---

# 우선순위 4 — Complete separation 대신 Partial Parameter Sharing

현재 cluster가 다르면 parameter도 완전히 분리됩니다.

그러나 현실에서는

> “공통 dynamics + cluster-specific correction”

구조가 더 자연스러울 수 있습니다.

따라서

```math
\Theta_m
=
\Theta_0+\Delta_m
```

로 두고

$$
\min
\mathcal L
+
\lambda
\sum_{m=1}^{M}
\|\Delta_m\|_F^2
$$

로 regularization할 수 있습니다.

* $\Theta_0$: 모든 cluster가 공유하는 global dynamics.
* $\Delta_m$: cluster $m$만의 correction.
* $\lambda$: 각 cluster가 global model에서 지나치게 멀어지는 것을 막는 regularization strength.

이것은 특히 데이터가 적을 때 중요합니다.

hard separate heads는 sample 수가 적어지면

$$
n_m \ll p
$$

상황이 생길 수 있지만 partial pooling은 다른 cluster에서 정보를 빌릴 수 있기 때문입니다.

---

# 우선순위 5 — Low-rank group adaptation

parameter 수를 더 줄이려면

```math
\Theta_m
=
\Theta_0+
UA_mV^\top
```

처럼 만들 수 있습니다.

이는 LoRA와 비슷한 아이디어입니다.

cluster마다 큰 $\Theta_m$ 전체를 학습하지 않고 작은 $A_m$만 바꾸므로

$$
\text{variance}\downarrow,
\qquad
\text{memory}\downarrow
$$

를 기대할 수 있습니다.

Traffic처럼 cluster 수가 수백 개로 남는 경우 특히 유용합니다.

---

# 우선순위 6 — Rolling correlation adaptation

deployment 환경에서는

```math
R_t
=
\lambda R_{t-1}
+
(1-\lambda)\hat R_t
```

형태로 correlation structure를 갱신할 수 있습니다.

* $\lambda\approx1$: 과거 관계를 오래 유지.
* 작은 $\lambda$: 최근 regime에 빠르게 적응.

cluster switching에 hysteresis를 추가하면 noisy correlation 때문에 head가 계속 바뀌는 문제도 줄일 수 있습니다.

---

# 우선순위 7 — Foundation model의 decoder/adapter로 MTLinear

이 방향은 저자들이 직접 future work로 제안한 부분입니다.

Moirai는 27B 이상의 관측치와 9개 domain을 사용해 하나의 universal forecaster가 다양한 frequency와 variable 수를 처리하는 방향을 제시했습니다. ([Proceedings of Machine Learning Research][7]) Time-MoE는 sparse MoE 구조와 대규모 pretraining으로 모델을 최대 2.4B parameter 수준까지 확장하고 Time-300B를 사용합니다. ([ICLR Proceedings][8])

따라서 매우 흥미로운 구조는

$$
X
\xrightarrow{\text{Foundation Encoder}}
Z
\xrightarrow{\text{Learned MT Grouping}}
\{Z_m\}
\xrightarrow{\text{MTLinear Decoder}}
\hat Y
$$

입니다.

이 경우 foundation model은 **보편적 representation**을 제공하고 MTLinear은 **target dataset의 variable-specific adaptation**을 담당하게 됩니다.

---

# 우선순위 8 — Point accuracy → probabilistic generalization

논문은 MSE/MAE 중심입니다.

실제 deployment에서는

$$
p(Y_{t+1:t+h}\mid X_{1:t})
$$

전체를 예측하거나

$$
Q_\tau(Y\mid X)
$$

형태의 quantile prediction이 더 유용할 수 있습니다.

일반화 성능을 평가할 때도 단순히

$$
\text{MSE}
$$

만 보는 것이 아니라

* CRPS,
* coverage,
* calibration error,
* interval width

를 보는 것이 필요합니다.

---

# 8-2. 2020년 이후 최신 관련 연구 비교

**중요:** 아래 표의 “성능”은 논문 간 raw MSE를 직접 비교하지 않습니다. 각 연구의 dataset/split/lookback/pretraining 조건이 다르므로, **일반화 설계와 variable interaction 방식**을 비교합니다.

| 연도 / 연구                   | 핵심 아이디어                                     | Cross-variate 처리                       | 일반화 관점                                                    | MTLinear과의 관계                                |
| ------------------------- | ------------------------------------------- | -------------------------------------- | --------------------------------------------------------- | -------------------------------------------- |
| **2020 PCGrad**           | conflicting gradient를 projection으로 수정       | task 단위                                | task interference 감소                                      | MTLinear의 MTL 이론적 출발점 중 하나                   |
| **2023 DLinear/NLinear**  | 매우 단순한 temporal linear mapping              | 사실상 channel-independent                | 단순성으로 과적합 억제                                              | MTLinear의 직접 backbone                        |
| **2023 PatchTST**         | patch token + channel independence          | 제한적                                    | self-supervised cross-dataset transfer                    | MTLinear이 “독립 channel만으로 충분한가?”를 재질문         |
| **2023 Crossformer**      | cross-time + cross-dimension attention      | 명시적                                    | complex dependency 표현                                     | MTLinear보다 expressive하지만 복잡                  |
| **2023 TSMixer**          | time/feature MLP mixing                     | 명시적 feature mixing                     | simple architecture + auxiliary/cross-variate information | MTLinear의 linear grouping과 상보적               |
| **2024 iTransformer**     | variable 자체를 token으로 사용                     | attention으로 variate relation           | variable/general lookback generalization                  | MTLinear Table 1의 핵심 비교대상                    |
| **2024 SOFTS**            | STAR aggregate-redistribute                 | global core를 통해 channel interaction    | distribution drift와 many-channel 효율성 고려                   | hard clustering 없는 global interaction의 대안    |
| **2024 Moirai**           | universal pretrained Transformer            | arbitrary multivariate inputs          | cross-domain zero-shot                                    | MTLinear decoder 확장에 적합                      |
| **2024 Tiny Time Mixers** | 약 1M parameter부터 시작하는 pretrained mixer      | multi-level channel modeling           | zero/few-shot + 다양한 resolution                            | small-data 일반화의 중요한 대안                       |
| **2025 MTLinear**         | correlation grouping + gradient balancing   | hard cluster                           | 간단·효율적이지만 static relation                                 | 본 논문                                         |
| **2025 DUET**             | temporal cluster + channel soft cluster     | frequency-domain learned soft relation | distribution heterogeneity 직접 대응                          | **MTLinear의 가장 직접적인 차세대 방향**                 |
| **2025 TimeMixer++**      | time/frequency multi-scale pattern mixing   | task-adaptive representation           | 여러 TS task로 generalization                                | static Pearson보다 richer representation       |
| **2025 Time-MoE**         | sparse MoE time-series foundation model     | 대규모 pretrained representation          | domain/context/horizon scaling                            | MTLinear을 lightweight adapter로 연결할 가능성       |
| **2026 SEER**             | bad/noisy patch 자동 filtering/replacement    | channel-adaptive representation        | missing, shift, anomaly, noise robustness                 | MTLinear이 아직 다루지 않는 corruption robustness 보완 |
| **2026 TFMixer**          | irregular TS의 time-frequency joint modeling | irregular multivariate modeling        | 비등간격·비동기 channel 일반화                                      | MTLinear의 regular sampling 가정 밖으로 확장         |

### 주요 연구 근거

DLinear/NLinear 계열을 제안한 2023 AAAI 연구는 단순한 한 층 linear model이 당시 Transformer 계열과 매우 강하게 경쟁할 수 있음을 보여주어 MTLinear의 기본 backbone 철학을 만들었습니다. ([AAAI Publications][9])

PatchTST는 patching과 channel independence를 사용하면서 self-supervised pretraining의 dataset 간 transfer 가능성을 보였습니다. ([ML Anthology][2])

Crossformer는 반대로 변수 간 의존성을 명시적으로 모델링하며 cross-time stage와 cross-dimension stage를 결합합니다. ([ICLR][5])

iTransformer는 각 variable의 전체 시계열을 하나의 token으로 뒤집어 넣고 self-attention으로 variable correlation을 학습하며, 논문 자체가 arbitrary lookback 및 variate generalization을 장점으로 제시합니다. ([ICLR Proceedings][10])

SOFTS는 channel independence가 distribution drift에 유리하지만 correlation을 놓친다는 문제를 출발점으로 하고, STAR라는 centralized aggregate–redistribute 구조로 linear complexity의 channel interaction을 제안했습니다. ([NeurIPS Proceedings][11])

Moirai는 단일 dataset 전용 모델에서 벗어나 arbitrary number of variates, cross-frequency, heterogeneous distributions를 처리하는 universal forecasting으로 연구축을 확장했습니다. ([Proceedings of Machine Learning Research][7])

Tiny Time Mixers는 compact pretrained model에서도 adaptive patching, resolution sampling, channel correlation modeling을 통해 zero/few-shot generalization을 강화할 수 있음을 보여줍니다. ([NeurIPS Proceedings][12])

DUET는 MTLinear과 매우 흥미로운 비교대상입니다. MTLinear이 **정적 Pearson hard-clustering**인 반면 DUET는 temporal distribution heterogeneity와 channel relationship을 각각 clustering하고, channel 측에서는 frequency-domain metric learning + soft clustering을 사용합니다. ([KDD][4])

TimeMixer++은 여러 time scale과 frequency resolution을 동시에 표현하여 forecasting뿐 아니라 여러 time-series task에 사용할 수 있는 general pattern machine을 지향합니다. ([ICLR Proceedings][13])

Time-MoE는 sparse expert activation을 통해 대규모 capacity와 inference efficiency를 동시에 겨냥하며, 대규모 cross-domain pretraining이 향후 일반화 연구의 또 다른 축임을 보여줍니다. ([ICLR Proceedings][8])

2026년의 SEER는 missing values, distribution shift, anomaly, white noise가 patch quality를 훼손한다는 문제를 직접 다루어 **“정상 benchmark에서의 평균 MSE”에서 “실제 환경 corruption에 대한 robustness”로 평가 축이 이동하고 있음**을 보여줍니다. 현재 확인 가능한 arXiv 논문과 공식 구현은 SEER를 ICML 2026 연구로 표시합니다. ([arXiv][14])

또한 TFMixer는 irregular multivariate series에서 non-uniform sampling과 variable asynchronicity를 대상으로 learnable NUDFT와 local patch mixing을 결합합니다. 이는 MTLinear이 기본적으로 가정하는 regular lookback matrix보다 훨씬 넓은 deployment 조건을 다룹니다. ([arXiv][15])

---

# MTLinear이 이후 연구에 미치는 영향

제가 보는 MTLinear의 연구적 영향은 **“linear model이 다시 강하다”는 주장 자체에는 있지 않습니다.** 이것은 이미 DLinear 계열에서 잘 알려졌습니다. ([AAAI Publications][9])

더 중요한 것은 다음 decomposition입니다.

```math
\boxed{
\text{MTS forecasting}
=
\text{Task Relation Identification}
+
\text{Selective Parameter Sharing}
+
\text{Optimization Balancing}
}
```

즉 이전에는 주로

> “channel을 섞을 것인가, 독립시킬 것인가?”

라는 이분법이었다면, MTLinear은

> **“어떤 channel끼리는 공유하고 어떤 channel끼리는 분리할 것인가?”**

라는 중간 문제를 명확하게 제시합니다.

이 방향은 이후 DUET 같은 **adaptive/soft channel grouping** 연구와 매우 자연스럽게 연결됩니다. ([arXiv][16])

---

# 앞으로 연구 시 가장 중요하게 고려할 점

일반화 성능을 중심으로 한다면 다음 형태의 **MTLinear-v2 연구**가 가장 논리적입니다.

$$
\boxed{
\begin{aligned}
&\text{Train-only relation estimation}\\
&\downarrow\\
&\text{Lag/Frequency/Nonlinear similarity}\\
&\downarrow\\
&\text{Dynamic soft clustering}\\
&\downarrow\\
&\text{Global + group-specific partial pooling}\\
&\downarrow\\
&\text{Scale-invariant gradient balancing}\\
&\downarrow\\
&\text{Drift-aware online update}
\end{aligned}
}
$$

특히 제가 우선 검증할 가설은 다음입니다.

```math
\boxed{
\Theta_g
=
\Theta_{\text{global}}
+
\Delta_g
}
```

와

$$
\boxed{
p(g\mid x_i,t)
\text{를 고정하지 않고 학습}
}
$$

을 함께 사용하는 것입니다.

그러면 MTLinear의 장점인 **낮은 모델 복잡도**는 유지하면서도,

* hard cluster 오류,
* 미래 correlation drift,
* 작은 group의 sample 부족,
* nonstationary relation

을 동시에 줄일 가능성이 있습니다.

---

## 추천 실험 프로토콜

이 후속 연구가 정말 “일반화가 개선되었다”고 말하려면 단순 Train/Validation/Test 한 번으로는 부족합니다.

시간순으로 $K$개의 rolling origin을 만들고

$$
\mathcal D_{\text{train}}^{(q)} < \mathcal D_{\text{valid}}^{(q)} < \mathcal D_{\text{test}}^{(q)}
$$

를 $q=1,\ldots,K$에 대해 반복해야 합니다.

그리고 모델 $A,B$의 각 fold 차이를

```math
d_q
=
M_q^{A}-M_q^{B}
```

로 두어 paired test를 해야 합니다.

추천하는 검증은

* 최소 5–10 random seeds,
* rolling-origin evaluation,
* train-only preprocessing/clustering,
* block bootstrap confidence interval,
* paired Wilcoxon test,
* 여러 모델 비교 시 Holm correction,
* mean MSE뿐 아니라 average rank와 normalized improvement,
* correlation-drift가 큰 구간의 별도 성능,
* 새로운 channel/domain/longer-lookback test

입니다.

이런 조건에서 개선이 유지되어야 **MTLinear의 다음 연구가 단순 benchmark tuning이 아니라 실제 generalization improvement**라고 주장할 수 있습니다.

---

# 최종 평가

이 논문은 **구조는 매우 단순하지만 연구 질문은 좋은 논문**입니다. MTLinear의 핵심은 “더 복잡한 neural network”가 아니라, 선형 forecasting의 gradient를 관찰하여 **어떤 변수들이 하나의 task를 형성해야 하는지**, 그리고 **어떤 task가 optimization을 과도하게 지배하는지를 분리해서 다룬 것**입니다.

다만 이론적으로 가장 조심해야 할 부분은

$$
\text{variable correlation}
\not\equiv
\text{gradient alignment}
$$

이라는 점입니다. 정확하게는

```math
\boxed{
g_{a,j}^{\top}g_{b,j}
=
4r_{a,j}r_{b,j}x_a^\top x_b
}
```

이므로 **input correlation, residual sign, residual magnitude, input norm**을 모두 봐야 합니다. 이 관찰을 반영한 dynamic/soft task grouping과 normalized gradient balancing이 오히려 이 논문에서 자연스럽게 도출되는 가장 강력한 후속 연구 주제라고 판단합니다.

또한 현재 최신 연구 흐름은 MTLinear이 제시한 정적 channel grouping에서 더 나아가 **DUET식 adaptive clustering, SOFTS식 효율적 cross-channel fusion, Moirai/Time-MoE식 cross-domain pretraining, SEER식 corruption robustness, TFMixer식 irregular multivariate forecasting**으로 확대되고 있습니다. 따라서 향후 연구에서는 단순히 benchmark MSE 몇 % 개선보다는 **“시간에 따라 관계가 변하고, sample이 적고, noise/shift가 발생하는 새로운 domain에서도 cluster와 parameter-sharing이 올바르게 유지되는가”**를 핵심 generalization criterion으로 삼는 것이 더 중요합니다. ([NeurIPS Proceedings][11])

---

## 참고한 논문·사이트

**원 논문 및 구현:**
[A Multi-Task Learning Approach to Linear Multivariate Forecasting — PMLR/AISTATS 2025](https://proceedings.mlr.press/v258/nochumsohn25a.html)
[A Multi-Task Learning Approach to Linear Multivariate Forecasting — arXiv:2502.03571](https://arxiv.org/abs/2502.03571)
[MTLinear — Official GitHub Implementation](https://github.com/azencot-group/MTLinear)

**MTL 이론:**
[Gradient Surgery for Multi-Task Learning — NeurIPS 2020](https://papers.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)
[Conflict-Averse Gradient Descent for Multi-task Learning — NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html)

**다변량/선형 forecasting 관련:**
[Are Transformers Effective for Time Series Forecasting? — AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/26317)
[A Time Series Is Worth 64 Words: Long-Term Forecasting with Transformers — ICLR 2023](https://mlanthology.org/iclr/2023/nie2023iclr-time/)
[Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting — ICLR 2023](https://iclr.cc/virtual/2023/poster/12023)
[TSMixer: An All-MLP Architecture for Time Series Forecasting — TMLR 2023](https://mlanthology.org/tmlr/2023/chen2023tmlr-tsmixer/)
[iTransformer: Inverted Transformers Are Effective for Time Series Forecasting — ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/2ea18fdc667e0ef2ad82b2b4d65147ad-Abstract-Conference.html)
[TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting — ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/a7ac8a21e5a27e7ab31a5f42a0117bdb-Abstract-Conference.html)
[SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion — NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/754612bde73a8b65ad8743f1f6d8ddf6-Abstract-Conference.html)
[Unified Training of Universal Time Series Forecasting Transformers (Moirai) — ICML 2024](https://proceedings.mlr.press/v235/woo24a.html)
[Tiny Time Mixers — NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/874a4d89f2d04b4bcf9a2c19545cf040-Abstract-Conference.html)
[DUET: Dual Clustering Enhanced Multivariate Time Series Forecasting — KDD 2025 listing](https://www.kdd.org/kdd2025/research-track-papers-2/)
[DUET — arXiv](https://arxiv.org/abs/2412.10859)
[TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis — ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2b187165e28fdfdc0ffb34d1bfff2b0c-Abstract-Conference.html)
[Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts — ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/558d48c1f08675daa636e09bfe94a89e-Abstract-Conference.html)

**2026년 확장 연구:**
[SEER: Transformer-based Robust Time Series Forecasting via Automated Patch Enhancement and Replacement — arXiv](https://arxiv.org/abs/2602.00589)
[SEER — Official Implementation](https://github.com/decisionintelligence/SEER/)
[Bridging Time and Frequency: A Joint Modeling Framework for Irregular Multivariate Time Series Forecasting — arXiv](https://arxiv.org/abs/2602.00582)
[TFMixer — Official Implementation](https://github.com/decisionintelligence/TFMixer)

[1]: https://proceedings.mlr.press/v258/nochumsohn25a.html "https://proceedings.mlr.press/v258/nochumsohn25a.html"
[2]: https://mlanthology.org/iclr/2023/nie2023iclr-time/ "https://mlanthology.org/iclr/2023/nie2023iclr-time/"
[3]: https://papers.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html "https://papers.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html"
[4]: https://www.kdd.org/kdd2025/research-track-papers-2/ "https://www.kdd.org/kdd2025/research-track-papers-2/"
[5]: https://iclr.cc/virtual/2023/poster/12023 "https://iclr.cc/virtual/2023/poster/12023"
[6]: https://mlanthology.org/tmlr/2023/chen2023tmlr-tsmixer/ "https://mlanthology.org/tmlr/2023/chen2023tmlr-tsmixer/"
[7]: https://proceedings.mlr.press/v235/woo24a.html "https://proceedings.mlr.press/v235/woo24a.html"
[8]: https://proceedings.iclr.cc/paper_files/paper/2025/hash/558d48c1f08675daa636e09bfe94a89e-Abstract-Conference.html "https://proceedings.iclr.cc/paper_files/paper/2025/hash/558d48c1f08675daa636e09bfe94a89e-Abstract-Conference.html"
[9]: https://ojs.aaai.org/index.php/AAAI/article/view/26317 "https://ojs.aaai.org/index.php/AAAI/article/view/26317"
[10]: https://proceedings.iclr.cc/paper_files/paper/2024/hash/2ea18fdc667e0ef2ad82b2b4d65147ad-Abstract-Conference.html "https://proceedings.iclr.cc/paper_files/paper/2024/hash/2ea18fdc667e0ef2ad82b2b4d65147ad-Abstract-Conference.html"
[11]: https://proceedings.neurips.cc/paper_files/paper/2024/hash/754612bde73a8b65ad8743f1f6d8ddf6-Abstract-Conference.html "https://proceedings.neurips.cc/paper_files/paper/2024/hash/754612bde73a8b65ad8743f1f6d8ddf6-Abstract-Conference.html"
[12]: https://proceedings.neurips.cc/paper_files/paper/2024/hash/874a4d89f2d04b4bcf9a2c19545cf040-Abstract-Conference.html "https://proceedings.neurips.cc/paper_files/paper/2024/hash/874a4d89f2d04b4bcf9a2c19545cf040-Abstract-Conference.html"
[13]: https://proceedings.iclr.cc/paper_files/paper/2025/hash/2b187165e28fdfdc0ffb34d1bfff2b0c-Abstract-Conference.html "https://proceedings.iclr.cc/paper_files/paper/2025/hash/2b187165e28fdfdc0ffb34d1bfff2b0c-Abstract-Conference.html"
[14]: https://arxiv.org/abs/2602.00589 "https://arxiv.org/abs/2602.00589"
[15]: https://arxiv.org/abs/2602.00582 "https://arxiv.org/abs/2602.00582"
[16]: https://arxiv.org/abs/2412.10859 "https://arxiv.org/abs/2412.10859"
