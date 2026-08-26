# Mixture-of-Linear-Experts for Long-term Time Series Forecasting

분석 기준은 사용자가 첨부한 **arXiv v3, 2024-05-01, 23쪽 PDF**와 AISTATS 2024/PMLR 공식 출판본입니다. 본 논문은 Ronghao Ni, Zinan Lin, Shuaiqi Wang, Giulia Fanti가 발표했으며 AISTATS 2024, PMLR Vol. 238, pp. 4672–4680에 게재되었습니다. 아래의 페이지 번호는 특별히 표시하지 않는 한 **첨부 PDF 기준**입니다.   ([Proceedings of Machine Learning Research][1])

---

## 1. Executive Summary — 10문장 이내

1. 이 논문의 핵심 문제의식은 DLinear·RLinear 같은 단순한 **linear-centric forecasting model**이 높은 평균 성능을 내더라도, 요일·계절·운전 regime처럼 시간에 따라 예측 규칙 자체가 바뀌는 시계열에는 하나의 선형 mapping만으로 대응하기 어렵다는 것입니다.
2. 저자들은 이를 해결하기 위해 하나의 선형 모델 대신 여러 개의 독립적인 linear expert를 두고, 시작 시각의 timestamp embedding을 입력받는 router가 expert들의 출력을 가중합하는 **Mixture-of-Linear-Experts(MoLE)**를 제안합니다.
3. MoLE의 router는 전체 입력 시계열이 아니라 첫 timestamp의 시간 정보를 사용하고, 각 변수(channel)마다 서로 다른 expert weight를 생성한다는 점이 일반적인 단일 gating 방식과 구별됩니다.
4. MoLE는 새로운 forecasting backbone이라기보다 DLinear, RLinear, RMLP와 같은 기존 linear-centric model을 여러 expert로 복제하여 적응적으로 조합하는 plugin 구조입니다.
5. 저자 실험에서 MoLE는 DLinear의 32/44, RLinear의 38/44, RMLP의 33/44 설정에서 MSE를 개선했습니다.
6. 또한 PatchTST 결과가 존재하여 비교 가능한 28개 설정 중 MoLE가 적용된 linear-centric model은 19개, 즉 68%에서 당시 비교군 기준 SOTA를 기록했으며 single-head linear model의 7/28, 25%보다 높았습니다.
7. 그러나 timestamp-aware router가 항상 우월했던 것은 아니며 Table 3에서는 random input이나 random output routing이 일부 데이터에서 TimeIn보다 더 좋은 결과를 내므로 단순히 “시간 정보를 넣으면 항상 좋아진다”는 결론은 성립하지 않습니다.
8. Figure 5는 특히 짧은 input history에서 timestamp-based expert selection의 이득이 커지고 긴 history에서는 single-head와의 격차가 작아지는 현상을 보여 줍니다.
9. 따라서 MoLE의 가장 강한 의미는 **저비용 linear model에 regime specialization을 도입했다는 것**이며, 논문이 실제로 검증한 일반화는 동일 데이터셋의 미래 hold-out 구간에 대한 일반화이지 cross-dataset, zero-shot, 강한 OOD distribution shift 일반화까지는 아닙니다. ([Proceedings of Machine Learning Research][1]) 

> **용어 — Linear-centric model:** 모델의 중심 계산이 거대한 Transformer가 아니라 시간축을 연결하는 하나 또는 소수의 선형층인 모델입니다.
> **Regime:** 시계열이 서로 다른 생성 규칙을 따르는 구간을 뜻합니다. 예를 들어 평일/주말, 정상 운전/부하 운전 등이 서로 다른 regime일 수 있습니다.
> **SOTA(State of the Art):** 해당 논문이 비교한 실험군에서 당시 가장 낮은 오차를 기록했다는 의미입니다. 2026년 현재 전체 시계열 분야의 최신 SOTA라는 의미는 아닙니다.

---

# 1-1. 연구의 목적과 필요성

이 연구의 출발점은 2023년 DLinear 연구에서 매우 단순한 선형 모델이 복잡한 Transformer 기반 LTSF 모델보다 강한 결과를 보였다는 사실입니다. DLinear 연구는 LTSF에서 복잡한 architecture 자체보다 적절한 시간축 mapping이 중요할 수 있다고 주장했습니다. ([AAAI Publications][2])

후속 RLinear/RMLP 연구에서는 이를 더 발전시켜 **linear mapping + RevIN + Channel Independence**가 상당한 성능을 낼 수 있다는 결과가 제시되었습니다. ([arXiv][3])

그러나 하나의 선형 mapping

$$
\mathbf y = W\mathbf x
$$

은 본질적으로 같은 $W$를 모든 시점에 적용합니다.

평일에는

$$
\mathbf y \approx W_{\text{weekday}}\mathbf x
$$

가 적절하고, 주말에는

$$
\mathbf y \approx W_{\text{weekend}}\mathbf x
$$

가 적절하다고 하더라도 single-head linear model은 사실상 하나의 절충된

$$
W_{\text{single}}
$$

을 학습해야 합니다.

MoLE가 해결하려는 문제는 바로 이 부분입니다.

즉,

$$
\boxed{
\text{하나의 복잡한 모델을 만들기보다 여러 개의 단순한 모델을 조건부로 조합한다}
}
$$

는 것이 설계 철학입니다. 첨부 PDF pp.1–2, 그리고 toy experiment pp.5, 14에서 이 아이디어가 명확히 드러납니다. 

> **비정상성(non-stationarity):** 시간에 따라 평균, 분산, 주기, 변수 간 관계 또는 예측식 자체가 변하는 현상입니다.
> **Mixture-of-Experts(MoE):** 하나의 모델이 모든 패턴을 처리하게 하지 않고 여러 expert를 둔 뒤 router가 상황별로 적합한 expert를 선택하거나 혼합하는 방식입니다.
> **Router/Gating:** 현재 입력에 어떤 expert를 얼마나 사용할지를 결정하는 작은 네트워크입니다.

---

# 2. 핵심 주장과 근거

| 핵심 주장                                           | 저자 근거                                                     | 위치                                    | 판단                                  |
| ----------------------------------------------- | --------------------------------------------------------- | ------------------------------------- | ----------------------------------- |
| 하나의 linear model보다 여러 temporal expert가 유리할 수 있음 | DLinear 32/44, RLinear 38/44, RMLP 33/44에서 MoLE가 MSE 개선   | p.6–7, **Table 2**                    | 강한 경험적 근거지만 모든 경우 개선은 아님            |
| 단순 parameter 증가만으로 설명되지는 않음                     | TimeIn/RandomIn/RandomOut ablation                        | p.8, **Table 3**                      | 부분적으로 지지되나 Random routing도 상당수 승리   |
| timestamp conditioning 효과는 input이 짧을수록 커짐       | input length 6→336 변화                                     | p.9, **Figure 5**                     | MoLE의 가장 흥미로운 메커니즘 근거               |
| head dropout은 timestamp overfitting 완화에 도움      | TimeIn + dropout이 긴 input에서 상대적으로 개선                      | p.9, **Figure 5**, p.17–19 **Fig.11** | 정규화 관점에서 설득력 있음                     |
| 큰 데이터셋에서 비교적 안정적인 향상                            | Weather/Electricity/Traffic에서 DLinear·RLinear 12/12 설정 개선 | p.6–7, **Table 2**                    | ETT처럼 작은 데이터에서는 훨씬 불안정              |
| 성능 향상을 위해 계산량이 폭발하지 않음                          | training 약 +26.45%, inference 약 +13.34%라고 보고              | p.20–21, **Table 10, Fig.12**         | 특정 GPU/implementation에 한정           |
| 당시 PatchTST 대비 경쟁력 있음                           | 비교 가능한 28개 설정 중 19개에서 SOTA                                | p.2, 6–7                              | **부분 비교만 가능**, 2026년의 SOTA를 의미하지 않음 |

특히 Table 2를 보면 성능 향상의 크기는 매우 균일하지 않습니다. 예를 들어 제가 원문 수치로 계산하면 Weather-96의 DLinear는 MSE가 $0.175\rightarrow0.147$로 약 **16.0% 감소**하지만, ETTh1-192에서는 $0.413\rightarrow0.453$으로 오히려 약 **9.7% 악화**합니다. 따라서 MoLE를 “항상 좋아지는 wrapper”라고 해석해서는 안 됩니다. 

---

# 2-1. 해결 문제, 수식, 모델 구조, 성능 향상과 한계

## 2-1-1. 문제 정의

논문은 입력을

$$
X\in\mathbb{R}^{c\times s}
$$

로 둡니다.

* $c$: channel 또는 변수 개수
* $s$: 과거 input timestamp 개수
* $p$: 미래 prediction timestamp 개수
* $X$: 과거 관측값
* $Y\in\mathbb{R}^{c\times p}$: 미래 예측
* $x_{\text{mark}}$: input timestamp 정보
* $y_{\text{mark}}$: prediction timestamp 정보

즉,

$$
f:
\mathbb{R}^{c\times s}
\rightarrow
\mathbb{R}^{c\times p}
$$

인 forecasting function을 학습합니다.

중요한 가정은 timestamp가 **regularly spaced**하다는 것입니다.

$$
t_{i+1}-t_i=\Delta t
$$

가 일정하다고 가정합니다. 첨부 PDF p.3입니다. 

> **Regularly spaced time series:** 모든 관측 사이의 시간 간격이 동일한 시계열입니다. 센서가 불규칙하게 누락되거나 측정 주기가 바뀌는 데이터에는 이 가정이 바로 적용되지 않습니다.

### Loss

논문은 MSE를 사용합니다.

```math
\mathrm{MSE}
=
\frac{1}{N}
\sum_{j=1}^{N}
\left(
Y_j-\hat{Y}_j
\right)^2
```

여기서

* $N$: 평가되는 전체 예측 원소 수
* $Y_j$: 논문 표기상 prediction
* $\hat Y_j$: corresponding ground truth

입니다.

주의할 점은 일반적인 머신러닝 표기에서는 $\hat Y$를 prediction으로 쓰는 경우가 많은데, 이 논문 설명에서는 반대 방향으로 서술되어 있습니다. 첨부 PDF p.6입니다.

---

## 2-1-2. DLinear expert

DLinear은 입력을 trend와 seasonal 부분으로 분해합니다.

개념적으로는

```math
X
=
X_{\mathrm{trend}}
+
X_{\mathrm{seasonal}}
```

이고,

```math
\hat Y
=
X_{\mathrm{trend}}W_{\mathrm{trend}}
+
X_{\mathrm{seasonal}}W_{\mathrm{seasonal}}
```

로 이해하는 것이 가장 자연스럽습니다.

* $X_{\mathrm{trend}}$: moving average를 통해 얻은 느린 변화
* $X_{\mathrm{seasonal}}$: $X-X_{\mathrm{trend}}$
* $W_{\mathrm{trend}}$: trend용 temporal linear mapping
* $W_{\mathrm{seasonal}}$: seasonal용 temporal linear mapping

> **중요한 원문 표기상의 주의점:** 논문 p.3에서는 $W_{\mathrm{trend}}X_{\mathrm{trend}}$ 형태로 적혀 있으면서 동시에 $X\in\mathbb R^{c\times s}$, $W\in\mathbb R^{s\times p}$라고 정의합니다. 이 차원을 그대로 적용하면 행렬곱이 맞지 않으므로, 실제 시간축 mapping의 의미는 $XW$ 또는 transpose convention으로 해석해야 합니다. 이는 제가 논문을 임의로 고친 것이 아니라 **원문 수식과 명시된 차원 사이의 표기 불일치**를 지적한 것입니다.

DLinear 자체는 AAAI 2023의 LTSF-Linear 계열에서 제안되었습니다. ([AAAI Publications][2])

---

## 2-1-3. RLinear expert

RLinear은 RevIN을 사용합니다.

```math
X_{\mathrm{norm}}
=
\text{RevIN}_{\mathrm{norm}}(X)
```

이후 temporal linear mapping을 적용하고,

```math
\hat Y
=
\text{RevIN}_{\mathrm{denorm}}
\left(
X_{\mathrm{norm}}W
\right)
```

처럼 해석할 수 있습니다.

> **RevIN(Reversible Instance Normalization):** 각 input sample의 통계량을 이용해 normalization한 후 prediction 단계에서 다시 원래 scale로 복원하는 방법입니다. train/test 사이의 level shift나 scale shift에 대한 민감도를 줄이는 것이 목적입니다.

이 특성 때문에 일반화 관점에서는 MoLE와 RLinear가 서로 다른 문제를 처리합니다.

$$
\boxed{
\text{RLinear: distribution level/scale shift}
}
$$

$$
\boxed{
\text{MoLE: 시간에 따른 prediction-rule specialization}
}
$$

으로 볼 수 있습니다. RLinear 연구는 RevIN과 Channel Independence의 중요성을 별도로 분석했습니다. ([arXiv][3])

---

## 2-1-4. RMLP expert

RMLP는 normalized signal에 residual MLP를 추가합니다.

```math
X_{\mathrm{norm}}
=
\text{RevIN}_{\mathrm{norm}}(X)
```

```math
H
=
X_{\mathrm{norm}}
+
\text{MLP}(X_{\mathrm{norm}})
```

```math
\hat Y
=
\text{RevIN}_{\mathrm{denorm}}
(HW).
```

즉 순수 선형성이 부족한 큰 데이터에서 작은 nonlinear correction을 추가합니다.

> **Residual connection:** 원래 입력 $X$에 학습된 변환 $F(X)$를 더해 $X+F(X)$로 만드는 구조입니다. 모델은 전체 관계를 처음부터 다시 학습하지 않고 “기본 관계에서 얼마나 수정할지”만 학습할 수 있습니다.

---

# 2-1-5. MoLE의 핵심 수식

MoLE에는 $n$개의 expert가 있습니다.

$i$번째 expert를

$$
H_i:
\mathbb R^{c\times s}
\rightarrow
\mathbb R^{c\times p}
$$

라고 정의합니다.

각 expert의 예측은

$$
Y_i=H_i(X),
\qquad i=1,\ldots,n
$$

입니다.

그다음 첫 timestamp의 embedding

$$
X_{\mathrm{mark}}\in\mathbb R^t
$$

를 router $M$에 넣습니다.

$$
W=M(X_{\mathrm{mark}})
$$

이며

$$
W\in\mathbb R^{c\times n}.
$$

즉 $W_{k,i}$는

> **channel $k$의 prediction에서 expert $i$를 얼마나 신뢰할 것인가**

를 나타냅니다.

논문은 각 channel에 대해 expert weight가 합쳐져 1이 되도록 구성한다고 설명합니다.

$$
\sum_{i=1}^{n}W_{k,i}=1.
$$

최종 mixture는

```math
\boxed{
Z
=
P
\left[
\sum_{i=1}^{n}
W_{:,i}\otimes Y_i
\right]
}
```

입니다. 첨부 PDF p.4, **Figure 2**입니다. 

여기서

* $Z\in\mathbb R^{c\times p}$: 최종 prediction
* $P$: optional postprocessing
* $W_{:,i}\in\mathbb R^c$: 모든 channel에 대한 expert $i$의 weight
* $Y_i\in\mathbb R^{c\times p}$: expert $i$의 prediction
* $\otimes$: channel-wise multiplication

입니다.

구체적으로

$$
a\in\mathbb R^c,\qquad
B\in\mathbb R^{c\times p}
$$

이면

$$
C=a\otimes B
$$

에 대해

$$
C_{kj}=a_kB_{kj}
$$

입니다.

즉 **한 expert에 전체 변수가 같은 weight를 갖는 것이 아니라 channel마다 다른 weight를 갖는 것**이 핵심입니다.

예를 들어 expert가 3개이고 한 channel에

```math
W_{k,:}
=
[0.7,0.2,0.1]
```

이라면 해당 channel prediction은

```math
Z_k
=
0.7Y_{1,k}
+
0.2Y_{2,k}
+
0.1Y_{3,k}.
```

다른 channel은

```math
W_{l,:}
=
[0.1,0.2,0.7]
```

일 수도 있습니다.

즉 동일한 timestamp에서도 서로 다른 물리 변수는 서로 다른 temporal expert를 사용할 수 있습니다.

---

# 2-1-6. 왜 timestamp만 router에 넣는가?

MoLE의 특징적인 설계는

$$
W=M(X)
$$

가 아니라

$$
\boxed{
W=M(X_{\mathrm{mark}})
}
$$

라는 것입니다.

즉 waveform 자체가 아니라 **입력 window 시작 시각**을 gating 정보로 사용합니다.

논문의 의도는 다음과 같습니다.

월요일 오전 8시,

$$
t=\text{Monday 08:00}
$$

와 토요일 오전 8시,

$$
t=\text{Saturday 08:00}
$$

에서 서로 다른 expert를 선택하도록 만드는 것입니다.

Router는 2-layer MLP이며 중간에 ReLU가 있습니다.

첨부 PDF p.6입니다.

> **MLP(Multi-Layer Perceptron):** 여러 fully-connected layer를 이어 붙인 작은 신경망입니다.
> **ReLU:** $\text{ReLU}(x)=\max(0,x)$인 비선형 활성화 함수입니다.

---

## Timestamp embedding

Appendix A, p.13에서는 hour, day-of-week, day-of-month, day-of-year 등의 값들을 $[-0.5,0.5]$로 선형 변환합니다.

요일의 경우

```math
e_{\mathrm{dow}}
=
\frac{\mathrm{index}}{6}-0.5
```

입니다.

따라서 Monday는

$$
-0.5
$$

Sunday는

$$
0.5
$$

가 됩니다.

여기에 중요한 잠재적 한계가 있습니다.

요일은 본질적으로 순환 변수이므로

$$
\text{Sunday}\rightarrow\text{Monday}
$$

가 가까워야 하지만 선형 embedding에서는

$$
0.5\leftrightarrow-0.5
$$

로 가장 멀리 떨어집니다.

따라서 일반화 개선 연구에서는

$$
\sin(2\pi t/T),\qquad
\cos(2\pi t/T)
$$

형태의 **cyclic embedding**이 더 자연스러운 후보입니다.

이 부분은 **논문의 주장이 아니라 제 후속 연구 제안**입니다.

---

# 2-1-7. Head dropout

과적합을 막기 위해 학습 중 일부 expert weight를 임의로 제거합니다.

논문 설명을 제가 수식으로 표현하면, mask

$$
m_i\sim\text{Bernoulli}(1-r)
$$

를 적용한 뒤

```math
\widetilde W_{k,i}
=
\frac{
m_iW_{k,i}
}{
\sum_jm_jW_{k,j}
}
```

처럼 남은 weight를 다시 합이 1이 되도록 normalize하는 구조입니다.

여기서

* $r$: head dropout rate
* $m_i=0$: 해당 expert를 이번 training iteration에서 제거
* $m_i=1$: 해당 expert를 유지

입니다.

**이 수식 자체는 논문에 그대로 적힌 것이 아니라 p.6의 verbal algorithm을 제가 수학적으로 다시 표현한 것입니다.**

---

# 2-1-8. 모델 전체 작동 순서

$$
X
\longrightarrow
H_1(X),H_2(X),\ldots,H_n(X)
$$

와 동시에

$$
t_0
\longrightarrow
\text{timestamp embedding}
\longrightarrow
M(\cdot)
\longrightarrow
W
$$

가 계산되고,

$$
\{Y_i\},W
\longrightarrow
\sum_iW_{:,i}\otimes Y_i
\longrightarrow
P(\cdot)
\longrightarrow
\hat Y
$$

가 됩니다.

즉 MoLE는 **dense mixture**에 가깝습니다. 모든 head의 prediction을 계산한 뒤 가중합하기 때문입니다.

이는 Time-MoE와 같은 이후의 **sparse MoE**가 일부 expert만 활성화하는 방식과 중요한 차이입니다. ICLR 2025의 Time-MoE는 sparse routing을 통해 높은 model capacity를 유지하면서 실제 활성 계산량을 제한하는 것을 핵심 목표로 합니다. ([ICLR Proceedings][4])

---

# 3. 핵심 주장에 대한 페이지/Figure/Table 지도

| 내용                                    | PDF 위치                        |
| ------------------------------------- | ----------------------------- |
| 문제 정의 및 regular sampling 가정           | **p.3, Sec. 3.1**             |
| DLinear/RLinear/RMLP 수식               | **p.3, Sec. 3.2, Fig.1**      |
| MoLE 전체 구조 및 핵심 mixture equation      | **p.4, Sec.4.1, Fig.2**       |
| Toy periodic regime                   | **p.5, Fig.3–4**              |
| 실험 split / metric / hyperparameter    | **p.5–6, Table 1**            |
| main forecasting results              | **p.7, Table 2**              |
| TimeIn/RandomIn/RandomOut             | **p.8, Table 3**              |
| input length 및 dropout generalization | **p.9, Fig.5**                |
| batch size/generalization gap         | **p.9, Fig.6**                |
| 저자 결론 및 open question                 | **p.9, Sec.6**                |
| timestamp embedding                   | **p.13, Appendix A**          |
| expert specialization 시각화             | **p.14, Fig.7**               |
| multi-seed robustness                 | **p.15, 23, Table 6**         |
| dropout 상세 ablation                   | **p.17–19, Fig.11**           |
| runtime/parameter count               | **p.20–21, Table 10, Fig.12** |

---

# 4. 저자가 보고한 내용과 제 해석의 분리

| 항목           | **저자가 직접 보고한 내용**                                                | **제 해석**                                               |
| ------------ | ---------------------------------------------------------------- | ------------------------------------------------------ |
| 연구 주제        | linear-centric LTSF에 MoE를 붙여 seasonal/non-stationary pattern을 처리 | linear model의 bias를 버리지 않고 conditional model로 확장한 접근   |
| expert 의미    | 서로 다른 temporal pattern에 specialize                               | 사실상 piecewise/soft-switching linear regression으로 해석 가능 |
| router       | starting timestamp embedding → channel-wise expert weights       | calendar regime이 예측식 변화의 proxy라는 강한 inductive bias     |
| 성능           | DLinear 32/44, RLinear 38/44, RMLP 33/44 개선                      | 평균적인 방향은 긍정적이나 효과 크기가 dataset-dependent                |
| timestamp 효과 | TimeIn이 25/44에서 최저 MSE                                           | 19/44에서는 다른 방식이 더 좋으므로 timestamp가 충분조건은 아님             |
| short input  | input이 짧을수록 TimeIn 이점 증가                                         | input 자체에서 regime을 식별할 정보가 부족할수록 calendar prior가 유용    |
| dropout      | long input에서 TimeIn overfitting 억제                               | router가 timestamp에 지나치게 의존할 수 있음을 오히려 보여 줌             |
| SOTA         | PatchTST와 비교 가능한 28개 중 19개 SOTA                                  | 당시 특정 benchmark/protocol에 국한된 주장                       |
| future work  | 언제 MoLE가 도움되는지와 필요한 head 수를 규명해야 함                               | 데이터-driven expert-number selection이 가장 중요한 미해결 문제      |

---

# 5. 통계적으로 취약한 부분과 비교 불가능한 수치

여기가 논문을 평가할 때 특히 중요합니다.

### ① Main Table 2는 사실상 single-seed 중심 결과입니다

Appendix D.2에서 저자들이 “single seed 2021 대신 2021/2022/2023 세 seed를 사용했다”고 설명하므로 main experiment는 기본적으로 single-seed 결과임을 알 수 있습니다.

Table 6에서는 세 seed 평균과 표준편차를 제시하지만 Weather2K는 제외됩니다.

Multi-seed 결과에서도 개선 비율은

* DLinear: 16/28 = **57.1%**
* RLinear: 21/28 = **75.0%**
* RMLP: 19/28 = **67.9%**

로 main Table 2보다 낮아집니다.

따라서 **“78% 이상 개선”이라는 headline 숫자를 seed-robust 확률로 해석하면 안 됩니다.** 

---

### ② 통계적 유의성 검정이 없습니다

평균과 standard deviation은 제시하지만,

$$
H_0:
E[L_{\mathrm{MoLE}}-L_{\mathrm{base}}]=0
$$

에 대한 paired test, bootstrap confidence interval, permutation test 등은 제공하지 않습니다.

따라서

$$
0.371\rightarrow0.368
$$

같은 작은 개선이 random-seed variability를 넘어선 통계적으로 확실한 개선인지 일괄적으로 판단할 수 없습니다.

---

### ③ “몇 개에서 이겼는가”는 효과 크기가 아닙니다

32/44라는 숫자는

$$
\mathbf 1(L_{\mathrm{MoLE}}<L_{\mathrm{base}})
$$

를 합한 **win count**입니다.

이는

```math
\Delta
=
\frac{L_{\mathrm{base}}-L_{\mathrm{MoLE}}}
{L_{\mathrm{base}}}
```

와 같은 상대적 개선 크기를 반영하지 않습니다.

0.1% 개선도 1승이고 20% 개선도 1승입니다.

---

### ④ PatchTST와의 비교는 완전한 apples-to-apples가 아닙니다

논문은 linear-centric models는 직접 재실험했지만 PatchTST는 **저자 보고값**을 사용했습니다.

또한 PatchTST/64는 look-back

$$
L=512
$$

이고 PatchTST/42는

$$
L=336
$$

인데 MoLE main experiment는 input length 336입니다.

따라서 모든 PatchTST column과 MoLE가 동일한 input-information budget을 썼다고 볼 수 없습니다. PatchTST의 핵심은 patching과 channel independence입니다. ([ML Anthology][5])

---

### ⑤ Weather2K 선택 대표성

Weather2K 전체 1865개 location 중 네 곳

$$
79,\ 114,\ 850,\ 1786
$$

만 임의로 선택했습니다.

이는 지역 다양성을 확보하려는 의도는 있지만,

```math
\text{Var}_{\text{station selection}}
(\text{performance})
```

은 측정하지 않았습니다.

따라서 “Weather2K에서 일반적으로 강하다”와 “이 네 station에서 강하다”는 구분해야 합니다.

---

### ⑥ Validation-selection bias 가능성

각 실험에서 여러 learning rate, head 수, dropout을 grid search하고 best validation loss를 선택합니다.

이는 정상적인 방법이지만 하나의 validation segment에 반복적으로 model selection을 수행하면

$$
\min_{\lambda\in\Lambda}
L_{\mathrm{val}}(\lambda)
$$

자체가 validation noise에 최적화될 수 있습니다.

Nested rolling validation이나 repeated temporal folds는 수행하지 않았습니다.

---

### ⑦ Toy dataset이 가설에 매우 우호적입니다

Toy dataset은 월–목은 frequency $f$, 금–일은 $2f$로 **명시적으로 calendar-aligned regime**을 만들었습니다.

따라서 timestamp router가 좋은 것은 구조적으로 예상 가능한 결과입니다.

이는 mechanism demonstration으로는 적절하지만 현실의 non-periodic regime shift까지 증명하지는 않습니다.

---

### ⑧ MSE만 사용합니다

MSE는 큰 오류에 제곱 패널티를 줍니다.

$$
L=(y-\hat y)^2
$$

이므로 outlier에 민감합니다.

MAE, MASE, probabilistic calibration, interval coverage 등의 결과는 main evaluation에 없습니다.

---

### ⑨ Runtime Table 10의 이례적 수치

Table 10에서 MoLE-DLinear의 6-head inference time이 **0.234 ms**로 적혀 있는데, 2–5 head가 약 0.828–0.844 ms이고 single-head가 0.727 ms입니다.

이 값은 주변 값과 매우 다릅니다.

**원문만으로 이것이 측정상의 특이 현상인지 오타인지 확인할 수 없으므로 결론을 만들어낼 수 없습니다.**

다만 저자 전체 결론은 평균적으로 training 약 26.45%, inference 약 13.34% overhead라는 것입니다.

---

# 6. 이 문서가 답하지 않는 질문

1. **몇 개의 expert가 최적인지 데이터만 보고 사전에 결정할 방법은 무엇인가?**
   저자 자신이 conclusion에서 명시적으로 open question으로 남깁니다.

2. **Regime이 calendar와 일치하지 않는다면 어떻게 되는가?**
   장비 열화, 시장 shock, 고장처럼 “화요일 오후”와 무관하게 발생하는 변화는 timestamp router만으로 탐지하기 어렵습니다.

3. **완전히 새로운 regime에서도 일반화하는가?**
   테스트에 존재하지만 training에는 전혀 없던 regime을 별도로 실험하지 않았습니다.

4. **Cross-dataset / zero-shot transfer가 가능한가?**
   검증하지 않았습니다.

5. **Irregular sampling이나 missing timestamp에서는 어떻게 동작하는가?**
   논문은 regular spacing을 가정합니다.

6. **Timestamp와 실제 process state를 함께 router에 넣으면 더 좋은가?**
   검증하지 않았습니다.

7. **Expert collapse가 발생할 수 있는가?**
   특정 expert 하나만 거의 항상 선택되는 문제에 대한 explicit load-balancing objective가 없습니다.

8. **Expert들의 차이가 정말 서로 다른 physical regime을 의미하는가?**
   Toy data에서는 확인하지만 실데이터에서는 causal/physical interpretation까지 입증하지 않습니다.

9. **시간이 지나면서 regime 자체가 변하는 concept drift에는 어떻게 대응하는가?**
   온라인 adaptation 실험이 없습니다.

10. **예측 불확실성은 어떻게 얻는가?**
    point MSE forecasting만 연구합니다.

> **Concept drift:** 입력과 출력 사이의 관계 $P(Y|X)$가 시간에 따라 변하는 현상입니다. 단순한 평균 이동보다 더 어려운 형태의 non-stationarity입니다.

---

# 7. 가장 중요한 그림 5개 해석

## Figure 2 — MoLE architecture, p.4

이 그림이 논문 전체의 핵심입니다.

기존에는

$$
X\rightarrow H(X)\rightarrow \hat Y
$$

였지만 MoLE에서는

$$
X\rightarrow
\{H_1(X),\ldots,H_n(X)\}
$$

로 여러 prediction rule을 만들고,

$$
t_0\rightarrow M(t_0)\rightarrow W
$$

가 이를 혼합합니다.

핵심 아이디어는 **representation을 엄청나게 복잡하게 만들지 않고 prediction rule의 다중성을 만드는 것**입니다.

제가 보기에 이것이 논문의 가장 중요한 설계 기여입니다.

---

## Figure 4 — Toy example prediction, p.5

Thursday→Friday에서 주파수가 바뀝니다.

single RLinear는 하나의 temporal mapping만 가지므로 transition 직후 prediction을 제대로 따라가지 못하지만, 2-head MoLE는 더 높은 주파수 패턴으로 전환합니다.

이 그림은

$$
\text{global compromise}
\quad\rightarrow\quad
\text{conditional specialization}
$$

이라는 MoLE의 직관을 가장 명확히 보여 줍니다.

다만 앞서 설명했듯 dataset 자체가 timestamp routing에 매우 유리하게 설계된 synthetic example입니다.

---

## Figure 5 — Input length와 generalization, p.9

이 그림이 **일반화 연구 관점에서 가장 중요합니다.**

input length가 매우 짧을 때는

$$
X_{t-s+1:t}
$$

자체가 regime을 판별하기에 정보가 부족합니다.

따라서

$$
P(\text{regime}|X,t)
$$

에서 $t$가 큰 추가 정보를 제공합니다.

반대로 input length가 길어지면

$$
P(\text{regime}|X)
$$

만으로도 waveform 내에서 변화 패턴을 파악할 수 있어 timestamp의 marginal utility가 감소합니다.

저자도 이 현상을 Effect 2로 해석합니다.

이 결과는 MoLE를 사용할 조건을 제시한다는 점에서 단순 benchmark 결과보다 더 가치가 있습니다.

---

## Figure 6 — Batch size와 generalization gap, p.9

중간 batch size에서 training loss는 낮은데 test loss가 상대적으로 나빠지는 형태를 관찰하고 저자들은 batch size 8을 선택합니다.

즉 이 그림은

$$
\text{training optimization quality}
\neq
\text{future generalization}
$$

임을 보여 줍니다.

다만 네 dataset 및 특정 architecture/settings에서 얻은 empirical observation이므로 batch size 8을 보편 법칙으로 해석할 수는 없습니다.

---

## Figure 7 — Expert specialization, p.14

Figure 7(a)는 시간에 따라 head mixing weight가 바뀌고, 7(c), 7(d)는 서로 다른 linear head가 서로 다른 frequency pattern을 학습했음을 보여 줍니다.

이것은 MoLE의 모델을 다음과 같이 해석하게 해 줍니다.

```math
f(X,t)
=
\sum_i
\pi_i(t)f_i(X)
```

즉 하나의 nonlinear black box라기보다

$$
\boxed{
\text{시간에 따라 부드럽게 전환되는 여러 개의 local linear model}
}
$$

입니다.

이 관점은 이후 regime-switching, dynamic coefficient model과 연결하기 매우 좋습니다.

---

# 8. 결론: 저자의 시사점과 후속 연구

저자들의 결론은 비교적 절제되어 있습니다.

MoLE가 linear-centric model의 단순성을 유지하면서 non-stationary/seasonal temporal patterns에 대한 적응성을 높일 수 있다고 주장하지만, **어떤 high-dimensional dataset에서 MoLE가 효과적인지**, 그리고 **필요한 head 수를 어떻게 사전에 예측할지**는 아직 해결되지 않았다고 명시합니다. 첨부 PDF p.9입니다. 

따라서 저자가 직접 제안한 미래 연구를 압축하면

$$
\boxed{
\text{dataset complexity}
\rightarrow
\text{expected benefit of MoLE}
}
$$

와

$$
\boxed{
\text{dataset structure}
\rightarrow
n_{\mathrm{experts}}
}
$$

사이의 관계를 찾는 것입니다.

---

# 8-1. 모델의 일반화 성능 향상 가능성

여기서는 “일반화”를 구분하는 것이 매우 중요합니다.

| 일반화 종류                  | MoLE에서 검증? | 평가                    |
| ----------------------- | ---------: | --------------------- |
| 같은 데이터셋의 미래 시간구간        |      **예** | 논문의 핵심 검증             |
| 다른 random seed          |    부분적으로 예 | Appendix Table 6      |
| 다른 prediction horizon   |          예 | 96/192/336/720        |
| 다른 input history length |    제한적으로 예 | Figure 5              |
| 새로운 channel             |        아니오 | 검증 없음                 |
| 새로운 station/domain      |        아니오 | zero-shot 실험 없음       |
| 강한 distribution shift   |  직접적으로 아니오 | calendar variation 중심 |
| 완전히 새로운 regime          |        아니오 | 검증 없음                 |
| irregular sampling      |        아니오 | regular-spacing 가정    |
| cross-dataset zero-shot |        아니오 | 검증 없음                 |

따라서 이 논문에서 “generalization이 좋아졌다”는 말을 가장 정확하게 표현하면

$$
\boxed{
\text{same-domain temporal hold-out generalization이 개선되는 경우가 많다}
}
$$

입니다.

**universal generalization**을 입증한 것은 아닙니다.

---

## 제가 제안하는 일반화 개선 방향

### A. Calendar-only router → state-aware router

현재는

$$
W=M(t)
$$

입니다.

이를

```math
W
=
M
\left(
t,\,
\phi(X)
\right)
```

로 바꿀 수 있습니다.

$\phi(X)$에는 미래정보 없이 과거 window에서 계산한

* mean
* variance
* slope
* spectral power
* recent change
* autocorrelation
* process regime indicator

등을 넣습니다.

그러면 단순 요일 변화뿐 아니라 **실제 state가 바뀌었을 때** expert routing이 바뀔 수 있습니다.

---

### B. 선형 timestamp embedding → cyclic embedding

```math
\phi_t
=
\left[
\sin\frac{2\pi t}{T},
\cos\frac{2\pi t}{T}
\right]
```

를 사용하면 주기 경계가 자연스럽습니다.

요일의 경우 Sunday와 Monday가 embedding 공간에서도 가깝게 됩니다.

---

### C. Independent experts → partial pooling

현재 각 head가 독립적이면 작은 데이터에서 parameter variance가 증가합니다.

따라서

```math
W_i
=
W_0+\Delta W_i
```

로 두고

$$
\lambda
\sum_i\|\Delta W_i\|_F^2
$$

를 penalty로 주는 것이 유용합니다.

이렇게 하면 모든 expert가 공통 구조 $W_0$를 공유하면서 필요한 부분만 다르게 학습합니다.

작은 데이터에서 특히 유리할 가능성이 높습니다.

---

### D. Smooth routing regularization

인접 timestamp에서 router가 불필요하게 급변하지 않도록

```math
\mathcal L_{\mathrm{smooth}}
=
\lambda
\sum_t
\|
W(t)-W(t-1)
\|_2^2
```

를 추가할 수 있습니다.

공정·교통·기상처럼 regime이 어느 정도 연속성을 갖는 문제에서는 강한 inductive bias가 될 수 있습니다.

---

### E. Dense MoLE → sparse top- $k$ MoLE

현재는 모든 head를 계산합니다.

이를

```math
\mathcal E_t
=
\text{TopK}(W_t)
```

로 두고

```math
\hat Y
=
\sum_{i\in\mathcal E_t}
W_iY_i
```

만 계산하면 expert 수를 늘리면서 inference cost를 제한할 수 있습니다.

이 방향은 이후 Time-MoE의 sparse-expert scaling 철학과 연결됩니다. ([ICLR Proceedings][4])

---

### F. Point prediction → probabilistic mixture

각 expert가

$$
p_i(Y|X)
$$

를 예측하도록 만들면

```math
p(Y|X,t)
=
\sum_i
\pi_i(t)p_i(Y|X)
```

가 됩니다.

서로 다른 regime의 불확실성을 자연스럽게 표현할 수 있습니다.

---

### G. 평가 방법 자체를 강화

일반화 연구라면 단일 6:2:2 또는 7:1:2 split보다

$$
T_1<T_2<T_3<T_4
$$

를 이용한 여러 rolling-origin validation을 권장합니다.

예를 들어

$$
\text{Train}_1\rightarrow\text{Valid}_1,
$$

$$
\text{Train}_2\rightarrow\text{Valid}_2,
$$

$$
\text{Train}_3\rightarrow\text{Valid}_3
$$

의 평균과 분산으로 hyperparameter를 선택하고 마지막 미래구간 test를 **한 번만** 사용하면 adaptive validation overfitting을 훨씬 줄일 수 있습니다.

---

# 8-2. 2020년 이후 최신 연구 비교

여기서 중요한 점은 **서로 다른 논문의 raw MSE를 그대로 줄 세우지 않는 것**입니다. dataset version, input length, normalization, tuning budget, metric aggregation 방식이 다르기 때문에 아래 비교는 **연구 방향과 일반화 전략 중심**입니다.

| 연구                                                                  |       연도 | 핵심 아이디어                                                | MoLE와의 관계                                              | 일반화 관점                                    |
| ------------------------------------------------------------------- | -------: | ------------------------------------------------------ | ------------------------------------------------------ | ----------------------------------------- |
| DLinear, *Are Transformers Effective for Time Series Forecasting?*  |     2023 | 단순 temporal linear mapping + decomposition             | MoLE backbone                                          | 저복잡도에 의한 일반화                              |
| PatchTST, *A Time Series Is Worth 64 Words*                         |     2023 | patching + channel independence                        | MoLE의 주요 Transformer 비교군                               | self-supervised transfer까지 연구             |
| RLinear/RMLP, *Revisiting LTSF: An Investigation on Linear Mapping* |     2023 | Linear + RevIN + CI                                    | 핵심 MoLE backbone                                       | distribution shift 대응                     |
| TimesNet                                                            |     2023 | 다중 주기를 2D variation으로 변환                               | 명시적 multi-period modeling                              | 다양한 TS task에 확장                           |
| **MoLE**                                                            | **2024** | timestamp-conditioned linear experts                   | 기준 논문                                                  | within-dataset regime specialization      |
| iTransformer                                                        |     2024 | 변수 자체를 token으로 만들어 cross-variate attention             | linear-centric 논쟁 이후 Transformer 재설계                   | variate generalization·long lookback 강조   |
| TimeMixer                                                           |     2024 | multi-scale decomposition + multiple future predictors | 여러 predictor 조합이라는 점에서 유사                              | 다양한 scale 변화 대응                           |
| Moirai                                                              |     2024 | 27B+ observations pretraining, universal forecaster    | 전혀 다른 scale의 generalization                            | cross-domain zero-shot                    |
| Chronos                                                             |     2024 | value tokenization + pretrained LM architecture        | foundation-model 방향                                    | unseen dataset zero-shot                  |
| FreqMoE                                                             |     2025 | frequency band별 expert + dynamic gating                | MoLE의 temporal expert idea를 frequency domain으로 확장하는 흐름 | 다양한 주기 특성 대응                              |
| Time-MoE                                                            |     2025 | sparse MoE foundation model, 최대 2.4B parameters        | MoE를 architecture scaling 수단으로 확장                      | large-scale zero-shot/general forecasting |
| M²FMoE                                                              |     2026 | Fourier/Wavelet multi-view + multi-resolution experts  | expert specialization을 extreme regime까지 확장             | 극단 이벤트 적응성 중점                             |

DLinear는 2023년 LTSF 연구에 “복잡한 모델이 반드시 좋은 것은 아니다”라는 중요한 기준점을 만들었습니다. ([AAAI Publications][2])

PatchTST는 patching과 channel-independence를 이용해 Transformer가 긴 history를 효율적으로 사용하는 방법을 제시했으며 self-supervised transfer도 보고했습니다. 이 부분은 MoLE가 다루지 않은 **cross-dataset representation generalization** 측면에서 중요합니다. ([ML Anthology][5])

RLinear/RMLP는 RevIN과 linear mapping을 결합하여 distribution shift에 대한 robustness를 강조했으며, MoLE가 직접 이 architecture를 expert로 사용합니다. ([arXiv][3])

TimesNet은 multi-periodicity 자체를 2차원 표현으로 변환한다는 점에서 MoLE와 문제 인식은 비슷하지만, expert switching이 아니라 representation 변환으로 해결합니다. ([OpenReview][6])

iTransformer는 Transformer 자체를 버리는 대신 variable을 token으로 재정의하여 multivariate correlation을 모델링하고 arbitrary lookback 및 variate generalization을 강조했습니다. ([ICLR Proceedings][7])

TimeMixer는 서로 다른 sampling scale에서 나타나는 micro/macro pattern을 분해하고 여러 predictor의 결과를 혼합한다는 점에서, “하나의 prediction rule로 모든 scale을 처리하지 않는다”는 MoLE의 철학과 연결됩니다. ([ICLR Proceedings][8])

Moirai는 LOTSA의 27B 이상 관측치를 이용해 cross-frequency, arbitrary-variate 문제를 해결하는 universal forecasting 방향으로 발전했으며 zero-shot에서도 full-shot model과 경쟁할 수 있음을 보고했습니다. 이는 MoLE의 in-dataset specialization보다 훨씬 넓은 일반화 문제를 다룹니다. ([Proceedings of Machine Learning Research][9])

Chronos 역시 다양한 domain과 synthetic Gaussian-process data를 이용한 사전학습으로 unseen dataset에 대한 zero-shot forecasting을 연구했습니다. ([ML Anthology][10])

2025년 AISTATS의 FreqMoE는 데이터를 frequency band별로 나누어 각 band에 expert를 할당하고 gating을 통해 결합합니다. 저자는 70개 평가 metric 중 51개에서 최고 결과를 보고했고 parameter 수를 50k 이하로 유지했다고 보고하지만, 이 수치는 MoLE의 44-setting win rate와 **직접 비교하면 안 됩니다**. 평가 프로토콜과 비교군이 다르기 때문입니다. ([Proceedings of Machine Learning Research][11])

ICLR 2025 Time-MoE는 sparse MoE를 billion-scale foundation model로 확장해 최대 2.4B parameter와 Time-300B pretraining corpus를 사용했습니다. 즉 MoLE가 “작은 linear expert의 specialization”을 연구했다면 Time-MoE는 “massive capacity를 sparse routing으로 확장하는 방법”을 연구합니다. ([ICLR Proceedings][4])

2026년 AAAI의 M²FMoE는 Fourier와 wavelet 양쪽의 frequency expert와 multi-resolution fusion을 이용해 **regular pattern이 아니라 extreme-event adaptation**까지 routing 대상을 넓혔습니다. 이는 2026년 현재 MoE forecasting 연구가 단순 calendar specialization을 넘어 frequency, resolution, extreme regime까지 확장되고 있음을 보여 주는 좋은 사례입니다. ([AAAI Publications][12])

---

# MoLE가 후속 연구에 미친 의미

MoLE의 역사적 의미를 과장하지 않고 표현하면 다음과 같습니다.

### 첫째, linear model과 MoE가 양립할 수 있음을 보여 주었습니다.

기존에는 MoE가 주로 거대한 language/vision model의 capacity scaling 기술로 인식되었습니다.

MoLE는

$$
\text{simple expert}
+
\text{adaptive router}
$$

만으로도 forecasting fidelity를 개선할 수 있다는 사례를 제시했습니다.

---

### 둘째, “모델 복잡성”보다 “전문가 분화 방식”이 중요하다는 방향을 강화했습니다.

2025년 FreqMoE에서는

$$
\text{time-conditioned expert}
\rightarrow
\text{frequency-conditioned expert}
$$

로 발전했고, ([Proceedings of Machine Learning Research][11])

2025년 Time-MoE에서는

$$
\text{dense small experts}
\rightarrow
\text{sparse large-scale experts}
$$

로 확장됐으며, ([ICLR Proceedings][4])

2026년 M²FMoE에서는

$$
\text{single temporal view}
\rightarrow
\text{multi-resolution + Fourier + Wavelet experts}
$$

로 확대됐습니다. ([AAAI Publications][12])

다만 이것을 모두 **MoLE의 직접적 인과 영향**이라고 단정할 근거는 충분하지 않습니다. 보다 정확하게는 MoLE가 이러한 **time-series-specific expert specialization 연구 흐름의 초기 대표 사례 중 하나**라고 보는 것이 타당합니다.

---

# 앞으로 연구할 때 가장 중요하게 고려할 점

제가 연구자 관점에서 우선순위를 매기면 핵심은 다음 관계입니다.

```math
\boxed{
\text{MoLE 2.0}
=
\text{shared linear prior}
+
\text{regime-aware experts}
+
\text{state-aware router}
+
\text{temporal regularization}
+
\text{strict time validation}
}
```

특히 작은 데이터라면 무조건 head 수를 늘리는 것보다

$$
W_i=W_0+\Delta W_i
$$

형태의 **partial pooling**을 사용하여 expert 사이에서 정보를 공유하는 것이 더 합리적입니다.

그리고 router는 단순 timestamp만 보게 하기보다

```math
g_t
=
g\left(
\underbrace{\text{calendar}_t}_{\text{known future}},
\underbrace{\text{past-state}(X_t)}_{\text{causal}},
\underbrace{\text{spectral state}(X_t)}_{\text{regime}}
\right)
```

처럼 구성하는 것이 일반화 가능성이 큽니다.

가장 중요한 것은 미래구간의 실제 값이나 미래에서 계산된 통계량을 router에 넣지 않는 것입니다. 그렇지 않으면 MoE가 성능을 높인 것이 아니라 **target/data leakage를 이용한 것**이 됩니다.

---

# 최종 평가

이 논문을 한 문장으로 평가하면,

$$
\boxed{
\text{“단순 선형 모델이 강하다”에서
“단순 선형 모델도 하나만 쓸 필요는 없다”로 넘어간 연구}
}
$$

라고 볼 수 있습니다.

논문의 가장 설득력 있는 부분은 단순히 Table 2의 평균 성능이 아니라 **Figure 5의 input-length 실험과 Figure 7의 expert specialization**입니다. 반대로 가장 조심해야 할 부분은 “78% 개선”과 “68% SOTA”라는 headline 숫자를 **통계적으로 보장된 보편적 일반화 성능**으로 확대 해석하는 것입니다. 이 논문이 입증한 것은 주로 동일 dataset 내 time-ordered future segment에 대한 성능 개선이며, 2024–2026의 Moirai, Chronos, Time-MoE 계열이 다루는 zero-shot/cross-domain generalization과는 구별해야 합니다. ([Proceedings of Machine Learning Research][9])

---

# 참고한 논문·사이트 전체

| 구분                | 참고자료 제목                                                                                                                                                                 |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **주 논문**          | **Ni et al., “Mixture-of-Linear-Experts for Long-term Time Series Forecasting,” AISTATS 2024, PMLR 238** ([Proceedings of Machine Learning Research][1])                |
| 공식 연구 소개          | **Microsoft Research — Mixture-of-Linear-Experts for Long-term Time Series Forecasting** ([Microsoft][13])                                                              |
| 공식 코드             | **RogerNi/MoLE — Official implementation of MoLE** ([GitHub][14])                                                                                                       |
| Linear baseline   | **Zeng et al., “Are Transformers Effective for Time Series Forecasting?”, AAAI 2023** ([AAAI Publications][2])                                                          |
| Linear/RevIN      | **Li et al., “Revisiting Long-term Time Series Forecasting: An Investigation on Linear Mapping,” 2023** ([arXiv][3])                                                    |
| Transformer       | **Nie et al., “A Time Series Is Worth 64 Words: Long-Term Forecasting with Transformers,” ICLR 2023** ([ML Anthology][5])                                               |
| Multi-period      | **Wu et al., “TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis,” ICLR 2023** ([OpenReview][6])                                                 |
| Transformer       | **Liu et al., “iTransformer: Inverted Transformers Are Effective for Time Series Forecasting,” ICLR 2024** ([ICLR Proceedings][7])                                      |
| Multiscale        | **Wang et al., “TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting,” ICLR 2024** ([ICLR Proceedings][8])                                             |
| Foundation model  | **Woo et al., “Unified Training of Universal Time Series Forecasting Transformers” (Moirai), ICML 2024** ([Proceedings of Machine Learning Research][9])                |
| Foundation model  | **Ansari et al., “Chronos: Learning the Language of Time Series,” TMLR 2024** ([ML Anthology][10])                                                                      |
| Frequency MoE     | **Liu, “FreqMoE: Enhancing Time Series Forecasting through Frequency Decomposition Mixture of Experts,” AISTATS 2025** ([Proceedings of Machine Learning Research][11]) |
| Sparse TS MoE     | **Shi et al., “Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts,” ICLR 2025** ([ICLR Proceedings][4])                                      |
| Extreme-event MoE | **Huang et al., “M²FMoE: Multi-Resolution Multi-View Frequency Mixture-of-Experts for Extreme-Adaptive Time Series Forecasting,” AAAI 2026** ([AAAI Publications][12])  |

특히 **2023~2026의 서로 다른 논문에서 보고된 MSE, “best N/M”, SOTA 비율은 동일한 실험 프로토콜이 아니므로 서로 직접적인 숫자 비교를 해서는 안 됩니다.** 위 비교는 연구 질문·architecture·일반화 방식의 변화에 초점을 맞춘 것입니다.

다음 단계로 가장 가치 있는 작업은 이 논문의 **MoLE 수식을 실제 PyTorch 수준으로 완전히 분해하고, leakage 없는 temporal Train/Validation/Test 환경에서 `Linear → Partial-Pooling MoLE → State-aware MoLE → Frequency-aware MoLE`를 단계적으로 구현·비교하는 것**입니다.

[1]: https://proceedings.mlr.press/v238/ni24a.html?utm_source=chatgpt.com "Mixture-of-Linear-Experts for Long-term Time Series Forecasting"
[2]: https://ojs.aaai.org/index.php/AAAI/article/view/26317?utm_source=chatgpt.com "Are Transformers Effective for Time Series Forecasting? | Proceedings of the AAAI Conference on Artificial Intelligence"
[3]: https://arxiv.org/abs/2305.10721?utm_source=chatgpt.com "Revisiting Long-term Time Series Forecasting: An Investigation on Linear Mapping"
[4]: https://proceedings.iclr.cc/paper_files/paper/2025/hash/558d48c1f08675daa636e09bfe94a89e-Abstract-Conference.html?utm_source=chatgpt.com "Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts"
[5]: https://mlanthology.org/iclr/2023/nie2023iclr-time/?utm_source=chatgpt.com "A Time Series Is Worth 64 Words: Long-Term Forecasting with Transformers | ML Anthology"
[6]: https://openreview.net/pdf/98c0a5bad8225b6d1baf5c74047c4d04bacfcfa1.pdf?utm_source=chatgpt.com "Published as a conference paper at ICLR 2023"
[7]: https://proceedings.iclr.cc/paper_files/paper/2024/hash/2ea18fdc667e0ef2ad82b2b4d65147ad-Abstract-Conference.html?utm_source=chatgpt.com "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting"
[8]: https://proceedings.iclr.cc/paper_files/paper/2024/hash/a7ac8a21e5a27e7ab31a5f42a0117bdb-Abstract-Conference.html?utm_source=chatgpt.com "TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting"
[9]: https://proceedings.mlr.press/v235/woo24a.html?utm_source=chatgpt.com "Unified Training of Universal Time Series Forecasting Transformers"
[10]: https://mlanthology.org/tmlr/2024/ansari2024tmlr-chronos/?utm_source=chatgpt.com "Chronos: Learning the Language of Time Series | ML Anthology"
[11]: https://proceedings.mlr.press/v258/liu25i.html?utm_source=chatgpt.com "FreqMoE: Enhancing Time Series Forecasting through Frequency Decomposition Mixture of Experts"
[12]: https://ojs.aaai.org/index.php/AAAI/article/view/39362?utm_source=chatgpt.com "M2FMoE: Multi-Resolution Multi-View Frequency Mixture-of-Experts for Extreme-Adaptive Time Series Forecasting | Proceedings of the AAAI Conference on Artificial Intelligence"
[13]: https://www.microsoft.com/en-us/research/publication/mixture-of-linear-experts-for-long-term-time-series-forecasting/?utm_source=chatgpt.com "Mixture-of-Linear-Experts for Long-term Time Series Forecasting - Microsoft Research"
[14]: https://github.com/RogerNi/MoLE?utm_source=chatgpt.com "GitHub - RogerNi/MoLE · GitHub"
