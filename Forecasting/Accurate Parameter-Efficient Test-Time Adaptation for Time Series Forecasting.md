# Accurate Parameter-Efficient Test-Time Adaptation for Time Series Forecasting

아래 검토는 **업로드된 arXiv v1 원문(2025-06-29, 11쪽)**을 1차 근거로 삼고, 웹에서 **공식 arXiv·AAAI·ICLR·PMLR·CVPR 자료와 2026년 후속 연구**를 교차 검증한 결과입니다. PETSA 원문이 직접 제시하지 않은 내용은 **“해석” 또는 “후속 연구 제안”**으로 명확히 구분했습니다. 원문 초록은 비정상성 때문에 고정된 사전학습 예측기의 성능이 저하되고, 기존 TTA의 비용 문제를 줄이기 위해 입력·출력의 작은 calibration module만 업데이트하는 PETSA를 제안한다고 설명합니다.  웹상의 arXiv 기록도 같은 내용을 확인합니다. ([arXiv][1])

# 1. Executive Summary — 10문장 이내

1. **PETSA(Parameter-Efficient Time-Series Adaptation)**는 시간에 따라 분포가 변하는 비정상 시계열에서 사전학습 예측기의 성능 저하를 줄이면서도, 테스트 시 전체 모델을 재학습하지 않도록 설계한 parameter-efficient TTA 방법입니다. 
2. 핵심 구조는 **source forecaster를 완전히 고정(frozen)**하고, 그 앞뒤에 **low-rank adapter + dynamic gating**으로 구성된 입력 및 출력 calibration module을 붙이는 것입니다. 
3. PETSA는 TAFAS의 시계열 TTA 설정을 계승하여 예측 후 순차적으로 관측되는 **부분 Ground Truth(PT)**와 더 늦게 확보되는 **전체 Ground Truth(T)**를 adaptation 신호로 사용합니다. 
4. adaptation loss는 이상치에 강한 **Huber loss**, 주기성을 보존하는 **frequency-domain loss**, 그리고 국소 평균·분산·상관구조를 맞추는 **patch-wise structural loss**를 결합합니다. 
5. 저자들은 ETTh1, ETTm1, ETTh2, ETTm2, Exchange, Weather와 여섯 종류의 forecasting backbone에서 PETSA가 TAFAS보다 더 많은 best-MSE 기록을 얻었다고 보고하며, 전체 count는 **PETSA 127회 대 TAFAS 88회**입니다. 
6. iTransformer–ETTh1 실험에서는 forecast window 720에서 PETSA가 TAFAS와 유사하거나 더 낮은 MSE를 유지하면서 adaptation parameter를 최대 **33.6배 적게** 사용했다고 보고합니다. 
7. 그러나 이 논문은 반복 실험의 표준편차·신뢰구간·통계검정 결과를 제시하지 않으며, “win count”는 효과 크기(effect size)가 아니라 순위 기반 집계라서 일반화 우수성을 확정하기에는 부족합니다.
8. 특히 arXiv v1 **Table 1의 일부 Avg 행은 바로 위 네 horizon 값의 산술평균과 일치하지 않는 내부 수치 오류가 확인되므로**, 해당 평균값을 그대로 근거로 사용해서는 안 됩니다. 
9. Appendix의 rank, gating initialization, loss ablation은 최적 설정이 horizon과 backbone에 따라 크게 달라지고 frequency loss가 오히려 성능을 악화시키는 경우도 보여주므로, PETSA의 일반화 성능은 **adapter 자체뿐 아니라 test-time hyperparameter selection에 강하게 의존할 가능성**이 있습니다. 
10. 2026년의 COSA와 Frequency-Aware Calibration 연구는 각각 더 단순한 output-space adaptation과 더 엄격한 matured-ground-truth-only protocol을 제안하고 있어, PETSA의 가장 중요한 후속 과제는 **“더 많은 adapter”가 아니라 “어떤 shift에서도 안전하게 작동하는 protocol·capacity·update rule의 자동 결정”**이라고 판단됩니다. ([ICLR Proceedings][2])

---

# 1-1. 연구의 목적과 필요성

## 연구가 출발한 문제

일반적인 시계열 예측은 과거 구간

$$
X=\{x_{t-L},x_{t-L+1},\ldots,x_{t-1}\}
$$

로부터 미래

$$
Y=\{x_t,x_{t+1},\ldots,x_{t+T-1}\}
$$

를 예측하도록

$$
Y=f_\theta(X)
$$

를 학습합니다. PETSA 원문도 이 문제를 동일하게 정의합니다. 

여기서

* $L$: 과거를 얼마만큼 보는지 나타내는 **look-back length**
* $T$: 앞으로 몇 시점을 예측하는지를 뜻하는 **forecast horizon**
* $x_t\in\mathbb R^V$: 시점 $t$에서 $V$개 변수의 관측값
* $f_\theta$: parameter $\theta$를 가진 사전학습 forecasting model입니다.

**용어 — Look-back window:** 현재 예측을 위해 모델에게 제공하는 과거 데이터 구간입니다.
**용어 — Forecast horizon:** 현재 시점부터 미래 몇 step까지 예측할지를 의미합니다.

문제는 실제 운용 데이터가 대개

$$
P_{\text{train}}(X,Y)\neq P_{\text{test},t}(X,Y)
$$

가 된다는 점입니다.

즉, 학습할 때의 데이터 분포와 실제 배포 후 시간 $t$에서의 데이터 분포가 달라집니다. 계절성 변화, 장비 상태 변화, 시장 regime 변화, 장기 trend 변화 같은 현상이 이에 해당합니다. PETSA는 이러한 **non-stationarity**가 pretrained forecaster의 정확도를 저하시킨다는 문제에서 출발합니다. 

**용어 — Non-stationarity, 비정상성:** 시계열의 평균, 분산, 주기, 변수 사이 관계 등이 시간에 따라 일정하지 않고 변하는 현상입니다.

**용어 — Distribution shift:** 학습 시점과 실제 사용 시점의 데이터 발생 확률분포가 달라지는 현상입니다.

---

## 왜 일반적인 fine-tuning으로 충분하지 않은가

배포된 모델을 매번 전체 fine-tuning하면 대략

```math
\theta_{t+1}
=
\theta_t-\eta\nabla_\theta\mathcal L_t
```

와 같이 전체 parameter $\theta$를 갱신해야 합니다.

큰 Transformer에서 $|\theta|$가 매우 크다면 gradient, optimizer state, activation까지 유지해야 하므로 inference 중 추가 비용이 커집니다.

PETSA의 생각은

$$
\theta_{\text{source}}\quad\text{는 고정},
$$

$$
\phi_t\quad\text{만 업데이트}
$$

하는 것입니다.

여기서 $\phi$는 작은 calibration adapter의 parameter입니다.

즉,

$$
|\phi|\ll|\theta_{\text{source}}|
$$

를 목표로 합니다.

**용어 — Parameter-efficient adaptation:** 전체 모델 parameter가 아니라 일부 작은 parameter 집합만 학습하여 계산량과 메모리를 줄이는 접근법입니다.

---

# 2. 핵심 주장과 근거

| 핵심 주장                                         | 저자가 제시한 근거                                                        | 위치                    | 검토                                            |
| --------------------------------------------- | ----------------------------------------------------------------- | --------------------- | --------------------------------------------- |
| PETSA는 parameter-efficient TSF-TTA이다          | source forecaster를 고정하고 입력·출력 low-rank calibration module만 update | p.2–3, Fig.2, Eq.(1)  | 구조적으로 타당                                      |
| 작은 adaptation capacity에서도 성능을 유지할 수 있다        | Huber + frequency + patch-wise structural loss                    | p.3–4, Eq.(2)–(6)     | 아이디어는 타당하지만 구성요소별 효과가 일관되지는 않음                |
| 다양한 backbone에 적용 가능하다                         | iTransformer, PatchTST, DLinear, OLS, FreTS, MICN 실험              | p.3–4, Table 1        | backbone 다양성은 장점                              |
| PETSA가 TAFAS보다 자주 최저 MSE를 기록한다                | PETSA 127 wins, TAFAS 88 wins                                     | p.3–4, Table 1, Fig.3 | **통계검정이 아닌 count**이며 tie도 포함                  |
| long horizon에서도 효율적이다                         | ETTh1/iTransformer horizon 720에서 최대 33.6× fewer parameters        | p.4, Fig.4            | parameter efficiency 근거는 강함                   |
| loss가 robustness와 구조 보존을 높인다                  | Huber, FFT spectral matching, patch statistics 사용                 | p.3–4                 | Appendix에서는 frequency term이 일부 설정에서 악화시킴      |
| rank가 adaptation capacity를 제어한다               | $r=8,16,32,64,128$ ablation                                       | p.9, Fig.11           | 지나친 rank 증가가 long horizon에서 악화 가능             |
| dynamic gating이 입력 조건에 맞춘 correction을 가능하게 한다 | gating initialization ablation                                    | p.10, Fig.12          | 상당한 hyperparameter sensitivity 존재             |
| PETSA는 범용적으로 더 효율적이다                          | OLS·Transformer 등에서 낮은 trainable-memory footprint                 | Fig.4–10              | **전체 GPU peak memory 또는 runtime 측정과 동일하지 않음** |

저자가 직접 설명한 architecture와 main result는 p.2–4에 명확히 제시되어 있습니다.  

---

# 2-1. 해결 문제, 제안 방법, 수식, 모델 구조

## 2-1-1. PETSA의 Test-Time Adaptation 설정

PETSA의 중요한 특징은 **미래 정답을 미리 보는 것이 아니라**, 시간이 흐른 뒤 실제 관측값이 순차적으로 도착하면 그것을 다음 adaptation에 사용하는 online 설정이라는 점입니다.

Fig.1의 순서는 다음과 같습니다.

```math
X_{t^*}
\rightarrow
\text{forecast}
\rightarrow
\hat Y_{t^*}
\rightarrow
\text{partial GT arrives}
\rightarrow
\text{adapt}
\rightarrow
\text{full GT arrives later}.
```

원문은 부분 실제값 PT와 이후 확보되는 full GT 모두를 lightweight calibration module update에 사용한다고 설명합니다. 

**용어 — Partial Ground Truth(PT):** 전체 예측 horizon이 끝나기 전에 이미 실제로 관측된 미래 구간입니다.

**중요한 해석:** 따라서 PETSA는 vision 분야 TENT처럼 **완전히 unlabeled test-time adaptation**인 것은 아닙니다. 시간 경과에 따라 실제 target을 획득하는 **delayed-supervision TTA**에 가깝습니다. 이 차이는 다른 TTA 연구와 수치를 비교할 때 매우 중요합니다.

---

# 2-1-2. 전체 구조 — Figure 2

PETSA는 다음 pipeline입니다.

$$
X_t
\rightarrow
C_{\text{in},\phi_{\text{in}}}
\rightarrow
X_t^{\text{cali}}
\rightarrow
f_{\theta}^{\text{frozen}}
\rightarrow
\hat Y_t
\rightarrow
C_{\text{out},\phi_{\text{out}}}
\rightarrow
\hat Y_t^{\text{cali}}
$$

여기서

* $C_{\text{in}}$: input calibration module
* $C_{\text{out}}$: output calibration module
* $\phi_{\text{in}},\phi_{\text{out}}$: test time에 학습되는 parameter
* $f_\theta$: source dataset에서 학습된 기존 forecaster
* $\theta$: PETSA adaptation 중 **고정되는 parameter**입니다.

Figure 2는 trainable module을 forecaster 앞뒤에 두고 source forecaster 자체는 frozen 상태로 유지하는 구조를 보여줍니다. 

**용어 — Calibration:** 원 모델을 다시 만드는 대신 입력이나 예측값에 작은 보정값을 추가하여 현재 데이터 분포에 맞추는 과정입니다.

---

# 2-1-3. Dynamic Low-Rank Calibration

논문의 Eq.(1)은 다음과 같습니다.

```math
\hat X_{t^*}^{\text{cali}}
=
X_{t^*}
+
\left[
\tanh\left(\alpha\odot X_{t^*}\right)W+b
\right]
```

```math
\hat Y_{t^*}^{\text{cali}}
=
\hat Y_{t^*}
+
\left[
\tanh\left(\alpha\odot\hat Y_{t^*}\right)W+b
\right].
```



핵심은

```math
\text{calibrated value}
=
\text{original value}
+
\text{learned residual correction}
```

입니다.

따라서 PETSA는 원 신호를 완전히 바꾸는 것이 아니라 **잔차 보정(residual correction)**을 학습합니다.

### 기호 설명

논문 표기 기준으로

$$
X_{t^*}\in\mathbb R^{B\times L\times V}
$$

입니다.

* $B$: batch size
* $L$: sequence length
* $V$: 변수 또는 channel 수
* $t^*$: 현재 adaptation이 일어나는 test-time index
* $\alpha\in\mathbb R^V$: 변수별 learnable gating parameter
* $\odot$: element-wise multiplication
* $\tanh(\cdot)$: 값을 $[-1,1]$로 제한하는 비선형 함수
* $W$: low-rank transformation
* $b$: learnable bias입니다.

### Dynamic gating

$$
g(X)=\tanh(\alpha\odot X)
$$

라고 쓰면 correction은 대략

$$
\Delta X=g(X)W+b
$$

이고 최종값은

$$
X^{\text{cali}}=X+\Delta X
$$

입니다.

즉 입력 상태가 바뀌면 $g(X)$도 바뀌므로 항상 동일한 correction을 주지 않습니다.

**용어 — Gating:** 보정량을 그대로 적용하지 않고 현재 입력에 따라 보정 강도 또는 방향을 조절하는 mechanism입니다.

---

# 2-1-4. Low-rank factorization

논문은

$$
W=A B_{\text{lr}}
$$

형태의 low-rank decomposition을 사용합니다.

원문에서는

$$
A\in\mathbb R^{L\times r},
\qquad
B\in\mathbb R^{r\times L\times V}
$$

라고 표기합니다. 

여기서

* $r$: low-rank dimension
* $r\ll L$로 만들면 전체 dense transformation보다 parameter를 줄일 수 있습니다.
* $A$: 저차원 latent subspace로 연결하는 factor
* $B_{\text{lr}}$: 다시 원 표현공간으로 변환하는 factor입니다.

제가 $B_{\text{lr}}$라고 바꿔 쓴 이유는 **논문 자체가 $B$를 batch size와 matrix factor 양쪽에 동시에 사용하여 표기가 혼동되기 때문**입니다.

**용어 — Low-rank:** 큰 행렬을 작은 두 행렬의 곱으로 근사해 parameter 수를 줄이는 방식입니다. LoRA가 대표적인 예입니다.

---

## 중요한 수식 표기상의 문제

원문 Eq.(1)은

$$
\hat Y_{t^*}\in\mathbb R^{B\times L\times V}
$$

라고 적습니다. 

하지만 앞선 forecasting 정의에서는 출력 horizon을 $T$라고 정의했으므로 일반적으로는

$$
\hat Y\in\mathbb R^{B\times T\times V}
$$

가 자연스럽습니다.

특히 실험은 horizon $96,192,336,720$을 사용하기 때문에 $T\neq L$인 경우가 존재할 수 있습니다.

따라서 **논문 v1의 Eq.(1)에는 output dimension 표기가 불명확합니다.** 이것이 구현 자체의 오류라는 증거는 없지만, 수학적 기술은 보다 명확하게 해야 합니다.

또한 $A\cdot B$가 3차원 tensor와 어떤 축을 따라 contraction되는지도 논문 본문에서 자세히 정의하지 않습니다.

---

# 2-1-5. Huber loss — Eq.(2)

PETSA는 MSE만 쓰지 않고 Huber loss를 사용합니다.

잔차를

$$
e=\hat Y^{\text{cali}}-Y
$$

라고 하면

```math
\mathcal L_{\text{Hub}}(e)
=
\begin{cases}
\dfrac{1}{2}e^2,
&
|e|<\delta,\\[6pt]
\delta\left(|e|-\dfrac{1}{2}\delta\right),
&
\text{otherwise}.
\end{cases}
```

논문에서는

$$
\delta=0.5
$$

로 고정합니다. 

작은 오차에서는

$$
\mathcal L\propto e^2
$$

이므로 MSE처럼 정밀하게 학습하고, 큰 오차에서는

$$
\mathcal L\propto |e|
$$

에 가까워지므로 이상치 하나가 gradient를 지나치게 크게 만들지 않습니다.

**용어 — Robust loss:** outlier가 존재해도 일부 극단적인 observation이 전체 학습을 지배하지 않도록 설계한 손실함수입니다.

---

# 2-1-6. Frequency-domain loss — Eq.(3)

논문은

```math
\mathcal L_{\text{freq}}
=
\left\|
\mathcal F(\hat Y^{\text{cali}})
-
\mathcal F(Y)
\right\|_1
```

을 사용합니다.

여기서

$$
\mathcal F(\cdot)=\text{FFT}(\cdot)
$$

입니다. 

* $\mathcal F$: Fast Fourier Transform
* $|\cdot|_1$: 각 frequency coefficient 차이의 절댓값 합
* $\hat Y^{\text{cali}}$: 보정 후 예측
* $Y$: 실제값입니다.

시간영역에서 두 신호가 조금 달라도 중요한 주기가 맞을 수 있고, 반대로 MSE가 낮아도 주기 구조가 왜곡될 수 있습니다.

따라서 PETSA는

$$
\text{time-domain fit}
+
\text{frequency-domain fit}
$$

을 동시에 요구합니다.

이 아이디어는 ICLR 2025의 **FreDF: Learning to Forecast in the Frequency Domain**과 연결됩니다. FreDF는 multi-step target의 시간적 correlation이 direct forecasting objective의 estimation bias를 유발할 수 있다는 관점에서 frequency-domain learning을 제안했습니다. ([ICLR Proceedings][3])

**용어 — FFT:** 시간에 따른 신호를 여러 주파수 성분의 조합으로 변환하는 알고리즘입니다.

---

# 2-1-7. Patch-wise Structural Loss — Eq.(4)

PETSA는

```math
\mathcal L_{\text{pw}}
=
\sum_{k\in
\{\text{corr},\text{mean},\text{var}\}}
\mathcal L_k
\left(
\hat Y^{\text{cali}},Y
\right)
```

를 사용합니다.

즉 patch 단위로

* $\text{corr}$: 상관구조
* $\text{mean}$: 평균
* $\text{var}$: 분산

을 맞추려고 합니다.

원문 Eq.(4)은 p.4에 제시되어 있습니다. 

이 아이디어의 원 연구인 **Patch-wise Structural Loss for Time Series Forecasting**은 point-wise MSE가 시계열 내부 구조를 충분히 반영하지 못한다고 보고, patch 수준 correlation·variance·mean을 비교합니다. ([Proceedings of Machine Learning Research][4])

**용어 — Patch:** 긴 시계열 전체를 한 번에 보는 대신 여러 작은 연속 구간으로 나눈 local segment입니다.

### 정확성상 중요한 한계

PETSA 논문 자체에는

$$
\mathcal L_{\text{corr}},
\quad
\mathcal L_{\text{mean}},
\quad
\mathcal L_{\text{var}}
$$

의 상세 수식이 다시 전개되어 있지 않습니다.

따라서 여기서 원 논문의 세부식을 제가 임의로 PETSA 공식인 것처럼 삽입하지 않겠습니다.

---

# 2-1-8. 최종 PETSA objective — Eq.(5), Eq.(6)

부분 GT에 대해서

```math
\mathcal L_{pt}
=
\mathcal L_{\text{Hub},pt}
+
\mathcal L_{\text{pw},pt}
+
\beta
\mathcal L_{\text{freq},pt},
```

전체 delayed GT에 대해서

```math
\mathcal L_T
=
\mathcal L_{\text{Hub},T}
+
\mathcal L_{\text{pw},T}
+
\beta
\mathcal L_{\text{freq},T}
```

를 사용하며,

```math
\boxed{
\mathcal L_{\text{PETSA}}
=
\mathcal L_T+\mathcal L_{pt}
}
```

입니다. 

* $pt$: partially observed target
* $T$: delayed full target을 뜻하는 subscript
* $\beta$: frequency loss의 상대적 가중치입니다.

여기서는 또 하나의 notation 충돌이 있습니다. 논문은 $T$를 forecast horizon에도 사용하고 full GT loss의 subscript에도 사용합니다. 문맥상 구분은 가능하지만 더 좋은 notation은 아닙니다.

---

# 2-1-9. PETSA가 parameter-efficient한 이유

전체 model $\theta$를 update한다면

$$
N_{\text{trainable}}=|\theta|
$$

입니다.

PETSA에서는

```math
N_{\text{trainable}}
=
|\phi_{\text{in}}|
+
|\phi_{\text{out}}|
\ll|\theta|.
```

논문의 input-module tensor 정의만 그대로 이용하면 한 adapter에서 대략

```math
N_{\text{adapter}}
=
Lr+rLV+LV+V
```

형태가 됩니다.

이는 **제가 논문에 주어진 tensor shape를 이용해 계산한 parameter-count 해석**이지, 저자가 명시적으로 제시한 공식은 아닙니다.

핵심은 dense transformation이 요구하는 큰 자유도를

$$
r\ll L
$$

인 subspace로 제한한다는 점입니다.

이 제한은 단순히 메모리 절감만 의미하지 않습니다.

작은 $r$은 adaptation 함수공간 자체를 제한하므로

$$
\text{variance 감소}
\quad\leftrightarrow\quad
\text{adaptation capacity 감소}
$$

라는 일종의 regularization 역할도 할 수 있습니다.

이 점은 **일반화 성능 측면에서 PETSA가 흥미로운 이유**입니다.

---

# 3. 주장별 Page / Figure / Table 위치

| 내용                                          | 위치                      |
| ------------------------------------------- | ----------------------- |
| 문제 설정 및 contribution                        | p.1, Introduction       |
| PT와 delayed full GT를 사용하는 online adaptation | **p.1, Figure 1**       |
| PETSA input/output adapter architecture     | **p.2, Figure 2**       |
| Dynamic calibration equation                | **p.3, Eq.(1)**         |
| Huber loss                                  | **p.3, Eq.(2)**         |
| Frequency-domain loss                       | **p.3, Eq.(3)**         |
| Patch-wise structural loss                  | **p.4, Eq.(4)**         |
| Partial/full adaptation losses              | **p.4, Eq.(5), Eq.(6)** |
| 6 datasets × 6 backbones 성능                 | **p.3, Table 1**        |
| PETSA vs TAFAS win counts                   | **p.4, Figure 3**       |
| iTransformer parameter/MSE efficiency       | **p.4, Figure 4**       |
| OLS parameter efficiency                    | p.7–9, Figures 5–10     |
| Low-rank rank ablation                      | **p.9, Figure 11**      |
| Gating initialization ablation              | **p.10, Figure 12**     |
| Loss/ $\beta$ ablation                       | **p.11, Figure 13**     |

---

# 4. 저자가 직접 보고한 결과와 제 해석의 분리

| 항목           | **저자가 보고한 내용**                                                 | **제 해석**                                                                       |
| ------------ | -------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| 연구주제         | parameter-efficient TTA for TSF                                | PEFT와 online TSF adaptation을 연결한 연구                                            |
| architecture | input/output low-rank gated calibrator + frozen forecaster     | 본체를 건드리지 않아 catastrophic drift 위험과 update cost를 제한할 수 있음                       |
| supervision  | partial GT + delayed full GT                                   | 완전한 unlabeled TTA보다 supervised online recalibration에 가까움                       |
| loss         | Huber + frequency + patch structural                           | point error, spectral structure, local statistics를 동시에 제약                      |
| Table 1      | PETSA 127 wins, TAFAS 88                                       | win count는 우세 방향을 보여주지만 효과 크기나 유의성을 의미하지 않음                                    |
| Fig.4        | horizon 720에서 최대 33.6× fewer parameters                        | parameter storage efficiency는 강한 장점이나 end-to-end inference efficiency와 동일하지 않음 |
| Fig.11       | rank에 따른 performance 비교                                        | capacity가 커질수록 항상 좋아지지는 않음                                                     |
| Fig.12       | gating init에 따른 MSE 비교                                         | hyperparameter가 horizon마다 달라 generalization 시 oracle tuning 위험이 있음             |
| Fig.13       | total loss가 항상 MSE보다 낫지는 않으며 $\beta$ tuning 필요                 | frequency loss는 universal inductive bias가 아님                                   |
| 결론           | 여러 backbone에서 fewer parameters로 competitive/better forecasting | “architecture generality”는 어느 정도 입증되었지만 “unseen shift generality”는 아직 입증되지 않음  |

---

# 5. 성능 결과를 어떻게 읽어야 하는가

저자들은 Table 1에서 PETSA가 전체적으로 **127 best-value counts**, TAFAS가 88이라고 보고합니다. 

그러나 PETSA가 **모든 개별 실험에서 TAFAS보다 낮은 MSE**인 것은 아닙니다.

즉 저자의 결론은 정확히는

> 여러 dataset/backbone/horizon에서 전반적으로 competitive 또는 better

에 가깝지,

> 모든 조건에서 strict dominance

가 아닙니다.

이 차이는 매우 중요합니다.

---

# 5-1. Table 1의 수치 검산에서 발견되는 문제

여기는 논문을 읽을 때 특히 주의해야 합니다.

arXiv v1 p.3의 Table 1에서 예를 들어 ETTm1 / iTransformer의 네 horizon 값은

$$
0.439,\quad0.508,\quad0.613,\quad0.485
$$

입니다. 

따라서 직접 평균하면

```math
\frac{
0.439+0.508+0.613+0.485
}{4}
=
0.51125
```

입니다.

그런데 표의 ETTm1 Avg는 iTransformer의 첫 값으로 **0.257**을 제시합니다. 

같은 방법으로 PDF에 인쇄된 값들을 검산하면 다음과 같습니다.

| Dataset / model         | 네 horizon의 단순평균 | Table의 Avg 첫 값 |
| ----------------------- | --------------: | -------------: |
| ETTh1 / iTransformer    |       $0.55625$ |        $0.557$ |
| ETTm1 / iTransformer    |   **$0.51125$** |    **$0.257$** |
| ETTh2 / iTransformer    |   **$0.31700$** |    **$0.220$** |
| ETTm2 / iTransformer    |   **$0.21425$** |    **$0.343$** |
| Exchange / iTransformer |   **$0.26075$** |    **$0.355$** |
| Weather / iTransformer  |       $0.25800$ |        $0.258$ |

ETTh1의 $0.55625$ 대 $0.557$ 정도는 원래 unrounded 값 때문에 생길 수 있는 정상적인 차이입니다.

반면 ETTm1 등의 차이는 rounding으로 설명할 수 없습니다.

### 판단

**arXiv v1의 Table 1에는 Avg row의 copy/alignment 또는 table-construction 오류가 있을 가능성이 매우 높습니다.**

다만 저자의 원 raw experiment 결과를 가지고 있지 않으므로 **어떤 값이 “진짜 최종값”인지는 단정할 수 없습니다.**

따라서 이 논문을 인용할 때는

* 개별 horizon cell은 별도로 확인하고,
* 문제가 있는 Avg row를 그대로 사용하지 않으며,
* 가능하면 저자 repository의 raw result로 재현해야 합니다.

---

# 5-2. 통계적으로 취약한 부분

## ① 반복 실험의 분산이 없다

Table 1은

$$
\text{MSE}=0.xxx
$$

형태의 point estimate만 보여줍니다.

그러나

$$
\bar m\pm s
$$

또는

$$
95\%\ \text{CI}
$$

가 없습니다.

따라서

$$
0.432 \text{ vs. }0.435
$$

같은 차이가 실제로 재현 가능한 차이인지 seed variation인지 판단하기 어렵습니다.

---

## ② 통계적 유의성 검정이 없다

예를 들어 여러 dataset/horizon에 대해 paired comparison을 한다면 최소한

* paired bootstrap confidence interval,
* Wilcoxon signed-rank test,
* hierarchical mixed-effects analysis

등을 고려할 수 있습니다.

논문에는 이런 검정이 제시되지 않습니다.

---

## ③ “127 wins vs 88 wins”는 독립적인 215회 승부가 아니다

Table의 count에는 **동률도 양쪽에 win으로 들어갈 수 있습니다.**

따라서

$$
127+88
$$

을 전체 독립 비교 횟수로 해석하면 안 됩니다.

또한 MSE가

$$
0.4300
\quad\text{vs.}\quad
0.4304
$$

였다가 소수점 세 자리에서 모두 $0.430$으로 표시되면 tie처럼 보일 수 있습니다.

즉 win count는

$$
\text{효과의 방향}
$$

은 어느 정도 보여주지만

$$
\text{효과의 크기}
$$

나

$$
\text{통계적 확실성}
$$

을 보여주지 않습니다.

---

## ④ hyperparameter 선택 protocol이 중요하지만 충분히 명확하지 않다

PETSA에는 최소한

$$
r,\quad
\alpha_{\text{init}},\quad
\beta
$$

라는 중요한 adaptation hyperparameter가 있습니다.

Figure 11–13을 보면 이들의 최적값은 horizon에 따라 달라집니다.   

따라서 진짜 deployment에서는

> 미래 test target을 모르는 상태에서 $r,\alpha,\beta$를 어떻게 정할 것인가?

가 매우 중요한 문제입니다.

ICML 2023의 **On Pitfalls of Test-Time Adaptation** 역시 TTA의 가장 큰 함정 중 하나가 test-time model/hyperparameter selection이라고 지적합니다. ([Proceedings of Machine Learning Research][5])

---

## ⑤ parameter memory와 실제 inference cost는 같은 개념이 아니다

Fig.4–10은 adaptation parameter와 MB annotation을 보여줍니다.

예를 들어 ETTh1 OLS horizon 720에서는 PETSA 0.21 MB, TAFAS 3.70 MB로 보고됩니다. 

하지만 deployment cost는

$$
\text{parameter storage}
\neq
\text{peak GPU memory}
\neq
\text{latency}
\neq
\text{FLOPs}
\neq
\text{energy}
$$

입니다.

PETSA v1은 이들을 모두 동일 수준으로 측정하지 않았습니다.

따라서 **“33.6× fewer trainable parameters”를 “33.6× faster”라고 바꾸어 말하면 잘못입니다.**

---

## ⑥ conventional unlabeled TTA와 직접 비교할 수 없다

TENT는 test data와 model 자체만을 이용하고 entropy minimization으로 adaptation합니다. ([arXiv][6])

반면 PETSA는 실제 target이 시간이 지나면서 부분·전체로 관측된다는 TSF 특성을 이용합니다.

따라서

$$
\text{TENT 성능}
\quad\text{vs.}\quad
\text{PETSA 성능}
$$

을 동일한 “TTA”라는 이유만으로 직접 숫자 비교하는 것은 불가능합니다.

**비교 불가능 사유:** supervision availability가 다릅니다.

---

# 6. 문서가 답하지 않는 질문

1. PETSA가 명시적으로 정의된 **abrupt shift, gradual drift, seasonal shift, variance shift, cross-variable dependency shift** 각각에서 얼마나 강한가?
2. test-time에 실제 GT가 늦게 도착하지 않거나 아예 unavailable하면 PETSA는 어떻게 작동하는가?
3. $r$, $\beta$, gating initialization을 **target test label 없이** 어떻게 선택해야 하는가?
4. adapter가 오랫동안 누적 update될 때 catastrophic forgetting이나 parameter drift가 발생하는가?
5. 분포가 원래 source regime으로 돌아왔을 때 PETSA가 자동으로 복원되는가?
6. calibration module 두 개가 동시에 학습되었을 때 input correction과 output correction의 역할을 식별할 수 있는가?
7. parameter MB 외에 wall-clock latency, FLOPs, peak GPU RAM, energy consumption은 얼마인가?
8. probabilistic forecasting이나 prediction interval에서도 adaptation이 calibration을 개선하는가?
9. missing values, irregular timestamps, variable set 변화에도 generalize하는가?
10. Table 1 Avg row의 수치 불일치는 단순 조판 오류인지 raw result 생성 pipeline의 오류인지?

이 질문들은 특히 **실제 산업 공정이나 장기간 online deployment**에서 중요합니다.

---

# 7. 가장 중요한 그림 5개 해석

## Figure 1 — PETSA가 언제 학습하는지를 이해하는 핵심 그림

Figure 1은 단순한 데이터 흐름 그림이 아니라 **PETSA의 supervision assumption 전체를 정의합니다.**

예측 직후에는 full target을 모르지만 시간이 흐르면 일부 실제값이 관측됩니다.

이를

$$
Y_{t:t+p}
$$

라고 하면 PETSA는 그것으로 먼저 adaptation하고, forecast horizon이 끝나면

$$
Y_{t:t+T}
$$

전체를 추가로 사용할 수 있습니다. 

### 해석

PETSA의 성공은 “unlabeled data에서 스스로 적응한다”기보다

> **시계열의 시간적 진행이 자연스럽게 delayed label을 제공한다**

는 사실을 적극적으로 활용한 결과입니다.

따라서 실제 deployment에서 label latency가 PETSA의 가정과 다르면 성능도 달라질 가능성이 큽니다.

---

# Figure 2 — PETSA의 가장 중요한 구조적 아이디어

Figure 2의 핵심은

$$
\boxed{
\text{adapt the boundary, freeze the core}
}
$$

라고 요약할 수 있습니다.

즉

$$
X
\rightarrow
\boxed{\text{Input Adapter}}
\rightarrow
\boxed{\text{Frozen Forecaster}}
\rightarrow
\boxed{\text{Output Adapter}}
\rightarrow
\hat Y
$$

입니다. 

### 해석

이 구조는 pretrained model의 기존 지식을 크게 훼손하지 않으면서 distribution mismatch를 앞뒤에서 보정하려는 설계입니다.

일반화 관점에서는 일종의 **capacity control**입니다.

전체 model을 update하는 것보다 자유도가 작기 때문에 적은 test data에서 과적합을 줄일 가능성이 있습니다.

하지만 두 adapter가 모두 자유롭게 움직이면 서로의 correction을 상쇄하거나 중복할 수도 있다는 점은 논문이 분석하지 않습니다.

---

# Figure 4 — 정확도 대 adaptation cost trade-off

ETTh1+iTransformer에서 PETSA는 여러 forecast horizon에서 TAFAS와 유사하거나 더 나은 MSE를 얻으면서 trainable parameter를 크게 줄입니다.

특히 horizon 720에서 최대 **33.6× fewer parameters**라고 저자가 강조합니다. 

### 해석

이 그림이 PETSA 논문의 가장 강한 증거입니다.

PETSA가 반드시 absolute best forecaster를 만드는 것이 아니라

$$
\frac{
\text{adaptation benefit}
}{
\text{additional trainable capacity}
}
$$

를 높였다는 것이 핵심입니다.

즉 **accuracy-efficiency Pareto frontier**를 개선하려는 연구입니다.

**용어 — Pareto frontier:** 한 목표를 더 개선하면 다른 목표를 희생해야 하는 상황에서, 더 이상 동시에 개선하기 어려운 최적 trade-off 경계를 말합니다.

---

# Figure 11 — Rank가 크다고 항상 좋은 것은 아니다

Figure 11은 ETTh1–OLS에서

$$
r\in\{8,16,32,64,128\}
$$

을 비교합니다. 

96, 192, 336 horizon에서는 rank 증가가 대체로 MSE를 낮추는 방향이지만, **720 horizon에서는 대략 $r=32$ 부근에서 가장 좋고 더 큰 rank에서 다시 악화되는 U형 경향**이 보입니다.

### 해석

이는 중요한 일반화 결과입니다.

$$
r\uparrow
\Rightarrow
\text{capacity}\uparrow
$$

이지만

$$
r\uparrow
\not\Rightarrow
\text{generalization}\uparrow
$$

입니다.

long-horizon에서는 제한된 delayed labels에 비해 adapter 자유도가 너무 커지면 transient pattern에 과적합할 가능성이 있습니다.

따라서 PETSA의 low-rank는 단순 압축뿐 아니라 **regularization hyperparameter**로 이해하는 것이 더 정확합니다.

---

# Figure 13 — Frequency loss가 항상 도움이 되는 것은 아니다

Figure 13은

$$
\beta\in[0,1]
$$

에 따른 loss ablation입니다. 

저자도 ETTh1 OLS에서는

$$
\beta=0
$$

일 때 가장 좋은 결과가 나왔다고 명시합니다.

즉 그 설정에서는

$$
\mathcal L_{\text{freq}}
$$

를 추가하지 않는 편이 더 좋았습니다. 반대로 FreTS에서는 다수 조건에서 $\beta=0.1$이 유리했다고 설명합니다. 

### 해석

이 결과는 PETSA의 specialized loss가 **universally optimal하지 않다**는 가장 직접적인 증거입니다.

주기가 강한 데이터에는 frequency constraint가 유리하지만, spectral structure가 불안정하거나 regime-dependent한 데이터에서는 오히려 잘못된 inductive bias가 될 수 있습니다.

**용어 — Inductive bias:** 모델이 어떤 형태의 패턴이 더 그럴듯하다고 미리 가정하도록 만드는 설계상의 편향입니다.

---

# 8. 논문의 결론과 저자 시사점

저자 결론의 핵심은 다음 세 가지입니다.

* 입력과 출력을 작은 gated calibration module로 보정하면 core forecaster를 고정한 채 TTA가 가능하다.
* Huber + frequency + patch-wise structural objective가 제한된 adapter capacity를 보완한다.
* 여러 forecasting architecture에서 기존 TTA보다 적은 trainable parameter로 competitive 또는 better MSE를 얻을 수 있다. 

### 중요한 사실

**arXiv v1 Conclusion에는 구체적인 “future work 계획”이 별도로 제시되어 있지 않습니다.**

따라서 아래 후속 연구 방향은 **저자가 명시한 계획이 아니라 제가 논문의 한계와 2025–2026 후속 연구를 바탕으로 도출한 연구 제안**입니다.

---

# 8-1. 모델의 일반화 성능 향상 가능성

PETSA에서 generalization을 높일 수 있는 가장 중요한 방향은 **더 큰 adapter를 만드는 것이 아닙니다.**

## A. Rank를 고정하지 말고 shift severity에 맞춰 동적으로 결정

현재는

$$
r=\text{fixed hyperparameter}
$$

입니다.

이를

```math
r_t
=
g(s_t)
```

처럼 만들 수 있습니다.

여기서 $s_t$는 distribution shift score입니다.

예를 들면

```math
s_t
=
\left\|
\mu_t-\mu_{\text{ref}}
\right\|_2
+
\lambda
\left\|
\Sigma_t-\Sigma_{\text{ref}}
\right\|_F
```

로 두고,

* shift가 작으면 작은 $r_t$,
* shift가 크면 큰 $r_t$

를 사용하는 방식입니다.

이렇게 하면 stationary 구간에서는 unnecessary adaptation을 줄일 수 있습니다.

---

## B. Frequency loss weight $\beta$를 자동화

Figure 13 때문에 고정 $\beta$는 위험합니다.

대신

```math
\beta_t
=
h\left(
\text{periodicity strength}_t
\right)
```

로 구성하는 것이 자연스럽습니다.

예를 들어 dominant spectral energy ratio를

```math
q_t
=
\frac{
\sum_{f\in\mathcal F_{\text{dom}}}|X_t(f)|^2
}{
\sum_f|X_t(f)|^2
}
```

라 정의하고

$$
\beta_t=\sigma(aq_t+c)
$$

로 만들 수 있습니다.

주기성이 명확한 구간에서만 frequency loss를 강하게 적용하는 것입니다.

---

## C. Input adapter와 output adapter를 항상 둘 다 사용하지 않기

2026년 ICLR의 COSA는 PETSA·TAFAS와 같은 dual input/output adapter가 correction의 해석을 어렵게 하고 비용을 늘릴 수 있다는 문제의식에서 **single output-space adapter**를 제안합니다. COSA는 frozen prediction에 context-conditioned residual만 더하는 간단한 구조이며, leakage-free protocol에서 기존 TTA 대비 개선을 보고합니다. ([ICLR Proceedings][2])

PETSA 후속 모델에서는

$$
z_t\in\{0,1,2\}
$$

를 두어

* $z_t=0$: no adaptation
* $z_t=1$: output-only
* $z_t=2$: input + output

처럼 shift에 따라 adaptation path 자체를 선택할 수 있습니다.

---

## D. “성능 향상”보다 먼저 negative adaptation을 방지

TTA에서 가장 위험한 것은

$$
\mathcal L_{\text{after}} > \mathcal L_{\text{before}}
$$

인 **negative adaptation**입니다.

따라서 실제 generalized PETSA는 update 전에

```math
\Delta_t
=
\text{estimated benefit of adaptation}
```

을 평가해

$$
\Delta_t > 0
$$

일 때만 parameter update를 허용하는 방향이 좋습니다.

이 관점은 2026년 금융 시계열 TTA 연구가 보여준 결과와도 맞습니다. 해당 연구에서는 금융 데이터에서 aggressive adaptation이 오히려 성능을 악화시킬 수 있고, 단순 normalization-statistics update가 더 안전한 경우가 있다고 보고합니다. ([arXiv][7])

---

## E. Continual adaptation의 forgetting 제어

vision TTA의 CoTTA는 오래 adaptation할수록 pseudo-label error accumulation과 catastrophic forgetting 문제가 생길 수 있음을 지적하고 일부 source weight를 stochastic하게 복원합니다. ([Open Access CVF][8])

PETSA도 장기 운용에서는 calibration parameter가 계속 drift할 수 있습니다.

따라서

```math
\mathcal L_{\text{total}}
=
\mathcal L_{\text{PETSA}}
+
\lambda
\|\phi_t-\phi_0\|_2^2
```

같은 anchor regularization을 고려할 수 있습니다.

* $\phi_0$: 초기 adapter parameter
* $\phi_t$: 현재 adapter
* $\lambda$: 원래 상태 유지 강도입니다.

---

# 8-2. 2020년 이후 관련 최신 연구 비교

## 연구 흐름

| 연도        | 연구                                            | 핵심 아이디어                                                                            | PETSA와의 관계                         | 직접 수치 비교 가능?                                          |
| --------- | --------------------------------------------- | ---------------------------------------------------------------------------------- | ---------------------------------- | ----------------------------------------------------- |
| 2020/2021 | **TENT**                                      | unlabeled test batch에서 entropy minimization, normalization affine parameter update | TTA의 대표적 출발점                       | **아니오** — vision classification / label assumption 다름 |
| 2022      | **LAME**                                      | parameter를 바꾸지 않고 output 자체를 보수적으로 adaptation                                      | PETSA보다 더 극단적 parameter efficiency | 아니오                                                   |
| 2022      | **EATA**                                      | selective update + anti-forgetting regularization                                  | PETSA가 다루지 않은 forgetting 문제 제기     | 아니오                                                   |
| 2022      | **CoTTA**                                     | continual shift, teacher averaging + stochastic source restoration                 | long-term PETSA에 참고 가능             | 아니오                                                   |
| 2023      | **TTAB / On Pitfalls of TTA**                 | TTA hyperparameter selection과 protocol inconsistency 지적                            | PETSA 평가 설계 검증에 매우 중요              | 아니오                                                   |
| 2024      | **LoRA for Time-Series Foundation Models**    | TS foundation model을 low-rank fine-tuning                                          | PETSA low-rank 설계의 TS precedent    | 아니오 — offline fine-tuning                             |
| 2024      | **Channel-Aware LoRA**                        | CI/CD trade-off를 low-rank adaptation으로 조정                                          | multivariate PETSA 확장에 참고          | 아니오                                                   |
| 2025      | **TAFAS**                                     | PT/POGT와 gated calibration을 이용한 TSF-TTA                                            | PETSA의 직접 기반                       | **부분적으로 가능**                                          |
| 2025      | **FreDF**                                     | frequency-domain objective                                                         | PETSA frequency loss의 이론적 기반       | loss 연구이지 TTA 자체는 아님                                  |
| 2025      | **Patch-wise Structural Loss**                | patch-level mean/variance/correlation matching                                     | PETSA structural term의 기반          | 직접 비교 불가                                              |
| 2025      | **PETSA**                                     | low-rank gated input/output adapters + composite loss                              | 기준 논문                              | —                                                     |
| 2026      | **COSA, ICLR 2026**                           | single output-space residual adapter + recent-GT context                           | PETSA dual adapter를 더 단순화하는 방향     | protocol 확인 후 제한적 비교                                  |
| 2026      | **Towards Principled TTA / FAC**              | matured-GT-only protocol + frequency-aware correction                              | PETSA evaluation protocol 자체를 재검토  | 기존 PETSA 표와 단순 비교 주의                                  |
| 2026      | **AdaNODEs**                                  | Neural ODE 기반 source-free TSF-TTA                                                  | nonlinear temporal dynamics 방향     | preprint, 설정 다름                                       |
| 2026      | **Non-stationary TS TTA → Financial Markets** | norm-affine TTA, uncertainty fallback, real financial regime                       | real-world external validity 문제 강조 | 설정 다름                                                 |

---

## 2020–2023: TTA 연구가 PETSA에 준 교훈

### TENT

TENT는

$$
\min_{\phi}
H(p_\phi(y\mid x))
$$

즉 prediction entropy를 낮추는 방식으로 test-time adaptation을 수행합니다.

학습 데이터나 test label 없이 normalization statistic과 affine parameter를 update합니다. ([arXiv][6])

PETSA와 비교하면 PETSA는 time-series의 delayed label이라는 추가 정보를 활용한다는 차이가 있습니다.

---

### LAME

LAME는 parameter update 자체를 하지 않고 prediction output을 조정합니다. 특히 test-time hyperparameter가 동일 scenario에서 고르지 않으면 기존 TTA가 catastrophic failure를 보일 수 있음을 지적했습니다. ([Open Access CVF][9])

이 관점은 PETSA의 Figure 11–13과 매우 잘 연결됩니다.

PETSA도

$$
r,\alpha_{\text{init}},\beta
$$

에 민감하기 때문입니다.

---

### EATA와 CoTTA

EATA는 모든 test sample을 무조건 backward하지 않고 informative sample을 선별하고, 중요한 parameter를 보존하는 regularization으로 forgetting을 줄입니다. ([Proceedings of Machine Learning Research][10])

CoTTA는 continually changing target distribution을 대상으로 error accumulation과 catastrophic forgetting을 직접 다룹니다. ([Open Access CVF][8])

PETSA는 **parameter efficiency는 잘 다루지만 long-term stability를 깊게 분석하지 않습니다.**

따라서 후속 PETSA 연구에서는

$$
\text{efficiency}
+
\text{accuracy}
+
\text{stability}
$$

세 축을 동시에 평가해야 합니다.

---

# 2024: PEFT가 시계열로 확장

**Low-Rank Adaptation of Time Series Foundational Models for Out-of-Domain Modality Forecasting**은 Lag-Llama, MOIRAI, Chronos 같은 TS foundation model에 LoRA를 적용하여 out-of-domain fine-tuning을 연구했습니다. ([arXiv][11])

**Channel-Aware Low-Rank Adaptation in Time Series Forecasting**은 channel independence가 distribution shift에 더 robust할 수 있지만 표현력이 약하고, channel dependence는 expressive하지만 overfitting될 수 있다는 trade-off를 low-rank adaptation으로 조절합니다. ([arXiv][12])

이 두 연구가 보여주는 중요한 흐름은

$$
\text{large adaptable capacity}
\rightarrow
\text{small structured adaptable subspace}
$$

입니다.

PETSA는 이 철학을 **offline fine-tuning이 아니라 test-time forecasting adaptation으로 이동시킨 것**으로 볼 수 있습니다.

---

# 2025: TAFAS → PETSA

TAFAS는 AAAI 2025에서 시계열 forecasting을 위한 TTA를 본격적으로 정식화하고,

* partially observed GT,
* gated calibration,
* non-stationary test distribution

을 사용했습니다. ([AAAI Publications][13])

PETSA는 이 framework 위에서

$$
\text{TAFAS}
+
\text{low-rank parameterization}
+
\text{specialized loss}
$$

로 발전시킨 연구입니다.

즉 PETSA의 가장 직접적인 기여는 새로운 TTA 문제를 처음 제안한 것이라기보다

> **TAFAS-type TSF-TTA를 더 작고 구조화된 parameter space에서 수행하도록 만든 것**

이라고 보는 것이 정확합니다.

---

# 2026: PETSA 이후 연구가 바꾸고 있는 방향

## COSA — 구조를 더 단순하게

ICLR 2026의 COSA는 **single output-space adapter**를 사용합니다. 공식 ICLR 자료는 frozen prediction과 최근 GT statistics를 context로 사용해 residual correction을 수행하며, leakage-free protocol 하에서 baseline 대비 13.91–17.03%, 기존 SOTA TTA 대비 10.48–13.05% 개선을 보고합니다. ([ICLR Proceedings][2])

다만 이 비율을 PETSA Table 1의 MSE와 직접 비교해서

> COSA가 PETSA보다 정확히 10.xx% 뛰어나다

고 단순 해석해서는 안 됩니다.

평균 방식과 protocol이 다를 수 있기 때문입니다.

그럼에도 연구 흐름은 명확합니다.

$$
\text{TAFAS dual calibration}
\rightarrow
\text{PETSA parameter-efficient dual calibration}
\rightarrow
\text{COSA simpler output correction}
$$

으로 이동하고 있습니다.

---

## Towards Principled TTA / FAC — “방법”보다 “평가 protocol”을 재검토

2026년 arXiv의 **Towards Principled Test-Time Adaptation for Time Series Forecasting**은 TSF-TTA 연구들이 revealed target을 사용하는 방법이 서로 다르며 protocol이 heterogeneous하다고 지적합니다.

그리고 **matured ground truth만 사용하는 더 깨끗한 protocol**을 제안합니다. ([arXiv][14])

이 연구가 PETSA 평가에 미치는 영향은 큽니다.

즉 앞으로는 단순히

$$
\text{MSE}_{\text{PETSA}} < \text{MSE}\_{\text{baseline}}
$$

만 확인하는 것이 아니라,

$$
\boxed{
\text{동일한 정보가 동일한 시점까지 실제로 사용 가능했는가?}
}
$$

를 검증해야 합니다.

**용어 — Matured ground truth:** 특정 예측이 끝난 뒤 실제로 충분히 시간이 지나 정상적으로 확보된 정답으로, 아직 완성되지 않은 미래구간을 섞지 않는 보다 보수적인 supervision 설정입니다.

---

# PETSA가 앞으로의 연구에 미치는 영향

PETSA의 가장 중요한 영향은 **“TTA의 성능과 adaptation parameter 수는 독립적으로 최적화할 수 있다”**는 방향을 명확히 보여준 것입니다.

기존 접근의 사고가

$$
\text{distribution shift}
\Rightarrow
\text{large model update}
$$

였다면 PETSA는

$$
\text{distribution shift}
\Rightarrow
\text{small targeted correction}
$$

으로 바꿉니다.

이 아이디어는 특히

* large foundation forecaster,
* edge deployment,
* industrial online forecasting,
* 모델 retraining이 어려운 장비,
* 데이터가 계속 drift하는 공정

에서 중요합니다.

다만 2026년 흐름을 보면 다음 단계의 경쟁은 **parameter 수를 더 줄이는 것 자체가 아니라**,

$$
\boxed{
\text{언제 적응할지}
+
\text{무엇을 적응할지}
+
\text{얼마나 적응할지}
}
$$

를 안전하게 자동화하는 방향으로 이동하고 있습니다.

---

# 8-3. 제가 제안하는 구체적인 후속 연구

가장 가치가 큰 실험은 다음과 같습니다.

### 1. Clean-protocol PETSA

PETSA를 그대로 재구현하되

$$
\text{POGT+full GT}
$$

와

$$
\text{matured GT only}
$$

를 별도 protocol로 나누어 비교해야 합니다.

그러면 PETSA의 이득 중 어느 정도가

* architecture에서 오는지,
* 더 빠른 target availability에서 오는지

분리할 수 있습니다.

---

### 2. Shift-aware PETSA

shift score를

```math
s_t
=
D(P_{\text{ref}},P_t)
```

로 계산하여

$$
s_t < \tau_1
\Rightarrow
\text{No Adapt}
$$

$$
\tau_1\le s_t < \tau_2
\Rightarrow
\text{Output Adapter only}
$$

$$
s_t\ge\tau_2
\Rightarrow
\text{Input + Output Adapter}
$$

형태로 바꾸는 연구입니다.

이 방식은 negative adaptation을 줄이면서 generalization을 높일 가능성이 큽니다.

---

### 3. Oracle-free hyperparameter study

가장 중요한 실험입니다.

$r,\beta,\alpha$를 test dataset에서 최적화하지 않고

* source validation에서 결정,
* 다른 dataset에서 결정,
* leave-one-dataset-out meta-selection

한 뒤 target dataset에 그대로 적용해야 합니다.

예를 들어

```math
(r^*,\beta^*,\alpha^*)
=
\arg\min_{r,\beta,\alpha}
\frac{1}{K-1}
\sum_{k\neq k^*}
\mathcal L_k
```

로 고르고 완전히 보지 않은 $k^*$ dataset에서 평가합니다.

이 성능이 유지되어야 **진정한 hyperparameter generalization**이라고 할 수 있습니다.

---

### 4. Controlled shift benchmark

단순 기존 dataset의 후반부를 평가하는 것 외에

$$
x_t' = a_t x_t + b_t+\epsilon_t
$$

형태로

* mean drift,
* scale drift,
* frequency drift,
* trend break,
* variance shift,
* channel relationship shift

를 강도별로 주입해야 합니다.

그러면

$$
\text{shift severity}
\rightarrow
\text{PETSA gain}
$$

의 함수 관계를 분석할 수 있습니다.

---

### 5. Stability–plasticity trade-off

adapter가 빨리 변하면 새로운 regime에는 적응하지만 기존 지식을 잃을 수 있습니다.

이를

$$
\min_\phi
\mathcal L_{\text{new}}(\phi)
+
\lambda
\|\phi-\phi_{\text{anchor}}\|_2^2
$$

로 정식화할 수 있습니다.

* 첫 항: 새 환경에 적응하는 **plasticity**
* 둘째 항: 기존 상태를 유지하는 **stability**

입니다.

이 문제는 PETSA를 단발성 benchmark가 아니라 수개월 이상 운영하는 online system으로 만들기 위해 필수적입니다.

---

# 최종 평가

이 논문은 **아이디어의 방향성은 매우 좋습니다.**

특히

$$
\boxed{
\text{Frozen Forecaster}
+
\text{Small Gated Low-Rank Calibration}
}
$$

이라는 설계는 실제 deployment에서 유용성이 높습니다.

다만 연구적 증거를 등급으로 나누면 다음과 같이 판단합니다.

| 항목                               | 평가                    |
| -------------------------------- | --------------------- |
| 구조적 참신성                          | **높음**                |
| parameter efficiency 근거          | **강함**                |
| 여러 backbone에서의 적용성               | **상당히 강함**            |
| 모든 조건에서의 정확도 우위                  | **입증되지 않음**           |
| 통계적 확실성                          | **부족**                |
| unseen distribution-shift 일반화    | **불충분**               |
| hyperparameter robustness        | **취약 가능성 있음**         |
| real-world deployment validation | **부족**                |
| Table 1 v1 데이터 신뢰성               | **일부 Avg row 재검증 필요** |
| 향후 연구 잠재력                        | **매우 높음**             |

따라서 PETSA를 연구 관점에서 가장 정확하게 평가하면,

> **“일반화 성능이 이미 완전히 해결된 TTA 방법”이라기보다는, forecasting TTA를 작은 저차원 adaptation subspace로 제한하여 accuracy–efficiency trade-off를 크게 개선한 연구이며, 다음 단계는 clean evaluation protocol, oracle-free adaptation, shift-aware capacity control을 통해 실제 generalization을 검증하는 것**

이라고 정리하는 것이 타당합니다.

---

# 참고한 논문·사이트 및 자료 제목

아래는 이번 답변에서 실제로 참고한 주요 원문·공식 자료입니다.

1. **Medeiros et al., “Accurate Parameter-Efficient Test-Time Adaptation for Time Series Forecasting”**, arXiv:2506.23424, 2025 — 업로드된 PDF 및 arXiv 원문. ([arXiv][1])
2. **Kim et al., “Battling the Non-stationarity in Time Series Forecasting via Test-time Adaptation”**, AAAI 2025 — TAFAS. ([AAAI Publications][13])
3. **Wang et al., “Tent: Fully Test-Time Adaptation by Entropy Minimization”**, ICLR 2021. ([OpenReview][15])
4. **Boudiaf et al., “Parameter-Free Online Test-Time Adaptation”**, CVPR 2022 — LAME. ([Open Access CVF][9])
5. **Niu et al., “Efficient Test-Time Model Adaptation without Forgetting”**, ICML 2022 — EATA. ([Proceedings of Machine Learning Research][10])
6. **Wang et al., “Continual Test-Time Domain Adaptation”**, CVPR 2022 — CoTTA. ([Open Access CVF][8])
7. **Zhao et al., “On Pitfalls of Test-Time Adaptation”**, ICML 2023. ([Proceedings of Machine Learning Research][5])
8. **Gupta et al., “Low-Rank Adaptation of Time Series Foundational Models for Out-of-Domain Modality Forecasting”**, 2024. ([arXiv][11])
9. **Nie et al., “Channel-Aware Low-Rank Adaptation in Time Series Forecasting”**, CIKM 2024. ([arXiv][12])
10. **Wang et al., “FreDF: Learning to Forecast in the Frequency Domain”**, ICLR 2025. ([ICLR Proceedings][3])
11. **Kudrat et al., “Patch-wise Structural Loss for Time Series Forecasting”**, ICML 2025. ([Proceedings of Machine Learning Research][4])
12. **Im & Kwon, “COSA: Context-aware Output-Space Adapter for Test-Time Adaptation in Time Series Forecasting”**, ICLR 2026. ([ICLR Proceedings][2])
13. **Wang et al., “Towards Principled Test-Time Adaptation for Time Series Forecasting”**, arXiv 2026 — Frequency-Aware Calibration 및 matured-GT protocol. ([arXiv][14])
14. **Dang et al., “AdaNODEs: Test Time Adaptation for Time Series Forecasting Using Neural ODEs”**, arXiv 2026. ([arXiv][16])
15. **Wu et al., “Test-Time Adaptation for Non-stationary Time Series: From Synthetic Regime Shifts to Financial Markets”**, arXiv 2026. ([arXiv][7])

특히 **PETSA 원문 v1의 Table 1 Avg 행 문제는 2차 자료의 주장이 아니라, 첨부된 PDF의 인쇄 수치를 직접 산술 검산하여 확인한 사항**입니다. 따라서 해당 수치를 후속 연구나 발표자료에 사용할 경우 원 저자의 코드/raw result를 재실행해 확인하는 것이 필요합니다.

[1]: https://arxiv.org/abs/2506.23424?utm_source=chatgpt.com "Accurate Parameter-Efficient Test-Time Adaptation for Time Series Forecasting"
[2]: https://proceedings.iclr.cc/paper_files/paper/2026/hash/2a8ce71baac4c89bf9ff479d8240c7d9-Abstract-Conference.html?utm_source=chatgpt.com "COSA: Context-aware Output-Space Adapter for Test-Time Adaptation in Time Series Forecasting"
[3]: https://proceedings.iclr.cc/paper_files/paper/2025/hash/1457fb1e5d72cdc4ecd88bc10f916095-Abstract-Conference.html?utm_source=chatgpt.com "FreDF: Learning to Forecast in the Frequency Domain"
[4]: https://proceedings.mlr.press/v267/kudrat25a.html?utm_source=chatgpt.com "Patch-wise Structural Loss for Time Series Forecasting"
[5]: https://proceedings.mlr.press/v202/zhao23d.html?utm_source=chatgpt.com "On Pitfalls of Test-Time Adaptation"
[6]: https://arxiv.org/abs/2006.10726?utm_source=chatgpt.com "Tent: Fully Test-time Adaptation by Entropy Minimization"
[7]: https://arxiv.org/abs/2602.00073?utm_source=chatgpt.com "Test-Time Adaptation for Non-stationary Time Series: From Synthetic Regime Shifts to Financial Markets"
[8]: https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Continual_Test-Time_Domain_Adaptation_CVPR_2022_paper.html?utm_source=chatgpt.com "CVPR 2022 Open Access Repository"
[9]: https://openaccess.thecvf.com/content/CVPR2022/html/Boudiaf_Parameter-Free_Online_Test-Time_Adaptation_CVPR_2022_paper.html?utm_source=chatgpt.com "CVPR 2022 Open Access Repository"
[10]: https://proceedings.mlr.press/v162/niu22a.html?utm_source=chatgpt.com "Efficient Test-Time Model Adaptation without Forgetting"
[11]: https://arxiv.org/abs/2405.10216?utm_source=chatgpt.com "Low-Rank Adaptation of Time Series Foundational Models for Out-of-Domain Modality Forecasting"
[12]: https://arxiv.org/abs/2407.17246?utm_source=chatgpt.com "Channel-Aware Low-Rank Adaptation in Time Series Forecasting"
[13]: https://ojs.aaai.org/index.php/AAAI/article/view/33965?utm_source=chatgpt.com "Battling the Non-stationarity in Time Series Forecasting via Test-time Adaptation | Proceedings of the AAAI Conference on Artificial Intelligence"
[14]: https://arxiv.org/abs/2605.17250?utm_source=chatgpt.com "Towards Principled Test-Time Adaptation for Time Series Forecasting"
[15]: https://openreview.net/pdf/4de0af9691a5dcc52de7de756676fded33d037ef.pdf?utm_source=chatgpt.com "Published as a conference paper at ICLR 2021"
[16]: https://arxiv.org/abs/2601.12893?utm_source=chatgpt.com "AdaNODEs: Test Time Adaptation for Time Series Forecasting Using Neural ODEs"
