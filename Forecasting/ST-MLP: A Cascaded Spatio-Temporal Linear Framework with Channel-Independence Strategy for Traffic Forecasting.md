# ST-MLP: A Cascaded Spatio-Temporal Linear Framework with Channel-Independence Strategy for Traffic Forecasting

## 분석 기준과 문헌 상태

아래 분석의 **페이지, Equation, Figure, Table 번호는 사용자가 첨부한 2023년 arXiv v1, 12쪽 PDF**를 기준으로 합니다.   
웹 검색으로 논문 자체와 후속 문헌도 교차검증했습니다.  
중요한 업데이트가 하나 있는데, 2023년의 이 preprint는 이후 연구가 확장되어 **2026년 6월 IEEE Transactions on Intelligent Transportation Systems, Vol. 27(6), pp. 6343–6355에 _“Channel-Independence for Traffic Forecasting: A Cascaded Spatio-Temporal MLP Framework”_라는 제목으로 정식 저널 출판**된 것이 확인됩니다.  
따라서 아래에서 **2023년 결과와 2026년 출판 사실은 혼합하지 않고 분리**해 다룹니다. ([arXiv][1])

---

# 1. Executive Summary — 10문장 이내

1. ST-MLP의 핵심 문제의식은 교통예측에서 STGNN의 구조가 지나치게 복잡해지는 반면 정확도 향상은 제한적이며, 시간에 따른 **distribution shift** 때문에 복잡한 채널 혼합 모델이 과적합될 수 있다는 것입니다.
2. 저자들은 각 센서의 과거 관측값을 다른 센서의 관측값과 직접 혼합하지 않는 **Channel-Independence(CI)**를 교통예측에 도입하면서도, 그래프·노드·시간 정보를 embedding 형태로 각 채널 내부에 주입하는 ST-MLP를 제안합니다.
3. 모델은 Time-in-Day와 Day-in-Week temporal embedding, predefined graph와 learnable node embedding으로 구성된 spatial embedding, 그리고 관측 시계열을 변환한 data embedding을 순차적으로 결합합니다.
4. Temporal → Spatial → Data embedding을 한 번에 concatenate하지 않고 **cascaded MLP**로 단계적으로 융합하는 것이 Figure 2의 핵심 구조입니다.
5. PEMS-BAY, PEMS04, PEMS07, PEMS08에서 12개 과거 시점으로 다음 12개 시점을 예측한 결과, ST-MLP는 대부분의 평균 MAE/RMSE/MAPE에서 STGNN 및 단순 시계열 baseline과 경쟁적이거나 가장 좋은 성능을 보였습니다.
6. 특히 Table 2에서는 channel mixing을 추가한 CM ST-MLP가 train error를 낮추면서도 test error를 크게 악화시켜, 저자들은 이를 CI가 distribution shift에 대해 더 강한 일반화 성능을 갖는 근거로 해석합니다.
7. 계산 효율에서도 ST-MLP의 epoch당 학습 시간이 네 데이터셋 모두 비교 STGNN보다 가장 짧았으며, PEMS07에서는 가장 가까운 경쟁 모델보다도 약 2.9배 빠릅니다.
8. 다만 논문에는 반복 seed에 대한 표준편차, confidence interval, 유의성 검정이 없고 distribution shift를 정량적으로 측정하지 않았으므로, **“CI가 distribution shift 때문에 일반화를 향상시킨다”는 인과적 해석은 아직 충분히 증명되지 않았습니다.**
9. 후속 연구 관점에서는 순수 CI를 유지하기보다 **CI backbone + 강하게 정규화된 sparse channel interaction + distribution-shift normalization + cross-city/rolling-time validation**을 결합하는 방향이 가장 유망합니다. 

---

# 1-1. 연구의 목적과 필요성

## 연구가 해결하려는 근본 문제

기존 교통 예측은 도로망을 graph로 표현한 뒤 GCN/GNN, recurrent network, attention 등을 복합적으로 결합하는 **Spatio-Temporal Graph Neural Network(STGNN)** 중심으로 발전했습니다. 문제는 구조가 복잡해질수록 계산량과 메모리는 증가하지만 정확도 상승 폭은 점점 작아진다는 것입니다. 저자들은 이 상황에서 “정말 센서의 관측값을 매 단계 복잡하게 서로 섞어야 하는가?”를 질문합니다. 이는 같은 시기 STID와 SimST에서도 나타난 문제의식입니다.  ([arXiv][2])

> **용어 — STGNN:** 도로의 센서를 node, 도로 연결관계를 edge로 만들고, 공간 관계와 시간 변화를 동시에 학습하는 Graph Neural Network 계열입니다.

두 번째 문제는 **distribution shift**입니다. 논문의 Figure 1은 Z-score normalization을 수행한 이후에도 PEMS07과 PEMS08에서 train과 test 시계열 분포가 다르게 나타날 수 있음을 보여줍니다. 즉,

$$
P_{\text{train}}(X,Y) \neq P_{\text{test}}(X,Y)
$$

가 될 수 있다는 문제입니다.

> **용어 — Distribution shift:** 학습할 때 본 데이터의 평균·분산·상관관계·조건부 관계와 실제 미래 데이터의 그것이 달라지는 현상입니다. 시계열에서는 계절성 변화, 공휴일, 사고, 도로 공사, 장기 추세 변화 등이 원인이 될 수 있습니다.

이 문제는 ST-MLP만의 주장이 아닙니다. RevIN은 시계열의 시간가변 평균·분산 자체가 forecasting 성능을 떨어뜨릴 수 있음을 보였고, Han et al.은 Channel-Dependent 모델이 더 높은 표현력을 가지지만 distribution drift에서는 CI보다 robustness가 떨어질 수 있다는 **capacity–robustness trade-off**를 분석했습니다. ([ML Anthology][3])

---

# 2. 논문의 핵심 주장과 근거

| 핵심 주장                                         | 저자가 제시한 근거                                       | 위치                         | 평가                                              |
| --------------------------------------------- | ------------------------------------------------ | -------------------------- | ----------------------------------------------- |
| 복잡한 STGNN이 항상 필요하지 않음                         | MLP/linear layer 중심 ST-MLP가 여러 STGNN보다 높은 평균 성능  | p.5–6, Table 1             | 상당히 설득력 있음. 단, 모든 horizon/metric에서 1위는 아님       |
| CI를 교통예측에 적용하면서 spatial information을 유지할 수 있음 | graph 정보를 관측 channel mixing이 아닌 embedding으로 주입   | p.2–4, Eq. (2)–(8), Fig. 2 | 논문의 가장 독창적인 설계                                  |
| Cascaded fusion이 단순 concat보다 좋음               | PEMS08 ablation에서 cascaded variant의 MAE가 일관되게 낮음 | p.6–7, Fig. 4              | 방향성은 확인되나 효과 크기가 작고 통계검정 없음                     |
| Time-in-Day가 특히 중요함                           | $E_{td}$ 제거 시 MAE 14.03 → 15.17                  | p.5–7, Fig. 4              | 매우 강한 ablation 결과                               |
| CI가 CM보다 일반화에 유리함                             | CM은 train error가 더 작지만 test error가 커짐            | p.7, Table 2               | 중요한 근거이나 CI/CM 외의 capacity 차이가 confound가 될 수 있음 |
| ST-MLP가 계산 효율적임                               | 모든 데이터셋에서 epoch당 training time 최소                | p.5, Fig. 3; p.12, Table 6 | 명확함. 단 inference latency/memory는 미보고            |
| CI가 distribution shift를 완화함                   | Fig.1의 shift + Table2의 generalization gap        | p.1, p.7                   | 정황증거는 있으나 shift 정량화/인과 검증 부족                    |
| 더 단순한 architecture 연구가 가치 있음                  | accuracy와 efficiency를 동시에 달성                     | Conclusion, p.7            | 후속 연구 흐름과 상당히 부합                                |



---

# 2-1. 해결 문제, 제안 방법, 수식, 모델 구조, 성능 및 한계

## A. 문제 정의

노드가 $N$개이고 과거 시점이 $T$개인 경우 논문은 입력을

$$
X\in\mathbb{R}^{N\times T}
$$

로 둡니다.

예측하려는 미래 $Q$개 시점은

$$
\hat X\in\mathbb{R}^{N\times Q}
$$

입니다.

원래 일반 정의는 traffic feature가 $C$개인

$$
X\in\mathbb{R}^{N\times T\times C}
$$

이지만, **실험에서는 $C=1$로 단순화**합니다. 이는 중요한 제한사항입니다.

Equation (1)은

$$
\hat X=f(G,X,T_{td},T_{dw})
$$

입니다.

기호는 다음과 같습니다.

* $G=(V,E)$: 도로망 graph입니다.
* $V$: 센서 또는 도로 node 집합입니다.
* $E$: node 간 연결 edge입니다.
* $X$: 과거 traffic observations입니다.
* $N$: sensor/node 수입니다.
* $T$: look-back window 길이입니다.
* $Q$: forecast horizon입니다.
* $T_{td}$: 각 입력 시점의 **Time in Day** index입니다.
* $T_{dw}$: 각 입력 시점의 **Day in Week** index입니다.
* $\hat X$: 예측값입니다.

> **용어 — Look-back window:** 미래를 예측하기 위해 모델이 뒤돌아보는 과거 구간입니다. 이 논문에서는 5분 간격 12개, 즉 1시간입니다.

실험에서는

$$
T=12,\qquad Q=12
$$

이므로 **과거 60분 → 미래 60분** 예측입니다. p.4의 implementation details에 명시되어 있습니다. 

---

## B. Channel Independence가 정확히 무엇인가

논문은 spatial node를 channel로 간주합니다.

입력이

$$
X_{\text{input}}\in\mathbb{R}^{N\times T}
$$

일 때 세 가지 연산을 정의합니다.

### Equation (2): Temporal mixing

$$
X_t=\text{TemporalMix}(X_{\text{input}})
$$

$$
X_t\in\mathbb{R}^{N\times T'}
$$

각 node $i$마다 자신의 시간축만 변환합니다.

즉,

$$
x_i(1:T)\rightarrow z_i(1:T')
$$

이며 다른 node $j$의 실제 관측값을 가져오지 않습니다.

---

### Equation (3): Channel mixing

$$
X_c=\text{ChannelMix}(X_{\text{input}})
$$

출력은

$$
X_c\in\mathbb{R}^{N'\times T}
$$

입니다.

여기서는 한 센서의 representation을 만들 때 다른 센서 정보까지 직접 섞습니다.

---

### Equation (4): Temporal-channel mixing

```math
X_{tc}
=
\text{TemporalChannelMix}(X_{\text{input}})
```

$$
X_{tc}\in\mathbb{R}^{N'\times T'}
$$

시간축과 센서축을 둘 다 섞습니다.

CI 모델은 원칙적으로 **Equation (2) 유형만 사용**합니다.

> **용어 — Channel Independence(CI):** node $i$의 미래값을 예측할 때 다른 node의 동적 관측 시계열 $X_j$를 직접 입력으로 섞지 않는 전략입니다.

여기서 매우 중요한 오해가 하나 있습니다.

**ST-MLP가 spatial information을 사용하지 않는 것은 아닙니다.**

오히려 graph와 node 정보를 **static/learnable embedding으로 변환한 뒤 node별 temporal representation에 넣습니다.**

따라서

> “다른 센서의 실측 시계열을 직접 섞지 않는다”

와

> “도로 공간구조를 전혀 이용하지 않는다”

는 전혀 다른 이야기입니다.

이 구분이 ST-MLP의 핵심입니다. p.2–4, Fig.2에서 확인됩니다. 

---

# C. Temporal embedding

하루를 $K$개 time slot으로 나눕니다.

5분 간격이므로

$$
K=\frac{24\times60}{5}=288
$$

입니다.

현재 Time-in-Day 위치를 one-hot vector로 만들면

$$
t_{td}\in\mathbb{R}^{1\times K}
$$

이고 학습 가능한 codebook을

$$
B_{td}\in\mathbb{R}^{K\times d_{td}}
$$

로 둡니다.

그러면

$$
t_{td}B_{td}
\in
\mathbb{R}^{1\times d_{td}}
$$

가 됩니다.

one-hot vector이므로 실제 의미는 복잡한 행렬 연산이 아니라 **해당 시간대 embedding 한 행을 선택하는 것**입니다.

모든 node가 동일한 clock time을 공유하므로 이를 $N$번 복제하여

$$
E_{td}\in\mathbb{R}^{N\times d_{td}}
$$

를 만듭니다.

Day-in-Week도 같은 방식으로

$$
B_{dw}\in\mathbb{R}^{7\times d_{dw}}
$$

를 사용합니다.

최종 temporal embedding은 Equation (5):

```math
E_t
=
\text{Concat}(E_{td},E_{dw})
```

이며

$$
E_t\in\mathbb{R}^{N\times d_t},
\qquad
d_t=d_{td}+d_{dw}
$$

입니다.

> **용어 — Embedding:** category 자체를 0,1 숫자로 사용하는 대신 학습 가능한 연속 벡터로 변환한 표현입니다. 비슷한 traffic pattern을 가진 시간대가 학습 과정에서 비슷한 벡터 표현을 갖게 될 수 있습니다.

Fig.4에서는 $E_{td}$를 제거했을 때 test MAE가 **14.03 → 15.17**로 크게 증가합니다. 약 **8.1% 악화**이므로 이 데이터에서는 복잡한 graph 정보보다 일중 주기 정보를 정확히 알려주는 것이 매우 중요한 것으로 해석할 수 있습니다. 

---

# D. Spatial embedding

ST-MLP의 spatial representation은 두 부분입니다.

## 1. 알려진 graph

adjacency/graph matrix:

$$
A\in\mathbb{R}^{N\times N}
$$

학습 가능한 graph codebook:

$$
B_{sp}\in\mathbb{R}^{N\times d_{sp}}
$$

Equation (6):

$$
E_{sp}=AB_{sp}
$$

따라서

$$
E_{sp}\in\mathbb{R}^{N\times d_{sp}}
$$

입니다.

node $i$의 행만 보면

```math
E_{sp}^{(i)}
=
\sum_{j=1}^{N}A_{ij}B_{sp}^{(j)}
```

라고 해석할 수 있습니다.

즉 node $i$의 graph embedding은 연결된 node들의 codebook representation을 $A_{ij}$로 가중합한 것입니다.

이 과정에서 graph topology는 들어가지만 **현재 시점의 traffic observation $X_j(t)$ 자체를 node $i$의 동적 값과 혼합하지 않는다는 점**이 핵심입니다.

---

## 2. 알려지지 않은 spatial information

도로 adjacency만으로 표현할 수 없는 sensor별 특성을 위해

$$
B_{su}\in\mathbb{R}^{N\times d_{su}}
$$

라는 학습 가능한 unknown spatial codebook을 둡니다.

node $i$에는

$$
E_{su}^{(i)}
$$

가 선택됩니다.

이를 predefined graph embedding과 결합하면 Equation (7):

```math
E_s^{(i)}
=
\text{Concat}
\left(
E_{sp}^{(i)},E_{su}^{(i)}
\right)
```

이고

$$
E_s\in\mathbb{R}^{N\times d_s},
\qquad
d_s=d_{sp}+d_{su}
$$

입니다.

이 설계는 “실제 도로 연결정보가 모든 functional correlation을 설명하지 못할 수 있다”는 가정에 대응합니다. 

---

# E. Data embedding

traffic data뿐 아니라 각 과거 시점의 time index를 같이 넣습니다.

$T_{td}$와 $T_{dw}$를 node마다 복제하여

$$
T'_{td},T'_{dw}\in\mathbb{R}^{N\times T}
$$

를 만들고, Equation (8):

$$
E_d=
\text{Linear}
\left(
\text{Concat}
[X,T'_{td},T'_{dw}]
\right)
$$

로 구성합니다.

출력은

$$
E_d\in\mathbb{R}^{N\times d_d}
$$

입니다.

즉 temporal embedding $E_t$가 **“현재가 월요일 오전 8시 30분 같은 어떤 시간 상태인가”**를 표현한다면, $E_d$는 **실제 최근 1시간 traffic trajectory와 그 trajectory 안의 timestamp 구조를 함께 압축**합니다.

---

# F. Cascaded architecture

Figure 2(a)의 핵심입니다.

논문 설명을 수식 형태로 명확하게 다시 적으면 다음과 같습니다.

> 아래 식은 원문의 별도 numbered equation이 아니라 **Figure 2와 본문 구조를 제가 수식화한 표현**입니다.

먼저,

$$
H_t=\text{MLP}_A(E_t)
$$

그다음 spatial 정보를 넣어

```math
E_{st}
=
\text{MLP}_B
\left(
\text{Concat}[H_t,E_s]
\right)
```

그리고 data embedding을 넣어

$$
E=
\text{MLP}_C
\left(
\text{Concat}[E_{st},E_d]
\right)
$$

마지막으로

```math
\hat X
=
EW_o+b_o
```

와 같은 linear projection으로

$$
\hat X\in\mathbb{R}^{N\times Q}
$$

를 생성합니다.

핵심은

$$
E_t\rightarrow(E_t,E_s)\rightarrow(E_t,E_s,E_d)
$$

순서로 information complexity를 증가시킨다는 것입니다.

단순 구조라면

$$
E=
\text{Concat}(E_t,E_s,E_d)
$$

한 뒤 바로 예측할 수도 있지만 저자들은 cascade가 더 좋은 결과를 냈다고 보고합니다.

---

## MLP block 내부

Figure 2(b)는

$$
\text{Linear}
\rightarrow
\text{Normalization}
\rightarrow
\text{ReLU}
\rightarrow
\text{Dropout}
$$

그리고 residual connection으로 구성됩니다.

이를 수식화하면 대략

```math
H^{(\ell+1)}
=
H^{(\ell)}
+
\text{Dropout}
\left[
\text{ReLU}
\left(
\text{Norm}
(
H^{(\ell)}W^{(\ell)}+b^{(\ell)}
)
\right)
\right]
```

입니다.

> **용어 — Residual connection:** 변환된 값에 원래 입력을 다시 더하는 구조로, 깊은 network에서 gradient 전달과 안정적인 학습을 돕습니다.

> **용어 — Dropout:** 학습 중 일부 hidden unit을 무작위로 제거하여 특정 neuron이나 feature에 과도하게 의존하는 것을 줄이는 regularization입니다.

한 가지 주의점이 있습니다. 제목에는 **“Linear Framework”**가 들어가지만 MLP 내부에 ReLU가 존재하므로 전체 ST-MLP는 엄밀하게 말해 **전역적으로 선형 모델이 아닙니다.** Linear layer 위주의 가벼운 architecture라는 의미에 더 가깝게 읽는 것이 적절합니다.

---

# G. 실제 hyperparameter

Appendix Table 5 기준 주요 설정은 다음과 같습니다.

$$
\text{batch size}=32
$$

$$
\text{learning rate}=0.002
$$

$$
\text{weight decay}=10^{-4}
$$

$$
\gamma=0.5
$$

$$
\text{milestones}=[1,50,80]
$$

$$
\text{epochs}=200
$$

MLP depth는

$$
n_A=1,\qquad n_B=1,\qquad n_C=3
$$

이고 embedding dimension은

$$
d_d=96,
\quad
d_{su}=32,
\quad
d_{sp}=32,
\quad
d_{td}=32
$$

입니다.

$d_{dw}$는 데이터셋별로 일부 다르게 설정됩니다.

즉 구조 자체는 상당히 얕습니다. 

---

# H. 성능 향상

## Table 1 — 12-step 평균 성능

| Dataset  | ST-MLP MAE |      RMSE |       MAPE | 중요한 비교                         |
| -------- | ---------: | --------: | ---------: | ------------------------------ |
| PEMS-BAY |   **1.56** |  **3.55** |  **3.50%** | MAE는 STID와 동률, RMSE/MAPE 우세    |
| PEMS04   |  **18.05** | **29.72** | **12.28%** | STID MAE 18.34 대비 약 1.58% 감소   |
| PEMS07   |  **19.51** | **32.61** |  **8.26%** | STID 대비 소폭 우세                  |
| PEMS08   |  **14.03** | **23.07** |  **9.25%** | MAE에서 STID 14.23 대비 약 1.41% 감소 |



여기서 중요한 점은 **“전체적으로 가장 좋다”와 “모든 숫자가 가장 좋다”는 다르다는 것**입니다.

예를 들어 PEMS-BAY 15분 MAE에서는

* GraphWaveNet: 1.31
* DCRNN: 1.31
* STID: 1.31
* **ST-MLP: 1.32**

이므로 ST-MLP가 모든 horizon에서 절대적으로 1위라는 표현은 정확하지 않습니다.

다만 전체 평균 성능과 계산비용을 함께 보면 경쟁력은 매우 높습니다.

---

# I. CI가 일반화를 높인다는 가장 중요한 근거 — Table 2

### PEMS-BAY

$$
MAE_{\text{CI,test}}=1.56
$$

$$
MAE_{\text{CM,test}}=1.75
$$

CI가 약 **10.9% 낮습니다.**

### PEMS04

$$
18.05\quad\text{vs.}\quad20.74
$$

약 **13.0% 낮습니다.**

### PEMS08

$$
14.03\quad\text{vs.}\quad15.58
$$

약 **10.0% 낮습니다.**

더 중요한 것은 train-test gap입니다.

| Dataset  | CI Train→Test MAE gap | CM Train→Test MAE gap |
| -------- | --------------------: | --------------------: |
| PEMS-BAY |      $1.56-1.41=0.15$ |      $1.75-1.40=0.35$ |
| PEMS04   |    $18.05-16.75=1.30$ |    $20.74-16.52=4.22$ |
| PEMS08   |   $14.03-14.11=-0.08$ |    $15.58-13.25=2.33$ |

즉 CM은 **training fit은 더 좋아졌는데 test generalization은 악화**됩니다.

이것이 이 논문에서 CI에 대한 가장 중요한 실험적 근거입니다. 

다만 **“CM → distribution shift를 학습함 → test 악화”라는 인과관계 자체가 직접 측정된 것은 아닙니다.**

---

# J. 계산 효율

Appendix Table 6의 실제 seconds/epoch입니다.

| Dataset  |  ST-MLP |  가장 빠른 다른 STGNN | ST-MLP 속도 우위 |
| -------- | ------: | --------------: | -----------: |
| PEMS-BAY | 18.22 s | StemGNN 28.51 s |  약 **1.56×** |
| PEMS04   |  5.34 s |  StemGNN 7.85 s |  약 **1.47×** |
| PEMS07   | 19.05 s | StemGNN 54.76 s |  약 **2.87×** |
| PEMS08   |  4.23 s |  StemGNN 6.87 s |  약 **1.62×** |

가장 느린 DCRNN/DGCRN 계열과 비교하면 일부 데이터셋에서는 **10배 이상** 차이가 납니다. 

다만 이것은 **training time/epoch**입니다.

* inference latency
* peak GPU memory
* FLOPs
* parameter count
* energy consumption
* 동일 accuracy까지 도달하는 총 wall-clock time

까지 비교한 것은 아닙니다.

---

# 3. 저자가 직접 보고한 결과와 제 해석의 분리

| 주제             | **저자가 직접 보고한 내용**                                        | **제 해석**                                                                           |
| -------------- | -------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| 연구 문제          | 복잡한 STGNN은 계산량 대비 accuracy gain이 작을 수 있음                 | traffic forecasting에서 architecture complexity보다 inductive bias가 더 중요할 수 있음         |
| CI             | CI가 distribution shift와 prediction variance 문제를 완화할 수 있음 | CI는 일종의 강한 regularization으로도 볼 수 있음                                                |
| Graph 이용       | predefined graph를 embedding으로 넣으면 CI를 유지할 수 있음           | raw observation mixing과 structural prior injection을 분리한 것이 핵심적 설계                  |
| Cascading      | 단순 concatenation보다 cascade가 좋음                           | 효과 자체는 존재하지만 Fig.4 complete 기준 차이는 14.10→14.03으로 크지 않음                             |
| $E_{td}$       | 가장 중요한 embedding                                         | 해당 PEMS 환경에서는 강한 daily periodicity가 복잡한 graph interaction보다 더 중요할 가능성              |
| CI vs CM       | CM은 train error가 낮지만 test error가 큼                       | capacity 증가가 spurious cross-node correlation까지 학습했을 가능성이 있음                        |
| Efficiency     | ST-MLP가 모든 비교 STGNN보다 epoch당 빠름                          | 실제 deployment superiority를 말하려면 inference/memory 결과가 추가로 필요                        |
| Generalization | CI가 robustness 향상에 도움이 됨                                 | 동일 도시·센서의 future split에서의 generalization이지 cross-city/zero-shot generalization은 아님 |



---

# 4. 통계적으로 취약한 부분 및 직접 비교하기 어려운 수치

## 4-1. 반복 실험 분산이 없습니다

Table 1과 Table 2는 모두 point estimate입니다.

즉

$$
\bar e \pm s
$$

형태의 mean ± standard deviation이 없습니다.

random seed를 $R$회 바꿨다면 최소한

```math
\bar e
=
\frac1R\sum_{r=1}^{R} e_r
```

와

```math
s
=
\sqrt{
\frac{1}{R-1}
\sum_{r=1}^{R}(e_r-\bar e)^2
}
```

를 보고해야 0.1%~1% 수준의 작은 차이가 실제 안정적인 개선인지 판단할 수 있습니다.

PEMS07에서 ST-MLP와 STID의 평균 MAE 차이는

$$
19.59-19.51=0.08
$$

밖에 되지 않습니다.

따라서 **통계적 유의성은 논문만으로 판단할 수 없습니다.**

---

## 4-2. CI vs CM 실험의 confounding

CM variant는 channel-mixing linear layer를 추가합니다.

그러면 CI와 CM의 차이는 동시에

1. channel interaction 유무,
2. parameter 수,
3. model capacity,
4. regularization strength

차이가 될 수 있습니다.

따라서

$$
\text{CM worse test}
\Rightarrow
\text{channel mixing 자체가 원인}
$$

이라는 결론은 완전히 식별되지 않습니다.

보다 엄밀하게는 **parameter-matched CI/CM** 실험이 필요합니다.

---

## 4-3. Distribution shift를 그림으로만 보여줍니다

Figure 1은 shift가 존재함을 시각적으로 제시하지만,

$$
D_{\mathrm{KL}}(P_{\text{train}}\Vert P_{\text{test}})
$$

Wasserstein distance,

$$
\text{MMD}
(P_{\text{train}},P_{\text{test}})
$$

Population Stability Index 등의 정량값이 없습니다.

그러므로 “shift가 큰 dataset일수록 CI gain이 커지는가?”를 직접 검증할 수 없습니다.

---

## 4-4. Cascaded 구조의 개선 폭은 일부 매우 작습니다

Fig.4 PEMS08:

* complete cascaded: **14.03**
* complete non-cascaded: **14.10**

상대 개선은 약

$$
\frac{14.10-14.03}{14.10}
\approx0.50\%
$$

입니다.

통계적 변동범위가 보고되지 않았으므로 이 차이가 seed variation보다 큰지 알 수 없습니다.

---

## 4-5. Ablation의 dataset 범위

저자들은 다른 dataset에서도 비슷한 결론이라고 설명하지만, 논문에 상세히 제시된 Fig.4 ablation은 주로 **PEMS08**입니다.

따라서

> $E_{td}$가 모든 도시와 모든 교통 환경에서 가장 중요한 feature다

라고 일반화하는 것은 근거보다 강한 주장입니다.

---

## 4-6. PEMS dataset의 domain diversity

네 데이터셋 모두 highway sensor 계열이며 상당수가 California PeMS에 기반합니다.

이는

* 다른 국가,
* 다른 도로구조,
* urban arterial,
* 사고·이벤트,
* 날씨,
* 새로운 sensor,
* topology 변경

환경을 충분히 대표하지 않습니다.

따라서 현재 증명된 generalization은 주로 **in-domain future generalization**입니다.

---

## 4-7. $C=1$

논문의 일반 정의는

$$
X\in\mathbb{R}^{N\times T\times C}
$$

이지만 실제 formulation에서는

$$
C=1
$$

로 둡니다.

따라서 speed + flow + occupancy처럼 **node별 다변량 feature가 동시에 존재할 때 CI를 어떻게 정의하고 적용할지**가 충분히 검증되지 않았습니다.

---

## 4-8. Efficiency 수치가 완전히 동일 의미는 아닙니다

Table 6은 **seconds/epoch**입니다.

하지만 모델마다

* convergence epoch
* parameter count
* FLOPs
* GPU utilization

이 다릅니다.

따라서 이것을 곧바로 “실제 시스템에서 N배 더 효율적이다”라고 해석해서는 안 됩니다.

---

## 4-9. Table 1 caption의 표기 불일치

본문 열은 **MAE, RMSE, MAPE**인데 Table 1 caption에는 “MAE, MSE and MAPE”라고 쓰여 있습니다.

실제 표의 숫자와 column heading을 보면 **RMSE가 맞으며 caption의 MSE는 오기일 가능성이 높습니다.** 

---

# 5. 이 문서가 답하지 않는 중요한 질문

1. **CI의 성능 향상이 실제 distribution shift 때문인가, 단순한 capacity reduction/regularization 때문인가?**
2. shift의 크기와 CI–CM 성능차 사이에 정량적 상관관계가 있는가?
3. 전혀 보지 못한 도시 또는 새로운 sensor에 대해서도 CI가 유리한가?
4. 사고·기상·공사·공휴일처럼 spatial interaction 자체가 순간적으로 강해지는 경우에도 CI가 좋은가?
5. static adjacency matrix가 오래되거나 잘못된 경우 $AB_{sp}$는 얼마나 robust한가?
6. $C>1$인 node-wise multivariate traffic에서는 channel을 sensor로 정의해야 하는가, variable로 정의해야 하는가?
7. missing sensor나 sensor failure 상황에서도 성능을 유지하는가?
8. 여러 random seed에서 ST-MLP의 작은 superiority가 유지되는가?
9. CI와 CM을 **동일 parameter budget**으로 맞추면 Table 2의 결론이 유지되는가?
10. inference latency, GPU memory, FLOPs, energy 측면에서도 efficiency 우위가 유지되는가?
11. 논문에서 명시적으로 수식화하지 않은 **training objective/loss function**은 정확히 무엇이며 baseline 모두 동일 loss였는가?
12. dynamic graph나 online adaptation을 도입하더라도 CI의 robustness 이점을 유지할 수 있는가?

---

# 6. 가장 중요한 그림 5개 해석

## ① Figure 1 — Train/Test distribution shift, p.1

**저자 보고:** Z-score normalization을 했는데도 PEMS07과 PEMS08에서 train과 test distribution이 달라집니다.

**해석:** 이 그림은 사실 ST-MLP 전체 연구의 출발점입니다. 단순히 normalization을 한 번 수행한다고

$$
P_{\text{train}}(X)=P_{\text{test}}(X)
$$

가 되는 것이 아님을 보여줍니다.

다만 정량 divergence가 없기 때문에 **illustrative evidence**이지 statistical proof는 아닙니다.

---

## ② Figure 2(a) — 전체 ST-MLP architecture, p.3

가장 중요한 구조는

$$
E_t
\rightarrow
\text{MLP}_A
\rightarrow
+E_s
\rightarrow
\text{MLP}_B
\rightarrow
+E_d
\rightarrow
\text{MLP}_C
\rightarrow
\text{Linear}
\rightarrow
\hat X
$$

입니다.

이 그림의 핵심은 **복잡한 GNN을 제거했다는 것 자체보다, 공간 정보를 동적 channel mixing에서 embedding conditioning 문제로 바꿨다는 것**입니다.

이 방식 덕분에 각 node prediction이 독립적인 형태를 유지하면서도 node identity와 graph topology는 알 수 있습니다.

---

## ③ Figure 2(c)+(d) — Temporal/Spatial embedding, p.3

Figure 2(c)는 clock information을 codebook에서 가져오는 과정이고, Figure 2(d)는

$$
A B_{sp}
$$

를 통해 graph topology를 embedding으로 변환하는 과정을 보여줍니다.

제가 보기에는 이 부분이 ST-MLP에서 가장 중요한 기술적 아이디어입니다.

즉,

$$
\boxed{\text{raw cross-node signal mixing}}
$$

을 제거하면서

$$
\boxed{\text{spatial structural information}}
$$

은 제거하지 않았습니다.

그래서 CI의 robustness와 graph prior를 동시에 가져갈 수 있습니다.

---

## ④ Figure 3 — Normalized training time, p.5

ST-MLP bar가 모든 데이터셋에서 가장 짧습니다.

Appendix Table 6의 absolute value와 함께 보면 PEMS07에서

$$
19.05 \text{ sec/epoch}
$$

이고 비교 대상 중 가장 빠른 StemGNN도

$$
54.76
$$

초입니다.

즉 약

$$
\frac{54.76}{19.05}\approx2.87
$$

배 차이가 납니다.

따라서 단순 architecture의 계산적 장점 자체는 상당히 명확합니다.

---

## ⑤ Figure 4 — Ablation, p.7

PEMS08 test MAE:

* Full cascade: **14.03**
* No cascade: **14.10**
* Without $E_{td}$: **15.17**
* Without $E_{dw}$: **14.11**
* Without $E_{sp}$: **14.11**
* Without $E_{su}$: **14.08**

가장 강하게 보이는 사실은 cascade보다 오히려 **Time-in-Day embedding의 중요성**입니다.

$E_{td}$ 제거 시

$$
\frac{15.17-14.03}{14.03}
\times100
\approx8.13\%
$$

악화되지만 cascade 제거는 약 0.5% 수준입니다.

따라서 이 Figure를 엄밀하게 해석하면

> “cascade가 중요하다”

보다

> **“traffic periodicity에 대한 적절한 inductive bias가 훨씬 중요하다”**

가 더 강하게 뒷받침됩니다.



---

# 7. 일반화 성능 관점에서 이 논문의 의미

ST-MLP가 일반화 측면에서 흥미로운 이유는 **모델이 알아야 할 정보와 굳이 직접 학습하지 않아도 되는 관계를 분리했다는 점**입니다.

복잡한 CM 모델은

```math
\hat y_i
=
f(x_1,x_2,\ldots,x_N)
```

처럼 모든 node correlation을 학습할 수 있습니다.

반면 CI는

```math
\hat y_i
=
f(x_i,z_i)
```

에 가깝고, $z_i$가

* time-of-day,
* day-of-week,
* node identity,
* graph-derived embedding

을 포함합니다.

즉 모델에게 필요한 context는 주면서 **관측된 cross-node correlation 자체를 자유롭게 외우는 능력은 제한**합니다.

이는 Han et al.이 설명한 CI의 capacity–robustness trade-off와 일치하는 방향입니다. CI는 capacity를 희생하지만 distribution drift에서 robust할 수 있습니다. ([arXiv][4])

---

# 7-1. 일반화 성능을 더 높일 수 있는가?

가능성이 높습니다. 그러나 **순수 CI를 무조건 더 강화하는 것**이 반드시 최선은 아닙니다.

향후 가장 합리적인 구조는 다음과 같습니다.

## ① CI backbone + 작은 residual channel mixer

기본 예측을

$$
\hat X_{\mathrm{CI}}
$$

로 만들고, cross-node information은 작은 residual만 허용합니다.

```math
\boxed{
\hat X
=
\hat X_{\mathrm{CI}}
+
\lambda(X)\,R_{\theta}(X)
}
```

여기서

* $R_\theta$: channel interaction을 학습하는 작은 residual network,
* $\lambda(X)$: 현재 regime에서 channel mixing을 얼마나 믿을지 결정하는 gate입니다.

그리고

$$
\lambda\approx0
$$

을 기본값으로 강하게 regularize합니다.

이렇게 하면 안정적인 시기에는 CI를 유지하고, 사고·정체처럼 실제 공간 상호작용이 중요한 시기만 CM을 사용하게 할 수 있습니다.

이는 CI와 channel-dependent 방식을 적대적인 둘 중 하나로 선택하지 않고 **adaptive bias–variance trade-off**로 바꾸는 접근입니다.

---

## ② RevIN과 결합

입력 instance $X$의 평균과 표준편차를

$$
\mu_X,\qquad \sigma_X
$$

라 하고

```math
\tilde X
=
\frac{X-\mu_X}{\sigma_X+\epsilon}
```

로 normalize한 뒤 예측하고 출력에서 inverse transform을 수행하는 방식입니다.

RevIN은 distribution shift에 대응하기 위해 이 아이디어를 명시적으로 제안했습니다. ([ML Anthology][3])

ST-MLP의 CI는 **cross-channel spurious correlation**을 줄이고,

RevIN은 **각 시계열의 time-varying level/scale shift**를 줄이므로 서로 보완적입니다.

---

## ③ Static graph보다 stable graph를 학습

ST-MLP의

$$
E_{sp}=AB_{sp}
$$

에서 $A$가 항상 정답이라고 가정하지 않고

```math
A^\ast
=
A+\Delta A
```

로 둘 수 있습니다.

단,

$$
\|\Delta A\|_1
$$

또는 low-rank constraint를 강하게 적용하여 graph가 시계열 noise를 따라 움직이지 않도록 해야 합니다.

핵심은 **dynamic raw channel mixing**이 아니라 **안정적인 structural correction**만 학습하는 것입니다.

---

## ④ Cross-city validation

현재 논문의 가장 큰 일반화 한계입니다.

다음과 같은 검증이 필요합니다.

$$
\text{Train: City A/B/C}
\rightarrow
\text{Test: unseen City D}
$$

또는

$$
\text{Train: 2022}
\rightarrow
\text{Validation: early 2023}
\rightarrow
\text{Test: late 2023}
$$

처럼 시간적으로 완전히 미래 구간을 사용해야 합니다.

2024–2025년 연구들은 이 문제가 실제로 중요한 다음 단계임을 보여줍니다.

---

# 8. 2020년 이후 최신 관련 연구 비교

| 연도   | 연구                            | 핵심 접근                                               | 일반화 관점                                    | ST-MLP와의 관계                                   |
| ---- | ----------------------------- | --------------------------------------------------- | ----------------------------------------- | --------------------------------------------- |
| 2020 | **MTGNN**                     | graph structure 자체를 학습 + graph/temporal convolution | 높은 capacity                               | ST-MLP가 단순화하려는 대표적 방향                         |
| 2021 | **ST-Norm**                   | spatial/temporal normalization                      | nonstationarity 완화                        | ST-MLP의 shift 문제의식과 보완적                       |
| 2022 | **RevIN**                     | instance 단위 normalization/denormalization           | distribution shift 직접 대응                  | CI와 함께 사용 가능                                  |
| 2022 | **STID**                      | spatial/time identity + MLP                         | 간단한 모델의 strong baseline                   | ST-MLP의 직접적인 사상적 전조                           |
| 2023 | **PatchTST**                  | patching + channel independence                     | long-history + robust CI                  | ST-MLP가 CI 아이디어를 spatial forecasting으로 확장     |
| 2023 | **Han et al. CI/CD analysis** | CI vs CD capacity–robustness 이론/실험                  | CI의 robustness 근거                         | ST-MLP의 CI 논리적 근거                             |
| 2023 | **STAEformer**                | adaptive ST embedding + vanilla Transformer         | embedding이 architecture보다 중요              | ST-MLP와 매우 유사한 연구 메시지                         |
| 2023 | **ST-MLP**                    | CI + graph embedding + cascaded MLP                 | in-domain temporal generalization         | 본 논문                                          |
| 2024 | **PreMixer**                  | MLP-Mixer + masked pretraining                      | 대규모 history에서 transferable representation | ST-MLP의 단순 MLP 방향을 pretraining으로 확대           |
| 2024 | **OpenCity**                  | large-scale pretraining + Transformer/GNN           | unseen city zero-shot                     | ST-MLP보다 훨씬 강한 cross-domain generalization 목표 |
| 2025 | **Cross-IDR**                 | cross-city distribution rectification               | target-city distribution adaptation       | ST-MLP가 직접 풀지 못한 cross-city shift 처리          |
| 2025 | **SLPF**                      | rank embedding + spatial transfer matrix            | unseen/unsensed node generalization       | CI와 별개로 spatial distribution shift를 직접 처리     |
| 2025 | **Fed-CI**                    | channel-independent federated traffic forecasting   | privacy + local generalization            | CI를 traffic system의 분산학습까지 확장한 흐름             |
| 2026 | **ST-MLP journal version**    | CI 기반 cascaded spatio-temporal MLP                  | 논문 아이디어의 peer-reviewed 확장                 | 2023 preprint의 후속 정식 저널화                      |

MTGNN은 latent variable 관계를 graph learning으로 적극적으로 학습하는 대표적인 고용량 접근입니다. ([arXiv][5])

ST-Norm은 spatial/temporal normalization을 별도로 설계해 multivariate dynamics를 안정화하는 방향을 제시했습니다. ([DOI][6])

STID는 2022년 이미 단순 MLP와 spatial/temporal identity만으로 강력한 성능을 얻을 수 있다고 보였으며, **ST-MLP의 “복잡성이 핵심은 아니다”라는 메시지와 직접 연결되는 연구 흐름**입니다. ([arXiv][2])

PatchTST는 channel independence를 Transformer에 적용해 각 channel을 독립적으로 처리하면서 parameter를 공유했고, CI가 단순한 MLP 트릭이 아니라 다른 architecture에도 적용 가능한 inductive bias임을 보여주었습니다. ([arXiv][7])

STAEformer 역시 복잡한 새로운 GNN을 만드는 대신 **좋은 spatio-temporal embedding + 비교적 표준적인 Transformer**로 SOTA를 달성했다는 논리이며, ST-MLP와 거의 같은 시기 독립적으로 “representation 설계가 architecture complexity보다 중요할 수 있다”는 방향을 강화했습니다. 이 논문은 CIKM 2023에 출판되었습니다. ([DOI][8])

PreMixer는 이 방향을 한 단계 더 발전시켜 MLP 기반 forecasting에 **masked pretraining**을 추가하고 대규모 traffic network 확장성을 목표로 합니다. 다만 현재 제가 확인한 주 출처는 2024 arXiv preprint이므로 peer-reviewed ST-MLP/Table 1과 수치를 직접 우열 비교하는 것은 피해야 합니다. ([arXiv][9])

OpenCity는 문제 정의를 더 크게 확장해, 특정 도시에서 잘 맞는 것보다 **새로운 도시에서 fine-tuning 없이 zero-shot forecasting**하는 것을 목표로 합니다. large-scale heterogeneous traffic data에서 pretraining하여 cross-city generalization을 확보하려는 방향입니다. ([arXiv][10])

Cross-IDR는 2025년 Knowledge-Based Systems에서 source-city knowledge를 그대로 transfer하는 것이 아니라 **target distribution 중심으로 representation을 rectification**해야 한다고 주장합니다. 이것은 ST-MLP가 Figure 1에서 제기했지만 직접 해결하지 않은 cross-domain distribution shift를 보다 명시적으로 다룬 연구입니다. ([ScienceDirect][11])

SLPF는 2025년 TMLR에서 **훈련 시 관측되지 않았거나 sensor가 없는 location**의 long-term prediction을 다루고 spatial transfer matrix를 사용해 sensed→unsensed location shift를 처리합니다. ST-MLP보다 훨씬 엄격한 spatial generalization 문제입니다. ([OpenReview][12])

또한 2025년의 _Channel-Independent Federated Traffic Prediction_은 CI를 federated traffic forecasting에까지 적용해 local information만으로 prediction하면서 communication cost를 크게 줄이는 방향을 제시했습니다. 다만 이는 현재 **arXiv preprint**이므로 결과를 정식 peer-reviewed 연구와 동일한 확정도로 취급하면 안 됩니다. ([arXiv][13])

---

# 8-1. 이 비교에서 보이는 연구 흐름

2020년대 초반은

$$
\text{더 정교한 graph}
+
\text{더 복잡한 temporal network}
$$

방향이 강했습니다.

이후 STID, DLinear, PatchTST, ST-MLP, STAEformer 등이 던진 메시지는 달라집니다.

$$
\boxed{
\text{복잡한 architecture}
\not\Rightarrow
\text{좋은 forecasting}
}
$$

대신

$$
\boxed{
\text{좋은 representation}
+
\text{적절한 inductive bias}
+
\text{distribution robustness}
}
$$

가 중요하다는 것입니다.

> **용어 — Inductive bias:** 모델이 데이터만 보기 전에 이미 갖고 있는 구조적 가정입니다. 예를 들어 ST-MLP의 “각 sensor의 동적 관측을 직접 섞지 말자”가 하나의 inductive bias입니다.

그리고 2024–2026년으로 갈수록 문제가 다시 변합니다.

단순히

> “PEMS08 test MAE가 얼마인가?”

가 아니라

> “새 도시에서도 작동하는가?”,
> “새로운 센서에서도 작동하는가?”,
> “장기간 distribution drift에서도 버티는가?”,
> “pretraining으로 transfer할 수 있는가?”

가 핵심 평가 기준으로 이동하고 있습니다. OpenCity, Cross-IDR, SLPF가 그 흐름을 잘 보여줍니다. ([arXiv][10])

---

# 8-2. ST-MLP가 이후 연구에 미치는 영향에 대한 평가

여기서는 **확인 가능한 사실과 추론을 분리해야 합니다.**

### 확인 가능한 사실

2023년 ST-MLP preprint의 핵심 CI+MLP 연구는 이후 확장되어 **2026 IEEE Transactions on Intelligent Transportation Systems의 정식 journal article**로 출판되었습니다. 따라서 2023 아이디어가 일회성 preprint에서 끝난 것은 아닙니다. ([Scholars@Duke][14])

또한 2025년에는 traffic forecasting에 CI를 명시적으로 사용하는 Fed-CI 같은 후속 방향도 등장했습니다. ([arXiv][13])

### 제 해석

ST-MLP의 가장 큰 영향은 특정 MLP block 자체라기보다 다음 관점에 있습니다.

$$
\boxed{
\text{Spatial dependency를 이용하는 것}
\neq
\text{매번 raw node signals를 직접 mixing하는 것}
}
$$

즉 spatial prior를

* graph propagation,
* attention,

으로만 구현해야 한다는 고정관념 대신

* identity embedding,
* graph-conditioned embedding,
* positional encoding,
* pretraining representation

으로 옮길 수 있다는 것입니다.

이 관점은 STID, STAEformer, PreMixer와 같은 연구 흐름과 매우 잘 맞습니다. 다만 이 모델들이 ST-MLP 때문에 등장했다고 **인과적으로 주장할 근거는 없으므로**, 이를 “직접적인 영향”이 아니라 **동시대 연구 패러다임의 수렴**으로 보는 것이 정확합니다.

---

# 9. 결론과 후속 연구

## 저자들이 직접 제시한 결론

저자들은 ST-MLP가

1. 단순 MLP 구조,
2. Channel Independence,
3. temporal/spatial/data embedding,
4. cascaded fusion

만으로 강한 traffic forecasting 성능과 높은 computational efficiency를 얻었다고 결론 내립니다.

후속 연구로는 특히

* 더 다양한 spatio-temporal dataset으로 확장,
* CI 기반 robustness 강화,
* CI의 빠른 convergence,
* over-smoothing 완화 가능성

을 조사할 필요가 있다고 명시합니다. p.7 Conclusion and Future Work에 해당합니다. 

> **용어 — Over-smoothing:** graph layer를 여러 번 통과하면서 서로 다른 node의 representation이 지나치게 비슷해져 node 고유 특성을 잃는 현상입니다.

---

# 제가 제안하는 가장 중요한 후속 연구 방향

일반화 성능만을 목표로 한다면 다음 순서가 가장 가치가 높습니다.

### 1순위 — CI와 CM을 binary choice로 보지 않기

```math
\hat X
=
\hat X_{CI}
+
g(X)\odot\hat R_{CM}
```

로 구성하고 $g(X)$에 강한 sparsity/regularization을 적용합니다.

정상 traffic에서는

$$
g(X)\approx0
$$

으로 CI를 유지하고, 실제 spatial propagation이 필요한 congestion/event regime에서만

$$
g(X)>0
$$

가 되게 합니다.

---

### 2순위 — Distribution shift를 실제 측정하기

train/test window마다

$$
D_{\mathrm{Wasserstein}},
\quad
\text{MMD},
\quad
\Delta\mu,
\quad
\Delta\sigma,
\quad
\Delta\rho
$$

를 계산합니다.

그다음

```math
\text{CI gain}
=
e_{\text{CM}}-e_{\text{CI}}
```

과 shift magnitude 간 관계를 검증해야 합니다.

이렇게 해야 “CI가 왜 좋은가?”라는 논문의 핵심 주장을 실제로 검증할 수 있습니다.

---

### 3순위 — 반드시 rolling-origin test

단일 70/10/20 split이 아니라

$$
D_1\rightarrow D_2,
\qquad
D_{1:2}\rightarrow D_3,
\qquad
D_{1:3}\rightarrow D_4
$$

처럼 여러 미래 구간에서 반복 평가해야 합니다.

그리고 각 모델에 대해 최소한

$$
\text{mean}\pm\text{std}
$$

를 보고해야 합니다.

작은 성능차에는 bootstrap confidence interval이나 forecasting error 차이에 대한 Diebold–Mariano 계열 검정을 추가하는 것이 좋습니다.

---

### 4순위 — Cross-city / unseen-node 평가

진정한 일반화라면 최소 두 종류를 분리해야 합니다.

**Temporal generalization**

$$
\text{same sensors, future period}
$$

**Spatial/domain generalization**

$$
\text{new sensors / new city}
$$

ST-MLP의 원 논문은 전자에 강한 증거를 주지만 후자는 거의 검증하지 않았습니다.

---

### 5순위 — Pretraining

ST-MLP의 단순성과 PreMixer/OpenCity의 pretraining을 결합하는 방향이 유망합니다.

예를 들어 historical traffic patches에 대해 masked reconstruction:

```math
\mathcal{L}_{pre}
=
\sum_{(i,t)\in\mathcal M}
\left|
X_{i,t}
-
\hat X_{i,t}
\right|
```

을 먼저 학습하고 ST-MLP forecasting을 fine-tune할 수 있습니다.

이렇게 하면 small-data city에서도 sensor-specific pattern을 처음부터 모두 학습하지 않아도 됩니다. ([arXiv][9])

---

# 최종 평가

이 논문의 가장 중요한 기여를 한 문장으로 요약하면 다음과 같습니다.

$$
\boxed{
\text{“공간 정보를 이용하기 위해 반드시 동적 센서값을 서로 섞을 필요는 없다.”}
}
$$

ST-MLP는 graph topology와 node identity를 **embedding으로 보존하면서 raw traffic signals의 cross-node mixing을 억제**해, 비교적 적은 계산량으로 좋은 future-test 성능을 얻었습니다. 특히 Table 2의 CI/CM 결과는 generalization 관점에서 매우 흥미롭습니다.

그러나 아직

$$
\boxed{
\text{CI}
\Rightarrow
\text{distribution-shift robustness}
}
$$

가 완전히 입증된 것은 아닙니다. 이를 강하게 증명하려면 **parameter-matched CI/CM, quantitative shift measurement, repeated seeds, rolling future tests, cross-city/unseen-node evaluation**이 필요합니다.

따라서 앞으로의 가장 유망한 연구 방향은 **순수 ST-MLP를 더 깊게 만드는 것**이 아니라,

$$
\boxed{
\text{CI backbone}
+
\text{sparse/adaptive spatial residual}
+
\text{shift normalization}
+
\text{pretraining}
+
\text{strict OOD evaluation}
}
$$

으로 발전시키는 것입니다.

---

## 참고한 논문·사이트 및 참고자료 제목

**주 분석 자료**

* **Wang et al. — *ST-MLP: A Cascaded Spatio-Temporal Linear Framework with Channel-Independence Strategy for Traffic Forecasting***, arXiv:2308.07496, 2023 — 첨부 PDF 및 arXiv.  ([arXiv][1])
* **Wang et al. — *Channel-Independence for Traffic Forecasting: A Cascaded Spatio-Temporal MLP Framework***, IEEE Transactions on Intelligent Transportation Systems, 2026. ([Scholars@Duke][14])

**비교 및 이론적 근거**

* Wu et al. — ***Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks***, 2020. ([arXiv][5])
* Deng et al. — ***ST-Norm: Spatial and Temporal Normalization for Multi-variate Time Series Forecasting***, KDD 2021. ([DOI][6])
* Kim et al. — ***Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift***, ICLR 2022. ([ML Anthology][3])
* Shao et al. — ***Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting***, CIKM 2022. ([arXiv][2])
* Nie et al. — ***A Time Series is Worth 64 Words: Long-term Forecasting with Transformers***, ICLR 2023. ([OpenReview][15])
* Han et al. — ***The Capacity and Robustness Trade-off: Revisiting the Channel Independent Strategy for Multivariate Time Series Forecasting***, 2023. ([arXiv][4])
* Liu et al. — ***Do We Really Need Graph Neural Networks for Traffic Forecasting?***, 2023. ([arXiv][16])
* Liu et al. — ***Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA for Traffic Forecasting***, CIKM 2023. ([DOI][8])
* Zhang et al. — ***PreMixer: MLP-Based Pre-training Enhanced MLP-Mixers for Large-scale Traffic Forecasting***, 2024 preprint. ([arXiv][9])
* Li et al. — ***OpenCity: Open Spatio-Temporal Foundation Models for Traffic Prediction***, 2024. ([arXiv][10])
* Yang et al. — ***Cross-city Transfer Learning for Traffic Forecasting via Incremental Distribution Rectification***, Knowledge-Based Systems, 2025. ([ScienceDirect][11])
* Liu et al. — ***Spatio-temporal Partial Sensing Forecast of Long-term Traffic***, Transactions on Machine Learning Research, 2025. ([OpenReview][12])
* Zhang et al. — ***Channel-Independent Federated Traffic Prediction***, arXiv:2508.04517, 2025 preprint. ([arXiv][13])

다음 단계로는 이 분석을 기반으로 **ST-MLP를 직접 구현할 수 있을 정도로 Figure 2를 tensor dimension 단위로 분해하고, PyTorch architecture + leakage 없는 실험 설계 + “CI → 일반화 향상”을 검증하는 ablation protocol**까지 연결하는 것이 가장 유용합니다.

[1]: https://arxiv.org/abs/2308.07496?utm_source=chatgpt.com "ST-MLP: A Cascaded Spatio-Temporal Linear Framework with Channel-Independence Strategy for Traffic Forecasting"
[2]: https://arxiv.org/abs/2208.05233?utm_source=chatgpt.com "Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting"
[3]: https://mlanthology.org/iclr/2022/kim2022iclr-reversible/?utm_source=chatgpt.com "Reversible Instance Normalization for Accurate Time-Series Forecasting Against Distribution Shift | ML Anthology"
[4]: https://arxiv.org/abs/2304.05206?utm_source=chatgpt.com "The Capacity and Robustness Trade-off: Revisiting the Channel Independent Strategy for Multivariate Time Series Forecasting"
[5]: https://arxiv.org/abs/2005.11650?utm_source=chatgpt.com "Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks"
[6]: https://doi.org/10.1145/3447548.3467330?utm_source=chatgpt.com "ST-Norm | Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining"
[7]: https://arxiv.org/abs/2211.14730?utm_source=chatgpt.com "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"
[8]: https://doi.org/10.1145/3583780.3615160?utm_source=chatgpt.com "Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA for Traffic Forecasting | Proceedings of the 32nd ACM International Conference on Information and Knowledge Management"
[9]: https://arxiv.org/abs/2412.13607?utm_source=chatgpt.com "PreMixer: MLP-Based Pre-training Enhanced MLP-Mixers for Large-scale Traffic Forecasting"
[10]: https://arxiv.org/abs/2408.10269?utm_source=chatgpt.com "OpenCity: Open Spatio-Temporal Foundation Models for Traffic Prediction"
[11]: https://www.sciencedirect.com/science/article/pii/S0950705125003831?utm_source=chatgpt.com "Cross-city transfer learning for traffic forecasting via incremental distribution rectification - ScienceDirect"
[12]: https://openreview.net/pdf?id=Ff08aPjVjD&utm_source=chatgpt.com "Published in Transactions on Machine Learning Research (08/2025)"
[13]: https://arxiv.org/abs/2508.04517?utm_source=chatgpt.com "Channel-Independent Federated Traffic Prediction"
[14]: https://scholars.duke.edu/publication/1817933?utm_source=chatgpt.com "Scholars@Duke publication: Channel-Independence for Traffic Forecasting: A Cascaded Spatio-Temporal MLP Framework"
[15]: https://openreview.net/pdf?id=Jbdc0vTOcol&utm_source=chatgpt.com "Published as a conference paper at ICLR 2023"
[16]: https://arxiv.org/abs/2301.12603?utm_source=chatgpt.com "Do We Really Need Graph Neural Networks for Traffic Forecasting?"
