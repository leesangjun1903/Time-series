# Chapter 3. Principal Components Analysis

> 교재 범위: Chapter 3, pp. 61–104.  
> 핵심 주제: PCA, sample principal components, covariance vs correlation matrix, component selection, scores, biplot, sample size, canonical correlation analysis(CCA).

## 1. Executive Summary — 10문장 이내

1. PCA는 상관된 $q$개 원변수를 서로 직교하는 새로운 선형결합으로 바꾸고, 가능한 한 적은 성분으로 전체 변동을 설명하려는 차원축소 방법입니다.
2. 첫 번째 주성분은 $a_1^\top x$의 분산을 최대화하는 방향이며, 그 해는 공분산행렬 또는 상관행렬의 가장 큰 고유값에 대응하는 고유벡터입니다.
3. 두 번째 이후 성분은 앞선 성분과 무상관이라는 제약 아래 남은 변동을 순차적으로 최대화합니다.
4. 공분산행렬 PCA는 원 단위의 변동 크기를 보존하고, 상관행렬 PCA는 모든 변수를 표준화하여 동일한 척도에서 비교합니다.
5. 성분 수 선택에는 누적 설명분산, Kaiser 기준, scree plot 등 여러 기준이 있지만 교재 예제에서도 서로 다른 답을 내므로 절대 규칙은 없습니다.
6. PCA의 큰 분산은 “예측에 중요한 정보”와 동일하지 않으며, target을 고려하지 않는 unsupervised 변환이라는 점을 기억해야 합니다.
7. CCA는 하나의 변수집합이 아니라 두 변수집합 사이에서 가장 강하게 연결되는 선형결합 쌍을 찾는 PCA의 친척 격 방법입니다.
8. 교재 heptathlon 사례에서 제1주성분은 공식 점수와 $-0.9931$의 매우 높은 상관을 보였지만, 부호는 PCA 고유벡터의 임의성 때문에 의미가 없습니다.
9. 2020년 이후에는 sparse PCA, integrated PCA, online PCA 등 해석성·다중 데이터 통합·streaming 환경을 위한 확장이 활발합니다.
10. 실제 예측 파이프라인에서는 PCA를 전체 데이터에 fit하지 말고 train에서만 fit하며, component 수는 validation 또는 nested CV로 정해야 일반화 성능을 정직하게 평가할 수 있습니다.

## 2. 해결하려는 문제

변수 수 $q$가 커지면 scatterplot matrix조차 해석하기 어려워지고, 변수끼리 강하게 상관되어 사실상 같은 정보를 반복 측정할 수 있습니다. PCA는 원변수 공간의 좌표축을 회전하여 **변동이 가장 큰 방향부터 새로운 축을 정의**합니다.

예를 들어 $q=800$개의 센서가 있지만 실제 공정상태가 10개 내외의 독립적인 변화 방향으로 움직인다면, 800개 원변수보다 10–30개 주성분으로 구조를 표현하는 것이 더 안정적일 수 있습니다.

**용어 설명 — dimension reduction**  
원래 변수 수보다 적은 새로운 변수로 데이터의 중요한 구조를 근사하는 과정입니다. 단순 feature selection과 달리 PCA는 원변수의 선형결합을 새 feature로 만듭니다.

## 3. PCA의 핵심 수학

중심화된 확률벡터 $X\in\mathbb R^q$를 생각하겠습니다. 첫 번째 주성분을

$$
Y_1=a_1^\top X
$$

로 둡니다.

- $X$: $q$개 원변수 벡터입니다.
- $a_1\in\mathbb R^q$: 제1주성분의 weight vector입니다.
- $Y_1$: 첫 번째 주성분입니다.

공분산행렬을 $\Sigma$라 하면

$$
\text{Var}(Y_1)=a_1^\top\Sigma a_1
$$

입니다. $a_1$의 길이를 마음대로 크게 하면 분산을 무한히 키울 수 있으므로

$$
a_1^\top a_1=1
$$

이라는 제약을 둡니다. 따라서 최적화 문제는

$$
\max_{a_1}\;a_1^\top\Sigma a_1
\qquad
\text{subject to }a_1^\top a_1=1
$$

입니다.

Lagrange multiplier $\lambda$를 사용하면

$$
L(a_1,\lambda)
=a_1^\top\Sigma a_1-\lambda(a_1^\top a_1-1)
$$

이고 미분조건은

$$
\Sigma a_1=\lambda a_1
$$

이 됩니다.

즉 $a_1$은 $\Sigma$의 고유벡터이고, 분산을 최대화하려면 가장 큰 고유값 $\lambda_1$에 대응하는 고유벡터를 택합니다.

**용어 설명 — eigenvalue / eigenvector**  
행렬 $A$에 대해 $Av=\lambda v$를 만족하는 방향 $v$가 고유벡터이고, 그 방향이 변환될 때 늘어나는 크기 $\lambda$가 고유값입니다. PCA에서는 고유벡터가 새로운 축의 방향이고 고유값이 그 축의 분산입니다.

## 4. 두 번째 이후 주성분

두 번째 주성분은

$$
Y_2=a_2^\top X
$$

이며 분산을 최대화하되 첫 번째와 무상관이어야 합니다.

$$
\text{Cov}(Y_1,Y_2)=0
$$

대칭 공분산행렬의 고유벡터는 서로 직교할 수 있으므로 $a_2$는 두 번째 큰 고유값 $\lambda_2$의 고유벡터가 됩니다. 이런 방식으로

$$
\lambda_1\ge\lambda_2\ge\cdots\ge\lambda_q\ge0
$$

순서의 축을 얻습니다.

전체 분산은

$$
\sum_{j=1}^{q}\lambda_j=\text{tr}(\Sigma)
$$

입니다.

- $\text{tr}(\Sigma)$: 공분산행렬 대각합, 즉 모든 원변수 분산의 합입니다.

따라서 $j$번째 주성분의 설명분산비는

$$
\text{EVR}_j=\frac{\lambda_j}{\sum_{k=1}^{q}\lambda_k}
$$

이고 첫 $m$개 누적 설명분산비는

$$
\text{CEVR}_m
=\frac{\sum_{j=1}^{m}\lambda_j}{\sum_{k=1}^{q}\lambda_k}
$$

입니다.

## 5. 표본 PCA와 score

실제로는 $\Sigma$를 모르므로 표본 공분산행렬 $S$를 고유분해합니다.

$$
S=V\Lambda V^\top
$$

- $V=[v_1,\ldots,v_q]$: 고유벡터를 열로 모은 행렬입니다.
- $\Lambda=\text{diag}(\lambda_1,\ldots,\lambda_q)$: 고유값 대각행렬입니다.

중심화 데이터 행렬을 $X_c$라 하면 score matrix는

$$
Z=X_cV
$$

입니다.

- $Z_{ij}$: 관측치 $i$의 $j$번째 principal component score입니다.
- $V$의 열벡터: 각 성분을 만드는 원변수 weight입니다.

## 6. Covariance PCA인가, Correlation PCA인가?

### Covariance PCA

원 단위의 변동량을 중요도로 인정합니다. 센서 A의 표준편차가 100이고 센서 B가 0.1이면 A가 PCA를 지배할 수 있습니다. 단위 자체가 물리적 의미를 가지며 “큰 변동이 실제로 중요”할 때 적절합니다.

### Correlation PCA

각 변수를 평균 0, 분산 1로 표준화한 뒤 PCA를 수행하는 것과 같습니다.

$$
z_{ij}=\frac{x_{ij}-\bar x_j}{s_j}
$$

- $s_j$: 변수 $j$의 표본 표준편차입니다.

변수 단위가 서로 다르거나 분산 크기의 차이가 단순 측정단위 때문이면 correlation PCA가 더 타당합니다.

**중요**  
표준화 여부는 preprocessing detail이 아니라 **PCA가 “중요한 방향”이라고 정의하는 기준 자체를 바꾸는 모델 선택**입니다.

## 7. 주성분 수를 어떻게 정하는가?

교재는 여러 기준을 소개합니다.

### 7.1 누적 설명분산

예를 들어 70–90% 정도의 분산을 보존하도록 $m$을 정하는 경험적 방식입니다. 하지만 predictive target에 필요한 약한 방향이 제거될 수 있습니다.

### 7.2 평균 고유값 기준

상관행렬 PCA에서는 전체 고유값 평균이 1이므로

$$
\lambda_j>1
$$

인 성분을 유지하는 Kaiser rule이 흔히 사용됩니다. 교재는 Jolliffe가 더 완화된 $0.7$ 기준을 제안했다는 점도 언급합니다.

### 7.3 Scree plot

고유값을 큰 순서로 그려 급격한 감소 후 완만해지는 `elbow`를 찾습니다.

**용어 설명 — scree plot**  
성분 번호에 따른 고유값의 감소를 그린 그림입니다. 산비탈 아래 쌓인 돌무더기(scree)처럼 완만해지는 지점 이후 성분을 noise에 가깝다고 보는 직관입니다.

### 교재의 중요한 교훈

Blood chemistry 예제에서는 기준에 따라 3, 4, 7개 등 서로 다른 성분 수가 제안됩니다. 즉 **component number는 데이터와 목적에 의존하는 선택**입니다.

## 8. Reconstruction 관점

첫 $m$개 성분만 사용하면 원데이터를 저차원 근사할 수 있습니다.

$$
\hat X_c=Z_mV_m^\top
$$

- $V_m$: 첫 $m$개 고유벡터입니다.
- $Z_m=X_cV_m$: 첫 $m$개 score입니다.
- $\hat X_c$: 중심화 원데이터의 rank-$m$ 근사입니다.

PCA는 squared reconstruction error를 최소화하는 최적 rank-$m$ 선형 근사와 연결됩니다. 따라서 차원축소는 “정보 삭제”가 아니라 **고유값이 작은 방향을 버리는 low-rank approximation**입니다.

## 9. Biplot

Biplot은 관측치 score와 변수 loading 정보를 같은 저차원 그림에 표시합니다. 가까운 관측치는 선택한 PCA 공간에서 유사하고, 같은 방향을 가리키는 변수 화살표는 양의 관련성을 가질 가능성이 큽니다.

단, biplot은 첫 2개 정도의 성분에 정보를 투영하므로 전체 데이터 구조를 완전히 나타내지 않습니다. 첫 두 성분 설명분산이 낮다면 해석은 특히 조심해야 합니다.

## 10. PCA와 예측의 관계

PCA는 target $y$를 보지 않습니다. 따라서

$$
\text{large }\text{Var}(Xv)
\not\Rightarrow
\text{large predictive information about }y
$$

입니다.

예를 들어 target을 결정하는 신호가 전체 분산은 작지만 안정적인 센서 조합에 존재한다면 PCA가 해당 방향을 후순위로 밀어버릴 수 있습니다. 이 때문에 supervised 목적에서는 PLS, supervised PCA, regularized regression 등을 대안으로 비교해야 합니다.

## 11. Canonical Correlation Analysis

PCA가 하나의 변수집합 내부 구조를 다룬다면 CCA는 두 변수집합

$$
x=(x_1,\ldots,x_{q_1})^\top,
\qquad
y=(y_1,\ldots,y_{q_2})^\top
$$

사이의 관계를 다룹니다.

첫 canonical variates를

$$
u_1=a_1^\top x,\qquad v_1=b_1^\top y
$$

로 정의하고

$$
\max_{a_1,b_1}\text{Corr}(u_1,v_1)
$$

을 풉니다.

전체 correlation matrix를 block으로

$$
R=
\begin{bmatrix}
R_{11}&R_{12}\\
R_{21}&R_{22}
\end{bmatrix}
$$

라 하면 교재에서 coefficient vector는 다음 행렬의 eigenvector로 얻습니다.

$$
E_1=R_{11}^{-1}R_{12}R_{22}^{-1}R_{21}
$$

$$
E_2=R_{22}^{-1}R_{21}R_{11}^{-1}R_{12}
$$

비영 고유값의 제곱근이 canonical correlations입니다.

- $R_{11}$: 첫 변수집합 내부 correlation matrix입니다.
- $R_{22}$: 둘째 변수집합 내부 correlation matrix입니다.
- $R_{12},R_{21}$: 두 집합 사이의 cross-correlation입니다.

**용어 설명 — canonical variate**  
각 변수집합의 여러 변수를 하나의 선형결합으로 압축한 값입니다. 두 집합의 canonical variate가 최대한 높은 상관을 갖도록 계수를 찾습니다.

## 12. 저자가 직접 보고한 결과

### 12.1 Head measurements PCA

25가족의 두 머리 측정치 예제에서 교재는 첫 주성분을 대략

$$
y_1=0.693x_1+0.721x_2
$$

으로, 둘째를

$$
y_2=-0.721x_1+0.693x_2
$$

로 보고합니다. 두 성분의 분산은 약 167.77과 28.33이며, 첫 성분이 전체 변동의 약 86%를 설명합니다. 저자들은 첫 성분을 전반적인 머리 크기, 둘째를 상대적 shape 차이로 해석합니다.

### 12.2 Olympic heptathlon

첫 주성분 score와 공식 heptathlon score 사이 상관은

$$
r=-0.9931
$$

입니다. 저자들은 음의 부호 자체는 고유벡터 부호를 $v$와 $-v$ 중 어느 쪽으로 선택하느냐의 임의성 때문에 중요하지 않다고 설명합니다.

### 12.3 Blood chemistry 성분 수

교재는 누적분산, eigenvalue threshold, scree, log-eigenvalue 등 기준이 서로 다른 성분 수를 제안하는 사례를 보여줍니다. 이는 단일 선택 규칙을 맹신해서는 안 된다는 직접적인 예입니다.

### 12.4 CCA head measurements

교재 계산에서 두 비영 고유값은 약 $0.621745$, $0.002888$이고 이에 따른 canonical correlation의 크기는 약

$$
R_1=\sqrt{0.621745}\approx0.7885,
\qquad
R_2=\sqrt{0.002888}\approx0.0537
$$

입니다. 첫 쌍은 강한 공통 크기 정보를 나타내고 둘째 쌍은 거의 관련성이 없습니다.

## 13. 해석: 저자 결과에서 무엇을 배워야 하는가?

Heptathlon의 $|r|=0.9931$은 PCA가 공식 scoring rule과 유사한 1차원 축을 발견했다는 강한 사례이지만, 이것이 PCA가 모든 supervised task에서 최적이라는 증거는 아닙니다. 반대로 blood chemistry 예제의 성분 수 불일치는 PCA가 objective한 계산법임에도 **모델 선택 단계에는 연구자의 목적과 판단이 남아 있음**을 보여줍니다.

CCA도 canonical correlation이 높다는 사실만으로 두 variable block 사이의 인과관계를 의미하지 않습니다. 또한 표본이 작고 변수 수가 크면 큰 canonical correlation이 우연히 나타날 수 있어 regularization 또는 permutation validation이 중요합니다.

## 14. 통계적으로 취약한 부분과 비교 불가능한 수치

1. **설명분산과 prediction 성능은 비교 불가능**합니다. 90% explained variance가 90% $R^2$를 뜻하지 않습니다.
2. **고유벡터 부호는 임의적**이므로 부호 자체를 모델 비교 기준으로 삼으면 안 됩니다.
3. **component selection heuristic**은 서로 다른 답을 낼 수 있습니다. scree의 elbow도 주관적일 수 있습니다.
4. **PCA sample stability**는 고유값 사이 간격이 작을수록 떨어집니다. 교재도 eigenvalue separation이 표본 크기 요구량에 영향을 준다고 설명합니다.
5. **CCA는 고차원에서 overfit 위험이 큽니다.** $R_{11}^{-1},R_{22}^{-1}$이 불안정하면 regularized CCA가 필요합니다.
6. 최신 sparse PCA 논문의 recovery error와 교재의 explained variance는 목적과 평가척도가 달라 직접 숫자 비교가 불가능합니다.

## 15. 문서가 직접 답하지 않는 질문과 답변

### 질문 1. PCA를 train+validation+test 전체에 fit해도 unsupervised이니 leakage가 아닌가?

leakage입니다. target을 보지 않더라도 test의 feature distribution을 이용해 mean, scale, principal direction을 학습합니다. 배포 시점에 알 수 없는 미래 분포를 미리 이용한 것이므로 PCA는 train에 fit해야 합니다.

### 질문 2. PCA 이후 회귀가 raw Ridge보다 항상 좋은가?

아닙니다. PCR은 variance가 큰 directions를 우선하지만 Ridge는 target과 연결된 모든 방향을 연속적으로 shrink합니다. target signal이 low-variance direction에 있으면 Ridge가 더 나을 수 있습니다.

### 질문 3. component 수는 explained variance로 정할까, validation 성능으로 정할까?

목표가 시각화·압축이면 explained variance가 자연스럽고, 목표가 예측이면 component 수를 hyperparameter로 두고 validation에서 결정하는 것이 더 직접적입니다.

## 16. 모델 일반화 성능 향상 가능성

PCA 기반 predictive pipeline에서 다음이 효과적일 수 있습니다.

- scaling 여부를 hyperparameter로 비교
- $m$을 nested CV 또는 time-series validation에서 선택
- covariance shrinkage PCA로 eigenvector 안정화
- outlier가 크면 robust PCA 검토
- sparse PCA로 해석성 향상, 단 weights와 loadings 차이를 명확히 구분
- 여러 데이터 block이 있으면 iPCA·multi-block PCA 후보
- drift가 있으면 online/incremental PCA로 subspace 변화 추적

특히 표본이 적고 $q$가 큰 데이터에서는 “더 많은 component = 더 많은 정보”가 아니라 variance inflation과 overfit을 가져올 수 있습니다.

## 17. 2020년 이후 관련 최신 연구 비교 분석

### 17.1 Integrated PCA

**Tang & Allen, “Integrated Principal Components Analysis”, Journal of Machine Learning Research, 2021.**

iPCA는 여러 데이터 matrix를 동시에 분석하여 각 데이터셋의 구조와 공통된 sample-level structure를 분리합니다. matrix-variate normal model과 Kronecker covariance를 사용하며, penalized covariance estimation으로 고차원 문제를 다룹니다. 저자들은 Alzheimer’s integrative genomics 사례에서 iPCA가 추출한 joint patterns가 cognition과 diagnosis에 높은 predictive information을 가진다고 보고합니다.

**교재 PCA와 차이**  
교재 PCA는 하나의 $X$에 대한 covariance eigen-decomposition이고, iPCA는 여러 $X_k$의 공유 구조를 명시적 확률모형으로 묶습니다.

### 17.2 Sparse PCA에 대한 현대적 비판

**Park, Ceulemans & Van Deun, “A critical assessment of sparse PCA (research): why (one should acknowledge that) weights are not loadings”, Behavior Research Methods, published 2023 / volume 2024.**

이 연구는 sparse PCA에서 weight를 sparse하게 만드는 방법과 loading을 sparse하게 만드는 방법이 더 이상 동등하지 않다는 점을 강조합니다. 또한 특정 simulation structure와 PCA-based initialization만 쓰면 연구결과가 지나치게 낙관적으로 보일 수 있음을 보고합니다.

**용어 설명 — sparse PCA**  
주성분이 모든 원변수의 작은 조합이 아니라 일부 변수에만 0이 아닌 coefficient를 갖도록 제약하여 해석성을 높이는 PCA 계열입니다.

### 비교표

| 방법 | 핵심 목적 | 장점 | 일반화/해석 위험 |
|---|---|---|---|
| Classical PCA | 최대분산 저차원 축 | 단순·안정·닫힌형 해 | target 무시, outlier 민감 |
| Sparse PCA | 축의 sparsity | 변수 해석 용이 | weights와 loadings 혼동, local optimum |
| iPCA | 여러 데이터 block의 공통 구조 | data integration | covariance model 가정 필요 |
| Online PCA | streaming subspace 추적 | drift 대응 | forgetting/learning rate 선택 필요 |

## 18. 실제 파이프라인 적용 시 고려할 점

```text
시간 또는 그룹 기준 split
  ↓
Train에서 missing/scaling fit
  ↓
PCA 후보 1: covariance PCA
PCA 후보 2: standardized correlation PCA
PCA 후보 3: shrinkage / sparse PCA
  ↓
Train 내부에서 component 수 후보 생성
  ↓
Validation에서 downstream metric 비교
  ↓
선택된 PCA를 Train(+Validation 재학습 정책에 따라)에서 최종 fit
  ↓
고정 transform으로 Test 변환
  ↓
Test metric 1회 보고
```

### 시계열에서는

random K-fold 대신 rolling/expanding validation이 필요합니다. PCA basis가 시간에 따라 변하는지 principal angle 또는 component loading drift를 추적하는 것도 좋습니다.

**용어 설명 — principal angle**  
두 subspace가 얼마나 다른 방향을 향하는지 측정하는 각도입니다. 시간 구간별 PCA subspace drift를 수치화할 수 있습니다.

## 19. 시사점과 후속 연구

교재의 PCA를 연구자 수준으로 확장할 때 핵심 질문은 “얼마나 많은 분산을 설명했는가?”에서 “그 subspace가 다른 표본에서도 재현되고 downstream task에도 유용한가?”로 이동해야 합니다. 후속 연구로는 (1) classical PCA vs shrinkage PCA의 OOS stability, (2) PCR vs Ridge/PLS의 target prediction, (3) sparse weights와 sparse loadings의 해석 차이, (4) chamber/time block을 통합하는 iPCA, (5) concept drift에서 online PCA를 비교할 수 있습니다.

## 20. 빠른 이해 점검

- 왜 $a^\top a=1$ 제약이 필요한가?
- 고유값이 PCA에서 “분산”이 되는 이유를 $\Sigma a=\lambda a$로 설명할 수 있는가?
- covariance PCA와 correlation PCA가 다른 답을 내는 이유는 무엇인가?
- explained variance가 높은 성분이 반드시 target 예측에 중요한 것은 왜 아닌가?
- CCA가 PCA와 다른 핵심 질문은 무엇인가?

## 21. 참고자료

### 교재
- Brian S. Everitt & Torsten Hothorn, *An Introduction to Applied Multivariate Analysis with R*, Springer, 2011, Chapter 3.
- I. T. Jolliffe, *Principal Component Analysis*, 2nd ed., Springer, 2002. 교재의 PCA 이론·성분 수 논의에서 주요 참고문헌.

### 2020년 이후 확장 연구 및 사이트
- Tiffany M. Tang & Genevera I. Allen, “Integrated Principal Components Analysis”, *Journal of Machine Learning Research*, Vol. 22, 2021. Source site: JMLR.
- S. Park, E. Ceulemans & K. Van Deun, “A critical assessment of sparse PCA (research): why (one should acknowledge that) weights are not loadings”, *Behavior Research Methods*, published online 2023, Vol. 56, 2024. Source site: Springer Nature.
