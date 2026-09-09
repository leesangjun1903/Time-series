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

> 표준화를 하지 않는 경우 (공분산 행렬 사용): PCA는 "데이터의 절대적인 변동 크기(Variance)"를 중요한 방향으로 정의합니다. 단위나 스케일이 큰 변수가 주성분을 완전히 지배하게 됩니다. 예를 들어 '연봉(원 단위)'과 '나이(세 단위)'를 함께 넣으면, PCA는 사실상 연봉의 변화만을 중요한 방향으로 인식합니다. 이는 단위의 크기 자체가 데이터의 실제 중요도를 반영할 때 올바른 선택이 됩니다.
>
> 표준화를 하는 경우 (상관계수 행렬 사용): PCA는 "변수 간의 선형적 연관성(Correlation)"을 중요한 방향으로 정의합니다. 모든 변수의 분산을 1로 맞추기 때문에, 단위와 상관없이 모든 변수가 동등한 가중치(영향력)를 가지고 주성분 형성에 기여합니다. 이는 단위의 차이가 무의미하고, 변수들 사이의 가려진 관계(패턴)를 찾고 싶을 때 올바른 선택이 됩니다.

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

> Kaiser Rule (카이저 규칙): 상관행렬을 이용한 PCA에서 고유값 $(\(\lambda \))$ 이 1보다 큰 주성분만 선택하는 방법입니다. 변수 1개가 가진 분산(1)보다 더 큰 분산을 설명하는 성분만 의미가 있다고 보는 표준적인 기준입니다.

> Jolliffe 기준: 카이저 규칙이 너무 엄격하여 유용한 정보가 버려질 수 있다는 점을 보완하기 위해, 고유값 기준을 0.7 이상으로 완화하여 적용하는 방식입니다. 표본 오차나 무작위 변동으로 인해 고유값이 낮게 측정될 수 있음을 감안한 제안입니다.

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

- $V_m$: 첫 $m$개 고유벡터입니다. 데이터의 공분산 행렬에서 고유값이 큰 순서대로 뽑은 첫 $\(m\)$ 개의 고유벡터(주성분 방향)들을 열로 가진 행렬입니다. ( $\(d \times m\)$ 크기)
- $Z_m=X_cV_m$: 첫 $m$개 score입니다. 원본 데이터를 $\(V_{m}\)$ 방향으로 투영(Projection)시켜 얻은 주성분 점수(Score)입니다. 데이터가 $\(m\)$ 차원으로 줄어든 실제 축소 행렬입니다.
- $\hat X_c$: 중심화 원데이터의 rank- $m$ 근사입니다. 축소된 $\(Z_{m}\)$ 에 다시 $\(V_{m}^{\top }\)$ 를 곱해 원래 차원으로 되돌린 행렬입니다. 정보 손실이 일어났기 때문에 완전히 똑같지는 않지만, 원본을 가장 잘 모사한 $\(m\)$ 개의 선형 결합(Rank- $\(m\)$ )이 됩니다.

> 주성분분석(PCA)은 고차원 데이터를 분산이 가장 큰 방향(주성분)으로 투영하여 차원을 줄입니다. 이때 상위 $\(m\)$ 개의 주성분만 남기고 나머지를 버린 뒤, 이를 다시 원래 공간으로 복원(Reconstruction)하면 원본 데이터와 가장 유사한 최적의 저차원 근사치 $(\(\hat{X}^{c}\))$ 를 얻을 수 있습니다.

> 수학적으로 기하학적 관점에서 보면, $\(\hat{X}^c = Z_m V_m^\top\)$ 는 원본 데이터 $\(X^{c}\)$ 와의 제곱 오차 거리(Reconstruction Error)를 최소화하는 최적의 $\(m\)$ 차원 부분공간(Subspace)을 찾은 결과물입니다.
> 데이터의 중요한 특징(큰 분산)을 가진 상위 $\(m\)$ 개 성분만 유지하고, 자잘한 변동을 가진 나머지 성분을 0으로 만들어 날려버렸기 때문에 일종의 데이터 노이즈 필터링 효과를 가집니다.

PCA는 squared reconstruction error를 최소화하는 최적 rank- $m$ 선형 근사와 연결됩니다. 따라서 차원축소는 “정보 삭제”가 아니라 **고유값이 작은 방향을 버리는 low-rank approximation**입니다.

> 데이터의 구조를 표현하는 데 기여도가 가장 낮은 불필요한 차원(노이즈)을 걸러내고, 핵심적인 기하학적 구조(구조적 정보)만을 남기는 최적의 데이터 압축 과정으로 이해하는 것이 정확합니다.

## 9. Biplot

Biplot은 관측치 score와 변수 loading 정보를 같은 저차원 그림에 표시합니다. 가까운 관측치는 선택한 PCA 공간에서 유사하고, 같은 방향을 가리키는 변수 화살표는 양의 관련성을 가질 가능성이 큽니다.

단, biplot은 첫 2개 정도의 성분에 정보를 투영하므로 전체 데이터 구조를 완전히 나타내지 않습니다. 첫 두 성분 설명분산이 낮다면 해석은 특히 조심해야 합니다.

> 도표 위에 점으로 표시되는 관측치들이 서로 가깝게 위치할수록 해당 데이터들이 서로 유사한 특징(패턴)을 가지고 있음을 의미합니다.
>
> 두 화살표가 이루는 각도가 좁을수록(같은 방향) 두 변수 간에 강한 양의 상관관계가 있을 가능성이 높습니다.
>
> 화살표의 길이는 해당 변수가 해당 주성분(PC) 공간에서 가지는 영향력(분산의 크기)을 나타냅니다. 화살표가 길수록 해당 주성분을 설명하는 데 중요한 변수입니다.
>
> 특정 관측치 점이 특정 변수 화살표 방향으로 멀리 치우쳐 있다면, 그 관측치는 해당 변수의 값이 상대적으로 매우 높다는 것을 시각적으로 유추할 수 있습니다.

> - 보완할 수 있는 접근 방법
>   - 스크리 산점도(Scree Plot) 확인: PC1, PC2뿐만 아니라 PC3, PC4 등 이후 주성분이 얼마나 많은 정보를 담고 있는지 누적 설명분산을 먼저 체크해야 합니다. 일반적으로 누적 설명분산이 70~80% 이상일 때 2차원 바이플롯의 신뢰도가 높습니다.
>   - 3차원 바이플롯 활용: 첫 두 성분의 설명분산이 너무 낮다면, PC3까지 포함하여 3차원 공간에 바이플롯을 시각화하는 것도 좋은 대안이 됩니다.
>   - 상관관계 행렬 병행: 변수 간의 관계(화살표 각도)를 확신하기 어렵다면, 실제 원본 데이터의 Pearson/Spearman 상관계수 행렬을 함께 띄워두고 수치를 검증하는 것이 안전합니다.

<img width="828" height="772" alt="image" src="https://github.com/user-attachments/assets/34d73f9e-f307-459b-96f4-ed03e6be8645" />

> 변수 간의 관계 (화살표 방향과 각도)
>   - x1과 x2: 두 화살표가 거의 같은 방향을 가리키며 겹쳐 있습니다. 이는 두 변수 간에 매우 강한 양의 상관관계가 있음을 뜻합니다.
>   - x3와 x4: 두 화살표가 서로 반대 방향(180도)을 향하고 있습니다. 이는 두 변수가 강한 음의 상관관계를 가짐을 보여줍니다.
>   - x1(또는 x2)과 x3: 두 화살표의 각도가 직각(90도)에 가깝습니다. 이는 두 변수가 서로 거의 독립적이며 상관관계가 없음을 나타냅니다.

> 변수의 영향력 (화살표 길이)
>   - x1, x2, x3, x4 변수는 화살표가 바깥쪽으로 길게 뻗어 있어, 이 주성분 공간(PC1, PC2)을 설명하는 데 기여도가 매우 높은 핵심 변수들입니다.
>   - 반면 x5는 화살표 길이가 상대적으로 매우 짧습니다. 이는 x5가 데이터의 주요 패턴(PC1, PC2)을 설명하는 데 큰 역할을 하지 못하는 변수임을 의미합니다.


## 10. PCA와 예측의 관계

PCA는 target $y$를 보지 않습니다. 따라서

$$
\text{large }\text{Var}(Xv)
\not\Rightarrow
\text{large predictive information about }y
$$

(큰 분산이 큰 예측력을 보장하지 않는다.)

입니다.

예를 들어 target을 결정하는 신호가 전체 분산은 작지만 안정적인 센서 조합에 존재한다면 PCA가 해당 방향을 후순위로 밀어버릴 수 있습니다. 이 때문에 supervised 목적에서는 PLS, supervised PCA, regularized regression 등을 대안으로 비교해야 합니다.

> PCA는 데이터가 어디로 가장 많이 흩어져 있는가(분산)만 볼 뿐, 그것이 우리가 맞추려는 정답(타겟 y)과 관련이 있는지는 전혀 신경 쓰지 않습니다.

> PCA는 데이터의 '설명력'을 오직 분산(Variance, 데이터가 퍼진 크기)으로만 판단합니다. 하지만 분산이 큰 신호가 언제나 예측에 중요한 것은 아닙니다.

> PLS (Partial Least Squares, 부분최소제곱)
> - x의 분산만 보는 PCA와 달리, x와 y의 상관관계(공분산)를 동시에 보면서 차원을 축소합니다. 정답 y를 예측하는 데 가장 도움이 되는 방향으로 데이터의 축을 회전시킵니다.

> Supervised PCA (지도형 PCA)
> - PCA를 하기 전에 1차 스크리닝(필터링)을 거칩니다. 타겟 y와 상관관계가 너무 낮은(아무 쓸모 없는) 특성들을 먼저 제거한 뒤, 남은 핵심 특성들로만 PCA를 수행합니다.

> Regularized Regression (규제 선형 회귀)
> - Ridge(릿지)나 Lasso(라쏘) 같은 모델을 사용합니다. 굳이 PCA처럼 축을 바꾸지 않고, 원래 특성들을 그대로 둔 상태에서 y 예측에 방해가 되거나 불필요한 특성의 영향력(가중치)을 0에 가깝게 줄여버립니다.

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

> 공분산 행렬을 각각 $\(\Sigma_{xx}\)$ ( $\(X\)$ 의 공분산), $\(\Sigma_{yy}\)$ ( $\(Y\)$ 의 공분산), $\(\Sigma_{xy}\)$ ( $\(X\)$ 와 $\(Y\)$ 의 상호 공분산)라고 하면 다음과 같이 정리됩니다. : $\(\text{Var}(u_1) = a_1^\top \Sigma_{xx} a_1\)\ , (\text{Var}(v_1) = b_1^\top \Sigma_{yy} b_1\)\ , (\text{Cov}(u_1, v_1) = a_1^\top \Sigma_{xy} b_1\)$

> 우리가 최대화하려는 목표는 두 변량의 상관계수(Correlation)입니다.

> $$\(\max _{a_{1},b_{1}}\text{Corr}(u_{1},v_{1})=\max _{a_{1},b_{1}}\frac{\text{Cov}(u_{1},v_{1})}{\sqrt{\text{Var}(u_{1})\text{Var}(v_{1})}}\)$$

> 이때 $\(a_{1}\)$ 과 $\(b_{1}\)$ 의 크기(스케일)가 바뀌어도 상관계수는 변하지 않으므로, 계산을 쉽게 하기 위해 분모를 1로 고정하는 제약 조건을 둡니다.

> 목적 함수: $\(\max_{a_1, b_1} a_1^\top \Sigma_{xy} b_1\)$ , 제약 조건: $\(a_1^\top \Sigma_{xx} a_1 = 1\), \(b_1^\top \Sigma_{yy} b_1 = 1\)$

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

> 제약 조건이 있는 최적화 문제를 풀기 위해 라그랑주 함수 $\(L\)$ 을 세웁니다. $(\(\lambda_1, \lambda_2\)$ 는 라그랑주 승수입니다.) (라그랑주 함수에서 제약 조건 앞에 $\(\frac{1}{2}\)$ 을 붙이는 이유는 나중에 미분할 때 계산을 더 깔끔하고 편리하게 만들기 위해서입니다.)

> $\(L(a_{1},b_{1},\lambda_{1},\lambda_{2})=a_{1}^{\top }\Sigma_{xy}b_{1}-\frac{\lambda_{1}}{2}(a_{1}^{\top }\Sigma_{xx}a_{1}-1)-\frac{\lambda_{2}}{2}(b_{1}^{\top }\Sigma_{yy}b_{1}-1)\)$

> 최대값을 찾기 위해 $\(a_{1}\)$ 과 $\(b_{1}\)$ 로 각각 편미분하여 0이 되는 지점을 찾습니다.

> $\(\frac{\partial L}{\partial a_1} = \Sigma_{xy} b_1 - \lambda_1 \Sigma_{xx} a_1 = 0 \implies \Sigma_{xy} b_1 = \lambda_1 \Sigma_{xx} a_1\)$

> $\(\frac{\partial L}{\partial b_1} = \Sigma_{yx} a_1 - \lambda_2 \Sigma_{yy} b_1 = 0 \implies \Sigma_{yx} a_1 = \lambda_2 \Sigma_{yy} b_1\)$ (여기서 $\(\Sigma_{yx} = \Sigma_{xy}^\top\)$ 입니다.)

> 위 두 식의 왼쪽에 각각 $\(a_{1}^{\top }\)$ 과 $\(b_{1}^{\top }\)$ 을 곱해보면 재미있는 사실을 알 수 있습니다.

> $\(a_1^\top \Sigma_{xy} b_1 = \lambda_1 a_1^\top \Sigma_{xx} a_1 = \lambda_1\)$ (제약 조건에 의해 뒤 항이 1이 됨)

> $\(b_1^\top \Sigma_{yx} a_1 = \lambda_2 b_1^\top \Sigma_{yy} b_1 = \lambda_2\)$

> 두 식의 값이 결국 우리가 구하려는 최대 상관계수 $\(\rho \)$ 로 같습니다. 즉, $\(\lambda_1 = \lambda_2 = \rho\)$ 입니다.

> 이제 식을 다시 정리하면 다음과 같은 연립 방정식이 됩니다.

> $\(\Sigma_{xy} b_1 = \rho \Sigma_{xx} a_1 \implies a_1 = \frac{1}{\rho} \Sigma_{xx}^{-1} \Sigma_{xy} b_1\)\$

> $(\Sigma_{yx} a_1 = \rho \Sigma_{yy} b_1\)$

> 1번 식을 2번 식에 대입하여 $\(a_{1}\)$ 을 소거합니다.

> $\(\Sigma_{yx}\left(\frac{1}{\rho }\Sigma_{xx}^{-1}\Sigma_{xy}b_{1}\right)=\rho \Sigma_{yy}b_{1}\)\$

> $(\Sigma_{yy}^{-1}\Sigma_{yx}\Sigma_{xx}^{-1}\Sigma_{xy}b_{1}=\rho ^{2}b_{1}\)$

> 같은 방식으로 $\(b_{1}\)$을 소거하면 $\(a_{1}\)$에 대한 식도 얻을 수 있습니다. $\(\Sigma_{xx}^{-1}\Sigma_{xy}\Sigma_{yy}^{-1}\Sigma_{yx}a_{1}=\rho ^{2}a_{1}\)$

결국 이 문제는 행렬의 일반화된 고유값 문제(Generalized Eigenvalue Problem)가 됩니다.

- $\(a_{1}\)$ 구하기: 행렬 $\(\Sigma_{xx}^{-1} \Sigma_{xy} \Sigma_{yy}^{-1} \Sigma_{yx}\)$ 의 가장 큰 고유값을 찾습니다.
  - 그 고유값이 바로 최대 상관계수의 제곱 $(\(\rho ^{2}\))$ 이 되며, 그때의 고유벡터가 바로 $\(a_{1}\)$ 입니다.
- $\(b_{1}\)$ 구하기: 구해진 $\(a_{1}\)$을 이용하여 $\(b_1 = \frac{1}{\rho} \Sigma_{yy}^{-1} \Sigma_{yx} a_1\)$ 공식으로 구하거나, 반대편 행렬의 가장 큰 고유벡터를 구합니다.
- 최종적으로 구한 고유벡터 $\(a_1, b_1\)$ 을 처음에 설정한 제약 조건 $(\(a_1^\top \Sigma_{xx} a_1 = 1\))$ 에 맞게 크기를 조절(선형 정규화)해주면 첫 번째 정식 변량의 가중치 벡터가 완성됩니다.

- $\(E_{1}\)$ 의 고유벡터는 첫 번째 변수 집합 $\(X\)$ 를 위한 가중치 벡터(coefficient vector) $\(a_{1}\)$ 이 됩니다.
- $\(E_{2}\)$ 의 고유벡터는 두 번째 변수 집합 $\(Y\)$ 를 위한 가중치 벡터(coefficient vector) $\(b_{1}\)$ 이 됩니다.
- 재미있는 점은, 두 행렬 $\(E_{1}\)$ 과 $\(E_{2}\)$ 는 곱하는 순서만 바뀐 꼴 $(\(AB\)$ 와 $\(BA\))$ 이기 때문에 영이 아닌 고유값(Eigenvalue)을 완벽하게 공유합니다.

따라서 $\(E_{1}\)$ (또는 $\(E_{2}\))$ 의 가장 큰 고유값을 찾아서 제곱근을 구하면, 그것이 바로 첫 번째 정식 변량 쌍 $\((u_1, v_1)\)$ 이 가질 수 있는 최대 상관계수가 됩니다.

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

> - 설명분산(Explained Variance)과 예측 성능(R² 등)은 서로 다른 개념이라 직접적으로 비교할 수 없습니다. 90%의 설명분산을 가졌다고 해서 예측 성능(R²)이 무조건 90%가 되는 것은 아닙니다.
>   - 설명분산(Explained Variance): 모델이 데이터의 '변화 추세(움직임)'를 얼마나 잘 따라가는지 봅니다. 예측값과 실제값의 평균적인 차이(편향)는 무시합니다. 즉, 모델이 실제보다 항상 10만큼 크게 예측하더라도, 오르내리는 모양새만 똑같다면 설명분산은 100%가 나올 수 있습니다.
>   - 결정계수( $\(R^{2}\)$ ):모델이 실제 데이터 값을 얼마나 '정확하게 맞추는지'를 봅니다. 여기서는 예측값이 실제값에서 벗어난 절대적인 거리(오차)를 그대로 계산합니다. 아무리 오르내리는 모양새를 잘 맞췄어도, 값이 통째로 밀려 있으면 $\(R^{2}\)$ 점수는 크게 떨어집니다. 심지어 음수(-)가 나오기도 합니다.

2. **고유벡터 부호는 임의적**이므로 부호 자체를 모델 비교 기준으로 삼으면 안 됩니다.

> - 고유벡터는 방향을 나타내는 벡터입니다. 선형대수학의 정의상 어떤 벡터 x가 고유벡터라면, 여기에 -1을 곱한 -x 역시 똑같은 고유값(Eigenvalue)을 가지는 고유벡터입니다.
> - 부호의 영향을 받지 않고 두 모델의 고유벡터를 비교하려면 다음과 같은 방법을 사용해야 합니다.
>   - 절댓값 비교: 고유벡터 안의 원소들에 절댓값을 씌워 크기(기여도)만 비교합니다.
>   - 코사인 유사도(Cosine Similarity): 두 벡터 사이의 각도를 측정합니다. 부호가 완전히 반대라면 코사인 유사도가 -1이 나오므로, 결과의 절댓값이 1에 가까운지 확인합니다.
>   - 내적(Dot Product)의 절댓값: 두 단위 고유벡터를 내적한 값에 절댓값을 씌워 1에 가까운지 확인합니다.

3. **component selection heuristic**은 서로 다른 답을 낼 수 있습니다. scree의 elbow도 주관적일 수 있습니다.

> - 주요 성분 분석(PCA)이나 요인 분석(Factor Analysis)에서 성분 개수를 고르는 기준들이 저마다 다르기 때문입니다.
>   - 스크리 도표의 엘보우 (Elbow Method) : 그래프가 완만해지기 직전의 꺾인 지점을 찾습니다. 그래프가 부드러운 곡선 형태면 어디가 꺾이는 지점인지 사람마다 다르게 보입니다.
>   - 카이저 규칙 (Kaiser's Rule) : 고유값(Eigenvalue)이 1보다 큰 성분만 남깁니다. 고유값이 1.01인 성분은 포함되고, 0.99인 성분은 버려지는 단판 승부식이라 데이터의 작은 변화에 민감합니다.
>   - 누적 분산 설명력 (Cumulative Variance Explained): 전체 데이터의 70%~80% 이상을 설명할 수 있을 때까지 성분을 뽑습니다. 70%로 할지 80%로 할지 기준 자체가 분석가의 마음입니다.

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
