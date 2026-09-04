# Chapter 1. Multivariate Data and Multivariate Analysis

> 교재 범위: Chapter 1, pp. 1–24.  
> 핵심 주제: 다변량 데이터 구조, 변수 척도, 결측치, 공분산·상관, 거리, 다변량 정규분포, Mahalanobis 거리.

## 1. Executive Summary — 10문장 이내

1. 다변량 분석의 출발점은 하나의 관측 단위에서 여러 변수를 동시에 측정했을 때, 각 변수를 따로 보지 않고 **변수 간 관계까지 함께 분석하는 것**입니다.
2. 데이터는 보통 $n$개의 관측치와 $q$개의 변수로 이루어진 행렬 $X\in\mathbb{R}^{n\times q}$로 표현됩니다.
3. 다변량 구조의 핵심 요약량은 공분산 행렬 $\Sigma$ 또는 상관행렬 $R$이며, 이 행렬들이 PCA, 요인분석, SEM, 혼합효과모형 등 뒤 장의 기반이 됩니다.
4. 변수의 척도가 다르면 공분산과 유클리드 거리는 단위의 영향을 크게 받으므로 표준화 또는 상관행렬 사용 여부를 판단해야 합니다.
5. 결측치를 단순 삭제하거나 평균으로 채우면 표본 크기·공분산·상관구조가 왜곡될 수 있으며, 교재는 다중대체를 더 적절한 일반적 해결책으로 소개합니다.
6. 다변량 정규분포는 평균벡터 $\mu$와 공분산행렬 $\Sigma$만으로 결합분포를 기술하며, 많은 고전적 추론법의 기준 분포가 됩니다.
7. Mahalanobis 거리는 변수의 분산과 변수 간 공분산을 함께 고려하므로 서로 다른 방향의 변동성을 반영한 이상치 탐지가 가능합니다.
8. 교재의 핵심 철학은 “다변량 분석은 복잡한 데이터 속에서 관계 구조를 이용하여 signal을 noise로부터 드러내는 작업”이라는 데 있습니다.
9. 2020년 이후에는 표본 수에 비해 변수가 많을 때의 **shrinkage covariance**, 자동화된 다중대체, 혼합형 데이터 결측 처리 등으로 이 고전적 기초가 확장되고 있습니다.

## 2. 목적과 필요성

한 관측치가 $q$개의 측정값을 가진다고 하겠습니다. 예를 들어 반도체 공정이라면 한 wafer 또는 run에 대해 온도, 압력, RF, gas flow, 공정시간, 계측값 등이 동시에 기록될 수 있습니다. 각 변수를 하나씩 따로 분석하면 “온도 자체의 분포”는 알 수 있지만, **온도가 높을 때 압력도 함께 높아지는지**, **특정 변수 조합이 비정상 상태를 만드는지**, **여러 센서가 사실상 같은 잠재 공정 상태를 측정하는지**는 놓칠 수 있습니다.

다변량 분석은 이 문제를 다음 두 관점으로 봅니다.

- 열(column) 관점: 변수와 변수 사이의 관계를 공분산·상관으로 표현합니다.
- 행(row) 관점: 관측치와 관측치 사이의 유사성·거리로 표현합니다.

**용어 설명 — 다변량(multivariate)**  
여러 확률변수가 동시에 분석 대상이 되는 상황입니다. 교재는 하나의 종속변수에 여러 설명변수를 넣는 일반 회귀를 엄밀히는 *multivariable*이라고 구분합니다. 실제 문헌에서는 두 표현이 혼용되므로 논문을 읽을 때 문맥을 확인해야 합니다.

## 3. 데이터 행렬과 확률변수

관측 데이터는

$$
X=
\begin{bmatrix}
x_{11} & \cdots & x_{1q}\\
\vdots & \ddots & \vdots\\
x_{n1} & \cdots & x_{nq}
\end{bmatrix}
$$

으로 씁니다.

- $n$: 관측 단위(unit)의 수입니다.
- $q$: 변수의 수입니다.
- $x_{ij}$: $i$번째 관측 단위에서 측정한 $j$번째 변수의 실제 관측값입니다.
- $X_j$: $j$번째 변수를 발생시키는 이론적 확률변수입니다.

여기서 중요한 구분은 **관측값 $x_{ij}$와 확률변수 $X_j$가 같지 않다**는 것입니다. $x_{ij}$는 이미 얻어진 숫자이고, $X_j$는 그 숫자를 만들어내는 확률적 메커니즘을 수학적으로 표현한 대상입니다.

## 4. 변수 척도와 분석 가능성

교재는 nominal, ordinal, interval, ratio의 네 척도를 소개합니다. 그러나 저자들은 “척도 분류만 보고 분석법을 기계적으로 제한하는 것”에는 상당히 실용적인 태도를 취합니다.

- **Nominal**: 순서가 없는 범주형 변수입니다. 예: chamber ID, recipe ID.
- **Ordinal**: 순서는 있지만 간격이 동일하다고 보장할 수 없습니다. 예: 불량 심각도 1–5.
- **Interval**: 차이는 의미가 있지만 절대적 0이 임의적입니다. 예: 섭씨 온도.
- **Ratio**: 0이 물리적으로 의미 있고 비율 비교가 가능합니다. 예: 시간, 길이, 질량.

**실무 해석**  
PCA나 유클리드 거리처럼 산술연산을 전제로 하는 방법에 범주형 변수를 그대로 숫자로 넣으면 “ID 4가 ID 2의 두 배”라는 가짜 기하구조를 만들 수 있습니다. 따라서 척도를 분류하는 이유는 단지 통계 교과서 규칙을 지키기 위해서가 아니라, **수식이 데이터에 부여하는 의미를 점검하기 위해서**입니다.

- 유클리드 거리 공식은 두 값의 차이를 제곱하여 계산합니다. 범주형 데이터인 '서울(1)'과 '부산(2)', '제주(3)'를 숫자로 넣으면, 알고리즘은 자동으로 서울과 제주의 거리(3-1=2)가 서울과 부산의 거리(2-1=1)보다 두 배 더 멀다고 판단합니다.
- PCA(주성분 분석)는 데이터의 분산(Variance)이 가장 큰 방향을 찾아 축을 회전합니다. 범주형 ID에 부여된 숫자의 크기와 간격은 아무런 의미가 없음에도 불구하고, 숫자의 차이가 크다는 이유만으로 알고리즘이 이를 중요한 정보(큰 분산)로 오인하게 만듭니다.

## 5. 결측치: 왜 단순 삭제가 위험한가

### 5.1 Complete-case analysis

하나라도 결측인 행을 모두 제거합니다. 구현은 쉽지만 두 문제가 있습니다.

1. $q$ (변수의 개수)가 커질수록 완전한 행이 빠르게 줄어들어 유효 표본 수가 크게 감소합니다.
2. 누락된 행이 관측된 행의 무작위 부분표본이 아니라면 추정치 자체가 편향될 수 있습니다.

**용어 설명 — MCAR(Missing Completely At Random)**  
결측 발생 여부가 관측된 값과 관측되지 않은 값 어느 것에도 의존하지 않는 강한 가정입니다. Complete-case가 일반적으로 안전하려면 이와 비슷한 조건이 필요합니다.

- 데이터의 누락(결측치)이 관측된 값이나 누락된 값 모두와 전혀 관계없이 완전히 무작위로 발생한 상태를 의미합니다

### 5.2 Available-case analysis

상관계수 $r_{jk}$를 계산할 때 변수 $j,k$가 둘 다 존재하는 행만 사용합니다. 표본을 더 많이 쓰는 장점이 있지만, 상관계수마다 서로 다른 표본으로 계산되므로 완성된 상관행렬이 **positive definite**하지 않을 수도 있습니다.

**용어 설명 — positive definite(양의 정부호)**  
대칭행렬 $A$가 모든 0이 아닌 벡터 $z$에 대해 $z^\top Az > 0$을 만족하는 성질입니다. 정상적인 비퇴화 공분산행렬에 기대되는 성질이며, 역행렬이나 고유값 분해가 필요한 PCA·요인분석·SEM에서 중요합니다.

- 상관행렬이 Positive Definite(또는 최소한 Positive Semi-definite)하지 않다는 것은 일부 고유값(Eigenvalue)이 음수로 나옵니다. 구조방정식 모형(SEM), 요인분석(Factor Analysis), 주성분분석(PCA) 등의 다변량 분석 알고리즘이 아예 작동하지 않고 에러를 내뿜습니다. 또한 회귀분석 시 행렬 연산(역행렬 계산)이 불가능해지거나 비정상적인 결과가 도출됩니다.

- 대안 : 분석에 포함된 모든 변수에 결측치가 하나도 없는 완벽한 행만 남기고 나머지는 모두 삭제합니다.

### 5.3 단일 대체의 함정

평균 대체는 결측값을 평균으로 몰아넣으므로 변동성을 줄입니다. 결과적으로 분산과 공분산을 0 방향으로 왜곡할 수 있습니다. 반대로 회귀모형의 예측값만 넣으면 설명변수와 대체된 변수 사이 관계가 지나치게 결정적이 되어 상관이 부풀려질 수 있습니다.

### 5.4 Multiple Imputation

교재가 권장하는 핵심 아이디어는 하나의 “정답 대체값”을 만들지 않는 것입니다. 결측값에 대해 $m>1$개의 가능한 값을 확률적으로 생성하고, 완성된 $m$개 데이터셋을 각각 분석한 뒤 결과를 결합합니다.

개념적으로 어떤 모수 $\theta$의 $m$개 추정치를 $\hat\theta_1,\ldots,\hat\theta_m$이라 하면 평균 추정치는

$$
\bar\theta=\frac{1}{m}\sum_{l=1}^{m}\hat\theta_l
$$

입니다. 실제 Rubin 결합법에서는 이 평균뿐 아니라 각 데이터셋 내부의 추정 불확실성과 데이터셋 사이의 대체 불확실성을 모두 사용하여 표준오차를 구성합니다.

- $m$: 대체 데이터셋 수입니다.
- $\hat\theta_l$: $l$번째 대체 데이터셋에서 얻은 모수 추정치입니다.
- $\bar\theta$: 대체 결과들을 통합한 중심 추정치입니다.

**중요**  
대체값은 실제 측정값이 아닙니다. 따라서 대체 후 행 수가 많아졌다고 해서 실제 정보량이 그만큼 증가한 것은 아닙니다.

- 루빈의 결합 규칙(Rubin's Rules)은 다중대체(Multiple Imputation)로 생성된 여러 개의 완전한 데이터셋을 하나로 결합할 때, 분석 결과의 불확실성을 과소평가하지 않기 위해 두 가지 종류의 분산을 모두 반영합니다.
  - 데이터셋 내부의 추정 불확실성 (Within-imputation variance, $\(W\)$ ): 각 데이터셋에서 얻은 추정치들의 분산(표준오차의 제곱)을 평균 낸 값입니다. 결측치가 없다고 가정했을 때 표본 추출로 인해 발생하는 일반적인 통계적 샘플링 오차를 의미합니다.
  - 데이터셋 사이의 대체 불확실성 (Between-imputation variance, $\(B\)$ ): 각 데이터셋마다 다르게 대체된 값들로 인해 발생하는 추정치들 간의 분산입니다. 즉, 결측치를 완벽하게 알지 못해서 생기는 추가적인 불확실성을 반영합니다.

- 최종 결합된 분산 $(\(T\))$ 은 이 둘을 더하고 대치 횟수 $(\(m\))$ 에 따른 수정 항을 추가하여 $\(T = W + (1 + \frac{1}{m})B\)$ 로 계산되며, 이 최종 분산의 제곱근 $(\(\sqrt{T}\))$ 이 말씀하신 최종 표준오차(Standard Error)가 됩니다.

## 6. 공분산: 함께 움직이는 정도

두 확률변수 $X_i,X_j$의 모집단 공분산은

$$
\text{Cov}(X_i,X_j)
=E\left[(X_i-\mu_i)(X_j-\mu_j)\right]
$$

입니다.

- $E[\cdot]$: 기대값입니다.
- $\mu_i=E[X_i]$: 변수 $X_i$의 모집단 평균입니다.
- $\mu_j=E[X_j]$: 변수 $X_j$의 모집단 평균입니다.
- $\text{Cov}(X_i,X_j)$: 두 변수가 각자의 평균에서 같은 방향으로 벗어나는 경향을 나타냅니다.

$i=j$이면

$$
\text{Cov}(X_i,X_i)=\text{Var}(X_i)=\sigma_i^2
$$

이므로 분산은 공분산의 특수한 경우입니다.

$q$개 변수의 모든 분산·공분산을 모으면

$$
\Sigma=
\begin{bmatrix}
\sigma_1^2 & \sigma_{12} & \cdots & \sigma_{1q}\\
\sigma_{21} & \sigma_2^2 & \cdots & \sigma_{2q}\\
\vdots & \vdots & \ddots & \vdots\\
\sigma_{q1} & \sigma_{q2} & \cdots & \sigma_q^2
\end{bmatrix}
$$

가 됩니다.

표본에서는

$$
S=\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar x)(x_i-\bar x)^\top
$$

으로 추정합니다.

- $x_i\in\mathbb R^q$: $i$번째 관측치의 변수 벡터입니다.
- $\bar x$: 표본 평균벡터입니다.
- $S$: 표본 공분산행렬입니다.
- $(\cdot)^\top$: 전치(transpose)입니다.

### 핵심 해석

독립이면 공분산은 0이지만, 공분산이 0이라고 독립인 것은 일반적으로 보장되지 않습니다. 공분산은 **선형적 동조**만 포착하기 때문입니다.

- 공분산(Covariance)은 두 변수가 '선형(직선) 관계'로 함께 움직이는지만 측정하기 때문에, 비선형적인 관계가 존재할 때는 공분산이 0이 되더라도 두 변수가 서로 독립이 아닐 수 있습니다.
  - 대표적인 예시: $\(Y = X^2\)$
  - $\(X\)$ 가 커질 때 $\(Y\)$ 가 직선 형태로 같이 커지거나 작아지는 것이 아니라, 포물선 형태(비선형)로 움직이기 때문에 선형 지표인 공분산은 이를 포착하지 못합니다.

## 7. 상관: 공분산의 단위 제거

공분산은 측정단위에 의존합니다. 이를 표준편차로 나누면

$$
\rho_{ij}=\frac{\sigma_{ij}}{\sigma_i\sigma_j}
$$

가 됩니다.

- $\rho_{ij}$: 모집단 Pearson 상관계수입니다.
- $\sigma_{ij}$: 공분산입니다.
- $\sigma_i,\sigma_j$: 각 변수의 표준편차입니다.

표본 상관행렬은

$$
R=D^{-1/2}SD^{-1/2}
$$

으로 쓸 수 있습니다.

- $D=\text{diag}(s_1^2,\ldots,s_q^2)$: 표본 분산을 대각에 둔 행렬입니다.
- $D^{-1/2}=\text{diag}(1/s_1,\ldots,1/s_q)$: 각 변수를 표준편차로 나누는 효과를 갖습니다.

**핵심 한계**  
$\rho\approx0$이 “관계 없음”을 뜻하지 않습니다. U자형처럼 강한 비선형 관계는 Pearson 상관이 0에 가까울 수 있습니다.

- U자형(이차함수) 관계나 원형 관계처럼 강력한 규칙성을 가진 비선형(Non-linear) 패턴이 존재해도, 피어슨 상관계수는 이를 감지하지 못하고 0에 가까운 값을 나타낼 수 있습니다.
- 데이터의 진짜 관계를 놓치지 않기 위해서는 통계치만 보는 것이 아니라 반드시 산점도(Scatter Plot)를 그려 눈으로 데이터의 시각적 패턴을 직접 확인해야 합니다. 비선형 관계가 의심될 때는 스피어먼(Spearman) 순위 상관계수를 쓰거나, 회귀분석 시 독립변수에 제곱항을 추가하는 등의 방법을 사용해야 합니다.

## 8. 거리: 관측치와 관측치의 차이를 수치화

가장 기본적인 유클리드 거리는

$$
d_{ij}=\sqrt{\sum_{k=1}^{q}(x_{ik}-x_{jk})^2}
$$

입니다.

- $d_{ij}$: 관측치 $i$와 $j$의 거리입니다.
- $x_{ik}$: 관측치 $i$의 $k$번째 변수입니다.
- $q$: 변수 수입니다.

변수 단위가 다르면 큰 스케일 변수 하나가 거리를 지배합니다. 따라서 필요하면 표준화 후 계산해야 합니다.

### Mahalanobis distance

한 관측치가 중심에서 얼마나 비정상적으로 떨어졌는지를 공분산까지 고려하여 측정하면

$$
d_i^2=(x_i-\bar x)^\top S^{-1}(x_i-\bar x)
$$

가 됩니다.

- $S^{-1}$: 공분산행렬의 역행렬입니다.
- $d_i^2$: squared Mahalanobis distance입니다.

유클리드 거리가 모든 방향을 같은 척도로 보는 반면, Mahalanobis 거리는 데이터가 원래 많이 퍼져 있는 방향의 차이는 덜 심각하게 보고, 원래 변동이 작은 방향의 차이는 더 크게 봅니다.

- Mahalanobis 거리는 이미 변동성이 큰(많이 퍼져 있는) 방향으로의 차이는 "그럴 수 있다"고 보아 거리를 작게 만들고, 변동성이 매우 작은 방향으로의 차이는 "일어나기 힘든 이례적인 일"로 보아 거리를 크게 계산합니다. 데이터 분포의 형태에 맞게 공간을 늘리거나 줄여서 거리를 재는 '통계적 거리'입니다.

## 9. 다변량 정규분포

$q$차원 벡터 $x$의 다변량 정규밀도는

$$
f(x;\mu,\Sigma)
=(2\pi)^{-q/2}|\Sigma|^{-1/2}
\exp\left[-\frac12(x-\mu)^\top\Sigma^{-1}(x-\mu)\right]
$$

입니다.

- $f(x;\mu,\Sigma)$: $x$에서의 확률밀도입니다.
- $\mu\in\mathbb R^q$: 평균벡터입니다.
- $\Sigma\in\mathbb R^{q\times q}$: 공분산행렬입니다.
- $|\Sigma|$: 행렬식입니다. 분포가 차지하는 전체 부피와 관련됩니다.
- $\Sigma^{-1}$: precision matrix라고도 하며, Mahalanobis 거리의 방향별 가중을 정합니다.
- $\exp(\cdot)$: 지수함수입니다.

> Precision matrix(정밀도 행렬)는 통계학과 확률론에서 공분산 행렬(covariance matrix)의 역행렬을 뜻합니다. 단일 변수에서는 분산의 역수(1/σ²)에 해당하며, 다변량 정규분포 등의 확률 모델에서 변수 간의 관계를 분석할 때 사용합니다.

> 두 확률변수 벡터가 다변량 정규분포를 따를 때, 두 벡터 사이의 공분산이 0(즉, 상관관계가 없음)이면 두 변수는 서로 통계적 독립입니다. (일반적인 분포에서는 성립하지 않지만, 정규분포에서는 성립하는 중요한 특성입니다.)

선형결합

$$
y=a^\top X
$$

은 다시 정규분포를 따르며

$$
E[y]=a^\top\mu,\qquad
\text{Var}(y)=a^\top\Sigma a
$$

입니다. 이 성질이 바로 PCA에서 “원변수의 선형결합”을 다룰 수 있게 하는 핵심 연결고리입니다.

> PCA는 원래 고차원 데이터 $\(X\)$ 에 가중치 벡터 $\(a\)$ 를 곱해 새로운 변수 $\(y = a^\top X\)$ 를 만드는 과정입니다. 원변수 $\(X\)$ 가 정규분포를 따른다면, 우리가 추출한 주성분 $\(y\)$ 역시 완벽한 정규분포를 따르게 됩니다. 덕분에 차원을 축소한 후에도 통계적 추정이나 가설 검정을 안정적으로 이어갈 수 있습니다.

> PCA의 목적은 데이터의 정보를 가장 잘 보존하는(분산이 가장 큰) 방향 $\(a\)$ 를 찾는 것입니다. 선형결합의 분산 공식이 다음처럼 행렬식으로 표현되기 때문에, $\(\text{Var}(y)=a^{\top }\Sigma a\)$ 우리는 제약조건 $(\(\Vert{}a\Vert{}_2=1\))$ 하에서 이 값을 극대화하는 라그랑주 승수법을 사용할 수 있게 됩니다. 이 수식을 풀면 결국 공분산 행렬 $\(\Sigma \)$ 의 고유값(Eigenvalue)과 고유벡터(Eigenvector) 문제로 귀결됩니다.

<details>

PCA에서 제약조건이 있는 분산 극대화 문제를 풀 때 라그랑주 승수법(Lagrange Multiplier Method)을 사용하는 이유와 구체적인 방법을 핵심만 명쾌하게 정리해 드립니다.
------------------------------
## 1. 왜 사용하는가? (제약조건이 있는 최적화)
PCA의 목적은 선형결합 $y = a^\top X$의 분산인 $a^\top \Sigma a$를 가장 크게 만드는 방향 벡터 $a$를 찾는 것입니다.
하지만 아무런 조건 없이 이 값을 크게 만들려고 하면, $a$의 크기(길이)를 무한히 키우기만 해도 분산이 무한히 커져 버립니다. 이는 우리가 원하는 '데이터의 핵심 방향'을 찾는 것과 아무 상관이 없습니다.
따라서 우리는 "방향만 고려하기 위해 벡터 $a$의 길이를 1로 고정한다"는 제약조건을 둡니다.

* 목적함수 (최대화할 값): $f(a) = a^\top \Sigma a$
* 제약조건: $g(a) = a^\top a - 1 = 0$ (즉, $\Vert{}a\Vert{}_2^2 = 1$)

이처럼 제약조건이 있는 상태에서 함수를 최적화(최대화/최소화)해야 할 때 사용하는 가장 표준적인 수학적 도구가 바로 라그랑주 승수법입니다.
------------------------------
## 2. 어떻게 사용하는가? (고유값 문제로의 변환 과정)
라그랑주 승수법은 목적함수와 제약조건을 하나의 라그랑주 함수(Lagrangian)로 합치면서 시작합니다. 새로운 변수 $\lambda$(라그랑주 승수)를 도입합니다.
## ① 라그랑주 함수 정의
$$L(a, \lambda) = a^\top \Sigma a - \lambda(a^\top a - 1)$$ 
## ② 벡터 $a$에 대해 미분하여 0이 되는 지점 찾기
행렬 미분 공식 ($\frac{\partial}{\partial a}(a^\top M a) = 2Ma$)을 적용하여 $a$로 편미분합니다.
$$\frac{\partial L}{\partial a} = 2\Sigma a - 2\lambda a = 0$$ 
## ③ 식 정리하기
양변을 2로 나누고 이항하면 PCA의 가장 아름다운 핵심 수식이 도출됩니다.
$$\mathbf{\Sigma a = \lambda a}$$ 
------------------------------
## 결론: 라그랑주 승수법이 준 놀라운 결과
라그랑주 승수법을 통해 제약조건 문제를 풀었더니, 결과가 선형대수학의 고유값(Eigenvalue) 문제로 완벽하게 변환되었습니다.

   1. 최적의 방향 $a$는? 공분산 행렬 $\Sigma$의 고유벡터(Eigenvector)입니다.
   2. 그때의 최대 분산 값은? 위 식의 양변에 왼쪽에 $a^\top$를 곱해보면 $a^\top \Sigma a = \lambda a^\top a = \lambda$ 가 됩니다. 즉, 분산의 크기가 곧 고유값(Eigenvalue) $\lambda$ 자체가 됩니다.

결국 데이터의 분산을 가장 잘 보존하는 첫 번째 주성분축은 공분산 행렬의 가장 큰 고유값에 대응하는 고유벡터가 됩니다.
  
</details>

## 10. 다변량 정규성 점검

각 변수의 Q-Q plot이 모두 직선에 가깝다고 해서 결합분포 전체가 다변량 정규라는 보장은 없습니다. 교재는 Mahalanobis 거리의 순서통계량을 $\chi_q^2$ 분위수와 비교하는 방법을 설명합니다.

> Q-Q plot은 개별 변수의 단변량(Univariate) 정규성만 확인해 줍니다. 하지만 다변량 정규분포가 되려면 변수들 사이의 상호작용(의존 구조)까지 정규성을 만족해야 합니다. 즉, 개별 변수들은 완벽한 정규분포를 가질지라도, 변수들이 결합하는 방식에 왜곡이 생기면 결합분포는 정규분포가 아닐 수 있습니다.

다변량 정규성이 맞으면 근사적으로

$$
d_i^2\sim\chi_q^2
$$

를 기대할 수 있습니다.

**용어 설명 — $\chi_q^2$ 분포**  
$q$개의 독립 표준정규변수를 제곱하여 더한 값의 분포입니다. 여기서는 다변량 정규 데이터에서 중심으로부터의 표준화된 제곱거리가 어느 정도 나와야 정상적인지 기준을 제공합니다.

> 다변량 정규분포(Multivariate Normal Distribution)를 따르는 데이터에서, 각 데이터 포인트의 마할라노비스 거리(Mahalanobis Distance)의 제곱 $\(d_{i}^{2}\)$ 은 근사적으로 자유도가 변수의 개수 $(\(q\))$ 인 카이제곱 분포 $(\(\chi _{q}^{2}\))$ 를 따르게 됩니다.

## 11. 교재의 핵심 주장과 근거

| 주장 | 교재의 근거 또는 예시 | 해석 |
|---|---|---|
| 변수를 따로 분석하면 구조를 놓친다 | 심리, 교육, 고고학, 환경 등 다수 예시 | 관계행렬이 정보의 일부이므로 단변량 분석만으로는 충분하지 않습니다. |
| 결측행 단순 삭제는 위험하다 | 정보손실과 MCAR가 아닐 때의 편향 설명 | 특히 $q$가 큰 데이터에서 완전행 비율이 급감합니다. |
| 평균 대체는 공분산을 왜곡한다 | 평균은 유지하지만 분산·공분산을 0 방향으로 축소 | PCA·요인분석 같은 공분산 기반 방법을 직접 왜곡합니다. |
| 상관은 산점도와 함께 봐야 한다 | 뒤 장 시각화와 연결 | 선형계수 하나로 비선형·이상치를 설명할 수 없습니다. |
| 다변량 정규성은 Mahalanobis 거리로 점검 가능하다 | $\chi^2$ Q-Q plot | 주변분포뿐 아니라 결합 구조를 함께 봅니다. |

## 12. 저자가 직접 보고한 결과 vs. 이 노트의 해석

### 12.1 저자 보고

- 20명의 chest–waist–hips 데이터는 표본이 너무 작아 다변량 정규성에 대해 강한 결론을 내리기 어렵다고 명시합니다.
- US air pollution 데이터에서 SO2와 precipitation의 정규확률도는 직선에서 상당히 벗어나고, manufacturing과 population에는 이상치가 보인다고 설명합니다.
- 같은 air pollution 데이터의 Mahalanobis-distance 기반 $\chi^2$ plot은 Chicago, Phoenix, Providence 같은 극단 관측치를 눈에 띄게 보여줍니다.
- 저자들은 다변량 분석의 일반 목적을 noise 속에서 signal을 발견·표시·추출하는 것이라고 정리합니다.

### 12.2 해석

이 장은 특정 예측모델을 “성능 향상”시키는 장이 아니라, 뒤의 모든 모델이 사용하는 **geometry와 uncertainty의 기초**를 만드는 장입니다. 특히 공분산행렬을 어떻게 추정하느냐가 PCA 방향, Mahalanobis 거리, Gaussian mixture, CFA/SEM 적합함수까지 연쇄적으로 영향을 줍니다. 따라서 현대 고차원 데이터에서 표본 공분산 $S$를 무비판적으로 쓰는 것은 교재 시대보다 훨씬 더 위험할 수 있습니다.

> PCA는 공분산 행렬의 고유벡터(Eigenvector)를 주성분 방향으로 설정합니다. 만약 공분산 행렬이 몇 개의 심한 이상치(Outlier)에 왜곡되어 추정된다면, 고유벡터는 데이터의 실제 주요 흐름이 아닌 이상치 방향으로 치우치게 됩니다. 추정 방식에 따라 데이터 공간을 투영하는 축(Axis) 자체가 완전히 달라집니다.  

> 마할라노비스 거리는 변수 간 상관관계와 분산을 반영하기 위해 공분산 행렬의 역행렬 $(\(\Sigma^{-1}\))$ 을 가중치로 사용합니다. 변수 대비 표본 수가 적어 공분산 행렬이 특이 행렬(Singular Matrix)에 가까워지면 역행렬이 불안정해집니다. 이로 인해 거리 값이 폭발하거나 왜곡되어, 이상치 탐지나 분류의 신뢰성이 무너집니다.

> GMM의 EM 알고리즘은 각 클러스터의 타원형 형상과 크기(공분산 행렬 $\(\Sigma_{k}\)$ )를 반복적으로 추정합니다. 공분산 구조를 어떻게 제약(Spherical, Tied, Diagonal, Full)하고 추정하느냐에 따라 클러스터의 경계가 완전히 바뀝니다. 데이터가 부족할 때 규제(Regularization) 없이 추정하면 특정 클러스터의 분산이 0으로 수렴하는 특이성(Singularity) 문제가 발생해 모델이 붕괴될 수 있습니다.  

> 구조방정식 모델링(SEM, Structural Equation Modeling)은 직접 측정할 수 없는 잠재변수와 관측변수 간의 복잡한 인과 관계를 동시에 분석하는 통합적 통계 기법입니다. SEM의 적합함수(ML, GLS 등)는 표본 공분산 행렬(\(S\))과 모델 추정 공분산 행렬 $(\(\Sigma(\theta)\))$ 의 차이를 최소화하는 방향으로 계산됩니다. 데이터의 비정규성이나 다중공선성을 고려하지 않고 표본 공분산을 그대로 투입하면, 카이제곱 $(\(\chi ^{2}\))$ 통계량이 과대추정되어 멀쩡한 모델이 기각되거나 표준오차가 왜곡되어 잘못된 가설 검정 결론에 도달합니다. (예: 다변량 비정규성일 때 Satorra-Bentler 조정 공분산 등이 필요한 이유입니다.)

- 현대 데이터 과학에서는 $\(S\)$ 를 그대로 쓰지 않고 다음과 같은 수축(Shrinkage) 및 정규화(Regularization) 기법을 필수적으로 적용합니다.
  - Ledoit-Wolf 수축 추정량: 표본 공분산 $\(S\)$ 와 항등행렬 $\(I\)$ (또는 다른 구조화된 타깃)를 가중 평균하여 고유값의 왜곡을 물리적으로 깎아냅니다.
  - Sparsity 가정을 활용한 Thresholding: 무의미한 작은 상관관계들을 강제로 0으로 만들어 행렬을 단순화합니다.
  - Graphical Lasso (Glasso): L1 규제(Lasso)를 공분산의 역행렬에 적용하여 핵심적인 인과/연관 관계만 남깁니다.

## 13. 통계적으로 취약한 부분과 비교 불가능한 수치

1. **소표본 정규성 판단**: $n=20$에서 Q-Q plot이 그럴듯해 보여도 검정력이 낮습니다. “정규성을 확인했다”가 아니라 “강한 위반을 확인하지 못했다”가 더 안전한 표현입니다.

> 표본 크기가 20개 정도로 작으면 오차 범위가 커집니다. 데이터가 정규분포와 꽤 다르게 생겼더라도, 통계적으로는 "이 정도는 작은 표본에서 우연히 발생할 수 있는 수준"이라며 넘어가게 됩니다.
> 표본 크기가 작을 때는 Q-Q plot의 점들이 직선 위에 예쁘게 놓여 있는 것처럼 보이기 쉽습니다. 하지만 이는 실제 데이터가 정규성을 만족해서가 아니라, 극단적인 값(아웃라이어)이나 분포의 왜곡을 보여줄 만큼 데이터의 수가 충분히 쌓이지 않았기 때문입니다. 몇 개의 점만으로는 분포의 진짜 모양을 알아내기 어렵습니다.

2. **Mahalanobis distance의 자기참조 문제**: 동일 데이터의 평균과 공분산으로 거리를 계산하면 극단값이 $S$ 자체를 왜곡하여 masking이 생길 수 있습니다. robust covariance가 대안입니다.

> 마할라노비스 거리(Mahalanobis Distance)를 계산할 때 일반적인 표본 평균과 표본 공분산 행렬 $(\(S\))$ 을 사용하면, 이상치(Outlier)가 평균과 공분산 자체를 자신 쪽으로 강하게 끌어당깁니다. 이로 인해 공분산 행렬이 비정상적으로 부풀려져 실제 이상치임에도 불구하고 거리가 작게 측정되는 마스킹 현상(Masking Effect)이 발생합니다.

> 따라서 robust covariance 대안은 : 

> MCD (Minimum Covariance Determinant) : 전체 데이터 중 오염되지 않은 일부분(예: 75%)의 데이터 서브셋을 찾아 공분산 행렬의 행렬식(Determinant)을 최소화하는 방식입니다.

> MVE (Minimum Volume Ellipsoid) : 데이터를 둘러싸는 가장 작은 부피의 타원체를 찾아 공분산을 추정합니다.

> 윈저화(Winsorization) 및 M-추정량(M-estimators) : 극단값의 가중치를 낮추거나 정상 범위의 최댓값/최솟값으로 대체하여 공분산을 계산합니다.

3. **상관과 인과의 혼동**: 공분산·상관은 방향성을 말하지 않습니다.

- 공분산과 상관계수는 두 변수 사이의 '선형적 관계성'만 나타낼 뿐, 원인과 결과의 방향성(인과관계)을 말해주지 않습니다. 원인과 결과의 방향성을 통계적으로 추론하려면 공분산이나 상관분석을 넘어 회귀분석(Regression Analysis)이나 구체적인 실험 설계가 필요합니다.

4. **교재의 다중대체 $m=3$ – $10$ 언급**: 이는 2011년 당시의 전형적 설명입니다. 오늘날에는 결측률과 요구되는 Monte Carlo 오차에 따라 더 많은 대체가 사용되기도 하므로 숫자를 고정 규칙으로 받아들이면 안 됩니다.

> Donald Rubin이 1987년에 처음 이 개념을 제안했을 당시에는 컴퓨터의 연산 능력이 한계가 있었기 때문에 $\(m = 3 \sim 5\)$ 정도의 작은 대체 수만으로도 충분히 효율적인 통계적 추정치를 얻을 수 있다고 보았습니다. 하지만 오늘날의 통계학계와 데이터 과학 분야에서는 이를 고정된 규칙으로 보지 않으며, 다음과 같은 이유로 더 많은 대체 수 $(\(m\))$ 를 권장하고 사용합니다.

> 현대적인 대체 수 $(\(m\))$ 기준 :
>   - 결측률 기준 (FMI 법칙): 분실 정보 비율(Fraction of Missing Information, FMI) 또는 결측률(%)과 비슷한 수준으로 $\(m\)$ 을 설정하는 것이 좋습니다. 예를 들어 데이터의 결측률이 20%라면 $\(m = 20\)$ , 결측률이 50%라면 $\(m = 50\)$ 으로 설정하는 식입니다 (Graham et al., 2007).
>   - 통계적 검정력(Power) 확보: 대체 수가 너무 적으면 통계적 검정력이 떨어지고, 하위 그룹 분석 시 표준오차가 과대평가될 수 있습니다. 높은 재현성을 확보하기 위해 최근에는 기본 설정을 $\(m = 20 \sim 100\)$ 으로 두고 분석하는 경우가 많습니다.

> 다중 대체법은 무작위 시뮬레이션(Monte Carlo) 기반 기술이기 때문에, 분석을 실행할 때마다 결과가 조금씩 달라지는 '시뮬레이션 오차'가 발생합니다. $\(m\)$ 의 크기를 키울수록 이 Monte Carlo 오차가 줄어들어 결과의 안정성과 신뢰성이 극대화됩니다.

5. **최신 imputation 논문의 RMSE와 교재 예제**: 서로 데이터셋·결측 메커니즘·목적함수가 다르므로 절대 수치를 직접 비교할 수 없습니다.

## 14. 문서가 직접 답하지 않는 질문과 답변

### 질문 1. $q$가 $n$과 비슷하거나 더 크면 $S^{-1}$은 어떻게 계산하는가?

표본 공분산은 rank가 부족해져 역행렬이 존재하지 않거나 극도로 불안정할 수 있습니다. 이때는

$$
\hat\Sigma_{\text{shrink}}
=(1-\alpha)S+\alpha T
$$

처럼 표본 공분산 $S$를 더 단순하고 안정적인 목표행렬 $T$ 쪽으로 수축시킬 수 있습니다.

- $\alpha\in[0,1]$: shrinkage 강도입니다.
- $T$: 대각행렬 또는 scaled identity 같은 안정적 target입니다.

이렇게 하면 고유값이 0 근처로 붕괴하는 문제를 완화하여 역행렬, PCA, 거리 계산을 안정화합니다.

> 목표행렬 $(\(T\))$ 의 종류: 주로 모든 자산의 분산 평균을 대각 성분으로 고정하고 공분산은 0으로 만드는 항등행렬 계열(Identity matrix)이나, 모든 자산의 상관관계를 동일하게 가정하는 일정 상관관계 행렬(Constant Correlation)이 사용됩니다.

### 질문 2. 이상치를 제거하면 항상 좋아지는가?

아닙니다. 이상치는 측정오류일 수도 있지만 실제 중요한 rare regime일 수도 있습니다. 따라서 “거리 큼 → 삭제”가 아니라 원인 확인, robust estimation, sensitivity analysis를 함께 해야 합니다.

> 원인 확인 (Root Cause Analysis): 단순 오타나 기계 결함(노이즈)인지, 아니면 자연 현상의 극단적 변동성(시그널)인지 분류하는 첫 단추입니다.

> 로버스트 추정 (Robust Estimation): 평균(Mean)이나 표준편차처럼 이상치에 취약한 통계량 대신, 중앙값(Median)이나 Huber Loss, RANSAC 알고리즘 등을 사용하여 이상치의 영향을 최소화하면서 모델의 안정성을 확보합니다.

> 민감도 분석 (Sensitivity Analysis): "이상치를 포함했을 때"와 "제거했을 때" 두 가지 버전으로 모델을 모두 돌려보고, 결론이 얼마나 쉽게 뒤집히는지 확인하여 분석 결과의 신뢰성을 검증합니다.

### 질문 3. 결측치를 먼저 채운 뒤 train/test를 나누면 되는가?

안 됩니다. 전체 데이터에서 imputation model을 학습하면 test 구간의 분포 정보를 train에 전달하는 leakage가 됩니다. 실전 ML에서는 **split 후 train에서만 imputer를 fit**하고 validation/test에는 transform만 해야 합니다.

> 결측치를 채우려면 평균값, 중앙값 등을 계산하거나 KNN, MICE 같은 예측 모델을 사용해야 합니다. 이때 전체 데이터를 기준으로 계산하면 테스트 데이터의 통계 정보(평균, 분산 등)가 학습 데이터에 미리 반영되어 버립니다. 결과적으로 모델이 실제 운영 환경(검증/테스트)에서보다 성능이 과도하게 좋게 나오는 '낙관적 편향'이 발생합니다.

## 15. 일반화 성능을 높일 수 있는 방향

이 장의 기초를 예측 파이프라인 관점으로 바꾸면 다음이 핵심입니다.

1. 공분산이 불안정하면 shrinkage 또는 robust covariance를 사용합니다.
2. 스케일링은 train 통계량으로만 수행합니다.
3. 결측대체도 train에서만 학습하고, missingness indicator가 예측정보를 가지는지 검증합니다.
4. 이상치 제거 여부를 validation에서 결정하되, 제거 기준 자체도 train에서 고정합니다.
5. 고차원에서는 raw correlation screening만 쓰지 말고 부트스트랩 안정성 또는 regularized precision matrix를 고려합니다.

이들은 모델을 복잡하게 만드는 것이 아니라 **입력 통계량의 variance를 줄여 일반화 오차를 낮추는 방법**이라는 점이 중요합니다.

> 고차원 데이터(High-dimensional data, $\(p \gg n\)$ ) 분석 시 단순히 raw correlation screening(상관관계 기반 스크리닝)만 사용하면 심각한 왜곡이나 가짜 양성(False Positive) 문제가 발생할 수 있습니다.

> 부트스트랩 안정성 기반 접근 (Stability Selection) : 데이터를 여러 번 무작위로 복원 추출(또는 50% 서브샘플링)하여 매번 스크리닝을 수행합니다. 각 변수가 최종 모델에 '얼마나 자주 선택되는지' 그 빈도(Selection Probability)를 계산합니다.

> 정규화된 정밀도 행렬 (Regularized Precision Matrix) : 고차원에서는 샘플 수가 부족해 역행렬을 직접 구할 수 없으므로, Graphical Lasso(Glasso) 같은 L1 정규화(Penalty) 기법을 적용합니다. 이를 통해 무관한 변수 간의 부분 상관관계를 0으로 만들어 행렬을 희소(Sparse)하게 만듭니다.

## 16. 2020년 이후 관련 최신 연구 비교 분석

### 16.1 Covariance shrinkage

**Ledoit & Wolf, “The Power of (Non-)Linear Shrinking: A Review and Guide to Covariance Matrix Estimation”, Journal of Financial Econometrics, 2022 (online 2020), Oxford Academic.**

고차원에서 표본 공분산의 고유값은 매우 noisy할 수 있으므로 linear/nonlinear shrinkage로 더 안정적인 공분산을 얻는 방향을 체계적으로 정리합니다. 교재의 $S$가 “표준 추정량”이라면 현대 관점은 **$S$를 그대로 쓰는 것이 항상 최선은 아니다**라는 것입니다.

**Hoff, McCormack & Zhang, “Core shrinkage covariance estimation for matrix-variate data”, Journal of the Royal Statistical Society Series B, 2023, Oxford Academic.**

행과 열의 구조를 동시에 갖는 matrix-variate 데이터에서는 separable covariance의 간결함과 비분리 구조의 유연성을 절충하는 core shrinkage를 제안합니다. 센서 $\times$ 시간 또는 wafer $\times$ 공정변수처럼 자연스러운 2차원 배열을 가진 데이터에 특히 관련됩니다.

### 16.2 Missing data

**Jarrett et al., “HyperImpute: Generalized Iterative Imputation with Automatic Model Selection”, ICML/PMLR, 2022.**

각 결측 변수의 조건부 모델과 하이퍼파라미터를 자동 선택하는 iterative imputation framework를 제시합니다. 고전적 chained imputation의 틀을 유지하면서 model selection을 자동화한다는 점이 특징입니다.

**Sun et al., “Deep learning versus conventional methods for missing data imputation: A review and comparative study”, Expert Systems with Applications, 2023, Elsevier.**

저자들의 실험에서는 제한된 크기의 tabular data에서 MICE와 missForest가 GAIN·VAE보다 더 안정적인 경우가 많았고, 특히 MAR/MNAR에서 deep generative 방법의 실패 가능성을 보고했습니다. 따라서 “딥러닝 대체 = 무조건 최신·우수”라고 가정하면 안 됩니다.

### 비교 정리

| 관점 | 교재(2011) | 2020년 이후 확장 | 실제 적용 판단 |
|---|---|---|---|
| 공분산 | 표본 $S$ 중심 | linear/nonlinear shrinkage, structured covariance | $q/n$이 크면 shrinkage를 기본 후보로 둡니다. |
| 결측 | complete/available case 비판, MI 권장 | 자동화 iterative imputation, ML/DL 비교 | 작은 tabular data에서는 복잡한 DL 대체가 항상 이점이 아닙니다. |
| 이상치 | Mahalanobis + 그래프 | robust covariance와 안정성 분석 중요 | 삭제보다 robust fit + sensitivity가 우선입니다. |
| 일반화 | 주로 통계적 추론 관점 | leakage-safe preprocessing과 OOS 검증 | 모든 전처리를 train-only로 학습합니다. |

## 17. 실제 파이프라인 적용 시 고려할 점

```text
1) 시간/그룹 구조를 고려하여 먼저 Train / Validation / Test 분할
2) 변수 타입 판정: numeric / ordinal / nominal / ID
3) Train 데이터에서 결측 메커니즘과 결측률 점검
4) Train에서 imputer fit → Validation/Test transform
5) Train에서 scaling 통계량 fit → Validation/Test transform
6) q/n 비율과 covariance condition number 점검
7) 필요하면 shrinkage/robust covariance 사용
8) correlation + scatter/conditional plot으로 비선형·이상치 확인
9) PCA/EFA/cluster/SEM 등 후속 모델 적용
10) 모든 선택을 독립 test set에서 마지막 한 번만 평가
```

**용어 설명 — condition number**  
행렬이 역행렬 계산이나 작은 데이터 변화에 얼마나 민감한지를 나타내는 수치입니다. 매우 크면 공분산 역행렬을 사용하는 방법이 불안정할 가능성이 큽니다.

## 18. 시사점과 후속 연구 계획

교재의 가장 중요한 시사점은 “다변량 분석의 성패는 화려한 모델보다 먼저 관계구조를 올바르게 정의하는 데 달려 있다”는 것입니다. 후속 연구에서는 (1) sample covariance와 shrinkage covariance의 downstream PCA/회귀 성능 비교, (2) 결측 메커니즘별 MICE·missForest·HyperImpute 비교, (3) classical Mahalanobis와 robust Mahalanobis의 이상치 안정성 비교, (4) $q/n$ 비율에 따른 고유값 왜곡과 일반화 성능의 관계를 연구할 가치가 있습니다.

추가로 공정 데이터라면 chamber·tool·recipe 같은 group structure를 covariance 자체에 반영하는 hierarchical/structured covariance 연구가 매우 실용적입니다.

## 19. 빠른 이해 점검

- 공분산이 0이면 독립이라고 말할 수 없는 이유를 한 문장으로 설명할 수 있는가?
- 왜 평균 대체가 PCA 결과까지 바꿀 수 있는가?
- 유클리드 거리와 Mahalanobis 거리의 차이를 “데이터가 많이 퍼진 방향”이라는 표현으로 설명할 수 있는가?
- $q\ge n$일 때 왜 $S^{-1}$이 문제가 되는지 설명할 수 있는가?

## 20. 참고자료

### 교재
- Brian S. Everitt & Torsten Hothorn, *An Introduction to Applied Multivariate Analysis with R*, Springer, 2011, Chapter 1.
- Donald B. Rubin, *Multiple Imputation for Nonresponse in Surveys*, Wiley, 1987. 교재가 결측치 논의에서 인용합니다.
- Joseph L. Schafer, “Multiple imputation: a primer”, Statistical Methods in Medical Research, 1999. 교재가 다중대체 개요로 인용합니다.

### 2020년 이후 확장 연구 및 사이트
- Olivier Ledoit & Michael Wolf, “The Power of (Non-)Linear Shrinking: A Review and Guide to Covariance Matrix Estimation”, *Journal of Financial Econometrics*, 2022. Source site: Oxford Academic.
- Peter Hoff, Andrew McCormack & Anru R. Zhang, “Core shrinkage covariance estimation for matrix-variate data”, *Journal of the Royal Statistical Society Series B*, 2023. Source site: Oxford Academic.
- Daniel Jarrett et al., “HyperImpute: Generalized Iterative Imputation with Automatic Model Selection”, *Proceedings of the 39th International Conference on Machine Learning*, PMLR 162, 2022. Source site: Proceedings of Machine Learning Research.
- Yige Sun et al., “Deep learning versus conventional methods for missing data imputation: A review and comparative study”, *Expert Systems with Applications*, Vol. 227, 2023. Source site: ScienceDirect / Elsevier.
