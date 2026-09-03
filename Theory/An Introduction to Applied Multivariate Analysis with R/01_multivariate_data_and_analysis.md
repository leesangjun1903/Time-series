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

## 5. 결측치: 왜 단순 삭제가 위험한가

### 5.1 Complete-case analysis

하나라도 결측인 행을 모두 제거합니다. 구현은 쉽지만 두 문제가 있습니다.

1. $q$가 커질수록 완전한 행이 빠르게 줄어들어 유효 표본 수가 크게 감소합니다.
2. 누락된 행이 관측된 행의 무작위 부분표본이 아니라면 추정치 자체가 편향될 수 있습니다.

**용어 설명 — MCAR(Missing Completely At Random)**  
결측 발생 여부가 관측된 값과 관측되지 않은 값 어느 것에도 의존하지 않는 강한 가정입니다. Complete-case가 일반적으로 안전하려면 이와 비슷한 조건이 필요합니다.

### 5.2 Available-case analysis

상관계수 $r_{jk}$를 계산할 때 변수 $j,k$가 둘 다 존재하는 행만 사용합니다. 표본을 더 많이 쓰는 장점이 있지만, 상관계수마다 서로 다른 표본으로 계산되므로 완성된 상관행렬이 **positive definite**하지 않을 수도 있습니다.

**용어 설명 — positive definite(양의 정부호)**  
대칭행렬 $A$가 모든 0이 아닌 벡터 $z$에 대해 $z^\top Az>0$을 만족하는 성질입니다. 정상적인 비퇴화 공분산행렬에 기대되는 성질이며, 역행렬이나 고유값 분해가 필요한 PCA·요인분석·SEM에서 중요합니다.

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

## 10. 다변량 정규성 점검

각 변수의 Q-Q plot이 모두 직선에 가깝다고 해서 결합분포 전체가 다변량 정규라는 보장은 없습니다. 교재는 Mahalanobis 거리의 순서통계량을 $\chi_q^2$ 분위수와 비교하는 방법을 설명합니다.

다변량 정규성이 맞으면 근사적으로

$$
d_i^2\sim\chi_q^2
$$

를 기대할 수 있습니다.

**용어 설명 — $\chi_q^2$ 분포**  
$q$개의 독립 표준정규변수를 제곱하여 더한 값의 분포입니다. 여기서는 다변량 정규 데이터에서 중심으로부터의 표준화된 제곱거리가 어느 정도 나와야 정상적인지 기준을 제공합니다.

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

## 13. 통계적으로 취약한 부분과 비교 불가능한 수치

1. **소표본 정규성 판단**: $n=20$에서 Q-Q plot이 그럴듯해 보여도 검정력이 낮습니다. “정규성을 확인했다”가 아니라 “강한 위반을 확인하지 못했다”가 더 안전한 표현입니다.
2. **Mahalanobis distance의 자기참조 문제**: 동일 데이터의 평균과 공분산으로 거리를 계산하면 극단값이 $S$ 자체를 왜곡하여 masking이 생길 수 있습니다. robust covariance가 대안입니다.
3. **상관과 인과의 혼동**: 공분산·상관은 방향성을 말하지 않습니다.
4. **교재의 다중대체 $m=3$–$10$ 언급**: 이는 2011년 당시의 전형적 설명입니다. 오늘날에는 결측률과 요구되는 Monte Carlo 오차에 따라 더 많은 대체가 사용되기도 하므로 숫자를 고정 규칙으로 받아들이면 안 됩니다.
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

### 질문 2. 이상치를 제거하면 항상 좋아지는가?

아닙니다. 이상치는 측정오류일 수도 있지만 실제 중요한 rare regime일 수도 있습니다. 따라서 “거리 큼 → 삭제”가 아니라 원인 확인, robust estimation, sensitivity analysis를 함께 해야 합니다.

### 질문 3. 결측치를 먼저 채운 뒤 train/test를 나누면 되는가?

안 됩니다. 전체 데이터에서 imputation model을 학습하면 test 구간의 분포 정보를 train에 전달하는 leakage가 됩니다. 실전 ML에서는 **split 후 train에서만 imputer를 fit**하고 validation/test에는 transform만 해야 합니다.

## 15. 일반화 성능을 높일 수 있는 방향

이 장의 기초를 예측 파이프라인 관점으로 바꾸면 다음이 핵심입니다.

1. 공분산이 불안정하면 shrinkage 또는 robust covariance를 사용합니다.
2. 스케일링은 train 통계량으로만 수행합니다.
3. 결측대체도 train에서만 학습하고, missingness indicator가 예측정보를 가지는지 검증합니다.
4. 이상치 제거 여부를 validation에서 결정하되, 제거 기준 자체도 train에서 고정합니다.
5. 고차원에서는 raw correlation screening만 쓰지 말고 부트스트랩 안정성 또는 regularized precision matrix를 고려합니다.

이들은 모델을 복잡하게 만드는 것이 아니라 **입력 통계량의 variance를 줄여 일반화 오차를 낮추는 방법**이라는 점이 중요합니다.

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
