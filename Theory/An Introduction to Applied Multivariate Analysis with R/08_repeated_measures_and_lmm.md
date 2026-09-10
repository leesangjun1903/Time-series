# Chapter 8. The Analysis of Repeated Measures Data

> 교재 범위: Chapter 8, pp. 225–258.  
> 핵심 주제: repeated measures, longitudinal dependence, linear mixed-effects model, random intercept, random slope, REML, random-effect prediction, dropout/missingness.

## 1. Executive Summary — 10문장 이내

1. 반복측정 데이터의 핵심 문제는 같은 대상에서 여러 번 얻은 관측값이 서로 독립이 아니라는 점입니다.
2. 이를 무시한 ordinary regression이나 단순 ANOVA는 회귀계수 자체뿐 아니라 특히 표준오차와 추론을 왜곡할 수 있습니다.
3. Linear Mixed-Effects Model(LMM)은 population-level fixed effect와 subject/group-specific random effect를 한 모델에 결합하여 평균구조와 상관구조를 동시에 설명합니다.
4. Random-intercept model은 개체마다 출발점이 다르도록 허용하고, 같은 개체 내 반복측정 사이에 공통 covariance를 부여합니다.
5. Random-slope model을 추가하면 개체마다 시간 또는 공정조건에 대한 반응속도까지 달라질 수 있으며, 반복측정의 분산·공분산도 측정시점에 따라 달라집니다.
6. variance component를 추정할 때 ML은 downward bias가 생길 수 있어 교재는 REML을 권장하며, REML model 간 likelihood-ratio comparison에는 fixed-effect structure가 같아야 한다는 제약이 있습니다.
7. random effect prediction은 집단 평균과 개체별 데이터 사이를 shrink하는 partial pooling 효과를 만들어 소표본 개체의 과적합을 줄입니다.
8. longitudinal dropout은 단순 결측이 아니라 상태가 나쁜 대상일수록 탈락하는 informative missingness일 수 있으므로 MCAR/MAR/MNAR 구분이 중요합니다.
9. 2020년 이후에는 mixed-effects neural network, longitudinal multiple imputation, Bayesian joint modeling처럼 LMM의 계층구조를 비선형 예측·결측·event process와 결합하는 연구가 활발합니다.
10. 실제 일반화 성능을 평가할 때는 “이미 본 subject의 미래”와 “완전히 새로운 subject”를 구분하고, row-level random split로 동일 subject가 train/test에 동시에 들어가는 leakage를 피해야 합니다.

## 2. 목적과 필요성

반복측정 데이터에서는 한 사람, wafer, chamber, machine, specimen에 대해 여러 시점 또는 여러 조건에서 같은 response를 측정합니다. 같은 대상에서 나온 두 측정치는 다른 대상끼리의 측정치보다 더 비슷한 것이 자연스럽습니다.

예를 들어 chamber별 VM 데이터를 생각하면 같은 chamber의 run들은 chamber 고유의 offset, aging state, calibration history를 공유합니다. 이 상관을 무시하면 “독립적인 표본이 많이 있다”고 착각하여 uncertainty를 지나치게 작게 평가할 수 있습니다.

**용어 설명 — repeated measures / longitudinal data**  
같은 unit에서 response를 여러 번 기록한 데이터입니다. 시간이 핵심 축이면 longitudinal이라고 부르는 경우가 많지만 반복측정은 시간 이외의 조건 반복도 포함할 수 있습니다.

## 3. 독립 회귀의 출발점과 문제

가장 단순한 모델은

$$
y_{ij}=\beta_0+\beta_1x_j+\epsilon_{ij}
$$

입니다.

- $i$: subject 또는 group index입니다.
- $j$: 반복측정 시점/조건 index입니다.
- $y_{ij}$: $i$번째 대상의 $j$번째 response입니다.
- $x_j$: time, slippage 등 반복 조건입니다.
- $\beta_0,\beta_1$: 전체 평균 intercept와 slope입니다.
- $\epsilon_{ij}$: residual error입니다.

일반 OLS는 보통 $\epsilon_{ij}$들이 서로 독립이고 같은 분산을 갖는다고 가정합니다. 그러나 같은 subject 안에서는 공통 원인이 존재하므로 이 가정이 현실적이지 않을 수 있습니다.

## 4. 저자가 직접 보고한 단순 회귀 결과: Timber slippage

교재의 timber data에 독립 OLS를 적용하면

$$
\hat\beta_0=3.516,
\qquad
\hat\beta_1=10.373
$$

이고

- intercept SE = 0.264
- slope SE = 0.283
- residual standard error = 1.65
- multiple $R^2=0.919$
- adjusted $R^2=0.918$

을 보고합니다.

저자들은 slippage effect가 크고 통계적으로 유의하다고 하면서도 **반복관측을 독립으로 보는 가정은 비현실적**이라고 바로 지적합니다. 이 예제의 중요한 교훈은 높은 $R^2$가 적절한 correlation model을 보장하지 않는다는 것입니다.

## 5. Random Intercept Model

개체마다 고유한 intercept $u_i$를 추가합니다.

$$
y_{ij}=(\beta_0+u_i)+\beta_1x_j+\epsilon_{ij}
$$

가정은

$$
u_i\sim N(0,\sigma_u^2),
\qquad
\epsilon_{ij}\sim N(0,\sigma^2)
$$

이며 서로 독립이라고 둡니다.

- $u_i$: subject $i$의 random intercept deviation입니다.
- $\sigma_u^2$: subject 간 intercept heterogeneity입니다.
- $\sigma^2$: subject 내 residual variance입니다.

### 왜 같은 subject의 측정치가 상관되는가?

전체 residual은

$$
r_{ij}=u_i+\epsilon_{ij}
$$

입니다. 따라서

$$
\text{Var}(r_{ij})=\sigma_u^2+\sigma^2
$$

이고 같은 subject의 서로 다른 두 시점 $j\ne j'$에 대해

$$
\text{Cov}(r_{ij},r_{ij'})=\sigma_u^2
$$

입니다.

따라서 intra-class correlation(ICC)은

$$
\rho_{ICC}
=\frac{\sigma_u^2}{\sigma_u^2+\sigma^2}
$$

가 됩니다.

**용어 설명 — ICC(Intra-Class Correlation)**  
전체 residual variance 중 subject/group 사이 차이가 차지하는 비율입니다. ICC가 크면 같은 subject의 반복측정들이 서로 강하게 닮습니다.

## 6. Compound Symmetry의 한계

Random-intercept model에서는 모든 시점의 residual variance가 같고 모든 시점쌍 covariance가 $\sigma_u^2$로 같습니다. 이를 compound symmetry라고 합니다.

하지만 실제 longitudinal data에서는 가까운 시점끼리 더 강하게 상관되고 시간이 지날수록 분산이 커질 수 있습니다. 따라서 random intercept 하나만으로는 충분하지 않을 수 있습니다.

**용어 설명 — compound symmetry**  
대각 원소는 모두 같은 variance, 비대각 원소는 모두 같은 covariance를 갖는 covariance structure입니다.

## 7. Random Intercept + Random Slope Model

개체마다 slope도 다르게 허용하면

$$
y_{ij}
=(\beta_0+u_{i1})
+(\beta_1+u_{i2})x_j
+\epsilon_{ij}
$$

입니다.

random effects vector를

$$
\begin{bmatrix}
u_{i1}\\u_{i2}
\end{bmatrix}
\sim N\left(
\begin{bmatrix}0\\0\end{bmatrix},
\begin{bmatrix}
\sigma_{u1}^2 & \sigma_{u1u2}\\
\sigma_{u1u2} & \sigma_{u2}^2
\end{bmatrix}
\right)
$$

로 둡니다.

- $u_{i1}$: intercept deviation입니다.
- $u_{i2}$: slope deviation입니다.
- $\sigma_{u1u2}$: intercept와 slope의 covariance입니다.

전체 residual variance는

$$
\text{Var}(u_{i1}+u_{i2}x_j+\epsilon_{ij})
=\sigma_{u1}^2
+2\sigma_{u1u2}x_j
+\sigma_{u2}^2x_j^2
+\sigma^2
$$

입니다.

두 시점 $j,j'$의 covariance는

$$
\text{Cov}(r_{ij},r_{ij'})
=\sigma_{u1}^2
+\sigma_{u1u2}(x_j+x_{j'})
+\sigma_{u2}^2x_jx_{j'}
$$

입니다.

이제 variance와 covariance가 $x$에 따라 변하므로 random-intercept model보다 훨씬 유연합니다.

## 8. 일반 행렬형 LMM

LMM은 compact하게

$$
y=X\beta+Zb+\epsilon
$$

로 쓸 수 있습니다.

- $y$: 모든 response를 모은 벡터입니다.
- $X$: fixed-effect design matrix입니다.
- $\beta$: population-level coefficient입니다.
- $Z$: random-effect design matrix입니다.
- $b$: subject/group-specific random-effect vector입니다.
- $\epsilon$: observation-level residual입니다.

가정은 보통

$$
b\sim N(0,G),
\qquad
\epsilon\sim N(0,R)
$$

이며

$$
\text{Var}(y)=ZGZ^\top+R
$$

가 됩니다.

이 식이 repeated-measures covariance를 설계하는 핵심입니다.

## 9. Fixed effect와 Random effect

### Fixed effect

관심 population 전체에 공통으로 적용되는 평균적 효과입니다. 예를 들어 slippage 증가에 따른 평균 load 증가율입니다.

### Random effect

각 subject가 population 평균에서 얼마나 벗어나는지를 확률변수로 표현합니다. 모든 subject마다 별도의 unrelated coefficient를 완전히 자유롭게 학습하는 것과 달리, 공통 분포 $N(0,G)$에서 나온다고 가정하여 **부분적으로 공유**합니다.

**용어 설명 — partial pooling**  
개별 group을 완전히 따로 추정(no pooling)하지도, 완전히 동일하게 묶어 추정(complete pooling)하지도 않고, group별 정보량에 따라 population 평균 쪽으로 적절히 shrink하는 방식입니다.

## 10. ML과 REML

교재는 variance component를 ordinary maximum likelihood로 추정하면 downward bias가 생길 수 있어 Restricted Maximum Likelihood(REML)를 흔히 권장한다고 설명합니다.

REML의 핵심 직관은 fixed-effect $\beta$ 추정에 소비된 자유도를 고려한 residual contrast를 통해 variance component를 추정한다는 것입니다.

**용어 설명 — variance component**  
$\sigma_u^2$, $\sigma_{u2}^2$, $\sigma^2$처럼 random effect와 residual의 variability를 나타내는 parameter입니다.

### 모델 비교 주의

REML로 적합한 두 모델을 likelihood-ratio test로 비교하려면 두 모델의 **fixed-effect structure가 같아야** 합니다. fixed effect가 다르면 REML likelihood가 서로 다른 transformed data space를 기준으로 하기 때문에 직접 비교가 적절하지 않습니다.

## 11. Random-effect prediction과 BLUP

교재는 random effect prediction을 별도 절에서 다룹니다. LMM에서 개체 random effect의 대표적 예측값은 BLUP입니다. Gaussian LMM에서 개념적으로

$$
\hat b
=GZ^\top V^{-1}(y-X\hat\beta),
\qquad
V=ZGZ^\top+R
$$

형태를 가집니다.

- $\hat b$: random-effect prediction입니다.
- $V$: marginal covariance of $y$입니다.

개체 데이터가 적거나 noisy하면 $\hat b$는 0, 즉 population mean 쪽으로 더 강하게 shrink됩니다. 데이터가 충분하면 개체별 deviation을 더 많이 반영합니다.

**용어 설명 — BLUP(Best Linear Unbiased Predictor)**  
주어진 LMM 가정 아래 random effect를 선형·불편 조건에서 평균제곱오차가 작도록 예측하는 값입니다. “best”는 모든 가능한 예측법 중 절대 최고라는 뜻이 아니라 해당 조건 안에서의 의미입니다.

## 12. Dropout과 Missingness

longitudinal study에서는 시간이 지나면서 일부 subject가 측정을 중단할 수 있습니다. 문제는 dropout 이유가 response와 관련될 수 있다는 것입니다.

- **MCAR**: dropout이 response/covariate와 무관합니다.
- **MAR**: 관측된 과거 정보에 조건부로 dropout이 미관측값에 더 이상 의존하지 않습니다.
- **MNAR**: 미관측 outcome 자체와 dropout이 연결됩니다.

교재는 repeated-measures data에서 missingness가 분석에 중요한 문제이며 단순 complete-case가 편향을 만들 수 있음을 1장과 연결해 설명합니다.

## 13. 저자가 직접 보고한 결과 vs. 이 노트의 해석

### 13.1 저자 보고

- Timber slippage 독립 OLS는 $R^2=0.919$와 slope $10.373$이라는 강한 관계를 보여주지만 독립성 가정이 부적절하다고 저자들이 지적합니다.
- random-intercept model은 same-subject covariance를 도입하지만 compound symmetry라는 강한 제약을 가집니다.
- random intercept + slope model은 시점/조건에 따라 variance와 covariance가 달라질 수 있어 더 현실적인 구조를 제공합니다.
- ML은 variance components를 과소추정하는 경향이 있어 REML을 흔히 권장합니다.
- dropout은 longitudinal data 해석에서 별도로 고려해야 합니다.

### 13.2 해석

이 장의 가장 중요한 실무 교훈은 **높은 point-prediction fit과 올바른 uncertainty model은 별개**라는 것입니다. OLS가 $R^2=0.919$여도 correlation structure를 무시하면 coefficient uncertainty와 hypothesis test가 틀릴 수 있습니다. 반대로 LMM은 단순히 standard error를 고치는 모델이 아니라 random effect를 통해 new/known group prediction을 구분하는 predictive model로도 볼 수 있습니다.

## 14. 통계적으로 취약한 부분

1. **Random-effect normality**: group effect가 heavy-tail 또는 multimodal이면 Gaussian $G$가 부적절할 수 있습니다.
2. **Random-effect structure 선택**: random slope를 무조건 추가하면 소표본에서 singular fit이 생길 수 있습니다.
3. **Within-subject residual autocorrelation**: random slope를 넣어도 시간적으로 남는 AR(1) correlation이 있을 수 있습니다.
4. **Informative dropout**: standard LMM만으로 MNAR를 해결할 수 없습니다.
5. **새 subject vs 기존 subject**: conditional prediction과 marginal prediction의 성능이 다릅니다.
6. **row-level cross-validation leakage**: 동일 subject의 일부 관측이 train과 test에 동시에 있으면 new-subject generalization을 과대평가합니다.
7. **REML-LRT 오용**: fixed effects가 다른 REML model의 likelihood를 직접 비교하면 안 됩니다.

## 15. 비교 불가능한 수치

- OLS $R^2=0.919$와 LMM likelihood/AIC는 같은 평가척도가 아닙니다.
- conditional $R^2$와 marginal $R^2$는 random effect 포함 여부가 달라 직접 같은 의미로 읽으면 안 됩니다.
- 최신 mixed-effects neural network의 RMSE/R²와 교재 timber example은 outcome 단위·데이터셋·분할이 달라 절대 성능을 비교할 수 없습니다.
- subject-level test와 row-level random test의 metric도 일반화 대상이 다르므로 직접 비교하면 안 됩니다.

## 16. 문서가 직접 답하지 않는 질문과 답변

### 질문 1. 새로운 subject에는 random effect가 없는데 어떻게 예측하는가?

완전히 새로운 subject라면 관측 history가 없으므로 기본적으로 $b=0$인 population-level prediction을 사용합니다. 몇 개 관측이 쌓이면 posterior/BLUP를 업데이트하여 personalized prediction을 만들 수 있습니다.

### 질문 2. chamber마다 random intercept만 둘지 slope도 둘지 어떻게 결정하는가?

domain logic, trajectory plot, likelihood/AIC, variance component 안정성, OOS group validation을 함께 봅니다. random slope variance가 거의 0이고 model이 singular하면 더 단순한 random-intercept model이 낫습니다.

### 질문 3. LMM과 neural network를 결합할 수 있는가?

가능합니다. fixed nonlinear part를 neural network로 두고 group random effects를 별도로 결합하는 mixed-effects neural network가 한 방향입니다. 다만 small-sample에서는 neural component가 overfit될 수 있어 strong regularization과 group-aware validation이 필요합니다.

## 17. 일반화 성능 향상 가능성

- subject/group 단위 split으로 leakage 방지
- known-subject future prediction과 new-subject prediction을 별도 평가
- random intercept/slope를 domain 구조에 맞게 제한
- REML 또는 Bayesian partial pooling으로 variance 안정화
- heteroskedastic residual 또는 AR(1) residual 비교
- missingness model과 analysis model compatibility 확보
- time-varying coefficient가 있으면 random slope 외 state-space/dynamic mixed model 비교
- nonlinear fixed effect가 명확하면 spline/GAMM 또는 mixed-effects NN 후보
- new group이 많다면 hierarchical prior로 group estimate shrinkage 강화

## 18. 2020년 이후 관련 최신 연구 비교 분석

### 18.1 Longitudinal LMM 실무 가이드

**Murphy, Weaver & Hendricks, “Accessible analysis of longitudinal data with linear mixed effects models”, Disease Models & Mechanisms, 2022.**

longitudinal data에서 LMM을 실용적으로 설명하고, repeated observations의 correlation을 무시하는 단순 분석이 false positive/false negative와 해석 문제를 만들 수 있음을 다룹니다. 교재의 핵심 메시지를 현대 분석 workflow 관점에서 보강합니다.

### 18.2 LMM을 목표분석으로 하는 Multiple Imputation

**Huque et al., “Multiple imputation methods for handling incomplete longitudinal and clustered data where the target analysis is a linear mixed effects model”, Biometrical Journal, 2020.**

random intercept와 random slope가 있는 LMM을 target analysis로 두고 7개 MI 방법을 simulation에서 비교합니다. 저자들은 imputation model과 analysis model의 compatibility가 중요하며, 적절히 compatible한 방법이 regression coefficient와 variance component를 일관되게 추정할 수 있음을 보고합니다.

**용어 설명 — congeniality / compatibility**  
imputation model이 최종 analysis model의 주요 구조를 충분히 포함하여 대체과정이 분석모형과 모순되지 않는 성질입니다.

### 18.3 Neural Networks + Mixed Effects

**Mandel, Ghosh & Barnett, “Neural Networks for Clustered and Longitudinal Data Using Mixed Effects Models”, Biometrics, published 2021 / Vol. 79, 2023.**

individual-level effect와 population-level nonlinear prediction을 함께 활용하기 위해 neural network와 mixed-effects structure를 결합하는 방법을 다룹니다. 모바일 health처럼 future prediction이 중요한 longitudinal data에 LMM 개념을 확장합니다.

### 18.4 실제 Mixed-Effects Neural Network 사례

**“Mixed-effects neural network modelling to predict longitudinal trends in fasting plasma glucose”, BMC Medical Research Methodology, 2024.**

779명 데이터에서 LME, back-propagation NN, mixed-effects NN(LMENN)을 비교했습니다. 저자 보고에 따르면 LMENN의 RMSE 범위는 train $0.447$–$0.471$, validation $0.525$–$0.552$, test $0.511$–$0.565$였고 test $R^2$는 $0.680$–$0.738$ 범위였습니다. 가장 좋은 보고 모델 중 하나는 test RMSE $0.511$, MAE $0.359$, $R^2=0.738$이었습니다. 저자들은 10-fold CV를 사용했지만 $n=779$가 neural network에 충분히 크지 않을 수 있어 overfitting과 generalizability 한계를 직접 언급합니다.

이 수치는 교재 timber 데이터와 **직접 비교할 수 없습니다**. outcome, sample, split, loss, 단위가 모두 다릅니다.

### 18.5 Bayesian Joint Modeling

**Alsefri et al., “Bayesian joint modelling of longitudinal and time to event data: a methodological review”, BMC Medical Research Methodology, 2020.**

89개 논문을 검토했고 longitudinal submodel로 LMM, event submodel로 proportional hazards를 결합하는 구조가 가장 흔했으며, MCMC 사용 비율이 93%였다고 보고합니다. dynamic prediction을 주목적으로 다룬 논문은 6개뿐이라고 지적합니다.

**Ouko, Mukaka & Ohuma, “Joint modelling of longitudinal data: a scoping review of methodology and applications for non-time to event data”, BMC Medical Research Methodology, 2025.**

검색된 4,681개 기록 중 74개 연구를 포함했고, 포함 연구의 86%가 2014–2024년에 출판된 것으로 보고합니다. 이는 LMM 기반 joint modeling이 time-to-event뿐 아니라 여러 longitudinal outcome을 함께 다루는 방향으로 확대되고 있음을 보여줍니다.

### 비교표

| 접근 | 평균구조 | group correlation | 비선형성 | missing/event 확장 | 주요 위험 |
|---|---|---|---|---|---|
| Classical LMM | 선형 | random effects | 제한적 | 별도 모델 필요 | misspecification |
| GAMM/spline LMM | 비선형 smooth | random effects | 중간 | 별도 | smoothing 선택 |
| Mixed-effects NN | NN | random effects | 높음 | 확장 가능 | overfit·표본 요구 |
| Joint model | LMM 등 | random effects | 모델에 따라 | longitudinal+event/outcome | 계산·association 가정 |
| Bayesian hierarchical | 유연 | posterior pooling | 유연 | 자연스러운 uncertainty | prior·MCMC 비용 |

## 19. 실제 파이프라인 적용 시 고려할 점

```text
1) 예측 대상 정의
   - 기존 subject의 미래인가?
   - 완전히 새로운 subject인가?
2) 그 정의에 맞춰 group/time split
3) Train-only imputation/scaling
4) trajectory plot으로 random intercept/slope 필요성 확인
5) LMM 후보 구성
   - RI
   - RI + RS
   - residual AR(1)/heteroskedastic 구조
6) REML로 variance 구조 안정화
7) fixed-effect 비교가 필요하면 적절한 ML/REML 전략 선택
8) Validation에서 marginal/conditional prediction 분리 평가
9) dropout/missing mechanism sensitivity
10) Test에서 group-aware OOS metric 보고
```

### VM·공정 데이터에 적용하면

$$
y_{it}
=\beta_0+\beta^\top x_{it}
+b_{0,g(i)}+b_{1,g(i)}t+\epsilon_{it}
$$

처럼 tool/chamber $g(i)$별 random intercept와 time slope를 둘 수 있습니다.

- $b_{0,g}$: chamber별 offset입니다.
- $b_{1,g}$: chamber별 drift slope입니다.

하지만 새로운 chamber가 test에 등장하면 해당 $b_g$를 미리 알 수 없으므로 population prediction과 few-shot adaptation을 구분해야 합니다. 이것이 group-aware generalization의 핵심입니다.

## 20. 시사점과 후속 연구 방향

교재의 LMM은 오늘날에도 반복측정·계층 데이터의 핵심 baseline입니다. 후속 연구는 (1) random coefficient drift와 state-space dynamic coefficient의 비교, (2) heteroskedastic random effect, (3) mixed-effects NN이 소표본에서 실제로 이득인지, (4) new-chamber cold-start prediction, (5) dropout/missingness와 target process의 joint modeling, (6) partial pooling strength와 OOS $R^2$의 관계를 연구할 수 있습니다.

특히 연구자가 될 관점에서는 “random effect를 넣었더니 fit이 좋아졌다”보다 **어떤 generalization target에 대해 어떤 수준의 pooling이 최적인가**를 질문하는 것이 중요합니다.

## 21. 빠른 이해 점검

- OLS가 $R^2=0.919$여도 repeated measures에 부적절할 수 있는 이유는 무엇인가?
- random intercept가 같은 subject의 covariance를 어떻게 만들어내는가?
- random slope가 추가되면 residual variance가 왜 $x_j$에 따라 달라지는가?
- partial pooling이 small group의 overfit을 줄이는 이유는 무엇인가?
- new-subject prediction과 known-subject prediction의 차이는 무엇인가?

## 22. 참고자료

### 교재
- Brian S. Everitt & Torsten Hothorn, *An Introduction to Applied Multivariate Analysis with R*, Springer, 2011, Chapter 8.
- Diggle, Heagerty, Liang & Zeger, *Analysis of Longitudinal Data*, 2003. 교재의 repeated-measures 이론 주요 참고문헌.
- Skrondal & Rabe-Hesketh, *Generalized Latent Variable Modeling*, 2004. 교재의 random-effects 관련 참고문헌.

### 2020년 이후 확장 연구 및 사이트
- Jessica I. Murphy, Nicholas E. Weaver & Audrey E. Hendricks, “Accessible analysis of longitudinal data with linear mixed effects models”, *Disease Models & Mechanisms*, 2022. Source site: PubMed Central / journal site.
- Md Hamidul Huque et al., “Multiple imputation methods for handling incomplete longitudinal and clustered data where the target analysis is a linear mixed effects model”, *Biometrical Journal*, 62(2), 444–466, 2020. Source sites: Wiley Online Library / PubMed Central.
- Francesca Mandel, Riddhi Pratim Ghosh & Ian Barnett, “Neural Networks for Clustered and Longitudinal Data Using Mixed Effects Models”, *Biometrics*, Vol. 79(2), 711–721, 2023; first published 2021. Source sites: Oxford Academic / PubMed.
- “Mixed-effects neural network modelling to predict longitudinal trends in fasting plasma glucose”, *BMC Medical Research Methodology*, Vol. 24, Article 313, 2024. Source site: Springer Nature / BMC.
- Maha Alsefri et al., “Bayesian joint modelling of longitudinal and time to event data: a methodological review”, *BMC Medical Research Methodology*, 20:94, 2020. Source site: Springer Nature / PubMed Central.
- Rehema K. Ouko, Mavuto Mukaka & Eric O. Ohuma, “Joint modelling of longitudinal data: a scoping review of methodology and applications for non-time to event data”, *BMC Medical Research Methodology*, 25:40, 2025. Source site: Springer Nature / BMC.
