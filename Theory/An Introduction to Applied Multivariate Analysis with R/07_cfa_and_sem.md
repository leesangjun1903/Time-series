# Chapter 7. Confirmatory Factor Analysis and Structural Equation Models

> 교재 범위: Chapter 7, pp. 201–224.  
> 핵심 주제: CFA, SEM, covariance structure, maximum likelihood estimation, identification, model fit, path models, causal interpretation의 한계.

## 1. Executive Summary — 10문장 이내

1. Confirmatory Factor Analysis(CFA)는 EFA에서 자유롭게 탐색하던 loading structure를 이론적으로 사전에 제한하여 “이 측정모형이 데이터의 공분산을 설명하는가?”를 검증합니다.
2. Structural Equation Model(SEM)은 CFA의 measurement model에 latent variable 사이의 regression/path relation을 추가한 더 일반적인 covariance structure model입니다.
3. 핵심 계산은 표본공분산 $S$와 model-implied covariance $\Sigma(\theta)$의 차이를 최소화하도록 parameter $\theta$를 추정하는 것입니다.
4. 최대우도 SEM에서는 $F_{ML}$을 최소화하고, $(N-1)F_{ML}$을 이용해 model fit을 평가할 수 있습니다.
5. parameter가 식별되지 않으면 서로 다른 $\theta$가 같은 $\Sigma(\theta)$를 만들 수 있으므로 고유한 추정이 불가능합니다.
6. 전통적 chi-square fit test는 큰 표본에서 사소한 misfit도 기각할 수 있으므로 RMSEA, SRMR, CFI/TLI류 지표와 residual, theory를 함께 봐야 합니다.
7. 교재 alienation SEM 예제는 chi-square와 RMSEA가 좋지 않은 반면 CFI/NFI/SRMR은 상대적으로 좋아, 적합도 지수들이 서로 다른 메시지를 줄 수 있음을 직접 보여줍니다.
8. 교재는 EFA에서 만든 모형을 같은 데이터로 CFA하면 안 되고 새로운 데이터에서 검증해야 한다고 명시합니다.
9. 2020년 이후 SEM 연구는 cross-validation, out-of-sample prediction, regularization, stability selection, Bayesian regularized SEM 등으로 설명 중심 SEM을 재현성과 예측 관점으로 확장하고 있습니다.
10. 일반화 성능을 높이려면 theory-driven specification, independent validation, regularization, misspecification sensitivity, multilevel/time structure를 함께 고려해야 합니다.

## 2. 목적과 필요성

EFA가 “몇 개 factor가 있고 어떤 변수가 어느 factor에 연결되는가?”를 탐색한다면 CFA는 연구자가 미리 정한 loading pattern을 데이터가 지지하는지 확인합니다. 예를 들어 6개 측정변수 중 4개는 `ability`, 2개는 `aspiration`을 측정하고 두 latent factor가 상관된다는 가설을 미리 정의할 수 있습니다.

SEM은 여기서 한 단계 더 나아가 latent variable 사이의 방향성 있는 회귀관계까지 표현합니다.

**용어 설명 — measurement model**  
latent variable이 어떤 observed variable을 통해 측정되는지를 정의하는 부분입니다. CFA가 주로 이 부분을 다룹니다.

**용어 설명 — structural model**  
latent variable 또는 observed variable 사이의 directional regression/path relation을 정의하는 부분입니다.

## 3. CFA의 기본 구조

관측벡터 $x$를

$$
x=\Lambda f+u
$$

로 두는 것은 EFA와 유사합니다. 차이는 CFA에서는 일부 loading을 이론에 따라 **사전에 0으로 고정**하거나 동일성 제약을 둔다는 점입니다.

예를 들어 두 factor $f_1,f_2$에 대해

$$
\begin{aligned}
X_1 &= \lambda_1f_1+0f_2+u_1,\\
X_2 &= \lambda_2f_1+0f_2+u_2,\\
X_3 &= 0f_1+\lambda_3f_2+u_3
\end{aligned}
$$

처럼 specification할 수 있습니다.

- $\lambda_j$: factor loading입니다.
- $u_j$: measurement-specific error/uniqueness입니다.
- 0으로 고정된 loading: 이론적으로 해당 factor가 그 관측변수를 직접 측정하지 않는다고 가정한다는 뜻입니다.

## 4. Model-implied covariance

SEM의 핵심은 모형이 관측변수의 covariance를 예측한다는 것입니다.

$$
\Sigma=\Sigma(\theta)
$$

- $\theta$: loading, regression coefficient, factor variance/covariance, error variance 등을 모두 모은 parameter vector입니다.
- $\Sigma(\theta)$: 해당 parameter에서 모형이 예측하는 covariance matrix입니다.

데이터의 sample covariance $S$와 $\Sigma(\theta)$가 비슷하도록 $\theta$를 추정합니다.

## 5. Maximum Likelihood Estimation

다변량 정규 가정 아래 교재가 제시하는 ML discrepancy는

$$
F_{ML}
=\log|\Sigma(\theta)|
-\log|S|
+\text{tr}\left[S\Sigma(\theta)^{-1}\right]
-q
$$

입니다.

- $|\Sigma(\theta)|$: model-implied covariance의 determinant입니다.
- $|S|$: sample covariance determinant입니다.
- $\text{tr}(\cdot)$: trace입니다.
- $q$: observed variable 수입니다.

$F_{ML}$을 최소화하는 $\hat\theta$를 찾습니다.

### Least-squares와 차이

단순히

$$
\sum_{i,j}(s_{ij}-\sigma_{ij}(\theta))^2
$$

를 최소화할 수도 있지만 변수 scale에 크게 좌우되고 covariance element의 sampling variance를 동일하게 취급하는 문제가 있습니다. ML은 확률모형에 기반한 추론을 가능하게 합니다.

## 6. Identification

교재는 model이 identified되기 위한 정의를

$$
\Sigma(\theta_1)=\Sigma(\theta_2)
\Rightarrow
\theta_1=\theta_2
$$

로 설명합니다.

즉 같은 관측 covariance를 만드는 서로 다른 parameter set이 없어야 합니다.

필요조건으로 free parameter 수 $t$가

$$
t<\frac{q(q+1)}{2}
$$

이어야 하지만, 이것만으로 identification이 보장되는 것은 아닙니다.

**용어 설명 — identification**  
관측 가능한 covariance 정보로 model parameter를 유일하게 결정할 수 있는 성질입니다. 식별되지 않은 parameter는 표본이 무한히 커져도 유일하게 추정할 수 없습니다.

## 7. Global chi-square fit test

최소화된 ML discrepancy로

$$
\chi^2=(N-1)F_{ML,\min}
$$

을 구성합니다.

자유도는

$$
\nu=\frac{q(q+1)}{2}-t
$$

입니다.

- $N$: sample size입니다.
- $t$: free parameter 수입니다.

귀무가설은 population covariance가 fitted model이 암시하는 covariance와 같다는 것입니다. 큰 $N$에서는 아주 작은 discrepancy도 유의해져 거의 모든 현실적 model을 기각할 수 있다는 한계가 있습니다.

## 8. Fit index는 하나만 보면 안 된다

교재는 likelihood-ratio statistic 외에 RMSR, Tucker-Lewis index, normed fit index 등을 언급합니다. 현대 SEM에서는 RMSEA, CFI, TLI, SRMR 등이 널리 사용되지만 fixed cutoff를 절대 규칙으로 쓰는 것은 위험합니다.

**용어 설명 — RMSEA**  
model의 자유도를 고려하여 population-level approximate misfit을 요약하는 지표입니다. 작은 값이 좋지만 threshold는 sample size와 model 조건에 따라 기계적으로 적용하면 안 됩니다.

**용어 설명 — SRMR**  
표준화된 observed correlation/covariance와 predicted 값의 residual 크기를 요약합니다. 작을수록 observed relation을 잘 재현합니다.

**용어 설명 — CFI/TLI**  
독립모형 같은 baseline model보다 현재 model이 얼마나 개선되었는지를 비교하는 incremental fit index입니다.

## 9. CFA 예제: Ability and Aspiration

교재의 Calsyn & Kenny 데이터는 556명의 학생에 대해 6개 manifest variable을 사용합니다.

- SCA, PPE, PTE, PFE → ability factor
- educational aspiration, college plans → aspiration factor
- 두 latent factor는 상관 허용

추정할 parameter는 loading 6개, specific variance 6개, factor correlation 1개로 총 13개입니다. $q=6$이면 covariance의 독립 원소 수는

$$
\frac{6(6+1)}{2}=21
$$

이므로 자유도는

$$
21-13=8
$$

입니다.

이 계산은 “관측 covariance 정보량과 free parameter 수의 차이”가 SEM 자유도가 된다는 직관을 보여줍니다.

## 10. Structural Equation Model의 구조

latent vector $\eta$가 다른 latent/exogenous variable $\xi$에 영향을 받는 구조는 일반적으로

$$
\eta=B\eta+\Gamma\xi+\zeta
$$

와 같은 형태로 쓸 수 있습니다.

- $B$: endogenous latent variables끼리의 path coefficient matrix입니다.
- $\Gamma$: exogenous variable에서 endogenous latent variable로 가는 coefficient입니다.
- $\zeta$: structural disturbance입니다.

측정모형과 구조모형을 결합하면 관측변수 covariance $\Sigma(\theta)$가 loading과 path parameter의 함수로 전개됩니다.

**용어 설명 — endogenous**  
모형 내부의 다른 변수에 의해 설명되는 변수입니다.

**용어 설명 — exogenous**  
모형 내에서 다른 변수의 결과로 설명되지 않고 외부에서 주어진 원인 쪽 변수입니다. 그렇다고 자동으로 causal intervention이 보장되는 것은 아닙니다.

## 11. 저자가 직접 보고한 결과: Alienation SEM

교재 alienation model의 초기 fit 결과는 다음과 같습니다.

$$
\chi^2=71.532,\qquad df=6,\qquad p=1.9829\times10^{-13}
$$

그리고

- GFI = 0.97514
- AGFI = 0.913
- RMSEA = 0.10831, 90% CI $(0.086636,0.13150)$
- NFI = 0.96644
- TLI/NNFI = 0.9226
- CFI = 0.96904
- SRMR = 0.021256

를 보고합니다.

일부 주요 structural coefficient는

$$
\beta_1=-0.61361,\qquad
\beta_2=-0.17447,\qquad
\beta_3=0.70463
$$

입니다.

저자들은 chi-square가 6 df에서 매우 크기 때문에 model fit이 좋지 않다고 평가합니다. 이후 1967과 1971의 anomia measurement error 사이 covariance를 허용하면

$$
\chi^2=6.359,\qquad df=5
$$

로 크게 개선됩니다.

그러나 교재는 단지 fit을 좋아 보이게 하려고 error covariance를 추가하는 행위를 비판적으로 다루며 **이론적 이유가 있어야 한다**고 강조합니다.

## 12. 저자 보고와 이 노트의 해석

### 저자 보고

위 alienation model의 fit index와 correlated error 추가 후 chi-square 개선은 교재에 직접 제시된 결과입니다. 교재는 또한 correlation data만으로 causal relation을 확정할 수 없으며, 누락된 변수 문제를 computer program이 해결해주지 못한다고 경고합니다.

### 해석

초기 model은 CFI/NFI와 SRMR만 보면 좋아 보일 수 있지만 RMSEA와 chi-square는 상당한 misfit을 가리킵니다. 이것은 fit index가 서로 다른 aspect를 측정하고 sample/model complexity에 다르게 반응하기 때문입니다. 따라서 “CFI > 특정 숫자이므로 model 승인” 같은 checklist 방식은 위험합니다.

또한 error covariance 추가는 data-driven modification index를 무제한 따라가면 training covariance에 overfit하는 과정이 될 수 있습니다. 새 표본에서 같은 residual correlation이 재현되는지 확인해야 합니다.

## 13. EFA → CFA에서 가장 중요한 일반화 원칙

교재는 EFA에서 얻은 structure를 CFA로 확인할 수 있지만 **같은 데이터를 사용해 생성하고 검증해서는 안 된다**고 명시합니다. 이것은 현대 ML의 train/validation separation과 정확히 같은 원리입니다.

```text
Dataset A
  ↓
EFA로 구조 탐색
  ↓
이론과 결합하여 CFA specification 고정
  ↓
Dataset B 또는 holdout sample
  ↓
CFA 검증
```

## 14. 통계적으로 취약한 부분

1. **큰 표본의 chi-square 민감성**: trivial misfit도 유의해질 수 있습니다.
2. **fit index cutoff 의존**: 보편적인 단일 threshold는 없습니다.
3. **modification index chasing**: 같은 데이터에 residual covariance/path를 계속 추가하면 overfit됩니다.
4. **identification을 자유도만으로 판단**: $t<q(q+1)/2$는 필요조건일 뿐 충분조건이 아닙니다.
5. **normality assumption**: non-normal data에서는 standard ML standard error와 chi-square가 왜곡될 수 있습니다.
6. **causal overinterpretation**: path arrow는 statistical direction specification이지 intervention evidence 자체가 아닙니다.
7. **measurement invariance 미검토**: group/time마다 loading/intercept가 달라지면 latent mean/path 비교가 왜곡될 수 있습니다.

## 15. 비교 불가능한 수치

- CFI, RMSEA, SRMR은 같은 scale의 성능지표가 아니므로 서로 평균내면 안 됩니다.
- in-sample covariance fit과 out-of-sample prediction error는 목적이 다릅니다.
- 최신 regularized SEM의 selection stability와 교재의 chi-square 감소량은 직접 비교할 수 없습니다.
- 서로 다른 model·sample에서 “CFI 0.97 vs 0.95”만 보고 우열을 단정하기 어렵습니다.

## 16. 문서가 직접 답하지 않는 질문과 답변

### 질문 1. SEM도 cross-validation이 가능한가?

가능합니다. train에서 parameter/model structure를 추정하고 validation/test에서 observed indicator를 이용해 prediction 또는 held-out covariance/log-likelihood를 평가할 수 있습니다. 2022년 SEM-based out-of-sample prediction 연구는 reflective SEM에서도 OOS prediction rule을 구성할 수 있음을 보여줍니다.

### 질문 2. parameter가 너무 많으면 어떻게 하는가?

LASSO/ridge-type regularized SEM, Bayesian shrinkage prior, stability selection을 사용할 수 있습니다. 단 regularization parameter 선택 자체가 또 하나의 model selection이므로 resampling이 필요합니다.

### 질문 3. SEM의 화살표를 causal graph로 봐도 되는가?

연구설계와 identification assumption이 갖춰진 경우에만 제한적으로 가능합니다. 단순 covariance data와 좋은 fit만으로 causal effect가 증명되지는 않습니다.

## 17. 일반화 성능 향상 가능성

- EFA와 CFA 데이터 분리
- predefined theory와 minimal modification
- robust ML / bootstrap SE for non-normality
- regularized SEM으로 불필요한 path 축소
- stability selection으로 selection false positive 완화
- Bayesian shrinkage로 small-sample variance 감소
- OOS prediction과 held-out covariance fit 병행
- multigroup/time measurement invariance 확인
- longitudinal이면 dynamic SEM 또는 mixed-effects/latent-growth model 비교

## 18. 2020년 이후 관련 최신 연구 비교 분석

### 18.1 SEM의 현재 방법론적 방향

**Zyphur, Bonner & Tay, “Structural Equation Modeling in Organizational Research: The State of Our Science and Some Proposals for Its Future”, Annual Review of Organizational Psychology and Organizational Behavior, 2023.**

현대 SEM의 확장을 latent interaction, nonlinear measurement model, multilevel SEM, cross-lagged panel, dynamic SEM, meta-analytic SEM 등으로 정리하고, cross-validation·regularization·causal evidence 문제를 중요한 향후 과제로 다룹니다.

### 18.2 SEM-Based Out-of-Sample Prediction

**de Rooij et al., “SEM-Based Out-of-Sample Predictions”, Structural Equation Modeling: A Multidisciplinary Journal, 2022.**

저자들은 reflective SEM에서도 out-of-sample prediction이 가능함을 보이고, 두 empirical example에서 SEM-based prediction이 linear regression보다 나은 사례를 보고합니다. simulation에서는 normality violation에 비교적 robust하지만 model misspecification에는 민감했다고 보고합니다.

**핵심 적용점**  
좋은 in-sample fit뿐 아니라 **새 관측치 prediction**을 별도 목표로 평가할 수 있습니다.

### 18.3 Regularized SEM with Stability Selection

**Li & Jacobucci, “Regularized Structural Equation Modeling With Stability Selection”, Psychological Methods, 2022.**

LASSO-type regularized SEM에서 하나의 sample에 따른 variable/path selection의 불안정을 줄이기 위해 반복 resampling과 stability selection을 결합합니다.

**용어 설명 — stability selection**  
데이터를 여러 번 부분표본화해 parameter가 얼마나 자주 선택되는지를 측정하고, 일정 빈도 이상 안정적으로 선택되는 항만 유지하는 방법입니다.

### 18.4 Bayesian Regularized SEM

**Sara van Erp, “Bayesian Regularized SEM: Current Capabilities and Constraints”, Psych, 2023.**

Bayesian prior를 이용해 parameter를 shrink하거나 sparsity를 유도하는 방법을 정리하고, 현재 구현 가능성과 한계를 검토합니다. 작은 표본·고차원 구조에서 유용할 수 있지만 prior sensitivity와 계산비용을 점검해야 합니다.

### 비교표

| 접근 | 핵심 목표 | 교재 SEM과의 차이 | 일반화 관점 |
|---|---|---|---|
| Classical ML SEM | covariance explanation | 교재 중심 | in-sample fit 중심 |
| OOS SEM prediction | 새 샘플 prediction | predictive rule 명시 | 직접 generalization 평가 |
| Regularized SEM | parameter shrinkage/selection | penalty 추가 | variance 감소 가능 |
| Stability selection | path selection 재현성 | resampling | false discovery 완화 |
| Bayesian regularized SEM | prior-based shrinkage | posterior inference | small-sample 안정화 가능 |

## 19. 실제 파이프라인 적용 시 고려할 점

```text
1) 이론적 measurement/structural model 작성
2) EFA를 했다면 CFA용 독립 sample 확보
3) Train에서 covariance와 model estimate
4) identification 확인
5) global fit + local residual 확인
6) modification은 theory 근거가 있는 경우만 제한적으로 수행
7) regularization/stability selection 후보 비교
8) Validation에서 held-out fit 또는 prediction 평가
9) group/time measurement invariance 점검
10) Test에서 최종 OOS performance와 uncertainty 보고
```

### 공정 데이터로 번역하면

latent `thermal state`, `RF state`, `chamber condition`을 여러 sensor indicator로 측정하고 이 latent state가 metrology target에 영향을 준다는 SEM을 만들 수 있습니다. 하지만 센서–latent loading이 chamber별로 다르면 global model이 잘못된 동일성 가정을 할 수 있으므로 multigroup 또는 multilevel SEM이 필요합니다.

## 20. 시사점과 후속 연구 방향

교재의 가장 중요한 메시지는 “모형 적합도가 좋아졌다는 이유만으로 scientific theory가 옳아지는 것은 아니다”라는 점입니다. 후속 연구로는 (1) classical SEM vs regularized SEM의 OOS covariance fit, (2) modification index path의 bootstrap stability, (3) latent score prediction과 raw ML prediction 비교, (4) chamber/time measurement invariance, (5) dynamic SEM과 mixed-effects model의 비교, (6) SEM의 causal claim을 intervention/temporal design과 결합하는 방향이 중요합니다.

## 21. 빠른 이해 점검

- EFA와 CFA의 가장 중요한 차이를 “사전에 0으로 고정하는 loading”으로 설명할 수 있는가?
- identification과 model fit은 왜 전혀 다른 문제인가?
- $\chi^2$가 큰 표본에서 너무 민감한 이유는 무엇인가?
- alienation 예제에서 CFI/SRMR과 RMSEA/chi-square가 다른 결론을 주는 이유를 설명할 수 있는가?
- correlated error를 추가할 때 이론적 이유가 필요한 이유는 무엇인가?

## 22. 참고자료

### 교재
- Brian S. Everitt & Torsten Hothorn, *An Introduction to Applied Multivariate Analysis with R*, Springer, 2011, Chapter 7.
- Bollen & Long, *Testing Structural Equation Models*, 1993. 교재의 identification/fit 논의 참고문헌.
- Jöreskog, Bentler, Browne 등의 covariance structure/SEM 고전 연구. 교재에서 이론 및 예제로 인용.

### 2020년 이후 확장 연구 및 사이트
- Michael J. Zyphur, Cavan V. Bonner & Louis Tay, “Structural Equation Modeling in Organizational Research: The State of Our Science and Some Proposals for Its Future”, *Annual Review of Organizational Psychology and Organizational Behavior*, Vol. 10, 2023. Source site: Annual Reviews.
- Mark de Rooij et al., “SEM-Based Out-of-Sample Predictions”, *Structural Equation Modeling: A Multidisciplinary Journal*, published online 2022. Source site: Taylor & Francis.
- X. Li & Ross Jacobucci, “Regularized Structural Equation Modeling With Stability Selection”, *Psychological Methods*, 27, 497–518, 2022. Source sites: APA journal / author publication page.
- Sara van Erp, “Bayesian Regularized SEM: Current Capabilities and Constraints”, *Psych*, 5(3), 814–835, 2023. Source site: MDPI.
