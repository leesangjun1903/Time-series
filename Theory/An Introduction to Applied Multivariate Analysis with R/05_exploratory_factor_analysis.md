# Chapter 5. Exploratory Factor Analysis

> 교재 범위: Chapter 5, pp. 135–162.  
> 핵심 주제: factor model, communality/uniqueness, principal factor, maximum likelihood factor analysis, factor number, rotation, factor scores, PCA와의 차이.

## 1. Executive Summary — 10문장 이내

1. Exploratory Factor Analysis(EFA)는 여러 관측변수 사이의 공분산이 소수의 관측되지 않은 latent factor와 변수별 고유오차로 생성된다고 가정하는 확률모형입니다.
2. 기본식 $x=\Lambda f+u$에서 $\Lambda$는 factor loading, $f$는 common factor, $u$는 변수별 specific component를 뜻합니다.
3. PCA가 전체 분산을 저차원 선형결합으로 압축하는 것과 달리 EFA는 **공통분산(common covariance)**을 잠재요인으로 설명하고 specific variance를 분리합니다.
4. factor loading은 관측변수와 요인 사이의 연결 강도를 나타내며 communality는 한 변수가 공통요인들로 설명되는 분산의 양입니다.
5. EFA는 회전(rotation)에 대해 해가 유일하지 않으므로 통계적으로 동일한 공분산 적합도를 갖는 여러 loading pattern이 존재합니다.
6. factor 수를 너무 적게 정하면 공통구조를 놓치고, 너무 많이 정하면 요인이 fragmentation되어 해석성과 재현성이 떨어질 수 있습니다.
7. 교재의 13개 drug-use 변수, $n=1634$ 예제에서는 최대우도 검정이 6-factor를 요구했지만 저자들은 마지막 두 요인을 해석하기 어렵고 3–4 factor도 residual이 작다는 점을 들어 formal test가 overfitting을 유도할 수 있다고 지적합니다.
8. 이는 큰 표본에서 아주 작은 covariance mismatch도 유의하게 검출된다는 고전적 적합도 검정의 한계를 잘 보여줍니다.
9. 2020년 이후에는 Bayesian regularized EFA, regularized bifactor analysis, Exploratory Graph Analysis(EGA)처럼 factor 수와 loading sparsity, 안정성을 함께 다루는 연구가 확장되고 있습니다.
10. 일반화 성능을 위해서는 한 데이터에서 탐색한 factor structure를 그대로 확정하지 말고 bootstrap stability, holdout CFA, regularization을 통해 새 표본에서 재현되는지를 확인해야 합니다.

## 2. 목적과 필요성

여러 설문문항, 여러 센서, 여러 행동지표가 서로 상관되어 있을 때 우리는 “이 변수들이 사실 몇 개의 공통 원인에 의해 함께 움직이는가?”를 묻게 됩니다. EFA는 관측변수의 correlation/covariance를 그대로 기술하는 데서 한 단계 더 나아가, 그 관계를 소수의 잠재요인으로 설명하려고 합니다.

예를 들어 20개의 공정 센서가 있지만 실제로는 `thermal state`, `gas-flow state`, `RF state` 같은 몇 개의 공통 상태가 센서들을 함께 움직인다고 생각할 수 있습니다. EFA는 이런 가설을 데이터에서 탐색하는 도구가 됩니다.

**용어 설명 — latent variable / latent factor**  
직접 측정되지 않지만 여러 관측변수의 공통 변동을 설명한다고 가정하는 숨은 변수입니다. “잠재요인이 실제 물리적 실체임”을 통계모형만으로 증명하는 것은 아닙니다.

## 3. 기본 k-factor model

평균을 제거한 $q$차원 관측벡터 $x$를

$$
x=\Lambda f+u
$$

로 표현합니다.

- $x\in\mathbb R^q$: 관측변수 벡터입니다.
- $f\in\mathbb R^k$: $k$개의 common factor 벡터입니다.
- $\Lambda\in\mathbb R^{q\times k}$: factor loading matrix입니다.
- $u\in\mathbb R^q$: 각 변수의 specific factor 또는 uniqueness component입니다.
- $k<q$: 원변수보다 적은 수의 요인을 사용합니다.

전형적인 표준화 설정에서

$$
E[f]=0,\qquad \text{Cov}(f)=I_k
$$

이며 specific component는

$$
E[u]=0,\qquad \text{Cov}(u)=\Psi
$$

로 둡니다. $\Psi$는 보통 대각행렬입니다.

또한

$$
\text{Cov}(f,u)=0
$$

을 가정합니다.

## 4. 공분산 분해

위 가정으로부터

$$
\Sigma
=\text{Cov}(x)
=\Lambda\Lambda^\top+\Psi
$$

가 됩니다.

이 식이 factor analysis 전체의 핵심입니다.

- $\Lambda\Lambda^\top$: 여러 관측변수가 common factor를 공유해서 생기는 공통 공분산입니다.
- $\Psi$: 각 변수에 특수한 잔여분산입니다.

$i$번째 변수에 대해

$$
\sigma_i^2
=\sum_{j=1}^{k}\lambda_{ij}^2+\psi_i
$$

입니다.

여기서

$$
h_i^2=\sum_{j=1}^{k}\lambda_{ij}^2
$$

를 communality라고 합니다.

- $\lambda_{ij}$: 변수 $i$가 factor $j$와 연결되는 loading입니다.
- $h_i^2$: 변수 $i$ 분산 중 common factors가 설명하는 부분입니다.
- $\psi_i$: uniqueness, 즉 factor model에 의해 공통적으로 설명되지 않는 변수별 분산입니다.

**용어 설명 — communality**  
관측변수 하나가 여러 공통요인으로부터 얼마나 많은 분산을 공유받는지 나타냅니다. 표준화 변수라면 $h_i^2$가 1에 가까울수록 공통요인으로 대부분 설명된다는 의미입니다.

## 5. 공분산이 어떻게 factor loading으로 만들어지는가?

서로 다른 변수 $x_i,x_j$의 공분산은 specific factor끼리 독립이라는 가정 아래

$$
\sigma_{ij}
=\sum_{l=1}^{k}\lambda_{il}\lambda_{jl}
$$

이 됩니다.

즉 두 변수가 같은 factor에 같은 방향의 큰 loading을 가지면 양의 공분산이 커집니다. 반대 부호의 loading을 가지면 음의 공분산이 만들어질 수 있습니다.

이 식을 통해 EFA는 “관측된 covariance matrix를 어떤 loading pattern이 설명할 수 있는가?”라는 inverse problem이 됩니다.

## 6. Scale invariance와 PCA와의 차이

교재는 factor model이 변수 rescaling에 대해 본질적으로 동일한 구조를 가질 수 있음을 설명합니다. 반면 PCA는 covariance matrix를 쓰느냐 correlation matrix를 쓰느냐에 따라 결과가 크게 달라질 수 있습니다.

### PCA

$$
\Sigma\approx V_m\Lambda_mV_m^\top
$$

처럼 전체 분산을 low-rank로 근사합니다.

### EFA

$$
\Sigma\approx\Lambda\Lambda^\top+\Psi
$$

처럼 공통분산과 변수별 고유분산을 분리합니다.

**핵심 차이**  
PCA는 데이터 변환/압축이고, EFA는 covariance를 설명하는 latent-variable model입니다. 둘 다 차원을 줄이지만 질문이 다릅니다.

## 7. Principal Factor Analysis

초기 uniqueness 추정치 $\hat\Psi$를 두고 reduced covariance matrix

$$
S^*=S-\hat\Psi
$$

를 만든 뒤, 이 행렬의 주요 eigenstructure를 통해 loading을 추정하고 다시 communality/uniqueness를 갱신하는 iterative 방식으로 이해할 수 있습니다.

**용어 설명 — reduced covariance matrix**  
관측 공분산 $S$에서 변수별 specific variance 추정치를 제거하여 공통요인으로 설명할 부분만 남기려는 행렬입니다.

## 8. Maximum Likelihood Factor Analysis

다변량 정규분포를 가정할 때 model-implied covariance

$$
\Sigma(\theta)=\Lambda\Lambda^\top+\Psi
$$

와 표본공분산 $S$의 차이를 최대우도 discrepancy로 최소화합니다. 교재가 사용하는 형태는

$$
F
=\log|\Lambda\Lambda^\top+\Psi|
+\text{tr}\left[S(\Lambda\Lambda^\top+\Psi)^{-1}\right]
-\log|S|-q
$$

입니다.

- $|\cdot|$: determinant입니다.
- $\text{tr}(\cdot)$: trace입니다.
- $q$: 관측변수 수입니다.

$F=0$에 가까울수록 model covariance가 sample covariance와 더 유사합니다.

### Heywood case

추정 과정에서 uniqueness가 0보다 작게 나오거나 communality가 관측분산을 넘어가는 비정상 해가 나타날 수 있습니다.

**용어 설명 — Heywood case**  
factor model의 parameter estimate가 admissible 범위를 벗어나는 현상입니다. 표본 부족, factor 수, model misspecification, 높은 상관 등 여러 원인이 있을 수 있습니다.

## 9. Factor 수 선택

ML factor analysis에서는 $k$개 factor가 충분하다는 귀무가설을 검정할 수 있습니다. 교재는

$$
U=N\min(F)
$$

을 사용하며

$$
N=n+1-\frac16(2q+5)-\frac23k
$$

이고 근사 자유도는

$$
\nu
=\frac12(q-k)^2-\frac12(q+k)
$$

입니다.

- $n$: 표본 수입니다.
- $q$: 관측변수 수입니다.
- $k$: factor 수입니다.
- $F$: 최소화된 ML discrepancy입니다.

작은 $k$부터 순차검정하여 기각되지 않는 첫 $k$를 선택할 수 있지만 교재 자체가 이 절차의 약점을 지적합니다. 여러 $k$에 대해 연속으로 검정하면서 critical value를 multiplicity에 맞게 조정하지 않기 때문입니다.

## 10. Rotation: 왜 같은 모형에서 여러 해가 생기는가?

orthogonal matrix $M$에 대해

$$
x=\Lambda f+u
$$

를

$$
x=(\Lambda M)(M^\top f)+u
$$

로 써도

$$
(\Lambda M)(\Lambda M)^\top
=\Lambda MM^\top\Lambda^\top
=\Lambda\Lambda^\top
$$

이므로 동일한 covariance를 예측합니다.

즉 loading matrix는 회전만으로 여러 형태가 가능하며, factor model은 기본적으로 rotationally indeterminate합니다.

**용어 설명 — Varimax rotation**  
각 factor에서 몇 개 변수는 큰 loading, 나머지는 작은 loading을 갖도록 loading 분산을 키워 “simple structure”를 찾는 대표적 orthogonal rotation입니다. factor 간 상관은 0으로 유지합니다.

**용어 설명 — oblique rotation**  
factor들이 서로 상관될 수 있도록 허용하는 rotation입니다. 실제 심리·생물학·공정 latent state가 독립이라는 보장이 없을 때 더 현실적일 수 있습니다.

## 11. Factor score

관측치마다 latent factor 값을 직접 관측한 것은 아니므로 $f_i$를 추정해야 합니다. 이를 factor score라고 합니다. 하지만 loading이 주어져도 score는 완전히 유일하지 않을 수 있으며 추정법에 따라 값이 달라질 수 있습니다.

따라서 factor score를 downstream regression feature로 사용할 때는 “진짜 latent variable을 관측했다”고 취급하지 않고 **추정오차가 있는 derived feature**로 봐야 합니다.

## 12. 저자가 직접 보고한 결과: Drug-use 예제

13개 substance에 대한 correlation matrix와 $n=1634$를 이용한 ML factor analysis에서 교재가 보고한 $k=1,\ldots,6$ factor 검정 p-value는

$$
0,
\;9.786\times10^{-70},
\;7.364\times10^{-28},
\;1.795\times10^{-11},
\;3.892\times10^{-6},
\;0.09753
$$

입니다.

따라서 형식적 검정만 보면 6-factor가 처음으로 기각되지 않습니다. 6-factor Varimax solution에서 교재는

- Factor 1: cigarettes, beer, wine, liquor, marijuana → “social/soft drug use”
- Factor 2: cocaine, tranquillizers, heroin → “hard drug use”
- Factor 3: 주로 amphetamine
- Factor 4: 주로 hashish
- Factor 5–6: 저자들이 설득력 있게 해석하지 않음

으로 설명합니다.

6-factor sufficient hypothesis에 대한 결과는

$$
\chi^2=22.41,\qquad df=15,\qquad p=0.0975
$$

이며 누적 설명 비율은 교재 출력에서 약 $0.549$까지 제시됩니다.

그러나 저자들은 13개 manifest variable에 6 factor는 만족스럽지 않고, 3-factor와 4-factor solution도 residual correlation이 작아 formal test가 이 사례에서 overfitting을 유도할 수 있다고 명시합니다.

## 13. 저자 결과와 이 노트의 해석 분리

### 저자 보고

위의 p-value와 $\chi^2$ 결과, factor interpretation, 3–4 factor residual도 작다는 관찰은 교재에 직접 보고된 내용입니다.

### 해석

이 예제는 통계적 유의성과 scientific usefulness가 다를 수 있다는 매우 좋은 사례입니다. $n=1634$처럼 큰 표본에서는 covariance mismatch가 작아도 검정이 이를 감지하여 추가 factor를 요구할 수 있습니다. 하지만 factor를 계속 늘리면 interpretability와 future-sample stability가 떨어질 수 있습니다. 따라서 factor 수는 goodness-of-fit만이 아니라 **parsimony, interpretability, bootstrap stability, replication**을 함께 보아야 합니다.

## 14. 통계적으로 취약한 부분

1. **Sequential testing multiplicity**: 교재가 직접 지적하듯 $k$를 늘려가며 반복검정할 때 다중검정 보정이 없습니다.
2. **Large-$n$ sensitivity**: 아주 작은 model misfit도 유의해질 수 있습니다.
3. **Rotational indeterminacy**: 같은 covariance fit을 갖는 여러 loading structure가 가능합니다.
4. **Factor reification**: factor를 실제 존재하는 실체처럼 해석하는 오류가 가능합니다.
5. **Normality assumption**: ML factor analysis의 전통적 추론은 multivariate normality에 의존합니다.
6. **Binary/ordinal items**: Pearson correlation과 Gaussian FA를 그대로 쓰는 것이 부적절할 수 있으며 polychoric/tetrachoric correlation 또는 categorical factor model이 필요할 수 있습니다.
7. **동일 데이터에서 탐색과 확인**: EFA로 factor structure를 정하고 같은 데이터에서 CFA로 “확인”하면 독립 검증이 아닙니다.

## 15. 비교 불가능한 수치 표시

- $\chi^2$ p-value와 explained variance는 서로 같은 성능척도가 아닙니다.
- EFA의 covariance residual과 supervised model의 RMSE/R²는 직접 비교할 수 없습니다.
- 최신 BREFA의 variable/factor recovery 성능과 교재 drug-use의 6-factor $p=0.0975$는 서로 데이터·목표가 달라 절대 수치 비교가 불가능합니다.
- EGA의 structural consistency와 EFA의 likelihood fit도 다른 개념입니다.

## 16. 문서가 답하지 않는 질문과 답변

### 질문 1. Factor 수를 오늘날에는 어떻게 더 안정적으로 정하는가?

한 가지 지표 대신 parallel analysis, bootstrap stability, information criterion, cross-validation, EGA 등 여러 정보를 함께 비교하는 것이 좋습니다. 특히 exploratory 연구라면 factor 수가 sample resampling마다 얼마나 유지되는지 보는 것이 중요합니다.

### 질문 2. Factor가 서로 상관될 수 있는데 Varimax를 써도 되는가?

이론적으로 factor 간 상관이 plausible하다면 oblique rotation을 함께 비교해야 합니다. Orthogonal rotation으로 강제로 0 상관을 만들면 실제 구조를 지나치게 단순화할 수 있습니다.

### 질문 3. Factor score를 ML feature로 쓰면 일반화가 좋아지는가?

가능하지만 보장되지 않습니다. noise dimension을 줄이고 shared signal을 압축하면 도움이 될 수 있지만, target signal이 uniqueness에 존재하면 오히려 손실됩니다. 따라서 raw regularized regression과 factor-score regression을 validation에서 비교해야 합니다.

## 17. 일반화 성능 향상 가능성

EFA를 구조 탐색에서 generalizable representation으로 바꾸려면 다음이 중요합니다.

- train-only covariance/correlation estimation
- shrinkage correlation 또는 robust correlation 사용
- factor 수를 bootstrap/parallel analysis로 안정화
- loading sparsity를 regularization으로 유도
- oblique/orthogonal rotation 모두 검토
- factor structure를 independent holdout CFA로 검증
- factor score downstream prediction은 nested validation으로 평가
- categorical 변수는 적절한 latent response model 사용

## 18. 2020년 이후 관련 최신 연구 비교 분석

### 18.1 Bayesian Regularized EFA(BREFA)

**Jinsong Chen, “A Bayesian Regularized Approach to Exploratory Factor Analysis in One Step”, Structural Equation Modeling: A Multidisciplinary Journal, 2021.**

BREFA는 factor 수가 알려지지 않은 상황에서 bi-level Bayesian sparse group selection을 사용하여 factor level과 loading level에서 exact zero 추정을 만들 수 있도록 설계합니다. 전통 EFA가 factor 수 선택과 loading estimation을 여러 단계로 나누는 반면, regularization을 통해 두 문제를 한 framework에서 다룹니다.

**용어 설명 — spike-and-slab / sparse prior**  
parameter가 정확히 0일 가능성과 0이 아닌 연속값일 가능성을 함께 표현하여 불필요한 loading을 제거하는 Bayesian 변수선택 아이디어입니다.

### 18.2 Regularized Exploratory Bifactor Analysis

**Jung, Seo & Park, “Regularized Exploratory Bifactor Analysis With Small Sample Sizes”, Frontiers in Psychology, 2020.**

작은 표본에서 exploratory bifactor structure를 추정할 때 ULS factor analysis와 regularized EFA 계열을 simulation으로 검토합니다. 전통 ML이 표본 부족에서 불안정할 수 있는 상황에 regularization을 도입한다는 점이 중요합니다.

**용어 설명 — bifactor model**  
모든 item에 영향을 주는 general factor와 특정 item subset에만 영향을 주는 group factor를 동시에 두는 구조입니다.

### 18.3 Exploratory Graph Analysis(EGA)

**Golino, Christensen & Garrido, “Exploratory Graph Analysis in context”, Psicologia: Teoria e Prática, 2022.**

EGA는 변수의 network structure와 community detection을 이용해 dimension을 추정하는 대안적 framework입니다. 이 논문은 network loadings, total entropy fit, dynamic EGA, bootstrap EGA, random-intercept EGA, hierarchical EGA 등 최근 확장을 정리합니다. 특히 bootstrap EGA는 item stability와 structural consistency를 통해 탐색된 차원이 resampling에서 얼마나 재현되는지를 평가합니다.

### 비교표

| 방법 | factor 수/구조 | regularization | 강점 | 주의점 |
|---|---|---|---|---|
| Classical EFA | 별도 선택 | 기본적으로 없음 | 해석 전통이 풍부 | rotation·factor 수 불안정 |
| BREFA | one-step Bayesian selection | 있음 | factor/loading sparsity | prior와 계산 복잡도 |
| Regularized bifactor EFA | bifactor 구조 | 있음 | small-sample 대응 | 구조가 실제로 bifactor인지 확인 필요 |
| EGA | network community로 dimension 추정 | graph estimation에 regularization 가능 | bootstrap stability | factor model과 동일한 생성가정은 아님 |

## 19. 실제 파이프라인 적용 시 고려할 점

```text
1) Train split에서 변수 타입과 correlation 종류 결정
2) Pearson / polychoric 등 적절한 association matrix 계산
3) sample size, q/n, condition number 확인
4) factor 수 후보: parallel analysis + ML + EGA/bootstrapping
5) factor extraction
6) orthogonal 및 oblique rotation 비교
7) loading stability bootstrap
8) factor meaning을 domain 지식과 연결
9) Validation 또는 별도 sample에서 CFA
10) downstream prediction이면 factor score와 raw feature 모델 비교
```

### 공정 데이터에서의 특별한 고려

센서 factor를 만들 때 tool/chamber/time이 섞여 있으면 하나의 global factor가 단지 장비 차이를 나타낼 수 있습니다. 따라서 group-centering, multilevel factor model, group별 loading invariance 검토가 필요합니다. future tool에 일반화하려면 기존 tool ID를 암묵적으로 encoding한 factor인지 확인해야 합니다.

## 20. 시사점과 후속 연구 방향

교재의 가장 중요한 경고는 factor model이 매력적인 설명을 제공하더라도 그것을 곧바로 실재하는 hidden cause로 취급해서는 안 된다는 것입니다. 후속 연구는 (1) ML-EFA vs BREFA vs EGA factor 수 안정성, (2) bootstrap loading reproducibility, (3) factor-score 예측과 raw Ridge/PLS 비교, (4) multilevel EFA로 chamber-specific vs common factors 분해, (5) 시계열 factor loading drift를 추적하는 dynamic factor model로 확장할 수 있습니다.

## 21. 빠른 이해 점검

- $\Sigma=\Lambda\Lambda^\top+\Psi$에서 두 항은 각각 어떤 분산을 의미하는가?
- communality와 uniqueness의 차이는 무엇인가?
- EFA와 PCA가 모두 차원축소처럼 보이지만 목적이 다른 이유는 무엇인가?
- 왜 rotation 후에도 model-implied covariance가 같은가?
- drug-use 예제에서 6-factor가 통계적으로 통과했는데도 저자들이 6-factor를 비판적으로 본 이유는 무엇인가?

## 22. 참고자료

### 교재
- Brian S. Everitt & Torsten Hothorn, *An Introduction to Applied Multivariate Analysis with R*, Springer, 2011, Chapter 5.
- Lawley & Maxwell, *Factor Analysis as a Statistical Method*. 교재의 factor analysis 이론 참고문헌.
- Jolliffe, *Principal Component Analysis*, PCA와 FA 비교에서 교재가 인용하는 참고문헌.

### 2020년 이후 확장 연구 및 사이트
- Jinsong Chen, “A Bayesian Regularized Approach to Exploratory Factor Analysis in One Step”, *Structural Equation Modeling: A Multidisciplinary Journal*, 28(4), 518–528, 2021. Source sites: Taylor & Francis; HKU Scholars Hub.
- Sunho Jung, Dong Gi Seo & Jungkyu Park, “Regularized Exploratory Bifactor Analysis With Small Sample Sizes”, *Frontiers in Psychology*, 11:507, 2020. Source site: Frontiers.
- Hudson Golino, Alexander P. Christensen & Luis Eduardo Garrido, “Exploratory Graph Analysis in context”, *Psicologia: Teoria e Prática*, 24(3), 2022. Source site: PePSIC/BVS.
