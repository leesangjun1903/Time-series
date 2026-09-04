# Chapter 2. Looking at Multivariate Data: Visualisation

> 교재 범위: Chapter 2, pp. 25–60.  
> 핵심 주제: scatterplot, bivariate boxplot, convex hull, chi-plot, glyph, scatterplot matrix, kernel density estimation, 3-D plot, trellis graphics, stalactite plot.

## 1. Executive Summary — 10문장 이내

1. 이 장의 핵심은 다변량 분석에서 그래프가 단순한 장식이 아니라 **모델을 만들기 전에 구조·이상치·비선형성을 발견하는 분석 도구**라는 것입니다.
2. 가장 기본적인 산점도는 두 변수 관계를 보여주며, 상관계수와 반드시 함께 보아야 이상치가 상관을 왜곡하는지 확인할 수 있습니다.
3. bivariate boxplot과 convex hull은 이변량 이상치에 더 구조적인 기준을 제공하지만, 이상치 제거 자체가 자동적으로 정당화되는 것은 아닙니다.
4. scatterplot matrix는 여러 변수의 모든 쌍 관계를 한 화면에서 비교하여 고차원 데이터의 비선형성·공선성·이상치를 찾는 데 유용합니다.
5. kernel density estimator(KDE)는 점의 밀도를 연속적인 확률밀도 형태로 근사하여 잠재적 군집을 시각적으로 드러낼 수 있습니다.
6. KDE에서는 kernel 모양보다 bandwidth $h$의 선택이 결과에 더 큰 영향을 주는 경우가 많습니다.
7. 3차원·glyph·star plot은 더 많은 변수를 한 번에 표시할 수 있지만, 정보량 증가가 곧 이해도 증가를 의미하지는 않습니다.
8. trellis graphics는 조건부 관계를 패널별로 분리하여 Simpson-type 혼합 효과를 발견하는 데 유용합니다.
9. 교재의 사례는 이상치 처리에 따라 상관이 $0.9553$에서 $0.7956$으로 크게 변할 수 있음을 직접 보여줍니다.
10. 2020년 이후 PaCMAP과 densMAP 같은 방법은 고차원 임베딩에서 local/global structure와 density 보존 문제를 다루지만, 교재의 직접 시각화와 목적함수가 달라 성능 수치를 단순 비교할 수 없습니다.

## 2. 목적과 필요성

다변량 데이터는 숫자 표만 보고 구조를 파악하기 어렵습니다. 예를 들어 상관행렬에 $r=0.8$이 적혀 있어도 실제 산점도를 보면 한 개의 극단점 때문에 상관이 커졌을 수 있고, 두 개의 군집이 섞여 생긴 가짜 선형관계일 수도 있습니다. 따라서 이 장의 목적은 **수치 요약 이전 또는 동시에 시각적 진단을 수행하는 습관**을 만드는 데 있습니다.

교재는 그래프의 목적을 크게 다음과 같이 봅니다: overview, story, hypothesis suggestion, model criticism. 이 장은 주로 앞의 세 가지를 다룹니다.

**용어 설명 — exploratory visualization**  
사전에 하나의 엄격한 가설만 검정하기보다 데이터가 가진 예상 밖 구조를 발견하기 위한 시각화입니다. 이 때문에 그래프에서 발견한 패턴은 후속 독립 검증 없이 확정적 결론으로 취급하면 안 됩니다.

## 3. Scatterplot과 상관계수

두 연속형 변수 $X,Y$의 산점도는 관측점 $(x_i,y_i)$를 직접 그립니다. 산점도를 통해 최소한 다음을 확인해야 합니다.

- 선형인지 비선형인지
- 분산이 $X$에 따라 달라지는지
- 한두 개 점이 관계를 지배하는지
- 두 개 이상의 군집이 섞였는지
- 범위 제한(range restriction)이 있는지

교재의 US air pollution 예제에서 manufacturing(`manu`)과 population(`popul`)의 Pearson 상관은 전체 데이터를 사용하면

$$
r=0.9553
$$

이지만 Chicago, Philadelphia, Detroit, Cleveland를 bivariate boxplot에서 이상치로 판단하여 제외하면

$$
r=0.7956
$$

으로 줄어듭니다.

이 결과는 “상관계수는 데이터의 그림 없이 해석하면 위험하다”는 점을 수치로 보여줍니다.

## 4. Bivariate Boxplot

단변량 boxplot을 2차원으로 확장한 개념으로, robust location, scale, correlation을 이용하여 중심부와 잠재적 이상치 경계를 타원으로 표현합니다. 교재의 설명에서 내부 `hinge`는 대략 50% 데이터를 포함하고, 외부 `fence` 밖의 점은 잠재적 이상치로 봅니다.

**용어 설명 — robust estimator**  
소수의 극단값이 들어와도 추정치가 크게 흔들리지 않도록 설계한 추정량입니다. 평균 대신 중앙값, 표준편차 대신 MAD를 쓰는 것이 대표적인 아이디어입니다.

### 왜 robust가 필요한가?

일반 평균과 공분산을 이용해 이상치를 찾으면 이상치가 평균과 공분산 자체를 끌어당겨 자신의 거리를 작게 만들 수 있습니다. 이를 **masking**이라고 합니다.

**용어 설명 — masking**  
이상치가 추정된 중심과 분산을 왜곡하여 자신 또는 다른 이상치가 정상처럼 보이게 만드는 현상입니다.

## 5. Convex Hull Trimming

2차원 점들을 모두 포함하는 가장 작은 볼록다각형의 꼭짓점을 제거한 뒤 상관을 다시 계산하는 방식입니다. 교재 예제에서는 convex hull을 제거한 상관이

$$
r=0.9225
$$

였습니다.

이 값은 bivariate boxplot으로 선택한 네 도시를 제거했을 때의 $0.7956$과 다릅니다. 즉 “어떤 점이 outlier인가?”라는 정의가 바뀌면 robust correlation도 달라집니다.

**통계적 교훈**  
이상치 처리 방법 자체가 분석자의 modeling choice이므로, 한 가지 처리 결과만 제시하기보다 원자료·robust 분석·제외 분석을 함께 보고 sensitivity를 확인하는 편이 안전합니다.

## 6. Chi-plot: 독립성의 시각적 진단

Chi-plot은 두 변수의 독립성 아래에서 transformed statistic $\chi_i$가 0 주변에서 비체계적으로 움직여야 한다는 성질을 이용합니다. 함께 쓰이는 $\lambda_i$는 각 점이 이변량 분포의 중심에서 얼마나 떨어졌는지를 나타냅니다.

교재의 `manu`와 `popul` 예제에서는 chi-plot의 중앙 독립성 band에 점이 거의 없어 독립에서 명확히 벗어나는 것으로 해석합니다.

**용어 설명 — independence**  
$X$를 안다고 해서 $Y$의 분포가 달라지지 않는 상태입니다. 상관 0보다 더 강한 개념입니다. 독립이면 적절한 조건 아래 공분산 0이지만, 공분산 0이 독립을 의미하지는 않습니다.

## 7. Scatterplot Matrix

$q$개 변수라면 가능한 변수쌍은

$$
\binom{q}{2}=\frac{q(q-1)}{2}
$$

개입니다. scatterplot matrix는 이들을 $q\times q$ 격자에 배치합니다.

- $q$: 변수 수입니다.
- $\binom{q}{2}$: 서로 다른 두 변수를 고르는 조합 수입니다.

교재의 air pollution 데이터에서는 `manu`와 `popul`의 강한 관계, `SO2`와 이 변수들의 관계, `precip`·`predays`와 `SO2` 사이의 비선형 가능성이 동시에 드러납니다. 이는 correlation matrix만으로는 놓칠 수 있는 정보입니다.

### 모델링으로 연결

산점도에서 곡률이 보인다면 회귀모형에 $x^2$, spline, kernel term을 고려할 근거가 생깁니다. 따라서 visualization은 단순 EDA가 아니라 **feature engineering의 근거**가 됩니다.

## 8. Kernel Density Estimation

### 8.1 1차원 KDE

교재가 소개하는 kernel density estimator는

$$
\hat f(x)=\frac{1}{nh}\sum_{i=1}^{n}K\left(\frac{x-x_i}{h}\right)
$$

입니다.

- $\hat f(x)$: 데이터에서 추정한 밀도입니다.
- $K(\cdot)$: kernel function입니다.
- $h>0$: bandwidth 또는 smoothing parameter입니다.
- $n$: 표본 수입니다.
- $x_i$: 실제 관측값입니다.

Gaussian kernel은

$$
K(u)=\frac{1}{\sqrt{2\pi}}e^{-u^2/2}
$$

입니다.

각 관측치 $x_i$ 위에 작은 봉우리 하나를 올리고 모두 합친다고 생각하면 됩니다.

### 8.2 Bandwidth의 역할

- $h$가 너무 작음: 작은 잡음까지 여러 peak로 표현되어 overfitting처럼 보입니다.
- $h$가 너무 큼: 실제 두 군집도 하나의 완만한 봉우리로 합쳐집니다.

**용어 설명 — bandwidth**  
각 관측치의 영향이 주변으로 얼마나 넓게 퍼질지를 정하는 폭입니다. KDE에서는 kernel 종류보다 bandwidth 선택이 결과 모양을 크게 좌우하는 경우가 많습니다.

### 8.3 2차원 KDE

이변량 데이터 $(x_i,y_i)$에서는

$$
\hat f(x,y)
=\frac{1}{nh_xh_y}\sum_{i=1}^{n}
K\left(\frac{x-x_i}{h_x},\frac{y-y_i}{h_y}\right)
$$

를 사용합니다.

- $h_x,h_y$: 각 좌표 방향의 smoothing 폭입니다.
- $K(u,v)$: 2차원 kernel입니다.

표준 이변량 Gaussian kernel은

$$
K(u,v)=\frac{1}{2\pi}\exp\left[-\frac12(u^2+v^2)\right]
$$

입니다.

## 9. KDE의 모델 구조와 한계

KDE는 학습 가능한 복잡한 neural network가 아니라 **비모수 밀도 추정기**입니다. 전체 구조는 다음과 같습니다.

```text
관측값
  ↓
각 관측치 중심에 kernel 배치
  ↓
bandwidth로 넓이 결정
  ↓
모든 kernel 평균
  ↓
연속적인 밀도 추정 \hat f
```

### 장점

- 정규분포 같은 특정 parametric shape를 강제하지 않습니다.
- multimodality를 시각화할 수 있습니다.
- 군집분석 전 데이터 구조를 점검하기 좋습니다.

### 한계

- 고차원에서는 필요한 표본 수가 급격히 늘어나는 curse of dimensionality가 있습니다.
- bandwidth에 민감합니다.
- density peak가 곧 실제 생성집단이라는 증거는 아닙니다.

**용어 설명 — curse of dimensionality**  
차원이 증가하면 공간의 부피가 매우 빠르게 커져 데이터가 희박해지고, 근접도·밀도 추정에 필요한 표본 수가 급격히 증가하는 현상입니다.

## 10. Bubble/Glyph/3-D Plot: 정보량과 인지부하의 충돌

Bubble plot은 $(x,y)$ 위치에 세 번째 변수의 크기를 원의 반지름 등으로 표현합니다. star plot이나 Chernoff face는 더 많은 변수를 symbol shape에 매핑합니다. 그러나 교재는 이런 그래프가 “많은 변수를 한 번에 보인다”는 이유만으로 효과적인 것은 아니라고 분명히 지적합니다.

3-D plot도 시점(rotation)에 따라 관계가 다르게 보이고, 정적인 인쇄물에서는 depth 판단이 어렵습니다. 따라서 고차원 정보를 억지로 3-D에 넣기보다 **차원축소 후 2-D 표시**가 더 나을 수 있으며, 이것이 다음 장 PCA로 자연스럽게 이어집니다.

## 11. Trellis Graphics와 조건부 관계

Trellis는 어떤 변수를 구간별로 나눈 뒤 동일한 그래프를 여러 panel에서 반복합니다. 교재의 air pollution 예제에서는 wind를 두 범위로 나누고 `SO2 ~ temp` 관계를 비교했을 때, 낮은 wind 구간에서는 온도가 증가할수록 오염이 감소하는 경향이 보이지만 높은 wind 구간에서는 관계가 약해 보인다고 설명합니다.

이것은 단일 전체 산점도에서 숨겨질 수 있는 **effect modification**을 찾는 방식입니다.

**용어 설명 — effect modification / interaction**  
$X$가 $Y$에 미치는 관계의 크기나 방향이 세 번째 변수 $Z$의 수준에 따라 달라지는 현상입니다. 회귀식에서는 흔히 $X\times Z$ interaction term으로 표현합니다.

## 12. Stalactite Plot과 반복적 이상치 진단

Stalactite plot은 처음부터 전체 표본의 평균·공분산만 쓰지 않고 작은 subset에서 시작해 subset 크기를 늘리면서 generalized distance 기준 이상치가 얼마나 지속적으로 유지되는지 보여줍니다. 목적은 masking을 완화하는 것입니다.

교재 air pollution 데이터에서 subset 크기가 커지면서 이상치 수가 줄고, 전체 41개 관측치를 사용했을 때 Chicago, Phoenix, Providence가 이상치로 남는 패턴을 보고합니다.

## 13. 저자가 직접 보고한 결과 vs. 이 노트의 해석

### 13.1 저자 보고

- `manu`–`popul` 전체 Pearson correlation: $0.9553$.
- bivariate boxplot에서 식별한 Chicago, Philadelphia, Detroit, Cleveland를 제거한 correlation: $0.7956$.
- convex hull point를 제거한 correlation: $0.9225$.
- CYG OB1 별 데이터의 bivariate KDE는 두 개의 서로 다른 cluster를 시사합니다.
- body measurement KDE에서는 waist/hips 관계가 두 group을 시사하며, 실제로 성별 두 집단이 존재합니다.
- 3-D와 star graphics가 항상 추가 정보를 주는 것은 아니며, scatterplot matrix가 많은 상황에서 더 유용하다고 결론 내립니다.

### 13.2 해석

세 상관값 $0.9553$, $0.7956$, $0.9225$는 “어떤 값이 진짜 상관인가?”를 결정해주는 것이 아니라 **robustness analysis의 필요성**을 보여줍니다. 실제 연구에서는 이상치를 임의 삭제하기보다 원자료 결과와 robust 추정 결과를 함께 보고해야 합니다. 또 KDE에서 두 peak가 보인다고 바로 두 population이라고 단정하지 않고, cluster stability나 domain label로 검증해야 합니다.

## 14. 통계적으로 취약한 부분과 비교 불가능한 수치

1. **그래프를 본 뒤 가설을 만든 뒤 같은 데이터로 검정**하면 double dipping이 생깁니다. 탐색에서 발견한 패턴은 새 데이터에서 검증해야 합니다.
2. **이상치 제거 후 상관 변화**는 예측 성능 향상과 동일하지 않습니다. 제거 기준이 test data까지 보고 정해졌다면 오히려 과대평가입니다.
3. **KDE peak 수**는 bandwidth에 따라 바뀔 수 있으므로 cluster 개수의 확정적 검정이 아닙니다.
4. **3-D·trellis 시각화**는 panel당 표본 수가 적으면 우연한 모양에 민감합니다.
5. **PaCMAP/densMAP 결과와 교재 scatterplot/KDE를 숫자로 직접 비교할 수 없습니다.** 전자는 고차원 embedding quality, 후자는 원변수 공간의 직접 관계 또는 density를 보여주는 도구로 목적함수가 다릅니다.

## 15. 문서가 직접 답하지 않는 질문과 답변

### 질문 1. 수백~수천 변수가 있으면 scatterplot matrix는 어떻게 하는가?

전 변수의 $q(q-1)/2$ 쌍을 그리는 것은 비현실적입니다. 먼저 domain-based selection, variance filter, supervised leakage-safe screening 또는 PCA를 통해 후보를 줄인 뒤 중요한 변수쌍을 확인합니다. 단, target을 사용해 변수를 고른 경우 그 선택은 train 내부에서만 해야 합니다.

### 질문 2. 임베딩에서 가까운 점은 원공간에서도 반드시 가까운가?

아닙니다. 모든 차원축소는 어떤 정보를 희생합니다. t-SNE/UMAP/PaCMAP처럼 2-D로 투영하는 방법은 local/global structure를 서로 다른 방식으로 보존하므로, 2-D 거리 자체를 원공간 거리의 정확한 대체물로 취급하면 안 됩니다.

### 질문 3. 시각화가 모델 일반화에 실제로 도움이 되는가?

간접적으로 매우 중요합니다. 시각화를 통해 leakage-like ID feature, temporal drift, nonlinear relation, heteroskedasticity, outlier regime를 발견하면 모델 specification을 개선할 수 있습니다. 그러나 시각화를 보고 모델을 반복 튜닝했다면 independent test set이 필요합니다.

## 16. 2020년 이후 관련 최신 연구 비교 분석

### 16.1 PaCMAP

**Wang, Huang, Rudin & Shaposhnik, “Understanding How Dimension Reduction Tools Work: An Empirical Approach to Deciphering t-SNE, UMAP, TriMap, and PaCMAP for Data Visualization”, JMLR, 2021.**

PaCMAP은 pairwise relation을 near, mid-near, further 구조로 조절하여 local structure뿐 아니라 global structure도 더 잘 보존하려는 임베딩입니다. 교재의 scatterplot matrix가 원변수 공간을 직접 보여주는 반면 PaCMAP은 고차원 구조를 2-D/3-D에 압축합니다.

### 16.2 densMAP

**Narayan, Berger & Cho, “Assessing single-cell transcriptomic variability through density-preserving data visualization”, Nature Biotechnology, 2021.**

densMAP은 t-SNE/UMAP류 시각화가 원공간의 local density를 왜곡할 수 있다는 문제를 다룹니다. 단순히 이웃만 보존하는 것이 아니라 **원공간의 밀도 차이도 임베딩에 반영**하려는 방향입니다.

### 비교

| 방법 | 주로 보존하려는 것 | 교재 방법과의 관계 | 주의점 |
|---|---|---|---|
| Scatterplot matrix | 원변수 쌍의 직접 관계 | 가장 해석 가능 | $q$가 크면 확장 어려움 |
| KDE contour | 원공간의 1-D/2-D density | 군집 후보 시각화 | bandwidth 민감 |
| PaCMAP | local + global geometric structure | 고차원 시각화 확장 | 2-D 거리 과해석 금지 |
| densMAP | neighborhood + density | KDE의 density 관점을 manifold embedding과 결합 | embedding hyperparameter 의존 |

## 17. 실제 파이프라인 적용 방향

```text
Train split 확정
  ↓
단변량 histogram / boxplot / missingness
  ↓
중요 변수 scatterplot + scatterplot matrix
  ↓
시간/장비/recipe 조건별 Trellis 또는 faceting
  ↓
KDE로 multimodality 후보 탐색
  ↓
robust distance로 outlier regime 확인
  ↓
고차원이면 PCA/PaCMAP 등 보조 임베딩
  ↓
발견한 가설을 feature/model 후보로 전환
  ↓
Validation에서 검증
  ↓
Test는 마지막 한 번만 사용
```

### 공정·시계열 데이터에서 특히 중요한 점

시간순 데이터는 무작위 scatterplot만 보면 drift를 놓칩니다. 따라서 각 변수의 `value vs time`, target residual vs time, chamber별 faceting을 먼저 보고, 그다음 변수쌍 관계를 보는 것이 좋습니다. 또한 future test 기간의 분포를 보고 preprocessing threshold를 정하면 leakage가 될 수 있으므로 시각화 단계에서도 train/validation/test 색을 구분하되 **의사결정은 train/validation에 한정**해야 합니다.

## 18. 일반화 성능 향상 가능성

이 장의 기술은 직접 예측기가 아니므로 “R²를 몇 % 높인다”는 식으로 말할 수 없습니다. 대신 다음 경로로 일반화에 기여할 수 있습니다.

- outlier-driven spurious correlation 제거 또는 robust 처리
- nonlinear pattern 발견 후 적절한 spline/kernel/tree 모델 선택
- subgroup interaction 발견 후 group-aware model 구성
- train–validation distribution shift 조기 발견
- density imbalance 발견 후 sample weighting 또는 stratified evaluation

즉 시각화는 모델의 **가설공간을 데이터 구조에 맞게 좁혀 variance와 misspecification을 줄이는 과정**입니다.

## 19. 시사점과 후속 연구 방향

교재의 핵심 메시지는 “좋은 그래프는 계산된 숫자의 보조물이 아니라 데이터의 질문 자체를 바꾼다”는 것입니다. 후속 연구로는 (1) scatterplot에서 발견한 비선형성을 spline/GAM이 실제 OOS 성능으로 회수하는지, (2) KDE에서 보인 multimodality가 실제 장비·recipe regime과 대응하는지, (3) PCA/PaCMAP/densMAP이 같은 공정 drift를 얼마나 안정적으로 보존하는지, (4) robust outlier 처리 전후 예측 성능과 calibration이 어떻게 변하는지를 비교할 수 있습니다.

## 20. 빠른 이해 점검

- 왜 $r=0.95$라는 숫자만 보고 강한 일반 관계라고 결론 내리면 안 되는가?
- KDE에서 $h$가 너무 작거나 클 때 각각 어떤 그림이 나오는가?
- scatterplot matrix와 PaCMAP은 어떤 정보를 서로 다르게 보존하는가?
- 이상치를 제거하는 것과 robust estimator를 쓰는 것의 차이는 무엇인가?

## 21. 참고자료

### 교재
- Brian S. Everitt & Torsten Hothorn, *An Introduction to Applied Multivariate Analysis with R*, Springer, 2011, Chapter 2.
- B. W. Silverman, *Density Estimation for Statistics and Data Analysis*, 1986. 교재의 KDE 이론 참고문헌.
- M. P. Wand & M. C. Jones, *Kernel Smoothing*, 1995. 교재의 비모수 밀도 추정 참고문헌.

### 2020년 이후 확장 연구 및 사이트
- Yingfan Wang, Haiyang Huang, Cynthia Rudin & Yaron Shaposhnik, “Understanding How Dimension Reduction Tools Work: An Empirical Approach to Deciphering t-SNE, UMAP, TriMap, and PaCMAP for Data Visualization”, *Journal of Machine Learning Research*, 22(201), 2021. Source site: JMLR.
- Ashwin Narayan, Bonnie Berger & Hyunghoon Cho, “Assessing single-cell transcriptomic variability through density-preserving data visualization”, *Nature Biotechnology*, 39, 765–774, 2021. Source site: Nature.
