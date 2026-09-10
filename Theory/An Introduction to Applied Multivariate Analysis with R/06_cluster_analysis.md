# Chapter 6. Cluster Analysis

> 교재 범위: Chapter 6, pp. 163–200.  
> 핵심 주제: cluster의 정의, agglomerative hierarchical clustering, single/complete/group-average linkage, k-means, finite mixture, Gaussian model-based clustering, graphical validation.

## 1. Executive Summary — 10문장 이내

1. 군집분석은 사전에 정답 label이 없는 데이터에서 관측치들이 자연스럽게 몇 개의 집단으로 나뉘는지 탐색하는 unsupervised 방법입니다.
2. Hierarchical clustering은 관측치 사이 거리와 cluster-to-cluster linkage 정의에 따라 작은 cluster를 단계적으로 합쳐 dendrogram을 만듭니다.
3. Single, complete, average linkage는 같은 원거리행렬을 사용해도 cluster geometry를 다르게 정의하므로 서로 다른 해를 만들 수 있습니다.
4. K-means는 각 관측치를 가장 가까운 centroid에 할당하며 within-group sum of squares를 최소화하지만 local optimum, scaling, 구형 cluster 가정에 민감합니다.
5. Model-based clustering은 데이터가 여러 확률분포의 mixture에서 생성되었다고 가정하고 likelihood와 posterior membership probability로 cluster를 추정합니다.
6. Gaussian mixture는 각 cluster의 평균·공분산을 통해 서로 다른 크기·모양·방향을 표현할 수 있어 k-means보다 유연합니다.
7. 교재의 gastroenterologist 예제에서는 BIC가 VEV covariance structure와 3개 cluster를 선택했지만, 저자들은 binary-response proportion에 Gaussian mixture를 적용하는 것 자체가 완전히 자연스럽지는 않다고 경고합니다.
8. 군집분석에는 “정답”이 없는 경우가 많으므로 한 번 나온 군집을 사실로 선언하기보다 안정성, 분리도, 외부 domain 의미를 검증해야 합니다.
9. 2020년 이후 deep clustering은 representation learning과 clustering objective를 결합해 고차원 비선형 데이터에 대응하지만, 더 복잡한 모델이 항상 더 재현성 높은 cluster를 만드는 것은 아닙니다.
10. 실제 파이프라인에서는 scaling, distance, $k$, initialization, bootstrap stability, holdout likelihood 또는 downstream usefulness를 함께 검증해야 합니다.

## 2. 목적과 필요성

군집분석의 질문은 supervised learning과 다릅니다. supervised learning은 $y$가 주어진 상황에서 $X\rightarrow y$를 학습하지만, clustering은 $y$ 없이 $X$ 자체의 구조에서 비슷한 관측치를 묶습니다.

예를 들어 wafer run이 여러 공정센서의 조합으로 세 개 operating regime을 형성할 수 있습니다. 이때 cluster가 실제 recipe, chamber, aging state 또는 fault mode와 연결된다면 이후 모델을 group-aware하게 설계할 수 있습니다.

**용어 설명 — unsupervised learning**  
정답 label 없이 입력 데이터의 구조 자체를 학습하는 방법입니다. “정답이 없다”는 뜻이지 평가할 수 없다는 뜻은 아니며, stability·likelihood·external validity로 품질을 평가할 수 있습니다.

## 3. Cluster라는 것은 데이터가 아니라 정의에도 의존한다

어떤 점들이 cluster인지 결정하려면 “비슷함”을 정의해야 합니다. Euclidean distance를 쓸 수도 있고 correlation, Mahalanobis, domain-specific distance를 쓸 수도 있습니다. 같은 데이터라도 거리가 달라지면 cluster가 달라집니다.

따라서 clustering 결과는

$$
\text{data} + \text{representation} + \text{distance} + \text{algorithm}
$$

의 함수라고 보는 것이 안전합니다.

## 4. Agglomerative Hierarchical Clustering

처음에는 각 관측치가 하나의 cluster입니다. 가장 가까운 두 cluster를 합치고 이를 반복하여 마지막에는 하나의 cluster가 됩니다.

### 4.1 Single linkage

$$
d(A,B)=\min_{i\in A,\;j\in B}d_{ij}
$$

- $A,B$: 두 cluster입니다.
- $d_{ij}$: 관측치 $i,j$ 사이 거리입니다.

두 cluster 사이 가장 가까운 한 쌍만 봅니다. 길게 이어진 chain 형태를 잘 만들 수 있습니다.

**용어 설명 — chaining effect**  
서로 매우 가까운 점들이 사슬처럼 이어져 실제로는 멀리 떨어진 영역까지 하나의 큰 cluster로 연결되는 single-linkage 특성입니다.

### 4.2 Complete linkage

$$
d(A,B)=\max_{i\in A,\;j\in B}d_{ij}
$$

두 cluster에서 가장 먼 점 쌍까지 가까워야 합치므로 비교적 compact한 cluster를 선호합니다.

### 4.3 Group-average linkage

$$
d(A,B)
=\frac{1}{n_An_B}
\sum_{i\in A}\sum_{j\in B}d_{ij}
$$

- $n_A,n_B$: 각 cluster의 관측치 수입니다.

single과 complete의 극단을 완화해 전체 pair distance의 평균을 봅니다.

## 5. Dendrogram 해석

Hierarchical clustering은 합쳐지는 순서와 높이를 dendrogram으로 표현합니다. 하지만 dendrogram의 가지를 특정 높이에서 자르는 것은 결국 cluster 수를 결정하는 모델 선택입니다.

**중요**  
높이가 크게 점프하는 지점이 있어도 그것이 실제 population cluster 수를 증명하지는 않습니다. linkage와 scaling에 따라 모양이 달라질 수 있습니다.

## 6. K-means

$K$개의 cluster $G_1,\ldots,G_K$와 각 cluster centroid $\mu_k$를 찾는 목적은

$$
\min_{G_1,\ldots,G_K}
\sum_{k=1}^{K}\sum_{i\in G_k}
\|x_i-\mu_k\|_2^2
$$

입니다.

교재의 변수별 표현으로는 within-group sum of squares가

$$
\text{WGSS}
=\sum_{j=1}^{q}\sum_{l=1}^{K}
\sum_{i\in G_l}
(x_{ij}-\bar x_j^{(l)})^2
$$

로 표현됩니다.

- $K$: cluster 수입니다.
- $G_l$: $l$번째 cluster의 관측치 집합입니다.
- $\bar x_j^{(l)}$: $l$번째 cluster에서 변수 $j$의 평균입니다.
- WGSS: cluster 내부 산포의 총합입니다.

### Lloyd-type iteration

```text
centroid 초기화
   ↓
각 점을 가장 가까운 centroid에 할당
   ↓
각 cluster 평균으로 centroid 갱신
   ↓
변화가 작아질 때까지 반복
```

각 단계에서 objective는 줄어들지만 global optimum을 보장하지 않습니다.

## 7. K-means가 암묵적으로 선호하는 구조

Euclidean squared distance를 사용하므로 각 cluster가 비슷한 크기의 구형 또는 타원보다는 구형에 가까운 geometry를 가질 때 잘 작동합니다. 매우 elongated하거나 density가 다른 cluster, non-convex cluster에는 부적절할 수 있습니다.

또한 변수 단위가 다르면

$$
\|x_i-\mu_k\|_2^2
=\sum_j(x_{ij}-\mu_{kj})^2
$$

에서 큰 단위 변수가 objective를 지배합니다. 따라서 scaling은 k-means의 결과 자체를 바꾸는 핵심 결정입니다.

## 8. Model-based Clustering: mixture model

데이터가 $K$개의 확률분포 중 하나에서 왔다고 가정합니다.

$$
f(x;p,\theta)
=\sum_{k=1}^{K}p_k g_k(x;\theta_k)
$$

- $p_k\ge0$: cluster $k$의 mixing proportion입니다.
- $\sum_{k=1}^{K}p_k=1$입니다.
- $g_k(x;\theta_k)$: cluster $k$의 probability density입니다.
- $\theta_k$: 평균·공분산 등 cluster parameter입니다.

관측치 $x_i$가 cluster $k$에서 왔을 posterior probability는

$$
\tau_{ik}
=P(C_i=k\mid x_i)
=\frac{p_k g_k(x_i;\theta_k)}
{\sum_{h=1}^{K}p_h g_h(x_i;\theta_h)}
$$

입니다.

**용어 설명 — posterior membership probability**  
관측치가 각 cluster에 속할 확률을 model parameter와 data를 이용해 계산한 값입니다. k-means처럼 hard label 하나만 주는 대신 membership uncertainty를 제공합니다.

## 9. Gaussian Mixture Model

각 성분을 다변량 정규분포로 두면

$$
g_k(x)
=\frac{1}{(2\pi)^{q/2}|\Sigma_k|^{1/2}}
\exp\left[-\frac12(x-\mu_k)^\top\Sigma_k^{-1}(x-\mu_k)\right]
$$

입니다.

- $\mu_k$: cluster $k$ 평균벡터입니다.
- $\Sigma_k$: cluster $k$ 공분산행렬입니다.

로그우도는

$$
\ell
=\sum_{i=1}^{n}
\log\left[
\sum_{k=1}^{K}p_k g_k(x_i;\theta_k)
\right]
$$

이며 EM algorithm으로 반복 추정하는 것이 일반적입니다.

### EM의 직관

- E-step: 현재 parameter로 $\tau_{ik}$를 계산합니다.
- M-step: $\tau_{ik}$를 soft weight처럼 사용해 $p_k,\mu_k,\Sigma_k$를 갱신합니다.

**용어 설명 — EM(Expectation-Maximization)**  
cluster label처럼 관측되지 않은 latent variable이 있는 likelihood 문제를 E-step과 M-step으로 번갈아 최적화하는 알고리즘입니다.

## 10. Covariance parameterization

교재의 `mclust` 계열에서는

$$
\Sigma_k=D_kA_kD_k^\top
$$

형태로 covariance를 분해하여 cluster의 volume, shape, orientation에 제약을 둡니다.

- $D_k$: 방향(orientation)을 정하는 eigenvector matrix입니다.
- $A_k$: 주축 방향의 상대적 variance, 즉 shape를 나타냅니다.
- 전체 scale은 volume parameter와 연결됩니다.

이런 제약을 통해 spherical, diagonal, ellipsoidal cluster를 다양한 복잡도로 비교할 수 있습니다.

## 11. BIC를 이용한 모델 선택

mixture model에서는 cluster 수 $K$와 covariance structure를 BIC로 비교할 수 있습니다. 일반적인 정의는

$$
\text{BIC}
=-2\ell(\hat\theta)+d\log n
$$

입니다.

- $\ell(\hat\theta)$: 최대화된 log-likelihood입니다.
- $d$: 자유 parameter 수입니다.
- $n$: 표본 수입니다.

단, R의 특정 패키지는 부호 convention을 달리해 “큰 BIC가 좋다”고 표시할 수 있으므로 **software 정의를 확인**해야 합니다.

## 12. 저자가 직접 보고한 결과

### 12.1 Romano-British pottery

교재는 pottery chemical composition에 k-means와 시각화를 적용하여 이미 알려진 지역/가마 구조와 군집이 어느 정도 대응하는지 살펴봅니다. 3-cluster solution은 세 가지 큰 구조를 상당히 반영하지만, 더 많은 cluster로 나누면 분리가 명확하지 않은 부분이 나타난다고 설명합니다.

### 12.2 Gastroenterologist model-based clustering

약 600명의 유럽 gastroenterologist 설문을 국가 단위 profile로 분석한 예에서, 교재의 BIC는

- covariance model: **VEV** = ellipsoidal, equal shape
- cluster 수: **3**

을 선택합니다.

저자는 세 cluster를 대략 다음과 같이 해석합니다.

1. 환자와 배우자에게 비교적 적극적으로 진단·예후를 알리는 국가군.
2. 배우자에게는 정보를 주지만 환자에게는 직접 묻는 경우를 제외하면 덜 공개하는 경향의 국가군.
3. 환자에게 bad news를 거의 알리지 않는 경향의 국가군.

동시에 저자들은 binary-response proportion에 Gaussian mixture를 적용하는 것이 모델 가정상 완전히 자연스럽지는 않다는 점을 명시적으로 경고합니다.

## 13. 해석: clustering에서 “발견”과 “만들어낸 구조”를 구분하기

Clustering algorithm은 어떤 partition이든 만들어낼 수 있습니다. 예를 들어 k-means에 $K=3$을 주면 실제로 세 population이 없어도 세 group을 만듭니다. 따라서 cluster의 존재는 algorithm output이 아니라 다음의 결합으로 판단해야 합니다.

- resampling stability
- separation / compactness
- alternative distance·algorithm에서도 재현되는지
- external variables와 의미 있는 관계가 있는지
- 새로운 데이터에서도 membership 구조가 유지되는지

## 14. 통계적으로 취약한 부분

1. **$K$의 사후 선택**: 여러 $K$를 보고 가장 그럴듯한 것을 고르면 불확실성을 과소평가할 수 있습니다.
2. **Initialization sensitivity**: k-means와 mixture EM은 local optimum에 빠질 수 있습니다.
3. **Scale sensitivity**: 표준화 여부가 cluster geometry를 바꿉니다.
4. **Cluster tendency 미확인**: 군집이 존재하는지 확인하지 않고 알고리즘부터 돌릴 수 있습니다.
5. **Mixture distribution misspecification**: Gaussian mixture가 bounded proportion, count, categorical data에 부적절할 수 있습니다.
6. **High-dimensional distance concentration**: $q$가 크면 Euclidean distance 차이가 줄어들어 nearest/farthest 구분이 불안정해질 수 있습니다.
7. **post-hoc naming**: 결과를 보고 그럴듯한 이름을 붙이는 것은 causal explanation이 아닙니다.

## 15. 비교 불가능한 수치

- k-means WGSS와 Gaussian-mixture BIC는 서로 다른 objective이므로 절대값 비교가 불가능합니다.
- silhouette, ARI, NMI, BIC는 각각 compactness, label agreement, information overlap, penalized likelihood를 측정하므로 같은 “성능점수”가 아닙니다.
- deep clustering benchmark accuracy와 교재의 pottery/gastroenterologist 사례는 데이터셋·label 사용 여부·embedding 구조가 달라 직접 비교하면 안 됩니다.

## 16. 문서가 직접 답하지 않는 질문과 답변

### 질문 1. Cluster가 실제로 있는지 먼저 확인할 수 있는가?

Hopkins statistic, gap statistic, multimodality check, pairwise-distance structure 같은 보조기법을 사용할 수 있지만 어떤 방법도 “진짜 cluster 존재”를 완전히 증명하지는 않습니다. 가장 중요한 것은 여러 resample과 알고리즘에서 구조가 안정적으로 재현되는지입니다.

### 질문 2. $K$를 어떻게 정해야 하는가?

k-means에서는 elbow, silhouette, gap statistic을 함께 보고, mixture에서는 BIC/ICL과 posterior uncertainty를 볼 수 있습니다. 최종적으로는 stability와 domain usefulness를 포함해야 합니다.

### 질문 3. clustering을 train/test로 나누는 것이 필요한가?

설명적 목적이면 전체 데이터 EDA도 가능하지만 “새 데이터에 cluster rule을 적용”하거나 downstream model을 평가하려면 분할이 필요합니다. train에서 centroids/mixture parameters를 학습하고 test를 fixed rule로 assign해야 일반화가 평가됩니다.

## 17. 일반화 성능 향상 가능성

- k-means++와 다중 초기화로 local optimum 민감도 감소
- train-only scaling과 robust scaling
- PCA/shrinkage covariance로 고차원 noise 완화
- bootstrap 또는 subsampling cluster stability
- Gaussian mixture에서 covariance regularization
- posterior probability가 낮은 borderline observation을 forced hard label로 과해석하지 않기
- group/time split에서 cluster reproducibility 확인
- downstream predictor에 cluster ID를 넣을 경우 cluster extraction도 train 내부에서 수행

## 18. 2020년 이후 관련 최신 연구 비교 분석

### 18.1 Deep Clustering survey

**Ren et al., “Deep Clustering: A Comprehensive Survey”, 2022, arXiv.**

이 survey는 deep clustering을 traditional single-view, semi-supervised, multi-view, transfer clustering 등으로 나누어 정리합니다. 현대 deep clustering의 핵심은 raw input에서 바로 거리를 계산하는 대신 neural representation $z=f_\phi(x)$를 학습하고 그 공간에서 clustering-friendly geometry를 만드는 데 있습니다.

개념적으로 joint objective는

$$
L(\phi,C)
=L_{\text{representation}}(\phi)
+\lambda L_{\text{cluster}}(f_\phi(X),C)
$$

처럼 볼 수 있습니다.

- $\phi$: representation network parameter입니다.
- $C$: cluster assignment 또는 prototype입니다.
- $\lambda$: 두 목적의 균형입니다.

### 18.2 Prior 관점의 최신 survey

**Lu et al., “A survey on deep clustering: from the prior perspective”, Vicinagearth, 2024.**

이 연구는 deep clustering의 성능을 단지 network architecture가 아니라 어떤 prior knowledge를 넣는지로 재정리합니다. data structure assumption, augmentation invariance, external prior 등이 clustering의 핵심 설계요소라고 봅니다.

**용어 설명 — contrastive clustering**  
같은 sample의 augmentation은 가까워지고 다른 sample은 적절히 분리되도록 representation을 학습하면서 cluster 구조를 함께 형성하는 계열입니다.

### 고전적 방법과 현대 방법 비교

| 방법 | representation | cluster model | 장점 | 위험 |
|---|---|---|---|---|
| Hierarchical | raw/scaled feature | linkage rule | dendrogram, 단순 | distance·linkage 민감 |
| K-means | raw 또는 사전 변환 | spherical centroid | 빠름 | local optimum, non-convex 구조 약함 |
| Gaussian mixture | raw 또는 변환 | probabilistic ellipsoid | uncertainty 제공 | Gaussian misspecification |
| Deep clustering | learned nonlinear embedding | centroid/prototype/network | 복잡한 구조 가능 | 표본 요구량, collapse, 재현성 |

## 19. 실제 파이프라인 적용 시 고려할 점

```text
1) Train / Validation / Test 또는 time/group split
2) Train에서 scaling·imputation
3) cluster tendency 확인
4) 후보 representation: raw / PCA / domain transform
5) 후보 알고리즘: hierarchical / k-means / GMM
6) K 범위와 random initialization 반복
7) internal metric + bootstrap stability
8) domain label은 외부 validation으로 사용
9) 새 샘플 assignment rule 고정
10) downstream task가 있으면 nested evaluation
```

### 공정 데이터 적용 예

cluster가 chamber ID를 거의 그대로 복원한다면 “새로운 process regime”을 찾은 것이 아니라 장비 차이를 다시 발견한 것일 수 있습니다. 반대로 chamber를 제거하고도 특정 temperature/RF pattern으로 cluster가 유지되며 future test 구간에서도 같은 centroids가 재현되면 새로운 latent regime일 가능성이 더 큽니다.

## 20. 시사점과 후속 연구 방향

교재의 핵심 시사점은 cluster analysis가 강력하지만 “어떤 한 알고리즘이 최선”인 분야가 아니라는 것입니다. 후속 연구로는 (1) raw vs PCA vs learned representation의 cluster stability, (2) k-means와 Gaussian mixture의 future membership consistency, (3) chamber/time split에서 cluster survival, (4) deep clustering이 소표본 공정데이터에서 실제로 classical method보다 나은지, (5) cluster-aware partial pooling 또는 mixture-of-experts predictor로 연결하는 연구를 진행할 수 있습니다.

## 21. 빠른 이해 점검

- single/complete/average linkage는 cluster 사이의 “거리”를 어떻게 다르게 정의하는가?
- k-means objective가 왜 scaling에 민감한가?
- Gaussian mixture가 k-means보다 membership uncertainty를 더 잘 표현하는 이유는 무엇인가?
- BIC가 3 cluster를 골랐다는 것이 “자연에 정확히 3개 집단이 존재한다”는 뜻은 왜 아닌가?

## 22. 참고자료

### 교재
- Brian S. Everitt & Torsten Hothorn, *An Introduction to Applied Multivariate Analysis with R*, Springer, 2011, Chapter 6.
- Everitt et al., *Cluster Analysis*, 2011. 교재가 더 상세한 clustering 논의로 권하는 참고문헌.
- Fraley & Raftery 계열의 model-based clustering / `mclust` 연구. 교재의 Gaussian mixture 분류 기반.

### 2020년 이후 확장 연구 및 사이트
- Yazhou Ren et al., “Deep Clustering: A Comprehensive Survey”, arXiv, 2022. Source site: arXiv.
- Yiding Lu, Haobin Li, Yunfan Li, Yijie Lin & Xi Peng, “A survey on deep clustering: from the prior perspective”, *Vicinagearth*, Vol. 1, Article 4, 2024. Source site: Springer Nature.
