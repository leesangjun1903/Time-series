# Chapter 4. Multidimensional Scaling

> 교재 범위: Chapter 4, pp. 105–134.  
> 핵심 주제: proximity data, classical MDS, non-metric MDS, stress, correspondence analysis.

## 1. Executive Summary — 10문장 이내

1. Multidimensional Scaling(MDS)은 원변수 자체보다 관측치 사이의 similarity/dissimilarity 정보를 받아, 그 관계를 저차원 좌표에 배치하는 방법입니다.
2. Classical MDS는 유클리드 거리행렬에서 내적행렬 $B$를 복원한 뒤 고유값 분해를 통해 좌표를 구합니다.
3. 원자료에서 계산된 유클리드 거리를 classical MDS에 넣으면 PCA와 밀접한 동등성이 생깁니다.
4. MDS의 좌표는 translation·rotation·reflection에 대해 유일하지 않으므로 축 자체에 고정된 실체적 의미를 부여하면 안 됩니다.
5. Non-metric MDS는 실제 거리값보다는 dissimilarity의 순서가 유지되도록 좌표를 찾으며, stress가 배치의 부적합 정도를 측정합니다.
6. 교재가 소개하는 stress의 경험적 기준은 역사적 rule of thumb이지 모든 현대 데이터에 적용되는 보편적 threshold가 아닙니다.
7. Correspondence analysis는 범주형 contingency table의 행과 열 관계를 chi-square distance 기반 저차원 공간에 표현합니다.
8. MDS는 주로 구조 설명과 시각화를 위한 방법이므로 supervised prediction의 R²와 같은 “성능 향상” 개념을 직접 적용해서는 안 됩니다.
9. 2020년 이후 PaCMAP과 densMAP 같은 nonlinear embedding은 local/global geometry 또는 density 보존을 명시적으로 다루지만, classical/non-metric MDS와 최적화 목적이 다릅니다.
10. 실제 파이프라인에서는 거리 정의, scaling, out-of-sample mapping 가능성, embedding stability를 반드시 함께 설계해야 합니다.

## 2. 목적과 필요성

어떤 데이터에서는 원변수 $X$보다 관측치 사이의 “얼마나 비슷한가”가 더 자연스러운 입력입니다. 예를 들어 사람들에게 정치인 두 명이 얼마나 비슷한지 1–9점으로 평가하게 했다면, 처음부터 우리가 갖는 것은 변수벡터가 아니라 pairwise dissimilarity입니다. MDS는 이런 proximity matrix를 사람이 볼 수 있는 2차원 또는 3차원 map으로 바꾸려는 방법입니다.

**용어 설명 — proximity**  
similarity와 dissimilarity를 포괄하는 용어입니다. similarity가 클수록 가깝다는 뜻일 수도 있고, dissimilarity가 클수록 멀다는 뜻일 수도 있으므로 분석 전에 방향을 명확히 해야 합니다.

## 3. Classical MDS의 문제 설정

$n$개 대상의 pairwise distance를 $D=(d_{ij})$로 알고 있다고 합시다. 목표는 각 대상을 $m$차원 좌표

$$
x_1,\ldots,x_n\in\mathbb R^m
$$

에 배치하여 좌표 사이의 유클리드 거리가 원래 $d_{ij}$를 잘 재현하게 하는 것입니다.

좌표행렬을

$$
X=
\begin{bmatrix}
x_1^\top\\
\vdots\\
x_n^\top
\end{bmatrix}
$$

라고 하면 inner-product matrix는

$$
B=XX^\top
$$

입니다.

- $B\in\mathbb R^{n\times n}$: 관측치끼리의 내적행렬입니다.
- $b_{ij}=x_i^\top x_j$: 관측치 $i,j$의 내적입니다.

유클리드 제곱거리는

$$
d_{ij}^2
=\|x_i-x_j\|^2
=b_{ii}+b_{jj}-2b_{ij}
$$

로 쓸 수 있습니다.

즉 거리를 알고 있으면 $B$를 복원하고, $B$를 factorization하여 좌표 $X$를 얻을 수 있다는 것이 핵심입니다.

## 4. Double Centering

좌표 중심을 원점으로 둔다고 합시다.

$$
\sum_{i=1}^{n}x_i=0
$$

중심화 행렬을

$$
J=I-\frac{1}{n}\mathbf 1\mathbf 1^\top
$$

라고 하면 classical MDS의 핵심 식은

$$
B=-\frac12 JD^{(2)}J
$$

입니다.

- $I$: $n\times n$ 단위행렬입니다.
- $\mathbf 1$: 모든 원소가 1인 $n$차원 벡터입니다.
- $D^{(2)}$: 원소가 $d_{ij}^2$인 squared-distance matrix입니다.
- $J$: 행·열 평균을 제거하는 centering matrix입니다.

교재가 원소별로 전개하는 식은 같은 내용을

$$
b_{ij}
=-\frac12\left(d_{ij}^2-d_{i\cdot}^2-d_{\cdot j}^2+d_{\cdot\cdot}^2\right)
$$

형태로 표현합니다.

**직관**  
거리만으로는 전체 점구성을 오른쪽으로 평행이동해도 변하지 않습니다. 따라서 평균 위치를 원점으로 고정한 뒤 좌표를 복원하는 것입니다.

## 5. Eigen-decomposition으로 좌표 복원

$B$를

$$
B=V\Lambda V^\top
$$

로 고유분해합니다. 양의 고유값 중 가장 큰 $m$개만 사용하면

$$
X_m=V_m\Lambda_m^{1/2}
$$

가 저차원 좌표가 됩니다.

- $V_m$: 선택된 $m$개 고유벡터입니다.
- $\Lambda_m$: 해당 양의 고유값 대각행렬입니다.
- $X_m$: $n\times m$ embedding coordinate입니다.

### 차원 수 선택

교재는 양의 고유값의 누적 비율

$$
P_m=\frac{\sum_{i=1}^{m}\lambda_i}{\sum_i\lambda_i}
$$

을 참고할 수 있고, 약 $0.8$을 하나의 합리적 수준으로 언급합니다. 그러나 이는 보편적 cutoff가 아니라 데이터와 시각화 목적에 따른 경험적 지침입니다.

## 6. 왜 좌표의 절대 방향은 의미가 없는가?

어떤 orthogonal matrix $Q$에 대해

$$
X^*=XQ
$$

로 회전·반사해도

```math
X^*(X^*)^\top
=XQQ^\top X^\top
=XX^\top
```

이므로 pairwise distance는 같습니다.

**용어 설명 — orthogonal transformation**  
$Q^\top Q=I$를 만족하는 회전 또는 반사 변환입니다. 길이와 각도를 보존합니다.

따라서 MDS 축 1을 “정치 성향”, 축 2를 “경제 정책”이라고 이름 붙이려면 외부 변수나 패턴에 의한 해석 근거가 필요합니다. 축의 부호와 방향 자체는 고유하지 않습니다.

## 7. PCA와 Classical MDS의 관계

중심화된 raw data $X_c$에서 유클리드 거리를 만들고 classical MDS를 수행하면 $B=X_cX_c^\top$을 복원하게 됩니다. PCA의 SVD

$$
X_c=U\Sigma V^\top
$$

에서 sample score가 $U\Sigma$이므로 classical MDS 좌표와 본질적으로 같은 geometry를 얻습니다.

차이는 출발점입니다.

- PCA: variable matrix $X$에서 시작합니다.
- Classical MDS: pairwise distance matrix $D$에서 시작합니다.

## 8. 비유클리드 거리와 음의 고유값

$D$가 실제 어떤 유클리드 공간에서도 정확히 구현되지 않는다면 $B$에 음의 고유값이 생길 수 있습니다. 이때 positive eigenvalue 부분만 사용해 근사할 수 있지만, 원래 dissimilarity가 유클리드 geometry와 얼마나 맞지 않는지 함께 확인해야 합니다.

**용어 설명 — Euclidean embeddability**  
주어진 거리행렬을 어떤 유클리드 공간의 점들 사이 거리로 정확히 표현할 수 있는 성질입니다.

## 9. Non-metric MDS

Non-metric MDS는 $\delta_{ij}$의 절대값보다 **순위**를 중요하게 봅니다. 관측 dissimilarity $\delta_{ij}$가 더 크면 fitted distance $d_{ij}$도 대체로 더 커지도록 monotonic transformation을 허용합니다.

교재가 제시하는 stress 형태는

```math
S(\hat X)
=
\sqrt{
\frac{\sum_{i < j}(\hat d_{ij}-d_{ij})^2}
{\sum_{i < j}d_{ij}^2}
}
```

과 같은 normalized discrepancy로 이해할 수 있습니다.

- $\hat d_{ij}$: dissimilarity order에 맞춘 disparity 또는 fitted target distance입니다.
- $d_{ij}$: 현재 low-dimensional configuration의 실제 거리입니다.
- $S$: 작을수록 proximity 구조를 잘 맞춘다는 뜻입니다.

**용어 설명 — disparity**  
원래 dissimilarity의 순서를 유지하도록 monotonic transformation한 값으로, non-metric MDS가 저차원 거리와 맞추려는 대상입니다.

교재는 Kruskal 계열의 경험적 규칙으로 stress 약 20% 이상은 poor, 10%는 fair, 5% 이하는 good에 가까운 해석을 소개하지만, 오늘날에는 데이터 크기·noise·거리구조가 다르므로 이를 절대 기준으로 쓰면 안 됩니다.

## 10. Correspondence Analysis

범주형 contingency table에서 행과 열의 profile 차이를 low-dimensional map으로 표현합니다. 단순 Euclidean distance가 아니라 expected count를 고려한 chi-square geometry를 사용합니다.

행 $i$의 profile을 $p_{ij}/p_{i+}$라고 할 때 두 행 $i,i'$ 사이의 chi-square distance는 개념적으로

```math
d^2(i,i')
=\sum_{j=1}^{c}
\frac{1}{p_{+j}}
\left(
\frac{p_{ij}}{p_{i+}}
-
\frac{p_{i'j}}{p_{i'+}}
\right)^2
```

로 표현할 수 있습니다.

- $p_{ij}$: 전체 빈도 중 cell $(i,j)$의 비율입니다.
- $p_{i+}$: 행 $i$의 주변 비율입니다.
- $p_{+j}$: 열 $j$의 주변 비율입니다.
- $c$: 열 범주 수입니다.

열의 빈도가 드문 범주일수록 $1/p_{+j}$로 차이를 더 크게 가중하게 됩니다.

## 11. 저자가 직접 보고한 결과

### 11.1 US House voting non-metric MDS

교재는 미국 하원의원 voting dissimilarity를 2차원에 배치했을 때 전반적으로 party line이 뚜렷하고, Republican 쪽 내부 variation이 더 커 보인다고 해석합니다. 또한 특정 의원(Rinaldo)이 상대적으로 Democrats에 가까운 위치에 나타나는 사례를 언급합니다.

### 11.2 World War II leaders

정치·전쟁 지도자에 대한 주관적 dissimilarity를 non-metric MDS로 표현하여, 관측된 판단 순서를 저차원 구조로 해석하는 예를 제시합니다. 이 사례는 raw numeric feature가 없어도 proximity만으로 map을 만들 수 있음을 보여줍니다.

### 11.3 Correspondence analysis

교재의 teenage relationships contingency table은 행과 열 범주가 갖는 association을 2차원 map으로 표현합니다. $r=3$, $c=5$이면 최대 비자명 차원이 $\min(r-1,c-1)=2$이므로 2차원에서 chi-square geometry를 완전히 표현할 수 있습니다.

## 12. 해석: 교재 사례를 어떻게 읽어야 하는가?

MDS map은 “실제 세계가 2차원이다”라는 뜻이 아닙니다. 원래 proximity에 들어 있는 중요한 구조를 2차원에 **근사적으로 표현**한 것입니다. 특히 axes는 회전 가능한 좌표계이므로, 하나의 축 이름보다 **점들 사이 상대 배치**가 먼저입니다.

Correspondence analysis도 가까운 범주끼리 인과적 관계가 있다고 말하는 방법이 아니라, contingency table에서 기대빈도 대비 어떤 association pattern이 있는지 보여주는 방법입니다.

## 13. 통계적으로 취약한 부분과 비교 불가능한 수치

1. **Stress cutoff는 보편적 품질점수가 아닙니다.** $n$, noise, proximity scale에 따라 기준이 달라질 수 있습니다.
2. **2-D map의 시각적 거리 과해석**은 위험합니다. 원래 고차원 geometry가 상당히 손실될 수 있습니다.
3. **축 해석의 비식별성**: rotation/reflection이 자유롭기 때문에 축 이름은 외부 근거가 필요합니다.
4. **주관적 dissimilarity 데이터**는 평가자간 차이와 measurement error를 포함할 수 있습니다.
5. **PaCMAP/densMAP의 trustworthiness류 지표와 MDS stress는 목적이 달라 직접 숫자 비교하면 안 됩니다.**
6. embedding을 여러 seed로 돌렸을 때 구조가 크게 달라진다면 한 번의 그림만으로 결론 내리면 안 됩니다.

## 14. 문서가 직접 답하지 않는 질문과 답변

### 질문 1. 새 관측치가 들어오면 기존 MDS map에 어떻게 넣는가?

Classical MDS는 기본적으로 training pairwise matrix 전체를 이용하는 transductive 성격이 강합니다. 새 점에는 기존 landmark와의 거리를 이용한 out-of-sample extension, Nyström-type 방법 또는 parametric mapping을 추가해야 합니다. 따라서 배포 시스템에서는 처음부터 OOS mapping이 필요한지 설계해야 합니다.

**용어 설명 — transductive**  
주어진 데이터셋 자체의 구조를 학습하는 데 초점을 두고, 새로운 샘플에 대한 직접적인 함수 $f(x)$가 자동으로 제공되지 않는 방식입니다.

### 질문 2. MDS와 UMAP/PaCMAP 중 무엇이 더 좋은가?

“더 좋다”는 하나의 답이 없습니다. 원거리까지 포함한 metric fidelity가 중요하고 distance matrix가 자연스러우면 MDS가 해석하기 좋습니다. 복잡한 manifold의 local neighborhood를 시각화하고 싶다면 nonlinear embedding이 유리할 수 있습니다. 비교는 목적별 보존지표와 stability로 해야 합니다.

### 질문 3. 거리 정의가 결과보다 더 중요할 수 있는가?

그렇습니다. Euclidean, correlation distance, Mahalanobis, cosine distance는 서로 전혀 다른 “비슷함”을 정의합니다. 잘못된 distance를 쓰면 최적화가 아무리 정확해도 잘못된 geometry를 충실히 재현합니다.

## 15. 일반화 성능 향상 가능성

MDS 자체는 주로 시각화/구조발견 방법이므로 predictive generalization을 직접 최적화하지 않습니다. 그러나 downstream pipeline에서는 다음이 중요합니다.

- scaling과 distance metric을 train-only 기준으로 확정
- landmark MDS로 OOS mapping 지원
- 여러 bootstrap sample에서 pairwise neighborhood 재현성 확인
- embedding dimension을 stress 하나가 아니라 downstream validation과 함께 결정
- supervised task라면 embedding이 target leakage를 만들지 않도록 target-independent fit 유지
- time-series에서는 과거 train point를 기준 landmark로 두고 미래점만 project

## 16. 2020년 이후 관련 최신 연구 비교 분석

### 16.1 PaCMAP

**Wang, Huang, Rudin & Shaposhnik, “Understanding How Dimension Reduction Tools Work: An Empirical Approach to Deciphering t-SNE, UMAP, TriMap, and PaCMAP for Data Visualization”, JMLR, 2021.**

PaCMAP은 local neighborhood만이 아니라 global structure까지 보존하기 위한 pair sampling과 loss 설계를 분석하고 제안합니다. Classical MDS가 pairwise distance의 metric reconstruction을 중시한다면, PaCMAP은 visualization-friendly neighborhood geometry를 더 적극적으로 최적화합니다.

### 16.2 densMAP

**Narayan, Berger & Cho, “Assessing single-cell transcriptomic variability through density-preserving data visualization”, Nature Biotechnology, 2021.**

densMAP은 UMAP류 embedding에서 원공간 local density가 왜곡될 수 있는 문제를 완화합니다. MDS와 직접 같은 알고리즘은 아니지만 “저차원 그림이 원데이터에서 무엇을 보존해야 하는가?”라는 동일한 핵심 질문을 더 현대적인 manifold setting에서 다룹니다.

### 비교표

| 방법 | 핵심 보존 대상 | 입력 | 주요 한계 |
|---|---|---|---|
| Classical MDS | metric distance | distance matrix | 비유클리드 거리, OOS mapping |
| Non-metric MDS | dissimilarity rank | ordinal proximity | local minimum, stress 해석 |
| Correspondence analysis | chi-square profile geometry | contingency table | sparse cell에 민감 |
| PaCMAP | local + global neighborhoods | high-dimensional vectors | hyperparameter와 embedding 해석 |
| densMAP | neighborhood + density | high-dimensional vectors | density preservation과 다른 구조 사이 trade-off |

## 17. 실제 파이프라인 적용 시 고려할 점

```text
1) 분석 목적 정의: metric fidelity / ranking / categorical association / visualization
2) Train에서 scaling과 distance metric 결정
3) pairwise distance 계산
4) Classical 또는 non-metric MDS 후보 비교
5) eigenvalue / stress / neighborhood preservation 점검
6) 여러 seed 또는 bootstrap으로 embedding stability 확인
7) 외부 label은 해석용으로 나중에 overlay
8) 새 데이터가 필요하면 OOS extension 설계
9) downstream predictor 사용 시 validation에서 실제 metric 확인
```

### 공정 데이터 적용 예

wafer/run 간 sensor profile distance를 정의하고 MDS map에서 chamber/recipe/time color를 입히면 domain shift나 regime cluster를 탐색할 수 있습니다. 그러나 chamber label을 이용해 embedding 자체를 조정한 뒤 같은 label separation을 성능처럼 제시하면 circular analysis가 됩니다.

## 18. 시사점과 추가 후속 연구

MDS의 가장 큰 시사점은 “관측치를 설명하는 원변수보다 관측치 사이 관계가 더 자연스러운 문제”가 존재한다는 것입니다. 후속 연구로는 (1) Euclidean·Mahalanobis·correlation distance가 공정 regime separation에 주는 영향, (2) classical MDS와 PaCMAP의 temporal neighborhood stability, (3) landmark MDS의 OOS reconstruction error, (4) distance learning과 MDS의 결합, (5) uncertainty-aware embedding 연구를 진행할 수 있습니다.

## 19. 빠른 이해 점검

- $B=-\frac12JD^{(2)}J$가 왜 “거리에서 좌표로 되돌아가는” 핵심인지 설명할 수 있는가?
- MDS map의 축 부호와 방향이 왜 고유하지 않은가?
- classical MDS와 non-metric MDS는 무엇을 각각 보존하려 하는가?
- stress가 작다는 사실과 prediction R²가 높은 것은 왜 비교할 수 없는가?

## 20. 참고자료

### 교재
- Brian S. Everitt & Torsten Hothorn, *An Introduction to Applied Multivariate Analysis with R*, Springer, 2011, Chapter 4.
- Young & Householder (1938), classical scaling의 초기 수학적 기반으로 교재가 인용하는 연구.
- Kruskal의 non-metric MDS/stress 계열 연구, 교재의 stress 해석 배경.

### 2020년 이후 확장 연구 및 사이트
- Yingfan Wang, Haiyang Huang, Cynthia Rudin & Yaron Shaposhnik, “Understanding How Dimension Reduction Tools Work: An Empirical Approach to Deciphering t-SNE, UMAP, TriMap, and PaCMAP for Data Visualization”, *Journal of Machine Learning Research*, 2021. Source site: JMLR.
- Ashwin Narayan, Bonnie Berger & Hyunghoon Cho, “Assessing single-cell transcriptomic variability through density-preserving data visualization”, *Nature Biotechnology*, 2021. Source site: Nature.
