# Fast Online Changepoint Detection via Functional Pruning CUSUM Statistics

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장
이 논문은 **Functional Online CuSUM (FOCuS)** 알고리즘을 제안합니다. 기존의 온라인 변화점 탐지 알고리즘들(MOSUM, Page-CUSUM 등)은 **윈도우 크기** 또는 **변화 크기($\mu_1$)** 를 사전에 지정해야 하며, 이 선택이 탐지 성능에 결정적인 영향을 미쳤습니다. FOCuS는 이러한 파라미터를 **모든 가능한 값에 대해 동시에** 계산하는 것과 동등하면서도, 반복당 평균 계산 비용이 관측 수의 **로그(logarithmic)** 에 비례하도록 합니다.

### 주요 기여
| 기여 항목 | 내용 |
|---|---|
| 알고리즘 설계 | 함수형 가지치기(functional pruning)를 이용한 Page-CUSUM 통계량의 정확한 효율적 계산 |
| 이론적 보장 | 반복당 평균 계산 비용 $O(\log n)$의 tight한 상한 증명 |
| 적용 범위 확장 | 사전 변화 평균 미지(unknown pre-change mean) 및 이상치(outlier) 존재 시나리오로 확장 |
| 실용적 검증 | AWS Cloudwatch 서버 CPU 이용률 데이터에서 SOTA 성능 달성 |
| 소프트웨어 공개 | R 패키지 공개 (https://github.com/gtromano/FOCuS) |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

온라인 변화점 탐지는 세 가지 주요 요건을 충족해야 합니다:
1. **순차적(sequential)**: 데이터를 실시간으로 처리
2. **상수 메모리(constant memory)**: 무한한 반복에도 메모리 사용량 고정
3. **고속 처리**: 적어도 평균적으로 데이터 도착 속도를 따라갈 것

기존 방법의 문제점:

**Page (1955)의 방법**: 사전/사후 변화 평균을 모두 알아야 하며, $\mu_1$ 추정이 틀리면 검출력이 크게 감소

$$\mathcal{Q}_{n,\mu_1} = \max_{0 \le s \le n} \sum_{t=s+1}^{n} \mu_1\left(x_t - \frac{\mu_1}{2}\right)$$

이는 재귀적으로:

```math
\mathcal{Q}_{n,\mu_1} = \max\left\{0,\; \mathcal{Q}_{n-1,\mu_1} + \mu_1\left(x_n - \frac{\mu_1}{2}\right)\right\}
```

**MOSUM**: 윈도우 크기 $w$를 고정해야 하며, 부적절한 선택 시 성능 저하

$$M_w(n) = \frac{1}{\sqrt{w}}|S(n-w, n)| $$

**Page-CUSUM (정확한 구현)**: 반복당 $O(n)$ 비용으로 $O(n^2)$ 전체 복잡도 → 실시간 처리 불가

$$P(n) = \max_{0 \le w < n} \frac{1}{\sqrt{w}}|S(n-w, n)| $$

### 2.2 제안하는 방법

#### 2.2.1 FOCuS $^0$ : 사전 변화 평균 $\mu_0$ 알려진 경우

핵심 아이디어는 식 (5)를 **모든 $\mu$ 값에 대해 동시에** 함수 $Q_n(\mu)$의 재귀로 풀겠다는 것입니다:

```math
Q_0(\mu) = 0, \quad Q_n(\mu) = \max\left\{0,\; Q_{n-1}(\mu) + \mu\left(x_n - \frac{\mu}{2}\right)\right\}
```

그리고 테스트 통계량은 $\max_\mu Q_n(\mu)$입니다.

**Proposition 1**에 의해:

$$\max_\mu Q_n(\mu) = \frac{1}{2}P(n)^2 = \frac{1}{2}\max_w M_w(n)^2$$

즉 FOCuS $^0$ 은 **모든 가능한 윈도우 크기의 MOSUM 통계량 중 최대값**을 계산하는 것과 동등합니다.

#### 2.2.2 핵심 구조: 구간별 이차함수(Piecewise Quadratic)

재귀 (6)은 구간별 이차함수(piecewise quadratic)를 구간별 이차함수로 매핑합니다. 시간 $n$에서 반복 $\tau$에 도입된 이차함수는:

$$\mu\left(\sum_{t=\tau+1}^{n} x_t - (n-\tau)\frac{\mu}{2}\right) = \mu\left((S_n - S_\tau) - (n-\tau)\frac{\mu}{2}\right) $$

여기서 $S_t = \sum_{j=1}^t x_j$. 각 이차함수는 3-튜플 $(\tau_i, s_i, l_i)$로 저장됩니다:
- $\tau_i$: 이차함수가 도입된 시간
- $s_i = S_{\tau_i}$: 해당 시점까지 관측값의 합
- $l_i$: $i$번째 이차함수가 최적인 $\mu$ 구간의 좌측 경계

새로운 이차함수(영선, zero-line)의 좌측 경계:

$$l = 2\max_\tau \frac{S_n - S_\tau}{n - \tau} $$

변화 탐지 조건 (시간 $n$에서 3-튜플 $(\tau, s, l)$에 대해):

$$(S_n - s)^2 \ge 2\lambda(n - \tau) $$

#### 2.2.3 FOCuS: 사전 변화 평균 $\mu_0$ 미지(Unknown Pre-change Mean)

Yu et al. (2020)의 로그 우도비(Log-likelihood Ratio) 통계량:

```math
LR_n = \max_{\substack{\tau \in \{1,\ldots,n-1\} \\ \mu_0, \mu_1 \in \mathbb{R}}} \left\{-\sum_{t=1}^{\tau}(x_t - \mu_0)^2 - \sum_{t=\tau+1}^{n}(x_t - \mu_1)^2\right\} - \max_{\mu \in \mathbb{R}}\left\{-\sum_{t=1}^{n}(x_t - \mu)^2\right\}
```

이를 $\mathcal{Q}\_{\tau,n}(\mu_0, \mu_1)$로 표현하고, $\mathcal{Q}\_n(\mu_0, \mu_1) = \max_{\tau} \mathcal{Q}_{\tau,n}(\mu_0, \mu_1)$이면:

```math
LR_n = 2\left\{\max_{\mu_0, \mu_1 \in \mathbb{R}} \mathcal{Q}_n(\mu_0, \mu_1) - \max_{\mu_0 \in \mathbb{R}} \mathcal{Q}_n(\mu_0, \mu_0)\right\}
```

시간 $n$에서 3-튜플 $(\tau_i, s_i, l_i)$로 정의된 이차함수에 대한 최대화:

$$\tau_i\left(\frac{S_i}{\tau_i}\right)^2 + (n-\tau_i)\left(\frac{S_n - S_i}{n - \tau_i}\right)^2 - n\left(\frac{S_n}{n}\right)^2$$

#### 2.2.4 FOCuS with Outliers (R-FOCuS)

Biweight loss를 이용한 강건한 손실 함수:

```math
F(x_t, \mu_1) = -\min\left\{\left(\frac{\mu_1}{2} - x_t\right)^2, K\right\}
```

해당 함수형 재귀:

```math
Q_n(\mu) = \max\left\{\max_{\mu_0}\sum_{t=1}^{n}F(x_t, \mu_0),\; Q_{n-1}(\mu) + F(x_n, \mu)\right\}
```

### 2.3 모델 구조

알고리즘의 핵심 구조는 두 단계입니다:

**Step 1: 구간 및 이차함수 갱신 (Algorithm 2)**
```
Input: Q_{n-1}(μ), x_n, S_{n-1}
1. S_n ← S_{n-1} + x_n
2. 새 이차함수 q_{k+1} = (n, S_n, ∞) 추가
3. 가지치기: 영선보다 낮은 이차함수 제거
4. 새 경계 l_{k+1} = max{0, 2(s_{k+1} - s_i)/(τ_{k+1} - τ_i)} 계산
```

**Theorem 2**: Algorithm 2의 최악 복잡도는 $O(T)$이며, 반복당 amortized 복잡도는 $O(1)$

**Step 2: 최대화**
- 저장된 이차함수를 순회하며 식 (10) 확인
- 반복당 평균 비용: $O(\log n)$ (Theorem 3, 4에 의해)

### 2.4 성능 향상

**Theorem 3 (FOCuS $^0$ )**: 데이터가 $X_i = \mu_i + \epsilon_i$ (연속 분포, 독립)를 따를 때, $\mu_i$ 상수 시:

```math
E(\#\mathcal{I}^0_{1:T}) \le \log(T) + 1
```

단일 변화점 존재 시:

```math
E(\#\mathcal{I}^0_{1:T}) \le 2(\log(T/2) + 1)
```

**Theorem 4 (FOCuS)**: 동일한 조건에서, $\mu_i$ 상수 시:

```math
E(\#\mathcal{I}_{1:n}) = 1 + \sum_{1}^{n-1}\frac{1}{t+1} \le (1 + \log(n))
```

단일 변화점 시: 

```math
E(\#\mathcal{I}_{1:n}) \le 2(1 + \log(n/2))
```

**실용적 성능**:
- 100만 관측값 처리: 1초 미만 (일반 PC)
- 100만번째 관측값 처리 ≈ 이차함수 15개 평가 비용
- Yu-CUSUM ( $O(n^2)$ ) 대비 $n > 100$에서 FOCuS가 더 빠름
- AWS Cloudwatch 데이터: R-FOCuS Precision 0.58, Recall 0.82 vs HTM Precision 0.50, Recall 0.76

### 2.5 한계

1. **단변량 제한**: 함수형 가지치기는 현재 **단변량 파라미터 함수**에만 작동
2. **엄밀한 온라인 알고리즘 아님**: 반복당 이차함수 수가 이론적으로 무한대가 될 수 있어 최악의 경우 비용이 유계되지 않음 (근사 버전인 FOCuS-Pp로 해결 가능)
3. **Gaussian 모델 가정**: 핵심 이론이 Gaussian 노이즈 기반
4. **단일 변화점 이론**: 복수의 미탐지 변화점 존재 시 이론적 보장 약화
5. **R-FOCuS 복잡도 이론 부재**: biweight loss 적용 시 기대 이차함수 수에 대한 이론적 상한 미제시 (경험적으로 $O(\log n)$ 관찰)

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 파라미터 무관 탐지 (Parameter-agnostic Detection)

FOCuS의 가장 핵심적인 일반화 기여는 **사전 파라미터 지정 없이** 모든 가능한 변화 크기를 동시에 고려한다는 점입니다.

기존 방법과 비교:

| 방법 | 파라미터 지정 필요 | 일반화 성능 |
|---|---|---|
| Page (1955) | $\mu_1$ 고정 | 특정 변화 크기에 최적화 |
| MOSUM | 윈도우 크기 $w$ 고정 | 특정 변화 패턴에 최적화 |
| Page-CUSUM (grid) | 이산적 $\mu_1$ 그리드 | 그리드 외 값에 성능 저하 |
| **FOCuS** | **없음** | **모든 변화 크기에 이론적 최적** |

특히 Proposition 1이 이를 뒷받침합니다:

$$\max_\mu Q_n(\mu) = \frac{1}{2}P(n)^2 = \frac{1}{2}\max_w M_w(n)^2$$

### 3.2 사전 변화 평균 미지 설정의 일반화

FOCuS(미지 사전 평균)는 FOCuS $^0$ (알려진 사전 평균)보다 일반적인 설정에서 작동합니다. 시뮬레이션에서:
- **훈련 데이터가 적을 때**: FOCuS가 FOCuS $^0$ 보다 우수 (사전 변화 평균을 실시간으로 추정하기 때문)
- **훈련 데이터가 충분할 때**: FOCuS $^0$ 가 미세한 변화에 더 민감 (추정 불확실성이 없으므로)

이는 **데이터 분포 shift에 대한 적응적 강건성**을 보여줍니다.

### 3.3 이상치에 대한 강건성 (R-FOCuS)

Biweight loss를 활용한 R-FOCuS는 outlier가 포함된 실제 데이터에서도 동작합니다:

```math
F(x_t, \mu_1) = -\min\left\{\left(\frac{\mu_1}{2} - x_t\right)^2, K\right\}
```

파라미터 $K$를 훈련 데이터에서 자동 추정함으로써 데이터 특성에 적응합니다.

### 3.4 다변량 데이터로의 확장 가능성

논문은 FOCuS 통계량을 여러 데이터 스트림에 독립적으로 적용하고 결합하는 방법을 논의합니다:
- **최대값 결합**: 소수 스트림의 sparse 변화 탐지에 적합
- **합 결합**: 다수 스트림의 dense 변화 탐지에 적합

$k$개 스트림 중 $k$개가 변화할 때의 변화 시뮬레이션:

$$\Delta_i = \frac{mZ_i}{\sqrt{\sum_{j=1}^k Z_j^2}}, \quad Z_1, \ldots, Z_k \sim \mathcal{N}(0,1)$$

**사전 평균 미지 시 FOCuS가 ocd(Chen et al., 2022) 대비 현저히 우수**한 결과를 보입니다 (Table 2 참조).

### 3.5 다양한 손실 함수로의 일반화

논문은 Gaussian log-likelihood 외에도 **구간별 이차(piecewise quadratic) 손실 함수**면 FOCuS 프레임워크를 적용할 수 있음을 명시합니다:
- Biweight loss (R-FOCuS)
- $L_1$ loss
- 그 외 구간별 이차 손실 함수

또한 특정 변화 패턴 제약(Hocking et al., 2020; Runge et al., 2020), 자기상관 노이즈 모델(Romano et al., 2020), 칼슘 이미징 데이터의 비볼록 역합성(Jewell et al., 2020) 등 다양한 도메인 응용이 가능합니다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 앞으로의 연구에 미치는 영향

**① 효율적 온라인 탐지의 새 기준점 설정**

FOCuS는 기존 방법들이 "계산 효율성 OR 통계 최적성" 중 하나를 선택해야 했던 trade-off를 해소합니다. 이는 향후 온라인 변화점 탐지 알고리즘의 **효율성 기준점(benchmark)** 이 됩니다.

**② 오프라인 알고리즘과의 연결**

논문은 오프라인 pDPA 알고리즘(Rigaill, 2015)과의 연결을 보여주며, FOCuS에 저장된 후보 변화점 집합이 pDPA의 집합을 포함한다고 증명합니다. 이는 **FPOP (Maidstone et al., 2017)** 및 **GFPOP (Hocking et al., 2020)** 등 기존 오프라인 알고리즘들의 복잡도 분석에도 응용될 수 있습니다.

**③ 실시간 이상 탐지 프레임워크**

R-FOCuS의 AWS 실험은 FOCuS 프레임워크가 산업 모니터링, 사이버 보안, 의료 신호 처리 등의 **실시간 이상 탐지 시스템**에 직접 활용 가능함을 보여줍니다.

**④ 함수형 가지치기의 범용화 가능성**

이 논문은 FPOP류 오프라인 아이디어를 온라인 설정에 성공적으로 이전했습니다. 유사한 전략이 **다른 통계 문제**(분산 변화, 자기상관 파라미터 변화 등)에 적용될 수 있다는 방향성을 제시합니다.

### 4.2 앞으로 연구 시 고려할 점

**① 다변량 파라미터로의 확장**

현재 함수형 가지치기는 단변량 파라미터 공간에서만 작동합니다. Runge (2020)의 아이디어를 고려하더라도, 고차원 파라미터 공간에서의 효율적인 가지치기 방법 개발이 핵심 과제입니다.

**② 복수 변화점에 대한 이론적 보장**

Theorem 4는 최대 1개의 변화점 상황에 제한됩니다. **다수 변화점** 또는 지속적으로 변하는 환경에서의 이론적 복잡도 분석이 필요합니다.

**③ 비정상성(Non-stationarity) 처리**

실제 데이터는 시간에 따라 변하는 분산, 계절성, 자기상관을 포함합니다. FOCuS의 통계적 이론은 주로 **i.i.d. Gaussian** 가정에 기반하므로, 이를 완화하는 방향의 연구가 필요합니다.

**④ 임계값(Threshold) 설정의 자동화**

논문은 고정 임계값을 사용하지만, 실제 응용에서 optimal threshold 설정은 여전히 어렵습니다. 베이지안 적응형 임계값 또는 데이터 기반 자동 보정 방법이 연구될 필요가 있습니다.

**⑤ 탐지 후 추론(Post-detection Inference)**

변화점 탐지 후 변화 시점의 불확실성 정량화(신뢰구간 제공)에 대한 연구가 부족합니다. FOCuS의 이차함수 구조가 이 문제에 활용될 수 있습니다.

**⑥ 메모리 효율성**

FOCuS의 메모리 사용량은 평균 $O(\log n)$이지만 최악의 경우 무한대가 될 수 있습니다. 임베디드 시스템이나 엣지 디바이스를 위한 **강한 메모리 제약 버전** 개발이 필요합니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 논문 | 방법 | 주요 특징 | FOCuS와의 관계 |
|---|---|---|---|
| **Yu et al. (2020)** "A note on online change point detection" (arXiv:2006.03283) | GLR 기반 온라인 탐지 | 통계적 최적성 증명; 정확 구현 $O(n)$/반복 | FOCuS가 동일 통계량을 $O(\log n)$으로 구현 |
| **Chen, Wang & Samworth (2022)** "High-dimensional, multiscale online changepoint detection" *JRSS-B* | ocd (online changepoint detection) | 고차원, 다중 스케일; sequential-Page 그리드 기반 | FOCuS는 미지 사전 평균 시 ocd보다 우수 (Table 2) |
| **Tickle, Eckley & Fearnhead (2021)** *JRSS-A* | 계산 효율적 고차원 다중 변화점 | 테러 발생 데이터 적용 | 다중 스트림 결합 전략 공유 |
| **Fisch, Eckley & Fearnhead (2021)** *J. Computational and Graphical Statistics* | 부분집합 다변량 집단/점 이상 탐지 | Sparse 이상 탐지 | FOCuS 통계량 결합 전략과 상호보완적 |
| **Hocking et al. (2020)** "GFPOP" *JMLR* | 그래프 제약 변화점 탐지 | 단조 변화 등 패턴 제약 가능 | FOCuS 재귀 구조 확장 가능성 |

### 비교 분석 핵심

**계산 효율성 측면**:

$$\text{Yu-CUSUM}: O(n^2) \succ \text{ocd}: O(n) \succ \mathbf{\text{FOCuS}}: O(n\log n) \text{ (전체)}$$

**통계 최적성 측면**:
- FOCuS는 Yu et al. (2020)과 동등한 GLR 통계량을 정확하게 계산
- ocd는 이산 그리드 근사를 사용하므로 그리드 외 변화에 약점 존재

**고차원 확장성**:
- ocd와 Fisch et al. (2021)은 다변량 설정에 직접 대응
- FOCuS는 현재 각 스트림 독립 적용 후 결합하는 간접 방식

---

## 참고자료

**주요 논문 (본문 직접 인용)**:
1. Romano, G., Eckley, I. A., Fearnhead, P., & Rigaill, G. (2023). Fast Online Changepoint Detection via Functional Pruning CUSUM Statistics. *Journal of Machine Learning Research*, 24, 1–36.
2. Yu, Y., Madrid Padilla, O. H., Wang, D., & Rinaldo, A. (2020). A note on online change point detection. *arXiv preprint arXiv:2006.03283*.
3. Chen, Y., Wang, T., & Samworth, R. J. (2022). High-dimensional, multiscale online changepoint detection. *Journal of the Royal Statistical Society (Series B)*, 84, 234–266.
4. Maidstone, R., Hocking, T., Rigaill, G., & Fearnhead, P. (2017). On optimal multiple changepoint algorithms for large data. *Statistics and Computing*, 27(2), 519–533.
5. Hocking, T., Rigaill, G., Fearnhead, P., & Bourque, G. (2020). Constrained dynamic programming and supervised penalty learning algorithms for peak detection in genomic data. *JMLR*, 21, 1–40.
6. Fearnhead, P., & Rigaill, G. (2019). Changepoint detection in the presence of outliers. *Journal of the American Statistical Association*, 114(525), 169–183.
7. Rigaill, G. (2015). A pruned dynamic programming algorithm to recover the best segmentations with 1 to kmax change-points. *Journal de la Societe Francaise de Statistique*, 156(4), 180–205.
8. Kirch, C., & Weber, S. (2018). Modified sequential change point procedures based on estimating functions. *Electronic Journal of Statistics*, 12(1), 1579–1613.
9. Fisch, A. T. M., Eckley, I. A., & Fearnhead, P. (2021). Subset multivariate collective and point anomaly detection. *Journal of Computational and Graphical Statistics*, 1–12.
10. Tickle, S. O., Eckley, I. A., & Fearnhead, P. (2021). A computationally efficient, high-dimensional multiple changepoint procedure with application to global terrorism incidence. *JRSS Series A*.
11. Page, E. S. (1954). Continuous inspection schemes. *Biometrika*, 41(1/2), 100–115.
12. Page, E. S. (1955). A test for a change in a parameter occurring at an unknown point. *Biometrika*, 42(3/4), 523–527.
13. Andersen, E. S. (1955). On the fluctuations of sums of random variables ii. *Mathematica Scandinavica*, 2, 195–223.
14. Melkman, A. A. (1987). On-line construction of the convex hull of a simple polyline. *Information Processing Letters*, 25(1), 11–12.
15. Ahmad, S., Lavin, A., Purdy, S., & Agha, Z. (2017). Unsupervised real-time anomaly detection for streaming data. *Neurocomputing*, 262, 134–147.
