# Unsupervised and Scalable Subsequence Anomaly Detection in Large Data Series

## 📌 참고 자료

- **주요 논문**: Boniol, P., Linardi, M., Roncallo, F., Palpanas, T., Meftah, M., & Remy, E. (2021). *Unsupervised and Scalable Subsequence Anomaly Detection in Large Data Series*. VLDB Journal. (첨부 PDF 전문)
- **관련 논문**: Boniol & Palpanas (2020). *Series2Graph: Graph-based Subsequence Anomaly Detection for Time Series*. PVLDB 13(11).
- **관련 논문**: Yeh et al. (2016). *Matrix Profile I: All Pairs Similarity Joins for Time Series*. ICDM.
- **관련 논문**: Senin et al. (2015). *Time Series Anomaly Discovery with Grammar-based Compression*. EDBT.
- **관련 논문**: Malhotra et al. (2015). *Long Short Term Memory Networks for Anomaly Detection in Time Series*. ESANN.

> ⚠️ **정확도 고지**: 2020년 이후 최신 연구 비교 분석 부분은 제가 직접 접근할 수 없는 외부 논문들을 포함하므로, 첨부된 논문 내 언급된 내용과 일반적으로 알려진 사실에 한해 기술하며, 불확실한 내용은 명시합니다.

---

## 1. 핵심 주장 및 주요 기여 요약

### 🔑 핵심 주장

NormA는 **비지도(Unsupervised)** 방식으로, 도메인 사전 지식 없이 **정상 행동 모델(Normal Model)**을 자동 구축하여 이상 부분 시퀀스를 탐지한다. 기존 discord 기반 방법의 핵심 한계인 **반복 이상치(recurrent anomalies) 탐지 실패** 문제를 해결하며, 정확도와 속도 모두에서 기존 방법을 압도한다.

### 🏆 주요 기여

| 기여 항목 | 내용 |
|-----------|------|
| 새로운 이상치 정의 | 최근접 이웃(NN) 거리가 아닌, 정상 모델까지의 거리 기반 이상 점수 |
| Normal Model 개념 형식화 | 반복 정상 패턴을 표현하는 가중 부분 시퀀스 집합 |
| NormA-SJ / NormA-smpl | 전체 계산 및 샘플링 기반 두 가지 알고리즘 변형 |
| NormA-mn | 다중 정상 행동 패턴을 가진 시계열 처리 확장 |
| 대규모 실험 검증 | 문헌상 가장 큰 규모의 실제 데이터셋(최대 2천만 포인트) 실험 |

---

## 2. 해결 문제 · 제안 방법 · 모델 구조 · 성능 · 한계

### 2.1 해결하고자 하는 문제

기존 **discord 기반** 방법의 세 가지 근본 한계:

1. **단일 이상치 가정**: 가장 먼 최근접 이웃(NN)을 가진 부분 시퀀스를 이상치로 정의 → 반복 이상치 탐지 실패
2. **$m^{th}$-discord의 파라미터 의존성**: 이상치 개수 $m$을 사전에 알아야 하며, 잘못된 $m$ 설정 시 대량의 False Positive 발생
3. **비정상성(Non-stationarity) 처리 불가**: 데이터의 통계적 특성이 시간에 따라 변화할 때 이상치/정상의 경계가 모호해짐

**기존 정의의 한계를 수식으로 표현:**

$$\text{Discord: } T_{i,\ell} = \arg\max_{T_{i,\ell} \in \mathbb{T}_\ell} \min_{j: |i-j| \geq \ell/2} \text{dist}(T_{i,\ell}, T_{j,\ell})$$

이 정의는 반복 이상치들이 서로를 최근접 이웃으로 삼아 거리가 작아지므로, 이상치임에도 불구하고 낮은 점수를 받는 문제가 발생한다.

---

### 2.2 제안하는 방법 (수식 포함)

#### 📐 Z-정규화 유클리드 거리

$$\text{dist}(A, B) = \sqrt{\sum_{k=1}^{\ell} \left(\frac{A_{k,1} - \mu_A}{\sigma_A} - \frac{B_{k,1} - \mu_B}{\sigma_B}\right)^2}$$

여기서 $\mu$와 $\sigma$는 각 시퀀스의 평균과 표준편차이다.

#### 📐 이상 점수 (Anomaly Score) 정의 (Definition 5)

```math
d_j = \sum_{(N^i_M, w^i) \in N_M} w^i \cdot \min_{x \in [0, \ell_{N_M} - \ell]} \left\{ \text{dist}(T_{j,\ell}, N^i_{M_{x,\ell}}) \right\}
```

- $N^i_M$: Normal Model의 $i$번째 부분 시퀀스 (클러스터 중심)
- $w^i$: $N^i_M$의 정규성 점수(normality score)
- $\ell_{N_M}$: Normal Model 길이 ($\ell_{N_M} > \ell$, 기본값: $3\ell \sim 4\ell$)

#### 📐 Data Series Join (Definition 6)

$$|A \bowtie_\ell B| = |B| - \ell + 1$$
$$(A \bowtie_\ell B)_{i,1} = \min\left(\text{dist}(B_{i,\ell}, A_{1,\ell}), \ldots, \text{dist}(B_{i,\ell}, A_{|A|-\ell+1,\ell})\right)$$

Join 연산을 통해 각 부분 시퀀스와 Normal Model 간의 최소 거리를 효율적으로 계산한다.

#### 📐 가중 이상 점수의 행렬 표현 (Algorithm 2)

$$d_j = \sum_{(N^i_M, w^i) \in N_M} w^i \cdot (N^i_M \bowtie_\ell T)_j $$

#### 📐 엔트로피 및 Description Length (MDL 원칙 적용)

$$H(T) = -\sum_{i=1}^{|T|} P(T = T_{i,1}) \log_2 P(T = T_{i,1}) $$

$$DL(T) = |T| \cdot H(T)$$

$$DL(T | \text{Center}(c)) = DL(T - \text{Center}(c)) $$

$$DLC(c | \text{Center}(c)) = DL(\text{Center}(c)) + \sum_{d \in c} DL(d | \text{Center}(c)) $$

$$\text{bitsave}(A) = \sum_{c \in A} \left[ DLC(c) - DLC(c | \text{Center}(c)) \right] $$

MDL 기반 덴드로그램 절단으로 **자동 클러스터 수 결정**이 가능하다.

#### 📐 클러스터 중심성 (Centrality)

$$\text{centrality}(c, \mathbb{C}) = \frac{1}{\sum_{x \in \mathbb{C}} \text{dist}(\text{Center}(c), \text{Center}(x))} $$

#### 📐 정규성 점수 (Norm Score)

$$\text{Norm}(c, \mathbb{C}) = \text{Frequency}(c)^2 \times \text{Coverage}(c) \times \text{centrality}(c, \mathbb{C}) $$

- **Frequency**: 클러스터 내 부분 시퀀스 수 (자주 등장 → 정상)
- **Coverage**: 클러스터가 전체 시계열에서 차지하는 범위
- **Centrality**: 클러스터 중심이 다른 클러스터들로부터 얼마나 중앙에 위치하는지

> ⚡ **설계 철학**: Frequency가 제곱으로 반영된 이유 → 높은 Coverage이지만 낮은 Frequency인 클러스터가 과대평가되는 것을 방지

#### 📐 다중 정상 행동 처리 (NormA-mn)

$$\tilde{d}_j = \left(\sum_{N^i_M} w^i (N^i_M \bowtie_\ell T)_j\right) - \beta_j $$

$$\beta_j = \frac{\sum_{k \in [I^b_{j,\tau}(T),\, I^e_{j,\tau}(T)]} d_k}{2\tau} $$

$\beta_j$는 시점 $j$ 주변 시간 구간 $\tau$ 내의 평균 거리로, 서로 다른 정상 구간 간의 교차 탐지 오류를 억제한다.

---

### 2.3 모델 구조

```
NormA 전체 파이프라인

입력: 데이터 시리즈 T, 이상 길이 ℓ
         │
         ▼
┌─────────────────────────────────────────┐
│       Step 1: Normal Model 구축          │
│  (Algorithm 4: CompNM)                  │
│                                         │
│  1-1. 후보 부분 시퀀스 선택              │
│       ├─ NormA-SJ: Self-Join 기반       │
│       │   (S^selfjoin: 가까운 이웃 있는  │
│       │    시퀀스만 선택, STOMP 활용)    │
│       └─ NormA-smpl: 균일 랜덤 샘플링  │
│           (S^sample, 선형 시간)          │
│                                         │
│  1-2. 계층적 군집화 (Complete Linkage)  │
│       + MDL 기반 자동 클러스터 수 결정  │
│       (bitsave 최대화 지점에서 절단)    │
│                                         │
│  1-3. 클러스터 Norm 점수 계산           │
│       Norm = Freq² × Coverage × Central │
│                                         │
│  출력: NM = {(N⁰_M,w⁰),...,(Nⁿ_M,wⁿ)} │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│       Step 2: 이상치 탐지               │
│  (Algorithm 2: CompAnom)               │
│                                         │
│  2-1. 각 N^i_M에 대해 Join 계산        │
│       N^i_M ⋈_ℓ T                      │
│                                         │
│  2-2. 가중 합산으로 이상 점수 계산      │
│       d_j = Σ wⁱ · (N^i_M ⋈_ℓ T)_j   │
│                                         │
│  2-3. Top-k 이상 점수 부분 시퀀스 반환 │
└─────────────────────────────────────────┘
         │
         ▼
출력: 이상 부분 시퀀스 순위 목록
```

---

### 2.4 성능 향상

#### 정확도

| 알고리즘 | 평균 P@k (전체 데이터셋) |
|----------|------------------------|
| GrammarViz (GV) | 0.62 |
| STOMP | 0.73 |
| DAD | 0.24 |
| LSTM-AD (지도학습) | 0.78 |
| LOF | 0.68 |
| Isolation Forest | 0.85 |
| **NormA-smpl** | **0.97** |
| **NormA-SJ** | **0.98** |

- Wilcoxon signed-rank test ($\alpha = 0.05$) 기반 Critical Difference Diagram에서 NormA-SJ, NormA-smpl이 통계적으로 유의미하게 우수함을 증명

#### 확장성

- NormA-smpl: 경쟁 알고리즘 대비 **1~2 오더(order)** 더 빠름
- DAD, LOF: 데이터 크기 ≥ 1M 포인트에서 8시간 초과(timeout)
- NormA-smpl: 2천만 포인트(Nasa Bearing) 데이터에도 적용 가능

**복잡도 비교:**

$$\text{NormA 이상치 탐지}: O\left((|T| - \ell + 1) \cdot \ell_{N_M} \cdot |N_M|\right), \quad |N_M| \ll |T|$$

$$\text{STOMP (기준)}: O(|T|^2)$$

---

### 2.5 한계

| 한계 | 설명 |
|------|------|
| **길이 파라미터 의존성** | 이상 부분 시퀀스 길이 $\ell$을 사용자가 입력해야 하며, 잘못 설정 시 성능 저하 가능. 가변 길이 이상치에는 직접 적용 불가 |
| **계층적 군집화 비용** | $S^{selfjoin}$ 사용 시 self-join 계산이 $O(T^2)$ 이며, 대규모 데이터에서는 NormA-smpl로 대체 필요 |
| **다중 정상 행동의 불균형** | 세그먼트 크기가 매우 불균등한 경우(예: 한 정상 패턴이 5%, 다른 패턴이 95%) NormA-mn 성능이 불안정함 (Table 5 참조) |
| **변화점(Change Point) 미지정** | NormA-mn의 $\beta_j$ 계산에서 $\tau$ 파라미터가 필요하며, 변화점이 복잡할수록 처리 어려움 |
| **클러스터링 과정의 메모리** | 완전 연결(complete linkage) 군집화가 $O(\ell S^2)$ 공간을 요구 |
| **SAX 기반 클러스터링** | SAX 이산화 과정에서 세밀한 패턴 구분이 손실될 수 있음 |

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재 논문에서 달성된 일반화 요소

#### (1) 도메인 무관성 (Domain-Agnosticism)

NormA는 **단 하나의 파라미터** $\ell$ (이상 길이)만 요구한다. 이는 ECG, 산업 센서, 교통 데이터 등 다양한 도메인에서 동일 알고리즘이 적용 가능함을 의미한다.

$$\text{입력}: (T, \ell) \xrightarrow{\text{NormA}} \text{이상 부분 시퀀스 순위 목록}$$

실험 결과, 심전도(MBA), 항공우주 엔지니어링(SED), 도시 교통(NTC), 기계 베어링(Nasa Bearing) 등 **이질적 도메인 전반에서** 일관되게 높은 성능을 기록했다.

#### (2) 다중 정상 행동 처리 (Multiple Normal Behaviors)

NormA-mn은 **여러 정상 패턴이 혼재**하는 시계열에서 일반화를 확장한다:

$$\tilde{d}_j = \left(\sum_{N^i_M} w^i (N^i_M \bowtie_\ell T)_j\right) - \beta_j$$

$\beta_j$가 로컬 평균을 차감함으로써, 국소적 정상 기준선이 달라지는 환경에서도 이상 점수가 올바르게 보정된다. 이중 정상(Double Normality)에서 평균 P@k 0.79, 삼중(Triple) 0.70, 사중(Quadruple) 0.69로 경쟁 방법 대비 일관되게 우위를 유지한다.

#### (3) 거리 척도 교체 가능성

논문은 Z-정규화 유클리드 거리를 기본으로 사용하지만, **DTW(Dynamic Time Warping)**, **SBD(Shape-Based Distance)** 등으로 교체해도 성능이 유사함을 실험으로 확인했다 (Figure 7). 이는 NormA 프레임워크 자체가 특정 거리 척도에 의존하지 않음을 의미한다.

#### (4) 샘플링 기반 일반화 (NormA-smpl)

NormA-smpl의 균일 랜덤 샘플링은 데이터의 통계적 분포를 간접적으로 반영하여, 다양한 데이터 특성에서 일반화된 Normal Model을 구축할 수 있게 한다. 샘플링 비율 $r$을 0.1~0.6 범위에서 변화시켜도 성능이 안정적으로 유지된다.

---

### 3.2 일반화 성능 향상을 위한 개선 가능성

#### (a) 가변 길이 이상치 탐지 (Variable-Length Anomaly)

현재 NormA는 고정 길이 $\ell$을 가정한다. **VALMOD** (Linardi et al., 2018, SIGMOD)처럼 가변 길이 모티프 탐색 기법을 Normal Model 구축에 결합하면, 다양한 길이의 이상치를 동시에 처리할 수 있다:

$$\ell^* = \arg\max_\ell \text{Anomaly Score}(T_{j,\ell}, N_M^{(\ell)})$$

#### (b) 온라인/스트리밍 환경 적응

현재 NormA는 전체 시계열이 주어지는 배치(Batch) 환경을 전제한다. 슬라이딩 윈도우 방식으로 Normal Model을 점진적으로 업데이트하면 **개념 표류(concept drift)**에 적응하는 일반화가 가능하다:

$$N_M^{(t+1)} = \alpha \cdot N_M^{(t)} + (1-\alpha) \cdot N_M^{\text{new}}$$

여기서 $\alpha \in (0,1)$은 망각 인자(forgetting factor)이다.

#### (c) 딥러닝 기반 표현 학습과의 결합

NormA의 Normal Model을 원시 부분 시퀀스 집합 대신, **오토인코더(Autoencoder)**나 **Contrastive Learning**으로 학습된 잠재 표현(latent representation)으로 대체하면, 더 복잡한 패턴의 정상 행동을 모델링할 수 있다:

```math
N_M^{\text{deep}} = \left\{ \left(f_\theta(N^i_M),\ w^i\right) \right\}
```

$$d_j = \sum_i w^i \cdot \text{dist}\left(f_\theta(T_{j,\ell}),\ f_\theta(N^i_M)\right)$$

#### (d) 자동 길이 추정

$\ell$을 자동으로 추정하는 방법(예: Matrix Profile의 주기성 분석, FFT 기반 지배 주파수 탐지)을 전처리 단계로 추가하면, 완전 무파라미터 시스템 구현이 가능하다.

#### (e) 다변수 시계열로의 확장

현재는 단변수(univariate) 시계열만 처리한다. 변수 간 상관 구조를 고려한 다변수 Normal Model 정의가 필요하다:

$$\text{dist}_{\text{multi}}(\mathbf{T}_{j,\ell}, \mathbf{N}^i_M) = \sum_{d=1}^{D} \lambda_d \cdot \text{dist}(T^{(d)}_{j,\ell}, N^{i,(d)}_M)$$

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미치는 영향

#### (1) 정상 행동 기반 패러다임의 확립

NormA는 이상치 탐지의 관점을 **"무엇이 가장 이상한가?"에서 "무엇이 정상과 얼마나 다른가?"** 로 전환했다. 이 패러다임 전환은 이후 연구들의 방향성에 근본적 영향을 미쳤다:

- **Series2Graph** (Boniol & Palpanas, PVLDB 2020): NormA의 저자들이 그래프 구조로 정상 행동을 표현하는 후속 연구
- 정상 행동 모델 자동 구축이라는 개념은 이후 딥러닝 기반 이상치 탐지의 설계 원칙으로도 채택됨

#### (2) 벤치마크 표준화

논문이 제공하는 **가장 포괄적인 데이터셋 실험** (MIT-BIH, NASA Bearing, NTC 등)은 이후 연구의 실험 설계 기준점을 제시했으며, 소스코드와 데이터셋 공개로 재현 가능성을 높였다.

#### (3) MDL 기반 자동 파라미터 설정

MDL을 클러스터 수 자동 결정에 활용한 방식은 시계열 분석의 다른 문제(세그멘테이션, 모티프 탐색)에서도 파라미터 자동화 연구에 영향을 주었다.

---

### 4.2 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ 아래 내용은 제가 직접 원문을 확인하지 못한 외부 논문들을 포함합니다. 논문의 존재 및 일반적 내용은 사실이나, 세부 수치나 방법론의 정확한 표현은 불확실할 수 있습니다. 따라서 명확히 확인 가능한 내용 위주로 기술하고, 불확실한 내용은 표시합니다.

#### (1) Series2Graph (Boniol & Palpanas, PVLDB 2020) ✅ 논문 내 직접 언급

NormA의 저자들이 제안한 후속 연구로, 시계열의 정상 행동을 **방향 그래프(directed graph)**로 표현한다. NormA 논문 자체에서 미래 비교 대상으로 언급되어 있다 (Section 7).

| 항목 | NormA | Series2Graph |
|------|-------|-------------|
| 정상 모델 표현 | 가중 부분 시퀀스 집합 | 방향 그래프 |
| 계산 방식 | Join 거리 계산 | 그래프 경로 분석 |
| 장점 | 해석 가능성 높음 | 복잡한 패턴 관계 표현 |

#### (2) TSB-UAD 벤치마크 (Paparrizos et al., PVLDB 2022) ⚠️ 내용 일부 불확실

시계열 이상치 탐지 알고리즘들을 통합 평가하는 대규모 벤치마크로, NormA를 포함한 다수 알고리즘을 비교한 것으로 알려져 있다. 다만, 구체적 비교 결과 수치는 직접 확인하지 못했으므로 제시하지 않는다.

#### (3) MERLIN (Nakamura et al., ICDM 2020) ⚠️ 내용 일부 불확실

**가변 길이 이상치** 탐지를 위한 알고리즘으로, NormA의 고정 길이 $\ell$ 가정을 극복하고자 하는 방향의 연구이다. NormA 대비 가변 길이 처리에서 강점이 있을 수 있으나, 정확한 성능 비교는 확인하지 못했다.

#### (4) 딥러닝 기반 방법들과의 비교 (일반적 동향)

2020년 이후 **Transformer 기반** 이상치 탐지(예: Anomaly Transformer, Wu et al., ICLR 2022)나 **확산 모델(Diffusion Model)** 기반 방법들이 등장했다. 이들은 복잡한 패턴 표현력에서 NormA보다 강점이 있을 수 있으나, 학습 데이터 요구량, 계산 비용, 해석 가능성 측면에서 NormA의 비지도 경량 접근법이 여전히 경쟁력을 가진다.

---

### 4.3 앞으로 연구 시 고려할 점

#### 🔬 방법론적 고려사항

| 고려 사항 | 구체적 내용 |
|-----------|-----------|
| **가변 길이 이상치** | 단일 $\ell$ 가정 극복을 위한 멀티스케일 Normal Model 설계 |
| **온라인 처리** | Normal Model의 점진적 업데이트 메커니즘 개발 |
| **다변수 확장** | 변수 간 상관 구조를 반영한 거리 함수 정의 |
| **자동 $\ell$ 추정** | FFT, Matrix Profile 기반 주기 탐지로 파라미터 제거 |
| **딥러닝 결합** | 오토인코더 잠재 공간에서 Normal Model 구축 |

#### 📊 실험 설계 고려사항

| 고려 사항 | 구체적 내용 |
|-----------|-----------|
| **평가 지표** | P@k 외에 ROC-AUC, F1-score, 탐지 지연(detection lag) 등 다양한 지표 병행 |
| **데이터셋 편향** | 논문에서 사용한 데이터셋 대부분이 심전도 중심 → 더 다양한 도메인 검증 필요 |
| **이상치 희소성 가정** | 이상치가 소수라는 가정이 성립하지 않는 환경(예: 이상치 비율 > 30%) 고려 |
| **노이즈 강건성** | 더 높은 수준의 Gaussian 노이즈(>25%)나 이상치 레이블 불확실성 처리 |

#### ⚖️ 실용적 고려사항

| 고려 사항 | 구체적 내용 |
|-----------|-----------|
| **설명 가능성** | Normal Model의 클러스터 중심 시각화를 통한 이상 원인 설명 체계 강화 |
| **실시간 적용** | 엣지 컴퓨팅 환경(메모리 제한)에서의 경량화 |
| **인간-AI 협력** | 전문가가 초기 Normal Model을 검토·수정하는 반지도학습 프레임워크 |
| **벤치마크 공정성** | 비교 알고리즘 간 파라미터 튜닝 기회의 공정한 제공 |

---

## 📋 최종 요약

```
NormA의 핵심 가치

  기존 방법의 한계          NormA의 해결책
  ─────────────────    ─────────────────────────
  단일 이상치 가정    →   Normal Model 기반 점수
  m 파라미터 필요    →   MDL 자동 클러스터 결정
  도메인 지식 필요   →   완전 비지도, 도메인 무관
  느린 계산 속도    →   샘플링 기반 선형 시간
  정상성 가정       →   NormA-mn 다중 정상 처리

  성과: P@k 평균 0.97~0.98 (기존 최고 0.85 대비)
        속도: 경쟁 방법 대비 1~2 오더 빠름
```
