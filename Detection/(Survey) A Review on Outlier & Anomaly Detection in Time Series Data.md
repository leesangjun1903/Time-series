# A Review on Outlier/Anomaly Detection in Time Series Data

---

## 1. 핵심 주장과 주요 기여

### 1.1 핵심 주장

본 논문(Blázquez-García et al., 2020)의 핵심 주장은 다음과 같다:

> 시계열 데이터에서의 이상치 탐지는 **입력 데이터 유형**, **이상치 유형**, **탐지 방법의 성질**이라는 세 가지 축으로 분류 가능한 체계적 분류체계(taxonomy)를 통해 구조화될 수 있으며, 이를 통해 연구자들이 문제에 적합한 기법을 선택할 수 있도록 안내해야 한다.

### 1.2 주요 기여

| 기여 항목 | 설명 |
|-----------|------|
| **분류 체계 제안** | 시계열 이상치 탐지 기법의 최초 전용 taxonomy 제시 |
| **포괄적 문헌 조사** | 점 이상치, 부분 수열 이상치, 전체 시계열 이상치로 구분하여 체계적 정리 |
| **소프트웨어 목록 제공** | 공개 소프트웨어/라이브러리 정리 (Table 9) |
| **미래 연구 방향 제시** | 비정규 샘플링, 점진적 학습, 동적 임계값 등 미탐구 영역 제안 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

- 시계열 이상치 탐지 분야의 **용어 혼재** (outlier, anomaly, discord 등)
- 기존 서베이들이 **시계열 전용** 분류체계를 제공하지 않음
- 탐지 기법 선택의 기준 부재

### 2.2 분류 체계 (Taxonomy)

**세 가지 축(Axis):**

```
Axis 1: 입력 데이터 유형
  ├─ 단변량 시계열 (Univariate)
  └─ 다변량 시계열 (Multivariate)

Axis 2: 이상치 유형
  ├─ 점 이상치 (Point Outlier)
  ├─ 부분 수열 이상치 (Subsequence Outlier)
  └─ 이상 시계열 (Outlier Time Series)

Axis 3: 탐지 방법의 성질
  ├─ 단변량 탐지 (Univariate Detection)
  └─ 다변량 탐지 (Multivariate Detection)
```

### 2.3 주요 탐지 방법 및 수식

#### (A) 점 이상치 탐지 — 모델 기반 (단변량)

가장 기본적인 점 이상치 판별 기준:

$$|x_t - \hat{x}_t| > \tau $$

- $x_t$: 시간 $t$에서의 관측값
- $\hat{x}_t$: 예측/추정 기댓값
- $\tau$: 사전 정의된 임계값

**추정 모델(Estimation Model):** 과거·현재·미래 데이터 사용

$$\{x_{t-k_1}, \ldots, x_t, \ldots, x_{t+k_2}\} \rightarrow \hat{x}_t$$

**예측 모델(Prediction Model):** 과거 데이터만 사용 (스트리밍 가능)

$$\{x_{t-k}, \ldots, x_{t-1}\} \rightarrow \hat{x}_t$$

#### (B) 점 이상치 탐지 — 밀도 기반

$$x_t \text{ is an outlier} \iff |\{x \in X \mid d(x, x_t) \leq R\}| < \tau $$

- $d$: 유클리드 거리
- $R$: 이웃 반경
- $\tau$: 최소 이웃 수 임계값

#### (C) 점 이상치 탐지 — 히스토그래밍 (Deviant)

```math
E_X(H^*_B) > E_{X-D}(H^*_{B-|D|})
```

- $H^*_B$: $B$개 버킷의 최적 히스토그램
- $D$: 이탈 집합(deviant set)
- $E_X(\cdot)$: 근사 오류 합계

#### (D) 다변량 점 이상치 탐지 — 모델 기반

$$\|\boldsymbol{x}_t - \hat{\boldsymbol{x}}_t\| > \tau $$

- $\boldsymbol{x}_t$: $k$차원 관측 벡터

#### (E) 다변량 점 이상치 탐지 — 유사도 기반

$$s(\boldsymbol{x}_t, \hat{\boldsymbol{x}}_t) > \tau $$

- $s$: 두 다변량 점 간의 비유사도 측도

#### (F) 부분 수열 이상치 탐지 — Discord 탐지

$$\forall S \in A, \quad \min_{D' \in A, D \cap D' = \emptyset} d(D, D') > \min_{S' \in A, S \cap S' = \emptyset} d(S, S') $$

- $D$: discord 후보 부분 수열
- $A$: 슬라이딩 윈도우로 추출된 모든 부분 수열 집합

#### (G) 부분 수열 이상치 탐지 — 유사도 기반

$$s(S, \hat{S}) > \tau $$

- $S$: 분석 대상 부분 수열
- $\hat{S}$: 정상성 기준(reference of normality)으로부터 얻은 기대값

#### (H) 부분 수열 이상치 탐지 — 예측 모델 기반

$$\sum_{i=p}^{p+n-1} |x_i - \hat{x}_i| > \tau $$

#### (I) 빈도 기반 이상치 탐지

$$|f(S) - \hat{f}(S)| > \tau $$

- $f(S)$: 부분 수열 $S$의 실제 출현 빈도
- $\hat{f}(S)$: 기대 출현 빈도

#### (J) 정보 이론 기반 이상치 탐지

$$I(S) \times f(S) > \tau $$

- $I(S)$: 부분 수열 $S$가 내포한 정보량
- $f(S)$: $S$의 출현 빈도

#### (K) 다변량 부분 수열 이상치 탐지

$$\sum_{i=p}^{p+n-1} \|\boldsymbol{x}_i - \hat{\boldsymbol{x}}_i\| > \tau $$

### 2.4 모델 구조 요약

**단변량 점 이상치 탐지 기법 분류:**

```
단변량 점 이상치
  ├─ 모델 기반
  │   ├─ 추정 모델: Median, MAD, EWMA, STL, GMM, ANN, ARIMA
  │   └─ 예측 모델: AR, ARIMA, CNN (DeepAnT), HTM, Student-t processes
  ├─ 밀도 기반: Distance-based (STORM)
  └─ 히스토그래밍: Deviant detection
```

**다변량 점 이상치 탐지 기법 분류:**

```
다변량 점 이상치
  ├─ 단변량 기법 적용
  │   ├─ 직접 적용: LSTM (Hundman et al., 2018)
  │   └─ 차원 축소 후 적용: PCA, ICA, Projection Pursuit
  └─ 다변량 기법 적용
      ├─ 모델 기반: Autoencoder, VAE-GRU (OmniAnomaly), CHMM, CNN
      ├─ 유사도 기반: Graph-based RBF (Cheng et al.)
      └─ 히스토그래밍: Multivariate deviant detection
```

### 2.5 성능 향상 및 한계

**성능 향상 측면:**

| 기법 | 향상 포인트 |
|------|-------------|
| DeepAnT (CNN) | 복잡한 시간적 패턴 학습 가능 |
| OmniAnomaly (VAE+GRU) | 스토캐스틱 정규화로 견고성 향상 |
| SPOT/DSPOT | 극값 이론으로 자동 임계값 설정 |
| HTM | 점진적 온라인 학습 |

**한계점:**

- **임계값 $\tau$ 선택 문제:** 대부분의 방법에서 수동 설정 필요
- **비정기 샘플링(Irregular Sampling) 미지원:** 거의 모든 기법이 등간격 가정
- **점진적 학습 부재:** 스트리밍 환경에서 모델 적응 능력 제한
- **용어 불일치:** outlier, anomaly, discord 등 혼용
- **레이블 부재 평가 어려움:** 비지도 방식의 성능 평가 기준 불명확

---

## 3. 모델의 일반화 성능 향상 가능성

논문에서 직접적으로 "일반화 성능"이라는 용어를 사용하지는 않지만, 일반화와 직결된 여러 논의를 포함하고 있다.

### 3.1 점진적 학습(Incremental Learning)의 필요성

논문은 스트리밍 환경에서의 **점진적 모델 업데이트** 필요성을 강조한다:

> "very few are able to adapt incrementally to the evolution of the stream"

일반화 관점에서, 고정 모델(fixed model)은 **개념 표류(Concept Drift)**에 취약하다:

$$P(X_{t+\Delta}) \neq P(X_t) \quad \text{for large } \Delta$$

점진적 학습은 이를 완화하여 **시간적 일반화(temporal generalization)** 성능을 향상시킬 수 있다. 논문에서 이에 해당하는 기법은:

- **HTM** (Ahmad et al., 2017): 뉴로모픽 구조로 점진적 업데이트
- **Student-t process** (Xu et al., 2016, 2017): 공분산 행렬 점진적 갱신

$$\Sigma_{t+1} = f(\Sigma_t, x_{t+1})$$

### 3.2 동적 임계값(Dynamic Threshold)의 역할

고정 임계값은 도메인 변화에 따른 분포 이동(distribution shift)에 취약하다. 동적 임계값은 일반화 성능 향상에 기여한다:

**극값 이론 기반 (SPOT, DSPOT):**

$$P(X_t > z_{q,t}) < q, \quad \forall t \geq 0$$

이 방법은 GPD(Generalized Pareto Distribution)를 가정하여 임계값을 자동으로 갱신한다:

$$z_{q,t} = z_{q,t-1} + g(x_t, \text{GPD parameters})$$

**Hundman et al. (2018)의 동적 임계값:**

$$\tau_t = f(\text{smoothed residuals from past data})$$

### 3.3 차원 축소의 일반화 기여

다변량 시계열에서 변수 간 상관관계를 활용하면 과적합을 방지할 수 있다:

**PCA 기반 접근 (Papadimitriou et al., 2005):**

$$\mathbf{X}_{reduced} = \mathbf{W}^T \mathbf{X}, \quad \mathbf{W} \in \mathbb{R}^{k \times p}, \, p < k$$

**ICA 기반 접근 (Baragona & Battaglia, 2007):**

$$\mathbf{X} = \mathbf{A} \cdot \mathbf{S}, \quad \mathbf{S}: \text{independent components}$$

이상치 판별:

$$\hat{x}_{it} = \mu_i, \quad \tau_i = 4.47\sigma_i$$

### 3.4 일반화 성능을 저해하는 요소와 개선 방향

| 저해 요소 | 현황 | 개선 방향 |
|-----------|------|-----------|
| 고정 모델 | 대부분의 기법 | 점진적/온라인 학습 |
| 고정 임계값 | 대다수 기법 | 동적/적응적 임계값 |
| 시간 정보 무시 | 일부 기법 (Cheng et al.) | 시간 정보 통합 |
| 유클리드 거리 고집 | 다수 기법 | DTW 등 대안 거리 활용 |
| 단변량 처리 | 다변량 시리즈에 단변량 적용 | 변수 간 상관관계 모델링 |

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미치는 영향

#### (1) Taxonomy의 표준화 역할

이 논문이 제시한 분류체계는 이후 연구들이 자신의 기법을 위치시키는 **기준 프레임**으로 활용되고 있다. 예를 들어, Transformer 기반 이상치 탐지 연구들도 이 taxonomy를 참조하여 point/subsequence 이상치 탐지로 분류한다.

#### (2) 딥러닝 기반 기법 연구 촉진

논문이 지적한 한계 (점진적 학습 부재, 다변량 상관관계 무시)는 이후 다음과 같은 연구들을 자극했다:

- **Transformer 기반:** Anomaly Transformer (Xu et al., 2022)
- **GNN 기반:** MTAD-GAT (Zhao et al., 2020)
- **Diffusion 기반:** ImDiffusion (Chen et al., 2023)

#### (3) 벤치마크 필요성 인식

논문에서 명시적 성능 비교 부재를 드러냄으로써, 이후 표준 벤치마크 연구 (e.g., TSB-UAD, ADBench)의 필요성을 환기시켰다.

### 4.2 앞으로 연구 시 고려할 점

#### (1) 비정기 샘플링 (Irregular Sampling)

논문이 지적한 미해결 과제:

$$T = \{t_1, t_2, \ldots, t_n\}, \quad \Delta t_i = t_{i+1} - t_i \neq \text{const}$$

**고려 방법:** Neural ODE, GRU-D, mTAND 등의 활용

#### (2) 동적·적응적 임계값

$$\tau_t = f(\text{historical distribution}, \text{concept drift})$$

임계값의 이론적 근거 강화 필요 (예: Conformal Prediction 활용)

#### (3) 변수 간 상관관계 모델링

$$\boldsymbol{x}_t \sim p(\boldsymbol{x}_t | \boldsymbol{x}_{t-1}, \ldots, \boldsymbol{x}_{t-k})$$

그래프 구조 (GNN)를 활용하여 변수 간 의존성을 명시적으로 모델링

#### (4) 레이블 없는 평가 지표

비지도 이상치 탐지의 평가를 위한 지표 개발 필요:

$$\text{Average Precision} = \sum_n (R_n - R_{n-1}) P_n$$

단, ground truth 없이 평가하는 방법론 연구 필요

#### (5) 설명 가능성 (Explainability)

탐지된 이상치의 원인 분석:

$$\text{anomaly score}(x_t) \rightarrow \text{attribution}(x_t^{(j)}), \, j \in \{1, \ldots, k\}$$

#### (6) 도메인 일반화 (Domain Generalization)

특정 도메인에서 학습된 모델이 다른 도메인에도 적용 가능하도록:

$$\min_\theta \mathbb{E}_{d \sim \mathcal{D}} [\mathcal{L}(\theta; \mathcal{D}_d)]$$

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의:** 아래 연구들은 제가 학습 데이터를 기반으로 알고 있는 내용을 정리한 것이며, 각 논문의 세부 수치/결과에 대해서는 원문 확인을 권장합니다.

### 5.1 주요 후속 연구 비교

| 논문 | 연도 | 핵심 기법 | Taxonomy상 위치 | 주요 기여 |
|------|------|-----------|-----------------|-----------|
| **MTAD-GAT** (Zhao et al.) | 2020 | GAT + GRU | 다변량 점 이상치, 다변량 탐지 | 변수 간·시간 간 attention |
| **Anomaly Transformer** (Xu et al.) | 2022 | Transformer + Association Discrepancy | 부분 수열 이상치 | Prior-Association 대비 |
| **TranAD** (Tuli et al.) | 2022 | Transformer + Adversarial Training | 다변량, 예측 기반 | 메타 학습 기반 빠른 적응 |
| **TimesNet** (Wu et al.) | 2023 | 2D 변환 + CNN | 단/다변량 점 이상치 | 시간적 변화 2D 표현 |
| **ImDiffusion** (Chen et al.) | 2023 | Diffusion Model | 다변량 점 이상치 | 생성 모델 기반 임퓨테이션 |

### 5.2 핵심 패러다임 변화

**Blázquez-García et al. (2020) 시점:**

$$\hat{x}_t = f_{\text{ARIMA/CNN/LSTM}}(x_{t-k}, \ldots, x_{t-1})$$

$$\text{anomaly if } |x_t - \hat{x}_t| > \tau$$

**2020년 이후 Transformer 기반:**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Association Discrepancy (Anomaly Transformer, Xu et al., 2022):**

$$\text{AssDis}(P, S; \mathbf{X}) = \left[\text{KL}(P_i \| S_i) + \text{KL}(S_i \| P_i)\right]_{i=1}^{N}$$

### 5.3 논문이 지적한 한계 vs. 최신 연구 대응

| 논문이 지적한 한계 | 최신 연구의 대응 |
|-------------------|-----------------|
| 점진적 학습 부재 | Online Anomaly Detection (OAD) 연구 증가 |
| 동적 임계값 부재 | Conformal Anomaly Detection (2021~) |
| 변수 상관관계 무시 | GNN 기반 (MTAD-GAT, GDN 등) |
| 시간 정보 통합 부족 | Transformer self-attention의 시간적 위치 인코딩 |
| 비정기 샘플링 | Neural ODE, mTAND 등 |
| 설명 가능성 부재 | SHAP 기반 이상치 기여도 분석 연구 |

---

## 참고 자료

### 본 논문
- **Blázquez-García, A., Conde, A., Mori, U., & Lozano, J. A.** (2020). *A review on outlier/anomaly detection in time series data*. arXiv:2002.04236v1. ACM Computing Surveys (최종 게재).

### 논문 내 인용 주요 문헌
- Hawkins, D. M. (1980). *Identification of Outliers*. Springer.
- Chandola, V., Banerjee, A., & Kumar, V. (2009). *Anomaly detection: A survey*. ACM Computing Surveys.
- Gupta, M., Gao, J., Aggarwal, C., & Han, J. (2014). *Outlier Detection for Temporal Data: A Survey*. IEEE TKDE.
- Ahmad, S. et al. (2017). *Unsupervised real-time anomaly detection for streaming data*. Neurocomputing.
- Munir, M. et al. (2019). *DeepAnT: A Deep Learning Approach for Unsupervised Anomaly Detection in Time Series*. IEEE Access.
- Su, Y. et al. (2019). *Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network*. KDD.
- Hundman, K. et al. (2018). *Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding*. KDD.
- Siffer, A. et al. (2017). *Anomaly Detection in Streams with Extreme Value Theory*. KDD.

### 2020년 이후 비교 연구 (학습 데이터 기반, 원문 확인 권장)
- Zhao, H. et al. (2020). *Multivariate Time-Series Anomaly Detection via Graph Attention Network*. ICDM.
- Xu, J. et al. (2022). *Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy*. ICLR.
- Tuli, S. et al. (2022). *TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data*. VLDB.
