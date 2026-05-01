# Anomaly Detection in Streams with Extreme Value Theory

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

본 논문(Siffer et al., KDD 2017)의 핵심 주장은 다음과 같습니다:

> **스트리밍 시계열 데이터에서 이상치(anomaly)를 탐지할 때, 데이터의 분포에 대한 가정 없이, 수동 임계값 설정 없이, 오직 극단값 이론(Extreme Value Theory, EVT)만을 이용하여 통계적으로 엄밀한 자동 임계값을 설정할 수 있다.**

### 주요 기여

| 기여 항목 | 내용 |
|---|---|
| **알고리즘 SPOT** | 정상(stationary) 분포를 따르는 스트림에서의 이상치 탐지 |
| **알고리즘 DSPOT** | 개념 표류(concept drift)가 있는 스트림에서의 이상치 탐지 |
| **분포 무가정(distribution-free)** | EVT의 POT 접근법으로 데이터 분포 가정 불필요 |
| **단일 파라미터** | 사용자가 설정하는 파라미터는 오직 위험 수준 $q$ (허용 오경보율) |
| **수치 최적화 개선** | Grimshaw 기법의 검색 구간 축소(Proposition 1) 및 수치 안정성 향상 |
| **스트리밍 최초 EVT 적용** | 저자들에 따르면 EVT를 스트리밍 이상치 탐지에 최초 적용 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**문제 정의:**

스트리밍 시계열 $(X_t)_{t \geq 0}$이 i.i.d. 관측값으로 구성될 때, 다음을 만족하는 임계값 $z_q$를 자동으로 설정하는 것:

$$\mathbb{P}(X > z_q) < q, \quad q \text{는 임의로 작은 양수}$$

**기존 방법의 한계:**
- **수동 임계값 설정**: 도메인 지식 필요, 동적 환경 부적합
- **분포 가정 필요**: Gaussian, Uniform 등 특정 분포 가정이 실제 데이터에 맞지 않을 수 있음
- **스트리밍 불가**: 기존 정적 이상치 탐지 기법들은 데이터를 여러 번 스캔해야 함
- **확률 하한 문제**: 학습 기반 방법(예: Laxhammar & Falkman, 2014)은 오경보율이 $\frac{1}{k+1}$ 이하로 내려갈 수 없음

---

### 2.2 제안 방법 및 수식

#### 2.2.1 이론적 배경: 극단값 이론 (EVT)

**극단값 분포 (EVD):**

Fisher-Tippett-Gnedenko 정리에 의해, 어떤 분포의 극단값도 아래 형태로 수렴합니다:

$$G_\gamma : x \mapsto \exp\left(-(1 + \gamma x)^{-\frac{1}{\gamma}}\right), \quad \gamma \in \mathbb{R}, \quad 1 + \gamma x > 0$$

여기서 $\gamma$는 극단값 지수(extreme value index)이며, 꼬리 형태를 결정합니다:

| 꼬리 형태 | 조건 | 예시 분포 |
|---|---|---|
| Heavy tail: $\mathbb{P}(X > x) \approx x^{-1/\gamma}$ | $\gamma > 0$ | Fréchet |
| Exponential tail: $\mathbb{P}(X > x) \approx e^{-x}$ | $\gamma = 0$ | Gamma |
| Bounded tail: $\mathbb{P}(X > x) = 0$ for $x \geq \tau$ | $\gamma < 0$ | Uniform |

#### 2.2.2 Peaks-Over-Threshold (POT) 접근법

**Pickands-Balkema-de Haan 정리 (EVT 제2정리):**

$$\bar{F}_t(x) = \mathbb{P}(X - t > x \mid X > t) \underset{t \to \tau}{\sim} \left(1 + \gamma \frac{x}{\sigma(t)}\right)^{-\frac{1}{\gamma}}$$

즉, 임계값 $t$를 초과하는 초과량 $Y = X - t$는 **일반화 파레토 분포(Generalized Pareto Distribution, GPD)**를 따릅니다.

**임계값 계산 수식 (핵심):**

$$\boxed{z_q \simeq t + \frac{\hat{\sigma}}{\hat{\gamma}}\left(\left(\frac{qn}{N_t}\right)^{-\hat{\gamma}} - 1\right)}$$

여기서:
- $t$: 초기 임계값 (실험적 상위 98% 분위수)
- $\hat{\gamma}, \hat{\sigma}$: GPD 파라미터 추정값 (최대우도추정, MLE)
- $n$: 총 관측 수
- $N_t$: 피크(초과) 개수 ($X_i > t$인 경우)
- $q$: 사용자 설정 위험 수준

#### 2.2.3 최대우도추정 (MLE)

GPD에 대한 로그우도함수:

$$\log \mathcal{L}(\gamma, \sigma) = -N_t \log \sigma - \left(1 + \frac{1}{\gamma}\right)\sum_{i=1}^{N_t} \log\left(1 + \frac{\gamma}{\sigma}Y_i\right)$$

#### 2.2.4 Grimshaw 기법 및 개선

최적화는 2변수 문제를 1변수 문제로 축소합니다. $x^\* = \gamma^\*/\sigma^*$라 하면:

$$u(x) \cdot v(x) = 1$$

여기서:

$$u(x) = \frac{1}{N_t}\sum_{i=1}^{N_t} \frac{1}{1 + xY_i}, \quad v(x) = 1 + \frac{1}{N_t}\sum_{i=1}^{N_t} \log(1 + xY_i)$$

**논문의 개선 (Proposition 1):** 해를 탐색하는 구간을 아래와 같이 축소:

$$x^* \leq 0 \quad \text{또는} \quad x^* \geq 2\frac{\bar{Y} - Y^m}{\bar{Y}Y^m}$$

따라서 탐색 구간은:

$$\left(-\frac{1}{Y^M}, 0\right) \quad \text{및} \quad \left[2\frac{\bar{Y} - Y^m}{\bar{Y}Y^m},\ 2\frac{\bar{Y} - Y^m}{(Y^m)^2}\right]$$

---

### 2.3 모델 구조

#### SPOT 알고리즘 (정상 스트림)

```
1. 초기화: n개 관측값으로 POT 수행 → z_q, t 계산
2. 스트리밍 단계:
   - X_i > z_q → 이상치 탐지 (anomaly)
   - t < X_i ≤ z_q → 피크(peak): GPD 업데이트, z_q 재계산
   - X_i ≤ t → 정상(normal): 카운터 증가
```

#### DSPOT 알고리즘 (개념 표류 대응)

변수 변환을 통해 로컬 행동을 제거:

$$X'_i = X_i - M_i, \quad M_i = \frac{1}{d}\sum_{k=1}^{d} X^*_{i-k}$$

여기서 $M_i$는 최근 $d$개의 **정상** 관측값의 이동 평균입니다. $X'_i$에 SPOT을 적용하여 개념 표류에 강건하게 동작합니다.

---

### 2.4 성능 향상 및 한계

#### 성능

| 측면 | 결과 |
|---|---|
| **정확도** | 이론적 임계값 대비 오차율 5% 이하 (n=1000 이상 시) |
| **침입 탐지 (MAWI)** | TPR 86%, FPR < 4% |
| **처리 속도** | 평균 약 351 µs/iteration (SPOT), 1000 obs/sec 이상 처리 가능 |
| **메모리** | 피크 저장 비율 약 1.9% (전체 스트림 대비) |

#### 한계

1. **i.i.d. 가정**: 관측값들이 독립 동일 분포를 따른다는 가정 필요
2. **단변량(univariate) 한정**: 다변량 시계열에 직접 적용 불가
3. **단봉(unimodal) 분포 가정**: 다봉(multimodal) 분포에는 부적합
4. **피크 집합 무한 성장**: 장기 모니터링 시 메모리 증가 (상한 설정으로 완화 가능)
5. **이동 평균의 한계**: DSPOT의 $M_i$는 단순 이동 평균으로, 복잡한 계절성 패턴에는 부적합

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 EVT 기반 분포 무가정의 일반화

EVT의 가장 강력한 일반화 특성은 **수렴 정리**에 있습니다. 초과량의 분포가 원래 데이터 분포에 관계없이 GPD로 수렴하므로:

$$\bar{F}_t(x) \xrightarrow{t \to \tau} \left(1 + \gamma \frac{x}{\sigma}\right)^{-1/\gamma}$$

이는 Gaussian, Pareto, Exponential 등 어떤 분포에서도 동일하게 작동하여 **분포 독립적 일반화(distribution-free generalization)**를 달성합니다.

### 3.2 파라미터 수렴 및 일반화 보장

MLE의 수렴 특성:
- $\gamma > -\frac{1}{2}$: $\sqrt{N_t}$ 속도로 정규분포에 수렴
- $-1 < \gamma < -\frac{1}{2}$: $N_t^{-\gamma}$ 속도로 초효율적(superefficient) 수렴

이로 인해 초기 배치 크기 $n$에 상관없이 충분한 데이터가 축적되면 임계값이 이론적 값으로 수렴합니다. 실험에서도 $n=300$부터 $n=5000$까지 모든 경우에서 오차가 같은 값으로 수렴하는 것이 확인되었습니다.

### 3.3 DSPOT을 통한 비정상성 환경 일반화

DSPOT의 변수 변환:

$$X'_i = X_i - M_i$$

이 변환은 **로컬 정상성(local stationarity)** 가정으로 완화하여, 전체 스트림이 정상적이지 않아도 로컬 변동 $X'_i$는 정상적이라고 가정합니다. 이는 개념 표류가 있는 실제 환경에서의 일반화 능력을 향상시킵니다.

### 3.4 일반화 성능의 한계와 개선 방향

| 한계 요인 | 현재 상태 | 개선 방향 |
|---|---|---|
| **i.i.d. 가정** | 필수 가정 | 시간 종속 EVT (stationary mixing 조건) 도입 |
| **단변량 한정** | 1차원만 처리 | 다변량 EVT 확장 (극단값 코퓰라 이론) |
| **이동 평균 단순성** | 단순 평균 | LSTM/TCN 기반 로컬 모델 대체 |
| **초기 배치 의존성** | $n \approx 1000$ 필요 | 베이지안 사전 분포 활용으로 소수 샘플 대응 |

---

## 4. 향후 연구 영향 및 고려사항

### 4.1 향후 연구에 미치는 영향

1. **EVT 기반 이상치 탐지의 표준화**: 스트리밍 환경에서 EVT를 이상치 탐지에 처음 체계적으로 적용하여 이후 연구의 기반이 됨
2. **자동 임계값 설정 패러다임**: 수동 임계값 없이 통계적으로 보장된 임계값 생성이라는 새로운 방향 제시
3. **오경보율 제어 프레임워크**: $q$라는 단일 파라미터로 FPR을 직접 제어하는 설계 원칙 제시
4. **복합 시스템 통합 가능성**: 더 복잡한 이상치 탐지 시스템의 서브모듈로 사용 가능

### 4.2 향후 연구 시 고려사항

**이론적 측면:**
- **비 i.i.d. 확장**: $\alpha$-mixing 또는 $\beta$-mixing 조건 하에서의 EVT 적용 연구
- **다변량 EVT**: 극단값 코퓰라(extreme value copula) 이론과의 결합
- **적응형 임계값 $t$**: 현재 고정된 98% 분위수 대신 데이터 적응적 선택 (Mean Excess Plot 안정화)

**실용적 측면:**
- **비정상성 패턴 모델링**: 단순 이동 평균 대신 계절성을 반영하는 STL 분해 또는 Fourier 분석 활용
- **피크 집합 관리**: 장기 운영 시 오래된 피크 제거 전략 (sliding window 피크 집합)
- **온라인 적응 속도**: 개념 표류 감지 메커니즘과의 결합 (ADWIN 등)
- **딥러닝과의 융합**: LSTM, Transformer로 로컬 모델 $M_i$를 정교하게 모델링 후 잔차에 EVT 적용

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

아래 연구들은 본 논문의 흐름을 잇거나 한계를 보완하는 연구들입니다. 단, 일부 세부 내용은 논문 원문에 접근하지 않고 알려진 정보를 바탕으로 서술하므로 세부 수치는 원문 확인을 권장합니다.

### 5.1 주요 후속 연구 비교

| 연구 | 방법 | SPOT 대비 개선점 | 한계 |
|---|---|---|---|
| **Xu et al. (2022), "Anomaly Detection for Multivariate Time Series through Modeling Temporal Dependence of Stochastic Variables"** | GNN + 통계 | 다변량, 시간 종속성 | EVT 미사용, 분포 가정 |
| **Garg et al. (2021), "An Evaluation of Anomaly Detection and Diagnosis in Multivariate Time Series"** (IEEE TNNLS) | 벤치마크 비교 | SPOT 포함 종합 평가 | 새 기법 제안 아님 |
| **Nakamura et al. (2020), "MERLIN: Parameter-Free Discovery of Arbitrary Length Anomalies in Massive Time Series Archives"** (ICDM 2020) | Matrix Profile 기반 | 파라미터 불필요 | 스트리밍 실시간 처리 어려움 |
| **Laptev et al. (후속 연구들)** | LSTM + 임계값 | 딥러닝 로컬 모델 | 학습 데이터 필요 |

### 5.2 EVT + 딥러닝 융합 연구 동향

2020년 이후 주목할 만한 방향은 **EVT와 딥러닝의 결합**입니다:

$$\hat{X}_t = f_\theta(X_{t-1}, \ldots, X_{t-k})$$
$$\epsilon_t = X_t - \hat{X}_t$$

잔차 $\epsilon_t$에 EVT(POT)를 적용하는 방식으로, 딥러닝이 로컬 패턴을 학습하고 EVT가 극단 잔차의 임계값을 통계적으로 설정합니다. 이는 DSPOT의 이동 평균 $M_i$를 딥러닝으로 대체한 것으로 볼 수 있습니다.

### 5.3 한계 보완 연구 방향 정리

```
SPOT/DSPOT의 한계
        │
        ├── i.i.d. 가정 → 시간 종속 EVT 연구 (mixing conditions)
        ├── 단변량 한정 → 다변량 EVT + Copula 연구  
        ├── 단순 로컬 모델 → LSTM/Transformer 잔차 + EVT
        └── 개념 표류 감지 → ADWIN + DSPOT 결합
```

---

## 참고 자료

1. **Siffer, A., Fouque, P.A., Termier, A., & Largouët, C. (2017).** "Anomaly Detection in Streams with Extreme Value Theory." *KDD 2017 - Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.* DOI: 10.1145/3097983.3098144. (HAL: hal-01640325) — **본 논문 원문**

2. **Beirlant, J., Goegebeur, Y., Segers, J., & Teugels, J. (2006).** *Statistics of Extremes: Theory and Applications.* John Wiley & Sons.

3. **Pickands, J. III. (1975).** "Statistical Inference Using Extreme Order Statistics." *The Annals of Statistics.*

4. **Balkema, A.A., & De Haan, L. (1974).** "Residual Life Time at Great Age." *The Annals of Probability.*

5. **Grimshaw, S.D. (1993).** "Computing Maximum Likelihood Estimates for the Generalized Pareto Distribution." *Technometrics, 35(2), 185–191.*

6. **Chandola, V., Banerjee, A., & Kumar, V. (2009).** "Anomaly Detection: A Survey." *ACM Computing Surveys.*

7. **Sadik, S., & Gruenwald, L. (2014).** "Research Issues in Outlier Detection for Data Streams." *ACM SIGKDD Explorations Newsletter, 15(1), 33–40.*

8. **Fisher, R.A., & Tippett, L.H.C. (1928).** "Limiting Forms of the Frequency Distribution of the Largest or Smallest Member of a Sample." *Mathematical Proceedings of the Cambridge Philosophical Society.*

9. **Gnedenko, B. (1943).** "Sur la Distribution Limite du Terme Maximum d'une Série Aléatoire." *Annals of Mathematics.*

> **정확도 주의사항**: 섹션 5(2020년 이후 최신 연구 비교)의 일부 후속 연구 세부 내용은 원문에 직접 접근하지 않은 상태에서 작성되었습니다. 해당 내용의 정확한 수치와 세부 방법론은 각 논문 원문을 직접 확인하시기 바랍니다.
