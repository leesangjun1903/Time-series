# Adaptive Conformal Anomaly Detection with Time Series Foundation Models for Signal Monitoring

## 📋 참고 자료

- **논문 원문**: Martinez Gil, N., O'Donncha, F., Gifford, W.M., Zhou, N., Patel, D.C., & Vaculin, R. (2026). "Adaptive Conformal Anomaly Detection with Time Series Foundation Models for Signal Monitoring." *Published as a conference paper at ICLR 2026.* arXiv:2604.20122v1
- **코드**: https://github.com/ibm-granite/granite-tsfm/tree/main/notebooks/hfdemo/adaptive_conformal_tsad
- Barber, R.F., Candès, E.J., Ramdas, A., & Tibshirani, R.J. (2023). "Conformal prediction beyond exchangeability." *The Annals of Statistics*, 51(2):816–845.
- Gibbs, I. & Candès, E. (2021). "Adaptive conformal inference under distribution shift." *NeurIPS*, 34:1660–1672.
- Zaffran, M., et al. (2022). "Adaptive conformal predictions for time series." *ICML*, pp. 25834–25866.
- Liu, Q. & Paparrizos, J. (2024). "The elephant in the room: Towards a reliable time-series anomaly detection benchmark." *NeurIPS Datasets and Benchmarks Track.*
- Ansari, A.F., et al. (2024). "Chronos: Learning the language of time series." arXiv:2403.07815.
- Ekambaram, V., et al. (2024). "TTMs: Fast multi-level tiny time mixers." arXiv:2401.03955.
- Auer, A., et al. (2025). "TiRex: Zero-shot forecasting across long and short horizons." arXiv:2505.23719.
- Boniol, P., et al. (2024). "Dive into time-series anomaly detection: A decade review." arXiv:2412.20512.

---

## 1. 핵심 주장과 주요 기여 (요약)

### 핵심 주장

이 논문은 ** $\mathcal{W}_1$ -ACAS (1-Wasserstein Adaptive Conformal Anomaly Score)**를 제안한다. 이 방법은 사전 학습된 시계열 파운데이션 모델(TSFM)의 예측 오차를 비적합도 점수(nonconformity score)로 활용하여, **추가 파인튜닝 없이** 온라인 이상 탐지를 수행한다. 핵심은 이상 점수가 **p-value (허위 경보율)로 직접 해석 가능**하며, **비교환성(non-exchangeability)** 하에서도 보정된(calibrated) 경보율 보장을 유지한다는 점이다.

### 주요 기여 4가지

| 특성 | 설명 |
|------|------|
| **해석 가능성** | 이상 점수 = p-value (허위 경보율과 직접 대응) |
| **분포 무관성** | 분위수 컨포멀 예측 기반, 중꼬리/복잡 분포에 강건 |
| **적응성** | Wasserstein 거리 기반 가중치 학습으로 분포 변화에 온라인 적응 |
| **Post-Hoc & 모델 무관** | 기존 TSFM에 바로 적용, 재학습 불필요 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

**산업 현장의 핵심 제약:**
1. **데이터 부족**: 충분한 학습 데이터 없이 즉각적 모니터링이 필요
2. **비교환성 위반**: 시계열 데이터는 표준 컨포멀 예측의 교환가능성 가정을 위반
3. **해석 불가능한 임계값**: 기존 방법들은 데이터 전체 통계에 의존하는 고정 임계값 사용
4. **분포 변화**: 비정상적(non-stationary) 환경에서의 적응 부재

### 2.2 배경: 컨포멀 이상 탐지

**비적합도 점수(Nonconformity Score):**

$$S = e(Y, h(X)), \quad e(Y, \hat{Y}) = |Y - \hat{Y}|$$

**표준 분할 컨포멀 탐지기:**

$$C_\alpha(X_{n+1}, Y_{n+1}) = \mathbf{1}[S_{n+1} > \hat{q}_\alpha], \quad \hat{q}_\alpha = Q_{1-\alpha}\left(\sum_{i=1}^n \frac{1}{n+1}\delta_{S_i} + \frac{1}{n+1}\delta_\infty\right) $$

**가중 컨포멀 분위수 (비교환성 확장):**

$$\mathbb{Q}_{1-\alpha}(\mathbf{s}, \mathbf{w}) = Q_{1-\alpha}\left(\sum_{i=1}^n \frac{w_i}{||\mathbf{w}||_1 + 1}\delta_{S_i} + \frac{1}{||\mathbf{w}||_1 + 1}\delta_\infty\right) $$

**명제 3.1** (Barber et al., 2023 직접 적용): 가중 탐지기 $A_{n+1} = \mathbf{1}[S_{n+1} > \hat{q}^w_\alpha]$의 허위 경보율 보장:

$$\mathbb{P}(A_{n+1} = 1) \leq \alpha + \sum_{i=1}^n \frac{w_i}{||\mathbf{w}||_1+1} d_{TV}(\mathbf{s}, \mathbf{s}^i) $$

여기서 $d_{TV}(\mathbf{s}, \mathbf{s}^i)$는 전체변동 거리(total variation distance).

### 2.3 제안 방법: 적응적 컨포멀 이상 점수

#### Step 1: 컨포멀 이상 점수 정의

과거 비적합도 점수 $\mathbf{s} = \{S_i\}\_{i=1}^t$와 가중치 $\mathbf{w}$가 주어질 때, 테스트 샘플 $S_{t+1}$에 대한 **컨포멀 p-value**:

$$\beta_\mathbf{w}(S_{t+1}) = \sup\{\alpha \in [0,1] : S_{t+1} \leq \mathbb{Q}_{1-\alpha}(\mathbf{s}, \mathbf{w})\} $$

이를 닫힌 형태(closed form)로 표현하면:

$$\beta_{t+1}(\mathbf{w}) := \frac{1 + \sum_{k=j_{t+1}}^n w_{\pi^{-1}(k)}}{|\mathbf{w}|+1}, \quad j_{t+1} = \sum_{i=1}^t \mathbf{1}[S_{t+1} \leq S_i] $$

여기서 $\pi$는 과거 점수의 오름차순 정렬 매핑, $\pi^{-1}(k)$는 역정렬 연산.

**명제 4.1**: $C_{\beta_\mathbf{w}}(X_{t+1}, Y_{t+1}) = \mathbf{1}[\beta_\mathbf{w}(S_{t+1}) < \alpha]$는 명제 3.1의 허위 경보율 보장을 만족.

#### Step 2: 최적 가중치 학습 (비교환성 대응)

$\beta_\mathbf{w}(S_{t+1}) \sim \mathcal{U}[0,1]$이 되도록 **1-Wasserstein 거리 최소화**:

$$\min_\mathbf{w} \mathcal{W}_1(F_{\beta_{t+1}(\mathbf{w})}, F_U) \quad \text{s.t.} \quad |\mathbf{w}| > \frac{1}{\alpha_c} - 1, \quad w_i \in [0,1], \forall i $$

이중 정의(dual definition)에 의해:

$$\mathcal{W}_1(F_{\beta_{t+1}(\mathbf{w})}, F_U) = \mathbb{E}_{\alpha \sim \mathcal{U}[0,1]} |\mathbb{P}(\beta_{t+1}(\mathbf{w}) \leq \alpha) - \alpha| $$

즉, 모든 허위 경보율에 대해 **보정 격차(calibration gap)를 균일하게 최소화**.

#### Step 3: 경험적 근사 및 최적화

$n_b$개의 배치 샘플로 경험적 CDF 근사:

$$\hat{F}_{\beta_{t+1}(\mathbf{w})}(\alpha) = \frac{1}{n_b}\sum_{j=1}^{n_b} \mathbf{1}[\beta_{t+j}(\mathbf{w}) \leq \alpha] $$

경험적 $\mathcal{W}_1$ 목적 함수:

$$\mathcal{W}_1(\hat{F}_{\beta_{t+1}(\mathbf{w})}, F_U) = \sum_{k=1}^{n_b} \int_{\frac{k-1}{n_b}}^{\frac{k}{n_b}} |\beta_{t+\hat{\pi}^{-1}(k)}(\mathbf{w}) - \alpha| d\alpha $$

**투영 경사 하강법(Projected Gradient Descent)** 업데이트:

$$\mathbf{w}_{t+n_b+1} = \Pi_{\mathbf{w} \in [0,1]^n, |\mathbf{w}| > \frac{1}{\alpha_c}-1}\left[\mathbf{w}_{t+n_b} - \frac{\gamma}{n_b}\sum_{i=1}^{n_b}\frac{\partial \mathcal{W}_1}{\partial \beta_{t+i}}\frac{\partial \beta_{t+i}(\mathbf{w}_{t+n_b})}{\partial w_k}\right] $$

편미분의 닫힌 형태:

```math
\frac{\partial \mathcal{W}_1}{\partial \beta_{t+i}} = \begin{cases} -\frac{1}{n_b}, & \text{if } \beta_{t+i} < \frac{\hat{\pi}(i)-1}{n_b} \\ 2\beta_{t+i} - \frac{2\hat{\pi}(i)-1}{n_b}, & \text{if } \frac{\hat{\pi}(i)-1}{n_b} \leq \beta_{t+i} \leq \frac{\hat{\pi}(i)}{n_b} \\ +\frac{1}{n_b}, & \text{if } \beta_{t+i} > \frac{\hat{\pi}(i)}{n_b} \end{cases}
```

$$\frac{\partial \beta_{t+i}(\mathbf{w})}{\partial w_k} = \frac{-\beta_{t+i}(\mathbf{w}) + \mathbf{1}[j_{t+i} \leq \pi(k)]}{||\mathbf{w}||_1 + 1} $$

#### Step 4: 다중 예측 지평선 집계

$D$개의 예측 지평선에서 독립적으로 알고리즘 실행, 중앙값으로 집계:

$$\bar{\beta}_{t+1} = \text{median}_{d \in [D]} \beta^d_{t+1}, \quad \beta^d_{t+1} = \beta_{\mathbf{w}^d}(S^d_{t+1}) $$

이는 과반수의 지평선별 탐지기가 이상으로 판단할 때만 경보 발생.

### 2.4 모델 구조

```
입력 시계열 Y_{0:t}
       ↓
슬라이딩 윈도우 (컨텍스트 길이 nc=52)
       ↓
사전학습 TSFM (TTM / Chronos / TiRex) — 파인튜닝 없음
       ↓
D=15 지평선별 예측값 Ŷ^d_{t+1}
       ↓
비적합도 점수: S^d_{t+1} = |Y_{t+1} - Ŷ^d_{t+1}|
       ↓
D개의 W1-ACAS 인스턴스 (병렬)
  → 가중치 w^d 적응적 업데이트 (ADAM, γ=0.001, nb=10)
  → β^d_{t+1} 계산
       ↓
중앙값 집계: β̄_{t+1} = median_d β^d_{t+1}
       ↓
이상 탐지: β̄_{t+1} < α_c (=0.01) → 이상 경보
```

**다변량 확장 (p-value 결합):**
- Fisher 방법: $\rho_{t+1} = 1 - F^{-1}\_{\chi^2_{2n_f}}(-2\sum_f \bar{\beta}^f_{t+1})$
- 조화평균 p-value (HMP): $\rho_{t+1} = \frac{n_f}{\sum_f 1/\bar{\beta}^f_{t+1}}$

### 2.5 성능 향상

**단변량 데이터셋 (7개: YAHOO, NEK, NAB, MSL, IOPS, STOCK, WSD) 결과:**

| 방법 | PA-F1 ↑ | Affiliation-F ↑ | FPR ↓ | CalErr ↓ |
|------|---------|----------------|-------|---------|
| TiRex + W1-ACAS | **0.925** | **0.897** | 0.084 | **0.025** |
| Chronos + W1-ACAS | 0.912 | 0.893 | **0.077** | **0.025** |
| TTM + W1-ACAS | 0.889 | 0.886 | 0.082 | 0.029 |
| CNN (semi-supervised) | 0.858 | 0.881 | 0.083 | 0.643 |
| POLY | 0.527 | 0.848 | 0.334 | 0.282 |

**핵심 성능 특징:**
- 임계값 의존 지표(PA-F1, Affiliation-F)에서 **최우수 성능**
- 반지도 학습 방법(CNN, USAD, OmniAnomaly) 대비 **CalErr에서 압도적 우위** (CNN: 0.643 vs W1-ACAS: 0.025)
- FPR-임계값 곡선에서 **가장 보수적이고 분산이 낮은** 임계값 제공
- 다변량 데이터셋(TAO, GECCO, LTDB, Genesis)에서도 **최고 또는 경쟁력 있는 성능**

### 2.6 한계

논문에서 명시적으로 인정하는 한계:

1. **임계값 독립 지표(AUC-PR, VUS-PR)에서의 경쟁력 제한**: MOMENT Zero-Shot이 일부 데이터셋에서 VUS-PR 기준 더 높음 (예: MOMENT ZS VUS-PR = 0.461 vs TiRex W1-ACAS = 0.438)
2. **예측 품질 의존성**: TSFM의 예측 오차 수준이 이상 탐지 성능과 직결 (TTM의 YAHOO 예측 오차가 높을수록 AD 성능 저하)
3. **최소 데이터 요구**: 유효 샘플 크기 제약 $|\mathbf{w}| > \frac{1}{\alpha_c} - 1$으로 인해 최소 $n_c = \lceil 1/\alpha_c - 1 \rceil$개의 정상 관측값 필요 (α_c=0.01이면 99개)
4. **컨텍스트 특성 미활용**: 현재 가중치는 과거 오차 분포만 활용, 입력 공변량(covariate) 정보 미반영
5. **단순 집계 방식**: 다변량의 경우 p-value 결합에 독립성 가정이 내포됨

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 성능의 이론적 기반

$\mathcal{W}_1$-ACAS의 일반화는 Barber et al. (2023)의 비교환성 하의 컨포멀 예측 경계에 직접 의존한다. 탐지기의 허위 경보율 상계는:

$$\mathbb{P}(A_{n+1} = 1) \leq \alpha + \sum_{i=1}^n \frac{w_i}{||\mathbf{w}||_1+1} d_{TV}(\mathbf{s}, \mathbf{s}^i)$$

이 경계에서 **일반화 향상 메커니즘**:
- $d_{TV}(\mathbf{s}, \mathbf{s}^i)$가 작은(= 현재 샘플과 교환 가능한) 과거 점수에 높은 $w_i$ 부여 → 상계의 초과분 최소화
- $||\mathbf{w}||_1$을 최대화 → 하계( $\alpha - \sum\_i \frac{w\_i}{||\mathbf{w}||_1+1}d\_{TV} - \frac{1}{||\mathbf{w}||_1+1}$ ) 개선

### 3.2 분포 변화(Distribution Shift) 하의 일반화

**W1 목적 함수가 균일 보정을 보장하는 원리:**

$$\mathcal{W}_1(F_{\beta_{t+1}(\mathbf{w})}, F_U) = \mathbb{E}_{\alpha \sim \mathcal{U}[0,1]} |\mathbb{P}(\beta_{t+1}(\mathbf{w}) \leq \alpha) - \alpha|$$

이를 최소화하면 **모든 $\alpha \in [0,1]$에 대해 동시에** $\mathbb{P}(\beta_{t+1}(\mathbf{w}) \leq \alpha) \approx \alpha$를 달성:
- 단일 오차율 최적화(ACI, SAOCP 등)와 달리 **전체 경보율 범위에서 보정**
- 기존 AdaptiveCI 계열은 핀볼 손실 최적화로 특정 분위수에 편향

**시뮬레이션 검증 (Random Shift & Jump Shift):**
- 연속 분포 변화: $\mu_{t+1} = \mu_t + \frac{1}{2}(\mu_t - \mu_{t-1}) + \frac{1}{2}\epsilon_t$
- 급격한 분포 변화: 500 스텝마다 단계적 점프
- 두 설정 모두에서 W1-ACAS가 지상진실(Ground Truth) p-value의 CDF에 가장 근접

### 3.3 제로샷(Zero-Shot) 일반화

**TSFM의 제로샷 능력 활용:**
- TTM(TSMixer 기반), Chronos(Transformer 기반), TiRex(xLSTM 기반) 등 다양한 아키텍처와 플러그인 방식으로 결합
- 각 TSFM의 제로샷 예측 오차를 비적합도 점수로 활용 → **모델 종류와 무관한 일반화**
- 표 5에서 세 TSFM의 MAE/RMSE가 유사한 수준 → 이상 탐지 성능도 유사하게 일반화됨을 확인

### 3.4 다변량 및 도메인 간 일반화

**p-value 집계를 통한 다변량 확장:**
- 각 특성(feature) 차원에 독립적으로 알고리즘 적용 → **특성 수에 무관한 확장성**
- Fisher 방법과 HMP의 차별화된 성능: Fisher($\mathcal{W}_1$-ACAS-F)는 GECCO, TAO에서 강세, HMP($\mathcal{W}_1$-ACAS-H)는 Genesis, LTDB에서 강세
- 다변량 데이터셋(TAO: 3특성, GECCO: 9특성, Genesis: 18특성)에서 **모두 경쟁력 있는 성능** 확인

### 3.5 주기적 패턴 포착에 의한 일반화

가중치 $\mathbf{w}$의 수렴 패턴 분석:
- 주기적 오차 패턴이 있는 신호에서 **동일 주기의 과거 관측값에 높은 가중치** 수렴
- 이를 통해 계절성, 주기성 등 반복 패턴이 있는 시계열에서 자연스럽게 일반화

### 3.6 일반화 향상을 위한 잠재적 개선 방향

논문이 향후 작업으로 제시:
- **공변량 특성 기반 가중치**: 현재는 오차 분포만 사용, 입력 컨텍스트 특성을 가중치 계산에 반영하면 국소적 일반화(local coverage) 향상 가능
- **더 정교한 지평선 집계**: 현재 중앙값 대신 가중 집계 또는 다른 p-value 결합 방법 탐색
- **컨텍스트 길이 최적화**: 현재 고정 52로 설정, 자동 컨텍스트 길이 선택으로 일반화 개선 가능

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

### 4.1 적응적 컨포멀 예측 연구와의 비교

| 방법 | 연도 | 적응 방식 | 비교환성 대응 | 이상탐지 적용 | 모든 α 보정 |
|------|------|-----------|--------------|--------------|------------|
| ACI (Gibbs & Candès) | 2021 | 분위수 업데이트 | ✓ | ✗ | ✗ (단일 α) |
| SAOCP (Zaffran et al.) | 2022 | 분위수 업데이트 | ✓ | ✗ | ✗ (단일 α) |
| CFOPI (Gibbs & Candès) | 2024 | 임의 분포 변화 | ✓ | ✗ | ✗ |
| Localized CP (Guan) | 2023 | 공변량 유사도 가중 | 부분적 | ✗ | ✗ |
| Neighborhood CP (Ghosh et al.) | 2023 | 이웃 가중 | 부분적 | ✗ | ✗ |
| **W1-ACAS (본 논문)** | **2026** | **W1 최소화 학습** | **✓** | **✓** | **✓** |

**핵심 차별점**: 기존 ACI/SAOCP 계열은 단일 오차율 $\alpha$에 최적화되는 반면, W1-ACAS는 모든 $\alpha \in [0,1]$에서 균일하게 보정된 p-value를 제공.

### 4.2 TSFM 기반 이상 탐지 연구와의 비교

| 방법 | 연도 | 파인튜닝 필요 | 해석가능 점수 | 분포 변화 적응 | 허위경보 제어 |
|------|------|-------------|--------------|--------------|-------------|
| MOMENT ZS (Goswami et al.) | 2024 | ✗ | ✗ | ✗ | ✗ |
| DeepAnT/CNN (Munir et al.) | 2018 | ✓ (semi-sup) | ✗ | ✗ | ✗ |
| OmniAnomaly (Su et al.) | 2019 | ✓ (semi-sup) | 부분적 | ✗ | ✗ |
| USAD (Audibert et al.) | 2020 | ✓ (semi-sup) | ✗ | ✗ | ✗ |
| **W1-ACAS** | **2026** | **✗** | **✓ (p-value)** | **✓ (온라인)** | **✓ (이론 보장)** |

**MOMENT ZS 대비**: MOMENT는 VUS-PR에서 0.461로 일부 경쟁력 있으나, CalErr=0.417로 임계값 선택이 불안정. W1-ACAS는 CalErr=0.025로 실용적 임계값 설정이 가능.

### 4.3 클래식 vs. 딥러닝 vs. W1-ACAS 비교

Liu & Paparrizos (2024) 벤치마크 결과와의 관계:
- **클래식 방법** (KShape, POLY, Sub-KNN 등): 전체 데이터셋 접근 필요 (비인과적), 스트리밍 불가. PA-F1 평균 0.47~0.54에 불과
- **딥러닝 반지도 방법**: 높은 PA-F1 달성 가능하나 CalErr 매우 높음 (CNN: 0.643), 즉 임계값 선택에 강한 도메인 지식 필요
- **W1-ACAS**: 인과적(causal), 스트리밍 가능, 임계값 직접 해석 가능, CalErr 최저 (0.025)

---

## 5. 미래 연구에 미치는 영향과 고려사항

### 5.1 미래 연구에 미치는 영향

#### 5.1.1 패러다임 전환: Post-Hoc 적응 컨포멀 프레임워크

이 논문은 **"TSFM 예측 + 컨포멀 보정 + 온라인 적응"** 의 결합 패러다임을 제시한다. 이는 다음 분야에 영향을 미칠 것으로 예상된다:

1. **산업 IoT 모니터링**: 데이터 부족 환경에서 즉각적 배포 가능한 프레임워크의 표준으로 자리잡을 가능성
2. **의료 신호 모니터링 (ECG, EEG)**: LTDB 실험 결과(W1-ACAS-H PA-F1=0.937)가 보여주듯 생체신호 이상 탐지에 유망
3. **금융 이상 탐지**: STOCK 데이터셋에서 거의 완벽한 성능 (AUC-PR=0.973) 시연

#### 5.1.2 컨포멀 예측 연구에의 기여

- **다목적(multi-level) 보정 목적 함수**: W1 거리를 통한 균일 보정이라는 새로운 목적 함수 제시 → 다른 보정 문제(회귀, 분류)에도 적용 가능성
- **비적합도 점수의 p-value 변환**: 임의의 이상 점수를 직접 해석 가능한 p-value로 변환하는 방법론적 기여

#### 5.1.3 TSFMs 생태계 연구 가속화

파운데이션 모델과 통계적 보장의 결합이라는 관점에서, 이 연구는 다음을 촉진할 것으로 예상:
- TSFM의 불확실성 정량화 연구 (현재 Chronos, TTM 등의 예측 불확실성 활용 방안)
- 더 나은 비적합도 점수 설계 (단순 절대 오차 이상의 스코어)
- 멀티모달 시계열 파운데이션 모델의 이상 탐지 적용

### 5.2 앞으로 연구 시 고려할 점

#### 5.2.1 방법론적 개선 방향

**① 컨텍스트 특성 기반 가중치 학습**

현재 가중치는 오직 비적합도 점수의 분포만 기반으로 학습된다. 향후:
- 입력 공변량 $X_t$의 특성(계절성, 추세, 스펙트럼 특성 등)을 가중치 학습에 통합
- 예: $w_i \propto \exp(-d_{\text{covariate}}(X_i, X_{t+1}) / \tau)$ 형태의 결합

**② 이상 오염 데이터 처리**

현재 보정 데이터에 이상치가 일부 포함된 경우 $\beta_\mathbf{w}(S_{t+1}) = \alpha + \alpha'$ (각주 6)로 처리하지만, 실제 환경에서 이상치 비율 $\alpha'$의 추정이 어려움. 강건 추정법 도입 필요.

**③ 비적합도 점수 설계 확장**

현재 $S^d_{t+1} = |Y_{t+1} - \hat{Y}^d_{t+1}|$ 외에도:
- 확률론적 예측 TSFM (Chronos 등)의 **분포 기반 비적합도 점수** (CRPS, NLL 등) 활용
- 다변량 상관관계를 반영한 **Mahalanobis 거리** 기반 점수

**④ 지평선 집계 방법론 심화**

현재 단순 중앙값 집계는 지평선 간 상관관계를 무시:
- 지평선별 상관관계를 반영한 **가중 집계** 또는 **계층적 p-value 결합** 연구 필요
- 이상의 지속 시간(duration)에 따른 최적 지평선 자동 선택

#### 5.2.2 평가 및 벤치마크 관련 고려사항

**① 임계값 독립 지표에서의 성능 개선 필요**

AUC-PR, VUS-PR에서 W1-ACAS는 MOMENT ZS 대비 열세인 경우 존재 (전체 평균 AUC-PR: W1-ACAS 0.344~0.355 vs MOMENT ZS 0.461). 이는 **임계값 독립적 순위(ranking) 능력** 강화 연구 필요를 시사.

**② 공정한 비교를 위한 프로토콜 정립**

논문에서도 언급하듯 기존 클래식/딥러닝 방법들은 전체 데이터셋에 접근 가능한 반면 W1-ACAS는 인과적(causal). 향후 벤치마크에서 **인과적/비인과적 설정을 분리하여 평가**하는 프로토콜 필요.

**③ 레이블 없는 테스트 환경에서의 임계값 선택**

현재 논문은 oracle 전략으로 최적 임계값 선택. 실제 배포 환경에서 레이블 없이 $\alpha_c$를 자동 설정하는 방법 연구 필요.

#### 5.2.3 계산 효율성 및 확장성

- 현재 15-step 예측 기준 **0.025 ± 0.012초/샘플/특성** (V100 GPU)
- 특성 수가 많은 고차원 다변량 시계열 (수백~수천 특성)에서 병렬화 구현 및 계산 복잡도 분석 필요
- 엣지 디바이스(edge device) 배포를 위한 경량화 연구 (현재 프레임워크는 GPU 의존)

#### 5.2.4 이론적 강화

- 현재 상계는 $d_{TV}(\mathbf{s}, \mathbf{s}^i)$에 의존하지만 실제 계산이 어려움. **Wasserstein 거리와 total variation 거리의 관계**를 통한 더 tight한 경계 도출 연구
- 학습된 가중치 $\mathbf{w}$의 수렴 속도와 최적성에 대한 이론적 분석
- 다변량 p-value 결합의 **가족별 오류율(FWER)** 또는 **FDR(False Discovery Rate)** 제어 보장 연구

#### 5.2.5 실용적 배포 고려사항

- **콜드 스타트(Cold Start) 문제**: $n_c = 1/\alpha_c - 1$개의 정상 샘플 수집 전까지의 탐지 전략
- **개념 드리프트(Concept Drift) 탐지와의 결합**: 분포 변화 감지 → 가중치 리셋 메커니즘
- **도메인 특화 비적합도 함수 설계**: 산업별(제조, 에너지, 의료 등) 맞춤 비적합도 함수와의 결합

---

> **⚠️ 주의**: 본 답변은 제공된 논문 원문(arXiv:2604.20122v1)에 기반하여 작성되었으며, 논문에 명시된 내용만을 인용하였습니다. "2020년 이후 관련 최신 연구 비교 분석" 섹션의 일부 표 내용(비교 방법들의 특성 분류)은 논문의 관련 연구 섹션과 실험 결과를 바탕으로 정리한 것이며, 논문에서 직접 이 형태로 제시하지 않은 내용은 논문의 실험 결과와 인용 맥락을 통해 추론하였음을 밝힙니다.
