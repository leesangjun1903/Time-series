# Multivariate Bayesian Structural Time Series Model 

---

## 1. Executive Summary (10문장 이내)

본 논문은 Jammalamadaka, Qiu, Ning (2018)이 제안한 **Multivariate Bayesian Structural Time Series (MBSTS)** 모델로, 기존 단변량 BSTS 모델을 다변량으로 확장한 연구이다.  
모델은 상태공간(State Space) 형태로 구성되며, **추세(Trend), 계절성(Seasonality), 순환(Cycle), 회귀(Regression)** 성분을 결합한다.  
Bayesian 패러다임과 MCMC 알고리즘을 통해 변수 선택(Variable Selection)과 모델 학습을 동시에 수행하여 과적합(Overfitting)을 방지한다.  
Spike and Slab 사전분포를 활용한 특성 선택(Feature Selection)과 Bayesian Model Averaging(BMA)이 핵심 방법론이다.  
순환 성분은 외부 충격(External Shock)에 의한 단기 변동을 감쇠 인자(Damping Factor)로 포착한다.  
7가지 시뮬레이션 모델 및 금융 실증 데이터(BOA, COF, JPM, WFC, 2006–2017)에 적용하여 성능을 검증하였다.  
비교 모델인 BSTS, ARIMAX, MARIMAX 대비 누적 일보 예측 오차(Cumulative One-Step-Ahead Prediction Error) 기준으로 MBSTS가 일관되게 우수하다.  
다변량 목표 시계열 간 상관관계가 높을수록 MBSTS의 성능 우위가 더욱 두드러진다.  
MCMC의 높은 연산 비용과 비가우시안 관측치 처리 미지원이 한계로 남는다.

---

### 1-1. 연구의 목적과 필요성

| 항목 | 내용 |
|------|------|
| **배경** | 경제 거래 데이터의 폭발적 증가로 다변량 의존 시계열의 동시 분석 필요성 증대 (p.1) |
| **문제** | 기존 BSTS(Scott & Varian, 2014)는 단변량 대상으로, 다중 목표 시계열 간 상관관계를 무시 |
| **목적** | 다중 상관 목표 시계열에 대해 특성 선택·예측·추론을 동시 수행하는 통합 Bayesian 프레임워크 구축 |
| **필요성** | 금융위기 등 외부 충격에 의한 단기 변동 포착, 과적합 방지, 공통 예측변수 활용의 유연성 확보 |

---

## 2. 핵심 주장과 근거 표

| 핵심 주장 | 방법론적 근거 | 실증 근거 | 위치 |
|-----------|-------------|-----------|------|
| 다변량 상관 포착으로 예측력 향상 | 관측 오차 공분산 $\Sigma_t$ 공동 모델링 (Eq.1) | Figure 6: $\rho$ 증가 시 MBSTS 우위 확대 | p.3, p.20 |
| Spike-Slab으로 과적합 방지 및 변수 선택 | $\gamma_{ij} \in \{0,1\}$ 사전분포 (Eq.12) | Figure 4(b): 포함 확률이 실제 생성 변수 정확 식별 | p.7, p.17 |
| 순환 성분으로 외부 충격 포착 | 감쇠 인자 $\varrho_{ii} \in (0,1)$ (Eq.8) | Figure 9(b): 2008–2012 금융위기 충격 반영 | p.5, p.26 |
| BMA로 임의적 변수 선택 회피 | 사후 예측 분포 적분 (Eq.31) | Figure 11: 원본 예측변수 사용 시 최저 오차 | p.11, p.28 |
| BSTS·ARIMAX·MARIMAX 대비 우월 | 성장 창문(Growing Window) 검증 | Figure 5, 6, 11 | p.18–21, p.28 |

---

## 2-1. 상세 설명

### 해결하고자 하는 문제

- **단변량 BSTS의 한계**: 각 시계열을 독립적으로 모델링하여 시계열 간 공분산 정보 손실
- **ARIMAX 계열의 한계**: 시계열 구조적 성분(추세·순환) 부재, Bayesian 변수 선택 미지원
- **고차원 예측변수 문제**: 후보 예측변수 풀에서 관련 변수만 자동 선택하는 메커니즘 필요

---

### 제안하는 방법 (수식 포함)

**기본 상태공간 모델** (p.3, Eq.1–3):

$$\tilde{y}_t = Z_t^T \alpha_t + \tilde{\epsilon}_t, \quad \tilde{\epsilon}_t \sim N_m(0, \Sigma_t) $$

$$\alpha_{t+1} = T_t \alpha_t + R_t \eta_t, \quad \eta_t \sim N_q(0, Q_t) $$

**일반 분해 모형** (p.3, Eq.4):

$$\tilde{y}_t = \tilde{\mu}_t + \tilde{\tau}_t + \tilde{\omega}_t + \tilde{\xi}_t + \tilde{\epsilon}_t, \quad \tilde{\epsilon}_t \overset{iid}{\sim} N_m(0, \Sigma_\epsilon) $$

**추세 성분** (p.4, Eq.5–6):

$$\tilde{\mu}_{t+1} = \tilde{\mu}_t + \tilde{\delta}_t + \tilde{u}_t, \quad \tilde{u}_t \overset{iid}{\sim} N_m(0, \Sigma_\mu) $$

$$\tilde{\delta}_{t+1} = \tilde{D} + \tilde{\rho}(\tilde{\delta}_t - \tilde{D}) + \tilde{v}_t, \quad \tilde{v}_t \overset{iid}{\sim} N_m(0, \Sigma_\delta) $$

**계절 성분** (p.4, Eq.7):

$$\tau_{t+1}^{(i)} = -\sum_{k=0}^{S_i - 2} \tau_{t-k}^{(i)} + w_t^{(i)}, \quad \tilde{w}_t \overset{iid}{\sim} N_m(0, \Sigma_\tau) $$

**순환 성분** (p.5, Eq.8):

$$\tilde{\omega}_{t+1} = \widehat{\varrho\cos(\lambda)}\tilde{\omega}_t + \widehat{\varrho\sin(\lambda)}\tilde{\omega}_t^* + \tilde{\kappa}_t, \quad \tilde{\kappa}_t \overset{iid}{\sim} N_m(0, \Sigma_\omega) $$

```math
\tilde{\omega}_{t+1}^* = -\widehat{\varrho\sin(\lambda)}\tilde{\omega}_t + \widehat{\varrho\cos(\lambda)}\tilde{\omega}_t^* + \tilde{\kappa}_t^*, \quad \tilde{\kappa}_t^* \overset{iid}{\sim} N_m(0, \Sigma_\omega)
```

여기서 $\varrho_{ii} \in (0,1)$은 감쇠 인자, $\lambda_{ii} = 2\pi/q_i$는 주파수

**회귀 성분** (p.5, Eq.9):

$$\xi_t^{(i)} = \beta_i^T x_t^{(i)} $$

**Spike Prior** (p.7, Eq.12):

$$\gamma \sim \prod_{i=1}^{m} \prod_{j=1}^{k_i} \pi_{ij}^{\gamma_{ij}} (1-\pi_{ij})^{1-\gamma_{ij}} $$

**Slab Prior** (p.7, Eq.13):

$$\beta|\gamma \sim N_K(b_\gamma, A_\gamma^{-1}), \quad \Sigma_\epsilon|\gamma \sim IW(v_0, V_0) $$

**사후 분포** (p.8, Eq.22, 27):

$$\beta|\hat{Y}^*, \Sigma_\epsilon, \gamma \sim N_K\!\left(\tilde{\beta}_\gamma, (\hat{X}_\gamma^T \hat{X}_\gamma + A_\gamma)^{-1}\right) $$

$$\Sigma_\epsilon|\tilde{Y}^*, \beta, \gamma \sim IW(v_0 + n,\ E_\gamma^T E_\gamma + V_0) $$

**사후 예측 분포** (p.11, Eq.31):

$$p(\hat{Y}|Y) = \int p(\hat{Y}|\tilde{\psi}) p(\tilde{\psi}|Y) d\tilde{\psi} $$

---

### 모델 구조

```
MBSTS 모델 구조
├── 관측 방정식 (Eq.1): ỹ_t = Z_t^T α_t + ε_t
├── 상태 전이 방정식 (Eq.2): α_{t+1} = T_t α_t + R_t η_t
└── 상태 성분 α_t = [μ̃_t^T, τ̃_t^T, ω̃_t^T, ξ̃_t^T]^T
    ├── 추세(Trend): 일반화 국소 선형 추세 (Eq.5–6)
    ├── 계절(Season): 다중 계절 주기 허용 (Eq.7)
    ├── 순환(Cycle): 감쇠 인자 포함 (Eq.8)
    └── 회귀(Regression): Spike-Slab 변수 선택 (Eq.9, 12–13)

추론 알고리즘 (Algorithm 1, p.11):
1. Kalman Filter/Smoother → α 샘플링
2. 역 Wishart → θ 샘플링
3. SSVS → γ 샘플링 (변수 선택)
4. 다변량 정규 → β 샘플링
5. 역 Wishart → Σ_ε 샘플링
```

---

### 성능 향상 및 한계

| 구분 | 내용 |
|------|------|
| **성능 향상** | 상관관계 높은 다변량 시계열에서 ARIMAX/MARIMAX/BSTS 대비 누적 예측 오차 명확히 낮음 (Figure 5, 6, 11) |
| **성능 향상** | 순환 성분으로 금융위기 충격 포착 (Figure 9b) |
| **성능 향상** | 자동 변수 선택으로 희소성 확보, 모델 크기 축소 (Figure 10, p.28) |
| **한계 ①** | MCMC 반복에 따른 높은 연산 비용 (p.31) |
| **한계 ②** | 목표 시계열 간 독립에 가까울 경우 BSTS 대비 우위 미미 (p.31) |
| **한계 ③** | 비가우시안(Non-Gaussian) 관측치 처리 미지원 (p.31) |
| **한계 ④** | 모델 구조 자체가 분석 기간 동안 불변임을 가정 (p.18) |
| **한계 ⑤** | 회귀 계수가 정적(Static)으로 고정됨 (p.5) |

---

## 3. 각 주장에 페이지/Figure·Table 번호 표시

| 주장 | 근거 위치 |
|------|-----------|
| MBSTS가 BSTS/ARIMAX/MARIMAX 대비 우월 | Figure 5 (p.19), Figure 6 (p.20), Figure 11 (p.29) |
| 상관관계 강할수록 MBSTS 우위 확대 | Figure 6(d)(f), p.21 |
| 순환 성분의 외부 충격 포착 | Figure 9(b), p.25–26 |
| Spike-Slab 변수 선택 정확성 | Figure 4(b), p.17–18 |
| 표본 크기 증가 시 추정 오차 감소 | Figure 2, Figure 3, p.15–16 |
| 90% 신용구간 포함률 확인 | Figure 4(a), p.17 |
| 실증 데이터 특성 선택 결과 | Figure 10, Table 1, Table 2, p.27–28 |
| 사후 예측 분포 기반 거래 전략 | Figure 12, p.29 |

---

## 4. 저자 보고 결과 vs. 해석 분리

### 연구 주제
- **저자 보고**: "다중 상관 시계열에 대한 추론과 예측을 위한 MBSTS 모델 제안" (Abstract, p.1)
- **검토자 해석**: 기존 BSTS의 단변량 한계를 구조적으로 극복하려는 시도로, 상태공간 모델의 다변량 확장과 Bayesian 변수 선택의 통합이라는 측면에서 방법론적 기여가 명확함

### 방법
- **저자 보고**: Kalman Filter + Spike-Slab + BMA의 5단계 MCMC 알고리즘 (Algorithm 1, p.11)
- **검토자 해석**: 각 단계의 조건부 켤레성(Conditional Conjugacy)을 활용한 설계로, 계산 효율성을 확보하면서도 비켤레 사전분포를 다루는 점에서 실용적 기여가 있음. 다만, 번-인(Burn-in) 200 샘플 결정이 시행착오에 의한 것임을 저자가 명시(p.15)

### 결과
- **저자 보고**: "시뮬레이션 및 실증 모두에서 MBSTS가 세 벤치마크 모델 대비 우수" (p.2, p.30)
- **검토자 해석**: 누적 예측 오차(Cumulative Absolute Error) 단일 지표 기반 비교로, RMSE, MAPE 등 다양한 평가 지표를 사용하지 않아 성능 우위의 일반성 주장에 제한이 있음

---

## 5. 통계적 취약점 및 비교 불가능한 수치

> ⚠️ 표시: 통계적으로 취약하거나 직접 비교가 어려운 항목

| 항목 | 문제점 |
|------|--------|
| ⚠️ **번-인 200 샘플** | "trial and error"에 의해 결정 (p.15) — 수렴 진단 기준(Gelman-Rubin 등) 미제시 |
| ⚠️ **누적 예측 오차만 사용** | RMSE, MAE, MAPE, CRPS 등 표준 지표 미사용 — 절대 수치 비교 불가 |
| ⚠️ **단일 실증 데이터셋** | 금융 데이터 1개(2006–2017)만 사용 — 도메인 일반화 불확실 |
| ⚠️ **시뮬레이션 반복 횟수 미명시** | 각 모델별 독립 시뮬레이션 횟수 불명확 — 변동성 추정 불안 |
| ⚠️ **MCMC 수렴 진단 부재** | 트레이스 플롯, 자기상관 함수 등 수렴 진단 결과 미제공 |
| ⚠️ **40% 예측 구간 선택 근거** | Figure 12의 40% 구간 선택 이유 미설명 (p.29) |
| ⚠️ **감쇠 인자 0.95 선택** | 교차검증으로 결정했다고 하나, 검증 기준 상세 미제시 (p.25) |
| ⚠️ **비계절화 처리 비교** | 원본 vs 계절 제거 예측변수 비교가 특정 기간에 한정 (Figure 11) |

---

## 6. 문서가 답하지 않는 질문

1. **MCMC 수렴 진단**: 체인이 실제로 수렴했는지에 대한 정량적 진단 결과는?
2. **비가우시안 확장**: 카운트 데이터, 이진 데이터 등 비가우시안 관측치에 어떻게 대응하는가?
3. **동적 회귀 계수**: 정적 회귀 계수($\beta_i$) 가정이 현실적으로 유효한 조건은?
4. **스케일 민감성**: 목표 시계열의 스케일 차이가 공분산 행렬 추정에 미치는 영향?
5. **모델 성분 선택 자동화**: 각 시계열에 어떤 성분을 포함할지 자동 결정하는 방법?
6. **대규모 $m$에서의 확장성**: 목표 시계열 차원 $m$이 매우 클 때 역 Wishart 추정의 안정성?
7. **시계열 길이 불균형**: 목표 시계열마다 관측 구간이 다를 때의 처리 방법?
8. **구조 변화(Structural Break)**: 모델 구조 자체가 바뀌는 경우 감지 및 대응 방법?
9. **다중 예측 수평선**: 일보 예측 외에 다단계(Multi-step) 예측 성능은?
10. **최적 MCMC 샘플 수**: 2,000 샘플 + 200 번-인의 적정성에 대한 이론적 근거?

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p.6) — 시뮬레이션 시계열 성분
**설명**: 추세(파란 선), 계절(빨간 점선), 순환(검은 선) 성분이 어떻게 개별적으로 생성되는지 시각화
**핵심 해석**: 세 성분의 시간적 패턴이 명확히 구별되며, 순환 성분은 감쇠 인자($\varrho = 0.97$)에 의해 시간이 지남에 따라 진폭이 줄어드는 특성이 뚜렷하게 나타남. 이는 외부 충격의 영향이 점진적으로 소멸하는 현실적 경제 현상을 반영함

---

### Figure 5 (p.19) — 6개 시뮬레이션 모델 누적 예측 오차 비교
**설명**: MBSTS, BSTS, ARIMAX, MARIMAX의 누적 일보 예측 오차를 6개 시뮬레이션 데이터셋에서 비교
**핵심 해석**:
- **Model 1–2** (추세만): MBSTS가 명확한 우위 없음 → 단순 추세 시계열에서는 성분 풍부성의 이점이 제한적
- **Model 3–4** (계절/순환 포함): MBSTS가 뚜렷하게 우수 → 복잡한 구조적 성분 포착 능력 확인
- **Model 5–6** (다차원): BSTS/ARIMAX 대비 MBSTS·MARIMAX의 다변량 우위 확인

⚠️ **통계적 주의**: 단일 시뮬레이션 데이터셋으로 비교하여 결과의 변동성(Variability) 미제시

---

### Figure 6 (p.20) — 다양한 상관계수에서의 누적 예측 오차
**설명**: $\rho \in \{0, 0.2, -0.3, 0.5, -0.6, 0.8\}$에서 네 모델 성능 비교

**핵심 해석**: 논문에서 가장 중요한 실험 결과 중 하나
$$\text{성능 격차} \propto |\rho|$$
상관관계의 절대값이 클수록 MBSTS의 우위가 명확히 증가함. $\rho = 0$일 때 MBSTS와 BSTS의 격차가 가장 작으며, $\rho = 0.8$ 또는 $\rho = -0.6$일 때 가장 큼. 이는 다변량 공동 모델링의 이론적 근거를 실험적으로 지지함

---

### Figure 9 (p.26) — BOA 주가의 상태 성분 분해
**설명**: BOA 최대 로그 수익률을 추세, 순환, 회귀 성분으로 분해 (90% 신용 구간 포함)

**핵심 해석**:
- **추세 성분(a)**: 2009년 정점 후 점진적 감소 → 금융위기 후 장기 평균 회복 과정 반영
- **순환 성분(b)**: 2008–2012년 구간에서 높은 진폭 → 서브프라임 모기지 위기의 충격이 감쇠하며 소멸하는 패턴 명확. 이는 순환 성분의 감쇠 인자($\varrho = 0.95$)가 적절히 작동함을 보여줌
- **회귀 성분(c)**: 주기적이지만 피크 없음 → 거시경제 지표 및 기술적 지표의 국소적 기여를 반영

---

### Figure 10 (p.27) — 경험적 사후 포함 확률 (4개 금융사)
**설명**: 각 금융사의 예측변수별 MCMC 포함 확률(Empirical Posterior Inclusion Probability)

**핵심 해석**:
- **MFI, EMV, CLV**: 4개 회사 모두에서 포함 확률 1 또는 근접 → 기술적 지표의 범용적 중요성
- **구글 추세**: 회사별로 다른 지표 선택 ("mobile/constr/comput" for BOA; "unempl/rental" for COF)
- 각 회사에 대한 최종 모델 크기가 전체 후보 예측변수 수(35개)보다 훨씬 작음 → 희소성 확보
- ⚠️ 포함 확률이 정확히 1.0으로 표시된 변수들은 MCMC 전체 반복에서 항상 선택됨을 의미하나, 이는 해당 변수의 진짜 중요성인지 모델 구조의 산물인지 추가 검증 필요

---

## 8. 결론, 시사점 및 후속 연구

### 저자 제시 시사점 (p.30–31)

1. MBSTS는 다중 상관 시계열 예측 실무에서 BSTS·ARIMAX·MARIMAX의 실용적 대안
2. 다변량 시계열 간 상관관계가 강할수록 MBSTS 채택이 특히 권고됨
3. 모델 적용 전 데이터 탐색과 전문가 지식 활용이 성분 선택에 중요

### 저자 제시 후속 연구 (p.31)

| 항목 | 내용 |
|------|------|
| **미해결 문제 ①** | 사전 정보(모델 크기, 계수 추정치)가 예측 성능에 미치는 영향 분석 |
| **미해결 문제 ②** | 비가우시안(Non-Gaussian) 관측치에 대한 모델 확장 |

---

### 8-1. 모델의 일반화 성능 향상 가능성

MBSTS의 일반화 성능 향상을 위한 방향은 다음과 같다:

#### (1) 구조적 한계 극복

**정적 회귀 계수의 동적 확장**:

현재 모델의 회귀 성분은 고정 계수 $\beta_i$를 사용하나, 시간 변화 계수로 확장 시:

$$\xi_t^{(i)} = \beta_{i,t}^T x_t^{(i)}, \quad \beta_{i,t+1} = \beta_{i,t} + \nu_t, \quad \nu_t \sim N(0, \Sigma_\beta) \tag{확장}$$

이를 통해 비정상(Non-stationary) 회귀 관계를 포착하고, 구조 변화에 강건한 모델 구성이 가능함

#### (2) 공분산 구조의 유연화

현재 $\Sigma_\epsilon$를 전체 기간 상수로 가정하나, 시간 변화 공분산으로 확장:

$$\Sigma_t = f(\text{GARCH}(\cdot)) \text{ 또는 } \Sigma_t \sim IW(\cdot) \text{ 동적 사전분포}$$

이는 금융 데이터의 변동성 군집(Volatility Clustering) 현상을 더 잘 반영함

#### (3) 비가우시안 관측치 처리

Count 데이터나 이진 데이터에 대한 일반화를 위한 확장:
- **포아송 관측 모델**: $y_t \sim \text{Poisson}(\exp(Z_t^T \alpha_t))$
- **음이항 모델**: 과분산(Overdispersion) 처리
- 보조 변수(Auxiliary Variable) 기법으로 MCMC 구조 유지 가능

#### (4) 온라인 학습 가능성

현재 Growing Window 방식은 전체 재학습을 요구하므로, Sequential Monte Carlo(SMC) 또는 Particle Filter 기반의 온라인 갱신으로 계산 효율성 개선이 가능함

#### (5) 대규모 차원에서의 희소 공분산

$m$이 클 때 역 Wishart의 차원의 저주(Curse of Dimensionality) 문제 해결을 위한 희소 Graphical Model 기반 $\Sigma_\epsilon$ 추정:

$$\Sigma_\epsilon^{-1} = \Omega \sim \text{Graphical Lasso Prior}$$

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> **주의**: 아래는 2020년 이후 관련 연구 흐름에 대한 일반적 분석이며, 특정 논문의 세부 수치는 직접 확인이 필요합니다. 확실하지 않은 구체적 수치는 제시하지 않습니다.

#### 최신 연구 흐름과의 비교

| 연구 방향 | MBSTS와의 관계 | 시사점 |
|-----------|---------------|--------|
| **Deep State Space Models** (예: DeepAR, LSTNet) | MBSTS는 해석 가능성(Interpretability) 우위; 딥러닝 모델은 비선형 패턴 포착 강점 | Hybrid 접근 가능성 |
| **Variational Inference 기반 베이즈** | MCMC 대비 계산 효율성 향상 | MBSTS의 MCMC 병목 해소 가능 |
| **Transformer 기반 시계열 예측** (예: Temporal Fusion Transformer) | MBSTS는 소표본에서, Transformer는 대용량 데이터에서 강점 | 데이터 크기에 따른 모델 선택 기준 필요 |
| **Sparse Bayesian VAR** | 목표 시계열 간 Granger 인과성 모델링 강점 | MBSTS의 회귀 성분과 통합 가능 |
| **베이즈 신경망 시계열** | 불확실성 정량화(Uncertainty Quantification) 공통 강점 | MBSTS의 신용구간 해석과 비교 연구 필요 |

#### 이 논문이 향후 연구에 미치는 영향

1. **다변량 Bayesian 시계열의 기준선(Baseline) 제공**: 구조적 성분 분해 + 변수 선택 통합 프레임워크는 후속 연구의 출발점이 됨

2. **금융 분야 적용 템플릿**: 포트폴리오 시계열의 공동 모델링 방법론으로 실무 적용 가능성 제시

3. **Spike-Slab의 다변량 확장 방향**: 단변량 BSTS의 Spike-Slab을 다변량으로 확장하는 방식이 후속 연구 (비가우시안 확장, 동적 계수 등)의 기초가 됨

#### 향후 연구 시 고려할 점

| 고려사항 | 구체적 내용 |
|----------|------------|
| **계산 효율성** | MCMC 대신 Variational Bayes 또는 INLA 적용으로 대규모 $m$ 처리 |
| **비정상성 처리** | 공적분(Cointegration) 관계를 가진 시계열에 대한 MBSTS 확장 필요 |
| **모델 불확실성** | 성분 선택(추세/순환 포함 여부) 자체를 Bayesian 방식으로 처리 |
| **평가 지표 다양화** | CRPS(Continuous Ranked Probability Score), WIS 등 확률 예측 지표 추가 |
| **재현성** | 공개 코드/데이터 부재 — 재현 가능한 연구(Reproducible Research) 기준 충족 필요 |
| **외부 충격의 내생적 감지** | 감쇠 인자와 순환 주기를 사전 고정하지 않고 데이터로부터 자동 추론하는 방법 |

---

## 참고 자료

**본 분석의 주요 참고 문헌 (논문 내 인용 기준)**:

1. **Jammalamadaka, S.R., Qiu, J., Ning, N.** (2018). *Multivariate Bayesian Structural Time Series Model*. arXiv:1801.03222v2 [stat.ML]
2. **Scott, S.L. & Varian, H.R.** (2014). Predicting the present with Bayesian structural time series. *International Journal of Mathematical Modelling and Numerical Optimisation*, 5(1-2):4–23
3. **Scott, S.L. & Varian, H.R.** (2015). Bayesian variable selection for nowcasting economic time series. In *Economic analysis of the digital economy*, pp.119–135, University of Chicago Press
4. **George, E.I. & McCulloch, R.E.** (1997). Approaches for Bayesian variable selection. *Statistica Sinica*, pp.339–373
5. **Hoeting, J.A., Madigan, D., Raftery, A.E., & Volinsky, C.T.** (1999). Bayesian model averaging: a tutorial. *Statistical Science*, pp.382–401
6. **Harvey, A.C.** (1990). *Forecasting, structural time series models and the Kalman filter*. Cambridge University Press
7. **Durbin, J. & Koopman, S.J.** (2002). A simple and efficient simulation smoother for state space time series analysis. *Biometrika*, 89(3):603–616
8. **Harvey, A.C., Trimbur, T.M., & Van Dijk, H.K.** (2007). Trends and cycles in economic time series: A Bayesian approach. *Journal of Econometrics*, 140(2):618–649
9. **Brodersen, K.H. et al.** (2015). Inferring causal impact using Bayesian structural time-series models. *The Annals of Applied Statistics*, 9(1):247–274
10. **Rossi, P.E., Allenby, G.M., & McCulloch, R.** (2012). *Bayesian statistics and marketing*. John Wiley & Sons
