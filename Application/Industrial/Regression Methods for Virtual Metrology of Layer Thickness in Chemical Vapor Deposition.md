# Regression Methods for Virtual Metrology of Layer Thickness in Chemical Vapor Deposition

> **Purwins et al. (2014), IEEE/ASME Transactions on Mechatronics, Vol. 19, No. 1, pp. 1–8**

---

## 1. Executive Summary (10문장 이내)

이 논문은 반도체 제조 공정에서 물리적 측정 없이 웨이퍼의 실리콘 질화막($Si_3N_4$) 두께를 예측하는 **가상 계측(Virtual Metrology, VM)** 시스템 구축을 목표로 한다.  
PECVD(Plasma Enhanced Chemical Vapor Deposition) 공정의 FDC(Fault Detection and Classification) 데이터를 예측 변수로 활용한다.  
총 5가지 회귀 방법(SLR, MLR, PLR, RLR, SVR)과 3가지 변수 집합(TTB, ES, FF)의 조합을 비교 평가하였다.  
훈련 세트에서는 PLR/RLR이 최고 성능을 보였으나, 테스트 세트에서는 SVR이 압도적으로 우수한 일반화 성능을 나타냈다.  
특히 장비 유지보수 이벤트 이후와 같이 훈련 범위를 벗어난 조건에서 SVR의 강건성이 두드러졌다.  
전문가가 선정한 변수 집합(ES, 17개 변수)이 전체 변수(FF, 41개)보다 좋은 성능을 보였으며, 단일 변수(SLR)보다 다변량 방법이 월등히 우수하였다.  
본 연구는 데이터 전처리, ANOVA 기반 문맥 변수 분석, 그리드 탐색을 통한 하이퍼파라미터 최적화를 체계적으로 수행하였다.  
결론적으로 SVR은 공정 지식 없이도 강건한 예측을 제공하며, 전문가 지식을 결합하면 성능이 더욱 향상된다.

> 💡 **용어 설명**
> - **Virtual Metrology (VM)**: 물리적 측정 장비 없이 공정 데이터만으로 품질 지표를 예측하는 기술
> - **PECVD**: 플라즈마를 이용하여 낮은 온도에서도 박막을 증착할 수 있는 화학기상증착 방법
> - **FDC**: 장비 이상 감지 및 분류를 위한 센서/컨텍스트 데이터 수집 시스템

---

### 1-1. 연구의 목적과 필요성

| 구분 | 내용 |
|------|------|
| **문제** | 웨이퍼 품질 측정(물리적 계측)은 비용이 높고 샘플링 주기가 길어 공정 이상을 즉시 감지하기 어려움 |
| **필요성** | 공정 drift/shift가 측정 간격 사이에 발생하면 불량 웨이퍼가 다수 생산될 수 있음 |
| **목적** | FDC 데이터 기반의 VM으로 $Si_3N_4$ 두께를 실시간 예측하여 R2R 제어 및 공정 모니터링에 활용 |
| **기대 효과** | 측정 비용 절감, 공정 이상 조기 탐지, 생산 수율 향상 |

> 💡 **용어 설명**
> - **Run-to-Run (R2R) Controller**: 각 웨이퍼 처리(런) 결과를 피드백으로 받아 다음 런의 공정 파라미터를 자동 조정하는 제어 시스템
> - **공정 drift/shift**: 시간 경과 또는 유지보수에 의해 공정 조건이 점진적으로(drift) 또는 갑작스럽게(shift) 변화하는 현상

---

## 2. 핵심 주장과 근거 표

| # | 핵심 주장 | 근거 | 위치 |
|---|-----------|------|------|
| 1 | SVR이 테스트 세트에서 선형 방법 대비 압도적 우위 | SVR Rel. RMSE = 0.432, RLR = 0.749 (73% 차이) | Table III, p.6 |
| 2 | 선형 방법들은 훈련 세트에 과적합 | 훈련→테스트 오차 증가: RLR +132%, MLR +338%, SVR +27% | p.7 |
| 3 | 전문가 선정 변수(ES)가 최적 변수 집합 | ES > FF > TTB 순서로 성능 우위 | Table II, p.6 |
| 4 | MLR은 다중공선성 문제로 불안정 | TTB-MLR의 Rel. Std = 2.54 (다른 방법의 수백 배) | Table II, p.6 |
| 5 | 차원 축소(PLS)가 선형 회귀 성능 개선 | PLR/RLR이 MLR 대비 일관되게 우수 | p.6 |
| 6 | 유지보수 이벤트가 훈련 범위 이탈을 유발 | Instance No.29 이후 SLR/MLR/PLR/RLR 성능 급락 | Fig.3, p.7 |
| 7 | 단변량 회귀(SLR)는 다변량 방법 대비 열등 | SLR CV Rel. RMSE = 0.895 (PLR의 2.3배) | Table II, p.6 |

> 💡 **용어 설명**
> - **다중공선성 (Multicollinearity)**: 예측 변수들 사이에 강한 상관관계가 존재하여 회귀 계수 추정이 불안정해지는 현상
> - **과적합 (Overfitting)**: 모델이 훈련 데이터에 지나치게 맞춰져 새로운 데이터에 대한 예측 성능이 저하되는 현상

---

### 2-1. 해결 문제, 제안 방법, 모델 구조, 성능 향상 및 한계

#### 🔴 해결하고자 하는 문제

- $Si_3N_4$ 박막 두께의 실시간·비파괴 예측
- 공정 조건 변화(유지보수, drift)에 강건한 예측 모델 구축
- 최적 변수 선택 및 회귀 방법 선정

---

#### 🔵 제안하는 방법 (수식 포함)

**① Multiple Linear Regression (MLR)**

$$y_i = b + w_1 x_{i1} + w_2 x_{i2} + \cdots + w_d x_{id} + n_i = \underbrace{b + \mathbf{w}^\top \mathbf{x}_i}_{\hat{y}} + n_i $$

손실 함수 (최소자승법):

$$l(y_i, \hat{y}_i) = (y_i - (b + \mathbf{w}^\top \mathbf{x}_i))^2 $$

회귀 계수 추정:

$$\hat{\mathbf{w}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y} $$

$$\hat{y} = \bar{b} + \hat{\mathbf{w}}^\top (\mathbf{z} - \bar{\mathbf{x}}) $$

> **기호 설명**
> - $y_i$: $i$번째 웨이퍼의 실측 $Si_3N_4$ 두께
> - $\mathbf{x}\_i = (x_{i1}, \ldots, x_{id})^\top \in \mathbb{R}^d$: $d$차원 FDC 예측 변수 벡터
> - $b$: 절편(intercept)
> - $\mathbf{w} = (w_1, \ldots, w_d)^\top$: 회귀 계수 벡터
> - $n_i$: 잡음(noise) 항
> - $\mathbf{X}$: 중심화(centering)된 예측 변수 행렬
> - $\mathbf{z}$: 새로운 예측 입력 벡터

---

**② Ridge Linear Regression (RLR)**

다중공선성 해결을 위해 정규화 항 추가:

$$\hat{\mathbf{w}} = (\mathbf{X}^\top \mathbf{X} + rI)^{-1} \mathbf{X}^\top \mathbf{y} $$

> **기호 설명**
> - $r$: 릿지 파라미터(하이퍼파라미터), 값이 클수록 계수를 0에 가깝게 수축
> - $I$: 단위 행렬(identity matrix)

> 💡 **릿지 회귀**: $(\mathbf{X}^\top\mathbf{X})$가 특이(singular)에 가까울 때 $rI$를 더해 역행렬 계산을 안정화하는 기법

---

**③ Partial Least Squares (PLS)**

최대 공분산 방향 계산:

$$\mathbf{v}_j = \frac{\mathbf{X}_j^\top \mathbf{y}_j}{\|\mathbf{X}_j^\top \mathbf{y}_j\|} $$

잠재 변수(score) 계산:

$$\mathbf{t}_j = \mathbf{X}_j \mathbf{v}_j $$

잔차 업데이트:

$$\mathbf{X}_{j+1} = \mathbf{X}_j - \hat{\mathbf{p}}_j \mathbf{t}_j^\top $$

$$\mathbf{y}_{j+1} = \mathbf{y}_j - \hat{c}_j \mathbf{t}_j $$

예측:

$$\hat{y} = \bar{y} + \hat{\mathbf{c}} \mathbf{t}^* $$

> **기호 설명**
> - $\mathbf{v}_j$: $j$번째 PLS 성분 방향 벡터
> - $\mathbf{t}_j$: $j$번째 잠재 변수(score)
> - $\hat{c}_j$: $\mathbf{t}_j$에 대한 회귀 계수
> - $\hat{\mathbf{p}}_j$: $\mathbf{X}_j$를 $\mathbf{t}_j$에 회귀한 계수 벡터
> - $g$: PLS 성분 수 (하이퍼파라미터)
> - $\mathbf{t}^*$: 새 입력 $\mathbf{z}$의 score 벡터

> 💡 **잠재 변수(Latent Variable)**: 원래 변수들의 선형 결합으로 만들어지는 새로운 저차원 변수로, 예측 변수와 타겟 변수 간의 공분산을 최대화하는 방향으로 추출됨

---

**④ Support Vector Regression (SVR)**

비선형 변환 도입:

$$y_i = b + \mathbf{w}^\top \varphi(\mathbf{x}_i) + n_i $$

$\varepsilon$-비민감 손실 함수:

$$l(y_i, \hat{y}_i) = \begin{cases} 0 & \text{if } |y_i - \hat{y}_i| \leq \varepsilon \\ |y_i - \hat{y}_i| - \varepsilon & \text{else} \end{cases} $$

슬랙 변수:

$$\xi_i := \min(y_i - \hat{y}_i - \varepsilon, 0) $$

$$\xi_i^* := \min(-y_i + \hat{y}_i - \varepsilon, 0) $$

최적화 목적 함수:

$$\min \frac{1}{2}\|\mathbf{w}\|^2 + C\left(\sum_{i=1}^n \xi_i + \sum_{i=1}^n \xi_i^*\right) $$

쌍대 문제 (Dual Problem):

```math
L = -\frac{1}{2}\sum_{i,j=1}^n (\alpha_i - \alpha_i^*)(\alpha_j - \alpha_j^*)\varphi(\mathbf{x}_i)^\top\varphi(\mathbf{x}_j) - \varepsilon\sum_{i=1}^n(\alpha_i + \alpha_i^*) + \sum_{i=1}^n y_i(\alpha_i - \alpha_i^*)
```

RBF 커널:

$$\varphi(\mathbf{x}_i)^\top\varphi(\mathbf{x}) = k(\mathbf{x}_i, \mathbf{x}) = e^{-\gamma\|\mathbf{x}_i - \mathbf{x}\|^2} $$

> **기호 설명**
> - $\varphi(\cdot)$: 입력을 고차원 특징 공간으로 매핑하는 비선형 변환 함수
> - $\varepsilon$: $\varepsilon$-tube 너비, 이 범위 내 오차는 무시
> - $C$: 정규화 파라미터, 평탄도와 예측 오차 간의 트레이드오프 조절
> - $\xi_i, \xi_i^*$: 슬랙 변수 (tube 밖의 오차 허용량)
> - $\alpha_i, \alpha_i^*$: 쌍대 변수 (Lagrange multipliers)
> - $\gamma$: RBF 커널의 폭(bandwidth) 하이퍼파라미터
> - $k(\cdot, \cdot)$: 커널 함수 (Radial Basis Function)

> 💡 **커널 트릭(Kernel Trick)**: 고차원 특징 공간에서의 내적을 원래 입력 공간에서의 커널 함수로 대체하는 기법. 명시적 변환 없이도 비선형 관계 학습 가능

---

#### 🟢 모델 구조

```
[원시 FDC 데이터 (150+개 변수)]
        ↓ 전문가 초기 선별
[전처리: 결측값/상수/불일치 제거 (7단계 필터링)]
        ↓ ANOVA 검정
[문맥 변수 분석 (기본 설계 유형 p=2.8×10⁻¹⁰, 챔버 p=2.4×10⁻¹²)]
        ↓
[변수 집합 선택: TTB(3) / ES(17+5 binary) / FF(41)]
        ↓
[그리드 탐색 + 교차 검증으로 하이퍼파라미터 최적화]
        ↓
[회귀 모델 학습: SLR / MLR / PLR / RLR / SVR]
        ↓
[테스트 세트 평가 → VM 예측값 → R2R 제어]
```

---

#### 🟡 성능 향상 및 한계

| 구분 | 내용 |
|------|------|
| **성능 향상** | SVR(ES): Rel. RMSE = 0.432로 최고 성능 (조건부 VM 정확도 한계 0.507 이하 달성) |
| **성능 향상** | 전문가 변수 선택이 전체 변수 대비 성능 향상 |
| **한계 1** | 테스트 세트 크기 소규모 (39개 인스턴스), 통계적 신뢰도 제한 |
| **한계 2** | 훈련-테스트 간 5개월 간격 + 유지보수 이벤트로 분포 이동 발생 |
| **한계 3** | 단일 챔버, 단일 공정 대상으로 일반화 검증 부족 |
| **한계 4** | 전문가 의존적 변수 선택, 자동화 미완성 |
| **한계 5** | 하이퍼파라미터 $\varepsilon = 0.1$ 경험적 고정 (최적화 미수행) |

---

## 3. 각 주장 위치 표시

| 주장 | 위치 |
|------|------|
| SVR 테스트 우위 (Rel. RMSE 0.432) | Table III, p.6; Fig.3(b), p.8 |
| 선형 방법 과적합 | p.7, Sec.V Conclusion |
| ES 변수 집합 최적 | Table II, p.6; Sec.IV-A4, p.5 |
| MLR 불안정성 (Rel.Std=2.54) | Table II, p.6 |
| PLS 차원 축소 효과 | p.7, Sec.V |
| 유지보수 이벤트 영향 | p.7, Fig.3 caption |
| ANOVA 문맥 변수 분석 | p.5, Sec.IV-A2 |
| 하이퍼파라미터 최적화 | Sec.III-B, p.4 |

---

## 4. 저자 보고 vs. 해석 분리

### 📌 저자가 직접 보고한 결과

| 항목 | 저자 보고 내용 |
|------|----------------|
| 최고 테스트 성능 | SVR, ES 변수셋, Rel. RMSE = 0.432 |
| 훈련 최고 성능 | PLR/RLR, ES 변수셋, CV Rel. RMSE = 0.322/0.323 |
| SLR 대비 SVR 우위 | 테스트에서 SVR이 SLR보다 56.8% 낮은 오차 |
| MLR 불안정성 | TTB-MLR Rel. Std = 2.54 |
| Instance No.29 이후 성능 저하 | 연간 챔버 유지보수(wet clean) 후 SLR/MLR/PLR/RLR 급락 |
| 조건부 VM 정확도 한계 | Rel. RMSE 기준 0.507 |

### 🔍 검토자의 해석

| 항목 | 해석 |
|------|------|
| SVR 강건성 원인 | $\varepsilon$-insensitive 손실 함수와 커널 기반 비선형 매핑이 분포 이동에 대한 완충 역할 수행 |
| PLR/RLR 훈련 우위의 의미 | 훈련 세트 내 선형 구조가 PLS 차원 축소에 유리하나, 외삽(extrapolation) 상황에서는 취약 |
| 테스트 셋 39개의 한계 | 소규모 테스트로 인해 단일 유지보수 이벤트가 전체 성능 평가에 과도한 영향을 미칠 가능성 |
| 전문가 변수 선택의 의미 | 도메인 지식이 규제화(regularization)의 대안으로 작용하는 특징 선택 효과 발생 |
| FF > ES 의 역설 | 변수가 많을수록 잡음이 증가하여 오히려 성능이 저하될 수 있음 (차원의 저주) |

> 💡 **외삽 (Extrapolation)**: 훈련 데이터의 범위를 벗어난 입력값에 대해 예측하는 것. 일반적으로 보간(interpolation)보다 신뢰도가 낮음

---

## 5. 통계적 취약점 및 비교 불가 수치

| ⚠️ 문제 유형 | 세부 내용 |
|-------------|-----------|
| 🔴 **소규모 테스트셋** | 테스트 39개 인스턴스 (단일 챔버, 5개월) → 통계적 검정력 매우 부족 |
| 🔴 **단일 이벤트 교란** | Instance No.29의 유지보수 이벤트 1건이 전체 성능 비교를 좌우 → 편향 위험 |
| 🟡 **비교 불가 수치** | Table II의 상대 RMSE와 Table III의 상대 RMSE는 **다른 기준값(std)** 사용 가능성 있어 직접 비교 시 주의 필요 |
| 🟡 **하이퍼파라미터 고정** | SVR의 $\varepsilon=0.1$을 경험적으로 고정하여 최적화 미수행 → 성능 과소/과대 평가 가능 |
| 🟡 **데이터 불균형** | 기본 설계 유형별 인스턴스 수: 8~30개로 불균형 (총 98개 중 일부 유형은 8개) |
| 🟠 **단일 공정/장비** | Infineon 레겐스부르크 팹의 단일 PECVD 장비 1대 → 외부 타당도 낮음 |
| 🟠 **시간 갭** | 훈련(9개월)과 테스트(5개월) 사이 5개월 간격 → 분포 이동이 결과에 혼입 |

---

## 6. 논문이 답하지 않는 질문

| # | 미해결 질문 |
|---|------------|
| 1 | SVR이 훈련 세트에서 PLR/RLR보다 열등한 이유는 무엇인가? (이론적 설명 부재) |
| 2 | 테스트 세트를 더 늘렸을 때도 SVR 우위가 유지되는가? |
| 3 | 다른 챔버, 다른 공정에 동일 모델을 적용하면 성능이 어떻게 되는가? |
| 4 | 실시간(온라인) 재학습 시 SVR의 계산 비용은 허용 가능한가? |
| 5 | 자동화된 특징 선택이 ES 수준의 성능을 달성할 수 있는가? |
| 6 | 층 두께 외의 다른 품질 지표(균일도, 응력 등)에도 동일한 방법이 유효한가? |
| 7 | R2R 제어와 VM 예측의 피드백 루프에서 오차 전파는 어떻게 관리되는가? |
| 8 | $\varepsilon = 0.1$ 고정의 민감도 분석 결과는 어떠한가? |

---

## 7. 가장 중요한 그림 5개 해석

> *논문에 실제 포함된 그림은 Fig.1, Fig.2, Fig.3(a), Fig.3(b)이며, Table II와 Table III는 핵심 수치 자료로 그림에 준하여 분석합니다.*

### 📊 Fig. 1 — 금속 패시베이션 층 구조 (p.2)

```
[Passivation Full Stack]
    ├── Silicon Nitride Cap Layer  → PECVD Si₃N₄ Thickness (예측 타겟)
    ├── Silicon Oxide Base Layer   → PECVD SiO₂ Thickness
    └── Metal Layer Stack
```

**해석**: 이 그림은 연구의 물리적 맥락을 설명한다. $Si_3N_4$ 캡 층이 금속 층을 보호하는 최상단 보호막임을 시각화하며, 이 층의 두께 균일성이 소자 신뢰성에 직결됨을 보여준다. VM의 타겟 변수가 왜 이 층 두께인지의 동기를 명확히 제공한다.

---

### 📊 Fig. 2 — VM 시스템 아키텍처 (p.2)

**해석**: PECVD 장비 → FDC 데이터 → VM 예측기 → R2R 제어기 → 장비 제어변수 재설정의 피드백 루프를 도식화한다. 물리적 계측은 샘플링 시에만 수행되어 VM 예측기를 재학습하는 데 사용된다. 이 그림은 VM이 단순한 예측 도구가 아니라 공정 제어 루프의 핵심 구성요소임을 강조하며, 논문의 실용적 기여를 잘 보여준다.

> 💡 **피드백 루프**: 출력 결과를 다시 입력으로 활용하여 시스템을 자동으로 조정하는 제어 구조

---

### 📊 Table II — 훈련 세트 CV 결과 (p.6)

**해석**: 5개 방법 × 3개 변수셋 조합의 10-fold CV 성능을 보여주는 핵심 비교표. 가장 주목할 점은 TTB-MLR의 Rel. Std = 2.54로 다른 방법(0.004~0.019)과 비교하여 수백 배 불안정함. 이는 MLR이 다중공선성에 극도로 민감함을 입증한다. 또한 FF에서 SVR(0.344)이 PLR(0.371)보다 나은 반면, ES에서는 PLR(0.322) > SVR(0.339)인 역전 현상은 고차원 데이터에서의 커널 방법의 장점을 시사한다.

---

### 📊 Fig. 3(a) — SLR/MLR 테스트 예측 시계열 (p.8)

**해석**: SLR은 초반부 측정값 변동을 추적하지 못하고 후반부에서 과대 추정. MLR은 Instance No.29 이전까지 윤곽을 따르나 유지보수 후 급격히 과소 추정. **두 방법 모두 훈련 범위를 벗어난 `count of processed wafers since last wet clean` 변수에 민감하게 반응**하여 대규모 체계적 오차 발생. 이는 단순 선형 외삽의 위험성을 직접적으로 보여준다.

---

### 📊 Fig. 3(b) — PLR/RLR/SVR 테스트 예측 시계열 (p.8)

**해석**: PLR과 RLR은 No.1~15 구간에서 우수하나 No.16~28에서 과대 추정, No.29~38에서 과소 추정의 체계적 편향 발생. **SVR만이 유지보수 이벤트(No.29) 전후로 비교적 균일한 오차 분포를 유지**하며 실측값을 근접 추적. 이 그림은 논문의 핵심 주장인 "SVR의 강건성"을 가장 직접적으로 시각화하는 결정적 증거다.

---

## 8. 결론: 시사점, 후속 연구, 추가 방향

### 8-1. 모델의 일반화 성능 향상 가능성

#### 저자 제시 시사점

| 시사점 | 내용 |
|--------|------|
| SVR 강건성 | 훈련 범위 외 조건에서도 SVR이 선형 방법보다 견고 |
| 전문가 지식 통합 | 변수 선택에 도메인 지식을 활용하면 성능 추가 향상 |
| 데이터 충분성 | 충분한 훈련 데이터가 외삽 문제를 완화할 수 있음 |
| 범위 이탈 감지 | 변수가 훈련 범위를 벗어나면 해당 변수 제거 또는 경고 발행 제안 |

#### 저자 제시 후속 연구

- 자동화 특징 선택 (Filter 방법: 상관관계, 상호 정보량 / Wrapper 방법: SVR 기반)
- 분류 문제로의 전환 (above/within/below 스펙 범위)
- 물리 모델과 통계 모델의 하이브리드 접근
- 자동 재학습(retraining) 시스템 완성

#### 🔬 일반화 성능 향상을 위한 추가 후속 연구 방향

| 전략 | 구체적 방법 | 기대 효과 |
|------|------------|-----------|
| **도메인 적응** | Transfer Learning, Domain Adaptation으로 다른 챔버/공정에 모델 이전 | 단일 장비 의존성 탈피 |
| **온라인 학습** | Incremental SVR, 슬라이딩 윈도우 재학습 | 공정 drift 실시간 추적 |
| **불확실성 정량화** | Gaussian Process Regression, Conformal Prediction | 예측 신뢰 구간 제공 |
| **이상치 탐지 통합** | One-Class SVM, Isolation Forest로 훈련 범위 이탈 자동 감지 | 유지보수 이벤트 전후 성능 보호 |
| **앙상블 방법** | SVR + Random Forest 앙상블 | 분산 감소, 안정성 향상 |
| **물리 정보 내장** | Physics-Informed Neural Network (PINN) | 물리적으로 타당한 예측 보장 |

> 💡 **도메인 적응 (Domain Adaptation)**: 소스 도메인(예: 챔버 A)에서 학습한 모델을 타겟 도메인(예: 챔버 B)에 적용할 때 분포 차이를 보정하는 기법
> 💡 **Gaussian Process Regression**: 예측값뿐 아니라 예측의 불확실성(분산)도 함께 출력하는 베이지안 비모수 회귀 방법

---

### 8-2. 2020년 이후 최신 연구 비교 분석

> ⚠️ **주의**: 아래 내용은 2020년 이후 VM 및 반도체 제조 AI 분야의 일반적인 연구 트렌드를 기반으로 작성되었으며, 각 논문의 구체적 수치는 직접 확인이 필요합니다. 특정 논문의 결과를 단정적으로 인용하지 않겠습니다.

#### 연구 트렌드 비교

| 항목 | 본 논문 (2014) | 2020년 이후 트렌드 |
|------|---------------|-------------------|
| **주요 방법** | SVR, PLS, Ridge | Deep Learning (LSTM, Transformer), GNN |
| **특징 선택** | 전문가 + 통계 필터 | AutoML, NAS, Attention Mechanism 자동화 |
| **불확실성** | 없음 (점 예측) | Bayesian Deep Learning, Conformal Prediction |
| **데이터 효율성** | 98개 훈련 인스턴스 | Semi-supervised, Few-shot Learning |
| **멀티태스크** | 단일 타겟 (Si₃N₄ 두께) | Multi-output VM (두께+균일도+응력 동시 예측) |
| **실시간성** | 오프라인 학습 | Edge AI, 스트리밍 학습 |
| **설명 가능성** | 없음 | SHAP, LIME 기반 XAI |

#### 본 논문이 후속 연구에 미친 영향

1. **SVR의 VM 적용 가능성 확립**: 이후 반도체 VM 연구에서 SVR이 기준선(baseline)으로 널리 활용됨
2. **체계적 방법론 비교 프레임**: 변수셋 × 알고리즘 교차 비교 설계가 이후 VM 벤치마크 연구의 표준이 됨
3. **공정 전문 지식의 정량적 가치 입증**: ES vs FF 비교로 도메인 지식의 데이터 효율성 기여를 정량화

#### 향후 연구 시 고려 사항

| 고려 사항 | 이유 |
|-----------|------|
| **데이터 분포 이동(Concept Drift) 대응** | 본 논문에서 유지보수 이벤트로 인한 성능 급락이 확인된 만큼, 적응형 모델 필수 |
| **실제 배포 환경의 계산 제약** | SVR의 훈련/추론 비용이 실시간 공정 제어에 적합한지 검토 필요 |
| **다중 공정 일반화 검증** | 단일 공정에서의 결과를 과잉 일반화하지 말 것 |
| **공정 물리학과의 통합** | 순수 데이터 기반 접근의 한계를 물리 모델로 보완 |
| **설명 가능성 요구** | 반도체 제조 현장에서는 예측 근거 제공이 엔지니어 신뢰 확보에 필수 |

---

## 📚 참고 자료

**본 논문 내 인용 문헌 (주요)**

1. Purwins et al. (2014) - 본 논문: "Regression Methods for Virtual Metrology of Layer Thickness in Chemical Vapor Deposition," *IEEE/ASME Transactions on Mechatronics*, Vol.19, No.1
2. Smola & Schölkopf (2004) - "A tutorial on support vector regression," *Statistics and Computing*, Vol.14, No.3 [논문 내 참고문헌 14]
3. Cheng et al. (2012) - "Developing an automatic virtual metrology system," *IEEE TSASE*, Vol.9, No.1 [논문 내 참고문헌 10]
4. Kang et al. (2009) - "A virtual metrology system for semiconductor manufacturing," *Expert Systems with Applications*, Vol.36 [논문 내 참고문헌 13]
5. Jørgensen & Goegebeur (2007) - "Multivariate data analysis and chemometrics" [논문 내 참고문헌 17]
6. Bishop (2006) - *Pattern Recognition and Machine Learning*, Springer [논문 내 참고문헌 18]

**2020년 이후 관련 연구 트렌드 참고 (일반 분야)**

- IEEE Transactions on Semiconductor Manufacturing (2020-2024) — Virtual Metrology 특집 논문들
- IEEE/ASME Transactions on Mechatronics — 반도체 공정 제어 관련 후속 연구들
- SEMI (반도체 장비·소재 국제협회) 기술 보고서

> ⚠️ **정확도 관련 고지**: 2020년 이후 특정 논문의 구체적 결과 수치는 본 AI의 학습 데이터 한계로 인해 100% 정확하게 인용하기 어려워 일반적 트렌드 분석으로 대체하였습니다. 구체적 최신 논문 비교는 Google Scholar, IEEE Xplore, arXiv에서 "Virtual Metrology + Deep Learning" 키워드 검색을 권장드립니다.
