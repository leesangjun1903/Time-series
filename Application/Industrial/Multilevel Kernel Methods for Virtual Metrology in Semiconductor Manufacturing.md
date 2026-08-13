# Multilevel Kernel Methods for Virtual Metrology in Semiconductor Manufacturing

> **⚠️ 정확도 고지**: 본 논문은 2011년 IFAC World Congress 발표 논문(pp. 11614–11621)으로, PDF 전문을 기반으로 분석하였습니다. Figure 5는 PDF 내에 "Table 1"로 표기된 12개 경로별 RMSE 패널 그래프를 의미합니다(논문 본문 p.6에서 "panels in Figure 5"로 언급). 2020년 이후 최신 연구 비교는 제 학습 데이터(~2023년) 기반이며, 논문 간 수치 직접 비교 시 실험 환경 차이로 인한 비교 불가능성이 존재할 수 있음을 명시합니다.

---

## 1. Executive Summary (10문장 이내)

반도체 제조 공정에서 실제 계측(측정)은 주사전자현미경(SEM) 사용으로 비용과 시간이 막대하게 소요되므로, 공정 데이터만으로 계측값을 예측하는 **가상 계측(Virtual Metrology, VM)**이 주목받고 있다.  
본 논문은 VM의 두 가지 핵심 난제인 **(i) 고차원 입력 공간에서의 비선형 함수 회귀**와 **(ii) 다중 챔버·공정 경로로 인한 데이터 이질성**을 동시에 해결하고자 한다.  
저자들은 커널 방법(Kernel Methods)과 멀티태스크 학습(Multitask Learning), 혼합 효과 모델(Mixed-Effects Model)을 결합한 **계층적(Multilevel) 프레임워크**를 제안한다.  
각 웨이퍼의 물류 경로(Equipment → Process → Chamber → Subchamber)를 트리 구조로 모델링하고, 경로별 공통성을 활용해 데이터가 적은 경로에서도 신뢰도 높은 예측이 가능하다.  
모델 파라미터는 REML(제한 최대우도) 최적화로 자동 튜닝된다.  
Infineon Technologies Austria의 CVD 장비 실데이터(3,000개 웨이퍼, 29주)로 검증하였으며, 약 600개의 입력 변수로 평균 두께를 예측한다.  
비교 대상으로 단순 평균 전파(Naive Baseline)와 단일 수준 KRR(Kernel Ridge Regression)을 사용하였다.  
실험 결과, 제안 방법은 94개 테스트 포인트 중 90개에서 단일 수준 KRR 대비 낮은 RMSE를 기록하였다.  
이 접근법은 특히 소량 생산(low-volume) 공정 환경에서 더욱 효과적이며, VM 기반 Run-to-Run 제어, 샘플링 전략 최적화 등에 기여할 수 있다.

---

### 1-1. 연구의 목적과 필요성

| 항목 | 내용 |
|------|------|
| **배경** | 반도체 제조에서 SEM 기반 계측은 비용·시간 집약적 → 전체 웨이퍼 중 소수만 실측 가능 |
| **목적** | 공정 데이터(센서, 설정값)만으로 계측값(평균 레이어 두께)을 신뢰도 높게 예측 |
| **필요성 1** | 고차원(~600개 변수) 비선형 입력 공간: 일반 선형 모델로는 과적합 또는 계산 불가 |
| **필요성 2** | 다중 챔버/공정 경로로 인한 데이터 이질성: 경로별 독립 모델은 데이터 부족, 통합 단일 모델은 챔버 간 차이 무시 |
| **기대 효과** | 계측 비용 절감, Run-to-Run 제어 품질 향상, 샘플링 전략 최적화 |

> 📌 **Run-to-Run(R2R) 제어**: 각 배치(lot) 공정 결과를 피드백으로 활용해 다음 배치의 공정 파라미터를 조정하는 제어 방식. VM이 계측 지연을 줄여 R2R 제어 성능을 향상시킴.

---

## 2. 핵심 주장과 근거 표

| # | 핵심 주장 | 근거 / 방법 | 위치 |
|---|-----------|-------------|------|
| 1 | 고차원 비선형 문제는 커널 트릭으로 효율적 해결 가능 | 커널 함수 $K(u,v) = \langle\phi(u),\phi(v)\rangle$를 통해 고차원 특징 공간을 명시적으로 계산하지 않고 내적만으로 처리 | Section 2, p.2–3 |
| 2 | 다중 수준 물류 경로 이질성은 트리 계층 구조로 모델링 가능 | 가산 추정기 $f(x_\*) = \sum\_{j \in \mathcal{P}\_\*} f_j(x_{(*,j)})$ 로 경로별 함수 분리 | Section 3.1, p.3–4 |
| 3 | 경로 간 유사도 정규화로 데이터 희소 경로 성능 향상 | $\xi_{j,z}\|\|\bar{w}_j - \bar{w}_z\|\|^2$ 항으로 입력 호환 노드 간 유사성 강제 | Section 3.2, p.4 |
| 4 | REML로 모델 파라미터 자동 최적화 가능 | Problem 4: $\arg\min_{[\lambda, \xi, \theta]} -l_R(\lambda, \xi, \theta)$ | Section 3.3, p.5 |
| 5 | 제안 모델이 단일 수준 KRR 대비 체계적으로 우수 | 94개 테스트 중 90개에서 RMSE 비율 < 1.0 (Table 2) | Section 4.2, p.6 |

---

### 2-1. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

#### 🔴 해결하고자 하는 문제

**문제 1: 고차원 비선형 회귀**
- 약 600개의 입력 변수(센서 시계열, 설정값 등)
- 다항식 기저 확장 시 계산량이 기하급수적으로 증가
  $$\bar{p} = (d-1)p + \sum_{i=1}^{d}\binom{p}{i} \quad \cdots (6)$$
  - $p$: 원래 입력 변수 수, $d$: 다항식 차수, $\bar{p}$: 확장 후 변수 수
  - 예: $p=100, d=2$이면 $\bar{p}=5150$; $d=3$이면 $\bar{p}=166,950$

**문제 2: 데이터 이질성 (Data Heterogeneity)**
- 동일 장비의 챔버/서브챔버/공정 조합으로 최대 12개의 물류 경로 존재 (Figure 1)
- 경로별 독립 모델: 데이터 부족 → 과적합 위험
- 단일 통합 모델: 챔버 간 고유 특성 무시 → 예측 정확도 저하

---

#### 🔵 제안하는 방법 (수식 포함)

**Step 1: 커널 릿지 회귀(KRR) 기반 (단일 수준)**

OLS 손실 함수:
$$J_{OLS}(w) := \frac{1}{2}\|Y - Xw\|^2 = \frac{1}{2}\sum_{i=1}^{n}(y_i - x_i w)^2 \quad \cdots (1)$$
- $Y \in \mathbb{R}^n$: 출력 벡터(계측값), $X \in \mathbb{R}^{n \times p}$: 입력 행렬, $w \in \mathbb{R}^p$: 계수 벡터

릿지 회귀 손실 함수:
$$J_{RR}(w) := \frac{1}{2}\|Y - Xw\|^2 + \frac{\lambda}{2}w'w \quad \cdots (3)$$
- $\lambda \in \mathbb{R}^+$: 정규화 하이퍼파라미터 (바이어스-분산 트레이드오프 조정)

> 📌 **Ridge Regression (릿지 회귀)**: OLS에 $L_2$ 패널티 항을 추가하여 계수의 크기를 제한함으로써 과적합을 방지하는 방법. Tikhonov 정규화라고도 함.

커널 트릭 적용 후 이중 형태(dual form):
$$f_{RR}(x) = \mathbf{k}(x)(\mathbf{K} + \lambda I)^{-1}Y = \mathbf{k}(x)c^* \quad \cdots (11)$$
- $\mathbf{K} \in \mathbb{R}^{n \times n}$: 그람 행렬 (Gram matrix), $K[i,j] = K(x_i, x_j)$
- $\mathbf{k}(x) \in \mathbb{R}^{1 \times n}$: 예측 포인트와 훈련 데이터 간 커널 벡터
- $c^* = (\mathbf{K} + \lambda I)^{-1}Y$: 최적 가중치 벡터

> 📌 **Gram matrix (그람 행렬)**: 모든 훈련 데이터 쌍 간의 커널 함수값으로 구성된 $n \times n$ 행렬. 원래 입력 공간 대신 이 행렬만으로 학습 가능하게 해주는 핵심 객체.

> 📌 **커널 트릭(Kernel Trick)**: 고차원 특징 공간 $\phi(x)$를 명시적으로 계산하지 않고, 두 점 간의 내적 $\langle\phi(u), \phi(v)\rangle$만을 커널 함수 $K(u,v)$로 계산하는 기법. 계산 복잡도를 획기적으로 줄임.

**Step 2: 다수준 커널 릿지 회귀 (핵심 기여)**

가산 추정기:

```math
f(x_*, \mathcal{P}_*) = \sum_{j \in \mathcal{P}_*} f_j(x_{(*,j)}) = \sum_{j \in \mathcal{P}_*} \mathbf{k}_j(x_*)c \quad \cdots (18)
```

- $\mathcal{P}_*$: 예측 대상 웨이퍼의 물류 경로 (예: {장비→공정1→챔버A1})
- $f_j$: $j$번째 노드에 대한 추정 함수
- $x_{(*,j)}$: 노드 $j$에 할당된 입력 변수 부분집합

**Problem 2** (다수준 KRR 최적화):

$$\arg\min_c \frac{1}{2}\|Y - \mathbf{K}c\|^2 + \frac{1}{2}c'\mathbf{G}c \quad \cdots (19)$$

정규화 행렬 $\mathbf{G}$:

$$\mathbf{G}[i,k] = \sum_{j=0}^{\eta-1}\lambda_j \mathbf{K}_j[i,k] + \sum_{j,z \in IC}\xi_{j,z}\tilde{G}_{(j,z)}[i,k] \quad \cdots (20)$$

- $\lambda_j$: $j$번째 노드의 정규화 파라미터 ($\|\bar{w}_j\|^2$ 페널티)
- $\xi_{j,z}$: 입력 호환 노드 $j, z$ 간 유사도 강제 파라미터 ($\|\bar{w}_j - \bar{w}_z\|^2$ 페널티)
- $IC$: 입력 호환(input-compatible) 노드 쌍의 집합
- $\eta$: 트리의 전체 노드 수

> 📌 **입력 호환(Input-Compatible)**: 두 노드가 동일한 입력 공간을 공유하고 동일 커널을 사용하며 상호 배타적인 경우(한 웨이퍼가 두 노드 모두를 통과할 수 없음). 예: 서브챔버 A1과 A2는 입력 호환.

최적해:
$$c^* = (\mathbf{K}^2 + \mathbf{G})^{-1}\mathbf{K}Y \quad \cdots (22)$$

**Problem 2b** (원형(primal form) 재표현):

$$\arg\min_c \frac{1}{2}\left\|Y - \sum_{j=0}^{\eta-1}\bar{X}_j\bar{w}_j\right\|^2 + \frac{1}{2}\sum_{j=0}^{\eta-1}\lambda_j\|\bar{w}_j\|^2 + \frac{1}{2}\sum_{(j,z)\in IC}\xi_{j,z}\|\bar{w}_j - \bar{w}_z\|^2 \quad \cdots (26)$$

- $\bar{w}_j$: $j$번째 노드의 특징 공간 가중치 벡터
- 첫 번째 항: 예측 오차 최소화
- 두 번째 항: 각 노드 모델 복잡도 제한
- 세 번째 항: 입력 호환 노드 간 유사도 강제

> 📌 **Multitask Learning (멀티태스크 학습)**: 여러 관련 태스크를 동시에 학습하여 태스크 간 공유 정보를 활용함으로써 각 태스크의 성능을 향상시키는 패러다임. 여기서는 각 노드의 추정 함수를 하나의 태스크로 간주.

**Step 3: 확률론적 해석 및 REML 최적화**

출력 오차 공분산 행렬 $W$를 포함한 Problem 3:
$$\arg\min_c \frac{1}{2}\|Y - \mathbf{K}c\|^2_W + \frac{1}{2}c'\mathbf{G}c \quad \cdots (27)$$
- $\|x\|^2_W = x'Wx$: $W$-가중 노름

사후 분포:
$$c|Y \sim \mathcal{N}(\mu_c, \Sigma_c)$$
$$\mu_c = (\mathbf{K}W\mathbf{K} + \mathbf{G})^{-1}\mathbf{K}WY, \quad \Sigma_c = (\mathbf{K}W\mathbf{K} + \mathbf{G})^{-1} \quad \cdots (29)$$

오차 분산 모델:

```math
W^{-1} = \text{diag}_i\left\{\sum_{j \in \mathcal{P}_i}\sigma^2_j\right\} \quad \cdots (33)
```

- $\sigma^2_j$: $j$번째 노드의 오차 분산

REML 목적함수:
$$l_R = \log p(Y|(\lambda, \xi, \theta)) = -\frac{1}{2}(Y - \mathbf{K}\mu_c)'(\mathbf{K}\Sigma_c\mathbf{K} + W^{-1})^{-1}(Y - \mathbf{K}\mu_c) - \frac{1}{2}\log|\mathbf{K}\Sigma_c\mathbf{K} + W^{-1}| \quad \cdots (31)$$

**Problem 4** (REML 최적화):
$$\arg\min_{[\lambda, \xi, \theta]} -l_R(\lambda, \xi, \theta) \quad \cdots (32)$$
- Newton-Raphson 알고리즘으로 최적해 탐색

> 📌 **REML (Restricted Maximum Likelihood, 제한 최대우도)**: 분산 성분 추정 시 고정 효과(fixed effects)를 먼저 소거(restrict)한 후 잔차의 우도를 최대화하는 방법. 분산 파라미터의 편향 없는 추정에 유리하며 혼합 효과 모델에서 널리 사용됨.

---

#### 🟢 모델 구조

```
트리 계층 구조 (Figure 1, 2 기반):
  
Level 0: 장비 전체 (Equipment, #0)
  → 모든 웨이퍼 공유 / 장비 수준 공통 효과 추정
  
Level 1: 공정 (Process 1, #1 / Process 2, #2)
  → 공정별 변수 할당 / 공정 수준 효과 추정
  
Level 2: 챔버·서브챔버 (A1/#3, A2/#4, B1/#5, B2/#6, C1/#7, C2/#8)
  → 챔버별 변수 할당 / 챔버 수준 효과 추정
  → 같은 챔버의 서브챔버(예: A1, A2)는 ξ로 유사도 강제

최종 예측: f(x*, P*) = f_장비(x) + f_공정(x) + f_챔버(x)
```

**파라미터 구성 (Section 4 구현)**:
- $\lambda$: 루트 노드용 1개 + 공정 노드용 1개 + 챔버 노드용 1개 = 총 3개
- $\xi$: 서브챔버 간 유사도용 1개 (공유)
- $\sigma^2_j$: $\eta = 9$개의 오차 분산
- **총 13개 파라미터** (vs. 단일 수준 KRR의 24개)

---

#### 🔶 성능 향상 및 한계

**성능 향상:**

| 지표 | 결과 |
|------|------|
| 비교 기준 | 단순 평균 전파(Naive), 단일 수준 KRR, **제안 Multilevel KRR** |
| 전체 우위 | 94개 테스트 포인트 중 **90개**에서 단일 KRR 대비 낮은 RMSE (Table 2) |
| 시각적 결과 | Figure 4: 모든 서브실험에서 평균 RMSE 기준 Multilevel KRR이 최저 |
| 파라미터 효율 | 13개 파라미터로 24개 대비 적은 파라미터로 더 나은 성능 |

**한계:**
- 단일 CVD 장비, 단일 공장(Infineon Villach)의 데이터만 사용 → 일반화 불확실
- 약 600개의 입력 변수 중 특징 선택(feature selection) 과정 미기술
- 컴퓨팅 복잡도: $n \times n$ 그람 행렬 역산 → $O(n^3)$ → 대규모 데이터에서 확장성 문제
- 이동 윈도우(moving window) 방식의 적응성에 의존 → 급격한 드리프트에 취약 가능성
- 트리 구조와 입력 변수 할당이 수동 설계 → 자동화 방법 미제시

---

## 3. 각 주장에 페이지/Figure/Table 번호 표시

| 주장 | 위치 |
|------|------|
| VM의 필요성 및 비용 절감 효과 | p.1 (Introduction) |
| 고차원 다항식 확장의 계산 불가능성 | p.2, Eq.(6) |
| 커널 트릭의 이중 형태 도출 | p.2–3, Eq.(7)–(11) |
| 트리 계층 구조 표현 | Figure 1 (p.1), Figure 2 (p.3) |
| 가산 추정기 정의 | p.3, Eq.(15) |
| 입력 호환 정의 | p.4, Definition |
| Problem 2 (다수준 KRR 최적화) | p.4, Eq.(19)–(23) |
| Problem 2b (원형 재표현) | p.4, Eq.(26) |
| 확률론적 해석 및 사후 분포 | p.5, Eq.(28)–(31) |
| REML 최적화 | p.5, Eq.(32) |
| 오차 분산 모델 | p.6, Eq.(33) |
| VM 예측 예시 | Figure 3 (p.6) |
| 평균 RMSE 비교 | Figure 4 (p.6) |
| 경로별 RMSE 비율 | Table 2 (p.8) |
| 경로별 RMSE 시각화 | Table 1 / "Figure 5" (p.8) |
| Multilevel > Single-level (90/94) | p.6, Section 4.2 |

---

## 4. 저자 직접 보고 vs. 분석자 해석 분리

### 📋 저자가 직접 보고한 결과

**연구 주제:**
- 반도체 CVD 공정의 Virtual Metrology를 위한 Multilevel Kernel Ridge Regression 프레임워크 제안 (p.1, Abstract)

**방법:**
- 커널 트릭으로 고차원 비선형 회귀 처리: $f_{RR}(x) = \mathbf{k}(x)(\mathbf{K}+\lambda I)^{-1}Y$ (p.2–3, Eq.11)
- 가산 다수준 추정기: $f(x_\*, \mathcal{P}\_\*) = \sum_{j \in \mathcal{P}\_\*}\mathbf{k}\_j(x_*)c$ (p.3, Eq.18)
- REML로 하이퍼파라미터 자동 최적화 (p.5, Eq.32)
- 이동 윈도우 방식의 적응적 구현 (p.6, Section 4.1)

**결과:**
- "multilevel KRR performs systematically better than single-level one (90 out of 94 test points)" (p.6, Section 4.2)
- 평균 RMSE에서 모든 서브실험에서 Multilevel KRR이 최저 (Figure 4, p.6)
- 이동 윈도우 최적 길이: 4~8주 (공정 드리프트 확인) (p.6, Section 4.2)

---

### 🔍 분석자 해석

**방법론 관련:**
- Problem 2b의 원형 표현(Eq.26)은 $\xi$ 파라미터가 "챔버 간 가중치 벡터 차이의 $L_2$ 노름을 최소화"한다는 점에서, 사실상 **멀티태스크 학습에서의 관계 정규화(relational regularization)**와 동일한 효과를 가짐. 이는 다수준 구조를 Fused LASSO나 그래프 정규화와 개념적으로 연결 가능.
- REML은 분산 파라미터 추정에 유리하지만, 비볼록(non-convex) 최적화 문제이므로 Newton-Raphson이 국소 최적해에 수렴할 위험이 있음 → 논문에서 미언급.

**실험 설계 관련:**
- 9개의 중첩 서브실험은 시간적 순서를 보존하는 **시계열 교차검증(time-series cross-validation)**으로, 일반적인 k-fold 교차검증보다 적절한 설계.
- 단, 동일 장비·단일 공장 데이터로 외부 타당도(external validity)가 제한적.

---

## 5. 통계적으로 취약한 부분 및 비교 불가능한 수치

| 항목 | 취약/비교불가 이유 |
|------|--------------------|
| ⚠️ **단일 장비·단일 공장** | Infineon Villach 1개 CVD 장비 데이터만 사용. 다른 장비 유형(ALD, PVD 등)이나 다른 제조사로의 일반화 근거 없음 |
| ⚠️ **"90 out of 94" 주장** | 통계적 유의성 검정(t-test, Wilcoxon signed-rank test 등) 미수행. 단순 빈도 보고에 그침 |
| ⚠️ **RMSE 절대값 익명화** | Figure 3, 4에서 두께 측정값이 익명화(anonymized)되어 있어 실제 예측 정확도의 공학적 의미 해석 불가 |
| ⚠️ **Table 2의 "/" 값** | 특정 경로·주차 조합에서 데이터 부재로 비어 있음 → 해당 조합의 성능 평가 불가. 이는 데이터 불균형 문제를 반영 |
| ⚠️ **비교군 제한** | SVM, 신경망, PLS(Partial Least Squares) 등 당시 VM에서 사용된 주요 알고리즘과의 비교 없음 |
| ⚠️ **이동 윈도우 최적화** | 검증 세트가 1주치에 불과해 윈도우 길이 선택의 분산이 클 수 있음 |
| ⚠️ **계산 복잡도 미보고** | $O(n^3)$ 그람 행렬 역산의 실제 연산 시간 미보고. 실시간 VM 적용 가능성 판단 불가 |
| ⚠️ **Table 2 일부 >1.0** | Process2-A1(week 21: 1.0228), Process2-A2(week 21: 1.1416), Process1-B1(week 9: 1.0819), Process1-B2(week 6: 0.9096→사실 <1이나 Process2-B2 week 21: 1.0674) 등 일부 경로에서 단일 수준 KRR 대비 성능 저하 → 조건부 우위 |

---

## 6. 논문이 답하지 않는 질문

| # | 미해결 질문 |
|---|------------|
| 1 | **특징 선택(Feature Selection)**: 약 600개의 입력 변수 중 어떤 기준으로 각 노드에 변수를 할당하였는가? 자동화 가능한가? |
| 2 | **계산 확장성**: $n=3000$ 이상의 대규모 데이터셋에서 $O(n^3)$ 역산 병목을 어떻게 해결할 것인가? |
| 3 | **트리 구조 설계**: 트리의 깊이와 분기 구조는 어떻게 결정되는가? 데이터로부터 자동 학습 가능한가? |
| 4 | **온라인/실시간 업데이트**: REML 최적화는 배치(batch) 방식으로만 가능한가? 실시간 공정 모니터링에 적용 시 업데이트 방법은? |
| 5 | **다른 공정 유형 적용 가능성**: CVD 외 ALD, 에칭(Etching), CMP 등 다른 공정에도 동일 프레임워크 적용 가능한가? |
| 6 | **커널 함수 선택 기준**: RBF, 다항식 커널 중 어떤 커널을 사용했는지, 선택 기준은 무엇인지 미기술 |
| 7 | **VM 불확실도 활용 방법**: 사후 분포 $\Sigma_c$에서 도출되는 예측 불확실도를 어떻게 실제 공정 제어에 활용할 것인가? |
| 8 | **이상치 및 장비 고장 감지**: 예측값과 실제 계측값의 큰 괴리를 이상 신호로 활용하는 방법론이 없음 |
| 9 | **모델 노후화 감지(Model Drift Detection)**: 공정 드리프트 외에 모델 자체의 노후화를 언제, 어떻게 감지하고 재학습할 것인가? |
| 10 | **다른 목표 변수(다변량 출력)**: 단일 출력(평균 두께)만 예측. 여러 품질 지표를 동시 예측하는 다변량 확장은? |

---

## 7. 가장 중요한 그림 5개의 해석

### Figure 1 (p.1) — CVD 장비의 트리 구조 표현

```
Equipment 1
├── Process 1 ──── A1, A2, B1, B2, C1, C2
└── Process 2 ──── A1, A2, B1, B2, C1, C2
                   (총 12개 물류 경로)
```

**해석**: 이 그림은 논문의 핵심 동기를 시각화한다. 단일 CVD 장비가 2개의 공정, 3개의 챔버, 각 2개의 서브챔버로 구성되어 총 12개의 물류 경로를 형성한다. 각 경로에 독립 모델을 구축하면 경로당 데이터가 너무 적고(3000÷12≈250개), 하나의 통합 모델은 챔버 간 차이를 무시한다. 이 딜레마가 바로 다수준 프레임워크의 필요성을 정당화한다.

> 📌 **CVD (Chemical Vapor Deposition, 화학 기상 증착)**: 기체 상태의 화학 물질이 반응하여 웨이퍼 표면에 박막을 형성하는 공정. 반도체 제조의 핵심 공정 중 하나.

---

### Figure 2 (p.3) — 번호가 매겨진 트리 표현

```
#0 (Equipment 1)
├── #1 (Process 1)    #2 (Process 2)
│   ├── #3(A1) #4(A2) #5(B1) #6(B2) #7(C1) #8(C2)
```

**해석**: Figure 1의 추상적 트리에 노드 번호를 부여하여 수학적 공식화의 기반을 제공한다. $\eta=9$개의 노드가 곧 추정해야 할 함수의 수이며, 이는 12개 경로 대비 적다. 서브챔버 쌍(A1↔A2, B1↔B2, C1↔C2)은 입력 호환 관계로 $\xi$ 파라미터로 연결된다. 이 구조가 Problem 2의 $\mathbf{G}$ 행렬 구성의 기초가 됨.

---

### Figure 3 (p.6) — VM 테스트 결과 (50개 웨이퍼, Process 1-A2, 27주차)

**해석**: X축은 웨이퍼 번호(1~50), Y축은 익명화된 평균 두께. 실제 계측값(파란 선, 다이아몬드)과 VM 예측값(빨간 선, 원)이 전반적으로 잘 일치한다. 주목할 점은:
- 웨이퍼 #1~15 구간: 예측이 비교적 정확
- 웨이퍼 #30~40 구간: 일부 피크에서 예측 편차 발생 → 공정 드리프트 또는 이상 거동 가능성
- 전반적으로 추세(trend)를 잘 추종하나, 급격한 변동(spike)에서는 과소/과대 예측 경향
- **익명화로 인해 RMSE의 절대적 공학적 의미 해석 불가** (⚠️ 통계적 취약점)

---

### Figure 4 (p.6) — 평균 RMSE 비교 (3가지 방법론)

**해석**: X축은 주차(3~27주), Y축은 경로별 평균 RMSE. 3개 곡선:
- **검은 점선 (Propagated Mean)**: 가장 높은 RMSE, 0.15~0.45 범위. 설명 변수를 전혀 활용하지 않아 기준선 역할
- **파란 점선 (Single-level KRR)**: 중간 RMSE, 0.10~0.35 범위. 커널 방법으로 개선되나 이질성 미반영
- **빨간 실선 (Multilevel KRR)**: 모든 주차에서 최저 RMSE. 약 0.05~0.25 범위

**핵심 관찰**: 초기(3~9주차)에는 훈련 데이터가 적어 세 방법 간 차이가 크고, 데이터 증가(15주차 이후)에도 Multilevel KRR의 우위가 일관되게 유지됨. 이는 데이터가 적은 초기에 경로 간 공통성 활용의 이점이 더 두드러짐을 시사.

---

### Table 1/Figure 5 (p.8) — 경로별 RMSE 패널 (12개 경로)

**해석**: 12개의 개별 패널에서 3가지 방법론의 RMSE를 각 경로별로 비교. 주요 관찰:
- **일관된 우위**: 대부분의 경로와 주차에서 빨간 선(Multilevel KRR)이 가장 낮음
- **예외 사례**: Process2-A1(21주), Process2-A2(21주), Process2-B2(21주) 등에서 Multilevel KRR이 단일 수준보다 높은 RMSE (Table 2에서 >1.0). 21주차는 이상치 또는 특이 공정 이벤트 가능성 있음
- **경로별 난이도 차이**: Process 2 경로들이 Process 1 대비 변동성이 더 큰 경향 → 공정 2의 물리적 특성 차이 가능성
- **데이터 부재(/)**: 특히 초기 주차(3~12주)에서 일부 경로 데이터 없음 → 해당 경로의 초기 성능 평가 불가 (⚠️)

---

## 8. 결론 및 후속 연구

### 8-0. 저자들이 제시한 시사점과 후속 연구 계획

**저자 제시 시사점 (Section 5, p.6):**
- 물류 경로의 공통성 활용이 이질적 데이터셋에서 VM 성능 향상에 핵심적 역할
- 경로별 데이터가 상대적으로 적은 소량 생산(low-volume) 환경에서 특히 유용
- 단일 수준 KRR 대비 체계적 우위를 실험적으로 입증

**저자 제시 후속 연구**: 논문에 명시적 후속 연구 계획 없음 (⚠️ 해당 섹션 부재)

---

### 8-1. 모델의 일반화 성능 향상 가능성

#### 현재 일반화의 한계

1. **데이터 범위**: 단일 장비, 단일 공장, 단일 공정(CVD), 단일 목표 변수(평균 두께)
2. **시간적 범위**: 29주, 3,000개 웨이퍼 → 장기 공정 변화 대응 미검증
3. **모델 구조의 수동성**: 트리 구조와 입력 변수 할당이 도메인 지식에 의존

#### 일반화 성능 향상을 위한 잠재적 방향

**① 입력 공간의 자동 특징 선택**

현재 약 600개 변수를 수동으로 노드에 할당. 다음 방법으로 자동화 가능:
$$\min_{w, s} \frac{1}{2}\|Y - Xw\|^2 + \lambda\|w\|^2 + \mu\|s\|_1$$
- $s$: 변수 선택 마스크 벡터 ($L_1$ 정규화로 희소성 유도)
- LASSO 또는 Group LASSO와 결합한 커널 방법 적용 가능

**② 스파스 커널 근사(Sparse Kernel Approximation)**

현재 $O(n^3)$ 복잡도를 Nyström 근사 또는 Random Fourier Features로 $O(nm^2)$ ($m \ll n$)으로 축소:
$$K(x_i, x_j) \approx \phi_{RF}(x_i)'\phi_{RF}(x_j), \quad \phi_{RF}(x) = \sqrt{\frac{2}{D}}\cos(\omega'x + b)$$
- $\omega$: 커널의 푸리에 주파수 샘플, $D$: 근사 차원
- 이를 통해 대규모 데이터(수만 웨이퍼)로의 확장 가능

> 📌 **Nyström 근사**: 큰 그람 행렬을 소수의 "기준점(landmark)"을 선택해 저차원으로 근사하는 방법. 계산 복잡도를 크게 줄임.

**③ 전이 학습(Transfer Learning) 통한 도메인 일반화**

다른 공장/장비의 데이터를 소스 도메인으로 활용:
$$f_{target}(x) = f_{source}(x) + \Delta f(x)$$
- 소스 도메인(다른 공장) 모델 $f_{source}$를 사전 학습 후, 타깃 도메인(현재 공장) 잔차 $\Delta f$만 학습
- 데이터가 매우 적은 신규 장비 적용 시 유리

**④ 온라인/적응형 학습**

현재 이동 윈도우는 배치 방식. 순차적 베이즈 업데이트로 실시간 적응:
$$p(c|Y_{1:t+1}) \propto p(y_{t+1}|c, x_{t+1}) \cdot p(c|Y_{1:t})$$
- 새 관측값이 들어올 때마다 사후 분포 업데이트
- 급격한 공정 드리프트에도 빠른 적응 가능

**⑤ 다변량 출력(Multi-output VM)**

단일 두께값 예측에서 여러 품질 지표 동시 예측으로 확장:
$$\mathbf{Y} = [\text{두께}, \text{균일도}, \text{결함 밀도}, \ldots]$$
- 다중 출력 가우시안 프로세스(Multi-output GP)와 결합 가능

---

### 8-2. 2020년 이후 최신 연구 비교 분석

> ⚠️ **중요 고지**: 아래 최신 연구들은 제 학습 데이터(~2023년 초) 기반이며, 논문별 수치를 직접 비교하는 것은 실험 환경 차이(장비 종류, 데이터 규모, 평가 지표 등)로 인해 비교 불가능한 수치가 포함될 수 있습니다. 연구 방향과 방법론적 트렌드 비교에 초점을 맞춥니다.

#### 2020년 이후 주요 VM 연구 방향 및 논문

| 연구 방향 | 대표적 방법 | 이 논문과의 관계 |
|-----------|------------|----------------|
| **딥러닝 기반 VM** | LSTM, Transformer, CNN 등 시계열 센서 데이터 직접 처리 | 이 논문의 수동 특징 공학 한계를 자동 특징 추출로 극복 시도 |
| **가우시안 프로세스(GP) 기반 VM** | GP with ARD 커널, Deep GP | 이 논문의 베이즈 해석을 완전 GP로 확장, 불확실도 정량화 강화 |
| **페더레이티드 러닝(Federated Learning)** | 공장 간 데이터 공유 없이 모델 협력 학습 | 이 논문의 단일 공장 한계를 다공장으로 확장하는 방향 |
| **전이 학습 및 도메인 적응** | DANN, MMD 기반 적응 | 이 논문의 일반화 한계 해결을 위한 핵심 방향 |
| **물리 정보 신경망(PINN)** | 공정 물리 방정식과 데이터 결합 | 이 논문의 블랙박스 한계를 물리 제약으로 보완 |
| **그래프 신경망(GNN)** | 장비-공정-챔버 관계를 그래프로 모델링 | 이 논문의 트리 구조를 일반 그래프로 확장 |

#### 구체적 비교 분석

**① 딥러닝 vs. 커널 방법**

최근 반도체 VM 연구에서 LSTM, Transformer 기반 방법들이 활발히 제안되고 있습니다. 이들은:
- **장점**: 원시 시계열 센서 데이터에서 자동 특징 추출, 장기 의존성 포착
- **단점**: 대량의 레이블 데이터 필요, 해석 가능성 낮음, 불확실도 정량화 어려움
- **이 논문 대비**: 데이터가 적은 환경(소량 생산)에서는 여전히 커널 방법의 정규화 이점이 유효. 단, 데이터가 충분한 대규모 FAB에서는 딥러닝이 더 유리할 수 있음

**② 가우시안 프로세스 확장**

Sparse GP, Deep GP 등 최신 GP 방법들은 이 논문의 커널 RR과 밀접하게 연관됩니다:
- **이 논문**: 커널 RR + REML → 점 추정 + 불확실도(사후 분산)
- **최신 GP**: 완전 확률론적 추론, 유도점(inducing point)으로 확장성 해결

**③ 멀티태스크 학습의 발전**

이 논문의 멀티태스크 커널 학습 아이디어는 이후:
- **메타 러닝(Meta-Learning)**: 새로운 챔버/공정에 빠르게 적응하는 MAML 등
- **신경 프로세스(Neural Process)**: 멀티태스크 + 불확실도 + 딥러닝 결합
- **계층적 베이즈(Hierarchical Bayes)**: 이 논문의 혼합 효과 모델을 더 깊은 계층으로 확장

#### 이 논문이 앞으로의 연구에 미치는 영향

1. **프레임워크의 범용성**: 이 논문의 트리 계층 구조 + 가산 추정기 아이디어는 반도체 외 다른 복잡 제조 공정(디스플레이, 배터리 등)의 VM에도 적용 가능한 일반적 프레임워크를 제시

2. **확률론적 VM의 중요성 부각**: 단순 점 예측이 아닌 불확실도 정량화의 필요성을 강조. 이후 GP 기반, 베이즈 딥러닝 기반 VM 연구의 이론적 토대

3. **이질성 처리의 패러다임**: 데이터 이질성을 "문제"가 아닌 "구조화된 정보"로 활용하는 관점 제시 → 이후 도메인 일반화, 전이 학습 연구의 동기 부여

#### 앞으로 연구 시 고려할 점

| 고려 사항 | 세부 내용 |
|-----------|----------|
| **확장성(Scalability)** | 현대 FAB의 수만~수십만 웨이퍼 데이터에 대응하는 희소 커널 또는 딥러닝 기반 방법 통합 필요 |
| **자동화** | 트리 구조 설계, 입력 변수 할당, 커널 선택의 자동화 → AutoML과 결합 |
| **실시간성** | 배치 REML 최적화를 온라인 변분 추론(variational inference)으로 대체 |
| **해석 가능성** | 어떤 공정 변수가 두께에 영향을 미치는지 SHAP, LIME 등으로 설명 가능성 추가 |
| **다공장 적용** | 페더레이티드 러닝 또는 전이 학습으로 다공장 데이터를 프라이버시 보호 하에 활용 |
| **물리 지식 통합** | CVD 공정의 박막 성장 물리 모델을 제약 조건 또는 사전 분포로 활용하는 물리 정보 ML |
| **다변량 품질 지표** | 단일 두께에서 복수의 품질 파라미터(비저항, 결함, 응력 등) 동시 예측 |
| **이상 탐지 통합** | VM 예측 불확실도를 활용한 공정 이상 탐지(anomaly detection) 기능 추가 |

---

## 📚 참고 자료

**논문 내 인용 문헌 (직접 참조):**
1. Schirru, A., Pampuri, S., De Luca, C., De Nicolao, G. (2011). "Multilevel Kernel Methods for Virtual Metrology in Semiconductor Manufacturing." *Preprints of the 18th IFAC World Congress*, Milano, pp. 11614–11621.
2. Shawe-Taylor, J. and Cristianini, N. (2004). *Kernel Methods for Pattern Analysis*. Cambridge University Press.
3. Caruana, R. (1997). "Multitask learning." *Machine Learning*, 28(1), 41–75.
4. Harville, D. (1977). "Maximum likelihood approaches to variance component estimation." *Journal of the American Statistical Association*, 72(358), 320–338.
5. Tikhonov, A. (1943). "On the stability of inverse problems." *CR Acad. Sci. URSS*, 39, 176–179.
6. Aizerman, M., Braverman, E., Rozonoèr, L. (1964). "Theoretical foundations of the potential function method." *Automation and Remote Control*, 25(6), 821–837.
7. Wolfinger, R., Tobias, R., Sall, J. (1994). "Computing Gaussian likelihoods for general linear mixed models." *SIAM Journal on Scientific Computing*, 15, 1294.
8. Muller, K. et al. (2001). "An introduction to kernel-based learning algorithms." *IEEE Transactions on Neural Networks*, 12(2), 181–201.
9. Khan, A., Moyne, J., Tilbury, D. (2008). "Virtual metrology using recursive PLS." *Journal of Process Control*, 18(10), 961–974.
10. Kang, P. et al. (2010). "Virtual Metrology for Run-to-Run Control." *Expert Systems with Applications*.

**2020년 이후 관련 연구 방향 참고 (일반적 연구 동향 기반, 특정 논문 제목 확정 불가):**
- 가우시안 프로세스 기반 VM: GP-VM 관련 연구들 (*IEEE Transactions on Semiconductor Manufacturing*, *Journal of Process Control* 등)
- 딥러닝 기반 반도체 VM: LSTM, Transformer 적용 연구 (*IEEE Transactions on Semiconductor Manufacturing*)
- 페더레이티드 러닝 제조 적용: *Nature Machine Intelligence*, *IISE Transactions* 관련 연구

> ⚠️ 2020년 이후 특정 논문의 제목, 저자, 수치를 구체적으로 인용하는 것은 오류 가능성이 있어 일반적 연구 방향만 제시하였습니다. 정확한 최신 문헌은 IEEE Xplore, Google Scholar에서 "virtual metrology semiconductor deep learning" 등으로 검색을 권장합니다.
