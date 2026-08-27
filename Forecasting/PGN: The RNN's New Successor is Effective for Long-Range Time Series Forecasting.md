# PGN: The RNN's New Successor is Effective for Long-Range Time Series Forecasting

---

## ⚠️ 사전 고지

본 논문은 arXiv:2409.17703v1 (NeurIPS 2024 채택)을 기반으로 분석하였습니다. 논문 원문에서 직접 확인 가능한 내용만을 기술하며, 불확실한 내용은 명시적으로 표기합니다.

---

## 1. Executive Summary (10문장 이내)

본 논문은 RNN의 순환 구조(recurrent structure)가 야기하는 긴 정보 전파 경로, 기울기 소실/폭발, 비효율적 순차 실행 문제를 근본적으로 해결하기 위해 **Parallel Gated Network(PGN)**을 제안한다.  
PGN은 Historical Information Extraction(HIE) 레이어를 통해 이전 시간 단계의 정보를 $\mathcal{O}(1)$의 경로로 직접 추출하고, 게이트 메커니즘으로 현재 정보와 융합한다.  
1D 모델링의 한계로 PGN 단독으로는 주기적 패턴 포착이 어려워, 저자들은 이를 확장한 **Temporal PGN(TPGN)** 프레임워크를 제안한다.  
TPGN은 입력 시계열을 1D→2D로 변환하여 장기 주기 정보와 단기 국소 정보를 두 브랜치로 분리 모델링한다.  
장기 브랜치는 PGN을 2D 입력의 열(column) 방향으로 적용하고, 단기 브랜치는 패치(patch) 기반 선형 레이어로 전역 표현을 추출한다.  
TPGN의 이론적 복잡도는 $\mathcal{O}(\sqrt{L})$로, 효율성과 성능을 동시에 달성한다.  
5개 벤치마크 데이터셋(ECL, Traffic, ETTh1, ETTh2, Weather) 실험에서 TPGN은 이전 최고 모델 대비 평균 MSE 12.35%, MAE 7.25% 개선을 달성하였다.  
Ablation study를 통해 두 브랜치 모두 필요하며, GRU/LSTM 대비 단일 게이트의 PGN이 더 우수함을 검증하였다.  
본 논문은 시간적 모델링(temporal modeling)에 집중하며, 다변수 간 관계 모델링은 향후 과제로 남긴다.

---

### 1-1. 연구의 목적과 필요성

**배경:** 장거리 시계열 예측은 에너지, 기후, 교통 등 다양한 분야에서 중요하나, 기존 RNN 기반 방법은 세 가지 근본적 문제를 가진다 (Introduction, p.2):

1. **긴 정보 전파 경로** $\mathcal{O}(L)$: 장기 의존성 포착 어려움
2. **기울기 소실/폭발(Gradient Vanishing/Exploding)**: 학습 불안정
3. **순차적 연산**: 이론적 복잡도 $\mathcal{O}(L)$임에도 실제 속도가 $\mathcal{O}(L^2)$의 Vanilla-Transformer보다 느릴 수 있음

> 💡 **기울기 소실/폭발(Gradient Vanishing/Exploding)**: 역전파(backpropagation) 과정에서 긴 시퀀스를 처리할 때 기울기값이 0에 수렴(소실)하거나 무한대로 발산(폭발)하는 현상. RNN에서 특히 심각하게 발생하여 학습을 어렵게 만든다.

**필요성:** 저자들은 "더 짧은 역사 입력으로 더 긴 미래를 예측하는 것"을 목표로 설정하며, 이를 위해 정보 전파 경로를 $\mathcal{O}(1)$으로 단축하는 새로운 패러다임이 필요하다고 주장한다.

---

## 2. 핵심 주장과 근거 표

| 핵심 주장 | 근거 | 위치 |
|-----------|------|-------|
| PGN은 정보 전파 경로를 $\mathcal{O}(1)$으로 줄인다 | HIE 레이어가 각 타임스텝에서 전체 이력을 병렬로 집약 | Section 3.1, p.4 |
| PGN은 RNN과 동일한 이론적 복잡도 $\mathcal{O}(L)$을 가지나 실제 속도가 빠르다 | 병렬 연산 가능, 순차 처리 불필요 | Section 3.3, p.6 |
| TPGN의 복잡도는 $\mathcal{O}(\sqrt{L})$ | $R \times P = L$이므로 각 브랜치 복잡도가 $\mathcal{O}(\sqrt{L})$ | Section 3.3, p.7 |
| TPGN이 SOTA 달성 | 5개 데이터셋 전 태스크에서 최고 성능, 평균 MSE 12.35% 개선 | Table 1, p.7 |
| 두 브랜치 모두 필요하다 | TPGN-long/TPGN-short 단독 사용 시 성능 저하 | Table 2, p.8 |
| PGN이 GRU/LSTM보다 우수 | TPGN-GRU/-LSTM 대비 TPGN이 전반적으로 더 낮은 MSE | Table 2, p.8 |
| TPGN은 예측 길이가 길어질수록 성능 저하가 완만하다 | Figure 3-8에서 타 모델 대비 곡선 기울기 완만 | Figure 3, p.8 |
| TPGN은 노이즈에 강건하다 | 노이즈 비율 10%까지 성능 소폭 저하 | Table 5, p.17 |

---

## 2-1. 상세 설명

### 해결하고자 하는 문제 (Section 1, p.1-2)

RNN의 순환 구조로 인한 세 가지 핵심 한계:
- **정보 전파 경로** $\mathcal{O}(L)$: 시퀀스 길이 $L$에 비례하여 정보 손실 증가
- **기울기 문제**: Pascanu et al. (2013)이 지적한 학습 불안정성
- **비효율적 순차 연산**: 병렬화 불가

---

### 제안하는 방법 및 수식

#### (1) PGN 수식 (Equation 1, p.4)

입력 신호 $X \in \mathbb{R}^L$ (길이 $L$)에 대해:

$$H = \text{HIE}(\text{Padding}(X))$$

$$G = \sigma(W_g[X, H] + b_g)$$

$$\hat{H} = \tanh(W_t[X, H] + b_t)$$

$$Out = G \odot H + (1 - G) \odot \hat{H}$$

**기호 설명:**
- $X \in \mathbb{R}^L$: 길이 $L$의 입력 신호
- $\text{Padding}(\cdot)$: 시퀀스 앞에 크기 $\mathbb{R}^{(L-1)}$의 영벡터(zero vector)를 채우는 연산
- $\text{HIE}(\cdot)$: 가중치 행렬 $W_h \in \mathbb{R}^{d_m \times (L-1)}$, 편향 $b_h \in \mathbb{R}^{d_m}$을 가지는 선형 레이어
- $H \in \mathbb{R}^{L \times d_m}$: HIE 레이어 출력 (각 타임스텝의 역사 정보 집약)
- $W_g, W_t \in \mathbb{R}^{d_m \times (d_m+1)}$: 게이트 메커니즘의 가중치 행렬
- $b_g, b_t \in \mathbb{R}^{d_m}$: 편향 벡터
- $G$: 게이트 값 (정보 선택 비율 결정)
- $\hat{H}$: 변환된 역사 정보
- $\odot$: 원소별 곱(element-wise product, Hadamard product)
- $\sigma(\cdot)$: 시그모이드 활성화 함수 (출력 범위 [0,1])
- $\tanh(\cdot)$: 쌍곡탄젠트 활성화 함수 (출력 범위 [-1,1])
- $Out \in \mathbb{R}^{L \times d_m}$: PGN 최종 출력
- $d_m$: 히든 유닛(hidden unit) 수 (모델 차원)

> 💡 **HIE (Historical Information Extraction) 레이어**: 슬라이딩 윈도우 방식으로 선형 레이어를 적용하여 각 타임스텝에서 이전 모든 타임스텝의 정보를 한 번에 병렬로 집약하는 구조. RNN의 순차적 전달 대신 직접 접근으로 $\mathcal{O}(1)$ 경로를 구현.

> 💡 **게이트 메커니즘(Gated Mechanism)**: $G$값이 1에 가까울수록 역사 정보 $H$를, 0에 가까울수록 변환된 정보 $\hat{H}$를 선택. LSTM의 여러 게이트와 달리 PGN은 단 하나의 게이트로 이를 처리.

---

#### (2) TPGN 입력 준비 모듈 수식 (Equation 2, p.5)

$$\mu_X = \frac{1}{L_h} \sum_{i=1}^{L_h} x_i, \quad \sigma^2_X = \frac{1}{L_h} \sum_{i=1}^{L_h} (x_i - \mu_X)^2$$

$$X_{1D}^{norm} = \begin{cases} X_{1D}, & norm = 0 \\ (X_{1D} - \mu_X)/\sigma_X, & norm = 1 \end{cases}$$

$$X_{2D} = \text{Reshape}([X_{1D}^{norm},\ TF_{enc}])$$

**기호 설명:**
- $X_{1D} = \{x_1, x_2, \ldots, x_{L_h}\} \in \mathbb{R}^{L_h \times C}$: 길이 $L_h$, $C$개 변수의 입력 시퀀스
- $TF_{enc} \in \mathbb{R}^{L_h \times C_{time}}$: 시간적 외부 특징(temporal external feature, 예: 시간/요일 등), $C_{time}$개 특징
- $\mu_X$: 시간축 평균
- $\sigma^2_X$: 시간축 분산
- $X_{1D}^{norm} \in \mathbb{R}^{L_h \times C}$: 정규화된 시계열
- $norm \in \{0, 1\}$: 정규화 여부 하이퍼파라미터 (데이터셋 특성에 따라 결정)
- $[\cdot]$: 연결(concatenation) 연산
- $\text{Reshape}(\cdot)$: 자연 주기 $P$에 따라 1D → 2D로 변환
- $X_{2D} \in \mathbb{R}^{R \times P \times C \times (1+C_{time})}$: 2D 변환 출력
- $R$: 2D 입력의 행(row) 수, $P$: 열(column) 수 (자연 주기)
- $R \times P = L_h$ (행과 열의 곱이 입력 길이)

> 💡 **1D→2D 변환(Reshape)**: 시계열을 주기 $P$에 따라 2차원 행렬로 재배열. 행(row) 방향은 단기 변화, 열(column) 방향은 장기 주기 패턴을 포착하는 구조를 만든다.

---

#### (3) 장기 정보 추출 브랜치 (Equation 3, p.6)

$$X_{long}^m = \text{PGN}(X_{2D}^m), \quad H_{long}^m = \text{Linear}_{long}(X_{long}^m)$$

**기호 설명:**
- $X_{2D}^m \in \mathbb{R}^{R \times P \times (1+C_{time})}$: 변수 $m$의 2D 입력
- $\text{PGN}(\cdot)$: $R$ 차원(행 방향)으로 PGN 적용
- $X_{long}^m \in \mathbb{R}^{R \times P \times d_m}$: PGN 출력
- $\text{Linear}_{long}(\cdot)$: 모든 행의 정보를 집약하는 선형 레이어
- $H_{long}^m \in \mathbb{R}^{P \times d_m}$: 장기 브랜치 최종 출력

---

#### (4) 단기 정보 추출 브랜치 (Equation 4, p.6)

$$H_{short}^m = \text{Linear}_{short}^{row}(X_{2D}^m), \quad H_{global}^m = \text{Linear}_{short}^{col}(H_{short}^m)$$

**기호 설명:**
- $\text{Linear}_{short}^{row}(\cdot)$: $P$ 차원(열 방향)으로 단기 정보를 패치로 집약
- $H_{short}^m \in \mathbb{R}^{R \times d_m}$: 패치 집약 결과
- $\text{Linear}_{short}^{col}(\cdot)$: 패치들을 전역 표현으로 추가 집약
- $H_{global}^m \in \mathbb{R}^{1 \times d_m}$: 시퀀스의 전역(global) 표현

> 💡 **패치(Patch)**: 연속된 여러 타임스텝을 하나의 단위로 묶는 방법. 로컬 단기 정보를 효율적으로 집약하며, PatchTST(Nie et al., 2023)에서 도입된 개념.

---

#### (5) 예측 모듈 (Equation 5, p.6)

$$Out^m = \text{Reshape}(\text{Linear}([H_{long}^m,\ H_{global}^m]))$$

**기호 설명:**
- $[\cdot]$: 두 브랜치 출력의 연결 연산
- $\text{Linear}(\cdot)$: 출력 차원 $\mathbb{R}^{P \times R_f}$ ($R_f \times P = L_f$)
- $L_f$: 예측 시계열 길이
- $Out^m \in \mathbb{R}^{L_f}$: 변수 $m$의 최종 예측값

---

### 모델 구조 (Figure 2, p.5)

```
입력 X_1D (L_h × C)
    ↓
[Input Preparation Module]
  - 정규화 (선택적)
  - TF_enc 연결
  - Reshape: 1D → 2D (R × P)
    ↓
┌─────────────────────────────────────────┐
│           TPGN Framework                │
│  ┌──────────────────────────────────┐   │
│  │  Long-term Branch                │   │
│  │  PGN(R방향) → Linear_long        │   │
│  │  출력: H^m_long (P × d_m)        │   │
│  └──────────────────────────────────┘   │
│  ┌──────────────────────────────────┐   │
│  │  Short-term Branch               │   │
│  │  Linear^row → Linear^col         │   │
│  │  출력: H^m_global (1 × d_m)      │   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
    ↓
[Forecasting Module]
  - Concat([H_long, H_global])
  - Linear → Reshape
    ↓
출력 Out^m (L_f)
```

---

### 성능 향상 (Table 1, p.7; Table 2, p.8)

| 데이터셋 | 평균 MSE 개선 (vs. 최선 기준 모델) |
|----------|-----------------------------------|
| ECL | 17.31% |
| Traffic | 9.38% |
| ETTh1 | 3.79% |
| ETTh2 | 12.26% |
| Weather | 19.09% |
| **전체 평균** | **12.35%** |

개별 모델 대비 TPGN의 평균 MSE 개선 범위: **14.08% ~ 37.44%** (Table 1 마지막 행)

---

### 한계 (Appendix A, p.12)

- **다변수 간 관계 모델링 미지원**: 시간적 차원(temporal dimension)에만 집중하며, 변수 간 상관관계 모델링은 별도로 다루지 않음
- **주기 결정의 수동성**: 자연 주기 $P$를 사전에 수동 설정해야 하며, 자동 탐색 미지원
- **단일 변수 실험 설정**: 다변수 데이터에서 단일 변수만 추출하여 실험 (일반화 한계)
- **norm 하이퍼파라미터**: 데이터셋 특성에 따라 수동 결정 필요

---

## 3. 각 주장별 위치 표시

| 주장 | 위치 |
|------|------|
| RNN의 세 가지 한계 | Section 1, p.1-2 |
| PGN이 정보 전파 경로를 $\mathcal{O}(1)$으로 줄임 | Section 3.1, p.4; Figure 1(l), p.2 |
| TPGN의 복잡도 $\mathcal{O}(\sqrt{L})$ | Section 3.3, p.6-7; Table 3, p.13 |
| TPGN이 모든 태스크에서 SOTA | Table 1, p.7 |
| 두 브랜치의 상보적 역할 검증 | Table 2, p.8 |
| PGN > GRU/LSTM | Table 2, p.8, Section 4.2, p.9 |
| 예측 길이 증가 시 성능 저하 완만 | Figure 3-8, p.8, 15-16 |
| 노이즈 강건성 | Table 5, p.17 |
| 효율성 비교 | Figure 4, p.9 |
| 다변수 모델링 미지원 (한계) | Appendix A, p.12 |

---

## 4. 저자 직접 보고 vs. 해석 분리

### 4-1. 저자가 직접 보고한 결과

**연구 주제:**
> "PGN을 RNN의 후계자(new successor)로 제안하고, 이를 기반으로 한 TPGN이 장거리 시계열 예측에서 SOTA를 달성함" (Abstract, p.1)

**방법:**
- PGN: HIE 레이어 + 단일 게이트 메커니즘, 정보 전파 경로 $\mathcal{O}(1)$
- TPGN: 두 브랜치 구조, 복잡도 $\mathcal{O}(\sqrt{L})$
- 수식 (1)-(5)로 형식화 (Section 3, p.4-6)

**결과 (Table 1, p.7):**
- 전체 평균 MSE 12.35%, MAE 7.25% 개선
- ECL 17.31%, Traffic 9.38%, ETTh1 3.79%, ETTh2 12.26%, Weather 19.09% MSE 개선
- 5회 반복 실험 평균값, 오차 막대 Table 4에 제시

**효율성 (Figure 4, p.9):**
- TPGN이 가장 낮은 시간/메모리 오버헤드는 아니나 "decent level"이라고 보고

---

### 4-2. 분석자의 해석

**긍정적 측면:**
- HIE 레이어는 본질적으로 **convolution과 유사한 슬라이딩 선형 연산**으로 볼 수 있으며, 이것이 병렬화를 가능하게 하는 핵심이다. 이는 RNN과의 차별화 포인트이나, 동시에 기존 TCN 계열과의 본질적 차이를 명확히 설명할 필요가 있다.
- 단일 게이트 PGN이 다중 게이트 GRU/LSTM보다 우수한 결과는, 게이트 수보다 **정보 전파 경로의 단축**이 성능에 더 중요함을 시사한다.
- Weather 데이터셋에서 19.09%의 높은 개선은, 날씨 데이터의 뚜렷한 주기성을 2D 변환이 효과적으로 포착하기 때문으로 해석된다.

**비판적 측면:**
- 단일 변수(univariate) 설정 실험은 실제 다변수 예측 시나리오와 괴리가 있으며, 결과의 일반화 가능성이 제한된다.
- ETTh1에서의 3.79% 개선은 다른 데이터셋 대비 상대적으로 작아, 데이터 특성(낮은 주기성)에 따라 TPGN의 이점이 가변적일 수 있다.
- "PGN이 RNN의 후계자"라는 강한 주장에 비해, 실제 다양한 도메인(NLP, 음성 등)에서의 검증은 부재하다.

---

## 5. 통계적 취약점 및 비교 불가능한 수치

### 5-1. ⚠️ 통계적으로 취약한 부분

| 항목 | 문제점 |
|------|--------|
| **단일 변수 실험** | 다변수 데이터에서 대표 단일 변수만 추출 (ECL: MT_320, Traffic: Node_862 등). 전체 데이터셋 대표성 의문 |
| **5회 반복 실험** (Table 4) | 실험 횟수가 통계적 유의성 검증(예: t-test, p-value)에 충분하지 않을 수 있음; 유의성 검정 결과 미보고 |
| **TPGN-short의 극단적 성능 저하** | ECL에서 MSE 0.7226 (Table 2) - 단기 브랜치만 사용 시 성능이 매우 나빠, 설계 균형이 장기 브랜치에 편중되어 있음 |
| **ETTh1 3.79% 개선** | 다른 데이터셋 대비 개선폭이 작아, 특정 데이터 특성에 민감할 가능성 |
| **norm 하이퍼파라미터** | 수동으로 결정하며, Table 6의 분산 비교 기준이 "약 2배"로 다소 자의적 |

### 5-2. ⚠️ 비교 불가능한 수치

| 항목 | 문제점 |
|------|--------|
| **효율성 비교 (Figure 4)** | TimesNet이 그래프에서 제외됨. "overhead가 너무 높아 제외"라고 설명하나, 이로 인해 비교 그룹이 선택적으로 구성됨 |
| **FiLM 시간 비교 제외** | "FiLM is not included in the time comparison chart" — 이유 불명확 |
| **TPGN-GRU/-LSTM 비교** | PGN 내부 구조를 GRU/LSTM으로 대체하는 것이 완전히 공정한 비교인지 불명확 (아키텍처 전반적 최적화 여부) |
| **"average improvement 12.35%"** | 어떤 기준 모델(per-task best)을 기준으로 하는지에 따라 수치가 크게 변할 수 있음 |
| **복잡도 $\mathcal{O}(\sqrt{L})$ 주장** | $R = P = \sqrt{L}$ 가정 하의 이론값이며, 실제 $R$과 $P$가 데이터의 자연 주기에 따라 결정되므로 항상 $\sqrt{L}$이 보장되지는 않음 |

---

## 6. 논문이 답하지 않는 질문

1. **다변수 시계열에서의 성능**: 논문은 단일 변수 설정만 실험하며, iTransformer, CrossGNN 등 다변수 모델과의 공정한 비교가 부재하다.

2. **자연 주기 $P$의 자동 결정 방법**: $P$를 어떻게 선택하는지 명확한 기준이 부족하며, 부적절한 $P$ 선택 시 성능 영향이 정량화되지 않았다.

3. **HIE 레이어의 이론적 표현력(expressiveness)**: HIE가 선형 레이어임을 고려할 때, 복잡한 비선형 시간 패턴을 포착하는 데 이론적 한계는 무엇인가?

4. **다른 도메인(NLP, 음성 등)에서의 적용 가능성**: "RNN의 후계자"라고 주장하나 시계열 예측 외 영역에서의 검증이 없다.

5. **PGN의 장기 의존성 포착 메커니즘**: HIE가 실제로 긴 의존성을 "이해"하는지, 단순히 이전 값들을 선형 조합하는 데 그치는지에 대한 해석 가능성(interpretability) 분석이 없다.

6. **배포 환경(distribution shift) 문제**: Train/Test 분포 차이가 클 때의 성능 저하에 대한 분석이 없다.

7. **다중 주기(multi-periodicity) 데이터**: 여러 주기가 혼재하는 데이터에서의 성능 (단일 $P$만 사용).

8. **모델 레이어 수 증가 시 성능/효율 트레이드오프**: 실험은 단일 레이어(single-layer) 기준으로 효율성 비교.

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p.2): 다양한 모델의 정보 전파 비교도

**해석:**
이 그림은 RNN(a), CNN(b,c), WITRAN(d), TimesNet(e), MICN(f), ModernTCN(g), Transformer(h), PatchTST(i), PDF(j), iTransformer(k), **PGN(l)**, **TPGN(m)** 총 13개 모델의 정보 전파 과정을 시각화한다.

- **RNN(a)**: 순차적 연결로 정보가 한 단계씩 전달 → 경로 $\mathcal{O}(L)$, 마지막 타임스텝까지 도달하기 위한 최대 경로 가장 길다
- **PGN(l)**: HIE 레이어가 각 타임스텝에서 직접 전체 이력을 집약 → 경로 $\mathcal{O}(1)$로 최단화, 병렬 처리 화살표 표시
- **TPGN(m)**: 두 브랜치(장기/단기) 분리. 장기 브랜치는 2D 입력의 열 방향으로 PGN 적용, 단기 브랜치는 패치 기반 집약
- **색상 진하기**: 더 진한 색은 더 많은 정보를 포함함을 나타내며, TPGN의 각 브랜치 출력이 정보를 효과적으로 집약함을 보여준다

**의의**: 본 논문의 핵심 동기와 PGN/TPGN의 설계 철학을 한 그림으로 설명. 모든 모델 중 TPGN만이 $\mathcal{O}(1)$ 전파 경로 + 2D 주기 정보 포착 + 병렬 연산을 동시에 달성함을 시각적으로 주장.

---

### Figure 2 (p.5): PGN과 TPGN의 구조도

**해석:**

**(a) PGN 구조:**
- 입력 $X$에서 Padding 후 HIE 레이어를 통해 $H$(역사 정보) 추출
- $X$와 $H$를 concat하여 sigmoid($G$)와 tanh($\hat{H}$)로 분리
- $G \odot H + (1-G) \odot \hat{H}$의 게이트 출력으로 최종 $Out$ 생성
- 핵심: **모든 타임스텝이 동시에 병렬 처리** (화살표가 수평으로 배치)

**(b) TPGN 전체 구조:**
- Input Preparation Module: Norm + Reshape (1D→2D)
- 장기 브랜치: PGN → Linear_long
- 단기 브랜치: Linear_short^row → Linear_short^col
- Forecasting Module: Concat → Linear → Reshape

**의의**: PGN이 단순한 선형 레이어(HIE)와 단일 게이트만으로 구성된 경량(lightweight) 구조임을 보여주며, TPGN이 두 브랜치의 출력을 후처리하여 예측하는 간결한 엔드-투-엔드 프레임워크임을 확인할 수 있다.

---

### Figure 3 (p.8): ECL 데이터셋에서 예측 길이별 성능 비교

**해석:**
- X축: 예측 길이 (168 → 1440), Y축: MSE/MAE
- **TPGN(빨간선)**: 전 구간에서 가장 낮은 MSE/MAE 유지, 그래프 기울기가 타 모델 대비 완만함
- **TimesNet(초록선)**: 예측 길이 증가에 따라 MSE가 급격히 상승 (168→1440: 0.28→0.67)
- **WITRAN(파란선)**: 성능 저하 추세가 TPGN보다 가파름
- **TimeMixer, iTransformer**: 중간 구간(336-720)에서 일부 교차하나 1440에서는 모두 TPGN보다 열세

**의의**: TPGN의 핵심 강점인 "긴 예측 길이에서의 안정성"을 시각적으로 증명. 특히 1440 예측에서 TPGN(0.2484) vs. 다른 모델들(0.32~0.90)의 격차가 두드러진다. 이는 TPGN의 $\mathcal{O}(1)$ 정보 전파와 주기 포착 능력이 장기 예측에서 특히 효과적임을 지지한다.

---

### Table 3 (p.13): 모델별 강점, 복잡도, 정보 전파 경로 비교

**해석 (Table 3의 주요 내용):**

| 모델 | 비점별 의미론적 정보 포착 | 주기 정보 직접 포착 | 최대 전파 경로 | 복잡도 | 병렬 처리 |
|------|------------------------|---------------------|---------------|--------|-----------|
| RNN | ✓ | ✗ | $\mathcal{O}(L)$ | $\mathcal{O}(L)$ | ✗ |
| WITRAN | ✓ | ✓(2D) | $\mathcal{O}(\sqrt{L})$ | $\mathcal{O}(\sqrt{L})$ | ✓– |
| Transformer | ✗ | ✗ | $\mathcal{O}(1)$ | $\mathcal{O}(L^2)$ | ✓ |
| PatchTST | ✓ | ✗ | $\mathcal{O}(1)$ | $\mathcal{O}((L/S)^2)$ | ✓ |
| **PGN (ours)** | **✓** | **✗** | **$\mathcal{O}(1)$** | **$\mathcal{O}(L)$** | **✓** |
| **TPGN (ours)** | **✓** | **✓(2D)** | **$\mathcal{O}(1)$** | **$\mathcal{O}(\sqrt{L})$** | **✓** |

**의의**: TPGN이 네 가지 기준 모두에서 가장 유리한 특성을 가지는 유일한 모델임을 표로 정리. 특히 $\mathcal{O}(1)$ 전파 경로 + $\mathcal{O}(\sqrt{L})$ 복잡도 + 2D 주기 포착 + 병렬 처리의 조합은 다른 모델에서 달성되지 않음. 그러나 이 표는 저자들이 자신의 방법에 유리하게 기준을 선택하였을 가능성이 있음에 주의.

---

### Figure 4 (p.9): 시간/메모리 오버헤드 비교

**해석:**
- 4개 서브플롯: (a) 예측 길이 변화 시 시간, (b) 예측 길이 변화 시 메모리, (c) 입력 길이 변화 시 시간, (d) 입력 길이 변화 시 메모리
- TPGN은 중간 수준의 시간/메모리 오버헤드를 보임
- DLinear가 가장 낮은 오버헤드를 가지나 성능은 열세
- TimesNet은 오버헤드가 매우 높아 그래프에서 제외됨 (⚠️ 선택적 비교)
- TPGN은 단일 레이어(1-layer) 구성임을 강조 — 타 모델은 다중 레이어 사용 가능

**의의**: TPGN이 효율성과 성능의 균형을 달성함을 보여주나, **TimesNet 제외**와 **단일 레이어 비교 조건**이 결과를 TPGN에 유리하게 만들 수 있다. 실제 동일 성능을 내기 위한 레이어 수를 고정한 비교가 더 공정할 것이다.

---

## 8. 결론 및 시사점

### 8-1. 저자가 제시한 시사점 및 후속 연구 계획 (Section 5, Appendix A, p.9, 12)

**시사점:**
- PGN이 RNN 구조의 근본 한계를 해결하는 새로운 패러다임임을 실증
- TPGN 프레임워크는 PGN 외 다른 모델(GRU, LSTM, MLP)로 대체 가능한 **범용 프레임워크**임을 입증
- 장기 예측에서 주기 정보 모델링이 성능 향상의 핵심임을 확인

**저자가 제시한 후속 연구:**
1. **다변수 관계 모델링 통합**: 변수 간 상관관계를 모델링하는 컴포넌트를 TPGN에 추가
2. **PGN 패러다임의 확장 적용**: 다른 시계열 분석 태스크 및 NLP 등 타 분야에서의 RNN 대체 검토

---

### 8-1. 모델의 일반화 성능 향상 가능성

현재 TPGN의 일반화 한계와 향상 가능성을 다음 관점에서 분석한다:

**현재 일반화 제약:**
1. **단일 변수 설정**: 다변수 데이터의 이종성(heterogeneity)이 성능에 부정적 영향을 줄 수 있어 단일 변수로 실험. 이는 실제 응용에서의 일반화를 제한한다.
2. **고정된 자연 주기 $P$**: 데이터마다 사전 설정 필요. 주기가 명확하지 않거나 다중 주기를 가지는 데이터에서 일반화 어려움.
3. **데이터셋 다양성 제한**: 5개 데이터셋 모두 에너지/교통/날씨 도메인으로, 금융, 의료, 제조 등 타 도메인에서의 일반화 미검증.

**일반화 향상 가능성:**

**(a) 다변수 확장:**
iTransformer [Liu et al., 2024]나 CrossGNN [Huang et al., 2023]의 변수 모델링 컴포넌트를 TPGN에 통합하면 다변수 시나리오에서의 일반화가 향상될 것이다. 저자들도 Appendix A에서 이를 명시적으로 언급.

**(b) 적응형 주기 탐색:**
TimesNet [Wu et al., 2023]처럼 FFT 기반으로 주기 $P$를 자동 탐색하거나, 학습 가능한(learnable) 주기 파라미터를 도입하면 다양한 데이터에 대한 일반화가 향상된다.

**(c) Pre-training 및 Fine-tuning:**
최근 시계열 foundation model 연구(예: TimesFM, MOIRAI 등)처럼 대규모 사전 학습 후 파인튜닝하는 방식을 PGN에 적용하면, 적은 데이터로도 높은 일반화 성능을 기대할 수 있다.

**(d) 분포 변화(Distribution Shift) 대응:**
학습-테스트 분포 차이가 클 때를 대비하여, 온라인 학습(online learning) 또는 도메인 적응(domain adaptation) 기법을 통합하면 실제 환경에서의 일반화가 향상된다.

**(e) 불규칙 시계열 지원:**
현재 TPGN은 균일 샘플링(uniform sampling) 기반. 결측값이나 불규칙 타임스텝에 대한 처리 메커니즘 추가가 필요하다.

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ 아래 분석은 논문 원문 내 인용 정보와 공개된 관련 연구를 기반으로 하며, 본 논문(2024년 9월 arXiv 제출) 이후 발표된 일부 연구는 저의 학습 데이터 범위(2024년 초까지)를 기반으로 합니다. 정확한 최신 성능 수치는 각 논문을 직접 확인하시기 바랍니다.

#### 주요 관련 연구 비교

| 연구 | 연도 | 패러다임 | 주요 특징 | TPGN과의 비교 |
|------|------|----------|-----------|---------------|
| Informer [Zhou et al.] | 2021 | Transformer | ProbSparse attention, $\mathcal{O}(L\log L)$ | TPGN이 성능/효율 모두 우세 (논문 내 비교) |
| Autoformer [Wu et al.] | 2021 | Transformer | Auto-correlation, 시계열 분해 | TPGN이 전반적 우세 |
| FEDformer [Zhou et al.] | 2022 | Transformer | 주파수 영역 변환 | TPGN 우세 |
| PatchTST [Nie et al.] | 2023 | Transformer | 패치 기반, 채널 독립 | TPGN이 MSE 평균 31.68% 개선 (Table 1) |
| iTransformer [Liu et al.] | 2024 | Transformer | 변수 차원 attention | TPGN이 MSE 평균 21.40% 개선 (Table 1) |
| DLinear [Zeng et al.] | 2023 | MLP | 단순 선형 분해 | TPGN이 MSE 평균 32.75% 개선 (Table 1) |
| TimesNet [Wu et al.] | 2023 | CNN | 1D→2D 변환 | TPGN이 MSE 평균 25.42% 개선 (Table 1) |
| TimeMixer [Wang et al.] | 2024 | MLP | 다중 스케일 혼합 | TPGN이 MSE 평균 36.15% 개선 (Table 1) |
| WITRAN [Jia et al.] | 2023 | RNN | 2D RNN, $\mathcal{O}(\sqrt{L})$ | TPGN이 MSE 평균 14.08% 개선 (Table 1) |
| ModernTCN [Luo & Wang] | 2024 | CNN | 대형 커널 CNN | TPGN이 MSE 평균 8.12% 개선 (Table 1) |
| PDF [Dai et al.] | 2024 | Transformer | 주기성 분리, 2D | TPGN이 MSE 평균 28.01% 개선 (Table 1) |

#### 동향 분석

**2020-2021: Transformer 지배기**
Informer, Autoformer, FEDformer가 등장하며 Transformer가 시계열 예측을 주도. 그러나 점별(point-wise) attention의 의미론적 정보 포착 한계가 지속 지적됨.

**2022-2023: Transformer 회의론 및 단순 모델의 부상**
DLinear [Zeng et al., 2023]이 "Are Transformers Effective for Time Series Forecasting?"에서 단순 선형 모델이 Transformer를 능가할 수 있음을 보여, 복잡성-성능 트레이드오프에 대한 재고를 촉구. PatchTST가 패치 기반 접근으로 Transformer의 한계를 일부 극복.

**2023-2024: 다양한 패러다임 경쟁**
TimesNet(1D→2D CNN), TimeMixer(다중 스케일 MLP), iTransformer(변수 차원 attention), WITRAN(2D RNN), ModernTCN(대형 커널), PDF(주기 분리) 등이 경쟁하며 특정 패러다임이 압도적이지 않음.

**PGN/TPGN의 위치:**
- 2024년 현재, PGN은 RNN의 순차성 문제를 병렬화로 해결한 **최초의 체계적 시도** 중 하나
- TPGN은 1D→2D 변환 + 이중 브랜치로 기존 연구의 장점을 통합

---

#### 앞으로의 연구에 미치는 영향

1. **RNN 재평가 촉진**: PGN은 RNN 계열 연구에 새로운 방향을 제시. 병렬화 가능한 순환 구조의 연구가 활성화될 것으로 예상.

2. **이중 브랜치 프레임워크의 확산**: 장기/단기 정보를 분리 모델링하는 TPGN의 접근법은 향후 범용 프레임워크로 채택될 가능성이 높다.

3. **복잡도-성능 트레이드오프 연구**: $\mathcal{O}(\sqrt{L})$ 복잡도 달성 방법론이 다른 효율적 모델 설계에 영감을 줄 수 있다.

---

#### 앞으로 연구 시 고려할 점

1. **공정한 다변수 비교 설정**: 단일 변수 실험 설정의 한계를 극복하고, 동일 조건에서 다변수 예측 성능 비교 필요.

2. **Foundation Model과의 통합**: TimesFM, MOIRAI 등 대규모 시계열 foundation model에 PGN 패러다임을 통합하거나, 사전 학습된 표현을 활용하는 연구.

3. **해석 가능성(Interpretability) 강화**: HIE 레이어가 어떤 역사 패턴을 학습하는지 시각화하고, 어텐션 가중치에 준하는 해석 도구 개발.

4. **온라인/지속 학습(Continual Learning) 적용**: 배포 환경에서 데이터 분포가 변화할 때 PGN의 성능 유지 방법.

5. **다양한 도메인 검증**: 금융 시계열(주가, 환율), 의료(심전도, 뇌파), 제조(센서 데이터) 등에서의 적용 가능성 검증.

6. **이론적 분석 강화**: PGN의 표현력(expressiveness), VC dimension, 일반화 오차 한계(generalization bound) 등 이론적 기초 연구.

7. **주기 자동 탐색**: 데이터별 수동 설정 $P$를 자동화하여 실용성 향상.

8. **NLP/음성 등 타 시퀀스 태스크**: "RNN의 후계자"라는 주장을 시계열 이외 영역에서도 검증.

---

## 참고 자료 (논문 내 인용 문헌 기반)

- Jia et al. (2023). "WITRAN: Water-wave Information Transmission and Recurrent Acceleration Network for Long-range Time Series Forecasting." NeurIPS 2023.
- Wu et al. (2023). "TimesNet: Temporal 2D-variation Modeling for General Time Series Analysis." ICLR 2023.
- Nie et al. (2023). "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers." ICLR 2023.
- Liu et al. (2024). "iTransformer: Inverted Transformers are Effective for Time Series Forecasting." ICLR 2024.
- Luo & Wang (2024). "ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis." ICLR 2024.
- Wang et al. (2024). "TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting." ICLR 2024.
- Dai et al. (2024). "Periodicity Decoupling Framework for Long-term Series Forecasting." ICLR 2024.
- Xu et al. (2024). "FITS: Modeling Time Series with 10k Parameters." ICLR 2024.
- Zeng et al. (2023). "Are Transformers Effective for Time Series Forecasting?" AAAI 2023.
- Hochreiter & Schmidhuber (1997). "Long Short-Term Memory." Neural Computation.
- Chung et al. (2014). "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling." arXiv:1412.3555.
- Vaswani et al. (2017). "Attention is All You Need." NeurIPS 2017.
- Pascanu et al. (2013). "On the Difficulty of Training Recurrent Neural Networks." ICML 2013.
- Tishby & Zaslavsky (2015). "Deep Learning and the Information Bottleneck Principle." IEEE ITW 2015.
- Zhou et al. (2021). "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting." AAAI 2021.
- 논문 원문: arXiv:2409.17703v1, Yuxin Jia et al., NeurIPS 2024.
- 코드 저장소: https://github.com/Water2sea/TPGN
