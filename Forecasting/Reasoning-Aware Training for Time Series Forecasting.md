# Reasoning-Aware Training for Time Series Forecasting

---

## 📌 참고 자료

**primary source:**
- Ahamed, M. A., Parmar, M., Goyal, P., Li, C.-L., Cheng, Q., Pfister, T., & Yoon, J. (2026). **"Reasoning-Aware Training for Time Series Forecasting"**. arXiv:2605.08625v1 [cs.LG], 9 May 2026.

**논문 내 인용된 핵심 관련 연구 (비교 분석에 활용):**
- Ansari et al. (2024, 2025): Chronos / Chronos-2
- Das et al. (2024): TimesFM
- Woo et al. (2024): MOIRAI
- Liu et al. (2024): iTransformer
- Nie et al. (2023): PatchTST
- Jin et al. (2024): Time-LLM
- Wei et al. (2022): Chain-of-Thought Prompting
- Hsieh et al. (2023): Distilling step-by-step
- Rombach et al. (2022): Latent Diffusion Models
- Tan et al. (2024): "Are language models actually useful for time series forecasting?"
- Ahamed & Cheng (2024): TimeMachine
- Ahamed et al. (2026): TFRBench
- Aksu et al. (2024): GIFT-Eval

---

## 1. 핵심 주장과 주요 기여 요약

### 1.1 핵심 주장

STRIDE(**S**trategic **T**ime-series **R**easoning **I**njected via **D**istilled **E**mbeddings)는 다음 두 가지 근본적 문제를 동시에 해결하고자 한다:

> **"LLM은 수치 예측기가 아닌, 의미론적 추론 생성기로 활용되어야 한다."**

| 기존 접근법 | 한계 |
|---|---|
| TSFM만 사용 | 블랙박스, 정성적 추론 불가 |
| LLM만 사용 | 모달리티 갭, 수치 정밀도 저하, 계산 병목 |
| STRIDE | LLM 추론을 연속 임베딩 공간에 주입하여 양쪽 장점 결합 |

### 1.2 주요 기여 (3가지)

**(i) 새로운 크로스 모달 프레임워크 (Novel Cross-Modal Framework)**
- 이산 토큰 병목 없이 LLM 추론을 TSFM 임베딩 공간에 직접 주입

**(ii) 추론 증류 파이프라인 (Reasoning Distillation Pipeline)**
- 경량 Student LLM이 역사적 맥락만으로 고품질 분석 전략 생성

**(iii) 수치·추론 성능 동시 향상**
- GIFT-Eval: MASE 0.674, CRPS 0.454 (SOTA)
- TFRBench in-domain MASE 0.615 vs. 기존 최강 0.765
- TFRBench out-of-domain MASE 0.724 vs. 기존 최강 0.778

---

## 2. 상세 분석: 문제 → 방법 → 구조 → 성능 → 한계

### 2.1 해결하고자 하는 문제

#### (A) TSFMs의 블랙박스 문제
기존 TSFM들은 조건부 분포를 다음과 같이 근사한다:

$$P(\mathbf{Y} \mid \mathbf{X})$$

여기서 $\mathbf{X} \in \mathbb{R}^{T \times V}$ (역사적 다변량 시계열), $\mathbf{Y} \in \mathbb{R}^{H \times V}$ (미래 예측값). 이 접근은 **정성적 추론, 외부 이벤트(휴일, 경제 충격) 반영이 불가**하다.

#### (B) LLM 직접 적용의 모달리티 갭(Modality Gap)
연속 수치 데이터를 이산 텍스트 토크나이저로 강제 변환 시:
- 수학적 관계 파괴 (e.g., "3.14159" → 여러 토큰으로 분절)
- 시퀀스 길이 폭발 → 계산 병목
- **"LLM 신기루(LLM Mirage)"**: LLM만으로는 TSFM을 의미있게 능가하지 못함 (Tan et al., 2024)

### 2.2 제안하는 방법 (수식 포함)

#### 새로운 예측 목표 재정의

$$P(\mathbf{Y} \mid \mathbf{X}, \mathbf{R})$$

여기서 $\mathbf{R}$은 **명시적 추론 사전(reasoning prior)** 으로, 크로스 채널 동역학, 트렌드 변화, 주기적 이벤트를 구조화된 형태로 포착한다.

---

#### Step 1: 추론 증류 (Reasoning Distillation)

Student LLM의 Cross-Entropy 손실:

$$\mathcal{L}_{CE} = -\sum_{j=1}^{M} \log P(R_{ref,j} \mid P_{in}, R_{ref,<j}; \Theta_{LLM})$$

- $M$: 참조 추론의 총 토큰 수
- $\Theta_{LLM}$: Student LLM의 학습 가능한 파라미터
- $P_{in}$: 역사적 시계열 + 메타데이터 + 베이스라인 추론으로 구성된 입력 프롬프트
- 입력 프롬프트 토큰에 대해서는 손실 마스킹 적용

---

#### Step 2: 크로스 모달 잠재 투영 (Cross-Modal Latent Projection)

Student LLM의 마지막 은닉 상태 $\mathbf{H} \in \mathbb{R}^{L \times d_{llm}}$에서 평균 풀링:

$$\mathbf{h}_R = \frac{1}{L} \sum_{i=1}^{L} \mathbf{H}_i$$

선형 투영으로 TSFM 임베딩 공간으로 변환:

$$\mathbf{e}_R = \mathbf{h}_R \mathbf{W}_{proj}, \quad \mathbf{W}_{proj} \in \mathbb{R}^{d_{llm} \times d_{ts}}$$

---

#### Step 3: 잠재 융합 및 분위수 예측 (Latent Fusion & Quantile Forecasting)

일반화된 융합 연산자:

$$\mathbf{E}_{fused} = \mathbf{e}_R \oplus \mathbf{E}_{TS}$$

- $\mathbf{E}_{TS}$: TSFM의 네이티브 시계열 임베딩
- $\oplus$: 시퀀스 프리픽싱(Chronos-2.0) 또는 초기 상태 치환(Timer-S1)

TSFM 디코더가 $\mathbf{E}_{fused}$를 처리하여 분위수 예측:

$$\hat{\mathbf{Y}}_q = \text{TSFM Decode}(\mathbf{E}_{fused}), \quad Q = \{0.1, 0.2, \ldots, 0.9\}$$

---

#### Step 4: 결합 최적화 목표 (Joint Optimization)

$$\mathcal{L}_{total} = \alpha \mathcal{L}_{CE} + \beta \mathcal{L}_{Quantile}$$

- $\alpha, \beta \in [0, 1]$ (실험에서 $\alpha = \beta = 1$ 사용)
- Student LLM은 LoRA(Low-Rank Adaptation)로만 최적화 (언어 지식 보존)
- 경사 흐름: TSFM 디코더 → 투영 레이어 → Student LLM (end-to-end)

---

### 2.3 모델 구조

```
┌────────────────────────────────────────────────────────────────┐
│                        STRIDE Architecture                     │
├────────────────────────────────────────────────────────────────┤
│  [훈련 단계]                                                    │
│                                                                │
│  역사 시계열 X ──→ Teacher LLM (Gemini-3.1-Pro)               │
│              ↑         + 미래 Y + 외부 이벤트 E               │
│              │    → Reference Reasoning R_ref                  │
│              │                                                │
│  X + 메타데이터 + R_base ──→ [Small-LLM: Gemma-3-4B-it]      │
│  (Large LLM이 생성한 R_base)     ↓ LoRA 파인튜닝              │
│                            Hidden States H ∈ R^{L×d_llm}     │
│                                 ↓ Mean Pooling                │
│                            h_R ∈ R^{d_llm}                   │
│                                 ↓ W_proj                      │
│                            e_R ∈ R^{d_ts}  ──→ L_CE          │
│                                 ↓                             │
│  X ──→ [TSFM Encoder] ──→ E_TS                               │
│              ↓                                                │
│         E_fused = e_R ⊕ E_TS                                  │
│              ↓                                                │
│         [TSFM Decoder] ──→ Ŷ_q ──→ L_Quantile               │
│                                                               │
│  L_total = α·L_CE + β·L_Quantile                              │
├────────────────────────────────────────────────────────────────┤
│  [평가 단계]                                                    │
│  X ──→ Large LLM → R_base → Small-LLM → R̂ (해석 가능 추론)  │
│                                ↓                              │
│                    H → h_R → e_R → E_fused → Ŷ_q (수치 예측) │
└────────────────────────────────────────────────────────────────┘
```

**주요 구성 요소:**
- **Teacher LLM**: Gemini-3.1-Pro (다중 에이전트 시스템으로 오라클 추론 생성, 훈련 시에만)
- **Large LLM (Baseline Provider)**: Gemini-3.1-Pro (베이스라인 추론 초안 생성)
- **Student Small-LLM**: Gemma-3-4B-it (LoRA 적용, 추론 생성 + 은닉 상태 제공)
- **Projection Matrix**: $\mathbf{W}\_{proj} \in \mathbb{R}^{d_{llm} \times d_{ts}}$ (학습 가능)
- **TSFM Backbone**: Chronos-2.0 또는 Timer-S1 (파인튜닝 가능)

---

### 2.4 이론적 기반: 분산 감소 정리 (Theorem 1)

미래 분포가 $K$개의 플로시블 궤적의 혼합이라 가정:

$$P(Y \mid X) = \sum_{i=1}^{K} \pi_i P_i(Y \mid X)$$

조건부 없는 분포의 분산 (총 분산의 법칙):

$$\text{Var}(Y \mid X) = \sum_{i=1}^{K} \pi_i \sigma_i^2 + \sum_{i=1}^{K} \pi_i (\mu_i - \bar{\mu})^2$$

추론 주입 후 분포가 진짜 모드 $k$로 붕괴된다면:

$$\text{Var}(Y \mid E_{fused}) = \sigma_k^2$$

따라서:

$$\text{Var}(Y \mid E_{fused}) \leq \text{Var}(Y \mid X)$$

등호는 추론 사전이 진짜 미래 모드를 성공적으로 고립할 때 성립. 이는 LLM의 추론이 예측 불확실성을 구조적으로 감소시킨다는 것을 수학적으로 보장한다.

---

### 2.5 성능 향상

#### GIFT-Eval 결과 (Figure 2 기반)

| 모델 | MASE ↓ | CRPS ↓ |
|---|---|---|
| **STRIDE (+Chronos-2.0)** | **0.674** | **0.454** |
| STRIDE (+Timer-S1) | 0.674 | 0.463 |
| Timer-S1 (베이스라인) | 0.693 | 0.485 |
| Chronos-2.0 (베이스라인) | 0.698 | 0.485 |
| TimesFM-2.5 | 0.705 | 0.490 |
| AutoARIMA | 1.074 | 0.912 |

→ Timer-S1 대비 MASE 약 2.7% 감소, CRPS 약 4.5% 감소

#### TFRBench 결과 (Table 1 기반)

| 모델 | In-Domain MASE ↓ | Out-Domain MASE ↓ |
|---|---|---|
| **STRIDE (+Chronos-2.0)** | **0.615** | **0.724** |
| Chronos-2.0 | 0.765 | 0.778 |
| Gemini-3.1-Pro | 0.854 | 1.408 |
| Claude-Sonnet-4 | 1.734 | 1.582 |

→ In-domain에서 Chronos-2.0 대비 **19.6% MASE 개선**
→ Out-of-domain에서 Chronos-2.0 대비 **6.9% MASE 개선**

#### 추론 품질 평가 (LLM-as-a-Judge, 1-5점)

| 모델 | 도메인 관련성 | 예측 정확성 | 이벤트 관련성 | 논리 일관성 |
|---|---|---|---|---|
| **STRIDE** | **4.62** | **3.30** | **2.33** | **4.92** |
| Gemini-3.1-Pro | 3.36 | 2.87 | 2.20 | 4.90 |
| Claude-Sonnet-4 | 3.81 | 2.27 | 2.04 | 2.84 |

---

### 2.6 한계 (Limitations)

논문이 명시하는 한계:

1. **Teacher 의존성**: 훈련 파이프라인이 Teacher LLM(TFRBench 기반)의 품질과 사실적 정확성에 본질적으로 의존. Teacher가 잘못된 추론을 생성하면 Bias 증가 위험.

2. **계산 오버헤드**: Small-LLM과 TSFM을 동시에 최적화하므로, 순수 수치 인코더 단독 배포 대비 추가 메모리 및 계산 비용 발생. (훈련: A100 16개 또는 H100 8개 필요)

3. **금융 예측의 경계 조건**: Amazon, Apple과 같이 **고변동성, 이벤트 주도** 도메인에서 성능이 미미하게 열화. 추론 시 실시간 외부 정보(거시경제 뉴스 등) 접근 불가로 인해 **과도하게 평탄화된(over-smoothed) 추론 사전**을 생성하여 TSFM의 갑작스러운 시장 충격 반응성을 억제.

4. **자동화 편향 위험**: 고도로 논리적으로 보이는 AI 생성 내러티브가 데이터 결함을 은폐하거나 자동화 편향을 조장할 가능성 존재 → 인간 감독 필수.

---

## 3. 일반화 성능 향상 가능성 (심층 분석)

### 3.1 일반화 메커니즘

STRIDE의 일반화 성능 향상은 다음 세 가지 메커니즘에 기반한다:

#### (A) 모달리티 불가지론적 추론 증류 (Model-Agnostic Reasoning Distillation)

Student LLM이 추출한 연속 은닉 상태는 특정 TSFM 아키텍처에 종속되지 않는다:

$$\mathbf{e}_R = \mathbf{h}_R \mathbf{W}_{proj}$$

$\mathbf{W}\_{proj}$는 임의의 TSFM의 임베딩 차원 $d_{ts}$에 맞게 학습된다. 이는 플러그-앤-플레이 특성을 보장한다:

- Chronos-2.0 (직접 다단계 예측기): 시퀀스 프리픽싱(sequence prefixing)으로 $\oplus$ 구현
- Timer-S1 (자기회귀 예측기): 초기 상태 치환(initial state substitution)으로 $\oplus$ 구현

#### (B) 귀납 편향으로서의 추론 (Reasoning as Inductive Bias)

Ablation 연구(Table 2, 3)가 보여주듯:

$$\text{Small-LLM 교체 실험:}$$

| Small-LLM | MASE | CRPS |
|---|---|---|
| Chronos-2.0 (베이스라인) | 0.698 | 0.485 |
| Qwen3-4B-Instruct | 0.685 | 0.460 |
| **Gemma-3-4B-it** | **0.674** | **0.454** |

$$\text{Large LLM 교체 실험:}$$

| Large LLM | MASE | CRPS |
|---|---|---|
| Chronos-2.0 (베이스라인) | 0.698 | 0.485 |
| Claude-Sonnet-4.5 | 0.684 | 0.471 |
| **Gemini-3.1-Pro** | **0.674** | **0.454** |

→ **추론이 특정 모델이 아닌 robust한 귀납 편향으로 작용함을 입증**

#### (C) 도메인 외 일반화 (Out-of-Domain Generalization)

TFRBench out-of-domain 데이터셋(GIFT-Eval 훈련 세트와 전혀 겹치지 않음)에서:

$$\text{MASE}_{out-domain}^{STRIDE} = 0.724 \quad \text{vs.} \quad \text{MASE}_{out-domain}^{Chronos-2.0} = 0.778$$

특히 NYC Taxi, Traffic 데이터셋에서 현저한 개선이 확인된다. **의미론적 추론이 도메인 횡단 전이 가능한(transferable) 귀납 편향**으로 기능함을 의미한다.

### 3.2 분산 감소의 일반화 이론적 함의

편향-분산 트레이드오프 관점:

- $\mathcal{L}_{CE}$ 최소화 → 분산 감소 (reasoning embedding이 Teacher의 논리와 정렬)
- $\mathcal{L}_{Quantile}$ 최소화 → 편향 감소 (잘못된 모드를 선택한 경우 큰 패널티)

모드 오지정(misspecification) 시 기대 제곱 오차:

$$\mathbb{E}[(Y - \hat{Y})^2 \mid e_{R,j}] = \underbrace{(\mu_j - \mu_k)^2}_{\text{Bias}^2} + \underbrace{\sigma_j^2}_{\text{Variance}} + \underbrace{\sigma_k^2}_{\text{Irreducible Error}}$$

$\mathcal{L}_{Quantile}$이 이 분포적 거리를 직접 패널티화하므로, 잘못된 추론은 역전파를 통해 Student LLM을 교정한다. **이 자기 교정 메커니즘이 새로운 도메인에서도 편향을 억제하여 일반화 성능을 유지**한다.

### 3.3 일반화 한계

- **금융 시계열**: 고변동성·이벤트 주도 도메인에서 추론이 과도하게 보수적으로 생성됨
- **실시간 외부 정보 의존성**: 추론 시 검색 에이전트 없이 역사적 맥락만 사용하므로, 급격한 분포 변화(distributional shift) 상황에서 적응력 제한

---

## 4. 최신 관련 연구 비교 분석 (2020년 이후)

### 4.1 TSFM 계열

| 연구 | 연도 | 핵심 방법 | STRIDE와 비교 |
|---|---|---|---|
| **Chronos** (Ansari et al.) | 2024 | 수치값 이산화 → 언어 모델로 처리 | STRIDE의 베이스 TSFM; 추론 부재 |
| **Chronos-2** (Ansari et al.) | 2025 | 단변량→범용 예측 확장 | STRIDE가 플러그인으로 개선 (MASE 0.698→0.674) |
| **TimesFM** (Das et al.) | 2024 | 디코더 전용 파운데이션 모델 | STRIDE+Chronos-2.0에 열위 (MASE 0.705 vs. 0.674) |
| **MOIRAI** (Woo et al.) | 2024 | 범용 시계열 Transformer, 통합 훈련 | MASE 0.728로 STRIDE에 열위 |
| **Timer-S1** (Liu et al.) | 2026 | 십억 규모 파운데이션, 직렬 스케일링 | STRIDE가 플러그인으로 개선 (MASE 0.693→0.674) |
| **PatchTST** (Nie et al.) | 2023 | 패칭으로 로컬 시간 의미 보존 | 제로샷 일반화 불가, 비교 제외 |
| **TimeMachine** (Ahamed & Cheng) | 2024 | Mamba 기반 선형 복잡도 | 특정 분포에만 강함, 해석성 부재 |

### 4.2 LLM 기반 시계열 예측

| 연구 | 연도 | 핵심 방법 | STRIDE와 비교 |
|---|---|---|---|
| **Time-LLM** (Jin et al.) | 2024 | LLM 재프로그래밍으로 시계열 처리 | 이산 토큰 병목 문제 유지; per-task 훈련 |
| **PromptCast** (Xue & Salim) | 2023 | 문장→문장 생성으로 예측 | 이산 텍스트 토큰으로 수치 정밀도 저하 |
| **LLMTime** (Gruver et al.) | 2023 | LLM을 제로샷 시계열 예측기로 | "LLM 신기루" 문제 (Tan et al., 2024에서 논박) |
| **MIRAI** (Ye et al.) | 2024 | LLM 에이전트로 이벤트 예측 | 이산 토큰 방식; 수치 예측보다 이벤트 중심 |

### 4.3 지식 증류 및 CoT

| 연구 | 연도 | 핵심 방법 | STRIDE와의 관계 |
|---|---|---|---|
| **Chain-of-Thought** (Wei et al.) | 2022 | 중간 추론 단계 유도 | STRIDE의 추론 생성 영감; 수치 예측과 미결합 |
| **Distilling Step-by-Step** (Hsieh et al.) | 2023 | 소형 LLM에 추론 능력 증류 | STRIDE 증류 파이프라인의 직접적 기반 |
| **Multi-teacher Distillation** (Tian et al.) | 2025 | 다중 교사로 추론 능력 전이 | STRIDE는 단일 교사 + 도메인 특화 |

### 4.4 설명 가능한 AI (XAI) for Time Series

| 연구 | 연도 | 접근법 | STRIDE와 비교 |
|---|---|---|---|
| **SHAP** (Lundberg & Lee) | 2017 | 특성 기여도 계산 | "어떤" 시점이 중요한지만 설명, "왜"는 불가 |
| **LIME** (Ribeiro et al.) | 2016 | 로컬 선형 근사 | 인과적 깊이 부족 |
| **XAI Survey** (Rojat et al.) | 2021 | 시계열 XAI 방법론 조사 | 사후 설명 vs. STRIDE의 내재적 해석 가능성 |

### 4.5 조건부 생성 모델

| 연구 | 연도 | 핵심 방법 | STRIDE와의 관계 |
|---|---|---|---|
| **Latent Diffusion** (Rombach et al.) | 2022 | 잠재 공간 조건부 생성 | STRIDE 이론적 기반 (조건 신호로서의 추론) |
| **ControlNet** (Zhang et al.) | 2023 | 구조적 제어 신호 주입 | STRIDE의 크로스 모달 융합 아키텍처 영감 |

---

## 5. 향후 연구에 미치는 영향 및 고려 사항

### 5.1 연구에 미치는 영향

#### (A) 패러다임 전환: 예측 + 설명의 통합
STRIDE는 수치 예측과 언어 추론을 **단일 최적화 목표**로 통합하는 새로운 연구 방향을 제시한다. 이는 기존의 "예측 후 설명(post-hoc explanation)" 패러다임에서 "설명이 예측을 주도하는(reasoning-guided prediction)" 패러다임으로의 전환을 의미한다.

**영향 받을 연구 영역:**
- 금융 시계열 예측 (해석 가능성이 규제 요건)
- 의료 시계열 (임상 의사결정 지원)
- 기후 모델링 (과학적 설명 필요)

#### (B) 크로스 모달 증류의 일반화
LLM 은닉 상태를 연속 임베딩 공간에 주입하는 기법은 시계열에 국한되지 않는다. 논문이 미래 연구로 언급하듯 **공간-시간 모델링, 다중 에이전트 강화학습** 등으로 확장 가능하다.

#### (C) 벤치마크 설계에의 영향
TFRBench (Ahamed et al., 2026)의 도입은 수치 정확도만 측정하던 기존 평가에서 **추론 품질을 포함한 다차원 평가**로의 패러다임 변화를 촉진한다.

### 5.2 향후 연구 시 고려할 점

#### (A) 교사 모델의 품질 보증
현재 Teacher LLM(Gemini-3.1-Pro)의 오라클 추론 품질에 전적으로 의존하므로:
- **교사 추론의 자동 검증** 메커니즘 연구 필요
- 여러 교사의 앙상블로 추론 다양성 및 견고성 향상 가능성 탐색
- Bias 항: $\text{Bias}(\hat{Y}) = \mu_j - \mu_k$ 를 최소화하는 교사 선택 기준 개발

#### (B) 실시간 외부 지식 통합
금융·의료 도메인에서의 한계를 극복하려면:
- 추론 단계에서 **검색 증강 생성(RAG)**과의 결합
- 시간 바인딩된 외부 이벤트를 동적으로 검색·통합하는 파이프라인 설계
- 단, 추론 지연(latency) 증가와의 트레이드오프 고려 필요

#### (C) 융합 연산자의 고도화
현재 $\oplus$는 단순 시퀀스 프리픽싱 또는 초기 상태 치환이다. 더 정교한 융합 메커니즘 연구 필요:
- **크로스-어텐션(cross-attention)** 기반 융합
- **게이팅 메커니즘**: 추론의 신뢰도에 따라 동적으로 융합 강도 조절

$$\mathbf{E}_{fused} = \lambda \cdot \mathbf{e}_R \oplus (1-\lambda) \cdot \mathbf{E}_{TS}, \quad \lambda = \sigma(\mathbf{W}_{gate}[\mathbf{e}_R; \mathbf{E}_{TS}])$$

#### (D) 추론 환각(Hallucination) 억제
LLM-as-a-Judge 결과에서 Event Relevance 점수가 상대적으로 낮음(2.33/5):
- 추론 내 환각된 이벤트를 사실 검증하는 자동화된 파이프라인
- Consistency 손실 항 추가: 추론 텍스트와 수치 예측 간 모순 패널티화

#### (E) 계산 효율성 개선
현재 훈련에 A100 16개 또는 H100 8개 요구:
- **추론 캐싱**: 유사 패턴에 대한 추론 임베딩 재사용
- **양자화(Quantization)**: Student LLM의 경량화
- **온디바이스 배포**: 엣지 환경에서의 적용 가능성 탐색

#### (F) 다변량 시계열의 채널 간 추론
현재 추론 생성이 채널별로 이루어지나, 채널 간 인과 구조를 더 명시적으로 모델링:
- 인과 그래프(causal graph) 정보를 추론 프롬프트에 통합
- 그래프 신경망(GNN)과 STRIDE의 결합

#### (G) 분포 이동(Distribution Shift) 적응
Out-of-domain 성능이 In-domain 대비 약 17.7% 하락 (0.615 → 0.724):
- **메타러닝(Meta-Learning)** 프레임워크와의 결합으로 빠른 도메인 적응
- 도메인 불변 추론 특성 학습을 위한 도메인 적대적 훈련(domain adversarial training)

#### (H) 평가 프레임워크의 표준화
LLM-as-a-Judge 방식의 주관성 문제:
- TFRBench 확장 및 표준화된 추론 평가 벤치마크 구축
- 인간 전문가 평가와의 비교를 통한 LLM 판사의 신뢰도 검증

---

## 결론 요약

STRIDE는 시계열 예측에서 **수치 정확도**와 **해석 가능성**을 동시에 달성하는 최초의 엔드-투-엔드 크로스 모달 프레임워크로, GIFT-Eval에서 SOTA를 달성하고 TFRBench에서 도메인 내외를 막론한 강력한 일반화 성능을 증명했다. 핵심 혁신은 이산 토큰 병목을 피하면서 LLM의 시맨틱 추론을 연속 잠재 공간에 주입하는 것이며, 이는 예측 분산을 이론적으로 보장된 방식으로 감소시킨다. 향후 외부 지식 통합, 융합 메커니즘 고도화, 계산 효율성 개선이 이 연구 방향의 핵심 과제가 될 것이다.
