# Generalizable Autoregressive Modeling of Time Series through Functional Narratives

---

## 1. Executive Summary (10문장 이내)

이 논문은 시계열 데이터를 시간 주기의 단순 연결(concatenation of time periods)로 처리하던 기존 Transformer 방식의 한계를 극복하기 위해, 시계열을 **시간의 함수(temporal function)**로 재해석하는 새로운 사전학습 프레임워크 **NoTS(Narratives of Time Series)**를 제안한다.  
핵심 아이디어는 원본 신호에 강도가 다른 열화 연산자(degradation operator)를 적용하여 점진적으로 단순화된 함수 시퀀스를 구성하고, Autoregressive(AR) Transformer가 이를 복원하도록 학습하는 것이다.  
이론적으로, 시계열을 시간 주기 시퀀스로 다루면 미분 연산자 같은 불연속 함수를 근사할 수 없음을 Theorem 1로 증명하고, 함수 시퀀스 기반 접근이 이를 해결하는 두 가지 충분조건(Proposition 1)을 제시한다.  
합성 데이터 실험에서 NoTS는 다른 사전학습 방법 대비 평균 **26% 성능 향상**을 보인다.  
실제 22개 데이터셋(분류, 이상 탐지, 결측값 보간)에 걸친 실험에서 NoTS-lw는 다른 사전학습 방법 대비 최대 **6% 평균 오류율 개선**을 달성한다.  
또한 기존 아키텍처(PatchTST, iTransformer)에 NoTS를 부착하면 추가 백본 수정 없이 성능이 향상된다.  
경량 모델 NoTS-lw는 전체 파라미터의 **1% 미만**만 학습해도 전체 성능의 **82%**를 유지하는 뛰어난 일반화 능력을 보인다.  
스케일링 실험에서 모델 크기가 증가할수록 성능이 개선되어 AR 모델의 **power law** 거동을 따를 가능성을 시사한다.

### 1-1. 연구의 목적과 필요성

| 구분 | 내용 |
|---|---|
| **핵심 문제** | 기존 Transformer 기반 시계열 모델은 시계열을 시간 주기(patch)의 연결로 처리하여 추세·주기성 등 비국소적(nonlocal) 함수적 특성을 손실함 |
| **일반화 한계** | 패치 길이, 슬라이싱 위치, 데이터셋 특성에 민감하여 범용 표현 학습이 어려움 |
| **이론적 공백** | 미분 연산자 등 불연속 sequence-to-sequence 함수를 시간 주기 시퀀스 방식으로는 Transformer가 근사할 수 없음 (Theorem 1, p.7) |
| **연구 목적** | 시계열을 함수로 재해석하고, 함수 공간에서의 AR 학습을 통해 더 광범위한 함수 근사 능력과 높은 일반화 성능을 갖는 범용 시계열 모델 개발 |

---

## 2. 핵심 주장과 근거 표

| # | 핵심 주장 | 근거 | 위치 |
|---|---|---|---|
| 1 | 시계열을 시간 주기 시퀀스로 처리하면 미분 연산자 같은 불연속 함수를 Transformer로 근사 불가 | Theorem 1: $\sup_{\mathbf{X} \in \mathcal{D}}\|f_P(\mathbf{X}) - f_{(A)}(\mathbf{X})\|_2^2 \geq T$ | p.7, Eq. 4 |
| 2 | 함수 시퀀스 구성 방식은 표현력 측면에서 두 가지 충분조건 하에 이 문제를 해결 가능 | Proposition 1: 표현적 $\mathcal{S}_i$ 또는 표현적 토크나이저 $\mathcal{E}$ 존재 시 근사 가능 | p.7, Proposition 1 |
| 3 | NoTS는 합성 데이터 특징 회귀에서 기존 방법 대비 최대 37.8% 성능 향상 | Table 1: H-index에서 NoTS 1.27 vs Next-period pred. 1.75 | p.8, Table 1 |
| 4 | 실제 22개 데이터셋에서 NoTS-lw가 타 사전학습 방법 대비 최대 6% 평균 오류율 개선 | Table 2: NoTS-lw avg error rate 15.10 vs SimMTM 16.14 (full fine-tuning) | p.9, Table 2 |
| 5 | <1% 파라미터 학습만으로 82% 성능 달성(context-aware generalization) | Table 2 상단 4행 (frozen 조건): NoTS-lw 18.51 avg error rate | p.9, Table 2 |
| 6 | 기존 아키텍처(PatchTST, iTransformer)에 NoTS 부착 시 성능 향상 | Table 2: PatchTST 21.78 → +NoTS 18.33, iTransformer 16.07 → 15.70 | p.9, Table 2 |
| 7 | 모델 크기 확장 시 Power Law 패턴 가능성 | Figure 3(C): 127k~2.1M 파라미터 스케일링 실험 | p.10, Fig. 3C |

---

## 2-1. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 향상 및 한계

### 해결하고자 하는 문제

기존 시계열 Transformer 방식은 시계열 $\mathbf{S} = [\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_T] \in \mathbb{R}^{C \times T}$를 길이 $L$의 패치로 잘라 시퀀스를 구성한다:

$$\mathbf{x}_i = \text{Tokenizer}([\mathbf{v}_{iL}, \ldots, \mathbf{v}_{(i+1)L}])$$

이는 두 가지 핵심 문제를 야기한다:
1. **비국소 함수 특성 손실**: 추세, 주기성 같은 전역 패턴이 패치 분할로 단절됨
2. **일반화 부족**: 패치 길이, 시작 위치, 데이터셋 특성에 민감

**이론적 증명 (Theorem 1, p.7)**: 미분 연산자 $A$에 의한 sequence-to-sequence 함수 $f_{(A)}$를 고려할 때, 임의의 Transformer $f_P \in \mathcal{T}_P^{h,m,r}$에 대해:

$$\sup_{\mathbf{X} \in \mathcal{D}} \left\| f_P(\mathbf{X}) - f_{(A)}(\mathbf{X}) \right\|_2^2 \geq T $$

가 성립하여, 시간 주기 기반 Transformer는 미분 연산자를 근사할 수 없음.

---

### 제안하는 방법 (수식 포함)

**Step 1: 열화 연산자(Degradation Operator) 구성**

원본 신호 $\mathbf{S}$에 강도 $k$의 열화 함수 $d_k(\cdot)$를 적용하여 점진적으로 단순화된 함수 시퀀스 생성:

$g_k(t) = (d_k \circ g)(t)$ 이며, $g_{k+1}(t)$는 $g_k(t)$보다 많거나 같은 정보를 포함한다.

**두 가지 열화 커널:**

- **Local smoothing (국소 평활화)**:

$$w_k[n] = \frac{1}{p_k}, \quad -0.5p_k \leq n \leq 0.5p_k, \quad w_k[n] = 0 \text{ elsewhere}$$

- **Global smoothing (전역 평활화, 저역 통과 필터)**:

$$w_k[n] = \text{sinc}(p_k n)$$

여기서 $\{p_k\}$는 $k$가 증가할수록 내림차순인 하이퍼파라미터 집합.

**Step 2: 함수 시퀀스 기반 AR 모델링**

$$p(g_1(t), g_2(t), \ldots, g_K(t)) = \prod_{k=1}^{K} p(g_k(t) \mid g_1(t), g_2(t), \ldots, g_{k-1}(t)) $$

**Step 3: 잠재 공간에서의 그룹 토큰 AR 모델링**

인코더 $\mathcal{E}$, 디코더 $\mathcal{D}$를 활용:
$$\mathbf{R}_k = \mathcal{E}(\mathbf{S}_k), \quad \mathbf{S}'_k = \mathcal{D}(\mathbf{R}'_k)$$

$$[\mathbf{R}'_2, \ldots, \mathbf{R}'_K] = \text{Transformer}([\mathbf{R}_1, \ldots, \mathbf{R}_{K-1}])$$

$$\text{mask}[\Omega_k] = \begin{cases} 0, & \bigcup_{m=1}^{k} \Omega_m \\ -\infty, & \text{elsewhere} \end{cases}$$

**Step 4: 학습 목적 함수 (Training Objective, Eq. 3)**

$$\mathcal{L} = \underbrace{\sum_{k=1}^{K-1} \mathcal{L}_{\text{recon}}(\mathbf{S}'_{k+1}, \mathbf{S}_k)}_{\text{AR reconstruction}} + \underbrace{\mathcal{L}_{\text{recon}}(\mathcal{D}(\mathcal{E}(\mathbf{S}_K)), \mathbf{S}_K)}_{\text{latent consistency term}} $$

여기서 $\mathcal{L}_{\text{recon}}$은 MAE(Mean Absolute Error).

---

### 모델 구조

```
[원본 신호 S]
    ↓ 열화 연산자 d_k (k=1,...,K)
[S_1, S_2, ..., S_K] (점진적으로 단순화)
    ↓ 인코더 ε (채널 독립 1D-ResNet)
[R_1, R_2, ..., R_K] (잠재 토큰 그룹)
    + 위치 임베딩 (Group / Degradation / Channel)
    ↓ AR Transformer (3-layer, 4-head, causal mask)
[R'_2, ..., R'_K]
    ↓ 디코더 D (ResNet)
[S'_2, ..., S'_K] → AR reconstruction loss
S_K → latent consistency loss
```

**적응 모듈 (Context-aware Adaptation):**
- **Channel adaptor**: 새로운 채널 그래프 처리 (선형 레이어 $\mathbb{R}^C \to \mathbb{R}^{C'}$ + 채널 임베딩 재초기화)
- **Task adaptor**: Deep visual prompt tuning 방식의 prompt token 삽입 + 태스크별 선형 레이어

---

### 성능 향상

| 실험 종류 | 비교 대상 | NoTS 향상 |
|---|---|---|
| fBm H-index 회귀 | Next-period pred. | +37.8% (Table 1) |
| fBm SSC(32D) | Next-period pred. | +31.4% (Table 1) |
| 실제 데이터 (full fine-tuning) | SimMTM | avg error rate: 16.14 → 15.10 (+6.5%) |
| PatchTST에 NoTS 부착 | PatchTST 단독 | avg error rate: 21.78 → 18.33 |
| iTransformer에 NoTS 부착 | iTransformer 단독 | avg error rate: 16.07 → 15.70 |

---

### 한계

| 한계 | 상세 내용 |
|---|---|
| **실험 규모** | 소규모 경량 모델(243k 파라미터)만 주요 실험에 사용; 대규모 모델/데이터셋 실험 미흡 |
| **이론적 가정** | Theorem 1, Proposition 1은 특정 구성 조건과 최소 시퀀스 길이 $T$를 전제 |
| **예측 태스크 미포함** | 시계열 예측(forecasting) 태스크 실험 부재 |
| **확산 모델 연결 미완** | Gaussian 열화는 컨볼루션 기반 대비 성능 열위; 이론적 연결 미완성 |
| **채널 의존성** | 채널 독립 설계로 채널 간 복잡한 상호작용 처리 제한적 |

---

## 3. 각 주장의 페이지/Figure/Table 번호

| 주장 | 위치 |
|---|---|
| 시계열을 함수로 재해석하는 새로운 관점 | p.1, Abstract; p.2, Introduction; Figure 1 |
| Theorem 1 (시간 주기 방식의 이론적 한계) | p.7, Section 4.1, Eq. 4 |
| Proposition 1 (함수 시퀀스의 두 충분조건) | p.7, Section 4.2 |
| 열화 연산자 정의 (Local/Global smoothing) | p.4-5, Section 3.2 |
| AR 학습 목적 함수 | p.6, Eq. 3 |
| 합성 데이터 성능 비교 | p.8, Table 1 |
| 실제 데이터 성능 비교 | p.9, Table 2 |
| Ablation study | p.9-10, Table 3 |
| Power law 스케일링 | p.10, Figure 3(C) |
| Context-aware adaptation 구조 | p.5-6, Section 3.3; Figure 2(B) |

---

## 4. 저자 보고 결과 vs. 해석 분리

### 저자가 직접 보고한 결과

**연구 주제**: 시계열을 함수로 모델링하는 AR 사전학습 방법 NoTS 제안

**수식 (저자 직접 제시)**:
- AR 목적 함수: Eq. 1, 2
- 열화 연산자: $\mathbf{S}_k = d_k(\mathbf{S}) = (\mathbf{S} * w_k)[n]$
- 학습 손실: Eq. 3
- Transformer 한계 정리: Eq. 4, 5

**결과 (저자 직접 보고)**:
- 합성 데이터 fBm H-index: NoTS $1.27 \pm 0.16$ vs Next-period $1.75 \pm 0.11$ (Table 1)
- 실제 데이터 평균 오류율: NoTS-lw 15.10 vs 최상위 경쟁 방법 16.05 (Table 2, full fine-tuning)
- Frozen 조건: NoTS-lw 18.51로 최고 성능
- 스케일링: $\log(L) \approx -0.69\log(E) + 1.18$ (Figure 3C, 저자 직접 기재)

---

### 검토자(내) 해석

1. **37.8% 개선의 맥락**: fBm H-index에서의 극적인 향상은 fBm의 복잡한 공분산 구조가 비국소 함수적 특성을 강하게 요구하기 때문으로, NoTS의 강점이 가장 잘 드러나는 조건임. 그러나 모든 태스크에서 균등한 개선이 아니며 (WAMP 0.98%, b.power 2.20%), **태스크 의존성이 존재**함.

2. **82% 성능 with <1% 파라미터의 의미**: 이는 prompt tuning만으로 달성한 성능으로, 비교 기준이 full fine-tuning이 아닌 다른 frozen 방법들임을 고려해야 함. **절대적 성능보다는 효율성 관점에서 해석**이 적절함.

3. **Power Law 주장의 잠정성**: 4개의 모델 크기만으로 power law를 주장하는 것은 **통계적으로 매우 제한적**이며, 저자 스스로 "pilot study"임을 명시함.

4. **이론과 실험의 간극**: Theorem 1의 증명은 특정 입력 함수 $g_M(t) = \sin(Mt)/M$에 대한 구성적 반례로, 실제 시계열이 이런 극단적 조건에 해당하는지는 불명확함.

---

## 5. 통계적으로 취약한 부분과 비교 불가능한 수치 ⚠️

| 항목 | 취약점 | 비고 |
|---|---|---|
| **Table 1 평균 개선율** | 3회 반복 평균이나 일부 지표(WAMP 32D)는 표준편차가 크고 개선 폭이 작음 (0.98%); 통계적 유의성 검정(t-test, p-value) 없음 | ⚠️ |
| **Power Law 주장** | 4개 데이터 포인트(127k, 243k, 641k, 2.1M)만으로 power law 주장; 훈련 데이터 고정 조건으로 스케일링 법칙과 다른 설정 | ⚠️ |
| **82% 성능 with <1% 파라미터** | "82% 평균 성능"의 기준이 불명확 (전체 fine-tuning 대비인지, 동일 frozen 조건 최고치 대비인지 모호) | ⚠️ |
| **Table 2 avg error rate** | 분류(↑)와 이상탐지(↑)는 높을수록 좋고, 보간(↓)은 낮을수록 좋은 지표를 단일 "average error rate"로 통합하는 방식의 타당성 검증 부재 | ⚠️ |
| **예측(forecasting) 태스크 미포함** | 시계열 예측은 가장 일반적인 벤치마크이나 실험 없음; 다른 방법들과 공정 비교 불가 | ⚠️ |
| **NoTS-lw vs 타 방법 아키텍처 동일성** | 동일 1D-ResNet 인코더 사용을 주장하나 NoTS는 추가 열화 연산 및 다중 입력 구조로 **연산량(FLOPs)** 비교 미제시 | ⚠️ |
| **UEA Table 9 평균값 오류 의심** | Table 9의 "Average" 행에 NoTS-lw가 62.78 → 88.08로 표기되어 있으나, 이는 parameter efficient tuning과 full fine-tuning이 혼재된 것으로 보임 | ⚠️ |

---

## 6. 문서가 답하지 않는 질문

| # | 미답변 질문 |
|---|---|
| 1 | 시계열 **예측(forecasting)** 태스크에서의 성능은? 가장 일반적인 벤치마크가 빠져 있음 |
| 2 | 열화 레벨 수 $K$와 열화 강도 $\{p_k\}$의 최적값 선택 기준은? (하이퍼파라미터 민감도 분석 부재) |
| 3 | Local smoothing과 Global smoothing을 **동시에** 사용하는 경우 각각의 기여도 분석? |
| 4 | **계산 비용(FLOPs, 학습 시간)** 비교 미제시: NoTS는 $K$개의 변형 신호를 동시에 처리하므로 기존 방법 대비 얼마나 비싼가? |
| 5 | 불규칙 시계열(irregular time series)이나 결측값이 많은 데이터에서의 적용 가능성? |
| 6 | 시계열 길이 $T$가 매우 짧은 경우(예: 100 미만) 이론적 조건 $T > 2$는 충족되나 실제 성능은? |
| 7 | **멀티모달** 또는 **이종 도메인** 데이터(예: 의료+금융 혼합) 사전학습 시의 일반화? |
| 8 | 열화 연산자의 채택이 특정 도메인(예: ECG vs. 주식)에 따라 달라져야 하는가? 데이터 의존적 설계의 자동화 방법? |
| 9 | **Gaussian 열화가 왜 성능이 낮은지** 이론적 설명이 "시계열은 본질적으로 noisy"라는 가설에 그침 |
| 10 | NoTS-lw의 스케일 업 시 **최적 모델 크기**와 **학습 데이터 양의 관계**? |

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p.1): 전체 개요

**(A) 시퀀스 구성 방식 비교**
- 왼쪽: 기존 방식 — 시계열을 시간 주기 $\mathbf{X} = [\ldots, \mathbf{x}_{iL}, \ldots]$로 분할
- 오른쪽: 제안 방식 — 시계열을 함수의 합성 $\mathbf{s} \sim g(t) = [\ldots, g^{(i)}(t), \ldots]$으로 표현

**(B) 기존 AR 방식**: "다음 시간 주기 예측" = 언어 모델의 "다음 단어 예측" 모방

**(C) NoTS 방식**: 강도가 다른 열화 연산자 $d_1, d_2, d_3$로 만든 시퀀스에서 AR Transformer가 "다음 함수" 예측 → 점진적 분산 복원

**해석**: 이 그림은 논문의 핵심 아이디어를 가장 직관적으로 전달한다. 기존 방식이 신호의 **수평적 분할(시간 축)**이라면, NoTS는 **수직적 분해(정보 밀도 축)**를 수행한다는 점에서 근본적으로 다른 귀납적 편향(inductive bias)을 도입함.

---

### Figure 2 (p.5): NoTS 아키텍처 전체 구조

**(A) 사전학습 파이프라인**:
```
원본 신호 → d_k 적용 → 인코더 ε → + [위치임베딩 + 열화임베딩] → AR Transformer → 디코더 → 다음 함수 예측
원본 신호 → 인코더 ε → [직접 연결] → latent consistency loss
```

**(B) 적응 파이프라인**:
- Channel adaptor: 새로운 채널 그래프를 MLP로 전처리 + 채널 임베딩 재초기화
- Task adaptor: Prompt token 삽입 (deep visual prompt tuning) + 태스크별 선형 레이어

**해석**: latent consistency term은 단순히 정규화 역할이 아니라, 원본 신호의 잠재 표현이 학습 과정에서 방치되지 않도록 하는 **분포 이동 방지** 메커니즘으로, ablation(Table 3)에서 이를 제거 시 성능이 크게 하락함을 확인. Prompt tuning 기반 task adaptor는 파라미터 효율성의 핵심.

---

### Figure 3A (p.10): AR 추론 과정 시각화

아래에서 위로: 가장 단순화된 신호 $\mathbf{S}_1$ → 점진적으로 분산이 복원되어 원본 신호 $\mathbf{S}_K$에 근접

**해석**: AR Transformer가 실제로 "이전에 보지 못한 정보"를 생성하고 있음을 정성적으로 보여줌. 붉은 박스로 강조된 부분에서 예측 신호가 입력 시퀀스에 없던 고주파 성분을 복원하는 것이 관찰됨. 그러나 이는 정성적 시각화에 불과하며, **정량적 "hallucination" 검증은 부재**.

---

### Figure 3B (p.10): 잠재 토큰 공간 시각화 (PCA)

- **AR 이전 (before AR transformer)**: 열화 정도가 높을수록 토큰이 밀집(clustered), 낮을수록 분산
- **AR 이후 (after AR transformer)**: 예측 토큰이 실제 토큰과 유사한 분산 패턴을 재현

**해석**: Transformer가 함수적 관계를 잠재 공간에서 올바르게 학습하고 있음을 시사. 특히 "group positions"로 색칠 시 방향이 크게 변하지 않는 것은 위치 정보가 안정적으로 인코딩됨을 보여줌. 다만 2D PCA는 고차원 정보를 크게 압축하므로 **해석에 주의 필요**.

---

### Figure 3C (p.10): 스케일링 파일럿 연구

4개 모델 크기(127k, 243k, 641k, 2.1M)에 대해 Test loss가 training epochs에 따라 수렴하며, 큰 모델일수록 낮은 loss에 수렴. 저자가 표기한 power law: $\log(L) \approx -0.69\log(E) + 1.18$

**해석**: **⚠️ 통계적 취약**: 4개 포인트만으로 power law를 주장하는 것은 과도한 일반화. 학습 데이터 고정 조건에서의 스케일링은 Kaplan et al. (2020)의 "compute-optimal" 스케일링 법칙과 설정이 다름. **잠재적 가능성을 시사하는 정도로만 해석**해야 함.

---

## 8. 결론: 시사점, 후속 연구 계획 및 추가 제안

### 8-1. 저자가 제시한 시사점

1. **범용 시계열 기반 모델의 대안**: NoTS는 특정 도메인이나 태스크에 종속되지 않는 범용 동역학 학습기(general-purpose dynamic learner)로서의 가능성을 보임
2. **함수적 관점의 패러다임 전환**: 시계열을 시간 주기가 아닌 함수의 합성으로 보는 새로운 귀납적 편향 제안
3. **파라미터 효율적 적응**: <1% 파라미터 학습으로 82% 성능 → 소규모 데이터셋에서의 실용성

### 저자가 명시한 후속 연구 계획 (p.10, Limitations)

1. 더 큰 모델, 더 큰 데이터셋, 더 어려운 태스크로 실험 확장
2. 확산 모델(diffusion model)과의 이론적 연결 심화 (특히 cold diffusion, Bansal et al., 2024)
3. 확률적 사건(stochastic events)에서의 NoTS 동작 이해 (rough path theory, Kidger et al., 2020)

---

### 추가 후속 연구 방향 (검토자 제안)

1. **예측 태스크 적용**: ETT, Weather, Exchange-Rate 등 표준 예측 벤치마크 적용
2. **열화 연산자 자동 설계**: Neural Architecture Search(NAS) 또는 meta-learning으로 데이터 의존적 최적 열화 연산자 자동 선택
3. **다중 해상도 연결**: Wavelet 변환과 NoTS의 결합 — 다중 스케일 분해와 AR 학습의 시너지
4. **대형 언어 모델(LLM)과의 연계**: NoTS로 사전학습된 표현을 LLM의 컨텍스트로 활용하는 멀티모달 프레임워크
5. **불규칙 시계열 확장**: Neural ODE/CDE와 결합하여 불균등 샘플링 시계열에 적용
6. **인과성(causality) 학습**: 함수 시퀀스 내 AR 관계가 변수 간 인과 구조를 암묵적으로 학습하는지 검증

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 방법 | NoTS와의 관계 |
|---|---|---|---|
| **PatchTST** (Nie et al., 2022) | 2022 | 시계열 패치 + BERT-style MAE | NoTS가 부착하여 성능 향상; 패치 방식의 한계를 NoTS가 이론적으로 비판 |
| **SimMTM** (Dong et al., 2024) | 2024 | 이웃 포인트 집계 기반 MAE | NoTS와 직접 비교; NoTS가 대부분 태스크에서 우위 |
| **iTransformer** (Liu et al., 2023b) | 2023 | 역전된 Transformer (변수 축 attention) | NoTS 부착 시 추가 성능 향상 확인 |
| **TimeGPT-1** (Garza & Mergenthaler-Canseco, 2023) | 2023 | Next-period prediction 기반 기반 모델 | NoTS가 직접 비교에서 우위 주장; 단, TimeGPT는 훨씬 대규모 학습 |
| **Chronos** (Ansari et al., 2024) | 2024 | 스케일링+양자화로 시계열 토큰화 | NoTS와 설계 철학이 다름; 예측 태스크에서의 직접 비교 부재 ⚠️ |
| **Lag-LLaMA** (Rasul et al., 2023) | 2023 | LLM 기반 시계열 예측 | NoTS는 비교하지 않음; 예측 태스크 비교 불가 ⚠️ |
| **MOIRAI** (Woo et al., 2024) | 2024 | 범용 시계열 예측 Transformer | NoTS와 목표는 유사하나 예측 중심 vs. NoTS는 표현 학습 중심 |
| **VAR (Visual Autoregressive)** (Tian et al., 2024) | 2024 | 이미지에서 다음 해상도 예측 | NoTS의 직접적 영감; 시계열로의 개념 이전 |
| **bioFAME** (Liu et al., 2023a) | 2023 | Fourier 기반 MAE for biosignals | NoTS와 직접 비교; 생체 신호에서 NoTS 우위 |

**본 논문이 앞으로의 연구에 미치는 영향:**
1. 시계열 사전학습의 패러다임을 "시간 축 분할"에서 "정보 밀도 분해"로 전환하는 선구적 시도
2. AR 모델의 적용 도메인을 언어·이미지에서 시계열 함수 공간으로 확장하는 이론적 토대 마련
3. 파라미터 효율적 적응(prompt tuning)과 범용 사전학습의 결합 방향 제시

**앞으로 연구 시 고려할 점:**
1. **공정한 계산 비용 비교 필요**: 열화 시퀀스 $K$개 처리 비용 vs. 단일 패치 처리 비용
2. **예측 태스크 포함 필수**: 시계열 연구의 표준 벤치마크 누락은 주요 약점
3. **대규모 사전학습 데이터 실험**: NoTS-lw는 경량 모델이나, 대규모 다중 도메인 사전학습에서의 검증이 없어 "foundation model"로의 확장 가능성이 실험적으로 미검증
4. **Hyperparameter 민감도**: $K$, $\{p_k\}$ 등의 선택이 성능에 미치는 체계적 분석 필요

---

## 참고 자료

- **논문 원문**: Liu, R., Ma, W., et al. "Generalizable Autoregressive Modeling of Time Series through Functional Narratives." arXiv:2410.08421v1, 2024.
- Nie, Y., et al. "A time series is worth 64 words." arXiv:2211.14730, 2022. (PatchTST)
- Dong, J., et al. "SimMTM: A simple pre-training framework for masked time-series modeling." NeurIPS 36, 2024.
- Garza, A., Mergenthaler-Canseco, M. "TimeGPT-1." arXiv:2310.03589, 2023.
- Liu, Y., et al. "iTransformer: Inverted Transformers are Effective for Time Series Forecasting." arXiv:2310.06625, 2023.
- Ansari, A.F., et al. "Chronos: Learning the Language of Time Series." arXiv:2403.07815, 2024.
- Tian, K., et al. "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction." arXiv:2404.02905, 2024.
- Yun, C., et al. "Are Transformers Universal Approximators of Sequence-to-Sequence Functions?" arXiv:1912.10077, 2019.
- Ismailov, V.E. "A three layer neural network can represent any multivariate function." JMAA, 523(1), 2023.
- Kaplan, J., et al. "Scaling Laws for Neural Language Models." arXiv:2001.08361, 2020.
- Woo, G., et al. "Unified Training of Universal Time Series Forecasting Transformers." arXiv:2402.02592, 2024.
- Jia, M., et al. "Visual Prompt Tuning." ECCV, 2022.
- Kidger, P., et al. "Neural Controlled Differential Equations for Irregular Time Series." NeurIPS 33, 2020.
- Bansal, A., et al. "Cold Diffusion." NeurIPS 36, 2024.
- Luo, S., et al. "Your Transformer May Not Be as Powerful as You Expect." NeurIPS 35, 2022.
