# TSMixer: An All-MLP Architecture for Time Series Forecasting

> **참고 자료:**
> - 원문 논문: Chen et al. (2023), "TSMixer: An All-MLP Architecture for Time Series Forecasting," *Transactions on Machine Learning Research* (09/2023). arXiv:2303.06053v5
> - OpenReview: https://openreview.net/forum?id=wbpxTuXgm0
> - GitHub 구현체: https://github.com/google-research/google-research/tree/master/tsmixer
> - 관련 참고문헌: Zeng et al. (2023), Nie et al. (2023), Lim et al. (2021), Salinas et al. (2020), Tolstikhin et al. (2021) 등 논문 내 인용 문헌 전체

---

## 1. Executive Summary (10문장 이내)

TSMixer(Time-Series Mixer)는 RNN이나 Transformer 없이 **MLP(Multi-Layer Perceptron)만을 적층**하여 시계열 예측을 수행하는 새로운 아키텍처다.  
핵심 설계 원리는 **시간 축(Time-mixing)**과 **변수 축(Feature-mixing)**을 교대로 적용하는 Mixing 연산이며, 이는 컴퓨터 비전의 MLP-Mixer에서 영감을 받았다.  
기존 Transformer 기반 다변량 모델들이 단순 단변량 선형 모델보다 성능이 낮다는 Zeng et al.(2023)의 발견에서 출발하여, 저자들은 선형 모델의 장점을 보존하면서 교차-변수 정보도 활용할 수 있는 구조를 설계하였다.  
이론적으로, 주기적이거나 Lipschitz 스무스한 시계열에 대해 선형 모델이 예측 오차를 bounded하게 제어할 수 있음을 증명하였다(Theorem 3.1).  
일반 장기 예측 벤치마크(ETT, Weather, Electricity, Traffic)에서 TSMixer는 **유일하게 단변량 SOTA 모델과 동등한 성능을 달성한 다변량 모델**이다.  
대규모 실세계 데이터셋인 M5(Walmart 소매 판매 데이터)에서는 DeepAR, TFT 등 산업 표준 모델을 능가하는 성능(WRMSSE 0.640)을 달성하였다.  
보조 정보(static feature, future feature) 통합을 위한 확장 버전 TSMixer-Ext도 제안되었으며, 이는 TFT 대비 우수한 성능을 보였다.  
파라미터 수(189K)와 추론 속도 측면에서 Transformer 계열(1.7M~2.9M) 대비 매우 효율적이다.  
실험 결과는 교차-변수 정보가 항상 유익한 것이 아니라 **데이터셋의 특성에 따라 선택적으로 활용**해야 함을 시사한다.  
TSMixer의 설계 패러다임은 단순성과 성능의 균형을 이룬 새로운 시계열 딥러닝 아키텍처의 방향을 제시한다.

### 1-1. 연구의 목적과 필요성

| 구분 | 내용 |
|------|------|
| **배경 문제** | Transformer 기반 다변량 모델이 단순 단변량 선형 모델보다 성능이 낮다는 역설적 발견(Zeng et al., 2023) |
| **핵심 질문 1** | 교차-변수 정보가 실제로 시계열 예측에 도움이 되는가? |
| **핵심 질문 2** | 교차-변수 정보가 불필요할 때 다변량 모델이 단변량 모델만큼 성능을 낼 수 있는가? |
| **필요성** | 실세계 데이터(M5 등)는 복잡한 교차-변수 상호작용과 보조 정보를 포함하며, 이를 효율적으로 처리하는 단순하고 강력한 모델이 필요 |
| **목적** | 선형 모델의 시간 패턴 학습 능력을 유지하면서 교차-변수 정보와 보조 정보를 효과적으로 활용하는 새로운 MLP 기반 아키텍처 제안 |

> 💡 **용어 설명**
> - **교차-변수 정보(Cross-variate information)**: 여러 변수 간의 상관관계. 예: 혈압과 체중의 관계
> - **보조 정보(Auxiliary information)**: 예측 외에 추가로 활용 가능한 정보. 예: 상품 카테고리(static), 프로모션 일정(future time-varying)

---

## 2. 핵심 주장과 근거 표

| # | 핵심 주장 | 근거 | 위치 |
|---|-----------|------|------|
| 1 | 선형 모델은 주기적·스무스한 시계열 예측에 이론적으로 강력한 후보다 | Theorem 3.1 수학적 증명 | p.5~6, Appendix A |
| 2 | 시간 축 MLP 적층(TMix-Only)만으로도 SOTA 단변량 모델과 동등한 성능 달성 | Table 3 비교 실험 | p.10, Table 3 |
| 3 | TSMixer는 유일하게 다변량 모델로서 단변량 모델과 경쟁력 있는 장기 예측 성능 보유 | Table 3, -0.66% MSE 개선(vs Linear) | p.10, Table 3 |
| 4 | 교차-변수 정보가 중요한 M5 데이터셋에서 TSMixer가 최고 성능 달성 | Table 4, WRMSSE 0.737 vs Autoformer 0.742 | p.12, Table 4 |
| 5 | 보조 정보 통합 시 TSMixer-Ext가 DeepAR, TFT를 능가 | Table 5, WRMSSE 0.640 vs TFT 0.670 | p.12, Table 5 |
| 6 | TSMixer는 긴 lookback window를 효과적으로 활용하는 일반화 성능 보유 | Fig. 5, 7 실험 결과 | p.11, Fig. 5; p.24, Fig. 7 |
| 7 | 파라미터 효율성: $O(L+C)$ 파라미터 증가율로 $O(LC)$ 대비 효율적 | 이론 분석 및 Table 6 | p.7, Table 6 |
| 8 | 일반 장기 예측 벤치마크는 교차-변수 정보 평가에 편향(dataset bias)이 있다 | Table 3 vs Table 4 비교 분석 | p.5, p.11 |

> 💡 **용어 설명**
> - **Lookback window**: 예측 시 참조하는 과거 시계열 구간의 길이 $L$
> - **WRMSSE**: Weighted Root Mean Squared Scaled Error. M5 대회 공식 평가 지표

---

## 2-1. 상세 기술 설명

### 해결하고자 하는 문제

1. **다변량 모델의 과적합 문제**: Transformer 기반 모델들이 단변량 선형 모델보다 성능이 낮음
2. **교차-변수 정보 활용 실패**: 기존 다변량 모델들이 교차-변수 정보를 효과적으로 활용하지 못함
3. **보조 정보 통합의 복잡성**: 정적 특성과 미래 시변 특성을 함께 처리하는 단순한 구조 부재
4. **벤치마크 편향**: 기존 학술 벤치마크가 교차-변수 정보의 유용성을 과소평가

### 제안하는 방법 (수식 포함)

#### (1) 선형 모델의 기본 예측 수식

$$\hat{Y} = AX \oplus b \in \mathbb{R}^{T \times C_x} $$

| 기호 | 설명 |
|------|------|
| $X \in \mathbb{R}^{L \times C_x}$ | 과거 관측 행렬 (lookback window) |
| $L$ | Lookback window 길이 |
| $C_x$ | 입력 변수(feature)의 수 |
| $A \in \mathbb{R}^{T \times L}$ | 학습 파라미터 행렬 |
| $b \in \mathbb{R}^{T \times 1}$ | 편향 벡터 |
| $T$ | 예측 horizon(미래 time step 수) |
| $\hat{Y}$ | 예측값 행렬 |
| $\oplus$ | Column-wise 덧셈 연산 |

#### (2) 주기 시계열에 대한 선형 모델의 최적 해 (수식 2)

주기 함수 $x(t) = x(t-P)$에 대해:

$$A_{ij} = \begin{cases} 1, & \text{if } j = L - P + (i \bmod P) \\ 0, & \text{otherwise} \end{cases}, \quad b_i = 0 $$

| 기호 | 설명 |
|------|------|
| $P$ | 시계열의 주기 ($P < L$) |
| $A_{ij}$ | 행렬 $A$의 $(i,j)$ 번째 원소 |
| $i \bmod P$ | $i$를 $P$로 나눈 나머지 |

> 💡 **용어 설명**
> - **Lipschitz 연속(Lipschitz smooth)**: 함수의 변화율이 일정 상수 $K$ 이하로 제한되는 성질. $\left|\frac{f(a)-f(b)}{a-b}\right| \leq K$

#### (3) Affine 변환된 주기 시계열 (수식 3)

$x(t) = a \cdot x(t-P) + c$에 대해:

$$A_{ij} = \begin{cases} a, & \text{if } j = L - P + (i \bmod P) \\ 0, & \text{otherwise} \end{cases}, \quad b_i = c $$

| 기호 | 설명 |
|------|------|
| $a, c \in \mathbb{R}$ | Affine 변환 상수 (스케일, 오프셋) |

#### (4) Theorem 3.1 (주요 이론적 결과)

$x(t) = g(t) + f(t)$, 여기서 $g(t)$는 주기 $P$의 주기 신호, $f(t)$는 Lipschitz 상수 $K$를 갖는 스무스 함수일 때, lookback window $L \geq P+1$인 선형 모델에 대해:

$$|y_i - \hat{y}_i| \leq K(i + \min(i, P)), \quad \forall i = 1, \ldots, T$$

이 정리는 선형 모델이 주기+스무스 트렌드로 분해 가능한 시계열을 오차 bounded하게 예측할 수 있음을 보장한다.

#### (5) Temporal Projection (수식 4)

$$\text{TP}_{L \to T}(X)_{*,i} = W_1 X_{*,i} + b_1, \quad \forall i = 1, \ldots, C $$

| 기호 | 설명 |
|------|------|
| $W_1 \in \mathbb{R}^{L \times T}$ | Temporal projection 가중치 |
| $b_1 \in \mathbb{R}^T$ | Temporal projection 편향 |
| $X_{*,i}$ | 입력 행렬 $X$의 $i$번째 열 (feature $i$의 시계열) |

#### (6) Time Mixing (수식 5)

$$\text{TM}(X)_{*,i} = \text{Norm}\left(X_{*,i} + \text{Drop}\left(\sigma\left(\text{TP}_{L \to L}(X)_{*,i}\right)\right)\right), \quad \forall i = 1, \ldots, C $$

| 기호 | 설명 |
|------|------|
| $\sigma(\cdot)$ | 활성화 함수 (ReLU 사용) |
| $\text{Drop}(\cdot)$ | Dropout 연산 |
| $\text{Norm}(\cdot)$ | Batch normalization 또는 Layer normalization |
| $\text{TP}_{L \to L}$ | 입출력 길이가 동일한 Temporal Projection |

> 💡 **용어 설명**
> - **Residual Connection(잔차 연결)**: 입력을 출력에 직접 더하는 연결. 그래디언트 소실 문제를 완화하고 깊은 학습을 가능하게 함
> - **Dropout**: 훈련 중 무작위로 뉴런을 비활성화하여 과적합을 방지하는 정규화 기법

#### (7) Feature Mixing

```math
U_{j,*} = \text{Drop}(\sigma(W_2 X_{j,*} + b_2))
```

```math
\text{FM}_{C \to C}(X)_{j,*} = \text{Norm}(X_{j,*} + \text{Drop}(W_3 U_{j,*} + b_3)), \quad \forall j = 1, \ldots, L
```

| 기호 | 설명 |
|------|------|
| $W_2, W_3 \in \mathbb{R}^{C \times C}$ | Feature mixing 가중치 행렬 |
| $b_2, b_3 \in \mathbb{R}^C$ | Feature mixing 편향 |
| $X_{j,*}$ | 입력 행렬 $X$의 $j$번째 행 (time step $j$의 모든 feature) |
| $U_{j,*}$ | 중간 은닉 표현 |

#### (8) Conditional Feature Mixing (수식 6, 7)

```math
V_{j,*} = \text{FR}_{C_s \to H}(\text{Expand}_L(S))
```

```math
\text{CFM}_{C \to H}(X, S)_{j,*} = \text{FM}_{C+H \to H}(X \oplus V)_{j,*}, \quad \forall j = 1, \ldots, L 
```

| 기호 | 설명 |
|------|------|
| $S \in \mathbb{R}^{1 \times C_s}$ | 정적(static) 보조 특성 |
| $\text{Expand}_L(\cdot)$ | 시간 차원으로 $L$번 반복 확장 |
| $V \in \mathbb{R}^{L \times H}$ | 정적 특성의 확장된 표현 |
| $H$ | 은닉층(hidden layer)의 크기 |
| $X \oplus V \in \mathbb{R}^{L \times (C+H)}$ | Feature 차원 방향 연결(concatenation) |

#### (9) Mixer Layer와 Conditional Mixer Layer (수식 8)

$$\text{Mix}_{C \to H}(X) = \text{FR}_{C \to H}(\text{TR}_{L \to L}(X))$$

$$\text{CMix}_{C \to H}(X, S) = \text{CFR}_{C \to H}(\text{TR}_{L \to L}(X), S) $$

### 모델 구조

```
입력: X ∈ R^{L×C}
    │
    ▼
┌─────────────────────────────────┐
│         Mixer Layer × N         │
│  ┌───────────────────────────┐  │
│  │      Time Mixing (TM)     │  │ ← 시간 축 방향 MLP (features 간 공유)
│  │  [2D BN → Transpose →     │  │
│  │   FC → ReLU → Drop →      │  │
│  │   Transpose → Residual]   │  │
│  └───────────────────────────┘  │
│  ┌───────────────────────────┐  │
│  │    Feature Mixing (FM)    │  │ ← 변수 축 방향 2-layer MLP (time steps 간 공유)
│  │  [2D BN → FC → ReLU →    │  │
│  │   Drop → FC → Drop →     │  │
│  │   Residual]               │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
    │
    ▼
Temporal Projection (FC): L → T
    │
    ▼
출력: Ŷ ∈ R^{T×C_y}
```

**TSMixer-Ext (보조 정보 포함):**

```
Historical (R^{L×Cx})  Future (R^{T×Cz})  Static (R^{1×Cs})
       │                    │                    │
  [Temporal Proj]     [Feature Mix]          [Repeat]
  [Feature Mix  ]           │                    │
       │                    │                    │
       └──────── Concatenate ────────────────────┘
                            │
                    [Conditional Mixer Layer × N]
                    (Feature Mixing conditioned on S)
                            │
                    [Fully-Connected]
                            │
                    출력: R^{T×Cy}
```

### 성능 향상 및 한계

| 구분 | 내용 |
|------|------|
| **성능 향상** | 다변량 SOTA 대비 최대 62.40% MSE 개선(vs Informer), TFT 대비 51.94% 개선(Table 3) |
| **성능 향상** | M5에서 TFT WRMSSE 0.670 → TSMixer-Ext 0.640 (약 4.5% 개선) (Table 5) |
| **파라미터 효율** | 189K (TSMixer) vs 1.7M (FEDformer), 2.9M (TFT) |
| **추론 속도** | 96 step/s (TSMixer) vs 22 step/s (TFT) (Table 6) |
| **한계 1** | 고변동성(high volatility), 비주기·비스무스 시계열에 대한 이론적 분석 부재 |
| **한계 2** | 학술 장기 예측 벤치마크에서 단변량 SOTA(PatchTST) 대비 일부 열세 (-1.53%) |
| **한계 3** | 해석 가능성(interpretability) 분석 미흡 |
| **한계 4** | 더 대규모 데이터셋으로의 확장성(scalability) 미검증 |

---

## 3. 각 주장의 페이지/Figure/Table 번호

| 주장 | 근거 위치 |
|------|-----------|
| 선형 모델의 이론적 강점 | p.5~6, Eq.(1)~(3), Theorem 3.1, Appendix A(p.17) |
| 시간-단계 의존(time-step-dependent) vs 데이터 의존 모델 비교 | p.6, Figure 2 |
| TMix-Only 구조 | p.7, Figure 3 |
| TSMixer 전체 아키텍처 | p.2, Figure 1 |
| 보조 정보 포함 TSMixer-Ext | p.9, Figure 4 |
| 장기 예측 벤치마크 성능 비교 | p.10, Table 3 |
| Lookback window 효과 | p.11~12, Figure 5; p.24, Figure 7 |
| M5 (보조 정보 없음) 성능 | p.12, Table 4 |
| M5 (보조 정보 포함) 성능 | p.12, Table 5 |
| 계산 비용 비교 | p.13, Table 6 |
| MLP 대안과의 비교 | p.23~24, Table 10, Appendix F |
| TSMixer-Ext 수식 상세 | p.19~20, Appendix B.3 |

---

## 4. 저자 보고 결과 vs 검토자 해석 분리

### 저자가 직접 보고한 결과

#### 연구 주제
- 저자들은 MLP만을 사용한 시계열 예측 아키텍처 TSMixer를 제안하며, 선형 모델 분석에서 출발하여 점진적으로 모델 용량을 확장하는 체계적 접근을 취함 (p.3)

#### 방법 (저자 직접 보고)
- Time-mixing과 Feature-mixing을 교대로 적용하는 구조 (p.6~7)
- 파라미터 증가율 $O(L+C)$ (p.7)
- 장기 예측: MSE 최소화, M5: Negative Binomial 분포 log-likelihood 최적화 (p.10~11)

#### 결과 (저자 직접 보고, Table 3)
- TSMixer가 Informer 대비 MSE 62.40% 개선, TFT 대비 51.94% 개선
- PatchTST 대비 TSMixer: -1.53% (즉, TSMixer가 1.53% 낮은 성능)
- Linear 대비 TSMixer: -0.66% (즉, TSMixer가 0.66% 낮은 성능)
- M5 with auxiliary (Table 5): TSMixer-Ext WRMSSE 0.640 ± 0.013 (Best)

### 검토자(내) 해석

| 항목 | 해석 |
|------|------|
| **PatchTST 대비 열세** | TSMixer가 PatchTST보다 -1.53% 낮다는 것은 엄밀히는 **패배**이나, 저자들은 이를 "competitive"로 표현. 실질적 차이가 통계적으로 유의미한지 검증 불충분 |
| **M5 결과 일반화** | M5는 단일 대회 데이터셋이며, 소매 도메인 특화. 다른 도메인(의료, 금융 등)으로의 일반화 여부는 미검증 |
| **Dataset bias 주장** | 저자들이 일반 장기 예측 벤치마크에 "편향"이 있다고 주장하나, 이 편향의 정도와 범위를 수량화한 증거는 제한적 |
| **Time-step-dependent 논거** | 이 개념이 TSMixer 설계의 핵심 근거이나, 실험적으로 직접 검증(ablation)된 것이 아니라 이론적 주장에 가까움 |
| **보조 정보의 기여** | Table 5의 ablation에서 static feature만 추가 시 0.657, future feature만 추가 시 0.697로 static feature의 기여가 더 크나, 두 결과 모두 표준편차가 겹쳐 통계적 해석 주의 필요 |

---

## 5. 통계적으로 취약한 부분 및 비교 불가능한 수치

| 구분 | 내용 | 문제점 |
|------|------|--------|
| ⚠️ **Table 3 일부 수치** | "*" 표시된 FEDformer, Autoformer, Informer, PatchTST의 수치는 Nie et al.(2023)에서 인용 | 동일한 실험 환경(하드웨어, 랜덤 시드, 데이터 전처리)이 보장되지 않을 수 있음 |
| ⚠️ **Table 5 Val WRMSSE** | TSMixer-Ext (static only, future only 조건)의 Val WRMSSE가 모두 0.000 ± 0.000 | 이상 수치로 기록 오류 가능성 높음. 저자들도 별도 설명 없음 |
| ⚠️ **M5 표준편차 중첩** | TSMixer (0.737±0.033) vs Autoformer (0.742±0.029): 표준편차 범위가 중첩됨 | 통계적 유의성 검증(p-value, 신뢰구간) 미제공으로 성능 차이의 유의성 불명확 |
| ⚠️ **Table 6 계산 비용** | TFT는 MXNet, 나머지는 PyTorch 구현 | 프레임워크 간 성능 차이 존재. 공정한 비교 불가 |
| ⚠️ **PatchTST와 비교** | TSMixer: MSE Imp. -1.53% | 음수 개선율: 엄밀히는 TSMixer가 PatchTST보다 낮은 성능. 저자들이 이를 "competitive"로 표현하는 것은 주관적 해석 |
| ⚠️ **Lookback window 실험** | Weather, Traffic 두 데이터셋에 한정(Fig. 5) | 모든 벤치마크에 대한 전체 결과는 Appendix Fig. 7에 있으나, 본문에서는 부분적으로만 제시 |
| ⚠️ **Theorem 3.1 한계** | Lipschitz 가정 범위 내에서만 오차 보장 | 고변동성(high-volatility) 시계열, 비선형 트렌드에 대한 이론적 보장 없음 (저자도 p.6에서 인정) |

---

## 6. 논문이 답하지 않는 질문

| # | 미해결 질문 |
|---|------------|
| 1 | 교차-변수 정보가 유익한 데이터셋과 그렇지 않은 데이터셋을 **사전에 식별**하는 방법은? |
| 2 | TSMixer의 **해석 가능성(interpretability)**: 어떤 시간 패턴이나 변수 간 관계를 학습하는지 분석 없음 |
| 3 | 더 **긴 예측 horizon** (T > 720)에서의 성능은 어떠한가? |
| 4 | **의료, 금융, 에너지** 등 다른 도메인으로의 일반화 성능은? |
| 5 | Lipschitz 조건을 벗어난 **고변동성·불규칙 시계열**에 대한 이론적 처리 방안은? |
| 6 | **Negative Binomial 분포 외 다른 분포** 가정이 필요한 도메인에서의 확장성은? |
| 7 | Table 5의 Val WRMSSE **0.000 ± 0.000** 수치의 의미는? (명백한 기록 오류로 추정) |
| 8 | TSMixer와 Mamba, SSM 등 **새로운 시퀀스 모델**과의 비교는? |
| 9 | **전이 학습(transfer learning)** 또는 **사전 훈련(pre-training)** 설정에서의 TSMixer 성능은? |
| 10 | Mixing layer 수 $N$과 hidden size $H$의 **최적 선택 기준**에 대한 이론적 가이드라인은? |

---

## 7. 가장 중요한 그림 5개의 해석

### Figure 1 (p.2): TSMixer 전체 아키텍처

**해석**: TSMixer의 전체 구조를 도식화한 핵심 그림이다. 입력 행렬에서 열(column)은 변수(feature), 행(row)은 시간 축을 나타낸다. **Time Mixing MLP**는 시간 축 방향으로 작동하며 모든 변수에 걸쳐 공유된다. **Feature Mixing MLP**는 변수 축 방향으로 작동하며 모든 시간 단계에 걸쳐 공유된다. 이 두 연산의 교대 적용이 TSMixer의 핵심이다. 파라미터 공유 덕분에 $O(L+C)$ 파라미터 증가율을 달성한다. Residual connection이 각 레이어에 적용되어 선형 모델의 능력을 보존하면서 복잡한 비선형 변환도 가능하게 한다. 최종 Temporal Projection이 lookback 길이 $L$에서 예측 길이 $T$로 매핑한다.

> 💡 **용어 설명**
> - **파라미터 공유(Parameter sharing)**: 동일한 가중치를 여러 위치에 반복 적용하여 모델 크기를 줄이는 기법

### Figure 2 (p.7): 시간-단계 의존 vs 데이터 의존 모델

**해석**: 이 그림은 TSMixer 설계의 이론적 근거를 시각화한다. **왼쪽(Time-step-dependent)**: 선형 모델에서 각 가중치 $w_i$는 입력 데이터에 무관하게 고정된 시간 위치에 대응한다. 이는 시계열의 시간적 패턴(주기성, 트렌드)을 직접 학습하기에 적합하다. **오른쪽(Data-dependent)**: LSTM의 gate나 Transformer의 attention weight $f_i(\mathbf{x})$는 입력 데이터에 따라 동적으로 결정된다. 이는 표현 용량은 높으나, 시간 위치 자체를 학습하는 것이 아니라 데이터 패턴에 과적합될 위험이 높다. 이 구분이 Transformer 계열이 단순 선형 모델에 뒤처지는 이유를 설명하는 핵심 논거다.

### Figure 4 (p.9): 보조 정보 포함 TSMixer-Ext 구조

**해석**: 실세계 데이터에서 흔히 활용 가능한 다양한 유형의 보조 정보를 통합하는 방법을 보여준다. **Align Stage**: 서로 다른 shape을 가진 Historical($\mathbb{R}^{L \times C_x}$), Future($\mathbb{R}^{T \times C_z}$), Static($\mathbb{R}^{1 \times C_s}$) 정보를 동일한 shape으로 정렬한다. Historical은 Temporal Projection으로 $T$ 길이로 변환하고, Static은 $T$번 반복(Repeat)한다. **Mixing Stage**: 정렬된 세 종류의 정보를 concatenate한 후 Conditional Mixer Layer를 $N$번 적용한다. Static feature가 Conditional Feature Mixing을 통해 각 Mixer Layer에 조건으로 주입된다. 이 설계는 이질적(heterogeneous) 입력을 단일 프레임워크에서 처리할 수 있게 한다.

### Figure 5 (p.12): Lookback window 크기 변화에 따른 성능

**해석**: Weather와 Traffic 데이터셋에서 $L = \{96, 336, 512, 720\}$ 변화에 따른 Linear 모델과 TSMixer의 MSE를 비교한다. **Linear 모델**: $L=96$에서 $L=336$으로 증가할 때 MSE가 급격히 개선되며, $L=720$에서 수렴하는 경향을 보인다. Theorem 3.1의 이론적 예측과 일치한다. **TSMixer**: $L=336$ 또는 $L=512$에서 최적 성능을 달성하고, $L=720$에서도 비슷한 수준을 유지한다. 이는 Transformer 계열이 $L>192$에서 과적합되는 것(Nie et al., 2023)과 대조적이다. TSMixer가 긴 lookback window를 효과적으로 활용하며, Linear 모델보다 더 낮은 MSE를 일관되게 달성함을 보여준다.

### Figure 7 (p.24, Appendix D): 전체 데이터셋에서의 Lookback window 효과

**해석**: ETTm2, Weather, Electricity, Traffic 4개 데이터셋에서 $T = \{96, 192, 336, 720\}$ 4가지 예측 horizon과 $L = \{96, 336, 512, 720\}$ 4가지 lookback window의 교차 실험 결과(총 16개 설정 × 4개 데이터셋)를 보여준다. 전반적으로 TSMixer(주황색)가 Linear(파란색) 모델보다 대부분의 설정에서 낮은 MSE를 달성한다. 특히 Electricity와 Traffic처럼 변수가 많은 데이터셋에서 TSMixer의 우위가 더 뚜렷하다. Window 크기 $L$이 커질수록 두 모델 모두 성능이 개선되지만, TSMixer의 수렴이 더 안정적이다. 이는 TSMixer가 더 많은 파라미터를 활용하면서도 과적합 없이 긴 시계열 패턴을 학습할 수 있음을 실증한다.

---

## 8. 결론: 시사점, 후속 연구, 추가 방향

### 저자들이 제시한 시사점

1. **단순성의 힘**: RNN/Attention 없이 MLP 적층만으로 SOTA 수준의 성능 달성 가능
2. **교차-변수 정보의 조건적 유익성**: 데이터 특성에 따라 교차-변수 정보가 유익하거나 해로울 수 있음
3. **벤치마크 편향 경고**: 일반 학술 장기 예측 벤치마크가 다변량 모델 평가에 충분하지 않을 수 있음
4. **실세계 적용 가능성**: 산업 표준 모델(DeepAR, TFT) 대비 우수한 성능과 효율성

### 저자들이 제시한 후속 연구 계획

- TSMixer의 **해석 가능성(interpretability)** 탐구
- **더 대규모 데이터셋**으로의 확장성 검토
- 더 혁신적인 시계열 예측 아키텍처를 위한 설계 패러다임 탐색

### 8-1. 모델의 일반화 성능 향상 가능성

#### 현재 일반화 강점

TSMixer는 여러 설계 선택이 일반화 성능 향상에 기여한다:

1. **파라미터 효율성** ( $O(L+C)$ ): 과적합 위험 최소화
2. **Residual Connection**: 불필요한 mixing 연산을 건너뛸 수 있어 최악의 경우에도 선형 모델 수준 보장
3. **높은 Dropout 비율**: Hyperparameter 튜닝 결과(Appendix E)에서 dropout=0.7~0.9 설정이 최적으로 선택됨 → 강력한 정규화 효과
4. **2D Batch Normalization**: 시간 및 변수 차원 모두에서 스케일 안정화

#### 일반화 향상 가능성 분석

| 방향 | 설명 | 기대 효과 |
|------|------|-----------|
| **가변 lookback window** | 고정 $L$ 대신 데이터 특성에 맞게 적응적으로 선택 | 다양한 도메인 적용성 향상 |
| **사전 훈련(Pre-training)** | 대규모 시계열 데이터로 사전 훈련 후 fine-tuning | few-shot 설정에서 일반화 향상 |
| **Reversible Instance Normalization 확장** | 이미 적용하나, 비정상(non-stationary) 시계열 처리 강화 | 분포 변화(distribution shift)에 대한 강건성 향상 |
| **Meta-learning 통합** | 다수의 time series 간 공통 패턴 학습 | M5처럼 30K+ 시계열 데이터에서 효과적 |
| **데이터 증강(Augmentation)** | 시계열 특화 증강(jitter, scaling, warping) 적용 | 소규모 데이터셋에서 일반화 향상 |

> 💡 **용어 설명**
> - **Distribution shift**: 훈련 데이터와 테스트 데이터의 분포가 달라지는 현상. 시계열에서는 계절성 변화, 추세 전환 등으로 발생
> - **Reversible Instance Normalization (RevIN)**: 입력을 정규화 후 예측하고, 출력을 다시 역정규화하는 기법. 분포 변화에 강건함

#### 주의할 일반화 한계

- Theorem 3.1은 Lipschitz 스무스 가정 하에서만 성립. 금융 데이터처럼 급격한 변동이 있는 시계열에서는 이론적 보장 없음
- M5 최적화에서 사용한 Negative Binomial 분포는 소매 판매 도메인 특화 가정

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

| 모델 | 발표 | 구조 | 주요 특징 | TSMixer와의 관계 |
|------|------|------|-----------|-----------------|
| **Informer** (Zhou et al., 2021) | AAAI 2021 | Transformer | ProbSparse Attention, 장기 예측 효율화 | TSMixer가 MSE 62.40% 개선 |
| **Autoformer** (Wu et al., 2021) | NeurIPS 2021 | Transformer | Auto-Correlation + Decomposition | TSMixer가 MSE 24.51% 개선 |
| **FEDformer** (Zhou et al., 2022) | ICML 2022 | Transformer | FFT 기반 주파수 분해 | TSMixer와 유사 성능, 파라미터는 약 9배 많음 |
| **LTSF-Linear** (Zeng et al., 2023) | AAAI 2023 | Linear | 단변량 선형 모델의 강점 입증 | TSMixer 설계의 직접적 동기 |
| **PatchTST** (Nie et al., 2023) | ICLR 2023 | Transformer | Patch 기반 단변량 Transformer | TSMixer와 동등~약간 우위, 교차-변수 정보 무시 |
| **TimesNet** (Wu et al., 2023) | ICLR 2023 | CNN (2D) | 1D→2D 변환으로 시간 패턴 추출 | TSMixer와 다른 방향의 MLP 대안 |
| **DLinear** (Zeng et al., 2023) | AAAI 2023 | Linear | 분해 기반 단순 선형 모델 | TSMixer의 베이스라인 |
| **iTransformer** (Liu et al., 2024) | ICLR 2024 | Transformer | 변수를 token으로 처리하는 역방향 Transformer | 교차-변수 정보 처리 방식에서 TSMixer와 대조 |
| **Mamba/S4** 계열 | 2022~2024 | SSM | 선형 복잡도 시퀀스 모델 | TSMixer에 SSM 통합 가능성 존재 |
| **TimesFM** (Das et al., 2024) | 2024 | Transformer | Google의 대규모 사전훈련 시계열 모델 | TSMixer 아키텍처의 사전훈련 확장 가능성 |

> 💡 **용어 설명**
> - **SSM(State Space Model)**: 상태 공간 모델. Mamba 등이 대표적이며, 선형 복잡도로 긴 시퀀스를 처리할 수 있는 모델 계열
> - **Patch-based Transformer**: 시계열을 일정 길이의 패치(patch)로 분할하여 각 패치를 하나의 token으로 처리하는 방식

#### TSMixer가 후속 연구에 미치는 영향

1. **MLP 기반 시계열 모델의 가능성 입증**: Transformer 없이도 SOTA 경쟁이 가능함을 보여줌으로써 MLP/Linear 계열 연구 활성화에 기여
2. **교차-변수 정보의 조건적 가치 재인식**: 무조건적인 다변량 모델 우위 가정에 이의를 제기하고, 데이터 특성에 맞는 모델 선택의 중요성 부각
3. **벤치마크 다양화 촉구**: M5와 같은 실세계 대규모 데이터셋을 평가 기준으로 사용하는 흐름 강화
4. **Mixing 패러다임의 확산**: 컴퓨터 비전의 MLP-Mixer를 시계열에 적용하는 접근이 후속 연구(예: TimeMixer, SCINet 등)에 영향

#### 앞으로 연구 시 고려할 점

| 고려 사항 | 구체적 내용 |
|-----------|-------------|
| **벤치마크 선택** | ETT, Weather 등 기존 학술 벤치마크만이 아니라 M5, 의료, 금융 등 다양한 실세계 데이터셋 포함 필수 |
| **교차-변수 정보 활용 전략** | 데이터셋 특성(변수 간 상관성 강도)에 따라 시간 mixing과 변수 mixing의 비중을 동적으로 조절하는 메커니즘 연구 |
| **이론적 분석 확장** | Lipschitz 가정을 넘어선 일반 시계열에 대한 오차 경계 분석, 특히 고변동성 데이터 |
| **대규모 사전훈련** | TSMixer 구조의 단순성과 효율성은 대규모 시계열 기반 모델(Foundation Model) 적용에 유리할 수 있음 |
| **통계적 유의성 검증** | 모델 비교 시 표준편차만이 아니라 통계적 유의성 검증(paired t-test, Diebold-Mariano test 등) 제공 필요 |
| **다양한 예측 목적** | Point 예측 외 확률적 예측(probabilistic forecasting), 이상 탐지(anomaly detection)로의 확장 |
| **SSM/Mamba 통합** | TSMixer의 time mixing을 SSM으로 대체하여 더 긴 시퀀스 의존성 포착 가능성 탐색 |
| **해석 가능성** | Attention weight 대신 mixing weight의 시각화, feature importance 분석 도구 개발 |

---

> **⚠️ 답변 정확도 주의사항**: 본 분석은 제공된 논문 PDF(arXiv:2303.06053v5)에 기반하며, iTransformer, TimesFM, Mamba 등 2024년 이후 발표된 연구와의 비교는 해당 논문들의 공개 정보를 바탕으로 한 추론을 포함합니다. 이 부분은 직접 인용이 아닌 검토자의 해석임을 명시합니다. Table 5의 Val WRMSSE 0.000 ± 0.000 수치는 논문 원문에 그대로 기재된 수치이며, 기록 오류 가능성에 대한 판단은 검토자의 해석입니다.
