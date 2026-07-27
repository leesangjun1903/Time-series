# Tiny Time Mixers (TTMs): Fast Pre-trained Models for Enhanced Zero & Few-Shot Forecasting of Multivariate Time Series

**분석 대상 (주 출처)**

Vijay Ekambaram, Arindam Jati, Pankaj Dayama, Sumanta Mukherjee, Nam H. Nguyen, Wesley M. Gifford, Chandra Reddy, Jayant Kalagnanam (IBM Research), *"Tiny Time Mixers (TTMs): Fast Pre-trained Models for Enhanced Zero/Few-Shot Forecasting of Multivariate Time Series"*, arXiv:2401.03955v8 [cs.LG], NeurIPS 2024. (업로드된 PDF 전문)

---

## 1. Executive Summary (10문장)

1. TTM은 시계열(TS) 파운데이션 모델이 수억~수십억 파라미터로 커지는 흐름에 반대로, **1M~5M 파라미터의 "tiny" 사전학습 모델**로 zero/few-shot 예측에서 대형 모델을 능가할 수 있음을 보인 연구다.
2. 백본은 self-attention을 배제한 **TSMixer(MLP-Mixer + gated attention)** 기반이며, 사전학습은 Monash·LibCity 공개 데이터 약 **1B 샘플**로 6×A100에서 **24–30시간**만에 완료된다 (§4.3, p.7).
3. 다중 해상도(초~일 단위) 이질 데이터를 극소 용량으로 학습하기 위해 **adaptive patching (AP)**, **diverse resolution sampling (DRS)**, **resolution prefix tuning (RPT)** 세 가지 기법을 제안했다 (§3.1.1, p.4–5).
4. 또한 **multi-level modeling**으로 백본은 channel-independent 사전학습, 경량 디코더는 fine-tuning 시 channel-mixing과 **exogenous mixer**를 활성화해 기존 TS 파운데이션 모델이 대부분 결여한 채널 상관·외생변수 처리를 지원한다 (§2.1, §3.2, Figure 2).
5. Zero-shot에서 $TTM_A$ 는 Moirai 대비 4–10%, TimesFM 대비 19% MSE 개선을 보고했고, $TTM_B$ (1M)는 Chronos 대비 17–32%, Lag-Llama 대비 40% 개선을 보고했다 (Table 1, Table 2).
6. Few-shot 5%에서는 GPT4TS 대비 15%, Time-LLM 대비 10% 개선, full-shot head probing에서는 Moment 대비 3–4% 개선을 보고했다 (Table 4, Table 5).
7. 계산 측면에서 $TTM_B$의 CPU 추론은 배치당 0.01초로 Chronos $_L$ 대비 **240,000배**, GPU 메모리는 0.06GB로 683배 유리하다고 보고했다 (Table 3).
8. Ablation의 핵심 발견은 "사전학습 데이터의 **양보다 해상도 다양성**"으로, DRS 적용이 37% 개선을 준 반면 데이터를 250M→1B로 4배 늘린 효과는 6%에 그쳤다 (Figure 3).
9. `[해석]` 다만 모든 표가 **단일 실행(single-run) MSE**이며 신뢰구간·시드 분산·유의성 검정이 전무하고, Table 2는 테스트셋의 **마지막 윈도 1개**만 평가한 값이라 통계적 신뢰도가 표마다 크게 다르다.
10. `[해석]` 결론적으로 이 논문의 진짜 기여는 "SOTA 정확도"보다 **"파라미터 규모가 아니라 사전학습 데이터의 해상도 다양성이 TS 일반화의 병목"이라는 가설을 실증한 점**이며, 이는 이후 TS 파운데이션 모델 설계의 축을 scale-law에서 data-diversity로 이동시킨 논거가 된다.

---

## 1-1. 연구의 목적과 필요성

| 구분 | 내용 | 근거 위치 |
|---|---|---|
| **문제의식 (배경)** | NLP/Vision의 대규모 사전학습 성공이 TS로 이전되지 못함. TS는 공개 데이터 부족, 도메인·해상도·채널 수·길이의 극단적 이질성 때문 | Abstract; §1, p.2; Appendix B.2, p.17 |
| **직접적 계기** | 2024년 초 Moment[10], TimesFM[3], Chronos[2], Moirai[35], Lag-Llama[26] 등 "large/massive" TS FM이 동시 출시 → 강한 zero-shot 벤치마크 확립 | §1, p.2 |
| **관찰된 공백 (1) 자원** | 이들 모델은 수억~수십억 파라미터 → 산업 현장의 비용·지연·인프라 제약과 충돌 | §1, p.2; Table 3 |
| **관찰된 공백 (2) 기능** | 기존 FM은 대부분 univariate/channel-independent → **cross-channel 상관과 exogenous 변수를 모델링하지 못함**. 그러나 이는 실제 산업 예측의 필수 요건 | §1, p.2; Appendix B.3, p.18 |
| **관찰된 공백 (3) 학습 방식** | 대부분 masking 기반 재구성 목적함수 사용 | §4.9, p.12 |
| **연구 질문 (원문)** | *"Can 'tiny' pre-trained models succeed in the TS domain too? If so, can they outperform the zero/few-shot forecasting results of 'large' TS pre-trained models...?"* | §1, p.2 |
| **목적** | (a) tiny 모델의 transfer learning 가능성 실증, (b) 다중 해상도 사전학습을 소용량에서 가능케 하는 아키텍처 제안, (c) 채널 상관·외생변수 통합, (d) CPU-only 환경 배포 가능성 확보 | §1 "Outline of TTM's key capabilities" (1)~(5), p.3 |

`[해석]` **필요성의 실질**: 이 논문의 동기는 순수 학술적이라기보다 **배포 경제성**에 가깝습니다. Table 3의 CPU 추론 시간(TTM 0.01초 vs Chronos 2,340초)은 "GPU 없이 엣지/온프레미스에서 시계열 FM을 돌린다"는 시나리오를 겨냥한 것이며, IBM의 엔터프라이즈 제품화 맥락(Apache 라이선스 가중치 별도 배포, Appendix C.1의 "enterprise-use 릴리스에서 LibCity 3개 데이터셋 제외" 언급)과 직결됩니다.

---

## 2. 핵심 주장과 근거 (근거 위치 포함) — 요구사항 §2 + §3 통합

| # | 핵심 주장 | 저자가 제시한 근거 (보고 수치 그대로) | 위치 (Page / Figure / Table) |
|---|---|---|---|
| C1 | 1M급 tiny 모델이 대형 TS FM의 zero-shot 정확도를 능가한다 | $TTM_A$(5M): Moirai 대비 **4–10%↑**, TimesFM 대비 **19%↑**. $TTM_B$(1M): $Moirai_S$ 6%↑(14X), $Moirai_L$ 4%↑(311X), TimesFM 15%↑(200X). 단 ** $Moirai_B$ 대비 1%↓** | p.7, **Table 1**; 전체판 **Table 13** (p.25) |
| C2 | 자기회귀형 대형 모델 대비 격차는 더 크다 | $TTM_B$ vs Chronos **17–32%↑** (8–709X 작음), vs Lag-Llama **40%↑** | p.8, **Table 2**; 전체판 **Table 15** (p.27) |
| C3 | LLM 기반 TS 모델도 능가한다 | vs LLMTime(70B) **26–36%↑**, 크기 14,000–70,000X↓ ; vs UniTime **29–31%↑** | **Table 20**, **Table 21** (p.31) |
| C4 | 계산 비용이 압도적으로 낮다 | $TTM_B$: GPU 4.7ms / 0.8M params / 0.06GB / CPU 0.01s. $Chronos_L$ 대비 298X·886X·683X·**240,000X** | p.9, **Table 3** |
| C5 | Few-shot 5%에서 LLM 기반 SOTA 능가 | $TTM_B$ vs GPT4TS **15%↑**(84X), vs Time-LLM(7B) **10%↑**(7,000X); PatchTST 17%↑, TimeMixer 31%↑, TimesNet 40%↑ | p.9, **Table 4**; 전체판 **Table 16** (p.28) |
| C6 | Full-shot head probing에서도 SOTA | vs Moment **3–4%↑**(348X), vs GPT4TS 4–6%↑, vs Time-LLM 9–10%↑ | p.10, **Table 5**; **Table 17–19** (p.28–29) |
| C7 | 채널 상관·외생변수 융합이 실효적이다 | $TTM_Q$ - CM이 BS/CC/APP/SER 4개 데이터에서 전 baseline 능가. f-imp 컬럼 **15–20%**, 본문 서술은 **"15–44%"** | p.10, **Table 6**; §4.6 |
| C8 | **데이터의 "양"보다 "해상도 다양성"이 중요** | PT(Monash) 0.511 → +DRS(250M) **0.322 (37%↑)** → PT(Full,1B)+DRS **0.303 (6%↑)** | p.11, **Figure 3**; §4.9 첫 bullet |
| C9 | AP는 데이터 적을 때, RPT는 데이터 많을 때 효과적 | AP: 250M에서 **3%**, 1B에서 **1.5%**. RPT: 250M에서 **0%**, 1B에서 **3%**. 짧은 컨텍스트(sl=96)에서 RPT **8%** | p.12, **Table 7**; **Table 23, 24, 25** (p.32) |
| C10 | 단일 FL 모델을 다른 FL로 적응 가능 | 짧은 적응(96→192)은 recursive가 우수(0.320 vs direct 0.325), 넓은 적응은 pruning이 안정적(336: 0.362 vs 0.359) | p.11, **Figure 4** |
| C11 | 학습된 표현이 해석 가능하다 | PCA 임베딩이 계절성 반영 순환 궤도 형성; BS 데이터에서 weathersit/season/holiday/temp에 높은 채널 attention | p.12, **Figure 5**; Appendix G (p.26) |
| C12 | 직접 예측(direct forecasting) 목적함수가 masking보다 우수 | **정량 근거 없음** — "We hypothesize" 수준의 서술만 존재 | §4.9 마지막 bullet, p.12 |

`[해석]` **C12는 주장으로 제시되었으나 ablation이 없습니다.** Moment[10]가 masking 기반이라는 사실은 있으나, 동일 아키텍처·동일 데이터에서 masking vs direct를 비교한 실험이 논문 어디에도 없습니다. 이는 §5에서 다시 다룹니다.

---

## 2-1. 문제 · 방법(수식) · 구조 · 성능 · 한계 상세

### (A) 해결하고자 하는 문제 — 형식적 정의

`[저자 보고]` §2 (p.3):

다변량 시계열 $\boldsymbol{X} \in \mathbb{R}^{c \times sl}$ ($c$: 채널 수, $sl$: 컨텍스트 길이)이 주어졌을 때, 미래값

$$\boldsymbol{Y} \in \mathbb{R}^{c' \times fl}, \qquad c' \le c$$

를 예측한다. $fl$은 예측 지평(forecast horizon), $c'$은 예측 대상 채널 수, 모델 예측은 $\hat{\boldsymbol{Y}} \in \mathbb{R}^{c' \times fl}$.

채널 분류:
- **Target variables (필수)**: 예측 대상
- **Exogenous variables (선택)**: 타깃에 영향을 주며, **예측 구간 전체에서 값이 알려져 있거나 추정 가능한** 채널

`[해석]` 이 정의에서 핵심은 `c' ≤ c`입니다. 즉 TTM은 "입력 채널 수 ≠ 출력 채널 수"를 아키텍처 수준에서 허용하며, 이는 Moirai의 any-variate flattening이나 Chronos의 완전 univariate 처리와 구조적으로 다른 지점입니다.

**실제로 해결해야 하는 세 가지 하위 문제:**

| 하위 문제 | 왜 어려운가 | TTM의 대응 |
|---|---|---|
| P1. 다중 해상도 이질성 | 초 단위~일 단위 데이터를 하나의 소용량 모델로 학습 시 underfitting | AP + RPT |
| P2. 해상도 편중 | 고해상도 데이터가 샘플 수를 지배 → 저해상도 편향 | DRS |
| P3. 채널 수 가변성 | 사전학습 데이터마다 채널 수가 달라 cross-channel 학습 불가 | Multi-level (사전학습=CI, 파인튜닝=CM) |

### (B) 제안 방법 — 수식

#### B-1. 전처리 (§2.2, p.4)

인스턴스 정규화(채널별 zero mean, unit std) 후 비중첩 패칭:

$$\boldsymbol{X}_p \in \mathbb{R}^{c \times n \times pl}, \qquad n = \frac{sl}{pl}, \quad \text{stride } s = pl$$

임베딩 레이어로 패치 은닉 차원 투영:

$$\boldsymbol{X}_h \in \mathbb{R}^{c \times n \times hf}, \qquad hf = fs \cdot pl, \quad ef = 2 \cdot hf, \quad fs = 3$$

(하이퍼파라미터는 Appendix D.1, p.21)

#### B-2. Adaptive Patching (AP) — §3.1.1, p.4–5

백본은 $L$개 레벨, 각 레벨당 $M$개 TTM 블록. 레벨 $i$의 patch partition은:

$$\boldsymbol{X}_h^{(i-1)} \in \mathbb{R}^{c \times n \times hf} \;\longrightarrow\; \boldsymbol{X}_h^{i} \in \mathbb{R}^{c \times (n \cdot K_i) \times (hf / K_i)}$$

$$K_i = 2^{(L-i)}, \qquad hf = m \cdot 2^{L-1} \;\; (m \in \mathbb{Z})$$

TSMixer 적용 후 patch merging으로 $\mathbb{R}^{c \times n \times hf}$ 복원.

`[저자 보고]` $L=3$, $M=2$ 설정에서 레벨별 구성은 $(n\cdot4,\, hf/4) \to (n\cdot2,\, hf/2) \to (n,\, hf)$ (Figure 2(b)). AP는 **백본에만 적용**하고 디코더에는 적용하지 않는다.

`[해석]` $TTM_B$ 기준 $pl=64$, $fs=3$ → $hf = 192 = m \cdot 2^{2}$ → $m=48$로 제약 $hf = m\cdot 2^{L-1}$을 만족합니다. 즉 $L$을 키우려면 $hf$가 $2^{L-1}$로 나누어떨어져야 하므로 **$L$의 확장성이 은닉 차원에 묶여 있는 설계**이며, 논문은 $L>3$을 시도하지 않았습니다.

#### B-3. Diverse Resolution Sampling (DRS) — §3.1.1, p.5

고해상도 데이터셋으로부터 저해상도 데이터셋을 생성:
1. **Averaging**: 비중첩 윈도 $k$개 평균
2. **Decimation**: $k$번째 샘플만 보존

$$k = \frac{\text{target resolution}}{\text{base resolution}}$$

`[저자 보고]` 예: 4초 해상도 → 분 단위 $k=15$, 시간 단위 $k=900$. 원본 고해상도 데이터셋도 풀에 유지. 적용 대상 데이터셋은 Table 8 (p.19)의 "+ Downsample" 표기 항목.

#### B-4. Resolution Prefix Tuning (RPT) — §3.1.1, p.5

해상도 → 고유 정수 매핑 → 임베딩 레이어 → $\mathbb{R}^{hf}$ → 채널 축 확장:

$$\boldsymbol{p}_{\text{res}} \in \mathbb{R}^{c \times 1 \times hf}, \qquad \boldsymbol{X}_h \leftarrow \big[\, \boldsymbol{p}_{\text{res}} \;\Vert\; \boldsymbol{X}_h \,\big] \in \mathbb{R}^{c \times (n+1) \times hf}$$

`[저자 보고]` prefix tuning[16] 개념 차용. 토큰 어휘는 `{1h, 15min, 10min, ..., OOV}` (Figure 2(b)).

`[해석]` OOV 토큰이 존재한다는 점은 **미학습 해상도에 대한 폴백**을 의미하지만, 논문에는 OOV 경로가 실제로 얼마나 성능을 유지하는지에 대한 실험이 **없습니다**. 이는 §6의 미해결 질문으로 이어집니다.

#### B-5. 사전학습 목적함수 (§3.1, p.4)

채널 독립(univariate) 방식. 다변량 데이터는 $(\boldsymbol{X}_1, \cdots, \boldsymbol{X}_N) \in \mathbb{R}^{c(=1)\times sl}$로 분해.

$$\mathcal{L} = \left\lVert \boldsymbol{Y} - \hat{\boldsymbol{Y}} \right\rVert_2^2$$

즉 **direct multi-step forecasting** (masking 재구성 아님, autoregressive 아님).

#### B-6. Exogenous Mixer — §3.2, p.6

예측 헤드 출력 $\hat{\boldsymbol{Y}} \in \mathbb{R}^{c \times fl}$에서, 외생 채널의 예측값을 **실제 미래값으로 치환**한 뒤 전치:

$$\hat{\boldsymbol{Y}}^{e} = \big[\hat{y}_0, \cdots, \hat{y}_{c'},\, y_{c'+1}, \cdots, y_{c}\big] \in \mathbb{R}^{fl \times c}$$

시차 상관 학습을 위해 stride $=1$ 중첩 패칭:

$$\hat{\boldsymbol{Y}}^{e,p} \in \mathbb{R}^{fl \times \Delta \times c}, \qquad \Delta = 2l + 1$$

($l$: 각 시점 양쪽으로 고려할 컨텍스트 길이; $\hat{\boldsymbol{Y}}^e$의 양끝에 길이 $l$의 zero-padding 필요 — 각주 5). 이후 channel-mixing 활성화된 vanilla TSMixer 블록 통과 → linear head → $\hat{\boldsymbol{Y}} \in \mathbb{R}^{c' \times fl}$로 reshape.

#### B-7. Fine-tuning 3가지 모드 (§3.2, p.6)

| 모드 | 사용 데이터 | 갱신 대상 |
|---|---|---|
| Zero-shot | 없음 | 없음 |
| Few-shot | train split의 5–10% | TTM Head (decoder + forecast head) |
| Full-shot (head probing) | train split 전체 | TTM Head |

`[저자 보고]` **백본은 항상 frozen이며 channel-independent로 동작.** 디코더만 channel-mixing 활성화 가능. 학습 가능 파라미터: 사전학습 1M → 파인튜닝 **0.3M** (Figure 2(a)).

### (C) 모델 구조 (§2.1, p.4 + Figure 2)

```
[입력 X ∈ R^{c×sl}]
   ↓ Instance Normalization
   ↓ Patching → X_p ∈ R^{c×n×pl}
   ↓ Embedding → X_h ∈ R^{c×n×hf}
   ↓ [RPT] concat resolution prefix → R^{c×(n+1)×hf}
┌──────────────────────────────────────────────┐
│ TTM Backbone (FROZEN in fine-tuning)         │
│   Level 1: [Patch Partition (n·4, hf/4)      │
│             → TSMixer Block ×M               │
│             → Patch Merge]                   │
│   Level 2: (n·2, hf/2) ×M                    │
│   Level 3: (n, hf) ×M                        │
│   * channel-independent 전용                  │
└──────────────────────────────────────────────┘
   ↓ X_h^L ∈ R^{c×n×hf}
┌──────────────────────────────────────────────┐
│ Slim TTM Decoder (백본의 10~20% 크기, 2 layers)│
│   * fine-tuning 시 channel-mixing 활성화 가능  │
│   * AP 미적용                                 │
└──────────────────────────────────────────────┘
   ↓
[Forecast Linear Head] → Ŷ ∈ R^{c×fl}
   ↓ (optional) Exogenous Mixer  ← 외생 채널 실제 미래값 주입
   ↓ Reverse Instance Normalization
[출력 Ŷ ∈ R^{c'×fl}]
```

**TSMixer 블록 내부** (Figure 2(b), Appendix A p.16): Inter-patch mixer → Intra-patch mixer → Inter-channel mixer (optional), 모두 gated attention 결합. Self-attention 없음 → 시퀀스 길이에 대한 $O(n^2)$ 제거.

**변형 모델** (§4.3, p.7):

| 변형 | 파라미터 | $sl$ | $pl$ | 사전학습 데이터 | 사전학습 시간 |
|---|---|---|---|---|---|
| $TTM_Q$ (Quick) | ~1M | 512 | 64 | Monash ~250M | 4–6h |
| $TTM_B$ (Base) | 1M | 512 | 64 | ~1B | 24–30h (6×A100) |
| $TTM_E$ (Enhanced) | 4M | 1024 | 128 | ~1B | 〃 |
| $TTM_A$ (Advanced) | 5M | 1536 | 128 | ~1B | 〃 |

`[저자 보고]` RPT는 $TTM_Q$를 제외한 전 변형에서 기본 활성화 (Appendix D.1).

### (D) 성능 향상 — 저자 보고 수치 정리

**지표 정의** (§4.1, p.6):
- $\text{f-imp}(\%)$ = 전 데이터셋 평균 MSE 개선율
- $\text{s-imp}(X) = \dfrac{\text{baseline params}}{\text{TTM params}}$

**Zero-shot (Table 1, sliding window, $FL \in \{96,192,336,720\}$)**

| Data | $TTM_B$ | $TTM_E$ | $TTM_A$ | $Moirai_S$ | $Moirai_B$ | $Moirai_L$ | TimesFM |
|---|---|---|---|---|---|---|---|
| ETTH1 | **0.394** | 0.404 | 0.400 | 0.400 | 0.434 | 0.510 | 0.479 |
| ETTH2 | 0.345 | 0.335 | **0.333** | 0.341 | 0.346 | 0.354 | 0.403 |
| ETTM1 | 0.386 | 0.380 | **0.362** | 0.448 | 0.382 | 0.390 | 0.429 |
| ETTM2 | 0.281 | 0.271 | **0.252** | 0.300 | 0.272 | 0.276 | 0.334 |
| Weather | 0.237 | 0.238 | **0.231** | 0.242 | 0.238 | 0.260 | – |
| Electricity | 0.205 | 0.194 | 0.192 | 0.233 | **0.188** | **0.188** | – |
| Size | 1M | 4M | 5M | 14M | 91M | 311M | 200M |

**Ablation 요약**

| 기법 | 250M PT | 1B PT | 짧은 컨텍스트 (sl=96) |
|---|---|---|---|
| AP (Table 23) | **+3%** | +1.5% | – |
| RPT (Table 24, 25) | 0% | **+3%** | **+8%** (Table 25) |
| DRS (Figure 3) | **+37%** | – | – |

### (E) 한계 — 저자 명시 vs 제가 추가로 식별한 것

**`[저자 보고]` Appendix H (p.26–27):**
1. 예측(forecasting) 태스크 전용. 분류·회귀·이상탐지 미지원.
2. **컨텍스트 길이마다 별도 모델 필요.** 비-Transformer 구조라 $sl$에 민감 → 3개 변형으로 대응 중.
3. **점 예측만 지원.** Lag-Llama·Moirai가 지원하는 확률적 예측 미지원 (분포 헤드 추가 예정).
4. 예측 지평(FL)도 원칙적으로 별도 사전학습 필요 (§4.3) → pruning/recursive로 완화.

**`[해석]` 저자가 명시하지 않은 구조적 한계:**

5. **"1M 모델"이라는 표현의 모호성.** §4.3에 *"In the direct approach, model parameter size varies across FLs and we report the average parameter size in the result tables"*라고 명시되어 있습니다. 즉 Table 1의 "1M"은 **FL별로 존재하는 4개 모델의 평균 크기**이며, 단일 모델이 4개 지평을 모두 커버하는 Moirai/Chronos와 배포 관점에서 동등하지 않습니다. TTM은 (변형 3개 × FL 4개) = 최대 12개 체크포인트 관리가 필요한 반면, 대형 FM은 1개입니다. **총 저장 용량 관점의 s-imp는 논문 수치보다 작아집니다.**
6. **Exogenous mixer의 강한 전제.** 외생 변수의 미래값이 예측 구간 전체에서 정확히 알려져 있어야 합니다. 실무에서는 외생 변수도 예측값(추정치)인 경우가 많은데, **추정 오차가 주입될 때의 성능 열화 실험이 없습니다.**
7. **백본 frozen의 상한.** 도메인 시프트가 큰 타깃(예: 사전학습에 없는 금융·의료 시계열)에서 0.3M 파라미터 헤드만으로 적응 가능한지 검증되지 않았습니다. Full fine-tuning 대비 head probing의 격차도 측정되지 않았습니다.
8. **채널 수 확장성.** Traffic은 862채널인데, decoder channel-mixing을 켰을 때의 비용·정확도는 보고되지 않았습니다 (D1은 외생변수가 없어 CM 실험 대상이 아님). CM 실험(Table 6)은 최대 107채널(SER)에 그칩니다.

---

## 3. 근거 위치 매핑 (Page / Figure / Table 인덱스)

`[표기 편의를 위해 §2 표에 통합했으나, 방법론 요소별로 재정리]`

| 요소 | 본문 위치 | 보조 자료 |
|---|---|---|
| 문제 정의 · 수식 | §2, p.3 | – |
| 전처리 (정규화·패칭) | §2.2, p.4 | Figure 2(a) |
| Multi-level 구조 | §2.1, p.4 | Figure 2(a) |
| Adaptive Patching | §3.1.1, p.4–5 | Figure 2(b); Table 7, 23 |
| Diverse Resolution Sampling | §3.1.1, p.5 | Figure 3; Table 8 (p.19) |
| Resolution Prefix Tuning | §3.1.1, p.5 | Figure 2(b); Table 7, 24, 25 |
| Fine-tuning 워크플로 | §3.2, p.6 | Figure 2(a) |
| Exogenous Mixer | §3.2, p.6 | Figure 2(c); Table 6 |
| 데이터셋 목록 | §4.1, p.6 | Table 8 (사전학습, p.19), Table 9 (평가, p.20) |
| 하이퍼파라미터 | Appendix D.1–D.2, p.21 | – |
| 런타임 실험 설정 | Appendix D.3, p.21 | Table 3 |
| **Baseline 수치 출처** | Appendix D.4, p.22 | **Table 10 (필독)** |
| Zero-shot 결과 | §4.4, p.7–8 | Table 1, 2, 11, 13, 15, 20, 21 |
| Few-shot 결과 | §4.5, p.8 | Table 4, 12, 16, 22 |
| Full-shot head probing | §4.5, p.8–9 | Table 5, 17, 18, 19 |
| 채널/외생 실험 | §4.6, p.9–10 | Table 6 |
| Ablation | §4.7, p.10–11 | Figure 3, 4; Table 7, 23, 24, 25 |
| 설명가능성 | §4.8, p.11 | Figure 5, 7; Appendix G, p.26 |
| 설계 근거 논의 | §4.9, p.11–12 | – |
| 한계·향후 계획 | Appendix H, p.26–27 | – |

---

## 4. 저자 보고 결과 vs 제 해석 (분리)

### 4-1. 연구 주제

| | 내용 |
|---|---|
| **`[저자 보고]`** | "tiny(1M~) 사전학습 모델로 대형 TS FM의 zero/few-shot 성능을 능가하면서, 채널 상관·외생변수까지 지원한다" |
| **`[해석]`** | 실제로 검증된 명제는 **"제한된 벤치마크(D1 7개 + D2 4개)에서, 고정 컨텍스트·고정 지평 조건 하에, MSE 기준으로"** 성립합니다. "능가한다"는 주장은 (a) 평가 프로토콜이 표마다 다르고, (b) 확률적 예측 능력을 비교에서 배제하며, (c) 대형 FM의 강점인 도메인 폭(Chronos/Moirai는 수십 도메인 커버)을 측정하지 않으므로, **"동등 조건에서의 우위"가 아니라 "이 벤치마크에서의 우위"**로 읽어야 합니다. 반대로 C4(계산 효율)와 C8(데이터 다양성 가설)은 프로토콜 의존성이 낮아 훨씬 견고한 기여입니다. |

### 4-2. 방법론

| | 내용 |
|---|---|
| **`[저자 보고]`** | AP·DRS·RPT는 "소용량 모델의 다중 해상도 사전학습"을 위한 novel enhancement. AP는 Swin Transformer[20]의 아이디어를 TS로 이식, RPT는 prefix tuning[16] 차용 |
| **`[해석]`** | 세 기법의 novelty 수준은 다릅니다. **AP는 계층적 멀티스케일 처리의 TS 적용**으로, 개념적으로 새롭진 않으나 "레벨별 patch/hidden 축의 factorized reshape($K_i = 2^{L-i}$)"라는 **파라미터 증가 없는 구현**이 실질 기여입니다(reshape이므로 추가 가중치 0). **RPT는 조건부 임베딩의 표준적 적용**이며 토큰 1개 비용($hf$개 파라미터)으로 3–8% 이득을 얻는 비용 효율이 강점입니다. **DRS는 아키텍처가 아니라 데이터 큐레이션 전략**인데, 정작 세 기법 중 효과가 압도적(37%)입니다 — 즉 **논문의 성능 대부분은 아키텍처가 아니라 데이터 처리에서 나옵니다.** 저자도 §4.9에서 이를 인정하지만, 제목과 abstract의 강조점은 여전히 아키텍처에 있습니다. |
| **`[해석]` 수식 관점** | 손실 $\mathcal{L}=\lVert Y-\hat Y\rVert_2^2$는 정규화된 스케일에서 계산되며, reverse instance norm 이후 원 스케일 오차와 일치하지 않습니다(§2.2는 "This process is reversed at the end **before computing the loss**"라고 하나, §3.1은 reverse-normalize 후 MSE라고 서술 — **두 서술이 미묘하게 충돌**합니다). 재현 시 확인 필요 지점입니다. |

### 4-3. 결과

| 항목 | `[저자 보고]` | `[해석]` |
|---|---|---|
| Table 1 zero-shot | $TTM_A$가 Moirai 대비 4–10%↑ | $TTM_B$는 ** $Moirai_B$ (91M)에 1% 뒤집니다.** 즉 "1M이 모든 대형 모델을 이긴다"는 성립하지 않으며, 정확히는 "5M이 311M을 이긴다"입니다. 또한 **Traffic이 Table 1에서 완전히 제외**(Moirai·TimesFM 사전학습 오염)되었는데, Traffic은 862채널로 가장 어려운 데이터셋이므로 평균이 TTM에 유리한 방향으로 편향됩니다. |
| Table 4 few-shot 15%↑ | GPT4TS 대비 15% | 데이터셋별로 보면 ETTH1(0.383 vs 0.682, **44%↑**) 하나가 평균을 견인하고, **Electricity(0.183 vs 0.178)와 Traffic(0.433 vs 0.434)에서는 TTM이 지거나 동률**입니다. 산술평균 f-imp는 단일 데이터셋 지배에 취약합니다. |
| Table 3 240,000X | $Chronos_L$ 대비 CPU 240,000배 | 이 수치는 **자기회귀 디코딩 vs 직접 예측의 구조적 차이**를 반영하는 것이지 TTM 최적화의 성과가 아닙니다. Chronos는 $fl$ 스텝을 순차 생성하므로 CPU에서 극단적으로 느립니다. 공정한 비교라면 GPT4TS(0.3s, 26X)나 TimesFM(0.4s, 46X) 같은 **비-자기회귀 baseline과의 격차**를 봐야 하며, 그 경우 26–46X로 여전히 인상적이지만 240,000X와는 차원이 다릅니다. |
| Figure 3 "quality > quantity" | DRS 37% vs 4배 데이터 6% | **혼동변수 존재**: DRS는 다양성만 늘린 게 아니라 **샘플 수 자체도 늘렸습니다**(Table 8에서 downsample 버전이 풀에 추가됨). 따라서 "PT(M) → PT(M)+DRS"의 37%는 다양성 효과와 증강 효과가 분리되지 않습니다. 진짜 "품질 vs 양"을 보려면 *총 샘플 수를 고정한 채* 해상도 다양성만 바꾸는 실험이 필요합니다. `[해석] 결론 자체는 그럴듯하나 실험 설계가 결론을 완전히 지지하지는 않습니다.` |
| Table 6 "15–44%" | 본문 서술 | 표의 f-imp 컬럼은 **15–20%**입니다. 44%는 GPT4TS의 APP 데이터셋 단일 셀(0.075 vs 0.042)에서 나온 값으로, **평균값 범위와 최댓값을 한 문장에 섞은 서술**입니다. |

---

## 5. 통계적으로 취약한 부분 & 비교 불가능한 수치

### 5-1. 통계적 취약점

| # | 문제 | 근거 | 심각도 |
|---|---|---|---|
| S1 | **분산 정보 전무.** 전 25개 표 어디에도 표준편차·신뢰구간·시드 수·유의성 검정 없음 | Table 1–25 전체 | **높음** — 0.394 vs 0.400( $Moirai_S$ ) 같은 1.5% 차이는 시드 노이즈와 구분 불가 |
| S2 | **Table 2·15·20은 테스트셋 "마지막 윈도 1개"만 평가.** 즉 (데이터셋, FL)당 표본 크기 $n=1$ | Table 2 caption ("over the last test-window"), §4.4 | **높음** — Chronos 대비 "32%↑"의 통계적 신뢰도는 sliding-window 결과와 비교 불가 수준으로 낮음. Table 15에서 ETTM1 FL=192 $TTM_B$=0.402인데 FL=96은 0.172 — 인접 지평 간 2.3배 변동은 단일 윈도 평가의 불안정성을 그대로 보여줌 |
| S3 | **f-imp의 산술평균 지배 현상.** 데이터셋별 개선율의 단순평균이라 극단값 1개가 헤드라인을 결정 | Table 4 (ETTH1 44% vs Electricity −3%) | 중간 |
| S4 | **caption 오류: Table 4는 " $FL\in\{96,192,336,720\}$ 평균"이라 하나, Table 16을 보면 ETTH1/ETTH2/Traffic은 FL=720이 "–"** → 실제로는 3개 지평 평균 | Table 4 caption vs Table 16, p.28 (검증: GPT4TS ETTH1 (0.543+0.748+0.754)/3 = 0.682 ✓) | 중간 — TTM·baseline 모두 동일 처리이므로 비교 자체는 공정하나, caption이 부정확 |
| S5 | **하이퍼파라미터 튜닝 비대칭.** TTM은 타깃 데이터셋별로 head dropout(0.7/0.2)·batch size(8/32/64)를 validation으로 선택 (Appendix D.2). 재현된 baseline들이 동등한 튜닝 예산을 받았는지 불명 | Appendix D.2, p.21 | 중간 |
| S6 | **확률 모델의 점 예측 평가.** Chronos(num_samples=20), Lag-Llama(num_samples=100)의 샘플 평균/중앙값을 MSE로 평가. CRPS 등 고유 지표 미보고 | Appendix D.3, p.21 | 중간 — 확률 모델에 불리한 방향의 체계적 편향 |
| S7 | **런타임과 정확도가 서로 다른 설정.** Table 3은 num_samples=1(최속), 정확도 표는 20/100. 즉 "이 속도로 이 정확도"인 조합은 실재하지 않음 | Appendix D.3 | 중간 — 논문이 명시하긴 함 |
| S8 | **모델 크기 수치 불일치.** Lag-Llama = 2.4M(Fig.1, Table 3) vs 3M(Table 2) / GPT4TS = 86M(Fig.1) vs 87M(Table 3) vs 84M(Table 4) / $Moirai_B$ = 91M(Table 1,3) vs Fig.1 미표기 | Figure 1, Tables 1–4 | 낮음 — s-imp 계산의 기준이 표마다 다름 |
| S9 | **단일 실행 ablation.** Table 23–25의 1.5%, 0%, 3% 차이는 재실행 분산 내일 가능성 배제 불가. 특히 Table 24에서 RPT는 250M 설정의 ETTM2·Weather·Electricity에서 **오히려 악화**(0.187→0.194, 0.154→0.160, 0.169→0.175) | Table 24, p.32 | 중간 |
| S10 | **사전학습 도메인 근접성.** Traffic을 평가에서 쓰면서 LibCity의 PEMS03/04/07/08, PEMS_BAY, LOS_LOOP, LOOP_SEATTLE, Q-TRAFFIC(모두 교통 점유율/속도)을 사전학습에 사용. 저자는 "Traffic 데이터셋 자체는 제외"라 하나 **도메인 누출 가능성은 남음** | Table 8 (p.19), §4.1 | 중간 |
| S11 | **평가된 가중치 ≠ 배포 가중치.** "the last three datasets in the Libcity section have been excluded from the pre-training process for the model releases intended for enterprise-use" — 배포판 성능 수치 없음 | Table 8 caption, Appendix C.1 | 중간 (재현성) |
| S12 | **C12(direct vs masking) 무근거.** ablation 없이 "We hypothesize"로만 주장 | §4.9, p.12 | 낮음 (주장 강도가 약해 정직함) |

### 5-2. 비교 불가능한 수치 (교차 인용 금지 목록)

| 비교 시도 | 왜 불가능한가 |
|---|---|
| **Table 1 ↔ Table 2** | 평가 프로토콜(전체 sliding window ↔ 마지막 윈도 1개)과 지평 집합($\{96,192,336,720\}$ ↔ $\{24,48,60,96,192\}$)이 모두 다름. "Moirai 대비 4%"와 "Chronos 대비 32%"를 같은 축에 놓을 수 없음 |
| **Figure 1의 X%↓ 라벨들** | caption이 "Full details in Tables [1–5]"라 명시 — 즉 **하나의 산점도에 sliding-window(T1), last-window(T2), few-shot(T4), head-probing(T5) 결과가 혼재**. 축(정확도 라벨) 자체가 이질적. Figure 1은 마케팅 도식으로 읽어야 하며 정량 비교 근거로 쓸 수 없음 |
| **컬럼별 f-imp 평균** | Table 1에서 TimesFM은 Weather·Electricity가 결측(사전학습 오염) → **4개 데이터셋 평균**, Moirai는 6개 데이터셋 평균. "TimesFM 15%↑"와 " $Moirai_S$ 6%↑"는 서로 다른 데이터 집합 위의 값 |
| **Table 14 (TTM zero-shot vs full-shot SOTA)** | full-shot baseline들은 **$sl=96$의 짧은 컨텍스트**로 학습된 반면 TTM은 $sl=512/1024/1536$. 저자가 caption에 명시했으나, "zero-shot이 full-shot을 이겼다"는 서술은 **컨텍스트 길이 4~16배 차이를 무시**한 표현 |
| **Table 22 (cross-transfer)** | baseline은 ETTH2→ETTH1 전이, $TTM_Q$는 Monash 1B→ETTH1. 소스 데이터가 근본적으로 다르므로 "IMP 17–43%"는 방법 비교가 아니라 **데이터 규모 비교** |
| **재현 수치 ↔ 인용 수치** | Table 10(p.22)에 따르면 Moirai zero-shot·Moment·Time-LLM·UniTime·SimMTM 계열·PatchTST·TimesNet·DLinear는 **원논문 표에서 그대로 인용**, Chronos·Lag-Llama·TimesFM·TSMixer·TimeMixer·iTransformer는 **저자가 재실행**. 전처리·split·정규화 관례가 다를 수 있어 두 그룹 간 상대 비교는 신뢰도가 낮음 |
| **s-imp(X)** | TTM 크기는 "FL별 모델의 평균 파라미터 수"(§4.3). 단일 모델로 전 지평을 커버하는 baseline과 비교 시 **분모 정의가 다름** |

---

## 6. 이 문서가 답하지 않는 질문

### 6-1. 저자가 스스로 인정한 것 (Appendix H)
- Q1. 예측 외 태스크(분류·이상탐지·대치)에서의 성능은?
- Q2. 컨텍스트 길이가 동적으로 변할 때 단일 모델로 대응 가능한가?
- Q3. 확률적 예측(예측구간, CRPS 캘리브레이션)은?

### 6-2. 논문이 다루지 않은 것 `[해석]`

**스케일링·설계**
- Q4. **파라미터를 10M, 50M, 100M으로 늘리면 어떻게 되는가?** 1M/4M/5M 세 점만으로는 "tiny가 충분하다"인지 "5M에서 포화되기 시작한다"인지 판별 불가. TS 도메인의 scaling law는 미측정.
- Q5. $L=3$, $M=2$, $fs=3$, decoder=2 layers는 어떻게 정해졌는가? 이 구조 하이퍼파라미터에 대한 ablation 없음.
- Q6. 디코더 크기(백본의 10–20%)의 근거는? 더 크게/작게 하면?
- Q7. AP의 $K_i = 2^{L-i}$가 아닌 다른 스케줄(예: 비균등 patch length)의 효과는?

**일반화·강건성**
- Q8. **사전학습에 없던 해상도(OOV 토큰 경로)에서 RPT는 어떻게 동작하는가?** Figure 2(b)에 OOV가 존재하나 실험 없음.
- Q9. **사전학습 도메인과 크게 다른 타깃**(금융 수익률, 의료 신호, 저빈도 수요 등)에서의 zero-shot은? 평가 D1은 전력·기상·교통에 집중.
- Q10. 결측치, 불규칙 샘플링, 짧은 시계열(cold start)은 어떻게 처리하는가?
- Q11. 분포 시프트·이상치·구조적 단절(regime change) 상황의 강건성은?
- Q12. **백본 전체 fine-tuning(unfrozen) 대비 head probing의 성능 격차**는 얼마인가? 이 값이 없으면 "frozen 백본으로 충분하다"는 설계 정당화가 불완전.

**Exogenous / Multivariate**
- Q13. **외생 변수 미래값이 부정확할 때**(추정치일 때) 성능은? Exogenous mixer의 핵심 전제가 검증되지 않음.
- Q14. Exogenous 실험(Table 6)에 왜 **가장 약한 변형인 $TTM_Q$만** 사용했는가? $TTM_B$ / $TTM_E$ / $TTM_A$ 결과 없음.
- Q15. Exogenous mixer의 $\Delta = 2l+1$에서 $l$의 값은? 논문에 구체적 수치 없음(Figure 2(c)에서 $l=1, \Delta=3$ 예시만 제시).
- Q16. 채널 수 수백~수천 규모(Traffic 862ch)에서 channel-mixing의 비용·정확도는?

**평가·재현**
- Q17. 시드 분산은 얼마인가? 보고된 개선폭이 노이즈를 넘는가?
- Q18. MSE 외 지표(MAE, MASE, sMAPE, CRPS)에서도 동일한 순위가 유지되는가?
- Q19. M4/M5, GIFT-Eval 같은 **광역 벤치마크**에서의 성능은? 11개 데이터셋은 TS FM 평가로는 좁음.
- Q20. Figure 5(b)의 gated attention 가중치가 **실제로 인과적 기여를 반영하는지**(faithfulness) — permutation/ablation 기반 검증 없음.
- Q21. 사전학습 총 FLOPs·에너지 비용은? "24–30시간 × 6 A100"만 제시, baseline과의 학습 비용 비교 없음.
- Q22. 엔터프라이즈 배포 가중치(LibCity 3개 제외)의 성능 수치는?

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p.2) — Size × CPU Inference Time × Accuracy 3축 요약

`[저자 보고]` 로그-로그 산점도. x축=모델 크기(M), y축=배치당 CPU 추론 시간(s). 각 baseline 라벨의 X%↓는 "TTM 대비 X% 덜 정확함". $TTM_B$는 (1M, $10^{-2}$ s)로 좌하단 극단에 위치.

`[해석]`
- **읽어야 할 구조**: 점들이 좌하→우상 대각선을 이루지 않습니다. $Chronos_T$ (8M, 2500s)와 GPT4TS(86M, 0.25s)를 보면 **크기와 추론 시간이 무관**하며, 진짜 결정 요인은 **디코딩 방식**입니다. 자기회귀(Chronos, Lag-Llama)는 크기와 무관하게 $10^1$ – $10^3$ s 대역, 비자기회귀(GPT4TS, TimesFM, Moirai)는 $10^{-1}$ – $10^1$ s 대역.
- **따라서 이 그림의 진짜 메시지**는 "작으면 빠르다"가 아니라 **"직접 예측(direct multi-step) 구조를 택하면 자기회귀 대비 3–5 orders of magnitude 빠르다"**입니다. TTM은 여기에 소형화를 더해 한 단계 더 내려간 것입니다.
- **⚠ 주의**: caption의 "Tables [1–5]"는 §5-2에서 지적했듯 이질적 프로토콜의 혼합입니다. X% 라벨을 정량 근거로 인용하지 마십시오.

### Figure 2 (p.3) — TTM 전체 아키텍처 (a)(b)(c)

`[저자 보고]` (a) 사전학습(1M 학습 파라미터, univariate) ↔ 파인튜닝(0.3M 학습 파라미터, multivariate) 워크플로 대비. 백본은 transfer 후 freeze. (b) $L=3, M=2$ 백본의 adaptive patching 상세 + RPT 경로. (c) $l=1$($\Delta=3$)일 때의 exogenous mixer — 초록색 control/exogenous 채널의 예측값이 알려진 실제값으로 치환됨.

`[해석]` **논문 전체에서 가장 정보 밀도가 높은 그림이며, 여기서 3개의 설계 결정을 읽어야 합니다.**
1. **(a)의 비대칭성**: 사전학습 경로는 `[1×sl]` 단일 채널, 파인튜닝 경로는 `[c×sl]` 다채널. 즉 **모델이 "언어"를 두 번 바꿉니다.** 이것이 "채널 수가 데이터셋마다 다르다"는 P3 문제의 우회책입니다 — 채널 상관을 사전학습에서 배우지 않고 **타깃 도메인에서만 배웁니다.** 장점은 사전학습 유연성, 단점은 **채널 상관에 대한 사전지식이 0**이라는 점입니다.
2. **(b)의 파라미터 무증가 트릭**: patch partition/merge는 reshape 연산이므로 **AP는 추가 파라미터를 전혀 쓰지 않고** 멀티스케일 처리를 얻습니다. "tiny"를 유지하면서 표현력을 늘린 핵심 장치입니다. `n = n+1` 표기는 RPT prefix가 패치 축에 1개 추가됨을 뜻합니다.
3. **(c)의 전제 노출**: "green channels ... are replaced by their **known true values**". 이 한 문장이 exogenous mixer의 적용 범위를 결정합니다 — 캘린더·프로모션·기상예보처럼 미래가 확정/고신뢰인 변수에만 안전하게 쓸 수 있습니다.

### Figure 3 (p.11) — 사전학습 데이터의 양 vs 해상도 다양성

`[저자 보고]` 3-bar. PT(Monash only) = **0.511** → PT(M)+DRS, 250M samples = **0.322 (37% IMP)** → PT(Full)+DRS, 1B samples = **0.303 (6% IMP)**. FL 96·192 zero-shot 평균 MSE.

`[해석]`
- **논문에서 가장 중요한 실험이자 가장 인용 가치가 높은 발견**입니다. 4배 데이터 증가(250M→1B)의 한계효용(6%)이 DRS 하나(37%)보다 **6배 작다**는 것은, TS 도메인에서 naive scaling이 비효율적임을 시사합니다. §4.9에서 저자는 TimesFM 300B / Moirai 27B time-points를 언급하며 이를 논거로 삼습니다.
- **⚠ 혼동변수 (§4-3 재강조)**: 첫 번째 막대에서 두 번째로 갈 때 **다양성과 샘플 수가 동시에 증가**합니다. Monash 원본은 250M보다 작았을 것이고 DRS로 250M이 되었습니다. 따라서 37% 중 얼마가 "다양성"이고 얼마가 "증강"인지 분리되지 않습니다. **총 샘플 수를 고정한 대조군이 필요합니다.**
- **로그 스케일 관점**: $0.511 \to 0.322$는 큰 도약이지만 $0.322 \to 0.303$은 수확 체감의 시작으로도, 혹은 log-linear scaling의 정상 구간으로도 읽힙니다. 데이터 점 2개(250M, 1B)로는 곡선 형태를 결정할 수 없어 **"양이 중요하지 않다"는 강한 결론은 과잉 해석**입니다.

### Figure 4 (p.11) — Forecast Length Adaptation (FLA)

`[저자 보고]` 3계열 × 4지평 MSE (D1 전체 평균):

| FL | Direct | Recursive (from FL 96) | Pruning (from FL 720) |
|---|---|---|---|
| 96 | **0.282** | 0.282 (동일 모델) | 0.297 |
| 192 | 0.325 | **0.320** | 0.333 |
| 336 | **0.359** | 0.561 | 0.362 |
| 720 | **0.412** | 0.593 | 0.412 (동일 모델) |

`[해석]`
- **명확한 교차점(crossover)이 FL 192와 336 사이에 존재**합니다. Recursive는 192까지는 direct를 미세하게 앞서지만(0.320 < 0.325), 336에서 **오차가 56% 폭증**(0.359 → 0.561)합니다. 이는 전형적인 **오차 누적(error accumulation)** 패턴으로, 96짜리 예측을 3~7회 반복 적용할 때 발생합니다.
- **Pruning은 전 구간에서 direct 대비 0–5% 이내로 안정적**입니다(96: +5.3%, 192: +2.5%, 336: +0.8%). 즉 **운영 관점의 실용적 결론은 "가장 긴 FL로 하나 학습해두고 필요시 pruning"**입니다. 이는 §5의 "체크포인트 12개 관리" 문제를 상당 부분 완화하는 답이며, 논문이 충분히 강조하지 않은 실용적 함의입니다.
- **⚠ 단일 실행이며 D1 평균값만 제시**되어, 데이터셋별로 교차점이 어디인지는 알 수 없습니다.

### Figure 5 (p.12) — 임베딩 PCA 투영 (a) + 채널 attention (b)

`[저자 보고]`
(a) Weather(10분), Traffic(1시간), Electricity(1시간) 3개 데이터셋 × 각 3개 비중첩 세그먼트(S-1/S-2/S-3)의 백본 출력 임베딩을 flatten 후 PCA. 제1·2 주성분 투영. 순환 궤도(cyclic orbits)가 계절성을 반영하며, **동일 해상도(1시간)인 Traffic·Electricity는 동심 궤도(concentric orbits)를, Weather는 다른 부분차원에서 궤도**를 형성.
(b) Bike Sharing fine-tuned 모델의 channel-mixing gated attention 평균 가중치 — `weathersit`, `season`, `holiday`, `temp`가 `cnt` 예측에 높은 기여.
(분석 절차는 Appendix G.1–G.2, p.26; 세그먼트 원본은 Figure 7, p.32)

`[해석]`
- **(a)의 함의**: 해상도가 같은 두 데이터셋이 같은 부분공간에 매핑된다는 것은, **모델이 도메인이 아니라 "주기 구조"로 표현을 조직**한다는 증거입니다. 이는 RPT의 설계 가정("해상도로 조건화하면 가중치를 분리할 수 있다")과 정합적입니다. 다만 **RPT를 끈 모델의 임베딩과 비교하지 않았으므로**, 이 구조가 RPT 덕분인지 데이터 자체의 성질인지는 이 그림만으로 판별 불가입니다.
- **(a)의 방법론적 약점**: flatten 후 PCA의 제1·2 주성분은 전체 분산의 일부만 설명합니다. **설명 분산 비율(explained variance ratio)이 보고되지 않아**, 궤도 구조가 지배적 구조인지 저차원 아티팩트인지 알 수 없습니다.
- **(b)의 함의와 한계**: 자전거 대여가 날씨·휴일에 좌우된다는 것은 **사전에 알려진 상식**이며, attention이 이를 재현한 것은 sanity check로서는 좋지만 **새로운 지식을 발견한 것은 아닙니다.** 더 중요한 문제는 **attention 가중치 = 인과적 중요도가 아니라는 점**(NLP 분야의 "Attention is not Explanation" 논쟁과 동일 쟁점)입니다. 해당 채널을 permute/ablate했을 때 실제로 MSE가 증가하는지 확인하는 **faithfulness 검증이 없습니다.** 따라서 "explainable"이라는 표현(§4.8 마지막 문장)은 근거보다 강합니다.

---

## 8. 결론

### 8-0. 저자가 제시한 시사점과 후속 계획

**`[저자 보고]` 시사점 (§5 Conclusions, p.12 + §4.9 p.11–12):**
1. 극소 모델도 이질적 데이터에 대한 사전학습이 가능하며, zero/few-shot SOTA를 달성할 수 있다.
2. **해상도 다양성이 데이터 양보다 중요**하다 — "This is an important observation and finding that resolution diversity in pretraining data is very crucial for time-series FMs."
3. TSMixer 계열 선택이 모델 크기 절감의 핵심 (self-attention의 $O(n^2)$ 회피).
4. Direct forecasting 목적함수가 masking 기반보다 zero-shot에 유리하다 (가설 수준).
5. CPU/GPU 양쪽 배포 지원으로 실무 채택 장벽을 낮춘다.

**`[저자 보고]` 후속 계획 (§5 + Appendix H, p.26–27):**
- (F1) 예측 외 다운스트림 태스크(분류, 회귀, 이상탐지) 확장
- (F2) 동적으로 변하는 컨텍스트 길이에 자동 적응하는 백본
- (F3) 분포 헤드(distribution head) 추가를 통한 확률적 예측 지원

---

### 8-1. 모델의 일반화 성능 향상 가능성 (중점)

#### (i) 논문이 제시한 일반화 메커니즘 정리

| 메커니즘 | 일반화에 기여하는 방식 | 측정된 효과 | 근거 |
|---|---|---|---|
| **DRS** | 해상도 marginal 분포를 균일화 → 고해상도 편향 제거 | **+37%** | Figure 3 |
| **AP** | 레벨마다 다른 유효 patch length → 데이터셋별 최적 granularity를 앙상블처럼 커버 | +3% (소데이터) / +1.5% (대데이터) | Table 23 |
| **RPT** | 해상도 조건화로 가중치 분리(weight decoupling) | +3% (대데이터) / **+8% (짧은 컨텍스트)** | Table 24, 25 |
| **Channel-independent 사전학습** | 채널 수 불변 → 임의 채널 수 데이터로 사전학습 가능 | 정량 미측정 | §3.1 |
| **Frozen backbone + slim head** | 파인튜닝 파라미터 0.3M → 5% 데이터로도 과적합 억제 | Table 4 | §3.2 |
| **Direct forecasting** | 오차 누적 제거 | 정량 미측정 (Fig.4가 간접 증거) | §4.9 |

#### (ii) `[해석]` 일반화 성능의 **현재 한계 지점** 진단

1. **일반화의 축이 "해상도"에만 정렬되어 있습니다.** AP·DRS·RPT 세 기법 모두 **temporal resolution** 축의 이질성을 겨냥합니다. 그러나 TS 이질성은 최소 5개 축이 있습니다: 해상도, **진폭/스케일 분포**, **계절 주기 구조**, **정상성 여부**, **채널 상관 구조**. 인스턴스 정규화가 스케일 축을 부분적으로 처리하지만, 나머지 3개 축은 다루어지지 않습니다.
2. **채널 상관에 대한 사전지식이 0입니다.** 백본이 항상 channel-independent이므로, 다변량 구조는 매 타깃 도메인에서 0.3M 파라미터로 **처음부터** 학습해야 합니다. 이것이 Table 6 실험이 소규모 데이터셋(BS 17,379 / CC 5,409 / APP 8,834 / SER 8,835)에서만 수행된 이유일 가능성이 있습니다 — 대규모·고차원 다변량에서는 검증되지 않았습니다.
3. **Table 24의 음의 전이 신호.** 250M 설정에서 RPT가 3개 데이터셋의 성능을 악화시켰다는 것은, **조건화 신호가 데이터가 부족할 때 오히려 용량을 낭비**할 수 있음을 시사합니다. 이는 tiny 모델의 근본적 트레이드오프(용량 vs 조건화)를 드러냅니다.

#### (iii) `[해석]` 일반화 향상을 위한 구체적 제안

**A. 해상도 축을 넘어선 조건화 (RPT의 일반화)**
RPT를 스칼라 해상도 임베딩에서 **다중 메타데이터 prefix**로 확장:

$$\boldsymbol{P} = \big[\,\boldsymbol{p}_{\text{res}} \,\Vert\, \boldsymbol{p}_{\text{domain}} \,\Vert\, \boldsymbol{p}_{\text{seasonality}} \,\Vert\, \boldsymbol{p}_{\text{stationarity}}\,\big] \in \mathbb{R}^{c \times k \times hf}$$

여기서 $\boldsymbol{p}_{\text{seasonality}}$는 학습 가능한 코드북이 아니라 **입력에서 계산된 통계량**(예: 자기상관 피크 위치, 스펙트럼 엔트로피)의 임베딩으로 두면 OOV 문제(Q8)도 동시에 완화됩니다. 비용은 $k \cdot hf$ 파라미터로 여전히 미미합니다.

**B. 사전학습 단계의 채널 상관 학습 — "가변 채널 그룹 마스킹"**
현재는 다변량을 완전히 분해합니다. 대안으로, 각 사전학습 배치에서 채널을 $g \sim \mathcal{U}\{1, G\}$개씩 무작위 그룹핑하여 inter-channel mixer를 확률적으로 활성화하면, **채널 수에 불변인 상관 사전지식**을 백본에 주입할 수 있습니다. Moirai의 any-variate flattening과 달리 그룹 크기를 제한하므로 계산량이 통제됩니다.

**C. Frozen backbone 가정의 검증과 완화**
Q12(full fine-tuning 대비 격차)를 먼저 측정한 뒤, 격차가 크다면 **LoRA류 저랭크 적응**을 백본에 적용:

$$W' = W + \frac{\alpha}{r} BA, \qquad B \in \mathbb{R}^{d\times r},\; A \in \mathbb{R}^{r \times d},\; r \ll d$$

1M 모델에서 $r=4$ 수준이면 추가 파라미터는 수만 개에 그쳐 "tiny" 정체성을 유지하면서 도메인 시프트 대응력을 확보할 수 있습니다.

**D. 컨텍스트 길이 불변성 (F2에 대한 구체적 경로)**
현재 $sl$마다 별도 모델이 필요한 이유는 patch 수 $n = sl/pl$이 MLP 입력 차원을 고정하기 때문입니다. Inter-patch mixer의 MLP를 **patch 축 pooling(예: attention pooling 또는 set-transformer 스타일 집약)**으로 대체하면 $n$ 가변성을 흡수할 수 있으며, 이는 MLP-Mixer 계열의 알려진 약점(입력 해상도 고정)에 대한 표준적 처방입니다.

**E. 스케일링 곡선 확보 (Q4)**
{1M, 5M, 15M, 50M} × {250M, 1B, 4B samples} 그리드로 $\text{MSE} \approx \alpha N^{-a} + \beta D^{-b} + \epsilon$ 형태를 적합하면, "tiny가 충분한 구간"의 경계를 정량화할 수 있습니다. Figure 3이 던진 질문에 대한 정면 답변이 됩니다.

**F. 평가 프로토콜 정비 (일반화 주장의 신뢰도 확보)**
- 최소 5시드 × sliding-window 전체 평가 + 신뢰구간
- MSE 외 MASE/CRPS 병기
- D1(7개) 외 광역 벤치마크 추가
- 사전학습 도메인과 **의도적으로 먼** hold-out 도메인 세트 구성 (Q9)

---

### 8-2. 2020년 이후 관련 연구 비교 분석 · 영향 · 향후 고려사항

> **⚠ 검증 범위 고지**: 아래 (A)는 **업로드된 논문의 참고문헌 목록과 Appendix B에서 직접 확인 가능한 내용**입니다. (B)는 **제 사전 지식에 기반하며 이 대화에서 검증할 수 없었습니다** — 인용 전 원문 확인을 권합니다. (C)는 제 분석입니다.

#### (A) 논문 내에서 검증 가능한 계보 (Appendix B, p.16–18 + References)

**B.1 계열 — 단일 도메인 학습 아키텍처의 진화**

| 연도 | 모델 | 논문이 서술한 위치 | 핵심 |
|---|---|---|---|
| 2020 | N-BEATS [23] (ICLR 2020) | Appendix B.1 | 다중 시계열 학습 가능하나 univariate |
| 2020 | DeepAR [28] (IJF 2020) | Appendix B.1 | 자기회귀 확률 예측, cross-channel 무시 |
| 2021 | Informer [44] (AAAI 2021) | B.1 | 장기 예측용 효율적 attention |
| 2021 | Autoformer [38] (NeurIPS 2021) | B.1 | 분해 + Auto-Correlation |
| 2022 | FEDformer [45] (ICML 2022) | B.1 | 주파수 강화 분해 |
| 2022 | DLinear [41] | B.1 | **"embarrassingly simple linear model"이 Transformer를 이김** → Transformer 유효성 논쟁 촉발 |
| 2023 | PatchTST [22] (ICLR 2023) | B.1 | 패칭 + channel-independence로 Transformer 복권 |
| 2023 | **TSMixer [6]** (KDD 2023, 동일 저자) | B.1, Appendix A | MLP-Mixer 기반, 2–3X 속도·메모리 절감 → **TTM의 직접 모체** |
| 2023 | TimesNet [37] (ICLR 2023) | B.1 | 다주기 분해 + Inception |
| 2024 | iTransformer [19] (ICLR 2024) | B.1 | 축 반전(variate token) |
| 2024 | TimeMixer [33] (ICLR 2024) | B.1 | 다중 스케일 mixing |

**B.2 계열 — TS 파운데이션 모델 (2024년 집중 출현)**

| 모델 | 사전학습 방식 | 규모 | 논문 서술 위치 |
|---|---|---|---|
| Lag-Llama [26] | decoder-only, lag를 covariate로, univariate | 2.4M | Appendix B.2 |
| Moment [10] (ICML'24) | **Transformer encoder + mask reconstruction**, "Time Series Pile" | 348M | B.2 |
| TimesFM [3] (ICML'24) | decoder-only causal attention, 실데이터+합성, **300B time-points** | 200M | B.2, §4.9 |
| Chronos [2] | **토큰화 + T5 LLM**, 자기회귀 샘플링 + dequantization | 8M–709M | B.2 |
| Moirai [35] (ICML'24) | encoder + any-variate flattening, LOTSA **27B time-points** | 14M–311M | B.2, §4.9 |
| TimeGPT-1 [8] | 클로즈드 소스 → 비교 제외 | – | B.2 |
| **TTM (본 논문)** | **direct forecasting, 1B samples, TSMixer** | **1M–5M** | – |

**B.3 계열 — LLM 기반 TS**: LLMTime [11] (NeurIPS'23, 텍스트화 zero-shot), GPT4TS [46] (NeurIPS'23, 임베딩·norm·출력층만 튜닝), Time-LLM [15] (ICLR'24, reprogramming), UniTime [18] (WWW'24). 논문의 비판: **cross-channel 상관 미모델링 + 거대 크기·느린 실행** (Appendix B.3, p.18).

**자기지도 전이 계열**: TS2Vec [40], CoST [36], TF-C [43], LaST [34], Ti-MAE [17], TST [42], SimMTM [5]. 논문의 비판: **데이터셋 쌍이 사전 선별되어(예: ETTH2→ETTH1) 진정한 out-of-domain 전이를 증명하지 못함** (Appendix B.2, p.17–18).

`[해석]` **TTM의 좌표**: 위 계보에서 TTM은 두 축의 교차점에 있습니다 — (1) **효율 축**: DLinear → TSMixer → TTM (Transformer 회피 노선), (2) **전이 축**: SimMTM → Moirai/Chronos → TTM (범용 사전학습 노선). 2024년 이전에는 이 두 축이 분리되어 있었고("효율적이려면 도메인 내 학습, 전이하려면 대형"), TTM은 **두 축이 양립 가능함을 처음 실증**했다는 것이 논문의 주장(§1 capability (1))입니다.

#### (B) 2024년 하반기 이후 흐름 — **미검증, 확인 필요**

> 아래는 제 사전 지식에 기반한 것으로, **이 대화에서 원문을 확인하지 못했습니다.** 세부 수치·발표 연도·저자에 오류가 있을 수 있으므로 인용 전 반드시 원문을 확인하시기 바랍니다. 확신도가 낮은 항목은 명시했습니다.

| 방향 | 대표 흐름 (미검증) | TTM과의 관계 |
|---|---|---|
| **MoE 기반 TS FM** | Time-MoE, Moirai-MoE 계열 — 희소 활성화로 "총 파라미터는 크되 활성 파라미터는 작게" | TTM과 **목표는 같고 수단이 반대**입니다. TTM은 dense-tiny, MoE는 sparse-large. 추론 지연 관점에서 직접 경쟁 관계 |
| **통합 TS 모델** | UniTS, Timer 계열 — 예측/분류/이상탐지/대치를 단일 모델로 | TTM의 후속 계획 F1과 동일 방향. TTM이 이 경쟁에서 뒤처진 영역 |
| **광역 벤치마크** | GIFT-Eval 등 다도메인·다지평 표준 벤치마크 제안 | §6 Q19에 대한 커뮤니티의 답. **ETT 7종 중심 평가의 한계가 이후 광범위하게 지적**됨 |
| **효율적 Chronos** | Chronos-Bolt 등 자기회귀 제거를 통한 대폭 가속 | TTM의 속도 우위(Table 3의 240,000X)를 **크게 잠식**했을 가능성. Table 3 수치는 2024년 초 시점의 스냅샷으로 읽어야 함 |
| **컨텍스트 길이 유연성** | 가변 컨텍스트 지원 아키텍처 | TTM의 F2 및 §6 Q2에 해당 |
| **확률적 예측 표준화** | CRPS/quantile loss 중심 평가 정착 | TTM의 최대 미비점 |

`[해석]` **가장 중요한 시간 민감성**: 이 논문의 **정확도 우위 주장(C1–C3)은 2024년 초 기준**이며, 이후 baseline들의 후속 버전이 나왔다면 재평가가 필요합니다. 반면 **효율 주장(C4)의 구조적 근거(비자기회귀 + 소형)**와 **데이터 다양성 가설(C8)**은 시간에 덜 민감한 기여입니다.

#### (C) `[해석]` 향후 연구에 미치는 영향

1. **"TS에는 LLM식 scaling law가 그대로 적용되지 않는다"는 반례 제시.** Figure 3은 TimesFM 300B·Moirai 27B 대비 1B 샘플·1M 파라미터로 경쟁 가능함을 보였고, 이는 **데이터 큐레이션 연구**를 아키텍처 연구와 동등한 지위로 끌어올렸습니다. 이후 TS FM 논문이 "우리는 어떤 다양성 축을 커버했는가"를 명시하게 만든 계기 중 하나로 평가할 수 있습니다.
2. **엣지/CPU 배포라는 새 설계 목표 정립.** Table 3의 CPU 열은 이전 TS FM 논문들이 보고하지 않던 항목입니다. "GPU 없이 돌아가는 파운데이션 모델"이라는 요구사항을 벤치마크에 편입시켰습니다.
3. **Exogenous 지원의 표준화 압력.** 대부분의 TS FM이 univariate·no-covariate였던 상황에서, TTM은 **외생변수 처리를 FM의 필수 요건으로 제기**했습니다. 실무(소매 수요, 에너지, 관측성)에서 이는 결정적입니다.
4. **비-Transformer 노선의 정당화.** DLinear가 던진 질문("Transformer가 정말 필요한가")을 **파운데이션 모델 스케일에서** 다시 확인했습니다.
5. **재현 가능성 기여.** 가중치와 코드가 공개되었고(Abstract), 사전학습 데이터 목록이 Table 8에 완전히 명시되어 있어 재현 장벽이 낮습니다.

#### (D) `[해석]` 이 논문을 활용/후속 연구할 때 반드시 고려할 점

| # | 고려사항 |
|---|---|
| 1 | **Table 1과 Table 2를 절대 함께 인용하지 말 것.** 프로토콜이 다릅니다. TTM을 baseline으로 쓸 때는 **sliding-window 전체 평가(Table 11/13)**를 기준으로 삼으십시오 |
| 2 | **"1M 파라미터"는 FL별 모델의 평균**입니다. 총 배포 비용을 비교할 때는 FL 개수를 곱하거나, pruning 전략(Figure 4)을 전제로 명시하십시오 |
| 3 | **재현 시 가중치 버전 확인.** 논문 평가 가중치와 엔터프라이즈 배포 가중치는 사전학습 데이터가 다릅니다(Table 8 caption) |
| 4 | **Exogenous 결과를 일반화하지 말 것.** $TTM_Q$ (가장 약한 변형) × 4개 소규모 데이터셋 결과입니다 |
| 5 | **MSE 단일 지표의 한계.** 확률 모델(Chronos/Moirai/Lag-Llama)과 비교할 때는 CRPS를 병기해야 공정합니다 |
| 6 | **Traffic 도메인 근접성 유의.** LibCity 교통 데이터로 사전학습한 모델을 Traffic에서 평가하는 것은 완전한 out-of-domain이 아닙니다 |
| 7 | **재실행 시 시드 분산부터 측정.** 논문의 1–5% 개선폭이 여러분의 환경에서 노이즈 범위를 넘는지 먼저 확인하십시오 |
| 8 | **Time-sensitivity.** 2024년 초 baseline 대비 결과입니다. 최신 비교가 필요하면 최근 버전으로 재실행이 필수입니다 |

---

## 참고자료 목록

**주 자료 (직접 분석)**
- Ekambaram, V., Jati, A., Dayama, P., Mukherjee, S., Nguyen, N. H., Gifford, W. M., Reddy, C., Kalagnanam, J. (IBM Research). *Tiny Time Mixers (TTMs): Fast Pre-trained Models for Enhanced Zero/Few-Shot Forecasting of Multivariate Time Series.* arXiv:2401.03955v8 [cs.LG], 2024-11-07. NeurIPS 2024 채택. — **업로드된 PDF, 본 답변의 모든 수치·인용의 유일한 검증된 출처**

**논문 내 참고문헌 중 본 답변에서 직접 언급한 항목** (모두 위 PDF의 References, p.13–15에서 확인)
- [2] Ansari et al., *Chronos: Learning the Language of Time Series*, arXiv:2403.07815
- [3] Das, Kong, Sen, Zhou, *A Decoder-Only Foundation Model for Time-Series Forecasting* (TimesFM), ICML
- [5] Dong et al., *SimMTM*, NeurIPS 2023
- [6] Ekambaram et al., *TSMixer: Lightweight MLP-Mixer Model for Multivariate Time Series Forecasting*, KDD 2023
- [7] Fanaee-T, *Bike Sharing Dataset*, UCI ML Repository
- [8] Garza & Mergenthaler-Canseco, *TimeGPT-1*
- [9] Godahewa et al., *Monash Time Series Forecasting Archive*, NeurIPS D&B 2021
- [10] Goswami et al., *MOMENT: A Family of Open Time-Series Foundation Models*, ICML 2024
- [11] Gruver et al., *Large Language Models Are Zero-Shot Time Series Forecasters* (LLMTime), NeurIPS 2023
- [13] Jablonka et al., *Machine Learning for Industrial Processes: Forecasting Amine Emissions from a Carbon Capture Plant*, Science Advances 9(1), 2023
- [15] Jin et al., *Time-LLM*, ICLR 2024
- [16] Li & Liang, *Prefix-Tuning*, ACL-IJCNLP 2021
- [18] Liu et al., *UniTime*, WWW 2024
- [19] Liu et al., *iTransformer*, ICLR 2024
- [20] Liu et al., *Swin Transformer*, ICCV 2021
- [22] Nie, Nguyen, Sinthong, Kalagnanam, *A Time Series Is Worth 64 Words* (PatchTST), ICLR 2023
- [23] Oreshkin et al., *N-BEATS*, ICLR 2020
- [26] Rasul et al., *Lag-Llama*, arXiv:2310.08278
- [27] BizITOps Dataset Repository, github.com/BizITObs/BizITObservabilityData
- [28] Salinas et al., *DeepAR*, IJF 36(3), 2020
- [30] Tolstikhin et al., *MLP-Mixer*, NeurIPS 2021
- [32] Wang et al., *LibCity*, SIGSPATIAL 2021
- [33] Wang et al., *TimeMixer*, ICLR 2024
- [35] Woo et al., *Unified Training of Universal Time Series Forecasting Transformers* (Moirai), ICML 2024
- [37] Wu et al., *TimesNet*, ICLR 2023
- [38] Wu et al., *Autoformer*, NeurIPS 2021
- [41] Zeng et al., *Are Transformers Effective for Time Series Forecasting?* (DLinear), arXiv:2205.13504
- [44] Zhou et al., *Informer*, AAAI 2021
- [45] Zhou et al., *FEDformer*, ICML 2022
- [46] Zhou et al., *One Fits All* (GPT4TS), NeurIPS 2023

**논문 내 명시된 데이터/코드 링크** (PDF Appendix C.1, D.4)
- Monash Forecasting Repository: `https://forecastingdata.org/`
- LOTSA (LibCity 하위집합): `https://huggingface.co/datasets/Salesforce/lotsa_data/tree/main`
- Autoformer 데이터 저장소: `https://github.com/thuml/Autoformer`
