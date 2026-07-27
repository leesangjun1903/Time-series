# TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables

## 1. Executive Summary (10문장 이내)

1. TimeXer는 예측 대상인 **내생변수(endogenous)** 와 예측할 필요는 없지만 정보를 제공하는 **외생변수(exogenous)** 를 구분해 다루는 "exogenous variables를 활용한 시계열 예측" 패러다임을 정면으로 다룬 Transformer 모델이다 (p.1, Abstract).
2. 기존 다변량 예측 모델은 모든 변수를 동등하게 취급하거나(iTransformer, Crossformer), 채널 독립 가정으로 외생 정보를 무시해(PatchTST) 이 설정에 최적화되어 있지 않다 (p.2, Table 1).
3. 핵심 아이디어는 **표현 수준의 비대칭성**이다: 내생변수는 patch-level 토큰으로, 외생변수는 series 전체를 하나로 압축한 variate-level 토큰으로 임베딩한다 (Eq. 2–3, p.4).
4. 두 granularity 사이의 정보 불일치를 해소하기 위해 ViT의 [CLS] 토큰에서 착안한 **learnable global token**을 내생변수마다 도입하여 "다리(bridge)" 역할을 시킨다 (p.4).
5. 구조적으로는 canonical Transformer를 **전혀 수정하지 않고**, patch-wise self-attention(내생 내부)과 variate-wise cross-attention(외생→global token)을 순차 적용한다 (Figure 2, Eq. 4–7).
6. 단기 전력가격 예측 5개 데이터셋(EPF)에서 평균 MSE 0.307 / MAE 0.265로 전 baseline 대비 1위이며, 2위 PatchTST(0.330/0.282) 대비 MSE 약 7.0% 개선을 보고한다 (Table 2, p.7).
7. 장기 다변량 예측 7개 벤치마크에서는 Traffic을 제외한 6개에서 1위를 기록했고, Traffic만 iTransformer(0.428)가 TimeXer(0.466)를 앞선다 (Table 3, p.7; Appendix G).
8. 외생변수를 zeros/random으로 대체해도 성능 저하가 0.307→0.330 수준(약 7.5%)에 그치는 반면, 내생변수를 훼손하면 0.307→1.125로 붕괴해 **내생 표현이 예측을 지배**함을 보인다 (Table 5, p.9).
9. 효율성 측면에서 외생 방향 복잡도가 $O(C)$로 iTransformer의 $O(C^2)$보다 유리하며, 3,850개 기상관측소·36개 외생변수의 대규모 실험(MSE 0.200)에서도 baseline을 앞선다 (Figure 4, Figure 5-Right, Appendix E).
10. 다만 **오차막대·유의성 검정이 사실상 부재**하고, 표 간 수치 불일치와 Weather 내생변수 정의 모순 등 검증상의 약점이 존재한다 (5절 참조).

### 1-1. 연구의 목적과 필요성

**목적**: "예측하지 않아도 되지만 정보는 제공하는 변수"를 Transformer가 **구조 변경 없이** 원리적으로 흡수하도록 만드는 것.

**필요성 (논문이 제시한 4가지 논거)**:

| 논거 | 내용 | 근거 위치 |
|---|---|---|
| ① 실용적 필연성 | 전력가격은 수급이라는 외부 요인에 종속되어, 과거 가격만으로는 원리적으로 예측 불가 | p.1, 1절 |
| ② 계산·노이즈 비용 | 외생변수를 내생변수와 동등 취급하면 시간·메모리 복잡도 증가 + 불필요한 내생→외생 상호작용 발생 | p.2 |
| ③ 인과적 시차 | 외부 요인은 내생 계열에 시차를 두고 인과적 영향을 미치므로 이를 추론할 수 있어야 함 | p.2 |
| ④ 이질성 대응 | 실제 외생 계열은 결측·시점 불일치·주기 불일치·길이 불일치를 겪음 | Figure 1 (Left), p.2 |

특히 ④는 기존 방법(NBEATSx, TiDE, TFT)이 "각 시점에서 내생·외생 feature를 concat 후 latent space로 사상"하기 때문에 **시간축 정렬을 강제**한다는 점에서 결정적 차별점으로 제시된다 (2.2절, p.3–4).

---

## 2. 핵심 주장과 근거 정리

| # | 저자의 핵심 주장 | 제시 근거 | 근거 위치 |
|---|---|---|---|
| C1 | 외생변수 전용 설계 패러다임이 필요하다 | 기존 8개 모델의 Univariate/Multivariate/Exogenous 지원 여부 비교표 | **Table 1**, p.3 |
| C2 | 내생=patch, 외생=variate의 비대칭 임베딩이 최적 | 외생을 patch로 바꾸면(Replace) EPF 평균 0.307→0.312, 장기 ETTh1 0.073→0.075 | **Table 4** (p.8), **Table 8** (p.16) |
| C3 | Global token이 필수적인 bridge다 | Global token 제거 시(Remove) EPF 평균 0.307→0.316 (MSE +2.9%) | **Table 4**, p.8 |
| C4 | Cross-attention이 단순 add/concat보다 우수 | Add 0.329, Concatenate 0.312 vs Ours 0.307 | **Table 4**, p.8 |
| C5 | 단기 외생 예측에서 SOTA | EPF 5개 전부 MSE/MAE 1위, 평균 0.307/0.265 | **Table 2**, p.7 |
| C6 | 다변량 예측으로 일반화 가능 | 7개 중 6개 1위, 세부 horizon 기준 1st Count 30(MSE)/22(MAE) | **Table 3** (p.7), **Table 12** (p.24) |
| C7 | 계열 이질성에 강건 | 외생 zeros 0.330 / random 0.333 vs 정상 0.307 | **Table 5**, p.9 |
| C8 | look-back 불일치를 수용 | 내생/외생 look-back을 독립적으로 {96…720} 변화시켜도 동작·개선 | **Figure 3**, p.8 |
| C9 | $C$가 클 때 효율적 | Cross-attention은 $O(C)$, iTransformer는 $O(C^2)$; ECL(320 exog) 메모리 우위 | **Figure 5-Right** (p.10), **Appendix E** (p.17) |
| C10 | 대규모 확장성 보유 | 3,850 관측소 + 36 외생, MSE 0.200 (iTrans. 0.207, PatchTST 0.208) | **Figure 4**, p.9 |
| C11 | Attention map이 해석 가능 | CO2 농도 ↔ Air Density 높은 attention, Max Wind Velocity 낮음 | **Figure 5-Left**, p.10 |
| C12 | 표현 품질이 우수 | 첫/마지막 블록 간 CKA 유사도 높음; iTransformer는 전체 CKA는 높으나 내생변수 표현은 미학습 | **Figure 9**, p.18 |

### 2-1. 문제·방법·구조·성능·한계 상세

#### (a) 해결하고자 하는 문제 (Problem Setting, p.4)

내생 계열 $x_{1:T} = \{x_1,\dots,x_T\} \in \mathbb{R}^{T\times 1}$ 과 외생 계열 집합

$$z_{1:T_{ex}} = \{z^{(1)}_{1:T_{ex}}, z^{(2)}_{1:T_{ex}}, \dots, z^{(C)}_{1:T_{ex}}\} \in \mathbb{R}^{T_{ex}\times C}$$

가 주어졌을 때, 미래 $S$ 스텝을 예측:

$$\hat{x}_{T+1:T+S} = \mathcal{F}_\theta\left(x_{1:T},\, z_{1:T_{ex}}\right) $$

핵심은 **$T_{ex} \neq T$ 를 명시적으로 허용**한다는 점이다 (p.4). 이는 기존 covariate 모델이 강제하는 시간축 정렬 제약을 푸는 것이 목표임을 뜻한다.

#### (b) 제안 방법 — 수식

**① 내생 임베딩 (Eq. 2)** — 비중첩 patch + 학습 가능 global token

$$
\begin{aligned}
\{s_1, s_2, \dots, s_N\} &= \mathrm{Patchify}(x) \\
\mathbf{P}_{\text{en}} &= \mathrm{PatchEmbed}(s_1, s_2, \dots, s_N) \in \mathbb{R}^{N \times D}\\
\mathbf{G}_{\text{en}} &= \mathrm{Learnable}(x) \in \mathbb{R}^{1 \times D}
\end{aligned}
$$

여기서 $N = \left\lfloor \dfrac{T}{P} \right\rfloor$ (patch 개수), $P$ = patch 길이, $D$ = 모델 차원.

**② 외생 임베딩 (Eq. 3)** — 계열 전체를 단일 토큰으로

$$\mathbf{V}_{\text{ex},i} = \mathrm{VariateEmbed}\left(z^{(i)}\right), \quad i \in \{1,\dots,C\} $$

$$\mathrm{VariateEmbed}: \mathbb{R}^{T_{ex}} \to \mathbb{R}^{D}, \qquad \mathbf{V}_{\text{ex}} = \{\mathbf{V}_{\text{ex},i}\}_{i=1}^{C}$$

**③ 내생 Self-Attention (Eq. 4)** — global token의 비대칭 역할

$$
\begin{aligned}
\text{Patch-to-Patch:}\quad & \widehat{\mathbf{P}}^{l,1}_{\text{en}} = \mathrm{LayerNorm}\left(\mathbf{P}^l_{\text{en}} + \mathrm{Self\text{-}Attention}\left(\mathbf{P}^l_{\text{en}}\right)\right) \\
\text{Global-to-Patch:}\quad & \widehat{\mathbf{P}}^{l,2}_{\text{en}} = \mathrm{LayerNorm}\left(\mathbf{P}^l_{\text{en}} + \mathrm{Cross\text{-}Attention}\left(\mathbf{P}^l_{\text{en}}, \mathbf{G}^l_{\text{en}}\right)\right) \\
\text{Patch-to-Global:}\quad & \widehat{\mathbf{G}}^l_{\text{en}} = \mathrm{LayerNorm}\left(\mathbf{G}^l_{\text{en}} + \mathrm{Cross\text{-}Attention}\left(\mathbf{G}^l_{\text{en}}, \mathbf{P}^l_{\text{en}}\right)\right)
\end{aligned}
$$

이 세 연산은 결국 concat 후 단일 self-attention과 등가로 축약된다 (Eq. 5):

$$\widehat{\mathbf{P}}^l_{\text{en}}, \widehat{\mathbf{G}}^l_{\text{en}} = \mathrm{LayerNorm}\left(\left[\mathbf{P}^l_{\text{en}}, \mathbf{G}^l_{\text{en}}\right] + \mathrm{Self\text{-}Attention}\left(\left[\mathbf{P}^l_{\text{en}}, \mathbf{G}^l_{\text{en}}\right]\right)\right) $$

$l \in \{0,\dots,L-1\}$, $\mathbf{P}^0_{\text{en}} = \mathbf{P}\_{\text{en}}$, $\mathbf{G}^0_{\text{en}} = \mathbf{G}_{\text{en}}$. 어텐션 맵 크기는 $(N+1)\times(N+1)$ (Figure 2c).

> 📌 **이것이 논문의 가장 우아한 지점**: "canonical Transformer를 수정하지 않았다"는 주장은 Eq. 4의 세 가지 개념적 경로가 Eq. 5의 단일 표준 self-attention으로 정확히 환원되기 때문에 성립한다.

**④ 외생→내생 Cross-Attention (Eq. 6)** — global token만이 query

$$\widehat{\mathbf{G}}^l_{\text{en}} = \mathrm{LayerNorm}\left(\widehat{\mathbf{G}}^l_{\text{en}} + \mathrm{Cross\text{-}Attention}\left(\widehat{\mathbf{G}}^l_{\text{en}}, \mathbf{V}_{\text{ex}}\right)\right) $$

어텐션 맵 크기는 $1 \times C$ (Figure 2d). 외생 토큰끼리의 상호작용은 **의도적으로 생략**된다.

**⑤ FFN (Eq. 7) 및 손실 (Eq. 8)**

$$\mathbf{P}^{l+1}_{\text{en}} = \mathrm{Feed\text{-}Forward}\left(\widehat{\mathbf{P}}^l_{\text{en}}\right), \quad \mathbf{G}^{l+1}_{\text{en}} = \mathrm{Feed\text{-}Forward}\left(\widehat{\mathbf{G}}^l_{\text{en}}\right) $$

$$\mathrm{Loss} = \sum_{i=1}^{S}\left\|x_i - \hat{x}_i\right\|_2^2, \quad \text{where } \hat{x} = \mathrm{Projection}\left(\left[\mathbf{P}^L_{\text{en}}, \mathbf{G}^L_{\text{en}}\right]\right) $$

**⑥ 병렬 다변량 확장 (p.6)**: 각 변수를 차례로 내생변수로 두고 나머지를 외생변수로 취급, channel independence + attention layer 공유로 병렬 수행.

#### (c) 모델 구조 요약 (Figure 2)

```
Endogenous x ──► Patchify ──► PatchEmbed ──┐
                 + Learnable Global Token  ├─► [P, G] ──► Self-Attn ──► LN
                                            │                  │
Exogenous z^(1..C) ──► VariateEmbed ──► V_ex┘                  ▼
                                              G ──► Cross-Attn(G, V_ex) ──► LN
                                                                │
                                              ──► Feed-Forward ──► LN  (× L blocks)
                                                                │
                                              ──► Linear Projection ──► x̂
```

- 블록 수 $L \in \{1,2,3\}$, $d_{model} \in \{128,256,512\}$, Adam lr $10^{-4}$, 10 epoch + early stopping, NVIDIA 4090 24GB 단일 GPU (Appendix A.2, p.14).
- Patch 길이: 장기 16, 단기 24 (비중첩).
- **중요**: $\mathbf{V}_{\text{ex}}$는 첫 레이어에서 한 번만 계산되어 **모든 블록에서 재사용**된다 (p.10). 이것이 효율성의 원천이자, 동시에 외생 표현이 레이어를 거쳐 정제되지 않는다는 구조적 트레이드오프다.

#### (d) 성능 향상

| 설정 | TimeXer | 2위 모델 | 개선율 |
|---|---|---|---|
| EPF 평균 MSE | **0.307** | PatchTST 0.330 | 7.0% |
| EPF 평균 MAE | **0.265** | PatchTST 0.282 | 6.0% |
| 장기 다변량 ECL | **0.171** | iTransformer 0.178 | 3.9% |
| 장기 다변량 ETTm2 | **0.274** | PatchTST 0.281 | 2.5% |
| 장기 다변량 Traffic | 0.466 | **iTransformer 0.428** | **-8.9% (패배)** |
| 대규모 Weather (3,850 st.) | **0.200** | iTransformer 0.207 | 3.4% |
| 외생 예측 설정 1st Count | 23/23 (MSE/MAE) | Crossformer 7/8 | — |

(출처: Table 2 p.7, Table 3 p.7, Figure 4 p.9, Table 11 p.23)

#### (e) 한계 — 저자 인정분과 미인정분 구분

**저자가 인정한 한계**
1. **Traffic 열세** (Appendix G, p.18): patch 토큰 다수 vs global 토큰 1개의 **불균형** 때문에 급변점(spike)의 정확한 수치를 못 맞추고 추세만 따라가며, MSE의 제곱항이 이 오차를 증폭한다고 설명. PatchTST도 같은 패턴을 보인다는 점을 근거로 제시.
2. **Concatenate 설계가 Traffic에서 우세** (Table 8, p.16): 외생 토큰 간 self-attention이 도움 되는 경우가 존재하나 ETTh1/ETTm1에서는 오히려 악화 → **데이터 의존적**.
3. **이차 복잡도** $O\!\left(\left(\tfrac{T}{P}+1\right)^2\right)$ 잔존 (Appendix E, p.17).
4. **내생 계열 품질 의존성** (Table 5, p.9): 내생 정보가 손상되면 성능이 붕괴 → 외생 정보만으로는 예측 불가.

**제가 판단하는 미인정 한계**

5. **`VariateEmbed`의 구현 미명세**: $\mathbb{R}^{T_{ex}} \to \mathbb{R}^{D}$ 는 **고정 입력 차원의 선형 사상**이다. 논문은 "임의의 look-back 길이 불일치를 수용"한다고 주장하지만(p.4), 이는 * 내생 $T$ 와 외생 $T_{ex}$ 가 다를 수 있다*는 의미일 뿐, **외생 변수들끼리 서로 다른 길이**를 갖는 경우를 처리할 수 있는지는 수식으로도 본문으로도 설명되지 않는다.
6. **주기 불일치의 실제 처리 방식이 회피적**: Figure 4 실험에서 3시간 간격 ERA5를 "baseline이 처리 못하므로 nearest 보간으로 시간별 변환"했다고 기술(p.9). 그렇다면 TimeXer도 동일 전처리를 받은 것인지, 아니면 원본 3시간 데이터를 그대로 썼는지 **불명확**하며, 후자라면 입력 정보량이 달라 공정 비교가 아니다.
7. **"causal"이라는 표현의 남용**: Abstract와 3절에서 "causal information"을 반복하지만, 인과 추론적 검증(개입, Granger 검정 등)은 전무하다. 실제 제시된 것은 attention weight 상관관계뿐이다.
8. **RevIN/정규화 미기술**: 비정상성 처리 방식이 본문에 없다.

---

## 3. 주장별 근거 위치 색인

| 주장 | 위치 |
|---|---|
| 문제 정의 및 4대 실무 이슈 | p.1–2, **Figure 1 (Left)** |
| EPF 레이더 차트 성능 미리보기 | **Figure 1 (Right)**, p.2 |
| 관련 연구 대비 기능 매트릭스 | **Table 1**, p.3 |
| Problem Setting, Eq. (1) | p.4 |
| 내생/외생 임베딩 Eq. (2)(3) | p.4–5 |
| 전체 아키텍처 | **Figure 2**, p.5 |
| Attention Eq. (4)(5)(6)(7) | p.5–6 |
| 손실 Eq. (8) | p.6 |
| 병렬 다변량 확장 | p.6 |
| 단기 EPF 결과 | **Table 2**, p.7 |
| 장기 다변량 결과 | **Table 3**, p.7 |
| Ablation (단기) | **Table 4**, p.8 |
| Look-back 길이 실험 | **Figure 3**, p.8 |
| 결측 강건성 | **Table 5**, p.9 |
| 대규모 확장성 | **Figure 4**, p.9 |
| Attention 해석 + 효율성 | **Figure 5**, p.10 |
| 데이터셋 명세 | **Table 6**, p.14 |
| 구현 상세 | Appendix A.2, p.14 |
| 중첩 patch ablation | **Table 7**, p.15 |
| Patch 길이 민감도 | **Figure 6**, p.15 |
| Ablation (장기) | **Table 8**, p.16 |
| 마스킹 비율 실험 | **Figure 7**, p.16 |
| 복잡도 이론 분석 | Appendix E, p.17 |
| Weather 효율성 + look-back | **Figure 8**, p.17 |
| CKA 표현 분석 | **Figure 9**, p.18 |
| Traffic 실패 사례 논의 | **Figure 10** + Appendix G, p.19 |
| 외생/미래외생 활용 showcase | **Figure 11**, p.20 |
| 정성적 예측 비교 | **Figure 12–14**, p.21 |
| 추가 baseline (단기, ±std) | **Table 9**, p.22 |
| 추가 baseline (GNN 계열) | **Table 10**, p.22 |
| 외생 설정 장기 전체 결과 | **Table 11**, p.23 |
| 다변량 장기 전체 결과 | **Table 12**, p.24 |

---

## 4. 저자 보고 결과 vs 제 해석 (분리)

### 4-1. 연구 주제

| 구분 | 내용 |
|---|---|
| **저자 보고** | "다변량/단변량과 구분되는 제3의 실용적 패러다임: forecasting with exogenous variables" (Abstract, p.1) |
| **제 해석** | 완전히 새로운 문제 정의라기보다, **계량경제학의 ARIMAX/SARIMAX 프레이밍을 딥러닝 아키텍처 설계 원리로 재수입**한 것에 가깝다. 논문 스스로 2.2절에서 이 계보를 인정한다. 기여의 실질은 "문제 발견"이 아니라 **"표현 granularity를 변수 역할에 따라 차등화한다"는 아키텍처 원리**이며, 이 원리 자체는 일반화 가치가 크다. |

### 4-2. 방법

| 구분 | 내용 |
|---|---|
| **저자 보고** | Eq. 4의 세 경로(P2P, G2P, P2G)를 Eq. 5의 단일 self-attention으로 축약 가능하므로 "Transformer 컴포넌트 무수정" (p.5) |
| **제 해석** | 수학적으로 타당하지만, **Eq. 4를 별도 제시한 것은 사후적 서술**이다. 실제 구현은 Eq. 5(concat 후 표준 attention)이며, Eq. 4는 그 해석에 불과하다. 즉 실질적 신규성은 **"global token 1개를 concat했다"** 한 줄이고, 나머지는 ViT의 [CLS] 토큰 재사용이다. 이를 폄하할 필요는 없다 — 오히려 **최소 개입으로 큰 효과**를 냈다는 것이 강점이다. |
| **저자 보고** | 외생변수를 variate-level로 압축하는 것이 "더 자연스럽다" (p.4, Appendix B.3) |
| **제 해석** | Table 8이 보여주는 것은 "patch로 바꾸면 약간 나쁘다"(ETTh1 0.073→0.075)이지, "variate가 원리적으로 옳다"가 아니다. 격차가 작다는 사실은 오히려 **성능이 아니라 계산 효율이 진짜 근거**임을 시사한다. 논문도 "patch-wise는 계산량이 크게 증가한다"를 함께 언급하는데(p.16), 이 두 근거의 비중을 명확히 구분하지 않은 것은 서술상의 약점이다. |
| **저자 보고** | Cross-attention으로 $O(C)$ 달성, iTransformer는 $O(C^2)$ (Appendix E) |
| **제 해석** | 정확하다. 다만 이는 **외생 토큰 간 상호작용을 포기한 대가**이며, 저자도 Traffic에서 이 포기가 손해였음을 Table 8로 자인한다. 즉 $O(C)$는 무료가 아니라 **표현력과의 명시적 트레이드오프**다. |

### 4-3. 결과

| 구분 | 내용 |
|---|---|
| **저자 보고** | EPF 5개 전부 SOTA, 평균 MSE 7.0% 개선 (Table 2) |
| **제 해석** | EPF에서의 우위는 **신뢰할 만하다**. Table 9의 5-seed 표준편차(NP 0.236±0.004, FR 0.385±0.005)에 비해 2위와의 격차(NP 0.031, FR 0.026)가 5–7σ 수준이기 때문이다. 다만 **baseline의 표준편차가 없어** 엄밀한 검정은 여전히 불가능하다. |
| **저자 보고** | "장기 다변량에서 대부분 SOTA" (p.7) |
| **제 해석** | **과대 서술이다.** ETTm1은 0.382 vs PatchTST 0.387로 격차 1.3%, Weather 외생 설정(Table 11)은 MSE가 소수 3자리에서 0.002로 동률이다. 이 수준은 seed 변동 범위 내일 개연성이 높다. 실질적으로 **의미 있는 우위는 ECL, ETTh2, EPF에 한정**된다고 보는 것이 정직하다. |
| **저자 보고** | Table 4에서 "모든 데이터셋에서 모든 설계 대비 우수" (p.7 "superior performance … across all datasets") |
| **제 해석** | **본문 주장과 표가 불일치한다.** Table 4에서 BE의 MSE는 Replace(0.376)가 Ours(0.379)보다 낮고, NP의 MAE는 Concatenate(0.266)가 Ours(0.268)보다 낮다. "평균적으로 우수"가 정확한 표현이다. |
| **저자 보고** | Table 5의 강건성은 "설계의 우수성" 때문 (p.9) |
| **제 해석** | **인과관계가 뒤집혀 있을 수 있다.** 외생변수를 0으로 채워도 7.5%만 나빠진다는 것은 강건성인 동시에 **모델이 외생 정보를 그다지 많이 쓰지 않는다**는 증거이기도 하다. 논문의 핵심 판매 포인트가 "외생변수 활용"인데, ablation은 외생 기여도가 전체 성능의 10% 미만임을 시사한다. 이 긴장은 논문에서 해소되지 않는다. |
| **저자 보고** | Figure 5-Left의 attention이 물리적으로 타당 (CO2 ↔ Air Density) (p.10) |
| **제 해석** | 저자 스스로 "유사한 형상의 계열이 높은 attention을 받는 경향"을 인정하는데(p.10), 이는 **attention이 인과가 아니라 형상 유사도를 포착**한다는 자백에 가깝다. CO2–공기밀도 사례는 단일 cherry-picked 예시이며 정량 지표(예: 알려진 인과 그래프와의 일치율)가 없다. **해석가능성 주장은 근거가 가장 약한 부분**이다. |

---

## 5. 통계적으로 취약한 부분 / 비교 불가능한 수치

### 5-1. 통계적 취약점

| # | 문제 | 상세 |
|---|---|---|
| S1 | **오차막대 거의 전무** | Table 9(EPF, TimeXer만, 5 seed)를 제외하면 Table 2·3·11·12 전부 단일 수치. Baseline의 분산은 **어디에도 없음**. |
| S2 | **유의성 검정 부재** | 전력가격 예측 벤치마크의 표준 관행인 **Diebold-Mariano 검정**이 없다. 이 벤치마크의 출처인 Lago et al. (2021) 자체가 DM 검정과 rMAE를 권고하는데, 논문은 벤치마크 데이터만 차용하고 평가 프로토콜은 차용하지 않았다. |
| S3 | **평가지표 비표준** | EPF 도메인 표준(sMAPE, rMAE, MASE) 대신 MSE/MAE만 사용 → 선행 EPF 문헌과 직접 대조 불가. |
| S4 | **하이퍼파라미터 탐색 비대칭** | TimeXer는 $L\in\{1,2,3\}$, $d_{model}\in\{128,256,512\}$ 탐색(p.14). Baseline의 탐색 범위·방법은 "TimesNet repo 기반 재현"이라고만 서술 → **탐색 예산 불균형 가능성**. |
| S5 | **"1st Count" 지표의 오도성** | Table 11(23/23), Table 12(30/22)의 카운트는 horizon 96/192/336/720이 강한 상관을 갖는데도 독립 승수처럼 집계된다. 실질 독립 표본은 데이터셋 수(7)에 가깝다. |
| S6 | **Figure 4·5의 수치 출처 미공개** | 막대/버블 차트에만 존재하고 대응 표가 없어 재현 검증 불가. Figure 4의 0.200 vs 0.207 격차(3.4%)는 오차막대 없이는 판단 불가. |
| S7 | **Figure 7 시작점 불일치** | 마스크 0%의 NP MSE가 Table 2의 0.236과 어긋나 보임(그래프 판독 기준 약 0.245). 다른 학습 설정일 가능성이 있으나 설명 없음. |

### 5-2. 비교 불가능 / 내부 모순 수치

| # | 항목 | 불일치 내용 |
|---|---|---|
| M1 | **PatchTST ECL** | Table 3: 0.205 / 0.290 ↔ Table 12 (Avg): **0.216 / 0.304**. 동일 설정인데 값이 다름. |
| M2 | **TiDE ECL MAE** | Table 3: **0.244** ↔ Table 12: **0.344**. Table 3 쪽이 오타로 보임(0.244면 TimeXer보다 우수해져 본문 서술과 모순). |
| M3 | **TimeXer Traffic / ETTh2** | Table 3: 0.466 / 0.367 ↔ Table 10: **0.467 / 0.366**. |
| M4 | **Weather 내생변수 정의 모순** | Appendix A.1 본문: "**Wet Bulb** factor를 내생변수로 사용" ↔ Table 6 및 4.4절 Figure 5: "**CO2-Concentration**". **둘 중 하나는 오류이며, 이는 Weather 결과 전체의 해석에 영향**을 준다. |
| M5 | **TFT 인용 오류** | Table 1은 "TFT **[16]**"으로 표기하나 [16]은 Lea et al.의 *Temporal Convolutional Networks*. 올바른 참조는 [20] (Lim et al.)이며 2.2절 본문은 [20]으로 정확히 표기 → 표 오기. |
| M6 | **Table 10의 GNN baseline** | MSGNet ETTh1이 "**0.0.452**"로 표기(오타). 또한 FourierGNN/MSGNet은 다수 셀이 '-'이며 원논문 인용값으로 추정 → **동일 프로토콜 재현이 아니므로 직접 비교 불가**. |
| M7 | **Table 11 Traffic Crossformer OOM** | 336/720에서 '-'인데 AVG 행도 '-'. 다른 모델은 4개 horizon 평균이므로 **평균 간 비교 자체가 성립하지 않음**. |
| M8 | **Weather 외생 설정 해상도 한계** | Table 11 Weather: TimeXer 0.002, iTrans 0.002, RLinear 0.002, TiDE 0.002 — **소수 3자리에서 전부 동률**. MAE로는 RLinear 0.029 < TimeXer 0.031로 **오히려 TimeXer가 열세**인데 본문은 언급하지 않음. |
| M9 | **표기 충돌** | 3절에서 $L$ = TimeXer 블록 수, $T$ = look-back. 그러나 Appendix E는 $O\!\left(\left(\tfrac{L}{P}+1\right)^2\right)$처럼 $L$을 look-back으로 사용 → **동일 기호의 이중 정의**. |
| M10 | **Table 3 Crossformer ETTm1** | Table 3: 0.512 ↔ Table 12 Avg: 0.513. |

---

## 6. 이 논문이 답하지 않는 질문

**아키텍처 관련**
1. Global token을 2개 이상으로 늘리면 Traffic의 불균형 문제가 실제로 해결되는가? (Appendix G에서 해법으로 **제안만 하고 실험은 없음**)
2. 서로 **길이가 다른 외생변수들**을 동시에 다룰 때 `VariateEmbed`는 변수별 개별 projector인가, 공유 projector + padding인가?
3. 외생변수마다 look-back을 **개별 최적화**하면 어떻게 되는가? (Figure 3은 전체 외생을 일괄 조정)
4. $L$(블록 수)과 $d_{model}$에 대한 민감도 분석이 없다 (patch 길이만 Figure 6에 존재).

**실험 관련**

5. **미래 외생값(known future covariates)** 활용의 정량 결과가 없다. Figure 11(f)는 "가장 좋다"고 정성 서술하나 MSE 수치 표가 부재 — TiDE/NBEATSx/TFT의 핵심 기능이므로 이 비교의 부재는 뼈아프다.
6. TFT·NBEATSx와의 **장기 예측** 비교가 없다 (Table 9는 EPF 단기에만 존재).
7. 진짜 **주기 불일치** 상황에서 보간 없이 실행한 결과가 없다 — Figure 4 실험도 보간 처리를 명시.
8. 진짜 **시점 불일치(temporal misalignment)** 전용 실험이 없다. Figure 1이 4대 문제로 제시했으나 실험은 결측(Table 5, Figure 7)과 길이 불일치(Figure 3)만 다룬다.
9. **확률적 예측 / 불확실성 정량화**를 지원하는가? (점예측만 수행)
10. **분포 이동(distribution shift)** 하에서의 성능, 정규화 기법(RevIN 등) 사용 여부.
11. **추론(inference) 지연시간**은? 보고된 것은 학습 시간(ms/iter)뿐.
12. Traffic처럼 $C=861$인 경우의 메모리 프로파일은? (Figure 5는 ECL의 $C=320$만)

**개념 관련**

13. Global token이 실제로 무엇을 학습하는가? (probing, attention rollout 등 분석 부재)
14. "causal information"이라는 표현의 인과적 근거는? (검증 없음)
15. **외생 정보의 실제 기여분**이 왜 10% 미만인가? — Table 5가 제기하지만 논문이 답하지 않는 가장 중요한 질문.

---

## 7. 가장 중요한 그림 5개 해석

### ① Figure 2 (p.5) — 아키텍처 도식
**저자 보고**: (a) 내생 임베딩이 다수 temporal token + 1개 global token 생성, (b) 외생은 변수당 1개 variate token, (c) 내생 self-attention $(N+1)\times(N+1)$, (d) cross-attention $1\times C$.

**제 해석**: 이 그림의 진짜 정보는 **어텐션 맵의 크기 비대칭**이다. $(N+1)^2$ vs $1\times C$ — 즉 모델이 내생 시간 구조에 압도적으로 많은 파라미터·연산을 배분하고, 외생 정보는 **단일 벡터로 요약되어 한 번에 주입**된다. Table 5의 "외생 제거해도 7.5%만 손실"은 이 설계의 필연적 귀결이지 우연이 아니다. 또한 (d)의 Query가 **global token 단 하나**라는 점은, 외생 정보가 patch별로 차등 전달되지 않고 **모든 patch에 동일하게 방송(broadcast)** 됨을 뜻한다 — 논문이 강조하는 "시차 인과 추론"과는 사실 상충하는 구조다.

### ② Figure 3 (p.8) — Look-back 길이 확장
**저자 보고**: 내생·외생 look-back을 {96, 192, 336, 512, 720}으로 독립 변화시켜도 동작하며, 길이 증가가 성능을 개선. 외생 확장보다 **내생 확장의 이득이 크고**, 둘 다 늘리면 최선.

**제 해석**: 세 패널의 y축 범위(0.12–0.17)가 동일한데, 좌측 패널(외생만 확장)의 개선폭이 우측(양쪽 확장) 대비 확연히 작다. 이는 다시 **외생 정보의 한계효용이 낮음**을 뒷받침한다. 또한 pred_len=720 곡선이 720 look-back에서도 여전히 하강 중이므로 **포화점에 도달하지 않았다** — 더 긴 컨텍스트에서 추가 이득 가능성이 남아 있다. 다만 이 실험은 데이터셋이 명시되지 않아(ECL 추정) 재현이 어렵다.

### ③ Figure 4 (p.9) — 대규모 확장성
**저자 보고**: 3,850개 관측소 기온(시간별, 2019–2020, NCEI) + ERA5 3×3 격자 4변수 = 36 외생(3시간 간격). 7일→3일 예측. MSE: TimeXer **0.200** < iTrans 0.207 < PatchTST 0.208 < DLinear 0.212 < RLinear 0.216.

**제 해석**: 논문에서 **가장 실무적으로 설득력 있는 실험**이다. 공간적으로 인접한 격자를 외생변수로 쓰는 것은 기상 도메인에서 물리적으로 정당한 설계다. 그러나 (i) 전체 성능 스프레드가 0.200–0.216(8%)로 **모든 모델이 사실상 비슷**하고, (ii) baseline은 보간된 시간별 데이터를 받았다고 명시되어 있어 **TimeXer의 입력 조건이 동일했는지 불명확**하며, (iii) 오차막대·표가 없다. "확장성 검증"보다는 "확장성 예비 증거"로 읽는 것이 적절하다.

### ④ Figure 5 (p.10) — Attention 해석 + 효율성
**저자 보고**: (Left) Weather에서 CO2 농도 예측 시 Air Density(변수 10)에 높은 attention, Maximum Wind Velocity(변수 12)에 낮은 attention → 물리적으로 타당. (Right) ECL 320 외생 환경에서 TimeXer가 iTransformer 대비 메모리 우위.

**제 해석**: 논문에서 **강점과 약점이 동시에 가장 선명한 그림**이다.
- 우측(효율성)은 강력하다. iTransformer는 모든 변수 토큰 간 self-attention이라 $O(C^2)$, TimeXer는 외생 토큰을 첫 레이어에서 한 번 만들고 재사용하므로 $O(C)$. 이 격차는 $C$가 커질수록 커지며 이론(Appendix E)과 실측이 일치한다.
- 좌측(해석가능성)은 약하다. 저자 본인이 "형상이 유사한 계열이 높은 attention을 받는다"고 인정하는데, 이는 **attention이 상관/유사도 검출기이지 인과 검출기가 아님**을 의미한다. 단일 사례에 물리적 해석을 붙인 것은 확증편향의 위험이 있다.
- 추출된 텍스트에서 각 모델의 (메모리, 시간) 라벨 대응이 모호하므로, 개별 수치를 인용할 때는 원 PDF 확인이 필요하다.

### ⑤ Figure 9 (p.18) — CKA 표현 분석
**저자 보고**: 첫 블록과 마지막 블록 출력 간 CKA 유사도 vs MSE 산점도. TimeXer가 높은 CKA와 낮은 MSE를 동시 달성. iTransformer는 전체 변수(iTrm-All) CKA는 높으나 **내생변수만(iTrm-En) 보면 표현이 잘 학습되지 않음** → 다변량 모델을 외생 설정에 그대로 쓰면 불필요한 노이즈가 유입된다는 증거.

**제 해석**: **논문의 핵심 가설(변수 역할 구분이 필요하다)을 가장 직접적으로 지지하는 증거**이며, Table 1의 개념적 주장을 경험적으로 뒷받침한다. 그러나 방법론적으로는 취약하다: (i) "높은 CKA = 좋은 성능"이라는 전제는 [36, 9]에서 빌려온 것이지 여기서 검증되지 않았고, (ii) 5개 데이터셋 × 5개 모델 = 25점으로 상관 주장을 하기엔 표본이 적으며, (iii) 실제로 산점도를 보면 **모델 간 CKA 순서와 MSE 순서가 항상 일치하지 않는다**(예: BE에서 iTrm-All은 CKA 0.65 부근인데 MSE는 PatchTST와 유사). **상관을 인과로 읽지 않도록 주의**해야 한다.

> (참고로 Figure 7(마스킹)과 Figure 11(미래 외생값 활용)도 중요하지만, 전자는 수치 표가 없고 후자는 정성적이어서 상위 5개에서 제외했습니다.)

---

## 8. 결론

### 8-1. 저자가 제시한 시사점과 후속 연구 계획

**시사점 (5절, p.10)**
1. Transformer는 **구조 변경 없이도** 외생변수를 흡수할 수 있다 — 필요한 것은 새 모듈이 아니라 **임베딩 전략**이다.
2. "per-patch-per-variate" 어텐션 재해석으로 시간 의존성과 변수 간 상관을 동시 포착 가능.
3. 외생변수 예측 패러다임은 다변량 예측을 **포함하는 상위 패러다임**으로 볼 수 있다 (channel independence 적용 시).
4. 결측·시점 불일치·이질성 등 실무 문제에 대응 가능성 확인.

**저자가 명시한 후속 계획 (산발적으로 언급)**
- Appendix G: patch 길이 증대 또는 **learnable token 수 증대**로 temporal/global 토큰 불균형 해소.
- Appendix E: **선형 어텐션** 도입으로 복잡도를 $O\!\left(\tfrac{L}{P} + C\right)$로 감축.
- Appendix B.3: 외생 토큰 간 상호작용의 **조건부 활용**(Traffic에서는 유효, ETT에서는 무효)에 대한 추가 탐구.

### 8-2. 모델의 일반화 성능 향상 가능성 (중점)

이 논문이 남긴 **일반화 관련 증거를 층위별로 정리**하면:

| 일반화 축 | 논문의 증거 | 강도 | 향상 여지 |
|---|---|---|---|
| **패러다임 일반화** (외생→다변량) | Table 3, 12에서 7개 중 6개 SOTA | 강 | Traffic 실패 원인(토큰 불균형) 해결 시 완전 일반화 가능 |
| **스케일 일반화** | Figure 4 (3,850 station, 36 exog) | 중 | 스케일링 법칙(파라미터 vs 데이터) 미측정 |
| **도메인 일반화** | 전력·기상·교통·전력소비 | 중 | 금융·의료·소매 등 미검증 |
| **입력 이질성 일반화** | Table 5, Figure 3, Figure 7 | 중~약 | 주기·시점 불일치 전용 실험 부재 |
| **전이/제로샷 일반화** | **없음** | — | 가장 큰 미개척지 |

**제가 제안하는 구체적 일반화 향상 경로 (7가지)**

**(1) Multi-global-token으로 병목 해소**
현재 $\mathbf{G}\_{\text{en}} \in \mathbb{R}^{1\times D}$ 단일 토큰이 $C$개 외생변수 정보 전체의 유일한 통로다. 이를 $K$개로 확장하면:

$$\mathbf{G}_{\text{en}} \in \mathbb{R}^{K\times D}, \quad \widehat{\mathbf{G}}^l_{\text{en}} = \mathrm{LayerNorm}\left(\widehat{\mathbf{G}}^l_{\text{en}} + \mathrm{Cross\text{-}Attention}\left(\widehat{\mathbf{G}}^l_{\text{en}}, \mathbf{V}_{\text{ex}}\right)\right)$$

여기서 어텐션 맵은 $K\times C$가 되어 **외생 정보를 여러 부공간으로 분해**할 수 있다. 복잡도는 $O(KC)$로 $K \ll C$ 이면 여전히 선형. 저자가 Appendix G에서 암시했으나 미실험한 방향이며, Traffic의 spike 예측 실패 해결의 1순위 후보다.

**(2) 시차 인식 외생 임베딩**
현재 `VariateEmbed`는 계열 전체를 단일 선형 사상으로 압축하므로 **"어느 시점의 외생 정보가 중요한가"** 정보가 소실된다. 논문이 주장하는 "systematic time lag 추론"을 실제로 구현하려면, 예컨대 lag-bank 형태:

$$\mathbf{V}_{\text{ex},i} = \sum_{k=1}^{K} \alpha_{i,k}\,\mathrm{Proj}_k\!\left(z^{(i)}_{t-\tau_k : t}\right)$$

처럼 복수 시차 윈도우를 학습 가중치 $\alpha_{i,k}$로 결합하는 설계가 자연스럽다. 이는 논문의 $O(C)$ 이점을 유지하면서 인과적 시차 표현력을 얻는다.

**(3) 미래 알려진 외생값(known future covariates)의 1급 시민화**
Figure 11(f)가 정성적으로 최선임을 보였으나 정량화하지 않았다. 실무(전력·수요예측)에서 day-ahead 예보는 **항상 사용 가능**하므로, 미래 외생 토큰 $\mathbf{V}^{\text{fut}}_{\text{ex}}$를 추가해

$$\widehat{\mathbf{G}}^l_{\text{en}} = \mathrm{LayerNorm}\!\left(\widehat{\mathbf{G}}^l_{\text{en}} + \mathrm{Cross\text{-}Attention}\!\left(\widehat{\mathbf{G}}^l_{\text{en}}, \left[\mathbf{V}_{\text{ex}}, \mathbf{V}^{\text{fut}}_{\text{ex}}\right]\right)\right)$$

로 확장하는 것은 구조 변경 없이 즉시 가능하다. **가장 낮은 비용으로 가장 큰 실무 이득**이 기대되는 방향.

**(4) 연속시간 임베딩으로 진짜 주기 불일치 대응**
고정 차원 선형 사상 $\mathbb{R}^{T_{ex}}\to\mathbb{R}^D$ 를 **timestamp 조건부 연속 인코더**(예: 시간 좌표를 입력으로 받는 attention pooling)로 교체하면, 보간 없이 3시간/1시간/불규칙 샘플링을 동시에 처리할 수 있다. 이는 Figure 1이 약속했으나 실험이 이행하지 못한 부분을 메운다.

**(5) 외생 토큰 상호작용의 게이팅**
Table 8은 외생 토큰 간 self-attention이 Traffic에는 유익, ETT에는 유해함을 보였다. 데이터셋별 이분법 대신, 학습 가능한 게이트 $g \in [0,1]$로

$$\mathbf{V}'_{\text{ex}} = (1-g)\,\mathbf{V}_{\text{ex}} + g\cdot\mathrm{Self\text{-}Attention}(\mathbf{V}_{\text{ex}})$$

를 두면 **데이터가 스스로 결정**하게 할 수 있다. 이는 논문이 남긴 가장 명백한 미봉착 지점이다.

**(6) 분포 이동에 대한 명시적 처리**
논문에 정규화 전략 서술이 없다. 비정상 시계열(전력가격은 대표적 체제 전환 데이터)에서 RevIN류 인스턴스 정규화의 유무는 성능 차이가 크므로, **ablation과 명세가 반드시 필요**하다.

**(7) 사전학습·전이 학습으로의 확장**
현재는 데이터셋별 from-scratch 학습이다. Global token은 구조적으로 [CLS]와 동형이므로 **마스킹 기반 사전학습의 자연스러운 앵커**가 될 수 있다. 다만 외생변수의 수·의미가 데이터셋마다 다르므로, 변수 개수에 불변인(permutation-invariant, cardinality-agnostic) 외생 인코더 설계가 선결 과제다.

### 8-2 (계속). 2020년 이후 관련 연구 비교 분석

> ⚠️ **정확도 고지**: 아래 비교는 논문 본문의 인용 정보와 제 사전 지식(2026년 5월 기준)에 근거합니다. 웹 검색을 사용할 수 없어 **최신 수치나 2025~2026년 발표 연구는 검증하지 못했습니다.** 확신도가 낮은 항목은 명시했습니다.

#### (i) 계보 비교표

| 연도 | 모델 | 표현 단위 | 외생변수 처리 | TimeXer와의 관계 |
|---|---|---|---|---|
| 2021 | **Autoformer** (Wu et al., NeurIPS 2021) [37] | point | 없음 | 분해+Auto-correlation. TimeXer의 baseline |
| 2021 | **TFT** (Lim et al., IJF 2021) [20] | point | **있음** (variable selection network, static/known/observed 구분) | TimeXer가 극복 대상으로 지목. Table 9에서 EPF 0.402로 열세 |
| 2022 | **Crossformer** (Zhang & Yan, ICLR 2023) [43] | patch × variate | 전 변수 동등 | TimeXer가 "모든 변수를 세밀하게 모델링하면 노이즈 유입" 논거로 반박 (p.7) |
| 2022 | **PatchTST** (Nie et al., ICLR 2023) [28] | patch | 없음 (channel independence) | TimeXer 내생 경로의 직접 조상 |
| 2023 | **DLinear** (Zeng et al., AAAI 2023) [41] | 선형 | 없음 | "Transformer 무용론" 대표. EPF에서 0.366으로 열세 |
| 2023 | **TiDE** (Das et al., TMLR) [5] | MLP encoder-decoder | **있음** (covariate projection) | 외생 전용 최신 baseline이나 EPF 0.412로 부진 |
| 2023 | **NBEATSx** (Olivares et al., IJF 2023) [29] | basis expansion | **있음** | EPF 프로토콜 원저. 0.330으로 2위권 |
| 2023 | **TimesNet** (Wu et al., ICLR 2023) [36] | 2D 변환 | 없음 | 다변량 baseline |
| 2023 | **iTransformer** (Liu et al., ICLR 2024) [23] | variate | 전 변수 동등 | TimeXer의 **가장 직접적 경쟁자**. Traffic에서 유일하게 승리 |
| 2024 | **TimeXer** (본 논문) | patch(내생) + variate(외생) | **있음, 비대칭 설계** | — |

#### (ii) TimeXer의 위치 규정

논문의 진짜 기여는 **"channel independence(PatchTST) ↔ channel dependence(iTransformer)"의 이분법을 깨고 제3의 축을 도입한 것**이다:

- PatchTST: 시간 표현 ○ / 변수 상관 ✗
- iTransformer: 시간 표현 ✗ / 변수 상관 ○
- Crossformer: 둘 다 ○, 그러나 $O(C \cdot N)$ 토큰으로 노이즈·비용 폭증
- **TimeXer: 둘 다 ○, 그러나 변수 역할에 따라 해상도를 차등화** → $O(N^2 + C)$

Figure 9(CKA)와 Appendix E(복잡도)가 각각 이 위치 설정의 **표현론적 근거**와 **계산론적 근거**를 제공한다.

#### (iii) 2024년 이후 흐름과의 접점 (사전 지식 기반, 검증 미수행)

제가 아는 범위에서 관련성이 높은 후속·병렬 흐름:

1. **시계열 기반모델(Foundation Models)**: Chronos (Ansari et al., 2024) [2], TimesFM (Das et al., ICML 2024) [6], Timer (Liu et al., ICML 2024) [25], Moirai (Woo et al., ICML 2024) 등. 이 중 **Moirai의 "any-variate attention"** 은 임의 개수의 공변량을 다루는 메커니즘을 제안한 것으로 알고 있으며, TimeXer의 variate token과 문제의식이 겹칩니다. 다만 세부 비교는 제가 검증하지 못했습니다.
2. **경량 MLP 계열**: TSMixer (Google, 2023), TimeMixer (ICLR 2024) 등이 Transformer 대비 경쟁력 있는 성능을 보고 — TimeXer의 "Transformer 유지" 전략이 계속 정당한지는 지속 검증이 필요합니다.
3. **LLM 기반 예측에 대한 회의론**: "Are Language Models Actually Useful for Time Series Forecasting?" (Tan et al., NeurIPS 2024)는 LLM 기반 예측기의 이득이 제한적임을 보고했습니다. 이는 TimeXer처럼 **도메인 특화 귀납 편향을 설계하는 접근의 상대적 가치를 높이는** 흐름입니다.
4. **TimeXer 자체의 후속 연구**: 저자 그룹(THUML)이 후속 작업을 냈는지는 **확인할 수 없었습니다.** 궁금하시면 웹 검색을 켜주시면 최신 인용 현황과 후속 논문을 조사해드릴 수 있습니다.

#### (iv) 향후 연구 시 고려할 점

| # | 고려사항 | 이유 |
|---|---|---|
| 1 | **평가 프로토콜을 도메인 표준에 맞출 것** | EPF는 Lago et al. (2021)이 rMAE + DM 검정을 권고. MSE/MAE만으로는 선행 연구와 대화 불가 |
| 2 | **최소 5 seed × 오차막대 × 유의성 검정 필수** | 본 논문의 장기 예측 격차(1~4%)는 검정 없이 판단 불가 |
| 3 | **baseline 튜닝 예산을 명시적으로 동일화** | S4 참조. 이것이 최근 시계열 예측 문헌의 재현성 위기의 핵심 |
| 4 | **"exogenous 기여도"를 별도 지표로 보고** | Table 5는 외생 기여가 <10%임을 시사. 외생 모델을 주장하려면 **외생 유무 델타**를 주 지표로 삼아야 함 |
| 5 | **look-back 96 고정 관행 재검토** | Figure 3은 720에서도 포화하지 않음을 보임. 짧은 컨텍스트는 Transformer에 불리한 인위적 제약 |
| 6 | **해석가능성 주장에 정량 지표 부여** | attention 시각화 단독으로는 불충분. 알려진 인과 구조와의 정합성 측정 필요 |
| 7 | **표–본문 정합성 자동 검증** | 본 논문의 M1~M6 같은 불일치는 리뷰 단계에서 걸러져야 하는 유형 |
| 8 | **비대칭 표현 원리의 타 도메인 이식 검토** | "역할이 다른 입력에는 다른 해상도를 준다"는 원리는 멀티모달·추천·센서 융합에도 적용 가능 |

### 최종 평가

TimeXer는 **아키텍처 혁신의 논문이 아니라 설계 원리의 논문**이다. 새 모듈은 사실상 learnable global token 하나뿐이며, 나머지는 표준 Transformer다. 그럼에도 가치가 있는 이유는 "**모든 입력을 동등하게 취급할 이유가 없다**"는 원리를 명확히 정식화하고, EPF에서 통계적으로도 상당히 견고한 우위(2위 대비 7%, 자체 표준편차의 5σ 이상)로 입증했기 때문이다.

반면 장기 다변량 예측에서의 우위는 **오차막대 부재와 표 간 수치 불일치로 인해 신뢰 구간이 넓다**. 그리고 논문의 가장 큰 아이러니는 — 외생변수 활용을 표방하면서, 자체 ablation(Table 5)이 외생 정보의 기여가 10% 미만임을 보여준다는 점이다. 이 긴장을 해소하는 것이 후속 연구의 가장 중요한 과제라고 봅니다.

---

## 참고자료 출처

**주 분석 대상 (업로드 문서)**
- Yuxuan Wang, Haixu Wu, Jiaxiang Dong, Guo Qin, Haoran Zhang, Yong Liu, Yunzhong Qiu, Jianmin Wang, Mingsheng Long. **"TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables."** NeurIPS 2024. arXiv:2402.19072v4 (2024-11-11). 코드: `https://github.com/thuml/TimeXer`

**논문 내에서 인용된 문헌 중 본 분석에 직접 사용한 것 (해당 논문의 참고문헌 번호 기준)**
- [5] Das et al., *Long-term Forecasting with TiDE: Time-series Dense Encoder*, arXiv:2304.08424
- [10] Dosovitskiy et al., *An Image is Worth 16x16 Words (ViT)*, arXiv:2010.11929
- [11] Hersbach et al., *The ERA5 Global Reanalysis*, QJRMS, 2020
- [14] Kornblith et al., *Similarity of Neural Network Representations Revisited (CKA)*, ICML 2019
- [15] Lago, Marcjasz, De Schutter, Weron, *Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark*, Applied Energy 293:116983, 2021 — **EPF 벤치마크 원저**
- [20] Lim, Arık, Loeff, Pfister, *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting*, IJF 2021
- [23] Liu et al., *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting*, arXiv:2310.06625 (ICLR 2024)
- [28] Nie et al., *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers (PatchTST)*, arXiv:2211.14730 (ICLR 2023)
- [29] Olivares et al., *Neural Basis Expansion Analysis with Exogenous Variables: Forecasting Electricity Prices with NBEATSx*, IJF 2023
- [32] Vaswani et al., *Attention Is All You Need*, NeurIPS 2017
- [36] Wu et al., *TimesNet*, ICLR 2023
- [37] Wu et al., *Autoformer*, NeurIPS 2021
- [41] Zeng et al., *Are Transformers Effective for Time Series Forecasting? (DLinear)*, AAAI 2023
- [43] Zhang & Yan, *Crossformer*, ICLR 2023

**8-2절 최신 동향 언급 시 참조한 외부 문헌 (제 사전 지식 기반, 이번 답변에서 웹으로 검증하지 않음)**
- Woo et al., *Unified Training of Universal Time Series Forecasting Transformers (Moirai)*, ICML 2024
- Chen et al., *TSMixer: An All-MLP Architecture for Time Series Forecasting*, 2023
- Wang et al., *TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting*, ICLR 2024
- Tan et al., *Are Language Models Actually Useful for Time Series Forecasting?*, NeurIPS 2024

**정확도 관련 고지**: 5절의 수치 불일치(M1~M10)는 업로드된 PDF의 추출 텍스트를 대조해 확인한 것이며, PDF 원본의 렌더링 상태에 따라 일부는 텍스트 추출 오류일 가능성도 있습니다. Figure 5의 개별 (메모리, 시간) 라벨 대응처럼 추출 텍스트만으로 확정할 수 없는 항목은 본문에 그 사실을 표시했습니다. 8-2절의 2024년 이후 동향은 검색 없이 작성되어 최신성이 보장되지 않았습니다.
