# Toto 2.0: Time Series Forecasting Enters the Scaling Era 

---

## 1. Executive Summary (10문장 이내)

Toto 2.0은 Datadog AI Research가 공개한 시계열 예측 파운데이션 모델 패밀리로, 4m~2.5B 파라미터에 걸쳐 **신뢰할 수 있는 스케일링**을 최초로 달성했다.  
핵심 질문인 "TSFM(Time Series Foundation Model)도 스케일링될 수 있는가?"에 대해 긍정적 답변을 제시한다.  
모델은 BOOM, GIFT-Eval, TIME 세 벤치마크 모두에서 새로운 SOTA를 달성했다.  
아키텍처상 가장 중요한 변화는 자기회귀 디코딩을 대체하는 **Contiguous Patch Masking(CPM)**으로, 단일 순방향 패스로 전체 예측 구간을 생성한다.  
출력 헤드는 Student-T 혼합 모델에서 **분위수(Quantile) 헤드**로 교체되어 대규모 학습의 수치 안정성을 확보했다.  
옵티마이저는 핀볼 손실에 최적화된 **NorMuon**을 채택했다.  
학습 데이터는 Datadog 내부 관측 메트릭과 합성 데이터만으로 구성되며, 공개 시계열 데이터는 사전학습에서 완전히 배제되었음에도 범용 벤치마크에서 1위를 달성했다.  
하이퍼파라미터는 10m 프록시 모델에서 탐색 후 **u-µP**를 통해 5개 목표 크기로 무조정 이전(zero-shot transfer)된다.  
22m 모델은 Toto 1.0과 동등한 성능을 7배 적은 파라미터로 달성하며, 추론 속도 역시 Toto 1.0 대비 전 사이즈에서 현격히 빠르다.  
저자들은 이를 TSFM 분야의 "GPT-2 모멘트"에 비유하며, 스케일링이 더 이상 연구 질문이 아닌 도구가 되었음을 선언한다.

### 1-1. 연구의 목적과 필요성

| 구분 | 내용 |
|------|------|
| **배경 문제** | NLP·비전 분야에서 스케일링 법칙은 확립되었으나, TSFM에서는 모델을 단순히 크게 만들어도 성능이 일관되게 향상되지 않았음 (경쟁 모델들: 큰 버전이 작은 버전보다 떨어지는 경우 발생) |
| **실용적 필요성** | Datadog 같은 관측 가능성(Observability) 플랫폼에서 CPU, 메모리, 레이턴시 등 수많은 시계열을 실시간 예측해야 하며, 모델 크기-비용-성능의 예측 가능한 트레이드오프가 필요 |
| **학문적 필요성** | TSFM이 "BERT 모멘트"(범용 성능 달성)에 도달했다는 평가(BERT²S Workshop, 2025)가 나오는 가운데, 다음 단계인 신뢰할 수 있는 스케일링 레시피 확립이 미해결 과제로 남아 있었음 |
| **오염 문제** | 기존 벤치마크(ETTh1 등)는 대부분의 TSFM 사전학습 데이터에 포함되어 공정한 평가가 어려우며, 오염(contamination)에 강건한 평가 방법 필요 |

> **💡 용어 설명**
> - **TSFM (Time Series Foundation Model)**: 다양한 도메인의 시계열 데이터에 대해 파인튜닝 없이 또는 최소한의 조정으로 예측을 수행할 수 있도록 대규모 데이터로 사전학습된 범용 모델
> - **스케일링 법칙 (Scaling Law)**: 모델 크기, 데이터 양, 연산량이 증가할수록 성능이 예측 가능하게 향상된다는 법칙 (Kaplan et al., 2020)
> - **관측 가능성 (Observability)**: 시스템 내부 상태를 외부 출력(메트릭, 로그, 트레이스 등)으로 추론할 수 있는 능력

---

## 2. 핵심 주장과 근거 표

| 핵심 주장 | 근거 | 위치 |
|-----------|------|-------|
| TSFM은 스케일링된다 | 4m→2.5B까지 모든 크기가 직전 크기보다 BOOM·GIFT-Eval에서 향상 | Figure 1, Section 5 |
| CPM이 자기회귀 디코딩보다 빠르고 정확하다 | 1,024스텝 예측에서 Toto 1.0 대비 극적인 레이턴시 감소 | Figure 9, Section 5.5 |
| 공개 데이터 없이도 범용 벤치마크 1위 달성 가능 | GIFT-Eval(23개 도메인) CRPS rank 1위, TIME rank 1위 | Figure 6, 8, Section 5.2, 5.4 |
| u-µP로 하이퍼파라미터 재탐색 없이 크기 이전 가능 | 10m 프록시에서 탐색된 설정이 2.5B까지 그대로 적용 | Section 4, Figure 4 |
| NorMuon이 핀볼 손실과 더 잘 맞는다 | 핀볼 기울기가 부호 값만 가져 AdamW의 분산 메커니즘이 제한적; NorMuon이 per-neuron 균형화로 이를 보완 | Section 2.3, Eq. (4),(5) |
| 큰 모델이 학습 컨텍스트를 넘어서도 장기 안정적이다 | 2.5B: 8,192스텝 지평선에서 r=0.818; Chronos-2는 r=0.310 | Figure 10, Section 5.6 |
| 합성 데이터가 공개 데이터보다 사전학습에 효과적 | 하이퍼파라미터 탐색에서 공개 데이터를 0으로 설정하는 것이 최적 | Section 3, 4.2 |

---

## 2-1. 상세 설명

### ① 해결하고자 하는 문제

1. **TSFM 스케일링 불안정**: 기존 모델 패밀리(Chronos, Moirai, TimesFM 등)는 더 큰 버전이 더 작은 버전보다 나쁠 때가 있음 (Figure 1 참조)
2. **자기회귀 디코딩의 비효율**: $H$-스텝 예측에 $K = H/P$ 번의 순차적 순방향 패스 필요 → 오류 누적 및 느린 추론
3. **SMM(Student-T Mixture)의 수치 불안정**: 대규모 활성화 값에서 발산하는 문제
4. **하이퍼파라미터 이전 불가**: 표준 파라미터화에서 최적 학습률이 모델 폭에 따라 최대 10배 차이 발생
5. **도메인 오염**: 공개 벤치마크 데이터가 많은 모델의 사전학습에 포함되어 공정 평가 어려움

---

### ② 제안하는 방법 및 핵심 수식

#### A. Contiguous Patch Masking (CPM) — Eq. (1)

$$\hat{\mathbf{p}}_i = \left[ f_\theta(\mathbf{p}_{1:N}, \mathbf{b}_{1:N}) \right]_i, \quad i \in \mathcal{M} $$

| 기호 | 설명 |
|------|------|
| $\hat{\mathbf{p}}_i$ | $i$번째 패치에 대한 예측값 |
| $f_\theta$ | 파라미터 $\theta$를 가진 Toto 2.0 트랜스포머 모델 |
| $\mathbf{p}_{1:N}$ | 컨텍스트 패치 시퀀스 ($N$개 패치, 각 패치 크기 $P=32$) |
| $\mathbf{b}_{1:N}$ | 이진 마스크 채널; $b_{i,k}=1$이면 미관측, $0$이면 관측 |
| $\mathcal{M}$ | 마스킹된 패치 인덱스 집합; 추론 시 $\mathcal{M} = \{N+1, \ldots, N+K\}$ |

> **💡 용어 설명**
> - **패치 (Patch)**: 연속된 타임스텝들을 하나의 토큰으로 묶은 단위. 여기서는 크기 $P=32$
> - **자기회귀 디코딩 (Autoregressive Decoding)**: 이전 출력을 다음 입력으로 사용하여 한 번에 하나씩 순차적으로 예측하는 방식

---

#### B. Quantile (Pinball) Loss — Eq. (2), (3)

$$\rho_\tau(y - \hat{q}_\tau) = (y - \hat{q}_\tau)\left(\tau - \mathbf{1}[y < \hat{q}_\tau]\right) $$

$$\mathcal{L}_{\text{quantile}} = \frac{1}{|\mathcal{T}|} \sum_{\tau \in \mathcal{T}} \rho_\tau(y - \hat{q}_\tau) $$

| 기호 | 설명 |
|------|------|
| $y$ | 실제 관측값 (target) |
| $\hat{q}_\tau$ | 분위수 수준 $\tau$에서의 예측 분위수 값 |
| $\tau$ | 분위수 수준; $\mathcal{T} = \{0.1, 0.2, \ldots, 0.9\}$ (9개) |
| $\mathbf{1}[y < \hat{q}_\tau]$ | 지시 함수; $y < \hat{q}_\tau$이면 1, 아니면 0 |
| $\mathcal{L}_{\text{quantile}}$ | 9개 분위수에 대한 평균 핀볼 손실 |

> **💡 용어 설명**
> - **핀볼 손실 (Pinball Loss)**: 분위수 회귀에서 사용되는 비대칭 손실 함수. 예측이 실제보다 낮을 때와 높을 때 다른 패널티를 부여하여 특정 분위수에 수렴하도록 학습
> - **CRPS (Continuous Ranked Probability Score)**: 예측 분포와 실제 관측값의 차이를 측정하는 확률적 예측 평가 지표. 낮을수록 좋음

---

#### C. Pinball Gradient (AdamW와의 비교 근거) — Eq. (4)

```math
\frac{\partial \rho_\tau(y - \hat{q})}{\partial \hat{q}} = g_\tau = \begin{cases} -\tau & y > \hat{q} \\ 0 & y = \hat{q} \\ 1 - \tau & y < \hat{q} \end{cases}
```

핀볼 기울기는 오직 3개의 값만 가지며(부호값 기울기), MSE 기울기 $\frac{\partial(y-\hat{q})^2}{\partial \hat{q}} = -2(y-\hat{q})$와 달리 오차 크기 정보를 전달하지 않음 → AdamW의 분산 기반 스텝 사이즈 조정이 제한적으로 작동

---

#### D. NorMuon 업데이트 규칙 — Eq. (5)

```math
v_t = \beta_2 v_{t-1} + (1 - \beta_2) \cdot \text{mean\_cols}(O_t \odot O_t)
```

$$W_t \leftarrow W_{t-1} - \eta \, O_t \Big/ \sqrt{v_t + \epsilon} $$

| 기호 | 설명 |
|------|------|
| $v_t$ | 시간 $t$에서의 per-row EMA 분산 추정값 |
| $\beta_2$ | 분산 EMA의 지수 감쇠율 (최적값: $0.999$) |
| $O_t$ | Newton–Schulz 반복(또는 Polar Express)으로 직교화된 행렬 |
| $\odot$ | 아다마르(원소별) 곱 |
| $\text{mean cols}(\cdot)$ | 각 행을 열 평균으로 축소하여 행별 스칼라 생성 |
| $\eta$ | 학습률 (NorMuon 최적값: $0.652$) |
| $\epsilon$ | 수치 안정화 소항 |
| $W_t$ | 시간 $t$에서의 가중치 행렬 |

> **💡 용어 설명**
> - **Muon**: Jordan et al. (2024)이 제안한 옵티마이저. 모멘텀 버퍼를 Newton–Schulz 반복으로 직교화하여 적용. Adam 대비 약 2배 연산 효율
> - **Newton–Schulz 반복**: 행렬의 특이값을 1로 수렴시키는 반복적 직교화 알고리즘
> - **NorMuon**: Muon에 per-neuron(행별) 정규화를 추가하여 핀볼 손실과 같은 부호값 기울기 환경에서 안정적인 스텝 크기 적응을 가능케 한 변형

---

#### E. Robust Causal Scaler (입력 정규화)

$$z_t = \text{asinh}\!\left(\frac{x_t - \mu_t}{\sigma_t}\right) = \log\!\left(\frac{x_t - \mu_t}{\sigma_t} + \sqrt{\left(\frac{x_t - \mu_t}{\sigma_t}\right)^2 + 1}\right)$$

| 기호 | 설명 |
|------|------|
| $x_t$ | 시간 $t$에서의 원시 입력값 |
| $\mu_t$ | 인과적(causal) 위치 추정값 |
| $\sigma_t$ | 인과적 스케일 추정값 |
| $z_t$ | 정규화된 입력; 모델이 이 공간에서 예측 후 역변환 |

> **💡 용어 설명**
> - **arcsinh 변환**: $|z| \ll 1$이면 항등 변환(선형 동작), $|z| \gg 1$이면 $\text{sign}(z)\log(2|z|)$(로그 압축)으로 동작하여, 부호 정보 손실 없이 극단적 스케일 변화를 처리

---

#### F. u-µP 가중치 파라미터화 (Section 4.3)

$$W = A_W \cdot w, \quad w_0 \sim \mathcal{N}(0,1)$$

$$w_{t+1} = w_t + C_W \cdot \Phi_t$$

```math
A_W \propto \frac{1}{\sqrt{\text{fan\_in}}}, \quad C_W \propto \frac{\eta}{\sqrt{\text{fan\_in}}}
```

| 기호 | 설명 |
|------|------|
| $A_W$ | 초기화 스케일 인수 |
| $C_W$ | 업데이트 스케일 인수 |
| $\text{fan in}$ | 해당 레이어의 입력 차원 수 |
| $\Phi_t$ | 옵티마이저의 스텝 방향 |
| $\eta$ | 학습률; $A_W$, $C_W$의 스케일링으로 모델 폭에 무관하게 최적값 고정 |

> **💡 용어 설명**
> - **µP (Maximal Update Parametrization)**: Yang et al. (2021)이 제안. 모델 폭에 관계없이 최적 학습률이 동일하도록 각 레이어를 재파라미터화하는 방법
> - **u-µP**: µP와 단위 스케일링(unit scaling)을 결합하여 디코더 전용 모델에서 이전 안정성을 높인 변형 (Blake et al., 2025)
> - **fan_in**: 뉴런의 입력 연결 수. 가중치 초기화 스케일 결정에 사용

---

### ③ 모델 구조 (Table 1, Figure 2)

```
입력 → Robust Causal Scaler (arcsinh 정규화)
     → Patch Embedding (크기 32) + CPM 마스크 채널
     → Input Residual MLP (2-layer SiLU + residual)
     → Variate-Time Transformer Decoder
        ├─ Time-axis: Causal Attention (시간축, 인과적)
        └─ Variate-axis: Full Attention (변수축, 전체)
        (PerDimScale 적용, 1/d_k 스케일링)
     → Output Residual MLP
     → Quantile Output Head (9개 분위수: 0.1~0.9)
     → 출력 역정규화 → 최종 예측
```

| 모델 | $d_\text{model}$ | $h$ (헤드수) | $L$ (레이어수) | 파라미터 |
|------|-----------|---------|------------|--------|
| 4m | 256 | 4 | 4 | ~4M |
| 22m | 512 | 8 | 6 | ~22M |
| 313m | 1024 | 16 | 24 | ~313M |
| 1B | 1536 | 24 | 36 | ~1B |
| 2.5B | 2048 | 32 | 48 | ~2.5B |

> 모든 사이즈: 헤드 차원 $d_\text{head}=64$, 컨텍스트 4,096 타임스텝, 패치 크기 32, 배치 크기 64

> **💡 용어 설명**
> - **Decoder-only Transformer**: GPT 계열처럼 이전 토큰만 참조하는(causal) 단방향 트랜스포머
> - **Variate-axis Attention**: 다변량 시계열에서 시간 차원이 아닌 변수(채널) 차원 간의 관계를 학습하는 어텐션
> - **SiLU**: Sigmoid Linear Unit. $\text{SiLU}(x) = x \cdot \sigma(x)$ 형태의 활성화 함수

---

### ④ 성능 향상 및 한계

#### 성능 향상

| 벤치마크 | 1위 모델 | CRPS Rank | 비교 2위 | 개선 폭 |
|----------|---------|-----------|---------|--------|
| BOOM | Toto 2.0 2.5B | 3.88 | Toto 2.0 1B (3.96) | 내부 경쟁 |
| GIFT-Eval (FM) | Toto 2.0 2.5B | 20.3 | Toto 2.0 1B (21.1) | PatchTST-FM r1(23.1) 대비 −2.8p |
| TIME | Toto 2.0 2.5B | 3.43 | Toto 2.0 313m (3.86) | Chronos-2(4.03) 대비 −0.6p |
| 추론 속도 | Toto 2.0 전 사이즈 | — | Toto 1.0 대비 극적 감소 | 1,024스텝: 1회 forward pass |

- 22m이 Toto 1.0(151m)을 **7배 적은 파라미터**로 능가
- 4m이 Toto 1.0과 Chronos-2에 근접한 성능을 **~38배 적은 파라미터**로 달성

#### 한계

| 한계 | 내용 |
|------|------|
| TIME 내 비단조성 | 313m이 MASE와 일부 rank 지표에서 1B을 앞섬 (Section 5.4) |
| 극장기 예측 | 2.5B도 8,192스텝에서 Pearson $r=0.818$로 하락; 적절히 피팅된 계절 모델에 뒤짐 |
| 고빈도 데이터 의존 | 학습 데이터가 Datadog 관측 메트릭 위주 — 특정 도메인 편향 가능성 |
| 공개 데이터 배제 | 사전학습에 공개 데이터 전혀 없음 — 일반화 근거가 합성 데이터의 품질에 의존 |
| 데이터 큐레이션 미성숙 | 최적 믹스가 직관적이지 않으며 원칙적 선택이 아닌 경험적 탐색으로 도달 |

---

## 3. 주장별 페이지/Figure 번호

| 주장 | 위치 |
|------|------|
| TSFM 스케일링 성공 주장 | p.1 Abstract, Figure 1 (p.1), Section 5 (p.9) |
| CPM 구조 및 수식 | Section 2.1 (p.2–3), Figure 2 (p.3), Eq.(1) |
| Quantile head & Pinball loss | Section 2.2 (p.3–4), Eq.(2),(3) |
| NorMuon 선택 근거 | Section 2.3 (p.4–5), Eq.(4),(5) |
| 학습 데이터 구성 | Section 3 (p.5–6), Figure 3 (p.6) |
| u-µP 하이퍼파라미터 이전 | Section 4 (p.6–9), Figure 4 (p.7), Table 1 (p.9) |
| BOOM 결과 | Section 5.1 (p.10), Figure 5 (p.10) |
| GIFT-Eval FM 결과 | Section 5.2 (p.10–11), Figure 6 (p.11) |
| GIFT-Eval FT/앙상블 결과 | Section 5.3 (p.11–12), Figure 7 (p.12) |
| TIME 결과 | Section 5.4 (p.13), Figure 8 (p.13) |
| 추론 레이턴시 | Section 5.5 (p.13–14), Figure 9 (p.14) |
| 장기 안정성 | Section 5.6 (p.14–15), Figure 10 (p.15) |
| 향후 연구 방향 | Section 6 (p.14–16) |

---

## 4. 저자 보고 결과 vs. 해석 분리

### 연구 주제

- **저자 직접 보고**: "We show that time series foundation models scale: a single training recipe produces reliable forecast-quality improvements from 4m to 2.5B parameters." (p.1)
- **내 해석**: 이는 TSFM 분야에서 LLM의 스케일링 법칙과 유사한 현상이 시계열 영역에서도 성립함을 보인 최초의 실증적 증거로, 향후 TSFM 연구 방향을 근본적으로 바꿀 수 있는 패러다임 전환이다.

### 방법

| 항목 | 저자 직접 보고 | 내 해석 |
|------|--------------|--------|
| CPM | "CPM addresses both [slowness and compounding error]" (p.3) | 마스크드 언어 모델링(MLM)을 시계열 패치에 적용한 아이디어로 BERT의 마스킹 전략과 유사하나, 추론 시 단방향으로 작동 |
| NorMuon 선택 | "NorMuon's row normalization reinstates the β₂ variance mechanism—now applied per neuron" (p.5) | 핀볼 손실의 부호값 기울기 문제를 인식하고 해당 특성에 맞는 옵티마이저를 이론적으로 분석하여 선택한 점이 주목할 만함 |
| 공개 데이터 배제 | "Our hyperparameter sweep found that public time series data was suboptimal at proxy model scale" (p.5) | 합성 데이터 품질이 공개 시계열보다 높을 수 있다는 역직관적 발견으로, TempoPFN 기반 다양한 합성 생성의 효과로 해석 |
| u-µP | "To our knowledge, this is the first application of µP to time series forecasting." (p.7) | 기존 LLM에서 검증된 µP를 TSFM에 최초 적용한 것으로, 기술 이전이 성공한 중요한 사례 |

### 결과

| 항목 | 저자 직접 보고 | 내 해석 |
|------|--------------|--------|
| BOOM 1위 | "Toto 2.0 2.5B CRPS rank: 3.88" (Figure 5) | Datadog 자체 데이터로 학습했으므로 이 벤치마크에서의 1위는 다소 유리한 조건. 단, 다른 범용 모델들도 같은 벤치마크에 접근 가능하여 비교 자체는 공정 |
| GIFT-Eval 1위 | "ranks first among foundation models on GIFT-Eval despite training only on synthetic and observability data" (p.10) | ⚠️ 사전학습에 공개 데이터가 없다는 점은 오히려 더 강력한 일반화 증거이지만, 파인튜닝(2.5B-FT)에는 GIFT-Eval train 데이터를 사용했으므로 파인튜닝 결과는 별도 평가 필요 |
| 추론 속도 | "313m runs at roughly the same latency as Chronos-2 (120m parameters)" (p.14) | 313m이 120m Chronos-2와 같은 레이턴시라는 것은 CPM의 병렬화 이점을 파라미터 증가 비용이 상쇄한 것으로, 절대 효율이 아닌 상대 효율로 해석해야 함 |

---

## 5. 통계적 취약점 및 비교 불가 수치 ⚠️

| 항목 | 취약점/비교불가 사유 |
|------|-------------------|
| **BOOM 1위** ⚠️ | Datadog의 내부 관측 메트릭으로 구성된 벤치마크에서 Datadog 메트릭으로 학습한 모델이 1위 — **평가 데이터와 학습 데이터 분포 겹침** 가능성 높음 |
| **Xihe-ultra 파라미터 수** ⚠️ | "~3B; not officially disclosed" (Figure 1 각주) — 추정치로 Figure 1 비교에서 Pareto 경계 판단의 불확실성 존재 |
| **Timer-s1 비교** ⚠️ | 8.3B MoE 모델(750m 활성 파라미터)을 파라미터 축에서 어떻게 표시할지 불명확; 활성 파라미터 기준이면 훨씬 작은 것으로 분류되어야 함 |
| **FnF 앙상블의 공정성** ⚠️ | Toto 2.0 FnF는 10개 외부 모델을 사용한 앙상블 + XGBoost 메타러너 — 단일 모델과의 직접 비교 부적절 |
| **TIME에서 313m > 1B** ⚠️ | 스케일링이 단조롭지 않은 유일한 지점 — 통계적 유의성 검증(예: bootstrap CI) 없이 확언하기 어려움 |
| **장기 안정성 실험** ⚠️ | "illustrative stability test... not extrapolation to genuinely novel dynamics" (p.15) — 합성 정현파 신호에서의 결과이며 실제 복잡한 시계열에 대한 일반화 불명확 |
| **파인튜닝 결과** ⚠️ | 2.5B-FT는 GIFT-Eval train 사용 → 제로샷 스케일링 주장 지지에 사용 불가 (저자도 명시: "not used to support the zero-shot scaling claim", p.11) |
| **CRPS vs. CRPS Rank** | 절대 CRPS 값에서 타 모델과 차이가 미미한 경우도 있음 (예: GIFT-Eval CRPS에서 상위 모델들의 차이가 0.01 이하) — rank 기반 지표가 실질적 차이를 과장할 수 있음 |

---

## 6. 논문이 답하지 않는 질문들

| 카테고리 | 미해결 질문 |
|----------|-----------|
| **스케일링 한계** | 2.5B보다 더 큰 모델(예: 10B, 100B)에서도 스케일링 법칙이 유지되는가? 수렴점이 존재하는가? |
| **데이터 효율** | 합성 데이터와 실제 데이터 중 품질 단위당 어느 것이 더 효과적인가? TempoPFN 데이터의 최적 생성 파라미터는 무엇인가? |
| **아키텍처 선택 근거** | Variate-axis attention을 마지막 레이어에만 배치한 것의 이론적 근거가 부족; ablation만 제시 |
| **도메인 이전** | 의료, 금융 등 Datadog 관측 메트릭과 전혀 다른 도메인에서의 실제 성능 보장 근거 부족 |
| **불확실성 보정** | 9개 분위수의 보정(calibration) 품질에 대한 체계적 분석 없음 |
| **전력/환경 비용** | 2.5B 모델 학습에 소요된 GPU 시간, 에너지 비용 미공개 |
| **CPM vs. 다른 마스킹 전략** | 비연속(non-contiguous) 마스킹, 랜덤 마스킹 등 대안과의 직접 비교 없음 |
| **NorMuon vs. AdamW on Quantile** | 동일 조건에서 NorMuon이 AdamW보다 얼마나 우수한지 정량적 ablation 결과 미제시 |
| **멀티모달 통합** | ARFBench 언급은 있으나 텍스트·로그·토폴로지와의 실제 통합 방법론 제시 없음 |
| **프로덕션 배포** | 실시간 스트리밍 시계열에서의 점진적 컨텍스트 업데이트 방법 불명확 |

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p.1) — CRPS Rank vs. Parameter Count

**무엇을 보여주는가**: BOOM(좌)과 GIFT-Eval(우)에서 파라미터 수에 따른 CRPS rank (낮을수록 좋음)를 경쟁 모델 패밀리와 비교

**핵심 관찰**:
- Toto 2.0만이 4m → 2.5B 전 구간에서 단조 개선 (Pareto 경계 위에 위치)
- 경쟁 모델들(Chronos Bolt, TimesFM-2.0 등): 더 큰 버전이 더 작은 버전보다 나쁜 역전 현상 발생
- BOOM에서 4m이 이미 Chronos Bolt(더 큰 모델)보다 우수

**해석**: 이 그림이 논문 전체의 핵심 주장을 시각화한다. 스케일링의 "신뢰성"이란 단순히 최대 모델이 1위를 차지하는 것이 아니라 모든 크기에서 단조 개선을 보이는 것임을 강조. ⚠️ BOOM에서 Datadog 데이터와 학습 데이터 분포 겹침 가능성 주의 요망.

---

### Figure 2 (p.3) — Toto 2.0 아키텍처 전체도

**무엇을 보여주는가**: 학습/추론 프로토콜(좌), 순방향 패스 구조(중), 입출력 헤드(우)를 통합 도시

**핵심 관찰**:
- 학습: CPM이 가변 길이 연속 마스크 스팬을 입력에 적용
- 추론: 전체 예측 구간을 마스크 토큰으로 채워 단 1회 순방향 패스로 해결
- 출력: 9개 분위수(0.1~0.9) fan-out 구조

**해석**: Toto 1.0의 자기회귀 방식 대비 추론 방식의 근본적 변화. 학습과 추론이 동일한 CPM 메커니즘 안에서 자연스럽게 통합되어, "마스크된 위치를 예측한다"는 하나의 목표로 양쪽을 포괄. 이는 BERT의 MLM을 생성 모델에 적용한 방식과 개념적으로 유사.

---

### Figure 9 (p.14) — Forward Pass Latency

**무엇을 보여주는가**: (좌) 파라미터 수 vs. 레이턴시 at 1,024스텝 예측, (우) 예측 길이 vs. 레이턴시

**핵심 관찰**:
- Toto 1.0: 파라미터 수 증가 시 레이턴시 $10^3$ ms 수준; 예측 길이에 선형 증가
- Toto 2.0 단일 패스: 전 사이즈에서 $10^1 \sim 10^2$ ms; 768스텝까지 **상수** 레이턴시
- Toto 2.0 2.5B 단일 패스: 4,096스텝에서도 Chronos-2보다 빠름

**해석**: CPM의 병렬화 이득이 파라미터 증가 비용을 상쇄함을 명확히 보여줌. 단, 우측 그래프에서 768스텝 이후 블록 디코딩으로 전환 시 레이턴시가 급증하는 점은 장기 예측 실시간 운용의 제약. ⚠️ 벤치마크 하드웨어 사양 미명시.

---

### Figure 10 (p.15) — 장기 안정성 (Multi-Scale Decomposition)

**무엇을 보여주는가**: 주기 500, 100, 20의 합성 다중 스케일 신호에서 2,048 / 4,096 / 8,192스텝 예측의 Pearson 상관계수 $r$

**핵심 관찰**:

| 모델 | 2,048 ($r$) | 4,096 ($r$) | 8,192 ($r$) |
|------|------------|------------|------------|
| 2.5B | 0.990 | 0.979 | 0.818 |
| 1B | 0.984 | 0.945 | 0.643 |
| 313m | 0.986 | 0.947 | 0.681 |
| 22m | 0.805 | 0.484 | 0.315 |
| 4m | 0.538 | 0.371 | 0.173 |
| Toto 1.0 | 0.816 | 0.627 | 0.333 |
| Chronos-2 | 0.663 | 0.457 | 0.310 |

**해석**: 큰 모델이 학습 컨텍스트(4,096스텝)를 두 배 초과하는 8,192스텝에서도 구조를 유지한다는 것은 모델이 단순 암기가 아닌 패턴의 외삽 능력을 획득했음을 시사. ⚠️ 저자 스스로 "illustrative"로 한정: 합성 정현파 신호이며 실제 도메인에 대한 일반화 주장 아님.

---

### Figure 6 (p.11) — GIFT-Eval Foundation Model 결과

**무엇을 보여주는가**: CRPS rank, MASE rank, 절대 CRPS, 절대 MASE에서 파운데이션 모델들의 순위

**핵심 관찰**:
- Toto 2.0 (2.5B, 1B, 313m): CRPS rank 20.3, 21.1, 21.4로 상위 3위 독점
- 다음 순위 PatchTST-FM r1: 23.1 (1.7포인트 격차)
- Toto 2.0 22m(26.8)이 Toto 1.0(35.1)을 8포인트 이상 능가
- ⚠️ 절대 CRPS 값: 상위 10개 모델 간 차이가 0.476~0.490으로 매우 작음 (0.014 범위)

**해석**: CRPS rank가 절대 성능보다 미세한 차이를 증폭시킬 수 있음. 그러나 공개 시계열 데이터를 사전학습에 사용하지 않고도 공개 데이터 기반 모델들을 능가했다는 점은 합성 데이터의 품질과 도메인 이전 능력의 강력한 증거.

---

## 8. 결론: 시사점, 후속 연구 계획, 추가 방향

### 저자 제시 시사점

| 시사점 | 내용 |
|--------|------|
| 스케일링 확립 | TSFM은 이제 스케일링 가능함이 실증됨; 더 많은 데이터와 더 큰 모델이 자연스러운 다음 단계 |
| "GPT-2 모멘트" | Toto 1.0이 BERT 모멘트였다면, Toto 2.0은 스케일링을 도구로 만든 GPT-2 모멘트 |
| 클래식 모델과의 격차 | 장기 예측에서 적절히 피팅된 계절 모델에 아직 미치지 못함 — 꼬리 거동, 체제 전환, 분포 외 예측 취약 |
| 데이터 큐레이션 | 스케일링이 해결된 지금, LLM 수준의 체계적 데이터 큐레이션을 TSFM에 적용할 때 |

### 저자 제시 후속 연구 계획

| 방향 | 구체 내용 |
|------|-----------|
| **메트릭 고유 모달리티** | 히스토그램, 분포형 데이터; 복합 계절성; 이종 주파수 다변량; 고차원 변수 선택 문제 |
| **멀티모달 세계 모델** | 메트릭 + 트레이스 + 로그 + 토폴로지 + 코드변경 통합; 사건 감지, 근본 원인 분석, 반사실 시뮬레이션 |
| **ARFBench 활용** | 소프트웨어 인시던트 응답 QA 벤치마크를 기반으로 한 멀티모달 추론 연구 |
| **지속 스케일링** | 더 많은 데이터, 더 큰 모델로의 확장 |

---

### 8-1. 모델의 일반화 성능 향상 가능성

#### 현재 일반화 성능의 근거

Toto 2.0이 공개 데이터 없이 사전학습했음에도 GIFT-Eval, TIME에서 1위를 달성한 메커니즘을 분석하면:

1. **TempoPFN 기반 합성 데이터의 다양성**: 비정상적 추세(nonstationary trends), 급격한 변화점(changepoints), 장범위 의존성(long-range dependencies)을 포함하여 실제 다양한 도메인의 패턴을 커버
2. **Robust Causal Scaler**: 수 자릿수에 걸친 스케일 변화를 arcsinh로 처리 — 어떤 도메인의 시계열도 같은 공간에서 모델링 가능
3. **CPM의 맥락 학습**: 가변 마스크 스팬 학습이 in-context learning 능력 강화 — 새 도메인에서도 컨텍스트로부터 패턴 추출 가능

#### 일반화 개선을 위한 방향

| 방향 | 설명 |
|------|------|
| **원칙적 데이터 믹싱** | 현재 최적 믹스는 경험적 탐색으로 도달. 정보 이론적 다양성 지표(예: MMD, KL 발산) 기반 믹싱 비율 결정 |
| **커리큘럼 학습** | 단순 합성 → 복잡 합성 → 실제 데이터 순서로 점진적 난이도 증가. LLM의 커리큘럼 전략 차용 |
| **메타러닝 통합** | MAML, ProtoNets 등 few-shot 학습을 미세조정 없는 도메인 적응에 활용 |
| **도메인 불변 표현 학습** | 도메인 레이블을 사용한 적대적 학습으로 도메인 독립적 특징 추출 |
| **컨텍스트 길이 확장** | RoPE, ALiBi 등 위치 인코딩 개선으로 학습 컨텍스트(4,096스텝)를 크게 넘어서는 일반화 강화 |
| **불확실성 정량화 개선** | 현재 9개 분위수는 극단 분위수(0.01, 0.99)를 다루지 못함 — 꼬리 거동 일반화 약점 존재 |
| **도메인 특화 파인튜닝 레시피** | 소수의 레이블된 예시로 빠른 적응이 가능한 PEFT(LoRA, prefix-tuning) 전략 연구 |

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

| 논문/모델 | 연도 | 핵심 기여 | Toto 2.0과의 관계 |
|-----------|------|-----------|-----------------|
| **Informer** (Zhou et al.) | 2021 | ProbSparse attention으로 장기 예측 효율화 | Toto 2.0은 패치 기반 접근으로 더 효율적인 시퀀스 압축 달성 |
| **PatchTST** (Nie et al., 2023) | 2023 | 패치 토큰화를 트랜스포머에 도입 | Toto 2.0의 패치 임베딩 개념의 선구자; 벤치마크에서 Toto 2.0에 의해 능가됨 |
| **TimesFM** (Das et al., 2024) | 2024 | 구글의 디코더 전용 시계열 FM | 유사한 패치 디코더 구조이나 스케일링 안정성 부재 (Figure 1에서 불안정) |
| **Chronos** (Ansari et al., 2024) | 2024 | 시계열을 토큰화하여 T5로 학습 | 언어 모델 백본 재사용; 스케일링 불안정 확인 (Chronos-2도 동일) |
| **Moirai** (Liu et al., 2025a) | 2025 | Salesforce의 범용 TSFM; 양방향 마스킹 | 유사한 멀티변량 패치 어텐션 구조이나 Toto 2.0에 의해 능가 |
| **TiRex** (Auer et al., 2025) | 2025 | xLSTM 기반 CPM 선구자 | Toto 2.0 CPM의 직접적 영감 출처; Toto 2.0이 더 긴 마스크 스팬($c_\text{max}=16$ vs. 5) 채용 |
| **Chronos-2** (Ansari et al., 2025) | 2025 | 단변량→다변량 확장 | TIME에서 CRPS rank 4.03으로 Toto 2.0 2.5B(3.43)에 뒤짐 |
| **Muon/NorMuon** (Jordan 2024; Li 2025) | 2024-25 | 행렬 직교화 기반 옵티마이저 | Toto 2.0이 TSFM 최초 채용; 핀볼 손실과의 이론적 적합성 분석 포함 |
| **u-µP** (Blake et al., 2025) | 2025 | 단위 스케일링 + µP 결합 | Toto 2.0의 스케일링 파이프라인 핵심; TSFM 최초 적용 |
| **TempoPFN** (Moroshan et al., 2025) | 2025 | PFN 프레임워크 기반 합성 시계열 생성 | Toto 2.0 합성 학습 데이터의 57.5%를 공급 |
| **TIME Benchmark** (Qiao et al., 2026) | 2026 | 오염 저항성 벤치마크 | Toto 2.0이 이 벤치마크에서도 1위 달성 — 오염 주장 반박 |

---

#### 앞으로의 연구에 미치는 영향

1. **스케일링 레시피 표준화**: Toto 2.0의 u-µP + NorMuon + CPM 조합이 TSFM 스케일링의 기준 레시피로 자리잡을 가능성. 향후 연구는 이 베이스라인 대비 개선을 입증해야 함

2. **합성 데이터 중심 패러다임 전환**: 공개 실제 데이터 없이 합성 데이터만으로 범용 벤치마크 1위 달성 — 향후 합성 데이터 생성 품질 연구가 핵심 방향이 될 것

3. **평가 방법론 재고**: TIME 같은 오염 저항성 벤치마크의 필요성을 강조. GIFT-Eval, ETTh1 등 기존 벤치마크의 포화 문제 부각

4. **확률적 예측 표준화**: 분위수 헤드가 TSFM의 확률 예측 표준으로 정착 중(Chronos-2, TimesFM도 채용) — SMM의 사실상 은퇴

#### 앞으로 연구 시 고려할 점

| 고려사항 | 이유 |
|----------|------|
| **오염 검증 의무화** | Toto 2.0 자체가 TIME 같은 오염 저항 벤치마크에서도 1위이지만, 미래 모델은 학습 데이터와 평가 데이터의 중복을 명시적으로 검증해야 |
| **스케일링 주장의 통계적 검증** | 단순 rank 비교가 아닌 bootstrap CI, permutation test 등으로 성능 차이의 유의성 검증 필요 |
| **에너지 효율 보고** | 2.5B 모델의 학습/추론 탄소 비용을 공개하는 것이 커뮤니티 규범으로 자리잡아야 |
| **다양한 도메인의 독립 검증** | 제3자가 Datadog 관련 없는 도메인(의료, 금융 등)에서 Toto 2.0을 재평가하는 독립 연구 필요 |
| **데이터 큐레이션 원칙화** | 경험적 탐색이 아닌 정보 이론 기반의 체계적 믹싱 전략 연구 시급 |
| **장기 예측과 불확실성의 정합성** | 분위수 헤드가 긴 예측 구간에서 적절히 넓어지는지(coverage) 체계적 검증 필요 |

---

## 참고자료

**논문 원문 (주 출처)**:
- Khwaja, E., Lettieri, C., Woo, G., et al. "Toto 2.0: Time Series Forecasting Enters the Scaling Era." arXiv:2605.20119v2, 2026.

**논문 내 인용 핵심 참고문헌**:
- Ansari et al. "Chronos: Learning the language of time series." TMLR, 2024.
- Ansari et al. "Chronos-2: From univariate to universal forecasting." arXiv:2510.15821, 2025.
- Auer et al. "TiRex: Zero-shot forecasting across long and short horizons." NeurIPS 2025.
- Blake et al. "u-µP: The unit-scaled maximal update parametrization." ICLR 2025.
- Blake et al. "Unit scaling: Out-of-the-box low-precision training." ICML 2023.
- Cohen et al. "Toto: Time series optimized transformer for observability." arXiv:2407.07874, 2024.
- Das et al. "A decoder-only foundation model for time-series forecasting." arXiv:2310.10688, 2024.
- Devlin et al. "BERT: Pre-training of deep bidirectional transformers." NAACL 2019.
- Jordan et al. "Muon: An optimizer for hidden layers in neural networks." 2024.
- Kaplan et al. "Scaling laws for neural language models." arXiv:2001.08361, 2020.
- Koenker & Bassett. "Regression quantiles." Econometrica, 1978.
- Li et al. "NorMuon: Making Muon more efficient and scalable." arXiv:2510.05491, 2025.
- Liu et al. "Moirai 2.0: When less is more for time series forecasting." arXiv:2511.11698, 2025.
- Montero-Manso et al. "FFORMA: Feature-based forecast model averaging." IJF, 2020.
- Moroshan et al. "TempoPFN: Synthetic pretraining of linear RNNs." arXiv:2510.25502, 2025.
- Nie et al. "A time series is worth 64 words." ICLR 2023.
- Qiao et al. "It's TIME: Towards the next generation of TSF benchmarks." arXiv:2602.12147, 2026.
- Radford et al. "Language models are unsupervised multitask learners." OpenAI TR, 2019.
- Yang et al. "Tuning large neural networks via zero-shot hyperparameter transfer." NeurIPS 2021.

**코드 및 가중치**:
- https://github.com/DataDog/toto
- https://huggingface.co/collections/Datadog/toto-20
