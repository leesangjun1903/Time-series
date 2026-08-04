# AutoTimes: Autoregressive Time Series Forecasters via Large Language Models

**참고 자료:**
- Liu, Y., Qin, G., Huang, X., Wang, J., & Long, M. (2024). *AutoTimes: Autoregressive Time Series Forecasters via Large Language Models*. NeurIPS 2024. arXiv:2402.02370v4
- 논문 내 인용 문헌 [1]–[50] (본문 참조)

---

## 1. Executive Summary (10문장 이내)

AutoTimes는 대형 언어 모델(LLM)을 시계열 예측기로 재활용하는 프레임워크로, NeurIPS 2024에 발표되었다.  
기존 LLM4TS 방법들은 LLM의 decoder-only 자기회귀(autoregressive) 특성을 무시하고 비자기회귀 방식으로 활용해 왔다는 근본적 문제를 지적한다.  
AutoTimes는 시계열 세그먼트를 언어 토큰의 임베딩 공간으로 투영하고, 다음 토큰 예측(next token prediction) 목표를 통해 자기회귀적으로 미래를 예측한다.  
LLM의 중간 레이어를 완전히 동결(freeze)하고 MLP 기반 임베딩·투영 레이어만 학습함으로써, 전체 파라미터의 0.1%만 학습 가능하게 유지한다.  
텍스트 타임스탬프를 위치 임베딩으로 활용해 시간적 정보를 인코딩하고 다변량 시계열을 정렬한다.  
시계열 자체를 프롬프트로 사용하는 **인컨텍스트 예측(in-context forecasting)** 패러다임을 최초로 제안한다.  
단일 모델로 가변 길이 입력과 임의 길이 예측을 지원하며, 기존 LLM4TS 대비 5배 이상의 학습/추론 속도를 달성한다.  
단기 예측(M4) 및 장기 예측(ETTh1, ECL 등) 벤치마크에서 최첨단 성능을 기록한다. 제로샷 일반화, 스케일링 특성 등 LLM의 고급 능력을 시계열 도메인으로 이전하는 데 성공하였다.  
확률적 예측 미지원 및 실세계 멀티모달 데이터셋 적용 미검증이 주요 한계로 남는다.

### 1-1. 연구 목적과 필요성

| 항목 | 내용 |
|------|------|
| **배경** | 시계열 파운데이션 모델 개발은 대규모 사전학습 데이터 부족과 확장 가능한 백본 부재로 제약 |
| **기회** | 자연어와 시계열의 순차적 구조 유사성 → LLM 활용 가능성 |
| **기존 문제** | 기존 LLM4TS 방법들은 LLM을 encoder처럼 사용(비자기회귀), decoder-only 구조 본질 훼손 |
| **핵심 필요성** | LLM의 자기회귀 능력을 온전히 활용해야만 제로샷 일반화, 가변 길이 예측, 스케일링 이점 확보 가능 |
| **실용적 필요성** | 기후, 경제, 에너지 등 다양한 도메인에서 범용 예측 모델에 대한 수요 증가 |

---

## 2. 핵심 주장과 근거 표

| # | 핵심 주장 | 근거 | 위치 |
|---|-----------|------|------|
| 1 | 기존 LLM4TS는 비자기회귀로 LLM 능력 미활용 | Decoder-only를 encoder처럼 사용 → 구조적 비일관성 | p.2, Fig.1(a) |
| 2 | 자기회귀 방식으로 임의 길이 예측 가능 | 다음 토큰 예측 목표 + 반복 생성 | p.5, Eq.(9) |
| 3 | 0.1% 학습 파라미터로 SOTA 달성 | MLP 0.79M / LLaMA-7B 7B 비교 | p.9, Fig.6 |
| 4 | 5× 이상 학습/추론 속도 향상 | FPT, TimeLLM 대비 시간 측정 | p.9, Fig.6 |
| 5 | 텍스트 타임스탬프가 성능 일관 향상 | 타임스탬프 유무 비교 실험 | p.21, Fig.9 |
| 6 | 인컨텍스트 예측으로 제로샷 대비 13.3% SMAPE 감소 | M4→M3 시나리오 검증 | p.8, Fig.4 |
| 7 | LLM 크기에 따른 스케일링 법칙 확인 | GPT-2~LLaMA-7B 성능 단조 증가 | p.9, Table 5, Fig.5 |
| 8 | 제로샷 일반화 성능 우수 | M4↔M3 전이 학습 | p.7, Table 4 |

### 2-1. 상세 설명

#### 해결하고자 하는 문제

기존 LLM4TS 방법(TimeLLM, FPT, UniTime 등)은 LLM을 비자기회귀 방식으로 사용한다. 구체적으로, 모든 lookback 토큰을 전역적으로 flatten하여 단일 스텝에서 예측값을 생성한다. 이는 LLM의 decoder-only 자기회귀 구조와 근본적으로 모순되며, LLM이 보유한 다중 스텝 생성 능력, 제로샷 일반화, 스케일링 특성을 충분히 활용하지 못한다.

#### 제안하는 방법 (수식 포함)

**Step 1: 시계열 토큰화**

단변량 시계열을 길이 $S$의 비중첩 세그먼트로 분할:

$$\mathbf{s}_i = \{x_{(i-1)S+1}, \ldots, x_{iS}\} \in \mathbb{R}^S, \quad i = 1, \ldots, N $$

**Step 2: 세그먼트 임베딩**

세그먼트를 LLM의 잠재 공간으로 독립적으로 임베딩:

$$\mathbf{SE}_i = \text{SegmentEmbedding}(\mathbf{s}_i) \in \mathbb{R}^D, \quad i = 1, \ldots, N $$

**Step 3: 타임스탬프 위치 임베딩**

텍스트 타임스탬프를 LLM으로 인코딩하여 위치 임베딩 추출 (`<EOS>` 토큰의 임베딩):

$$\mathbf{TE}_i = \text{SelectLast}\!\left(\text{LLM}(\text{TimestampTemplate}(\mathbf{s}_i))\right) \in \mathbb{R}^D $$

**Step 4: 토큰 임베딩 합산**

$$\mathbf{E}_i = \mathbf{SE}_i + \mathbf{TE}_i $$

**Step 5: LLM 레이어를 통한 다음 토큰 예측**

$$\{\hat{\mathbf{E}}_2, \ldots, \hat{\mathbf{E}}_{N+1}\} = \text{LLMLayers}(\{\mathbf{E}_1, \ldots, \mathbf{E}_N\}) $$

$$\hat{\mathbf{s}}_i = \text{SegmentProjection}(\hat{\mathbf{E}}_i), \quad i = 2, \ldots, N+1 $$

**Step 6: MSE 학습 목표**

$$\mathcal{L}_{\text{MSE}} = \frac{1}{NS} \sum \|\mathbf{s}_i - \hat{\mathbf{s}}_i\|_2^2, \quad i = 2, \ldots, N $$

**Step 7: 자기회귀적 임의 길이 예측 (추론)**

$$\hat{\mathbf{s}}_i = \text{LLMForecaster}(\mathbf{s}_{<i}), \quad i = 1, \ldots, \frac{F}{S} $$

**인컨텍스트 예측 공식화:**

$$\mathcal{C} = \{\text{tsp}^{(j)} = \mathbf{x}_{\leq t_j} \mid \text{earlier historical time series}\}, \quad j = 1, \ldots, m, \; t_j \leq L $$

$$f: (\mathcal{C}, \mathbf{x}_{1:L}, \mathbf{a}_{1:L+F}) \mapsto \hat{\mathbf{x}}_{L+1:L+F} $$

#### 모델 구조

```
[시계열 입력] → [세그먼트 분할(길이 S)]
     ↓
[SegmentEmbedding MLP] + [TextualTimestamp → LLM → TE_i]
     ↓ (덧셈)
[Token Embedding E_i]
     ↓
[LLM 중간 레이어 (완전 동결)]
     ↓
[SegmentProjection MLP]
     ↓
[예측 세그먼트 출력] → 반복 자기회귀 생성
```

- **학습 가능 파라미터**: SegmentEmbedding + SegmentProjection (각 2-layer MLP, 총 ~0.79M for LLaMA-7B)
- **LLM**: 완전 동결 (freeze)
- **호환 LLM**: GPT-2, OPT 시리즈, LLaMA (모든 decoder-only LLM)

#### 성능 향상

| 비교 대상 | 성능 향상 |
|-----------|-----------|
| vs. TimeLLM (장기 one-for-all) | 평균 9.12% MSE 감소 (Table 3) |
| vs. FPT (제로샷 M4→M3) | SMAPE 12.75 vs. 13.06 |
| 인컨텍스트 vs. 제로샷 | 13.3% SMAPE 감소 (Fig.4) |
| 단기 M4 | OWA 0.850 (모든 방법 중 최고) |
| 학습 속도 (LLaMA-7B 기준) | AutoTimes 0.354 vs. TimeLLM 1.896 s/iter (5.4×) |

#### 한계

- 확률적 예측(probabilistic forecasting) 미지원 (Section G)
- 이산 언어 토큰이 아닌 연속 임베딩 공간 매핑 → 언어 토큰과의 정렬 제한
- 실세계 멀티모달 데이터셋(뉴스-주가, 로그-측정치 등) 미검증
- 고급 LoRA 적용 효과 탐색 중
- 인컨텍스트 예측의 프롬프트 선택 전략 체계적 이론 미확립

---

## 3. 주장별 페이지/Figure/Table 번호

| 주장 | 위치 |
|------|------|
| 비자기회귀 LLM4TS의 구조적 모순 | p.2, Figure 1(a), Table 1 (p.3) |
| 자기회귀 시계열 토큰화 방법 | p.4–5, Figure 3, Eq.(2)–(9) |
| 타임스탬프 위치 임베딩 | p.4–5, Figure 3, Eq.(4), Figure 9 (p.21) |
| 인컨텍스트 예측 | p.6, Figure 4, Eq.(10)–(12) |
| 단기 예측 SOTA | p.7, Table 2 |
| 장기 one-for-all SOTA | p.7, Table 3; p.17, Table 10 |
| 제로샷 일반화 | p.7, Table 4; p.20, Table 16 |
| 효율성 비교 (5× 속도) | p.9, Figure 6 |
| 스케일링 법칙 | p.9, Table 5, Figure 5 |
| 어블레이션 (LLM 유용성) | p.10, Table 6 |
| LoRA 적용 효과 | p.10, Table 7 |
| 자기회귀 vs. FlattenHead | p.23, Table 21 |
| 가변 lookback 길이 | p.20–21, Figure 8 |

---

## 4. 저자 보고 결과 vs. 해석자 분석 분리

### 저자가 직접 보고한 결과

**연구 주제:**
> "AutoTimes repurposes LLMs as Autoregressive Time series forecasters, which projects time series into the embedding space of language tokens and autoregressively generates future predictions with arbitrary lengths." (Abstract, p.1)

**방법 (저자 직접 제시 수식):**

$$\mathbf{E}_i = \mathbf{SE}_i + \mathbf{TE}_i \quad \text{(Eq.5)}$$

$$\mathcal{L}_{\text{MSE}} = \frac{1}{NS} \sum \|\mathbf{s}_i - \hat{\mathbf{s}}_i\|_2^2 \quad \text{(Eq.8)}$$

**저자 보고 수치:**
- M4 단기 예측: OWA **0.850** (Table 2, p.7)
- 장기 one-for-all: TimeLLM 대비 **9.12% 평균 MSE 감소** (p.7)
- 제로샷 M4→M3: SMAPE **12.75** (Table 4, p.7)
- 인컨텍스트 예측: 제로샷 대비 **13.3% SMAPE 감소** (p.8)
- 학습 파라미터: **0.79M** (전체의 **0.1%**) (p.9)
- 훈련/추론 속도: **5× 이상** 향상 (p.9, Figure 6)
- LLaMA-7B 기준 학습: **0.354 s/iter** vs. TimeLLM **1.896 s/iter** (Figure 6)

### 해석자(검토자)의 분석

1. **방법론적 강점**: TE_i를 사전 계산(pre-computed)하여 런타임 오버헤드를 제거한 설계는 공학적으로 영리하나, 타임스탬프 템플릿의 최적 설계에 대한 이론적 근거가 부족하다.

2. **공정성 이슈**: one-for-all 벤치마크는 AutoTimes에게 구조적으로 유리하다. AutoTimes는 단일 모델로 모든 예측 길이를 처리하도록 설계된 반면, 비교 모델들은 각 예측 길이별로 별도 훈련되었다. 즉, AutoTimes의 one-for-all 성능과 다른 모델의 one-for-one 성능을 직접 비교하는 것은 설정 자체가 비대칭이다. (단, one-for-one 결과도 Table 12에서 별도 제공)

3. **오차 누적 완화 주장**: 자기회귀가 오차 누적을 완화한다고 주장하지만, 이에 대한 정밀한 이론적 증명은 없으며 경험적 결과로만 지지된다.

4. **타임스탬프 효과**: Figure 9의 타임스탬프 임베딩 효과는 일관되나, 그 메커니즘(주기성 인식인지, 절대 위치 정렬인지)에 대한 해석적 분석이 부족하다.

5. **스케일링 법칙**: GPT-2(124M)에서 LLaMA-7B까지 단조 성능 향상이 관찰되나, 더 큰 모델(70B급)에서의 거동은 미검증이다.

---

## 5. 통계적 취약점 및 비교 불가능 수치

⚠️ **통계적으로 취약하거나 주의가 필요한 부분:**

| 항목 | 문제점 |
|------|--------|
| **Table 3 (one-for-all vs. one-for-one 혼재)** | AutoTimes는 단일 모델, 타 모델은 각 길이별 독립 훈련 → **비대칭 비교** |
| **제로샷 비교 (Table 4)** | FPT 이외 LLM4TS 방법과 미비교 (TimeLLM, UniTime 제외됨) |
| **5× 속도 향상** | 배치 크기 224 고정 조건(ETTh1), 하드웨어 A100 특정 → 일반화 제한 |
| **Table 16 M3 Others** | AutoTimes SMAPE 5.79 vs. FPT **4.81** → AutoTimes가 열위인 항목 (본문에서 평균만 강조) |
| **인컨텍스트 13.3% 감소** | M4→M3 단일 시나리오, 단일 프롬프트 전략(m=1) 기준 → 다른 도메인 일반화 미검증 |
| **LoRA 효과 (Table 7)** | 일부 설정에서 개선 미미(ETTh1 Pred-192 MSE: 0.391 → 0.391) |
| **Table 9 표준편차** | 3개 시드만 사용, 통계적 검정(t-test 등) 미제공 |
| **Weather Pred-96 (Table 10)** | TimeLLM(MSE 0.149)이 AutoTimes(0.153)보다 우수 → 본문의 "80% datasets" 주장과 세부 불일치 |
| **인컨텍스트 P.4 결과** | 비관련 프롬프트 사용 시 성능 저하(13.98 vs. 13.61) → 잘못된 프롬프트의 위험성 |

---

## 6. 논문이 답하지 않는 질문

| 번호 | 미해결 질문 |
|------|------------|
| Q1 | 자기회귀 방식이 오차 누적을 완화하는 메커니즘은 무엇인가? (이론적 증명 부재) |
| Q2 | 세그먼트 길이 $S$의 최적값을 이론적으로 결정하는 방법은? |
| Q3 | 70B 이상 초대형 LLM에서도 스케일링 법칙이 유지되는가? |
| Q4 | 타임스탬프 임베딩이 성능을 향상시키는 정확한 메커니즘은 무엇인가? (주기성 인식? 절대 위치?) |
| Q5 | 확률적 예측(구간 예측, 불확실성 정량화)으로 확장 가능한가? |
| Q6 | 뉴스-주가, 로그-측정치 등 실세계 멀티모달 데이터에서 성능은? |
| Q7 | 인컨텍스트 예측을 위한 최적 프롬프트 선택 알고리즘이 존재하는가? |
| Q8 | 비영어권 또는 다국어 LLM 사용 시 성능 차이가 있는가? |
| Q9 | Channel Independence 가정이 강한 멀티변량 상관 데이터에서의 성능 한계는? |
| Q10 | AutoTimes의 추론 지연(latency) 대비 스트리밍 예측 적용 가능성은? |

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p.2) — 비자기회귀 vs. 자기회귀, 프롬프팅 메커니즘

**해석:**
- **(a)** 기존 방법들(Non-Autoregressive): lookback 토큰을 flatten+project하여 단일 스텝에서 모든 예측값을 생성. LLM의 decoder 구조를 encoder처럼 사용한다는 구조적 모순을 시각화.
- **(b)** AutoTimes(Autoregressive): 이전 토큰들을 순차적으로 받아 다음 토큰을 예측. 언어 모델의 생성 방식과 완전히 일치.
- **프롬프팅**: 기존 방법은 자연어 프롬프트를 시계열 앞에 concat → 모달리티 격차(modality gap) 발생. AutoTimes는 시계열 자체를 프롬프트로 사용(in-context forecasting) → 격차 해소.
- **논문 전체 논지의 핵심**을 한 그림에 압축한 도식.

---

### Figure 3 (p.5) — AutoTimes 전체 아키텍처

**해석:**
- 좌측: 타임스탬프 텍스트 → LLM → `<EOS>` 임베딩 추출 (사전 계산, 런타임 오버헤드 없음)
- 중앙: 시계열 세그먼트 $s_1, s_2, \ldots, s_7$ → SegmentEmbedding
- 결합: $\mathbf{E}_i = \mathbf{SE}_i + \mathbf{TE}_i$ (위치 임베딩으로 절대 시간 정보 주입)
- 상단: LLM 중간 레이어 통과(동결) → SegmentProjection → 예측 세그먼트
- **핵심 통찰**: 타임스탬프가 컨텍스트 길이를 늘리지 않으면서도 시간 정보를 효과적으로 주입하는 설계.

---

### Figure 4 (p.8) — 인컨텍스트 예측 vs. 제로샷 예측

**해석:**
- 좌측(제로샷): 소스 도메인에서 훈련한 모델을 타겟 도메인에 직접 적용
- 우측(인컨텍스트): 타겟 도메인의 이전 시계열 구간을 프롬프트로 concat → lookback 이전 컨텍스트 확장
- 바 차트: M3의 모든 하위 집합(Yearly, Quarterly, Monthly, Others)에서 일관된 SMAPE 감소 확인
  - 특히 Yearly: 21.52 → 17.03, Others: 8.46 → 5.33 (각각 약 21%, 37% 감소)
- **시사점**: 자기회귀 일관성이 확보되어야 시계열 프롬프트가 효과적으로 작동함을 실증.

---

### Figure 5 (p.9) — LLM 크기별 효율성 비교

**해석:**
- X축: 훈련 시간(ms/iter), Y축: MSE, 원의 크기: 파라미터 수
- GPT-2(124M) → LLaMA-7B로 갈수록 MSE 감소(성능 향상)하지만 훈련 시간도 증가
- **Pareto 최적점**: OPT-1.3B가 성능과 효율성의 균형점 (ECL MSE 0.164, Traffic MSE 0.397)
- LLaMA-7B가 최고 성능이지만 가장 느림 → **성능-비용 트레이드오프** 실용적 가이드 제공
- 스케일링 법칙이 시계열 도메인에서도 성립함을 시각적으로 확인.

---

### Figure 6 (p.9) — AutoTimes vs. FPT vs. TimeLLM 효율성

**해석:**
- 3개 측면 비교: 훈련 시간(s/iter), 추론 시간(s/iter), 학습 가능 파라미터(M)
- **GPT-2 기준**:
  - 훈련: AutoTimes 0.035 vs. FPT 0.284 → **8.1×** 빠름
  - 파라미터: AutoTimes 0.44M vs. FPT 7.01M → **16×** 적음
- **LLaMA-7B 기준**:
  - 훈련: AutoTimes 0.354 vs. TimeLLM 1.896 → **5.4×** 빠름
  - 파라미터: AutoTimes 0.79M vs. TimeLLM 45.66M → **58×** 적음
- **핵심**: LLM 동결 + 최소 파라미터 학습이 효율성의 근본 원천임을 정량적으로 증명.

---

## 8. 결론, 시사점 및 후속 연구

### 저자들이 제시한 시사점

1. **자기회귀 일관성의 중요성**: LLM의 decoder-only 구조를 훼손하지 않아야 다중 스텝 생성, 제로샷 일반화, 스케일링 이점을 온전히 활용할 수 있다.
2. **시계열 프롬프팅 패러다임**: 시계열 자체가 가장 효과적인 프롬프트 → 언어 프롬프트의 모달리티 격차 문제 해소.
3. **실용적 파운데이션 모델**: 단일 모델로 가변 길이 시나리오 처리 → 도메인 적응 비용 최소화.

### 저자 제시 후속 연구 계획

- 고급 LoRA 기법의 추가 탐색 (Token transition의 미세 정렬)
- 더 큰 언어 백본 활용
- 실세계 멀티모달 데이터셋(뉴스-주가, 로그-측정치) 적용
- 더 정교한 임베딩·투영 레이어 설계

---

### 8-1. 모델의 일반화 성능 향상 가능성

#### 현재 일반화 성능 분석

AutoTimes는 세 가지 일반화 능력을 실증하였다:

| 일반화 유형 | 결과 | 근거 |
|------------|------|------|
| **도메인 전이 (제로샷)** | M4→M3 SMAPE 12.75 | Table 4, p.7 |
| **가변 lookback 길이** | 384→672 시 평균 9.3% MSE 감소 | Fig.8, p.21 |
| **가변 예측 길이** | 단일 모델로 96~720 스텝 예측 | Table 3, p.7 |

#### 일반화 성능 향상 가능성 (검토자 분석)

**1. LoRA 기반 도메인 정렬 강화**

Table 7에서 LoRA 적용 시 일관된 성능 향상이 확인되었다. 특히 장기 예측(Pred-720)에서 ECL MSE: 0.216 → 0.202 (6.5% 감소)로 개선폭이 크다. 이는 **LoRA로 LLM의 토큰 전이 패턴을 시계열의 미래 외삽에 맞게 미세 조정**하면 일반화 성능이 추가 향상될 수 있음을 시사한다.

**2. 인컨텍스트 프롬프팅의 체계적 최적화**

Table 19-20에서 확인된 바와 같이, **주기성을 반영한 Ahead-Period 프롬프트**가 단순 lookback 확장보다 우수하다(P.2 SMAPE 11.80 vs. P.0 13.61). 향후 자동화된 프롬프트 검색(prompt retrieval) 알고리즘 개발 시 일반화 성능이 추가 향상될 여지가 있다.

**3. 멀티도메인 공동 사전학습**

현재는 단일 소스 도메인에서 훈련 후 타겟 도메인으로 전이하는 방식이다. 여러 도메인의 시계열을 동시에 사전학습하면 더 강건한 표현 학습이 가능하며, 이는 Moirai(Woo et al., 2024)나 MOMENT(Goswami et al., 2024) 등의 사전학습 방식에서 이미 효과가 검증되었다.

**4. 더 큰 LLM 백본 활용**

Table 5, Figure 5에서 스케일링 법칙이 확인되었다. GPT-2(124M) → LLaMA-7B(7B)로 증가 시 ECL MSE: 0.173 → 0.159 (8.1% 개선). 70B급 모델(LLaMA-3, Mistral-8×7B 등) 적용 시 추가 성능 향상이 기대되나, 계산 비용과의 트레이드오프 분석이 필요하다.

**5. Channel Independence 가정 완화**

현재는 변량 독립(Channel Independence) 가정 하에서 타임스탬프 임베딩으로만 암묵적 변량 정렬을 수행한다. 강한 변량 간 상관관계(예: 기상 데이터의 온도-습도 상관)를 명시적으로 모델링하면 일반화 성능이 향상될 수 있다. iTransformer[22]가 역전된 어텐션으로 이를 해결한 사례가 참고가 된다.

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

**⚠️ 주의**: 아래 비교는 논문 내 인용 및 공개 정보를 기반으로 하며, 2024년 이후 발표된 일부 논문의 세부 수치는 확인되지 않을 수 있습니다.

#### 주요 관련 연구 계보

| 연구 | 발표 | 핵심 기여 | AutoTimes와의 관계 |
|------|------|-----------|-------------------|
| **FPT (One Fits All)** [49] | 2023 | LLM을 범용 시계열 표현 추출기로 사용 | 비자기회귀 방식 → AutoTimes가 극복 대상으로 제시 |
| **LLMTime** [13] | 2023 | 시계열을 숫자 토큰으로 표현, 제로샷 예측 | 자기회귀이나 숫자 토큰화의 해상도 제한 |
| **TimeLLM** [15] | 2023 | 자연어 프롬프트로 LLM 재프로그래밍 | 비자기회귀, 모달리티 격차 문제 |
| **TEMPO** [7] | 2023 | 소프트 프롬프팅 기반 GPT 미세조정 | LLM 미동결, 높은 학습 비용 |
| **TEST** [34] | 2023 | 텍스트 프로토타입 정렬 임베딩 | 비자기회귀 |
| **UniTime** [21] | 2023 | 다중 도메인 통합 예측기 | LLM 미동결, 비자기회귀 |
| **Tan et al.** [35] | 2024 | LLM4TS의 LLM 실제 유용성 재검토 | AutoTimes 설계 근거 제공 (비자기회귀 비판) |
| **Moirai (Woo et al.)** [41] | 2024 | 범용 시계열 예측 Transformer 통합 사전학습 | 도메인 불가지론적 사전학습의 대안 방향 |
| **Timer (Liu et al.)** [24] | 2024 | 대규모 시계열 사전학습 Transformer | AutoTimes와 같은 청화대 연구팀, 상호보완적 접근 |

#### AutoTimes의 위치와 영향

```
LLMTime (자기회귀, 숫자 토큰)
    ↓ 해상도 한계
FPT/TimeLLM (고해상도 임베딩, 비자기회귀)
    ↓ 구조적 모순
AutoTimes (고해상도 임베딩 + 자기회귀) ← 현 논문
    ↓ 향후 방향
멀티도메인 사전학습 + 자기회귀 + 고급 정렬
```

#### 향후 연구에 미치는 영향

1. **패러다임 전환 촉진**: 비자기회귀 LLM4TS의 한계를 명확히 지적함으로써, 향후 LLM4TS 연구에서 자기회귀 일관성 유지가 표준 설계 원칙으로 자리잡을 가능성이 높다.

2. **인컨텍스트 예측의 개척**: 시계열 도메인에서 인컨텍스트 학습을 최초로 체계화. 향후 Few-shot 시계열 예측 연구의 기준점 역할.

3. **효율성 기준 제고**: 0.1% 학습 파라미터로 SOTA 달성 → 파라미터 효율적 적응이 LLM4TS의 필수 평가 기준으로 정립될 것.

#### 앞으로 연구 시 고려할 점

| 고려 사항 | 설명 |
|-----------|------|
| **벤치마크 공정성** | one-for-all vs. one-for-one 설정의 명확한 분리 및 표준화 필요 |
| **오차 누적 이론화** | 자기회귀 예측의 오차 누적 특성에 대한 이론적 분석 필요 |
| **확률적 예측 통합** | 불확실성 정량화는 실제 의사결정에서 필수 → AutoTimes 확장의 핵심 과제 |
| **LLM 선택 가이드** | 도메인/데이터 특성에 따른 최적 LLM 선택 기준 연구 필요 |
| **비영어권 LLM** | 한국어, 중국어 등 비영어권 LLM의 시계열 처리 능력 비교 |
| **실시간 스트리밍** | 산업 현장의 실시간 데이터에 대한 온라인 학습 적응 방법 |
| **멀티모달 통합** | 뉴스, 이미지, 음성 등 다양한 모달리티와의 결합 가능성 |
| **재현성** | 공개 코드(github.com/thuml/AutoTimes)를 통한 독립 재현 및 검증 필요 |

---

**참고 자료 목록:**

1. Liu, Y., Qin, G., Huang, X., Wang, J., & Long, M. (2024). *AutoTimes: Autoregressive Time Series Forecasters via Large Language Models*. NeurIPS 2024. arXiv:2402.02370v4
2. Jin et al. (2023). *Time-LLM: Time Series Forecasting by Reprogramming Large Language Models*. arXiv:2310.01728 [논문 내 참조 [15]]
3. Zhou et al. (2023). *One Fits All: Power General Time Series Analysis by Pretrained LM*. arXiv:2302.11939 [논문 내 참조 [49]]
4. Gruver et al. (2023). *Large Language Models are Zero-Shot Time Series Forecasters*. arXiv:2310.07820 [논문 내 참조 [13]]
5. Touvron et al. (2023). *LLaMA: Open and Efficient Foundation Language Models*. arXiv:2302.13971 [논문 내 참조 [36]]
6. Woo et al. (2024). *Unified Training of Universal Time Series Forecasting Transformers*. arXiv:2402.02592 [논문 내 참조 [41]]
7. Tan et al. (2024). *Are Language Models Actually Useful for Time Series Forecasting?* arXiv:2406.16964 [논문 내 참조 [35]]
8. Liu et al. (2024). *Timer: Transformers for Time Series Analysis at Scale*. arXiv:2402.02368 [논문 내 참조 [24]]
9. Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685 [논문 내 참조 [14]]
10. Nie et al. (2022). *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*. arXiv:2211.14730 [논문 내 참조 [26]]
