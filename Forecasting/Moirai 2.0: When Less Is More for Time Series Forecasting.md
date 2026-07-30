# Moirai 2.0: When Less Is More for Time Series Forecasting

**📚 참고 자료**
- **논문 원문**: Liu, C., Aksu, T., Liu, J., Liu, X., et al. (2025). *Moirai 2.0: When Less Is More for Time Series Forecasting*. arXiv:2511.11698v1. Salesforce AI Research.
- **관련 벤치마크**: Aksu et al. (2024). *GIFT-EVAL: A Benchmark for General Time Series Forecasting Model Evaluation*. arXiv:2410.10393.
- **선행 모델**: Woo et al. (2024). *Unified Training of Universal Time Series Forecasting Transformers (Moirai 1.0)*. ICML 2024.

> ⚠️ **주의사항**: 본 논문은 2025년 11월 공개된 arXiv 프리프린트로, 피어 리뷰를 거치지 않았습니다. 일부 비교 수치와 해석은 이 점을 감안하여 읽어야 합니다.

---

## 1. Executive Summary (10문장 이내)

Moirai 2.0은 Salesforce AI Research에서 개발한 디코더 전용(decoder-only) 시계열 파운데이션 모델로, 3,600만 개의 시계열(~295B 관측값)로 사전 학습되었다. 핵심 철학은 **"Less Is More"**로, 복잡한 구조를 단순화함으로써 정확도와 효율성을 동시에 향상시켰다. Moirai 1.0의 마스크 인코더, 다중 패치 크기, 혼합 분포 출력을 각각 디코더 전용 구조, 단일 패치, 분위수 손실(quantile loss)로 대체하였다. 모델은 $n_q = 9$개의 분위수를 예측하며, 자기회귀적 다중 분위수 디코딩(autoregressive multi-quantile decoding)을 통해 불확실성을 보존한다. GIFT-EVAL 벤치마크(97개 태스크, 55개 데이터셋)에서 37개 파운데이션 모델 중 5위를 기록하였다. 전임 모델인 Moirai 1.0-Large 대비 약 **30배 작고 2배 빠르면서도 더 높은 성능**을 달성하였다. 에블레이션 연구 결과, 디코더 전용 구조와 재귀적 다중 분위수 디코딩이 성능 향상에 가장 크게 기여하였다. 반면, 파라미터 수 증가에 따른 성능 향상은 관찰되지 않아 **데이터 스케일링이 모델 스케일링보다 더 중요**함을 시사한다. 장기 예측 지평(long-horizon)에서의 성능 저하와 다변량 예측 미지원이 현재의 주요 한계이다. 코드와 평가 세부 사항을 공개하여 후속 연구를 지원한다.

### 1-1. 연구의 목적과 필요성

| 구분 | 내용 |
|------|------|
| **실용적 필요성** | 클라우드 인프라, 금융, 에너지, 의료 등 광범위한 도메인에서의 시계열 예측 수요 증가 |
| **기술적 한계 (Moirai 1.0)** | 마스크 인코더로 인한 비효율적 데이터 활용(전체 토큰의 15%만 손실 계산에 참여), 다중 패치로 인한 학습 복잡성, 혼합 분포 출력의 최적화 불안정성 |
| **연구 목적** | 단순하고 효율적인 구조로 더 높은 정확도와 추론 속도를 동시에 달성하는 범용 시계열 파운데이션 모델 개발 |
| **시장 맥락** | GIFT-EVAL 벤치마크 출시 이후 25개 이상의 파운데이션 모델이 제출되는 경쟁적 환경 (p.1) |

---

## 2. 핵심 주장과 근거 표

| # | 핵심 주장 | 근거 | 페이지/Figure |
|---|-----------|------|---------------|
| 1 | 디코더 전용 구조가 마스크 인코더보다 데이터 효율이 높다 | T개 토큰에서 T-1개 손실 계산 vs. 마스킹률 15%에서 단 1개 손실 | p.6 |
| 2 | 분위수 손실이 분포 NLL 손실보다 안정적이고 효과적이다 | 에블레이션에서 v1→v2 전환 시 MASE 0.85→0.744, CRPS 0.58→0.553으로 단일 변경 중 최대 향상 | Table 2, p.10 |
| 3 | Moirai 2.0 Small이 Moirai 1.0 Large보다 30배 작고 2배 빠르면서도 성능이 우수하다 | GIFT-EVAL MASE: Moirai 2.0(0.728) < Moirai Large(CRPS 기준 하위 랭크) | Figure 2, 5, p.9 |
| 4 | 파라미터 수 증가가 성능 향상으로 이어지지 않는다 | Small(11.4M, MASE 0.728) > Base(87.1M, 0.732) > Large(305M, 0.743) | Table 1, p.10 |
| 5 | 자기회귀적 다중 분위수 디코딩이 오차 누적을 줄인다 | v2→v3 전환 시 MASE 0.744→0.736, CRPS 0.553→0.533 | Table 2, p.10 |
| 6 | 단일 패치 크기가 다중 패치보다 효율적이다 | 계산 효율성 및 정확도 향상 (정량적 단독 비교는 에블레이션에서 분리되지 않음) | p.6 |
| 7 | KV 캐시 활용 시 최대 17배 추론 속도 향상 가능 | 컨텍스트 10K, 예측 길이 10K 시 17× 속도 향상 케이스 스터디 | p.6 |

### 2-1. 해결하고자 하는 문제 / 제안 방법 / 모델 구조 / 성능 / 한계

#### 🔴 해결하고자 하는 문제

1. **비효율적 데이터 활용**: Moirai 1.0의 마스크 인코더는 15% 토큰만 학습에 기여
2. **확률적 예측의 복잡성**: 혼합 분포(mixture of distributions) 출력의 분산 붕괴·폭발 문제
3. **다중 패치 복잡성**: 다양한 시간 주파수 처리의 어려움
4. **추론 효율성**: 반복 쿼리 시 마스크 인코더의 KV 재계산 낭비

#### 🟢 제안하는 방법 (수식 포함)

**① 입력 투영 (Input Projection)**

$$\mathbf{z}_i = \text{PatchEmbed}(\hat{\mathbf{x}}_i) = \text{SiLU}(\mathbf{W}(\hat{\mathbf{x}}_i) + \mathbf{b}) + \hat{\mathbf{x}}_i \in \mathbb{R}^d, \quad i = 1, \ldots, T $$

여기서 $\hat{\mathbf{x}}\_i = \mathbf{x}\_i \| \mathbf{m}\_i$ (결측값 이진 지시자 연결), $p_{in}$은 입력 패치 크기, $d$는 임베딩 차원.

**② 분위수 손실 (Quantile/Pinball Loss)**

단일 시간 스텝 $t$, 분위수 레벨 $q \in \mathcal{Q}$에 대한 손실:

$$\ell_q(y_t, \hat{y}_t^{(q)}) = \begin{cases} q(y_t - \hat{y}_t^{(q)}) & \text{if } y_t \geq \hat{y}_t^{(q)} \\ (1-q)(\hat{y}_t^{(q)} - y_t) & \text{if } y_t < \hat{y}_t^{(q)} \end{cases}$$

전체 학습 손실 ($\mathcal{Q} = \{0.1, 0.2, \ldots, 0.9\}$, $H = Kp$):

$$\mathcal{L}_{\mathcal{Q}} = \frac{1}{H|\mathcal{Q}|} \sum_{t=1}^{H} \sum_{q \in \mathcal{Q}} \ell_q(y_t, \hat{y}_t^{(q)}) $$

$$= \frac{1}{H|\mathcal{Q}|} \sum_{t=1}^{H} \sum_{q \in \mathcal{Q}} \left[ q \max\left(y_t - \hat{y}_t^{(q)}, 0\right) + (1-q) \max\left(\hat{y}_t^{(q)} - y_t, 0\right) \right] $$

**③ 자기회귀적 다중 분위수 디코딩 (Algorithm 1)**

- **Expand**: 이전 스텝 $t$의 $m$개 분위수 각각에 대해 다음 스텝 예측 → $m^2$개 후보 생성:

```math
\mathcal{S}_{t+1} \leftarrow \left\{ \hat{y}_{t+1}^{(q_1, q_2)} : q_1, q_2 \in \mathcal{Q} \right\}, \quad |\mathcal{S}_{t+1}| = m^2
```

- **Collapse**: 각 분위수 레벨 $q$에 대해:

$$\hat{y}_{t+1}^{(q)} \leftarrow \text{Quantile}_q(\mathcal{S}_{t+1})$$

**④ 정규화 전략**

- 시계열 앞 **30%** 구간으로만 정규화 통계 계산 → 나머지 70%는 인과적 사전학습 태스크에 활용 (미래 정보 누출 방지)
- 학습 시 50% 패치 랜덤 마스킹 적용

#### 🔵 모델 구조

| 구성 요소 | 세부 내용 |
|-----------|-----------|
| **기본 아키텍처** | Decoder-only Transformer (인과적 멀티헤드 셀프 어텐션 + GLU FFN) |
| **위치 인코딩** | Rotary Positional Encodings (RoPE) |
| **정규화** | RMS Norm |
| **입력 처리** | 단일 패치 크기, 결측값 이진 지시자 연결, 잔차 블록 투영 |
| **출력** | $\mathbb{R}^d \rightarrow \mathbb{R}^{n_{token} \times n_q \times p}$ (9개 분위수, 멀티 토큰 예측) |
| **학습 최적화** | AdamW ($lr=10^{-3}$, $\beta_1=0.9$, $\beta_2=0.98$, weight decay $10^{-1}$), bf16 혼합 정밀도, 100K 스텝, 배치 256 |
| **모델 크기** | Small: 11.4M / Base: 87.1M / Large: 305M 파라미터 |

#### 🟡 성능 향상

| 비교 대상 | MASE | CRPS | 파라미터 | 속도 |
|-----------|------|------|----------|------|
| Moirai 1.0 Small | 0.946 | 0.650 | — | — |
| Moirai 1.0 Large | GIFT-EVAL 하위 | — | ~305M 이상 | 기준 |
| **Moirai 2.0 Small** | **0.728** | **0.516** | **11.4M** | **2× 빠름** |
| GIFT-EVAL 전체 순위 | 5위/37개 모델 (MASE) | 6위/37개 모델 (CRPS) | — | — |

#### 🔴 한계

- 다변량 예측(cross-variate) 미지원 → 독립 단변량으로 처리
- 공변량(covariate) 미지원
- 장기 예측 지평에서 성능 저하 (단기 4위 → 중기 6위 → 장기 8위)
- 파라미터 스케일링의 역효과 (Large가 Small보다 성능 낮음)
- Nature 도메인 성능 취약

---

## 3. 각 주장에 페이지/Figure/Table 번호 표시

| 주장 | 근거 위치 |
|------|-----------|
| 디코더 전용 구조의 데이터 효율성 | p.6, Section 3.3 |
| 분위수 손실의 우수성 | p.4, Eq.(2)(3); Table 2, p.10 |
| GIFT-EVAL 5위 달성 | p.2, p.9, Figure 2 |
| 30× 작고 2× 빠름 | p.9, Section 5.2, Figure 5 |
| 파라미터 스케일링 실패 | Table 1, p.10, Section 5.3 |
| 에블레이션 결과 | Table 2, p.10, Section 5.4 |
| 도메인별 결과 | Figure 3, p.9 |
| 예측 길이별 결과 | Figure 4, p.9 |
| KV 캐시 17× 속도 향상 | p.6, Section 3.3 |
| Nature 도메인 취약성 | p.9, Figure 3 |

---

## 4. 저자 직접 보고 결과 vs. 해석 분리

### 📊 연구 주제

| 구분 | 내용 |
|------|------|
| **저자 보고** | 시계열 파운데이션 모델의 아키텍처 단순화(디코더 전용, 단일 패치, 분위수 손실)와 새 사전학습 데이터셋으로 정확도·효율성 동시 향상 |
| **본 해석** | 이 연구는 "더 많은 것이 더 좋다"는 스케일링 법칙의 맹목적 적용에 대한 반례를 제시하며, 시계열 도메인 특유의 귀납적 편향(inductive bias)이 범용 LLM 스케일링 전략보다 더 중요할 수 있음을 시사함 |

### 📊 방법 (수식)

| 구분 | 내용 |
|------|------|
| **저자 보고** | Eq.(2)(3)의 분위수 손실, Algorithm 1의 자기회귀 다중 분위수 디코딩, 30%/70% 분할 정규화 |
| **본 해석** | 30%/70% 분할은 인과성 보장을 위한 실용적 해법이나, 최적 분할 비율의 이론적 근거는 제시되지 않음. 알고리즘 1의 expand-collapse 방식은 beam search의 변형으로, 계산 비용이 $O(m^2)$으로 증가하므로 $m$이 클 경우 효율성 trade-off 발생 |

### 📊 결과

| 구분 | 내용 |
|------|------|
| **저자 직접 보고** | GIFT-EVAL MASE 0.728(5위), CRPS 0.516(6위); Small이 Large보다 성능 우수; 전임 모델 대비 30× 소형화, 2× 속도 향상 |
| **본 해석** | 벤치마크 순위는 평가 시점(2025년 11월)의 스냅샷이며, 리더보드가 지속적으로 업데이트되므로 상대적 우위는 변동 가능. 또한 효율성 비교는 12개 선별 태스크 기반으로, 전체 97개 태스크 대상 추론 시간은 미보고 |

---

## 5. 통계적으로 취약한 부분 및 비교 불가능한 수치

| ⚠️ 취약점 | 설명 |
|-----------|------|
| **효율성 비교의 편향** | 추론 속도 실험은 12개 태스크만 사용, 전체 97개 태스크 대상 결과 미보고 (p.9). 특정 태스크 선택 기준이 성능 해석에 영향을 줄 수 있음 |
| **단일 패치 크기 단독 효과 미분리** | 에블레이션에서 단일 패치 vs. 다중 패치의 직접 비교가 없음. 다른 변경사항과 혼재 |
| **KV 캐시 속도 향상** | "케이스 스터디"로만 보고(p.6), 공식 벤치마크 평가 아님. 특정 컨텍스트 길이(10K, 예측 1K/10K)에서의 결과로 일반화 어려움 |
| **파라미터 수 증가의 역효과** | 3개 크기(Small/Base/Large) 비교만 수행. 중간 크기 다수 실험이나 통계적 유의성 검정 없음 |
| **CRPS 최적화 직접성** | 저자는 분위수 손실이 CRPS와 직접 정렬된다고 주장하나, 이는 근사적 관계이며 엄밀한 이론적 동치는 아님 |
| **내부 Salesforce 데이터** | 사전학습 데이터의 2.15M 시계열이 내부 telemetry 데이터로, 재현 불가능하며 외부 검증 어려움 |
| **Nature 도메인 취약 원인** | "사전학습 데이터에서의 과소 표현"으로 추정하나, 정량적 분석 없음 |

---

## 6. 문서가 답하지 않는 질문

| # | 미답 질문 |
|---|-----------|
| 1 | 30%/70% 정규화 분할 비율은 어떻게 결정되었는가? 다른 비율 대비 성능 차이는? |
| 2 | 단일 패치 크기의 최적값은 얼마이며, 어떻게 선택되었는가? |
| 3 | 분위수 레벨 9개($\{0.1, ..., 0.9\}$)의 선택 근거와 다른 레벨 수 대비 성능은? |
| 4 | 다변량 예측을 독립 단변량으로 처리할 때의 구체적인 성능 손실은 얼마인가? |
| 5 | 내부 Salesforce 데이터를 제외한 경우의 성능은 어떻게 변하는가? |
| 6 | Chronos-Mixup 생성 데이터 30M 시계열의 기여도는 단독으로 얼마인가? |
| 7 | 장기 예측 성능 저하의 근본 원인은 오차 누적인가, 사전학습 데이터 분포 편향인가? |
| 8 | Few-shot 또는 Fine-tuning 시나리오에서의 성능은 평가되었는가? |
| 9 | Autoregressive Multi-Quantile Decoding의 확장(expand) 깊이를 2 이상으로 할 경우의 효과는? |
| 10 | 멀티 토큰 예측(ntoken)의 최적값과 그 선택 기준은? |

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p.3): Moirai 2.0 아키텍처 개요

**구성 패널 해석:**

- **패널 1 (전체 파이프라인)**: 시계열 → 패치 분할 → 스케일링 → 랜덤 마스킹(패널 3) → 잔차 블록 투영 → 스택 Transformer(RMS Norm + Causal Self-Attention + GLU FFN) → 출력 투영 → 디스케일링 → 분위수 예측. 디코더 전용 특성상 Rotary PE가 인과적 순서를 보장.

- **패널 2 (분위수 손실)**: 실제 분위수 레이블 없이 각 지점 $y_t$를 예측된 모든 분위수와 비교. 이는 분위수 교차(crossing) 문제를 직접 방지하며 정렬과 간격을 강제.

- **패널 3 (랜덤 마스킹)**: 50% 패치를 랜덤 마스킹하여 결측 패턴에 대한 강건성 향상. 단독 사용 시 성능이 소폭 하락(v3→v4)하지만 멀티 토큰 예측과의 시너지로 최종적으로 유효.

- **패널 4 (자기회귀 다중 분위수 디코딩)**: 각 스텝에서 9→81개로 확장 후 다시 9개로 압축하는 depth-2 beam search. 중앙값으로 붕괴시키는 TimesFM 방식보다 불확실성 정보를 더 잘 보존.

**본 해석**: 이 아키텍처 설계는 LLM의 성공 요소(디코더 전용, 인과적 어텐션)를 시계열 도메인에 맞게 재적용한 것으로, 패널 4의 expand-collapse 전략은 참신하나 $m=9$일 때 매 스텝 9번의 추가 forward pass가 필요하여 추론 비용 증가 가능성 있음.

---

### Figure 2 (p.7): GIFT-EVAL 벤치마크 전체 결과

**저자 보고 해석:**
- 좌측(MASE): Chronos-2 > TimesFM-2.5 > TiRex > FlowState-9.1M > **Moirai 2.0** 순위
- 우측(CRPS): Chronos-2 > TiRex > TimesFM-2.5 > FlowState-9.1M > granite-flowstate-r1 > **Moirai 2.0** (6위)
- Moirai 2.0은 Moirai Large(오렌지색)보다 현저히 우수한 성능을 보임

**본 해석:**
- 점선으로 표시된 "Data Leakage" 기준선 아래 일부 모델이 위치하는 것은 벤치마크 공정성 문제를 시사
- Moirai 2.0(11.4M)이 Moirai Large(305M, ~26배 큰)보다 우수하다는 것은 단순 파라미터 증가의 한계를 명확히 보여줌
- ⚠️ CRPS에서 6위임에도 효율성(속도·크기) 고려 시 상위권으로 볼 수 있으나, 1~4위 모델과의 절대적 성능 격차는 기재되지 않음

---

### Figure 4 (p.8): 예측 기간별 성능 순위

**저자 보고 해석:**
- 단기(short): 4위 / 중기(medium): 6위 / 장기(long): 8위
- 장기 예측에서 FlowState, Sundial, Kairos 계열 대비 상대적으로 불리

**본 해석:**
- 이 트렌드는 자기회귀 디코딩 방식의 본질적 한계(오차 누적)를 반영
- 멀티 토큰 예측으로 완화를 시도하나, 근본적인 해결은 이루어지지 않음
- FlowState(Flow-based), Sundial 등 비자기회귀(non-autoregressive) 혹은 하이브리드 모델이 장기 예측에서 강점을 보이는 패턴과 일치
- ⚠️ "단기/중기/장기"의 정확한 정의(시간 스텝 수)가 논문에 명시되지 않아 절대적 비교 어려움

---

### Figure 5 (p.9): 속도-크기-정확도 3차원 트레이드오프

**저자 보고 해석:**
- X축: 추론 시간(GPU초, 로그 스케일), Y축: 모델 크기(파라미터 M), 주석: MASE/CRPS 랭크
- Moirai 2.0(★): 소형(~11M), 빠른 추론, MASE 랭크 5위
- Moirai Large(★): 대형(~300M), 느린 추론, MASE 랭크 11위
- Granite-FlowState-R1: Moirai 2.0보다 작고 정확하나 약 3배 느림
- Kairos-50M: 가장 빠르나 5배 큰 파라미터에 낮은 정확도

**본 해석:**
- Pareto 프론티어 관점에서 Moirai 2.0은 속도-정확도 trade-off에서 경쟁력 있는 위치
- ⚠️ 추론 시간이 12개 선별 태스크 기준이므로 그래프의 절대적 수치를 실제 배포 환경에 그대로 적용하는 것은 부적절
- 그래프에서 "데이터 누출(Data Leakage)" 모델 제외 처리가 정확하게 이루어졌는지 확인 필요

---

### Table 2 (p.10): 에블레이션 연구

**저자 보고 해석:**

| 변경 사항 | MASE 변화 | CRPS 변화 | 효과 크기 |
|-----------|-----------|-----------|-----------|
| 1.0→v0: 디코더 전용 구조 | 0.946→0.929 | 0.65→0.647 | 소 |
| v0→v1: 새 학습 데이터 | 0.929→0.850 | 0.647→0.580 | **대** |
| v1→v2: 분위수 손실 | 0.850→0.744 | 0.580→0.553 | **최대** |
| v2→v3: 재귀 디코딩 | 0.744→0.736 | 0.553→0.533 | 중 |
| v3→v4: 랜덤 마스킹 추가 | 0.736→0.772 | 0.533→0.560 | **역효과** |
| v4→v5: 멀티 토큰 예측 | 0.772→0.739 | 0.560→0.527 | 중-대 |
| v5→Final: 잔차 블록 | 0.739→0.728 | 0.527→0.516 | 소 |

**본 해석:**
- 새 학습 데이터(v0→v1)와 분위수 손실(v1→v2)이 성능 향상의 핵심 동인으로, 아키텍처 변경보다 **데이터 품질과 손실 함수가 더 중요**함을 시사
- v3→v4에서 랜덤 마스킹 단독 적용 시 역효과 발생은 흥미로운 관찰이나, 왜 멀티 토큰 예측과의 조합에서만 효과적인지 이론적 설명이 부족
- ⚠️ 각 변경사항은 순차적으로 적용되어 상호작용 효과(interaction effect)가 분리되지 않음. 진정한 인과 관계 파악을 위해서는 factorial design이 필요

---

## 8. 결론, 시사점, 후속 연구

### 저자 제시 시사점

1. **"Less Is More" 원칙 검증**: 아키텍처 단순화가 성능 향상을 가져올 수 있음
2. **데이터 스케일링 우선**: 파라미터 수 증가보다 데이터 다양성·양의 확장이 더 효과적
3. **장기 예측 및 데이터 스케일링**: 미래 연구의 핵심 방향으로 명시
4. **에이전틱 솔루션**: LLM의 추론 능력과 시계열 분석 통합
5. **멀티모달 파운데이션 모델**: 텍스트·이미지·시계열 통합 모델 개발

### 저자 제시 후속 연구 계획

- 사전학습 데이터 스케일링과 모델 용량 간의 균형 연구
- 장기 예측 지평을 위한 아키텍처 혁신
- 고품질 다변량·공변량 데이터셋 구축 (합성 데이터 생성 포함)
- LLM 기반 에이전틱 시계열 분석 시스템
- 텍스트/이미지/시계열 멀티모달 파운데이션 모델

---

### 8-1. 모델 일반화 성능 향상 가능성

**현재 일반화 관련 주요 발견:**

논문에서 일반화 성능은 다음 차원에서 분석됨:

| 차원 | Moirai 2.0 성능 | 일반화 강약점 |
|------|-----------------|---------------|
| 도메인 | 대부분 Top-10 진입 (Figure 3) | Nature 도메인 취약 ⚠️ |
| 예측 길이 | 단기 4위, 장기 8위 (Figure 4) | 장기 일반화 취약 ⚠️ |
| 주파수 | Daily~Minutely 우수, Yearly·Secondly 중간 (Figure 7) | 고주파·저주파 극단 취약 |
| 안정성 | High stability 6위 (Figure 12) | 불안정 시계열 상대적 취약 |

**일반화 향상을 위한 분석 및 제언:**

**① 도메인 불균형 문제**

Nature 도메인 취약성은 사전학습 데이터의 도메인 분포 불균형을 명시적으로 시사한다. 향후 연구에서는 도메인별 균형 샘플링(domain-balanced sampling) 전략이 필요하다:

$$\mathcal{L}_{balanced} = \sum_{d \in \mathcal{D}} w_d \cdot \mathcal{L}_{\mathcal{Q}}^{(d)}, \quad w_d \propto \frac{1}{\text{freq}(d)}$$

**② 장기 예측 일반화**

현재 자기회귀 방식은 장기 예측에서 오차가 누적되는 구조적 한계를 가짐. 다음 접근법이 효과적일 수 있음:
- **Temporal hierarchical modeling**: 다중 시간 스케일 표현 학습
- **Long-context Transformer**: Sparse attention (Longformer, BigBird 방식)
- **Non-autoregressive 병렬 디코딩** (단, 분위수 일관성 보장 필요)

**③ 분포 외 일반화 (Out-of-Distribution)**

- 학습 시 30%/70% 분할 정규화는 분포 이동(distribution shift)에 일부 대응하나, 극단적 비정상성(severe nonstationarity)에는 한계
- Reversible Instance Normalization [Kim et al., 2021]의 더 정교한 변형이나 적응적 정규화 연구 필요

**④ 파라미터 스케일링과 일반화의 역설**

Small(11.4M)이 Large(305M)보다 우수한 현상은 **이중 강하(double descent)** 현상과 관련될 수 있다. 데이터 대비 과도한 파라미터가 오히려 과적합(overfitting) 혹은 사전학습 데이터 분포에 대한 과도한 특화를 야기할 수 있다. 이는 Kaplan et al. [2020]의 언어 모델 스케일링 법칙이 시계열 도메인에 직접 적용되지 않음을 보여주며, **도메인 특화 스케일링 법칙** 연구가 필요함을 시사한다.

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ 아래 비교는 논문 내 인용 정보와 공개된 아카이브 정보 기반이며, 논문 외 정보는 제한적으로 확인된 사실만 기술합니다.

#### 주요 경쟁 모델과의 비교

| 모델 | 기관 | 아키텍처 | 출력 유형 | GIFT-EVAL 순위(MASE) | 특징 |
|------|------|---------|-----------|----------------------|------|
| **Chronos-2** [Ansari et al., 2025] | Amazon | Encoder-only | Quantile | 1위 | 단변량→범용 확장 |
| **TimesFM-2.5** [Das et al., 2024+] | Google | Decoder-only | Point | 2위 | 대규모 Google 데이터 |
| **TiRex** [Auer et al., 2025] | JKU | xLSTM | Quantile | 3위 | 비Transformer, 장단기 균형 |
| **FlowState** [Graf et al., 2025] | IBM | Enc-Dec Hybrid | Flow-based | 4위 | 샘플링 주파수 불변 |
| **Moirai 2.0** (본 논문) | Salesforce | Decoder-only | Quantile | 5위 | 극소형·고속 |
| **Kairos** [Feng et al., 2025] | — | Transformer | Point | ~7위 | 적응형·일반화 |
| **Sundial** [Liu et al., 2025] | Tsinghua | Decoder-only | Flow | ~8위 | 대용량 사전학습 |

#### 핵심 트렌드 분석

**① 분위수 예측으로의 수렴**

Chronos(분포) → Chronos-Bolt → Chronos-2(분위수), TiRex(분위수), Moirai 2.0(분위수) 등 최신 고성능 모델들이 분위수 출력으로 수렴하는 경향. 이는 CRPS 메트릭과의 직접적 정렬 때문으로 해석.

**② 아키텍처 다양성의 증가**

2023년 이전: Transformer 일변도 → 2025년: xLSTM(TiRex), PFN(TabPFN-TS), Flow-based(FlowState) 등 다양한 구조 경쟁. Transformer가 최선이라는 가정이 도전받고 있음.

**③ 스케일링 법칙의 한계**

Moirai 2.0의 발견(소형 > 대형)은 GIFT-EVAL 논문[Aksu et al., 2024]에서도 관찰된 패턴과 일치. 시계열 도메인에서는 언어 모델의 스케일링 법칙이 성립하지 않을 수 있으며, 데이터 다양성이 모델 크기보다 중요할 수 있음.

**④ 벤치마크 생태계의 성숙**

GIFT-EVAL(97 tasks, 55 datasets)과 FEV-BENCH의 등장으로 표준화된 평가가 가능해졌으나, 벤치마크 특화(benchmark overfitting) 위험도 증가.

#### 앞으로의 연구에 미치는 영향

| 영향 | 구체적 내용 |
|------|-----------|
| **아키텍처 설계** | "단순함이 최선"이라는 패러다임 강화. 연구자들은 복잡한 아키텍처보다 학습 전략·데이터·손실 함수 최적화에 더 집중할 가능성 |
| **데이터 중심 AI** | 사전학습 데이터의 규모(36M 시계열)와 다양성이 모델 크기보다 중요함을 실증. 고품질 시계열 데이터셋 구축 연구가 활성화될 것 |
| **분위수 예측 표준화** | 분위수 손실의 효과성 재입증으로 차세대 파운데이션 모델의 기본 출력 방식으로 자리잡을 가능성 |
| **효율성 경쟁** | 30× 소형화 달성으로 엣지 배포(edge deployment) 가능성을 보여줌. 경량 시계열 파운데이션 모델 연구 가속화 |
| **멀티모달 통합** | 저자의 미래 방향 제시가 LLM+시계열 통합 연구(JoLT, ChatTime, ChatTS 등)에 대한 관심을 더욱 높일 것 |

#### 앞으로 연구 시 고려할 점

| 고려 사항 | 구체적 방향 |
|-----------|------------|
| **도메인 균형 사전학습** | Nature, Secondly 등 희귀 도메인의 의도적 과샘플링(oversampling) 전략 필요 |
| **장기 예측 특화 설계** | 자기회귀 오차 누적 문제 해결을 위한 계층적 표현 학습이나 비자기회귀 방식 탐구 |
| **다변량 관계 모델링** | 독립 단변량 처리의 한계를 극복하기 위해 cross-variate attention 메커니즘과 고품질 다변량 데이터 필요 |
| **스케일링 법칙 재정립** | 시계열 도메인 특화 스케일링 법칙 연구: 최적의 모델 크기-데이터 비율 탐색 |
| **공정한 벤치마크 설계** | 내부 독점 데이터(Salesforce telemetry)가 포함된 모델의 평가 공정성 담보 방안 |
| **불확실성 보정 평가** | CRPS 외에도 Expected Calibration Error(ECE), Reliability Diagram 등 보정 평가 지표 추가 |
| **적응형 추론 전략** | 예측 지평에 따라 자기회귀 스텝 수를 동적으로 조정하는 적응형 디코딩 연구 |

---

**📌 최종 참고문헌 목록**

1. Liu, C., Aksu, T., Liu, J., Liu, X., et al. (2025). *Moirai 2.0: When Less Is More for Time Series Forecasting*. arXiv:2511.11698v1.
2. Aksu, T., et al. (2024). *GIFT-EVAL: A Benchmark for General Time Series Forecasting Model Evaluation*. arXiv:2410.10393.
3. Woo, G., et al. (2024). *Unified Training of Universal Time Series Forecasting Transformers*. ICML 2024.
4. Ansari, A.F., et al. (2024). *Chronos: Learning the Language of Time Series*. TMLR.
5. Ansari, A.F., et al. (2025). *Chronos-2: From Univariate to Universal Forecasting*. arXiv:2510.15821.
6. Das, A., et al. (2024). *A Decoder-Only Foundation Model for Time-Series Forecasting*. ICML 2024.
7. Auer, A., et al. (2025). *TiRex: Zero-Shot Forecasting Across Long and Short Horizons*. arXiv:2505.23719.
8. Graf, L., et al. (2025). *FlowState: Sampling Rate Invariant Time Series Forecasting*. arXiv:2508.05287.
9. Feng, K., et al. (2025). *Kairos: Towards Adaptive and Generalizable Time Series Foundation Models*. arXiv:2509.25826.
10. Liu, Y., et al. (2025). *Sundial: A Family of Highly Capable Time Series Foundation Models*. arXiv:2502.00816.
11. Kaplan, J., et al. (2020). *Scaling Laws for Neural Language Models*. arXiv:2001.08361.
12. Kim, T., et al. (2021). *Reversible Instance Normalization for Accurate Time-Series Forecasting*. ICLR 2021.
13. Nie, Y., et al. (2023). *A Time Series is Worth 64 Words: Long-Term Forecasting with Transformers (PatchTST)*. ICLR 2023.
