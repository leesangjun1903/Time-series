# Seeking SOTA: Time-Series Forecasting Must Adopt Taxonomy-Specific Evaluation to Dispel Illusory Gains

> **⚠️ 중요 고지:** 이 논문은 arXiv 프리프린트(arXiv:2603.15506v1, 2026년 3월)이며, NeurIPS 2025 Position Paper Track에서 **Reject** 판정을 받은 상태입니다. 아래 분석은 제공된 PDF 원문에 기반하며, 불확실한 부분은 명시합니다.

---

## 1. Executive Summary (10문장 이내)

1. 본 논문은 AI/ML 시계열 예측(TSF) 분야의 벤치마크 평가 방식에 근본적인 문제가 있음을 주장하는 **포지션 페이퍼**이다.
2. 현재 표준 LTSF(Long-horizon TSF) 벤치마크(ETT, Traffic, Electricity 등 9개)는 강하고 지속적인 주기성(periodicity)을 지닌 데이터로 편중되어 있다.
3. 이러한 데이터 특성 때문에, 복잡한 딥러닝·트랜스포머·LLM 기반 모델이 단순한 선형 모델(AR, DLinear)이나 통계 모델(ARIMA)보다 실질적으로 우월하지 않음을 Table 3에서 실증적으로 보인다.
4. 저자들은 현재 SOTA 경쟁이 "벤치마크 아티팩트(artifact)"에 의한 **착시적 성과(illusory gains)**임을 지적한다.
5. "Seeking SOTA Athlete" 사고 실험을 통해, 일반론자(generalist) 모델이 집계 지표에서 SOTA를 달성하지만 개별 작업에서는 전문가보다 열등할 수 있음을 직관적으로 설명한다.
6. 트랜스포머 자기-어텐션 메커니즘의 순열 불변성(permutation invariance)은 시간적 순서 모델링에 구조적으로 부적합하다는 선행 연구[68]를 지지한다.
7. MSE/MAE 단독 사용의 한계(스케일 의존성, 계절성 무시 등)를 지적하며, MASE·OWA 등 다양한 지표 사용을 권장한다.
8. 저자들은 커뮤니티에 두 가지를 요구한다: (I) 구조적 변화(structural break), 개념 드리프트(concept drift), 시변 변동성(time-varying volatility) 등 다양한 비정상성을 포함하는 벤치마크 개발, (II) 모든 딥러닝 논문에 적절한 고전 기준선(classical baseline) 의무 포함.
9. TimeLLM, CALF 등 고인용 논문의 결과 불일치 및 재현성 문제를 구체적 사례로 제시하며 "연쇄 오류(cascading errors)" 위험을 경고한다.
10. GIFT-Eval[1] 같은 표준화된 제3자 벤치마크 플랫폼 채택을 통한 투명하고 재현 가능한 평가 프로토콜 구축을 최종 권고한다.

### 1-1. 연구의 목적과 필요성

**목적:** 현행 LTSF 벤치마크 평가 패러다임의 구조적 결함을 진단하고, 분류체계별(taxonomy-specific) 평가 체계로의 전환을 촉구하는 것이다.

**필요성:**
- Goodhart의 법칙("측정치가 목표가 되면, 좋은 측정치가 아니게 된다")이 TSF 연구에 현실화되고 있음 (p.1)
- 2021년 Informer[73]가 도입한 9개 데이터셋이 사실상 표준이 되었으나, 이 데이터셋들은 모두 강한 주기성을 공유하는 좁은 스펙트럼을 대표함 (Table 1, p.2)
- 수천 개의 파라미터를 가진 SparseTSF[37]가 수억 파라미터의 LLM 기반 모델과 경쟁하는 역설적 현상이 지속됨 (p.2)
- 잘못된 평가 관행이 후속 연구에 연쇄적으로 전파되는 "cascading errors" 문제가 심각함 (p.7)

---

## 2. 핵심 주장과 근거 표

| # | 핵심 주장 | 근거 | 위치 |
|---|-----------|------|-------|
| 1 | 표준 LTSF 벤치마크는 주기성 편향을 가짐 | 9개 데이터셋 모두 강한 ACF 주기 패턴 확인; train/test 경계에서 패턴 연속성 시각화 | Figure 1, Figures 3–9, Table 1 (p.2) |
| 2 | 복잡한 DL 모델이 단순 통계 모델보다 실질적으로 우월하지 않음 | AR(1951), Auto-AR이 ETT 4개 데이터셋에서 대다수 최신 모델과 동등하거나 우수 | Table 3 (p.8) |
| 3 | 트랜스포머 자기-어텐션은 시계열에 구조적으로 부적합 | 순열 불변성 문제; Zeng et al.[68]의 DLinear가 트랜스포머 계열 모두를 능가한 선례 | §3 (p.4–5) |
| 4 | 집계 지표 기반 SOTA는 착시적 성과를 유발 | Seeking SOTA Athlete 사고 실험; Stein의 역설과의 유비 | §4.1–4.2, Table 4 (p.5–6) |
| 5 | 평가 지표 선택이 부적절함 | MSE/MAE의 스케일 의존성, 계절성 무시 등 4가지 한계 기술 | Table 6, §D.2 (p.21) |
| 6 | 고전적 기준선 생략이 만연함 | TimeLLM 등 500+ 인용 논문의 재현 불일치; 나이브 기준선 미포함 사례 | §5.1.2 (p.8–9) |
| 7 | 도메인 특화 SOTA 기준선 생략 | WEATHER 데이터셋 논문들이 GraphCast/WeatherBench 무시 | §5.1.2 (p.9) |
| 8 | LLM은 시계열 시간적 추론에 취약 | GPT-4가 zero-shot 원인 추론에서 무작위 추측 수준; Merrill et al.(2024)[46] | §3 (p.4) |

### 2-1. 해결 문제, 제안 방법, 모델 구조, 성능, 한계

#### 해결하고자 하는 문제

**핵심 문제:** 시계열 예측 연구에서 보고되는 SOTA가 진정한 방법론적 발전이 아니라 특정 벤치마크의 데이터 특성(강한 주기성)에 최적화된 **아티팩트**일 가능성.

구체적 문제들:
1. 주기성 편향 벤치마크로 인한 DL 모델의 과대평가
2. 고전 기준선 부재로 인한 상대적 우월성 착시
3. 부적절한 지표(MSE/MAE 단독 사용)로 인한 왜곡된 성능 비교
4. 연쇄 오류(cascading errors)의 누적

#### 제안하는 방법 (이 논문은 새 모델을 제안하지 않는 포지션 페이퍼임)

**분석 프레임워크:**

시계열 분해 관점:

$$X(t) = P(t) + T(t), \quad \text{where } P(t) = P(t+w) \text{ for some period } w$$

여기서 $P(t)$는 주기 성분, $T(t)$는 추세 성분이다. SparseTSF 같은 경량 모델이 효과적인 이유는, 강한 $P(t)$가 존재할 때 모델이 단순히 $f: X(t) \rightarrow T(t)$ 매핑만 학습하면 되기 때문이다 (p.2–3).

**평가 지표:**

$$\text{MSE} = \frac{1}{H}\sum_{i=1}^{H}(y_{T+i} - \hat{y}_{T+i})^2$$

$$\text{MAE} = \frac{1}{H}\sum_{i=1}^{H}|y_{T+i} - \hat{y}_{T+i}| $$

$$\text{MAPE} = \frac{100}{H}\sum_{i=1}^{H}\frac{|y_{T+i} - \hat{y}_{T+i}|}{|y_{T+i}|} $$

$$\text{sMAPE} = \frac{200}{H}\sum_{i=1}^{H}\frac{|y_{T+i} - \hat{y}_{T+i}|}{|y_{T+i}| + |\hat{y}_{T+i}|} $$

$$\text{MASE} = \frac{1}{H}\sum_{i=1}^{H}\frac{|y_{T+i} - \hat{y}_{T+i}|}{\frac{1}{T-H-m}\sum_{j=1}^{T-m}|y_j - y_{j-m}|} $$

$$\text{OWA} = \frac{1}{2}\left[\frac{\text{sMAPE}}{\text{sMAPE}_{\text{Naïve2}}} + \frac{\text{MASE}}{\text{MASE}_{\text{Naïve2}}}\right] $$

저자들은 MSE/MAE의 한계를 지적하며 MASE, OWA 등 scale-free 지표 사용을 권고한다 (Table 5, p.21).

**제안하는 권고 사항 (4가지):**
1. 과업별 평가 및 데이터 관련성 명시
2. 표준화·투명·재현 가능한 평가 프로토콜 (GIFT-Eval[1] 채택)
3. 강건한 기준선 및 목표 정렬 지표 의무화
4. 멀티모달·맥락 인식·다양성 벤치마크 개발

#### 모델 구조

> ⚠️ 이 논문은 새로운 모델을 제안하지 않습니다. 기존 모델들을 비교 분석하는 포지션 페이퍼입니다.

비교된 모델 범주 (§2):
- **Specialist Transformer**: Informer, Autoformer, FEDformer, Pyraformer, PatchTST
- **TSF LLMs/FMs**: GPT4TS(OFA), TimeLLM, TEMPO, TimesFM, LLM4TS, MOMENT, TTM
- **간단한 기준선**: DLinear, SparseTSF, AR, Auto-AR

#### 성능 향상 주장

저자들이 보고한 핵심 결과 (Table 3, p.8):

| 모델 | ETTh1-96 MSE | ETTh2-96 MSE |
|------|-------------|-------------|
| AR (1951) | **0.357** | 0.271 |
| Auto-AR (2008) | 0.357 | **0.269** |
| DLinear* (원 논문) | 0.375 | 0.289 |
| iTransformer (2024) | 0.386 | 0.297 |
| Informer (2021) | 0.865 | 3.755 |
| LLM4TS (2024) | 0.371 | 0.269 |

> **⚠️ 통계적 취약점:** Table 3의 결과는 저자들이 직접 실험한 것이 아니라 대부분 기존 논문 및 Xu et al.[67]에서 재인용한 수치임. 공정한 하이퍼파라미터 튜닝 여부, 동일 데이터 전처리 여부 불확실.

#### 한계

저자들이 명시적으로 인정한 한계:
1. 포지션 페이퍼로서 새로운 모델이나 방법론을 제시하지 않음
2. Table 3의 일부 결과는 재현 구현(reimplementation)에서 원 논문 값과 불일치함 (SparseTSF*, TTMA* 등)
3. 사고 실험(Table 4)의 수치는 ChatGPT-4o와 Gemini-2.5 Pro의 추정치를 평균한 **가설적 값**임
4. 고전 방법론의 한계(고차원 설정, 외생 변수 풍부 환경)를 충분히 논의하지 않음 (Reviewer 2 지적)
5. GIFT-Eval 등 최신 대규모 벤치마크 이니셔티브와의 연계 부족 (Area Chair 지적)
6. **NeurIPS 2025 Position Paper Track에서 Reject** (평균 점수 5.67/10)

---

## 3. 각 주장별 위치 표시

| 주장 | 위치 |
|------|------|
| 표준 벤치마크의 주기성 편향 | p.1–3, Figure 1, Table 1 |
| 시계열 분해 수식 $X(t) = P(t) + T(t)$ | p.2–3 |
| MSE/MAE 정의 | p.4, Equation (1) |
| LLM의 시간적 추론 한계 | p.4–5, §3 |
| 트랜스포머 순열 불변성 문제 | p.4, §3 |
| Seeking SOTA Athlete 사고 실험 | p.5–6, §4.1, Table 4 (Appendix C.1) |
| Stein의 역설과의 유비 | p.6, §4.2 |
| Cascading errors 위험 | p.7, §5 |
| 평가 지표 부적절성 | p.7, §5.1.1 |
| 고전 기준선 생략 문제 | p.8–9, §5.1.2, Table 3 |
| AR이 최신 모델과 동등한 성능 | p.8, Table 3 (맨 하단 행) |
| 도메인 특화 기준선 생략 (GraphCast) | p.9, §5.1.2 |
| 비정상성 유형 분류 | Figure 2 (p.4), Table 7 (p.22) |
| 평가 지표 비교 | Table 5 (p.21), Table 6 (p.21) |
| ETT 데이터셋 ACF 분석 | Figures 3–6, Appendix §E.1 |
| NeurIPS 심사 결과 | Appendix F, Table 10 (p.30) |

---

## 4. 연구 주제·방법·결과: 저자 보고 vs. 독자적 해석

### 연구 주제

| 구분 | 내용 |
|------|------|
| **저자 보고** | "표준 LTSF 벤치마크가 주기성 지배적 데이터에 편향되어 DL 모델의 진정한 성능을 오도한다" |
| **내 해석** | 이는 벤치마크 편향의 문제를 넘어, AI/ML 커뮤니티의 인센티브 구조(SOTA 경쟁, 논문 출판 압력)가 평가 엄밀성보다 성능 수치 향상을 우선시하게 만드는 구조적 문제를 반영한다. |

### 방법

| 구분 | 내용 |
|------|------|
| **저자 보고** | ACF 분석으로 주기성 입증; 기존 논문에서 결과 수집·통합; 사고 실험(Seeking SOTA Athlete); Stein의 역설과의 유비 적용 |
| **내 해석** | 방법론 자체가 실험적이기보다 **논증적(argumentative)**이며, 핵심 증거인 Table 3은 이질적 출처의 결과를 통합한 것으로 직접 통제된 실험이 아니다. 이는 비교 공정성에 근본적 한계를 내포한다. |

### 결과

| 구분 | 내용 |
|------|------|
| **저자 직접 보고** | AR(1951)이 ETTh1에서 MSE=0.357(horizon 96)로 최신 모델들(iTransformer 0.386, TimeMixer 0.375 등)보다 우수하거나 동등; SparseTSF 재현 결과가 원 논문 값과 상이(ETTh1-96: 0.359→0.363) |
| **내 해석** | AR의 "우수성"은 진정한 우월성이 아닐 수 있음. 이 데이터셋들이 AR에 유리한 강한 자기상관 구조를 가지기 때문이며, 실제로 더 복잡한 비정상성(금융 시계열 등)에서는 결과가 역전될 가능성이 높다. 이 점은 저자들도 간접적으로 인정한다. |

---

## 5. 통계적 취약점 및 비교 불가능한 수치

> ⚠️ = 통계적으로 취약 | 🚫 = 비교 불가능

| 항목 | 문제 유형 | 설명 |
|------|-----------|------|
| **Table 3 결과 통합** | ⚠️🚫 | 결과가 여러 논문, 여러 재현 구현에서 수집됨. 동일한 데이터 전처리, 하이퍼파라미터 튜닝, 시드 설정 보장 불가. 저자 직접 비교 실험 없음. |
| **SparseTSF 원본 vs. 재현** | ⚠️ | ETTh1-96: 0.359(원본*) vs. 0.363(재현). 차이의 원인(하이퍼파라미터, 랜덤 시드 등) 미분석. |
| **DLinear 세 버전의 불일치** | ⚠️🚫 | DLinear*=0.375, 재인용([61])=0.397, 저자 재현=0.379. 동일 모델의 큰 수치 편차가 설명 부족. |
| **Table 4 (Seeking SOTA)** | 🚫 | 저자의 스포츠 성적 일부가 ChatGPT-4o/Gemini-2.5 Pro 추정치 평균. 완전히 가설적 수치. |
| **통계적 유의성 검정 부재** | ⚠️ | 어떤 비교에서도 p-value, 신뢰구간, Friedman 검정 등 통계적 유의성 검정 없음. |
| **단일 데이터 분할** | ⚠️ | 고정된 단일 train/test split만 사용. 다중 분할 또는 시계열 교차검증(walk-forward) 미적용. |
| **AR 결과의 맥락 부재** | ⚠️ | AR의 차수(order) $d$ 설정 방식("with $d=0$" 표기) 불명확. ARIMA와의 관계 미설명. |
| **ETTm1, ETTm2의 FM 결과 누락** | 🚫 | Table 3에서 ETTm1/ETTm2에 대한 FM 계열(SparseTSF 포함) 결과가 "-"로 표시. 불완전 비교. |

---

## 6. 논문이 답하지 않는 질문

1. **비주기성 데이터에서의 DL 우월성 임계점:** 비정상성이 얼마나 복잡해야 DL 모델이 통계 모델보다 실질적으로 우월해지는가? 정량적 기준이 없음.

2. **최적 고전 기준선 선택 방법:** 특정 데이터셋의 비정상성 유형에 따라 어떤 기준선(AR, ARIMA, ETS, 계절 나이브 등)을 선택해야 하는지 구체적 가이드라인 부재.

3. **트랜스포머의 우위 조건:** Reviewer 3이 지적했듯, 어떤 조건(데이터 특성, 지평선 길이 등)에서 트랜스포머 계열이 통계 모델보다 명확히 우월한가?

4. **LLM의 잠재적 역할:** 외생 텍스트 정보 통합, 제로샷 전이 등 LLM이 실질적으로 기여할 수 있는 구체적 TSF 시나리오는?

5. **고전 방법의 확장성 한계:** 862채널의 Traffic 같은 고차원 다변량 설정에서 ARIMA 계열의 실용적 적용 방법과 한계는?

6. **분류체계(taxonomy) 정의:** "taxonomy-specific evaluation"의 구체적 분류 기준(어떤 속성을 기준으로 데이터를 분류할 것인가)이 명확히 제시되지 않음.

7. **새 벤치마크 구축 방법:** 다양한 비정상성을 포함하는 벤치마크를 어떻게 설계할 것인지 구체적 방법론 부재.

8. **재현성 보장 메커니즘:** 권장하는 GIFT-Eval 외에 커뮤니티 레벨의 재현성 보장을 위한 구체적 메커니즘은?

9. **계산 효율성 vs. 성능:** DL 모델의 높은 계산 비용이 특정 도메인(실시간 예측 등)에서 정당화되는 조건은?

10. **확률적 예측 평가:** 점 예측(point forecasting) 비판에 집중하며, 불확실성 정량화(CRPS, 구간 커버리지)에 대한 체계적 분석 부재.

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p.2): 5개 데이터셋 목표 변수 시각화

**구성:** ETTh2, ETTm2, Traffic, Exchange, ILI의 전체 시계열(상단)과 train/test 분할 경계 확대(하단)

**핵심 메시지:**
- ETTh2/ETTm2/Traffic: train에서 test로 경계를 넘어도 주기적 패턴이 **끊김 없이 연속**됨. 이는 테스트 데이터가 훈련 데이터의 주기를 단순 연장한 것임을 시사.
- Exchange Rate: 뚜렷한 주기성 없는 랜덤워크 성격의 시계열. 이 데이터셋이 벤치마크에 포함됨에도 주기성 기반 모델과 동일 지표로 비교됨의 문제를 시각적으로 드러냄.
- ILI: 강한 연간 계절성(겨울 독감 피크). 계절 나이브 모델이 강력한 기준선이 될 수 있음을 직관적으로 보여줌.

**내 해석:** 이 그림은 논문의 핵심 논증을 가장 직관적으로 지지하는 시각화다. 특히 train/test 경계의 패턴 연속성은 "테스트 데이터가 훈련 데이터와 다른 도전을 제시하지 않는다"는 주장의 강력한 시각적 증거다.

---

### Figure 2 (p.4): 비정상성 유형 분류 도식

**구성:** Trend, Seasonal, Random Walk, Changepoint, Heteroscedastic, Cyclostationary 6가지 비정상성의 개략적 시뮬레이션

**핵심 메시지:**
- 실제 시계열이 가질 수 있는 비정상성의 스펙트럼을 도식화함.
- 현재 LTSF 벤치마크는 이 중 "Seasonal"에 과도하게 집중되어 있음.
- Changepoint, Heteroscedastic, Random Walk 등은 현재 벤치마크에서 체계적으로 과소 대표됨.

**내 해석:** 이 그림은 논문의 "taxonomy-specific evaluation" 개념의 이론적 기반을 제공한다. 그러나 각 비정상성 유형에 대한 정량적 정의나 테스트 통계 기반 분류 기준이 제시되지 않아, 실제 데이터셋 분류에 적용하기 어렵다는 한계가 있다.

---

### Figure 3 (p.23): ETTh1 ACF 분석 (대표)

**구성:** ETTh1의 7개 채널 각각에 대한 ACF 플롯(훈련 vs. 테스트), 전체 시계열, 확대 시계열

**핵심 메시지:**
- HUFL, MUFL 채널: 훈련과 테스트 모두 **lag=24**에서 dominant peak (일간 주기성)
- 패턴이 train/test 경계를 넘어 거의 동일하게 유지됨
- 이는 테스트 예측이 단순히 훈련 데이터의 주기를 연장하는 것으로 충분함을 의미

**내 해석:** ACF 분석은 논문의 가장 엄밀한 실증적 근거다. 그러나 저자들이 인정하듯, ACF peak의 존재가 모든 채널에서 동일 강도는 아니며(MULL, LUFL 등은 상이한 주기), 이는 벤치마크 편향 주장의 보편성을 다소 약화시킨다.

---

### Table 3 (p.8): ETT 4개 데이터셋 장기 예측 MSE 비교

**구성:** 2021–2025년 발표 30개 모델의 horizon {96, 192, 336, 720} MSE 값

**핵심 메시지:**
- **AR(1951)**: ETTh1에서 MSE=0.357 (horizon 96) — 최신 모델들 중 최고 성능
- **Auto-AR(2008)**: ETTh2에서 MSE=0.269 (horizon 96) — LLM4TS(0.269)와 동일
- DL 모델들의 성능 개선이 2021→2025년 동안 매우 점진적 (소수점 3째 자리 수준)
- TimeLLM 등 LLM 기반 모델들이 AR보다 대체로 열등하거나 동등

**내 해석:**

> ⚠️ 이 표의 가장 중요한 통계적 한계: 각 모델의 결과가 서로 다른 구현, 다른 논문에서 수집되어 직접 비교의 공정성이 보장되지 않는다.

그럼에도 불구하고, AR이 수십 년 후의 최신 모델들과 경쟁하는 패턴은 다양한 재현 시도에서 반복적으로 나타났다는 점에서 벤치마크 편향의 실증적 증거로서 설득력이 있다.

---

### Figure 9 (p.29): ILI 데이터셋 시각화

**구성:** 국가 독감 주간 데이터의 전체 시계열(2002–2020)과 train/test 경계 ±60주 확대

**핵심 메시지:**
- 연간 독감 계절성(겨울 피크)이 20년간 매우 규칙적으로 반복
- train/test 경계 전후의 패턴이 동일 계절 사이클을 따름
- "계절 나이브" 모델(전년 동기 값 반복)이 이미 매우 강력한 기준선임을 시사

**내 해석:** ILI 데이터셋은 역설적으로 현 벤치마크 문제를 가장 극명히 보여주는 사례다. 독감 예측이라는 의료적으로 중요한 응용에서, 실제 도전(팬데믹 급격한 변화, COVID-19 같은 구조적 변화)은 이 데이터셋에 전혀 반영되지 않는다. 이는 저자들의 "실세계 복잡성 부재" 주장을 강하게 지지한다.

---

## 8. 결론: 시사점, 후속 연구, 추가 방향

### 저자들이 제시한 시사점

저자들의 핵심 시사점(§6, Appendix B):

1. **평가의 투명성 확보:** 모든 TSF 논문이 나이브·통계·단순 ML 기준선을 반드시 포함해야 함
2. **표준화된 프로토콜:** GIFT-Eval[1] 같은 중립적 제3자 플랫폼 채택
3. **다양한 지표 사용:** MSE/MAE 단독 사용 지양, MASE·OWA 추가 보고
4. **벤치마크 다양화:** 구조적 변화, 개념 드리프트 등을 포함하는 새 데이터셋 개발
5. **커뮤니티 문화 변화:** 리뷰어·편집자가 기준선 포함 여부를 심사 필수 요건으로 만들어야 함

### 저자들의 후속 연구 계획

> ⚠️ 저자들은 이 포지션 페이퍼에서 구체적인 후속 연구 계획을 명시하지 않았습니다. 아래는 논문에서 간접적으로 유추할 수 있는 방향입니다.

- 다양한 비정상성 유형을 체계적으로 포함하는 새 벤치마크 구축
- taxonomy 기반 모델 선택 가이드라인 개발
- 멀티모달 TSF (텍스트+시계열) 평가 체계 연구

### 8-1. 모델의 일반화 성능 향상 가능성

논문에서 일반화 성능과 관련된 핵심 논점:

**저자 주장:** 현재 DL 모델들은 주기성 지배 데이터에서 "일반화"처럼 보이는 성능을 보이지만, 이는 실제로 강한 주기 패턴을 암기(memorization)하는 것에 가깝다. Stein의 역설 유비를 통해, 글로벌 TSF FM들의 "평균적 일반화"가 개별 과업의 최적 성능을 희생하는 결과를 낳을 수 있음을 지적한다 (§4.2, p.6).

**일반화 성능 향상을 위한 저자 권고:**

$$\text{진정한 일반화} \neq \text{주기성 데이터에서의 평균 성능 향상}$$

진정한 일반화를 위해 필요한 요소들:
- 구조적 변화(Changepoint)가 있는 데이터에서의 적응력
- 시간 변동 변동성(Heteroscedasticity) 환경에서의 불확실성 정량화
- 분포 드리프트(Concept Drift) 상황에서의 온라인 학습 능력
- 외생 변수 통합 능력

**내 해석 및 추가 관점:**

일반화 성능 향상 가능성은 다음 방향에서 탐색될 수 있다:

1. **메타-학습(Meta-Learning) 접근:** 데이터의 비정상성 유형을 자동 감지하여 적절한 모델을 선택하는 메타 프레임워크. 예: MAML 기반 TSF 적응형 모델.

2. **조건부 정규화(Conditional Normalization):** 입력 데이터의 통계적 특성(ACF 피크 강도, ADF 검정 결과 등)을 조건으로 한 적응형 정규화.

3. **비정상성 유형 인식 손실 함수:** 데이터의 비정상성 유형에 따라 가중치가 달라지는 손실 함수 설계.

$$\mathcal{L}_{\text{adaptive}} = \sum_{k} w_k(\text{type}) \cdot \text{MASE}_k$$

4. **앙상블 다양성 강화:** 주기성에 특화된 모델과 구조적 변화 대응 모델을 병렬로 구성하여 데이터 특성에 따라 동적 가중 결합.

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ 아래 비교는 제공된 PDF의 참고문헌 목록에서 확인 가능한 논문들 중심으로 작성되었습니다. 참고문헌에 없는 논문에 대한 외부 지식은 명확히 구분하여 표시합니다.

#### 논문이 직접 참조한 2020년 이후 주요 연구

| 논문 | 연도 | 핵심 주장 | 본 논문과의 관계 |
|------|------|-----------|-----------------|
| Zeng et al. [68] "Are Transformers Effective for TSF?" (AAAI 2023) | 2023 | DLinear가 트랜스포머 계열 모두를 능가 | **핵심 지지 선행연구** |
| Tan et al. [58] "Are Language Models Actually Useful for TSF?" | 2024 | LLM 구성 요소 제거해도 성능 유지 또는 향상 | **핵심 지지 선행연구** |
| Merrill et al. [46] "Language Models Still Struggle to Zero-Shot Reason about Time Series" | 2024 | GPT-4가 원인 추론에서 랜덤 수준 | **지지** |
| Xu et al. [67] "Specialized Foundation Models Struggle to Beat Supervised Baselines" (ICLR 2025) | 2025 | FM들이 지도학습 기준선을 능가하지 못함 | **핵심 지지 선행연구** |
| Lin et al. [37] SparseTSF (ICML 2024) | 2024 | <1k 파라미터로 SOTA 달성 | **벤치마크 편향의 구체적 증거** |
| Aksu et al. [1] GIFT-Eval | 2024 | 일반화된 TSF 모델 평가를 위한 표준 벤치마크 | **본 논문이 권장하는 방향** |
| Qiu et al. [53] TFB | 2024 | 포괄적·공정한 TSF 벤치마크 | **본 논문 방향과 정렬** |

#### 본 논문이 앞으로의 연구에 미치는 영향

**긍정적 영향:**
1. **벤치마크 설계 재고 촉진:** 새 TSF 논문들이 고전 기준선 포함을 표준 관행으로 채택하도록 압력
2. **평가 지표 다양화:** MASE, OWA 등 scale-free 지표 사용 확대
3. **비주기성 데이터 수요 창출:** 구조적 변화, 금융 시계열, 의료 데이터 등 다양한 특성의 벤치마크 개발 동기 부여
4. **재현성 문화 강화:** 코드 공개, 데이터 분할 고정 등을 표준 요건으로 요구하는 리뷰 관행 형성

**한계로 인한 영향 제약:**
1. 논문 자체가 NeurIPS에서 Reject 받아 공식 출판 영향력이 제한될 수 있음
2. 구체적인 대안 벤치마크나 방법론을 제시하지 않아, 실천적 변화를 이끌기 어려울 수 있음

#### 앞으로 연구 시 고려할 점

1. **데이터 특성 명시 의무화:**
   - 제안 모델의 전제 조건(어떤 비정상성 유형에서 유효한가)을 명확히 기술
   - ADF 검정, KPSS 검정 결과, ACF/PACF 분석 등 데이터 특성 보고 표준화

2. **기준선 완전성 체크리스트:**
   - 나이브 기준선 (랜덤워크, 계절 나이브)
   - 통계 기준선 (적절히 튜닝된 ARIMA, ETS)
   - 단순 ML 기준선 (DLinear, 단층 MLP)
   - 도메인 SOTA (날씨: GraphCast, 금융: GARCH 계열)

3. **통계적 유의성 검정 의무화:**
   - Diebold-Mariano 검정 등 예측 비교 전용 검정 적용
   - 다중 비교 보정(Bonferroni, Holm 등)
   - 다중 데이터 분할(walk-forward validation)

4. **계산 비용 대비 효과(cost-benefit) 보고:**
   - FLOPs, 학습 시간, 추론 시간 대비 성능 향상 보고
   - 탄소 배출량 보고 (ML 지속가능성 관점)

5. **비정상성 유형별 세분화 평가:**
   - 주기성 강도에 따른 성능 프로파일 (ACF peak 강도 기반)
   - 구조적 변화 존재 여부에 따른 별도 성능 보고

6. **재현성 인프라 활용:**
   - GIFT-Eval[1] 같은 표준화 플랫폼 사용 권장
   - 모든 실험 결과를 고정된 공개 데이터 분할로 보고

---

## 참고 자료

본 분석에 직접 참조된 자료:

- **주 논문:** Saqur, R., Bergmeir, C., Horvath, B., Schmidt, D., Rudzicz, F., & Lyons, T. (2026). *Seeking SOTA: Time-Series Forecasting Must Adopt Taxonomy-Specific Evaluation to Dispel Illusory Gains.* arXiv:2603.15506v1.

- **논문 내 핵심 참고문헌:**
  - [1] Aksu et al. "GIFT-Eval: A Benchmark for General Time Series Forecasting Model Evaluation." arXiv:2410.10393, 2024.
  - [37] Lin et al. "SparseTSF: Modeling Long-Term Time Series Forecasting with *1k* Parameters." ICML 2024.
  - [46] Merrill et al. "Language Models Still Struggle to Zero-Shot Reason about Time Series." arXiv:2404.11757, 2024.
  - [53] Qiu et al. "TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods." VLDB 2024.
  - [58] Tan et al. "Are Language Models Actually Useful for Time Series Forecasting?" arXiv:2406.16964, 2024.
  - [67] Xu et al. "Specialized Foundation Models Struggle to Beat Supervised Baselines." ICLR 2025.
  - [68] Zeng et al. "Are Transformers Effective for Time Series Forecasting?" AAAI 2023.
  - [73] Zhou et al. "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting." AAAI 2021.
  - [14] Efron & Morris. "Stein's Paradox in Statistics." Scientific American, 1977.
  - [23][24] Hewamalage et al. "Forecast Evaluation for Data Scientists: Common Pitfalls and Best Practices." DMKD, 2023.
  - [27] Hyndman & Athanasopoulos. *Forecasting: Principles and Practice.* OTexts, 3rd ed., 2021.
