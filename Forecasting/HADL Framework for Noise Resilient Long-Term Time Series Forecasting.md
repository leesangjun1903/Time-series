# HADL Framework for Noise Resilient Long-Term Time Series Forecasting

> **⚠️ 사전 안내**: 본 논문은 2025년 2월 arXiv 프리프린트(arXiv:2502.10569v1)로, 아직 동료 심사(peer review)를 거치지 않은 논문입니다. 이 점을 감안하여 결과 해석에 주의가 필요합니다.

---

## 1. Executive Summary (10문장 이내)

HADL(Haar-Approximation Discrete cosine transform Low-rank)은 장기 시계열 예측(LTSF)에서 **노이즈 내성, 경량 설계, 예측 정확도**를 동시에 달성하는 프레임워크다.  
실세계 금융·에너지·경제 시계열 데이터는 높은 수준의 시간적 노이즈를 포함하며, 기존 모델들은 이를 체계적으로 다루지 못했다.  
HADL은 Haar 웨이블릿 기반 DWT로 입력 시퀀스를 절반으로 압축하며 노이즈를 제거하고, 이어서 DCT로 주파수 도메인에서 장기 패턴을 추출한다.  
최종 예측은 저랭크(low-rank) 선형 레이어 단 하나로 수행되어, 역변환 없이 시간 도메인 출력이 가능하다.  
실험 결과, HADL은 17K~50K 파라미터만으로 PatchTST(706K~5184K), FrNet(116K~7486K) 등 대형 모델과 동등하거나 우수한 MSE를 달성한다.  
노이즈 내성 테스트에서 ETTh1, ETTh2 데이터셋 기준 최저 MAV(각각 0.013, 0.008)를 기록하며 가장 안정적인 성능을 보였다.  
반면, 고차원 Traffic 데이터셋에서는 Transformer 계열 모델 대비 열세를 보이는 한계도 확인되었다.  
L1 정규화와 채널 혼합(channel mixing) 배제 전략이 과적합 억제에 기여한다.  
결론적으로 HADL은 노이즈가 많은 실세계 환경에서 경량·고효율 예측 모델의 유력한 대안이다.

---

### 1-1. 연구의 목적과 필요성

**목적**: 장기 시계열 예측(LTSF)에서 ① 노이즈 내성, ② 경량 설계, ③ 높은 예측 정확도를 **통합적으로** 해결하는 프레임워크 제안

**필요성 (Section 1, p.1)**:

| 문제 | 설명 |
|------|------|
| 시간적 노이즈 | 금융·에너지 데이터의 변동성으로 인해 기저 패턴 식별이 어려움 |
| 룩백 윈도우 딜레마 | 짧으면 정보 부족, 길면 노이즈 누적 |
| 대형 모델의 비효율 | Transformer 계열은 파라미터·메모리 비용이 크고 노이즈에 취약 |
| 기존 경량 모델의 한계 | DFT 기반 모델(FITS 등)은 균일한 컷오프 주파수 가정으로 정확도 손실 |
| 선행 연구 공백 | 노이즈 내성 + 경량화 + 정확도를 동시에 달성한 LTSF 모델 부재 |

> 💡 **LTSF(Long-Term Time Series Forecasting)**: 수십~수백 스텝 이후의 미래 값을 예측하는 작업으로, 단기 예측보다 노이즈와 장기 패턴 포착이 어려움

---

## 2. 핵심 주장과 근거 표

| 핵심 주장 | 방법론적 근거 | 실험적 근거 | 위치 |
|-----------|-------------|------------|------|
| DWT로 노이즈 감소 및 입력 압축 | Haar 필터로 근사 계수만 유지, 세부 계수(노이즈) 폐기 | Table 4: 파라미터 20~40% 감소, MSE 동등 이상 | p.4, Table 4 |
| DCT로 장기 패턴 추출 | 에너지 집약(energy compaction)으로 코사인 성분에 신호 집중 | Table 9: DCT 적용 시 MSE 0.362 vs 미적용 0.371 (ETTh1, H=96) | p.9, Table 9 |
| 저랭크 레이어로 일반화 향상 | 가중치 행렬 $W \approx PQ$ 분해로 과적합 억제 | Table 5: Low-Rank MSE 0.362 vs Standard 0.442 (ETTh1, H=96) | p.8, Table 5 |
| 노이즈 내성 최고 수준 | DWT+DCT 전처리로 노이즈 영향 사전 차단 | Table 3: ETTh1 MAV=0.013, ETTh2 MAV=0.008 (최저) | p.7, Table 3 |
| 경쟁적 예측 정확도 | 단일 저랭크 레이어로 경량화 | Table 1: ETTm2 전 horizon 최고 또는 2위 MSE | p.7, Table 1 |
| Traffic 고차원에서 한계 | 채널 수 862개의 복잡한 상관관계를 단일 레이어로 포착 불가 | Table 1: Improvement −0.045~−0.048 (음수) | p.6, Table 1 |

---

## 2-1. 상세 설명

### 해결하고자 하는 문제

**(Section 1, p.1~2)**

기존 LTSF 모델들이 공통적으로 직면하는 세 가지 문제:

1. **노이즈-룩백 딜레마**: 룩백 윈도우 확장 시 노이즈 누적으로 정확도 개선이 미미
2. **경량화-성능 트레이드오프**: 경량 MLP 모델들도 충분히 큰 룩백 창 요구
3. **통합 솔루션 부재**: 노이즈 내성 + 경량화 + 정확도를 동시에 달성한 모델 없음

---

### 제안하는 방법 (수식 포함)

#### Step 1: 이산 웨이블릿 변환 (DWT) — Eq. (1a), (2), (3)

$$A_T, \_ = \text{DWT}(X_T) $$

$$A_T = X_T * \left[\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}\right] $$

$$D_T = X_T * \left[-\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}\right] $$

| 기호 | 의미 |
|------|------|
| $X_T \in \mathbb{R}^{C \times L}$ | 입력 다변량 시계열 (채널 수 $C$, 룩백 길이 $L$) |
| $A_T \in \mathbb{R}^{C \times L/2}$ | 근사 계수 (저주파, 신호 핵심) — **유지** |
| $D_T \in \mathbb{R}^{C \times L/2}$ | 세부 계수 (고주파, 주로 노이즈) — **폐기** |
| $*$ | 컨볼루션 연산 |
| $\left[\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}\right]$ | Haar 근사 필터 (평균 연산) |
| $\left[-\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}\right]$ | Haar 세부 필터 (차분 연산) |

> 💡 **Haar 웨이블릿**: 가장 단순한 웨이블릿 함수로, 인접한 두 값의 평균과 차이를 계산. 평균(근사 계수)은 신호의 전반적 추세를, 차이(세부 계수)는 급격한 변화(노이즈 포함)를 나타냄

> 💡 **DWT(이산 웨이블릿 변환)**: 신호를 시간-주파수 두 도메인에서 동시에 분석하는 기법. 짧은 시간에 발생하는 주파수 변화를 포착 가능하며, STFT와 달리 웨이블릿 함수의 직교성으로 정보 중복이 없음

---

#### Step 2: 이산 코사인 변환 (DCT) — Eq. (1b), (4)

$$A_F = \frac{2}{L} \times \text{DCT}(A_T) $$

| 기호 | 의미 |
|------|------|
| $A_F \in \mathbb{R}^{C \times L/2}$ | 주파수 도메인 표현 |
| $\frac{2}{L}$ | 길이 기반 정규화 상수 (스케일 독립성·수치 안정성 보장) |
| $\text{DCT}(\cdot)$ | DCT Type II 변환 |

> 💡 **DCT(이산 코사인 변환)**: 신호를 코사인 함수의 합으로 표현하는 주파수 변환. DFT와 달리 실수 성분만 사용하며, 에너지 집약(energy compaction) 특성으로 적은 계수로 신호의 핵심 정보를 표현 가능

> 💡 **에너지 집약(Energy Compaction)**: 신호 에너지의 대부분이 소수의 저주파 계수에 집중되는 DCT의 특성. 이를 통해 노이즈(주로 고주파)와 신호(저주파)를 효과적으로 분리

---

#### Step 3: 저랭크 레이어 — Eq. (1c), (5)~(6)

$$\hat{Y}_T = A_F P Q + B $$

$$\dot{P}(t) = -\nabla_P \mathcal{L}(P(t)Q(t)) $$

$$\dot{Q}(t) = -\nabla_Q \mathcal{L}(P(t)Q(t)) $$

| 기호 | 의미 |
|------|------|
| $\hat{Y}_T \in \mathbb{R}^{C \times H}$ | 예측값 (예측 길이 $H$) |
| $P \in \mathbb{R}^{L/2 \times r}$ | 저랭크 인수 행렬 1 |
| $Q \in \mathbb{R}^{r \times H}$ | 저랭크 인수 행렬 2 |
| $r$ | 랭크 ( $r \ll \min(L/2, H)$ ), 실험에서 $r=40$ 사용 |
| $B \in \mathbb{R}^H$ | 편향 벡터 |
| $W \approx PQ$ | 전체 가중치 행렬의 저랭크 근사 |
| $\mathcal{L}$ | 손실 함수 |
| $\nabla_{P,Q}$ | $P, Q$에 대한 기울기 |

> 💡 **저랭크 근사(Low-Rank Approximation)**: 고차원 행렬 $W$를 두 작은 행렬 $P \times Q$의 곱으로 표현하는 기법. 파라미터 수를 $L/2 \times H$에서 $(L/2 + H) \times r$로 대폭 감소시킴. LoRA 등에서 차용된 개념

---

#### Parseval's Theorem (역변환 생략 근거) — Eq. (7)

$$\sum_{n=0}^{N-1} |(A_T)_n|^2 = \frac{1}{N}\sum_{n=0}^{N-1} |(A_F)_n|^2 $$

> 💡 **Parseval's 정리**: 시간 도메인과 주파수 도메인에서 신호의 총 에너지가 보존됨을 보장하는 수학 정리. HADL은 이를 근거로 역DCT 없이 주파수 도메인에서 직접 예측 수행

---

#### 노이즈 주입 및 평가 지표 — Eq. (8)~(11)

$$X_{\text{train}} := X_{\text{train}} + \mathcal{N}(0,1) \cdot \eta $$

$$\text{Imp.} = \text{MSE}_{\text{baseline}} - \text{MSE}_{\text{ours}} $$

$$\text{NRR} = \frac{\text{MSE}_{\eta=x}}{\text{MSE}_{\eta=0.0}} $$

$$\text{MAV} = \frac{1}{n}\sum_{i=1}^{n}|\text{NRR}_i - 1| $$

| 기호 | 의미 |
|------|------|
| $\mathcal{N}(0,1)$ | 표준 정규 분포에서 샘플링한 노이즈 |
| $\eta \in \{0.0, 0.3, 0.7, 1.7, 2.3\}$ | 노이즈 강도 |
| NRR | 노이즈 내성 상대지표 (1에 가까울수록 노이즈 영향 작음) |
| MAV | NRR의 평균 절대 변동성 (낮을수록 노이즈 내성 높음) |

---

### 모델 구조 (Figure 1, p.3)

```
입력 X_T ∈ R^{C×L}
    ↓
[Haar DWT 분해]
    ↓ (근사 계수 유지, 세부 계수 폐기)
A_T ∈ R^{C×L/2}
    ↓
[DCT Type-II 변환 + 정규화 (2/L)]
    ↓
A_F ∈ R^{C×L/2}
    ↓
[단일 저랭크 레이어 (PQ+B)]  ← 유일한 학습 파라미터
    ↓
Ŷ_T ∈ R^{C×H}  (역변환 없이 직접 예측)
```

**설계 원칙**:
- 채널 혼합(channel mixing) **배제**: 채널 간 노이즈 전파 방지
- 채널별 개별 레이어 대신 **단일 공유 레이어**: 과적합 억제
- L1 정규화: 불필요한 계수 수축

---

### 성능 향상 및 한계

| 구분 | 내용 |
|------|------|
| **성능 향상** | ETTh2 전 horizon MSE 개선(+0.001~+0.003) |
| **성능 향상** | FLOPs 0.004~0.011B (PatchTST 0.336~0.479B 대비 ~100배 감소) |
| **성능 향상** | 파라미터 17.6K~49.5K (FrNet 116K~7486K 대비 대폭 감소) |
| **노이즈 내성** | ETTh1 MAV=0.013, ETTh2 MAV=0.008 (최저) |
| **한계** | Traffic 고차원(862채널)에서 −0.045~−0.048 음수 개선 |
| **한계** | 짧은 룩백 창(L<192)에서 DWT 압축 후 정보 손실 |
| **한계** | ETTm2 노이즈 내성에서 PatchTST(0.014)에 뒤처짐(0.022) |

---

## 3. 각 주장의 근거 위치

| 주장 | 위치 |
|------|------|
| 노이즈가 LTSF 성능을 저해함 | p.1, Section 1 |
| DWT가 입력을 절반 압축하며 노이즈 제거 | p.3, Eq.(1a)(2)(3); p.4, Section 3.1 |
| DCT의 에너지 집약 장점 | p.4, Section 3.2, Eq.(4) |
| 저랭크 레이어의 일반화 효과 | p.5, Section 3.3, Eq.(5)(6); Table 5 |
| Parseval 정리로 역변환 생략 정당화 | p.5, Section 3.4, Eq.(7) |
| 멀티변량 예측 결과 | p.7, Table 1; p.16, Table 10 |
| 파라미터/FLOPs 비교 | p.7, Table 2 |
| 노이즈 내성 실험 | p.7, Table 3; p.18, Table 12 |
| Haar 분해 효과 검증 | p.8, Table 4; Appendix C.1, Table 7 |
| 저랭크 레이어 효과 검증 | p.8, Table 5; Figure 3 |
| DCT 효과 검증 | p.9, Figure 2; Table 9 |
| Traffic에서의 한계 | p.6, Section 5; Table 1 (Imp. 열) |

---

## 4. 저자 보고 결과 vs. 해석 분리

### 저자가 직접 보고한 결과

**연구 주제** (Abstract, p.1):
> "We propose a novel framework that addresses these challenges by integrating the Discrete Wavelet Transform (DWT) and Discrete Cosine Transform (DCT) to perform noise reduction and extract robust long-term features."

**방법** (Section 3, p.3):
$$A_{T,-} = \text{DWT}(X_T), \quad A_F = \frac{2}{L} \times \text{DCT}(A_T), \quad \hat{Y}_T = \text{LowRankLayer}(A_F)$$

**결과** (Section 5, p.6~7):
- ETTh2: "+0.001 to +0.003 MSE improvement across all horizons with only 17–50K parameters"
- ETTh1 MAV=0.013, ETTh2 MAV=0.008 (최저 노이즈 변동성)
- Traffic: "negative improvements (−0.045 to −0.048)"
- FLOPs: "0.004–0.011B" (경쟁 모델 대비 압도적 경량)

---

### 필자(리뷰어)의 해석

1. **ETTm1 결과의 해석**: Table 1에서 ETTm1 일부 horizon(H=96,192)에서 HADL이 DLinear, PatchTST에 뒤처지는데(−0.012~−0.005), 이는 ETTm1이 15분 단위 고해상도 데이터로 단기 주기 패턴이 강해 DCT 기반 장기 패턴 추출의 이점이 제한적이기 때문으로 보임

2. **저랭크의 실질적 기여**: Table 5에서 표준 선형 레이어 대비 MSE가 ETTh1 H=96 기준 0.442→0.362로 18% 감소했는데, 이는 단순 파라미터 감소 효과를 넘어 정규화 효과(implicit regularization)에 의한 것으로 해석됨

3. **노이즈 주입 실험의 한계**: 노이즈를 훈련 데이터에만 주입하고 검증·테스트 세트는 변경하지 않는 설계는 실제 테스트 시 노이즈 내성을 평가하는 것이 아닌, 노이즈 훈련 후 클린 데이터 예측 능력을 측정하는 것임 → 실제 운용 환경에서의 노이즈 내성과는 다를 수 있음

4. **Traffic 열세 원인**: 이 논문은 채널 혼합을 배제하는데, Traffic(862채널)은 센서 간 공간적 상관관계가 핵심인 데이터셋으로, 채널 독립 가정이 근본적으로 불리함

---

## 5. 통계적 취약점 및 비교 불가능한 수치

### ⚠️ 통계적 취약점

| 항목 | 문제점 |
|------|--------|
| **단일 실험 결과** | 각 실험에 대한 평균±표준편차 미보고. 랜덤 시드에 따른 변동성 불명확 |
| **노이즈 단일 샘플** | Eq.(8)의 노이즈 $\mathcal{N}(0,1)\cdot\eta$가 매 실험마다 다른 랜덤 샘플. 노이즈 주입의 재현성 불명확 |
| **ETTm2 H=720 DCT 비교** | Table 9에서 w/DCT=0.440, w/o DCT=0.438로 DCT **없는 것**이 오히려 낮음. 저자는 이를 명시적으로 논의하지 않음 ⚠️ |
| **ETTh2 H=336 저랭크 비교** | Table 5에서 Low-Rank=0.364, Standard=0.361로 표준 레이어가 더 낮음. 저자 설명 없음 ⚠️ |

### ⚠️ 비교 불가능한 수치

| 항목 | 이유 |
|------|------|
| **룩백 창 고정 L=512** | 일부 베이스라인 모델의 공식 논문은 서로 다른 최적 L을 사용하므로, L=512 고정이 일부 모델에게 불리할 수 있음 |
| **Traffic vs. ETT 성능** | Traffic(862채널)과 ETT(7채널)은 채널 수 차이가 100배 이상으로 직접 비교 불가 |
| **Robustness 노이즈 범위** | Table 3의 헤더는 η={0.0, 0.3, 0.7, 1.3, 1.7}인데 본문(Section 4)에서는 η={0.0, 0.3, 0.7, 1.7, 2.3}으로 불일치 ⚠️ |
| **SparseTSF와의 공정한 비교** | SparseTSF는 1K 파라미터 초경량 설계로, 17~50K의 HADL과 설계 목적 자체가 다름 |

---

## 6. 논문이 답하지 않는 질문

| 번호 | 미해결 질문 |
|------|------------|
| Q1 | **최적 랭크 $r$ 선택 기준**: Table 8에서 $r$=40을 기본값으로 사용하나, 데이터셋별 최적 $r$ 자동 선택 방법이 없음 |
| Q2 | **다중 DWT 분해 수준**: 초장기 룩백(L>720) 시 여러 레벨의 Haar 분해를 어떻게 통합하는지 미제시 |
| Q3 | **세부 계수(Detail Coefficient) 활용**: $D_T$를 완전히 폐기하는데, 이것이 항상 노이즈인지에 대한 이론적 검증 부재 |
| Q4 | **Weather, Electricity 데이터셋**: 주요 결과표(Table 1)에서 제외되고 Appendix(Table 10)에만 존재. 이 데이터셋에서의 성능이 왜 주요 분석에서 누락되었는지 불명확 |
| Q5 | **비선형 패턴 대응**: DCT+저랭크는 본질적으로 선형 변환 체계. 고도로 비선형적인 시계열(예: 주식)에서의 한계 미논의 |
| Q6 | **L1 정규화 강도 선택**: 정규화 계수(lambda) 값과 선택 기준 미공개 |
| Q7 | **채널 독립 가정의 타당성**: 7채널(ETT)에서는 효과적이나, 중간 규모(21~321채널)에서의 적정성 미검증 |
| Q8 | **실시간 스트리밍 적용**: 온라인 학습이나 슬라이딩 윈도우 환경에서의 계산 효율성 미분석 |

---

## 7. 가장 중요한 그림/표 5개 해석

### [1] Figure 1 (p.3) — HADL 아키텍처 전체 흐름도

**해석**: 좌측부터 입력(L) → Haar 분해(L/2) → DCT(L/2) → 저랭크 레이어(L/2 → H) → 출력(H) 순서로 구성. 핵심은 **처리 과정에서 채널(C) 차원은 독립적으로 유지**되며, 유일한 학습 파라미터가 저랭크 행렬 $P, Q$뿐이라는 점. 역변환 단계가 없어 구조가 매우 단순함. 이는 모델의 해석 가능성(interpretability)을 높이는 동시에 파라미터 효율성을 극대화하는 설계 철학을 시각적으로 보여줌

---

### [2] Table 1 (p.7) — 멀티변량 예측 성능 비교

**해석**: HADL(w/ L1)이 ETTh1·ETTh2에서 Transformer 계열과 동등하거나 우수한 MSE를 달성함. 특히 ETTh2 H=96에서 MSE=0.271로 FrNet(0.269)과 거의 동일하면서 파라미터 수는 약 8배 적음(17.6K vs. 116K). **그러나 Traffic에서의 Imp.=-0.045~-0.048은 채널 독립 가정의 구조적 한계를 명확히 드러냄.** 한편 SparseTSF(1K 파라미터)가 HADL(17~50K)보다 대체로 성능이 낮은 것은 극단적 경량화의 비용을 보여줌

---

### [3] Table 3 (p.7) — 노이즈 내성 테스트 (NRR/MAV)

**해석**: HADL(w/o 정규화)이 ETTh1(MAV=0.013), ETTh2(MAV=0.008)에서 모든 모델 중 최저 MAV 달성. NRR이 η<1.3에서 1.000~1.016으로 안정적으로 유지되는 반면, iTransformer는 Traffic에서 η=1.7시 NRR=1.692로 급격히 악화. **주목할 점**: 정규화 미적용(w/o) 버전이 적용(w/) 버전보다 대체로 낮은 MAV를 보임 → DWT+DCT 자체의 노이즈 내성이 L1 정규화보다 더 강력하다는 증거. ETTm2에서 HADL(0.022)이 PatchTST(0.014)에 뒤처지는 이유는 ETTm2의 15분 고해상도 데이터 특성상 단일 DWT 레벨이 충분하지 않을 수 있음

---

### [4] Figure 2 (p.9) — DCT 적용/미적용 저랭크 행렬 히트맵

**해석**: DCT 적용 시(a) 저랭크 행렬이 **주기적이고 구조적인 패턴**(물결 모양)을 보임. 반면 DCT 미적용 시(b) 행렬이 **더 균일하고 단순한** 패턴을 가짐. 이는 DCT가 신호의 구조적 특성을 저랭크 행렬에 효과적으로 인코딩함을 시사. 두 경우 모두 시계열의 핵심 성분을 포착하지만, DCT 적용 시 더 명확한 주파수 특성이 시각화됨. MSE 비교(0.362 vs 0.371)는 통계적으로 미미한 차이이지만, 가중치 구조의 질적 차이를 보여줌

---

### [5] Figure 3 (p.15) — 저랭크 vs. 표준 선형 행렬 히트맵

**해석**: 저랭크 행렬(a)은 **집중적이고 구조화된** 가중치 분포(특정 주파수 성분에 집중)를 보이는 반면, 표준 선형 행렬(b)은 **광범위하고 분산된** 가중치를 가짐. 표준 레이어의 넓은 가중치 분포는 노이즈 성분까지 학습하는 과적합 경향을 의미. 저랭크의 집중된 가중치는 가장 중요한 주파수 성분만 선택적으로 학습함을 보여줌. Table 5의 성능 개선(ETTh1 H=96: 0.442→0.362)이 이 구조적 차이에서 비롯됨을 확인

---

## 8. 결론 및 후속 연구

### 8-1. 저자 제시 시사점과 후속 연구 계획

**저자의 시사점** (Section 7, p.9):
- HADL은 노이즈 내성·경량성·정확도를 동시에 달성한 최초의 통합 LTSF 프레임워크
- 단일 저랭크 레이어가 역변환 없이도 시간 도메인 예측 가능함을 실증

**저자가 인정한 한계 및 향후 방향**:

| 한계 | 향후 방향 |
|------|-----------|
| 짧은 룩백 창 성능 저하 | 최적 룩백 창 자동 탐색 |
| 초장기 룩백에서 단일 DWT 레벨 부족 | 다중 Haar 분해 레벨 통합 |
| 세부 계수 폐기로 정밀 정보 손실 | 선택적 세부 계수 활용 전략 |

---

### 8-1. 모델의 일반화 성능 향상 가능성 (중점 분석)

현재 HADL의 일반화 메커니즘은 세 층위에서 작동함:

**1. DWT를 통한 구조적 정규화**
Haar 분해 자체가 입력 노이즈를 물리적으로 제거하여 학습 데이터의 품질을 높임. 이는 **데이터 수준의 정규화**로 볼 수 있음

**2. 저랭크 레이어의 암묵적 정규화**
$$W \approx PQ, \quad r \ll \min(L/2, H)$$
이 분해는 가중치 공간을 저차원 다양체(manifold)로 제한하여 과적합을 억제. 이론적으로 Schotthöfer et al. (2022, [31])의 저랭크 복권 이론과 연결됨

**3. L1 정규화**
불필요한 주파수 계수의 계수를 0으로 수렴시켜 희소한(sparse) 주파수 표현 학습

**일반화 향상을 위한 추가 가능성**:

- **적응형 랭크 선택**: AdaLoRA([23]) 방식으로 데이터셋별 최적 $r$ 자동 탐색 → 다양한 데이터셋에서의 일반화 향상 기대
- **다중 DWT 레벨 통합**: 다양한 주파수 해상도에서 정보 추출로 데이터 특성 다양성 대응
- **채널 어댑터**: 고차원 데이터셋을 위한 경량 채널 상관관계 학습 모듈 추가 (Traffic 한계 극복)
- **메타 학습(Meta-Learning)**: 다양한 도메인의 시계열에서 공통 패턴을 사전 학습하여 새로운 데이터셋에 빠른 적응

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ 아래 비교는 논문에 인용된 문헌 정보와 공개된 연구 정보를 바탕으로 하며, 직접 접근하지 않은 논문에 대해서는 논문 내 인용 정보만을 기반으로 기술함

| 모델 | 연도 | 주요 접근 | 파라미터 규모 | HADL 대비 특징 |
|------|------|-----------|--------------|----------------|
| **DLinear** [7] | 2023 | 추세-계절성 분해 + 선형 레이어 | ~1K~8K | 간단하나 노이즈 처리 없음 |
| **PatchTST** [8] | 2022 | Transformer + 시계열 패치 | 706K~5184K | 정확도 높으나 고파라미터 |
| **iTransformer** [9] | 2023 | 역전된 Transformer (특성-패치) | ~948K~3789K | 채널 상관관계 포착 강점 |
| **FITS** [11] | 2023 | DFT + 선형 보간 | ~5K | 10K 파라미터 경량, 균일 컷오프 한계 |
| **SparseTSF** [12] | 2024 | 희소 크로스-피어리어드 | ~1K | 초경량이나 복잡 패턴 약함 |
| **FrNet** [10] | 2024 | 주파수 기반 회전 네트워크 | 116K~7486K | 주파수 처리 강점, 고파라미터 |
| **FreTS** [15] | 2024 | 주파수 도메인 MLP | ~16K~17K | 주파수 처리 유사, 노이즈 처리 취약 |
| **ModernTCN** [16] | 2024 | 현대적 순수 CNN 구조 | 930K~6042K | 시간적 패턴 강점, 고파라미터 |
| **HADL** (본 논문) | 2025 | DWT+DCT+저랭크 | 17K~50K | 노이즈 내성 + 경량화 통합 |

**HADL이 연구에 미치는 영향**:

1. **노이즈-인식 전처리의 중요성 재조명**: 단순한 아키텍처 개선보다 입력 품질 향상이 성능에 크게 기여함을 실증

2. **저랭크 기법의 LTSF 확장**: NLP에서 검증된 LoRA 계열 기법이 시계열 예측에서도 유효함을 보임 → 향후 Transformer 기반 LTSF에 저랭크 어텐션 적용 연구 촉진 예상

3. **변환 도메인 앙상블의 효과**: DWT+DCT의 시간-주파수 복합 변환이 단독 변환 대비 우수함을 확인

**향후 연구 시 고려할 점**:

| 고려 사항 | 세부 내용 |
|-----------|-----------|
| **고차원 채널 대응** | 862채널 Traffic에서의 열세를 극복할 경량 채널 상관관계 모듈 필요 |
| **적응형 분해 레벨** | 데이터셋 특성에 따라 DWT 분해 레벨을 자동 조정하는 메커니즘 |
| **비정상(Non-stationary) 시계열** | 분포가 시간에 따라 변하는 데이터에서의 일반화 검증 필요 |
| **실제 노이즈 유형** | 가우시안 노이즈만 테스트됨 → 임펄스 노이즈, 계절적 노이즈 등 다양한 노이즈 유형 검증 |
| **엔드-투-엔드 학습 가능한 웨이블릿** | 고정 Haar 웨이블릿 대신 학습 가능한 웨이블릿 필터 탐구 |
| **불균일 샘플링** | 현재 모델은 규칙적 샘플링 가정 → 의료·금융 등 불규칙 데이터 대응 필요 |

---

## 참고 자료

**논문 원문**:
- Dey, A., Kusch, J., & Al Machot, F. (2025). *HADL Framework for Noise Resilient Long-Term Time Series Forecasting*. arXiv:2502.10569v1 [cs.LG].

**논문 내 인용 문헌 (주요)**:
- [7] Zeng et al. (2023). *Are Transformers Effective for Time Series Forecasting?* AAAI 2023.
- [8] Nie et al. (2022). *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*. arXiv:2211.14730.
- [9] Liu et al. (2023). *iTransformer: Inverted Transformers are Effective for Time Series Forecasting*. arXiv:2310.06625.
- [10] Zhang et al. (2024). *FrNet: Frequency-based Rotation Network for Long-term Time Series Forecasting*. KDD 2024.
- [11] Xu et al. (2023). *FITS: Modeling Time Series with 10k Parameters*. arXiv:2307.03756.
- [12] Lin et al. (2024). *SparseTSF: Modeling Long-term Time Series Forecasting with 1k Parameters*. arXiv:2405.00946.
- [15] Yi et al. (2024). *Frequency-domain MLPs are More Effective Learners in Time Series Forecasting*. NeurIPS 2024.
- [16] Luo & Wang (2024). *ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis*. ICLR 2024.
- [22] Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685.
- [26] Ahmed et al. (1974). *Discrete Cosine Transform*. IEEE Transactions on Computers.
- [27] Daubechies & Bates (1993). *Ten Lectures on Wavelets*.
- [31] Schotthöfer et al. (2022). *Low-rank Lottery Tickets: Finding Efficient Low-rank Neural Networks via Matrix Differential Equations*. NeurIPS 2022.

**코드 저장소**:
- https://github.com/forgee-master/HADL
