# SWIFT: Mapping Sub-series with Wavelet Decomposition Improves Time Series Forecasting

> **📌 면책 고지**: 본 분석은 제공된 논문 PDF(arXiv:2501.16178v3)를 직접 분석한 결과입니다. 논문에 명시되지 않은 내용은 추정임을 명확히 표시합니다. 2020년 이후 최신 연구 비교는 제 학습 데이터(~2024년 초) 기반이며, 일부 최신 논문의 세부 수치는 확인되지 않을 수 있습니다.

---

## 1. Executive Summary (10문장 이내)

SWIFT(Sub-series with Wavelet Decomposition for Time Series Forecasting)는 자원 제약적 엣지 환경에서의 장기 시계열 예측(LTSF)을 위해 설계된 경량 모델이다.  
기존 Transformer 기반 모델들은 높은 예측 성능을 보이지만 연산 비용이 크고, 반대로 기존 경량 선형 모델들은 비정상(non-stationary) 시계열 처리에 취약하다는 이중 한계를 동시에 해결하고자 한다.  
SWIFT는 세 가지 핵심 기법을 결합한다:  
① Haar 웨이블릿 기반 1차 이산 웨이블릿 변환(DWT)을 통한 무손실 다운샘플링,  
② 학습 가능한 1D 합성곱 필터를 통한 크로스-밴드 정보 융합,  
③ 저주파·고주파 서브시리즈를 단일 공유 선형 계층 또는 얕은 MLP로 매핑하는 전략이다.  
7개 실제 데이터셋(ETT 4종, Weather, Electricity, Traffic)에 대한 실험에서 SWIFT는 SOTA 수준의 예측 성능을 달성했다.  
SWIFT-Linear의 파라미터 수는 18.1K로, 단일 선형 모델 대비 25% 수준이며 FITS의 15%에 불과하다.  
이상 탐지 태스크에서도 SMD 데이터셋에서 99.92% F1을 기록하며 범용성을 입증했다.  
웨이블릿 도메인 합성곱은 기존 시간 도메인 합성곱 대비 수용 필드(receptive field)를 지수적으로 확장하면서 파라미터는 선형적으로만 증가시킨다.  
공유 가중치 전략의 타당성은 코사인 유사도 및 선형 회귀 분석을 통해 이론적·실험적으로 검증되었다.  
SWIFT는 에너지 스케줄링, 지능형 교통 시스템 등 지연 민감 응용 분야에 즉시 배포 가능한 실용적 모델을 제시한다.

---

### 1-1. 연구의 목적과 필요성

**배경 문제 (Introduction, p.1-2)**

| 기존 모델 유형 | 장점 | 한계 |
|---|---|---|
| Transformer/LLM 기반 | 높은 예측 정확도, 장거리 의존성 포착 | 높은 연산 비용, 엣지 배포 불가 |
| 경량 선형 모델 (DLinear, FITS) | 빠른 추론, 적은 파라미터 | **비정상 시계열에서 성능 급락** |

> **비정상(Non-stationary) 시계열**: 시간에 따라 평균, 분산, 자기상관 구조가 변화하는 시계열. 실제 세계의 대부분의 시계열(주가, 교통량, 기상 등)이 이에 해당하며, FFT 기반 방법(FITS 등)은 신호가 정상(stationary)이라는 가정 하에 동작하므로 이런 데이터에 취약함.

**Figure 1 (p.2)**에서 합성 비정상 신호에 대한 예측 결과를 비교한 결과:
- SWIFT: MSE = 0.0129
- FITS: MSE = 0.0376 (약 2.9배 열등)
- iTransformer: MSE = 0.0384 (약 3.0배 열등)

**핵심 필요성**: 엣지 디바이스에서 실시간 동작 가능한 초경량 모델이면서도, 비정상 시계열을 효과적으로 처리할 수 있는 모델의 부재.

---

## 2. 핵심 주장과 근거 표

| # | 핵심 주장 | 근거 | 위치 |
|---|---|---|---|
| 1 | DWT는 시계열의 무손실 다운샘플링 방법으로 파라미터 효율을 4배 향상 | Haar DWT로 $T \to T/2$ 압축 후 성능 유지 실험 | p.2, Table 3 |
| 2 | 저주파·고주파 서브시리즈는 동일 표현 공간에서 매핑 가능 | 코사인 유사도($Sim_{s,l}$) 88~97%, LR MSE≈0 | p.8, Table 6 |
| 3 | 단일 공유 선형 계층이 분리 매핑과 동등한 성능 달성 | Share vs Split 절제 실험, IMP≤0.003 | p.8, Table 5 |
| 4 | 웨이블릿 도메인 합성곱은 지수적 수용 필드 확장을 선형 파라미터 증가로 달성 | $\ell$-level 분해 시 수용 필드 $2^\ell \cdot k$, 파라미터 $\ell \cdot 2 \cdot k^2$ | p.5, Figure 3 |
| 5 | SWIFT는 SOTA 대비 100배 이상 경량이면서 동등 또는 우월한 예측 성능 | Table 1, 2, 3의 포괄적 비교 | p.7, Table 1-3 |
| 6 | Haar 웨이블릿이 다른 웨이블릿(DB2, Sym4, Coif) 대비 최적 성능 | 웨이블릿 절제 실험 | p.15, Table 10 |
| 7 | 채널 독립(Not CI) 전략이 채널 혼합(CI)보다 대부분 우수 | CI 절제 실험 | p.14, Table 9 |

---

### 2-1. 상세 설명

#### 📌 해결하고자 하는 문제

1. **비정상 시계열 처리 실패**: FFT 기반 모델(FITS)은 신호의 정상성(stationarity)을 가정하여 시변(time-varying) 특성을 포착하지 못함
2. **파라미터 효율과 성능의 트레이드오프**: 고성능 모델은 수억 개의 파라미터를 요구하여 엣지 디바이스에 배포 불가
3. **선형 모델의 과소적합**: 단일 선형 계층의 표현력 한계로 복잡한 패턴 학습에 실패

---

#### 📌 제안하는 방법 (수식 포함)

**[Step 1] 인스턴스 정규화 (p.4)**

$$\tilde{\mathcal{X}} = \mathcal{X} - \bar{\mathcal{X}}$$

$$\hat{\mathcal{Y}} = f(\tilde{\mathcal{X}}) + \bar{\mathcal{X}}$$

> - $\mathcal{X}$: 입력 시계열
> - $\tilde{\mathcal{X}}$: 정규화된 시계열
> - $\bar{\mathcal{X}}$: 시계열의 평균값
> - $f(\cdot)$: 모델 함수
> - $\hat{\mathcal{Y}}$: 최종 예측 출력

ReVIN 방식의 학습 가능한 정규화도 지원:

$$\mathcal{X} = \gamma \left( \frac{\mathcal{X} - \mathbb{E}_t[\mathcal{X}]}{\sqrt{\text{Var}[\mathcal{X}] + \epsilon}} \right) + \beta$$

> - $\gamma, \beta$: 학습 가능한 스케일/시프트 파라미터
> - $\epsilon$: 수치 안정성을 위한 소수

> **인스턴스 정규화(Instance Normalization)**: 각 샘플(시퀀스) 개별적으로 평균과 분산을 계산하여 정규화하는 기법. 배치 정규화(Batch Normalization)와 달리 배치 크기에 독립적으로 동작하며, 시계열의 분포 이동(distribution shift) 문제를 완화함.

---

**[Step 2] DWT 분해 (p.3-4)**

Haar 웨이블릿의 스케일 함수와 웨이블릿 함수의 재귀 관계:

$$\phi(t) = \sum_n h_\phi[n] \sqrt{2} \phi(2t - n)$$

$$\psi(t) = \sum_n h_\psi[n] \sqrt{2} \phi(2t - n)$$

> - $\phi(t)$: 스케일 함수(저주파 성분 표현)
> - $\psi(t)$: 웨이블릿 함수(고주파 성분 표현)
> - $h_\phi[n]$: 저역통과 필터 계수
> - $h_\psi[n]$: 고역통과 필터 계수

근사 계수와 세부 계수의 재귀 공식:

$$W_\phi[j, k] = h_\phi[-n] * W_\phi[j+1, n]$$

$$W_\psi[j, k] = h_\psi[-n] * W_\phi[j+1, n]$$

> - $W_\phi[j, k]$: $j$차 분해의 근사 계수(저주파)
> - $W_\psi[j, k]$: $j$차 분해의 세부 계수(고주파)
> - $j$: 웨이블릿 분해 차수
> - $k$: 시간 도메인에서의 이동(shift) 인덱스

Haar 웨이블릿 필터 (1차 분해, $j=1$):

$$h_\phi[n] = \{1/\sqrt{2},\ 1/\sqrt{2}\}$$

$$h_\psi[n] = \{1/\sqrt{2},\ -1/\sqrt{2}\}$$

입력 시계열에 DWT 적용:

$$\mathcal{Y}_\mathcal{L}, \mathcal{Y}_\mathcal{H} = \text{DWT}(\mathbf{X}) $$

> - $\mathbf{X} \in \mathbb{R}^{N \times T}$: 입력 시계열 ($N$: 변수 수, $T$: 과거 윈도우 길이)
> - $\mathcal{Y}_\mathcal{L} \in \mathbb{R}^{N \times T/2}$: 근사 계수 (저주파 성분, 추세와 완만한 변화 포착)
> - $\mathcal{Y}_\mathcal{H} \in \mathbb{R}^{N \times T/2}$: 세부 계수 (고주파 성분, 급격한 변화와 노이즈 포착)

> **DWT(이산 웨이블릿 변환, Discrete Wavelet Transform)**: 신호를 저주파(근사)와 고주파(세부) 성분으로 분리하는 수학적 변환. FFT와 달리 시간-주파수 국소화(time-frequency localization) 특성을 가져 비정상 신호 분석에 적합함.

---

**[Step 3] 서브시리즈 연결 (p.4)**

$$\mathbf{Y} = [\mathcal{Y}_\mathcal{L}; \mathcal{Y}_\mathcal{H}] \in \mathbb{R}^{N \times 2 \times T/2} $$

> - 저주파·고주파 성분을 새로운 차원으로 연결하여 시간-주파수 표현을 구성

---

**[Step 4] 학습 가능한 합성곱 필터 (p.5)**

$$\mathbf{Y_C} = \text{Conv}(\mathbf{Y}) + \mathbf{Y} $$

> - $\mathbf{Y_C}$: 합성곱 후 잔차 연결된 출력
> - $\text{Conv}(\cdot)$: 입력 채널 2, 출력 채널 2의 1D 합성곱 레이어
> - 잔차 연결(residual connection)로 원본 정보 보존

웨이블릿 도메인 합성곱의 수용 필드 확장:
- $\ell$-level 분해, 커널 크기 $k$: 수용 필드 $= 2^\ell \cdot k$
- 파라미터 수 $= \ell \cdot 2 \cdot k^2$ (선형 증가)
- 기존 방법: 파라미터 수 $\propto$ (수용 필드) $^2$ (이차 증가)

---

**[Step 5] 공유 서브시리즈 매핑 (p.5)**

$$\mathbf{Y'} = \mathbf{Y_C} \mathbf{W} + \mathbf{b} $$

> - $\mathbf{W} \in \mathbb{R}^{T/2 \times T'/2}$: 공유 가중치 행렬 (저·고주파 성분 모두에 적용)
> - $\mathbf{b} \in \mathbb{R}^{T'/2}$: 편향 벡터
> - $\mathbf{Y'} \in \mathbb{R}^{N \times 2 \times T'/2}$: 매핑 결과
> - $T'$: 예측 구간 길이

---

**[Step 6] IDWT로 최종 예측 복원 (p.5)**

$$\mathbf{Y'_L} = \mathbf{Y'}_{:,0,:}, \quad \mathbf{Y'_H} = \mathbf{Y'}_{:,1,:}$$

$$\hat{\mathbf{Y}} = \text{IDWT}(\mathbf{Y'_L}, \mathbf{Y'_H}), \quad \hat{\mathbf{Y}} \in \mathbb{R}^{N \times T'} $$

> - $\mathbf{Y'_L}$: 예측된 저주파 계수
> - $\mathbf{Y'_H}$: 예측된 고주파 계수
> - $\hat{\mathbf{Y}}$: 최종 시계열 예측값
> - IDWT: 역 이산 웨이블릿 변환 (웨이블릿 도메인 → 시간 도메인 복원)

---

**[공유 가중치 분석 수식] (p.8)**

코사인 유사도:

$$Sim_{a,b} = \frac{\sum_{i=1}^n a_i b_i}{\sqrt{\sum_{i=1}^n a_i^2} \sqrt{\sum_{i=1}^n b_i^2}}$$

선형 회귀 관계:

$$W_s = \beta_l W_l + \beta_h W_h + \epsilon$$

> - $W_s$: Share 전략의 공유 가중치 행렬
> - $W_l$: Split 전략의 저주파 성분용 가중치 행렬
> - $W_h$: Split 전략의 고주파 성분용 가중치 행렬
> - $\beta_l, \beta_h$: 선형 회귀 계수
> - $\epsilon$: 잔차 (실험에서 MSE ≈ 0)

---

#### 📌 모델 구조 (Figure 2, p.3)

```
입력 X ∈ R^{T×N}
    ↓
[인스턴스 정규화]
    ↓
[DWT (Haar, 1차)]
    ↓─────────────────────────────┐
 Y_L ∈ R^{N×T/2}    Y_H ∈ R^{N×T/2}
    └──────────── 연결 ────────────┘
         Y ∈ R^{N×2×T/2}
              ↓
    [Conv1D Block (잔차 연결)]
              ↓
    [공유 Linear 또는 MLP]
    (채널 독립 또는 비독립)
              ↓
    Y' ∈ R^{N×2×T'/2}
              ↓
    [IDWT 복원]
              ↓
    [역 인스턴스 정규화]
              ↓
    출력 Ŷ ∈ R^{T'×N}
```

---

#### 📌 성능 향상

**파라미터 효율 (Table 3, p.7)**

| 모델 | 파라미터 수 | MACs | 학습 시간/에폭 |
|---|---|---|---|
| DLinear | 138.4K | 44.61M | 19.062s |
| FITS | 116.2K | 1189.91M | 25.070s |
| CycleNet/Linear | 123.7K | 22.42M | 28.268s |
| **SWIFT/Linear** | **18.1K** | **11.09M** | **18.571s** |
| SWIFT/MLP | 53.1K | 33.53M | 19.717s |

> **MACs(Multiply-Accumulate Operations)**: 곱셈-누산 연산 횟수로 모델의 실제 연산량을 측정하는 지표. 파라미터 수와 별개로 추론 속도에 직접 영향을 줌.

**예측 성능 (Table 1-2, p.7)**
- ETT 데이터셋 4종에서 SWIFT/Linear가 최다 데이터셋에서 1위 또는 2위
- Weather: SWIFT/MLP MSE=0.216 (전체 평균), 최고 성능
- Traffic: SWIFT/MLP 평균 MSE=0.394, 최고 성능

---

#### 📌 한계

1. **SMAP/MSL 이상 탐지 성능 저조**: 이진 이벤트 데이터에서 시간-주파수 표현의 한계 (Table 8, p.14)
2. **1차 Haar 분해만 사용**: 다중 해상도 정보 활용 미흡 (저자 스스로 후속 연구 과제로 제시)
3. **비교 기준 제한**: Solar-Energy 등 일부 데이터셋은 결과만 언급되고 표에 미포함
4. **단일 랜덤 시드**: STD가 5e-4 미만이나 통계적 유의성 검증 부족

---

## 3. 각 주장에 페이지/Figure/Table 번호 표시

| 주장 | 근거 위치 |
|---|---|
| DWT가 비정상 시계열 처리에 우수 | Figure 1 (p.2), Table 4 (p.8) |
| SWIFT-Linear 파라미터 = FITS의 15% | Table 3 (p.7) |
| 공유 가중치 전략 유효성 | Table 5 (p.8), Table 6 (p.8), Figure 4 (p.6) |
| Haar 웨이블릿이 최적 | Table 10 (p.15) |
| 채널 독립 전략이 우수 | Table 9 (p.14) |
| Conv 레이어 필수성 | Table 4 (p.8) |
| 이상 탐지 성능 | Table 8 (p.14) |
| MLP가 고차원 데이터에 유리 | Table 2 (p.7), Appendix A (p.11) |
| LTSF 문제 정의 | Section 3.1 (p.3) |
| DWT 수식 | Section 3.2 (p.3), Section 4.2 (p.4-5) |

---

## 4. 저자 보고 결과 vs. 해석 분리

### 4-1. 저자가 직접 보고한 결과

**연구 주제**: 웨이블릿 분해 기반 경량 시계열 예측 모델 (p.1)

**방법**: DWT + Conv1D + 공유 선형/MLP + IDWT (Section 4, p.4-6)

**성능**:
- SWIFT-Linear 파라미터: 18.1K (FITS의 15.6%, DLinear의 13.1%)
- ETTh1 평균 MSE: 0.403 (FITS 0.407 대비 개선)
- Weather 평균 MSE: SWIFT/MLP 0.216 (FITS 0.218 대비 개선)
- SMD 이상 탐지 F1: 99.92%
- 공유 가중치 $W_s$와 $W_l$의 코사인 유사도: 88.2~97.3%

### 4-2. 검토자(필자)의 해석

> ⚠️ 이하는 논문 내용을 바탕으로 한 비판적 해석으로, 저자의 공식 주장이 아닙니다.

**긍정적 측면**:
- 웨이블릿 도메인에서의 공유 매핑 전략은 파라미터 정규화 효과를 가지며, 구조적 위험 최소화(SRM) 관점에서 일반화 성능 향상에 이론적 근거가 있음
- Haar 웨이블릿 선택은 연산 효율과 경계 효과(edge effect) 최소화 면에서 실용적으로 타당

**비판적 측면**:
- **성능 개선폭이 미미**: ETT 데이터셋에서 SWIFT/Linear 대 최고 기준 모델의 MSE 차이는 0.000~0.015 수준으로 실용적 유의성 불명확
- **선택적 비교**: 일부 최신 모델(TimeMixer, TSMixer 등)이 비교에서 제외됨
- **단일 시드 실험**: 재현성 및 통계적 신뢰도 제한적
- **합성 비정상 데이터셋의 대표성**: Figure 1의 실험이 논문에서 자체 생성한 데이터라 실제 비정상성을 완전히 대표하지 못할 수 있음

---

## 5. 통계적 취약 부분 및 비교 불가능한 수치 ⚠️

| 항목 | 문제 | 위치 |
|---|---|---|
| STD = 0.000 (실제 < 5e-4) | 소수점 3자리 반올림으로 실제 분산 숨겨짐, 통계적 유의성 검증 없음 | Table 1, 2 각주 |
| 단일 랜덤 시드(seed=2023) | 결과의 재현성 및 분산 추정 불가 | Table 9, 10 각주 |
| Figure 1의 합성 데이터 | 저자가 직접 생성한 데이터로 편향 가능성 존재 | p.2, Appendix F |
| 이상 탐지 F1 비교 | 일부 모델(DGHL, OCSVM 등)은 LTSF 모델이 아닌 전용 이상 탐지 모델로 직접 비교 부적절 | Table 8 |
| MACs 비교 기준 불일치 가능성 | 룩백 윈도우=720 고정 조건에서만 비교하여 다른 설정에서의 효율 일반화 불확실 | Table 3 |
| Weather 개선폭 | SWIFT/MLP 0.216 vs FITS 0.218: 차이 0.002, 실용적 유의성 불명확 | Table 2 |
| Traffic Not CI vs CI | Not CI 성능이 더 좋다고 하나, 채널 수(862)가 많은 경우 CI의 이점이 왜 나타나지 않는지 설명 불충분 | Table 9 |

---

## 6. 논문이 답하지 않는 질문

| # | 미해결 질문 |
|---|---|
| 1 | **다중 해상도 DWT 성능**: 1차 분해만 사용했는데, 2차 이상 분해 시 성능 변화는? (후속 연구로만 언급) |
| 2 | **장기 의존성의 한계**: 극도로 긴 시퀀스(T > 2000)에서의 성능 저하 여부 |
| 3 | **도메인 외 일반화**: 의료 시계열, 금융 고빈도 데이터 등 새로운 도메인에서의 성능 |
| 4 | **모델 선택 기준**: Linear vs MLP 선택의 명확한 정량적 기준 부재 (100채널 기준이라고 하나 모호) |
| 5 | **Haar 우위의 이론적 설명**: 실험적으로 Haar가 우수함을 보였으나 수학적 증명 부재 |
| 6 | **분포 이동(distribution shift) 대응**: ReVIN 외 고급 정규화 기법과의 비교 없음 |
| 7 | **실제 엣지 디바이스 벤치마크**: MCU, Raspberry Pi 등 실제 하드웨어에서의 추론 시간 미측정 |
| 8 | **전이 학습 가능성**: 사전 훈련-파인튜닝 패러다임에서의 SWIFT 적용 가능성 |
| 9 | **비정상성의 정도별 성능**: 다양한 수준의 비정상성을 가진 데이터에서의 체계적 실험 부재 |
| 10 | **SMAP/MSL 성능 저하의 근본 원인**: 시간-주파수 표현이 이진 이벤트에 왜 부적합한지 이론적 분석 부재 |

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p.2) — 합성 비정상 신호 예측 비교

**해석**: 3개 모델의 예측 시작점(96번째 타임스텝)부터의 예측 결과를 시각화.
- SWIFT(MSE=0.0129): 진폭 변화와 위상이 ground truth에 근접하게 추적
- FITS(MSE=0.0376): 고주파 성분의 위상 오차 누적, 비정상적 진폭 변화 추적 실패
- iTransformer(MSE=0.0384): 전체적인 패턴 포착 실패, 평탄화 경향

**의의**: SWIFT의 DWT 기반 접근이 FFT 기반(FITS) 및 attention 기반(iTransformer) 대비 비정상 신호에서 구조적으로 우수함을 직관적으로 보여주는 핵심 동기 그림.

> ⚠️ **통계적 주의**: 단일 합성 데이터셋, 단일 예시로 일반화 위험 있음.

---

### Figure 2 (p.3) — SWIFT 전체 구조도

**해석**: SWIFT의 데이터 흐름을 완전히 도식화.
- **좌측 경로**: 입력 → 인스턴스 정규화 → DWT → Y_L, Y_H 연결 → Conv1D 블록 → Linear/MLP (채널 독립 또는 비독립) → IDWT → 역 정규화 → 출력
- **핵심 혁신**: DWT로 $T \to T/2$ 압축 후 2×(T/2) 형태로 재구성하여 동일 공유 레이어 적용
- 채널 독립(Channel Independent) 경로: N개 채널 각각에 동일 가중치 적용 → 파라미터 공유 극대화

**의의**: 논문의 세 가지 핵심 기여(DWT, Conv1D, 공유 매핑)가 어떻게 하나의 파이프라인으로 결합되는지 이해하는 데 필수적인 그림.

---

### Figure 3 (p.5) — 웨이블릿 도메인에서의 합성곱 수용 필드

**해석**: 1차 DWT 후 합성곱의 수용 필드 확장 메커니즘을 도식화.
- 원본 신호 $x_0, x_1, ..., x_7$ (8개 포인트)
- DWT 후: 저주파 $a_0, a_1, a_2, a_3$과 고주파 $b_0, b_1, b_2, b_3$ (각 4개)
- 커널 크기 2의 합성곱이 웨이블릿 도메인에서 적용될 때, 원본 신호 기준 수용 필드 = 4 (커널 크기의 2배)
- 수식: $\ell$-level 분해 시 수용 필드 $= 2^\ell \cdot k$

**의의**: SWIFT가 적은 파라미터로 넓은 시간적 컨텍스트를 포착할 수 있는 핵심 이유를 기하학적으로 설명하는 그림.

---

### Figure 4 (p.6) — 가중치 맵 시각화 (ECL 데이터셋)

**해석**: ECL 데이터셋에서 학습된 세 가중치 행렬 $W_s$, $W_l$, $W_h$의 히트맵.
- $W_s$ (공유 전략): 전체적으로 균일한 패턴 (왼쪽)
- $W_l$ (분리 전략, 저주파): $W_s$와 매우 유사한 패턴 — 코사인 유사도 97.3% (Table 6)
- $W_h$ (분리 전략, 고주파): 오른쪽 하단에 집중된 다른 패턴 — $W_l$과 낮은 유사도

**의의**: 저주파 성분이 예측에 지배적 역할을 한다는 것을 시각적으로 확인. 공유 가중치 전략이 사실상 저주파 패턴을 중심으로 학습되며, 고주파 정보를 보조적으로 통합함을 보여줌.

---

### Figure 5/6/7 (p.13) — DWT 분해 시각화 (Traffic, Weather, ETT)

**해석** (세 그림을 하나로 묶어 해석):

각 그림은 3개 패널로 구성:
1. **입력 원본 시계열** (전체 길이 T)
2. **근사 계수 $Y_L$** (저주파, 길이 T/2): 전체적 추세와 주기적 패턴이 명확히 보존
3. **세부 계수 $Y_H$** (고주파, 길이 T/2): 국소적 변동과 노이즈 포착

**데이터셋별 특성**:
- **Traffic (Figure 5)**: 고주파 성분에 뚜렷한 일주기 패턴이 존재
- **Weather (Figure 6)**: 저주파 성분이 계절성을 잘 보존, 고주파 성분은 미세 기상 변동
- **ETT (Figure 7)**: 저주파 성분의 장기 추세가 명확, 고주파에 다양한 스케일의 변동

**의의**: DWT가 실제 다양한 도메인의 시계열에서 의미 있는 시간-주파수 분해를 수행함을 시각적으로 검증. SWIFT의 설계 가정(저·고주파 성분이 미래 성분을 예측하는 데 상호 보완적)의 직관적 근거 제공.

---

## 8. 결론 및 후속 연구

### 8-1. 저자 제시 시사점 및 후속 연구 계획 (p.8)

**시사점**:
1. DWT 기반 무손실 다운샘플링이 파라미터 효율과 예측 성능을 동시에 달성하는 핵심 메커니즘
2. 웨이블릿 도메인에서 저·고주파 성분은 공유 표현 공간에서 매핑 가능 (새로운 발견)
3. 엣지 컴퓨팅 환경에서 고성능 예측 AI의 민주화 가능성 제시

**저자 후속 연구 계획**:
- 이상 탐지, 분류 등 다양한 시계열 태스크로 확장
- 시간-주파수 도메인의 대규모 신경망 탐색 및 SWIFT 스케일업
- 다중 해상도 웨이블릿 변환 적용으로 멀티스케일 정보 활용

---

### 8-1. 모델 일반화 성능 향상 가능성 (중점)

**현재 일반화 근거**:

SWIFT의 공유 가중치 전략은 **구조적 위험 최소화(Structural Risk Minimization, SRM)** 관점에서 가설 공간(hypothesis space)을 제약하는 암묵적 정규화 효과를 가짐 (p.7).

> **구조적 위험 최소화(SRM)**: 경험적 오류(training error)와 모델 복잡도(model capacity) 사이의 균형을 최적화하는 학습 이론 원칙. 더 단순한 모델(낮은 VC 차원)이 같은 훈련 오류라면 더 나은 일반화 성능을 보장함.

이론적 근거(Appendix A, p.11):

$$|y_{t+1} - \text{MLP}_\theta(\mathbf{y}_{t:t-k})| \leq \epsilon + \mathcal{O}(\lambda^k) $$

> - $\epsilon$: MLP의 범용 근사 오차 (Universal Approximation Theorem으로 임의 소)
> - $\lambda^k$: 잠재 상태의 영향 감쇠 항 ($\lambda < 1$, $k$ 증가 시 지수적 감소)
> - 이 부등식은 충분히 큰 룩백 윈도우 $k$에서 MLP가 결합 시스템을 효과적으로 근사할 수 있음을 보장

**일반화 향상을 위한 추가 방향** (필자 제안):

1. **적응형 웨이블릿 학습**: 고정 Haar 필터 대신 데이터에 최적화된 학습 가능한 웨이블릿 필터 (AdaWaveNet 방향)

2. **메타 학습(Meta-Learning) 통합**: MAML 등을 통해 새로운 시계열 도메인에 빠르게 적응하는 능력 부여

3. **앙상블 다중 해상도**: 다양한 차수의 DWT 결과를 앙상블하여 다양한 시간 스케일의 패턴을 포착

4. **도메인 불변 표현 학습**: 웨이블릿 계수 공간에서 도메인 어댑테이션(Domain Adaptation) 적용

5. **데이터 증강(Wavelet Augmentation)**: 웨이블릿 계수 치환/혼합을 통한 훈련 데이터 다양화 (Arabi et al., 2024가 제안한 Wave-Mask/Mix 방향)

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 이하 비교는 논문 본문의 참고문헌과 필자의 학습 데이터(~2024년 초)를 기반으로 함. 일부 논문의 정확한 수치는 원본 확인 필요.

#### 주요 관련 연구 계보

```
DLinear (Zeng et al., 2023)
    ├─ "단순 선형 모델도 Transformer 능가" 발견
    │
FITS (Xu et al., 2024)
    ├─ 주파수 도메인 보간, 10K 파라미터
    │   └─ 한계: FFT의 정상성 가정
    │
CycleNet (Lin et al., 2024)
    ├─ 주기적 패턴의 명시적 모델링
    │
SWIFT (Xie & Cao, 2025) ← 본 논문
    ├─ DWT + 공유 매핑
    │
AdaWaveNet (Yu et al., 2025)
    └─ 학습 가능한 리프팅 기반 웨이블릿
```

#### 2020년 이후 주요 논문 비교

| 논문 | 연도 | 핵심 방법 | 파라미터 규모 | SWIFT 대비 |
|---|---|---|---|---|
| Informer (Zhou et al.) | 2021 | ProbSparse Attention | 수백만 | 100배+ 초과 |
| Autoformer (Wu et al.) | 2021 | Auto-Correlation + 분해 | 수백만 | 100배+ 초과 |
| FEDformer (Zhou et al.) | 2022 | 주파수 강화 Attention | 수백만 | 100배+ 초과 |
| DLinear (Zeng et al.) | 2023 | 단순 선형 분해 | ~138K | 7.6배 초과 |
| PatchTST (Nie et al.) | 2023 | 패치 + Transformer | ~수백만 | 100배+ 초과 |
| iTransformer (Liu et al.) | 2024 | 역전된 Transformer | ~수백만 | 100배+ 초과 |
| FITS (Xu et al.) | 2024 | FFT 보간 | ~116K | 6.4배 초과 |
| CycleNet (Lin et al.) | 2024 | 주기 모델링 | ~124K | 6.8배 초과 |
| WPMixer (Murad et al.) | 2024 | 웨이블릿 패치 혼합 | 미확인 | 비교 없음⚠️ |
| AdaWaveNet (Yu et al.) | 2025 | 학습 가능 웨이블릿 | 미확인 | 비교 없음⚠️ |
| **SWIFT** | **2025** | **DWT + 공유 선형** | **18.1K** | **기준** |

> ⚠️ WPMixer, AdaWaveNet의 정확한 파라미터 수는 논문에 직접 명시되지 않아 확인 불가.

#### SWIFT의 연구사적 위치와 영향

**기여**:
1. "웨이블릿 계수의 공유 매핑 가능성"이라는 새로운 발견은 향후 경량 시계열 모델 설계의 원칙적 기반이 될 수 있음
2. 엣지 컴퓨팅에서의 시계열 예측 연구를 위한 새로운 벤치마크 제시 (18.1K 파라미터)
3. 웨이블릿 도메인에서의 합성곱이 시간-주파수 분석과 딥러닝을 통합하는 효과적 방법임을 실증

**향후 연구 시 고려할 점**:

1. **다중 해상도 웨이블릿 분해**: SWIFT는 1차 분해만 사용. 계층적 DWT와 딥러닝의 통합이 미해결 과제
2. **학습 가능한 웨이블릿**: 고정 Haar 필터의 한계를 극복하기 위한 데이터 적응형 웨이블릿 설계
3. **비정상성의 정량적 측정**: 어느 정도의 비정상성에서 DWT 기반이 FFT 기반을 압도하는지 체계적 연구 필요
4. **멀티모달 시계열**: 텍스트, 이미지 등 외부 정보와 웨이블릿 표현의 결합
5. **인과성 보존**: 예측 시 미래 정보 누출(data leakage) 방지를 위한 웨이블릿 적용 방식 검토
6. **하드웨어 최적화**: Haar 웨이블릿의 연산을 하드웨어 가속(FPGA, NPU)에 최적화하는 구현 연구
7. **통계적 엄밀성**: 다중 랜덤 시드, 부트스트랩 신뢰구간 등 통계적 검증 강화

---

## 참고 자료

### 논문 본문 내 인용 (직접 분석)
- **주 논문**: Xie, W. & Cao, F. (2026). "SWIFT: Mapping Sub-series with Wavelet Decomposition Improves Time Series Forecasting." arXiv:2501.16178v3.

### 논문에서 인용된 주요 참고문헌
- Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). "Are Transformers Effective for Time Series Forecasting?" AAAI 2023.
- Xu, Z., Zeng, A., & Xu, Q. (2024). "FITS: Modeling Time Series with 10k Parameters." ICLR 2024.
- Nie, Y. et al. (2023). "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers." ICLR 2023.
- Liu, Y. et al. (2024). "iTransformer: Inverted Transformers are Effective for Time Series Forecasting." ICLR 2024.
- Lin, S. et al. (2024). "CycleNet: Enhancing Time Series Forecasting through Modeling Periodic Patterns." arXiv:2409.18479.
- Kim, T. et al. (2021). "Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift." ICLR 2021.
- Yu, H., Guo, P., & Sano, A. (2025). "AdaWaveNet: Adaptive Wavelet Network for Time Series Analysis." arXiv:2405.11124.
- Murad, M. M. N., Aktukmak, M., & Yilmaz, Y. (2024). "WPMixer: Efficient Multi-Resolution Mixing for Long-Term Time Series Forecasting." arXiv:2412.17176.
- Hornik, K., Stinchcombe, M., & White, H. (1989). "Multilayer Feedforward Networks are Universal Approximators." Neural Networks.
- Takens, F. (1981). "Detecting Strange Attractors in Turbulence." Lecture Notes in Mathematics, Springer.
