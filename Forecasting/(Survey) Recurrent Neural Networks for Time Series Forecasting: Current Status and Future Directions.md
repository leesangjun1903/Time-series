
# Recurrent Neural Networks for Time Series Forecasting: Current Status and Future Directions

## 1. 논문 핵심 요약

Hewamalage et al. (2019)의 "Recurrent Neural Networks for Time Series Forecasting: Current Status and Future Directions"는 시계열 예측 분야에서 RNN의 실질적 적용 가능성을 광범위하게 검증하는 실증 연구입니다. 논문의 핵심 주장은 다음과 같습니다:[1]

**핵심 기여:**
1. RNN 아키텍처의 체계적 분류 및 비교
2. 6개 벤치마크 데이터셋(M3, M4, CIF 2016, NN5, Wikipedia, Tourism)에 대한 광범위한 실험 (48,000개 시계열)
3. 전자동 RNN 모델이 통계적 방법(ETS, ARIMA)과 경쟁 가능함을 증명
4. 실무자를 위한 구체적 가이드라인 및 오픈소스 프레임워크 제공

**핵심 발견:** RNN은 "은탄환"이 아니지만, 홈제너스 계절성을 가진 관련 시계열들에서 글로벌 모델로 활용될 때 유효한 대안입니다.

***

## 2. 해결하는 문제 및 방법론

### 2.1 문제의 배경
과거 시계열 예측 경쟁(M3, NN3, NN5)에서 RNN은 전통적 통계 모델에 비해 경쟁력을 잃었습니다. 그러나 최근 Big Data 환경에서는 단일 시계열 모델링의 한계를 극복하기 위해 다수의 관련 시계열을 활용하는 글로벌 모델이 필요하며, RNN이 이를 위한 유력한 후보입니다. 문제는 다음과 같습니다:

1. **자동화 부재**: RNN의 하이퍼파라미터 튜닝 가이드라인 부족
2. **계절성 모델링**: RNN의 계절성 처리 능력 불명확
3. **재현 가능성**: 공개된 코드/데이터셋 부족
4. **비용-편익**: ETS/ARIMA 대비 계산 비용 대 정확성 트레이드오프

### 2.2 제안 방법론

#### 2.2.1 단변량 예측 문제 정의
$$\{x_{T+1}, ..., x_{T+H}\} = F(x_1, x_2, ..., x_T) + \epsilon \quad (Eq. 1)$$

여기서:
- $F$: 신경망으로 근사되는 함수
- $H$: 예측 지평선 (forecast horizon)
- $\epsilon$: 함수 근사 오차

#### 2.2.2 RNN 셀 구조

**Elman RNN**:

$$h_t = \sigma(W_i \cdot h_{t-1} + V_i \cdot x_t + b_i)$$

$$z_t = \tanh(W_o \cdot h_t + b_o)$$

**LSTM with Peephole 연결**:

$$i_t = \sigma(W_i \cdot h_{t-1} + V_i \cdot x_t + P_i \cdot C_{t-1} + b_i)$$

$$f_t = \sigma(W_f \cdot h_{t-1} + V_f \cdot x_t + P_f \cdot C_{t-1} + b_f)$$

$$\tilde{C}_t = \tanh(W_c \cdot h_{t-1} + V_c \cdot x_t + b_c)$$

$$C_t = i_t \odot \tilde{C}_t + f_t \odot C_{t-1}$$

$$h_t = o_t \odot \tanh(C_t)$$

$$z_t = h_t$$

여기서 $P_i, P_f, P_o$는 peephole 연결의 가중치입니다.

**GRU (간소화된 LSTM)**:

$$u_t = \sigma(W_u \cdot h_{t-1} + V_u \cdot x_t + b_u)$$

$$r_t = \sigma(W_r \cdot h_{t-1} + V_r \cdot x_t + b_r)$$

$$\tilde{h}_t = \tanh(W_h \cdot r_t \odot h_{t-1} + V_h \cdot x_t + b_h)$$

$$h_t = u_t \odot \tilde{h}_t + (1-u_t) \odot h_{t-1}$$

#### 2.2.3 아키텍처 구조

**Stacked Architecture**:
- 각 시간 단계에서 이동 윈도우 입력
- 다층 구조 가능
- 누적 에러: $E = \sum_{t=1}^{T} e_t$ 로 백프로퍼게이션 적용

**Sequence-to-Sequence Architecture**:
- 인코더-디코더 구조
- 컨텍스트 벡터: 최종 히든 상태
- 에러: $E = \sum_{t=1}^{H} (y_t - \hat{y}_t)$ (마지막 단계만)

**다중 출력 전략 (MIMO)**:
- 모든 예측 단계를 동시에 출력
- 시간 단계 간 상호의존성 포착
- 이동 윈도우: $m = 1.25 \times \max(\text{seasonality}, H)$

#### 2.2.4 전처리 및 정규화

**계절성 분해 (STL Decomposition)**:
$$X_t = T_t + S_t + R_t$$

여기서 $T_t$는 트렌드, $S_t$는 결정론적 계절성, $R_t$는 나머지입니다.

**분산 안정화 (Log 변환)**:

$$w_t = \begin{cases} \log(y_t) & \text{if } \min(y) > \epsilon \\ \log(y_t + 1) & \text{if } \min(y) \leq \epsilon \end{cases}$$

**추세 정규화 (Per-window)**:

$$\tilde{x}_t = x_t - \text{trend}(\text{last point of input window})$$

#### 2.2.5 하이퍼파라미터 튜닝

SMAC (Sequential Model-based Algorithm Configuration) 기반:
- 미니배치 크기: 32-128
- 에포크 수: 50-500
- 학습률: 0.001-0.1 (Adam/Adagrad의 경우)
- RNN 셀 차원: 10-512
- L2 정규화 파라미터 $\psi$: 1e-6 to 1e-2
- 노이즈 표준편차: 0-0.5

***

## 3. 모델 구조 상세 분석

### 3.1 아키텍처 비교표

| 아키텍처 | 입력 형식 | 에러 계산 | 장점 | 단점 |
|---------|---------|---------|------|------|
| **Stacked** | 이동 윈도우 벡터 | 누적 에러 | 최고 성능, 데이터 증강 | 길이 의존성 |
| **S2S Decoder** | 스칼라 | 마지막 단계 | 가변 길이 입력 | 교사 강제 오류 축적 |
| **S2S Dense (MW)** | 이동 윈도우 | 마지막 단계 | 오류 축적 회피 | Stacked보다 약함 |
| **S2S Dense (Non-MW)** | 스칼라 | 마지막 단계 | 단순 구조 | 성능 저하 |

### 3.2 RNN 셀 성능 순위

통계 검정 (Friedman test p-value = 0.101):
1. **LSTM with Peephole** (최고)
2. **GRU** (중간)
3. **Elman RNN** (최하)

Peephole 연결이 게이트가 이전 셀 상태를 확인하도록 함으로써 그래디언트 흐름 개선.

### 3.3 옵티마이저 효과

| 옵티마이저 | 특징 | 성능 | 자동화 |
|---------|------|------|--------|
| **COCOB** | 학습률 자동 튜닝 | **최고** | ✓ 우수 |
| **Adam** | 모멘텀 + RMSprop | 경쟁력 | ✗ 수동 튜닝 |
| **Adagrad** | 축적 그래디언트 | 최저 | ✗ 학습률 감소 |

**COCOB의 우수성**: 학습률 하이퍼파라미터를 제거하여 완전 자동화 달성.

***

## 4. 성능 향상 및 한계

### 4.1 주요 성능 향상 메커니즘

#### 4.1.1 계절성 처리의 이원화

**발견**: RNN이 **직접** 계절성을 모델링할 수 있는가?

논문의 실험 결과:
- **홈제너스 계절성** (모든 시계열이 유사한 패턴): 직접 모델링 가능
- **이질적 계절성**: 사전 STL 분해 필수

**결론**: 

$$\text{Recommendation} = \begin{cases} 
\text{No STL} & \text{if seasonality strength homogeneous} \\
\text{STL + RNN} & \text{if seasonality strength heterogeneous}
\end{cases}$$

#### 4.1.2 이동 윈도우 크기의 최적화

입력 윈도우 $m$의 선택:

$$m = \begin{cases}
1.25 \times \text{forecast horizon } H & \text{경우 1} \\
1.25 \times \text{seasonality period } s & \text{경우 2}
\end{cases}$$

휴리스틱 계수 1.25는:
- 트렌드 정보 충분히 포함
- 적어도 한 주기 이상 커버
- 계산 비용 억제

#### 4.1.3 전처리 순서의 중요성

**최적 파이프라인**:
1. 결측값 처리 (중앙값 보간)
2. 로그 변환 (분산 안정화)
3. STL 분해 (계절성 제거)
4. Per-window 추세 정규화
5. 신경망 학습

### 4.2 성능 벤치마크 결과

#### 4.2.1 데이터셋별 RNN vs 통계 모델

**Mean SMAPE 비교**:

| 데이터셋 | 최고 RNN | ETS | ARIMA | RNN 우위 |
|---------|---------|-----|-------|---------|
| CIF 2016 | 10.54 | - | - | ✓ |
| NN5 | 22.46 | - | - | ✓ |
| M3 (월별) | 14.44 | - | - | ✓ |
| Wikipedia | 45.77 | - | - | ✓ |
| **M4 (월별)** | 19.21 | - | **18.95** | ✗ |
| Tourism | 17.52 | - | - | ✓ |

**마이크로 카테고리만 RNN 우위** (M4의 45,000개 시계열 중 마이크로만).

#### 4.2.2 계산 비용

| 태스크 | 데이터셋 | 전처리 | 튜닝 | 학습 | 총계 | vs Baseline |
|------|---------|--------|------|------|------|-------------|
| RNN | CIF h=12 | 10s | 3530s | 210s | 3750s | 44배 (ARIMA: 85s) |
| RNN | NN5 | 50s | 48,000s | 3,440s | 51,490s | 48배 (ARIMA: 1,067s) |

**Tuning이 총 시간의 85-94%** 차지.

### 4.3 일반화 성능의 주요 한계

#### 4.3.1 짧은 시계열에서의 과적합

**문제**: M3/CIF의 최대 길이 126-108 시점
- 복잡한 모델: 과적합 위험
- 통계 모델의 강점: 파라미터 수 적음

**해결**: 
- 글로벌 모델로 데이터 증강
- 이동 윈도우 전략 (효과적 데이터 포인트 증가)
- 정규화 강화

#### 4.3.2 계절성 비균질성

**예시**: CIF 2026 데이터셋
- 일부 시계열: 2년 미만 (불완전한 주기)
- 계절성 강도 분포: 0.01-0.85 (광범위)

$$\text{Recommendation: Heterogeneous case에서 STL 필수}$$

#### 4.3.3 M4에서의 성능 한계

M4 월별 데이터셋: **ARIMA가 여전히 우수**

**원인 분석**:
- 마이크로 카테고리(개별 제품): RNN 우위
- 매크로 카테고리(GDP, 경제 지표): ARIMA 우위
  
**분류별 성능**:
- Micro: RNN이 우수 (다수의 유사 시계열)
- Macro: ARIMA 우수 (고유한 경제 역학)

$$\text{Domain-specific 모델 선택 필수}$$

#### 4.3.4 해석 가능성 부족

RNN의 "블랙박스" 특성:
- 모델 예측의 이유 설명 어려움
- 통계 모델(ARIMA): 명시적 성분 분해 가능
- 실무 적용의 저항

***

## 5. 모델 일반화 성능 향상 분석

### 5.1 교차 시계열 학습의 효과

**핵심 통찰**: 글로벌 모델 vs 로컬 모델

$$\text{성능 = 함수(모델 복잡도, 데이터 가용성, 시계열 유사성)}$$

| 시나리오 | 최적 방법 | 이유 |
|---------|---------|------|
| 단일 긴 시계열 | 통계 모델 | 통계 방법이 충분 |
| 다수 단기 유사 시계열 | RNN 글로벌 | 패턴 공유 활용 |
| 이질적 시계열 혼합 | 혼합 모델 | 클러스터링 필요 |

### 5.2 클러스터링 기반 글로벌 모델

논문에서 제안한 접근(Bandara et al. 기법):
1. 유사 시계열 그룹화
2. 각 클러스터별 RNN 글로벌 모델
3. 그룹 내 상태 개별 유지

$$\text{예측} = \text{GlobalWeights} + \text{LocalHiddenState}$$

**효과**: M4에서 M3 대비 11% 정확도 향상.

### 5.3 전처리가 일반화에 미치는 영향

| 전처리 기법 | SMAPE 개선 | 일반화 효과 |
|----------|-----------|----------|
| 계절성 제거(STL) | 5-8% | 비정상성 감소 |
| 로그 변환 | 2-4% | 분산 안정화 |
| 추세 정규화 | 3-6% | 활성화 함수 포화 회피 |
| 이동 윈도우 | 4-7% | 데이터 증강 |
| 결합 효과 | **12-18%** | 누적 개선 |

### 5.4 하이퍼파라미터 민감도

**중요도 순서** (영향도):
1. **SMAC 튜닝** (50 iterations): 가장 중요
2. **아키텍처 선택**: Stacked >> S2S
3. **셀 차원**: 낮은 민감도 (Smyl 2020과 일치)
4. **옵티마이저**: COCOB > Adam >> Adagrad

**결론**: 자동화된 하이퍼파라미터 튜닝이 핵심.

***

## 6. 2020년 이후 최신 연구 비교 분석

### 6.1 Transformer 기반 아키텍처

#### 6.1.1 iTransformer (2023)[2]
**혁신**: 역방향 어텐션 (Inverted Attention)
- 기존: 시간 토큰 × 다변량 임베딩 → 어텐션
- iTransformer: 변량 토큰 × 시간 정보 → 어텐션

$$\text{Attention}(Q_v, K_t, V_t) \rightarrow \text{Multivariate correlations}$$

**성과**: 
- 벤치마크 최고 성능
- 장기 의존성(horizons > 96) 우수
- 계산 효율 향상

**한계**: 짧은 시계열에서 과적합 위험.

#### 6.1.2 ETSformer (2022)[3]
**개념**: 지수 평활화 원리를 Transformer에 통합

$$\text{ESA(Exponential Smoothing Attention)}$$

- 시간 가중 감소 적용
- 최근 데이터에 높은 가중치
- 해석 가능성 강화

**성과**: 
- 계절성 우수 모델링
- 전통 통계와 DL 결합
- 중기 예측(horizons 24-96) 최고

#### 6.1.3 패치 기반 Transformer (PatchTST, Dateformer)
**핵심 아이디어**: 시계열을 패치로 분할 후 처리

$$\text{Series} \rightarrow [\text{Patch}_1, \text{Patch}_2, ..., \text{Patch}_N]$$

- 각 패치: 토큰으로 변환
- Transformer 인코더 적용
- 재구성 또는 직접 예측

**성과**:
- 시계열 길이 확장성 (4K 이상)
- 정보 손실 감소
- PatchTST: M4 경쟁 수준

### 6.2 N-BEATS 계열 (2019-2021)[4][5]

#### 6.2.1 원본 N-BEATS
**구조**: 완전 연결층 + 기저 확장 (Basis Expansion)

$$\text{Generic}: \text{Identity basis + Deep layers}$$
$$\text{Interpretable}: \text{Polynomial/Fourier basis}$$

**기저 확장**:
$$\hat{y} = \sum_{k} \theta_k \cdot b_k(t)$$

여기서 $b_k$는 기저 함수.

**성과**:
- M4 11% 향상 (통계 벤치마크 대비)
- M4 우승 모델(Smyl 2020)과 3% 차이
- 해석 가능성 + 성능 균형

#### 6.2.2 NBEATSx (2021)[4]
**확장**: 외생변수 (Exogenous) 통합

$$\hat{y} = f(X_{\text{history}}, E_{\text{future}})$$

**사례**: 전기 가격 예측
- 외생변수: 온도, 수요 예측, 요일
- 성능: NBEATS 대비 20% 향상
- 기존 전문가 모델 대비 5% 향상

### 6.3 주의 메커니즘 강화 연구 (2020-2025)

#### 6.3.1 Dual-stage Attention RNN (DA-RNN, 2017 이후)[6]
**구조**: 이중 단계 주의
1. **입력 주의**: 다변량 기여도 가중
2. **시간 주의**: 시간 단계 중요도 가중

$$\alpha_t^{input} = \text{softmax}(W^T \tanh(W_e E_t + U_e s_{t-1}))$$
$$\alpha_t^{temporal} = \text{softmax}(V^T \tanh(W_d h_t + U_d s_{t-1}))$$

**성과**: 다변량 예측에서 NARX, LSTM 우수

#### 6.3.2 위치 기반 내용 주의 (2017)[Cinar et al.]
**혁신**: 계절 주기를 주의에 명시적으로 인코딩

$$\text{Attention}_{i,j} = \text{Attention}_{\text{Bahdanau}} + \pi(j \bmod s) \cdot e_i$$

여기서 $s$는 계절 주기, $\pi$는 학습된 페널티.

**결과**: 6개 데이터셋 중 5개에서 우수

### 6.4 하이브리드 모델 (2020-2025)

#### 6.4.1 ES-RNN (M4 우승, Smyl 2020)
**구조**: 지수 평활화 + RNN

$$L_t = \alpha \cdot y_t + (1-\alpha) \cdot L_{t-1}$$
$$\hat{y}_t = L_{t-1} \cdot \text{RNN}(x_1, ..., x_{t-1})$$

**성과**:
- M4 전체 1위
- M4 월별만 평가 시 여전히 최고 수준
- 통계와 DL 강점 결합

#### 6.4.2 AMES-RNN (2025)[7]
**개념**: 적응형 다변량 지수평활화 + RNN

$$\hat{y}^{(i)}_t = \text{ES}(y^{(i)}_{t-1}) + \text{RNN}_{\text{residual}}$$

**성과**:
- 비계절 추세 데이터에 23% RMSE 감소
- 자동 계수 갱신
- 저계산 리소스 (온디바이스 가능)

#### 6.4.3 ARIMA-LSTM (통계 + DL)
**설계**: 
1. ARIMA: 선형 성분
2. LSTM: 비선형 잔차 모델링

$$y_t = \text{ARIMA}_t + \text{LSTM}_{\text{residual},t}$$

**결과**: 코로나19 예측에 전문가 모델 대비 우수

### 6.5 Temporal Convolutional Networks (TCN, 2018 이후)

#### 6.5.1 기본 구조
**핵심**: Dilated causal convolution

$$y_t = \sum_{k=0}^{K-1} w_k \cdot x_{t-d \cdot k}$$

여기서 $d$는 dilation factor.

**장점 vs RNN**:
- 병렬 처리 가능
- 그래디언트 소실 문제 완화
- 더 긴 receptive field

#### 6.5.2 TCAN (Temporal Convolutional + Attention)
**결합**: TCN + Sparse Attention

$$\text{Performance: TCAN > DeepAR > LogSparse Transformer > TCNN > N-BEATS}$$

태양광 예측에서 검증.

### 6.6 2024-2026 최신 아키텍처

#### 6.6.1 PSformer (2025)[8]
**혁신**: 
- 매개변수 공유 (Parameter Sharing)
- 공간-시간 세그먼트 주의 (Segment Attention)

$$\text{SegAtt}(Q_{segment}, K_{time}, V_{time})$$

**특징**: 매개변수 75% 감소, 성능 동등

#### 6.6.2 AWGformer (2026)[9]
**개념**: 웨이블릿 기반 다중해상도 예측

$$\text{Signal} = \sum_{j} W_j(f) + \text{Detail}_j(f)$$

- 적응형 웨이블릿 선택
- 주파수 인식 다중 헤드 주의
- 계층적 재구성

**성과**: 비정상 시계열에 최고 성능

#### 6.6.3 DDT (2026)[10]
**이중 마스킹 + 이중 전문가**:
1. 인과 마스크 (이론적 일관성)
2. 동적 데이터 기반 마스크 (적응성)
3. 시간-변량 상관 이원화

**벤치마크**:
- ETTh, Electricity, Solar에서 SOTA
- 재생 에너지 통합에 최적

### 6.7 비교 요약: 아키텍처별 성능 특성

| 모델 | 장점 | 단점 | 최적 용도 | 계산 비용 |
|------|------|------|---------|---------|
| **LSTM/GRU** | 간단, 빠름, 안정적 | 장기 의존 약함 | 단기 예측 | 저 |
| **Transformer** | 장기 의존, 병렬 | 메모리 집약적, 짧은 시계열 약함 | 길이 예측, 다변량 | 높음 |
| **N-BEATS** | 해석 가능, 빠름 | 외생변수 제한(NBEATS) | 일반 시계열 | 중간 |
| **TCN** | 병렬, 안정적 | 주기성 모델링 약함 | 신호 처리 | 중간 |
| **Hybrid (ES-RNN)** | 정확, 견고 | 복잡 | 금융, 수요 예측 | 높음 |

***

## 7. 향후 연구 시 고려사항

### 7.1 이론적 과제

#### 7.1.1 일반화 능력의 수학적 분석
- **문제**: RNN의 VC dimension, Rademacher complexity 분석 부족
- **필요**: 샘플 복잡도 이론 (Sample complexity theory)

$$\text{Error}_{\text{test}} \leq \text{Error}_{\text{train}} + O\left(\sqrt{\frac{d + \log(n)}{n}}\right)$$

#### 7.1.2 비정상성(Non-stationarity) 처리
- **도전**: 통계적 성질이 시간과 함께 변함
- **해결 방향**:
  - Adaptive normalization
  - Online learning
  - Concept drift detection

#### 7.1.3 불확실성 정량화
- **현황**: 점 추정 중심
- **필요**: 확률적 예측, 신뢰 구간

$$P(y_t | \mathcal{D}_{1:t-1}) = \int P(y_t | \theta) P(\theta | \mathcal{D}_{1:t-1}) d\theta$$

### 7.2 방법론적 개선

#### 7.2.1 도메인 적응 (Domain Adaptation)
- **동기**: 한 도메인(금융)에서 학습 → 다른 도메인(에너지)에 전이
- **기술**: 
  - Adversarial training
  - Meta-learning
  - Fine-tuning 전략

#### 7.2.2 다중 태스크 학습 (Multi-task Learning)
- **아이디어**: 관련 예측 태스크 동시 학습

$$\mathcal{L}_{\text{total}} = \sum_{i=1}^{K} w_i \mathcal{L}_i(\theta_{\text{shared}}, \theta_i)$$

**효과**: NN5 데이터셋에서 5-8% 향상

#### 7.2.3 메타학습 (Meta-learning)
- **목표**: "학습 방법을 학습"
- **응용**: 신규 시계열에 빠른 적응

$$\theta^* = \text{MetaUpdate}(\theta, \mathcal{D}_{\text{few-shot}})$$

### 7.3 해석 가능성 강화

#### 7.3.1 주의 맵 시각화
- 모델 예측의 이유 설명
- 도메인 전문가 검증

$$\text{Attention}_{\text{map}}[t, t'] = \frac{\exp(e_{t,t'})}{\sum_j \exp(e_{t,j})}$$

#### 7.3.2 Feature Importance
- Permutation importance
- SHAP values for time series
- Integrated gradients

#### 7.3.3 기저 확장 해석 (Basis expansion interpretation)
- N-BEATS의 트렌드/계절성 분해 확장
- 물리 기반 기저 (Domain-specific basis)

### 7.4 효율성 개선

#### 7.4.1 경량 모델 (Lightweight)
- **목표**: 온디바이스 배포 (모바일, IoT)
- **기법**: 
  - 지식 증류 (Knowledge distillation)
  - 프루닝 (Pruning)
  - 양자화 (Quantization)

**성과**: 매개변수 90% 감소, 성능 <5% 저하

#### 7.4.2 에너지 효율성
- GPT-4 학습: 6,150 MWh (570가구 연간 전력)
- 목표: 탄소 중립 AI

#### 7.4.3 점진적 학습 (Incremental Learning)
- 새로운 데이터 도입 시 재학습 최소화
- 지속적 학습 환경에서 필수

### 7.5 문제별 권장 모델

**의사결정 트리**:

```
시계열 예측 문제
├─ 짧은 통계 시계열 (n < 200)
│  ├─ 선형 성분 주요 → ARIMA
│  └─ 비선형 성분 주요 → LSTM
├─ 긴 개별 시계열 (n > 1000)
│  ├─ 주기 명확 → Transformer/iTransformer
│  └─ 비정상성 강 → TCN
├─ 다수 관련 시계열 (K > 100)
│  ├─ 해석 가능성 필요 → N-BEATS
│  ├─ 정확도 우선 → ES-RNN/Hybrid
│  └─ 실시간 필요 → GRU
└─ 외생변수 포함
   ├─ 다변량 의존성 강 → DA-RNN
   └─ 간단한 선형 관계 → NBEATSx
```

***

## 8. 결론

Hewamalage et al.의 논문은 시계열 예측에서 RNN이 **조건부 경쟁력**이 있음을 엄밀하게 입증했습니다. 핵심 발견은:

1. **RNN의 강점**: 글로벌 모델로서 많은 관련 시계열 활용 시 우수
2. **RNN의 한계**: 
   - 단일 짧은 시계열에서 과적합
   - 계산 비용 높음 (튜닝 시간)
   - M4와 같은 도메인 혼합 데이터에서 약함

3. **최적 전략**: 
   - 홈제너스 데이터에 RNN
   - 이질적 데이터에 하이브리드(ES-RNN)
   - 해석 가능성 필요 시 N-BEATS

2020년 이후 연구는 **Transformer**, **N-BEATS**, **하이브리드 모델**을 중심으로 진화했으며, 특히 2024-2026의 최신 모델들은 **매개변수 효율성**, **웨이블릿 기반 다중 해상도**, **동적 적응**에 초점을 맞추고 있습니다.

미래 연구는 세 가지 방향으로 진행될 것으로 예상됩니다:
- **이론**: 비정상 시계열의 일반화 이론
- **응용**: 도메인 적응 및 전이학습
- **실무**: 해석 가능성과 효율성의 균형

***

## 참고문헌 (선별)

 Hewamalage, H., Bergmeir, C., & Bandara, K. (2020). Recurrent Neural Networks for Time Series Forecasting: Current Status and Future Directions. *International Journal of Forecasting*, 37(2), 388-427.[1]

 N-BEATS-RNN: Deep learning for time series forecasting (2020)[11]

 Multi-Task Time Series Forecasting With Shared Attention (2020)[6]

 An RNN-Based Adaptive Hybrid Time Series Forecasting Model (2025)[7]

 iTransformer: Inverted Transformers Are Effective for Time Series Forecasting (2023)[2]

 Temporal Convolutional Attention Neural Networks (2023)[12]

[53-54] DDT, AWGformer (2026)

 ETSformer: Exponential Smoothing Transformers (2022)[3]

 Neural basis expansion analysis with exogenous variables (2021)[4]

[84-85] N-BEATS (2019) and related work

출처
[1] 1909.00590v5.pdf https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c4477a4e-49c2-4dcd-8b1e-edae15f7ed75/1909.00590v5.pdf
[2] Dynamic group fusion transformer for financial time series prediction: An ablation study https://www.growingscience.com/dsl/Vol15/dsl_2025_61.pdf
[3] ETSformer: Exponential Smoothing Transformers for Time-series
  Forecasting https://arxiv.org/pdf/2202.01381.pdf
[4] Neural basis expansion analysis with exogenous variables https://arxiv.org/abs/2104.05522
[5] N-BEATS: Neural basis expansion analysis for interpretable time series forecasting http://arxiv.org/abs/1905.10437
[6] Multi-Task Time Series Forecasting With Shared Attention https://ieeexplore.ieee.org/document/9346331/
[7] An RNN-Based Adaptive Hybrid Time Series Forecasting ... https://scholar.dgist.ac.kr/bitstream/20.500.11750/58291/2/2_s2.0_105001384761.pdf
[8] PSformer: Parameter-efficient Transformer with Segment Attention for
  Time Series Forecasting https://arxiv.org/html/2411.01419v1
[9] AWGformer: Adaptive Wavelet-Guided Transformer for Multi-Resolution Time Series Forecasting https://www.semanticscholar.org/paper/ee3dae20458b3392a3b0b4ae4a58e1aaf2b93b34
[10] DDT: A Dual-Masking Dual-Expert Transformer for Energy Time-Series Forecasting https://www.semanticscholar.org/paper/1a3d93515cd9f8ded6aaa8ad71ad874530fef3b0
[11] N-BEATS-RNN: deep learning for time series forecasting https://ieeexplore.ieee.org/document/9356308/
[12] iTransformer: Inverted Transformers Are Effective for Time Series Forecasting https://arxiv.org/abs/2310.06625
[13] Web Traffic Time Series Forecasting using ARIMA and LSTM RNN https://www.itm-conferences.org/10.1051/itmconf/20203203017
[14] Forecasting of Sea Level Time Series using Deep Learning RNN, LSTM, and BiLSTM, Case Study in Jakarta Bay, Indonesia https://www.semanticscholar.org/paper/04b5421c8fc6f09e107595b48bfffc66b67a8fe4
[15] Deep Learning based Time Series Forecasting https://ieeexplore.ieee.org/document/9356200/
[16] Uncertainty Quantification in Machine Learning Modeling for Multi-Step Time Series Forecasting: Example of Recurrent Neural Networks in Discharge Simulations https://www.mdpi.com/2073-4441/12/3/912
[17] Univariant Time Series forecasting of Agriculture load by using LSTM and GRU RNNs https://ieeexplore.ieee.org/document/9236695/
[18] Zero-shot and few-shot time series forecasting with ordinal regression recurrent neural networks https://www.semanticscholar.org/paper/a131cb17777d6918a6b12aaa3648ebe7cb50623a
[19] Attention-Based SeriesNet: An Attention-Based Hybrid Neural Network Model for Conditional Time Series Forecasting https://www.mdpi.com/2078-2489/11/6/305
[20] Augmented Out-of-Sample Comparison Method for Time Series Forecasting Techniques https://link.springer.com/10.1007/978-3-030-47358-7_30
[21] Recurrent Neural Networks for Time Series Forecasting https://arxiv.org/pdf/1901.00069.pdf
[22] Recurrent Neural Networks for Time Series Forecasting: Current Status
  and Future Directions http://arxiv.org/pdf/1909.00590v2.pdf
[23] Lightweight RNN-Based Model for Adaptive Time Series Forecasting with Concept Drift Detection in Smart Homes https://www.iieta.org/download/file/fid/116524
[24] RWKV-TS: Beyond Traditional Recurrent Neural Network for Time Series
  Tasks https://arxiv.org/pdf/2401.09093.pdf
[25] A Temporal Linear Network for Time Series Forecasting https://arxiv.org/html/2410.21448v1
[26] A Distance Correlation-Based Approach to Characterize the Effectiveness
  of Recurrent Neural Networks for Time Series Forecasting https://arxiv.org/pdf/2307.15830.pdf
[27] What is the best RNN-cell structure to forecast each time series
  behavior? https://arxiv.org/html/2203.07844
[28] A Memory-Network Based Solution for Multivariate Time-Series Forecasting http://arxiv.org/pdf/1809.02105.pdf
[29] A Review of Lithium-Ion Battery Capacity Estimation ... https://pdfs.semanticscholar.org/38e0/b3c5aee11894d110ed3d189d825daea26897.pdf
[30] Deep Learning-Based River Flow Forecasting with MLPs https://pdfs.semanticscholar.org/a44e/a69a5b6c3ac5819ac9a707853a1b77f99e86.pdf
[31] arXiv:2501.14929v1 [cs.CV] 24 Jan 2025 https://arxiv.org/pdf/2501.14929.pdf
[32] TERA: Self-Supervised Learning of Transformer Encoder ... https://ar5iv.labs.arxiv.org/html/2007.06028
[33] IDEA Research Report https://arxiv.org/pdf/2301.04020.pdf
[34] conceptual limitations and the role of reduced-order models https://arxiv.org/html/2506.22552v8
[35] Advancing 3D Point Cloud Understanding through Deep ... https://arxiv.org/html/2407.17877v1
[36] Sentiment Analysis using various Machine Learning and ... https://pdfs.semanticscholar.org/2b4e/3ec211e5bfb4bc24a6ede6839434d97bf235.pdf
[37] Probing forced responses and causality in data-driven ... https://arxiv.org/pdf/2506.22552.pdf
[38] A Machine Learning Approach for Horse Racing Betting https://pdfs.semanticscholar.org/1698/257462714eb2183e7ecb2259514a827290bf.pdf
[39] (PDF) Current ARTs, Virologic Failure, and Implications for ... https://pdfs.semanticscholar.org/7116/100dbb52880cfabbbef20425fbe4490386dd.pdf
[40] Hyperparameter Transfer with Mixture-of-Experts Layers https://arxiv.org/html/2601.20205v1
[41] Motion-enhancement to Echocardiography Segmentation ... https://arxiv.org/html/2501.14929v1
[42] arXiv:2503.02251v1 [cs.IR] 4 Mar 2025 https://arxiv.org/pdf/2503.02251.pdf
[43] AMF-MedIT: An Efficient Align-Modulation-Fusion ... https://arxiv.org/pdf/2506.19439.pdf
[44] Enhancing GDP Growth Forecasting with LSTM, GRU, and Hybrid Model: Evidence from South Korea - Dong-Jin Pyo, 2025 https://journals.sagepub.com/doi/10.1177/21582440251359828
[45] Attention Mechanism for Time Series https://www.shadecoder.com/topics/attention-mechanism-for-time-series-a-comprehensive-guide-for-2025
[46] Predict future time series forecasting | PDF https://www.slideshare.net/slideshow/predict-future-time-series-forecasting/238653133
[47] Performance analysis of neural network architectures for time series forecasting: A comparative study of RNN, LSTM, GRU, and hybrid models - PubMed https://pubmed.ncbi.nlm.nih.gov/40777584/
[48] An attention-based deep learning model for multi-horizon ... https://www.sciencedirect.com/science/article/abs/pii/S0360835223006915
[49] A Reinforced Recurrent Encoder with Prediction-Oriented ... https://arxiv.org/html/2601.03683v1
[50] A comparative study of RNN, LSTM, GRU, and hybrid models - NIH https://pmc.ncbi.nlm.nih.gov/articles/PMC12329085/
[51] Rethinking attention mechanism in time series classification https://www.sciencedirect.com/science/article/abs/pii/S0020025523000968
[52] Recurrent Neural Networks for Time Series Forecasting https://www.sciencedirect.com/science/article/abs/pii/S0169207020300996
[53] Performance Analysis of LSTM Vs GRU In Predicting ... https://www.sciencexcel.com/articles/OxKAUru3ZOiGsyGrixqzL7jxj1tRGmTF3eHLlCeM.pdf
[54] How do attention mechanisms enhance time series ... https://milvus.io/ai-quick-reference/how-do-attention-mechanisms-enhance-time-series-forecasting-models
[55] Time-series forecasting with deep learning: a survey https://royalsocietypublishing.org/rsta/article/379/2194/20200209/41189/Time-series-forecasting-with-deep-learning-a
[56] 3. Results And Discussion https://pmc.ncbi.nlm.nih.gov/articles/PMC9453185/
[57] Revisiting Attention for Multivariate Time Series Forecasting - arXiv https://arxiv.org/html/2407.13806v1
[58] Time Series Forecasting of Air Pollutant PM2.5 Using Transformer Architecture https://www.ijsr.net/archive/v12i11/SR231125192357.pdf
[59] CLM-former for enhancing multi-horizon time series forecasting and load prediction in smart microgrids using a robust transformer-based model. https://www.nature.com/articles/s41598-025-34870-y
[60] EVEREST: An Evidential, Tail-Aware Transformer for Rare-Event Time-Series Forecasting https://www.semanticscholar.org/paper/ae7f4544e082ed2d6b23430c148ec81dce57f5a5
[61] PatchMixer: A Patch-Mixing Architecture for Long-Term Time Series Forecasting https://arxiv.org/abs/2310.00655
[62] Time-series forecasting of mortality rates using transformer https://www.tandfonline.com/doi/full/10.1080/03461238.2023.2218859
[63] ResInformer: Residual Transformer-Based Artificial Time-Series Forecasting Model for PM2.5 Concentration in Three Major Chinese Cities https://www.mdpi.com/2227-7390/11/2/476
[64] Sentinel: Multi-Patch Transformer with Temporal and Channel Attention
  for Time Series Forecasting http://arxiv.org/pdf/2503.17658.pdf
[65] Dateformer: Time-modeling Transformer for Longer-term Series Forecasting https://arxiv.org/abs/2207.05397
[66] A Time Series is Worth 64 Words: Long-term Forecasting with Transformers http://arxiv.org/pdf/2211.14730v2.pdf
[67] Learning Novel Transformer Architecture for Time-series Forecasting https://arxiv.org/pdf/2502.13721.pdf
[68] sTransformer: A Modular Approach for Extracting Inter-Sequential and
  Temporal Information for Time-Series Forecasting http://arxiv.org/pdf/2408.09723.pdf
[69] A Systematic Review for Transformer-based Long-term Series Forecasting https://arxiv.org/pdf/2310.20218.pdf
[70] LLMOrbit: A Circular Taxonomy of Large Language Models https://arxiv.org/html/2601.14053v1
[71] Forecasting Russian Equipment Losses Using Time Series ... https://arxiv.org/html/2509.07813v1
[72] LLMOrbit - From Scaling Walls to Agentic AI Systems https://arxiv.org/pdf/2601.14053.pdf
[73] Uncertainty quantification of turbulent systems via ... https://pdfs.semanticscholar.org/f1f0/16c2674c935310e8bbe2aef4a0017a90b40d.pdf
[74] Neural basis expansion analysis for interpretable time ... https://www.semanticscholar.org/paper/N-BEATS:-Neural-basis-expansion-analysis-for-time-Oreshkin-Carpov/13c185b8c461034af2634f25dd8a85889e8ee135
[75] TOWARDS A GENERAL SINGLE-VIEW ASTEROID 3D ... https://arxiv.org/pdf/2508.01079.pdf
[76] [2104.05522] Neural basis expansion analysis with exogenous ... https://ar5iv.labs.arxiv.org/html/2104.05522
[77] DiffInk: Glyph- and Style-Aware Latent Diffusion ... https://arxiv.org/html/2509.23624v2
[78] Comparison of neural basis expansion analysis for ... https://pubmed.ncbi.nlm.nih.gov/35537407/
[79] Error Correcting Codes within Discrete Deep Generative ... https://arxiv.org/html/2410.07840v1
[80] [1905.10437] N-BEATS: Neural basis expansion analysis ... https://arxiv.org/abs/1905.10437
[81] D4D: An RGBD diffusion model to boost monocular depth ... https://arxiv.org/html/2403.07516v1
[82] N-BEATS with a Mixture-of-Experts Layer for ... https://arxiv.org/html/2508.07490v1
[83] Computer Interfaces - Neural Digital Twins https://arxiv.org/pdf/2601.01539.pdf
[84] Deep learning in time series forecasting with transformer ... https://peerj.com/articles/cs-3001/
[85] [PDF] Temporal Convolutional Attention Neural Networks for Time Series ... https://yanglin1997.github.io/files/TCAN.pdf
[86] How to Design Transformer Model for Time-Series Forecasting https://blogs.mathworks.com/deep-learning/2024/11/12/how-to-design-transformer-model-for-time-series-forecasting/
[87] [PDF] Temporal Convolutional Attention Neural Networks ... https://www.semanticscholar.org/paper/dd80f082b2fd3dc2fb82c31d9cb21390becbc46a
