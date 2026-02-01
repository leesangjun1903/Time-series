
# An RNN-Based Adaptive Hybrid Time Series Forecasting Model for Driving Data Prediction

## 1. 핵심 주장 및 주요 기여도 요약

"An RNN-Based Adaptive Hybrid Time Series Forecasting Model for Driving Data Prediction" (Seo & Kim, 2025)은 차량 주행 데이터의 시계열 예측을 위한 Adaptive Multivariate Exponential Smoothing-Recurrent Neural Networks (AMES-RNN) 모델을 제안한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f063bd5d-55a1-4bc1-89a1-646ddb6aa14d/2_s2.0_105001384761.pdf)

**논문의 핵심 주장**은 통계적 모델과 심층학습 모델의 장점을 결합한 하이브리드 방식이 비계절성, 가산 추세 특성을 갖는 차량 데이터 예측에서 기존 MES-LSTM 대비 23.0~55.1%의 성능 향상을 달성할 수 있다는 것이다. 특히 조향각 예측에서 99th percentile 기준 51.7% 개선을 보여준다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f063bd5d-55a1-4bc1-89a1-646ddb6aa14d/2_s2.0_105001384761.pdf)

**주요 기여**는 다음과 같다:

| 기여 요소 | 설명 | 성능 영향 |
|---------|------|---------|
| SC-Updater (평활 계수 온라인 업데이터) | RNN 기반 회귀 모델로 최적 평활 계수 동적 추정 | Type 1→Type 2→AMES-RNN 단계적 개선 |
| 확장된 안정성 조건 (Eq. 8) | 기존 식 (3)보다 더 큰 계수 범위 허용 | ES 모델 표현력 증가 |
| 미래 암시 정보 통합 | 전방 30m 도로 곡률을 입력에 추가 | 조향각 예측 정확도 대폭 향상 |
| 계산 효율성 | <35MB 메모리, <13ms 실행 시간 | 온라인 예측 가능 |

***

## 2. 해결하는 문제 및 제안 방법

### 2.1 문제 정의

논문이 해결하는 두 가지 핵심 문제는:

**문제 1: MES-LSTM의 적응성 부족**

MES-LSTM은 훈련 시 결정된 고정 평활 계수 $\Theta_0 = (\alpha_0, \beta_0, \phi_0)$를 사용한다. 이는 다음 두 가지 한계를 야기한다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f063bd5d-55a1-4bc1-89a1-646ddb6aa14d/2_s2.0_105001384761.pdf)

- 운전 데이터의 특성이 시간에 따라 변할 때(예: 일정 속도 주행 → 급가속), 고정 계수는 이러한 변화에 대응하지 못한다.
- 댐핑 추세 계수 $\phi$가 항상 1로 고정되어, 진정한 댐핑 추세 행동을 반영하지 못한다.

**문제 2: Instance Normalization의 주행 데이터 부적합성**

기존 시계열 예측 방법들이 사용하는 instance normalization $\hat{x} = \frac{x-\mu}{\sigma}$는 다음과 같은 문제를 야기한다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f063bd5d-55a1-4bc1-89a1-646ddb6aa14d/2_s2.0_105001384761.pdf)

일정 속도 주행 시 표준편차 $\sigma \approx 0$이 되어, 정규화 후 데이터가 마치 극도의 변동성을 가진 것처럼 변환된다. 또한 "부드러운 회전 vs 급회전" 같이 패턴 형태는 비슷하지만 변화율이 다른 두 신호는 정규화 후 거의 동일해진다. 이는 다양한 운전 행동 정보를 손실시킨다.

### 2.2 제안하는 방법: AMES-RNN 구조

#### 2.2.1 지수평활 모델 (Exponential Smoothing)

비계절성, 가산 추세를 갖는 시계열에 대한 ES 모델: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f063bd5d-55a1-4bc1-89a1-646ddb6aa14d/2_s2.0_105001384761.pdf)

$$y_{t+1|t} = l_t + b_t$$

$$l_t = \alpha y_t + (1-\alpha)(l_{t-1} + b_{t-1})$$

$$b_t = \beta(l_t - l_{t-1}) + (1-\beta)b_{t-1}$$

여기서 $\alpha, \beta \in $은 평활 계수이고, $\phi \in $은 댐핑 인자이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f063bd5d-55a1-4bc1-89a1-646ddb6aa14d/2_s2.0_105001384761.pdf)

#### 2.2.2 SC-Updater를 통한 적응형 계수 추정

AMES-RNN의 핵심 혁신은 다음 식으로 표현되는 적응형 평활 계수 업데이트이다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f063bd5d-55a1-4bc1-89a1-646ddb6aa14d/2_s2.0_105001384761.pdf)

$$\Theta_t^{(i)} = \Theta_0^{(i)} + \Delta\Theta_t^{(i)}$$

여기서 $\Delta\Theta_t^{(i)} = (\Delta\alpha_t^{(i)}, \Delta\beta_t^{(i)}, \Delta\phi_t^{(i)})$는 SC-Updater-i (RNN 기반 회귀 모델)에 의해 추정된다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f063bd5d-55a1-4bc1-89a1-646ddb6aa14d/2_s2.0_105001384761.pdf)

SC-Updater의 구조: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f063bd5d-55a1-4bc1-89a1-646ddb6aa14d/2_s2.0_105001384761.pdf)
- 입력: 최근 $L_E$ 시간 스텝의 다변량 시계열 데이터 $Y_E^{(i)}(t)$와 미래 정보 $Info_{future}(t)$
- 은닉층: RNN (LSTM 또는 GRU)
- 출력층: Tanh 활성화로 [-1, 1] 범위의 정규화된 변화값 생성
- 최종 출력: 선형 변환을 통해 실제 제약 범위로 스케일링

#### 2.2.3 개선된 안정성 조건

기존 MES-LSTM의 제약 조건 (식 3): [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f063bd5d-55a1-4bc1-89a1-646ddb6aa14d/2_s2.0_105001384761.pdf)

$$0 \leq \alpha, \beta; \quad \alpha \leq 1; \quad \beta \leq (1-\alpha)/(1-\phi)$$

AMES-RNN의 확장된 조건 (식 8): [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f063bd5d-55a1-4bc1-89a1-646ddb6aa14d/2_s2.0_105001384761.pdf)

$$0 \leq \alpha \leq 2 \quad (8a)$$

$$0 \leq \beta \quad (8b)$$

$$\beta \leq \frac{\alpha(1-\phi)}{1-\alpha} \quad (8c)$$

$$0 \leq \phi \leq 1 \quad (8d)$$

이 확장은 ES 모델의 안정성을 유지하면서도 $\alpha$의 범위를 에서 로 확장함으로써 모델의 표현력을 증가시킨다. [mdpi](https://www.mdpi.com/2078-2489/11/6/305)

#### 2.2.4 손실함수 설계

SC-Updater 훈련을 위한 손실함수: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f063bd5d-55a1-4bc1-89a1-646ddb6aa14d/2_s2.0_105001384761.pdf)

$$Loss^{(i)} = \frac{1}{...} \sum_{\tau}(\delta_{\tau}^{(i)} - \hat{\delta}_{\tau}^{(i)})^2 + StabLoss^{(i)}$$

여기서 $StabLoss^{(i)} = 0$ if 식 (8c)를 만족하고, 그렇지 않으면 위반도에 비례하여 증가한다. 이는 ES 모델 안정성을 보장하는 동시에 최적 계수를 학습한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f063bd5d-55a1-4bc1-89a1-646ddb6aa14d/2_s2.0_105001384761.pdf)

#### 2.2.5 Trend Predictor 훈련

전처리된 입력: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f063bd5d-55a1-4bc1-89a1-646ddb6aa14d/2_s2.0_105001384761.pdf)

$$Y_{pre}^{(i)} = Y_T^{(i)} - y_{t+1|t}$$

$$B_F^{(i)} = Y_F^{(i)} - y_{t+1|t}$$

여기서 $y_{t+1|t}$는 훈련된 SC-Updater를 이용한 ES 모델의 추정값이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f063bd5d-55a1-4bc1-89a1-646ddb6aa14d/2_s2.0_105001384761.pdf)

Trend Predictor의 손실함수: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f063bd5d-55a1-4bc1-89a1-646ddb6aa14d/2_s2.0_105001384761.pdf)

$$Loss = \sum_{i=1}^{k} \sum_{f=1}^{F} (b_{t+f}^{(i)} - \hat{b}_{t+f}^{(i)})^2$$

***

## 3. 모델 구조 및 알고리즘

### 3.1 AMES-RNN 전체 구조

| 컴포넌트 | 역할 | 입력 | 출력 |
|---------|------|------|------|
| **SC-Updater-i** | 각 시계열 유형별 평활 계수 동적 추정 | $Y_E^{(i)}, Info_{future}$ | $\Theta_t^{(i)}$ |
| **ES Model** | 수준 및 추세 성분 분해 | $Y_T^{(i)}, \Theta_t^{(i)}$ | $y_{t+1\|t}, \{b_t\}$ |
| **Trend Predictor (RNN)** | 미래 추세 성분 예측 | $Y_{pre}, Info_{future}$ | $\hat{B}_F$ |
| **Output Fusion** | 최종 예측값 생성 | $y_{t+1\|t}, \hat{B}_F$ | $Y_F$ |

### 3.2 AMES-RNN 알고리즘 (Algorithm 2) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f063bd5d-55a1-4bc1-89a1-646ddb6aa14d/2_s2.0_105001384761.pdf)

```
procedure TS_Forecast:
  // 1단계: 각 시계열 유형별 수준 성분 추출
  for i = 1, 2, ..., k do
    Y_E^{(i)}(t) ← {y_{t-L_E+1}^{(i)}, ..., y_t^{(i)}}  // 과거 L_E 시점
    
  // 2단계: SC-Updater를 통한 적응형 평활 계수 추정
  for i = 1, 2, ..., k do
    Θ_t^{(i)} ← SC-Updater-i(Y_E^{(i)}(t), Info_future(t))  // RNN 회귀
    
  // 3단계: ES 모델로 수준 및 추세 분해
  for i = 1, 2, ..., k do
    Y_T^{(i)}(t) ← {y_{t-L_T+1}^{(i)}, ..., y_t^{(i)}}
    ŷ_{t+1|t}^{(i)} ← ES(Y_T^{(i)}, Θ_t^{(i)})  // 1단계 선행 ES 예측
    Y_{pre}^{(i)} ← Y_T^{(i)} - ŷ_{t+1|t}^{(i)}  // 전처리
    
  // 4단계: RNN을 통한 다변량 추세 예측
  B_F(t) ← RNN(Y_pre, Info_future(t))  // F단계 선행 추세 예측
  
  // 5단계: 최종 다단계 예측
  for i = 1, 2, ..., k do
    Ŷ_F^{(i)} ← B_F^{(i)} + ŷ_{t+1|t}^{(i)}
    
  return Y_F(t) = {Ŷ_F^{(1)}, ..., Ŷ_F^{(k)}}
```

### 3.3 데이터셋 생성 (슬라이딩 윈도우) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f063bd5d-55a1-4bc1-89a1-646ddb6aa14d/2_s2.0_105001384761.pdf)

SC-Updater와 Trend Predictor 훈련용 데이터 추출: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f063bd5d-55a1-4bc1-89a1-646ddb6aa14d/2_s2.0_105001384761.pdf)

$$\tau \in [\max(L_E, L_T, L_F), L_n - F]$$

시간 $\tau$에서 추출된 샘플:
- **입력 윈도우**: $Y_E^{(i)}(\tau) = \{y_{\tau-L_E+1}^{(i)}, ..., y_\tau^{(i)}\}$ (크기 $L_E$)
- **출력 윈도우**: $Y_F^{(i)}(\tau) = \{y_{\tau+1}^{(i)}, ..., y_{\tau+F}^{(i)}\}$ (크기 $F$)

그 후 최적화 문제를 풀어 $\delta_\tau^{(i)}$를 결정: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f063bd5d-55a1-4bc1-89a1-646ddb6aa14d/2_s2.0_105001384761.pdf)

$$\min_{\delta} \|y_{\tau+1}^{(i)} - y_{\tau+1|\tau}^{(i)}\|^2 \quad \text{s.t.} \quad \Theta \in \mathcal{C}$$

***

## 4. 성능 향상 및 실험 결과

### 4.1 실험 설정 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f063bd5d-55a1-4bc1-89a1-646ddb6aa14d/2_s2.0_105001384761.pdf)

| 항목 | 내용 |
|------|------|
| **데이터 소스** | IPG CarMaker 차량 시뮬레이터 |
| **테스트 신호** | 조향각(steering angle), 페달 위치(pedal position) |
| **샘플링 속도** | 10 Hz |
| **미래 정보** | 전방 30m 도로 곡률 (1m 간격) |
| **예측 지평선(F)** | 6 단계 (0.6초) |
| **노이즈 추가** | 조향각: ±0.1°, 페달: ±0.5, 곡률: ±125 m⁻¹ |

### 4.2 주요 성능 결과 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f063bd5d-55a1-4bc1-89a1-646ddb6aa14d/2_s2.0_105001384761.pdf)

**페달 위치 예측 (MAE 백분위수)**:

| 모델 | 90th 백분위수(m) | 99th 백분위수(m) | 개선도 |
|------|-----------------|-----------------|--------|
| MES-LSTM | 기준 | 기준 | - |
| Type 1 | -8.7% | -13.2% | ES 출력 개선 |
| Type 2 | -15.3% | -22.8% | 계수 범위 확장 |
| **AMES-RNN** | **-23.0%** | **-32.4%** | **SC-Updater 적용** |

**조향각 예측**:

| 모델 | 90th 백분위수(°) | 99th 백분위수(°) | 개선도 |
|------|-----------------|-----------------|--------|
| MES-LSTM | 기준 | 기준 | - |
| Type 1 | -18.9% | -19.4% | - |
| Type 2 | -31.7% | -28.3% | - |
| AMES-RNN (미래정보 없음) | -42.2% | -43.1% | - |
| **AMES-RNN** | **-55.1%** | **-51.7%** | **미래정보 통합** |

### 4.3 개별 개선 효과 분석 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f063bd5d-55a1-4bc1-89a1-646ddb6aa14d/2_s2.0_105001384761.pdf)

각 개선 사항의 단계적 기여:

1. **$y_{t+1\|t}$로 대체** (Type 1): MES-LSTM 대비 8.7~18.9% 개선
   - 이유: Trend Predictor가 ES 예측값이 아닌 순수 추세만 예측

2. **안정성 조건 확장** (Type 2): Type 1 대비 추가 6.6~12.8% 개선
   - 이유: 더 넓은 계수 범위에서 최적해 탐색 가능

3. **SC-Updater 적용** (AMES-RNN): Type 2 대비 추가 7.3~13.4% 개선
   - 이유: 온라인 매개변수 적응으로 변동하는 데이터 특성 추적

4. **미래 정보 통합**: AMES-RNN 대비 조향각 13% 추가 개선
   - 이유: 도로 곡률과 조향각 간 높은 상관관계(Figure 11 참조)

### 4.4 ES 모델 안정성 검증 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f063bd5d-55a1-4bc1-89a1-646ddb6aa14d/2_s2.0_105001384761.pdf)

SC-Updater가 추정한 계수의 안정성 조건 만족도 (Figure 10):
- 식 (8a) 만족: 100% (α, β가 제약 범위 내)
- 식 (8b) 만족: 100% (β ≥ 0)
- 식 (8c) 만족: 100% (M = α(1-β)/(1-α) - β ≥ 0)
- 식 (8d) 만족: 100% (0 ≤ φ ≤ 1)

### 4.5 계산 효율성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f063bd5d-55a1-4bc1-89a1-646ddb6aa14d/2_s2.0_105001384761.pdf)

| 메트릭 | MES-LSTM | AMES-RNN | 증가율 |
|------|----------|----------|--------|
| 메모리 | 18 MB | 32 MB | +77% |
| FLOPs | 17×10³ | 35×10³ | +106% |
| 평균 실행 시간 | 5.2 ms | 12.8 ms | +146% |

결론: <35MB 메모리와 <13ms 실행 시간으로 **100ms 샘플링 간격에서 온라인 예측 가능** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f063bd5d-55a1-4bc1-89a1-646ddb6aa14d/2_s2.0_105001384761.pdf)

***

## 5. 일반화 성능(Generalization Performance) 심층 분석

### 5.1 일반화 성능 향상 메커니즘

#### 5.1.1 적응형 매개변수 업데이트의 역할

AMES-RNN이 일반화 성능을 향상시키는 핵심 메커니즘은 **온라인 적응형 매개변수 추정**이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f063bd5d-55a1-4bc1-89a1-646ddb6aa14d/2_s2.0_105001384761.pdf)

고정 계수 모델의 문제:
- 훈련 데이터의 특성에 최적화된 $\Theta_0$는 테스트 시 마주치는 새로운 운전 패턴에 부적합
- 급격한 가속/감속이나 회전 각속도 변화 시 분해 오류 발생

적응형 SC-Updater의 해결책:
- 각 시간 단계에서 현재 데이터 특성을 반영한 $\Theta_t^{(i)}$ 추정
- RNN의 기억 능력으로 최근 추세 변화 포착
- **훈련-테스트 분포 변화(distribution shift)에 대한 견고성 증가**

실험 증거 (Table 7): [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f063bd5d-55a1-4bc1-89a1-646ddb6aa14d/2_s2.0_105001384761.pdf)
- Type 2 (고정 계수): ES 예측 1-단계 오류 = 0.48 (페달 위치)
- AMES-RNN (적응 계수): ES 예측 1-단계 오류 = 0.31 (-35% 개선)

#### 5.1.2 확장된 안정성 조건의 표현력 증대

기존 식 (3)의 계수 범위 제약이 과도하게 좁았다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f063bd5d-55a1-4bc1-89a1-646ddb6aa14d/2_s2.0_105001384761.pdf)

$\alpha \in$ 제약의 문제: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f063bd5d-55a1-4bc1-89a1-646ddb6aa14d/2_s2.0_105001384761.pdf)
- 빠르게 변하는 신호(급회전, 급가속)에 대응하기 위해 더 큰 $\alpha$ 필요
- 슈퍼스무싱(over-smoothing) 또는 under-smoothing 양극단

확장된 범위 (식 8)의 이점:
- $\alpha \in$ 허용으로 다양한 신호 변화율 대응 [mdpi](https://www.mdpi.com/2078-2489/11/6/305)
- Type 2 도입으로 6.6~12.8% 추가 개선
- **더 광범위한 데이터 특성에 대한 적응성**

#### 5.1.3 비인과적 MA 필터링의 역할

훈련 데이터 생성 시 $\delta_\tau^{(i)}$에 비인과적(non-causal) MA 필터 적용: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f063bd5d-55a1-4bc1-89a1-646ddb6aa14d/2_s2.0_105001384761.pdf)

$$\delta'\_\tau = \frac{1}{2K+1} \sum_{j=-K}^{K} \delta_{\tau+j}$$

효과:
- 급격한 계수 변화를 평활화하여 **과적합(overfitting) 방지**
- 더 안정적인 학습 신호 제공
- RNN이 장기 추세를 포착하도록 유도

Figure 5 예시: 필터 적용 전 진동 → 필터 적용 후 부드러운 곡선

#### 5.1.4 미래 정보 통합의 불확실성 감소

미래 암시 정보(forward road curvature)의 역할: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f063bd5d-55a1-4bc1-89a1-646ddb6aa14d/2_s2.0_105001384761.pdf)

입력 융합 (Input Fusion):
$$Y_E^{(i)} \oplus Info_{future} \rightarrow [Y_E^{(i)}, Info_{future}] \in \mathbb{R}^{L_E+r}$$

효과:
- 조향각 예측에서 13% 추가 개선
- 도로 곡률과 조향각 간 높은 상관성으로 인한 불확실성 감소
- **외생 변수의 효과적 활용으로 일반화 능력 증강**

***

### 5.2 일반화 성능 한계

#### 5.2.1 현재 제약 조건

논문의 실험은 **IPG CarMaker 시뮬레이션 환경**에서만 수행되었다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f063bd5d-55a1-4bc1-89a1-646ddb6aa14d/2_s2.0_105001384761.pdf)
- 실제 센서 노이즈의 복잡성 미흡
- 제한된 환경 변동성 (기후, 도로 상태 등 미반영)
- 특정 가상 드라이버 모델의 특성에만 최적화

#### 5.2.2 분포 외 일반화(Out-of-Distribution Generalization) 우려 [arxiv](https://arxiv.org/html/2503.13868v1)

시계열 연구의 공통 문제:
- 훈련 분포: 특정 운전 조건(시간대, 날씨, 도로)
- 테스트 분포: 미지의 조건 → **분포 변화(domain shift)** 발생

AMES-RNN의 취약점:
- SC-Updater가 훈련 데이터의 계수 분포에만 기반
- 극단적 운전 상황(응급 제동, 극단 회전) 대응 미흡

#### 5.2.3 데이터 규모의 영향

차량 주행 데이터의 수집 난제: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f063bd5d-55a1-4bc1-89a1-646ddb6aa14d/2_s2.0_105001384761.pdf)
- 대규모 다중 차량 데이터 부재 → RNN이 Transformer보다 유리 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f063bd5d-55a1-4bc1-89a1-646ddb6aa14d/2_s2.0_105001384761.pdf)
- 개인화된 드라이버 모델 개발 시 샘플 크기 제한
- **소규모 데이터셋에서의 일반화** 여전히 도전 과제

***

### 5.3 최신 연구와의 비교를 통한 일반화 성능 컨텍스트

| 모델 | 연도 | 방식 | 일반화 전략 | 주요 강점 |
|------|------|------|-----------|---------|
| **ETSformer** [arxiv](https://arxiv.org/pdf/2202.01381.pdf) | 2022 | Transformer + ES 분해 | 다중 계절성 처리 | 장기 의존성 |
| **ES-dRNN** [arxiv](https://arxiv.org/pdf/2112.02663.pdf) | 2021 | ES + dilated RNN | 다중 계절성, 계층 구조 | 불확실성 구간 |
| **CNN-Bayes LSTM** [ieeexplore.ieee](https://ieeexplore.ieee.org/document/9924783/) | 2022 | CNN-LSTM + 베이지안 | 불확실성 정량화 | 소규모 데이터 |
| **AMES-RNN** | 2025 | 적응형 ES + RNN | 온라인 계수 조정 | **온라인 배포, 낮은 복잡도** |
| **Chronos** [arxiv](https://arxiv.org/pdf/2501.07034.pdf) | 2025 | 기초 모델 + 미세조정 | 사전학습 + 전이학습 | **극단 효율성, 영역 전이** |

**비교 분석**:
- ETSformer, ES-dRNN: 계절성 있는 데이터에 우수, 복잡도 높음
- CNN-Bayes LSTM: 불확실성 정량화 우수, 계산 비용 증가
- **AMES-RNN: 차량 도메인 특화, 실시간 배포 가능** ← 명확한 강점
- **Chronos: 기초 모델 접근으로 뛰어난 일반화** ← 향후 벤치마크

***

## 6. 모델의 한계

### 6.1 기술적 한계

1. **비계절성, 가산 추제 가정**
   - 계절성 있는 도시 교통 흐름 등에 적용 불가
   - 승수적 추제(multiplicative trend) 시스템 미지원

2. **시뮬레이션 데이터 기반**
   - 현실 센서 노이즈, 신호 손실 미포함
   - 극한 주행 상황(고속, 오프로드) 미검증

3. **온라인 학습의 장기 안정성**
   - 개념 표류(concept drift) 처리 미흡
   - 매우 긴 배포 기간 동안의 누적 오류 분석 부재

### 6.2 실험적 한계

1. **제한된 도메인**
   - 정상적 차량 이동만 평가
   - 교통 상호작용(차량 간 간섭) 미포함

2. **단일 데이터셋 검증**
   - 공개 데이터셋(nuScenes, Argoverse) 비교 부재
   - 도메인 간 일반화 검증 부재

3. **미래 정보의 접근성**
   - HD 지도(고정밀 맵) 가용성 가정
   - 사전 정보 없는 환경에서의 성능 미평가

***

## 7. 향후 연구에 미치는 영향 및 고려사항

### 7.1 단기 연구 개선사항

**1. 실제 주행 데이터 검증**
- **필요성**: 시뮬레이션과 실제 차량의 동역학 차이 존재
- **구현 방법**: 
  - 공개 데이터셋(nuScenes, Waymo Open, Argoverse 2) 활용
  - 실제 드라이빙 시뮬레이터에서 수집한 데이터
- **기대 효과**: 모델의 실용성 증명

**2. 확장된 데이터 특성 지원**
- **계절성 있는 데이터**: 공식 (수정된 알고리즘 필요)
- **다중 계절성**: ETSformer 스타일 접근법 통합
- **비정상성(non-stationary) 처리**: 차분화 또는 적응형 전처리

**3. 개념 표류(Concept Drift) 처리**
- **온라인 재학습 전략**:
  - 검증 오류 모니터링으로 재훈련 트리거
  - 슬라이딩 윈도우 방식의 점진적 업데이트
  - 극단 탐지 후 긴급 모델 재설정
- **참고 연구**:  에너지 도메인 온라인 적응 프레임워크 [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0952197623006838)

**4. 미래 정보의 다양화**
- 시선 추적(eye-gaze) 데이터
- 자동차 간 통신(V2X) 정보
- 교통 상호작용 신호(surrounding vehicle trajectories)

### 7.2 중기 연구 방향

**1. 다중 예측 지평선 및 불확실성 정량화**
- **한계 극복**: 점 예측 → 확률적 예측으로 확장
- **방법**: 
  - 베이지안 추론 통합 (CNN-Bayes LSTM 스타일)
  - 또는 분위수 회귀 (Quantile Regression)
- **응용**: 위험 기반 경로 계획, 동적 제약 조건

**2. 도메인 전이 학습(Transfer Learning)**
- **목표**: 한 차종/운전자 → 다른 차종/운전자 모델 적응
- **전략**:
  - 기초 모델 사전학습 (다양한 드라이버로부터)
  - 미세조정 (타겟 드라이버 소규모 데이터)
  - **참고**:  Chronos 기초 모델의 33.75% 개선 사례 [arxiv](https://arxiv.org/pdf/2501.07034.pdf)

**3. 설명 가능성(Interpretability) 개선**
- **중요성**: 자율주행에서는 예측 신뢰성이 안전과 직결
- **접근법**:
  - SC-Updater의 학습된 계수 변화 해석
  - 주의(Attention) 메커니즘 추가로 입력 중요도 시각화
  - SHAP 값 기반 피처 기여도 분석

### 7.3 장기 연구 비전

**1. 다중 차종/운전자 기초 모델 구축**
- **기대 효과**: 개인화 모델의 빠른 적응 → 안전 시스템 개선
- **기술**: 메타 러닝 또는 다중 작업 학습

**2. 엣지 컴퓨팅 최적화**
- **목표**: 현재 <13ms → <5ms 달성 (더 높은 샘플링 속도)
- **방법**: 모델 경량화(distillation), 양자화(quantization)

**3. 자율주행 시스템의 폐루프(Closed-loop) 검증**
- **현재**: 오픈루프 다단계 예측 평가
- **향후**: 실제 자율주행 제어 루프에서의 성능 검증
- **기대**: 예측 정확도 개선 → 자동차 안전도 증가

***

## 8. 2020년 이후 관련 최신 연구 비교 분석

### 8.1 하이브리드 시계열 모델 진화

| 연도 | 모델 | 핵심 아이디어 | 시계열 특성 | 성능 | 실무 적용 |
|------|------|-----------|----------|------|---------|
| 2020 | Attention-SeriesNet [mdpi](https://www.mdpi.com/2078-2489/11/6/305) | CNN+LSTM + 주의 메커니즘 | 다중 조건 | 기준 | 제한적 |
| 2020 | CNN-LSTM (에너지) [mdpi](https://www.mdpi.com/2073-4433/11/5/457) | EWT+LSTM | 파동분해 | 향상 | 중간 |
| 2021 | MES-LSTM [arxiv](https://arxiv.org/pdf/2112.08618.pdf) | 통계+심층학습 | 비계절, 다변량 | 우수 | 제한적 |
| 2021 | ES-dRNN [arxiv](https://arxiv.org/pdf/2112.02663.pdf) | 확장 ES + dilated RNN | 다중 계절성 | 최우수 | 낮음 |
| 2022 | ETSformer [arxiv](https://arxiv.org/pdf/2202.01381.pdf) | Transformer + ES 분해 | 계절성+추세 | 최우수 | 중간 |
| 2022 | CNN-Bayes LSTM [ieeexplore.ieee](https://ieeexplore.ieee.org/document/9924783/) | 베이지안+하이브리드 | 불확실성 | 우수 | 낮음 |
| 2022 | LSTM-Prophet [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC9202617/) | STL+RNN+Prophet | 계절추세분해 | 우수 | 중간 |
| 2023 | HyVAE [arxiv](https://arxiv.org/pdf/2303.07048.pdf) | 변분 오토인코더 + 추세 | 지역패턴+동역학 | 우수 | 낮음 |
| **2025** | **AMES-RNN** | **적응형 ES+RNN** | **비계절+가산+온라인** | **우수** | **높음** |
| 2025 | Chronos (차량) [arxiv](https://arxiv.org/pdf/2501.07034.pdf) | 기초 모델 + 미세조정 | 도메인 무관 | 최우수 | 높음 |

### 8.2 차량 궤적 예측 최신 동향

**RNN 기반 방법의 진화**:
1. **초기 (2019-2021)**: CNN-LSTM 시퀀스 모델 (기본) [pure.kaist.ac](https://pure.kaist.ac.kr/en/publications/vehicle-trajectory-prediction-with-convolutional-neural-network-a)
2. **발전 (2022-2023)**: Graph 기반 상호작용 모델 (다중 에이전트) [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2024W/MULA/papers/Liu_LAformer_Trajectory_Prediction_for_Autonomous_Driving_with_Lane-Aware_Scene_Constraints_CVPRW_2024_paper.pdf)
3. **현재 (2024-2025)**: 기초 모델 + 트랜스포머 기반 (확장성) [arxiv](https://arxiv.org/html/2509.10570v1)

**AMES-RNN의 위치**:
- **강점**: 낮은 복잡도, 온라인 실시간 성능, 차량 도메인 특화
- **약점**: 기초 모델 대비 제한된 도메인 전이, 불확실성 정량화 미흡

### 8.3 온라인 학습/적응형 모델

최신 동향:, [arxiv](https://arxiv.org/pdf/2509.03810.pdf)
- **개념 표류 대응**: 동적 재훈련, 혼합 모델 앙상블
- **특성 공간 적응**: 입력 특성 표현 업데이트 (전체 모델 업데이트 대신)
- **비효율성 극복**: 샘플별 업데이트 → 미니배치 또는 윈도우 기반 업데이트

**AMES-RNN의 혁신**:
- SC-Updater의 **RNN 기반 회귀 방식** (직접 계수 최적화 vs. 휴리스틱 규칙)
- **안정성 제약 명시적 포함** (손실함수)
- **다변량 상호작용 유지** (개별 계수 적응 후 결합)

### 8.4 일반화 성능 관점

최신 연구 경향:,, [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10824647/)

**도전 과제**:
1. 작은 데이터셋에서 과적합
2. 분포 변화에 취약
3. 극한 상황(tail events) 예측 부정확

**AMES-RNN의 대응**:
1. **온라인 매개변수 조정** → 훈련-테스트 분포 차이 완화
2. **비인과적 MA 필터** → 안정적 학습 신호
3. **미래 정보 활용** → 불확실성 감소

***

## 결론

AMES-RNN은 지수평활의 수학적 견고성과 RNN의 유연성을 효과적으로 결합한 차량 주행 데이터 예측 모델이다. 특히 온라인 환경에서 23~55% 성능 개선과 <13ms의 실행 시간을 달성함으로써 **실제 차량 제어 시스템의 실용적 배포**에 그 가치를 증명하였다.

그러나 시뮬레이션 환경 기반 평가, 비계절성 가정, 분포 외 일반화 한계 등으로 인해 다음 단계 연구는 **실제 주행 데이터 검증**, **개념 표류 처리**, **도메인 전이 학습**에 집중해야 한다. 아울러 최신 기초 모델(Chronos) 패러다임과의 통합을 고려할 때, AMES-RNN의 온라인 적응 메커니즘은 **기초 모델 미세조정 전략에 보완적 가치**를 제공할 수 있을 것으로 기대된다.

***

## 참고문헌

 Seo, J.H., Kim, K.-D. "An RNN-Based Adaptive Hybrid Time Series Forecasting Model for Driving Data Prediction," *IEEE Access*, vol. 13, pp. 54177-54191, March 2025. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f063bd5d-55a1-4bc1-89a1-646ddb6aa14d/2_s2.0_105001384761.pdf)

 Attention-Based SeriesNet (2020), MDPI Information. [mdpi](https://www.mdpi.com/2078-2489/11/6/305)

 Hybrid LSTM Data-Driven Model with Empirical Wavelet Transform (2020), MDPI Atmosphere. [mdpi](https://www.mdpi.com/2073-4433/11/5/457)

 REMD-LSTM (2021), PMC NIH. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC8178659/)

 MES-LSTM (2021), arXiv:2112.08618. [arxiv](https://arxiv.org/pdf/2112.08618.pdf)

 Hybrid Variational Autoencoder (2023), arXiv:2303.07048. [arxiv](https://arxiv.org/pdf/2303.07048.pdf)

 LSTM-Prophet Hybrid (2022), PMC NIH. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC9202617/)

 ES-dRNN (2021), arXiv:2112.02663. [arxiv](https://arxiv.org/pdf/2112.02663.pdf)

 Chronos 차량 추적 (2025), arXiv:2501.07034. [arxiv](https://arxiv.org/pdf/2501.07034.pdf)

 ETSformer (2022), arXiv:2202.01381. [arxiv](https://arxiv.org/pdf/2202.01381.pdf)

 ACyLeR (2024), IEEE Access. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10824647/)

 Data Augmentation for Time Series (2024), IJASEIT. [ijaseit.insightsociety](https://ijaseit.insightsociety.org/index.php/ijaseit/article/view/18550)

 CNN-Bayes LSTM (2022), IEEE. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/9924783/)

 Bi-LSTM for Malaria (2025), UINSU. [jurnal.uinsu.ac](https://jurnal.uinsu.ac.id/index.php/zero/article/view/26043)

 Out-of-Distribution Generalization Survey (2025), arXiv. [arxiv](https://arxiv.org/html/2503.13868v1)

 Large Foundation Models for Trajectory Prediction (2025), arXiv. [arxiv](https://arxiv.org/html/2509.10570v1)

 LAformer (2024), CVPR. [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2024W/MULA/papers/Liu_LAformer_Trajectory_Prediction_for_Autonomous_Driving_with_Lane-Aware_Scene_Constraints_CVPRW_2024_paper.pdf)

 Online Adaptation for Energy (2023), Elsevier. [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0952197623006838)

 Online Time Series Prediction with Feature Adjustment (2025), arXiv:2509.03810. [arxiv](https://arxiv.org/pdf/2509.03810.pdf)

 CNN-Seq2Seq 차량 궤적 (2020), KAIST. [pure.kaist.ac](https://pure.kaist.ac.kr/en/publications/vehicle-trajectory-prediction-with-convolutional-neural-network-a)
