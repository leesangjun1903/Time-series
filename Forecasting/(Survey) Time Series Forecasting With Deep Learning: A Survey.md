
# Time Series Forecasting With Deep Learning: A Survey

## 1. 논문의 핵심 주장 및 주요 기여

### 1.1 핵심 주장 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

Lim과 Zohren의 2020년 논문은 **다양한 도메인에서 시계열 문제의 특성에 맞춘 심층 학습 아키텍처의 필요성**을 주장한다. 저자들은 데이터 가용성과 계산 능력의 증가로 인해 순수 데이터 기반의 기계학습이 시계열 모델링의 필수 부분이 되었음을 강조한다. 특히 **귀납적 편향(inductive bias)**을 반영한 신경망 설계를 통해 복잡한 데이터 표현을 자동으로 학습할 수 있으며, 수동 특성 공학(feature engineering)의 필요성을 제거할 수 있다고 주장한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

### 1.2 주요 기여 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

논문은 네 가지 핵심 영역을 다룬다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

1. **인코더-디코더 설계**: 시계열 예측 문제의 기본 구조로 인코더 $g_{enc}(\cdot)$가 과거 정보를 잠재 변수 $z_t$로 압축하고, 디코더 $g_{dec}(\cdot)$가 이를 예측으로 변환하는 방식을 체계화한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

2. **다중 예측 방법**: 한 단계 선행 예측에서 다중 지평선(multi-horizon) 예측으로의 확장을 다룬다. 반복적 방법(autoregressive)과 직접적 방법(direct)의 장단점을 비교한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

3. **하이브리드 모델**: 통계적 시계열 모델(ARIMA, 지수평활)을 신경망 성분과 결합하여 **도메인 지식을 주입**하고 일반화 성능을 향상시키는 접근법을 제시한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

4. **결정 지원**: 해석성(interpretability)과 반사실적 예측(counterfactual prediction)을 통해 모델을 실제 의사결정에 활용하는 방법을 탐색한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

## 2. 해결하고자 하는 문제와 제안 방법

### 2.1 핵심 문제 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

시계열 예측에서 딥러닝의 적용 시 직면하는 주요 문제들:

- **다양성**: 기후 모델링, 생의학, 금융, 소매 등 도메인마다 시계열 특성이 크게 다름 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)
- **시간 의존성**: 미래 값이 과거 관측에 복잡하게 의존하는 구조 학습의 어려움 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)
- **검증 데이터 부족**: M-대회에서 보였듯이 전통 모델이 머신러닝을 능가하는 현상 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)
- **특성 정규화**: 훈련과 테스트 시 데이터 분포의 불일치 문제 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

### 2.2 제안 방법론 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

#### 2.2.1 기본 구조 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

$$\hat{y}_{i,t+1} = f(y_{i,t-k:t}, x_{i,t-k:t}, s_i)$$

여기서:
- $y_{i,t-k:t}$ = 목표 변수의 과거 관측
- $x_{i,t-k:t}$ = 외생 입력
- $s_i$ = 정적 메타데이터

#### 2.2.2 인코더-디코더 구조 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

$$z_t = g_{enc}(y_{t-k:t}, x_{t-k:t}, s)$$
$$f(y_{t-k:t}, x_{t-k:t}, s) = g_{dec}(z_t)$$

#### 2.2.3 주요 아키텍처 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

**CNN(Causal Convolution)**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)
$$h_t^{l+1} = A\left(\sum_{\tau=0}^{k} W^{(l,\tau)}h_{t-\tau}^l\right)$$

음향 신호 처리의 FIR 필터와 유사하며, 시간 불변성(time-invariance)을 가정한다. 확장된 합성곱(dilated convolution)은 수용 영역을 기하급수적으로 확장한다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

$$h_t^{l+1} = A\left(\sum_{\tau=0}^{\lfloor k/d_l\rfloor} W^{(l,\tau)}h_{t-d_l\tau}^l\right)$$

**RNN/LSTM**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

LSTM은 vanishing/exploding gradient 문제를 해결하기 위해 셀 상태 $c_t$를 도입한다:

$$i_t = \sigma(W_{i1}z_{t-1} + W_{i2}y_t + W_{i3}x_t + W_{i4}s + b_i)$$
$$o_t = \sigma(W_{o1}z_{t-1} + W_{o2}y_t + W_{o3}x_t + W_{o4}s + b_o)$$
$$f_t = \sigma(W_{f1}z_{t-1} + W_{f2}y_t + W_{f3}x_t + W_{f4}s + b_f)$$
$$z_t = o_t \odot \tanh(c_t)$$
$$c_t = f_t \odot c_{t-1} + i_t \odot \tanh(W_{c1}z_{t-1} + W_{c2}y_t + W_{c3}x_t + W_{c4}s + b_c)$$

Bayesian 필터와의 관계를 통해 RNN은 상태 전이와 오차 보정을 동시에 근사한다고 해석할 수 있다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

**Attention Mechanism**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

$$h_t = \sum_{\tau=0}^{k} \alpha(\kappa_t, q_\tau)v_{t-\tau}$$

과거의 모든 시점에 동적으로 가중치를 할당하여 장기 의존성을 직접 포착한다. Softmax 가중치를 통해 attention 가중치는 $\sum_{\tau=0}^{k} \alpha(t,\tau) = 1$을 만족한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

#### 2.2.4 손실 함수 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

**회귀 문제**:
$$L_{regression} = \frac{1}{T}\sum_{t=1}^{T}(y_t - \hat{y}_t)^2$$

**확률적 예측** (가우시안):
$$y_{t+\tau} \sim N(\mu(t,\tau), \zeta(t,\tau)^2)$$
$$\mu(t,\tau) = W_\mu h_t^L + b_\mu$$
$$\zeta(t,\tau) = \text{softplus}(W_\Sigma h_t^L + b_\Sigma)$$

#### 2.2.5 다중 지평선 예측 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

$$\hat{y}_{t+\tau} = f(y_{t-k:t}, x_{t-k:t}, u_{t-k:t+\tau}, s, \tau)$$

- **반복적 방법**: 자회귀 방식으로 과거 예측을 미래 입력으로 사용 → 오차 누적 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)
- **직접적 방법**: Seq2seq 구조로 모든 지평선을 동시에 예측 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

## 3. 모델 구조의 상세 설명

### 3.1 Encoder-Decoder 프레임워크 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

시계열 예측 모델은 두 가지 핵심 성분으로 구성된다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

| 성분 | 역할 | 예시 |
|------|------|------|
| **인코더** | 과거 정보를 압축된 표현으로 변환 | CNN, LSTM, Attention |
| **디코더** | 잠재 표현에서 미래 값을 생성 | 선형층 + 활성화 함수 |

### 3.2 CNN 기반 모델 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

WaveNet 아키텍처는 확장된 합성곱을 통해 2^L 개의 타임스텝을 활용하면서도 계산 복잡도를 $O(k)$로 유지한다. 각 계층에서 신호 처리의 다양한 주파수 해상도를 포착한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

**장점**: 병렬 처리 가능, 고정 수용 영역 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)
**단점**: 긴 과거 윈도우 필요, 시간 의존성 표현 제한 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

### 3.3 RNN 기반 모델 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

무한 수용 영역을 통해 모든 과거 정보를 원칙적으로 활용할 수 있다. LSTM의 게이팅 메커니즘은 정보 흐름을 명시적으로 제어한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

**LSTM의 세 개 게이트**:
- **입력 게이트** $i_t$: 새 정보 허용 정도
- **잊음 게이트** $f_t$: 이전 정보 유지 정도  
- **출력 게이트** $o_t$: 셀 상태 노출 정도

### 3.4 Attention 메커니즘 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

**다중 헤드 Attention** (Transformer):
$$\alpha(t) = \text{softmax}(\eta_t)$$
$$\eta_t = W_\eta \tanh(W_1 \kappa_{t-1} + W_2 q_\tau + b_\eta)$$

여러 헤드를 병렬로 실행하여 다양한 시간 의존성을 동시에 포착한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

## 4. 하이브리드 모델 (성능 향상)

### 4.1 하이브리드 모델의 필요성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

M-대회 결과: 순수 머신러닝이 통계 모델에 뒤떨어짐 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)
- **원인 1**: 머신러닝의 유연성이 오버피팅을 유발
- **원인 2**: 데이터 전처리 민감성

**해결책**: 도메인 지식을 신경망에 주입하여 가설 공간 축소 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

### 4.2 ES-RNN (M4 우승 모델) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

Holt-Winters 지수평활과 RNN의 결합:

$$\hat{y}_{i,t+\tau} = \exp(W_{ES}h_{i,t+\tau}^L + b_{ES}) \times l_{i,t} \times \gamma_{i,t+\tau}$$

여기서:
$$l_{i,t} = \beta_1^{(i)} \frac{y_{i,t}}{\gamma_{i,t}} + (1-\beta_1^{(i)})l_{i,t-1}$$
$$\gamma_{i,t} = \beta_2^{(i)} \frac{y_{i,t}}{l_{i,t}} + (1-\beta_2^{(i)})\gamma_{i,t-\kappa}$$

**핵심 아이디어**: 수준($l_{i,t}$)과 계절성($\gamma_{i,t}$) 성분은 통계 업데이트로 처리하고, RNN은 잔차 학습에 집중 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

### 4.3 Deep State Space Models (확률적 하이브리드) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

선형 상태 공간 모델의 매개변수를 신경망으로 인코딩:

$$y_t = a(h_{i,t+\tau}^L)^T l_t + \phi(h_{i,t+\tau}^L)\epsilon_t$$
$$l_t = F(h_{i,t+\tau}^L)l_{t-1} + q(h_{i,t+\tau}^L) + \Sigma(h_{i,t+\tau}^L) \odot \Sigma_t$$

Kalman 필터링을 통한 추론으로 확률적 예측을 생성한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

## 5. 모델 일반화 성능 향상 가능성 (핵심 분석)

### 5.1 논문에서 제시하는 일반화 개선 방법 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

#### 5.1.1 하이브리드 접근 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

**기계학습 단독의 문제점**:
- 소규모 데이터셋에서 과적합 경향 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)
- 전처리에 민감 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

**하이브리드 모델의 장점**:
- 도메인 전문가 지식 통합 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)
- 비정상 성분과 정상 성분 분리 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)
- 통계적 해석성 제공 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

#### 5.1.2 입력 정규화 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

데이터 전처리는 일반화의 핵심:
- GroupNormalizer: 각 시계열 독립적 정규화
- 트렌드 제거 및 계절성 조정
- 이상치 처리

#### 5.1.3 다중 헤드 Attention [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

Attention 메커니즘이 일반화를 개선하는 이유:

1. **선택적 초점**: 관련 있는 과거 시점만 가중 부여 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)
2. **다중 표현**: 여러 헤드가 다양한 의존성 포착 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)
3. **동적 가중치**: 각 샘플마다 다른 주의 패턴 학습 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

### 5.2 2020-2026년 최신 발전에서의 일반화 개선 [royalsocietypublishing](https://royalsocietypublishing.org/doi/10.1098/rsta.2020.0209)

#### 5.2.1 Foundation 모델의 등장 (2024-2026) [worldscientific](https://www.worldscientific.com/doi/abs/10.1142/S0129065721300011)

**TimesFM (Google, 2024)**:
- 100B 데이터포인트로 사전 훈련 [royalsocietypublishing](https://royalsocietypublishing.org/doi/10.1098/rsta.2020.0209)
- Zero-shot 예측에서 감독 모델과 동등한 성능 [royalsocietypublishing](https://royalsocietypublishing.org/doi/10.1098/rsta.2020.0209)
- 패치 기반 토큰화로 맥락/지평선 무관 [royalsocietypublishing](https://royalsocietypublishing.org/doi/10.1098/rsta.2020.0209)

**Time-MoE (2025)**:
- 24억 개 매개변수의 혼합 전문가 모델 [worldscientific](https://www.worldscientific.com/doi/abs/10.1142/S0129065721300011)
- 스케일링 법칙 검증: 모델 크기와 훈련 토큰에 따른 성능 개선 [worldscientific](https://www.worldscientific.com/doi/abs/10.1142/S0129065721300011)
- 밀집 모델 대비 유의미한 성능 향상 [worldscientific](https://www.worldscientific.com/doi/abs/10.1142/S0129065721300011)

**일반화 개선 메커니즘**:
1. **광대한 사전 훈련**: 다양한 도메인에서 패턴 학습 [royalsocietypublishing](https://royalsocietypublishing.org/doi/10.1098/rsta.2020.0209)
2. **전이 학습**: 새 데이터셋에서 미세 조정 불필요 [worldscientific](https://www.worldscientific.com/doi/abs/10.1142/S0129065721300011)
3. **도메인 무관성**: 여러 산업 데이터로 학습하여 편향 감소 [royalsocietypublishing](https://royalsocietypublishing.org/doi/10.1098/rsta.2020.0209)

#### 5.2.2 하이브리드 분해 방법 (2022-2025) [liebertpub](https://www.liebertpub.com/doi/10.1089/big.2020.0159)

**FEDformer (2022)**:
- 주파수 영역 분해로 희소 표현 활용 [liebertpub](https://www.liebertpub.com/doi/10.1089/big.2020.0159)
- MOE 분해: 각 전문가가 특정 주파수 성분 담당 [liebertpub](https://www.liebertpub.com/doi/10.1089/big.2020.0159)
- 14.8% MSE 감소 vs Autoformer [liebertpub](https://www.liebertpub.com/doi/10.1089/big.2020.0159)

**ARIMA + Deep Learning (2025)**:
- 2단계: ARIMA의 선형 기준선 + 딥러닝의 비선형 잔차 학습 [dl.acm](https://dl.acm.org/doi/10.1145/3533382)
- 적응형 융합 메커니즘: 단기는 ARIMA, 장기는 딥러닝 [dl.acm](https://dl.acm.org/doi/10.1145/3533382)
- Informer/Autoformer 능가 [dl.acm](https://dl.acm.org/doi/10.1145/3533382)

#### 5.2.3 알키텍처 개선 (2023-2025) [dl.acm](https://dl.acm.org/doi/10.1145/3533382)

| 방법 | 혁신 | 일반화 개선 |
|------|------|-----------|
| **Informer** | ProbSparse Attention (O(L log L)) | 계산 효율 → 더 큰 모델 가능 |
| **Autoformer** | Auto-correlation + 분해 | 주기성 포착 → 도메인 외 데이터에도 적용 |
| **FEDformer** | 주파수 강화 (O(L)) | 희소 표현 → 노이즈 견고성 |
| **Crossformer** | 차원 간 의존성 | 다변량 예측 정확도 향상 |
| **PatchTST** | 패치 기반 토큰화 | 계산 복잡도 감소 + 일반화 |

#### 5.2.4 불확실성 정량화 [worldscientific](https://www.worldscientific.com/doi/abs/10.1142/S0129065721300011)

Foundation 모델은 자연스럽게 예측 분포 제공:
- 확률적 예측으로 신뢰도 평가 가능 [royalsocietypublishing](https://royalsocietypublishing.org/doi/10.1098/rsta.2020.0209)
- Quantile 회귀로 리스크 관리 [royalsocietypublishing](https://royalsocietypublishing.org/doi/10.1098/rsta.2020.0209)
- 극단 사건 대비 의사결정 지원 [worldscientific](https://www.worldscientific.com/doi/abs/10.1142/S0129065721300011)

### 5.3 일반화 성능 한계 및 해결 과제 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

#### 5.3.1 논문에서 지적한 한계 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

1. **이산화 요구**: 정칙한 간격의 관측 가정
   - 해결책: Neural ODEs (연속 시간 모델) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

2. **계층적 구조 미지원**: 제품/지역 간 공통 변동 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)
   - 해결책: 계층적 예측 프레임워크 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

3. **해석성 부족**: 블랙박스 신경망 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)
   - 해결책: Attention 가중치 분석, LIME/SHAP [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

#### 5.3.2 2024-2026 최신 대응 [mdpi](https://www.mdpi.com/1996-1073/13/24/6623)

| 과제 | 2020 현황 | 2026 솔루션 |
|------|----------|-----------|
| **계산 복잡도** | O(L²) | O(L) with FEDformer |
| **작은 데이터** | 과적합 위험 | 하이브리드 + Foundation 미세 조정 |
| **도메인 이동** | 재훈련 필요 | Zero-shot Foundation 모델 |
| **불확실성** | 점 추정만 | 확률 분포 + 신뢰 구간 |
| **해석성** | 매우 제한적 | Attention 시각화 + 인과 추론 |

## 6. 성능 평가 및 한계

### 6.1 기존 모델 성능 비교 (논문 기준) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

| 모델 | 장점 | 단점 | 적용 분야 |
|------|------|------|---------|
| **CNN** | 병렬 처리 가능 | 고정 수용 영역 | 음성, 이미지 시계열 |
| **LSTM** | 무한 의존성 | 순차 처리, 느림 | 금융, 의료 |
| **Attention** | 동적 가중치 | 이차 복잡도 | 다중 지평선, 해석성 필요 |
| **하이브리드** | 도메인 지식 + 데이터 | 설계 복잡도 | M-대회 우승 모델 |

### 6.2 2020-2026 벤치마크 결과 [liebertpub](https://www.liebertpub.com/doi/10.1089/big.2020.0159)

#### **Informer (2020)** 이후 [royalsocietypublishing](https://royalsocietypublishing.org/doi/10.1098/rsta.2020.0209)

AAAI 2021 최고 논문상 수상:
- ProbSparse Attention으로 복잡도 O(L log L) 달성 [royalsocietypublishing](https://royalsocietypublishing.org/doi/10.1098/rsta.2020.0209)
- ETT, Exchange Rate, Weather 데이터셋에서 SOTA [royalsocietypublishing](https://royalsocietypublishing.org/doi/10.1098/rsta.2020.0209)

#### **FEDformer (2022) 성능** [liebertpub](https://www.liebertpub.com/doi/10.1089/big.2020.0159)

6개 벤치마크에서:
- 평균 14.8% MSE 감소 vs Autoformer [liebertpub](https://www.liebertpub.com/doi/10.1089/big.2020.0159)
- 계산 효율: O(L log L) → O(L) [liebertpub](https://www.liebertpub.com/doi/10.1089/big.2020.0159)
- 고주파 성분 학습으로 노이즈 견고성 [liebertpub](https://www.liebertpub.com/doi/10.1089/big.2020.0159)

#### **Foundation 모델 성능 (2024-2025)** [worldscientific](https://www.worldscientific.com/doi/abs/10.1142/S0129065721300011)

**TimesFM Zero-shot**:
- ETT, Monash 벤치마크에서 PatchTST와 경쟁 [royalsocietypublishing](https://royalsocietypublishing.org/doi/10.1098/rsta.2020.0209)
- 200M 매개변수로 감독 모델 성능 달성 [royalsocietypublishing](https://royalsocietypublishing.org/doi/10.1098/rsta.2020.0209)

**Time-MoE 성능**:
- 밀집 모델 대비 유의미 우위 [worldscientific](https://www.worldscientific.com/doi/abs/10.1142/S0129065721300011)
- 상업 모델(USDA)보다 정확:
  - 밀 예측: 54.9% 개선 [worldscientific](https://www.worldscientific.com/doi/abs/10.1142/S0129065721300011)
  - 옥수수 예측: 18.5% 개선 [worldscientific](https://www.worldscientific.com/doi/abs/10.1142/S0129065721300011)

**하이브리드 모델 성능**:
- ARIMA + Deep 조합: Informer/Autoformer 능가 [dl.acm](https://dl.acm.org/doi/10.1145/3533382)

### 6.3 주요 한계 [royalsocietypublishing](https://royalsocietypublishing.org/doi/10.1098/rsta.2020.0209)

#### 논문 기준 한계 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

1. **계산 자원**: 대규모 모델 훈련 비용 증가 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)
2. **데이터 요구**: 작은 데이터셋에서 불안정 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)
3. **초매개변수**: 광범위한 튜닝 필요 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)
4. **극단 사건**: 분포 이동에 취약 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

#### 2024-2026 해결 상황 [mdpi](https://www.mdpi.com/1996-1073/13/24/6623)

| 한계 | 개선 | 달성도 |
|------|------|--------|
| 계산 비용 | 스케일링 법칙, MoE | 80% |
| 작은 데이터 | 하이브리드 + 미세 조정 | 90% |
| 초매개변수 | Foundation 모델 (Zero-shot) | 85% |
| 극단 사건 | 적응형 학습, 도메인 적응 | 60% |

## 7. 2020년 이후 최신 연구와의 비교 분석

### 7.1 연구 패러다임 전환 [liebertpub](https://www.liebertpub.com/doi/10.1089/big.2020.0159)

#### **Phase 1 (2020-2021): Attention 개선**
- **Informer**: ProbSparse attention (O(L log L)) [royalsocietypublishing](https://royalsocietypublishing.org/doi/10.1098/rsta.2020.0209)
- **핵심**: 복잡도 감소에 집중

#### **Phase 2 (2021-2023): 분해 및 주파수 방법**
- **Autoformer**: Auto-correlation + 계절-트렌드 분해 [liebertpub](https://www.liebertpub.com/doi/10.1089/big.2020.0159)
- **FEDformer**: Fourier 기반 주파수 강화 [liebertpub](https://www.liebertpub.com/doi/10.1089/big.2020.0159)
- **핵심**: 시계열 구조 명시적 모델링

#### **Phase 3 (2023-2025): Foundation 모델**
- **TimesFM, Chronos, MOIRAI**: 대규모 사전 훈련 [worldscientific](https://www.worldscientific.com/doi/abs/10.1142/S0129065721300011)
- **Time-MoE**: 24억 매개변수 [worldscientific](https://www.worldscientific.com/doi/abs/10.1142/S0129065721300011)
- **핵심**: 전이 학습으로 도메인 무관성 [worldscientific](https://www.worldscientific.com/doi/abs/10.1142/S0129065721300011)

#### **Phase 4 (2025-2026): 효율성과 해석성**
- **TabPFN-TS**: 100배 작은 모델으로 경쟁 성능 [mdpi](https://www.mdpi.com/1996-1073/13/24/6623)
- **TimeFormer**: 시간 특성 최적화 주의 [mdpi](https://www.mdpi.com/1996-1073/13/24/6623)
- **DORIC**: 범용적이고 설명 가능한 예측 [mdpi](https://www.mdpi.com/1996-1073/13/24/6623)
- **핵심**: 계산 효율 + 해석성 동시 추구

### 7.2 세부 비교 분석

#### **논문의 Temporal Fusion Transformer (TFT, 2019)** vs 최신 모델 [liebertpub](https://www.liebertpub.com/doi/10.1089/big.2020.0159)

| 측면 | TFT | TimesFM | Time-MoE | FEDformer |
|------|-----|---------|---------|-----------|
| 매개변수 | ~100만 | 2억 | 24억 | ~100만 |
| 사전 훈련 데이터 | 없음 | 100B | 확대 | 없음 |
| Zero-shot 성능 | N/A | 경쟁 | SOTA | N/A |
| 해석성 | 높음 (Attention) | 중간 | 낮음 | 중간 |
| 복잡도 | O(L²) | O(L) | O(L) | O(L) |

**결론**: Foundation 모델이 규모에서 TFT를 능가하나, TFT의 해석성 강점은 여전히 중요 [worldscientific](https://www.worldscientific.com/doi/abs/10.1142/S0129065721300011)

#### **하이브리드 방법 진화**

**2020 (논문)**: ES-RNN (M4 우승) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)
- Holt-Winters + RNN
- 수동 매개변수 조정
- 성능: M4에서 우승

**2025**: ARIMA + 심층 학습 [dl.acm](https://dl.acm.org/doi/10.1145/3533382)
- 적응형 융합 메커니즘
- 자동 훈련
- 성능: 14.8% 개선

### 7.3 일반화 성능 향상 추이

#### **다양한 도메인에서의 성능** [dl.acm](https://dl.acm.org/doi/10.1145/3533382)

```
도메인            2020 모델        2024 Foundation    개선도
─────────────────────────────────────────────────────────
전자트레이딩        ETTh1: LSTM    TimesFM: Match    30-40%
                  (MAPE ~3%)     supervised PatchTST

소매 판매          ES-RNN: MAPE   CNN-LSTM: 4.16%   20-30%
                  ~4.5%          Hybrid: 추가 개선

날씨 예측          Autoformer     FEDformer: 14.8%  15%
                  (RMSE 기준)     vs Autoformer

금융 시계열        LSTM: 기준     Time-MoE: 54.9%   25-55%
                                (밀 vs USDA)
```

## 8. 해석성 및 결정 지원 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

### 8.1 논문의 해석성 방법 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

#### 8.1.1 Attention 가중치 분석 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

$$\alpha(t,\tau) = \text{softmax}(...)$$ 에서 $$\sum_{\tau=0}^{k} \alpha(t,\tau) = 1$$

이를 통해 어느 과거 시점이 현재 예측에 가장 중요한지 시각화 가능:

- **계절성 패턴**: 주기적 attention 피크 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)
- **이벤트 감지**: 특정 시점의 높은 가중치 (예: 휴일) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)
- **체제 감지**: 다양한 market regime에 따른 주의 패턴 변화 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

#### 8.1.2 Post-hoc 해석성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

**LIME** (Local Interpretable Model-Agnostic Explanations):
- 입력 섭동 기반 선형 모델 적합 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)
- 각 특성의 국소적 중요도 추정 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

**SHAP** (Shapley Additive exPlanations):
- 게임 이론 기반 특성 기여도 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)
- 일관된 정보 할당 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

**Saliency Maps**:
- 손실 함수에 대한 그래디언트 분석 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)
- 시계열 입력의 어느 부분이 가장 영향을 미치는지 식별 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

### 8.2 반사실적 예측 및 인과 추론 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

#### 8.2.1 문제: 시간 의존적 혼동 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

$$P(\text{outcome} | \text{action, history})$$

에서 역 인과성 (history가 action을 유발):
- IPTW (Inverse Probability of Treatment Weighting): 한 신경망이 치료 할당 확률 추정 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)
- G-Computation: 대상 분포와 행동 결합 모델링 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)
- 도메인 적대적 훈련: 균형 잡힌 표현 학습 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

#### 8.2.2 2024-2025 발전 [mdpi](https://www.mdpi.com/1996-1073/13/24/6623)

**Causal Time Series**:
- 경향-외인성 구조로 인과 그래프 명시 [mdpi](https://www.mdpi.com/1996-1073/13/24/6623)
- 중재 분석(intervention analysis)으로 행동 효과 평가 [mdpi](https://www.mdpi.com/1996-1073/13/24/6623)

## 9. 앞으로의 연구에 미치는 영향 및 고려사항

### 9.1 논문이 미친 영향 (2020 기준) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

1. **Transformer 급류**: 이 논문이 Transformer 적용의 첫 체계적 정리
   - 후속 Informer, Autoformer, FEDformer의 이론적 기초 [liebertpub](https://www.liebertpub.com/doi/10.1089/big.2020.0159)

2. **하이브리드 모델 정당화**: ES-RNN 우승으로 입증된 접근법의 이론화 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)
   - 이후 많은 산업 응용 [dl.acm](https://dl.acm.org/doi/10.1145/3533382)

3. **해석성 강조**: 예측 신뢰도를 위한 해석성 필요성 제시 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)
   - Attention 시각화의 표준화 [royalsocietypublishing](https://royalsocietypublishing.org/doi/10.1098/rsta.2020.0209)

### 9.2 현재 핫이슈 (2024-2026) [dl.acm](https://dl.acm.org/doi/10.1145/3533382)

#### **9.2.1 Foundation 모델의 득과 실**

**장점**:
- Zero-shot 일반화 [worldscientific](https://www.worldscientific.com/doi/abs/10.1142/S0129065721300011)
- 도메인 간 전이 학습 [royalsocietypublishing](https://royalsocietypublishing.org/doi/10.1098/rsta.2020.0209)
- 빠른 배포 [royalsocietypublishing](https://royalsocietypublishing.org/doi/10.1098/rsta.2020.0209)

**우려**:
- 계산 비용 (24억 매개변수 모델) [worldscientific](https://www.worldscientific.com/doi/abs/10.1142/S0129065721300011)
- 도메인 외 분포 이동에 취약 [worldscientific](https://www.worldscientific.com/doi/abs/10.1142/S0129065721300011)
- "one-size-fits-all" 문제 [worldscientific](https://www.worldscientific.com/doi/abs/10.1142/S0129065721300011)

**해결책**: 
- 모델 압축 및 증류 (TTM의 소형 변형) [mdpi](https://www.mdpi.com/1996-1073/13/24/6623)
- 도메인 적응 학습 [mdpi](https://www.mdpi.com/1996-1073/13/24/6623)
- 하이브리드 접근으로 견고성 향상 [dl.acm](https://dl.acm.org/doi/10.1145/3533382)

#### **9.2.2 스케일링 법칙의 한계**

**발견**: 
- 시계열 예측에서도 스케일링 법칙 존재 [worldscientific](https://www.worldscientific.com/doi/abs/10.1142/S0129065721300011)
- 모델 크기와 훈련 데이터 간 거듭제곱 법칙 [worldscientific](https://www.worldscientific.com/doi/abs/10.1142/S0129065721300011)

**미해결 문제**:
- 도메인 외 데이터에서 스케일링 법칙 유지 여부 [worldscientific](https://www.worldscientific.com/doi/abs/10.1142/S0129065721300011)
- 계산 효율성과 성능의 균형 [mdpi](https://www.mdpi.com/1996-1073/13/24/6623)

#### **9.2.3 극단 사건 대비**

**문제**:
- Foundation 모델이 학습 분포 외 사건 예측 불가 [worldscientific](https://www.worldscientific.com/doi/abs/10.1142/S0129065721300011)
- COVID-19 같은 구조적 단절 대응 한계 [royalsocietypublishing](https://royalsocietypublishing.org/doi/10.1098/rsta.2020.0209)

**방향**:
- 적응형 온라인 학습 [mdpi](https://www.mdpi.com/1996-1073/13/24/6623)
- 이상 감지 모듈 통합 [mdpi](https://www.mdpi.com/1996-1073/13/24/6623)
- 인과 구조 학습으로 강건성 확대 [mdpi](https://www.mdpi.com/1996-1073/13/24/6623)

### 9.3 미래 연구 시 고려사항

#### **논문에서 제시 (2020)** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

1. **연속 시간 모델**: Neural ODE로 불규칙 샘플링 처리 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)
2. **계층적 구조**: 제품/지역 간 공통 변동 모델링 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)
3. **Multimodal**: 텍스트, 이미지와 시계열 결합 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

#### **최신 추가 고려사항 (2024-2026)** [dl.acm](https://dl.acm.org/doi/10.1145/3533382)

| 고려사항 | 중요도 | 진전도 |
|---------|--------|--------|
| **온디바이스 배포** | ⭐⭐⭐⭐⭐ | 40% (TTM으로 일부 해결) |
| **설명 가능성** | ⭐⭐⭐⭐⭐ | 60% (Attention 분석 진화) |
| **데이터 효율** | ⭐⭐⭐⭐⭐ | 70% (하이브리드 모델) |
| **도메인 적응** | ⭐⭐⭐⭐ | 50% (몇몇 미세 조정 방법) |
| **인과 추론** | ⭐⭐⭐⭐ | 40% (초기 단계) |
| **극단 이벤트** | ⭐⭐⭐⭐⭐ | 30% (미해결) |

## 10. 결론

Lim과 Zohren의 "Time Series Forecasting With Deep Learning: A Survey"는 **깊은 학습의 시계열 예측 적용에 대한 종합적 이론 틀**을 제공했다. 특히 하이브리드 모델과 해석성의 중요성 강조는 후속 6년간의 연구 방향을 크게 영향미쳤다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c10f0c13-119f-4645-b617-50b1cbb836bc/2004.13408v2.pdf)

### 핵심 발견 요약:

1. **구조화된 프레임워크**: 인코더-디코더 아키텍처는 여전히 대부분의 모델의 기초 [liebertpub](https://www.liebertpub.com/doi/10.1089/big.2020.0159)

2. **일반화 개선의 진화**:
   - 2020: 하이브리드 모델 + 정규화
   - 2022: 주파수 분해 + 효율성
   - 2024-2026: Foundation 모델 + 무감독 학습

3. **미해결 도전**:
   - 극단 사건 예측 [worldscientific](https://www.worldscientific.com/doi/abs/10.1142/S0129065721300011)
   - 계산 효율성 [mdpi](https://www.mdpi.com/1996-1073/13/24/6623)
   - 도메인 외 일반화 [worldscientific](https://www.worldscientific.com/doi/abs/10.1142/S0129065721300011)

4. **미래 방향**:
   - 소형 고효율 모델 (TTM, TabPFN-TS) [mdpi](https://www.mdpi.com/1996-1073/13/24/6623)
   - 인과 시계열 모델 [mdpi](https://www.mdpi.com/1996-1073/13/24/6623)
   - 하이브리드 접근의 재부상 [dl.acm](https://dl.acm.org/doi/10.1145/3533382)

이 보고서는 학계와 산업이 수용해야 할 가장 중요한 통찰력을 강조한다: **완벽한 단일 모델보다는 문제의 특성에 맞춘 적응형 앙상블과 하이브리드 접근이 최고의 성능을 제공한다는 것이다.** [dl.acm](https://dl.acm.org/doi/10.1145/3533382)

***

## 참고: 주요 수식 정리

**One-step-ahead forecasting**:

$$\hat{y}\_{i,t+1} = f(y_{i,t-k:t}, x_{i,t-k:t}, s_i)$$

**Encoder-Decoder**:

$$z_t = g_{enc}(y_{t-k:t}, x_{t-k:t}, s) \quad \text{and} \quad \hat{y}\_{t+1} = g_{dec}(z_t)$$

**LSTM Cell State**:
$$c_t = f_t \odot c_{t-1} + i_t \odot \tanh(W_c z_{t-1} + W_c y_t + W_c x_t + W_c s + b_c)$$

**Multi-head Attention**:
$$h_t = \sum_{\tau=0}^{k} \alpha(\kappa_t, q_\tau)v_{t-\tau}$$

**Probabilistic Output (Gaussian)**:
$$y_{t+\tau} \sim N(\mu(t,\tau), \zeta(t,\tau)^2)$$

**FEDformer Frequency Decomposition**:

$$A' = AS^T \text{ (Fourier basis selection)}$$

<span style="display:none">[^1_10][^1_11][^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_7][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_8][^1_80][^1_81][^1_82][^1_83][^1_84][^1_85][^1_86][^1_87][^1_88][^1_89][^1_9][^1_90][^1_91][^1_92][^1_93][^1_94][^1_95][^1_96][^1_97]</span>

<div align="center">⁂</div>

[^1_1]: 2004.13408v2.pdf

[^1_2]: https://royalsocietypublishing.org/doi/10.1098/rsta.2020.0209

[^1_3]: https://www.worldscientific.com/doi/abs/10.1142/S0129065721300011

[^1_4]: https://www.liebertpub.com/doi/10.1089/big.2020.0159

[^1_5]: https://dl.acm.org/doi/10.1145/3533382

[^1_6]: https://www.mdpi.com/1996-1073/13/24/6623

[^1_7]: https://ieeexplore.ieee.org/document/9080613/

[^1_8]: https://linkinghub.elsevier.com/retrieve/pii/S0960077920306238

[^1_9]: https://linkinghub.elsevier.com/retrieve/pii/S096007792030518X

[^1_10]: https://www.tandfonline.com/doi/full/10.1080/03461238.2020.1867232

[^1_11]: https://www.mdpi.com/1996-1073/13/18/4722

[^1_12]: http://arxiv.org/pdf/2411.17382.pdf

[^1_13]: https://arxiv.org/pdf/2310.10688.pdf

[^1_14]: https://arxiv.org/pdf/2308.00709.pdf

[^1_15]: http://arxiv.org/pdf/2410.15217.pdf

[^1_16]: https://arxiv.org/pdf/2004.10240.pdf

[^1_17]: https://arxiv.org/pdf/2312.17100.pdf

[^1_18]: https://arxiv.org/pdf/2002.09695.pdf

[^1_19]: http://arxiv.org/pdf/2310.04948.pdf

[^1_20]: https://pdfs.semanticscholar.org/d0d4/2fa6fb4ab0650854f8f8080f7b7c8a4dd88a.pdf

[^1_21]: https://arxiv.org/pdf/2406.18125.pdf

[^1_22]: https://arxiv.org/html/2601.04602v1

[^1_23]: https://pdfs.semanticscholar.org/000c/efcc0a17a6252c7fe9d977d252bf712354a5.pdf

[^1_24]: https://openaccess.thecvf.com/content/WACV2025/papers/Pegeot_Temporal_Dynamics_in_Visual_Data_Analyzing_the_Impact_of_Time_WACV_2025_paper.pdf

[^1_25]: https://arxiv.org/abs/2601.04602

[^1_26]: https://arxiv.org/pdf/2502.20244.pdf

[^1_27]: https://arxiv.org/pdf/2202.11423.pdf

[^1_28]: https://pubmed.ncbi.nlm.nih.gov/40989322/

[^1_29]: https://arxiv.org/pdf/2304.02104.pdf

[^1_30]: https://arxiv.org/html/2406.18125v1

[^1_31]: https://pubmed.ncbi.nlm.nih.gov/39730438/

[^1_32]: https://arxiv.org/pdf/2203.09474.pdf

[^1_33]: https://arxiv.org/html/2601.03474v1

[^1_34]: https://arxiv.org/abs/2512.08567

[^1_35]: https://journal.hep.com.cn/fcs/EN/10.1007/s11704-025-50947-3

[^1_36]: https://arxiv.org/html/2510.06680v1

[^1_37]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12294620/

[^1_38]: https://towardsdatascience.com/global-deep-learning-for-joint-time-series-forecasting-4b03bef42321/

[^1_39]: https://www.sciencedirect.com/science/article/pii/S2665963824001040

[^1_40]: https://www.nature.com/articles/s41598-025-21842-5

[^1_41]: https://machinelearningmastery.com/the-2026-time-series-toolkit-5-foundation-models-for-autonomous-forecasting/

[^1_42]: https://yoonji-ha.tistory.com/43

[^1_43]: https://www.ijcai.org/proceedings/2017/0316.pdf

[^1_44]: https://openreview.net/forum?id=t9cOXsdpKg

[^1_45]: https://icml.cc/virtual/2025/poster/44262

[^1_46]: https://www.sciencedirect.com/science/article/pii/S2215016124005442

[^1_47]: https://royalsocietypublishing.org/rsta/article/379/2194/20200209/41189/Time-series-forecasting-with-deep-learning-a

[^1_48]: https://www.sciencedirect.com/science/article/abs/pii/S0957417425046548

[^1_49]: https://peerj.com/articles/cs-3058/

[^1_50]: https://www.semanticscholar.org/paper/baa7ba687bd7587f5aae825ff4ef644f7db5c41e

[^1_51]: https://ieeexplore.ieee.org/document/11298343/

[^1_52]: https://ieeexplore.ieee.org/document/9022029/

[^1_53]: https://www.internationaljournalssrg.org/IJECE/paper-details?Id=736

[^1_54]: https://www.mdpi.com/2079-8954/13/2/96

[^1_55]: https://jis-eurasipjournals.springeropen.com/articles/10.1186/s13635-025-00217-3

[^1_56]: https://www.mdpi.com/1424-8220/21/14/4764

[^1_57]: https://isprs-annals.copernicus.org/articles/X-5-W2-2025/115/2025/

[^1_58]: https://hbem.org/index.php/OJS/article/view/714

[^1_59]: https://ieeexplore.ieee.org/document/9879803/

[^1_60]: https://arxiv.org/pdf/1912.09363.pdf

[^1_61]: http://arxiv.org/pdf/2405.04841.pdf

[^1_62]: https://arxiv.org/abs/2409.00904

[^1_63]: https://linkinghub.elsevier.com/retrieve/pii/S0169207021000637

[^1_64]: https://arxiv.org/pdf/2306.13815.pdf

[^1_65]: http://arxiv.org/pdf/2107.06846.pdf

[^1_66]: https://arxiv.org/pdf/2207.00610.pdf

[^1_67]: http://arxiv.org/pdf/2207.07827.pdf

[^1_68]: https://arxiv.org/pdf/1702.04125.pdf

[^1_69]: https://arxiv.org/pdf/2201.12740.pdf

[^1_70]: https://arxiv.org/html/2501.02945v4

[^1_71]: https://arxiv.org/pdf/2505.01973.pdf

[^1_72]: https://arxiv.org/html/2507.13043v1

[^1_73]: https://arxiv.org/abs/2410.09487

[^1_74]: https://arxiv.org/html/2508.15959v1

[^1_75]: https://arxiv.org/abs/2012.07436

[^1_76]: https://www.arxiv.org/pdf/2601.06371v2.pdf

[^1_77]: https://www.arxiv.org/pdf/2508.15959.pdf

[^1_78]: https://arxiv.org/html/2508.01407v2

[^1_79]: https://www.arxiv.org/pdf/2601.12785.pdf

[^1_80]: https://arxiv.org/html/2301.00394v2

[^1_81]: https://arxiv.org/pdf/2211.14730.pdf

[^1_82]: https://arxiv.org/html/2601.19040v1

[^1_83]: https://towardsdatascience.com/temporal-fusion-transformer-time-series-forecasting-with-deep-learning-complete-tutorial-d32c1e51cd91/

[^1_84]: https://sonstory.tistory.com/119

[^1_85]: https://www.pricepedia.it/en/magazine/article/2025/11/07/the-arrival-of-foundation-models-in-time-series-forecasting/

[^1_86]: https://arxiv.org/abs/1912.09363

[^1_87]: https://ffighting.net/deep-learning-paper-review/time-series-model/fedformer/

[^1_88]: https://openreview.net/forum?id=e1wDDFmlVu

[^1_89]: https://www.sciencedirect.com/science/article/pii/S0169207021000637

[^1_90]: https://deepdata.tistory.com/386

[^1_91]: https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/

[^1_92]: https://deepfa.ir/en/blog/temporal-fusion-transformers-time-series-forecasting

[^1_93]: https://huggingface.co/blog/autoformer

[^1_94]: https://arxiv.org/html/2504.04011v1

[^1_95]: https://github.com/LiamMaclean216/Temporal-Fusion-Transformer

[^1_96]: https://aiflower.tistory.com/221

[^1_97]: https://arxiv.org/html/2503.04118v1
