# GluonTS: Probabilistic Time Series Models in Python

### 요약

Amazon Web Services의 연구팀이 개발한 **GluonTS: Probabilistic Time Series Models in Python** (2019)는 시계열 모델링과 예측을 위한 첫 번째 현대적 딥러닝 기반 통합 라이브러리이다. 이 논문은 깊은 신경망을 시계열 분석에 적용할 때 필요한 표준화된 인프라 부재를 해결하였으며, 동시에 확률적 모델링과 전통적 통계 기법의 융합을 가능하게 한다. GluonTS는 모듈성, 확장성, 재현 가능성이라는 세 가지 설계 원칙을 중심으로 구축되었으며, DeepAR, 상태공간 모델, Transformer 등 다양한 최신 아키텍처를 제공한다.

***

### 1. 핵심 주장과 문제 제시

**해결하고자 한 문제:**
전통적으로 시계열 모델링은 개별 시계열마다 독립적인 모수를 추정하는 지역 모델(local models) 중심이었다. 그러나 최근 깊은 신경망의 발전으로 대규모 시계열 컬렉션에서 단일 글로벌 모델을 학습하여 모든 시계열에서 모수를 공유하는 방식이 우수함이 입증되었다. 그럼에도 불구하고, 이러한 딥러닝 기반 시계열 모델링을 체계적으로 개발하고 벤치마크할 수 있는 통합 라이브러리가 존재하지 않았다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/02359309-88cd-4b2c-95ac-abec700a6031/1906.05264v2.pdf)

**주요 기여:**
- 시계열 모델링을 위한 첫 번째 현대 딥러닝 기반 통합 프레임워크 제공 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/02359309-88cd-4b2c-95ac-abec700a6031/1906.05264v2.pdf)
- 확률적 모델링(probabilistic modeling)과 신경망 아키텍처의 자유로운 결합 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/02359309-88cd-4b2c-95ac-abec700a6031/1906.05264v2.pdf)
- 11개 공개 데이터셋에 걸친 대규모 벤치마크 실험 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/02359309-88cd-4b2c-95ac-abec700a6031/1906.05264v2.pdf)
- 재현 가능한 과학적 실험을 위한 완전한 구성 추적 시스템 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/02359309-88cd-4b2c-95ac-abec700a6031/1906.05264v2.pdf)

***

### 2. 해결하고자 하는 문제와 제안 방법

#### 2.1 형식적 문제 정의

GluonTS가 다루는 세 가지 핵심 시계열 문제는 다음과 같이 정의된다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/02359309-88cd-4b2c-95ac-abec700a6031/1906.05264v2.pdf)

**예측(Forecasting):**

$$p(z_{i,T_i+1:T_i+H}|z_{i,1:T_i}, x_{i,1:T_i}, \theta)$$

여기서 $Z = \{z_{i,1:T_i}\}\_{i=1}^N$ 은 $N$개의 시계열이고, $X = \{x_{i,1:T_i}\}_{i=1}^N$은 공변량 벡터이며, $H$는 예측 지평이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/02359309-88cd-4b2c-95ac-abec700a6031/1906.05264v2.pdf)

**평활(Smoothing) 또는 결측값 대체:**

$$p(z_{i,j_1:j_k}|z_{i,k_1:k_l}, x_{i,k_1:k_l}, \theta)$$

여기서 $\{j_1, \ldots, j_k\}$는 결측값의 인덱스이고 $\{k_1, \ldots, k_l\}$는 관측된 값의 인덱스이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/02359309-88cd-4b2c-95ac-abec700a6031/1906.05264v2.pdf)

**이상 탐지(Anomaly Detection):**
누적분포함수(CDF)로부터 p-값을 계산하거나 음의 로그 우도(log-likelihood)를 이용하여 비정상적 시점을 식별한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/02359309-88cd-4b2c-95ac-abec700a6031/1906.05264v2.pdf)

#### 2.2 모델 구조와 아키텍처

**2.2.1 생성 모델 (Generative Models)**

생성 모델은 $p(z_{i,1:T_i}|x_{i,1:T_i}, \theta)$를 직접 모델링한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/02359309-88cd-4b2c-95ac-abec700a6031/1906.05264v2.pdf)

**상태공간 모델 (State Space Models, SSM):**
$$l_t = F_t l_{t-1} + g_t \epsilon_t, \quad \epsilon_t \sim N(0, 1)$$
$$z_t = y_t + \eta_t, \quad y_t = a_t^T l_{t-1} + b_t, \quad \eta_t \sim N(0, \sigma_\eta^2)$$

상태공간 모델은 숨겨진 상태 $l_t \in \mathbb{R}^D$를 통해 수준(level), 추세(trend), 계절성(seasonality)을 인코딩한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/02359309-88cd-4b2c-95ac-abec700a6031/1906.05264v2.pdf)

**심층 상태공간 모델 (Deep State Space Models, DeepState):**
$$h_{i,t} = h(h_{i,t-1}, x_{i,t}; \theta)$$
$$p(z_{i,1:T_i}|x_{i,1:T_i}, \theta) = p_{SS}(z_{i,1:T_i}|\theta_{i,1:T_i}, \theta)$$

RNN(특히 LSTM)을 사용하여 공변량 $x_{i,t}$를 상태공간 모델의 시간 변화 매개변수 $\theta_{i,t}$로 매개변수화한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/02359309-88cd-4b2c-95ac-abec700a6031/1906.05264v2.pdf)

**2.2.2 판별 모델 (Discriminative Models)**

판별 모델은 조건부 분포 $p(z_{i,T_i+1:T_i+H}|z_{i,1:T_i}, x_{i,1:T_i}, \theta)$를 직접 모델링하며, 생성 모델보다 유연하다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/02359309-88cd-4b2c-95ac-abec700a6031/1906.05264v2.pdf)

**Sequence-to-Sequence 모델:**
- 인코더는 컨텍스트 $z_{i,1:T_i}, x_{i,1:T_i}$ 를 숨겨진 상태로 인코딩 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/02359309-88cd-4b2c-95ac-abec700a6031/1906.05264v2.pdf)
- 디코더는 숨겨진 상태와 미래 특징 $x_{i,T_i+1:T_i+H}$ 를 결합하여 예측 생성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/02359309-88cd-4b2c-95ac-abec700a6031/1906.05264v2.pdf)
- 장점: 미래에 사용 불가능한 공변량 사용 가능, 오류 누적 방지 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/02359309-88cd-4b2c-95ac-abec700a6031/1906.05264v2.pdf)

**신경 정량 회귀 (Neural Quantile Regression):**

$$\hat{q}_\tau = \text{decoder}(z_{i,1:T_i}, x_{i,1:T_i}; \theta)$$

Quantile loss로 훈련: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/02359309-88cd-4b2c-95ac-abec700a6031/1906.05264v2.pdf)

$$L_\tau(y, \hat{q}\_\tau) = (\tau - \mathbb{1}_{y < \hat{q}_\tau})(y - \hat{q}_\tau)$$

**Transformer 아키텍처:**
Self-attention 메커니즘만 사용하여 모든 시간 단계 간 상호작용을 모델링한다. 분포를 직접 학습하기 위해 2.3절의 분포 컴포넌트를 적용한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/02359309-88cd-4b2c-95ac-abec700a6031/1906.05264v2.pdf)

**2.2.3 자동회귀 모델 (Auto-regressive Models)**

$$p(z_{i,T_i+1:T_i+H}|z_{i,1:T_i}, x_{i,1:T_i}) = \prod_{t=1}^H p(z_{i,T_i+t}|z_{i,1:T_i+t-1}, x_{i,1:T_i+t})$$

**DeepAR:**
- LSTM/GRU 셀 사용 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/02359309-88cd-4b2c-95ac-abec700a6031/1906.05264v2.pdf)
- 매개변수적 분포 또는 분위수 함수의 유연한 매개변수화 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/02359309-88cd-4b2c-95ac-abec700a6031/1906.05264v2.pdf)
- 관련 지연값(lags)을 입력으로 포함 (예: 시간별 데이터의 경우 1, 24, 48, 168 지연) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/02359309-88cd-4b2c-95ac-abec700a6031/1906.05264v2.pdf)

**WaveNet:**
- 확장된 인과 합성곱(dilated causal convolutions) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/02359309-88cd-4b2c-95ac-abec700a6031/1906.05264v2.pdf)
- 분산 신호는 소프트맥스 분포로, 연속 신호는 임의의 분포로 모델링 가능 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/02359309-88cd-4b2c-95ac-abec700a6031/1906.05264v2.pdf)

**2.2.4 확률 분포 컴포넌트**

GluonTS는 다양한 확률 분포을 지원한다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/02359309-88cd-4b2c-95ac-abec700a6031/1906.05264v2.pdf)

- **매개변수적 분포**: 가우시안, Student-t, 감마, 음이항
- **변환 분포**: 미분 가능한 전단사 변환 활용

$$y_t = T(u_t; \phi), \quad u_t \sim P_0(\cdot; \psi)$$

전단사 변환의 야코비안이 계산 가능하고 학습 가능한 매개변수에 의존할 수 있다. 박스-콕스 변환도 포함: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/02359309-88cd-4b2c-95ac-abec700a6031/1906.05264v2.pdf)

$$y_t^{(\lambda)} = \begin{cases} \frac{y_t^{\lambda} - 1}{\lambda} & \lambda \neq 0 \\ \ln(y_t) & \lambda = 0 \end{cases}$$

- **혼합 분포**: 기본 분포들의 임의의 혼합

***

### 3. 모델 구조의 상세 분석

#### 3.1 라이브러리 아키텍처

| 컴포넌트 | 설명 | 역할 |
|---------|------|------|
| **Data I/O** | DatasetRepository, 합성 데이터 생성기 | 데이터 로드 및 전처리 |
| **Transformation Pipeline** | Box-Cox, 분할, 패딩, 결측값 표시 | 시계열 특정 특징 공학 |
| **Distribution Output** | 분포 추상화, 변환, 혼합 | 확률적 예측 생성 |
| **Forecast Object** | 표본 경로 또는 분위수 저장 | 통일된 인터페이스 |
| **Evaluator** | MASE, MAPE, sMAPE, Weighted QuantileLoss | 모델 평가 및 비교 |
| **Backtest** | Train-validation-test 분할 | 재현 가능한 실험 |

#### 3.2 모델 훈련 및 예측 파이프라인

**손실 함수:**
GluonTS는 다양한 손실 함수를 지원한다:

1. **최대 우도 추정**: 생성 모델의 표준 접근법
2. **분위수 손실**: 

$$\mathcal{L}_\tau = (\tau - \mathbb{1}_{y < \hat{q}_\tau})(y - \hat{q}_\tau)$$

3. **CRPS (Continuous Ranked Probability Score)**:

$$\text{CRPS}(F, y) = \int_{-\infty}^{\infty} (F(z) - \mathbb{1}_{z \geq y})^2 dz$$

#### 3.3 예측 객체와 평가

- **표본 경로 기반**: 자동회귀 모델이 생성한 다수의 샘플 경로 (예: 1,000개)
- **분위수 기반**: 특정 분위수 집합 (예: 0.1, 0.5, 0.9)
- **공통 인터페이스**: `.quantile()` 메서드로 임의의 분위수 추출 가능 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/02359309-88cd-4b2c-95ac-abec700a6031/1906.05264v2.pdf)

***

### 4. 성능 향상과 벤치마크

#### 4.1 벤치마크 결과

**테스트 데이터셋 (11개):**
- SP500-returns, electricity, M4 (6개 빈도), parts, traffic, wiki10k

**평가 지표:**
- CRPS (Continuous Ranked Probability Score) - 확률적 예측 평가 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/02359309-88cd-4b2c-95ac-abec700a6031/1906.05264v2.pdf)

| 모델 | 설명 | 특징 |
|------|------|------|
| **Auto-ARIMA** | 고전적 통계 방법 | 데이터셋별 강력한 성능 |
| **Prophet** | Facebook의 시계열 라이브러리 | 장기 의존성 포착 제한 |
| **NPTS** | 비모수적 기준 모델 | 단순성과 해석성 |
| **Transformer** | Self-attention 기반 | 복잡한 패턴 학습 가능 |
| **CNN-QR** | 확장된 인과 합성곱 | 신경망 기반 정량 회귀 |
| **DeepAR** | LSTM 자동회귀 | 다양한 데이터셋에서 경쟁력 있는 성능 |

**핵심 발견:** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/02359309-88cd-4b2c-95ac-abec700a6031/1906.05264v2.pdf)
- 신경망 기반 모델이 ARIMA/ETS와 경쟁력 있음
- 모든 데이터셋을 지배하는 단일 최적 모델 없음
- 데이터셋 특성에 따라 최적 모델이 다름 (유연한 툴킷 필수성 증명)

#### 4.2 이상 탐지 응용

DeepAR을 이용한 이상 탐지: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/02359309-88cd-4b2c-95ac-abec700a6031/1906.05264v2.pdf)
1. 예측된 분포의 CDF에서 p-값 계산
2. 음의 로그 우도로 비정상성 정의
3. 훈련 데이터의 로그 우도 분포에서 백분위수 설정 (99, 99.9, 99.99)
4. 테스트 시점의 로그 우도와 비교

***

### 5. 일반화 성능 향상과 관련된 분석

#### 5.1 글로벌 모델의 장점

**전통적 접근 vs. 글로벌 모델:**

전통적 지역 모델:
$$\theta_i = \arg\max_{\theta_i} p(z_{i,1:T_i}|\theta_i)$$

글로벌 신경망 모델 (GluonTS):
$$\theta = \arg\max_{\theta} \sum_{i=1}^N p(z_{i,1:T_i}|f(x_{i,1:T_i}; \theta))$$

글로벌 모델의 이점: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/02359309-88cd-4b2c-95ac-abec700a6031/1906.05264v2.pdf)
1. **데이터 공유**: 모든 시계열의 정보를 활용하여 모수 추정
2. **고차 특징 추출**: 깊은 신경망의 표현 학습 능력
3. **도메인 간 지식 전이**: 다양한 시계열 유형에서 공통 패턴 학습
4. **계산 효율성**: 단일 모델로 모든 시계열 처리

#### 5.2 계절성 및 트렌드 처리

**특징 공학:**
- 시간 변화 공변량 (예: 가격, 프로모션)
- 시간 독립 공변량 (예: 제품 카테고리, 브랜드)
- 자동 지연값 선택 (시간 단위별 관련 지연)

**변환:**
- Box-Cox 정규화
- 계절성-트렌드 분해 (향후 연구)

#### 5.3 확률적 예측의 이점

GluonTS의 확률적 모델링: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/02359309-88cd-4b2c-95ac-abec700a6031/1906.05264v2.pdf)
1. **불확실성 정량화**: 점 추정이 아닌 전체 분포 제공
2. **의사결정**: 위험 회피적(risk-averse) 및 위험 추구적(risk-seeking) 정책 모두 지원
3. **합계 예측**: 개별 표본 경로를 합산하여 집계 통계 계산 가능

예: 재고 예측에서 특정 신뢰도 수준의 안전 재고 결정 가능

***

### 6. 한계와 제약

#### 6.1 모델 선택의 어려움

**한계:**
벤치마크 결과에서 모든 데이터셋에서 최상의 성능을 보이는 단일 모델이 없음. 이는 다음을 의미한다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/02359309-88cd-4b2c-95ac-abec700a6031/1906.05264v2.pdf)
- 각 문제에 대해 여러 모델을 시도해야 함
- 하이퍼파라미터 튜닝 비용 증가

#### 6.2 데이터 요구사항

**제약:**
- 신경망 기반 모델은 충분한 훈련 데이터 필요 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/02359309-88cd-4b2c-95ac-abec700a6031/1906.05264v2.pdf)
- 지역 모델(ARIMA)이 작은 데이터셋에서 더 나을 수 있음

#### 6.3 해석 가능성

**문제:**
- 신경망 모델의 블랙박스 특성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/02359309-88cd-4b2c-95ac-abec700a6031/1906.05264v2.pdf)
- Attention 가중치 시각화 가능하지만 완전한 이해 어려움

#### 6.4 추가 연구 필요 분야

논문에서 명시한 한계: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/02359309-88cd-4b2c-95ac-abec700a6031/1906.05264v2.pdf)
1. **절제 연구(Ablation Studies)**: 각 컴포넌트의 기여도 분석
2. **통제 데이터 실험**: 알려진 패턴으로 모델 성능 평가
3. **이상 탐지와 예측 정확도의 관계**: 정량적 분석 필요
4. **장기 예측 안정성**: 외삽 능력 개선

***

### 7. 2020년 이후 최신 연구 비교 분석

#### 7.1 Foundation Models의 등장 (2023-2025)

| 모델 | 출시 | 주요 특징 | 일반화 성능 |
|------|------|---------|----------|
| **TimeGPT** | 2023 | Transformer 기반, 다양한 데이터로 사전학습 | Zero-shot 성능 우수 |
| **TimesFM (Google)** | 2024 | 패치 기반 토큰화, 200M 파라미터 | 대규모 데이터셋에서 경쟁력 |
| **TTM** | 2024 | Task-specific 변형, 컴팩트 모델 집합 | 특정 작업에 최적화 |
| **TimelyGPT** | 2023 | 건강관리 시계열 특화, 외삽 가능한 임베딩 | 불규칙 샘플링 데이터에 강함 |

**핵심 특징:** [arxiv](https://arxiv.org/html/2504.04011v1)
- **대규모 사전학습**: 수십억 개의 시계열 데이터 활용
- **Zero-shot 성능**: 미세조정 없이도 다양한 작업에 적용 가능
- **전이 학습**: 작은 데이터셋에서 미세조정으로 빠른 적응 [forecasters](https://forecasters.org/wp-content/uploads/TransferL_Progress2_2022-SAS_Kin-G.-Olivares.pdf)

#### 7.2 Transformer 아키텍처 발전

| 모델 | 년도 | 개선사항 | 성능 향상 |
|------|------|---------|----------|
| **Informer** | 2020 | ProbSparse Attention (O(L log L)) | 장시간 의존성 포착 |
| **PatchTST** | 2023 | 패치 임베딩, 채널 독립성 | 다양한 길이 시계열 처리 |
| **ETSFormer** | 2022 | 지수 평활 + Transformer | 통계적 기초와 심층학습 결합 |
| **iTransformer** | 2023 | 채널 중심 처리 | 다변량 의존성 향상 |
| **FreEformer** | 2025 | 주파수 도메인 강화 | 다양한 도메인에서 최고 성능 [arxiv](https://arxiv.org/abs/2501.13989) |

**성능 비교 (2025년 최신 벤치마크):** [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12453695/)
- **iTransformer**: RMSE = 1.43, MedianAbsE = 1.21 (최고 수준)
- **PatchTST**: MAE = 1.24 (안정적 성능)
- **Informer**: 장기 패턴 포착에 우수
- **TCN/BiTCN**: 단기 예측에는 한계

#### 7.3 일반화 성능 향상 메커니즘

**1. 전이 학습 (Transfer Learning):** [forecasters](https://forecasters.org/wp-content/uploads/TransferL_Progress2_2022-SAS_Kin-G.-Olivares.pdf)

매개변수 기반 전이: [forecasters](https://forecasters.org/wp-content/uploads/TransferL_Progress2_2022-SAS_Kin-G.-Olivares.pdf)
$$\theta_l^{(T)} = \theta_l^{(S)} + \Delta\theta_l$$

소스 데이터셋에서 사전학습된 모수 $\theta^{(S)}$를 목표 데이터셋에서 미세조정:
- 영점 샷(Zero-shot): 미세조정 없이 직접 적용
- 미세조정(Fine-tuning): 소수의 훈련 데이터로 빠른 적응 [forecasters](https://forecasters.org/wp-content/uploads/TransferL_Progress2_2022-SAS_Kin-G.-Olivares.pdf)

**효과:** [forecasters](https://forecasters.org/wp-content/uploads/TransferL_Progress2_2022-SAS_Kin-G.-Olivares.pdf)
- 계산 비용 대폭 감소
- 데이터 부족 시나리오에서 성능 향상
- 도메인 간 지식 공유

**2. 멀티모달 확장:** [arxiv](https://arxiv.org/pdf/2406.08627.pdf)

Time-MMD 데이터셋 (시간 + 텍스트): [arxiv](https://arxiv.org/pdf/2406.08627.pdf)
- 수치 시계열 + 관련 텍스트 정보 통합
- 독립적 모달리티 모델링 후 가중치 결합: [arxiv](https://arxiv.org/pdf/2406.08627.pdf)

$$\hat{z}_{t+1} = w_1 \cdot f_{TS}(z_{1:t}) + w_2 \cdot f_{LLM}(text_{1:t})$$

**성능 개선:** [arxiv](https://arxiv.org/pdf/2406.08627.pdf)
- 평균 MSE 15% 감소, 텍스트 풍부한 도메인에서 40% 감소
- 95% 이상의 실험에서 단일모달 모델 능가

**3. 메타 러닝 (Meta-Learning):** [arxiv](https://arxiv.org/pdf/2401.13968.pdf)

MANTRA (Meta-Transformer Networks): [arxiv](https://arxiv.org/pdf/2401.13968.pdf)
- 빠른 학습자(Fast Learners): 데이터 분포의 다양한 측면 학습
- 느린 학습자(Slow Learner): 적응 표현 제공
- 동적 환경에서 변화 적응 [arxiv](https://arxiv.org/pdf/2401.13968.pdf)

**4. 자기 지도 학습 (Self-Supervised Learning):** [arxiv](https://arxiv.org/pdf/2303.18205.pdf)

대조 학습 (Contrastive Learning): [arxiv](https://arxiv.org/pdf/2303.18205.pdf)
- 과거에서 미래 예측 (next-step prediction) [arxiv](https://arxiv.org/pdf/2303.18205.pdf)
- 마스크된 재구성 (masked reconstruction)
- 표현 학습 개선

**TiDE (Time series Dense Encoder):** [sciencedirect](https://www.sciencedirect.com/science/article/pii/S073658452500064X)
- 인코더-디코더 구조
- 간단하지만 효과적 (WAPE 2.40%)

#### 7.4 도메인 특화 응용

**1. 제조 시계열:** [sciencedirect](https://www.sciencedirect.com/science/article/pii/S073658452500064X)
- N-BEATS, PatchTST 우수 성능
- BiTCN 다변량 예측에 효과적 (WAPE 44.77%)

**2. 기상 예측:** [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12453695/)
- iTransformer: RMSE 1.43, MAPE 0.66
- 장기 패턴 포착 능력 (2025년 연구) [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12453695/)

**3. 금융:** [arxiv](https://arxiv.org/html/2601.13082v1)
- Transformer > LSTM (10/10 데이터셋, RMSE)
- LSTM-Transformer 하이브리드 모형 (2025) [mdpi](https://www.mdpi.com/2413-4155/7/1/7)

**4. 프로세스 모델 예측:** [arxiv](https://arxiv.org/abs/2512.07624)
- Time Series Foundation Models (TSFMs)
- Zero-shot 성능이 기존 모델 능가 (MAE/RMSE)
- 미세조정 이득 미미한 경우도 있음 [arxiv](https://arxiv.org/abs/2512.07624)

#### 7.5 GluonTS와 최신 연구의 비교

| 측면 | GluonTS (2019) | 최신 연구 (2024-2025) |
|-----|---------------|--------------------|
| **아키텍처** | LSTM, CNN, Transformer | Transformer 중심, 패치 기반 |
| **학습 방식** | 지도학습 | 사전학습 + 미세조정, 자기 지도 |
| **데이터 규모** | 수백만 시계열 | 수십억 시계열 |
| **모델 크기** | 중간 규모 | 200M+ 파라미터 (Foundation Models) |
| **일반화** | 글로벌 모델 공유 | 사전학습된 표현 전이 |
| **도메인 적응** | 하이퍼파라미터 조정 | Zero-shot 또는 경량 미세조정 |
| **확률 예측** | 배포 컴포넌트 | 자동 불확실성 정량화 |

***

### 8. 앞으로의 연구에 미치는 영향과 고려사항

#### 8.1 GluonTS의 지속적 영향

**1. 프로토타이핑 가속화:**
- 신축적인 모델 조립 ("mix-and-match")
- 신규 아키텍처 빠른 구현 및 벤치마크
- Amazon SageMaker 통합으로 프로덕션 경로 단순화 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/02359309-88cd-4b2c-95ac-abec700a6031/1906.05264v2.pdf)

**2. 재현 가능성 표준:**
- 인간이 읽을 수 있는 구성 직렬화 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/02359309-88cd-4b2c-95ac-abec700a6031/1906.05264v2.pdf)
- 완전한 실험 추적 및 재현
- 학술 커뮤니티의 재현성 위기 해결에 기여

**3. 개방형 커뮤니티 표준:**
- AWS에서 활발한 개발 및 유지보수
- 다양한 아키텍처 통합
- PyTorch 백엔드 지원으로 확장성 증가

#### 8.2 앞으로의 연구 방향

**1. Domain-Specific Fine-tuning:**

Foundation Model 기반 특화: [arxiv](https://arxiv.org/abs/2403.14735)

$$\theta_{\text{target}} = \text{PEFT}(\theta_{\text{FM}}, \text{Domain Data})$$

경량 파라미터 효율적 미세조정 (PEFT):
- LoRA (Low-Rank Adaptation): 기존 가중치에 낮은 순위 업데이트 추가
- Prefix Tuning: 프롬프트처럼 동작하는 매개변수
- 1-10% 데이터로도 특화 모델 성능 달성 가능

**2. 하이브리드 모델 아키텍처:**

물리 정보 기반 신경망 (Physics-Informed Neural Networks, PINNs):

$$\mathcal{L} = \mathcal{L}\_{\text{data}} + \lambda \mathcal{L}_{\text{physics}}$$

예: 에너지 시장 예측에 공급-수요 법칙 제약 추가

**3. 장기 예측 능력:** [nature](https://www.nature.com/articles/s41467-025-63786-4)

외삽 가능성(Extrapolation) 개선: [arxiv](https://arxiv.org/pdf/2312.00817.pdf)
- xPos (Extrapolatable Position Embedding) 임베딩
- Trend와 Seasonality 명시적 인코딩
- 6,000 타임스텝까지 안정적 예측 달성 [arxiv](https://arxiv.org/pdf/2312.00817.pdf)

**4. 불확실성 정량화:** [arxiv](https://arxiv.org/pdf/2511.23260.pdf)

Per-step 확률 분포 학습:
$$p(y_t | x_{1:t}) = \mathcal{N}(\mu_t(x_{1:t}; \theta), \sigma_t^2(x_{1:t}; \theta))$$

이중 분기 네트워크 (Dual-branch) 아키텍처로 평균과 분산 동시 학습

**5. 데이터 효율성:** [ijcai](https://www.ijcai.org/proceedings/2025/1187.pdf)

Few-shot 학습 (TimePFN):
- Prior-data Fitted Networks (PFN) 개념 적용
- 합성 데이터로 다양한 분포 학습
- 수십 개 샘플로도 효과적 예측 [arxiv](https://arxiv.org/pdf/2502.16294.pdf)

자기 지도 표현 학습:
- 음성 Foundation Models (wav2vec 2.0, HuBERT)의 일반화 능력
- 센서 시계열 작업에도 전이 가능 [arxiv](https://arxiv.org/abs/2509.00221)

#### 8.3 실무 적용 시 고려사항

**1. 모델 선택 전략:**
```
if data_size < 10k:
    use(StatisticalModels)  # ARIMA, ETS
elif data_size < 100k:
    use(TransferLearning(FoundationModel))  # Fine-tune TimesFM
else:
    ensemble(SOTA_Models)  # Informer, iTransformer, PatchTST
```

**2. 계산 비용 고려:**
- Transformer 모델: 훈련 시간 1-2시간 (수백만 시계열) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/02359309-88cd-4b2c-95ac-abec700a6031/1906.05264v2.pdf)
- Foundation Model 미세조정: 수 분-수십 분 [forecasters](https://forecasters.org/wp-content/uploads/TransferL_Progress2_2022-SAS_Kin-G.-Olivares.pdf)
- 추론: 거의 실시간 (마이크로초 단위)

**3. 프로덕션 배포:**

GluonTS의 상태 비저장 API: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/02359309-88cd-4b2c-95ac-abec700a6031/1906.05264v2.pdf)
```python
predictor = estimator.train(train_dataset)
forecast = predictor(test_item)  # Stateless
```

이점: 병렬 처리, 분산 배포, 쉬운 확장

**4. 모니터링 및 재훈련:**
- 시간 변화하는 패턴 (concept drift) 감시
- 분기별 재훈련 권장
- MASE, MAPE로 지속적 성능 추적

**5. 설명 가능성:** [research](https://research.google/blog/time-series-foundation-models-can-be-few-shot-learners/)

Foundation Model의 강점: [research](https://research.google/blog/time-series-foundation-models-can-be-few-shot-learners/)
- Zero-shot 성능 우수 → 신뢰성 증가
- 일반적 패턴 학습 → 이상 탐지 강화
- 한계: Attention 가중치 해석의 어려움

***

### 9. 핵심 결론

**GluonTS의 유산 (2019-2025):**

1. **패러다임 전환**: 지역 모델 중심에서 글로벌 신경망 모델 중심으로
2. **도구화**: 학술 연구를 프로덕션 시스템으로 전환하는 다리 역할
3. **표준화**: 확률적 시계열 모델링의 모듈화 및 벤치마크 기준 제시

**향후 가장 중요한 발전:**

1. **Foundation Model 시대**: 대규모 사전학습으로 일반화 능력 극대화
2. **경량화**: 20-200M 파라미터로도 경쟁력 유지 (TimesFM, TimePFN)
3. **멀티모달화**: 시계열 + 텍스트 + 메타데이터 통합
4. **적응성**: 동적 환경에서의 빠른 적응 (메타 러닝)

**실무 권장사항:**

| 상황 | 추천 접근법 | 이유 |
|-----|-----------|------|
| 소규모 데이터 (<10k) | ARIMA/Prophet 또는 Foundation Model 전이 | 신경망 과적합 위험 |
| 중규모 데이터 (10k-100k) | Informer, PatchTST + 미세조정 | 최적의 성능-비용 트레이드오프 |
| 대규모 데이터 (>100k) | TimesFM Zero-shot 또는 앙상블 | 최고 성능, 계산 효율 |
| 불규칙 샘플링 | TimelyGPT 또는 연속시간 모델 | 건강관리, 센서 데이터 |
| 멀티모달 | MM-TSFlib (시계열+텍스트) | 15-40% 성능 향상 |

***

### 참고 자료
<span style="display:none">[^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_80][^1_81][^1_82][^1_83][^1_84][^1_85][^1_86][^1_87][^1_88][^1_89][^1_90][^1_91][^1_92][^1_93]</span>

<div align="center">⁂</div>

[^1_1]: 1906.05264v2.pdf

[^1_2]: https://arxiv.org/html/2504.04011v1

[^1_3]: https://arxiv.org/abs/2403.14735

[^1_4]: https://forecasters.org/wp-content/uploads/TransferL_Progress2_2022-SAS_Kin-G.-Olivares.pdf

[^1_5]: https://arxiv.org/abs/2501.13989

[^1_6]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12453695/

[^1_7]: https://arxiv.org/pdf/2406.08627.pdf

[^1_8]: https://arxiv.org/pdf/2401.13968.pdf

[^1_9]: https://arxiv.org/pdf/2303.18205.pdf

[^1_10]: https://www.ijcai.org/proceedings/2025/1187.pdf

[^1_11]: https://www.sciencedirect.com/science/article/pii/S073658452500064X

[^1_12]: https://arxiv.org/html/2601.13082v1

[^1_13]: https://www.mdpi.com/2413-4155/7/1/7

[^1_14]: https://arxiv.org/abs/2512.07624

[^1_15]: https://www.nature.com/articles/s41467-025-63786-4

[^1_16]: https://arxiv.org/pdf/2312.00817.pdf

[^1_17]: https://arxiv.org/pdf/2511.23260.pdf

[^1_18]: https://arxiv.org/pdf/2502.16294.pdf

[^1_19]: https://arxiv.org/abs/2509.00221

[^1_20]: https://research.google/blog/time-series-foundation-models-can-be-few-shot-learners/

[^1_21]: https://icml.cc/virtual/2025/poster/44262

[^1_22]: https://ojs.aaai.org/index.php/AAAI/article/view/17325

[^1_23]: https://arxiv.org/abs/2207.05397

[^1_24]: http://www.sciencepublishinggroup.com/journal/paperinfo?journalid=146\&doi=10.11648/j.ajtas.20200904.18

[^1_25]: http://link.springer.com/10.1007/s11207-020-1595-3

[^1_26]: https://dl.acm.org/doi/10.1145/3437802.3437827

[^1_27]: http://link.springer.com/10.1007/s12080-020-00451-0

[^1_28]: https://inmateh.eu/volumes/volume-61--no-2--2020/article/61-07-zeying-xu-prediction-model-of-ammonia-concentration-in-yellow-feather-broilers-house-durin

[^1_29]: https://meetingorganizer.copernicus.org/EGU2020/EGU2020-3054.html

[^1_30]: https://www.semanticscholar.org/paper/b0528c13b684547b8a13241a4d212c32f176cbff

[^1_31]: https://www.semanticscholar.org/paper/540ec5368c90c59a0caa81cf11dc547c1a3e7165

[^1_32]: https://pubs.geoscienceworld.org/ssa/srl/article/91/5/2631/587730/California-Historical-Intensity-Mapping-Project

[^1_33]: https://www.ssrn.com/abstract=3574846

[^1_34]: https://arxiv.org/pdf/1906.05264.pdf

[^1_35]: https://arxiv.org/pdf/2402.16516.pdf

[^1_36]: https://arxiv.org/pdf/2310.10688.pdf

[^1_37]: https://arxiv.org/pdf/2401.13912.pdf

[^1_38]: https://arxiv.org/pdf/2307.01616.pdf

[^1_39]: http://arxiv.org/pdf/2405.13522.pdf

[^1_40]: https://arxiv.org/pdf/2312.17100.pdf

[^1_41]: https://dl.acm.org/doi/pdf/10.5555/3455716.3455832

[^1_42]: https://arxiv.org/abs/1906.05264

[^1_43]: https://proceedings.neurips.cc/paper/2020/file/2f2b265625d76a6704b08093c652fd79-Paper.pdf

[^1_44]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10312385/

[^1_45]: https://www.sciencedirect.com/science/article/pii/S2667305325000742

[^1_46]: https://github.com/awslabs/gluonts

[^1_47]: https://journal.hep.com.cn/fcs/EN/10.1007/s11704-025-50947-3

[^1_48]: https://arxiv.org/abs/2306.04901

[^1_49]: https://www.jmlr.org/papers/v21/19-820.html

[^1_50]: https://www.sciencedirect.com/science/article/abs/pii/S0957417425031008

[^1_51]: https://arxiv.org/pdf/2308.13222.pdf

[^1_52]: https://arxiv.org/html/2407.16445v2/

[^1_53]: https://arxiv.org/pdf/2508.11004.pdf

[^1_54]: https://pdfs.semanticscholar.org/7f99/cabdf824b49121275507233165d600f95878.pdf

[^1_55]: https://arxiv.org/pdf/2506.13201.pdf

[^1_56]: https://arxiv.org/html/2407.17877v1

[^1_57]: https://arxiv.org/html/2503.10198v1

[^1_58]: https://arxiv.org/html/2502.17495v1

[^1_59]: https://arxiv.org/html/2510.08202v1

[^1_60]: https://ar5iv.labs.arxiv.org/html/1906.05264

[^1_61]: https://arxiv.org/html/2509.09176

[^1_62]: https://arxiv.org/pdf/2203.09474.pdf

[^1_63]: https://ieeexplore.ieee.org/document/11154197/

[^1_64]: https://www.mdpi.com/1424-8220/25/3/652

[^1_65]: https://www.mdpi.com/2227-7390/13/5/814

[^1_66]: https://ieeexplore.ieee.org/document/10926918/

[^1_67]: https://www.mdpi.com/2571-9394/7/3/41

[^1_68]: https://ieeexplore.ieee.org/document/11168679/

[^1_69]: https://ieeexplore.ieee.org/document/11239713/

[^1_70]: https://arxiv.org/html/2411.01419v1

[^1_71]: https://arxiv.org/pdf/2310.20218.pdf

[^1_72]: http://arxiv.org/pdf/2410.23749.pdf

[^1_73]: http://arxiv.org/pdf/2211.14730v2.pdf

[^1_74]: https://arxiv.org/pdf/2502.13721.pdf

[^1_75]: https://arxiv.org/html/2312.00817v3

[^1_76]: https://peerj.com/articles/cs-3001/

[^1_77]: https://aigrowthclub.kr/ai-term-transformer-아키텍처/

[^1_78]: https://www.sciencedirect.com/science/article/pii/S1574013725001595

[^1_79]: https://www.pricepedia.it/en/magazine/article/2025/11/07/the-arrival-of-foundation-models-in-time-series-forecasting/

[^1_80]: https://zsunn.tistory.com/entry/AI-최신-기술-이해-및-실습-Transformers-Self-Attention-GPT-BERT-등

[^1_81]: https://research.aimultiple.com/time-series-foundation-models/

[^1_82]: https://hyperlab.hits.ai/blog/titans-transformer

[^1_83]: https://www.kjas.or.kr/journal/view.html?doi=10.5351%2FKJAS.2024.37.5.583

[^1_84]: https://www.videns.ai/en-ca/blog/lessor-des-modeles-fondamentaux-dans-les-series-temporelles-un-changement-de-paradigme-ou-juste-un-autre-engouement

[^1_85]: https://calmmimiforest.tistory.com/110

[^1_86]: https://arxiv.org/html/2502.21245v1

[^1_87]: https://www.semanticscholar.org/paper/Explainable-transformers-in-financial-forecasting-Govindaraj-Jaganathan/ab9569490438dbd97729cca6e09e71d8b7ed4e6d

[^1_88]: https://arxiv.org/html/2505.12761v2

[^1_89]: https://arxiv.org/html/2509.22359v2

[^1_90]: https://arxiv.org/abs/2506.15705

[^1_91]: https://arxiv.org/pdf/2509.04162.pdf

[^1_92]: https://arxiv.org/html/2510.00742v3

[^1_93]: https://arxiv.org/html/2310.04948v3
