# Conformal PID Control for Time Series Prediction

# 주요 주장 및 기여 요약

**Conformal PID Control**은 시계열 예측에서 불확실성을 정량화하기 위해 *온라인 상황*에서 작동하는 새로운 알고리즘 프레임워크를 제안한다.  
핵심 기여:
- 분포 변화(seasonality, trend, distribution shift)를 *사전 예측*하며 적응하는 **Scorecasting (D 제어)** 모듈 도입.  
- 과거 커버리지 오차를 누적하여 보정하는 **Error Integration (I 제어)** 기법 제시.  
- 실시간으로 분위수(quantile)를 추적하는 **Quantile Tracking (P 제어)** 알고리즘 제안.  
- 이 세 모듈을 결합한 **Proportional–Integral–Derivative (PID) 제어** 관점의 해석 및 이론적 보장 제공.[1]

# 상세 설명

## 해결하고자 하는 문제  
전통적 *Conformal Prediction*은 데이터가 i.i.d.일 때만 엄격한 보장을 제공하며, 시계열 예측에서 분포 이동이 발생하면 예측 불확실성이 무너지게 된다.  
사용자는 제한된 가정 하에 장기적 평균 커버리지 보장을 달성하면서, 예측 구간을 가능한 한 좁게 유지하고자 한다.

## 제안 방법

### 1) Quantile Tracking (P 제어)  
모든 시점 t까지의 conformal score $$s_t$$의 $$\alpha$$ 분위수를 온라인 그래디언트 하강법으로 추적.  
최적화 문제:  

$$
\min_{q\in\mathbb{R}} \sum_{t=1}^T \ell_\alpha(s_t - q),
$$  

$$\ell_\alpha(z)=\alpha z_+ + (1-\alpha)(-z)_+$$인 분위수 손실을 사용.  

업데이트식:  

$$
q_{t+1} = q_t + \eta(\mathbb{1}\{s_t > q_t\} - \alpha),
$$  

이는 $$P$$ 제어에 해당하며, 보장을 위해 점차 보정하며 장기적 커버리지를 달성.[1]

### 2) Error Integration (I 제어)  
과거 커버리지 오차 $$\mathrm{err}_i = \mathbb{1}\{y_i \notin C_i\} - \alpha$$의 누적합을 사용하여 분위수 보정을 안정화.  

$$
q_{t+1} = r_t\Bigl(\sum_{i=1}^t \mathrm{err}_i\Bigr),
$$  

여기서 $$r_t$$은 비선형 포화 함수(예: $$\tanh$$ 기반)로, 과도한 수정 방지 및 안정적 수렴을 보장한다.[1]

### 3) Scorecasting (D 제어)  
현재 시점의 conformal score 변동 경향을 예측하는 추가 모델 $$g_t$$을 학습하여,  

$$
C_t = \{y: s_t(x_t,y)\le q_t\},\quad q_{t+1} = g_t + r_t\Bigl(\sum_{i=1}^t \mathrm{err}_i\Bigr).
$$  

이는 *미리* 분포 이동 신호를 반영하여 예측 구간 크기를 조정하며, 추세(residualized trend)를 제거해 보다 날카로운 예측 구간을 생성한다.[1]

## 모델 구조  
- **Base Forecaster**: AR, Theta, Prophet, Transformer 등 다양한 모델 가능.  
- **Quantile Tracker (P)**: 위 업데이트식 적용.  
- **Integrator (I)**: $$\tanh$$-기반 포화 함수로 누적 오차를 조정.  
- **Scorecaster (D)**: 1-penalized Quantile Regression, Theta 모델 등 예측 잔차를 학습.

## 성능 향상 및 한계  
실험(- **COVID-19 주별 사망자 예측**: CDC Ensemble 대비 커버리지 20%→70% 회복, 과소예측 시점 보정 - **전력 수요 예측**: Adaptive Conformal Inference 대비 더 작은 예측 구간과 안정적 커버리지 유지 )이 모든 설정에서 보장된 장기 커버리지와 구간 효율성 개선을 보였다.  
다만, Scorecasting이 잔여 신호가 없거나 과도하게 복잡할 경우 오히려 분산을 증가시켜 예측 구간의 변동성을 키울 수 있다.  

## 일반화 성능 향상 관점  
Scorecasting은 잔여 에러에 내재한 *계절성·추세·외생 변수* 신호를 예측하여 제거하므로,  
- **데이터 분포 변화**에 대한 **사전 적응**이 가능  
- **기저 예측기(Base Forecaster)**의 구조적 한계를 보완  
이를 통해 다양한 데이터를 횡단 적용할 때 모델의 **일반화 성능**을 크게 개선할 수 있다.[1]

# 향후 연구 방향 및 고려 사항

- **적응적 하이퍼파라미터 튜닝**: Integrator 상수($$\mathrm{KI}, \mathrm{Csat}$$) 및 학습률 $$\eta$$를 온라인으로 조절하는 메커니즘 개발.  
- **커널 기반 통합**: 과거 오차 통합 시 *로컬 윈도우* 또는 *그룹별* 가중 커널 적용으로 지역 커버리지 보강.  
- **고차원 시계열**: 다변량 예측 설정에서 PID 컨트롤러 모듈 간 상호작용 및 확장성 연구.  
- **리스크 제어 확장**: 제안된 기법을 **Conformal Risk Control** 및 다양한 손실 함수 환경으로 일반화.  
- **Scorecasting 모델 설계**: 문제별 특성에 맞춘 맞춤형 잔차 예측기(예: RNN, Transformer) 선정 및 안정성 분석.  

위 고려사항은 Conformal PID Control 프레임워크의 **실제 적용성**과 **강건성**을 한층 더 높이는 데 기여할 것으로 기대된다.  

   파일:2307.16895v1.pdf (Abstract 및 Introduction)[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/83b0d3b9-1c6d-4705-8d4e-ab6bf1d8ca16/2307.16895v1.pdf)
