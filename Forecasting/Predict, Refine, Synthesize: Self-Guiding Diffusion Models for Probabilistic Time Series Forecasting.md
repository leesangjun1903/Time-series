# Predict, Refine, Synthesize: Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting

## 핵심 주장 및 주요 기여  
이 논문은 **TSDiff**, 즉 *Self-Guiding Unconditional Diffusion Model for Time Series*를 제안하여,  
- **단일 무조건(unconditional) 확산 모델**이 다양한 시계열 예측, 결측치 보간(imputation), 합성 데이터 생성 작업에 모두 적용 가능함을 보인다.[1]
- **관찰치 자기 안내(observation self-guidance)** 메커니즘을 통해, 추가 네트워크나 별도 학습 없이도 학습된 무조건 모델을 조건부 모델처럼 활용할 수 있도록 한다.[1]
- 생성된 암묵적 확률 밀도를 활용해 **기존 예측기의 출력 예측을 사후 정제(refinement)** 하는 방식을 제안하여, 계산 비용을 줄이면서 성능을 향상시킨다.[1]
- 합성 샘플의 예측 품질을 평가하기 위한 **Linear Predictive Score (LPS)**를 도입하고, TSDiff로 생성된 샘플이 기존 생성 모델(TimeVAE, TimeGAN)을 능가함을 보여준다.[1]

## 해결하고자 하는 문제  
기존 시계열 확산 모델은 주로 특정 예측(imputation/forecasting) 작업에 특화된 **조건부 모델**로 학습되어,  
- 새로운 작업에 맞춰 모델을 다시 학습해야 하며  
- 무조건 생성(unconditional generation) 능력을 활용할 수 없다는 한계가 있다.[1]

논문은 이 한계를 극복하기 위해 **단일 무조건 확산 모델 학습**만으로 모든 하위 작업을 처리할 수 있는지를 탐구한다.[1]

## 제안 방법  

### 1. 무조건 확산 모델(TSDiff) 학습  
- 시계열 $$y \in \mathbb{R}^L$$를 모델링하기 위해 **Denoising Diffusion Probabilistic Model (DDPM)** 프레임워크를 사용한다.[1]
- S4 계층과 $$1\times1$$ 컨볼루션을 결합한 아키텍처(SSSD 기반)를 채택하여, 입력 시퀀스 길이 $$L$$와 시차(lags)를 채널 차원에 함께 제공한다.[1]

### 2. 관찰치 자기 안내(Observation Self-Guidance)  
조건부 예측 $$\,p(y_{\text{target}}\mid y_{\text{obs}})$$는 베이즈 법칙을 적용한 **가이드드 역확산(reverse diffusion)** 으로 구현한다:[1]

$$
p(x_{t-1}\mid x_t, y_{\text{obs}})
\;=\;
\mathcal{N}\bigl(x_{t-1}\mid x_t + s^2_t\nabla_{x_t}\log p(y_{\text{obs}}\mid x_t),\,\Sigma_t\bigr),
$$  

여기서 $$s$$는 가이드 스케일, $$\nabla_{x_t}\log p(y_{\text{obs}}\mid x_t)$$는 **자기 안내**를 위해 무조건 모델을 재활용해 계산된 관찰치 우도(가우시안 또는 비대칭 라플라스)이다.[1]
- **MSE Self-Guidance**: 관찰치에 대한 가우시안 우도로 접근하여 평균 제곱 오차 형태의 안내 신호를 사용한다.[1]
- **Quantile Self-Guidance**: 연속 순위 점수(CRPS)에 최적화된 **비대칭 라플라스** 우도를 사용해 다양한 분위수를 고려한 안내를 수행한다.[1]

### 3. 예측 정제(Prediction Refinement)  
기존 예측기 $$g$$의 출력 $$\hat y = g(y_{\text{obs}})$$을 **에너지 기반 모델(EBM)** 으로 보고,  

$$
E(y) = -\log p(y) + R(y, \hat y),\quad
R=\text{MSE or Quantile Loss},
$$  

를 최소화하거나 Langevin 몬테카를로 샘플링으로 탐색하여 $$\hat y$$를 개선한다.[1]

## 모델 구조  
- **입력**: 관찰 시퀀스와 시차값을 채널 차원으로 포함한 노이즈 시퀀스 $$x_t\in\mathbb{R}^{L\times C}$$  
- **핵심 계층**: S4 모듈(시간 차원 처리) + $$1\times1$$ 컨볼루션(채널 차원 처리)  
- **출력**: 입력과 동일한 차원의 노이즈 예측  
- **Guidance**: 역확산 단계에서 네트워크 출력으로부터 관찰치 우도 점수(score)를 계산해 추가  

## 성능 향상  
- **Forecasting**: 8개 벤치마크 데이터셋에서 TSDiff-Q(Quantile Self-Guidance)가 Task-Specific 확산 모델(CSDI)과 유사하거나 우수한 지속적 순위 점수(CRPS)를 달성.[1]
- **Missing Values**: 학습 시 결측치 시나리오를 정의할 필요 없이, 단일 무조건 모델로 3가지 결측 시나리오 모두에서 경쟁력 있는 성능을 보임.[1]
- **Refinement**: 단일 iteration만으로 단순 포인트 예측기(Seasonal Naive, Linear) 성능을 즉시 개선, 복잡 예측기(DeepAR, Transformer)에도 이득을 제공.[1]
- **Synthetic Data**: 제안된 **LPS**에서 TimeVAE, TimeGAN 대비 크게 우수, 다운스트림 DeepAR/Transformer 학습 시에도 실제 데이터 성능에 근접.[1]

## 한계  
- **높은 계산 비용**: 반복적 역확산 과정 및 gradient 계산으로 인해 추론 시간이 증가함.  
- **비용-성능 균형**: self-guidance와 refinement 단계별 반복 횟수, guidance 스케일 선택이 성능에 민감함.  
- **단일 데이터셋 학습**: 각 데이터셋별로 모델을 재학습해야 하며, 대규모 다중 태스크 학습은 미검증 상태임.  

## 일반화 성능 향상 가능성  
- **Task-Agnostic 학습**: 무조건 훈련만으로 예측·보간·합성이 가능하므로, 새로운 시계열 도메인으로 빠르게 적용할 수 있는 잠재력이 높다.[1]
- **Quantile Guidance**: 분위수 수준별 안내는 불확실성 분포를 더 풍부히 반영, CRPS 최적화에 유리해 일반화된 예측 신뢰도를 높인다.[1]
- **EBM Refinement**: 다양한 예측기와 결합해 사후 정제를 수행, 기존 시스템에 손쉬운 통합이 가능해 실제 산업 적용에서 확장성이 높다.[1]

## 향후 연구 및 고려 사항  
- **가속화된 확산 솔버**(e.g., DPM-Solver, Fast Guided Solvers)로 추론 비용 절감[1]
- **모멘텀 기반 MCMC**(Hamiltonian/Underdamped Langevin) 도입으로 refinement 효율 향상  
- **다중 시계열·멀티태스크 훈련**: 하나의 TSDiff로 여러 시계열 간 패턴 공유 및 전이 학습  
- **다양한 역문제(inverse problems)**로의 확장: 이미지 복원, 신호 분리 등 다른 도메인에 self-guidance 적용  
- **Guidance 스케일 자동 튜닝**: 메타 학습 기반 스케일 적응으로 성능 민감도 완화  

TSDiff는 **무조건 확산 모델**이 시계열 예측·보간·합성 작업 전반에서 *범용 기저(foundation model)* 역할을 할 수 있음을 입증하며, 다양한 실제 응용에 빠르게 적용 가능한 *Self-Guiding Diffusion* 패러다임을 제시한다.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/360661c3-f822-4322-8c59-ac73a10e8277/2307.11494v3.pdf)
