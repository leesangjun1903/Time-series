# Dish-TS: A General Paradigm for Alleviating Distribution Shift in Time Series Forecasting

**핵심 주장 및 기여**  
Dish-TS는 시계열 예측(Time Series Forecasting, TSF)에서 입력(lookback)과 출력(horizon) 구간 간, 그리고 동일 구간 내 시간에 따른 분포 변화를 체계적으로 정의하고 이를 완화하는 **Dual-CONET** 기반의 일반 신경망 패러다임을 제안한다.[1]
- 분포 변화를 **intra-space shift**(입력 구간 내 변화)와 **inter-space shift**(입력↔출력 구간 간 변화)로 구분·정의  
- **CONET**(Coefficient Net)으로 각 구간의 분포를 학습 가능한 레벨·스케일 계수로 추정  
- 두 개의 CONET을 이용해 입력·출력 분포를 별도로 정규화/역정규화하는 **Dual-CONET 프레임워크** 설계  
- Prior knowledge(평균) 기반 추가 손실로 출력 분포 예측 정확도 향상 유도  
- 다양한 SOTA 모델(In​former, Autoformer, N-BEATS)과 결합해 평균 20∼28% 이상의 성능 향상 달성  

***

## 1. 해결하고자 하는 문제  
비정상(non-stationary) 시계열 데이터는 관측 빈도·시간 흐름에 따라 통계량(평균·표준편차)이 달라져 기존 TSF 모델의 일반화 성능 및 예측 정확도를 크게 떨어뜨린다.  
- 기존 연구는 구간 내 분포 변화(intra-space shift)만 불완전하게 정량화하거나, 입력·출력 구간이 동일 분포라고 가정해 inter-space shift를 무시함.[1]
- 센서 샘플링 주기와 실제 분포가 일치하지 않아 고정 통계량 기반 정규화는 표현력·신뢰성이 부족.

***

## 2. 제안 방법  
### 2.1 CONET: 분포 계수 학습망  
임의 시계열 윈도우 $$x\in\mathbb{R}^L$$에 대해  

$$
(\alpha,\;\beta)=\mathrm{CONET}(x)
$$

- $$\alpha\in\mathbb{R}$$: 전체 스케일(level) 계수  
- $$\beta\in\mathbb{R}$$: 변동(fluctuation) 계수  
- CONET은 fully-connected 등 자유로운 신경망 구조로 설계 가능.[1]

### 2.2 Dual-CONET 프레임워크  
- **BACK-CONET**: 입력 구간 분포 $$\mathcal{X}_{\mathrm{in}}$$ 추정  
- **HORI-CONET**: 출력 구간 분포 $$\mathcal{X}_{\mathrm{out}}$$ 예측  
- 정규화→예측 모델 $$F$$→역정규화 과정:
  1. 입력 정규화: $$\tilde{x}\_{t-L+1:t}=(\,x_{t-L+1:t}-\alpha_b)/\beta_b$$  
  2. 예측: $$\hat{y}\_{t+1:t+H}=F(\tilde{x}_{t-L+1:t})$$  
  3. 역정규화: $$\hat{y}=\beta_h\,\hat{y}_{t+1:t+H}+\alpha_h$$  
- 이로써 intra-, inter-space shift를 모두 완화.[1]

### 2.3 Prior Knowledge 유도 학습  
출력 분포 예측이 어려워지므로, 실제 향후 구간 평균 $$\bar{y}$$를 soft target으로 추가 손실 구성:  

$$
\mathcal{L}
=\underbrace{\|\hat{y}-y\|_2^2}_{\text{MSE}}
+\lambda\,
\|\alpha_h-\bar{y}\|_2^2
$$

$$\lambda$$로 prior guidance 강도 조절.[1]

***

## 3. 모델 구조  
- CONET: 2개의 fully-connected layer + LeakyReLU  
- Dual-CONET: BACK-CONET, HORI-CONET 독립 학습  
- Forecast 모델 $$F$$: Informer/Autoformer/N-BEATS 등 자유롭게 결합 가능  
- End-to-end 학습.

***

## 4. 성능 향상 및 한계  
### 4.1 성능 향상  
- **Univariate forecasting**: 평균 MSE 28.6% 개선  
- **Multivariate forecasting**: 평균 MSE 21.9% 개선[1]
- RevIN 대비 10∼36% 추가 개선[1]
- 긴 예측(horizon↑), 긴 과거정보(lookback↑) 모두에서 일관된 성능 유지

### 4.2 한계  
- Horizon 길이 대폭 증가 시 HORI-CONET 학습 불안정  
- 단순 CONET 설계(선형 투영+편차)로는 복잡한 분포에는 표현력 부족 가능  
- Prior guidance 비율 $$\lambda$$ 민감도 존재

***

## 5. 일반화 성능 향상 관점  
Dual-CONET의 분포 계수 학습은 모델이 시계열의 비정상성을 직접 모델링하도록 유도해, 다양한 시계열 유형 및 샘플링 주기 변화에도 견고한 예측을 가능하게 한다. 특히 inter-space shift 완화가 일반화 성능을 크게 끌어올린다.

***

## 6. 향후 연구에 미치는 영향 및 고려 사항  
Dish-TS는 **분포 변화 정량화+완화** 패러다임을 제시함으로써, TSF 연구에서 비정상성 문제를 체계적으로 다루는 새로운 길을 열었다.  
- **향후 연구**:  
  - 더 복잡한 분포(비정규 분포, 계절성 등) 추정을 위한 CONET 구조 확장  
  - 적응형 $$\lambda$$ 스케줄링 또는 메타러닝 적용  
  - 비선형/비가우시안 분포 측정 기법 통합  
- **고려점**:  
  - Horizon 구간 길이에 따른 prior guidance 효과 분석  
  - 실제 산업 데이터(불완전·결측)에서 CONET 안정성 검증  
  - 다중 시계열 간 상호작용 고려한 분포 학습 확장  

Dish-TS는 분포 변화를 직접 모델링함으로써 TSF의 일반화 성능 한계를 효과적으로 넘어서며, 향후 다양한 시계열 예측 과제에서 핵심 기법으로 자리잡을 수 있다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8558d4d5-eae7-450f-a518-802e5686cf3f/2302.14829v3.pdf)
