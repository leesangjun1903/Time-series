# Pathformer: Multi-scale Transformers with Adaptive Pathways for Time Series Forecasting

**주요 주장**  
Pathformer는 시계열 예측을 위해 **다중 스케일(Multi-Scale) Transformer** 구조와 입력 시계열의 동적 특성에 따라 스케일을 선택하는 **적응적 경로(Adaptive Pathways)**를 결합하여, 기존 모델들이 고정된 단일 스케일이나 스케일 간 상호작용을 충분히 고려하지 못하는 한계를 극복한다.[1]

**주요 기여**  
- 다중 스케일 분할(Multi-Scale Division)과 **이중 주의 메커니즘(Dual Attention)**을 결합해 서로 다른 시계열 해상도와 거리에 대한 전역 상관관계 및 국부 세부 정보(local details)를 동시에 학습.  
- **Adaptive Pathways**: 입력 시계열의 추세(trend)와 계절성(seasonality)을 분해한 후, 경로 라우터(Router)가 스케일별 가중치를 생성하고 Top-K 스케일을 선택하여 모델이 각 샘플에 최적화된 스케일 조합을 자동으로 구성.  
- 11개 실제 데이터셋에서 기존 최고 성능 모델 대비 평균 MSE 8.1%, MAE 6.4% 절감 및 전이 학습(Transfer Learning) 평가에서 강력한 **일반화 성능** 입증.[1]

***

## 1. 해결하려는 문제  
현실 시계열 데이터는 다양한 주기와 변화 폭을 가지므로,  
- **단일 스케일** 또는 **고정된 멀티스케일** 모델은 특정 시계열 특성을 포착하기 어렵고,  
- **스케일 선택**에 드는 수작업 튜닝 비용이 크며, 데이터마다 최적 스케일이 상이함.  

이로 인해 모델의 예측 정확도 및 **미지 데이터 일반화 성능**이 제한된다.[1]

***

## 2. 제안 방법

### 2.1 모델 구조 개요  
Pathformer는 다음 세 부분으로 구성된다:[1]
1) Instance Normalization  
2) 여러 개의 **Adaptive Multi-Scale Block(AMS Block)** 쌓기  
3) Predictor (Fully Connected Layer)

각 AMS Block은  
- **Multi-Scale Transformer Block**: 다양한 패치 크기로 분할 후, 이중 주의(Dual Attention) 적용  
- **Adaptive Pathways**: Multi-Scale Router + Aggregator  

### 2.2 Multi-Scale Transformer Block  
- 패치 크기 집합 $$S = \{S_1, \dots, S_M\}$$ 정의  
- 각 $$S$$에 대해 시계열 $$X \in \mathbb{R}^{H \times d}$$를 길이 $$S$$ 패치로 분할하여 $$X_i \in \mathbb{R}^{S \times d}$$ 생성  
- **Intra-Patch Attention**(식 1):  

$$
    \mathrm{Attn}^{\text{intra}} = \mathrm{Softmax} \big(Q^{\text{intra}}(K^{\text{intra}})^T / d_m\big)\,V^{\text{intra}}
  $$  

- **Inter-Patch Attention**(식 3):  

$$
    \mathrm{Attn}^{\text{inter}} = \mathrm{Softmax} \big(Q^{\text{inter}}(K^{\text{inter}})^T / d_m\big)\,V^{\text{inter}}
  $$  

- 최종 Dual Attention 결과 $$\mathrm{Attn} = \mathrm{Inter} + \mathrm{Intra}$$ 으로 결합해 다양한 스케일의 전역 및 국부 패턴 학습.[1]

### 2.3 Adaptive Pathways  
1) **Temporal Decomposition**:  
   - **계절성(Seasonality)**: DFT로 주파수 변환 후 상위 $$K_f$$ 주파수만 선택, IDFT로 복원  
   - **추세(Trend)**: 잔여 $$(X - X^\text{sea})$$에 다중 커널 평균 풀링 후 가중치 합(식 5)  
2) **Multi-Scale Router**:  

$$
     R(X^\text{trans}) = \mathrm{Softmax}(X^\text{trans}W_r + \mathrm{Softplus}(X^\text{trans}W_\text{noise}))
   $$  
   
   $$-$$ Top-K 스케일 선택해 경로 가중치 생성.[1]
3) **Aggregator**: 선택된 스케일 출력 $$\{X_i^\text{out}\}$$에 대해 가중합(식 7) 수행.

***

## 3. 성능 향상 및 한계

### 3.1 성능 향상  
- 11개 데이터셋에서 88개 실험 중 81회 1위, 5회 2위 달성. PatchTST 대비 MSE 8.1%, MAE 6.4% 개선.[1]
- 전이 학습 실험: Pre-train → Fine-tune, Pathformer (Part-tuning) 방식은 컴퓨팅 자원 52% 절감하면서 거의 동일한 정확도 유지, 다른 모델 Full-tuning 대비 우수한 일반화 성능.[1]

### 3.2 한계  
- 패치 크기 후보 풀에 의존해 최적 스케일이 사전에 정의되어야 함.  
- 복잡한 구조로 인한 연산 비용 증가.  
- 매우 긴 시계열(수백만 길이)나 실시간 예측에서는 효율성 및 메모리 최적화 필요.

***

## 4. 일반화 성능 향상 관점  
- **Adaptive Pathways**는 시계열별 동적 특성(주기·추세)에 맞춰 스케일을 선택하므로, **이질적 데이터셋 간 전이** 시에도 적절한 스케일 조합이 자동으로 적용되어 일반화 성능이 크게 개선된다.[1]
- Part-tuning 전략을 통해 라우터 파라미터만 재조정, 모델 전체 재학습 부담을 경감하면서도 높은 정확도 유지.

***

## 5. 향후 연구 방향 및 고려 사항  
- 패치 크기 후보 풀 자동 탐색(AutoML) 및 라우터 초기화 최적화.  
- 효율적 구조 설계를 위한 연산 경량화 연구.  
- 멀티모달 시계열(텍스트·이미지 포함) 예측으로 확장.  
- 온라인 러닝에 대응하는 적응적 패치 분할 및 경로 업데이트 메커니즘 개발.  
- 불확실성 정량화(Uncertainty Quantification) 및 설명 가능성(Explainability) 강화.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5cddda8c-e827-4124-aebc-b0b59ce1c58e/2402.05956v5.pdf)
