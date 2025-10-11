# MICN: Multi-scale Local and Global Context Modeling for Long-term Series Forecasting

## **1. 핵심 주장과 주요 기여**

### **핵심 주장**
MICN은 기존 Transformer 기반 방법들의 **이차 복잡도(quadratic complexity)** 문제를 해결하면서도 **선형 복잡도(linear complexity)**로 더 뛰어난 성능을 달성할 수 있다는 것입니다.[1]

### **주요 기여**
- **컨볼루션 기반 구조**: Self-attention을 효율적으로 대체하는 MICN 구조로 선형 계산 복잡도와 메모리 비용 달성[1]
- **다중 분기 프레임워크**: 복잡하고 가변적인 시계열 데이터의 복잡한 temporal pattern을 깊이 있게 분석하는 다중 분기 구조[1]
- **Local-Global 구조**: 정보 집약과 장기 의존성 모델링을 위한 새로운 구조로 self-attention과 Auto-correlation 메커니즘을 능가[1]
- **성능 향상**: 다변량 예측에서 17.2%, 단변량 예측에서 21.6%의 상대적 성능 향상 달성[1]

## **2. 해결하고자 하는 문제와 제안 방법**

### **문제 정의**
시계열 예측에서 좋은 예측 방법은 다음 두 가지 특성을 가져야 한다고 정의합니다:[1]
1. **Local features 추출 능력**: 단기 변화를 측정
2. **Global correlations 모델링 능력**: 장기 트렌드를 측정

### **제안 방법**

#### **Multi-scale Hybrid Decomposition (MHDecomp)**
기존의 단일 커널 사용 방식을 개선하여 여러 다른 커널을 사용:

$$X_t = \text{mean}(\text{AvgPool}(\text{Padding}(X), \text{kernel}_1), ..., \text{AvgPool}(\text{Padding}(X), \text{kernel}_n))$$

$$X_s = X - X_t$$[1]

#### **Local Module (다운샘플링 컨볼루션)**

$$Y^{local,i}\_{s,l} = \text{Conv1d}(\text{Avgpool}(\text{Padding}(Y_{s,l}), \text{kernel}_i), \text{kernel}_i)$$[1]

여기서 stride = kernel_i로 설정하여 local features를 압축합니다.

#### **Global Module (등거리 컨볼루션)**

$$Y'^{i}\_{s,l} = \text{Norm}(Y^{local,i}\_{s,l}) + \text{Dropout}(\text{Tanh}(\text{IsometricConv}(Y^{local,i}_{s,l})))$$

$$Y^{global,i}\_{s,l} = \text{Norm}(Y_{s,l-1}) + \text{Dropout}(\text{Tanh}(\text{Conv1dTranspose}(Y'^{i}_{s,l}, \text{kernel}_i)))$$[1]

#### **복잡도 분석**
- **다운샘플링 컨볼루션**: $$O(iD^2 \frac{L}{i}) = O(LD^2)$$
- **등거리 컨볼루션**: $$O(\frac{L^2D^2}{i^2}) = O(cD^2)$$ (where c is constant)
- **전체 복잡도**: $$O(LD^2)$$ - 시퀀스 길이에 대해 선형[1]

## **3. 모델 구조**

### **전체 아키텍처**
1. **Multi-scale Hybrid Decomposition**: 입력 시계열을 trend-cyclical과 seasonal 부분으로 분해
2. **Trend-cyclical Prediction Block**: 선형 회귀 또는 평균을 사용한 트렌드 예측
3. **Seasonal Prediction Block**: MIC 레이어를 사용한 복잡한 계절성 패턴 모델링[1]

### **MIC (Multi-scale Isometric Convolution) Layer**
- **여러 분기**: 서로 다른 스케일 크기($$i = I/4, I/8, ...$$)로 다양한 temporal pattern 모델링
- **Local-Global Module**: 각 분기에서 local features와 global correlations 추출
- **Merge 연산**: Conv2d를 사용하여 다양한 패턴을 가중합으로 융합[1]

## **4. 성능 향상 및 실험 결과**

### **주요 성능 지표**
- **다변량 예측**: FEDformer 대비 평균 17.2% MSE 감소[1]
- **단변량 예측**: FEDformer 대비 평균 21.6% MSE 감소[1]
- **특히 Weather 데이터셋**에서 예측 길이별로 53-75%의 상대적 MSE 감소 달성[1]

### **6개 벤치마크 데이터셋에서의 성능**
ETT, Electricity, Exchange, Traffic, Weather, ILI 데이터셋 모든 예측 길이 설정에서 state-of-the-art 성능 달성[1]

### **Ablation Studies 결과**
- **Local-Global vs Auto-correlation**: Local-Global 구조가 더 우수한 성능 입증[1]
- **등거리 컨볼루션 vs Self-attention**: 짧은 시퀀스에서 등거리 컨볼루션이 더 효과적[1]
- **입력 길이 영향**: 입력 길이가 증가할수록 MICN의 성능이 점진적으로 향상[1]

## **5. 일반화 성능 향상 가능성**

### **핵심 일반화 메커니즘**

#### **등거리 컨볼루션의 일반화 우위**
"등거리 컨볼루션의 커널은 모든 훈련 데이터에 의해 결정되며, 이는 **전역 temporal inductive bias**(translation equivariance 등)를 도입하고 self-attention보다 **더 나은 일반화**를 달성할 수 있다"[1]

#### **강건성 분석**
- **노이즈 주입 실험**: 10%까지의 노이즈에 대해 좋은 강건성 보여줌
- **데이터 이상 변동**에 대한 우수한 대처 능력 (예: 장비 손상으로 인한 비정상적인 전력 데이터)[1]

#### **다양한 데이터셋에서의 일관된 성능**
- **주기성이 없는 Exchange 데이터셋**에서도 51% 평균 MSE 감소 달성
- "MICN이 실제 애플리케이션의 다양한 시계열 예측 작업에 잘 대처할 수 있음을 보여준다"[1]

#### **하이퍼파라미터 민감도**
다양한 $$i$$ 값(12,16 / 16,24 / 12,24 등)에서 유사한 성능을 보여 모델의 강건성 입증[1]

## **6. 한계점**

### **주요 한계사항**
- **단순한 트렌드 예측**: 선형 회귀나 평균을 사용한 trend-cyclical 예측은 복잡한 트렌드 정보에 대해 제한적[1]
- **데이터셋 의존성**: 일부 경우에서 masked self-attention이 더 효과적일 수 있으며, 이는 해당 데이터셋과 관련이 있다고 언급[1]
- **활성화 함수 선택**: Tanh 활성화 함수나 컨볼루션 연산 사용으로 인해 Auto-Correlation보다 약간 더 많은 시간이 소요될 수 있음[1]

## **7. 앞으로의 연구에 미치는 영향**

### **시계열 예측 분야에 미치는 영향**

#### **패러다임 전환**
- **효율성과 성능의 균형**: 선형 복잡도로 더 나은 성능을 달성하는 새로운 방향 제시
- **Local-Global 모델링**: CNN과 Transformer의 장점을 결합한 새로운 접근법

#### **기술적 기여**
- **등거리 컨볼루션**: Self-attention의 효과적인 대안으로 활용 가능
- **다중 스케일 분해**: 복잡한 시계열 패턴 분석을 위한 새로운 방법론

### **향후 연구 시 고려사항**

#### **모델 개선 방향**
1. **고급 트렌드 예측 방법**: 현재의 단순한 선형 회귀를 넘어서는 더 정교한 트렌드 예측 방법 개발 필요
2. **데이터셋별 최적화**: 서로 다른 특성을 가진 데이터셋에 대한 맞춤형 접근법 연구
3. **계산 효율성 최적화**: 활성화 함수와 컨볼루션 연산의 최적화를 통한 추가적인 효율성 개선

#### **확장 가능성**
1. **다른 도메인 적용**: 의료, 금융, 에너지 등 다양한 실제 응용 분야로의 확장
2. **멀티모달 시계열**: 여러 모드의 데이터를 동시에 처리하는 방법론 개발
3. **실시간 처리**: 스트리밍 데이터에 대한 실시간 예측 성능 최적화

#### **이론적 발전**
1. **일반화 이론**: 등거리 컨볼루션의 일반화 성능에 대한 더 깊은 이론적 분석
2. **최적 스케일 선택**: 다양한 시계열 특성에 따른 최적 스케일 크기 결정 방법론
3. **아키텍처 자동 설계**: 데이터 특성에 따른 자동 아키텍처 최적화 방법

이 논문은 시계열 예측 분야에서 효율성과 성능을 동시에 달성하는 새로운 방향을 제시하며, 특히 일반화 성능 측면에서 중요한 기여를 하고 있습니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/92259a0c-0748-4beb-b1d9-c2122d48183e/2222_micn_multi_scale_local_and_glo.pdf)
