# MSGNet: Learning Multi-Scale Inter-Series Correlations for Multivariate Time Series Forecasting

## **1. 핵심 주장과 주요 기여**

**MSGNet**의 핵심 주장은 다변량 시계열에서 **서로 다른 시간 스케일에 따른 변동하는 계열 간 상관관계(inter-series correlations)**를 캐치하는 것이 예측 정확도 향상에 결정적이라는 것입니다.[1]

### **주요 기여**

1. **다중 스케일 계열 간 상관관계 모델링**: 기존 연구들이 간과한 서로 다른 시간 스케일에서의 변동하는 계열 간 상관관계를 효과적으로 발견하고 포착하는 새로운 MSGNet 구조를 제안[1]

2. **혁신적인 모듈 조합**: 계열 간 상관관계(adaptive graph convolution)와 계열 내 상관관계(multi-head attention) 모델링을 동시에 수행하는 통합 프레임워크 도입[1]

3. **우수한 실증 성능과 일반화 능력**: 실제 데이터셋에서 기존 딥러닝 모델들을 일관되게 능가하며, 특히 분포 외 샘플(out-of-distribution samples)에서도 강력한 일반화 성능을 보임[1]

## **2. 해결하고자 하는 문제와 제안 방법**

### **문제 정의**
기존 딥러닝 모델들은 **다변량 시계열에서 서로 다른 시간 스케일에 따라 변동하는 계열 간 상관관계**를 정확히 기술하지 못합니다. 예를 들어 금융 시장에서 자산 간 상관관계는 시장 불안정기에는 증가하지만 경제 성장기에는 감소하는 패턴을 보입니다.[1]

### **제안 방법**

**MSGNet**은 세 가지 핵심 구성요소를 통합한 프레임워크입니다:

#### **1) Scale Identification (스케일 식별)**
Fast Fourier Transform(FFT)을 사용하여 주요 주기성을 시간 스케일로 감지합니다:

$$F = \text{Avg}(\text{Amp}(\text{FFT}(X_{\text{emb}})))$$

$$f_1, \ldots, f_k = \arg\text{Top}_k(F), \quad s_i = \frac{L}{f_i}$$[1]

여기서 $$F$$는 각 주파수의 진폭이고, Top-k 연산을 통해 가장 두드러진 k개 스케일을 선택합니다.

#### **2) Multi-scale Adaptive Graph Convolution**
각 시간 스케일 $$i$$에 대해 적응적 인접 행렬을 학습합니다:

$$A^i = \text{SoftMax}(\text{ReLU}(E^i_1(E^i_2)^T))$$[1]

그리고 Mixhop graph convolution을 적용합니다:

$$H^i_{\text{out}} = \sigma\left(\parallel_{j \in P} (A^i)^j H^i\right)$$[1]

여기서 $$P$$는 인접행렬의 거듭제곱 집합이고, $$\parallel$$은 열 단위 연결을 나타냅니다.

#### **3) Multi-head Attention and Scale Aggregation**
각 스케일에서 Multi-head Attention을 적용하고:

$$\hat{X}^i_{\text{out}} = \text{MHA}_s(\hat{X}^i)$$[1]

Mixture of Experts 전략으로 스케일을 통합합니다:

$$\hat{X}_{\text{out}} = \sum_{i=1}^k \hat{a}_i \hat{X}^i_{\text{out}}$$

여기서 $$\hat{a}\_1, \ldots, \hat{a}\_k = \text{SoftMax}(F_{f_1}, \ldots, F_{f_k})$$[1]

## **3. 모델 구조**

MSGNet은 **ScaleGraph Block**들을 쌓은 구조로, 각 블록은 다음 4단계를 수행합니다:[1]

1. **입력 시계열의 스케일 식별**: FFT를 통한 주요 주기성 탐지
2. **스케일별 계열 간 상관관계 학습**: Adaptive graph convolution 적용  
3. **계열 내 상관관계 캐처**: Multi-head attention 메커니즘
4. **다중 스케일 표현 통합**: SoftMax 기반 적응적 집계

**Input Embedding**은 1-D convolution과 positional encoding, timestamp embedding을 결합하여 생성됩니다:

$$X_{\text{emb}} = \alpha \text{Conv1D}(\hat{X}\_{t-L:t}) + PE + \sum_{p=1}^P SE_p$$[1]

**Residual Connection**을 통해 안정적인 학습이 보장됩니다:

$$X^l = \text{ScaleGraphBlock}(X^{l-1}) + X^{l-1}$$[1]

## **4. 성능 향상 및 한계**

### **성능 향상**

1. **우수한 예측 정확도**: 8개 데이터셋에서 5개 데이터셋에서 최고 성능, 2개에서 두 번째 성능을 달성했습니다. Flight 데이터셋에서는 TimesNet 대비 MSE를 21.5%, MAE를 13.7% 감소시켰습니다[1]

2. **계산 효율성**: TimesNet보다 운영 효율성이 뛰어나며, 입력 시간이 증가해도 일정한 시간 비용을 유지합니다[1]

3. **스케일별 해석 가능성**: 학습된 인접 행렬이 실제 현상을 반영합니다. 예를 들어, 공항 간 거리가 먼 경우 장기 스케일에서 강한 영향을 미치지만 단기에서는 약해집니다[1]

### **한계**

1. **계산 복잡도**: DLinear, Autoformer 같은 단순 모델보다 GPU 메모리와 계산 자원을 더 많이 사용합니다[1]

2. **하이퍼파라미터 민감성**: 스케일 수(k), Mixhop order 등 핵심 하이퍼파라미터의 최적 설정이 필요합니다[1]

## **5. 일반화 성능 향상**

MSGNet의 **일반화 성능 향상**은 특히 주목할 만한 특징입니다:

### **Out-of-Distribution (OOD) 샘플에 대한 강건성**

COVID-19 팬데믹이라는 극단적 외부 충격을 활용한 실험에서, MSGNet은 **가장 적은 성능 저하(21.29%)**를 보였습니다. 이는 다른 모델들(TimesNet: 26.47%, DLinear: 42.29%, MTGnn: 45.74%)에 비해 월등히 뛰어난 결과입니다.[1]

### **일반화 능력의 원리**

1. **다중 계열 간 상관관계**: MSGNet이 포착하는 다중 계열 간 상관관계 중 일부는 OOD 상황에서도 여전히 유효하게 작동합니다[1]

2. **Mixhop Graph Convolution의 표현력**: Two-hop Delta Operator와 같은 다양한 특징을 학습할 수 있어 고정된 계열 간 상관관계 패턴에 제약받지 않습니다[1]

3. **스케일별 로버스트성**: 외부 교란이 발생해도 특정 스케일의 계열 간 상관관계만 변화하고 다른 상관관계는 영향받지 않아 전체적인 일반화 성능이 유지됩니다[1]

## **6. 미래 연구에 미치는 영향 및 고려사항**

### **연구 영향**

1. **다중 스케일 모델링의 중요성**: 시계열 예측에서 서로 다른 시간 스케일의 계열 간 상관관계를 고려하는 것이 핵심적임을 실증적으로 입증했습니다

2. **그래프 신경망과 어텐션의 효과적 결합**: GNN과 Transformer의 장점을 살린 하이브리드 아키텍처의 가능성을 제시했습니다

3. **해석 가능한 AI**: 학습된 인접 행렬을 통해 모델의 의사결정 과정을 시각화할 수 있어 설명 가능한 AI 연구에 기여할 수 있습니다

### **향후 연구 고려사항**

1. **계산 효율성 최적화**: 더 경량화된 아키텍처 개발을 통한 실용성 향상 필요

2. **동적 스케일 학습**: 시간에 따라 변화하는 스케일 패턴을 더 정교하게 모델링하는 방법 연구

3. **다양한 도메인 적용**: 금융, 에너지, 교통 등 다양한 시계열 예측 태스크에서의 일반화 성능 검증

4. **이론적 분석 강화**: 다중 스케일 계열 간 상관관계의 이론적 기반 구축 및 수렴성 보장

MSGNet은 시계열 예측 분야에서 **다중 스케일 계열 간 상관관계**라는 새로운 연구 방향을 제시하며, 특히 분포 외 샘플에 대한 강건성을 통해 실제 응용에서의 신뢰성을 크게 향상시켰다는 점에서 중요한 의미를 갖습니다.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/839d44aa-8942-47fb-899e-a8ef542771f6/2401.00423v1.pdf)
