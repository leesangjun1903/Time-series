# Dynamic and Multi-faceted Spatio-temporal Deep Learning for Traffic Speed Forecasting

### 1. 핵심 주장 및 주요 기여

본 논문(KDD 2021)은 교통 속도 예측을 위한 **DMSTGCN(Dynamic and Multi-faceted Spatio-Temporal Graph Convolution Network)**을 제안합니다. 핵심 주장은 기존 방법들이 정적 인접 행렬(static adjacency matrix)을 사용하여 도로 구간 간의 공간적 관계를 모델링하지만, 실제로는 시간대에 따라 두 도로 구간 간의 영향이 동적으로 변화한다는 것입니다. 또한 교통 속도뿐만 아니라 교통량(traffic volume)과 같은 보조 데이터(auxiliary data)를 효과적으로 통합해야 한다고 주장합니다.[1]

**주요 기여는 세 가지입니다:**

1. **동적 공간 의존성 학습 방법**: 정적 인접 행렬 대신 시간대별로 변화하는 동적 그래프를 학습하는 dynamic graph constructor와 dynamic graph convolution을 제안했습니다.[1]

2. **Multi-faceted fusion module**: 보조 특징(auxiliary feature)의 hidden states를 주 특징(primary feature)의 hidden states와 공간적·시간적으로 통합하는 범용 프레임워크를 제공합니다.[1]

3. **실증적 검증**: 실제 데이터셋에서 최첨단(state-of-the-art) 성능을 달성했으며, 명시적이고 해석 가능한 동적 공간 관계를 발견했습니다.[1]

### 2. 해결하고자 하는 문제

**핵심 문제**:[1]

1. **정적 그래프의 한계**: 기존 DGNN(Dynamic Graph Neural Network) 기반 방법들은 사전 정의되거나 학습된 정적 인접 행렬을 사용합니다. 하지만 두 도로 구간의 영향은 하루 중 시간대에 따라 동적으로 변화합니다. 예를 들어, 두 구간의 속도 패턴이 아침 러시아워에는 유사하지만 저녁 러시아워에는 완전히 다를 수 있습니다.[1]

2. **다중 측면 보조 데이터 활용 부족**: 교통 속도는 현재 속도뿐만 아니라 교통량 같은 다른 요인의 영향도 받습니다. 교통량이 급증한 후 교통 속도가 급락하는 패턴이 관찰되는데, 기존 방법은 이러한 다중 측면 데이터를 효과적으로 통합하지 못합니다.[1]

**기술적 과제**:[1]

- 교통 속도의 동적이고 암묵적인 공간 패턴 모델링
- 시간대별 그래프 생성 시 파라미터 폭발 문제(complexity $$O(TNN)$$ where $$T$$는 시간 슬롯 수, $$N$$은 노드 수)
- 보조 정보가 교통 속도 예측에 미치는 시공간적 특성 모델링의 어려움

### 3. 제안하는 방법

#### 3.1 Dynamic Graph Construction (수식 포함)

**문제**: 각 시간대별로 노드 쌍 간의 관계를 모델링하는 것은 계산 복잡도가 $$O(TNN)$$로 매우 높습니다.[1]

**해결책**: Tucker 분해(Tucker decomposition)에서 영감을 받아 인접 텐서(adjacency tensor)를 구성합니다.[1]

**수식**:[1]

세 개의 학습 가능한 임베딩 행렬과 하나의 코어 텐서를 정의합니다:
- $$E_t \in \mathbb{R}^{N_t \times d}$$: 시간 슬롯 임베딩
- $$E_s \in \mathbb{R}^{N_s \times d}$$: 소스 노드 임베딩  
- $$E_e \in \mathbb{R}^{N_e \times d}$$: 타겟 노드 임베딩
- $$E_k \in \mathbb{R}^{d \times d \times d}$$: 코어 텐서

인접 텐서는 다음과 같이 계산됩니다:

$$
A'_{t,i,j} = \sum_{o=1}^{d} \sum_{q=1}^{d} \sum_{r=1}^{d} E_k^{o,q,r} E_t^{t,o} E_e^{i,q} E_s^{j,r}
$$

$$
A''_{t,i,j} = \max(0, A'_{t,i,j})
$$

$$
A_{t,i,j} = \frac{e^{A''_{t,i,j}}}{\sum_{n=1}^{N_s} e^{A''_{t,i,n}}}
$$

이 방법은 복잡도를 $$O(N_t N N)$$에서 $$O(d^3 + N_t d + Nd)$$로 크게 감소시키며, 교통의 주기성(periodicity)을 고려하여 같은 시간대는 동일한 그래프를 공유합니다.[1]

#### 3.2 Dynamic Graph Convolution

**수식**:[1]

$$
H_l = \sum_{k=0}^{K} (A_{\phi(t)})^k H_l^t W_k
$$

여기서:
- $$H_l^t$$: $$l$$번째 블록의 temporal convolution layer 출력
- $$W_k$$: depth $$k$$에 대한 파라미터
- $$K$$: 최대 diffusion 단계
- $$A_{\phi(t)}$$: 시간 $$t$$에서의 동적 인접 행렬

이는 DCRNN의 diffusion 과정에서 영감을 받았으며, 시간대별로 다른 그래프에서 이웃 노드의 hidden states를 집계합니다.[1]

#### 3.3 Multi-faceted Fusion Module

**수식**:[1]

보조 특징과 주 특징 간의 상호 의존성을 모델링하기 위해 "propagation and aggregation" 패러다임을 따릅니다:

$$
H_{l}^{p'} = \text{Aggregate}(H_l^p, \text{Propagate}(H_l^a))
$$

**Propagation**:

$$
H_l^{ap} = \text{Propagate}(H_l^a) = A_{\phi(t)}^{ap} H_l^a W
$$

**Aggregation**:

$$
H_l^{p'} = \text{Aggregate}(H_l^p, H_l^{ap}) = H_l^p + H_l^{ap}
$$

여기서 $$A^{ap} \in \mathbb{R}^{N_t \times N_p \times N_a}$$는 주 노드와 보조 노드 간의 동적 그래프를 나타냅니다.[1]

**다중 보조 특징으로 확장**:

$$
H_l^{p'} = H_l^p + \sum_{m=1}^{M} A_{\phi(t)}^{apm} H_l^{am} W
$$

이 방법은 보조 노드와 주 노드가 정렬되지 않은(unaligned) 경우에도 작동하며, 전통적인 concatenation 방법보다 범용적입니다.[1]

#### 3.4 Temporal Convolution Layer

**수식**:[1]

Gating mechanism이 적용된 dilated convolution을 사용합니다:

$$
H_l^t = \tanh(W_{f,l} \star F_l) \odot \sigma(W_{g,l} \star F_l)
$$

여기서:
- $$\star$$: dilated convolution 연산
- $$\odot$$: element-wise multiplication
- $$\sigma(\cdot)$$: sigmoid 함수

#### 3.5 전체 손실 함수

**수식**:[1]

MAE(Mean Absolute Error)를 목적 함수로 사용합니다:

$$
L = \frac{1}{Q \times N_p} \sum_{i=t+1}^{t+Q} \sum_{j=1}^{N_p} |X_{i,j}^p - \hat{X}_{i,j}^p|
$$

### 4. 모델 구조

**DMSTGCN의 전체 아키텍처**:[1]

1. **입력 레이어**: 주 특징과 보조 특징을 fully connected layer로 변환
   - $$F_1^p = W_{in}^p X^p + b_{in}^p$$
   - $$F_1^a = W_{in}^a X^a + b_{in}^a$$

2. **$$L$$개의 블록**: 각 블록은 parallel 구조로 구성
   - Primary part: Temporal convolution + Dynamic graph convolution
   - Auxiliary part: Temporal convolution + Dynamic graph convolution
   - Multi-faceted fusion module로 두 part 통합

3. **Residual connections**:[1]
   - $$F_{l+1}^p = H_l^{p'} + F_l^p$$
   - $$F_{l+1}^a = H_l^a + F_l^a$$

4. **Skip connections**: 각 temporal convolution 후 hidden states를 출력 레이어로 연결

5. **출력 레이어**:[1]
   - $$H = ||\_{l=1}^{L} \text{reshape}(H_l^{p,t})$$
   - $$\hat{X}\_{t+1:t+Q}^p = W_{fc2} \cdot \text{ReLU}(W_{fc1} \cdot \text{ReLU}(H) + b_{fc1}) + b_{fc2}$$

**하이퍼파라미터**:[1]
- 블록 수: 8개
- Dilated ratio:[1]
- Graph convolution max depth: 2
- Channel size: 32
- Hidden dimension: 16
- Batch size: 64
- Learning rate: 0.001

### 5. 성능 향상

**실험 결과** (3개 데이터셋: PeMSD4, PeMSD8, England):[1]

DMSTGCN은 모든 데이터셋에서 일관되게 최고 성능을 달성했습니다:

**PeMSD4 (Horizon 12)**:
- DMSTGCN: MAE 1.8787, MAPE 0.0415, RMSE 4.3814
- 차선책(GWNet): MAE 1.9550, MAPE 0.0446, RMSE 4.5560
- 개선율: 약 3.9% MAE 감소[1]

**PeMSD8 (Horizon 12)**:
- DMSTGCN: MAE 1.5522, MAPE 0.0358, RMSE 4.0522
- 개선율: GWNet 대비 약 3.2% MAE 감소[1]

**England (Horizon 12)**:
- DMSTGCN: MAE 3.2554, MAPE 0.0493, RMSE 7.8060
- 개선율: MTGNN 대비 약 2.4% MAE 감소[1]

**Ablation Study 결과**:[1]

1. **DMSTGCN vs. w/o dynamic graph**: 동적 그래프가 정적 그래프보다 우수함을 입증
2. **DMSTGCN vs. w/o auxiliary part**: 보조 특징의 중요성 확인
3. **DMSTGCN vs. w/o multi-faceted fusion**: Graph-based fusion이 단순 concatenation보다 효과적

**Unaligned Auxiliary Information 실험**:[1]

보조 정보 비율을 0%, 25%, 50%, 75%, 100%로 변경한 실험에서 보조 정보가 많을수록 성능이 향상되어, unaligned 상황에서도 모델이 효과적으로 작동함을 확인했습니다.

### 6. 일반화 성능 향상 가능성

**강점**:[1]

1. **데이터 다양성**: 3개의 서로 다른 지역(San Francisco, San Bernardino, England), 다양한 샘플링 속도(5분, 15분), 다른 기간(2개월, 6개월)의 데이터셋에서 일관된 성능을 보였습니다.

2. **Unaligned data 처리**: 보조 노드와 주 노드가 공간적으로 정렬되지 않은 경우에도 작동하여, 실제 환경의 다양한 시나리오(예: 쇼핑몰 고객 흐름 예측 시 주변 도로 교통량 활용)에 적용 가능합니다.[1]

3. **범용 프레임워크**: 교통 속도 예측뿐만 아니라 다중 측면 시공간 그래프 시퀀스를 다루는 다른 문제에도 적용 가능한 일반적 프레임워크입니다.[1]

4. **해석 가능성**: 학습된 동적 그래프가 실제 교통 패턴(예: 러시아워 시간대의 높은 상관관계, 비러시아워의 낮은 상관관계)을 명시적으로 포착하여 해석 가능성을 제공합니다.[1]

5. **저랭크 구조 활용**: Tucker 분해를 통해 공간 의존성의 저랭크 특성을 활용하여, 파라미터 수를 크게 줄이면서도 효과적인 학습이 가능합니다.[1]

**일반화 한계 요소**:

논문에서 명시적으로 다루지는 않았지만, 다음과 같은 일반화 한계가 있을 수 있습니다:

1. **교통 주기성 가정**: 같은 시간대가 동일한 그래프를 공유한다는 가정은 특수한 이벤트(사고, 행사 등)가 발생할 때 한계가 있을 수 있습니다.

2. **센서 밀도**: 실험에 사용된 데이터셋은 모두 센서 기반으로, 센서가 희소한 지역에서의 일반화 성능은 불확실합니다.

3. **데이터 품질 의존성**: 결측값 보간(linear interpolation)에 의존하므로, 결측률이 높은 경우 성능 저하 가능성이 있습니다.

### 7. 한계

**명시된 한계**:

논문에서 직접적으로 명시한 한계는 없지만, 다음과 같은 기술적 한계를 유추할 수 있습니다:

1. **계산 복잡도**: Tucker 분해로 복잡도를 줄였지만, 여전히 3차원 텐서 연산이 필요하며, 대규모 네트워크(노드 수가 매우 많은 경우)에서는 계산 부담이 클 수 있습니다.[1]

2. **하이퍼파라미터 민감성**: 8개 블록, 특정 dilated ratio 패턴 등 많은 하이퍼파라미터가 있어, 새로운 도메인에 적용 시 튜닝이 필요합니다.[1]

3. **장기 예측 성능**: 실험은 최대 Horizon 12까지만 수행되었으며(약 1시간), 더 장기 예측에서의 성능은 검증되지 않았습니다.[1]

4. **보조 데이터 의존성**: 교통량과 같은 보조 데이터가 없는 경우 성능 저하가 예상되며, ablation study에서 보조 데이터 제거 시 성능이 감소함을 보였습니다.[1]

5. **외부 요인**: 날씨, 특수 이벤트, 사고 등 외부 요인을 직접적으로 모델링하지 않습니다. 실험에서는 시간 정보(time in a day, day in a week)만 입력으로 사용했습니다.[1]

### 8. 향후 연구에 미치는 영향 및 고려사항

**학술적 영향**:[1]

1. **동적 그래프 학습의 새로운 패러다임**: Tucker 분해를 활용한 동적 그래프 구성 방법은 시공간 그래프 신경망 연구에 새로운 방향을 제시했습니다. 이후 연구에서 텐서 분해 기반 동적 그래프 학습 방법론이 발전할 가능성이 높습니다.

2. **Multi-modal fusion 연구**: Graph 관점에서 unaligned multi-modal 데이터를 융합하는 방법론은 교통 예측뿐만 아니라 다른 시공간 예측 문제(날씨 예측, 에너지 수요 예측 등)에도 적용 가능합니다.

3. **해석 가능한 딥러닝**: 학습된 동적 그래프를 통해 명시적이고 해석 가능한 공간 관계를 발견한 점은, 블랙박스 모델에 대한 우려가 큰 교통 시스템 분야에서 중요한 기여입니다.

**향후 연구 시 고려사항**:

1. **더 복잡한 시간적 패턴 모델링**:
   - 주기성뿐만 아니라 비정기적 이벤트(사고, 공사, 특수 행사)를 모델링하는 방법 연구 필요
   - Attention mechanism이나 transformer 구조를 활용하여 더 세밀한 시간적 패턴 포착

2. **확장성(Scalability) 개선**:
   - 수천~수만 개의 노드를 가진 대규모 네트워크에 적용하기 위한 효율적인 그래프 학습 방법 연구
   - Graph sampling, hierarchical graph 구조 등을 고려

3. **Few-shot learning 및 Transfer learning**:
   - 센서 데이터가 부족한 지역이나 새로 설치된 센서에 대한 예측 성능 개선
   - 학습된 dynamic graph embedding을 다른 지역이나 도시로 전이하는 연구

4. **불확실성 정량화(Uncertainty Quantification)**:
   - 점 추정(point estimation)뿐만 아니라 예측의 불확실성을 정량화하는 방법 연구
   - Bayesian deep learning, conformal prediction 등을 활용

5. **인과 관계(Causality) 모델링**:
   - 상관관계뿐만 아니라 인과 관계를 명시적으로 모델링하여 더 강건한 예측 수행
   - Granger causality, structural causal model 등을 graph neural network와 결합

6. **다중 작업 학습(Multi-task Learning)**:
   - 교통 속도, 교통량, 사고 예측 등 여러 관련 작업을 동시에 학습하여 상호 보완
   - 각 작업 간의 관계를 동적 그래프로 모델링

7. **실시간 적응(Online Adaptation)**:
   - 교통 패턴이 시간에 따라 변화(예: 코로나19 이후 재택근무 증가)하는 상황에서 모델을 실시간으로 업데이트하는 방법
   - Continual learning, meta-learning 기법 활용

8. **외부 지식 통합**:
   - 도로 네트워크 구조, POI(Point of Interest) 정보, 인구통계학적 데이터 등 외부 지식을 효과적으로 통합
   - Knowledge graph embedding과 결합

9. **공정성(Fairness) 고려**:
   - 모든 지역에 대해 공평한 예측 성능을 보장하는 방법 연구
   - 데이터가 부족한 지역에 대한 편향 완화

10. **에너지 효율성**:
    - 실시간 교통 예측 시스템에서 계산 비용과 예측 정확도의 trade-off 최적화
    - Model compression, pruning, knowledge distillation 기법 활용

이 논문은 동적 그래프 학습과 다중 측면 데이터 융합을 통해 교통 속도 예측의 새로운 지평을 열었으며, 시공간 예측 문제 전반에 걸쳐 영향력 있는 연구 방향을 제시했습니다.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/df445dd2-6553-470a-b3af-6a8dc2c4bcbf/3447548.3467275.pdf)
