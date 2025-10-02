# FourierGNN: Rethinking Multivariate Time Series Forecasting from a Pure Graph Perspective

**핵심 주장**  
FourierGNN은 순수 그래프 신경망만으로 시계열의 시공간(spatiotemporal) 상호의존성을 통합 학습할 수 있음을 입증한다. 입력된 다변량 시계열을 각 시점·변수 값을 노드로 간주해 완전연결(hypervariate) 그래프를 구성한 뒤, 푸리에 공간에서 동작하는 **Fourier Graph Operator(FGO)** 를 반복 적용하여 예측 문제를 그래프 노드 예측으로 재구성한다.[1]

**주요 기여**  
1. **하이퍼베리아트 그래프 구조** 정의  
   - 길이 $$T$$, 변수 개수 $$N$$의 입력 윈도우를 $$N\!T$$ 노드의 완전연결 그래프로 표현하여 시공간 상호의존성을 단일 그래프에 통합.[1]
2. **푸리에 그래프 연산자(FGO)** 제안  
   - 노드 특성에 대한 그래프 합성곱을 푸리에 공간의 행렬 곱으로 대체하여 시간 영역의 $$O(n^2)$$ 복잡도를 $$O(n\log n)$$으로 감소.[1]
3. **FourierGNN 모델 구조**  
   - 입력 노드를 d차원 임베딩 후 DFT→연속 FGO 레이어→IDFT→FFN을 통해 다음 $$L$$스텝 예측 수행.[1]
4. **이론적 분석**  
   - FGO가 시간 도메인의 그래프 합성곱과 동등함을 증명(명제1)하여 모델의 타당성과 표현력을 보장.[1]
5. **실험적 검증**  
   - 7개 공개 데이터셋에서 평균 10% 이상 정확도 개선, 파라미터 20~30% 절감 및 학습시간 15~25% 단축.[1]

***

## 1. 해결 문제  
기존 GNN 기반 시계열 예측은  
- **공간 네트워크(GCN/GAT)** 와 **시간 네트워크(LSTM/TCN/Transformer)** 를 별도로 설계해야 하며,  
- 이 둘의 결합 호환성 부재로 **통합 시공간 의존성** 학습에 한계가 있었다.[1]

FourierGNN은 이 문제를  
- **순수 그래프 관점**으로 재정의하고,  
- 시간 네트워크 없이도 하나의 그래프 네트워크에서 시공간 관계를 동시에 학습하도록 설계하였다.[1]

***

## 2. 제안 방법

### 2.1 Hypervariate Graph  
- 입력 $$\mathbf{X}\_t\in\mathbb{R}^{N\times T}$$의 각 값 $$x_{n,t}$$를 개별 노드로 보고 $$NT$$개의 노드를 완전연결.[1]
- 그래프 $$\mathcal{G}_t=(\mathbf{X}^G_t,\mathbf{A}^G_t)$$에서  

$$\mathbf{A}^G_t=\mathbf{1}_{NT\times NT}$$, $$\mathbf{X}^G_t\in\mathbb{R}^{NT\times1}$$  

### 2.2 Fourier Graph Operator (FGO)  
- 그래프 합성곱 $$\mathbf{A}\mathbf{X}\mathbf{W}$$를 DFT 기반 행렬곱 $$\mathcal{F}(\mathbf{X})\,S\,$$로 대체.[1]
- 계산 복잡도:  

$$
    \text{DFT/IDFT }O(n\log n) + \text{ 행렬곱 }O(nd^2)
    \;\ll\;O(n^2d + nd^2)
  $$  

- $$S\in\mathbb{C}^{d\times d}$$는 푸리에 공간의 학습 가능한 커널.[1]

### 2.3 모델 구조  
1. **임베딩**: $$\mathbf{X}^G_t\!E\to\mathbf{X}^G_t\in\mathbb{R}^{NT\times d}$$  
2. **DFT**: $$\mathbf{X}^G_t\to\widehat{\mathbf{X}}^G_t\in\mathbb{C}^{NT\times d}$$  
3. **FGO 반복**:

$$
     \widehat{\mathbf{Y}}^G_t
     = \sum_{k=0}^K 
       \widehat{\mathbf{X}}^G_t\,S_k + b_k
   $$

4. **IDFT**: $$\widehat{\mathbf{Y}}^G_t\to\mathbf{Y}^G_t\in\mathbb{R}^{NT\times d}$$  
5. **FFN**: $$\mathbf{Y}^G_t\to\mathbf{\hat{Y}}_t\in\mathbb{R}^{N\times L}$$  

수식 내 DFT/IDFT $$\mathcal{F},\mathcal{F}^{-1}$$, 네트워크 깊이 $$K$$, 바이어스 $$b_k$$.[1]

***

## 3. 성능 향상 및 한계

### 3.1 성능 향상  
- 7개 벤치마크(METR-LA, Traffic, Solar, Wiki 등)에서 MAE 9.4%↓, RMSE 10.9%↓.[1]
- 파라미터 20~32% 절감, 학습시간 15~23% 단축.[1]
- 다중 스텝 예측(Multi-step)에서도 기존 GNN 모델 대비 MAE·RMSE 약 30% 개선.[1]

### 3.2 일반화 성능 향상  
- **n-불변 FGO**(parameter $$S\in\mathbb{C}^{d\times d}$$)를 도입해 그래프 크기 변화에 불변하고 과적합 완화.[1]
- **노드 임베딩** 및 **동적 FGO**(각 확산 차수별 다른 $$S_k$$) 구성이 일반화 성능에 핵심 기여.[1]

### 3.3 한계  
- **복잡도**: 실세계 대규모 $$N\!T$$ 그래프에서는 여전히 메모리 부담.  
- **비선형성 처리**: 푸리에 공간의 복소수 연산이 일부 데이터에서 최적화 난이도 상승.  
- **시간 불변성 가정**: 주기적·비정상 시계열엔 DFT 기반 한계 가능성.

***

## 4. 앞으로의 연구 영향 및 고려 사항

- **순수 그래프 기반 시계열 예측**의 새로운 방향 제시: 그래프 신경망 단일 구조로 시공간 통합 학습 가능.  
- **주파수 도메인 연산**의 확장: DWT, 웨이블릿 등 다양한 변환과 결합한 모델 연구.  
- **비정상·비주기 데이터** 대응: 적응형 스펙트럼 변환 기법 필요.  
- **효율성 제고**: 희소 DFT, 메시지 패싱 최적화로 대규모 적용성 강화.  
- **이론적 확장**: FGO의 수렴성과 안정성 해석 연구.  

이 논문은 **다변량 시계열 예측** 및 **그래프 신경망** 연구에 새로운 패러다임을 제공하며, 후속 연구에서는 주파수 변환 기반 모델의 일반화·효율성·안정성 강화에 초점을 맞출 필요가 있다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/55e65316-d7d4-4627-89a5-03e3211fa4d2/2311.06190v1.pdf)
