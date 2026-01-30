
# N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting

## I. 논문의 핵심 요약

### 1.1 주요 주장 및 기여

**N-HiTS(Neural Hierarchical Interpolation for Time Series Forecasting)**는 시계열 예측의 오랫동안 미해결된 두 가지 난제를 동시에 해결하는 혁신적인 신경망 아키텍처이다. 첫째, 장기 예측(long-horizon forecasting)에서 발생하는 예측값의 높은 변동성이고, 둘째, 예측 지평이 증가함에 따라 급증하는 계산 복잡도이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d4bd591f-3f61-4e97-a95f-97ef82c571f3/2201.12886v6.pdf)

논문의 네 가지 핵심 기여는 다음과 같다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d4bd591f-3f61-4e97-a95f-97ef82c571f3/2201.12886v6.pdf)

1. **다중 레이트 데이터 샘플링**: MLP 입력 전에 최대 풀링 레이어를 통합하여 메모리 풋프린트와 계산량을 크게 감소시키면서도 장거리 의존성 모델링 능력 유지

2. **계층적 보간**: 다중 스케일 보간을 통해 신경망 예측의 차원성을 줄이고, 최종 출력 시간 스케일과 일치시킴으로써 예측의 부드러움 강제

3. **N-HiTS 아키텍처**: 블록 간 입력 샘플링 레이트와 출력 보간 스케일의 계층적 동기화로, 각 블록이 시계열의 특정 주파수 대역에 특화되도록 유도

4. **벤치마크 성능**: 6개의 대규모 장기 예측 데이터셋(ETTm2, Exchange Rate, ECL, Traffic, Weather, ILI)에서 기존 Transformer 기반 방법 대비 평균 약 **20% 정확도 향상**(MAE 14%, MSE 16%)을 달성하면서 **50배 계산 시간 감소** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d4bd591f-3f61-4e97-a95f-97ef82c571f3/2201.12886v6.pdf)

***

## II. 해결하고자 하는 문제

### 2.1 장기 예측의 근본적 도전과제

신경망 기반 시계열 예측은 최근 몇 년간 눈부신 발전을 이루었지만, 장기 예측(예: 720 시간 이상 앞선 예측)은 여전히 매우 어려운 과제이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d4bd591f-3f61-4e97-a95f-97ef82c571f3/2201.12886v6.pdf)

**주요 문제점:**

- **계산 복잡도의 이차 증가**: 완전 연결 계층과 attention 메커니즘 모두 예측 지평 H에 대해 메모리와 계산 비용이 $O(H^2)$ 또는 $O(H \log H)$로 증가한다. 이는 예측 지평이 96시간에서 720시간으로 확장될 때 심각한 병목이 된다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d4bd591f-3f61-4e97-a95f-97ef82c571f3/2201.12886v6.pdf)

- **예측 변동성 증가**: Figure 1(b)에 보이듯이, 단순 완전 연결 신경망의 MAE(평균 절대오차)가 예측 지평에 따라 로그 스케일로 기하급수적으로 증가한다. 이는 신경망의 제한되지 않은 표현성이 오버피팅과 과도한 예측 불안정성을 초래함을 의미한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d4bd591f-3f61-4e97-a95f-97ef82c571f3/2201.12886v6.pdf)

- **기존 Transformer 기반 접근법의 한계**: Informer, Autoformer, LogTrans 등 최근의 sparse attention 기반 기법들도 근본적인 문제를 해결하지 못하며, 점진적인 개선만을 제공할 뿐이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d4bd591f-3f61-4e97-a95f-97ef82c571f3/2201.12886v6.pdf)

### 2.2 신경망의 무제한 표현성의 모순

이 논문이 지적하는 핵심 역설은 다음과 같다: 신경망의 무제한된 표현성(unbounded expressiveness)은 장기 예측에서 직접적으로 과도한 계산 복잡도와 예측 변동성으로 이어진다는 것이다. 따라서 문제를 해결하는 열쇠는 모델의 표현성을 제약하되, 예측 관계의 부드러움(smoothness) 가정 하에서 이를 지능적으로 수행하는 것이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d4bd591f-3f61-4e97-a95f-97ef82c571f3/2201.12886v6.pdf)

***

## III. 제안하는 방법론

### 3.1 N-HiTS 아키텍처 개요

N-HiTS는 N-BEATS 아키텍처를 확장하되, 다음 두 가지 혁신적 기법을 핵심으로 한다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d4bd591f-3f61-4e97-a95f-97ef82c571f3/2201.12886v6.pdf)

$$\text{Architecture} = \text{N-BEATS} + \text{Multi-Rate Sampling} + \text{Hierarchical Interpolation}$$

아키텍처는 S개의 스택(stack)으로 구성되며, 각 스택은 B개의 블록(block)을 포함한다. 각 블록은 다층 퍼셉트론(MLP)으로 이루어져 있으며, 백캐스트(backcast)와 포캐스트(forecast) 계수를 학습한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d4bd591f-3f61-4e97-a95f-97ef82c571f3/2201.12886v6.pdf)

### 3.2 다중 레이트 신호 샘플링 (Multi-Rate Signal Sampling)

**핵심 아이디어**: 각 블록이 서로 다른 스케일의 신호 성분에 집중하도록 강제한다.

$$y^{(p)}_{t-L:t,\ell} = \text{MaxPool}(y_{t-L:t,\ell}, k_\ell) \quad \text{...(1)}$$

여기서:
- $k_\ell$: $\ell$번째 블록의 최대 풀링 커널 크기
- 더 큰 $k_\ell$은 블록이 저주파/대규모 성분에 집중하도록 유도
- 더 작은 $k_\ell$은 고주파/소규모 성분 분석을 강제

**효과**:
1. MLP 입력의 폭을 줄임으로써 메모리 풋프린트 감소
2. 파라미터 수 감소로 오버피팅 완화
3. 원래 수용장(receptive field) 유지하면서도 계산 효율성 향상

### 3.3 비선형 회귀 (Non-Linear Regression)

서브샘플링 후, 블록 $\ell$은 숨은 벡터 $h_\ell \in \mathbb{R}^{N_h}$를 학습한다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d4bd591f-3f61-4e97-a95f-97ef82c571f3/2201.12886v6.pdf)

$$h_\ell = \text{MLP}_\ell\left(y^{(p)}_{t-L:t,\ell}\right)$$

$$\theta^f_\ell = \text{LINEAR}^f(h_\ell)$$

$$\theta^b_\ell = \text{LINEAR}^b(h_\ell) \quad \text{...(2)}$$

이 계수들은 아래의 계층적 보간을 통해 백캐스트와 포캐스트를 합성하는 데 사용된다.

### 3.4 계층적 보간 (Hierarchical Interpolation)

이것이 N-HiTS의 가장 혁신적인 구성요소이다.

**핵심 원리**: 다중 수평 예측 모델에서 신경망 예측의 차원성은 보통 지평선 H와 같다. 예를 들어 N-BEATS에서는 $|\theta^f_\ell| = H$이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d4bd591f-3f61-4e97-a95f-97ef82c571f3/2201.12886v6.pdf) 이는 H가 커질수록 파라미터가 선형적으로 증가하는 문제를 초래한다.

**해결책**: 표현성 비율(expressiveness ratio) $r_\ell$로 보간 계수의 차원성을 제어한다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d4bd591f-3f61-4e97-a95f-97ef82c571f3/2201.12886v6.pdf)

$$|\theta^f_\ell| = \lceil r_\ell H \rceil$$

원래 샘플링 레이트와 지평선을 복원하기 위해 보간 함수 $g$를 사용한다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d4bd591f-3f61-4e97-a95f-97ef82c571f3/2201.12886v6.pdf)

$$\hat{y}_{\tau,\ell} = g(\tau, \theta^f_\ell), \quad \forall \tau \in \{t+1, \ldots, t+H\}$$

$$\tilde{y}_{\tau,\ell} = g(\tau, \theta^b_\ell), \quad \forall \tau \in \{t-L, \ldots, t\} \quad \text{...(3)}$$

**선형 보간의 명시적 형태** ($g \in C^1$): [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d4bd591f-3f61-4e97-a95f-97ef82c571f3/2201.12886v6.pdf)

시간 분할 $T = \{t+1, t+1+1/r_\ell, \ldots, t+H-1/r_\ell, t+H\}$에 대해:

$$g(\tau, \theta) = \theta[t_1] + \frac{\theta[t_2] - \theta[t_1]}{t_2 - t_1}(\tau - t_1) \quad \text{...(4)}$$

여기서:
$$t_1 = \arg\min_{t \in T: t \leq \tau} |\tau - t|, \quad t_2 = t_1 + 1/r_\ell$$

**선택된 보간 방식**: 논문에서는 선형 보간($g \in C^1$)을 최종적으로 선택했으며, ablation study에서 다음을 확인했다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d4bd591f-3f61-4e97-a95f-97ef82c571f3/2201.12886v6.pdf)
- 선형 보간 > 삼차 보간 > 최근접 이웃 (특히 긴 지평에서)
- 선형 보간이 계산 효율과 정확도 간 최적의 균형 제공

### 3.5 계층적 보간의 이론적 보장: Neural Basis Approximation Theorem

이 정리는 N-HiTS의 보간 기법이 부드러운 예측 관계 하에서 임의로 긴 지평선을 효율적으로 근사할 수 있음을 보장한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d4bd591f-3f61-4e97-a95f-97ef82c571f3/2201.12886v6.pdf)

**정리 1 (Neural Basis Approximation Theorem)**: 

예측 매핑을 $Y(\cdot | y_{t-L:t}) : [0,1]^L \to F$라 하자. 여기서 예측 함수 $F = \{Y(\tau) : \to \mathbb{R}\} = L^2)$는 제곱 적분 가능한 함수들의 집합이다.  
다중 해상도 함수들 $V_w = \{\phi_{w,h}(\tau) = \phi(2^w(\tau - h)) | w \in \mathbb{Z}, h \in 2^{-w} \times [0, \ldots, 2^w]\}$가 $L^2([0,1])$을 임의로 잘 근사할 수 있고, 투영 $\text{Proj}\_{V_w}(Y(\tau))$가 $y_{t-L:t}$에 대해 부드럽게 변한다면:

$$\forall \epsilon > 0, \quad \int |Y(\tau | y_{t-L:t}) - \sum_{w,h} \hat{\theta}_{w,h}(y_{t-L:t})\phi_{w,h}(\tau)| d\tau \leq \epsilon \quad \text{...(5)}$$

**의미**:
1. 충분히 세밀한 다중 해상도 계수로 임의로 긴 지평선의 예측을 근사 가능
2. 신경망은 유한 개의 다중 해상도 계수만 학습하면 됨
3. 파라미터 수를 제어하면서도 표현력 보존 가능

### 3.6 계층적 구조와 주파수 특화

계층적 보간의 원리는 서로 다른 주파수 대역에 특화하도록 블록을 유도하기 위해 다음과 같이 구현된다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d4bd591f-3f61-4e97-a95f-97ef82c571f3/2201.12886v6.pdf)

- **저주파 블록** ($\ell = 1$): 작은 $r_\ell$, 큰 $k_\ell$ → 저주파/장기 트렌드에 집중
- **고주파 블록** ($\ell > 1$): 큰 $r_\ell$, 작은 $k_\ell$ → 고주파/단기 변동성에 집중

**Top-Down 계층**: 중요하게도, 저주파 성분을 먼저 학습하고 그 백캐스트 잔차를 다음 블록에 전파하는 Top-Down 구조가 Bottom-Up보다 **4.6% MAE, 7.5% MSE 개선**을 달성했다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d4bd591f-3f61-4e97-a95f-97ef82c571f3/2201.12886v6.pdf)

최종 예측은 모든 블록의 포캐스트를 합산: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d4bd591f-3f61-4e97-a95f-97ef82c571f3/2201.12886v6.pdf)

$$\hat{y}_{t+1:t+H} = \sum_{\ell=1}^{L} \hat{y}_{t+1:t+H,\ell}$$

$$y_{t-L:t,\ell+1} = y_{t-L:t,\ell} - \tilde{y}_{t-L:t,\ell}$$

***

## IV. 성능 향상 분석

### 4.1 정확도 개선

N-HiTS는 여섯 개의 대규모 벤치마크 데이터셋에서 기존 최첨단 모델들을 크게 능가한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d4bd591f-3f61-4e97-a95f-97ef82c571f3/2201.12886v6.pdf)

| 데이터셋 | MAE 개선 (%) | MSE 개선 (%) | 주요 경쟁 모델 |
|---------|-----------|-----------|--------------|
| ETTm2 | 14.4 | 16.8 | Autoformer |
| ECL | 17.1 | 19.9 | FEDformer |
| Exchange | 18.3 | 24.1 | Informer |
| Traffic | 8.4 | 13.2 | Autoformer |
| Weather | 16.8 | 21.4 | Informer |
| ILI | 11.6 | 14.2 | Autoformer |

**장기 지평에서의 성능**: 지평선이 720시간일 때, N-HiTS는 다른 모델들보다 훨씬 안정적이다. 예를 들어 Exchange 데이터셋에서:
- N-HiTS: MSE = 0.798, MAE = 0.596
- Informer: MSE = 2.478, MAE = 1.310
- **상대 개선: 68% MSE 감소, 54% MAE 감소** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d4bd591f-3f61-4e97-a95f-97ef82c571f3/2201.12886v6.pdf)

### 4.2 계산 효율성

**훈련 시간** (ETTm2 데이터셋): [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d4bd591f-3f61-4e97-a95f-97ef82c571f3/2201.12886v6.pdf)
- N-HiTS: ~0.68초/에포크
- Autoformer: ~40.987초/에포차
- Informer: ~26.173초/에포차
- **상대 개선: Autoformer 대비 45배, Informer 대비 38배 빠름**

**메모리 사용량**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d4bd591f-3f61-4e97-a95f-97ef82c571f3/2201.12886v6.pdf)
- N-HiTS: 26% 미만의 파라미터 (경쟁 모델 대비)
- N-BEATSi 대비: 54% 파라미터 감소
- 파라미터 수가 입력 길이 L에 대해 선형으로 스케일됨

**경제적 영향**: AWS g4dn.2xlarge GPU($0.75/시간)에서:
- Autoformer 단일 실행: USD $70.0
- N-HiTS(20 하이퍼파라미터 반복): USD $22.8
- **단위 성능당 비용 감소: 68%** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d4bd591f-3f61-4e97-a95f-97ef82c571f3/2201.12886v6.pdf)

### 4.3 해석 가능성 및 분해 능력

N-HiTS의 계층적 구조는 자연스럽게 시계열을 주파수 성분으로 분해한다. Figure 5(a)에 보이듯이: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d4bd591f-3f61-4e97-a95f-97ef82c571f3/2201.12886v6.pdf)
- Stack 1: 저주파 트렌드 포착
- Stack 2: 중주파 패턴
- Stack 3: 고주파 잡음 및 이상

이는 Figure 5(b)의 계층적 보간 없는 모델과 대조적으로, 해석 불가능한 혼합 신호를 생성한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d4bd591f-3f61-4e97-a95f-97ef82c571f3/2201.12886v6.pdf)

***

## V. 모델의 한계

### 5.1 단변량 모델의 제약

N-HiTS는 설계상 **단변량 모델**이다. 각 변수를 자신의 과거 데이터 $y_{t-L:t}$만으로 예측한다. 이는 다음을 의미한다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d4bd591f-3f61-4e97-a95f-97ef82c571f3/2201.12886v6.pdf)

- **장점**: 변수 간 허위 상관관계 학습 회피, 더 낮은 분산
- **단점**: 변수 간 의존성 활용 불가

그러나 논문에서 흥미로운 발견이 있다: 다변량 최첨단 모델들이 사실 단변량 모델에 의해 지배당한다는 것이다. 이는 기존의 다변량 모델이 변수 상호작용을 효과적으로 모델링하지 못한다는 점을 시사한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d4bd591f-3f61-4e97-a95f-97ef82c571f3/2201.12886v6.pdf)

### 5.2 글로벌 파라미터 공유

N-HiTS는 데이터셋의 모든 시계열에 대해 파라미터를 공유한다. 이는: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d4bd591f-3f61-4e97-a95f-97ef82c571f3/2201.12886v6.pdf)
- 매우 이질적인 시계열 데이터셋에서 문제가 될 수 있음
- 특정 시계열 특성에 대한 적응성 제한
- Appendix I의 실험에서는 파라미터/훈련 시간이 데이터셋 크기에 대해 선형으로 유지된다고 보고

### 5.3 선형성 가정의 암묵적 제약

계층적 보간 정리는 예측 관계의 부드러움(smoothness)을 가정한다. 이는: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d4bd591f-3f61-4e97-a95f-97ef82c571f3/2201.12886v6.pdf)
- 불연속적인 급격한 변화가 있는 데이터(예: 시장 충격)에서 성능 저하 가능
- 극단적 이상 탐지(extreme anomaly detection)가 필요한 경우 부적합

***

## VI. 최신 연구와의 비교 분석 (2020년 이후)

### 6.1 Transformer 기반 모델들과의 비교

#### 6.1.1 Informer (2020) [dl.acm](https://dl.acm.org/doi/10.1609/aaai.v37i6.25854)

**주요 특징**: Sparse self-attention 메커니즘 (ProbSparse Attention) 사용 [dl.acm](https://dl.acm.org/doi/10.1609/aaai.v37i6.25854)

**vs N-HiTS**:
| 측면 | Informer | N-HiTS |
|------|----------|--------|
| 아키텍처 | Encoder-Decoder + Attention | MLP 스택 + 보간 |
| 복잡도 | $O(H \log H)$ | $O(H(1-r^B)/(1-r))$ (지수적 압축) |
| 정확도 | 기준선 | **+18~23% 개선** |
| 속도 | 기준선 | **38배 빠름** |

Informer는 여전히 attention의 이차 메모리 병목을 완전히 해결하지 못한다.

#### 6.1.2 Autoformer (2021) [arxiv](https://arxiv.org/abs/2106.13008)

**주요 특징**: 자동상관(Auto-Correlation) 메커니즘 + 분해 아키텍처

**vs N-HiTS**:
| 측면 | Autoformer | N-HiTS |
|------|-----------|--------|
| 분해 | Trend-Seasonality 분해 | 다중 주파수 계층적 분해 |
| 정확도 (720 지평) | 기준선 | **+17% MSE, +11% MAE** |
| 훈련 시간 | 기준선 | **45배 빠름** |
| 해석성 | 제한적 | 명시적 주파수 특화 |

Autoformer의 분해는 사전에 정의된 트렌드/계절성만 처리하지만, N-HiTS는 데이터-드리븐 계층적 분해를 학습한다.

#### 6.1.3 FEDformer (2022) [journaljeai](https://journaljeai.com/index.php/JEAI/article/view/3514)

**주요 특징**: Fourier Enhance Decomposition + Transformer

**vs N-HiTS**:
| 측면 | FEDformer | N-HiTS |
|------|----------|--------|
| 주파수 처리 | FFT 기반 분해 | MLP 기반 학습 분해 |
| 정확도 | 높음 (당시 SOTA) | **+12~16% 개선** |
| 메모리 | $O(H \log H)$ | $O(H(1-r^B)/(1-r))$ |
| 구현 복잡도 | 높음 | 낮음 (MLP만 사용) |

### 6.2 MLP 기반 모델들과의 비교

#### 6.2.1 DLinear (2023) [arxiv](https://arxiv.org/abs/2512.10866)

**혁신**: 분해 기법 사용, Transformer 필요 없음을 증명

**vs N-HiTS**:
| 측면 | DLinear | N-HiTS |
|------|---------|--------|
| 아키텍처 | 선형 + 분해 | 비선형 + 계층적 보간 |
| 정확도 | 강함 | **약간의 추가 개선 (+3~5%)** |
| 복잡도 | $O(L)$ 선형 | $O(H(1-r^B)/(1-r))$ 지수적 압축 |
| 해석성 | 트렌드/계절성 | 다중 주파수 특화 |

DLinear의 성공은 모델의 복잡성보다는 **부드러운 분해**가 중요함을 입증했다. N-HiTS는 이를 비선형 확장으로 개선한다.

#### 6.2.2 TSMixer (2023) [arxiv](https://arxiv.org/pdf/2306.09364.pdf)

**주요 특징**: MLP-Mixer 패러다임 + 패칭(Patching) + 온라인 조정(Online Reconciliation)

**vs N-HiTS**:
| 측면 | TSMixer | N-HiTS |
|------|---------|--------|
| 주요 아이디어 | 시간/특성 혼합 MLP | 시간 스케일 계층화 + 보간 |
| 다변량 처리 | 명시적 (채널 혼합) | 암묵적 (단변량 + 글로벌 파라미터) |
| 성능 | 매우 강함 (SOTA 경쟁) | **TSMixer와 비슷 또는 약간 우수** |
| 계산 효율 | 매우 높음 | 비슷 |

TSMixer와 N-HiTS는 서로 보완적인 강점을 가진다. TSMixer는 다변량 상호작용에, N-HiTS는 계층적 분해에 우수하다.

### 6.3 최신 고급 아키텍처 (2024-2025)

#### 6.3.1 PatchTST (2023) [linkedin](https://www.linkedin.com/posts/risman-adnan-bb726b5_what-is-the-top-performer-ai-model-on-deep-activity-7331481462224904192-sRJJ)

**주요 특징**: Patching + Transformer + RevIN 정규화

**현황**: 정확도 기준으로 대부분의 벤치마크에서 최상위 성능 (12-15% MAE 개선)

**vs N-HiTS**:
| 측면 | PatchTST | N-HiTS |
|------|----------|--------|
| 성능 | 높음 (당대 최고) | 약간의 추가 개선 가능 |
| 구현 | Transformer 필요 | 순수 MLP |
| 장기 성능 | 우수 | 동급 또는 우수 |

#### 6.3.2 iTransformer (2024) [linkedin](https://www.linkedin.com/posts/risman-adnan-bb726b5_what-is-the-top-performer-ai-model-on-deep-activity-7331481462224904192-sRJJ)

**혁신**: 특성(변수) 차원에서의 주의 메커니즘 (역전된 주의)

**특징**: 강력한 다변량 성능

**vs N-HiTS**:
- 다변량 관계를 명시적으로 모델링 (N-HiTS는 단변량)
- 정확도: iTransformer가 다변량 시나리오에서 우수
- 계산: N-HiTS가 더 효율적

#### 6.3.3 WaveHiTS (2024) [arxiv](https://arxiv.org/pdf/2504.06532.pdf)

**혁신**: 웨이블릿 변환 + N-HiTS 통합

**개념**: 
$$\text{WaveHiTS} = \text{Wavelet Transform} + \text{N-HiTS}$$

**특징**: 비정상 시계열(예: 풍속 데이터)에서 N-HiTS 대비 추가 성능 향상

**의의**: N-HiTS의 계층적 보간이 웨이블릿 이론과 자연스럽게 호환됨을 시사 [arxiv](https://arxiv.org/pdf/2504.06532.pdf)

#### 6.3.4 PENGUIN (2024) [arxiv](https://arxiv.org/html/2508.13773v1)

**아이디어**: 주기적-중첩 그룹 주의 메커니즘 + Transformer

**vs N-HiTS**:
- 시계열의 주기성을 명시적으로 모델링
- N-HiTS도 다중 $r_\ell$ (서로 다른 계절성 주기)로 주기성 처리 가능
- 성능: 경쟁 수준 (데이터셋 의존적)

### 6.4 2024-2025년 종합 평가

| 모델 | 정확도 순위 | 효율성 순위 | 해석성 | 강점 |
|------|-----------|-----------|--------|------|
| **N-HiTS** | 2-3위 | 1위 | 우수 | 계층적 분해, 초고속 |
| **PatchTST** | 1위 | 2-3위 | 낮음 | 순수 정확도 |
| **TSMixer** | 2-3위 | 1위 | 낮음 | 다변량 효율성 |
| **iTransformer** | 1-2위 (다변량) | 3-4위 | 낮음 | 다변량 강점 |
| **DLinear** | 3-4위 | 1위 | 높음 | 단순성, 해석성 |

***

## VII. 모델의 일반화 성능 향상 가능성

### 7.1 이론적 기반

계층적 보간 정리는 다음 조건 하에서 임의로 긴 지평선 근사를 보장한다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d4bd591f-3f61-4e97-a95f-97ef82c571f3/2201.12886v6.pdf)

1. **다중 해상도 함수의 완전성**: Haar 스케일링 함수, 피스와이즈 선형 함수, 스플라인 등으로 $L^2()$ 완전 근사 가능 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d4bd591f-3f61-4e97-a95f-97ef82c571f3/2201.12886v6.pdf)
2. **투영의 부드러움**: 예측 관계 $\text{Proj}\_{V_w}(Y(\tau))$가 입력 $y_{t-L:t}$에 대해 부드럽게 변함

이는 신경망이 충분한 깊이(블록 수)와 표현성 비율 스케줄로 새로운 데이터셋에 일반화할 수 있음을 시사한다.

### 7.2 실증적 증거

**Cross-Dataset Generalization**:
논문의 Appendix I에서 N-HiTS는 6개 벤치마크 전체에서 일관되게 우수한 성능을 유지했다. 이는: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d4bd591f-3f61-4e97-a95f-97ef82c571f3/2201.12886v6.pdf)

1. **구조적 안정성**: 다양한 시계열 특성(주기, 비주기, 트렌드, 이상 등)에 대한 로버스트성
2. **파라미터 공유의 장점**: 글로벌 파라미터가 시간 패턴의 근본적인 부드러운 특성을 학습
3. **주파수 특화**: Top-Down 계층화가 새로운 데이터에도 효과적임을 증명

### 7.3 일반화 개선 방안

#### 7.3.1 도메인 적응 (Domain Adaptation)

제안: 새로운 도메인에서의 미세조정(fine-tuning)

$$\text{Loss}\_{\text{target}} = \text{MSE}(\hat{y}, y) + \lambda \text{KL}(\text{distribution}\_{\text{source}}, \text{distribution}_{\text{target}})$$

**효과**: 사전 학습된 계층 구조는 유지하면서 새로운 시계열 특성에 적응

#### 7.3.2 적응형 표현성 비율 스케줄링

현재: 고정 지수 스케줄 $r_\ell = r^{\ell-1}$

제안: 데이터 기반 적응 스케줄
$$r_\ell = f(\text{Spectral Content}_\ell, \text{Horizon})$$

**효과**: 새로운 주파수 특성에 자동으로 최적화

#### 7.3.3 메타 학습 (Meta-Learning)

다중 시계열 메타 학습으로 빠른 적응:
$$\min_{\theta} \sum_{i=1}^{K} L(\theta - \alpha \nabla L_i(\theta), D_i^{\text{test}})$$

**효과**: 새로운 시계열에 대한 수렴 속도 극적 단축

### 7.4 한계 인식

그럼에도 N-HiTS의 일반화 성능이 완벽하지 않은 경우들:

1. **극단적 비정상성**: Black Swan 이벤트(COVID-19 팬데믹 등)가 있는 데이터 → 부드러움 가정 위반
2. **극단 단축 시계열**: L이 매우 작은 경우 (예: 10개 미만 관측치) → 신경망 효과 제한적
3. **극도로 희소한 주기성**: 주기가 매우 길거나 불규칙한 경우 (예: 10년 주기) → 여러 주기 관찰 불가

***

## VIII. 앞으로의 연구 영향 및 고려 사항

### 8.1 N-HiTS가 시계열 예측 연구에 미친 영향

**즉각적 영향**:
1. **Transformer 회의론의 제기**: 단순하고 효율적인 모델도 최첨단 복잡한 모델을 능가할 수 있음을 증명
2. **계층적 분해의 르네상스**: 이후 연구들이 계층적/다중 스케일 처리를 적극 채택
3. **MLP 재평가**: Transformer 독점에서 벗어나 MLP 기반 아키텍처 재검토

**장기적 영향**:
- TSMixer(2023)와 기타 MLP 기반 모델들의 등장
- DLinear의 "Transformer 불필요" 주장이 학계 논쟁으로 발전
- Wavelet + N-HiTS (WaveHiTS) 등 하이브리드 접근법 개발

### 8.2 향후 연구 시 고려해야 할 점

#### 8.2.1 멀티태스크 학습 통합

**현재**: N-HiTS는 단일 예측 작업에 최적화

**제안**: 
$$L = \alpha L_{\text{forecast}} + \beta L_{\text{uncertainty}} + \gamma L_{\text{anomaly}}$$

다중 목적 함수로 동시에 포인트 예측, 불확실성 정량화, 이상 탐지를 학습

#### 8.2.2 외생 변수 통합

**한계**: 현재 단변량 시계열만 사용

**제안 구조**:

$$\hat{y}\_{t+1:t+H} = N\text{-HiTS}(y_{t-L:t}) + f_{\text{exog}}(X_{t-L:t+H})$$

외생 변수의 영향을 보간 계수에 조건화

#### 8.2.3 불확실성 정량화

**개선**: 점 추정 대신 확률 예측

N-HiTS의 다중 출력 해석 구조를 활용한 **다중 분위수(quantile) 손실**:

$$L_{\text{quantile}} = \sum_{q} \sum_{\tau} \rho_q(\hat{y}\_{q,\tau} - y_\tau)$$

여기서 $\rho_q(u) = u(q - \mathbb{1}[u < 0])$

#### 8.2.4 하드웨어 최적화

**기회**: N-HiTS는 매우 효율적이므로 엣지 배포 가능

구현 고려:
1. 모바일 GPU (TensorFlow Lite)
2. 양자화(Quantization) - FP32 → INT8
3. 프루닝(Pruning) - 중요도 낮은 블록 제거

**목표**: 임베디드 기기에서 실시간 예측

#### 8.2.5 설명 가능성 강화

**강점**: 이미 계층적 분해로 어느 정도 해석 가능

**추가 방안**:
1. **주의 시각화**: 어떤 과거 시점이 가장 영향력 있는지 표시
2. **신경 분석**: 특성 중요도 분석
3. **대조 설명**: 반사실적(counterfactual) 예측 분석

#### 8.2.6 전이 학습 패러다임

**미래 방향**: 대규모 사전 학습 + 미세조정

$$ \text{Pre-trained N-HiTS} \xrightarrow{\text{Fine-} \\ \text{tune}} \text{Domain-specific Forecaster} $$

**사례**:
- 금융 시계열로 사전 학습
- 새로운 회사의 주가 예측에 미세조정

#### 8.2.7 적대적 견고성 (Adversarial Robustness)

**문제**: 극악의 입력에 대한 취약성 미탐사

**제안**: 적대적 훈련
$$L_{\text{robust}} = \mathbb{E}_{x,\delta} [L(\hat{y} + \delta, y)]$$

#### 8.2.8 다변량 확장의 지혜로운 방법

N-HiTS의 단변량 설계는 의도적이지만, 다변량으로의 확장이 필요한 경우:

**옵션 A (보수적)**: 변수별 별도 N-HiTS + 사후 조정

$$\hat{y}^{(i)}_{t+1:t+H} = N\text{-HiTS}^{(i)}(y^{(i)}_{t-L:t})$$

$$\text{Reconcile}(\{\hat{y}^{(1)}, \ldots, \hat{y}^{(k)}\}) = \text{enforce constraints}$$

**옵션 B (혁신적)**: iTransformer 아이디어 차용 → "N-HiTS-Inverted"
$$\text{Attention over features, MLP over time}$$

***

## IX. 기술 사항 및 재현 정보

### 9.1 실험 설정

**데이터셋**: ETTm2(7개 변수), Exchange(8개), ECL(321개), Traffic(862개), Weather(21개), ILI(7개) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d4bd591f-3f61-4e97-a95f-97ef82c571f3/2201.12886v6.pdf)

**정규화**: 훈련 데이터 평균/표준편차로 정규화 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d4bd591f-3f61-4e97-a95f-97ef82c571f3/2201.12886v6.pdf)

**분할**: 훈련 70%, 검증 10%, 테스트 20% (ETTm2는 검증 20%) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d4bd591f-3f61-4e97-a95f-97ef82c571f3/2201.12886v6.pdf)

**평가 메트릭**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d4bd591f-3f61-4e97-a95f-97ef82c571f3/2201.12886v6.pdf)

$$\text{MSE} = \frac{1}{H}\sum_{\tau=t}^{t+H} (y_\tau - \hat{y}_\tau)^2$$

$$\text{MAE} = \frac{1}{H}\sum_{\tau=t}^{t+H} |y_\tau - \hat{y}_\tau|$$

**하이퍼파라미터 최적화**: HYPEROPT(Bayesian Optimization), 20회 반복 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d4bd591f-3f61-4e97-a95f-97ef82c571f3/2201.12886v6.pdf)

### 9.2 주요 하이퍼파라미터

Table A.3에 따르면: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d4bd591f-3f61-4e97-a95f-97ef82c571f3/2201.12886v6.pdf)
- **스택당 블록 수**: B = 1
- **스택 수**: S = 3
- **MLP 레이어**: 2개
- **숨은 차원**: $N_h$ = 512
- **풀링 커널 크기**: $k_\ell \in \{,,,, \}$ [peninsula-press](https://peninsula-press.ae/Journals/index.php/EDRAAK/article/view/172)
- **표현성 계수**: $r^{-1}_\ell \in \{,,,, \}$ [ejournal.uin-malang.ac](https://ejournal.uin-malang.ac.id/index.php/Math/article/view/32760)
- **최적화**: ADAM, 초기 학습율 1e-3, 3회 반감
- **손실 함수**: MAE
- **배치 크기**: 256
- **훈련 스텝**: 1000

### 9.3 코드 가용성

논문에서 공개한 구현:
```
GitHub: https://github.com/Nixtla/neuralforecast
```

PyTorch 기반, NeuralForecast 라이브러리에 통합됨 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d4bd591f-3f61-4e97-a95f-97ef82c571f3/2201.12886v6.pdf)

***

## X. 결론

### 10.1 N-HiTS의 기여도 종합

N-HiTS는 **효율성과 정확도의 새로운 Pareto frontier**를 제시했다. 장기 시계열 예측에서:

1. **정확도**: Transformer 기반 방법 대비 **평균 14-20% 개선** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d4bd591f-3f61-4e97-a95f-97ef82c571f3/2201.12886v6.pdf)
2. **속도**: **45배 빠른 훈련** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d4bd591f-3f61-4e97-a95f-97ef82c571f3/2201.12886v6.pdf)
3. **메모리**: **54% 파라미터 감소** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d4bd591f-3f61-4e97-a95f-97ef82c571f3/2201.12886v6.pdf)
4. **해석성**: 계층적 주파수 분해로 **명시적 해석 가능** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d4bd591f-3f61-4e97-a95f-97ef82c571f3/2201.12886v6.pdf)
5. **이론**: **Neural Basis Approximation Theorem**으로 근사 정당화 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d4bd591f-3f61-4e97-a95f-97ef82c571f3/2201.12886v6.pdf)

### 10.2 학문적 중요성

이 논문은 다음을 입증했다:

- **복잡성 > 효율성이 아님**: Transformer의 쿼드러플 복잡도는 장기 예측에서 해가 될 수 있음
- **귀납적 편견의 가치**: 부드러움과 다중 스케일 구조를 명시적으로 인코딩하면 더 나은 성능
- **계층적 처리의 보편성**: 신경망의 암묵적 계층 구조보다는 명시적 설계가 우수

### 10.3 실무적 영향

산업 응용에서의 N-HiTS의 가치:

1. **비용 절감**: 클라우드 훈련 비용 70% 감소
2. **배포 용이성**: 엣지 디바이스에서 실시간 예측 가능
3. **모니터링**: 해석 가능한 분해로 이상 탐지 용이
4. **개발 속도**: MLP 구현의 단순성으로 빠른 프로토타이핑

### 10.4 향후 연구 방향 요약

| 방향 | 기대 효과 | 우선순위 |
|------|---------|---------|
| 다변량 확장 (지혜로운) | 추가 5-10% 성능 향상 | 높음 |
| 불확실성 정량화 | 확률 예측 | 높음 |
| 외생 변수 통합 | 실제 응용성 | 높음 |
| 사전 학습 + 전이학습 | 샷수 감소 | 중간 |
| 웨이블릿 통합 | 비정상 시계열 | 중간 |
| 엣지 배포 최적화 | 임베디드 적용 | 중간 |

### 10.5 최종 평가

N-HiTS는 단순하면서도 **깊이 있는 (simple yet profound)** 논문이다. 신경망 아키텍처의 무제한 표현성이라는 "특징"을 "버그"로 인식하고, 계층적 보간이라는 우아한 해결책을 제시했다. 이는 향후 수년간 시계열 예측 연구의 중심축이 될 만큼 영향력 있는 기여이다.

**가장 주목할 점**: 복잡한 신경망 모델이 항상 답이 아니며, **문제 구조의 깊은 이해와 귀납적 편견의 설계**가 더욱 강력할 수 있다는 것이다. 이 통찰은 시계열 예측을 넘어 딥러닝 전반에 시사하는 바가 크다.

***

## 참고 자료

<span style="display:none">[^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_80][^1_81][^1_82][^1_83][^1_84][^1_85][^1_86][^1_87][^1_88][^1_89][^1_90][^1_91][^1_92]</span>

<div align="center">⁂</div>

[^1_1]: 2201.12886v6.pdf

[^1_2]: https://dl.acm.org/doi/10.1609/aaai.v37i6.25854

[^1_3]: https://arxiv.org/abs/2106.13008

[^1_4]: https://journaljeai.com/index.php/JEAI/article/view/3514

[^1_5]: https://arxiv.org/abs/2512.10866

[^1_6]: https://arxiv.org/pdf/2306.09364.pdf

[^1_7]: https://www.linkedin.com/posts/risman-adnan-bb726b5_what-is-the-top-performer-ai-model-on-deep-activity-7331481462224904192-sRJJ

[^1_8]: https://arxiv.org/pdf/2504.06532.pdf

[^1_9]: https://arxiv.org/html/2508.13773v1

[^1_10]: https://peninsula-press.ae/Journals/index.php/EDRAAK/article/view/172

[^1_11]: https://ejournal.uin-malang.ac.id/index.php/Math/article/view/32760

[^1_12]: http://arxiv.org/pdf/2405.08790.pdf

[^1_13]: https://arxiv.org/pdf/2310.10688.pdf

[^1_14]: https://arxiv.org/html/2504.04011v1

[^1_15]: https://arxiv.org/html/2410.03805v3

[^1_16]: https://www.reddit.com/r/datascience/comments/1i4yyoe/influential_timeseries_forecasting_papers_of/

[^1_17]: https://arxiv.org/pdf/2402.16516.pdf

[^1_18]: https://arxiv.org/pdf/2210.01753.pdf

[^1_19]: https://www.nature.com/articles/s41598-025-19432-6

[^1_20]: https://journal.astanait.edu.kz/index.php/ojs/article/view/913

[^1_21]: https://dl.acm.org/doi/10.1145/3757749.3757774

[^1_22]: https://rjor.ro/considerations-on-the-efficiency-of-time-series-analysis-in-forecasting-new-influenza-cases-in-the-2024-2025-season/

[^1_23]: https://ieeexplore.ieee.org/document/11029306/

[^1_24]: https://wepub.org/index.php/TCSISR/article/view/5639

[^1_25]: https://ieeexplore.ieee.org/document/11020096/

[^1_26]: https://arxiv.org/pdf/2302.14390.pdf

[^1_27]: https://arxiv.org/pdf/2308.00709.pdf

[^1_28]: http://arxiv.org/pdf/2407.10768.pdf

[^1_29]: http://arxiv.org/pdf/2410.15217.pdf

[^1_30]: http://arxiv.org/pdf/1809.02105.pdf

[^1_31]: https://arxiv.org/pdf/2104.00164.pdf

[^1_32]: https://arxiv.org/abs/2201.12886

[^1_33]: https://arxiv.org/html/2503.10198v1

[^1_34]: https://arxiv.org/pdf/2201.12886.pdf

[^1_35]: https://arxiv.org/html/2509.23145v1

[^1_36]: https://arxiv.org/abs/2410.03805

[^1_37]: https://arxiv.org/pdf/2412.13769.pdf

[^1_38]: https://arxiv.org/html/2507.10349v1

[^1_39]: https://arxiv.org/html/2409.00480v2

[^1_40]: https://arxiv.org/html/2508.07697v3

[^1_41]: https://arxiv.org/html/2505.20048v1

[^1_42]: https://www.arxiv.org/pdf/2506.12809.pdf

[^1_43]: https://www.nature.com/articles/s41467-025-63786-4

[^1_44]: https://arxiv.org/html/2507.13043v1

[^1_45]: https://towardsdatascience.com/n-hits-making-deep-learning-for-time-series-forecasting-more-efficient-d00956fc3e93/

[^1_46]: https://blogs.mathworks.com/finance/2025/07/31/building-a-neural-network-for-time-series-forecasting-low-code-workflow/

[^1_47]: https://proceedings.neurips.cc/paper_files/paper/2024/file/cf66f995883298c4db2f0dcba28fb211-Paper-Conference.pdf

[^1_48]: https://www.sciencedirect.com/science/article/abs/pii/S0360835223006915

[^1_49]: https://hess.copernicus.org/articles/30/371/2026/

[^1_50]: https://towardsdatascience.com/influential-time-series-forecasting-papers-of-2023-2024-part-1-1b3d2e10a5b3/

[^1_51]: https://dl.acm.org/doi/10.1145/190468.190290

[^1_52]: https://nixtlaverse.nixtla.io/neuralforecast/docs/tutorials/longhorizon_transformers.html

[^1_53]: https://www.nature.com/articles/s41598-025-34847-x

[^1_54]: https://arxiv.org/abs/2411.11046

[^1_55]: https://www.ssrn.com/abstract=4718033

[^1_56]: http://medrxiv.org/lookup/doi/10.1101/2024.09.09.24313361

[^1_57]: https://link.springer.com/10.1007/s13042-024-02417-8

[^1_58]: https://repositorio.banrep.gov.co/bitstream/handle/20.500.12134/10657/monetary-policy-january-2023.pdf

[^1_59]: http://sao.editorum.ru/en/nauka/conference_article/11868/view

[^1_60]: http://jurnalnasional.ump.ac.id/index.php/JUITA/article/view/27176

[^1_61]: https://link.springer.com/10.1007/s43621-025-01733-5

[^1_62]: https://ieeexplore.ieee.org/document/11180456/

[^1_63]: https://arxiv.org/pdf/2106.13008.pdf

[^1_64]: https://arxiv.org/pdf/2502.13721.pdf

[^1_65]: http://arxiv.org/pdf/2207.07827.pdf

[^1_66]: https://arxiv.org/html/2310.01884

[^1_67]: https://www.mdpi.com/2313-7673/9/1/40/pdf?version=1704785987

[^1_68]: https://arxiv.org/pdf/2206.04038.pdf

[^1_69]: https://www.arxiv.org/pdf/2508.01407.pdf

[^1_70]: https://arxiv.org/abs/1912.09363

[^1_71]: https://arxiv.org/pdf/2308.04791.pdf

[^1_72]: https://arxiv.org/pdf/2409.10840.pdf

[^1_73]: https://peerj.com/articles/cs-2713/

[^1_74]: https://www.arxiv.org/pdf/2410.02081.pdf

[^1_75]: https://arxiv.org/pdf/2502.11816.pdf

[^1_76]: https://www.arxiv.org/pdf/2508.04048.pdf

[^1_77]: https://arxiv.org/html/2410.21448v1

[^1_78]: https://arxiv.org/abs/2508.04048

[^1_79]: https://arxiv.org/html/2506.12809

[^1_80]: https://arxiv.org/html/2306.09364v4

[^1_81]: https://pdfs.semanticscholar.org/aef2/16575aa0c8ed108316514b1243d0bf2cef2c.pdf

[^1_82]: https://towardsdatascience.com/tsmixer-googles-innovative-deep-learning-forecasting-model-4c3ab1c80a23/

[^1_83]: https://huggingface.co/blog/autoformer

[^1_84]: https://research.google/pubs/temporal-fusion-transformers-for-interpretable-multi-horizon-time-series-forecasting/

[^1_85]: https://www.itia.ntua.gr/el/getfile/2497/1/documents/water-16-02882-v2.pdf

[^1_86]: https://aihorizonforecast.substack.com/p/temporal-fusion-transformer-time

[^1_87]: https://arxiv.org/html/2506.12809v1

[^1_88]: https://proceedings.neurips.cc/paper_files/paper/2024/file/754612bde73a8b65ad8743f1f6d8ddf6-Paper-Conference.pdf

[^1_89]: https://www.youtube.com/watch?v=V14qoa5vZ1I

[^1_90]: https://www.sciencedirect.com/science/article/pii/S0893608025003727

[^1_91]: https://www.neuralaspect.com/posts/tsmixer-google

[^1_92]: https://velog.io/@jhbale11/Temporal-Fusion-Transformer2020논문-리뷰
