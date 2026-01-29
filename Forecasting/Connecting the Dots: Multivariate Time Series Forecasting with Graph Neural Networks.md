
# Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks

## 1. 논문의 핵심 주장 및 기여

### 1.1 핵심 주장

본 논문의 중심 명제는 **다변수 시계열 데이터의 변수 간 잠재 공간 종속성(latent spatial dependencies)이 기존 방법들에 의해 제대로 활용되지 못하고 있다**는 것이다. 논문 저자들은 기존의 통계적 방법(VAR, GP)과 심층학습 방법(LSTNet, TPA-LSTM)이 변수 쌍 간의 명시적인 종속성을 모델링하지 못한다고 비판한다. 이러한 한계를 극복하기 위해, 저자들은 그래프 신경망(GNN)이 관계 종속성 처리에서 뛰어난 능력을 보이지만 사전정의된 그래프 구조를 필요로 한다는 점에 주목하고, **다변수 시계열에서 그래프 구조를 자동으로 학습하는 엔드-투-엔드 프레임워크**를 제안한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/60e37a15-43ed-4dfd-8d34-40fe6f0c1aa9/2005.11650v1.pdf)

### 1.2 주요 학술적 기여

논문의 공식적 기여는 네 가지로 정리된다:

1. **학문적 개척성**: 다변수 시계열 데이터를 그래프 기반 관점에서 포괄적으로 다룬 첫 연구 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/60e37a15-43ed-4dfd-8d34-40fe6f0c1aa9/2005.11650v1.pdf)
2. **그래프 학습 모듈**: 단방향 관계를 추출하는 새로운 그래프 학습 메커니즘 제안
3. **혼합 홉 전파 계층(Mix-hop Propagation Layer)**: 그래프 합성곱 네트워크의 과도한 평활화(over-smoothing) 문제 해결
4. **일반적 프레임워크**: 사전정의 그래프 구조가 있는 경우와 없는 경우 모두를 처리할 수 있는 유연한 아키텍처

## 2. 해결하고자 하는 문제 및 제안 방법

### 2.1 핵심 문제 정의

논문은 두 가지 주요 도전 과제를 식별한다:

**도전 과제 1**: 미지의 그래프 구조 - 기존 GNN 방법들은 시계열 예측을 위해 사전정의된 그래프 구조에 의존하지만, 대부분의 다변수 시계열에서는 명시적인 그래프 구조가 존재하지 않는다. 따라서 데이터로부터 변수 간의 숨겨진 관계를 발견해야 한다.

**도전 과제 2**: 그래프 학습과 GNN 학습의 동시성 - 그래프 구조를 학습하는 것과 GNN 모델을 학습하는 것을 별개로 처리하는 기존 접근 방식은 최적이 아니며, 엔드-투-엔드 방식으로 동시에 최적화되어야 한다.

### 2.2 제안 방법론

#### 2.2.1 그래프 학습 계층

논문의 핵심 혁신은 적응적 그래프 인접 행렬을 학습하는 메커니즘이다. 그 수식은 다음과 같다:

$$M_1 = \tanh(\alpha E_1 \Theta_1) \quad \cdots (1)$$

$$M_2 = \tanh(\alpha E_2 \Theta_2) \quad \cdots (2)$$

$$A = \text{ReLU}(\tanh(\alpha(M_1 M_2^T - M_2 M_1^T))) \quad \cdots (3)$$

여기서 $E_1, E_2$는 학습 가능한 노드 임베딩, $\Theta_1, \Theta_2$는 모델 파라미터, $\alpha$는 활성화 함수의 포화율을 제어하는 하이퍼파라미터이다.

**단방향성 달성**: 식 (3)의 $M_1 M_2^T - M_2 M_1^T$ 차이와 ReLU 활성화는 비대칭 성질을 구현한다. 이는 $A_{vu}$가 양수이면 대각 원소 $A_{uv}$를 0으로 설정하여 방향성을 보장한다.

이후 희소성을 위해 각 노드마다 상위-k 이웃을 선택한다:

$$\text{idx} = \text{argtopk}(A[i, :]) \quad \cdots (5)$$

$$A[i, -\text{idx}] = 0 \quad \cdots (6)$$

**외부 정보 통합**: 변수의 정적 특성이 주어진 경우, $E_1 = E_2 = Z$ (정적 특성 행렬)로 설정하여 외부 지식을 용이하게 통합할 수 있다.

#### 2.2.2 Mix-hop 전파 계층

그래프 합성곱의 과도한 평활화 문제를 해결하기 위해 제안된 혼합 홉 전파는 두 단계로 구성된다:

**정보 전파 단계**:
$$H^{(k)} = \beta H_{\text{in}} + (1-\beta)\tilde{A}H^{(k-1)} \quad \cdots (7)$$

여기서:
- $\beta$: 원래 노드 상태를 유지하는 비율
- $\tilde{A} = \tilde{D}^{-1}(A + I)$: 정규화 인접 행렬
- $\tilde{D}\_{ii} = 1 + \sum_j A_{ij}$: 차수 행렬

**정보 선택 단계**:
$$H_{\text{out}} = \sum_{k=0}^{K} H^{(k)} W^{(k)} \quad \cdots (8)$$

이 구조의 이점은 $K=2, W^{(0)}=0, W^{(1)}=-1, W^{(2)}=1$일 때 인접한 두 홉 간의 델타 차이를 표현할 수 있다는 것이다:

$$H_{\text{out}} = \Delta(H^{(2)}, H^{(1)}) = H^{(2)} - H^{(1)} \quad \cdots (9)$$

#### 2.2.3 확장 인셉션 계층

시간적 패턴을 포착하기 위해 두 가지 전략을 결합한 확장 인셉션 계층이 제안된다:

**다중 커널 크기 인셉션**: 시계열 데이터의 내재 주기성(7, 12, 24, 28, 60 등)을 고려하여 1×2, 1×3, 1×6, 1×7 필터 크기를 사용한다:

$$z = \text{concat}(z \star f^{1×2}, z \star f^{1×3}, z \star f^{1×6}, z \star f^{1×7}) \quad \cdots (12)$$

**확장된 합성곱**: 긴 시간 의존성을 처리하기 위해 확장 인수(dilation factor) $d$를 사용한다:

$$z \star f^{1×k}(t) = \sum_{s=0}^{k-1} f^{1×k}(s) z(t - d \times s) \quad \cdots (13)$$

수용 영역(receptive field)은 지수적으로 증가한다:

$$R = 1 + (c-1)\frac{q^m - 1}{q - 1} \quad \cdots (11)$$

여기서 $q > 1$은 확장 지수 인수이다.

### 2.3 모델 구조

MTGNN의 전체 아키텍처는 다음 요소들로 구성된다:

 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/60e37a15-43ed-4dfd-8d34-40fe6f0c1aa9/2005.11650v1.pdf)

1. **1×1 초기 합성곱**: 입력을 잠재 공간으로 투영 ($\mathbb{R}^{N \times T \times D} \to \mathbb{R}^{N \times T \times d}$)
2. **그래프 학습 계층**: 모든 그래프 합성곱 모듈에서 사용될 적응적 인접 행렬 계산
3. **교대 모듈**: m개의 그래프 합성곱과 시간 합성곱 모듈을 교대로 배치
4. **잔차 및 스킵 연결**: 그래프 합성곱 입출력 간 잔차, 시간 합성곱 이후 스킵 연결
5. **출력 모듈**: 1×1 합성곱을 통해 원하는 차원으로 변환

### 2.4 학습 알고리즘

#### 2.4.1 교과 학습 전략

다중 단계 예측에서 장기 예측이 단기 예측보다 훨씬 큰 손실을 생성하므로, 모델이 장기 예측에만 과도하게 집중하는 문제를 해결하기 위해 교과 학습을 적용한다:

1단계 예측으로 시작 → 점진적으로 Q단계로 확장

이를 통해 모델은 먼저 쉬운 과제에서 좋은 출발점을 찾고, 난이도가 증가함에 따라 단계적으로 학습한다.

#### 2.4.2 부그래프 샘플링

메모리 오버플로우를 방지하기 위해 각 반복에서 노드를 m개 그룹으로 무작위 분할:

$$\text{시간 복잡도: } O(N^2) \to O((N/m)^2)$$

이는 대규모 그래프 처리를 가능하게 하면서도 각 노드가 한 그룹 내에서 다른 모든 노드와의 유사도 점수를 계산하고 업데이트할 기회를 보장한다.

## 3. 성능 향상 및 한계

### 3.1 실험 결과

#### 3.1.1 단일 단계 예측 (Single-step Forecasting)

 [dl.acm](https://dl.acm.org/doi/10.1145/3394486.3403118)

MTGNN은 3개 벤치마크 데이터셋에서 SOTA 달성:
- **Traffic**: RSE에서 3.88~7.24% 개선
- **Solar-Energy**: 모든 예측 지평에서 우수한 성능
- **Electricity**: 일관적인 개선

유일한 예외는 **Exchange-Rate** 데이터셋으로, 이는 8개 노드의 매우 작은 그래프 크기와 제한된 학습 데이터 때문이다.

#### 3.1.2 다중 단계 예측 (Multi-step Forecasting)

 [semanticscholar](https://www.semanticscholar.org/paper/645054d31fa26b29bbfb0cf73b75f8906c359415)

MTGNN은 사전정의 그래프 없이도 최첨단 GNN 방법들과 경쟁력 있는 성능을 달성한다:
- METR-LA: MAE 2.69 (DCRNN: 2.77, Graph WaveNet: 2.69)
- PEMS-BAY: MAE 1.32 (MRA-BGCN: 1.29)

이는 특히 중요한데, DCRNN, STGCN, MRA-BGCN 등은 모두 사전정의된 도로 네트워크 거리 정보를 활용하지만 MTGNN은 이런 정보 없이 동등한 성능을 달성한다.

### 3.2 절제 연구(Ablation Study) 결과

 [semanticscholar](https://www.semanticscholar.org/paper/2a8beb7836b4e999e452448220573c0133834a55)

METR-LA 데이터셋에서 수행된 절제 연구는 각 구성 요소의 중요성을 정량화한다:

| 제거 항목 | MAE 변화 | RMSE 변화 | 해석 |
|---------|---------|---------|------|
| 그래프 합성곱 (w/o GC) | +3.5% | +5.4% | **가장 중요**: 변수 간 정보 흐름 |
| Mix-hop 정보 선택 (w/o Mix-hop) | +0.9% | +1.0% | 중요: 각 홉에서의 유용 정보 필터링 |
| 인셉션 다중 필터 (w/o Inception) | +0.2% | +0.3% | 중간: 다양한 시간 주기 포착 |
| 교과 학습 (w/o CL) | +0.4% | +0.3% | 중요: 수렴 안정성 및 최적점 향상 |

### 3.3 그래프 학습 방법 비교

논문은 다양한 그래프 구성 방법을 비교한다:

| 방법 | 구성 | MAE | RMSE | MAPE | 특징 |
|-----|-----|-----|------|------|------|
| 사전정의 (Pre-defined) | 도로 거리 | 2.9017 | 6.1288 | 0.0836 | 기준선 |
| 전역 (Global-A) | $A = \text{ReLU}(W)$ | 2.8457 | 5.9900 | 0.0805 | N² 파라미터 |
| 무방향 (Undirected-A) | $A = \text{ReLU}(\tanh(\alpha M_1M_1^T))$ | 2.7736 | 5.8411 | 0.0783 | 대칭 관계 |
| 방향 (Directed-A) | $A = \text{ReLU}(\tanh(\alpha M_1M_2^T))$ | 2.7758 | 5.8217 | 0.0783 | 일방향 관계 |
| 동적 (Dynamic-A) | $A_t = \text{SoftMax}(\tanh(X_t W_1)\tanh(W_2^T X_t^T))$ | 2.8124 | 5.9189 | 0.0794 | 시간 스텝별 변화 |
| **단방향 (Uni-directed-A)** | $A = \text{ReLU}(\tanh(\alpha(M_1M_2^T - M_2M_1^T)))$ | **2.7715** | **5.8070** | **0.0778** | **제안 방법** |

MTGNN의 단방향 그래프 학습이 모든 지표에서 최고 성능을 달성한다.

### 3.4 사례 연구: 학습된 그래프의 해석성

특히 흥미로운 점은 학습된 그래프가 사전정의 그래프보다 더 나은 예측 성능을 제공한다는 것이다. METR-LA 데이터셋의 노드 55를 중심으로 한 사례 연구에서: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/60e37a15-43ed-4dfd-8d34-40fe6f0c1aa9/2005.11650v1.pdf)

- **사전정의 이웃**: 지리적으로 가장 가까운 노드 (동시 상관관계 높음)
- **학습된 이웃**: 더 멀리 떨어져 있지만 같은 도로에 위치한 노드 (선행 상관관계 - 미래 교통 상황 예측에 더 유용)

이는 MTGNN이 **인과적 관계(causal relationships)**를 자동으로 발견할 수 있음을 시사한다.

### 3.5 모델의 한계

1. **Exchange-Rate 데이터셋 성능 저조**: 매우 작은 그래프(8노드)와 제한된 샘플(7,588개)에서는 그래프 학습의 이점이 제한적
2. **정적 노드 특성**: 동적 시간 스텝 간 종속성을 명시적으로 모델링하지 못함
3. **안정성 문제**: 동적 그래프 학습은 수렴이 어려워 교과 학습과 같은 보조 전략 필요
4. **계산 복잡도**: 부그래프 샘플링에도 불구하고 그래프 학습 계층의 $O(N^2)$ 복잡도는 매우 큰 그래프에서 병목

## 4. 모델의 일반화 성능 향상 가능성

### 4.1 일반화의 강점

#### 4.1.1 구조 독립성
MTGNN은 사전정의 그래프의 유무와 관계없이 작동하는 진정한 범용 프레임워크이다. 이는 다양한 응용 분야에 대한 적응성을 의미한다:
- 도로 네트워크 정보 없는 교통 예측
- 센서 간 물리적 근접성이 불분명한 환경 데이터
- 금융 시계열의 복잡한 상호작용 패턴

#### 4.1.2 외부 지식 통합의 용이성
변수 속성, 카테고리, 도메인 지식 등을 노드 임베딩으로 쉽게 통합할 수 있다. 이는 다른 GNN 방법들보다 더 유연한 설계를 가능하게 한다.

#### 4.1.3 단방향 관계 모델링
인과 관계를 반영하는 단방향 그래프는 더 나은 해석성과 물리적 의미를 제공한다.

### 4.2 일반화의 한계 및 개선 방안

#### 4.2.1 현재 한계
1. **고정 그래프 구조**: 학습 후 그래프는 고정되므로 테스트 시점의 새로운 종속성을 포착하지 못함
2. **정적 임베딩**: 노드 임베딩 $E_1, E_2$는 시간 불변이므로 동적 관계 변화 미포착
3. **분포 이동(Distribution Shift)**: 학습 데이터와 다른 특성의 테스트 데이터에서 성능 저하

#### 4.2.2 향상 방안

**1) 동적 그래프 학습**
최신 연구(TimeGNN 2023)는 시간 스텝별로 그래프를 재학습하는 방식을 제안한다:
$$A_t = \text{Function}(E_1(t), E_2(t), \Theta_1(t), \Theta_2(t))$$
이를 통해 시간에 따라 변하는 변수 간 관계를 포착할 수 있다. [arxiv](https://arxiv.org/html/2307.14680)

**2) 다중 스케일 그래프**
여러 수준의 상호작용을 모델링:
- 직접 이웃: 즉각적 영향
- 2홉 이웃: 중간 효과
- 다홉: 장거리 영향

**3) 전이 학습(Transfer Learning)**
대규모 일반 시계열 데이터에서 사전학습된 그래프 구조와 임베딩을 활용하여 새로운 데이터셋에 대한 수렴 속도와 성능 향상:
$$A_{\text{new}} = A_{\text{pretrained}} + \Delta A$$

**4) 주의 메커니즘 기반 동적 가중치**
각 노드와 시간 스텝에서 이웃의 중요도를 동적으로 조정:
$$A_{ij}^{(t)} = \text{Attention}(i, j, t) \cdot A_{ij}$$

**5) 데이터 증강**
학습된 GNN의 특성을 활용하여 부족한 시계열 데이터를 증강하는 GEANN 방식 적용 [arxiv](https://arxiv.org/abs/2307.03595)

**6) 하이브리드 모델**
통계 방법(ARIMA, VAR)의 해석성과 GNN의 표현 능력을 결합:
$$\hat{Y} = \lambda Y_{\text{ARIMA}} + (1-\lambda) Y_{\text{MTGNN}}$$

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 스펙트럼 영역 기반 접근 (StemGNN, 2020)

**기본 아이디어**: 시간 영역이 아닌 주파수 영역에서 관계와 시간 종속성을 동시에 모델링

**핵심 수식**:
$$\text{Graph Fourier Transform (GFT)}: \tilde{X} = U^T X$$
$$\text{Discrete Fourier Transform (DFT)}: \hat{X} = \mathcal{F}(X)$$

**StemGNN vs MTGNN**:

| 측면 | MTGNN | StemGNN |
|------|-------|---------|
| 그래프 학습 | 시간 영역에서 학습 | 스펙트럼 영역에서 학습 |
| 그래프 구조 | 동적 업데이트 | 초기 한 번만 학습 |
| 성능 | 3개 데이터셋 SOTA | 9개 데이터셋에서 평균 8.1% MAE 개선 |
| 해석성 | 높음 (단방향 명시) | 중간 (주파수 분석) |
| 확장성 | 중간 (O(N²)) | 높음 |

StemGNN은 더 광범위한 데이터셋에서 SOTA를 달성했지만, MTGNN의 동적 그래프 학습이 특정 애플리케이션에서 더 적응적일 수 있다. [semanticscholar](https://www.semanticscholar.org/paper/645054d31fa26b29bbfb0cf73b75f8906c359415)

### 5.2 동적 그래프 학습 (TimeGNN, 2023)

**핵심 혁신**: 시간이 진행됨에 따라 노드 간 상호작용 패턴의 진화를 포착

**구조**:
1. 노드 임베딩의 시간적 진화 모델링
2. 시간 의존적 그래프 구성
3. 동적 그래프에서의 정보 전파

**MTGNN과의 차이**:
- MTGNN: 학습 후 고정 그래프
- TimeGNN: 각 시점에서 새로운 그래프 구성
- 성능: TimeGNN이 특히 변수 간 관계가 급격히 변하는 데이터셋에서 우수 [arxiv](https://arxiv.org/abs/2307.14680)

### 5.3 순수 그래프 관점 재검토 (FourierGNN, 2023)

**주요 기여**:
- GNN과 시간 신경망의 분리를 버리고, 순수 그래프 변환으로 통합
- 푸리에 기반 그래프 변환으로 공간-시간 의존성 동시 포착
- 계산 복잡도 감소

**성능**: 여러 벤치마크에서 MTGNN, StemGNN 능가 [arxiv](https://arxiv.org/pdf/2311.06190.pdf)

### 5.4 다중 스케일 관계 학습 (MSGNet, 2024)

**혁신점**:
- 다양한 시간 스케일에서 시계열 간 상관관계 변화 포착
- 주파수 영역 분석으로 주기적 패턴 분해
- 적응형 그래프 합성곱

**MTGNN과의 차이**: MTGNN은 단일 시간 스케일의 관계를 학습하지만, MSGNet은 여러 수준의 관계를 계층적으로 모델링 [arxiv](https://arxiv.org/pdf/2401.00423.pdf)

### 5.5 인과관계 기반 접근 (TEGNN, 2020)

**핵심 개념**: 전이 엔트로피(Transfer Entropy)를 사용하여 변수 간의 **인과 관계**를 명시적으로 모델링

**수식**:
$$TE(X \to Y) = \sum P(y_t, x_t^{(k)}, y_t^{(l)}) \log \frac{P(y_t | x_t^{(k)}, y_t^{(l)})}{P(y_t | y_t^{(l)})}$$

**장점**: 
- 상관성이 아닌 인과성 포착
- 더 물리적으로 의미 있는 그래프 구조
- 외부 변수의 영향 명시적 모델링

**단점**:
- 계산 복잡도 증가
- 인과성 추론의 불확실성 [ieeexplore.ieee](https://ieeexplore.ieee.org/document/9837007/)

### 5.6 확장 가능한 적응형 접근 (SAGDFN, 2024)

**문제 해결**: 기존 STGNN은 수백 개 센서로 제한되지만, 실제 대규모 시스템은 수천 개 이상의 노드 보유

**솔루션**:
- 적응형 그래프 확산 메커니즘
- 계층적 구조 활용
- 부분 그래프 학습

**성능**: 1000+ 노드 데이터셋에서도 선형 시간 복잡도 유지 [arxiv](http://arxiv.org/pdf/2406.12282.pdf)

### 5.7 사전학습 모델 (Pre-training Enhanced STGNN, 2022)

**혁신**:
- 매우 긴 시간 이력(예: 1년)에서 사전학습
- 세그먼트 수준 표현 생성
- 짧은 시간의 STGNN에 맥락 정보 제공

**성능 향상**: 
- 수렴 속도 30% 단축
- 예측 정확도 5-7% 개선
- 콜드 스타트 문제 완화 [dl.acm](https://dl.acm.org/doi/10.1145/3394486.3403118)

### 5.8 다중 과제 학습 (G-MTL, 2023)

**개념**: 여러 시계열의 특성을 동시에 예측 (예: 하나의 역학계에서 여러 변수)

**아키텍처**:
- 자기주의로 전역 동적 종속성 포착
- 그래프주의로 국소 종속성 포착
- 과제 간 특성 공유

**MTGNN과의 차이**: MTGNN은 단일 변수 예측에 최적화되지만, G-MTL은 다중 과제에서 일반화 성능 향상 [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC10453913/)

## 6. 향후 연구에 미치는 영향 및 고려사항

### 6.1 학술적 영향

#### 6.1.1 패러다임 변화
MTGNN은 다변수 시계열 연구에 "그래프 중심" 관점을 도입했다. 이전 주류인 RNN/LSTM 중심 접근에서 구조 학습 중심으로의 전환을 주도했으며, 현재 수십 편의 후속 논문에 영감을 제공했다. [arxiv](https://arxiv.org/html/2410.22377v4)

#### 6.1.2 이론적 기초 부재
MTGNN은 실증적 성공에도 불구하고, **왜 그래프 신경망이 시계열에 효과적인가**에 대한 이론적 설명이 부족하다. 최근 연구들이 이를 보충하려 시도하고 있다:
- 스펙트럼-시간 그래프 합성곱의 표현 성능 [arxiv](https://arxiv.org/html/2305.06587v3)
- 과도한 압착(over-squashing) 현상과의 관계 [arxiv](https://arxiv.org/html/2506.15507v2)

### 6.2 기술적 고려사항

#### 6.2.1 Over-smoothing 문제
Mix-hop 층으로 어느 정도 해결되었지만, 매우 깊은 GNN에서는 여전히 문제: [arxiv](https://arxiv.org/html/2506.15507v2)

정보 손실 방지:

```math
H_{t}^{(l)}=\alpha H_{t}^{(l-1)}+(1-\alpha )\text{Aggregate}\left(\left\{H_{i}^{(l-1)}\right\}_{i\in N(t)}\right)
```

동적 $\alpha_t$를 학습하는 방식도 제안됨.

#### 6.2.2 하이퍼파라미터 민감도
매개변수 연구(Appendix A.4)에서:
- 이웃 개수 k: 작은 값이 더 나음 (단일 변수는 제한된 이웃만 의존)
- 유지 비율 β: 0.05가 최적 (높으면 이웃 정보 무시)
- 전파 깊이 K: 2-3 충분 (더 이상은 over-smoothing)

### 6.3 응용 분야별 고려사항

#### 6.3.1 교통 예측
MTGNN이 가장 큰 성능 향상을 보인 분야:
- 도로 간 인과관계 자동 발견
- 사전정의 네트워크 불필요
- 새로운 도로 추가 시 용이한 재적응

**향후 고려**: 
- 사고, 이벤트 등 외생 변수 통합
- 실시간 온라인 학습
- 다중 도시 간 전이 학습

#### 6.3.2 에너지 시스템
태양광/풍력 발전량, 전력 수요 예측:
- 지리적 분산과 기상 의존성
- 시간 변화하는 수요 패턴
- 그리드 안정성 중요

**MTGNN 적용 시 고려**:
- 계절성 명시적 모델링
- 시간대별 그래프 구조 변화
- 예측 불확실성 정량화

#### 6.3.3 금융 데이터
주식 가격, 암호화폐 등:
- 고빈도 거래로 인한 동적 관계
- 외생 정보(뉴스, 거래량) 중요
- 극단적 사건(flash crash) 처리

**도전**:
- 동적 그래프 학습의 안정성
- 과거 그래프 구조의 적응성 감소
- 설명 가능성 요구

#### 6.3.4 환경/기후
지표수 예측, 오염도 등 다지점 관측:
- 물리적 법칙 제약
- 장시간 의존성
- 결측치 처리

**기회**:
- 물리 정보 신경망(PINN)과 결합
- 부족한 데이터에 대한 일반화
- 인과 관계 발견으로 과학적 통찰 [nature](https://www.nature.com/articles/s41598-024-75385-2)

### 6.4 미래 연구 방향

#### 6.4.1 동적 및 적응형 그래프
```
단계별 개선:
1단계: 시간 윈도우별 정적 그래프 학습
2단계: 연속 시간에서의 그래프 진화
3단계: 사건 기반 그래프 업데이트
```

이러한 진화는 이미 TimeGNN 등에서 시작됨. [arxiv](https://arxiv.org/html/2307.14680)

#### 6.4.2 확률적 예측
현재 MTGNN은 점 추정(point estimate)만 제공:

$$\hat{Y}\_t = \text{Network}(X_{t-P:t})$$

향후:

$$P(\hat{Y}\_t | X_{t-P:t}) = \text{Network}(X_{t-P:t})$$

정량 회귀(quantile regression) 또는 정규화 흐름(normalizing flows) 통합

#### 6.4.3 설명 가능한 AI
모형 해석성 요구 증가:
- 중요 변수 식별
- 의사결정 경로 추적
- 그래프 구조의 물리적 의미 검증

MTGNN의 명시적 그래프는 이 점에서 유리하지만, 심층 네트워크에서는 해석성 저하

#### 6.4.4 극한 사건 예측
대부분의 GNN은 평균 오차에 최적화되어 극값 성능 저하:
- 이례탐지(anomaly detection) 결합
- 불균형 손실함수(class imbalance) 처리
- 강건성(robustness) 향상

#### 6.4.5 다중 모달 데이터
시계열 외 영상, 텍스트, 그래프 정보 통합:
- 멀티모달 GNN
- 이질적 정보 융합
- 교차 모달 전이 학습

#### 6.4.6 기초 모델(Foundation Models) 적용
최근 트렌드인 사전학습 대규모 모델:
- 100만 시계열에서 사전학습
- 새로운 도메인에 미세조정(fine-tuning)
- 제로샷(zero-shot) 일반화

### 6.5 실제 배포 시 고려사항

#### 6.5.1 계산 효율성
```
MTGNN 복잡도:
- 그래프 학습: O(N²s) 
- 그래프 합성곱: O(M) (M = 간선 수)
- 시간 합성곱: O(Nl·c_i·c_o/d)
```

대규모 배포 시:
- GPU 메모리 최적화 필수
- 부그래프 샘플링 전략 개선 필요
- 온라인 학습 지원 구현

#### 6.5.2 온라인 학습
실시간 스트리밍 데이터:
1. 고정 시간창에서 증분 그래프 업데이트
2. 개념 변화(concept drift) 감지
3. 과거 데이터에 대한 재학습 필요성 판단

#### 6.5.3 모니터링 및 유지보수
배포 후:
- 분포 변화 모니터링 (데이터, 패턴)
- 성능 저하 자동 감지
- A/B 테스트를 통한 모델 비교
- 재학습 일정 결정

## 7. 결론

MTGNN은 다변수 시계열 예측에 대한 중요한 패러다임 전환을 제시했다. 사전정의 그래프 없이도 데이터로부터 변수 간의 숨겨진 관계를 자동으로 학습하는 능력은 광범위한 응용을 가능하게 했다.

### 핵심 강점
1. **일반성**: 다양한 그래프 구조 유무와 관계없이 작동
2. **해석성**: 단방향 그래프로 인과관계 명시
3. **유연성**: 외부 지식 통합 용이
4. **성능**: 3개 벤치마크 SOTA, 다른 2개와 경쟁력

### 주요 한계
1. **정적 구조**: 학습 후 그래프 고정으로 동적 변화 미포착
2. **이론 부족**: 왜 효과적인지에 대한 수학적 설명 부족
3. **확장성**: O(N²) 복잡도로 매우 큰 그래프 처리 어려움
4. **데이터 요구**: 작은 그래프나 제한된 샘플에서 성능 저하

### 향후 발전
2020년 이후 5년간의 발전은 다음 방향을 지시한다:
- **동적 그래프 학습** (TimeGNN 등): 변수 관계의 시간적 진화 포착
- **다중 스케일 모델링** (MSGNet 등): 여러 수준의 상호작용 동시 처리
- **인과관계 명시화** (TEGNN 등): 상관성 넘어 인과성 추론
- **확장성 개선** (SAGDFN 등): 대규모 실제 시스템 적용
- **기초 모델 활용**: 사전학습을 통한 일반화 성능 향상

MTGNN은 단지 하나의 모델이 아니라 시계열 연구의 새로운 방향을 제시한 이정표이며, 그로부터 파생된 수십 개의 후속 연구들이 지속적으로 한계를 극복하고 있다.

***

## 참고문헌 및 인용 출처

<span style="display:none">[^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_80][^1_81][^1_82][^1_83][^1_84][^1_85][^1_86][^1_87][^1_88][^1_89][^1_90][^1_91]</span>

<div align="center">⁂</div>

[^1_1]: 2005.11650v1.pdf

[^1_2]: https://dl.acm.org/doi/10.1145/3394486.3403118

[^1_3]: https://www.semanticscholar.org/paper/645054d31fa26b29bbfb0cf73b75f8906c359415

[^1_4]: https://www.semanticscholar.org/paper/2a8beb7836b4e999e452448220573c0133834a55

[^1_5]: https://arxiv.org/html/2307.14680

[^1_6]: https://arxiv.org/abs/2307.03595

[^1_7]: https://arxiv.org/abs/2307.14680

[^1_8]: https://arxiv.org/pdf/2311.06190.pdf

[^1_9]: https://arxiv.org/pdf/2401.00423.pdf

[^1_10]: https://ieeexplore.ieee.org/document/9837007/

[^1_11]: http://arxiv.org/pdf/2406.12282.pdf

[^1_12]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10453913/

[^1_13]: https://arxiv.org/html/2410.22377v4

[^1_14]: https://arxiv.org/html/2305.06587v3

[^1_15]: https://arxiv.org/html/2506.15507v2

[^1_16]: https://www.nature.com/articles/s41598-024-75385-2

[^1_17]: http://arxiv.org/pdf/2405.18036.pdf

[^1_18]: https://linkinghub.elsevier.com/retrieve/pii/S0952197626001521

[^1_19]: https://www.semanticscholar.org/paper/3c90e747017bd2417d24e500f7d9bc37c052ca24

[^1_20]: https://ieeexplore.ieee.org/document/9416768/

[^1_21]: https://www.mdpi.com/1999-5903/18/1/26

[^1_22]: https://www.semanticscholar.org/paper/971602286459a3fee502456403af65e6c008ccae

[^1_23]: https://link.springer.com/10.1007/978-3-030-59419-0_44

[^1_24]: http://www.ije.ir/article_108448.html

[^1_25]: https://arxiv.org/pdf/2307.14680.pdf

[^1_26]: https://arxiv.org/pdf/2206.13816.pdf

[^1_27]: https://arxiv.org/abs/2112.03273

[^1_28]: https://pdfs.semanticscholar.org/fb76/33dcdeb50f0abdd840d44e23e4afa44a2fde.pdf

[^1_29]: https://arxiv.org/pdf/2508.07122.pdf

[^1_30]: https://arxiv.org/list/math/new

[^1_31]: https://arxiv.org/abs/2508.02069

[^1_32]: https://arxiv.org/pdf/2512.08567.pdf

[^1_33]: https://arxiv.org/list/physics/new

[^1_34]: https://arxiv.org/html/2511.05179v1

[^1_35]: https://arxiv.org/html/2307.03759v3

[^1_36]: https://arxiv.org/list/math.PR/new

[^1_37]: https://arxiv.org/abs/2410.22377

[^1_38]: https://peerj.com/articles/cs-3097/

[^1_39]: https://www.biorxiv.org/content/10.1101/2023.05.17.541153v1.full.pdf

[^1_40]: https://ar5iv.labs.arxiv.org/abs/2405.18693v1

[^1_41]: https://www.sciencedirect.com/science/article/abs/pii/S0952197625020822

[^1_42]: https://www.sciencedirect.com/science/article/abs/pii/S0925231225002164

[^1_43]: https://www.research-collection.ethz.ch/entities/publication/1ab7ad0a-0180-4067-8200-e00b4e91c3b2

[^1_44]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11490608/

[^1_45]: https://www.sciencedirect.com/science/article/abs/pii/S0952197623014756

[^1_46]: https://fre.snu.ac.kr/wp-content/uploads/sites/9/2024/04/240426_Graph_Neural_Network_for_Time-series_Forecasting-1.pdf

[^1_47]: https://www.sciencedirect.com/science/article/abs/pii/S0925231225010720

[^1_48]: https://www.youtube.com/watch?v=_uD9ooOFqss

[^1_49]: https://www.sciencedirect.com/science/article/abs/pii/S0306261924021275

[^1_50]: https://arxiv.org/abs/2307.03759

[^1_51]: https://www.sciencedirect.com/science/article/pii/S0952197625003045

[^1_52]: https://preregister.science/papers_20neurips/67_paper.pdf

[^1_53]: https://ieeexplore.ieee.org/document/8663347/

[^1_54]: https://ieeexplore.ieee.org/document/9338393/

[^1_55]: https://www.mdpi.com/2072-4292/12/7/1097

[^1_56]: https://ieeexplore.ieee.org/document/9288309/

[^1_57]: https://ieeexplore.ieee.org/document/9239613/

[^1_58]: https://meetingorganizer.copernicus.org/GSTM2020/GSTM2020-56.html

[^1_59]: https://ieeexplore.ieee.org/document/9353969/

[^1_60]: https://www.semanticscholar.org/paper/591b9bc83fa44fbc9fca1df7ac32e2a00ac34e9e

[^1_61]: https://arxiv.org/pdf/2103.07719.pdf

[^1_62]: http://arxiv.org/pdf/2106.02930.pdf

[^1_63]: https://arxiv.org/pdf/2309.05305.pdf

[^1_64]: https://arxiv.org/pdf/2401.03988.pdf

[^1_65]: https://arxiv.org/pdf/2302.01018.pdf

[^1_66]: https://arxiv.org/pdf/2210.16270.pdf

[^1_67]: https://arxiv.org/pdf/2312.07777.pdf

[^1_68]: https://pdfs.semanticscholar.org/8b84/cf8d293f745c243962b3607767f8c6337536.pdf

[^1_69]: https://arxiv.org/html/2411.05793v1

[^1_70]: https://www.semanticscholar.org/paper/Spectral-Temporal-Graph-Neural-Network-for-Cao-Wang/645054d31fa26b29bbfb0cf73b75f8906c359415

[^1_71]: https://arxiv.org/pdf/2509.23816.pdf

[^1_72]: https://arxiv.org/html/2405.18036v1

[^1_73]: https://arxiv.org/pdf/2312.02159.pdf

[^1_74]: https://arxiv.org/pdf/2302.11313.pdf

[^1_75]: https://arxiv.org/abs/2405.18036

[^1_76]: https://www.semanticscholar.org/paper/A-Multi-View-Multi-Task-Learning-Framework-for-Time-Deng-Chen/c242e09e8c5be4c8ad6ac923cc3b004b102a56b2

[^1_77]: https://arxiv.org/pdf/2405.18036.pdf

[^1_78]: https://arxiv.org/pdf/2301.10569.pdf

[^1_79]: https://www.semanticscholar.org/paper/ForecastGrapher:-Redefining-Multivariate-Time-with-Cai-Wang/60560da7e2483d4f788c228ebd5e226ae1f40002

[^1_80]: https://papers.nips.cc/paper/2020/file/cdf6581cb7aca4b7e19ef136c6e601a5-Paper.pdf

[^1_81]: https://mlg-europe.github.io/2023/papers/215.pdf

[^1_82]: https://www.themoonlight.io/en/review/forecastgrapher-redefining-multivariate-time-series-forecasting-with-graph-neural-networks

[^1_83]: https://proceedings.nips.cc/paper_files/paper/2020/file/cdf6581cb7aca4b7e19ef136c6e601a5-Paper.pdf

[^1_84]: https://grlplus.github.io/papers/58.pdf

[^1_85]: https://www.microsoft.com/en-us/research/publication/spectral-temporal-graph-neural-network-for-multivariate-time-series-forecasting/

[^1_86]: https://www.net.in.tum.de/fileadmin/TUM/NET/NET-2023-11-1/NET-2023-11-1_20.pdf

[^1_87]: https://www.scielo.br/j/cr/a/fTQZddyndYCFXh3sSQpchjG/

[^1_88]: https://emanuelerossi.co.uk/assets/pdf/intel_tgn.pdf

[^1_89]: https://arxiv.org/abs/2103.07719

[^1_90]: https://openreview.net/pdf?id=pHCdMat0gI

[^1_91]: https://dl.acm.org/doi/10.1007/978-981-96-9818-9_15
