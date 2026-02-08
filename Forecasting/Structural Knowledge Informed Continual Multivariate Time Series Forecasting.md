# Structural Knowledge Informed Continual Multivariate Time Series Forecasting

다변량 시계열(Multivariate Time Series, MTS) 예측은 금융, 의료, 교통, 에너지 관리 등 다양한 산업 분야에서 의사결정의 핵심적인 역할을 수행하며 지난 수십 년간 비약적인 발전을 거듭해 왔다.1  
초기 통계적 모델인 ARIMA나 VAR에서 시작하여 최근의 딥러닝 기반 모델인 Transformer, Graph Neural Networks(GNN), 그리고 대규모 시계열 기초 모델(Foundation Models)에 이르기까지 그 기술적 궤적은 매우 가파르다.1  
그러나 실제 환경에서 발생하는 시계열 데이터는 데이터의 분포가 시간에 따라 변화하는 비정상성(Non-stationarity)과 더불어, 시스템의 운영 로직이나 환경적 요인이 변화하는 '레짐(Regime)'의 전환이라는 고유한 특성을 지닌다.1  
이러한 환경에서 모델을 순차적으로 학습시킬 때 발생하는 가장 치명적인 문제는 새로운 지식을 습득하는 과정에서 과거에 학습했던 정보가 소실되는 '치명적 망각(Catastrophic Forgetting)' 현상이다.1

최근 발표된 "Structural Knowledge Informed Continual Multivariate Time Series Forecasting" 연구는 이러한 치명적 망각 문제를 해결하기 위해 구조적 지식(Structural Knowledge)을 학습의 가이드로 삼는 새로운 프레임워크인 SKI-CL(Structural Knowledge Informed Continual Learning)을 제안하였다.1  
본 보고서는 해당 논문의 핵심 주장과 주요 기여를 분석하고, 모델의 기술적 구조와 수식, 성능 향상 요인 및 한계를 상세히 다룬다.  
특히 2020년 이후 발표된 최신 연구들과의 비교 분석을 통해 모델의 일반화 성능 향상 가능성과 향후 연구 방향에 대한 심도 있는 고찰을 제공하고자 한다.

## **SKI-CL의 핵심 주장 및 주요 기여 요약**

본 연구의 핵심 주장은 다변량 시계열의 변수 간 의존성(Variable Dependency) 구조가 레짐에 따라 변화하더라도, 물리적 제약이나 도메인 지식과 같은 '구조적 지식'을 활용하면 각 레짐의 고유한 특성을 명확히 식별하고 보존할 수 있다는 점이다.1  
기존의 지속 학습(Continual Learning) 방법론들이 주로 모델의 파라미터를 고정하거나 단순히 과거 데이터를 무작위로 재생(Replay)하는 데 그쳤다면, SKI-CL은 시계열의 '관계적-시간적 역학(Relational-Temporal Dynamics)'을 보존하는 데 집중한다.1

이 논문의 주요 기여는 네 가지로 요약될 수 있다.  
첫째, 지속 학습 패러다임 내에서 MTS 예측 및 의존성 구조 추론을 수행하기 위한 SKI-CL 프레임워크를 최초로 제시하였다.1  
둘째, 입력 윈도우의 세밀한 단위에서 변수 간 관계를 동적으로 파악하는 그래프 구조 학습 모듈을 개발하고, 이를 구조적 지식과 정합시키는 일관성 규제(Consistency Regularization) 기법을 도입하였다.1  
셋째, 데이터의 시간적 커버리지를 극대화하여 각 레짐의 핵심적인 역학을 효율적으로 보존하는 '표현 정합 메모리 재생(Representation-Matching Memory Replay)' 체계를 제안하였다.1  
넷째, 합성 데이터 및 실제 벤치마크 데이터셋을 통한 광범위한 실험을 통해 기존의 최첨단(SOTA) 모델들을 능가하는 예측 정확도와 의존성 추론 성능을 입증하였다.1

## **해결하고자 하는 문제: 치명적 망각과 의존성 소실**

MTS 예측 모델은 단순히 과거의 수치적 추세를 파악하는 것을 넘어 변수 간의 복잡한 상관관계를 모델링해야 한다. 예를 들어, 전력망 데이터에서 특정 변압기의 부하 변화는 인접한 노드들에 연쇄적인 영향을 미치며, 이러한 물리적 연결 구조는 예측의 정확성을 높이는 중요한 정보가 된다.1 그러나 실시간으로 데이터가 축적되는 환경에서는 계절적 요인이나 장치의 교체 등으로 인해 데이터의 생성 메커니즘이 변화하는 레짐 시프트(Regime Shift)가 발생한다.1

기본적인 모델 업데이트 방식인 순차적 학습(Sequential Training)을 수행할 경우, 모델은 현재 유입되는 레짐의 의존성 구조에 과도하게 적응(Overfitting)하게 된다. 이 과정에서 이전에 학습했던 레짐의 의존성 패턴이 파라미터 업데이트 과정에서 지워지며, 결과적으로 과거 데이터에 대한 예측 성능이 급격히 저하되는 치명적 망각이 발생한다.1 공동 학습(Joint Training)은 모든 과거 데이터를 저장하고 재학습해야 하므로 계산 복잡도와 저장 공간 측면에서 비효율적이며, 데이터의 실시간성을 반영하기 어렵다는 한계가 있다.1 SKI-CL은 이러한 안정성-가소성 딜레마(Stability-Plasticity Dilemma)를 구조적 지식이라는 외부 닻(Anchor)을 통해 해결하고자 한다.1

## **SKI-CL의 제안 방법론 및 모델 구조**

SKI-CL 프레임워크는 크게 '동적 그래프 추론 모듈', '구조적 지식 기반 규제', '그래프 기반 예측 모듈', 그리고 '표현 정합 샘플 선택'의 네 가지 구성 요소로 나뉜다.1

### **동적 그래프 추론 모듈 (Dynamic Graph Inference Module)**

기존의 많은 그래프 신경망 기반 시계열 예측 모델이 전체 데이터셋에 대해 하나의 정적인 그래프를 학습하거나, 학습된 파라미터에 고정된 그래프를 사용하는 것과 달리, SKI-CL은 입력 윈도우 단위에서 변수 간 관계를 실시간으로 파악한다.1  
변수 $i$의 시간적 인코딩을 $z_i = \Phi(X_{i, t-\tau:t-1})$라고 할 때, 두 변수 $i, j$ 사이의 엣지 가중치 $\hat{A}_{ij}$는 다음과 같이 생성된다 :

$$\\hat{A}\_{ij} \= \\Psi(z\_i | z\_j)$$

여기서 $||$는 연결(Concatenation) 연산을 의미하며, $\Phi$는 시간적 패턴을 추출하는 합성곱 계층, $\Psi$는 다층 퍼셉트론(MLP)이다. 이 방식은 레짐 내에서의 미세한 변화뿐만 아니라 레짐 전환 시 발생하는 급격한 의존성 변화에 대응할 수 있는 가소성을 모델에 부여한다.

### **구조적 지식의 통합과 일관성 규제**

구조적 지식 $A \\in \\mathbb{R}^{N \\times N}$은 각 레짐의 정체성을 정의하는 참조점 역할을 한다. 논문은 지식의 형태에 따라 두 가지 규제 손실 함수를 제안한다.1

첫째, 엣지가 이진(Binary)인 경우, 모델이 예측한 엣지 확률 $\\theta\_{ij}$가 베르누이 분포를 따르도록 유도하며 이진 교차 엔트로피(Binary Cross Entropy) 손실을 적용한다 1:

$$\mathcal{L}_G = \sum_{i,j} -A_{ij} \log \theta_{ij} - (1 - A_{ij}) \log(1 - \theta_{ij})$$

둘째, 엣지가 연속(Continuous) 값인 경우, 가중치의 강도를 직접 정합시키기 위해 평균 제곱 오차(MSE) 손실을 사용한다 1:

$$\mathcal{L}_G = \frac{1}{N^2} \sum_{i,j} ||A_{ij} - \hat{A}_{ij}||^2$$

이러한 규제는 모델이 데이터의 노이즈에 휩쓸리지 않고 도메인 지식에 부합하는 관계를 학습하도록 강제함으로써 학습의 안정성을 보장한다.1 지식이 부분적으로만 관측된 경우(Partially Observed)에도 알려진 항목에 대해서만 일관성을 유지하도록 설계되어 범용성을 높였다.1

### **그래프 기반 예측 모듈: TGConv 블록**

예측 모듈은 여러 개의 시간적 그래프 합성곱(Temporal Graph Convolution, TGConv) 블록으로 구성된다. 각 블록은 팽창 인과 합성곱(Dilated Causal Convolution)을 통해 단변량 시간 역학을 캡처하고, 이후 메시지 패싱(Message-passing) 연산을 통해 변수 간 정보를 교환한다.1 메시지 패싱 과정은 다음과 같은 수식으로 표현된다 1:

$$\text{MessagePassing}(r_i) = W_1 r_i + W_2 \sum_{j \in \mathcal{N}(i)} e_{j,i} r_j$$

여기서 $r_i$는 시간적 표현, $e_{j,i}$는 학습된 엣지 가중치, $W$는 학습 가능한 가중치 행렬이다. 최종적으로 총 학습 목표 함수는 예측 오차 $\mathcal{L}_F$와 구조 규제 오차 $\mathcal{L}_G$의 합으로 정의된다 :

$$\mathcal{L}_{total} = \mathcal{L}_F + \lambda \mathcal{L}_G = \frac{1}{\tau'} \sum_{t'=t}^{t+\tau'-1} ||\hat{X}_{:,t'} - X_{:,t'}||^2 + \lambda \mathcal{L}_G$$

### **표현 정합 메모리 재생 (Representation-matching Memory Replay)**

지속 학습의 핵심은 과거의 핵심 샘플을 효율적으로 저장하고 재생하는 것이다. SKI-CL은 최대 엔트로피 원리에 기반하여, 각 레짐의 표현 공간을 가장 다양한 모드(Mode)로 분할하고 각 모드를 대표하는 샘플을 선택한다.1  
데이터 분포의 유사성을 측정하기 위해 CORAL(Deep Correlation Alignment) 거리를 사용하며, 이는 표현의 2차 통계량인 공분산 행렬 $C$를 정합시키는 방식이다 :

$$D(\cdot, \cdot) = \frac{1}{4q^2} ||C_{(\cdot)} - C_{(\cdot)}||_F^2$$

이 방식은 단순히 시간 순서대로 샘플을 저장하는 것보다 레짐의 전체적인 역학 구조를 훨씬 더 잘 보존하며, 적은 메모리 예산(예: 1%)으로도 높은 지식 보존 효과를 낸다.1

## **성능 향상 및 실험 결과 분석**

SKI-CL은 교통(Traffic-CL), 태양광(Solar-CL), 인간 행동 인식(HAR-CL), 합성 데이터(Synthetic-CL) 등 네 가지 주요 벤치마크에서 뛰어난 성능을 보였다.1 성능 지표로는 모든 학습 레짐에 대한 평균 정확도인 AP(Average Performance)와 과거 학습 레짐에서의 성능 저하 정도인 AF(Average Forgetting)를 사용하였다.1

| 데이터셋 | 모델 | AP (MAE ↓) | AF (MAE ↓) | 구조 정합도 (AP ↑) |
| :---- | :---- | :---- | :---- | :---- |
| Traffic-CL | MTGNN (er) | 15.79 | 2.76 | 0.10 (Prec.) |
|  | GTS (der++) | 15.84 | 1.15 | 0.64 (Prec.) |
|  | **SKI-CL** | **15.23** | **1.51** | **0.88 (Prec.)** |
| Solar-CL | ESG (er) | 2.01 | 0.24 | 0.35 (MAE) |
|  | **SKI-CL** | **1.75** | **0.09** | **0.17 (MAE)** |

실험 결과에 따르면, SKI-CL은 예측 오차(MAE, RMSE) 측면에서 SOTA 모델들보다 우수한 AP를 기록했을 뿐만 아니라, 특히 '학습된 의존성 구조'가 실제 구조적 지식과 얼마나 일치하는지를 나타내는 구조 정합도에서 압도적인 성능을 보였다.1 이는 모델이 단순히 수치적 예측에 최적화된 것이 아니라, 데이터 이면의 인과적·물리적 관계를 정확히 이해하고 있음을 시사한다.1 또한, 메모리 재생 기법 중에서도 제안된 표현 정합 방식이 Random Replay나 Herding 방식보다 일관되게 우수한 성능을 나타냈다.1

## **모델의 일반화 성능 향상 가능성 집중 분석**

본 연구에서 가장 주목해야 할 점은 구조적 지식의 도입이 모델의 '일반화 성능'을 어떻게 향상시키는가이다. 시계열 데이터에서 일반화란 학습 단계에서 보지 못한 새로운 레짐에 적응하는 능력(가소성)과, 과거에 학습했던 다양한 패턴을 잊지 않고 적절히 호출하는 능력(안정성)의 결합이다.3

### **구조적 지식의 앵커링 효과 (Anchoring Effect)**

일반적인 딥러닝 모델은 데이터의 노이즈나 일시적인 변동에 민감하게 반응하여 과적합되는 경향이 있다.11 특히 지속 학습 환경에서는 현재 유입되는 데이터의 편향(Bias)이 모델 전체의 파라미터를 오염시킬 위험이 크다. SKI-CL은 구조적 지식을 일종의 '정답지'가 아닌 '제약 조건'으로 활용함으로써, 모델이 학습하는 의존성 구조가 지나치게 자유롭게 변형되는 것을 막는다.1 이러한 앵커링 효과는 모델이 레짐 간의 공통적인 물리적 속성을 파악하게 하여, 유사한 특성을 가진 미래의 레짐이나 관측되지 않은 환경에 대한 대응력을 높인다.1

### **표현 공간의 다양성 보존**

제안된 표현 정합 메모리 재생 방식은 레짐 내의 단일한 평균 패턴이 아닌, 분포의 극단이나 다양한 변동 모드를 보존한다.1 이는 모델이 '대표적인 상황'뿐만 아니라 '예외적인 상황'에 대한 기억도 유지하게 함으로써, 테스트 단계에서 입력되는 시계열의 변동성이 크더라도 과거에 학습했던 유사한 모드를 찾아내어 정확하게 예측할 수 있는 기반을 마련한다.1 이는 통계적 거리를 직접 최소화하는 CORAL 기법의 특성상, 데이터의 분포 이동(Distribution Shift)에 대한 강건성을 직접적으로 향상시키는 결과를 낳는다.1

### **동적 추론을 통한 유연한 적응**

정적인 그래프 구조를 사용하는 모델은 레짐이 바뀔 때마다 그래프 자체를 다시 학습해야 하므로 일반화에 한계가 있다. 반면 SKI-CL의 동적 그래프 추론 모듈은 윈도우 단위의 특징을 기반으로 관계를 실시간 추론하므로, 한 번도 보지 못한 미세한 변화가 발생하더라도 인코더가 추출한 특징을 통해 관계를 유추할 수 있다.1 이러한 설계는 '훈련 레짐'과 '테스트 레짐' 사이의 간극을 좁히는 핵심적인 요소가 된다.1

## **모델의 한계점**

SKI-CL이 가진 뛰어난 성능에도 불구하고 실제 적용 환경에서는 몇 가지 한계가 존재한다.

첫째, 구조적 지식의 가용성과 품질에 대한 의존성이다. 논문은 물리적 제약이나 상관관계 등의 사전 지식이 가용함을 가정하고 있으나, 복잡한 시스템에서는 이러한 지식을 정의하기 어렵거나 지식 자체가 노이즈를 포함할 수 있다.1 지식의 품질이 낮을 경우 오히려 모델의 가소성을 저해하는 역효과를 낼 가능성이 있다.

둘째, 계산 비용의 문제다. 입력 윈도우마다 동적으로 그래프를 추론하고 메시지 패싱을 수행하는 구조는 변수 개수(![][image15])의 제곱에 비례하는 계산 복잡도를 가진다.1 대규모 시계열(예: 수천 개 이상의 변수) 환경에서는 실시간 예측에 어려움이 있을 수 있으며, 이를 해결하기 위한 그래프 서브샘플링이나 클러스터링 기법과의 결합이 필요해 보인다.13

셋째, 메모리 관리의 정교함이다. 표현 정합을 위한 분할 알고리즘은 그리디(Greedy) 방식으로 최적의 분할을 보장하지 못할 수 있으며, 레짐의 수가 극단적으로 많아질 경우 고정된 메모리 예산 내에서 각 레짐에 할당되는 샘플 수가 너무 적어져 지식 보존 효과가 약화될 우려가 있다.1

## **2020년 이후 관련 최신 연구 비교 분석**

SKI-CL의 위치를 명확히 하기 위해 최근 시계열 예측 분야를 주도하고 있는 최신 모델들과 비교 분석을 수행하였다.

### **Transformer 및 MLP 계열 모델 (PatchTST, iTransformer, DLinear)**

2023년 전후로 등장한 PatchTST와 iTransformer는 시계열을 패치 단위로 나누거나 변수 차원을 반전시키는 등 혁신적인 접근으로 SOTA를 경신했다.14 DLinear는 단순한 선형 계층만으로도 복잡한 모델을 압도할 수 있음을 보여주었다.14 그러나 이들은 대부분 '정적 학습' 환경을 가정한다. SKI-CL의 실험 결과에 따르면, 이러한 강력한 모델들도 순차 학습 환경(seq)에서는 SKI-CL보다 높은 망각률(AF)을 보였으며, 특히 변수 간의 명시적인 구조 학습 기능이 없어 레짐 시프트에 취약한 모습을 보였다.1

### **현대적 합성곱 모델 (ModernTCN)**

ModernTCN은 대형 커널과 가변적 수용 영역(ERF)을 통해 합성곱 신경망의 성능을 Transformer 수준으로 끌어올렸다.18 효율성 면에서는 우수하지만, SKI-CL처럼 외부의 구조적 지식을 명시적으로 통합하여 지속 학습의 안정성을 확보하는 기법은 포함되어 있지 않다.21 따라서 데이터의 물리적 연결성이 중요한 도메인에서는 SKI-CL이 더 신뢰할 수 있는 대안이 된다.

### **시계열 기초 모델 (TimesFM, Chronos)**

2024년과 2025년에 걸쳐 Google, Amazon 등이 발표한 시계열 기초 모델들은 수십억 개의 데이터로 사전 학습되어 제로샷(Zero-shot) 예측에서 놀라운 성능을 보여준다.4 하지만 이러한 대규모 모델은 특정 도메인의 물리적 규칙이나 실시간으로 변화하는 레짐의 특수성을 즉각적으로 반영하기에는 너무 무겁다.25 SKI-CL은 특정 산업 현장에서의 '도메인 특화 지속 학습'을 위한 모델로서, 기초 모델이 제공하지 못하는 구조적 설명력과 적응력을 제공한다는 점에서 차별화된다.1

| 비교 항목 | SKI-CL | iTransformer | ModernTCN | TimesFM (Found.) |
| :---- | :---- | :---- | :---- | :---- |
| 주요 메커니즘 | 동적 그래프 \+ 구조 지식 | 차원 반전 Attention | 현대적 TCN 구조 | 대규모 사전학습 |
| 지속 학습 대응 | 명시적 (Memory Replay) | 부재 (Fine-tuning 의존) | 부재 | In-Context Learning |
| 구조적 설명력 | 매우 높음 | 낮음 | 낮음 | 낮음 |
| 제로샷 능력 | 보통 | 보통 | 보통 | 매우 높음 |

## **향후 연구에 미치는 영향 및 고려할 점**

SKI-CL 연구는 시계열 예측 모델이 단순한 수치 최적화 도구에서 '지식 집약적 지능 체계'로 진화해야 함을 보여주었다.

### **연구에 미치는 영향**

첫째, 그래프 구조 학습과 지속 학습의 결합 가능성을 확인시켜 주었다. 이전의 지속 학습 연구들이 주로 분류(Classification) 작업에 치중되어 있었으나, 본 연구는 회귀(Regression) 성격이 강한 시계열 예측에서도 '관계 보존'이 망각 방지의 핵심임을 입증하였다.1

둘째, 메모리 재생 기법의 패러다임을 바꿨다. 무작위 샘플링이 아닌 '표현 공간의 기하학적 분포'를 고려한 샘플 선택이 시계열 데이터에서 얼마나 효과적인지를 보여줌으로써, 샘플 선택 알고리즘 연구에 새로운 방향을 제시하였다.1

셋째, 설명 가능한 AI(XAI)의 실질적 구현 사례를 제시하였다. 모델이 추론한 그래프 구조를 시각화하여 도메인 지식과 비교함으로써, 예측 결과에 대한 신뢰성을 확보할 수 있는 방법론적 근거를 마련하였다.1

### **향후 연구 시 고려할 점**

앞으로의 연구에서는 '구조적 지식의 동적 진화'를 고려해야 한다. 실제 시스템에서는 변수 자체가 추가되거나 삭제되는 구조적 변화가 발생할 수 있다.27 고정된 변수 집합을 가정하는 현재의 프레임워크를 넘어, 그래프의 노드가 변하는 환경에서도 지식을 전이할 수 있는 '가변적 구조 학습' 기법이 필요하다.28

또한, 대규모 언어 모델(LLM)과의 결합도 흥미로운 지점이다. 최근 기초 모델들이 텍스트 형태의 설명을 통해 시계열의 패턴을 이해하려는 시도가 늘고 있는 만큼, SKI-CL의 구조적 지식 입력을 텍스트 기반의 프롬프트나 시맨틱 정보와 결합한다면 더욱 정교한 레짐 식별이 가능할 것이다.23

마지막으로, 모델의 효율성을 위한 아키텍처 개선이다. FITS와 같은 모델이 보여준 주파수 영역에서의 보간 기법을 SKI-CL의 그래프 합성곱 과정에 도입한다면, 계산 복잡도를 획기적으로 낮추면서도 복잡한 상관관계를 모델링할 수 있는 효율적인 지속 학습 모델이 탄생할 수 있을 것으로 기대된다.18

## **결론 및 제언**

"Structural Knowledge Informed Continual Multivariate Time Series Forecasting"은 시계열 데이터의 비정상성과 레짐 전환이라는 현실적인 난제를 해결하기 위해 구조적 지식을 전면에 내세운 독창적인 연구다.1 SKI-CL은 동적 그래프 학습을 통해 가소성을 확보하고, 구조 일관성 규제와 표현 정합 메모리 재생을 통해 안정성을 유지함으로써 치명적 망각 문제를 성공적으로 완화하였다.1

이 프레임워크는 단순히 과거의 수치를 기억하는 것을 넘어, 데이터 이면에 숨겨진 변수 간의 물리적·논리적 관계를 보존한다는 점에서 매우 강력한 일반화 성능 잠재력을 지닌다.1 비록 계산 복잡도와 지식 의존성이라는 숙제가 남아있으나, 2020년 이후 쏟아지는 최신 기술들과의 융합을 통해 그 한계를 극복할 수 있을 것이다. 데이터가 끊임없이 유입되고 환경이 급변하는 4차 산업혁명 시대에, SKI-CL과 같은 지식 기반의 지속 학습 기술은 자율 주행, 스마트 시티, 실시간 금융 관제 등 높은 신뢰성이 요구되는 분야에서 필수적인 기술적 기반이 될 것으로 확신한다.

#### **참고 자료**

1. 2402.12722v1.pdf  
2. Deep learning for time series forecasting: a survey of recent advances, 2월 8, 2026에 액세스, [https://journal.hep.com.cn/fcs/EN/10.1007/s11704-025-50947-3](https://journal.hep.com.cn/fcs/EN/10.1007/s11704-025-50947-3)  
3. Online Continual Learning for Time Series: a Natural Score-driven, 2월 8, 2026에 액세스, [https://arxiv.org/pdf/2601.12931](https://arxiv.org/pdf/2601.12931)  
4. Benchmarking Foundation Models for Time-Series Forecasting \- MDPI, 2월 8, 2026에 액세스, [https://www.mdpi.com/2813-0324/11/1/32](https://www.mdpi.com/2813-0324/11/1/32)  
5. (PDF) Continual Learning for Time Series Forecasting: A First Survey, 2월 8, 2026에 액세스, [https://www.researchgate.net/publication/382421681\_Continual\_Learning\_for\_Time\_Series\_Forecasting\_A\_First\_Survey](https://www.researchgate.net/publication/382421681_Continual_Learning_for_Time_Series_Forecasting_A_First_Survey)  
6. Report on "Online Adaptive Multivariate Time Series Forecasting" by, 2월 8, 2026에 액세스, [https://www.researchgate.net/publication/399996889\_Report\_on\_Online\_Adaptive\_Multivariate\_Time\_Series\_Forecasting\_by\_Amal\_Saadallah\_Hanna\_Mykula\_and\_Katharina\_Morik](https://www.researchgate.net/publication/399996889_Report_on_Online_Adaptive_Multivariate_Time_Series_Forecasting_by_Amal_Saadallah_Hanna_Mykula_and_Katharina_Morik)  
7. STRUCTURAL KNOWLEDGE INFORMED CONTINUAL, 2월 8, 2026에 액세스, [https://openreview.net/pdf/f2a793171fe4f7efe9861755bc0609e985afb7fb.pdf](https://openreview.net/pdf/f2a793171fe4f7efe9861755bc0609e985afb7fb.pdf)  
8. Structural Knowledge Informed Continual Multivariate Time Series, 2월 8, 2026에 액세스, [https://openreview.net/forum?id=URCfZ2NgaR](https://openreview.net/forum?id=URCfZ2NgaR)  
9. Foundation models for time series forecasting \- arXiv, 2월 8, 2026에 액세스, [https://arxiv.org/pdf/2507.08858](https://arxiv.org/pdf/2507.08858)  
10. Online Time Series Forecasting with Theoretical Guarantees, 2월 8, 2026에 액세스, [https://openreview.net/pdf/e8aca67557b5958853062ccc2bf7e90fdf8bc749.pdf](https://openreview.net/pdf/e8aca67557b5958853062ccc2bf7e90fdf8bc749.pdf)  
11. NeurIPS Poster Selective Learning for Deep Time Series Forecasting, 2월 8, 2026에 액세스, [https://neurips.cc/virtual/2025/poster/116357](https://neurips.cc/virtual/2025/poster/116357)  
12. Multivariate Probabilistic Time Series Forecasting with Correlated, 2월 8, 2026에 액세스, [https://neurips.cc/virtual/2024/poster/94440](https://neurips.cc/virtual/2024/poster/94440)  
13. Graph Deep Learning for Time Series Forecasting \- arXiv, 2월 8, 2026에 액세스, [https://arxiv.org/html/2310.15978v2](https://arxiv.org/html/2310.15978v2)  
14. Extraordinary Mixture of SOTA Models for Time Series Forecasting, 2월 8, 2026에 액세스, [https://arxiv.org/html/2510.23396v1](https://arxiv.org/html/2510.23396v1)  
15. Continual Learning for Time Series Forecasting, 2월 8, 2026에 액세스, [https://itise.ugr.es/PresentacionPDF/ITISE2024\_Slides\_presentation\_2474.pdf](https://itise.ugr.es/PresentacionPDF/ITISE2024_Slides_presentation_2474.pdf)  
16. COMRES: SEMI-SUPERVISED TIME SERIES FORECASTING, 2월 8, 2026에 액세스, [https://proceedings.iclr.cc/paper\_files/paper/2025/file/6ca155a091aedc939d289df8f16f6c75-Paper-Conference.pdf](https://proceedings.iclr.cc/paper_files/paper/2025/file/6ca155a091aedc939d289df8f16f6c75-Paper-Conference.pdf)  
17. Tailored Architectures for Time Series Forecasting: Evaluating Deep, 2월 8, 2026에 액세스, [https://arxiv.org/html/2506.08977v1](https://arxiv.org/html/2506.08977v1)  
18. \[ICLR 2024\] Advancements in Time Series Forecasting \- LG AI ..., 2월 8, 2026에 액세스, [https://www.lgresearch.ai/blog/view?seq=424](https://www.lgresearch.ai/blog/view?seq=424)  
19. Multi-Scale and Interpretable Daily Runoff Forecasting with IEWT, 2월 8, 2026에 액세스, [https://www.mdpi.com/2073-4441/18/2/183](https://www.mdpi.com/2073-4441/18/2/183)  
20. ModernTCN: A Modern Pure Convolution Structure for General Time, 2월 8, 2026에 액세스, [https://openreview.net/forum?id=vpJMJerXHU](https://openreview.net/forum?id=vpJMJerXHU)  
21. ModernTCN Revisited: A Critical Look at the Experimental Setup in, 2월 8, 2026에 액세스, [https://openreview.net/forum?id=R20kKdWmVZ](https://openreview.net/forum?id=R20kKdWmVZ)  
22. ModernTCN Revisited: A Critical Look at the Experimental Setup in, 2월 8, 2026에 액세스, [https://openreview.net/pdf?id=R20kKdWmVZ](https://openreview.net/pdf?id=R20kKdWmVZ)  
23. Time series foundation models can be few-shot learners, 2월 8, 2026에 액세스, [https://research.google/blog/time-series-foundation-models-can-be-few-shot-learners/](https://research.google/blog/time-series-foundation-models-can-be-few-shot-learners/)  
24. The arrival of foundation models in time series forecasting \- PricePedia, 2월 8, 2026에 액세스, [https://www.pricepedia.it/en/magazine/article/2025/11/07/the-arrival-of-foundation-models-in-time-series-forecasting/](https://www.pricepedia.it/en/magazine/article/2025/11/07/the-arrival-of-foundation-models-in-time-series-forecasting/)  
25. Lightweight Online Adaption for Time Series Foundation Model, 2월 8, 2026에 액세스, [https://icml.cc/virtual/2025/poster/44485](https://icml.cc/virtual/2025/poster/44485)  
26. Structural Knowledge Informed Continual Learning for Multivariate, 2월 8, 2026에 액세스, [https://openreview.net/forum?id=B1TnT6lUnU](https://openreview.net/forum?id=B1TnT6lUnU)  
27. GinAR+: A Robust End-to-End Framework for Multivariate Time, 2월 8, 2026에 액세스, [https://www.computer.org/csdl/journal/tk/2025/08/11002729/26GmQWuN0g8](https://www.computer.org/csdl/journal/tk/2025/08/11002729/26GmQWuN0g8)  
28. Large Language Models for Financial and Investment Management, 2월 8, 2026에 액세스, [https://web.media.mit.edu/\~xdong/paper/jpm24b.pdf](https://web.media.mit.edu/~xdong/paper/jpm24b.pdf)  
29. ‪Anderson Schneider‬ \- ‪Google Scholar‬, 2월 8, 2026에 액세스, [https://scholar.google.com.tw/citations?user=KLyaFtUAAAAJ\&hl=th](https://scholar.google.com.tw/citations?user=KLyaFtUAAAAJ&hl=th)  
30. A High-Fidelity Benchmark for Multimodal Time Series Forecasting, 2월 8, 2026에 액세스, [https://arxiv.org/html/2509.24789v1](https://arxiv.org/html/2509.24789v1)  
31. A Unified Benchmark for Automatically Describing Time Series \- arXiv, 2월 8, 2026에 액세스, [https://arxiv.org/html/2509.05215v1](https://arxiv.org/html/2509.05215v1)  
32. FITS: Modeling Time Series with 10k Parameters \- ResearchGate, 2월 8, 2026에 액세스, [https://www.researchgate.net/publication/372248505\_FITS\_Modeling\_Time\_Series\_with\_10k\_Parameters](https://www.researchgate.net/publication/372248505_FITS_Modeling_Time_Series_with_10k_Parameters)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAcAAAAXCAYAAADHhFVIAAAAaElEQVR4XmNgGHigAMT30QVh4C0Q/0cXpAx0AnECuiAI/IDSIPsckSVmAjETlA2SdEWSY6iF0v0MeFwKkihEFwSBPAaELmEgNkGSA0u8g7IfI0uAwDMgPsQAsT8TTQ4MAoBYDF1w6AAA4oAS3/pLqloAAAAASUVORK5CYII=>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAABCklEQVR4XmNgGAWDDSgA8X10QRIBGwMeM94C8X90QRLBSgbKzRikoBOIE9AFSQDMQLwIiAPQJUDgB5QGhZ0jsgSRQBqIH0HZIDPakeQYZgIxE5QNknRFkiMWIEcqiH0Vic9QC6X7GciPfVUkNsiMOCQ+HIAkLqALkgicGXA4UoQBIiGGLkEiOMKAw4LzDAiJcgZIpMGAJxBrI/HxAZAZsMhGASCJHVD2KyRxRqgcVldhASB18eiCIAALuz/oEkCwEYgfoAtiAW4MxDsEA5xCF8ACTjJQYAEujSAf+0DZIDULEFLEg10MkFSGDhIYIIYaAPFDIH6OIksCQM5E6CASiBcCsTi6xCggGgAAiGY16EuggsoAAAAASUVORK5CYII=>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAXCAYAAAA7kX6CAAAAqklEQVR4XmNgGHngNRB/AuL/SPg7VJwoYMIA0dSJLkEIuDFANE5Dl8AGpID4JxC/AOJqBojG5UB8Ccq2RChFgDgGiGQGlA9zaj2UrwPlY7gAFhAwgK4RBNDVYBXEp5EHSQzsN5AgI5SPTyMGAGl+B2Wja7wI5fND+RjAHogfMiBMB2GQYRHIivABshOAJwNE42x0CVzgDxC/BOInQPwYSoMSxA9kRaOAVgAA62g7rQNfQXYAAAAASUVORK5CYII=>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAXCAYAAADUUxW8AAAAn0lEQVR4XmNgGAUg8AaI/wHxfyT8FohjkRURAjCNBMEiIJ6JJoZNcysQLwZidWTBKgaIwp1IYuiao6D8I0hicICuGJ2/EcrnRRKDA3TF6Py7aHwUgK6YEB8FJDBAJB2gfGTFjFD2aSgfK3jKAFHEB6Vhmr8gsfGCcCB+wYDQDMItKCqIAHj9SAiQpfkDED8D4sdQDAoHUNpORVY0CugNABaZPmrN/QAWAAAAAElFTkSuQmCC>

[image5]: <data:image/png;base64,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>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAAA9CAYAAAAQ2DVeAAAFGElEQVR4Xu3dWehtUxgA8IVERGS6JcUDwoOkeBFKpkiKePKgeKBISSHD34MhlJS8KJlKSV4UMoZEIh4UZYhklkxlHtbn7N1/3e+e/3CGe89x7u9XX3utb529z97nZX/tffZepQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMBQ19T4JycBAJgPH9Y4oyjYAADmnoINAGDOKdgAAOacgg0AYM4p2AAA5ty4Bdv7NT6u8WmNz2p8UePLEWNRbVNjn5wEABjXuAXbvmWwbsThaWw115Xl9aKwWTTb1XimxiU1Dkpjk4jf+9caH+QBAGBxPV/j5zIonOIq2XMbD6/LDmW5+BrVATX+zsnOYam/V+pvLbZv2n91yygI32zyAMCEjqpxcU4umIfL+EXb6TnRydt6KfWH+TYnZiwfwziuzIlq2xpv5CQAMJ7vy/iFzP9Nf5zT+t9W/s3WU7DldWZtGvtzVU5UH+UEADCe9mQ9jRP3vNtQlou2uKo4qfybrVWwfVM2XWeWplWoX536J6Q+ADCmOFH/kZNbgbhNN61CJW9jtYIt/nu3f9l0nVk5tltOY39irtfeuzXOr3FBGTzUAABMIE7UO+XkVqIv2CYtVvL6qxVsZ3XLWGf3dmBG3u6W+RjGcW1OAACTu72s70QdnzmuxiE1zk1jrXtqPJji/hr31bi3G583fcH2bB4YQf4NVyrY2s9F++ymHz5J/ZNTf9ra/czHEPL+DHuooHV9TgAAk4uT9J05meQTebxba3Ppi6dJYxTxCo5x1mvldVcq2M5r2rHODU0/5P7xqd/Kx5zjtOWPDrVzGdy27MU68QqOVt6fC1M/W8oJAGByudDIXqmxa8q9kPqL4OUaf+bkCPLvOKxgy+9wi378vr1dmnY4uAwejthcfkv9OIYTm37en7DWrfOlnAAAJrNj2bTQCLvV+K5rDxtfRJMeZ14/F2x7lI1fKhuiWOvXu6NbvtUt+ytxebvTspQTZfBdt6Vcvz+9tfZnKScAgMm0t89ytJ/pHVjjiRqnNLl5EP/DO6nGqzWOSGNriamm1ipC1iNvoy/YYnaE/v127ywPlye7XETM4BAuWh7+z7T2LYsre7Hdh5rci10u4rUul/cn/JgTyVJOAADj269sfMvryBpP13igyYVcMPye+ptTXN2JJxjztEZxZeqpslzoxD7u2bRHEZ8/NSfHkL83X2Fbj7yNS8vgOGcl709Y66GDpZwAALaMOHE/UgZPe0ZhtyVF4TOscBiWCyvlh/kqJyaQv3ecgu29Gpc1/dhmfghgS8r707ZXspQTAMBii1uC4acaPzT5Q2vc2vR7X+fEKh4tg6uMo7orJzrTKNh6Z3bL9zfKzs7NNW4pmx7jMEs5AQAstseadlssfN60e/d1y8vb5ArOKYM38I/q7rK+omUaxikmN6ejcwIAILTFUbRv6tp5Gq2/ymC8j9XsXQb/1VuvY2r8Upa3He9sAwCg83jTjhkS+mLsxiY/iv6py0kCAIBG/sN9FEz9/9qmJbYXsW0X094+AMDCuiInymBKLFe5AADmxEqF2Up5AAAW2Gq3QVcbAwBgC1ntqt2XOQEAAAAAdGJi9g1lMF9pdm0Z3A6NCe8BAJihePJ035zsxOTrAADMWDtH6bD3vgEAMGMX13i9a+cCLfcBAJgTu3bLdvJ5AADmwEvd8oMab7YDAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAsuH8BMLoyiQVsg6gAAAAASUVORK5CYII=>

[image7]: <data:image/png;base64,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>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAZCAYAAAABmx/yAAAArklEQVR4XmNgGAWjAAh+AvEnID4BxBZA/AeIHwPxQmRF6CAKiA2A2BqI/wPxUag4iA3COAHIdBBYxICq8AsQhyDxQYARmVMLpe8xoGrkQGLDwAd0ARAAaTqALogEQLYFowuCAEijPbogIRDBgD8gtgHxWyBmQZe4zIBboxgQKwPxWSD2R5MDh+wUdEE0gMtgvMATiG+iCxIDfgCxLBDvQ5cgBAqB+BQQy6BLDGcAAOiUIXxJ/Af+AAAAAElFTkSuQmCC>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABUAAAAYCAYAAAAVibZIAAAA+klEQVR4Xu2SMetBYRTGT7EaWJAPwIcgi8liN6AsJotPoP7FqnwEBhl9BqMyyGCRMlooEuF/jvfcOh3v9Vpkub964n1+PbfrXgAB3yCE2WMeIjt2GcxRuRU7YqtcWbgnJxY2vJGNBqalS485+A/fXfSqC8kYzDCp+grmzk7Tx8R0KemAGeZUf8ZM2YWVW6rzC3Uww5rohpgIZsAuLdxCfPclD2b4J7oZf7bZFfkcBfPTnaTADEd83ghXZdfk80U4JzSku0tguqLPsuthCpiScE5oeMDcVE//CHITi3NCQwrdjcZzcS1c0MjveZFb6/ITaEhv1ga5gIBf8A/vQUX4wN6bNQAAAABJRU5ErkJggg==>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAAXCAYAAAD+4+QTAAABCUlEQVR4XmNgGAVUBiJAPBOIvdAlqAGEgfg/GtZAUUEh0GWAGMoF5cMs8YOroAIAGdiMxH8KxO+R+BSDSwwQS2gKQBZcQRekJshkgFgCihOagd8MdAqqv+iCeMBiIH4DxP+AeAUQLwPifQwQc2yR1MGBPgNmqiIG/ALiLjSxJCBmQRMDg4MM5AUVSA8PlK0MpX2hNAYAKb6GLogE0hgwHQHKrMhi75DYGGAXAyJXwzAonGcBsQEQc0LFLGAaoKCVARKHl4H4GwMkNLACewaIgaxAbM0AKQxBfHQLBWEakAAoPnqQ+I5IbKoBkAP40AWpCUCGo8cRVQEo/L8C8Wcg/gHEgajSo4ACAACKdkIEK4NEtgAAAABJRU5ErkJggg==>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAAXCAYAAAD+4+QTAAABE0lEQVR4XmNgGAVUBiJAPBOIvdAlqAGEgfg/GtZAUUEh0GWAGMoF5cMs8YOroAIAGdiMxH8KxO+R+BSDSwwQS2gKQBZcQRekJshkgFgCihOagd8M5AcVExArIPFBqRMrAFnwF12QAMgB4i9AHA3EQkB8lwGSr1yRFcGAPgNmqiIE7gFxO7ogA57QOMiARxIL2A3Er9EFoQCnOSCJa+iCSCCNAVUziA0qcrABrEG1iwGRq2H4HxDPAmIDIOaEillA1adA+UQDewaIgaxAbM0AiTQQH91CQZgGIFgNFUcGH6BiML0UgzgG7AaBggmbONkAZJgpmthjID6JJkYRAGW+z0D8A4i3AfFDIGYEYltkRaOAZAAA9ZBFGVyDqIoAAAAASUVORK5CYII=>

[image12]: <data:image/png;base64,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>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAYCAYAAAAlBadpAAAAsUlEQVR4XmNgGJZAGogLgHgmECshiVshsTHAYiD+D8S3gdgbiFWBeBoQPwdiS6gcVgCS+AfE/OgSQFDJAJG/hC4BAn8Y8JgKBSD5IHTBD1AJTnQJNIBhuC5U8Ba6BBaAofkvVBCbPwkCkEYME4kFZGtmZoBofIkugQVgtYAYmy2AOAFdEATuMkA0g1yBDYDEX6ELIgOQZlAiQTfACIhfo4lhBbsZEF74CqVTUVSMgqEIAG1gK0HBSgf2AAAAAElFTkSuQmCC>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAAA2CAYAAAB6H8WdAAAE9ElEQVR4Xu3dW6g1YxgA4M8hZ0mOJeVHRLmRkmM7SeKCIleOoZ+4IBfKIZQLyQW3P6EkuZLCn/O6VCgpkUN/crpQzuJPDvM2a+Xbr5m11957rbXXzvPU2/red9ZeM7P3rnmbmfVNKQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADBzF+QCAACLYVsTDzbxd14AAMBi0bABACw4DRsAwILTsAEALDgNGwDAgtOwAQAsOA3bbBxR2t/ttXkBAMCkLm7irdI2Fd818fryxatyaC5Q/hy+ftXEu/UCAGB+XsyFVbo9FzaRU1O+PeX/V79X4/eHr2828UdVBwDW6MvSnmmK+LmJH4bjb5vYvXrfyDQuI25p4oRc3CROS/nLKR+5u7S/q8uaOKqJH0t7Nu7h6j0b4anSbteFTVzaxF+l3ac467geXf8XXTUAYI1GDVvtvI5a5Aek2lrFZ+2Si5vA6Sl/JeXhp9I2bFn+fc5brP/cXCzT2a78GTtTDgCsUxxs78vF0tYvqvJfqvF6PVQ250H9zJS/lvL3StuwdclNzTzFumPbukxju+rP+LCJq5u4qYlXqzoAsA59B+y6kYtLe9N+WHrfehfZWSmvv7BwYxm/T9fkwpw8W8Zv1xW5sAbjPh8AWKdoxPoOtlE/Zzh+vomDq2XT0LfesGOC2Ahnp/yNahz7M26fxtm1zO4ScWxT16XbLsem/LCU91nrfgMAE4gD7b252Hi0LD8If16NR+KS1yQH6m9K9/u6atMyap6mEbWllA+qcdf7++zIhcZJuZDk7coRXyToEsuOzsUOu+XCUHxhYiWT7jcAsAZ9B9qo1/dnvVONR/Zo4qVc7BCNyGO5WPrXvciWUj6oxuMatruq8bh7Ab/OhSmIbdorF4fqv3Hfth9Y2suq43T9bF27shoDAKvUdaCNqT1iyofaI00ck2rZ3k0ckotjdK170S2lfFCNHy/9+1Tf/7ejGmd9P78en5V/50arPdPEkVX+RDXOVtquvHyfJh4Yji+pFwAAq3NGWX6gjfuo4tLn01Vt5Lgm7snFZNwZpuygMvl752n0WKVP84KhpZQPUv5bE89V+eHlv5PHxv2AfeLJC7MQ+1R/E/jk0k7dUrs15bWV/lZ5+f1N7D8c1/f5AQAzlg/KXSZ9FNGgiW25OENxxmcSo8cqxf1cXfuylPJBykM0ty+UduqSLvWZrPPL8m3raxSnIaYkieapawqXcFU1vrMah5X+9nl5nfddjgUAZiDur9qai8knudAjH+BnbbXri7ONXfftLaV8kPJJfJwLldVu5zSNa6BX2q68POcAwBzFpKh97siFHnmy2VmLBmm1DUTf+5dSPkj5JPo+O/RNbjsPfdu1b1n5kVr1z8al1ni02fFVDQCYs8tzYYHtWdpLcn3NSJxVinv54ozayA3D11lenvwgF0r7zNGN1HfZ+Ptc6HBKLgAATGp0w39XwzaqjR6IHkaPVbquzPaxSl0T0o77MsK8xJx5tZi6BQBgZk6sxrlhWykHAGDGYs64J6u8bshirrg6v74sfyYoAAAboG7QbivLL/3FsrjXDQCADVQ3bPHtxY+G4y1pGQAAGyCmmPiitPPI/TqsvV3aRu3m4SsAAAsqmrWduQgAwGK4pbQN2/Ym9kvLAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAApuofkjwLDbj3OkoAAAAASUVORK5CYII=>

[image15]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAAYCAYAAAD3Va0xAAAA3UlEQVR4XmNgGAWkgnlA/BmI/0PxAhRZCPjLgJAHYWdUaVSArBAb2AfEKuiC6IARiLcD8XoGiEFBqNJggMsCFJAPxCZQNi5X/UEXwAbeIrE/MEAM4kMSUwPiTiQ+ToDsAlA4gPg3kcSWATEPEh8rAIXPZjQxdO9h8yoGQA4fZDGQ5m4o/xeSHE7wDl0ACmCu0gbiFjQ5rACXs3czQOTuATEnmhwGYAHiveiCUMDEgBlWWAEzEL8B4pPoEkjgGxB/RxdEBquA+CMDJP2A0g0oL2ED+kCcjS44CkYBEAAABi803bhnVOIAAAAASUVORK5CYII=>
