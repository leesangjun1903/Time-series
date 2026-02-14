# Do We Really Need Graph Neural Networks for Traffic Forecasting?

- 이 논문은 교통 예측에서 **GNN 기반 STGNN이 반드시 필요하지 않으며**, 단순한 GNN‑free 공간 모듈과 일반적인 시계열 인코더만으로도 비슷한 정확도를 더 높은 효율로 달성할 수 있다고 주장한다.[^1_1][^1_2]
- SimST라는 프레임워크를 제안하여 (1) 로컬 근접성(Local Proximity)과 (2) 글로벌 상관(Global Correlation)을 GNN의 메시지 패싱 없이 모델링하고, 노드 기반 배치 샘플링으로 일반화를 개선하면서, 5개 벤치마크에서 최대 39배 높은 TPS(throughput)를 달성한다.[^1_3][^1_1]

***

## 1. 해결하고자 하는 문제

- 현행 교통 예측 SOTA는 대부분 STGNN으로, 시간 방향은 RNN/TCN/Transformer, 공간 방향은 GNN(특히 GCN류, adaptive adjacency)을 사용한다.[^1_4][^1_5][^1_1]
- 문제점:
    - Dense/Adaptive adjacency 사용 시 시간 복잡도가 $O(L|V|^{2}D_m)$에 달해, 노드 수가 많거나 그래프가 조밀할수록 추론이 비효율적이다.[^1_1]
    - 실시간·대규모 ITS(교통 정보 시스템)에서 레이턴시 제약 때문에 실제 운영이 어렵다.[^1_2][^1_1]
- 연구 질문:
    - “메시지 패싱 GNN 없이도, 공간 구조를 충분히 활용하면서 교통 예측 정확도를 유지할 수 있는가?”[^1_1][^1_3]

***

## 2. 제안 방법: SimST (수식·모델 구조 중심)

### 2.1 기본 설정 (문제 정의)

그래프와 시계열은 다음과 같이 정의된다.[^1_1]

- 센서 그래프: $G = (V, E)$, $|V|$는 센서 수.
- 입력 시계열: $X_{T_h} \in \mathbb{R}^{|V| \times T_h \times F}$.
- 예측 대상: $\hat{Y}_{T_f} \in \mathbb{R}^{|V| \times T_f \times 1}$.
- 예측 함수:

$$
G, X_{T_h} \xrightarrow{\ \Theta\ } \hat{Y}_{T_f}
$$
- 학습 손실 (MAE):

$$
\mathcal{L}(\Theta)
= \frac{1}{|V|} \sum_{v_i \in V} \left| \hat{Y}_{T_f}^{(v_i)} - Y_{T_f}^{(v_i)} \right|.
$$
- 인접행렬 $A$는 도로 네트워크 거리의 가우시안 커널로 정의:

```math
A_{ij} =
\begin{cases}
\exp\left(-\dfrac{\text{dist}(v_i, v_j)^2}{s^2}\right) & \text{if } \text{dist}(v_i, v_j) \le r,\\[4pt]
0 & \text{otherwise.}
\end{cases}
```

[file:1]


### 2.2 기존 GNN 메시지 패싱 (비교를 위한 기준)

전통적인 GCN‑스타일 레이어는 다음과 같다.[file:1]

$$
H^{(l)} = \sigma(\tilde{A} H^{(l-1)} W^{(l)})
$$

- $\tilde{A}$: 정규화 인접행렬,
- $W^{(l)}$: 가중치,
- $\sigma$: 활성함수.

SimST는 이 메시지 패싱을 직접 사용하지 않고, 두 모듈로 **기능만 근사**한다.[file:1]

### 2.3 Local Proximity Modeling (로컬 공간 상관)

목표: 한 레이어 GNN의 “1‑hop 이웃 집계” 기능을, **ego‑graph + MLP**로 대체.[file:1]

1. 정규화 인접행렬:

$$
\tilde{A} = \tilde{D}^{-\frac{1}{2}} (A + I)\, \tilde{D}^{-\frac{1}{2}}
$$

2. 노드 $v$의 정방향·역방향 1‑hop 이웃을, $\tilde{A}$의 가중치 기준 top‑ $k$ 로 선택:
    - $N^{1}_f(v)$: forward 방향 이웃,
    - $N^{1}_b(v)$: backward 방향 이웃.[file:1]
3. ego‑graph 특성 행렬 구성:

$$
X^{T_h}_{G_v}
= \text{COMBINE}\Big(
  X^{T_h}_v,\;
  \{X^{T_h}_{v_f}: v_f \in N^{1}_f(v)\},\;
  \{X^{T_h}_{v_b}: v_b \in N^{1}_b(v)\},\;
  X^{T_h}_{\text{avg}f},\;
  X^{T_h}_{\text{avg}b}
\Big)
$$

- 차원: $X^{T_h}_{G_v} \in \mathbb{R}^{T_h \times (2k+3)}$.

[file:1]

4. 각 시점별로 MLP로 임베딩:

$$
H^{T_h}_{G_v} = \text{MLP}\big(X^{T_h}_{G_v}\big) \in \mathbb{R}^{T_h \times D_m}.
$$

특징:

- 메시지 합산($\tilde{A}H$) 대신, **입력 전처리 시점에 ego‑graph를 구성**하므로 훈련·추론 시간에 의존성이 없다.[file:1]
- 복잡도: 평균 degree $R$에 대해 $O(|V|R)$.[file:1]


### 2.4 Global Correlation Learning (글로벌 공간 상관)

목표: 여러 GNN 레이어를 쌓아 **원거리 노드 상관**을 얻는 기능을, **정적 센서 임베딩**으로 대체.[file:1]

1. 센서 임베딩 테이블:

$$
E \in \mathbb{R}^{|V| \times D_n}, \quad E_v \sim \text{random init}
$$

2. 임베딩을 hidden 차원으로 매핑:

$$
H = \text{MLP}(E) \in \mathbb{R}^{|V| \times D_m}, \quad H_v \in \mathbb{R}^{D_m}.
$$

3. 학습 후에는 코사인 유사도

```math
\displaystyle \text{sim}(v_i, v_j) = \frac{E_{v_i}\cdot E_{v_j}}{\|E_{v_i}\|_2\|E_{v_j}\|_2}
```

가 물리적 거리와 반비례하는 경향을 보여, “근접할수록 유사도↑”라는 지리학적 법칙을 재현한다.[file:1]

역할:

- 로컬 ego‑graph에서 얻기 어려운 **장거리·구조적 유사성**(예: 고속도로–램프 교차점)을 캡처.[file:1]
- noisy adjacency에서 GNN이 오염되기 쉬운 경우, 데이터 기반으로 관계를 재학습하는 보완 모듈 역할.[file:1][]
- over‑squashing(멀리 있는 정보가 좁은 bottleneck에 압축되는 현상)을 완화하는 직접 연결 구조.[file:1]


### 2.5 Temporal Encoder \& Predictor

SimST는 시간 모듈에 대해 **아키텍처 비종속적(agnostic)** 하며, 세 가지 백본을 실험한다.[file:1][]

- 입력: $H^{T_h}_{G_v} \in \mathbb{R}^{T_h \times D_m}$.

1. GRU 기반:
    - GRU로 시간 축을 압축: $h_{G_v} \in \mathbb{R}^{D_m}$.
2. WaveNet 기반 (TCN):
    - dilation convolution으로 $T_h \to 1$ 요약: $h_{G_v} \in \mathbb{R}^{D_m}$.
3. Causal Transformer:
    - self‑attention에 causal mask를 적용해 현재 시점이 미래를 보지 못하도록 제약.
    - 최종적으로 마지막 시점 representation만 사용: $h_{G_v} \in \mathbb{R}^{D_m}$.[file:1]
4. 예측기:
    - 공간 요약과 위치 임베딩을 concat:

$$
z_v = h_{G_v} \,\Vert\, H_v \in \mathbb{R}^{2D_m}
$$

- MLP predictor:

$$
\hat{Y}_{T_f}^{(v)} = \text{MLP}(z_v).
$$

### 2.6 Node-based Batch Sampling (훈련 전략)

기존 STGNN:

- 배치 단위: 그래프 전체.
- 입력: $X_{T_h} \in \mathbb{R}^{B \times |V| \times T_h \times F}$.[file:1]

문제:

- SimST는 node-wise 모델이라 실제 batch size가 $B^\* = B \cdot |V|$가 되어, gradient 노이즈가 감소·일반화 악화 및 메모리 폭증.[file:1]
- 동일 시점의 모든 노드가 한 배치에 들어가 **샘플 다양성 부족**.[file:1]

제안:

- 배치 단위를 “노드–시간 윈도우”로 변경:
    - 입력: $X_{T_h} \in \mathbb{R}^{B^\* \times T_h \times F}$, $B^\* \ll B|V|$.
- 경험적으로 $B^\* \approx 1024$에서 최적 성능; 너무 큰 $B^\*$는 generalization gap을 유발.[file:1]
- node-based sampling이 동일한 $B^\*$에서 graph-based 대비 검증 MAE와 수렴 안정성이 모두 개선.[file:1]


### 2.7 복잡도 비교

- GCN 기반 STGNN (predefined sparse $A$):

$$
O(L|E|D_m + L|V|D_m^2).
$$
- adaptive dense adjacency 사용 시:

$$
O(L|V|^2D_m + L|V|D_m^2).
$$
- SimST:
    - Local Proximity: $O(|V|R)$, $R$ = 평균 degree.
    - Global Correlation: $O(|V|D_nD_m)$.[file:1]

따라서 노드 수가 커질수록 SimST는 **선형 스케일링**을 보장해, 실시간/대규모 환경에 적합하다.[file:1][]

***

## 3. 모델 구조, 성능 향상 및 한계

### 3.1 전체 구조 (단일 노드 관점)

각 노드 $v$에 대해:[file:1]

1. 과거 $T_h$ 구간의 ego‑graph 시계열 $X^{T_h}_{G_v}$ 구성.
2. MLP로 로컬 특징 추출 → $H^{T_h}_{G_v}$.
3. Temporal encoder(GRU/WaveNet/Causal Transformer) → $h_{G_v}$.
4. 위치 임베딩 $H_v$ 추출.
5. $[h_{G_v} \Vert H_v]$를 MLP predictor에 입력 → $\hat{Y}_{T_f}^{(v)}$.

이 때 **모든 노드가 동일한 파이프라인을 공유**하며, 그래프 전체가 아니라 node‑wise로 처리되므로 병렬화와 메모리 측면에서 유리하다.[file:1]

### 3.2 성능: 정확도

5개 벤치마크(PeMSD4/7/8, LA, BAY)에서 비교 결과:[file:1][]

- MAE, RMSE, MAPE에서 SimST‑GRU/CT/WN이 GWNET, STGNCDE, AGCRN 등 강력한 STGNN과 **동급 혹은 소폭 우위**의 성능을 보인다.
- 예: PeMSD7에서 SimST‑GRU는 MAE 20.14로, 가장 강력한 baseline보다 유의하게 낮은 MAE를 달성(유의수준 0.05).[file:1]
- 요일(day‑of‑week) feature 추가 시, 모든 데이터셋에서 MAE가 추가로 감소하여, **간단한 prior feature + SimST**가 SOTA STGNN을 상회하는 경우도 나타난다.[file:1]


### 3.3 성능: 효율성

- TPS(throughput per second) 기준:
    - SimST‑WaveNet이 GWNET 대비 3.3–5.6×, AGCRN 대비 5.8–6.6×, STGODE/STGNCDE 대비 11–39× 빠르다.[file:1][]
- 파라미터 수:
    - SimST variants는 약 130–170K 수준으로, 수백 K–수 M 파라미터를 쓰는 STGNN보다 작다.[file:1]
- GPU 메모리:
    - SimST‑CT는 모든 데이터셋에서 약 2GB 수준으로 고정되는 반면, GWNET 등은 $|V|$ 증가에 따라 3GB→11GB까지 증가.[file:1]

이로부터, **정확도를 유지하면서 실시간·대규모 배치에 필요한 효율성**을 확보했다는 것이 핵심 성과이다.[file:1][]

### 3.4 한계 및 저자 논의

논문에서 명시한 한계:[file:1]

- (1) **공간 모듈의 근사성**:
    - 노드 기반 배치 샘플링을 끄면, SimST‑GRU/CT는 STGNCDE, GWNET 등을 따라잡지 못한다.
    - 즉, 두 공간 모듈만으로는 GNN의 기능을 완전히 대체하지 못하며, 훈련 전략이 성능에 필수적이다.
- (2) **이웃 정보가 풍부한 경우의 성능 격차**:
    - LA처럼 평균 degree가 높은 데이터셋에서, SimST‑CT가 SOTA STGNN 대비 약간 뒤쳐진다.
- (3) **도메인 일반성의 미검증**:
    - 실험은 모두 교통 예측에 한정되어 있고, 공기질·전력망 등 다른 spatio‑temporal 도메인에 대한 검증은 향후 과제로 남는다.

저자들은 이를 보완하기 위해 **pretrained 모델·knowledge distillation**을 활용한 향후 연구를 제안한다.[file:1][]

***

## 4. 일반화 성능 향상 관점에서의 분석

SimST는 여러 구성 요소가 **일반화(특히 distribution shift·데이터 sparsity)**에 유리한 메커니즘을 내포한다.

### 4.1 Node-based Batch Sampling의 일반화 효과

- 큰 batch size는 sharp minima에 수렴해 generalization gap을 늘린다는 기존 결과(Keskar et al. 2017)를 반영해, $B^\*$를 의도적으로 작게 설정.[file:1]
- 그래프 기반 배치는 동일 시간대의 모든 노드를 같은 배치에 넣어, 배치 간 상관이 높고 sample 다양성이 낮다.
- node‑based 배치는 서로 다른 노드·시간 윈도우가 섞여 들어가, gradient 노이즈·다양성을 확보해 검증 MAE 감소와 더 smooth한 학습 곡선을 보인다.[file:1]
- 특히 PeMSD4처럼 구조가 희소한 데이터에서 generalization 개선 효과가 더 크게 나타난다.[file:1]

요약하면, SimST의 중요한 일반화 요인은 “모델 구조”뿐 아니라 “batch 구성 전략”에 있다.

### 4.2 Global Correlation Embedding의 일반화 역할

- adjacency 기반 GNN은 **그래프 구조가 노이즈/불완전**할 때 오히려 성능이 악화될 수 있다.[file:1][]
- SimST의 센서 위치 임베딩은:
    - 원거리지만 기능적으로 유사한 노드(예: 모두 고속도로 교차점)를 embedding 공간에서 가깝게 두어, 구조 노이즈에 덜 민감한 표현을 학습.[file:1]
    - case study에서, embedding 유사도가 물리적 거리와 일관된 패턴(가까울수록 유사도↑)을 보이며, 이는 **데이터 기반으로 “부정확한 그래프”를 자동 보정하는 효과**를 시사.[file:1]
- 이는 센서 추가·삭제, 도로망 변경 등 환경 변화에도 비교적 robust한 spatial representation을 제공해, **도시 확장/센서 재배치 상황에서의 일반화 잠재력**을 보여준다.


### 4.3 로컬 근접 모듈과 overfitting 균형

- top‑ $k$ 이웃 수를 늘리면 초기에는 성능 향상(k=0→3) 후, 더 크면 오히려 악화(k=4)되는 overfitting 패턴이 관찰된다.[file:1]
- 이는 로컬 정보가 너무 많아질 경우, noise·중복을 학습하는 방향으로 치우칠 수 있음을 보여주며, SimST는 적당한 $k$를 통해 **복잡도–일반화의 균형점**을 찾을 수 있음을 시사한다.


### 4.4 도메인 전환·데이터 shift 측면의 잠재력

- SimST는 GNN 메시지 패싱에 의존하지 않기 때문에, road graph 정보가 빈약하거나 품질이 낮은 새로운 도시로 이식할 때 **“그래프 재설계” 비용을 줄이고** 빠르게 적응할 여지가 있다.[file:1][]
- FlashST(2024)와 같은 prompt‑tuning 기반 traffic prediction이 등장하면서, 대규모 pretraining된 spatio‑temporal 모델 + SimST 스타일의 구조 단순화/임베딩 조합은 **도시 간 zero‑shot/low‑shot generalization** 연구와 자연스럽게 연결될 수 있다.[][]

***

## 5. 2020년 이후 관련 최신 연구와의 비교 분석

SimST와 최근 연구들을 “GNN 필요성, 효율성, 일반화” 관점에서 정리하면 다음과 같다.

### 5.1 선택된 주요 논문들

| 논문 | 연도 | 핵심 아이디어 | GNN 의존성 |
| :-- | :-- | :-- | :-- |
| STGODE / STGNCDE | 2021–2022 | Neural ODE/CDE를 STGNN에 결합, 연속 시간 dynamics 모델링.[file:1][] | 강함 (GCN 기반) |
| SimST (본 논문) | 2023 | GNN‑free 로컬/글로벌 모듈 + node‑based sampling, 선형 복잡도.[file:1][] | 없음 (GNN 제거) |
| HD‑TTS 등 missing data용 STGNN | 2024 | 공간·시간 계층 downsampling으로 missing data에 robust한 STGNN.[] | 강함 |
| STSM (regions without data) | 2024 | 관측 없는 region을 위한 selective masking 기반 spatio‑temporal forecasting.[][] | 중간 (GRU+local neighbor) |
| SpecSTG | 2024 | spectral domain diffusion + fast spectral conv로 확률적 STG forecasting.[] | 강함 (spectral GNN) |
| FlashST | 2024 | pretraining + prompt‑tuning으로 distribution shift에 robust한 traffic prediction.[] |  |
| Efficient distillation for traffic prediction | 2025 | GNN teacher → MLP student distillation, 효율성과 성능 동시 확보.[] |  |

### 5.2 비교 관찰

1. GNN 중심 vs GNN‑free/경량화:
    - 2020–2022 STGODE/STGNCDE, 다양한 STGNN survey들은 GNN을 교통 예측의 “기본 모듈”로 전제.[][][]
    - SimST는 node‑wise MLP + 임베딩 구조만으로 STGNN와 동급 성능을 보이며, 이후 distillation 기반 “그래프‑less student” 연구들(Efficient Spatio‑Temporal Distillation 등)을 직접적으로 자극했다.[file:1][][]
2. 효율성:
    - STGODE/STGNCDE는 continuous‑time expressiveness를 얻는 대신, training/inference 시간이 SimST 대비 10–30× 이상 길다.[file:1]
    - SpecSTG는 spectral conv로 속도를 개선하려 하지만 여전히 그래프 연산에 의존.[]
    - SimST와 distillation 계열은 **“그래프 없이도, 혹은 teacher‑student 구조로, 실시간 요구를 만족”**하는 방향을 강조한다.[file:1][]
3. 일반화·distribution shift:
    - FlashST, STSM, HD‑TTS 등은 “데이터 결손, 새로운 지역, domain shift”에 초점을 맞추고, masking·prompt·hierarchical pooling으로 일반화를 개선.[][][]
    - SimST는 이들과 달리 **훈련 배치 구성(node‑based)**과 **embedding 기반 global 구조 학습**으로 일반화 robustness를 올리는 점이 특징이다.[file:1]
    - SimST 아이디어는, FlashST 같은 foundation 모델의 **경량한 adaptor**로 결합될 가능성이 크다(예: pretrain된 temporal backbone + SimST‑style spatial modules).[][]

***

## 6. 향후 연구에 미치는 영향과 고려할 점

### 6.1 영향

1. “GNN이 항상 필요하다”는 가정에 대한 재고:
    - SimST는 교통 예측에서 **GNN‑free 구조가 STGNN과 동급 성능**이라는 실증을 제공해, 이후 GNN distillation·graph‑less 네트워크 연구의 근거를 강화한다.[file:1][][]
2. 효율성 중심 설계 패러다임 강화:
    - TPS·파라미터 수·메모리 사용을 공동 평가 척도로 삼으며, 실시간 ITS/엣지 디바이스에서 deploy‑가능한 모델을 설계하는 흐름을 가속한다.[file:1][]
3. 학습 전략의 중요성 부각:
    - node‑based sampling이 성능에 미치는 영향을 체계적으로 보여줌으로써, **batch sampling을 모델 설계의 핵심 축**으로 인식하게 한다.[file:1]

### 6.2 앞으로 연구 시 고려할 점 (연구자 관점 제안)

1. GNN vs 비‑GNN 설계 선택 기준 정립
    - 도로망 규모, 그래프 품질, 레이턴시 요구, 하드웨어 자원에 따라:
        - 고품질 그래프·레이턴시 덜 민감 → STGNN/SpecSTG 등.
        - 대규모·노이즈 그래프·실시간 요구 → SimST/MLP‑student 계열이 유리.[file:1][][]
2. Pretraining + SimST 결합
    - FlashST 등 large spatio‑temporal foundation model을 temporal backbone으로 사용하고, SimST 스타일 로컬/글로벌 모듈을 plug‑in 하는 구조:
        - 도시 간 transfer, 비슷한 도메인(공기질, 전력망 등)으로의 zero‑shot 전이 검증.[][file:1]
3. Distillation·Teacher–Student 설계
    - 복잡한 STGNN (teacher)이 학습한 공간 관계를, SimST/MLP (student)에 distill:
        - Efficient Traffic Prediction Through Spatio‑Temporal Distillation와 유사한 방향에서, SimST의 node‑wise 구조를 student로 활용.[][file:1]
4. Graph 품질·missing data 환경에서의 검증
    - STSM, HD‑TTS처럼 데이터 결손/그래프 불완전성이 큰 시나리오에서 SimST‑style 임베딩과 GNN 기반 모델을 체계적으로 비교:
        - adjacency를 점진적으로 degrade하는 실험 설계.
        - 센서 추가/삭제, 도로 확장 등 real deployment 시나리오를 반영한 평가.[][][file:1]
5. 이론적 분석
    - SimST가 어떤 조건 하에서 GNN의 스펙트럴 필터링/메시지 패싱 효과를 근사할 수 있는지에 대한 이론 정식화:
        - ego‑graph + MLP가 구현 가능한 polynomial graph filter 클래스,
        - sensor embedding이 구조적 동형성을 어떻게 암묵적으로 모사하는지에 대한 분석.[file:1][]

***

위 구조를 바탕으로, 사용자의 연구 주제(예: distillation, foundation model, missing‑data robustness)에 맞게 SimST를 비교 축으로 삼아 설계·분석을 진행하면, **모델 단순성과 일반화 사이의 trade‑off를 정교하게 다루는 후속 연구**를 설계하는 데 도움이 될 것이다.[file:1][][]
<span style="display:none">[^1_10][^1_11][^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_6][^1_7][^1_8][^1_9]</span>

<div align="center">⁂</div>

[^1_1]: 2301.12603v1.pdf

[^1_2]: https://arxiv.org/abs/2301.12603

[^1_3]: https://arxiv.org/pdf/2301.12603.pdf

[^1_4]: https://arxiv.org/pdf/1709.04875.pdf

[^1_5]: https://www.semanticscholar.org/paper/Graph-Neural-Network-for-Traffic-Forecasting:-A-Jiang-Luo/bad3534cc797606d1fe3cb09713407783e77cac4

[^1_6]: https://link.springer.com/10.1007/s11042-023-17248-y

[^1_7]: https://ieeexplore.ieee.org/document/10336800/

[^1_8]: https://www.hindawi.com/journals/jat/2023/8962283/

[^1_9]: https://ieeexplore.ieee.org/document/10048915/

[^1_10]: https://www.semanticscholar.org/paper/20a9bb23cafb79259052f1f00fc34a677484ec55

[^1_11]: https://ieeexplore.ieee.org/document/10422690/

[^1_12]: https://dl.acm.org/doi/10.1145/3580305.3599890

[^1_13]: https://www.semanticscholar.org/paper/ecc79fb9325ecb619717f57e132f492b13a30444

[^1_14]: https://repositorio.banrep.gov.co/bitstream/handle/20.500.12134/10657/monetary-policy-january-2023.pdf

[^1_15]: http://arxiv.org/pdf/2401.08119.pdf

[^1_16]: https://arxiv.org/pdf/2405.17898.pdf

[^1_17]: https://arxiv.org/html/2501.10796v1

[^1_18]: https://arxiv.org/pdf/2311.08635.pdf

[^1_19]: http://arxiv.org/pdf/2403.16495.pdf

[^1_20]: https://www.mdpi.com/2079-9292/9/9/1474

[^1_21]: https://www.semanticscholar.org/paper/Do-We-Really-Need-Graph-Neural-Networks-for-Traffic-Liu-Liang/334de342e7c4823d7b91e6915089d6a16339f5f6

[^1_22]: https://arxiv.org/html/2412.09972v1

[^1_23]: https://arxiv.org/html/2512.17352v1

[^1_24]: https://arxiv.org/html/2402.10634v3

[^1_25]: https://arxiv.org/html/2511.14720v1

[^1_26]: https://arxiv.org/html/2509.18115v1

[^1_27]: https://arxiv.org/html/2511.05179v1

[^1_28]: https://arxiv.org/html/2401.00713v2

[^1_29]: https://arxiv.org/pdf/2201.05760.pdf

[^1_30]: https://arxiv.org/html/2410.22377v2

[^1_31]: https://arxiv.org/html/2408.06762v1

[^1_32]: https://arxiv.org/html/2501.10459v1

[^1_33]: https://openreview.net/forum?id=2ppuWD3dkie

[^1_34]: https://www.scribd.com/document/867206454/Do-We-Really-Need-Graph-Neural-Networks-for-Traffic-Forecasting

[^1_35]: https://arxiv.org/html/2401.10518v1

[^1_36]: https://arxiv.org/pdf/2101.11174.pdf

[^1_37]: https://openreview.net/pdf?id=2ppuWD3dkie

[^1_38]: https://openproceedings.org/2024/conf/edbt/paper-123.pdf

[^1_39]: https://ui.adsabs.harvard.edu/abs/2024SPIE13064E..0MG/abstract

[^1_40]: http://arxiv.org/abs/2301.12603

[^1_41]: https://www.sciencedirect.com/science/article/abs/pii/S0925231225010720

[^1_42]: https://ascelibrary.org/doi/10.1061/9780784483565.046

[^1_43]: https://www.nature.com/articles/s41598-024-78335-0

