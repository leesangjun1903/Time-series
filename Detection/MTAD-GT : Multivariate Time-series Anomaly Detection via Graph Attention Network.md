# Multivariate Time-series Anomaly Detection via Graph Attention Network

- MTAD-GAT은 다변량 시계열 이상 탐지에서 “변수 간 상관관계”와 “시간적 의존성”을 그래프 어텐션으로 명시적으로 학습해, 기존 방법의 잦은 false alarm 문제를 줄이는 것을 핵심 주장으로 한다.[^1_1][^1_2]
- 이를 위해 (1) feature‑oriented GAT, (2) time‑oriented GAT, (3) GRU, (4) 예측( forecasting ) + 재구성( reconstruction )을 합친 joint objective, (5) POT 기반 자동 임계값을 결합한 self‑supervised 프레임워크를 제안하고, NASA SMAP/MSL 및 자사 TSA 데이터셋에서 SOTA 대비 F1을 최대 9% 향상시킨다.[^1_2][^1_1]
- GAT attention score를 활용해 어떤 변수 간 상관이 깨졌는지 시각화함으로써, 이상 탐지뿐 아니라 root‑cause diagnosis(원인 분석)에도 유용함을 보인다.[^1_1]

***

## 2. 문제, 방법(수식), 구조, 성능·한계 (자세한 설명)

### 2.1 해결하고자 하는 문제

- 입력: 길이 $n$, 변수 수 $k$인 다변량 시계열 슬라이딩 윈도우

$$
\mathbf{x} \in \mathbb{R}^{n \times k}
$$

에 대해, 각 시점 $t$가 이상인지 여부 $y_t \in \{0,1\}$를 예측하는 문제이다.[^1_1]
- 기존 한계:
    - LSTM‑기반 예측/재구성 모델들은 변수 간 공간 상관관계를 암묵적으로만 모델링해, 동시적 스파이크처럼 “함께 움직이는” 패턴을 잘못 이상으로 보거나, 상관이 깨지는 경우를 포착하지 못한다.[^1_2][^1_1]
    - Reconstruction‑only / Forecast‑only 모델은 각각 “주기 깨짐” 또는 “확률적 변동”에 취약해, 시나리오에 따라 성능 편차가 크다.[^1_1]

MTAD‑GAT의 목표는

1) 변수 간 구조( feature graph )와 시간 축 구조( temporal graph )를 모두 GAT로 명시적으로 학습하고,
2) 예측‑오차와 재구성‑확률을 결합한 스코어로 강인한 이상 점수를 정의해, 다양한 환경에서 안정적인 성능과 해석 가능성을 확보하는 것이다.[^1_2][^1_1]

***

### 2.2 제안 방법: 수식 중심 정리

#### (1) 전처리

- min–max 정규화:

$$
\tilde{\mathbf{x}} = 
\frac{\mathbf{x} - \min(X_{\text{train}})}
     {\max(X_{\text{train}}) - \min(X_{\text{train}})}
$$

[^1_1]

- 학습 데이터에 대해 SR‑CNN 기반 univariate 이상 탐지로 의심 시점 값을 근방 정상 값으로 치환해, 훈련 시 outlier 영향 감소.[^1_1]


#### (2) 1‑D Convolution

- 각 변수별 1‑D CNN (kernel size 7)을 적용해 지역 패턴을 추출:

$$
\mathbf{h}^{(conv)} = \text{Conv1D}(\tilde{\mathbf{x}})
$$

여기서 $\mathbf{h}^{(conv)} \in \mathbb{R}^{n \times k}$.[^1_1]

#### (3) Graph Attention Layer (GAT) 기본식

노드 표현 $\mathbf{v}_i$ 에 대해, 단일‑헤드 GAT는 다음을 사용한다.[^1_2][^1_1]

- 이웃 $j$에 대한 비정규화 attention:

$$
e_{ij} = \text{LeakyReLU}\big(\mathbf{w}^\top [\mathbf{v}_i \oplus \mathbf{v}_j]\big)
$$

- 정규화 attention:

$$
\alpha_{ij} =
\frac{\exp(e_{ij})}{\sum_{l \in \mathcal{N}(i)} \exp(e_{il})}
$$

- 출력 노드 표현:

$$
\mathbf{h}_i = \sigma\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij}\mathbf{v}_j\right)
$$

여기서 $\sigma$는 비선형 함수(논문에서는 sigmoid 표기), $\oplus$는 concatenation, $\mathcal{N}(i)$는 이웃 집합이다.[^1_1]

#### (4) Feature‑oriented GAT

- 노드: 각 **변수(채널)** $i \in \{1,\dots,k\}$.
    - 노드 특징: 해당 변수의 길이 $n$ 시계열 $\mathbf{x}\_i = [x\_{0,i}, \dots, x_{n-1,i}] \in \mathbb{R}^n$.

[^1_1]
- 그래프: 완전 그래프 (모든 변수 쌍 연결).
- 출력: 변수별 representation 행렬 $\mathbf{H}^{(f)} \in \mathbb{R}^{k \times n}$.

[^1_1]

이 층은 CPU–Memory, 입력량–출력량, GC–처리량과 같은 “변수 간 인과/상관”을 학습해, 어떤 변수의 패턴이 normal일 때 다른 변수들이 어떻게 따라야 하는지를 포착한다.[^1_1]

#### (5) Time‑oriented GAT

- 노드: 각 **시점** $t \in \{0,\dots,n-1\}$.
    - 노드 특징: 시점 $t$에서의 $k$-차원 벡터 $\mathbf{x}_t \in \mathbb{R}^k$.[^1_1]
- 그래프: 슬라이딩 윈도우 내 시점들 간 완전 그래프.
- 출력: $\mathbf{H}^{(t)} \in \mathbb{R}^{n \times k}$.[^1_1]

이는 Transformer의 self‑attention과 유사하게, 단기 인접 시점뿐 아니라 멀리 떨어진 시점 간 장기 의존성도 직접 attention으로 모델링한다.[^1_2][^1_1]

#### (6) GAT + Conv Fusion, GRU

- feature‑GAT의 출력을 transpose하여 시점 기준으로 맞추면,
$\mathbf{H}^{(f)\top} \in \mathbb{R}^{n \times k}$.
- 세 소스를 concat:

$$
\mathbf{H}^{(cat)}_t = 
\big[
\mathbf{h}^{(conv)}_t \;\oplus\;
\mathbf{h}^{(f)\top}_t \;\oplus\;
\mathbf{h}^{(t)}_t
\big] \in \mathbb{R}^{3k}
$$

따라서 $\mathbf{H}^{(cat)} \in \mathbb{R}^{n \times 3k}$.[^1_1]

- 이를 GRU에 입력:

$$
\mathbf{h}^{(gru)}_t = \text{GRU}(\mathbf{H}^{(cat)}_{0:t})
\quad (\mathbf{h}^{(gru)}_t \in \mathbb{R}^{d_1})
$$

여기서 $d_1=300$.[^1_1]

#### (7) Forecasting‑based branch

- GRU 마지막 hidden state (또는 전체 시퀀스)를 바탕으로, **다음 시점** $\mathbf{x}_{n}$ 예측:

$$
\hat{\mathbf{x}}_{n} = f_{\text{fc}}\big(\mathbf{h}^{(gru)}_{0:n-1}\big) \in \mathbb{R}^k
$$

[^1_1]

- loss: RMSE (실 구현은 MSE 기반):

$$
\mathcal{L}_{\text{for}} =
\sqrt{\sum_{i=1}^{k} \big(x_{n,i} - \hat{x}_{n,i}\big)^2}
$$

[^1_1]

#### (8) Reconstruction‑based branch (VAE)

- Encoder $q_\phi(\mathbf{z}|\mathbf{x})$: 전체 윈도우 $\mathbf{x}$를 잠재 벡터 $\mathbf{z}\in\mathbb{R}^{d_3}$로 매핑.[^1_1]
- Decoder $p_\theta(\mathbf{x}|\mathbf{z})$: 잠재 공간에서 시계열을 재구성.
- Evidence Lower Bound 기반 VAE loss:

```math
\mathcal{L}_{\text{rec}} = 
- \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}
\left[
\log p_\theta(\mathbf{x}|\mathbf{z})
\right]
+
D_{\text{KL}}\big(q_\phi(\mathbf{z}|\mathbf{x}) \,\|\, p(\mathbf{z})\big)
```

[^1_2][^1_1]

#### (9) Joint optimization

$$
\mathcal{L} = \mathcal{L}_{\text{for}} + \mathcal{L}_{\text{rec}}
$$

두 브랜치를 동시에 업데이트해, “국소 단일‑시점 예측능력”과 “전역 시퀀스 분포 모델링”을 동시에 학습한다.[^1_1]

#### (10) Inference \& Anomaly score

테스트 시, 각 시점/변수 $i$에 대해

- 예측값 $\hat{x}_i$ 와 실제값 $x_i$의 제곱오차,
- 재구성 기반 “정상일 확률” $p_i$ (또는 reconstruction likelihood)
를 사용해 변수별 스코어 $s_i$를 정의한다.[^1_1]

논문이 제시한 최종 윈도우 수준 anomaly score:

```math
\text{score} = \sum_{i=1}^{k} s_i
=
\sum_{i=1}^{k}
\frac{( \hat{x}_i - x_i )^2
+ \gamma\,(1 - p_i)}{1 + \gamma}
```

여기서 $\gamma$는 예측‑오차와 재구성‑확률의 비중을 조절하는 하이퍼파라미터로, 실험에서는 $\gamma=0.8$이 최적.[^1_1]

시점별 score 시퀀스에 EVT 기반 Peak‑Over‑Threshold(POT)로 임계값을 자동 추정해 이상/정상을 결정한다.[^1_3][^1_1]

***

### 2.3 모델 구조 요약

전체 파이프라인은 다음과 같다.[^1_2][^1_1]

1. 입력 전처리: min–max 정규화 + SR‑기반 노이즈/이상 클리닝.
2. 1‑D CNN: 각 변수의 로컬 패턴 추출.
3. Feature‑GAT: 변수 그래프(완전 그래프) 상에서 변수 간 attention.
4. Time‑GAT: 시점 그래프(완전 그래프) 상에서 시점 간 attention.
5. Concatenation: Conv + 두 GAT 출력을 시점 기준으로 concat.
6. GRU: 장기 시퀀스 의존성 모델링.
7. Forecast head (FC): 다음 시점 다변량 값 예측.
8. VAE head: 전체 윈도우 재구성.
9. Joint training: 두 loss 합으로 학습.
10. Inference: 예측오차 + 재구성확률을 결합한 score → POT로 thresholding.

***

### 2.4 성능 향상 및 한계

#### 성능 향상

- 데이터셋: NASA SMAP, MSL, Microsoft TSA(프로덕션 Flink 모니터링 데이터).[^1_1]
- F1‑score:
    - SMAP: 0.9013로 OmniAnomaly(0.8434) 등 대비 약 1%p 이상 개선.[^1_1]
    - MSL: 0.9084로 OmniAnomaly(0.8989) 등 대비 소폭 우위.[^1_1]
    - TSA: 0.7975로 기존 최고 대비 F1 9%p 향상.[^1_1]
- 지연 허용(δ)별 F1: 첫 이상 발생 후 δ 시점 이내에 한 번이라도 탐지하면 성공으로 보는 segment‑level metric에서, MTAD‑GAT는 OmniAnomaly보다 일관되게 높은 F1을 기록하며, 특히 δ=10일 때 상대적 개선 폭이 13~54%로 큼.[^1_1]
- Ablation:
    - feature‑GAT 제거 시 평균 F1 약 3.2% 저하, TSA에서 특히 큰 감소 → 변수 간 관계 모델링이 실질적으로 성능에 기여.[^1_1]
    - time‑GAT 제거 시에도 2.5% 감소 → GRU만으로는 포착하기 어려운 비인접 시점 간 의존성을 attention이 보완.[^1_1]
    - forecast만 / reconstruction만 사용하는 경우 모두 joint보다 성능이 떨어지며, reconstruction‑only는 주기 붕괴와 같이 값 분포는 정상이나 패턴이 깨진 경우를 놓치고, forecast‑only는 stochastic noise에 민감.[^1_1]
- 진단 성능:
    - Top‑k root cause 후보에서 HitRate, NDCG@5가 높게 나타나, attention weight와 feature별 score를 통해 원인 후보 정렬이 가능함을 보임.[^1_1]


#### 한계

- 완전 그래프 가정: feature‑GAT, time‑GAT 모두 fully‑connected graph를 사용해, 변수/시점 수가 많아지면 $O(k^2)$, $O(n^2)$ 복잡도로 확장성이 떨어질 수 있다.[^1_1]
- 도메인 prior 미활용: 실제 시스템에는 이미 알려진 인과 구조(예: topology, 물리 모델)가 있지만, MTAD‑GAT는 전적으로 데이터로부터 학습하여, 적은 데이터나 domain constraint가 강한 환경에서 sub‑optimal할 수 있다.[^1_1]
- false positive 사례:
    - 드문 대규모 트래픽 피크처럼, 실제로는 정상적 workload 변화지만 과거에 거의 없었던 패턴이 나오면 anomaly로 탐지하는 경향이 있다.[^1_1]
- open‑set/generalization 관점: 모든 이상 유형을 포괄적으로 학습하지 못하고, 학습 시 관찰된 normal/“유사 이상” 분포에 상당 부분 의존하므로, 미지 유형의 이상(open‑set anomaly)에 대한 이론적 보장은 없다.[^1_4][^1_1]

***

## 3. “모델 일반화 성능 향상 가능성” 관련 논점

MTAD‑GAT 자체의 일반화 및 이후 연구에서 이를 확장하는 방향을, (1) 구조적 일반화, (2) 분포 변화·open‑set, (3) 학습·튜닝 안정성 측면에서 정리한다.

### 3.1 구조적 일반화: 그래프·어텐션 설계

1. **그래프 구조에 prior/학습 통합**
    - MTAD‑GAT는 fully‑connected 그래프에서 attention이 “유효한” 엣지를 soft하게 선택하도록 학습하지만, 도메인에서 주어진 topology(예: 네트워크 토폴로지, 공정 플로우, 물리 계통)를 반영하면 불필요한 edge를 줄이고, 스파스한 attention을 유도해 과적합을 완화할 수 있다.[^1_5][^1_1]
    - 이후 연구에서는 동적 그래프 학습과 attention을 결합해, 시간에 따라 변하는 inter‑variable 관계를 학습해 일반화를 개선하고 있다(CAN, MTGFlow, CGAD 등).[^1_6][^1_7][^1_8]
2. **다중 스케일·주파수 도메인 통합**
    - MTAD‑GAT의 time‑GAT는 윈도우 내부에서 한 스케일의 self‑attention만 사용한다.
    - TopoGDN, TFAD‑GAT, CATCH, PGMA 등은 multi‑scale temporal convolution, time‑frequency 특징, periodic graph를 결합해 다양한 주기·스케일의 패턴에 robust하도록 설계하여, 데이터셋 변화에도 성능 저하가 덜한 경향을 보고한다.[^1_9][^1_10][^1_11][^1_3]
3. **하이브리드 아키텍처**
    - TSA‑Net, GTAD, GRAN, GAT+Informer 기반 모델 등은 GAT + TCN/Informer/LSTM을 조합해 시계열 길이, 변수 수, 노이즈 특성이 다른 다양한 산업 데이터에서 안정적 성능을 보인다.[^1_12][^1_13][^1_14][^1_15]
    - 이는 MTAD‑GAT 구조를 보다 “모듈형”으로 재설계해, 도메인별로 temporal backbone(Conv, TCN, Transformer 등)을 교체할 여지를 시사한다.

### 3.2 분포 변화·Open‑set 일반화

1. **라벨 희소, 분포 이동 상황**
    - 실제 산업 환경에서는 정상 분포가 서서히 이동하고, 새로운 이상 유형이 등장한다.
    - MTGFlow는 dynamic graph + normalizing flow를 사용해 “density gap” 기반 이상 탐지를 수행함으로써, 레이블이 전혀 없는 환경에서도 다양한 anomaly type에 대해 상대적으로 robust한 성능을 보인다.[^1_7][^1_16]
    - MOSAD는 few‑shot labeled anomaly만 있는 open‑set TSAD를 정의하고, contrastive / deviation / generative heads를 결합해 unseen anomaly 클래스까지 탐지하는 일반화 프레임워크를 제시한다.[^1_17][^1_4]
2. **Self‑supervised / contrastive pretraining**
    - MTAD‑GAT는 예측/재구성이라는 self‑supervised signal을 이미 사용하지만, 최근 연구에서는 마스킹, permutation, 인과 방향 예측 등 다양한 pretext task를 병렬로 학습해 representation의 도메인 전이 성능을 높이는 경향이 있다(FMUAD, MEMTO 등).[^1_18][^1_19]
    - 이러한 multi‑task self‑supervision 위에 MTAD‑GAT의 GAT 구조를 얹으면, feature/temporal 관계 일반화가 강화될 가능성이 높다.

### 3.3 학습·튜닝 안정성 및 벤치마크 관점

- 논문은 $\gamma$에 대한 sensitivity가 낮고, 0.4~1.0 범위에서 항상 경쟁력 있는 성능을 보인다고 보고해 “튜닝 안정성” 측면의 강점을 보여준다.[^1_1]
- 하지만 mTSBench, MTAD toolkit 같은 대규모 벤치마크 연구에서는 모델 선택과 하이퍼파라미터 설정에 따라 동일 모델의 성능 편차가 상당하다는 점을 지적하며, MTAD‑GAT도 포함된 여러 모델을 공통 프로토콜로 비교한다.[^1_20][^1_21][^1_22]
- 일반화 성능을 진지하게 논하려면,
    - 더 다양한 도메인(의료, 금융, 네트워크 보안),
    - missing value, irregular sampling, high‑dimensional (>100 channel) 설정,
    - online/streaming 업데이트
까지 포함한 평가가 필요하며, 최근 benchmark 논문들이 그 방향으로 움직이고 있다.[^1_23][^1_24][^1_20]

***

## 4. 2020년 이후 관련 최신 연구 비교 분석

MTAD‑GAT 이후, 그래프·어텐션 기반 다변량 시계열 이상 탐지는 여러 방향으로 확장되었다. 주요 공개·오픈액세스 연구 몇 편을 중심으로 비교한다.

### 4.1 그래프+시계열 하이브리드 계열

| 모델 | 핵심 아이디어 | MTAD‑GAT 대비 특징 |
| :-- | :-- | :-- |
| GTAD (Graph+TCN, 2022)[^1_14] | TCN으로 temporal dependency, GNN으로 sensor correlation 학습 | GRU 대신 dilated TCN, 더 긴 temporal field, F1 >95% 보고 |
| GraphAD (2022)[^1_25] | entity‑wise graph를 학습해 KPI 간 구조를 반영 | feature‑graph를 정적으로가 아니라 데이터로부터 학습 |
| GRAN (Graph Recurrent Attention Network, 2024)[^1_13] | 개선된 GAT + LSTM 결합 | attention을 시계열 정보에 condition하여 가중치 할당 개선 |
| GAT+Informer (ICS, 2024)[^1_15] | GAT로 inter‑dependency, Informer로 long‑range forecast | MTAD‑GAT의 GRU를 long‑horizon Informer로 대체, 고차원 ICS에 강함 |

이 계열은 MTAD‑GAT의 “그래프+RNN” 틀을 유지하되, temporal backbone을 TCN·Informer로 대체해 긴 시계열과 고차원 데이터에서의 일반화를 개선한다.

### 4.2 GAT 확장 및 토폴로지 활용 계열

| 모델 | 요약 | 공통점/차이점 |
| :-- | :-- | :-- |
| TopoGDN (2024)[^1_26][^1_3] | multi‑scale temporal conv + topology‑aware augmented GAT | MTAD‑GAT의 dual GAT를 topology 정보로 강화, coarse attention 한계를 극복 |
| TFAD‑GAT (2024)[^1_9] | time‑frequency feature + dynamic GAT | 주파수 도메인의 변화를 함께 사용, 주기/비주기 패턴 generalization 개선 |
| CGAD (Entropy Causal Graph AD, 2024)[^1_8] | transfer entropy 기반 causal graph + GAT | 상관이 아닌 인과 구조를 반영해 spurious correlation 문제 완화 |

이들은 MTAD‑GAT가 fully‑connected GAT에 맡겼던 “구조 학습” 부분을 explicit topology, causal graph 등으로 보강해, 스케일과 도메인 변화에 더 잘 일반화되도록 한 것으로 볼 수 있다.

### 4.3 self‑supervised / open‑set / label‑free 계열

| 모델 | 설정 | 기여 |
| :-- | :-- | :-- |
| FMUAD (2022)[^1_19] | forecast‑based multi‑aspect unsupervised | spatial vs temporal anomaly 유형을 분리해 score 설계, 다양한 anomaly type에 robust |
| MTGFlow (2023–24)[^1_7][^1_16] | label‑free, dynamic graph + flow | “zero label” 환경에서 density‑based anomaly score, 여러 MTSAD SOTA 초과 |
| MEMTO (2023)[^1_18] | memory‑guided Transformer | latent/입력 space 양쪽에서 deviation score 사용, 재구성‑기반과 forecast‑기반의 일반화 결합 |
| MOSAD (open‑set TSAD, 2023–24)[^1_4][^1_17] | few‑shot labeled anomaly, unseen anomaly class | multi‑head network로 open‑set anomaly 대응, 일반화 이론 논의 포함 |

MTAD‑GAT의 joint objective 아이디어(예측+재구성)는 FMUAD, MEMTO, TranAD류 모델에서 변형·확장되어, 다양한 self‑supervised loss와 결합되고 있다.[^1_27][^1_18]

### 4.4 최근 리뷰·벤치마크

- Deep Learning TSAD Survey (2022), Online MTS Survey (2024) 는 MTAD‑GAT를 대표적인 GAT‑기반 MTSAD로 분류하며, “explicit inter‑variable modeling”이라는 장점을 강조하는 한편, high‑dimensional/online setting에 대한 확장 필요성을 지적한다.[^1_28][^1_23]
- MTAD toolkit, MTAD/mTSBench (2024–25) 는 MTAD‑GAT 포함 수십 개 모델을 통합 구현·비교해, 데이터셋·metric에 따라 ranking이 크게 바뀐다는 점을 보여주며 model‑selection/generalization 문제를 독립 주제로 제기한다.[^1_21][^1_22][^1_20]

***

## 5. 앞으로의 연구에 미치는 영향과 향후 고려사항

### 5.1 영향

1. **“변수=노드” 관점의 표준화**
    - MTAD‑GAT는 “각 univariate 시계열을 그래프의 노드로 보고, 상관관계를 attention으로 학습”하는 패러다임을 분명히 제시했고, 이후 대부분의 GNN‑기반 MTSAD 연구가 이 틀을 기본 출발점으로 삼는다.[^1_8][^1_3][^1_1]
2. **joint objective (forecast + reconstruction)의 실용성 입증**
    - 단일 loss로는 특정 anomaly type에 취약하다는 점을 실험적으로 보여주고, 두 paradigm을 결합한 joint loss가 실제 산업 데이터(TSA)에서 의미 있게 성능을 끌어올릴 수 있음을 입증했다.[^1_27][^1_1]
3. **해석 가능한 graph attention을 통한 진단**
    - feature‑GAT의 attention matrix를 root‑cause ranking에 활용하는 사례를 명시적으로 보여줌으로써, anomaly detection 모델을 “설명 가능한 진단 도구”로 사용하는 후속 연구(예: GTAD, TSA‑Net, Graph spatiotemporal process 등)에 영향을 주었다.[^1_14][^1_24][^1_1]
4. **산업 적용 및 제품화**
    - Microsoft Azure multivariate anomaly detection 서비스에 MTAD‑GAT가 탑재되면서, 연구‑산업 간 브리지 역할을 했고, 이후 여러 벤더/오픈소스에서 유사한 구조의 서비스를 제공하는 계기가 되었다.[^1_29][^1_21]

### 5.2 앞으로 연구 시 고려할 점 및 제안

연구자로서 MTAD‑GAT를 확장·비판적으로 계승하려면 다음 방향을 특히 의식하는 것이 좋다.

1. **그래프 구조 학습과 스케일 문제**
    - 변수 수가 수백~수천인 시나리오에서 fully‑connected GAT는 연산·메모리 측면에서 한계가 명확하다.
    - 스파스 그래프(Topology, causal graph, learned k‑NN), multi‑graph(상관, 인과, 물리 constraint 병렬) 구조를 사용해, MTAD‑GAT 스타일의 dual GAT를 스케일업하는 설계가 중요하다.[^1_11][^1_30][^1_8]
2. **도메인 prior와 causal inference 통합**
    - 단순 상관보다 인과 구조를 반영하면 distribution shift나 intervention(시스템 변경) 이후에도 더 잘 일반화할 수 있다. CGAD류 접근처럼 transfer entropy, Granger causality 등을 이용한 causal graph + GAT 조합은 유망한 후속 방향이다.[^1_8]
3. **Open‑set, continual, label‑scarce 환경**
    - 실제 운영 환경에서는 unseen anomaly type이 필연적이므로, MOSAD·MTGFlow처럼 few‑shot / zero‑shot open‑set 설정을 고려한 설계가 필요하다.[^1_4][^1_7]
    - MTAD‑GAT의 예측/재구성 joint objective를 유지하면서, memory module, prototype‑based clustering, online update를 결합해 continual TSAD로 확장하는 것도 자연스러운 연구 주제다.[^1_18][^1_5]
4. **언어·지식 모델과의 결합**
    - 최근 mTSBench 등에서는 LLM‑기반 multivariate anomaly 설명·진단 시도가 등장하고 있다.[^1_20]
    - MTAD‑GAT의 attention map, anomaly score, 메타데이터를 LLM의 입력으로 사용해, root‑cause hypothesis, remediation plan 등을 자연어로 생성하는 “time‑series + LLM” 하이브리드도 연구 가치가 크다.
5. **평가 프로토콜과 실제 비용 반영**
    - 논문은 segment‑level F1, delay‑aware F1를 사용했지만, 최근 연구는
        - false alarm cost,
        - miss cost,
        - investigation effort,
        - online latency
등을 반영한 composite metric과 salience를 제안한다.[^1_22][^1_20]
    - 향후 연구에서는 MTAD‑GAT 계열 모델들을 이런 비용 인식(metric‑aware) 학습/평가 틀에서 다시 재검토할 필요가 있다.

요약하면, MTAD‑GAT는 “그래프 어텐션 + joint forecast/reconstruction”이라는 강력한 설계 패턴을 제시했고, 이후 연구는 이를 더 큰 스케일, 더 어려운 데이터 조건(open‑set, missing, online), 더 풍부한 도메인 지식과 결합하는 방향으로 확장하고 있다.[^1_24][^1_7][^1_3][^1_1]
<span style="display:none">[^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48]</span>

<div align="center">⁂</div>

[^1_1]: 2009.02040v1.pdf

[^1_2]: https://arxiv.org/abs/2009.02040

[^1_3]: https://arxiv.org/html/2408.13082v1

[^1_4]: https://arxiv.org/pdf/2310.12294v2.pdf

[^1_5]: https://arxiv.org/html/2401.05800v1

[^1_6]: https://arxiv.org/pdf/2306.07114.pdf

[^1_7]: https://arxiv.org/pdf/2208.02108.pdf

[^1_8]: https://arxiv.org/html/2312.09478v2

[^1_9]: https://ieeexplore.ieee.org/document/10652754/

[^1_10]: https://arxiv.org/html/2410.12261v2

[^1_11]: https://arxiv.org/pdf/2509.17472.pdf

[^1_12]: https://www.mdpi.com/1424-8220/26/3/1062

[^1_13]: https://ieeexplore.ieee.org/document/10674110/

[^1_14]: https://pubmed.ncbi.nlm.nih.gov/35741480/

[^1_15]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10935277/

[^1_16]: https://arxiv.org/html/2312.11549v2

[^1_17]: https://arxiv.org/pdf/2310.12294.pdf

[^1_18]: http://arxiv.org/pdf/2312.02530.pdf

[^1_19]: https://arxiv.org/pdf/2201.04792.pdf

[^1_20]: https://arxiv.org/html/2506.21550v1

[^1_21]: https://github.com/OpsPAI/MTAD

[^1_22]: https://arxiv.org/abs/2401.06175

[^1_23]: https://www.sciencedirect.com/science/article/pii/S0952197624014817

[^1_24]: https://www.sciencedirect.com/science/article/pii/S1566253524000332

[^1_25]: https://arxiv.org/pdf/2205.11139.pdf

[^1_26]: https://dl.acm.org/doi/10.1145/3627673.3679614

[^1_27]: https://peerj.com/articles/cs-2172/

[^1_28]: https://arxiv.org/abs/2211.05244

[^1_29]: https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/introducing-multivariate-anomaly-detection/2260679

[^1_30]: https://www.arxiv.org/pdf/2509.17235.pdf

[^1_31]: https://link.springer.com/10.1007/978-3-030-91445-5_12

[^1_32]: http://www.inderscience.com/link.php?id=10058553

[^1_33]: https://link.springer.com/10.1007/s10489-025-06650-8

[^1_34]: https://www.mdpi.com/1424-8220/24/5/1522

[^1_35]: https://linkinghub.elsevier.com/retrieve/pii/S0141938225003087

[^1_36]: https://link.springer.com/10.1007/978-981-95-6203-9_10

[^1_37]: https://arxiv.org/pdf/2108.03585.pdf

[^1_38]: https://arxiv.org/pdf/2503.11255.pdf

[^1_39]: https://dl.acm.org/doi/pdf/10.1145/3611643.3613896

[^1_40]: https://arxiv.org/html/2512.13735v1

[^1_41]: https://arxiv.org/html/2310.12294v1

[^1_42]: https://arxiv.org/html/2509.04449v1

[^1_43]: https://iy322.tistory.com/60

[^1_44]: https://ai-scholar.tech/en/articles/time-series/mv_ts_graph

[^1_45]: https://github.com/mangushev/mtad-gat

[^1_46]: https://dl.acm.org/doi/10.1145/3696410.3714941

[^1_47]: https://ieeexplore.ieee.org/document/10024505/

[^1_48]: https://github.com/dheiver/Multivariate-Time-series-Anomaly-

