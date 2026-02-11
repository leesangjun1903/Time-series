# Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting

### 1. 핵심 주장 및 주요 기여 (간결 요약)

- **핵심 주장**: 멀티변량 시계열(MTS) 예측에서 최근 복잡한 Spatial-Temporal GNN(STGNN)들은 성능 향상이 크지 않은 데 비해 지나치게 복잡하다. 근본적인 병목은 “공간·시간 축에서의 샘플 비식별성(indistinguishability)”이며, 이를 풀어주기만 하면 단순한 MLP만으로도 SOTA STGNN을 능가할 수 있다는 것이 논문의 핵심 주장이다.[^1_1][^1_2][^1_3]
- **주요 기여**

1. **공간·시간 비식별성 개념 제시**: 서로 다른 센서나 서로 다른 시점에서 역사 구간은 매우 유사하지만 미래 구간은 다른 샘플 쌍이 많으며, 단순 회귀 모델은 이를 구분하지 못한다는 점을 체계적으로 분석.
2. **STID 모델 제안**: 각 시계열(노드)과 시각(하루 내 시간, 요일)에 대해 **학습 가능한 ID 임베딩**을 붙인 후, 단순 MLP로 예측하는 **Spatial-Temporal Identity (STID)**라는 매우 간단한 베이스라인을 제안.[^1_3][^1_4][^1_1]
3. **성능·효율 동시 달성**: STGCN, DCRNN, Graph WaveNet, MTGNN, ST-Norm 등 다양한 STGNN 대비, 교통·전력 데이터셋(PEMS04/07/08/BAY, Electricity)에서 **대부분의 horizon에서 최상 성능**과 **최고 수준의 학습 속도**를 동시에 달성.
4. **연구 방향 전환의 메시지**: “그래프+시퀀스”라는 정형 STGNN 디자인에서 벗어나, **샘플을 구분해 주는 정보(정체성, 주기성 등)를 얼마나 잘 제공하는가**가 핵심이라는 인사이트를 제시하여 이후 단순 선형/MLP 모델 열풍(DLinear·Spatiotemporal-Linear·SimpleTM 등)과도 맥이 닿는 흐름을 형성.[^1_5][^1_6][^1_7]

***

## 2. 논문의 상세 내용

### 2.1 해결하고자 하는 문제: 공간·시간 비식별성

#### 2.1.1 문제 설정

멀티변량 시계열 데이터는 다음과 같이 표현된다.[^1_1]

$$
\mathbf{X} \in \mathbb{R}^{T \times N},
$$

여기서 $T$는 전체 타임스텝 수, $N$은 변수(센서) 개수이다. 예측 문제는 과거 $P$ 시점의 값으로부터 앞 $F$ 시점의 값을 예측하는 것:

$$
\mathbf{X}_{t-P:t} \in \mathbb{R}^{P \times N} \;\longrightarrow\; \mathbf{Y}_{t:t+F} \in \mathbb{R}^{F \times N}.
$$

각 변수 $i$에 대해 한 샘플은

$$
\mathbf{X}^i_{t-P:t} \in \mathbb{R}^P,\quad \mathbf{Y}^i_{t:t+F} \in \mathbb{R}^F
$$

로 표현된다.

#### 2.1.2 공간·시간 비식별성의 정의

논문이 지적하는 핵심 병목은 다음 두 가지 “비식별성”이다.[^1_3][^1_1]

1. **공간(spatial) 비식별성**
    - 서로 다른 센서 $i \neq j$에서 같은 시각 창 $W_1$을 보면,
$\mathbf{X}^i_{t-P:t} \approx \mathbf{X}^j_{t-P:t}$ 이지만
$\mathbf{Y}^i_{t:t+F} \not\approx \mathbf{Y}^j_{t:t+F}$ 인 경우가 많다.
    - 단순 MLP 회귀는 입력이 거의 같으면 출력도 비슷하게 내보내기 때문에, **센서 간 차이를 반영하기 어렵다**.
2. **시간(temporal) 비식별성**
    - 같은 센서 $i$에서도 서로 다른 시점 창 $W_2, W_3$에서
$\mathbf{X}^i_{t_2-P:t_2} \approx \mathbf{X}^i_{t_3-P:t_3}$ 이지만
$\mathbf{Y}^i_{t_2:t_2+F} \not\approx \mathbf{Y}^i_{t_3:t_3+F}$ 가 될 수 있다.
    - 특히 교통·전력 데이터의 **일/주기적 패턴** 때문에, 비슷한 과거 패턴이라도 요일/시간대에 따라 미래가 크게 달라진다.

STGNN의 GCN 모듈은 노드 구조를 통해 사실상 **“노드 ID”를 주입**하기 때문에 공간 비식별성을 어느 정도 해소한다는 분석이 선행 연구에서 제시되었고, 본 논문은 여기에 더해 **시간 축에서도 유사한 문제가 존재**하며, 이를 정체성(Identity) 임베딩으로 직접 해결하자고 제안한다.[^1_8][^1_1][^1_3]

***

### 2.2 제안 방법: STID (수식 포함)

#### 2.2.1 공간·시간 ID 정의

논문은 다음과 같은 학습 가능한 임베딩 행렬을 정의한다.[^1_1][^1_3]

- 공간 ID(센서 ID):

$$
E \in \mathbb{R}^{N \times D}, \quad E_i \in \mathbb{R}^D
$$

- 하루 내 시간(Time-of-Day) ID:

$$
T^{\text{TiD}} \in \mathbb{R}^{N_d \times D}, \quad T^{\text{TiD}}_{t} \in \mathbb{R}^D
$$

- $N_d$: 하루를 샘플링 주기에 따라 나눈 슬롯 수 (예: 5분 간격이면 288)
- 요일(Day-of-Week) ID:

$$
T^{\text{DiW}} \in \mathbb{R}^{N_w \times D}, \quad T^{\text{DiW}}_{t} \in \mathbb{R}^D, \quad N_w = 7
$$

모든 행렬은 랜덤 초기화된 **학습 파라미터**이며, 동일한 시간대/요일에 대해 공유된다.

#### 2.2.2 임베딩 및 피처 결합

각 샘플 $\mathbf{X}^i_{t-P:t}$에 대해, 먼저 과거 구간을 단일 벡터로 임베딩한다.[^1_3][^1_1]

$$
\mathbf{H}^i_t = \mathrm{FC}_{\text{embedding}}(\mathbf{X}^i_{t-P:t}) \in \mathbb{R}^D.
$$

이후 공간·시간 ID를 단순 **concatenation**으로 붙여 하나의 표현을 만든다:

$$
\mathbf{Z}^i_t = \mathbf{H}^i_t \,\Vert\, E_i \,\Vert\, T^{\text{TiD}}_t \,\Vert\, T^{\text{DiW}}_t \in \mathbb{R}^{4D},
$$

여기서 $\Vert$는 벡터 연결(concatenation)을 의미한다.

요약하면,

- $\mathbf{H}^i_t$: “내용(content)” – 과거 패턴
- $E_i$: “위치(where)” – 센서/변수 ID
- $T^{\text{TiD}}_t, T^{\text{DiW}}_t$: “언제(when)” – 시각·요일 ID
를 하나의 벡터로 묶어 **샘플이 어떤 센서에서 어떤 시간대·요일에 관측된 것인지**를 명시적으로 표현한다.


#### 2.2.3 MLP 인코더와 회귀 헤드

이후 $\mathbf{Z}^i_t$에 대해 $L$개의 MLP 레이어를 residual 연결로 쌓는다.[^1_1]

$$
(\mathbf{Z}^i_t)^{l+1} = \mathrm{FC}_2^{(l)}\bigl( \sigma(\mathrm{FC}_1^{(l)}((\mathbf{Z}^i_t)^l)) \bigr) + (\mathbf{Z}^i_t)^l,\quad l=0,\dots,L-1,
$$

여기서

- $(\mathbf{Z}^i_t)^0 = \mathbf{Z}^i_t$,
- $\sigma$는 비선형 활성함수(ReLU 등),
- 각 $\mathrm{FC}_1^{(l)}, \mathrm{FC}_2^{(l)}$는 일반적인 fully connected 층이다.

마지막으로, 회귀층을 통해 길이 $F$의 미래를 출력한다:

$$
\hat{\mathbf{Y}}^i_{t:t+F} = \mathrm{FC}_{\text{reg}}\bigl( (\mathbf{Z}^i_t)^L \bigr) \in \mathbb{R}^F.
$$

#### 2.2.4 학습 목적 함수

손실함수로는 **Mean Absolute Error(MAE)**를 사용한다.[^1_1]

$$
\mathcal{L}(\hat{\mathbf{Y}}, \mathbf{Y}) = 
\frac{1}{NF} \sum_{i=1}^{N} \sum_{j=1}^{F} \left| \hat{Y}^i_{t+j-1} - Y^i_{t+j-1} \right|.
$$

Adam 최적화를 사용하여 모든 FC 층과 ID 임베딩 파라미터를 end-to-end로 학습한다.

***

### 2.3 모델 구조 (아키텍처 관점에서 정리)

STID의 전체 파이프라인은 다음과 같이 요약할 수 있다.[^1_3][^1_1]

1. **입력 슬라이딩 윈도우**
    - 각 센서 $i$, 시점 $t$에 대해 길이 $P$의 과거 구간 $\mathbf{X}^i_{t-P:t}$를 샘플로 생성.
2. **임베딩 레이어**
    - 1D 벡터 $\mathbf{X}^i_{t-P:t}$를 FC로 투사하여 $\mathbf{H}^i_t \in \mathbb{R}^D$ 생성.
3. **ID 부착(Spatial-Temporal Identity Attachment)**
    - 센서 ID $E_i$, 시간대 ID $T^{\text{TiD}}_t$, 요일 ID $T^{\text{DiW}}_t$를 추출해 $\mathbf{H}^i_t$와 연결하여 $\mathbf{Z}^i_t \in \mathbb{R}^{4D}$ 구성.
4. **다층 MLP 인코더**
    - residual MLP 블록을 $L$단 쌓아 표현을 정제.
    - GCN/Conv/RNN/Attention 없이 **오직 FC + 비선형 + residual**로 구성된 극단적으로 단순한 구조.
5. **회귀 헤드**
    - 마지막 표현을 FC에 통과시켜 길이 $F$의 미래 예측 시퀀스 $\hat{\mathbf{Y}}^i_{t:t+F}$ 출력.
6. **학습 및 추론 효율**
    - 그래프 구조(인접행렬)나 복잡한 time–space fusion 연산이 없으므로, GPU·CPU 모두에서 매우 빠른 학습 속도와 메모리 효율을 가진다.[^1_4][^1_1]

***

### 2.4 성능 향상과 한계

#### 2.4.1 성능 향상

실험은 5개 MTS 벤치마크에서 수행된다.[^1_1]

- **교통(traffic)**: PEMS04, PEMS07, PEMS08, PEMS-BAY (5분 간격, 수백 개 센서, 수개월 길이)
- **전력(Electricity)**: 336개 시계열, 1시간 간격

비교 대상에는 VAR, HI(History Inertia), LSTM, DCRNN, STGCN, Graph WaveNet, AGCRN, StemGNN, GMAN, MTGNN, ST-Norm 등 **전형적인 STGNN 및 RNN/CNN 계열** 대부분이 포함된다.[^1_1]

주요 결과는 다음과 같다.[^1_9][^1_3][^1_1]

- **정확도**
    - 모든 데이터셋과 horizon(@3, @6, @12, avg)에 대해 **대부분의 MAE/RMSE/MAPE에서 1위**를 기록.
    - 그래프 구조를 사용하지 않음에도, 그래프 기반 모델(DCRNN, STGCN, Graph WaveNet, GMAN 등)을 전반적으로 상회.
- **효율성**
    - 에폭당 학습 시간이 기존 STGNN 대비 **수 배 이상 빠름** (예: PEMS04에서 DCRNN은 ~95초/epoch, STID는 ~5초/epoch 수준).[^1_1]
    - Electricity처럼 입력 길이 $P$가 커지는 경우에도 연산량 증가가 완만하여, **장기 히스토리에서도 실용적인 속도**를 유지.


#### 2.4.2 한계

논문 자체 및 후속 연구 관점에서 볼 수 있는 한계는 다음과 같다.

1. **그래프 구조를 완전히 무시**
    - STID는 그래프를 사용하지 않기 때문에, **공간 구조가 매우 중요한 도메인**(예: 도로망의 세밀한 topology, 전력망 구조)에선 STGNN 기반이 더 유리할 수 있다.
2. **ID 임베딩의 해석 가능성/전이성 부족**
    - 학습된 $E, T^{\text{TiD}}, T^{\text{DiW}}$는 t-SNE 시각화에서 의미 있는 클러스터를 형성하지만,[^1_1]
**새로운 센서/새로운 시간대**에 대해 어떻게 일반화될지, 전이 학습이 가능한지는 명시적으로 다루지 않는다.
3. **복잡한 비선형 동역학에 대한 표현력**
    - MLP만 사용하므로, **비선형 장기 의존성(long-range dependency)**을 transformer나 dilated CNN만큼 유연하게 포착하지 못할 수 있다. 다만 실험 데이터셋 범위에서는 충분했다는 것이 이 논문의 포인트.
4. **이상치, 분포 변화 등 현실적 요인 미고려**
    - 데이터 분포 shift, 이상치, 결측 등은 별도 처리 없이 학습하며, 이 부분은 이후 ST-SSDL, STTS-EAD 같은 self-supervised/이상치-통합 모델에서 확장되고 있다.[^1_10][^1_8]

***

## 3. STID와 “일반화 성능” 관점에서의 의미

질문에서 특히 강조한 **“모델의 일반화 성능 향상 가능성”**을 STID 관점에서 정리하면 다음과 같다.

### 3.1 왜 ID 임베딩이 일반화를 돕는가?

1. **샘플 조건부 분포를 명시적으로 구분**
    - 기본 MLP는 $p(\mathbf{Y} \mid \mathbf{X})$를 하나의 거대한 혼합 분포로 보며,
센서·시간대별로 다른 조건부 $p(\mathbf{Y} \mid \mathbf{X}, i, t)$를 구분하지 못한다.
    - STID는 $(i, t_{\text{timeofday}}, t_{\text{weekday}})$를 임베딩으로 인코딩하여,
사실상 **여러 “모드”를 하나의 네트워크 안에서 공유 파라미터로 학습**하게 만든다.
    - 이는 **조건부 분포의 multi-modality를 설명 가능한 방식으로 분해**하는 것과 유사하여,
**노이즈보다 구조적인 차이에 더 잘 적응**하게 만든다.
2. **파라미터 공유와 데이터 효율성**
    - 센서별로 독립 모델을 학습하는 대신,
**하나의 MLP가 모든 센서·시간대를 공유**하면서도 ID 임베딩으로 context를 구분한다.
    - 데이터가 적은 센서나 드문 시간대도, **유사한 임베딩 클러스터에 속한 다른 샘플로부터 통계력을 공유**할 수 있어,
소규모 데이터 영역의 일반화를 개선한다.
3. **임베딩 공간의 구조적 일반화**
    - 실험에서 공간 임베딩 $E$는 지리적으로 인접한 센서끼리 클러스터를 형성하고,[^1_1]
시간 임베딩 $T^{\text{TiD}}, T^{\text{DiW}}$는 일/주기 구조(근접 시간대, 평일 vs 주말)를 잘 반영한다.
    - 이는 모델이 단순 “ID lookup table”이 아니라,
**연속적인 표현 공간에서 근접한 노드/시간대를 비슷하게 다루는 inductive bias**를 획득했음을 의미하며,
도메인 내 분포 변화에 대한 **로컬 일반화(local generalization)**를 촉진한다.

### 3.2 일반화 측면의 잠재적 리스크

1. **ID에 과도하게 의존하는 “암기(memorization)” 위험**
    - 극단적으로는, $\mathbf{X}$보다 $E, T^{\text{TiD}}, T^{\text{DiW}}$에 더 의존해서
특정 센서·요일 패턴을 “암기”할 수 있다.
    - 이 경우, **새로운 기간(예: 신규 월, 휴일 패턴 변화)** 또는
**새로운 센서(그래프 확장)**에 대한 일반화는 제한적일 수 있다.
2. **Out-of-distribution(OOD) 시나리오에 대한 한계**
    - 훈련에 등장하지 않은 요일/시간대 조합, 혹은 완전히 다른 주기성을 가진 데이터셋에 전이할 때
ID 임베딩은 더 이상 의미 있는 prior가 아닐 수 있다.
    - 이는 이후 DLinear·Spatiotemporal-Linear·MixLinear처럼
**데이터의 주기/트렌드 구조 자체를 더 강하게 모델링하는 선형/주파수 기반 모델**들에서 개선 방향으로 다루어진다.[^1_11][^1_7][^1_5]
3. **그래프/물리 구조의 미활용으로 인한 한계**
    - 공간 ID만으로는 **물리적 이웃 관계나 네트워크 흐름 제약**(예: 상류·하류 도로 구조)을 직접적으로 표현하지 못한다.
    - 이 때문에 ST-Hyper, LMHR-Enhanced STGNN, ASTCRF 등의 후속 연구는
**STID의 “정체성 임베딩” 아이디어 + 동적인 그래프/고차 의존성**을 결합하는 방향을 취한다.[^1_12][^1_13][^1_14]

***

## 4. 2020년 이후 관련 최신 연구 비교 분석 및 향후 연구에의 영향

### 4.1 주요 관련 연구(2020년 이후) – 간단 리스트

(요청하신 대로, 개별 논문은 “제목 – 저자 – 링크 – 1–2문장 요약” 형식으로 정리합니다. 모두 open-access 소스만 사용.)

- **Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting (STID)**
Zezhi Shao et al., CIKM 2022.[^1_2][^1_3][^1_1]
https://arxiv.org/abs/2208.05233
→ 공간·시간 비식별성을 핵심 병목으로 규정하고, 학습 가능한 공간/시간 ID + MLP만으로 STGNN을 능가하는 간단 베이스라인을 제안.
- **ST-Norm: Spatial and Temporal Normalization for Multi-Variate Time Series Forecasting**
Jinliang Deng et al., KDD 2021 (STID가 인용).[^1_8]
→ 공간·시간별 분포 차이를 정규화(normalization)로 보정하여, 간단한 구조로도 STGNN 수준 성능을 보임. “정규화/identity 등 데이터 재표현이 핵심”이라는 점에서 STID와 철학 공유.
- **Are Transformers Effective for Time Series Forecasting? (DLinear)**
Zeng et al., AAAI 2023.[^1_15][^1_7]
https://arxiv.org/pdf/2205.13504.pdf
→ 기존 LTSF Transformer들이 과대평가되었음을 보이고, 단순 선형(Linear/DLinear/NLinear) 모델이 다수 벤치마크에서 SOTA를 달성한다는 결과를 제시. STID처럼 “단순 모델 vs 복잡 모델” 논쟁에 큰 영향을 줌.
- **A Time Series is Worth 64 Words: Long-term Forecasting with Transformers (PatchTST)**
Nie et al., ICLR 2023.[^1_16][^1_17][^1_18][^1_19]
https://arxiv.org/abs/2211.14730
→ 채널 독립 + patch 토큰화를 사용하는 Transformer로, LTSF에서 기존 Transformer 계열 대비 큰 개선을 달성. STID의 “ID/채널 구분”과 유사하게, **채널 단위 분리와 shared parameter**를 강조.
- **Spatiotemporal-Linear: Towards Universal Multivariate Time Series Forecasting**
Zhu et al., arXiv 2023.[^1_5]
https://arxiv.org/abs/2312.14869
→ 복잡한 deep 모델 대신, 단일 선형 구조로 공간·시간 정보를 동시에 반영하는 “보편적” MTS 예측기를 제안. STID의 단순성 철학을 더 나아가 선형 수준으로 밀어붙인 사례.
- **An Analysis of Linear Time Series Forecasting Models**
Li et al., arXiv 2024.[^1_15]
https://arxiv.org/pdf/2403.14587.pdf
→ DLinear류 선형 모델들의 장단점을 체계적으로 분석. STID와 함께 “간단한 베이스라인의 재평가” 트렌드를 공고히 함.
- **A Simple Baseline for Multivariate Time Series Forecasting (SimpleTM)**
OpenReview preprint.[^1_6]
https://openreview.net/pdf/e7ccd35be296940a46939659136fb42d3cdc1fb1.pdf
→ 신호 처리 기반 토크나이제이션 + 얕은 attention으로 구성된 또 다른 “simple yet effective” 모델을 제안, 여러 벤치마크에서 SOTA에 근접하거나 능가.
- **Modeling Multivariate Time Series Correlation with Variate Embedding (VE)**
arXiv 2024.[^1_20]
https://arxiv.org/abs/2409.06169
→ 변수(채널) 간 상관을 학습 가능한 **variate embedding**으로 표현하는 모델. STID의 공간 ID와 아이디어적으로 매우 유사하며, 다양한 데이터셋에서 강한 성능과 상관 구조 학습을 보여줌.
- **Interpretable Spatial Identity Neural Network-based Epidemic Prediction (ISID)**
Hao et al., Sci. Rep. 2023.[^1_21]
https://www.nature.com/articles/s41598-023-45177-1 (open access)
→ STID의 공간 ID 아이디어를 전염병 예측에 적용해 간결한 구조와 높은 해석 가능성을 얻음. STID 아이디어의 실제 응용·해석 가능성 확장을 보여주는 사례.
- **Spatiotemporal-Linear, MixLinear, vLinear, OLinear 등 경량 선형 계열**
MixLinear (arXiv 2024), vLinear·OLinear (arXiv 2023–2025) 등.[^1_22][^1_23][^1_11]
→ 시간·주파수 도메인을 활용하거나 직교 변환을 사용하는 등 다양한 방식으로 **극단적으로 경량이면서도 높게 일반화되는 선형/준선형 모델**을 제안.
- **STD-PLM: Understanding Both Spatial and Temporal Properties of Spatial-Temporal Data with PLM**
arXiv 2024.[^1_24]
→ 대형 PLM을 재프로그램하여 ST 데이터의 공간·시간 토큰을 동시에 이해하도록 하는 방식. STID의 “spatial/temporal identity를 토큰으로 본다”는 관점을 거대 모델까지 확장하는 흐름.
- **ST-SSDL, STTS-EAD 등 self-supervised/이상치 통합 ST 학습**
ST-SSDL (arXiv 2025), STTS-EAD (arXiv 2025).[^1_10][^1_8]
→ 단순 예측 손실에 더해 deviation·contrastive/self-supervised loss를 추가해 **분포 변화·이상치에 강인한 일반화**를 지향.


### 4.2 STID와 최신 연구 흐름의 비교·분석

#### 4.2.1 “단순 모델의 재발견” 축

- STID, DLinear, Spatiotemporal-Linear, SimpleTM, MixLinear, vLinear, OLinear 등은 모두
**복잡한 attention·GCN·RNN 없이도 SOTA에 근접하거나 능가**함을 보이는 공통점이 있다.[^1_23][^1_7][^1_11][^1_22][^1_6][^1_5][^1_1]
- 차이는 다음과 같이 볼 수 있다.

| 모델 계열 | 핵심 아이디어 | STID와의 관계 |
| :-- | :-- | :-- |
| STID | 공간/시간 ID + MLP | **정체성(Identity) 주입**의 효용을 최초로 강하게 주장 |
| DLinear | 트렌드/계절성 분해 + 선형 | 구조적 분해를 강조, ID 대신 series level의 통계 구조에 초점 |
| Spatiotemporal-Linear | 선형 변환 하나로 ST 동시 처리 | STID의 “그래프 없이도 ST 정보 표현 가능”을 선형 형태로 일반화 |
| SimpleTM | 얕은 attention + 신호처리 기반 토큰 | 복잡도를 크게 줄이면서도 attention의 장점 유지 |
| MixLinear/vLinear/OLinear | 시간·주파수·직교 변환을 활용한 극경량 선형 | **일반화·효율성 극대화**를 위해 파라미터 수를 극단적으로 줄임 |

이 흐름 속에서 STID는

- “**샘플을 구분하기 위한 최소한의 조건(context)만 잘 줘도, 복잡한 구조 없이 강력한 예측이 가능하다**”
- “그래프나 transformer 도입 전에, 일단 ID/정규화/분해 같은 단순한 재표현을 먼저 해보라”

는 메시지를 가장 이른 시점(2022)부터 강하게 던진 논문 중 하나로 볼 수 있다.

#### 4.2.2 “정체성/임베딩 기반 ST 표현” 축

- STID의 공간 ID $E$, 시간 ID $T^{\text{TiD}}, T^{\text{DiW}}$는
**“어느 노드/어느 시점인지”를 나타내는 토큰**과 같다.
- 이후 연구에서는 유사한 아이디어를 다양한 방식으로 확장한다.

예시:

- **Variate Embedding (VE)**: 각 변수에 대한 learnable embedding을 도입해, multivariate 상관 구조를 캡쳐.[^1_20]
→ STID의 $E$와 거의 동일한 착상을, 보다 일반적 MTS 모델에 적용.
- **ISID**: 전염병 예측에서 **공간 ID만 사용**하여, 간결한 구조와 해석 가능성(예: region 간 감염 관계)을 동시에 달성.[^1_21]
- **STD-PLM/STD-Reprogramming류**: 공간·시간 토큰화를 통해 PLM이 ST 데이터를 이해하도록 재프로그램.[^1_25][^1_24]
→ STID의 “ID를 토큰으로 본다”는 관점이 foundation model 수준으로 확장된 사례.

이러한 계열과 비교하면, STID는

- **그래프·Transformer 없이도**,
- **정체성 임베딩만으로도 상당한 spatial–temporal 구조를 끌어낼 수 있다**는 **lower bound**를 제시한 셈이며,
이는 이후 임베딩/토큰 설계 연구에 강한 기준점으로 작용한다.


#### 4.2.3 “복잡 STGNN/Transformer vs Simple Baseline” 축의 재조정

- STID는 STGNN을 완전히 대체한다고 주장하지는 않지만,
**같은 벤치마크에서 간단한 STID조차 이기지 못하는 복잡 STGNN은 설계 재검토가 필요하다**는 강한 메시지를 던진다.[^1_26][^1_1]
- 이후 STNet, ST-Hyper, ASTCRF, LMHR-Enhanced STGNN 등은
단순히 GCN+RNN을 쌓는 대신,
    - 동적 그래프 학습,
    - hypergraph를 통한 고차 spatial-temporal 스케일,
    - long-term history representation
등을 도입하여, **복잡성 증가가 실제로 STID·DLinear 등 simple baseline을 의미 있게 넘어서도록** 설계되고 있다.[^1_13][^1_14][^1_12][^1_26]
- Transformer 계열에서도 PatchTST, Gateformer 등은
“채널 독립성, 패치 기반 attention, gating” 등을 통해
**local/long-range 패턴을 모두 잡되, 파라미터 공유·효율성을 유지**하는 방향으로 진화하고 있다.[^1_27][^1_17][^1_16]

이 전체 흐름을 보면, STID는

- “단순 baseline vs 복잡 모델” 논쟁에서 **baseline 쪽의 수준을 끌어올려 준 논문**
- 이후 복잡 모델 설계 시에도 **STID/DLinear/PatchTST 수준의 strong baseline을 반드시 넘도록 설계해야 한다**는 사실상의 커뮤니티 표준을 형성하는 데 기여했다고 볼 수 있다.

***

## 5. 앞으로의 연구에 미치는 영향과 연구 시 고려할 점

### 5.1 향후 연구에 미치는 영향

1. **강력한 베이스라인으로서의 STID**
    - 교통·전력·역학 등 spatial-temporal MTS가 등장하는 도메인에서,
STID는 **“그래프 없이도 이 정도는 나와야 한다”는 기준선**을 제공한다.
    - 이는 새로운 STGNN/Transformer 모델 제안 시,
**STID·DLinear·PatchTST와의 비교가 사실상 필수**가 되었음을 의미한다.[^1_7][^1_19][^1_9]
2. **ID/정규화/분해 기반 전처리·표현 설계의 중요성 부각**
    - STID, ST-Norm, DLinear, Spatiotemporal-Linear, VE 등은
공통적으로 **데이터의 단순한 재표현(Identity, Normalization, Decomposition, Embedding)이 모델 구조 못지않게 중요**함을 보여준다.[^1_7][^1_8][^1_20][^1_5]
    - 이는 앞으로도
        - 도메인 특화 ID(예: 도로 타입, 구간 길이, 도시 구획 등)
        - 계절성/트렌드 분해
        - 통계적 정규화
등의 설계가 핵심 연구 주제가 될 것임을 시사한다.
3. **해석 가능성과 경량화의 교차점**
    - ISID처럼, STID 계열 모델은 구조가 간단하고 ID 임베딩이 명시적이어서
**해석 가능성(어떤 센서/시간대가 어떤 영향?)**을 얻기 용이하다.[^1_21]
    - 동시에 파라미터 수와 연산량이 작아 **엣지 디바이스·실시간 예측**에도 적합하다.
    - 이는 **설명 가능하고 경량인 ST 예측 모델**이라는 방향성에 힘을 실어준다.

### 5.2 앞으로 연구 시 고려할 점 (연구자 관점의 체크리스트)

연구/모델 설계를 하실 때 특히 고려할 만한 포인트를 정리하면 다음과 같다.

1. **반드시 강력한 단순 baseline부터 구축**
    - 새 STGNN/Transformer를 설계하기 전에,
        - STID (공간·시간 ID + MLP),
        - DLinear/Spatiotemporal-Linear (선형 분해/주파수 기반),
        - PatchTST(Simple Patch Transformer)
등을 **동일 전처리·평가 설정에서 재현**해 보는 것이 바람직하다.[^1_16][^1_7][^1_1]
    - 제안 모델이 이들 대비 얼마나 이득을 주는지 **정량/정성 모두에서 설득력 있게 제시**해야 한다.
2. **정체성(ID) 설계의 일반화 가능성 평가**
    - STID처럼 ID를 도입할 때,
        - 새로운 센서가 추가되는 경우,
        - 다른 도시/국가로 이식하는 경우,
        - 달라진 주기(예: 팬데믹·정책 변화)의 등장 등
에서 **ID가 얼마나 재사용·전이 가능한지**를 별도로 평가할 필요가 있다.
    - 예: 새 노드에 대한 cold-start를 위해
        - 초기화 전략(평균/근접 노드 임베딩 상속),
        - meta-learning·hypernetwork 기반 ID 생성 등.
3. **그래프·토폴로지 정보와의 결합**
    - STID는 그래프를 버리고 ID만 사용하지만,
네트워크 구조가 중요한 도메인에선
        - STID-style ID + 동적 그래프 학습 (ASTCRF, LMHR 등),[^1_14][^1_12]
        - Hypergraph 기반 multi-scale ST 모형 (ST-Hyper)[^1_13]
처럼 **ID와 그래프를 함께 사용하는 방향**이 유망하다.
    - 연구 설계 시,
        - ID만으로 충분한지,
        - 그래프가 실제로 추가 정보를 주는지
를 ablation으로 분명히 보여주는 것이 좋다.
4. **OOD, 이상치, 분포 변화에 대한 강인성**
    - ST-SSDL, STTS-EAD 등은 self-supervised deviation learning·이상치 탐지와 예측을 결합하여
**분포 변화에도 잘 버티는 ST 모델**을 추구한다.[^1_8][^1_10]
    - STID류 simple baseline 위에
        - contrastive loss,
        - deviation-aware loss,
        - anomaly-aware training
을 올려 OOD에 대한 일반화를 개선하는 것이 유의미한 연구 방향이다.
5. **벤치마크·평가 세트의 다양화와 공정성**
    - STID가 쓰는 교통/전력 데이터셋(PEMS, Electricity)은 이제 사실상 표준 벤치마크가 되었지만,[^1_9][^1_1]
        - 기후(Weather), 에너지 부문 다른 데이터,
        - 금융·주식, 헬스케어, IoT 센서 네트워크 등
**도메인 다양성**을 확대하여 평가하는 것이 필요하다.
    - 또한, DLinear 논쟁에서 드러났듯,
        - 전처리(정규화),
        - window/horizon 설정,
        - metric 계산 방식
등에 따라 결과가 크게 달라질 수 있으므로,
재현 가능한 스크립트와 코드 공개(GitHub/BasicTS 등)를 함께 제공하는 것이 점점 필수에 가깝다.[^1_4][^1_7]

***

요약하면, 이 논문은 **“공간·시간 정체성(ID)이라는 매우 단순한 신호만으로도 MTS 예측의 핵심 병목을 상당 부분 해결할 수 있다”**는 점을 명쾌하게 보여준 작업입니다. 이후 DLinear·Spatiotemporal-Linear·PatchTST·VE·ISID·ASTCRF 등 다양한 연구들이 **단순/경량 모델과 정체성·임베딩 기반 표현의 조합**을 탐색하고 있으며, 앞으로도

- 강력한 simple baseline 구축,
- 정체성/정규화/분해 기반 재표현 설계,
- 그래프·self-supervision·OOD 대응 기법과의 결합

이 멀티변량 시계열 예측 연구의 핵심 축이 될 가능성이 매우 높습니다.
<span style="display:none">[^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46]</span>

<div align="center">⁂</div>

[^1_1]: https://arxiv.org/abs/2208.05233

[^1_2]: https://arxiv.org/abs/2208.05233v1

[^1_3]: https://openreview.net/pdf/60a1ed0119f5f7af41e7b817e8511416b8c3bc41.pdf

[^1_4]: https://github.com/GestaltCogTeam/STID

[^1_5]: https://arxiv.org/html/2312.14869v1

[^1_6]: https://openreview.net/pdf/e7ccd35be296940a46939659136fb42d3cdc1fb1.pdf

[^1_7]: https://github.com/honeywell21/DLinear

[^1_8]: https://arxiv.org/html/2510.04908v1

[^1_9]: https://paperswithcode.com/paper/spatial-temporal-identity-a-simple-yet

[^1_10]: https://arxiv.org/pdf/2501.07814.pdf

[^1_11]: https://arxiv.org/html/2410.02081v1

[^1_12]: https://arxiv.org/html/2505.14737v1

[^1_13]: https://dl.acm.org/doi/10.1145/3746252.3761281

[^1_14]: https://dl.acm.org/doi/10.1145/3675165

[^1_15]: https://arxiv.org/pdf/2403.14587.pdf

[^1_16]: https://arxiv.org/abs/2211.14730

[^1_17]: https://arxiv.org/pdf/2211.14730.pdf

[^1_18]: https://huggingface.co/docs/transformers/model_doc/patchtst

[^1_19]: https://github.com/yuqinie98/PatchTST

[^1_20]: https://arxiv.org/html/2409.06169v1

[^1_21]: https://www.nature.com/articles/s41598-023-45177-1

[^1_22]: https://arxiv.org/html/2601.13768v1

[^1_23]: https://arxiv.org/html/2505.08550v1

[^1_24]: http://arxiv.org/pdf/2407.09096.pdf

[^1_25]: https://arxiv.org/pdf/2507.11558.pdf

[^1_26]: https://arxiv.org/pdf/2206.09113.pdf

[^1_27]: https://arxiv.org/html/2505.00307v2

[^1_28]: 2208.05233v2.pdf

[^1_29]: https://arxiv.org/html/2501.08620v4

[^1_30]: https://arxiv.org/html/2501.08620v1

[^1_31]: https://arxiv.org/html/2505.11625v1

[^1_32]: https://dl.acm.org/doi/10.1145/3511808.3557702

[^1_33]: https://ieeexplore.ieee.org/document/11228734/

[^1_34]: https://ieeexplore.ieee.org/document/11228192/

[^1_35]: https://ieeexplore.ieee.org/document/10888419/

[^1_36]: https://linkinghub.elsevier.com/retrieve/pii/S0020025525007807

[^1_37]: https://linkinghub.elsevier.com/retrieve/pii/S1051200425004828

[^1_38]: https://www.mdpi.com/1424-8220/24/14/4473

[^1_39]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11670686/

[^1_40]: https://arxiv.org/pdf/2302.01701.pdf

[^1_41]: https://arxiv.org/pdf/2109.12218.pdf

[^1_42]: https://huggingface.co/blog/autoformer

[^1_43]: https://www.datasciencewithmarco.com/blog/patchtst-a-breakthrough-in-time-series-forecasting

[^1_44]: https://letter-night.tistory.com/450

[^1_45]: https://openreview.net/forum?id=PARkZPsb9x

[^1_46]: https://secundo.tistory.com/107

