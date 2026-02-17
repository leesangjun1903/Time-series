# LightCTS: A Lightweight Framework for Correlated Time Series Forecasting

## 1. 핵심 주장 및 주요 기여

LightCTS는 **상관된 시계열(Correlated Time Series, CTS) 예측을 위한 경량화 딥러닝 프레임워크**로, 기존 모델들이 정확도 개선이 정체된 반면 계산 복잡도만 증가하는 문제를 해결합니다. 이 논문의 핵심 주장은 **리소스 제약 환경(예: MCU)에서 배포 가능한 경량 모델로도 최신 모델과 동등한 정확도를 달성할 수 있다**는 것입니다.[^1_1]

주요 기여는 다음과 같습니다:[^1_1]

- **L-TCN (Light Temporal Convolutional Network)**: 그룹화 전략을 통한 경량 시간 연산자
- **GL-Former (GlobalLocal TransFormer)**: 전역 및 지역 공간 상관관계를 모두 포착하는 경량 공간 연산자
- **Plain Stacking 아키텍처**: 기존 교대 적층 대신 시간/공간 연산자를 순차적으로 배치
- **Last-shot Compression**: 시간적 특징의 마지막 타임스텝만 유지하여 계산량 대폭 감소


## 2. 문제 정의 및 제안 방법

### 해결하고자 하는 문제

**문제 1: 계산 복잡도 증가와 정확도 정체**
AutoCTS가 GwNet 대비 MAE를 0.06 mph만 개선했지만, 수백 GPU 시간이 필요하고 CO₂ 배출량이 증가합니다.[^1_1]

**문제 2: 엣지 디바이스 배포 불가능**
STM32F4 MCU는 3MB 메모리만 제공하지만, GwNet 같은 모델은 배포할 수 없습니다.[^1_1]

**문제 3: 기존 경량화 기법의 부적용성**
컴퓨터 비전의 2D/3D 컨볼루션 경량화 기법은 CTS의 1D 시간 컨볼루션과 그래프 컨볼루션에 직접 적용 불가능합니다.[^1_1]

### CTS 예측 문제 정식화

**단일 스텝 예측(Single-step Forecasting):**

$$
\hat{X}_{t+P+Q} \leftarrow \text{SF}(X_{t+1}, \ldots, X_{t+P})
$$

**다중 스텝 예측(Multi-step Forecasting):**

$$
\{\hat{X}_{t+P+1}, \ldots, \hat{X}_{t+P+Q}\} \leftarrow \text{MF}(X_{t+1}, \ldots, X_{t+P})
$$

여기서 $X \in \mathbb{R}^{N \times T \times F}$는 $N$개 시계열, $T$개 타임스텝, $F$개 특징을 가진 CTS입니다.[^1_1]

### 제안 방법론

#### 2.1 L-TCN (Light Temporal Convolutional Network)

**Dilated Causal Convolution (DCC):**

$$
H'[i; t; d] = \sum_{k=0}^{K-1} \left( H[i; t - \delta \times k; :] \cdot W_d[k; :] \right)
$$

**Shuffled Group TCN:**

$$
\text{SGTCN}(H \mid G_T) = \text{concat}\left(\{\text{TCN}(H_j \mid \delta, K)\}_{j=1}^{G_T}\right)
$$

**L-TCN Layer with Gating:**

$$
\text{L-TCN}(H) = \tanh(\text{SGTCN}_o(H \mid G_T)) \odot \sigma(\text{SGTCN}_g(H \mid G_T))
$$

시간 복잡도가 $O(\frac{D^2}{G_T} \cdot N \cdot P)$로 표준 TCN의 $\frac{1}{G_T}$ 배로 감소합니다.[^1_1]

#### 2.2 Last-shot Compression

각 L-TCN 레이어의 마지막 타임스텝 특징만 유지합니다:

$$
O_b = H_b[:, P-1, :]
$$

$$
H = \sum_{b=1}^{L_T} O_b
$$

**Squeeze-and-Excitation (SE) Module:**

$$
H_T = H \cdot \sigma(W_{s2} \cdot \text{ReLU}(W_{s1} \cdot H^\circ))
$$

여기서 $H^\circ = \text{GlobalAvgPool}(H) \in \mathbb{R}^D$입니다.[^1_1]

#### 2.3 GL-Former (GlobalLocal TransFormer)

**Multi-Head Attention:**

$$
\text{MHA}(H_{\text{PE}}) = \text{concat}\left(\{\text{head}_i(H_{\text{PE}})\}_{i=1}^h\right)
$$

$$
\text{head}_i(H_{\text{PE}}) = \text{softmax}(H_I) \cdot V_i
$$

$$
H_I = (Q_i \cdot K_i^T) / \sqrt{D/h}
$$

**Local Attention with Mask:**

$$
H_I = \text{mask}\left((Q_i \cdot K_i^T) / \sqrt{D/h}, M\right)
$$

마스크 함수는 인접 행렬 $M$을 이용해 관련 노드 쌍만 유지합니다.[^1_1]

**Light MHA (L-MHA):**

$$
\text{L-MHA}(H_{\text{PE}}) = \text{concat}\left(\{\text{MHA}(H_{\text{PE}}^j)\}_{j=1}^{G_M}\right)
$$

복잡도가 $\frac{1}{G_M}$ 배로 감소합니다.[^1_1]

#### 2.4 손실 함수

$$
\mathcal{L} = \text{MAE}(\hat{Y}, Y) = \frac{1}{N \times L} \sum_{i=1}^N \sum_{j=1}^L |\hat{Y}_{ij} - Y_{ij}|
$$

## 3. 모델 구조

### 전체 아키텍처

LightCTS는 **Plain Stacking 패턴**을 채택합니다:[^1_1]

1. **Embedding Module**: 원시 CTS 데이터를 잠재 표현으로 변환 (1개 CNN 레이어)
2. **T-Operator Module**: $L_T$개 L-TCN 레이어 (예: $L_T=4$, dilation rates $[1,2,4,8]$)
3. **Last-shot Compression**: $N \times P \times D \rightarrow N \times D$ 차원 축소
4. **S-Operator Module**: $L_S$개 GL-Former attention 블록 (global ↔ local 교대)
5. **Aggregation \& Output**: 2개 fully-connected 레이어

$$
\hat{Y} = \text{ReLU}((H_S + H_T) \cdot W_1^o + b_1^o) \cdot W_2^o + b_2^o
$$

### 복잡도 분석

| 구성 요소 | 시간 복잡도 | 공간 복잡도 |
| :-- | :-- | :-- |
| 표준 TCN | $O(D^2 \cdot N \cdot P)$ | $O(D^2)$ |
| L-TCN | $O(\frac{D^2}{G_T} \cdot N \cdot P)$ | $O(\frac{D^2}{G_T})$ |
| 표준 Transformer S-op | $O(D \cdot N \cdot P \cdot (N+D))$ | $O(D^2)$ |
| GL-Former | $O(\frac{D \cdot N \cdot (N+D)}{G_M})$ | $O(\frac{D^2}{G_M})$ |

Last-shot compression으로 S-operator 입력이 $N \times P \times D$에서 $N \times D$로 감소하여 시간 복잡도가 $\frac{1}{P}$ 배 감소합니다.[^1_1]

## 4. 성능 향상 및 한계

### 성능 향상

**다중 스텝 예측 (PEMS08 데이터셋):**

- LightCTS: MAE 14.63, RMSE 23.49, MAPE 9.43%
- AutoCts (SOTA): MAE 14.82, RMSE 23.64, MAPE 9.51%
- **FLOPs**: AutoCts 808M → LightCTS 70M (**11.5배 감소**)
- **Parameters**: AutoCts 366K → LightCTS 177K (**2.07배 감소**)
- **Latency**: AutoCts 3.7s → LightCTS 0.4s (**9.25배 개선**)[^1_1]

**메모리 제약 환경 (3MB):**

- LightCTS: MAE 14.63, Latency 0.4s
- GwNet: MAE 17.40 (19% 성능 저하), Latency 1.0s
- AutoCts/AgCrn: 배포 불가능[^1_1]

**단일 스텝 예측 (Solar-Energy):**

- 3rd step RRSE: LightCTS 0.1714 vs AutoCts 0.1750
- **FLOPs**: AutoCts 2237M → LightCTS 169M (**13.2배 감소**)[^1_1]


### 한계점

1. **하이퍼파라미터 튜닝의 복잡성**: $G_T$, $G_M$, $G_F$, $D$, $L_S$ 등 여러 하이퍼파라미터 조정 필요[^1_1]
2. **극도로 작은 임베딩 크기에서의 성능 저하**: 2MB 메모리 제약 시 MAE가 16.70으로 증가 (14% 성능 저하)[^1_1]
3. **도메인별 인접 행렬 의존성**: GL-Former의 local attention이 사전 정의된 공간 정보(인접 행렬) 필요[^1_1]
4. **장기 예측 한계**: Last-shot compression이 매우 긴 시계열에서 정보 손실 가능성[^1_1]

## 5. 일반화 성능 향상 가능성

### 5.1 구조적 일반화 메커니즘

**Cross-Period Sparse Forecasting**
L-TCN의 dilated convolution은 다양한 주기성을 자동으로 포착하여 다른 데이터셋에 일반화됩니다. 예를 들어, dilation rates $[1,2,4,8]$은 5분 간격 교통 데이터와 10분 간격 태양광 데이터 모두에 효과적입니다.[^1_1]

**Adaptive Graph Learning**
GL-Former는 데이터 기반 인접 행렬 학습으로 도메인별 공간 패턴을 자동 발견합니다. 실험 결과, 학습된 인접 행렬이 도메인별 상호작용 동역학을 포착했습니다.[^1_1]

### 5.2 실증적 일반화 증거

LightCTS는 **6개 벤치마크 데이터셋**에서 일관된 성능을 보였습니다:[^1_1]


| 데이터셋 | 도메인 | 노드 수 (N) | LightCTS 순위 |
| :-- | :-- | :-- | :-- |
| PEMS04 | 교통 흐름 | 307 | 1st (MAE/RMSE) |
| PEMS08 | 교통 흐름 | 170 | 1st (MAE/RMSE) |
| METR-LA | 교통 속도 | 207 | 1st (MAE) |
| PEMS-BAY | 교통 속도 | 325 | 1st (MAE) |
| Solar-Energy | 태양광 발전 | 137 | 1st (RRSE/CORR) |
| Electricity | 전력 소비 | 321 | 1st (대부분 메트릭) |

**Transfer Learning 가능성**
Plain stacking 구조는 T-operator와 S-operator를 분리하여 **모듈별 전이 학습**이 가능합니다. 예: 교통 데이터로 학습한 L-TCN을 에너지 데이터에 재사용.[^1_1]

### 5.3 일반화 개선 전략

**1. Knowledge Distillation 확장**
LightCTS★ 버전은 Tafd (Temporal-Attentive Feature Distillation)와 Caad (Channel-wise Adaptive Attention Distillation)를 추가하여 초저자원 환경에서 일반화 향상.[^1_2][^1_3]

**2. Multi-Domain Pre-training**
다양한 도메인 데이터로 사전 학습 후 fine-tuning하면 새로운 도메인에 빠른 적응 가능. Time Series Foundation Models (TSFMs) 패러다임과 결합 가능.[^1_4]

**3. Probabilistic Forecasting**
현재 결정론적 예측을 확률적 예측으로 확장하여 불확실성 정량화 시 일반화 향상.[^1_5]

## 6. 향후 연구 방향 및 영향

### 6.1 연구에 미치는 영향

**1. 경량화 연구의 새 방향 제시**
LightCTS는 "경량화 = 정확도 손실"이라는 통념을 깨고, **구조적 최적화로 정확도 유지 가능**함을 증명했습니다. 이는 Mobile-friendly AI의 새로운 패러다임을 제시합니다.[^1_6][^1_1]

**2. 엣지 컴퓨팅 활성화**
3MB MCU에서도 배포 가능한 설계는 IoT 센서 네트워크의 온디바이스 예측을 현실화했습니다. 풍력 터빈, 스마트 시티 센서 등에 직접 적용 가능합니다.[^1_1]

**3. 그린 AI 기여**
계산량 11.5배 감소는 CO₂ 배출 감소로 이어져 환경 친화적 AI 연구를 촉진합니다.[^1_1]

### 6.2 향후 연구 시 고려할 점

#### 고려사항 1: 동적 그래프 구조 학습

**문제**: 현재 GL-Former는 정적 인접 행렬을 사용하여 시간에 따른 공간 관계 변화를 포착하지 못합니다.[^1_1]

**해결 방향**:

- **Dynamic Adjacency Learning**: 각 타임스텝마다 인접 행렬을 업데이트하는 메커니즘 도입.[^1_5]
- **Temporal Graph Networks (TGNs)**: 연속 시간 그래프 표현 학습 통합.[^1_7]


#### 고려사항 2: Transformer 기반 모델과의 통합

**문제**: LightCTS는 Transformer의 self-attention을 공간 차원에만 적용하지만, 최신 모델들은 시간-공간 통합 attention을 사용합니다.[^1_8][^1_4]

**해결 방향**:

- **PatchTST 통합**: 시계열을 패치로 분할하여 시간적 self-attention 적용 후 GL-Former로 공간 모델링.[^1_4]
- **CITRAS 방식**: Covariate-informed attention으로 외부 변수 통합.[^1_8]


#### 고려사항 3: Foundation Model로의 확장

**문제**: LightCTS는 특정 데이터셋에서 학습하지만, 최근 연구는 다중 도메인 사전 학습을 강조합니다.[^1_9][^1_4]

**해결 방향**:

- **Multi-Domain Pre-training**: 교통, 에너지, 기상 데이터를 혼합하여 사전 학습.
- **TFMAdapter 방식**: 경량 어댑터로 foundation model을 task-specific하게 조정.[^1_4]

$$
\text{Adapter Output} = \text{LightCTS}_{\text{pretrained}} + \alpha \cdot \text{FC}(\text{Covariates})
$$

#### 고려사항 4: 장기 예측 성능 개선

**문제**: Last-shot compression이 horizon이 매우 긴 경우 (예: 720 steps) 성능 저하 가능성.[^1_5][^1_1]

**해결 방향**:

- **Hierarchical Compression**: 다중 해상도 temporal feature 유지.[^1_10]
- **Residual Stacking**: RS-GLinear 방식으로 잔차 연결 강화.[^1_11]

$$
\hat{Y}_{\text{long}} = \text{GL-Former}(H_T^{\text{short}}) + \text{Residual}(H_T^{\text{long}})
$$

#### 고려사항 5: 불확실성 정량화

**문제**: 결정론적 예측은 신뢰 구간을 제공하지 않아 실제 응용에서 제한적입니다.[^1_5]

**해결 방향**:

- **Monte Carlo Dropout**: 추론 시 dropout 유지하여 확률 분포 추정.[^1_12]
- **Gaussian Process Module**: 구조화된 노이즈 모델링으로 불확실성 학습.[^1_10]

$$
p(\hat{Y} \mid X) = \mathcal{N}(\mu_{\text{LightCTS}}(X), \sigma^2_{\text{GP}}(X))
$$

#### 고려사항 6: 자동 하이퍼파라미터 최적화

**문제**: $G_T$, $G_M$, $D$ 등 수동 튜닝이 필요합니다.[^1_1]

**해결 방향**:

- **Multi-objective Bayesian Optimization**: 정확도와 FLOPs를 동시 최적화하는 Pareto front 탐색.
- **AutoML 통합**: Neural Architecture Search (NAS)로 데이터셋별 최적 구조 자동 발견.[^1_13]


## 7. 2020년 이후 관련 최신 연구 비교

### 7.1 경량화 시계열 예측 모델 (2023-2025)

#### SparseTSF (2024)[^1_14]

- **핵심**: Cross-Period Sparse Forecasting으로 **1K 파라미터**만 사용
- **비교**: LightCTS (177K) 대비 극도 경량화, 하지만 공간 상관관계 무시
- **차별점**: LightCTS는 CTS (다변량)를 다루지만 SparseTSF는 단변량 특화


#### Lite-STGNN (2025)[^1_5]

- **핵심**: Trend-seasonal decomposition + Top-K adjacency learning
- **성능**: 720 step 장기 예측에서 SOTA, Transformer 대비 훈련 속도 빠름
- **한계**: LightCTS와 유사한 접근이지만 Last-shot compression 부재


#### RS-GLinear (2025)[^1_11]

- **핵심**: Residual-Stacked Gaussian Linear 아키텍처
- **장점**: 금융/역학 데이터에서 Transformer 대비 견고성 향상
- **제한**: 그래프 구조 미활용, 순수 linear 기반


### 7.2 Transformer 기반 SOTA 모델 (2023-2025)

#### CITRAS (2025)[^1_8]

- **핵심**: Covariate-informed cross-variate attention
- **기여**: 외부 변수(날씨, 휴일 등)를 attention에 통합
- **LightCTS 통합 가능성**: GL-Former에 covariate attention 추가

$$
\text{Attention}(Q, K, V, C) = \text{softmax}\left(\frac{QK^T + f(C)}{\sqrt{d_k}}\right) V
$$

#### iTransformer (2025)[^1_15][^1_16]

- **핵심**: 변수 간 관계를 명시적으로 모델링하는 inverted attention
- **성능**: 기상 예측에서 MedianAbsE 1.21 달성
- **단점**: 계산량 여전히 높음 (LightCTS 대비 5-10배)


#### PatchTST (2023-2025)[^1_15][^1_4]

- **핵심**: 시계열을 패치로 분할하여 attention 길이 $\lceil N/p \rceil$로 감소
- **전이 학습**: 한 데이터셋에서 학습 후 다른 데이터셋에 일반화
- **LightCTS 융합**: L-TCN을 patch 단위로 적용 가능


### 7.3 공간-시간 그래프 신경망 (2024-2025)

#### ST-GWNN (2024)[^1_7]

- **핵심**: Graph Wavelet Neural Network으로 지역적 특징 학습
- **장점**: Spectral CNN 대비 계산 복잡도 감소
- **비교**: GL-Former의 local attention과 유사하지만 wavelet 기반


#### STM-Graph (2025)[^1_17]

- **핵심**: 원시 spatio-temporal 데이터를 GNN 그래프로 자동 변환하는 프레임워크
- **OpenStreetMap 통합**: 실제 도시 특징 반영
- **LightCTS 활용**: STM-Graph로 전처리 → LightCTS로 예측


### 7.4 Foundation Models (2024-2025)

#### TFMAdapter (2025)[^1_4]

- **핵심**: Time Series Foundation Model에 lightweight adapter 추가로 covariate 활용
- **패러다임**: Pretrain on diverse domains → Adapt to specific task
- **LightCTS 통합**:
    - TSFM을 teacher model로 사용하여 LightCTS knowledge distillation
    - LightCTS를 adapter로 활용하여 TSFM의 covariate 처리 강화


#### MOMENT / Sundial (2024-2025)[^1_9]

- **핵심**: T5 인코더 기반 대규모 사전 학습
- **한계**: 추론 비용 높음, 엣지 배포 불가
- **보완 관계**: LightCTS가 MOMENT의 경량 배포 버전 역할 가능


### 7.5 비교 종합표

| 모델 | 연도 | FLOPs (M) | Params (K) | 주요 기법 | 공간 모델링 | 엣지 배포 |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| LightCTS | 2023 | **70** | 177 | Plain stacking, Last-shot | ✓ (GL-Former) | ✓ |
| AutoCTS | 2021 | 808 | 366 | NAS | ✓ (GCN) | ✗ |
| SparseTSF | 2024 | **~30** | **1** | Cross-period sparse | ✗ | ✓ |
| Lite-STGNN | 2025 | **~100** | **~50** | Top-K adjacency | ✓ (Top-K GNN) | ✓ |
| iTransformer | 2025 | ~500 | ~400 | Inverted attention | ✓ (Cross-variate) | ✗ |
| CITRAS | 2025 | ~600 | ~500 | Covariate-informed | ✓ (Smoothed attention) | ✗ |
| PatchTST | 2023 | ~300 | ~250 | Patching | ✗ (단변량) | △ |

### 7.6 최신 트렌드와 LightCTS의 위치

**트렌드 1: Foundation Model + Lightweight Adapter**
최신 연구는 대규모 사전 학습과 경량 배포를 분리합니다. LightCTS는 adapter 역할로 활용 가능합니다.[^1_4]

**트렌드 2: Probabilistic Forecasting**
Gaussian Process 기반 불확실성 추정이 확산됩니다. LightCTS에 Monte Carlo dropout 추가 필요합니다.[^1_10]

**트렌드 3: Longer Horizons (720+ steps)**
Lite-STGNN은 720 step 예측을 표준화했습니다. LightCTS의 last-shot compression은 이를 위해 개선 필요합니다.[^1_5]

**트렌드 4: Graph Learning Automation**
STM-Graph 같은 자동 그래프 구성 도구가 등장했습니다. LightCTS와 파이프라인 통합 가능합니다.[^1_17]

**트렌드 5: Multimodal Integration**
CITRAS는 텍스트/수치 covariate를 통합합니다. LightCTS에 multimodal encoder 추가 시 응용 확장됩니다.[^1_8]

## 결론

LightCTS는 **"경량화와 정확도의 균형"**이라는 딜레마를 plain stacking, L-TCN, GL-Former, last-shot compression으로 해결한 선구적 연구입니다. 2025년 최신 연구들(Lite-STGNN, RS-GLinear, CITRAS)은 LightCTS의 경량화 철학을 계승하면서 장기 예측, 불확실성 정량화, foundation model 통합으로 확장하고 있습니다. 향후 연구는 **동적 그래프 학습, 확률적 예측, multimodal covariate 통합, AutoML 기반 하이퍼파라미터 최적화**를 중점적으로 다뤄야 하며, LightCTS를 foundation model의 경량 adapter로 재구성하는 방향이 유망합니다.[^1_6][^1_11][^1_8][^1_4][^1_5][^1_1]
<span style="display:none">[^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39]</span>

<div align="center">⁂</div>

[^1_1]: 2302.11974v2.pdf

[^1_2]: https://vbn.aau.dk/en/publications/lightcts-lightweight-correlated-time-series-forecasting-enhanced-/

[^1_3]: https://forskning.ruc.dk/en/publications/lightcts-lightweight-correlated-time-series-forecasting-enhanced-/

[^1_4]: https://arxiv.org/html/2509.13906v1

[^1_5]: https://arxiv.org/html/2512.17453v1

[^1_6]: https://dl.acm.org/doi/10.1145/3589270

[^1_7]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11680986/

[^1_8]: https://arxiv.org/html/2503.24007v1

[^1_9]: https://arxiv.org/pdf/2510.03519.pdf

[^1_10]: https://arxiv.org/abs/2511.19657

[^1_11]: https://arxiv.org/abs/2510.03788

[^1_12]: https://www.mdpi.com/2076-3417/15/21/11580

[^1_13]: https://arxiv.org/pdf/2112.11174.pdf

[^1_14]: https://arxiv.org/pdf/2405.00946.pdf

[^1_15]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12453695/

[^1_16]: https://peerj.com/articles/cs-3001/

[^1_17]: https://arxiv.org/html/2509.10528v1

[^1_18]: https://iaj.aktuaris.or.id/index.php/iaj/article/view/28

[^1_19]: https://link.springer.com/10.1007/s13042-025-02778-8

[^1_20]: https://csitjournal.khmnu.edu.ua/index.php/csit/article/view/447

[^1_21]: https://ejurnal.stmik-budidarma.ac.id/jurikom/article/view/8495

[^1_22]: https://link.springer.com/10.1007/s11250-025-04388-6

[^1_23]: https://pub.isae.in/index.php/jae/article/view/1117

[^1_24]: https://www.impaxon.com/agriculture/1/2

[^1_25]: https://arxiv.org/pdf/2302.11974.pdf

[^1_26]: http://arxiv.org/pdf/2405.19661.pdf

[^1_27]: https://arxiv.org/pdf/2303.18205.pdf

[^1_28]: http://arxiv.org/pdf/2412.17603.pdf

[^1_29]: https://arxiv.org/pdf/2402.02399.pdf

[^1_30]: https://arxiv.org/html/2506.14831v2

[^1_31]: https://arxiv.org/html/2602.00731v1

[^1_32]: https://arxiv.org/html/2510.07041v1

[^1_33]: https://arxiv.org/html/2601.12380v1

[^1_34]: https://arxiv.org/html/2511.12104v1

[^1_35]: https://arxiv.org/pdf/2509.08679.pdf

[^1_36]: https://github.com/ai4cts/lightcts

[^1_37]: https://forskning.ruc.dk/en/publications/lightcts-lightweight-correlated-time-series-forecasting-enhanced-

[^1_38]: https://openreview.net/forum?id=9UWMXVpmtm

[^1_39]: https://www.sciencedirect.com/science/article/abs/pii/S0925231225010720

