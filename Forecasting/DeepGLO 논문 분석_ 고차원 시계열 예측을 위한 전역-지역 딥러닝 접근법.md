<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# DeepGLO 논문 분석: 고차원 시계열 예측을 위한 전역-지역 딥러닝 접근법

## 1. 핵심 주장과 주요 기여

DeepGLO(Sen et al., 2019)는 **"전역적으로 생각하고 지역적으로 행동하는"** 고차원 시계열 예측 프레임워크입니다. 이 논문은 기존 딥러닝 기반 시계열 모델들이 예측 시 오직 개별 시계열의 과거 데이터만 활용하는 한계를 극복하기 위해, 전역 패턴과 지역 특성을 동시에 활용하는 하이브리드 모델을 제안합니다.[^1_1]

### 주요 기여

- **LeveledInit**: 다양한 스케일을 가진 시계열 데이터를 사전 정규화 없이 학습할 수 있는 Temporal Convolution Network(TCN) 초기화 방식[^1_1]
- **TCN-MF**: TCN으로 정규화된 행렬 분해 모델로, 전역 시간적 패턴을 비선형적으로 포착[^1_1]
- **DeepGLO**: 전역 모델의 출력을 지역 TCN의 공변량으로 활용하여 전역 및 지역 특성을 통합[^1_1]
- **실험 결과**: 110K 이상의 차원을 가진 위키피디아 데이터셋에서 기존 방법 대비 25% 이상의 WAPE 개선[^1_1]


## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**문제 1: 스케일 다양성 문제**[^1_1]
고차원 시계열 데이터에서 개별 시계열의 스케일이 수 자릿수 차이날 수 있습니다. 예를 들어 소매 수요 예측에서 인기 상품과 틈새 상품의 수요 차이가 극심한 경우입니다. 기존 딥러닝 모델(LSTM, TCN)은 학습 성공을 위해 각 시계열을 적절히 정규화해야 하며, 이는 예측 성능에 큰 영향을 미칩니다.[^1_1]

**문제 2: 전역 패턴 활용 부재**[^1_1]
기존 딥러닝 모델들은 전체 데이터셋으로 학습되지만, 예측 시에는 오직 해당 시계열의 과거 값만 사용합니다. 그러나 주식 시장이나 소매 수요 예측에서는 유사 항목들의 과거 값이 예측에 유용할 수 있습니다.[^1_1]

### 2.2 제안하는 방법

#### **LeveledInit (Leveled Initialization)**

TCN의 간단한 초기화 방식으로, 모든 필터 가중치를 $1/k$ (k는 필터 크기)로 설정하고 편향을 0으로 초기화합니다.[^1_1]

**Proposition 1**: 필터 크기가 $k=2$이고 $d$개 레이어를 가진 TCN에서 LeveledInit를 적용하면, 모든 활성화 함수가 ReLU이고 입력이 비음수일 때, 출력 예측값 $\hat{y}_j$는 과거 $l$ 시점의 평균과 같습니다:[^1_1]

$\hat{y}_j = \mu(y_J) = \frac{1}{l} \sum_{i \in J} y_i$

여기서 $J = \{j-l, ..., j-1\}$이고 $l = 2(k-1)2^{d-1}$입니다.[^1_1]

이 초기화는 모델이 학습 초기에 과거 값들의 평균을 예측하도록 하여, 이후 학습에서 스케일에 무관한 변동성을 학습할 수 있게 합니다.[^1_1]

#### **TCN-MF (TCN-regularized Matrix Factorization)**

고차원 시계열 행렬 $\mathbf{Y}^{(tr)} \in \mathbb{R}^{n \times t}$를 저랭크 인수분해합니다:[^1_1]

$\mathbf{Y}^{(tr)} \approx \mathbf{F}\mathbf{X}^{(tr)}$

여기서 $\mathbf{F} \in \mathbb{R}^{n \times k}$, $\mathbf{X}^{(tr)} \in \mathbb{R}^{k \times t}$이고 $k \ll n$입니다.[^1_1]

**시간적 정규화**는 TCN $\mathcal{T}_X(\cdot)$을 사용하여 $\mathbf{X}^{(tr)}$의 시간 구조를 보존합니다:[^1_1]

$R(\mathbf{X}^{(tr)} | \mathcal{T}_X(\cdot)) := \frac{1}{|J|} L_2(\mathbf{X}[:, J], \mathcal{T}_X(\mathbf{X}[:, J-1]))$

여기서 $J = \{2, ..., t\}$이고, $L_2$는 제곱 손실입니다.[^1_1]

**전역 손실 함수**:[^1_1]

$L_G(\mathbf{Y}^{(tr)}, \mathbf{F}, \mathbf{X}^{(tr)}, \mathcal{T}_X) := L_2(\mathbf{Y}^{(tr)}, \mathbf{F}\mathbf{X}^{(tr)}) + \lambda_T R(\mathbf{X}^{(tr)} | \mathcal{T}_X(\cdot))$

여기서 $\lambda_T$는 정규화 파라미터입니다.[^1_1]

#### **DeepGLO: 하이브리드 모델**

DeepGLO는 전역 TCN-MF 모델의 예측값을 지역 TCN의 입력 공변량으로 제공합니다[^1_1]. 지역 TCN $\mathcal{T}_Y(\cdot|\Theta_Y)$는 $r+2$ 차원 입력을 받습니다[^1_1]:

1. 원본 시계열의 과거 값
2. $r$차원 원본 공변량
3. 전역 모델의 예측값

**학습 절차** (Algorithm 4):[^1_1]

1. TCN-MF로 전역 인수 $\mathbf{F}$, $\mathbf{X}^{(tr)}$, $\mathcal{T}_X$ 학습
2. 전역 모델 예측 $\hat{\mathbf{Y}}^{(g)}$ 생성
3. 확장된 공변량 생성: $\mathbf{Z}' \in \mathbb{R}^{n \times (r+1) \times t}$, 여기서 $\mathbf{Z}'[:, 1, :] = \hat{\mathbf{Y}}^{(g)}$
4. $\mathcal{T}_Y$를 $\mathbf{Y}^{(tr)}$과 $\mathbf{Z}'$로 학습

### 2.3 모델 구조

**TCN 아키텍처**:[^1_1]

- Causal convolution layers with dilation
- 레이어 $i$의 dilation: $\text{dil}(i) = 2^{i-1}$
- 동적 범위 (look-back): $l' = 1 + 2(k-1)2^{d-1}$
- 실험 설정: 대부분 데이터셋에서  구조, 필터 크기 $k=7$ 사용[^1_2][^1_1]

**TCN-MF 구조**:[^1_1]

- 인수분해 랭크 $k$: electricity(64), traffic(64), wiki(256), PeMSD7(M)(64)
- $\mathcal{T}_X$: 공변량 없이 기저 시계열에 대한 TCN
- 교대 학습: $\mathbf{F}$, $\mathbf{X}^{(tr)}$ 업데이트 → $\mathcal{T}_X$ 업데이트

**DeepGLO 전체 구조**:[^1_1]

- 전역 컴포넌트: TCN-MF ($\mathbf{F}$, $\mathbf{X}$, $\mathcal{T}_X$)
- 지역 컴포넌트: $\mathcal{T}_Y$ (시간 특징 7개 포함: 시각, 요일 등)
- 예측: 다단계 lookahead 방식으로 $\tau$ 시점 예측


### 2.4 성능 향상

**정량적 결과** (Table 2):[^1_1]


| 데이터셋 | DeepGLO (정규화) | DeepGLO (비정규화) | 차선책 |
| :-- | :-- | :-- | :-- |
| electricity (n=370) | 0.133/0.453/0.162 | **0.082/0.341/0.121** | 0.086/0.259/0.141 (DeepAR-정규화) |
| traffic (n=963) | 0.166/0.210/0.179 | **0.148/0.168/0.142** | 0.140/0.201/0.114 (DeepAR-정규화) |
| wiki (n=115,084) | 0.569/3.335/1.036 | **0.237/0.441/0.395** | 0.212/0.316/0.296 (Local TCN-비정규화) |

*WAPE/MAPE/SMAPE 지표*

**주요 개선사항**:[^1_1]

- Wiki 데이터셋 (110K+ 차원): 비정규화 설정에서 25% WAPE 개선
- 비정규화 설정에서 일관되게 우수한 성능 (스케일 문제 해결)
- DeepAR, LSTM, 기본 TCN은 비정규화 설정에서 성능 저하


### 2.5 한계

**논문에서 명시된 한계**:

1. **랭크 선택**: 인수분해 랭크 $k$는 수동으로 선택되며, 데이터셋마다 다름 (64~1,024)[^1_1]
2. **계산 복잡도**: 전역 모델과 지역 모델을 순차적으로 학습해야 하므로 단일 모델 대비 학습 시간 증가[^1_1]
3. **그래프 구조 미활용**: STGCN과 같은 방법은 시계열 간 관계를 나타내는 가중 그래프를 사용하지만, DeepGLO는 이를 활용하지 않음[^1_1]
4. **롤링 예측 시 업데이트**: TCN-MF는 재학습 없이 롤링 예측 가능하지만, 새로운 관측값 통합을 위해 $\mathbf{X}[:, t_1+1:t_2]$를 최적화해야 함[^1_1]

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 이론적 메커니즘

**스케일 불변성**:[^1_1]
LeveledInit는 초기 예측을 과거 평균으로 설정하여, 모델이 절대 스케일 대신 상대적 변동성을 학습하도록 유도합니다. 이는 새로운 스케일의 시계열에 대한 일반화를 개선합니다.

**저차원 표현 학습**:[^1_1]
TCN-MF는 $n$개 시계열을 $k$개 기저 시계열의 선형 결합으로 표현 ($k \ll n$)합니다. 이러한 차원 축소는:

- 과적합 방지 (파라미터 수 감소)
- 전역 패턴 포착 (기저 시계열이 공통 동역학 학습)
- 새로운 시계열에 대한 전이 학습 가능성

**하이브리드 아키텍처**:[^1_1]
전역 모델이 거시적 패턴을 포착하고 지역 모델이 미시적 변동을 조정함으로써, 양 극단의 과적합/과소적합을 피합니다.

### 3.2 실험적 증거

**다양한 도메인에서의 성능**:[^1_1]

- 전력 소비 (electricity): 370 시계열
- 교통 (traffic): 963 시계열
- 웹 트래픽 (wiki): 115,084 시계열
- 교통 센서 (PeMSD7): 228 시계열

모든 도메인에서 상위 2위 이내 성능을 보여 도메인 간 일반화 능력 입증.[^1_1]

**롤링 예측 성능**:[^1_1]
7개 예측 윈도우(electricity, traffic) 및 4개 윈도우(wiki)에 대한 롤링 검증에서 일관된 성능을 유지하여, 시간적 분포 변화에 대한 강건성을 보였습니다.

### 3.3 일반화 제한 요소

**데이터 의존성**:
매우 이질적인 시계열(공통 패턴 부족)에서는 전역 모델의 이점이 제한적일 수 있습니다.[^1_1]

**외부 공변량 의존**:
시간 특징(요일, 시각 등) 7개를 사용하며, 도메인별 공변량 설계가 성능에 영향을 미칩니다.[^1_1]

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 연구 영향

**1. 전역-지역 패러다임의 확산**[^1_1]
DeepGLO는 시계열 예측에서 "전역 패턴 + 지역 조정" 프레임워크를 제시하였으며, 이는 이후 연구들에 영향을 미쳤습니다:

- LGnet (2019): 메모리 네트워크를 통한 전역 패턴 탐색과 LSTM 기반 지역 예측 결합[^1_3][^1_4]
- Deep Factors (2019): 전역 인수와 지역 모델 결합[^1_1]

**2. 비정규화 학습의 중요성**[^1_1]
LeveledInit는 스케일 다양성 문제 해결을 위한 초기화 기법의 중요성을 보여주었으며, 이는 실무 응용에서 전처리 부담을 크게 줄였습니다.

**3. 행렬 분해와 딥러닝의 통합**[^1_1]
TRMF의 선형 시간 정규화를 TCN의 비선형 정규화로 확장하여, 전통적 방법과 딥러닝의 융합 가능성을 제시했습니다.

### 4.2 향후 연구 고려사항

**1. 자동 랭크 선택**
현재 인수분해 랭크 $k$는 수동 선택됩니다. 자동 랭크 결정 메커니즘(예: 정보 기준, 교차 검증)이 필요합니다.

**2. 그래프 구조 통합**
시계열 간 명시적 관계 그래프(예: 공간적 인접성, 의미적 유사성)를 통합하여 전역 모델을 향상시킬 수 있습니다.

**3. 온라인 학습 및 적응**
개념 이동(concept drift)에 대응하기 위한 온라인 업데이트 메커니즘 개발이 필요합니다.

**4. 해석 가능성 향상**
기저 시계열의 의미 해석 및 전역-지역 기여도 분석 도구 개발이 필요합니다.

**5. 확장성 개선**
매우 고차원 데이터(백만 개 이상 시계열)에 대한 메모리 효율적 구현이 요구됩니다.

## 5. 2020년 이후 관련 최신 연구 비교

### 5.1 Transformer 기반 접근법

**STHD (Scalable Transformer for High-Dimensional MTS, 2024)**[^1_5][^1_6]

- **차이점**: Transformer의 관계 행렬 희소성(Relation Matrix Sparsity)과 ReIndex 학습 전략을 통해 고차원 채널 의존성 포착
- **DeepGLO와의 관계**: DeepGLO는 TCN 기반이지만, STHD는 Transformer의 self-attention으로 채널 간 의존성을 명시적으로 모델링
- **장점**: 더 긴 시퀀스와 복잡한 채널 상호작용 포착 가능
- **단점**: DeepGLO 대비 계산 비용이 높으며, STHD는 여전히 스케일 다양성 문제 해결에 명시적 초기화 기법을 사용하지 않음

**iTransformer, PatchTST, Informer (2023-2024)**[^1_7][^1_8]

- 시계열 특화 Transformer 아키텍처로 장기 의존성 포착 강화
- DeepGLO의 TCN 대비 긴 시퀀스에서 우수하지만, 고차원 데이터에서는 계산 복잡도 문제


### 5.2 전역-지역 하이브리드 모델

**LGnet (Joint Modeling of Local and Global Temporal Dynamics, 2019)**[^1_4][^1_3]

- **유사점**: 전역-지역 동역학 결합 (메모리 모듈로 전역 패턴, LSTM으로 지역 예측)
- **차이점**: 결측치가 있는 다변량 시계열에 특화, 적대적 학습(adversarial training) 사용
- **DeepGLO와의 관계**: 동시기 유사 아이디어이지만, DeepGLO는 행렬 분해 + TCN, LGnet은 메모리 네트워크 + LSTM

**Cluster-and-Conquer Framework (2021)**[^1_9]

- **접근법**: 3단계 프레임워크 (단변량 파라미터 추정 → 클러스터링 → 다변량 예측)
- **차이점**: 명시적 클러스터링 후 각 클러스터를 다변량 시계열로 처리
- **DeepGLO와의 관계**: 클러스터링이 DeepGLO의 행렬 분해와 유사한 역할 (시계열 그룹화)
- **장점**: 더 해석 가능한 구조


### 5.3 Mamba 및 State Space Models

**Bi-Mamba+ (2024)**[^1_10]

- **아키텍처**: 양방향 Mamba (State Space Model 기반)로 시계열 요소 간 상호작용 포착
- **차이점**: RNN의 재귀 구조 대신 SSM으로 장기 의존성 효율적 모델링
- **DeepGLO 대비**: 더 최신 아키텍처이지만, 고차원 전역-지역 분리 전략은 명시적이지 않음


### 5.4 고차원 함수형 시계열

**HDFTS (High-Dimensional Functional Time Series) 모델 (2023-2025)**[^1_11][^1_12][^1_13][^1_2]

- **접근법**: 이중 인수 구조(dual-factor structure)로 고차원 함수형 시계열 분해
- **DeepGLO와의 관계**: 행렬 분해 개념 공유하지만, 함수형 데이터(곡선)에 특화
- **차이점**: DeepGLO는 이산 시계열, HDFTS는 연속 함수형 관점


### 5.5 로버스트 및 이상치 처리

**RTNMFFM (Robust Temporal NMF, 2023)**[^1_14]

- **접근법**: $L_{2,1}$ 노름을 사용한 비음수 행렬 분해(NMF)로 이상치와 결측치에 강건
- **DeepGLO와의 관계**: 모두 행렬 분해 사용하지만, RTNMFFM은 이상치 강건성 강조
- **차이점**: DeepGLO는 TCN 정규화, RTNMFFM은 자기회귀 정규화 + $L_{2,1}$ 노름


### 5.6 최근 기법 통합

**VARMAformer (2025)**[^1_15]

- **혁신**: 전통적 VARMA(Vector AutoRegressive Moving Average)를 Transformer에 통합
- **의의**: DeepGLO처럼 전통 통계 기법(ARIMA 계열)과 딥러닝 융합
- **차이점**: 선형 VARMA vs. 비선형 행렬 분해


### 5.7 대규모 사전학습 모델

**LSTSMs (Large-Scale Time Series Models, 2024-2025)**[^1_16]

- **모델**: Moirai, TimeGPT, Timer, LLM4TS 등 사전학습된 대규모 모델
- **접근법**: 다양한 도메인의 대규모 데이터로 사전학습 후 파인튜닝
- **DeepGLO와의 관계**: DeepGLO는 개별 데이터셋 학습, LSTM은 전이 학습 접근
- **실험 결과**: LLM4TS_FS가 여러 벤치마크에서 최고 성능 달성
- **트레이드오프**: LSTM은 더 많은 데이터와 계산 필요, DeepGLO는 가볍고 해석 가능


### 5.8 실무 응용

**간헐적 시계열 예측 (2025)**[^1_17]

- **문제**: 많은 0값을 포함한 간헐적 데이터 (예: 희소 수요)
- **결과**: D-Linear가 최고 성능, Tweedie 분포 사용 효과적
- **DeepGLO 적용**: 소매 수요 예측에 적합하지만, 확률적 예측 헤드 추가 필요


### 5.9 종합 비교

| 측면 | DeepGLO (2019) | 최신 연구 (2020-2025) |
| :-- | :-- | :-- |
| **전역-지역 통합** | 행렬 분해 + TCN 하이브리드 | LGnet(메모리+LSTM), STHD(Transformer 희소성) |
| **스케일 다양성** | LeveledInit ⭐ | 대부분 정규화 필요 또는 명시적 해결 없음 |
| **아키텍처** | TCN 기반 | Transformer(STHD, iTransformer), Mamba(Bi-Mamba+), SSM |
| **전통 기법 융합** | TRMF → TCN-MF | VARMAformer(VARMA+Transformer) |
| **확장성** | 115K 시계열 검증 | STHD(초고차원), HDFTS(함수형) |
| **사전학습** | 없음 | LSTSMs(Moirai, TimeGPT, Timer) 등장 |
| **강건성** | 비정규화 학습 가능 | RTNMFFM($L_{2,1}$ 노름), 이상치 처리 강화 |
| **해석 가능성** | 기저 시계열 해석 가능 | 대부분 블랙박스, 일부 attention 시각화 |

### 5.10 DeepGLO의 현재 위치

**강점 유지**:

1. **비정규화 학습**: LeveledInit는 여전히 독창적이며, 최신 연구에서 거의 다루지 않음
2. **경량 및 효율성**: Transformer 대비 빠른 학습/추론
3. **해석 가능성**: 기저 시계열 추출로 도메인 지식 통합 가능

**추월된 영역**:

1. **장기 의존성**: Transformer 및 Mamba가 더 긴 시퀀스에서 우수
2. **채널 상호작용**: STHD의 희소 attention이 더 명시적
3. **전이 학습**: 사전학습 모델들이 소규모 데이터에서 더 효과적

**향후 발전 방향**:

- DeepGLO + Transformer 하이브리드 (전역: Transformer, 지역: TCN)
- LeveledInit를 Transformer/Mamba에 적응
- 사전학습된 인코더 + DeepGLO 디코더 구조


## 결론

DeepGLO는 2019년에 제안된 선구적인 고차원 시계열 예측 프레임워크로, 전역-지역 하이브리드 접근법과 비정규화 학습 기법을 통해 실무적 문제를 효과적으로 해결했습니다. 2020년 이후 Transformer 기반 모델, 사전학습 모델, State Space Models 등 새로운 아키텍처가 등장했지만, DeepGLO의 핵심 아이디어인 "전역 패턴 포착 + 지역 조정"과 "스케일 불변 초기화"는 여전히 관련성이 높습니다. 향후 연구는 DeepGLO의 효율성과 해석 가능성을 최신 아키텍처(Transformer, Mamba)와 결합하는 방향으로 발전할 것으로 예상됩니다.[^1_5][^1_10][^1_16][^1_1]
<span style="display:none">[^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38]</span>

<div align="center">⁂</div>

[^1_1]: 1905.03806v2.pdf

[^1_2]: https://academic.oup.com/jrsssa/advance-article/doi/10.1093/jrsssa/qnaf144/8276869

[^1_3]: https://arxiv.org/abs/1911.10273

[^1_4]: https://arxiv.org/pdf/1911.10273.pdf

[^1_5]: https://dl.acm.org/doi/10.1145/3627673.3679757

[^1_6]: https://arxiv.org/html/2408.04245v1

[^1_7]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12453695/

[^1_8]: https://peerj.com/articles/cs-3001/

[^1_9]: https://arxiv.org/pdf/2110.14011.pdf

[^1_10]: https://arxiv.org/pdf/2404.15772.pdf

[^1_11]: https://www.tandfonline.com/doi/full/10.1080/01621459.2024.2413201

[^1_12]: https://arxiv.org/pdf/2305.19749.pdf

[^1_13]: https://www.frontiersin.org/journals/applied-mathematics-and-statistics/articles/10.3389/fams.2025.1600278/full

[^1_14]: https://www.mdpi.com/1099-4300/26/1/92

[^1_15]: https://arxiv.org/abs/2509.04782

[^1_16]: https://arxiv.org/html/2507.02907v1

[^1_17]: https://arxiv.org/html/2601.14031v1

[^1_18]: https://www.mdpi.com/2072-4292/16/11/1915

[^1_19]: https://ieeexplore.ieee.org/document/10480246/

[^1_20]: https://ieeexplore.ieee.org/document/10814884/

[^1_21]: https://ieeexplore.ieee.org/document/10636928/

[^1_22]: https://link.springer.com/10.1007/s00521-020-05129-6

[^1_23]: https://linkinghub.elsevier.com/retrieve/pii/S0957417420305285

[^1_24]: https://arxiv.org/abs/2403.03850

[^1_25]: https://arxiv.org/pdf/2304.08424.pdf

[^1_26]: http://arxiv.org/pdf/2407.10768.pdf

[^1_27]: http://arxiv.org/pdf/2212.02567.pdf

[^1_28]: https://arxiv.org/pdf/2501.07048.pdf

[^1_29]: http://arxiv.org/pdf/2412.13769.pdf

[^1_30]: https://arxiv.org/pdf/2509.26468.pdf

[^1_31]: https://arxiv.org/html/2602.08588v1

[^1_32]: https://arxiv.org/pdf/2502.08302.pdf

[^1_33]: https://arxiv.org/html/2409.02891v1

[^1_34]: https://arxiv.org/html/2511.04988v2

[^1_35]: https://arxiv.org/pdf/2409.10030.pdf

[^1_36]: https://openreview.net/forum?id=CCV9RqCCoQ

[^1_37]: https://arxiv.org/html/2402.01999v1

[^1_38]: https://research.ibm.com/publications/a-method-for-high-dimensional-probabilistic-multivariate-time-series-forecasting

