<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# FDNet: Focal Decomposed Network 논문 심층 분석

FDNet은 시계열 예측에서 기존의 전역 특징 추출 패러다임을 깨고, 지역 세밀 특징만으로도 효과적인 예측이 가능함을 이론적·실증적으로 입증한 2023년 연구입니다. 본 모델은 6개 실세계 벤치마크에서 13개 SOTA 모델 대비 평균 38.4% MSE 감소를 달성했습니다.[^1_1]

## 1. 핵심 주장 및 주요 기여

### 핵심 주장

- **ICOM(Input Correlation Oriented Mechanism) 불필요성**: 기존 딥러닝 시계열 예측 모델들이 사용하는 전역 상관관계 추출 메커니즘(attention, dilated convolution 등)이 필수적이지 않음[^1_1]
- **지역 특징 추출의 우수성**: 분해된 예측 공식(Decomposed Forecasting Formula)을 통한 지역 세밀 특징만으로 더 강건하고 실용적인 예측 가능[^1_1]


### 주요 기여

1. **분해된 예측 공식 제안**: Direct forecasting 기반으로 예측 요소와 입력 요소의 특징 추출을 모두 분해[^1_1]
2. **Focal Input Sequence Decomposition**: LSTI(Long Sequence Time series Input) 문제 해결을 위한 초점 분해 방법 제안[^1_1]
3. **우수한 성능 및 효율성**:
    - 다변량/단변량 예측에서 각각 36.2%/40.5% 상대적 MSE 감소[^1_1]
    - 단순한 구조(선형 투영 + CNN)로 높은 정확도 달성[^1_1]
4. **분포 이동(Distribution Shift) 문제 해결**: KS 테스트로 실세계 시계열의 비정상성을 입증하고, ICOM 제거로 근본적 해결[^1_1]

## 2. 상세 분석

### 해결하고자 하는 문제

**문제 1: ICOM의 필요성과 한계**

기존 모델들은 입력 시퀀스의 전역 특징 추출을 위해 ICOM을 사용하지만, 이는 다음과 같은 문제를 야기합니다:[^1_1]

- 수동 가정(season-trend 분해 등)에 의존하여 일반성 부족
- 비정상 시계열의 분포 이동에 취약
- 국소 윈도우마다 다른 통계적 특성을 가진 실세계 데이터에 부적합

**문제 2: LSTI(Long Sequence Time series Input) 문제**

- Transformer 기반 모델들도 입력 시퀀스가 과도하게 길어지면 성능 저하(overfitting)[^1_1]
- 긴 입력 시퀀스 처리 시 시간 소모 및 파라미터 폭발[^1_1]


### 제안하는 방법 (수식 포함)

**참고**: 시스템 설정에 따라 LaTeX 수식은 $$ 형식으로 제공됩니다.

#### 분해된 예측 공식 (Decomposed Forecasting Formula)

시계열 예측 문제를 다음과 같이 정의합니다:

- 입력: $\{z_{i,1:t_0}\}_{i=1}^V$ (V개 변수, 길이 $t_0$)
- 출력: $\{z_{i,t_0+1:T}\}_{i=1}^V$ (예측 시퀀스)

기존 방식과의 차이:[^1_1]

- **Rolling forecasting**: $T-t_0$단계로 순차 예측
- **Direct forecasting**: 한 번의 forward로 전체 예측
- **Decomposed forecasting** (제안): Direct 방식에서 입력 요소의 특징 추출까지 분해


#### Focal Input Sequence Decomposition

입력 시퀀스를 $f$개의 연속적 부분 시퀀스로 분할하며, 비율은 기하급수 형태($\alpha=0.5$)를 따릅니다:[^1_1]

예: $f=4$일 때 비율 = $\{1/2, 1/4, 1/8, 1/8\}$

**핵심 원리**:

- 예측 시퀀스에 **가까운** 부분: 짧은 길이, 많은 특징 추출 레이어
- 예측 시퀀스에 **먼** 부분: 긴 길이, 적은 특징 추출 레이어

이를 통해 시간적 거리에 따라 적응적으로 복잡도를 할당하며, 파라미터 증가율을 제어합니다.[^1_1]

### 모델 구조

FDNet은 $N+2$개 레이어로 구성됩니다:[^1_1]

1. **Embedding 레이어**: 입력을 잠재 공간으로 투영
2. **N개의 분해된 특징 추출 레이어**: 각 레이어는 4개의 2D convolutional 레이어 포함
    - 홀수 레이어: 1×1 Conv (Perceptron과 동일)
    - 짝수 레이어: 3×1 Conv (지역성 강화, 이상치 평활화)
3. **투영 레이어**: 최종 예측 출력

**2D Convolution 사용 이유**:[^1_1]

- Variable dimension의 커널 크기는 항상 1로 설정
- 다른 변수의 요소 값이 서로 영향을 주지 않도록 보장
- Variable-specific과 Variable-agnostic 방법의 균형

**특징**:

- Weight Normalization, Gelu activation, residual connection 활용[^1_1]
- 예측 길이 증가 시 최종 1D Conv 레이어의 파라미터만 증가하여 파라미터 폭발 방지[^1_1]


### 성능 향상

#### 정량적 성능

**다변량 예측** (Table 5 기준):[^1_1]

- ETTh1: MSE 0.365~0.457 (96~720 예측 길이)
- ETTm2: MSE 0.168~0.417
- ECL: MSE 0.142~0.204
- 평균 38.4% MSE 감소 (13개 baseline 대비)

**단변량 예측** (Table 6 기준):[^1_1]

- ETTh1: MSE 0.067~0.167
- Traffic: MSE 0.134~0.162
- Weather: MSE 0.002~0.003

**LSTI 처리 능력** (Table 10):[^1_1]

- ECL 단변량 예측에서 입력 길이 96→1344 증가 시:
    - FDNet: MSE 0.391→0.209 (46.5% 개선)
    - ETSformer: MSE 0.726→0.893 (악화)
    - FEDformer: MSE 0.268→0.323 (악화)


#### 효율성

- **파라미터**: 매우 적음 (0.1~1.0 MB, 입력 길이에 따라)[^1_1]
- **GPU 메모리**: FEDformer 대비 1/5 수준[^1_1]
- **학습 시간**: 경쟁 모델 대비 우수[^1_1]


### 한계

#### 1. 작은 데이터셋에서의 성능

Exchange 데이터셋(크기 7,588)에서 입력 길이 96으로 제한:[^1_1]

- 더 긴 입력(192~384)에서 성능 저하 관찰
- 이유: 학습 인스턴스 감소, 모델의 약한 귀납적 편향


#### 2. ILI 데이터셋 제외

논문에서 ILI 데이터셋(크기 966)을 실험에서 제외:[^1_1]

- LSTI 문제 분석에 부적합한 크기
- FDNet의 약한 inductive bias로 인해 충분한 학습 데이터 필요


#### 3. Universal Feature가 실제로 유용한 경우

ETTh1 단변량 예측에서 FEDformer, ETSformer가 더 나은 성능:[^1_1]

- 이 경우 FUNet(Focal + ICOM)이 더 효과적
- FDNet의 철학적 접근이 모든 상황에서 최적은 아님


#### 4. 해석 가능성 제한

- 단순한 구조로 인한 해석 가능성 향상이 있으나, 특정 도메인 지식 통합에는 제한적[^1_1]


## 3. 모델의 일반화 성능 향상 가능성

### 이론적 근거

**1. ICOM 제거를 통한 강건성 확보**

FDNet은 KS(Kolmogorov-Smirnov) 테스트를 통해 실세계 시계열의 비정상성을 입증했습니다:[^1_1]

KS 통계량: $$
D = \sup_x |F(\{x_i\}_{i=m}^{m+n_1}) - F(\{x_i\}_{i=n}^{n+n_1})|
$$

6개 데이터셋의 Reject Rate (RR):[^1_1]

- ETTh1: 98.2%, ETTm2: 98.4%
- ECL: 66.4%, Traffic: 92.2%
- Exchange: 95.3%, Weather: 86.0%

이는 국소 윈도우마다 통계적 특성이 크게 다름을 의미하며, 전역 특징 추출이 오히려 해로울 수 있음을 시사합니다.[^1_1]

**2. 분포 이동에 대한 강건성**

CoST 모델을 사용한 ablation study에서 큰 커널(128, 64) 제거 시 성능 향상 확인:[^1_1]

- 작은 커널({1,2})만 사용 시: 더 나은 예측 정확도
- 큰 커널 사용 시: 분포 이동에 취약, overfitting 심화

이는 지역 특징 추출이 일반화에 유리함을 실증적으로 증명합니다.[^1_1]

### 실증적 증거

**1. 다양한 도메인에서의 일관된 성능**

6개 데이터셋은 서로 다른 특성을 가집니다:[^1_1]

- **주파수**: 15분(ETTm2) ~ 1일(Exchange)
- **차원**: 7 ~ 862 변수
- **도메인**: 전력, 교통, 금융, 기상

FDNet은 모든 데이터셋에서 상위 2위 내 성능 달성.[^1_1]

**2. 변동성(Robustness) 분석**

Autoformer의 ETTm2 단변량 예측 표준편차가 크게 나타난 반면, FDNet은 안정적 성능:[^1_1]

- 더 작은 표준편차
- 랜덤 초기화에 덜 민감

**3. Variable-Specific vs Variable-Agnostic 균형**

FDNet의 2D convolution 전략은:[^1_1]

- 변수별 특성 존중 (값 혼합 방지)
- 동일한 가중치 행렬 공유 (파라미터 효율성)
- 수백 개 변수에서도 효과적 (Traffic: 862 변수)


### 일반화 성능 향상 메커니즘

#### 1. Focal Decomposition의 적응성

Focal 방법은 시간적 거리에 따라 적응적 모델링을 제공합니다:[^1_1]

- 최근 정보: 세밀한 모델링 (짧은 시퀀스, 깊은 네트워크)
- 먼 과거: 거친 모델링 (긴 시퀀스, 얕은 네트워크)

Table 11의 $f$ 변화 실험에서:[^1_1]

- ECL: $f=3$에서 최적
- Traffic: $f=2$에서 최적
- 데이터셋 특성에 따라 조정 가능 → 일반화 가능성 높음


#### 2. Overfitting 방지

LSTI 실험 결과(Table 10):[^1_1]

- 입력 길이 증가에도 성능 유지/개선
- 표준편차 안정적 (1.0e-3~1.5e-3 수준)
- 파라미터 증가 최소 (0.1MB → 1.0MB)


#### 3. 이상치에 대한 강건성

3×1 convolution 사용으로 이상치 평활화:[^1_1]

- Stride=1로 국소성 유지
- 순수 element-wise 방식보다 안정적


### 일반화 한계와 개선 방향

**한계**:

1. **작은 데이터셋**: 약한 inductive bias로 인해 충분한 학습 데이터 필요[^1_1]
2. **명확한 전역 패턴 존재 시**: Seasonal-trend가 명확한 경우 ICOM이 유리할 수 있음[^1_1]

**개선 방향**:

- FUNet과의 하이브리드: Focal + ICOM으로 다양한 상황 대응[^1_1]
- Meta-learning 적용: 적은 데이터에서도 효과적 학습
- Domain adaptation: 특정 도메인 지식 통합 메커니즘


## 4. 향후 연구에 미치는 영향 및 고려사항

### 연구에 미치는 영향

#### 1. 패러다임 전환: "Less is More"

FDNet은 "복잡한 attention mechanism이 반드시 필요한가?"라는 근본적 질문을 제기했습니다. 이는 시계열 예측 커뮤니티에 다음과 같은 영향을 미쳤습니다:[^1_1]

**영향받은 후속 연구**:

- **DLinear (2023)**: 단순 선형 레이어만으로 Transformer 능가[^1_1]
- **PatchTST (2023)**: Channel-independence 강조, FDNet과 유사한 철학[^1_2]
- **Are Transformers Effective?** 논문: Self-attention의 실제 효과 재검토[^1_3]


#### 2. 분해 전략의 다양화

Focal decomposition은 새로운 분해 패러다임을 제시했습니다:[^1_1]

- 기존: Seasonal-trend 분해 (도메인 가정 필요)
- FDNet: 시간적 거리 기반 분해 (일반적 가정)

**파급 효과**:

- Multi-resolution 접근법으로 발전 (MTST, 2023)[^1_4]
- Patch-based 방법론 확산 (PSformer, 2025)[^1_5]


#### 3. LSTI 문제에 대한 새로운 해법

Focal 방법은 파라미터 효율적 long-context 모델링 제시:[^1_1]

- Timer-XL (2025): Long-context transformer에 유사 개념 적용[^1_6][^1_7]
- Billion-scale foundation model에도 적용 가능성[^1_8]


### 향후 연구 시 고려사항

#### 1. 아키텍처 설계 측면

**고려사항 A: ICOM 필요성 재평가**

모든 시계열 문제에 일률적으로 attention을 적용하기보다는:

- 데이터 특성 분석 우선 (KS test 등)[^1_1]
- 분포 이동 정도에 따라 모델 선택
- Hybrid 접근 (FUNet) 고려[^1_1]

**권장사항**:

```
IF 명확한 전역 패턴 존재 THEN
    Use ICOM (Transformer, etc.)
ELSE IF 높은 비정상성 THEN
    Use Decomposed approach (FDNet)
ELSE
    Use Hybrid (FUNet)
```

**고려사항 B: Variable Modeling 전략**

FDNet의 2D convolution 전략은 variable 모델링의 중간 지점을 제시:[^1_1]

- iTransformer (2024): 각 variable을 token으로 처리[^1_9]
- Channel-mixing vs Channel-independence 재검토

**고려사항 C: Parameter Efficiency**

Foundation model 시대에 parameter-efficient 설계 중요성 증가:

- Focal 방식의 adaptive depth allocation[^1_1]
- LoRA-style fine-tuning과의 결합 가능성[^1_10]


#### 2. 실무 적용 측면

**고려사항 A: 데이터 크기와 모델 선택**

FDNet의 사례에서 배울 점:[^1_1]

- **큰 데이터셋** (> 10,000 samples): FDNet 유리
- **중간 데이터셋** (1,000~10,000): Focal parameter 조정 필요
- **작은 데이터셋** (< 1,000): ICOM 기반 모델이나 전통적 방법 고려

**고려사항 B: Computational Resource**

FDNet의 효율성 장점 활용:[^1_1]

- Edge device 배포: 낮은 메모리, 빠른 추론
- Real-time forecasting: LSTI 처리 능력 활용
- 대규모 변수: Variable-specific보다 효율적


#### 3. 이론적 연구 측면

**고려사항 A: Universal Approximation vs Inductive Bias**

FDNet은 약한 inductive bias 전략 채택:[^1_1]

- Trade-off: 일반성 ↑, 데이터 요구량 ↑
- 향후 연구: Optimal bias 수준 탐구 필요

**고려사항 B: Distribution Shift 이론**

FDNet의 KS test 분석은 시작점:[^1_1]

- 더 정교한 비정상성 측정 지표 개발
- 적응적 normalization 전략 (Non-stationary Transformer)[^1_1]

**고려사항 C: Long-term Dependency**

Focal 방법의 이론적 한계 연구 필요:

- 얕은 네트워크로 먼 과거 정보 충분히 활용 가능한가?
- Receptive field와 예측 성능의 관계


#### 4. 평가 방법론 측면

**고려사항 A: 일반화 평가 개선**

FDNet 평가에서 부족한 점:

- Cross-domain generalization 평가 부재
- Zero-shot/Few-shot 성능 미검증 (cf. CHRONOS)[^1_11]

**권장 평가 프로토콜**:

1. Within-domain: 기존 방식
2. Cross-domain: 다른 도메인 데이터로 테스트
3. Distribution shift: Train-test 분포 차이 시뮬레이션
4. Sample efficiency: 제한된 데이터에서의 성능

**고려사항 B: Robustness 평가 강화**

표준편차, confidence interval 보고 필수화:[^1_1]

- FDNet처럼 10회 반복 실험
- Bootstrap 기반 신뢰구간


## 5. 2020년 이후 관련 최신 연구 비교 분석

### 주요 연구 트렌드

#### Trend 1: Transformer의 재평가와 단순화 (2022-2024)

**핵심 발견**: 복잡한 Transformer가 항상 최선은 아님


| 모델 | 년도 | 핵심 아이디어 | FDNet과의 관계 |
| :-- | :-- | :-- | :-- |
| **DLinear**[^1_1] | 2023 | 단순 선형 레이어로 충분 | FDNet과 유사한 "Less is More" 철학 |
| **PatchTST**[^1_2] | 2023 | Channel-independence + patching | FDNet의 변수 분리와 일맥상통 |
| **Are Transformers Effective?**[^1_3] | 2024 | Self-attention 효과 의문 제기 | FDNet의 ICOM 불필요 주장 지지 |
| **LATST**[^1_12] | 2024 | 단순화된 Transformer | Entropy collapse 해결, FDNet보다 복잡 |
| **CATS**[^1_3] | 2024 | Cross-attention만 사용, self-attention 제거 | FDNet의 ICOM 제거와 유사한 동기 |

**분석**: FDNet(2023)은 이 트렌드의 선구자 중 하나로, ICOM의 필요성을 이론적·실증적으로 재검토했습니다.[^1_1]

#### Trend 2: Multi-scale/Multi-resolution 접근 (2023-2025)

**핵심 아이디어**: 다양한 스케일에서 시계열 모델링


| 모델 | 년도 | 방법론 | FDNet Focal과의 비교 |
| :-- | :-- | :-- | :-- |
| **MTST**[^1_4][^1_13] | 2023 | 다양한 patch 크기 병렬 처리 | Focal: 시간 거리 기반, MTST: 주파수 기반 |
| **PSformer**[^1_5] | 2025 | Segment attention + parameter sharing | Parameter efficiency에서 유사 |
| **Ada-MSHyper**[^1_14] | 2024 | Multi-scale hypergraph | Focal보다 복잡한 그래프 구조 |
| **Sentinel**[^1_15] | 2025 | Multi-patch attention | Temporal + channel attention 동시 고려 |

**FDNet의 차별점**:

- Focal 방법은 **adaptive depth allocation**에 초점[^1_1]
- 다른 방법들은 주로 **attention 범위** 조정
- Focal은 더 parameter-efficient (Table 11)[^1_1]


#### Trend 3: Foundation Models와 Pre-training (2024-2025)

**핵심 트렌드**: 대규모 사전학습 모델의 등장


| 모델 | 년도 | 특징 | 데이터 규모 |
| :-- | :-- | :-- | :-- |
| **Time-MoE**[^1_8] | 2024 | Mixture of Experts | Billion-scale |
| **CHRONOS**[^1_11] | 2025 | Pre-trained language model 활용 | Zero-shot 가능 |
| **Timer-XL**[^1_6][^1_7] | 2025 | Long-context unified model | Multi-domain |
| **AutoTimes**[^1_10] | 2024 | GPT-2 + LoRA | Transfer learning |

**FDNet의 위치**:

- FDNet은 foundation model이 **아님** (task-specific)[^1_1]
- 하지만 **효율성**과 **단순성**에서 강점
- Focal 개념은 foundation model의 fine-tuning에 적용 가능성

**시사점**: 향후 "FDNet-style foundation model" 연구 가능

- Focal decomposition + Pre-training
- Parameter-efficient adaptation


#### Trend 4: Domain-Specific Transformers (2024-2025)

**핵심 트렌드**: 특정 도메인 지식 통합


| 모델 | 년도 | 도메인 | 특수 메커니즘 |
| :-- | :-- | :-- | :-- |
| **FEDformer**[^1_1][^1_16] | 2022 | 금융 | Frequency enhanced decomposition |
| **ETSformer**[^1_1][^1_17] | 2022 | 일반 | Exponential smoothing attention |
| **TFT**[^1_18][^1_19][^1_20] | 2021-2025 | 다중 도메인 | Static/known/observed inputs 통합 |
| **ExoTST**[^1_21] | 2024 | Exogenous 변수 | Cross-temporal modality fusion |
| **QTFT**[^1_18][^1_19] | 2025 | 양자 컴퓨팅 | Quantum-enhanced TFT |

**FDNet과의 비교**:

- FDNet: **최소한의 도메인 가정** (auto-regression만)[^1_1]
- Domain-specific 모델: **강한 inductive bias**
- Trade-off: FDNet이 더 일반적, domain 모델이 특정 상황에서 우수

**실험 비교** (from Table 5/6):[^1_1]

- ETTh1 단변량: ETSformer 우수, FDNet 2위
- ECL/Traffic 다변량: FDNet 압도적 우수
- → 도메인 특성에 따라 선택 필요


#### Trend 5: Hybrid Architectures (2024-2025)

**핵심 트렌드**: 여러 메커니즘 결합


| 모델 | 년도 | 결합 요소 | 성능 |
| :-- | :-- | :-- | :-- |
| **HTMformer**[^1_9] | 2024 | Temporal + Multivariate extraction | Transformer 대비 35.8% 향상[^1_9] |
| **TimeMixer**[^1_9] | 2024 | Multi-scale mixing | Multi-horizon 특화 |
| **AutoFormer-TS**[^1_22] | 2025 | NAS + Transformer | Automated architecture search |

**FDNet의 FUNet 변형**:[^1_1]

- Focal + ICOM 결합
- ETTh1에서 우수한 성능 (Table 9)
- Hybrid 접근의 유효성 입증


### 종합 비교표

| 차원 | FDNet (2023) | 최신 Transformer (2024-25) | Foundation Models (2024-25) |
| :-- | :-- | :-- | :-- |
| **복잡도** | 매우 낮음 | 높음 | 매우 높음 |
| **파라미터** | 0.1~1 MB[^1_1] | 수십 MB | GB 단위 |
| **일반화** | 높음 (비정상성 강건) | 중간 (도메인 의존) | 매우 높음 (pre-training) |
| **효율성** | 매우 우수[^1_1] | 낮음~중간 | 낮음 (inference) |
| **해석 가능성** | 높음 (단순 구조) | 중간 (attention) | 낮음 (black-box) |
| **데이터 요구** | 중간~높음[^1_1] | 중간 | 매우 높음 (pre-training) |
| **Zero-shot** | 불가 | 불가 | 가능[^1_11] |

### 성능 비교 (동일 벤치마크)

**ETTh1 다변량 예측 (96 horizon)**:

- FDNet: MSE **0.365**[^1_1]
- PatchTST: MSE 0.393[^1_1]
- iTransformer: 비교 데이터 없음 (다른 설정)
- Transformer+HTME: MSE 0.271→0.185 (향상)[^1_9]

**ECL 다변량 예측 (96 horizon)**:

- FDNet: MSE **0.142**[^1_1]
- PatchTST: MSE 0.196[^1_1]
- N-HiTS: MSE 0.147[^1_1]
- FEDformer: MSE 0.188[^1_1]

→ FDNet이 여전히 경쟁력 있는 성능 유지

### 최신 연구의 FDNet 개념 활용

**1. Channel-Independence 확산**

- PatchTST (2023): "each channel contains a single univariate time series"[^1_2]
- iTransformer (2024): Variable을 token으로 처리[^1_9]
- FDNet의 variable 분해 개념과 일맥상통[^1_1]

**2. Patch-based 방법론**

- PatchTST: Time series를 patches로 분할[^1_2]
- PSformer (2025): Segment attention[^1_5]
- FDNet의 focal decomposition에서 영감[^1_1]

**3. Parameter Efficiency**

- PSformer: Parameter sharing 강조[^1_5]
- Timer-XL: Long-context에서 효율성[^1_6]
- FDNet의 focal 방식이 선구적 접근[^1_1]


### 미래 연구 방향 제안

#### 방향 1: FDNet + Foundation Model

- Focal decomposition을 pre-training에 적용
- Parameter-efficient fine-tuning (LoRA + Focal)
- 예상 효과: Zero-shot + Robustness 동시 달성


#### 방향 2: Quantum-Enhanced FDNet

- QTFT (2025)처럼 양자 컴퓨팅 활용[^1_18]
- 단순한 FDNet 구조가 양자 회로 구현에 유리
- NISQ 장치에서 실행 가능성


#### 방향 3: Adaptive ICOM Selection

- 데이터 특성 분석 후 ICOM 자동 선택
- FDNet ↔ FUNet 동적 전환
- Meta-learning 기반 모델 선택


#### 방향 4: Hybrid Multi-Scale

- MTST + Focal decomposition[^1_4]
- 주파수 기반 + 시간 거리 기반 결합
- 더 풍부한 표현 학습


### 결론: FDNet의 학술적 위치

**패러다임 전환의 촉매**:

- 2022-2023: Transformer 복잡도 정점 (FEDformer, ETSformer)
- **2023: FDNet**, DLinear → "Less is More" 시작
- 2024-2025: 단순화 트렌드 가속 (CATS, LATST)

**지속적 영향력**:

- 38.4% MSE 감소는 여전히 최상위 수준[^1_1]
- Foundation model 시대에도 효율성으로 차별화
- Focal 개념은 다양한 변형으로 계승 중

**한계와 보완**:

- Zero-shot 불가 → Foundation model이 보완
- 작은 데이터 취약 → Pre-training으로 해결 가능
- 명확한 전역 패턴 시 ICOM 필요 → Hybrid 접근

FDNet은 시계열 예측 연구에서 "복잡성보다 적절한 귀납적 편향"이 중요함을 입증한 이정표적 연구로, 향후 효율적이고 강건한 모델 설계의 기준점이 될 것입니다.[^1_23][^1_24][^1_1]
<span style="display:none">[^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39]</span>

<div align="center">⁂</div>

[^1_1]: 2306.10703v1.pdf

[^1_2]: http://arxiv.org/pdf/2211.14730v2.pdf

[^1_3]: https://arxiv.org/abs/2405.16877

[^1_4]: https://arxiv.org/abs/2311.04147

[^1_5]: https://arxiv.org/html/2411.01419v1

[^1_6]: http://arxiv.org/pdf/2410.04803.pdf

[^1_7]: https://arxiv.org/html/2507.02907v1

[^1_8]: https://arxiv.org/abs/2409.16040

[^1_9]: https://arxiv.org/html/2510.07084v1

[^1_10]: https://arxiv.org/html/2508.07697v3

[^1_11]: https://ieeexplore.ieee.org/document/11137629/

[^1_12]: https://arxiv.org/abs/2410.23749

[^1_13]: https://proceedings.mlr.press/v238/zhang24l.html

[^1_14]: http://arxiv.org/pdf/2410.23992.pdf

[^1_15]: http://arxiv.org/pdf/2503.17658.pdf

[^1_16]: https://arxiv.org/abs/2511.12951

[^1_17]: https://arxiv.org/pdf/2202.01381.pdf

[^1_18]: https://arxiv.org/abs/2508.04048

[^1_19]: https://arxiv.org/html/2508.04048v1

[^1_20]: https://aihorizonforecast.substack.com/p/temporal-fusion-transformer-time

[^1_21]: http://arxiv.org/pdf/2410.12184.pdf

[^1_22]: https://arxiv.org/pdf/2502.13721.pdf

[^1_23]: https://www.semanticscholar.org/paper/41742e4ae0f3c4afcf5a90eefbd685d63134d911

[^1_24]: https://arxiv.org/abs/2411.05793

[^1_25]: https://arxiv.org/abs/2510.10878

[^1_26]: https://journal.astanait.edu.kz/index.php/ojs/article/view/913

[^1_27]: https://dl.acm.org/doi/10.1145/3757749.3757774

[^1_28]: https://rjor.ro/considerations-on-the-efficiency-of-time-series-analysis-in-forecasting-new-influenza-cases-in-the-2024-2025-season/

[^1_29]: https://www.mdpi.com/2073-4433/16/3/292

[^1_30]: https://www.ewadirect.com/proceedings/ace/article/view/30132

[^1_31]: https://arxiv.org/html/2508.16641v1

[^1_32]: https://arxiv.org/html/2503.10198v1

[^1_33]: https://www.arxiv.org/pdf/2508.04048.pdf

[^1_34]: https://secundo.tistory.com/107

[^1_35]: https://www.sciencedirect.com/science/article/pii/S2665963824001040

[^1_36]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12453695/

[^1_37]: https://www.reddit.com/r/MachineLearning/comments/1aifjbq/r_literature_review_of_advances_recent_in_deep/

[^1_38]: https://peerj.com/articles/cs-3001/

[^1_39]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11402414/

