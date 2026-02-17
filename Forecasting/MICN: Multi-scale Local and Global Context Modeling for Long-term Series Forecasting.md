# MICN: Multi-scale Local and Global Context Modeling for Long-term Series Forecasting

## 1. 핵심 주장 및 주요 기여 요약

MICN(Multi-scale Isometric Convolution Network)은 장기 시계열 예측(Long-term Series Forecasting)을 위해 CNN 기반 접근법을 사용하여 Transformer의 self-attention 메커니즘을 효율적으로 대체한 모델입니다. 주요 기여는 다음과 같습니다:[^1_1]

- **선형 복잡도 달성**: self-attention의 $O(L^2)$ 복잡도를 $O(LD^2)$로 감소시켜 계산 효율성을 크게 향상[^1_1]
- **다중 스케일 프레임워크**: 복잡한 시간적 패턴을 별도로 모델링하는 다중 분기 구조 제안[^1_1]
- **Local-Global 구조**: downsampling convolution으로 국소 특징을 추출하고 isometric convolution으로 전역 상관관계를 포착[^1_1]
- **SOTA 성능**: 다변량 예측에서 17.2%, 단변량 예측에서 21.6%의 상대적 MSE 개선[^1_1]


## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**Transformer 기반 모델의 한계**:

- Self-attention의 $O(L^2)$ 시간 및 공간 복잡도로 인한 비효율성[^1_1]
- CNN처럼 국소 특징을 targeted하게 모델링하지 못함[^1_1]
- 많은 토큰 쌍 간 계산이 불필요하고 중복적[^1_1]

**TCN의 한계**:

- Receptive field 크기 제한으로 전역 관계 모델링에 많은 레이어 필요[^1_1]
- 네트워크 복잡도 증가 및 학습 난이도 상승[^1_1]


### 2.2 제안하는 방법 (수식 포함)

#### Multi-scale Hybrid Decomposition (MHDecomp)

시계열을 trend-cyclical 부분과 seasonal 부분으로 분해:

```math
X_t = \text{mean}(\text{AvgPool}(\text{Padding}(X))_{\text{kernel}_1}, \ldots, \text{AvgPool}(\text{Padding}(X))_{\text{kernel}_n})
```

$X_s = X - X_t$

여기서 $X \in \mathbb{R}^{I \times d}$이며, 다양한 kernel 크기를 사용하여 다중 스케일 패턴을 분리합니다.[^1_1]

#### Trend-Cyclical Prediction Block

두 가지 방법 제안:

**선형 회귀 방식 (MICN-regre)**:
$Y_t^{\text{regre}} = \text{regression}(X_t)$

**평균 방식 (MICN-mean)**:
$Y_t^{\text{mean}} = \text{mean}(X_t)$

여기서 $Y_t \in \mathbb{R}^{O \times d}$는 trend 부분 예측입니다.[^1_1]

#### Seasonal Prediction Block

**Embedding**:
$X_s^{\text{emb}} = \sum(\text{TFE} + \text{PE} + \text{VE}(\text{Concat}(X_s, X_{\text{zero}})))$

여기서 TFE는 시간 특징 인코딩, PE는 위치 인코딩, VE는 값 임베딩입니다.[^1_1]

**Local Module (Downsampling Convolution)**:

$Y_s^{\text{local},i} = \text{Conv1d}(\text{Avgpool}(\text{Padding}(Y_{s,l}))\_{\text{kernel}=i})_{\text{kernel}=i, \text{stride}=i}$

여기서 $i \in \{\frac{I}{4}, \frac{I}{8}, \ldots\}$는 다양한 스케일 크기를 나타냅니다.[^1_1]

**Global Module (Isometric Convolution)**:
$Y_s^{\prime,i} = \text{Norm}(Y_s^{\text{local},i} + \text{Dropout}(\text{Tanh}(\text{IsometricConv}(Y_s^{\text{local},i}))))$

$Y_s^{\text{global},i} = \text{Norm}(Y_{s,l-1} + \text{Dropout}(\text{Tanh}(\text{Conv1dTranspose}(Y_s^{\prime,i})_{\text{kernel}=i})))$

Isometric convolution은 길이 $S$의 시퀀스에 $S-1$의 패딩을 추가하고 kernel 크기 $S$를 사용합니다.[^1_1]

**Merge Operation**:
$Y_s^{\text{merge}} = \text{Conv2d}(Y_s^{\text{global},i}, i \in \{\frac{I}{4}, \frac{I}{8}, \ldots\})$

$Y_s = \text{Norm}(Y_s^{\text{merge}} + \text{FeedForward}(Y_s^{\text{merge}}))$

**최종 예측**:
$Y_{\text{pred}} = Y_s + Y_t$

### 2.3 모델 구조

**전체 아키텍처**:

1. **Input**: 과거 시계열 $X \in \mathbb{R}^{I \times d}$
2. **MHDecomp Block**: 다중 스케일 분해로 $X_t$와 $X_s$ 생성
3. **Trend-cyclical Prediction Block**: 간단한 회귀 또는 평균으로 $Y_t$ 예측
4. **Seasonal Prediction Block**:
    - Embedding layer
    - N개의 MIC (Multi-scale Isometric Convolution) layers
    - 각 MIC layer는 여러 분기(branch)를 포함
    - 각 분기는 Local-Global 모듈로 구성
5. **Output**: $Y_{\text{pred}} \in \mathbb{R}^{O \times d}$

**MIC Layer의 특징**:

- 다중 분기 구조: 각 분기는 서로 다른 스케일 크기 $i$를 사용
- Local module: downsampling으로 국소 특징 압축
- Global module: isometric convolution으로 전역 상관관계 모델링
- Merge: Conv2d로 다양한 패턴 통합[^1_1]


### 2.4 성능 향상

**다변량 예측 (Multivariate)**:

- ETTm2: 기존 FEDformer 대비 12% MSE 감소 (96 예측 길이)
- Electricity: 14% MSE 감소
- Exchange: 31% MSE 감소
- Traffic: 12% MSE 감소
- Weather: 26% MSE 감소
- ILI: 17% MSE 감소
- 전체 평균: **17.2% MSE 개선**[^1_1]

**단변량 예측 (Univariate)**:

- Weather 데이터셋에서 특히 뛰어난 성능:
    - 96 예측: 53% MSE 감소
    - 192 예측: 75% MSE 감소
    - 336 예측: 44% MSE 감소
    - 720 예측: 56% MSE 감소
- 전체 평균: **21.6% MSE 개선**[^1_1]

**계산 효율성**:

- 복잡도: $O(LD^2)$ (Transformer의 $O(L^2)$보다 선형)
- 메모리 사용량: $O(LD^2)$
- 학습 및 추론 속도: Transformer 대비 5-10배 빠름[^1_2][^1_1]


### 2.5 한계

**논문에서 언급된 한계**:

1. **복잡한 Trend 모델링**: ETTm2와 같이 복잡한 trend-cyclical 정보를 가진 데이터에서 단순 선형 회귀가 충분하지 않을 수 있음[^1_1]
2. **하이퍼파라미터 선택**: 다중 스케일 크기 $i$의 수동 선택이 필요하며, 자동화된 방법 부재[^1_1]
3. **극단적 노이즈 민감도**: 10% 이상의 노이즈 주입 시 성능 저하 관찰[^1_1]
4. **주기성이 없는 데이터**: Exchange 데이터셋과 같이 명확한 주기성이 없는 경우 성능 향상 제한적[^1_1]
5. **수동 아키텍처 설계**: AutoML 기법을 사용한 자동 구조 최적화 미적용[^1_1]

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 능력 강화 요인

**1. Multi-scale Branch Structure**

- 다양한 시간적 패턴을 별도로 학습하여 복잡하고 가변적인 데이터에 대한 일반화 능력 향상[^1_1]
- 각 분기가 서로 다른 주기성과 패턴을 포착하여 미지의 데이터에 대한 적응력 증가

**2. Inductive Bias**

- Convolution 연산의 translation equivariance 특성으로 전역 시간적 inductive bias 도입[^1_1]
- Self-attention(요소 간 곱으로 상관관계 획득)보다 kernel 학습을 통한 일반화가 더 효과적

**3. 입력 길이 증가에 따른 성능 향상**

- 입력 길이가 증가할수록 MICN의 예측 성능이 지속적으로 향상[^1_1]
- Transformer 기반 모델은 입력이 길어질수록 반복적인 단기 패턴으로 인해 성능 저하
- 장기 시간 의존성을 효과적으로 포착하는 능력 입증

**4. Robustness Analysis 결과**

- 10%까지의 노이즈에 대해 강건한 성능 유지[^1_1]
- 이상 데이터나 설비 장애로 인한 비정상적 변동에 대한 우수한 대응력


### 3.2 일반화 성능 향상 전략

**1. Cross-domain Transfer Learning**

- 사전 학습된 MICN 모델을 다양한 도메인에 전이하여 zero-shot 또는 few-shot 학습 가능성 탐구
- 최신 연구(Chronos, Time-MoE)처럼 foundation model 접근법 적용 가능[^1_3][^1_4]

**2. Meta-learning 통합**

- 다양한 데이터셋에서 학습하여 새로운 시계열 패턴에 빠르게 적응하는 메타 학습 프레임워크 개발

**3. Uncertainty Quantification**

- 확률적 예측 및 불확실성 추정 기능 추가로 신뢰성 있는 일반화 성능 제공
- TiDE 접근법처럼 MLE 기반 loss function 사용[^1_2]

**4. Automated Architecture Search**

- Neural Architecture Search (NAS)를 활용한 데이터셋별 최적 스케일 크기 및 분기 수 자동 탐색


## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 학계 및 산업계에 미치는 영향

**1. 효율성과 성능의 균형**

- MICN은 Transformer의 높은 복잡도 문제를 해결하면서도 우수한 성능을 달성하여, "Attention is all you need"에 대한 재고찰 촉진[^1_1]
- 최신 연구에서도 simple linear models(DLinear)이나 MLP 기반 모델(TiDE)이 경쟁력 있음을 입증[^1_5][^1_2]

**2. Convolution의 재부상**

- TCN, MICN 등 convolution 기반 접근법의 효과성 재확인
- PatchTST와 같은 패칭 기법과 결합 가능성[^1_5]

**3. Decomposition-based 접근법 검증**

- 시계열 분해(trend-seasonal decomposition)의 중요성 강조
- Autoformer, FEDformer에 이어 분해 기반 방법론의 지속적 발전[^1_1]


### 4.2 향후 연구 시 고려사항

**1. Foundation Models와의 통합**

- **LLM 기반 시계열 예측**: Chronos, TimeGPT와 같은 사전 학습된 foundation models이 부상[^1_6][^1_7][^1_4]
- **고려사항**: MICN의 convolution 구조를 foundation model의 backbone으로 활용하거나, LLM의 semantic understanding과 결합 가능성 탐구

**2. Mixture of Experts (MoE) 아키텍처**

- **최신 동향**: Time-MoE는 24억 파라미터까지 확장하여 SOTA 달성[^1_3]
- **고려사항**: MICN의 다중 분기 구조를 MoE 프레임워크로 확장하여 스케일업

**3. Quantum Machine Learning 융합**

- **새로운 방향**: QuLTSF, QLSTM 등 양자 기계학습 기반 시계열 예측 연구 등장[^1_8][^1_9]
- **고려사항**: MICN의 convolution 연산을 양자 회로로 구현하여 계산 효율성 추가 향상 가능성

**4. Multimodal Time Series Forecasting**

- **트렌드**: 텍스트, 이미지 등 다중 모달 정보를 활용한 예측[^1_10][^1_6]
- **고려사항**: MICN에 cross-modal attention 또는 fusion module 추가하여 다중 소스 데이터 통합

**5. Long-horizon Forecasting 극대화**

- **도전 과제**: 720+ 시간 단계 예측에서의 정확도 유지
- **고려사항**:
    - Hierarchical 또는 recursive 분해 구조 적용(LiNo)[^1_11]
    - Future-Guided Learning 방식으로 분포 변화 대응[^1_12]

**6. 해석 가능성 (Interpretability) 강화**

- **필요성**: 실무 적용 시 모델 결정 과정 이해 필수
- **고려사항**:
    - Attention 시각화처럼 convolution kernel 가중치 분석 도구 개발
    - Feature importance 및 패턴 기여도 정량화

**7. Domain-specific Adaptation**

- **응용 분야**:
    - 금융: 고빈도 거래 데이터 예측[^1_13][^1_14]
    - 의료: 환자 바이탈 사인 모니터링[^1_15]
    - 에너지: 태양광/풍력 발전량 예측[^1_16]
    - 기상: 극한 기상 현상 예측[^1_17]
- **고려사항**: 각 도메인의 특성(비정상성, 이상치, 계절성)에 맞춘 맞춤형 convolution kernel 설계

**8. 실시간 및 온라인 학습**

- **도전 과제**: Concept drift 대응 및 실시간 업데이트[^1_18]
- **고려사항**:
    - Incremental learning 기법 통합
    - 온라인 모델 갱신을 위한 경량화

**9. Probabilistic 및 Distributional Forecasting**

- **트렌드**: Point forecast를 넘어 불확실성 정량화[^1_19]
- **고려사항**:
    - Quantile regression 또는 conformal prediction 통합
    - Ensemble 방법으로 예측 구간 생성

**10. 벤치마크 표준화**

- **현황**: TSPP, ProbTS와 같은 통합 벤치마킹 프레임워크 등장[^1_20][^1_19]
- **고려사항**: 공정한 비교를 위한 표준화된 평가 프로토콜 준수


## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 Transformer 기반 모델

| 모델 | 연도 | 주요 특징 | MICN과의 비교 |
| :-- | :-- | :-- | :-- |
| **Informer**[^1_1] | 2021 | ProbSparse self-attention, $O(L \log L)$ | MICN이 12-31% MSE 개선, 선형 복잡도 우위 |
| **Autoformer**[^1_1] | 2021 | Auto-correlation, 시계열 분해 | MICN의 Local-Global이 Auto-correlation 대비 우수 |
| **FEDformer**[^1_1] | 2022 | 주파수 영역 분해, Fourier transform | MICN이 평균 17.2% MSE 개선 |
| **PatchTST**[^1_5][^1_17] | 2022 | Patching 기법, vanilla attention | 패치 토큰화로 효율성 향상, MICN과 상호보완적 |
| **Former**[^1_17] | 2023-2024 | 강력한 장기 의존성 포착 | Transformer 계열 중 최고 성능, MICN과 유사 |
| **iTransformer**[^1_17] | 2024 | 변수 간 관계 모델링 강화 | 기상 데이터에서 우수, MICN과 비슷한 안정성 |
| **LATST**[^1_21] | 2024 | LSE Attention 메커니즘 | Electricity, Traffic, Weather에서 SOTA, MICN과 경쟁 |

**핵심 통찰**:

- 최근 연구는 "Are Transformers Effective for Time Series?"라는 질문 제기[^1_9][^1_5]
- 단순 선형 모델(DLinear)이 복잡한 Transformer를 능가하는 경우 발견
- MICN은 이러한 논쟁에서 convolution 기반 대안의 효과성 입증


### 5.2 MLP 및 선형 모델

| 모델 | 연도 | 주요 특징 | MICN과의 비교 |
| :-- | :-- | :-- | :-- |
| **DLinear**[^1_5][^1_22] | 2023 | 단순 선형 매핑 | 트렌드 강한 데이터에서 우수, MICN은 비선형성 추가로 향상 |
| **TiDE**[^1_5][^1_2] | 2023 | MLP encoder-decoder, MLE loss | Transformer 대비 5-10배 빠름, MICN과 유사한 효율성 |
| **NLinear** | 2023 | Normalization + Linear | 분포 변화에 강건, MICN의 분해 접근법과 상호보완 |

**핵심 통찰**:

- Google Research의 TiDE 연구는 MLP가 Transformer를 능가할 수 있음을 입증[^1_2]
- MICN은 MLP와 convolution의 장점을 결합하여 더욱 표현력 있는 모델 제공


### 5.3 Foundation Models 및 Pre-trained 모델

| 모델 | 연도 | 주요 특징 | MICN과의 관계 |
| :-- | :-- | :-- | :-- |
| **Chronos**[^1_7][^1_4][^1_23] | 2024 | Decoder-only, zero-shot 예측 | LLM 접근법, MICN은 task-specific 학습 |
| **Time-MoE**[^1_3] | 2024 | 24억 파라미터, MoE 구조 | 대규모 확장성, MICN은 경량화 |
| **Sundial**[^1_24] | 2025 | 대규모 사전 학습 모델 | Foundation model 방향, MICN은 효율적 특화 모델 |
| **TimeGPT** | 2024 | GPT-style time series model | 범용성 강조, MICN은 도메인별 최적화 |

**핵심 통찰**:

- Foundation model이 zero-shot 및 few-shot 학습에서 강점[^1_23][^1_4]
- MICN의 효율적 아키텍처는 리소스 제약 환경에서 유리
- 향후 MICN을 foundation model의 backbone으로 활용 가능성


### 5.4 Hybrid 및 Novel 접근법

| 모델 | 연도 | 주요 특징 | MICN과의 연관성 |
| :-- | :-- | :-- | :-- |
| **VARMAformer**[^1_25] | 2025 | VARMA + Transformer | 통계 모델 통합, MICN의 분해 접근법과 유사 철학 |
| **LiNo**[^1_11] | 2025 | Recursive residual decomposition | 선형/비선형 패턴 분리, MICN의 Local-Global과 공명 |
| **DisenTS**[^1_26] | 2024 | Disentangled channel modeling | 채널별 패턴 분리, MICN의 다중 분기와 유사 |
| **Future-Guided Learning**[^1_12] | 2025 | 예측-피드백 결합 | 적응적 학습, MICN에 통합 가능 |
| **LSTM variants**[^1_15][^1_14][^1_8] | 2024-2025 | RNN 기반, 도메인 특화 | 전통적 접근, MICN이 일반적으로 우수 |

**핵심 통찰**:

- Hybrid 모델이 통계적 방법과 딥러닝 결합으로 해석 가능성과 성능 균형[^1_27]
- MICN의 분해 + convolution 접근법은 이러한 트렌드와 일치


### 5.5 성능 비교 요약 (주요 벤치마크)

**ETTm2 데이터셋 (96 예측 길이, Multivariate)**:

- MICN-regre: **0.179 MSE** ✓
- FEDformer: 0.203 MSE
- Autoformer: 0.255 MSE
- Informer: 0.365 MSE
- PatchTST: ~0.19 MSE (추정)[^1_5]

**Electricity 데이터셋 (96 예측 길이, Multivariate)**:

- MICN-regre: **0.164 MSE** ✓
- FEDformer: 0.193 MSE
- Autoformer: 0.201 MSE
- LATST: ~0.16 MSE (유사)[^1_21]

**Weather 데이터셋 (96 예측 길이, Univariate)**:

- MICN-regre: **0.0029 MSE** ✓
- FEDformer: 0.0062 MSE
- Former: ~0.003 MSE[^1_17]
- iTransformer: ~0.003 MSE[^1_17]


### 5.6 최신 트렌드 및 MICN의 위치

**2024-2025년 주요 트렌드**:

1. **Foundation Models**: 대규모 사전 학습 및 zero-shot 능력[^1_7][^1_6][^1_3]
2. **Hybrid Approaches**: 통계 + ML 결합[^1_25][^1_27]
3. **Efficiency Focus**: 경량 모델 및 빠른 추론[^1_2]
4. **Multimodal Integration**: 텍스트, 이미지 등 통합[^1_6][^1_10]
5. **Probabilistic Forecasting**: 불확실성 정량화[^1_19][^1_2]
6. **Domain Adaptation**: 특정 분야 최적화[^1_14][^1_15][^1_16]
7. **No Champions Paradigm**: 단일 최고 모델 없음, 데이터 특성에 따라 달라짐[^1_22]

**MICN의 강점**:

- ✅ 효율성: 선형 복잡도로 대규모 데이터 처리 가능
- ✅ 성능: 2023년 ICLR 발표 당시 SOTA 달성
- ✅ 해석 가능성: Convolution kernel 분석 가능
- ✅ 일반화: 6개 벤치마크에서 일관된 성능

**MICN의 개선 영역**:

- ❌ Zero-shot 능력: Foundation models 대비 부족
- ❌ 멀티모달: 단일 시계열 데이터만 처리
- ❌ Probabilistic: 불확실성 정량화 미지원
- ❌ AutoML: 하이퍼파라미터 자동 탐색 부재


## 결론

MICN은 2023년 발표 당시 장기 시계열 예측 분야에서 convolution 기반 접근법의 효과성을 입증한 중요한 연구입니다. 선형 복잡도 달성, 다중 스케일 모델링, Local-Global 구조라는 핵심 기여를 통해 Transformer의 효율성 문제를 해결하면서도 우수한 성능을 달성했습니다.[^1_1]

2024-2025년 현재, foundation models와 hybrid 접근법이 부상하는 가운데, MICN의 효율적 아키텍처는 여전히 리소스 제약 환경과 실시간 애플리케이션에서 경쟁력을 유지하고 있습니다. 향후 연구는 MICN의 convolution 구조를 foundation model의 backbone으로 활용하거나, MoE 및 multimodal learning과 결합하여 차세대 시계열 예측 시스템을 구축하는 방향으로 발전할 것으로 전망됩니다.[^1_10][^1_6][^1_3][^1_2]
<span style="display:none">[^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38]</span>

<div align="center">⁂</div>

[^1_1]: 2222_micn_multi_scale_local_and_glo.pdf

[^1_2]: https://research.google/blog/recent-advances-in-deep-long-horizon-forecasting/

[^1_3]: https://arxiv.org/abs/2409.16040

[^1_4]: https://github.com/amazon-science/chronos-forecasting

[^1_5]: https://ar5iv.labs.arxiv.org/html/2304.08424

[^1_6]: https://arxiv.org/html/2508.07697v3

[^1_7]: https://arxiv.org/pdf/2310.10688.pdf

[^1_8]: https://ieeexplore.ieee.org/document/11364891/

[^1_9]: http://arxiv.org/pdf/2412.13769.pdf

[^1_10]: http://arxiv.org/pdf/2405.13522.pdf

[^1_11]: https://arxiv.org/html/2410.17159v2

[^1_12]: https://www.nature.com/articles/s41467-025-63786-4

[^1_13]: https://dl.acm.org/doi/10.1145/3757749.3757774

[^1_14]: https://www.ewadirect.com/proceedings/ace/article/view/22255

[^1_15]: https://www.nature.com/articles/s41598-025-86418-9

[^1_16]: https://ieeexplore.ieee.org/document/10969243/

[^1_17]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12453695/

[^1_18]: https://arxiv.org/pdf/2305.19837.pdf

[^1_19]: http://arxiv.org/pdf/2310.07446.pdf

[^1_20]: https://arxiv.org/pdf/2312.17100.pdf

[^1_21]: https://arxiv.org/html/2410.23749v1

[^1_22]: https://arxiv.org/html/2502.14045v1

[^1_23]: https://ieeexplore.ieee.org/document/11137629/

[^1_24]: https://arxiv.org/html/2510.02729v1

[^1_25]: https://arxiv.org/abs/2509.04782

[^1_26]: http://arxiv.org/pdf/2410.22981.pdf

[^1_27]: https://thestatisticsassignmenthelp.com/blog-details/time-series-analysis-2025-trends-forecasting-applications

[^1_28]: https://arxiv.org/html/2509.23145v1

[^1_29]: https://arxiv.org/abs/2411.05793

[^1_30]: https://arxiv.org/html/2510.14510

[^1_31]: https://rjor.ro/considerations-on-the-efficiency-of-time-series-analysis-in-forecasting-new-influenza-cases-in-the-2024-2025-season/

[^1_32]: https://link.springer.com/10.1007/s00477-025-03098-7

[^1_33]: https://jurnal.unipasby.ac.id/index.php/tibuana/article/view/9942

[^1_34]: https://www.ewadirect.com/proceedings/aemps/article/view/24830

[^1_35]: https://www.influxdata.com/time-series-forecasting-methods/

[^1_36]: https://timeserieslab.com

[^1_37]: https://www.businessresearchinsights.com/market-reports/time-series-forecasting-market-114943

[^1_38]: https://www.bohrium.com/paper-details/forecasting-multistep-daily-stock-prices-for-long-term-investment-decisions-a-study-of-deep-learning-models-on-global-indices/938991595804426292-2472

