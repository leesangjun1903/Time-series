# Cross-LKTCN: Modern Convolution Utilizing Cross-Variable Dependency for Multivariate Time Series Forecasting

## 1. 핵심 주장과 주요 기여

Cross-LKTCN은 다변량 시계열 예측에서 **교차 시간 의존성(cross-time dependency)**과 **교차 변수 의존성(cross-variable dependency)**을 동시에 효과적으로 포착하는 순수 합성곱 기반 모델입니다.[^1_1]

### 주요 기여

- **Depth-wise Large Kernel Convolution 도입**: 컴퓨터 비전의 대형 커널 2D 합성곱에서 영감을 받아, 시계열 예측에 대형 커널 1D 합성곱을 도입하여 수용 영역(receptive field)을 확대하고 장기 시간 의존성 모델링 능력을 개선했습니다.[^1_1]
- **Successive Point-wise Group Convolution FFNs**: 그룹 합성곱의 분리 속성을 활용하여 교차 변수 의존성을 효율적으로 포착하는 두 개의 연속적인 point-wise 그룹 합성곱 피드포워드 네트워크를 제안했습니다.[^1_1]
- **State-of-the-art 성능 달성**: 9개의 실제 벤치마크에서 기존 합성곱 기반 모델 대비 27.4%, 교차 변수 방법 대비 52.3%의 상대적 성능 향상을 달성했습니다.[^1_1]


## 2. 상세 분석

### 해결하고자 하는 문제

기존 다변량 시계열 예측 방법들은 다음과 같은 한계가 있었습니다:[^1_1]

1. **교차 변수 의존성 간과**: Transformer 기반 방법들과 선형 모델들은 주로 cross-time dependency에만 집중하고 변수 간 복잡한 의존성을 충분히 고려하지 못했습니다.
2. **제한된 수용 영역**: 기존 합성곱 기반 모델들은 제한된 수용 영역으로 인해 장기 의존성을 효과적으로 모델링하지 못했습니다.
3. **높은 계산 복잡도**: MTGNN, Crossformer 같은 교차 변수 방법들은 그래프 합성곱이나 어텐션 메커니즘을 사용하여 계산 복잡도가 높았습니다.

### 제안하는 방법

#### Patch-style Embedding Strategy

입력 시계열 $X_{in} \in \mathbb{R}^{M \times L}$을 $N$개의 패치로 분할하고 $D$차원 임베딩 벡터로 변환합니다:[^1_1]

$X_{emb} = \text{Conv1d}(\text{Padding}(X_{in}))_{\text{kernel size}=P, \text{stride}=S, \text{channels}: 1 \rightarrow D}$

여기서 패치 수는 $N = L//S$이며, 패치 크기는 $P$, 스트라이드는 $S$입니다.[^1_1]

#### Cross-LKTCN Block 구조

각 Cross-LKTCN 블록은 residual 방식으로 구성됩니다:[^1_1]

$Z_{i+1} = \text{Block}(Z_i) + Z_i$

**1) Depth-wise Large Kernel Convolution** (Cross-time Dependency 포착):[^1_1]

$Z^{time1}\_i = \text{BN}(\text{DW1Conv1d}(Z_i))_{\text{kernel size=large size, groups}=(M \times D)}$

$Z^{time2}\_i = \text{BN}(\text{DW2Conv1d}(Z_i))_{\text{kernel size=small size, groups}=(M \times D)}$

$Z^{time}_i = Z^{time1}_i + Z^{time2}_i$

Structural Re-parameterization 기법을 활용하여 대형 커널의 최적화 문제를 해결합니다.[^1_1]

**2) Successive Point-wise Group Convolution FFNs** (Cross-variable Dependency 포착):[^1_1]

첫 번째 ConvFFN (각 변수의 독립적인 표현 학습):

$Z^{variable1'}\_i = \text{Drop}(\text{PW1}\_1\text{Conv1d}(Z^{time}\_i))_{\text{kernel size}=1, \text{groups}=M}$

$Z^{variable1'}_i = \text{GELU}(Z^{variable1'}_i)$

$Z^{variable1}\_i = \text{Drop}(\text{PW1}\_2\text{Conv1d}(Z^{variable1'}\_i))_{\text{groups}=M}$

두 번째 ConvFFN (교차 변수 의존성 포착):

$Z^{variable2'}\_i = \text{Drop}(\text{PW2}\_1\text{Conv1d}(Z^{variable2}\_i))_{\text{kernel size}=1, \text{groups}=D}$

$Z^{variable2'}_i = \text{GELU}(Z^{variable2'}_i)$

$Z^{variable}\_i = \text{Drop}(\text{PW2}\_2\text{Conv1d}(Z^{variable2'}\_i))_{\text{groups}=D}$

채널 수 변화: $(D \times M) \rightarrow (r \times D \times M) \rightarrow (D \times M)$, 여기서 $r$은 FFN ratio입니다.[^1_1]

#### 최종 예측

$\hat{X} = \text{Head}(\text{Flatten}(Z))$

여기서 $\hat{X} \in \mathbb{R}^{M \times T}$는 예측 길이 $T$의 출력입니다.[^1_1]

### 모델 구조

Cross-LKTCN의 전체 구조는 다음과 같이 구성됩니다:[^1_1]

1. **RevIN**: Distribution shift 완화를 위한 정규화
2. **Patch-style Embedding**: 지역성 강화 및 의미 정보 집계
3. **Backbone**: $K$개의 Cross-LKTCN 블록 스택
4. **Linear Head**: 최종 예측 생성

### 성능 향상

9개의 실제 데이터셋에서 실험한 결과:[^1_1]


| 비교 대상 | MSE 감소율 | MAE 감소율 |
| :-- | :-- | :-- |
| 합성곱 기반 모델 (MICN, SCINet) | 27.4% | 15.3% |
| 교차 변수 방법 (MTGNN, Crossformer) | 52.3% | 33.5% |

**주요 데이터셋 결과** (예측 길이 96):[^1_1]

- Electricity: MSE 0.129 (PatchTST 대비 동등)
- ILI: MSE 1.347 (PatchTST 1.319 대비 근소한 차이)
- Traffic: MSE 0.373 (PatchTST 0.360 대비 경쟁력 있는 성능)
- Exchange: MSE 0.080 (PatchTST 0.093 대비 우수)

**계산 복잡도 비교**:[^1_1]


| 모델 | Cross-time | Cross-variable |
| :-- | :-- | :-- |
| Cross-LKTCN | $O(\frac{L}{S}MD)$ | $O(\frac{L}{S}MD^2 + \frac{L}{S}DM^2)$ |
| Crossformer | $O(\frac{L^2}{S^2}MD)$ | $O(M^2\frac{L}{S}D)$ |
| MICN | $O(LD^2)$ | - |

Cross-LKTCN은 선형 복잡도를 유지하면서 패치 임베딩의 $\frac{1}{S}$ 계수로 실질적인 계산량을 크게 감소시켰습니다.[^1_1]

### 한계

논문에서 명시적으로 언급된 한계는 제한적이지만, 다음과 같은 잠재적 한계를 추론할 수 있습니다:

1. **주기성이 불명확한 데이터**: Exchange 데이터셋에서 SCINet이 더 우수한 성능을 보인 것처럼, 계층적 구조를 가진 모델이 다중 스케일 정보를 더 잘 활용할 수 있는 경우가 있습니다.[^1_1]
2. **매우 긴 시퀀스**: 커널 크기를 51까지 확장했지만, 극도로 긴 시퀀스에서는 여전히 제한이 있을 수 있습니다.
3. **하이퍼파라미터 민감도**: 커널 크기, 패치 크기, 그룹 수 등 여러 하이퍼파라미터의 최적화가 필요합니다.

## 3. 모델의 일반화 성능 향상 가능성

### 입력 길이에 따른 성능

Cross-LKTCN은 입력 길이가 증가함에 따라 지속적으로 성능이 향상되는 것을 보여주었습니다. 이는 모델이 더 긴 히스토리에서 유용한 정보를 효과적으로 추출하고 장기 의존성을 포착할 수 있음을 나타냅니다. 반면, 일부 Transformer 기반 모델들은 반복되는 단기 패턴으로 인해 입력 길이 증가 시 성능 저하를 겪었습니다.[^1_1]

### 커널 크기의 영향

Ablation study 결과, 커널 크기를 3에서 51로 증가시킬 때 일관된 성능 향상이 관찰되었습니다:[^1_1]

- ILI 데이터셋 (예측 길이 24): MSE 1.906 → 1.687 → 1.347
- ETTh1 데이터셋 (예측 길이 96): MSE 0.381 → 0.367 → 0.368
- Electricity 데이터셋 (예측 길이 96): MSE 0.143 → 0.133 → 0.130

이는 대형 커널이 수용 영역을 효과적으로 확대하여 교차 시간 의존성을 더 잘 포착함을 의미합니다.[^1_1]

### 교차 변수 의존성 포착의 효과

Successive point-wise group convolution FFNs의 ablation study 결과:[^1_1]

1. **M Groups + D Groups** (제안 방법): 최고 성능
2. **M Groups only**: 변수 간 의존성을 포착하지 못해 성능 저하
3. **D Groups only**: 변수별 깊은 표현 학습 부족으로 성능 저하
4. **No Group**: 분리된 모델링 불가로 가장 낮은 성능

이러한 설계는 모델이 다양한 데이터셋에서 변수 간 관계를 적응적으로 학습할 수 있도록 하여 일반화 성능을 향상시킵니다.[^1_1]

### Channel-independence vs. Cross-variable 접근

Cross-LKTCN은 channel-independent 방법(PatchTST, DLinear)과 달리 명시적으로 교차 변수 의존성을 모델링하여, 변수 간 상호작용이 중요한 데이터셋에서 더 나은 일반화 능력을 보입니다. 특히 Exchange 데이터셋처럼 주기성이 불명확한 경우, 교차 변수 정보를 추가로 활용하여 PatchTST보다 우수한 성능을 달성했습니다.[^1_1]

## 4. 향후 연구에 미치는 영향과 고려사항

### 연구에 미치는 영향

**1) 합성곱의 재조명**

Cross-LKTCN은 시계열 예측 분야에서 합성곱 기반 모델의 잠재력을 재확인시켰습니다. 대형 커널과 modern architectural design을 결합하여 Transformer와 경쟁할 수 있는 성능을 달성함으로써, 합성곱이 여전히 효과적인 선택지임을 입증했습니다.[^1_1]

**2) 효율성과 성능의 균형**

선형 계산 복잡도를 유지하면서 state-of-the-art 성능을 달성한 것은 실용적인 응용에서 중요한 의미를 가집니다. 특히 자원이 제한된 환경이나 실시간 예측이 필요한 경우에 유용합니다.[^1_1]

**3) 교차 변수 의존성의 중요성**

교차 변수 방법들 대비 52.3%의 MSE 감소를 달성함으로써, 효율적인 교차 변수 의존성 모델링이 다변량 시계열 예측에서 핵심적임을 보여주었습니다.[^1_1]

### 향후 연구 시 고려사항

**1) 커널 크기 최적화**

- **동적 커널 크기**: 데이터셋의 특성에 따라 커널 크기를 자동으로 조정하는 메커니즘 연구
- **계층적 커널**: 다중 스케일 패턴을 더 효과적으로 포착하기 위한 다양한 커널 크기의 조합

**2) 교차 변수 모델링 개선**

- **Sparse Cross-variable Interaction**: 모든 변수 간 상호작용이 아닌 중요한 관계만 선택적으로 모델링
- **시간 변화 변수 관계**: 시간에 따라 변하는 변수 간 관계를 동적으로 포착

**3) 확장성 연구**

- **대규모 변수**: 수백 개 이상의 변수를 가진 시계열에서의 성능과 효율성
- **초장기 예측**: 예측 길이가 매우 긴 경우(T > 1000)의 모델 적응

**4) 도메인 특화 개선**

- **불규칙 샘플링**: 비균일하게 샘플링된 시계열 데이터 처리
- **결측값 처리**: 교차 변수 의존성을 활용한 결측값 보완 메커니즘

**5) 이론적 분석**

- 왜 대형 커널이 시계열에서 효과적인지에 대한 이론적 근거
- 그룹 합성곱의 표현력(expressiveness)에 대한 형식적 분석


## 5. 2020년 이후 관련 최신 연구 비교 분석

### Transformer 기반 방법의 진화

**PatchTST (2022)**[^1_2][^1_3]

- **핵심 아이디어**: 시계열을 subseries-level 패치로 분할하고 channel-independence 전략 사용
- **Cross-LKTCN과의 비교**: PatchTST는 각 변수를 독립적으로 처리하여 교차 변수 의존성을 간과하지만, Cross-LKTCN은 이를 명시적으로 모델링합니다. 성능은 대체로 비슷하지만, 교차 변수 정보가 중요한 데이터셋에서 Cross-LKTCN이 우위를 보입니다.[^1_1]

**Crossformer (2023)**[^1_4]

- **핵심 아이디어**: Cross-dimension dependency를 위한 two-stage attention 메커니즘
- **Cross-LKTCN과의 비교**: Crossformer는 $O(\frac{L^2}{S^2}MD)$의 이차 복잡도를 가지는 반면, Cross-LKTCN은 선형 복잡도로 52.3% 더 나은 MSE를 달성합니다.[^1_1]

**iTransformer (2024)**[^1_5]

- **핵심 아이디어**: 독립적인 시계열을 토큰으로 임베딩하여 self-attention으로 다변량 상관관계 포착
- **특징**: Channel을 토큰으로 사용하는 혁신적인 접근으로 SOTA 성능 달성
- **Cross-LKTCN과의 차이**: iTransformer는 attention 기반이지만, Cross-LKTCN은 순수 합성곱으로 더 효율적입니다


### 선형 및 MLP 기반 방법

**DLinear (2022)**[^1_1]

- **핵심 아이디어**: 단순한 선형 모델이 복잡한 Transformer를 능가할 수 있음을 입증
- **Cross-LKTCN과의 비교**: Cross-LKTCN은 대부분의 데이터셋에서 DLinear를 능가하며, 특히 교차 변수 의존성이 중요한 경우 뚜렷한 우위를 보입니다.[^1_1]

**TSMixer (2023)**[^1_6]

- **핵심 아이디어**: Multi-layer perceptrons를 스택한 all-MLP 아키텍처
- **특징**: 간단하면서도 효과적인 구조로 빠른 추론 속도

**TTM (Tiny Time Mixers) (2024)**[^1_7]

- **핵심 아이디어**: 경량 pre-trained 모델(1M 파라미터부터)로 zero/few-shot 예측 가능
- **혁신**: Adaptive patching, resolution prefix tuning, multi-level modeling
- **의의**: CPU만으로도 실행 가능한 실용적 솔루션 제시


### 교차 변수 의존성 모델링

**Leddam (2024)**[^1_8]

- **핵심 아이디어**: Learnable decomposition 전략과 dual attention module (inter-series dependencies와 intra-series variations 동시 포착)
- **성능**: 11.87%~48.56% MSE 감소 달성
- **Cross-LKTCN과의 유사성**: 둘 다 교차 변수와 교차 시간 의존성을 모두 중요시하지만, Leddam은 attention 기반, Cross-LKTCN은 합성곱 기반

**TimeCNN (2024)**[^1_5]

- **핵심 아이디어**: Timepoint-independent 접근으로 각 시간 포인트마다 독립적인 합성곱 커널 사용
- **성능**: 계산 요구량 60.46% 감소, 파라미터 57.50% 감소, 추론 속도 3~4배 향상
- **Cross-LKTCN과의 비교**: 둘 다 합성곱 기반이지만, TimeCNN은 시간 포인트별 독립성을 강조하고, Cross-LKTCN은 대형 커널로 장기 의존성을 포착

**DUET (2024)**[^1_9][^1_10]

- **핵심 아이디어**: Temporal과 channel 차원에서 dual clustering 수행
- **특징**: TCM (Temporal Clustering Module)으로 이질적 시간 패턴 처리, CCM (Channel Clustering Module)으로 주파수 도메인에서 채널 관계 포착
- **실험**: 10개 응용 도메인의 25개 데이터셋에서 SOTA 달성

**SOFTS (2024)**[^1_11][^1_12]

- **핵심 아이디어**: Series-core fusion with STAR (STar Aggregate-Redistribute) 모듈
- **혁신**: 중앙화된 전략으로 모든 시리즈를 global core representation으로 집계 후 재분배
- **특징**: 선형 복잡도로 SOTA 성능 달성


### 하이브리드 및 특수 접근법

**TimeCMA (2024)**[^1_13]

- **핵심 아이디어**: LLM-empowered multivariate time series forecasting via cross-modality alignment
- **혁신**: Dual-modality encoding으로 disentangled yet weak time series embeddings와 entangled yet robust prompt embeddings를 결합
- **의의**: LLM을 시계열 예측에 효과적으로 통합

**HDMixer (2024)**[^1_14]

- **핵심 아이디어**: Length-Extendable Patcher (LEP)와 Hierarchical Dependency Explorer (HDE)
- **특징**: 순수 MLP 기반으로 패치 내 단기, 패치 간 장기, 변수 간 복잡한 상호작용을 모두 모델링

**DisenTS (2024)**[^1_15]

- **핵심 아이디어**: Disentangled channel evolving pattern modeling
- **특징**: 다변량 시계열 데이터 내 다양한 패턴을 분리된 방식으로 모델링


### 최신 트렌드 (2025)

**GLinear (2025)**[^1_16]

- **핵심 아이디어**: 주기적 패턴을 활용한 data-efficient 아키텍처
- **의의**: 단순성과 정교함의 균형 추구

**MTS-UNMixers (2024)**[^1_17]

- **핵심 아이디어**: Channel-time dual unmixing으로 critical bases와 coefficients로 분해
- **특징**: 역사적 시리즈와 미래 시리즈 간 명시적 매핑 구축

**Gateformer (2025)**[^1_18]

- **핵심 아이디어**: Gating mechanism으로 temporal dependency embeddings와 global temporal embeddings 통합
- **특징**: Variate-wise representation 생성으로 intra-series와 inter-series 의존성 모두 포착


### 종합 비교 분석

| 접근 방식 | 대표 모델 | 주요 특징 | 계산 복잡도 | Cross-variable 처리 |
| :-- | :-- | :-- | :-- | :-- |
| **Transformer** | PatchTST | Patching + Channel-independence | $O(L^2)$ | 간접적 |
| **Transformer** | Crossformer | Two-stage attention | $O(L^2M^2)$ | 명시적 |
| **Convolution** | Cross-LKTCN | Large kernel + Group conv | $O(LMD)$ | 명시적 |
| **Convolution** | TimeCNN | Timepoint-independent kernel | 낮음 | 명시적 |
| **MLP** | TSMixer | Stacked MLPs | $O(LD^2)$ | 혼합 |
| **MLP** | TTM | Pre-trained lightweight | 매우 낮음 | Multi-level |
| **Linear** | DLinear | Simple decomposition | $O(L)$ | 없음 |
| **Hybrid** | DUET | Dual clustering | - | 명시적 |
| **Hybrid** | SOFTS | Series-core fusion | $O(LMD)$ | 중앙화 전략 |
| **LLM-based** | TimeCMA | Cross-modality alignment | 높음 | LLM-empowered |

### 연구 방향성 분석

**2020-2022**: Transformer 우세 시기

- Informer, Autoformer, FEDformer 등이 attention 메커니즘으로 장기 의존성 포착에 집중

**2022-2023**: 단순성의 역습

- DLinear이 단순한 선형 모델이 복잡한 Transformer를 능가할 수 있음을 입증
- "Are Transformers Effective for Time Series Forecasting?" 논쟁 촉발

**2023-2024**: 교차 변수 의존성의 재발견

- Cross-LKTCN, Crossformer, Leddam 등이 교차 변수 모델링의 중요성 강조
- 합성곱 기반 방법의 부활 (대형 커널 활용)

**2024-2025**: 다양화와 전문화

- LLM 통합 (TimeCMA)
- 경량 pre-trained 모델 (TTM)
- Clustering 기반 접근 (DUET)
- Quantum ML 적용 시도 (QuLTSF)[^1_19]


### Cross-LKTCN의 위치와 의의

Cross-LKTCN은 다음과 같은 점에서 중요한 기여를 했습니다:

1. **합성곱의 현대화**: 컴퓨터 비전의 대형 커널 기법을 시계열에 성공적으로 적용하여 합성곱의 가능성을 재조명[^1_1]
2. **효율성과 성능의 최적 균형**: 선형 복잡도로 Transformer와 경쟁하는 성능 달성[^1_1]
3. **교차 변수 의존성의 효율적 구현**: 그룹 합성곱을 활용한 novel한 접근으로 기존 교차 변수 방법들을 크게 능가[^1_1]
4. **후속 연구 촉발**: TimeCNN 등 합성곱 기반 교차 변수 모델링 연구의 발판 마련

### 향후 연구 방향 제언

1. **Foundation Models**: TTM처럼 대규모 데이터로 pre-training한 합성곱 기반 foundation model 개발
2. **Adaptive Architecture**: 데이터 특성에 따라 커널 크기, 그룹 수 등을 자동으로 조정하는 Neural Architecture Search (NAS) 적용
3. **Multimodal Integration**: TimeCMA처럼 다른 modality (텍스트, 이미지)와의 결합 탐구
4. **Theoretical Understanding**: 대형 커널이 시계열에서 효과적인 이유에 대한 수학적 분석 및 일반화 이론 연구
5. **Real-world Robustness**: 불규칙 샘플링, 결측값, distribution shift 등 실제 환경의 도전 과제 해결[^1_20]

Cross-LKTCN은 2023년 시계열 예측 분야에서 합성곱 기반 방법의 부활을 이끈 중요한 연구이며, 효율성과 성능을 동시에 추구하는 현대 시계열 예측 모델 설계의 좋은 예시를 제공합니다.[^1_1]
<span style="display:none">[^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37]</span>

<div align="center">⁂</div>

[^1_1]: 2306.02326v1.pdf

[^1_2]: https://github.com/yuqinie98/PatchTST

[^1_3]: https://github.com/PatchTST/PatchTST

[^1_4]: https://www.semanticscholar.org/paper/fb45d31cc89207aec392dbac8908cc24db2df871

[^1_5]: https://arxiv.org/html/2410.04853v1

[^1_6]: https://arxiv.org/pdf/2303.06053.pdf

[^1_7]: https://arxiv.org/abs/2401.03955

[^1_8]: https://arxiv.org/abs/2402.12694

[^1_9]: https://dl.acm.org/doi/10.1145/3690624.3709325

[^1_10]: https://arxiv.org/html/2412.10859v1

[^1_11]: https://arxiv.org/abs/2404.14197

[^1_12]: https://proceedings.neurips.cc/paper_files/paper/2024/file/754612bde73a8b65ad8743f1f6d8ddf6-Paper-Conference.pdf

[^1_13]: https://ojs.aaai.org/index.php/AAAI/article/view/34067

[^1_14]: https://ojs.aaai.org/index.php/AAAI/article/view/29155

[^1_15]: http://arxiv.org/pdf/2410.22981.pdf

[^1_16]: http://arxiv.org/pdf/2501.01087.pdf

[^1_17]: https://arxiv.org/html/2411.17770v1

[^1_18]: https://arxiv.org/html/2505.00307v2

[^1_19]: http://arxiv.org/pdf/2412.13769.pdf

[^1_20]: https://arxiv.org/html/2506.08660v1

[^1_21]: https://ieeexplore.ieee.org/document/10729516/

[^1_22]: https://ejournal.unesa.ac.id/index.php/mathunesa/article/view/59059

[^1_23]: https://ieeexplore.ieee.org/document/10485787/

[^1_24]: https://arxiv.org/pdf/2109.06489.pdf

[^1_25]: https://arxiv.org/pdf/2502.03571.pdf

[^1_26]: https://arxiv.org/html/2402.12694v3

[^1_27]: https://arxiv.org/html/2505.20774v1

[^1_28]: https://arxiv.org/html/2408.11306v2

[^1_29]: https://arxiv.org/html/2502.10721v1

[^1_30]: https://arxiv.org/html/2410.18318v1

[^1_31]: https://arxiv.org/abs/2306.02326

[^1_32]: https://arxiv.org/pdf/2508.05454.pdf

[^1_33]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11059412/

[^1_34]: https://www.ibm.com/think/tutorials/sktime-multivariate-time-series-forecasting

[^1_35]: https://ar5iv.labs.arxiv.org/html/2306.02326

[^1_36]: https://neurips.cc/virtual/2024/poster/96390

[^1_37]: http://arxiv.org/abs/2306.02326

