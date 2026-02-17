# WinNet:time series forecasting with a window-enhanced period extracting and interacting

## 1. 핵심 주장 및 주요 기여

WinNet은 CNN 기반의 장기 시계열 예측 모델로, Transformer 및 MLP 기반 모델들의 한계를 극복하고 SOTA 성능을 달성합니다. 이 논문의 핵심 기여는 다음과 같습니다:[^1_1]

- **단일 합성곱 계층 백본**: 하나의 합성곱 계층만으로 예측 네트워크를 구성하여 메모리와 계산 복잡도를 크게 감소시킴
- **주기 창(Periodic Window) 기반 재구성**: FFT로 추출한 다중 짧은 주기들의 최소공배수를 사용하여 시계열을 2D 텐서로 재구성
- **주기-추세와 진동 항의 상관관계 활용**: 기존 모델들이 이 두 항을 독립적으로 처리한 것과 달리, DCB를 통해 주기적 이웃 정보를 집계

성능 면에서 WinNet은 TimesNet 대비 다변량 시계열에서 MSE 18.5%, MAE 12.0% 개선을 달성했습니다.[^1_1]

## 2. 문제 정의 및 제안 방법

### 해결하고자 하는 문제

시계열 예측은 시스템의 미래 상태가 시간에 따라 변화하고 불확실성을 가지기 때문에 매우 어려운 과제입니다. 기존 방법들의 주요 문제점은:[^1_1]

- **Transformer 기반 모델**: 높은 계산 비용과 주기성 포착의 어려움
- **DLinear (MLP 기반)**: 단순한 선형 변환으로는 복잡한 주기 패턴 포착 한계
- **기존 분해 방법**: 추세와 계절 항을 독립적으로 처리하여 상관관계 손실


### 제안하는 방법

#### Inter-Intra Period Encoder (I2PE)

1D 시계열을 주기 창 크기에 따라 2D 텐서로 변환합니다:[^1_1]

$$
\hat{X}_{1D} = \text{Permute}(\text{RevIN}(X_{1D}))
$$

$$
X^{\text{row}}_{2D} = \text{Reshape}(\text{Linear}(\hat{X}_{1D}))
$$

$$
X^{\text{col}}_{2D} = \text{Transpose}(X^{\text{row}}_{2D})
$$

여기서 $X_{1D} \in \mathbb{R}^{sl \times c}$는 원본 시퀀스, $X^{\text{row}}\_{2D} \in \mathbb{R}^{c \times n \times w}$는 행이 주기 창을 나타내는 intra-period, $X^{\text{col}}_{2D} \in \mathbb{R}^{c \times w \times n}$는 열이 주기 창을 나타내는 inter-period입니다.[^1_1]

#### Two-Dimensional Period Decomposition (TDPD)

기존 1D 분해와 달리 2D 텐서에 대한 주기 분해를 수행합니다:[^1_1]

$$
X_{\text{period}} = \text{AvgPool2D}_{k \times k}(\text{TrendPadding}(X_{2D}))
$$

$$
X_{\text{osc}} = X - X_{\text{period}}
$$

AvgPool2D는 주기 창 내부의 추세(intra-correlation)와 인접 창들 간의 장기 주기 변화(inter-correlation)를 동시에 추출합니다.[^1_1]

#### Decomposition Correlation Block (DCB)

주기-추세 항과 진동 항의 국소 상관관계를 합성곱 커널로 학습합니다:[^1_1]

$$
X^{CI}_{\text{period}}, X^{CI}_{\text{osc}} = \text{CI}(X_{\text{period}}), \text{CI}(X_{\text{osc}})
$$

$$
X^{CI}_{\text{input}} = \text{Concat}(X^{CI}_{\text{period}}, X^{CI}_{\text{osc}})
$$

$$
\hat{X}^{CI}_{\text{output}} = \text{Dropout}(\text{Sigmoid}(\text{Conv2D}(X^{CI}_{\text{input}})))
$$

$$
\hat{X}^{CI}_{\text{output}} = \text{CA}(\hat{X}^{CI}_{\text{output}})
$$

여기서 CI(·)는 채널 독립성 전략, CA(·)는 채널 집계를 의미합니다.[^1_1]

#### Series Decoder

Inter-period와 intra-period를 융합하여 전역 주기성을 추출합니다:[^1_1]

$$
\hat{X}^{\text{fusion}}_{i,j} = \hat{X}^{\text{row}}_{i,j} + \hat{X}^{\text{col}}_{j,i}, \quad i, j \in (1, 2, ..., w)
$$

$$
\hat{X}^{\text{res}}_{i,j} = \hat{X}_{i,j} + X^{\text{row}}_{i,j}
$$

$$
\hat{X}^{\text{final}} = \text{Permute}(\text{Linear}(\text{Reshape}(\hat{X}^{\text{res}}_{2D})))
$$

### 모델 구조

전체 아키텍처는 다음과 같은 순차적 흐름으로 구성됩니다:[^1_1]

1. **입력 전처리**: RevIN 정규화 후 선형 변환
2. **I2PE 블록**: 2D 텐서 변환 (intra-period와 inter-period 생성)
3. **TDPD 블록** (병렬): 각각에 대해 주기-추세 및 진동 항 분해
4. **DCB 블록** (병렬): 분해된 항들의 상관관계 학습
5. **Series Decoder**: 최종 예측 생성

## 3. 성능 향상 및 한계

### 성능 향상

**다변량 예측** (입력 길이 512):[^1_1]

- ETTm1: MSE 0.345 (TimesNet 대비 15.2% 개선)
- Traffic: MSE 0.417 (TimesNet 대비 32.3% 개선)
- Electricity: MSE 0.159 (TimesNet 대비 20.5% 개선)

**단변량 예측** (입력 길이 336):[^1_1]

- PatchTST 대비: MSE 8.2%, MAE 5.0% 개선
- TimesNet 대비: MSE 12.3%, MAE 8.1% 개선
- DLinear 대비: MSE 18.9%, MAE 13.1% 개선

**계산 효율성** (Traffic 데이터셋 기준):[^1_1]

- FLOPs: 851.3K (TimesNet의 0.026%)
- 파라미터: 830.8K (TimesNet의 0.18%)
- 학습 시간: 17ms/iter (TimesNet의 3.5%)


### 모델의 일반화 성능 향상 가능성

**입력 길이 강건성**:
Ablation study에서 입력 길이 {24, 48, 96, 192, 336, 512, 720}에 대해 실험한 결과, WinNet은 다양한 look-back window에서 안정적인 성능을 보였습니다. 특히 긴 입력 길이에서 더 우수한 성능을 나타냈는데, 이는 주기 창 방식이 장기 주기성을 효과적으로 포착하기 때문입니다.[^1_1]

**도메인 간 일반화**:
9개의 벤치마크 데이터셋(에너지, 교통, 경제, 날씨, 전력, 질병)에서 일관되게 SOTA 성능을 달성했습니다. 이는 제안된 주기 추출 메커니즘이 도메인에 구애받지 않고 작동함을 시사합니다.[^1_1]

**주기 패턴 자동 추출**:
FFT 기반 주기 추출과 MLP 변환을 통해 데이터의 주기를 자동으로 학습합니다. 이는 수동 하이퍼파라미터 튜닝 없이도 다양한 데이터셋에 적용 가능함을 의미합니다.[^1_1]

### 한계점

**주기성이 약한 데이터**:
모델이 FFT를 통한 주기 추출에 의존하므로, 명확한 주기 패턴이 없는 비정상적이거나 카오틱한 시계열에서는 성능이 제한될 수 있습니다.[^1_1]

**창 크기 선택의 민감성**:
Ablation study에서 주기 창 크기 {18, 24, 32}에 대한 실험 결과, 최적 창 크기 선택이 성능에 영향을 미칩니다. ETTm1 데이터에서 크기 24가 가장 우수했으나, 다른 크기도 경쟁력 있는 결과를 보였습니다.[^1_1]

**매우 긴 예측 구간**:
720 스텝 예측에서는 여전히 개선의 여지가 있으며, 특히 ILI 같은 작은 데이터셋에서 더 두드러집니다.[^1_1]

**해석 가능성**:
CNN 기반 특징 추출이 Attention 메커니즘보다 해석하기 어려울 수 있습니다.

## 4. 향후 연구에 미치는 영향 및 고려사항

### 향후 연구에 미치는 영향

**CNN 재조명**:
WinNet은 Transformer와 MLP 지배적인 환경에서 CNN의 잠재력을 재확인시켰습니다. 단일 합성곱 계층만으로도 SOTA 성능을 달성할 수 있음을 보여, 효율성과 성능의 균형을 추구하는 연구에 새로운 방향을 제시합니다.[^1_1]

**분해 전략의 진화**:
2D 주기 분해(TDPD)는 기존 1D 분해 방법의 한계를 극복했습니다. 이는 시계열 분해 연구에서 다차원 접근의 중요성을 강조하며, 향후 더 복잡한 분해 메커니즘 개발의 기반이 됩니다.[^1_1]

**상관관계 모델링**:
주기-추세와 진동 항 간의 lag-correlation이 강하고 주기적 패턴을 가진다는 발견은 중요한 통찰입니다. DCB는 이 두 항을 독립적으로 처리하던 기존 방식(DLinear, MICN)을 넘어서며, 향후 다항 상호작용 모델링 연구에 영향을 줄 것입니다.[^1_1]

### 2020년 이후 관련 최신 연구 비교 분석

#### Transformer 기반 모델들

**PatchTST (2023)**:[^1_2][^1_3]

- Patching 메커니즘으로 시계열을 세그먼트화하여 채널 독립성 유지
- WinNet과 비교: PatchTST는 전역 attention에 의존하나, WinNet은 국소 합성곱으로 효율성 우위 (FLOPs 51.1G vs 851.3K)[^1_1]
- 성능: WinNet이 장기 예측에서 평균 8.2% MSE 개선[^1_1]

**TimesNet (2022)**:[^1_4]

- 2D 변환과 Inception 블록을 사용한 다중 주기 추출
- WinNet과 유사점: 2D 변환 사용, 주기성 중시
- WinNet 차이점: 단일 합성곱 vs 복잡한 Inception, TDPD vs 단순 변환
- WinNet이 계산 복잡도에서 압도적 우위 (3240.7G FLOPs vs 851.3K)[^1_1]

**Time-LLM (2023)**:[^1_5]

- LLM을 시계열 예측에 재프로그래밍하여 few-shot 및 zero-shot 학습 우수
- 다른 패러다임: 사전학습된 언어 모델 활용 vs WinNet의 특화된 CNN
- 장점: 일반화 능력, 단점: 계산 비용

**Peri-midFormer (2024)**:[^1_6]

- 주기적 피라미드 구조로 계층적 주기 표현
- WinNet과의 공통점: 주기성을 명시적으로 모델링
- 차이점: Transformer 기반 vs CNN 기반


#### MLP/Linear 기반 모델들

**DLinear (2023)**:[^1_1]

- 단순 선형 계층으로 Transformer 능가
- WinNet 대비: 주기 상관관계 미활용, 복잡한 패턴 포착 한계
- WinNet이 11.9% MSE 개선 (단입력), 18.9% 개선 (장입력 단변량)[^1_1]

**TSMixer (2023)**:[^1_7]

- 경량 MLP-Mixer로 Transformer 수준 정확도, 80% 적은 파라미터
- WinNet과 유사한 효율성 추구 철학
- 차이점: 채널 의존적 전략 vs WinNet의 주기 중심 접근


#### 주기성 관련 연구

**Dynamic Periodic Event Graphs (2025)**:[^1_8]

- 시계열 분해 + 주파수 분석으로 주기 자동 추출
- WinNet과 유사: 자동화된 주기 추출
- 응용 차이: 그래프 기반 표현 vs 직접 예측

**Unsupervised Periodic Detection (2024)**:[^1_9]

- 레이블 없이 주기 패턴 탐지, 45-50% 성능 향상
- WinNet과 보완적: 전처리 단계로 활용 가능


#### 최신 트렌드 (2024-2025)

**Foundation Models**:[^1_10][^1_11]

- 대규모 사전학습 모델이 zero-shot 예측에서 감독학습 모델 수준 도달
- WinNet의 위치: 특화된 작업에서 여전히 경쟁력, 효율성 우위

**Mamba 기반 모델들**:[^1_12][^1_13]

- Bi-Mamba+, ISMRNN: RNN과 Mamba 결합으로 장기 의존성 포착
- 새로운 아키텍처 경쟁, WinNet은 단순성과 효율성으로 차별화


### 향후 연구 시 고려사항

**1. 주기-추세와 진동 항의 상호작용 심화**:
논문이 명시한 대로, 향후 연구는 이 두 항을 분리 학습하는 대신 상호작용을 더 깊이 탐구해야 합니다. 예를 들어:[^1_1]

- Attention 메커니즘을 추가하여 동적 가중치 학습
- 비선형 상호작용 모델링
- 시간에 따른 상관관계 변화 포착

**2. 적응적 주기 창 크기**:
현재 FFT로 고정된 주기 창을 사용하나, 데이터 분포 변화에 따라 동적으로 조정하는 메커니즘이 필요합니다.[^1_1]

**3. 분포 변화(Distribution Shift) 대응**:
실세계 시계열은 비정상성을 보이므로, RevIN 외에 추가적인 정규화 또는 도메인 적응 기법 통합이 필요합니다.

**4. 멀티모달 통합**:
Beyond Trend and Periodicity (2024) 연구처럼 텍스트 단서를 통합하는 방향성 고려.[^1_14]

**5. 하이브리드 아키텍처**:
CNN의 국소성과 Transformer의 전역 모델링을 결합한 하이브리드 접근이 유망합니다. WinNet의 효율적인 CNN 블록을 다른 모델의 특정 계층에 통합하는 연구가 가능합니다.

**6. Few-shot 및 Zero-shot 학습**:
Time-LLM의 강점을 WinNet에 통합하여 적은 데이터로도 일반화 가능한 모델 개발.[^1_5]

**7. 해석 가능성 향상**:
학습된 주기 창과 합성곱 필터의 의미를 시각화하고 해석하는 방법론 개발이 필요합니다.

**8. 실시간 예측 및 온라인 학습**:
WinNet의 낮은 계산 비용을 활용하여 스트리밍 데이터에 대한 온라인 학습 및 실시간 예측 시스템 구축.[^1_1]

**9. 확률적 예측**:
현재 점 예측에 집중하나, ProbTS (2024)처럼 분포 예측 기능 추가 고려.[^1_15]

**10. 도메인 특화 최적화**:
각 응용 분야(에너지, 금융, 의료)의 특성에 맞춘 WinNet 변형 개발이 유망합니다.

WinNet은 단순성, 효율성, 성능의 균형을 달성하여 시계열 예측 분야에 중요한 기여를 했으며, CNN 기반 방법의 잠재력을 재확인시켰습니다. 향후 연구는 이 기반 위에 적응성, 해석 가능성, 멀티모달 통합을 강화하는 방향으로 진행될 것으로 예상됩니다.[^1_1]
<span style="display:none">[^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26]</span>

<div align="center">⁂</div>

[^1_1]: 3254_WinNet_time_series_foreca.pdf

[^1_2]: https://arxiv.org/html/2506.06005v1

[^1_3]: https://arxiv.org/html/2510.23396v1

[^1_4]: https://arxiv.org/html/2404.01340v1

[^1_5]: https://arxiv.org/pdf/2310.01728.pdf

[^1_6]: https://arxiv.org/abs/2411.04554

[^1_7]: https://www.linkedin.com/posts/risman-adnan-bb726b5_what-is-the-top-performer-ai-model-on-deep-activity-7331481462224904192-sRJJ

[^1_8]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11888914/

[^1_9]: https://arxiv.org/html/2406.00566v1

[^1_10]: https://arxiv.org/pdf/2310.10688.pdf

[^1_11]: https://arxiv.org/abs/2310.10688

[^1_12]: https://arxiv.org/pdf/2404.15772.pdf

[^1_13]: http://arxiv.org/pdf/2407.10768.pdf

[^1_14]: http://arxiv.org/pdf/2405.13522.pdf

[^1_15]: http://arxiv.org/pdf/2310.07446.pdf

[^1_16]: https://arxiv.org/pdf/2312.17100.pdf

[^1_17]: http://arxiv.org/pdf/2412.13769.pdf

[^1_18]: http://arxiv.org/pdf/2310.04948.pdf

[^1_19]: https://arxiv.org/abs/2310.01728

[^1_20]: https://arxiv.org/html/2508.07697v3

[^1_21]: https://arxiv.org/html/2411.05793v1

[^1_22]: https://arxiv.org/pdf/2506.12953.pdf

[^1_23]: https://www.reddit.com/r/datascience/comments/1i4yyoe/influential_timeseries_forecasting_papers_of/

[^1_24]: https://www.sohu.com/a/639166184_121119001/

[^1_25]: https://arxiv.org/abs/2503.10198

[^1_26]: https://tarlton.info/09-Citations/zhangPeriodicTimeSeries2020

