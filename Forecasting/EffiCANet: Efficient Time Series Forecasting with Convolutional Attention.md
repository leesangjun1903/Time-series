# EffiCANet: Efficient Time Series Forecasting with Convolutional Attention

> **Xinxing Zhou et al., "EffiCANet: Efficient Time Series Forecasting with Convolutional Attention"**, arXiv:2411.04669v1 (2024)[^1_1][^1_2]

***

## 1. 핵심 주장 및 주요 기여 요약

EffiCANet은 **정확도와 계산 효율성의 균형**이라는 기존 딥러닝 시계열 예측 모델의 근본적 한계를 해결하기 위해 제안된 **하이브리드 합성곱-어텐션 네트워크**입니다. 9개의 벤치마크 데이터셋에서 최대 **MAE 10.02% 감소** 및 표준 대형 커널 합성곱 대비 **FLOPs 26.2% 절감**을 동시에 달성하였습니다.[^1_3][^1_1]

**세 가지 핵심 기여:**

- **TLDC (Temporal Large-kernel Decomposed Convolution):** 대형 커널을 소형 커널로 분해하여 수용 영역을 확장하면서도 계산 비용을 획기적으로 절감
- **IVGC (Inter-Variable Group Convolution):** 비동기·지연(lag) 관계를 포함한 변수 간 동적 의존성을 그룹 합성곱으로 모델링
- **GTVA (Global Temporal-Variable Attention):** Squeeze-and-Excitation(SE) 원리를 시간·변수 이중 경로로 분리하여 전역적 특징 선택 수행[^1_4][^1_1]

***

## 2. 해결 문제 및 제안 방법 (수식 포함)

### 해결하고자 하는 문제

현존 모델들은 두 가지 핵심 과제를 동시에 해결하지 못합니다:[^1_1]

- **과제 1:** Transformer는 장기 의존성 포착에 뛰어나지만 연산 비용이 높고, TCN은 효율적이지만 수용 영역(receptive field)이 제한적
- **과제 2:** 다변량 시계열에서 변수 간 관계는 비동기(asynchrony)·지연(lead-lag) 효과로 인해 복잡하게 얽혀 있으며, 기존 모델은 이를 독립적으로만 처리


### 문제 정의 (수식)

입력 다변량 시계열 $\mathbf{X} \in \mathbb{R}^{M \times T}$에 대해, 모델 $\mathcal{F}$는 과거 $H$-스텝 윈도우로부터 미래 $\tau$-스텝을 예측합니다:

```math
\hat{\mathbf{X}}_{t_0:t_0+\tau} = \mathcal{F}_\Phi(\mathbf{X}_{t_0-H:t_0}) 
```

여기서 $\Phi$는 모델 파라미터를 나타냅니다.[^1_1]

***

## 3. 모델 구조 상세 설명

### 전체 아키텍처

입력 데이터는 **패칭 및 임베딩** → **$L$개의 스택 블록 (TLDC → IVGC → GTVA)** → **예측 헤드(Predict Head)** 순서로 처리됩니다. 각 블록의 출력은 다음과 같이 정의됩니다:[^1_1]

```math
\mathbf{Z}^{(l)} = f^{(l)}_{\text{Block}}(\mathbf{Z}^{(l-1)}), \quad \mathbf{Z}^{(0)} = \mathbf{X}_{\text{emb}}
```

***

### 모듈 1: TLDC

대형 커널 크기 $K$를 모사하기 위해 두 단계의 계층적 합성곱을 수행합니다:[^1_1]

**1단계 - 깊이별 합성곱 (DW Conv):** 지역 단기 의존성 포착

```math
\mathbf{X}_{\text{local}} = \text{DW-Conv}(\mathbf{X}_{\text{emb}})
```

**2단계 - 깊이별 팽창 합성곱 (DW-D Conv):** 팽창률 $d$를 활용한 장기 의존성 확장

```math
\mathbf{X}_{\text{dilated}} = \text{DW-D-Conv}(\mathbf{X}_{\text{local}})
```

**요소별 합산으로 단·장기 특징 결합:**

```math
\mathbf{X}_{\text{combined}} = \mathbf{X}_{\text{dilated}} + \mathbf{X}_{\text{local}}
```

**계산 복잡도 비교:**


| 방법 | 파라미터 수 | FLOPs |
| :-- | :-- | :-- |
| 표준 대형 커널 | $(M \times D)(K+1)$ | $2MDK N$ |
| TLDC | $MD(2d+1+\lceil K/d \rceil)$ | $2MD(2d-1+\lceil K/d \rceil)N$ |

복잡도 감소 비율은 팽창률에 반비례( $O(1/d)$ )하며, $K=55$, $d=5$ 조건에서 파라미터 **61% 감소**, FLOPs **64% 감소**.[^1_1]

***

### 모듈 2: IVGC

시간 윈도우 크기 $W$로 세그멘테이션 후 표준 패딩과 헤드-테일 패딩 두 가지 방식을 병렬 적용합니다:[^1_1]

**패딩 길이 계산:**

```math
N_{\text{pad1}} = \begin{cases} 0 & \text{if } N \equiv 0 \pmod{W} \\ W - (N \bmod W) & \text{otherwise} \end{cases}
```

```math
N_{\text{left\_pad2}} = \lfloor W/2 \rfloor, \quad N_{\text{right\_pad2}} = W - \lfloor W/2 \rfloor + N_{\text{pad1}} 
```

**두 경로의 그룹 합성곱 후 합산:**

```math
\mathbf{Y} = \text{Conv}(\mathbf{Y}_{\text{padded1}} + \mathbf{Y}_{\text{padded2}})
```

***

### 모듈 3: GTVA

SE(Squeeze-and-Excitation) 원리를 **시간 어텐션**과 **변수 어텐션** 이중 경로로 분리합니다:[^1_1]

**시간 어텐션:**

```math
\mathbf{T}_{\text{pool}} = \text{AvgPool}(\mathbf{Y}_{\text{temp}}) \in \mathbb{R}^{N \times D} 
```

```math
\mathbf{T}_{\text{atten}} = \sigma\!\left(\mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \cdot \mathbf{T}_{\text{pool}})\right)
```

**변수 어텐션:**

```math
\mathbf{V}_{\text{pool}} = \text{AvgPool}(\mathbf{Y}_{\text{var}}) \in \mathbb{R}^{M \times D}
```

```math
\mathbf{V}_{\text{atten}} = \sigma\!\left(\mathbf{W}_4 \cdot \text{ReLU}(\mathbf{W}_3 \cdot \mathbf{V}_{\text{pool}})\right)
```

**이중 어텐션 결합 (Hadamard 곱):**

```math
\mathbf{Y}_{\text{out}} = \sigma(\mathbf{T}_{\text{atten}} \odot \mathbf{V}_{\text{atten}} \odot \mathbf{Y})
```

**블록 간 피드백 연결 (잔차와 달리 곱셈 방식):**

```math
\mathbf{X}'_{\text{emb}} = \mathbf{Y}_{\text{out}} \odot \mathbf{X}_{\text{emb}}
```

***

## 4. 성능 향상 결과

### 벤치마크 데이터셋 성능

72개 실험 평가에서 **51회 1위, 13회 2위** 달성:[^1_1]


| 데이터셋 | MSE 감소 (vs 2위) | MAE 감소 (vs 2위) |
| :-- | :-- | :-- |
| ETTh2 | 4.70% | 2.53% |
| ILI | **10.02%** | 3.34% |
| ETTm1 (avg) | 0.287→0.347 (1위) | - |
| Electricity (avg) | 0.156 (1위) | 0.252 (1위) |
| Traffic (avg) | 0.387 (1위) | 0.278 (1위) |

### 계산 효율성

ECL 데이터셋 기준으로 TLDC는 표준 대형 커널 합성곱(TLC) 대비 파라미터 **32.9% 절감**, FLOPs **26.2% 절감**을 달성하면서 예측 정확도(MAE·MSE 편차 0.8% 이내)를 거의 유지합니다.[^1_3][^1_1]

***

## 5. 일반화 성능 향상 가능성

### 설계 차원에서의 일반화 강점

- **피드백 곱셈 연결 (수식 22):** 기존 잔차 덧셈(additive residual)과 달리 요소별 곱셈으로 각 블록 출력이 입력과 동적으로 상호작용하여 **레이어 간 누적 특징 정제**가 발생, 장기 예측 일반화에 유리[^1_1]
- **IVGC의 이중 패딩 전략:** 표준 패딩과 헤드-테일 패딩을 동시에 사용함으로써 시간 윈도우 경계에서 발생하는 정보 소실을 보완하여 **다양한 시간 패턴 분포에 강인**[^1_1]
- **채널 독립 처리 (Channel Independence):** 패칭 임베딩 단계에서 변수 차원을 unsqueeze하여 각 변수를 독립 처리함으로써 변수 수(M)가 다른 데이터셋에 유연하게 적응[^1_1]


### 일반화의 한계

- **파라미터 민감도:** 블록 수 $L > 1$ 이상에서 과적합이 관찰되며, 단일 블록($L=1$)이 최적인 데이터셋이 많아 **깊은 모델의 일반화 능력이 불안정**[^1_1]
- **비정상(Non-stationary) 시계열 취약:** 논문 저자들 스스로도 "비정상 시계열 최적화"를 미래 과제로 언급하며, 분포 이동(distribution shift)이 심한 데이터에서의 일반화는 미해결 과제[^1_3][^1_1]
- **하이퍼파라미터 데이터셋 의존성:** $K=55$, $d=5$, $W=4$, $r=16$ 등의 최적값이 ETTh1 기준이며, **다른 도메인 데이터셋에서는 재탐색이 필요**[^1_1]

***

## 6. 최신 관련 연구 비교 분석 (2020년 이후)

| 모델 | 출판 연도 | 핵심 방법 | 장점 | 단점 (vs EffiCANet) |
| :-- | :-- | :-- | :-- | :-- |
| **Informer** | 2021 | ProbSparse 어텐션 | 장기 의존성 포착 | 계산 비용 높음, 변수 간 관계 미흡 |
| **Autoformer** [^1_5] | 2021 | 분해 기반 자기상관 Transformer | 추세·계절성 분리 | 비선형 변수 관계 한계 |
| **PatchTST** [^1_6] | 2023 | 패치 기반 채널 독립 Transformer | 패턴 국소화 | FLOPs 과다, 불규칙 시계열 취약 |
| **DLinear/NLinear** | 2023 | 단순 선형 레이어 | 초경량, 해석 용이 | 비선형·다변수 데이터에서 성능 열세 |
| **ModernTCN** [^1_7] | 2024 | 대형 커널 TCN 현대화 | 수용 영역 확장 | 직접 대형 커널 사용으로 비효율 |
| **ConvTimeNet** [^1_8] | 2024 | 깊이별·점별 합성곱 계층 | 전역 시퀀스 모델링 | IVGC 방식의 동적 변수 관계 미반영 |
| **iTransformer** [^1_9] | 2024 | 역전된 Transformer (variate token) | 다변량 상관 향상 | 시간 차원 역전으로 국소 패턴 약화 |
| **EffiCANet** | 2024 | TLDC + IVGC + GTVA 하이브리드 | 정확도+효율 균형, 동적 변수 관계 | 비정상 시계열, 블록 수 과적합 |

[^1_10][^1_11][^1_1]

EffiCANet의 가장 차별화된 포인트는 **대형 커널의 효과를 분해 합성곱으로 근사**하는 동시에, **변수 간 비동기·지연 패턴을 이중 패딩 그룹 합성곱**으로 처리한다는 점입니다. 이는 ModernTCN의 수용 영역 확장 전략보다 계산 효율이 높고, iTransformer의 variate token 방식보다 동적 시간 변화에 유연합니다.[^1_11][^1_1]

***

## 7. 미래 연구에 미치는 영향 및 고려 사항

### 연구적 영향

1. **합성곱-어텐션 하이브리드 설계 방향 제시:** "어텐션 대신 단순 선형·MLP"라는 2023년 트렌드를 넘어, 구조적으로 최적화된 합성곱과 경량 어텐션의 결합이 Transformer를 대체할 수 있음을 실증[^1_12][^1_1]
2. **분해 합성곱의 일반성:** TLDC의 DW+DW-D 분해 전략은 시계열 외에도 **시공간 예측(spatio-temporal forecasting), 이상 탐지(anomaly detection)** 등으로 확장 가능[^1_1]
3. **Edge AI·IoT 적용 가능성:** 26.2% FLOPs 절감은 산업용 엣지 디바이스에서의 실시간 예측 가능성을 열었으며, 이 방향의 후속 연구를 촉진[^1_3][^1_1]

### 후속 연구 시 반드시 고려해야 할 점

- **비정상 시계열 처리:** RevIN(Reversible Instance Normalization)  또는 적응형 분해 기법(예: Autoformer의 추세-계절 분리)을 TLDC와 결합하여 분포 이동에 대한 견고성 향상 필요[^1_13][^1_1]
- **과적합 억제 전략:** 블록 수 증가에 따른 과적합을 SAMformer 의 **Sharpness-Aware Minimization(SAM)** 기법으로 완화하거나, 드롭아웃·배치 정규화 방식을 재검토해야 함[^1_14][^1_1]
- **해석 가능성(Interpretability) 강화:** IVGC의 합성곱 가중치로 변수 의존성을 시각화하는 접근은 긍정적이나, 의료·금융 등 고위험 도메인 적용을 위해서는 **SHAP, Grad-CAM 수준의 설명 체계**가 요구됨[^1_3][^1_1]
- **변수 수 확장성:** Traffic(862개 변수)에서 강점을 보이지만, 수천 개 변수 환경에서의 IVGC 그룹 합성곱 확장성은 추가 검증이 필요하며, **sparse graph 구조** 도입을 검토할 수 있음
- **시계열 파운데이션 모델과의 결합:** TimesFM, Mamba4Cast  등 제로샷 파운데이션 모델이 부상하는 현시점에서, EffiCANet의 효율적 합성곱 백본을 **파인튜닝 가능한 경량 어댑터**로 활용하는 방향도 유망한 연구 주제[^1_15]
<span style="display:none">[^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43]</span>

<div align="center">⁂</div>

[^1_1]: 2411.04669v1.pdf

[^1_2]: https://arxiv.org/abs/2411.04669

[^1_3]: https://www.themoonlight.io/en/review/efficanet-efficient-time-series-forecasting-with-convolutional-attention

[^1_4]: https://arxiv.org/html/2411.04669

[^1_5]: https://www.ijcai.org/proceedings/2024/0275.pdf

[^1_6]: https://arxiv.org/pdf/2304.08424.pdf

[^1_7]: https://arxiv.org/pdf/2310.10688.pdf

[^1_8]: https://arxiv.org/abs/2404.14197

[^1_9]: https://link.springer.com/10.1007/s13042-024-02417-8

[^1_10]: https://arxiv.org/html/2411.04669v1

[^1_11]: https://openreview.net/forum?id=vpJMJerXHU

[^1_12]: https://www.semanticscholar.org/paper/41742e4ae0f3c4afcf5a90eefbd685d63134d911

[^1_13]: https://link.springer.com/10.1007/s10462-024-10989-8

[^1_14]: https://arxiv.org/html/2402.12694v3

[^1_15]: https://arxiv.org/abs/2410.09385

[^1_16]: https://arxiv.org/abs/2402.01533

[^1_17]: https://arxiv.org/pdf/2402.01533.pdf

[^1_18]: https://arxiv.org/html/2506.17253v3

[^1_19]: https://arxiv.org/pdf/2410.09385.pdf

[^1_20]: https://arxiv.org/html/2402.05370v1

[^1_21]: https://arxiv.org/html/2505.19090v1

[^1_22]: https://arxiv.org/abs/2405.17951

[^1_23]: https://arxiv.org/html/2505.00307v2

[^1_24]: https://arxiv.org/pdf/2311.15599.pdf

[^1_25]: https://arxiv.org/abs/2405.10877

[^1_26]: https://dl.acm.org/doi/10.1145/3729531

[^1_27]: https://www.mdpi.com/2227-7390/12/23/3666

[^1_28]: https://ieeexplore.ieee.org/document/10762340/

[^1_29]: https://ieeexplore.ieee.org/document/10796427/

[^1_30]: https://journal.lembagakita.org/ijsecs/article/view/2774

[^1_31]: https://jurnal.kdi.or.id/index.php/ef/article/view/1110

[^1_32]: http://arxiv.org/pdf/2411.04669v1.pdf

[^1_33]: https://arxiv.org/pdf/2307.01616.pdf

[^1_34]: http://arxiv.org/pdf/2503.17410.pdf

[^1_35]: https://arxiv.org/html/2504.00120v1

[^1_36]: http://arxiv.org/pdf/2410.22981.pdf

[^1_37]: https://arxiv.org/pdf/2312.17100.pdf

[^1_38]: https://dl.acm.org/doi/10.5555/3692070.3693436

[^1_39]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11059412/

[^1_40]: https://blog.csdn.net/tMb8Z9Vdm66wH68VX1/article/details/144203019

[^1_41]: https://arxiv.org/html/2407.13806v1

[^1_42]: https://www.sciencedirect.com/science/article/abs/pii/S0925231220300606

[^1_43]: http://www.arxiv.org/abs/2411.04669

