<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# TACTiS: Transformer-Attentional Copulas for Time Series 논문 상세 분석

## 1. 핵심 주장과 주요 기여

TACTiS는 Transformer 구조와 비모수적 코퓰라(non-parametric copula)를 결합하여 고차원 다변량 시계열의 결합 예측 분포를 추정하는 혁신적인 방법론입니다. 이 논문은 Transformer 기반의 attention 메커니즘이 코퓰라의 속성을 모방하도록 학습될 수 있음을 이론적으로 증명하고, 실세계 데이터셋에서 최첨단 성능을 달성했습니다.[^1_1]

### 주요 기여

- **Attentional Copulas 제안**: 임의 개수의 확률 변수에 대해 비모수적 코퓰라를 추정하는 attention 기반 구조를 개발했습니다[^1_1]
- **이론적 보장**: Attentional copulas가 유효한 코퓰라로 수렴함을 이론적으로 증명했습니다[^1_1]
- **높은 유연성**: 불규칙 샘플링, 결측값, 다양한 샘플링 빈도, 예측 및 보간을 단일 프레임워크에서 처리할 수 있습니다[^1_1]
- **확장성**: Bagging 기법을 통해 수백 개의 시계열로 확장 가능합니다[^1_1]


## 2. 상세 분석

### 2.1 해결하고자 하는 문제

TACTiS는 실세계 시계열 데이터의 다음과 같은 특성을 모두 처리할 수 있는 범용 예측 방법을 개발하고자 합니다:[^1_1]

1. **다변량 확률 과정의 결합 특성화**: 임의의 시간 지평에서 궤적 예측
2. **비확률적 공변량의 존재**: 정적 또는 시변 조건 변수
3. **다양한 샘플링 빈도**: 불규칙 샘플링 또는 정렬되지 않은 변수
4. **임의 결측값**: 임의 시점 및 변수에 대한 결측값 처리
5. **다양한 도메인**: $\mathbb{R}, \mathbb{R}_+, \mathbb{N}, \mathbb{N}_+, \mathbb{Z}$ 등 왜도와 두터운 꼬리를 가진 주변 분포

기존 ARIMA, LSTM 기반 방법들은 이러한 복잡한 특성을 종합적으로 다루기 어렵고, 완전한 예측 분포를 제공하지 못하는 한계가 있었습니다.[^1_1]

### 2.2 제안하는 방법 (수식 포함)

#### 문제 정의

$m$개의 다변량 시계열 $\mathcal{S} = \{X_1, ..., X_m\}$가 주어지고, 각 $X = \{x_i \in \mathbb{R}^{l_i}\}_{i=1}^n$는 길이가 임의인 단변량 시계열들의 집합입니다. 목표는 다음 결합 분포를 추론하는 것입니다:[^1_1]

$P\left(\{x_i^{(m)}\}_{i=1}^n \mid \{x_i^{(o)}, C_i, t_i\}_{i=1}^n\right)$

여기서 $x_i^{(o)}$와 $x_i^{(m)}$는 각각 관측된 요소와 결측된 요소를 나타냅니다.[^1_1]

#### Encoder 구조

Encoder는 표준 Transformer와 유사하지만, 관측 및 결측 토큰을 모두 동시에 인코딩합니다:[^1_1]

**Input Embedding**:
$e_{ij} = \text{Embed}_{\theta_{emb}}(x_{ij} \cdot m_{ij}, c_{ij}, m_{ij})$

**Positional Encoding**:
$e'_{ij} = \frac{e_{ij}}{\sqrt{d_{emb}}} + p_{ij}$

여기서 $p_{ij}$는 시간 스탬프 $t_{ij}$에 기반한 위치 인코딩입니다.[^1_1]

#### Decoder: Copula 기반 밀도 추정

모델은 다음과 같은 코퓰라 기반 구조를 사용합니다:[^1_1]

$g_{\phi}(x_1^{(m)}, ..., x_{n_m}^{(m)}) = c_{\phi_c}(F_{\phi_1}(x_1^{(m)}), ..., F_{\phi_{n_m}}(x_{n_m}^{(m)})) \times \prod_{k=1}^{n_m} f_{\phi_k}(x_k^{(m)})$

여기서:

- $c_{\phi_c}$: 코퓰라 밀도
- $F_{\phi_k}$: $k$번째 변수의 주변 누적분포함수(CDF)
- $f_{\phi_k}$: $k$번째 변수의 주변 밀도함수

**주변 분포 모델링**: Deep Sigmoidal Flows (DSF)를 사용하여 주변 CDF를 모델링합니다. DSF는 단조증가하고 연속적이며 미분 가능한 함수로, \$\$로 매핑됩니다:[^1_1]

$\phi_k = \text{MarginalParams}_{\theta_F}(z_k^{(m)})$

**코퓰라 밀도의 자기회귀 분해**: 임의의 순열 $\pi = [\pi_1, ..., \pi_{n_m}]$에 따라 코퓰라 밀도를 분해합니다:[^1_1]

$c_{\phi_c^\pi}(u_1, ..., u_{n_m}) = c_{\phi_{c1}^\pi}(u_{\pi_1}) \times c_{\phi_{c2}^\pi}(u_{\pi_2} | u_{\pi_1}) \times \cdots \times c_{\phi_{cn_m}^\pi}(u_{\pi_{n_m}} | u_{\pi_1}, ..., u_{\pi_{n_m-1}})$

여기서 $u_{\pi_k} = F_{\phi_{\pi_k}}(x_{\pi_k}^{(m)})$이고, 첫 번째 조건부 밀도 $c_{\phi_{c1}^\pi}$는 $\mathcal{U}$의 밀도입니다.[^1_1]

**Attention 기반 조건화**: 각 조건부 분포의 파라미터는 attention 메커니즘을 통해 얻어집니다. 메모리는 관측 토큰과 순열상 선행하는 결측 토큰의 표현으로 구성됩니다:[^1_1]

$k = \text{Key}_{\theta_k}(z, u)$
$v = \text{Value}_{\theta_v}(z, u)$
$q = \text{Query}_{\theta_q}(z_{\pi_k}^{(m)})$

Attention 연산 후 최종 파라미터를 얻습니다:[^1_1]

$z'' = \text{LayerNorm}(\text{FeedForward}_{\theta_{FF}}(z') + z')$
$\phi_{ck}^\pi = \text{DistParams}_{\theta_{dist}}(z'')$

#### 학습 절차

모든 순열에 대한 기대 음의 로그 우도를 최소화합니다:[^1_1]

$\arg\min_{\Theta} \mathbb{E}_{\pi \sim \Pi, X \sim \mathcal{S}} \left[-\log g_{\phi^\pi}(x_1^{(m)}, ..., x_{n_m}^{(m)})\right]$

**정리 1 (이론적 보장)**: Attentional copula $c_{\phi_c^\pi}$는 위 목적함수를 최소화하는 파라미터 $\Theta$에서 유효한 코퓰라입니다.[^1_1]

**증명 핵심**: 순열에 대한 기하평균과 산술평균의 관계를 이용하여, 최적해에서 모델이 순열 불변성을 갖게 되고, 첫 번째 요소가 $\mathcal{U}$이므로 모든 주변 분포가 $\mathcal{U}$이 됨을 보입니다.[^1_1]

### 2.3 모델 구조

TACTiS의 전체 구조는 다음과 같이 구성됩니다:[^1_1]

1. **Input Embedding Layer**: 토큰 값, 공변량, 마스크를 벡터 표현으로 변환
2. **Positional Encoding**: 시간 스탬프 정보 추가
3. **Transformer Encoder**: Multi-head self-attention과 feed-forward 네트워크의 스택
4. **Marginal Flow Networks**: 각 변수의 주변 분포를 모델링하는 DSF
5. **Attentional Copula Decoder**: Attention 기반으로 변수 간 의존성 모델링

**Scalability를 위한 Temporal Transformer (TACTiS-TT)**: 변수 축과 시간 축을 분리하여 attention을 계산하여 복잡도를 $O([n \cdot l_{max}]^2)$에서 $O(n^2 \cdot l_{max} + n \cdot l_{max}^2)$로 감소시킵니다.[^1_1]

### 2.4 성능 향상

5개의 실세계 데이터셋(electricity, fred-md, kdd-cup, solar-10min, traffic)에서 CRPS-Sum 지표로 평가한 결과:[^1_1]


| 모델 | 평균 순위 |
| :-- | :-- |
| TACTiS-TT | **1.6 ± 0.2** |
| GPVar | 2.7 ± 0.2 |
| TimeGrad | 3.6 ± 0.3 |
| TempFlow | 3.9 ± 0.2 |

TACTiS-TT는 5개 데이터셋 중 3개에서 최고 성능을 달성했으며, 전체적으로 가장 낮은 평균 순위를 기록했습니다.[^1_1]

### 2.5 한계

논문에서 언급된 한계점들은 다음과 같습니다:[^1_1]

1. **초기화 민감성**: 실험 중 모델이 때때로 자명한 코퓰라(uniform marginals with no correlation)에 갇히는 문제가 발견되었습니다
2. **계산 복잡도**: 샘플링 단계에서 bagging이 적용되지 않아 추론 시 리소스가 많이 필요합니다
3. **이산 도메인 미지원**: 현재 연속 변수에만 적용 가능합니다
4. **시간별 위치 인코딩**: 일반적인 sinusoidal encoding을 사용하며, 시계열 특화 인코딩(휴일, 요일 등)은 향후 연구로 남겨두었습니다

## 3. 일반화 성능 향상 가능성

### 3.1 모델의 일반화 메커니즘

TACTiS는 여러 측면에서 뛰어난 일반화 성능을 보입니다:

**1. 변수 개수에 대한 일반화**: Attention 메커니즘 덕분에 훈련 시와 다른 개수의 시계열에 재훈련 없이 적용 가능합니다. Bagging 실험(Table 2)에서 bag size $b=2$부터 $b=30$까지 성능이 일관되게 유지되는 것이 확인되었습니다.[^1_1]

**2. 의존성 구조 재사용**: 코퓰라 접근법은 주변 분포와 의존성 구조를 분리하여 학습합니다. 이는 다음과 같이 표현됩니다:[^1_1]

$p(X_1 = x_1, ..., X_d = x_d) = c(F_1(x_1), ..., F_d(x_d)) \times f_1(x_1) \times \cdots \times f_d(x_d)$

이 분리는 학습된 의존성 구조를 유사한 표현을 가진 다른 변수 집합에 재사용할 수 있게 합니다.[^1_1]

**3. 결측 패턴에 대한 강건성**: 훈련 중 무작위로 마스킹된 값에 대해 학습하므로, 다양한 결측 패턴(예측, 보간, 또는 이들의 조합)에 적용 가능합니다.[^1_1]

### 3.2 Foundation Models로의 확장 가능성

논문은 TACTiS가 Foundation Models로 발전할 수 있는 가능성을 제시합니다:[^1_1]

> "TACTiS could be trained on time series from a wealth of domains, reusing the same attentional copula, but fine-tuning its encoder to new, unforeseen domains."

이는 다음과 같은 시나리오를 가능하게 합니다:

- 다양한 도메인의 시계열 데이터로 사전 훈련
- 동일한 attentional copula 재사용
- 새로운 도메인에 대해 encoder만 fine-tuning
- Cold-start 문제 해결: 관측이 거의 없는 상황에서도 합리적인 예측


### 3.3 실험적 검증

**Unaligned/Non-uniform 샘플링 실험**: Bivariate noisy sine process에서 관측 시점이 무작위로 분산되어 있어도 정확한 예측을 생성했습니다. 이는 모델이 시간 정렬에 의존하지 않고 일반화할 수 있음을 보여줍니다.[^1_1]

**보간 실험**: Stochastic volatility process에서 25개 time point gap에 대한 조건부 사후 분포를 ground truth와 매우 유사하게 추정했습니다. Energy score 기준으로 Oracle (ground truth 샘플링) 다음으로 우수한 성능을 보였습니다.[^1_1]

## 4. 향후 연구에 미치는 영향과 고려사항

### 4.1 연구에 미치는 영향

**1. 코퓰라 기반 딥러닝의 새로운 방향**: TACTiS는 비모수적 코퓰라를 neural network로 학습할 수 있음을 보였으며, 이는 경제학, 금융, 보험 분야에서 널리 사용되던 parametric copula 방법론에 대한 대안을 제시합니다.[^1_1]

**2. Attention 메커니즘의 새로운 응용**: Attention이 단순히 시퀀스 모델링뿐 아니라 확률적 의존성 구조를 학습하는 데 사용될 수 있음을 입증했습니다.[^1_1]

**3. 유연한 시계열 모델링 패러다임**: 마스킹 기반 접근법은 예측, 보간, 역추적 등 다양한 작업을 단일 프레임워크로 통합할 수 있는 길을 열었습니다.[^1_1]

### 4.2 향후 연구 시 고려사항

**1. 위치 인코딩 개선**:

- 시계열 특화 positional encoding (휴일, 요일, 계절성 등) 개발 필요[^1_1]
- 현재 sinusoidal encoding은 일반적이지만 도메인 특화 정보를 충분히 활용하지 못함

**2. 효율성 향상**:

- Sparse attention, linear attention 등 최신 Transformer 효율화 기법 적용[^1_1]
- 샘플링 단계에서의 계산 비용 감소 (조건부 독립성 학습 기반)

**3. 학습 동역학 연구**:

- 초기화 문제 해결을 위한 auxiliary task 개발[^1_1]
- 자명한 코퓰라에 갇히는 현상 방지 메커니즘

**4. 이산 변수 지원**:

- 현재 연속 변수만 지원하므로, 이산 도메인을 위한 decoder 확장 필요[^1_1]

**5. 조건부 독립성 활용**:

- 더 효율적인 샘플링을 위해 변수 간 조건부 독립성 구조 학습[^1_1]


## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 Transformer 기반 시계열 예측 모델

**Informer (2021)**: ProbSparse self-attention으로 $O(L \log L)$ 복잡도 달성. TACTiS와 달리 단일 시계열에 초점을 맞추며 코퓰라 기반 의존성 모델링은 없음.[^1_2]

**Temporal Fusion Transformer (TFT, 2021)**: Gating 메커니즘으로 불필요한 공변량 억제. 분위수 손실만 평가하며 전체 결합 분포는 모델링하지 않음.[^1_1]

**Autoformer (2021)**: Auto-correlation 메커니즘 도입하여 주기성 감지. 점예측과 단일 시점 분위수에 집중, TACTiS의 다변량 결합 분포 추정과는 다른 접근.[^1_3]

**PatchTST (2023)**: Channel-independent patching으로 지역적 시간 임베딩 학습. TACTiS는 channel 간 의존성을 명시적으로 모델링하여 대조적.[^1_4][^1_3]

**iTransformer (2024)**: 변수를 토큰으로, 시간을 특징으로 다루는 역전된 구조. TACTiS는 각 관측치를 개별 토큰으로 다루어 불규칙 샘플링 지원.[^1_3]

### 5.2 확률적 다변량 예측 모델

**TimeGrad (2021)**: Diffusion model 기반 자기회귀 예측. TACTiS 벤치마크에서 평균 순위 3.6으로 TACTiS(1.6)보다 낮은 성능.[^1_1]

**TempFlow (2021)**: Normalizing flow 기반 조건부 분포 모델링. 평균 순위 3.9로 TACTiS에 비해 성능 낮음.[^1_1]

**Latent Diffusion Transformer (LDT, 2024)**: 고차원 데이터를 잠재 공간으로 압축하여 diffusion 적용. TACTiS의 copula 접근법과 달리 latent space 생성 방식 사용.[^1_5]

**Multi-scale Attention Normalizing Flow (MANF, 2022)**: Multi-scale attention과 normalizing flow 결합. 비자기회귀 방식으로 누적 오차 방지, TACTiS도 자기회귀 코퓰라로 유사한 목표 달성.[^1_6]

### 5.3 코퓰라 기반 시계열 방법

**GPVar (2019)**: LSTM 기반 Gaussian copula process. TACTiS 벤치마크에서 평균 순위 2.7로 두 번째로 우수하나, parametric Gaussian copula의 표현력 제한.[^1_1]

**CopulaCPTS (2022)**: 다단계 예측을 위한 copula 기반 conformal prediction. 불확실성 정량화에 초점, TACTiS는 전체 결합 분포 모델링에 집중.[^1_7]

**CoCAI (2024)**: Copula와 diffusion model 결합하여 이상 탐지 수행. TACTiS와 유사하게 copula 사용하나 목적이 예측보다는 이상 탐지.[^1_8]

**Copula-based Spatio-temporal Graph Model (2024)**: Graph neural network와 copula 결합, 공간-시간 의존성 모델링. TACTiS의 attentional copula와 달리 그래프 구조 기반.[^1_9]

### 5.4 TACTiS의 차별점

| 특징 | TACTiS | 다른 최신 모델들 |
| :-- | :-- | :-- |
| **코퓰라 유형** | 비모수적 attentional copula | Parametric (Gaussian) 또는 미사용 |
| **불규칙 샘플링** | Native 지원 | 대부분 미지원 또는 제한적 |
| **작업 유연성** | 예측/보간/역추적 통합 | 주로 예측만 지원 |
| **이론적 보장** | 유효한 코퓰라 수렴 증명 | 대부분 경험적 검증만 |
| **변수 간 의존성** | 명시적 attention 기반 모델링 | Channel-independent 또는 암묵적 |
| **확장성** | Bagging으로 수백 개 시계열 | 제한적 |

### 5.5 최신 트렌드와 TACTiS의 위치

**Foundation Models 방향 (2024-2025)**: TimePFN, GPHT 등이 다양한 데이터셋에서 사전 훈련하여 일반화 성능 향상. TACTiS는 이 방향으로 확장 가능성을 논의했으나 아직 구현되지 않음.[^1_10][^1_11]

**효율성 개선**: Fredformer, PSformer 등이 주파수 편향 완화, 파라미터 효율성 개선. TACTiS는 temporal transformer로 일부 효율성 확보하나 추가 개선 여지 있음.[^1_12][^1_13]

**State Space Models 통합 (2024-2025)**: MAT 등이 Mamba와 Transformer 결합. TACTiS는 순수 Transformer 기반으로 이러한 최신 구조 미통합.[^1_14]

**결론**: TACTiS(2022)는 코퓰라 기반 확률적 예측에서 독보적 위치를 차지하며, 특히 불규칙 샘플링과 유연한 작업 지원에서 강점을 보입니다. 그러나 2023-2025년의 foundation model, 효율성 개선, state space model 통합 등의 트렌드를 반영한 후속 연구가 필요합니다.[^1_15][^1_5][^1_14][^1_3]
<span style="display:none">[^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38]</span>

<div align="center">⁂</div>

[^1_1]: 2202.03528v2.pdf

[^1_2]: https://ojs.aaai.org/index.php/AAAI/article/view/17325

[^1_3]: https://arxiv.org/html/2503.06928v1

[^1_4]: http://arxiv.org/pdf/2211.14730v2.pdf

[^1_5]: https://ojs.aaai.org/index.php/AAAI/article/view/29085

[^1_6]: https://arxiv.org/abs/2205.07493

[^1_7]: https://arxiv.org/html/2212.03281v4

[^1_8]: https://arxiv.org/html/2507.17796v1

[^1_9]: https://www.sciencedirect.com/science/article/abs/pii/S156849462400098X

[^1_10]: https://dl.acm.org/doi/10.1145/3637528.3671855

[^1_11]: https://arxiv.org/pdf/2502.16294.pdf

[^1_12]: https://dl.acm.org/doi/10.1145/3637528.3671928

[^1_13]: https://arxiv.org/html/2411.01419v1

[^1_14]: https://ieeexplore.ieee.org/document/10823516/

[^1_15]: https://arxiv.org/html/2510.03129v3

[^1_16]: https://www.arxiv.org/pdf/2510.03129v2.pdf

[^1_17]: https://arxiv.org/pdf/2509.26468.pdf

[^1_18]: https://arxiv.org/html/2512.18661v1

[^1_19]: https://arxiv.org/pdf/2502.08302.pdf

[^1_20]: https://arxiv.org/html/2402.01000v3

[^1_21]: https://www.mdpi.com/2072-4292/16/11/1915

[^1_22]: https://arxiv.org/abs/2406.02486

[^1_23]: https://www.ijcai.org/proceedings/2024/608

[^1_24]: http://www.proceedings.com/079017-2794.html

[^1_25]: https://ieeexplore.ieee.org/document/10533212/

[^1_26]: http://arxiv.org/pdf/2410.23749.pdf

[^1_27]: https://arxiv.org/pdf/2401.13968.pdf

[^1_28]: http://arxiv.org/pdf/2408.09723.pdf

[^1_29]: https://arxiv.org/abs/2207.05397

[^1_30]: https://arxiv.org/pdf/2502.13721.pdf

[^1_31]: https://www.sciencedirect.com/science/article/pii/S1574013725001595

[^1_32]: https://www.kjas.or.kr/journal/view.html?doi=10.5351%2FKJAS.2024.37.5.583

[^1_33]: https://github.com/ddz16/TSFpaper

[^1_34]: https://dataspace.princeton.edu/handle/88435/dsp01kk91fp583

[^1_35]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12453695/

[^1_36]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11059412/

[^1_37]: https://peerj.com/articles/cs-3001/

[^1_38]: https://openreview.net/forum?id=4QcFfTu6UT

