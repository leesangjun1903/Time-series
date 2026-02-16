
# Learning to Rotate: Quaternion Transformer for Complicated Periodical Time Series Forecasting

## 1. 핵심 주장과 주요 기여

Quaternion Transformer (Quatformer)는 복잡한 주기 패턴을 가진 시계열 예측의 근본적인 문제를 해결하기 위해 제안된 프레임워크입니다. 이 논문은 실세계 시계열 데이터의 **다중 주기(multiple periods)**, **가변 주기(variable periods)**, **위상 이동(phase shifts)**이라는 세 가지 복잡한 주기적 패턴을 효과적으로 모델링하는 새로운 접근법을 제시합니다.[^1_1]

주요 기여는 다음 세 가지로 요약됩니다:[^1_1]

- **Learning-to-Rotate Attention (LRA)**: 쿼터니언을 활용하여 학습 가능한 주기와 위상 정보를 attention 메커니즘에 통합
- **Trend Normalization**: 시계열의 추세가 천천히 변하는 특성을 반영한 새로운 정규화 기법
- **Decoupling Attention with Global Memory**: 선형 복잡도로 계산 효율성을 달성하면서도 예측 정확도를 유지


## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 Transformer 기반 모델들은 시계열 예측에서 두 가지 주요 한계를 보입니다:[^1_1]

1. **복잡한 주기 패턴 처리 실패**: Autoformer와 같은 최신 모델도 auto-correlation을 통해 고정된 주기만 추정할 수 있으며, 시간 지연 집계(time delay aggregation)는 위상 이동을 처리하지 못함
2. **계산 복잡도**: Dot-product attention의 $O(N^2)$ 복잡도는 긴 시계열 모델링을 방해함

### 2.2 제안하는 방법 (수식 포함)

#### Rotatory Softmax-Kernel

논문의 핵심은 쿼터니언을 활용한 새로운 커널 함수입니다:[^1_1]

$$
\text{SM}_{\text{rot}}(\phi(\mathbf{x},m), \psi(\mathbf{y},n)) = \exp(\text{Re}[\phi(\mathbf{x},m)^H \psi(\mathbf{y},n)])
$$

여기서 회전 함수는:

$$
\phi(\mathbf{x},m) = \tilde{\mathbf{x}} e^{i2\pi\omega m}, \quad \psi(\mathbf{y},n) = \tilde{\mathbf{y}} e^{j2\pi\omega n}
$$

$\tilde{\mathbf{x}}, \tilde{\mathbf{y}}$는 실수 벡터의 쿼터니언 형식이고, $\omega = 1/T$는 주기 $T$에 대응하는 주파수입니다.[^1_1]

이 커널은 세 가지 핵심 속성을 만족합니다:[^1_1]

1. **Boundedness**: $\|\phi(\mathbf{x},m)\| = \|\mathbf{x}\|$ (norm-invariant)
2. **Initial-value invariance**: $\phi(\mathbf{x}, 0) = \mathbf{x}$
3. **Period-independent and phase-dependent translation**: 주기 번역에는 불변이지만 위상에는 의존

#### Learning-to-Rotate Attention

실세계 데이터의 복잡성을 다루기 위해, 논문은 데이터 기반으로 여러 주파수와 위상을 학습합니다:[^1_1]

**주파수/위상 생성**:

$$
\{\omega^Q_1, \ldots, \omega^Q_P\}, \{\theta^Q_1, \ldots, \theta^Q_P\} = \text{Conv}(\mathbf{Q}; W^Q_\omega), \text{Conv}(\mathbf{Q}; W^Q_\theta)
$$

**시리즈 회전**:

$$
\Phi_p(\mathbf{Q}, \text{pos}^Q) = \tilde{\mathbf{Q}} e^{i(2\pi\omega^Q_p \text{pos}^Q + \theta^Q_p)}, \quad p = 1, 2, \ldots, P
$$

**Attention 계산**:

$$
\mathbf{S} = \text{softmax}\left(\frac{1}{P\sqrt{d}} \sum_{p=1}^P \text{Re}[\Phi_p(\mathbf{Q}, \text{pos}^Q) \Psi_p(\mathbf{K}, \text{pos}^K)^H]\right)
$$

**정규화 항**:[^1_1]

$$
\mathcal{L}_\omega = \frac{1}{P(N-1)} \sum_{p=1}^P \sum_{n=0}^{N-2} (\omega^{(n+1)}_p - \omega^{(n)}_p)^2
$$

$$
\mathcal{L}_\theta = \frac{1}{PN} \sum_{p=1}^P \sum_{n=0}^{N-1} |\theta^{(n)}_p|
$$

최종 손실 함수는:

$$
\mathcal{L} = \mathcal{L}_{\text{pred}} + \lambda_1 \mathcal{L}_\omega + \lambda_2 \mathcal{L}_\theta
$$

#### Trend Normalization

기존 Layer Normalization과 달리, Trend Normalization은 시계열의 천천히 변하는 추세를 모방합니다:[^1_1]

$$
\frac{\gamma}{\sigma} \odot (\mathbf{X} - \text{MovingAvg}(\mathbf{X})) + \mathbf{T}
$$

여기서 추세 $\mathbf{T}$는 다항 함수로 학습됩니다:

$$
\mathbf{T} = \sum_{i=0}^p \boldsymbol{\beta}_i \text{pos}^i
$$

#### Decoupling Attention

선형 복잡도 달성을 위해, 논문은 길이 $c$의 latent series $\mathbf{M}$을 도입합니다:[^1_1]

$$
\mathbf{H} = \text{LR-Attn}(\mathbf{X}, \mathbf{M}'), \quad \mathbf{M}' = \text{LR-Attn}(\mathbf{M}, \mathbf{Y})
$$

복잡도: $O(2cN)$ (선형)

**Momentum 업데이트**:[^1_1]

$$
\mathbf{M} \leftarrow \alpha \mathbf{M} + (1-\alpha) \frac{1}{B} \sum_{i=1}^B \mathbf{M}'_i
$$

### 2.3 모델 구조

Quatformer는 encoder-decoder 아키텍처를 따릅니다:[^1_1]

- **Encoder**: $N$개의 동일한 레이어, 각각 Decoupling LR-Attention → Trend Norm → Feed Forward → Trend Norm
- **Decoder**: $M$개의 레이어, encoder 출력에 대한 cross-attention 추가
- **입력**: 관측 시계열 (길이 $I$)
- **출력**: 예측 시계열 (길이 $O$)


### 2.4 성능 향상

6개 벤치마크 데이터셋에서 광범위한 실험 결과:[^1_1]


| 데이터셋 | MSE 개선 (평균) | 최대 개선 |
| :-- | :-- | :-- |
| ETTh | 8.5% | 13.2% |
| ETTm | 9.2% | 18.5% |
| Weather | 12.4% | 17.1% |
| Exchange | 4.5% | 9.1% |
| Traffic | 2.3% | 5.2% |
| Electricity | 3.1% | 7.1% |

**전체 평균**: 8.1% MSE 개선, 최대 18.5% 개선 (Autoformer 대비)[^1_1]

**계산 효율성**: Decoupling attention을 통해 메모리 사용량과 실행 시간이 선형적으로 증가하여, 긴 시계열(10,000+ 길이)도 효율적으로 처리 가능.[^1_1]

### 2.5 한계점

논문에서 명시적으로 언급된 한계:[^1_1]

1. **잔차 성분 미반영**: 현재 프레임워크는 추세와 계절 성분에 집중하며, 잔차 성분(노이즈)을 통합하여 예측 불확실성을 추정하는 부분은 향후 과제
2. **하이퍼파라미터 민감도**: latent period 수 $P$, momentum coefficient $\alpha$, 정규화 계수 $\lambda_1, \lambda_2$ 등 여러 하이퍼파라미터 조정 필요
3. **단변량 시계열 성능**: 일부 데이터셋에서 단변량 예측 시 TCN이나 LSTM과 비교해 큰 우위를 보이지 않음[^1_1]

## 3. 모델의 일반화 성능 향상 가능성

Quatformer의 일반화 성능은 여러 측면에서 우수합니다:

### 3.1 구조적 일반화

**채널 독립성 (Channel-independence)**: 각 변수를 독립적으로 처리하여 다양한 차원의 데이터에 적용 가능. 이는 6개의 서로 다른 도메인(전력, 날씨, 교통, 금융 등)에서 일관된 성능 향상으로 입증됩니다.[^1_1]

**정규화 메커니즘**: $\mathcal{L}\_\omega$는 주파수가 천천히 변하도록, $\mathcal{L}_\theta$는 위상 이동이 희소하도록 제약하여 과적합 방지.[^1_1]

### 3.2 전이 학습 가능성

Global Memory의 momentum 업데이트는 mini-batch를 통해 전체 데이터셋의 전역 패턴을 학습합니다. 이는:[^1_1]

- 새로운 시계열에 빠르게 적응
- 유사한 주기 패턴을 가진 도메인 간 전이 가능


### 3.3 스케일링 특성

Decoupling attention의 선형 복잡도 $O(cN)$는 매우 긴 시계열(예: 7,680+ 길이)에서도 안정적인 성능을 보여, 대규모 데이터로의 확장성이 우수합니다.[^1_1]

## 4. 앞으로의 연구에 미치는 영향과 고려사항

### 4.1 연구에 미치는 영향

**주기성 모델링의 새로운 패러다임**: Quatformer는 쿼터니언을 시계열 예측에 적용한 선구적 연구로, 기하학적 변환을 통한 시간적 패턴 표현의 새로운 방향을 제시합니다.[^1_2][^1_1]

**Attention 메커니즘의 재해석**: 단순히 position encoding이 아닌, 도메인 특화 정보(주기, 위상)를 kernel 함수에 내장하는 접근법은 다른 sequential 데이터에도 적용 가능합니다.

**효율성과 정확도의 균형**: Decoupling attention은 선형 복잡도로 긴 시계열을 처리하면서도 성능을 유지하는 실용적 해결책을 제공합니다.[^1_1]

### 4.2 향후 연구 시 고려사항

#### (1) 불확실성 정량화

논문이 제안한 대로, 확률적 예측을 위한 잔차 성분 모델링이 필요합니다. Latent Diffusion Transformer와 같은 확률적 접근법과의 결합을 고려할 수 있습니다.[^1_3][^1_1]

#### (2) 멀티스케일 패턴 통합

Multi-resolution Time-Series Transformer (MTST)나 PeriodNet처럼 여러 해상도의 주기 패턴을 동시에 모델링하는 방향으로 확장 가능합니다.[^1_4][^1_5]

#### (3) Foundation Model로의 확장

TEMPO나 GPT4TS처럼 대규모 사전학습을 통한 일반화 성능 향상을 고려해야 합니다. Quatformer의 global memory 메커니즘은 이러한 방향과 잘 호환됩니다.[^1_6][^1_7]

#### (4) 쿼터니언의 해석 가능성

쿼터니언의 4개 성분(실수부, 3개 허수부)이 시계열의 어떤 특성을 표현하는지에 대한 심층 분석이 필요합니다.[^1_8][^1_2]

#### (5) 하이브리드 아키텍처

Mamba와 Transformer를 결합한 MAT처럼, State Space Model과 LRA의 결합으로 장단기 의존성을 더 효과적으로 포착할 수 있습니다.[^1_9]

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 Transformer 기반 시계열 모델의 진화

#### 1세대 (2020-2021): 효율성 개선

- **Informer (2021)**: ProbSparse attention으로 $O(L \log L)$ 복잡도 달성[^1_1]
- **Autoformer (2021)**: Auto-correlation과 series decomposition 도입[^1_1]


#### 2세대 (2022): 도메인 특화 설계

- **Quatformer (2022)**: 주기성에 특화된 쿼터니언 기반 attention[^1_1]
- **ETSformer (2022)**: 지수 평활(exponential smoothing) 원리 통합[^1_10]
- **FEDformer (2022)**: 주파수 도메인 처리[^1_11]


#### 3세대 (2023-현재): 단순화와 재검토

- **DLinear/Are Transformers Effective? (2023)**: 단순한 선형 모델이 복잡한 Transformer를 능가할 수 있음을 보임[^1_12][^1_13]
- **PatchTST (2023)**: Patch 기반 토큰화로 효율성과 성능 개선[^1_14]
- **TEMPO (2023)**: Generative pre-training 접근법[^1_6]


#### 4세대 (2024-2025): 멀티모달과 Foundation Models

- **Fredformer (2024)**: 주파수 편향(frequency bias) 완화[^1_15]
- **TKAT (2024)**: Kolmogorov-Arnold Networks 통합[^1_16]
- **Timer/Moment (2024)**: 대규모 사전학습 모델[^1_7]
- **HTMformer (2025)**: Hybrid Time-Multivariate 전략으로 35.8% 평균 성능 향상[^1_17]
- **PENGUIN (2025)**: Periodic-nested group attention 메커니즘[^1_18]


### 5.2 주기성 모델링 접근법 비교

| 모델 | 연도 | 주기 처리 방법 | 복잡도 | 주요 특징 |
| :-- | :-- | :-- | :-- | :-- |
| Autoformer[^1_1] | 2021 | Auto-correlation, 고정 주기 탐지 | $O(L \log L)$ | Time delay aggregation |
| **Quatformer**[^1_1] | 2022 | **쿼터니언 회전, 학습 가능 주기/위상** | **$O(cN)$** | **가변 주기, 위상 이동 처리** |
| ETSformer[^1_10] | 2022 | 지수 평활 attention | $O(L \log L)$ | 통계적 기법 통합 |
| MTST[^1_4] | 2023 | Multi-branch, 다중 해상도 | $O(L^2)$ | 상대적 위치 인코딩 |
| Skip-Timeformer[^1_19] | 2024 | Skip-time interaction | $O(L \log L)$ | 강한 주기성에 특화 |
| PeriodNet[^1_5] | 2025 | Period attention, sparse period attention | $O(L^2)$ | 인접 주기 간 유사도 |
| PENGUIN[^1_18] | 2025 | Periodic-nested group attention | $O(L^2)$ | 다중 주기성 명시적 모델링 |

### 5.3 Quatformer의 차별화 포인트

**1. 이론적 엄밀성**: 쿼터니언의 수학적 속성(boundedness, initial-value invariance, phase-dependent translation)을 명시적으로 증명[^1_1]

**2. 유연성**: 고정된 주기 가정 없이 CNN을 통해 시간에 따라 변하는 주파수와 위상을 학습[^1_1]

**3. 실용성**: Decoupling attention으로 선형 복잡도를 달성하면서도 평균 8.1% 성능 향상[^1_1]

**4. 검증 범위**: 6개 벤치마크, 4가지 예측 길이에서 일관된 우수 성능[^1_1]

### 5.4 최신 연구 트렌드와의 관계

**Transformer 회의론 대응**: DLinear 논문이 제기한 "Transformer가 시계열 예측에 효과적인가?"라는 질문에 대해, Quatformer는 도메인 특화 설계(주기성)를 통해 긍정적 답변을 제공합니다.[^1_13][^1_12][^1_1]

**Foundation Model 통합 가능성**: TEMPO, Timer와 같은 사전학습 모델의 인코더로 Quatformer의 LRA를 활용하면, 주기적 패턴이 강한 도메인에서 전이 학습 성능을 향상시킬 수 있습니다.[^1_7][^1_6]

**하이브리드 접근법**: MAT가 Mamba와 Transformer를 결합한 것처럼, LRA와 State Space Model을 결합하면 장기 의존성과 주기성을 동시에 효과적으로 포착할 수 있습니다.[^1_9]

**쿼터니언 신경망의 부활**: 최근 쿼터니언 기반 시계열 압축 연구와 함께, Quatformer는 쿼터니언 신경망의 시계열 분야 적용 가능성을 재조명했습니다.[^1_2][^1_8]

### 5.5 향후 통합 연구 방향

1. **Quatformer + PatchTST**: Patch 토큰화와 LRA의 결합으로 효율성과 주기성 모델링 강화
2. **Quatformer + Fredformer**: 주파수 편향 완화와 주기 학습의 시너지
3. **Quatformer + Diffusion Models**: 확률적 예측을 위한 latent diffusion 통합[^1_3]
4. **Multi-resolution LRA**: 여러 스케일의 쿼터니언 회전 동시 적용

## 결론

Quatformer는 복잡한 주기 패턴을 가진 시계열 예측에서 쿼터니언 기반 회전을 통한 새로운 패러다임을 제시했습니다. 2022년 발표 이후, 이 연구는 주기성 모델링의 중요성을 강조하며, PeriodNet, PENGUIN 등 후속 연구에 영향을 미쳤습니다. 향후 연구는 Foundation Model과의 통합, 불확실성 정량화, 멀티스케일 패턴 처리를 중심으로 발전할 것으로 예상됩니다.[^1_5][^1_18][^1_1]
<span style="display:none">[^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37]</span>

<div align="center">⁂</div>

[^1_1]: 3534678.3539234.pdf

[^1_2]: https://arxiv.org/pdf/2403.11722.pdf

[^1_3]: https://ojs.aaai.org/index.php/AAAI/article/view/29085

[^1_4]: https://arxiv.org/abs/2311.04147

[^1_5]: https://arxiv.org/abs/2511.19497

[^1_6]: https://arxiv.org/abs/2310.04948

[^1_7]: https://arxiv.org/html/2507.02907v1

[^1_8]: https://publikationen.fhb.fh-swf.de/servlets/MCRFileNodeServlet/fhswf_derivate_00002747/Time%20series%20compression%20using%20quaternion%20valued%20neural%20networks.pdf

[^1_9]: https://ieeexplore.ieee.org/document/10823516/

[^1_10]: https://arxiv.org/pdf/2202.01381.pdf

[^1_11]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12453695/

[^1_12]: https://towardsdatascience.com/influential-time-series-forecasting-papers-of-2023-2024-part-1-1b3d2e10a5b3/

[^1_13]: https://github.com/cure-lab/LTSF-Linear

[^1_14]: http://arxiv.org/pdf/2211.14730v2.pdf

[^1_15]: https://dl.acm.org/doi/10.1145/3637528.3671928

[^1_16]: https://arxiv.org/abs/2406.02486

[^1_17]: https://arxiv.org/html/2510.07084v1

[^1_18]: https://arxiv.org/html/2508.13773v1

[^1_19]: https://www.ijcai.org/proceedings/2024/608

[^1_20]: https://www.scitepress.org/DigitalLibrary/Link.aspx?doi=10.5220/0013395500003890

[^1_21]: https://www.ssrn.com/abstract=4718033

[^1_22]: https://ieeexplore.ieee.org/document/10352988/

[^1_23]: https://www.semanticscholar.org/paper/fb45d31cc89207aec392dbac8908cc24db2df871

[^1_24]: https://dl.acm.org/doi/10.1145/3637528.3671855

[^1_25]: https://arxiv.org/html/2411.01419v1

[^1_26]: https://arxiv.org/pdf/2502.13721.pdf

[^1_27]: http://arxiv.org/pdf/2503.17658.pdf

[^1_28]: http://arxiv.org/pdf/2410.12184.pdf

[^1_29]: http://arxiv.org/pdf/2410.23992.pdf

[^1_30]: https://arxiv.org/pdf/2206.05495.pdf

[^1_31]: https://arxiv.org/html/2508.16641v1

[^1_32]: https://arxiv.org/html/2602.00589v1

[^1_33]: https://arxiv.org/html/2511.19497v1

[^1_34]: https://arxiv.org/pdf/2209.14551.pdf

[^1_35]: https://openreview.net/forum?id=kHEVCfES4Q\&noteId=mrNbq9EkQa

[^1_36]: https://www.sciencedirect.com/science/article/pii/S1574013725001595

[^1_37]: https://milvus.io/ai-quick-reference/how-do-attention-mechanisms-enhance-time-series-forecasting-models

