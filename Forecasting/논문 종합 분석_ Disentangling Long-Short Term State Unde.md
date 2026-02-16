<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

## 논문 종합 분석: Disentangling Long-Short Term State Under Unknown Interventions for Online Time Series Forecasting

### 1. 핵심 주장과 주요 기여

이 논문은 온라인 시계열 예측(online time series forecasting)에서 장기(long-term) 및 단기(short-term) 상태를 분리하는 새로운 프레임워크를 제안합니다. 핵심 주장은 기존 방법들이 장단기 상태를 적절히 분리하지 못해 비정상성(nonstationarity)에 효과적으로 적응하지 못한다는 것입니다.[^1_1]

주요 기여는 다음과 같습니다:

- **이론적 기여**: 알려지지 않은 개입(unknown interventions) 하에서 장단기 잠재 상태의 블록별 식별 가능성(block-wise identifiability)을 증명[^1_1]
- **방법론적 기여**: 장단기 분리 모델(LSTD: Long Short-Term Disentanglement model) 제안[^1_1]
- **실증적 기여**: 6개 벤치마크 데이터셋에서 최신 기법들을 능가하는 성능 달성, 특히 Exchange 데이터셋에서 44% 향상[^1_1]


### 2. 해결하고자 하는 문제와 제안 방법

## 해결하고자 하는 문제

기존 온라인 시계열 예측 방법들은 순차적으로 도착하는 데이터에서 장기 의존성을 유지하면서 단기 변화에 적응하는 것이 어렵습니다. 특히 FSNet, OneNet, D3A 같은 최신 방법들도 장단기 상태가 이미 잘 분리되어 있다고 가정하지만, 실제로는 이 가정이 충족되지 않아 비정상 환경에 효과적으로 적응하지 못합니다.[^1_1]

## 데이터 생성 프로세스

논문은 다음과 같은 데이터 생성 프로세스를 정의합니다:

$$
x_t = g(z_t)
$$

여기서 $z_t \in \mathbb{R}^n$은 장기 잠재 상태 $z^s_t \in \mathbb{R}^{n_s}$와 단기 잠재 상태 $z^d_t \in \mathbb{R}^{n_d}$로 구성됩니다 ($n = n_s + n_d$).[^1_1]

**장기 잠재 변수 생성:**

$$
z^s_{t,i} = f^s_i(\{z^s_{t-\tau,k} | z^s_{t-\tau,k} \in Pa(z^s_{t,i})\}, \varepsilon^s_{t,i}), \quad \varepsilon^s_{t,i} \sim p_{\varepsilon^s_{t,i}}
$$

**단기 잠재 변수 생성 (개입 포함):**

$$
z^d_{t,j} = \begin{cases} f^d_j(\{z^d_{t-\tau,k} | z^d_{t-\tau,k} \in Pa(z^d_{t,j})\}, \varepsilon^d_{t,j}) & \text{if } I = 0 \\ f^d_j(\varepsilon^d_{t,j}) & \text{if } I = 1 \end{cases}
$$

여기서 $I \sim B(I, \theta)$는 개입 발생 여부를 나타내는 베르누이 분포입니다.[^1_1]

## 식별 가능성 이론 (Theorem 1)

다음 세 가지 가정 하에서 장단기 잠재 변수의 블록별 식별 가능성이 보장됩니다:

**A1 (매끄럽고 양의 조건부 독립 밀도):**

$$
\log p(z_{t-\tau+1:t} | z_{t-\tau}) = \sum_{k=1}^{n_s} \log p(z_{t-\tau+1:t,k} | z_{t-\tau})
$$

**A2 (비특이 야코비안):** 생성 함수 $g$는 거의 모든 곳에서 비특이 야코비안 행렬을 가지며 역함수가 존재합니다.[^1_1]

**A3 (선형 독립):** 단기 잠재 변수에 대한 벡터 함수들이 선형 독립입니다:

$$
\bar{v}_{t-\tau,l} = \frac{\partial^2 \log p(z^d_{t-\tau+1:t} | z^d_{t-\tau})}{\partial z^d_{t-\tau+1:t,k} \partial z^d_{t-\tau,l}}
$$

## 모델 구조: LSTD

### Variational Sequential Autoencoder

모델의 Evidence Lower Bound (ELBO):

$$
\text{ELBO} = \mathbb{E}_{q(z^s_{1:H}|x_{1:H})} \mathbb{E}_{q(z^d_{1:H}|x_{1:H})} \ln p(x_{1:H}|z^s_{1:H}, z^d_{1:H}) - D_{KL}(q(z^s_{1:H}|x_{1:H}) || p(z^s_{1:H})) - D_{KL}(q(z^d_{1:H}|x_{1:H}) || p(z^d_{1:H}))
$$

### 장기 사전 네트워크 (Long-Term Prior Networks)

역전이 함수 $\{r^s_i\}$를 통해 추정된 노이즈:

$$
\hat{\epsilon}^s_{t,i} = r^s_i(\hat{z}^s_{t,i}, \hat{z}^s_{t-1})
$$

변환 $\kappa^s := \{\hat{z}^s_{t-1}, \hat{z}^s_t\} \rightarrow \{\hat{z}^s_{t-1}, \hat{\epsilon}^s_t\}$의 야코비안:

$$
J_{\kappa^s} = \begin{bmatrix} I & 0 \\ M & \text{diag}(\frac{\partial r^s_i}{\partial \hat{z}^s_{t,i}}) \end{bmatrix}
$$

이를 통해 장기 사전 확률을 다음과 같이 추정합니다:

$$
\log p(\hat{z}^s_{1:t}) = \log p(\hat{z}^s_1) + \sum_{\tau=2}^t \left( \sum_{i=1}^{n_s} \log p(\hat{\epsilon}^s_{\tau,i}) + \sum_{i=1}^{n_s} \log |\frac{\partial r^s_i}{\partial \hat{z}^s_{\tau,i}}| \right)
$$

### 매끄러움 제약 (Smooth Constraint)

장기 의존성 보존을 위해 시계열을 두 구간으로 나누어 어텐션 가중치의 유사성을 제약합니다:

$$
A^{z^s}_h = \text{Softmax}\left(\frac{z^s_{1:H/2} (z^s_{1:H/2})^T}{\sqrt{n_s}}\right), \quad A^{z^s}_e = \text{Softmax}\left(\frac{z^s_{H/2:H} (z^s_{H/2:H})^T}{\sqrt{n_s}}\right)
$$

$$
\mathcal{L}_m = ||A^{z^s}_h - A^{z^s}_e||_2
$$

### 중단된 의존성 제약 (Interrupted Dependency Constraint)

단기 변수의 개입을 강제하여 분리를 촉진합니다:

$$
\mathcal{L}_s = \sum_{(i,j) \in \{1,\ldots,n_d\}} \sum_{\tau \in \{2,\ldots,H-1\}} ||\frac{\partial \hat{\epsilon}^d_{H,i}}{\partial \hat{z}^d_{\tau-1,j}}||_1
$$

### 총 손실 함수

$$
\mathcal{L} = \mathcal{L}_R + \mathcal{L}_P + \beta \mathcal{L}_K + \alpha \mathcal{L}_m + \gamma \mathcal{L}_s
$$

여기서 $\mathcal{L}_R$은 재구성 손실, $\mathcal{L}_P$는 예측 손실, $\mathcal{L}_K = \mathcal{L}^s_K + \mathcal{L}^d_K$는 KL 발산 항입니다.[^1_1]

### 3. 성능 및 한계

## 성능 향상

실험 결과는 다음을 보여줍니다:

- **Exchange 데이터셋**: MSE 0.039, 최고 경쟁 모델 대비 44% 향상[^1_1]
- **Weather 데이터셋**: MSE 0.153 (Len=1), 기존 방법 대비 크게 감소[^1_1]
- **ECL 데이터셋**: MSE 2.112 (Len=1), 단기 예측에서 우수한 성능[^1_1]
- **Traffic 데이터셋**: MSE 0.231 (Len=1)[^1_1]


## 모델의 한계

1. **ETT 데이터셋에서의 제한적 성능**: ETTh2 및 ETTm1 데이터셋에서는 2위 성능을 보였는데, 이는 이 데이터셋에 알려지지 않은 개입이 적기 때문일 수 있습니다[^1_1]
2. **계산 복잡도**: 모델 효율성 비교에서 LSTD는 메모리 사용량이 다른 베이스라인보다 높습니다. 이는 학습 과정에서 사전 분포를 포함해야 하기 때문입니다[^1_1]
3. **가정의 한계**: 이론적 결과는 충분 조건을 제공하며, 일부 가정이 충족되지 않아도 모델이 작동할 수 있지만, 가정이 엄격하게 정의된 시나리오 부분집합에서만 식별 가능성이 보장됩니다[^1_1]

### 4. 모델의 일반화 성능 향상 가능성

이 논문은 여러 측면에서 일반화 성능 향상에 기여합니다:

## 이론적 보장

블록별 식별 가능성 이론은 모델이 장단기 상태를 올바르게 분리할 수 있는 이론적 근거를 제공합니다. 야코비안 행렬의 블록 대각 구조:[^1_1]

$$
J_{h_z} = \begin{bmatrix} A := \frac{\partial z^s_t}{\partial \hat{z}^s_t} & B := \frac{\partial z^s_t}{\partial \hat{z}^d_t} = 0 \\ C := \frac{\partial z^d_t}{\partial \hat{z}^s_t} = 0 & D := \frac{\partial z^d_t}{\partial \hat{z}^d_t} \end{bmatrix}
$$

이 구조는 장단기 상태가 독립적으로 식별될 수 있음을 보장합니다.[^1_1]

## 다양한 도메인에서의 적용

6개 벤치마크 데이터셋(ETTh2, ETTm1, Weather, ECL, Traffic, Exchange)에서 일관된 성능을 보여 다양한 도메인(전기, 기상, 교통, 금융)에서의 일반화 능력을 입증했습니다.[^1_1]

## 입력 수평선 변화에 대한 강건성

Ablation 연구는 입력 수평선이 20에서 60으로 변할 때도 모델이 안정적인 성능을 유지함을 보여줍니다. 이는 제한된 샘플로도 비정상 데이터에 적응할 수 있음을 의미합니다.[^1_1]

## 모델 불가지론적 설계

LSTD는 다양한 인코더 백본(FSNet, MLP)과 결합할 수 있는 일반적인 프레임워크입니다. 이러한 유연성은 다양한 응용 분야에 적응할 수 있는 일반화 잠재력을 높입니다.[^1_1]

### 5. 미래 연구에 미치는 영향과 고려사항

## 연구에 미치는 영향

**인과 표현 학습과 시계열 예측의 통합**: 이 논문은 인과 표현 학습(Causal Representation Learning)의 이론적 틀을 온라인 시계열 예측에 성공적으로 적용한 첫 사례 중 하나입니다. 이는 두 분야 간의 다리 역할을 하며, 향후 연구가 이론적 보장과 실용적 성능을 모두 추구할 수 있는 방향을 제시합니다.[^1_1]

**알려지지 않은 개입 모델링**: 실제 시계열 데이터에서 발생하는 알려지지 않은 개입(예: 정책 변화, 시장 충격)을 명시적으로 모델링하는 접근법은 다음 영역에서 중요한 영향을 미칠 것입니다:

- 금융 시계열 분석 (갑작스러운 정책 변화)[^1_1]
- 의료 데이터 분석 (치료 개입)
- 공급망 관리 (외부 충격)

**해석 가능성 강화**: 장단기 상태의 명시적 분리는 모델의 예측이 장기 트렌드에서 비롯되는지 단기 변동에서 비롯되는지 이해할 수 있게 하여 블랙박스 모델의 한계를 극복합니다.[^1_1]

## 향후 연구 시 고려할 점

**1. 다양한 비정상성 유형 처리**

논문은 개입으로 인한 비정상성에 초점을 맞추지만, 실제 데이터에는 점진적 드리프트, 계절적 변화, 구조적 변화 등 다양한 유형의 비정상성이 존재합니다. 향후 연구는 다음을 탐구해야 합니다:[^1_1]

- 복합적 비정상성 패턴 처리
- 개입의 강도 및 지속 시간 모델링
- 여러 시간 척도에서의 변화 감지

**2. 개입 감지 메커니즘 개선**

현재 모델은 중단된 의존성 제약을 통해 개입을 간접적으로 감지하지만, 명시적인 개입 감지 모듈을 추가하면 다음이 가능합니다:

- 개입 시점의 정확한 식별
- 개입 유형 분류
- 개입 영향 정량화

**3. 확장성 및 효율성**

메모리 사용량 문제를 해결하기 위해 다음을 고려해야 합니다:[^1_1]

- 경량 사전 네트워크 설계
- 점진적 학습 전략
- 모델 압축 기법 적용

**4. 다변량 시계열에서의 변수 간 관계**

현재 모델은 각 변수의 장단기 분리에 초점을 맞추지만, 변수 간 인과 관계도 중요합니다:

- 변수 간 장단기 인과 구조 학습
- 동적 그래프 구조 변화 모델링
- 교차 변수 개입 효과 분석

**5. 이론적 가정의 완화**

실제 응용에서 모든 가정이 충족되지 않을 수 있으므로:

- 부분적 식별 가능성 연구
- 가정 위반 시 성능 저하 정량화
- 적응적 가정 검증 메커니즘


### 6. 2020년 이후 관련 최신 연구 비교 분석

## 온라인 시계열 예측 방법론

**FSNet (2022) **: 부분 도함수로 단기 정보를 특성화하고 연상 메모리로 장기 의존성을 보존합니다. LSTD와 비교하여 장단기 상태를 명시적으로 분리하지 않아 Exchange 데이터셋에서 LSTD(MSE 0.039) 대비 FSNet(MSE 0.113)으로 성능이 낮습니다.[^1_2][^1_1]

**OneNet (2024) **: 강화 학습 기반으로 시간적 상관관계와 교차 변수 의존성 모델을 동적으로 조정합니다. LSTD는 OneNet보다 Exchange 데이터셋에서 약 18% 향상된 성능을 보입니다.[^1_2][^1_1]

**D3A (2024) **: 시간적 분포 변화를 먼저 감지한 후 공격적으로 모델을 업데이트합니다. 그러나 장단기 분리 없이는 환경 변화에 효과적으로 적응하지 못합니다.[^1_2][^1_1]

**TOT Framework (2025) **: 온라인 예측에 대한 이론적 보장을 제공하며, 잠재 변수가 베이즈 위험을 감소시킴을 증명합니다. LSTD와 유사하게 잠재 변수 식별에 초점을 맞추지만, LSTD는 장단기 분리에 특화되어 있습니다.[^1_3][^1_2]

**CEP (2025) **: 반복적 개념 드리프트를 다루기 위한 프라이버시 보존 프레임워크로, 경량 통계 유전자를 사용하여 개념 식별과 예측을 분리합니다. 역사적 데이터를 저장하지 않고도 20% 이상의 예측 오류 감소를 달성합니다.[^1_4]

**ADAPT-Z (2025) **: 잠재 특징 공간 수정을 통해 분포 변화를 근본적으로 해결하며, 다단계 예측에서의 지연된 피드백 문제를 효과적으로 해결합니다.[^1_5]

## 인과 표현 학습 접근법

**Temporally Disentangled Representation Learning (TDRL, 2022) **: 정상 환경에서 시간 지연 잠재 인과 프로세스의 식별 가능성을 확립했습니다. LSTD는 이를 확장하여 알려지지 않은 개입 하의 비정상 환경을 다룹니다.[^1_6][^1_1]

**NCTRL (2023) **: 알려지지 않은 비정상성 하에서 시간 지연 잠재 인과 변수를 재구성합니다. LSTD와 유사한 목표를 가지지만, LSTD는 온라인 예측에 특화되어 실시간 적응에 더 적합합니다.[^1_7]

**SYNC (2025) **: 정적-동적 인과 표현 학습을 통해 시간 인식 인과 표현을 학습하며, 동적 인과 요인과 인과 메커니즘 드리프트를 통합한 시간 인식 구조적 인과 모델을 설계합니다.[^1_8]

**CReP (2025) **: 원인 관련, 효과 관련, 비인과적 표현으로 원 공간을 세 가지 직교 잠재 요인으로 분해하며, 시공간 정보 변환 메커니즘을 통해 인과 상호작용과 미래 상태 예측을 동시에 모델링합니다.[^1_9]

**CausalFormer (2024) **: 시간적 우선순위 제약 하에서 다중 커널 인과 컨볼루션을 사용하여 시계열의 인과 표현을 학습하는 해석 가능한 트랜스포머 기반 모델입니다.[^1_10]

## 비정상성 및 개념 드리프트 처리

**SSR4OTS (2025) **: 자기 지도 학습과 데이터 재생을 통해 데이터 증강 및 자기 증류 메커니즘으로 예측 성능을 향상시킵니다. 분포 변화에 대한 모델의 안정성을 강화하지만 인과 구조를 명시적으로 모델링하지 않습니다.[^1_11]

**OASIS (2025) **: 금융 시계열 예측에서 실제 및 가상 개념 드리프트를 모두 처리하는 온라인 적응 시스템입니다. LSTD의 개입 기반 접근법과 유사하게 드리프트 유형을 구분합니다.[^1_12]

**EAMDrift (2023) **: 예측 불가능한 패턴을 처리하기 위해 성능 메트릭에 따라 각 예측에 가중치를 부여하여 여러 개별 예측기의 예측을 결합합니다.[^1_13]

## 비교 우위 및 독창성

LSTD의 주요 차별점:

1. **이론적 기초**: 알려지지 않은 개입 하에서 장단기 상태의 블록별 식별 가능성에 대한 엄격한 증명을 제공합니다. 이는 TOT나 TDRL과 유사하지만 개입 모델링에 특화되어 있습니다.[^1_3][^1_6][^1_1]
2. **명시적 장단기 분리**: FSNet, OneNet과 달리 장단기 상태를 명시적으로 분리하여 각각에 대한 적절한 업데이트 전략을 적용합니다.[^1_1]
3. **제약 기반 분리**: 매끄러움 제약과 중단된 의존성 제약을 통해 장단기 분리를 강제하는 독창적인 접근법을 사용합니다.[^1_1]
4. **실증적 우수성**: 특히 개입이 빈번한 금융 데이터(Exchange)에서 기존 방법들을 크게 능가합니다.[^1_1]

## 향후 통합 가능성

최신 연구들과의 통합을 통해 다음과 같은 확장이 가능합니다:

- **CEP의 프라이버시 보존 메커니즘 통합**: 역사적 데이터 저장 없이 LSTD의 장단기 분리 활용[^1_4]
- **CReP의 공간-시간 정보 변환과 결합**: 더 풍부한 인과 표현 학습[^1_9]
- **SSR4OTS의 자기 지도 학습 전략 적용**: 제한된 데이터에서의 성능 향상[^1_11]
- **TOT의 베이즈 위험 경계와 통합**: 더 강력한 이론적 보장[^1_3]


## 결론

이 논문은 온라인 시계열 예측에서 인과 표현 학습의 이론적 엄밀함과 실용적 성능을 성공적으로 결합한 중요한 연구입니다. 2020년 이후 최신 연구들과 비교할 때, 알려지지 않은 개입 하에서의 장단기 상태 분리라는 독특한 관점을 제시하며, 특히 금융 및 비정상 환경에서의 예측 성능을 크게 향상시킵니다. 향후 연구는 다양한 비정상성 유형 처리, 계산 효율성 개선, 그리고 최신 프라이버시 보존 및 자기 지도 학습 기법과의 통합을 통해 이 프레임워크를 더욱 발전시킬 수 있을 것입니다.[^1_2][^1_3][^1_1]
<span style="display:none">[^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38]</span>

<div align="center">⁂</div>

[^1_1]: 2502.12603v1.pdf

[^1_2]: https://arxiv.org/abs/2510.18281

[^1_3]: https://arxiv.org/pdf/2510.18281.pdf

[^1_4]: https://arxiv.org/abs/2506.14790

[^1_5]: https://arxiv.org/html/2509.03810v1

[^1_6]: https://openreview.net/pdf?id=Vi-sZWNA_Ue

[^1_7]: https://arxiv.org/abs/2310.18615

[^1_8]: https://arxiv.org/abs/2506.17718

[^1_9]: https://ui.adsabs.harvard.edu/abs/2025CmPhy...8..242C/abstract

[^1_10]: https://arxiv.org/html/2406.16708v1

[^1_11]: https://ieeexplore.ieee.org/document/11086851/

[^1_12]: https://sol.sbc.org.br/index.php/eniac/article/view/38818

[^1_13]: https://arxiv.org/pdf/2305.19837.pdf

[^1_14]: https://jrucs.iq/index.php/JAUCS/article/view/722

[^1_15]: https://iaj.aktuaris.or.id/index.php/iaj/article/view/28

[^1_16]: https://www.semanticscholar.org/paper/08d68bf827f5b45fae571fdacb9346faab95ff8c

[^1_17]: https://arxiv.org/abs/2502.12603

[^1_18]: https://ieeexplore.ieee.org/document/10787008/

[^1_19]: https://ieeexplore.ieee.org/document/11137629/

[^1_20]: https://arxiv.org/pdf/2310.10688.pdf

[^1_21]: http://arxiv.org/pdf/2405.13522.pdf

[^1_22]: http://arxiv.org/pdf/2410.22981.pdf

[^1_23]: http://arxiv.org/pdf/2412.17603.pdf

[^1_24]: http://arxiv.org/pdf/2410.18959.pdf

[^1_25]: https://arxiv.org/html/2503.22747v1

[^1_26]: http://arxiv.org/pdf/2501.01087.pdf

[^1_27]: https://www.arxiv.org/pdf/2602.03981.pdf

[^1_28]: https://arxiv.org/pdf/2509.09176.pdf

[^1_29]: https://arxiv.org/html/2602.03981v1

[^1_30]: https://arxiv.org/pdf/2406.08627.pdf

[^1_31]: https://doaj.org/article/fd5ec18695264344892ab4569a92189e

[^1_32]: https://arxiv.org/html/2502.12603v1

[^1_33]: https://arxiv.org/pdf/2601.15514.pdf

[^1_34]: https://arxiv.org/html/2510.14049v1

[^1_35]: https://royalsocietypublishing.org/rsta/article/379/2194/20200209/41189/Time-series-forecasting-with-deep-learning-a

[^1_36]: https://www.sciencedirect.com/science/article/pii/S1574013725001595

[^1_37]: https://www.nature.com/articles/s41467-025-63786-4

[^1_38]: https://forecastio.ai/blog/time-series-forecasting

