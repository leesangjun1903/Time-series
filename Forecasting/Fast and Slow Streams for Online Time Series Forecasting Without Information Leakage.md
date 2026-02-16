# Fast and Slow Streams for Online Time Series Forecasting Without Information Leakage

## 핵심 주장 및 주요 기여

이 논문은 기존 온라인 시계열 예측(OTSF) 연구의 두 가지 핵심 문제를 제기합니다. 첫째, **정보 누출(information leakage)** 문제로, 모델이 이미 역전파에 사용된 시점을 재평가하여 성능을 과대평가합니다. 둘째, **실용성 문제**로, 예측 시퀀스가 대부분 이미 관측된 시점을 포함하고 미래는 단 한 스텝만 예측합니다.[^1_1]

저자들은 OTSF 문제를 재정의하고, 이중 스트림 프레임워크(DSOF)를 제안합니다. 이 프레임워크는 경험 재생(Experience Replay)을 사용하는 느린 스트림과 시간차 학습(Temporal Difference Learning)을 활용하는 빠른 스트림을 결합하여, teacher-student 모델을 잔차 학습 전략으로 업데이트합니다.[^1_1]

## 해결하고자 하는 문제

### 기존 OTSF 설정의 문제점

기존 연구에서는 시간 $t = i$에서 모델이 $t = i - L - H + 2$부터 $t = i - H + 1$까지의 데이터를 입력받아 $t = i - H + 2$부터 $t = i + 1$까지 예측합니다. 예를 들어, $L = 5$, $H = 4$일 때, $t = 10$에서 $t = 3$ ~ $t = 7$의 데이터로 $t = 8$~$t = 11$을 예측하고, $t = 11$에서 ground truth를 받아 파라미터를 업데이트합니다.[^1_1]

이 설정의 문제는 $t = 12$ 에서 평가할 때 $t = 9$ ~ $t = 11$의 데이터가 이미 $t = 11$에서 역전파에 사용되었다는 점입니다. 이는 편향된 평가 결과를 초래하며 실제 성능을 과대평가합니다.[^1_1]

### 재정의된 OTSF 설정

논문은 두 가지 핵심 기준을 제시합니다:[^1_1]

1. 모델은 역전파에 사용되지 않은 시점에서만 평가되어야 함
2. 출력 윈도우 $H$는 미지의 미래 시점 수와 동일해야 함

새로운 설정에서 $t = 10$일 때 $t = 6$~$t = 10$의 데이터로 $t = 11$~$t = 14$를 예측하고, $t = 11$에서는 $t \leq 11$까지의 ground truth만 사용하여 업데이트합니다. 평가는 $t = 14$ 이후에만 가능합니다.[^1_1]

## 제안하는 방법

### Teacher-Student 모델 구조

**Teacher 모델**은 주어진 입력 시퀀스에 대해 coarse prediction을 생성합니다:[^1_1]

$$
\hat{X}^{(T,i)}_{i+1:i+H} = f^{(i)}_T(X_{i-L+1:i}; \theta^{(T)})
$$

**Student 모델**은 경량 MLP로, teacher의 예측을 정제합니다:[^1_1]

$$
\hat{X}^{(S,i)}_{i+1:i+H} = f^{(i)}_S(\text{Concat}(X_{i-L+1:i}, \hat{X}^{(T,i)}_{i+1:i+H}); \theta^{(S)})
$$

**최종 예측**은 두 모델의 합으로 계산됩니다:[^1_1]

$$
\hat{X}^{(i)}_{i+1:i+H} = \hat{X}^{(T,i)}_{i+1:i+H} + \hat{X}^{(S,i)}_{i+1:i+H}
$$

### 느린 스트림: Experience Replay

느린 스트림은 경험 재생(ER)을 사용하여 완전한 ground truth로 모델을 업데이트합니다. 재생 버퍼 $B$는 고정 크기 $N_B$를 가지며 FIFO 방식으로 작동합니다. $t = i$에서 ground truth $x_i$를 받으면, $(X_{i-H-L-1:i-H}, X_{i-H+1:i})$ 쌍이 버퍼에 추가됩니다.[^1_1]

이 방법은 완전한 label sequence를 기다려야 하므로 "느린" 업데이트로 간주됩니다. ER은 재앙적 망각(catastrophic forgetting)을 방지하고 최근 데이터의 노이즈에 과적합하는 것을 억제합니다.[^1_1]

### 빠른 스트림: Temporal Difference Learning

빠른 스트림은 불완전한 ground truth 문제를 해결하기 위해 시간차 학습에서 영감을 받았습니다. $t = i-1$에서 $\hat{X}^{(i-1)}_{i:i+H-1}$을 예측한 후, $t = i$에서 ground truth $x_i$가 도착하면, teacher 모델의 예측으로 **pseudo label**을 생성합니다:[^1_1]

$$
\tilde{X}^{(i)}_{i:i+H-1} = [x_i, \hat{X}^{(T,i)}_{i+1:i+H-1}]
$$

**Temporal Difference Loss**는 기하학적 감쇠 인자 $\gamma$를 사용하여 먼 미래 예측의 영향을 줄입니다:[^1_1]

$$
\ell^{(i-1)}_{TD}(\hat{X}^{(i-1)}_{i:i+H-1}, \tilde{X}^{(i)}_{i:i+H-1}) = \frac{1}{H}\sum_{j=1}^{H}\gamma^{j-1}\|\hat{x}^{(i)}_j - \tilde{x}^{(i)}_j\|^2
$$

이 접근법은 완전한 ground truth를 기다리지 않고 최신 정보를 활용하여 근미래 예측을 개선합니다.[^1_1]

### 통합 알고리즘

온라인 학습 단계에서 새로운 데이터 포인트가 도착할 때마다:[^1_1]

1. 재생 버퍼 업데이트 및 ER 수행 (배치 크기 $N_b$로 무작위 샘플링)
2. Student 모델을 TD loss로 업데이트 (Teacher 모델은 고정)
3. 다음 스텝 예측 생성

## 성능 향상 및 일반화 능력

### 실험 결과

논문은 6개 벤치마크 데이터셋(Electricity, ETTh2, ETTm1, Exchange, Traffic, Weather)에서 실험을 수행했습니다. DSOF는 DLinear, FITS, FSNet, OneNet, iTransformer, PatchTST, NSTrans 등 다양한 backbone 아키텍처에 적용되었습니다.[^1_1]

**주요 성과**:[^1_1]

- Electricity 데이터셋에서 FSNet과 NSTransformer를 teacher 모델로 사용 시 MSE가 50% 이상 감소
- 대부분의 데이터셋과 아키텍처에서 배치 학습 대비 일관된 성능 향상
- 분포 변화가 있는 환경에서 효과적으로 적응


### 일반화 성능 향상 메커니즘

**Bias-Variance Trade-off**: ER과 TD loss의 상호작용은 편향-분산 균형(bias-variance trade-off)을 달성합니다. ER은 역사적 데이터로 학습하여 예측 가능한 패턴에서 안정적이고 낮은 분산을 제공하지만, 급격한 분포 변화에는 시대에 뒤떨어진 가정을 할 수 있습니다. 반면 TD loss는 pseudo label을 사용하여 급격한 변화를 빠르게 감지하지만 노이즈에 민감하여 높은 분산을 보입니다.[^1_1]

**개념 표류(Concept Drift) 처리**: 실험 분석에 따르면, 분포 전환 직후에는 TD loss가 더 나은 성능을 보이며, 새로운 분포에 안정화된 후에는 ER이 우수합니다. 두 구성요소를 결합함으로써 각각의 강점을 활용하여 다양한 데이터 분포에서 더 정확하고 신뢰할 수 있는 예측을 제공합니다.[^1_1]

**재앙적 망각 방지**: ER은 연속 학습에서 일반적인 접근법으로, 최근 샘플을 재생하여 과거 지식을 유지하면서 새로운 패턴을 학습합니다. 이는 모델이 최신 데이터에만 과적합하는 것을 방지합니다.[^1_1]

## 한계

1. **계산 비용**: DSOF는 다른 온라인 학습 프레임워크보다 효율성이 떨어집니다. 특히 Traffic 데이터셋과 같이 특징 차원이 큰 경우 wall time과 processor time이 증가합니다.[^1_1]
2. **하이퍼파라미터 민감도**: 일부 경우 배치 학습이 온라인 접근법을 능가하는데, 이는 정규화 선택과 하이퍼파라미터가 배치 학습에 최적화되어 있고 온라인 학습에는 추가 최적화가 필요하기 때문입니다.[^1_1]
3. **Student 모델의 선택적 효과**: Student 모델을 포함하는 것이 항상 최적의 성능을 보장하지는 않습니다. 일부 경우 teacher 모델만 사용하는 것이 더 나은 결과를 제공합니다.[^1_1]
4. **제한된 적용 범위**: 논문은 더 긴 예측 기간으로의 확장, 더 나은 모델 아키텍처 식별, 더 고급 반지도 학습 기법 사용 등 탐구할 영역이 여전히 남아있음을 인정합니다.[^1_1]

## 앞으로의 연구에 미치는 영향

### 이론적 기여

**정보 누출 문제 제기**: 이 논문은 기존 OTSF 연구의 근본적인 평가 결함을 지적했습니다. 재정의된 설정은 향후 연구의 새로운 기준이 될 가능성이 높습니다.[^1_1]

**강화학습과 시계열 예측의 연결**: TD learning을 OTSF에 적용한 것은 두 분야 간의 새로운 연결고리를 제시합니다. 이는 강화학습의 다른 기법들도 시계열 예측에 적용될 수 있는 가능성을 시사합니다.[^1_1]

### 실용적 기여

**모델 독립적 프레임워크**: DSOF는 다양한 backbone 아키텍처에 적용 가능한 일반적 프레임워크입니다. 이는 기존 예측 모델들을 온라인 학습 환경에 쉽게 적응시킬 수 있게 합니다.[^1_1]

**실시간 응용 가능성**: 전력 소비 모니터링, 기후 모델링, 소매, 주식 시장 예측 등 다양한 실시간 응용 분야에서 활용 가능합니다.[^1_1]

## 앞으로 연구 시 고려할 점

### 1. 계산 효율성 개선

현재 DSOF의 주요 약점은 계산 비용입니다. 향후 연구는 다음을 고려해야 합니다:[^1_1]

- ER을 대체할 수 있는 더 낮은 런타임 복잡도를 가진 방법 개발
- 더 작은 재생 배치 크기 사용 최적화
- 고차원 데이터에 대한 효율적인 처리 메커니즘


### 2. 개념 표류 감지 통합

논문은 명시적인 개념 표류 감지 메커니즘을 포함하지 않습니다. Detect-and-Adapt 방법과 같이 개념 표류를 먼저 식별한 후 적응하는 접근법과의 통합이 유용할 수 있습니다.[^1_2][^1_3][^1_4][^1_1]

### 3. Pseudo Label 생성 개선

현재는 teacher 모델의 예측만 사용하지만, 더 고급 반지도 학습 기법을 활용하여 pseudo label의 품질을 향상시킬 수 있습니다.[^1_1]

### 4. 더 긴 예측 horizon 탐구

논문은 $H = 1, 24, 48$에 초점을 맞췄습니다. 더 긴 예측 기간에서의 성능과 한계를 탐구할 필요가 있습니다.[^1_1]

### 5. 멀티모달 통합

최근 연구는 텍스트 단서를 활용한 시계열 예측을 탐구하고 있습니다. DSOF 프레임워크를 멀티모달 데이터와 통합하는 것도 유망한 방향입니다.[^1_5]

## 2020년 이후 관련 최신 연구 비교 분석

### Concept Drift 처리 방법

**Detect-and-Adapt (2024)**: 개념 표류를 먼저 감지한 후 다른 학습 방식으로 모델을 적응시킵니다. DSOF와 달리 명시적 표류 감지를 사용하지만, 정보 누출 문제를 다루지 않습니다.[^1_6][^1_4]

**OASIS (2025)**: 금융 시계열에서 실제 및 가상 개념 표류를 구분하여 처리합니다. DSOF는 모든 유형의 표류를 통합된 방식으로 처리합니다.[^1_7]

**Continuous Evolution Pool (CEP, 2025)**: 원시 샘플 대신 통계적 유전자를 사용하여 개인정보를 보호하면서 재발하는 개념 표류를 처리합니다. 역사적 ground truth 없이 20% 이상 오류 감소를 달성했습니다.[^1_3]

**Proactive Model Adaptation (Proceed, 2024)**: 훈련 샘플과 테스트 샘플 간의 개념 표류를 추정하고 사전에 모델을 적응시킵니다. DSOF가 반응적인 반면, Proceed는 능동적 접근입니다.[^1_4]

### 온라인 학습 전략

**SSR4OTS (2025)**: 자기지도 학습과 데이터 재생을 사용하여 데이터 부족 및 분포 변화 문제를 해결합니다. DSOF와 유사하게 데이터 재생을 사용하지만, 자기지도 학습 전략은 추가적입니다.[^1_8]

**LSTD (2025)**: 알려지지 않은 개입 하에서 장단기 상태를 분리합니다. DSOF의 빠른/느린 스트림과 개념적으로 유사하지만, 인과 추론 관점을 취합니다.[^1_2]

**ADAPT-Z (2025)**: 파라미터 업데이트 대신 잠재 특징 공간 수정을 통해 분포 변화를 처리합니다. 이는 DSOF의 파라미터 기반 접근과 근본적으로 다릅니다.[^1_9]

**TOT (2025)**: 잠재 변수를 이론적 보장과 함께 사용하는 프레임워크를 제공합니다. 시간 디코더와 독립 노이즈 추정기를 사용합니다.[^1_6]

### Temporal Difference Learning 응용

**기초 이론 (1997, 2020)**: TD learning의 수렴 증명 및 그래디언트 분할로서의 해석을 제공합니다. DSOF는 이러한 이론적 토대를 실용적인 시계열 예측에 적용합니다.[^1_10][^1_11]

**CaTT (2025)**: 시간적 일관성을 활용한 대조 학습을 제안하며, 모든 시간 스텝을 병렬로 대조합니다. DSOF와는 다른 시간적 정보 활용 방식입니다.[^1_12]

### Foundation Models 및 Zero-Shot 접근

**CHRONOS (2024-2025)**: 사전 학습된 언어 모델을 시계열 예측에 적용하여 제로샷 예측을 가능하게 합니다. DSOF는 온라인 적응에 초점을 맞추는 반면, CHRONOS는 사전 학습과 전이 학습에 중점을 둡니다.[^1_13][^1_14]

**Context is Key (2025)**: 텍스트 정보를 수치 데이터와 통합하는 벤치마크를 제공합니다. 향후 DSOF를 멀티모달 설정으로 확장할 가능성을 시사합니다.[^1_15]

### 핵심 차별점

DSOF의 주요 차별점은 **정보 누출 문제를 명시적으로 해결**하고, **ER과 TD learning을 결합**하여 안정성과 적응성의 균형을 달성한다는 것입니다. 대부분의 최신 연구는 개념 표류 처리나 모델 아키텍처 개선에 초점을 맞추지만, 평가 방법론의 근본적 결함을 다루지 않습니다.[^1_1]

이 논문은 온라인 시계열 예측 커뮤니티에 평가 프로토콜을 재고하도록 촉구하며, 향후 연구가 더 엄격하고 실용적인 기준을 따르도록 영향을 미칠 것입니다.[^1_1]
<span style="display:none">[^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39]</span>

<div align="center">⁂</div>

[^1_1]: 9019_Fast_and_Slow_Streams_for.pdf

[^1_2]: https://arxiv.org/abs/2502.12603

[^1_3]: https://arxiv.org/abs/2506.14790

[^1_4]: http://www.arxiv.org/abs/2412.08435

[^1_5]: https://arxiv.org/pdf/2406.08627.pdf

[^1_6]: https://arxiv.org/abs/2510.18281

[^1_7]: https://sol.sbc.org.br/index.php/eniac/article/view/38818

[^1_8]: https://ieeexplore.ieee.org/document/11086851/

[^1_9]: https://arxiv.org/html/2509.03810v1

[^1_10]: https://arxiv.org/abs/2010.14657

[^1_11]: https://www.mit.edu/~jnt/Papers/J063-97-bvr-td.pdf

[^1_12]: https://arxiv.org/html/2410.15416v2

[^1_13]: https://ieeexplore.ieee.org/document/11137629/

[^1_14]: https://arxiv.org/pdf/2310.10688.pdf

[^1_15]: http://arxiv.org/pdf/2410.18959.pdf

[^1_16]: https://jrucs.iq/index.php/JAUCS/article/view/722

[^1_17]: https://iaj.aktuaris.or.id/index.php/iaj/article/view/28

[^1_18]: https://www.semanticscholar.org/paper/08d68bf827f5b45fae571fdacb9346faab95ff8c

[^1_19]: https://ieeexplore.ieee.org/document/10787008/

[^1_20]: http://arxiv.org/pdf/2405.13522.pdf

[^1_21]: http://arxiv.org/pdf/2410.22981.pdf

[^1_22]: https://arxiv.org/pdf/2305.19837.pdf

[^1_23]: http://arxiv.org/pdf/2412.17603.pdf

[^1_24]: https://arxiv.org/html/2503.22747v1

[^1_25]: http://arxiv.org/pdf/2501.01087.pdf

[^1_26]: https://www.arxiv.org/pdf/2602.03981.pdf

[^1_27]: https://arxiv.org/pdf/2509.09176.pdf

[^1_28]: https://arxiv.org/html/2602.03981v1

[^1_29]: https://doaj.org/article/fd5ec18695264344892ab4569a92189e

[^1_30]: https://arxiv.org/pdf/2304.01512.pdf

[^1_31]: https://arxiv.org/html/2511.12104v1

[^1_32]: https://arxiv.org/abs/2304.01512

[^1_33]: https://royalsocietypublishing.org/rsta/article/379/2194/20200209/41189/Time-series-forecasting-with-deep-learning-a

[^1_34]: https://www.sciencedirect.com/science/article/pii/S1574013725001595

[^1_35]: https://www.nature.com/articles/s41467-025-63786-4

[^1_36]: https://forecastio.ai/blog/time-series-forecasting

[^1_37]: https://minkull.github.io/publications/OliveiraEtAlICTAI2017.pdf

[^1_38]: https://github.com/ddz16/TSFpaper

[^1_39]: https://en.wikipedia.org/wiki/Temporal_difference_learning

