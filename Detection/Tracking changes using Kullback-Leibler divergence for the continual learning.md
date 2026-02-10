# Tracking changes using Kullback-Leibler divergence for the continual learning

- 이 논문은 **레이블이나 분류기 없이도** 다차원 데이터 스트림에서 분포 변화(컨셉 드리프트)를 감지하기 위해, 인접 청크 간 Kullback–Leibler(KL) 발산을 계산하고 그 시계열의 기울기를 임계값으로 판단하는 “KLD” 검출기를 제안한다.[^1_1][^1_2]
- 분포 형태에 대한 가정 없이 적용 가능하고, 단일 제어 파라미터 $\alpha$가 민감도–거짓 경보(trade‑off)를 직관적으로 조절하며, 시뮬레이션 데이터에서 다양한 점진적 드리프트를 빠르게 포착할 수 있음을 실험적으로 보인다.[^1_3][^1_1]

***

## 1. 핵심 주장과 주요 기여 (간결 요약)

- **핵심 주장**

1) 컨셉 드리프트를 “입력–출력 분포의 변동” 자체로 모니터링하면, 분류기 성능 기반(에러율 기반) 방법보다 더 일반적이고 조기 탐지가 가능하다.[^1_1]
2) 이러한 분포 변화를 KL 발산 기반 거리로 표현하고, 시간축에서의 변화율(1차 미분)을 통계적 임계 규칙으로 감지하면, 빠르고 해석 가능한 드리프트 검출기가 된다.[^1_1]
- **주요 기여**[^1_3][^1_1]
    - 스트림 데이터 청크 간 KL 발산을 이용해 분포 변화를 연속적으로 모니터링하는 일반적 프레임워크 제안.
    - KL 기반 컨셉 드리프트 검출기 KLD 정의 (pmf 추정, 가중 KL, 스무딩, 1차 도함수, 임계값 결정 규칙 포함).
    - 단일 실수 제어 파라미터 $\alpha$를 통한 간단한 but 튜너블한 의사결정 규칙 제안 및 민감도 분석.
    - Stream-learn로 생성한 비정상 이진 데이터 스트림(4/6 feature, 20 drifts)에서, 결정트리/가우시안 NB 기반 검출과 시각적으로 비교하여 KLD가 분포 변화를 잘 추적함을 보임.

***

## 2. 논문의 문제 정의, 방법(수식), “모델” 구조, 성능·한계

### 2.1 해결하고자 하는 문제

1. **문제 상황**
    - 스트리밍/continual learning 환경에서 데이터의 생성 분포가 시간에 따라 변하는 **컨셉 드리프트**가 발생하면, 고정 모델의 성능이 급격히 저하된다.[^1_1]
    - 기존 드리프트 검출은
        - (a) 분류기의 예측 에러(DDM, EDDM, ADWIN 등)에 의존하거나,
        - (b) 특정 통계 검정(KS test, chi‑square 등)에 기반해,
        - 대부분 **레이블** 또는 **특정 분포 가정**·검정 구조에 묶여 있다.[^1_4][^1_1]
2. **연구 질문**
    - “**레이블 없이, 분포 형태 가정 없이, 다차원 스트림에서 드리프트를 빠르고 일반적으로 감지할 수 있는가?**”[^1_1]
    - 특히, “인접 청크의 **원시 데이터(topological/빈도 구조)** 만으로 드리프트 시점을 포착할 수 있는가?”[^1_1]

### 2.2 제안 방법: KLD 검출기 (수식 포함)

#### (1) 데이터 스트림과 청크 정의

- 스트림을 청크 시퀀스로 본다.

$$
\mathcal{S} = \{S_1, S_2, \dots, S_k, \dots\}, 
\quad S_i = \{z_i^{(1)}, \dots, z_i^{(K)}\},\; z_i^{(k)} = (u_i^{(k)}, y_i^{(k)}),
$$

여기서 $u_i^{(k)} \in \mathcal{U} \subset \mathbb{R}^p$, $y_i^{(k)} \in \mathcal{Y}$ (이 논문에서는 이진 분류).[^1_1]


#### (2) 각 청크에서의 경험적 pmf 추정

- 입력 공간 $\mathcal{U}$를 다차원 정규 격자 $J$개 bin으로 등분: $b_i^j, j=1,\dots, J$.[^1_1]
- 각 bin과 클래스 $l$에 대해 조건부 확률 추정:

```math
\Pr\big(y_i = l \mid \{u_i : u_i^{(k)} \in b_i^j,\ \forall k\}\big)
=
\frac{\sum_{k=1}^{K} \mathbf{1}\{u_i^{(k)} \in b_i^j\ \wedge\ y_i^{(k)} = l\}}
     {\sum_{k=1}^{K} \mathbf{1}\{u_i^{(k)} \in b_i^j\}}
\tag{3}
```

(분모가 0인 bin은 smoothing으로 대응; KL 발산 수치 안정성 위해 $\varepsilon>0$ 추가).[^1_1]


#### (3) bin별 KL 발산

- 청크 $S_i$와 $S_t$에 대해, 같은 위치의 bin $b_i^j, b_t^j$의 **조건부 pmf** 차이를 KL로 측정:

```math
d_{i,t}^j 
= KL\!\left(
    \Pr\big(y_i \mid \{u_i: u_i^{(k)} \in b_i^j\}\big)
    \parallel
    \Pr\big(y_t \mid \{u_t: u_t^{(k)} \in b_t^j\}\big)
  \right)
```
- 기본 KL 정의는

$$
KL(p \parallel q)=\sum_x p(x)\log\frac{p(x)}{q(x)}.[file:1]
\tag{1}
$$


#### (4) 청크 간 전체 거리 (가중 KL)

- 단순 평균:

$$
D(S_i, S_t) = \frac{1}{J}\sum_{j=1}^{J} d_{i,t}^j.
\tag{5}
$$
- 제안하는 **가중 평균**: bin별 샘플링 확률 $\gamma_i^j$를 가중치로 사용.

$$
\gamma_i^j = \frac{1}{K}\sum_{k=1}^{K} \mathbf{1}\{u_i^{(k)} \in b_i^j\},
$$

$$
D(S_i, S_t) = \frac{1}{J}\sum_{j=1}^{J} \gamma_i^j\, d_{i,t}^j.
\tag{6}
$$
- KLD 알고리즘은 인접 청크만 사용하므로 $D(S_k, S_{k+1})$ 시퀀스를 생성:

$$
\{D(S_1,S_2), D(S_2,S_3), \dots, D(S_k,S_{k+1}), \dots\}.
$$


#### (5) 스무딩과 1차 도함수 기반 의사결정 규칙

- 원시 시퀀스는 노이즈·아웃라이어에 민감하므로, 이동평균(윈도우 5) 또는 LOWESS로 스무딩하여 연속 곡선 $\tilde{D}(t)$를 얻는다.[^1_1]
- $\tilde{D}(t)$의 1차 미분 값을 이산 시퀀스 $l_i$로 두고, 평균과 표준편차를 계산:

$$
\bar{l} = \frac{1}{n}\sum_{i=1}^{n} l_i,\quad
\sigma(l) = \sqrt{\frac{1}{n}\sum_{i=1}^{n} (l_i - \bar{l})^2}.
$$
- **드리프트 검출 규칙** (임계값 기반 outlier rule):

$$
l_i \notin [\bar{l} - \alpha\sigma(l),\ \bar{l} + \alpha\sigma(l)]
\tag{7}
$$

이면 시간 $i$를 “critical point(컨셉 드리프트 발생)”로 간주하고, 집합 $C$에 추가.[^1_1]
- $\alpha$는 hyper-parameter로, 작을수록 민감하지만 false positive 증가, 클수록 보수적이며 false negative 증가; 실험적으로 $\alpha\in[1.5,2]$에서 좋은 성능 보고.[^1_1]


#### (6) 알고리즘 구조 (의사코드 관점 “모델 구조”)

KLD는 학습 모델이라기보다는 **통계적 모니터링 파이프라인**에 가깝다.[^1_1]

1. 초기 청크 $S_0$ 도착 → 입력 공간 분할, pmf 추정.
2. 새 청크 $S_i$ 도착: pmf 추정.
3. 이전 청크 $S_{i-1}$와의 bin별 KL($d_{i-1,i}^j$) 계산, 가중 합으로 $D(S_{i-1},S_i)$.
4. 지금까지의 $\{D\}$ 시퀀스를 스무딩, 1차 도함수 $l_i$ 계산.
5. $l_i$가 식 (7)을 만족하면 시점 $i$를 드리프트로 기록.

이 전체 절차가 “모델 구조”라고 볼 수 있으며, 내부에 별도의 딥러닝/분류기 구조는 없다.[^1_1]

### 2.3 성능 향상 및 한계

#### (1) 실험 설정 요약

- 데이터: Stream-learn 라이브러리로 생성한 이진 분류 스트림 8개.[^1_5][^1_1]
    - feature 수: 4, 6 (각각 4개 데이터셋).
    - 청크 수: 10,000, 각 청크 250 샘플.
    - 시뮬레이션된 드리프트 수: 20 (sigmoid spacing 99 → 점진적 드리프트).
- J(빈 수): 경험적으로 $J = 5 \times p$ 로 고정.[^1_1]
- 비교 기준:
    - 자체 KL 곡선과 스무딩 곡선의 시각적 비교.
    - 결정트리(DT), Gaussian Naive Bayes(GNB) 분류기의 성능 변화 곡선(정확도 derivative)과 비교.[^1_1]


#### (2) 관찰된 장점 (성능 의미)

- KL 거리 시퀀스의 스무딩 곡선에서, **수직 점선(시뮬레이션 드리프트 위치)** 주변에 뚜렷한 피크·변곡이 나타나, 드리프트 타이밍을 잘 포착한다.[^1_1]
- 일부 구간에서는 DT·GNB가 낮은 정확도를 보이는 반면, KLD의 기울기 곡선은 여전히 변화를 강하게 신호하여, **모델 성능 기반 검출보다 더 민감하게 분포 변화를 포착**하는 사례를 보인다.[^1_1]
- $\alpha$ 민감도 분석 결과:
    - $\alpha < 1.2$: 검출된 critical point 수가 20을 크게 초과 → false positive 다수.
    - $\alpha \approx 1.5\sim 1.75$: 20개 근처로 수렴, 드리프트 위치와 잘 정렬.
    - $\alpha > 2.5$: 검출 수가 줄어들어 누락 증가.[^1_1]


#### (3) 한계 및 제약

- **데이터 종류**:
    - 이 논문에서는 이진 출력, 인공 데이터(시뮬레이션)를 대상으로 했으며, 실제 산업 스트림·다중 클래스·회귀에는 아직 검증되지 않았다.[^1_1]
- **성능 지표**:
    - 탐지 delay, precision/recall, F1 등 정량 지표 대신, 대부분 시각적 비교와 critical point 개수만 보고한다. SOTA 검출기 대비 체계적 벤치마크는 “향후 과제”로 남김.[^1_1]
- **고차원·희소 데이터**:
    - 격자 기반 pmf 추정은 $p$가 커질수록 빈이 급상승하여 **희소성·계산비용** 문제가 생길 수 있음. 논문에서도 J가 크면 빈이 비어 KL 수치 불안정이 커지고, 비용이 증가한다고 언급.[^1_1]
- **파라미터 튜닝**:
    - $\alpha$, J, 스무딩 파라미터 등을 어떻게 자동·이론적으로 선택할지에 대한 가이드라인은 제한적이며, 경험적 선택에 의존.
- **컨셉 종류**:
    - 주로 점진적(incremental) 드리프트를 대상으로 했고, 갑작스러운(sudden)·주기적(recurring)·가상(virtual) 드리프트에 대한 분석은 간단히 언급만 존재.[^1_1]

***

## 3. 모델의 일반화 성능 향상 가능성

이 논문은 “일반화 성능(테스트 성능)”을 직접 최적화하지는 않지만, **drift detection → 적응(adaptation)** 관점에서 일반화 향상 가능성을 시사한다.[^1_6][^1_1]

### 3.1 일반화 측면의 강점

1. **레이블 비의존성 → label‑scarce 환경 일반화**
    - KLD는 **입력–출력 쌍의 경험적 분포**를 직접 비교하므로, 예측 에러 라벨 피드백을 즉시 요구하지 않는다 (단, pmf 추정에 레이블은 쓰지만, “정답만 있으면 되는” 상황에서도 쓸 수 있고, pure input‑drift로 확장 가능).[^1_1]
    - 실제 환경에서 레이블 딜레이가 큰 경우, 에러 기반 검출기보다 **분포 기반 KLD가 더 빠르게 변화 징후를 포착 → 재학습/적응 시점을 앞당겨 일반화 성능 유지에 유리**하다.[^1_7][^1_4]
2. **분포 가정 불필요 → 도메인 일반성**
    - KLD는 임의의 이산 pmf에 적용 가능하며, 입력·출력 분포의 형태(가우시안 등)에 대한 가정을 거의 두지 않는다.[^1_3][^1_1]
    - 이는 비선형·복잡한 데이터(예: 센서, 로그, 복합 피처)에 대해서도 동일한 프레임워크를 사용할 수 있다는 의미이며, **continual learning에서의 도메인 전이·도메인 쉬프트에 넓게 적용 가능**하다.[^1_8]
3. **드리프트 조기 탐지 → forgetting/재학습 시점 최적화**
    - continual learning에서 catastrophic forgetting과의 균형을 위해, **언제** 과거 지식을 잊고 새 지식을 배울지 결정하는 것이 핵심이다.[^1_8][^1_7]
    - KLD가 분포 변화를 조기에 검출하면, 다음과 같은 구체 전략이 가능하다.
        - 메모리 버퍼에서 과거 샘플 비중을 줄이고 최근 샘플 비중을 늘리는 **전략적 샘플 선택 및 망각** (최근 연구: Strategic Selection and Forgetting).[^1_5]
        - task‑aware/unsupervised regularization 강도 조정: drift가 클수록 과거 task parameter anchoring을 완화하여 적응성 증가.
    - 이런 메커니즘은 결과적으로 새로운 분포에 대한 **out‑of‑distribution 일반화 성능** 및 in‑distribution 성능 회복을 향상시킬 수 있다.[^1_9][^1_5]
4. **단순 제어 파라미터 $\alpha$ → 운영 환경에서의 실용적 일반화**
    - $\alpha$는 “얼마나 변화해야 drift로 볼 것인가”를 의미하는 매우 직관적인 파라미터이며, 운영 중에도 경험적으로 조정 가능하다.[^1_1]
    - 예: predictive maintenance처럼 false negative(드리프트 미탐지) 비용이 큰 도메인에서는 $\alpha$를 낮추고, false positive 비용이 큰 도메인에서는 $\alpha$를 높이는 식으로 **도메인 별 일반화 기준**을 맞출 수 있다.[^1_10]

### 3.2 일반화 향상을 위한 한계와 보완 필요 지점

1. **검출 → 적응 전략의 부재**
    - 이 논문은 “언제 drift가 발생했는가”만 제공하고, 모델을 어떻게 업데이트해야 **generalization error**를 최소화할지에 대한 구체 전략(예: 메모리 샘플 선택, 재훈련 스케줄, 파라미터 정규화)은 제시하지 않는다.[^1_1]
    - 최신 continual learning 연구에서는,
        - drift 감지 후 메모리 업데이트와 파인튜닝을 결합하는 전략적 선택·망각 (Strategic Selection and Forgetting).[^1_5]
        - OOD 검출과 CL을 결합해, 분포 외 샘플을 식별하고 별도 대처하는 프레임워크 (OOD for CL).[^1_9]
등과 같이 “검출–적응”을 통합하여 일반화 성능을 최대화하는 방향으로 나아가고 있다.
2. **고차원 표현 공간에서의 KLD 안정성**
    - CNN/Transformer representation처럼 고차원 연속 공간에서는, 단순 격자 binning＋pmf는 차원의 저주로 인해 KL 추정이 불안정해질 수 있고, 이는 drift detection의 신뢰도 저하 → 적응 전략이 잘못된 일반화로 이어질 수 있다.[^1_11][^1_4]
    - 최근 연구들은
        - 딥 표현을 **가우시안(또는 혼합 가우시안)**으로 모델링하고, KL 대신 Fréchet distance(FDD)나 Wasserstein distance 등 더 견고한 거리 척도를 사용해 drift를 감지하는 방식을 제안한다 (DriftLens).[^1_11]
3. **실제 태스크의 성능 곡선과의 연계 필요**
    - 논문은 “분포 변화 곡선 vs 분류기 accuracy 곡선”을 시각적으로만 비교하고,
        - KLD 기반 적응을 적용했을 때 vs 적용하지 않았을 때의 장기 average accuracy / forgetting rate / stability‑plasticity trade‑off 등, CL 핵심 지표는 제시하지 않는다.[^1_1]
    - 유저 관점에서는 “KLD를 써서 실제 모델 일반화 성능을 얼마나 올릴 수 있는지”가 중요하므로, 향후 실증 연구가 필요하다.

***

## 4. 향후 연구에의 영향 및 앞으로 고려할 점

(2020년 이후 관련 최신 연구와의 비교 포함)

### 4.1 이 논문의 위치와 영향

1. **정보이론 기반 드리프트 모니터링의 명시적 정교화**
    - 기존에도 KL은 drift 측정에 간헐적으로 사용되었지만, 이 논문은[^1_12][^1_10]
        - 다차원 스트림,
        - 격자 기반 pmf,
        - 스무딩 및 1차 도함수 임계값 규칙
를 결합한 **완전한 파이프라인**으로 제시했다는 점에서, 이후 KL 기반 drift 연구의 하나의 baseline·참조점 역할을 한다.[^1_10][^1_1]
2. **레이블 의존적 검출기에서 레이블 비의존·분포 기반 검출기로의 전환**
    - 2020년 이후 drift detection 서베이·연구들에서는, 레이블이 제한된 상황에서의 **unsupervised/weakly-supervised drift detection**이 핵심 주제로 부상했다.[^1_13][^1_4]
    - 이 논문은 **분포 기반·모델 독립적** 접근의 대표적인 예로 인용되며,
        - drift를 “에러 곡선”이 아니라 “데이터 분포 곡선”으로 측정하는 방향을 강화한다.[^1_13][^1_10]

### 4.2 2020년 이후 관련 최신 연구 비교 분석

아래는 대표적인 관련 오픈 액세스 연구들이다 (개념적 비교 중심).


| 연구 (연도) | 핵심 아이디어 | KLD 논문과의 관계 |
| :-- | :-- | :-- |
| Hinder et al., “Overview of unsupervised drift detection methods” (2024)[^1_13] | 다양한 distance 기반·cluster 기반·representation 기반 unsupervised drift 방법을 분류·비교. | KLD와 같은 distribution distance 기반 방법을 하나의 큰 축으로 분류하고, 장단점을 체계적으로 정리. KLD는 distance 기반, chunked, global‑threshold 방식의 예시로 위치. |
| Geco et al., “DriftLens” (2024)[^1_11] | 딥 표현(embedding)을 Gaussian으로 모델링하고, Fréchet Distance 등을 이용해 unsupervised로 drift 감지·설명 제공. | KLD가 원시 입력+그리드 pmf를 쓰는 반면, DriftLens는 representation space + closed‑form distance를 사용해 고차원·비정형 데이터에 더 적합. KLD는 아이디어는 유사하지만 구현은 더 단순·고전적. |
| Zhang et al., “Continual Learning with Strategic Selection and Forgetting” (2024)[^1_5] | KS test, KL divergence 등을 사용해 drift를 감지하고, drift 발생 시 샘플 선택·메모리 업데이트·파인튜닝을 전략적으로 수행. | KLD가 “drift detection만” 다루는 반면, 이 논문은 detection→selection→forgetting→fine‑tuning 전체를 설계하여 실제 CL 모델의 성능 개선을 보인다. KLD는 여기서 drift distance의 한 사례로 통합될 수 있음. |
| Yu et al., efficient drift detection under limited labeling (2022–2024)[^1_14] | generative+discriminative 통합, KL 기반 최적화로 label 부족 환경에서의 drift 감지 성능 개선. | KLD에서 보인 “분포 기반 접근”을 한 단계 더 발전시켜, 생성 모델을 활용해 drift와 task 성능을 함께 고려. |
| Ponti et al., DC‑KL (2017, 개념적으로 연관)[^1_10] | 여러 분류기 출력 분포 간 DC-KL로 drift/분쟁 탐지. | KLD는 원시 데이터의 분포를, DC-KL은 모델 출력 분포를 비교. CL 세팅에서는 두 방향을 결합하는 것이 유망. |

요약하면, Basterrech \& Woźniak(2022)의 KLD 검출기는 **정보이론적 거리로 drift를 수치화한다는 아이디어를 명확한 파이프라인으로 제시**했고, 이후 연구들은

- 고차원 딥 표현 공간,
- label scarcity,
- drift detection–adaptation 통합 설계
쪽으로 이를 확장·일반화하고 있다.[^1_11][^1_13][^1_5]


### 4.3 앞으로 연구 시 고려할 점 (구체 제안)

1. **검출–적응 통합 설계**
    - KLD를 실제 continual learning 모델에 연결해,
        - drift 검출 시 **replay 버퍼 구성 변경**,
        - 학습률/정규화 세기 조정,
        - task‑specific classifier 재훈련 스케줄
을 자동화하는 정책을 설계해야 한다.[^1_7][^1_5]
    - 예: “KLD 스코어”를 reward로 하는 reinforcement learning 기반 적응 정책.
2. ** representation space 에서의 KLD/대안 거리 활용**
    - 원시 feature 대신, 사전학습 모델의 embedding 또는 CL 모델의 중간표현에 대해 다변량 Gaussian/Mixture를 맞추고,
        - KL, Jensen–Shannon, Wasserstein, Fréchet distance 등을 비교·선택하는 방식이 필요하다.[^1_11][^1_1]
    - 이렇게 하면 이미지·텍스트·그래프 데이터에 대한 **일반화된 drift detection**과 CL가 가능해진다.
3. **고차원·희소 환경에서의 안정적 pmf 추정**
    - 격자 대신
        - k‑NN density,
        - kernel density estimation,
        - normalizing flows/VAEs 기반 density
로 분포를 추정한 후 KL을 계산하거나,[^1_15]
        - 아예 KL 대신 MMD, energy distance, Maximum Concept Discrepancy(MCD, 2024) 같은 분포 거리로 대체하는 방안을 탐색할 수 있다.[^1_7]
4. **이론적 보장과 성능 지표**
    - KLD 기반 검출의 false alarm rate, detection delay에 대한 이론적 bound를 Statistical Learning Theory 관점에서 분석한 연구들이 등장하고 있어, 이를 KLD 프레임워크에 적용해 **신뢰 가능한 경보 시스템**으로 만드는 것이 중요하다.[^1_16]
    - drift detection 성능을 **downstream CL generalization**(예: average accuracy, backward transfer, forgetting)와 직접 연결하는 평가 지표도 필요하다.[^1_17]
5. **실제 산업/도메인 사례 적용**
    - 논문이 강조한 predictive maintenance, intrusion detection, 추천시스템, online advertising 등에서,
        - KLD‑기반 검출 + 도메인 특화 적응 정책이 실제로 비용 감소·성능 향상에 얼마만큼 기여하는지 사례 연구가 요구된다.[^1_18][^1_10]

***

요청하신 것처럼, 위의 수식은 모두 LaTeX 형식으로 작성했으며, 실제 논문 구현에서는 $\LaTeX$ 수식을 마크다운에서 $`$...$`$로 감싸 렌더링하면 됩니다.
<span style="display:none">[^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35]</span>

<div align="center">⁂</div>

[^1_1]: 2210.04865v1.pdf

[^1_2]: https://arxiv.org/abs/2210.04865

[^1_3]: https://arxiv.org/pdf/2210.04865.pdf

[^1_4]: https://onlinelibrary.wiley.com/doi/pdfdirect/10.1002/widm.1381

[^1_5]: https://arxiv.org/pdf/2412.16264.pdf

[^1_6]: https://ieeexplore.ieee.org/document/9945547/

[^1_7]: https://arxiv.org/pdf/2407.05375.pdf

[^1_8]: https://arxiv.org/pdf/2104.01678.pdf

[^1_9]: https://arxiv.org/html/2512.19725v1

[^1_10]: https://digitalcommons.chapman.edu/cgi/viewcontent.cgi?article=1199\&context=engineering_articles

[^1_11]: https://arxiv.org/html/2406.17813v2

[^1_12]: https://d-nb.info/1338111809/34

[^1_13]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11220237/

[^1_14]: https://arxiv.org/pdf/2312.10841.pdf

[^1_15]: http://arxiv.org/pdf/2302.05380.pdf

[^1_16]: https://www.academia.edu/122267974/Tracking_changes_using_Kullback_Leibler_divergence_for_the_continual_learning

[^1_17]: https://arxiv.org/html/2505.17902v3

[^1_18]: https://www.mdpi.com/1099-4300/20/10/775/pdf

[^1_19]: https://www.semanticscholar.org/paper/Tracking-changes-using-Kullback-Leibler-divergence-Basterrech-Wo'zniak/d742eb6b21766d7a6b8b904929715457df1ab363

[^1_20]: https://www.semanticscholar.org/paper/Tackling-Virtual-and-Real-Concept-Drifts:-An-Model-Oliveira-Minku/0e1c64b1a293d069ea61e41cb8aeb2e95222be9a

[^1_21]: https://www.semanticscholar.org/paper/A-Self-Organizing-Clustering-System-for-Shift-Basterrech-Clemmensen/4dec33625519b4663e86aa8c52db2581b9722399

[^1_22]: https://www.semanticscholar.org/paper/An-Empirical-Insight-Into-Concept-Drift-Detectors-Lapinski-Krawczyk/c483ac3ef4082e19e281fe9f5057843fbe03da50

[^1_23]: https://arxiv.org/html/2412.16264v3

[^1_24]: https://www.semanticscholar.org/paper/458ba5fea4eee8c789ee4d7660f77e754b9aaa0b

[^1_25]: https://arxiv.org/html/2312.02901v2

[^1_26]: https://arxiv.org/html/2410.04183v2

[^1_27]: https://arxiv.org/pdf/2303.09331.pdf

[^1_28]: https://arxiv.org/pdf/2006.12822.pdf

[^1_29]: https://deeplearn.org/arxiv/321318/tracking-changes-using-kullback-leibler-divergence-for-the-continual-learning

[^1_30]: https://deep.ai/publication/tracking-changes-using-kullback-leibler-divergence-for-the-continual-learning

[^1_31]: https://arxiv.org/abs/1504.01044

[^1_32]: https://www.youtube.com/watch?v=lYLg4e6Ixr8

[^1_33]: https://calmmimiforest.tistory.com/120

[^1_34]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10773756/

[^1_35]: https://www.arxiv.org/pdf/2411.02464.pdf

