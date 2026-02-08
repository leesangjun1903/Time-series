# OneNet: Enhancing Time Series Forecasting Models under Concept Drift by Online Ensembling

## 핵심 주장과 주요 기여 (간결 요약)

**핵심 주장**: 시계열 온라인 예측에서 개념 드리프트가 계속 발생할 때, "하나의 모델"만으로는 시간 축 의존성(cross-time)과 변수 간 의존성(cross-variable)을 동시에 견고하게 다루기 어렵다. 따라서 서로 다른 inductive bias를 가진 여러 예측기를 온라인 앙상블하고, 그 가중치를 온라인 convex programming(OCP)+오프라인 RL로 동적으로 조정하면, 개념 드리프트 하에서의 예측 성능과 적응 속도를 크게 향상시킬 수 있다는 것이 핵심 주장이다.[1]

**주요 기여**
- **OneNet 구조 제안**: 시간 의존성 전용 forecaster와 변수 의존성 전용 forecaster 두 개를 유지하면서, OCP 블록이 가변적인 앙상블 가중치를 학습하는 two-stream 온라인 앙상블 프레임워크를 제안.[1]
- **OCP + RL 기반 가중치 업데이트**: 전통적인 Exponentiated Gradient Descent(EGD) 기반 OCP가 갖는 slow switch 현상(분포가 급변할 때 가중치 전환이 느린 문제)을 완화하기 위해, 장기 이력은 EGD로, 단기 이력은 오프라인 RL(RvS 스타일)로 반영해 가중치를 합성하는 OCP 블록을 설계하고, 이론적 regret bound와 실험으로 타당성 입증.[1]
- **성능 향상**: 4개 벤치마크(ETTh2, ETTm1, WTH, ECL)에서 기존 SOTA 온라인 적응 방식 FSNet 대비, 누적 MSE를 평균 53.1%, MAE를 34.5% 감소시킴.[1]
- **설계 선택에 대한 체계적 분석**: 변수 독립 구조, 인스턴스 정규화, seasonal-trend 분해, frequency-domain augmentation 등 현대 시계열 모델 설계 요소가 개념 드리프트 하에서의 견고성/일반화에 어떻게 영향을 주는지 광범위한 실험으로 분석.[1]

## 논문이 해결하고자 하는 문제

### 2.1 문제 설정: 개념 드리프트 하의 온라인 시계열 예측

데이터는 다변량 시계열 $$x_t \in \mathbb{R}^M$$이며, 길이 $$L$$짜리 look-back 윈도우와 길이 $$H$$짜리 forecast horizon에 대해, 전형적인 배치 예측에서는

$$
(x_i)_{i=1}^{L} \mapsto (x_i)_{i=L+1}^{L+H}
$$

를 학습한다.[1]

온라인 예측에서는 각 시점 $$t$$마다

$$
x_{t-L+1:t} \mapsto \hat x_{t+1:t+H}
$$

을 예측한 뒤 실제 $$x_{t+1:t+H}$$가 나중에 공개되고, 이 새 샘플로 모델을 한 번 더 업데이트한다.

온라인 단계의 손실은 채널 평균 MSE:

$$
\mathcal{L} = \frac{1}{M}\sum_{j=1}^{M} \left\| \hat x^{(j)}_{t+1:t+H} - x^{(j)}_{t+1:t+H} \right\|_2^2
$$

[1]

현실 데이터에서는 분포가 시간이 지나면서 바뀌는 **개념 드리프트**가 발생해, 과거 데이터로 학습한 모델이 현재 분포에는 잘 맞지 않게 된다. 전통적 대응은:
- 단순 온라인 SGD/OnlineTCN: 매 스텝 파라미터만 계속 업데이트[1]
- continual learning 류(ER, DER++, MIR, TFCL 등): 버퍼/정규화로 catastrophic forgetting 완화[1]
- FSNet: fast/slow 두 스트림으로 최근 패턴 적응과 과거 패턴 재사용을 결합[2]

그러나 모두 **하나의 백본**에만 의존하며, "어떤 inductive bias(모델 가정)가 현재 구간에 더 맞는지"를 명시적으로 바꾸지는 않는다.[1]

### 2.2 Cross-time vs Cross-variable 가정의 트레이드오프

논문은 다음과 같은 경험적 관찰을 제시한다:[1]

**Cross-time(변수 독립) 모델** (예: PatchTST, Time-TCN):
- 각 채널을 독립된 단변량 시계열로 보고 시간 축 상의 의존성만 모델링.
- 변수 수 $$M$$이 매우 큰 ECL(321 변수) 같은 환경에서는 **매우 견고**하며, 드리프트 하에서도 성능이 안정적.[1]

**Cross-variable 모델** (예: 고전적인 TCN, FSNet 백본):
- 각 시점에서 $$M$$-차원 벡터 전체를 입력으로 받아 변수 간 상관을 직접 모델링.
- 변수 수가 작은 ETTh2/ETTm1(7 변수)에서는 cross-variable 의존성이 중요해 성능이 더 좋지만,
- 변수 수가 큰 환경에서는 overfitting 및 드리프트에 더 취약.[1]

실험적으로, 온라인 적응 과정에서 두 유형의 모델의 MSE를 시점별로 그려 보면, **어느 한 모델이 항상 우월하지 않고, 시점/환경에 따라 더 잘 맞는 모델이 계속 바뀐다**는 사실을 보여준다.[1]
→ 따라서 **"하나의 모델을 잘 업데이트하는 것"만으로는 부족하고**, 시간에 따라 선호되는 inductive bias를 **데이터 기반으로 온라인 선택/조합**할 필요가 있다.

## 제안 방법: OneNet (수식·구조 중심)

### 3.1 OCP 기반 온라인 앙상블 문제 정식화

다변량 입력 $$x \in \mathbb{R}^{L \times M}$$, 타겟 $$y \in \mathbb{R}^{H \times M}$$, $$d$$개의 전문가(모델) $$f_i$$가 있을 때, 각 전문가의 예측을

$$
\tilde y_i = f_i(x) \in \mathbb{R}^{H \times M}, \quad i = 1,\dots, d
$$

라 두고, 이들의 선형 결합

$$
\hat y = \sum_{i=1}^{d} w_i \tilde y_i
$$

의 MSE를 최소화하도록, 가중치 $$w \in \Delta^d$$ (d-차원 simplex)를 온라인으로 갱신하는 문제를 푼다:

$$
\min_{w \in \Delta^d} \; L(w) := \left\| \sum_{i=1}^{d} w_i f_i(x) - y \right\|_2^2
$$

[1]

**전통 OCP에서 자주 쓰이는 Exponentiated Gradient Descent(EGD) 기반 업데이트**:
전문가 $$i$$의 시간 $$t$$에서의 손실 $$\ell_{t,i} = \| f_i(x_t) - y_t\|\_2^2$$ 라 두면,  
초기 $$w_{1,i} = \frac{1}{d}$$ 로 두고,

$$
w_{t+1,i} = \frac{w_{t,i} \exp(-\eta \ell_{t,i})}{Z_t},\quad
Z_t = \sum_{j=1}^{d} w_{t,j} \exp(-\eta \ell_{t,j})
$$

[1]

이 알고리즘은 고전적인 regret bound

$$
\sum_{t=1}^{T} L(w_t) - \inf_{u \in \Delta^d} \sum_{t=1}^{T} L(u) \le \mathcal{O}(\sqrt{T \log d}) 
$$

를 만족한다는 것이 잘 알려져 있다.[1]

하지만 EGD는 **역사 전체의 누적 성능**에 기반해 천천히 가중치를 바꾸므로, **분포가 급변할 때는 전환이 느린 slow switch 현상**을 보인다. 논문은 간단한 toy 예제로, 손실이 50 step마다 뒤집히는 두 전문가 상황에서 $$\eta$$를 어떻게 조절해도 "한쪽 환경에서는 좋지만 다른 쪽 구간에서 전환이 너무 느리다"는 trade-off를 시각적으로 보여준다.[1]

### 3.2 K-step 재초기화와 한계

EGD의 느린 전환을 줄이는 간단한 아이디어로, **K-step마다 $$w$$를 균등 분포로 리셋**(re-initialize)하는 알고리즘을 제안하고, 이 경우 **짧은 구간** $$|I|$$ 에 대한 regret bound가 더 타이트해짐을 이론적으로 보인다(비공식 Proposition 2).[1]

- 길이 $$|I| = T^n$$ ($$0 \le n \le 1$$ ) 인 구간에 대해, $$K = T^{2n/3}$$ 로 잡으면

$$
  R(I) = \mathcal{O}(T^{2n/3})
  $$
  
  으로, 전체 구간 regret $$\mathcal{O}(\sqrt{T})$$ 보다 짧은 구간에서는 더 좋은 bound를 얻는다.[1]
- 하지만:
  - K를 어떻게 잡을지 어려움
  - K-step 밖의 장기 정보는 버려져, 긴 지평에서는 오히려 EGD보다 손해.

→ **장기 이력(안정성)**과 **단기 이력(빠른 적응)**을 동시에 활용할 수 있는 메커니즘이 필요하다.

### 3.3 RL을 결합한 OCP 블록 (OCP + Offline RL)

이를 위해 OneNet은 **두 종류의 가중치**를 도입한다.[1]

1. **장기 가중치 $$w_t$$**:  
   - EGD로 전체 역사에 기반해 업데이트  
   - 느리지만 안정적

2. **단기 bias $$b_t$$**:  
   - 최근 몇 step(실험에서는 바로 직전 step)에서의 전문가 예측과 실제 성능을 입력으로 받는 **offline RL policy**로 산출  
   - 빠르게 전환

구체적으로, $$d=2$$ (두 전문가)인 경우, 시점 $$t$$에서:

- 입력 특징: 각 전문가의 예측과 현재 장기 가중치를 곱한 것과 실제 타깃을 concat

$$
  \text{input}_t = [w_{t,1} \tilde y_{1,t} \otimes \cdots \otimes w_{t,d} \tilde y_{d,t} \otimes y_t]
  $$

- 두 층 MLP $$f_{\text{rl}}$$를 policy로 사용:

$$
  b_t = f_{\text{rl}}(\text{input}_t) \in \mathbb{R}^d
  $$

- 최종 앙상블 가중치는

$$
  \tilde w_{t,i} = \frac{w_{t,i} + b_{t,i}}{\sum_{j=1}^{d} (w_{t,j} + b_{t,j})} 
  $$
  
  [1]
- RL policy의 학습은 행동-가중치의 ground-truth가 없으므로, **예측 손실을 직접 최소화**하는 방식:

$$
  \min_{\theta_{\text{rl}}} \; \left\| \sum_{i=1}^{d} \tilde w_{t,i}(\theta_{\text{rl}})\, f_i(x_t) - y_t \right\|_2^2
  $$

여기서 사용한 RL 프레임워크는 RvS(RL via supervised learning)를 차용하여, value/정책 평가 대신 **조건부 행동 분포를 지도 학습 형식**으로 학습하는 구조를 따른다.[1]

### 3.4 OneNet 모델 구조

#### 3.4.1 두 개의 forecaster: Cross-time vs Cross-variable

- **입력**: 다변량 시계열 $$x \in \mathbb{R}^{L \times M}$$
- **Cross-time forecaster $$f_1$$**:
  - 각 채널을 독립적으로 처리하는 **변수 독립 구조** (Time-FSNet 또는 Time-TCN 스타일)
  - encoder: $$x \mapsto z_1 \in \mathbb{R}^{M \times d_m}$$ (각 채널별 representation)
  - prediction head:  

$$
    z_1 \mapsto \tilde y_1 \in \mathbb{R}^{M \times H}
    $$
    
(채널마다 길이 $$H$$인 예측)[1]

- **Cross-variable forecaster $$f_2$$**:
  - 모든 채널을 함께 처리하는 **변수 상호작용 구조** (FSNet/TCN 스타일)
  - encoder: $$x \mapsto z_2 \in \mathbb{R}^{L \times d_m}$$
  - 마지막 시점 representation $$z_{2,L}\in \mathbb{R}^{d_m}$$을 사용
  - prediction head:

$$
    z_{2,L} \mapsto \tilde y_2 \in \mathbb{R}^{M \times H}
    $$
    
(이 head의 파라미터 수는 대략 $$d_m \times M \times H$$ 로, $$M$$이 클수록 무거워짐)[1]

두 forecaster는 상호보완적인 inductive bias를 가진다:
- $$f_1$$: 변수 독립 → 대규모 $$M$$ 데이터에서 robust, 드리프트에 덜 민감[1]
- $$f_2$$: 변수 간 상호작용 → $$M$$이 작고 변수 간 상관이 중요한 데이터에서 우수

#### 3.4.2 변수별 앙상블 가중치와 OCP 블록

- 각 변수 $$j$$에 대해 cross-time/cross-variable 두 전문가에 대한 가중치 $$w_{t,j} \in \mathbb{R}^2$$를 유지:

$$
  \hat y^{(j)}_t = w_{t,j,1} \tilde y^{(j)}_{1,t} + w_{t,j,2} \tilde y^{(j)}_{2,t}
  $$
  
  [1]

- $$w_{t,j}$$는 EGD로 장기 업데이트, $$b_{t,j}$$는 RL policy로 단기 보정, 최종 $$\tilde w_{t,j} = \text{Normalize}(w_{t,j} + b_{t,j})$$.
- 이 모든 과정을 담당하는 모듈이 **OCP 블록**으로, OneNet 전체 아키텍처에서는 두 forecaster 위에 얹힌 작은 "가중치 제어 네트워크"라고 볼 수 있다.[1]

#### 3.4.3 Decoupled training 전략

단순히 전체 시스템을 joint loss

$$
\mathcal{L}_{\text{joint}} = \mathcal{L}\big( \tilde w_1 \tilde y_1 + \tilde w_2 \tilde y_2,\, y \big)
$$

만으로 학습하면, 어느 한 forecaster가 계속 더 잘하면 다른 쪽의 gradient가 거의 사라져 **한 쪽 모델이 거의 학습되지 않는 collapse**가 발생한다.[1]

이를 막기 위해 loss를 두 부분으로 **decouple**한다:

1. 각 forecaster 자체의 예측력 향상:

$$
   \mathcal{L}_{\text{forecaster}} = \mathcal{L}(\tilde y_1, y) + \mathcal{L}(\tilde y_2, y)
   $$

2. OCP 블록(가중치)의 최적화:

$$
   \mathcal{L}_{\text{OCP}} = \mathcal{L}\big( \tilde w_1 \tilde y_1 + \tilde w_2 \tilde y_2,\, y \big)
   $$
   
   [1]

이렇게 하면:
- 두 forecaster는 **항상 독립적으로 일정 수준 이상 학습**됨 → 어떤 환경으로 드리프트하더라도 둘 중 하나는 좋은 베이스라인이 됨.
- OCP 블록은 이미 reasonable한 두 예측을 입력으로 받아, 적절한 조합 규칙만 학습하면 됨.

## 성능 향상 및 한계 (특히 일반화 관점)

### 4.1 정량적 성능 향상

논문은 4개의 대표적 multivariate benchmark(ETTh2, ETTm1, WTH, ECL)와 여러 forecast horizon $$H \in \{1,24,48\}$$에서 누적 MSE/MAE를 비교하였다.[1]

**기본 비교 대상**
- OnlineTCN, ER/MIR/DER++/TFCL (online continual learning 기법들)[1]
- FSNet (ICLR 2023, Fast & Slow learning Networks)[2]
- PatchTST (Transformer 기반, 변수 독립 patch 모델)[3]
- Time-TCN (변수 독립 TCN), DLinear 등[1]

**주요 결과 (MSE 기준)**:
- OneNet 평균 누적 MSE는 약 **0.747**, FSNet은 **1.594**로 **53.1% 감소**.[1]
- MAE도 0.293 vs 0.448로 **34.5% 감소**.
- 특히 변수 수가 큰 **ECL(321 변수)**에서:
  - FSNet: $$H=48$$ 시 MSE 7.034
  - OneNet: 동일 조건 MSE 2.201 (약 69% 감소)[1]
- 변수 수가 작은 ETTh2/ETTm1에서도, 단일 cross-time, cross-variable 모델보다 일관되게 더 낮은 MSE를 달성.

**Delayed feedback setting** (실제 환경처럼 타깃이 $$H$$ step 뒤에야 주어져, $$H$$ step마다 한 번 업데이트):
- 모든 기법 성능이 악화되지만, FSNet은 $$H=48$$에서 일부 데이터셋에서 TCN보다 못한 성능까지 보인다.
- OneNet은 이 설정에서도 가장 낮은 평균 MSE/MAE를 유지하며, online 피드백이 늦어져도 **상대적인 우위**를 잃지 않는다.[1]

### 4.2 Ensembling 방법 비교와 OCP 블록의 효과

단순히 "두 모델을 섞기만" 해도 좋아질 수 있으므로, 다양한 앙상블 방식과 비교한다:[1]

- 평균( $$\frac{1}{2}(\tilde y_1 + \tilde y_2)$$ )
- gating (출력 기반 softmax weight)
- mixture-of-experts (입력 $$x$$ 기반 softmax)
- linear regression로 최적 가중치 추정
- EGD만 사용하는 OCP
- RL-W (RL로만 weight를 학습, 장기 이력 미사용)

**결과**:
- 단순 평균이나 gating도 FSNet 단일 모델보다는 개선을 가져오지만,
- 입력 기반 MoE는 오히려 불안정하고 평균보다 못한 경우도 발생.
- EGD와 RL-W는 각각 나름 성능 향상을 보여, **장기 이력(역사)과 단기 성능 모두 유의미**함을 시사.
- **OCP(EGD)+RL을 결합한 OneNet의 OCP 블록이 가장 좋은 성능**을 보여, 두 정보를 함께 사용하는 설계가 타당함을 실험적으로 입증.[1]

### 4.3 일반화·견고성 관점에서의 논문 분석

논문이 보여주는 일반화 관련 포인트는 크게 세 가지다.[1]

1. **모델 bias의 다양화가 시간적 일반화에 기여**
   - cross-time vs cross-variable 한쪽만 쓰면, 특정 데이터셋·시점에서는 우수하지만 **개념 드리프트 이후**에는 급격히 성능이 나빠질 수 있다.
   - 서로 다른 bias를 가진 두 forecaster를 모두 유지하고, 환경에 따라 가중치를 조정하면,
     - 최악의 경우에도 "둘 중 하나"의 성능 이상을 유지할 수 있는 **hedging 효과**를 제공.
     - 이는 **시간에 따른 분포 변화에 대한 일반화**로 해석 가능 (dataset 간 generalization이 아니라, same dataset 내 time-wise generalization).

2. **OCP+RL로 "빠르게 전환할 수 있는" 일반화**
   - EGD의 slow switch 문제는 급격한 환경 전환 시 장기 이력에 과도하게 끌려가 **적시에 bias를 바꾸지 못하는 일반화 실패**로 볼 수 있다.
   - OneNet의 RL bias $$b_t$$는 최근 손실 패턴만을 반영하므로,  
     - 개념이 급변하면 단기적으로 빠르게 **"현재 더 잘 맞는 모델" 쪽으로 가중치를 당겨오는 역할**을 한다.
     - 이때도 장기 이력 $$w_t$$가 완전히 사라지지 않아, 노이즈/일시적 이상치에 대한 과도한 적응(오버피팅)을 방지.
   - 이는 online 환경에서 **"fast but not too myopic"한 generalization** trade-off를 구현한 것으로 볼 수 있다.[1]

3. **설계 요소들의 robust/generalizable choice에 대한 통찰**
   - 인스턴스 정규화(instance norm), seasonal-trend 분해, frequency-domain augmentation 등 현대 LTSF 설계들이 항상 "드리프트 하에서 좋은 것"은 아니다.
   - 예를 들어, PatchTST에서 instance norm은 드리프트가 있을 때 "배치 학습 + 적응 불가능" 상황에서는 분포 차이를 줄여 robustness를 높여주지만,  
     온라인 적응이 가능한 상황에서는 **적응 속도를 방해**해, 일부 데이터셋/긴 horizon에서 오히려 성능을 악화시키는 결과를 보였다.[1]
   - 이는 향후 "**일반화 + 빠른 적응**을 동시 달성할 수 있는 새로운 정규화/전처리 설계" 필요성을 제기한다.

### 4.4 한계

논문이 명시하거나 암묵적으로 드러나는 한계는 다음과 같다.[1]

- **파라미터·연산량 증가**:
  - 두 개의 forecaster + OCP 블록 → FSNet 단일 모델보다 파라미터와 추론 시간이 증가.
  - 이를 완화하기 위해 저자들은 파라미터 효율적인 변형인 **OneNet-**(PatchTST + Time-FSNet, projection head 경량화)을 제안하지만, 여전히 single backbone 대비 복잡하다.
- **두 bias에 대한 의존**:
  - "cross-time vs cross-variable" 두 종류의 bias가 유효하지 않은 특수 도메인(예: 그래프 구조가 중요한 시계열)에선 그대로 적용하기 어렵고,  
    이 경우 어떤 전문가 셋을 쓸지 도메인 설계가 필요하다.
- **레이블 가용성 가정**:
  - 온라인 설정에서 매 스텝(or $$H$$ 스텝마다) 타깃 $$y$$가 주어진다고 가정.  
    실제 산업 환경에서는 라벨 지연이 더 크거나, 일부 시점은 라벨이 없는 semi-supervised 상황이 많다.
- **명시적 드리프트 검출 없음**:
  - D³A 같은 "드리프트 검출 + 적응(Detect-then-Adapt)" 계열과 달리, OneNet은 **implicit하게** 가중치를 조정할 뿐, 명시적 드리프트 탐지·해석 기능은 없다.[4]
- **벤치마크 한정**:
  - ETTh2/ETTm1/WTH/ECL 네 가지 전형적인 benchmark에서 좋은 결과를 보이지만,  
    고차원 금융 시계열, graph-structured time series, 초장기 horizon 등으로의 일반화는 추가 검증 필요.

## 2020년 이후 관련 최신 연구와의 비교 분석

여기서는 OneNet과 유사한 문제(온라인 시계열, 개념 드리프트, 모델/가중치 적응)를 다루는 2020년 이후 대표 연구들과 비교한다.

### 5.1 FSNet (Learning Fast and Slow for Online Time Series Forecasting)[2]

**핵심 아이디어**:
- Complementary Learning Systems(CLS) 이론에서 영감.  
- 하나의 백본 네트워크에:
  - 레이어별 adapter를 붙여 **fast learning**을 수행하고,
  - associative memory를 통해 과거 패턴을 **retrieve**하여 **slow backbone**을 보완.

**문제 초점**:
- "하나의 모델"이 새 패턴에 빠르게 적응하면서 과거 패턴을 잊지 않도록 하는 continual learning 설계.

**OneNet과 차이점**:
- FSNet은 **단일 백본의 파라미터 업데이트**에 초점,
- OneNet은 **서로 다른 inductive bias를 가진 두 백본의 가중치(ensemble weight)를 온라인으로 선택/조합**.
- 논문 실험에서는 PatchTST(변수 독립 Transformer)를 단순 온라인 fine-tuning 했을 때조차 FSNet보다 좋은 성능을 보인다는 점을 지적하고,  
  그 위에서 OneNet의 앙상블이 FSNet을 크게 능가함을 보여줌.[1]

**요약**:  
FSNet은 "모델 파라미터 수준의 fast/slow 적응", OneNet은 "모델 bias 수준의 fast/slow 선택"이라고 볼 수 있으며, OneNet은 FSNet류 방법을 **전문가 중 하나로 포함하는 상위 메타-프레임워크**로 해석 가능하다.

### 5.2 PatchTST (A Time Series is Worth 64 Words)[3]

**핵심**:
- channel-independent patch Transformer로, 시계열을 patch 단위 token으로 변환하고, 각 채널을 독립적으로 모델링하여 LTSF에서 매우 강력한 성능을 달성.

**개념 드리프트 관점**:
- 원 논문은 주로 배치 학습/장기 예측에 초점을 맞추고, 온라인 드리프트 처리는 별도 다루지 않음.
- OneNet 논문은 PatchTST를 온라인 fine-tuning 했을 때:
  - FSNet보다 좋은 성능을 보이는 경우가 많고,
  - 특히 **변수 독립 구조가 드리프트에 대한 robustness에 크게 기여**함을 실험으로 분석한다.[1]

**OneNet과 관계**:
- OneNet은 이 통찰을 일반화해, "변수 독립 cross-time 전문가"를 하나의 스트림으로 채택하고,  
  "변수 상호작용 cross-variable 전문가"와 OCP 블록으로 앙상블한다.
- 따라서 OneNet은 PatchTST류의 강력한 백본과 매우 상호보완적인 역할을 할 수 있으며, 실제로 PatchTST를 포함한 Three-expert 구성(OneNet-TCN+Patch)도 제시된다.[1]

### 5.3 D³A: Addressing Concept Shift in Online TS Forecasting (Detect-then-Adapt)[4]

**핵심 아이디어**:
- 많은 온라인 적응 방법들이 "계속 조금씩 업데이트"만 하다 보니, 누적 드리프트 하에서 점진적인 성능 저하를 보인다는 점을 지적.
- **개념 드리프트 탐지 후, 적절한 시점에 더 큰 적응**(예: 재학습, 재파라미터화)을 수행하는 Detect-then-Adapt 프레임워크 제안.

**OneNet과 비교**:
- D³A는 명시적 drift detection + adaptation pipeline, OneNet은 **암묵적 weight reallocation**에 기반.
- 두 접근은 상호보완적:  
  - OneNet의 OCP 블록을 D³A의 "drift 감지 후 재초기화 또는 전문가 교체" 모듈과 결합할 수 있음.
  - 예: drift가 감지되면 특정 전문가를 재학습하거나 교체하고, OCP 블록은 언제나 최신 전문가들을 online ensemble.

### 5.4 LEAF: Two-Stage Meta-Learning for Concept Drift (IJCAI 2025)[5]

**문제 정의**:
- 개념 드리프트를 macro-drift(큰 구조 변화)와 micro-drift(미세한 변화)로 나누어,  
  online forecasting에서 둘을 동시에 다루는 meta-learning 프레임워크 제안.

**핵심**:
- 다양한 prediction 모델에 대해 meta-parameters를 학습해, 새로운 drift 환경에 빠르게 적응할 수 있는 초기화/업데이트 규칙을 학습.

**OneNet과 비교**:
- LEAF는 **update rule/meta-parameter**를 학습하는 meta-learning 접근,
- OneNet은 **서로 다른 구조(전문가)들의 비율**을 조정하는 weight-learning 접근.
- 둘은 orthogonal한 축:  
  - 예: 각 전문가 자체는 LEAF로 meta-trained 되고, OneNet OCP 블록이 이들을 online ensemble.

### 5.5 모델 선택·앙상블 계열: SimpleTS, Online Explainable Model Selection 등

- **SimpleTS**:[6]
  - 다양한 시계열 모델을 미리 클러스터링하고, 입력 시계열을 유형별로 분류한 뒤, 그 유형에 가장 잘 맞는 모델을 선택하는 효율적인 모델 선택 프레임워크.
- **Online Explainable Model Selection**:[7]
  - 각 모델이 잘 작동하는 Region of Competence(RoC)를 온라인으로 클러스터링해, 시점별로 적절한 모델(또는 소수 모델 앙상블)을 선택하고, 선택 근거를 설명 가능하게 제공.

**OneNet과의 차이**:
- 위 방법들은 주로 **dataset-level 또는 segment-level** 모델 선택에 초점,  
- OneNet은 **두 개의 고정 전문가에 대해, 매우 fine-grained(시점·변수별) weight 조정**을 수행.
- 또한 OneNet은 모델 family를 두 개로 고정해 두고, RL+OCP로 weights를 학습하는 반면, SimpleTS류는 많은 후보 모델 군에서 "하나(or 소수)를 선택"한다는 점에서 다르다.

### 5.6 Foundation TS 모델 + Online Adaption (AdapTS 등)[8][9]

- 최근에는 LLM 스타일의 **Time-series Foundation Model**들이 등장하면서,  
  - 대규모 코퍼스에서 사전학습된 모델을 zero-shot/ few-shot으로 사용하고,  
  - domain별 online adaptor(예: AdapTS, ELF-Forecaster 등)를 붙여 실시간 데이터에 적응시키는 방향이 활발하다.
- 이들 또한 "**기본 모델 + online adapter/weighting**"이라는 점에서 OneNet의 사상과 가깝다:
  - Foundation forecaster와 lightweight online forecaster를 ensemble하는 weighting 네트워크를 학습하여, drift를 감지하면 더 잘 맞는 쪽으로 weight를 이동시킨다.[9]
- **OneNet과의 연결 가능성**:
  - Foundation forecaster를 cross-time 전문가, domain-specific 모델을 cross-variable 전문가로 보고, OCP+RL 블록으로 앙상블하면, **대규모 사전학습 + 온라인 드리프트 적응**을 동시에 달성할 수 있다.

## 향후 연구에의 영향과 고려할 점

OneNet이 앞으로의 연구에 주는 시사점과, 후속 연구에서 고려할 포인트를 정리하면 다음과 같다.

### 6.1 연구 방향에 대한 영향

1. **"단일 모델 업데이트"에서 "모델 bias 앙상블/선택"으로의 패러다임 이동**
   - FSNet, ER 계열은 하나의 모델이 모든 드리프트를 감당하도록 만든 반면,  
     OneNet은 모델 간 bias 차이를 적극 활용해 **온라인 앙상블**을 수행한다.
   - 향후 연구에서는:
     - 다양한 구조(예: Transformer, GNN, CNN, Linear, Graph-TS 등)[10]
     - 다양한 데이터 전처리/정규화 조합
     을 **전문가 풀(pool)**로 두고, OneNet식 OCP+RL 블록으로 online model selection/ensemble을 하는 방향이 유력하다.

2. **Online weight control에 RL·OCP·meta-learning 접목**
   - OneNet은 OCP(EGD)와 offline RL(RvS)을 결합한 첫 사례 중 하나로,  
     향후에는:
     - distribution-aware exploration,
     - delayed reward(지연 피드백),
     - risk-sensitive objective
     를 고려한 RL 기반 weight updater들이 더 발전할 여지가 있다.
   - 또한 LEAF 같은 meta-learning 접근과 결합하면,[5]
     "**가중치 업데이트 규칙 자체**를 meta-learn"하는 고차원적인 online ensemble 학습이 가능하다.

3. **Normalization/전처리의 역할 재고찰**
   - 논문이 보인 것처럼, instance norm, seasonal-trend decomposition, frequency augmentation 등은 **드리프트/온라인 적응** 맥락에서 다시 평가되어야 한다.[1]
   - 즉, "배치 LTSF 성능을 올려준 테크닉"이 온라인 환경에서도 항상 좋다는 보장은 없으며,
     - drift-robust normalization,
     - 빠른 re-centering/re-scaling이 가능한 온라인 통계 추정
     등이 새로운 연구 주제로 부상할 수 있다.

### 6.2 향후 연구 시 고려할 점 (실무·연구 관점)

1. **전문가 풀 설계**
   - 어떤 전문가들을 앙상블에 포함할지에 따라 성능·복잡도가 크게 달라진다.
   - 실무에서는:
     - 하나는 파라미터가 적고 빠른 cross-time "안전한 기본 모델",
     - 다른 하나는 복잡하지만 고성능 cross-variable/graph-based 모델,
     - 필요시 Foundation forecaster까지 포함
     하는 식으로 **성능–비용–robustness trade-off**를 설계해야 한다.

2. **파라미터 효율성과 latency**
   - 온라인 서비스에서는 추론 지연과 메모리가 매우 중요하므로,  
     OneNet- 같이 projection head를 공유/경량화 하거나,  
     adapter/prompt 기반 parameter-efficient tuning 기법을 활용해
     - "두 개(혹은 그 이상) 모델을 모두 full update"하는 것은 피하고,
     - encoder는 동결, 작은 head/adapter만 업데이트하는 식의 구조를 고민해야 한다.

3. **레이블 지연·결측에 대한 대응**
   - OneNet은 true $$y$$가 곧바로(또는 예측 horizon 이후 즉시) 주어진다고 가정한다.
   - 실제로는:
     - 더 긴 지연,
     - 일부 시점의 라벨 미존재,
     - semi-supervised 또는 unsupervised drift detection[11]
     가 공통적이므로,
     - "라벨이 없는 구간에서도 drift를 감지하고, weight를 선제적으로 조정"하는 방식,
     - self-supervised loss(예: reconstruction, consistency)를 weight updater에 활용하는 방식
     이 중요해질 것이다.

4. **설명 가능성 및 드리프트 분석**
   - Online Explainable Model Selection처럼, 왜 특정 시점에 어떤 전문가가 선택됐는지 해석 가능한 메커니즘이 점점 요구된다.[7]
   - OCP+RL weight의 시간적 궤적을 분석하면,
     - 어느 시점에 어떤 종류의 drift가 발생했는지,
     - cross-time vs cross-variable 의존성이 언제 중요해졌는지
     를 간접적으로 파악할 수 있다.
   - 향후 연구에서는 weight dynamics를 기반으로 한 **drift explanation**과 **root-cause analysis**가 중요한 주제가 될 것이다.

5. **통합 벤치마크·평가 프레임워크**
   - 최근 TSPP 같은 통합 벤치마크 도구들이 제안되고 있으나,[12]
     여전히 "온라인·드리프트·레이블 지연" 등을 포괄하는 표준 벤치마크는 부족하다.
   - OneNet과 LEAF, D³A, FSNet, AdapTS류 방법들을 **동일한 온라인 환경 세팅**에서 fair하게 비교할 수 있는 프레임워크 구축이 앞으로의 커뮤니티 과제다.

**정리하면**, 이 논문은 "개념 드리프트 하의 온라인 시계열 예측"에서 **하나의 모델을 잘 업데이트하는 것**에서 한 단계 나아가, **서로 다른 inductive bias를 가진 여러 모델을 온라인 앙상블하면서, 그 가중치를 OCP+RL로 데이터 기반으로 조정**하는 프레임워크를 제시했고, 이를 통해 기존 SOTA 대비 큰 폭의 성능 향상과 드리프트에 대한 일반화·견고성을 보여주었다.[1]

향후 연구에서는 이 아이디어를 더 큰 전문가 풀, foundation TS 모델, meta-learning, drift detection, parameter-efficient adaptation과 결합하는 방향으로 확장하는 것이 유망해 보인다.[13][8][9]

<span style="display:none">[^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45]</span>

<div align="center">⁂</div>

[^1_1]: https://arxiv.org/abs/2309.12659

[^1_2]: https://arxiv.org/abs/2202.11672

[^1_3]: https://openreview.net/forum?id=q-PbpHD3EOk

[^1_4]: https://arxiv.org/abs/2211.14730

[^1_5]: http://arxiv.org/pdf/2211.14730v2.pdf

[^1_6]: https://arxiv.org/html/2403.14949v1

[^1_7]: https://www.ijcai.org/proceedings/2025/0542.pdf

[^1_8]: https://dl.acm.org/doi/10.14778/3611540.3611561

[^1_9]: https://ieeexplore.ieee.org/document/10302609/

[^1_10]: https://arxiv.org/pdf/2310.10688.pdf

[^1_11]: https://arxiv.org/pdf/2502.12920.pdf

[^1_12]: https://ieeexplore.ieee.org/document/10636792/

[^1_13]: https://arxiv.org/html/2406.17813v2

[^1_14]: https://www.sciencedirect.com/science/article/abs/pii/S0016003223006361

[^1_15]: https://arxiv.org/pdf/2312.17100.pdf

[^1_16]: 2309.12659v1.pdf

[^1_17]: https://dl.acm.org/doi/10.14778/3583140.3583155

[^1_18]: https://arxiv.org/abs/2302.10347

[^1_19]: https://arxiv.org/abs/2304.03877

[^1_20]: https://link.springer.com/10.1007/978-3-031-43424-2_10

[^1_21]: https://dl.acm.org/doi/10.1145/3580305.3599533

[^1_22]: https://link.springer.com/10.1007/s11042-023-17637-3

[^1_23]: https://arxiv.org/pdf/2202.11672.pdf

[^1_24]: http://arxiv.org/pdf/2310.19322.pdf

[^1_25]: http://arxiv.org/pdf/2405.14252.pdf

[^1_26]: http://arxiv.org/pdf/2405.13522.pdf

[^1_27]: http://arxiv.org/pdf/2405.10877.pdf

[^1_28]: http://arxiv.org/pdf/2412.13769.pdf

[^1_29]: https://arxiv.org/html/2507.13043v1

[^1_30]: https://arxiv.org/html/2601.12931v1

[^1_31]: https://arxiv.org/html/2501.08620v1

[^1_32]: https://arxiv.org/html/2412.08435v3

[^1_33]: https://arxiv.org/pdf/2509.03810.pdf

[^1_34]: https://arxiv.org/html/2211.05244v3

[^1_35]: https://arxiv.org/html/2501.08620v3

[^1_36]: https://arxiv.org/html/2510.15404v1

[^1_37]: https://arxiv.org/pdf/2211.14730.pdf

[^1_38]: https://arxiv.org/pdf/2410.09190.pdf

[^1_39]: https://github.com/yuqinie98/PatchTST

[^1_40]: https://openreview.net/forum?id=Jbdc0vTOcol

[^1_41]: https://iclr.cc/virtual/2023/poster/11198

[^1_42]: https://huggingface.co/docs/transformers/model_doc/patchtst

[^1_43]: https://www.sciencedirect.com/science/article/pii/S0957417422019522

[^1_44]: https://openreview.net/references/pdf?id=2mNAE8QDNd

[^1_45]: https://dl.acm.org/doi/10.1145/3691338
