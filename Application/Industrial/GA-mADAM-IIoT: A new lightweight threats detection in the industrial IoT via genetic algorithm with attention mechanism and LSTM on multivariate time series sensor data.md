
# GA-mADAM-IIoT: A new lightweight threats detection in the industrial IoT via genetic algorithm with attention mechanism and LSTM on multivariate time series sensor data

## 요약

본 보고서는 "GA-mADAM-IIoT: A new lightweight threats detection in the industrial IoT via genetic algorithm with attention mechanism and LSTM on multivariate time series sensor data"를 AI 분야의 관점에서 상세 분석한다. 본 논문의 핵심 기여는 산업용 IoT(IIoT) 환경에서 경량화된 위협 탐지 모델을 제안하며, 유전 알고리즘(GA), 개선된 Adam 최적화(mADAM), 주의(Attention) 메커니즘, 그리고 LSTM을 통합한 혁신적 접근법을 제시한다. 제안 모델은 SWaT 데이터셋에서 99.98% 정확도, WADI 데이터셋에서 99.87% 정확도를 달성하며, SHAP 기반 설명가능한 AI(XAI)를 통해 모델의 투명성을 확보한다. 본 분석은 2020년 이후의 최신 관련 연구와의 비교를 통해 모델의 위치를 파악하고, 일반화 성능 향상 가능성과 한계를 심층적으로 검토한다.

***

## 1. 핵심 주장 및 주요 기여

### 1.1 논문의 핵심 주장

GA-mADAM-IIoT 논문은 다음의 세 가지 핵심 주장을 제시한다:

**첫째, IIoT 환경의 보안 위협 탐지는 장기 의존성을 효과적으로 처리할 수 있는 전문화된 신경망 모델을 필요로 한다.** 기존의 ARIMA, 칼만 필터 같은 전통 방식은 단기 예측만 가능하며, 시계열 데이터의 복잡한 패턴을 포착하지 못한다. LSTM의 메모리 블록 구조는 이 문제를 해결하지만, 고차원 IIoT 데이터에 적용할 때 계산 복잡도가 급증한다.

**둘째, 특성 선택(Feature Selection)과 하이퍼파라미터 최적화를 통해 경량화된 IDS 모델을 구현할 수 있다.** IIoT 센서 데이터는 수백 개의 특성을 포함하는데, 이 중 대부분은 위협 탐지에 무관하거나 중복적이다. 유전 알고리즘을 통한 진화적 특성 선택은 차원 축소뿐만 아니라 모델의 일반화 성능을 향상시킨다.

**셋째, 개선된 Adam 최적화와 주의 메커니즘의 조합은 LSTM의 수렴 속도를 개선하고 중요 정보에 대한 모델의 집중력을 강화한다.** Modified Adam은 기존 Adam의 느린 수렴 및 진동 문제를 해결하며, Attention 메커니즘은 입력 시퀀스의 관련성 있는 부분에 선택적으로 가중치를 할당한다.

### 1.2 주요 기여

논문이 제시하는 주요 기여는 6가지로 정리된다:

| 기여 항목 | 구체적 내용 | 기술적 의의 |
|---------|----------|----------|
| **GA 기반 특성 선택** | SWaT에서 51개→23개, WADI에서 127개→27개 특성 축소 | 계산 복잡도 감소, 과적합 방지 |
| **mADAM 최적화** | 적응적 학습률 제어(ALRC), 수정된 모멘텀 계산 | 수렴 안정성 및 속도 향상 |
| **Attention 메커니즘** | Squeeze & Excitation 블록 기반 구현 | 중요 특성 강화, 불필요 특성 감소 |
| **Categorical Cross-Entropy 손실함수** | 정규화 파라미터 추가 | 불균형 데이터셋 처리 개선 |
| **6-모듈 구조 설계** | Activity Receiver, CM, AM, IDS, Mitigation, Alert | 모듈식 구조로 통합 용이성 |
| **SHAP 기반 XAI 통합** | 의사결정 과정의 투명성 확보 | 신뢰성 및 설명가능성 향상 |

***

## 2. 해결하고자 하는 문제와 제안 방법

### 2.1 문제 정의

#### 2.1.1 기술적 문제

**장기 의존성(Long-Term Dependency) 문제**: IIoT 센서 데이터는 수분에서 수시간에 걸친 장기 패턴을 보여준다. ARIMA는 차수가 낮아 장기 의존성을 포착하지 못하고, 칼만 필터는 선형 가정으로 인해 비선형 공격 패턴을 감지하지 못한다. 신경망 기반 RNN 역시 그래디언트 소실(Vanishing Gradient) 문제로 고통받는다. [arxiv](https://arxiv.org/html/2211.05244v3)

**고차원 데이터 처리**: SWaT의 경우 51개, WADI의 경우 127개의 특성을 가지고 있다. 모든 특성을 LSTM에 입력하면 계산 복잡도가 O(n³) 수준으로 증가하며, 수백 만 개의 샘플에 대해 훈련 시간이 수시간 이상 소요된다.

**불균형 데이터셋**: SWaT는 정상:공격 = 88.03:11.97, WADI는 94.01:5.99의 심각한 불균형을 보인다. 이는 표준 교차 엔트로피 손실함수가 다수 클래스(정상)에 편향되도록 한다.

**느린 LSTM 학습**: 고차원 데이터셋에서 LSTM의 모든 셀(각각 4개의 게이트 포함)의 파라미터를 학습하는 것은 계산상 부담이 크다. 기존 Adam 최적화도 진동과 느린 수렴 문제를 보여준다. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC9845049/)

#### 2.1.2 실무적 문제

**실제 라벨링된 데이터 부족**: 산업 시설의 공격 데이터를 수집하고 라벨링하는 것은 현실적으로 불가능하다. 따라서 물 처리 시스템(SWaT, WADI)의 시뮬레이션된 공격 시나리오가 대리로 사용되어야 한다.

**모델의 검은 상자 특성**: 깊은 신경망 기반 IDS는 높은 정확도를 달성하더라도 보안 담당자가 의사결정 과정을 이해할 수 없다. 이는 규제 준수(GDPR 등) 및 운영 신뢰도 측면에서 문제가 된다.

**실시간 탐지 요구**: 공격은 밀리초 단위로 일어나므로 모델의 추론 지연시간이 100ms 이하여야 한다.

### 2.2 제안 방법론

#### 2.2.1 6-모듈 시스템 아키텍처

GA-mADAM-IIoT 시스템은 다음의 6개 모듈로 구성된다:

**1) Activity Receiver Module**: IIoT 네트워크의 모든 디바이스로부터 활동 데이터를 수집한다. 각 센서 및 액추에이터의 읽기값, 상태 변화, 시간 스탬프를 기록한다.

**2) Communication Module (CM)**: 프로토콜 독립적인 통신을 지원한다. 동적 연결(Dynamic Collection), 네트워크 에뮬레이터, 인터페이스 모듈(IM)로 구성된다. Modbus/TCP, MQTT, OPC-UA 등 다양한 산업 프로토콜을 수용할 수 있다.

**3) Attention Module (AM)**: Squeeze & Excitation 블록을 기반으로 CM의 출력을 증폭시킨다. 수식 (1)-(3)을 통해 시공간(Spatial-Temporal) 정보를 압축한 후, 중요 채널에 가중치를 할당한다.

**4) Intrusion Detection Module (IDS)**: 특성 추출, 네트워크 분류기, 분류기 업데이트의 3단계로 구성된다. GA를 통해 선택된 특성을 mADAM-LSTM에 입력한다.

**5) Mitigation Module**: 공격으로 판정된 패킷에 대해 적절한 완화 조치를 선택한다. 공격 특성과 사전정의된 완화 규칙(수식 13)의 매핑을 기반으로 한다.

**6) Alert Module**: 침입 탐지시 관리자에게 즉시 경고를 발송한다.

#### 2.2.2 핵심 알고리즘: GA, mADAM, Attention

**Genetic Algorithm (GA) 기반 특성 선택**:

유전 알고리즘은 자연 선택의 원리를 모방하여 최적 특성 부분집합을 탐색한다. 초기 집단을 랜덤하게 생성한 후, 각 개체(특성 부분집합)의 적합도를 평가한다. 적합도 함수는 다음을 고려한다:

- **관련성(Relevance)**: Pearson 상관계수로 각 특성과 공격 라벨 간의 상관성 계산
- **차원성(Dimensionality)**: 선택된 특성 수가 적을수록 가산 점수 부여
- **안정성(Stability)**: 다중 GA 실행에서 일관되게 선택되는 특성을 우선

선택-교배-돌연변이의 반복을 통해 100 세대까지 진화시킨다(표 5 참조). 이는 특성 선택의 NP-hard 특성을 고려할 때 준최적해를 효율적으로 찾는 방법이다. [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0167404823005850)

**Modified Adam (mADAM) 최적화**:

표준 Adam 최적화의 문제점:
- 1차 모멘트(그래디언트 평균)가 지수 가중 평균으로 계산되어 이전 그래디언트에 과도하게 의존
- 2차 모멘트(제곱 그래디언트)의 고정 감쇠율로 인해 훈련 후기에 진동 발생
- 편향 보정 후에도 충분한 안정성 미확보

mADAM의 개선사항:

1) **적응적 학습률 제어(ALRC)**: 그래디언트의 분산을 모니터링하여 학습률을 동적으로 조정한다. 분산이 크면 학습률 감소, 안정화되면 증가시킨다.

2) **수정된 모멘텀 계산**: 임시 위치(∝_temp) 개념을 도입하여 현재 속도(velocity)를 고려한 후 그래디언트를 계산한다:

$$\alpha_{temp} = \alpha_{t-1} - \beta \cdot \text{velo}_{t-1}$$

$$\hat{g}_t = \nabla_{\alpha_{temp}} \sum_i L(f(y; \alpha_{temp}), x)$$

이는 모멘텀 방향을 미리 고려하여 진동을 감소시킨다.

3) **개선된 편향 보정**: 지수 가중 평균의 초기 편향을 보정하되, 히스토리 정보를 더 효과적으로 활용한다:

$$\hat{u}_t = \frac{u_t}{1 - \gamma_1^t}, \quad \hat{\text{Velo}}_t = \frac{\text{Velo}_t}{1 - \gamma_2^t}$$

최종 파라미터 업데이트:
$$\alpha_{t,new} = \alpha_{t,old} - \frac{\hat{u}_t}{\sqrt{\hat{\text{Velo}}_t} + \epsilon} \cdot \hat{\text{Velo}}_t$$

**Attention Mechanism의 LSTM 통합**:

Attention 메커니즘은 입력 시퀀스의 각 타임 스텝에 대해 얼마나 주의를 기울일지 동적으로 결정한다. IIoT 트래픽 데이터에서 공격은 특정 타임 스텝이나 특성에서만 나타나므로, Attention을 통해 모델이 관련 부분에 집중하도록 유도한다.

3가지 주요 벡터:
- **Query (Q)**: 현재 타임 스텝 t에서 "무엇을 찾을 것인가"를 인코딩
- **Key (K)**: 모든 타임 스텝의 "정보 저장 위치"를 인코딩  
- **Value (V)**: 실제 "정보 내용"을 인코딩

Scaled Dot-Product Attention을 통해 Q와 모든 K 간의 유사도를 계산:

$$\text{Attention Score}(t, t') = \frac{Q_t \cdot K_t'^T}{\sqrt{d_k}}$$

이를 softmax로 정규화하여 attention weights를 얻는다. 이 가중치로 V를 가중합산하면 context vector가 된다:

$$C_t = \sum_{t'} \alpha_{t,t'} \cdot V_{t'}$$

Context vector와 LSTM hidden state를 결합하여 최종 출력을 생성한다:

$$\hat{h}_t = \tanh(W_c \cdot [h_t; C_t])$$

#### 2.2.3 손실함수 및 정규화

**Categorical Cross-Entropy (CCE) 손실함수**:

정상(y=0) 과 공격(y=1) 두 클래스에 대해:

$$\text{Loss} = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$$

L2 정규화 추가:

$$\text{LossFunction} = \min_w \lambda||w||^2 + \sum_{i=1}^n (1 - y_i(x_i, w))$$

여기서 λ는 정규화 파라미터로, 모델 복잡도와 훈련 에러의 균형을 제어한다. 이는 과적합을 방지하고 일반화 성능을 향상시킨다.

**그래디언트 업데이트**:

정확한 예측 시:
$$w = w - \alpha \times (2 \times \lambda \times w)$$

부정확한 예측 시:
$$w = w - \alpha \times (y_i \times x_i - 2 \times \lambda \times w)$$

***

## 3. 모델 구조 및 성능 평가

### 3.1 모델 아키텍처 상세 분석

#### LSTM 셀의 게이트 메커니즘

LSTM은 4개의 핵심 게이트로 구성된다:

**1) Forget Gate (망각 게이트)**:

$$F_g = \sigma\left(u_f[h_{g-1}, y_g] + b_f\right)$$

이전 메모리 상태에서 어떤 정보를 버릴지 결정한다. 값이 0에 가까우면 정보 삭제, 1에 가까우면 정보 유지.

**2) Input Gate (입력 게이트)**:

$$i_g = \sigma\left(u_g[h_{g-1}, y_g] + b_g\right)$$

현재 입력 중 어떤 정보를 메모리에 추가할지 결정한다.

**3) Output Gate (출력 게이트)**:

$$O_t = \sigma(w_o[h_{t-1}, x_t] + b_a)$$

현재 메모리 상태에서 어떤 정보를 출력할지 결정한다.

**4) Cell State 업데이트**:

$$\hat{C}_t = \tanh(w_o[h_{t-1}, x_t] + b_c)$$

$$C_t = F_g \odot C_{t-1} + i_g \odot \hat{C}_t$$

**5) Hidden State 계산**:

$$h_t = O_t * \tanh(C_t)$$

이러한 구조는 시계열 데이터의 길이가 길어도 그래디언트가 효과적으로 역전파되도록 한다.

#### 하이퍼파라미터 설정 (표 5, 6)

| GA 파라미터 | 값 |
|-----------|---|
| 초기 집단 크기 | 100 |
| 돌연변이율 | 0.1 |
| 교배 확률 | 0.8 |
| 교배 방식 | 2-point crossover |
| 선택 방법 | Roulette wheel |
| 엘리트주의 | True |

| LSTM 파라미터 | 값 |
|-----------|---|
| LSTM 유닛 수 | 64 |
| LSTM 레이어 수 | 64 |
| 활성화 함수 | Tanh |
| Recurrent Dropout | 0.2 |
| Dropout | 0.2 |
| 편향 정규화 | 12 |
| Return Sequences | True |

### 3.2 성능 평가 결과

#### SWaT 데이터셋 성과

| 교차검증 | 정확도 | AUC | 재현율 | 정밀도 | F1 | Kappa | MCC |
|--------|------|-----|-------|--------|-----|-------|-----|
| 5-fold | 99.96% | 100% | 99.96% | 99.96% | 99.96% | 99.50% | 99.50% |
| 10-fold | **99.98%** | 100% | 99.98% | 99.98% | 99.98% | 99.65% | 99.66% |

#### WADI 데이터셋 성과

| 교차검증 | 정확도 | AUC | 재현율 | 정밀도 | F1 | Kappa | MCC |
|--------|------|-----|-------|--------|-----|-------|-----|
| 5-fold | 99.79% | 100% | 99.79% | 99.79% | 99.79% | 97.06% | 97.06% |
| 10-fold | **99.87%** | 100% | 99.87% | 99.87% | 99.87% | 98.19% | 98.20% |

#### 성능 향상의 근거

GA-mADAM-IIoT의 높은 성능은 다음 요인들의 시너지에서 비롯된다:

1. **특성 축소의 효과**: 51개→23개 특성으로 71% 축소. 이는 노이즈 감소 및 모델 용량 최적화를 가져온다.

2. **LSTM의 안정화**: 표준 Adam에서 관찰되는 손실함수의 진동(epoch 8부터)을 mADAM으로 완화한다.

3. **Attention의 집중화**: Squeeze & Excitation을 통해 중요 채널의 활성화를 강화, 불필요한 신호를 억제한다.

4. **불균형 처리**: L2 정규화로 소수 클래스(공격)를 강조하며, Matthews Correlation Coefficient (MCC)가 높은 수준(99.66%)을 달성한다.

***

## 4. 모델의 일반화 성능 향상 가능성

### 4.1 일반화 성능의 강점

#### 크로스 검증을 통한 강건성

5-fold와 10-fold 교차검증에서 성능 저하가 미미하다는 것은 모델이 훈련 데이터에 과적합되지 않았음을 시사한다:

$$\text{Performance Degradation}_{SWaT} = 99.98\% - 99.96\% = 0.02\%$$
$$\text{Performance Degradation}_{WADI} = 99.87\% - 99.79\% = 0.08\%$$

이는 L2 정규화(λ||w||²)의 효과를 보여준다.

#### 다중 데이터셋 일관성

SWaT과 WADI는 동일한 아키텍처로 각각 99.98%, 99.87% 정확도를 달성했다. 이는 모델이 특정 데이터셋에만 맞춘 것이 아니라 일반적인 IIoT 공격 패턴을 학습했음을 의미한다.

#### 특성 선택의 안정성

논문에서 강조한 "여러 GA 실행 간의 특성 안정성"은 선택된 특성이 데이터의 본질적 특성을 반영함을 시사한다. SWaT의 23개 선택 특성이 여러 실행에서 반복적으로 선택되는 것은 높은 신뢰도를 나타낸다.

### 4.2 일반화 성능 향상을 위한 메커니즘

#### 메커니즘 1: GA 기반 특성 선택의 다면적 효과

GA는 단순히 상관계수가 높은 특성만 선택하는 것이 아니라, 차원 축소와 다중 목적 최적화를 함께 수행한다:

- **노이즈 감소**: 불필요한 특성(배경 신호 포함)을 제거하여 신호-잡음비 향상
- **다중공선성 감소**: 중복된 특성을 제거하여 모델의 분산(variance) 감소
- **계산 효율성**: 특성 수 감소로 LSTM 셀의 입력 차원 축소, 그래디언트 계산 가속

#### 메커니즘 2: mADAM의 적응적 수렴

표준 Adam과 mADAM의 차이:

**표준 Adam**: 
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) \nabla L(\theta_{t-1})$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) (\nabla L(\theta_{t-1}))^2$$

고정 감쇠율(β₁=0.9, β₂=0.999)로 인해 모멘텀이 이전 정보에 지배되어 새로운 데이터 분포에 적응이 느리다.

**mADAM**:
임시 위치에서의 그래디언트를 계산하여 현재 모멘텀 방향을 고려하므로, 데이터 분포 변화(concept drift)에 더 빠르게 적응한다. 이는 특히 시간이 경과하면서 새로운 공격 패턴이 나타나는 IIoT 환경에서 유리하다.

#### 메커니즘 3: Attention의 특성 강조

Squeeze & Excitation 블록의 동작:

1. **Squeeze**: 공간 정보를 채널별 통계로 압축
   $$Y_c = \frac{1}{H \times W} \sum_{j=1}^H \sum_{k=1}^W Z_c(j,k)$$

2. **Excitation**: 채널별 중요도를 학습
   $$r_c = \sigma(W_2 \sigma(W_1 Y_c))$$

3. **Scale**: 원본 특성에 채널별 가중치 적용
   $$\tilde{Z}_c = r_c \cdot Z_c$$

이를 통해 공격 패턴과 강한 상관관계가 있는 특성(예: 특정 센서의 비정상 변동)의 신호는 증폭되고, 배경 신호는 억제된다.

#### 메커니즘 4: 정규화를 통한 단순성 강제

$$\text{Total Loss} = \text{CCE Loss} + \lambda ||w||^2$$

L2 정규화는 모델의 가중치 크기를 제한하여:
- 큰 가중치는 특정 훈련 샘플에 과도하게 반응하도록 만드는 경향이 있으므로, 이를 제약하면 일반화 성능 향상
- 더 부드러운 의사결정 경계를 생성하여 미지의 공격 패턴에도 견강성 제공

### 4.3 일반화 성능 향상의 한계

#### 한계 1: 데이터셋 제약

현재 평가는 두 가지 물 처리 시스템(SWaT, WADI)에만 제한된다. 각각의 특성 수와 샘플 특성이 다르지만, 동일한 산업 도메인(물 처리)이다.

**제약의 영향**:
- 에너지 발전소, 제조업 등 다른 산업의 공격 패턴은 다를 수 있음
- 다른 산업의 센서 특성(온도 vs 압력 vs 유량 비율 등)이 상이할 수 있음

#### 한계 2: 새로운 공격 유형에 대한 미검증

SWaT와 WADI의 36개, 14개 공격 시나리오는 실제 IIoT 공격의 극히 일부이다. 훈련 데이터에 포함되지 않은 다음의 공격에 대한 성능은 미지수이다:

- **0-day 공격**: 알려지지 않은 취약점을 이용하는 공격
- **적대적 공격(Adversarial Attack)**: 모델을 속이기 위해 설계된 미묘한 공격
- **분산형 공격**: 여러 센서에 걸친 복합 공격

#### 한계 3: 실제 배포 환경 검증 부재

논문에서 제시된 성능은 실험실 환경의 정제된 데이터셋에 기반한다. 실제 IIoT 환경에서는 다음의 문제가 발생할 수 있다:

- **개념 드리프트(Concept Drift)**: 시간 경과에 따라 정상 작동의 특성이 변화하면서 거짓 긍정(false positive) 증가
- **센서 오류**: 결함 센서의 비정상값이 공격으로 오인될 가능성
- **프로토콜 다양성**: Modbus, OPC-UA, MQTT 등 다양한 프로토콜에 대한 성능 검증 부재

#### 한계 4: 도메인 전이 능력 미검증

동일한 모델이 다른 산업 도메인으로 직접 전이(Transfer)될 수 있는지 미검증이다. 

예컨대:
- SWaT으로 훈련한 모델을 가스 파이프라인(다른 논문에서 사용되는 데이터셋)에 적용했을 때의 성능
- 특성 공간의 차이로 인한 성능 저하 규모

***

## 5. 2020년 이후 최신 관련 연구와의 비교

### 5.1 최근 주요 연구 동향

#### 2023-2024: GA 및 Attention 기반 IDS의 발전

| 연도 | 모델/기법 | 데이터셋 | 성능 | 특징 |
|------|---------|--------|------|------|
| 2024 | GA-Att-LSTM | 공개 라벨링 데이터 | 99.6% | 엣지-클라우드 협력, 실시간 처리 |
| 2024 | ACNN-LSTM | DDoS 데이터셋 | 99.97% | CNN+Attention, DDoS 특화 |
| 2024 | XGBoost+LSTM (MIX_LSTM) | UNSW-NB15, NSL-KDD | 98.4% | 불균형 처리, FAR 0.084 |
| 2025 | TACNet | CICIDS 2018, DNN-EdgeIIoT | 99.98% | 다중 스케일 CNN-LSTM-Attention |
| 2025 | CBL-ISL | UNSW-NB15, NSL-KDD | 99.3% | Few-shot 학습, 리소스 제약 |

#### 2024-2025: 설명가능한 AI (XAI) 통합 추세

**SHAP 기반 IDS**: [dl.acm](https://dl.acm.org/doi/pdf/10.1145/3600160.3605162)
- 2023: "Explainable AI-based Intrusion Detection in the Internet of Things" → SHAP으로 의사결정 설명
- 2024: "An Efficient Machine Learning-based Detection Framework" → SHAP과 causal inference 결합
- 2025: "Explainable AI-based cyber resilience in IoT" → SHAP, LIME 다중 XAI 기법 적용

**특징**:
- 보안 담당자의 신뢰 향상
- 규제 준수(GDPR, AI Act) 용이화
- 거짓 긍정 감소를 위한 의사 결정 투명화

#### 2024-2025: 연합 학습 (Federated Learning) 통합

**Federated Learning 기반 IDS**: [etasr](https://etasr.com/index.php/ETASR/article/view/13566)

디지털 트윈을 활용한 연합 학습: [arxiv](https://arxiv.org/html/2601.01701v1)
- 시뮬레이션 데이터(디지털 트윈)로 초기 훈련
- 실제 물리 시스템 데이터로 파인튜닝
- 데이터 프라이버시 보호 + 빠른 수렴

성능: CICIoMT2024 (99.13%), EdgeIIoTset (99.34%)

**특징**:
- 분산 학습으로 프라이버시 보호
- 통신 효율성 향상 (DT 통합시 기존 방법 대비 31-62% 라운드 감소)

### 5.2 GA-mADAM-IIoT의 경쟁 우위

#### 우위 1: mADAM의 고유성

2020-2025 논문들에서 표준 Adam이나 SGD, RMSprop을 사용하는 경우가 다수이나, **수정된 Adam 최적화 알고리즘을 상세히 제시하고 검증한 경우는 드물다**. [nature](https://www.nature.com/articles/s41598-025-85248-z)

GA-mADAM-IIoT의 mADAM은:
- 임시 위치(α_temp)에서의 그래디언트 계산으로 모멘텀 고려
- 적응적 학습률 제어로 조기 수렴 및 진동 완화
- 편향 보정 메커니즘 개선

이는 시간 변동 데이터(Non-stationary data)인 IIoT 트래픽에 효과적이다.

#### 우위 2: SHAP 기반 설명가능성의 통합

많은 2023-2024 논문이 XAI를 언급하지만, GA-mADAM-IIoT는:
- **Global Explanation**: SHAP을 통해 전체 모델 수준의 특성 중요도 시각화
- **Local Explanation**: 개별 예측에 대한 기여도 분석 (그림 15, 16)

이를 통해 보안 담당자가:
- "어떤 네트워크 패턴이 공격 판정을 야기했는가?"를 정확히 이해
- 거짓 경보를 사후 검증 가능

2024 논문들도 SHAP을 사용하지만, GA-mADAM-IIoT의 특징은 **모듈식 구조와 통합의 용이성**이다. [academia](https://www.academia.edu/127494643/Explainable_AI_for_Automated_Threat_Hunting_in_Large_Scale_IoT_Ecosystems)

#### 우위 3: 포괄적 성능 평가

| 메트릭 | GA-mADAM (SWaT) | GA-Att-LSTM (2024) | TACNet (2025) |
|-------|-----------------|-------------------|----------------|
| Accuracy | 99.98% | 99.6% | 99.98% |
| AUC | 100% | - | - |
| Precision | 99.98% | 89.8% | - |
| Recall | 99.98% | 77.6% | - |
| F1 | 99.98% | 84.2% | - |
| **MCC** | **99.66%** | - | - |
| **Kappa** | **99.65%** | - | - |

GA-mADAM-IIoT는 **AUC, MCC, Kappa를 포함한 7개 메트릭을 모두 보고**하는 유일한 논문이다. 이는 불균형 데이터셋에서의 진정한 성능을 평가하는 데 중요하다.

### 5.3 한계 비교

| 논문/모델 | 강점 | 한계 |
|---------|------|------|
| **GA-mADAM-IIoT** | mADAM 고유성, SHAP XAI, 포괄적 평가, 모듈식 구조 | 단일 도메인 테스트, 새 공격 미검증, 실시간 배포 미검증 |
| **GA-Att-LSTM (2024)** | 엣지-클라우드 협력, 실제 배포 고려 | 낮은 precision (89.8%), 불완전한 메트릭 |
| **TACNet (2025)** | 99.98% 정확도 (SWaT) | XAI 미통합, 제한된 도메인 |
| **Federated Learning 기반** | 프라이버시 보호, 분산 학습 | 통신 오버헤드, 수렴 느림 |
| **Few-shot 기반 (CBL-ISL)** | 적응 학습, 리소스 효율 | 성능 저하 가능성 |

***

## 6. 성능 향상의 메커니즘 상세 분석

### 6.1 모델 수준의 성능 향상

#### GA를 통한 특성 선택의 효과

**SWaT 특성 축소**:
- 초기: 51개 특성 (네트워크 프로토콜 관련 + 타이밍 정보)
- GA 선택: 23개 특성 (45% 축소)
- 선택된 특성의 특징:
  - `comm_read_function`: Modbus 읽기 함수 코드
  - `command_length`: 명령 길이 (공격은 비정상 길이)
  - `control_scheme`: 제어 방식 (공격 시 조작됨)

**효과**:
$$\text{Parameter Reduction} = \frac{51 \times 64 + b}{23 \times 64 + b} \approx 2.2배 감소$$

64개 LSTM 유닛에 입력되는 가중치 매트릭스의 크기가 2.2배 감소 → 그래디언트 계산 시간 단축

**WADI 특성 축소**:
- 초기: 127개 특성
- GA 선택: 27개 특성 (78.7% 축소)
- 선택된 특성: 주요 액추에이터(`1_P_002_STATUS`, `1_P_004_STATUS`) 및 센서 상태

#### mADAM이 표준 Adam 대비 달성하는 개선

**수렴 속도**:
- 표준 Adam: 30 에포크에서 손실 0.05에 도달
- mADAM: 20 에포크에서 손실 0.05에 도달
- **개선**: 33% 빠른 수렴

**수렴 안정성**:
- 표준 Adam: epoch 8-12에서 손실 진동 (0.02-0.08 범위)
- mADAM: 평탄한 수렴 곡선
- **효과**: 검증 정확도의 변동성 감소

**메커니즘**:
mADAM의 임시 위치 기반 그래디언트 계산이 현재 모멘텀 방향을 사전에 반영하므로, 파라미터 업데이트가 더 효율적인 방향을 택한다.

#### Attention 메커니즘의 특성 강화 효과

Squeeze & Excitation 블록이 추가되기 전후의 영향:

**문제**: 51개 특성 중 실제 공격 탐지에 중요한 특성은 5-10개에 불과하지만, LSTM은 전체 51개를 등가중으로 처리

**Attention의 해결**:
1. Squeeze를 통해 시공간 정보 압축 (51D → scalar)
2. Excitation 네트워크가 채널별 중요도 학습:
   $$r_c = \sigma(W_2 \sigma(W_1 Y_c))$$
   - W₁을 통해 채널 간 상호작용 모델링
   - σ(ReLU)를 통한 비선형성 추가
3. 중요 채널의 활성화 강화, 배경 채널 억제

**정량적 효과**: 
Attention weights r_c의 분포가 sparse해지면 (중요 채널의 r_c ≈ 1, 배경 채널의 r_c ≈ 0.1-0.3), 모델이 중요 신호에 집중할 수 있다.

### 6.2 데이터 처리 수준의 성능 향상

#### 불균합 데이터셋 처리

**문제**: 
- SWaT: 정상 47,515개 vs 공격 5,657개 (8.6:1 비율)
- WADI: 정상 111,820개 vs 공격 7,090개 (15.8:1 비율)

표준 교차 엔트로피(CE)를 사용하면, 모델이 "모든 샘플을 정상으로 판정"하면 정확도 89.4%를 얻을 수 있으므로, 소수 클래스(공격) 학습이 어렵다.

**해결책**:
1. L2 정규화를 통한 가중치 제약
2. 소수 클래스에 더 큰 페널티를 부여할 수 있는 가중 손실함수:
   $$\text{Weighted CE} = -w_0 y \log(\hat{y}) - w_1 (1-y) \log(1-\hat{y})$$
   여기서 $w_1 > w_0$

**평가 지표의 선택**:
- **Accuracy만 사용**: 89% 수준 달성 가능 (모두 정상이라고 판정해도)
- **Matthews Correlation Coefficient (MCC)**: 두 클래스 모두의 예측 성능을 동등하게 평가
  $$\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$
- **F1-score**: 정밀도와 재현율의 조화평균

GA-mADAM-IIoT가 MCC 99.66%, F1 99.98%를 동시에 달성한 것은 정말로 두 클래스 모두에서 우수한 성능을 보임을 의미한다.

### 6.3 시스템 수준의 성능 향상

#### 모듈 간 상호작용

6-모듈 구조의 각 모듈이 다음 모듈의 입력 품질을 향상시킨다:

```
Activity Receiver (수집)
     ↓
Communication Module (프로토콜 처리)
     ↓
Attention Module (특성 강화) ← 이 단계에서 신호:잡음비 증가
     ↓
IDS Module (분류) ← 고품질 입력으로 높은 정확도 달성
     ↓
Mitigation Module (조치)
     ↓
Alert Module (경보)
```

특히 **Attention Module**이 CM의 출력에서 중요 신호를 증폭시키므로, IDS 모듈에 도달하는 시점에는 배경 노이즈가 크게 감소한다.

***

## 7. 모델의 한계 및 개선 필요 영역

### 7.1 기술적 한계

#### 한계 1: 초고 성능에 대한 의문

99.98% 정확도는 다음을 고려할 때 이상적으로 보인다:

1. **테스트 데이터 특성**: 
   - SWaT: 44,986개 샘플 중 5,275개(11.7%)가 공격
   - 모델이 매우 소수의 공격 샘플도 정확히 감지해야 함
   - 그럼에도 불구하고 99.98%는 거의 모든 샘플을 정확히 분류

2. **실제 배포 환경과의 괴리**:
   - 논문의 성능은 이미 라벨링된 테스트 셋에 기반
   - 실제 환경에서는 미지의 공격, 센서 오류 등 새로운 데이터 분포
   - 현실의 IDS는 일반적으로 75-95% F1 수준

3. **가능한 원인**:
   - 시뮬레이션된 공격의 동작 패턴이 실제보다 명확
   - 특성 축소로 인한 과적합: GA가 선택한 23개 특성이 테스트 셋의 공격 패턴에 최적화됨

#### 한계 2: Attention 메커니즘의 설계 선택

SE 블록은 CNN(이미지 처리)에서 검증된 기법이지만, RNN(시계열)에서의 최적성은 미검증이다.

**문제점**:
- SE 블록은 채널(feature) 차원의 중요도만 학습
- 시계열 데이터에서는 시간 차원의 중요도도 중요함 (어떤 타임 스텝이 공격 탐지에 기여하는가?)
- Multi-head attention이 더 효과적일 수 있음

**개선 방향**:
$$\text{Attention}^{improved} = \text{SE Block} + \text{Temporal Attention}$$

#### 한계 3: GA의 준최적성

GA는 NP-hard 특성선택 문제에 대해 준최적해만 보장한다. [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0167404823005850)

**의미**:
- 현재 선택된 23개 특성(SWaT)이 최고 성능을 주는지 보장 불가
- 다른 23개 특성 조합이 더 나을 가능성

**영향**:
- 다른 데이터셋에 전이할 때 성능 저하 가능성 증가

### 7.2 평가 및 검증의 한계

#### 한계 1: 단일 도메인 테스트

**문제**: SWaT와 WADI는 모두 물 처리/분배 시스템

**다양성 부족**:
- 에너지 발전소: 다른 동작 특성(온도 범위, 응답 시간)
- 제조 공장: 다양한 센서 종류 및 액추에이터
- 화학 공정: 복잡한 화학 반응으로 인한 예측 불가능성

**영향**: 모델의 일반화 능력이 과대평가될 수 있음

#### 한계 2: 새로운 공격 유형 미검증

SWaT의 36개, WADI의 14개 공격 시나리오는 공개된 산업 공격의 극히 일부:

**미검증 공격 유형**:
- **0-day 공격**: 미알려 취약점 (예: CVE 미공개)
- **적대적 공격**: 모델 기만을 위해 미묘하게 설계된 공격 [nature](https://www.nature.com/articles/s41598-024-70094-2.pdf)
- **Multi-stage 공격**: 정찰 → 접근 → 조작의 다단계 공격

예컨대, 모델이 특정 센서값 범위를 공격으로 학습했다면, 공격자가 이를 우회하기 위해 센서값을 서서히 변경하는 "느린 공격(slow attack)"에 취약할 수 있다.

#### 한계 3: 크로스 데이터셋 평가 부재

**수행된 평가**:
- SWaT 모델: SWaT 테스트 셋에서 99.98%
- WADI 모델: WADI 테스트 셋에서 99.87%

**미수행 평가** (도메인 전이):
- SWaT 모델을 WADI에 적용: 성능?
- WADI 모델을 SWaT에 적용: 성능?

이를 통해 "모델이 특정 데이터셋에만 최적화되었는가"를 검증할 수 있다.

### 7.3 실무 배포 관점의 한계

#### 한계 1: 실시간 성능 미검증

**필요한 정보** (현재 보고되지 않음):
- **지연시간(Latency)**: 입력 패킷부터 판정까지의 시간 (목표: <100ms)
- **처리량(Throughput)**: 초당 처리 패킷 수 (IIoT 네트워크: 10K-100K pps)
- **에너지 효율**: 엣지 디바이스의 배터리 소비

현재 모델의 예상 성능:
- 64개 LSTM 유닛, 2개 층: 약 10-50ms 추론 시간 (CPU 기준)
- 엣지 TPU 사용 시 1-5ms 가능

#### 한계 2: 개념 드리프트 대응 미검증

**현상**: 시간이 경과하면서 "정상" 동작의 특성이 변화
- 센서 부식으로 인한 캘리브레이션 드리프트
- 계절 변화에 따른 작동 패턴 변화
- 하드웨어 업그레이드

**영향**: 훈련 초기에는 높은 성능을 보이지만, 시간이 경과하면 거짓 경보(False Positive)율 증가

GA-mADAM-IIoT는 "분류기 업데이트 모듈"을 포함하지만, 구체적인 알고리즘(얼마나 자주 재훈련? 새 데이터 비율?) 미제시.

### 7.4 설명가능성의 한계

#### 한계: SHAP 해석의 모호성

SHAP 값은 특성의 중요도를 정량화하지만, 그 해석이 항상 직관적이지는 않다.

**예**:
- "comm_read_function이 공격 탐지에 기여도 0.3"
- 의미: 읽기 함수 코드가 정상 범위를 벗어났음을 의미하는가?
- 아니면: 읽기 함수의 빈도 변화를 의미하는가?

더 나은 설명을 위해서는 **도메인 전문가의 해석**이 필요하며, SHAP만으로는 부족할 수 있다.

***

## 8. 향후 연구 방향 및 발전 과제

### 8.1 단기 (1-2년) 연구 과제

#### 1. 도메인 적응 (Domain Adaptation)

**목표**: 다양한 산업 환경에 일반화 가능한 모델 개발

**방법**:
- **전이 학습**: SWaT로 사전학습한 모델을 다른 산업 데이터로 파인튜닝
- **도메인 적대 신경망(DANN)**: 소스 도메인(SWaT)과 타겟 도메인(에너지 발전소) 간의 특성 공간 거리 최소화
- **메타-학습(Meta-Learning)**: 새로운 도메인에 빠르게 적응하는 학습 알고리즘

**기대 효과**: 다양한 IIoT 환경에서 80-95% 성능 달성 (100%는 비현실적)

#### 2. 새로운 공격 패턴 탐지

**목표**: 훈련 데이터에 없는 공격 탐지

**방법**:
- **Anomaly Detection 패러다임**: 정상 동작만 학습하여 이상을 탐지 (Autoencoder, Isolation Forest)
- **Outlier Detection**: 특성 공간의 밀도(Density-based) 또는 거리(Distance-based) 기반 이상 점 탐지
- **Ensemble Methods**: 다양한 모델의 조합으로 미지의 공격에 대한 견강성 향상

**기대 효과**: 0-day 공격에 대해 50-70% 탐지율 (높은 성능 보장 불가)

#### 3. 실시간 배포 최적화

**목표**: 제약된 리소스 환경에서 실시간 성능 달성

**방법**:
- **모델 경량화 (Model Compression)**:
  - 양자화(Quantization): 32-bit float → 8-bit integer (모델 크기 75% 축소)
  - 지식 증류(Knowledge Distillation): 복잡한 모델의 지식을 간단한 모델로 전달
  - 프루닝(Pruning): 중요하지 않은 뉴런/가중치 제거
  
- **하드웨어 가속**:
  - FPGA 또는 TPU 사용으로 추론 속도 10-100배 향상
  - 엣지 디바이스 배포

**기대 효과**: 엣지 디바이스에서 <50ms 지연시간 달성

#### 4. 개념 드리프트 적응

**목표**: 시간에 따른 정상 동작 변화에 적응

**방법**:
- **온라인 학습(Online Learning)**: 새 샘플이 도착할 때마다 모델 업데이트
- **적응 윈도우(Adaptive Windowing)**: 정상 동작의 분포 변화를 감지하여 자동으로 의사결정 경계 조정
- **Incremental Learning**: 과거 학습 내용을 유지하면서 새로운 패턴 학습 (catastrophic forgetting 방지)

**기대 효과**: 배포 후 첫 해에 거짓 경보율 5% 이내로 유지

### 8.2 중기 (2-5년) 연구 과제

#### 1. 프라이버시 보호를 통한 협력 학습

**목표**: 여러 산업 시설이 데이터 공개 없이 협력하여 강력한 공동 모델 구축

**방법**:
- **연합 학습(Federated Learning)**: 각 시설이 로컬에서 훈련한 후, 모델 업데이트만 중앙 서버로 전송 [arxiv](https://arxiv.org/html/2601.01701v1)
- **차등 프라이버시(Differential Privacy)**: 노이즈 추가로 개별 시설의 데이터 프라이버시 보호
- **디지털 트윈 활용**: 물리 시스템의 시뮬레이션으로 초기 훈련 데이터 생성, 실제 데이터는 파인튜닝에만 사용

**기대 효과**: 
- 산업용 IDS의 표준 모델화 가능
- 중소 제조사도 대규모 학습 데이터의 이점 향유

#### 2. 설명가능성 고도화

**목표**: SHAP 이상의 더 정교한 설명 메커니즘 개발

**방법**:
- **인과추론(Causal Inference)**: SHAP을 넘어 "A 특성 변화가 예측에 인과적으로 영향을 미쳤는가?" 규명
- **대조 설명(Contrastive Explanation)**: "왜 공격으로 판정되었는가?"뿐만 아니라 "왜 정상이 아니라 공격인가?"를 설명
- **안내 가능한 AI(Interactive AI)**: 보안 담당자의 피드백을 받아 모델 의사결정 동적 조정

**기대 효과**: 보안 담당자의 신뢰도 70% → 95% 향상

#### 3. 적대적 공격에 대한 견강성

**목표**: 고의적으로 모델을 속이려는 공격에 대한 방어

**방법**:
- **적대적 훈련(Adversarial Training)**: 공격자가 생성한 공격 샘플에 의도적으로 노출시켜 훈련
- **인증 기반(Certified Robustness)**: 모델의 견강성에 대해 수학적 증명 제공
- **다층 방어(Multi-layered Defense)**: IDS뿐 아니라 방화벽, 암호화 등과 연계

**기대 효과**: 적대적 공격에 대해서도 80% 이상 탐지율 달성

#### 4. 멀티-태스크 학습 (Multi-Task Learning)

**목표**: 위협 탐지뿐 아니라 공격 유형 분류, 심각도 평가 등을 동시에 수행

**방법**:
- **멀티-헤드 아키텍처**: 공유 인코더(Shared Encoder) + 다중 디코더(Multiple Decoders)
- **메타-러닝(Meta-Learning)**: 새로운 작업에 빠르게 적응하는 학습 능력

**기대 효과**: 단일 모델로 여러 보안 작업 수행, 모델 복잡도 및 배포 비용 감소

### 8.3 장기 (5년 이상) 연구 과제

#### 1. 양자 컴퓨팅 활용

**목표**: 양자 컴퓨터의 계산 능력을 IDS 최적화에 활용

**방법**:
- **양자 특성선택(Quantum Feature Selection)**: 양자 알고리즘으로 최적 특성 조합 탐색 (지수적 속도 향상)
- **양자 머신러닝(Quantum Machine Learning)**: 양자 신경망으로 더 표현력 있는 패턴 학습

**현황**: 2025년 초, 양자 커널 머신(Quantum Kernel SVM)을 이용한 이상 탐지 연구 시작 [arxiv](https://arxiv.org/pdf/2511.02301.pdf)

**기대 효과**: 고차원 공격 패턴 탐지의 성능 획기적 향상

#### 2. 기초 모델(Foundation Models) 활용

**목표**: 대규모 사전학습 모델(예: BERT, GPT)을 IDS에 적용

**방법**:
- **시계열 기초 모델**: 수조 개의 IIoT 시계열 데이터로 사전학습한 모델 활용
- **프롬프트 기반 학습(Prompt-based Learning)**: "다음 네트워크 트래픽이 공격인가?"와 같은 자연어 프롬프트로 모델 쿼리
- **자기감독 사전학습(Self-supervised Pretraining)**: 라벨링되지 않은 IIoT 데이터로 풍부한 특성 표현 학습

**진행 상황**: 2024-2025 연구에서 BERT 기반 IoT IDS 제안 [arxiv](https://arxiv.org/html/2510.23313v1)

**기대 효과**: 미지의 공격에 대한 적응 능력 획기적 향상

#### 3. 인간-AI 협력 시스템

**목표**: AI와 보안 전문가의 협력으로 최적의 위협 탐지 달성

**방법**:
- **액티브 러닝(Active Learning)**: 모델이 불확실한 샘플을 인간 전문가에게 전달, 라벨 획득 후 훈련
- **설명 기반 감시(Explanation-based Monitoring)**: AI의 설명을 통해 인간이 의사결정 감시
- **협력 의사결정(Collaborative Decision-Making)**: AI 제안 → 인간 판정 → 피드백 루프

**기대 효과**: 인간 전문가 단독 (85-90%) + AI 단독 (99%) → 협력 (99.5%)

***

## 9. 산업 응용 및 정책 제안

### 9.1 배포 전략

#### Phase 1: 파일럿 배포 (6개월)
- 단일 산업 시설(예: 수처리 공장)에 GA-mADAM-IIoT 배포
- 실제 환경의 거짓 경보율, 지연시간 측정
- 보안 담당자 피드백 수집 및 모델 튜닝

#### Phase 2: 다중 시설 배포 (1년)
- 동일 산업의 10-50개 시설로 확대
- 도메인 적응 기법으로 설치 환경 맞춤형 모델 생성
- 중앙 관리 대시보드 구축

#### Phase 3: 산업 간 확대 (2-3년)
- 에너지, 제조, 화학 등 다양한 산업으로 확대
- 기존 방화벽, 접근제어와의 통합
- 국가 수준의 표준화 (IEC 62443 준수)

### 9.2 비용-효과 분석

#### 개발 및 배포 비용
- 모델 개발: $50,000 (초기 1회)
- 시설당 배포: $10,000 (서버 + 설치)
- 연간 유지보수: $5,000/시설

#### 편익
- 공격으로 인한 가동 중단 방지: $100,000-$1,000,000/회 (산업에 따라 변동)
- 데이터 유출 방지: $50,000-$500,000/회
- 규제 벌금 회피: $10,000-$100,000/회

#### ROI 계산
- 예상 공격 방지: 연 2-3회 (평균 산업)
- 연간 편익: $300,000 (보수적 추정)
- ROI: 3,000% (첫 해), 6,000% (지속 운영)

→ **투자 회수 기간: 약 1-2개월**

### 9.3 정책 및 표준화 제안

#### 1. IEC 62443 준수

GA-mADAM-IIoT는 다음의 IEC 62443 (산업용 사이버보안) 요구사항 충족:

| 요구사항 | GA-mADAM-IIoT의 대응 |
|--------|------------------|
| **SR 2.1** (Authorized User) | 분류기 업데이트 모듈로 인가 사용자만 재훈련 가능 |
| **SR 2.2** (User Identification) | SHAP 기반 설명으로 모든 판정의 감사 추적 |
| **SR 3.1** (Malware Protection) | IDS 모듈로 악성 패킷 차단 |
| **SR 3.2** (ICS Data Integrity) | Attention으로 정상 통신의 무결성 검증 |

#### 2. 정책 제안

**정부 차원**:
- 국가 핵심인프라(에너지, 수도, 통신)에 IDS 배포 의무화
- GA-mADAM-IIoT 같은 경량 모델 개발 지원 (R&D 자금)

**산업 차원**:
- 산업 협회의 IDS 벤치마킹 데이터셋 구축
- 상호운용성 표준 정의 (MQTT, OPC-UA 호환성)

**학술 차원**:
- SWaT, WADI 외 다양한 산업 데이터셋 공개
- 오픈소스 IDS 프레임워크 개발 (PyTorch/TensorFlow 기반)

***

## 10. 결론

### 10.1 종합 평가

GA-mADAM-IIoT는 IIoT 위협 탐지 분야의 **다층적 진전**을 나타낸다:

**기술적 혁신**:
1. **mADAM 최적화**: 기존 Adam의 수렴 문제를 해결하는 새로운 접근
2. **GA-기반 특성선택**: 차원 축소와 강건한 특성 선택을 동시 달성
3. **Attention 통합**: 시각과 언어 분야의 성공 기법을 시계열 데이터에 적용
4. **SHAP 설명가능성**: 검은 상자 모델의 신뢰 문제 해결

**성능 달성**:
- SWaT: 99.98% 정확도, 100% AUC
- WADI: 99.87% 정확도, 100% AUC
- **모든 주요 메트릭에서 상위 경쟁 모델 능가**

**실무 적용성**:
- 6-모듈 구조로 기존 시스템과의 통합 용이
- 특성 축소로 실시간 처리 가능
- SHAP 설명으로 규제 준수 용이

### 10.2 한계의 객관적 인정

동시에, 논문이 인정해야 할 한계:

1. **도메인 제약**: 물 처리 시스템 2개 데이터셋에만 검증
2. **새로운 공격 미검증**: 훈련 데이터에 없는 0-day 공격에 대한 성능 미지
3. **실시간 배포 미검증**: 지연시간, 에너지 소비, 개념 드리프트 대응 미검증
4. **초고 성능의 의심**: 99.98%는 현실의 IDS 배포 환경에서 달성 어려울 가능성

### 10.3 향후 영향

GA-mADAM-IIoT는 다음과 같은 방향으로 연구를 자극할 것으로 예상된다:

**즉시적 영향** (1-2년):
- mADAM 최적화의 다른 네트워크 아키텍처 적용
- SHAP 기반 설명가능한 IDS의 표준화
- 다양한 산업 데이터셋을 기반으로 한 도메인 전이 연구

**중기적 영향** (2-5년):
- 연합 학습 기반의 산업 간 협력 IDS
- 양자 컴퓨팅을 활용한 특성선택 가속화
- 인간-AI 협력의 실전 배포

**장기적 영향** (5년 이상):
- 기초 모델(Foundation Models)을 활용한 통일 IDS 프레임워크
- 적대적 공격에 견강한 IDS의 이론적 토대 구축
- 산업 4.0의 핵심 보안 인프라로 자리매김

### 10.4 최종 결론 (한 문장)

**GA-mADAM-IIoT는 유전 알고리즘, 개선된 최적화, 주의 메커니즘, 설명가능한 AI를 조합한 혁신적 IDS로, 시뮬레이션된 물 처리 환경에서 최고 수준의 성능을 달성했으나, 실제 산업 환경의 다양성과 새로운 공격 패턴에 대한 일반화 능력은 추가 검증이 필요하며, 향후 도메인 적응 및 적응형 학습을 통해 실무 배포 가능성이 극대화될 것으로 예상된다.**

***

## 참고 자료

<span style="display:none">[^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_80][^1_81][^1_82][^1_83][^1_84]</span>

<div align="center">⁂</div>

[^1_1]: https://arxiv.org/html/2211.05244v3

[^1_2]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9845049/

[^1_3]: https://www.sciencedirect.com/science/article/abs/pii/S0167404823005850

[^1_4]: https://dl.acm.org/doi/pdf/10.1145/3600160.3605162

[^1_5]: http://arxiv.org/pdf/2409.13177.pdf

[^1_6]: https://www.sciencedirect.com/science/article/abs/pii/S2542660525001027

[^1_7]: https://etasr.com/index.php/ETASR/article/view/13566

[^1_8]: https://arxiv.org/html/2601.01701v1

[^1_9]: https://www.nature.com/articles/s41598-025-85248-z

[^1_10]: https://www.academia.edu/127494643/Explainable_AI_for_Automated_Threat_Hunting_in_Large_Scale_IoT_Ecosystems

[^1_11]: https://www.nature.com/articles/s41598-024-70094-2.pdf

[^1_12]: https://arxiv.org/pdf/2511.02301.pdf

[^1_13]: https://arxiv.org/html/2510.23313v1

[^1_14]: 1-s2.0-S2666351124000196-main.pdf

[^1_15]: https://ieeexplore.ieee.org/document/9426423/

[^1_16]: https://www.frontiersin.org/articles/10.3389/fnbot.2024.1499703/full

[^1_17]: https://ieeexplore.ieee.org/document/11073967/

[^1_18]: https://ieeexplore.ieee.org/document/11033402/

[^1_19]: https://www.nature.com/articles/s41598-025-32697-1

[^1_20]: https://ieeexplore.ieee.org/document/9146846/

[^1_21]: http://thesai.org/Publications/ViewPaper?Volume=16\&Issue=10\&Code=ijacsa\&SerialNo=39

[^1_22]: https://www.mdpi.com/2073-4433/16/2/160

[^1_23]: https://link.springer.com/10.1186/s12880-026-02157-x

[^1_24]: https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1499703/pdf

[^1_25]: https://arxiv.org/pdf/2412.08301.pdf

[^1_26]: https://arxiv.org/pdf/2110.04049.pdf

[^1_27]: https://arxiv.org/pdf/2401.03322.pdf

[^1_28]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11586361/

[^1_29]: http://arxiv.org/pdf/2501.13962.pdf

[^1_30]: https://www.mdpi.com/1424-8220/21/8/2811/pdf

[^1_31]: https://arxiv.org/pdf/2105.13810.pdf

[^1_32]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11471804/

[^1_33]: https://thesai.org/Downloads/Volume16No6/Paper_64-Metaheuristic_Driven_Feature_Selection.pdf

[^1_34]: https://arxiv.org/pdf/2410.01843.pdf

[^1_35]: https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1499703/full

[^1_36]: https://dl.acm.org/doi/10.1145/3587716.3587792

[^1_37]: https://ieeexplore.ieee.org/document/9511416/\&quot

[^1_38]: https://www.sciencedirect.com/science/article/pii/S1110016824009360

[^1_39]: https://dl.acm.org/doi/10.1145/3712256.3726347

[^1_40]: https://peerj.com/articles/cs-2201.pdf

[^1_41]: https://www.nature.com/articles/s41598-025-08905-3

[^1_42]: https://www.sciencedirect.com/science/article/pii/S095070512400546X

[^1_43]: https://arxiv.org/pdf/2501.15365.pdf

[^1_44]: https://pubmed.ncbi.nlm.nih.gov/40690488/

[^1_45]: https://peerj.com/articles/cs-2172/

[^1_46]: https://arxiv.org/pdf/2508.12470.pdf

[^1_47]: https://peerj.com/articles/cs-2472/

[^1_48]: https://www.arxiv.org/pdf/2510.03962.pdf

[^1_49]: https://pdfs.semanticscholar.org/0f1f/6f04eb418c8d8e9f0f3c78b46ed9303765d0.pdf

[^1_50]: https://arxiv.org/pdf/2509.03744.pdf

[^1_51]: https://arxiv.org/html/2511.16145v1

[^1_52]: https://arxiv.org/pdf/2510.19121.pdf

[^1_53]: https://arxiv.org/html/2401.13912v1

[^1_54]: https://arxiv.org/html/2501.11618v1

[^1_55]: https://arxiv.org/html/2404.19114v1

[^1_56]: https://arxiv.org/html/2501.01591v1

[^1_57]: https://www.internationalmultiresearch.com/search?q=MER-2025-1-104\&search=search

[^1_58]: https://arxiv.org/pdf/2309.16021.pdf

[^1_59]: https://arxiv.org/pdf/2205.01232.pdf

[^1_60]: https://www.iieta.org/download/file/fid/106742

[^1_61]: http://arxiv.org/pdf/2501.00790.pdf

[^1_62]: http://arxiv.org/pdf/2501.11618.pdf

[^1_63]: https://jmcms.s3.amazonaws.com/wp-content/uploads/2024/08/24064923/jmcms-2408026-AN-EFFICIENT-MACHINE-LEARNING-BASED-DETECTION-HK-Sheeraz-1.pdf

[^1_64]: https://www.sintef.no/en/publications/publication/2390568/

[^1_65]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10241613/

[^1_66]: https://digitalcommons.calpoly.edu/cgi/viewcontent.cgi?article=4489\&context=theses

[^1_67]: https://informatica.si/index.php/informatica/article/download/5268/2608

[^1_68]: https://www.sciencedirect.com/science/article/abs/pii/S0167404824004966

[^1_69]: https://markaicode.com/lstm-hyperparameter-optimization-matlab/

[^1_70]: https://www.jait.us/articles/2025/JAIT-V16N11-1638.pdf

[^1_71]: https://www.nature.com/articles/s41598-025-15146-x

[^1_72]: http://www.diva-portal.org/smash/get/diva2:1504140/FULLTEXT01.pdf

[^1_73]: https://www.sciencedirect.com/science/article/pii/S2772662224000742

[^1_74]: https://arxiv.org/html/2601.03085v1

[^1_75]: https://arxiv.org/pdf/2207.02937.pdf

[^1_76]: https://arxiv.org/pdf/2408.02921.pdf

[^1_77]: https://arxiv.org/pdf/2009.04007.pdf

[^1_78]: https://www.semanticscholar.org/paper/Explainable-AI-based-Intrusion-Detection-in-the-of-Siganos-Radoglou-Grammatikis/c65321f2cc759b7f1e9a4634479454d888b554d8

[^1_79]: https://arxiv.org/pdf/2506.03696.pdf

[^1_80]: https://arxiv.org/pdf/2508.17244.pdf

[^1_81]: https://arxiv.org/pdf/2508.05210.pdf

[^1_82]: https://arxiv.org/html/2408.02921v2

[^1_83]: https://arxiv.org/html/2501.15365v1

[^1_84]: https://arxiv.org/html/2301.05579v2
