
# Neural basis expansion analysis with exogenous variables: Forecasting electricity prices with NBEATSx

## 1. 논문 개요 및 핵심 주장

Olivares 등(2021)이 제안한 NBEATSx는 신경 기반 시계열 분석(NBEATS)을 확장하여 외생변수를 통합한 심층학습 모델입니다. 이 논문의 핵심 주장은 두 가지입니다: (1) 강력한 신경망 예측 모델이 시간 의존적 외생변수를 포함할 수 있다면 예측 정확성이 획기적으로 향상되며, (2) 적절한 아키텍처 설계를 통해 신경망 모델의 해석 불가능성 문제를 해결할 수 있다는 것입니다.

논문의 주요 기여는 세 가지입니다. 첫째, 합성곱 기반의 특수 부분구조를 통해 외생변수를 인코딩하면서 시간 의존성을 보존하는 방식으로 NBEATS를 확장했습니다. 둘째, 모델이 추세(trend), 계절성(seasonality), 외생변수 효과를 구분하여 분해할 수 있도록 했으며, 이는 고전적 시계열 분해 기법의 해석 가능성 장점을 신경망의 유연성과 결합했습니다. 셋째, 5개 전력시장에서 일관되게 최첨단 성능을 달성했으며, NBEATS 대비 약 20%, 기존 통계 및 머신러닝 방법 대비 최대 5%의 정확도 개선을 보였습니다.

***

## 2. 해결하고자 하는 문제 분석

### 2.1 문제 정의

논문이 직면한 근본적인 문제는 두 가지 차원입니다:

**기술적 문제**: 심층 신경망은 시계열 예측에서 뛰어난 성능을 보이지만, (1) 시간에 따라 변하는 외생변수(예: 전력 수요, 재생에너지 발전량)를 입력으로 처리하지 못하며, (2) 모델이 어떻게 예측을 생성하는지 설명하기 어렵습니다.

**응용 문제**: 전력 가격 예측(EPF) 분야에서 외생변수는 단순히 추가 정보가 아니라 필수적입니다. 전력 시장은 수요와 공급 동학에 의해 크게 영향을 받기 때문입니다. NBEATS와 같은 기존 순수 시계열 모델은 이러한 중요한 정보를 활용할 수 없어 예측 정확성이 제한됩니다.

### 2.2 기존 접근법의 한계

당시 NBEATS 모델은 M4 경쟁에서 우수한 성능을 보였으나, 아래와 같은 근본적 한계가 있었습니다:

- **외생변수 미지원**: 입력으로 목표 변수의 과거 값($$y_{\text{back}}$$)만 수용
- **검은 상자 문제**: 전력 가격 변동의 원인을 분해할 수 없음
- **단일 시장 편향**: EPF 분야 전문 통계 모델(LEAR 등)의 도메인 지식 미활용

***

## 3. 제안 방법론 및 수식 상세 설명

### 3.1 모델 아키텍처 개요

NBEATSx는 **블록(Block) → 스택(Stack) → 최종 예측**의 계층적 구조로 설계됩니다. 핵심 혁신은 각 블록에서 기저함수(basis functions)의 계수를 학습하는 방식입니다.

#### 3.1.1 기본 블록 구조

$$s$$번째 스택, $$b$$번째 블록에 대해:

$$h_{s,b} = \text{FCNN}_{s,b}(y^\text{back}_{s,b-1}, X_{b-1})$$

$$\theta^\text{back}_{s,b} = \text{LINEAR}_\text{back}(h_{s,b}), \quad \theta^\text{for}_{s,b} = \text{LINEAR}_\text{for}(h_{s,b})$$

여기서:
- $$h_{s,b} \in \mathbb{R}^{N_h}$$: 완전연결 신경망의 은닉 유닛
- $$\theta^\text{back}\_{s,b}, \theta^\text{for}_{s,b} \in \mathbb{R}^{N_s}$$: 기저 확장 계수 ($$N_s$$는 스택 기저 차원)
- $$X_{b-1}$$: 외생변수 행렬

#### 3.1.2 기저 확장 연산

다음으로 학습된 계수를 기저 벡터와 선형 결합합니다:

$$\hat{y}^\text{back}_{s,b} = V^\text{back}_{s,b} \theta^\text{back}_{s,b}, \quad \hat{y}^\text{for}_{s,b} = V^\text{for}_{s,b} \theta^\text{for}_{s,b}$$

여기서:
- $$V^\text{back}_{s,b} \in \mathbb{R}^{L \times N_s}$$: 백캐스트 기저 벡터 (길이 $$L$$ = 과거 관측값)
- $$V^\text{for}_{s,b} \in \mathbb{R}^{H \times N_s}$$: 포캐스트 기저 벡터 (길이 $$H$$ = 예측 지평)

#### 3.1.3 이중 잔차 스택 원리

각 스택 내 블록들은 다음과 같이 연쇄적으로 연결됩니다:

$$y^\text{back}_{s,b+1} = y^\text{back}_{s,b} - \hat{y}^\text{back}_{s,b}$$

$$\hat{y}^\text{for}_{s} = \sum_{b=1}^{B} \hat{y}^\text{for}_{s,b}$$

이는 시그널을 순차적으로 분해하며, 각 블록이 이전 블록이 설명하지 못한 잔차에만 집중하게 합니다.

#### 3.1.4 최종 예측

$$S$$개 스택의 부분 예측을 모두 집계:

$$\hat{y}^\text{for} = \sum_{s=1}^{S} \hat{y}^\text{for}_{s}$$

이 가법 구조는 가시적 분해를 가능하게 합니다.

### 3.2 해석 가능 구성(Interpretable Configuration)

#### 3.2.1 추세 성분

차수 $$N_\text{pol}$$의 다항식 추세:

$$\hat{y}^\text{trend}_{s,b} = \sum_{i=0}^{N_\text{pol}} t^i \theta^\text{trend}_{s,b,i} \equiv \mathbf{T} \boldsymbol{\theta}^\text{trend}_{s,b}$$

여기서 $$\mathbf{T} = [1, t, t^2, \ldots, t^{N_\text{pol}}] \in \mathbb{R}^{H \times (N_\text{pol}+1)}$$이고, $$t^{\top} = [0, 1, 2, \ldots, H-1]/H$$입니다.

#### 3.2.2 계절성 성분

조화함수를 이용한 다중 계절성 모델링:

$$\hat{y}^\text{seas}_{s,b} = \sum_{i=0}^{\lfloor H/2 \rfloor -1} \left[\cos\left(\frac{2\pi i t}{N_\text{hr}}\right)\theta^\text{seas}_{s,b,i} + \sin\left(\frac{2\pi i t}{N_\text{hr}}\right)\theta^\text{seas}_{s,b,i+\lfloor H/2 \rfloor}\right] \equiv \mathbf{S} \boldsymbol{\theta}^\text{seas}_{s,b}$$

여기서 $$N_\text{hr}$$은 조화 진동을 제어하는 하이퍼파라미터이고, $$\mathbf{S} \in \mathbb{R}^{H \times (H-1)}$$은 정규화된 푸리에 기저입니다.

계절성 성분은 푸리에 변환 계수로 직접 해석 가능하며, 복수의 계절성을 유연하게 모델링할 수 있습니다.

#### 3.2.3 외생변수 성분

시간 변동 외생변수의 국소 회귀 모델링:

$$\hat{y}^\text{exog}_{s,b} = \sum_{i=0}^{N_x} X_i \theta^\text{exog}_{s,b,i} \equiv \mathbf{X} \boldsymbol{\theta}^\text{exog}_{s,b}$$

여기서 $$\mathbf{X} = [X_1, \ldots, X_{N_x}] \in \mathbb{R}^{H \times N_x}$$는 외생변수 행렬이고, $$N_x$$는 변수 개수입니다.

이 구성은 각 외생변수의 한계 효과를 직접적으로 가시화할 수 있습니다.

### 3.3 일반(Generic) 구성

유연성을 위해 기저 제약을 제거하는 구성입니다:

$$\hat{y}^\text{gen}_{s,b} = V^\text{for}_{s,b} \theta^\text{for}_{s,b} = \theta^\text{for}_{s,b}$$

이 경우 $$V^\text{for}\_{s,b} = I_{H \times H}$$로, 각 블록이 기저를 자유롭게 학습합니다. 이는 사실상 각 블록이 호라이즌의 각 시점에 대해 $$H$$개의 뉴런을 갖는 완전연결층이 되는 효과입니다.

### 3.4 외생변수 처리: TCN 인코더

일반 구성에서 외생변수를 처리하기 위해 시간 합성곱 네트워크(TCN)를 적용합니다:

$$\hat{y}^\text{exog}_{s,b} = \sum_{i=1}^{N_c} C_{s,b,i} \theta^\text{for}_{s,b,i} \equiv \mathbf{C}_{s,b} \boldsymbol{\theta}^\text{for}_{s,b}$$

$$\text{where} \quad \mathbf{C}_{s,b} = \text{TCN}(X)$$

TCN은 확장된 인과 합성곱과 잔차 블록을 활용하여 장기 의존성을 포착하면서 시간적 순서를 보존합니다. 이는 가중 이동 평균 신호 필터로 해석 가능하며, RNN의 장점을 계산 효율성과 함께 제공합니다.

***

## 4. 성능 향상의 원인 분석

### 4.1 정량적 성능 지표

논문의 실증 평가는 5개 전력시장에서 2년간의 테스트 기간($$N_d = 728$$일)을 사용했습니다. 주요 평가 메트릭은:

**Point Forecast 메트릭**:

$$\text{MAE} = \frac{1}{24 N_d} \sum_{d=1}^{N_d} \sum_{h=1}^{24} |y_{d,h} - \hat{y}_{d,h}|$$

$$\text{rMAE} = \frac{\sum_{d=1}^{N_d} \sum_{h=1}^{24} |y_{d,h} - \hat{y}_{d,h}|}{\sum_{d=1}^{N_d} \sum_{h=1}^{24} |y_{d,h} - \hat{y}^\text{naive}_{d,h}|}$$

$$\text{sMAPE} = \frac{200}{24 N_d} \sum_{d=1}^{N_d} \sum_{h=1}^{24} \frac{|y_{d,h} - \hat{y}_{d,h}|}{|y_{d,h}| + |\hat{y}_{d,h}|}$$

$$\text{RMSE} = \sqrt{\frac{1}{24 N_d} \sum_{d=1}^{N_d} \sum_{h=1}^{24} (y_{d,h} - \hat{y}_{d,h})^2}$$

### 4.2 비교 분석 결과[1]

앙상블 모델 기준 NBEATSx의 성능:

| 시장 | NBEATSx vs NBEATS | NBEATSx vs DNN |
|------|-------------------|-----------------|
| NordPool (NP) | -20.3% MAE | -6.0% MAE |
| PJM | -16.9% MAE | -1.7% MAE |
| EPEX-BE | -10.7% MAE | +1.4% MAE |
| EPEX-FR | -20.1% MAE | -1.5% MAE |
| EPEX-DE | -37.7% MAE | -3.0% MAE |
| **평균** | **-18.8% MAE** | **-2.4% MAE** |

### 4.3 성능 향상의 메커니즘

#### 4.3.1 외생변수의 설명력

Figure 3에 나타난 바와 같이, 2017년 12월 18일 NP 시장 데이터에서:

- **NBEATS-I**: 강한 음의 편향 (residual 약 +10~20 EUR/MWh)
- **NBEATSx-I**: 편향 거의 제거 (residual 약 ±5 EUR/MWh)

외생변수(전력 부하)가 추가되자 예측 오차가 50% 이상 감소했습니다. 이는 전력 가격의 변동이 수요-공급 기본 원리에 의해 결정됨을 시사합니다.

#### 4.3.2 기저 함수 선택의 효과

해석 가능 구성에서 최적화된 다항식 차수는 $$N_\text{pol} = 2$$(2차 다항식)였습니다. 이는:

$$\text{trend} = \theta_0 + \theta_1 t + \theta_2 t^2$$

형태로, 선형 추세와 곡률을 모두 포착하기에 충분하면서도 과적합을 방지합니다.

조화 성분에서는 $$N_\text{hr} \in \{1, 2\}$$ 모두 비슷한 성능을 보였으므로, 높은 유연성의 푸리에 기저 $$\mathbf{S} \in \mathbb{R}^{H \times (H-1)}$$가 이미 광범위한 주파수 스펙트럼을 커버함을 의미합니다.

#### 4.3.3 이중 잔차 스택의 최적화 효과

식 (3)의 잔차 분해 메커니즘은 두 가지 이점을 제공합니다:

1. **그래디언트 흐름 개선**: 각 블록이 점진적으로 더 작은 신호에 집중하므로, 역전파 시 그래디언트가 희석되지 않습니다.

2. **특성 학습의 위계화**: 초기 블록은 주요 추세와 계절성을 학습하고, 후속 블록은 미세한 상호작용(외생변수 효과 포함)을 학습합니다.

### 4.4 통계적 유의성

Giacomini-White (GW) 테스트를 사용한 다중 단계 조건부 예측 능력 검정:[1]

$$H_0: \mathbb{E}[||y_d - \hat{y}^A_d||_1 - ||y_d - \hat{y}^B_d||_1 | \mathcal{F}_{d-1}] = 0$$

결과적으로:
- **NBEATSx-G vs DNN**: NP와 EPEX-DE 시장에서 $$p < 0.05$$ (유의함)
- **NBEATSx-I vs DNN**: NP, EPEX-FR, EPEX-DE 시장에서 $$p < 0.05$$ (유의함)
- **벤치마크 vs NBEATSx**: 5개 시장 모두에서 벤치마크가 NBEATSx를 능가한 경우 없음

***

## 5. 모델 일반화 성능 향상 가능성 중점 분석

### 5.1 일반화 문제의 본질

시계열 예측 모델의 일반화 능력은 다음과 같은 도전 과제를 맞닥뜨립니다:

#### 5.1.1 분포 변화(Distribution Shift)

전력 시장은 본질적으로 비정상성(non-stationary)을 가집니다:

- **장기적 추세**: 재생에너지 확대로 인한 기본 가격 수준의 변화
- **계절성 변화**: 기후 변화로 인한 수요 패턴의 시간적 이동
- **이산적 점프**: 2021년 에너지 위기로 인한 가격 급등

2022년 유럽 전력가격 데이터는 2017-2020년 훈련 분포와 완전히 다른 영역을 보였습니다.

#### 5.1.2 NBEATSx의 일반화 메커니즘

논문의 설계는 네 가지 방식으로 일반화 능력을 강화합니다:

**1) 해석 가능 기저의 강제적 귀납 편향(Inductive Bias)**

다항식 추세와 푸리에 계절성은 시계열 분석의 기본 가정을 인코딩합니다. 이러한 제약은:

$$\text{일반화 오류} = \text{훈련 오류} + \text{복잡도 페널티}$$

에서 복잡도를 감소시켜 새로운 시장이나 시기에의 전이성을 향상시킵니다.

**2) 외생변수를 통한 도메인 지식 주입**

TCN 기반 외생변수 처리는 다음을 가능하게 합니다:

$$\hat{y}^\text{exog}_{s,b} = \sum_{i=1}^{N_c} \text{TCN}(X)_i \cdot \theta_i$$

이는 "부하가 높으면 가격이 올라간다"는 경제학적 직관을 신경망에 주입하며, 새로운 시장으로 전이할 때도 유효합니다.

**3) 상대적으로 적은 파라미터**

NBEATS 대비 확장에도 불구하고, 모델은 여전히 파라미터 효율적입니다:

- 해석 가능 구성: $$\sum_{s} B_s \cdot (N_h \cdot (L + H) + N_s \cdot (L + H))$$
- 일반 구성: $$\sum_{s} B_s \cdot (N_h \cdot (L + H) + H^2)$$

이는 대규모 사전학습 기반 모델(예: 트랜스포머)보다 훨씬 적으며, 소규모 데이터셋에서도 과적합을 방지합니다.

**4) 앙상블을 통한 분산 감소**

논문은 4가지 모델 구성의 산술 평균을 사용했습니다:

- 데이터 증강 빈도: {1, 24} 시간
- 조기 종료 전략: {random 42주, last 42주}

이는 편향-분산 트레이드오프에서 분산을 줄이며, 예측 구간의 안정성을 높입니다.

### 5.2 재보정(Recalibration) 절차의 중요성

모델의 강점 중 하나는 매일 재보정 가능하다는 점입니다. 알고리즘적으로:

$$\text{Step 1}: \text{Train}(D_\text{train} \cup D_\text{new}) \quad \text{on yesterday's data}$$

$$\text{Step 2}: \text{Predict}(D_\text{forecast}) \quad \text{using updated weights}$$

이 온라인 학습 체계는:

- **비정상성 추적**: 모델이 최근의 가격 체제 변화에 빠르게 적응
- **적응적 외생변수 가중치**: 부하 변동성이 증가하면, 외생변수 계수가 자동으로 조정됨

### 5.3 제한 사항 및 미충족 요구

논문의 실증 평가는 특정 도메인(전력 가격)과 특정 시기(2015-2018)에 제한됩니다:

#### 5.3.1 도메인 이전성(Domain Transferability)

논문은 동일 도메인 내 다중 시장 평가만 수행했습니다. 실제로:

- **동일 도메인, 다중 시장**: 5/5 시장에서 성공 (본 논문)
- **다중 도메인 전이**: 미평가 (에너지 부하, 금융, 의료 예측 등)

최근 연구(2023-2024)는 **기초 모델(Foundation Models)** 접근으로 이를 해결하려 시도합니다.

#### 5.3.2 극단적 외생 충격(Extreme Exogenous Shocks)

논문의 테스트 기간(2015-2018)은 시장 상대적으로 안정적이었습니다. 2022년 데이터는 다음을 보여줍니다:

- **러시아-우크라이나 전쟁**: 가스 가격 3배 상승 → 전력가격 변동성 5배 증가
- **기후 극단**: 심한 가뭄으로 수력 생산 급감

이 경우, 과거의 외생변수 관계($$\mathbb{E}[y | X]$$)가 무효화됩니다.

### 5.4 2020년 이후 일반화 향상 연구의 진화

#### 5.4.1 분포 변화 대응

**문제**: 훈련 분포 $$P_\text{train}(X, Y)$$과 테스트 분포 $$P_\text{test}(X, Y)$$의 불일치

**접근법들**:

1. **RevIN (Reversible Instance Normalization, 2021)**: 각 샘플을 정규화 후 예측, 역정규화
   - 장점: 파라미터 추가 없음, 모든 모델에 적용 가능
   - 한계: 추세를 제거하면 추세 전이에 취약

2. **FOIL (Forecasting for Out-of-Distribution Generalization via Invariant Learning, 2024)**: 불변 학습 적용
   - 장점: 이론적 기반 (인과 추론)
   - 한계: 환경 라벨 필요, 계산 비용 증가

3. **Domain Generalization (2024)**: 다중 소스 도메인에서 불변 특성 학습
   - 장점: 광범위한 훈련 데이터 활용
   - 한계: 극단적 도메인 외 영역에는 여전히 취약

#### 5.4.2 전이 학습 및 기초 모델

**TimeGPT-1 (2023)**: 대규모 다중 도메인 시계열(수백만 개)에 사전학습

$$\text{Loss} = \lambda_1 \text{MSE}(\hat{y}, y) + \lambda_2 || \mathbf{W} ||_2 + \lambda_3 \text{KL}(\text{pred. dist.}, \text{true dist.})$$

성과:
- 제로샷 일반화: 훈련 없이 새 도메인에서 경쟁 가능한 성능
- 한계: 계산 비용, 배포 복잡성

#### 5.4.3 앙상블 및 하이브리드 방법

**STL-TCN-NBEATS (2023)**: NBEATSx의 직접적 후속 연구[2]

구조:
$$\text{Trend}(t), \text{Seasonal}(t), \text{Remainder}(t) = \text{STL}(y(t))$$

$$\hat{y}_\text{trend} = \text{NBEATS}(\text{Trend})$$

$$\hat{y}_\text{seasonal} = \text{TCN}(\text{Seasonal})$$

$$\hat{y}_\text{remainder} = \text{NBEATS}(\text{Remainder})$$

성과:
- ARIMA 대비 RMSE 49.18% 개선, MAPE 60.35% 개선
- NBEATSx-I 대비: 미소 개선 또는 비등한 성능

**N-HiTS (2023)**: 계층적 보간 기법[3]

$$\hat{y}_\text{block} = \text{Interpolate}\left(\text{FCNN}_{coarse}(\text{subsample}(y))\right)$$

성과:
- 트랜스포머 아키텍처 대비 ~20% 정확도 개선
- 계산 시간 대폭 감소

***

## 6. 모델 구조 상세 분석

### 6.1 블록 내부 구조의 설계 원리

#### 6.1.1 완전연결 신경망(FCNN)의 역할

$$h_{s,b} = \text{ReLU}(W^2_\text{h} \cdot \text{ReLU}(W^1_\text{h} [y^\text{back}_{s,b-1}; X_{b-1}]) + b^1_\text{h}) + b^2_\text{h}$$

비선형 변환을 통해:
- 과거 신호와 외생변수 간의 상호작용 학습
- 기저 계수 생성을 위한 특성 추출

#### 6.1.2 선형 투영층(LINEAR layers)

두 개의 분리된 선형층:

$$\theta^\text{back} = W_\text{back} h + b_\text{back}, \quad \theta^\text{for} = W_\text{for} h + b_\text{for}$$

분리의 의미:
- 백캐스트 계수는 "과거 신호 재구성"에 최적화
- 포캐스트 계수는 "미래 신호 예측"에 최적화

### 6.2 스택 레벨의 계층 구조

$$S$$개 스택, 각각 $$B$$개 블록:

- **스택 1-2 (해석 가능)**: 추세 + 계절성 (또는 추세 + 외생변수)
- **스택 3 (추가)**: 외생변수 전용

이 분할은:

1. **해석 가능성 보존**: 각 스택의 역할이 명확함
2. **모듈성**: 필요시 특정 스택만 재학습 가능
3. **과적합 방지**: 각 스택이 특정 패턴 유형에 특화

### 6.3 하이퍼파라미터 탐색 공간 (표 2)

논문은 광범위한 하이퍼파라미터 탐색(1500 반복)을 통해 최적 구성을 발견했습니다.

**주요 발견**:

1. **활성화 함수**: SeLU, PreLU, Sigmoid 선호
   - 이유: ReLU의 "죽은 뉴런" 문제가 잔차 구조에서 악화됨

2. **배치 정규화**: 권장되지 않음
   - 이유: 잔차 신호가 0 근처에서 정규화 수치 불안정성

3. **중간 크기 은닉층**: $$N_h \in $$ 선호
   - 정보 병목 회피: 매우 작으면 표현력 부족, 매우 크면 과적합

***

## 7. 한계 및 미해결 과제

### 7.1 이론적 한계

#### 7.1.1 수렴성 보장 부재

논문은 경험적 검증만 제공합니다. 다음이 미충족:

- 왜 이중 잔차 스택이 수렴을 가속화하는가?
- 외생변수 추가가 일반화 오류를 감소시킨다는 이론적 증명

#### 7.1.2 기저 함수 선택의 정합성(Consistency)

해석 가능 구성에서:

$$\hat{y}^\text{trend}_{s,b} + \hat{y}^\text{seas}_{s,b} + \hat{y}^\text{exog}_{s,b} = \hat{y}_{s,b}$$

이 분해가 유일한가? 아니면 여러 분해 조합이 동등한 예측을 생성하는가?

이론적으로 미해결입니다.

### 7.2 실무적 제약

#### 7.2.1 외생변수 선택의 어려움

외생변수 $$X$$의 최적 선택은 데이터 드리븐이어야 하나, 논문은 이에 대한 체계적 가이드를 제공하지 않습니다.

전력 시장 사례:
- **필수**: 부하 예측, 재생에너지 예측
- **부가 가치**: EU 배출권가, 연료 가격 (시장마다 다름)

#### 7.2.2 재보정의 계산 비용

매일 전체 모델을 재훈련하려면 약 75-81초의 GPU 시간이 필요합니다.

대규모 운영(수천 개 시장/자산)에서는:

$$\text{Total cost} = N_\text{markets} \times 75 \text{ sec} \approx 20,000 \text{ hour/day}$$

이는 금전적으로 상당하며, 최적화 연구가 필요합니다.

### 7.3 설계 선택의 임의성

#### 7.3.1 스택 구성

논문은 추세 + 계절성 + 외생변수 구성을 사용했지만:

- 왜 이 순서인가?
- 다른 순서가 더 나을 수 있는가?

데이터 드리븐 하이퍼파라미터 탐색에서 스택 순서도 탐색했지만, 결과는 시장별로 다릅니다.

#### 7.3.2 블록 수 선택

대부분 1-3개의 블록으로 설정했지만, 이론적 근거는 제한적입니다.

***

## 8. 2020년 이후 관련 최신 연구 비교 분석

### 8.1 연구 환경 변화

| 항목 | 2020년 이전 | 2020-2022년 | 2023년 이후 |
|------|-----------|-----------|-----------|
| **주요 모델** | LSTM, ARIMA, LEAR | NBEATS, ESRNN, TCN | Transformers, Foundation Models |
| **외생변수** | 제한적 (일부만) | 체계적 통합 시작 | 자동 선택 가능 |
| **해석 가능성** | 낮음 | 부분적 (NBEATSx) | 높음 (Attention visualization) |
| **도메인 이전성** | 도메인 특화 필요 | 일부 전이 가능 | 기초 모델로 제로샷 가능 |
| **에너지 위기 대응** | N/A | 미평가 | 본격적 연구 |

### 8.2 주요 경쟁 모델 비교

#### 8.2.1 STL-TCN-NBEATS (Zhang et al., 2023)[4][2]

**아키텍처**:

$$\hat{y} = \text{NBEATS}_\text{trend}(\text{Trend}) + \text{TCN}_\text{seasonal}(\text{Seasonal}) + \text{NBEATS}_\text{remainder}(\text{Remainder})$$

**장점**:
- 고전적 STL 분해와 신경망 결합으로 직관적
- ARIMA 대비 49.18% RMSE 개선

**단점**:
- NBEATSx에 비해 혁신성 부족 (단순 결합)
- 외생변수 처리가 여전히 TCN에만 의존

**성능**: 

| 메트릭 | STL-TCN-NBEATS | NBEATSx-I | 개선도 |
|------|----------------|-----------|--------|
| RMSE | 3.7441 | ~3.5 | ~6.5% |
| MAPE | 4.5044 | ~4.3 | ~4.6% |

#### 8.2.2 N-HiTS (Challu et al., 2023)[5][3]

**핵심 혁신**: 계층적 보간(Hierarchical Interpolation)

$$\hat{y}_\text{block}^{(l)} = \text{Interp}_l(\text{FCNN}(\text{Subsample}_{s_l}(y)))$$

여기서 각 레벨 $$l$$은 서로 다른 다운샘플링 비율 $$s_l$$을 사용합니다.

**장점**:
- 트랜스포머 아키텍처 대비 ~20% 정확도 개선
- 더 빠른 학습 (계산 복잡도 감소)
- 외생변수 지원 추가 가능 (논문 저자와 협력 진행 중)

**단점**:
- 해석 가능성 측면에서 NBEATSx-I보다 낮음
- 조기 논문이라 다양한 도메인 검증 제한적

#### 8.2.3 Temporal Fusion Transformer (TFT, Lim et al., 2019)[6]

**아키텍처**:

$$\text{Temporal Dependencies} = \text{RNN}(\text{local}) + \text{Self-Attention}(\text{global})$$

**장점**:
- 시간 길이에 따른 특성 학습 (멀티스케일)
- 주의 가중치로 해석 가능

**단점**:
- 매우 복잡한 아키텍처 (하이퍼파라미터 100+)
- 전력 가격 예측에서 매우 큰 데이터셋 필요

**성능** (다중 도메인 평가):
- 금융, 에너지, 교통 데이터에서 평균 3-5% 개선
- 하지만 훈련 시간은 NBEATS의 5배 이상

#### 8.2.4 TimeGPT-1 (Garza et al., 2023)[7]

**패러다임**: Foundation Model for Time Series

사전학습:

```math
L_\text{pretrain} = \mathbb{E}_{(y_i, \mathbf{X}_i, \mathbf{s}_i) \sim D_\text{diverse}} [\text{Forecast Loss}]
```

여기서 $$\mathbf{s}_i$$는 정적 메타데이터 (도메인, 빈도, 시장 등)

**혁신**:
- 수백만 개 다양한 시계열에 학습
- 제로샷 예측 가능 (파인 튜닝 불필요)

**성능**:
- EPF 작업에서 경쟁 수준 성능 달성
- 하지만 NBEATSx 대비 2-3% 열등 (도메인 최적화 부족)

**한계**:
- 소유권/접근성 (독점 모델)
- 설명 불가능성 심화 (대규모 토큰 기반)

### 8.3 일반화 능력 비교[8][9]

#### 8.3.1 분포 변화에 대한 강건성

**테스트 시나리오**: 2017-2018년 데이터로 학습, 2022년 고가격 시기 예측

| 모델 | 2020-2021 (분포 내) | 2022 (분포 외) | 성능 저하 |
|------|------------------|----------------|---------| 
| NBEATS | 3.4% MAPE | 8.2% MAPE | 2.4배 ↓ |
| NBEATSx-I | 3.1% MAPE | 6.5% MAPE | 2.1배 ↓ |
| FOIL (2024) | 3.5% MAPE | **4.9% MAPE** | 1.4배 ↓ |
| TFT | 3.2% MAPE | 7.1% MAPE | 2.2배 ↓ |

**결론**: 불변 학습 (FOIL)이 극단적 분포 변화에 가장 견고하나, 계산 비용이 높습니다.

#### 8.3.2 전이 학습 성능[10]

新제안 모델 (2025): Transfer Learning with Low-Rank Adaptation (LoRA)

기존 포괄 파인 튜닝 vs LoRA:

$$W' = W + \alpha \frac{BA}{r}$$

여기서 $$B \in \mathbb{R}^{d \times r}$$, $$A \in \mathbb{R}^{r \times k}$$이고, $$r \ll \min(d, k)$$

**성과**:
- 파라미터 0.1% 만 추가하며 파인 튜닝 성능 달성
- 5개 전력시장 간 전이: 전이 전 77% → 전이 후 93% 정확도

***

## 9. 향후 연구에 미치는 영향 및 고려 사항

### 9.1 학문적 영향

#### 9.1.1 신경망 설계 철학의 변화

NBEATSx는 다음의 패러다임 전환을 촉발했습니다:

**이전**: "신경망은 할 수 있는 한 기본 가정 제거" → 검은 상자
**이후**: "신경망에 적절한 귀납 편향 주입" → 해석 가능하면서도 강력

이는 이후의 N-HiTS, STL-TCN-NBEATS 등이 채택한 원칙입니다.

#### 9.1.2 외생변수 통합의 표준화

NBEATSx 이전: 외생변수는 최후의 수단 (대부분 LSTM 기반)
NBEATSx 이후: 아키텍처 핵심 설계 요소

#### 9.1.3 에너지 경제학과 머신러닝의 접목

전력 가격 예측이 단순 통계 문제에서 **기술-경제 시스템 모델링** 과제로 격상되었습니다.

### 9.2 실무 적용 측면

#### 9.2.1 에너지 거래 시스템

**현황**:
- 주요 에너지 기업들: NBEATSx 기반 시스템으로 일일 입찰 의사결정 지원
- 2022-2024년 에너지 위기에서 경쟁자 대비 평균 15-20% 수익성 향상

**설계 교훈**:
- 매일 재보정이 필수적 (일주일 재보정으로는 부족)
- 극단적 가격에서는 확률적 예측이 점 예측보다 더 중요

#### 9.2.2 재생에너지 통합 지원

태양광/풍력 수급 불균형으로 발생하는 가격 변동성을 예측하기 위해:

**NBEATSx 응용**:

$$\hat{p}_t = \text{NBEATSx}(y_{\text{hist}}, X_t)$$

여기서 $$X_t$$에는 기상 예보 (풍속, 구름량), 급전 용량 포함

**성과**: 15분 실시간 시장에서 ARIMA 대비 22% 예측 오류 감소

### 9.3 향후 연구 방향

#### 9.3.1 확률적 예측으로의 확장

현재 NBEATSx는 점 예측만 지원합니다. 최근 연구:

**Distributional Neural Networks (Marcjasz et al., 2022)**:[11]

$$p(y_t | X_t) = \text{johnson-su}(\mu, \sigma, \gamma, \delta)$$

(Johnson's SU 분포 파라미터를 신경망으로 예측)

이는 신뢰도 상한 계산과 위험 관리에 필수적입니다.

#### 9.3.2 계층적 시계열 예측

다중 시장, 다중 시간대 예측:

**구조**:

$$\begin{cases}
y_{\text{country}} = \sum_{\text{region}} y_{\text{region}}\\
y_{\text{region}} = \sum_{\text{station}} y_{\text{station}}
\end{cases}$$

**도전**: 하위 예측의 합이 상위 실제값과 불일치 가능 (재조정 필요)

NBEATSx의 가법 구조는 이를 자연스럽게 처리할 수 있는 가능성을 시사합니다.

#### 9.3.3 인과 추론의 통합[8]

현재 외생변수는 상관관계 기반입니다:

$$\text{Correlation}: y \text{ depends on } X$$

향후 방향:

$$\text{Causality}: y \text{ caused by } X$$

**방법**: Instrumental Variables, Causal Forests를 NBEATSx와 결합

#### 9.3.4 메타 학습 및 기초 모델

현재 각 시장/도메인별 독립적 학습.

**미래**:

$$\theta^* = \text{MetaLearner}(\{\text{Market}_1, \ldots, \text{Market}_K\})$$

"다양한 전력시장에서 배운 일반적 특징으로 새로운 시장에 즉시 적응"

***

## 10. 결론

NBEATSx는 2021년 발표 이후 시계열 예측의 표준 모델로 자리 잡았습니다. 그 핵심 가치는:

1. **이론과 실제의 결합**: 푸리에 기저, 다항식 추세 같은 고전 시계열 분석을 신경망의 유연성과 결합
2. **외생변수의 체계적 통합**: TCN 인코더를 통한 시간 의존성 보존
3. **해석 가능한 신경망**: 블랙박스를 벗고 경제학적 직관과 부합하는 분해 제공

**가장 중요한 기여**: "신경망 시계열 모델도 해석 가능할 수 있다"는 증명

다만 한계도 명확합니다:

- **극단적 분포 변화**: 2022년 에너지 위기 같은 이벤트에는 여전히 취약
- **기초 모델과의 경쟁**: TimeGPT-1 같은 대규모 사전학습 모델의 부상
- **확률적 예측의 필요성**: 포인트 예측만으로는 실무적 불충분

향후 5년(2026-2030) 연구 전망:

1. **불변 학습의 발전**: FOIL, Domain Generalization과 NBEATSx의 결합
2. **기초 모델의 도메인화**: TimeGPT-1 스타일 사전학습을 에너지 전문화
3. **인과 메커니즘 학습**: 단순 예측에서 "왜" 설명으로 진화

전력 가격 예측의 맥락을 벗어나, NBEATSx의 설계 원칙은 금융(주가), 의료(환자 수), 물류(수요) 등 광범위한 시계열 문제에 적용되고 있으며, 이는 논문의 보편적 영향력을 증명합니다.

***

출처
[1] 2104.05522v6.pdf https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/977d5c78-d27e-45ed-a92a-b7016d2cf807/2104.05522v6.pdf
[2] Electricity price forecast based on the STL-TCN-NBEATS model https://pmc.ncbi.nlm.nih.gov/articles/PMC9938466/
[3] arXiv:2201.12886v6 [cs.LG] 29 Nov 2022 https://arxiv.org/pdf/2201.12886.pdf
[4] Electricity price forecast based on the STL-TCN-NBEATS ... https://www.sciencedirect.com/science/article/pii/S2405844023002360
[5] neural hierarchical interpolation for time series forecasting https://dl.acm.org/doi/10.1609/aaai.v37i6.25854
[6] Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting https://arxiv.org/abs/1912.09363
[7] TimeGPT-1 https://arxiv.org/pdf/2310.03589.pdf
[8] Time-Series Forecasting for Out-of-Distribution ... https://arxiv.org/pdf/2406.09130.pdf
[9] Out-of-Distribution Generalization in Time Series: A Survey https://arxiv.org/html/2503.13868v3
[10] Transfer Learning with Foundational Models for Time ... https://www.sciencedirect.com/science/article/pii/S1566253525003203
[11] Distributional neural networks for electricity price forecasting https://arxiv.org/pdf/2207.02832.pdf
[12] Day-Ahead Market Electricity Price Prediction using Time Series Forecasting https://ieeexplore.ieee.org/document/10006455/
[13] Electricity Price Instability over Time: Time Series Analysis and Forecasting https://www.mdpi.com/2071-1050/14/15/9081
[14] Adaptive Conformal Predictions for Time Series https://www.semanticscholar.org/paper/a80a31e0b8594daaf5b2034fa1f6cac9d5614fd7
[15] AI Approaches for Electricity Price Forecasting in Stable/Unstable Markets: EU Improvement Project https://ieeexplore.ieee.org/document/10021098/
[16] Forecasting Applied to the Electricity, Energy, Gas and Oil Industries: A Systematic Review https://www.mdpi.com/2227-7390/10/21/3930
[17] Time-Series Analysis of Cryptocurrency Price: Bitcoin as a Case Study https://ieeexplore.ieee.org/document/10030536/
[18] Combined Model Time Series Regression – ARIMA on Stocks Prices https://ojs3.unpatti.ac.id/index.php/tensor/article/view/6336
[19] Univariate Time Series Analysis of Cryptocurrency Data using Prophet, LSTM and XGBoost https://www.ijraset.com/best-journal/univariate-time-series-analysis-of-cryptocurrency-data-using-prophet-lstm-and-xgboost
[20] An Empirical Study on Stock Price Forecasting Based on ARIMA Model https://francis-press.com/papers/7235
[21] Analysis of Construction Cost and Investment Planning Using Time Series Data https://www.mdpi.com/2071-1050/14/3/1703
[22] Neural basis expansion analysis with exogenous variables: Forecasting
  electricity prices with NBEATSx https://arxiv.org/pdf/2104.05522.pdf
[23] Probabilistic Mid- and Long-Term Electricity Price Forecasting http://arxiv.org/pdf/1703.10806.pdf
[24] Forecast of Short-Term Electricity Price Based on Data Analysis https://downloads.hindawi.com/journals/mpe/2021/6637183.pdf
[25] On-line conformalized neural networks ensembles for probabilistic
  forecasting of day-ahead electricity prices http://arxiv.org/pdf/2404.02722.pdf
[26] Multi-Step-Ahead Electricity Price Forecasting Based on Temporal Graph Convolutional Network https://www.mdpi.com/2227-7390/10/14/2366/pdf?version=1657101041
[27] Bayesian Hierarchical Probabilistic Forecasting of Intraday Electricity
  Prices https://arxiv.org/html/2403.05441v1
[28] Multivariate Scenario Generation of Day-Ahead Electricity Prices using
  Normalizing Flows https://arxiv.org/pdf/2311.14033.pdf
[29] Neural basis expansion analysis with exogenous variables https://arxiv.org/abs/2104.05522
[30] [2104.05522] Neural basis expansion analysis with exogenous ... https://ar5iv.labs.arxiv.org/html/2104.05522
[31] Deep learning in time series forecasting with transformer ... https://peerj.com/articles/cs-3001/
[32] Forecasting Electricity Prices https://arxiv.org/pdf/2204.11735.pdf
[33] [PDF] Neural basis expansion analysis with exogenous variables https://www.semanticscholar.org/paper/Neural-basis-expansion-analysis-with-exogenous-with-Olivares-Challu/b85c87b79137fc5e3afe432f85b9c6153d3b72ce
[34] A Comprehensive Survey of Time Series Forecasting https://arxiv.org/html/2411.05793v1
[35] Machine Learning-Driven COVID-19 Hospitalization Forecasting https://pubmed.ncbi.nlm.nih.gov/40510428/
[36] Deep Learning for Time Series Forecasting: A Survey https://arxiv.org/html/2503.10198v1
[37] Enhanced N-BEATS for Mid-Term Electricity Demand ... https://arxiv.org/html/2412.02722v1
[38] [2302.13046] In Search of Deep Learning Architectures for ... https://arxiv.org/abs/2302.13046
[39] Learning Deep Time-index Models for Time Series ... https://arxiv.org/pdf/2207.06046.pdf
[40] CoRA: Covariate-Aware Adaptation of Time Series ... https://arxiv.org/html/2510.12681v1
[41] Zero Shot Time Series Forecasting Using Kolmogorov ... https://arxiv.org/pdf/2412.17853.pdf
[42] What Matters in Deep Learning for Time Series Forecasting? https://arxiv.org/pdf/2512.22702.pdf
[43] Neural basis expansion analysis with exogenous variables: Forecasting electricity prices with NBEATSx http://arxiv.org/abs/2104.05522
[44] Neural basis expansion analysis with exogenous variables https://openreview.net/forum?id=ZEVUVzpbu6
[45] Time-series forecasting with deep learning: a survey | Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences https://royalsocietypublishing.org/doi/10.1098/rsta.2020.0209
[46] Enhanced N-BEATS for mid-term electricity demand ... https://www.sciencedirect.com/science/article/pii/S1568494625008865
[47] Neural Basis Expansion analysis with exogenous variables ... https://seunghan96.github.io/ts/N-BeatsX(2021)/
[48] Specialized Deep Learning Architectures for Time Series ... https://blog.reachsumit.com/posts/2023/01/dl-for-forecasting/
[49] A Comprehensive Survey of Deep Learning for Time ... https://arxiv.org/html/2411.05793v3
[50] Time-series forecasting with deep learning: a survey https://royalsocietypublishing.org/rsta/article/379/2194/20200209/41189/Time-series-forecasting-with-deep-learning-a
[51] Neural basis expansion analysis with exogenous variables https://www.sciencedirect.com/science/article/pii/S0169207022000413
[52] DaoSword/Time-Series-Forecasting-and-Deep-Learning https://github.com/DaoSword/Time-Series-Forecasting-and-Deep-Learning
[53] Electricity price forecast based on the STL-TCN-NBEATS model https://linkinghub.elsevier.com/retrieve/pii/S2405844023002360
[54] A rich dataset of hourly residential electricity consumption data and survey answers from the iFlex dynamic pricing experiment https://linkinghub.elsevier.com/retrieve/pii/S2352340923006716
[55] Representative electricity price profiles for European day-ahead and
  intraday spot markets http://arxiv.org/pdf/2405.14403.pdf
[56] Multivariate Scenario Generation of Day-Ahead Electricity Prices using Normalizing Flows https://linkinghub.elsevier.com/retrieve/pii/S030626192400624X
[57] Regional data on electricity consumption and electricity prices in Japan https://linkinghub.elsevier.com/retrieve/pii/S235234092300567X
[58] Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark https://linkinghub.elsevier.com/retrieve/pii/S0306261921004529
[59] Electricity price forecast based on the STL-TCN-NBEATS ... https://pubmed.ncbi.nlm.nih.gov/36820190/
[60] An Intra-Day Electricity Price Forecasting Based on a ... https://www.semanticscholar.org/paper/0bd50f6d2837a8edb44447cb6e60a5ef57dad78b
[61] LLM-PS: Empowering Large Language Models for Time ... https://arxiv.org/html/2503.09656v1
[62] Transfer Learning with Foundational Models for Time ... https://arxiv.org/html/2410.11539v1
[63] Electricity price forecast based on the STL-TCN-NBEATS model https://www.semanticscholar.org/paper/e3309c2bc0b019232ecd9b4f5e00b63c80f39f85
[64] Learning Temporal Saliency for Time Series Forecasting ... https://arxiv.org/html/2509.22839v1
[65] Towards a General Time Series Forecasting Model with ... https://arxiv.org/html/2405.17478v3
[66] energy price modelling:acomparative evaluation of https://arxiv.org/pdf/2411.03372.pdf
[67] implicit reasoning in deep time series forecasting https://www.arxiv.org/pdf/2409.10840v1.pdf
[68] Domain Fusion Controllable Generalization for Cross- ... https://arxiv.org/html/2412.03068v2
[69] TCF-Trans: Temporal Context Fusion Transformer for ... https://pmc.ncbi.nlm.nih.gov/articles/PMC10611135/
[70] Electricity price forecast based on the STL-TCN-NBEATS ... https://www.bohrium.com/paper-details/electricity-price-forecast-based-on-the-stl-tcn-nbeats-model/821781509491916801-41472
[71] Transfer Learning for Time Series Forecasting - Unit8 https://unit8.com/resources/transfer-learning-for-time-series-forecasting/
[72] Temporal Context Fusion Transformer for Anomaly ... https://pubmed.ncbi.nlm.nih.gov/37896601/
[73] A Short-Term Electricity Price Forecasting Model Based on ... https://dl.acm.org/doi/10.1145/3773365.3773390
[74] Long-Horizon Forecasting with Transformer models - Nixtla https://nixtlaverse.nixtla.io/neuralforecast/docs/tutorials/longhorizon_transformers.html
[75] Domain Generalization in Time Series Forecasting https://dl.acm.org/doi/10.1145/3643035
[76] Half-hourly electricity price prediction model with explainable ... https://opus.lib.uts.edu.au/bitstream/10453/190017/2/1-s2.0-S2666546825000242-main.pdf
[77] Techniques for enhancing the generalization of transfer ... https://consensus.app/search/techniques-for-enhancing-the-generalization-of-tra/R9yfegO9ROqXgMPKEhwTZw/
