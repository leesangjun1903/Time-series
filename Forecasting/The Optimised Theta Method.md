# The Optimised Theta Method

# 1. 논문의 핵심 주장과 주요 기여 (간결 요약)

**핵심 주장**  
“The Optimised Theta Method (OTM)” 논문은 기존 Theta 방법의 가장 큰 제약(사전에 고정된 두 개의 theta 계수)을 제거하고, **두 번째(theta>1) 단기 성분을 데이터 기반으로 최적화**함으로써, 대규모 시계열군에 대해 **더 정확하고 견고한 자동 예측법**을 제공할 수 있다고 주장한다.[1][2]

**주요 기여**

1. **일반화된 Theta 조합식 제시**  
   임의의 두 theta 라인 $\(Z_t(\theta_1), Z_t(\theta_2)\)$ 을 가중합으로 결합하면서도, 원 시계열 $\(y_t\)$ 를 정확히 재구성할 수 있는 **유일한 가중치 공식**을 제시:

$$
   Y_t = \omega Z_t(\theta_1) + (1-\omega)Z_t(\theta_2), \quad 
   \omega(\theta_1,\theta_2) = \frac{\theta_2 - 1}{\theta_2 - \theta_1}, \quad \theta_1 \le 1 \le \theta_2
   $$ 

[1]

2. **Optimised Theta Method(OTM) 정의**  
   장기 성분 $\(\theta_1=0\)$ 을 고정하고, 단기 성분 $\(\theta_2=\theta \ge 1\)$ 를 **검증 구간(out-of-sample) 손실 최소화**를 통해 선택하는 **최적화 기반 Theta 모델**을 제안.[1]

3. **Generalised Rolling Origin Evaluation(GROE)**  
   단일 고정 origin 대신 여러 origin을 굴리며 검증하는 **일반화된 rolling-origin 평가 프레임워크**를 정의하고, OTM의 $\(\theta\)$ 선택에 적용.[1]

4. **대규모 실험(M3 3003개 시계열)**  
   M3-Competition 전체 데이터(3003개)에 대해, 다양한 GROE 스킴과 손실함수(sAPE, AE, SE)를 비교한 결과, OTM이 **원래 Theta, ETS, Damped, ARIMA 등 대부분의 고전 통계 모델을 일관되게 상회**함을 보임.[2][1]

5. **모델 재구성 성질 유지 + 확장성**  
   새로운 OTM은  
   - $\(\theta=2\)$일 때 고전 Theta에 수렴,  
   - $\(\theta=1\)$ 일 때 두 번째 라인 모형(예: SES)을 직접 적용한 것과 같아지는 등,  
   **기존 모형들을 포함하는 상위(super) 프레임워크**임을 명확히 보여준다.[1]

***

# 2. 논문이 다루는 문제, 제안 방법, 모델 구조, 성능 및 한계

## 2.1 해결하고자 하는 문제

기존 Assimakopoulos & Nikolopoulos(2000)의 Theta 방법은 다음과 같은 한계를 갖는다.[1]

1. **Theta 계수 고정**:  
   - 장기 성분: $\(\theta_1 = 0\)$ (직선 회귀)  
   - 단기 성분: $\(\theta_2 = 2\)$ (곡률 2배)  
   이 값들은 **경험적으로 정해진 상수**로, 데이터 특성에 따른 적응이 없다.

2. **결합 가중치 고정 (0.5–0.5)**  
   두 theta 라인의 예측을 항상 50:50으로 결합:

$$
   \hat{y}_{t+h} = 0.5 \hat{Z}_{t+h}^{(0)} + 0.5 \hat{Z}_{t+h}^{(2)}
   $$
   
   이 선택은 원 시계열 재구성을 보장하지만, **예측 정확도 최적화 관점에서 최선이라는 보장은 없다**.[1]

3. **단기/장기 성분 활용의 비효율**  
   Theta 방법은 “국소 곡률 조작”이라는 매우 강력한 아이디어를 가지고 있으나, **어떤 곡률(어떤 $\(\theta\)$ )이 어떤 시계열·예측 지평에 가장 적합한지**에 대한 체계적인 탐색/최적화 절차가 없었다.[3][1]

4. **대규모 시계열군을 위한 자동화 최적화 부재**  
   M3처럼 수천 개의 시계열을 다루는 상황에서, **자동·견고·계산 가능**하면서도 개별 시계열에 맞게 튜닝되는 방법이 필요하다.

논문은 위 문제를 “**Theta 계수 선택과 결합 구조를 이론적으로 일반화하고, 단기 성분 $\(\theta\)$ 를 데이터 기반으로 최적화**하는 문제”로 정식화한다.[1]

***

## 2.2 제안하는 방법 (수식 포함)

### 2.2.1 기본 Theta 라인의 정의

Nikolopoulos et al.(2012)의 결과를 이용해, Theta 라인은 다음과 같이 정의된다.[1]

- 원 시계열: $\(y_t\), \(t = 1,\dots,n\)$
- 선형 회귀 추세: $\(\hat{\alpha} + \hat{\beta} t\)$

Theta 라인:

$$
Z_t(\theta) = \theta y_t + (1-\theta)(\hat{\alpha} + \hat{\beta} t)
= \theta (y_t - \hat{\alpha} - \hat{\beta} t) + \hat{\alpha} + \hat{\beta} t
$$

성질:
- 평균과 기울기는 $\(y_t\)$ 와 동일
- $\(\theta < 1\)$ : **곡률 축소 → 장기 추세 강조**
- $\(\theta > 1\)$ : **곡률 확대 → 단기 변동 강조**[1]

### 2.2.2 일반화된 Theta 조합식

두 개의 theta 라인을 임의의 가중치로 결합:

$$
Y_t = \omega Z_t(\theta_1) + (1-\omega) Z_t(\theta_2), \quad 0 \le \omega \le 1
$$

원 데이터 재구성 조건 $\(Y_t = y_t\)$ ( $\(t=1,\dots,n\)$ ) 을 만족시키는 **유일한 해**는 다음과 같다.[1]

- 제약: $\(\theta_1 \le 1 \le \theta_2\)$
- 가중치:

$$
\omega(\theta_1,\theta_2) = \frac{\theta_2 - 1}{\theta_2 - \theta_1}, \qquad
1-\omega = \frac{1-\theta_1}{\theta_2 - \theta_1}
$$

검증:  
- $\(\theta_1=0,\theta_2=2\)$ 이면 $\(\omega = \frac{2-1}{2-0} = 0.5\)$ , 즉 기존 Theta와 일치.[1]
- 이 구조 아래에서 항상

$$
  \omega Z_t(\theta_1) + (1-\omega) Z_t(\theta_2) = y_t
  $$
  
  가 성립.

### 2.2.3 Optimised Theta Method(OTM)의 특수화

논문은 **장기 성분은 고정, 단기 성분만 최적화**하는 전략을 취한다.[1]

- 장기 성분: $\(\theta_1 = 0 \Rightarrow Z_t(0) = \hat{\alpha} + \hat{\beta} t\)$
- 단기 성분: $\(\theta_2 = \theta \ge 1\)$

이때 식 (4)에서

$$
\omega = \frac{\theta - 1}{\theta - 0} = 1 - \frac{1}{\theta}
$$

따라서 결합식은

$$
Y_t = \left(1 - \frac{1}{\theta}\right) Z_t(0) + \frac{1}{\theta} Z_t(\theta)
= \left(1 - \frac{1}{\theta}\right) (\hat{\alpha} + \hat{\beta} t) + \frac{1}{\theta} Z_t(\theta)
$$

예측 $\(k\)$ 스텝 앞:

$$
\hat{Y}_{t+k|t} =
\left(1 - \frac{1}{\theta}\right) (\hat{\alpha} + \hat{\beta} (t+k)) + \frac{1}{\theta} \hat{Z}_{t+k|t}(\theta)
$$

여기서 $\(\hat{Z}_{t+k|t}(\theta)\)$ 는 보통 **SES(Simple Exponential Smoothing)**으로 외삽한다.[1]

특기할 점:

- $\(\theta = 1\)$ 이면 $\(Z_t(1) = y_t\), \(\omega = 0\)$ , 따라서

$$
  \hat{Y}\_{t+k|t} = \hat{Z}_{t+k|t}(1)
  $$
  
  즉 **두 번째 라인에 쓰는 외삽 방법(예: SES)을 원 시계열에 직접 적용**한 것과 동일 → OTM은 해당 외삽법의 상위 일반화.[1]

- $\(\theta = 2\)$ 이면 기존 Theta와 동일.

### 2.2.4 Generalised Rolling Origin Evaluation (GROE)

$\(\theta\)$ 를 선택하는 핵심은, 단순 1-step ahead in-sample 적합이 아니라, **다양한 origin에서의 다단계 예측 성능을 평균**하는 것이다.[1]

- 전체 길이: $\(n\)$
- 첫 origin: $\(n_1\)$
- origin 이동 간격: $\(m\)$
- 각 origin에서의 예측 길이: $\(H\)$
- origin 개수: $\(p\)$, 최대는

$$
  p_{\max} = 1 + \left\lfloor \frac{n - n_1}{m} \right\rfloor
  $$

GROE 손실함수:

$$
\ell(\theta) =
\sum_{i=1}^{p}
\sum_{j=1}^{\min(H,\, n-n_i)}
g\big( y_{n_i + j}, \hat{Y}_{n_i + j \mid n_i}(\theta) \big)
$$

여기서 $\(g(\cdot)\)$ 는 SE, AE, sAPE 중 하나:

- 제곱오차(SE): $\( (a-b)^2 \)$
- 절대오차(AE): $\( |a-b| \)$
- 대칭 APE(sAPE):

$$
  sAPE(a,b) = \frac{2|a-b|}{|a|+|b|}
  $$

** $\(\theta\)$ 추정 전략**

- $\(\theta \in \Theta = \{1, 1.5, 2, 2.5, \dots, 5\}\)$ 와 같이 **유한 격자**로 제한
- 각 $\(\theta\)$ 에 대해 식 (8)을 계산하여 **brute-force 탐색으로 최소값을 주는 $\(\hat\theta\)$ 선택**
- 이 방식은 **수천 개 시계열에서도 계산 가능**하며, 곡률 변화에 대한 민감도가 완만하기 때문에 격자화의 실효성이 높음.[1]

***

## 2.3 모델 구조 요약

OTM의 전체 파이프라인은 다음과 같다.[1]

1. **Seasonality Test (Step 0)**  
   ACF 기반의 계절성 유의성 검정.

2. **Deseasonalization (Step 1)**  
   계절성이 있으면 **고전적인 곱셈형 분해**로 계절성 제거.

3. ** $\(\theta\)$ 추정 (Step 2)**  
   - 사전 정의한 GROE 설정 $\((n_1, m, H, p)\)$ 과 손실 $\(g\)$ 선택  
   - $\(\theta \in \Theta\)$ 에 대해 식 (8)을 계산, 최소값을 주는 $\(\hat\theta\)$ 선택

4. **Decomposition (Step 3)**  
   - 장기선: $\(Z_t(0) = \hat{\alpha} + \hat{\beta} t\)$ 
   - 단기선: $\(Z_t(\hat{\theta})\)$

5. **Extrapolation (Step 4)**  
   - $\(Z_t(0)\)$ : 선형 추세를 직선으로 외삽  
   - $\(Z_t(\hat{\theta})\)$ : 주로 SES로 외삽 (논문에서 기본 선택)

6. **Combination (Step 5)**  
   - 식 (5)에 따라 두 라인 예측을 결합.

7. **Reseasonalization (Step 6)**  
   - 1단계에서 제거한 계절성 재적용.

구조적으로 보면, OTM은

- (1) **통계적 선형 추세 모델(LR)**  
- (2) **단기 시점용 SES(또는 Holt, Damped 등 교체 가능) 기반 곡률 증폭 시계열**  
- (3) **GROE 기반 교차검증 최적화**  

를 합성한 **“검증 기반 파라미터 선택이 붙은 2-모델 앙상블”**이다.[3][1]

***

## 2.4 성능 향상 및 한계

### 2.4.1 성능 향상 (M3-Competition)

**벤치마크**

- 기존 Theta
- Naïve, Seasonal Naïve
- SES, Holt/Holt-Winters
- Damped trend, ETS(자동 선택), ARIMA(auto.arima)[1]

**평가 지표**

- sMAPE
- MASE  
  (Hyndman & Koehler의 스케일 불변 절대 오차 지표)[1]

**핵심 결과**

1. **전체 3003개 시계열 기준**  
   - 기존 Theta의 sMAPE ≈ 13.09, MASE ≈ 2.19[1]
   - 최선의 OTM 변형(예: sAPE 비용, GROE 접근 (d))의 sMAPE ≈ 12.85, MASE ≈ 2.09 수준으로 **일관된 개선**.[1]

2. **빈도별**  
   - 연간, 분기, 월간 모든 부분집합에서 OTM이 기존 Theta 및 대부분의 벤치마크를 상회  
   - “Other” 빈도(잡다한 비정형 빈도)에서는 Damped/ETS가 근소하게 좋은 경우도 있으나, OTM도 경쟁력 유지.[1]

3. **검증 스킴 비교**  
   - GROE에서 origin을 자주 굴리는 접근 (예: (d): rolling origin, \(m=1\))이 고정 origin보다 **일반적으로 더 좋은 성능**  
   - $\(\,n_1 = n-h\)$ (마지막 h 구간을 validation용으로 남기는 설정)가 \(n-2h\)보다 대부분의 빈도에서 우수.[1]

4. **랭크 기반 통계 검정 (MCB Test)**  
   - 다중 비교(Multiple Comparisons with the Best)에서,  
     OTM의 상위 몇 개 변형(approach (a)–(d))은 서로 통계적으로 유의한 차이가 없고, 모두 Theta보다 **유의하게 우수**.[1]

요약하면, OTM은

- 기존 Theta 대비 **소폭이지만 일관된 예측 정확도 향상**  
- ETS, ARIMA, Damped 등과 비교해도 **가장 견고한 베이스라인 중 하나**임을 실증적으로 보여준다.[3][1]

### 2.4.2 한계

1. **계산 비용 증가**  
   - 각 시계열마다 여러 $\(\theta\)$ 후보 × 여러 origin에 대해 반복적으로 피팅·예측해야 하므로,  
   - 기존 Theta보다 계산량이 상당히 크며, ETS/ARIMA 자동선택과 유사한 수준의 비용.[1]

2. **단일 단기 Theta 라인만 최적화**  
   - 논문에서는 $\(\theta_1=0\)$ 으로 고정, $\(\theta_2=\theta\)$ 하나만 최적화  
   - 다수의 Theta 라인(예: $\(\theta \in \{-1,0,1,2,3\}\)$ )을 동시에 최적화하는 일반화는 남겨둔 상태.[1]

3. **외삽 모형 선택은 고정(SES)**  
   - 주된 실험은 SES 기반 OTM에 집중  
   - Holt, Damped 등으로 교체 시 “Other” 빈도에서 일부 향상은 있으나, 계산 시간 급증.[1]

4. **이론적 일반화 성능 분석의 부족**  
   - $\(\theta\)$ 선택이 검증 기반이라는 점에서 **경험적으로는 과적합 방지에 유리하지만**,  
   - 통계적 학습이론 관점(일반화 오차 bound 등)의 이론 분석은 제공되지 않는다.

***

# 3. 모델의 일반화 성능 향상 가능성 관점에서의 해석

OTM은 여러 측면에서 **일반화 성능(보지 않은 미래 구간에 대한 예측력)을 개선하는 설계를 채택**하고 있다.

## 3.1 곡률 제어를 통한 바이어스–분산 트레이드오프 조절

Theta 라인의 정의

$$
Z_t(\theta) = \theta (y_t - \hat{\alpha} - \hat{\beta} t) + \hat{\alpha} + \hat{\beta} t
$$

을 보면, $\(\theta\)$ 는 **잔차의 곡률을 확대/축소하는 조절자** 역할을 한다.[1]

- $\(\theta \approx 1\)$ : 원래 시계열과 거의 동일 → 분산↑, 바이어스↓  
- $\(\theta \gg 1\)$ : 고주파 변동을 과도하게 키워 **과적합 위험**  
- $\(\theta \to 1^+\)$ : 장기 추세 + 제한된 단기 파형만 반영 → 바이어스와 분산의 균형

OTM은 이 $\(\theta\)$ 를 out-of-sample 검증으로 선택함으로써, **각 시계열에 맞는 바이어스–분산 트레이드오프를 자동으로 조정**한다.

## 3.2 GROE: 다수 origin 기반의 교차검증형 튜닝

일반적인 교차검증과 유사하게, GROE는

- 서로 다른 origin에서
- 고정 길이 $\(H\)$ 의 예측 구간에 대해
- 동일한 $\(\theta\)$ 를 테스트한 후 손실을 평균

함으로써, **특정 지역적 패턴에 대한 과적합을 줄이고, 전체 시계열의 전반적 구조를 잘 반영하는 $\(\theta\)$ 를 선택**한다.[1]

이 점은 기존 Theta가

- $\(\theta=2\)$ 를 모든 시계열·모든 지평에 대해 **전역 고정**했던 것과 대비된다.  
  → OTM은 같은 간단한 구조를 유지하면서도 **데이터와 지평에 맞춰 자동 적응하는 하이퍼파라미터 튜닝**을 수행한다.

## 3.3 구조적 재구성 제약의 정규화 효과

식 (5)에서 보았듯, 어떤 $\(\theta\)$ 를 택하더라도

$$
Y_t = \left(1 - \frac{1}{\theta}\right) (\hat{\alpha}+\hat{\beta}t) + \frac{1}{\theta} Z_t(\theta)
$$

은 **원 데이터와 동일한 추세/평균을 가지며, 전체적으로 $y_t$ 를 정확히 재구성**한다.[1]

이는 모형이

- 임의의 비정형 패턴을 마음대로 fitting하는 대신,
- “선형 추세 + 곡률 조정된 잔차”라는 **강한 구조적 제약** 아래에서만 표현을 허용한다는 의미로,  
- 통계적 학습이론 관점에서 보면 **모델 클래스의 용량을 제한하는 정규화(regularization)**와 유사한 역할을 한다.

따라서, OTM의 자유도는 “ $\(\theta\)$ 하나 + SES 파라미터” 수준으로 낮고, 재구성 제약까지 있으므로, **과적합 위험이 상대적으로 낮으면서도, $\(\theta\)$ 조정을 통해 충분한 유연성**을 확보할 수 있다.[3][1]

## 3.4 경쟁 연구와의 비교 (2020년 이후)

2020년 이후 연구들을 보면, Theta/OTM 계열은 **여전히 강력한 일반화 성능을 가진 베이스라인**으로 자리하고 있으며, 다음과 같은 흐름이 나타난다.

1. **일반화된 자동 Theta (Spiliotis et al., 2020)**  
   - “Generalizing the Theta method for automatic forecasting”에서는 Theta를 **자동 예측 알고리즘으로 일반화**하고, M4 등 대규모 데이터셋에서 여러 설정을 비교.[4]
   - 자동화된 Theta 변형은 **여전히 다양한 데이터셋에서 경쟁력 있는 일반화 성능**을 보여, OTM과 마찬가지로 “데이터 기반 구성요소 선택”의 중요성을 강조.

2. **Wisdom of the Data (Petropoulos & Spiliotis, 2021)**  
   - “The Wisdom of the Data: Getting the Most Out of Univariate Time Series Forecasting”에서는 Theta 계열을 포함한 여러 데이터 변환/조합 전략(곡률 조작, temporal aggregation, bootstrap 등)을 체계적으로 비교.[5]
   - 핵심 메시지: **동일 시계열을 여러 관점(곡률, 집계 수준 등)에서 바라보고 예측을 결합하면 일반화 성능이 향상**된다는 것 → Theta/OTM의 “다른 곡률의 theta 라인을 결합”이라는 아이디어와 맥락을 같이 한다.

3. **데이터 중심 예측(Deja vu, 2020)**  
   - “Déjà vu: A data-centric forecasting approach through time series cross-similarity”에서는 모형 가정 대신 **유사한 시계열의 미래 경로를 집계**하는 방식으로 예측.[6]
   - 통계 모형 불확실성을 회피하면서도, cross-similarity를 통해 일반화 성능을 확보한다는 점에서, **OTM이 ‘모델 구조의 정규화 + 검증 최적화’를 통해 일반화를 도모하는 것과 상보적 접근**이다.

4. **주간 시계열을 위한 앙상블 베이스라인(2020)**  
   - “A Strong Baseline for Weekly Time Series Forecasting”에서는 Theta, TBATS, DHR-ARIMA, RNN을 Lasso 기반 스태킹으로 결합해 **강력한 주간 예측 베이스라인**을 제시한다.[7]
   - 여기서도 Theta는 **여전히 핵심 구성요소**이며, 앙상블 구조 속에서 좋은 일반화 성능을 제공.

5. **소지역 인구 예측(2023)**  
   - Australian SA2 소지역 인구를 예측하는 연구에서, ARIMA, ETS, THETA, LightGBM, XGBoost 및 이들의 앙상블을 비교한 결과, **THETA 계열이 여전히 강력하며, 앙상블 내 중요한 구성원**임이 보고된다.[8]

6. **구조적 Theta (2024)**  
   - “The structural Theta method”는 Theta를 **구조적 시계열 모델(state-space, structural model) 프레임워크 내에서 재해석**하여, 다중 오차 소스를 고려한 변형을 제시하고 M4/M5급 데이터에서 성능 분석을 수행.[9]
   - 이는 OTM이 가졌던 “state space와의 연결”을 더 명확히 하며, **구조적 추론을 통해 일반화 성능 및 불확실성 평가를 동시에 개선**하려는 시도이다.[9][3]

7. **조합 및 견고성 향상 (2020+ 다수 연구)**  
   - forecast 조합/메타-러닝(FFORMA 류) 연구에서, Theta/OTM 계열은 **기본 base learner로 가장 자주 포함**되며,  
   - 2024년 농산물 예측에서 DOTM(Dynamic Optimised Theta Model)을 포함해 ETS, ARIMA, CES를 median 결합(SCUM)한 방법이 개별 모델보다 우수한 결과를 보이는 등, **Theta/OTM이 견고한 “building block”으로 활발히 활용**되고 있다.[10][11][7]

이러한 흐름은, OTM과 그 후속 연구들이

- **간단한 구조와 낮은 자유도**를 유지하면서도,
- **데이터 기반 최적화(θ, 조합 가중치, 구조 선택 등)를 통해 일반화 성능을 높이는 방향**으로 진화하고 있음을 보여준다.

***

# 4. 앞으로의 연구에 대한 영향과 향후 고려 사항

## 4.1 논문의 학문적·실무적 영향

1. **Theta 방법의 이론적 “정규형” 제공**  
   - 단순한 경험적 기법이었던 Theta를  
     - 일반화된 조합식(식 (3),(4))  
     - 검증 기반 파라미터 선택(식 (8))  
     로 재정의함으로써, 이후 연구(동적 OTM, 구조적 Theta, 자동 Theta 등)의 **이론적 토대**를 제공하였다.[4][9][3][1]

2. **대규모 예측 시스템의 강력한 베이스라인**  
   - M3에서 입증된 성능 + OTM의 추가 개선으로,  
   - M4/M5, 산업용 forecasting 시스템에서 Theta/OTM은 **“버릴 수 없는 최소 기준선(baseline)”**으로 자리 잡았고, deep learning 모델과 비교 연구에서도 여전히 강력한 비교군으로 사용된다.[12][5][7][4]

3. **“데이터에서 정보 추출 후 결합”이라는 패러다임 강화**  
   - Theta/OTM, temporal aggregation, bootstrap 등 다양한 변환 기반 예측들이 “Wisdom of the Data” 프레임에서 통합적으로 논의되고 있으며, 이는 **단일 복잡 모델 대신, 여러 단순 관점의 조합으로 일반화를 얻는 패턴**을 강화했다.[5][4]

## 4.2 앞으로 연구 시 고려할 점

OTM/Theta 계열을 확장·활용하는 미래 연구 방향은 다음과 같다.

### 4.2.1 Horizon-aware / Multi-theta 확장

- 논문에서도 “forecasting horizon에 따라 서로 다른 theta 라인을 선택”하는 확장을 future work로 제안한다.[1]
- 연구 아이디어:
  - $\(\theta(h)\)$ : 예측 지평 \(h\)에 따라 다른 $\(\theta\)$  사용
  - 다중 Theta 라인 $\(\{\theta_k\}\)$ 을 두고, horizon별 가중치를 학습  
  - M4/M5 수준의 대규모 데이터에서 horizon-aware OTM의 일반화 성능 분석

### 4.2.2 외삽 모형 선택의 자동화

- 현재 OTM은 기본적으로 SES를 사용하고, Holt/Damped를 실험적으로 비교하는 수준에 머무른다.[1]
- 향후:
  - $\(Z_t(\theta)\)$ 에 대해 ETS, ARIMA, TBATS, Prophet 등 **여러 외삽 모형을 후보군으로 두고, GROE 또는 meta-learning으로 선택/조합**하는 구조
  - Spiliotis et al.(2020)의 자동 Theta, Montero-Manso류 FFORMA와 결합하여, **“Automatic Optimised Theta + Model Selection”** 프레임워크 구축[4]

### 4.2.3 Deep/Global 모델과의 하이브리드

- 최근 deep learning 기반 global forecasting(N-BEATS, N-HiTS 등)이 부상했지만,  
  - Makridakis et al.(2018) 및 후속 비교 연구에서는 통계적 방법(Theta, ARIMA, Comb)이 여전히 강력하다는 결과가 다수 보고된다.[13][14][12]
- 향후 방향:
  - **Global DL 모델의 출력에 Theta/OTM을 후처리(잔차 보정, drift 보정)**  
  - 반대로, **Theta/OTM 예측을 DL 모델의 feature 또는 residual input으로 사용하는 하이브리드**  
  - NHITS 등과 비교 시, 특정 조건(짧은 지평, 이상치/비정상성)에서 Theta가 더 낫다는 최근 결과를 높이 활용해, **조건부 앙상블/모델 선택 전략** 설계.[15][13][5]

### 4.2.4 불확실성 추정 및 구조적 해석

- Structural Theta(2024)와 state-space 해석 연구들은 Theta를 구조적 시계열모형으로 재해석하며,  
  - **오차 분해, 구성요소 수준의 불확실성, 베이지안 추론** 등의 장점을 제공한다.[9][3]
- 향후:
  - OTM을 state-space 형태로 완전히 재정의하여,
    - $\(\theta\)$ 를 **상태변수/하이퍼파라미터로 베이지안 추론**
    - 예측 구간, 위험 기반 의사결정 등으로 확장

### 4.2.5 일반화 성능 평가 프레임의 고도화

- 현재 OTM 및 후속 연구는 주로 sMAPE/MASE 평균, 랭크 기반 MCB 테스트 등으로 평가.[4][1]
- 앞으로는:
  - **조건부 성능(시계열 특성별, horizon별, 데이터 품질별)**을 세분하여 분석
  - Deep vs Statistical 비교 연구에서 강조되는 것처럼,  
    - “평균적인 성능”이 아닌 **task-aware, aspect-aware generalization**을 측정하는 프레임 (예: 변동성, 이상치 비율, 구조 변화 빈도 등)에 Theta/OTM을 위치시킬 필요가 있다.[14][13][5]

***

# 5. 요약

- 이 논문은 Theta 방법을 **이론적으로 일반화**하고, **단기 곡률 성분의 theta 계수를 검증 기반으로 최적화**하는 Optimised Theta Method(OTM)을 제시한다.[1]
- 핵심 수학적 기여는
  - 임의 두 Theta 라인의 **유일한 재구성 가중치 공식** $\(\omega(\theta_1,\theta_2)\)$ 제시,
  - 장기선( $\(\theta_1=0\)$ ) 유지 + 단기선( $\(\theta_2=\theta\)$ ) 최적화 구조 정의,
  - 일반화된 rolling-origin 손실 함수(GROE)를 통한 $\(\theta\)$ 선택 절차이다.[1]
- M3 전체 데이터에 대한 실험에서, OTM은 기존 Theta와 주요 통계 모형(ETS, ARIMA, Damped)을 **일관되게 상회하는 일반화 성능**을 보이며, 계산 비용은 증가하지만 여전히 실무적으로 허용 가능한 수준이다.[3][1]
- 2020년 이후 관련 연구들은
  - Theta/OTM을 기반으로 한 자동화, 구조화(Structural Theta), 앙상블, deep hybrid 등을 활발히 전개하고 있으며,
  - 여전히 Theta 계열이 **대규모 시계열 예측에서 매우 강력한, 그리고 해석 가능한 베이스라인**임을 확인하고 있다.[7][8][5][9][4]
- 향후 연구에서는
  - horizon-aware 다중 Theta,
  - 외삽 모형 선택의 자동화 및 meta-learning,
  - deep/global 모델과의 하이브리드,
  - state-space/베이지안 확장,
  - 조건부 일반화 성능 분석  
  등을 중심으로 OTM의 아이디어를 확장·통합하는 것이 유망하다.

### 출처
[1] 1503.03529v1.pdf https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4e84c581-066e-4225-bfcb-e4a5cf8d82a6/1503.03529v1.pdf
[2] The Optimised Theta Method https://arxiv.org/pdf/1503.03529.pdf
[3] Models for optimising the theta method and their relationship to state ... https://orca.cardiff.ac.uk/id/eprint/86781/8/fiorucci%20et%20al%202016.pdf
[4] Generalizing the Theta method for automatic forecasting https://ideas.repec.org/a/eee/ejores/v284y2020i2p550-558.html
[5] Getting the Most Out of Univariate Time Series Forecasting https://pdfs.semanticscholar.org/8f8a/039451735b38d55225b78dbabd5dd31585f6.pdf
[6] D\'ej\`a vu: A data-centric forecasting approach through time series
  cross-similarity https://arxiv.org/pdf/1909.00221.pdf
[7] A Strong Baseline for Weekly Time Series Forecasting http://www.arxiv.org/pdf/2010.08158v1.pdf
[8] Development and evaluation of probabilistic forecasting methods for small area populations https://journals.sagepub.com/doi/pdf/10.1177/23998083231178817
[9] The structural Theta method and its predictive performance ... https://www.sciencedirect.com/science/article/pii/S0169207024000906
[10] Enhancing Agricultural Commodity Forecasting: A Median-Based Combination of Time Series Models https://journaljeai.com/index.php/JEAI/article/view/2782
[11] Short-term load forecasting using Theta method https://pdfs.semanticscholar.org/797b/3f4524307d18049d6fec08a659bb5114d7a6.pdf
[12] Statistical and Machine Learning forecasting methods https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0194889
[13] Forecasting with Deep Learning: Beyond Average of Average of Average
  Performance http://arxiv.org/pdf/2406.16590.pdf
[14] Are Statistical Methods Obsolete in the Era of Deep ... https://arxiv.org/html/2505.21723v1
[15] Universal Time-Series Representation Learning: A Survey https://arxiv.org/html/2401.03717v3
[16] Time series forecasting : advances on Theta method https://www.semanticscholar.org/paper/138b29ffec7a3152bb0271ec185cd683a06a3c77
[17] Reliability Forecasting for Simulation-based Workforce Planning https://www.semanticscholar.org/paper/d0195f705668105757583741acec29cf4aaa5406
[18] Study To Optimise Interior Field Flow Of A Lobe Pump https://www.semanticscholar.org/paper/6d2fe167885a8d30d5a6d8481f256b8364583432
[19] The Optimised Theta Method https://www.semanticscholar.org/paper/8c2d8cebc120d50ffe1dbaad71b1640a5bacba6d
[20] Forecasting methods in oil & gas sector. Optimised Theta model and application on annual oil and gas demand data of european countries https://polynoe.lib.uniwa.gr/xmlui/handle/11400/1483
[21] Short-term load forecasting using Theta method https://www.e3s-conferences.org/10.1051/e3sconf/20198401004
[22] Optimised extreme gradient boosting model for short term electric load demand forecasting of regional grid system https://www.nature.com/articles/s41598-022-22024-3
[23] Forecasting the Chinese energy security index price: A Gaussian process regression-based machine learning framework enhanced by Bayesian optimisation and cross-validation https://journals.sagepub.com/doi/10.1177/03019233251404104
[24] Machine learning based crop water demand forecasting using minimum climatological data http://link.springer.com/10.1007/s11042-019-08533-w
[25] A loss discounting framework for model averaging and selection in time
  series models https://arxiv.org/pdf/2201.12045.pdf
[26] The Wisdom of the Data: Getting the Most Out of Univariate Time Series Forecasting https://www.mdpi.com/2571-9394/3/3/29/pdf
[27] Improving forecasting by subsampling seasonal time series https://arxiv.org/pdf/2101.00827.pdf
[28] Optimizing accuracy and diversity: a multi-task approach to forecast
  combinations https://arxiv.org/pdf/2310.20545.pdf
[29] Integrating Physics-Informed Deep Learning and ... https://arxiv.org/pdf/2410.04299.pdf
[30] A Framework to Produce Local Explanations for Global ... https://arxiv.org/pdf/2111.07001.pdf
[31] Deep Neural networks for solving high-dimensional ... https://www.arxiv.org/pdf/2601.13256.pdf
[32] Boosting global time series forecasting models: a two- ... https://arxiv.org/html/2502.08600v2
[33] ThetA -- fast and robust clustering via a distance parameter https://arxiv.org/abs/2102.07028
[34] Time series forecasting : advances on Theta method https://www.semanticscholar.org/paper/Time-series-forecasting-:-advances-on-Theta-method-Fiorucci/138b29ffec7a3152bb0271ec185cd683a06a3c77
[35] Generalizing the Theta method for automatic forecasting https://www.semanticscholar.org/paper/Generalizing-the-Theta-method-for-automatic-Spiliotis-Assimakopoulos/f06b07c2090cf5d60e3b0f5e291ede55ebeed3ad
[36] Dynamically Weighted Momentum with Adaptive Step ... https://arxiv.org/html/2510.25042v1
[37] Optimized Theta Model - Nixtla https://nixtlaverse.nixtla.io/statsforecast/docs/models/optimizedtheta.html
[38] pde-constrained deep kernel learning in - https ://ris.utwen te.nl https://ris.utwente.nl/ws/portalfiles/portal/482607693/2501.18258v1.pdf
[39] forecTheta: Forecasting Time Series by Theta Models https://cran.r-project.org/web/packages/forecTheta/forecTheta.pdf
[40] Time Series Forecasting with Theta model https://www.kaggle.com/code/kkhandekar/time-series-forecasting-with-theta-model
[41] ThetaEvolve: Test-time Learning on Open Problems https://arxiv.org/abs/2511.23473
[42] Models for optimising the theta method and their ... https://www.sciencedirect.com/science/article/pii/S0169207016300243
[43] The Theta Model - statsmodels 0.14.6 https://www.statsmodels.org/stable/examples/notebooks/generated/theta-model.html
[44] Deep Learning for the Approximation of a Shape Functional https://arxiv.org/abs/2110.02112
[45] Models for optimising the theta method and their ... https://ideas.repec.org/a/eee/intfor/v32y2016i4p1151-1161.html
[46] Easily Employ A Theta Model For Time Series https://towardsdatascience.com/easily-employ-a-theta-model-for-time-series-b94465099a00/
[47] Multi-Objective Loss Balancing for Physics-Informed Deep ... https://www.sciencedirect.com/science/article/pii/S0045782525001860
