# The theta model: a decomposition approach to forecasting

# 1. 논문의 핵심 주장과 주요 기여 (간결 요약)

Assimakopoulos & Nikolopoulos(2000)의 “The theta model: a decomposition approach to forecasting”의 핵심은 다음 두 가지이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/09ebe41f-bbac-4021-8e69-291220d9a540/theta.pdf)

1. **핵심 주장**
   - 시계열의 **국소 곡률(local curvature)**을 직접 조정하는 계수 $\( \theta \)$ 를 2차 차분에 곱해 새로운 시계열(Theta‑lines)을 만들면,  
     - 원래 시계열의 **평균과 기울기(추세)는 유지**하면서  
     - **장기 추세 성분과 단기 변동 성분을 분리**할 수 있고,  
   - 이들을 각각 예측한 후 단순 결합하면, 기존 단변량 기법(ARIMA, ES 등)보다 **간단하면서도 경쟁력 있는 예측 정확도**를 얻을 수 있다는 것이다.

2. **주요 기여**
   - **새로운 분해 방식**: 전통적 추세–계절–불규칙 분해 대신, **곡률 조정 기반(Theta‑lines) 분해**를 제안.
   - **단순 구조**: 두 개의 Theta‑line ( $\(\theta=0,2\)$ )만으로도 매우 단순한 모델 구조를 유지.
   - **실증 성과**: M3 대회 3003개 시계열에 적용했을 때 특히 **월별·미시 경제 시계열에서 우수한 성능**을 보이며 사실상의 우승 모델로 자리잡음. [research.bangor.ac](https://research.bangor.ac.uk/en/impacts/enabling-effective-and-fast-decision-making-in-organisations-fore/)
   - 이후 연구에서 Hyndman & Billah가 이 방법이 **drift를 가진 단순 지수평활(SES with drift)**와 동치임을 보이며, 예측 구간 및 최대우도 추정을 연결했다는 점도 간접 기여. [sktime](https://www.sktime.net/en/v0.19.2/api_reference/auto_generated/sktime.forecasting.theta.ThetaForecaster.html)

***

# 2. 논문 내용 상세 설명

## 2.1 해결하고자 하는 문제

전통적 분해 접근(추세–계절–불규칙)은 실무에서 잘 쓰이지 않는데, 그 이유는: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/09ebe41f-bbac-4021-8e69-291220d9a540/theta.pdf)

- 오차 성분 분리 어려움
- 추세–계절 각 성분의 예측이 불안정
- 복잡한 절차에 비해 성능 이점이 크지 않음

저자들은:

- **“모형을 적용하기 전에, 데이터에 내재된 유용한 정보를 더 많이 끌어내릴 수 없을까?”**라는 문제의식에서 출발한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/09ebe41f-bbac-4021-8e69-291220d9a540/theta.pdf)
- 시계열에 내재된 정보는 직관적으로 **장기(추세)와 단기(최근 변동)**로 나눌 수 있으므로,
- 이 둘을 **명시적으로 분리·강조한 뒤 각각을 예측·결합**하는 방법을 제안한 것이 Theta 모델이다.

즉, **단변량(univariate) 시계열에서 최대한 단순한 구조로 장·단기 정보를 모두 활용하는 예측법**을 만들려는 것이 목표다.

***

## 2.2 제안 방법 – Theta‑계수와 수식

### 2.2.1 국소 곡률에 Theta 적용

원 시계열 $\( X_t \)$ 의 **2차 차분**은

$$
\Delta^2 X_t = X_t - 2X_{t-1} + X_{t-2}
$$

이다. 저자들은 여기에 **Theta 계수 $\( \theta \)$ **를 곱해 국소 곡률을 조정한 새로운 시계열 $\(X_t^{(\theta)}\)$ 를 정의한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/09ebe41f-bbac-4021-8e69-291220d9a540/theta.pdf)

$$
\Delta^2 X_t^{(\theta)} = \theta \, \Delta^2 X_t
$$

이를 적분(합산)해 복원하면,  
- **평균과 전체 기울기(직선 추세)는 원 데이터와 동일**하게 유지하면서,  
- 곡률(굽은 정도)만 조절된 새로운 시계열이 얻어진다.[1, 부록 A,B]

직관적으로:

- $\( 0 < \theta < 1 \)$ : **곡률 감소 → 장기 추세(부드러운 선) 강조**
- $\( \theta > 1 \)$ : **곡률 증가 → 단기 변동(최근 패턴) 강조**
- $\( \theta = 0 \)$ : 완전히 **직선 추세(OLS 회귀 직선)**가 됨
- $\( \theta = 1 \)$ : 원래 시계열 유지

이렇게 생성된 시계열을 **Theta‑line**이라 부른다.

***

### 2.2.2 두 개의 Theta‑line으로의 분해

논문에서 M3 대회에 사용한 **대표적인 설정**은 다음과 같다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/09ebe41f-bbac-4021-8e69-291220d9a540/theta.pdf)

1. **두 개의 Theta‑line 사용**  
   - $\( \theta_1 = 0 \)$ : 완전히 평탄화된 **직선 추세선**  
   - $\( \theta_2 = 2 \)$ : **곡률 2배**로 단기 변동을 강조한 선

2. 이 두 선의 단순 평균이 원 시계열을 재구성하도록 설계:

$$
X_t = \frac{1}{2} L_t(\theta=0) + \frac{1}{2} L_t(\theta=2)
$$

여기서 $\(L_t(\theta)\)$ 는 $\(\theta\)$ 에 해당하는 Theta‑line이다.[부록 B] [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/09ebe41f-bbac-4021-8e69-291220d9a540/theta.pdf)

3. 부록 B에서 더 일반적인 식을 제시하지만, 핵심 아이디어는  
   - **1 이하의 $\(\theta\)$ **와 **1 이상(대개 >1)의 $\(\theta\)$ **를 짝으로 두고  
   - 이 둘의 적절한 가중 평균이 **원 시계열을 정확히 복원**하도록 만드는 것이다. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC11310609/)

***

### 2.2.3 Theta‑line의 명시적 표현 (후속 연구에서 정리된 형태)

후속 연구(Nikolopoulos et al., Hyndman 등)는 Theta‑line을 **선형 회귀 추세와 원 시계열의 선형 결합**으로 명시적으로 표현했다.: [robjhyndman](https://robjhyndman.com/papers/Theta.pdf)

원 시계열 $\(Y_t\)$ , OLS 추세가

$$
\hat{Y}_t = A_n + B_n t
$$

일 때, 임의의 $\(\theta\)$ 에 대한 Theta‑line은

$$
Z_t(\theta) = \theta Y_t + (1-\theta)(A_n + B_n t)
$$

으로 쓸 수 있다. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC11310609/)

- $\(\theta=0\)$ : $\(Z_t(0) = A_n + B_n t\)$ → **순수 직선 추세**
- $\(\theta=1\)$ : $\(Z_t(1) = Y_t\)$  → 원 시계열
- $\(\theta=2\)$ : 직선 추세에서 원 시계열 방향으로 **곡률 2배**로 휜 선

이 표현은 Theta‑line이 **“직선 추세 + 곡률 조정된 잔차”**라는 점을 명확히 보여준다.

***

## 2.3 모델 구조와 절차

M3 대회에서 사용한 **실질적인 Theta 모델 파이프라인**은 다음과 같다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/09ebe41f-bbac-4021-8e69-291220d9a540/theta.pdf)

1. **계절성 검정 (Step 0)**  
   - 1년 주기 래그(월별: 12, 분기별: 4)의 자기상관 계수에 대한 t‑검정으로 계절성 여부 판단. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/09ebe41f-bbac-4021-8e69-291220d9a540/theta.pdf)

2. **비계절화 (Step 1)**  
   - 고전적 곱셈 분해(classical multiplicative decomposition)로 계절 성분 제거 → **계절조정 시계열** 생성.

3. **Theta 분해 (Step 2)**  
   - 계절조정 시계열을  
     - $\( \theta=0 \)$ → **직선 추세선 $\(L(\theta=0)\)$ **  
     - $\( \theta=2 \)$  → **단기 변동 강조선 $\(L(\theta=2)\)$ **  
     으로 분해.

4. **각 성분 예측 (Step 3)**  
   - $\(L(\theta=0)\)$ : 시간에 대한 **단순 선형 회귀(linear trend)**로 외삽.  
   - $\(L(\theta=2)\)$ : **단순 지수평활(SES)**으로 외삽.

5. **결합 (Step 4)**  
   - 두 예측치를 **동일 가중 평균**:

$$
   \hat{X}_{t+h} = \frac{1}{2} \hat{L}_{t+h}(\theta=0)
                 + \frac{1}{2} \hat{L}_{t+h}(\theta=2)
   $$

6. **재계절화 (Step 5)**  
   - 계절성이 있던 시계열의 경우, Step 1에서 얻은 계절지수로 다시 곱해서 원 스케일로 복원.

이 구조의 중요한 점:

- **모델 자체는 매우 단순**하지만,
- 분해를 통해 **장기 추세(직선)와 단기 패턴(지수평활)**을 명시적으로 분리·결합한다는 것.

***

## 2.4 성능 향상 및 한계

### 2.4.1 성능 향상

- Theta 모델은 M3 경쟁에서 **월별·미시 경제 시계열에서 특히 우수한 성능**을 보였고, 사실상 **우승 모델로 간주**된다. [research.bangor.ac](https://research.bangor.ac.uk/en/impacts/enabling-effective-and-fast-decision-making-in-organisations-fore/)
- 이후 여러 벤치마크에서 **ARIMA, ETS와 함께 대표적인 “클래식 베이스라인”**으로 자리잡았고, M4 대회에서도 강력한 기준선으로 사용된다. [stat.berkeley](https://www.stat.berkeley.edu/~ryantibs/timeseries-f23/lectures/advanced.pdf)
- 최근 대규모 벤치마크(예: NHITS vs classical models)에서도,  
  - 전체 평균 성능에서는 최신 딥러닝(예: NHITS)에 약간 밀리지만,  
  - **이상치 구간·특정 조건(짧은 시계열, 이상 관측 포함 등)에서는 여전히 Theta가 더 견고**한 경우가 보고된다. [arxiv](https://arxiv.org/html/2506.07987v1)

성능 향상의 메커니즘은 논문에서 다음과 같이 해석한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/09ebe41f-bbac-4021-8e69-291220d9a540/theta.pdf)

- $\( \theta=0 \)$ 직선 추세는 **장기적인 레벨과 기울기 정보를 “잃지 않고” 가져옴**.
- $\( \theta=2 \)$ 선은 **최근 변동성을 증폭**하여, 직선 추세만 쓸 때 무시되는 **최근 패턴을 보완**.
- 두 예측의 결합은 **결합 예측(combining forecasts)**의 장점(오차 상쇄, 모델 불확실성 평균화)을 동시에 활용.[Clemen 1989 인용] [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/09ebe41f-bbac-4021-8e69-291220d9a540/theta.pdf)

***

### 2.4.2 한계 및 비판

1. **모수 선택과 구조의 경험적 성격**  
   - $\(\theta=0,2\)$ , 동등 가중, $\(\theta=2\)$ 에 SES 적용 등은 **이론적 최적성 근거보다는 경험적·휴리스틱**에 가깝다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/09ebe41f-bbac-4021-8e69-291220d9a540/theta.pdf)
   - 이후 연구는 이 부분을 **최적화·일반화**하는 방향으로 발전. [arxiv](https://arxiv.org/pdf/1503.03529.pdf)

2. **모형 해석의 복잡성**  
   - 원 논문 수식은 상당히 복잡하며, 본질적으로는 단순한 구조(SES with drift)임에도 불필요하게 난해하게 보일 수 있다.  
   - Hyndman & Billah(2001)는 이를 **SES with drift로 단순화**해 보여주며, 이 관점에서 보면 Theta 모델 자체가 새로운 확률 모형이라기보다 **기존 지수평활의 재해석**이라는 비판도 가능하다. [robjhyndman](https://robjhyndman.com/papers/Theta.pdf)

3. **비선형·복잡 시계열에 대한 한계**  
   - 선형 추세 + 지수평활 구조이기 때문에, **강한 비선형성, 구조적 변동, 복잡한 다중 계절성**이 있는 시계열에 대해서는 한계가 뚜렷하다. [mdpi](https://www.mdpi.com/1099-4300/23/9/1163/pdf)
   - 이러한 상황에서는 TBATS, Prophet, 딥러닝 기반 글로벌 모델 등이 더 유리하다는 결과 다수. [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0957417419306128)

4. **전역(global) 정보 활용의 부재**  
   - Theta 모델은 **완전히 단변량·시계열별(local) 모델**로,  
   - 근래 딥러닝 기반 **글로벌 모델(global time-series model)**들이 다수 시계열 간 공통 패턴을 학습하는 것과 대비된다. [ar5iv.labs.arxiv](https://ar5iv.labs.arxiv.org/html/1710.03222)
   - 따라서 **크로스시리즈 일반화** 측면에서는 구조적 제약이 있다.

***

# 3. Theta 모델의 일반화 성능 향상 가능성

Theta 모델 자체는 단순하지만, 그 구조(장·단기 성분 분해 후 결합)는 **일반화 성능을 개선하기 좋은 여러 특징**을 가진다.

## 3.1 구조적 정규화(structural regularization)로서의 역할

1. **장기 추세의 선형화**  
   - $\( \theta=0 \)$ 성분은 **OLS 직선 추세**로, 매우 제한된(저차원) 함수 공간에 속한다.  
   - 이는 장기 추세에 대한 **구조적 정규화**로 작용해, 과적합을 방지하고 외삽 안정성을 높인다. [robjhyndman](https://robjhyndman.com/papers/Theta.pdf)

2. **단기 변동의 완만한 모형화**  
   - $\( \theta=2 \)$ 성분은 SES라는 **저복잡도(1개 smoothing parameter)** 모형에 의해 예측된다.  
   - 장기와 단기를 각각 “simple model”로 설명하고, 이를 결합하므로 **과도한 유연성 없이도 다양한 패턴을 흡수**한다.

3. **결합 예측의 분산 감소**  
   - 서로 다른 특성(장기 vs 단기)을 가진 두 예측을 결합하면,  
   - 예측 오차의 상관이 완벽하지 않을 때 **분산 감소 → 일반화 성능 개선** 효과. [arxiv](https://arxiv.org/pdf/2406.16590.pdf)

이러한 특성 때문에, Theta는 딥러닝 글로벌 모델과 비교해도 **짧은 시계열, 이상치·변동성 높은 데이터**에서는 여전히 견고한 성능을 보인다. [repository.londonmet.ac](https://repository.londonmet.ac.uk/10423/)

***

## 3.2 이후 발전된 Theta 계열 연구와 일반화 성능

### 3.2.1 최적화·일반화된 Theta (Optimised / Generalized Theta)

- **Optimised Theta Method (OTM)**:  
  - 두 Theta‑line의 $\(\theta_1, \theta_2\)$ , 가중치 $\(\omega\)$ 를 **손실 최소화 기반으로 최적화**해, 원 Theta보다 예측 정확성을 높인 일반화된 형태. [arxiv](https://arxiv.org/pdf/1503.03529.pdf)
  - 일반형:

$$
    Y_t = \omega Z_t(\theta_1) + (1-\omega) Z_t(\theta_2),
    $$

$$
    \omega(\theta_1,\theta_2) = \frac{\theta_2 - 1}{\theta_2 - \theta_1}
    $$

    로, 항상 원 시계열을 재조합할 수 있도록 설계. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC11310609/)

- **Generalizing the Theta method for automatic forecasting (Spiliotis et al., 2020)** [pure.unic.ac](https://pure.unic.ac.cy/en/publications/generalizing-the-theta-method-for-automatic-forecasting/)
  - (i) 선형뿐 아니라 **비선형 추세**까지 고려  
  - (ii) 추세 기울기(slope)를 조정하는 메커니즘 추가  
  - (iii) 기본 모형의 **가법/곱셈 구조** 모두 허용  
  - 결과적으로 Theta를 **자동 시계열 예측용 일반 프레임워크**로 확장, M/M3/M4 전체에서 원 Theta 및 다른 고전 방법보다 향상된 성능을 보고.

→ 이 계열은 **모수 공간을 확장하되, 여전히 선형·저복잡도 구조를 유지**하므로,  
   - 과적합 위험을 크게 늘리지 않으면서  
   - 다양한 패턴에 대응하는 **유연성–일반화의 균형**을 개선한 것이라 볼 수 있다.

***

### 3.2.2 구조적 Theta, 하이브리드 Theta, 짧은 시계열용 개선

- **Structural Theta Method (Sbrana, 2024)** [sciencedirect](https://www.sciencedirect.com/science/article/pii/S0169207024000906)
  - Theta 방식을 **구조적 시계열 모형(상태공간) 관점**에서 재구성,  
  - 여러 Theta 변형과 통합해 **예측 간신(precision)과 구조 해석 가능성**을 동시에 추구.  
  - 다양한 Theta 변종(OTM, DOTM 등)과 비교해 구조적 Theta가 경쟁력 있는 예측 성능을 보임.

- **Hybrid Theta & Temporal Aggregation (박사논문, 2022)** [researchportal.bath.ac](https://researchportal.bath.ac.uk/en/studentTheses/improving-forecasting-through-a-hybrid-theta-method-and-its-integ/)
  - 계절성 검정·분해·추세외삽을 확장한 **하이브리드 Theta**를 제안하고,  
  - 이를 **multiple temporal aggregation (TA)**와 통합해,  
  - 특히 **트렌드성 데이터에서 TA와 Theta의 상호작용이 항상 이득이 되지 않음**을 보이면서,  
  - 어떤 조건에서 일반화 성능이 개선/악화되는지 정량적으로 분석.

- **  $\(\theta\)$ -comb: Improved Theta for short series (Mattera, 2025)** [repository.londonmet.ac](https://repository.londonmet.ac.uk/10423/)
  - 단기 시계열(Human Development Index 등)에서  
  - Theta의 **단기 성분 예측을 여러 방법으로 생성 후 결합(θ‑comb)**하는 방식 제안.  
  - 데이터가 짧아 딥러닝·복잡 모형이 과적합되기 쉬운 상황에서 **예측 정확성 향상**을 보고.

→ 이들 연구는 **“단순한 Theta 구조를 유지하면서, 조합/상태공간/aggregation을 조정해 일반화 성능을 체계적으로 개선”**하는 방향이라고 볼 수 있다.

***

### 3.2.3 Theta + 딥러닝 하이브리드: Theta‑ARNN (TARNN)

- **Theta Autoregressive Neural Network (TARNN)** [medrxiv](http://medrxiv.org/lookup/doi/10.1101/2020.10.01.20205021)
  - COVID‑19 확산 예측에서,  
  - **Theta 모델 + Autoregressive Neural Network (ARNN)** 하이브리드 제안.
  - 아이디어:
    1. Theta 모델로 **장기 추세·전체 구조**를 먼저 설명
    2. 남은 잔차(residual)에 대해 **비선형 ARNN**을 학습
  - 이 구조는  
    - Theta가 **저차원 선형 구조**를 담당하고  
    - ARNN이 **고차원 비선형 잔차**만을 학습하도록 제한함으로써,  
    - 딥러닝 단독보다 **과적합 위험이 줄고 일반화가 개선**될 가능성이 크다.
  - 논문에서는 TARNN이 다수의 기존 단일·하이브리드 모형보다 코로나 확산 단·장기 예측에서 더 나은 성능을 보였음을 보고. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC9454152/)

→ Theta는 **딥러닝 전처리/스캐폴딩(skeleton) 모형**으로서, 구조적 편향(bias)을 주어 딥러닝의 분산(variance)을 줄이는 **바이어스–분산 절충**을 실현하는 도구로 해석할 수 있다.

***

## 3.3 최신 글로벌·딥러닝 기반 모델과의 비교 (2020년 이후)

최근 연구는 **수십만 개 시계열을 동시에 학습하는 글로벌 딥러닝 모델**에 집중되고 있다. [arxiv](https://arxiv.org/html/2401.03717v3)

대표적으로:

- **RNN/LSTM 기반 글로벌 모델**: Bandara et al.(2020), Elsworth(2020) 등 [arxiv](https://arxiv.org/pdf/2003.05672.pdf)
- **NHITS, DeepAR, PatchTST, TimesNet 등 딥러닝 시계열 모델**: Cerqueira et al.(2024), 여러 벤치마크 [arxiv](https://arxiv.org/pdf/2406.16590.pdf)
- **사전학습(Foundation) 시계열 모델 (예: Delphyne)**: 2024 이후 연구 [arxiv](https://arxiv.org/html/2410.11539v1)

이들 연구는 공통적으로:

- 평균적으로는 **딥러닝 글로벌 모델이 고전 로컬 모델(ARIMA, ETS, Theta 등)을 상회**하지만,
- **데이터 길이가 짧거나, 이상치가 많거나, 특정 조건(1-step 예측 등)**에서는  
  - 여전히 Theta, ETS 등이 더 견고하거나  
  - 성능 격차가 미미하다고 보고한다. [stat.berkeley](https://www.stat.berkeley.edu/~ryantibs/timeseries-f23/lectures/advanced.pdf)

따라서:

- **큰·풍부한 데이터셋**: 글로벌 딥러닝 + 요약형(temporal aggregation, embedding) 접근이 우위  
- **짧고 잡음이 많으며 구조적 정규화가 필요한 상황**:  
  - Theta 계열(OTM, θ‑comb, structural Theta, TARNN 등)이  
  - **모델 복잡도 대비 일반화 성능이 매우 좋은 선택**으로 남아 있다.

***

# 4. 2020년 이후 관련 최신 연구 비교 분석 (오픈 액세스 중심)

요청하신 바와 같이, Theta 및 시계열 예측 관련 **2020년 이후 주요 개방형 연구**를 간단 표 형식으로 정리한다 (일부는 preprint / open-access 저널).

| 연도 | 논문 / 링크 | 핵심 내용 & Theta와의 관계 |
|------|-------------|----------------------------|
| 2020 | Spiliotis et al., “Generalizing the Theta method for automatic forecasting” (EJOR) [ideas.repec](https://ideas.repec.org/a/eee/ejores/v284y2020i2p550-558.html) | Theta의 추세/구조를 일반화해 자동 예측용 프레임워크로 확장. OTM 등 변형 포함. M, M3, M4 전체에서 원 Theta·다른 고전 기법보다 향상된 성능 보고. |
| 2020 | Hyndman, “Unmasking the Theta method” (preprint) [robjhyndman](https://robjhyndman.com/papers/Theta.pdf) | Theta가 **단순 지수평활(SES) with drift**와 수학적으로 동치임을 증명하고, 상태공간 표현·예측구간 제공. Theta의 이론적 기반을 간명하게 정리. |
| 2021 | Chakraborty et al., “Theta Autoregressive Neural Network: A Hybrid Time Series Model for Pandemic Forecasting (TARNN)” [ieeexplore.ieee](https://ieeexplore.ieee.org/document/9533747/) | Theta + ARNN 하이브리드로 COVID‑19 예측. 구조적(Theta) + 비선형(ARNN) 결합으로 여러 기존 모형보다 우수한 단·장기 예측 성능. 일반화 측면에서 구조적 bias + 딥러닝 variance 절충의 사례. |
| 2021 | Gastinger et al., “Ensemble Learning for Time Series Forecasting” [arxiv](https://arxiv.org/pdf/2104.11475.pdf) | 수십 개 모델(ARIMA, ETS, Theta 포함)을 조합한 앙상블 기법을 대규모 벤치마크에 적용. Theta는 여전히 강력한 베이스라인이며, 대부분의 top 성능은 **앙상블(종종 Theta 포함)**이 차지함을 보임. |
| 2021 | Time Series Modelling (Special Issue, Entropy) [mdpi](https://www.mdpi.com/1099-4300/23/9/1163/pdf) | 현대 시계열 모형 개관. Theta는 ARIMA, ETS와 함께 대표적 고전 모델로 다수 인용. |
| 2022 | PhD thesis “Improving forecasting through a hybrid Theta method and its integration with temporal aggregation” [researchportal.bath.ac](https://researchportal.bath.ac.uk/en/studentTheses/improving-forecasting-through-a-hybrid-theta-method-and-its-integ/) | Theta의 계절성 처리·추세외삽을 확장한 하이브리드 Theta 제안, TA와 통합. 다양한 실데이터에서 기존 Theta·벤치마크보다 향상된 성능, 단 **트렌드 강한 시계열에서 TA가 항상 이득은 아님**을 보임. |
| 2023 | Kamalov, “Deep learning for COVID‑19 forecasting: State‑of‑the‑art review” [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC9454152/) | COVID 예측 딥러닝 리뷰에서 **Theta‑ARNN(TARNN)**을 포함해 전통+딥러닝 하이브리드를 비교. 일부 상황에서 딥러닝 단독보다 하이브리드가 더 안정적임을 강조. |
| 2024 | Cerqueira et al., “Beyond Average of Average of Average Performance” / NHITS 평가 논문 [arxiv](https://arxiv.org/pdf/2406.16590.pdf) | NHITS vs ARIMA, ETS, Theta 등 대규모 비교. 전체 평균은 NHITS가 상회하지만, **이상치·특정 조건에서 Theta/ETS가 NHITS보다 우수**한 사례 명확히 제시 → Theta의 견고성 확인. |
| 2024 | Wang et al., “Estimating the Temporal Epidemiological Trends of Tuberculosis using Advanced Theta Methods” [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC11310609/) | TB 발생 데이터에 SARIMA와 여러 Theta 변형(STM, DOTM, DSTM, OTM)을 비교. OTM 등 최적화 Theta가 전통 Theta와 ARIMA보다 개선된 예측 성능을 보임. Theta‑line의 명시적 식 $\(Z_t(\theta)=\theta Y_t+(1-\theta)(A_n+B_nt)\)$ 제시. |
| 2024 | Sbrana, “The structural Theta method and its predictive performance” [sciencedirect](https://www.sciencedirect.com/science/article/pii/S0169207024000906) | Theta를 구조적 시계열(상태공간) 모델로 재정식화하고, 기존 Theta 변형과의 예측 성능 비교. 구조적 Theta가 다수의 벤치마크에서 경쟁력 있는 성능 확보. |
| 2024 | ModelRadar / aspect-based forecast evaluation [arxiv](https://arxiv.org/html/2504.00059v1) | 다양한 예측 모델(Theta 포함)을 **측면별(예: horizon, 이상치 비율 등)**로 평가하는 프레임워크. 특정 조건에서 Theta가 여전히 강력한 선택지임을 보여줌. |
| 2025 | Mattera, “Forecasting human development with an improved Theta method (θ‑comb)” [repository.londonmet.ac](https://repository.londonmet.ac.uk/10423/) | 짧은 시계열(HDI)에 대해 Theta의 단기 성분 예측을 여러 방법으로 생성 후 조합(θ‑comb). 원 Theta 및 기타 접근보다 향상된 단기 예측 성능 및 일반화 보고. |
| 2025 | Bosch, “Multi-layer Stack Ensembles for Time Series Forecasting” [arxiv](https://arxiv.org/pdf/2511.15350.pdf) | Theta, AutoETS, DeepAR 등 11개 베이스 모델을 포함하는 스택 앙상블 제안. Theta는 여전히 유용한 구성 요소로 사용되며, 앙상블이 단일 모델 대비 일반화 성능 개선. |

***

# 5. 앞으로의 연구에 대한 영향과 연구 시 고려할 점

## 5.1 Theta 모델이 시계열 연구에 미친 영향

1. **단순하지만 강력한 베이스라인의 표준화**
   - Theta는 M3 이후 **“반드시 이겨야 하는” 고전 베이스라인**이 되었다. [stat.berkeley](https://www.stat.berkeley.edu/~ryantibs/timeseries-f23/lectures/advanced.pdf)
   - 많은 최신 연구(딥러닝, foundation models 등)는 실험 설계 시 **ARIMA·ETS·Theta**를 공통 baseline으로 포함한다. [arxiv](https://arxiv.org/html/2506.06288v1)

2. **분해 + 결합이라는 설계 패턴 보급**
   - Theta의 “곡률 조정 → 장·단기 분해 → 별도 예측 → 결합” 구조는  
   - 이후 **Prophet, TBATS, LTSTA, 다양한 hybrid/ensemble** 설계에서 공통적으로 등장하는 패턴이다. [arxiv](https://arxiv.org/pdf/2511.15350.pdf)

3. **구조적 단순성의 가치 재조명**
   - 대규모 딥러닝 모델들이 등장한 이후에도, Theta는  
   - 짧은 시계열, 데이터가 희소한 사회지표, 의료·역학 데이터 등에서 **“단순하지만 일반화가 뛰어난” 대안**으로 계속 사용되고 있다. [journals.plos](https://journals.plos.org/digitalhealth/article?id=10.1371%2Fjournal.pdig.0000598)

***

## 5.2 앞으로 연구 시 고려할 점 (특히 “일반화 성능 향상” 관점에서)

1. **구조적 편향(Theta) + 표현력(딥러닝) 결합**
   - Theta가 제공하는 **선형 추세·저복잡도 구조**를 딥러닝의 인풋 혹은 잔차 모형으로 사용하면,  
   - 딥러닝이 “모든 것”을 학습하는 대신 **잔차의 비선형성만 학습**하게 되어 일반화에 유리하다.  
   - TARNN, Hybrid Theta + TA는 이런 방향의 초기 사례이며, [ieeexplore.ieee](https://ieeexplore.ieee.org/document/9533747/)
   - 앞으로는 **foundation time‑series model + Theta 기반 구조적 prior** 결합이 유망하다. [arxiv](https://arxiv.org/html/2410.11539v1)

2. **조건부(측면별) 평가를 통한 모델 선택**
   - 최근 연구(ModelRadar, NHITS 평가 등)는 평균 성능이 아니라  
     - 예측 시차, 이상치 빈도, 계절성 등 **조건별 성능**을 평가할 것을 강조한다. [arxiv](https://arxiv.org/html/2504.00059v1)
   - Theta는 특히  
     - **단기 vs 장기 horizon별 가중 조정**,  
     - **이상치·구조변화 구간에서의 안정성**,  
     - **짧은 시계열**  
     에 강점을 보이므로,  
   - 향후 연구에서도 **“언제 Theta/Theta 변형이 최선인가?”**를 조건별로 명시하는 것이 중요하다.

3. **일반화 이론과의 연결**
   - 현재 Theta 연구는 주로 경험적 성능 중심이지만,  
   - Neural ODE, RNN, global models 등에 대한 **일반화 이론(예: Lipschitz, flat minima, NTK)**이 발전하고 있어, [arxiv](https://arxiv.org/pdf/2508.18920.pdf)
   - Theta‑식 분해/결합을 **복잡 모형의 정규화 레이어**로 활용할 경우,  
   - 이론적 일반화 경계와 연결해 분석할 여지가 크다.

4. **짧은 시계열·저빈도 데이터에 특화된 설계**
   - HDI, 질병발생, 매크로 지표 등 **연/분기 단위의 짧은 시계열**은 딥러닝에 부적합한 경우가 많다. [mdpi](https://www.mdpi.com/1099-4300/23/9/1163/pdf)
   - 이 영역에서는  
     - Theta, OTM, θ‑comb, structural Theta, Theta+TA 등  
     - **저복잡도·구조적 모형**이 일반화 측면에서 핵심 역할을 계속할 것이다.

5. **실무적 관점에서의 비용–편익 고려**
   - Uber 등의 사례에서 Theta 계열이 **대규모 서비스에서 막대한 경제적 가치**를 창출한 바 있다. [research.bangor.ac](https://research.bangor.ac.uk/en/impacts/enabling-effective-and-fast-decision-making-in-organisations-fore/)
   - 실무에서는  
     - 구현 난이도, 계산 비용, 모니터링·설명 가능성까지 포함한 **총 소유 비용(TCO)** 관점에서  
     - 딥러닝 foundation model vs Theta 계열(및 그 앙상블)의 **비용–편익 비교**가 중요한 연구 주제가 될 것이다.

***

## 5.3 요약적 제언

- **연구 설계 측면**  
  - 새로운 딥러닝/글로벌 모델 제안 시,  
    - **ARIMA, ETS, (Generalized) Theta, OTM**을 반드시 강력한 baseline으로 포함하고,  
    - **조건별(측면별) 성능 분석**을 통해 “언제 전통 모델이 더 낫거나 동등한가?”를 명시하는 것이 필요하다. [ideas.repec](https://ideas.repec.org/a/eee/ejores/v284y2020i2p550-558.html)

- **모형 개발 측면**  
  - Theta의 핵심 아이디어(곡률 조정, 장·단기 분해, 단순 결합)를  
    - 딥러닝의 아키텍처 설계(예: trend/residual 분리 레이어),  
    - 사전학습 representation의 구조적 prior,  
    로 통합하는 방향이 **일반화–해석 가능성–성능을 모두 고려한 설계**에 유리하다.

- **응용 측면**  
  - 데이터가 짧거나, 이상치·변동성이 크거나, 해석 가능성이 중요한 분야(보건, 공공정책, 지속가능성 지표 등)에서는  
    - 여전히 Theta 및 그 변형(OTM, θ‑comb, structural Theta, TARNN 등)이  
    - **1차 후보 모델**로 충분히 고려될 가치가 있다.

이와 같이, 원 논문은 단순한 분해 기법을 제안한 것처럼 보이지만,  
결과적으로 **“구조적 분해 + 결합”을 통한 일반화 친화적 시계열 예측 패턴**을 제시했고,  
이는 이후 20년 이상 시계열 예측 연구·실무에서 핵심적인 설계 패턴으로 자리 잡고 있다.


<span style="display:none">[^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_80][^1_81][^1_82][^1_83][^1_84][^1_85][^1_86][^1_87][^1_88][^1_89][^1_90][^1_91][^1_92][^1_93][^1_94][^1_95]</span>

<div align="center">⁂</div>

[^1_1]: theta.pdf

[^1_2]: https://research.bangor.ac.uk/en/impacts/enabling-effective-and-fast-decision-making-in-organisations-fore/

[^1_3]: https://www.sktime.net/en/v0.19.2/api_reference/auto_generated/sktime.forecasting.theta.ThetaForecaster.html

[^1_4]: https://robjhyndman.com/papers/Theta.pdf

[^1_5]: https://www.statsmodels.org/stable/examples/notebooks/generated/theta-model.html

[^1_6]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11310609/

[^1_7]: https://www.stat.berkeley.edu/~ryantibs/timeseries-f23/lectures/advanced.pdf

[^1_8]: https://arxiv.org/pdf/2511.15350.pdf

[^1_9]: https://arxiv.org/html/2506.07987v1

[^1_10]: https://arxiv.org/pdf/2406.16590.pdf

[^1_11]: https://arxiv.org/pdf/1503.03529.pdf

[^1_12]: https://pure.unic.ac.cy/en/publications/generalizing-the-theta-method-for-automatic-forecasting/

[^1_13]: https://ideas.repec.org/a/eee/ejores/v284y2020i2p550-558.html

[^1_14]: https://www.mdpi.com/1099-4300/23/9/1163/pdf

[^1_15]: https://www.sciencedirect.com/science/article/abs/pii/S0957417419306128

[^1_16]: https://arxiv.org/html/2506.06288v1

[^1_17]: https://ar5iv.labs.arxiv.org/html/1710.03222

[^1_18]: https://arxiv.org/html/2401.03717v3

[^1_19]: https://arxiv.org/pdf/2104.11475.pdf

[^1_20]: https://repository.londonmet.ac.uk/10423/

[^1_21]: https://www.sciencedirect.com/science/article/pii/S0377221720300242

[^1_22]: https://www.sciencedirect.com/science/article/abs/pii/S0377221720300242

[^1_23]: https://www.sciencedirect.com/science/article/pii/S0169207024000906

[^1_24]: https://researchportal.bath.ac.uk/en/studentTheses/improving-forecasting-through-a-hybrid-theta-method-and-its-integ/

[^1_25]: http://medrxiv.org/lookup/doi/10.1101/2020.10.01.20205021

[^1_26]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9454152/

[^1_27]: https://ieeexplore.ieee.org/document/9533747/

[^1_28]: https://arxiv.org/pdf/2003.05672.pdf

[^1_29]: https://arxiv.org/html/2410.11539v1

[^1_30]: https://arxiv.org/html/2504.00059v1

[^1_31]: https://journals.plos.org/digitalhealth/article?id=10.1371%2Fjournal.pdig.0000598

[^1_32]: https://arxiv.org/pdf/2508.18920.pdf

[^1_33]: https://arxiv.org/html/2506.04690v1

[^1_34]: https://www.semanticscholar.org/paper/8fca6d69b8498c0b747fc490d63c95c23b0e87ef

[^1_35]: https://www.aclweb.org/anthology/2020.emnlp-main.73

[^1_36]: http://www.isca-speech.org/archive/VCC_BC_2020/abstracts/VCC2020_paper_36.html

[^1_37]: https://www.banglajol.info/index.php/DUJS/article/view/54612

[^1_38]: https://ieeexplore.ieee.org/document/9182578/

[^1_39]: http://econ-environ-geol.org/index.php/ojs/article/view/475

[^1_40]: https://www.semanticscholar.org/paper/3a211a97b7dac8b24e4d9753f112d7924fd11e8f

[^1_41]: http://www.ije.ir/article_108448.html

[^1_42]: https://link.aps.org/doi/10.1103/PhysRevA.104.012401

[^1_43]: https://www.aclweb.org/anthology/2020.acl-main.213.pdf

[^1_44]: http://arxiv.org/pdf/1310.8499.pdf

[^1_45]: https://arxiv.org/pdf/1911.12436.pdf

[^1_46]: https://arxiv.org/html/2503.23612v1

[^1_47]: https://arxiv.org/pdf/1903.04933.pdf

[^1_48]: https://arxiv.org/html/2412.05657v2

[^1_49]: https://arxiv.org/pdf/2104.03739.pdf

[^1_50]: https://arxiv.org/pdf/2502.12207.pdf

[^1_51]: https://arxiv.org/html/2503.13505v2

[^1_52]: https://www.semanticscholar.org/paper/Real-time-forecasts-and-risk-assessment-of-novel-A-Chakraborty-Ghosh/48ad6b5c6f5e3a29d7d59191079ea0686bf6d9d9

[^1_53]: https://arxiv.org/html/2601.20556v1

[^1_54]: https://www.biorxiv.org/content/10.64898/2025.12.12.694026v1.full.pdf

[^1_55]: https://arxiv.org/html/2503.13544v3

[^1_56]: https://www.biorxiv.org/content/10.1101/2023.12.12.571268v1.full.pdf

[^1_57]: https://arxiv.org/html/2503.13505v1

[^1_58]: https://arxiv.org/html/2601.13352v1

[^1_59]: https://arxiv.org/html/2410.06851v1

[^1_60]: https://arxiv.org/html/1802.03308v9

[^1_61]: https://pmc.ncbi.nlm.nih.gov/articles/PMC7248977/

[^1_62]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8861707/

[^1_63]: https://ceur-ws.org/Vol-3885/paper50.pdf

[^1_64]: https://www.emergentmind.com/topics/autoregressive-neural-operators

[^1_65]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12329085/

[^1_66]: https://www.nature.com/articles/s41598-025-09970-4

[^1_67]: https://courses.cs.washington.edu/courses/cse599i/20au/resources/L04_nade.pdf

[^1_68]: https://mathdoyun.tistory.com/30

[^1_69]: https://www.sciencedirect.com/science/article/abs/pii/S0010482519301428

[^1_70]: https://www.nature.com/articles/s42005-023-01416-5

[^1_71]: https://www.frontiersin.org/journals/astronomy-and-space-sciences/articles/10.3389/fspas.2022.1031407/full

[^1_72]: https://www.tandfonline.com/doi/full/10.1080/02664763.2023.2179567

[^1_73]: https://wildlife.onlinelibrary.wiley.com/doi/10.1002/jwmg.21985

[^1_74]: https://journal.universitasmulia.ac.id/index.php/seminastika/article/view/275

[^1_75]: http://jecei.sru.ac.ir/article_1477.html

[^1_76]: http://jurnal.sttmcileungsi.ac.id/index.php/jenius/article/view/159

[^1_77]: https://formative.jmir.org/2021/9/e28028

[^1_78]: https://ieeexplore.ieee.org/document/9456509/

[^1_79]: https://astesj.com/v06/i01/p147/

[^1_80]: https://journal.upgris.ac.id/index.php/JIU/article/view/9559

[^1_81]: https://downloads.hindawi.com/journals/complexity/2021/5963516.pdf

[^1_82]: http://arxiv.org/pdf/2305.08124.pdf

[^1_83]: http://arxiv.org/pdf/2203.10702.pdf

[^1_84]: https://arxiv.org/pdf/2405.11111.pdf

[^1_85]: https://arxiv.org/pdf/1905.12118.pdf

[^1_86]: https://arxiv.org/pdf/1810.09996.pdf

[^1_87]: https://arxiv.org/html/2510.02729v1

[^1_88]: https://arxiv.org/html/2205.11235v4

[^1_89]: https://arxiv.org/html/2409.04399v1

[^1_90]: https://arxiv.org/html/2502.08600v2

[^1_91]: https://arxiv.org/html/2502.00818v2

[^1_92]: https://arxiv.org/pdf/2402.04094.pdf

[^1_93]: https://arxiv.org/html/2601.00970v1

[^1_94]: https://www.nature.com/articles/s41467-025-63786-4

[^1_95]: https://discovery.researcher.life/article/forecasting-human-development-with-an-improved-theta-method-based-on-forecast-combination/b8e69c1387443182bb79f27741b64aea
