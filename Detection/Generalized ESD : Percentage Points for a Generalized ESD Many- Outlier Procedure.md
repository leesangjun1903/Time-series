# Percentage Points for a Generalized ESD Many- Outlier Procedure

### 1. 핵심 주장과 주요 기여 (간결 요약)

- 이 논문의 핵심 주장은 **여러 개(outliers ≤ k)의 이상치를 탐지하는 ESD(many-outlier) 절차를 일반화한 “generalized ESD many‑outlier test”를 제안**하여,
    - (i) **$H_0$** (outlier 없음) 뿐 아니라
    - (ii) 실제로 $l$개의 outlier가 존재하는 모든 대립가설 $H_l,\, l=1,\dots,k-1$
에서도, “추가로 더 많은 outlier를 선언할 확률(type I error)”을 유의수준 $\alpha$로 **동시에 제어**하도록 만든다는 점입니다.[^1_1]
- 두 번째 기여는 **일반화된 ESD 통계량의 임계값(percentage points)을 $t$-분포 변환으로 근사하는 방법**을 제안하고, **Monte Carlo 시뮬레이션으로 정확도를 검증**하여,
$n\ge 25, k\le 10$ 범위에서 **실용적으로 충분한 정확도**를 가지는 표를 제공했다는 점입니다.[^1_1]
- 세 번째 기여는 **기존 ESD many‑outlier 절차의 편향(실제 outlier 수보다 “과도 탐지”)을 줄이고, outlier 수를 더 정확하게 추정**한다는 점입니다. 이는 특히 **masking / swamping 문제를 줄이면서 일반화 가능한 검정 절차**를 제공한다는 의미에서, 이후 수많은 outlier·anomaly detection 연구의 기본 빌딩블록이 되었습니다.[^1_2][^1_3]

***

### 2. 논문이 다루는 문제, 제안 방법(수식 포함), “모델 구조”, 성능 및 한계

#### 2.1 해결하고자 하는 문제 (기존 ESD many‑outlier의 한계)

가정:

- 데이터 $x_1,\dots,x_n$는 **정규분포**에서 오며, 그 중 최대 $k$개까지 outlier일 수 있다고 가정.
- 목표: **1개 이상 $k$개 이하의 outlier를 탐지**하면서,
    - (i) outlier가 전혀 없을 때 잘못 탐지할 확률,
    - (ii) 실제로 $l$개 존재할 때 $l$개보다 더 많이 탐지할 확률
모두를 유의수준 $\alpha$로 제어하는 절차를 만드는 것.

기존 Rosner(1975)의 ESD many‑outlier 절차는, 통계량

$$
R_1,\dots,R_k
$$

(“extreme Studentized deviates”)를 정의하고, 각 $R_i$에 대해 같은 백분위수 수준 $\beta$를 쓰도록 설계되어 있습니다. 구체적으로 첫 번째 통계량은[^1_1]

$$
R_1 = \frac{\max_i |x_i - \bar{x}|}{s},
$$

$\bar{x}$는 표본평균, $s$는 표준편차입니다. 가장 극단값을 제거하고 남은 표본에서 같은 방식으로 $R_2, R_3, \dots$를 계산합니다.

기존 절차는 다음을 만족하도록 $\beta$와 임계값 $\lambda_i(\beta)$를 선택합니다.

$$
\Pr(R_i > \lambda_i(\beta)\mid H_0) = \beta,\quad i=1,\dots,k,
$$

$$
\Pr\left(\bigcup_{i=1}^k \{R_i > \lambda_i(\beta)\} \middle| H_0\right) = \alpha.
$$

그러나 이렇게 하면:

- **$H_0$** (outlier 없음)에서의 type I error는 $\alpha$로 잘 맞지만,
- 실제로 $l$개의 outlier가 존재할 때($H_l$)
$\Pr(\text{실제 이상치 개수 }l\text{보다 더 많이 선언})$가 $\alpha$보다 커지는,
즉 **“과도 탐지(swamping)”** 문제가 발생합니다.[^1_1]

따라서 논문이 해결하려는 핵심 문제는:

1. **모든 $H_l$ ($l=0,\dots,k-1$)에서 type I error를 $\alpha$로 제어**하는 many‑outlier 검정 설계,
2. 이 검정을 실무에서 쉽게 쓸 수 있도록 **폭넓은 $(n,k)$ 조합에 대한 임계값을 근사·표준화**하는 것입니다.

#### 2.2 일반화된 ESD many‑outlier 절차: 핵심 수식

먼저 기존과 동일하게, 다음과 같이 successively reduced sample을 정의합니다.

- $I_0$: 원 표본의 인덱스 집합,
- $I_1$: $R_1$ 계산 후 가장 극단값 하나 제거,
- $\dots$,
- $I_k$: $k$번 제거 후의 표본.

각 단계 $l$에서,

- 표본평균 $\bar{x}^{(l)}$,
- 표준편차 $s^{(l)}$,
- Studentized 값

$$
y_i = \frac{x_i - \bar{x}^{(l)}}{s^{(l)}},\quad i\in I_l
$$

를 정의하면, $R_{l+1} = \max_{i\in I_l} |y_i|$입니다.

이때, 논문이 제안하는 **“generalized ESD many‑outlier 절차”**는, 각 $l=0,\dots,k-1$에 대해 임계값 $\lambda_1,\dots,\lambda_k$를 다음을 만족하도록 선택합니다.[^1_1]

$$
\Pr\left( \bigcup_{i=l+1}^k \{R_i > \lambda_i\} \middle| H_l\right) = \alpha,\quad l=0,1,\dots,k-1.
$$

해석:

- $H_l$: 실제로 **정확히 $l$개의 outlier**가 존재하는 가설.
- $H_l$ 하에서 $R_{l+1},\dots,R_k$는 **남은 표본(정상 데이터 + 나머지 outlier 후보)에 기반한 통계량**.
- 식 (1)은
“이미 $l$개를 outlier로 선언했다면, 그 이후 단계에서 **추가 outlier를 잘못 선언할 확률**이 $\alpha$가 되도록 하라”는 제약입니다.

검정 규칙 자체는 기존 ESD와 구조가 거의 같습니다.

- $R_i \le \lambda_i$ 인 모든 $i=1,\dots,k$ 이면 “outlier 없음” 선언.
- 아니라면 $l = \max\{i : R_i > \lambda_i\}$로 두고,
“가장 극단값부터 $l$개를 outlier로 선언”합니다.


#### 2.3 임계값 $\lambda_i$의 $t$-분포 근사 (수식)

직접 $\Pr(\bigcup_{i=l+1}^k \{R_i > \lambda_i\}\mid H_l)$를 계산하는 것은 난해하므로, 논문은 다음 **근사 가설**을 둡니다.[^1_1]

$$
\Pr\left( \bigcap_{i=l+1}^k \{R_i \le \lambda_i\} \middle| H_l\right) 
\approx \Pr\left(R_{l+1} \le \lambda_{l+1}\mid H_l\right).
$$

즉, 여러 단계의 결합확률이 **실질적으로 첫 번째 새로운 통계량 $R_{l+1}$**에 의해 지배된다고 보는 가정입니다.

또한 Thompson(1935)의 결과에 따르면,
정규분포 표본에서 표본 평균·표준편차로 정규화한

$$
y_i = \frac{x_i - \bar{x}^{(l)}}{s^{(l)}}
$$

는 $t$-분포의 비선형 변환으로 근사됩니다.[^1_1]

$$
y_i \sim \frac{t_{n-l-2}\,(n-l-1)}{\sqrt{(n-l-2 + t_{n-l-2}^2)\,(n-l)}}.
$$

여기서 $t_{d}$는 자유도 $d$인 Student $t$-분포입니다.

이제 Bonferroni 근사

$$
\Pr\left(\max_{i\in I_l} y_i > \lambda_{l+1}\right) 
\approx (n-l)\Pr(y_i > \lambda_{l+1})
$$

을 적용하면, 두–단측(one-sided) 문제에서

$$
1 - \frac{\alpha}{n-l} = \Pr(y_i \le \lambda_{l+1})
$$

이 됩니다.

식 (3)–(4)를 결합하면, **one‑sided outlier 검정**에서

$$
\lambda_{l+1} =
\frac{t_{n-l-2,\,p}\,(n-l-1)}
{\sqrt{[\,n-l-2 + t_{n-l-2,\,p}^{2}\,]\,(n-l)}},
\quad p = 1 - \frac{\alpha}{n-l},
$$

를 얻습니다.[^1_1]

양측(two‑sided) 문제(논문에서 실제 사용하는 경우)에서는 $\alpha$ 대신 $\alpha/2$를 사용하여

$$
\lambda_{l+1} =
\frac{t_{n-l-2,\,p}\,(n-l-1)}
{\sqrt{[\,n-l-2 + t_{n-l-2,\,p}^{2}\,]\,(n-l)}},
\quad p = 1 - \frac{\alpha/2}{n-l}.
$$

이 식 (6)이 **generalized ESD 임계값 $\lambda_{l+1}$의 폐형식 근사**이며,
논문은 여기에 기반해 Table 3의 광범위한 percentage points를 계산합니다.[^1_1]

#### 2.4 절차(“모델 구조”) 요약

알고리즘 관점에서 generalized ESD는 다음 구조(모델)를 갖습니다.

1. **입력**
    - 데이터: $x_1,\dots,x_n$ (정규 분포 가정)
    - 최대 outlier 개수: $k$
    - 유의수준: $\alpha$
2. **초기화**
    - $I_0 = \{1,\dots,n\}$
    - 현재 표본 크기 $n_0 = n$
3. **반복 ( $i = 1,\dots,k$ )**

4. $\bar{x}^{(i-1)}, s^{(i-1)}$ 계산
5. $R_i = \max_{j\in I_{i-1}} |x_j - \bar{x}^{(i-1)}|/s^{(i-1)}$
6. 그 최대를 달성하는 지수 $j^\*$를 선택, $I_i = I_{i-1}\setminus\{j^*\}$
7. 식 (6)을 이용해 $\lambda_i$ 계산($l+1=i$로 치환)
1. **판정**
    - $l = \max\{i : R_i > \lambda_i\}$ (없으면 0)
    - $l=0$이면 outlier 없음,
    - $l\ge 1$이면, 제거된 순서 상 상위 $l$개를 outlier로 선언.

부록의 FORTRAN 서브루틴은 위 절차를 그대로 구현한 것입니다.[^1_1]

***

### 3. 성능 평가: 타입 I 에러, 검정력, “일반화 성능” 관점

#### 3.1 타입 I 에러 (Monte Carlo, Table 1)

논문은 $\alpha = 0.05$에서 다음 조건 하에 Monte Carlo 시뮬레이션을 수행합니다.[^1_1]

- $n = 10,15,20,25,30,50,100$,
- $k^* = \min(\lfloor n/2\rfloor, 10)$,
- 각 조합에 대해 2000회 반복,
- 데이터는 $N(0,1)$에서 생성,
- generalized ESD 절차(식 (1), (6) 사용)를 적용하고

$$
\hat{\alpha}(n,k) = \Pr\left(\bigcup_{i=1}^k \{R_i > \lambda_i\}\right)
$$

를 추정.

결과:

- $n \le 15$에서는 실제 $\hat{\alpha}$가 0.10 ~ 0.13 수준까지 올라가
**이론적 $\alpha=0.05$의 약 2 ~ 2.5배** 수준으로 오버슈트합니다.
- $n \ge 25$에서는 $\hat{\alpha} \approx 0.05$에 매우 근접하여,
**type I error가 잘 보정**됩니다.[^1_1]

⇒ **일반화 성능 관점**에서 보면,

- **표본 크기가 충분히 크면** ($n \ge 25$),
다양한 $k$ (최대 10)와 다양한 outlier 패턴에 대해
“nominal type I error를 안정적으로 유지”하는 **잘 일반화된 검정**이라고 볼 수 있습니다.
- 반대로 **작은 표본**에서는 t‑분포 근사와 Bonferroni 근사가 부정확해져,
“훈련 환경(대규모 Monte Carlo) 밖의 작은 $n$ 상황”으로 **일반화가 잘 안 되는** 셈입니다.


#### 3.2 파워(검정력) 비교: 기존 ESD vs Generalized ESD (Table 2)

$n=25, k=2$에서, 두 점 $x_{24}\sim N(\gamma_1,1)$, $x_{25}\sim N(\gamma_2,1)$에 인위적 shift를 주며 Monte Carlo를 수행했습니다.[^1_1]

- 1‑outlier case: $(\gamma_1,\gamma_2) = (0,2), (0,4), (0,6)$
    - 기존 ESD는 “2개 outlier 탐지” 확률이 0.04, 0.10, 0.12.
    - generalized ESD(GEN)는 각각 0.01, 0.04, 0.05.
⇒ **기존 ESD가 실제 1개인 상황에서 2개로 “과대 추정”하는 경향**이 훨씬 강함.
- 2‑outlier case: $(2,2), (2,4), (2,6), (4,4), (4,6), (6,6)$ 등
    - 실제 outlier 2개를 정확히 탐지할 power는
기존 ESD가 generalized ESD보다 대체로 더 큼.
(예: $(4,4)$에서 ESD의 “2개 정확 탐지”는 0.64, GEN은 0.48 정도.)[^1_1]

요약하면,

- **“outlier 수를 과대 선언하는 실수를 줄이는 것”**이 목표라면 GEN이 유리하고,
- **“실제로 outlier가 많이 존재할 때 최대한 많이 잡아내는 것”**이 목표라면 기존 ESD가 약간 더 강력합니다.

일반화 성능 측면에서는,

- GEN은 다양한 $H_l$ (실제 outlier 개수 $l$)에 대해 “잘못해서 더 많이 탐지하는 오류”를 $\alpha$로 제어하기 때문에,
**“실제 환경에서의 outlier 개수 분포가 다양할 때 더 안정적인 동작(robust generalization)”**을 합니다.
- 대신 다수 outlier 시에는 일부 파워를 희생한다는 **trade‑off**가 존재합니다.


#### 3.3 논문에서 지적한 한계

논문은 명시적으로 다음 한계를 언급합니다.[^1_1]

1. **작은 표본($n<25$)에서의 근사 부정확성**
    - $\hat{\alpha}$가 nominal $\alpha$보다 2배 이상 커질 수 있음.
    - 이는 **일반화 성능 저하**로 볼 수 있어,
작은 $n$ 영역에 대해서는 “수정된 근사”나 “직접 Monte Carlo 표 구축”이 필요.
2. **정규성 가정**
    - 절차 전체가 “정규분포 + Studentized deviate” 구조 위에서 설계됨.
    - 비정규 분포, heavy tail, skewness 환경에서의 **robustness/generalization**은 별도 연구 과제로 남김.
3. **robust estimator와의 비교 미비**
    - median, trimmed mean 같은 **robust 추정량 기반 방법들과의 체계적 비교**가 부족하다고 지적하며,
향후 연구 과제로 제시.
4. **비정규/다변량 상황의 확장 필요성**
    - 다변량 outlier, 비정규 GLM 등으로의 확장은 향후 과제로 남겨 둠.

***

### 4. “모델의 일반화 성능 향상 가능성”에 대한 해석

이 논문이 다루는 대상은 **예측 모델**이 아니라 **통계적 검정(검출기)**이지만, AI/ML 맥락에서 “일반화 성능”을 다음처럼 해석할 수 있습니다.

1. **데이터 정제 단계에서의 일반화**
    - 학습 데이터에서 generalized ESD로 outlier를 제거(또는 down‑weight)하면,
        - noise·센서 오류·입력 실수 등 **훈련 데이터의 label/feature contamination**을 줄일 수 있고,
        - 그 결과 **downstream 모델(회귀/분류/딥러닝)의 generalization error를 감소**시킬 수 있음.
    - 특히 “outlier 개수 상한 $k$만 주고, 실제 개수는 모르는 상황”에서
기존 ESD보다 **과대 제거를 덜 하므로**,
“진짜 유효한 rare but informative sample”이 덜 손실될 가능성이 큼.
2. **운영 단계(online monitoring)에서의 일반화**
    - 브라우저 로그, 시계열 모니터링 등에서 generalized ESD 기반 threshold를 쓰면
        - **다양한 outlier 빈도·패턴**에 대해
“거짓 경보(false alarm) 확률을 통제”하면서,
        - 데이터 분포가 어느 정도 바뀌어도
nominal $\alpha$ 기준의 동작을 유지하는 **distribution‑robust behavior**를 기대할 수 있음.
3. **근사 기반 구조의 이점**
    - 임계값이 $t$-분포 백분위수만으로 표현되므로,
        - 다양한 $n,k,\alpha$에 대해 **새로 Monte Carlo를 돌리지 않고도**
일반적으로 사용 가능(즉, **구현·적용 측면에서의 generalization**).
    - 이는 이후 여러 도메인(공정 제어, 시간 시리즈, 네트워크 트래픽, 스마트 그리드 등)으로
generalized ESD가 폭넓게 일반화·이식되는 기반이 됩니다.[^1_4][^1_5][^1_3]

***

### 5. 2020년 이후 관련 최신 연구 비교 분석 (open‑access 위주)

Rosner(1983)의 generalized ESD는 이후 **시계열·스트리밍 이상탐지, 통계적 outlier 검정 이론, AI 기반 anomaly detection pipeline**의 핵심 building block으로 많이 사용됩니다. 2020년 이후 open‑access 논문들 중, Rosner의 방법과 연관성이 높은 대표적인 것들을 정리하면 아래와 같습니다.

아래 목록은 **제목 – 저자 – 링크 – 1–2문장 요약** 형식이며, 이후 간단 비교 분석을 덧붙입니다.

***

#### 5.1 대표 논문·보고서 목록

1. **“generalized\_ESD: Detect outliers using the generalized Extreme Studentized Deviate (ESD) test” (R 패키지 modern)**
    - **Authors**: Foster Lab (R 패키지 문서)
    - **Link**: https://rdrr.io/github/fosterlab/modern/man/generalized_ESD.html[^1_6]
    - **요약**: Rosner(1983)의 generalized ESD를 그대로 구현한 R 함수로, **max\_outliers와 $\alpha$**를 인자로 받아 outlier 인덱스와 통계량을 반환. 바이오네트워크 추론 등에서 **robust network inference를 위한 전처리**로 사용.
2. **“Recursive ESD for streaming time series data” (비공식 명칭, arXiv:1911.05376v2, 2020)**
    - **Authors**: Shafiee 등
    - **Link**: https://arxiv.org/pdf/1911.05376.pdf[^1_7]
    - **요약**: time‑series에서 generalized ESD를 **슬라이딩 윈도우 + 재귀 업데이트** 구조로 만들어 “R‑ESD”라는 **streaming anomaly detection** 알고리즘을 제안. Twitter의 SH‑ESD보다 더 빠르고 안정적인 검출 성능을 보임.
3. **“Outlier Detection using AI: A Survey” (2021)**
    - **Authors**: 여러 저자
    - **Link**: https://arxiv.org/pdf/2112.00588.pdf[^1_8]
    - **요약**: 통계적·거리 기반·밀도 기반·딥러닝 기반 outlier detection을 광범위하게 정리한 survey. ESD 계열 검정(Grubbs, generalized ESD 등)을 **고전적 통계 기반 방법의 대표 예**로 소개.
4. **“Outlier Detection in Data Streams — A Comparative Study” (Procedia CS, 2021)**
    - **Authors**: Duraj 등
    - **Link**: (open PDF via ScienceDirect)[^1_9]
    - **요약**: 데이터 스트림에서 여러 통계 기반 알고리즘(Seasonal‑Hybrid ESD 등)을 비교. generalized ESD는 계절성 보정을 결합한 SH‑ESD 형식으로 시계열 이상탐지에 사용되며, **다른 anomaly detection 알고리즘 대비 장단점**을 실증적으로 비교.
5. **“Time Series Anomaly Detection for Smart Grids: A Survey” (arXiv:2107.08835, 2021)**
    - **Authors**: Tang 등
    - **Link**: https://arxiv.org/pdf/2107.08835.pdf[^1_5]
    - **요약**: 스마트 그리드 전력 데이터에서의 시계열 이상탐지 기법을 종합적으로 정리. generalized ESD는 **훈련 데이터 정제 및 이상 load 패턴 탐지를 위한 통계 기반 baseline**으로 사용됨.
6. **“LightESD: Fully-Automated and Lightweight Anomaly Detection ...” (arXiv, 약 2023)**
    - **Authors**: 여러 저자
    - **Link**: https://arxiv.org/pdf/2305.12266.pdf[^1_10]
    - **요약**: generalized ESD의 **정규성 가정과 deviant point 취약성**을 지적하고, 분포 가정 완화 및 threshold 조정을 통해 **데이터 분포에 더 강건한 LightESD**를 제안. 시계열 anomaly benchmark에서 기존 ESD/SH‑ESD보다 더 일관된 성능을 보여 “generalization 향상”을 주장.
7. **“Dive into Time-Series Anomaly Detection: A Decade Review” (arXiv, 2024)**
    - **Authors**: 여러 저자
    - **Link**: https://arxiv.org/html/2412.20512v1[^1_11]
    - **요약**: 지난 10년 간 시계열 anomaly detection 연구를 종합 review. Extreme Studentized Deviate (ESD) 계열 방법을 **단순·설명 가능한 통계 기반 baseline**으로 위치시키면서, deep learning 기반 방법과의 비교에서 **해석 가능성과 통계적 보장 측면의 장점**을 언급.
8. **“Hypothesis testing for detecting outlier evaluators” (2024)**
    - **Authors**: Li 등
    - **Link**: https://pmc.ncbi.nlm.nih.gov/articles/PMC11661559/[^1_12]
    - **요약**: 평가자 신뢰도 분석에서 “outlier evaluator”를 가설검정 방식으로 검출하는 방법을 제안하면서, Rosner(1983)의 generalized ESD를 **다중 outlier 검정의 기본 레퍼런스**로 인용. outlier evaluator 검정도 **type I error 제어를 명시적으로 설계**한다는 점에서 Rosner의 철학을 계승.
9. (참고: 2019, 2020 직전 연구지만 이론적으로 근접)
**“Multiple Outlier Detection Tests for Parametric Models” (Mathematics, 2019)**
    - **Authors**: Garren \& Kharin
    - **Link**: https://www.mdpi.com/2227-7390/8/12/2156[^1_13]
    - **요약**: 다양한 parametric family(정규, 지수 등)에 대해 **다중 outlier 검정**을 제안하고, Rosner의 generalized ESD와 함께 검정력·masking에 대한 성능 비교를 수행. 많은 상황에서 새로운 방법이 더 높은 파워를 보이나, generalized ESD는 여전히 **간단하고 잘 이해되는 baseline**으로 유지됨.

***

#### 5.2 Rosner(1983)와 이후 연구의 비교·분석

1. **스트리밍·시계열로의 확장 (R‑ESD, SH‑ESD, LightESD)**
    - Rosner는 **i.i.d. 정규 데이터의 batch 단일 표본**에서 many‑outlier를 다루었으나,
2020년대 연구는 이를 **시계열·데이터 스트림·계절성**까지 확장합니다.[^1_7][^1_5][^1_9]
    - R‑ESD는 generalized ESD 통계량을 **재귀 업데이트**로 계산하여,
streaming data에서도 **type I error를 제어하면서 실시간 이상탐지**를 가능케 합니다.[^1_7]
    - LightESD는 **정규성 가정 완화 및 deviant point 취약성 개선**을 통해,
다양한 분포·현실 데이터셋에서 **더 나은 “분포 간 generalization”**을 목표로 합니다.[^1_10]
2. **분포 가정과 robust·AI 모델과의 결합**
    - Rosner(1983)는 분명히 “정규성·독립성”을 가정하며, 비정규 상황의 robustness는 향후 과제로 남겼습니다.[^1_1]
    - 이후 AI/outlier detection 서베이들은 generalized ESD를[^1_11][^1_8]
        - **해석 가능하고 통계적으로 잘 이해된 baseline**으로 보면서,
        - 대규모·고차원·비선형 구조를 갖는 데이터에 대해서는
**deep generative model, representation learning, graph model** 등과 결합해야 한다고 강조합니다.
    - 예를 들어 robust regression + t‑기반 threshold로 outlier를 검출하는 최근 프레임워크들은,
Rosner식 “Studentized residual + t‑distribution + $\alpha$-기반 임계값” 철학을 계승하면서,
**모수 추정 단계에 robust learner를 넣어 generalization을 강화**합니다.[^1_14]
3. **many‑outlier 이론의 일반화**
    - MDPI Mathematics 논문은 다양한 parametric family에서 Rosner, Hawkins 등 고전 many‑outlier 검정들과 새 방법을 비교·분석합니다.[^1_13]
    - 결과적으로 generalized ESD는
        - 구현이 단순하고,
        - masking/swamping에 대한 특성이 잘 알려져 있으며,
        - broad한 $(n,k)$ 범위 테이블을 통해
여전히 **실무 baseline**으로 자리 유지.
    - 그러나 특정 분포(예: heavy‑tailed, skewed)나 특정 contamination 시나리오에서는 새 메서드가
**더 높은 검정력 및 더 나은 generalization**을 보여,
앞으로는 “Rosner test + 분포·모델 특화 many‑outlier test”의 **hybrid 사용**이 유망합니다.
4. **AI 관점에서의 일반화 성능**
    - Outlier Detection using AI 서베이 및 최근 시계열 anomaly review는,[^1_8][^1_11]
        - generalized ESD를 **단순·해석 가능한 thresholding layer**로 보고,
        - deep model이 생성한 residual·feature score 위에 generalized ESD (또는 변형)를 올려
**통계적 유의성 기반의 decision rule**을 구성하는 전략을 논의합니다.
    - 이는 Rosner의
“Studentized deviate + $t$-분포 + $\alpha$-기반 임계값” 구조가
**신경망 기반 representation 위에 얹혀도 여전히 유효**하다는 점에서,
**모델 generalization을 지키는 guardrail**로 기능합니다.

***

### 6. 앞으로의 연구에 미치는 영향과, 연구 시 고려할 점

#### 6.1 영향

1. **표준적인 many‑outlier 검정의 역할**
    - generalized ESD는 **다수 outlier 탐지에서 type I error를 전 가설($H_l$)에 대해 제어**하는
구조를 처음 정교하게 제시했습니다.[^1_1]
    - 이후 “다중 outlier 검정” 이론의 상당수가
        - (i) type I error의 전체적 제어,
        - (ii) masking/swamping trade‑off,
        - (iii) percent point 표 구축
측면에서 Rosner의 프레임을 기준점으로 사용하고 있습니다.[^1_13]
2. **도메인별 anomaly detection baseline**
    - NIST EDA 핸드북, 실험 설계 교과서, 시계열 anomaly detection 라이브러리 등은
generalized ESD를 **기본적인 outlier detection 툴**로 널리 채택하고 있습니다.[^1_3][^1_15][^1_2]
    - 시계열, 스마트 그리드, 네트워크 트래픽, 생물학 데이터 전처리 등에서
“정규성(또는 근사 정규성)이 어느 정도 성립”하는 상황이면
generalized ESD 또는 그 변형이 **여전히 강력한 baseline**입니다.[^1_4][^1_5][^1_6]
3. **AI/ML에서 generalization 관리용 통계 계층**
    - 딥러닝/ML 모델의 출력(residual, confidence score 등)에
generalized ESD 계층을 얹어
        - **OOD(out‑of‑distribution) 샘플**,
        - **distribution shift 전조**,
        - **label noise**
를 검출·제거하는 방향의 연구가 활발합니다.[^1_16][^1_17][^1_11]
    - 이는 단순 threshold를 넘어, **명시적인 $\alpha$ 기반 generalization 통제**를 제공한다는 점에서
안전성·신뢰성 연구 방향과 직결됩니다.

#### 6.2 앞으로 연구 시 고려할 점 (특히 AI 연구자의 시각에서)

1. **작은 표본·비정규 분포에 대한 개선**
    - Rosner가 명시한 한계: $n<25$에서 type I error가 크게 inflate 됨.[^1_1]
    - 향후 연구 과제:
        - 작은 $n$에서의 **정확한 percentile 테이블** (고성능 Monte Carlo / quasi‑MC / importance sampling),
        - 혹은 **bootstrap‑기반 재보정**으로 실질적 $\hat{\alpha}$를 target $\alpha$에 맞추는 방법.
        - heavy‑tailed / skewed 분포에 대해
generalized ESD를 **robust variance estimator + 변환**과 결합하는 방안.
2. **다변량·고차원 데이터로의 확장**
    - Caroni \& Prescott(1992) 등에서는 multivariate extension이 제안되었지만,
고차원 / sparse / graph‑structured 데이터에 대한 체계적 many‑outlier 이론은 아직 부족합니다.[^1_18]
    - AI 맥락에서는,
        - representation space에서 Mahalanobis distance나 graph Laplacian residual에 generalized ESD를 적용하는
“many‑outlier in representation space”를 설계할 수 있음.
        - 이때 **공분산 추정의 불안정성**과 **multiple testing 문제**를 동시에 다루는 이론이 필요.
3. **스트리밍·비정상(non‑stationary) 환경에서의 이론화**
    - R‑ESD, LightESD 등은 실무적으로 좋은 성능을 보이지만,[^1_19][^1_10][^1_7]
“일련의 streaming test들에서 장기적으로 type I error를 어떻게 제어할 것인가?”에 대한
엄밀한 이론은 아직 제한적입니다.
    - AI 연구자는
        - **concept drift + point outlier**를 동시에 고려한 framework에서,
        - generalized ESD 계열 검정을 **channel 분리(예: residual channel vs drift channel)** 구조로 통합하는 연구를 고려할 수 있습니다.[^1_14]
4. **robust estimator·베이지안·graphical model과의 통합**
    - robust regression, high‑breakdown multivariate method, mixed graphical model 기반 outlier 검정 등과
generalized ESD의 결합은 기대가 큽니다.[^1_20][^1_21]
    - 예:
        - robust GLM으로 residual을 얻고, 그 residual에 generalized ESD를 적용
→ 모델링 단계에서 비정규성/contamination을 처리하고,
검정 단계에서 통계적 유의성을 정량화.
        - 베이지안 outlier model에서 posterior predictive residual에 generalized ESD를 얹어
**Bayes + frequentist hybrid** 형태의 anomaly detector 설계.
5. **AI generalization 관점에서의 활용 전략**

연구자가 딥러닝/ML 모델의 일반화 성능 관점에서 generalized ESD를 사용할 때 고려할 포인트는 다음과 같습니다.

- **학습 데이터 정제**
    - generalized ESD를 label별·feature별로 적용해 **극단값을 제거/재가중**하면,
training set contamination을 줄여 test 성능을 높일 수 있음.
    - 다만 이때도 **정규성 가정이 대략 맞는 residual space**에서 사용하는 것이 바람직.
- **validation 단계에서의 robust metric**
    - validation residual에 generalized ESD를 적용해 **극단 validation sample을 제거한 후**
성능을 기록하면, hyper‑parameter 선택이 “rare but pathological 사례”에 과도하게 끌려가지 않게 할 수 있음.
- **online monitoring \& OOD detection**
    - 실제 배포 환경에서 모델 residual score에 generalized(or Light) ESD를 적용해,
**type I controlled OOD alarm 시스템**을 설계할 수 있음.
    - 이때 Rosner식 $\alpha$-제어는, **false alarm budget을 명시적으로 설정**하는 역할을 함.

***

정리하면, Rosner(1983)의 generalized ESD many‑outlier 절차는

- “여러 개 outlier를 동시에 고려하면서, 모든 관련 가설($H_l$)에 대해 type I error를 제어하는 검정”을 처음 체계적으로 제시했고,
- $t$-분포 기반 근사로 광범위한 $(n,k)$ 조합에서 실용적인 임계값을 제공했으며,
- 이후 스트리밍·시계열·AI 기반 anomaly detection까지 폭넓게 **일반화·확장**되었습니다.

향후 연구에서는 **작은 표본·비정규성·고차원·스트리밍**이라는 네 가지 축에서
이 절차를 어떻게 수정·통합할지, 그리고 **deep model의 generalization을 보장하는 통계 계층**으로
어떻게 활용할지가 중요한 과제가 될 것입니다.
<span style="display:none">[^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37]</span>

<div align="center">⁂</div>

[^1_1]: https://www.stat.cmu.edu/technometrics/80-89/VOL-25-02/v2502165.pdf

[^1_2]: https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/esd.htm

[^1_3]: https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h3.htm

[^1_4]: https://pdfs.semanticscholar.org/1748/af3ecbb33ff6025c70d3048abe4fce8d835b.pdf

[^1_5]: https://arxiv.org/pdf/2107.08835.pdf

[^1_6]: https://rdrr.io/github/fosterlab/modern/man/generalized_ESD.html

[^1_7]: https://arxiv.org/pdf/1911.05376.pdf

[^1_8]: https://arxiv.org/pdf/2112.00588.pdf

[^1_9]: https://www.sciencedirect.com/science/article/pii/S1877050921017841/pdf?md5=75a2d4a311c239e863a983f1f1b98424\&pid=1-s2.0-S1877050921017841-main.pdf

[^1_10]: https://arxiv.org/pdf/2305.12266.pdf

[^1_11]: https://arxiv.org/html/2412.20512v1

[^1_12]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11661559/

[^1_13]: https://www.mdpi.com/2227-7390/8/12/2156

[^1_14]: https://arxiv.org/html/2512.12289v1

[^1_15]: https://pyastronomy.readthedocs.io/en/latest/pyaslDoc/aslDoc/outlier.html

[^1_16]: https://arxiv.org/pdf/2202.01197.pdf

[^1_17]: https://arxiv.org/pdf/2008.09245.pdf

[^1_18]: https://www.tqmp.org/RegularArticles/vol08-2/p108/p108.pdf

[^1_19]: https://arxiv.org/html/2502.18038v1

[^1_20]: https://arxiv.org/pdf/0808.0657.pdf

[^1_21]: https://arxiv.org/pdf/2103.02366.pdf

[^1_22]: v2502165.pdf

[^1_23]: https://arxiv.org/pdf/1910.10426.pdf

[^1_24]: http://arxiv.org/pdf/1912.02724.pdf

[^1_25]: https://arxiv.org/pdf/2502.08593.pdf

[^1_26]: https://arxiv.org/pdf/1812.09178.pdf

[^1_27]: https://www.biorxiv.org/content/10.1101/2022.11.14.516449v1.full-text

[^1_28]: https://www.biorxiv.org/content/10.1101/2022.11.14.516449v2.full-text

[^1_29]: https://arxiv.org/pdf/2102.09350.pdf

[^1_30]: https://www.biorxiv.org/content/10.1101/2022.11.14.516449v1.full.pdf

[^1_31]: https://pdfs.semanticscholar.org/5322/8d880f128509366128ebc7816d02ecf21962.pdf

[^1_32]: https://real-statistics.com/students-t-distribution/identifying-outliers-using-t-distribution/generalized-extreme-studentized-deviate-test/

[^1_33]: https://www.graphpad.com/quickcalcs/grubbs1/

[^1_34]: https://rdrr.io/github/skinnider/modern/man/generalized_ESD.html

[^1_35]: https://www.youtube.com/watch?v=KGWbbAUcC0I

[^1_36]: https://pro.arcgis.com/en/pro-app/latest/tool-reference/space-time-pattern-mining/understanding-outliers-in-time-series-analysis.htm

[^1_37]: https://ouci.dntb.gov.ua/en/works/7WaVoP5l/

