# Chapter 2. Looking at Multivariate Data: Visualisation

> 교재 범위: Chapter 2, pp. 25–60.  
> 핵심 주제: scatterplot, bivariate boxplot, convex hull, chi-plot, glyph, scatterplot matrix, kernel density estimation, 3-D plot, trellis graphics, stalactite plot.

## 1. Executive Summary — 10문장 이내

1. 이 장의 핵심은 다변량 분석에서 그래프가 단순한 장식이 아니라 **모델을 만들기 전에 구조·이상치·비선형성을 발견하는 분석 도구**라는 것입니다.
2. 가장 기본적인 산점도는 두 변수 관계를 보여주며, 상관계수와 반드시 함께 보아야 이상치가 상관을 왜곡하는지 확인할 수 있습니다.
3. bivariate boxplot과 convex hull은 이변량 이상치에 더 구조적인 기준을 제공하지만, 이상치 제거 자체가 자동적으로 정당화되는 것은 아닙니다.
4. scatterplot matrix는 여러 변수의 모든 쌍 관계를 한 화면에서 비교하여 고차원 데이터의 비선형성·공선성·이상치를 찾는 데 유용합니다.
5. kernel density estimator(KDE)는 점의 밀도를 연속적인 확률밀도 형태로 근사하여 잠재적 군집을 시각적으로 드러낼 수 있습니다.
6. KDE에서는 kernel 모양보다 bandwidth $h$의 선택이 결과에 더 큰 영향을 주는 경우가 많습니다.
7. 3차원·glyph·star plot은 더 많은 변수를 한 번에 표시할 수 있지만, 정보량 증가가 곧 이해도 증가를 의미하지는 않습니다.
8. trellis graphics는 조건부 관계를 패널별로 분리하여 Simpson-type 혼합 효과를 발견하는 데 유용합니다.
9. 교재의 사례는 이상치 처리에 따라 상관이 $0.9553$에서 $0.7956$으로 크게 변할 수 있음을 직접 보여줍니다.
10. 2020년 이후 PaCMAP과 densMAP 같은 방법은 고차원 임베딩에서 local/global structure와 density 보존 문제를 다루지만, 교재의 직접 시각화와 목적함수가 달라 성능 수치를 단순 비교할 수 없습니다.

## 2. 목적과 필요성

다변량 데이터는 숫자 표만 보고 구조를 파악하기 어렵습니다. 예를 들어 상관행렬에 $r=0.8$이 적혀 있어도 실제 산점도를 보면 한 개의 극단점 때문에 상관이 커졌을 수 있고, 두 개의 군집이 섞여 생긴 가짜 선형관계일 수도 있습니다. 따라서 이 장의 목적은 **수치 요약 이전 또는 동시에 시각적 진단을 수행하는 습관**을 만드는 데 있습니다.

교재는 그래프의 목적을 크게 다음과 같이 봅니다: overview, story, hypothesis suggestion, model criticism. 이 장은 주로 앞의 세 가지를 다룹니다.

**용어 설명 — exploratory visualization**  
사전에 하나의 엄격한 가설만 검정하기보다 데이터가 가진 예상 밖 구조를 발견하기 위한 시각화입니다. 이 때문에 그래프에서 발견한 패턴은 후속 독립 검증 없이 확정적 결론으로 취급하면 안 됩니다.

## 3. Scatterplot과 상관계수

두 연속형 변수 $X,Y$의 산점도는 관측점 $(x_i,y_i)$를 직접 그립니다. 산점도를 통해 최소한 다음을 확인해야 합니다.

- 선형인지 비선형인지
- 분산이 $X$에 따라 달라지는지
- 한두 개 점이 관계를 지배하는지
- 두 개 이상의 군집이 섞였는지
- 범위 제한(range restriction)이 있는지

교재의 US air pollution 예제에서 manufacturing(`manu`)과 population(`popul`)의 Pearson 상관은 전체 데이터를 사용하면

$$
r=0.9553
$$

이지만 Chicago, Philadelphia, Detroit, Cleveland를 bivariate boxplot에서 이상치로 판단하여 제외하면

$$
r=0.7956
$$

으로 줄어듭니다.

이 결과는 “상관계수는 데이터의 그림 없이 해석하면 위험하다”는 점을 수치로 보여줍니다.

## 4. Bivariate Boxplot

단변량 boxplot을 2차원으로 확장한 개념으로, robust location, scale, correlation을 이용하여 중심부와 잠재적 이상치 경계를 타원으로 표현합니다. 교재의 설명에서 내부 `hinge`는 대략 50% 데이터를 포함하고, 외부 `fence` 밖의 점은 잠재적 이상치로 봅니다.

> 중심점 (Robust Location): 데이터의 중심을 나타내며, 이상치에 영향을 받지 않는 로버스트(Robust)한 중앙값 공간(예: Medianpolish 또는 최소공분산결정(MCD) 중심)을 사용합니다.

**용어 설명 — robust estimator**  
소수의 극단값이 들어와도 추정치가 크게 흔들리지 않도록 설계한 추정량입니다. 평균 대신 중앙값, 표준편차 대신 MAD를 쓰는 것이 대표적인 아이디어입니다.

### 왜 robust가 필요한가?

일반 평균과 공분산을 이용해 이상치를 찾으면 이상치가 평균과 공분산 자체를 끌어당겨 자신의 거리를 작게 만들 수 있습니다. 이를 **masking**이라고 합니다.

**용어 설명 — masking**  
이상치가 추정된 중심과 분산을 왜곡하여 자신 또는 다른 이상치가 정상처럼 보이게 만드는 현상입니다.

<img width="692" height="563" alt="image" src="https://github.com/user-attachments/assets/7f25fc82-54a7-409a-8334-f2f71affceaa" />

> 시각화 그래프 해석 가이드
> - 🟢 로버스트 중심 (Robust Center): 우측 상단 (5, 5)에 무리지어 있는 이상치 집단에 전혀 끌려가지 않고, 실제 데이터의 청정 구역 한가운데에 정확히 위치합니다.
> - 🔵 내부 힌지 (Inner Hinge, 푸른색 점선 타원): 데이터의 대략 50%를 감싸는 중심부 상자 영역입니다. 단변량 박스플롯의 IQR 상자에 대응됩니다.
> - 🔴 외부 펜스 (Outer Fence, 붉은색 실선 타원): 통계적 확률(카이제곱 분포 기준 97.5%~99%)에 따라 설정된 이상치 차단 경계선입니다. 단변량 박스플롯의 수염(Whiskers) 끝부분에 해당합니다.
> - ❌ 잠재적 이상치 (Outliers, 붉은색 X 표시): 일반 공분산 행렬을 썼다면 타원을 찌그러뜨려 마스킹 뒤에 숨었을 변칙 데이터들이 로버스트 타원 밖으로 완벽하게 격리·검출되었습니다.

## 5. Convex Hull Trimming

2차원 점들을 모두 포함하는 가장 작은 볼록다각형의 꼭짓점을 제거한 뒤 상관을 다시 계산하는 방식입니다. 교재 예제에서는 convex hull을 제거한 상관이

$$
r=0.9225
$$

였습니다.

이 값은 bivariate boxplot으로 선택한 네 도시를 제거했을 때의 $0.7956$과 다릅니다. 즉 “어떤 점이 outlier인가?”라는 정의가 바뀌면 robust correlation도 달라집니다.

**통계적 교훈**  
이상치 처리 방법 자체가 분석자의 modeling choice이므로, 한 가지 처리 결과만 제시하기보다 원자료·robust 분석·제외 분석을 함께 보고 sensitivity를 확인하는 편이 안전합니다.

> Convex Hull Trimming(볼록 껍질 다듬기)은 데이터 세트에서 이상치(Outlier)를 제거하고 데이터의 본질적인 상관관계를 분석하기 위해 사용하는 기하학적 데이터 전처리 기법입니다. 데이터의 중심축에서 멀리 떨어진, 분포의 가장자리에 위치한 극단적인 값들을 효과적으로 제거합니다. 특정 변수 하나만의 극단값이 아니라, 두 변수의 관계(2차원 공간) 속에서 외곽에 존재하는 이상치를 유기적으로 제거할 수 있습니다.

## 6. Chi-plot: 독립성의 시각적 진단

Chi-plot은 두 변수의 독립성 아래에서 transformed statistic $\chi_i$가 0 주변에서 비체계적으로 움직여야 한다는 성질을 이용합니다. 함께 쓰이는 $\lambda_i$는 각 점이 이변량 분포의 중심에서 얼마나 떨어졌는지를 나타냅니다.

교재의 `manu`와 `popul` 예제에서는 chi-plot의 중앙 독립성 band에 점이 거의 없어 독립에서 명확히 벗어나는 것으로 해석합니다.

**용어 설명 — independence**  
$X$를 안다고 해서 $Y$의 분포가 달라지지 않는 상태입니다. 상관 0보다 더 강한 개념입니다. 독립이면 적절한 조건 아래 공분산 0이지만, 공분산 0이 독립을 의미하지는 않습니다.

<img width="1491" height="1330" alt="image" src="https://github.com/user-attachments/assets/23af8992-eaa5-4780-907d-2b90243030b0" />

> 핵심 축의 의미
> - $\(\chi _{i}\)$ (세로축): 두 변수가 서로 독립인지 아닌지를 나타내는 변환된 통계량입니다. 두 변수가 완전히 독립이라면 $\(\chi _{i}\)$ 값들은 0을 중심으로 아무런 패턴 없이 무작위(비체계적)로 분포하게 됩니다. 만약 0에서 벗어나 특정한 형태(U자, 역U자, 직선 등)를 띤다면 독립성이 깨졌음을 의미합니다.
> - $\(\lambda _{i}\)$ (가로축): 각 데이터 포인트가 이변량 분포의 중심(Center)으로부터 얼마나 멀리 떨어져 있는지를 나타내는 척도입니다. 값의 범위는 -1에서 1 사이이며, 0에 가까울수록 중심에 위치하고 $\(\pm 1\)$ 에 가까울수록 외곽(꼬리 부분)에 위치함을 뜻합니다.

> 독립 데이터 (위쪽 그래프) : $\(\chi_{i}\)$ 값이 빨간색 점선(신뢰 한계선, Control Limits: 약 ± 0.15) 내부에 무체계적이고 무작위로 모여 있습니다. 이는 두 변수가 서로 독립임을 증명합니다.

> 양의 상관관계 데이터 (아래쪽 그래프) : 점들이 신뢰 한계선을 뚫고 상단의 윗부분 $(\(\chi_i > 0\))$ 으로 한데 쏠려 이동해 있습니다. 특히 중심에서 멀어질수록 $(\(\lambda _{i}\)$ 가 ± 1에 가까워질수록) $\(\chi _{i}\)$ 통계량이 1에 가까운 최상단에 밀집하는 독특한 패턴을 보여줍니다.

## 7. Scatterplot Matrix

$q$개 변수라면 가능한 변수쌍은

$$
\binom{q}{2}=\frac{q(q-1)}{2}
$$

개입니다. scatterplot matrix는 이들을 $q\times q$ 격자에 배치합니다.

- $q$: 변수 수입니다.
- $\binom{q}{2}$: 서로 다른 두 변수를 고르는 조합 수입니다.

교재의 air pollution 데이터에서는 `manu`와 `popul`의 강한 관계, `SO2`와 이 변수들의 관계, `precip`·`predays`와 `SO2` 사이의 비선형 가능성이 동시에 드러납니다. 이는 correlation matrix만으로는 놓칠 수 있는 정보입니다.

### 모델링으로 연결

산점도에서 곡률이 보인다면 회귀모형에 $x^2$, spline, kernel term을 고려할 근거가 생깁니다. 따라서 visualization은 단순 EDA가 아니라 **feature engineering의 근거**가 됩니다.

> 대각선 원소 (Diagonal): $\(q \times q\)$ 격자에서 행과 열이 같은 대각선 자리는 자기 자신과의 산점도가 되므로, 보통 산점도 대신 변수명이나 해당 변수의 히스토그램(Histogram) 또는 밀도 그림(Density Plot)을 배치합니다.

> 대각선을 기준으로 위쪽(상삼각)과 아래쪽(하삼각)은 x축과 y축만 바뀐 대칭적인 그림이 됩니다. 예를 들어 1번 변수와 2번 변수의 산점도는 (1행 2열)과 (2행 1열)에 각각 축이 바뀐 채로 두 번 나타납니다.

> 대칭되는 부분을 제외하고 서로 다른 변수 쌍만 순수하게 그린 개수가 $\(\binom{q}{2} = \frac{q(q-1)}{2}\)$ 개가 됩니다.

<img width="590" height="575" alt="image" src="https://github.com/user-attachments/assets/4a1fd62d-44cd-4803-9cc6-d58cabb24a15" />

> 시각화 격자 구조 해설
> - 대각선 : 자기 자신과의 관계이므로 보통 해당 변수의 분포(히스토그램 또는 밀도 곡선)나 변수 이름이 들어갑니다.

> - 하삼각 : 우리가 찾고자 하는 서로 다른 변수 쌍 3개의 산점도입니다.

> - 상삼각 : 하삼각과 축(X축, Y축)만 바뀐 채 대칭을 이루는 똑같은 변수 쌍의 산점도입니다. (도구에 따라 이 자리에 상관계수 숫자를 띄우기도 합니다.)

변수의 개수 $(\(q\))$ 가 너무 많아지면 격자가 촘촘해져 시각적으로 확인하기 어려워지므로, 보통 5~10개 이하의 변수를 분석할 때 가장 유용합니다.

## 8. Kernel Density Estimation

### 8.1 1차원 KDE

교재가 소개하는 kernel density estimator는

$$
\hat f(x)=\frac{1}{nh}\sum_{i=1}^{n}K\left(\frac{x-x_i}{h}\right)
$$

입니다.

- $\hat f(x)$: 데이터에서 추정한 밀도입니다.
- $K(\cdot)$: kernel function입니다.
- $h>0$: bandwidth 또는 smoothing parameter입니다.
- $n$: 표본 수입니다.
- $x_i$: 실제 관측값입니다.

Gaussian kernel은

$$
K(u)=\frac{1}{\sqrt{2\pi}}e^{-u^2/2}
$$

입니다.

각 관측치 $x_i$ 위에 작은 봉우리 하나를 올리고 모두 합친다고 생각하면 됩니다.

### 8.2 Bandwidth의 역할

- $h$가 너무 작음: 작은 잡음까지 여러 peak로 표현되어 overfitting처럼 보입니다.
- $h$가 너무 큼: 실제 두 군집도 하나의 완만한 봉우리로 합쳐집니다.

**용어 설명 — bandwidth**  
각 관측치의 영향이 주변으로 얼마나 넓게 퍼질지를 정하는 폭입니다. KDE에서는 kernel 종류보다 bandwidth 선택이 결과 모양을 크게 좌우하는 경우가 많습니다.

> 구간을 나누는 대신, 모든 개별 데이터 $(\(x_{i}\))$ 의 위치에 부드러운 봉우리(커널 함수)를 하나씩 세웁니다. 데이터가 몰려 있는 곳은 봉우리들이 겹치면서 더 높아지고, 데이터가 없는 곳은 낮아집니다. 이 봉우리들을 모두 더한 뒤 평균을 내면 매끄러운 곡선이 완성됩니다.

### 8.3 2차원 KDE

이변량 데이터 $(x_i,y_i)$에서는

$$
\hat f(x,y)
=\frac{1}{nh_xh_y}\sum_{i=1}^{n}
K\left(\frac{x-x_i}{h_x},\frac{y-y_i}{h_y}\right)
$$

를 사용합니다.

- $h_x,h_y$: 각 좌표 방향의 smoothing 폭입니다.
- $K(u,v)$: 2차원 kernel입니다.

표준 이변량 Gaussian kernel은

$$
K(u,v)=\frac{1}{2\pi}\exp\left[-\frac12(u^2+v^2)\right]
$$

입니다.

> 1차원 KDE가 데이터 밀도를 '매끄러운 곡선'으로 나타냈다면, 2차원 KDE는 '부드러운 지형도(산)'를 만듭니다.
> - 평면 위 데이터 지점마다 표준 이변량 가우시안 커널(둥근 종 모양의 천막)을 하나씩 칩니다.
> - 데이터가 빽빽하게 모여 있는 곳은 천막들이 겹치면서 거대한 산(높은 밀도)을 이루고, 데이터가 없는 곳은 평지가 됩니다.
> - 이 지형을 위에서 내려다보며 높이가 같은 곳을 선으로 이으면 우리가 흔히 보는 밀도 등고선(Contour plot)이 됩니다.

<img width="1764" height="730" alt="image" src="https://github.com/user-attachments/assets/1c17470c-c8e6-46fd-ac16-c354c576ee83" />

> 시각화 그래프 해석

> - 왼쪽 (2D 등고선과 산점도): 검은색 점들은 실제 관측값 $\((x_i, y_i)\)$ 입니다.
>   - 점들이 빽빽하게 뭉쳐 있는 중심부로 갈수록 연두색(높은 밀도값 f̂(x,y))을 띱니다.
>   - 데이터가 없는 외곽 지역은 짙은 남색(낮은 밀도)으로 표현됩니다.
>   - 위에서 내려다본 이 등고선이 바로 2차원 KDE의 대표적인 시각화 형태입니다.

> - 오른쪽 (3D 밀도 표면): 관측값 위에 올린 작은 가우시안 봉우리들이 모두 합쳐져 만들어진 3차원 지형도입니다.
>   - 데이터가 밀집된 두 구역이 각각 하나의 거대한 '산봉우리'를 형성하고 있는 모습을 직관적으로 확인할 수 있습니다.

## 9. KDE의 모델 구조와 한계

KDE는 학습 가능한 복잡한 neural network가 아니라 **비모수 밀도 추정기**입니다. 전체 구조는 다음과 같습니다.

```text
관측값
  ↓
각 관측치 중심에 kernel 배치
  ↓
bandwidth로 넓이 결정
  ↓
모든 kernel 평균
  ↓
연속적인 밀도 추정 \hat f
```

### 장점

- 정규분포 같은 특정 parametric shape를 강제하지 않습니다.
- multimodality를 시각화할 수 있습니다.
- 군집분석 전 데이터 구조를 점검하기 좋습니다.

### 한계

- 고차원에서는 필요한 표본 수가 급격히 늘어나는 curse of dimensionality가 있습니다.
- bandwidth에 민감합니다.
- density peak가 곧 실제 생성집단이라는 증거는 아닙니다.

**용어 설명 — curse of dimensionality**  
차원이 증가하면 공간의 부피가 매우 빠르게 커져 데이터가 희박해지고, 근접도·밀도 추정에 필요한 표본 수가 급격히 증가하는 현상입니다.

> KDE 수식을 보면 차원 d 가 대역폭의 지수 $(\(h^{d}\))$ 로 들어갑니다. 고차원 공간 전체를 커버하기 위해 대역폭 $\(h\)$ 를 키우면 밀도가 너무 뭉뚱그려져(Over-smoothing) 데이터의 세부 특징을 잃어버립니다. 반대로 특징을 잡기 위해 $\(h\)$ 를 작게 유지하면, 데이터가 없는 텅 빈 공간에서는 밀도가 0으로 수렴해 버리는 현상이 발생합니다.

## 10. Bubble/Glyph/3-D Plot: 정보량과 인지부하의 충돌

Bubble plot은 $(x,y)$ 위치에 세 번째 변수의 크기를 원의 반지름 등으로 표현합니다. star plot이나 Chernoff face는 더 많은 변수를 symbol shape에 매핑합니다. 그러나 교재는 이런 그래프가 “많은 변수를 한 번에 보인다”는 이유만으로 효과적인 것은 아니라고 분명히 지적합니다.

> 버블 플롯에서 변수 크기를 반지름에 매핑하면, 데이터가 2배 커질 때 원의 면적은 4배로 커져 시청자가 효과를 과장되게 인지합니다. (면적에 매핑하더라도 인간은 면적의 비례 관계를 정확히 계산하지 못합니다.) 스타 플롯의 뾰족한 모양이나 체르노프 페이스의 눈 크기, 입 모양 등은 ‘위치’에 비해 인간의 눈이 그 정량적인 차이를 미세하게 식별하기 어렵습니다.

> 모든 변수가 동등한 중요도를 가짐에도 불구하고, 어떤 시각적 요소에 매핑되느냐에 따라 특정 변수가 더 과장되어 보일 수 있습니다. (체르노프 페이스에서 '입의 미소 짓는 각도'나 '눈의 크기'는 인간이 진화론적으로 가장 먼저 주목하는 시각 자극(얼굴 표정)입니다. 반면 '귀의 크기'나 '머리카락 길이'는 상대적으로 덜 주목받습니다.)

> 스타 플롯은 다각형의 꼭짓점 순서를 어떻게 배치하느냐에 따라 전체적인 기하학적 형태가 완전히 달라집니다. (똑같은 데이터임에도 변수 순서만 바꾸면 완전히 다른 도형처럼 보이기 때문에, 데이터의 본질적인 패턴이 아니라 '우연히 정해진 변수 순서'에 의해 시청자가 오도될 위험이 큽니다.)

<img width="1742" height="490" alt="image" src="https://github.com/user-attachments/assets/eb192742-9588-4ee1-a605-4f7e46c86060" />

> 1. Star Plot (스타 플롯) : Data A의 닫힌 도형 면적을 보시면 Data B나 Data C에 비해 압도적으로 거대해 보입니다. 인간의 눈은 각 축의 개별 수치(Feature 3이 낮다는 점 등)보다 전체 다각형의 면적과 부피를 먼저 보기 때문에, 데이터가 실제보다 왜곡되어 과장되게 인지됩니다.
> 2. Chernoff Face Concept : (체르노프 페이스 개념 플롯)수치를 얼굴 대신 원의 크기(눈 크기 대용)와 색상 변화(입 모양 대용)로 치환해 보았습니다. Feature 3이 커서 원의 크기가 거대한 Data B와 색상이 붉은 Data C에 직관적으로 눈길이 먼저 쏠립니다. 정작 가장 핵심적인 수치인 수직 위치(Feature 1)의 차이는 형태적 자극에 묻혀 정량적 비교가 불가능해집니다.
> 3. Parallel Coordinates : (평행 좌표계 — 대안) 평행 좌표계(Parallel Coordinates)는 수많은 변수를 가진 다변량 데이터를 왜곡 없이 한눈에 파악하기 위해 고안된 대표적인 다차원 시각화 기법입니다. 4개의 수직 축에 데이터 값이 꺾은선으로 정직하게 표현됩니다. 면적이나 심볼 크기의 착시 없이, 어떤 데이터가 어떤 변수에서 높고 낮은지(예: Data A는 Feature 1, 2가 높고 3이 낮음)를 '공간상의 위치' 채널만을 활용해 가장 정확하게 식별할 수 있습니다.

3-D plot도 시점(rotation)에 따라 관계가 다르게 보이고, 정적인 인쇄물에서는 depth 판단이 어렵습니다. 따라서 고차원 정보를 억지로 3-D에 넣기보다 **차원축소 후 2-D 표시**가 더 나을 수 있으며, 이것이 다음 장 PCA로 자연스럽게 이어집니다.

> PCA는 무작정 데이터를 구겨 넣는 것이 아니라, "가장 중요한 방향"을 찾아 그 축으로 데이터를 투영(Projection)합니다.
> - 분산(Variance)의 최대화: 데이터가 가장 널리 퍼져 있는 축(정보량이 가장 많은 축)을 찾아 PC1(제1주성분)로 지정합니다.
> - 직교성(Orthogonality): PC1과 겹치지 않는(수직인) 나머지 방향 중 가장 분산이 큰 축을 PC2(제2주성분)로 지정합니다.
> - 2D 시각화: 이렇게 찾은 PC1과 PC2를 X축, Y축으로 삼아 데이터를 뿌리면, 고차원 데이터의 전체적인 실루엣(클러스터, 트렌드)을 왜곡이 가장 적은 2D 화면으로 관찰할 수 있습니다.

## 11. Trellis Graphics와 조건부 관계

Trellis는 어떤 변수를 구간별로 나눈 뒤 동일한 그래프를 여러 panel에서 반복합니다. 교재의 air pollution 예제에서는 wind를 두 범위로 나누고 `SO2 ~ temp` 관계를 비교했을 때, 낮은 wind 구간에서는 온도가 증가할수록 오염이 감소하는 경향이 보이지만 높은 wind 구간에서는 관계가 약해 보인다고 설명합니다.

이것은 단일 전체 산점도에서 숨겨질 수 있는 **effect modification**을 찾는 방식입니다.

**용어 설명 — effect modification / interaction**  
$X$가 $Y$에 미치는 관계의 크기나 방향이 세 번째 변수 $Z$의 수준에 따라 달라지는 현상입니다. 회귀식에서는 흔히 $X\times Z$ interaction term으로 표현합니다.

> Trellis 그래프(또는 격자 그래프, Facet 그래프)는 하나의 데이터셋을 특정 조건(변수의 구간)에 따라 여러 개의 패널로 쪼개어 시각화함으로써, 데이터 전체를 하나로 묶어봤을 때는 놓치기 쉬운 상호작용(Interaction) 효과나 효과수정(Effect Modification)을 찾아내는 데 매우 탁월한 도구입니다.

> 제3의 변수(상황 변수)의 값이나 구간에 따라 그래프를 나누기 때문에, 해당 변수의 영향을 분리해서 볼 수 있습니다.

> 전체 산점도에서는 두 변수의 관계가 단순히 '양의 상관관계'처럼 보일 수 있지만, Trellis를 통해 변수 구간별로 나누어 보면 어떤 구간에서는 양의 관계, 다른 구간에서는 음의 관계로 나타나는 식의 '효과수정' 현상을 명확하게 포착할 수 있습니다.

<img width="4770" height="1224" alt="image" src="https://github.com/user-attachments/assets/c0cb8684-0aaf-4f4c-afc9-425f60212781" />
 
> 1. Combined Plot (맨 왼쪽): 모든 데이터를 하나의 평면에 뭉뚱그려 그리면 기울기가 거의 0에 가까운 평평한 빨간 선이 나옵니다. 조절 변수 Z를 고려하지 않으면 *"X와 Y는 무관하다"*는 통계적 오류(결론)에 빠지게 됩니다.
> 2. Trellis Panels (오른쪽 3개): 변수 Z의 수준(Low, Medium, High)에 따라 데이터를 분할하는 순간 숨겨져 있던 진짜 관계가 드러납니다.
>   - Low Z: X가 증가할수록 Y가 감소하는 파란색 음(-)의 관계
>   - Medium Z: X와 Y 사이에 아무런 경향이 없는 주황색 무상관 관계
>   - High Z: X가 증가할수록 Y도 빠르게 증가하는 초록색 양(+)의 관계

## 12. Stalactite Plot과 반복적 이상치 진단

Stalactite plot은 처음부터 전체 표본의 평균·공분산만 쓰지 않고 작은 subset에서 시작해 subset 크기를 늘리면서 generalized distance 기준 이상치가 얼마나 지속적으로 유지되는지 보여줍니다. 목적은 masking을 완화하는 것입니다.

교재 air pollution 데이터에서 subset 크기가 커지면서 이상치 수가 줄고, 전체 41개 관측치를 사용했을 때 Chicago, Phoenix, Providence가 이상치로 남는 패턴을 보고합니다.

> 이상치의 오염 가능성이 낮은 아주 작은 크기의 깨끗한 부분집합(Subset)을 먼저 선택합니다. 이 Subset을 기준으로 평균과 공분산을 계산하고, 이를 기반으로 전체 데이터의 일반화 거리(Generalized Distance 또는 마할라노비스 거리)를 측정합니다.
>
> 거리가 가까운 순으로 관측치를 하나씩 추가하면서 Subset의 크기를 점점 늘려나갑니다. Subset의 크기(X축)가 커짐에 따라 각 관측치(Y축)가 기준치 이상의 먼 거리를 유지하는지 추적합니다.

> 이 과정에서 끝까지 기준치 밖에 머무는 관측치들이 행(Row) 방향으로 길게 표시되며, 그 모양이 마치 동굴의 종유석과 닮았다고 하여 이 이름이 붙었습니다. 초기 무작위 무오염 집단에서 시작해 전향적으로 나아가는 Forward Search 기법의 대표적인 시각화 사례입니다.

<img width="989" height="589" alt="image" src="https://github.com/user-attachments/assets/cad69eaa-486a-461f-953e-5bd7021f3417" />

> X축 (Subset Size m): 분석에 사용되는 깨끗한 데이터 집합(Subset)의 크기입니다. 최초 10개의 데이터로 시작해 최종 100개까지 순차적으로 크기를 늘려갑니다.
>
> Y축 (Observations): 데이터의 개별 관측치(샘플)들입니다. 이상치로 판정된 지속 시간이 길수록 위쪽에 배치되도록 정렬했습니다.
>
> 색상 표현 (Black = Outlier): 해당 Subset 크기(m)를 기준으로 계산한 마할라노비스 거리가 임계값을 초과(이상치로 판정)했을 때 검은색 블록으로 표시됩니다.

## 13. 저자가 직접 보고한 결과 vs. 이 노트의 해석

### 13.1 저자 보고

- `manu`–`popul` 전체 Pearson correlation: $0.9553$.
- bivariate boxplot에서 식별한 Chicago, Philadelphia, Detroit, Cleveland를 제거한 correlation: $0.7956$.
- convex hull point를 제거한 correlation: $0.9225$.
- CYG OB1 별 데이터의 bivariate KDE는 두 개의 서로 다른 cluster를 시사합니다.
- body measurement KDE에서는 waist/hips 관계가 두 group을 시사하며, 실제로 성별 두 집단이 존재합니다.
- 3-D와 star graphics가 항상 추가 정보를 주는 것은 아니며, scatterplot matrix가 많은 상황에서 더 유용하다고 결론 내립니다.

### 13.2 해석

세 상관값 $0.9553$, $0.7956$, $0.9225$는 “어떤 값이 진짜 상관인가?”를 결정해주는 것이 아니라 **robustness analysis의 필요성**을 보여줍니다. 실제 연구에서는 이상치를 임의 삭제하기보다 원자료 결과와 robust 추정 결과를 함께 보고해야 합니다. 또 KDE에서 두 peak가 보인다고 바로 두 population이라고 단정하지 않고, cluster stability나 domain label로 검증해야 합니다.

## 14. 통계적으로 취약한 부분과 비교 불가능한 수치

1. **그래프를 본 뒤 가설을 만든 뒤 같은 데이터로 검정**하면 double dipping이 생깁니다. 탐색에서 발견한 패턴은 새 데이터에서 검증해야 합니다.

> 데이터 분할 (Data Splitting): 전체 데이터를 처음부터 탐색용(Exploratory Set)과 검증용(Confirmation Set)으로 나눕니다. 탐색용 데이터로 마음껏 그래프를 그리고 가설을 찾은 뒤, 확정된 가설은 한 번도 보지 않은 검증용 데이터로만 통계 검정을 수행합니다.
>
> 사전 등록 (Preregistration): 학술 연구나 임상 시험에서는 데이터를 수집·보기 전에 어떤 가설을 검정할지, 어떤 통계 기법을 쓸지 미리 등록하여 사후에 가설을 조작하는 것을 원천 차단합니다.

2. **이상치 제거 후 상관 변화**는 예측 성능 향상과 동일하지 않습니다. 제거 기준이 test data까지 보고 정해졌다면 오히려 과대평가입니다.

> 데이터셋의 이상치가 오류(Noise)가 아니라 실제 발생하는 희귀한 현상(예: 금융 사기, 장비 고장, 극단적 기후)일 경우, 이를 제거하면 모델은 중요한 현실 세계의 패턴을 학습할 기회를 잃어버립니다.
>
> 이상치 판단 기준(예: IQR 공식의 계수, 임계값 등)은 오직 학습 데이터(Train Data)만을 기준으로 수립해야 합니다. 테스트 데이터는 이 기준을 그대로 적용받아 필터링되거나, 원본 상태 그대로 평가받아야 합니다.
>
> 이상치를 무조건 제거하기보다, 이상치에 영향을 덜 받는 로바스트 회귀(Robust Regression)나 트리 기반 모델(Random Forest, XGBoost 등)을 사용하는 것이 안전합니다.

3. **KDE peak 수**는 bandwidth에 따라 바뀔 수 있으므로 cluster 개수의 확정적 검정이 아닙니다.

> 대역폭을 너무 크게 설정하면 데이터의 세부 구조가 뭉개져 피크가 너무 적게(대개 1개로) 나타납니다. 대역폭을 너무 작게 설정하면 데이터의 노이즈(noise)까지 모두 피크로 인식하여 불필요하게 많은 피크가 생성됩니다.

> KDE의 시각적 직관성을 살리면서도 군집 개수를 더 객관적으로 검정하기 위해 다음과 같은 방법들을 함께 활용합니다.
>   - 실루엣 스코어 (Silhouette Score): 군집 내부의 응집도와 군집 간의 분리도를 계산하여 적절한 군집 수를 평가합니다.
>   - 엘보우 방법 (Elbow Method): 군집 수에 따른 SSE(오차제곱합)의 감소율이 꺾이는 지점을 찾습니다.
>   - 딥스 테스트 (Dip Test): 데이터가 단봉 분포(Unimodal, 피크가 1개)인지 다봉 분포(Multimodal)인지 통계적으로 유의성을 검정(p-value 확인)합니다.
>   - 실버만의 부트스트랩 검정 (Silverman's Bootstrap Test): KDE 환경에서 피크 개수의 유의성을 판단하기 위해 부트스트랩 샘플링을 활용하는 구체적인 통계적 검정 방법입니다.

4. **3-D·trellis 시각화**는 panel당 표본 수가 적으면 우연한 모양에 민감합니다.

> 데이터의 양이 부족하면 특정 패널에서 나타나는 패턴이 실제 경향이 아니라 우연에 의한 왜곡(노이즈)일 가능성이 커집니다. 이를 통계학에서는 소표본 문제(Small sample problem) 또는 과적합 위험이라고 부릅니다.
>
> 이 문제를 완화하고 시각화의 신뢰도를 높이기 위한 주요 접근법은 다음과 같습니다.
>   - 패널 통합 (Binning & Aggregation) : 복잡한 범주나 촘촘한 구간을 더 넓게 병합하여 패널당 표본 수를 확보합니다.
>   - 차원 축소 및 변수 단순화 : 꼭 필요한 경우가 아니라면 2차원 Trellis 플롯으로 전환하고, 색상(Color)이나 크기(Size)를 활용해 추가 차원을 표현하는 것이 안전합니다.
>   - 통계적 평활화 (Smoothing) 추가 : 각 패널에 산점도만 그리기보다 LOESS(국소 회귀)나 스플라인(Spline) 같은 평활화 곡선/곡면을 함께 시각화합니다.
>   - 신뢰구간 (Confidence Interval) 표시 : 데이터가 부족한 패널일수록 신뢰구간이 넓게 표시되므로, 보는 사람이 "이 패널의 결과는 불확실성이 크다"는 것을 직관적으로 인지할 수 있게 합니다.

5. **PaCMAP/densMAP 결과와 교재 scatterplot/KDE를 숫자로 직접 비교할 수 없습니다.** 전자는 고차원 embedding quality, 후자는 원변수 공간의 직접 관계 또는 density를 보여주는 도구로 목적함수가 다릅니다.

## 15. 문서가 직접 답하지 않는 질문과 답변

### 질문 1. 수백~수천 변수가 있으면 scatterplot matrix는 어떻게 하는가?

전 변수의 $q(q-1)/2$ 쌍을 그리는 것은 비현실적입니다. 먼저 domain-based selection, variance filter, supervised leakage-safe screening 또는 PCA를 통해 후보를 줄인 뒤 중요한 변수쌍을 확인합니다. 단, target을 사용해 변수를 고른 경우 그 선택은 train 내부에서만 해야 합니다.

> Supervised Leakage-safe Screening (누수 방지 감독형 스크리닝) : 타겟 변수(Y)와의 상관관계나 정보 이득(Information Gain) 등을 계산하여 상관관계가 높은 변수만 남기는 방법입니다. 이때 반드시 데이터를 Train/Test로 먼저 분할한 후, Train 데이터셋으로만 스크리닝 기준을 세워야 데이터 누수(Data Leakage)를 막을 수 있습니다.

### 질문 2. 임베딩에서 가까운 점은 원공간에서도 반드시 가까운가?

아닙니다. 모든 차원축소는 어떤 정보를 희생합니다. t-SNE/UMAP/PaCMAP처럼 2-D로 투영하는 방법은 local/global structure를 서로 다른 방식으로 보존하므로, 2-D 거리 자체를 원공간 거리의 정확한 대체물로 취급하면 안 됩니다.

> - t-SNE (t-Distributed Stochastic Neighbor Embedding) : 고차원 공간에서의 데이터 간 거리(유클리드 거리)를 확률 분포로 변환하고, 저차원 공간에서도 이 확률 분포가 최대한 유지되도록 최적화합니다. 이 과정에서 멀리 있는 점보다 가까운 이웃 간의 관계(Local Structure)를 보존하는 데 특화되어 있습니다.
> - UMAP (Uniform Manifold Approximation and Projection) : 퍼지 심벌릭 그래프(Fuzzy Simplicial Set)와 리만 기하학을 기반으로 고차원 데이터의 위상 구조(Topology)를 학습한 뒤 이를 저차원 그래프로 투영합니다.
> - PaCMAP (Pairwise Controlled Manifold Approximation Projection) : 데이터 쌍(Pair)을 세 가지 유형(가까운 이웃, 중간 거리 쌍, 먼 쌍)으로 나누어 각각 다른 제약 조건을 부여해 최적화하는 최신 차원 축소 기법입니다.

> 임베딩(저차원) 공간에서 가깝다고 해서 원공간(고차원)에서 반드시 가까운 것은 아닙니다. 이를 차원 축소의 거짓 발견(False Discovery / False Positive) 현상이라고 합니다.
>
> 고차원의 방대한 정보(거리, 각도, 위상 등)를 2차원이나 3차원이라는 극단적으로 좁은 공간에 구겨 넣는 과정에서 필연적으로 정보의 왜곡과 손실이 발생하기 때문입니다.
>
> - 고차원 공간은 저차원에 비해 공간의 볼륨이 기하급수적으로 넓습니다. 고차원의 수많은 데이터가 2차원으로 내려오면, 원공간에서는 서로 적당히 떨어져 있던 점들이 저차원 공간의 한정된 면적 때문에 어쩔 수 없이 뭉치게 되면서 가깝게 보일 수 있습니다.
> - 특히 t-SNE나 UMAP 같은 알고리즘은 국소적 이웃(Local Structure) 관계를 강제로 유지하려다 보니, 원공간에서는 연속적이거나 큰 의미가 없는 데이터 뭉치가 저차원에서는 마치 독립적이고 뚜렷한 군집인 것처럼 시각적 착시를 일으키기도 합니다.
> - t-SNE는 멀리 떨어진 군집 간의 거리를 거의 보존하지 못하며, UMAP이나 PaCMAP이 이를 개선했음에도 불구하고 저차원 축에서의 절대적인 거리가 원공간의 수치적 거리를 대변하지는 못합니다.

> 따라서 시각화 플롯(2-D Scatter plot)에서의 거리는 데이터의 전반적인 경향성과 대략적인 군집 형태를 파악하는 참고용으로만 보아야 하며, 실제 거리 기반의 정량적 분석(K-NN 분류, 클러스터링 등)은 반드시 차원 축소 전의 원공간 데이터나 고차원 임베딩 자체를 대상으로 수행해야 합니다.

### 질문 3. 시각화가 모델 일반화에 실제로 도움이 되는가?

간접적으로 매우 중요합니다. 시각화를 통해 leakage-like ID feature, temporal drift, nonlinear relation, heteroskedasticity, outlier regime를 발견하면 모델 specification을 개선할 수 있습니다. 그러나 시각화를 보고 모델을 반복 튜닝했다면 independent test set이 필요합니다.

> 시각화는 모델의 수학적 목적 함수를 직접 최적화하지는 않지만, 데이터의 본질적인 문제를 파악하고 올바른 모델링 방향을 설정(Specification)하는 데 결정적인 역할을 합니다.
>
> 데이터 수집 과정에서 실수로 포함된 식별자(ID)나 타겟 변수의 힌트(Leakage)를 시각화(예: Feature Importance vs Cardinallity 산점도)로 찾아내어, 학습 데이터에만 과적합되는 현상을 방지합니다.
>
> 시간에 따른 피처나 타겟의 분포 변화를 라인 차트나 시계열 히트맵으로 확인하여, 과거 데이터에만 치우치지 않도록 적절한 검증 전략(Time-series split)이나 가중치를 적용할 수 있게 합니다.
>
> Heteroskedasticity(이분산성) 대응: 잔차(Residual) 분석 시각화를 통해 오차의 분산이 일정하지 않음을 파악하고, 타겟 변수 변환(Log 변환 등)이나 가중 최소제곱법(WLS) 등을 적용해 모델의 신뢰성을 높입니다.
>
> Outlier regime(이상치 영역) 제거: 상자 수염 그림(Boxplot)이나 차원 축소(t-SNE/UMAP) 시각화를 통해 단순 노이즈인 이상치와 실제 중요한 희귀 케이스를 분리하여 모델이 노이즈를 학습하지 않도록 방지합니다.

## 16. 2020년 이후 관련 최신 연구 비교 분석

### 16.1 PaCMAP

**Wang, Huang, Rudin & Shaposhnik, “Understanding How Dimension Reduction Tools Work: An Empirical Approach to Deciphering t-SNE, UMAP, TriMap, and PaCMAP for Data Visualization”, JMLR, 2021.**

PaCMAP은 pairwise relation을 near, mid-near, further 구조로 조절하여 local structure뿐 아니라 global structure도 더 잘 보존하려는 임베딩입니다. 교재의 scatterplot matrix가 원변수 공간을 직접 보여주는 반면 PaCMAP은 고차원 구조를 2-D/3-D에 압축합니다.

### 16.2 densMAP

**Narayan, Berger & Cho, “Assessing single-cell transcriptomic variability through density-preserving data visualization”, Nature Biotechnology, 2021.**

densMAP은 t-SNE/UMAP류 시각화가 원공간의 local density를 왜곡할 수 있다는 문제를 다룹니다. 단순히 이웃만 보존하는 것이 아니라 **원공간의 밀도 차이도 임베딩에 반영**하려는 방향입니다.

### 비교

| 방법 | 주로 보존하려는 것 | 교재 방법과의 관계 | 주의점 |
|---|---|---|---|
| Scatterplot matrix | 원변수 쌍의 직접 관계 | 가장 해석 가능 | $q$가 크면 확장 어려움 |
| KDE contour | 원공간의 1-D/2-D density | 군집 후보 시각화 | bandwidth 민감 |
| PaCMAP | local + global geometric structure | 고차원 시각화 확장 | 2-D 거리 과해석 금지 |
| densMAP | neighborhood + density | KDE의 density 관점을 manifold embedding과 결합 | embedding hyperparameter 의존 |

## 17. 실제 파이프라인 적용 방향

```text
Train split 확정
  ↓
단변량 histogram / boxplot / missingness
  ↓
중요 변수 scatterplot + scatterplot matrix
  ↓
시간/장비/recipe 조건별 Trellis 또는 faceting
  ↓
KDE로 multimodality 후보 탐색
  ↓
robust distance로 outlier regime 확인
  ↓
고차원이면 PCA/PaCMAP 등 보조 임베딩
  ↓
발견한 가설을 feature/model 후보로 전환
  ↓
Validation에서 검증
  ↓
Test는 마지막 한 번만 사용
```

### 공정·시계열 데이터에서 특히 중요한 점

시간순 데이터는 무작위 scatterplot만 보면 drift를 놓칩니다. 따라서 각 변수의 `value vs time`, target residual vs time, chamber별 faceting을 먼저 보고, 그다음 변수쌍 관계를 보는 것이 좋습니다. 또한 future test 기간의 분포를 보고 preprocessing threshold를 정하면 leakage가 될 수 있으므로 시각화 단계에서도 train/validation/test 색을 구분하되 **의사결정은 train/validation에 한정**해야 합니다.

## 18. 일반화 성능 향상 가능성

이 장의 기술은 직접 예측기가 아니므로 “R²를 몇 % 높인다”는 식으로 말할 수 없습니다. 대신 다음 경로로 일반화에 기여할 수 있습니다.

- outlier-driven spurious correlation 제거 또는 robust 처리
- nonlinear pattern 발견 후 적절한 spline/kernel/tree 모델 선택
- subgroup interaction 발견 후 group-aware model 구성
- train–validation distribution shift 조기 발견
- density imbalance 발견 후 sample weighting 또는 stratified evaluation

즉 시각화는 모델의 **가설공간을 데이터 구조에 맞게 좁혀 variance와 misspecification을 줄이는 과정**입니다.

## 19. 시사점과 후속 연구 방향

교재의 핵심 메시지는 “좋은 그래프는 계산된 숫자의 보조물이 아니라 데이터의 질문 자체를 바꾼다”는 것입니다. 후속 연구로는 (1) scatterplot에서 발견한 비선형성을 spline/GAM이 실제 OOS 성능으로 회수하는지, (2) KDE에서 보인 multimodality가 실제 장비·recipe regime과 대응하는지, (3) PCA/PaCMAP/densMAP이 같은 공정 drift를 얼마나 안정적으로 보존하는지, (4) robust outlier 처리 전후 예측 성능과 calibration이 어떻게 변하는지를 비교할 수 있습니다.

## 20. 빠른 이해 점검

- 왜 $r=0.95$라는 숫자만 보고 강한 일반 관계라고 결론 내리면 안 되는가?

> 상관계수는 직선 형태의 관계만 포착합니다. 만약 두 변수가 곡선 형태(예: 이차함수 형태)로 완벽하게 묶여 있더라도, 선형 패턴이 아니면 r 값이 낮게 나올 수 있습니다. 반대로, 데이터가 꺾이는 지점이나 특정 구간의 왜곡 때문에 실제로는 곡선 관계임에도 불구하고 선형 상관계수가 우연히 매우 높게 나타날 수 있습니다.

- KDE에서 $h$가 너무 작거나 클 때 각각 어떤 그림이 나오는가?

<img width="638" height="223" alt="스크린샷 2026-09-04 오후 3 29 41" src="https://github.com/user-attachments/assets/0cc3322a-08ca-444f-a139-96ab6a88a5d8" />

<img width="1489" height="463" alt="image" src="https://github.com/user-attachments/assets/0a42684b-f7d8-499b-bb6f-2792e7bd8f58" />

> 너무 작은 대역폭 (빨간색 선): 데이터 포인트 하나하나를 전부 봉우리로 만들어서 톱날처럼 뾰족하고 거친 모양이 됩니다.
>
> 적절한 대역폭 (초록색 선): 자잘한 노이즈는 무시하고, 데이터가 가진 원래의 큰 두 개의 봉우리 모양을 부드럽게 잘 살려냅니다.
>
> 너무 큰 대역폭 (파란색 선): 세부 정보를 너무 많이 지워버려서 두 봉우리가 뭉개진 완만한 하나의 산 모양이 됩니다.

- scatterplot matrix와 PaCMAP은 어떤 정보를 서로 다르게 보존하는가?

> 산점도 행렬은 원래 차원의 축(Feature) 정보와 변수 간의 개별 관계를 정확하게 보존하는 반면, PaCMAP은 데이터 공간 전체의 구조(지역적/전역적 구조)와 고차원적 기하학적 형태를 유기적으로 보존합니다.

- 이상치를 제거하는 것과 robust estimator를 쓰는 것의 차이는 무엇인가?

> 이상치를 제거하는 것은 전통적인 통계 기법(예: 평균, 평균제곱오차 기반 선형 회귀)을 그대로 사용하기 위해, 방해가 되는 데이터를 사전에 골라내어 버리는 방법입니다. Robust Estimator 는 데이터에 이상치가 포함되어 있어도 통계적 왜곡을 계산 자체에서 스스로 견뎌내도록 설계된 알고리즘을 적용하는 방법입니다. 실무에서는 이상치의 원인이 단순 입력 오류라면 제거하는 것이 좋고, 데이터 고유의 변동성이거나 원인을 알 수 없다면 로바스트 추정량을 도입하는 것이 안전합니다.

## 21. 참고자료

### 교재
- Brian S. Everitt & Torsten Hothorn, *An Introduction to Applied Multivariate Analysis with R*, Springer, 2011, Chapter 2.
- B. W. Silverman, *Density Estimation for Statistics and Data Analysis*, 1986. 교재의 KDE 이론 참고문헌.
- M. P. Wand & M. C. Jones, *Kernel Smoothing*, 1995. 교재의 비모수 밀도 추정 참고문헌.

### 2020년 이후 확장 연구 및 사이트
- Yingfan Wang, Haiyang Huang, Cynthia Rudin & Yaron Shaposhnik, “Understanding How Dimension Reduction Tools Work: An Empirical Approach to Deciphering t-SNE, UMAP, TriMap, and PaCMAP for Data Visualization”, *Journal of Machine Learning Research*, 22(201), 2021. Source site: JMLR.
- Ashwin Narayan, Bonnie Berger & Hyunghoon Cho, “Assessing single-cell transcriptomic variability through density-preserving data visualization”, *Nature Biotechnology*, 39, 765–774, 2021. Source site: Nature.
