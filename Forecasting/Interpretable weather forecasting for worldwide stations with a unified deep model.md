# Interpretable weather forecasting for worldwide stations with a unified deep model

**검토 대상:** Haixu Wu 외, *Interpretable weather forecasting for worldwide stations with a unified deep model*, **Nature Machine Intelligence 5, 602–611 (2023)**. 아래의 본문 페이지는 학술지 쪽수이며, Extended Data 그림은 첨부 PDF의 페이지를 병기했습니다. 보충자료와 후속 연구에서 확인한 내용은 원논문의 결과와 구분합니다. Interpretable-weather-forecasti…

## 1. Executive summary — 9문장

이 연구의 목적은 불규칙하게 흩어진 기상관측소들을 개별적으로 예측할 때 발생하는 정보 단절과 계산 부담을 줄이고, 여러 관측소가 함께 기상 변화를 예측하는 통합모델을 만드는 것이다.  
저자들이 제안한 **Corrformer**는 한 관측소에서 반복되는 시간적 패턴과 여러 관측소 사이에서 시차를 두고 나타나는 패턴을 지역부터 전지구까지의 계층구조에서 결합한다.  
이를 통해 모든 관측소 쌍을 직접 비교하지 않으면서 관측소 수에 대해 선형적으로 증가하는 공간 모델링 비용을 달성했다고 보고한다. **[본문 pp.602–606, Figs.1–2]** Interpretable-weather-forecasti… Interpretable-weather-forecasti…

주요 실험은 전지구 3,850개, 중국 지역 34,040개, 동계올림픽 경기장 24개 관측소에서 수집한 자료를 사용하며, 과거 48시간으로 미래 24시간을 예측한다.  
Corrformer는 본문 Table 1의 네 예측 과제 모두에서 비교군보다 낮은 평균제곱오차와 평균절대오차를 기록했다.  
또한 약 9–10%의 관측소로 학습한 모델을 재학습 없이 더 많은 관측소에 적용한 결과는 관측망 확장에 대한 적응 가능성을 보여준다. **[본문 pp.605–608, Table 1; Extended Data Fig.5]** Interpretable-weather-forecasti… Interpretable-weather-forecasti… Interpretable-weather-forecasti…

다만 관측소 사이의 시차에서 추론한 진행 방향이 실제 기상현상의 인과적 이동 경로임을 입증한 것은 아니다.  
또한 관측소를 추가하는 실험을 새로운 기후권·미래 기후·관측 이력이 없는 장소에 대한 일반화 검증으로 확대해석하거나, 일부 비교 결과를 수치예보 전반에 대한 우위로 해석해서는 안 된다.  
연구의 핵심 가치는 **대규모 비정규 관측망의 효율적인 공동예측**이며, 저자가 제시한 물리정보와 데이터 기반 학습의 결합을 발전시키되 지리적·시간적 외삽과 운영 조건을 분리 검증하는 것이 중요한 후속 방향이다. **[본문 p.609; Extended Data Fig.5; 보충 Table 10]** Interpretable-weather-forecasti… Interpretable-weather-forecasti… [Springer Nature](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs42256-023-00667-9/MediaObjects/42256_2023_667_MOESM1_ESM.pdf)

---

## 2. 연구 목적과 필요성: 무엇을 해결하는 논문인가?

### 2.1 저자가 설정한 문제

**[저자 보고]** 관측소 하나씩 별도 모델로 예측하면 주변 관측소가 먼저 포착한 기상 변화를 충분히 활용하지 못하고, 관측소가 많아질수록 학습·저장·운영 부담도 커진다. 반대로 모든 관측소 사이의 관계를 직접 계산하면 계산량이 급격히 증가한다. 기존의 규칙적인 격자에 맞춘 기상예측 구조를 불규칙한 관측소 배치에 그대로 적용하기도 어렵다. **[p.602]** Interpretable-weather-forecasti…

**[해석]** 이 논문의 질문은 “전지구 대기의 모든 상태를 재현할 수 있는가?”보다는 다음에 가깝다.

> **관측소의 제한된 과거 관측만으로, 다른 관측소에서 먼저 나타난 유용한 패턴을 계산 효율적으로 가져와 각 관측소의 미래를 예측할 수 있는가?**

따라서 제목의 **worldwide**는 전세계에 분포한 관측소를 대상으로 한다는 뜻이지, 해양과 대기 상층까지 빈틈없이 예측한다는 뜻은 아니다. 또한 **unified model**의 핵심은 여러 관측소를 공동으로 처리한다는 것이며, 모든 자료셋과 기상변수에 동일한 가중치를 사용하는 범용 모델의 검증으로 읽어서는 안 된다. **[문제 정의 p.603; 실험 구성 pp.607–608]** Interpretable-weather-forecasti… Interpretable-weather-forecasti… Interpretable-weather-forecasti…

### 2.2 실험 자료와 예측 대상

| 자료 | 관측소 수·기간 | 예측 변수 | 해석할 때의 주의점 |
|---|---|---|---|
| Global | 3,850개, 2019–2020년 | 시간별 **평균 풍속**, 시간별 기온 | 두 변수는 Global wind와 Global temp의 별도 과제로 제시된다. **[pp.607–608]** Interpretable-weather-forecasti… |
| Regional | 중국 34,040개, 2018–2019년 | 시간별 **최대 풍속** | Global wind의 평균 풍속과 예측 대상 자체가 다르다. **[p.608]** Interpretable-weather-forecasti… |
| Olympics | 경기장 24개, 2019–2020년 | 시간별 **최대 풍속** | 경기장 자료이지만, 이 표의 시험자료가 2022년 대회 기간의 실시간 평가자료라는 뜻은 아니다. **[p.608]** Interpretable-weather-forecasti… |

주요 자료 분할은 시간순 **학습:검증:시험 = 7:1:2**이다. **[p.608]** Interpretable-weather-forecasti…

> **용어 — 관측소와 격자:** 관측소는 실제 센서가 있는 특정 위치이고, 격자는 공간을 일정한 간격으로 나눈 계산 지점이다. 두 방식은 위치뿐 아니라 하나의 값이 대표하는 공간 범위도 다를 수 있다.

---

## 3. 핵심 주장과 근거

| 핵심 주장 | 저자가 제시한 근거 | 연구자로서의 해석·한계 |
|---|---|---|
| 시간과 공간의 상관관계를 하나의 구조에서 모델링한다. | 관측소 단위 자기상관, 지역 단위 교차상관을 계층적 트리에서 결합한다. **[pp.604–606, Figs.2·5, 식(5)]** Interpretable-weather-forecasti… Interpretable-weather-forecasti… | 새로움은 자기상관 자체보다는 **대규모 관측소의 시간·공간 정보를 결합하는 방식**에 있다. |
| 대규모 관측망에도 계산이 가능하다. | 전체 multi-correlation 복잡도는 $\mathcal O(NL\log L)$이며, 지역 자료에서 StemGNN은 메모리 부족을 보였다. **[pp.606·608·610, Table 1, 식(11)]** Interpretable-weather-forecasti… Interpretable-weather-forecasti… Interpretable-weather-forecasti… | 특정 밀집 관계 모델보다 효율적이라는 근거다. 모든 그래프 모델보다 항상 빠르다는 증명은 아니다. |
| 비교 모델보다 예측 오차가 낮다. | 본문 Table 1의 네 과제에서 MSE·MAE 모두 최저값을 보고한다. **[p.606, Table 1]** Interpretable-weather-forecasti… | 해당 데이터·기간·예측길이·비교군 안에서의 결과다. 다른 자료의 수치와 직접 순위를 매길 수 없다. |
| 기상현상의 진행 방향을 해석할 수 있다. | 관측소별 상대적 지연을 정렬한 방향이 ERA5의 변화 양상과 유사하다는 사례를 제시한다. **[pp.607–609, Fig.4b]** Interpretable-weather-forecasti… Interpretable-weather-forecasti… | **관계의 가시화**는 제공하지만, 물리적 인과관계의 식별과는 다르다. |
| 새 관측소에 재학습 없이 적응한다. | 전지구 350개·지역 3,404개로 학습하고 관측소를 점진적으로 추가한다. **[Extended Data Fig.5, PDF p.15]** Interpretable-weather-forecasti… | 관측망 크기 변화에 대한 유용한 증거다. 새로운 기후권이나 관측 이력이 없는 위치에 대한 검증은 아니다. |
| 실제 서비스에 활용 가능하다. | 2022년 동계올림픽 예보 서비스에 배치했다고 보고한다. **[p.608]** Interpretable-weather-forecasti… | 운영 경험의 근거이지만, 대회 기간 전체의 오경보율·서비스 가용성·의사결정 효과를 정량적으로 입증하는 자료와는 구별해야 한다. |

> **용어 — 상관관계와 인과관계:** 상관관계는 두 값이 함께 또는 시차를 두고 변하는 관계다. 인과관계는 한 요인의 변화가 다른 요인의 변화를 실제로 일으킨다는 뜻이며, 상관관계만으로 확정할 수 없다.

---

## 4. 제안 방법과 모델 구조: 수식 중심 설명

아래에서 **“원문 식”**은 논문에 제시된 수식을 뜻하고, **“해설용 재표기”**는 동일한 연산을 읽기 쉽게 정리한 것이다.

### 4.1 예측 문제의 정의

**[저자 보고: p.603]** $N$개 관측소가 각각 $C$개 기상변수를 관측한다고 하자.

$$
x_t\in\mathbb R^{N\times C},
\qquad
X_{\text{past}}\in\mathbb R^{N\times I\times C},
\qquad
\widehat X_{\text{future}}\in\mathbb R^{N\times O\times C}.
$$

이를 함수로 표현하면 다음과 같다.

```math
\widehat X_{\text{future}}
=
f_\theta
\left(
X_{\text{past}},
x_{\text{space}},
x_{\text{time}}
\right).
```

**기호:** $t$는 현재 시각, $I$는 과거 입력 길이, $O$는 미래 예측 길이, $\theta$는 학습되는 모델 매개변수다. $x_{\text{space}}$와 $x_{\text{time}}$은 각각 위치와 시간 정보다. 본문 주요 실험에서는 $I=48$, $O=24$이다. Interpretable-weather-forecasti… Interpretable-weather-forecasti…

**[해석]** 여러 관측소의 시계열을 입력으로 함께 받기 때문에, 한 관측소에서 먼저 관측된 변화가 다른 관측소의 예측에 사용될 수 있다.

### 4.2 입력 임베딩: 관측값·위치·시간을 결합한다

**원문 식(1), p.604:**

```math
x_{\text{en}}
=
\text{Embed}_{\text{data}}(X_{\text{past}})
+
\text{Embed}_{\text{space}}(x_{\text{space}})
+
\text{Embed}_{\text{time}}(x_{\text{time}}).
```

**기호:**  
$x_{\text{en}}\in\mathbb R^{N\times I\times d_{\text{model}}}$은 인코더 입력이고, $d_{\text{model}}$은 내부 특징의 차원이다. 위치 정보는 경도·위도·고도, 시간 정보는 연·월·일·시로 구성된다. 각 $\text{Embed}$는 해당 입력을 공통 특징 차원으로 변환하는 연산이며, 공간·시간 차원이 맞도록 확장해서 더한다. Interpretable-weather-forecasti…

> **용어 — 임베딩:** 원래의 관측값이나 좌표를 모델이 계산하기 좋은 숫자 벡터로 바꾸는 과정이다.  
> **용어 — 브로드캐스팅:** 예를 들어 한 관측소의 위치 벡터를 모든 시간 지점에 공통으로 적용하도록 차원을 맞추는 연산이다.

### 4.3 시계열 분해와 인코더–디코더

Corrformer는 Autoformer의 분해 구조를 바탕으로 한다. 이동평균으로 완만한 추세를 추출하고, 나머지 변동을 별도로 처리한다. **[p.604, Fig.1; Autoformer §3.1]** Interpretable-weather-forecasti… [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2021/file/bcc0d400288793e8bdcd7c19a8ac0c2b-Paper.pdf)

**해설용 재표기:**

$$
z_{\text{trend}}=\text{MovingAvg}_{k}(z),
\qquad
z_{\text{season}}=z-z_{\text{trend}}.
$$

**기호:** $z$는 입력 또는 중간 시계열 특징, $k$는 이동평균 창 길이, $z_{\text{trend}}$는 추세 성분, $z_{\text{season}}$은 추세를 제거한 나머지 성분이다.

> **용어 — 이동평균:** 주변 시점의 값을 평균 내어 빠른 변동을 완화하는 방법이다.  
> **용어 — seasonal 성분:** 여기서는 봄·여름 같은 계절만 뜻하지 않는다. 추세를 제외한 반복적·변동적 성분을 가리킨다.

**[저자 보고]** 분해는 상관분석에 유리한 **광의 정상성**을 강화하기 위한 것이다. 인코더에서는 과거의 추세를 제거하면서 특징을 추출하고, 디코더에서는 반복 성분의 예측과 누적된 추세 예측을 더한다. **[p.604]** Interpretable-weather-forecasti… Interpretable-weather-forecasti…

> **용어 — 광의 정상성:** 평균이 시간에 따라 변하지 않고, 두 시점 사이의 공분산이 절대적인 날짜가 아니라 두 시점의 간격에만 의존하는 성질이다.

**[해석]** 이동평균을 뺐다고 이 성질이 수학적으로 보장되지는 않는다. 급격한 기상 전환이나 계절별 변동성 차이는 잔차에도 남을 수 있다.

전체 흐름은 다음과 같다.

| 단계 | 수행하는 연산 |
|---|---|
| 인코더 | Multi-correlation → 특징 변환층 → 반복적 분해·잔차 연결 |
| 디코더 초기화 | 최근 $I/2$ 길이의 과거 정보와 미래 $O$ 길이의 초기 자리값을 결합 |
| 디코더 갱신 | 디코더 내부 multi-correlation → 인코더 정보와의 multi-correlation → 특징 변환 |
| 최종 출력 | 반복·변동 성분 예측 + 누적된 추세 예측 |

**[원문 식(2)–(3), p.604, Fig.1]** Interpretable-weather-forecasti…

> **용어 — 인코더·디코더:** 인코더는 과거 자료에서 예측에 필요한 특징을 추출하고, 디코더는 이를 이용해 미래 값을 만든다.  
> **용어 — 잔차 연결:** 변환 결과에 입력을 다시 더하여 기존 정보를 보존하는 연결이다.

실제 구현은 **인코더 2층·디코더 1층**, 숨김차원은 전지구 768·지역 256·올림픽 512이며, 이동평균 창은 25이다. **[보충 p.4, Table 1]** [Springer Nature](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs42256-023-00667-9/MediaObjects/42256_2023_667_MOESM1_ESM.pdf)

### 4.4 다중규모 트리: 관측소를 지역별로 묶는다

**[저자 보고]** 지리적 위치에 따른 계층적 이웃 군집화로 관측소를 묶는다. 가장 아래의 잎 노드는 개별 관측소이고, 상위 노드는 하위 노드 시계열의 평균이다. **[p.605, 원문 식(4), Fig.2a]** Interpretable-weather-forecasti…

**해설용 재표기:**

```math
T_{0,j}=Z_j,
\qquad
T_{i,j}
=
\frac{1}{N_i}
\sum_{k\in\mathcal C(i,j)}T_{i-1,k}.
```

**기호:** $Z_j$는 $j$번째 관측소의 시계열 특징, $i$는 트리의 규모 단계, $j$는 해당 단계의 노드 번호다. $\mathcal C(i,j)$는 노드 $(i,j)$의 자식 집합이고, $N_i$는 자식 수다. 실제 multi-correlation에서는 이러한 구조를 $Q,K,V$ 각각에 적용한다.

> **용어 — 계층적 군집화:** 가까운 관측소들을 작은 집단으로 묶고, 이 집단들을 다시 더 큰 집단으로 묶는 방식이다.  
> **용어 — pivot 시계열:** 여러 관측소를 비교하기 위한 기준 시계열이다. 여기서는 상위 노드에 해당하는 지역 평균 특징이 그 역할을 한다.

**[해석]** 논문의 “학습된 다중규모 구조”를 **트리의 연결 자체가 예측 손실로 자유롭게 학습된다**는 뜻으로 읽으면 부정확하다. 지리적 계층구조 위에서 특징과 시차 관계를 학습하는 설계로 이해하는 것이 적절하다. **[pp.605·609]** Interpretable-weather-forecasti… Interpretable-weather-forecasti…

### 4.5 시간 모델링: 자기상관으로 반복되는 부분을 모은다

**[저자 보고]** 한 시계열과 이를 시간축에서 이동시킨 시계열의 유사성을 계산하여 반복 길이를 찾는다. 이를 빠르게 계산하기 위해 FFT를 사용한다. **[pp.605·609, Fig.5a–b]** Interpretable-weather-forecasti… Interpretable-weather-forecasti…

**원문 식(6):**

```math
R_{xx}(\tau)
=
\text{FFT}^{-1}
\left(
\text{FFT}(x)\,
\overline{\text{FFT}(x)}
\right)(\tau).
```

**기호:** $x$는 길이 $L$의 시계열, $\tau$는 시간 지연, $R_{xx}(\tau)$는 지연 $\tau$에서의 자기상관 값이다. $\text{FFT}$와 $\text{FFT}^{-1}$는 고속 푸리에 변환과 역변환이고, 위쪽 막대는 복소켤레를 뜻한다.

> **용어 — 자기상관:** 동일한 시계열이 일정 시간 뒤에도 얼마나 비슷한지 측정한다.  
> **용어 — FFT·Wiener–Khinchin 정리:** FFT는 신호를 주파수 성분으로 빠르게 표현한다. 이 정리는 주파수 영역의 정보를 이용해 자기상관을 계산할 수 있게 해준다.  
> **주의:** 이 식의 값을 자동으로 $[-1,1]$ 범위의 정규화된 Pearson 상관계수라고 해석해서는 안 된다.

이후 상관값이 높은 지연 $M$개를 선택하여 정보를 합친다.

**원문 식(7)의 해설용 재표기:**

```math
\{\tau_1,\ldots,\tau_M\}
=
\text{argTopM}_{\tau\in\{1,\ldots,L\}}
R_{Q,K}(\tau),
\qquad
M=\lfloor c\log L\rfloor.
```

```math
\alpha_m
=
\frac{\exp(R_{Q,K}(\tau_m))}
{\sum_{r=1}^{M}\exp(R_{Q,K}(\tau_r))},
\qquad
\text{AutoCorr}(Q,K,V)
=
\sum_{m=1}^{M}
\alpha_m\,\text{Roll}(V,\tau_m).
```

**기호:** $Q,K,V\in\mathbb R^{L\times d_{\text{model}}}$는 상관관계 계산과 정보 집계에 쓰이는 특징표현이다. $Q,K$로 지연별 유사성을 구하고, $V$를 이동·집계한다. $c$는 선택할 지연 수를 조절하는 설정값, $\alpha_m$은 합이 1인 가중치다. $\text{Roll}$은 시계열을 순환 이동시킨다. **[p.609]** Interpretable-weather-forecasti…

> **용어 — Softmax:** 여러 점수를 양수로 바꾸고 합을 1로 정규화하는 연산이며, 위 $\alpha_m$ 식에 해당한다.  
> **용어 — 순환 이동:** 끝을 넘어간 원소를 반대쪽에 다시 넣는 이동이다. 여기서는 주기적 패턴을 정렬하는 계산 장치이지 실제 미래 관측을 입력한다는 뜻은 아니다.

### 4.6 공간 모델링: 기준 시계열과의 시차로 관측소를 정렬한다

**[저자 보고]** 모든 관측소 쌍의 관계를 구하는 대신, 각 지역의 pivot과 각 자식 노드 사이의 시차를 구한다. **[pp.606·609, Fig.2c, 원문 식(8)–(9)]** Interpretable-weather-forecasti… Interpretable-weather-forecasti…

```math
\tau_k
=
\text{argmax}_{\tau\in\{1,\ldots,L\}}
R_{T_{i,j},T^{k}_{i,j}}(\tau),
\qquad
\pi=\text{argsort}(\tau_1,\ldots,\tau_{N_i}).
```

**기호:** $T_{i,j}$는 pivot, $T^k_{i,j}$는 그 지역의 $k$번째 자식 시계열이다. $\tau_k$는 pivot과 가장 잘 맞는 상대적 지연이고, $\pi$는 지연에 따라 정렬한 노드 순서다.

> **용어 — 교차상관:** 서로 다른 두 시계열이 시간차를 두고 얼마나 비슷한지 측정한다. 예를 들어 비슷한 풍속 변화가 A에서 먼저, B에서 나중에 나타나는 관계를 찾을 수 있다.

그다음 시계열을 시간적으로 맞추고, 정렬된 관측소 방향으로 정보를 집계한 뒤 원래 순서로 되돌린다.

**원문 식(10)의 해설용 재표기:**

```math
(\mathcal D,\pi)
=
\text{Order}\left(\text{AvgPool}(Q_i),K_i\right),
```

```math
U_i
=
\text{CausalConv}
\left(
\text{Rearrange}
\left(
\text{Align}(V_i,\mathcal D),\pi
\right)
\right),
```

```math
\text{CrossCorr}(Q_i,K_i,V_i)
=
\text{Align}
\left(
\text{Rearrange}(U_i,\pi^{-1}),
-\mathcal D
\right).
```

**기호:** $\mathcal D$는 각 노드의 지연 집합, $\text{AvgPool}$은 평균을 구하는 연산, $\text{Align}$은 시간 정렬, $\text{Rearrange}$는 노드 순서 변경이다.  
$\pi^{-1}$는 원래 순서를 복원하는 역순열이며, $U_i$는 집계된 특징이다. **[p.609, Fig.5c]** Interpretable-weather-forecasti…

> **용어 — causal convolution:** 정해진 순서의 앞쪽 정보만 이용하는 합성곱이다. 이 논문에서는 **추론된 관측소 순서에 따른 일방향 집계**를 뜻하며, 실제 기상현상의 인과관계가 확인되었다는 뜻은 아니다.

**[해석]** 이 설계의 효율성은 복잡한 관측소 관계를 하나의 순서로 압축하는 데서 나온다. 반면 여러 기상현상이 동시에 지나가거나 상관관계가 양방향·분기 형태일 때, 단일 pivot과 최대 상관 지연 하나로 충분한지는 별도 검증이 필요하다.

### 4.7 전체 결합과 계산복잡도

**원문 식(5)의 축약 재표기:**

```math
\text{MultiCorr}(Q,K,V)
=
\text{AutoCorr}_{\text{leaf}}(Q,K,V)
+
\sum_{i=0}^{S-1}
\mathcal B_i
\left(
\text{CrossCorr}_{i}(Q,K,V)
\right).
```

**기호:** $S$는 트리 규모 단계 수, $\mathcal B_i$는 해당 규모의 결과를 관측소 차원에 맞게 확장하는 연산이다.  
개별 관측소의 시간 패턴과 여러 규모의 공간 패턴을 더한다. **[p.606]** Interpretable-weather-forecasti…

논문은 공간 연산의 비용을 다음과 같이 정리한다.

**원문 식(11), p.610:**

```math
\mathcal O
\left[
\sum_{i=1}^{S}
\left(\prod_{r=i+1}^{S}N_r\right)
\left(
N_iL\log L
+
N_i\log N_i
+
N_iL
\right)
\right]
=
\mathcal O(NL\log L).
```

**기호:** $N=\prod_{i=1}^{S}N_i$이고, $N_i$는 규모 $i$에서의 분기 수다. 괄호 안의 세 항은 각각 상관계산, 정렬, 집계 비용에 해당한다.  
논문의 단순화에는 $N_i>1$, $L\log L>\log N_i$라는 설정 조건이 사용된다. Interpretable-weather-forecasti…

> **용어 — 계산복잡도:** 입력 규모가 커질 때 필요한 연산량이 어떻게 증가하는지 나타낸다. 실제 실행시간 자체를 뜻하지는 않는다.

**[해석]** 이 결과는 중요한 장점이지만, 모든 모델 크기·트리 구성·자료 수집 과정까지 포함한 무조건적인 선형 시간 보장은 아니다.  
특히 관측망 변경에 따른 전처리와 트리 재구성 비용은 별도로 측정해야 한다.

---

## 5. 성능 향상: 저자 보고와 재계산의 구분

### 5.1 지표의 의미

논문은 평균제곱오차와 평균절대오차를 사용한다. **[p.608]** Interpretable-weather-forecasti…

```math
\text{MSE}
=
\frac{1}{|\mathcal E|}
\sum_{e\in\mathcal E}(\hat y_e-y_e)^2,
\qquad
\text{MAE}
=
\frac{1}{|\mathcal E|}
\sum_{e\in\mathcal E}|\hat y_e-y_e|.
```

**기호:** $\mathcal E$는 평가 대상 예측값의 집합, $y_e$는 관측값, $\hat y_e$는 예측값이다.

> **용어 — MSE·MAE:** MSE는 큰 오차에 더 강한 벌점을 주고, MAE는 오차의 절댓값을 그대로 평균한다. 따라서 한 모델이 MSE에서 우수해도 MAE에서는 열세일 수 있다.

### 5.2 본문 Table 1의 결과

아래의 오차값은 **저자 보고**, 마지막 열의 감소율은 **제가 Table 1에서 재계산한 값**이다. 비교 모델은 각 과제에서 Corrformer를 제외하고 **MSE가 가장 낮은 모델**로 골랐다. **[p.606, Table 1]** Interpretable-weather-forecasti…

| 과제 | 비교 모델 | 비교 모델 MSE | Corrformer MSE | Corrformer MAE | MSE 상대 감소율 |
|---|---|---:|---:|---:|---:|
| Global wind | N-HiTS | 4.028 | **3.889** | **1.304** | **3.45%** |
| Global temp | N-BEATS | 9.203 | **7.709** | **1.888** | **16.23%** |
| Regional wind | FreDo | 4.568 | **4.160** | **1.320** | **8.93%** |
| Olympics wind | GDBT — 원문 표기 | 7.820 | **7.605** | **1.825** | **2.75%** |

계산식은 다음과 같다.

```math
\text{MSE 감소율}(\%)
=
100
\frac{
\text{MSE}_{\text{baseline}}-\text{MSE}_{\text{Corrformer}}
}{
\text{MSE}_{\text{baseline}}
}.
```

**기호:** $\text{baseline}$은 위 표에서 선택한 비교 모델이다.

**[해석]** 전지구 기온에서 개선 폭이 가장 크고, 전지구 풍속과 올림픽 풍속에서는 상대적 개선 폭이 작다.  
특히 **“MSE 16.23% 감소”는 “기온 예측 정확도 16.23% 상승”이나 “기온 오차가 16.23% 줄었다”와 동일한 표현이 아니다.**

### 5.3 구성요소가 실제로 기여하는가?

구성요소 제거 실험에서 Global temp의 MSE는 완전모델 **7.709**, 자기상관 제거 **9.207**, 교차상관 제거 **8.120**, 트리 제거 **7.991**이다. **[보충 p.9, Table 5]** [Springer Nature](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs42256-023-00667-9/MediaObjects/42256_2023_667_MOESM1_ESM.pdf)

> **용어 — 구성요소 제거 실험, ablation:** 특정 모듈을 빼고 다시 실험하여 그 모듈의 기여를 살펴보는 방법이다.

**[해석]** 이 과제에서는 시간적 반복 패턴 모델링의 기여가 특히 크다.  
다만 각 모듈을 제거한 손실 증가량을 단순 합산하여 독립적인 기여율로 해석하면 안 된다. 모듈들 사이에 상호작용이 있기 때문이다.

### 5.4 보충자료까지 읽어야 피할 수 있는 오해

보충자료에는 **240시간까지의 예측**, **시점별 가우스 확률예측**, **계절균형 자료 분할**도 존재한다. 따라서 “24시간 밖은 실험하지 않았다”, “확률예측이 전혀 없다”, “계절 영향을 고려하지 않았다”는 평가는 부정확하다. **[보충 Fig.4, Tables 9·11]** [Springer Nature](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs42256-023-00667-9/MediaObjects/42256_2023_667_MOESM1_ESM.pdf)

> **용어 — 가우스 분포:** 평균과 표준편차로 모양이 정해지는 종 모양의 확률분포다. 시점별 분포를 출력하는 것과 여러 지역·시간의 날씨를 일관된 하나의 확률 시나리오로 만드는 것은 다르다.

### 5.5 가장 중요한 비교상의 단서: 지역 수치예보보다 모든 지표가 좋은가?

올림픽 자료의 더 강한 비교에서는 다음 결과가 보고된다. **[보충 p.14, Table 10]** [Springer Nature](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs42256-023-00667-9/MediaObjects/42256_2023_667_MOESM1_ESM.pdf)

| 모델 | MSE | MAE |
|---|---:|---:|
| 지역 NWP + MOS | 7.928 | **1.796** |
| 단일 Corrformer | **7.605** | 1.825 |
| Corrformer 앙상블 | **7.209** | **1.773** |

> **용어 — NWP:** 대기 상태와 물리방정식을 이용하는 수치예보다.  
> **용어 — MOS:** 수치예보 결과에 남는 체계적 오차를 통계적으로 보정하는 후처리다.  
> **용어 — 앙상블:** 여러 예측 결과를 결합하는 방식이다.

**[해석]** 단일 Corrformer는 **MSE에서 우수하지만 MAE에서는 열세**다. 앙상블은 입력길이가 다른 6개 모델을 결합하므로 단일 모델과 계산량·관측 이력 조건이 다르다. 따라서 가장 정확한 결론은 **“유력한 국지예측 대안이지만, 모든 평가 지표에서 일관되게 수치예보를 능가한 것은 아니다”**이다. [Springer Nature](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs42256-023-00667-9/MediaObjects/42256_2023_667_MOESM1_ESM.pdf)

---

## 6. 중요한 그림 선정과 해석

### 6.1 Figs.2·5 — 논문의 실제 기여를 가장 잘 보여주는 그림

**위치:** Fig.2는 p.604, Fig.5는 p.608.

**[저자 보고]** Fig.2는 트리·자기상관·교차상관의 개념을, Fig.5는 FFT 계산, 지연 선택, 정렬, 집계, 원상복구의 순서를 보여준다.  
Interpretable-weather-forecasti… Interpretable-weather-forecasti…

**[해석]** 핵심은 단순히 “공간 정보를 추가했다”가 아니다. **시간적으로 어긋난 비슷한 패턴을 먼저 정렬한 뒤 공간 정보를 결합한다**는 점이다.  
동시에 이 그림은 복잡한 공간 관계를 정렬된 사슬로 바꾸는 정보 압축이 일어남을 보여준다. 효율성의 원천과 표현력의 잠재적 제약이 같은 설계에서 나온다.

### 6.2 Fig.3 — 세계적인 자료와 세계적인 일반화는 다르다

**위치:** p.605.

**[저자 보고]** 전지구·중국 지역·경기장의 서로 다른 관측소 배치와 계층적 묶음을 제시한다. Interpretable-weather-forecasti…

**[해석]** 관측소 분포는 균일하지 않다. 이 그림은 다양한 공간 규모를 다룬다는 근거지만, 관측이 드문 지역이나 해양에서 동일한 성능을 갖는다는 근거는 아니다.  
후속 연구에서는 전체 평균뿐 아니라 **관측소 밀도·기후권·고도별 성능**을 함께 보고할 필요가 있다.

### 6.3 Fig.4a — 예측 사례로는 유용하지만 극한현상 검증으로는 부족하다

**위치:** p.607.

**[저자 보고]** 지역 풍속 변화와 한 관측소의 미래 24시간 곡선을 비교하여, Corrformer가 바람의 진행과 약화 과정을 더 잘 포착한다고 설명한다. **[해설 pp.608]** Interpretable-weather-forecasti… Interpretable-weather-forecasti…

**[해석]** 이 그림은 평균 오차만으로 보이지 않는 **변화의 도착 시점과 최고값 표현**을 보여준다는 장점이 있다. 그러나 극한풍 전체에 대한 탐지율, 오경보율, 최고값 편향이 없으면 “위험 기상을 안정적으로 잘 예측한다”는 결론까지 확장할 수 없다.

### 6.4 Fig.4b — 해석가능성의 핵심이면서 가장 조심해야 할 그림

**위치:** p.607; 본문 설명 p.609.

**[저자 보고]** 학습된 진행 방향이 ERA5에서 관찰되는 기온 변화와 닮았으며, 북유럽에서는 산맥과 관련된 것으로 설명한 지그재그 형태가 나타난다. Interpretable-weather-forecasti…

**[해석]** 시차가 존재한다고 반드시 공기가 그 방향으로 이동한 것은 아니다. 이를 보여주는 **해설용 반례**를 생각할 수 있다.

```math
u_s(t)
=
a_s\cos(\omega t+\phi_s)+b_s.
```

**기호:** $u_s(t)$는 위치 $s$의 신호, $a_s$는 진폭, $\omega$는 공통 주기 변화의 각주파수, $\phi_s$는 위치별 위상차, $b_s$는 평균 수준이다.

이 식에서는 위치 사이에 물리적 전달을 넣지 않아도 $\phi_s$가 다르면 교차상관에서 순서가 검출된다. 따라서 일변화의 위치별 위상차나 공통 외부 요인도 진행 방향처럼 보일 수 있다.

> **용어 — 위상차:** 같은 주기적 변화가 서로 다른 시점에 최고점이나 최저점에 도달하는 차이다.

이는 Fig.4b의 해석이 틀렸다는 증명이 아니라, **그림만으로 가능한 대안 설명을 배제할 수 없다는 지적**이다. 일변화 제거 전후 비교와 독립적인 바람장·이동속도 검증이 필요하다.

### 6.5 Extended Data Fig.4 — 효율성은 측정 조건과 함께 읽어야 한다

**위치:** 첨부 PDF p.14.

**[저자 보고]** 배치 크기 1, 내부 채널 512에서 관측소 수에 따른 메모리·실행시간을 측정하며, 실행시간은 1,000회 반복 평균이다. 그래프는 대체로 선형적인 증가를 보여준다.  Interpretable-weather-forecasti…

**[해석]** 이는 확장성의 좋은 실증 근거다. 다만 본문의 “약 3만 관측소를 0.5초 이내 예측”이라는 설명과 모든 설정을 동일시하면 안 된다.  
자료 수집, 품질관리, 트리 갱신까지 포함한 전체 서비스 지연시간도 별개다. **[본문 p.608]** Interpretable-weather-forecasti…

### 6.6 Extended Data Fig.5 — 일반화와 관련해 가장 중요한 그림

**위치:** 첨부 PDF p.15.

**[저자 보고]** Global은 350개, Regional은 3,404개 관측소로 학습한 뒤 관측소를 점진적으로 추가하면서 동일한 모델을 평가한다. Interpretable-weather-forecasti…

**[해석: 긍정적 의미]** 고정된 관측소 수에만 맞는 모델이 아니라, 새로운 관측소의 과거 정보를 받아 처리할 수 있음을 보여준다. Global wind와 Regional wind에서는 전체 관측소 학습 기준선에 가까운 결과를 보인다.

**[해석: 제한]** 다음은 구분해야 한다.

| 이 그림으로 뒷받침되는 것 | 이 그림만으로 뒷받침되지 않는 것 |
|---|---|
| 재학습 없이 관측소 수를 늘려 추론할 수 있다. | 학습하지 않은 대륙이나 기후권에서도 같은 성능을 유지한다. |
| 제한된 학습 관측소에서 얻은 매개변수를 다른 관측소에 적용할 수 있다. | 관측 이력이 없는 위치도 동일한 방식으로 예측할 수 있다. |
| 일부 과제에서 전체 학습 기준선에 가까운 오차를 보인다. | 모든 변수에서 전체 학습과 통계적으로 동등하다. |
| 관측망 확장 실험의 성능 곡선을 제공한다. | 관측소 추가 자체가 고정된 시험 대상의 성능을 개선한다. |

마지막 구분이 특히 중요하다. **가로축이 바뀌면 평가에 포함되는 관측소 집합도 달라지므로**, 곡선 변화에는 모델 효과뿐 아니라 평가 대상 구성의 변화가 섞일 수 있다. 또한 Global temp 곡선은 전체 관측소 학습 기준선보다 높은 오차를 보여, 완전한 동등성으로 해석하기 어렵다. **[Extended Data Fig.5]** Interpretable-weather-forecasti…

---

## 7. 통계적으로 취약한 부분과 직접 비교하면 안 되는 수치

### 7.1 통계적 신뢰성

실험은 5회 반복되었고, Corrformer MSE의 표준편차는 과제 순서대로 **0.075·0.219·0.132·0.024**이다. 따라서 반복 실험이나 변동성 보고가 전혀 없다는 비판은 맞지 않는다. **[보충 p.15, Table 12]** [Springer Nature](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs42256-023-00667-9/MediaObjects/42256_2023_667_MOESM1_ESM.pdf)

다만 다음 문제가 남는다.

| 점검 항목 | 판단 |
|---|---|
| **Table 1의 “95%·99% confidence interval”** | 각주에서는 우월성의 유의 수준으로 설명하지만, 실제 구간의 상·하한은 제시하지 않는다. 예보가 맞을 확률이나 예측구간의 포함률로 해석하면 안 된다. **[p.606, Table 1 각주]** Interpretable-weather-forecasti… |
| **유의성 검정의 재현 가능성** | 본문·보충자료에서 검정 종류, 독립 표본 단위, 다중비교 보정 방식을 명확히 확인하지 못했다. 따라서 유의성 표기를 독립적으로 재현했다고 말할 수 없다. |
| **학습 반복과 날씨 표본의 불확실성** | 같은 시험기간에서의 학습 반복 변동은 다른 해·다른 기상사건에서도 성능이 유지되는지에 대한 불확실성과 다르다. |
| **시공간 의존성** | 가까운 관측소와 연속된 시간의 오차를 모두 독립 표본으로 처리하면 근거가 과장될 수 있다. 이 논문 자체가 강한 시공간 상관을 전제로 한다. **[pp.602–603]** Interpretable-weather-forecasti… |
| **평균 성능과 취약 집단** | 전체 평균이 낮아도 특정 기후권·고산지대·극한현상의 성능이 나쁠 수 있다. 본문 Table 1만으로는 이를 판별할 수 없다. |

> **용어 — 다중비교:** 여러 모델·변수·예측시간을 동시에 비교하는 것이다. 비교 횟수가 많으면 우연히 좋아 보이는 결과가 나올 가능성도 커진다.

### 7.2 비교 불가능하거나 조건을 맞춰야 하는 수치

**첫째, 자료셋 간 오차를 직접 순위화하면 안 된다.** Global wind는 평균 풍속이고 Regional·Olympics wind는 최대 풍속이며, 기간·지역·관측밀도도 다르다. 예를 들어 3.889와 7.605의 차이만으로 전지구 풍속 예측이 올림픽 풍속 예측보다 두 배 정확하다고 말할 수 없다. **[pp.607–608]** Interpretable-weather-forecasti… Interpretable-weather-forecasti…

**둘째, 재분석자료와 실제 예보를 같은 종류의 예측으로 취급하면 안 된다.** Table 1은 저자가 “prediction”과 “reanalysis”로 표기한 자료를 함께 비교한다. 재분석은 과거 대기를 재구성한 자료이므로, 예보 발행 시점에 이용 가능한 정보만 사용하는 운영 예보와 구분해야 한다. **[p.606, Table 1; GenCast의 평가 조건 설명]** Interpretable-weather-forecasti… [Nature](https://www.nature.com/articles/s41586-024-08252-9)

> **용어 — 재분석:** 관측과 수치모델을 결합해 과거 대기 상태를 일관된 방식으로 다시 추정한 자료다. 원시 관측이나 당시 실제 발행된 예보와 같지 않다.

**셋째, 격자 예보를 관측소로 보간한 결과와 관측소를 직접 학습한 모델은 대표하는 공간 규모가 다르다.** 논문은 보간과 고도 보정을 수행하지만, 이 비교에는 모델의 예측능력뿐 아니라 격자–관측소 표현 차이도 포함된다. **[p.608]** Interpretable-weather-forecasti…

**넷째, 계절균형 분할은 독립적인 미래 연도 검증과 다르다.** 보충자료의 분기별 분할을 모으는 방식에서는 일부 시험시점보다 뒤의 다른 분기 자료가 학습에 포함될 수 있다. 이는 계절별 평가를 보완하지만, 전체 시간축에서 미래로만 예측하는 검증을 대체하지 않는다. **[보충 p.15, §J.5]** [Springer Nature](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs42256-023-00667-9/MediaObjects/42256_2023_667_MOESM1_ESM.pdf)

**추가 확인이 필요한 표시:** Extended Data Fig.3a–b의 제목에는 2019년 8월 사례가 적혀 있어, 주요 시간순 분할의 시험기간 사례로 읽기 어렵다. 날짜 오기나 별도 사례 선정일 수 있으므로 확인이 필요하지만, 이것만으로 본문 Table 1의 자료 누출을 단정할 수는 없다. **[첨부 PDF p.13]** Interpretable-weather-forecasti…

---

## 8. 2020년 이후 관련 연구 비교: 무엇이 발전했고, 무엇을 배워야 하는가?

아래는 **각 후속 연구 저자의 보고**를 요약한 것이다. 서로 다른 자료의 오차를 모아 하나의 종합 순위를 만든 것은 아니다.

### 8.1 방법과 일반화 관점의 비교

| 연구 | 해당 저자의 방법·결과 | Corrformer와의 차이 및 일반화에 대한 해석 |
|---|---|---|
| **Autoformer, 2021** | 분해 구조와 자기상관 기반 시계열 예측을 제안한다. **[§3.1–3.2, pp.3–5]** [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2021/file/bcc0d400288793e8bdcd7c19a8ac0c2b-Paper.pdf) | Corrformer의 직접적인 방법론적 기반이다. Corrformer의 추가 기여는 대규모 관측소 간 공간관계와 계층적 결합이다. |
| **Pangu-Weather·GraphCast, 2023** | Pangu는 3차원 대기 구조를 반영한 Transformer와 여러 시간 간격의 모델을 사용한다. GraphCast는 그래프 신경망으로 전지구 중기예보를 수행한다. **[Pangu Fig.1; GraphCast 초록, p.1416]** [Nature](https://www.nature.com/articles/s41586-023-06185-3) | 여러 상층·지표 변수를 가진 전지구 대기장 예측이 중심이다. Corrformer의 관측소 단기예측과 목적·입력·평가지표가 달라 **원시 오차의 직접 비교는 불가**하다. 다만 대규모 대기 정보를 관측소 모델의 조건으로 제공하는 방향은 유용하다. |
| **MetNet-3, 2023** | 관측소 일부를 입력에서 가리는 훈련과, 학습에 사용하지 않은 관측소 평가를 구분한다. 24시간 다변수 예측을 수행하며 NWP 초기 상태도 입력에 사용한다. **[Fig.1, §2.2]** [arXiv](https://arxiv.org/pdf/2306.06079) | Corrformer 후속 연구에 특히 중요한 비교다. **관측 이력이 제공되는 새 관측소**와 **입력에도 없는 위치**를 구별해 평가하는 설계를 참고할 수 있다. |
| **GenCast, 2024 온라인 발표·2025 권호** | 조건부 확산모델로 전지구 15일 앙상블을 생성하며, 개별 지점의 분포뿐 아니라 함께 발생하는 날씨 시나리오를 평가한다. **[Figs.1·3]** [Nature](https://www.nature.com/articles/s41586-024-08252-9) | Corrformer의 시점별 확률 출력에서 더 나아가, 여러 관측소·시간의 오차가 함께 움직이는 구조를 모델링할 필요성을 보여준다. |
| **Aurora, 2025** | 다양한 지구과학 자료로 사전학습한 뒤 기상·대기질·파랑 등으로 미세조정한다. **[Figs.1·5]** [Nature](https://www.nature.com/articles/s41586-025-09005-y) | 일반화는 모델 구조뿐 아니라 **학습자료의 다양성과 전이학습**에서 얻을 수 있다는 방향이다. 그러나 미세조정 성공이 임의의 새 관측소에 대한 무학습 예측을 의미하지는 않는다. |
| **Aardvark Weather, 2025** | 관측 입력 → 대기 상태 추정 → 미래 대기장 → 관측소 예측의 전체 연결을 학습한다. 학습에는 ERA5를 이용하지만, 배치 시 NWP 입력 없이 예측하도록 설계한다. **[Figs.1·5]** [Nature](https://www.nature.com/articles/s41586-025-08897-0) | 관측소만의 과거 정보에서 출발한 Corrformer보다 입력의 대기 정보 범위를 넓힌다. 관측소 출력에 맞춘 전체 모델 조정이라는 방향이 중요하지만, 관측소 정확도 평가와 미관측 위치 일반화는 별개다. |
| **S²Transformer, 2025 공개 원고·2026 게재정보** | 지역 내부·지역 사이의 구조화된 주의집중으로 관측소 관계를 처리한다. 공개 원고 Table 2의 NCEI 풍속 MSE는 Corrformer **3.89**, S²Transformer **3.61**이며 MAE는 둘 다 표시 정밀도에서 **1.30**이다. **[p.8, Table 2]** [arXiv](https://arxiv.org/pdf/2509.19648) | 관측소 예측이라는 직접적인 후속 비교다. **모든 지표에서 개선된 것은 아니며**, 다른 자료에서의 성능을 새로운 기후권에 대한 무학습 전이로 읽으면 안 된다. |
| **SIGMAformer, 2026** | 가우스 혼합 패턴 추출과 동적 시공간 상관을 결합하며, Corrformer 대비 풍속 MSE **1.83%**, MAE **2.68%** 감소를 보고한다. **[Table 1, Fig.6]** [Nature](https://www.nature.com/articles/s41612-026-01385-w) | 패턴에 따라 관계를 달리 처리하는 직접 후속 방향이다. 다만 Corrformer 등 기존 비교값은 원 연구에서 가져왔다고 명시하므로, 동일 환경에서 모든 모델을 재실행한 비교와 구분해야 한다. |

> **용어 — 주의집중, attention:** 여러 입력 중 현재 예측에 어떤 정보를 얼마나 사용할지 가중치를 정하는 방식이다.  
> **용어 — 확산모델:** 잡음으로 시작한 값을 반복적으로 정제해 가능한 미래 표본을 만드는 생성모델이다.  
> **용어 — 사전학습·미세조정:** 넓은 자료로 공통 특징을 먼저 학습하고, 이후 특정 지역·변수·목적에 맞게 추가 학습하는 과정이다.  
> **용어 — 가우스 혼합:** 여러 종 모양 분포의 결합으로 복잡한 패턴을 표현한다. 이것을 사용했다고 예측 불확실성이 자동으로 잘 보정되는 것은 아니다.

### 8.2 일반화 관점에서 중요한 변화

**관측소 추가에서 관측 결손과 미관측 위치로.** MetNet-3의 Fig.1은 입력에서 관측소를 가리는 훈련, 보지 못한 위치에서의 평가, 관측값을 제공하는 국지평가를 구분한다. **[해석]** Corrformer의 다음 실험도 “새 관측소”라는 하나의 이름으로 묶지 말고, **그 관측소의 과거 값을 제공하는지**부터 구분해야 한다. [arXiv](https://arxiv.org/pdf/2306.06079)

**단일 자료셋 학습에서 다양한 자료의 공통 학습으로.** Aurora는 서로 다른 변수·해상도·자료원으로 학습한 표현을 다른 과제로 조정한다. **[해석]** Corrformer 역시 관측소 수만 늘리는 것보다 여러 해·기후권·변수를 학습하는 것이 일반화에 더 중요한 연구 축일 수 있다. 다만 실제 개선은 별도 실험으로 확인해야 한다. **[Aurora Fig.1]** [Nature](https://www.nature.com/articles/s41586-025-09005-y)

**평균 오차에서 사건·운영 조건의 평가로.** SIGMAformer는 관측소별 훈련자료의 상위 5% 기준으로 극한사건을 정의해 별도 평가한다. **[해석]** 이러한 사건별 검증은 Corrformer의 사례 그림을 정량적 극한현상 평가로 발전시키는 데 유용하다. **[SIGMAformer Table 2]** [Nature](https://www.nature.com/articles/s41612-026-01385-w)

**더 긴 관측 기록과 최근 운영 자료로.** WEATHER-5K는 5,672개 관측소의 10년 자료를 제공하고, RealBench의 2026년 사전공개 연구는 2025년 자료와 운영 초기 상태·실측 관측·극한사건 평가를 강조한다. **[해석]** 다만 특정 연도가 진정한 학습 밖 자료인지는 각 모델의 학습 종료 시점을 확인해야 하며, 단순히 “2025년 자료”라는 이유만으로 모든 모델에 독립 시험자료가 되는 것은 아니다. [arXiv](https://arxiv.org/html/2406.14399v1)

### 8.3 이 논문의 후속 연구에 대한 영향

**확인 가능한 영향**은 S²Transformer와 SIGMAformer처럼 Corrformer를 직접 비교 대상으로 삼고, 관측소 간 관계 모델링을 발전시키는 연구가 이어졌다는 점이다. 반면 Aurora나 Aardvark가 Corrformer에서 직접 파생되었다고 단정할 근거는 여기서 제시하지 않는다. [arXiv](https://arxiv.org/pdf/2509.19648)

**[종합 해석]** Corrformer가 남긴 중요한 문제 설정은 “관측소 예측을 수많은 독립 시계열의 집합으로 보지 말고, 공유되는 시공간 관계를 가진 하나의 예측 문제로 다루자”는 것이다. 이후 연구의 경쟁점은 단순한 평균 오차 개선을 넘어, **새 위치·새 기후·관측 결손·불확실성에 대해 어떤 관계 표현이 유지되는가**로 이동할 필요가 있다.

---

## 9. 문서가 충분히 답하지 않는 질문

| 질문 | 아직 필요한 증거 |
|---|---|
| **학습하지 않은 대륙이나 기후권에서도 성능이 유지되는가?** | 인접 관측소가 학습에 섞이지 않는 지리적 구역 제외 실험이 필요하다. Extended Data Fig.5의 관측소 추가 실험과는 다르다. |
| **새 관측소에 과거 48시간 기록이 없으면 어떻게 예측하는가?** | 현재 문제 정의는 과거 관측을 입력으로 요구한다. 관측이 없는 위치에 대한 별도 추론 구조와 평가가 필요하다. **[p.603]** Interpretable-weather-forecasti… |
| **관측소 고장이나 관측망 급변에 얼마나 강한가?** | 관측소 전체 제거, 연속 결측, 센서 편향, 트리 변경 전후의 성능·비용 평가가 필요하다. |
| **결측값 처리에서 미래 정보가 사용되지 않았는가?** | 보충 B.1의 평균 대체에 사용한 통계가 훈련구간에서만 계산됐는지, 대체된 시험 정답을 평가에서 제외했는지 확인이 필요하다. **[보충 p.3]** [Springer Nature](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs42256-023-00667-9/MediaObjects/42256_2023_667_MOESM1_ESM.pdf) |
| **학습된 진행 방향은 실제 전달 경로인가?** | 공통 일변화와 외부 요인을 통제한 분석, 독립적인 이동 방향·속도 자료와의 정량 비교가 필요하다. |
| **몇 년 뒤의 기후나 드문 극한현상에서도 성능이 유지되는가?** | 단일한 2년 기록의 분할을 넘어 여러 미래 연도를 순차 평가해야 한다. **[자료 범위 pp.607–608]** Interpretable-weather-forecasti… Interpretable-weather-forecasti… |
| **2022년 실제 서비스의 효과는 어느 정도였는가?** | 대회 기간 전체의 예측 기록, 경보 기준, 누락·오경보, 운영 지연, 의사결정 개선 자료가 필요하다. 배치 사실만으로는 정량 효과를 알 수 없다. |
| **완전한 독립 재현이 가능한가?** | 저자는 코드와 전지구 자료의 공개 경로를 제시하지만, 지역·올림픽 전체 자료에는 CMA의 허가가 필요하다고 명시한다. **[p.610]** Interpretable-weather-forecasti… |

---

## 10. 결론과 후속 연구: 일반화 성능 향상을 중심으로

### 10.1 저자들이 직접 제시한 시사점과 계획

**[저자 보고]** 관측소의 부분 관측만으로도 대규모 공동예측과 해석 가능한 관계 추출이 가능하며, 데이터 기반 AI가 기상과학에 유용한 접근이 될 수 있다고 결론짓는다. 명시적인 후속 계획은 **데이터 기반 방법과 물리정보 기반 방법의 결합을 탐색하여 Corrformer를 개선하는 것**이다. **[p.609, Conclusion]** Interpretable-weather-forecasti…

> **용어 — 물리정보 기반 학습:** 관측자료뿐 아니라 물리법칙, 지형, 보존 성질, 알려진 이동 특성 등을 모델 구조나 학습 조건에 반영하는 접근이다.

아래는 이 계획을 구체화한 **제 연구 제안**이며, 원논문이 이미 입증한 성능 향상이 아니다.

### 10.2 우선순위 1: 일반화를 먼저 정확히 측정한다

가장 먼저 해야 할 일은 모델을 크게 만드는 것이 아니라 **일반화의 종류를 분리하는 것**이다.

| 검증 축 | 권장 실험 |
|---|---|
| 시간 일반화 | 과거 여러 해로 학습하고 이후의 독립된 해를 순차 평가 |
| 관측소 일반화 | 학습에서 제외한 관측소를 평가하되, 과거 관측 제공 여부를 분리 |
| 지역·기후 일반화 | 특정 지역·기후권 전체를 학습에서 제외 |
| 운영 일반화 | 실제 자료 도착 지연, 결측, 센서 교체, 관측망 변경을 반영 |
| 극한현상 일반화 | 전체 평균과 별도로 강풍·폭염·급변 사건을 평가 |

오차의 신뢰구간은 관측값 하나씩이 아니라 **기상사건·시간 구간·공간 구역을 묶어 재표집**하는 방식으로 추정하는 것이 타당한 출발점이다. 이는 서로 연관된 관측을 독립 표본처럼 취급하는 문제를 줄이기 위한 제안이다.

> **용어 — 블록 부트스트랩:** 연속된 자료 묶음을 통째로 다시 뽑아 불확실성을 추정하는 방법이다. 시간적·공간적 의존성을 어느 정도 유지하려는 목적이 있다.

### 10.3 우선순위 2: 관측 누락과 불균등한 지역 분포를 학습에 반영한다

MetNet-3의 관측소 가리기 훈련을 참고하여, Corrformer도 일부 관측소나 연속 시간 구간이 없는 상태를 학습하게 할 수 있다. 다만 입력 결측과 평가 정답의 결측은 구별해야 한다. [arXiv](https://arxiv.org/pdf/2306.06079)

**[제안 수식: 원논문에 없는 학습 목적]**

```math
\mathcal L_{\text{obs}}
=
\frac{
\sum_{e\in\mathcal B}
m_e\,w_{g(e)}\,(\hat y_e-y_e)^2
}{
\sum_{e\in\mathcal B}
m_e\,w_{g(e)}
}.
```

**기호:** $\mathcal B$는 한 학습 묶음의 예측 항목 집합, $m_e$는 실제 정답 관측이 있으면 1이고 없으면 0인 표시다.  
$g(e)$는 해당 항목의 지역·기후 집단, $w_{g(e)}>0$은 집단별 가중치다. 분모가 양수인 묶음에 적용한다.

이 목적은 **보간한 값을 실제 관측처럼 학습하는 문제를 줄이고, 관측소가 많은 지역에 학습이 편중되는 현상을 완화**하려는 것이다.  
다만 지역 간 균형을 높이는 것이 기존 전체 평균을 반드시 개선하는 것은 아니므로, 전체 평균과 지역별 평균을 함께 보고해야 한다.

### 10.4 우선순위 3: 단일 pivot·단일 순서의 제약을 완화한다

**[제안]** 고정된 지리적 계층구조를 유지하되, 한 지역에서 여러 pivot과 여러 유력한 지연을 사용할 수 있다.  
이렇게 하면 서로 다른 기상 과정이 겹칠 때 하나의 평균과 하나의 순서에 과도하게 의존하는 문제를 줄일 가능성이 있다.

물리정보는 단순히 손실함수에 항을 하나 더 넣는 방식보다, **어떤 연결과 지연이 해당 변수에서 물리적으로 타당한지**를 검토하는 데 활용하는 편이 중요하다.  
기온의 위상 이동, 실제 바람의 이동, 기압계의 이동은 동일한 현상이 아니기 때문이다.

이 방향은 구조화된 공간관계를 확장한 S²Transformer와 패턴별 관계를 강조한 SIGMAformer와 연결되지만, 복잡도를 늘리는 만큼 고정된 계산 예산에서 실제 일반화 개선이 있는지 검증해야 한다. [arXiv](https://arxiv.org/pdf/2509.19648)

### 10.5 우선순위 4: 전지구 대기 정보를 관측소 보정과 결합한다

**[제안]** 관측소의 최근 과거만으로는 멀리서 접근하는 대규모 기상 변화를 충분히 알기 어려울 수 있다.  
전지구 예보를 배경 정보로 사용하고 Corrformer 계열 모델이 관측소 수준의 차이를 보정하는 구성을 검토할 수 있다.  
전지구 상태와 국지 출력의 연결이라는 관점에서는 Aardvark도 중요한 참고다. **[Aardvark Figs.1·5]** [Nature](https://www.nature.com/articles/s41586-025-08897-0)

**제안 수식:**

```math
\hat y_{s,t+h}
=
\text{Interp}(F_{t,h},r_s)
+
g_\theta
\left(
X_{t-I+1:t},
r_s,
m,
F_{t,h}
\right).
```

**기호:** $s$는 관측소, $h$는 예측 선행시간, $F_{t,h}$는 시각 $t$에 생성한 전지구 예보장, $r_s$는 관측소 위치다.  
$\text{Interp}$는 예보장을 해당 위치로 변환하는 연산, $g_\theta$는 관측소 자료를 이용한 잔차 보정 모델, $m$은 입력 결측 정보다.

> **용어 — 잔차 보정:** 배경 예보에서 남은 오차를 별도 모델이 추정해 더하거나 빼는 방식이다.

이는 원래의 “관측소 자료만으로 예측”이라는 조건을 바꾸는 제안이다. 따라서 성능이 좋아지더라도 **추가 입력·계산 자원 덕분인지, 구조 개선 덕분인지**를 구분해야 한다.

### 10.6 우선순위 5: 평균 예측뿐 아니라 함께 발생하는 위험을 모델링한다

**[제안]** 확률예측을 더 발전시킬 때는 각 지점의 분포만이 아니라 여러 관측소에서 강풍이 동시에 발생할 가능성, 강풍이 얼마나 지속되는지, 이동 경로의 불확실성을 평가해야 한다. GenCast의 공동 시나리오 생성은 이 방향의 중요한 비교 대상이다. **[GenCast Figs.1·3]** [Nature](https://www.nature.com/articles/s41586-024-08252-9)

> **용어 — 공동분포:** 여러 위치와 시간의 값이 어떤 조합으로 함께 나타나는지 설명하는 확률분포다.  
> **용어 — 확률 보정:** 예를 들어 80% 확률이라고 예측한 사건이 장기적으로 실제 약 80% 발생하는지를 뜻한다.

대형 생성모델을 즉시 도입하기보다는, 먼저 관측소 사이에 연관된 오차 표본을 만드는 비교적 단순한 확장부터 평가할 수 있다. 평균 오차, 사건 탐지, 확률 보정, 실행비용을 함께 비교해야 확장의 실질적 가치를 판단할 수 있다.

### 최종 판단

**Corrformer는 “수만 개의 불규칙 관측소를 하나의 효율적인 예측 구조로 다룰 수 있다”는 점에서 의미 있는 연구다.**  
본문 비교군에 대한 오차 개선과 관측소 수 증가에 대한 적응 실험은 이 기여를 뒷받침한다. Interpretable-weather-forecasti…

그러나 **관측망 확장에 대한 적응**, **새 기후·지역으로의 일반화**, **물리적 인과관계의 발견**은 서로 다른 주장이다.  
후속 연구의 가장 중요한 목표는 이들을 분리해 검증하면서, 장기간·다변수 학습과 물리적 배경 정보가 어떤 조건에서 일반화 성능을 실제로 높이는지 밝히는 것이다.

---

## 참고자료: 분석에 직접 사용한 원문과 출처

| 번호 | 자료 제목·서지정보 | 확인한 출처 |
|---|---|---|
| 1 | Wu et al. **Interpretable weather forecasting for worldwide stations with a unified deep model**. *Nature Machine Intelligence* 5, 602–611, 2023. DOI: 10.1038/s42256-023-00667-9. | 사용자 첨부 원문 및 Extended Data. Interpretable-weather-forecasti… |
| 2 | **Supplementary Information for “Interpretable weather forecasting for worldwide stations with a unified deep model”**. 2023. | Nature / Springer Nature 보충자료. [Springer Nature](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs42256-023-00667-9/MediaObjects/42256_2023_667_MOESM1_ESM.pdf) |
| 3 | Wu et al. **Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting**. NeurIPS, 2021. | NeurIPS 공식 논문집. [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2021/file/bcc0d400288793e8bdcd7c19a8ac0c2b-Paper.pdf) |
| 4 | Lam et al. **Learning skillful medium-range global weather forecasting**. *Science* 382, 1416–1421, 2023. | 저자 공개 원고, arXiv:2212.12794 — GraphCast. [arXiv](https://arxiv.org/abs/2212.12794) |
| 5 | Bi et al. **Accurate medium-range global weather forecasting with 3D neural networks**. *Nature* 619, 533–538, 2023. | Nature — Pangu-Weather. [Nature](https://www.nature.com/articles/s41586-023-06185-3) |
| 6 | Andrychowicz et al. **Deep Learning for Day Forecasts from Sparse Observations**. 2023. | arXiv:2306.06079 — MetNet-3. [arXiv](https://arxiv.org/pdf/2306.06079) |
| 7 | Price et al. **Probabilistic weather forecasting with machine learning**. *Nature* 637, 84–90, 2025; 온라인 발표 2024. | Nature — GenCast. [Nature](https://www.nature.com/articles/s41586-024-08252-9) |
| 8 | Bodnar et al. **A foundation model for the Earth system**. *Nature* 641, 1180–1187, 2025. | Nature — Aurora. [Nature](https://www.nature.com/articles/s41586-025-09005-y) |
| 9 | Allen et al. **End-to-end data-driven weather prediction**. *Nature* 641, 1172–1179, 2025. | Nature — Aardvark Weather, 현재 온라인 저자 표기 기준. [Nature](https://www.nature.com/articles/s41586-025-08897-0) |
| 10 | Chen et al. **S²Transformer: Scalable Structured Transformers for Global Station Weather Forecasting**. | 수치 분석: arXiv:2509.19648 공개 원고. 2026 TMLR 게재정보: 공저자 Xiucheng Li의 공식 연구 목록. [arXiv](https://arxiv.org/pdf/2509.19648) |
| 11 | Kim & Suk. **SIGMAformer: a spatiotemporal Gaussian mixture correlation transformer for global weather forecasting**. *npj Climate and Atmospheric Science*, 2026. | Nature / npj Climate and Atmospheric Science. [Nature](https://www.nature.com/articles/s41612-026-01385-w) |
| 12 | Han et al. **WEATHER-5K: A Large-scale Global Station Weather Dataset Towards Comprehensive Time-series Forecasting Benchmark**. 2024. | arXiv:2406.14399. [arXiv](https://arxiv.org/html/2406.14399v1) |
| 13 | Li et al. **RealBench: Benchmarking Data-Driven Numerical Weather Forecasting Under Operational Conditions and Extreme Event Challenges**. 2026. | arXiv:2605.24945, 사전공개 연구. [arXiv](https://arxiv.org/abs/2605.24945) |
