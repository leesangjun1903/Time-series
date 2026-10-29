# FreEformer: Frequency Enhanced Transformer for Multivariate Time Series Forecasting

**분석 대상:** *FreEformer: Frequency Enhanced Transformer for Multivariate Time Series Forecasting*, arXiv:2501.13989v1, 2025년 1월 23일 공개본입니다. 아래 페이지 번호는 **첨부 PDF의 첫 장을 1쪽으로 센 번호**이며, Figure·Table·식 번호도 첨부본을 기준으로 합니다.

원문 파일: 2501.13989v1.pdf

**구분 원칙:** **[저자 보고]**는 원문에 제시된 내용, **[해석]**은 그 결과나 수식에서 도출한 분석, **[제안]**은 아직 검증되지 않은 후속 실험입니다. 수치는 원문 표를 사용했으며, 직접 계산한 비율은 별도로 표시했습니다. 관련 연구 비교는 **2026년 10월 29일 기준으로 확인한 원 논문과 공식 학회 자료**를 사용합니다.

---

## 1. Executive summary — 8문장

1. **FreEformer는 각 변수의 전체 주파수 스펙트럼을 하나의 표현으로 만들고, 변수 간 관계를 Transformer로 학습하는 다변량 시계열 예측 모델입니다.** [pp.1–4, Figure 2]
2. 저자들은 일반적인 attention을 주파수 표현에 적용하면 attention 행렬의 랭크가 낮아지는 현상을 관찰하고, 이를 표현 다양성을 제한하는 문제로 제시합니다. [p.4, Figures 3–4]
3. 해결책은 softmax attention 행렬에 양수로 변환한 학습 가능 행렬을 더한 다음, 각 행을 $L_1$ 정규화하는 **enhanced attention**입니다. [p.5, 식 (9)]
4. 모델은 입력 정규화, 푸리에 변환, 실수부·허수부의 독립적인 Transformer 처리, 역푸리에 변환, 시간 영역 예측 헤드로 구성됩니다. [pp.3–4, 식 (5)–(7)]
5. 저자들은 18개 데이터셋에서 평가했으며, 특히 PEMS 계열과 여러 MAE 결과에서 강한 성능을 보고합니다. [p.6, Tables 2–3; pp.21–23, Tables 13–15]
6. 제안 attention을 다른 모델에 적용한 실험에서는 iTransformer와 PatchTST의 평균 MSE가 각각 5.9%, 9.9% 개선되었다고 보고합니다. [p.7, Table 9]
7. ECL·Traffic에서는 학습 데이터를 5%까지 줄여도 비교 모델보다 낮은 MAE를 보이지만, 이는 두 데이터셋의 특정 예측 설정에서 확인된 결과입니다. [p.19, Figure 13]
8. **[해석] 이 논문은 주파수 기반 변수 표현과 attention 보강의 유용성을 뒷받침하지만, 모든 데이터셋·지표에서의 최고 성능, 랭크 증가에 따른 일반화 향상, 새로운 데이터셋으로의 zero-shot 전이를 보장하지는 않습니다.** [Tables 2–3, 9, 21; 식 (10), (13)]

> **용어 풀이**
>
> - **다변량 시계열:** 여러 센서나 지표가 시간에 따라 함께 변하는 데이터.
> - **주파수 스펙트럼:** 신호를 느린 변화와 빠른 진동 등 여러 주파수 성분으로 나눈 표현.
> - **Attention:** 다른 입력에서 가져올 정보의 비중을 계산하는 연산.
> - **랭크(rank):** 행렬이 표현할 수 있는 서로 독립적인 방향의 수.
> - **일반화:** 학습에 사용하지 않은 데이터에서도 성능을 유지하는 능력.
> - **Zero-shot:** 대상 데이터셋으로 추가 학습하지 않고 예측하는 설정.

---

## 2. 연구 배경·목표·필요성

### 2.1 해결하려는 문제

**[저자 보고] 첫 번째 문제는 시계열의 전역적인 패턴을 어떻게 표현할 것인가입니다.**  
저자들은 인접한 시간 구간의 관측값 모양이 달라도, 정규화한 주파수 스펙트럼에서는 유사한 성분이 나타날 수 있다고 봅니다.  
Figure 1은 ECL과 PEMS04의 일부 사례를 통해 이러한 관찰을 제시합니다. [p.1, Figure 1]

**두 번째 문제는 여러 변수의 주파수 표현 사이에서 유용한 관계를 학습하는 것입니다.**  
기존 주파수 모델의 선형층 중심 처리와 주파수 패치 기반 처리에서 더 나아가, 각 변수의 전체 스펙트럼을 토큰으로 만들어 변수 간 의존성을 학습하려고 합니다. [pp.1–2, §§1–2]

> **용어 풀이 — 토큰·패치**
>
> **토큰**은 Transformer가 하나의 처리 단위로 받아들이는 벡터입니다. FreEformer에서는 한 변수의 전체 주파수 정보를 담습니다. **패치**는 연속된 시간점이나 주파수 구간을 작게 나눈 묶음입니다.

**세 번째 문제는 주파수 표현에 일반 attention을 적용했을 때 나타나는 낮은 랭크입니다.**  
저자들은 주파수 스펙트럼의 희소성과 softmax가 큰 값에 집중하는 성질을 가능한 원인으로 제시합니다.  
다만 이 원인 설명은 관찰에 기반한 가설이며, “주파수의 희소성은 반드시 낮은 attention 랭크를 만든다”는 일반 정리로 증명되지는 않습니다. [p.4, Figure 3, §4.2]

> **용어 풀이 — 희소성·softmax**
>
> **희소성**은 정보나 에너지가 일부 성분에 집중된 상태입니다. **Softmax**는 여러 점수를 양수 가중치로 바꾸고 합을 1로 만드는 함수로, 점수 차이가 크면 일부 항목에 가중치가 집중될 수 있습니다.

### 2.2 연구의 목적과 필요성

연구의 목적은 다음 세 가지로 정리할 수 있습니다.

| 목적 | 논문에서 취한 접근 | 근거 |
|---|---|---|
| 전역적인 시간 패턴 표현 | 시간축 DFT로 전체 주파수 스펙트럼 구성 | pp.2–4, 식 (1)–(6) |
| 변수 간 관계 학습 | 전체 스펙트럼을 변수 토큰으로 만들고 변수축 attention 적용 | pp.3–4, Figure 2 |
| 낮은 attention 랭크 보완 | softmax 이후 학습 행렬 추가와 행별 정규화 | p.5, 식 (9)–(13) |

**[해석] 이 연구의 차별점은 주파수 영역을 사용한다는 사실 자체보다, “전체 스펙트럼의 변수 토큰화”와 “입력 기반 attention에 데이터셋에서 학습한 관계를 더하는 방식”의 결합에 있습니다.** 주파수 표현의 효과와 attention 수정의 효과를 나누어 평가해야 연구의 기여를 정확하게 판단할 수 있습니다. [pp.3–7, Tables 5–9]

---

## 3. 핵심 주장과 근거

| 핵심 주장 | 저자가 제시한 근거 | 근거가 뒷받침하는 범위와 주의점 | 위치 |
|---|---|---|---|
| 주파수 표현은 예측에 유용한 전역 정보를 제공한다 | 인접 구간 스펙트럼 시각화, 시간 영역 모델과의 비교 | 선택된 사례와 실험 설정의 근거이며, 모든 시계열의 스펙트럼 안정성을 입증하지는 않음 | Figure 1, p.1; Table 6, p.7 |
| 전체 스펙트럼의 변수 간 관계 학습이 효과적이다 | 선형층·Transformer·학습 차원·패치 사용 여부 비교 | 시험한 구조 중 최종 구성이 대체로 우수 | Table 5, p.7 |
| 실수부·허수부를 별도 가중치로 처리하는 방식이 유리하다 | 결합 처리, 가중치 공유, 가중치 비공유 비교 | 비공유 방식이 최저 또는 동률이나, 일반적인 복소 신경망 전체에 대한 우위는 아님 | Table 1, p.4 |
| 학습 행렬 추가가 attention의 랭크를 높인다 | 랭크 부등식과 행렬 시각화 | 예시에서 증가하지만, 부등식만으로 모든 경우의 증가를 보장하지 않음 | 식 (10), p.5; Figure 4, p.4 |
| 새로운 attention이 gradient 조절을 유연하게 한다 | Jacobian 유도와 시각화 | 추가 학습 경로와 스케일 조절은 확인되지만, gradient 소실 해결이나 일반화 향상의 보장은 아님 | 식 (11)–(13), Figure 5, p.5 |
| 다양한 예측 벤치마크에서 높은 성능을 낸다 | 18개 데이터셋의 MSE·MAE 비교 | PEMS와 MAE에서 강하나, 모든 데이터셋·지표에서 최고는 아님 | Tables 2–4, p.6 |
| 다른 Transformer에도 적용할 수 있다 | 기존 모델의 attention만 교체 | 모듈의 적용 범용성을 뒷받침하며, 데이터셋 간 전이 일반화와는 구분해야 함 | Table 9, p.7 |
| 적은 학습 데이터에서도 유리하다 | 학습 비율을 100%에서 5%까지 축소 | Figure 13의 직접 근거는 ECL·Traffic에 한정 | Figure 13, p.19 |

> **용어 풀이 — Gradient·Jacobian**
>
> **Gradient**는 오차를 줄이려면 파라미터를 어느 방향으로 바꿔야 하는지 알려주는 미분값입니다. **Jacobian**은 입력이나 파라미터의 각 성분을 조금 바꿀 때 출력의 각 성분이 얼마나 달라지는지 모은 행렬입니다.

---

## 4. 연구 방법론과 데이터

### 4.1 데이터 소스

**[저자 보고] 실제 결과가 제시된 데이터셋은 다음 18개입니다.** 데이터 설명은 Appendix D, 통계는 Table 11에 있습니다. [pp.14–15]

| 데이터셋 | 변수 수 | 내용·관측 간격 |
|---|---:|---|
| ETTh1, ETTh2 | 각각 7 | 전력 변압기 관련 측정값, 1시간 |
| ETTm1, ETTm2 | 각각 7 | 전력 변압기 관련 측정값, 15분 |
| Weather | 21 | Max Planck 기상 관측 자료, 10분 |
| ECL | 321 | 고객별 전력 소비, 1시간 |
| Traffic | 862 | 도로 점유율, 1시간 |
| Exchange | 8 | 환율, 1일 |
| Solar-Energy | 137 | 태양광 발전량, 10분 |
| PEMS03 | 358 | 교통 자료, 5분 |
| PEMS04 | 307 | 교통 자료, 5분 |
| PEMS07 | 883 | 교통 자료, 5분 |
| PEMS08 | 170 | 교통 자료, 5분 |
| ILI | 7 | 인플루엔자 유사 질환 관련 자료, 1주 |
| COVID-19 | 55 | 코로나19 입원 자료, 1일 |
| METR-LA | 207 | 교통 센서 자료, 본문 설명은 5분 |
| NASDAQ | 12 | 주가지수와 경제 지표, 1일 |
| Wiki | 99 | 선택한 Wikipedia 문서의 조회수, 1일 |

**원문 내부의 주의점은 두 가지입니다.**

- p.5의 데이터셋 열거에는 **ECG**도 들어가지만, Table 11과 결과표에는 해당 실험이 없습니다. 위 18개에는 ECG를 포함하지 않았습니다.
- METR-LA의 관측 간격은 p.14 본문에서 **5분**, Table 11에서는 **1일**로 다르게 기재되어 있습니다.

또한 COVID-19는 전체 길이 335, ILI는 966, Wiki는 730으로 보고됩니다. 이러한 짧은 데이터셋의 작은 성능 차이는 반복 실험과 평가 구간의 영향을 특히 주의해서 해석해야 합니다. [p.15, Table 11]

### 4.2 분할·학습·평가 설정

| 항목 | 원문 설정 |
|---|---|
| 데이터 분할 | 시간순으로 학습·검증·테스트 분할 |
| ETT·PEMS | 6:2:2 |
| COVID-19의 S2 설정 | 6:2:2 |
| 그 외 | 7:1:2 |
| 장기 예측의 기본 입력 길이 | $T=96$ |
| 일반 장기 예측 길이 | $\tau\in\{96,192,336,720\}$ |
| PEMS 예측 길이 | $\tau\in\{12,24,48,96\}$ |
| 단기 S1 | 입력 12, 예측 $\{3,6,9,12\}$ |
| 단기 S2 | 입력 36, 예측 $\{24,36,48,60\}$ |
| 학습 환경 | PyTorch, NVIDIA RTX 4090 한 대 |
| 최적화 | Adam, 초기 학습률 $\{10^{-4},5\times10^{-4}\}$ |
| 주요 차원 | $d=16$, $D\in\{128,256,512\}$ |
| 학습 종료 | 최대 50 epoch, 검증 성능이 10 epoch 연속 개선되지 않으면 조기 종료 |
| 주요 평가 지표 | MSE, MAE |
| 비교 모델 수치 | 원 논문 보고값을 사용하거나, 없으면 공식 코드 실행 |

**근거:** p.6, Tables 2–3; p.15, Appendices D–E.

> **용어 풀이**
>
> - **입력 길이·lookback:** 예측할 때 참고하는 과거 관측의 길이.
> - **예측 길이·horizon:** 미래 몇 시점까지 예측하는지 나타내는 길이.
> - **MSE:** 오차의 제곱을 평균한 값으로, 큰 오차에 민감한 지표.
> - **MAE:** 오차의 절댓값을 평균한 지표.
> - **Epoch:** 학습 데이터를 한 차례 모두 사용하는 학습 단위.

**중요한 구분:** Tables 2–3의 평균은 기본적으로 **서로 다른 네 예측 길이에 대한 평균**입니다. 네 번의 독립적인 반복 실험 평균을 의미하지 않습니다.

---

## 5. 제안 방법과 모델 구조 — 수식 중심 설명

### 5.1 전체 구조와 기호

원문은 푸리에 변환 설명에서 $N$을 신호 길이로 사용하다가, 모델 설명에서는 변수 수로 사용합니다. 혼동을 피하기 위해 아래에서는 **$N$을 변수 수, $T$를 입력 길이로 통일**합니다.

| 기호 | 의미 |
|---|---|
| $X\in\mathbb{R}^{N\times T}$ | 입력 다변량 시계열 |
| $N$ | 변수 수 |
| $T$ | 과거 입력 길이 |
| $\tau$ | 미래 예측 길이 |
| $d$ | 차원 확장 크기, 기본값 16 |
| $D$ | Transformer의 은닉 차원 |
| $L$ | Transformer 블록 수 |
| $F=\lfloor T/2\rfloor+1$ | 실수 입력에서 보존할 독립 주파수 계수 수 |
| $\hat Y\in\mathbb{R}^{N\times\tau}$ | 최종 예측 |

**[저자 보고] 본문 식 (5)–(7)을 기준으로 한 처리 순서는 다음과 같습니다.** [pp.3–4, Figure 2]

| 단계 | 연산 | 출력 크기 | 역할 |
|---|---|---|---|
| 1 | RevIN 정규화 | $N\times T$ | 입력의 통계적 수준 차이 완화 |
| 2 | 차원 확장 | $N\times d\times T$ | 학습 가능한 표현 공간 확장 |
| 3 | 시간축 DFT, 독립 주파수 유지 | $N\times d\times F$ | 복소 주파수 표현 구성 |
| 4 | 실수부·허수부 각각 펼치기와 선형 투영 | 각각 $N\times D$ | 변수마다 하나의 토큰 생성 |
| 5 | 별도의 $L$개 enhanced Transformer 블록 | 각각 $N\times D$ | 변수 간 주파수 관계 학습 |
| 6 | 선형 투영·재배열·복소수 재결합 | $N\times d\times F$ | 스펙트럼 복원 |
| 7 | 역변환 `irfft` | $N\times d\times T$ | 시간 영역 표현 복원 |
| 8 | 잔차 연결·예측 헤드·역정규화 | $N\times\tau$ | 미래 시계열 출력 |

> **용어 풀이 — RevIN·은닉 차원·잔차 연결**
>
> **RevIN**은 입력에서 인스턴스별 통계 정보를 정규화하고 출력에서 복원하는 방식입니다. **은닉 차원**은 모델 내부 벡터의 크기입니다. **잔차 연결**은 변환 전 표현을 변환 후 표현에 더해 원래 정보를 전달하는 연결입니다. RevIN은 FreEformer가 새로 제안한 구성요소가 아닙니다. [seharanul17.github.io](https://seharanul17.github.io/RevIN/?utm_source=chatgpt.com)

**표기상 주의:** Figure 2에는 중간 표현이 $N\times d\times D$로 표시되지만, 본문 식 (6)은 펼친 뒤의 토큰을 $N\times D$로 정의합니다. 위 설명은 **본문 식 (6)을 기준**으로 했습니다.

### 5.2 차원 확장과 푸리에 변환

#### ① 차원 확장

본문의 $\bar x=x\times\phi_d$를 성분별로 풀어쓰면 다음과 같습니다.

$$
\bar X_{n,r,t}=X^{\text{norm}}_{n,t}\phi_r,
\qquad
\bar X\in\mathbb{R}^{N\times d\times T}.
$$

- $X^{\text{norm}}$: 정규화한 입력.
- $\phi_r$: 학습 가능한 벡터 $\phi_d\in\mathbb{R}^d$의 $r$번째 성분.
- $n$: 변수 인덱스.
- $r$: 확장 차원 인덱스.
- $t$: 시간 인덱스.

**[해석] 이 연산은 새로운 관측 정보를 추가하지 않습니다.** 동일 시계열의 학습 가능한 배수들을 만들어, 이후 연산이 사용할 표현 공간을 확장합니다. [p.3, §4.1]

#### ② DFT와 역DFT

하나의 길이 $T$ 시계열 $x[t]$에 대해 다음이 성립합니다.

```math
\mathcal F[k]
=
\sum_{t=0}^{T-1}
x[t]e^{-j2\pi kt/T},
```

```math
x[t]
=
\frac{1}{T}
\sum_{k=0}^{T-1}
\mathcal F[k]e^{j2\pi kt/T}.
```

- $x[t]$: 시점 $t$의 값.
- $\mathcal F[k]$: $k$번째 복소 주파수 계수.
- $j=\sqrt{-1}$: 허수 단위.
- $k$: 주파수 인덱스.
- $T$: 신호 길이.

이는 원문 식 (1)–(2)의 길이 기호를 $T$로 통일한 것입니다. [pp.2–3]

실수 입력에는 켤레 대칭이 있으므로 다음 관계를 이용할 수 있습니다.

```math
\mathcal F[T-k]=\mathcal F[k]^*,
\qquad
F=\left\lfloor\frac{T}{2}\right\rfloor+1
=
\left\lceil\frac{T+1}{2}\right\rceil.
```

- $(\cdot)^*$: 복소켤레.
- $F$: 중복되지 않는 주파수 계수의 수.
- $\lfloor\cdot\rfloor$, $\lceil\cdot\rceil$: 각각 내림과 올림.

**따라서 독립적인 절반의 주파수 계수를 유지하는 것은, 저주파 성분만 선택하는 저역통과 필터와 다릅니다.** [p.4, 식 (6)]

> **용어 풀이 — 복소수·켤레 대칭**
>
> 복소수는 실수부와 허수부로 구성됩니다. 주파수 계수의 두 성분은 진동의 크기와 위치 정보를 함께 나타냅니다. **켤레 대칭**은 실수 신호의 양·음 주파수 계수가 서로 대응하여 일부 계수가 중복되는 성질입니다.

### 5.3 실수부·허수부의 독립 처리

실수부의 초기 토큰은 다음처럼 만듭니다.

```math
H_R^{(0)}
=
\text{Linear}_R
\left(
\text{Flatten}\left(\text{Re}(\mathcal F_+)\right)
\right)
\in\mathbb{R}^{N\times D}.
```

허수부도 별도의 가중치로 처리합니다.

```math
H_I^{(0)}
=
\text{Linear}_I
\left(
\text{Flatten}\left(\text{Im}(\mathcal F_+)\right)
\right)
\in\mathbb{R}^{N\times D}.
```

- $\mathcal F_+\in\mathbb{C}^{N\times d\times F}$: 독립 주파수 부분.
- $\text{Re}$, $\text{Im}$: 실수부와 허수부 추출.
- $\text{Flatten}$: 변수별 $d\times F$ 성분을 하나의 벡터로 펼치는 연산.
- $H_R^{(0)}$, $H_I^{(0)}$: 두 분기의 초기 변수 토큰.
- $\text{Linear}_R$, $\text{Linear}_I$: 서로 다른 학습 가능한 선형층.

각 분기는 다음과 같이 $L$개 블록을 통과합니다.

```math
H_R^{(\ell+1)}
=
\text{TrmBlock}_R^{(\ell)}(H_R^{(\ell)}),
\qquad
H_I^{(\ell+1)}
=
\text{TrmBlock}_I^{(\ell)}(H_I^{(\ell)}).
```

- $\ell=0,\ldots,L-1$: 블록 인덱스.
- $\text{TrmBlock}$: enhanced attention, 잔차 연결, LayerNorm, FFN으로 구성한 블록.

**핵심은 attention의 대상이 변수라는 점입니다.** 기본 구조에서 주파수 점 하나하나를 attention 토큰으로 취급하지 않습니다. [p.4, 식 (6); p.3, Figure 2]

> **용어 풀이 — LayerNorm·FFN**
>
> **LayerNorm**은 내부 표현의 통계적 크기를 정리하는 정규화입니다. **FFN**은 각 토큰에 적용되는 순방향 신경망으로, 토큰 내부 특징을 변환합니다.

이후 실수부·허수부를 원래 스펙트럼 크기로 되돌리고 `irfft`로 $\tilde X$를 얻습니다. 최종 출력은 다음과 같습니다.

```math
\hat Y
=
\text{DeNorm}
\left(
\text{Linear}_{\text{head}}
\left(
\text{Flatten}(\tilde X+\bar X)
\right)
\right).
```

- $\tilde X\in\mathbb{R}^{N\times d\times T}$: 학습된 스펙트럼을 역변환한 표현.
- $\bar X$: 차원 확장한 입력.
- $\text{Linear}_{\text{head}}$: 변수별 $dT$차원 표현을 미래 $\tau$개 값으로 변환하는 예측층.
- $\text{DeNorm}$: 입력 정규화의 통계 정보를 이용한 역정규화.

이는 원문 식 (7)에 해당합니다. [p.4]

### 5.4 핵심 기술: enhanced attention

#### ① 일반 attention

```math
A
=
\text{Softmax}
\left(
\frac{QK^\top}{\sqrt D}
\right),
\qquad
\text{Attn}(Q,K,V)=AV.
```

- $Q,K,V\in\mathbb{R}^{N\times D}$: 토큰을 선형 변환하여 만든 query, key, value 행렬.
- $A\in\mathbb{R}^{N\times N}$: 변수 간 attention 가중치.
- $D$: 특징 차원.
- Softmax는 행별로 적용됩니다.

이는 원문 식 (8)입니다. [p.4]

> **용어 풀이 — Query·Key·Value**
>
> **Query**는 어떤 정보를 찾는지, **Key**는 각 입력이 어떤 정보를 담는지, **Value**는 실제로 가져와 결합할 내용을 나타내는 벡터입니다.

#### ② 학습 행렬 추가

FreEformer는 다음과 같이 수정합니다.

$$
S=\text{Softplus}(B),
\qquad
\text{Softplus}(z)=\log(1+e^z),
$$

$$
C=\text{Norm}(A+S),
\qquad
\text{EnhAttn}(Q,K,V)=CV.
$$

- $B\in\mathbb{R}^{N\times N}$: 추가 학습 파라미터.
- $S$: Softplus를 적용하여 원소를 양수로 만든 행렬.
- $\text{Norm}$: 행별 $L_1$ 정규화.
- $C$: 최종 attention 행렬.

모든 항이 양수이므로, 각 행을 그 행의 원소 합으로 나누면 됩니다. 이는 원문 식 (9)입니다. [p.5]

**수정 위치가 중요합니다.** 학습 행렬을 **softmax를 거친 attention 확률에 더합니다.**

#### ③ 두 종류의 관계를 결합한다는 해석

**[해석: 식 (9)의 대수적 재표현]** 다음과 같이 정의하면,

$$
s_i=\sum_{q=1}^{N}S_{iq},
\qquad
P_{iq}=\frac{S_{iq}}{s_i},
\qquad
\lambda_i=\frac{1}{1+s_i},
$$

최종 attention은 다음과 같습니다.

```math
C_{iq}
=
\lambda_i A_{iq}
+
(1-\lambda_i)P_{iq}.
```

- $i,q$: 변수 인덱스.
- $s_i$: 학습 행렬 $S$의 $i$번째 행의 합.
- $P$: 행별로 정규화한 학습 관계.
- $\lambda_i$: 입력 기반 attention과 학습 관계의 혼합 비중.

따라서 이 구조는 다음 두 요소를 결합합니다.

- $A$: **현재 입력에 따라 달라지는 변수 관계**.
- $P$: **학습 데이터 전체에서 얻은, 입력 샘플과 무관한 변수 관계**.

$p.5$에서 저자가 제시한 “MLP와 attention의 결합”이라는 설명을 수식으로 명확히 풀어쓴 것입니다. **$P$와 $\lambda_i$는 학습 후 입력 샘플마다 새로 결정되는 값이 아닙니다.**

> **용어 풀이 — 정적 관계·귀납적 편향**
>
> 여기서 **정적 관계**는 현재 입력과 무관하게 학습된 관계를 뜻합니다. **귀납적 편향**은 모델이 어떤 구조를 더 쉽게 학습하도록 만드는 설계상의 선호입니다. 이런 관계를 공유하는 것이 도움이 되는지는 데이터에 따라 검증해야 합니다.

### 5.5 이론 분석: 무엇이 성립하고, 무엇은 보장되지 않는가

#### A. 주파수 선형변환과 시간 영역 합성곱

**[저자 보고] Theorem 1**은 다음 주파수 선형변환을 다룹니다.

$$
\tilde{\mathcal F}=W\mathcal F+b.
$$

그 시간 영역 표현은 다음과 같습니다.

```math
\text{IDFT}(\tilde{\mathcal F})
=
\sum_{i=0}^{T-1}
\Omega_i\circledast\mathcal M_i(x)
+
\text{IDFT}(b),
```

$$
w_i=
[\text{diag}(W,i),\text{diag}(W,i-T)],
\qquad
\Omega_i=\text{IDFT}(w_i),
$$

```math
\mathcal M_i(x)[t]
=
x[t]e^{-j2\pi it/T}.
```

- $W\in\mathbb{C}^{T\times T}$: 주파수 영역 선형변환 행렬.
- $b\in\mathbb{C}^{T}$: 편향 벡터.
- $\text{diag}(W,i)$: $W$의 $i$번째 대각선 성분을 추출한 벡터.
- $[\cdot,\cdot]$: 벡터 이어 붙이기.
- $\Omega_i$: 역DFT로 얻은 합성곱 필터.
- $\mathcal M_i(x)$: 원 신호에 복소 진동을 곱한 변조 신호.
- $\circledast$: 순환 합성곱.

**근거:** pp.3, 11, 식 (3)–(4), Appendix A.

> **용어 풀이 — 순환 합성곱·변조**
>
> **순환 합성곱**은 신호의 끝과 처음을 연결한 것으로 보고 수행하는 필터 연산입니다. **변조**는 신호에 진동 성분을 곱하여 주파수 위치를 이동시키는 연산입니다.

**[해석] 이 정리의 의미는 “일반적인 주파수 선형변환이 여러 변조 신호에 대한 순환 합성곱의 합과 대응한다”는 것입니다.**  
비선형 FFN·LayerNorm·attention을 포함한 전체 FreEformer가 하나의 합성곱과 같다는 뜻이나, 일반화 오차가 감소한다는 보장은 아닙니다.

#### B. 랭크 분석

원문 식 (10)은 다음 부등식입니다.

$$
\left|
\text{rank}(A)-\text{rank}(S)
\right|
\le
\text{rank}(A+S)
\le
\text{rank}(A)+\text{rank}(S).
$$

행합이 양수이므로 행 정규화는 랭크를 바꾸지 않습니다.

$$
\text{rank}(C)=\text{rank}(A+S).
$$

- $\text{rank}$: 행렬의 랭크.
- $A$: 입력 기반 attention.
- $S$: 양수로 변환한 학습 행렬.
- $C$: 정규화 후 attention.

**[해석] 이 부등식은 항상 랭크가 증가한다는 정리가 아닙니다.** 또한 실제 출력에 대해서는 다음 제한이 있습니다.

$$
\text{rank}(CV)
\le
\min\{\text{rank}(C),\text{rank}(V)\}.
$$

따라서 attention 행렬의 랭크가 높아져도, value 행렬 $V$가 담은 정보나 최종 예측 정확도가 자동으로 개선되지는 않습니다. [p.5, 식 (10)에 대한 수학적 해석]

**증명 표기에도 주의가 필요합니다.** 부록 식 (25)는 열공간의 포함관계를 합집합 $\cup$으로 표현하지만, 일반적으로 필요한 것은 열공간의 **합**입니다. 랭크 부등식 자체는 타당하나, 첨부본의 해당 증명 전개에는 오류가 있습니다. [pp.11–12, 식 (25)–(31)]

> **용어 풀이 — 열공간**
>
> 행렬의 열벡터들을 여러 비율로 더해서 만들 수 있는 모든 벡터의 집합입니다.

#### C. Gradient 분석

한 행에 대해 다음과 같이 정의합니다.

$$
p=\text{Softmax}(a),
\qquad
u=p+b,
\qquad
s=\|u\|_1=1+\|b\|_1,
\qquad
c=\frac{u}{s}.
$$

- $a\in\mathbb{R}^{N}$: softmax 이전 attention 점수.
- $p$: 일반 attention의 한 행.
- $b$: $S=\text{Softplus}(B)$의 해당 행.
- $u$: 두 행을 더한 벡터.
- $s$: 그 행의 합.
- $c$: 최종 enhanced attention의 한 행.

원문의 Jacobian은 다음과 같습니다.

```math
\frac{\partial p}{\partial a}
=
\text{Diag}(p)-pp^\top,
```

```math
\frac{\partial c}{\partial a}
=
\frac{1}{s}
\left(
\text{Diag}(p)-pp^\top
\right),
```

```math
\frac{\partial c}{\partial b}
=
\frac{1}{s^2}
\left(
sI-u\mathbf 1^\top
\right).
```

- $\text{Diag}(p)$: $p$를 대각 원소로 갖는 행렬.
- $I$: 단위행렬.
- $\mathbf 1$: 모든 성분이 1인 열벡터.

**근거:** p.5, 식 (11)–(13); pp.12–13, Appendix C.

**[해석] 정확한 결론은 “기존 경로의 미분 스케일을 조절하고, 추가 학습 경로를 제공한다”는 것입니다.**  
$s>1$이므로 기존 점수 $a$에 대한 Jacobian은 $1/s$배로 **축소**됩니다. 따라서 “모든 gradient를 키워 소실을 해결한다”는 설명은 부정확합니다.

또한 실제 원파라미터를 $\beta$라 하고 $b=\text{Softplus}(\beta)$라 하면,

```math
\frac{\partial c}{\partial\beta}
=
\frac{\partial c}{\partial b}
\text{Diag}(\sigma(\beta))
```

가 됩니다.

- $\beta$: $B$의 해당 행.
- $\sigma$: sigmoid 함수.

이는 Figure 5의 $\partial c/\partial b$만 보고 실제 학습 파라미터의 gradient 크기를 단정할 수 없다는 뜻입니다.

### 5.6 학습 손실

**[저자 보고] 학습에는 CARD에서 사용한 가중 $L_1$ 손실을 적용합니다.**

```math
\mathcal L
=
\frac{1}{\tau}
\sum_{t=1}^{\tau}
t^{-\alpha}
\left\|
\hat Y_{:t}-Y_{:t}
\right\|_1,
\qquad
\alpha=0.5.
```

- $\tau$: 예측 길이.
- $t$: 미래 예측 시점.
- $\hat Y_{:t},Y_{:t}\in\mathbb{R}^{N}$: 시점 $t$의 모든 변수에 대한 예측과 정답.
- $\|\cdot\|_1$: 절댓값의 합.
- $t^{-\alpha}$: 먼 미래의 오차에 더 작은 가중치를 부여하는 항.

**근거:** p.20, Appendix I.4.

이 손실은 평가 지표인 MSE와 다릅니다. 논문은 다른 모델에도 같은 손실을 적용한 추가 비교를 제공합니다. [p.34, Table 28]

---

## 6. 주요 결과와 데이터 포인트

### 6.1 예측 성능: 강점과 예외

아래 값은 **각 표에서 지정한 네 예측 길이의 평균**입니다.

| 데이터·설정 | FreEformer MSE / MAE | 비교 모델 MSE / MAE | 해석 | 근거 |
|---|---:|---:|---|---|
| ECL | **0.162 / 0.251** | CARD 0.168 / 0.258 | 두 지표 모두 우수 | Table 2, p.6 |
| PEMS03 | **0.102 / 0.206** | Leddam 0.107 / 0.210 | 두 지표 모두 우수 | 동일 |
| PEMS04 | **0.094 / 0.196** | Leddam 0.103 / 0.210 | 두 지표 모두 우수 | 동일 |
| PEMS07 | **0.080 / 0.167** | Leddam 0.084 / 0.180 | 두 지표 모두 우수 | 동일 |
| PEMS08 | **0.110 / 0.194** | Leddam 0.122 / 0.211 | 두 지표 모두 우수 | 동일 |
| Traffic | 0.435 / **0.251** | iTransformer **0.428** / 0.282 | MAE 우세, MSE 열세 | 동일 |
| Exchange | 0.354 / 0.399 | Fredformer **0.333 / 0.391** | 두 지표 모두 비교 모델이 우수 | 동일 |
| ILI, S1 | **1.140 / 0.585** | iTransformer 1.437 / 0.659 | 단기 예측에서 큰 개선 | Table 3, p.6 |

표의 반올림값으로 직접 계산하면, ECL의 MSE는 CARD 대비 **약 3.6%**, PEMS04는 Leddam 대비 **약 8.7%**, ILI S1은 iTransformer 대비 **약 20.7%** 낮습니다.  
이 비율은 **본 보고서의 재계산**이며, 통계적 유의성을 뜻하지 않습니다.

다음 예외도 중요합니다.

- **ETTh1:** FreEformer MSE 0.433, Leddam 0.431.
- **ETTh2:** FreEformer MSE 0.372, TimeMixer 0.364.
- **Solar:** FreEformer MSE 0.217, TimeMixer 0.216.
- **추가 주파수 모델 비교의 Traffic:** FreEformer MSE 0.435, FreDF 0.421.  
  [p.6, Tables 2, 4]

**[해석] 따라서 “전반적으로 경쟁력이 높고 여러 MAE 결과에서 강하다”는 평가는 타당하지만, “모든 데이터셋·지표에서 최고”라는 해석은 표와 일치하지 않습니다.**

### 6.2 구성요소별 결과

> **용어 풀이 — 구성요소 제거·교체 실험(ablation)**
>
> 모델의 일부만 없애거나 바꾸어, 성능 향상에 어떤 요소가 기여했는지 확인하는 실험입니다.

| 비교 | 주요 결과 | 해석·주의점 | 근거 |
|---|---|---|---|
| 실수부·허수부 처리 | Weather MSE: 비공유 0.239, 공유 0.240, 결합 0.243 | 이 구조에서는 독립 가중치가 유리하지만 차이는 작음 | Table 1, p.4 |
| 주파수 패치 사용 여부 | COVID-19 MSE: 최종 구성 1.892, 패치 사용 2.029 | 시험한 패치 설정에서는 전체 스펙트럼 처리가 유리 | Table 5, p.7 |
| 주파수 영역 대 시간 영역 | Enhanced attention에서 저자 보고 평균 MSE 개선 8.4% | 시간 영역 변형은 허수부 분기도 제거하므로 용량 차이가 섞임 | Table 6, p.7 |
| 예측 헤드 | Weather: 시간 0.239, 주파수 0.245; ECL: 시간 0.162, 주파수 0.160 | 시간 헤드가 항상 더 좋은 것은 아님 | Table 7, p.7 |
| 손실함수 통제 | 동일 가중 $L_1$ 손실에서도 여러 평균 결과가 우수 | 예컨대 Traffic 0.435 대 Fredformer 0.436은 작은 차이 | Table 28, p.34 |

**[저자 해석]** 패치가 주파수의 전역 정보를 제한할 수 있다는 설명을 제시합니다.  
**[분석상 한계]** 패치가 실제로 “잡음을 추가했다”는 현상 자체를 별도로 측정한 것은 아닙니다. [pp.5–6, §5.2]

### 6.3 다른 모델로 attention을 이식한 결과

저자는 다음 평균 MSE 개선율을 보고합니다. [p.7, Table 9]

| 대상 모델 | 저자 보고 개선율 |
|---|---:|
| iTransformer | 5.9% |
| PatchTST | 9.9% |
| Leddam | 1.4% |
| Table 9의 Fredformer 열 | 3.8%로 설명되는 결과 |

**마지막 항목은 원문 표기 문제가 있습니다.** 표에는 Fredformer가 있지만, 본문은 3.8%의 모델명을 FreEformer로 적었습니다.

또한 다음 예외가 있습니다.

- Fredformer·ETTm1: **0.384 → 0.385**, 소폭 악화.
- Leddam·Weather: **0.242 → 0.242**, 표시 정밀도상 동률.
- PatchTST·METR-LA: **0.335 → 0.335**, 표시 정밀도상 동률.

**[해석] 이 실험은 제안 모듈의 적용 범용성을 뒷받침하는 중요한 증거입니다.** 다만 “모든 모델의 모든 조건에서 항상 향상”으로 표현해서는 안 됩니다.

### 6.4 계산 효율

ECL, 입력 96·예측 96·배치 크기 16 조건에서 다음 수치가 보고됩니다. [p.28, Table 22]

| 모델 | 파라미터 수 | 학습 시간 | 학습 GPU 메모리 |
|---|---:|---:|---:|
| FreEformer | 11.05M | 24.89 ms/iteration | 1.46 GB |
| iTransformer | 4.83M | 10.66 ms/iteration | 0.70 GB |
| Leddam | 8.56M | 65.19 ms/iteration | 1.44 GB |

**[해석] 일부 복잡한 모델과 비교하면 정확도와 속도의 절충이 좋지만, 가장 작거나 빠른 모델은 아닙니다.**  
배포 판단에서는 예측 오차뿐 아니라 입력 변수 수와 메모리 제한을 함께 고려해야 합니다.

---

## 7. 가장 중요한 그림 선정과 해석

| 그림 | 읽어야 할 내용 | 해석과 한계 |
|---|---|---|
| **Figure 2, p.3** | 정규화 → DFT → 실수·허수 분기 → 변수 간 enhanced attention → 역변환 → 예측 | 모델 이해의 출발점입니다. 핵심 attention 대상은 변수이며, 정확한 중간 차원은 본문 식 (6)을 함께 확인해야 합니다. |
| **Figures 3–4, p.4** | 일반 attention과 enhanced attention의 행렬 구조·랭크 | 랭크 문제의 관찰과 보강 효과를 보여줍니다. 하지만 대표 시각화이며 전체 데이터·head·layer의 통계 분포는 아닙니다. |
| **Figure 13, p.19** | ECL·Traffic에서 학습 비율 감소에 따른 MAE | 일반화 중에서도 **학습 데이터 효율**에 가장 직접적인 그림입니다. 5%까지 우세하지만 두 데이터셋의 Input-96/Predict-96 결과입니다. |
| **Figure 1, p.1** | 인접 구간의 정규화된 주파수 진폭과 변수 간 유사성 | 주파수 표현을 사용하는 동기를 보여줍니다. 진폭 예시만으로 전체 복소 스펙트럼이나 모든 시계열의 안정성을 입증하지는 않습니다. |
| **Figure 5, p.5** | 두 경로의 Jacobian 시각화 | 추가 학습 경로가 attention을 조절하는 방식을 보여줍니다. 학습 수렴 속도나 실제 원파라미터 gradient의 우위를 직접 입증하지는 않습니다. |
| **Figure 14, p.20** | MAE, 학습 시간, 학습 메모리의 관계 | 실용적인 비용 비교에 유용합니다. 원 크기는 메모리를 나타내며, 서로 다른 하위 그림 사이의 원 크기를 직접 비교하면 안 됩니다. |

### Figure 4를 정확하게 읽는 방법

이 그림의 랭크는 다음과 같습니다.

| 데이터 | 별도로 학습한 vanilla 모델 | Enhanced 모델 내부 $A$ | 학습 행렬 $S$ | 최종 $C$ |
|---|---:|---:|---:|---:|
| Weather | 3 | 18 | 21 | 21 |
| ECL | 137 | 40 | 321 | 321 |

따라서 **동일한 enhanced 모델 안에서의 변화는 Weather 18→21, ECL 40→321**입니다. 첫 열의 vanilla 모델과 이후 세 열을 구분해야 합니다.

또한 Figures 13–14는 **세로축이 뒤집혀 있으므로 위쪽일수록 더 작은 MAE**입니다. [각 Figure의 캡션]

---

## 8. 일반화 성능·통계적 취약성·미해결 질문

### 8.1 일반화는 어디까지 검증되었는가?

| 일반화의 의미 | 원문 근거 | 판단 |
|---|---|---|
| 같은 데이터셋의 미래 구간 예측 | 시간순 분할, Tables 2–3 | 여러 도메인에서 근거가 있음 |
| 여러 데이터셋에서 각각 학습했을 때의 성능 | 18개 벤치마크 | 적용 범위의 넓이를 뒷받침 |
| 적은 학습 데이터로 학습 | Figure 13 | ECL·Traffic의 특정 설정에서 근거가 있음 |
| 초기화에 따른 안정성 | Table 12 | 6개 데이터셋에서 FreEformer의 7회 반복 결과 제공 |
| 다른 모델에 모듈 적용 | Table 9 | attention 모듈의 범용성에 대한 근거 |
| 새로운 데이터셋으로 학습 없이 전이 | FreEformer의 해당 실험 없음 | 미검증 |
| 새로운 변수 수·센서 구성·변수 순서 | 명시적 평가 없음 | 미검증 |
| 급격한 분포 변화·결측·노이즈 | 명시적 강건성 평가 없음 | 미검증 |

**[해석] 이 논문에서 가장 잘 뒷받침되는 일반화는 “각 데이터셋에서 학습한 모델이 해당 데이터셋의 미래 구간을 예측하는 능력”입니다.**

#### 학습 행렬 $B$와 일반화의 관계

$B$는 $N\times N$이므로 행렬 하나당 $N^2$개의 파라미터를 갖습니다. 변수 수나 변수의 의미가 바뀌면 동일한 행렬을 그대로 사용하는 방법이 자동으로 정의되지 않습니다. [식 (9)에 대한 해석]

또한 변수 순서를 바꾸려면 학습된 관계 행렬도 대응하도록 바꾸어야 합니다. 위치 정보가 없는 일반 self-attention의 성질을 고정된 $B$가 있는 구조에 그대로 적용할 수는 없습니다.

> **용어 풀이 — 순열 등변성**
>
> 입력 변수의 순서를 바꾸면 출력도 같은 순서로 바뀌는 성질입니다. 출력이 아예 변하지 않는 “순열 불변성”과 다릅니다.

이는 **저자가 명시한 한계가 아니라, 제안 수식에서 도출한 분석**입니다.

### 8.2 적은 데이터와 사전학습 모델 비교의 경계

Table 21은 FreEformer와 Timer 등을 비교하지만, 다음 조건이 다릅니다. [p.27]

- FreEformer: 사전학습 없음, 입력 길이 **Best**.
- Timer: 사전학습 사용, 입력 길이 **672**.
- Zero-shot 모델들: 서로 다른 사전학습 자료와 입력 길이 사용.
- FreEformer 결과는 100%·20%·5% 학습 열에 있으며, **0% zero-shot 열에는 없습니다.**

| 데이터·학습 비율 | FreEformer MSE | Timer MSE | 표에서 관찰되는 결과 |
|---|---:|---:|---|
| ECL, 100% | **0.124** | 0.136 | FreEformer 우세 |
| ETTm1, 20% | **0.307** | 0.321 | FreEformer 우세 |
| ECL, 5% | 0.145 | **0.132** | Timer 우세 |
| Traffic, 5% | 0.425 | **0.361** | Timer 우세 |
| PEMS03, 5% | 0.232 | **0.125** | Timer 우세 |
| PEMS04, 5% | 0.815 | **0.135** | Timer 우세 |

**[조건이 달라 직접적인 일반화 순위로 해석할 수 없음]**  
이 표는 FreEformer가 일부 조건에서 경쟁력이 있음을 보여주지만, “적은 데이터에서 항상 사전학습 모델보다 우수하다”거나 “zero-shot 일반화가 우수하다”는 결론을 지지하지 않습니다.

### 8.3 통계적으로 취약하거나 비교에 주의할 부분

| 항목 | 원문에서 확인되는 사실 | 분석상 한계 |
|---|---|---|
| 반복 실험 | 6개 데이터셋에서 FreEformer만 7개 seed의 표준편차 제시 | 모든 비교 모델의 반복 결과·paired 검정·신뢰구간이 없음 |
| 작은 성능 차이 | COVID-19 S2 MSE 8.435 대 Leddam 8.439 | 상대 차이는 약 **0.047%**로 재계산되며, 유의한 우위인지 판단 불가 |
| 비교 수치의 출처 | 기존 논문 수치와 공식 코드 실행값을 함께 사용 | 동일한 튜닝·학습 예산으로 통제한 전체 비교는 아님 |
| 주파수/시간 영역 실험 | 시간 변형에서 DFT·IDFT와 허수부 분기를 제거 | 주파수 표현 효과와 모델 용량 효과가 분리되지 않음 |
| 랭크 시각화 | 일부 attention 행렬의 랭크 제시 | 수치 허용오차·선정 기준·전체 분포가 충분히 보고되지 않음 |
| 데이터셋 간 MSE | 서로 다른 측정 대상과 예측 구간 | 다른 데이터셋의 원시 MSE 크기를 직접 비교하면 안 됨 |
| 사전학습 모델 비교 | 입력 길이·사전학습·학습 비율 차이 | 표의 수치만으로 모델 계열 전체의 일반화 우열 판단 불가 |
| MASE | 식 (55)의 분모가 예측 구간 정답의 차분으로 기재되고 인덱스가 혼재 | 다른 연구의 MASE와 정의·구현 확인 없이 비교하면 안 됨 |

**근거:** p.15 Appendix E; p.18 Table 12·식 (55); p.27 Table 21; pp.6–7 Table 6.

예를 들어 Table 12의 ECL 예측 길이 96 MSE는 $0.133\pm0.001$, Traffic은 $0.395\pm0.002$입니다. 이는 **FreEformer의 초기화 변화에 따른 안정성**을 보여주지만, 비교 모델보다 통계적으로 유의하게 우수하다는 검정 결과는 아닙니다.

> **용어 풀이 — Seed·표준편차·신뢰구간**
>
> **Seed**는 무작위 초기화 등을 재현하기 위한 설정값입니다. **표준편차**는 반복 결과가 흩어진 정도이고, **신뢰구간**은 추정한 차이의 불확실성을 표현하는 구간입니다.

추가로 재현 시 확인할 문서상 불일치도 있습니다.

- PEMS 상세 결과의 Table 14 캡션은 예측 길이를 잘못 적고 있어, 실제 행과 Table 2의 설명을 함께 확인해야 합니다. [p.22]
- Table 30 캡션의 $d=16$ 차원 확장 설명은 본문 §4.1의 학습 벡터 설명과 일치하지 않습니다. [pp.3, 34]

### 8.4 저자가 직접 언급한 한계와 후속 방향

첨부본에는 별도의 **Limitations 절이 없습니다.** 따라서 위의 분석상 한계를 저자가 직접 인정한 내용으로 표현해서는 안 됩니다.

원문에서 직접 확인할 수 있는 관련 내용은 다음과 같습니다.

1. **미래 주파수 스펙트럼을 직접 예측하는 데 어려움이 있다**고 설명합니다. [p.7, Prediction Head]
2. Fourier가 모든 경우에 최선은 아닙니다. PEMS03의 평균 MSE는 Fourier 0.102, Haar 0.100, discrete Meyer 0.098입니다. [p.29, Table 23]
3. **더 강건한 기저 함수를 탐색하는 것**을 명시적인 후속 연구 방향으로 제시합니다. [p.19, Appendix I.1]

> **용어 풀이 — 기저 함수·Wavelet**
>
> **기저 함수**는 신호를 분해하고 표현할 때 사용하는 기본 패턴입니다. Fourier는 진동 성분을, **wavelet**은 시간상 위치가 제한된 파형을 이용하여 신호를 표현합니다.

### 8.5 문서가 답하지 않는 질문

- **랭크 증가 자체가 성능 향상의 원인인가?** 추가 파라미터나 정적 변수 혼합의 효과와 구분되지 않습니다.
- **변수 간 관계가 바뀌면 학습된 $B$는 어떻게 작동하는가?** 관계가 안정적인 구간과 급변하는 구간을 나눈 검증이 없습니다.
- **새로운 변수 수나 센서 집합으로 어떻게 전이하는가?** 현재 $N\times N$ 파라미터의 이식 방법이 제시되지 않습니다.
- **주파수·시간 변형의 파라미터 수와 계산량을 맞추어도 이득이 유지되는가?**
- **저데이터 성능이 ECL·Traffic과 일부 PEMS 조건에서 다르게 나타나는 이유는 무엇인가?**
- **결측·관측 노이즈·주기 변화·위상 변화에 얼마나 강한가?**
- **모든 비교 모델을 같은 데이터 처리·손실·튜닝 예산으로 다시 학습해도 작은 성능 차이가 유지되는가?**
- **예측 오차 감소가 실제 운영 비용 절감이나 의사결정 개선으로 이어지는가?** 논문은 이 후속 효과를 측정하지 않습니다.

> **용어 풀이 — 위상**
>
> 같은 주기의 진동이라도 봉우리와 골이 시간상 어디에 위치하는지를 나타내는 정보입니다.

---

## 9. 2020년 이후 관련 연구 비교 분석

아래는 관련 흐름을 **2021–2026년의 대표 연구**로 정리한 것입니다. 연도는 가능한 경우 정식 학회 발표 기준을 사용했습니다.  
**서로 다른 논문에서 보고한 MSE를 합쳐 순위를 매기지는 않았습니다.**

### 9.1 구조와 연구 목적의 비교

| 연구 | 원 저자들이 제안한 핵심 | FreEformer와의 비교 분석 |
|---|---|---|
| **Informer·Autoformer, 2021** | Informer는 효율적인 장기 attention, Autoformer는 분해와 주기 기반 자기상관 집계에 초점 | FreEformer는 전체 스펙트럼을 담은 **변수 토큰 사이의 관계**와 attention 표현력에 초점. [ojs.aaai.org](https://ojs.aaai.org/index.php/AAAI/article/view/17325?utm_source=chatgpt.com) |
| **FEDformer, 2022** | 계절·추세 분해와 주파수 표현을 결합 | 주파수 사용 자체는 기존 흐름에 속하며, FreEformer의 차이는 변수 토큰화와 attention 보강에 있음. [proceedings.mlr.press](https://proceedings.mlr.press/v162/zhou22g.html?utm_source=chatgpt.com) |
| **RevIN, 2022** | 입력별 통계 정보를 제거하고 출력에서 복원 | FreEformer에 이미 포함된 요소이므로 정규화의 효과와 신규 모듈의 효과를 구분해야 함. [seharanul17.github.io](https://seharanul17.github.io/RevIN/?utm_source=chatgpt.com) |
| **DLinear, 2023** | 단순 선형 모델로 복잡한 Transformer의 필요성을 재검토 | 추가 복잡성의 가치를 평가할 때 동일 입력 길이·정규화·학습 예산의 선형 기준선이 중요. [ojs.aaai.org](https://ojs.aaai.org/index.php/AAAI/article/view/26317?utm_source=chatgpt.com) |
| **PatchTST, 2023** | 시간 패치와 변수 독립 처리, 공유 가중치 사용 | FreEformer의 변수 결합과 대비되는 접근이며, 변수 관계가 바뀔 때 비교할 중요한 기준선. [arxiv.org](https://arxiv.org/abs/2211.14730?utm_source=chatgpt.com) |
| **FreTS, 2023** | 주파수 영역 MLP로 변수·시간 의존성 학습 | 주파수 표현의 철학은 가깝고, 핵심 비교는 MLP 기반 학습과 변수 간 attention의 차이. [proceedings.nips.cc](https://proceedings.nips.cc/paper_files/paper/2023/hash/f1d16af76939f476b5f040fd1398c0a3-Abstract-Conference.html?utm_source=chatgpt.com) |
| **iTransformer, 2024** | 각 변수의 전체 시계열을 토큰화하고 변수 간 attention 적용 | 가장 직접적인 구조적 선행 연구로, FreEformer는 주파수 표현과 enhanced attention을 추가. [proceedings.iclr.cc](https://proceedings.iclr.cc/paper_files/paper/2024/hash/2ea18fdc667e0ef2ad82b2b4d65147ad-Abstract-Conference.html?utm_source=chatgpt.com) |
| **FITS, 2024** | 복소 주파수 영역의 보간을 이용하는 작은 모델 | 주파수 표현의 이점이 큰 attention 없이도 얻어지는지 확인할 경량 기준선. [proceedings.iclr.cc](https://proceedings.iclr.cc/paper_files/paper/2024/hash/701251e1db4a2e4dd2ef23f5265d5936-Abstract-Conference.html?utm_source=chatgpt.com) |
| **Fredformer, 2024** | 에너지가 큰 주파수에 대한 편중을 줄이고 주파수 대역을 고르게 학습 | 주파수 에너지 편향과 attention의 낮은 랭크는 서로 다른 문제이므로 구분해야 함. [arxiv.org](https://arxiv.org/abs/2406.09009?utm_source=chatgpt.com) |
| **FAN, 2024** | 주요 주파수 성분을 정규화하고 입력·출력 간 변화도 예측 | 스펙트럼의 안정성이 약해지는 조건을 연구할 때 관련성이 높음. [proceedings.neurips.cc](https://proceedings.neurips.cc/paper_files/paper/2024/hash/37c6d0bc4d2917dcbea693b18504bd87-Abstract-Conference.html?utm_source=chatgpt.com) |
| **FreDF, 2025** | 미래 정답의 상관관계를 고려하여 주파수 영역에서 학습 | 입력 표현을 바꾸는 FreEformer와 학습 목표를 바꾸는 접근을 구분할 필요가 있음; 2024년 공개본은 이미 FreEformer에 인용됨. [proceedings.iclr.cc](https://proceedings.iclr.cc/paper_files/paper/2025/hash/1457fb1e5d72cdc4ecd88bc10f916095-Abstract-Conference.html?utm_source=chatgpt.com) |
| **Chronos-2, 2025** | Group attention과 사전학습을 이용한 단변량·다변량·보조변수 기반 zero-shot 예측 | 데이터셋별 학습을 넘어 새로운 과제로의 전이를 평가해야 한다는 비교 기준을 제공. [arxiv.org](https://arxiv.org/abs/2510.15821?utm_source=chatgpt.com) |
| **DTAF, 2026** | 시간 영역 안정화와 주파수 차분을 결합하여 비정상성에 대응 | 스펙트럼이 안정적인 상황과 변화하는 상황을 나누어 FreEformer를 평가할 필요성을 제시. [ojs.aaai.org](https://ojs.aaai.org/index.php/AAAI/article/view/39585?utm_source=chatgpt.com) |

> **용어 풀이**
>
> - **변수 독립 처리:** 각 변수의 예측 과정에서 다른 변수와 직접 정보를 섞지 않는 방식.
> - **비정상성:** 평균·변동성·주기 구조 등이 시간에 따라 달라지는 현상.
> - **보조변수·covariate:** 예측 대상과 함께 사용하는 외부 정보나 관련 변수.
> - **Group attention:** 같은 그룹의 여러 시계열이나 변수 사이에서 정보를 공유하는 attention.

### 9.2 일반화 관점에서 중요한 비교 축

#### A. iTransformer와의 비교: 두 변경의 효과를 분리해야 한다

**[해석]** FreEformer를 이해하는 가장 유용한 비교는 다음 두 축입니다.

1. 시간 표현을 주파수 표현으로 바꾸는 효과.
2. 일반 attention을 enhanced attention으로 바꾸는 효과.

논문의 Table 6은 이 구분을 시도하지만, 시간 영역 변형에서 허수부 분기도 없어집니다. 후속 연구에서는 **두 분기의 구조와 파라미터 예산을 맞춘 비교**가 필요합니다. [pp.6–7, Table 6] iTransformer의 원래 기여는 변수 토큰화라는 점에서 이 통제 실험의 출발점이 됩니다. [proceedings.iclr.cc](https://proceedings.iclr.cc/paper_files/paper/2024/hash/2ea18fdc667e0ef2ad82b2b4d65147ad-Abstract-Conference.html?utm_source=chatgpt.com)

#### B. Fredformer와의 비교: 에너지 편향과 랭크는 같은 개념이 아니다

Fredformer는 작은 진폭이지만 예측에 중요한 주파수를 놓치는 문제를 다룹니다. FreEformer는 attention 행렬의 낮은 랭크를 주요 문제로 설정합니다. [arxiv.org](https://arxiv.org/abs/2406.09009?utm_source=chatgpt.com)

**[해석]** 랭크가 높아졌다는 사실만으로 작은 에너지의 중요한 주파수까지 더 잘 학습했다고 판단할 수는 없습니다. 두 현상을 함께 측정하면 제안 attention의 실제 작동 원리를 더 명확하게 확인할 수 있습니다.

#### C. FAN·DTAF와의 비교: 스펙트럼의 변화 자체를 평가해야 한다

FAN은 입력과 출력 사이의 주요 주파수 성분 변화를 모델링합니다. DTAF는 시간 영역과 주파수 영역의 비정상성을 명시적으로 다룹니다. [proceedings.neurips.cc](https://proceedings.neurips.cc/paper_files/paper/2024/hash/37c6d0bc4d2917dcbea693b18504bd87-Abstract-Conference.html?utm_source=chatgpt.com)

**[해석]** FreEformer의 Figure 1이 보여주는 스펙트럼 일관성이 약해질 때 성능이 유지되는지가 중요한 후속 질문입니다. 평균·분산 정규화만으로 주기나 변수 관계의 변화까지 해결했다고 볼 근거는 없습니다.

#### D. Chronos-2와의 비교: 일반화의 평가 과제가 다르다

Chronos-2는 관련 시계열·변수·보조변수 사이의 정보 공유와 사전학습을 통해 대상 데이터셋의 추가 학습 없이 예측하는 설정을 다룹니다. [arxiv.org](https://arxiv.org/abs/2510.15821?utm_source=chatgpt.com)

**[해석]** FreEformer의 적은 데이터 실험과 이러한 zero-shot 실험은 다른 질문에 답합니다. 후속 비교에서는 **데이터셋별 학습, 소량 미세조정, 학습 없는 전이**를 각각 분리해야 합니다. 최신 모델과의 우열은 동일한 평가 조건에서 다시 검증해야 합니다.

---

## 10. 결론·시사점·빠른 검증과 후속 연구

### 10.1 저자들이 제시한 결론과 계획

**[저자 보고]** 저자들은 주파수 영역의 enhanced Transformer가 다변량 예측에 유용하며, 제안 attention이 기존 Transformer 예측 모델에도 적용될 수 있다고 결론짓습니다. 또한 FreEformer가 시계열 예측 연구의 강한 기준 모델이 되기를 기대합니다. [p.7, §6]

명시적인 후속 연구 방향은 **더 강건한 기저 함수 탐색**입니다. Fourier 외 기저가 일부 데이터셋에서 더 좋은 결과를 보였다는 관찰에 근거합니다. [p.19, Appendix I.1; p.29, Table 23]

### 10.2 학술적·실용적 시사점

**학술적 시사점 — [해석]**

- 주파수 표현과 변수 토큰화를 결합하는 비교적 명확한 설계를 제시합니다.
- Attention의 정규화 이후를 수정하는 작은 모듈도 예측 성능에 영향을 줄 수 있음을 보여줍니다.
- 입력에 따라 달라지는 관계와 데이터셋에서 학습한 정적 관계를 함께 사용하는 구조를 분석할 출발점을 제공합니다.
- 랭크·Jacobian 같은 수학적 성질과 실제 일반화 성능 사이의 관계를 더 엄밀하게 검증할 연구 과제를 남깁니다. [식 (9)–(13), Tables 5–9]

**실용적 시사점 — [해석]**

- 기존 Transformer를 이미 사용하는 경우, 전체 모델을 다시 설계하기 전에 attention 모듈 교체를 시험할 가치가 있습니다.
- 변수 수가 고정된 데이터에서 학습한 관계를 활용하는 방식이므로, 센서 구성이나 변수 정의가 자주 바뀌는 환경에서는 별도의 적응 검증이 필요합니다.
- 계산 비용은 일부 복잡한 모델보다 낮지만, 경량 모델이나 iTransformer보다 높을 수 있으므로 실제 자원 제한을 함께 고려해야 합니다. [Tables 9, 22]

### 10.3 가장 유망한 기술과 빠른 검증 방법

다음은 **추가 성능을 예측하는 주장이 아니라, 핵심 아이디어를 확인하기 위한 실험 제안**입니다.

| 우선순위 | 검증할 기술 | 최소 실험 | 확인할 질문 |
|---|---|---|---|
| **1** | Softmax 이후 학습 행렬 추가 | 기존 iTransformer에서 일반 attention과 enhanced attention만 비교 | 동일 학습 설정에서 MSE·MAE 개선이 반복되는가? |
| **2** | 입력 관계와 정적 관계의 결합 | $A$만, $S$만, $A+S$를 비교 | 입력 적응성과 정적 변수 관계 중 무엇이 기여하는가? |
| **3** | 전체 스펙트럼의 변수 토큰화 | 작은 모델에서 시간 표현과 Fourier 표현을 비교하되 분기·파라미터 예산 통제 | 이득이 주파수 좌표계 자체에서 나오는가? |
| **4** | 기저 함수 선택 | 같은 처리 구조에서 Fourier·Haar·Meyer 교체 | 어떤 데이터 특성에서 Fourier가 유리하거나 불리한가? |
| **5** | 실수부·허수부 가중치 비공유 | 공유·비공유 구조를 파라미터 예산을 고려해 비교 | 분리 처리의 효과가 추가 용량과 구분되는가? |

**추천하는 시작점은 변수 수가 작은 Weather에서 1·2번을 먼저 검증하는 것입니다.** 이후 ECL로 확장하면 변수 수 증가에 따른 성능과 비용 변화를 함께 볼 수 있습니다. 이는 데이터 규모와 제안 구조를 고려한 실험 설계 제안입니다. [Table 11, 식 (9)]

실험에서는 입력 길이·예측 길이·손실함수·학습률 탐색 범위를 같게 하고, 같은 seed 집합으로 비교하는 것이 좋습니다. 랭크만 기록하지 말고 **예측 오차, 반복 변동, 학습·테스트 성능 차이, 계산 비용**을 함께 기록해야 합니다.

### 10.4 일반화 성능 향상을 위한 후속 연구 방향

**[제안] 첫째, 시간에 따라 관계가 변하는 조건을 분리해 평가해야 합니다.**  
안정적인 구간과 주기·변수 관계가 바뀌는 구간을 나누어, 정적 관계 $S$가 언제 유용하고 언제 제한이 되는지 확인할 수 있습니다.

**둘째, 가변적인 변수 집합으로의 전이를 연구해야 합니다.**  
변수 누락·추가·순서 변경을 구분하여 평가하고, $N\times N$ 행렬을 새로운 변수 집합에 적용하는 방법을 정의해야 합니다. 이는 현재의 데이터셋별 성능을 더 넓은 일반화 주장으로 확장하기 위한 핵심 조건입니다.

**셋째, 주파수 표현의 효과와 모델 용량의 효과를 분리해야 합니다.**  
동일 파라미터 수·연산량·학습 예산에서 시간 영역, Fourier, wavelet을 비교하면 표현 방식의 기여가 더 분명해집니다.

**넷째, 랭크와 성능 사이의 인과적 관계를 검증해야 합니다.**  
학습 행렬 외에 고정 행렬이나 단순한 변수 혼합도 통제군으로 두면, 랭크 증가 때문인지 추가 혼합 경로 때문인지 구분하는 데 도움이 됩니다.

**최종 평가:** FreEformer는 **전체 주파수 표현을 이용한 변수 간 학습과, 정적 관계를 결합한 attention 보강**을 검증할 가치가 높은 방법입니다. 현재 증거는 여러 벤치마크의 예측 정확도와 일부 데이터 효율을 지지하며, **새 도메인·새 변수 집합·급격한 분포 변화까지 포함하는 일반화 향상은 후속 검증 과제**로 남아 있습니다.

---

## 11. 참고자료 및 출처

### 분석 대상과 공식 구현

1. Yue et al. **FreEformer: Frequency Enhanced Transformer for Multivariate Time Series Forecasting.** arXiv:2501.13989v1, 2025.  
   [arXiv 원문 페이지](https://arxiv.org/abs/2501.13989v1)
2. **FreEformer 공식 구현 저장소.**  
   [GitHub — jackyue1994/FreEformer](https://github.com/jackyue1994/FreEformer)

### 관련 연구 비교에 사용한 원 논문·공식 자료

3. Zhou et al. **Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting.** AAAI, 2021.  
   [공식 논문 페이지](https://ojs.aaai.org/index.php/AAAI/article/view/17325)
4. Wu et al. **Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting.** NeurIPS, 2021.  
   [공식 논문 페이지](https://papers.nips.cc/paper/2021/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html)
5. Zhou et al. **FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting.** ICML, 2022.  
   [공식 논문 페이지](https://proceedings.mlr.press/v162/zhou22g.html)
6. Kim et al. **Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift.** ICLR, 2022.  
   [저자 공식 프로젝트 페이지](https://seharanul17.github.io/RevIN/)
7. Zeng et al. **Are Transformers Effective for Time Series Forecasting?** AAAI, 2023.  
   [공식 논문 페이지](https://ojs.aaai.org/index.php/AAAI/article/view/26317)
8. Nie et al. **A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.** ICLR, 2023.  
   [논문 페이지](https://arxiv.org/abs/2211.14730)
9. Yi et al. **Frequency-domain MLPs are More Effective Learners in Time Series Forecasting.** NeurIPS, 2023.  
   [공식 논문 페이지](https://proceedings.nips.cc/paper_files/paper/2023/hash/f1d16af76939f476b5f040fd1398c0a3-Abstract-Conference.html)
10. Liu et al. **iTransformer: Inverted Transformers Are Effective for Time Series Forecasting.** ICLR, 2024.  
    [공식 논문 페이지](https://proceedings.iclr.cc/paper_files/paper/2024/hash/2ea18fdc667e0ef2ad82b2b4d65147ad-Abstract-Conference.html)
11. Xu et al. **FITS: Modeling Time Series with $10k$ Parameters.** ICLR, 2024.  
    [공식 논문 페이지](https://proceedings.iclr.cc/paper_files/paper/2024/hash/701251e1db4a2e4dd2ef23f5265d5936-Abstract-Conference.html)
12. Piao et al. **Fredformer: Frequency Debiased Transformer for Time Series Forecasting.** KDD, 2024.  
    [논문 페이지](https://arxiv.org/abs/2406.09009)
13. Ye et al. **Frequency Adaptive Normalization For Non-stationary Time Series Forecasting.** NeurIPS, 2024.  
    [공식 논문 페이지](https://proceedings.neurips.cc/paper_files/paper/2024/hash/37c6d0bc4d2917dcbea693b18504bd87-Abstract-Conference.html)
14. Wang et al. **FreDF: Learning to Forecast in the Frequency Domain.** ICLR, 2025.  
    [공식 논문 페이지](https://proceedings.iclr.cc/paper_files/paper/2025/hash/1457fb1e5d72cdc4ecd88bc10f916095-Abstract-Conference.html)
15. Ansari et al. **Chronos-2: From Univariate to Universal Forecasting.** arXiv:2510.15821, 2025.  
    [논문 페이지](https://arxiv.org/abs/2510.15821)
16. Lu et al. **Towards Non-Stationary Time Series Forecasting with Temporal Stabilization and Frequency Differencing.** AAAI, 2026.  
    [공식 논문 페이지](https://ojs.aaai.org/index.php/AAAI/article/view/39585)

**데이터 출처 표기는 FreEformer Appendix D에 근거했습니다.** 그 절은 Max Planck 기상 관측소, California PeMS, CDC FluView, DCRNN의 METR-LA 자료, NASDAQ 데이터셋, Wikipedia Web Traffic 2018–19 등을 출처로 제시합니다. 본 보고서의 데이터 통계와 비교 모델 성능은 해당 외부 데이터나 코드를 새로 실행한 결과가 아니라 **첨부 논문의 보고값**입니다.
