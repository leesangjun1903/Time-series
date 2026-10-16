# CARD: Channel Aligned Robust Blend Transformer for Time Series Forecasting

분석 대상은 첨부된 **ICLR 2024판, 총 39쪽**이며, 아래의 페이지 번호는 PDF에 인쇄된 번호입니다. **‘저자 보고’는 논문의 주장·실험 결과, ‘재계산’은 인쇄된 수치의 산술 검산, ‘해석·제안’은 이를 바탕으로 한 비판적 분석**을 뜻합니다. 재계산은 모델을 다시 학습한 재현 실험이 아닙니다.

## 1. Executive summary — 8문장

1. CARD의 연구 목적은 여러 시계열 변수 사이의 유용한 관계를 활용하면서도, 잡음이나 우연한 상관관계까지 학습하는 과적합을 줄이는 것입니다.  
2. 연구의 출발점은 변수를 독립적으로 예측하면 학습은 안정적이지만 다른 변수의 정보를 놓치고, 변수들을 함께 처리하면 표현력은 높아지지만 잡음에 취약해질 수 있다는 상충관계입니다. **[p.1, Introduction]** ICLR-2024-card-channel-aligned-…  
3. 이를 해결하기 위해 CARD는 시간 구간·변수·학습된 특징 사이의 어텐션을 결합하고, 지수이동평균과 동적 투영을 이용해 계산 부담과 학습 불안정성을 완화합니다. **[pp.4–5, Eqs.3–7, Figure 2]** ICLR-2024-card-channel-aligned-… ICLR-2024-card-channel-aligned-…  
4. Token blend는 토큰 배열의 전체 크기를 유지하면서 같은 헤드의 인접 토큰 정보를 재배열하여, 서로 다른 시간 규모의 패턴을 학습하도록 유도합니다. **[pp.5–6, Figure 3]** ICLR-2024-card-channel-aligned-…  
5. 최종 학습 손실은 미래 예측 오차에 $l^{-1/2}$의 가중치를 적용한 절대오차로, 더 먼 미래의 불확실한 신호에 과도하게 맞추지 않도록 설계되었습니다. **[pp.6–7, Eqs.8–12]** ICLR-2024-card-channel-aligned-…  
6. 저자는 입력 길이 96인 장기 예측에서 데이터셋별 평균 MSE는 7개 중 6개, 평균 MAE는 7개 모두에서 최고 성능을 보고하며, 단변량 단기 예측과 이상 탐지·결측값 복원에서도 유용성을 제시합니다. **[pp.7–8, Tables 1–2; p.21, Table 13; p.39, Table 30]** ICLR-2024-card-channel-aligned-… ICLR-2024-card-channel-aligned-… ICLR-2024-card-channel-aligned-…  
7. 다만 표 사이의 수치 불일치, 일부 평균값의 산술 오류, 손실 변경 시 성능이 악화되는 사례가 있어, 논문의 모든 우월성 표현을 그대로 받아들이기는 어렵습니다. **[Tables 1, 7, 10, 19–20]** ICLR-2024-card-channel-aligned-… ICLR-2024-card-channel-aligned-… ICLR-2024-card-channel-aligned-…  
8. 따라서 CARD의 가장 타당한 기여는 **“변수 간 정보 공유와 잡음 억제를 함께 설계하면 기존 예측 벤치마크에서 효과가 있다”는 경험적 증거**이며, 새로운 변수·새로운 도메인·강한 분포 변화까지 일반화한다는 보장은 별도의 검증이 필요합니다. **[해석; Appendix L, Q, R의 평가 범위에 근거]** ICLR-2024-card-channel-aligned-… ICLR-2024-card-channel-aligned-… ICLR-2024-card-channel-aligned-…  

> **용어 설명**  
> **과적합**: 학습 자료에만 잘 맞고 새로운 자료에서는 성능이 떨어지는 현상입니다.  
> **어텐션(attention)**: 현재 예측에 필요한 다른 정보에 더 큰 가중치를 주어 결합하는 연산입니다.  
> **MSE / MAE**: 각각 오차의 제곱 / 절댓값을 평균한 지표이며, 작을수록 좋습니다.

---

## 2. 연구의 목적과 필요성

### 2.1 무엇이 문제인가?

**저자 보고.** 논문은 다변량 시계열 예측에서 두 설계의 장단점에 주목합니다.

| 설계 | 예측에 사용하는 정보 | 장점 | 논문이 지적하는 약점 |
|---|---|---|---|
| **채널 독립형, CI** | 각 변수 자신의 과거 | 다른 변수의 잡음에 덜 영향받음 | 변수 간 유용한 의존관계를 놓침 |
| **채널 의존형, CD** | 여러 변수의 과거를 함께 사용 | 복잡한 변수 관계를 표현할 수 있음 | 우연한 상관관계와 잡음까지 학습할 수 있음 |

CARD는 “CI가 좋은가, CD가 좋은가”를 이분법적으로 선택하기보다, **CD의 정보 활용 능력을 유지하면서 CI가 보이는 안정성에 접근하려는 모델**입니다. **[p.1, Introduction]** ICLR-2024-card-channel-aligned-…

> **용어 설명**  
> **채널(channel)**: 전력 사용량, 온도, 습도처럼 함께 관측되는 개별 변수입니다.  
> **다변량 시계열**: 여러 변수의 값이 시간에 따라 함께 기록된 자료입니다.  
> **허위 상관관계(spurious correlation)**: 실제로 예측에 안정적으로 도움이 되는 관계가 아니라, 특정 자료 구간에서 우연히 나타난 관계입니다.

**중요한 구분.** CI가 반드시 “변수마다 완전히 별개의 모델을 학습한다”는 뜻은 아닙니다. PatchTST는 변수별 입력을 독립적으로 처리하지만 임베딩과 Transformer 가중치를 공유합니다. 따라서 **가중치 공유**와 **입력 정보의 변수 간 공유**는 서로 다른 개념입니다. **[PatchTST, p.1 및 모델 설명]** [arXiv](https://arxiv.org/abs/2211.14730)

### 2.2 이 문제를 일반화 관점에서 어떻게 읽어야 하는가?

**해석.** CARD가 다루는 일반화는 다음 세 수준으로 나누어 읽는 것이 적절합니다.

| 일반화 수준 | 확인하려는 것 | CARD의 증거 수준 |
|---|---|---|
| 같은 벤치마크의 미래 구간 | 학습하지 않은 시간 구간을 잘 예측하는가? | 직접적인 실험이 있음 |
| 학습 조건 변화 | 초기화, 입력 길이, 모델 크기가 달라도 안정적인가? | 반복 실험과 민감도 분석이 있음 |
| 새로운 환경 | 처음 보는 채널·도메인·샘플링 방식·분포 변화에서도 작동하는가? | 제한적이며, 체계적인 검증은 부족함 |

특히 여러 데이터셋에서 **각각 학습하여** 좋은 성능을 얻었다는 사실과, 한 데이터셋에서 학습한 모델을 **새로운 데이터셋에 그대로 적용해** 좋은 성능을 얻었다는 사실은 같지 않습니다. CARD의 주요 결과는 전자에 해당합니다. **[p.7, §5.1; pp.24–32, Appendix L; p.37, Appendix R.2]** ICLR-2024-card-channel-aligned-… ICLR-2024-card-channel-aligned-… ICLR-2024-card-channel-aligned-…

> **용어 설명**  
> **분포 변화(distribution shift)**: 학습 때와 실제 사용 때 자료의 평균·변동성·변수 관계 등이 달라지는 현상입니다.  
> **도메인(domain)**: 전력, 교통, 금융 등 자료가 생성되는 응용 환경입니다.

---

## 3. 핵심 주장과 근거

아래에서 **‘통계주의’는 효과가 없다는 뜻이 아니라, 제시된 자료만으로 통계적 유의성이나 인과적 기여를 확정하기 어렵다는 뜻**입니다.

| 핵심 주장 | 저자가 제시한 근거 | 위치 | 비판적 해석 |
|---|---|---|---|
| 변수 간 관계를 활용하면 예측력이 좋아진다 | 채널 어텐션을 제거한 모델의 MSE 증가 | pp.35–36, Table 26 | 구성 요소를 **순차적으로 제거**하므로 채널 어텐션만의 독립 효과로 해석하면 안 됨 |
| Token blend가 다중 시간 규모의 정보를 활용한다 | Blend size 1보다 큰 설정에서 대체로 낮은 오차 | p.9, Figure 4 | 큰 blend가 항상 좋지는 않으며, 명시적인 주파수·규모 분해를 증명한 것은 아님 |
| EMA가 학습을 안정화한다 | 강한 평활화에서 여러 설정의 오차 표준편차 감소 | pp.36–37, Figure 40 | 평균 정확도보다 **반복 학습 간 변동성**에 대한 근거 |
| 신호 감쇠 손실이 다른 모델에도 유용하다 | 여러 Transformer·CNN·MLP 계열에서 개선 | p.8, Table 3; p.23, Tables 19–20 | **반례가 존재**하고, MSE→MAE 변경과 시간 가중치 변경의 효과가 섞여 있음 |
| 장기 예측에서 높은 성능을 보인다 | Table 1의 평균 MSE 6/7, 평균 MAE 7/7 최고 | p.8, Table 1 | Table 7과의 수치 불일치 및 일부 근소한 차이에 **통계주의** 필요 |
| 예측 이외의 작업에도 표현이 유용하다 | 이상 탐지, 결측값 복원 성능 | p.8, Table 2; p.39, Table 30 | 작업별 재학습 결과이며, 범용 사전학습 표현의 전이를 입증한 것은 아님 |
| 더 긴 입력이나 적절한 훈련 자료 선택이 도움이 된다 | 입력 길이 증가, 훈련 자료 70% 실험 | p.37, Table 27; p.38, Table 28 | 개선은 비단조적이고 데이터셋별로 다르며, 자료 선택 효과의 원인은 확정되지 않음 |

근거: 구성 요소·평활화 실험, 주요 성능표, 손실 전이 실험 및 자료 크기 실험. ICLR-2024-card-channel-aligned-… ICLR-2024-card-channel-aligned-… ICLR-2024-card-channel-aligned-… ICLR-2024-card-channel-aligned-…

> **용어 설명**  
> **다중 규모(multiscale)**: 짧은 변동, 하루 주기, 장기 추세처럼 서로 다른 시간 범위의 패턴입니다.  
> **절제 실험(ablation)**: 모델의 특정 구성 요소를 제거하거나 바꿔 그 역할을 확인하는 실험입니다.  
> **비단조적**: 설정값을 늘릴 때 성능이 계속 좋아지거나 계속 나빠지는 것이 아니라 방향이 바뀐다는 뜻입니다.

---

## 4. 제안 방법과 모델 구조 — 수식 중심 설명

### 4.1 문제 정의와 기호

논문의 §3에서는 $L$이 입력 길이이고 $T$가 예측 길이지만, §4에서는 $L$을 예측 길이로 다시 사용합니다. 아래에서는 혼동을 피하기 위해 **입력 길이는 $L$, 예측 길이는 $T$로 통일**합니다.

$$
A=[a_{t-L+1},\ldots,a_t]\in\mathbb R^{C\times L},
\qquad
\hat Y=[\hat a_{t+1},\ldots,\hat a_{t+T}]
\in\mathbb R^{C\times T}.
$$

여기서 $a_t$는 시점 $t$의 $C$개 변수 관측값, $A$는 과거 입력, $\hat Y$는 미래 예측입니다. **[p.3, §3]** ICLR-2024-card-channel-aligned-…

이후 사용하는 주요 기호는 다음과 같습니다.

| 기호 | 의미 |
|---|---|
| $C$ | 채널, 즉 변수 개수 |
| $L,\ T$ | 과거 입력 길이, 미래 예측 길이 |
| $P,\ S$ | 패치 길이, 패치를 이동시키는 간격 |
| $N$ | 추가 토큰을 제외한 패치 개수 |
| $d$ | 토큰의 전체 특징 차원 |
| $H,\ d_h=d/H$ | 어텐션 헤드 수, 헤드당 특징 차원 |
| $r$ | 동적 투영 후 요약 표현의 개수 |
| $\alpha$ | EMA의 새 관측값 가중치 |
| $h_3$ | Token blend 크기 |

### 4.2 패치 기반 토큰화

**저자 방법 — Eq.1.**

입력을 길이 $P$의 구간으로 자르면 다음과 같습니다.

$$
N=\left\lfloor\frac{L-P}{S}+1\right\rfloor,
\qquad
\tilde X\in\mathbb R^{C\times N\times P}.
$$

각 패치를 특징 벡터로 바꾸고 위치 정보를 더한 뒤 추가 토큰을 붙입니다.

$$
X=[T_0,\ F_1(\tilde X)+E]
\in\mathbb R^{C\times(N+1)\times d}.
$$

$F_1$은 길이 $P$의 패치를 $d$차원으로 바꾸는 학습 가능한 투영, $E$는 위치 임베딩, $T_0$는 추가 토큰입니다. 점 하나씩 처리하는 대신 **여러 시점의 국소 패턴을 하나의 입력 단위로 묶는 설계**입니다. **[pp.3–4, Eq.1]** ICLR-2024-card-channel-aligned-…

> **용어 설명**  
> **패치(patch)**: 연속된 관측값을 묶은 짧은 구간입니다.  
> **토큰(token)**: 모델이 하나의 단위로 처리하는 특징 벡터입니다.  
> **임베딩(embedding)**: 원래 자료를 학습하기 좋은 숫자 벡터로 바꾸는 과정입니다.

**해석상 주의.** 저자는 $T_0$를 더 긴 이력의 요약 정보를 넣을 수 있는 자리라고 설명하지만, 부록의 기본 코드는 학습 가능한 토큰을 반복해서 붙입니다. 따라서 **$T_0$가 존재한다는 사실만으로 입력 범위 밖의 장기 이력을 실제로 활용했다고 볼 수는 없습니다.** **[p.4; p.16, Appendix B 코드]** ICLR-2024-card-channel-aligned-… ICLR-2024-card-channel-aligned-…

### 4.3 시간 토큰과 은닉 특징에 대한 이중 어텐션

먼저 토큰을 질의·키·값으로 투영합니다.

$$
Q=F_q(X),\qquad K=F_k(X),\qquad V=F_v(X).
$$

$F_q,F_k,F_v$는 학습 가능한 투영이며, 각 헤드에서 특정 채널 $c$를 고정하면

$$
Q_i^{c:},K_i^{c:},V_i^{c:}
\in\mathbb R^{(N+1)\times d_h}
$$

가 됩니다. $i$는 헤드 번호입니다. **[p.4, Eq.2]** ICLR-2024-card-channel-aligned-…

> **용어 설명**  
> **질의 $Q$**: 어떤 정보를 찾는지 나타내는 표현입니다.  
> **키 $K$**: 각 정보가 질의와 얼마나 관련 있는지 비교하기 위한 표현입니다.  
> **값 $V$**: 관련도에 따라 실제로 결합되는 정보입니다.  
> **헤드(head)**: 서로 다른 관계를 병렬적으로 학습하는 어텐션의 하위 단위입니다.

#### A. 시간 토큰 사이의 어텐션

**원문 Eq.3을 그대로 옮기면**

```math
A_{i1}^{c:}
=
\text{softmax}\left(
\frac{
\text{EMA}(Q_i^{c:})
\text{EMA}(K_i^{c:})^\top
}{
\sqrt d
}
\right),
```

$$
O_{i1}^{c:}=A_{i1}^{c:}V_i^{c:}.
$$

$A_{i1}^{c:}$의 크기는 $(N+1)\times(N+1)$이며, **어느 과거 패치를 참고할지** 결정합니다. 위의 $\sqrt d$는 일반적인 Transformer 표기로 교정한 것이 아니라 **논문에 인쇄된 분모**입니다. **[p.4, Eqs.3, 5]** ICLR-2024-card-channel-aligned-…

#### B. 은닉 특징 차원 사이의 어텐션

**원문 Eq.4–5.**

```math
A_{i2}^{c:}
=
\text{softmax}\left(
\frac{
(Q_i^{c:})^\top K_i^{c:}
}{
\sqrt N
}
\right),
\qquad
O_{i2}^{c:}=V_i^{c:}A_{i2}^{c:}.
```

$A_{i2}^{c:}$의 크기는 $d_h\times d_h$입니다. 시간 위치끼리 비교하는 첫 번째 경로와 달리, 두 번째 경로는 **패치 표현을 구성하는 학습된 특징끼리의 관계**를 계산합니다. 저자는 이를 패치 내부의 국소 정보를 활용하는 장치로 설명합니다. **[p.4, Eqs.4–5]** ICLR-2024-card-channel-aligned-…

> **용어 설명**  
> **은닉 특징(hidden feature)**: 온도·습도 같은 원래 변수가 아니라, 신경망 내부에서 학습된 특징입니다. 따라서 **은닉 특징에 대한 어텐션과 원래 채널에 대한 어텐션은 다릅니다.**  
> **Softmax**: 여러 점수를 합이 1인 가중치로 바꾸는 함수입니다.

#### C. EMA의 역할

$$
y_j=\alpha x_j+(1-\alpha)y_{j-1},
\qquad 0 < \alpha < 1.
$$

$x_j$는 평활화 전 표현, $y_j$는 평활화된 표현입니다. $\alpha$가 작을수록 이전 표현을 더 오래 유지하므로 평활화가 강해집니다. CARD는 이를 **$Q$와 $K$에 적용**하며, 기본 설계에서는 $\alpha$를 학습하지 않습니다. **[p.4, EMA 설명 및 각주]** ICLR-2024-card-channel-aligned-… ICLR-2024-card-channel-aligned-…

> **용어 설명**  
> **지수이동평균, EMA**: 최근 값에 더 큰 비중을 주되 과거 값도 점차 작아지는 비중으로 반영하는 평활화 방법입니다. 갑작스러운 변동에 덜 민감한 표현을 만들 수 있습니다.

### 4.4 채널 어텐션과 동적 투영

채널 수가 많을 때 모든 변수 쌍을 비교하면 계산량이 커집니다. CARD는 특정 토큰 위치에서 $C$개의 키와 값을 $r$개의 요약 표현으로 압축합니다.

헤드와 토큰 위치의 첨자를 생략하면 다음과 같습니다.

$$
P_k=\text{softmax}(F_{pk}(K)),
\qquad
P_v=\text{softmax}(F_{pv}(V)),
$$

$$
\tilde K=P_k^\top K,
\qquad
\tilde V=P_v^\top V.
$$

여기서

$$
K,V\in\mathbb R^{C\times d_h},
\qquad
P_k,P_v\in\mathbb R^{C\times r},
\qquad
\tilde K,\tilde V\in\mathbb R^{r\times d_h}.
$$

$F_{pk},F_{pv}$는 현재 입력에 따라 요약 가중치를 만드는 투영입니다. 이후 원래 $C$개 질의가 $r$개 요약 키를 참조하므로, 채널 관계를 계산하는 핵심 행렬이 $C\times C$에서 $C\times r$로 줄어듭니다. **[p.5, Eqs.6–7, Figure 2]** ICLR-2024-card-channel-aligned-…

> **용어 설명**  
> **동적 투영(dynamic projection)**: 모든 입력에 동일한 고정 요약을 쓰는 대신, 현재 입력에서 계산한 가중치로 정보를 요약하는 방식입니다.  
> 여기서 요약은 **가중합**이며, 반드시 채널 방향의 단순 평균이라는 뜻은 아닙니다.

**원문 해석상 주의.** 본문은 압축된 $Q,\tilde K,\tilde V$를 Eqs.3–5에 적용한다고 간략히 서술하지만, 이를 은닉 어텐션의 $Q^\top\tilde K$에 그대로 대입하면 $C\ne r$일 때 행렬 크기가 맞지 않습니다. **Figure 2와 부록 코드는 은닉 특징 경로에서는 압축 전 $Q,K,V$를 사용**합니다. 따라서 재현할 때는 본문의 축약된 설명만으로 구현하면 안 됩니다. **[p.5, Figure 2; p.17, Appendix B]** ICLR-2024-card-channel-aligned-… ICLR-2024-card-channel-aligned-…

### 4.5 Token blend: 무엇을 섞는가?

일반적인 다중 헤드 어텐션은 **같은 시간 위치에서 서로 다른 헤드의 특징을 이어 붙입니다.** CARD는 이를 바꾸어, **같은 헤드의 인접 토큰 정보가 함께 들어가도록 재배열**합니다. **[pp.5–6, Figure 3]** ICLR-2024-card-channel-aligned-…

$M=N+1$이라 두면, 부록 코드와 차원 관계에 맞는 재배열은 다음과 같이 요약할 수 있습니다.

$$
C\times H\times M\times d_h
\rightarrow
C\times h_1\times M\times h_3\times d_h
\rightarrow
C\times M\times h_3\times h_1\times d_h
\rightarrow
C\times M\times d,
$$

$$
h_1h_3=H.
$$

$h_3$는 blend size이고, $h_1$은 남는 헤드 묶음 수입니다. $h_3=1$이면 표준적인 헤드 결합으로 돌아갑니다.

**내부 불일치.** p.6 본문에는 $h_1=Hh_3$로 인쇄되어 있지만, 이는 앞뒤의 차원 관계와 맞지 않습니다. p.17 코드는 `n_heads // merge_size`를 사용하므로 위에서는 **$h_1=H/h_3$라는 코드·차원 일치 해석**을 사용했습니다. **[p.6, §3.4; p.17, token blend 코드]** ICLR-2024-card-channel-aligned-… ICLR-2024-card-channel-aligned-…

**해석.** 이 연산은 출력 토큰 수와 전체 특징 크기를 유지하는 **재배열**입니다. 따라서 “명시적인 다운샘플링”, “이동평균”, “수학적으로 보장된 저역통과 필터”와 동일시하면 안 됩니다. 다중 규모 정보를 배우기 쉬운 표현 구조를 제공한다는 것이 더 정확한 설명입니다.

> **용어 설명**  
> **다운샘플링**: 시간 해상도를 낮춰 관측점 수를 줄이는 작업입니다.  
> **저역통과 필터**: 빠른 변동은 줄이고 느린 변화는 상대적으로 남기는 연산입니다.

### 4.6 전체 정보 흐름

부록의 계산 순서를 함수 형태로 요약하면 다음과 같습니다.

$$
Z_c=B_c(X),\qquad
Z_t=B_t(Z_c),
$$

```math
X^+
=
\text{BN}
\left(
X+\text{Dropout}\bigl(F(Z_c+Z_t)\bigr)
\right).
```

$B_c$는 채널 어텐션 블록, $B_t$는 시간 토큰 어텐션 블록, $F$는 특징 투영, $\text{BN}$은 배치 정규화입니다.  
즉 기본 설계는 **채널 처리 → 시간 처리 → 채널 경로를 다시 더하는 구조**, 논문의 표기로는 ** $c\rightarrow t+c$ **입니다. 마지막에는 채널별 토큰을 펼쳐 선형 출력층에서 미래 $T$개 값을 한 번에 예측합니다. **[p.16, Appendix B; p.36, Figure 38·Table 25]** ICLR-2024-card-channel-aligned-… ICLR-2024-card-channel-aligned-…

입력은 채널별 평균·표준편차로 표준화하고, 예측 후 원래 척도로 되돌립니다. 따라서 최종 성능에는 어텐션뿐 아니라 이러한 입력 처리도 함께 기여합니다. **[p.16; p.18, Appendix D]** ICLR-2024-card-channel-aligned-… ICLR-2024-card-channel-aligned-…

> **용어 설명**  
> **잔차 연결**: 변환 결과에 원래 입력을 더하는 연결입니다.  
> **Dropout**: 학습 중 일부 특징을 무작위로 꺼 과도한 의존을 줄이는 방법입니다.  
> **Encoder-only**: 별도의 Transformer 디코더 없이 입력 표현에서 예측을 생성한다는 뜻입니다. 출력용 선형층이 있다고 해서 encoder–decoder Transformer가 되는 것은 아닙니다.

---

## 5. 신호 감쇠 손실과 이론적 근거

### 5.1 저자가 제안한 목적함수

예측 오차 벡터를 다음과 같이 정의합니다.

$$
e_l(A)=\hat a_{t+l}(A)-a_{t+l},
\qquad l=1,\ldots,T.
$$

일반적인 MSE 학습 목적은

```math
\mathcal L_{\text{MSE}}
=
\mathbb E_A\left[
\frac1T\sum_{l=1}^{T}\|e_l(A)\|_2^2
\right]
```

입니다. $\mathbb E_A$는 입력 이력에 대한 평균, $\|\cdot\|_2^2$는 채널별 제곱오차의 합입니다. 이 목적은 모든 미래 위치에 동일한 가중치를 줍니다. **[p.6, Eq.8; 예측 길이 기호만 $T$로 변경]** ICLR-2024-card-channel-aligned-…

저자는 다음과 같은 확률 과정을 가정합니다.

$$
a_{t+1}\sim\mathcal N(G(a_t),\sigma^2I).
$$

$G$는 상태 전이 함수, $\sigma^2$는 새로 추가되는 잡음의 분산, $I$는 단위행렬입니다. 저자는 $G$의 Lipschitz 상수가 1이라는 조건에서 미래 불확실성을 분석하고, 관측된 현재 상태를 조건으로 미래 공분산이 다음처럼 제한된다고 주장합니다.

```math
\Sigma_l
=
\text{Cov}(a_{t+l}\mid a_t)
\preceq l\sigma^2I.
```

이로부터 가우시안 우도 기반의 근사로 $l^{-1}$ 가중 제곱오차를 제시하고, 이상치에 더 강한 절대오차로 바꾸어 최종적으로 다음 손실을 사용합니다. **[pp.6–7, Eqs.9–12]** ICLR-2024-card-channel-aligned-…

```math
\boxed{
\mathcal L_{\text{CARD}}
=
\mathbb E_A
\left[
\frac1T
\sum_{l=1}^{T}
l^{-1/2}\|e_l(A)\|_1
\right]
}
```

$\|\cdot\|_1$은 채널별 절대오차의 합이고, $l^{-1/2}$는 예측 위치에 따라 미리 정해진 가중치입니다.

> **용어 설명**  
> **1차 마르코프 과정**: 다음 상태가 현재 상태에 의해 결정되고, 현재를 알면 더 과거의 상태는 추가 정보를 주지 않는 확률 모형입니다.  
> **Lipschitz 상수 1**: 입력 두 개 사이의 거리를 출력에서 더 크게 확대하지 않는다는 조건입니다.  
> **공분산 행렬**: 각 변수의 불확실성과 변수들이 함께 변하는 정도를 나타냅니다.  
> **$\preceq$**: 행렬 차이가 양의 준정부호라는 뜻으로, 모든 방향에서 분산이 더 크지 않다는 비교입니다.

### 5.2 이 손실이 실제로 의미하는 것

**해석.** 최종 손실은 **학습된 불확실성에 따른 가중치가 아닙니다.** “예측 거리가 멀수록 불확실성이 커진다”는 가정을 $l^{-1/2}$라는 고정 함수로 구현한 것입니다.

예를 들어,

$$
w_1=1,\qquad
w_{720}=\frac1{\sqrt{720}}\approx0.0373.
$$

따라서 720번째 미래 위치의 **개별 오차**는 첫 번째 위치보다 작은 가중치를 받습니다. 이는 먼 미래 구간 전체의 총가중치가 3.73%라는 뜻은 아닙니다.

라플라스 분포의 오차 크기 매개변수가 $b_l=b_0\sqrt l$라고 가정하면 음의 로그우도에 나타나는 절대오차 항이 이 가중치와 연결됩니다. 하지만 CARD는 $b_l$을 자료에서 추정하거나 예측구간을 출력하지 않으므로, 이를 **불확실성이 보정된 확률 예측 모델**이라고 부르는 것은 과도합니다. **[p.7, Eq.12에 대한 해석]** ICLR-2024-card-channel-aligned-…

> **용어 설명**  
> **라플라스 분포**: 중심 부근이 뾰족하고 정규분포보다 꼬리가 두꺼운 분포로, 절대오차 손실과 연결됩니다.  
> **음의 로그우도**: 관측된 자료가 모델에서 얼마나 그럴듯한지 측정하는 학습 기준입니다.  
> **불확실성 보정(calibration)**: 예를 들어 “90% 예측구간”이 실제로 약 90%의 관측값을 포함하도록 맞는 성질입니다.

### 5.3 이론적으로 취약한 부분

#### A. Eq.9의 공분산 행렬 부등식은 추가 설명이 필요함

**검토 결과.** 일반적인 벡터 함수가 1-Lipschitz라는 사실만으로

$$
\text{Cov}(G(a_t))\preceq\text{Cov}(a_t)
$$

가 따라오지는 않습니다.

간단한 반례로 입력 공분산과 좌표 교환 행렬을

```math
\Sigma=
\begin{pmatrix}
1&0\\
0&0
\end{pmatrix},
\qquad
R=
\begin{pmatrix}
0&1\\
1&0
\end{pmatrix}
```

라 두고 $G(x)=Rx$라고 하면, $G$는 거리를 보존하므로 1-Lipschitz이지만

```math
R\Sigma R^\top=
\begin{pmatrix}
0&0\\
0&1
\end{pmatrix}
\npreceq
\Sigma.
```

여기서 $\Sigma$는 입력 공분산이고, $R$은 두 좌표를 바꾸는 행렬입니다. **이 반례는 논문의 Eq.9 중간 유도가 명시된 조건만으로 충분하지 않다는 뜻**입니다. 최종 미래 분산 상계가 다른 증명으로도 성립할 수 없다는 뜻은 아닙니다. **[검토 대상: p.6, Eq.9]** ICLR-2024-card-channel-aligned-…

#### B. 분산의 상계와 실제 분산은 다름

저자는 분산의 상계를 이용해 우도 손실의 하한에 해당하는 형태를 만듭니다. 그러나 **하한 목적함수를 최소화하는 것과 실제 우도를 최소화하는 것은 일반적으로 같은 문제가 아닙니다.** 또한 모든 자료에서 예측 불확실성이 정확히 $\sqrt l$에 비례한다고 실험적으로 확인한 것도 아닙니다. **[해석; p.6, Eq.11]** ICLR-2024-card-channel-aligned-…

#### C. 부록의 일반화 분석은 전체 CARD에 대한 보장이 아님

Appendix J는 1차원 자기회귀 모형의 2단계 예측을 예로 들어, 매개변수 오차 상계의 상수가 개선될 수 있다고 설명합니다. 이는 손실 설계의 동기를 보여주는 예이지, **다층 CARD 전체나 새로운 도메인에 대한 일반화 정리**는 아닙니다. **[pp.22–24, Eqs.13–16]** ICLR-2024-card-channel-aligned-… ICLR-2024-card-channel-aligned-… ICLR-2024-card-channel-aligned-…

---

## 6. 성능 향상: 보고값과 재계산을 구분한 평가

### 6.1 장기 예측 — Table 1 기준

**저자 설정.** 입력 길이는 96, 예측 길이는 $\{96,192,336,720\}$이며, 각 설정을 10회 반복합니다. 기본 모델은 인코더 블록 2개, 헤드 차원 8, 동적 투영 차원 8을 사용하고, 데이터셋에 따라 모델 크기와 blend size 등이 달라집니다. **[p.7, §5.1; p.18, Table 6]** ICLR-2024-card-channel-aligned-… ICLR-2024-card-channel-aligned-…

아래는 **p.8 Table 1의 표시값만 사용**한 재계산입니다. MSE와 MAE에서 가장 좋은 비교 모델은 서로 다를 수 있습니다.

```math
\text{상대 오차 감소율}
=
\frac{
\text{비교 모델 오차}-\text{CARD 오차}
}{
\text{비교 모델 오차}
}\times100.
```

| 데이터셋 | CARD MSE / MAE | 가장 낮은 비교 모델 오차: MSE / MAE | 재계산 감소율: MSE / MAE |
|---|---:|---|---:|
| ETTm1 | 0.383 / 0.383 | MICN 0.387 / FiLM 0.399 | 1.03% / 4.01% |
| ETTm2 | 0.271 / 0.316 | PatchTST 0.283 / 0.327 | 4.24% / 3.36% |
| ETTh1 | 0.443 / 0.429 | MICN・FEDformer 0.440 / PatchTST 0.444 | **−0.68% / 3.38%** |
| ETTh2 | 0.367 / 0.390 | PatchTST・FiLM 0.384 / 0.406 | 4.43% / 3.94% |
| Weather | 0.240 / 0.262 | MICN 0.243 / PatchTST 0.280 | 1.23% / 6.43% |
| Electricity | 0.169 / 0.258 | MICN 0.187 / MICN・TimesNet 0.295 | 9.63% / 12.54% |
| Traffic | 0.450 / 0.278 | PatchTST 0.488 / MICN 0.316 | 7.79% / 12.03% |

원자료: **[p.8, Table 1]**. 음수는 CARD의 오차가 더 크다는 뜻입니다. ICLR-2024-card-channel-aligned-…

**해석.** Electricity와 Traffic에서 비교적 큰 개선이 보이는 것은 다수 변수의 관계를 활용한다는 설계 동기와 부합합니다. 하지만 Weather의 가장 강한 비교 모델 대비 MSE 개선은 **1.23%**이고, ETTh1에서는 평균 MSE가 오히려 높습니다. 따라서 “고차원 자료에서 항상 큰 폭으로 우월하다”거나 “모든 데이터셋·지표에서 최고”라고 일반화할 수 없습니다.

### 6.2 손실 함수의 전이 효과

**저자 보고.** Table 3에서 ETTm1의 FEDformer MSE는 0.448→0.413, Autoformer는 0.588→0.523으로 낮아집니다. 반면 ETTh1의 Autoformer는 **0.496→0.514로 악화**됩니다. 후자는 인쇄값 기준 약 **3.63% 증가**입니다. **[p.8, Table 3]** ICLR-2024-card-channel-aligned-…

**해석.** 신호 감쇠 손실은 여러 모델에서 유용한 학습 선택지이지만, 보편적으로 우월한 손실로 검증된 것은 아닙니다. 또한 실험은 기본적으로 **일반 MSE에서 시간 가중 MAE로 변경**하므로, 개선을 전부 시간 감쇠 가중치 덕분이라고 분리해서 말하기 어렵습니다.

### 6.3 장기 예측 외의 결과

| 작업 | 저자 보고값 및 재계산 | 해석상 한계 |
|---|---|---|
| **M4 단기 예측** | 평균 SMAPE: CARD 11.614, PatchTST 11.807 → **재계산 1.63% 감소** | M4는 단변량이므로 채널 간 어텐션의 효과를 입증하는 결과는 아님 |
| **이상 탐지** | 평균 F1: CARD 0.890, TimesNet 0.864 | 모든 데이터셋에서 최고는 아님 |
| **SMAP 이상 탐지** | F1 0.715→0.857: **14.2%포인트**, 상대 개선은 약 **19.9%** | ‘14.2% 향상’과 ‘14.2%포인트 향상’을 구분해야 함 |
| **Electricity 결측값 복원** | 평균 MSE 0.092→0.055, MAE 0.210→0.151 | 무작위 점 결측 복원이며, 연속 결측 구간을 포함한 미래 예측과는 다른 작업 |

근거: **[p.21, Table 13; p.8, Table 2; p.39, Table 30]**. ICLR-2024-card-channel-aligned-… ICLR-2024-card-channel-aligned-… ICLR-2024-card-channel-aligned-… ICLR-2024-card-channel-aligned-…

> **용어 설명**  
> **SMAPE**: 실제값과 예측값의 크기를 함께 사용해 정규화한 절대오차 지표입니다.  
> **F1**: 이상이라고 판정한 결과의 정확성과 실제 이상을 찾아낸 비율을 함께 평가합니다.  
> **결측값 복원(imputation)**: 이미 관측된 구간 안에서 빠진 값을 추정하는 작업으로, 미래 예측과는 구분됩니다.

---

## 7. 통계적 취약점과 비교 불가능한 수치

### 7.1 원문 내부에서 확인되는 수치 불일치

아래는 추정으로 고친 결과가 아니라 **인쇄된 표에서 직접 확인되는 문제**입니다.

| 항목 | 원문에서 확인되는 내용 | 판정 |
|---|---|---|
| **Table 1 vs. Table 7** | Traffic의 CARD 평균이 Table 1에서는 **0.450 / 0.278**, Table 7에서는 **0.453 / 0.282** | 어느 것이 최종 실험값인지 문서만으로 확정 불가 |
| **“MAE 100% 최고” 주장** | Table 7의 Weather, 예측 길이 720에서 CARD MAE **0.353**, PatchTST **0.347** | 적어도 인쇄된 표는 100% 최고 주장을 뒷받침하지 않음 |
| **Table 10 평균값** | ETTh2 DLinear MSE는 0.289, 0.383, 0.448, 0.605인데 평균은 **0.259**로 표시 | 네 표시값의 산술평균은 **0.43125**이므로 평균 행에 명백한 불일치 |
| **손실 효과의 일반화** | Autoformer의 ETTh1 MSE가 손실 변경 후 악화 | “모든 모델에서 개선”이라는 해석은 불가 |

근거: **[p.8, Tables 1·3; p.19, Table 7; p.20, Table 10]**. ICLR-2024-card-channel-aligned-… ICLR-2024-card-channel-aligned-… ICLR-2024-card-channel-aligned-… ICLR-2024-card-channel-aligned-… ICLR-2024-card-channel-aligned-…

**중요:** 이러한 불일치가 CARD의 모든 성능 개선을 무효화하지는 않습니다. 그러나 **세부 순위, 승률, 개선율을 정확한 확정값으로 인용하려면 원시 결과나 정정표가 필요**합니다.

### 7.2 반복 실험이 의미하는 것과 의미하지 않는 것

**통계주의 ① — 10개 시드는 10개의 독립적인 현실 환경이 아니다.**  
Table 8은 10개 무작위 시드에 대한 오차 통계를 제공합니다. 이는 초기화·학습 무작위성에 대한 안정성이지, 새로운 지역·계절·산업 환경에 대한 불확실성을 직접 측정한 것이 아닙니다. 또한 본문은 표준편차, 표 제목은 표준오차라는 표현을 혼용합니다. **[pp.18–19, Appendix E·Table 8]** ICLR-2024-card-channel-aligned-… ICLR-2024-card-channel-aligned-…

> **용어 설명**  
> **시드(seed)**: 무작위 초기화와 학습 과정을 재현하기 위한 시작값입니다.  
> **표준편차**: 반복 결과 자체가 얼마나 퍼지는지 나타냅니다.  
> **표준오차**: 평균값 추정이 얼마나 불확실한지 나타내며, 표준편차와 같은 값이 아닙니다.

**통계주의 ② — 근소한 차이의 유의성을 확인할 정보가 부족하다.**  
CARD의 변동성만으로 두 모델의 차이가 유의한지 판단할 수는 없습니다. 비교 모델과 동일한 반복 조건의 결과, 대응되는 오차 차이, 신뢰구간 등이 필요합니다. 따라서 0.001 수준의 차이를 곧바로 확실한 우월성으로 읽어서는 안 됩니다. **[평가 대상: p.19, Table 8; p.20, Table 10]** ICLR-2024-card-channel-aligned-… ICLR-2024-card-channel-aligned-…

### 7.3 구성 요소의 효과를 분리하기 어려운 부분

**통계주의 ③ — Table 26은 순차 제거 실험이다.**  
Electricity 평균 MSE는 전체 CARD 0.168 → 은닉 어텐션 제거 0.186 → 추가로 채널 어텐션 제거 0.198입니다. 따라서 마지막 차이는 **이미 은닉 어텐션이 없는 모델에서 채널 어텐션을 제거한 효과**입니다. 이를 전체 CARD에서 채널 경로만 제거한 효과로 해석하면 안 됩니다. **[p.36, Table 26]** ICLR-2024-card-channel-aligned-…

**통계주의 ④ — 손실 변경의 두 효과가 섞여 있다.**  
최소한 다음 네 조합을 같은 조건에서 비교해야 시간 감쇠 자체의 효과를 분리할 수 있습니다.

| 기본 오차 | 시간 가중치 없음 | 시간 감쇠 가중치 있음 |
|---|---|---|
| 제곱오차 | 일반 MSE | 가중 MSE |
| 절대오차 | 일반 MAE | CARD형 가중 MAE |

이는 **추가 검증 제안**입니다. 원문의 주요 손실 비교를 이와 같은 완전한 요인 분해 실험으로 간주해서는 안 됩니다. **[검토 대상: Tables 3, 19–21]** ICLR-2024-card-channel-aligned-… ICLR-2024-card-channel-aligned-…

### 7.4 직접 비교하면 안 되는 수치

| 비교 | 왜 직접 비교하기 어려운가? |
|---|---|
| **Table 1과 Table 10의 개선율** | Table 1은 입력 길이 96, Table 10은 CARD의 긴 입력 및 비교 모델의 최적 보고·탐색 결과를 사용 |
| **장기 예측 MSE와 M4 SMAPE** | 자료, 예측 길이, 척도와 지표가 다름 |
| **결측 복원 MSE와 미래 예측 MSE** | 이용 가능한 정보와 문제 자체가 다름 |
| **CARD와 최신 기반 모델의 논문별 종합 점수** | 학습 자료, 사전학습 여부, 정규화, 입력 길이, 데이터 누출 통제 등이 다름 |
| **서로 다른 하드웨어·배치 크기의 실행시간** | 계산 환경이 다르면 시간 차이를 구조의 효율 차이로 분리할 수 없음 |

CARD 자체도 입력 길이 96의 비교와 최적 입력 길이 비교를 별도로 보고합니다. 공정 비교를 연구 주제로 다룬 TFB 역시 통일되지 않은 평가 절차의 문제를 강조합니다. **[CARD pp.19–20, Appendix G; TFB, 초록]** ICLR-2024-card-channel-aligned-… [arXiv](https://arxiv.org/abs/2403.20150)

### 7.5 효율성과 “선형 시간” 표현의 한계

**저자 보고.** 고정 stride에서 CARD의 시간 길이에 대한 복잡도는 $O(L^2/S^2)$로 설명됩니다. $S$를 $\sqrt L$ 규모로 증가시키면 거의 선형처럼 만들 수 있다는 논의가 있지만, 이는 **입력 길이가 증가할 때 패치 이동 간격도 바꾸는 조건**입니다. 고정 $S$에서 자동으로 선형이 되는 것은 아닙니다. **[p.33, Appendix N]** ICLR-2024-card-channel-aligned-…

실측에서도 CARD가 항상 빠르지는 않습니다. ETTh1 추론 시간은 CARD 0.0046초, PatchTST 0.0021초, DLinear 0.0009초로 보고됩니다. 이 값은 논문이 지정한 동일 실험 환경 안에서만 비교해야 합니다. **[p.33, Table 24]** ICLR-2024-card-channel-aligned-…

---

## 8. 가장 중요한 그림과 해석

### Figure 1과 Figure 2 — 모델의 핵심은 ‘세 축의 관계’를 분리해 처리하는 것

**위치:** p.3 Figure 1, p.5 Figure 2.

Figure 1은 패치 입력, 어텐션 블록, 채널별 출력의 전체 구성을 보여주고, Figure 2는 시간·채널 축 처리 안에 **토큰 방향과 은닉 특징 방향의 두 어텐션 경로**가 존재함을 보여줍니다. **[저자 설계]** ICLR-2024-card-channel-aligned-… ICLR-2024-card-channel-aligned-…

**해석.** CARD의 차별점은 단순히 “채널 어텐션을 추가했다”가 아닙니다. **시간 위치의 관계, 변수의 관계, 학습된 특징의 관계를 서로 다른 축에서 처리하고 결합한다**는 점이 중요합니다. 다만 이것이 실제 물리적 시간 지연을 추정하거나 비동기 센서를 자동 정렬한다는 뜻은 아닙니다.

### Figure 3 — 다중 규모를 토큰 수 감소 없이 유도

**위치:** p.6 Figure 3.

그림은 표준 헤드 결합과 인접 토큰 중심의 재배열을 대비합니다. **[저자 설계]** ICLR-2024-card-channel-aligned-…

**해석.** 가장 중요한 부분은 **배열의 최종 크기를 그대로 유지한다는 점**입니다. 명시적인 계층형 다운샘플링 없이 다음 블록이 더 넓은 범위의 정보를 조합하도록 유도합니다. 반대로, 이 그림만으로 특정 주파수 대역이 분리되거나 장기 패턴이 반드시 더 잘 보존된다고 주장할 수는 없습니다.

### Figure 4 — “섞을수록 좋다”가 아니라 최적 혼합 범위가 존재

**위치:** p.9 Figure 4.

Blend size를 1, 2, 4, 8, 16으로 바꾸었을 때, 일반적인 헤드 결합에 해당하는 1보다 큰 설정이 대체로 유리하지만 증가에 따른 개선이 일관되지는 않습니다. 저자는 지나친 혼합이 유용한 정보를 과도하게 평활화할 수 있다고 추측합니다. **[저자 보고]** ICLR-2024-card-channel-aligned-…

**해석.** 일반화 향상을 위해서는 “가장 큰 blend”보다 **자료의 주기·급변·잡음 수준에 맞는 blend**가 중요합니다. 이는 후속 연구에서 고정 크기 대신 입력에 따라 혼합 규모를 선택할 동기가 됩니다.

### Figure 40 — 정확도보다 학습 변동성에 관한 그림

**위치:** p.37 Figure 40.

이 그림은 EMA 설정에 따른 **MSE·MAE의 표준편차**를 보여줍니다. 저자는 강한 평활화가 여러 설정에서 반복 학습 간 변동성을 줄인다고 보고합니다. **[저자 보고]** ICLR-2024-card-channel-aligned-…

**해석.** 이 그림을 “EMA가 모든 경우 평균 예측 오차를 줄인다”는 증거로 읽으면 안 됩니다. 보다 직접적인 결론은 **학습 결과의 재현성과 안정성을 높일 가능성**입니다.

### Figure 36과 Figure 37 — 설명 가능한 단서는 있지만 인과적 증거는 아님

**위치:** pp.34–35, Figures 36–37.

저자는 어텐션 맵이 매끄럽고, 입력 패치와 미래 곡선의 DTW 점수에 정성적으로 대응한다고 설명합니다. **[pp.33–35, Appendix O]** ICLR-2024-card-channel-aligned-… ICLR-2024-card-channel-aligned-… ICLR-2024-card-channel-aligned-…

> **용어 설명**  
> **어텐션 맵**: 어떤 입력 위치에 높은 가중치가 부여되었는지 보여주는 그림입니다.  
> **DTW, Dynamic Time Warping**: 시간축을 부분적으로 늘이거나 줄여 두 곡선의 모양 차이를 비교하는 방법입니다.

**해석.** 선택된 예시의 시각적 대응만으로 “모델이 실제 원인을 찾았다”고 결론 내릴 수 없습니다. 상관계수·통계검정·입력 제거 실험이 제시되지 않았고, DTW가 거리인지 변환된 유사도인지에 따라 점수의 방향도 구분해야 합니다.

---

## 9. 2020년 이후 관련 연구 비교

아래는 **2026년 공개 논문까지 확인한 선택적 비교**입니다. 각 논문의 성능 우월성 주장은 해당 논문의 실험 조건에 한정하며, 서로 다른 논문의 숫자를 이어 붙여 통합 순위를 만들지 않았습니다.

### 9.1 2020–2023: CARD가 대응한 연구 흐름

| 연구 | 해당 저자가 제안·보고한 내용 | CARD와의 차이 및 일반화 관점의 해석 |
|---|---|---|
| **N-BEATS, ICLR 2020** | 잔차 연결과 기저 확장으로 예측하는 MLP 계열 구조. **[p.1, 초록]** [OpenReview](https://openreview.net/forum?id=r1ecqn4YwB) | 복잡한 어텐션 없이도 강한 예측이 가능하다는 비교 기준. CARD의 이점은 단순 모델보다 복잡하다는 사실이 아니라 추가 관계 학습의 실익으로 보여야 함 |
| **Informer, AAAI 2021** | 희소 어텐션과 압축을 통해 긴 입력의 계산 부담을 줄임. **[초록]** [AAAI Publications](https://ojs.aaai.org/index.php/AAAI/article/view/17325) | 효율적 장기 의존성에 초점. CARD는 여기에 변수 관계와 잡음 강건성 문제를 전면에 둠 |
| **Autoformer, NeurIPS 2021 / FEDformer, ICML 2022** | 각각 분해·자기상관, 주파수 기반 분해 구조를 사용. **[각 p.1, 초록]** [NeurIPS](https://proceedings.neurips.cc/paper/2021/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html) | 명시적인 추세·주기 처리와 달리 CARD는 token blend로 규모 정보를 암묵적으로 조합 |
| **RevIN, ICLR 2022** | 입력별 통계를 제거한 뒤 출력에서 복원하여 분포 변화에 대응. **[초록]** [OpenReview](https://openreview.net/forum?id=cGDAkQo1C0p) | CARD도 사용하므로, 전체 일반화 개선을 CARD 어텐션만의 효과로 돌리면 안 됨 |
| **DLinear, AAAI 2023** | 단순 선형 계열 모델이 당시 여러 Transformer보다 강한 결과를 보임. **[초록]** [AAAI Publications](https://ojs.aaai.org/index.php/AAAI/article/view/26317) | 반드시 포함해야 할 강한 단순 기준선. 다만 “Transformer는 원천적으로 부적합”하다는 정리는 아님 |
| **PatchTST, ICLR 2023** | 패치와 채널 독립 처리, 공유 가중치, 마스킹 기반 사전학습·전이 실험. **[p.1 및 모델·자기지도학습 설명]** [arXiv](https://arxiv.org/abs/2211.14730) | CARD는 채널 간 정보를 추가하지만, **전이 일반화의 증거에서는 PatchTST가 이미 별도 실험을 제시했다는 점**을 고려해야 함 |
| **Crossformer, ICLR 2023** | 시간·변수 의존성을 처리하는 2단계 어텐션과 계층형 구조. **[초록 및 모델 설명]** [OpenReview](https://openreview.net/forum?id=vSVLM2j9eie) | CARD와 가장 가까운 선행 계열. CARD는 encoder-only, 은닉 특징 어텐션, 크기를 유지하는 blend, 강건 손실을 결합 |
| **Capacity–Robustness Trade-off, 2023 사전공개·2024 TKDE** | CI와 CD의 표현력·강건성 상충관계를 분석. **[초록]** [arXiv](https://arxiv.org/abs/2304.05206) | CARD의 문제 설정을 이해하는 직접적인 배경. 채널 공유의 이익과 해를 동시에 평가해야 한다는 관점을 제공 |

> **용어 설명**  
> **기저 확장**: 여러 기본 패턴을 조합해 복잡한 곡선을 표현하는 방식입니다.  
> **자기지도 사전학습**: 정답 라벨 없이 일부 입력을 가리거나 변형한 뒤 복원하도록 학습하는 방식입니다.  
> **전이학습**: 다른 자료에서 학습한 표현이나 모델을 새로운 문제에 활용하는 방법입니다.

### 9.2 2024–2026: 일반화의 기준이 어떻게 확장되었는가?

| 연구 | 저자 방법·평가와 위치 | CARD에 대한 비교 해석 |
|---|---|---|
| **iTransformer, ICLR 2024** | 변수 하나의 전체 이력을 하나의 토큰으로 만들고 변수 사이에 어텐션 적용. **[p.4, Figure 4]**; 일부 변수로 학습한 뒤 다른 변수까지 예측하는 평가. **[p.7, Figure 5]** [ICLR Proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/file/2ea18fdc667e0ef2ad82b2b4d65147ad-Paper-Conference.pdf) | CARD는 시간 패치와 채널 축을 번갈아 처리. 후속 비교에서는 평균 MSE 외에 **미관측 채널 일반화**를 공통 평가해야 함 |
| **SAMformer, ICML 2024** | 얕은 채널 어텐션 모델에 sharpness-aware minimization을 적용해 최적화 안정성을 개선. **[pp.3–5, §2.3–2.4]** [arXiv](https://arxiv.org/pdf/2402.10198) | CARD의 EMA·강건 손실은 표현과 오차 가중치를 조정하고, SAMformer는 매개변수 주변의 손실 민감도를 조정. 상호 보완 가능성은 있지만 결합 효과는 미검증 |
| **TimeMixer, ICLR 2024** | 여러 해상도의 시계열을 명시적으로 만들고 추세·계절 성분을 규모 간 혼합. **[pp.3–5, Figure 1]** [ICLR Proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/file/a7ac8a21e5a27e7ab31a5f42a0117bdb-Paper-Conference.pdf) | CARD의 암묵적 재배열과 명시적 다중 해상도 처리의 대조군. 계산량·급변 보존·분포 변화에 따른 효과를 함께 비교해야 함 |
| **TimeXer, NeurIPS 2024** | 예측 대상의 패치와 외생 변수 표현을 구분하고 전역 토큰을 통해 결합. **[§3, Figure 2]** [arXiv](https://arxiv.org/html/2402.19072v3) | 모든 채널을 비슷한 지위로 처리하는 것과, 예측 대상·과거에만 알려진 변수·미래에도 알려진 변수를 구분하는 것은 다름 |
| **Moirai, ICML 2024** | 다양한 자료의 통합 학습, 가변 채널 수를 위한 any-variate attention, 확률적 출력. **[p.4, Figure 2, §3.1.2]** [arXiv](https://arxiv.org/pdf/2402.02592) | 데이터셋별 CARD 학습보다 문제 범위가 넓음. CARD의 구조를 범용화하려면 가변 채널, 채널 순서, 확률 예측을 별도로 다뤄야 함 |
| **TimeFilter, ICML 2025** | 패치별 시공간 그래프에서 불필요한 관계를 제거하고 필요한 관계를 유지. **[초록]** [Proceedings of Machine Learning Research](https://proceedings.mlr.press/v267/hu25ac.html) | CARD의 고정된 정보 혼합 구조를 넘어, **어떤 위치에서 어떤 관계를 사용할지 선택**하는 방향 |
| **Chronos-2, 2025** | 시간 어텐션과 관련 시계열 그룹 사이의 어텐션을 결합하여 단변량·다변량·공변량 포함 예측 지원. **[pp.5–6, §3.2]** [arXiv](https://arxiv.org/pdf/2510.15821) | 축을 분리해 정보를 공유한다는 구조적 공통점이 있으나, 대규모 사전학습과 새로운 자료에 대한 적용이라는 평가 범위는 다름 |
| **ChannelTokenFormer, ICLR 2026** | 채널 관계·비동기 샘플링·시험 시 연속 결측을 함께 다루는 구조. **[pp.5–7, Figures 2–3, Table 1]** [ICLR Proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/file/f924dab7a06d0463fe238904678acaee-Paper-Conference.pdf) | CARD가 충분히 평가하지 않은 현실 조건을 명시적인 문제로 삼음. 단, 해당 Table 1에는 CARD가 없어 **직접 수치 우열을 말할 수 없음** |

> **용어 설명**  
> **Sharpness**: 모델 매개변수를 조금 바꿨을 때 손실이 얼마나 급격히 커지는지 나타내는 성질입니다.  
> **외생 변수**: 예측 대상 밖에서 제공되는 보조 정보입니다. 예를 들어 전력 예측에서 달력이나 날씨 예보가 해당할 수 있습니다.  
> **기반 모델(foundation model)**: 다양한 자료로 미리 학습해 여러 후속 문제에 재사용하는 모델입니다.  
> **제로샷 예측**: 대상 데이터셋으로 추가 학습하지 않고 예측하는 것입니다.  
> **비동기 샘플링**: 채널마다 관측 간격이나 관측 시점이 다른 상황입니다.

### 9.3 비교를 명확히 하는 대표 수식: CARD와 SAMformer

CARD는 예측 위치별 오차의 가중치를 바꾸는 반면, SAM 계열 목적은 개념적으로 다음과 같습니다.

$$
\min_\theta
\max_{\|\delta\|_2 < \rho}
\mathcal L_{\text{train}}(\theta+\delta).
$$

$\theta$는 모델 매개변수, $\delta$는 작은 매개변수 교란, $\rho$는 허용 교란 크기입니다. 즉 **매개변수가 조금 달라져도 손실이 지나치게 나빠지지 않는 해**를 찾습니다. **[SAMformer p.4, §2.3]** [arXiv](https://arxiv.org/pdf/2402.10198)

**해석.** CARD는 “어떤 미래 오차에 집중할 것인가”, SAMformer는 “얼마나 불안정한 해를 피할 것인가”에 개입합니다. 두 방향을 결합할 연구 가치는 있지만, 추가 연산량과 함께 검증해야 하며 결합하면 반드시 일반화가 좋아진다고 단정할 수는 없습니다.

### 9.4 CARD가 후속 연구에 미친 영향: 확인된 것과 추론

**확인된 관계.** ChannelTokenFormer의 ICLR 2026 논문은 CARD를 채널 의존형 선행 연구로 직접 인용하면서, 기존 설계가 충분히 다루지 못한 비동기 관측과 연속 결측 문제를 논의합니다. 따라서 CARD가 **채널 간 정보 활용 연구의 선행 기준점으로 자리 잡았다는 점**은 확인할 수 있습니다. **[ChannelTokenFormer pp.1–3 및 참고문헌의 CARD 항목]** [ICLR Proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/file/f924dab7a06d0463fe238904678acaee-Paper-Conference.pdf)

**해석.** 더 넓은 연구적 의미는 다음과 같습니다.

> 질문이 **“채널을 섞을 것인가?”**에서 **“언제, 어떤 채널을, 어떤 시간 규모에서, 얼마나 신뢰하며 섞을 것인가?”**로 이동하고 있습니다.

TimeFilter의 관계 선택, Moirai의 가변 채널 처리, Chronos-2의 그룹 정보 공유는 이러한 관점에서 비교할 수 있습니다. 다만 구조가 유사하다는 이유만으로 이들이 CARD에서 직접 파생되었다고 주장하지는 않습니다. [Proceedings of Machine Learning Research](https://proceedings.mlr.press/v267/hu25ac.html)

---

## 10. 문서가 답하지 않는 질문

아래 질문은 논문의 실험 범위에서 충분히 답해지지 않습니다.

| 남아 있는 질문 | 왜 중요한가? | 필요한 추가 검증 |
|---|---|---|
| **어떤 채널 관계가 안정적으로 유용한가?** | 채널 수가 많다는 사실과 유용한 관계가 많다는 사실은 다름 | 무관·잡음 채널 추가, 관계 구조가 바뀌는 합성 자료 실험 |
| **처음 보는 채널이나 바뀐 채널 수에서도 작동하는가?** | 실제 센서망에서는 센서가 추가·제거됨 | 채널 일부만 학습, 새로운 채널 평가, 채널 수 변화 |
| **채널 순서를 바꿔도 같은 의미의 예측을 하는가?** | 채널 축의 평활화·혼합이 배열 순서에 영향을 받을 수 있음 | 채널과 메타데이터를 함께 순열 변경하는 실험 |
| **$l^{-1/2}$가 실제 불확실성을 잘 근사하는가?** | 고정 가중치가 자료 특성과 맞지 않을 수 있음 | 미래 거리별 잔차 분산·분포 측정, 학습형 불확실성과 비교 |
| **먼 미래의 중요한 급변을 희생하지 않는가?** | 평균 오차 개선과 중요한 사건 예측은 다른 목표 | 최대 부하, 급변, 최장기 구간을 별도 평가 |
| **70% 훈련 자료의 이점은 왜 생기는가?** | 분포 변화, 최근성, 학습 횟수 등 원인이 섞일 수 있음 | 최근 70%·무작위 70%·초기 70% 비교, 업데이트 수 통제 |
| **새 도메인·연속 결측·비동기 입력에 전이되는가?** | 실제 적용의 핵심 일반화 조건 | 도메인 외 평가, 연속 결측, 채널별 다른 샘플링 주기 |
| **표의 불일치는 어떤 실험 버전에서 발생했는가?** | 세부 순위와 효과 크기 재현에 필수 | 확정 코드 버전, 원시 시드별 로그, 정정된 결과표 공개 |

이 질문들은 각각 논문의 제한된 채널 절제 실험, 고정 감쇠 함수 비교, 자료 크기 실험, 무작위 결측 복원 실험에서 확장되는 것입니다. **[Tables 21, 26, 28, 30]** ICLR-2024-card-channel-aligned-… ICLR-2024-card-channel-aligned-… ICLR-2024-card-channel-aligned-… ICLR-2024-card-channel-aligned-…

---

## 11. 결론과 후속 연구 방향 — 일반화 성능을 중심으로

### 11.1 저자가 실제로 제시한 시사점과 후속 계획

**저자의 결론.** CARD는 시간·채널·은닉 특징의 관계, 다중 규모 정보, 잡음에 강한 학습 손실을 함께 설계하는 것이 시계열 예측에 유용하다고 주장합니다. **[p.9, Conclusion]** ICLR-2024-card-channel-aligned-…

다만 결론 절의 제목에는 “Future Works”가 포함되어 있어도, 본문에는 상세한 후속 연구 계획이 나열되어 있지 않습니다. 구체적으로 확인되는 계획은 부록의 다음 두 가지입니다.

| 저자가 명시한 후속 연구 | 위치 |
|---|---|
| 자료 특성에 맞는 다른 표준화·정규화 방법의 분석 | p.18, Appendix D |
| 훈련 자료의 양과 분포 변화가 예측에 미치는 영향의 추가 탐구 | p.37, Appendix R.2 |

ICLR-2024-card-channel-aligned-… ICLR-2024-card-channel-aligned-…

**대규모 사전학습, 인과 모델링, 제로샷 기반 모델 개발은 이 논문이 구체적인 후속 계획으로 약속한 내용이 아닙니다.** 아래는 별도의 연구 제안입니다.

### 11.2 제안 1 — 채널 독립·의존 예측을 입력에 따라 선택

**제안.** 모든 입력에서 동일한 강도로 채널을 섞기보다, 채널 관계가 신뢰할 만할 때만 공유하도록 설계할 수 있습니다.

```math
\hat a_{c,t+l}
=
\bigl(1-g_{c,l}(A)\bigr)
f_{\text{CI},c,l}(A_c)
+
g_{c,l}(A)
f_{\text{CD},c,l}(A).
```

$A_c$는 채널 $c$의 과거, $f_{\text{CI},c,l}$은 해당 채널만 사용하는 예측기, $f_{\text{CD},c,l}$은 전체 채널을 사용하는 예측기입니다. $g_{c,l}(A)\in[0,1]$는 입력·채널·예측 거리에 따라 정보 공유 강도를 선택하는 함수입니다.

**기대하는 일반화 효과는 가설입니다.** 관계가 안정적이면 CD를 활용하고, 관계가 바뀌거나 잡음 채널이 많으면 CI 쪽으로 이동하도록 유도할 수 있습니다. 검증에서는 무관 채널 추가, 상관관계 반전, 일부 채널 고장 같은 조건을 포함해야 합니다.

> **용어 설명**  
> **게이트(gate)**: 여러 정보 경로 중 어느 경로를 얼마나 사용할지 결정하는 학습 가능한 가중치입니다.

### 11.3 제안 2 — 고정 시간 감쇠를 자료 기반 불확실성으로 확장

**제안.** 고정 $l^{-1/2}$ 대신 채널·입력·예측 거리별 오차 규모를 예측하는 방법을 비교할 수 있습니다.

```math
\mathcal L_{\text{adaptive}}
=
\sum_{c=1}^{C}\sum_{l=1}^{T}
\left[
\frac{|e_{c,l}|}{b_{c,l}(A)}
+
\log b_{c,l}(A)
\right],
\qquad
b_{c,l}(A)>0.
```

$e_{c,l}$은 채널 $c$의 $l$단계 예측 오차이고, $b_{c,l}(A)$는 모델이 추정하는 양의 오차 규모입니다. 이는 라플라스 음의 로그우도에서 상수항을 제외한 형태입니다. $\log b$ 항은 오차 규모를 무작정 키워 잔차 항을 줄이는 것을 억제합니다.

이 방향은 **CARD의 고정 불확실성 가정을 검증 가능한 예측 대상으로 바꾸는 것**입니다. 평가에는 평균 MSE·MAE뿐 아니라 예측구간의 실제 포함률, 최장기 오차, 급변 구간의 오차가 필요합니다. 학습형 불확실성이 오히려 중요한 어려운 사례를 회피하는지도 점검해야 합니다.

### 11.4 제안 3 — 채널 수·채널 순서·도메인이 바뀌는 일반화 평가

**제안.** 범용적인 다변량 모델을 지향한다면 다음 성질을 검증할 가치가 있습니다.

$$
f(\Pi A)=\Pi f(A).
$$

$\Pi$는 채널 순서를 바꾸는 순열 행렬이고 $f$는 예측 모델입니다. 이 조건은 **입력 채널의 배열 순서만 바꾸면 출력도 같은 방식으로 재배열될 뿐, 예측의 의미는 달라지지 않아야 한다**는 뜻입니다.

CARD가 이 성질을 만족한다고 논문은 증명하지 않습니다. 따라서 채널 정보와 메타데이터를 함께 재배열하는 실험, 미관측 채널 평가, 채널 추가·삭제 평가를 별도로 수행해야 합니다. Moirai의 any-variate 설계와 iTransformer의 미관측 변수 실험은 이 비교를 위한 유용한 참고점입니다. **[Moirai p.4; iTransformer p.7, Figure 5]** [arXiv](https://arxiv.org/pdf/2402.02592)

> **용어 설명**  
> **순열 등변성(permutation equivariance)**: 입력 순서를 바꾸면 출력 순서도 대응해서 바뀌되, 각 대상에 대한 계산 의미는 유지되는 성질입니다.

### 11.5 제안 4 — 분포 변화와 데이터 선택을 분리해 검증

**저자 관찰.** 훈련 자료 70%를 사용할 때 ETTm2·ETTh2·Traffic의 성능이 좋아지는 결과가 보고됩니다. 그러나 어떤 70%를 선택했는지와 성능 개선의 원인이 충분히 분리되어 있지 않습니다. **[pp.37–38, Table 28]** ICLR-2024-card-channel-aligned-… ICLR-2024-card-channel-aligned-…

**제안.** 최근 자료만 쓰기, 오래된 자료의 가중치 줄이기, 변화 시점별 모델을 쓰기를 비교하되, 학습 업데이트 수와 검증 절차를 통제해야 합니다. 특히 여러 시점에서 학습·평가 구간을 앞으로 이동시키는 평가가 단일 분할보다 이 질문에 직접적입니다.

> **용어 설명**  
> **이동 시점 평가(rolling-origin evaluation)**: 여러 과거 시점을 예측 시작점으로 삼아, 그때까지의 자료로 학습하고 그 이후를 예측하는 평가입니다.

### 11.6 제안 5 — 기반 모델·현실 결측 환경으로 확장하되 평가 절차를 먼저 고정

**제안.** 패치 복원 사전학습이나 다양한 채널 관계를 가진 합성 자료로 CARD 계열 표현을 사전학습한 뒤, 새로운 도메인에서의 성능을 평가할 수 있습니다. 그러나 사전학습 자료가 시험 자료와 겹치면 전이 성능을 과대평가할 수 있으므로, 자료 출처와 시간 구간의 중복을 통제해야 합니다.

공정한 비교 절차에는 TFB, 제로샷과 사전학습 자료 분리를 고려하는 데는 GIFT-Eval, 비동기·연속 결측 조건에는 ChannelTokenFormer의 평가 설정을 참고할 수 있습니다. [arXiv](https://arxiv.org/abs/2403.20150)

### 최종 판단

**CARD는 “채널 의존형 모델도 적절한 정보 압축·표현 재배열·손실 설계를 결합하면 강한 예측 모델이 될 수 있다”는 설득력 있는 연구 방향을 제시합니다.** 특히 채널 간 정보를 무조건 제거하는 대신, 잡음에 취약한 정보 결합을 제어하려는 접근이 중요합니다. **[pp.1–2, 연구 동기·기여; p.9, 결론]** ICLR-2024-card-channel-aligned-… ICLR-2024-card-channel-aligned-… ICLR-2024-card-channel-aligned-…

그러나 **강건한 벤치마크 학습, 통계적으로 확인된 모델 우월성, 새로운 환경으로의 일반화는 서로 다른 주장**입니다. 후속 연구의 우선순위는 더 많은 평균 순위 1위를 얻는 것보다, **채널 관계와 자료 분포가 달라졌을 때 어떤 정보를 신뢰하고 어떤 정보를 버려야 하는지 규명하는 것**이어야 합니다. 동시에 원문의 수치 불일치를 해소하고, 손실·구조·정규화의 효과를 분리하는 재현 실험이 그 출발점입니다.

---

## 참고자료 — 실제 분석·비교에 사용한 문헌

아래는 본문에서 사용한 참고자료의 전체 제목입니다. 첨부 논문의 참고문헌 전체를 다시 나열한 목록은 아닙니다.

1. **Wang et al. — “CARD: Channel Aligned Robust Blend Transformer for Time Series Forecasting.”** ICLR 2024. 첨부 PDF, 본문 및 Appendix A–S. ICLR-2024-card-channel-aligned-…  
2. **Oreshkin et al. — “N-BEATS: Neural basis expansion analysis for interpretable time series forecasting.”** ICLR 2020, OpenReview. [OpenReview](https://openreview.net/forum?id=r1ecqn4YwB)  
3. **Zhou et al. — “Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting.”** AAAI 2021, AAAI 공식 출판물. [AAAI Publications](https://ojs.aaai.org/index.php/AAAI/article/view/17325)  
4. **Wu et al. — “Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting.”** NeurIPS 2021, 공식 proceedings. [NeurIPS](https://proceedings.neurips.cc/paper/2021/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html)  
5. **Zhou et al. — “FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting.”** ICML 2022, PMLR. [Proceedings of Machine Learning Research](https://proceedings.mlr.press/v162/zhou22g.html)  
6. **Kim et al. — “Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift.”** ICLR 2022, OpenReview. [OpenReview](https://openreview.net/forum?id=cGDAkQo1C0p)  
7. **Zeng et al. — “Are Transformers Effective for Time Series Forecasting?”** AAAI 2023, AAAI 공식 출판물. [AAAI Publications](https://ojs.aaai.org/index.php/AAAI/article/view/26317)  
8. **Nie et al. — “A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.”** ICLR 2023, OpenReview·arXiv. [arXiv](https://arxiv.org/abs/2211.14730)  
9. **Zhang and Yan — “Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting.”** ICLR 2023, OpenReview. [OpenReview](https://openreview.net/forum?id=vSVLM2j9eie)  
10. **Han, Ye, and Zhan — “The Capacity and Robustness Trade-off: Revisiting the Channel Independent Strategy for Multivariate Time Series Forecasting.”** 2023 사전공개, IEEE TKDE 2024. [arXiv](https://arxiv.org/abs/2304.05206)  
11. **Liu et al. — “iTransformer: Inverted Transformers Are Effective for Time Series Forecasting.”** ICLR 2024, 원문 PDF. [ICLR Proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/file/2ea18fdc667e0ef2ad82b2b4d65147ad-Paper-Conference.pdf)  
12. **Ilbert et al. — “SAMformer: Unlocking the Potential of Transformers in Time Series Forecasting with Sharpness-Aware Minimization and Channel-Wise Attention.”** ICML 2024, 원문 PDF. [arXiv](https://arxiv.org/pdf/2402.10198)  
13. **Wang et al. — “TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting.”** ICLR 2024, 원문 PDF. [ICLR Proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/file/a7ac8a21e5a27e7ab31a5f42a0117bdb-Paper-Conference.pdf)  
14. **Wang et al. — “TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables.”** NeurIPS 2024, arXiv 원문. [arXiv](https://arxiv.org/html/2402.19072v3)  
15. **Woo et al. — “Unified Training of Universal Time Series Forecasting Transformers.”** ICML 2024, Moirai 원문. [arXiv](https://arxiv.org/pdf/2402.02592)  
16. **Hu et al. — “TimeFilter: Patch-Specific Spatial-Temporal Graph Filtration for Time Series Forecasting.”** ICML 2025, PMLR. [Proceedings of Machine Learning Research](https://proceedings.mlr.press/v267/hu25ac.html)  
17. **Ansari et al. — “Chronos-2: From Univariate to Universal Forecasting.”** 2025, arXiv 기술보고서. [arXiv](https://arxiv.org/pdf/2510.15821)  
18. **Jang et al. — “Towards Robust Real-World Multivariate Time Series Forecasting: A Unified Framework for Dependency, Asynchrony, and Missingness.”** ICLR 2026, ChannelTokenFormer 공식 proceedings PDF. [ICLR Proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/file/f924dab7a06d0463fe238904678acaee-Paper-Conference.pdf)  
19. **Qiu et al. — “TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods.”** PVLDB 2024, 원문·arXiv. [arXiv](https://arxiv.org/abs/2403.20150)  
20. **Aksu et al. — “GIFT-Eval: A Benchmark For General Time Series Forecasting Model Evaluation.”** 2024 사전공개, arXiv 확인판. [arXiv](https://arxiv.org/abs/2410.10393)
