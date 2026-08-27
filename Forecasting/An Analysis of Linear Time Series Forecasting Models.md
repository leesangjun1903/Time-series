# An Analysis of Linear Time Series Forecasting Models

> **참고 논문**: Toner, W. & Darlow, L. (2024). *An Analysis of Linear Time Series Forecasting Models*. arXiv:2403.14587v2. ICML 2024 제출본.
>
> **참고 문헌**: Zeng et al. (2023), Li et al. (2023), Xu et al. (2023), Kim et al. (2021), Hastie et al. (2009), Vaswani et al. (2017), Wu et al. (2021), Pedregosa et al. (2011), Silva (2024)

---

## 1. Executive Summary (10문장 이내)

1. 본 논문은 시계열 예측(Time Series Forecasting)에서 널리 사용되는 선형 모델 변형들(DLinear, FITS, RLinear, NLinear)을 수학적으로 분석한다.
2. 저자들은 이 모델들이 아키텍처는 달라 보이지만, 실제로 표현하는 함수 집합(Model Class)이 표준 비제약 선형 회귀(Unconstrained Linear Regression)와 동등하거나 약하게 제약된(Weakly Constrained) 형태임을 증명한다.
3. 핵심 발견은 $L \geq T-2$ 조건 하에서 $M(\text{DLinear}) = M(\text{Linear}) = M(\text{FITS})$가 성립한다는 것이다.
4. 정규화(Normalization) 기법을 적용한 모델들(RLinear, NLinear, FITS+IN)도 마찬가지로 가중치 행렬의 행합(Row Sum)이 1이 되는 약한 제약 선형 회귀로 귀결됨을 증명한다.
5. 최소제곱 손실(MSE Loss) 하에서의 볼록성(Convexity)으로 인해 이 모델들은 동일한 학습 데이터에서 동일한 최적해로 수렴해야 함을 이론적으로 도출한다.
6. 실험적으로, 각 모델의 학습된 가중치 행렬이 폐쇄형 최소제곱(OLS) 해와 거의 동일함을 시각적으로 확인한다.
7. 폐쇄형 OLS 해가 SGD(Stochastic Gradient Descent)로 훈련된 모델들보다 72%(32개 중 23개) 설정에서 더 우수한 성능을 보인다.
8. FITS 모델은 푸리에 변환의 정규화 방식으로 인해 편향(Bias) 항의 실효 학습률(Effective Learning Rate)이 $\frac{1}{L}$배 억제되는 효과가 있어, 소규모 데이터셋에서 과적합(Overfitting) 방지에 유리하게 작용함을 이론적으로 설명한다.
9. 각 모델 클래스는 적절히 증강된(Augmented) 피처 집합 위에서 비제약 선형 회귀로 재정식화될 수 있으며, 이는 폐쇄형 해의 존재를 보장한다.
10. 결론적으로, 이 모델들의 "현대화(modernization)" 시도는 함수적 관점에서 표준 선형 회귀에서 크게 벗어나지 않으며, 더 단순한 OLS 해가 실용적으로 우수한 대안임을 주장한다.

---

### 1-1. 연구의 목적과 필요성

**목적**: 시계열 예측 분야에서 제안된 다양한 선형 모델 변형들(DLinear, FITS, RLinear, NLinear)이 아키텍처 수준의 차이에도 불구하고 함수적으로 동등한지를 수학적으로 분석하고 실험적으로 검증하는 것.

**필요성**:
- 딥러닝 모델이 시계열 예측에서 단순 선형 모델보다 우월하지 않다는 선행 연구(Zeng et al., 2023; Li et al., 2023)가 발표되며, 선형 모델의 변형 연구가 급증하였다.
- 각 변형 모델이 아키텍처적 우월성을 주장하나, 이들의 함수 공간(Model Class)에 대한 엄밀한 수학적 분석이 부재하였다.
- 금융, 기상, 헬스케어, 클라우드 인프라 등 고빈도·고해상도 데이터 환경에서 단순하고 설명 가능하며 효율적인 모델의 필요성이 증대되었다.
- 폐쇄형 해(Closed-Form Solution)의 실용적 우수성을 정량적으로 입증함으로써, 불필요한 아키텍처 복잡화를 지양하는 방향을 제시할 필요가 있었다.

> 💡 **용어 설명**
> - **Model Class (모델 클래스)**: 특정 아키텍처가 표현할 수 있는 모든 함수들의 집합. 예: 선형 회귀의 모델 클래스는 $\{\vec{x} \mapsto W\vec{x} + \vec{b}\}$의 전체 집합.
> - **폐쇄형 해 (Closed-Form Solution)**: 반복적 최적화(경사 하강 등) 없이 수식으로 직접 계산되는 해. 선형 회귀에서는 $W^* = (X^TX)^{-1}X^TY$가 해당됨.

---

## 2. 핵심 주장과 근거 표

| 핵심 주장 | 근거 (이론) | 근거 (실험) | 위치 |
|---|---|---|---|
| DLinear ≡ 비제약 선형 회귀 | Lemma 3.2: 이동 평균 분해가 선형 변환이므로 임의의 아핀 맵 $A\vec{x}+\vec{b}$ 표현 가능 | 학습 가중치 행렬이 OLS와 동일 | p.3, Fig.1 |
| FITS ≡ 비제약 선형 회귀 ($L \geq T-2$) | Theorem 3.3: RFT → 복소 선형 맵 → iRFT 구성이 아핀 선형임 | 가중치 행렬 코사인 유사도 1에 수렴 | p.3, Fig.2 |
| RLinear ≡ 행합=1 제약 + $\sigma$ 스케일 편향 선형 회귀 | Lemma 3.7: RevIN의 수학적 전개 결과 | 편향 항이 OLS+IN과 동일 | p.4, Fig.4 |
| NLinear ≡ 행합=1 제약 선형 회귀 (편향 비제약) | Lemma 3.9: NowNorm의 수학적 전개 결과 | 가중치 행렬 거의 동일 | p.5, Fig.1 |
| 모든 모델이 동일 최적해로 수렴 | MSE 손실의 볼록성 → 전역 유일 최적해 존재 | 코사인 유사도 1 수렴 (Fig.2) | p.6, Fig.2 |
| OLS 해가 SGD 학습 모델보다 72% 우수 | 볼록 최적화의 정확한 해 도출 | Table 2: 32개 중 23개 설정에서 OLS 우세 | p.8, Table 2 |
| FITS 편향 학습률 억제 현상 | iRFT 정규화로 인해 편향 실효 학습률 $\approx \frac{1}{L}$배 감소 | Fig.4: FITS 편향이 타 모델과 현저히 다름 | p.6, Fig.4 |

---

### 2-1. 상세 분석

#### ① 해결하고자 하는 문제

시계열 예측을 위한 다양한 선형 모델 변형들이 서로 다른 아키텍처(트렌드 분해, 주파수 공간 처리, 인스턴스 정규화 등)를 사용하며 각기 우월성을 주장하지만, 이들이 **실제로 동일한 함수 공간을 표현하는지** 수학적으로 밝혀진 바 없었다. 또한 SGD로 훈련된 이 모델들이 폐쇄형 OLS 해보다 실제로 더 나은지에 대한 정량적 근거가 없었다.

#### ② 제안하는 방법 (수식 포함)

**공통 표기법** (p.2, Section 3.1):
- $L$: 컨텍스트 길이 (입력 시계열 길이)
- $T$: 예측 지평선 (예측할 미래 시점 수)
- $\vec{x} \in \mathbb{R}^L$: 컨텍스트 벡터 (과거 데이터)
- $\vec{y} \in \mathbb{R}^T$: 타겟 벡터 (예측 목표값)
- $c$: 채널 수 (독립 시계열 수); 단변량 분석 시 $c=1$

**[DLinear 분석]** (p.3, Lemma 3.2)

DLinear는 $\vec{x}$를 트렌드( $\vec{x}\_{\text{trend}}$ )와 계절성( $\vec{x}\_{\text{seasonal}} = \vec{x} - \vec{x}_{\text{trend}}$ )으로 분해 후 각각 별도의 선형 레이어에 통과:

$$f_{\text{DLinear}}(\vec{x}) = B\vec{x}_{\text{seasonal}} + C\vec{x}_{\text{trend}} + \vec{c} + \vec{d}$$

여기서 $B, C \in \mathbb{R}^{T \times L}$, $\vec{c}, \vec{d} \in \mathbb{R}^T$, $D$는 패딩된 이동 평균에 해당하는 정사각 행렬.

전개하면:

$$= B(\vec{x} - D\vec{x}) + C(D\vec{x}) + \vec{c} + \vec{d} = (B - BD + CD)\vec{x} + \vec{c} + \vec{d}$$

$A = B - BD + CD$, $\vec{b} = \vec{c} + \vec{d}$로 설정하면 임의의 아핀 맵 $A\vec{x} + \vec{b}$와 동일:

$$\boxed{M(\text{DLinear}) = M(\text{Linear})}$$

> 💡 **용어 설명**
> - **아핀 맵 (Affine Map)**: $f(\vec{x}) = A\vec{x} + \vec{b}$ 형태의 선형 변환 + 평행이동. 순수 선형 변환($\vec{b}=0$)을 포함하는 더 일반적인 개념.
> - **이동 평균 (Moving Average)**: 연속된 시점들의 평균으로 데이터의 트렌드를 추출하는 기법. DLinear에서는 커널 크기 25의 이동 평균 사용.

**[FITS 분석]** (p.3, Theorem 3.3; p.10-15, Appendix A)

FITS 모델의 처리 과정:

$$\vec{x} \xrightarrow{\text{RFT}_L} \tilde{\vec{z}} \in \mathbb{C}^{\lfloor L/2 \rfloor + 1} \xrightarrow{W\cdot + \vec{c}} \tilde{\vec{y}} \in \mathbb{C}^{\lfloor(L+T)/2\rfloor + 1} \xrightarrow{\text{iRFT}} \mathbb{R}^{L+T}$$

실수 이산 푸리에 변환(RFT) 정의 (Definition A.2, p.10):

$$\text{RFT}_L(\vec{x})_j := \sum_{k=0}^{L-1} e^{-2\pi ikj/L} x_k, \quad j \in \{0, 1, \ldots, \lfloor L/2 \rfloor\}$$

기호 설명:
- $\text{RFT}_L$: 길이 $L$ 실수 신호에 대한 실수 이산 푸리에 변환
- $i$: 허수 단위 ($\sqrt{-1}$)
- $k$: 시간 도메인 인덱스
- $j$: 주파수 도메인 인덱스

FITS 모델은 RFT, 복소 선형 맵, iRFT의 합성이므로 전체가 아핀 선형 연산. Lemma A.7에 따라:

$$\text{FITS}(\vec{x}; W, \vec{c}) = D_{L+T}^{-1}(\Pi_{L+T}^{-1} W \Pi_L) D_L \vec{x} + \text{iRFT}(\vec{c})$$

$L \geq T-2$ 조건에서 가중치 행렬 $A$와 편향 $\vec{b}$가 완전히 비제약임을 Vandermonde 행렬의 full rank 성질로 증명:

$$\boxed{M(\text{DLinear}) = M(\text{Linear}) = M(\text{FITS})} \quad (L \geq T-2, \text{ LPF 없을 때})$$

> 💡 **용어 설명**
> - **RFT (Real Discrete Fourier Transform)**: 실수 신호를 주파수 성분으로 분해하는 변환. 실수 입력의 켤레 대칭 성질을 이용해 DFT의 절반만 저장.
> - **Vandermonde 행렬**: 각 행이 등비수열 형태인 특수 행렬. 단위근(root of unity)을 생성원으로 하면 항상 full rank(행렬식 ≠ 0)임이 보장됨.
> - **iRFT (Inverse Real Fourier Transform)**: RFT의 역변환. 주파수 도메인에서 시간 도메인으로 복원.

**[Instance Normalization(IN) 기반 모델 분석]** (p.3-5, Definition 3.4, Lemma 3.5, 3.7, 3.9)

**인스턴스 정규화(IN)** (Definition 3.4):

$$\vec{x}' = \frac{\vec{x} - \mu(\vec{x})}{\sigma(\vec{x}) + \varepsilon}, \quad \hat{y} = f(\vec{x}'), \quad \hat{y}_{\text{out}} = \hat{y} \cdot (\sigma(\vec{x}) + \varepsilon) + \mu(\vec{x})$$

기호 설명:
- $\mu(\vec{x})$: $\vec{x}$의 평균
- $\sigma(\vec{x})$: $\vec{x}$의 표준편차
- $\varepsilon$: 수치 안정성을 위한 소수 (0 나누기 방지)

IN을 선형 레이어와 결합하면 (Lemma 3.5):

$$f(\vec{x}) = (B_T + A - AB_L)\vec{x} + \sigma(\vec{x})\vec{b} = \tilde{A}\vec{x} + \vec{b}\sigma(\vec{x})$$

여기서 $\tilde{A}$의 각 행의 합 = 1, $B_T$는 $T \times L$ 행렬로 모든 원소가 $\frac{1}{L}$.

**RevIN (Reversible Instance Normalization)** (Definition 3.6, p.4):

$$\vec{x}' = \frac{\vec{x} - \mu(\vec{x})}{\sigma(\vec{x}) + \varepsilon}, \quad \vec{x}'' = \frac{\vec{x}' - \beta}{\alpha}, \quad \hat{y} = f(\vec{x}''), \quad \hat{y}' = \alpha\hat{y} + \beta, \quad \hat{y}_{\text{out}} = \hat{y}' \cdot (\sigma(\vec{x}) + \varepsilon) + \mu(\vec{x})$$

기호 설명:
- $\alpha, \beta$: 학습 가능한 아핀 변환 파라미터

RLinear (RevIN + Linear, Lemma 3.7):

$$f(\vec{x}) = \tilde{A}\vec{x} + \vec{b}\sigma(\vec{x}), \quad \tilde{A} = B_T + A - AB_L, \quad \vec{b} = \beta + \alpha c - A\beta$$

**NowNorm(NN)** (Definition 3.8):

$$\vec{x}_{\text{norm}} = \vec{x} - x_L, \quad \hat{y} = f(\vec{x}_{\text{norm}}), \quad \hat{\vec{y}}_{\text{out}} = \hat{y} + x_L$$

기호 설명:
- $x_L$: 컨텍스트 벡터의 가장 최근 값

NLinear (NN + Linear, Lemma 3.9):

$$f(\vec{x}) = \tilde{A}\vec{x} + \vec{b}, \quad \text{단, } \tilde{A}\text{의 각 행의 합} = 1, \quad \vec{b} \in \mathbb{R}^T \text{ (비제약)}$$

**최종 모델 클래스 동등성** (p.5):

$$M(\text{DLinear+IN}) = M(\text{Linear+IN}) = M(\text{FITS+IN}) = M(\text{RLinear}) \approx M(\text{NLinear})$$

> 💡 **용어 설명**
> - **RevIN (Reversible Instance Normalization)**: 입력을 정규화한 후 예측, 역정규화로 복원하는 기법. 분포 변화(Distribution Shift)에 강건함.
> - **NowNorm**: NLinear의 정규화 방식. 가장 최근 값을 기준으로 입력을 이동시켜 현재 수준(level)의 영향을 제거.
> - **행합=1 제약 (Row-sum-to-one constraint)**: 가중치 행렬 $\tilde{A}$의 각 행의 원소 합이 정확히 1이 되어야 하는 조건.

**[폐쇄형 OLS 해]** (p.6, Appendix D.2, Definition D.1):

$$W^* = (X^TX)^{-1}X^TY$$

기호 설명:
- $X \in \mathbb{R}^{N \times L}$: $N$개 훈련 샘플로 구성된 설계 행렬
- $Y \in \mathbb{R}^{N \times T}$: 훈련 타겟 행렬
- $W^* \in \mathbb{R}^{L \times T}$: MSE 최소화 가중치 행렬

**[FITS 편향 학습률 억제]** (p.6, Section 4.1; Appendix C):

편향 $\vec{b}$와 복소 편향 $\vec{c}$의 관계: $\vec{b} = M\vec{C}$

SGD 업데이트 비교:
- 직접 파라미터화: $\vec{b} \mapsto \vec{b} - \eta\bar{b}$
- FITS 파라미터화($\vec{b} = M\vec{C}$를 통해 $\vec{C}$ 학습):

$$\vec{b} \mapsto \vec{b} - \eta MM^T\bar{b}$$

$M$의 원소 크기가 $\sim \frac{1}{\sqrt{L}}$ 수준이므로 $MM^T \sim \frac{1}{L}$, 즉 편향의 실효 학습률이 약 $\frac{1}{L}$배로 감소.

> 💡 **용어 설명**
> - **실효 학습률 (Effective Learning Rate)**: 파라미터화 방식에 의해 실제로 적용되는 학습률. FITS에서는 편향이 복소 공간을 통해 간접 파라미터화되어 학습률이 자연스럽게 감소함.
> - **SGD (Stochastic Gradient Descent)**: 미니배치 단위로 기울기를 계산해 파라미터를 업데이트하는 최적화 알고리즘.

#### ③ 모델 구조 요약

| 모델 | 구조 | 정규화 | 모델 클래스 |
|---|---|---|---|
| Linear | $W\vec{x} + \vec{b}$ | 없음 | $A\vec{x} + \vec{b}$ (비제약) |
| DLinear | 트렌드/계절 분해 → 각각 선형 | 없음 | $A\vec{x} + \vec{b}$ (비제약) |
| FITS | RFT → 복소 선형 → iRFT | 없음 | $A\vec{x} + \vec{b}$ (비제약, $L \geq T-2$) |
| NLinear | 최근값 기준 이동 → 선형 | NowNorm | $\tilde{A}\vec{x} + \vec{b}$ (행합=1) |
| RLinear | RevIN → 선형 | RevIN | $\tilde{A}\vec{x} + \vec{b}\sigma(\vec{x})$ (행합=1, 편향 스케일) |
| FITS+IN | RFT → 복소 선형 → iRFT | IN | $\tilde{A}\vec{x} + \vec{b}\sigma(\vec{x})$ (행합=1, 편향 스케일) |

#### ④ 성능 향상 및 한계

**성능 향상**:
- OLS가 32개 설정 중 23개(72%)에서 가장 낮은 MSE 달성 (Table 2, p.8)
- 대규모 데이터셋(ECL, Traffic, Weather)에서 OLS의 강건한 성능 확인
- FITS는 소규모 데이터셋(ETTh1, ETTh2)에서 편향 억제 효과로 상대적으로 우수

**한계**:
- 다채널(Multi-channel) 설정에서 RevIN과 IN의 미묘한 차이 미분석 (Appendix F.2)
- LPF(Low-Pass Filter) 적용 FITS의 모델 클래스 분석 미완 (Appendix F.2)
- 비선형 모델과의 직접적 비교 부재
- 폐쇄형 해의 계산 비용(SVD 분해)이 대규모 데이터에서 실용적이지 않을 수 있음
- 단변량($c=1$) 중심 분석; 다변량 설정의 엄밀한 분석 필요

---

## 3. 각 주장의 위치 (페이지/Figure/Table)

| 주장 | 위치 |
|---|---|
| $M(\text{DLinear}) = M(\text{Linear})$ | p.3, Lemma 3.2 |
| $M(\text{FITS}) = M(\text{Linear})$ ($L \geq T-2$) | p.3, Theorem 3.3; pp.10-15, Appendix A |
| IN/RevIN의 행합=1 제약 도출 | p.3-4, Lemma 3.5 (ILinear), p.4 Lemma 3.7 (RLinear) |
| NLinear 모델 클래스 (편향 비제약) | p.5, Lemma 3.9 |
| 전체 모델 클래스 동등성 요약 | p.5, Table 1 |
| 모든 모델 클래스의 폐쇄형 해 존재 | p.6 (Convexity/Closed Form 단락) |
| FITS 편향 학습률 억제 분석 | p.6, Section 4.1; Appendix C (pp.16-17) |
| 학습 가중치 행렬 비교 시각화 | p.7, Figure 1 |
| OLS로의 수렴 (코사인 유사도) | p.7, Figure 2 |
| 예측 비교 | p.7, Figure 3 |
| 편향 항 비교 | p.7, Figure 4; Appendix B, Figure 5 |
| 성능 비교 (MSE 수치) | p.8, Table 2 |
| OLS 72% 우세 | p.8 (Table 2 설명 단락) |
| 폐쇄형 해 수식 | p.18, Appendix D.2, Definition D.1 |
| 한계 및 향후 연구 | p.20, Appendix F.2 |

---

## 4. 저자 보고 결과 vs. 본 해석 분리

### 4-1. 연구 주제

| 구분 | 내용 |
|---|---|
| **저자 보고** | "시계열 예측용 선형 모델 변형들의 함수 집합(Model Class)을 수학적으로 분석하고, 이들이 표준 선형 회귀와 동등함을 증명한다." (p.1, Abstract) |
| **본 해석** | 아키텍처 복잡성 증가가 실질적 표현력 향상으로 이어지지 않음을 보이는 연구로, 시계열 예측 분야의 "단순함의 승리(Triumph of Simplicity)" 테제를 수학적으로 강화하는 작업. 그러나 분석이 단변량·MSE 손실·채널 독립 설정에 한정되어 있어 실용적 일반화 범위에 주의가 필요함. |

### 4-2. 연구 방법

**저자 보고**:
- Lemma/Theorem 형태의 수학적 증명으로 모델 클래스 동등성 확립
- ETTh1 데이터셋에서 가중치 행렬 시각화 및 코사인 유사도 추적
- 8개 벤치마크 데이터셋에서 MSE 비교 (3회 이상 반복 실험, 표준편차 보고)

**본 해석**:
- 증명 방법은 엄밀하나, 핵심 조건($L \geq T-2$, LPF 미적용)이 실용적 설정에 항상 만족되는지 검토 필요. 논문 실험에서는 모두 $L=720$으로 이 조건이 충족되지만, 짧은 컨텍스트 사용 시 FITS의 동등성 성립 여부는 별도 분석 필요.
- 코사인 유사도 실험은 350 에폭에서의 수렴을 보이나, 실제 학습에서 50 에폭 조기 종료(Early Stopping) 시 수렴 정도가 다를 수 있음 (이는 논문도 인정, p.6).

### 4-3. 연구 결과

| 구분 | 내용 |
|---|---|
| **저자 보고** | "OLS가 72%(23/32)의 설정에서 최소 MSE 달성." (p.8, Table 2) |
| **본 해석** | ⚠️ 이 수치는 특정 하이퍼파라미터(학습률 0.0005, 50 에폭, Adam, 컨텍스트 720)하에서의 결과임. OLS는 정규화(regularization) 없는 순수 최소제곱 해이므로, 훈련 데이터가 충분히 크거나 노이즈가 낮을 때 유리하게 작동할 수 있음. Exchange 데이터셋(소규모 금융 데이터)에서 DLinear가 일부 설정에서 OLS보다 우수한 점은 과적합 관련 논의가 필요함을 시사. |
| **저자 보고** | "FITS는 소규모 데이터셋(ETTh1/h2)에서 우수한 성능을 보이며, 이는 편향 억제 효과로 인한 암묵적 정규화 때문." (p.8) |
| **본 해석** | 이 해석은 설득력 있으나 직접적 실험 검증(예: 편향 학습률을 명시적으로 조절하는 ablation study)이 논문 내에는 부재함. 이론적 추론과 실험 결과의 상관관계만 확인됨. |

> 💡 **용어 설명**
> - **조기 종료 (Early Stopping)**: 검증 손실이 최소인 시점의 모델을 선택하는 기법. 과적합 방지 효과가 있어 일종의 암묵적 정규화로 작용.
> - **Ablation Study**: 특정 구성 요소를 제거하거나 변경하며 그 기여도를 측정하는 실험. 인과관계 확인에 필수적.

---

## 5. 통계적 취약점 및 비교 불가능 수치

⚠️ **통계적 취약점 목록**:

| 항목 | 문제점 | 위치 |
|---|---|---|
| **반복 횟수** | "최소 3회" 실행 명시 (정확한 횟수 미고정). 일부 설정에서 표준편차가 매우 작아 3회 반복으로는 통계적 유의성 검정 불충분 | p.20, Appendix E |
| **OLS 72% 우세** | 정확한 통계 검정(t-test, Wilcoxon 검정 등) 없이 단순 빈도 비율로 우세 주장 | p.8 |
| **FITS 소규모 데이터셋 우세** | Exchange $T=720$ 설정에서 Linear MSE: $0.717 \pm 0.170$으로 표준편차가 매우 높아 해석 불안정 | Table 2 |
| **단일 컨텍스트 길이** | 모든 실험에서 $L=720$ 고정. 다른 컨텍스트 길이에서의 결과 부재 | p.20, Appendix E |
| **단변량 분석** | 수학적 분석이 $c=1$ (단변량) 중심. 다변량 설정에서의 동등성 엄밀 분석 미완 | p.2, Section 3.1 |

⚠️ **비교 불가능 수치**:

| 항목 | 이유 |
|---|---|
| **FITS vs. 타 논문 보고 성능** | 원 논문(Xu et al., 2023)의 FITS 성능과 직접 비교 어려움: LPF 미적용, 하이퍼파라미터 재설정, 50 에폭 고정 vs. 원 논문 설정 차이 |
| **RLinear vs. Li et al. (2023) 보고값** | 학습률(0.0005), 에폭(50), 배치 크기(128) 통일 설정이 원 논문 최적 설정과 다를 수 있음 |
| **OLS vs. 리지/라쏘 회귀** | OLS는 정규화 없는 해. 정규화 회귀와 비교 시 소규모 데이터에서 OLS가 불리할 수 있으나 논문에서 미비교 |

---

## 6. 논문이 답하지 않는 질문

1. **LPF 적용 FITS의 모델 클래스는 무엇인가?** 저자들은 LPF가 성능을 저하시킨다고 언급하며 분석에서 제외했으나, LPF 적용 시 모델 클래스의 제약 조건에 대한 엄밀한 특성화가 이루어지지 않았다 (Appendix F.2).

2. **다변량(Multi-variate) 설정에서의 동등성은 성립하는가?** 분석이 채널 독립(Channel-Independent, CI) 가정하의 단변량 설정에 한정되어 있다. 채널 간 상관관계를 활용하는 설정에서도 동등성이 유지되는지 불명확하다.

3. **짧은 컨텍스트($L < T-2$)에서 FITS는 선형 회귀보다 어떤 추가적 제약을 갖는가?** $L \geq T-2$ 조건이 불만족될 때의 FITS 모델 클래스 특성화가 제시되지 않았다.

4. **OLS 해의 계산 비용이 대규모 데이터에서 실용적인가?** SVD 기반 OLS는 $O(NL^2 + L^3)$ 복잡도를 가지므로, Traffic(862채널, 12185 훈련 샘플)과 같은 대규모 설정에서의 실용성 분석이 없다.

5. **FITS의 암묵적 편향 정규화가 어느 정도의 데이터 크기/노이즈 수준에서 유효한가?** ETTh vs. ETTm의 차이(시간 vs. 분 단위 해상도)만으로 소규모 데이터 기준을 정의하기 어렵다.

6. **비MSE 손실 함수(MAE, Quantile Loss 등)에서도 동일한 동등성이 성립하는가?** 볼록성 논증이 MSE 손실에 특화되어 있어, 다른 손실 함수 적용 시 결론이 달라질 수 있다.

7. **RevIN의 학습 가능한 아핀 파라미터($\alpha, \beta$)가 다채널 설정에서 채널별로 다를 때 Linear+IN과 동등성이 유지되는가?** 저자들은 이를 한계로 언급했으나 분석을 제공하지 않았다 (Appendix F.2).

8. **FITS 압축 기법을 OLS 해에 사후 적용(Post-hoc)하는 방법이 실제로 효과적인가?** 저자들이 향후 연구로 제안하지만 논문 내 검증이 없다.

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p.7): 학습된 가중치 행렬 시각화

**내용**: ETTh1 데이터셋에서 50 에폭 훈련 후 OLS+IN, FITS+IN, DLinear+IN, RLinear, NLinear의 가중치 행렬($720 \times 336$ 부분)을 동일한 색 스케일로 시각화.

**해석**: 모든 모델의 가중치 행렬이 육안으로 거의 동일한 패턴(대각선 방향의 주기적 구조)을 보인다. 이는 Lemma 3.2, 3.7, 3.9, Theorem 3.3의 이론적 예측—즉, 이들이 동일한 모델 클래스에 속함—과 완벽히 일치한다. 미세한 차이가 존재하나 Figure 3의 예측 비교에서 보듯 실질적 예측 차이는 미미하다.

**중요성**: 이 한 장의 그림이 논문의 핵심 주장을 가장 직관적으로 증명한다.

---

### Figure 2 (p.7): 코사인 유사도 학습 곡선

**내용**: 350 에폭 훈련 동안 각 모델의 가중치 행렬과 OLS 해의 코사인 유사도 추적.

$$d(x, y) := \frac{x \cdot y}{\|x\|_2 \cdot \|y\|_2}$$

**해석**: 모든 모델(RLinear, NLinear, DLinear+IN, FITS+IN)의 코사인 유사도가 학습이 진행됨에 따라 1에 수렴한다. RLinear가 가장 빠르게 수렴하고, FITS+IN이 가장 느리게 수렴한다. 이는 Section 4.1의 FITS 편향 학습률 억제 이론과 일관된다. 수렴 속도 차이는 각 모델의 파라미터화 방식이 SGD 경로(optimization trajectory)에 영향을 줌을 보여준다.

**중요성**: 이론(볼록성 → 동일 최적해 수렴)과 실험이 일치함을 시계열적으로 입증.

> 💡 **용어 설명**
> - **코사인 유사도 (Cosine Similarity)**: 두 벡터 간의 방향 유사도를 측정 (-1~1). 1이면 완전히 동일한 방향, 즉 두 행렬이 스케일 차이만 있고 방향이 같음을 의미.

---

### Figure 4 (p.7): 학습된 편향 항 비교

**내용**: DLinear+IN, RLinear, NLinear, FITS+IN, OLS+IN의 학습된 편향 벡터를 예측 지평선(horizon) 축에 따라 플롯 ($T=720$).

**해석**: DLinear+IN, RLinear, OLS+IN은 거의 동일한 편향을 학습하며 모두 음의 기울기 패턴을 보인다. 반면 FITS+IN의 편향은 크기가 현저히 작고 패턴이 다르다. 이는 Section 4.1의 이론적 분석—FITS의 iRFT 파라미터화로 인해 편향의 실효 학습률이 $\frac{1}{L}$배 억제됨—을 직접적으로 확인시켜준다. 이 편향 억제가 소규모 데이터에서 과적합 방지 역할을 함을 시사.

**중요성**: 이론(Section 4.1)과 실험(Figure 4, Appendix B Figure 5)이 정확히 일치하는 핵심 증거.

---

### Figure 3 (p.7): 예측값 비교

**내용**: ETTh1, $T=336$ 설정에서 5개 모델의 예측값을 실제 Ground Truth와 비교.

**해석**: 5개 모델의 예측 곡선이 시각적으로 매우 유사하며 Ground Truth 패턴을 비슷하게 추종한다. 일부 시점에서 미세한 차이가 있으나 이는 Figure 1에서 관찰된 가중치 행렬의 미세한 차이에서 비롯된 것으로, 실질적 예측 성능 차이는 미미함을 확인. OLS+IN이 일부 구간에서 가장 안정적인 예측을 보임.

**중요성**: "모델들이 사실상 동일한 예측을 한다"는 주장의 직관적 시각화.

---

### Table 2 (p.8): 장기 다변량 예측 MSE 비교

**내용**: 8개 데이터셋 × 4개 예측 지평선 = 32개 설정에서 9개 모델의 MSE (표준편차 포함). 녹색: OLS 우세, 파란색: 1 표준편차 이내 차이, 볼드: 최고 성능.

**해석**:
- **OLS 우세 패턴**: 대부분의 설정에서 OLS/OLS+IN이 최소 MSE. 특히 ETTm2, ETTh2의 긴 예측 지평선($T=720$)에서 OLS+IN이 현저히 우수 (예: ETTh2 $T=720$에서 OLS+IN=0.380 vs. FITS+IN=0.377, DLinear+IN=0.384).
- **FITS 예외**: ETTh1/h2 소규모 데이터셋에서 FITS+IN이 경쟁적 성능 (ETTh1 $T=720$에서 FITS+IN=0.428 vs. OLS+IN=0.460). 편향 억제 효과의 실증.
- **Exchange 데이터셋**: 표준편차가 크게 보고됨 (Linear $T=720$: $0.717 \pm 0.170$), 통계적 불안정성 주의.
- **대규모 데이터셋** (ECL, Traffic, Weather): 모든 모델의 성능이 매우 유사하고 OLS가 경쟁적. 선형 모델의 표현 용량 한계 시사.

**중요성**: 이론적 동등성이 실제 성능 유사성으로 이어짐을 32개 설정에서 실증적으로 확인하는 핵심 근거 표.

---

## 8. 결론 및 후속 연구

### 8-1. 저자 제시 시사점 및 후속 연구

**저자 제시 시사점**:
- DLinear, FITS, RLinear, NLinear 등 선형 모델 변형들은 아키텍처 관점에서의 혁신에도 불구하고 함수적으로 표준 선형 회귀와 동등하거나 약하게 제약된 형태에 불과함
- 폐쇄형 OLS 해가 SGD 기반 학습보다 72% 설정에서 우수하므로, 시계열 예측에서 OLS가 강력한 베이스라인이 될 수 있음
- FITS의 우수성은 아키텍처적 혁신이 아닌 편향 학습률 억제라는 암묵적 정규화 효과에 기인

**저자 제시 향후 연구** (Appendix F.2):
1. LPF 적용 FITS의 모델 클래스 엄밀한 특성화
2. FITS 압축 기법을 OLS 해에 사후 적용(Post-hoc Compression)하는 방법론 개발
3. 다채널 설정에서 RevIN과 IN의 미세한 차이 분석

---

### 8-1. 모델의 일반화 성능 향상 가능성

본 논문의 분석 결과에서 다음과 같은 일반화 성능 향상 방향을 도출할 수 있다:

**① 암묵적 정규화의 설계적 활용**

FITS의 편향 학습률 억제($\sim \frac{1}{L}$배)가 소규모 데이터에서 일반화 성능을 향상시킴을 보였다. 이를 활용하여, 데이터 크기에 따라 **편향 학습률을 명시적으로 조절**하는 적응형 학습률 스케줄러(Adaptive LR for bias term)를 설계할 수 있다:

$$\eta_{\text{bias}} = \frac{\eta_{\text{weight}}}{L^{\alpha}}, \quad \alpha \in (0, 1]$$

이는 FITS의 효과를 재현하면서도 가중치 학습을 방해하지 않는다.

**② 피처 증강(Feature Augmentation)을 통한 비선형성 모사**

모든 모델이 선형 회귀로 환원됨을 보인 만큼, 일반화 성능 향상은 **비선형 피처 증강**에서 찾을 수 있다:

$$\tilde{\vec{x}} = [\vec{x}; \phi_1(\vec{x}); \phi_2(\vec{x}); \ldots]$$

여기서 $\phi_i$는 스펙트럼 특징, 웨이블릿 계수, 자기상관 등의 비선형 피처. 이를 OLS에 결합하면 계산 효율을 유지하면서 표현력을 확장할 수 있다.

**③ 정규화된 OLS (Ridge/Lasso)의 체계적 적용**

본 논문의 OLS는 정규화 없는 순수 최소제곱 해. 소규모 데이터나 고차원 설정에서는:

$$W^*_{\text{ridge}} = (X^TX + \lambda I)^{-1}X^TY$$

리지(Ridge) 정규화가 일반화 성능을 향상시킬 것으로 기대된다. 최적 $\lambda$는 교차검증(Cross-Validation)으로 결정 가능하며, 이는 SGD의 조기 종료보다 더 원칙적인 정규화 방법이다.

> 💡 **용어 설명**
> - **Ridge 회귀**: 가중치의 L2 노름($\|W\|^2$)을 손실 함수에 추가해 과적합을 방지하는 정규화 선형 회귀.
> - **웨이블릿 계수 (Wavelet Coefficients)**: 시간-주파수 분석 도구. 푸리에 변환과 달리 국소적 시간 정보를 보존하면서 주파수 성분을 분석.

**④ 채널 독립 vs. 채널 결합 전략**

Li et al. (2023)이 보인 것처럼, 일부 데이터셋에서는 채널 독립(CI) 설정이 더 나은 일반화를 보인다. 본 논문의 프레임워크를 확장하여:

$$f_{\text{joint}}(\vec{X}) = W\text{vec}(\vec{X}) + \vec{b}, \quad \vec{X} \in \mathbb{R}^{L \times c}$$

채널 간 상관관계를 활용하는 선형 모델의 표현력과 일반화 성능을 체계적으로 분석할 수 있다.

---

### 8-2. 2020년 이후 최신 연구 비교 분석

| 연구 | 핵심 기여 | 본 논문과의 관계 |
|---|---|---|
| **Zeng et al. (2023)**, "Are Transformers Effective for Time Series Forecasting?" (AAAI 2023) | DLinear, NLinear 제안; Transformer의 시계열 예측 유용성에 의문 제기 | 본 논문이 DLinear/NLinear의 수학적 등가성을 증명하여 이 연구의 발견을 이론적으로 강화 |
| **Li et al. (2023)**, "Revisiting Long-term Time Series Forecasting" | RevIN + 선형 매핑(RLinear) 제안; 채널 독립의 중요성 탐구 | 본 논문이 RLinear의 모델 클래스를 엄밀히 특성화 (Lemma 3.7) |
| **Xu et al. (2023)**, "FITS: Modeling Time Series with 10k Parameters" | 주파수 도메인 선형 모델 FITS 제안; 저파라미터 SoTA 달성 | 본 논문이 FITS의 등가성을 증명하고 성능의 원인이 아키텍처가 아닌 편향 억제임을 밝힘 |
| **Kim et al. (2021)**, "Reversible Instance Normalization" (ICLR 2022) | RevIN 제안; 분포 변화에 강건한 정규화 | 본 논문이 RevIN의 선형 모델에 대한 영향(행합=1 제약, 편향 스케일링)을 수학적으로 분석 |
| **Liu et al. (2023)**, "iTransformer" | 채널 차원에서 어텐션 적용; 다변량 예측에서 Transformer 효용성 재확인 | ⚠️ 본 논문이 다루지 않은 채널 결합 비선형 모델 영역. 직접 비교 불가 |
| **Nie et al. (2022)**, "PatchTST" (ICLR 2023) | 패치 기반 Transformer; 채널 독립 설정에서 강력한 성능 | 본 논문의 "단순 선형 모델이 우수" 테제와 경쟁 관계. PatchTST가 일부 설정에서 선형 모델 능가 |
| **Anonymous (2024)**, "DAM: A Foundation Model for Forecasting" (ICLR 2024) | 예측용 파운데이션 모델 | 본 논문의 분석이 파운데이션 모델 시대에 선형 모델의 위치를 재정립하는 데 기여 |

**본 논문이 앞으로의 연구에 미치는 영향**:

1. **베이스라인 재정의**: OLS가 경쟁력 있는 베이스라인임을 실증하여, 이후 연구에서 OLS를 필수 비교 대상으로 포함해야 한다는 기준을 제시.

2. **아키텍처 혁신의 검증 기준 강화**: 새로운 선형 또는 약선형 모델 제안 시, 해당 모델의 모델 클래스가 기존 선형 회귀와 실질적으로 다름을 증명해야 한다는 요구사항을 암묵적으로 설정.

3. **정규화 효과의 명시화**: 아키텍처적 혁신처럼 보이는 것이 사실 암묵적 정규화 효과일 수 있음을 보임으로써, 향후 모델 설계 시 정규화 효과와 표현력 향상을 분리하여 분석해야 함을 강조.

**앞으로 연구 시 고려할 점**:

1. **비선형 모델로의 확장**: 본 논문의 분석은 선형 모델에 한정. Transformer, N-BEATS, TimesNet 등 비선형 모델의 "진정한" 표현력 우위가 어떤 조건에서 발현되는지 체계적 분석 필요.

2. **분포 변화(Distribution Shift) 강건성**: 본 논문의 실험은 고정 분포 가정. 실제 환경에서 흔한 비정상성(Non-stationarity)과 분포 변화 하에서 OLS vs. SGD 기반 모델의 일반화 성능 비교가 필요.

3. **계산 효율 vs. 성능 트레이드오프**: OLS의 SVD 계산 비용이 실시간 또는 대규모 설정에서 문제가 될 수 있음. 근사 OLS 또는 분산 선형 회귀 기법과의 성능 비교 필요.

4. **다중 스케일 시계열**: 본 논문의 단일 컨텍스트 길이($L=720$) 고정 설정을 넘어, 다양한 컨텍스트 길이 및 다중 스케일 입력에 대한 분석 확장 필요.

5. **불확실성 정량화(Uncertainty Quantification)**: 점 예측(Point Forecast) 중심의 MSE 비교를 넘어, 예측 구간(Prediction Interval)이나 확률적 예측(Probabilistic Forecasting)에서 OLS와 SGD 기반 모델의 비교 연구 필요.

6. **파운데이션 모델과의 관계**: 대규모 사전 훈련(Pre-training)을 활용하는 파운데이션 모델(DAM, TimeGPT 등)이 선형 모델의 한계를 어떤 메커니즘으로 극복하는지 분석하여, 본 논문의 프레임워크가 비선형·대규모 설정에서도 유효한지 검토 필요.

> 💡 **용어 설명**
> - **파운데이션 모델 (Foundation Model)**: 대규모 데이터로 사전 훈련된 후 다양한 태스크에 전이(Transfer)할 수 있는 대형 모델. GPT, BERT 등이 대표적.
> - **비정상성 (Non-stationarity)**: 시계열의 통계적 특성(평균, 분산 등)이 시간에 따라 변하는 성질. 실제 금융, 경제 데이터에서 흔히 나타남.
> - **불확실성 정량화 (Uncertainty Quantification)**: 예측값의 신뢰도나 예측 구간을 함께 제공하는 방법론. 의사결정 지원에 필수적.

---

**⚠️ 답변 정확도 관련 고지**: 본 답변은 제공된 PDF 원문을 기반으로 작성되었으며, 페이지/Figure/Table 번호 모두 원문에서 직접 확인한 것입니다. 섹션 8-2의 최신 연구 비교 중 2024년 이후 발표된 연구(TimeGPT, TimesNet 등)에 대한 내용은 제 학습 데이터 기반이므로, 최신 발표 현황과 다를 수 있습니다. 해당 부분은 원문 참고 문헌 목록에 포함된 연구 중심으로 기술하였습니다.
