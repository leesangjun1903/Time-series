# Distillation of Synthetic Data for Time Series Foundation Models
---

## 1. Executive Summary (10문장 이내)

본 논문은 Meta AI의 Niloy Biswas와 Noureddine El Karoui가 저술한 연구로, 시계열 파운데이션 모델(TSFM) 사전학습의 효율성을 향상시키는 **합성 데이터 증류(Synthetic Data Distillation, SDD)** 기법을 제안한다.  
기존 사전학습은 단일 실현 궤적(realized trajectory)을 목표값으로 사용하여 확률적 경사(stochastic gradient)에 불필요한 분산이 발생한다는 문제가 있다.  
SDD는 단일 실현값 대신 조건부 예측 분포(conditional forecast distribution) 전체를 손실 함수에 활용하여 이 분산을 제거한다.  
이는 라오-블랙웰화(Rao-Blackwellization)에 해당하며, 경사의 기댓값은 유지하면서 뢰브너 편순서(Loewner partial ordering) 하에서 공분산을 감소시킴이 이론적으로 증명된다.  
SDD는 가우시안 프로세스, 선형 가우시안 상태공간 모델, 선형 SDE, i.n.i.d. 시계열 등 다수의 합성 데이터 생성기에 대해 해석적으로(closed-form) 계산 가능하다.  
실험은 4M에서 2.5B 파라미터 규모의 Toto-2 아키텍처 5종에서 수행되었으며, SDD는 연속 패치 마스킹(CPM) 조건에서 Status Quo 대비 $1.62\times$ ~ $1.86\times$의 수렴 가속을 달성하였다.  
계산 비용 측면에서는 동일한 검증 손실 도달에 38 ~ 46% 적은 FLOPs를 소요한다.  
사전 데이터 생성 시 조건부 분포 통계량을 캐싱하면 학습 단계에서 SDD와 Status Quo의 연산 비용은 동일하다.  
교사 강제(teacher forcing) 조건에서는 효과가 $1.19\times$ ~ $1.33\times$로 줄어드는데, 이는 이미 많은 위치를 평균화하여 분산이 낮기 때문이다.  
SDD는 TSFM을 넘어 표 형식 파운데이션 모델(Tabular Foundation Models) 등 합성 데이터로 사전학습되는 다양한 구조화 데이터 모델로 확장 가능하다.

> **💡 용어 설명**
> - **시계열 파운데이션 모델(TSFM)**: 다양한 도메인의 시계열 데이터로 사전학습된 대규모 신경망으로, 제로샷 예측(별도 파인튜닝 없이 새로운 데이터에 바로 적용)이 가능한 모델
> - **제로샷 예측(Zero-shot Forecasting)**: 학습에 사용되지 않은 새로운 데이터에 대해 추가 훈련 없이 바로 예측하는 능력
> - **합성 데이터(Synthetic Data)**: 실제 관측값이 아닌 수학적 모델(예: 가우시안 프로세스)로 인공 생성한 데이터

---

### 1-1. 연구의 목적과 필요성

**배경**: 실세계 시계열 데이터는 텍스트에 비해 공개적으로 이용 가능한 양이 절대적으로 부족하다(p.1). 이에 따라 합성 데이터 생성 기법이 TSFM 사전학습의 핵심 구성요소로 자리잡았다.

**문제**: 기존 사전학습은 합성 궤적의 단일 실현 미래값 $y_{t+1:t+h}$를 목표값으로 사용하므로, 동일한 과거 $y_{0:t}$에서 출발하더라도 미래 실현값의 무작위성이 확률적 경사에 **불필요한 분산(noise)**을 추가한다.

**필요성**: 합성 데이터를 사용할 경우, 데이터 생성 과정 $\pi_\alpha$가 알려져 있으므로 조건부 예측 분포를 정확히 계산할 수 있다. 이를 활용하면 실현값의 샘플링 노이즈를 완전히 제거하여 학습을 가속할 수 있다.

---

## 2. 핵심 주장과 근거 표

| 핵심 주장 | 근거 | 위치 |
|---|---|---|
| SDD는 경사의 기댓값을 변경하지 않으면서 분산을 감소시킨다 | Proposition 1: Rao-Blackwellization 이론적 증명 | p.3-4, Eq.(4) |
| SDD는 Status Quo 대비 더 빠른 수렴을 달성한다 | 5종 Toto-2 모델에서 $1.62\times$~$1.86\times$ 가속, 38~46% FLOPs 절감 | p.4, Figure 2 |
| SDD의 이점은 모델 스케일에 무관하게 일관적이다 | 4M~2.5B 전 모델 크기에서 SDD가 항상 우세 | p.4, Figure 2 |
| 대부분의 합성 생성기에 대해 SDD는 해석적으로 계산 가능하다 | GP, ARIMA/DLM, OU 프로세스, i.n.i.d. 시계열 등에 대한 closed-form 도출 | p.3, Table 1; Appendix B, Table 2 |
| SDD와 Status Quo의 학습 단계 연산 비용은 동일하다 | 조건부 분포 통계량을 데이터 생성 시 사전 캐싱 가능 | p.3, Section 2.3 |
| 교사 강제(TF) 하에서도 SDD가 우세하나 효과는 작다 | TF: $1.19\times$~$1.33\times$, CPM: $1.62\times$~$1.85\times$ | p.11-12, Figure 3, Appendix D.1 |
| 배치 크기가 클수록 SDD의 상대적 이점은 감소한다 | $B=64$: $1.26\times$, $B=512$: $1.13\times$ (Table 4) | p.12-13, Table 4 |
| 시퀀스 길이가 길수록 SDD 이점이 증가한다 | $T=128$: $1.26\times$, $T=1024$: $1.34\times$ | p.13, Table 4 |

---

## 2-1. 상세 설명

### 해결하고자 하는 문제

합성 시계열 데이터로 TSFM을 사전학습할 때, 기존 방법은 확률적 미래 궤적의 **단일 실현값**을 목표(target)로 사용한다:

$$\frac{1}{B}\sum_{b=1}^{B} \ell\!\left(f_\theta(y_{0:t}^{(b)}),\, y_{t+1:t+h}^{(b)}\right) $$

> **기호 설명**
> - $f_\theta$: 가중치 $\theta$를 가진 TSFM
> - $B$: 배치 크기
> - $\ell$: 손실 함수 (MSE, pinball loss, cross-entropy 등)
> - $h$: 예측 지평(forecast horizon)
> - $y_{0:t}^{(b)}$: $b$번째 샘플의 과거 시계열값
> - $y_{t+1:t+h}^{(b)}$: $b$번째 샘플의 미래 시계열값(단일 실현값)

이 방식은 동일한 $y_{0:t}$에 대해 여러 미래 궤적이 가능함에도 단 하나만 사용하므로, 경사 추정에 **샘플링 노이즈**가 개입된다.

---

### 제안하는 방법 (SDD)

**핵심 아이디어**: 단일 실현값 $y_{t+1:t+h}$ 대신, 데이터 생성 과정 $\pi_\alpha$로부터의 조건부 분포 전체를 손실 계산에 활용한다.

**증류 손실(Distilled Loss) 정의**:

$$\ell_{\text{distill}}^{(\pi_\alpha, h)}(\theta,\, y_{0:t}) := \mathbb{E}_{\pi_\alpha}\!\left[\ell\!\left(f_\theta(y_{0:t}),\, Y_{t+1:t+h}\right) \;\middle|\; y_{0:t}\right] $$

> **기호 설명**
> - $\pi_\alpha$: 하이퍼파라미터 $\alpha$를 가진 데이터 생성 과정
> - $Y_{t+1:t+h}$: 미래 시계열의 확률변수 (단순 실현값이 아닌 랜덤 변수)
> - $\mathbb{E}\_{\pi_\alpha}[\cdot | y_{0:t}]$: $y_{0:t}$가 주어졌을 때 $\pi_\alpha$ 하에서의 조건부 기댓값

**핵심 등식 (Law of Total Expectation)**:

$$\mathbb{E}_{\pi_\alpha}\!\left[\ell\!\left(f_\theta(y_{0:t}),\, Y_{t+1:t+h}\right)\right] = \mathbb{E}_{\pi_\alpha}\!\left[\mathbb{E}_{\pi_\alpha}\!\left[\ell\!\left(f_\theta(y_{0:t}),\, Y_{t+1:t+h}\right) \;\middle|\; y_{0:t}\right]\right] $$

이에 따라 $\ell_{\text{distill}}$은 $\ell$과 **동일한 기댓값**을 가지면서, 조건부 분포를 적분하여 샘플링 노이즈를 제거한다.

> **💡 용어 설명**
> - **라오-블랙웰화(Rao-Blackwellization)**: 추정량의 기댓값은 유지하면서 분산을 줄이는 통계적 기법. 충분 통계량(sufficient statistic)으로 조건화하여 불필요한 무작위성을 제거
> - **조건부 기댓값(Conditional Expectation)**: 특정 정보(여기서는 $y_{0:t}$)가 주어졌을 때의 기댓값

---

### 손실 함수별 증류 형태 (Table 1, p.3)

| 손실 $\ell$ | 기존 형태 | 증류 형태 $\ell_{\text{distill}}$ |
|---|---|---|
| Squared Error | $(\hat{y}\_{t+i} - Y_{t+i})^2$ | $(\hat{y}_{t+i} - \mu)^2 + s^2$ |
| Absolute Error | $\lvert\hat{y}\_{t+i} - Y_{t+i}\rvert$ | $s\!\left[2m(z) + z(2F_0(z)-1)\right]$ |
| Pinball ($\tau$) | $(Y_{t+i}-\hat{y}\_{t+i})(\tau - \mathbf{1}\{Y_{t+i} < \hat{y}_{t+i}\})$ | $s\!\left[m(z)+z(F_0(z)-\tau)\right]$ |
| Cross-entropy | $-\log\hat{p}(\text{bin}(Y_{t+i}))$ | $-\sum_k \mathbb{P}(Y_{t+i}\in\text{bin}_k)\log\hat{p}_k$ |

> **기호 설명**
> - $\mu = \mathbb{E}\_{\pi_\alpha}[Y_{t+i}|y_{0:t}]$: $y_{0:t}$가 주어졌을 때 $Y_{t+i}$의 조건부 평균
> - $s > 0$: 조건부 표준편차
> - $z = (\hat{y}_{t+i} - \mu)/s$: 표준화된 예측 오차
> - $Z = (Y_{t+i} - \mu)/s$: 표준화된 확률변수 (밀도 $f_0$, CDF $F_0$ 보유)
> - $m(z) = \int_z^\infty u\,f_0(u)\,du$: 상위 부분 평균(upper partial mean)
> - $\hat{p}_k$: 빈(bin) $k$의 예측 확률

> **💡 용어 설명**
> - **Pinball Loss (분위수 손실)**: 분위수 예측에 사용되는 비대칭 손실함수. 레벨 $\tau$에서 과소예측과 과대예측에 다른 페널티 부여
> - **CRPS (Continuous Ranked Probability Score)**: 확률적 예측의 품질을 평가하는 지표로, 분위수 손실의 합으로 표현 가능

---

### 분산 감소 보장 (Proposition 1, p.3-4)

$$\text{Cov}_{\pi_\alpha}[\nabla_\theta L_{\text{distill}}(\theta)] = \text{Cov}_{\pi_\alpha}[\nabla_\theta L(\theta)] - \mathbb{E}_{\pi_\alpha}[\text{Cov}_{\pi_\alpha}(\nabla_\theta L(\theta)|y_{0:t})] \preceq \text{Cov}_{\pi_\alpha}[\nabla_\theta L(\theta)] $$

> **기호 설명**
> - $L(\theta) = \ell(f_\theta(y_{0:t}), y_{t+1:t+h})$: Status Quo 손실
> - $L_{\text{distill}}(\theta) = \ell_{\text{distill}}^{(\pi_\alpha,h)}(\theta, y_{0:t})$: 증류 손실
> - $\text{Cov}\_{\pi_\alpha}[\cdot]$: $\pi_\alpha$ 분포 하에서의 공분산 행렬
> - $\preceq$: 뢰브너 편순서 (Loewner partial order) — $A \preceq B$는 $B-A$가 양반정치(PSD) 행렬임을 의미

> **💡 용어 설명**
> - **뢰브너 편순서(Loewner Partial Ordering)**: 행렬 $A \preceq B$는 $B-A$가 양반정치 행렬(모든 고유값이 0 이상)임을 의미. 공분산 행렬 비교에 사용
> - **양반정치 행렬(Positive Semi-Definite Matrix)**: 모든 고유값이 0 이상인 대칭 행렬

---

### 모델 구조

- **기반 모델**: Toto-2 [참고문헌 9] — 패치 길이 $P=32$의 next-patch predictor
- **아키텍처**: 패치 간 인과적 시간 어텐션(causal time attention) + 채널 간 변수 어텐션(variate attention)을 교차 배치
- **모델 크기**: 4.1M, 21.9M, 312.7M, 1041.0M, 2454.3M 파라미터
- **출력 헤드**: 9개 분위수(decile)를 출력하는 분위수 헤드(quantile head)
- **학습 방식**: Contiguous Patch Masking (CPM) 또는 Teacher Forcing
- **옵티마이저**: AdamW, weight decay $10^{-4}$, gradient clipping 1.0, lr= $10^{-5}$ , cosine decay

> **💡 용어 설명**
> - **Contiguous Patch Masking (CPM)**: 연속된 패치(시계열 구간)를 무작위로 가리고 해당 구간을 예측하도록 학습하는 방식
> - **Teacher Forcing**: 매 시점에서 실제 과거값을 모델 입력으로 사용하여 다음 시점을 예측하도록 학습하는 방식

---

### 실험 조건 및 성능 향상

**실험 데이터**: 길이 $T=512$의 단변량 가우시안 프로세스 궤적

$$y_{0:T-1} \mid (\kappa, \alpha, a, c) \sim \mathcal{N}\!\left(g,\, K_{\kappa\alpha} + \tilde{\sigma}^2 I_T\right), \quad \tilde{\sigma}^2 = \sigma^2 + 10^{-4} $$

**가우시안 프로세스 조건부 분포**:

$$Y_q \mid y_{0:t} \sim \mathcal{N}(\mu, \Sigma), \quad \mu = g_q + K_{qc}\widetilde{K}_{cc}^{-1}(y_{0:t} - g_c), \quad \Sigma = K_{qq} + \tilde{\sigma}^2 I - K_{qc}\widetilde{K}_{cc}^{-1}K_{cq} $$

**SDD 분위수 헤드 손실**:

$$\ell_{\text{distill}} = \frac{1}{hK}\sum_{i=1}^{h}\sum_{k=1}^{K} s_i\!\left[\phi(z_{ik}) + z_{ik}\!\left(\Phi(z_{ik}) - \tau_k\right)\right], \quad z_{ik} = \frac{\hat{y}_{t+i,\tau_k} - \mu_i}{s_i} $$

> **기호 설명**
> - $K = 9$: 분위수 개수 (9개 십분위수)
> - $\tau_k$: $k$번째 분위수 레벨
> - $\phi$: 표준정규분포 밀도함수
> - $\Phi$: 표준정규분포 CDF
> - $\mu_i, s_i$: 식 (11)의 조건부 분포의 평균과 표준편차

**성능 요약** (Figure 2, p.4):

| 모델 크기 | SDD Speed-up | FLOPs 절감 |
|---|---|---|
| 4M | $1.75\times$ | ~43% |
| 22M | $1.74\times$ | ~43% |
| 313M | $1.86\times$ | ~46% |
| 1B | $1.62\times$ | ~38% |
| 2.5B | $1.64\times$ | ~39% |

---

### 한계

1. **실험 범위**: 가우시안 프로세스 단일 데이터 생성기에서만 실험됨; 다른 합성 생성기에 대한 실험 부재
2. **생산 환경 미검증**: 실제 TSFM 사전학습에서 합성 데이터와 실세계 데이터가 혼합될 때의 효과 미확인
3. **단변량 한정**: 실험이 모두 단변량($C=1$) 시계열에 한정됨
4. **비교가능성 제한**: 교사 강제와 CPM 결과는 직접 비교 불가(학습 체제 상이)
5. **에러바 과소추정**: 동일 학습 코퍼스를 고정하여 사용하므로 에러바에 데이터 샘플링 변동성이 반영되지 않음 (Appendix C)

---

## 3. 각 주장의 위치 표시

| 주장 | 위치 |
|---|---|
| SDD의 동기 및 정의 | p.1-2, Section 2.2 |
| 증류 손실 수식 $\ell_{\text{distill}}$ | p.2, Eq.(3); p.3, Table 1 |
| Rao-Blackwellization 이론적 증명 | p.3-4, Proposition 1, Eq.(4) |
| 해석적으로 다루기 쉬운 생성기 분류 | p.3, Section 2.3; Appendix B, Table 2, Table 3 |
| CPM 실험 결과 ( $1.62\times$ ~ $1.86\times$ ) | p.4, Figure 2 |
| 교사 강제 결과 ( $1.19\times$ ~ $1.33\times$ ) | p.11-12, Figure 3, Appendix D.1 |
| 요인 분석 (노이즈/배치/시퀀스 길이) | p.12-13, Table 4, Appendix D.2 |
| 한계 및 미래 연구 방향 | p.5, Section 4 |

---

## 4. 저자 직접 보고 vs. 해석 분리

### 저자가 직접 보고한 결과

- **Figure 2 (p.4)**: "SDD attains the same validation loss as Status Quo while spending 38-46% fewer FLOPs, a convergence speed-up of $1.6\times$ to $1.85\times$ depending on model size."
- **Figure 3 / Appendix D.1 (p.11-12)**: "the speed-up is $1.19\times$ to $1.33\times$ rather than $1.6\times$ to $1.85\times$"
- **Table 4 (p.13)**: 관측 노이즈 $\sigma=0.1, 0.25$: $1.32\times$; 배치 $B=64$: $1.26\times$, $B=512$: $1.13\times$; 시퀀스 $T=128$: $1.26\times$, $T=1024$: $1.34\times$
- **Proposition 1 (p.3-4)**: Rao-Blackwellization에 의한 공분산 감소의 수학적 증명
- **Appendix C (p.11)**: "error bars carry no data-sampling variability and understate the spread that retraining on a freshly drawn corpus would show."

### 리뷰어 해석

1. **일반화 가능성의 제한**: 모든 실험이 가우시안 프로세스 단일 생성기에서 수행되었다. 저자들은 이를 한계로 인정하나, 실세계 혼합 데이터 환경이나 비가우시안 생성기에서의 성능은 검증되지 않았다.

2. **배치 크기 상호작용**: SDD의 이점이 배치 크기 증가에 따라 감소하는 현상은 이론적으로 예측 가능하나, 실제 대규모 학습(배치 크기 수천~수만)에서의 효용은 상당히 제한될 수 있다.

3. **CPM vs TF 체제 간 직접 비교 불가**: 논문 자체가 명시하듯 두 마스킹 체제의 수치는 직접 비교될 수 없어, 방법론 선택에 따라 SDD 이득이 크게 달라진다.

4. **수렴 후 성능 격차**: CPM 조건에서 수렴 후 CRPS 개선이 −0.90%~−2.16%로, 수렴 가속(speed-up)에 비해 최종 성능 개선폭은 제한적으로 보인다.

---

## 5. 통계적으로 취약한 부분 및 비교 불가능한 수치

> ⚠️ **통계적 취약점**

| 항목 | 내용 | 근거 위치 |
|---|---|---|
| **에러바 과소추정** | 동일 고정 캐시 코퍼스 사용으로 데이터 샘플링 변동성이 에러바에 반영 안됨 | Appendix C, p.11 |
| **소규모 반복 수**: 요인 분석에서 시드 3개만 사용 | 통계적 유의성 평가에 충분하지 않을 수 있음 | Appendix C, p.11 |
| **단일 데이터 생성기**: GP만 사용 | 다른 생성기(ARIMA, OU 등)에서의 결과 부재 | Section 3, p.4 |
| **단변량 한정**: $C=1$ | 다변량 시계열에서의 효과 미검증 | Appendix C, p.10 |

> ⚠️ **비교 불가능한 수치**

| 항목 | 이유 |
|---|---|
| Section 3 (CPM) vs Appendix D.1 (TF) CRPS 절댓값 | 학습 체제가 달라 직접 수치 비교 불가 (p.11 명시) |
| Section 3 vs Table 4 factor sweeps 속도향상 수치 | 예산 및 배치 크기 상이; Table 4는 Section 3의 1/3~1/6 예산 (p.11 명시) |
| 본 논문 학습률 ( $10^{-5}$ ) vs 원 Toto-2 학습률 ( $O(10^{-2})$ ) | 옵티마이저 스케일링 방식이 달라 직접 비교 불가 (Appendix C, p.10) |
| Training objective gap vs Held-out metric gap | Training 목표값의 CPM 쌍은 $s_i^2$ 보정 후에도 완전히 비교 불가 (Appendix C, p.10) |

---

## 6. 논문이 답하지 않는 질문

1. **혼합 데이터 환경**: 실세계 데이터와 합성 데이터가 혼합된 실제 사전학습 코퍼스에서 SDD의 계산 절감 효과가 유지되는가? (Section 4에서 미래 연구로 언급만 됨)

2. **비가우시안 생성기**: ARIMA, Ornstein-Uhlenbeck, Geometric Brownian Motion 등 비가우시안 생성기에서 SDD의 실질적 이득은 얼마인가?

3. **다변량 시계열**: 채널 수 $C > 1$인 다변량 설정에서 SDD 이득이 유지되는가? 채널 간 의존성은 어떻게 처리되는가?

4. **다운스트림 태스크 성능**: 검증 손실 개선이 실제 제로샷 예측 벤치마크 성능으로 전이되는가? (논문은 합성 데이터 검증 손실만 보고)

5. **몬테카를로 근사 최적 수**: 비해석적(intractable) 생성기에 대해 몇 개의 MC 샘플이 최적의 분산-비용 균형을 제공하는가?

6. **생산 규모 검증**: 수백만 스텝, 대규모 배치, 실제 TSFM 코퍼스 크기에서의 결과가 없다.

7. **파인튜닝 이후 성능**: SDD로 사전학습된 모델이 특정 도메인 파인튜닝 후 성능이 SQ 기반 모델과 어떻게 다른가?

8. **최적 학습률 선택**: SDD의 분산 감소로 인해 더 높은 학습률 사용이 가능한지, 그 경우 추가 이득이 있는지 미검토.

---

## 7. 가장 중요한 그림 5개 해석

### 📊 Figure 1 (p.2) — SDD vs Status Quo 직관적 비교

**해석**: 단일 시계열 궤적에서 두 방법의 차이를 시각화한다. 파란색 실선은 과거 관측 $y_{0:t}$이고, 검정 파선은 조건부 평균 $\mu = \mathbb{E}\_{\pi_\alpha}[Y_{t+1:t+h}|y_{0:t}]$, 검정 점선은 단일 실현 미래 $y_{t+1:t+h}$이다. 핵심 메시지: Status Quo는 조건부 평균으로부터 크게 벗어난 단일 실현값을 목표로 사용하여 경사 추정에 노이즈를 유발하는 반면, SDD는 조건부 평균을 목표로 사용하여 이 노이즈를 제거한다. 그림자(음영) 영역은 조건부 분포의 불확실성을 나타내며, SDD가 전체 분포 정보를 활용함을 직관적으로 보여준다.

---

### 📊 Figure 2 (p.4) — 5종 모델 크기 전체 학습 성능 비교 (메인 결과)

**해석**: X축은 누적 학습 연산량(FLOPs, $6ND$ 추정), Y축은 held-out CRPS이다. 각 색상이 하나의 모델 크기를 나타내며, 실선이 Status Quo, 점선이 SDD이다. **핵심 관찰**:
1. 모든 모델 크기에서 SDD 점선이 Status Quo 실선의 왼쪽에 위치: 동일한 CRPS 도달에 더 적은 FLOPs 소요
2. 같은 FLOPs에서 SDD가 항상 낮은 CRPS 달성
3. Speed-up은 313M 모델에서 최대($1.86\times$)이고, 1B 모델에서 최소($1.62\times$)
4. 두 굵은 선(endpoints 연결)은 모델 크기 스케일링 특성을 보여줌

---

### 📊 Figure 3 (p.12) — 교사 강제(Teacher Forcing) 조건 결과

**해석**: CPM과 달리 교사 강제 조건에서는 5개 패널로 분리하여 표시하는데, 이는 모델 크기 간 CRPS 범위가 두 방법 간 차이보다 100배 크기 때문이다. **핵심 관찰**:
1. Speed-up이 $1.19\times$~$1.33\times$로 CPM($1.62\times$~$1.85\times$) 대비 현저히 작음
2. 초반에 SDD가 열고 간 격차가 학습 후반에 Status Quo가 일부 회복
3. 이론적 예측과 일치: TF는 이미 매 스텝에서 $T$개 위치 평균으로 노이즈 억제가 이루어지므로, SDD의 추가 분산 감소 효과가 제한적
4. TF가 CPM보다 절댓값 CRPS가 낮은 것은 메트릭이 TF 방식으로 계산되기 때문

---

### 📊 Table 1 (p.3) — 손실 함수별 증류 형태

**해석**: 가장 중요한 이론적 기여를 압축한 표이다. 각 행은 일반적으로 사용되는 손실 함수($\ell$)에 대해 기존 실현값 기반 형태와 해석적 증류 형태를 대조한다. **핵심 관찰**:
1. **제곱 오차**: 증류 손실은 조건부 평균 $\mu$와의 MSE에 $\theta$-독립적 상수 $s^2$ 추가. 최적화 관점에서 $s^2$ 항이 경사에 기여하지 않으므로, 사실상 조건부 평균으로의 예측을 목표로 함
2. **Cross-entropy**: 단일 원-핫 벡터 대신 조건부 확률 분포에 대한 cross-entropy로, 소프트 레이블(soft label) 학습에 해당
3. 모든 증류 손실은 해석적으로 계산 가능하며, 무한 개의 미래 궤적을 적분한 결과와 동등

---

### 📊 Table 4 (p.13) — 요인별 SDD Speed-up

**해석**: 3개 요인(관측 노이즈 $\sigma$, 배치 크기 $B$, 시퀀스 길이 $T$)에 따른 SDD 가속 효과를 정리한다. **핵심 관찰**:
1. **관측 노이즈**: $\sigma$가 증가할수록 speed-up이 $1.32\times$에서 $1.22\times$로 소폭 감소. 노이즈 증가는 양방 arm의 CRPS를 동시에 증가시키나 상대적 차이는 비교적 유지됨
2. **배치 크기**: $B$ 증가 시 speed-up이 $1.26\times \to 1.13\times$로 단조 감소. 배치 평균화가 SDD가 제거하려는 노이즈를 이미 부분적으로 억제하기 때문
3. **시퀀스 길이**: $T$ 증가 시 speed-up이 $1.26\times \to 1.34\times$로 단조 증가. 긴 문맥이 조건부 분포를 더 정확하게 결정하여 증류 신호의 품질 향상

---

## 8. 결론: 시사점, 후속 연구 계획 및 방향

### 8-1. 저자가 제시한 시사점 및 후속 연구 계획 (Section 4, p.5)

**저자 시사점**:
- SDD는 TSFM 사전학습을 위한 합성 데이터 활용을 개선하는 원리적(principled) 방법이다.
- 계산 비용 증가 없이 $1.62\times$ ~ $1.86\times$의 학습 가속을 달성 가능하다.
- 이론적 보장(Rao-Blackwellization)이 실증적 결과와 일치한다.

**저자가 제시한 후속 연구**:
1. 더 넓은 범위의 합성 데이터 생성기에서의 수치 실험
2. 합성 데이터와 실세계 데이터가 혼합된 생산 규모 사전학습에서의 효과 검증
3. TSFM을 넘어 표 형식 파운데이션 모델(TabPFN 등)로의 적용

---

### 8-1. 모델의 일반화 성능 향상 가능성 (중점)

SDD는 손실 함수의 분산을 줄임으로써 경사 방향의 **신호 대 잡음비(SNR)**를 높인다. 이는 단순한 학습 가속을 넘어 일반화 성능 향상으로 이어질 수 있는 여러 메커니즘을 제공한다.

**① 조건부 분포 학습**: SDD는 모델이 단일 실현값의 편향된 신호 대신 **진정한 조건부 예측 분포**를 학습하도록 유도한다. 이는 특정 실현값의 이상치(outlier)에 과적합되는 것을 방지하여 일반화를 향상시킨다.

**② 분위수 헤드의 수렴 특성**: Appendix A 분석에 따르면, pinball loss에 대한 증류 손실 $\ell_{\text{distill}}$의 최솟값이 정확히 $\hat{y}_{t+i} = \mu + sF_0^{-1}(\tau)$, 즉 **진정한 조건부 $\tau$-분위수**에 위치한다. SDD는 분위수 헤드가 학습하는 목표를 변경하지 않으면서 더 빠르게 최적에 수렴하도록 한다.

**③ 일반화 한계**: 현재 실험은 합성 데이터(GP)로만 진행되었다. 사전학습된 모델의 **제로샷 일반화 성능** 향상 여부를 확인하려면 M4, ETTh, Traffic 등 실세계 벤치마크에서의 평가가 필요하다. SDD로 수렴된 모델이 반드시 더 좋은 제로샷 성능을 보인다는 보장은 현재 논문에서 제공되지 않는다.

**④ 데이터 다양성과의 상호작용**: 실제 TSFM 사전학습에서는 다양한 도메인의 실세계 데이터가 포함된다. SDD는 tractable한 합성 데이터 부분에만 적용되므로, 혼합 코퍼스에서의 전체적 일반화 향상은 합성 데이터의 비중에 비례할 것으로 예측된다.

**⑤ 시퀀스 길이와 일반화**: Table 4 결과에서 시퀀스 길이 $T$가 길수록 SDD 이득이 증가한다는 것은, 더 긴 문맥을 갖는 실제 응용에서 일반화 성능 향상 잠재력이 더 크다는 것을 시사한다.

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> **⚠️ 중요 주의사항**: 아래 비교는 논문 내 참고문헌과 공개된 연구들을 기반으로 합니다. 논문 자체가 2026년 9월 제출이므로, 2025-2026년 논문들도 포함됩니다.

| 연구 | 연도 | 방법 | 합성 데이터 활용 | SDD와의 관계 |
|---|---|---|---|---|
| **Chronos** [참고문헌 1] | 2024 | T5 기반, 시계열을 토큰화하여 언어모델 사전학습 | 실세계+합성 혼합 | SDD 적용 가능한 i.n.i.d. 생성기 사용; Status Quo 방식 |
| **TimesFM** [참고문헌 5] | 2024 | Decoder-only 파운데이션 모델, teacher forcing | GP 기반 합성 | SDD의 teacher forcing 버전 적용 가능 |
| **ForecastPFN** [참고문헌 6] | 2023 | PFN 기반 제로샷 예측, 순수 합성 사전학습 | i.n.i.d. 합성만 사용 | Table 3에 분류됨; SDD 직접 적용 가능 |
| **TempoPFN** [참고문헌 11] | 2025 | Linear RNN 기반, 합성 사전학습 | KernelSynth + ForecastPFN 생성기 포함 | Table 3에 분류; SDD 적용 가능 |
| **Chronos-2** [참고문헌 2] | 2025 | 다변량으로 확장 | 혼합 | SDD의 다변량 확장 가능성 연구 필요 |
| **Moirai 2.0** [참고문헌 10] | 2025 | CPM 기반, 패치 예측 | 혼합 | CPM 방식에서 SDD 적용 시 본 논문과 유사한 이득 예상 |
| **Toto 2.0** [참고문헌 9] | 2026 | 스케일링 시대의 TSFM | GP 기반 | 본 논문의 실험 기반 모델; SDD 공식 적용 연구 |
| **TiRex-2** [참고문헌 14] | 2026 | 다변량+스트리밍으로 일반화 | 미상 | SDD의 다변량 적용 연구와 연계 가능 |
| **Dataset Distillation** [참고문헌 17] | 2018 | 데이터셋 압축 방법론 | 해당 없음 | SDD의 개념적 모티베이션 |
| **Transformers can do Bayesian inference (TabPFN)** [참고문헌 12] | 2022 | 사전 적합 네트워크(PFN), Bayesian 추론 | 순수 합성 | SDD의 이론적 기반과 유사한 방향 |

---

### 앞으로의 연구 영향 및 고려사항

#### 이 논문이 미치는 영향

1. **합성 데이터 활용 패러다임 전환**: 단순히 더 많은 합성 데이터를 생성하는 방향에서, **기존 데이터를 더 효율적으로 활용**하는 방향으로 연구 관심을 이동시킬 것이다.

2. **이론-실증 연결**: Rao-Blackwellization이라는 고전 통계이론을 대규모 딥러닝 사전학습에 적용한 선례로, 유사한 분산 감소 기법 연구를 촉진할 것이다.

3. **다른 모달리티로의 파급**: 시계열 외에도 합성 데이터로 사전학습되는 모든 구조화 데이터 모델(표 형식, 그래프 등)에 확장 가능하다.

4. **데이터 생성기 설계 기준 변화**: SDD와의 호환성(tractability)이 새로운 합성 데이터 생성기 설계의 중요 기준이 될 것이다.

#### 후속 연구 시 고려할 점 (리뷰어 추가 제안)

1. **제로샷 벤치마크 평가**: Monash, M4, ETTh 등 표준 벤치마크에서 SDD 사전학습 모델의 제로샷 성능을 직접 측정하여 downstream 일반화 이득 확인 필요.

2. **혼합 코퍼스에서의 최적 가중치**: 실세계 데이터(Status Quo 손실)와 합성 데이터(SDD 손실)를 혼합할 때 최적 비율 탐색.

3. **MC 근사와 SDD 간의 트레이드오프 분석**: 비해석적 생성기에 대해 샘플 수 $M$에 따른 분산 감소 vs 연산 비용 균형을 실험적으로 정량화.

4. **적응적 SDD**: 학습 진행에 따라 SDD와 Status Quo의 비율을 동적으로 조정하는 커리큘럼 학습 방법 탐구.

5. **다변량 확장**: 채널 간 의존성이 있는 다변량 시계열에서 조건부 분포의 해석적 계산 및 SDD 적용 방법 개발.

6. **Diffusion/Flow 기반 TSFM**: 확률적 생성 모델 기반의 TSFM에서 SDD 원리를 적용하는 방법 탐구.

7. **Continual Pretraining**: 이미 사전학습된 모델을 새로운 도메인 데이터로 계속 학습할 때 SDD의 효과.

---

## 참고문헌

본 분석에 직접 활용된 논문 내 참고문헌:

1. Ansari et al. (2024). *Chronos: Learning the language of time series.* TMLR.
2. Ansari et al. (2025). *Chronos-2: From univariate to universal forecasting.* arXiv:2510.15821.
3. Bhatia, R. (1997). *Matrix Analysis.* Springer. (Loewner 편순서 이론)
4. Blackwell, D. (1947). *Conditional expectation and unbiased sequential estimation.* Annals of Mathematical Statistics.
5. Das et al. (2024). *A decoder-only foundation model for time-series forecasting.* ICML.
6. Dooley et al. (2023). *ForecastPFN: Synthetically-trained zero-shot forecasting.* NeurIPS.
7. Hollmann et al. (2023). *TabPFN.* ICLR.
8. Hollmann et al. (2025). *Accurate predictions on small data with a tabular foundation model.* Nature.
9. Khwaja et al. (2026). *Toto 2.0: Time series forecasting enters the scaling era.* arXiv:2605.20119.
10. Liu et al. (2025). *Moirai 2.0.* arXiv:2511.11698.
11. Moroshan et al. (2025). *TempoPFN.* arXiv:2510.25502.
12. Müller et al. (2022). *Transformers can do Bayesian inference.* ICLR.
13. Nie et al. (2023). *A time series is worth 64 words.* ICLR.
14. Podest et al. (2026). *TiRex-2.* arXiv:2607.01204.
15. Rao, C.R. (1945). *Information and the accuracy attainable in the estimation of statistical parameters.* Bulletin of the Calcutta Mathematical Society.
16. Vaswani et al. (2017). *Attention is all you need.* NeurIPS.
17. Wang et al. (2018). *Dataset distillation.* arXiv:1811.10959.
18. Xie et al. (2026). *Cauker.* ICLR. arXiv:2508.02879.

**원본 논문**: Biswas, N., & El Karoui, N. (2026). *Distillation of Synthetic Data for Time Series Foundation Models.* arXiv:2609.09586v1.
