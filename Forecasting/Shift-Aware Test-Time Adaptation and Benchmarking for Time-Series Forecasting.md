# Shift-Aware Test-Time Adaptation and Benchmarking for Time-Series Forecasting

> **참고 논문**: Grover, S., & Etemad, A. (2025). "Shift-Aware Test-Time Adaptation and Benchmarking for Time-Series Forecasting." *Proceedings of the 42nd International Conference on Machine Learning (ICML 2025)*, PMLR 267.
> **코드/데이터**: https://github.com/shivam-grover/DynaTTA

---

## 1. Executive Summary (10문장 이내)

1. 실세계의 시계열(time-series) 데이터는 **비정상성(non-stationarity)**을 가지며, 학습 시점과 추론 시점 사이에 분포가 달라지는 **분포 이동(distribution shift)** 문제가 빈번하게 발생한다.
2. 기존 Test-Time Adaptation(TTA) 연구는 컴퓨터 비전·언어 분류 태스크에 집중되어 있으며, 회귀 기반의 시계열 예측(TSF)에 대한 TTA는 크게 미개척 상태이다.
3. 기존 TSF-TTA 방법(예: TAFAS)은 **고정된 적응률(fixed adaptation rate)**과 **분포 이동 강도를 고려하지 않는 게이팅(gating)** 메커니즘을 사용하여 과적응 또는 과소적응 문제를 가진다.
4. 또한 기존 표준 TSF 데이터셋은 테스트 분할에서 분포 이동이 미미하거나 반복적이어서 TTA 방법 평가에 부적합하다.
5. 본 논문은 이 두 문제를 해결하기 위해 **DynaTTA**와 **TTFBench** 두 가지를 제안한다.
6. DynaTTA는 MSE Z-score, 단기 임베딩 드리프트(RTAB), 장기 임베딩 드리프트(RDB) 세 가지 지표로 실시간 분포 이동을 추정하고, 이를 기반으로 적응률과 게이팅을 동적으로 조절한다.
7. TTFBench는 기존 표준 데이터셋의 테스트 분할에 추세(trend), 계절성(seasonality), 레짐 변화(regime shift), 국소 잡음(local noise) 등 다양한 섭동을 주입하여 1,000개 변형을 생성한 최초의 TSF-TTA 전용 벤치마크이다.
8. DynaTTA는 5개 백본 모델(PatchTST, iTransformer, DLinear, FreTS, MICN)과 표준 및 TTFBench 데이터셋에서 기존 방법 대비 일관된 성능 향상을 달성한다.
9. 표준 벤치마크에서 iTransformer 대비 최대 7.21%, TAFAS 대비 최대 6.1%, TTFBench에서 PatchTST 대비 최대 8.41%, TAFAS 대비 최대 4.39%의 MSE 감소를 기록했다.
10. DynaTTA는 백본 파라미터를 동결(frozen) 상태로 유지하며 재학습 없이 어떤 사전학습 TSF 모델에도 모듈 방식으로 적용 가능하다.

---

### 1-1. 연구의 목적과 필요성

| 구분 | 내용 |
|------|------|
| **핵심 문제** | 실세계 시계열 데이터의 비정상성과 분포 이동이 모델 일반화 성능을 저하시킴 |
| **기존 TTA의 한계** | (1) 분류 태스크 중심 설계로 회귀·시계열 예측에 직접 적용 불가 (2) 고정 적응률로 이동 강도에 따른 유연한 대응 불가 (3) 적절한 벤치마크 부재 |
| **필요성** | 분포 이동의 심각도를 실시간으로 감지하고 적응 전략을 동적으로 조절하는 TSF 전용 TTA 프레임워크와 평가 환경 필요 |

> 💡 **비정상성(Non-stationarity)**: 시계열 데이터의 평균, 분산 등 통계적 특성이 시간에 따라 변하는 성질. 예를 들어, 에너지 소비 패턴이 계절이나 사회 변화로 인해 점차 달라지는 현상.

> 💡 **분포 이동(Distribution Shift)**: 모델을 학습한 데이터(소스 분포)와 실제 추론 시점의 데이터(타겟 분포)가 달라지는 현상. 예를 들어, 코로나19 이전 데이터로 학습한 교통량 예측 모델이 이후 데이터에 부정확해지는 경우.

> 💡 **Test-Time Adaptation(TTA)**: 모델을 재학습하지 않고, 테스트 시점에서 들어오는 데이터만을 활용하여 모델을 소량 업데이트하는 기법.

---

## 2. 핵심 주장과 근거 표

| 핵심 주장 | 근거/방법 | 위치 |
|-----------|-----------|------|
| 고정 적응률은 TSF-TTA에 부적합 | 과적응/과소적응 문제 발생; 동적 조절 필요성 시각화 | p.1-2, Figure 1 |
| 기존 벤치마크는 TSF-TTA 평가에 불충분 | 표준 데이터셋(Electricity, ETTh2 등)의 테스트 분할이 학습 분포와 유사 | p.2, Figure 2 |
| DynaTTA의 동적 적응률이 성능 향상 | 표준/TTFBench 데이터셋에서 모든 백본 대비 일관된 MSE 감소 | p.4, Table 1, 4 |
| TTFBench가 더 도전적이고 현실적인 평가 제공 | 1,000개 다양한 섭동 변형 생성; 베이스라인 MSE가 표준 대비 증가 | p.4, Table 4; p.3 |
| 각 컴포넌트(SCG, DAR, WU, MSE, RTAB, RDB)가 모두 기여 | Ablation study에서 각 요소 제거 시 성능 저하 확인 | p.4, Table 2 |
| DynaTTA는 랜덤 시드에 강건 | 10회 반복 실험에서 표준편차가 매우 작음(≤0.0007) | p.4, Table 3 |

---

## 2-1. 상세 설명

### 2-1-a. 해결하고자 하는 문제

**문제 1 (p.1-2)**: 기존 TSF-TTA 방법(특히 TAFAS, Kim et al., 2025)의 두 가지 결함:
- **고정 적응률(Fixed LR)**: 분포 이동의 심각도에 무관하게 동일한 학습률 적용 → 가벼운 이동에서 과적응, 급격한 이동에서 과소적응
- **비동적 게이팅(Non-dynamic Gating)**: 사전학습 모델과 적응 모델의 기여도 조절이 분포 이동 강도에 연동되지 않음

**문제 2 (p.2, Figure 2)**: 기존 표준 TSF 벤치마크(ETTh1/2, Electricity, Exchange Rate 등)는 학습/검증/테스트 분할 간 분포 차이가 미미하거나 반복적이어서 TTA 방법의 실질적인 평가가 불가능.

---

### 2-1-b. 제안 방법 및 수식

#### ① 분포 이동 추정 (p.2, 10-11)

**[수식 1] 시각 t에서의 예측 MSE:**

$$\text{MSE}_t^{(H)} = \frac{1}{CH} \sum_{c=1}^{C} \sum_{l=1}^{H} \left(\hat{x}_{t+l-1}^{(c)} - x_{t+l-1}^{(c)}\right)^2 $$

- $C$: 채널(변수) 수
- $H$: 예측 지평선(horizon) 길이
- $\hat{x}_{t+l-1}^{(c)}$: 채널 $c$의 시각 $t+l-1$ 예측값
- $x_{t+l-1}^{(c)}$: 채널 $c$의 시각 $t+l-1$ 실제값

**[수식 2] MSE Z-score (분포 이동 추정):**

$$z_t = \frac{\text{MSE}_t^{(L)} - \mu_t}{\sigma_t + \epsilon}$$

- $\mu_t$: 롤링 버퍼 내 MSE들의 평균
- $\sigma_t$: 롤링 버퍼 내 MSE들의 표준편차
- $\epsilon$: 수치 안정성을 위한 소상수

> 💡 **Z-score**: 어떤 값이 평균에서 얼마나 떨어져 있는지를 표준편차 단위로 표현한 수치. Z-score가 높으면 최근 예측 오류가 평소보다 크게 증가했다는 신호.

**[수식 3] 부분 MSE (partial MSE for RTAB):**

$$E_i = \text{MSE}_t^{(l)} = \frac{1}{C \cdot l} \sum_{c=1}^{C} \sum_{j=1}^{l} \left(\hat{x}_{t+j-1}^{(c)} - x_{t+j-1}^{(c)}\right)^2 $$

- $l$: 현재까지 관측된 미래 스텝 수 ($l < H$)

**[수식 4] RTAB 임베딩 가중 평균을 위한 가중치:**

$$w_i^{(t)} = \frac{\alpha_i \cdot \beta}{\sum_{j \in \mathcal{B}_t} \alpha_j \cdot \beta}, \quad \text{where } \beta = \frac{1}{E_i + \epsilon} $$

- $\alpha_i \in (0,1]$: 신뢰도 계수. 부분 MSE이면 $\alpha_i = l/H$, 완전 MSE이면 $\alpha_i = 1$
- $\mathcal{B}_t$: 시각 $t$에서 RTAB에 저장된 인덱스 집합
- $\beta$: MSE의 역수 (예측 정확도가 높은 임베딩에 더 큰 가중치 부여)

> 💡 **RTAB(Real-Time Adaptation Buffer)**: 최근 입력 임베딩과 그 MSE를 저장하는 단기 메모리 버퍼. 최근의 분포 변화를 빠르게 감지하는 역할.

> 💡 **RDB(Reference Distribution Buffer)**: MSE가 가장 낮았던 K개의 임베딩을 장기 보존하는 버퍼. 소스 분포의 안정적 대리 역할.

---

#### ② 동적 적응률 조정 (p.3)

**[수식 5] 이동 점수 기반 적응률 승수:**

$$\lambda_t = 1 + \left(\frac{\alpha_{\max}}{\alpha_{\min}} - 1\right) \cdot \frac{1}{1 + e^{-\kappa S_t}}$$

- $S_t$: 세 가지 Z-score 정규화된 지표들의 합산 이동 점수
- $\alpha_{\max}$: 허용되는 최대 적응률 (설정값: 0.01)
- $\alpha_{\min}$: 허용되는 최소 적응률 (설정값: 0.0005)
- $\kappa$: 민감도 조절 스케일링 파라미터

**[수식 6] 지수 평활화 기반 적응률 업데이트 (논문 Eq. 1):**

$$\alpha_{t+1} = \alpha_t + \eta \left(\alpha_{\text{target}} - \alpha_t\right) $$

- $\alpha_t$: 현재 시각의 적응률
- $\eta$: 지수 평활화 계수
- $\alpha_{\text{target}} = \alpha_{\min} \cdot \lambda_t$: 목표 적응률

> 💡 **지수 평활화(Exponential Smoothing)**: 최신 값과 이전 값을 가중 평균하여 갑작스러운 변화를 완화하는 기법. $\eta$가 클수록 최신 정보를 더 빠르게 반영.

**[수식 7] 워밍업 계수:**

$$\gamma_t = \min\left(1, \frac{n_t}{\alpha_{\text{warm}} \cdot H}\right)$$

- $n_t$: 현재까지 관측된 샘플 수
- $\alpha_{\text{warm}}$: 조절 가능한 워밍업 파라미터

워밍업 적용 후 목표 적응률: $\alpha_{\text{target}} = \alpha_{\text{base}} \cdot [1 + \gamma_t(\lambda_t - 1)]$

---

#### ③ 이동 조건부 게이팅 (p.3)

$$\mathbf{X}_{\text{cal}} = \mathbf{X} + \tanh(\phi_{\text{dynamic}}) \circ (\mathbf{W} * \mathbf{X} + \mathbf{b})$$

- $\phi_{\text{dynamic}} = \phi_{\text{base}} + f_{\text{gate}}(\mathbf{m}_t)$: 동적 게이팅 파라미터
- $\phi_{\text{base}} \in \mathbb{R}^C$: 초기 0으로 설정된 학습 가능한 기저 파라미터
- $f_{\text{gate}}$: 소형 MLP (이동 추정치를 게이팅 신호로 변환)
- $\mathbf{m}_t \in \mathbb{R}^d$: MSE Z-score, RTAB 거리, RDB 거리를 포함한 이동 지표 벡터
- $\mathbf{W}, \mathbf{b}$: 시간 보정을 위한 학습 가능한 가중치와 편향
- $\circ$: 원소별 곱셈(element-wise multiplication)
- $*$: 변수별 시간 변환(variable-wise temporal transformation)
- $\tanh$: 게이팅 신호를 $(-1, 1)$ 범위로 제한하는 활성화 함수

> 💡 **게이팅(Gating)**: 어떤 정보를 얼마나 통과시킬지 조절하는 메커니즘. 여기서는 적응 모듈의 영향력을 분포 이동 심각도에 따라 조절.

---

### 2-1-c. 모델 구조 (p.3, Figure 4)

```
[입력 Xt-L:t-1]
      ↓
[입력 어댑터 (W,b) + 게이팅] → 조정된 입력
      ↓
[동결된(Frozen) 사전학습 백본] → 중간 출력 + 임베딩 ft
      ↓
[출력 어댑터 (W',b') + 게이팅] → 최종 예측 Ŷt^adapt
      ↑                      ↑
[RTAB/RDB/MSE 버퍼 업데이트] → mt 계산
      ↓
[적응률 조정] + [게이팅 파라미터 업데이트]
(과거 예측에 대한 실제값이 점진적으로 공개되면 Backprop)
```

**핵심 설계 원칙**:
- 백본은 항상 동결(frozen) 상태 유지
- 두 개의 경량 어댑터(입력 정규화, 출력 역정규화) 만 업데이트
- 레이블(정답)이나 학습 데이터 접근 없이 테스트 스트림만 사용

---

### 2-1-d. 성능 향상 및 한계

**성능 향상 (p.4, Table 1, 4)**:
- 표준 벤치마크: iTransformer 대비 최대 7.21%, TAFAS 대비 최대 6.1% MSE 감소
- TTFBench: PatchTST 대비 최대 8.41%, TAFAS 대비 최대 4.39% MSE 감소
- 모든 5개 백본과 5개 데이터셋에서 일관된 성능 향상

**한계**:
- 비교 기준(baseline)이 TAFAS 단 하나로 제한됨 (다른 TTA 방법들과의 비교 없음)
- 합성(synthetic) 섭동으로 생성된 TTFBench가 실제 분포 이동을 완전히 재현하는지 검증 미흡
- DLinear처럼 내부 임베딩이 없는 모델의 경우 MSE 신호만 활용하는 축소 모드로 동작

---

## 3. 각 주장의 위치 표시

| 주장 | 위치 |
|------|------|
| 기존 TTA의 고정 적응률 문제 | p.1 (Problem 1), Figure 1 (p.2) |
| 기존 벤치마크의 분포 이동 부재 | p.2 (Problem 2), Figure 2 (p.2) |
| DynaTTA 구조 및 작동 원리 | p.2-3, Figure 4 (p.3) |
| TTFBench 구성 방법 | p.3 (Section 3), Appendix A.3 (p.15-18) |
| 표준 벤치마크 성능 결과 | p.4, Table 1 |
| TTFBench 성능 결과 | p.4, Table 4 |
| Ablation study | p.4, Table 2 |
| 랜덤 시드 민감도 | p.4, Table 3 |
| 적응률 진화 시각화 | Appendix, Figure A1 (p.12) |
| 게이팅 값 진화 시각화 | Appendix, Figure A2 (p.13), Figure A3 (p.14) |
| TTFBench 섭동 예시 | Appendix, Figure A5 (p.19), A6 (p.20) |

---

## 4. 저자 보고 결과 vs. 해석자 분석

### 저자가 직접 보고한 결과

| 항목 | 저자 보고 내용 |
|------|---------------|
| **연구 주제** | TSF를 위한 동적 TTA 프레임워크(DynaTTA) 및 최초 TSF-TTA 벤치마크(TTFBench) 제안 |
| **방법** | MSE Z-score + RTAB + RDB 기반 분포 이동 추정 → 동적 적응률($\lambda_t$, Eq.1) + 이동 조건부 게이팅($\phi_{\text{dynamic}}$) |
| **성능(표준)** | DynaTTA는 모든 백본과 데이터셋에서 Base 및 TAFAS 대비 일관되게 낮은 MSE 달성 (최대 7.21%↓ vs Base, 6.1%↓ vs TAFAS) |
| **성능(TTFBench)** | TTFBench에서도 일관된 향상 (최대 8.41%↓ vs Base, 4.39%↓ vs TAFAS) |
| **Ablation** | 모든 컴포넌트 기여; 워밍업(WU)과 RDB가 가장 중요 (Table 2) |
| **랜덤 시드** | 표준편차 ≤ 0.0007 수준으로 결과 안정적 (Table 3) |

### 해석자의 분석 및 평가

| 항목 | 해석자 분석 |
|------|------------|
| **방법의 강점** | 세 가지 보완적 신호(예측 오류 + 단기 임베딩 + 장기 임베딩)를 조합하여 다양한 시간 스케일의 분포 변화를 포착하는 설계가 타당함 |
| **비교 기준의 제한성** | TAFAS(2025) 단 하나만 비교 대상으로 제시. 도메인 적응, 지속적 학습 등 인접 방법론과의 비교 부재 |
| **TTFBench의 타당성 문제** | 합성 섭동의 현실 대표성을 별도 검증 없이 가정. 실제 세계의 분포 이동이 4가지 구조적 성분으로 충분히 설명되는지 의문 |
| **계산 비용 미보고** | 세 개의 메모리 버퍼 유지와 MLP 기반 게이팅의 추론 시간 오버헤드가 논문에서 정량적으로 보고되지 않음 |
| **개선 폭의 절대값** | MSE 기준 개선 폭이 수 백분의 일 ~ 수 퍼센트 수준으로, 실용적 유의성은 도메인에 따라 다를 수 있음 |

---

## 5. 통계적으로 취약한 부분과 비교 불가능한 수치

### ⚠️ 통계적으로 취약한 부분

| 문제 | 상세 설명 |
|------|-----------|
| **단일 baseline 비교** | TAFAS만 비교 대상으로 제시. 다른 TTA 방법(TEST: Test-time self-training, TTT 등)과의 비교 없어 우위를 일반화하기 어려움 |
| **통계 검정 부재** | Table 1, 4의 MSE 수치에 신뢰 구간, p-value 등 통계적 유의성 검정이 제시되지 않음 |
| **랜덤 시드 실험 범위 제한** | 랜덤 시드 민감도(Table 3)를 ETTh1/2, ETTm1에만 보고; Weather, Exchange에 대한 시드 실험 없음 |
| **Ablation 범위 제한** | Ablation study(Table 2)가 iTransformer 백본과 3개 데이터셋에만 한정. 다른 백본에서도 동일 결론인지 불명확 |

### ⛔ 비교 불가능한 수치

| 문제 | 상세 설명 |
|------|-----------|
| **TTFBench vs 표준 데이터 절대 수치** | TTFBench의 MSE는 합성 섭동으로 인해 표준 데이터셋 MSE보다 전반적으로 높음. 두 설정의 절대 MSE 수치는 직접 비교 불가 |
| **DLinear의 제한된 모드** | DLinear는 내부 임베딩이 없어 MSE 신호만 활용. 다른 백본과 동일 조건 비교가 아님 |
| **사전학습 설정 의존성** | TAFAS 논문의 사전학습 설정을 그대로 차용했기 때문에, 사전학습 방법이 다른 모델들과 직접 비교 어려움 |

---

## 6. 논문이 답하지 않는 질문

| 번호 | 미답 질문 |
|------|-----------|
| 1 | DynaTTA의 추론 시간 오버헤드(세 버퍼 유지 + 실시간 계산)는 얼마인가? |
| 2 | 하이퍼파라미터($\kappa$, $\eta$, $K_{\text{RDB}}$, $\alpha_{\text{warm}}$ 등)의 최적값 선택 방법과 민감도 분석은? |
| 3 | 완전히 새로운 분포(out-of-distribution)로의 급격한 이동이나 개념 표류(concept drift)에 대한 적응 한계는? |
| 4 | TTFBench의 합성 섭동이 실제 세계의 분포 이동을 얼마나 충실하게 재현하는가? |
| 5 | RTAB/RDB 버퍼 크기가 성능에 미치는 영향(버퍼 크기 Ablation)은? |
| 6 | 채널 수가 매우 많은 고차원 시계열(예: 수백 개 변수)에서의 확장성은? |
| 7 | 단변량(univariate) 시계열에 적용 시 성능은? |
| 8 | 분포가 지속적으로 이동하는 온라인 스트리밍 환경에서 버퍼 포화(buffer saturation) 문제는? |

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p.2) — DynaTTA vs. TAFAS 동적 적응 비교

**해석**: X축은 시각 스텝, Y축은 예측값. 시계열이 낮은 이동(Low Shift) → 높은 이동(High Shift) → 낮은 이동 → 소스 분포 복귀의 사이클을 보인다. TAFAS는 모든 구간에서 동일한 LR=고정값으로 동작하여 분포 복귀 시에도 불필요한 적응을 지속한다. 반면 DynaTTA는 이동 심각도에 따라 LR과 게이팅(G)을 실시간 조절(낮은 이동 시 LR↓·G↓, 높은 이동 시 LR↑·G↑, 복귀 시 LR↓·G↓)하여 Ground Truth에 더 가깝게 추적한다. 이는 동적 조절의 핵심 동기를 직관적으로 보여주는 그림.

---

### Figure 2 (p.2) — 표준 데이터셋의 분포 이동 부재

**해석**: Electricity, ETTh2 등 표준 벤치마크의 학습/검증/테스트 분할을 시각화. 테스트 데이터가 학습 데이터와 통계적으로 매우 유사하거나(Electricity), 반복적인 계절 패턴(ETTh2)을 보임. Exchange Rate는 스케일이 작아 미세한 변화만 존재. 이는 기존 데이터셋으로는 TTA 방법의 효과를 충분히 평가하기 어렵다는 TTFBench의 필요성을 명확히 보여줌.

---

### Figure 4 (p.3) — DynaTTA 전체 구조

**해석**: 완전한 DynaTTA 파이프라인을 도식화. 좌측의 세 메모리 버퍼(RTAB, RDB, MSE)에서 이동 지표 $\mathbf{m}_t$가 계산되고, 이것이 우측의 적응률 조정과 게이팅 파라미터를 동시에 업데이트. 동결된 백본을 중심으로 입력 어댑터(정규화)와 출력 어댑터(역정규화)만 업데이트되어 사전학습 지식을 보존하면서 적응하는 구조가 명확히 표현됨. 과거 예측에 대한 점진적 실제값 공개를 활용하는 Backprop 흐름도 포함.

---

### Figure A1 (p.12) — 워밍업 유무에 따른 적응률 진화

**해석**: ETTh2(H=96), ETTh1(H=336), ETTm1(H=96) 세 데이터셋에서 워밍업 없음($\alpha_{\text{warm}}=0$)과 있음($\alpha_{\text{warm}}=1$)의 적응률 변화를 비교. 워밍업 없는 경우(좌측 열), 초기 스텝에서부터 높은 적응률 변동을 보여 신뢰할 수 없는 초기 버퍼 정보에 과민 반응. 워밍업 적용 시(우측 열), 초기에는 적응률이 낮게 시작하여 충분한 데이터가 쌓인 후 점진적으로 활성화. 이는 Ablation에서 워밍업이 가장 중요한 컴포넌트임을 지지하는 시각적 증거.

---

### Figure A4 (p.18) — TTFBench 섭동 전후 채널 간 상관관계 보존

**해석**: Exchange Rate 데이터셋에서 섭동 적용 전후의 채널 간 Pearson 상관계수 행렬을 비교. 섭동 후에도 채널 쌍 간의 상관 구조(양/음의 상관관계 패턴)가 거의 동일하게 유지됨을 보여줌. 이는 $g^{(c)}(t) = \text{sign}(\bar{\rho}^{(c)}) \cdot g(t)$ 공식으로 글로벌 섭동 신호를 채널 상관성에 맞게 부호화한 설계의 효과를 검증함. TTFBench가 단순한 랜덤 노이즈 추가가 아닌 현실적인 다변량 관계를 보존한 벤치마크임을 지지.

---

## 8. 결론: 시사점, 후속 계획, 추가 연구 방향

### 8-0. 저자 제시 시사점 (p.4, Section 6)

- **실용적 가치**: 재학습 없이 어떤 사전학습 TSF 모델에도 적용 가능하여 모델 노후화 방지 및 재학습 비용 절감
- **범용성**: 날씨, 금융, 에너지 등 시계열이 편재하는 실세계 응용에서 일반화 성능을 높임
- **벤치마크 기여**: TTFBench를 공개하여 이후 TSF-TTA 연구의 표준 평가 기반 제공

### 8-1. 모델 일반화 성능 향상 가능성 (중점)

DynaTTA의 일반화 성능과 관련된 현재의 강점과 잠재적 개선 방향:

**현재 기여**:
- 백본 파라미터를 동결하여 **재앙적 망각(catastrophic forgetting)** 방지 → 소스 도메인 지식 보존
- RDB를 통한 소스 분포의 장기 메모리 유지 → 분포가 원점 복귀 시 적응 해제(적응 비활성화) 가능
- 모듈형 설계로 새로운 백본 아키텍처에 즉시 적용 가능

> 💡 **재앙적 망각(Catastrophic Forgetting)**: 신경망이 새로운 작업이나 데이터를 학습할 때 이전에 학습한 정보를 덮어쓰는 현상. TTA에서 과도한 적응은 사전학습 지식을 손상시킬 수 있음.

**일반화 성능 향상을 위한 추가 방향**:

1. **메타 러닝 기반 초기화**: MAML 등 메타 러닝으로 어댑터 파라미터의 초기값을 학습하면, 새로운 분포에서 더 빠르고 안정적인 적응 가능
2. **불확실성 정량화 통합**: 예측의 불확실성을 적응 신호로 활용하면(단순 MSE 외에 분포 기반 신호 추가) 더 정밀한 이동 감지 가능
3. **다중 소스 도메인 학습**: 사전학습 시 다양한 분포의 데이터를 혼합하면 어댑터가 더 넓은 변화 범위에 대응 가능
4. **희소 적응(Sparse Adaptation)**: LoRA처럼 어댑터 파라미터의 극소 일부만 업데이트하여 오버피팅 위험 감소

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> **⚠️ 주의**: 아래 비교는 논문 내 참조 문헌을 기반으로 하며, 2025년 이후 발표된 논문에 대해서는 확인이 제한됩니다. 정확도가 불확실한 부분은 명시합니다.

| 연구 | 핵심 방법 | DynaTTA와 비교 |
|------|-----------|---------------|
| **TENT** (Wang et al., ICLR 2020) | 엔트로피 최소화로 배치 정규화 업데이트 | 분류 태스크 특화, 회귀/순차 데이터에 직접 적용 불가 |
| **TTT** (Sun et al., ICML 2020) | 자기지도 보조 태스크로 모델 업데이트 | 보조 태스크 설계 필요; TSF에선 적절한 보조 태스크 정의 어려움 |
| **TAFAS** (Kim et al., 2025) | 게이팅 캘리브레이션 모듈 + 고정 LR | **DynaTTA의 직접 비교 대상**; 동적 조절 부재가 핵심 약점 |
| **Christou et al.** (Test time learning for TSF, 2024) | 테스트 시간 학습 기반 TSF | 고정 적응률 사용; DynaTTA는 동적 조절로 차별화 |
| **Gong et al.** (SIGKDD 2025) | 불확실성 인식 프로토타입 + 엔트로피 비교 | **시계열 분류**에 집중; TSF(회귀) 설정과 다름 |

**DynaTTA가 앞으로의 연구에 미치는 영향**:

1. **TSF-TTA 분야 개척**: 최초로 체계적인 TSF-TTA 방법론과 벤치마크를 동시 제시하여 후속 연구의 기준선(baseline) 역할
2. **TTFBench의 표준화 가능성**: 공개된 코드와 1,000개 섭동 변형 벤치마크가 TSF-TTA 연구 커뮤니티의 표준 평가 플랫폼이 될 가능성
3. **모듈형 설계 패러다임**: 백본을 동결하고 경량 어댑터만 업데이트하는 패러다임이 다른 순차적 예측 태스크(예: 이상 탐지, 임팩트 예측)로 확장 유도

**향후 연구 시 고려할 점**:

| 고려사항 | 이유 |
|----------|------|
| **다양한 baseline 비교 필수화** | TSF-TTA 분야가 발전함에 따라 더 많은 방법과의 공정한 비교 필요 |
| **실제 세계 분포 이동 데이터셋 구축** | TTFBench의 합성 섭동을 보완하는 실제 이동 데이터 필요 |
| **적응의 계산 효율성 보고** | 엣지 디바이스·실시간 시스템 배포를 위한 latency/FLOPs 분석 필요 |
| **MAE 외 다양한 평가 지표** | MSE 외에 MASE, SMAPE, Winkler Score 등으로 평가 다각화 |
| **장기 연속 드리프트 시나리오** | 점진적으로 변화하는 분포(gradual drift)에서의 안정성 검증 필요 |
| **멀티태스크/멀티도메인 일반화** | 단일 어댑터가 매우 다른 도메인에서도 효과적인지 검증 필요 |

---

**참고 자료 목록**:
1. Grover, S., & Etemad, A. (2025). *Shift-Aware Test-Time Adaptation and Benchmarking for Time-Series Forecasting*. ICML 2025, PMLR 267. (본 분석의 주요 대상 논문)
2. Kim, H., et al. (2025). *Battling the non-stationarity in time series forecasting via test-time adaptation*. arXiv:2501.04970. (TAFAS, 논문 내 주요 비교 대상)
3. Wang, D., et al. (2020). *TENT: Fully test-time adaptation by entropy minimization*. ICLR 2020. (논문 내 인용)
4. Nie, Y., et al. (2023). *A time series is worth 64 words: Long-term forecasting with transformers*. ICLR 2023. (PatchTST, 논문 내 인용)
5. Liu, Y., et al. (2023). *iTransformer: Inverted transformers are effective for time series forecasting*. ICLR 2023. (논문 내 인용)
6. Zeng, A., et al. (2023). *Are transformers effective for time series forecasting?* AAAI 2023. (DLinear, 논문 내 인용)
7. Ragab, M., et al. (2023). *AdaTime: A benchmarking suite for domain adaptation on time series data*. ACM TKDD. (논문 내 인용)
8. Xiao, Z., & Snoek, C. G. (2024). *Beyond model adaptation at test time: A survey*. arXiv:2411.03687. (논문 내 인용)
