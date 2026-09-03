# Data Augmentation Policy Search for Long-Term Forecasting

---

## 1. Executive Summary (10문장 이내)

본 논문은 장기 시계열 예측(Long-Term Forecasting)을 위한 자동 데이터 증강(Automatic Data Augmentation) 프레임워크인 **TSAA(Time-Series AutoAugment)**를 제안한다.  
딥러닝 기반 시계열 예측 모델은 과파라미터화(overparameterized) 경향이 있어 과적합에 취약하며, 시각(vision) 분야와 달리 시계열에 특화된 자동 증강 기법은 부재했다.  
TSAA는 이중 수준 최적화(bi-level optimization) 문제를 실용적으로 완화하여, 부분 사전학습(partial pre-training)과 베이지안 최적화(Bayesian optimization) 기반 정책 탐색, ASHA(Asynchronous Successive Halving) 기반 가지치기(pruning)를 결합한 2단계 프레임워크로 구성된다.  
13종의 시계열 고유 변환(transformation)으로 구성된 사전(dictionary)을 정의하고, TPE(Tree-structured Parzen Estimator)와 EI(Expected Improvement)를 활용해 최적 증강 정책을 자동으로 탐색한다.  
6개 벤치마크 데이터셋과 다수의 기준 모델(Informer, Autoformer, FEDformer, N-BEATS 등)에 걸친 광범위한 실험에서, TSAA는 단변량·다변량 설정 모두에서 대부분의 예측 설정을 개선한다.  
특히 Weather 데이터셋 다변량 96 horizon에서 MSE 23.73% 감소, 단변량에서 최대 75% 감소를 달성하는 등 두드러진 성과를 보인다.  
Fast AutoAugment, RandAugment 대비 더 일관된 성능 향상(평균 MSE 3.33% 감소)을 제공한다.  
다만 Exchange와 같이 랜덤 워크(random walk) 특성이 강한 금융 데이터에서는 개선 효과가 제한적이다.  
코드는 공개 저장소(https://github.com/azencot-group/TSAA)에 공개되어 있으며, 2025년 2월 Transactions on Machine Learning Research(TMLR)에 게재되었다.

---

### 1-1. 연구의 목적과 필요성

| 측면 | 내용 |
|------|------|
| **문제 배경** | 딥러닝 시계열 예측 모델은 과파라미터화 경향이 있어 소규모 데이터셋에서 과적합 위험이 높음 (p.1) |
| **기존 한계** | 이미지·자연어 처리 분야에서는 AutoAugment 등 자동 증강이 발전했으나, 시계열 예측(TSF)에 특화된 자동 증강 연구는 현저히 부족 (p.1, p.2) |
| **필요성** | 시계열 데이터는 트렌드, 계절성, 잡음 등 고유한 특성을 가져 이미지 증강 기법을 그대로 적용하기 어려움 (p.5) |
| **목적** | 장기 시계열 예측에 특화된 효율적이고 구현 용이한 자동 증강 프레임워크 개발 |

> **💡 용어 설명**
> - **과파라미터화(Overparameterized)**: 모델의 파라미터 수가 학습 데이터 수보다 훨씬 많은 상태. 학습 데이터에는 완벽히 맞지만 새로운 데이터에는 잘 일반화되지 않는 과적합이 발생하기 쉬움.
> - **데이터 증강(Data Augmentation)**: 원본 데이터를 변형하여 인위적으로 학습 데이터를 늘리는 기법. 모델의 일반화 성능 향상에 효과적.

---

## 2. 핵심 주장과 근거 표

| 핵심 주장 | 근거 | 위치 |
|-----------|------|------|
| TSAA는 기존 베이스라인 모델의 성능을 대부분의 설정에서 향상시킨다 | 다변량 39/48, 단변량 32/48 지표에서 최고 성능 달성 | Table 1, 2 (p.8-9) |
| 이중 수준 최적화를 2단계 분리 최적화로 완화하면 효율적 | ASHA+Bayesian Opt 적용 시 800→170 에폭으로 감소 | App. D (p.19) |
| β=0.5(절반 사전학습)이 최적의 계산-성능 비율을 달성 | β=0.5: 5.3% 개선, β=1.0(완전 증강): 5.1% 개선 | Fig. 4A (p.10) |
| 시계열 증강에서 Trend Downscale, Jittering, Mixup, Smoothing이 효과적 | 대부분 데이터셋에서 Top-5 변환에 포함됨 | Fig. 3 (p.9) |
| TSAA는 Fast AutoAugment, RandAugment보다 일관성 있게 우수 | 평균 MSE 개선: TSAA 3.33%, RandAugment 1.67%, Fast AA -9.5% | Table 3 (p.12) |
| 랜덤 워크 특성이 강한 데이터에는 TSAA 효과가 제한적 | Exchange 데이터셋에서 일부 음수 개선율 관찰 | App. E.5, Fig. 7 (p.24-26) |

---

## 2-1. 핵심 내용 상세 설명

### ① 해결하고자 하는 문제

- 시계열 예측 모델(Transformer 계열 등)의 **과적합** 문제
- 이미지 분야 자동 증강과 달리, **시계열에 특화된 자동 증강 정책 탐색 방법 부재**
- 이중 수준 최적화의 **계산 비용 과다** 문제

> **💡 용어 설명**
> - **이중 수준 최적화(Bi-level Optimization)**: 두 개의 최적화 문제가 중첩된 구조. 상위 문제의 해가 하위 문제에 의존하고, 하위 문제의 해가 상위 문제에 영향을 주는 형태.

---

### ② 제안하는 방법 (수식 포함)

**[이중 수준 최적화 공식화]** (p.4, Eq. 3-4)

$$\min_{\theta} \mathcal{L}_{\text{val}}(\omega, \theta) $$

$$\text{subject to} \quad \min_{\omega} \mathbb{E}_{p_\theta}[\mathcal{L}_{\text{tr}}(\omega, \theta)] $$

- $\theta$: 증강 정책(augmentation policy)
- $p_\theta$: 증강 정책의 분포
- $\omega$: 신경망 가중치
- $\mathcal{L}_{\text{val}}$: 검증 손실
- $\mathcal{L}_{\text{tr}}$: 학습 손실

> **💡 용어 설명**
> - **검증 손실(Validation Loss)**: 학습에 사용되지 않은 데이터에서의 예측 오차. 모델의 일반화 성능 측정에 사용.

---

**[베이지안 최적화: TPE]** (p.3, Eq. 2)

$$p(x|y) = \begin{cases} l(x) & y < y^* \\ g(x) & y \geq y^* \end{cases} $$

- $x$: 탐색 파라미터 (TSAA 맥락에서는 정책 $\theta$)
- $y$: 목적함수 평가값 ($f(x) = \mathcal{L}_{\text{val}}$)
- $y^*$: 임계 점수(threshold score)
- $l(x)$: 성능 개선이 발생한 파라미터 분포 ($y < y^*$인 경우)
- $g(x)$: 성능 개선이 없는 파라미터 분포 ($y \geq y^*$인 경우)
- $l(x)/g(x)$를 최대화하는 것이 **Expected Improvement(EI)** 최적화에 해당

> **💡 용어 설명**
> - **Tree-structured Parzen Estimator (TPE)**: 베이지안 최적화의 한 방법. 과거 평가 결과를 두 확률 분포( $l(x)$, $g(x)$ )로 모델링하여, 다음으로 시도할 파라미터를 효율적으로 선택.
> - **Expected Improvement (EI)**: 현재까지의 최솟값보다 더 좋은 값을 얻을 기대 향상량. 탐색과 활용의 균형을 위한 획득 함수(acquisition function).

---

**[정책 탐색 공간]** (p.5, Eq. 5-6)

$$\theta \sim p_\theta := \prod_{j=1}^{k} p(\theta_j), \quad \theta_j \in \Theta, \quad \Theta := \{\theta_1, \ldots, \theta_k\} $$

$$\theta_j = T_{j,n}(x_{n-1}, m_{j,n}) \circ \cdots \circ T_{j,1}(x_0, m_{j,1}) $$

- $k$: 선택되는 서브 정책(sub-policy) 수 (실험에서 $k=3$)
- $\theta_j$: $j$번째 서브 정책
- $T_{j,i}$: $j$번째 서브 정책의 $i$번째 변환 함수
- $x_0$: 원본 입력 데이터
- $x_{i-1}$: 이전 변환의 출력
- $m_{j,i}$: $j$번째 서브 정책의 $i$번째 변환 강도(magnitude), 범위 $(0, 1]$
- $n$: 각 서브 정책 내 변환 수 (실험에서 $n=2$)

---

**[공유 가중치 및 자원 파라미터]** (p.4)

$$\omega_{\text{share}} := \omega(\lfloor \beta K \rfloor), \quad R := K - \lfloor \beta K \rfloor$$

- $\beta$: 부분 사전학습 비율 ($\beta = 0.5$)
- $K$: 베이스라인 모델의 실제 활성 학습 에폭 수 ($K \leq 10$)
- $\omega_{\text{share}}$: 공유 가중치 (전체 학습의 절반 시점 가중치)
- $R$: ASHA에 사용되는 최대 자원(에폭 수)

> **💡 용어 설명**
> - **공유 가중치(Shared Weights)**: 여러 증강 정책 탐색 시 매번 처음부터 학습하는 대신, 미리 부분 학습된 가중치를 공유하여 탐색 비용을 줄이는 기법.

---

**[ASHA 복잡도]** (App. D, p.19)

$$\mathcal{O}((1-\beta)K \cdot T_{\max})$$

- $T_{\max}$: 최대 시도(trial) 횟수 (실험에서 $T_{\max}=100$)
- $(1-\beta)K$: 파인튜닝에 사용되는 에폭 수

---

### ③ 모델 구조

TSAA는 두 단계로 구성된다 (Fig. 2, p.5):

```
입력 시계열
    ↓
[Step 1: 공유 가중치 생성]
  - 베이스라인 모델을 βK 에폭만큼 부분 학습
  - ω_share 저장
    ↓
[Step 2: 반복 분리 최적화 (Tmax회)]
  ┌─────────────────────────────────┐
  │ TPE로 증강 정책 θ 탐색         │ ← Eq. 3 해결
  │ (EI 기반 베이지안 최적화)       │
  │         ↓                      │
  │ ω_share에서 파인튜닝 + ASHA    │ ← Eq. 4 해결
  │ (저성능 실험 조기 종료)         │
  └─────────────────────────────────┘
    ↓
k개 최적 정책으로 p_θ* 구성
    ↓
θ* ~ p_θ* 로 최종 파인튜닝 → ω*
```

**13종 시계열 변환 사전 (Table 4, p.17):**

| 변환 | 설명 |
|------|------|
| Jittering | 백색 잡음 추가 |
| Trend scale (Up/Down) | 트렌드 성분 스케일 조정 |
| Seasonality scale (Up/Down) | 계절성 성분 스케일 조정 |
| Scale (Up/Down) | 전체 시계열 스케일 조정 |
| Smooth | 저역통과 필터링 |
| Noise Scale | 고역통과 필터링 + 스케일 |
| Permutation | 두 비중첩 시간 구간 교환 |
| Dynamic Time Stretching (DTS) | 시간 구간 길이 조작 |
| Window Warping (Up/Down) | 전체 윈도우 길이 조작 |
| Mixup | 두 시계열 선형 보간 |
| Flip | 시계열 부호 반전 |
| Reverse | 시간 순서 역전 |
| Identity | 원본 유지 |

> **💡 용어 설명**
> - **Mixup**: 두 데이터 샘플을 선형으로 혼합하는 증강 기법. $\tilde{x} = \lambda x_1 + (1-\lambda) x_2$로 표현되며, 모델이 클래스/값 간의 선형 관계를 학습하도록 유도.
> - **Dynamic Time Stretching (DTS)**: 시계열의 서로 다른 구간의 길이를 비균일하게 늘리거나 줄여 시간 축을 변형하는 기법.
> - **저역통과 필터링(Low-pass Filtering)**: 느리게 변화하는 성분(트렌드, 계절성 등)만 남기고 고주파 잡음을 제거하는 신호 처리 기법. Smoothing에 해당.

---

### ④ 성능 향상 및 한계

**성능 향상:**

| 설정 | 최대 개선 | 데이터셋 |
|------|-----------|---------|
| 다변량 MSE | 23.73% ↓ | Weather (horizon 96) |
| 단변량 MSE | 75.00% ↓ | Weather (horizon 192) |
| 단변량 Exchange MSE | 39.43% ↓ | Exchange (horizon 720) |
| 평균 horizon 720 | 21.74% ↓ | 전체 평균 |

**한계:**

- Exchange 데이터셋 (랜덤 워크 특성)에서 일부 음수 개선율
- 다변량 Exchange, horizon 336/720에서 기존 베이스라인 대비 성능 저하 (Table 1, p.8)
- 계산 비용: $T_{\max}=100$에서 Electricity/336 설정 시 최대 170 에폭 소요 (App. D, p.19)
- 종자 수(seed)가 3개로 통계적 신뢰도 제한

---

## 3. 주장별 페이지/Figure/Table 번호

| 주장 | 근거 위치 |
|------|-----------|
| TSAA 프레임워크 개요 | Fig. 2 (p.5), Algorithm 1 (p.6) |
| 다변량 예측 성능 | Table 1 (p.8), Tables 6-9 (p.20) |
| 단변량 예측 성능 | Table 2 (p.9), Tables 10-14 (p.21) |
| 최적 증강 정책 분석 | Fig. 3 (p.9) |
| β 파라미터 선택 | Fig. 4A (p.10) |
| Tmax 수렴 분석 | Fig. 4B (p.10) |
| TSAA vs Fast AA vs RandAugment | Table 3 (p.12) |
| 실제 예측 시각화 | Fig. 5 (p.11) |
| 랜덤 워크 한계 분석 | Fig. 7 (p.26), App. E.5 (p.24) |
| 표준편차 포함 결과 | Tables 15-16 (p.22-23) |
| PatchTST, iTransformer 비교 | Tables 17-19 (p.24) |

---

## 4. 저자 보고 결과 vs. 해석 분리

### 저자가 직접 보고한 결과

**연구 주제 (p.1-2):**
> "we introduce a time-series automatic augmentation approach named TSAA, which is both efficient and easy to implement."

**방법 (p.4-6):**
- 이중 수준 최적화를 Eq. 3-4로 공식화
- 베이지안 TPE+EI로 정책 탐색 (Eq. 2)
- ASHA로 저성능 실험 조기 종료

**결과 (p.8-9):**
- "TSAA achieves the best results in 39 error metrics" (다변량)
- "TSAA obtained the best models for 32 error metrics" (단변량)
- Weather 96 다변량: MSE 23.73% 감소 (0.236→0.180)
- 평균 MSE 감소: TSAA 3.33%, RandAugment 1.67%, Fast AA –9.5% (Table 3)

### 내 해석 및 분석

1. **β=0.5 선택의 함의**: 저자들은 β=0.5가 계산 효율과 성능의 최적 균형점이라고 주장하나, 실제로는 4개 horizon 중 2개에서만 β=0.5가 최선이었다 (Fig. 4A). 이는 데이터셋/모델에 따라 최적 β가 다를 수 있음을 시사한다.

2. **Trend Downscale의 우세**: Fig. 3에서 Trend Downscale이 ETTm2, Weather, Electricity에서 30% 이상의 비율을 차지한다. 이는 딥러닝 모델이 트렌드를 과추정(overestimate)하는 구조적 편향이 있을 가능성을 시사하며, 이는 모델 설계 개선의 방향성을 제공한다.

3. **Exchange 실패 원인**: 랜덤 워크 분석(App. E.5)은 타당하나, 단 5개의 합성 데이터 실험으로 검증하여 통계적 강건성이 부족하다.

4. **비교 대상의 제한성**: 2021-2022년 모델(Autoformer, FEDformer 등)이 주 비교 대상이며, TimesNet, DLinear, PatchTST 등 보다 최신 SOTA 모델과의 비교는 부록에서만 일부 수행된다.

---

## 5. 통계적으로 취약한 부분과 비교 불가능한 수치

### ⚠️ 통계적 취약점

| 항목 | 문제점 |
|------|--------|
| **실험 반복 횟수** | 종자(seed) 3개만 사용 → 95% 신뢰구간 미보고, 통계적 유의성 검정 없음 (p.7) |
| **랜덤 워크 한계 실험** | 합성 데이터 5개 셋업만으로 검증 (App. E.5, p.25) — 표본 크기 매우 작음 |
| **β 선택 실험** | ILI + ETTm2 데이터셋에만 국한, 타 데이터셋 일반화 불확실 (p.10) |
| **η 비교** | η=2 vs η=3 차이 0.12%로 통계적으로 비유의적일 가능성 높음 (p.10-11) |
| **표준편차** | Tables 15-16에서 제공되나 주 결과 Tables 1-2에서는 누락 |

### ⚠️ 비교 불가능한 수치

| 항목 | 이유 |
|------|------|
| **베이스라인 재현 수치** | "reported baseline results may slightly differ from the reported values" (p.7) — 원저자 환경과 상이 |
| **TSAA vs PatchTST/iTransformer** | 부록(Tables 17-19)에서만 보고, 다른 데이터셋에서는 비교 없음 |
| **단변량 Exchange** | FEDformer-f 기준 MSE -75.91%(악화) 등 극단적 수치는 변동성이 크게 관찰됨 (Fig. 9) |
| **Tmax 실험** | ILI 데이터셋에만 국한 (Fig. 4B), 다른 데이터셋으로의 일반화 불명확 |
| **계산 비용 비교** | TSAA의 총 계산 비용과 베이스라인 반복학습 비용의 공정한 비교 부재 |

---

## 6. 논문이 답하지 않는 질문

1. **최적 하이퍼파라미터의 이전 가능성**: $\beta=0.5$, $k=3$, $n=2$, $\eta=3$이 모든 데이터셋/모델 조합에서 최적인가? 각 설정별 최적값은 다를 수 있다.

2. **증강 정책의 전이 학습 가능성**: 한 데이터셋에서 찾은 최적 정책을 다른 유사 데이터셋에 직접 적용할 수 있는가?

3. **랜덤 워크 데이터에 대한 대안**: Exchange 같은 금융 데이터에 효과적인 증강 전략은 무엇인가?

4. **증강의 이론적 보장**: 특정 변환이 시계열의 분포적 특성(distribution)을 보존함을 보장하는 이론적 근거가 있는가?

5. **확장성**: 수천 개 변수를 가진 초대규모 다변량 시계열(예: 스마트 그리드, 센서 네트워크)에서도 TSAA가 효율적인가?

6. **증강 없는 대형 데이터셋**: 충분히 큰 데이터셋에서도 TSAA가 유의미한 개선을 제공하는가, 아니면 소규모 데이터에서만 효과적인가?

7. **다른 손실 함수**: MSE 외 다른 손실 함수(MAE, MAPE, SMAPE 등)를 사용 시 동일한 정책이 최적인가?

8. **실시간 적용**: 스트리밍 시계열 환경에서의 온라인 정책 탐색 가능성은?

---

## 7. 가장 중요한 그림 5개 해석

### **Figure 1 (p.3): DA 정책 예시 및 ASHA 동작**

**[Fig. 1A]** Electricity 데이터에 두 가지 서브 정책 적용 예시:
- 상단: "Upscale + Smooth" — 진폭을 키우고 고주파 성분을 제거하여 트렌드가 명확해짐
- 하단: "Jittering + Downscale" — 잡음을 추가하고 진폭을 줄여 다양성을 증가

**[Fig. 1B]** ASHA 동작 시각화:
- 파란색(Baseline): 전체 학습 완료
- 여러 색상 곡선: 서로 다른 정책으로 시작한 실험들
- Rung 0, Rung 1에서 저성능 실험이 조기 종료되어 계산 자원 절약
- 해석: ASHA가 없으면 모든 실험을 끝까지 학습해야 하지만, ASHA를 통해 유망한 정책에 자원을 집중할 수 있음

> **💡 용어 설명**
> - **Rung(런그)**: ASHA에서 모델 체크포인트를 생성하는 에폭 시점. 각 런그에서 성능 하위 $1/\eta$ 비율의 실험이 종료됨.

---

### **Figure 2 (p.5): TSAA 전체 프레임워크 구조**

TSAA의 2단계 파이프라인을 시각화:
- **Step 1**: 입력 → 부분 학습 → $\omega_{\text{share}}$ 생성
- **Step 2**: TPE가 정책 $\theta$를 제안 → ASHA와 함께 파인튜닝 → 반복 → 최적 정책 분포 $p_{\theta^\*}$ 구성 → 최종 $\omega^\*$ 획득

핵심 해석: 정책 탐색($\theta$ 최적화)과 가중치 학습($\omega$ 최적화)을 **교대로 반복**함으로써 이중 수준 최적화의 계산 부담을 줄임. 공유 가중치 덕분에 매 탐색 시 처음부터 학습할 필요 없음.

---

### **Figure 3 (p.9): 데이터셋별 최적 변환 비율**

**다변량(상단)과 단변량(하단)에서 상위 5개 변환의 선택 비율:**

- **Trend Downscale**: ETTm2, Weather, Electricity에서 30% 이상 선택됨 → 딥러닝 모델이 트렌드를 과대추정하는 경향을 보정하는 효과
- **Jittering**: 트렌드나 계절성을 변형하지 않으면서 다양성 증가
- **Smoothing**: Jittering의 반대 효과, 잡음 감소
- **Mixup**: 다변량 4개, 단변량 3개 데이터셋에서 Top-5에 포함 → 분포 다양성 증가에 효과적
- **데이터셋별 차이**: ILI(주간 독감 데이터)는 Seasonality 관련 변환이 더 선호됨

---

### **Figure 4 (p.10): β 및 Tmax 민감도 분석**

**[Fig. 4A: β 선택]**
- x축: full(β=0.0), half(β=0.5), last epoch, none(β=1.0)
- y축: 정규화된 평균 성능
- 결론: β=0.0(전체 증강학습)과 β=0.5(절반 사전학습+증강)이 유사한 성능이지만, β=0.5가 **훨씬 적은 계산 비용**으로 달성 가능

**[Fig. 4B: Tmax 선택]**
- x축: Tmax ∈ {100, 150, 200, 250}
- y축: 정규화된 평균 MSE
- 결론: Tmax 증가 시 성능 향상되나 체감(diminishing returns). Transformer 계열 1% MSE 감소, N-BEATS 7.25% 감소 (Tmax 100→250). **실용적 기본값: Tmax=100**

---

### **Figure 5 (p.11): 예측 시각화 비교**

네 가지 모델(Informer, Autoformer, FEDformer-f, NBEATS-G)의 동일 예측 대상에 대한 결과 비교:
- 파란색: 실제값(ground truth)
- 주황색: 증강 없는 예측
- 초록색: TSAA 적용 예측

**관찰:**
1. **Informer (ETTm2)**: TSAA가 Trend Downscale + Permutation 정책으로 예측 정렬 개선
2. **Autoformer (ETTm2)**: Permutation + Mixup으로 이상치(outlier) 감소
3. **FEDformer-f (Weather)**: Seasonality Downscale + Noise Scale + Downscale 정책이 진폭 과대추정 보정
4. **NBEATS-G (Weather)**: Flip + Seasonality Downscale 정책이 효과적

핵심 해석: 같은 데이터셋 내에서도 모델마다 다른 최적 정책이 선택되며, 이는 **모델 특성에 맞는 맞춤형 증강**의 필요성을 시사한다.

---

## 8. 결론: 시사점, 후속 연구, 추가 방향

### 8-1. 모델의 일반화 성능 향상 가능성

#### 저자가 제시한 시사점 (p.12, Conclusion)

1. **이중 수준 최적화의 실용적 완화**: 복잡한 bi-level 문제를 2단계 분리 최적화로 단순화하면서도 효과적인 정책 탐색 가능
2. **플러그인(plug-in) 활용성**: TSAA는 기존 TSF 모델에 추가 변경 없이 적용 가능하여 범용성 높음
3. **변환 분석의 인사이트**: Trend Downscale의 우세는 딥러닝 모델의 트렌드 과추정 편향을 시사, 향후 모델 설계에 활용 가능

#### 저자가 제시한 후속 연구 계획 (p.12)

1. **End-to-End 학습 가능한 bi-level 최적화** 탐색 (DADA, Deep AutoAugment 방향)
2. **학습 가능한(learnable) DA 모듈** 도입 — 합성곱 모델의 필터와 유사한 방식

#### 일반화 성능 향상 관련 분석

TSAA의 일반화 성능 향상은 다음 메커니즘에서 비롯된다:

| 메커니즘 | 내용 |
|----------|------|
| **분포 다양성 확장** | Mixup, Jittering 등이 학습 데이터의 분포를 확장하여 미관측 패턴에 대한 강건성 향상 |
| **편향 보정** | Trend Downscale이 딥러닝 모델의 트렌드 과추정 편향을 상쇄 |
| **정규화 효과** | DA는 암묵적 정규화(implicit regularization) 역할을 하여 과적합 방지 |
| **후기 단계 적용** | β=0.5로 후반부에 DA를 적용하면 초반에 학습된 기초 표현을 유지하면서 다양성 주입 가능 |

**한계**: 랜덤 워크 특성이 강한 데이터(Exchange)에서는 증강이 오히려 노이즈 데이터를 더 주입하여 일반화를 저해. 이는 **데이터 특성에 따른 증강의 선택적 적용** 필요성을 시사한다.

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> **⚠️ 주의**: 아래 비교는 제공된 논문의 내용과 일반적으로 알려진 연구 동향을 바탕으로 작성되었습니다. 각 연구의 정확한 수치는 해당 원문을 반드시 확인하시기 바랍니다.

#### 관련 최신 연구 동향

| 연구 | 핵심 기여 | TSAA와의 관계 |
|------|-----------|--------------|
| **PatchTST** (Nie et al., 2023) | 시계열을 패치(patch) 단위로 토큰화하여 Transformer 적용 | TSAA가 PatchTST에 적용 시 소폭 개선 (Table 18, App.) |
| **iTransformer** (2024) | 변수(variate) 차원에서 어텐션 적용 | TSAA 적용 시 일부 개선, ILI에서 최대 13.86% 감소 (Table 19) |
| **Crossformer** (2023) | 크로스 차원 어텐션 | TSAA 적용 시 ETTm2에서 87.94% MSE 감소 (Table 17) — 단, Crossformer 자체가 약한 기준점 |
| **DLinear/NLinear** (Zeng et al., 2023) | 단순 선형 모델의 경쟁력 입증 | TSAA 논문에서 직접 비교 없음 — **비교 공백** |
| **TimesNet** (Wu et al., 2023) | 시계열을 2D로 변환하여 비전 기법 적용 | TSAA와 직접 비교 없음 — **비교 공백** |
| **FrAug** (Chen et al., 2023) | 주파수 도메인 증강 | TSAA의 증강 사전에 주파수 변환 미포함 — 통합 가능성 |

#### TSAA가 앞으로의 연구에 미치는 영향

1. **시계열 자동 증강 연구의 기반 마련**: 시계열 특화 변환 사전과 자동 정책 탐색을 결합한 첫 번째 포괄적 프레임워크로서, 후속 연구의 기준점(baseline) 역할

2. **정책 분석의 인사이트 제공**: Trend Downscale의 우세성 발견은 딥러닝 TSF 모델의 트렌드 편향 연구로 이어질 수 있음

3. **플러그인 프레임워크의 모범 사례**: 특정 모델 구조에 독립적인 증강 프레임워크 설계 방향 제시

#### 앞으로 연구 시 고려할 점

1. **더 강력한 베이스라인과의 비교**: Mamba, TimeMixer 등 2024-2025년 최신 SOTA 모델과의 비교 필요

2. **주파수 도메인 증강 통합**: FrAug(Chen et al., 2023)와 같은 주파수 기반 변환을 증강 사전에 추가하면 성능 향상 가능성

3. **학습 가능한 증강 모듈**: 고정 매그니튜드 대신 그래디언트 기반으로 최적 매그니튜드를 학습하는 방향 (저자들도 언급)

4. **메타 학습(Meta-learning) 연계**: 새로운 데이터셋에 빠르게 최적 정책을 적응시키는 메타 학습 기반 정책 전이

5. **비정상(Non-stationary) 시계열 처리**: 랜덤 워크 등 비정상 데이터에 대한 특화 증강 전략 개발 필요

6. **효율성 개선**: $T_{\max}=100$의 탐색 비용을 줄이기 위한 One-shot NAS(Neural Architecture Search) 방식의 증강 탐색 도입

7. **다운스트림 태스크 다양화**: 현재 예측에 특화되어 있으나, 이상 탐지(anomaly detection), 분류(classification) 등으로 확장

8. **연속 학습(Continual Learning) 환경**: 시간에 따라 분포가 변화하는 실제 환경에서의 온라인 정책 업데이트 메커니즘 필요

---

## 참고자료

**본 논문:**
- Nochumsohn, L., & Azencot, O. (2025). *Data Augmentation Policy Search for Long-Term Forecasting*. Transactions on Machine Learning Research. arXiv:2405.00319v2. https://openreview.net/forum?id=Wnd0XY0twh

**논문 내 주요 인용 문헌:**
- Cubuk et al. (2019). *AutoAugment: Learning augmentation strategies from data*. CVPR.
- Cubuk et al. (2020). *RandAugment*. CVPR Workshops.
- Lim et al. (2019). *Fast AutoAugment*. NeurIPS.
- Li et al. (2020a). *A system for massively parallel hyperparameter tuning (ASHA)*. MLSys.
- Bergstra et al. (2011). *Algorithms for hyper-parameter optimization (TPE)*. NeurIPS.
- Wen et al. (2020). *Time series data augmentation for deep learning: A survey*. IJCAI.
- Nie et al. (2023). *PatchTST: A time series is worth 64 words*. ICLR.
- Zeng et al. (2023). *Are transformers effective for time series forecasting?* AAAI.
- Wu et al. (2021). *Autoformer*. NeurIPS.
- Zhou et al. (2022). *FEDformer*. ICML.
- Fama, E.F. (1965). *The behavior of stock-market prices*. Journal of Business.
- Akiba et al. (2019). *Optuna*. KDD.

**코드 저장소:** https://github.com/azencot-group/TSAA
