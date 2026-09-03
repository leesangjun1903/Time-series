# Towards Principled Test-Time Adaptation for Time Series Forecasting

> **참고 자료**: Wang et al. (2026). "Towards Principled Test-Time Adaptation for Time Series Forecasting." arXiv:2605.17250v1 [cs.LG], 17 May 2026. Stony Brook University.

> **⚠️ 주의**: 본 논문은 2026년 5월 arXiv 프리프린트로, 아직 동료 심사(peer review)를 거치지 않은 상태입니다. 일부 인용 문헌(Im & Kwon, 2026; Huang et al., 2026 등)도 미래 날짜로 표기되어 있어, 실제 출판 여부를 독립적으로 확인할 수 없었습니다.

---

## 1. Executive Summary (10문장 이내)

본 논문은 시계열 예측(Time Series Forecasting, TSF)에서 **테스트 타임 적응(Test-Time Adaptation, TTA)**의 프로토콜 수준 문제를 재검토한다.  
기존 TSF-TTA 방법들은 부분적으로 관측된 정답(POGT)과 완전히 관측된 정답(Matured GT)을 혼합하여 사용하거나, 스트리밍 방식으로 현재 배치의 정답으로 어댑터를 업데이트하는 등 이질적인 프로토콜을 사용해왔다.  
저자들은 이 두 방식 모두 프로토콜 수준의 결함을 가진다고 주장하며, **오직 완전히 성숙한 정답(Matured Ground Truth)만을 사용하는 더 원칙적인 적응 프로토콜**을 제안한다.  
또한 기존 어댑터들의 예측 보정이 주파수 도메인에서 매끄럽고 구조화되지 않은 스펙트럼 수정을 보인다는 진단을 제시한다.  
이를 바탕으로 주파수 도메인에서 직접 예측 보정을 파라미터화하는 **Frequency-Aware Calibration(FAC)**을 제안한다.  
FAC는 입력/출력 FreqGCM(주파수 도메인 게이트 보정 모듈)을 통해 FFT → 복소 아핀 마스크 → iFFT → tanh 게이트의 경량 파이프라인을 구성한다.  
실험 결과, FAC는 6개 데이터셋, 5개 소스 예측기, 4개 예측 지평선에서 경쟁력 있는 성능을 달성하면서, 비교 대상 어댑터보다 **훨씬 적은 훈련 가능 파라미터**를 사용한다.  
TAFAS 대비 FAC의 파라미터 수는 최대 약 320배 적다(예: Weather, H=720: 11,097,114 vs 34,482).  
본 연구는 프로토콜 설계와 주파수 도메인 적응이라는 두 축에서 TSF-TTA의 원칙적 기반을 강화하는 데 기여한다.

### 1-1. 연구의 목적과 필요성

| 항목 | 내용 |
|------|------|
| **핵심 문제** | 실세계 시계열은 비정상성(non-stationarity)으로 인한 분포 변화(distribution shift)에 노출되어 있으며, 이는 사전 훈련된 예측기의 성능을 저하시킨다. |
| **기존 접근의 한계** | ① 학습 가능한 정규화 모듈(RevIN 등)은 훈련 단계에 초점 ② 온라인 학습은 예측기 자체를 처음부터 재훈련해야 함 |
| **TSF-TTA의 등장** | 사전 훈련된 소스 예측기를 동결하고, 가벼운 어댑터만 테스트 시점에 업데이트하는 방식이 대안으로 부상 |
| **기존 TSF-TTA의 문제** | ① POGT 혼합 감독 방식: 공개된 값을 감독으로 사용 vs. 단순 입력 문맥으로 사용 중 어느 것이 더 유익한지 불분명 ② 스트리밍 적응 방식: 현재 배치의 감독이 이후 배치 예측 범위와 겹쳐 "성숙하지 않은" 감독 사용 우려 |
| **연구 목적** | 원칙적이고 검증 가능한 단일 프로토콜(성숙한 GT만 사용)을 정립하고, 해당 프로토콜 하에서 효과적인 경량 어댑터를 제안 |

---

## 2. 핵심 주장과 근거 표

| # | 핵심 주장 | 근거 | 근거 유형 | 위치 |
|---|-----------|------|-----------|------|
| 1 | POGT 기반 적응이 단순한 직접 예측(후기 샘플)보다 일관되게 우월하지 않다 | 동일 미니배치 내 첫 샘플의 조정 예측 vs. 마지막 샘플의 직접 예측 MSE 비교 | 실증 분석 | Fig. 2b, Eq. 10-11 |
| 2 | 스트리밍 방식의 감독은 이후 배치의 예측 범위와 겹칠 수 있어 "성숙하지 않음" | $H \geq B_k + B_{k+1}$일 때 감독-예측 범위 교집합이 공집합이 아님을 수식으로 증명 | 수학적 증명 | Eq. 15, p.7 |
| 3 | 성숙한 GT만으로도 충분한 적응 신호가 된다 | 원래 TAFAS(혼합 감독) vs. Matured-only TAFAS MSE 비교 | 실험 비교 | Table 4, Table A.1 |
| 4 | 기존 어댑터의 주파수 도메인 보정은 매끄럽고 구조화되지 않음 | TAFAS, PETSA의 스펙트럼 보정 크기 분포 분석 | 주파수 분석 | Fig. 4, Fig. A.2 |
| 5 | FAC는 경쟁력 있는 성능을 더 적은 파라미터로 달성 | MSE 비교(Table 1), 파라미터 수 비교(Table 2) | 실험 결과 | Table 1, 2, 3 |
| 6 | 입력+출력 보정이 출력만의 보정보다 일반적으로 우월 | 입력+출력 vs. 출력만 MSE 비교 | 절제 실험 | Table A.2 |

---

## 2-1. 상세 설명

### 해결하고자 하는 문제

1. **프로토콜 불순함(Protocol Impurity)**: 기존 TSF-TTA는 POGT(부분 관측 정답)를 혼합하거나 성숙하지 않은 정답을 감독 신호로 사용하여 평가 기준이 이질적임.
2. **주파수 도메인 보정의 비효율**: 기존 어댑터들은 시간 도메인에서 보정하거나, 주파수 손실을 보조 항으로만 활용하여 스펙트럼 구조화된 보정이 약함.

---

### 제안하는 방법 및 수식

#### ① 롤링 TSF-TTA 공식화 (Eq. 1-3)

$$\mathbf{X}_k^{[j]} = [\mathbf{x}_{t_k+j-L}, \ldots, \mathbf{x}_{t_k+j-1}], \quad \hat{\mathbf{Y}}_k^{[j]} = [\hat{\mathbf{y}}^{[j]}_{t_k+j}, \ldots, \hat{\mathbf{y}}^{[j]}_{t_k+j+H-1}]$$

- $L$: 룩백 윈도우 길이 (look-back length)
- $H$: 예측 지평선 (forecasting horizon)
- $k$: 미니배치 인덱스
- $t_k$: $k$번째 미니배치의 첫 예측 대상 직전 글로벌 시간 인덱스
- $\mathbf{x}_t \in \mathbb{R}^C$: 시간 $t$에서의 다변량 관측값 ($C$: 변수 수)
- $\hat{\mathbf{y}}^{[j]}_t \in \mathbb{R}^C$: $j$번째 샘플의 시간 $t$에서의 예측값

> 💡 **미니배치(Mini-batch)**: 연속된 롤링 윈도우들을 묶어 한 번에 처리하는 단위. 배치 크기 $B_k$만큼 연속 샘플이 포함됨.

#### ② 성숙한 미니배치 집합 정의 (Eq. 16)

$$\mathcal{M}_k := \{m < k : t_m + B_m + H - 1 \leq t_k\}$$

- $\mathcal{M}_k$: 시간 $k$에서 완전히 관측 완료된(성숙한) 미니배치들의 인덱스 집합
- $B_m$: $m$번째 미니배치의 크기
- 조건 해석: 미니배치 $m$의 모든 샘플의 예측 지평선이 현재 $k$번째 배치 시작 전에 완전히 관측 완료됨

#### ③ 원칙적 적응 목적 함수 (Eq. 17-18)

$$\mathcal{L}_k^{\text{clean}} = \sum_{m \in \mathcal{S}_k} \omega_{k,m} \mathcal{L}_m^{\text{matured}}, \quad \mathcal{S}_k \subseteq \mathcal{M}_k$$

$$\mathcal{L}_m^{\text{matured}} = \frac{1}{B_m} \sum_{j=1}^{B_m} \text{MSE}(\hat{\mathbf{Y}}_m^{[j]}, \mathbf{Y}_m^{[j]})$$

- $\mathcal{S}_k$: 적응에 사용할 성숙 미니배치들의 부분집합
- $\omega_{k,m} \geq 0$: 결합 가중치

#### ④ FAC 입력 FreqGCM (Eq. 19-20)

$$\boldsymbol{\mathcal{X}}_k^{[j]} = \text{FFT}\left(\mathbf{X}_k^{[j]}\right)$$

$$\Delta\boldsymbol{\mathcal{X}}_k^{[j]} = \boldsymbol{\mathcal{X}}_k^{[j]} \odot \mathbf{W}_{\text{in}} + \mathbf{B}_{\text{in}}$$

$$\mathbf{X}_k^{\text{FAC},[j]} = \mathbf{X}_k^{[j]} + \tanh(\alpha_{\text{in}}) \cdot \text{iFFT}\left(\Delta\boldsymbol{\mathcal{X}}_k^{[j]}\right)$$

- $\mathbf{W}_{\text{in}} \in \mathbb{C}^{L/2+1}$: 복소 아핀 마스크의 곱셈 가중치 (원소별 스케일 및 위상 이동)
- $\mathbf{B}_{\text{in}} \in \mathbb{C}^{L/2+1}$: 복소 아핀 마스크의 덧셈 편향 (근-영 성분 보정)
- $\alpha_{\text{in}}$: 학습 가능한 게이트 스칼라 (보정 크기 전체 제어)
- $\odot$: 원소별 곱셈 (Hadamard product)
- $\tanh(\alpha_{\text{in}})$: 보정 크기를 $(-1, 1)$ 범위로 제한하는 게이트

> 💡 **FFT/iFFT (고속 푸리에 변환/역변환)**: 시간 도메인 신호를 주파수 성분으로 분해하거나 복원하는 알고리즘. 실제 구현에서는 rFFT/irFFT를 사용하여 실수 출력을 보장.
>
> 💡 **복소 아핀 마스크 (Complex Affine Mask)**: 각 푸리에 계수에 복소수 곱셈(위상 이동 + 크기 조정)과 복소수 덧셈(영 근방 성분 보정)을 적용하는 연산.
>
> 💡 **잔차 형식 (Residual Form)**: 원본 신호에 보정량을 더하는 방식으로, 보정이 작을 때 원본 신호를 보존함.

#### ⑤ FAC 출력 FreqGCM (Eq. 21)

$$\hat{\boldsymbol{\mathcal{Y}}}_k^{[j]} = \text{FFT}\left(\hat{\mathbf{Y}}_k^{[j]}\right)$$

$$\Delta\hat{\boldsymbol{\mathcal{Y}}}_k^{[j]} = \hat{\boldsymbol{\mathcal{Y}}}_k^{[j]} \odot \mathbf{W}_{\text{out}} + \mathbf{B}_{\text{out}}$$

$$\hat{\mathbf{Y}}_k^{\text{FAC},[j]} = \hat{\mathbf{Y}}_k^{[j]} + \tanh(\alpha_{\text{out}}) \cdot \text{iFFT}\left(\Delta\hat{\boldsymbol{\mathcal{Y}}}_k^{[j]}\right)$$

- $\mathbf{W}\_{\text{out}}, \mathbf{B}_{\text{out}}$: 출력 FreqGCM의 복소 아핀 파라미터
- $\alpha_{\text{out}}$: 출력 게이트 스칼라

#### ⑥ FAC 적응 손실 (Eq. 22)

$$\mathcal{L}_{m(k)}^{\text{FAC}} = \text{MSE}\left(\{\hat{\mathbf{Y}}_{m(k)}^{\text{FAC},[j]}\}_{j=1}^{B_{m(k)}}, \{\mathbf{Y}_{m(k)}^{[j]}\}_{j=1}^{B_{m(k)}}\right)$$

- $m(k)$: 단계 $k$에서 가장 최근에 성숙한 미니배치 인덱스

---

### 모델 구조

```
현재 미니배치 입력 X_k
        ↓
[1] PAAS (선택적): 주기 추정 → 미니배치 크기 결정
        ↓
[2] Matured-GT-Only 적응:
    성숙한 배치 X_{m(k)} → Input FreqGCM → 동결된 소스 예측기 → Output FreqGCM
    → MSE(FAC 예측, 성숙 정답) → 파라미터 업데이트
        ↓
[3] 업데이트된 FAC로 현재 배치 재예측:
    X_k → (업데이트된 Input FreqGCM) → 소스 예측기 → (업데이트된 Output FreqGCM) → Ŷ_k^FAC
```

> 💡 **PAAS (Periodicity-Aware Adaptation Scheduling)**: 시계열의 주기성을 감지하여 미니배치 크기를 결정하는 스케줄링 방법 (TAFAS에서 도입).
>
> 💡 **동결된 소스 예측기 (Frozen Source Forecaster)**: 사전 훈련 후 파라미터를 고정한 예측 모델. FAC는 이 모델을 수정하지 않고 전후에 경량 모듈을 추가.

---

### 파라미터 스케일링 비교 (Table 3)

| 어댑터 | 파라미터 스케일링 |
|--------|-----------------|
| TAFAS | $O(C(L^2 + H^2))$ |
| TAFAS (PatchTST) | $O(L^2 + H^2)$ |
| PETSA | $O(Cr(L + H))$ |
| **FAC (제안)** | $\mathbf{O(C(L + H))}$ |

- $C$: 변수(채널) 수, $L$: 룩백 길이, $H$: 예측 지평선, $r$: 저랭크 차원

> 💡 **저랭크 적응 (Low-Rank Adaptation, LoRA)**: 고차원 행렬을 두 저차원 행렬의 곱으로 근사하여 훈련 파라미터 수를 줄이는 기법. PETSA에서 활용.

---

### 성능 향상 및 한계

**성능 향상** (Table 1, Table 2):
- FAC는 대부분의 설정에서 최고 또는 2위 MSE 달성
- 파라미터 수: FAC 2,758 ~ 34,482 (ETT H=96~720) vs. TAFAS 130,382 ~ 11,097,114

**한계**:
1. **피드백 지연**: 미니배치 전체가 성숙할 때까지 기다려야 하므로, 긴 지평선/큰 배치에서 지연 발생 (p.13)
2. **런타임 오버헤드**: FFT/iFFT 연산으로 파라미터 효율이 항상 속도 효율로 이어지지 않음 (Table A.4)
3. **단일 성숙 배치**: 현재 구현은 가장 최근 성숙 배치 하나만 사용하여 히스토리 활용 제한

---

## 3. 각 주장별 페이지/Figure/Table 번호

| 주장 | 위치 |
|------|------|
| POGT 혼합 감독 프로토콜 분석 | p.4-5, Eq. 4-9 |
| POGT 사용이 직접 예측보다 일관되게 우월하지 않음 | p.5-6, Fig. 2 (p.6), Eq. 10-11 |
| 스트리밍 적응의 비성숙 감독 문제 | p.7, Eq. 14-15 |
| 성숙 GT 기반 원칙적 프로토콜 정의 | p.7, Eq. 16-18 |
| FAC 구조 설계 | p.8-9, Fig. 3 (p.8), Eq. 19-22 |
| 주요 성능 결과 | p.10, Table 1 |
| 파라미터 효율 비교 | p.11, Table 2, Table 3 |
| POGT 필요성 분석 (실험적) | p.11, Table 4 |
| 주파수 도메인 보정 분석 | p.12, Fig. 4 |
| 결론, 한계, 미래 연구 | p.13 |
| 입력+출력 vs. 출력만 비교 | p.17-18, Table A.2, Table A.3 |
| 런타임 비교 | p.19, Table A.4 |

---

## 4. 저자 보고 결과 vs. 해석 분리

### 저자가 직접 보고한 결과

**[방법론]**
- FAC는 입력/출력 FreqGCM을 통해 주파수 도메인에서 직접 보정을 파라미터화함 (p.8-9)
- 적응 신호: 오직 성숙한 GT만 사용 ($\mathcal{L}^{\text{clean}}_k$, Eq. 17)
- 파라미터 스케일링: $O(C(L+H))$ (Table 3)

**[수치 결과]**
- ETTh1, H=96, DLinear: Base 0.4695 → FAC 0.4554 (MSE 감소) (Table 1)
- Weather, H=720, DLinear: FAC 파라미터 수 34,482 (TAFAS 11,097,114의 약 1/322) (Table 2)
- 원래 TAFAS vs. Matured-only TAFAS on ETTh1, H=720, DLinear: 0.6820 vs. 0.6926 (Table 4) — 원래 TAFAS가 약간 우세
- Weather, H=720, DLinear: TAFAS 적응 시간 $3.60 \pm 0.25$ ms, FAC $4.25 \pm 0.41$ ms (Table A.4)

**[주파수 분석]**
- TAFAS, PETSA는 주파수에 따라 완만하게 변하는 보정 스펙트럼을 보임
- FAC는 특정 주파수 성분에서 국소화된 피크를 보임 (Fig. 4, Fig. A.2)

---

### 분석자(나)의 해석

**[긍정적 해석]**
- FAC의 파라미터 효율( $O(C(L+H))$ )은 긴 지평선 설정에서 특히 유리하며, 이는 실시간 배포 시나리오에서 중요한 실용적 장점임
- "성숙한 GT만 사용" 프로토콜은 데이터 누수(data leakage) 방지 측면에서 방법론적으로 더 엄격하고 재현 가능한 평가 기준을 제공함

**[유보적 해석]**
- Table 4에서 원래 TAFAS(POGT 포함)가 일부 설정에서 Matured-only TAFAS보다 낮은 MSE를 달성하는 경우가 있음. 저자는 이를 "일관되지 않다"고 표현하지만, H=720에서 DLinear 기준 0.6820 vs. 0.6926으로 약 1.5% 차이가 있어 응용에 따라 무시하기 어려울 수 있음
- FAC의 런타임이 PatchTST 기준 $18.16 \pm 0.29$ ms로 TAFAS의 $2.97 \pm 0.12$ ms 대비 약 6배 느린 점은 단순히 "FFT 오버헤드" 로 설명되고 있으나, 실시간 예측이 중요한 응용에서는 중요한 단점임

**[불확실한 해석]**
- FAC의 "국소화된 주파수 피크"가 실제로 더 정확한 예측에 기여하는지, 아니면 단순히 더 공격적인 보정인지는 현재 실험만으로는 완전히 구분하기 어려움 (저자들도 p.12-13에서 이를 명시적으로 인정함)

---

## 5. 통계적으로 취약한 부분과 비교 불가능한 수치

| 항목 | 취약점 | 설명 |
|------|--------|------|
| ⚠️ **단일 시드 실험** | 통계적 유의성 검정 없음 | Table 1의 MSE 비교에 표준편차, p-value, confidence interval이 제시되지 않아 차이의 유의성을 판단하기 어려움 |
| ⚠️ **COSA 미포함** | 불완전한 비교 | Im & Kwon (2026)의 COSA는 "설계 변경이 필요하다"는 이유로 주요 실험에서 제외됨 (각주 2, p.10). 이는 가장 최신 경쟁 방법과의 직접 비교가 없음을 의미 |
| ⚠️ **DynaTTA 미포함** | 불완전한 비교 | Grover & Etemad (2025)의 DynaTTA도 본 프로토콜 하 비교 대상에 포함되지 않음 |
| ⚠️ **런타임 측정 환경 이질성** | 비교 가능성 제한 | Table A.4는 RTX 4070 Super/4090 혼용 환경에서 측정되었으며, 어떤 GPU를 각 방법에 사용했는지 명시되지 않음 |
| ⚠️ **소규모 수치 차이** | 실용적 유의성 불명확 | 예: ETTm2, H=96, DLinear에서 FAC(0.1560) vs. PETSA(0.1583)의 차이 0.0023은 실용적 유의성이 낮을 수 있음 |
| ⚠️ **Exchange H=720 제외** | 비대칭 평가 | Exchange 데이터셋에서 H=720은 성숙 배치가 없어 제외됨 — 특정 조건에서 방법이 적용 불가한 제약 |
| ⚠️ **MICN 제외** | 아키텍처 다양성 제한 | "아키텍처 분류가 모호하다"는 이유로 제외되었으나, 멀티스케일 합성곱 기반 모델과의 비교가 없음 (각주 1, p.9) |

---

## 6. 논문이 답하지 않는 질문

1. **FAC의 적응 속도(convergence)**: 몇 번의 업데이트 스텝 이후 안정적인 성능에 도달하는가? 초기 수렴 곡선이 제시되지 않음.

2. **다중 성숙 배치 활용의 효과**: Eq. 17은 다수의 성숙 배치를 사용할 수 있도록 일반화되어 있지만, 실제로 가장 최근 배치 하나만 사용함. 여러 배치를 조합할 때의 성능 변화가 제시되지 않음.

3. **분포 변화 강도에 따른 성능 변화**: 약한/강한 분포 변화 상황별로 FAC의 효과가 어떻게 달라지는지 체계적으로 분석되지 않음.

4. **단변량 시계열에서의 성능**: 모든 실험이 다변량 설정에서 이루어지며, 단변량 TSF-TTA 시나리오에서의 효과는 검증되지 않음.

5. **하이퍼파라미터 민감도**: $\alpha_{\text{in}}, \alpha_{\text{out}}$의 초기값, 학습률 등의 하이퍼파라미터 민감도 분석이 부재.

6. **FAC가 특정 주파수 성분을 선택하는 이유**: 어떤 데이터 특성이 특정 주파수 피크로 이어지는지 해석 가능성(interpretability) 분석이 없음.

7. **더 긴 테스트 시퀀스에서의 안정성**: 매우 긴 테스트 시퀀스에서 어댑터의 파라미터가 누적 업데이트로 인해 과적합되는지 여부.

8. **COSA와의 공정 비교**: 설계 변경 없이 COSA를 성숙 GT 프로토콜에 맞추는 방법이 있는지, 그 성능이 어떻게 되는지.

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p.2): 적응 감독 프로토콜 비교

**해석**: 두 가지 기존 TSF-TTA 프로토콜의 시각적 비교.
- **(a) 혼합 감독**: 현재 배치의 POGT(파란색)와 과거 성숙 배치의 GT를 함께 사용. 첫 번째 샘플의 예측이 POGT를 기반으로 조정(빨간색)됨.
- **(b) 스트리밍 적응**: 현재 배치의 전체 예측 범위(빨간 빗금)가 감독 신호로 사용됨.

**핵심 메시지**: 두 방식 모두 현재 배치의 관측값을 감독으로 사용한다는 공통점이 있으며, 이는 프로토콜 수준의 문제를 내포함. 스트리밍 방식에서 현재 배치의 감독이 이후 배치의 예측 범위와 겹칠 수 있음.

---

### Figure 2 (p.6): POGT 조정 예측 vs. 직접 예측 비교

**해석**: 미니배치 내 샘플 위치별 평균 겹침 영역 MSE를 보여줌.
- **파란 곡선(Before TAFAS)**: 적응 전 직접 예측의 MSE. 배치 내 후기 샘플일수록(더 많은 관측 문맥 보유) MSE가 낮아지는 경향.
- **주황 곡선(After TAFAS)**: TAFAS 적응 후 예측의 MSE. 전반적으로 개선되지만, 후기 직접 예측보다 나쁜 경우가 많음.
- **빨간 점선**: 첫 번째 샘플의 조정 예측 MSE — 대부분의 경우 후기 직접 예측보다 높음.
- **녹색 점선**: 마지막 샘플의 직접 예측 MSE — 대부분의 경우 첫 번째 샘플의 조정 예측보다 낮음.

**핵심 메시지**: POGT를 감독으로 사용한 조정이 단순히 더 많은 과거 데이터를 입력 문맥으로 사용한 직접 예측보다 일관되게 우월하지 않음. 이는 POGT 기반 적응의 원칙적 필요성에 의문을 제기.

> ⚠️ **통계적 취약점**: 이 비교는 $B_k=97$인 배치만을 대상으로 하며, 통계적 유의성 검정이 없음.

---

### Figure 3 (p.8): FAC 전체 구조 개요

**해석**: FAC의 세 단계 파이프라인을 보여줌.
1. **[선택] PAAS**: 주기 추정으로 미니배치 크기 결정
2. **성숙 GT 적응**: 최근 성숙 배치를 가져와 입력 FreqGCM → 동결 예측기 → 출력 FreqGCM → MSE 손실 → 파라미터 업데이트
3. **재예측**: 업데이트된 FAC로 현재 배치 재예측

**핵심 메시지**: FAC는 TAFAS/PETSA와 유사하게 입력/출력 양쪽에 보정 모듈을 배치하지만, 각 모듈이 시간 도메인 대신 주파수 도메인에서 직접 작동함. 전체 설계가 "동결 예측기를 수정하지 않는다"는 TTA 원칙을 유지.

---

### Figure 4 (p.12): 주파수 도메인 보정 스펙트럼 (H=720)

**해석**: 적응 전후 예측 차이를 rFFT로 분석한 스펙트럼.
- **TAFAS(주황), PETSA(녹색)**: 저주파에서 고주파로 갈수록 완만하게 감소하는 스무스한 스펙트럼. 주파수 선택성이 낮음.
- **FAC(파란)**: 특정 주파수 위치에서 뚜렷한 피크를 보이며, 더 국소화된 스펙트럼 보정 패턴.

**핵심 메시지**: FAC의 원소별 복소 아핀 마스크 설계가 개별 푸리에 계수를 독립적으로 조정할 수 있게 하여, 더 구조화된 주파수 선택적 보정을 가능하게 함.

> ⚠️ **해석 주의**: 저자들이 명시적으로 언급하듯, 보정 크기가 크다고 예측이 더 정확한 것은 아님. 이 분석은 보정의 구조적 특성을 보여주는 것이지, 성능 측정이 아님.

---

### Figure A.2 (p.20): 예측 지평선별 추가 주파수 스펙트럼

**해석**: H∈{96, 192, 336, 720} 모든 지평선에 걸쳐 5개 데이터셋-예측기 조합의 스펙트럼을 보여줌.
- FAC의 국소화된 피크 패턴은 단일 H=720뿐 아니라 다양한 지평선에서 일관되게 나타남.
- Weather+PatchTST 조합에서는 세 방법 간 차이가 상대적으로 작음 — 데이터셋과 소스 예측기에 따라 스펙트럼 특성이 달라짐을 시사.
- 짧은 지평선(H=96)에서는 전반적으로 보정 크기가 크고, 긴 지평선에서는 구조가 더 복잡해지는 경향.

**핵심 메시지**: FAC의 주파수 선택적 보정 패턴은 특정 설정에 국한되지 않으며 일반적으로 관찰됨. 단, 그 정도는 소스 예측기와 데이터셋에 따라 달라짐.

---

## 8. 결론 및 후속 연구

### 8-1. 연구자들이 제시한 시사점 및 후속 연구 계획

**시사점** (p.13):
1. 성숙한 GT 기반의 원칙적 프로토콜이 TSF-TTA 평가의 표준으로 자리잡을 수 있음
2. 주파수 도메인 직접 파라미터화가 경량 적응의 효과적인 설계 방향임
3. POGT 없이도 충분한 적응이 가능함

**저자 제시 미래 연구 방향** (p.13):
1. **부분 성숙 활용**: 미니배치 전체가 성숙하기 전에 이미 관측 가능한 타겟을 활용하는 더 유연한 구현
2. **다중 성숙 배치 활용**: Eq. 17의 일반화된 형태로 여러 성숙 배치를 가중 결합
3. **캐시된 주파수 표현**: 과거 성숙 배치의 FFT 표현을 캐시하여 반복 연산 감소 및 런타임 효율 개선

---

### 8-1. 모델의 일반화 성능 향상 가능성 (중점)

#### 현재 FAC의 일반화 제한 요인

| 제한 요인 | 상세 설명 |
|-----------|-----------|
| **단일 성숙 배치 의존** | 가장 최근 성숙 배치 하나만 사용하므로, 국소적 분포 변화에 과적합 위험 |
| **피드백 지연** | 긴 지평선에서 성숙 지연으로 인해 현재 분포 변화에 늦게 반응 |
| **고정 복소 마스크 구조** | $\mathbf{W}\_{\text{in}}, \mathbf{B}_{\text{in}}$이 모든 주파수 성분에 동일한 형식 적용 |

#### 일반화 성능 향상을 위한 방향

**① 다중 성숙 배치 가중 결합**

$$\mathcal{L}_k^{\text{clean}} = \sum_{m \in \mathcal{S}_k} \omega_{k,m} \mathcal{L}_m^{\text{matured}}$$

가중치 $\omega_{k,m}$을 시간적 거리, 분포 유사도 등으로 설계하면 더 안정적인 적응이 가능. 예를 들어 지수 감소 가중치 $\omega_{k,m} \propto \exp(-\lambda(k-m))$를 도입하면 최근 배치를 더 중시하면서도 과거 정보를 활용 가능.

**② 메타 학습 기반 초기화 (MAML 스타일)**

소스 예측기 훈련 단계에서 FAC의 초기 파라미터를 "빠른 적응에 유리한" 초기점으로 메타 학습하면, 테스트 시 소수의 성숙 배치로도 빠른 적응이 가능.

**③ 주파수 선택적 마스킹 (Sparse Frequency Selection)**

모든 푸리에 성분을 보정하는 현재 방식 대신, 도메인 지식을 활용하여 중요한 주파수 대역만 선택적으로 보정하는 희소(sparse) 마스크를 사용하면 과적합 위험 감소 및 일반화 향상.

**④ 채널 간 상관관계 활용**

현재 FAC는 원소별 연산으로 채널 간 독립성을 가정하나, 다변량 시계열에서 채널 간 상관관계를 주파수 도메인에서 포착하는 그룹 주파수 마스크를 도입하면 더 풍부한 구조 학습 가능.

**⑤ 불확실성 추정 기반 적응 강도 제어**

예측의 불확실성이 높은 상황에서 $\tanh(\alpha)$ 게이트를 더 보수적으로 유지하는 적응적 게이트 스케줄링을 통해 안정적인 일반화 보장.

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 아래 비교는 논문에 인용된 문헌들과 일반적으로 알려진 연구 동향을 기반으로 작성되었으며, 일부 2025-2026년 논문은 프리프린트 상태이거나 독립적 확인이 어렵습니다.

#### 연구 흐름 분류

```
[분포 변화 대응 방법론]
    ├── 훈련 시간 정규화
    │     ├── RevIN (Kim et al., 2022, ICLR)
    │     ├── Dish-TS (Fan et al., 2023, AAAI)
    │     └── SAN (Liu et al., 2023, NeurIPS)
    ├── 온라인 예측
    │     ├── OneNet (Zhang et al., 2023, NeurIPS)
    │     ├── Act-Now (Liang et al., 2024)
    │     ├── DSOF (Lau et al., 2025, ICLR)
    │     └── ADAPT-Z (Huang et al., 2026, ICLR)
    └── 테스트 타임 적응 (TSF-TTA) ← 본 논문의 영역
          ├── TAFAS (Kim et al., 2025, AAAI)
          ├── PETSA (Medeiros et al., 2025, ICML Workshop)
          ├── DynaTTA (Grover & Etemad, 2025, ICML Workshop)
          ├── COSA (Im & Kwon, 2026, ICLR)
          └── FAC [본 논문] (Wang et al., 2026)
```

#### 주요 연구 비교표

| 방법 | 연도 | 핵심 아이디어 | 파라미터 효율 | 프로토콜 명확성 | 주파수 활용 |
|------|------|---------------|--------------|----------------|------------|
| RevIN | 2022 | 역방향 인스턴스 정규화 | 높음 | N/A (훈련 시) | ✗ |
| OneNet | 2023 | 온라인 앙상블 | 중간 | 낮음 (온라인) | ✗ |
| TAFAS | 2025 | POGT+성숙 GT 혼합, PAAS | 낮음 ( $O(C(L^2+H^2))$ ) | 낮음 (혼합 감독) | 부분 (보조 손실) |
| PETSA | 2025 | 저랭크 어댑터 + 주파수 손실 | 중간 ( $O(Cr(L+H))$ ) | 낮음 (혼합 감독) | 부분 (손실 항) |
| DynaTTA | 2025 | 동적 적응률, 변화 조건 게이팅 | 미공개 | 낮음 | ✗ |
| COSA | 2026 | 출력 공간 스트리밍 어댑터 | 중간 | 낮음 (스트리밍) | ✗ |
| **FAC** | **2026** | **주파수 직접 파라미터화 + 성숙 GT** | **매우 높음 ( $O(C(L+H))$ )** | **높음 (성숙 GT만)** | **직접 (파라미터화)** |

> ⚠️ **비교 주의사항**: DynaTTA, COSA, ADAPT-Z 등 2025-2026년 논문들은 현재 arXiv 또는 워크샵 논문으로, 최종 출판 전 결과일 수 있습니다.

#### 본 논문이 앞으로의 연구에 미치는 영향

**긍정적 영향**:
1. **프로토콜 표준화**: "성숙한 GT만 사용" 원칙은 TSF-TTA 연구의 공정한 비교 기반을 제공. 향후 연구에서 이 프로토콜을 표준 벤치마크로 채택할 가능성이 높음.
2. **주파수 도메인 TTA 방향 제시**: 주파수 도메인에서 직접 어댑터를 설계하는 아이디어는 향후 다양한 확장 연구의 출발점이 될 수 있음.
3. **경량 어댑터 설계 기준 제시**: $O(C(L+H))$ 스케일링이 시간 도메인 대비 어떤 장점을 갖는지 명확하게 보여줌.

**제한적 영향**:
1. COSA와의 공정 비교 부재는 후속 연구에서 반드시 보완되어야 할 갭.
2. 통계적 유의성 검증의 부재는 주장의 신뢰도를 제한함.

#### 앞으로 연구 시 고려해야 할 사항

| 고려 사항 | 세부 내용 |
|----------|-----------|
| **통계적 엄밀성** | 다중 랜덤 시드, bootstrap confidence interval, 유의성 검정 필수화 |
| **프로토콜 통일** | 본 논문이 제안한 "성숙 GT만 사용" 프로토콜 또는 이에 준하는 명확한 기준을 명시해야 함 |
| **실제 배포 환경 고려** | 지연(latency), 메모리, 에너지 효율 등 실용적 지표를 포함한 종합 평가 |
| **다양한 분포 변화 유형** | 점진적/급격/계절적/주기적 변화 등 다양한 비정상성 유형별 성능 분석 |
| **소스 예측기 다양화** | 현재 5개 예측기 외 최신 기초 모델(foundation model) 기반 예측기와의 결합 가능성 탐색 |
| **해석 가능성** | 주파수 선택적 보정이 어떤 실제 시계열 패턴(계절성, 트렌드 등)에 대응하는지 해석 연구 |
| **온라인-TTA 통합** | OTSF와 TSF-TTA의 경계를 명확히 하거나 통합하는 이론적 프레임워크 개발 |
| **멀티모달 확장** | 외부 공변량(날씨, 경제 지표 등)을 주파수 도메인 보정에 통합하는 방향 |

---

## 참고 자료

1. **본 논문**: Wang, H., Xu, R., Kementzidis, G., Cho, K., Ramirez Villarreal, S., & Deng, Y. (2026). "Towards Principled Test-Time Adaptation for Time Series Forecasting." arXiv:2605.17250v1.

2. **TAFAS**: Kim, H., Kim, S., Mok, J., & Yoon, S. (2025). "Battling the Non-Stationarity in Time Series Forecasting via Test-Time Adaptation." *AAAI 2025*, 39(17):17868–17876.

3. **PETSA**: Medeiros, H. R., Sharifi-Noghabi, H., Oliveira, G. L., & Irandoust, S. (2025). "Accurate Parameter-Efficient Test-Time Adaptation for Time Series Forecasting." *ICML 2025 Workshop*.

4. **DynaTTA**: Grover, S., & Etemad, A. (2025). "Shift-Aware Test Time Adaptation and Benchmarking for Time-Series Forecasting." *ICML 2025 Workshop*.

5. **COSA**: Im, J., & Kwon, H.-Y. (2026). "COSA: Context-Aware Output-Space Adapter for Test-Time Adaptation in Time Series Forecasting." *ICLR 2026*.

6. **RevIN**: Kim, T., Kim, J., Tae, Y., Park, C., Choi, J.-H., & Choo, J. (2022). "Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift." *ICLR 2022*.

7. **iTransformer**: Liu, Y., et al. (2024). "iTransformer: Inverted Transformers are Effective for Time Series Forecasting." *ICLR 2024*.

8. **PatchTST**: Nie, Y., et al. (2023). "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers." *ICLR 2023*.

9. **DLinear**: Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). "Are Transformers Effective for Time Series Forecasting?" *AAAI 2023*.

10. **FreTS**: Yi, K., et al. (2023). "Frequency-domain MLPs are More Effective Learners in Time Series Forecasting." *NeurIPS 2023*.

11. **DSOF**: Lau, Y.-Y. A., Shao, Z., & Yeung, D.-Y. (2025). "Fast and Slow Streams for Online Time Series Forecasting without Information Leakage." *ICLR 2025*.

12. **Liang et al. (2025) TTA Survey**: Liang, J., He, R., & Tan, T. (2025). "A Comprehensive Survey on Test-Time Adaptation under Distribution Shifts." *International Journal of Computer Vision*, 133(1):31–64.
