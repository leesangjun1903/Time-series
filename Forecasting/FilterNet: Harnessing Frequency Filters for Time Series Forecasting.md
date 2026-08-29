# FilterNet: Harnessing Frequency Filters for Time Series Forecasting

---

## ⚠️ 사전 고지

본 분석은 제공된 PDF 원문(arXiv:2411.01623v2)에 기반합니다. 논문에 명시되지 않은 내용은 추측하지 않고 명확히 표시합니다. 2020년 이후 최신 연구 비교는 제 학습 데이터(2024년 초까지) 기반이며, 일부 정보는 불완전할 수 있습니다.

---

## 1. Executive Summary (10문장 이내)

FilterNet은 신호 처리(signal processing)의 주파수 필터(frequency filter) 개념을 시계열 예측(time series forecasting)에 직접 적용한 심층 학습 프레임워크이다.  
기존 Transformer 기반 모델들이 고주파(high-frequency) 신호에 취약하고 계산 비효율적이며 전체 주파수 스펙트럼을 활용하지 못한다는 문제를 해결하고자 한다.  
FilterNet은 두 종류의 학습 가능한 주파수 필터, 즉 **Plain Shaping Filter(PaiFilter)**와 **Contextual Shaping Filter(TexFilter)**를 핵심 구성요소로 제안한다.  
PaiFilter는 무작위 초기화된 학습 가능한 파라미터를 통해 신호를 필터링하며, TexFilter는 입력 데이터에 의존적인 필터를 동적으로 생성한다.  
두 필터는 각각 MLP와 Transformer의 선형·어텐션 연산을 근사적으로 대체할 수 있음을 이론적으로 설명한다.  
인스턴스 정규화(instance normalization)와 고속 푸리에 변환(FFT)을 통해 비정상성(non-stationarity) 문제를 완화하고 주파수 도메인 변환을 수행한다.  
8개의 실제 벤치마크 데이터셋에서 실험한 결과, FilterNet은 효과성과 효율성 모두에서 최신 기법들을 능가하는 성능을 보인다.  
모델의 계산 복잡도는 $\mathcal{O}(\log L)$로 Transformer( $\mathcal{O}(L^2)$ ) 대비 뛰어난 효율을 달성한다.  
특히 소규모 데이터셋에서는 PaiFilter가, 대규모 데이터셋에서는 TexFilter가 강점을 보인다.  
본 연구는 신호처리 기반 접근법이 딥러닝 시계열 예측에서 유망한 방향임을 제시한다.

### 1-1. 연구의 목적과 필요성

**[필요성 — Abstract, p.1; Introduction, p.2]**

| 문제 | 설명 |
|------|------|
| 고주파 취약성 | Transformer 기반 모델은 고주파 신호를 제대로 처리하지 못함 (Figure 1: iTransformer MSE=1.1e-01 vs FilterNet MSE=2.7e-05) |
| 계산 비효율성 | Self-attention의 $\mathcal{O}(L^2)$ 복잡도로 인해 긴 시계열에 부적합 |
| 전체 스펙트럼 미활용 | 기존 모델들은 저주파 성분 위주로 처리하여 중·고주파 정보 손실 |
| 비정상성 | 시계열 데이터의 분포 이동(distribution shift)으로 인한 예측 성능 저하 |

> 💡 **비정상성(Non-stationarity)**: 시계열 데이터의 평균, 분산 등 통계적 특성이 시간에 따라 변하는 성질. 예를 들어 에너지 소비 패턴이 계절에 따라 달라지는 경우가 해당됩니다.

> 💡 **주파수 스펙트럼(Frequency spectrum)**: 신호를 주파수 성분으로 분해했을 때 각 주파수의 크기(진폭)를 나타낸 것. 저주파는 천천히 변하는 추세, 고주파는 빠르게 변하는 노이즈/패턴에 해당합니다.

---

## 2. 핵심 주장과 근거 표

| 핵심 주장 | 근거 | 위치 |
|-----------|------|------|
| Transformer는 전체 주파수 스펙트럼을 활용하지 못함 | 합성 신호 실험: iTransformer MSE=1.1e-01, FilterNet MSE=2.7e-05 | Figure 1, p.2 |
| PaiFilter가 MLP 선형 매핑을 근사함 | 합성곱 정리(Convolution Theorem)를 통해 주파수 필터=순환 합성곱 등가 관계 증명 | Eq.(1), p.3 |
| TexFilter가 Transformer 어텐션을 근사함 | 자기 어텐션이 글로벌 순환 합성곱 형태임을 선행 연구[35] 인용 | Appendix B, p.13 |
| 채널 공유(channel-shared) 필터가 채널 고유(channel-unique) 필터보다 우수 | Table 2: ETTh1, Exchange 데이터셋에서 일관된 성능 우위 | Table 2, p.8 |
| FilterNet이 8개 벤치마크에서 SOTA 달성 | Table 1 (L=96), Table 4 (추가 비교), Table 5 (L=336) | Table 1, 4, 5 |
| 계산 효율성 우수 ( $\mathcal{O}(\log L)$ ) | Exchange: 0.5GB, 1.5s/epoch; iTransformer: 0.6GB, 2.6s/epoch | Figure 6, p.9 |
| 인스턴스 정규화가 필수적 | Ablation: W/O Norm 제거 시 성능 하락 | Figure 10, p.17 |
| 전체 통과(all-pass) 필터가 저역 통과(low-pass)보다 효과적 | FITS(저역 통과 필터 기반)보다 일관적으로 우수한 성능 | Table 1, p.7 |

> 💡 **All-pass filter(전체 통과 필터)**: 모든 주파수 성분을 통과시키는 필터. 반면 **Low-pass filter(저역 통과 필터)**는 낮은 주파수만 통과시키고 높은 주파수는 차단합니다.

---

## 2-1. 자세한 논문 설명

### 🔴 해결하고자 하는 문제

**[p.2, Introduction]**

1. **고주파 취약성**: Transformer 모델은 self-attention 특성상 고주파 노이즈에 민감
2. **계산 비효율**: $\mathcal{O}(L^2)$ 복잡도
3. **전체 스펙트럼 미활용**: 저주파 위주 처리로 인한 정보 손실
4. **비정상성 대응 부재**: 분포 이동에 따른 성능 저하

### 🔵 제안하는 방법 (수식 포함)

#### 기본 주파수 필터 정의 [p.3, Eq.(1)]

$$\mathcal{Y}[k] = \mathcal{H}[k]\mathcal{X}[k] \leftrightarrow y[n] = h[n] \circledast x[n]$$

| 기호 | 설명 |
|------|------|
| $\mathcal{X}[k]$ | 입력 시계열 $x[n]$의 푸리에 변환 |
| $\mathcal{H}[k]$ | 주파수 도메인에서의 필터 |
| $\mathcal{Y}[k]$ | 출력 신호의 푸리에 변환 |
| $h[n]$ | $\mathcal{H}[k]$의 역 푸리에 변환 |
| $\circledast$ | 순환 합성곱(circular convolution) 연산 |

> 💡 **합성곱 정리(Convolution Theorem)**: 주파수 도메인에서의 점별 곱셈(point-wise multiplication)은 시간 도메인에서의 합성곱 연산과 동일합니다. 이를 통해 계산을 훨씬 효율적으로 수행할 수 있습니다.

#### 인스턴스 정규화 [p.4, Eq.(2)(3)]

$$\text{Norm}(\mathbf{X}) = \left[\frac{X_i^{1:L} - \text{Mean}_L(X_i^{1:L})}{\text{Std}_L(X_i^{1:L})}\right]_{i=1}^{N}$$

$$\text{InverseNorm}(\mathbf{P}) = \left[P_i^{L+1:L+\tau} \times \text{Std}_L(X_i^{1:L}) + \text{Mean}_L(X_i^{1:L})\right]_{i=1}^{N}$$

| 기호 | 설명 |
|------|------|
| $\mathbf{X} \in \mathbb{R}^{N \times L}$ | 입력 시계열 ($N$: 변수 수, $L$: 룩백 윈도우 길이) |
| $\text{Mean}_L(\cdot)$ | 시간 축 방향 평균 |
| $\text{Std}_L(\cdot)$ | 시간 축 방향 표준편차 |
| $\mathbf{P} \in \mathbb{R}^{N \times \tau}$ | 예측값 |
| $\tau$ | 예측 시간 단계 수 |

#### FilterBlock 공통 정의 [p.5, Eq.(4)]

$$\text{FilterBlock}(\mathbf{Z}) = \mathcal{F}^{-1}(\mathcal{F}(\mathbf{Z}) \mathcal{H}_{filter})$$

| 기호 | 설명 |
|------|------|
| $\mathcal{F}$ | 푸리에 변환(FFT) |
| $\mathcal{F}^{-1}$ | 역 푸리에 변환(IFFT) |
| $\mathcal{H}_{filter}$ | 학습 가능한 주파수 필터 |

#### ① Plain Shaping Filter (PaiFilter) [p.5, Eq.(5)(8)]

$$\text{FilterBlock}(\mathbf{Z}) = \mathcal{F}^{-1}(\mathcal{F}(\mathbf{Z})\mathcal{H}_\phi)$$

$$\mathcal{Z} = \mathcal{F}(\mathbf{Z}), \quad \mathcal{S} = \mathcal{Z} \odot_L \mathcal{H}_\phi, \quad \mathcal{H}_\phi \in \{\mathcal{H}_\phi^{(Uni)}, \mathcal{H}_\phi^{(Ind)}\}, \quad \mathbf{S} = \mathcal{F}^{-1}(\mathcal{S})$$

| 기호 | 설명 |
|------|------|
| $\mathcal{H}_\phi$ | 무작위 초기화 후 학습되는 주파수 필터 파라미터 |
| $\mathcal{H}_\phi^{(Uni)} \in \mathbb{C}^{1 \times L}$ | 채널 공유형(universal) 필터 |
| $\mathcal{H}_\phi^{(Ind)} \in \mathbb{C}^{N \times L}$ | 채널 고유형(individual) 필터 |
| $\odot_L$ | $L$ 차원 방향 원소별 곱(element-wise product) |
| $\mathbf{S} \in \mathbb{R}^{N \times L}$ | PaiFilter 출력 |

#### ② Contextual Shaping Filter (TexFilter) [p.6, Eq.(6)(9)]

$$\text{FilterBlock}(\mathbf{Z}) = \mathcal{F}^{-1}(\mathcal{F}(\mathbf{Z})\mathcal{H}_\varphi(\mathcal{F}(\mathbf{Z})))$$

$$\mathcal{Z} = \mathcal{F}(\mathbf{Z}), \quad \mathcal{E} = \kappa(\mathcal{Z}), \quad \mathcal{H}_\varphi(\mathcal{Z}) = \sigma(\mathcal{E} \odot_D \mathcal{W}_{1:K}), \quad \mathcal{W}_{1:K} = \prod_{i=1}^{K} W_i$$

$$\mathcal{S} = \mathcal{E} \odot_D \mathcal{H}_\varphi(\mathcal{Z}), \quad \mathbf{S} = \mathcal{F}^{-1}(\mathcal{S})$$

| 기호 | 설명 |
|------|------|
| $\mathcal{H}_\varphi(\cdot)$ | 입력 데이터로부터 필터를 생성하는 신경망 |
| $\kappa: \mathbb{C}^L \mapsto \mathbb{C}^D$ | 선형 밀집 임베딩(dense embedding) 연산 |
| $\mathcal{E}$ | 임베딩 결과 |
| $W_{1:K} \in \mathbb{C}^{1 \times D}$ | $K$개의 학습 가능한 복소수 파라미터 |
| $\sigma$ | 활성화 함수 |
| $\odot_D$ | $D$ 차원 방향 원소별 곱 |
| $\mathbf{S} \in \mathbb{R}^{N \times D}$ | TexFilter 출력 |

> 💡 **복소수 파라미터(Complex-valued parameters)**: 실수부와 허수부를 모두 갖는 숫자. 주파수 도메인에서 신호는 진폭(amplitude)과 위상(phase)을 동시에 표현하기 위해 복소수로 표현됩니다.

#### Feed-forward Network (FFN) [p.5, Eq.(7)]

$$\mathbf{P} = \text{FFN}(\mathbf{S}), \quad \hat{\mathbf{Y}} = \text{InverseNorm}(\mathbf{P})$$

| 기호 | 설명 |
|------|------|
| $\mathbf{S}$ | 주파수 필터 블록의 출력 |
| $\mathbf{P}$ | FFN의 출력 (인스턴스 정규화된 예측값) |
| $\hat{\mathbf{Y}}$ | 최종 예측값 |

### 🟢 모델 구조

**[Figure 2, p.4]**

```
입력 X
  ↓
① Instance Normalization (비정상성 제거)
  ↓
② Frequency Filter Block (두 가지 선택)
   ├─ Plain Shaping Filter (PaiFilter): FFT → H_φ 곱셈 → IFFT
   └─ Contextual Shaping Filter (TexFilter): FFT → κ 임베딩 → H_φ(Z) 생성 → 곱셈 → IFFT
  ↓
③ Feed-Forward Network (FFN)
  ↓
④ Inverse Instance Normalization
  ↓
출력 Ŷ
```

> 💡 **FFT (Fast Fourier Transform, 고속 푸리에 변환)**: 시간 도메인 신호를 주파수 도메인으로 변환하는 알고리즘. 계산 복잡도가 $\mathcal{O}(N \log N)$으로 효율적입니다.

### 🟡 성능 향상

**[Table 1, p.7; Figure 6, p.9]**

- **정확도**: 8개 벤치마크 중 대부분에서 SOTA 달성 (Table 1 기준 빨간색/파란색 표시)
- **효율성**: 복잡도 $\mathcal{O}(\log L)$, Exchange 데이터셋에서 1.5s/epoch (iTransformer 2.6s 대비 약 42% 빠름)
- **메모리**: 0.5GB (DLinear와 동일, Transformer류보다 적음)

### 🔴 한계

논문에서 **명시적으로 한계를 기술한 섹션이 없습니다.** 다만 다음을 추론할 수 있습니다:
- Traffic 데이터셋(862개 변수, 주로 저주파)에서는 iTransformer가 더 우수함 (Table 1, p.7) — **논문이 간접적으로 인정**
- 초고차원 다변량(862개 이상) 또는 완전히 불규칙한 신호에 대한 성능 검증 미흡
- 비선형 패턴 포착 능력이 FFN에 의존하는 구조적 한계 (Appendix A, p.13)

---

## 3. 각 주장별 페이지/Figure/Table 번호

| 주장 | 위치 |
|------|------|
| Transformer의 고주파 취약성 | p.2, Figure 1(b)(c) |
| 주파수 필터 = 순환 합성곱 | p.3, Eq.(1) |
| FilterNet 전체 구조 | p.4, Figure 2 |
| 인스턴스 정규화 수식 | p.4, Eq.(2)(3) |
| FilterBlock 정의 | p.5, Eq.(4) |
| PaiFilter 수식 | p.5-6, Eq.(5)(8) |
| TexFilter 수식 | p.6, Eq.(6)(9) |
| 두 필터 구조 비교 | p.5, Figure 3 |
| 주요 실험 결과 | p.7, Table 1 |
| 트렌드/주기 신호 모델링 | p.8, Figure 4 |
| 채널 전략 비교 | p.8, Table 2 |
| 예측 시각화 | p.8, Figure 5 |
| 효율성 분석 | p.9, Figure 6 |
| 주파수 필터 스펙트럼 시각화 | p.9, Figure 7, 8 |
| Ablation Study | p.17, Figure 10 |
| 대역폭(bandwidth) 분석 | p.16-17, Figure 9 |
| 추가 비교 결과 | p.18, Table 4, 5 |

---

## 4. 저자 직접 보고 결과 vs. 검토자 해석

### 저자가 직접 보고한 결과

**연구 주제**:
> "we explore a novel perspective of enlightening signal processing for deep time series forecasting" (p.1, Abstract)

**방법 (저자 직접 기술)**:

$$\text{FilterBlock}(\mathbf{Z}) = \mathcal{F}^{-1}(\mathcal{F}(\mathbf{Z})\mathcal{H}_\phi) \quad \text{[Eq.5, p.5]}$$

$$\text{FilterBlock}(\mathbf{Z}) = \mathcal{F}^{-1}(\mathcal{F}(\mathbf{Z})\mathcal{H}_\varphi(\mathcal{F}(\mathbf{Z}))) \quad \text{[Eq.6, p.5]}$$

**성능 (저자 직접 보고)**:
- "The average improvement of FilterNet over all baseline models is **statistically significant at the confidence of 95%**" (p.7, Section 5.2)
- "FilterNet surpasses other Transformer models, regardless of dataset size" (p.9, Section 5.3)
- 복잡도: $\mathcal{O}(\log L)$ (p.9, Section 5.3)

**한계 (저자 직접 보고)**:
- "iTransformer [17], which achieves the best results on the Traffic dataset (862 variables) but not on smaller datasets" (p.7) — FilterNet이 Traffic에서 최고 성능이 아님을 간접 인정

### 검토자(본 분석)의 해석

- **⚠️ 95% 신뢰도 주장의 통계적 불명확성**: 어떤 검정 방법(t-test, Wilcoxon 등)을 사용했는지 본문에 명시되어 있지 않습니다. (→ Section 5 통계적 취약성 참조)
- **⚠️ TexFilter vs PaiFilter 선택 기준 불명확**: 논문은 "이하 실험에서는 PaiFilter를 FilterNet으로 지칭한다"고 했지만, 실제 상황에서의 선택 가이드라인이 부족합니다.
- **검토자 해석**: Traffic 데이터셋에서의 열위는 FilterNet의 전체 스펙트럼 활용 전략이 주로 저주파로 구성된 데이터셋에서 오히려 노이즈를 학습할 수 있음을 시사합니다 (Figure 7(c) 참조).
- **검토자 해석**: PaiFilter와 MLP의 등가성 주장(Appendix B)은 직관적이지만, 완전한 수학적 등가 증명이 아닌 개념적 유사성 수준에 머뭅니다.

---

## 5. 통계적으로 취약한 부분 및 비교 불가능한 수치

| 유형 | 내용 | 위치 |
|------|------|------|
| ⚠️ **통계적 취약** | "95% 신뢰도의 통계적 유의성" 주장이지만 사용된 통계 검정 방법 미기재 | p.7, Section 5.2 |
| ⚠️ **통계적 취약** | FITS 결과는 "5회 실험 평균"으로 보고되었으나, 다른 모델들의 재현 횟수가 명시되지 않음 | p.6, C.2 및 Table 1 |
| ⚠️ **비교 불가** | 일부 baseline 결과는 원 논문 그대로 인용(FreTS, MICN, Autoformer 등), 동일 하이퍼파라미터 조건 보장 불가 | p.15, C.2 |
| ⚠️ **비교 불가** | FreTS 결과는 min-max 정규화 대신 instance normalization을 적용하여 수정 보고 — 원 논문과 공정한 비교가 아닐 수 있음 | p.15, C.2 |
| ⚠️ **비교 불가** | Table 1과 Table 4는 동일한 설정이지만 포함된 baseline이 다름 (직접 비교 불가) | Table 1(p.7), Table 4(p.18) |
| ⚠️ **통계적 취약** | 합성 데이터 실험(Figure 1)은 단일 데이터 포인트 기반으로 일반화 가능성 불명확 | p.2, Figure 1 |
| ⚠️ **통계적 취약** | Ablation Study(Figure 10)에서 오차 범위(error bar) 미표시 | p.17, Figure 10 |

---

## 6. 문서가 답하지 않는 질문

1. **TexFilter와 PaiFilter의 자동 선택 기준**: 언제 TexFilter를, 언제 PaiFilter를 선택해야 하는지에 대한 명확한 가이드라인이 없음
2. **복소수 필터의 초기화 방법**: PaiFilter의 무작위 초기화 분포(예: 가우시안, 균등 분포)가 명시되지 않음
3. **필터 해석 가능성(interpretability)**: 학습된 필터가 실제로 어떤 물리적 의미를 갖는지 분석 부재
4. **비정상 시계열에 대한 이론적 보장**: 인스턴스 정규화가 비정상성을 완전히 해결하는지에 대한 이론적 분석 미흡
5. **다른 도메인(텍스트, 이미지 등)으로의 확장 가능성**: 시계열 이외 도메인에서의 적용 가능성 미언급
6. **이상값(outlier)에 대한 견고성**: 극단적 이상값이 포함된 데이터에서의 성능 검증 미흡
7. **전이 학습(transfer learning) 가능성**: 사전 학습된 필터를 다른 도메인에 적용하는 방법 미탐색
8. **K (TexFilter의 학습 가능한 파라미터 수)의 최적값**: 하이퍼파라미터 K에 대한 감도 분석 없음
9. **실시간 스트리밍 데이터 적용 가능성**: 온라인 학습(online learning) 설정에서의 성능 미검증
10. **통계 검정 방법**: 95% 신뢰도 주장에 사용된 구체적 통계 검정 방법

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p.2): 합성 다중 주파수 신호에서의 성능 비교

**내용**: (a) 저-중-고 주파수 3성분 합성 신호 스펙트럼, (b) iTransformer 예측 결과(MSE=1.1e-01), (c) FilterNet 예측 결과(MSE=2.7e-05)

**해석**:
- FilterNet의 MSE가 iTransformer 대비 약 **4,000배** 낮음
- iTransformer는 고주파 성분을 포착하지 못해 예측이 실제 신호와 크게 불일치
- FilterNet은 저/중/고 주파수 성분 모두를 정확히 재현

> ⚠️ **검토자 주의**: 이는 합성(synthetic) 신호에서의 결과로, 실제 복잡한 데이터에서의 일반화를 보장하지 않습니다.

---

### Figure 2 (p.4): FilterNet 전체 아키텍처

**내용**: 인스턴스 정규화 → 주파수 필터 블록(PaiFilter 또는 TexFilter) → FFN → 역 인스턴스 정규화 구조

**해석**:
- 모듈형(modular) 설계로 PaiFilter와 TexFilter를 플러그인 방식으로 교체 가능
- 인스턴스 정규화가 별도 블록으로 분리되어 비정상성 처리와 필터링을 독립적으로 수행
- FFT/IFFT 래핑(wrapping)을 통해 시간-주파수 도메인 변환이 자동화됨

---

### Figure 5 (p.8): ETTh1 데이터셋 예측 시각화 비교

**내용**: (a) FilterNet, (b) iTransformer, (c) PatchTST의 예측값 vs 실제값 비교 (L=96, τ=96)

**해석**:
- FilterNet(a)의 예측선(파란색)이 실제값(주황색)과 가장 잘 일치
- iTransformer(b)와 PatchTST(c)는 변동 패턴을 따라가지 못하거나 진폭을 과대/과소 추정
- 특히 시계열의 하강 구간에서 FilterNet이 더 빠르게 반응함

> ⚠️ **검토자 주의**: 단일 케이스 시각화는 모델의 일반적 우수성을 증명하지 않으며, 저자들이 유리한 사례를 선택했을 가능성 배제 불가

---

### Figure 6 (p.9): 효율성 비교 (Exchange & Electricity)

**내용**: X축=학습 시간(s/epoch), Y축=MSE, 원의 크기=메모리 사용량

**해석**:
- FilterNet은 Exchange(8변수)에서 0.5GB, 1.5s/epoch로 DLinear(0.5GB, 0.9s)와 유사한 효율을 보이면서 더 낮은 MSE 달성
- Electricity(321변수)에서 FilterNet(0.6GB, 14.2s)은 PatchTST(1.3GB, 168.4s) 대비 메모리 2배 절약, 속도 약 12배 향상
- 효율성-정확도 트레이드오프에서 Pareto 최적에 가까운 위치

---

### Figure 7 (p.9): 학습된 필터의 주파수 응답 특성

**내용**: Weather, ETTh1, Traffic 데이터셋에서 학습된 필터의 주파수 스펙트럼 (x=주파수, y=진폭)

**해석**:
- **(a) Weather**: 중간 주파수 대역에도 상당한 진폭을 보임 → 날씨 데이터의 복잡한 주기성 포착
- **(b) ETTh1**: 저주파에서 고주파까지 고르게 활성화 → 전체 스펙트럼 활용 확인
- **(c) Traffic**: 주로 저주파 대역에 집중 → 교통 데이터의 저주파 특성 반영
- 이는 iTransformer가 Traffic에서 좋은 성능을 보이는 이유를 설명(저주파 위주 처리 모델이 저주파 데이터에 유리)

> 💡 **주파수 응답 특성(Frequency response characteristics)**: 필터가 각 주파수 성분을 얼마나 증폭(또는 감쇄)시키는지를 나타내는 특성. 높은 진폭 = 해당 주파수를 강하게 처리함을 의미합니다.

---

## 8. 결론: 시사점, 후속 연구 계획 및 추가 제언

### 8-1. 저자가 제시한 시사점과 후속 연구 계획

**[p.9, Section 6 Conclusion Remarks]**

**저자 시사점**:
1. 신호 처리 관점이 딥러닝 시계열 예측에 새로운 방향을 제시함
2. 주파수 필터를 직접 설계 원리로 활용하는 것이 효과적임
3. 단순한 구조(PaiFilter)도 복잡한 모델(iTransformer)과 경쟁하거나 능가 가능

**저자의 명시적 후속 연구 언급**:
> "We hope this work can facilitate more future research integrating signal processing techniques or filtering processes with deep learning on time series modeling and accurate forecasting." (p.9)

→ 매우 일반적 수준의 언급으로, **구체적인 후속 연구 계획은 명시되어 있지 않습니다.**

---

### 모델의 일반화 성능 향상 가능성

**현재 한계**:
- 8개 특정 벤치마크(주로 전력, 기상, 교통 도메인)에서만 검증됨
- 모든 데이터셋이 $L=96$ 또는 $L=336$의 고정된 룩백 윈도우 사용
- 채널 수가 7~862개 범위로 한정; 수천 개 변수의 초고차원 데이터 미검증

**일반화 향상 가능 방향** (검토자 제안):

1. **도메인 적응(Domain Adaptation)**: 사전 학습된 필터를 타 도메인에 이전하는 전이 학습 프레임워크 개발
2. **메타 학습(Meta-Learning)**: 새로운 시계열 데이터에 빠르게 적응하는 few-shot 필터 학습

$$\theta^* = \arg\min_\theta \mathbb{E}_{\mathcal{T} \sim p(\mathcal{T})} \left[ \mathcal{L}_{\mathcal{T}}(f_\theta) \right]$$

여기서 $\mathcal{T}$는 다양한 시계열 태스크 분포.

3. **적응형 대역폭**: 고정된 룩백 윈도우 대신 데이터 특성에 따라 동적으로 대역폭을 조정하는 메커니즘
4. **불규칙 시계열 처리**: 결측값이나 불균일 샘플링 시계열에 대한 확장

> 💡 **일반화 성능(Generalization performance)**: 모델이 학습 데이터가 아닌 새로운 데이터에서도 좋은 성능을 내는 능력. 과적합(overfitting)의 반대 개념입니다.

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 이하 비교는 제 학습 데이터(2024년 초까지) 기반입니다. FilterNet 논문(2024년 11월)은 NeurIPS 2024에 발표되어 후속 피인용 연구가 아직 제한적입니다.

#### 관련 최신 연구 비교표

| 모델 | 발표연도 | 핵심 방법 | FilterNet과의 관계 |
|------|----------|-----------|-------------------|
| **Informer** [14] | AAAI 2021 | ProbSparse Attention, $\mathcal{O}(L\log L)$ | FilterNet이 대부분 데이터셋에서 우위 (Table 4) |
| **Autoformer** [15] | NeurIPS 2021 | Auto-correlation + Series Decomposition | FilterNet이 일관적으로 우위 (Table 4) |
| **FEDformer** [25] | ICML 2022 | 주파수 도메인 Sparse Attention | FilterNet이 더 단순하면서 성능 동등 이상 |
| **FiLM** [31] | NeurIPS 2022 | Fourier + Legendre 필터 (저역 통과) | FilterNet은 전체 통과 필터로 확장, 성능 우위 |
| **DLinear** [12] | AAAI 2023 | 단층 선형 모델 + 시계열 분해 | FilterNet이 유사 효율성으로 높은 정확도 |
| **PatchTST** [16] | ICLR 2023 | Patch 기반 Transformer, 채널 독립성 | FilterNet이 소규모 데이터에서 경쟁력 |
| **FreTS** [13] | NeurIPS 2023 | 주파수 도메인 MLP | FilterNet과 유사 방향성, FilterNet이 전반적 우위 (Table 4) |
| **FITS** [32] | 2023 | 주파수 보간 (저역 통과, 10k 파라미터) | FilterNet의 all-pass 전략이 일관적으로 우위 |
| **iTransformer** [17] | ICLR 2024 | 역전(Inverted) Transformer, 변수 토큰화 | Traffic(고변수) 제외 대부분에서 FilterNet 우위 |
| **TimeMixer** | 2024 | 다중 스케일 MLP 믹서 | FilterNet과 직접 비교 없음 (⚠️ 불확실) |
| **TimesFM** (Google) | 2024 | 대형 기초 모델(Foundation Model) | FilterNet과 비교 불가 (설계 목적 상이) |

> 💡 **Foundation Model(기초 모델)**: 대규모 데이터로 사전 학습된 후 다양한 하위 태스크에 적용 가능한 범용 AI 모델. GPT가 텍스트 분야의 대표적 예입니다.

#### 연구 영향 및 향후 고려사항

**FilterNet이 향후 연구에 미치는 영향**:

1. **패러다임 전환 가능성**: Transformer 중심에서 신호처리 기반 접근으로의 방향 제시
2. **효율성 기준 상향**: $\mathcal{O}(\log L)$ 복잡도가 새로운 효율성 벤치마크가 될 수 있음
3. **주파수 도메인 설계 원리 확산**: 후속 연구에서 FFT 기반 레이어 설계 증가 예상

**향후 연구 시 고려할 점**:

| 고려사항 | 설명 |
|----------|------|
| **Foundation Model과의 통합** | FilterNet 필터 구조를 대형 시계열 기초 모델의 효율적 레이어로 활용 |
| **비정상 신호 이론적 분석** | 인스턴스 정규화만으로 비정상성이 해결되는지에 대한 이론적 보장 필요 |
| **다중 해상도(Multi-resolution) 확장** | 단일 대역폭 대신 웨이블릿(wavelet) 기반 다중 해상도 필터 통합 가능성 탐색 |
| **이상값 견고성** | 금융 시계열처럼 극단적 이상값이 자주 발생하는 도메인에서의 견고성 검증 필요 |
| **해석 가능성(Explainability)** | 학습된 필터의 주파수 응답을 통해 도메인 전문 지식과 연결하는 연구 필요 |
| **온라인 학습 적용** | 실시간 스트리밍 데이터에서 필터를 점진적으로 업데이트하는 메커니즘 개발 |
| **공정한 비교 프로토콜** | 다른 정규화 방법 사용 baseline과의 불공정 비교 문제 해결을 위한 표준화 필요 |

---

## 📚 참고자료 및 출처

본 분석에 사용된 주요 참고자료:

1. **원본 논문**: Yi, K., Fei, J., Zhang, Q., He, H., Hao, S., Lian, D., & Fan, W. (2024). "FilterNet: Harnessing Frequency Filters for Time Series Forecasting." *38th Conference on Neural Information Processing Systems (NeurIPS 2024)*. arXiv:2411.01623v2.

2. **참조된 논문들** (논문 내 인용 기준):
   - [12] Zeng et al., "Are Transformers Effective for Time Series Forecasting?" AAAI 2023.
   - [13] Yi et al., "Frequency-domain MLPs are More Effective Learners in Time Series Forecasting." NeurIPS 2023.
   - [14] Zhou et al., "Informer." AAAI 2021.
   - [15] Wu et al., "Autoformer." NeurIPS 2021.
   - [16] Nie et al., "PatchTST." ICLR 2023.
   - [17] Liu et al., "iTransformer." ICLR 2024.
   - [19] Kim et al., "Reversible Instance Normalization." ICLR 2021.
   - [25] Zhou et al., "FEDformer." ICML 2022.
   - [31] Zhou et al., "FiLM." NeurIPS 2022.
   - [32] Xu et al., "FITS." arXiv:2307.03756, 2023.
   - [35] Guibas et al., "Adaptive Fourier Neural Operators." arXiv:2111.13587, 2021.

3. **코드 저장소**: https://github.com/aikunyi/FilterNet

---

> ⚠️ **최종 정확도 고지**: 본 분석은 제공된 PDF 원문에 충실하게 작성되었습니다. Section 8-2의 최신 연구 비교(TimeMixer, TimesFM 등)는 제 학습 데이터 기반으로 일부 정보가 불완전하거나 업데이트가 필요할 수 있습니다. FilterNet 이후 발표된 논문들과의 비교는 실제 피인용 현황을 직접 확인하시기 바랍니다.
