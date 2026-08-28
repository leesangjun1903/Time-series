# Neural Fourier Modelling: A Highly Compact Approach to Time-Series Analysis

> **⚠️ 면책 고지**: 본 논문은 2024년 10월 arXiv 프리프린트(arXiv:2410.04703v1)로, **동료 심사(peer review)가 완료되지 않은 상태**입니다. 일부 주장은 추후 수정될 수 있습니다.

---

## 1. Executive Summary (10문장 이내)

Neural Fourier Modelling(NFM)은 시계열 데이터를 시간 도메인이 아닌 **푸리에(주파수) 도메인에서 직접** 모델링하는 새로운 접근법이다.  
핵심 아이디어는 푸리에 변환(FT)의 두 가지 특성, 즉 ① 유한 길이 시계열을 함수 공간에서 연속 시간 요소로 취급하는 능력과 ② 주파수 도메인 내 데이터 조작(리샘플링, 시간 범위 확장) 능력을 학습 메커니즘으로 재해석하는 것이다.  
NFM은 두 가지 핵심 모듈인 **LFT(Learnable Frequency Tokens)**와 **INFF(Implicit Neural Fourier Filter)**를 도입하여 주파수 보간(interpolation) 및 외삽(extrapolation)을 수행한다.  
예측(forecasting), 이상 탐지(anomaly detection), 분류(classification) 세 가지 태스크에서 SOTA 수준의 성능을 달성하였다.  
특히 모든 태스크에서 파라미터 수가 40K 미만(예측 27K, 이상 탐지 6.6K, 분류 37K)으로, 수백만 파라미터를 사용하는 기존 모델들과 경쟁하거나 이를 능가한다.  
NFM은 훈련 시와 다른 샘플링 속도로 테스트할 때도 강인한 성능을 보이는 **해상도 불변성(resolution-invariance)** 특성을 갖는다.  
이는 모델이 이산 신호를 함수 공간의 연속 시간 요소로 처리하기 때문에 가능하다.  
기존 FITS 모델의 일반화 버전으로, 다변량 시계열 처리, 가변 길이 입출력 적응, 확장성 등을 개선하였다.  
다만 불규칙 샘플링 시계열 처리에는 아직 한계가 있으며, FFT의 등간격 샘플링 요구 조건이 현재의 제약 사항이다.

---

### 1-1. 연구의 목적과 필요성

| 구분 | 내용 |
|------|------|
| **핵심 문제** | 기존 신경망 기반 시계열 분석은 주로 **시간 도메인**에서 이루어지며, 주파수 표현은 보조적 특성으로만 활용됨 |
| **기술적 공백** | 주파수 보간·외삽을 **직접적인 핵심 학습 메커니즘**으로 활용한 연구 부재 |
| **실용적 필요성** | 엣지 컴퓨팅 등 저자원 환경에서 수억 개 파라미터를 가진 모델은 배포 불가능 |
| **일반화 문제** | 테스트 시 훈련과 다른 샘플링 속도를 만날 경우 기존 모델들의 성능이 급격히 저하됨 |
| **연구 동기** | 푸리에 도메인에서 직접 학습 시 함수-함수 매핑(function-to-function mapping)이 가능하여 **해상도 불변성**이라는 귀납적 편향(inductive bias)을 자연스럽게 획득 가능 |

> 📝 **귀납적 편향(Inductive Bias)**: 모델이 학습 데이터 외의 상황에도 잘 동작하도록 유도하는 사전 가정이나 구조적 제약. 예를 들어 CNN의 지역성(locality) 가정이 대표적 귀납적 편향이다.

---

## 2. 핵심 주장과 근거 표

| 핵심 주장 | 근거 | 위치 |
|-----------|------|------|
| 푸리에 도메인에서 직접 시계열 모델링 가능 | DFT의 수학적 성질: 이산 신호 → 연속 시간 함수 공간 표현 | Section 3, Eq.(2)(3) |
| 주파수 조작을 학습 메커니즘으로 재해석 | 제로 패딩/절단 → 주파수 보간, 제로 인터리빙 → 주파수 외삽 | Figure 1, Section 3.1 |
| LFT가 효과적 스펙트럴 사전(prior) 학습 | Ablation: INFF-only(79.4%) → INFF+LFT(90.9%), 14.5%↑ | Figure 4, Table 11 |
| INFF가 우수한 글로벌 합성곱 연산자 | 4가지 SOTA NFF 대비 정확도 및 해상도 강인성 우월 | Figure 4, Appendix B |
| 극도의 경량성 달성 | 예측 27K, 이상 탐지 6.6K, 분류 37K 파라미터로 SOTA | Table 1, 2, 3 |
| 해상도 불변성 | 미학습 SR=1/2에서 분류 정확도 0.7%↓ (CKConv 7.95%↓, NRDE 8.31%↓) | Table 3, Table 4 |
| FITS의 일반화 버전 | 다변량 처리, 가변 길이, 확장성 모두 개선 | Appendix A, Figure 7 |

---

## 2-1. 상세 설명

### 🔴 해결하고자 하는 문제

1. **시간 도메인 의존성**: 기존 모델들은 시간 도메인에서만 작동하거나 주파수를 보조 특성으로만 사용
2. **대규모 파라미터 요구**: 실용적 배포에 부적합한 수백만~수십억 파라미터
3. **샘플링 속도 변화 취약성**: 훈련 시 보지 못한 샘플링 속도에서 성능 급락
4. **가변 길이 처리 불가**: 기존 모델들의 예측 헤드(prediction head)가 입력 길이에 종속

---

### 🟢 제안하는 방법 및 수식

#### (1) 기본 표기 및 보간/외삽 인수

$$\frac{L}{N} = \frac{T_y f_y}{T_x f_x} = m_\tau m_f \tag{1}$$

- $L$: 출력 시계열 길이
- $N$: 입력 시계열 길이  
- $T_y, T_x$: 출력/입력 시간 범위(timespan)
- $f_y, f_x$: 출력/입력 샘플링 속도
- $m_\tau := T_y/T_x$: **보간 인수(interpolation factor)** — 시간 범위 확장 비율 (예측 시 $m_\tau > 1$)
- $m_f := f_y/f_x$: **외삽 인수(extrapolation factor)** — 샘플링 속도 변화 비율 (이상 탐지 시 $m_f > 1$)

> 📝 **주파수 보간(Frequency Interpolation)**: 주파수 스펙트럼을 확장하여 더 긴 시간 범위의 신호를 생성. 시간 도메인에서는 업샘플링에 해당.
> 
> 📝 **주파수 외삽(Frequency Extrapolation)**: 기존 주파수 성분 밖으로 스펙트럼을 확장. 시간 도메인에서는 더 높은 해상도 신호 생성에 해당.

---

#### (2) 이산 푸리에 변환 (DFT / IDFT)

$$X[k] = \mathcal{F}(x) := \sum_{n=0}^{N-1} x[n] e^{-j2\pi kn/N} \tag{2}$$

$$x[n] = \mathcal{F}^{-1}(X) := \frac{1}{N} \sum_{k=0}^{N-1} X[k] e^{j2\pi kn/N} \tag{3}$$

- $x[n]$: 시간 도메인 이산 신호 ($n$은 시간 인덱스)
- $X[k]$: 주파수 도메인 복소수 스펙트럼 ($k$는 주파수 인덱스)
- $e^{-j2\pi kn/N}$: 복소 지수 함수 (오일러 공식 기반)
- $j$: 허수 단위 ($j = \sqrt{-1}$)

> 📝 **DFT(이산 푸리에 변환)**: 유한 길이 이산 신호를 복소수 형태의 주파수 성분으로 분해하는 수학적 도구. FFT는 DFT를 $\mathcal{O}(N \log N)$ 복잡도로 효율적으로 계산하는 알고리즘.

---

#### (3) LFT (Learnable Frequency Tokens)

$$V[k \in I_{K_L}] = \mathcal{F}(\text{InstanceNorm}(\phi(\tau_n)))$$

$$Z_0[k] = (\bar{Z}_0[k] + V[k])$$

$$z_0[n] = \mathcal{F}^{-1}(Z_0[k]) \tag{5}$$

- $V[k] \in \mathbb{C}^d$: 학습 가능한 주파수 토큰 (복소수 벡터)
- $\phi: \mathbb{R} \to \mathbb{R}^d$: **INR(암묵적 신경 표현, Implicit Neural Representation)** — 시간 위치를 특징 벡터로 매핑하는 MLP
- $\tau_n = \{n/f_y \mid n \in I_L\}$: 샘플링된 시간 위치
- $\bar{Z}_0 \in \mathbb{C}^{K_L \times d}$: 제로 초기화된 확장 스펙트럼 표현
- $Z_0[k]$: LFT가 추가된 스펙트럼 임베딩
- $\text{InstanceNorm}$: 인스턴스 정규화 — DC 성분(직류 성분) 편향 제거
- $K_L := \lfloor L/2 \rfloor + 1$: 켤레 대칭성 활용 후의 주파수 성분 수

> 📝 **INR(Implicit Neural Representation, 암묵적 신경 표현)**: 데이터를 격자 구조 대신 연속 함수로 표현하는 방법. NeRF(Neural Radiance Fields)가 대표적 사례. 좌표(시간, 공간)를 입력받아 해당 위치의 특징값을 출력한다.
>
> 📝 **DC 성분(DC Component)**: 주파수 0에 해당하는 성분으로 신호의 평균값을 나타냄. DC 성분이 지배적이면 고주파 학습이 방해된다.

---

#### (4) INFF (Implicit Neural Fourier Filter)

$$\hat{z} = \mathcal{F}^{-1}(\mathcal{R}(z_0) \odot \mathcal{F}(z)) \tag{8}$$

$$\mathcal{R}(z_0) := \mathcal{W}(\mathcal{F}(\text{InstanceNorm}(\phi(\tau_n) + z_0))) \tag{9}$$

- $\hat{z}$: INFF를 통해 변조된 임베딩 토큰
- $\mathcal{R}[k] \in \mathbb{C}^d$: 주파수 필터 계수 (푸리에 도메인에서 정의)
- $\odot$: **아다마르 곱(Hadamard product)** — 원소별 곱셈
- $\mathcal{W}: \mathbb{C}^d \to \mathbb{C}^d$: 복소수 값 MLP
- $z_0$: 초기 스펙트럼 임베딩 (INFF 계산 조건화에 사용)
- $\phi(\tau_n) + z_0$: INR 출력과 현재 임베딩의 결합 → 인스턴스 적응성 부여

> 📝 **합성곱 정리(Convolution Theorem)**: 시간 도메인에서의 합성곱 연산이 주파수 도메인에서의 원소별 곱셈과 동치임을 말하는 정리. 이를 통해 글로벌 합성곱을 $\mathcal{O}(N \log N)$ 복잡도로 효율적 계산 가능.

---

#### (5) 손실 함수

**예측 손실:**

$$\mathcal{L}_{Forecasting} = \underbrace{\lambda \frac{1}{L} \sum_{n=0}^{L-1} \|\hat{y}[n] - y[n]\|_2}_{\mathcal{L}_{TD}} + \underbrace{(1-\lambda) \frac{1}{K_L} \sum_{k=0}^{K_L-1} \left((\hat{Y}_{Real}[k] - Y_{Real}[k])^2 + (\hat{Y}_{Imag}[k] - Y_{Imag}[k])^2\right)^{1/2}}_{\mathcal{L}_{FD}} \tag{13}$$

**이상 탐지 손실:**

$$\mathcal{L}_{AD} = \underbrace{\lambda \frac{1}{N} \sum_{n=0}^{N-1} \|\hat{x}[n] - x[n]\|_2}_{\mathcal{L}_{TD}} + \underbrace{(1-\lambda) \frac{1}{K_N} \sum_{k=0}^{K_N-1} \left((\hat{X}_{Real}[k] - X_{Real}[k])^2 + (\hat{X}_{Imag}[k] - X_{Imag}[k])^2\right)^{1/2}}_{\mathcal{L}_{FD}} \tag{14}$$

- $\lambda = 0.5$: 시간/주파수 도메인 손실 균형 계수
- $\mathcal{L}_{TD}$: 시간 도메인 MSE 손실 (지역적 정밀도)
- $\mathcal{L}_{FD}$: 주파수 도메인 손실 (전역적 구조, Focal Frequency Loss 변형)
- $Y_{Real}[k], Y_{Imag}[k]$: 시퀀스 $y$의 주파수 표현의 실수부/허수부
- $\hat{Y}\_{Real}[k], \hat{Y}_{Imag}[k]$: 예측 시퀀스의 주파수 표현의 실수부/허수부

> 📝 **MSE(Mean Squared Error)**: 예측값과 실제값의 차이를 제곱하여 평균한 손실 함수. 시간 도메인에서 점별(point-wise) 오차를 측정한다.
>
> 📝 **Focal Frequency Loss**: 이미지 복원 연구에서 유래한 손실 함수로, 주파수 도메인에서 재구성이 어려운 성분(고주파)에 더 큰 가중치를 부여한다.

---

### 🟡 모델 구조

```
입력 x [N×c]
    ↓ (Projection: ℝ^c → ℝ^d, SIREN 기반 비선형 투영)
x̄ [N×d]
    ↓ LFT Block (주파수 확장 + 스펙트럴 사전 학습)
    ├── DFT → 확장(M) → 역DFT
    └── INR(ϕ) → InstanceNorm → DFT → V[k] 추가
z₀ [L×d]
    ↓ × l번 반복: Mixer Block
    ├── LayerNorm + MLP Channel Mixer
    └── INFF Global Convolution
        ├── DFT → R(z₀) 계산 (INR + ComplexMLP)
        └── 아다마르 곱 → 역DFT
z_l [L×d]
    ↓ Predictor P (task별 선형 레이어)
y (최종 출력)
```

**태스크별 설정:**
- 예측: $m_f=1, m_\tau>1$ → 주파수 보간
- 이상 탐지: $m_f>1, m_\tau=1$ → 주파수 외삽 (다운샘플→원본 복원)
- 분류: $m_f=1, m_\tau=1$ → 재구성 후 전역 평균 풀링+FC

---

### 🔵 성능 향상 및 한계

**성능 향상:**

| 태스크 | NFM 파라미터 | 주요 성능 | 비교 대상 최대 파라미터 |
|--------|------------|----------|----------------------|
| 예측 | 27K | ETTm1 MSE: 0.345 (1위/공동) | TimesNet ~0.3B |
| 이상 탐지 | 6.6K | SMD F1: 84.32 (1위) | TimesNet ~28M |
| 분류 | 37K | SC-MFCC ACC: 94.23% | Transformer 800K |

**한계:**
- ❌ **불규칙 시계열 처리 불가**: FFT는 등간격 샘플링 필수
- ❌ 분류에서 CKConv(95.27%)에 약간 뒤짐
- ❌ 원시 파형 분류에서 S4(96.17%)에 열등
- ❌ ETTh2 예측에서 FITS, PatchTST에 일부 열등 (Table 7)

---

## 3. 각 주장에 페이지 및 Figure/Table 번호 표시

| 주장 | 근거 위치 |
|------|-----------|
| NFM 전체 아키텍처 | Figure 2, 3 (p.3, 5) |
| 푸리에 도메인 조작의 시각적 설명 | Figure 1 (p.1) |
| 예측 성능 비교 (평균) | Table 1 (p.7) |
| 예측 성능 비교 (전체) | Table 7 (p.24) |
| 이상 탐지 성능 | Table 2, Table 8 (p.8, 24) |
| 분류 성능 및 미학습 SR 비교 | Table 3 (p.8) |
| 미학습 샘플링 속도 예측 성능 | Table 4, Table 9 (p.8, 25) |
| LFT/INFF Ablation Study | Figure 4, Table 10, 11 (p.9, 26) |
| INFF 시각화 | Figure 5 (p.9) |
| 확장성(Scaling) 분석 | Figure 6 (p.10) |
| FITS 대비 파라미터 비교 | Figure 7, Appendix A (p.15) |
| 다양한 NFF 설계 비교 | Figure 8, Appendix B (p.16) |
| 통계적 유의성 (랜덤 시드) | Figure 10 (p.27-28), Appendix E.5 |
| 하이퍼파라미터 설정 | Table 5 (p.19) |
| 데이터셋 요약 | Table 6 (p.23) |

---

## 4. 저자 보고 결과 vs. 해석 분리

### 저자가 직접 보고한 결과

**연구 주제:**
- 푸리에 도메인에서 직접 시계열 분석을 수행하는 경량 모델 NFM 제안

**방법 (수식):** (위의 Eq.1-14 참조)

**결과 (저자 직접 보고):**
- 예측: 27K 파라미터로 7개 벤치마크에서 SOTA 수준 달성 (Table 1, p.7)
- 이상 탐지: 6.6K 파라미터로 SMD, MSL, PSM에서 1위 (Table 2, p.8)
- 분류 (MFCC): 37K로 94.23%, CKConv(95.27%), S4(93.96%) 사이 (Table 3, p.8)
- 미학습 SR=1/2에서 분류 정확도 0.7%↓ (NRDE 8.31%↓ 대비) (Table 3, p.8)
- LFT 추가 시 INFF-only 대비 14.5%↑ 성능 향상 (p.9)
- 통계적 유의성: 예측 MSE 표준편차 $10^{-4} \sim 10^{-3}$ 수준 (Figure 10, p.27-28)

---

### 검토자 해석

> ⚠️ 이하는 검토자의 해석으로 저자의 직접 주장과 구분됩니다.

1. **파라미터 효율성의 구조적 원인**: NFM의 파라미터가 입출력 길이와 독립적인 이유는 예측 헤드가 특징-대-특징(feature-to-feature) 투영을 사용하기 때문. 반면 PatchTST의 경우 $L=720$에서 전체 파라미터의 95% 이상이 예측 헤드에 집중된다는 점이 흥미롭다.

2. **FITS와의 관계**: NFM을 "FITS의 완전한 재설계이자 일반화"라고 저자들이 주장하지만, 이는 과장일 수 있다. FITS가 여전히 매우 짧은 시계열 + 낮은 컷오프 주파수 환경에서는 NFM보다 낮은 파라미터 수로 유사한 성능을 낼 수 있음.

3. **해상도 불변성의 한계**: 저자들은 NFM이 함수-함수 매핑을 학습하기 때문에 해상도 불변성을 가진다고 주장하지만, 단순 주파수 도메인 처리만으로 완전한 연속 시간 모델링이 달성된다고 보기는 어렵다. S4 모델도 SR=1/2에서 2.14%↓로 NFM(0.7%↓)보다 크지만, 이는 S4가 상태 공간 모델로서 다른 메커니즘을 사용하기 때문.

4. **이상 탐지 프레임**: 다운샘플링-복원으로 이상 탐지를 수행하는 것은 독창적이나, 해당 설계 선택(다운샘플링 인수 $d_r$ 등)이 데이터셋 특성에 민감할 수 있음.

---

## 5. 통계적으로 취약한 부분 및 비교 불가능한 수치

### ⚠️ 통계적 취약점

| 항목 | 문제점 |
|------|--------|
| **기존 베이스라인 통계 미제시** | NFM만 랜덤 시드 반복 실험(Figure 10) 제공, 베이스라인은 단일 실행 결과만 보고 |
| **TimesNet 비교** | NFM은 lookback=720, TimesNet은 lookback=96 사용 → **동일 조건 비교 불가** (p.19 명시) |
| **iTransformer 재실험** | 원래 lookback=96 결과를 저자들이 lookback=720으로 재실험하여 보고 → 베이스라인 최적 조건 상이 |
| **FITS 파라미터 설정** | FITS의 "10K 설정"이 아닌 더 많은 파라미터 설정(~0.2M)으로 비교 (p.19-20) |
| **분류 SR=1/2 비교** | 연속 시간 모델(ODE-RNN 등)과의 비교에서 해당 모델들이 raw waveform 처리 불가(~) 표시로 비교 제외 |
| **이상 탐지 코드 수정** | 저자들이 베이스라인 공식 코드의 "버그"를 수정하여 재실험 → 원 논문과 수치 상이, 독립 검증 필요 |
| **통계 검정 미실시** | 랜덤 시드 실험의 평균±표준편차만 제공, t-검정 등 공식 통계 검정 없음 |

### ⚠️ 비교 불가능한 수치

- **SC-raw(N=16K) 분류**: Transformer, ODE 계열 모델들이 GPU 메모리 부족으로 실험 불가(~로 표시) → 실질적 비교 범위 제한
- **ADformer 비교**: 저자들이 ADformer의 이상 점수 기준을 변경(joint criterion → 단순 재구성 오차)하여 수치 변경 → 원 ADformer 논문과 직접 수치 비교 불가 (Table 2 각주)
- **채널 독립 vs. 채널 혼합**: NFM과 일부 베이스라인이 채널 독립 전략 사용, 다른 모델들은 채널 간 상관 학습 → 공정성 논란 여지

---

## 6. 논문이 답하지 않는 질문

| # | 미해결 질문 |
|---|------------|
| 1 | **불규칙 시계열** 처리를 위한 구체적 해결 방안은? (한계로만 언급, 방향 미제시) |
| 2 | 최적 다운샘플링 인수 $d_r$ 선택 기준은? 데이터별 민감도는? |
| 3 | $\lambda=0.5$ (시간/주파수 손실 균형)의 최적화 여부 및 민감도 분석 |
| 4 | 멀티변량 시계열에서 채널 간 상관관계를 활용하는 방법 (현재 채널 독립 전략 사용) |
| 5 | 매우 긴 시계열(>16K) 또는 매우 짧은 시계열(<100)에서의 성능 |
| 6 | 비정상(non-stationary) 시계열에서의 주파수 표현 안정성 |
| 7 | INFF의 INR 주파수 스케일 $w_0$ 선택이 성능에 미치는 영향의 체계적 분석 |
| 8 | 실제 온디바이스(edge) 환경 배포 시 지연 시간(latency) 벤치마크 |
| 9 | 도메인 이동(domain shift) 시나리오에서의 일반화 성능 |
| 10 | 시계열 길이가 변하는 온라인 학습(online learning) 시나리오 적용 가능성 |

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p.1) — 푸리에 도메인 조작

**내용**: 상단은 제로 패딩/절단(리샘플링), 하단은 제로 인터리빙(시간 범위 확장)

**해석**: 이 그림은 NFM의 핵심 아이디어를 직관적으로 보여준다. 푸리에 도메인에서의 단순한 조작(0 삽입)이 시간 도메인에서 업샘플링 또는 시간 범위 확장과 동치임을 보여, "제로를 학습 가능한 계수로 대체"한다는 NFM의 철학적 기반을 제공한다. 즉, 시계열 예측을 "미래 주파수 성분 보간"으로 재정의하는 패러다임 전환을 이 한 그림이 담고 있다.

---

### Figure 2 (p.3) — NFM 전체 워크플로우

**내용**: 입력 $x[n]$에서 출력 $y[n]$, $z[n]$으로의 전체 처리 흐름 (3D 시각화)

**해석**: $m_\tau > 1$ (보간, 예측)과 $m_f > 1$ (외삽, 해상도 변경)이 동시에 또는 선택적으로 작동하는 방식을 보여준다. LFT가 스펙트럴 확장을 담당하고 INFF가 이를 정제한다는 계층적 처리가 명확히 표현되어 있다. NFM이 이산 신호를 연속 시간 함수 공간에서 처리한다는 추상적 개념을 가장 잘 시각화한 그림이다.

---

### Figure 3 (p.5) — NFM 아키텍처 상세

**내용**: LFT 블록, MLP 채널 믹서, INFF 글로벌 합성곱의 데이터 플로우

**해석**: 각 모듈의 입출력 차원($\mathbb{R}^{N \times d}$, $\mathbb{C}^{K_L \times d}$ 등)과 연산 순서가 명확히 표시되어 있다. 특히 LFT에서의 확장 연산 $M$과 INFF에서의 아다마르 곱이 구분되어 있어, 왜 NFM의 파라미터 수가 입출력 길이와 독립적인지 이해할 수 있다. 복소수 도메인과 실수 도메인 간 전환(DFT/IDFT)이 어디서 발생하는지도 확인 가능하다.

---

### Figure 4 (p.9) — Ablation Study 및 NFF 비교

**내용**: SpeechCommand 데이터셋에서 다양한 구성의 파라미터 수, FLOP, 메모리, ACC 비교

**해석**: LFT의 중요성이 극명하게 드러난다. INFF만 사용 시 79.4%이던 정확도가 LFT 추가 시 90.9%로 급등(14.5%↑)하며, 심지어 FNO+LFT(84.8%), AFNO+LFT(84.7%), AFF+LFT(88.4%)보다도 INFF+LFT가 우월하다. 또한 SR=1/2(미학습 해상도)에서는 모드 적응형(mode-adaptive) NFF(FNO 11.6%↓, GFN 9.5%↓)의 성능 저하가 인스턴스 적응형(AFNO 2.7%↓, AFF 2.9%↓)보다 훨씬 크며, INFF+LFT는 0.7%↓로 가장 강인하다. 이는 INFF의 설계 철학(인스턴스 + 모드 적응성 동시 달성)의 우수성을 입증한다.

---

### Figure 6 (p.10) — 확장성(Scaling) 분석

**내용**: 히든 차원과 깊이 변화에 따른 NFM 성능 및 파라미터 수 변화

**해석**: (a) 예측 태스크에서 NFM은 단 27K 파라미터(히든=36, 깊이=1)로 FITS(16K), N-Linear(69K), iTransformer(304K), PatchTST(548K)보다 작거나 유사한 크기로 경쟁력 있는 MSE를 달성한다. (b) 분류에서도 37K로 S4(400K), CKConv(100K)과 경쟁한다. 중요한 점은 파라미터를 늘릴수록 성능이 단조롭게 향상되어 확장성이 있음을 보여주며, 이는 NFM이 단순 경량화 트릭이 아닌 구조적 효율성을 가짐을 시사한다.

---

## 8. 결론: 시사점, 후속 연구, 추가 방향

### 8-1. 저자 제시 시사점 및 후속 연구 계획

**저자 제시 시사점 (Section 5, p.10):**
- 푸리에 도메인에서의 직접 모델링이 연속 시간 특성과 경량성을 동시에 달성 가능
- 40K 이하 파라미터로 SOTA 달성 → 저자원 온디바이스 학습 가능성 제시
- 주파수 보간/외삽이 다양한 시계열 태스크에 통합 프레임워크 제공

**저자 제시 후속 연구 (Limitations, p.10):**
- **불규칙 시계열 처리**: FFT의 등간격 요구 조건 극복 방안 연구 중
  - 현재 한계: DFT 행렬 직접 계산은 속도·메모리 모두 비효율적
  - 향후 방향: NFFT(Non-uniform FFT) 또는 Gappy FFT 등 적용 탐색

---

### 모델의 일반화 성능 향상 가능성 (심층 분석)

#### 현재 일반화 강점

NFM이 일반화 성능에서 강점을 보이는 이유는 구조적으로 세 가지이다:

1. **함수-함수 매핑 학습**: DFT가 이산 신호를 함수 공간의 연속 요소로 변환하므로, 다른 샘플링 속도의 신호도 동일한 함수 표현 공간에서 처리 가능

2. **INR 기반 LFT**: SIREN 활성 함수( $\alpha = \sin(\cdot)$ )와 푸리에 특성(Fourier features)의 결합이 스펙트럴 편향(spectral bias) 문제를 완화하여 고주파 성분도 학습 가능

3. **인스턴스 + 모드 적응성**: INFF가 입력 인스턴스( $z_0$ )와 주파수 모드( $\phi(\tau_n)$ ) 모두에 조건화되어 분포 이동(distribution shift)에 강인

#### 일반화 향상을 위한 추가 개선 가능성

| 방향 | 현재 한계 | 개선 방안 |
|------|-----------|-----------|
| 불규칙 시계열 | FFT는 등간격 필수 | NFFT, Sparse CTFT, 신경 적분(neural quadrature) 활용 |
| 도메인 이동 | RevIN이 일부 데이터에서 불안정 | 적응적 정규화(adaptive normalization) 연구 |
| 채널 간 의존성 | 채널 독립 전략 사용 | Cross-channel 주파수 상호작용 모듈 개발 |
| 분포 밖(OOD) 일반화 | 미학습 주파수 범위 취약 | 주파수 증강(frequency augmentation) 결합 |
| 다중 스케일 | 단일 주파수 해상도 | 멀티 스케일 주파수 분해(다중 해상도 분석) |

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 방법 | NFM과의 관계 |
|------|------|-----------|--------------|
| **FEDformer** (Zhou et al.) | 2022 | 주파수 향상 Transformer, 랜덤 모드 선택 | NFM이 더 가벼운 대안으로 유사 성능 |
| **PatchTST** (Nie et al.) | 2022 | 시계열 패치(patch) 기반 Transformer | NFM의 경쟁 모델; 파라미터 효율성에서 NFM 우위 |
| **TimesNet** (Wu et al.) | 2022 | 2D 시간적 변화 모델링 | 수억 파라미터 vs. NFM의 27K |
| **iTransformer** (Liu et al.) | 2023 | 역전 Transformer (역전된 어텐션) | NFM이 일부 데이터셋에서 경쟁 |
| **FITS** (Xu et al.) | 2023 | 10K 파라미터 주파수 선형 모델 | NFM의 직접 전구체; NFM이 일반화 버전 |
| **FreTS** (Yi et al.) | 2024 | 주파수 도메인 MLP 예측 | NFM과 가장 유사한 철학; 핵심 메커니즘 상이 |
| **FourierGNN** (Yi et al.) | 2024 | 그래프 관점 주파수 믹싱 | 다변량 특화; NFM은 더 일반적 |
| **S4** (Gu et al.) | 2021 | 구조적 상태 공간 모델 | 긴 시퀀스 처리에서 강점; NFM이 분류에서 경쟁 |
| **CKConv** (Romero et al.) | 2021 | 연속 커널 합성곱 | 분류에서 NFM보다 약간 우위(95.27% vs 94.23%) |

> 📝 **상태 공간 모델(State Space Model, SSM)**: 시스템의 내부 상태를 변수로 표현하여 시계열을 모델링하는 방법. S4, Mamba 등이 대표적이며, 특히 긴 시퀀스에서 효율적이다.

#### NFM이 앞으로의 연구에 미치는 영향

1. **극단적 경량화 패러다임의 검증**: 40K 미만 파라미터로 SOTA 가능성 입증 → 엣지 AI 연구 가속화
2. **주파수 도메인 완전 모델링의 타당성 확립**: 보조 특성이 아닌 주 학습 공간으로서 주파수 도메인 재조명
3. **해상도 불변 학습 방법론**: 테스트 타임 도메인 일반화(test-time domain generalization) 연구에 영향
4. **FITS 계열 연구 확장**: 선형 → 비선형, 단변량 → 다변량으로의 체계적 발전 경로 제시

#### 앞으로 연구 시 고려할 점

1. **불규칙 시계열**: NFFT나 신경 미분방정식(Neural ODE)과의 결합으로 FFT 제약 극복 필요
2. **더 강한 베이스라인**: Mamba(2024), TimesFM(Google, 2024) 등 최신 모델과 비교 필요
3. **사전 훈련(Pre-training)**: NFM의 함수 공간 표현이 대규모 시계열 기반 모델(foundation model) 구축에 활용 가능한지 탐색
4. **인과성(Causality)**: 주파수 도메인 처리의 비인과적 특성(미래 정보 누출 가능성) 주의
5. **복소수 학습**: INFF의 복소수 MLP 최적화가 실수 MLP와 다를 수 있으므로 복소수 역전파 안정성 연구 필요
6. **해석 가능성**: 학습된 INFF 계수가 실제로 의미 있는 주파수 특성을 인코딩하는지 체계적 분석 필요

---

## 참고 자료

**논문 원문:**
- Kim, M., Hioka, Y., & Witbrock, M. (2024). *Neural Fourier Modelling: A Highly Compact Approach to Time-Series Analysis*. arXiv:2410.04703v1.

**논문 내 인용 주요 참고문헌:**
- Xu, Z., Zeng, A., & Xu, Q. (2023). FITS: Modeling time series with 10k parameters. arXiv:2307.03756.
- Nie, Y., et al. (2022). A time series is worth 64 words. ICLR 2023.
- Wu, H., et al. (2022). TimesNet. ICLR 2023.
- Liu, Y., et al. (2023). iTransformer. ICLR 2024.
- Yi, K., et al. (2024b). FreTS: Frequency-domain MLPs. NeurIPS 2024.
- Sitzmann, V., et al. (2020). Implicit neural representations with periodic activation functions. NeurIPS 2020.
- Li, Z., et al. (2020). Fourier Neural Operator. ICLR 2021.
- Guibas, J., et al. (2021). AFNO. arXiv:2111.13587.
- Romero, D. W., et al. (2021). CKConv. ICLR 2022.
- Gu, A., et al. (2021). S4. ICLR 2022.
- Jiang, L., et al. (2021). Focal frequency loss. ICCV 2021.
- Kim, T., et al. (2021). RevIN. ICLR 2022.
- Zeng, A., et al. (2023). N-Linear. AAAI 2023.
- Rahaman, N., et al. (2019). On the spectral bias of neural networks. ICML 2019.
- Tancik, M., et al. (2020). Fourier features. NeurIPS 2020.

**코드 저장소:**
- https://github.com/minkiml/NFM
