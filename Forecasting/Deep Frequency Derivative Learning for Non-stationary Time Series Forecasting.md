# Deep Frequency Derivative Learning for Non-stationary Time Series Forecasting
## 논문 분석 보고서

> **⚠️ 정확도 고지**: 본 분석은 제공된 PDF 원문에 기반하며, 원문에 명시되지 않은 내용은 추정임을 명시합니다. 2020년 이후 외부 논문 비교 분석 부분은 제 학습 데이터 기준(~2024년 초)의 일반적 지식을 활용하며, 일부 최신 수치는 부정확할 수 있습니다.

---

## 1. Executive Summary (10문장 이내)

1. 대부분의 실세계 시계열 데이터는 **비정상성(non-stationarity)**을 가지며, 이는 시간에 따라 분포가 변하는 **분포 이동(distribution shift)** 문제를 야기한다.
2. 기존 정규화 방법들(RevIN, Dish-TS 등)은 평균·표준편차 등 통계량만을 이용해 시계열을 변환하며, 이는 이론적으로 주파수 스펙트럼의 **영주파수 성분(zero frequency component)**만을 활용하는 것과 동치임을 저자들이 증명한다.
3. 영주파수만 활용하면 데이터의 분포 정보를 온전히 활용하지 못하는 **정보 활용 병목(information utilization bottleneck)**이 발생한다.
4. 저자들은 **전체 주파수 스펙트럼**을 활용한 새로운 가역 변환인 **주파수 도함수 변환(FDT, Frequency Derivative Transformation)**을 제안한다.
5. FDT는 시계열을 주파수 도메인에서 미분하여 더 정상적(stationary)인 표현으로 변환한다.
6. 이를 기반으로 한 **DERITS** 프레임워크는 **차수 적응형 푸리에 합성곱 네트워크(OFCN)**와 **병렬 적층 아키텍처**로 구성된다.
7. OFCN은 차수별로 적응적 주파수 필터링을 수행하고, 푸리에 합성곱으로 주파수 도메인 의존성을 학습한다.
8. 실험 결과 DERITS는 7개 실세계 데이터셋에서 기존 Transformer 기반 모델 대비 MAE/RMSE 평균 **20% 이상** 감소를 달성한다.
9. DERITS는 NSTransformer 등 기존 방법 대비 훈련 속도가 **수 배 이상** 빠른 효율성도 보인다.
10. 저자들은 주파수 도함수 관점이 비정상 시계열 예측의 새로운 연구 방향을 제시할 수 있다고 주장한다.

---

### 1-1. 연구의 목적과 필요성

**목적**: 비정상 시계열에서 발생하는 분포 이동 문제를 주파수 전체 스펙트럼을 활용한 미분 기반 변환으로 해결하고, 정확한 예측 성능을 달성하는 것.

**필요성**:
- 교통, 날씨, 금융 등 실세계 시계열은 대부분 비정상적이며, 수백만 타임스텝에 걸쳐 분포가 이동함 (p.1, Section 1).
- 기존 정규화 방법들은 평균/표준편차만 조정하는데, 이는 수식적으로 **영주파수 성분 정규화**에 불과함 → 정보 손실 발생 (p.1, Appendix B.1).
- 분포 이동에는 **공변량 이동(covariate shift)**과 **조건부 이동(conditional shift)** 두 가지가 존재하며, 기존 방법들은 이를 완전히 해결하지 못함 (p.3, Section 3).

> 📌 **비정상성(Non-stationarity)**: 시계열의 평균, 분산, 자기상관 등 통계적 성질이 시간에 따라 변하는 특성. 예: 주식 가격, 기상 데이터.
>
> 📌 **분포 이동(Distribution Shift)**: 모델 훈련 시와 테스트 시의 데이터 분포가 달라지는 현상으로, 모델 성능 저하의 주요 원인.
>
> 📌 **영주파수 성분(Zero Frequency Component)**: 푸리에 변환에서 $f=0$일 때의 성분으로, 시계열 신호의 평균값과 수학적으로 동치임. (Appendix B.1 증명)

---

## 2. 핵심 주장과 근거 표

| # | 핵심 주장 | 근거 | 위치 |
|---|-----------|------|-------|
| 1 | 기존 정규화(평균/표준편차 조정)는 영주파수 성분만 활용 | DFT에서 $\mathcal{X}[0] = \frac{1}{N}\sum_{n=0}^{N-1}x[n]$ (평균값과 수학적 동치) | p.8, Appendix B.1, Eq.(12-13) |
| 2 | FDT로 더 정상적인 표현 획득 가능 | 트렌드 신호의 분포 이동량이 미분 후 감소함을 수식으로 증명 | p.9, Appendix D |
| 3 | DERITS가 SOTA 대비 우수한 예측 성능 | 7개 데이터셋, MAE/RMSE 기준 Transformer 계열 대비 평균 20%↓ | p.5, Table 1 |
| 4 | FDT가 기존 정규화(RevIN, Dish-TS)보다 효과적 | LTSF-Linear backbone에서 FDT 적용 시 일관된 성능 향상 | p.6, Table 2 |
| 5 | 다차수 병렬 아키텍처가 단일 차수보다 우수 | 개별 차수 DERITS vs 다차수 DERITS 비교 | p.6, Figure 3 |
| 6 | $k=0$(미분 없음) 대비 미분($k \geq 1$) 적용 시 성능 향상 | Table 3 결과: Exchange k=2에서 MAE 0.035 (k=0: 0.041) | p.6, Table 3 |
| 7 | DERITS의 계산 효율성이 NS-Transformer 계열보다 우수 | 훈련 시간 비교: DERITS 12.57s vs NS-FEDformer 137.7s (H=96) | p.7, Table 4 |

---

### 2-1. 상세 설명

#### 🔴 해결하고자 하는 문제

비정상 시계열 예측에서의 분포 이동 문제:
- **기존 방법의 한계**: RevIN, Dish-TS 등이 사용하는 평균/표준편차 기반 정규화는 이론적으로 주파수 스펙트럼의 영주파수 성분($f=0$)만을 활용하는 것과 동치.
- **결과**: 시계열의 전체 분포 정보를 활용하지 못하는 정보 병목 현상 발생 → 예측 성능 저하.

> 📌 **정규화(Normalization)**: 데이터의 스케일이나 분포를 조정하는 전처리 과정. 시계열에서는 주로 평균 빼기/표준편차 나누기 형태로 사용.

---

#### 🟡 제안하는 방법 (수식 포함)

**[단계 1] 도메인 변환 (Domain Transformation)**

시간 도메인 신호 $X(t)$를 주파수 도메인으로 변환:

$$\mathcal{X}(f) = \mathcal{F}(X(t)) = \int_{-\infty}^{\infty} X(t)e^{-j2\pi ft}dt $$

$$= \underbrace{\int_{-\infty}^{\infty} X(t)\cos(2\pi ft)dt}_{\text{실수부 } Re(\mathcal{X})} + j\underbrace{\int_{-\infty}^{\infty} X(t)\sin(2\pi ft)dt}_{\text{허수부 } Im(\mathcal{X})}$$

- $\mathcal{F}$: 고속 푸리에 변환(FFT)
- $f$: 주파수 변수
- $t$: 적분 변수 (시간)
- $j$: 허수 단위 ($j = \sqrt{-1}$)

> 📌 **고속 푸리에 변환(FFT, Fast Fourier Transform)**: 신호를 시간 도메인에서 주파수 도메인으로 변환하는 알고리즘. 신호를 구성하는 주파수 성분들을 분해하여 분석 가능하게 함.

---

**[단계 2] 주파수 도함수 연산자 (Fourier Derivative Operator)**

**Definition 1** (Fourier Derivative Operator):

$$\mathcal{R}(\mathcal{X}(f)) := (j2\pi f)\mathcal{X}(f) $$

$k$차수 확장:

$$\mathcal{R}_k(\mathcal{X}(f)) = (j2\pi f)^k \mathcal{X}(f) $$

- $k$: 미분 차수 (order)
- $(j2\pi f)^k$: $k$차 주파수 도함수 가중치

**$k$차 FDT 최종 정의**:

$$\text{FDT}_k(X(t)) = (j2\pi f)^k \mathcal{F}(X(t)) $$

> 📌 **도함수/미분(Derivative)**: 함수의 변화율을 나타냄. 시계열에서 1차 미분은 인접 시점 간의 차이(변화량)에 해당. 미분을 통해 트렌드 제거 → 더 정상적인 신호로 변환.

---

**[명제 1] 주파수 도메인 미분 = 시간 도메인 미분 (등가성)**

$$\mathcal{F}\left(\frac{d^k X(t)}{dt^k}\right) = (j2\pi f)^k \mathcal{X}(f) $$

- $\frac{d^k}{dt^k}$: 시간 $t$에 대한 $k$차 미분 연산자

**증명 스케치** (Appendix B.2, p.8-9):

$$X(t) = \frac{1}{2\pi}\int_{-\infty}^{\infty}\mathcal{X}(f)e^{j2\pi ft}df $$

양변을 $t$로 미분:

$$\frac{dX(t)}{dt} = \mathcal{F}^{-1}((j2\pi f)\mathcal{X}(f)) $$

이를 $k$번 반복하면:

$$\frac{d^k X(t)}{dt^k} = \mathcal{F}^{-1}((j2\pi f)^k \mathcal{X}(f)) $$

---

**[역변환] Inverse FDT (iFDT)**

예측 결과를 시간 도메인으로 복원:

$$\text{iFDT}_k(\mathcal{X}(f)) = \mathcal{R}_k^{-1}(\mathcal{X}(f)) = \mathcal{F}^{-1}\left(\frac{1}{(j2\pi f)^k}\mathcal{X}(f)\right) $$

- $\mathcal{R}_k^{-1}$: $k$차 FDO의 역연산 (시간 도메인의 적분 연산과 동치)
- $\mathcal{F}^{-1}$: 역 푸리에 변환

> 📌 **가역 변환(Reversible Transformation)**: 원래 신호를 완벽하게 복원할 수 있는 변환. DERITS는 FDT→예측→iFDT의 과정에서 정보 손실 없이 복원 가능.

---

**[OFCN] 차수 적응형 주파수 필터**

$$\mathcal{H}'^k_t = \mathbf{m}_k \odot \mathbf{v}_k \mathcal{X}'^k_t = \overbrace{[\underbrace{1,\cdots,1}_{S/2^{(K-k)}},0,\cdots,0]}^{S} \odot \mathbf{v}_k \mathcal{X}'^k_t $$

- $\mathbf{m}_k$: 길이 $S$의 마스크 벡터 (차수 $k$에 따라 유지할 주파수 수 결정)
- $\mathbf{v}_k$: 차수 $k$에 대해 학습 가능한 랜덤 초기화 벡터
- $\mathcal{X}'^k_t$: 진폭 기준 내림차순 정렬된 주파수 성분
- $S$: 전체 주파수 수
- $K$: 총 브랜치 수 (최대 차수)
- $\odot$: 원소별 곱(Hadamard product)
- **의미**: 저차수일수록 더 많은 고주파 노이즈가 포함되므로, 저차수 브랜치에서 더 많은 주파수를 필터링

> 📌 **고주파 노이즈(High-frequency Noise)**: 신호에서 빠르게 변동하는 성분으로 예측에 불필요한 잡음. 저주파 성분이 주요 패턴을 담음.

---

**[OFCN] 푸리에 합성곱**

$$\mathcal{H}^k_t = \text{FourierConvolution}(\mathcal{H}'^k_t) = \mathcal{H}'^k_t \mathbf{W}_k $$

- $\mathbf{W}_k$: 차수 $k$의 주파수 도메인 합성곱을 위한 가중치 행렬 (브랜치 간 파라미터 비공유)
- **이론적 근거**: 합성곱 정리(Convolution Theorem)에 의해, 주파수 도메인에서의 곱셈 = 시간 도메인에서의 전역 합성곱

> 📌 **합성곱 정리(Convolution Theorem)**: "두 신호의 합성곱의 푸리에 변환 = 각 신호 푸리에 변환의 점별 곱". 이를 이용하면 주파수 도메인에서 단순 행렬곱으로 전역적 의존성 학습 가능.

---

**[전체 아키텍처] 병렬 적층 구조**

각 브랜치 $k$에서:

$$\mathcal{X}^k_t = \text{FDT}_k(\mathbf{X}_t), \quad k=1,2,\cdots,K $$

$$\mathcal{H}^k_t = \text{Order-adaptiveFourierConvolution}(k, \mathcal{X}^k_t) $$

$$\mathbf{H}^k_t = \text{iFDT}_k(\mathcal{H}^k_t), \quad k=1,2,\cdots,K $$

최종 예측:

$$\hat{\mathbf{Y}}_t = \text{MultilayerPerceptron}(\mathbf{H}^1_t, \mathbf{H}^2_t, \cdots, \mathbf{H}^K_t) $$

- $\mathbf{X}_t$: 타임스텝 $t$에서의 lookback 윈도우 입력 ($\in \mathbb{R}^{L \times D}$)
- $\hat{\mathbf{Y}}_t$: 예측 결과 ($\in \mathbb{R}^{H \times D}$)
- $L$: lookback 윈도우 길이, $H$: 예측 길이, $D$: 변량 수

---

#### 🟢 모델 구조 (Figure 2, p.4)

```
입력 Xₜ
  │
  ├─[Branch k=1]─ FFT ─ FDO(j2πf)¹ ─ OFCN(Filter+Conv) ─ iFDO ─ IFFT ─ H¹ₜ ─┐
  ├─[Branch k=2]─ FFT ─ FDO(j2πf)² ─ OFCN(Filter+Conv) ─ iFDO ─ IFFT ─ H²ₜ ─┤
  ├─      ...                                                                   ├─ MLP → Ŷₜ
  └─[Branch k=K]─ FFT ─ FDO(j2πf)ᴷ ─ OFCN(Filter+Conv) ─ iFDO ─ IFFT ─ Hᴷₜ ─┘
```

**3단계 구성**:
1. **Fourier Derivative Transformation (FDT)**: 시간→주파수 변환 후 주파수 도함수 적용
2. **Order-adaptive Fourier Convolution Network (OFCN)**: 차수별 필터링 + 주파수 합성곱
3. **Inverse FDT (iFDT)**: 예측 결과를 시간 도메인으로 복원

---

#### 🔵 성능 향상

| 비교 대상 | 성능 향상 | 출처 |
|-----------|-----------|------|
| Transformer 계열 (최고 기준) | MAE/RMSE 평균 20%↑ 감소 | p.6, Section 5.2 |
| FreTS, PatchTST | 대부분 데이터셋에서 DERITS 우세 | p.5, Table 1 |
| RevIN, Dish-TS (FDT 단독) | 모든 데이터셋에서 FDT 일관된 개선 | p.6, Table 2 |
| 훈련 속도 (H=96, Exchange) | DERITS: 12.57s vs NS-FEDformer: 137.7s | p.7, Table 4 |

#### 🔴 한계

1. **고차수 정보 손실**: 차수 $k$가 높아질수록 정보 손실로 인해 오히려 성능 저하 (Table 3: k=3 < k=2 in Exchange) (p.6)
2. **복소수 처리 복잡성**: FDT 출력이 복소수값이어서 직접 시각화가 어려움 (p.7, Section 5.5)
3. **최적 차수 $k$의 수동 설정**: 기본값 $k=2$로 설정하며, 데이터셋별 최적 차수 탐색 미흡 (p.8, Appendix A.3)
4. **단일 GPU 실험**: NVIDIA RTX 3090 단일 GPU 환경에서만 검증 (p.6)
5. **lookback 길이 증가 시 노이즈 증가 가능성** 언급 (p.7, Figure 4 분석)

---

## 3. 각 주장별 페이지/Figure/Table 번호

| 주장 | 위치 |
|------|------|
| 기존 정규화 = 영주파수 정규화 동치 | p.1 (Abstract), p.8 (Appendix B.1, Eq.12-13) |
| FDT 정의 및 수식 | p.3-4 (Section 4.1, Eq.1-5) |
| 병렬 적층 아키텍처 | p.4 (Section 4.2, Figure 2, Eq.6-9) |
| OFCN 설명 | p.4-5 (Section 4.3, Eq.10-11) |
| 전체 예측 성능 비교 | p.5-6 (Table 1) |
| 정규화 기법 비교 | p.6 (Table 2) |
| FDT 차수 영향 분석 | p.6 (Table 3) |
| 다차수 vs 단일 차수 비교 | p.6 (Figure 3) |
| Lookback 길이 영향 | p.7 (Figure 4) |
| 계산 효율성 | p.7 (Table 4) |
| 시각화 (원본 vs 도함수 신호) | p.7 (Figure 5) |
| 예측 사례 연구 | p.7 (Figure 6) |
| ILI 추가 실험 | p.9 (Table 6) |
| Exchange 추가 실험 | p.9 (Table 7) |
| 분포 이동 감소 이론적 분석 | p.9 (Appendix D) |

---

## 4. 저자 보고 결과 vs. 해석 분리

### 저자가 직접 보고한 결과 (원문 기반)

**연구 주제** (p.1, Abstract):
> "We propose to utilize the whole frequency spectrum to transform time series to make full use of data distribution from the frequency perspective."

**핵심 수식** (p.3-5):

$$\text{FDT}_k(X(t)) = (j2\pi f)^k \mathcal{F}(X(t)) \quad \text{[Eq.3]}$$

$$\mathcal{H}^k_t = \mathcal{H}'^k_t \mathbf{W}_k \quad \text{[Eq.11]}$$

**실험 결과** (p.6, Section 5.2):
> "Compared with the best results of transformer-based models, DERITS has an average decrease of more than 20% in MAE and RMSE."

**효율성** (p.7, Table 4):
> DERITS 훈련 시간: H=96에서 12.57초 (NS-FEDformer: 137.7초, NS-Autoformer: 44.41초)

**FDT 차수 실험** (p.6, Table 3):
- Exchange 데이터셋, H=96: k=0(0.041 MAE) → k=2(0.035 MAE, 최고) → k=3(0.036 MAE)

---

### 🔍 해석자(리뷰어)의 분석

1. **이론적 기여의 실용적 의의**: 평균 정규화 = 영주파수 정규화라는 등가성은 수학적으로 타당하나, **실제 시계열에서 표준편차 정규화가 영주파수 이상을 포함하는지 여부**에 대한 논의가 부족함. 표준편차는 신호의 분산과 관련되며 단순히 영주파수만과 동치는 아닐 수 있음.

2. **20% 성능 향상 수치**: Transformer 기반 모델 중 "최고 성능"을 기준으로 한 수치. FreTS, PatchTST 대비로는 개선폭이 훨씬 작음 (Table 1에서 소수점 2-3자리 차이). **비교 기준 선택에 따라 수치가 달라질 수 있음**.

3. **다차수 병렬 구조의 실제 기여**: Figure 3에서 multi-order가 individual-order보다 우수하나, 이는 **파라미터 수 증가 효과**와 **다차수 정보 융합 효과**를 구분하지 않음. 파라미터 수를 동일하게 맞춘 비교가 필요함.

4. **효율성 비교의 공정성**: Table 4의 훈련 시간 비교는 NS-Transformer 계열과만 비교하며, **FreTS, PatchTST 등 더 경량화된 모델과의 속도 비교는 제시되지 않음**.

---

## 5. 통계적 취약점 및 비교 불가능한 수치

> ⚠️ **통계적 취약 부분**

| 문제 유형 | 내용 | 위치 |
|-----------|------|-------|
| ⚠️ 통계적 유의성 검증 없음 | MAE/RMSE 수치 비교만 제시, 표준편차/신뢰구간/p-value 없음 | Table 1-3 전반 |
| ⚠️ 단일 시드 실험 가능성 | 실험 반복 횟수 미기재, 랜덤 시드 설정 미보고 | p.6, Section 5.1 |
| ⚠️ 비교 불공정: 파라미터 수 미공개 | DERITS의 파라미터 수 vs 비교 모델 비교 없음 | Section 5 전반 |
| ⚠️ "20% 이상 감소" 기준 불명확 | 어떤 모델 대비, 어떤 데이터셋 평균인지 불명확 | p.6, Section 5.2 |
| ⚠️ Traffic 데이터셋 예측 길이 상이 | Traffic만 H∈{48,96,192,336}, 나머지는 H∈{96,192,336,720} | p.5, Table 1 |
| ⚠️ 일부 비교 불가능 수치 | Table 7에서 NSTransformer는 메모리 부족('-')으로 일부 결과 없어 직접 비교 불가 | p.9, Table 7 |
| ⚠️ ILI 데이터셋 소규모 | ILI 데이터셋 샘플 수 966개로 매우 소규모 → 일반화 신뢰성 의문 | p.8, Table 5 |
| ⚠️ 효율성 비교 범위 제한 | 훈련 시간 비교가 Exchange 데이터셋, NS-Transformer 계열에만 한정 | p.7, Table 4 |

---

## 6. 논문이 답하지 않는 질문

| # | 미해결 질문 |
|---|------------|
| 1 | **최적 차수 $K$ 자동 결정 방법**은 무엇인가? 현재 기본값 $k=2$는 경험적으로 설정됨. |
| 2 | **단변량(univariate) vs 다변량(multivariate) 성능 차이**는? 논문은 다변량 위주 실험만 수행. |
| 3 | **복소수 주파수 표현에서의 위상(phase) 정보** 활용은? 현재 진폭(amplitude) 기준 정렬만 사용. |
| 4 | **조건부 이동(conditional shift)**에 대한 FDT의 이론적 효과**는? Appendix D는 공변량 이동만 분석. |
| 5 | **초장기 예측(H>720)**에서의 성능은? 실험은 H≤720으로 제한. |
| 6 | **비규칙적 샘플링(irregular sampling)** 시계열에 FDT 적용 가능성은? |
| 7 | **다른 백본 모델(Mamba, TimesNet 등)**에 FDT를 플러그인으로 적용 시 효과는? (LTSF-Linear 외) |
| 8 | **이산 신호에서 FDT의 정밀한 동작**은? 논문은 연속 신호 기반 수식만 제시. |
| 9 | **표준편차 정규화와 영주파수의 동치 여부**에 대한 완전한 수학적 분석이 누락됨. |
| 10 | **분포 이동 측정 지표(Wasserstein distance 등)**를 이용한 FDT의 정량적 이동 감소량은? |

---

## 7. 가장 중요한 그림 5개 해석

### 📊 Figure 1 (p.1) — 기존 방법 vs 제안 방법 비교

**구성**: (a) 영주파수 성분만 활용하는 기존 방법, (b) 전체 주파수 스펙트럼 활용하는 DERITS

**해석**:
- (a)에서 기존 방법은 주파수 스펙트럼 중 $f=0$ 부분(직류 성분, 즉 평균)만을 이용해 정규화. 이는 시계열의 저주파-고주파 구조를 무시.
- (b)에서 DERITS는 모든 주파수 성분을 활용하여 변환을 수행함으로써, 시계열의 주기성, 추세, 노이즈 등 전체 분포 구조를 반영.
- **핵심 메시지**: 분포 정보의 완전 활용 여부가 예측 성능의 근본적 차이를 만든다는 연구의 출발점.

---

### 📊 Figure 2 (p.4) — DERITS 전체 아키텍처

**구성**: 좌측 FDT 블록 → 중앙 OFCN 블록 → 우측 iFDT 블록의 병렬 적층 구조

**해석**:
- 각 브랜치 $k$가 독립적인 FDT($j2\pi f$의 $k$승 가중)를 수행하여 서로 다른 차수의 도함수 표현을 학습.
- OFCN 내부의 마스크($m_k$)와 가중치($v_k$, $W_k$)는 브랜치별로 독립 학습 → **차수별 특화 학습** 가능.
- iFDT로 복원 후 MLP로 융합 → 다차수 보완적 정보 통합.
- **아키텍처의 핵심 강점**: 수학적으로 엄밀한 가역성(reversibility)을 보장하는 설계.

> 📌 **가역성(Reversibility)**: 변환 후 역변환을 통해 원래 신호를 완벽히 복원 가능한 성질. 정보 손실 없이 변환 공간에서 학습이 가능함을 보장.

---

### 📊 Figure 3 (p.6) — 다차수 vs 개별 차수 성능 비교

**구성**: Exchange, Weather, ILI 3개 데이터셋에서 1차/2차/3차 개별 모델 vs 다차수 융합 DERITS 비교 (MAE 기준)

**해석**:
- 세 데이터셋 모두에서 multi-order DERITS가 어떤 individual-order보다도 일관되게 낮은 MAE 달성.
- 특히 ILI 데이터셋에서 격차가 가장 두드러짐 → 소규모 데이터셋에서 다차수 융합의 효과가 더 큼.
- Exchange에서는 2차 individual-order가 3차보다 우수하지만, multi-order는 둘 다 초과 → 차수 간 상호 보완적 정보가 존재함을 시사.
- **해석 주의**: 파라미터 수 통제 없는 비교이므로 성능 향상 원인이 정보 융합인지 파라미터 증가인지 불분명.

---

### 📊 Figure 5 (p.7) — 원본 신호 vs FDT 도함수 신호 시각화

**구성**: Weather 데이터셋(a)과 Exchange 데이터셋(b)에서 원본 시계열과 FDT 적용 후 시간 도메인으로 복원한 신호 비교

**해석**:
- **원본 신호(위)**: 뚜렷한 추세(trend) 변화와 비정상적 진동이 관찰됨.
- **도함수 신호(아래)**: 추세가 제거되고 변화량(gradient) 중심의 신호로 변환 → 시각적으로 더 정상적(stationary)인 패턴.
- Appendix D의 이론: 선형 추세 $c_1 t$는 미분 후 상수 $c_1$이 되어 분포 이동이 제거됨.
- **한계**: 복소수 FDT 출력을 다시 iFDT로 복원하여 시각화한 것이므로, 직접적인 주파수 도메인 시각화가 아님.

---

### 📊 Figure 6 (p.7) — 예측 사례 연구 (DERITS vs NSTransformer)

**구성**: Exchange 데이터셋에서 lookback=96, 예측 길이=96으로 설정한 예측 결과 vs 실제값 비교

**해석**:
- **DERITS(a)**: 시계열이 급격히 분포 이동하는 구간(중반부 상승)에서도 Ground Truth와 근접하게 추종.
- **NSTransformer(b)**: 동일 구간에서 예측값이 실제값과 크게 벗어나는 현상 관찰.
- **시사점**: FDT를 통해 비정상 구간에서도 안정적인 학습 공간을 제공함으로써 분포 이동에 강건한 예측이 가능함을 시각적으로 확인.
- **해석 주의**: 단일 사례(case study)로, 통계적 대표성은 보장되지 않음.

---

## 8. 결론 및 후속 연구

### 8-0. 저자들이 제시한 시사점과 후속 연구 계획

**저자 시사점** (p.7-8, Section 6):
1. 비정상 시계열 예측 문제를 주파수 도함수 관점으로 재해석하는 새로운 패러다임 제시.
2. 주파수 전체 스펙트럼을 활용한 가역 변환(FDT)이 분포 이동 완화에 효과적임을 이론적·실험적으로 검증.
3. "distribution shifts and non-stationarity are actually a pervasive and crucial topic" → 향후 관련 연구 촉진 기대.

**후속 연구 계획** (원문에 명시된 내용):
- 저자들은 구체적 후속 연구 계획을 논문에 명시하지 않음. 다만 코드 공개 예정 언급 ("The codes will be publicly available soon", p.8, Appendix A.3).

---

### 8-1. 모델의 일반화 성능 향상 가능성 🎯

#### 현재 모델의 일반화 한계

1. **데이터셋 다양성 제한**: 7개 데이터셋 모두 정형화된 벤치마크. 의료, 음성, 지진 등 도메인별 특수성이 강한 데이터에서의 검증 부재.
2. **입력 길이 의존성**: Figure 4에서 lookback 길이 변화에 따른 성능 변동이 관찰 → 최적 lookback 길이가 데이터셋별로 상이.
3. **다변량 교차 의존성 미활용**: OFCN의 Fourier convolution이 변수 간 관계보다 시간적 의존성에만 집중.

#### 일반화 향상을 위한 구체적 방향

**① 적응적 차수 선택 메커니즘**
- 현재 기본값 $k=2$를 데이터셋별로 자동 결정하는 메타학습(meta-learning) 또는 신경 아키텍처 탐색(NAS) 적용 가능.
- 예: 입력 신호의 정상성 검정(ADF Test) 결과를 기반으로 $k$ 자동 조정.

> 📌 **ADF 검정(Augmented Dickey-Fuller Test)**: 시계열의 단위근(unit root) 유무를 검정하는 통계적 방법. 비정상성 수준을 수치화하여 최적 미분 차수 결정에 활용 가능.

**② 도메인 적응(Domain Adaptation) 연계**
- FDT의 역변환 가역성을 이용하여, 소스 도메인에서 학습한 모델을 타겟 도메인에 적응시키는 전이 학습 가능성 탐색.

**③ 변수 간 그래프 구조 도입**
- OFCN을 그래프 신경망(GNN)과 결합하여 다변량 시계열의 변수 간 관계를 주파수 도메인에서 함께 학습 (FourierGNN [Yi et al., 2024]과의 결합 가능성).

**④ 불규칙 샘플링 확장**
- 연속 웨이블릿 변환(CWT) 또는 비균일 FFT(NUFFT)를 FDT에 통합하여 의료·금융 데이터의 불규칙 샘플링 문제 해결.

> 📌 **비균일 FFT(NUFFT, Non-Uniform FFT)**: 균일하지 않은 시간 간격의 신호에도 FFT를 적용할 수 있게 하는 알고리즘.

**⑤ 데이터 증강과의 결합**
- FDT 공간에서의 주파수 혼합(Frequency Mixing) 증강으로 학습 데이터의 다양성 향상 → 분포 외(out-of-distribution) 일반화 능력 강화.

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 아래 내용은 2024년 초까지의 학습 데이터 기반이며, 일부 성능 수치는 각 논문의 원문을 직접 확인하시기 바랍니다.

#### 주요 관련 연구 비교

| 논문 | 연도 | 핵심 방법 | DERITS와의 차별점 |
|------|------|-----------|------------------|
| **RevIN** [Kim et al., 2021, ICLR] | 2021 | 인스턴스별 평균/표준편차 정규화 후 역정규화 | 영주파수만 활용 (DERITS가 이론적으로 상위 개념) |
| **Autoformer** [Wu et al., 2021, NeurIPS] | 2021 | 자기상관 메커니즘 + 분해 | 시간 도메인 분해, 주파수 도메인 미분 미사용 |
| **FEDformer** [Zhou et al., 2022, ICML] | 2022 | 주파수 강화 Attention | 주파수 필터링에 집중, 도함수 개념 없음 |
| **Dish-TS** [Fan et al., 2023, AAAI] | 2023 | 학습 가능한 통계량으로 분포 이동 처리 | 여전히 통계 기반, 주파수 전체 활용 안 함 |
| **PatchTST** [Nie et al., 2023, ICLR] | 2023 | 패치 기반 Transformer | 비정상성 문제 직접 해결 안 함 |
| **FreTS** [Yi et al., 2023, NeurIPS] | 2023 | 주파수 도메인 MLP | 도함수 변환 없이 주파수 학습 |
| **SAN** [Liu et al., 2023, NeurIPS] | 2023 | 슬라이스 통계 기반 적응적 정규화 | 여전히 통계 기반 접근 |
| **FourierGNN** [Yi et al., 2024, NeurIPS] | 2024 | 그래프+FFT 결합 | 변수 간 관계 중점, 비정상성 처리 미흡 |
| **DERITS** (본 논문) | 2024 | 주파수 도함수 변환 + 병렬 적층 | 전체 주파수 스펙트럼 + 수학적 등가성 이론 + 도함수로 비정상성 해소 |

#### 주요 트렌드 분석

**2020-2022**: Transformer 아키텍처 시계열 적용 전성기 (Informer, Autoformer, FEDformer)

**2022-2023**: Transformer 한계 지적 및 단순 모델 재조명 (DLinear, PatchTST), 주파수 도메인 학습 부상 (FreTS)

**2023-2024**: 비정상성과 분포 이동 문제의 심층 연구 (SAN, DERITS), 상태공간모델(Mamba) 시계열 적용 시작

#### DERITS가 앞으로의 연구에 미치는 영향

1. **이론적 기여**: 평균 정규화 = 영주파수 정규화 등가성 증명은 시계열 정규화 방법론 연구의 새로운 분석 틀 제공.
2. **방법론적 기여**: FDT의 플러그인 적용 가능성(Table 2에서 LTSF-Linear 백본에 적용)은 다양한 모델에 통합 가능성 시사.
3. **연구 방향**: 주파수 도메인에서의 비정상성 처리라는 새로운 방향을 열어 후속 연구 유도 가능.

#### 앞으로 연구 시 고려할 점

**① 공정한 비교 설계**
- 파라미터 수, GPU 메모리, 훈련 시간을 동시에 통제한 비교 실험 필수.
- 통계적 유의성 검증(paired t-test, Wilcoxon test 등) 포함.

**② 최신 경쟁 모델과의 비교 확대**
- Mamba 기반 시계열 모델 (예: S-Mamba, TimeMachine), iTransformer [Liu et al., 2024], TimesNet 등과의 비교 필요.

> 📌 **Mamba**: 선형 복잡도의 상태공간모델(SSM) 기반 아키텍처로 2023년 이후 시계열 분야에 빠르게 적용되고 있는 최신 딥러닝 모델.

**③ 적용 도메인 다양화**
- 의료 시계열(ECG, EEG), 금융 고빈도 데이터, 기후 예측 등 도메인별 검증.

**④ 이산 신호에서의 엄밀한 수학적 분석**
- 논문의 수식은 연속 신호 기반이나 실제 구현은 이산 FFT 사용 → 두 사이의 근사 오차 분석 필요.

**⑤ 해석 가능성(Interpretability) 연구**
- 각 주파수 성분과 브랜치가 실제 어떤 시계열 패턴(계절성, 추세, 노이즈)에 대응하는지 분석.

**⑥ 온라인 학습(Online Learning) 확장**
- 실시간 분포 이동 감지 및 FDT 차수 동적 업데이트를 통한 온라인 시계열 예측 시스템 구축.

---

## 📚 참고 자료 (논문 원문 인용 기반)

### 논문 내 직접 인용된 주요 참고문헌:
1. **Fan et al. (2024)** - "Deep Frequency Derivative Learning for Non-stationary Time Series Forecasting" *(본 논문)*, arXiv:2407.00502v1
2. **Kim et al. (2021/2022)** - "Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift", ICLR
3. **Yi et al. (2023b)** - "Frequency-domain MLPs are more effective learners in time series forecasting", NeurIPS 2023
4. **Yi et al. (2024)** - "FourierGNN: Rethinking Multivariate Time Series Forecasting from a Pure Graph Perspective", NeurIPS 2024
5. **Fan et al. (2023)** - "Dish-TS: A General Paradigm for Alleviating Distribution Shift in Time Series Forecasting", AAAI 2023
6. **Liu et al. (2022b)** - "Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting", NeurIPS 2022
7. **Liu et al. (2023)** - "Adaptive Normalization for Non-stationary Time Series Forecasting: A Temporal Slice Perspective", NeurIPS 2023
8. **Nie et al. (2023)** - "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers", ICLR 2023
9. **Zhou et al. (2022)** - "FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting", ICML 2022
10. **Wu et al. (2021)** - "Autoformer: Decomposition Transformers with Auto-Correlation for Long-term Series Forecasting", NeurIPS 2021
11. **Zeng et al. (2022/2023)** - "Are Transformers Effective for Time Series Forecasting?", arXiv:2205.13504
12. **Katznelson (1970)** - "An Introduction to Harmonic Analysis", Cambridge University Press (합성곱 정리 근거)
13. **Nussbaumer and Nussbaumer (1982)** - "The Fast Fourier Transform", Springer

### 외부 참고 자료 (해석에 활용, 원문 미수록):
- Vaswani et al. (2017). "Attention Is All You Need." NeurIPS. *(Transformer 기반 모델 이해)*
- Salinas et al. (2020). "DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks." *International Journal of Forecasting.*
