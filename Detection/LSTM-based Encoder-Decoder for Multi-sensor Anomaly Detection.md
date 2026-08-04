# LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection

> **참고 자료**
> - Malhotra et al. (2016). *LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection*. arXiv:1607.00148v2. ICML 2016 Anomaly Detection Workshop.
> - 논문 내 인용 문헌들 (Hochreiter & Schmidhuber 1997; Sutskever et al. 2014; Cho et al. 2014; Malhotra et al. 2015 등)
> - ⚠️ **주의**: 2020년 이후 최신 연구 비교(8-2절)는 제 학습 데이터 기반이며, 일부 수치는 원본 논문 미확인 정보가 포함될 수 있음을 사전 고지합니다.

---

## 1. Executive Summary (10문장 이내)

1. 본 논문은 다중 센서 시계열 데이터에서 비정상(anomaly)을 탐지하기 위한 **LSTM 기반 Encoder-Decoder 모델(EncDec-AD)**을 제안한다.
2. 기계 장비의 센서 데이터는 외부 제어 변수나 환경 조건에 의해 **본질적으로 예측 불가능(unpredictable)**한 경우가 많아, 예측 오차 기반 기존 방법이 무력화된다.
3. EncDec-AD는 **정상 시계열만으로 훈련**되어 정상 패턴을 재구성(reconstruct)하는 법을 학습하며, 재구성 오차가 높은 구간을 이상으로 판단한다.
4. 인코더는 입력 시계열을 고정 차원 벡터로 압축하고, 디코더는 이를 이용해 시계열을 역순으로 재구성한다.
5. 재구성 오차 벡터에 **다변량 정규분포(Gaussian likelihood)**를 피팅하여 이상 점수를 산출한다.
6. 실험은 Power Demand, Space Shuttle, ECG(공개 데이터)와 실제 엔진 데이터(Engine-P, Engine-NP) 총 5개 데이터셋에서 수행되었다.
7. **예측 불가능한 Engine-NP** 데이터에서 기존 LSTM-AD($F_{0.05}=0.03$)를 크게 압도하며 $F_{0.1}=0.93$을 달성했다.
8. 주기적·비주기적·준주기적 시계열, 단기(L=30)~장기(L=500) 시계열 모두에서 강건한 성능을 보였다.
9. 단, 예측 가능한 데이터셋에서는 예측 기반 모델(LSTM-AD)이 더 높은 $F_\beta$를 기록하여 적용 시나리오를 고려해야 한다.
10. 본 연구는 이상 데이터가 희소하거나 부재한 실제 산업 환경에서 비지도/준지도 방식 이상 탐지의 가능성을 열었다.

### 1-1. 연구의 목적과 필요성

| 구분 | 내용 |
|------|------|
| **현실적 문제** | 굴착기·엔진 등 기계의 센서 데이터는 외부 부하, 수동 조작 등 미관측 요인으로 인해 예측이 불가능함 (p.1, Figure 1) |
| **기존 방법의 한계** | EWMA, SVR, LSTM-AD 등 예측 오차 기반 모델은 시계열이 예측 불가능할 때 작동 불가 (p.1) |
| **이상 데이터 희소성** | 기계는 정기 유지보수로 이상 데이터가 극히 드물어 분류 모델 학습이 곤란함 (p.2) |
| **연구 목적** | 정상 데이터만으로 학습 가능하며, 예측 가능·불가능 시계열 모두에서 이상을 탐지하는 범용 모델 개발 |

---

## 2. 핵심 주장과 근거 표

| # | 핵심 주장 | 근거 / 실험 결과 | 위치 |
|---|-----------|-----------------|------|
| 1 | EncDec-AD는 예측 불가능 시계열에서도 이상 탐지 가능 | Engine-NP: $F_{0.1}=0.83$, TPR/FPR= $\infty$ | Table 2, p.3 |
| 2 | 재구성 기반 접근이 예측 기반보다 범용적 | LSTM-AD의 Engine-NP $F_{0.05}=0.03$ vs EncDec-AD $F_{0.1}=0.93$ | p.4, Sec 3.2 |
| 3 | 정상 데이터만으로 훈련 가능 | 훈련셋 $s_N$은 정상 시퀀스만 사용 | p.2, Sec 2.1 |
| 4 | 단기~장기 시계열 모두 처리 가능 | L=30(Engine) ~ L=500(Space Shuttle) | p.1 Abstract, Table 2 |
| 5 | Positive Likelihood Ratio > 1.0 (모든 데이터셋) | Power: 33.0, Space Shuttle: 4.9, Engine-NP: $\infty$ | Table 2, p.3 |
| 6 | 예측 가능 데이터에서는 LSTM-AD가 더 우수 | Space Shuttle LSTM-AD $F_{0.1}=0.84$ > EncDec-AD $F_{0.05}=0.81$ | p.4, Sec 3.2 |

### 2-1. 세부 설명

#### 해결하고자 하는 문제

- 외부 변수(부하, 환경, 수동 조작)가 센서에 포착되지 않아 **본질적으로 예측 불가능한 시계열**에서의 이상 탐지
- 이상 데이터가 희소하여 **지도학습 기반 분류가 불가능한 상황**

#### 제안하는 방법 (수식 포함)

**① 모델 훈련 목적함수**

$$\mathcal{L} = \sum_{X \in s_N} \sum_{i=1}^{L} \left\| \mathbf{x}^{(i)} - \mathbf{x}'^{(i)} \right\|^2$$

여기서 $s_N$은 정상 훈련 시퀀스 집합, $\mathbf{x}^{(i)}$는 원래 값, $\mathbf{x}'^{(i)}$는 재구성 값 (p.2, Sec 2.1)

**② 디코더 출력 계산**

$$\mathbf{x}'^{(i)} = \mathbf{w}^T \mathbf{h}_D^{(i)} + \mathbf{b}$$

여기서 $\mathbf{w} \in \mathbb{R}^{c \times m}$, $\mathbf{b} \in \mathbb{R}^m$, $c$는 LSTM 유닛 수, $m$은 센서 차원 수 (p.2)

**③ 재구성 오차 벡터**

$$\mathbf{e}^{(i)} = \left| \mathbf{x}^{(i)} - \mathbf{x}'^{(i)} \right|$$

**④ 이상 점수 (Mahalanobis distance 기반)**

$$a^{(i)} = \left( \mathbf{e}^{(i)} - \boldsymbol{\mu} \right)^T \boldsymbol{\Sigma}^{-1} \left( \mathbf{e}^{(i)} - \boldsymbol{\mu} \right)$$

여기서 $\boldsymbol{\mu}$, $\boldsymbol{\Sigma}$는 검증셋 $v_{N1}$의 오차 벡터로부터 MLE로 추정된 정규분포 파라미터 (p.2, Sec 2.2)

**⑤ 임계값 최적화 기준**

$$F_\beta = \frac{(1 + \beta^2) \times P \times R}{\beta^2 P + R}$$

$\beta < 1$로 설정하여 정밀도(Precision) 우선 최적화 ($v_{N2}$, $v_A$ 검증셋에서 $\tau$, $c$ 선택) (p.2-3)

#### 모델 구조

```
입력 시계열 X = {x(1), x(2), ..., x(L)}
        ↓
[LSTM Encoder] → h_E(L) (고정 차원 벡터 표현)
        ↓ (초기 상태로 전달)
[LSTM Decoder] → 역순 재구성 {x'(L), x'(L-1), ..., x'(1)}
        ↓
[Linear Layer] w^T h_D + b
        ↓
재구성 오차 e(i) → N(μ, Σ) 피팅 → 이상 점수 a(i)
```

- 인코더/디코더: 각 단일 은닉층, $c$개의 LSTM 유닛 (p.3)
- 역순 재구성: Sutskever et al. (2014) 방식 채택 (p.2)
- 옵티마이저: Adam (Kingma & Ba, 2014) (p.3)

#### 성능 향상 및 한계

| 구분 | 내용 |
|------|------|
| **성능 향상** | 예측 불가 시계열(Engine-NP)에서 $F_{0.1}$: 0.03 → 0.93 (LSTM-AD 대비) |
| **강건성** | 주기적/비주기적/준주기적, L=30~500 모두 처리 |
| **한계 ①** | 예측 가능 데이터에서는 LSTM-AD가 더 우수 |
| **한계 ②** | Engine 데이터의 경우 PCA로 차원 축소 후 단변량 처리 (12→1차원, 분산 설명 72%/61%) |
| **한계 ③** | 이상 레이블의 정확한 위치 미확인 (수리 날짜 기준 근사 레이블) |
| **한계 ④** | ECG 데이터에서 $F_\beta=0.65$로 상대적으로 낮은 성능 |
| **한계 ⑤** | 단일 은닉층만 실험, 깊은 구조 미탐색 |

---

## 3. 주장별 페이지/Figure/Table 번호

| 주장 | 근거 위치 |
|------|-----------|
| 예측 불가능 시계열 문제 정의 | p.1, Figure 1(a)(b) |
| EncDec-AD 모델 구조 | p.2, Figure 2 |
| 훈련 목적함수 | p.2, Sec 2.1 |
| 이상 점수 산출 방법 | p.2, Sec 2.2 |
| 데이터셋 특성 | p.3, Table 1 |
| 전체 성능 결과 | p.3, Table 2 |
| 정상/이상 재구성 시각화 | p.3-4, Figure 3(a)-(j) |
| LSTM-AD vs EncDec-AD 비교 | p.4, Sec 3.2 Observation 3) |

---

## 4. 저자 보고 결과 vs. 분석자 해석 분리

### 저자가 직접 보고한 결과

| 항목 | 저자 보고 내용 |
|------|--------------|
| **연구 주제** | LSTM Encoder-Decoder를 이용한 다중 센서 시계열 이상 탐지 |
| **방법** | 정상 시계열 재구성 학습 후 Mahalanobis 거리 기반 이상 점수 산출 |
| **Engine-NP 결과** | EncDec-AD: P=1.0, R=0.01, $F_{0.05}=0.83$, TPR/FPR= $\infty$ (Table 2) |
| **Engine-NP LSTM-AD 비교** | LSTM-AD: P=0.03, R=0.07, $F_{0.05}=0.03$, TPR/FPR=1.9 (p.4) |
| **Space Shuttle LSTM-AD** | $F_{0.1}=0.84$ (LSTM-AD) vs $F_{0.05}=0.81$ (EncDec-AD) (p.4) |
| **Positive LR** | 모든 데이터셋에서 1.0 초과 (Table 2) |

### 분석자(필자)의 해석

| 항목 | 해석 |
|------|------|
| **일반화 가능성** | 저자는 "robust"라 주장하지만, 5개 데이터셋 중 3개는 단변량이고 엔진 데이터는 비공개(proprietary)라 독립적 재현 및 일반화 검증이 어려움 ⚠️ |
| **Recall 값** | 대부분의 데이터셋에서 R이 매우 낮음(0.005~0.08). $\beta<1$ 설정으로 Precision 우선화했지만, 실제 안전 critical 시스템에서는 높은 Recall이 필수적일 수 있음 |
| **PCA 전처리** | 12차원 엔진 데이터를 PCA 1성분으로 축소(Engine-P: 72%, Engine-NP: 61% 분산 설명). 39%의 정보 손실이 성능에 미치는 영향 미분석 |
| **역순 재구성** | Sutskever et al. (2014)의 방식 채택 이유를 실험적으로 검증하지 않음 |
| **임계값 설정** | ECG에서 검증 이상 시퀀스($v_A$) 없이 $\tau = \mu_a + \sigma_a$로 휴리스틱 설정 → 신뢰도 저하 ⚠️ |

---

## 5. 통계적 취약점 및 비교 불가능한 수치 ⚠️

| 취약점 유형 | 구체적 내용 |
|------------|-------------|
| **⚠️ 샘플 수 극히 적음** | ECG 이상 시퀀스: $N_a=1$개. 단 1개 이상 샘플로 성능 평가 불가 신뢰성 있음 |
| **⚠️ 엔진 데이터 비공개** | Engine-P/NP는 TCS 내부 데이터. 독립 재현 불가, 외부 검증 불가 |
| **⚠️ $F_\beta$ 기준 불통일** | Power Demand는 $\beta=0.1$, 나머지는 $\beta=0.05$ 사용 → 데이터셋 간 직접 비교 불가 |
| **⚠️ TPR/FPR = $\infty$** | Engine-NP, ECG에서 FPR=0이므로 수치적으로 $\infty$ 보고. 분모가 0인 경우의 통계적 의미 불명확 |
| **⚠️ 하이퍼파라미터 탐색 미보고** | $c$ (LSTM 유닛 수) 선택 범위, 탐색 방법론 미공개 |
| **⚠️ 통계적 유의성 검증 없음** | 랜덤 시드, 반복 실험, 신뢰 구간 등 미보고 |
| **⚠️ LSTM-AD 비교의 공정성** | $\beta$ 값이 다른 조건에서 비교($F_{0.1}$ vs $F_{0.05}$) |

---

## 6. 논문이 답하지 않는 질문

| # | 미해결 질문 |
|---|-----------|
| 1 | 인코더/디코더의 **레이어 깊이**를 증가시키면 성능이 어떻게 변하는가? (단일 레이어만 실험) |
| 2 | **역순 재구성(reverse reconstruction)**이 순방향보다 항상 우월한가? 비교 실험 없음 |
| 3 | **온라인(실시간) 이상 탐지**에 적용 시 계산 비용과 지연(latency)은? |
| 4 | 이상의 **종류(point anomaly vs. contextual anomaly vs. collective anomaly)**에 따른 성능 차이는? |
| 5 | PCA 대신 **원본 12차원 다변량 시계열**을 직접 처리하면 성능이 향상되는가? |
| 6 | **임계값 $\tau$의 시간적 안정성**: 계절 변화, 드리프트가 있을 때 재학습 주기는? |
| 7 | **데이터 불균형** 비율이 다를 때 성능 변화는? (현재 이상 시퀀스 비율이 데이터셋마다 상이) |
| 8 | 정상 데이터에 **소량의 오염(contamination)**이 포함될 경우 모델 강건성은? |
| 9 | **Attention 메커니즘** 추가 시 성능 개선 가능성은? |
| 10 | 훈련에 사용된 정상 시퀀스 수가 줄어들 때 성능 저하 임계점은? |

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1. 수동 제어 센서 readings (p.1)

- **(a) Predictable**: 이진(high/low) 외부 제어를 갖는 Engine-P의 제어 변수. 명확한 패턴 존재
- **(b) Unpredictable**: 연속 범위 내에서 빈번히 변동하는 Engine-NP의 제어 변수. 극도의 불규칙성
- **해석**: 논문의 핵심 동기를 시각화. 동일한 유형의 제어 변수라도 특성에 따라 예측 가능성이 완전히 달라짐을 보여주며, 예측 기반 방법의 한계와 재구성 기반 방법의 필요성을 직관적으로 설득함

---

### Figure 2. LSTM Encoder-Decoder 추론 단계 (p.2)

$$\mathbf{h}_D^{(3)} = \mathbf{h}_E^{(3)}, \quad \mathbf{x}'^{(3)} = \mathbf{w}^T \mathbf{h}_D^{(3)} + \mathbf{b}$$

- $L=3$ 시퀀스 예시로 인코더-디코더 정보 흐름을 명확히 도식화
- **인코더**: $\mathbf{x}^{(1)} \to \mathbf{x}^{(2)} \to \mathbf{x}^{(3)}$ 순방향 처리, $\mathbf{h}_E^{(3)}$에 전체 시퀀스 압축
- **디코더**: $\mathbf{h}_D^{(3)}$에서 시작하여 $\mathbf{x}'^{(3)} \to \mathbf{x}'^{(2)} \to \mathbf{x}'^{(1)}$ 역순 재구성
- **해석**: 디코더가 인코더의 최종 은닉 상태만을 초기값으로 사용하는 병목(bottleneck) 구조가 핵심. 이 압축이 정상 패턴의 본질만을 포착하게 하여 이상 탐지를 가능하게 함

---

### Figure 3(b) & 3(h). Power-A 및 Engine-NP-A (p.3)

- **Power-A (3b)**: 첫째 날 전력 수요가 비정상적으로 낮음(빨간색 표시). 재구성 시퀀스(녹색)는 정상 주중 패턴으로 복원하여 높은 이상 점수(빨간색 하단 그래프) 발생
- **Engine-NP-A (3h)**: 예측 불가능한 시계열임에도 정상 시퀀스(3g)와 이상 시퀀스(3h)에서 재구성 오차가 명확히 다름
- **해석**: 예측 불가능한 시계열(Engine-NP)에서도 인코더-디코더가 정상 범위 내 변동 패턴은 학습하고, 비정상 구조는 재구성 실패함을 시각적으로 입증. EncDec-AD의 핵심 작동 원리 검증

---

### Figure 3(c) & 3(d). Space Shuttle 정상/이상 (p.3)

- **Space Shuttle-N (3c)**: TEK17 - 정상적인 주기적 valve 개폐 패턴. 재구성이 매우 정확
- **Space Shuttle-A (3d)**: TEK14 - 이상 구간(빨간색)에서 밸브 동작 패턴이 변형됨. 이상 점수가 해당 구간에서 급등
- **해석**: L=500이라는 장시간 시퀀스에서도 이상 위치를 정확히 포착. LSTM 인코더의 장기 의존성 학습 능력이 실제 이상 탐지에 유효함을 보여줌

---

### Figure 3(i) & 3(j). ECG 정상/이상 (p.3)

- **ECG-N (3i)**: 정상 심전도 패턴. 재구성 오차가 낮고 이상 점수가 점선(임계값) 이하
- **ECG-A (3j)**: PVC(조기 심실 수축) 구간에서 파형이 변형. 해당 시점에서 이상 점수가 임계값을 초과
- **해석**: 준주기적(quasi-periodic) 시계열에 대한 적용 가능성 실증. 단 데이터셋에 이상 샘플이 1개뿐이고($N_a=1$), $v_A$ 검증셋 없이 휴리스틱 임계값 사용으로 통계적 신뢰도는 낮음 ⚠️

---

## 8. 결론: 시사점, 후속 연구, 추가 방향 제안

### 8-0. 저자들이 제시한 시사점 및 후속 연구 (p.4-5, Sec 5)

| 구분 | 내용 |
|------|------|
| **핵심 시사점** | 재구성 기반 이상 탐지가 예측 기반보다 범용적; 예측 불가 시계열에서 유효 |
| **실용적 시사점** | 이상 데이터 없이 정상 데이터만으로 훈련 가능; 정기 유지보수 환경에 적합 |
| **후속 연구 방향 (저자 언급)** | 다층 구조 탐색, 어텐션 메커니즘 도입, 더 다양한 도메인 검증 (직접 명시 없이 Discussion에서 암시) |

---

### 8-1. 모델의 일반화 성능 향상 가능성

#### 현재 일반화의 한계

```
① 단일 은닉층만 실험
② PCA로 다변량 → 단변량 축소 (정보 손실 39%)
③ 하이퍼파라미터(c, L) 수동 설정
④ 비공개 데이터셋 포함으로 외부 검증 불가
⑤ 소규모 데이터셋 (ECG: Na=1)
```

#### 일반화 향상을 위한 구체적 방향

**① 아키텍처 개선**
- **계층적(hierarchical) Encoder-Decoder**: 다중 시간 스케일의 패턴 포착
- **Bidirectional LSTM**: 순방향+역방향 컨텍스트 활용
- **Attention 메커니즘**: 어느 시점이 재구성에 중요한지 가중치 부여

$$\alpha^{(i)} = \frac{\exp(\mathbf{v}^T \tanh(\mathbf{W}_a \mathbf{h}_E^{(i)}))}{\sum_j \exp(\mathbf{v}^T \tanh(\mathbf{W}_a \mathbf{h}_E^{(j)}))}$$

**② 전처리 개선**
- PCA 대신 **Variational Autoencoder(VAE)** 기반 잠재 공간 학습으로 정보 손실 최소화
- 원본 다변량 시계열 직접 처리

**③ 훈련 강건성**
- **오염 강건 훈련(Robust Training)**: 정상 데이터에 소량 이상이 섞여도 견딜 수 있는 손실 함수
- **데이터 증강**: 정상 시퀀스에 noise 추가, time warping 등

**④ 임계값 자동화**
- 현재 수동 $\tau$ 설정을 **적응형 임계값(adaptive threshold)** 으로 대체
- 베이지안 최적화(Bayesian Optimization)로 $\tau$, $c$ 자동 탐색

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 아래 연구들은 제 학습 데이터(~2024년 초)를 기반으로 정리한 것으로, 일부 세부 수치는 원본 논문을 직접 확인하시기 바랍니다. 불확실한 내용에는 ⚠️ 표시를 했습니다.

#### 주요 후속 연구 비교

| 연구 | 방법 | EncDec-AD 대비 개선점 | 한계 |
|------|------|----------------------|------|
| **TadGAN** (Geiger et al., 2020) | GAN 기반 재구성 + Critic 점수 | 비지도 임계값 설정 자동화, 생성적 다양성 | GAN 훈련 불안정성 |
| **USAD** (Audibert et al., 2020) | Encoder + 두 개의 Decoder 적대적 훈련 | False Positive 감소, 속도 향상 | 다변량 상관관계 처리 제한 |
| **MTAD-GAT** (Zhao et al., 2020) | Graph Attention Network + GRU | 센서 간 공간적 상관관계 모델링 | 그래프 구조 사전 정의 필요 |
| **Anomaly Transformer** (Xu et al., 2021, NeurIPS) | Attention의 Association Discrepancy 이용 | Transformer의 전역 컨텍스트 포착 | 계산 비용 높음 |
| **TimesNet** (Wu et al., 2023, ICLR) | 1D→2D 변환으로 시간적 변동 모델링 | 다중 주기성 동시 처리 | 복잡한 전처리 |
| **PatchTST** (Nie et al., 2023, ICLR) | Transformer + Patch 기반 표현 | 장기 의존성 효율적 포착 | 이상 탐지 전용 설계 아님 |

#### EncDec-AD가 미친 영향

```
1. 재구성 기반 이상 탐지 패러다임 정립
   → VAE-LSTM, GAN 기반 방법들의 이론적 선구자

2. "정상 데이터만으로 훈련" 원칙 확립
   → 준지도/비지도 이상 탐지의 표준 프로토콜화

3. 다중 센서 처리 필요성 제기
   → 이후 MTAD-GAT 등 센서 간 상관관계 모델링 연구 촉발

4. 산업 IoT 도메인 적용 가능성 실증
   → 제조업, 항공, 의료 등 다양한 도메인 연구 확장
```

#### 앞으로 연구 시 고려할 점

| 고려사항 | 권장 방향 |
|---------|-----------|
| **벤치마크 표준화** | SMD, MSL, SMAP, PSM 등 표준 벤치마크 사용 (EncDec-AD는 비표준 데이터 혼재) |
| **평가 지표 일관성** | $F_\beta$ $\beta$ 값 통일, Point-Adjust 방식 명시, PR-AUC/ROC-AUC 병행 보고 |
| **다변량 상관관계** | 센서 간 시공간 상관관계를 명시적으로 모델링 (GNN, Graph Attention 등) |
| **계산 효율성** | 엣지 디바이스 배포를 위한 모델 경량화 (Knowledge Distillation, Quantization) |
| **적응형 학습** | 정상 패턴 드리프트에 대응하는 온라인/연속 학습 메커니즘 |
| **설명 가능성** | 어느 센서, 어느 시점에서 이상이 발생했는지 해석 가능한 출력 제공 |
| **Few-shot 이상 탐지** | 극소수 이상 샘플만으로도 임계값 최적화 가능한 메타러닝 접근 |

---

## 참고 자료 목록

1. **Malhotra, P. et al.** (2016). *LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection*. arXiv:1607.00148v2. ICML 2016 Anomaly Detection Workshop.
2. **Hochreiter, S. & Schmidhuber, J.** (1997). Long short-term memory. *Neural computation*, 9(8):1735–1780.
3. **Sutskever, I., Vinyals, O., & Le, Q.V.** (2014). Sequence to sequence learning with neural networks. *NIPS 27*.
4. **Cho, K. et al.** (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. arXiv:1406.1078.
5. **Malhotra, P. et al.** (2015). Long short term memory networks for anomaly detection in time series. *ESANN 2015*.
6. **Kingma, D.P. & Ba, J.** (2014). Adam: A method for stochastic optimization. arXiv:1412.6980.
7. **Keogh, E., Lin, J., & Fu, A.** (2005). HOT SAX. *IEEE ICDM 2005*.
8. **Sakurada, M. & Yairi, T.** (2014). Anomaly detection using autoencoders with nonlinear dimensionality reduction. *MLSDA 2014*.
9. **Audibert, J. et al.** (2020). USAD: UnSupervised Anomaly Detection on Multivariate Time Series. *KDD 2020*. ⚠️
10. **Xu, J. et al.** (2021). Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy. *NeurIPS 2021*. ⚠️
11. **Geiger, A. et al.** (2020). TadGAN: Time Series Anomaly Detection Using Generative Adversarial Networks. *IEEE BigData 2020*. ⚠️
12. **Zhao, H. et al.** (2020). Multivariate Time-series Anomaly Detection via Graph Attention Network. *ICDM 2020*. ⚠️
