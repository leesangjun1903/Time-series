# CrossAD: Time Series Anomaly Detection with Cross-scale Associations and Cross-window Modeling

> **참고 자료**: Beibu Li, Qichao Shentu et al., "CrossAD: Time Series Anomaly Detection with Cross-scale Associations and Cross-window Modeling," arXiv:2510.12489v1, NeurIPS 2025. (제공된 PDF 원문 직접 인용)

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

CrossAD는 시계열 이상 탐지에서 기존 멀티스케일 방법들이 간과하던 **두 가지 핵심 문제**를 동시에 해결하는 새로운 프레임워크다.

| 기존 문제 | CrossAD의 해결책 |
|---|---|
| 스케일 간 연관성(cross-scale association)을 무시 | Cross-scale Reconstruction |
| 고정 슬라이딩 윈도우로 인한 전역 정보 손실 | Cross-window Modeling (Query Library + Global Context) |

### 주요 기여 (3가지)

1. **Cross-scale Reconstruction**: 거친 스케일(coarse-grained)에서 세밀한 스케일(fine-grained) 시계열을 재구성함으로써 스케일 간 연관성을 명시적으로 모델링
2. **Cross-window Modeling**: 쿼리 라이브러리(Query Library)와 전역 멀티스케일 컨텍스트(Global Multi-scale Context)를 통해 슬라이딩 윈도우의 경계를 초월한 정보 공유
3. **State-of-the-art 성능**: 7개 실세계 데이터셋, 9개 평가 지표에서 18개 베이스라인 대비 최고 성능 달성

---

## 2. 해결하고자 하는 문제, 제안 방법(수식 포함), 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

**Problem 1: 스케일 간 연관성 무시**

기존 멀티스케일 방법들(TimesNet, TimeMixer 등)은 각 스케일을 독립적으로 모델링하거나 단순 피처 퓨전 전략을 사용한다. 그러나 정상 시계열에서는 거친 스케일이 세밀한 스케일을 특정 방식으로 복원할 수 있는 반면, **이상(anomaly) 발생 시 이 cross-scale association이 깨진다**는 점을 활용하지 못했다.

**Problem 2: 고정 윈도우 크기의 한계**

슬라이딩 윈도우 기반 접근법은 현재 윈도우 내부의 정보만 활용 가능하므로, 시계열 전체의 전역적 맥락(global context)을 파악하기 어렵다.

---

### 2.2 제안 방법 및 수식

#### 문제 정의

다변량 시계열 $\mathcal{X} = (x_1, x_2, \ldots, x_T) \in \mathbb{R}^{T \times C}$가 주어졌을 때, 테스트 시계열 $\mathcal{X}\_{test}$의 각 시점 $x_t$에 대해 이상 여부 $\hat{\mathcal{Y}}\_{test} = (y_1, y_2, \ldots, y_{T'})$, $y_t \in \{0,1\}$를 출력하는 것이 목표.

---

#### Stage 1: Multi-scale Generation and Embedding

원본 시계열 $\mathbf{X}\_m$에 $m$가지 평균 풀링(Average Pooling)을 적용하여 $m$개의 멀티스케일 시계열 $\{\mathbf{X}\_0, \mathbf{X}\_1, \ldots, \mathbf{X}\_{m-1}\}$을 생성한다. 각 $\mathbf{X}\_i \in \mathbb{R}^{T_i}$이고 $T_i < T_{i+1}$ (거친 순서). 각 스케일에 패치 임베딩(Patch Embedding)을 적용하여 $\mathbf{h}^0_i \in \mathbb{R}^{P_i \times d}$를 얻고, 이를 결합:

$$\mathbf{H}^0 = \{\mathbf{h}^0_0, \ldots, \mathbf{h}^0_{m-1}\} \in \mathbb{R}^{P \times d}, \quad P = \sum_{i=0}^{m-1} P_i$$

---

#### Stage 2: Scale-Independent Encoding

마스크 어텐션 메커니즘(Mask Attention)을 정의:

$$\text{MaskAttn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}, \mathbf{M}) = \text{softmax}\!\left(\frac{(\mathbf{Q}\mathbf{W}_q)(\mathbf{K}\mathbf{W}_k)^\top}{\sqrt{d}} + \mathbf{M}\right)(\mathbf{V}\mathbf{W}_v) $$

**Scale-Independent Mask** $\mathbf{M}\_1 \in \mathbb{R}^{P \times P}$: 각 스케일 내부 블록 $\mathbf{B}\_i = \mathbf{0}_{P_i \times P_i}$ (어텐션 허용), 블록 외부는 $-\infty$ (어텐션 차단). 이를 통해 스케일 간 간섭 없이 각 스케일의 시간 의존성을 독립적으로 학습:

$$\hat{\mathbf{H}}^l = \text{LayerNorm}(\mathbf{H}^{l-1} + \text{MaskAttn}(\mathbf{H}^{l-1}, \mathbf{H}^{l-1}, \mathbf{H}^{l-1}, \mathbf{M}_1))$$

$$\mathbf{H}^l = \text{LayerNorm}(\hat{\mathbf{H}}^l + \text{Feedforward}(\hat{\mathbf{H}}^l)) $$

---

#### Stage 3: Next-scale Generation (Cross-scale Reconstruction)

인코더 출력 $\mathbf{h}^L_i \in \mathbb{R}^{P_i \times d}$를 보간(Interpolation)으로 다음 스케일 차원에 맞게 조정:

$$\mathbf{z}^0_i = \text{interpolate}_i(\mathbf{h}^L_i), \quad i = 0, \ldots, m-1 $$

**Cross-scale Mask** $\mathbf{M}\_2$: 대각 블록 $\mathbf{B}\_i = \mathbf{0}\_{P_{i+1} \times P_{i+1}}$, 하삼각(lower triangular) 요소 = 0 → 각 스케일이 자신보다 거친 스케일 정보만 참조 가능.

디코더 과정:

$$\tilde{\mathbf{Z}}^l = \text{LayerNorm}(\mathbf{Z}^{l-1} + \text{MaskAttn}(\mathbf{Z}^{l-1}, \mathbf{Z}^{l-1}, \mathbf{Z}^{l-1}, \mathbf{M}_2)) $$

$$\hat{\mathbf{Z}}^l = \text{LayerNorm}(\tilde{\mathbf{Z}}^l + \text{Attention}(\tilde{\mathbf{Z}}^l, \mathbf{G}', \mathbf{G}'))$$

$$\mathbf{Z}^l = \text{LayerNorm}(\hat{\mathbf{Z}}^l + \text{FeedForward}(\hat{\mathbf{Z}}^l)) $$

여기서 $\mathbf{G}'$는 전역 멀티스케일 컨텍스트(Cross-window Modeling에서 생성).

**최적화 목적 함수** (Cross-scale Reconstruction):

$$\theta^* = \arg\min_\theta \sum_{i=1}^{m} \left\|\mathbf{X}_i - \hat{\mathbf{X}}_i\right\|_2^2, \quad \hat{\mathbf{X}}_i = f(\{\mathbf{X}_0, \ldots, \mathbf{X}_{i-1}\};\, \theta) $$

---

#### Cross-window Modeling

**Sub-series Representation (쿼리 라이브러리)**

쿼리 라이브러리 $\mathbf{Q} = \{q_1, \ldots, q_n\}$, $q_i \in \mathbb{R}^{S \times d}$를 구성. Period-aware Router가 입력 시계열을 푸리에 변환:

$$\mathbf{X}^t_{\text{period}} = \mathcal{F}^{-1}(\mathbf{A}^t_{\text{top-k}}, \Phi) $$

Gumbel-Softmax 게이팅으로 현재 윈도우에 적합한 쿼리를 동적으로 선택:

$$\mathbf{q}^t = \sum_{i=1}^n \frac{\exp((\mathbf{u}_i + g_i)/\tau)}{\sum_{j=1}^n \exp((\mathbf{u}_j + g_j)/\tau)} q_i, \quad \mathbf{u} = \text{MLP}(\mathbf{X}_{\text{period}}) $$

Cross-scale 서브시리즈 표현:

$$\mathbf{R}^t = \text{Cross-Attention}(\mathbf{q}^t, \mathbf{H}^{L,t}, \mathbf{H}^{L,t}) $$

**Global Multi-scale Context (전역 컨텍스트)**

$K$개의 프로토타입 $\mathbf{G} = \{g_1, \ldots, g_K\}$, $g_i \in \mathbb{R}^{S \times d}$를 구성하고 EMA(지수 이동 평균)로 업데이트:

$$d^t_i = \text{distance}(\mathbf{R}^t, g_i), \quad i = 1, \ldots, K $$

$$g_i \leftarrow \text{EMA}(g_i, \mathbf{R}), \quad i = \arg\min[d^t_1, \ldots, d^t_K] $$

($\alpha = 0.95$: EMA 감쇠율)

---

#### Anomaly Score

모든 스케일의 재구성 오차를 보간(interpolation)으로 원본 길이에 맞추고 평균화:

$$\text{AnomalyScore}(\mathbf{X}_m) = \frac{1}{m}\sum_{i=1}^m \text{interpolate}\!\left((\mathbf{X}_i - f(\{\mathbf{X}_0, \ldots, \mathbf{X}_{i-1}\};\,\theta))^2\right) $$

최종 임계값은 SPOT(Extreme Value Theory 기반 자동 임계값 결정)을 사용.

---

### 2.3 모델 구조 요약

```
입력 X_m
    │
    ├─① Multi-scale Generation (Average Pooling × m)
    │   → {X_0, X_1, ..., X_{m-1}}
    │
    ├─② Patch Embedding + Scale-Independent Encoding (Encoder, M_1)
    │   → H^L = {h^L_0, ..., h^L_{m-1}}
    │
    ├─③ [Cross-window Modeling]
    │   Period-aware Router → Gumbel-Softmax Query 선택
    │   Cross-Attention(q^t, H^{L,t}) → R^t
    │   EMA Update → Global Multi-scale Context G'
    │
    ├─④ Up-interpolation + Cross-scale Decoding (Decoder, M_2 + Attention(G'))
    │   → {X̂_1, X̂_2, ..., X̂_m}
    │
    └─⑤ Anomaly Score Interpolation
        → 최종 이상 점수 + SPOT 임계값 결정
```

---

### 2.4 성능 향상

#### 주요 실험 결과 (Table 1, VUS-PR 기준)

| Dataset | TimesNet (2위권) | CrossAD | 향상 |
|---|---|---|---|
| SMD | 0.2040 | **0.2344** | +14.9% |
| MSL | 0.2731 | **0.3144** | +15.1% |
| SMAP | 0.1352 | **0.1443** | +6.7% |
| SWaT | 0.1158 | **0.4767** | +311% |
| PSM | 0.4373 | **0.5596** | +28.0% |
| GECCO | 0.4578 | **0.6211** | +35.7% |
| SWAN | 0.9160 | **0.9171** | +0.1% |

#### 어블레이션 스터디 결과 (Table 3, 평균 VUS-PR)

| 구성 | 평균 VUS-PR | 향상 |
|---|---|---|
| Row 1: 기본 단일스케일 | 0.3665 | - |
| Row 2: +멀티스케일 | 0.3962 | +8.13% |
| Row 3: +크로스스케일 재구성 | 0.4326 | +9.18% |
| Row 4: +전역 컨텍스트(직접) | 0.4597 | +6.27% |
| Row 6: CrossAD (전체) | **0.4984** | +8.42% |

#### 효율성 비교 (Table 4, GECCO 데이터셋)

| 방법 | 추론 시간(s) | 파라미터(M) | VUS-PR |
|---|---|---|---|
| ModernTCN | **0.57** | **0.05** | 0.4819 |
| AnomalyTransformer | 15.05 | 4.74 | 0.0278 |
| CrossAD | 0.80 | 0.93 | **0.6211** |

CrossAD는 AnomalyTransformer 대비 추론 속도 약 **18.8배 빠르면서** VUS-PR은 22.4배 높다.

---

### 2.5 한계 (논문 Section G 직접 인용)

1. **하이퍼파라미터 자동화 부재**: 서브시리즈 쿼리 수($n$)와 프로토타입 수($K$)의 최적값이 데이터셋마다 다르지만, 현재는 고정된 통합 파라미터를 사용함. 데이터셋 특성에 맞는 자동 조정 메커니즘이 필요.
2. **훈련 시간 비용**: 전역 멀티스케일 컨텍스트의 지속적 업데이트로 인해 훈련 시간이 증가 (약 270초, Table 4).
3. **온라인 탐지 한계**: EMA 기반 전역 컨텍스트 업데이트 구조가 배치 학습 방식에 최적화되어 있어, 실시간 온라인 탐지 환경에서의 적용 가능성 검토가 필요.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화를 높이는 설계 요소들

#### (1) 전역 멀티스케일 컨텍스트 (Global Multi-scale Context)

가장 중요한 일반화 메커니즘이다. 학습 중 모든 윈도우를 순회하며 프로토타입 $\{g_1, \ldots, g_K\}$를 EMA로 업데이트하는 방식은, **다양한 서브시리즈 패턴을 점진적으로 누적**하는 효과를 낸다.

$$g_i \leftarrow \alpha g_i + (1-\alpha)\mathbf{R}^t \quad (\alpha = 0.95)$$

이 메커니즘 덕분에 모델은 단일 윈도우의 지역적 패턴에 과적합(overfit)되지 않고, 시계열 전체에 걸친 글로벌 패턴을 학습한다. 어블레이션에서 이 컴포넌트만으로 **VUS-ROC +3.18%, VUS-PR +10.4%** 향상이 확인됨.

#### (2) 채널 독립(Channel-Independent) 학습

논문이 채택한 채널 독립 방식은 각 채널을 독립적으로 처리함으로써:
- 채널 수가 다른 데이터셋에 유연하게 적용 가능
- 특정 채널 간 상관관계에 과적합되는 현상 방지
- 다양한 도메인(서버, 우주, 수처리 등)으로의 전이 용이

#### (3) 학습 가능한 서브시리즈 쿼리 + Period-aware Router

t-SNE 시각화(Figure 7)에서 확인할 수 있듯, 학습 가능한 서브시리즈 쿼리는 **랜덤 쿼리 대비 뚜렷한 클러스터링 구조**를 형성한다. 이는 모델이 시계열 데이터의 내재적 구조를 학습했음을 의미하며, 유사한 패턴을 가진 새로운 데이터에 대한 일반화 가능성을 높인다.

푸리에 변환 기반의 Period-aware Router는 데이터의 주기적 특성을 자동으로 감지하여 적절한 쿼리를 선택하므로, 다양한 주기성을 가진 데이터에 적응적으로 동작한다.

#### (4) 파라미터 민감도 (Figure 6)

논문은 패치 크기, 윈도우 크기, 히든 차원($d_\text{model}$)에 대한 민감도 분석을 수행했으며, CrossAD가 이러한 하이퍼파라미터에 **상대적으로 둔감(robust)**함을 보였다. (단, 패치 크기 = 1일 때는 성능 저하 발생 — 단일 포인트는 패턴 정보 부재)

#### (5) 스케일 수($m$)에 따른 안정적 성능 (Table 6)

스케일 수를 $m=1$에서 $m=5$로 증가시킬수록 성능이 향상되지만, 특정 수 이상에서 **성능이 수렴(stable)**하는 경향이 나타났다. 이는 모델이 과도한 스케일 추가에도 불안정해지지 않음을 의미한다.

#### (6) TSB-AD 및 UCR 벤치마크에서의 일반화 확인

- **TSB-AD-U** (40개 다양한 공개 데이터셋): VUS-PR = 0.45로 1위 (Sub-PCA 0.42, SubShapeAD 0.40 대비 우월)
- **UCR 벤치마크** (250개 서브데이터셋): 평균 $\alpha$ quantile = **4.40%** (Timer 12.5%, DADA 7.60% 대비 우월)

이는 CrossAD가 특정 도메인에 특화되지 않고 **광범위한 데이터셋에서 일관된 성능**을 보임을 입증한다.

### 3.2 일반화 향상을 위한 미래 가능성

| 방향 | 설명 |
|---|---|
| 자동 하이퍼파라미터 조정 | 데이터셋 특성에 맞는 $n$, $K$ 자동 탐색 (논문 한계로 명시) |
| 파운데이션 모델 통합 | GPT4TS 등 사전학습 LM과 결합하여 제로샷 일반화 가능성 |
| 온라인 학습 확장 | 스트리밍 데이터에서 프로토타입 업데이트 전략 개선 |
| 도메인 적응 | 소수의 타깃 도메인 데이터로 파인튜닝하는 퓨샷(few-shot) 학습 |

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4.1 카테고리별 비교

#### 재구성 기반 (Reconstruction-based)

| 논문 | 연도 | 핵심 아이디어 | CrossAD와의 차이 |
|---|---|---|---|
| **OmniAnomaly** (Su et al., KDD 2019) | 2019 | 확률적 RNN + Planar Normalizing Flow | 단일 스케일, 순환 구조 → 장기 의존성 약함 |
| **CAE-Ensemble** (Campos et al., PVLDB 2022) | 2022 | 다양성 기반 합성곱 앙상블 | 단일 스케일, 고정 윈도우 |
| **MEMTO** (Song et al., NeurIPS 2024) | 2024 | 메모리 가이드 Transformer | 단일 스케일, 전역 컨텍스트 없음 |
| **DADA** (Shentu et al., ICLR 2025) | 2025 | 적응적 병목 + 이중 적대적 디코더 | 멀티스케일 미지원, CrossAD 팀 동일 저자 |
| **CrossAD** | 2025 | Cross-scale Reconstruction + Cross-window | 스케일 간 연관성 + 전역 컨텍스트 명시적 모델링 |

#### 대조 기반 (Contrastive-based)

| 논문 | 연도 | 핵심 아이디어 | CrossAD와의 차이 |
|---|---|---|---|
| **Anomaly Transformer** (Xu et al., ICLR 2022) | 2022 | Prior-Series Association Discrepancy | 단일 스케일, 추론 느림(15.05s) |
| **DCdetector** (Yang et al., KDD 2023) | 2023 | 이중 어텐션 대조 표현 학습 | 단일 스케일, VUS-PR 낮음 |

#### 멀티스케일 기반 (Multi-scale)

| 논문 | 연도 | 핵심 아이디어 | CrossAD와의 차이 |
|---|---|---|---|
| **TimesNet** (Wu et al., arXiv 2022) | 2022 | 시계열 → 2D 텐서 변환, 주기 기반 | 스케일 독립 처리, 교차 스케일 연관성 미지원 |
| **Pyraformer** (Liu et al., ICLR 2022) | 2022 | 피라미드 어텐션으로 멀티스케일 | 이상탐지 전용 아님 |
| **Pathformer** (Chen et al., ICLR 2024) | 2024 | 계절성/트렌드 기반 적응적 패치 | 단순 피처 퓨전, 스케일 간 재구성 없음 |
| **TimeMixer** (Wang et al., 2024) | 2024 | 분해 가능한 멀티스케일 믹싱 | 스케일 혼합이지만 교차 스케일 연관성 없음 |
| **MtsCID** (Xie et al., WWW 2025) | 2025 | 거친/미세 스케일 내부 및 변수 간 의존성 포착 | 변수 내부 초점, 전역 컨텍스트 부재 |
| **MAD-TS** (Lu et al., ACML 2022) | 2022 | 어텐션 기반 순환 오토인코더 멀티스케일 | 레이어 기반 스케일, 명시적 교차 스케일 연관성 없음 |
| **MODEM** (Zhong et al., ICLR) | - | 확산 모델 + 주파수 분해 멀티스케일 | 비정상성 대응에 집중, 교차 스케일 재구성 없음 |
| **CrossAD** | 2025 | 크로스스케일 재구성 + 크로스윈도우 | **스케일 간 연관성 명시적 모델링 최초** |

#### 예측 기반 (Forecasting-based)

| 논문 | 연도 | 핵심 아이디어 | CrossAD와의 차이 |
|---|---|---|---|
| **GDN** (Deng & Hooi, AAAI 2021) | 2021 | 그래프 신경망 기반 예측 오차 | 그래프 구조 필요, 재구성 기반 아님 |

#### 사전학습/파운데이션 모델 기반

| 논문 | 연도 | 핵심 아이디어 | CrossAD와의 차이 |
|---|---|---|---|
| **GPT4TS** (Zhou et al., NeurIPS 2023) | 2023 | 사전학습 LM을 시계열에 적용 | 대용량 파라미터, GECCO에서 우수하나 SWaT에서 열세 |
| **Timer** (Liu et al., 2024) | 2024 | 대규모 Transformer 시계열 분석 | 파운데이션 모델, UCR에서 CrossAD에 열세 |

### 4.2 핵심 차별화 요소 정리

```
기존 멀티스케일 방법:
  스케일 1 ──── 독립 처리 ──── 이상 점수 1
  스케일 2 ──── 독립 처리 ──── 이상 점수 2  →  단순 합산
  스케일 3 ──── 독립 처리 ──── 이상 점수 3

CrossAD:
  스케일 1 (거친)
      │  Cross-scale Association
      ▼
  스케일 2 ──── 재구성 오차 ──── 이상 점수 2
      │  Cross-scale Association
      ▼
  스케일 3 (세밀) ──── 재구성 오차 ──── 이상 점수 3
      │
  +전역 컨텍스트(Cross-window)
      │
  최종 이상 점수 (보간 평균)
```

---

## 5. 앞으로의 연구에 미치는 영향 및 고려 사항

### 5.1 앞으로의 연구에 미치는 영향

#### (1) 패러다임 전환: 연관성 기반 이상 탐지

CrossAD는 기존의 "단일 스케일 재구성 오차 최소화" 패러다임에서 벗어나, **스케일 간 연관성의 붕괴(disruption)를 이상의 신호로 해석**하는 새로운 패러다임을 제시한다. 이는 이후 연구들이 다양한 차원의 "연관성 붕괴"를 탐지 신호로 활용하는 방향으로 발전할 수 있음을 시사한다.

#### (2) 전역 컨텍스트 프로토타입 학습의 가능성

EMA 기반 프로토타입 업데이트 메커니즘은 **온라인 연속 학습(continual learning)**과 결합하면, 시계열의 분포 변화(distribution shift)에 적응하는 모델 개발로 이어질 수 있다.

#### (3) 크로스윈도우 개념의 확장

쿼리 라이브러리 기반의 크로스윈도우 정보 공유는 시계열 예측, 분류, 이상 탐지 전반에 적용 가능한 **범용 멀티윈도우 컨텍스트 모듈**로 발전할 수 있다.

#### (4) 평가 지표 다양화의 필요성

CrossAD가 VUS-PR/VUS-ROC를 중심 지표로 채택한 것은, 기존 점 조정(point adjustment) F1 스코어의 한계를 극복하려는 시도다. 이는 향후 연구에서 **더 공정하고 견고한 평가 지표 체계** 수립을 촉진할 것으로 보인다.

### 5.2 앞으로 연구 시 고려할 점

#### 방법론적 고려 사항

| 고려 사항 | 세부 내용 |
|---|---|
| **자동 스케일 수 결정** | 데이터 특성(샘플링 주파수, 이상 지속 시간)에 따른 최적 스케일 수 $m$ 자동 결정 알고리즘 필요 |
| **하이퍼파라미터 자동화** | $n$(쿼리 수), $K$(프로토타입 수) 등을 데이터 기반으로 자동 탐색하는 메타러닝 또는 NAS 방법 적용 |
| **비대칭 이상 탐지** | 현재 이상 비율(AR)이 낮은 데이터셋(GECCO: 1.25%, UCR: 0.6%)에서의 성능 개선을 위한 클래스 불균형 처리 |
| **온라인/스트리밍 적용** | EMA 업데이트가 배치 학습에 최적화되어 있으므로, 실시간 스트리밍 환경에서 프로토타입 갱신 전략 재설계 필요 |
| **다변량 채널 간 관계** | 현재 채널 독립(channel-independent) 방식을 채택하므로, 채널 간 상관관계 활용 시 추가 성능 향상 가능 |

#### 평가 및 실험 설계 고려 사항

| 고려 사항 | 세부 내용 |
|---|---|
| **공정한 지표 선택** | VUS-PR, VUS-ROC, Affiliation-F1 등 편향 없는 지표 병행 사용 필수 (기존 PA-F1의 과대평가 문제 인식) |
| **도메인 이전(Transfer)** | 특정 도메인 학습 후 다른 도메인에서 제로샷/퓨샷 적용 실험 필요 |
| **이상 유형별 분석** | Point, Contextual, Shapelet, Seasonal, Trend 등 유형별 성능 분리 평가 (CrossAD는 Figure 4에서 일부 수행) |
| **계산 자원 제약 환경** | 엣지 디바이스나 제한된 메모리 환경에서의 경량화 버전 연구 |
| **벤치마크 다양성** | TSB-AD, UCR 외 금융, 의료(ECG), 산업 IoT 등 도메인 특화 벤치마크에서 추가 검증 필요 |

#### 이론적 고려 사항

| 고려 사항 | 세부 내용 |
|---|---|
| **Cross-scale Association 이론화** | 정상 시계열의 스케일 간 연관성이 얼마나 일관적인지에 대한 이론적 근거 부족 → 통계적 특성 분석 필요 |
| **프로토타입 수렴 보장** | EMA 기반 업데이트가 전역 최적해로 수렴하는지에 대한 이론적 분석 필요 |
| **이상 탐지 임계값 민감도** | SPOT 자동 임계값 결정의 이상 비율 의존성 분석 및 대안 방법 탐색 |

---

## 참고 자료

1. **Beibu Li, Qichao Shentu et al.** (2025). "CrossAD: Time Series Anomaly Detection with Cross-scale Associations and Cross-window Modeling." arXiv:2510.12489v1. NeurIPS 2025. *(본 분석의 주요 원전)*

2. **Xu et al.** (2022). "Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy." ICLR 2022.

3. **Yang et al.** (2023). "DCdetector: Dual Attention Contrastive Representation Learning for Time Series Anomaly Detection." KDD 2023.

4. **Wu et al.** (2022). "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis." arXiv:2210.02186.

5. **Wang et al.** (2024). "TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting." arXiv:2405.14616.

6. **Chen et al.** (2024). "Pathformer: Multi-scale Transformers with Adaptive Pathways for Time Series Forecasting." ICLR 2024.

7. **Shentu et al.** (2025). "Towards a General Time Series Anomaly Detector with Adaptive Bottlenecks and Dual Adversarial Decoders." ICLR 2025.

8. **Liu & Paparrizos** (2024). "The Elephant in the Room: Towards a Reliable Time-Series Anomaly Detection Benchmark." NeurIPS 2024.

9. **Paparrizos et al.** (2022). "Volume under the Surface: A New Accuracy Evaluation Measure for Time-Series Anomaly Detection." VLDB 2022.

10. **Su et al.** (2019). "Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network." KDD 2019.

11. **Wu & Keogh** (2023). "Current Time Series Anomaly Detection Benchmarks are Flawed and are Creating the Illusion of Progress." TKDE 2023.

12. **Siffer et al.** (2017). "Anomaly Detection in Streams with Extreme Value Theory." KDD 2017. (SPOT 방법)
