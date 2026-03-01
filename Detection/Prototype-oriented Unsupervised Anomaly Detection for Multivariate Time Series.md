# Prototype-oriented Unsupervised Anomaly Detection for Multivariate Time Series

---

## 1. 핵심 주장과 주요 기여 요약

**핵심 주장:** 기존 비지도 이상 탐지(UAD) 방법들은 각 다변량 시계열(MTS)마다 개별 파라미터 세트를 학습하므로 계산 비용이 크고 새로운 MTS에 대한 적응력이 제한적이다. PUAD는 다수의 MTS에 걸쳐 공유되는 **프로토타입(prototype)** 그룹을 학습하고, 최적 수송(Optimal Transport)을 활용하여 다양한 정상 패턴을 대표하는 프로토타입을 효율적으로 인덱싱함으로써 이 문제를 해결한다.

**주요 기여:**
1. 다수 MTS의 다양한 정상 패턴을 프로토타입 그룹으로 추출하는 확률적 프레임워크(PUAD) 제안
2. 분포 간 OT 거리를 활용하여 프로토타입 학습을 안내하는 **Prototype-oriented Optimal Transport (POT)** 모듈 개발
3. **글로벌 프로토타입**(여러 MTS 공유 패턴)과 **로컬 프로토타입**(새로운 MTS 고유 패턴)을 정의하여 메타 이상 탐지에서의 적응 능력 강화
4. 5개 공개 MTS 데이터셋에서 전통적 이상 탐지 및 메타 이상 탐지 모두에서 SOTA 성능 달성

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

대규모 시스템(CDN, 데이터센터 서버 등)에서 생성되는 MTS 데이터에서 비지도 방식으로 이상점을 탐지하는 것이 목표이다. 구체적인 기술적 도전은 다음과 같다:

- **다양한 정상 패턴 모델링:** 각 디바이스(예: 동영상 서버 vs. 쇼핑 서버)가 서로 다른 정상 모드를 가지므로, 고정된 파라미터 세트 하나로 모든 패턴을 포착하기 어렵다.
- **새로운 MTS에 대한 적응:** 기존 방법은 새로운 MTS에 적응하기 위해 대량의 데이터로 파라미터를 재학습해야 하며, 관측 데이터가 적을 때 성능이 급격히 저하된다.
- **계산 효율성:** "one-for-one" 방식(각 MTS마다 개별 모델)은 MTS 수가 많을 때 파라미터 수가 폭발적으로 증가한다.

### 2.2 제안하는 방법 (수식 포함)

#### 2.2.1 문제 정의

MTS를 $\boldsymbol{x} = (x_1, x_2, \ldots, x_T) \in \mathbb{R}^{V \times T}$로 정의하며, $x_t \in \mathbb{R}^V$는 시점 $t$에서의 $V$-차원 관측값이다. 목표는 특정 시점 $x_t$가 이상(anomalous)인지 여부를 비지도 방식으로 판별하는 것이다.

#### 2.2.2 Prototype-oriented Optimal Transport (POT) 모듈

**프로토타입 정의:** 글로벌 프로토타입 $\boldsymbol{\beta}_g = [\boldsymbol{b}_g^1, \boldsymbol{b}_g^2, \ldots, \boldsymbol{b}_g^{K_g}] \in \mathbb{R}^{K_g \times d_f}$와 로컬 프로토타입 $\boldsymbol{\beta}_l = [\boldsymbol{b}_l^1, \boldsymbol{b}_l^2, \ldots, \boldsymbol{b}_l^{K_l}] \in \mathbb{R}^{K_l \times d_f}$를 결합하여 $\boldsymbol{\beta} = [\boldsymbol{\beta}_g; \boldsymbol{\beta}_l] \in \mathbb{R}^{(K_g+K_l) \times d_f}$로 정의한다.

**임베딩 분포:** $N_j$개의 샘플에 대한 경험적 분포:

$$P_{\theta_0} = \sum_{i=1}^{N_j} \frac{1}{N_j} \delta_{\theta_0^i}, \quad \theta_0^i \in \mathbb{R}^{d_f} $$

**프로토타입 분포:**

$$P_{\boldsymbol{\beta}_g} = \sum_{i=1}^{K_g} \frac{1}{K_g} \delta_{\boldsymbol{b}_g^i}, \quad \boldsymbol{b}_g^i \in \mathbb{R}^{d_f} $$

**OT를 통한 수송 확률 행렬 획득:** Sinkhorn 알고리즘을 사용하여 $P_{\theta_0}$에서 $P_{\boldsymbol{\beta}_g}$로의 최적 수송 계획을 구한다:

$$\boldsymbol{M}^* = \text{OT}(P_{\theta_0}, P_{\boldsymbol{\beta}_g}) = \min_{\boldsymbol{M}} \langle \boldsymbol{M}, \boldsymbol{C} \rangle \stackrel{\text{def.}}{=} \sum_i^{N_j} \sum_j^{K_g} M_{ij} C_{ij}$$

여기서 비용 행렬 $\boldsymbol{C} \in \mathbb{R}\_{\geq 0}^{N_j \times K_g}$이며, $C_{ij} = \sqrt{(\theta_0^i - \beta_g^j)^2}$ (유클리드 거리)이다.

**프로토타입 기반 잠재 표현 재구성:**

$$\boldsymbol{\theta}_0' = \boldsymbol{M} \times \boldsymbol{\beta}_g, \quad \boldsymbol{\theta}_0' \in \mathbb{R}^{N_j \times d_f} $$

**OT 손실 함수:**

$$\mathcal{L}_{OT} = \min_{\boldsymbol{\beta}_g} \mathbb{E}_{\theta \sim F_\phi(\mathcal{D}_x)} \left[ \sum_i^{N_j} \sum_j^{K_g} M_{ij} C_{ij} + \sum_i^{N_j} \sum_j^{K_g} M_{ij} \ln(M_{ij}) \right] = \min_{\boldsymbol{\beta}_g} \mathbb{E}_{\theta \sim F_\phi(\mathcal{D}_x)} \left[ \text{OT}(P_{\theta_0}, P_{\boldsymbol{\beta}_g}) \right] $$

#### 2.2.3 프로토타입 가이드 확률적 생성 모델

계층적 VAE에서 영감을 받아 다음과 같은 생성 과정을 정의한다:

$$\theta_0 \sim \mathcal{N}(0, \boldsymbol{I})$$

$$\theta_1 \sim \mathcal{N}\left(\mathcal{F}_1^\mu(\boldsymbol{\theta}_0'), \mathcal{F}_1^\sigma(\boldsymbol{\theta}_0')\right)$$

$$\boldsymbol{x}' \sim \mathcal{N}\left(\mathcal{F}_2^\mu(\theta_1), \mathcal{F}_2^\sigma(\theta_1)\right) $$

$$\boldsymbol{\theta}_0' = \boldsymbol{M} \times \boldsymbol{\beta}_g, \quad \boldsymbol{M} = \text{POT}(\theta_0, \boldsymbol{\beta}_g)$$

여기서 $\mathcal{F}_1^\mu, \mathcal{F}_1^\sigma, \mathcal{F}_2^\mu, \mathcal{F}_2^\sigma$는 완전 연결 네트워크로 구현된 비선형 함수이다.

#### 2.2.4 Transformer 기반 추론 모델

변분 분포를 다음과 같이 정의한다:

$$q(\theta_0, \theta_1 \mid \boldsymbol{x}, \boldsymbol{\theta}_0') = q(\theta_0 \mid \boldsymbol{x}) q(\theta_1 \mid \boldsymbol{\theta}_0', \boldsymbol{x}) $$

Transformer를 사용하여 시계열의 시간적 의존성을 인코딩한다:

$$\boldsymbol{h}_0 = \text{Transformer}(\boldsymbol{x}), \quad \boldsymbol{h}_1 = f(\boldsymbol{W}_{h_0 h_1} \boldsymbol{h}_0 + \boldsymbol{b}_{h_0 h_1}) $$

추론 과정:

$$q(\theta_0 \mid \boldsymbol{x}) \sim \mathcal{N}\left(\theta_0 \mid \tilde{\mu}_{\theta_0}, \text{diag}(\tilde{\sigma}_{\theta_0}^2)\right)$$

$$q(\theta_1 \mid \boldsymbol{\theta}_0', \boldsymbol{x}) \sim \mathcal{N}\left(\theta_1 \mid \tilde{\mu}_{\theta_1}, \text{diag}(\tilde{\sigma}_{\theta_1}^2)\right) $$

$$\tilde{\mu}_{\theta_1} = f(\tilde{\boldsymbol{W}}_{\theta_1}^\mu \boldsymbol{\theta}_0' + \tilde{\boldsymbol{V}}_{\theta_1}^\mu \boldsymbol{h}_1 + \tilde{\boldsymbol{b}}_{\theta_1}^\mu)$$

#### 2.2.5 학습 목적 함수

ELBO와 OT 손실을 결합한 최종 목적함수:

$$\mathcal{L} = \mathbb{E}_{q(\theta_0), q(\theta_1)}[\log p(\boldsymbol{x} \mid \theta_1)] - \rho_1 D_{KL}\left(q(\theta_1 \mid \boldsymbol{x}, \boldsymbol{\theta}_0') \| p(\theta_1 \mid \boldsymbol{\theta}_0')\right) - \rho_2 D_{KL}\left(q(\theta_0 \mid \boldsymbol{x}) \| p(\theta_0)\right) - \rho_3 \mathcal{L}_{OT} $$

여기서 $\rho_1, \rho_2, \rho_3 > 0$은 beta-VAE에서 영감을 받은 하이퍼파라미터로, 학습 초기 $K$ 에폭 동안 0에서 1로 점진적으로 증가시킨다.

#### 2.2.6 이상 점수

재구성 확률을 이상 점수로 사용:

$$S_t = \log p(\boldsymbol{x} \mid \theta_0, \theta_1) $$

$S_t$가 특정 임계값 이하이면 이상으로 분류하며, Peaks-Over-Threshold 방법으로 임계값을 선택한다.

### 2.3 모델 구조

PUAD의 전체 구조는 다음 세 모듈로 구성된다 (Figure 2 참조):

1. **인코더 (Encoder):** 3-layer Transformer (hidden dimension 512)로 MTS에서 시간적 의존성을 포착하여 $\boldsymbol{h}_0, \boldsymbol{h}_1$을 추출하고, 변분 분포 $q(\theta_0 \mid \boldsymbol{x})$, $q(\theta_1 \mid \boldsymbol{\theta}_0', \boldsymbol{x})$를 추론
2. **POT 모듈 (잠재 공간):** 잠재 표현 $\theta_0$와 프로토타입 $\boldsymbol{\beta}$ 사이의 OT 계산 → 수송 확률 행렬 $\boldsymbol{M}$ 획득 → 프로토타입 정보가 반영된 $\boldsymbol{\theta}_0' = \boldsymbol{M} \times \boldsymbol{\beta}_g$ 생성
3. **디코더 (Decoder):** 계층적 확률 생성 모델로 $\boldsymbol{\theta}_0' \rightarrow \theta_1 \rightarrow \boldsymbol{x}'$ 순서로 MTS를 재구성

**구현 세부사항:**
- 글로벌 프로토타입 $K_g = 10$, 로컬 프로토타입 $K_l = 2$, 프로토타입 차원 $d_f = 256$
- 잠재 변수 $\theta_0, \theta_1$ 차원: 512
- 슬라이딩 윈도우 크기: 20
- 학습률: 0.00002, 배치 크기: 256

### 2.4 성능 향상

**일반 이상 탐지 (Table 1):** 5개 데이터셋(SMD, MSL, PSM, SMAP, DND) 모두에서 SOTA F1-score 달성:

| 데이터셋 | PUAD | 차상위 방법 | 차상위 F1 |
|--------|------|-----------|---------|
| SMD | **96.16** | TranAD | 96.05 |
| MSL | **95.04** | TranAD | 94.94 |
| PSM | **98.14** | Anomaly Transformer | 97.89 |
| SMAP | **96.72** | Anomaly Transformer | 96.69 |
| DND | **86.62** | GmVRNN | 85.58 |

**메타 이상 탐지 (Table 2):** 제한된 관측(1~200 샘플)으로 새로운 MTS에 적응하는 시나리오에서 PUAD는 일관되게 최고 성능을 달성하였다. 예를 들어, SMD에서 1개 샘플만으로도 F1 95.68%를 달성한 반면, 프리트레인된 Anomaly Transformer는 91.58%에 그쳤다.

**파라미터 효율성 (Figure 3(b)):** "one-for-all" 방식인 PUAD는 "one-for-one" 방법 대비 파라미터 수가 현저히 적으면서도 더 높은 F1-score를 달성하였다.

**시간 효율성 (Table 3):** PUAD의 테스트 시간은 경쟁 모델 대비 최소 수준이다 (예: SMD에서 $3.30 \times 10^{-5}$ sec/sample vs. GmVRNN $7.02 \times 10^{-5}$).

**Ablation Study (Figure 3(a)):** 
- Transformer → Transformer+VAE → +POT → PUAD로 각 구성요소 추가 시 F1이 단조적으로 향상
- POT 모듈 추가 시 평균 7.11% 향상, OT 정규화 추가 시 평균 4.52% 추가 향상

### 2.5 한계

논문에서 명시적으로 언급된 한계는 제한적이나, 분석을 통해 다음 한계를 도출할 수 있다:

1. **프로토타입 수의 수동 설정:** $K_g$와 $K_l$을 하이퍼파라미터로 고정(10, 2)하며, 데이터 복잡도에 따른 자동 조정 메커니즘이 없다.
2. **변수 간 공간적 상관관계 모델링 부재:** Transformer가 시간축 의존성은 포착하지만, 변수 간(inter-variable) 그래프 구조 등 공간적 의존성을 명시적으로 모델링하지 않는다.
3. **스트리밍/온라인 학습 미지원:** 프로토타입과 모델 파라미터가 오프라인으로 학습되며, 개념 변화(concept drift)에 대한 동적 대응이 불분명하다.
4. **임계값 설정 의존성:** Peaks-Over-Threshold 방법의 초기 확률 $p$를 경험적으로 0.01로 설정하며, 이에 대한 민감도 분석이 부족하다.
5. **평가 프로토콜의 제약:** Point-adjust F1-score를 사용하는데, 이 메트릭이 실제 이상 탐지 성능을 과대평가할 수 있다는 비판이 최근 연구에서 제기되고 있다.

---

## 3. 모델의 일반화 성능 향상 가능성 (중점 분석)

### 3.1 일반화를 위한 핵심 설계

PUAD의 일반화 성능은 크게 세 가지 메커니즘에 의해 달성된다:

**(1) 글로벌 프로토타입을 통한 전이 가능한 패턴 학습**

글로벌 프로토타입 $\boldsymbol{\beta}_g$는 여러 MTS에 걸쳐 공유되는 통계적 시간 의존성을 포착하도록 학습된다. 이는 메타러닝에서의 "전이 가능한 패턴(transferable patterns)"과 유사한 역할을 수행하며, OT 손실(Eq. 5)을 통해 프로토타입이 다양한 정상 패턴의 대표점이 되도록 유도된다:

$$\mathcal{L}_{OT} = \min_{\boldsymbol{\beta}_g} \mathbb{E}_{\theta \sim F_\phi(\mathcal{D}_x)} \left[ \text{OT}(P_{\theta_0}, P_{\boldsymbol{\beta}_g}) \right]$$

이 최적화는 프로토타입이 전체 학습 데이터의 잠재 표현 분포를 대표하도록 만들어, 새로운 MTS에서도 유사한 정상 패턴을 인식할 수 있게 한다.

**(2) 로컬 프로토타입을 통한 빠른 적응**

새로운 MTS에 대해서는 소량의 샘플에서 로컬 프로토타입 $\boldsymbol{\beta}_l$만을 학습한다. 이때 글로벌 프로토타입과 기존 모델 파라미터는 고정되므로:
- 학습해야 할 파라미터가 $K_l \times d_f$개(논문에서 $2 \times 256 = 512$개)로 극히 적다
- 과적합 위험이 낮고, 수 개의 샘플로도 효과적인 적응이 가능하다
- 글로벌 프로토타입이 축적한 지식을 훼손하지 않는다

**(3) OT 기반 프로토타입 선택 메커니즘**

OT는 $\theta_0$에서 프로토타입으로의 최적 수송 계획을 제공하여, 현재 MTS와 가장 관련이 깊은 프로토타입에 높은 가중치를 자동으로 부여한다. 이 메커니즘은:
- 글로벌과 로컬 프로토타입 사이의 중요도를 자동으로 균형 잡는다
- 글로벌 프로토타입이 현재 MTS에 부적합하더라도 로컬 프로토타입이 보완하고, 반대의 경우도 마찬가지이다

### 3.2 일반화 성능의 실험적 검증

**메타 이상 탐지 실험 (Table 2):**

PUAD는 1개 샘플만으로도 일관된 성능을 보여준다:
- SMD: 1샘플 F1 95.68% → 200샘플 F1 95.51% (변동 폭 0.67%p 이내)
- MSL: 1샘플 F1 91.72% → 200샘플 F1 93.64%
- SMAP: 1샘플 F1 95.20% → 200샘플 F1 96.12%

이는 글로벌 프로토타입이 충분한 전이 정보를 이미 내포하고 있어, 소량의 로컬 적응만으로도 우수한 성능이 가능함을 입증한다.

반면, 프리트레인된 one-for-one 모델들(Table 8)은 동일 설정에서 PUAD보다 상당히 낮은 성능을 보였다 (예: Anomaly Transformer SMD 1샘플: 91.58% vs. PUAD 95.68%).

### 3.3 일반화 성능의 추가 향상 방향

1. **프로토타입 수 자동 결정:** 비모수적(nonparametric) 방법이나 Dirichlet Process를 활용하여 데이터 복잡도에 따라 프로토타입 수를 자동 조정
2. **계층적/다중 해상도 프로토타입:** 시간 스케일에 따라 다단계 프로토타입을 학습하여 다양한 시간 해상도의 패턴을 포착
3. **도메인 적응 기법 강화:** 도메인 불변 표현 학습(domain-invariant representation)과 결합하여 도메인 간 일반화 향상
4. **그래프 구조 통합:** 변수 간 의존성을 그래프 신경망으로 모델링하여 공간적 일반화 능력 강화
5. **컨티뉴얼 러닝(Continual Learning) 통합:** 새로운 MTS가 지속적으로 도착하는 시나리오에서 프로토타입을 점진적으로 업데이트하는 메커니즘

---

## 4. 향후 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미치는 영향

1. **"One-for-all" 패러다임의 확산:** PUAD는 개별 모델을 각 MTS마다 학습하는 "one-for-one" 접근법의 한계를 극복하는 실용적 대안을 제시하였다. 이는 대규모 시스템(수만 대 서버) 운영 시 배포 비용을 크게 절감할 수 있다.

2. **프로토타입 기반 시계열 이상 탐지의 선구적 연구:** MTS 이상 탐지에 프로토타입 개념을 최초로 도입하여, 향후 프로토타입/메모리 기반 시계열 연구의 이론적 기반을 마련하였다.

3. **메타 이상 탐지 벤치마크 정립:** 제한된 데이터로 새로운 MTS에 적응하는 "메타 이상 탐지" 실험 설정을 체계화하여, 향후 연구에서의 표준 평가 프레임워크를 제시하였다.

4. **OT와 확률적 생성 모델의 결합:** 최적 수송을 VAE 기반 생성 모델의 잠재 공간에서 활용하는 방법론은 이상 탐지 외에도 시계열 생성, 예측, 클러스터링 등 다양한 과제에 확장 가능하다.

### 4.2 향후 연구 시 고려할 점

1. **평가 메트릭 재검토:** 최근 Kim et al. (2022, "Towards a Rigorous Evaluation of Time-Series Anomaly Detection", AAAI 2022)에서 point-adjust F1-score의 문제점이 지적되었다. 향후 연구에서는 더 엄밀한 평가 프로토콜(예: range-based F1, VUS 등)을 채택해야 한다.

2. **개념 변화(Concept Drift) 대응:** 실제 운영 환경에서는 정상 패턴 자체가 시간에 따라 변화한다. 프로토타입을 동적으로 갱신하거나 불필요한 프로토타입을 제거하는 메커니즘이 필요하다.

3. **해석 가능성(Interpretability) 강화:** 수송 확률 행렬 $\boldsymbol{M}$이 어떤 프로토타입과 연관되는지를 분석하면 이상의 원인을 해석할 수 있으나, 이를 체계화하는 방법론이 더 발전해야 한다.

4. **다양한 이상 유형 대응:** 점 이상(point anomaly), 맥락 이상(contextual anomaly), 집합 이상(collective anomaly) 등 다양한 유형에 대한 프로토타입의 차별적 효과 분석이 필요하다.

5. **스케일링 문제:** 변수 차원($V$)이 수백~수천으로 증가할 때 프로토타입 차원과 OT 계산의 확장성에 대한 연구가 필요하다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 방법 | 연도 | 핵심 접근법 | PUAD와의 비교 |
|------|------|-----------|------------|
| **OmniAnomaly** (Su et al., 2019) | 2019 | 확률적 RNN + 정규화 플로우 | 시간적 의존성 모델링에 집중하나 다양한 MTS 패턴 통합 모델링 불가. PUAD가 SMD에서 F1 96.16% vs. 85.22%로 크게 상회 |
| **THOC** (Shen et al., 2020) | 2020 | 계층적 일-클래스 네트워크 | 시간 계층 구조를 활용하나 one-for-one 방식이므로 새로운 MTS 적응에 약함. PUAD가 전 데이터셋에서 우수 |
| **Anomaly Transformer** (Xu et al., 2021) | 2021 | Association discrepancy + Transformer | Attention 기반 association을 이상 점수로 활용. PSM에서 97.89%로 강력하나 PUAD(98.14%)가 상회. 메타 이상 탐지에서 PUAD가 현저히 우수 |
| **InterFusion** (Li et al., 2021) | 2021 | 계층적 VAE + inter-metric/temporal 임베딩 | 변수 간, 시간 간 의존성을 계층적으로 모델링하나 one-for-one 방식. PUAD가 전반적으로 우수 |
| **TranAD** (Tuli et al., 2022) | 2022 | 적대적 학습 + Transformer | SMD에서 96.05%로 강력하나 PUAD(96.16%)가 상회. SMAP에서는 89.15%로 PUAD(96.72%)에 크게 뒤처짐 |
| **GmVRNN** (Dai et al., 2022) | 2022 | 가우시안 혼합 변분 RNN | 동일한 one-for-all 방식이나 단일 파라미터 세트의 한계. PUAD가 프로토타입을 통해 다양성 포착에서 우위 |
| **VGCRN** (Chen et al., 2022) | 2022 | 변분 그래프 합성곱 순환 네트워크 | 변수 간 그래프 구조를 명시적으로 모델링. PUAD는 그래프 구조를 활용하지 않으나 프로토타입으로 보완. 두 접근법의 결합이 유망 |
| **TSMAE** (Gao et al., 2022) | 2022 | 메모리 증강 오토인코더 | 메모리 벡터로 정상 패턴을 저장하나, 코사인 유사도 기반 메모리 접근은 OT 기반 접근보다 프로토타입 균형 조절이 어려움. PUAD가 전 데이터셋에서 우수 |

### 최근 연구 트렌드와의 관계

- **Foundation Model 기반 이상 탐지** (2023~): GPT-style 또는 사전학습된 시계열 기반 모델(예: TimeGPT, Lag-Llama)이 등장하고 있으며, PUAD의 프로토타입 개념은 이러한 foundation model의 in-context learning이나 prompt tuning과 결합될 가능성이 있다.
- **Self-supervised/Contrastive Learning 기반 이상 탐지** (2021~): TS2Vec (Yue et al., 2022), DCdetector (Yang et al., 2023) 등 대조 학습 기반 방법이 부상하고 있으며, 프로토타입을 대조 학습의 앵커로 활용하는 확장 연구가 가능하다.
- **GNN 기반 MTS 이상 탐지** (2022~): VGCRN (Chen et al., 2022), GDN (Deng & Hooi, 2021) 등 그래프 구조를 활용하는 연구가 활발하며, PUAD에 변수 간 그래프 구조를 통합하면 일반화 성능을 추가로 개선할 수 있다.

---

## 참고자료

1. **Li, Y., Chen, W., Chen, B., Wang, D., Tian, L., & Zhou, M.** (2023). "Prototype-oriented unsupervised anomaly detection for multivariate time series." *Proceedings of the 40th International Conference on Machine Learning (ICML 2023)*, PMLR 202. — 본 논문 원문
2. **Xu, J., Wu, H., Wang, J., & Long, M.** (2021). "Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy." *arXiv:2110.02642*
3. **Tuli, S., Casale, G., & Jennings, N. R.** (2022). "TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data." *arXiv:2201.07284*
4. **Dai, L., Chen, W., et al.** (2022). "Switching Gaussian Mixture Variational RNN for Anomaly Detection of Diverse CDN Websites." *IEEE INFOCOM 2022*
5. **Chen, W., Tian, L., Chen, B., et al.** (2022). "Deep Variational Graph Convolutional Recurrent Network for Multivariate Time Series Anomaly Detection." *ICML 2022*
6. **Su, Y., Zhao, Y., et al.** (2019). "Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network." *ACM SIGKDD 2019*
7. **Tanwisuth, K., Fan, X., et al.** (2021). "A Prototype-oriented Framework for Unsupervised Domain Adaptation." *NeurIPS 2021*
8. **Guo, D., Tian, L., et al.** (2022). "Learning Prototype-oriented Set Representations for Meta-learning." *ICLR 2022*
9. **Peyré, G., Cuturi, M., et al.** (2019). "Computational Optimal Transport: With Applications to Data Science." *Foundations and Trends in Machine Learning*
10. **Shen, L., Li, Z., & Kwok, J.** (2020). "Timeseries Anomaly Detection Using Temporal Hierarchical One-Class Network." *NeurIPS 2020*
11. **Gao, H., et al.** (2022). "TSMAE: A Novel Anomaly Detection Approach for IoT Time Series Data Using Memory-augmented Autoencoder." *IEEE Trans. Network Science and Engineering*
12. **Gong, D., et al.** (2019). "Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder for Unsupervised Anomaly Detection." *ICCV 2019*
13. PUAD 공식 GitHub 리포지토리: https://github.com/BoChenGroup/PUAD
