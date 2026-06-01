
# PDFormer: Propagation Delay-Aware Dynamic Long-Range Transformer for Traffic Flow Prediction

> **논문 정보:**
> - **저자:** Jiawei Jiang, Chengkai Han, Wayne Xin Zhao, Jingyuan Wang
> - **발표:** AAAI 2023 (The Thirty-Seventh AAAI Conference on Artificial Intelligence)
> - **arXiv:** [2301.07945](https://arxiv.org/abs/2301.07945)
> - **공식 게재:** [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/25556)
> - **코드:** [GitHub - BUAABIGSCity/PDFormer](https://github.com/BUAABIGSCity/PDFormer)

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

교통 흐름 예측의 근본적 과제는 교통 데이터 내의 복잡한 시공간 의존성을 효과적으로 모델링하는 것이다. GNN 기반 모델들이 유망한 방법으로 떠올랐으나, 세 가지 주요 한계가 존재한다: ① 공간 의존성을 정적 방식으로만 모델링하여 동적인 도시 교통 패턴 학습이 제한됨, ② 단거리 공간 정보만 고려하여 장거리 공간 의존성 포착 불가, ③ 교통 상황이 위치 간에 전파될 때 발생하는 **시간 지연(Propagation Delay)**을 무시함.

이를 해결하기 위해, 새로운 전파 지연 인식 동적 장거리 트랜스포머(PDFormer)를 제안하여 정확한 교통 흐름 예측을 수행한다.

### 주요 기여 (4가지)

| 기여 | 설명 |
|------|------|
| ① 동적 공간 자기 주의 | 핵심 기술 기여로, 동적 공간 의존성을 포착하는 새로운 공간 자기 주의(SSA) 모듈을 설계. 로컬 지리적 이웃과 전역 의미적 이웃 정보를 다른 그래프 마스킹 방법을 통해 자기 주의 상호작용에 통합하여 단거리 및 장거리 공간 의존성을 동시에 포착. |
| ② 지연 인식 특징 변환 | 이 모듈을 기반으로, 지연 인식 특징 변환 모듈을 추가 설계하여 역사적 교통 패턴을 공간 자기 주의에 통합하고 공간 정보 전파의 시간 지연을 명시적으로 모델링. |
| ③ 시간 자기 주의 | 시간 자기 주의 모듈을 채택하여 교통 데이터의 동적 시간 패턴을 식별. |
| ④ 해석 가능성 및 실험 검증 | 6개의 실제 공개 교통 데이터셋에 대한 광범위한 실험 결과로 최신 성능을 달성하고 경쟁력 있는 계산 효율성을 보임. 또한, 학습된 시공간 주의 맵을 시각화하여 모델의 높은 해석 가능성을 제공. |

---

## 2. 해결하고자 하는 문제 / 제안 방법 / 모델 구조 / 성능

### 2-1. 해결하고자 하는 문제

기존 GNN 기반 방법들은 공간 의존성을 정적으로만 모델링하여 동적인 특성을 반영하지 못하며, 단거리 공간 정보에만 집중하여 중요한 장거리 공간 의존성을 포착하지 못한다. 또한, 위치 간 교통 상황 변화의 전파 지연을 설명하지 못하는 것이 핵심 한계이다.

---

### 2-2. 문제 정의 (수식)

도로망 그래프 $\mathcal{G} = (\mathcal{V}, \mathcal{E}, A)$와 교통 흐름 텐서 $X = (X_1, X_2, \ldots, X_T) \in \mathbb{R}^{T \times N \times C}$가 주어질 때, 시각 $t$에서의 $N$개 노드의 교통 흐름 $X_t \in \mathbb{R}^{N \times C}$ ($C$는 교통 흐름 차원)을 기반으로, 역사적 관측값 $X$로부터 미래의 교통 흐름을 예측하는 것이 문제이다.

수식으로 표현하면:

$$f: (X_1, X_2, \ldots, X_T; \mathcal{G}) \rightarrow (\hat{X}_{T+1}, \hat{X}_{T+2}, \ldots, \hat{X}_{T+T'})$$

---

### 2-3. 제안하는 방법

#### (A) 공간 자기 주의 모듈 (Spatial Self-Attention, SSA)

핵심 기술 기여로, 동적 공간 의존성을 포착하는 공간 자기 주의 모듈을 설계. 이 모듈은 서로 다른 그래프 마스킹 방법을 통해 로컬 지리적 이웃과 전역 의미적 이웃 정보를 자기 주의 상호작용에 통합하여 교통 데이터의 단거리 및 장거리 공간 의존성을 동시에 포착한다.

두 가지 그래프 마스킹 행렬:


- **이진 지리적 마스킹 행렬 $M_{geo}$** (단거리): 두 노드 간의 홉(hop) 수가 임계값 $\lambda$보다 작으면 가중치 1, 그렇지 않으면 0으로 설정.
- **이진 의미적 마스킹 행렬 $M_{sem}$** (장거리): 동적 시간 왜곡(Dynamic Time Warping, DTW)을 노드 간에 적용하여 유사도가 가장 높은 Top-K 노드를 선택하고 가중치를 1로, 나머지는 0으로 설정. 학습 가능 파라미터 $W_Q^S, W_K^S, W_V^S \in \mathbb{R}^{d \times d'}$를 사용.


표준 Self-Attention 연산에 마스킹을 적용한 수식:

$$\text{Attention}(Q, K, V, M) = \text{softmax}\left(\frac{QK^T}{\sqrt{d'}} + M\right)V$$

여기서 $M \in \{M_{geo}, M_{sem}\}$, $Q = XW_Q^S$, $K = XW_K^S$, $V = XW_V^S$

두 개의 마스킹 기반 주의 결과를 결합하면:

$$\text{SSA}(X) = \text{Concat}(\text{Attention}_{geo}, \text{Attention}_{sem}) \cdot W_O$$

#### (B) 지연 인식 특징 변환 모듈 (Delay-Aware Feature Transformation)

이 모듈은 공간 자기 주의에 역사적 교통 패턴을 통합하고 공간 정보 전파의 시간 지연을 명시적으로 모델링한다.

지리적 및 의미적 공간 마스크를 주의 메커니즘에 도입하여 단거리 및 장거리 동적 공간 의존성을 모두 포착하며, 실제 도로의 전파 지연을 고려하는 지연 인식 특징 변환 모듈을 사용한다.

지연 인식 특징 변환의 핵심 수식 (논문 내 표현 기반):

$$\tilde{X}_t^{(k)} = X_{t - \Delta_k} \cdot W_k^{delay}$$

여기서 $\Delta_k$는 $k$번째 지연 인덱스, $W_k^{delay}$는 학습 가능한 변환 행렬이며, 여러 시간 지연된 입력을 집계하여 최종 지연 인식 표현을 생성한다:

$$Z^{delay} = \sum_{k=1}^{K} \alpha_k \cdot \tilde{X}_t^{(k)}$$

#### (C) 시간 자기 주의 모듈 (Temporal Self-Attention, TSA)

공간 자기 주의 모듈이 동적 장거리 공간 의존성을 포착하고, 시간 자기 주의 모듈은 교통 데이터의 동적 시간 패턴을 발견한다.

$$\text{TSA}(X) = \text{softmax}\left(\frac{Q_T K_T^T}{\sqrt{d_T}}\right)V_T$$

---

### 2-4. 전체 모델 구조

```
입력: 도로망 그래프 G, 역사적 교통 흐름 X (T × N × C)
           ↓
    [입력 임베딩 레이어]
           ↓
    ┌──────────────────────────────────────┐
    │     Spatial-Temporal Encoder Layer   │  (L번 반복)
    │  ┌─────────────────────────────────┐ │
    │  │ Spatial Self-Attention (SSA)    │ │
    │  │  ├─ M_geo (지리적 단거리 마스크) │ │
    │  │  └─ M_sem (의미적 장거리 마스크) │ │
    │  ├─────────────────────────────────┤ │
    │  │ Delay-Aware Feature Transform   │ │
    │  │  └─ 복수 시간 지연 집계          │ │
    │  ├─────────────────────────────────┤ │
    │  │ Temporal Self-Attention (TSA)   │ │
    │  └─────────────────────────────────┘ │
    └──────────────────────────────────────┘
           ↓
    [출력 예측 레이어]
           ↓
    예측값: X̂ (T' × N × C)
```

또한, 공간 정보 전파에서 시간 지연을 명시적으로 모델링하는 지연 인식 특징 변환 모듈을 설계하고, 6개의 실제 데이터셋에서 광범위한 실험을 통해 모델의 우수성을 검증하며 학습된 주의 맵을 시각화하여 모델의 해석 가능성을 제공한다.

---

### 2-5. 성능 향상

6개의 실제 공개 교통 데이터셋에 대한 광범위한 실험 결과에서 최신(SOTA) 성능을 달성하면서도 경쟁력 있는 계산 효율성을 보이며, 학습된 시공간 주의 맵을 시각화하여 모델을 고도로 해석 가능하게 만든다.

실험에 사용된 주요 데이터셋은 **PeMS03, PeMS04, PeMS07, PeMS08, METR-LA, PEMS-BAY** 이며, 평가 지표로는 MAE, RMSE, MAPE를 사용한다.

다중 스텝 및 단일 스텝 교통 흐름 예측 실험을 6개 실제 공개 데이터셋에서 수행하여, 최신 모델을 크게 능가하는 결과를 보이고 경쟁력 있는 계산 효율성을 보이며, 시각화 실험에서도 학습된 시공간 주의를 통해 높은 해석 가능성을 확인하였다.

---

### 2-6. 한계점

논문 자체에서 명시한 한계 및 연구자 커뮤니티에서 지적되는 한계:

1. **전처리 의존성**: 의미적 마스킹 행렬 $M_{sem}$ 생성에 DTW(Dynamic Time Warping)를 사용하는데, 이는 데이터 전처리 단계에서 계산 비용이 상당하며 사전에 고정되어 있어 온라인 적응이 어렵다.

2. **확장성 문제**: 자기 주의 메커니즘의 시간 복잡도가 $\mathcal{O}(N^2)$이므로, 노드 수 $N$이 매우 큰 대규모 도시 도로망에는 확장성의 한계가 존재한다.

3. **미래 연구 방향으로서의 한계 인정**: 향후 연구로 PDFormer를 풍력 발전 예측과 같은 다른 시공간 예측 과제에 적용할 것과, 교통 예측에서 데이터 부족 문제를 해결하기 위한 사전 학습(pre-training) 기법도 탐색할 것임을 명시하였다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 현재 모델이 일반화에 기여하는 요소

| 요소 | 일반화 기여 내용 |
|------|----------------|
| **동적 의미 마스크 ($M_{sem}$)** | DTW 기반 유사 노드 선택으로, 도로망 구조가 달라도 **데이터 기반으로 관계를 학습**하여 다양한 도시에 적응 가능 |
| **장거리 의존성 포착** | 지리적으로 멀리 있어도 의미적으로 유사한 노드 간 패턴을 포착하여 다양한 도로 위상에 일반화 가능 |
| **전파 지연 모델링** | 전파 지연 인식 동적 장거리 트랜스포머는 자기 주의 메커니즘으로 시공간 상관관계를 추출하고 교통 지연 인식 모듈을 통합하여, 노드 간 정보 전파의 시간 지연을 설명한다. 이 설계는 도시마다 전파 지연 패턴이 달라도 데이터에서 학습 가능하게 한다. |

### 3-2. 일반화 성능 향상을 위한 미래 방향

향후 PDFormer를 풍력 발전 예측 등 다른 시공간 예측 과제에 적용할 계획이며, 교통 예측에서 **데이터 부족 문제를 해결**하기 위한 **사전 학습(pre-training) 기법** 탐색도 예정되어 있다. 이는 일반화 성능 향상의 핵심 방향이다.

구체적인 일반화 향상 가능성:

- **Pre-training + Fine-tuning 패러다임 적용**: 대규모 교통 데이터로 사전 학습 후 소규모 도시 데이터에 파인튜닝함으로써, 데이터가 부족한 새로운 도시에도 적용 가능
- **그래프 구조 불가지론적 설계**: 의미 마스크를 통해 고정된 그래프 토폴로지 없이도 관계를 학습하므로, 그래프 구조가 다른 환경에도 적응 용이
- **멀티 모달 데이터 통합**: 기상 데이터, 이벤트 정보 등을 입력으로 통합하면 다양한 도시 환경에서 일반화 성능 강화 가능

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

### 4-1. 주요 관련 연구 비교표

| 모델 | 연도/학회 | 핵심 방법 | 공간 모델링 | 시간 모델링 | PDFormer 대비 특징 |
|------|-----------|-----------|------------|------------|-------------------|
| **DCRNN** | 2018 / ICLR | 확산 그래프 + GRU | 정적 GNN | RNN | PDFormer의 비교 베이스라인 |
| **STGCN** | 2018 / IJCAI | GCN + 1D-Conv | 정적 GCN | CNN | 정적·단거리 공간에 한계 |
| **Graph WaveNet** | 2019 / IJCAI | 적응적 그래프 + WaveNet | 동적 GCN | 팽창 Conv | 적응 그래프 선구자 |
| **AGCRN** | 2020 / NeurIPS | 적응적 GCN + RNN | 동적 GCN | GRU | 적응형 그래프 합성곱 순환 네트워크(AGCRN)로 교통 예측을 수행. |
| **ASTGNN** | 2021 / IEEE TITS | 동적 그래프 Conv + 시간 주의 | 동적 GNN | Self-Attention | 동적 그래프 + 주의 메커니즘 결합 |
| **DSTAGNN** | 2022 / ICML | 동적 시공간 그래프 | 동적 GNN | GNN | 동적 공간-시간 그래프 학습 |
| **PDFormer** | **2023 / AAAI** | 전파 지연 인식 Transformer | **동적+단/장거리** | Self-Attention | **전파 지연 명시적 모델링** |
| **STAEformer** | 2023 / CIKM | 시공간 적응 임베딩 + Vanilla Transformer | 임베딩 기반 | Self-Attention | 시공간 적응 임베딩 기법으로 Vanilla Transformer의 성능을 향상시키며, 교통 시계열 데이터의 복잡한 시공간 역학과 시간적 정보를 포착한다. |
| **STWave** | 2023 / ICDE | Wavelet 분리-융합 | 그래프 주의 | 이산 웨이블릿 변환 | 교통 데이터를 장기 트렌드와 단기 이벤트로 분리하기 위해 이산 웨이블릿 변환(DWT)을 사용하며, 2채널 시공간 네트워크로 시공간 특징을 모델링. |
| **DDGFormer** | 2024 | 거리·방향 인식 Self-Attention | 동적 그래프 | Self-Attention | PDFormer 이후, DDGFormer는 거리 및 방향 인식 자기 주의 모듈과 동적 그래프를 결합한 모델. |
| **MGTEFormer** | 2024 | 다중 세분도 시간 임베딩 | Transformer | 다중 세분도 임베딩 | PDFormer 및 STAEformer와 같은 Transformer 모델들과 동일한 과제를 대상으로 비교 실험. |

### 4-2. PDFormer의 위치

PDFormer는 교통 흐름 예측의 복잡한 시공간 의존성을 해결하기 위해 특별히 설계된 전파 지연 인식 동적 장거리 Transformer를 소개한다.

Transformer 기반 교통 흐름 예측 모델들이 부상하고 있으며, 그 중 STAEFormer는 유사한 임베딩 기법을 활용하고, PDFormer는 유사한 마스킹 전략을 도입하였다.

---

## 5. 앞으로의 연구에 미치는 영향과 고려할 점

### 5-1. 앞으로의 연구에 미치는 영향

#### ✅ 이론적 기여
1. **전파 지연의 명시적 모델링 패러다임 제시**: 교통 예측 분야에서 시간 지연을 명시적으로 설계에 반영한 것은 이후 연구에 큰 영향을 미쳐, PDFormer는 공간 정보 전파의 시간 지연을 활용하는 교통 지연 인식 특징 변환 예측 모델로 제시되었다.

2. **이중 마스킹 전략의 확산**: PDFormer가 지리적 및 의미적 공간 마스크를 주의 메커니즘에 도입하여 단거리와 장거리 동적 공간 의존성을 모두 포착한 전략은 이후 여러 연구에서 참조·확장되고 있다.

3. **해석 가능한 AI 방향**: 학습된 시공간 주의 맵을 시각화하여 모델을 높은 해석 가능성을 갖추게 한 것은 XAI(설명 가능 AI) 관점에서도 교통 분야 연구에 영향을 미친다.

#### ✅ 실용적 기여
- **오픈소스 공개**: LibCity 프레임워크 기반으로 개발된 코드를 오픈소스 교통 예측 라이브러리로 공개하여, 후속 연구자들이 쉽게 재현하고 비교 실험을 수행할 수 있다.

---

### 5-2. 앞으로의 연구 시 고려할 점

#### 🔬 방법론적 고려사항

| 고려 항목 | 세부 내용 |
|-----------|-----------|
| **Pre-training 전략** | 교통 예측에서 데이터 부족 문제를 해결하기 위한 사전 학습 기법 탐색이 중요하다. LLM 기반 접근 혹은 자기 지도 학습(Self-Supervised Learning)과의 결합이 유망하다. |
| **확장성 (Scalability)** | Self-Attention의 $\mathcal{O}(N^2)$ 복잡도를 해결하기 위해, Sparse Attention, Linear Attention, 또는 계층적 구조 도입이 필요하다. |
| **멀티 도메인 일반화** | PDFormer를 풍력 발전 예측 등 다른 시공간 예측 과제에 적용하는 연구가 예고되어 있으며, 도메인 일반화 연구가 중요해진다. |
| **동적 지연 추정** | 현재 DTW 기반 전처리로 고정된 지연 구조를 사용하는데, 실시간으로 지연 값을 추정하는 온라인 방식으로의 발전이 필요하다. |
| **이질적 교통 데이터 통합** | 도로 유형, 날씨, 사고 데이터 등 이질적 정보를 통합하는 멀티모달 접근법과의 결합 |
| **LLM과의 결합** | 사전 학습된 언어 모델(LM)이 시계열 데이터로 훈련되지 않았음에도 예측 성능을 향상시킬 수 있음이 최근 연구에서 확인되어, LLM과 PDFormer 구조 결합이 유망한 방향이다. |
| **지속 학습 (Continual Learning)** | 교통 패턴의 비정상성(non-stationarity)에 대응하기 위해, 모델이 새로운 데이터에 점진적으로 적응하는 지속 학습 기법이 중요하다. |

#### 🧪 실험적 고려사항

- **공정한 비교 기준 설정**: 다양한 데이터셋과 메트릭에서 일관된 비교가 중요하며, LibCity와 같은 통합 벤치마크 활용이 권장된다.
- **계산 효율성 vs. 예측 정확도 트레이드오프**: 대규모 실시간 시스템에서는 예측 정확도뿐 아니라 지연 시간(latency)도 핵심 지표임을 고려해야 한다.
- **Cross-city 일반화 평가**: 특정 도시로 학습 후 다른 도시에서 평가하는 Zero-shot / Few-shot 시나리오에서의 성능 평가가 앞으로 중요해진다.

---

## 📚 참고 자료 (출처 목록)

| # | 제목 / 출처 | URL |
|---|------------|-----|
| 1 | **PDFormer 논문 (arXiv)** - Jiawei Jiang et al., 2023 | https://arxiv.org/abs/2301.07945 |
| 2 | **PDFormer (AAAI OJS 공식)** - AAAI 2023 | https://ojs.aaai.org/index.php/AAAI/article/view/25556 |
| 3 | **PDFormer (ACM DL)** | https://dl.acm.org/doi/10.1609/aaai.v37i4.25556 |
| 4 | **PDFormer GitHub (공식 코드)** - BUAABIGSCity | https://github.com/BUAABIGSCity/PDFormer |
| 5 | **PDFormer Quick Review** - Liner.com | https://liner.com/review/pdformer-propagation-delayaware... |
| 6 | **PDFormer ResearchGate** | https://www.researchgate.net/publication/367280811 |
| 7 | **Lab Seminar Slides (SlideShare)** - 2024 | https://www.slideshare.net/slideshow/20240710_labseminar... |
| 8 | **STAEformer 논문 (arXiv)** - Liu et al., CIKM 2023 | https://arxiv.org/pdf/2308.10425 |
| 9 | **Multi-scale ST Transformer (Scientific Reports, 2025)** | https://www.nature.com/articles/s41598-025-33625-z |
| 10 | **STTLM (Sensors, 2024)** - Ma et al. | https://www.mdpi.com/1424-8220/24/17/5502 |
| 11 | **MGTEFormer (PMC, 2024)** | https://pmc.ncbi.nlm.nih.gov/articles/PMC11678995/ |
| 12 | **TARGCN (Complex & Intelligent Systems, 2024)** | https://link.springer.com/article/10.1007/s40747-024-01601-1 |
| 13 | **Traffic Flow Prediction Review (MaxAPress, 2025)** | https://www.maxapress.com/article/doi/10.48130/dts-0025-0027 |
| 14 | **PatchSTG (arXiv, 2024)** | https://arxiv.org/html/2412.09972v1 |
| 15 | **DeepAI - PDFormer** | https://deepai.org/publication/pdformer-propagation-delay-aware... |
| 16 | **Scite.ai - PDFormer** | https://scite.ai/reports/pdformer-propagation-delay-aware-dynamic-long-range-PQwXxXLQ |

---

> ⚠️ **정확도 주의사항**: 본 답변에서 수식 일부(특히 지연 인식 특징 변환 모듈의 세부 수식)는 논문 전문의 완전한 접근이 제한된 관계로, 논문의 공개 설명과 관련 리뷰 자료를 기반으로 재구성하였습니다. 정확한 수식의 완전한 형태는 반드시 [공식 논문 PDF](https://arxiv.org/pdf/2301.07945)를 직접 확인하시기 바랍니다.
