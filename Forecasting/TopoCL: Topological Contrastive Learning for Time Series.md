
# TopoCL: Topological Contrastive Learning for Time Series

## 논문 기본 정보
- **저자**: Namwoo Kim, Hyungryul Baik, Yoonjin Yoon
- **게재**: arXiv:2502.02924, Submitted to TNNLS (under review)
- **분야**: Machine Learning (cs.LG), Artificial Intelligence (cs.AI)
- **발표일**: 2025년 2월 5일

---

## 1. 핵심 주장 및 주요 기여 (요약)

시계열 표현의 범용적 학습(Universal time series representation learning)은 분류, 이상 탐지, 예측 등 실세계 응용에서 중요하면서도 도전적인 과제이다. 그러나 대조 학습(CL)에서의 데이터 증강 과정은 계절 패턴이나 시간적 의존성을 왜곡하여, 의미 정보의 손실을 초래하는 핵심 문제가 있다.

이를 해결하기 위해, TopoCL(Topological Contrastive Learning for time series)을 제안하며, 이는 변환에 불변(invariant)하는 데이터의 위상적 특성을 포착하는 지속적 호몰로지(persistent homology)를 통합하여 정보 손실을 완화한다.

**주요 기여 3가지:**

1. 위상적 특성을 통합하는 것이 시계열 데이터의 강건하고 변별력 있는 표현을 포착하는 데 효과적임을 입증하였다.
2. 시계열 데이터의 토폴로지를 고려한 새로운 표현 학습 프레임워크를 설계하였으며, 이는 시계열 표현 학습의 맥락에서 지속적 호몰로지와 대조 학습을 결합한 최초의 접근법이다.
3. 시계열 분류, 이상 탐지, 예측 및 전이 학습의 4가지 다운스트림 태스크에 대한 광범위한 실험을 수행하여 제안 모델의 유효성을 입증하였다.

---

## 2. 해결하고자 하는 문제

### 2.1 문제 정의

범용 시계열 표현 학습은 분류, 이상 탐지, 예측 등에서 가치 있지만 도전적이다. 최근 대조 학습(CL)이 시계열 표현을 다루기 위해 활발히 탐구되고 있으나, CL의 데이터 증강 과정이 계절 패턴이나 시간적 의존성을 왜곡하여 의미 정보 손실을 초래하는 핵심 문제가 있다.

추가로, 시계열 데이터 분석은 데이터의 불완전성, 노이즈 취약성, 복잡성 등으로 인해 어렵고, 딥러닝 모델을 적용하려면 잘 레이블된 데이터가 필요하다는 추가적 장벽이 있다.

### 2.2 기존 방법의 한계

기존 시계열 대조 학습 방법들의 구체적 한계:

| 기존 방법 | 한계 |
|-----------|------|
| **TS2Vec** (Yue et al., 2022) | 계층적 대조 학습이지만 증강에 의한 의미 정보 손실 문제 미해결 |
| **TNC** (Tonekaboni et al., 2021) | 특정 수준의 세분성에서만 시간적 지역 평활성 활용 |
| **TS-TCC** (Eldele et al., 2021) | 타임스탬프 레벨에서만 인스턴스 대조 수행 |
| **CoST** (Woo et al., 2022) | 시간-주파수 도메인 분리에 의존, 토폴로지 미고려 |
| **TF-C** (Zhang et al., 2022) | 시간-주파수 일관성에 기반하나 구조적 위상 특성 미포착 |

---

## 3. 제안 방법 (TopoCL Framework)

### 3.1 전체 파이프라인

TopoCL은 시간(time) 모달리티와 토폴로지(topology) 모달리티의 정보를 통합하여 활용한다. 먼저 시계열 데이터로부터 위상적 특성을 구성한다. 시계열 데이터에 지연 임베딩(delay embedding)을 적용하고, 지속적 호몰로지를 계산하며, 간단한 위상 모듈을 설계하여 위상 정보를 효과적으로 인코딩한다.

### 3.2 단계별 방법론

#### Step 1: 지연 임베딩 (Delay Embedding)

시계열 $x = \{x_1, x_2, \ldots, x_T\}$에 대해 Takens의 지연 임베딩 정리를 적용하여 고차원 점 구름(point cloud)을 생성한다:

$$\mathbf{v}_i = (x_i, x_{i+\tau}, x_{i+2\tau}, \ldots, x_{i+(d-1)\tau})$$

여기서 $\tau$는 시간 지연(time delay), $d$는 임베딩 차원(embedding dimension)이다.

#### Step 2: 지속적 호몰로지 (Persistent Homology) 계산

지연 임베딩으로 생성된 점 구름에 대해 Vietoris-Rips 복합체를 구성하고, 필트레이션 파라미터 $\epsilon$을 변화시키며 위상적 특징을 추적한다:

$$\text{VR}_\epsilon = \{\sigma \subseteq X \mid \text{diam}(\sigma) \leq \epsilon\}$$

시계열 데이터의 시간적 및 위상적 속성을 별도의 모달리티로 취급하며, 구체적으로 지속적 호몰로지를 계산하여 시계열 데이터의 위상적 특징을 구성하고 이를 지속성 다이어그램(persistence diagram)으로 표현한다.

지속성 다이어그램은 다음과 같이 정의된다:

$$\text{PD} = \{(b_i, d_i)\}_{i=1}^{n}$$

여기서 $b_i$는 위상적 특징의 출생(birth) 시점, $d_i$는 사망(death) 시점을 나타낸다. $H_0$ (연결 성분)과 $H_1$ (루프/구멍)의 두 가지 차원을 사용한다.

#### Step 3: 토폴로지 인코더 ($f_\theta^{topo}$)

지속성 다이어그램을 인코딩하기 위한 신경망을 설계한다. Ablation 연구에서 (5) avg-pool 대신 symmetric aggregate function (max-pool)을 사용하는 것이 토폴로지 인코더의 핵심 설계임을 확인하였다.

지속성 다이어그램의 각 점 $(b_i, d_i)$를 변환하여:

$$\mathbf{h}_i^{topo} = \text{MLP}([b_i, d_i, d_i - b_i])$$

$$\mathbf{z}^{topo} = \text{MaxPool}\left(\{\mathbf{h}_i^{topo}\}_{i=1}^{n}\right)$$

#### Step 4: 결합 학습 목적함수 (Joint Learning Objective)

증강된 동일 시계열 버전이 특징 공간에서 가깝게 임베딩되도록 하면서 시간과 토폴로지 간의 대응 관계를 보존하는 결합 목적에 초점을 맞춘다.

전체 손실 함수는 두 가지 구성요소의 결합이다:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{time}} + \lambda \cdot \mathcal{L}_{\text{time-topo}}$$

**(a) 시간 모달리티 대조 손실 ($\mathcal{L}_{\text{time}}$)**:

인스턴스 판별을 위한 대조 손실은 토폴로지 모달리티에는 적용되지 않는다. 대신, 시간 모달리티에 대조 손실을 적용하고, 토폴로지 모달리티를 활용하여 시계열 표현 학습을 강화한다.

$$\mathcal{L}_{\text{time}} = \mathcal{L}_{\text{instance}} + \mathcal{L}_{\text{temporal}}$$

여기서 인스턴스 대조 손실과 시간 대조 손실은 각각:

$$\mathcal{L}_{\text{instance}} = -\log \frac{\exp(\text{sim}(\mathbf{z}_i^t, \mathbf{z}_i^{t'})/\tau)}{\sum_{j=1}^{N} \exp(\text{sim}(\mathbf{z}_i^t, \mathbf{z}_j^{t'})/\tau)}$$

**(b) 시간-토폴로지 정렬 손실 ($\mathcal{L}_{\text{time-topo}}$)**:

시간-토폴로지 정렬은 시간적 특징과 그에 대응하는 위상적 특징을 통합함으로써 포괄적이고 상보적인 의미 정보를 학습할 수 있게 한다는 가정에 기반한다.

$$\mathcal{L}_{\text{time-topo}} = -\log \frac{\exp(\text{sim}(\mathbf{z}_i^{time}, \mathbf{z}_i^{topo})/\tau)}{\sum_{j=1}^{N} \exp(\text{sim}(\mathbf{z}_i^{time}, \mathbf{z}_j^{topo})/\tau)}$$

### 3.3 모델 구조 다이어그램

```
시계열 입력 x
├─── [데이터 증강] ──→ x', x'' (두 augmented view)
│    ├─── [시간 인코더 f_θ^time] ──→ z^time (시간 표현)
│    └─── 시간 모달리티 대조 손실 (L_time)
│
├─── [지연 임베딩] ──→ 점 구름 (Point Cloud)
│    ├─── [지속적 호몰로지] ──→ 지속성 다이어그램 (PD)
│    └─── [토폴로지 인코더 f_θ^topo] ──→ z^topo (위상 표현)
│
└─── [시간-토폴로지 정렬] ──→ L_time-topo
     └─── 최종 손실: L_total = L_time + λ · L_time-topo
```

---

## 4. 성능 향상 및 실험 결과

### 4.1 평가 태스크 및 성능

분류, 이상 탐지, 예측, 전이 학습의 4가지 다운스트림 태스크에 대한 광범위한 실험을 수행하였으며, TopoCL이 최신 성능(state-of-the-art)을 달성하였다.

| 태스크 | 데이터셋 | 비교 방법 |
|--------|---------|----------|
| **분류** | 128 UCR 데이터셋 | Ablation 5가지 변형 비교 |
| **이상 탐지** | Yahoo, KPI | TS2Vec에 TopoCL 적용 |
| **예측** | 다양한 시계열 벤치마크 | 기존 SOTA 대비 |
| **전이 학습** | 크로스-도메인 | 일반화 성능 평가 |

이상 탐지 태스크에서는 Yahoo와 KPI 데이터셋을 사용하며, TopoCL을 TS2Vec에 적용하여 일반 및 콜드스타트 설정에서 평가를 수행하였다.

### 4.2 Ablation 연구

128개 UCR 데이터셋에서 TopoCL과 5가지 변형을 비교하였다: (1) w/o time-topology alignment, (2) w/o time domain contrastive loss, (3) w/o $H_0$, (4) w/o $H_1$, (5) avg-pool 대체.

Ablation 결과는 시간 모달리티 대조 손실과 시간-토폴로지 대응의 결합 학습이 이러한 역량을 향상시킴을 보여준다.

---

## 5. 모델의 일반화 성능 향상 가능성 (핵심 분석)

### 5.1 위상적 불변성에 의한 증강 강건성

TopoCL은 다양한 데이터 증강 기법에 걸쳐 일관된 성능을 보여주며, 이는 증강으로 인한 정보 손실을 완화하는 데 효과적임을 시사한다.

**핵심 통찰**: 지속적 호몰로지는 본질적으로 연속 변환에 대해 안정적(stable)이라는 수학적 성질을 가진다. 이는 **stability theorem**으로 공식화된다:

$$d_B(\text{PD}(f), \text{PD}(g)) \leq \|f - g\|_\infty$$

여기서 $d_B$는 병목 거리(bottleneck distance)이다. 이 성질 덕분에, 데이터 증강으로 인한 작은 섭동은 위상적 특징의 작은 변화만을 유발하므로, 표현의 강건성이 보장된다.

### 5.2 크로스-모달 정렬의 일반화 효과

시간과 토폴로지를 별도 모달리티로 취급하고 결합 학습 목적을 제안하여 양자의 이해를 향상시키며, 데이터 증강에 대한 강건성도 개선한다. TopoCL은 분류, 예측, 이상 탐지, 전이 학습 등 다수의 태스크에 걸쳐 평가되어, 보편성(universality), 일반화 능력(generalization capability), 유효성을 입증하였다.

### 5.3 제한된 데이터에서의 일반화

제안된 방법은 시간-토폴로지 정렬이 없는 경우보다 일관되게 우수한 성능을 보이며, 이는 위상적 특성의 활용이 소량의 데이터로도 데이터의 내재적 특성을 포착할 수 있음을 시사한다.

### 5.4 전이 학습에서의 일반화

전이 학습 태스크에서도 평가되어, 한 도메인에서 학습한 표현이 다른 도메인에서도 유효함을 확인하였다. 이는 위상적 특징이 도메인 독립적인 구조적 정보를 인코딩하기 때문이다.

---

## 6. 한계점

검색된 정보와 논문의 구조적 특성에 기반하여, 다음과 같은 한계를 추론할 수 있다:

1. **계산 비용**: 지속적 호몰로지 계산의 시간 복잡도가 $O(n^3)$ (또는 최적화된 경우에도 상당한 비용)으로, 대규모 시계열에 대한 확장성 문제가 있을 수 있음
2. **지연 임베딩 하이퍼파라미터**: $\tau$ (지연)과 $d$ (차원)의 선택이 성능에 영향을 미칠 수 있으며, 이에 대한 자동 선택 메커니즘이 필요
3. **다변량 시계열 확장**: 다변량 시계열에서의 위상적 특징 추출 방법에 대한 추가 연구 필요
4. **위상 인코더의 단순성**: MLP + MaxPool 구조의 위상 인코더가 복잡한 위상 정보를 충분히 포착하지 못할 수 있음
5. **이론적 보장의 부재**: 시간-토폴로지 정렬이 표현 학습을 향상시키는 이론적 근거에 대한 엄밀한 분석이 부족

---

## 7. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 아이디어 | TopoCL과의 차이 |
|------|------|-------------|----------------|
| **TS2Vec** (Yue et al.) | 2022 (AAAI) | 임의의 시맨틱 레벨에서 시계열 표현을 학습하는 범용 프레임워크로, 증강된 컨텍스트 뷰에 대해 계층적으로 대조 학습을 수행 | 토폴로지 정보 미활용; TopoCL의 기반 아키텍처로 사용 |
| **TNC** (Tonekaboni et al.) | 2021 | 먼 세그먼트를 음성 쌍, 인접 세그먼트를 양성 쌍으로 가정하며, 신호 생성 과정의 국소 평활성을 활용하여 정상 특성을 가진 시간적 이웃을 정의 | 위상적 구조 미고려; 단일 수준 세분성 |
| **TS-TCC** (Eldele et al.) | 2021 | 약한/강한 증강으로 두 뷰를 생성하여 시간적 대조 모듈로 강건한 시간적 표현 학습 | 증강 의존적; 토폴로지 미활용 |
| **CoST** (Woo et al.) | 2022 (ICLR) | 시간 도메인과 주파수 도메인 대조 손실을 모두 활용하여 분리된 계절-추세 표현을 학습 | 주파수 기반 분리 vs. 토폴로지 기반 보완 |
| **TF-C** (Zhang et al.) | 2022 (NeurIPS) | 시간 및 주파수 기반 표현을 모두 학습하고 새로운 시간-주파수 일관성 아키텍처를 제안 | 주파수 모달리티 vs. 토폴로지 모달리티 |
| **SoftCLT** (2024, ICLR) | 2024 | 소프트 인스턴스 및 시간 대조 손실로 유사도 기반 가중치 부여 | 증강 자체의 정보 손실 문제 미해결 |
| **Series2Vec** (Foumani et al.) | 2024 | 시간 및 스펙트럼 도메인에서 두 시리즈 간 유사도를 예측하는 자기 지도 태스크로 훈련되며, 수작업 데이터 증강 필요성을 제거 | 증강 불필요 접근 vs. 증강 보완 접근 |
| **TopoGCL** (2024, AAAI) | 2024 | 확장 지속성(extended persistence)을 활용한 topo-topo CL 모드로 국소 및 전역 잠재 위상 정보를 포착하는 새로운 대조 모드 채택 (그래프 도메인) | 그래프 vs. 시계열; extended persistence vs. standard persistence |
| **High-TS** (2024) | 2024 | 다중 스케일 임베딩 모듈과 단순 복합체 임베딩 모듈로 구성되며, 단순 복합체의 higher-order interaction을 구축하고 TDL로 학습, 대조 학습으로 서로 다른 모달리티의 표현을 정렬 | 단순 복합체 기반 vs. 지속적 호몰로지 기반 |
| **CaTT** (2024) | 2024 | 시간적 일관성으로부터 시계열 표현을 학습 | 위상 특성 미활용; 효율성에 초점 |

### 핵심 차별점 분석

TopoCL이 기존 연구와 구별되는 가장 중요한 점은:

1. **시계열에서의 TDA-CL 최초 결합**: 시계열 표현 학습의 맥락에서 지속적 호몰로지와 대조 학습을 결합한 최초의 접근법이다.

2. **크로스-모달 설계**: 기존 방법들이 시간-주파수(time-frequency) 크로스-모달 학습에 초점을 맞춘 반면, TopoCL은 시간-토폴로지(time-topology) 크로스-모달 학습이라는 새로운 축을 제안한다.

3. **증강 불변 보완 정보**: 위상적 특징은 수학적으로 연속 변환에 대해 안정적이므로, 증강으로 인한 정보 손실을 구조적으로 보완한다.

---

## 8. 앞으로의 연구에 미치는 영향 및 고려사항

### 8.1 연구 영향

1. **시계열 분석에서의 TDA 활용 확대**: TopoCL은 TDA(Topological Data Analysis)가 시계열 대조 학습에서 유효한 보완적 정보원임을 실증적으로 보여줌으로써, 후속 연구에서 위상적 특징을 활용하는 새로운 연구 방향을 개척

2. **멀티-모달 시계열 표현 학습의 새 패러다임**: 시간-토폴로지 크로스-모달 학습의 성공은 시간-주파수, 시간-그래프, 시간-위상 등 다양한 크로스-모달 조합의 가능성을 시사

3. **증강 강건성의 이론적 기반 마련**: 위상적 안정성 정리에 기반한 증강 강건성은 향후 대조 학습에서 증강 설계의 이론적 프레임워크를 제공

### 8.2 앞으로 연구 시 고려할 점

| 고려 사항 | 상세 내용 |
|----------|----------|
| **확장성 개선** | 대규모 시계열에 대한 지속적 호몰로지의 계산 효율화 (approximate PH, GPU-accelerated PH 등) |
| **적응적 임베딩** | 지연 임베딩 파라미터 $(\tau, d)$의 자동 최적화 메커니즘 개발 (mutual information 기반 등) |
| **이론적 분석 강화** | 시간-토폴로지 정렬이 표현 학습에 미치는 정보 이론적 분석 (mutual information bound, generalization bound 등) |
| **고차원 위상 특징** | $H_0, H_1$을 넘어 $H_2$ 이상의 고차 호몰로지 활용 가능성 탐구 |
| **다변량 확장** | 다변량 시계열에서의 채널 간 위상적 관계 모델링 (simplicial complex 활용 등) |
| **연속적 미분 가능한 TDA** | 지속적 호몰로지의 미분 가능한 구현을 통한 end-to-end 최적화 강화 |
| **Foundation Model 통합** | 대규모 시계열 기반 모델(foundation model)에 위상적 특성을 통합하는 연구 |
| **실시간 응용** | 스트리밍 시계열에 대한 온라인 위상 특성 업데이트 메커니즘 개발 |
| **도메인 특화 위상 특성** | 의료, 금융, IoT 등 특정 도메인에서 의미 있는 위상적 특성의 식별 및 활용 |

### 8.3 후속 연구 방향 제안

1. **Differentiable TDA Pipeline**: 전체 파이프라인(지연 임베딩 → PH 계산 → 인코딩)을 미분 가능하게 만들어 진정한 end-to-end 학습 달성

2. **Multi-Resolution Topology**: 다중 해상도에서의 위상적 특성을 동시에 활용하는 계층적 위상 학습

3. **Topology-Aware Augmentation**: 위상적 특성을 보존하도록 설계된 데이터 증강 전략 개발

4. **Zero-shot/Few-shot Transfer**: 위상적 특성의 도메인 불변성을 활용한 zero-shot 및 few-shot 전이 학습 탐구

---

## 참고 자료

1. **Kim, N., Baik, H., & Yoon, Y.** (2025). "TopoCL: Topological Contrastive Learning for Time Series." arXiv:2502.02924. https://arxiv.org/abs/2502.02924
2. **Kim, N., Baik, H., & Yoon, Y.** (2025). TopoCL: Topological Contrastive Learning for Time Series (HTML version). https://arxiv.org/html/2502.02924v1
3. **Yue, Z. et al.** (2022). "TS2Vec: Towards Universal Representation of Time Series." AAAI 2022. https://arxiv.org/abs/2106.10466
4. **Zhou, X. et al.** (2024). "TopoGCL: Topological Graph Contrastive Learning." AAAI 2024. https://arxiv.org/html/2406.17251v1
5. **Foumani, N. M. et al.** (2024). "Series2vec: Similarity-based Self-supervised Representation Learning for Time Series Classification." Data Mining and Knowledge Discovery. https://link.springer.com/article/10.1007/s10618-024-01043-w
6. **SoftCLT** (2024). "Soft Contrastive Learning for Time Series." ICLR 2024. https://proceedings.iclr.cc/paper_files/paper/2024/file/ccc48eade8845cbc0b44384e8c49889a-Paper-Conference.pdf
7. **Wang, L. et al.** (2024). "Higher-order Cross-structural Embedding Model for Time Series Analysis." https://arxiv.org/html/2410.22984
8. **CaTT** (2024). "Contrast All The Time: Learning Time Series Representation from Temporal Consistency." https://arxiv.org/html/2410.15416v2
9. **AutoTCL** (2024). "Parametric Augmentation for Time Series Contrastive Learning." https://arxiv.org/html/2402.10434v1

---

> **⚠️ 정확성 주의**: 본 분석에서 수식의 세부 사항(특히 토폴로지 인코더의 구체적 구조, 손실 함수의 정확한 하이퍼파라미터 등)은 논문의 PDF 전문 확인 후 수정이 필요할 수 있습니다. arXiv abstract 및 HTML 버전에서 공개된 정보를 기반으로 작성되었으며, 전체 논문을 직접 확인하시는 것을 권장합니다.
