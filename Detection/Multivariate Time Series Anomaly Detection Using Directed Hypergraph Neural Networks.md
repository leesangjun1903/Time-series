# Multivariate Time Series Anomaly Detection Using Directed Hypergraph Neural Networks

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장
기존 그래프 신경망(GNN) 기반 다변량 시계열 이상 탐지 방법들은 **변수-변수 관계(variable-variable relationships)**, 즉 두 변수 간의 쌍별(pairwise) 관계만을 모델링할 수 있다. 그러나 실제 복잡 시스템에서는 한 변수의 행동이 **여러 변수의 그룹에 의해 동시에 영향**을 받는 **변수-그룹 관계(variable-group relationships)**가 존재한다. 이 논문은 이러한 한계를 극복하기 위해 **방향성 하이퍼그래프(Directed Hypergraph)**를 활용한 새로운 이상 탐지 방법 **DHG-AD**를 제안한다.

### 주요 기여
1. **방향성 하이퍼그래프를 통한 변수-그룹 관계 모델링**: 양의 상관(P-DHG)과 음의 상관(N-DHG) 두 종류의 방향성 하이퍼그래프를 구축하여 변수와 변수 그룹 간의 관계를 포괄적으로 포착
2. **이중 하이퍼그래프 접근법(Dual Hypergraph Approach)**: 양의 상관과 음의 상관을 동시에 모델링하여 다변량 상호작용의 포괄적 표현 학습
3. **두 개의 실세계 데이터셋(Exathlon, SMD)에서 모든 평가 지표에서 최고 성능** 달성

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**다변량 시계열 이상 탐지(Multivariate Time Series Anomaly Detection)** 문제를 다룬다.

- **문제 정의**: 다변량 시계열 $\mathcal{X} = \{x_1, \ldots, x_T\}$에서 각 시점 $t$의 데이터 포인트 $x_t \in \mathbb{R}^n$에 대해 이상 여부 레이블 $y_t \in \{0, 1\}$을 예측
- **비지도 학습 기반**: 정상 상태의 훈련 데이터 $\mathcal{X}\_{train}$으로 학습하고, 테스트 데이터 $\mathcal{X}_{test}$에서 학습된 정상 패턴과의 편차를 기반으로 이상을 탐지
- **핵심 한계**: 기존 GNN 기반 방법(MTAD-GAT, GDN, STGAT-MAD 등)은 그래프의 엣지로 두 변수 간 관계만 표현 가능하며, 한 변수가 여러 변수 그룹에 의해 동시에 영향을 받는 관계를 포착하기 어려움

### 2.2 제안하는 방법 (수식 포함)

#### (1) 입력 전처리

슬라이딩 윈도우 $W_t = \{x_{t-w+1}, \ldots, x_t\}$ (크기 $w$)를 $K$개의 겹치는 세그먼트 $\{S_1, \ldots, S_K\}$로 분할한다. 각 세그먼트의 시작 시점은 고정 스트라이드 $\eta$에 의해 결정된다:

$$t_k = t_{k-1} + \eta, \quad t_1 = t - w + 1, \quad \eta \leq l_s$$

#### (2) 피어슨 상관 행렬 계산

각 세그먼트 $S_k$ 내에서 변수 $i$와 $j$의 상관 계수:

$$\rho_k(i, j) = \frac{\sum_{t=t_k}^{t_k+l_s-1} (x_t^i - \mu_k^i)(x_t^j - \mu_k^j)}{\sqrt{\sum_{t=t_k}^{t_k+l_s-1} (x_t^i - \mu_k^i)^2} \sqrt{\sum_{k=t_k}^{t_k+l_s-1} (x_t^j - \mu_k^j)^2}} $$

#### (3) P-DHG 구축 (양의 상관 그룹)

각 노드 $v_i$에 대해 하이퍼아크 $e_i = (e_i^{\text{tail}}, e_i^{\text{head}})$를 정의:

$$e_i^{\text{tail}} = \{v_i\} \cup \{v_j : 1 \leq j \leq n, j \neq i \mid \rho_k(i,j) > \tau_P\} $$

$$e_i^{\text{head}} = \{v_i\} $$

#### (4) N-DHG 구축 (음의 상관 그룹)

$$e_i^{\text{tail}} = \{v_i\} \cup \{v_j : 1 \leq j \leq n, j \neq i \mid \rho_k(i,j) < \tau_N\} $$

$$e_i^{\text{head}} = \{v_i\} \tag{8}$$

#### (5) 결합 행렬(Incidence Matrices)

```math
H^{\text{tail}}(i,j) = \begin{cases} 1, & \text{if } v_i \in e_j^{\text{tail}}, \\ 0, & \text{otherwise.} \end{cases}
```

```math
H^{\text{head}}(i,j) = \begin{cases} 1, & \text{if } v_i \in e_j^{\text{head}}, \\ 0, & \text{otherwise.} \end{cases}
```

#### (6) 하이퍼아크 가중치

$$w(e_j) = \frac{\text{occurrences of } e_j \text{ in all segments}}{K} $$

#### (7) 방향성 하이퍼그래프 컨볼루션

노드 표현 행렬 $Z \in \mathbb{R}^{n \times d}$:

$$Z = \Delta F \Theta $$

라플라시안 행렬 $\Delta$:

$$\Delta = I - \frac{\Pi^{1/2} Q \Pi^{-1/2} + \Pi^{1/2} Q^T \Pi^{-1/2}}{2} $$

전이 확률 행렬 $Q$:

$$Q = (D_v^{\text{tail}})^{-1} H^{\text{tail}} \mathcal{W}_{\text{arc}} (D_e^{\text{head}})^{-1} (H^{\text{head}})^T $$

여기서 $\Pi \in \mathbb{R}^{n \times n}$은 방향성 하이퍼그래프 상의 랜덤 워크의 정상 분포를 나타내는 대각 행렬, $\Theta \in \mathbb{R}^{(K \times l_s) \times d}$는 학습 가능한 가중치이다.

#### (8) 재구성(Reconstruction)

통합 노드 표현: $U = [Z_P \| Z_N] \in \mathbb{R}^{n \times 2d}$

재구성된 윈도우:

$$\hat{W}_t = \sigma(U \mathcal{W}_{fc}) $$

여기서 $\mathcal{W}_{fc} \in \mathbb{R}^{2d \times w}$는 완전 연결 층의 가중치, $\sigma(\cdot)$는 시그모이드 활성화 함수이다.

#### (9) 손실 함수 (훈련)

$$\mathcal{L}_{\text{MSE}} = \frac{1}{w} \sum_{u=t-w+1}^{t} \| x_u - \hat{x}_u \|_2^2 $$

#### (10) 이상 점수 (테스트)

$$\alpha_t = \frac{1}{l_a} \sum_{u=t-l_a+1}^{t} \| x_u - \hat{x}_u \|_2^2 $$

$\alpha_t > \delta$이면 시점 $t$에서 이상 발생으로 판단.

### 2.3 모델 구조

DHG-AD는 **세 가지 주요 구성 요소**로 이루어진다:

| 구성 요소 | 설명 |
|---------|------|
| **방향성 하이퍼그래프 구축** | 입력 윈도우를 $K$개의 겹치는 세그먼트로 분할 → 피어슨 상관 기반으로 P-DHG와 N-DHG 구축 |
| **방향성 하이퍼그래프 컨볼루션** | P-DHG와 N-DHG 각각에 대해 하이퍼그래프 컨볼루션을 적용하여 노드 표현 $Z_P$, $Z_N$ 생성 |
| **재구성** | $Z_P$와 $Z_N$을 연결(concatenate)하여 통합 표현 $U$를 생성하고, 완전 연결 층을 통해 입력 윈도우를 재구성 |

노드 특징은 각 변수의 모든 세그먼트 값을 연결하여 $f_i = [s_1^i \| s_2^i \| \ldots \| s_K^i]$로 구성되며, 차원은 $K \times l_s$이다.

### 2.4 성능 향상

**두 개의 실세계 데이터셋에서의 성능 비교 결과**:

| 데이터셋 | 메트릭 | DHG-AD | 2위 방법 | 향상폭 |
|--------|--------|--------|---------|-------|
| **Exathlon** | AUC-F1 ${\text{PA\%K}}$ | **0.7761** | 0.6896 (MTAD-GAT) | +12.5% |
| | AUPRC | **0.6827** | 0.6406 (MSCRED) | +6.6% |
| | AUROC | **0.8538** | 0.8014 (MTAD-GAT) | +6.5% |
| **SMD** | AUC-F1 ${\text{PA\%K}}$ | **0.5685** | 0.5594 (STGAT-MAD) | +1.6% |
| | AUPRC | **0.4103** | 0.3843 (STGAT-MAD) | +6.8% |
| | AUROC | **0.8127** | 0.8077 (OmniAnomaly) | +0.6% |

**Ablation Study 결과**: P-DHG와 N-DHG를 모두 사용한 전체 모델이 어느 한쪽만 사용한 변형보다 일관되게 우수한 성능을 보임.

### 2.5 한계

1. **SMD 데이터셋에서의 제한적 개선**: 28개 독립 머신, 짧은 이상 구간(평균 90 시점), 높은 확률적 변동성으로 인해 성능 개선 폭이 상대적으로 작음
2. **하이퍼파라미터 민감도**: 상관 임계값 $\tau_P$, $\tau_N$에 대한 민감도가 일부 시계열에서 관찰됨 (예: Exathlon의 app6, app9)
3. **두 개의 데이터셋만으로 실험**: 일반화 성능에 대한 보다 광범위한 검증이 필요
4. **Pearson 상관만 사용**: 비선형 관계를 포착하기 어려울 수 있음
5. **시간적 의존성 모델링의 한계**: RNN/Transformer 등의 시간적 모델링 없이 세그먼트 기반 상관만으로 시간적 패턴을 포착

---

## 3. 모델의 일반화 성능 향상 가능성

논문에서 제시하고 있는 일반화 성능 향상 방향과 추가적으로 고려할 수 있는 전략을 정리한다.

### 3.1 논문에서 제안한 일반화 향상 방향

#### (1) 하이퍼엣지 수준 증강(Hyperedge-level Augmentation)
Wei et al. (2022)에서 제안된 하이퍼그래프 대조 학습의 증강 전략을 P-DHG와 N-DHG에 적용할 수 있다. 하이퍼아크를 무작위로 드롭, 교란, 합성함으로써 모델이 **미세한 구조적 노이즈에 불변(invariant)인 표현**을 학습하도록 유도하여 일반화와 강건성을 향상시킬 수 있다. 이는 결합 행렬(incidence matrices)을 확률적으로 수정하는 것만으로 구현 가능하다.

#### (2) 교차 뷰 일관성 정규화(Cross-view Consistency Regularization)
Xia et al. (2022)에 영감을 받아, P-DHG와 N-DHG에서 생성된 표현 간의 정렬을 위한 경량 대조 손실(contrastive loss)을 추가할 수 있다. 이를 통해 양의 상관과 음의 상관 변수-그룹 관계를 더 일관되게 통합하여 통합 노드 표현을 날카롭게 하고, 편차를 더 명확하게 만들 수 있다.

### 3.2 추가적으로 고려 가능한 일반화 전략

| 전략 | 설명 | 기대 효과 |
|-----|------|---------|
| **적응적 임계값 학습** | $\tau_P$, $\tau_N$을 고정값 대신 데이터 기반으로 학습하거나 적응적으로 조정 | 데이터셋별 수동 튜닝 없이 일반화 |
| **비선형 상관 측정** | Pearson 상관 대신 mutual information, distance correlation 등 비선형 의존성 측정 | 비선형 변수-그룹 관계 포착 |
| **시간적 모델링 통합** | GRU/LSTM/Transformer 등과 하이퍼그래프 컨볼루션 결합 | 시간적 의존성과 변수-그룹 관계의 동시 모델링 |
| **다중 스케일 세그먼트** | 다양한 길이의 세그먼트를 사용한 멀티스케일 하이퍼그래프 구축 | 다양한 시간 해상도에서의 관계 포착 |
| **전이 학습/도메인 적응** | 한 도메인에서 학습한 하이퍼그래프 구조를 다른 도메인으로 전이 | 새로운 시스템에 대한 빠른 적응 |

### 3.3 파라미터 민감도와 일반화의 관계

논문의 실험 결과에 따르면:
- $\tau_P \in [0.7, 0.9]$, $\tau_N \in [-0.7, -0.5]$ 범위가 일반적으로 적절
- 높은 상관 임계값은 약한 상관(노이즈)을 필터링하여 탐지 정확도 향상에 기여
- 그러나 일부 시계열(app6, app9)은 $\tau_P$에 민감하여, **데이터 특성에 따른 적응적 임계값 설정**이 일반화에 중요

---

## 4. 논문이 향후 연구에 미치는 영향 및 고려할 점

### 4.1 향후 연구에 미치는 영향

1. **변수-그룹 관계 모델링의 새로운 패러다임**: GNN 기반 방법이 쌍별 관계에 국한되었던 것에서 벗어나, 하이퍼그래프를 통해 고차(higher-order) 관계를 모델링하는 방향을 제시
2. **이중 하이퍼그래프 접근법의 확장 가능성**: 양/음 상관 외에도 시차 상관(lagged correlation), 비선형 의존성 등 다양한 관계 유형으로 확장 가능
3. **방향성 하이퍼그래프의 시계열 분석 적용**: 방향성 하이퍼그래프가 시계열 분석에서 인과적/방향적 변수 관계를 효과적으로 표현할 수 있음을 보여줌
4. **평가 방법론에 대한 기여**: PA%K와 같은 엄격한 평가 지표를 채택하여 과대평가 문제를 완화

### 4.2 향후 연구 시 고려할 점

| 고려 사항 | 상세 설명 |
|---------|---------|
| **확장된 벤치마크** | SWaT, WADI, PSM, MSL, SMAP 등 더 다양한 도메인의 데이터셋에서의 검증 필요 |
| **계산 복잡도** | 하이퍼그래프 구축 시 $O(n^2 \cdot K)$의 상관 계산이 필요하며, 변수 수가 매우 많은 경우의 확장성 분석 필요 |
| **동적 하이퍼그래프** | 현재 윈도우별로 하이퍼그래프를 재구축하지만, 시간에 따른 하이퍼그래프 구조의 변화를 명시적으로 모델링하는 방법 고려 |
| **해석 가능성** | 하이퍼아크를 통해 어떤 변수 그룹이 이상에 기여했는지 해석하는 방법론 개발 |
| **온라인/스트리밍 적용** | 실시간 시스템에서의 효율적인 하이퍼그래프 업데이트 및 추론 전략 |
| **노이즈 강건성** | 센서 노이즈, 결측값 등 현실적 데이터 품질 문제에 대한 강건성 평가 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 방법 | 관계 모델링 | DHG-AD와의 비교 |
|-----|------|---------|-----------|--------------|
| **MTAD-GAT** (Zhao et al.) | 2020 | Graph Attention Network | 변수-변수 (쌍별) | DHG-AD는 변수-그룹 관계를 포착하여 Exathlon에서 AUC-F1 +12.5% 향상 |
| **GDN** (Deng & Hooi) | 2021 | Graph Deviation Network | 변수-변수 (학습된 그래프) | 그래프 구조 학습으로 관계를 발견하나, 고차 관계는 미모델링 |
| **AnomalyTransformer** (Xu et al.) | 2022 | Transformer + Anomaly Attention | 시간적 관계 중심 | 변수 간 관계 명시적 모델링 부족; DHG-AD 대비 두 데이터셋 모두에서 낮은 성능 |
| **STGAT-MAD** (Zhan et al.) | 2022 | Spatial-Temporal GAT | 변수-변수 (시공간) | 시공간 구조를 포착하나 여전히 쌍별 관계에 한정 |
| **MSCRED** (Zhang et al.) | 2019 | Multi-Scale CNN-RNN + Signature Matrix | 변수-변수 (상관 행렬) | 다중 스케일 상관을 포착하나, 그룹 관계는 미모델링 |
| **PA%K 평가** (Kim et al.) | 2022 | 평가 방법론 | — | DHG-AD가 채택한 엄격한 평가 프레임워크; PA의 과대평가 문제 해결 |
| **Directed HGNN** (Ma et al.) | 2024 | 방향성 하이퍼그래프 표현 학습 | 고차 방향성 관계 | DHG-AD의 핵심 컨볼루션 연산의 이론적 기반 제공 |
| **하이퍼그래프 증강** (Wei et al.) | 2022 | 하이퍼그래프 대조 학습 증강 | — | DHG-AD의 일반화 향상을 위한 미래 연구 방향으로 제안됨 |
| **하이퍼그래프 대조 협업 필터링** (Xia et al.) | 2022 | 교차 뷰 대조 학습 | — | P-DHG/N-DHG 간 교차 뷰 정규화의 이론적 기반 |

### 주요 차별점 요약

기존 GNN 기반 방법들(MTAD-GAT, GDN, STGAT-MAD)은 모두 **그래프 엣지를 통한 쌍별 변수 관계**만 모델링한다. DHG-AD는 **하이퍼아크(hyperarc)를 통해 한 변수와 여러 변수 그룹 간의 관계**를 직접 모델링함으로써, 복잡한 다변량 상호작용을 보다 자연스럽게 포착한다. 특히, 양/음 상관을 분리하여 두 개의 하이퍼그래프를 사용하는 이중 접근법은 기존 연구에서 시도되지 않은 새로운 관점이다.

---

## 참고자료

1. **Ha, T. W., & Kim, M. H. (2025).** "Multivariate Time Series Anomaly Detection Using Directed Hypergraph Neural Networks." *Applied Artificial Intelligence*, 39(1), e2538519. DOI: [10.1080/08839514.2025.2538519](https://doi.org/10.1080/08839514.2025.2538519) — 본 분석의 주요 논문
2. **Zhao, H. et al. (2020).** "Multivariate Time-Series Anomaly Detection via Graph Attention Network." *IEEE ICDM 2020*.
3. **Deng, A., & Hooi, B. (2021).** "Graph Neural Network-Based Anomaly Detection in Multivariate Time Series." *AAAI 2021*.
4. **Xu, J. et al. (2022).** "Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy." *ICLR 2022*.
5. **Zhan, J. et al. (2022).** "STGAT-MAD: Spatial-Temporal Graph Attention Network for Multivariate Time Series Anomaly Detection." *IEEE ICASSP 2022*.
6. **Kim, S. et al. (2022).** "Towards a Rigorous Evaluation of Time-Series Anomaly Detection." *AAAI 2022*.
7. **Ma, Z., Zhao, W., & Yang, Z. (2024).** "Directed Hypergraph Representation Learning for Link Prediction." *AISTATS 2024*.
8. **Wei, T. et al. (2022).** "Augmentations in Hypergraph Contrastive Learning: Fabricated and Generative." *NeurIPS 2022*.
9. **Xia, L. et al. (2022).** "Hypergraph Contrastive Collaborative Filtering." *ACM SIGIR 2022*.
10. **Su, Y. et al. (2019).** "Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network." *ACM SIGKDD 2019*.
11. **Zhang, C. et al. (2019).** "A Deep Neural Network for Unsupervised Anomaly Detection and Diagnosis in Multivariate Time Series Data." *AAAI 2019*.
12. **Gallo, G. et al. (1993).** "Directed Hypergraphs and Applications." *Discrete Applied Mathematics*, 42(2-3), 177–201.
