# Learn Hybrid Prototypes for Multivariate Time Series Anomaly Detection

> **출처:** Shen, Ke-Yuan. "Learn Hybrid Prototypes for Multivariate Time Series Anomaly Detection." Published as a conference paper at ICLR 2025.

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장
기존 재구성(reconstruction) 기반 다변량 시계열 이상 탐지(MTSAD) 모델은 **과잉 일반화(over-generalization)** 문제를 겪으며, 메모리 뱅크를 사용하는 모델(MEMTO 등)도 **시점(point) 프로토타입**만 학습하여 **구간(interval) 이상**과 **주기(periodical) 이상**을 탐지하지 못한다. H-PAD는 **다중 스케일 패치 프로토타입**과 **주기 프로토타입**을 결합한 **하이브리드 프로토타입**을 학습함으로써 이 문제를 해결한다.

### 주요 기여
1. **하이브리드 프로토타입 프레임워크:** 패치 프로토타입(지역 정보)과 주기 프로토타입(전역 정보)을 결합하여 점 이상, 구간 이상, 주기 이상을 모두 탐지
2. **다중 스케일 지역-전역 특징 학습:** 패치를 통한 지역 특징과 주기를 통한 전역 정보를 종합적으로 고려
3. **복합 이상 점수 설계:** 입력 공간의 재구성 오류와 특징 공간에서의 프로토타입 거리를 결합한 이상 점수로 탐지 정확도 향상

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**과잉 일반화(Over-generalization):** 재구성 기반 모델은 정상 시계열만으로 학습되지만, 추론 시 비정상 데이터까지 잘 재구성하여 높은 재구성 오류를 통한 이상 탐지가 불가능해지는 문제가 발생한다.

**기존 메모리 기반 접근의 한계:** MEMTO(Song et al., 2024)는 시점 단위 정상 프로토타입을 메모리 뱅크에 저장하여 과잉 일반화를 완화하려 했으나:
- 시점 프로토타입은 이웃 시점들 간의 **내재적 연관성**을 포착하지 못함
- **구간 이상**(짧은 시간 동안의 급격한 변동)을 식별하지 못함
- **주기 이상**(장기적 주기성 붕괴)을 탐지할 수 없음

### 2.2 제안하는 방법 (H-PAD)

H-PAD는 **다중 스케일 패치 프로토타입 학습 브랜치**와 **주기 프로토타입 학습 브랜치**의 두 가지 경로로 구성된다.

#### (A) 다중 스케일 패치 프로토타입 학습

**평균 풀링(Average Pooling):** 주어진 시계열 $\mathbf{X} \in \mathbb{R}^{L \times C}$를 크기 $z \in \{1, 2, \cdots, m\}$의 패치로 나누고 평균 처리한다:

$$\mathbf{x}_i^z = \frac{1}{z} \sum_{t=t_0}^{i \cdot z} \mathbf{x}_t $$

여기서 $t_0 = (i-1) \cdot z + 1$이며, 새로운 시리즈 $\mathbf{X}^z = \{\mathbf{x}_1^z, \mathbf{x}_2^z, \cdots, \mathbf{x}_{L_z}^z\}$가 생성되고, $L_z = \lceil \frac{L}{z} \rceil$이다.

**인코딩:** Transformer 인코더를 통해 특징 공간으로 임베딩한다:

$$\mathbf{Q}^z = Encoder(\mathbf{X}^z) $$

여기서 $\mathbf{Q}^z \in \mathbb{R}^{L_z \times D}$ ($D > C$)이다.

**프로토타입 업데이트:** 랜덤 초기화된 패치 프로토타입 $\mathbf{B}^z = \{\mathbf{b}_1^z, \mathbf{b}_2^z, \cdots, \mathbf{b}_M^z\} \in \mathbb{R}^{M \times D}$는 유사도 행렬과 게이트 메커니즘을 통해 업데이트된다. 유사도 행렬:

$$v_{ij}^z = \frac{\exp\left(\langle \mathbf{b}_i^z, \mathbf{q}_j^z \rangle / \tau\right)}{\sum_{r=1}^{L_z} \exp\left(\langle \mathbf{b}_i^z, \mathbf{q}_r^z \rangle / \tau\right)} $$

업데이트 게이트 $\psi$를 통한 프로토타입 업데이트:

$$\mathbf{b}_i^z = (\mathbf{1}_D - \psi) \circ \mathbf{b}_i^z + \psi \circ \sum_{j=1}^{L_z} v_{ij}^z \mathbf{q}_j^z $$

$$\psi = \sigma\left(\mathbf{U}_1^z \mathbf{b}_i^z + \mathbf{U}_2^z \sum_{k=1}^{L_z} v_{ik}^z \mathbf{q}_k^z\right) $$

여기서 $\sigma$는 시그모이드 함수, $\mathbf{U}_1^z$와 $\mathbf{U}_2^z$는 학습 가능한 행렬이다.

**쿼리 재구성:** 업데이트된 프로토타입으로 쿼리 시리즈를 재구성한다:

$$\hat{\mathbf{q}}_j^z = \sum_{k=1}^{M} w_{jk}^z \mathbf{b}_k, \quad \text{where} \quad w_{jk}^z = \frac{\exp\left(\langle \mathbf{q}_j^z, \mathbf{b}_k^z \rangle / \tau\right)}{\sum_{r=1}^{M} \exp\left(\langle \mathbf{q}_j^z, \mathbf{b}_r^z \rangle / \tau\right)} $$

다중 스케일 재구성 결과의 통합:

$$\hat{\mathbf{q}}_t = Linear\left(ReLU\left(Linear\left(\hat{\mathbf{q}}_{t_i}^{z_1}, \cdots, \hat{\mathbf{q}}_{t_i}^{z_2}, \cdots, \hat{\mathbf{q}}_{t_i}^{z_m}\right)\right)\right) $$

#### (B) 주기 프로토타입 학습

**주기 분할(Period Division):** FFT를 활용하여 시계열의 주요 주기를 추출한다:

$$\mathbf{A} = Avg(Amp(FFT(\mathbf{X})))$$

$$\{f_1, f_2, \ldots, f_K\} = argTopK(\mathbf{A}), \quad p_i = \left\lceil \frac{L}{f_i} \right\rceil \quad (i = 1, 2, \cdots, K)$$

**주기 프로토타입 업데이트:** 각 변수에 대해 하나의 주기 프로토타입 $\mathbf{b}^p \in \mathbb{R}^D$를 학습한다:

$$\mathbf{b}^p = (\mathbf{1}_D - \psi) \circ \mathbf{b}^p + \psi \circ \sum_{j=1}^{N} v_j^p \mathbf{q}_j^p $$

$$\psi = \sigma\left(\mathbf{U}_1^p \mathbf{b}^p + \mathbf{U}_2^p \sum_{k=1}^{N} v_k^p \mathbf{q}_k^p\right), \quad v_j^p = \frac{\exp\left(\langle \mathbf{b}^p, \mathbf{q}_j^p \rangle / \tau\right)}{\sum_{r=1}^{N} \exp\left(\langle \mathbf{b}^p, \mathbf{q}_r^p \rangle / \tau\right)} $$

**주기 쿼리 재구성:** 변수 간 상관관계를 고려하여:

$$\hat{\mathbf{q}}_i^p = \sum_{j=1}^{C} w_{ij}^p \mathbf{b}_j^p, \quad \text{where} \quad w_{ij}^p = \frac{\exp\left(\langle \mathbf{q}_i^p, \mathbf{b}_j^p \rangle / \tau\right)}{\sum_{n=1}^{C} \exp\left(\langle \mathbf{q}_i^p, \mathbf{b}_n^p \rangle / \tau\right)} $$

#### (C) 최종 재구성 및 손실 함수

최종 재구성: $\hat{\mathbf{X}} = \gamma \hat{\mathbf{X}}_z + (1-\gamma) \hat{\mathbf{X}}_p$

**총 손실 함수:**

$$LOSS = \alpha_1 \mathcal{L}_{rec} + \alpha_2 \mathcal{L}_{ent} + \alpha_3 \mathcal{L}_{prd} $$

여기서:
- **재구성 손실:** $\mathcal{L}_{rec} = \|\mathbf{X} - \hat{\mathbf{X}}\|_F$ (Frobenius 노름)
- **엔트로피 손실** (희소성 제약): $\mathcal{L}\_{ent} = \sum_{z=z_1}^{z_m} \sum_{j=1}^{L_z} \sum_{i=1}^{M} -w_{ji}^z \log(w_{ji}^z)$
- **주기 손실:** $\mathcal{L}\_{prd} = \sum_{m=1}^{K} \sum_{i=1}^{C} \sum_{j=1}^{N} \|\mathbf{b}\_i^{p_m} - \mathbf{q}_{ij}^{p_m}\|_2$

#### (D) 이상 점수

입력 공간 점수:

$$s_r(t) = \|\hat{\mathbf{x}}_t - \mathbf{x}_t\|_2 $$

패치 특징 공간 점수:

$$s_z(t) = \sum_{j=1}^{m} \frac{1}{z_j} \left\|\mathbf{q}\_t - \mathbf{b}_{sim}^{z_j}\right\|_2 $$

주기 특징 공간 점수:

$$s_p(t) = \sum_{k=1}^{K} \sum_{i=1}^{C} \left\|Encoder\left(O_{i,\rho(t)}^{p_k}\right) - \mathbf{b}_i^{p_k}\right\|_2 $$

최종 통합 점수:

$$s(t) = softmax(s_z(t) + \beta s_p(t)) \times s_r(t) $$

### 2.3 모델 구조

H-PAD의 전체 아키텍처는 두 개의 병렬 브랜치로 구성된다 (Figure 2 참조):

| 구성 요소 | 설명 |
|---------|------|
| **좌측 브랜치 (패치 프로토타입)** | 평균 풀링 → Transformer 인코더 → 프로토타입 업데이트 → 다중 스케일 쿼리 재구성 → 디코더 |
| **우측 브랜치 (주기 프로토타입)** | FFT 기반 주기 분할 → Transformer 인코더 → 주기별 프로토타입 업데이트 → 쿼리 재구성 → 디코더 |
| **통합** | $\hat{\mathbf{X}} = \gamma\hat{\mathbf{X}}_z + (1-\gamma)\hat{\mathbf{X}}_p$ |

### 2.4 성능 향상

7개 벤치마크 데이터셋(MSL, SMAP, PSM, SMD, SWaT, NIPS_TS_Water, NIPS_TS_Swan)에서 평가:

- **F1 점수 기준:** 5개 주요 데이터셋 평균 **96.86%**, 모든 데이터셋에서 95% 이상 달성
- **AUC-ROC 기준:** 7개 데이터셋 평균 **72.83%**로 최고 성능
- **AUC-PR 기준:** 7개 데이터셋 평균 **33.12%**로 최고 성능
- MEMTO 대비 F1 평균 1.87%p 향상 (94.99% → 96.86%)
- 어블레이션 실험에서 패치 프로토타입 제거 시 평균 F1이 96.86% → 83.08%로 13.78%p 하락, 주기 프로토타입 제거 시 96.86% → 96.11%로 0.75%p 하락

### 2.5 한계

1. **계산 효율성:** MEMTO 대비 MACs가 약 3~9배, 학습 시간이 약 7~15배, 메모리 사용량이 약 1.3~10배 증가 (Table 6)
2. **하이퍼파라미터 의존성:** $\gamma$, $\beta$, $\alpha_1$, $\alpha_2$, $\alpha_3$, 스케일 수, 주기 수, 프로토타입 수 등 다수의 하이퍼파라미터가 존재하며, 데이터셋별 최적값이 다름
3. **PA 평가의 한계:** F1 점수는 Point Adjustment(PA) 기반으로, PA가 부정확한 평가를 초래할 수 있다는 점을 저자 스스로 인정하고 AUC-ROC/AUC-PR을 추가 보고
4. **특정 데이터셋에서의 성능 제한:** NIPS_TS_Water에서 AUC-PR이 7.30%로 다른 방법(DMamba: 46.32%) 대비 크게 낮음

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 성능 향상을 위한 H-PAD의 설계 전략

**(1) 과잉 일반화 억제 메커니즘:**
- **다중 스케일 패치 프로토타입:** 다양한 크기의 패치에서 정상 패턴을 학습하므로, 단일 이상 시점이 여러 스케일의 정상 프로토타입 가중 합으로 재구성될 때 비정상 정보가 "평활화(smoothed off)"되어 높은 재구성 오류가 발생
- **엔트로피 손실($\mathcal{L}_{ent}$):** 재구성 가중치에 희소성 제약을 부여하여, 가장 관련성 높은 프로토타입만 재구성에 참여하도록 유도 → 과잉 일반화 가능성 감소

**(2) 하이퍼파라미터 강건성:**
- 프로토타입 수에 대한 민감도 분석(Figure 4(a))에서 F1 점수의 변동폭이 작아 **모델의 강건성(robustness)**이 입증됨
- 주기 수(Figure 4(c))에 대해서도 유사한 안정성을 보임

**(3) 다양한 이상 유형 탐지 능력:**
시각화 분석(Figure 11)에서 H-PAD가 점 이상, 맥락 이상, 주기 이상, 집단 이상, 트렌드 이상 등 다양한 유형의 이상을 효과적으로 탐지함을 확인

### 3.2 일반화 성능의 현재 한계 및 개선 방향

**(1) 도메인 간 일반화:**
- 현재 7개 데이터셋에서 검증되었으나, 데이터셋별로 하이퍼파라미터 $(D, z, K, M)$의 최적값이 달라 새로운 도메인에 적용 시 별도의 튜닝이 필요
- **개선 방향:** 자동 하이퍼파라미터 탐색(AutoML), 메타 학습 기반 적응적 파라미터 조정

**(2) 비정상(non-stationary) 시계열 처리:**
- FFT 기반 주기 분할은 정상적(stationary) 주기 구조를 가정하므로, 시간에 따라 주기가 변화하는 시계열에서는 성능 저하 가능
- D3R(Wang et al., 2023)의 동적 분해 접근이나 DMamba(Chen et al., 2024)의 다단계 디트렌딩 메커니즘과의 결합을 고려할 수 있음

**(3) 계산 효율성 대비 일반화 트레이드오프:**
- 다중 스케일과 다중 주기를 동시에 처리하면서 발생하는 높은 계산 비용(Table 6)이 대규모 실시간 시스템에서의 활용을 제한
- 저자들도 "향후 전체 프레임워크를 최적화하여 성능을 유지하면서 학습 시간과 메모리 소비를 줄일 계획"이라고 언급

---

## 4. 향후 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미치는 영향

1. **프로토타입 학습 패러다임의 확장:** 시점 단위 프로토타입에서 다중 스케일·주기 프로토타입으로의 확장은 메모리 기반 이상 탐지 연구의 새로운 방향을 제시
2. **다중 유형 이상 탐지의 통합 프레임워크:** 점 이상, 구간 이상, 주기 이상을 하나의 프레임워크에서 처리하는 접근법으로, 향후 더 다양한 이상 유형을 포함하는 연구의 토대
3. **특징 공간과 입력 공간의 이중 이상 점수 설계:** 재구성 오류뿐만 아니라 특징 공간에서의 프로토타입 거리를 결합하는 점수 체계는 보다 정교한 이상 탐지 기준 설계에 영감

### 4.2 향후 연구 시 고려할 점

1. **효율성 개선:** 경량화된 인코더 구조(예: 선형 어텐션), 프로토타입 수의 적응적 결정, 계층적 처리 등을 통한 계산 비용 절감
2. **적응적 주기 탐지:** 정적 FFT 대신 시간 변화에 적응하는 주기 추출 메커니즘 필요
3. **공정한 평가 프로토콜:** PA 기반 F1 외에 AUC-ROC, AUC-PR, Affiliation 메트릭 등 다양한 평가 기준의 병행 사용 필요
4. **변수 간 상관관계 모델링 강화:** 현재 주기 프로토타입에서 변수 간 상관관계를 고려하지만, 그래프 신경망(GNN) 등과의 결합으로 더 정교한 변수 간 의존성 모델링 가능
5. **온라인/스트리밍 환경 적용:** 현재 오프라인 배치 학습 방식에서 실시간 스트리밍 환경으로의 확장 연구 필요

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 모델 | 연도 | 핵심 접근법 | H-PAD와의 비교 |
|------|------|-----------|-------------|
| **THOC** (Shen et al.) | 2020 | 시간적 계층 원-클래스 네트워크 | 계층적 시간 표현을 학습하나, 프로토타입 기반 재구성 없음. H-PAD 대비 평균 F1 약 8.9%p 낮음 (88.01% vs 96.86%) |
| **Anomaly Transformer** (Xu et al.) | 2022 | Association discrepancy 기반 정상-비정상 상관 차이 학습 | 어텐션 기반이나 메모리 프로토타입 부재. 과잉 일반화 문제 잔존 (Figure 5(a)). 평균 F1: 94.09% |
| **DCdetector** (Yang et al.) | 2023 | 이중 어텐션 대조 학습 + 패치 기반 지역 정보 | 패치 학습을 사용하나 프로토타입 메모리 없음. F1은 유사(94.37%)하나 AUC-ROC/PR에서 H-PAD가 우수 |
| **D3R** (Wang et al.) | 2023 | 동적 분해 + 확산 재구성으로 비정상 시계열 처리 | 비정상 시계열에 강점이 있으나, 평균 F1(91.73%)에서 H-PAD보다 낮음 |
| **MEMTO** (Song et al.) | 2024 | 메모리 가이드 트랜스포머 + 시점 프로토타입 | H-PAD의 직접적 선행 연구. 시점 프로토타입만 사용하여 구간/주기 이상 탐지 한계. 평균 F1: 94.99% |
| **DMamba** (Chen et al.) | 2024 | 선택적 상태공간 모델 + 다단계 디트렌딩 | 장거리 의존성에 강점이나, 평균 F1(78.51%)이 매우 낮음. 다만 AUC 기준 NIPS_TS 데이터셋에서 우수 |
| **GSC_MAD** (Zhang et al.) | 2024 | 그래프 구조 변화 기반 이상 탐지 | 변수 간 상관관계 모델링에 강점. 평균 F1(95.10%)로 H-PAD(96.86%)에 근접 |

### 주요 비교 인사이트

- **메모리/프로토타입 기반 접근:** MemAE(2019) → MNAD(2020) → MEMTO(2024) → **H-PAD(2025)**로 이어지는 진화 과정에서, H-PAD는 프로토타입의 단위를 시점에서 패치+주기로 확장한 점이 핵심 혁신
- **평가 메트릭의 중요성:** PA 기반 F1에서는 여러 모델이 유사한 성능을 보이나, AUC-ROC/PR에서는 상당한 차이가 존재. H-PAD는 AUC 평균에서 최고 성능을 달성하여 보다 공정한 평가에서의 우위를 보임
- **효율성 vs 성능 트레이드오프:** MEMTO 대비 높은 계산 비용이 가장 큰 단점으로, 향후 모델 경량화가 실용적 적용의 관건

---

### 참고자료

1. Shen, K.-Y. "Learn Hybrid Prototypes for Multivariate Time Series Anomaly Detection." ICLR 2025.
2. Song, J., Kim, K., Oh, J., & Cho, S. "MEMTO: Memory-guided Transformer for Multivariate Time Series Anomaly Detection." NeurIPS 2024.
3. Xu, J., Wu, H., Wang, J., & Long, M. "Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy." ICLR 2022.
4. Yang, Y., Zhang, C., Zhou, T., Wen, Q., & Sun, L. "DCdetector: Dual Attention Contrastive Representation Learning for Time Series Anomaly Detection." KDD 2023.
5. Wang, C., et al. "D3R: Drift Doesn't Matter: Dynamic Decomposition with Diffusion Reconstruction for Unstable Multivariate Time Series Anomaly Detection." NeurIPS 2023.
6. Chen, J., et al. "Joint Selective State Space Model and Detrending for Robust Time Series Anomaly Detection." IEEE Signal Processing Letters, 2024.
7. Zhang, Z., Geng, Z., & Han, Y. "Graph Structure Change-Based Anomaly Detection in Multivariate Time Series of Industrial Processes." IEEE TII, 2024.
8. Gong, D., et al. "Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder for Unsupervised Anomaly Detection." ICCV 2019.
9. Park, H., Noh, J., & Ham, B. "Learning Memory-Guided Normality for Anomaly Detection." CVPR 2020.
10. Wu, H., et al. "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis." ICLR 2023.
