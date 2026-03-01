# TSA-Net: Multivariate Time Series Anomaly Detection Based on Two-Stage Temporal Attention 

---

## 1. 핵심 주장 및 주요 기여 요약

TSA-Net은 산업용 다변량 시계열(MTS) 이상 탐지를 위한 **경량 2단계 시공간 어텐션 프레임워크**이다. 기존 방법들이 높은 훈련 비용과 느린 수렴 속도로 인해 동적 산업 환경에서의 빈번한 재훈련에 부적합한 문제를 해결하고자 한다.

### 주요 기여 (3가지)

| 기여 | 내용 |
|------|------|
| **① RepVGG-TCN + GAT 시공간 특징 추출기** | 구조적 재매개변수화(Structural Reparameterization)를 통해 훈련 시 다중 분기 구조로 풍부한 특징 추출, 배포 시 단일 분기로 압축하여 효율성 확보 |
| **② 캐스케이드 피드백 메커니즘** | 1단계의 초기 예측을 사전 지식(Prior Knowledge)으로 2단계 입력에 주입하여 Coarse-to-Fine 반복 정제 수행 |
| **③ 적응형 게이트 융합 전략** | 시간적·공간적 특징의 중요도를 동적으로 가중하여 복합 이상 탐지 성능 향상 |

**핵심 결과**: F1 점수 약 **7% 향상**, 훈련 시간 최대 **99% 감소** (Transformer 기반 모델 대비).

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

1. **높은 훈련 비용 및 느린 수렴**: Transformer 기반 모델(TranAD, EST Transformer 등)은 self-attention의 $O(n^2)$ 복잡도로 인해 대규모 산업 데이터에서 재훈련이 비현실적
2. **시공간 의존성의 불균형 모델링**: 시간 중심 모델(LSTM, TCN)은 변수 간 공간적 상관관계를 무시하고, GNN 기반 모델(D-GATAD 등)은 동적 그래프 갱신으로 추가 계산 비용 발생
3. **느슨한 다단계 결합**: 기존 2단계 방법(DTAAD, MAD-STA 등)은 단계 간 정보를 명시적으로 피드백하지 않아 반복적 오류 수정이 불가

### 2.2 제안 방법 (수식 포함)

#### (A) 문제 정의

다변량 시계열 $\mathcal{X} = \{\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_T\}$, $\mathbf{x}_t \in \mathbb{R}^m$에 대해 각 시점 $t$의 이상 점수 $S_t$를 계산하고, 임계값 $\tau$와 비교하여 이상 레이블을 결정:

```math
y_t = \begin{cases} 1, & S_t > \tau \\ 0, & S_t \leq \tau \end{cases}
```

#### (B) RepVGG-TCN (시간적 특징 추출)

**인과 패딩(Causal Padding)**으로 시간적 인과성 보장:

$$P = (K - 1) \times d $$

여기서 $K$는 커널 크기, $d$는 확장 인자(dilation factor)이다.

**총 수용 영역(Total Receptive Field)**:

$$1 + (K - 1) \times (2^L - 1) \geq R $$

$L$은 레이어 수, $R$은 입력 시퀀스 길이. 이 조건으로 최소 3개 레이어 필요.

- **Local TCN**: 확장 인자 $d = 1$ (세밀한 국소 패턴)
- **Global TCN**: 지수적 확장 $d = 2^i$ (장거리 의존성)

**핵심 혁신 — 구조적 재매개변수화**: 훈련 시 $3 \times 1$ Conv + $1 \times 1$ Conv + Identity의 다중 분기 구조를 사용하고, 추론 시 단일 $3 \times 1$ Conv로 수학적으로 병합하여 메모리 접근 비용과 지연 시간을 최소화.

#### (C) GAT (공간적 특징 추출)

완전 연결 속성 그래프 $G = (V, E)$에서 노드 $i$와 이웃 $j$ 간 어텐션 계수:

$$e_{ij} = \text{LeakyReLU}\left(\mathbf{a}^T[\mathbf{W}\mathbf{h}_i \| \mathbf{W}\mathbf{h}_j]\right) $$

정규화된 어텐션 가중치:

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}_i} \exp(e_{ik})} $$

각 헤드의 노드 특징 갱신:

$$\mathbf{h}'_i = \text{ReLU}\left(\sum_{j \in \mathcal{N}_i} \alpha_{ij} \mathbf{W}\mathbf{h}_j\right) $$

$K$개 헤드의 멀티헤드 결합:

$$H_i = \text{Concat}\left(\mathbf{h}'^{(1)}_i, \mathbf{h}'^{(2)}_i, \ldots, \mathbf{h}'^{(K)}_i\right) $$

완전 연결 그래프를 사용하되, 어텐션 메커니즘이 **적응적 소프트 희소화(soft-sparsification)** 필터 역할을 수행하여 무관한 센서 쌍의 $\alpha_{ij} \approx 0$으로 노이즈 전파를 억제.

#### (D) 적응형 게이트 융합(Adaptive Gated Fusion)

시간적 특징 $\mathbf{T}$와 공간적 특징 $\mathbf{G}$를 채널 차원으로 결합:

$$\mathbf{C} = \text{Concat}([\mathbf{T}, \mathbf{G}], \text{dim}=1) \in \mathbb{R}^{B \times 2F \times W} $$

게이팅 가중치 생성:

$$\mathbf{A} = \text{Sigmoid}(\text{Conv1d}(\mathbf{C})) \in \mathbb{R}^{B \times F \times W} $$

최종 융합 특징:

$$\mathbf{F}' = \mathbf{A} \odot \mathbf{T} + (1 - \mathbf{A}) \odot \mathbf{G} $$

여기서 $\odot$은 원소별 곱셈. 각 시점에서 시간적/공간적 정보의 중요도를 동적으로 조절.

#### (E) Encoder-Decoder

**적응형 멀티헤드 셀프 어텐션**: 헤드 수 $h$를 입력 차원 $D$의 최대 약수(1 제외)로 동적 설정.

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

**수정된 FFN** (LeakyReLU 활성화):

$$\text{FFN}(x) = \text{LeakyReLU}(xW_1 + b_1)W_2 + b_2 $$

#### (F) 캐스케이드 피드백 메커니즘

1단계 예측 $\hat{\mathbf{X}}^{(1)}$을 원래 입력 $\mathbf{X}$에 사전 지식으로 주입:

$$\mathbf{X}' = \mathbf{X} + \hat{\mathbf{X}}^{(1)}_{\text{reshaped}} = \mathbf{X} + \text{permute}(\hat{\mathbf{X}}^{(1)}, (0, 2, 1)) $$

2단계는 증강된 입력 $\mathbf{X}'$를 동일 구조로 처리하여 최종 예측 $\hat{\mathbf{X}}^{(2)}$ 생성. 이는 잔차(residual)가 아닌 **예측 자체를 피드백**하여 노이즈 유입 방지.

#### (G) 목적 함수

두 단계의 MSE 손실을 가중 결합:

$$L(\Theta_1, \Theta_2) = \lambda L_1 + (1 - \lambda) L_2 $$

$\lambda = 0.8$으로 설정하여 1단계에 높은 가중치를 부여, 안정적인 조대(coarse-grained) 기반 학습 보장.

#### (H) 이상 점수 및 임계값

시스템 수준 이상 점수 (평균 집계):

$$S_t = \frac{1}{m} \sum_{j=1}^{m} s_{t,j} $$

**Peaks-Over-Threshold (POT)** 방법으로 적응적 임계값 $\text{thr}_{\text{POT}}$ 결정:

$$y_t = \begin{cases} 1 & \text{if } S_t \geq \text{thr}_{\text{POT}} \\ 0 & \text{otherwise} \end{cases} $$

### 2.3 모델 구조 요약

```
[입력 MTS] ──→ Stage 1 ┬── Local TCN (d=1, RepVGG) ──┐
                        └── GAT ─────────────────────┘
                              │
                        Adaptive Gated Fusion → Encoder-Decoder → X̂⁽¹⁾
                              │                                      │
                              │      ┌──── Feedback (X' = X + X̂⁽¹⁾) ┘
                              │      ▼
               Stage 2 ┬── Global TCN (d=2ⁱ, RepVGG) ──┐
                        └── GAT ────────────────────────┘
                              │
                        Adaptive Gated Fusion → Encoder-Decoder → X̂⁽²⁾
                              │
                        Loss = λL₁ + (1-λ)L₂
```

### 2.4 성능 향상

#### 탐지 성능 (Table 2)

| 데이터셋 | TSA-Net F1 | 최고 비교 기준선 F1 | 향상 |
|---------|-----------|---------------|------|
| **SMD** | **0.9782** | USAD: 0.9493 | +2.89%p |
| **SMAP** | **0.9297** | TranAD: 0.8950 | +3.47%p |
| MSL | 0.9389 | GDN: 0.9588 | −1.99%p |

AUC는 모든 데이터셋에서 0.986–0.995 범위의 높은 값 유지.

#### 계산 효율성 (Table 3)

| 데이터셋 | TSA-Net 수렴 시간(s) | MTAD-GAT(s) | TranAD(s) | 대비 속도 |
|---------|------------------|-------------|-----------|---------|
| SMD | **545** | 45,600 | 905 | ~80× vs MTAD-GAT |
| SMAP | **59** | 4,713 | 628 | ~80× vs MTAD-GAT |
| MSL | **38** | 5,563 | 520 | ~146× vs MTAD-GAT |

#### 추론 효율성 (Table 4)

구조적 재매개변수화 후 TCN 모듈 지연: **0.79ms → 0.34ms (2.31× 가속)**. 전체 시스템: **4.35ms/샘플**, ~230 samples/s 처리량으로 산업 SCADA 시스템(1–100Hz) 요구 충족.

### 2.5 한계

논문에서 명시적으로 인정한 한계 및 분석에서 도출된 한계:

1. **완전 연결 그래프의 계산 병목**: 공간 모듈에서 $O(N^2)$ 복잡도의 완전 그래프 구성은 초고차원 환경에서 추론 처리량을 제약 (논문 Section 5에서 명시)
2. **MSL 데이터셋에서의 차선 성능**: GDN(0.9588) 대비 F1이 약 2%p 낮음 — 55차원의 고차원 환경에서 완전 연결 그래프의 노이즈 전파가 영향을 미쳤을 가능성
3. **제한된 데이터셋 다양성**: SMD, SMAP, MSL 3개 벤치마크에서만 검증 — 의료, 금융, 교통 등 다른 도메인 데이터로의 일반화는 미검증
4. **정적 하이퍼파라미터 $\lambda = 0.8$**: 두 단계 간 손실 가중치가 고정되어 데이터 특성에 따른 적응이 불가
5. **비정상(non-stationary) 시계열 처리**: 개념 드리프트(concept drift)에 대한 명시적 대응 전략 부재

---

## 3. 모델의 일반화 성능 향상 가능성 (중점 분석)

TSA-Net의 아키텍처는 일반화 성능 향상과 관련하여 여러 설계적 장점과 개선 여지를 동시에 갖고 있다.

### 3.1 일반화를 촉진하는 현재 설계 요소

**(1) 구조적 재매개변수화(Structural Reparameterization)의 암묵적 정규화 효과**

훈련 시 다중 분기 구조($3\times1$ Conv, $1\times1$ Conv, Identity)는 서로 다른 스케일의 특징을 병렬로 학습하여 **다양한 특징 부분공간(feature subspace)**을 탐색한다. 이는 단일 분기 대비 경사 전파 경로를 다양화하여 **과적합 방지** 효과를 가진다. Ablation 실험(Table 5)에서 `w/o RepVGG`가 모든 데이터셋에서 성능 하락(SMD: 0.9655, SMAP: 0.8812, MSL: 0.8530)을 보인 것이 이를 뒷받침한다.

**(2) 적응형 게이트 융합의 도메인 독립적 설계**

수식 (10)의 게이팅 메커니즘은 시간적·공간적 특징의 상대적 중요도를 데이터 기반으로 자동 조정한다:

$$\mathbf{F}' = \mathbf{A} \odot \mathbf{T} + (1 - \mathbf{A}) \odot \mathbf{G}$$

이 메커니즘은 도메인에 특화된 사전 지식 없이도 **데이터의 내재적 특성에 맞게 적응**할 수 있어, 새로운 도메인(의료 IoT, 스마트 그리드 등)으로의 이전 시 별도의 구조 변경 없이 적용 가능하다. 시각화 분석(Figure 7)에서 정상 구간과 이상 구간에서 게이팅 가중치가 극적으로 변화하는 것은 이 메커니즘의 적응력을 실증적으로 보여준다.

**(3) 캐스케이드 피드백의 반복 정제**

수식 (13)의 사전 지식 주입은 2단계 네트워크가 처음부터 패턴을 학습하는 부담을 줄이고, 1단계가 포착하지 못한 **미세 이상(subtle anomaly)**에 집중하게 한다. 이는 본질적으로 **부스팅(boosting)**과 유사한 원리로, 서로 다른 난이도의 이상 패턴에 대한 일반화 능력을 향상시킨다.

**(4) 적응형 멀티헤드 전략**

고정된 헤드 수 대신 입력 차원 $D$에 따라 동적으로 $h$를 설정하는 전략은 **다양한 차원의 데이터셋에 자동 적응**할 수 있게 한다. Table 6에서 제안 전략($h=19$, Test MSE = 0.0025)이 단일 헤드($h=1$, MSE = 0.0030) 및 완전 분할($h=38$, MSE = 0.0031) 모두를 능가함을 확인.

**(5) 가우시안 노이즈 증강 및 Dropout**

전처리 단계에서 가우시안 화이트 노이즈를 추가하고, TCN(Dropout 0.2) 및 GAT(Dropout 0.2)에서 드롭아웃을 적용하여 모델의 **강건성(robustness)** 및 일반화를 촉진.

### 3.2 일반화 성능 향상을 위한 개선 방향

**(1) 희소 그래프 학습(Sparse Graph Learning)으로의 전환**

현재 완전 연결 그래프 $G = (V, E)$는 $O(N^2)$ 복잡도를 가지며, 초고차원 환경(수백~수천 센서)에서 노이즈 전파와 계산 비용 문제를 야기한다. 향후 **학습 가능한 희소 인접 행렬(learnable sparse adjacency matrix)** 또는 **그래프 샘플링(graph sampling)** 기법을 도입하면:
- 무관한 센서 쌍의 연결을 명시적으로 제거하여 노이즈 강건성 향상
- MSL과 같은 고차원 데이터셋에서의 일반화 성능 개선 기대

**(2) 도메인 적응 및 전이 학습(Transfer Learning) 통합**

논문에서 인용한 Wu et al. [25]의 교차 도메인 지식 증류(cross-domain knowledge distillation) 접근법을 TSA-Net의 경량 백본과 결합하면, **소스 도메인에서 학습한 시공간 패턴을 타겟 도메인으로 효과적으로 이전**할 수 있다. RepVGG의 구조적 재매개변수화는 이 과정에서 추론 비용 증가 없이 풍부한 사전 훈련 표현을 유지하는 이점을 제공.

**(3) 적응적 손실 가중치 $\lambda$**

현재 고정된 $\lambda = 0.8$ 대신, 훈련 과정에서 두 단계의 손실 비율을 동적으로 조정하는 **불확실성 기반 가중치(uncertainty-based weighting)** 또는 **GradNorm** 전략을 적용하면, 다양한 데이터셋 특성에 자동 적응할 수 있다.

**(4) 개념 드리프트(Concept Drift) 대응**

산업 시스템의 운영 조건은 시간에 따라 변화한다. **온라인 학습(online learning)** 또는 **지속적 학습(continual learning)** 메커니즘을 통합하여, 데이터 분포 변화에 대한 모델의 적응력을 강화할 수 있다. TSA-Net의 빠른 수렴 속도(SMAP: 59초, MSL: 38초)는 주기적 재훈련을 현실적으로 가능하게 하는 장점.

**(5) 다양한 도메인 벤치마크로의 확장 검증**

현재 우주(SMAP/MSL)와 서버(SMD)에 한정된 실험을 **SWaT/WADI(산업 제어), HAI(전력), Yahoo/KPI(IT 운영)** 등으로 확장하여 일반화 성능을 입증할 필요가 있다.

---

## 4. 향후 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미치는 영향

**(1) "훈련-추론 분리(Train-Infer Decoupling)" 패러다임의 시계열 분야 확산**

TSA-Net은 컴퓨터 비전에서 성공한 **구조적 재매개변수화(RepVGG)**를 시계열 이상 탐지에 최초로 적용한 사례로, 이 기법이 다른 시계열 태스크(예측, 분류)로도 확산될 가능성을 시사한다. 특히 **엣지 디바이스 배포**가 필수적인 산업 IoT 시나리오에서 이 패러다임은 핵심 설계 원칙이 될 수 있다.

**(2) 캐스케이드 피드백의 방법론적 기여**

기존 2단계 방법(TranAD의 적대적 훈련, DTAAD의 병렬 분기)과 달리, TSA-Net의 **직렬 캐스케이드 + 사전 지식 주입** 방식은 다단계 이상 탐지의 새로운 설계 방향을 제시한다. 이는 객체 탐지의 Cascade R-CNN과 유사한 "coarse-to-fine" 원리를 시계열에 적용한 것으로, 후속 연구에서 3단계 이상의 다단계 정제로 확장될 수 있다.

**(3) 효율성과 정확도의 새로운 Pareto 최적 기준 설정**

F1 ~7% 향상과 훈련 시간 99% 감소를 동시에 달성함으로써, 후속 연구가 달성해야 할 **효율성-정확도 트레이드오프의 새로운 기준선(baseline)**을 제시.

### 4.2 향후 연구 시 고려할 점

| 고려 사항 | 세부 내용 |
|----------|----------|
| **평가 프로토콜의 공정성** | Point-Adjust(PA) 등 논란이 있는 평가 메트릭 대신, 범위 기반(range-based) F1 또는 VUS(Volume Under the Surface) 등 최신 메트릭 병행 필요 |
| **해석가능성(Interpretability)** | GAT 어텐션 가중치의 시각화(Figure 7)가 초보적 해석을 제공하나, Tang et al. [26]의 gradient-based 해석이나 SHAP 통합으로 이상 원인의 변수 수준 귀인(attribution) 강화 필요 |
| **확장성(Scalability)** | $N > 100$ 센서 환경에서 완전 연결 GAT의 $O(N^2)$ 복잡도 해결 → 희소 그래프 학습, GraphSAGE 스타일 샘플링 고려 |
| **비정상 시계열** | 개념 드리프트 및 분포 이동에 대한 온라인 적응 전략 통합 |
| **다중 도메인 벤치마크** | 우주/서버 데이터 외에 SWaT, WADI, HAI, PSM 등 산업 제어 데이터셋으로 일반화 검증 |
| **비지도 학습의 한계** | 이상 레이블의 품질에 민감한 POT 임계값 결정의 한계 → 자기 지도 학습(self-supervised) 기반 대안 탐색 |
| **다변량 이상의 유형 구분** | 점 이상(point anomaly) vs. 문맥 이상(contextual anomaly) vs. 집단 이상(collective anomaly)에 대한 차별화된 탐지 성능 분석 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

아래 표는 TSA-Net과 2020년 이후 발표된 주요 MTS 이상 탐지 방법들을 체계적으로 비교한다.

| 방법 | 연도 | 핵심 아키텍처 | 시공간 모델링 | 다단계 설계 | 효율성 전략 | TSA-Net 대비 차이점 |
|------|------|------------|------------|----------|----------|-----------------|
| **USAD** [40] | 2020 | 이중 AE (적대적) | 시간만 | 단일 단계 | AE 경량 구조 | 공간 모델링 부재; TSA-Net이 SMD에서 F1 +2.89%p |
| **GDN** [41] | 2021 | GNN + 예측 | 학습 그래프 | 단일 단계 | Attention 기반 | MSL에서 GDN이 F1 우세(0.9588 vs 0.9389); 단, 수렴 시간 20× 느림 |
| **TranAD** [13] | 2022 | Transformer (적대적) | 시간만 (self-attention) | 2단계 (적대적 증폭) | Focus score | Self-attention $O(n^2)$ 비용; TSA-Net 대비 SMD 수렴 1.66× 느림 |
| **CST-GL** [22] | 2023 | 상관 인식 시공간 그래프 | 동적 그래프 | 단일 단계 | - | 세밀한 그래프 융합 but 높은 계산 비용 |
| **MST-GAT** [16] | 2023 | GAT + TCN | 정적 그래프 + TCN | 단일 단계 | - | 정적 그래프 사전 의존; RepVGG 같은 배포 최적화 없음 |
| **DTAAD** [3] | 2024 | 이중 TCN + Transformer | 시간만 (병렬 분기) | 2단계 (병렬) | 경량 TCN | **병렬** 독립 분기 → 초기 예측 오류의 반복 수정 불가; TSA-Net은 **직렬** 캐스케이드로 이를 해결 |
| **SiET** [18] | 2024 | Spatial-enhanced Transformer | 공간 증강 어텐션 | 단일 단계 | - | 무거운 멀티헤드 어텐션; 서브스페이스 붕괴 위험 |
| **EST Transformer** [19] | 2025 | 시공간 Transformer | Self-attention 기반 | 단일 단계 | - | 높은 계산 오버헤드; 산업 실시간 배포 부적합 |
| **D-GATAD** [21] | 2025 | 동적 그래프 + GAT | 매 시점 그래프 갱신 | 단일 단계 | - | 매 시점 $O(N^2)$ 그래프 재구성 → 실시간 부적합 |
| **GSTA-DeSVDD** [28] | 2025 | GAT + GAN + DeepSVDD | 그래프 기반 | 다단계 | - | 느슨한 단계 결합; TSA-Net의 피드백 메커니즘 부재 |
| **Res2Coder** [34] | 2025 | 2단계 잔차 AE | 시간만 | 2단계 (잔차) | AE 기반 | 잔차 피드백은 노이즈 유입 위험; TSA-Net은 예측 자체를 피드백 |
| **TSA-Net (본 논문)** | 2026 | RepVGG-TCN + GAT + Transformer | 이중 분기 (TCN + GAT) | 2단계 (직렬 캐스케이드) | 구조적 재매개변수화 + 적응형 헤드 | **유일하게** 훈련-추론 아키텍처 분리와 직렬 사전 지식 주입을 결합 |

### 주요 차별화 분석

**TSA-Net vs. TranAD**: 두 모델 모두 2단계 구조를 사용하지만, TranAD는 적대적 훈련을 통해 이상 신호를 증폭하는 반면, TSA-Net은 사전 지식 주입으로 반복 정제한다. TranAD의 self-attention은 $O(T^2)$ 복잡도이나, TSA-Net의 RepVGG-TCN은 $O(T)$로 선형적.

**TSA-Net vs. DTAAD**: DTAAD의 병렬 2분기 TCN은 독립적으로 특징을 추출한 후 최종 단계에서만 융합하여, 1단계 오류를 2단계가 수정할 수 없다. TSA-Net의 직렬 캐스케이드는 1단계 예측을 2단계 입력에 명시적으로 주입하여 이 문제를 해결.

**TSA-Net vs. D-GATAD**: D-GATAD는 매 시간 단계마다 전역 정보 기반으로 그래프 구조를 갱신하여 동적 의존성을 포착하지만, 이로 인한 계산 오버헤드가 실시간 배포를 어렵게 한다. TSA-Net은 완전 연결 그래프 + 소프트 희소화로 유사한 적응 효과를 훨씬 낮은 비용에 달성.

---

## 참고 자료

본 분석은 다음 자료를 기반으로 작성되었습니다:

1. **주 논문**: Wu, H., Le, W., Jia, Z.-H., Zhao, H., Zhang, S., & Zhang, Z.-S. (2026). "TSA-Net: Multivariate Time Series Anomaly Detection Based on Two-Stage Temporal Attention." *Sensors*, 26(3), 1062. https://doi.org/10.3390/s26031062
2. Ding, X. et al. (2021). "RepVGG: Making VGG-style ConvNets Great Again." *IEEE/CVF CVPR*, pp. 13733–13742.
3. Yu, L.R. et al. (2024). "DTAAD: Dual TCN-Attention Networks for Anomaly Detection in Multivariate Time Series Data." *Knowledge-Based Systems*, 295, 111849.
4. Tuli, S. et al. (2022). "TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data." *Proc. VLDB Endow.*, 15(11), 2568–2581.
5. Zhao, H. et al. (2020). "Multivariate Time-series Anomaly Detection via Graph Attention Network." *IEEE ICDM 2020*, pp. 841–850.
6. Gao, C. et al. (2025). "Dynamic Graph-based Graph Attention Network for Anomaly Detection in Industrial Multivariate Time Series Data." *Applied Intelligence*, 55, 517.
7. Gao, Y. et al. (2025). "EST Transformer: Enhanced Spatiotemporal Representation Learning for Time Series Anomaly Detection." *J. Intell. Inf. Syst.*, 63, 783–805.
8. Wang, H. et al. (2025). "Res2Coder: A Two-stage Residual Autoencoder for Unsupervised Time Series Anomaly Detection." *Applied Intelligence*, 55, 804.
9. Audibert, J. et al. (2020). "USAD: Unsupervised Anomaly Detection on Multivariate Time Series." *ACM SIGKDD 2020*, pp. 3395–3404.
10. Deng, A. & Hooi, B. (2021). "Graph Neural Network-Based Anomaly Detection in Multivariate Time Series." *AAAI 2021*, 35, 4027–4035.
11. Li, J. et al. (2025). "Enhanced Anomaly Detection of Industrial Control Systems via Graph-driven Spatio-temporal Adversarial Deep Support Vector Data Description." *Expert Systems with Applications*, 270, 126573.
12. Zheng, Y. et al. (2023). "Correlation-aware Spatial–Temporal Graph Learning for Multivariate Time-series Anomaly Detection." *IEEE TNNLS*, 35, 11802–11816.
