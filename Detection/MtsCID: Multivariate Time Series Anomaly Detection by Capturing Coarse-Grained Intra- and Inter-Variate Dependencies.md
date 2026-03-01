# MtsCID: Multivariate Time Series Anomaly Detection by Capturing Coarse-Grained Intra- and Inter-Variate Dependencies

## 종합 분석 보고서

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장
MtsCID는 기존 다변량 시계열(MTS) 이상 탐지 방법들이 **지나치게 세밀한 세분화(excessively fine granularity)**에 집중하여 시계열 내부(intra-variate)의 시간적 의존성과 변량 간(inter-variate) 관계에서 **두드러진(salient) 패턴을 포착하지 못하는 한계**를 지적하고, **조대립(coarse-grained)** 수준에서 이 두 가지 의존성을 동시에 학습하는 이중 네트워크(dual-network) 아키텍처를 제안한다.

### 주요 기여 3가지

1. **시간-주파수 교차 학습 체계(Time-Frequency Interleaved Learning Scheme)**: 주파수 도메인에서 변량 간 시간 단계를 정렬하고, 시간 도메인에서 조대립 시간적 의존성을 학습하는 교차 처리 방식을 도입하여 정상 패턴 포착 능력을 향상
2. **이중 네트워크 기반 이상 탐지 구조**: 시간적 의존성 학습을 위한 t-AutoEncoder와 변량 간 관계 학습을 위한 i-Encoder + 정현파 프로토타입 상호작용 모듈(p-i Module)로 구성된 이중 네트워크 아키텍처 제안
3. **7개 공개 데이터셋에서의 광범위한 실험**: 9개 SOTA 기준선과 비교하여 동등 이상의 성능 달성, 특히 GECCO 데이터셋에서 F1 기준 42.12% 이상의 성능 향상 달성

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**다변량 시계열 이상 탐지(MTS Anomaly Detection)**는 웹 애플리케이션 운영에서 장애 관리를 위해 필수적이다. 이 과제는 다음 이유로 **준지도 학습(semi-supervised learning)** 문제로 정의된다:

- 데이터 레이블링의 노동 집약성
- 이상 데이터의 희소성
- 정상 데이터만으로 모델 훈련

**기존 방법의 한계:**
| 방법 유형 | 대표 방법 | 한계 |
|---|---|---|
| 근접성 기반(Proximity-based) | iForest, DeepSVDD, DAGMM | 동적 시간 의존성과 복잡한 변량 간 관계 포착 불가 |
| 시간 기반(Temporal-based) | THOC, AT, DCdetector | 지나치게 세밀한 시간 단계에 집중하여 두드러진 패턴 놓침 |
| 시공간 기반(Spatiotemporal-based) | InterFusion, MEMTO, STEN | 변량 간 관계의 세밀한 처리로 인한 준최적 성능 |

**핵심 문제 정의:**

주어진 부분 수열 집합 $D = \{X^1, \ldots, X^N\}$에서 각 $X^i \in \mathbb{R}^{T \times C}$는 관측 부분 수열 $[x^i_1, \ldots, x^i_L]$이며, $x^i_t \in \mathbb{R}^C$는 시간 $t$에서의 다변량 관측 벡터이다. 훈련 데이터가 오직 정상 관측으로만 구성된 상태에서, 각 시간 단계의 이상 여부를 식별하는 것이 목표이다.

---

### 2.2 제안하는 방법 (수식 포함)

#### (A) 전체 모델 구조 (MtsCID)

MtsCID는 **이중 분기(dual-branch)** 구조로 구성된다:

- **상위 분기 (Upper Branch)**: t-AutoEncoder — 변량 내부 시간적 의존성 학습
- **하위 분기 (Lower Branch)**: i-Encoder + p-i Module — 변량 간 관계 학습

---

#### (B) Temporal AutoEncoder Network (t-AutoEncoder)

입력 $X \in \mathbb{R}^{B \times L \times C}$에 대해:

**1단계: 주파수 도메인 변환 (DFT)**

$$H = \text{DFT}(X) $$

여기서 $H \in \mathbb{R}^{B \times f \times C}$는 주파수 성분의 실수부와 허수부를 포함한다.

**2단계: fc-Linear를 통한 잠재 공간 투영**

$$Q = K = V = HW $$

여기서 $W^{(r)}, W^{(i)} \in \mathbb{R}^{C \times d}$는 실수부와 허수부에 대한 학습 가능한 파라미터이다.

**3단계: fc-Transformer를 통한 주파수 성분 간 의존성 학습**

$$\hat{H} = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V $$

**4단계: 역 DFT를 통한 시간 도메인 복원**

$$Z = \text{LayerNorm}\left(\text{iDFT}(\hat{H} + V)\right) $$

**5단계: 다중 스케일 패치(Multi-Scale Patch) 생성 및 ts-Attention**

$$Z^{p_i} = \text{Patch}(Z) $$

여기서 $Z^{p_i} \in \mathbb{R}^{(B \times d) \times n_i \times p_i}$이며, $n_i$는 패치 수, $p_i$는 패치 크기이다.

**6단계: 어텐션 맵을 통한 조대립 시간적 의존성 포착**

$$A^{p_i} = \text{Softmax}\left(\frac{Z^{p_i} {Z^{p_i}}^T}{\sqrt{p_i}}\right) M^{p_i} $$

여기서 $M^{p_i} \in \mathbb{R}^{n_i \times p_i}$는 학습 가능한 파라미터이다.

**7단계: 어텐션 맵 평균화 및 언패치**

$$\hat{Z} = \text{Unpatch}\left(\frac{1}{m}\sum_{i=1}^{m} A^{p_i}\right) $$

**8단계: 디코더를 통한 재구성**

$$\hat{X} = \text{Decoder}(\hat{Z}) $$

---

#### (C) Inter-variate Dependency Encoder Network (i-Encoder)

**1단계: 1D 합성곱을 통한 국소 시간 의존성 포착**

$$T = \text{Conv1d}(X) $$

여기서 커널 크기 $k$를 사용하여 $T \in \mathbb{R}^{B \times C \times L}$을 생성한다.

**2단계: 주파수 도메인 변환**

$$E = \text{DFT}(T) $$

**3단계: 변량 간 fc-Transformer를 통한 관계 학습**

$$J = \text{Softmax}\left(\frac{EE^T}{\sqrt{f}}\right)E $$

> 주목할 점: 여기서 어텐션은 **변량(variate) 차원**을 따라 연산되어, 변량 간 관계를 포착한다.

**4단계: 시간 도메인 복원 및 잔차 연결**

$$O = \text{LayerNorm}(\text{iDFT}(J) + X) $$

---

#### (D) Sinusoidal Prototypes Interaction Module (p-i Module)

고정된 정현파 프로토타입 $M \in \mathbb{R}^{L \times C}$를 정의한다:

$$M_{i,j} = \cos\left(\frac{2\pi}{L} \cdot i \cdot j\right) $$

여기서 $i = 0, 1, \ldots, L-1$이고 $j = 0, 1, \ldots, C-1$이다.

표현 $O$와 프로토타입 간의 상호작용:

$$w_{ti} = \frac{\exp(\langle O_{:,t,:}, M_{i,:} \rangle / \tau)}{\sum_{j=1}^{L} \exp(\langle O_{:,t,:}, M_{j,:} \rangle / \tau)} $$

**기존 MEMTO[30]와의 차이점:** MEMTO는 동적 메모리 업데이트 메커니즘을 사용하지만, MtsCID는 **고정된 정현파 프로토타입**을 사용하여 훈련 불안정성을 제거하고 추가적인 2단계 훈련/클러스터링 과정이 불필요하다.

---

#### (E) 학습 태스크 및 손실 함수

**시간 의존성 재구성 손실:**

$$L_{t\text{-}rec} = \frac{1}{B}\sum_{s=1}^{B} \|X^s - \hat{X}^s\|_2^2 $$

**프로토타입 기반 엔트로피 정규화 손실:**

$$L_{i\text{-}ent} = \frac{1}{B}\sum_{s=1}^{B}\sum_{t=1}^{L}\sum_{i=1}^{C} -w_{t,i} \log(w_{t,i}) $$

**최종 목적 함수:**

$$L = L_{t\text{-}rec} + \lambda L_{i\text{-}ent} $$

여기서 $\lambda$는 가중치 하이퍼파라미터이다 (기본값 $\lambda = 0.1$).

---

#### (F) 이상 점수 계산 (추론 단계)

**시간 편차(Temporal Deviation, TD):** 입력 $X_{t,:}$와 재구성 $\hat{X}_{t,:}$ 간의 거리

**관계 편차(Relationship Deviation, RD):** $O_{t,:}$와 가장 가까운 메모리 단계 $m_{s,:}$ 간의 거리

**최종 이상 점수:**

$$\text{AScore}(X) = \text{Softmax}([RD(O_{t,:}, M_{:,:})]) \circ [TD(X_{t,:}, \hat{X}_{t,:})] $$

여기서 $\circ$는 원소별 곱셈이며, $\text{AScore}(X) \in \mathbb{R}^L$은 각 시간 단계의 이상 점수이다.

---

### 2.3 성능 향상 결과

#### Point-Adjustment F1 Score 비교 (주요 데이터셋)

| 방법 | SMD | MSL | SMAP | SWaT | PSM |
|---|---|---|---|---|---|
| iForest | 53.64 | 66.45 | 55.53 | 47.02 | 83.48 |
| THOC | 84.99 | 89.69 | 90.68 | 85.13 | 89.54 |
| AT | 92.88 | 94.10 | 96.61 | 95.40 | 98.19 |
| DCdetector | 86.28 | 92.21 | 96.57 | 98.02 | 98.24 |
| MEMTO | 91.84 | 94.13 | 96.52 | 96.26 | **98.50** |
| STEN | 83.29 | 92.42 | 96.49 | 95.31 | 97.88 |
| **MtsCID** | **93.39** | **95.13** | **97.32** | 96.91 | **98.54** |

#### GECCO 데이터셋에서의 특출한 성능

| 방법 | F1 |
|---|---|
| AT | 44.53 |
| DCdetector | 37.08 |
| MEMTO | 54.25 |
| STEN | 36.34 |
| **MtsCID** | **77.10** |

GECCO에서 **2위 대비 42.12% 향상**이라는 두드러진 성능 차이를 보인다.

---

### 2.4 한계점

논문에서 저자들이 명시한 한계점:

1. **독립적 학습의 정보 손실**: 시간적 의존성(상위 분기)과 변량 간 관계(하위 분기)가 훈련 중 **독립적으로 학습**되어, 두 분기 간의 상호 보완적 정보가 손실될 수 있다.
2. **주파수 도메인 정보의 활용 미흡**: 주파수 도메인 정보의 활용이 아직 충분히 개발되지 않았다.
3. **SWAN 데이터셋 성능**: 다른 데이터셋에서 우수한 성능을 보이는 반면, SWAN 데이터셋에서는 AF-F1 기준으로 경쟁 방법들과 비슷한 수준에 그친다.

---

## 3. 모델의 일반화 성능 향상 가능성

MtsCID의 일반화 성능과 관련된 설계 요소와 실험적 근거를 중점적으로 분석한다.

### 3.1 일반화 성능 향상에 기여하는 설계 요소

#### (1) 조대립 처리(Coarse-Grained Processing)의 일반화 효과

세밀한 시간 단계 수준 대신 **패치(patch) 수준의 어텐션 맵**을 사용함으로써:
- 노이즈에 대한 강건성(robustness) 향상
- 특정 데이터셋에 과적합되는 경향 감소
- 다양한 데이터셋에서 안정적인 성능 (7개 데이터셋 중 6개에서 최고 F1)

**Ablation 실험 근거 (Table 4):**
- 다중 스케일 패치 어텐션 제거 시 ($\text{MtsCID}_\text{co}$): SMD 93.13→93.39, MSL 94.91→95.13 등 모든 데이터셋에서 성능 하락
- 합성곱 제거 시 ($\text{MtsCID}_\text{ao}$): 마찬가지로 성능 하락

#### (2) 주파수 도메인 처리의 일반화 기여

주파수 도메인에서 작업하는 것이 일반화에 기여하는 두 가지 이유:
1. **시간 단계별 변동에 덜 민감**: 개별 시간 단계에 의한 영향이 줄어들어, 더 안정적인 패턴 학습 가능
2. **표현 공간의 축소**: 연속적인 시계열 값의 주파수 도메인 표현이 모집단(population)을 줄여, 재구성의 결정성(determinism) 향상

**실험 근거:** $\text{MtsCID}_\text{td}$(시간 도메인에서만 작동하는 변형)는 모든 데이터셋에서 MtsCID보다 낮은 성능을 보인다 (예: SMD 91.22 vs 93.39).

#### (3) 고정 정현파 프로토타입의 안정성

- MEMTO[30]의 동적 메모리 업데이트는 훈련 불안정성 문제가 있지만, MtsCID의 **고정 정현파 프로토타입**은 이를 해결
- 학습 파라미터가 줄어들어 과적합 위험 감소
- 추가적인 2단계 훈련/클러스터링이 불필요하여 다양한 데이터셋에 쉽게 적용 가능

#### (4) 다중 스케일 패치의 범용성

$p_i \in \{10, 20\}$으로 설정된 다중 스케일 패치는:
- 다양한 시간적 해상도(granularity)를 동시에 포착
- 서로 다른 주기성을 가진 데이터셋에 대해 범용적으로 적용 가능

### 3.2 일반화 성능의 실험적 검증

#### 민감도 분석(Sensitivity Analysis)

| 분석 항목 | 범위 | 결과 |
|---|---|---|
| 손실 가중치 $\lambda$ | $10^{-3} \sim 10^2$ | 성능에 거의 둔감 |
| 멀티스케일 패치 설정 | [5,10], [10,20], [5,10,20] | 약간의 변동만 존재 |
| 합성곱 커널 크기 | {1, 3, 5, 7, 9} | 커널 크기 5에서 대부분 최적 |

이러한 하이퍼파라미터 둔감성은 **다양한 도메인에 대한 높은 일반화 가능성**을 시사한다.

#### 확장성(Scalability)

MSL 데이터셋에서 MtsCID는 모든 비교 대상 기준선보다 **훈련 및 테스트 시간 모두에서 우수한 효율성**을 보여, 실제 애플리케이션에 대한 강한 확장성을 입증하였다.

### 3.3 일반화 성능 향상을 위한 향후 방향

1. **자기지도 학습(Self-supervised Learning) 통합**: 두 분기 간의 상호 정보를 활용하면 일반화 성능이 더 향상될 가능성이 있다.
2. **주파수 도메인 정보의 심층 활용**: 현재 DFT/iDFT를 통한 기본적인 주파수 처리만 사용 중이므로, 더 정교한 주파수 분석(예: Wavelet, STFT) 활용 가능
3. **도메인 적응(Domain Adaptation)**: 현재는 데이터셋별 훈련이 필요하나, 사전학습 모델을 통한 도메인 간 전이 가능성 탐색
4. **다양한 이상 유형 대응**: SWAN 데이터셋에서의 상대적 약점을 개선하기 위해 다양한 이상 패턴에 대한 적응력 강화

---

## 4. 해당 논문이 앞으로의 연구에 미치는 영향과 고려할 점

### 4.1 연구에 미치는 영향

#### (1) 세분화 수준(Granularity)에 대한 새로운 관점 제시
기존 연구들이 더 세밀한 시간 단계 수준의 분석에 집중했다면, MtsCID는 **"세밀함이 항상 좋은 것은 아니다"**라는 중요한 메시지를 전달한다. 이는 향후 시계열 분석 전반에 걸쳐 적절한 세분화 수준의 선택이 중요한 연구 주제가 될 수 있음을 시사한다.

#### (2) 시간-주파수 교차 처리의 패러다임
주파수 도메인에서 변량 간 관계를 학습하고, 시간 도메인에서 시간적 의존성을 학습하는 **교차 처리 방식**은 시계열 분석의 다른 태스크(예측, 분류 등)에도 적용 가능한 범용적 접근법이다.

#### (3) 고정 프로토타입 기반 학습의 실용성
동적 메모리 업데이트 대신 **고정 정현파 프로토타입**을 사용하여 훈련 안정성과 단순성을 달성한 것은, 산업 현장에서의 실용적 배포를 용이하게 한다.

#### (4) 이중 네트워크 구조의 효과성 검증
시간적 의존성과 변량 간 관계를 별도의 네트워크에서 학습하는 접근은, 복잡한 다변량 시계열에서 각 유형의 의존성을 명시적으로 분리하여 학습할 수 있음을 보여준다.

### 4.2 앞으로 연구 시 고려할 점

#### (1) 두 분기 간 상호작용 메커니즘
현재 두 분기는 독립적으로 학습되며 추론 시에만 결합된다. **교차 주의(cross-attention)**나 **대조 학습(contrastive learning)**을 통해 훈련 중 두 분기 간의 상호작용을 강화하면 더 풍부한 표현 학습이 가능할 것이다.

#### (2) 적응적 패치 크기 선택
현재 패치 크기는 고정되어 있지만($p_i \in \{10, 20\}$), 데이터의 특성에 따라 **적응적으로 패치 크기를 결정**하는 메커니즘이 필요하다. 특히 서로 다른 주기성을 가진 변량에 대해 개별적인 패치 크기를 적용할 수 있다.

#### (3) 온라인/스트리밍 환경에서의 적용
현재 MtsCID는 고정 길이 부분 수열(sliding window of length 100)을 사용한다. 실시간 시스템에서는 **점진적 학습(incremental learning)**이나 **온라인 적응** 메커니즘이 필요하다.

#### (4) 해석 가능성(Interpretability)
이상 탐지에서 **어느 변량이 이상에 기여했는지**를 파악하는 것이 운영 현장에서 중요하다. 현재 MtsCID는 시간 단계 수준의 이상 점수만 제공하며, 변량 수준의 원인 분석(root cause analysis) 기능이 부족하다.

#### (5) 이상 유형의 다양성 대응
GECCO에서는 두드러진 성능을 보이지만, SWAN에서의 AF-F1은 상대적으로 낮다. 다양한 이상 유형(점 이상, 컨텍스트 이상, 집단 이상 등)에 대한 모델의 강건성을 체계적으로 분석하고 개선할 필요가 있다.

#### (6) 대규모 시계열에 대한 확장성
현재 실험은 최대 51개 변량(SWaT)에 대해 수행되었다. 산업 현장에서는 수백~수천 개의 변량이 존재할 수 있으므로, 이에 대한 확장성 검증이 필요하다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

아래 표는 MtsCID와 2020년 이후 발표된 주요 MTS 이상 탐지 방법들을 체계적으로 비교 분석한 것이다.

### 5.1 방법론 비교

| 방법 | 발표 연도/학회 | 접근 방식 | 의존성 포착 | 도메인 | 핵심 특징 |
|---|---|---|---|---|---|
| **THOC** [29] | 2020 / NeurIPS | Temporal | Intra-variate | Time | 계층적 클러스터링 기반 다중 스케일 시간 특성 통합 |
| **USAD** [3] | 2020 / KDD | Temporal | Intra-variate | Time | 적대적 학습 기반 오토인코더 |
| **InterFusion** [15] | 2021 / KDD | Spatiotemporal | Intra + Inter | Time | 계층적 VAE로 시간적/변량간 의존성 동시 학습 |
| **AT (Anomaly Transformer)** [36] | 2021 / ICLR | Temporal | Intra-variate | Time | Prior-association과 series-association 간의 불일치 학습 |
| **TranAD** [32] | 2022 / arXiv | Temporal | Intra-variate | Time | 심층 Transformer + 적대적 훈련 |
| **TFAD** [38] | 2022 / CIKM | Temporal | Intra-variate | Time + Freq | 시간-주파수 분해 기반 이상 탐지 |
| **DCdetector** [37] | 2023 / KDD | Temporal | Intra-variate | Time | 이중 어텐션 대조 표현 학습 |
| **MTGFlow** [41] | 2023 / AAAI | Unsupervised | Intra + Inter | Time | 그래프 기반 정규화 흐름 |
| **MEMTO** [30] | 2024 / NeurIPS | Spatiotemporal | Intra + Inter | Time | 메모리 기반 Transformer, 동적 프로토타입 |
| **STEN** [5] | 2024 / ECML-PKDD | Spatiotemporal | Intra + Inter | Time | 자기지도 시공간 정규성 학습 |
| **Nam et al.** [22] | 2024 / WWW | Temporal | Intra-variate | Time + Freq | 시간-주파수 세분화 불일치 해결 |
| **Wang et al.** [33] | 2024 / WWW | Unsupervised | Intra-variate | Freq | VAE의 주파수 관점 재방문 |
| **MtsCID (본 논문)** | 2025 / WWW | Spatiotemporal | Intra + Inter | **Time + Freq (교차)** | 조대립 이중 네트워크 + 고정 정현파 프로토타입 |

### 5.2 핵심 차별화 포인트 분석

#### MtsCID vs AT [36]
- **AT**: 단일 네트워크에서 prior/series association 간의 불일치를 통해 이상 탐지
- **MtsCID**: 이중 네트워크로 시간적 의존성과 변량 간 관계를 분리하여 학습
- **성능 차이**: SMD에서 MtsCID(93.39) > AT(92.88), GECCO에서 MtsCID(77.10) >> AT(44.53)

#### MtsCID vs DCdetector [37]
- **DCdetector**: 패치 기반 이중 어텐션을 대조 학습으로 훈련
- **MtsCID**: 패치 기반이지만 주파수 도메인 처리를 추가하고, 변량 간 관계를 별도 네트워크로 학습
- **성능 차이**: SMD에서 MtsCID(93.39) >> DCdetector(86.28), GECCO에서 MtsCID(77.10) >> DCdetector(37.08)

#### MtsCID vs MEMTO [30]
- **MEMTO**: 단일 Transformer + 동적 메모리 업데이트 (2단계 훈련 + 클러스터링 필요)
- **MtsCID**: 이중 네트워크 + 고정 정현파 프로토타입 (추가 단계 불필요)
- **장점**: 훈련 안정성 향상, 더 간단한 학습 파이프라인
- **성능 차이**: 대부분 데이터셋에서 MtsCID가 우세하며, 특히 GECCO(77.10 vs 54.25)에서 큰 격차

#### MtsCID vs STEN [5]
- **STEN**: 부분 수열 순서 예측 + 거리 예측을 통한 자기지도 학습
- **MtsCID**: 주파수 도메인 교차 처리 + 조대립 패치 어텐션
- **성능 차이**: SMD에서 MtsCID(93.39) >> STEN(83.29)

#### MtsCID vs TFAD [38] / Nam et al. [22]
- 이 방법들도 주파수 도메인을 활용하지만, MtsCID는 주파수 도메인을 **변량 간 정렬 및 관계 학습**에 특화하여 사용하는 점에서 차별화됨

### 5.3 종합 비교 분석표

| 평가 차원 | AT | DCdetector | MEMTO | STEN | **MtsCID** |
|---|---|---|---|---|---|
| 시간적 의존성 학습 | ✅ 강 | ✅ 강 | ✅ 중 | ✅ 중 | ✅ **강** |
| 변량 간 관계 학습 | ❌ | ❌ | ✅ 중 | ✅ 중 | ✅ **강** |
| 주파수 도메인 활용 | ❌ | ❌ | ❌ | ❌ | ✅ **교차 처리** |
| 조대립 처리 | ❌ | 부분적 | ❌ | ❌ | ✅ **다중스케일** |
| 훈련 안정성 | ✅ | ✅ | △ (동적메모리) | ✅ | ✅ **고정프로토타입** |
| 추가 훈련 단계 | 불필요 | 불필요 | 2단계 필요 | 불필요 | **불필요** |
| 계산 효율성 | 중 | 중 | 낮 | 중 | **높** |
| 다양한 데이터셋 일반화 | 중 | 중 | 중 | 중 | **높** |

---

## 결론

MtsCID는 다변량 시계열 이상 탐지에서 **조대립(coarse-grained) 수준의 변량 내부 및 변량 간 의존성 학습**이라는 새로운 관점을 제시하여, 7개 데이터셋에서 SOTA 수준의 성능을 달성하였다. 특히 **시간-주파수 교차 처리**, **고정 정현파 프로토타입**, **다중 스케일 패치 어텐션** 등의 설계가 일반화 성능 향상에 핵심적으로 기여하였다. 향후 연구에서는 두 분기 간의 상호작용 강화, 적응적 패치 크기, 변량 수준 해석 가능성, 온라인 학습 확장 등이 주요 과제가 될 것이다.

---

## 참고자료

1. **논문 원문:** Xie, Y., Zhang, H., & Babar, M. A. (2025). "Multivariate Time Series Anomaly Detection by Capturing Coarse-Grained Intra- and Inter-Variate Dependencies." *WWW '25, April 2025, Sydney, Australia.* arXiv:2501.16364v1 [cs.LG]
2. **GitHub 코드 저장소:** https://github.com/ilwoof/MtsCID/
3. **Xu, J. (2021).** "Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy." arXiv:2110.02642 (AT)
4. **Yang, Y. et al. (2023).** "DCdetector: Dual Attention Contrastive Representation Learning for Time Series Anomaly Detection." *KDD 2023.* (DCdetector)
5. **Song, J. et al. (2024).** "Memto: Memory-guided Transformer for Multivariate Time Series Anomaly Detection." *NeurIPS 2024.* (MEMTO)
6. **Chen, Y. et al. (2024).** "Self-supervised Spatial-Temporal Normality Learning for Time Series Anomaly Detection." *ECML-PKDD 2024.* (STEN)
7. **Shen, L. et al. (2020).** "Timeseries Anomaly Detection Using Temporal Hierarchical One-Class Network." *NeurIPS 2020.* (THOC)
8. **Li, Z. et al. (2021).** "Multivariate Time Series Anomaly Detection and Interpretation Using Hierarchical Inter-Metric and Temporal Embedding." *KDD 2021.* (InterFusion)
9. **Nam, Y. et al. (2024).** "Breaking the Time-Frequency Granularity Discrepancy in Time-Series Anomaly Detection." *WWW 2024.*
10. **Wang, Z. et al. (2024).** "Revisiting VAE for Unsupervised Time Series Anomaly Detection: A Frequency Perspective." *WWW 2024.*
11. **Zhang, C. et al. (2022).** "TFAD: A Decomposition Time Series Anomaly Detection Architecture with Time-Frequency Analysis." *CIKM 2022.*
12. **Audibert, J. et al. (2020).** "USAD: Unsupervised Anomaly Detection on Multivariate Time Series." *KDD 2020.*
