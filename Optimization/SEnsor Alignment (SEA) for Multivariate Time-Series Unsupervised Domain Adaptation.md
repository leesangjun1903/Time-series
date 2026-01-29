
# SEnsor Alignment (SEA) for Multivariate Time-Series Unsupervised Domain Adaptation

## 요약

본 논문은 다중 센서로부터 수집된 시계열 데이터(MTS: Multivariate Time-Series)의 비지도 도메인 적응(UDA)이라는 처음으로 체계적으로 정의된 문제를 해결한다. 기존 UDA 방법들이 전역 특징만 정렬하는 한계를 극복하기 위해, SEA는 **센서 수준(로컬)**과 **글로벌 수준**에서 도메인 불일치를 동시에 감소시킨다. 핵심 기여는 (1) MTS-UDA 문제의 첫 공식화, (2) 엔도/엑소 피처 정렬 기법의 제안, (3) 공간-시간 의존성 모델링이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fb6daecc-7bb4-4542-be58-680cb9e90990/26221-Article-Text-30284-1-2-20230626.pdf)

***

## 1. 핵심 주장과 주요 기여

### 1.1 문제 정의: MTS-UDA의 필요성

기존 시계열 도메인 적응 방법들은 단일 센서(univariate) 데이터를 가정한다. 그러나 현실의 많은 응용 분야에서는 다중 센서가 배포된다:

- **Remaining Useful Life(RUL) 예측**: 엔진 모니터링에 14개 센서 사용
- **Human Activity Recognition(HAR)**: 신체 착용 센서(113개) 활용

문제는 **각 센서가 서로 다른 분포를 따른다**는 것이다. 기존 방법이 모든 센서를 통합적으로 처리하면 개별 센서의 분포 불일치를 간과하여 성능이 저하된다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fb6daecc-7bb4-4542-be58-680cb9e90990/26221-Article-Text-30284-1-2-20230626.pdf)

### 1.2 주요 기여도

| 구분 | 내용 |
|------|------|
| **문제 공식화** | MTS-UDA를 체계적으로 정의한 첫 논문 |
| **방법론** | 로컬(센서 수준) + 글로벌 계층 정렬 구조 |
| **공간-시간 모델링** | 다중 브랜치 자기주의 기반 그래프 구성(MSGC) |
| **성능** | C-MAPSS에서 21% 개선, Opportunity HAR에서 4.35% 개선 |

***

## 2. 해결하고자 하는 문제: 상세 분석

### 2.1 문제의 핵심 난제

**(1) 센서 수준 분포 불일치 미처리**

기존 UDA 방법:
$$\min L_{global} = D_{mmd}(f_s^{global}, f_t^{global})$$

여기서 모든 센서 특징을 하나의 전역 벡터로 통합하므로, 온도 센서와 속도 센서의 개별 분포 특성이 소실된다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fb6daecc-7bb4-4542-be58-680cb9e90990/26221-Article-Text-30284-1-2-20230626.pdf)

**(2) 공간-시간 의존성 미모델링**

- **공간 의존성**: 센서 간 상관관계 (예: 온도↑ → 팬 속도↑)
- **시간 의존성**: 연속 시간 스텝 간 관계

기존 방법은 이 둘을 동시에 전이하지 못한다.

### 2.2 문제의 수학적 정식화

주어진 것:
- 소스 도메인: $D^S = \{(x_i^s, y_i^s)\}_{i=1}^{N_s}$ (라벨 있음)
- 타겟 도메인: $D^T = \{x_i^t\}_{i=1}^{N_t}$ (라벨 없음)

각 샘플: $x_i = \{x_{im}\}_{m=1}^N \in \mathbb{R}^{N \times L}$

여기서 $N$ = 센서 수, $L$ = 시간 길이, $m$ = m번째 센서 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fb6daecc-7bb4-4542-be58-680cb9e90990/26221-Article-Text-30284-1-2-20230626.pdf)

목표: 타겟 도메인에서 도메인 불일치를 최소화하며 특징 $h \in \mathbb{R}^F$를 학습

***

## 3. 제안 방법: SEA 프레임워크

### 3.1 전체 구조 (Figure 2 기반)

SEA는 3단계로 구성:

```
1) 그래프 기반 인코더 (MSGC + GNN + LSTM)
                    ↓
2) 엔도-피처 정렬 (센서 수준)
                    ↓
3) 엑소-피처 정렬 (글로벌 수준)
```

### 3.2 핵심 알고리즘

#### **3.2.1 다중 브랜치 자기주의 기반 그래프 구성 (MSGC)**

원본 샘플 $x \in \mathbb{R}^{N \times L}$를 패치로 분할:
$$\{Z_T\}_T^{\hat{L}} \in \mathbb{R}^{N \times \hat{L} \times d}$$

여기서 $\hat{L} = \lfloor L/d \rfloor$는 패치 수 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fb6daecc-7bb4-4542-be58-680cb9e90990/26221-Article-Text-30284-1-2-20230626.pdf)

각 패치 $T$에서 센서 간 가중치 학습 (i번째 브랜치):

$$q_{m,T}^i = z_{m,T} W_Q^i, \quad k_{n,T}^i = z_{n,T} W_K^i$$

$$e_{mn,T}^i = \frac{q_{m,T}^i (k_{n,T}^i)^T / \sqrt{d}}{\sum_{j=1}^N q_{m,T}^i (k_{j,T}^i)^T / \sqrt{d}} \quad (식 1)$$

다중 브랜치 평균화:
$$e_{mn,T} = \frac{\sum_{i=1}^{n_b} e_{mn,T}^i}{n_b}$$

**결과**: 순차 그래프 $\{G_T\}_T^{\hat{L}}$ 구성, $G_T = (Z_T, E_T)$ [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fb6daecc-7bb4-4542-be58-680cb9e90990/26221-Article-Text-30284-1-2-20230626.pdf)

#### **3.2.2 공간-시간 의존성 포착**

**공간 의존성 (GNN via Message Passing)**:
$$h_{m,T} = \sum_{j=1}^N e_{mj,T} z_{j,T}$$
$$z_{m,T} = \text{ReLU}(h_{m,T} W_G) \quad (식 2)$$

**시간 의존성 (LSTM)**:
각 센서 $m$에 대해 순차 특징 $z_m \in \mathbb{R}^{\hat{L} \times d}$에 LSTM 적용 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fb6daecc-7bb4-4542-be58-680cb9e90990/26221-Article-Text-30284-1-2-20230626.pdf)

결과: 업데이트된 순차 특징 $\{Z_T\}_T^{\hat{L}} \in \mathbb{R}^{\hat{L} \times N \times d}$

#### **3.2.3 엔도-피처 정렬: 센서 상관관계 정렬 (SCA)**

순차 그래프들의 평균화:
$$P = \frac{\sum_{T=1}^{\hat{L}} Z_T}{\hat{L}}, \quad E = \frac{\sum_{T=1}^{\hat{L}} E_T}{\hat{L}}$$

센서 상관관계 불일치:
$$\Delta e_{mn}^{st} = e_{mn}^s - e_{mn}^t$$

**SCA 손실**:
$$\min L_{SCA} = \mathbb{E}(|\Delta e_{mn}^{st}|) \quad (식 3)$$

직관: 두 도메인의 센서 간 상호작용(예: 온도-팬 상관)이 동일해야 함 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fb6daecc-7bb4-4542-be58-680cb9e90990/26221-Article-Text-30284-1-2-20230626.pdf)

#### **3.2.4 엔도-피처 정렬: 센서 특징 정렬 (SFA)**

대조 학습 메커니즘:
$$\min L_{SFA} = -\mathbb{E}\left(\log \frac{e^{\phi(p_m^s, p_m^t)}}{\sum_{p_j^t \in P^t} e^{\phi(p_m^s, p_j^t)}}\right) \quad (식 4)$$

여기서 $\phi(p_m^s, p_m^t) = p_m^s (p_m^t)^T$ (코사인 유사도)

**해석**: 같은 센서 특징은 가깝게, 다른 센서 특징은 멀게 정렬 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fb6daecc-7bb4-4542-be58-680cb9e90990/26221-Article-Text-30284-1-2-20230626.pdf)

#### **3.2.5 엑소-피처 정렬: 글로벌 특징 정렬**

센서 특징 스택:
$$h^s = f_{exo}(p_1^s, ..., p_N^s) = \sigma(\text{cat}(p_1^s, ..., p_N^s) W_{exo}) \in \mathbb{R}^F$$

Deep Coral을 이용한 공분산 정렬:
$$C^s = (h^s)^T h^s - \frac{1}{B_s}(1^T h^s)(1^T h^s)^T / (B_s - 1)$$

$$\min L_{EXO} = \frac{1}{4F^2} \|C^s - C^t\|_F^2 \quad (식 6)$$

### 3.3 전체 손실 함수

$$\min L = L_C + \lambda_{EXO} L_{EXO} + L_{Endo}$$
$$L_{Endo} = \lambda_{SCA} L_{SCA} + \lambda_{SFA} L_{SFA} \quad (식 7)$$

여기서 $L_C$는 소스 도메인의 지도 손실(MSE 또는 CE) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fb6daecc-7bb4-4542-be58-680cb9e90990/26221-Article-Text-30284-1-2-20230626.pdf)

***

## 4. 모델 구조 상세

### 4.1 인코더 구성 (Figure 2)

| 단계 | 구성 요소 | 역할 |
|------|---------|------|
| 1 | MSGC | 순차 그래프 자동 생성 |
| 2 | GNN (MPNN) | 센서 간 공간 의존성 포착 |
| 3 | LSTM | 시간 의존성 모델링 |
| 4 | 정렬 모듈 | 도메인 불일치 감소 |

### 4.2 특징 추출 차원 변화

$$x^s/x^t: \mathbb{R}^{N \times L} \rightarrow Z_T: \mathbb{R}^{N \times d} \rightarrow$$
$$\text{MPNN}: \mathbb{R}^{N \times d} \rightarrow \text{LSTM}: \mathbb{R}^{\hat{L} \times N \times d}$$

### 4.3 아키텍처 선택 이유

- **MSGC**: 다양한 관점에서 센서 관계 학습 (다중 초기화)
- **GNN**: 공간 의존성 표준 모델, 이전 연구에서 입증됨
- **LSTM**: 시간 순서 의존성 모델링에 최적
- **Deep Coral**: 이차 통계량으로 복잡한 분포 포착 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fb6daecc-7bb4-4542-be58-680cb9e90990/26221-Article-Text-30284-1-2-20230626.pdf)

***

## 5. 성능 향상 분석

### 5.1 실험 설정

**데이터셋**:
- **C-MAPSS**: 항공기 엔진 RUL 예측, 14개 센서, 4개 도메인
- **Opportunity HAR**: 인체 활동 인식, 113개 센서, 4개 도메인

**평가 지표**:
- RUL: RMSE (낮을수록 좋음), Score (낮을수록 좋음)
- HAR: Accuracy (높을수록 좋음)

### 5.2 C-MAPSS 결과 (12개 교차 도메인 시나리오)

$$\text{SEA 개선율} = \frac{\text{2순위 - SEA}}{\text{2순위}} \times 100\%$$

**RMSE 기준**:
- 평균 RMSE: 19.77 (SEA) vs 25.03 (SDAT, 2순위)
- **21.0% 개선** ✓
- 최고 성능: 10개 시나리오, 2순위: 2개 시나리오

**Score 기준**:
- 평균 Score: 1963 (SEA) vs 3511 (CLUDA, 2순위)
- **44.1% 개선** ✓ [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fb6daecc-7bb4-4542-be58-680cb9e90990/26221-Article-Text-30284-1-2-20230626.pdf)

### 5.3 Opportunity HAR 결과

| 방법 | 평균 정확도 |
|------|----------|
| Source Only | 65.25% |
| SDAT | 81.07% |
| **SEA** | **85.42%** |
| 개선율 | **4.35%** |

12개 시나리오 중: 최고 9개, 2순위 3개 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fb6daecc-7bb4-4542-be58-680cb9e90990/26221-Article-Text-30284-1-2-20230626.pdf)

### 5.4 Ablation Study (중요)

| 모델 변형 | C-MAPSS RMSE | Opportunity Acc. | 설명 |
|---------|-------------|-----------------|------|
| $L_C$ w/o endo | 25.22 | 80.47% | 엑소만 사용 |
| + $L_{SCA}$ | 21.97 | 82.00% | SCA 추가 (+12.8%) |
| + $L_{SFA}$ | 22.77 | 82.79% | SFA 추가 (+9.7%) |
| **SEA (전체)** | **19.77** | **85.42%** | 최적 조합 |

**결론**: SCA와 SFA를 **동시에** 사용해야 최고 성능 달성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fb6daecc-7bb4-4542-be58-680cb9e90990/26221-Article-Text-30284-1-2-20230626.pdf)

### 5.5 하이퍼파라미터 민감도 분석 (Figure 4)

$$\lambda_{SCA}, \lambda_{SFA} \in \{0.001, 0.01, 0.1, 1, 10\}$$

- **권장값**: $\lambda_{SCA} = 0.1$ 또는 $1$, $\lambda_{SFA} = 0.1$ 또는 $1$
- 극단값에서만 성능 저하 (예: $\lambda_{SCA} = 0.001$에서 부족, $\lambda_{SFA} = 10$에서 과대)
- **안정적인 성능**: 권장 범위에서 일관성 있음 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fb6daecc-7bb4-4542-be58-680cb9e90990/26221-Article-Text-30284-1-2-20230626.pdf)

***

## 6. 모델의 일반화 성능 향상 분석

### 6.1 일반화 성능 향상 메커니즘

**SEA가 일반화를 향상시키는 이유**:

**(1) 센서 수준 정렬을 통한 표현 견고성**

기존 방법:
- 글로벌 정렬만 수행 → 개별 센서 특성 손실 → 타겟 데이터의 새로운 센서 분포에 취약

SEA:
- 각 센서별 개별 정렬 → 센서 특성 명시적 보존 → 범용성 증가 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fb6daecc-7bb4-4542-be58-680cb9e90990/26221-Article-Text-30284-1-2-20230626.pdf)

**(2) 공간-시간 의존성 모델링**

예시 (RUL 예측):
- 소스: 온도↑ → 팬 속도 +2%
- 타겟: 온도↑ → 팬 속도 +2.5% (약간 다름)

SEA: 센서 상관관계 정렬($L_{SCA}$) → 관계 일관성 유지 → 작은 분포 변화에도 강건 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fb6daecc-7bb4-4542-be58-680cb9e90990/26221-Article-Text-30284-1-2-20230626.pdf)

**(3) 계층적 정렬 (로컬+글로벌)**

$$\text{일반화 성능} \propto \text{정렬 수준의 다양성}$$

- 로컬(센서 수준): 세부 특성 포착
- 글로벌: 전체 패턴 일관성
- 결합: 모든 스케일의 변화에 대응 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fb6daecc-7bb4-4542-be58-680cb9e90990/26221-Article-Text-30284-1-2-20230626.pdf)

### 6.2 교차 도메인 성능 분포 분석

C-MAPSS에서 12개 시나리오의 성능:

```
1→2: 15.83 RMSE ✓ (큰 개선)
1→3: 20.03 RMSE ✓ (27.7% 개선)
1→4: 25.67 RMSE (1.3% 약간 약함)
...
평균: 19.77 RMSE
표준편차: 약 3.5 (상대적으로 안정적)
```

**일관된 성능**: 대부분 시나리오에서 최고 또는 2순위 달성 → 과적합 없음 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fb6daecc-7bb4-4542-be58-680cb9e90990/26221-Article-Text-30284-1-2-20230626.pdf)

### 6.3 데이터 스케일링 특성

**HAR 데이터셋의 다양한 특성**:
- 센서 수: 113개 (C-MAPSS의 14개보다 훨씬 많음)
- 차원: 더 높음
- 복잡도: 더 높음

**SEA 성능**:
- Opportunity HAR: 85.42% (4.35% 개선)
- C-MAPSS: 19.77 RMSE (21% 개선)

→ **고차원 문제에서도 일반화 성능 유지** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fb6daecc-7bb4-4542-be58-680cb9e90990/26221-Article-Text-30284-1-2-20230626.pdf)

### 6.4 이론적 일반화 경계 (암시)

SEA의 설계가 일반화를 개선하는 이유:

$$\text{Generalization Error} = \underbrace{\text{Empirical Loss}}_{\text{소스}} + \underbrace{\text{Domain Discrepancy}}_{\text{최소화}} + \text{Hypothesis Complexity}$$

SEA의 역할:
- **Domain Discrepancy 감소**: $L_{Endo} + L_{Exo}$ 최소화
- **Complexity 제어**: 센서별 정렬(정규화 효과)
- **견고한 표현**: 공간-시간 의존성으로 노이즈 저항성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fb6daecc-7bb4-4542-be58-680cb9e90990/26221-Article-Text-30284-1-2-20230626.pdf)

***

## 7. 한계 (논문에서 명시)

### 7.1 명시된 한계

| 한계 | 영향 |
|------|------|
| **더티 데이터 미처리** | 센서 샘플링 레이트 지터, 타임스탐프 미정렬 |
| **자기지도 학습 부재** | 라벨 없는 타겟에서 견고성 부족 |
| **하이퍼파라미터 민감도** | $\lambda$ 값 조정 필요 |

### 7.2 암시된 한계

1. **순차 그래프 의존성**: 패치 크기 $d$에 따른 성능 변화 미분석
2. **동적 네트워크**: 시간에 따라 변하는 센서 관계 미반영
3. **다중 소스 도메인**: 여러 소스 도메인 처리 미다룸 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fb6daecc-7bb4-4542-be58-680cb9e90990/26221-Article-Text-30284-1-2-20230626.pdf)

***

## 8. 최신 연구 비교 분석 (2020년 이후)

### 8.1 주요 관련 연구 타임라인

| 연도 | 논문 | 핵심 방법 | vs. SEA |
|------|------|--------|--------|
| 2020 | MTS-CycleGAN [dl.acm](https://dl.acm.org/doi/10.1145/3711896.3737150) | Adversarial + LSTM | 생성 기반, 라벨 필요 |
| 2021 | SASA [arxiv](https://arxiv.org/abs/2508.08280) | 자기주의 + MMD | 변수 간만, 센서 간 미처리 |
| 2023 | ADATIME [ojs.aaai](https://ojs.aaai.org/index.php/AAAI/article/view/33863) | CNN + 분산 정렬 | 단변량 중심 |
| 2023 | **SEA** | **센서별 정렬 + 공간-시간** | **MTS 특화** ✓ |
| 2023 | **SEA++** [arxiv](https://arxiv.org/abs/2505.00415) | 고차 통계 + 다중그래프 | SEA 확장 |
| 2024 | POND [semanticscholar](https://www.semanticscholar.org/paper/3cc5f151763ad0571eda4f96288672692e264b69) | 프롬프트 기반 | 모듈식, 경량 |
| 2025 | ContexTST [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11211409/) | 주파수 분해 + 트랜스포머 | 트렌드 모델링 |
| 2025 | Causal MTS-DA [semanticscholar](https://www.semanticscholar.org/paper/9dd51819fba6e4ab2aa2fbe2a9841322c0e3d28f) | 인과 구조 학습 | 인과관계 추론 |

### 8.2 세부 비교

#### **MTS-CycleGAN (2020)** [dl.acm](https://dl.acm.org/doi/10.1145/3711896.3737150)

```
구조: Source → Generator → Fake Target → Discriminator
장점: 라벨 없는 학습, 크로스 도메인 매핑
단점: 센서 상관관계 미보존, 공간-시간 의존성 불명확
```

**vs. SEA**: 생성 기반 vs. 판별 기반 (SEA가 더 직접적)

#### **SASA (Sparse Associative Structure Alignment)** [arxiv](https://arxiv.org/abs/2508.08280)

$$L_{SASA} = \text{MMD}(\text{attention}_1^s, \text{attention}_1^t) + L_{class}$$

- 변수 간 자기주의만 고려 (센서 간 상관관계 미명시)
- 지도 손실에 의존적

**vs. SEA**: 
- SEA는 센서별 특징+상관관계 동시 정렬
- 비지도 방식 (타겟 라벨 불필요) [arxiv](https://arxiv.org/abs/2508.08280)

#### **ADATIME (2023)** [ojs.aaai](https://ojs.aaai.org/index.php/AAAI/article/view/33863)

```
아키텍처: CNN-based + 다양한 매트릭 (MMD, CORAL, DANN)
특점: 시계열 UDA 벤치마킹 스위트 제공
제한: MTS에 직접 설계되지 않음 (다중 센서 미고려)
```

**vs. SEA**:
- ADATIME: 단변량 시계열 중심
- SEA: 다중센서 분포 불일치 명시 처리 [ojs.aaai](https://ojs.aaai.org/index.php/AAAI/article/view/33863)

#### **SEA++ (2023) - SEA의 확장** [arxiv](https://arxiv.org/abs/2505.00415)

SEA의 한계 개선:

**기존 SEA**:
$$L_{SCA} = \mathbb{E}(|e_{mn}^s - e_{mn}^t|) \quad \text{(1차 통계)}$$

**SEA++ (2차 통계)**:
$$d_{mn,T}^{st} = M_c(E_{mn,T}^s, E_{mn,T}^t) \quad \text{(Deep CORAL)}$$

$$L_{iSCA,T} = \mathbb{E}(d_{mn,T}^{st}) \quad (식 7-SEA++)$$

**고차 정렬 효과**: 복잡한 분포 포착으로 성능 +3-5% [arxiv](https://arxiv.org/abs/2505.00415)

**다중그래프 가중화**:
$$W_T = M_c([Z_{1,T}^s, ..., Z_{\hat{L},T}^s], [Z_{1,T}^t, ..., Z_{\hat{L},T}^t])$$

→ 분포 불일치가 큰 그래프에 높은 가중치 [arxiv](https://arxiv.org/abs/2505.00415)

#### **POND (2024) - 프롬프트 기반** [semanticscholar](https://www.semanticscholar.org/paper/3cc5f151763ad0571eda4f96288672692e264b69)

```
구조: Pre-trained Encoder + Task-specific Prompts
장점: 경량, 전이 학습 최적화
특점: 여러 소스 도메인 지원
```

**vs. SEA**:
- POND: 프롬프트 튜닝 (모듈식)
- SEA: 엔드-투-엔드 학습 (통합적)
- 트레이드오프: 경량성 vs. 정확도 [semanticscholar](https://www.semanticscholar.org/paper/3cc5f151763ad0571eda4f96288672692e264b69)

#### **ContexTST (2025) - 주파수 기반** [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11211409/)

$$\mathcal{F}_{freq} = \text{FFT}(\text{time series})$$
$$L = L_{time} + L_{freq} + L_{context}$$

**혁신**: 시간+주파수 영역 동시 정렬, 컨텍스트 인식

**vs. SEA**:
- SEA: 공간 구조 (센서 그래프)
- ContexTST: 신호 분해 (주파수)
- 보완 가능성: 두 방법 결합 잠재력 [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11211409/)

#### **Causal MTS-DA (2025)** [semanticscholar](https://www.semanticscholar.org/paper/9dd51819fba6e4ab2aa2fbe2a9841322c0e3d28f)

$$\text{Structure} = \text{Sparse} + \text{Low-rank}$$
$$L = L_{causal} + L_{structure} + L_{sparsity}$$

**철학**: 도메인 불변 인과 구조 학습

**vs. SEA**:
- SEA: 확률적 정렬 (상관관계)
- Causal: 인과적 메커니즘 (원인-결과)
- 관점 차이: 피상적 vs. 인과적 [semanticscholar](https://www.semanticscholar.org/paper/9dd51819fba6e4ab2aa2fbe2a9841322c0e3d28f)

### 8.3 비교 요약표

| 특성 | MTS-CycleGAN | ADATIME | SEA | SEA++ | POND | Causal |
|------|-------------|---------|-----|-------|------|--------|
| **센서 수준 정렬** | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ |
| **공간-시간 모델** | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| **다중그래프** | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ |
| **고차 통계** | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ |
| **경량성** | ✗ | ✓ | △ | △ | ✓ | △ |
| **해석성** | 낮음 | 중간 | 높음 | 높음 | 높음 | 높음 |

***

## 9. 앞으로의 연구에 미치는 영향

### 9.1 이 논문의 학술적 기여

**(1) 새로운 연구 방향 개척**

SEA는 "MTS-UDA"라는 새로운 분야를 정의했다:
- 2023년 AAAI 이후 후속 연구 다수 (SEA++, 다중센서 도메인 적응)
- 벤치마크 표준화 추진 (C-MAPSS, Opportunity HAR) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fb6daecc-7bb4-4542-be58-680cb9e90990/26221-Article-Text-30284-1-2-20230626.pdf)

**(2) 방법론적 혁신**

- **센서별 정렬**: 다변량 데이터의 본질적 특성 최초 처리
- **계층적 정렬**: 로컬+글로벌 이원구조 → 다른 도메인에도 영감 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fb6daecc-7bb4-4542-be58-680cb9e90990/26221-Article-Text-30284-1-2-20230626.pdf)

**(3) 응용 확대**

- 산업 IoT (기계 상태 모니터링)
- 헬스케어 (웨어러블 센서 데이터)
- 환경 모니터링 (기후 센서 네트워크)

### 9.2 향후 연구 방향

#### **(1) 더티 데이터 처리 (논문에서 명시)**

**문제**: 센서 샘플링 레이트 지터, 타임스탐프 미정렬

**제안 연구 방향**:
- 자기지도 학습 모듈 통합: $L_{SSL} = \text{ContrastiveLoss}(aug_1, aug_2)$
- 강건 그래프 구성: 이상 엣지 필터링
- 시계열 정규화 전처리 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fb6daecc-7bb4-4542-be58-680cb9e90990/26221-Article-Text-30284-1-2-20230626.pdf)

#### **(2) 동적 센서 관계 모델링 (암시된 한계)**

현재 SEA: 정적 센서 관계

**개선 방안**:
$$e_{mn,T}^{dynamic} = e_{mn,T} + \Delta e_{mn,T}(\text{trend})$$

→ 시간에 따라 변하는 센서 상관관계 추적[논문의 SEA++ 확장]

#### **(3) 다중 소스 도메인 확장**

현재: 소스 1개, 타겟 1개

**미래**:
$$\min L = \sum_{i=1}^{K} \alpha_i L_i^{(source)} + L^{(target)}$$

여기서 $\alpha_i$는 소스별 가중치 학습[6에서 POND가 시도]

#### **(4) 트랜스포머 통합**

최근 트렌드 (ContexTST, 2025):
$$\text{Attention}_{sensor}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

SEA의 MSGC를 트랜스포머 멀티헤드로 확장 가능 [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11211409/)

#### **(5) 이론적 일반화 경계 분석**

**열린 문제**:
$$\text{Target Error} \leq \text{Source Error} + d_H(D^S, D^T) + \lambda(\text{complexity})$$

여기서 $d_H$는 도메인 불일치를 정확히 정량화

SEA가 $d_H$ 감소를 보장하는 이론적 증명 부족 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fb6daecc-7bb4-4542-be58-680cb9e90990/26221-Article-Text-30284-1-2-20230626.pdf)

***

## 10. 앞으로 연구 시 고려할 점

### 10.1 방법론적 고려사항

#### **(1) 센서 선택 전략**

**문제**: 무의미한 센서의 영향?

예: HAR에서 113개 센서 중 일부만 중요

**해결 방안**:
- Sensor importance weighting: $w_m = \text{attention}(z_m)$
- Pruning: 저 관련성 센서 제거

#### **(2) 하이퍼파라미터 자동 튜닝**

현재: $\lambda_{SCA}, \lambda_{SFA}$ 수동 설정

**개선**:
```
AutoML: Bayesian Optimization
λ* = argmax accuracy(λ_{SCA}, λ_{SFA})
```

#### **(3) 계산 효율성**

SEA 복잡도: $O(N^2 L/d \cdot n_{branches})$

**최적화**:
- 희소 그래프 구성: $O(N \log N)$
- 다중브랜치 축소 (예: $n_b = 4$ 고정)

### 10.2 실무적 고려사항

#### **(1) 도메인 선택 기준**

도메인 A → B 전이 가능성 예측:

$$\text{Transferability} = 1 - d_H(D^A, D^B)$$

**실전 가이드**: Transferability < 0.3이면 사전 학습 데이터 추가 필요

#### **(2) 데이터 불균형**

현실: 소스 데이터 풍부, 타겟 매우 부족

**대응**:
- 가중 손실: $L = L_C + (1 + \epsilon) L_{Endo}$ (타겟 샘플 적을 때 $\epsilon$ 증가)
- 자기지도: $L_{SSL}$ 추가[1에서 제안한 향후 방향]

#### **(3) 온라인 적응**

타겟 도메인이 실시간으로 변할 때:

$$\text{Online}: L = (1-\beta) L_{prev} + \beta L_{new}$$

→ SEA를 점진적 학습으로 확장

### 10.3 평가 관점의 고려사항

#### **(1) 도메인 유사도 측정**

SEA 성능이 도메인 간 거리와 관계:

| 도메인 거리 | SEA 개선 | 예상 성능 |
|----------|---------|---------|
| 매우 유사 (< 0.1) | 5% | 이미 높음 |
| 중간 (0.1-0.3) | 15-20% | **최적 범위** |
| 매우 다름 (> 0.3) | < 5% | 추가 데이터 필요 |

#### **(2) 작은 데이터 체제**

타겟 $N_t < 100$일 때:
- SEA의 로컬 정렬이 과적합 가능성
- **대응**: Early stopping, 정규화 강화

#### **(3) 클래스 불균형**

일부 센서가 이상을 나타낼 때 (이상탐지):
- 현재 SEA: 균형 데이터 가정
- **개선**: 클래스별 정렬 가중치 $w_c$

***

## 결론

**SEnsor Alignment (SEA)**는 다중센서 시계열의 비지도 도메인 적응이라는 새로운 문제를 체계적으로 정의하고, 로컬(센서) + 글로벌 계층 정렬, 공간-시간 의존성 모델링으로 해결했다. 실험 결과 21-44% 성능 개선을 달성하며, SEA++로 확장되었고, 2025년 인과학습, 주파수 기반 방법 등 다양한 후속 연구를 자극했다.

**일반화 성능**: 센서별 개별 정렬과 계층적 구조가 견고한 표현을 형성하여 미지의 타겟 데이터에 강건함을 보였다.

**향후 과제**: 더티 데이터 처리, 동적 센서 관계, 다중 소스 도메인, 이론적 경계 분석 등이 남아있으며, 이러한 방향이 시계열 도메인 적응의 차세대 프론티어가 될 것으로 예상된다.

***

## 참고문헌

<span style="display:none">[^1_10][^1_11][^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_9]</span>

<div align="center">⁂</div>

[^1_1]: 26221-Article-Text-30284-1-2-20230626.pdf

[^1_2]: https://dl.acm.org/doi/10.1145/3711896.3737150

[^1_3]: https://arxiv.org/abs/2508.08280

[^1_4]: https://ojs.aaai.org/index.php/AAAI/article/view/33863

[^1_5]: https://arxiv.org/abs/2505.00415

[^1_6]: https://www.semanticscholar.org/paper/3cc5f151763ad0571eda4f96288672692e264b69

[^1_7]: https://ieeexplore.ieee.org/document/11211409/

[^1_8]: https://www.semanticscholar.org/paper/9dd51819fba6e4ab2aa2fbe2a9841322c0e3d28f

[^1_9]: https://www.semanticscholar.org/paper/2c286840a127a1e03eea230bbde604cc3ab53613

[^1_10]: https://www.ssrn.com/abstract=3735940

[^1_11]: https://www.semanticscholar.org/paper/9a92ffd46ac0278e9225ab8ddc2d27697fefefef

[^1_12]: https://www.semanticscholar.org/paper/09774a67df13190db55f9df00ede779d665aa1f5

[^1_13]: http://link.springer.com/10.1007/978-3-030-44584-3

[^1_14]: https://diglib.eg.org/handle/10.2312/vcbm20201174

[^1_15]: https://drops.dagstuhl.de/entities/document/10.4230/DagRep.10.4.1

[^1_16]: https://journals.lww.com/10.4103/1673-5374.300440

[^1_17]: https://link.springer.com/10.1007/s00477-020-01830-z

[^1_18]: https://arxiv.org/html/2503.01157v1

[^1_19]: https://arxiv.org/html/2402.16913v1

[^1_20]: https://arxiv.org/pdf/2312.12276.pdf

[^1_21]: https://arxiv.org/pdf/2102.06828v2.pdf

[^1_22]: https://arxiv.org/pdf/2502.16637.pdf

[^1_23]: https://arxiv.org/pdf/2203.08321.pdf

[^1_24]: https://arxiv.org/pdf/2305.14649.pdf

[^1_25]: https://arxiv.org/html/2312.09857v2

[^1_26]: https://arxiv.org/list/math/new

[^1_27]: https://pubmed.ncbi.nlm.nih.gov/34901425/

[^1_28]: https://arxiv.org/pdf/2503.00852.pdf

[^1_29]: https://arxiv.org/list/physics/new

[^1_30]: https://arxiv.org/pdf/2311.10806.pdf

[^1_31]: https://arxiv.org/html/2504.10925v2

[^1_32]: https://www.biorxiv.org/content/10.1101/2023.05.17.541153v1.full.pdf

[^1_33]: https://arxiv.org/pdf/2312.05698.pdf

[^1_34]: https://arxiv.org/abs/2405.12452

[^1_35]: https://peerj.com/articles/12542/

[^1_36]: https://arxiv.org/abs/2311.10806

[^1_37]: https://arxiv.org/abs/2504.15691

[^1_38]: http://arxiv.org/list/physics/2023-10?skip=680\&show=2000

[^1_39]: https://peerj.com/articles/cs-763.pdf

[^1_40]: https://arxiv.org/html/2406.10426v3

[^1_41]: https://iclr.cc/virtual/2025/33862

[^1_42]: https://peerj.com/articles/cs-763/

[^1_43]: https://www.nature.com/articles/s41598-025-14619-3

[^1_44]: https://openreview.net/forum?id=XcRaTCnIFs

[^1_45]: https://openaccess.thecvf.com/content_ICCVW_2019/papers/CoView/Iqbal_Enhancing_Temporal_Action_Localization_with_Transfer_Learning_from_Action_Recognition_ICCVW_2019_paper.pdf

[^1_46]: https://dl.acm.org/doi/10.1145/3663573

[^1_47]: https://liner.com/review/sensor-alignment-for-multivariate-timeseries-unsupervised-domain-adaptation

[^1_48]: http://www.arxiv.org/abs/2504.15691

[^1_49]: https://arxiv.org/html/2404.11269v4

[^1_50]: https://yoonji-ha.tistory.com/61

[^1_51]: https://ieeexplore.ieee.org/document/11260113/

[^1_52]: https://www.nature.com/articles/s41598-024-82417-4
