# Chronos2: From Univariate to Universal Forecasting

### **1. 핵심 주장 및 주요 기여**

Chronos-2는 Amazon Web Services 연구팀이 2025년 10월 발표한 시계열 예측 기초 모델(Foundation Model)로서, 기존 대부분의 사전학습 모델들이 단변량(univariate) 예측에만 국한된 근본적 한계를 극복하는 데 중점을 두고 있습니다.[1]

**핵심 혁신 사항:**

- **보편적 적용 가능성**: 단변량, 다변량, 공변량(covariate) 포함 예측을 동일한 아키텍처로 처리하는 유일한 제로샷(zero-shot) 모델
- **그룹 어텐션 메커니즘(Group Attention)**: 관련 시계열 간의 정보 공유를 효율적으로 수행
- **인콘텍스트 학습(ICL) 강화**: 배치 내 다양한 시계열 간 지식 전이를 활성화
- **합성 데이터 기반 다변량 학습**: Multivariatizer를 통해 단변량 데이터로부터 구조화된 다변량 데이터 생성

논문은 산업 현장에서 다변량 공변량 정보(예: 에너지 가격 예측 시 부하와 재생에너지 발전량, 소매 판매 예측 시 프로모션과 휴일 정보)의 중요성이 과거 모델들에 의해 간과되어 왔다는 점을 강조합니다.[1]

***

### **2. 해결하는 문제 및 기술적 방법론**

#### **2.1 문제 정의**

현존하는 사전학습 모델(Chronos-Bolt, TimesFM-2.5, Moirai-2.0 등)의 핵심 제약사항:

| 기능 | Chronos-2 | Toto-1.0 | Moirai-1.0 | TabPFN-TS | TimesFM-2.5 |
|------|-----------|----------|-----------|-----------|------------|
| 다변량 예측 | ✓ | ✓ | ✓ | ✗ | ✗ |
| 과거 공변량 | ✓ | ✓ | ✗ | ✗ | ✗ |
| 미래 공변량 | ✓ | ✗ | ✓ | ✓ | ✗ |
| 범주형 공변량 | ✓ | ✗ | ✗ | ✓ | ✗ |
| 교차학습(Cross-learning) | ✓ | ✗ | ✗ | ✗ | ✗ |
| 메모리 복잡도 | O(V) | O(V) | O(V²) | O(V) | - |

Chronos-2는 메모리 효율성( $O(V)$ )을 유지하면서 모든 예측 기능을 통합한 첫 모델입니다.[1]

#### **2.2 정규화 및 토큰화 (Tokenization)**

**Robust Scaling 공식:**

$$\tilde{v}_{t,d} = \sinh^{-1}\left(\frac{v_{t,d} - \mu_d}{\sigma_d}\right) \quad \text{for } t \in \{1, \ldots, T\}$$

$$\tilde{w}_{t,d} = \sinh^{-1}\left(\frac{w_{t,d} - \mu_d}{\sigma_d}\right) \quad \text{for } t \in \{T+1, \ldots, T+H\}$$

여기서 $\mu_d$, $\sigma_d$는 과거 관측치의 평균과 표준편차입니다. 이 sinh 역함수 변환은 극단값의 영향을 감소시켜 수렴을 안정화합니다.[1]

**메타 특성(Meta Features):**
- 시간 인덱스: $j = \left[-\frac{T}{C}, -\frac{T-1}{C}, \ldots, 0, \ldots, \frac{H-1}{C}\right]$
- 마스크: $m_d \in \{0,1\}$ (관측 여부 및 미래 공변량 구분)

#### **2.3 패칭 및 임베딩**

입력 $u_d$를 길이 $P$의 비중첩 패치로 분할하고, 메타 특성과 연결:

$$h_p = f_\phi^{\text{in}}\left(\left[u^p, j^p, m^p\right]\right)$$

여기서 $f_\phi^{\text{in}}: \mathbb{R}^{3P} \to \mathbb{R}^{D_{\text{model}}}$은 잔차 네트워크입니다.[1]

#### **2.4 핵심 아키텍처: Time & Group Attention**

**Time Attention 계층:**
자기 어텐션을 시간축에 적용하여 같은 시계열 내 패치 간 정보 집계

**Group Attention 계층 (혁신):**

그룹 ID 벡터 $g \in \mathbb{R}^B$에 기반한 2D 어텐션 마스킹을 통해 동일 그룹 내에서만 정보 교환:

$$\text{GroupAttn}(Q, K, V \mid g) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} \odot M_{g,g}\right)V$$

여기서 $M_{g,g}$는 그룹 일치성 마스크입니다.[1]

**그룹의 다양한 의미:**
- 단일 시계열 (최소 그룹화)
- 관련 시계열들의 집합 (교차학습 시 여러 제품의 판매량)
- 다변량 시계열의 변량들 (공유 역학 특성)
- 목표변수 + 과거 공변량 + 미래 공변량 (완전 예측 작업)

#### **2.5 손실 함수 및 학습**

**Quantile 회귀 목적함수:**

$$\sum_{q \in Q} \left[q \cdot \max(\hat{z} - z_q, 0) + (1-q) \cdot \max(z_q - \hat{z}, 0)\right]$$

21개 quantile 예측 ($Q = \{0.01, 0.05, 0.1, \ldots, 0.9, 0.95, 0.99\}$)으로 확률적 예측 제공.[1]

#### **2.6 역정규화 (Inference)**

$$\hat{y}_{t,d}^q = \mu_d + \sigma_d \cdot \sinh(\hat{z}_{t,d}^q)$$

정규화된 예측을 원래 스케일로 변환합니다.[1]

***

### **3. 모델 구조의 세부 설명**

#### **3.1 Encoder-Only Transformer 설계**

Chronos-2는 T5 인코더를 기반으로 하며, 다음 특징을 가집니다:

- **위치 임베딩**: RoPE (Rotary Position Embeddings) 사용
- **교대 계층**: Time Attention → Group Attention → Feed-Forward 구조 반복
- **특수 토큰**: REG 토큰 (구분자 및 어텐션 싱크 역할)
- **멀티헤드 구조**: 다양한 시간 및 그룹 차원에서의 관계 학습

#### **3.2 Quantile Head**

미래 패치의 모든 목표 변수를 단일 포워드 패스에서 처리하여 장기 예측 효율화:

$$\hat{Z} \in \mathbb{R}^{H \times D \times |Q|}$$

이를 통해 반복 예측의 오차 누적(error accumulation) 문제를 완화합니다.[1]

#### **3.3 학습 파이프라인 (2단계)**

**Stage 1**: 컨텍스트 길이 2,048, 낮은 출력 패치 수

**Stage 2**: 컨텍스트 길이 8,192로 확장, 높은 출력 패치 수로 장기 계절성 학습

이는 고주파 데이터의 긴 계절성(예: 24시간 주기의 시간별 데이터)을 효과적으로 포착합니다.[1]

***

### **4. 성능 향상의 실증 분석**

#### **4.1 벤치마크 결과**

**fev-bench (100개 작업, 다변량 & 공변량 강조):**

$$\text{Win Rate} = 90.7\% \quad \text{(Skill Score: 47.3\%)}$$

95% 신뢰 구간에서 TiRex (80.8%, 42.6%), TimesFM-2.5 (75.9%, 42.3%)를 통계적으로 유의미하게 초과.[1]

**GIFT-Eval (97개 작업, 고주파 & 장기 예측):**

가중 정량 손실(WQL)에서 81.9% 승률, 51.4% 스킬 스코어

**Chronos Benchmark II (27개 작업, 짧은 컨텍스트):**

79.8% 승률, 46.6% 스킬 스코어로 모든 기존 모델 초과[1]

#### **4.2 인콘텍스트 학습의 일반화 이득**

**단변량 작업에서의 교차학습 효과:**

Chronos Benchmark II에서 최대 ~1.0% 포인트 스킬 점수 향상 (짧은 컨텍스트에서 특히 강함)[1]

**다변량 작업:**

직관적으로 더 큰 이득이 예상되지만, 실제로는 ~0.3% 포인트 modest gain만 관찰

→ Takens 임베딩 정리의 시사점: 충분히 긴 단변량 기록으로도 시스템 역학 대부분 복구 가능[1]

**공변량 작업 (최대 이득):**

fev-bench 공변량 부분집합에서 **7.0% 포인트 스킬 점수 향상** (40.0% → 47.0%)

에너지 도메인: 51.3% (ICL) vs 46.5% (단변량)
소매 도메인: 48.6% (ICL) vs 44.3% (단변량)[1]

***

### **5. 모델의 일반화 성능 향상 가능성 (핵심 분석)**

#### **5.1 합성 데이터의 역할**

Chronos-2의 다변량/공변량 기능은 **전적으로 합성 데이터**에 의존합니다. 절제 연구(ablation study) 결과:

| 학습 데이터 | fev-bench | GIFT-Eval | Chronos II |
|-----------|-----------|-----------|-----------|
| 실제 + 합성 | 47.3% | 51.4% | 46.6% |
| 합성만 (Chronos-2-Synth) | 45.9% | 50.4% | 46.4% |
| **성능 저하** | **-1.4pp** | **-1.0pp** | **-0.2pp** |

→ 합성 데이터 접근의 유효성을 증명하며, 실제 데이터가 선택적일 수 있음을 시사합니다.[1]

#### **5.2 Multivariatizer 설계의 중요성**

**공시적 다변량화(Cotemporaneous)**: 동시점 선형/비선형 변환으로 순간 상관성 유도

**순차적 다변량화(Sequential)**: 시간축 의존성 생성 (래그-리드 관계, 공적분)

이 두 메커니즘이 조합될 때 가장 다양한 다변량 구조를 생성합니다.[1]

#### **5.3 모델 크기와 효율성**

28M 파라미터 소형 모델 실적:

| 벤치마크 | 기본(120M) | 소형(28M) | 저하 |
|---------|----------|----------|------|
| fev-bench | 47.3% | 45.3% | 2.0pp |
| GIFT-Eval | 51.4% | 50.4% | 1.0pp |
| Chronos II | 46.6% | 44.1% | 2.5pp |

→ 약 75% 파라미터 감소로 2-2.5% 성능 저하만 발생, 즉 **리소스 제약 환경에서의 실현 가능성 높음**[1]

#### **5.4 장시간 컨텍스트 후훈련의 영향**

8,192 길이 확장 후훈련 vs 2,048 기본:

GIFT-Eval에서 **0.3pp** 향상 (고주파 데이터의 장기 계절성 학습)

→ 아키텍처 개선보다는 **데이터 규모와 다양성**이 일반화 성능의 주요 결정요인임을 시사합니다.[1]

***

### **6. 한계 및 제약사항**

#### **6.1 다변량 모델링의 한계**

다변량 작업에서 단변량 모드와의 성능 차이가 미미한 것은:

- **채널 독립성(Channel-independence) 가정의 강건성**: 최근 연구(PatchTST, iTransformer)가 채널별 독립적 처리의 효과성을 보여줌
- **제한된 동적 상관성**: 합성 데이터 생성 방식이 실제 산업의 복잡한 다변량 의존성을 완전히 포착하지 못할 가능성[1]

#### **6.2 공변량 유형의 제한**

- **텍스트 입력 미지원**: 뉴스, 이벤트 설명 등 비정형 정보 활용 불가
- **범주형 변수의 처리**: 다변량 목표의 경우 서수 인코딩(ordinal encoding)에 의존 (정보 손실 위험)[1]

#### **6.3 극단값(극한 분위수) 처리**

0.01, 0.99 분위수 예측 추가로 희귀 사건 커버리지 개선하지만:
- 분포의 극단부 모델링 부정확성
- 이상 탐지 작업에서의 제한적 성능[1]

#### **6.4 계산 오버헤드**

- GPU 배치 처리 기반: 단일 시계열의 실시간 예측에서 지연
- Group attention의 배치 크기 의존성: 그룹 크기가 불균형할 때 메모리 비효율[1]

***

### **7. 앞으로의 연구에 미치는 영향**

#### **7.1 향후 아키텍처 설계의 방향**

1. **멀티모달 확장**: 시계열 + 텍스트 + 카테고리 변수의 통합 학습 프레임워크
   - 예: 소매 수요 예측에 소셜 미디어 감정, 뉴스 이벤트 통합

2. **검색-기반 인콘텍스트 학습(Retrieval-augmented forecasting)**
   - Group attention의 유연성을 활용하여 유사 시계열을 자동 선택
   - 메타데이터 기반 희소 그룹핑으로 확장성 개선[1]

3. **적응형 정규화(Adaptive normalization)**
   - sinh 변환 외 도메인별 최적 정규화 학습
   - 계절성, 추세, 이상치 프로파일에 따른 동적 스케일링[1]

#### **7.2 데이터 생성 및 합성의 진화**

1. **인과 기반 합성 데이터**: TCM 확장으로 실제 경제 인과관계 모델링
2. **도메인별 특화 생성기**: 에너지, 금융, 의료 각 분야의 특성 반영
3. **메타 학습**: 합성 데이터 생성 알고리즘 자체를 학습[1]

#### **7.3 실무 배포 전략**

1. **Cold-start 문제 해결**: 짧은 이력의 신규 제품/지점에서의 성능 향상
   - TimesFM-ICF의 in-context fine-tuning 아이디어 통합
   
2. **도메인 적응(Domain adaptation)**
   - 에너지, 금융, 소매 등 산업별 소규모 미세조정 데이터셋 개발
   - Transfer learning의 효율적 체계화[2][1]

3. **불확실성 정량화**: 95% 신뢰 구간 예측의 정확성 검증
   - 극단 분위수 보정 메커니즘[1]

#### **7.4 이론적 기여**

1. **Takens 임베딩과 현대 DL의 관계**: 
   - 다변량 vs 단변량의 성능 수렴 현상에 대한 수학적 분석
   - 언제 다변량 모델링이 필수인지의 이론적 기준 제시[1]

2. **그룹 주의의 일반화 특성**:
   - 배치 내 다중 분포(multi-distribution) 학습의 일반화 경계
   - 인콘텍스트 학습의 샘플 복잡도 분석[1]

***

### **8. 2020년 이후 관련 최신 연구 비교 분석**

#### **8.1 시계열 기초 모델의 진화 타임라인**

| 연도 | 모델 | 주요 혁신 | 한계 |
|------|------|---------|------|
| 2023 | N-BEATS+ (추적 불가) | 통계 기반 결합 | 학습 기반 접근 없음 |
| 2024-Q1 | TimeGPT, TimesFM | 대규모 사전학습, 단변량 강화 | 공변량 미지원 |
| 2024-Q2 | Moirai-1.0 | 다변량 자기주의 도입 | 플래팅으로 인한 확장성 문제 |
| 2024-Q3 | Toto-1.0 | 교차 변량 주의, 공변량 초기 지원 | 미래 공변량 미지원 |
| 2024-Q4 | Time-MoE, TiRex | 스케일 확대, 향상된 ICL | 메모리 효율성 한계 |
| 2025-Q1 | **Chronos-2** | **통합 O(V) 아키텍처** | 텍스트 공변량 미지원 |
| 2025-Q2 | TimesFM-ICF | In-context fine-tuning | 런타임 길이 불변 |

#### **8.2 핵심 아키텍처 비교**

**TimesFM (Google Research, 2024):**
- 200M 파라미터, 100B time-points 학습
- 디코더만(decoder-only) 구조
- **장점**: 빠른 추론 (배치당 1.4초 vs Chronos-2의 3.6초)
- **한계**: 공변량 지원 없음, ICL 제한적[3]

**Moirai-1.0 (Salesforce, 2024):**
- 다변량 시계열의 동시 모델링 시도
- 내부 플래팅(flattening): D개 변량을 단일 시퀀스로 변환
- **문제**: 메모리 복잡도 O(V²), 고차원에서 비효율[1]

**Toto-1.0 (Anthropic, 2025):**
- 교차-변량(cross-variate) 자기주의 도입
- 과거 공변량만 지원
- **제약**: 미래 공변량(배운 예보자 값 등)이 실제로 중요한 경우 미지원[1]

**TiRex (2025):**
- 향상된 in-context learning
- 긴 시간대 vs 짧은 시간대 균형
- **성능**: fev-bench에서 80.8% 승률 (Chronos-2의 90.7%에 미미)[1]

#### **8.3 학습 데이터 전략의 진화**

| 모델 | 실제 데이터 | 합성 데이터 | 혼합 비율 |
|------|----------|----------|---------|
| TimeGPT | 주로 | - | 미공개 |
| TimesFM | 100B points | 적음 | ~90% 실제 |
| Moirai | 선택적 | 일부 | 미공개 |
| **Chronos-2** | **선택적** | **다변량은 100%** | **도메인별** |

Chronos-2의 혁신: **실제 데이터의 가용성이 다변량 예측 성능에 미치는 영향 최소화**[1]

#### **8.4 문제별 성능 분석**

**공변량 관련 작업 (가장 차별화되는 영역):**

| 모델 | 스킬 스코어 |
|------|-----------|
| Chronos-2 | **47.0%** |
| TabPFN-TS | 40.9% |
| TiRex | 38.7% |
| Toto-1.0 | 35.1% |
| 통계 앙상블 | 19.7% |

**7% 포인트 이상의 격차**는 공변량 활용에서 Chronos-2의 우월성을 명확히 증명합니다.[4][1]

#### **8.5 In-Context Learning (ICL) 패러다임의 변화**

**TimesFM (2024) → TimesFM-ICF (2025)의 발전:**

TimesFM-ICF는 계속 사전학습을 통해 배치 내 관련 예시에서 학습하는 능력을 추가하여, **지도 미세조정 없이 거의 동등한 성능** 달성. 이는 Chronos-2의 그룹 주의 기반 접근보다는 **다른 패러다임**이지만, 같은 목표(산업 배포의 단순화)를 추구합니다.[5][2]

**Chronos-2의 우위:**
- 구조적으로 통합된 다중 작업 지원 (별도 학습 불필요)
- 메모리 효율 O(V) vs TimesFM-ICF의 시퀀스 길이 배가

***

### **9. 향후 연구 시 고려사항**

#### **9.1 방법론적 고려**

1. **통제 실험 설계**
   - 공변량의 정보량(정보 게인 정량화) 측정
   - 다변량 동적의 강도(correlation strength)에 따른 성능 곡선 분석

2. **도메인 특수성 연구**
   - 금융(높은 자기상관) vs 에너지(강한 계절성) vs 소매(공변량 지배적)
   - 각 분야에서의 최적 모델 크기, 컨텍스트 길이

3. **적대적 평가(Adversarial evaluation)**
   - 분포 외(out-of-distribution) 공변량에 대한 robustness
   - 공변량이 실제로는 비정보적인 경우의 과적합[1]

#### **9.2 기술적 개선 방향**

1. **동적 그룹핑**: 
   - 메타데이터나 임베딩 기반으로 그룹 할당 자동화
   - 런타임에 그룹 구조 조정[1]

2. **희소 주의(Sparse Attention)**:
   - 매우 큰 배치(예: 10,000+ 시계열)에서의 O(V) 메모리 유지
   - 블록 희소(block-sparse) 그룹 주의[6]

3. **적응형 분위수 선택**:
   - 고정 21개 분위수 대신 작업 난이도에 따른 동적 선택
   - 극단값 예측 정확도 향상[1]

#### **9.3 평가 지표의 재고**

1. **보정(Calibration) 메트릭**:
   - 분위수 예측의 확률적 보정 지수(PIT, probability integral transform)
   - 실제 커버리지 vs 명목 신뢰도 비교

2. **인과 평가**:
   - 공변량이 실제로 예측에 사용되었는지 확인 (주의 가중치 분석)
   - 허위 공변량에 대한 모델 감수성[1]

3. **계산 복잡도-성능 트레이드오프**:
   - 파레토 최적선 도출 (지연 vs 정확성)
   - 배포 시나리오별 최적 모델 선택 기준[1]

***

### **10. 결론 및 임상적 의의**

**Chronos-2의 위치:**

Chronos-2는 시계열 기초 모델 분야에서 **"보편성(Universality)" 패러다임의 실현**으로 평가됩니다. 단순히 다변량과 공변량을 추가로 지원하는 것이 아니라:

1. **통일된 아키텍처**: 모든 예측 작업을 구조적으로 처리 → 배포 복잡도 극적 감소
2. **확장 가능한 설계**: O(V) 메모리로 고차원 시나리오 수용
3. **합성 데이터의 유효성 입증**: 실제 다변량 데이터의 부족을 극복 가능

**산업 영향:**

- **에너지**: 부하 + 기후 공변량의 통합으로 30일 선도 시간 예측 정확도 향상
- **소매**: 프로모션 효과의 정량화를 통한 동적 가격 책정 최적화
- **금융**: 다자산 포트폴리오 리스크 관리 자동화
- **클라우드**: CPU/메모리/I/O의 동시 예측으로 리소스 할당 효율화[1]

**미해결 과제:**

- 극단 분위수의 정확도 개선 필요
- 텍스트/이미지 공변량의 통합
- 적응형 알고리즘(배치 크기, 시간대에 따른 자동 조정)

**5년 전망:**

현재 추세(TimesFM-ICF, TiRex)와 Chronos-2의 결합은 **"멀티모달 기초 모델"** 시대를 열 것으로 예상됩니다. 여기서 시계열은 텍스트, 이미지, 정형 데이터와 통합되어 진정한 "보편적" 예측 인공지능을 구성할 것입니다.

***

**참고 자료:**

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/603eb397-8660-48ea-9c6b-49f19fb8b7ff/2510.15821v1.pdf)
[2](https://research.google/blog/time-series-foundation-models-can-be-few-shot-learners/)
[3](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/)
[4](https://arxiv.org/abs/2406.16964)
[5](https://research.google/pubs/in-context-finetuning-for-time-series-foundation-models/)
[6](https://arxiv.org/html/2408.04245v1)
[7](https://ieeexplore.ieee.org/document/11137629/)
[8](https://ieeexplore.ieee.org/document/11050326/)
[9](https://arxiv.org/abs/2507.02907)
[10](https://arxiv.org/abs/2402.07570)
[11](https://www.mdpi.com/2673-4591/68/1/1)
[12](https://dl.acm.org/doi/10.1145/3637528.3671855)
[13](https://arxiv.org/abs/2405.14252)
[14](https://rjor.ro/considerations-on-the-efficiency-of-time-series-analysis-in-forecasting-new-influenza-cases-in-the-2024-2025-season/)
[15](https://ieeexplore.ieee.org/document/11207370/)
[16](http://arxiv.org/pdf/2410.15217.pdf)
[17](https://arxiv.org/pdf/2402.16516.pdf)
[18](https://arxiv.org/html/2410.11674)
[19](https://arxiv.org/pdf/2310.10688.pdf)
[20](http://arxiv.org/pdf/2308.08469.pdf)
[21](https://arxiv.org/html/2503.22747v1)
[22](http://arxiv.org/pdf/2310.04948.pdf)
[23](https://arxiv.org/pdf/2403.05798.pdf)
[24](https://www.scitepress.org/Papers/2025/133638/133638.pdf)
[25](https://pmc.ncbi.nlm.nih.gov/articles/PMC9026292/)
[26](https://arxiv.org/html/2401.00230v1)
[27](https://icml.cc/virtual/2025/poster/43518)
[28](https://arxiv.org/html/2401.00230v2)
[29](https://arxiv.org/abs/2410.24087)
[30](https://github.com/google-research/timesfm/)
[31](https://huggingface.co/blog/informer)
[32](https://openreview.net/forum?id=uxzgGLWPj2)
[33](https://github.com/amazon-science/chronos-forecasting)
[34](https://orbilu.uni.lu/bitstream/10993/59995/1/AAAI_AI4TS_2024_final_JX.pdf)
[35](https://arxiv.org/html/2410.24087v1)
[36](https://openreview.net/forum?id=Z5FJsp1U3Z&noteId=5N5JjGUW0m)
[37](https://github.com/thuml/iTransformer)
[38](https://openreview.net/forum?id=ryIHtXE9uG)
[39](https://arxiv.org/html/2506.06005v1)
[40](https://github.com/mkdirer/Multivariate-Time-Series-Forecasting-Using-Transformers)
[41](https://arxiv.org/html/2507.02907v1)
[42](https://arxiv.org/html/2512.07705v1)
[43](https://arxiv.org/html/2501.06386v1)
[44](https://arxiv.org/abs/2408.16896)
[45](https://arxiv.org/abs/2308.08469)
[46](https://arxiv.org/abs/2308.09884)
[47](https://arxiv.org/abs/2509.23695)
[48](https://arxiv.org/html/2508.07697v3)
[49](https://arxiv.org/abs/2512.07705)
[50](https://arxiv.org/html/2508.19609v1)
[51](https://arxiv.org/abs/2010.02803)
[52](https://arxiv.org/abs/2511.15447)
[53](https://arxiv.org/abs/2409.16040)
[54](https://arxiv.org/abs/2510.07084)
[55](https://arxiv.org/abs/2511.19694)
[56](https://arxiv.org/html/2509.00616v1)
[57](https://arxiv.org/abs/2505.00307)
[58](https://premierscience.com/pjs-25-1179/)
[59](https://www.sciencedirect.com/science/article/pii/S1566253525003203)
