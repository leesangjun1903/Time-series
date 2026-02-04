
# Time-Series Forecasting for Out-of-Distribution Generalization Using Invariant Learning

## 1. 핵심 주장과 주요 기여 요약

이 논문은 **FOIL(Forecasting for Out-of-distribution TS generalization via Invariant Learning)** 프레임워크를 제안합니다. 핵심 주장은 시계열 예측(TSF) 모델이 역사적 훈련 데이터와 미래 테스트 데이터의 분포 불일치에 대응하기 위해 불변학습(Invariant Learning, IL)을 적용해야 한다는 것입니다.

주요 기여는 다음과 같습니다:

- **OOD-TSF 문제 공식화**: 시계열 예측에서 OOD 일반화를 다룬 첫 연구
- **불변학습 적용의 핵심 도전 식별**: 미관측 변수(unobserved variables)로 인한 충분성 가정 위반과 환경 레이블 부재 문제 명시적 제시
- **FOIL 프레임워크**: 모델 불가지론적(model-agnostic) 접근으로 다양한 백본 모델 지원
- **경험적 성과**: 최대 85% 성능 향상 달성

***

## 2. 해결하고자 하는 문제

### 2.1 문제의 본질

시계열 데이터는 동적 특성으로 인해 **분포 시프트(distribution shift)**를 겪습니다. 기존 모델들은 경험적 위험 최소화(ERM)를 사용하여 훈련 데이터의 모든 상관관계를 포착하지만, 이 중 많은 상관관계가 테스트 분포에서 유지되지 않아 OOD 일반화 능력이 부족합니다.

$$P_{\text{train}}(X, Y) \neq P_{\text{test}}(X, Y)$$

### 2.2 불변학습 직접 적용의 문제

**첫 번째 도전**: 미관측 변수 문제

기존 불변학습은 다음 가정을 전제합니다:

$$Y = g(X_I) + \epsilon$$

여기서 $X_I$는 불변 특성(invariant features)이고, Y는 $X_I$에 의해 충분히 결정됩니다. 그러나 시계열 데이터에는 항상:

$$Y = q(Y_{\text{suf}}, Z)$$

로 표현되는 **미관측 핵심 변수** Z(유행병 발생, 정책 변화 등)가 존재하므로 충분성 가정이 위반됩니다.

**두 번째 도전**: 환경 레이블 부재

시계열 데이터는 명시적 환경 레이블이 없고, 기존 환경 추론 방법들(HRM, EIIL, ZIN 등)은 고차원 입력과 시계열 특성을 고려하지 않습니다.

***

## 3. 제안하는 방법과 모델 구조

### 3.1 핵심 아이디어: Sufficiently Predictable 부분에 IL 적용

FOIL의 핵심은 **전체 Y가 아닌 Y_suf(충분히 예측가능한 부분)**을 대상으로 불변학습을 수행하는 것입니다:

$$Y = q(Y_{\text{suf}}, Z) = \alpha(Z)(Y_{\text{suf}}) + \beta(Z)\mathbf{1}$$

여기서 $Y_{\text{suf}}$는 입력 X에 의해 결정되며, Wold 분해 정리에 의해 다음을 따릅니다.

### 3.2 FOIL의 세 가지 핵심 모듈

#### (1) Label Decomposing Component (CLD)

미관측 변수 Z의 영향을 완화하기 위해 **Instance Residual Normalization (IRN)**을 도입:

$$\tilde{Y}_t = \frac{Y_t - \mu(Y_t)}{\sigma(Y_t)}, \quad \tilde{\hat{Y}}_t = \frac{\hat{Y}_t - \mu(\hat{Y}_t)}{\sigma(\hat{Y}_t)}$$

$$\tilde{\text{Res}}_t = \tilde{Y}_t - \tilde{\hat{Y}}_t$$

**Surrogate Loss**(대리 손실)는 다음과 같이 정의됩니다:

$$\ell_{\text{suf}}(\hat{Y}, Y) = \text{MSE}(\tilde{\text{Res}}, 0) = \frac{1}{h}\sum_{j=1}^{h}(\tilde{\text{Res}}_{t+j})^2$$

이는 기존 정규화 방법과 다르게 Y와 $\hat{Y}$ 간 평균과 분산을 직접 정렬하여 Z의 영향을 제거합니다.

#### (2) Time-Series Environment Inference Module (MTEI)

**EM 알고리즘** 기반으로 환경을 자동 추론하되, **시간 인접성 구조**를 보존합니다.

**M-Step** - 환경별 회귀기 최적화:

$$\min_{\{\rho^{(e)}\}} L_{\text{TEI}} = \sum_{e \in E_{\text{infer}}} \frac{1}{|D_e|}\sum_{(X,Y) \in D_e} \ell_{\text{suf}}(\rho^{(e)}(\phi^*(X)), Y)$$

**E-Step** - 두 단계 환경 레이블 재할당:

**Step 1** (거리 기반 재할당):
$$E_{\text{infer}}(t) \leftarrow \arg\min_{e} \ell_{\text{suf}}(\rho^{(e)}(\phi^*(X_t)), Y_t)$$

**Step 2** (시간 인접성 보존):
$$E_{\text{infer}}(t) \leftarrow \text{mode}\{E_{\text{infer}}(t+j)\}_{j=-r}^{r}$$

시간적으로 인접한 인스턴스가 유사한 환경을 갖도록 하는 **라벨 전파(label propagation)** 메커니즘으로 오버피팅 방지.

#### (3) Time-Series Invariant Learning Module (MTIL)

추론된 환경 간에서 불변 표현을 학습합니다. 이론적 목표:

$$\phi^* = \arg\max_{\phi} I(Y_{\text{suf}}; \phi(X)) - I(Y_{\text{suf}}; E^*_{\text{learn}} | \phi(X))$$

첫 번째 항은 **충분성**(sufficiency)을 보장(상호정보량 최대화), 두 번째 항은 **불변성**(invariance)을 보장(조건부 독립성).

실제 구현 손실함수:

$$L_{\text{TIL}} = \underbrace{\mathbb{E}_{e \in E^*_{\text{infer}}} R^{(e)}_{\text{suf}}(\rho, \phi)}_{\text{Y}_{\text{suf}} \text{ 충분성}} + \underbrace{\lambda_1 R_{\text{ERM}}(\rho, \phi)}_{\text{원본 Y 기반}} + \underbrace{\lambda_2 \text{Var}_{e \in E^*_{\text{infer}}}\{R^{(e)}_{\text{suf}}(\rho, \phi)\}}_{\text{불변성}}$$

여기서:
- 첫 번째 항: 각 환경에서의 손실 (Y_suf 기반)
- 두 번째 항: ERM 손실 - μ(Y_suf), σ(Y_suf)의 영향 조절 (λ₁로 균형)
- 세 번째 항: 환경 간 손실 분산 - 불변성 강제 (λ₂로 조절)

### 3.3 통합 모델 구조

$$f_\theta(X) = \rho(\phi(X))$$

여기서 $\phi(\cdot)$는 백본 모델(Informer, Crossformer, PatchTST 등), $\rho(\cdot)$는 최상위 회귀기입니다. MTIL과 MTEI는 $\phi(X)$ 표현 공간에서 작동하므로 **모델 불가지론적** 특성을 유지합니다.

***

## 4. 성능 향상 및 실험 결과

### 4.1 실험 설정

| 항목 | 상세 |
|------|------|
| **데이터셋** | Exchange (일일 환율), ILI (주간 독감), ETTh1/ETTh2 (시간별 변압기 온도) |
| **백본 모델** | Informer, Crossformer, PatchTST |
| **평가지표** | MSE, MAE |
| **지평선 길이** | 24 ~ 720 (데이터셋별 상이) |

### 4.2 주요 성과 (표 1)

| 백본 | 데이터셋 | FOIL 개선율(%) |
|------|---------|----------------|
| **Informer** | Exchange | 80.58% (MSE), 61.24% (MAE) |
| | ILI | 75.80% (MSE), 57.37% (MAE) |
| | ETTh1 | 85.38% (MSE), 68.14% (MAE) |
| | ETTh2 | 78.03% (MSE), 58.82% (MAE) |
| **Crossformer** | Exchange | 61.90% (MSE), 45.06% (MAE) |
| | ILI | 79.99% (MSE), 64.03% (MAE) |
| **PatchTST** | Exchange | 30.60% (MSE), 21.11% (MAE) |

특히 ETTh1(Informer)에서 **85% MSE 개선**은 FOIL의 놀라운 효과를 보여줍니다.

### 4.3 기존 방법과의 비교 (표 2)

16개 기준선 대비 FOIL의 성능:

| 비교 대상 | FOIL 개선율 |
|----------|-----------|
| ERM (기본) | 평균 10% 이상 |
| 기존 OOD 방법 (GroupDRO, IRM, VREx 등) | 15~20% 이상 |
| 하이브리드 방법 (IRM+RevIN, EIIL+RevIN) | 11% 이상 |

특히 일반 OOD 방법들의 낮은 성능(IRM 등)은 직접 적용의 부적절성을 입증합니다.

### 4.4 Ablation Study (그림 3a)

| 모델 변형 | 성능 저하 |
|----------|---------|
| FOIL \ Suf (대리손실 제거) | **가장 큼** - 미관측 변수 처리의 중요성 증명 |
| FOIL \ TEI (환경 추론 제거) | 큼 |
| FOIL \ LP (라벨 전파 제거) | 중간 |

각 모듈이 독립적으로 기여함을 확인.

### 4.5 추론된 환경의 타당성 (그림 3b)

ILI 데이터셋 사례연구: 2개 환경 추론 결과

- **환경 1**: 겨울과 H1N1-09 기간이 주요 구성 → 독감 계절성 반영
- **환경 2**: 여름 기간이 주도적 → 정상 계절 변화 반영

추론된 환경이 도메인 전문 지식과 일치하는 의미 있는 환경임을 보여줍니다.

***

## 5. 일반화 성능 향상 가능성 (중점)

### 5.1 일반화 개선의 메커니즘

**1. 인과적 특성 학습**
- 불변 특성 $X_I$는 Y와의 인과 관계를 유지 ($X_I \to Y$)
- 변이 특성은 환경의 영향으로 인한 허위 상관 제거 ($X_V \leftarrow E \to X_I \to Y$)

**2. 미관측 변수의 영향 완화**
- Surrogate loss를 통해 $\mu(Y_{\text{suf}}), \sigma(Y_{\text{suf}})$만 학습
- Z의 불확실한 영향은 버리고 안정적인 패턴만 보존

**3. 환경별 다양성 활용**
- 다양한 추론된 환경에서 불변성 검증
- 모든 환경에서 성능 저하 최소화하는 robust 표현 학습

### 5.2 단기 vs 장기 예측의 차이

| 예측 기간 | FOIL 효과 |
|----------|----------|
| **단기 (24~96 단계)** | 우수 (30~85% 개선) |
| **장기 (336~720 단계)** | 중간 (10~30% 개선) |

**이유**: 장기 예측은 미관측 변수의 영향이 커져 $Y_{\text{suf}}$의 비중 감소.

### 5.3 OOD 성능의 이론적 보장

불변성 원칙(Invariance Principle)에 기반:

$$\text{최소 환경 위험} \leq \text{임의의 미지 테스트 위험}$$

FOIL은 최악의 성능을 최소화하는 **worst-case 학습**을 수행하여 OOD 일반화를 보장합니다.

***

## 6. 모델의 한계

### 6.1 방법론적 한계

1. **미관측 변수의 부분적 처리만 가능**
   - IRN은 가정 기반의 휴리스틱으로, Z를 완전히 제거하지 못함
   - 복잡한 비선형 미관측 변수에 대해 제한적

2. **환경 추론의 정확도 문제**
   - EM 알고리즘이 지역 최솟값에 빠질 수 있음
   - 자동 추론된 환경이 진정한 causal 환경과 불일치 가능성
   - 환경 개수 k의 사전 선택 필요 (튜닝 매개변수 2~10)

3. **장기 예측의 약화**
   - 불확실성이 증가할수록 불변 특성 학습 어려움
   - $Y_{\text{suf}}$의 비중이 급격히 감소

### 6.2 기술적 제약

1. **계산 비용**
   - 다중 회귀기 $\{\rho^{(e)}\}$로 인한 매개변수 수 증가
   - EM 반복으로 학습 시간 증가

2. **확장성 문제**
   - 고차원 데이터나 매우 큰 시계열에 대한 성능 평가 제한적
   - 환경 개수에 따른 선형적 복잡도 증가

3. **데이터 요구사항**
   - 충분한 환경적 다양성 필요
   - 작은 데이터셋에서 과적합 위험

### 6.3 가정의 제약

- 방정식 (2)의 가정: $Y = \alpha(Z)(Y_{\text{suf}}) + \beta(Z)\mathbf{1}$ 
  - 시계열 전체에 걸친 Z의 일정한 영향 가정
  - 시간 가변 미관측 변수에 제한적

***

## 7. 2020년 이후 관련 최신 연구 비교 분석

### 7.1 시간 분포 이동 처리 방법들

#### RevIN (Kim et al., 2021, ICLR)
- **접근**: 인스턴스 정규화로 입력 비정상성 처리
- **특징**: 간단하고 효율적, 하지만 입력만 처리
- **FOIL 대비**: RevIN은 출력 Y의 미관측 변수 영향을 다루지 않음

#### NST (Liu et al., 2022, NeurIPS)
- **접근**: Non-Stationary Transformer로 평균/분산 변화 모델링
- **특징**: 적응형 정규화, 모델 내재화
- **FOIL 대비**: 모델 특화, IL의 causal 원칙 미적용

#### Dish-TS (Fan et al., 2023, AAAI)
- **접근**: 입력/출력 공간 분포를 분리하여 학습
- **특징**: 마진 및 조건부 분포 모두 처리
- **FOIL 대비**: OOD 특화 아님, 환경 개념 미포함

### 7.2 OOD 일반화 관련 연구들

#### DIVERSIFY (Lu et al., 2023, ICLR)
- **목표**: 시계열 분류/탐지를 위한 OOD 학습
- **특징**: 적대적 훈련으로 최악의 잠재 분포 식별
- **FOIL 대비**: 분류 작업 중심, 예측 아님; 환경 추론 미포함

#### ShifTS (Zhao et al., 2025, ICLR 제출)
- **목표**: 개념 드리프트 + 시간 이동 동시 처리
- **특징**: 소프트 어텐션으로 불변 패턴 추출
- **FOIL 대비**: 개념 드리프트 명시적 처리 추가; IL 기반 아님
- **상보성**: FOIL의 불변학습 + ShifTS의 개념 드리프트 결합 가능

#### ContexTST (Liu et al., 2025, arXiv)
- **목표**: Cross-domain 시계열 예측
- **특징**: Mixture-of-Experts + context-aware transformer
- **FOIL 대비**: 도메인 전이 중심, OOD 시프트 일반화 아님

#### TFPS (Sun et al., 2024, arXiv)
- **목표**: 패치 수준 분포 변화 처리
- **특징**: 패턴별 전문가 학습, 시간/주파수 이중 인코더
- **FOIL 대비**: IL 기반 아님, 하지만 패턴 다양성 보존에 유사

#### APT (Li et al., 2025, arXiv)
- **목표**: 분포 시프트 하 예측
- **특징**: 타임스탐프 조건 프로토타입 학습으로 전역 분포 특성 주입
- **FOIL 대비**: 경량 플러그인, 초매개변수 자동 생성 가능

### 7.3 FOIL의 위치

```
────────────────────────────────────────────────────────────
                    시계열 예측 방법 분류
────────────────────────────────────────────────────────────

1. 비정상성 처리 (RevIN, NST)
   └─ 통계적 접근, 모델 내재화 가능

2. 분포 분리 학습 (Dish-TS)
   └─ 입출력 공간 처리, 구조적 접근

3. OOD 일반화 (DIVERSIFY, ShifTS, APT)
   ├─ 적대적 훈련
   ├─ 소프트 어텐션
   └─ 프로토타입 학습

4. **IL 기반 OOD (FOIL)** ⭐
   ├─ Causal 원칙 적용
   ├─ 미관측 변수 명시적 처리
   ├─ 환경 자동 추론
   └─ 모델 불가지론적

5. 도메인 적응 (ContexTST)
   └─ Cross-domain 전이
────────────────────────────────────────────────────────────
```

### 7.4 성능 비교 요약

| 방법 | 비정상성 | 환경 추론 | IL 기반 | 미관측 변수 | 다중 백본 | 성능 개선 |
|------|---------|----------|--------|-----------|---------|---------|
| RevIN | ✓ | ✗ | ✗ | ✗ | ✓ | 중간 |
| NST | ✓ | ✗ | ✗ | ✗ | ✗ | 중간 |
| Dish-TS | ✓ | ✗ | ✗ | ✗ | ✓ | 중간 |
| DIVERSIFY | ✗ | ✓ | ✗ | ✗ | ✗ | 중간 |
| ShifTS | ✓ | ✗ | ✓(부분) | ✗ | ✓ | 중상 |
| FOIL | ✓ | **✓** | **✓** | **✓** | **✓** | **우수** |

***

## 8. 앞으로의 연구에 미치는 영향과 고려할 점

### 8.1 이론적 기여와 영향

**1. Causal Inference in TSF**
- 불변학습의 인과 원칙을 시계열에 처음 적용
- 미관측 변수 처리의 새로운 패러다임 제시
- OOD 일반화의 이론적 기초 마련

**2. 구조적 인과 모델(SCM) 확장**
- 기존 SCM: 정적 데이터 중심
- FOIL의 SCM: 시간 축을 고려한 동적 환경 모델
- 이는 다른 시계열 작업(분류, 이상탐지)으로 확장 가능

**3. 환경 개념의 재정의**
- 시계열에서 "환경" = 시간적 맥락(temporal context)
- 시간 인접성 보존으로 인과적 타당성 강화

### 8.2 방법론적 개선 방향

**1. 미관측 변수의 더 정교한 처리**
$$Y = \sum_{t=1}^{\tau} \alpha_t(Z)Y_{\text{suf},t} + \beta_t(Z) + \epsilon_t$$
- 시간 가변 미관측 변수 처리
- Wold 분해의 일반화

**2. 적응형 초매개변수 학습**
- λ₁, λ₂를 데이터 기반으로 자동 결정
- 메타 학습(meta-learning) 적용

**3. 약한 환경 정보 활용**
- 부분적 환경 레이블만으로도 학습 가능하게 확장
- 준감독(semi-supervised) 환경 추론

**4. 다변수 예측 체계화**
- 현재: 다변수 → 다중 단변수 예측
- 향후: 채널 간 불변성, 채널별 환경 다양성 모델링

### 8.3 응용 및 실무적 고려사항

**1. 도메인 특화 적용**
- **금융**: 마켓 체제 변화 (2008 금융위기, COVID-19 등)
- **의료**: 신종 질병 발생, 의료 정책 변화
- **에너지**: 계절 변화, 신재생 에너지 비중 증가
- **교통**: 도시 계획 변화, 이벤트 영향

**2. 해석성(Interpretability) 강화**
- 추론된 환경이 실제 causal mechanism을 얼마나 반영하는지 검증
- 불변 특성과 변이 특성의 명시적 식별

**3. 공정성(Fairness) 평가**
- 부분 집단(subgroup)별 성능 차이 분석
- 특정 환경에서의 차별적 성능 완화

### 8.4 미래 통합 연구 방향

**1. ShifTS와의 결합**
$$L_{\text{총}} = L_{\text{FOIL}} + L_{\text{ShifTS}}$$
- 개념 드리프트(concept drift) + 시간 이동(temporal shift) 동시 처리
- 더 광범위한 분포 시프트 대응

**2. 테스트 타임 적응(Test-Time Adaptation)**
- FOIL 기반 온라인 환경 추론
- 테스트 데이터에 대한 동적 모델 조정

**3. 사전학습(Pre-training) 통합**
- 대규모 시계열 코퍼스 사전학습 + FOIL 미세조정
- 데이터 효율성 향상

**4. 멀티태스크 학습**
- 여러 관련 시계열 작업의 불변 특성 공유
- 전이 학습 강화

### 8.5 연구자들을 위한 구체적 제안

**단기 (1-2년)**
1. 다양한 도메인에서 FOIL 성능 벤치마킹
2. 환경 추론 정확도 평가 지표 개발
3. 초매개변수 민감도 분석 및 자동 튜닝

**중기 (2-3년)**
1. ShifTS와의 결합 프레임워크
2. 약한 감독 환경 추론 방법
3. 해석 가능한 불변 특성 추출

**장기 (3년 이상)**
1. 시계열 분류/이상탐지로 IL 패러다임 확장
2. 동적 환경 모델의 이론적 분석
3. 멀티모달 시계열(센서 fusion)에의 적용

***

## 결론

FOIL은 시계열 예측의 OOD 일반화에 **불변학습의 인과적 원칙**을 처음 적용한 획기적 연구입니다. 미관측 변수의 명시적 처리와 시간 인접성을 보존하는 환경 추론이 이 논문의 독특한 강점이며, 최대 85%의 성능 개선은 그 효과를 입증합니다.

다만 미관측 변수의 부분적 처리, 환경 추론의 정확도, 장기 예측의 한계 등은 향후 개선의 여지가 있습니다. 특히 최근의 ShifTS, APT 등 관련 연구들과의 결합이 더욱 견고한 OOD 일반화 시스템을 구축할 수 있을 것으로 기대됩니다.

이 연구는 단순히 성능 개선을 넘어 **시계열 예측에서 인과추론과 OOD 일반화의 새로운 패러다임**을 제시함으로써 학계와 산업에 중대한 영향을 미칠 것으로 예상됩니다.

<span style="display:none">[^1_1][^1_10][^1_11][^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_2][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_3][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_4][^1_5][^1_6][^1_7][^1_8][^1_9]</span>

<div align="center">⁂</div>

[^1_1]: 2406.09130v1.pdf

[^1_2]: https://link.springer.com/10.1186/s12982-025-01206-0

[^1_3]: http://arxiv.org/pdf/2406.09130.pdf

[^1_4]: http://arxiv.org/pdf/2308.02282v1.pdf

[^1_5]: http://arxiv.org/pdf/2410.07018.pdf

[^1_6]: https://arxiv.org/pdf/2503.13868.pdf

[^1_7]: https://arxiv.org/html/2503.01157v1

[^1_8]: https://arxiv.org/pdf/2209.07027.pdf

[^1_9]: https://arxiv.org/pdf/2302.14829.pdf

[^1_10]: https://arxiv.org/pdf/2405.00946.pdf

[^1_11]: https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0146101

[^1_12]: https://arxiv.org/html/2503.13868v3

[^1_13]: https://arxiv.org/abs/2510.14814

[^1_14]: https://journals.plos.org/plosone/article/file?type=printable\&id=10.1371%2Fjournal.pone.0146101

[^1_15]: https://arxiv.org/html/2410.09836v2

[^1_16]: https://arxiv.org/abs/2410.09836

[^1_17]: https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0254948

[^1_18]: https://arxiv.org/html/2601.14968v1

[^1_19]: https://arxiv.org/abs/2511.17628

[^1_20]: https://arxiv.org/html/2407.17877v1

[^1_21]: https://arxiv.org/html/2511.13022v1

[^1_22]: https://arxiv.org/html/2601.20280v1

[^1_23]: https://www.biorxiv.org/content/10.1101/2023.12.18.572116v1.full.pdf

[^1_24]: https://arxiv.org/html/2510.14814v1

[^1_25]: https://arxiv.org/abs/2511.12945

[^1_26]: https://openreview.net/forum?id=1qXuNwZ1mb

[^1_27]: https://icml.cc/virtual/2025/poster/45216

[^1_28]: https://openreview.net/pdf/26fbc434477b1a9ad6098b6ddaed0a1bbc8b30b5.pdf

[^1_29]: https://proceedings.mlr.press/v235/liu24ae.html

[^1_30]: https://icml.cc/virtual/2024/poster/34011

[^1_31]: https://openreview.net/forum?id=Klx0Rq9vbC

[^1_32]: https://arxiv.org/abs/2406.09130

[^1_33]: https://liner.com/review/timeseries-forecasting-for-outofdistribution-generalization-using-invariant-learning

[^1_34]: https://www.sciencedirect.com/science/article/pii/S266731852500008X

[^1_35]: https://www.youtube.com/watch?v=sWwrJSfG6TI

[^1_36]: https://aihorizonforecast.substack.com/p/influential-time-series-forecasting-8c3

[^1_37]: https://tsood-generalization.com

[^1_38]: https://openreview.net/forum?id=Dxl0EuFjlf

[^1_39]: https://ieeexplore.ieee.org/document/11134510/
