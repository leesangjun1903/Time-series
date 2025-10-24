# Local Evaluation of Time Series Anomaly Detection Algorithms

### 1. 핵심 주장 및 주요 기여

이 논문은 **시계열 이상 탐지(Time Series Anomaly Detection)** 알고리즘을 평가할 때 기존 평가 메트릭의 심각한 문제점을 드러내고, 이를 해결하기 위한 **Affiliation Metrics(소속 메트릭)**을 제안합니다.[1]

핵심 주장은 다음과 같습니다:[1]

- 기존 정밀도(Precision)와 재현율(Recall)은 시계열의 연속적 시간 특성을 고려하지 못함
- 최근 개발된 범위 기반 메트릭(RP/RR, TaP/TaR)은 매개변수를 필요로 하고 해석 가능성이 낮으며, **적대적 알고리즘(Adversary Algorithm)**으로 쉽게 조작 가능

**주요 기여:**[1]
- 적대적 예측으로 기존 메트릭을 속일 수 있음을 실증적으로 증명
- 파라미터 없고 해석 가능한 Affiliation Metrics 제안
- 점 이상(Point Anomaly)과 범위 이상(Range Anomaly) 모두 처리 가능
- 폐쇄형(Closed-form) 이론적 특성 도출

---

### 2. 해결하는 문제 및 방법론

#### 2.1 기존 메트릭의 문제점

기존 평가 방식이 직면한 두 가지 근본적 한계:[1]

1. **시간 인접성 미인식(Unawareness of Temporal Adjacency)**: 지면 진실(Ground Truth)과 예측이 1샘플만 차이나도 거짓 양성(FP)과 거짓 음성(FN)으로 분류되어 실제로는 좋은 예측도 낮은 점수를 받음

2. **이벤트 지속시간 미인식(Unawareness of Event Duration)**: 긴 이벤트 정확히 탐지하는 것이 단일 샘플 이상 탐지보다 훨씬 높은 보상을 받아 평가 불균형 발생

#### 2.2 Affiliation Metrics의 핵심 방법론

**3단계 구조:**[1]

**(1) 평균 방향 거리(Average Directed Distance) 정의**

$$dist(X,Y) := \frac{1}{|X|} \int_{x \in X} dist(x,Y) dx$$

여기서 $$dist(x,Y) := \min_{y \in Y} |x-y|$$

이는 두 집합 간의 부드러운 변화(Smooth Variation)를 보장하며, 시간 단위의 물리적 의미를 유지합니다.[1]

**예시 (Figure 2a):** 예측 시계열에서 지면 진실까지의 평균 거리는 18초(좋은 정밀도), 역방향은 76.5초(재현율 측정)

****[1]

**(2) 소속 구역(Affiliation Zone) 분할**

시간축을 각 지면 진실 이벤트 $$gt_j$$에 대한 소속 구역 $$I_j$$로 분할합니다.[1]

각 구역에서 개별 정밀도/재현율 거리 계산:

$$D_{precision}^j := dist(pred \cap I_j, gt_j)$$

$$D_{recall}^j := dist(gt_j, pred \cap I_j)$$

이러한 **국소적(Local) 설계**는 예측 밀집도가 전체 점수에 미치는 영향을 제한하여 적대적 알고리즘 방지 기능을 제공합니다.[1]

****[1]

****[1]

** 무작위 표본 비교를 통한 정규화**

거리를 확률 로 변환하기 위해 무작위 예측 기준과 비교합니다.[1]

**개별 정밀도 확률:**

$$P^{precision}_j := \frac{1}{|pred \cap I_j|} \int_{x \in pred \cap I_j} \overline{F}^{precision}_j(dist(x, gt_j)) dx$$

여기서 $$\overline{F}^{precision}_j(d) := 1 - F^{precision}_j(d^-)$$는 생존 함수(Survival Function)

**개별 재현율 확률:**

$$P^{recall}_j := \frac{1}{|gt_j|} \int_{y \in gt_j} \overline{F}^{y,recall}_j(dist(y, pred \cap I_j)) dy$$

**평균화:**

$$P^{precision} := \frac{1}{|S|} \sum_{j \in S} P^{precision}_j$$

$$P^{recall} := \frac{1}{n} \sum_{j=1}^{n} P^{recall}_j$$

여기서 $$S = \{j \in [1,n]: pred \cap I_j \neq \emptyset\}$$[1]

***

### 3. 모델 구조 및 성능 분석

#### 3.1 적대적 알고리즘에 대한 강건성

**적대적 예측 구성** (4.2절):[1]

1. "Trivial 이벤트"(단순 임계값으로 탐지 가능한 이벤트) 식별
2. Trivial 이벤트 내에서: 최대 수의 예측 이벤트 생성 (양성/음성 샘플 교대)
3. Trivial 이벤트 외부: 모든 샘플을 양성으로 라벨

**결과 (표 5):**[1]

| 메트릭 | NYC-Taxi Adversary 성능 |
|--------|----------------------|
| RP/RR | 0.88 정밀도 / 0.85 재현율 |
| TaP/TaR | 0.93 정밀도 / 1.00 재현율 |
| **Affiliation** | **0.54 정밀도 / 1.00 재현율** |

RP/RR과 TaP/TaR은 적대적 알고리즘을 높이 평가하지만, **Affiliation Metrics는 정밀도를 ~0.5로 유지**(무작위 예측 수준).[1]

**이론적 증명** (4.4절):

전체 구간을 이상으로 예측할 때:[1]

$$P^{precision} = \frac{1}{2} + \frac{p^2}{2}$$

$$P^{recall} = 1$$

여기서 $$p = |gt_j|/|I_j|$$ (매우 작은 $$p$$에서 정밀도 ≈ 0.5)

#### 3.2 정밀도 0.5의 의미

Affiliation Metrics는 정밀도/재현율 < 0.5를 **무작위 예측보다 나쁜 상태**, ~0.5를 **무작위 예측 수준**으로 해석합니다.[1]

**단일 무작위 예측 시의 기댓값** (4.4절):[1]

$$E[P^{precision}] = \frac{1}{2} + \frac{p^2}{2} \approx 0.5 \quad (p \ll 1)$$

$$E[P^{recall}] = 0.5$$

***

### 4. 일반화 성능 향상 가능성

#### 4.1 현재 메트릭 설계의 강점

**파라미터 프리(Parameter-free) 설계:**[1]

- RP/RR: 4개 파라미터 필요
- TaP/TaR: 3개 파라미터 필요
- **Affiliation: 0개** - 모든 데이터셋에 균일하게 적용 가능

이는 **과적합 위험 감소** 및 **일반화 성능 향상**을 의미합니다.[1]

**이벤트별 해석 가능성** (4.3절):

표 6의 SWaT 데이터셋 분석:
- iForest vs seq2seq 알고리즘 비교
- **21/35 이벤트**에서 seq2seq 우수
- **개별 이벤트 수준**의 상세 분석 가능

일반화 성능 개선을 위한 시사점:[1]

1. **국소적 해석(Local Interpretability)**: 각 이벤트별 강약점 파악 가능 → 앙상블 설계 최적화
2. **강건성 이론**: 적대적 전략에 대한 수학적 증명으로 평가 신뢰성 보장
3. **표준화된 비교**: 파라미터 튜닝 없이 알고리즘 공정 비교 가능

#### 4.2 일반화 성능의 수학적 기초

**정위치 예측(Positioned Predictions) 분석** (4.4절, Figure 4):

지면 진실이 소속 구역 중앙에 위치할 때, 예측 위치별 메트릭:[1]

| 위치 | 정밀도 공식 | 재현율 |
|------|---------|--------|
| (a) 구역 경계 | 0 | p/4 |
| (b) 중간 거리 | 1/2 - p/2 | 1/2 - p/2 + ... |
| (c) 이벤트 시작 | 1 | $$1 - p + \frac{16}{9p}\left(p-\frac{1}{3}\right)_+^2$$ |
| (d) 이벤트 중심 (최적) | 1 | $$1 - \frac{p}{2} + \frac{1}{2p}\left(p-\frac{1}{2}\right)_+^2$$ |

**$$p \ll 1$$ 영역에서:**[1]
- 거리 예측은 0에 수렴
- 정확한 예측은 1에 수렴
- 점진적 성능 향상

이는 메트릭이 **표현력(Expressiveness)** 있음을 보장합니다.[1]

---

### 5. 한계 및 고려사항

#### 5.1 메트릭의 한계

**지면 진실 품질 의존성** (5절):[1]

- 각 라벨이 단일 이상을 나타내야 함
- 이벤트 시작점과 종료점 판정의 주관성 존재 (예: 수처리 시스템, 제어 시스템)
- 단편화된 이벤트 병합 필요 (Wu & Keogh, 2021)

**통계적 특성의 미개발** (5절):[1]

> "The affiliation metrics focus on a single aspect of the evaluation: the assessment of the proximity between the predicted and the ground truth labels... further research is needed to gather theoretical bounds on the variance."

F-score 등 단일값 요약 메트릭의 분산 경계 미정의

**배포 단계의 고려사항** (5절):[1]

- 정밀도/재현율 외 추가 지표 필요:
  - 이벤트 개수
  - 예측 방향성 (조기 탐지 vs 지연 탐지)
  - 대화형 시각화 도구 필요

#### 5.2 적용 범위의 제한

**점 이상과 범위 이상:**

- 논문은 두 가지 처리 가능하나, 부드러운 거리 정의 방식에서 이상 종류에 따른 미세 조정 필요

**다변량 데이터:**

- 논문 사례는 단일 시계열만 포함
- 다변량 확장 시 거리 메트릭 재검토 필요

---

### 6. 미래 연구에 미치는 영향

#### 6.1 학문적 기여

**평가 메트릭 분야:**[1]

- "매개변수 자유(Parameter-free) + 국소적 해석 가능" 설계 패러다임 제시
- 이상 탐지 벤치마크 개선 표준으로 활용 가능

**강건성 이론:**

적대적 알고리즘 방어 메커니즘의 수학적 형식화는 다른 시계열 평가 문제(변화점 탐지, 분할)로 확장 가능

#### 6.2 실무 적용 시 고려점

1. **메트릭 선택 가이드:**
   - 알려진 적대적 위협 → Affiliation Metrics
   - 실시간 배포 요구 → 추가 성능 지표 병행

2. **향후 개선 방향:**
   - 대화형 시각화 라이브러리 개발
   - 다변량/고차원 확장
   - 운영 환경의 비용-편익 분석 프레임워크

3. **벤치마킹 재정의:**
   - 기존 NAB, HAI 데이터셋 재평가
   - Affiliation Metrics로 신규 알고리즘 평가 표준화

***

### 7. 종합 평가

**논문의 강점:**

- 명확한 문제 정의 (적대적 알고리즘 취약성 증명)
- 이론적 엄밀성 (폐쇄형 생존 함수, 기댓값 증명)
- 실용적 유용성 (파라미터 없음, 직관적 해석)

**개선 필요 영역:**

- 다변량 시계열 확장
- 분산 경계 도출
- 운영 환경 검증 (스트리밍 데이터 등)

이 논문은 시계열 이상 탐지 알고리즘 평가의 **신뢰성 위기**를 진단하고 근거 있는 해결책을 제시하여, 향후 벤치마킹 표준으로 확립될 가능성이 높습니다.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6fc4cb9c-9208-4f92-b6c3-bd3646b61cca/2206.13167v1.pdf)
