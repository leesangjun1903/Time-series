

# 다양한 물류 흐름을 고려한 효율적인 Fab 공정 배치 결정 및 평가 방법

**저자:** 서용진, 아주대학교 대학원 산업공학과 공학 박사 학위 논문

**주요 참고자료:** Suh, Y. J., & Choi, J. Y. (2021). "Efficient Fab facility layout with spine structure using genetic algorithm under various material-handling considerations." *International Journal of Production Research*, 1-14. [36]

---

## 1. 핵심 주장과 주요 기여 (간결 요약)

### 핵심 주장
기존의 반도체 Fab 공정 배치는 **총 물류 이동거리 최소화**만을 목적으로 하여, OHT(Overhead Hoist Transport)와 트랙 구성 특성에 의해 발생하는 **물류 정체(congestion) 위험**을 고려하지 못하였다. 본 연구는 스파인(Spine) 구조 Fab에서 OHT에 의한 물류 흐름을 **순방향(Forward)**, **역방향(Reverse)**, **횡단방향(Cross)**으로 세분화하고, 중앙 통로의 정체 위험을 최소화하는 공정 배치 방법을 제안한다.

### 주요 기여
1. **유전자 알고리즘(GA) 기반 공정 배치 결정 방법** — 물류 흐름의 방향성을 고려한 적합도 함수를 설계하여 중앙 통로 혼잡 위험도(ConDeg)를 20~31% 개선하면서도, 최적 해 대비 평균 약 6%의 오차를 가지는 근사 해를 매우 효율적으로(계산 시간 최대 11,459배 단축) 탐색
2. **AHP(Analytic Hierarchy Process) 기반 배치 대안 평가 방법** — 생산·물류 통합 관점에서 객관적이고 일관성 있는 평가 기준을 도출하고, 복수 배치 대안의 체계적 비교·선정 방법론 제시

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

반도체 Fab의 설비 배치 문제(Facility Layout Problem, FLP)는 **NP-hard** 문제로, 공정 수 $N$이 증가하면 배치 대안 수가 $N!$로 증가한다. 기존 연구의 한계는 다음과 같다:

- **전체 물류 이동거리 최소화**만 고려하여, AMHS(OHT)의 물리적 특성(일방 주행, 순차 주행, 트랙 회차)에 의한 **중앙 통로 물류 정체 위험**을 반영하지 못함
- 시뮬레이션 기반 검증은 **초기 데이터 확보 및 모델링에 과다한 시간**이 소요되어, 기획 단계에서 빈번한 배치 수정에 부적합
- 공정(설비) 배치와 AMHS 배치가 **서로 다른 조직에서 순차적**으로 진행되어, 물류 흐름의 집중·정체 위험이 충분히 고려되지 못함

### 2.2 제안하는 방법 (수식 포함)

#### 방법 1: GA 기반 공정 배치 결정

**(1) 물류 흐름의 세분화**

스파인 구조 Fab에서 OHT에 의한 물류 흐름을 3가지로 구분:
- **순방향 흐름(Forward flow):** OHT 주행 방향과 차기 공정 목적지 방향이 동일
- **역방향 흐름(Reverse flow):** OHT 주행 방향과 반대 → 중앙 통로 회차(Short Cut) 트랙 경유 → **중앙 통로 점유**
- **횡단 방향 흐름(Cross flow):** 차기 공정이 중앙 통로 건너편 → **중앙 통로 횡단** → **중앙 통로 점유**

**(2) 적합도 함수 (Fitness Function)**

$$Fval = w_1 \cdot FitValue1 + w_2 \cdot FitValue2 + w_3 \cdot FitValue3 + FitValue4 \tag{1}$$

여기서:
- $FitValue1$: 순방향 흐름에 의한 물류 이동거리
- $FitValue2$: 역방향 흐름에 의한 물류 이동거리
- $FitValue3$: 횡단 방향 흐름에 의한 물류 이동거리
- $FitValue4$: 제약 조건 위반에 대한 페널티 값
- $w_1, w_2, w_3$: 각 흐름 성분에 대한 가중치

전체 물류 이동거리: $FitSum = FitValue1 + FitValue2 + FitValue3$

**(3) 공정 간 거리 (직각 거리)**

공정 $i$와 공정 $j$의 중심 좌표를 각각 $(x_i, y_i)$, $(x_j, y_j)$라 할 때:

$$d_{ij} = |x_i - x_j| + |y_i - y_j| \tag{2}$$

**(4) 제약 조건 (페널티 함수)**

상부 세그먼트 $p$에 속한 공정 $u$의 폭을 $l_{pu}$, 하부 세그먼트 $q$에 속한 공정 $v$의 폭을 $l_{qv}$라 할 때:

$$g_1 = \sum l_{pu} - W \leq 0 \tag{3}$$
$$g_2 = \sum l_{qv} - W \leq 0$$

$$FitValue4 = \sum_{k=1}^{2} R_k \cdot \max(g_k, 0) \tag{4}$$

여기서 $R_k$는 매우 큰 값의 페널티 계수이며, $W$는 Fab의 폭이다.

**(5) 혼잡 위험도 지표 (Congestion Degree)**

$$ConDeg = \frac{FitValue2 + FitValue3}{FitSum} \tag{5}$$

이 지표는 전체 물류 이동거리 중 **중앙 통로를 경유하는 역방향 및 횡단 방향 흐름의 비율**로, 중앙 통로의 OHT 정체·혼잡 위험도를 예측한다.

**(6) 성능 평가 지표**

$$\% \, error = \frac{ConDeg_{GA \, using \, (1,5,5)} - ConDeg_{optimal \, solution}}{ConDeg_{optimal \, solution}} \times 100 \tag{7}$$

$$\% \, improvement = \frac{ConDeg_{GA \, using \, (1,1,1)} - ConDeg_{GA \, using \, (1,5,5)}}{ConDeg_{GA \, using \, (1,1,1)}} \times 100 \tag{8}$$

#### 방법 2: AHP 기반 배치 대안 평가

대안의 중요도를 다음과 같이 결정:

$$\text{중요도} = f\left(\text{중요도}_{평가기준1}, \text{중요도}_{평가기준2}, \cdots, \text{중요도}_{평가기준n}\right) \tag{9}$$

쌍대 비교 행렬의 구성 요소 특성:

$$a_{ii} = 1, \quad a_{ij} = \frac{1}{a_{ji}}, \quad i = 1 \sim 4 \tag{10}$$

일관성 지수(CI) 및 일관성 비율(CR):

$$CI = \frac{\lambda - n}{n - 1} = \frac{4.220 - 4}{4 - 1} = 0.073 \tag{12}$$

$$CR = \frac{CI}{RI} = \frac{0.073}{0.90} = 0.082 \tag{13}$$

$CR \leq 0.1$이므로 일관성이 만족되는 수준으로 검증되었다.

### 2.3 모델 구조

#### GA 모델 구조

| 구성 요소 | 설명 |
|---|---|
| **염색체(Chromosome)** | 공정 수 $N$의 1차원 배열, 상부 베이 좌측→시계 방향 순서의 공정 배열 정보 |
| **초기 해 집단** | 크기 $Q$의 임의 순열 조합 생성 |
| **적합도 함수** | 식 (1)의 가중 물류 이동거리 + 페널티 |
| **선택(Selection)** | 엘리티즘(Elitism) 기반 |
| **교차 연산** | 일점/이점 교차 (염색체 중복 회피 설계 적용) |
| **돌연변이 연산** | 임의 두 위치의 요소 교환 |
| **종료 조건** | 해 개선 미비 또는 효율성 임계 도달 시 종료 |

#### AHP 모델 구조

| 계층 | 내용 |
|---|---|
| **목표 (Level 1)** | Fab 배치 대안의 비교 우위 평가 |
| **평가 기준 (Level 2)** | 물류 이동거리(0.495), 인접성(0.181), 확장성(0.086), 재공(0.237) |
| **대안 (Level 3)** | 기존 연구 3개 배치안 + GA 배치안 (대안 4) |

### 2.4 성능 향상

#### GA 기반 배치 결정의 성능

| 구성 | 공정 수 | ConDeg 개선율 (%improvement) | 최적해 대비 오차 (%error) | 계산 시간 비율 (GA/Exact) |
|---|---|---|---|---|
| Conf.1 | 8 | **평균 20.0%** | 평균 6.1% | 약 1/2.5배 |
| Conf.2 | 10 | **평균 30.6%** | 평균 5.8% | 약 1/93배 |
| Conf.3 | 12 | **평균 31.4%** | 평균 6.0% | **약 1/11,459배** |

- 완전 요인 실험(6인자, 2수준, 64개 조합)을 통해 최적 매개변수 $(w_1, w_2, w_3, Q, r_c, r_m) = (1, 5, 5, 200, 0.9, 0.05)$ 도출
- 일점·이점 교차 연산 간 유의차 없음 ($p\text{-value} = 0.634$, paired t-test)

#### AHP 기반 평가 결과

| 대안 | 중요도 | 순위 |
|---|---|---|
| 대안 1 (Chung & Jang, 2007) | 0.313 | 2 |
| 대안 2 (Kim & Jang, 2013) | 0.083 | 4 |
| 대안 3 (Yang & Kuo, 2003) | 0.144 | 3 |
| **대안 4 (GA 배치, 본 연구)** | **0.460** | **1** |

### 2.5 한계

1. **공정 간 근접/회피 배치 제약** (예: 진동 제거용 독립 기반 배치) 미고려
2. **임의 크기의 공정 수 $N$**에 대한 일반화된 배치 방법론으로의 확장 미비
3. OHT 트랙의 실제 **비대칭적 왕복 거리**(회차 포함)를 동일 거리로 가정
4. Fab의 **초대형화(mega-size)**, 건물 간·층간 연결에 의한 혼잡 분산배치 미고려
5. AHP의 쌍대 비교에서 **전문가 5인의 주관적 판단**에 의존 (설문 규모 제한)
6. **동적 물류 현상**(실시간 정체, 교착 상태 등)의 확률적 모델링은 부재

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재 모델의 일반화 수준

본 연구에서는 8, 10, 12개 공정에 대해 **각각 30개의 서로 다른 물류흐름 밀도행렬(Flow Density Matrix)**을 생성하여 **총 90개 인스턴스**에서 실험을 수행하였다. 이를 통해:

- 다양한 물류 흐름 패턴에서도 GA가 **일관되게 ConDeg를 20~31% 개선**
- 최적해 대비 **평균 6% 이내 오차**를 안정적으로 유지
- 공정 수 증가(8→10→12)에 따라 개선율이 오히려 **증가하는 경향** (20%→30.6%→31.4%), 이는 문제 규모가 커질수록 제안 방법의 효과가 더 두드러짐을 시사

### 3.2 일반화 성능 향상을 위한 방향

1. **공정 수의 확장**: 현재 최대 12개 공정에서 실험하였으나, 실제 대형 Fab은 더 많은 세부 공정을 가질 수 있음. $N$이 더 커질 때 GA의 수렴성과 정확도에 대한 추가 검증이 필요

2. **다양한 Fab 구조로의 확장**: 스파인 구조에 한정되어 있으나, 최근 **멀티 스파인(multi-spine)**, **루프(loop)**, **매트릭스(matrix)** 구조 등 다양한 Fab 형태에 대한 적용 가능성 연구 필요

3. **가중치의 적응적 결정**: 현재 완전 요인 실험으로 $(w_1, w_2, w_3) = (1, 5, 5)$를 고정하였으나, Fab의 특성(중앙 통로 폭, 베이 수, OHT 대수 등)에 따라 **가중치를 자동 조정하는 메타 학습(meta-learning) 기법** 적용 가능

4. **동적 물류 특성 반영**: 정적 물류흐름 밀도행렬 기반에서 벗어나, **시간 변동적(time-varying) 물류 패턴**을 반영할 수 있는 확장 필요

5. **추가 제약 조건 통합**: 진동 회피, 클린룸 등급, 유틸리티 배관 등 실제 산업 현장의 **다양한 물리적 제약**을 적합도 함수에 통합

6. **하이브리드 최적화**: GA와 **강화학습(Reinforcement Learning)**, **시뮬레이션 최적화** 등을 결합하여 동적 환경에서의 일반화 성능 향상

---

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 4.1 연구에 미치는 영향

1. **물류 흐름 방향성 개념의 도입**: 기존 FLP 연구가 총 이동거리 최소화에 집중한 것과 달리, OHT의 물류 흐름을 순방향·역방향·횡단방향으로 세분화한 것은 반도체 Fab 배치 연구에 **새로운 패러다임**을 제시

2. **기획 단계에서의 실용적 도구 제공**: 시뮬레이션 대비 극히 짧은 시간(수 초)에 근사 최적해를 탐색할 수 있어, 빈번한 설계 변경이 요구되는 Fab 기획 실무에 **즉시 적용 가능**한 방법론

3. **생산·물류 통합 관점의 평가 프레임워크**: AHP 기반 평가 모델은 반도체 산업뿐 아니라, 유사한 AMHS를 사용하는 **디스플레이, 2차전지** 등 타 산업의 배치 문제에도 확장 적용 가능

4. **Fab 생애주기 전체 커버리지**: 신규 기획 단계에서는 GA 기반 결정, 운영 단계에서는 AHP 기반 평가로, Fab의 **전 생애주기에 걸친 종합적 배치 방법론** 제시

### 4.2 앞으로 연구 시 고려할 점

1. **대규모 문제로의 확장성(Scalability)**: 공정 수가 20개 이상으로 증가할 경우 GA의 수렴 속도와 해의 품질 변화에 대한 체계적 분석 필요

2. **실제 OHT 트랙 토폴로지 반영**: 본 연구에서는 왕복 거리를 동일하게 가정하였으나, 실제 회차(Short Cut) 트랙 포함 시 비대칭 거리를 반영하는 정교한 모델 필요

3. **다목적 최적화(Multi-objective Optimization)**: ConDeg 최소화와 총 이동거리 최소화를 **파레토 최적(Pareto Optimal)** 관점에서 동시 추구하는 접근 필요

4. **확률적·동적 모델링**: 생산량 변동, 설비 고장, 제품 믹스 변경 등 **불확실성(uncertainty)**하의 강건한(robust) 배치 결정 방법 연구 필요

5. **디지털 트윈(Digital Twin) 연계**: GA 기반 배치 결정과 시뮬레이션을 **디지털 트윈 환경에서 통합** 운영하여, 실시간 배치 최적화 구현

6. **AI/ML 기법과의 융합**: 딥러닝 기반 물류 흐름 예측, 강화학습 기반 OHT 디스패칭과의 연계를 통한 **종합적 Fab 최적화 프레임워크** 구축

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

본 논문의 참고문헌 및 Appendix A에 수록된 2019~2020년 최신 연구와의 비교:

| 비교 항목 | 본 연구 (Suh & Choi, 2021) [36] | Hwang & Jang (2020) [26] | Lee et al. (2020) [27] | Wu et al. (2019) [28] | Chae & Regan (2020) [29] | Chen et al. (2020) [15] |
|---|---|---|---|---|---|---|
| **Fab 구조** | Spine (스파인) | Spine (Mega fab) | Spine (Mega fab) | Spine | Double Row (AGV) | Loop |
| **주요 방법** | GA + AHP | Q(λ)-learning (시뮬레이션 검증) | Machine Learning (교통 제어) | Fuzzy Logic + Hungarian Method | Mixed MIP | Simulation-Optimization + GA |
| **목적 함수** | ConDeg (물류 정체 위험도) 최소화 | Average delivery time | Delivery/Transfer/Queued time | Delivery time, Vehicle utilization | Solution time (계산 효율) | Throughput (컨베이어) |
| **물류 흐름 세분화** | ✅ 순방향/역방향/횡단방향 | ❌ | ❌ | ❌ | ❌ | ❌ |
| **배치 결정 vs 운영 최적화** | **배치 결정 (Planning)** | 운영 최적화 (Operation) | 운영 최적화 | 운영 최적화 | 배치 결정 | 배치 결정 |
| **산업 적용성** | 기획 단계 즉시 적용 가능 (수 초 내 해 탐색) | 시뮬레이션 기반 검증 필요 | 대규모 학습 데이터 필요 | 시뮬레이션 기반 | 소규모 문제에 적합 | 컨베이어 시스템 한정 |

### 차별점 분석

1. **유일하게 물류 흐름의 방향성을 세분화**: [26], [27], [28] 등 최신 연구들은 OHT의 경로 가이드·디스패칭 등 **운영(Operation) 단계**의 최적화에 집중하고 있으나, 본 연구는 **기획(Planning) 단계**에서 물류 흐름의 방향적 특성을 배치 결정에 반영

2. **계산 효율성의 실용적 우위**: MIP 기반 연구 [29]는 소규모 문제에서만 적용 가능하고, 시뮬레이션 기반 연구 [15], [26]은 모델링 시간이 과다한 반면, 본 연구의 GA는 12개 공정 기준 **약 3.4초**에 근사 최적해 도출

3. **배치와 평가의 통합 프레임워크**: 대부분의 기존 연구가 배치 결정 또는 운영 최적화 중 하나에만 초점을 맞추는 반면, 본 연구는 **GA(결정) + AHP(평가)**의 통합 방법론을 제시하여 Fab 생애주기 전체를 커버

4. **2020년 이후 추가적인 관련 동향**: 최근에는 강화학습(RL)과 딥러닝을 결합한 **지능형 AMHS 제어** 연구가 활발하며 [26, 27], 이러한 운영 단계 기법과 본 연구의 기획 단계 방법론을 **연계·통합**하는 것이 향후 중요한 연구 방향이 될 것으로 판단됨

---

## 참고자료 (본 분석에서 인용한 출처)

1. 서용진 (2021). "다양한 물류 흐름을 고려한 효율적인 Fab 공정 배치 결정 및 평가 방법." 아주대학교 대학원 산업공학과 공학 박사 학위 논문.
2. Suh, Y. J., & Choi, J. Y. (2021). "Efficient Fab facility layout with spine structure using genetic algorithm under various material-handling considerations." *International Journal of Production Research*, 1-14. [36]
3. Hwang, I., & Jang, Y. J. (2020). "Q(λ) learning-based dynamic route guidance algorithm for overhead hoist transport systems in semiconductor fabs." *International Journal of Production Research*, 58(4), 1199-1221. [26]
4. Lee, S., Kim, Y., Kahng, H., et al. (2020). "Intelligent traffic control for autonomous vehicle systems based on machine learning." *Expert Systems with Applications*, 144, 113074. [27]
5. Wu, L., Zhang, Z., & Zhang, J. (2019). "A Hybrid Vehicle Dispatching Approach for Unified Automated Material Handling System in 300mm Semiconductor Wafer Fabrication System." *IEEE Access*, 7, 174028-174041. [28]
6. Chae, J., & Regan, A. C. (2020). "A mixed integer programming model for a double row layout problem." *Computers & Industrial Engineering*, 140, 106244. [29]
7. Chen, T. L., et al. (2020). "Solving the layout design problem by simulation-optimization approach." *Simulation Modelling Practice and Theory*, 102192. [15]
8. Yang, T., Su, C.T., & Hsu, Y.R. (2000). "Systematic layout planning: A Study on semiconductor wafer fabrication facilities." *International Journal of Operation & Production Management*, 20(11), 1359-1371. [23]
9. Kim, J. H., Yu, G. J., & Jang, Y. J. (2016). "Semiconductor fab layout design analysis with 300-mm fab data." *Computers & Industrial Engineering*, 99, 330-346. [34]
10. Singh, S. P. & Sharma, R. R. K. (2005). "A review of different approaches to the facility layout problems." *International Journal of Advanced Manufacturing Technology*, 30(5-6), 425-433. [1]
