# Can Competition Enhance the Proficiency of Agents Powered by Large Language Models in the Realm of News-driven Time Series Forecasting?

## 1. 핵심 주장과 주요 기여

이 논문은 뉴스 기반 시계열 예측(news-driven time series forecasting)에서 멀티 에이전트 경쟁 메커니즘이 LLM 기반 에이전트의 혁신적 사고 능력과 예측 성능을 크게 향상시킬 수 있음을 입증합니다. 기존 멀티 에이전트 토론 프레임워크가 겪는 **Degeneration-of-Thought (DoT)** 문제와 **Wrong Logic Propagation Error**를 해결하기 위해, 사회과학의 경쟁 이론에서 영감을 얻은 새로운 접근법을 제시합니다.[^1_1]

주요 기여는 다음과 같습니다:[^1_1]

- **정보 비대칭(Information Asymmetry)**, **경쟁 인식(Competitive Awareness)**, **적자생존(Survival of the Fittest)**을 통합한 경쟁 메커니즘 제안
- 오도하는 정보를 식별하기 위한 **Multi-Stage Reflection (MSR)** 전략 설계
- 4개 데이터셋에서 MAE 31.03%, MSE 36.31%, RMSE 2.48%, MAPE 18.14% 평균 성능 향상


## 2. 해결하고자 하는 문제와 제안 방법

### 2.1 핵심 문제

뉴스 기반 시계열 예측의 핵심 과제는 서로 다른 뉴스 이벤트가 시계열 변동에 미치는 영향을 측정하는 것입니다. 기존 멀티 에이전트 토론 프레임워크는 다음 두 가지 한계를 보입니다:[^1_1]

1. **DoT(Degeneration-of-Thought) 문제**: 여러 라운드의 토론 후 모델의 높은 확신도로 인해 새로운 사고가 부족해지는 현상
2. **Wrong Logic Propagation Error**: 토론 중 잘못된 논리가 전파되어 에이전트가 오도될 수 있음

### 2.2 제안 방법론

#### 2.2.1 문제 정식화

시계열 $X$를 슬라이딩 윈도우를 통해 $S$개의 부분 시계열 $X_1, X_2, ..., X_S$로 분할합니다. 주어진 시계열 $X_s = \{x_1, x_2, ..., x_t\}$에 대해, 모델은 뉴스 데이터베이스 $D$에서 관련 뉴스 $N_s = \{n_1, n_2, ..., n_d\}$를 선택하고, 이를 바탕으로 시점 $t+1$의 값 $\hat{y}_{s,t+1}$을 예측합니다.[^1_1]

#### 2.2.2 Multi-Indicator Evaluation (MIE)

경쟁 인식을 유도하기 위해 각 에이전트의 성능을 다음과 같이 평가합니다:[^1_1]

$EM_i^{(e)} = \{rank_i^{(e)}, top_i^{(e)}, ave_i^{(e)}\}$ (1)

여기서:

- $rank_i^{(e)}$: 라운드 $e$에서 에이전트 $i$의 순위
- $ave_i^{(e)}$: 전체 평균 대비 성능 증감률
- $top_i^{(e)}$: 최고 성능 대비 성능 차이

누적 점수(Cumulative Score)는 다음과 같이 계산됩니다:[^1_1]

$M_i^{(e+1)} = M_i^{(e)} + M_i^{(e)} \times (1 - MMN(MAPE_i^{(e)}))$ (2)

#### 2.2.3 Information Asymmetry (IA)

에이전트는 자신의 논리를 다음과 같이 공개합니다:[^1_1]

$PL^{(e)} = \{pl_1^{(e)}, pl_2^{(e)}, ..., pl_I^{(e)}\}$ (3)

$pl_i^{(e)} = LLM_L(P_{IA}, X, logic_i^{(e)}, EM^{(e)}, target)$ (7)

이 메커니즘은 에이전트가 전체 논리를 공개하거나, 부분적으로 공개하거나, 오도하는 정보를 제공할 수 있도록 허용하여 정보 비대칭을 구현합니다.[^1_1]

#### 2.2.4 Multi-Stage Reflection (MSR)

**1단계**: 기존 방법으로 논리 업데이트[^1_1]

$L_i^{(e+1)'} = LLM_L(P_{ref}, X, PL_{-i}^e, L_i^{(e)}, EM^{(e)})$ (8)

**2단계**: 업데이트된 부분 추출 및 평가[^1_1]

$\delta_i^{(e+1)} = \{\delta_1, \delta_2, ..., \delta_U\} = diff(L_i^{(e+1)'}, L_i^{(e)})$ (4)

각 업데이트 부분 $\delta_u$에 대해 다음과 같이 판단합니다:[^1_1]

$$
\begin{cases} 
ID(\delta_u) = good, & \text{if } IR(L_i^{(e+1)'} - \delta_u) \leq IR(L_i^{(e+1)'}) \\
ID(\delta_u) = bad, & \text{if } IR(L_i^{(e+1)'} - \delta_u) > IR(L_i^{(e+1)'})
\end{cases}
$$ 

(10)

**3단계**: "bad"로 표시된 부분 재평가 후 최종 논리 결정:[^1_1]

$L_i^{(e+1)} = L_i^{(e+1)'} - \delta_{i,bad}^{(e+1)}$ (5)(11)

#### 2.2.5 최종 예측 집계

$E$번째 라운드 후 $I'$개의 에이전트가 남으면, 최종 예측값은 다음과 같이 계산됩니다:[^1_1]

$\hat{y}_{s,t+1} = \sum\_{i=1}^{I'} \frac{M_i^{(E)}}{\sum_j M_j^{(E)}} \times \hat{y}_i^{s,t+1}$ (6)

### 2.3 모델 구조

프레임워크는 4단계로 구성됩니다:[^1_1]

1. **뉴스 필터링 단계**: 각 에이전트가 논리 $L_i^{(e)}$에 따라 뉴스 세트 $N_i^{(e)}$ 선택 (GPT-4o 사용)
2. **시계열 예측 단계**: 파인튜닝된 Llama 2-7B ($LLM_{S,i}^{(e)}$)로 예측 수행
3. **에이전트 성능 평가 단계**: MIE로 평가하고 SF(Survival of Fittest)로 하위 $(1-\alpha)\%$ 제거
4. **토론 및 반성 단계**: IA와 OOSR(Opponent-Oriented Self-Reflection)을 통해 논리 업데이트

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 도메인 간 일반화

논문은 전기 부하, 환율, 교통량, 비트코인 가격이라는 4개의 서로 다른 도메인에서 일관되게 우수한 성능을 보였습니다. 각 도메인에서의 개선율:[^1_1]

- 전기(Electricity): MAPE 6.71%
- 환율(Exchange): MAE 4.41×10³, MAPE 0.63%
- 교통(Traffic): MAE 1.56×10²
- 비트코인(Bitcoin): MAPE 2.83%


### 3.2 혁신적 사고 능력 향상

IA 컴포넌트가 있는 모델은 논리 유사도를 낮게 유지하여 더 다양한 에이전트 논리를 생성합니다. Logic Update Degree (LUD) 분석 결과, IA가 있는 모델은 에이전트가 여러 경쟁 라운드를 거치면서 더 의미 있는 논리 업데이트를 생성함을 보여줍니다.[^1_1]

### 3.3 적응적 경쟁 강도

논문은 사회과학의 발견과 일치하는 **U자형 관계**를 발견했습니다: 중간 수준의 경쟁도(40-70%)를 가진 에이전트가 평균적으로 가장 좋은 성능을 보이며, 경쟁과 협력 사이의 균형을 형성합니다. 이는 모델이 다양한 환경에 적응할 수 있는 잠재력을 시사합니다.[^1_1]

### 3.4 LLM 간 전이 가능성

GLM-4-130B, DeepSeek-V2.5, GPT-4o 등 서로 다른 LLM에서 유사한 성능을 보여 제안된 방법이 다양한 LLM에 효과적으로 적용될 수 있음을 입증했습니다.[^1_1]

## 4. 성능 향상 및 한계

### 4.1 성능 향상

**Ablation Study 결과**:[^1_1]

- IA 제거: 성능 대폭 저하 (Electricity 데이터셋에서 RMSE 450.71 vs 364.52)
- MIE 제거: 경쟁 인식 약화로 성능 저하
- SF 제거: 20.49% 성능 저하
- MSR 제거: 오도하는 정보 식별 능력 저하로 성능 감소

**기존 방법 대비**:[^1_1]

- Agent Discussion (AD) 대비: MAE 6.74%, MSE 43.55%, RMSE 32.41%, MAPE 31.87% 개선
- Agent Collaboration (AC) 대비: MAE 24.93%, MSE 57.37%, RMSE 40.71%, MAPE 52.69% 개선


### 4.2 한계

논문은 다음과 같은 한계를 명시하고 있습니다:[^1_1]

1. **메커니즘 해석 부족**: 정보 비대칭과 경쟁 인식이 혁신적 사고를 촉진하는 근본적인 메커니즘에 대한 추가 연구 필요
2. **수학적 지식 통합 부족**: 다변량 시계열 관련 수학적 지식(deep auto-regressive modeling, time stationarity analysis, co-integration testing, DTW algorithms) 미통합
3. **계산 자원 요구**: 멀티 에이전트 경쟁 모델은 높은 계산 자원과 긴 계산 시간 필요
4. **장문 추론 한계**: 복잡한 장문 추론 작업에서 여러 제약 존재

## 5. 향후 연구에 미치는 영향과 고려사항

### 5.1 향후 연구에 미치는 영향

**1. 멀티 에이전트 시스템 설계 패러다임**[^1_1]

- 경쟁 메커니즘이 협력적 토론만큼 중요함을 입증
- 정보 비대칭을 의도적으로 설계하여 시스템 성능 향상 가능
- 사회과학 이론(경쟁과 혁신)을 AI 시스템에 적용하는 새로운 관점 제시

**2. LLM 기반 시계열 예측**[^1_2][^1_3]

- 뉴스와 같은 비구조화된 데이터를 시계열 예측에 통합하는 효과적인 방법 제시
- 기존 연구(Wang et al., 2024b "From News to Forecast")를 경쟁 메커니즘으로 확장하여 성능 크게 개선

**3. 오류 전파 방지 메커니즘**

- MSR의 정량적 지표 활용은 LLM의 판단 가능성을 높이는 중요한 접근법
- 파인튜닝된 소규모 LLM을 보조 도구로 활용하는 전략 제시


### 5.2 향후 연구 시 고려사항

**1. 이론적 기반 강화**

- Chain-of-Thought 파인튜닝과 distillation을 통합하여 경쟁 메커니즘의 제어 가능성 향상 필요
- 정보 비대칭과 혁신 사이의 인과 관계에 대한 체계적 분석 필요

**2. 수학적/통계적 지식 통합**

- Auto-regressive 모델링, stationarity analysis, co-integration testing 등 시계열 이론 통합
- DTW(Dynamic Time Warping) 알고리즘 등 시계열 유사도 측정 방법 적용

**3. 계산 효율성 최적화**

- CPU 환경에서도 실행 가능한 경량화 모델 개발 (TTM 모델 참고)[^1_4]
- 에이전트 수와 경쟁 라운드 수의 최적 균형점 탐색

**4. 평가 메트릭 다양화**

- MAPE, MAE 외에 도메인 특화 메트릭 개발
- 급격한 변동(sudden change) 감지 능력 평가 지표 추가

**5. 윤리적 고려사항**

- 의도적으로 오도하는 정보를 생성하는 IA 메커니즘의 윤리적 함의 고려
- 실제 응용 시 투명성과 설명 가능성 확보 방안 마련


## 6. 2020년 이후 관련 최신 연구 비교 분석

### 6.1 LLM 기반 시계열 예측 연구

**TimeLLM (Jin et al., 2023)**

- 시계열을 언어 공간으로 재프로그래밍하여 LLM의 추론 능력 활용[^1_1]
- 단일 에이전트 접근법으로 멀티 에이전트 경쟁의 이점 부재

**GPT4TS (Zhou et al., 2024)**[^1_1]

- Frozen Pretrained Transformer 방식으로 전이학습 활용
- 본 논문 대비 Electricity 데이터셋에서 MAPE 6.72% vs 6.71%로 유사하나, 뉴스 정보 미활용

**From News to Forecast (Wang et al., 2024b)**[^1_3][^1_2][^1_1]

- 본 논문의 직접적 기반 연구로, LLM 에이전트를 사용하여 관련 뉴스 필터링
- 경쟁 메커니즘 미도입으로 본 논문 대비 성능 낮음 (AC baseline)

**DCATS (2025)**[^1_5][^1_6]

- 메타데이터를 활용한 데이터 중심적 에이전트 프레임워크
- 교통량 데이터에서 6% 오류 감소, 데이터 품질 개선에 집중

**TimeCAP (2025)**[^1_7]

- 두 개의 독립적 LLM 에이전트 사용: 하나는 맥락 요약, 다른 하나는 예측
- 멀티모달 인코더 활용하나 경쟁 메커니즘 부재


### 6.2 멀티 에이전트 시스템 연구

**Multi-Agent Debate (Du et al., 2023)**[^1_1]

- 토론을 통한 LLM의 사실성과 추론 향상
- DoT 문제로 인해 강력한 프롬프트를 가진 단일 에이전트에 뒤처짐

**Encouraging Divergent Thinking (Liang et al., 2024)**[^1_1]

- 멀티 에이전트 토론으로 다양한 사고 촉진
- 본 논문의 AD baseline보다 성능 낮음

**ContestTrade (2024)**[^1_8]

- 내부 경쟁 메커니즘 기반 멀티 에이전트 트레이딩 시스템
- 금융 도메인에 특화, 시계열 예측보다는 자산 배분에 집중

**CompeteAI (Zhao et al., 2024)**[^1_1]

- LLM 기반 에이전트의 경쟁 역학 연구
- 주로 사회 시뮬레이션에 초점, 작업 성능 향상 미흡

**MERIT (2025)**[^1_9]

- 비지도 시계열 표현 학습을 위한 멀티 에이전트 협력
- 3개의 LLM 에이전트가 positive views 생성, 경쟁보다는 협력 중심


### 6.3 차별화 요소

본 논문은 다음과 같은 독특한 기여를 제공합니다:

1. **작업 성능 향상을 위한 경쟁**: 대부분의 경쟁 메커니즘 연구가 사회 시뮬레이션에 집중한 반면, 본 논문은 실제 예측 작업 성능 향상에 초점[^1_1]
2. **정보 비대칭의 전략적 활용**: 일반적인 협력적 에이전트 시스템과 달리, 의도적으로 정보 비대칭과 오도를 허용하여 혁신 촉진[^1_1]
3. **정량적 반성 메커니즘**: MSR에서 파인튜닝된 소규모 LLM으로 정량적 지표를 생성하여 오도하는 논리 식별[^1_1]
4. **사회과학 이론 검증**: LLM 멀티 에이전트 시스템에서 경쟁 강도와 성능 간 U자형 관계를 발견하여 사회과학 이론과의 일치성 확인[^1_1]
5. **종합적 평가**: 4개의 서로 다른 도메인(전기, 환율, 교통, 비트코인)에서 일관된 성능 향상 입증[^1_1]

이러한 차별화 요소들은 본 논문이 단순히 기존 방법을 개선하는 것을 넘어, LLM 기반 멀티 에이전트 시스템 설계에 대한 새로운 패러다임을 제시함을 보여줍니다.
<span style="display:none">[^1_10][^1_11][^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35]</span>

<div align="center">⁂</div>

[^1_1]: 2504.10210v1.pdf

[^1_2]: https://arxiv.org/html/2409.17515v3

[^1_3]: https://arxiv.org/abs/2409.17515

[^1_4]: https://arxiv.org/abs/2401.03955

[^1_5]: https://arxiv.org/html/2508.04231

[^1_6]: https://arxiv.org/html/2508.04231v1

[^1_7]: http://arxiv.org/pdf/2502.11418.pdf

[^1_8]: https://arxiv.org/html/2508.00554v3

[^1_9]: https://aclanthology.org/2025.findings-acl.1231.pdf

[^1_10]: https://arxiv.org/html/2503.01013v2

[^1_11]: https://arxiv.org/html/2602.01776v1

[^1_12]: https://arxiv.org/pdf/2508.04231.pdf

[^1_13]: https://arxiv.org/html/2511.13233v1

[^1_14]: https://arxiv.org/html/2501.06322v1

[^1_15]: https://arxiv.org/abs/2410.17462

[^1_16]: https://ieeexplore.ieee.org/document/10982626/

[^1_17]: https://arxiv.org/abs/2402.04852

[^1_18]: https://arxiv.org/abs/2411.06735

[^1_19]: https://www.mdpi.com/2227-7390/12/23/3666

[^1_20]: http://poster-openaccess.com/article_detail.php?paper_id=762\&conf=ICIC\&year=2024

[^1_21]: https://dl.acm.org/doi/10.1145/3709026.3709103

[^1_22]: http://www.proceedings.com/079017-4463.html

[^1_23]: https://arxiv.org/html/2410.11674

[^1_24]: https://arxiv.org/html/2504.02119v1

[^1_25]: https://arxiv.org/pdf/2503.09656.pdf

[^1_26]: https://arxiv.org/pdf/2403.05798.pdf

[^1_27]: https://arxiv.org/html/2409.17515

[^1_28]: https://arxiv.org/pdf/2402.04852.pdf

[^1_29]: http://arxiv.org/pdf/2503.03594.pdf

[^1_30]: https://www.sciencedirect.com/science/article/pii/S0893608025012432?dgcid=rss_sd_all

[^1_31]: https://openreview.net/pdf?id=iroZNDxFJZ

[^1_32]: https://proceedings.neurips.cc/paper_files/paper/2024/file/6ed5bf446f59e2c6646d23058c86424b-Paper-Conference.pdf

[^1_33]: https://www.arxiv.org/abs/2508.04231

[^1_34]: https://openreview.net/forum?id=ytBLqEkK9D

[^1_35]: https://blog.milvus.io/ai-quick-reference/how-do-agents-compete-in-a-multiagent-system

