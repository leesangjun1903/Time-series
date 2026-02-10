# Detecting drifts in data streams using Kullback-Leibler (KL) divergence measure for data engineering applications

- 이 논문은 ETL 파이프라인 단계에서 **데이터 필드 분포 드리프트를 조기 탐지**하기 위해 KL divergence의 특수형인 Population Stability Index(PSI)를 체계적으로 도입·정당화한 데이터 엔지니어링 관점의 연구다.[^1_1][^1_2][^1_3]
- 네 가지 현실적 시뮬레이션(광고 응답, 매출, 예금, 신용카드 행동점수)을 통해 PSI가 단순 규칙 기반 ETL 검증으로는 놓치기 쉬운 분포 변화와 시스템 오류를 **가볍고 확장 가능한 한 지표로 모니터링**할 수 있음을 보이고, PSI 임계값·bin 개수에 따른 민감도 및 운영상 활용 가이드를 제시한다.[^1_3][^1_1]

***

## 1. 논문의 문제 정의와 핵심 아이디어

### (1) 해결하고자 하는 문제

- 기존 ETL은 스키마·파일 형식·타입 일관성은 잘 검증하지만, **동일 필드의 분포가 시간에 따라 어떻게 달라지는지**는 보지 못한다.[^1_1]
- 복수 소스가 하나의 특성(예: 예금액, 점수)을 공급할 때, 단위/scale/부호 차이, 잘못된 집계, 모델 입력 오류 등으로 **분포가 비정상적으로 바뀌어도 “유효 값”으로 통과**하는 문제가 발생한다.[^1_1]
- 그 결과, downstream ML/규칙 엔진이 **훈련 당시와는 전혀 다른 분포의 입력**으로 자동 의사결정을 내리며, 이 문제는 주로 성능 저하나 이상한 KPI 변동을 통해서야 뒤늦게 발견된다.[^1_1]

논문이 다루는 구체적 질문은 다음과 같다.[^1_2][^1_1]

- 두 시점(또는 두 집단)에서 동일 필드의 분포가 “얼마나 달라졌는지”를 수량화하는 **가볍고 해석 가능한 지표**를,
- **대량·고빈도 데이터 스트림을 처리하는 ETL 파이프라인 내부**에 올려 실시간에 가깝게 돌릴 수 있는가?


### (2) 제안하는 핵심 아이디어

- 정보이론적 거리인 KL divergence를 출발점으로, 그 **대칭형 변형이자 업계에서 널리 쓰이는 PSI**를 데이터 엔지니어링용 드리프트 지표로 재공식화·정당화한다.[^1_1]
- 모든 “모델 입력 필드”에 대해, 기준 분포 $Q$ (예: 과거 한 달)와 현재 분포 $P$ 사이의 PSI를 계산하고, 필드·bin 단위로 드리프트를 감지해 **ETL 레벨에서 파이프라인을 중단 또는 경고**하는 구조를 제안한다.[^1_3][^1_1]

***

## 2. 방법론: 수식, 구조, 실험

### (1) KL divergence와 PSI 수식

이산 확률변수 $x \in \{x_1,\dots,x_B\}$에 대해, 실제 분포 $P(x)$와 기대(기준) 분포 $Q(x)$ 사이의 KL divergence는 다음과 같이 정의된다.[^1_1]

$$
D_{\mathrm{KL}}(P(x)\,\|\,Q(x)) = \sum_{i=1}^{B} P(x_i)\,\ln \frac{P(x_i)}{Q(x_i)}
$$

이는 비대칭이므로, 저자들은 대칭형을 유도한다.[^1_1]

$$
\begin{aligned}
D(P,Q) 
&= D_{\mathrm{KL}}(Q\,\|\,P) + D_{\mathrm{KL}}(P\,\|\,Q) \\
&= \sum_{i=1}^{B} \bigl(P(x_i)-Q(x_i)\bigr)\,\ln \frac{P(x_i)}{Q(x_i)}
\end{aligned}
$$

이 합이 바로 PSI로 정의된다.[^1_1]

$$
\mathrm{PSI} = \sum_{i=1}^{B} \bigl(P(x_i)-Q(x_i)\bigr)\,\ln \frac{P(x_i)}{Q(x_i)}
$$

여기서

- $B$: bin 개수(예: decile, demi-decile),
- $P(x_i)$: “현재/실제” 샘플에서 bin $i$에 속한 비율,
- $Q(x_i)$: “기준/기대” 샘플에서 bin $i$에 속한 비율이다.[^1_1]

실무적 해석용으로 흔히 사용하는 규칙도 재확인한다.[^1_4][^1_1]

- $\mathrm{PSI} < 0.1$: 분포 유사, 유의한 드리프트 없음.
- $0.1 \le \mathrm{PSI} \le 0.2$: 중간 수준 드리프트, 검토 필요.
- $\mathrm{PSI} > 0.2$: 유의한 분포 변화, 즉각적인 조사·조치 필요.


### (2) 제안 “모델/시스템” 구조

이 논문은 예측 모델 구조를 새로 제안하지 않고, **데이터 엔지니어링·MLOps 파이프라인의 한 모듈**로 PSI 기반 드리프트 감지기를 위치시킨다. 구조를 단계로 정리하면:[^1_1]

1. 기준 분포 설정
    - 각 필드별로 기준 기간(예: 지난 해 동일 월, 안정적인 기간)을 선택해 **baseline 분포 $Q$**를 구한다.[^1_1]
2. 실시간/배치 데이터 집계
    - ETL이 수신하는 각 배치(또는 시간 창)에서 동일 필드의 **현재 분포 $P$**를 집계한다.
3. PSI 계산
    - 각 필드를 일정 bin 수 $B$로 구간화(decile 또는 demi-decile)하고, 위 수식으로 PSI와 bin별 “partial PSI”를 계산한다.[^1_1]
4. 임계값 기반 의사결정
    - 필드 전체 PSI 또는 특정 bin의 partial PSI가 임계값을 넘는다면,
        - ETL 파이프라인 중단/보류,
        - 경고 알림 및 수동 검토,
        - 문제 필드만 격리·보정하는 자동 규칙 실행 등을 트리거한다.[^1_1]

이 구조의 특징은 다음과 같다.[^1_3][^1_1]

- **모델 비종속적**: 어떤 ML 모델, 심지어 규칙 기반 시스템에도 동일하게 적용 가능.
- **스케일 적합성**: binning + 집계 수준 연산이므로 SQL, Spark 등 분산 시스템에서 쉽게 구현 가능.
- **설명 가능성**: 각 bin의 기여(partial PSI)를 통해 “어느 구간에서 얼마나 분포가 움직였는지” 직관적 해석이 가능.


### (3) 시뮬레이션과 성능 결과

저자들은 네 개의 대표적인 비즈니스 필드를 선택해, “현실적인 오류 시나리오”를 인위적으로 주입한 후 PSI로 탐지 가능성을 평가한다.[^1_1]

- 기준 샘플($Q$) 설정: 각 필드에 대해 샘플 크기 $100{,}000$으로 평균과 표준편차를 가진 분포 생성.[^1_1]
    - 광고 응답(Ad response): 평균 8000, 표준편차 1000
    - 매출(Sales volume): 평균 350,000
    - 예금(Deposits): 평균 75,000
    - 신용카드 행동점수(CBM score): 평균 610
- 타깃 샘플($P$) 생성: 네 가지 현실적 데이터 문제를 주입.[^1_1]
    - 광고 응답: 플랫폼 누락을 모사하기 위해 10%를 결측으로 설정.
    - 매출: 1사분위(저가 구간)의 50% 관측치에 가격 10% 증가(이중 집계·마이그레이션 오류).
    - 예금: 새 지점 시스템이 금액을 “천 단위”로 보고하여 20% 관측치의 scale mismatch.
    - CBM 점수: 상위 25% 고객의 10%에서 점수를 50점 낮추는 오류(모델 입력 문제).

이후 각 필드에 대해 decile(10 bins)과 demi-decile(20 bins)로 PSI를 계산한 결과는 다음과 같다.[^1_1]


| 필드 | PSI@10 bins | PSI@20 bins | 해석 |
| :-- | --: | --: | :-- |
| Ad response | 0.067 | 0.111 | decile 기준 “무시 가능”, 더 세분화 시 “중간 수준” 드리프트 감지 |
| Sales volume | 0.125 | 0.134 | 일관된 중간 수준 드리프트 |
| Deposits | 0.225 | 0.341 | 명확한 심각 드리프트 (scale 오류 반영) |
| CBM score | 0.171 | 0.228 | 중간~심각 드리프트, 상위 구간에서 집중적 변화 포착 |

주요 관찰점:[^1_1]

- **bin 수 증가 → PSI 증가**: 더 세분화할수록 작은 형태의 분포 변형도 민감하게 포착.
- **부분 PSI 해석력**: 예를 들어 Ad response의 경우, 전체 PSI는 낮지만 첫 번째 bin(하위 구간)에서만 큰 PSI가 나와 “특정 구간 이상치”를 정확히 지목한다.
- **다양한 오류 유형에 대한 민감도**: 결측, scale mismatch, 국소 구간 shift 등 서로 다른 유형의 문제에 일관되게 반응.

논문이 직접적인 “모델 정확도 향상 수치”를 제시하지는 않지만, 드리프트를 조기에 탐지해 잘못된 데이터를 모델에 전달하지 않음으로써 **운영 모델의 성능 붕괴와 잘못된 자동 의사결정(예: 5만 건의 잘못된 카드 갱신 거절)을 방지**하는 것을 주된 성능 기여로 주장한다.[^1_1]

***

## 3. 일반화 성능 관점에서의 해석

### (1) PSI 기반 모니터링이 일반화에 기여하는 경로

이 논문은 “일반화 성능”을 명시적으로 분석하지 않지만, 제안 구조는 다음과 같은 경로로 모델 일반화와 안정성에 기여한다.[^1_5][^1_4][^1_1]

1. **분포 이동의 조기 탐지 → 학습/운영 분포 괴리 관리**
    - 훈련 시점의 입력 분포 $Q$와 운영 시점의 입력 분포 $P$ 차이를 PSI로 정량화하면, 일반화 이론에서 말하는 “train–test distribution shift”를 운영 단계에서 상시 계측하는 것과 같다.
    - $\mathrm{PSI}$가 증가하는 순간을 모델 재학습·feature 엔지니어링·규칙 재설계의 트리거로 사용하면, 모델이 **더 이상 대표성이 없는 데이터**로 운영되는 기간을 최소화할 수 있다.
2. **feature 단위의 국소 drift 파악 → 취약 feature 보완**
    - bin별 partial PSI를 보면, 특정 구간(예: 상위 점수 구간, 고액 예금 구간)에서만 분포 변화가 심한지 알 수 있다.[^1_1]
    - 이는 “일반화가 취약한 영역”을 나타내며, 그 구간에 대해
        - 데이터 수집 강화,
        - calibration,
        - 지역적 재가중(reweighting) 등 일반화 개선 기법을 설계하는 근거가 된다.
3. **모델 불변성/로버스트니스 목표치로 PSI 사용**
    - 실제로 2020년 이후 연구에서는 PSI를 **다중 클라이언트/도메인 간 분포 불일치 측정치**로 사용해, 클러스터형 퍼스널라이즈드 FL 구조나 drift-aware 파이프라인의 핵심 구성요소로 사용하는 경향이 뚜렷하다.[^1_6][^1_7][^1_8][^1_4][^1_5]
    - 이 논문은 ETL 수준에서 PSI를 표준 도입하자고 제안함으로써, 이후 학습/모델링 단계에서 **“허용 가능한 분포 변형 범위”를 수량화한 제약 조건**을 설계할 수 있는 기반을 제공한다.

요약하면, 이 논문은 **PSI를 모델 외부에서의 ‘환경 감지 센서’로 표준화**하려는 시도로 볼 수 있고, 이는 일반화 성능을 간접적으로지만 강하게 규제하는 메커니즘을 제공한다.

### (2) 모델 일반화 향상을 위한 한계와 보완 필요 지점

한계 역시 명확하다.[^1_9][^1_4][^1_5][^1_1]

- PSI는 **입력 분포의 1차적 차이만 본다**.
    - 입력–레이블 관계(개념 드리프트) 변화는 PSI만으로는 직접 포착하기 어렵다.
    - 예를 들어, CBM 점수 분포는 그대로인데 “같은 점수에서 연체율이 두 배로 증가”하면 PSI는 작게 나와도 모델은 심각하게 망가진다.
- binning 의존성
    - 어떤 binning을 선택하느냐에 따라 PSI 값과 민감도가 크게 변한다. 논문도 bin 개수 증가 시 PSI가 계속 증가함을 보고하고, “최적 binning은 향후 연구 과제”라고 명시한다.[^1_1]
- 임계값의 경험적·도메인 의존성
    - $\mathrm{PSI} < 0.1,\ 0.1\sim0.25,\ >0.25$와 같은 규칙은 금융 업계의 경험 법칙에 가깝고, 대규모 스트리밍·멀티모달 환경에서 false positive/negative cost를 반영한 이론적 기준은 부재하다.[^1_4][^1_1]

따라서 일반화 성능 측면에서 PSI는

- “**필수지만 충분하지 않은**” 모듈이며,
- 레이블 없는 drift 탐지를 위한 통계적 지표로써,
- 레이블 기반 성능 모니터링, representation-level drift 검출, 불확실성 추적 등과 결합될 때 비로소 **완전한 일반화 모니터링 체계**를 이룰 수 있다.

***

## 4. 2020년 이후 관련 최신 연구와의 비교

2020년 이후 PSI·KL 기반 드리프트 탐지 연구 흐름을 이 논문과 대비해 정리하면 다음과 같다.

### (1) PSI 확장·활용 연구

- **Representation-level feature drift 관리**
    - 최근 representation learning 기반 feature drift 연구는 PSI와 KS, Energy Distance 등을 baseline 지표로 사용하면서, latent representation 상의 drift를 직접 측정하는 방식을 제안한다.[^1_10][^1_11]
    - 이 논문은 **원시 feature 분포 수준의 PSI**에 머무는 반면, 최신 연구는 encoder를 포함한 **representation space 단에서의 일반화 붕괴**를 다룬다는 점에서 범위가 다르다.
- **연합학습(FL)에서의 PSI 기반 클라이언트 군집화**
    - Clust-PSI-PFL은 PSI로 클라이언트 간 레이블 분포 차이를 측정하고, 유사한 클라이언트들을 클러스터링해 **per-cluster 모델**을 학습함으로써 non-IID 상황에서 최대 18% 정확도 향상을 보고한다.[^1_6]
    - Kurian \& Allali는 ETL 레벨 분포 차이 탐지에 집중하고, FL 환경·학습 알고리즘에는 들어가지 않는다. PSI의 위치가 “데이터 파이프라인 전단 vs. 분산 학습 중간”이라는 차이가 있다.
- **드리프트 탐지 툴킷에서의 PSI**
    - 2024년 이후 “Open-Source Drift Detection Tools in Action” 및 NannyML/Arize 등의 실무 튜토리얼은 PSI를 KS, Jensen–Shannon, Hellinger 등과 조합한 **복수 지표 체계**를 권장하면서, PSI의 장단점을 요약한다.[^1_12][^1_5][^1_4]
    - 이 논문은 PSI 단일 지표에 집중하지만, 최근 툴 연구는 **멀티 메트릭·멀티 레벨(입력, 출력, 에러)의 drift 모니터링**을 강조하는 방향이다.


### (2) KL divergence 기반 스트림 드리프트 탐지

- **KL 기반 온·라인 스트림 감지(KLD)**
    - Basterrech \& Wozniak(2022)은 히스토그램 기반으로 연속적인 데이터 chunk의 분포를 추정하고, KL divergence와 임계값 $\alpha$로 concept drift를 감지하는 KLD 알고리즘을 제안한다.[^1_13]
    - 이는 예측 성능을 직접 감지하려는 “모델-연계” 접근이고, Kurian \& Allali는 모델과 분리된 “데이터 분포 감시”에 집중한다.
- **하이브리드(통계 + representation) drift pipeline**
    - Transformer–Autoencoder 에러, PSI, JSD, 불확실성, 규칙 위반 등을 통합한 Trust Score 기반 실시간 drift pipeline이 제안되기도 한다.[^1_7][^1_8]
    - Kurian \& Allali의 기여는 이 복합 체계의 “PSI 모듈”에 해당하는 부분을 **수학적으로 정리하고, ETL 내 임계값·binning 전략을 구체적 사례로 보여준 점**이라고 볼 수 있다.


### (3) 비교 요약 표

| 축 | Kurian \& Allali (2024) | Basterrech \& Wozniak (2022)[^1_13] | Clust-PSI-PFL (2025)[^1_6] | Drift 툴/파이프라인 연구[^1_12][^1_7][^1_8][^1_4][^1_5] |
| :-- | :-- | :-- | :-- | :-- |
| 지표 | PSI (KL 대칭형) | KL divergence | PSI | PSI + JSD + KS + 기타 거리 |
| 초점 | ETL 내 데이터 필드 분포 드리프트 | 온·라인 concept drift | FL에서 non-IID 클라이언트 군집 | 운영 모델 전체 drift 모니터링 |
| 모델 의존성 | 모델 비종속 | base predictor 필요 | FL 학습 loop 내 | 모델/feature/에러 모두 사용 |
| 일반화와의 연결 | 입력 분포 안정성 → 간접 | 예측 성능 붕괴 직접 감지 | 클러스터별 모델로 non-IID 완화 | drift–성능 상관분석, retrain 정책 |
| 공헌 위치 | 데이터 파이프라인 전단 | 스트림 학습 알고리즘 | 연합학습 알고리즘 | MLOps/도구/엔드투엔드 파이프라인 |

Kurian \& Allali 논문은 **“PSI를 데이터 엔지니어링 레벨의 표준 드리프트 지표로 재정리한 2024년대의 대표적 실무 지향 논문”**이라 볼 수 있고, 이후 연구들은 이를 포함한 다양한 거리·representation 지표를 조합해 보다 직접적으로 일반화·성능을 다룬다.

***

## 5. 앞으로의 연구 영향과 향후 고려사항

### (1) 이 논문이 미치는 영향

1. **데이터 엔지니어링–ML 사이의 단절 완화**
    - PSI를 ETL 단계에 통합하자는 제안은 모델러가 아니라 데이터 엔지니어가 **분포 드리프트를 1차로 책임지고 감지**하도록 역할을 재정의한다.[^1_1]
    - 이는 “데이터 관측성(data observability)”과 “모델 운영(MLOps)” 사이에 **공유 언어(PSI)**를 제공한다는 점에서 의미가 크다.
2. **간단하지만 강력한 baseline의 확립**
    - PSI는 계산이 단순하고 해석이 직관적이라, 고급 representation-level 방식과 비교할 때도 **항상 baseline으로 포함될 가치**가 있다.[^1_5][^1_4][^1_1]
    - 이후 연구들에서 PSI는 거의 모든 drift detection 툴과 논문에서 기본 지표로 등장하는데, 이 논문은 그 중에서도 **대규모 데이터 엔지니어링 맥락**을 강조한 사례로 많이 인용될 가능성이 있다.[^1_12][^1_4][^1_5]
3. **비즈니스 리스크–통계 지표의 연결**
    - 카드 자동갱신, 예금, 광고 응답 등 구체 사례를 통해 “PSI 값 ↔ 실제 운영 리스크(잘못된 자동 의사결정·고객 경험 악화)”를 직관적으로 연결해 준다.[^1_1]
    - 이후 공정성·규제 영역(예: 신용평점 under drift)에서 PSI 기반 규정·감독 기준 설계의 근거로 활용될 여지가 있다.[^1_14][^1_15][^1_9]

### (2) 앞으로 연구 시 고려할 점

1. **PSI 임계값의 이론화 및 비용 기반 최적화**
    - 현재 임계값은 경험 법칙이다.
    - 향후 연구에서는
        - false positive/false negative에 대한 **비용 함수 $C_{\mathrm{FP}},C_{\mathrm{FN}}$**를 정의하고,
        - drift 빈도·강도 분포를 모델링해, 기대 비용을 최소화하는 $\tau^\ast$를

$$
\tau^\ast = \arg\min_{\tau} \mathbb{E}\bigl[C_{\mathrm{FP}}(\tau) + C_{\mathrm{FN}}(\tau)\bigr]
$$

형태로 구하는 방향이 필요하다.
2. **적응적 binning 및 고차원 확장**
    - 균일 decile 대신,
        - 데이터 밀도 기반 binning,
        - quantile + domain knowledge 혼합 binning,
        - 다변량 PSI(공분포에 대한 상대엔트로피 근사) 등을 탐구할 필요가 있다.[^1_9][^1_4][^1_5][^1_1]
3. **PSI와 representation-level drift, 성능 지표의 통합**
    - 입력 PSI, latent drift(예: DriftLens), 예측 불확실성, 에러율(정확도, AUC 등)을 하나의 **joint drift–성능 모형**으로 결합하면,
        - 어떤 유형의 입력 drift가 실제로 일반화 성능을 얼마나 악화시키는지,
        - 어떤 feature의 drift는 허용 가능하고, 어떤 것은 즉각 조치가 필요한지
를 데이터 기반으로 규정할 수 있다.[^1_8][^1_16][^1_17][^1_13][^1_7]
4. **도메인별 특화 연구: 금융, 의료, 공공 데이터**
    - 금융: 신용점수/사기 탐지에서 PSI는 이미 규제기관·감독기관에 친숙한 지표이므로, PSI 기반 drift 경보를 **규제 보고 체계**에 편입하는 연구가 가능하다.[^1_14][^1_9][^1_4]
    - 의료·공공: 코호트 변화, 정책 변화 등에 의한 분포 이동을 PSI로 추적하고, fairness·bias와 연계하는 연구가 필요하다.[^1_15]
5. **실제 대규모 스트리밍 환경에서의 벤치마킹**
    - 논문은 시뮬레이션 기반이므로, 대형 로그/센서/거래 데이터에서
        - PSI vs 다른 거리(JSD, Hellinger, EMD 등)의 감도·지연시간·운영 비용,
        - 온라인 업데이트/슬라이딩 윈도우에서의 안정성
을 체계적으로 비교하는 벤치마크 연구가 뒤따라야 한다.[^1_18][^1_12][^1_5]

***

## 6. 정리: 연구자로서의 활용 포인트

연구자 관점에서 이 논문은 다음과 같이 활용할 수 있다.

- **데이터 엔지니어링–MLOps 파이프라인 설계 시**: 최소한의 drift 센서로 PSI 모듈을 도입하는 설계 근거.
- **새 drift detection 알고리즘 제안 시**: PSI를 baseline으로 삼고,
    - 검출 지연, false alarm rate, 성능 유지율에서의 우수성을 정량 비교하는 실험 설계에 참고.
- **일반화/로버스트니스 이론 연구 시**: 현실 시스템에서 실제로 채택되는 간단한 지표(PSI)를 고려해, “실무적으로 관측 가능한 신호” 위에 있는 이론 모델을 세우는 방향으로의 브리지 역할.

필요하다면, 위 내용을 바탕으로 PSI 기반 drift 모듈을 포함한 **엔드투엔드 MLOps/데이터 관측성 아키텍처** 스케치를 LaTeX 문서 형태로도 정리해 드릴 수 있다.
<span style="display:none">[^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35]</span>

<div align="center">⁂</div>

[^1_1]: s42488-024-00119-y.pdf

[^1_2]: https://digitalcommons.chapman.edu/cgi/viewcontent.cgi?article=1199\&context=engineering_articles

[^1_3]: https://colab.ws/articles/10.1007%2Fs42488-024-00119-y

[^1_4]: https://www.fiddler.ai/blog/measuring-data-drift-population-stability-index

[^1_5]: https://www.nannyml.com/blog/population-stability-index-psi

[^1_6]: https://arxiv.org/html/2512.20363

[^1_7]: https://arxiv.org/html/2508.07085v1

[^1_8]: https://arxiv.org/pdf/2508.07085.pdf

[^1_9]: https://www.mdpi.com/2227-9091/7/2/53/pdf?version=1557125420

[^1_10]: https://arxiv.org/html/2505.10325v1

[^1_11]: https://arxiv.org/html/2505.10325v2

[^1_12]: https://arxiv.org/pdf/2404.18673.pdf

[^1_13]: https://arxiv.org/pdf/2210.04865.pdf

[^1_14]: https://arxiv.org/pdf/2511.03807.pdf

[^1_15]: https://www.kihasa.re.kr/api/kihasa/file/download?seq=32609

[^1_16]: https://arxiv.org/html/2406.17813v2

[^1_17]: https://arxiv.org/html/2505.17902v3

[^1_18]: https://towardsdatascience.com/drift-detection-in-robust-machine-learning-systems/

[^1_19]: https://arxiv.org/html/2404.18673v2

[^1_20]: https://arxiv.org/html/2512.20631v1

[^1_21]: https://arxiv.org/html/2601.15544v1

[^1_22]: https://arxiv.org/html/2601.08928v1

[^1_23]: https://www.arxiv.org/pdf/2601.08928.pdf

[^1_24]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11126395/

[^1_25]: https://onlinelibrary.wiley.com/doi/10.1002/psp.2637

[^1_26]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9877951/

[^1_27]: https://pmc.ncbi.nlm.nih.gov/articles/PMC5889633/

[^1_28]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11122687/

[^1_29]: https://onlinelibrary.wiley.com/doi/pdfdirect/10.1111/gcb.16841

[^1_30]: https://onlinelibrary.wiley.com/doi/pdfdirect/10.1111/padr.12464

[^1_31]: https://www.geeksforgeeks.org/data-science/population-stability-index-psi/

[^1_32]: https://velog.io/@h-go-getter/두-분포의-차이를-검증하는-성능-지표-PSIPopulation-Stability-Index

[^1_33]: https://abluesnake.tistory.com/163

[^1_34]: https://arize.com/blog-course/population-stability-index-psi/

[^1_35]: https://d-nb.info/1338111809/34

