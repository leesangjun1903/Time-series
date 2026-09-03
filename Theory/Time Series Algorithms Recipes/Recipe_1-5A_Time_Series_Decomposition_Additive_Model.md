# Recipe 1-5A. Time Series Decomposition — Additive Model

> 학습 수준: **대학/대학원 → 연구자 발전형**  
> 원문 위치: **PDF 36–39 / book pp.21–24**  
> 기준 문헌: *Time Series Algorithms Recipes* (Apress, 2023)

## 1. Executive Summary — 10문장 이내

1. 이 절은 관측값을 trend, seasonality, remainder 같은 구조적 성분으로 나누어 이해하는 방법을 다룹니다.
2. 분해는 그래프를 보기 좋게 만드는 기법이 아니라 데이터 생성 구조를 가설화하는 통계적 도구입니다.
3. 가법·승법 여부와 계절 주기는 도메인 지식과 out-of-sample 검증으로 결정해야 합니다.
4. 분해 후 residual에 자기상관이 남는다면 아직 예측 가능한 구조를 제거하지 못한 것입니다.
5. 최근 연구는 MSTL, DLinear, TimeMixer처럼 다중 계절성과 학습 가능한 decomposition으로 확장되고 있습니다.
6. 본 학습자료는 Recipe 1-5A “Time Series Decomposition — Additive Model”의 원문을 출발점으로 하되, 대학원 수준의 수학적 정의와 2020년 이후 연구를 연결합니다.

## 2. 목적과 필요성

이 Recipe를 공부하는 목적은 Python API를 외우는 것이 아니라, 해당 절이 가정하는 데이터 생성 구조와 평가 조건을 이해하는 것입니다. 시계열 연구에서는 동일한 알고리즘이라도 시간축 처리, validation 방식, feature availability에 따라 성능이 크게 달라집니다. 특히 연구자로 발전하려면 “코드가 실행된다”와 “통계적으로 타당한 예측 실험이다”를 구분해야 합니다.

> **용어 설명 — detrending**  
> 추세 성분을 추정한 뒤 원자료에서 제거하여 단기 변동을 분석하는 과정입니다.

> **용어 설명 — white noise**  
> 평균이 안정되고 시간 자기상관이 거의 없는 예측 불가능한 오차 과정입니다.

## 3. 이론 강의


### 3.1 가법 분해

가법 모델은

$$
Y_t=T_t+S_t+R_t
$$

로 표현합니다. $T_t$는 trend, $S_t$는 seasonal component, $R_t$는 remainder입니다. 계절 진폭이 시계열 수준에 크게 비례하지 않을 때 자연스럽습니다.

- 시계열 데이터에서 remainder(잔차 또는 불규칙 성분, Irregular/Residual component)란 전체 데이터에서 트렌드(Trend)와 계절성(Seasonality)을 제거하고 남은 예측 불가능한 무작위 변동 성분을 말합니다.

잔차가 진정한 잡음에 가까우려면 최소한

$$
\text{E}[R_t]\approx0,\qquad \text{Cov}(R_t,R_{t-k})\approx0\;(k\neq0)
$$

가 기대됩니다. 따라서 분해가 끝났다는 이유만으로 $R_t$를 white noise라고 부르면 안 되고, ACF·Ljung–Box·이분산성 검사를 추가해야 합니다.

> **용어 설명 — remainder/residual**  
> 추세와 계절 구조를 제거하고 남은 부분입니다. 구조가 충분히 제거되지 않았다면 여전히 예측 가능한 신호를 포함할 수 있습니다.




### 수식 기호 상세 해설

- $Y_t$: 관측값, $T_t$: trend, $S_t$: seasonality, $R_t$: residual입니다.
- 가법식의 `+`는 각 성분의 효과 크기가 원래 데이터 단위에서 더해진다는 뜻입니다.
- $\text{E}[R_t]$: residual의 기대평균이며 좋은 분해라면 0에 가까운 것이 바람직합니다.
- $\text{Cov}(R_t,R_{t-k})$: residual과 $k$시점 전 residual의 공분산입니다.
- $k\neq0$: 같은 시점이 아닌 시간 지연에서 잔차 상관이 남지 않는지를 확인한다는 뜻입니다.

### 3.3 무엇을 학습하거나 추정하는가

이 절의 핵심 추정 대상은 데이터의 시간 구조를 설명하는 **파라미터 또는 상태**입니다. 통계모델이라면 계수와 오차분산을 추정하고, tree/deep model이라면 예측손실을 최소화하는 함수 $f_\theta$를 학습합니다. 공통적인 연구 목표는 새로운 미래 구간에서의 기대 손실

$$
R_{\text{future}}(f)=\text{E}\left[L(Y_{t+h},f(\mathcal{I}_t))\right]
$$

을 낮추는 것입니다. 예측 위험식의 기호는 다음과 같습니다. $R_{\text{future}}(f)$는 아직 보지 못한 미래 분포에서의 기대 예측손실, $\text{E}[\cdot]$는 그 분포에 대한 기대값, $Y\_{t+h}$는 $h$단계 미래의 실제값, $f$는 예측함수, $\mathcal{I}\_t$는 시점 $t$까지 이용 가능한 정보, $L(\cdot,\cdot)$은 실제값과 예측값의 오차를 수치화하는 손실함수입니다. $f=f_\theta$로 쓸 때 $\theta$는 데이터에서 학습하는 모델 파라미터 전체를 뜻합니다.

> **용어 설명 — information set $\mathcal{I}_t$**  
> 예측 순간에 현실적으로 알고 있는 모든 정보의 집합입니다. 미래 target이나 미래에만 측정되는 sensor 값을 포함하면 데이터 누수입니다.

## 4. 원문이 직접 보고한 내용

저자는 가법 분해에서 trend, seasonality, residual이 합으로 결합된다고 설명하고 quarterly retail turnover를 분해합니다.

이 문단은 **저자의 보고를 요약한 것**이며, 아래의 비판적 해석과 구분해야 합니다.

## 5. 비판적 해석 및 연구자 관점

분해는 예측 전처리라기보다 구조 진단 도구입니다. residual에 예측 가능한 구조가 남는지 확인해야 합니다.

또한 이 절의 결과를 다른 Recipe와 비교할 때는 데이터셋, target scale, forecast horizon, split, metric이 동일한지 확인해야 합니다. 조건이 다르면 숫자의 크기만으로 모델 우열을 말할 수 없습니다.

## 6. 통계적으로 취약한 부분과 비교 불가능한 수치

고전 seasonal_decompose는 edge에서 trend 추정이 불안정하고 multiple seasonality·시간가변 계절성을 잘 다루지 못합니다. residual whiteness 검증이 없습니다.

**비교 가능성 판정:** 이 Recipe의 숫자는 동일한 데이터·동일한 forecast origin·동일한 horizon·동일한 metric을 사용한 실험끼리만 직접 비교해야 합니다. 다른 target이나 단위가 다른 RMSE는 직접 크기 비교를 하지 않습니다.

## 7. 문서가 답하지 않는 질문과 답변

**질문 1. 분해한 residual은 white noise인가요?**  
아닙니다. ACF/Ljung–Box/ARCH test 등으로 남은 구조를 확인해야 합니다.

**질문 2. additive와 multiplicative 중 무엇을 선택하나요?**  
계절 진폭이 수준과 함께 커지는지, 로그 변환 후 분산이 안정되는지, validation forecasting error가 개선되는지를 종합합니다.

**질문 3. 분해를 전체 데이터에 먼저 해도 되나요?**  
forecasting 평가에서는 위험합니다. centered smoother나 STL이 미래값을 사용할 수 있으므로 각 training fold 안에서 fit해야 합니다.

## 8. 2020년 이후 관련 최신 연구 비교 분석

고전적 단일 계절 decomposition은 여러 계절 주기를 충분히 다루지 못할 수 있습니다. MSTL은 STL을 다중 계절성으로 확장했고, DLinear는 decomposition을 단순 선형 forecasting과 결합했으며, TimeMixer는 여러 sampling scale에서 trend/seasonal 성분을 섞는 구조를 제안했습니다. 연구 방향은 '분해 후 예측'에서 '분해 자체를 학습 구조에 통합'하는 쪽으로 확장되었습니다.

| 연구 | 핵심 기술 변화 | 이 Recipe에 주는 의미 |
|---|---|---|
| MSTL, 2021/2022 | multiple seasonal-trend decomposition | 시간/일/주 등 복수 계절성에 적합 |
| DLinear, 2023 | decomposition + linear forecasting | 복잡한 Transformer가 항상 필요한 것은 아님 |
| TimeMixer, 2024 | multiscale decomposition/mixing | 여러 시간 scale을 동시에 모델링 |

### 최신 연구 결과를 읽을 때의 주의점

논문에서 “SOTA”라고 보고된 수치는 특정 benchmark, split, horizon, preprocessing, tuning budget에 종속됩니다. 특히 foundation model은 pretraining corpus가 benchmark와 얼마나 겹치는지, zero-shot인지 fine-tuned인지에 따라 비교 조건이 달라집니다. 따라서 **서로 다른 논문의 숫자를 한 표에 놓고 단순 순위화하지 않습니다.**

## 9. 실제 파이프라인 적용 방향

1. train 구간에서만 trend/seasonal 구조를 추정합니다.  
2. 단일/다중 계절 주기 후보를 도메인과 periodogram으로 확인합니다.  
3. additive와 log-additive(multiplicative) 후보를 비교합니다.  
4. residual whiteness와 heteroskedasticity를 검사합니다.  
5. 분해를 사용한 예측과 사용하지 않은 예측을 rolling-origin으로 비교합니다.

### 일반화 성능을 높이기 위한 연구 방향

- 모델 복잡도를 먼저 키우기보다 **시간순 교차검증과 leakage 제거**로 평가분산을 줄입니다.
- 예측 horizon별로 필요한 구조가 다르므로 $h=1$과 long-horizon을 분리하여 튜닝합니다.
- 여러 seed 또는 여러 rolling origin에서 평균과 표준편차를 보고 selection noise를 줄입니다.
- 작은 데이터에서는 low-capacity statistical/tree baseline을 유지하고, deep/foundation model이 실제로 유의한 개선을 주는지 검증합니다.
- domain shift가 예상되면 최근 구간 가중, calibration, covariate shift 진단, fine-tuning을 별도 실험합니다.

## 10. 후속 연구 계획

이 Recipe를 단독 실습으로 끝내지 않고, 동일 데이터에 대해 **고전 baseline → leakage-safe validation → 최신 모델**의 순서로 비교하는 것이 좋습니다. 후속 연구에서는 (1) rolling-origin 반복평가, (2) residual diagnostic, (3) 모델 복잡도 대비 성능, (4) domain shift 구간 성능, (5) uncertainty calibration을 공통 실험 프로토콜로 고정합니다. 이렇게 해야 모델 이름이 바뀌어도 동일한 연구 질문 아래에서 비교할 수 있습니다.

## 11. 참고자료 및 출처

- **Time Series Algorithms Recipes: Implement Machine Learning and Deep Learning Techniques with Python** — Akshay R. Kulkarni, Adarsha Shivananda, Anoosh Kulkarni, V. Adithya Krishnan, Apress, 2023. https://doi.org/10.1007/978-1-4842-8978-5
- **Forecast evaluation for data scientists: common pitfalls and best practices** — Hewamalage, Ackermann, Bergmeir, published online 2022; Data Mining and Knowledge Discovery 37 (2023). https://doi.org/10.1007/s10618-022-00894-5
- **MSTL: A Seasonal-Trend Decomposition Algorithm for Time Series with Multiple Seasonal Patterns** — Bandara, Hyndman, Bergmeir, International Journal of Operational Research 52(1), 2025 (preprint 2021). https://doi.org/10.1504/IJOR.2025.143957
- **Are Transformers Effective for Time Series Forecasting?** — Zeng, Chen, Zhang, Xu, AAAI 2023. https://doi.org/10.1609/aaai.v37i9.26317
- **TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting** — Wang et al., ICLR 2024. https://arxiv.org/abs/2405.14616
- **NHITS: Neural Hierarchical Interpolation for Time Series Forecasting** — Challu et al., AAAI 2023. https://doi.org/10.1609/aaai.v37i6.25854

---

### 학습 체크포인트

이 절을 공부한 뒤에는 **정의 → 수식 → 가정 → 추정 대상 → 평가 설계 → 누수 가능성 → 최신 연구와의 차이**를 자신의 말로 설명할 수 있어야 합니다. 코드를 외우는 것보다 “어떤 조건에서 이 방법이 맞고, 어떤 조건에서 실패하는가”를 설명하는 능력이 연구 단계의 목표입니다.
