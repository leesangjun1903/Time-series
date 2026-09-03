# Recipe 2-5. Grid Search Hyperparameter Tuning for ARIMA

> 학습 수준: **대학/대학원 → 연구자 발전형**  
> 원문 위치: **PDF 68–74 / book pp.54–60**  
> 기준 문헌: *Time Series Algorithms Recipes* (Apress, 2023)

## 1. Executive Summary — 10문장 이내

1. 이 절은 비교적 적은 파라미터로 시간 의존성을 설명하는 통계적 forecasting 원리를 학습하는 부분입니다.
2. ARIMA·ETS 계열의 강점은 해석 가능성, 작은 데이터에서의 안정성, 빠른 추정입니다.
3. 정상성·차분·lag 선택과 residual diagnostics를 올바르게 이해하는 것이 알고리즘 이름을 외우는 것보다 중요합니다.
4. 단일 test 구간에 맞춰 하이퍼파라미터를 선택하면 일반화 성능을 과대평가할 수 있습니다.
5. 최신 deep/foundation model과 비교할 때도 고전 모델은 반드시 유지해야 할 강한 baseline입니다.
6. 본 학습자료는 Recipe 2-5 “Grid Search Hyperparameter Tuning for ARIMA”의 원문을 출발점으로 하되, 대학원 수준의 수학적 정의와 2020년 이후 연구를 연결합니다.

## 2. 목적과 필요성

이 Recipe를 공부하는 목적은 Python API를 외우는 것이 아니라, 해당 절이 가정하는 데이터 생성 구조와 평가 조건을 이해하는 것입니다. 시계열 연구에서는 동일한 알고리즘이라도 시간축 처리, validation 방식, feature availability에 따라 성능이 크게 달라집니다. 특히 연구자로 발전하려면 “코드가 실행된다”와 “통계적으로 타당한 예측 실험이다”를 구분해야 합니다.

> **용어 설명 — stationarity**  
> 확률분포의 핵심 특성이 시간 이동에 따라 변하지 않는 성질입니다.

> **용어 설명 — lag**  
> 현재보다 몇 시점 이전의 관측값을 뜻합니다.

> **용어 설명 — AIC/BIC**  
> likelihood 적합도와 모델 복잡도를 함께 평가하는 정보기준입니다.

## 3. 이론 강의


### 3.1 하이퍼파라미터 선택은 최적화 문제가 아니라 추정 문제입니다

ARIMA 후보 집합 $\mathcal{H}$에서 RMSE가 가장 작은 조합을 선택한다고 쓰면

$$
\hat{h}=\arg\min_{h\in\mathcal{H}}\frac{1}{n_{\text{val}}}\sum_{t\in\text{Val}}(Y_t-\hat{Y}^{(h)}_t)^2.
$$

핵심은 $\text{Val}$이 **최종 Test와 달라야 한다는 점**입니다. Test를 이용해 $h$를 선택하고 같은 Test에서 성능을 보고하면 selection-induced optimism이 발생합니다.

> **용어 설명 — validation overfitting**  
> 많은 후보를 반복적으로 비교하면서 검증 구간 특성에 우연히 잘 맞는 모델을 선택하는 현상입니다. 검증 데이터도 사실상 학습 과정의 일부가 됩니다.

시계열에서는 단일 validation 구간보다 rolling-origin CV가 안정적입니다.

$$
\text{CV}(h)=\frac{1}{K}\sum_{k=1}^{K}L\left(Y_{t_k+1:t_k+H},\hat{Y}^{(h)}_{t_k+1:t_k+H}\right).
$$

마지막 untouched test는 단 한 번만 사용해야 합니다.


### 3.2 이 Recipe에서 반드시 바로잡아야 할 평가 설계

책의 grid search는 `test_data`를 각 후보의 RMSE 계산에 반복 사용합니다. 이는 수식으로

$$
\hat{h}=\arg\min_h L_{\text{test}}(h)
$$

를 수행한 뒤 다시 $L_{\text{test}}(\hat{h})$를 최종 test 성능이라고 보고하는 구조입니다. 이 값은 독립 test가 아니라 **튜닝된 validation 성능**입니다. 연구용 구현은 Train → Validation/CV → 최종 Test의 3단 구조로 바꿔야 합니다.

### 수식 기호 상세 해설

- $\mathcal{H}$: 탐색할 모든 하이퍼파라미터 후보의 집합입니다.
- $h$: 이 식에서는 forecast horizon이 아니라 하나의 하이퍼파라미터 설정을 나타내는 기호입니다.
- $\hat{h}$: validation loss가 가장 작은 최종 선택 설정입니다.
- $n_{\text{val}}$: validation 관측 개수, $\text{Val}$: validation 시간 인덱스 집합입니다.
- $Y_t$: 실제값, $\hat{Y}^{(h)}_t$: 설정 $h$로 만든 예측값입니다.
- $K$: rolling-origin 평가 횟수, $t_k$: $k$번째 forecast origin, $H$: 각 origin에서 평가하는 forecast horizon 길이입니다.
- $L(\cdot,\cdot)$: MAE·MSE 등 검증 손실함수입니다.

### 3.3 무엇을 학습하거나 추정하는가

이 절의 핵심 추정 대상은 데이터의 시간 구조를 설명하는 **파라미터 또는 상태**입니다. 통계모델이라면 계수와 오차분산을 추정하고, tree/deep model이라면 예측손실을 최소화하는 함수 $f_\theta$를 학습합니다. 공통적인 연구 목표는 새로운 미래 구간에서의 기대 손실

$$
R_{\text{future}}(f)=\text{E}\left[L(Y_{t+h},f(\mathcal{I}_t))\right]
$$

을 낮추는 것입니다. 예측 위험식의 기호는 다음과 같습니다. $R_{\text{future}}(f)$는 아직 보지 못한 미래 분포에서의 기대 예측손실, $\text{E}[\cdot]$는 그 분포에 대한 기대값, $Y_{t+h}$는 $h$단계 미래의 실제값, $f$는 예측함수, $\mathcal{I}_t$는 시점 $t$까지 이용 가능한 정보, $L(\cdot,\cdot)$은 실제값과 예측값의 오차를 수치화하는 손실함수입니다. $f=f_\theta$로 쓸 때 $\theta$는 데이터에서 학습하는 모델 파라미터 전체를 뜻합니다.

> **용어 설명 — information set $\mathcal{I}_t$**  
> 예측 순간에 현실적으로 알고 있는 모든 정보의 집합입니다. 미래 target이나 미래에만 측정되는 sensor 값을 포함하면 데이터 누수입니다.

## 4. 원문이 직접 보고한 내용

저자는 p,d,q를 각각 0–3 범위에서 grid search하고 ARIMA(1,2,0)을 best configuration으로 선택하여 RMSE 725.218을 보고합니다.

이 문단은 **저자의 보고를 요약한 것**이며, 아래의 비판적 해석과 구분해야 합니다.

## 5. 비판적 해석 및 연구자 관점

grid search의 핵심 연구 교훈은 “더 많이 탐색”이 아니라 “validation과 test를 엄격히 분리”하는 것입니다.

또한 이 절의 결과를 다른 Recipe와 비교할 때는 데이터셋, target scale, forecast horizon, split, metric이 동일한지 확인해야 합니다. 조건이 다르면 숫자의 크기만으로 모델 우열을 말할 수 없습니다.

## 6. 통계적으로 취약한 부분과 비교 불가능한 수치

**중대한 평가 누수**가 있습니다. 여러 ARIMA 조합을 `test_data` RMSE로 선택하고 같은 test RMSE를 최종 성능으로 보고하므로 test가 validation 역할을 해버립니다. 이 725.218은 독립적인 최종 일반화 추정치가 아닙니다.

**비교 가능성 판정:** 이 Recipe의 숫자는 동일한 데이터·동일한 forecast origin·동일한 horizon·동일한 metric을 사용한 실험끼리만 직접 비교해야 합니다. 다른 target이나 단위가 다른 RMSE는 직접 크기 비교를 하지 않습니다.

## 7. 문서가 답하지 않는 질문과 답변

**질문 1. 정상성이 반드시 필요한가요?**  
ARMA의 고전 이론에는 중요하지만 ARIMA는 차분을 통해 비정상성을 다룹니다. 예측 목적에서는 모델 residual과 OOS 성능을 함께 봐야 합니다.

**질문 2. ACF/PACF만으로 차수를 정해도 되나요?**  
초기 후보에는 유용하지만 noisy finite sample에서는 불확실합니다. 정보기준과 rolling-origin을 함께 사용해야 합니다.

**질문 3. ARIMA가 최신 모델보다 낡아서 쓸 필요가 없나요?**  
그렇지 않습니다. 작은 데이터·강한 seasonality·해석성·빠른 재학습이 중요하면 여전히 매우 강한 baseline입니다.

## 8. 2020년 이후 관련 최신 연구 비교 분석

2020년 이후에도 ARIMA/ETS는 강한 baseline입니다. 다만 N-BEATS, N-HiTS, PatchTST, Chronos, TimesFM 등은 대규모·다양한 데이터나 장기 horizon에서 강점을 보였습니다. 최신 비교에서는 고전 모델이 특정 도메인과 충분한 feature engineering에서 foundation model과 대등할 수 있다는 결과도 있어, 모델 세대만으로 우열을 결정해서는 안 됩니다.

| 연구 | 핵심 기술 변화 | 이 Recipe에 주는 의미 |
|---|---|---|
| N-BEATS, ICLR 2020 | residual MLP forecasting | 통계 baseline 대비 강한 global nonlinear 모델 |
| N-HiTS, AAAI 2023 | hierarchical interpolation | long horizon 효율성 개선 |
| Chronos, 2024 | tokenized probabilistic pretrained forecasting | 새 데이터에 zero-shot 적용 가능 |
| TimesFM, ICML 2024 | patched decoder foundation model | ARIMA/ETS와 같은 고전 baseline을 zero-shot에서 비교 |
| TimeGPT benchmark, 2026 | SARIMAX/Prophet/XGBoost와 실제 rolling 비교 | 충분한 feature engineering을 한 SARIMAX가 foundation model과 대등할 수 있음 |

### 최신 연구 결과를 읽을 때의 주의점

논문에서 “SOTA”라고 보고된 수치는 특정 benchmark, split, horizon, preprocessing, tuning budget에 종속됩니다. 특히 foundation model은 pretraining corpus가 benchmark와 얼마나 겹치는지, zero-shot인지 fine-tuned인지에 따라 비교 조건이 달라집니다. 따라서 **서로 다른 논문의 숫자를 한 표에 놓고 단순 순위화하지 않습니다.**

## 9. 실제 파이프라인 적용 방향

1. naive/seasonal-naive를 먼저 계산합니다.  
2. train에서 변환·차분 차수를 결정하되 validation으로 확인합니다.  
3. AICc/BIC는 후보 축소에 사용하고 최종 선택은 rolling-origin error로 확인합니다.  
4. residual ACF/Ljung–Box와 분산 안정성을 확인합니다.  
5. untouched test에서 horizon별 MAE/RMSE/MASE를 보고합니다.

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
- **N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting** — Oreshkin et al., ICLR 2020. https://arxiv.org/abs/1905.10437
- **NHITS: Neural Hierarchical Interpolation for Time Series Forecasting** — Challu et al., AAAI 2023. https://doi.org/10.1609/aaai.v37i6.25854
- **Are Transformers Effective for Time Series Forecasting?** — Zeng, Chen, Zhang, Xu, AAAI 2023. https://doi.org/10.1609/aaai.v37i9.26317
- **Chronos: Learning the Language of Time Series** — Ansari et al., Transactions on Machine Learning Research, 2024. https://arxiv.org/abs/2403.07815
- **A decoder-only foundation model for time-series forecasting** — Das, Kong, Sen, Zhou, ICML 2024. https://research.google/pubs/a-decoder-only-foundation-model-for-time-series-forecasting/
- **Benchmarking a time-series foundation model (TimeGPT) for real-world forecasting applications** — Machine Learning with Applications, 2026. https://doi.org/10.1016/j.mlwa.2025.100801

---

### 학습 체크포인트

이 절을 공부한 뒤에는 **정의 → 수식 → 가정 → 추정 대상 → 평가 설계 → 누수 가능성 → 최신 연구와의 차이**를 자신의 말로 설명할 수 있어야 합니다. 코드를 외우는 것보다 “어떤 조건에서 이 방법이 맞고, 어떤 조건에서 실패하는가”를 설명하는 능력이 연구 단계의 목표입니다.
