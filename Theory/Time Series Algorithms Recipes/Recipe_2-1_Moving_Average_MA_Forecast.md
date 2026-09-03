# Recipe 2-1. Moving Average (MA) Forecast

> 학습 수준: **대학/대학원 → 연구자 발전형**  
> 원문 위치: **PDF 48–52 / book pp.34–38**  
> 기준 문헌: *Time Series Algorithms Recipes* (Apress, 2023)

## 1. Executive Summary — 10문장 이내

1. 이 절은 비교적 적은 파라미터로 시간 의존성을 설명하는 통계적 forecasting 원리를 학습하는 부분입니다.
2. ARIMA·ETS 계열의 강점은 해석 가능성, 작은 데이터에서의 안정성, 빠른 추정입니다.
3. 정상성·차분·lag 선택과 residual diagnostics를 올바르게 이해하는 것이 알고리즘 이름을 외우는 것보다 중요합니다.
4. 단일 test 구간에 맞춰 하이퍼파라미터를 선택하면 일반화 성능을 과대평가할 수 있습니다.
5. 최신 deep/foundation model과 비교할 때도 고전 모델은 반드시 유지해야 할 강한 baseline입니다.
6. 본 학습자료는 Recipe 2-1 “Moving Average (MA) Forecast”의 원문을 출발점으로 하되, 대학원 수준의 수학적 정의와 2020년 이후 연구를 연결합니다.

## 2. 목적과 필요성

이 Recipe를 공부하는 목적은 Python API를 외우는 것이 아니라, 해당 절이 가정하는 데이터 생성 구조와 평가 조건을 이해하는 것입니다. 시계열 연구에서는 동일한 알고리즘이라도 시간축 처리, validation 방식, feature availability에 따라 성능이 크게 달라집니다. 특히 연구자로 발전하려면 “코드가 실행된다”와 “통계적으로 타당한 예측 실험이다”를 구분해야 합니다.

> **용어 설명 — stationarity**  
> 확률분포의 핵심 특성이 시간 이동에 따라 변하지 않는 성질입니다.

> **용어 설명 — lag**  
> 현재보다 몇 시점 이전의 관측값을 뜻합니다.

> **용어 설명 — AIC/BIC**  
> likelihood 적합도와 모델 복잡도를 함께 평가하는 정보기준입니다.

## 3. 이론 강의


### 3.1 이동평균의 두 가지 의미를 구분해야 합니다

책의 Recipe 2-1은 rolling mean을 사용합니다.

$$
M_t=\frac{1}{w}\sum_{j=0}^{w-1}Y_{t-j}
$$

여기서 $w$는 window length입니다. 이는 smoothing 또는 단순한 예측 규칙으로 사용할 수 있습니다. 그러나 시계열 이론의 **MA($q$) 확률모형**은 전혀 다른 개념입니다.

$$
Y_t=\mu+\varepsilon_t+\theta_1\varepsilon_{t-1}+\cdots+\theta_q\varepsilon_{t-q}
$$

> **용어 설명 — rolling mean vs. MA(q)**  
> rolling mean은 관측값 자체를 평균내는 필터이고, MA($q$) 모델은 과거의 예측 오차(innovation)를 선형 결합하는 확률모형입니다. 이름이 같아 초보자가 가장 자주 혼동하는 부분입니다.

rolling mean의 분산 감소 효과는 독립 잡음 가정에서 대략 $\text{Var}(M_t)=\sigma^2/w$이지만, 시계열에 자기상관이 있으면 이 식은 달라집니다. 또한 centered moving average를 전체 데이터에 적용하면 미래 관측을 사용하는 누수가 발생할 수 있으므로 forecasting에서는 trailing window를 사용해야 합니다.

- 시계열 예측(Forecasting) 모델을 구축할 때 중심 이동 평균(Centered Moving Average)을 사용하면 미래 데이터가 과거로 유출되는 데이터 누수(Data Leakage)가 발생합니다. 따라서 예측 시점 이전의 데이터만을 활용하는 후방 이동 평균(Trailing/Causal Moving Average)을 반드시 사용해야 합니다.
  - 후방 이동 평균(Trailing MA): $\(t\)$ 시점의 평균을 구하기 위해 $\(t-2\), \(t-1\), \(t\)$ 시점 등 과거와 현재의 데이터만 사용합니다. 실전 예측(Out-of-sample forecasting)에 바로 적용할 수 있는 올바른 방법입니다.


### 수식 기호 상세 해설

- $M_t$: 시점 $t$에서의 rolling moving average입니다.
- $w$: 평균에 포함하는 window 길이, $j$: window 안의 상대적 lag 인덱스입니다.
- $Y_{t-j}$: 현재부터 $j$시점 과거의 관측값입니다.
- 두 번째 식의 $\mu$: 과정의 평균 수준, $\varepsilon_t$: 현재 innovation, $\theta_j$: $j$번째 과거 innovation의 계수입니다.
- $q$: 확률모형 MA($q$)에서 사용하는 과거 innovation의 최대 lag입니다.
- 따라서 첫 번째 moving average는 **평활 연산**, 두 번째 MA($q$)는 **확률모형**으로 서로 다른 개념입니다.

> - MA(q) 모델의 이노베이션(Innovation)이란 현재 시점의 시계열 데이터를 설명하기 위해 모델에 투입되는 예측 불가능한 백색 잡음(white noise)이나 충격(shock) 오차항을 뜻합니다.

### 3.3 무엇을 학습하거나 추정하는가

이 절의 핵심 추정 대상은 데이터의 시간 구조를 설명하는 **파라미터 또는 상태**입니다. 통계모델이라면 계수와 오차분산을 추정하고, tree/deep model이라면 예측손실을 최소화하는 함수 $f_\theta$를 학습합니다. 공통적인 연구 목표는 새로운 미래 구간에서의 기대 손실

$$
R_{\text{future}}(f)=\text{E}\left[L(Y_{t+h},f(\mathcal{I}_t))\right]
$$

을 낮추는 것입니다. 예측 위험식의 기호는 다음과 같습니다. $R_{\text{future}}(f)$는 아직 보지 못한 미래 분포에서의 기대 예측손실, $\text{E}[\cdot]$는 그 분포에 대한 기대값, $Y\_{t+h}$는 $h$단계 미래의 실제값, $f$는 예측함수, $\mathcal{I}\_t$는 시점 $t$까지 이용 가능한 정보, $L(\cdot,\cdot)$은 실제값과 예측값의 오차를 수치화하는 손실함수입니다. $f=f_\theta$로 쓸 때 $\theta$는 데이터에서 학습하는 모델 파라미터 전체를 뜻합니다.

> **용어 설명 — information set $\mathcal{I}_t$**  
> 예측 순간에 현실적으로 알고 있는 모든 정보의 집합입니다. 미래 target이나 미래에만 측정되는 sensor 값을 포함하면 데이터 누수입니다.

## 4. 원문이 직접 보고한 내용

저자는 US GDP에 5기간 rolling mean을 적용해 원자료와 평활 시계열을 비교합니다. 정량적 holdout forecast 평가는 제시하지 않습니다.

이 문단은 **저자의 보고를 요약한 것**이며, 아래의 비판적 해석과 구분해야 합니다.

- 정량적 홀드아웃 예측 평가(Quantitative Holdout Forecast Evaluation)는 과거 데이터의 일부를 예측 모델 학습에서 제외(Holdout)한 후, 해당 기간의 실제 데이터와 모델의 예측치를 비교하여 모델의 객관적인 성과를 정량적으로 측정하는 기법입니다.

## 5. 비판적 해석 및 연구자 관점

rolling mean은 강력한 baseline/smoother지만 MA(q)와 구분해야 합니다. 작은 데이터에서 복잡한 모델보다 안정적일 수 있습니다.

또한 이 절의 결과를 다른 Recipe와 비교할 때는 데이터셋, target scale, forecast horizon, split, metric이 동일한지 확인해야 합니다. 조건이 다르면 숫자의 크기만으로 모델 우열을 말할 수 없습니다.

## 6. 통계적으로 취약한 부분과 비교 불가능한 수치

이동평균 smoothing과 MA(q) 확률모형이 명시적으로 구분되지 않아 개념 혼동 위험이 있습니다. holdout 성능이 없어 forecasting 성능을 비교할 수 없습니다.

**비교 가능성 판정:** 이 Recipe의 숫자는 동일한 데이터·동일한 forecast origin·동일한 horizon·동일한 metric을 사용한 실험끼리만 직접 비교해야 합니다. 다른 target이나 단위가 다른 RMSE는 직접 크기 비교를 하지 않습니다.

## 7. 문서가 답하지 않는 질문과 답변

**질문 1. 정상성이 반드시 필요한가요?**  
ARMA의 고전 이론에는 중요하지만 ARIMA는 차분을 통해 비정상성을 다룹니다. 예측 목적에서는 모델 residual과 OOS 성능을 함께 봐야 합니다.

- 고전 ARMA는 데이터 자체의 정상성을 요구하는 반면, ARIMA는 차분(Differencing)을 통해 비정상 데이터(단위근을 가진 데이터)를 처리합니다.
- 예측 목적의 머신러닝/딥러닝 모델 역시 원본 데이터의 정상성보다는 잔차(Residual)의 정상성과 아웃오브샘플(OOS, Out-of-Sample) 성능을 중심으로 평가하는 것이 실무적으로 옳습니다.

**질문 2. ACF/PACF만으로 차수를 정해도 되나요?**  
초기 후보에는 유용하지만 noisy finite sample에서는 불확실합니다. 정보기준과 rolling-origin을 함께 사용해야 합니다.

- ACF와 PACF는 이론적 분포를 가정한 초기 후보 식별에는 유용하지만, 유한하고 노이즈가 많은 실제 데이터(noisy finite sample)에서는 샘플 변동성이 커서 차수를 명확히 잘라내기(cut-off) 어렵기 때문입니다.

**질문 3. ARIMA가 최신 모델보다 낡아서 쓸 필요가 없나요?**  
그렇지 않습니다. 작은 데이터·강한 seasonality·해석성·빠른 재학습이 중요하면 여전히 매우 강한 baseline입니다.

- 압도적인 연산 속도: 딥러닝 모델(예: Informer, PatchTST 등)이 학습에 수 시간에서 수일이 걸릴 때, ARIMA는 수초 내에 학습과 예측을 끝냅니다.
- 명확한 해석 가능성: 결과의 원인을 파악하기 힘든 '블랙박스' 딥러닝과 달리, ARIMA는 과거의 내 값(AR)과 과거의 오차(MA) 중 어느 것이 예측에 영향을 주었는지 수학적으로 완벽히 설명할 수 있습니다.
- 적은 데이터로도 작동: 최신 대형 모델은 수만 개 이상의 데이터 포인트가 있어야 과적합(Overfitting)을 피할 수 있지만, ARIMA는 수십~수백 개의 데이터만으로도 안정적인 예측값을 냅니다.
- 강력한 기준점(Baseline): 새로운 복잡한 모델을 도입했을 때, 그 모델이 정말 우수한지 평가하려면 ARIMA보다 성능이 좋은지 확인하는 것이 업계 표준입니다.

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

> Seasonal Naive (계절성 나이브) 모델 : 가장 최근의 '동일한 계절' 또는 '동일한 주기'의 관측치를 미래 예측값으로 사용하는 방법입니다.

> AICc / BIC를 통한 후보 축소 (Filter Step): 수많은 파라미터 조합(예: ARIMA의 p, d, q 차수 또는 변수 조합) 중 연산 비용을 줄이고 과적합을 방지하기 위해 상위 후보군을 빠르게 선별합니다. 

> 잔차의 자기상관 확인 (ACF / Ljung-Box Test) : 잔차에 정보가 남아있지 않고 무작위적인지(백색잡음) 확인합니다.
>   - 잔차 ACF (Autocorrelation Function) 플롯 : 모든 시차(Lag)에서 ACF 값이 신뢰구간(보통 파란 점선, $\(\pm 1.96/\sqrt{T}\))$ 안에 들어와야 합니다. 신뢰구간을 벗어나는 시차가 없다면 자기상관이 없는 백색잡음으로 봅니다.
>   - 융-박스 검정 (Ljung-Box Test) - 가설: $\(H_{0}\)$ (잔차들이 독립적이다 / 자기상관이 없다) vs $\(H_{1}\)$ (잔차들이 독립적이지 않다). p-value > 0.05이면 귀무가설을 채택하여 잔차에 자기상관이 없는 백색잡음 상태로 판단합니다.

### 일반화 성능을 높이기 위한 연구 방향

- 모델 복잡도를 먼저 키우기보다 **시간순 교차검증과 leakage 제거**로 평가분산을 줄입니다.
- 예측 horizon별로 필요한 구조가 다르므로 $h=1$과 long-horizon을 분리하여 튜닝합니다.
- 여러 seed 또는 여러 rolling origin에서 평균과 표준편차를 보고 selection noise를 줄입니다.
- 작은 데이터에서는 low-capacity statistical/tree baseline을 유지하고, deep/foundation model이 실제로 유의한 개선을 주는지 검증합니다.
- domain shift가 예상되면 최근 구간 가중, calibration, covariate shift 진단, fine-tuning을 별도 실험합니다.

> 최근 구간 가중 (Recency Weighting) : 데이터가 시간에 따라 점진적으로 변하는 Concept Drift나 시간 경과에 따른 가중치 최적화를 위함입니다.

> Calibration (예측 확률 보정): 도메인이 변화하면 모델이 출력하는 확률값(Confidence)의 신뢰도가 떨어져 과잉 확신(Overconfidence) 또는 과소 확신이 발생합니다. 배포 전 단계에서 Platt Scaling이나 Isotonic Regression을 활용하여 예측 확률을 실제 정확도와 일치시킵니다.

> Covariate Shift 진단: 학습 데이터 $(\(P(X)\))$ 와 최근 데이터 $(\(P'(X)\))$ 의 입력 분포 차이를 정량적으로 확인합니다.
>   - Adversarial Validation(적대적 검증): 데이터가 원래 학습 데이터인지 최근 데이터인지 분류하는 이진 분류기를 학습시킵니다. 이때 AUC가 0.5에 가깝다면 차이가 없는 것이고, 1.0에 가깝다면 심각한 Covariate Shift가 발생한 것입니다.
>   - 주요 피처별로 PSI(Population Stability Index)나 KL-Divergence를 계산하여 어떤 피처가 변했는지 추적합니다.

> Fine-tuning (미세 조정): 변화한 도메인의 패턴을 모델에 직접 학습시켜 적응(Domain Adaptation)시킵니다.
>   - 기존 지식을 잊어버리는 파괴적 망각(Catastrophic Forgetting)을 방지하기 위해 학습률(Learning Rate)을 매우 낮게 설정합니다.
>   - 앞단의 특징 추출 레이어는 동결(Freeze)하고, 출력층(Classification Head) 위주로 먼저 학습하는 방안을 비교 실험합니다.

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
