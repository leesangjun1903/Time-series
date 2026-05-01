# Anomaly detection algorithm of single variable time-series data based on dynamic parametrization for subsurface fluid data anomaly detection

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

본 논문은 지하 유체(지하수위) 시계열 데이터의 이상 탐지를 위해 **동적 파라미터 조정 기반 단일 변수 시계열 이상 탐지 알고리즘(ADSV-DPT: Anomaly Detection algorithm of Single Variable time-series Data based on Dynamic Parameter Tuning)**을 제안합니다.

기존 머신러닝 기반 방법들(LSTM, GRU, OC-SVM 등)의 높은 계산 비용, 복잡한 파라미터 조정 필요성, 비정상적(non-stationary) 데이터에 대한 취약성 문제를 **경량화된 통계 기반 적응형 알고리즘**으로 해결할 수 있다는 것이 핵심 주장입니다.

### 주요 기여

| 기여 항목 | 설명 |
|-----------|------|
| **강건한 중심/분산 추정** | 극값의 영향을 최소화하기 위해 평균 대신 Median + MAD 사용 |
| **동적 파라미터 업데이트** | 슬라이딩 윈도우 기반 실시간 파라미터 갱신 |
| **동적 임계값 설정** | 데이터 분포 변화에 자동 적응하는 임계값 계산 |
| **실용적 성능** | 평균 Precision, Recall, F1-score 모두 85% 이상 달성 |
| **낮은 계산 비용** | 대규모 실시간 모니터링 시스템에 적합한 경량 알고리즘 |

---

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

중국 지진청(CEA)의 지하 유체 관측망은 분당 1회, 하루 1,440개의 고빈도 수위 데이터를 수집합니다. 이 방대한 데이터에서 세 가지 주요 이상 유형을 자동으로 탐지하는 것이 목표입니다:

- **Jump anomaly**: 단일 또는 다중 지점의 급격한 상승/하강 (지진파, 강수, 기압 변화 등에 의해 발생)
- **Step anomaly**: 짧은 시간 내 급격하고 지속적인 수준 변화 (강수, 배수로 막힘, 정전 등)
- **Steep anomaly**: 선형적 급격한 상승 또는 하강 (기상 현상, 기기 고장 등)

**기존 방법들의 한계:**

| 방법 유형 | 한계 |
|-----------|------|
| 전통 통계 (ARIMA, 웨이블릿) | 수동 파라미터 튜닝 필요, 자동화 수준 낮음 |
| 딥러닝 (LSTM, GRU) | 높은 훈련 비용, 복잡한 구조, 대규모 라벨링 데이터 필요 |
| SPOT/DSPOT | MLE 계산 비용 높고, 비정상 데이터에서 불안정 |
| KNN, OC-SVM | 복잡한 혼합 이상 패턴에 취약 |

---

### 2.2 제안 방법 (ADSV-DPT) 및 수식

#### (1) 중앙값(Median) 계산

$$\text{median} = \text{median}(\text{data window}) $$

#### (2) 중앙 절대 편차(MAD) 계산

$$\text{MAD} = \text{median}(|\text{data window} - \text{median}|) $$

> MAD는 평균 대신 중앙값을 기준으로 하므로, 극단값(outlier)이 분산 추정에 미치는 영향을 크게 줄입니다.

#### (3) 현재 윈도우의 표준편차 계산 (윈도우 크기 조정용)

$$\sigma = \text{std}(\text{data window}) $$

#### (4) 윈도우 크기 동적 조정

$$\text{new size} = \max(\text{min window size},\ \text{window size} - 1) \quad \text{if } \sigma > 5 $$

$$\text{new size} = \min(\text{max window size},\ \text{window size} + 1) \quad \text{if } \sigma < 1 $$

$$\text{window size} = \text{new size} $$

> 데이터 변동성이 클 때( $\sigma > 5$ )는 윈도우를 축소하여 반응성 향상, 변동성이 작을 때( $\sigma < 1$ )는 윈도우를 확대하여 안정성 확보

#### (5) 동적 임계값 설정

$$\text{threshold high} = \text{median} + \text{sensitivity factor} \times \text{MAD} $$

$$\text{threshold low} = \text{median} - \text{sensitivity factor} \times \text{MAD} $$

#### (6) 이상 판별 조건

$$\text{Anomaly} \iff \text{data point} > \text{threshold high} \quad \text{or} \quad \text{data point} < \text{threshold low} $$

#### (7) 평가 지표

```math
\begin{cases} \text{Accuracy} = \dfrac{TP + TN}{TP + FP + TN + FN} \\[10pt] \text{Precision} = \dfrac{TP}{TP + FP} \\[10pt] \text{Recall} = \dfrac{TP}{TP + FN} \\[10pt] F_1\text{-score} = \dfrac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} \end{cases}
```

---

### 2.3 모델 구조

알고리즘의 전체 처리 흐름은 다음과 같습니다:

```
[데이터 입력]
     ↓
[파라미터 초기화]
: initial_window_size=10, min=10, max=1000, sensitivity_factor=5~10
     ↓
[데이터 포인트 추가 → 윈도우에 누적]
     ↓
[σ 계산 → 윈도우 크기 동적 조정]
  σ > 5  →  window_size - 1
  σ < 1  →  window_size + 1
     ↓
[median, MAD 업데이트]
     ↓
[동적 임계값 계산]
  threshold_high = median + sensitivity_factor × MAD
  threshold_low  = median - sensitivity_factor × MAD
     ↓
[이상 판별]
  data_point > threshold_high OR data_point < threshold_low
  → True(이상) / False(정상)
     ↓
[다음 데이터 포인트로 반복]
```

**핵심 파라미터 설정:**

| 파라미터 | 값 |
|----------|-----|
| initial_window_size | 10 |
| min_window_size | 10 |
| max_window_size | 1000 |
| sensitivity_factor | 5~10 |

---

### 2.4 성능 결과

#### Jump 이상 탐지

| 알고리즘 | Accuracy | Precision | Recall | F1-score |
|----------|----------|-----------|--------|----------|
| **ADSV-DPT** | **0.932** | **0.975** | **0.955** | **0.965** |
| PELT | 0.754 | 0.816 | 0.908 | 0.860 |
| KNN | 0.775 | 0.826 | 0.926 | 0.873 |
| OC-SVM | 0.840 | 0.936 | 0.891 | 0.913 |

#### Step 이상 탐지

| 알고리즘 | Accuracy | Precision | Recall | F1-score |
|----------|----------|-----------|--------|----------|
| **ADSV-DPT** | **0.861** | **0.881** | **0.934** | **0.907** |
| PELT | 0.772 | 0.814 | 0.883 | 0.847 |
| KNN | 0.740 | 0.772 | 0.915 | 0.838 |
| OC-SVM | 0.703 | 0.760 | 0.867 | 0.810 |

#### Steep 이상 탐지

| 알고리즘 | Accuracy | Precision | Recall | F1-score |
|----------|----------|-----------|--------|----------|
| **ADSV-DPT** | **0.999** | **0.972** | **0.938** | **0.955** |
| PELT | 0.983 | 0.536 | 0.281 | 0.369 |
| KNN | 0.985 | 0.441 | 0.516 | 0.476 |
| OC-SVM | 0.955 | 0.239 | 0.768 | 0.365 |

#### 혼합(Mixed) 이상 탐지

| 알고리즘 | Accuracy | Precision | Recall | F1-score |
|----------|----------|-----------|--------|----------|
| **ADSV-DPT** | **0.924** | **0.954** | **0.947** | **0.950** |
| PELT | 0.741 | 0.855 | 0.799 | 0.826 |
| KNN | 0.773 | 0.514 | 0.301 | 0.380 |
| OC-SVM | 0.538 | 0.811 | 0.521 | 0.635 |

> **특히 Steep 이상 탐지에서 PELT(F1: 0.369), KNN(F1: 0.476), OC-SVM(F1: 0.365) 대비 ADSV-DPT(F1: 0.955)가 압도적 우위**

---

### 2.5 한계점

논문에서 명시적 또는 암묵적으로 인정된 한계:

1. **단일 변수(univariate) 전용**: 다변수 시계열(예: 수위 + 수온 + 유량 동시 분석)에 직접 적용 불가
2. **Step 이상에서의 오인식 존재**: 정상 데이터의 자연적인 변동과 step 이상의 경계가 모호한 경우 일부 오류 발생 (논문 Discussion 참고)
3. **임계값 $\sigma$ 기준의 임의성**: $\sigma > 5$ 또는 $\sigma < 1$ 의 기준값이 경험적으로 설정되어 있어 다른 도메인 적용 시 재조정 필요
4. **레이블 불균형 문제**: 실험 데이터가 타겟 샘플링으로 구성되어 실제 관측 환경(심각한 불균형)을 완전히 반영하지 못할 가능성
5. **소규모 실험 데이터셋**: 총 180개 인스턴스(카테고리당 60개)로 일반화 검증에 한계
6. **지진 전조와의 직접적 연관성 미검증**: 이상 탐지 레이블링 자동화에 집중하며, 실제 지진 예측 성능은 별도 검증 필요

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재 일반화 성능의 강점

ADSV-DPT는 다음과 같은 설계적 특성으로 인해 **태생적으로 일반화 능력이 높습니다:**

#### (a) 분포 가정 없는(Distribution-free) 설계

Median과 MAD는 비모수적(non-parametric) 통계량으로, 데이터가 정규 분포를 따른다는 가정이 없습니다. 이는 다양한 유형의 지하수위 데이터(절대값이 크게 다른 여러 관측소)에도 적용 가능합니다. 실제로 논문은 절대값이 아닌 **상대적 변화**에 초점을 맞추어 이 강점을 명시합니다.

#### (b) 로컬 적응성(Local Adaptivity)

슬라이딩 윈도우 메커니즘이 전역 분포 가정 없이 **로컬 데이터 특성**에 적응하기 때문에, 비정상(non-stationary) 시계열에 강건합니다:

$$\text{median}\_{t} = \text{median}(\{x_{t-w+1}, \ldots, x_t\})$$

$$\text{MAD}_{t} = \text{median}(|x_i - \text{median}_{t}|), \quad i \in [t-w+1, t]$$

이 로컬 추정값은 계절성 변동, 조수 영향, 장기 트렌드 등 다양한 비정상 패턴에서도 안정적으로 작동합니다.

#### (c) 자기 조정 윈도우(Self-adjusting Window)

표준편차 $\sigma$에 기반한 윈도우 크기 동적 조정은 알고리즘이 데이터 특성에 따라 자동으로 적응 범위를 조절합니다:

- 고변동성 구간: 윈도우 축소 → 빠른 반응
- 저변동성 구간: 윈도우 확대 → 안정적 기준선 확보

#### (d) 혼합 이상 탐지 성능

다양한 이상 유형이 혼합된 환경에서 F1-score 0.950을 달성한 것은 알고리즘의 **교차 도메인 일반화 가능성**을 시사합니다.

---

### 3.2 일반화 성능 향상을 위한 잠재적 방향

논문이 명시적으로 언급하거나 분석을 통해 도출 가능한 향후 개선 방향:

#### (a) 다변수 확장

현재 단일 변수 전용이지만, 다음과 같이 확장 가능:

$$\text{MAD}_{multi} = \text{median}\left(\left|\mathbf{x}_i - \text{median}(\mathbf{X})\right|_2\right)$$

여러 센서 데이터를 동시에 처리하는 다변수 버전을 개발하면 지하수위 + 수온 + 유량 등의 복합 이상을 탐지 가능합니다.

#### (b) 적응형 sensitivity_factor

현재 sensitivity_factor는 수동 설정(5~10)입니다. 이를 다음과 같이 자동화하면 일반화 성능 향상 가능:

$$\text{sensitivity factor}_t = f\left(\frac{\text{MAD}_t}{\text{MAD}_{global}}\right)$$

또는 베이지안 최적화(Bayesian Optimization)를 통해 관측소별 최적값 자동 도출 가능합니다.

#### (c) 머신러닝과의 하이브리드 통합

논문 결론에서 직접 언급: *"investigating the integration of statistical methods with machine learning techniques to identify a broader spectrum of anomaly patterns"*

예를 들어, ADSV-DPT를 전처리 필터로, LSTM을 후처리 분류기로 결합:

$$P(\text{anomaly} | x_t) = \alpha \cdot P_{ADSV-DPT}(x_t) + (1-\alpha) \cdot P_{LSTM}(x_t)$$

#### (d) 도메인 전이 학습 가능성

ADSV-DPT는 훈련 데이터 없이 동작하는 **비지도 방식**이므로, 새로운 관측소에 즉시 적용 가능합니다. 다만, sensitivity_factor와 윈도우 크기의 초기 설정을 메타러닝(meta-learning)으로 자동화하면 zero-shot 일반화가 가능합니다.

#### (e) 불균형 데이터 대응

실제 환경에서 이상 데이터는 전체의 0.1% 미만일 수 있습니다. 다음과 같은 접근으로 일반화 성능 향상 가능:
- SMOTE 기반 오버샘플링과 결합
- 적응형 임계값을 이상 탐지 비율에 따라 보정

---

## 4. 향후 연구에 미치는 영향과 고려 사항

### 4.1 연구에 미치는 영향

#### (a) 지진 전조 연구에 대한 기여

지하수위 이상 자동 레이블링 시스템의 실용적 토대를 제공합니다. 향후 대규모 관측 네트워크에서 수집된 데이터를 자동으로 이상 분류함으로써:
- 지진 전조 패턴 데이터베이스 구축 가능
- 머신러닝 기반 지진 예측 모델의 훈련 데이터 품질 향상

#### (b) 경량 이상 탐지 알고리즘 연구 방향 제시

딥러닝 의존도를 낮추고 **해석 가능성(interpretability)**이 높은 통계 기반 방법의 실용성을 재확인했습니다. IoT 환경, 엣지 컴퓨팅 적용 가능성을 보여줍니다.

#### (c) 비정상 시계열 처리의 표준 접근법

동적 윈도우 + MAD 조합은 다른 비정상 환경 모니터링 도메인(대기 오염, 하천 수위, 전력 소비 등)에도 적용 가능한 **일반화된 프레임워크**를 제시합니다.

---

### 4.2 향후 연구 시 고려할 점

#### (a) 데이터셋 규모 및 다양성 확장

- **현재 한계**: 180개 인스턴스(관측소당 60개)는 통계적 일반화에 충분하지 않음
- **권고사항**: 중국 전역 수백 개 관측소의 다년간 데이터로 실험 확장 필요
- **다양한 지질 조건**(단층 근접, 지하수 심도 등)에 따른 성능 변화 검증 필요

#### (b) 윈도우 조정 임계값의 이론적 정당화

현재 $\sigma > 5$ 및 $\sigma < 1$ 기준은 경험적 설정입니다. 향후 연구에서:
- 통계적 검증(예: likelihood ratio test) 기반 임계값 도출
- 도메인별 최적 임계값 민감도 분석(sensitivity analysis) 필요

#### (c) 실시간 처리 시스템 구현 검증

논문은 알고리즘의 실시간 적합성을 주장하지만, 실제 스트리밍 환경에서의 **지연 시간(latency)**, **처리량(throughput)**, **메모리 사용량** 측정이 부재합니다.

#### (d) 레이블 노이즈 대응

수동 레이블링 데이터를 Ground Truth로 사용하고 있으나, 인간 전문가의 판단에도 오류가 포함될 수 있습니다. **노이즈 레이블 학습(learning with noisy labels)** 기법과의 결합을 고려해야 합니다.

#### (e) 지진과의 인과관계 검증

이상 탐지 알고리즘의 성능이 높더라도, 탐지된 이상이 실제 지진 전조인지에 대한 **역학적(causal) 검증**이 필요합니다. 구체적으로:

$$P(\text{earthquake} | \text{anomaly detected}) \gg P(\text{earthquake})$$

임을 통계적으로 입증해야 합니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

논문에서 직접 인용된 2020년 이후 연구들을 중심으로 비교합니다.

### 5.1 논문 내 인용된 2020년 이후 주요 연구

| 연구 | 방법 | 적용 도메인 | ADSV-DPT 대비 |
|------|------|-------------|---------------|
| Yan et al. (2021) *J. Hydrol.* | LSTM + EWMA control chart | 지하수위 (1996 여강 지진) | 더 높은 정확도 가능하나 훈련 비용 매우 높음 |
| Liu et al. (2021) *Prog. Geophys.* | GRU | 가스 압력 이상 (원촨 지진 전조) | 비선형 패턴에 강하나 레이블 데이터 필요 |
| Song et al. (2022) *IEEE Trans.* | Improved Isolation Forest | 초분광 이상 탐지 | 고차원 데이터에 적합하나 단일 변수에 과사양 |
| Yerima & Bashar (2022) *IWSSIP* | Semi-supervised OC-SVM | 스팸 탐지 | 경계면 기반으로 복잡 혼합 이상에 취약 |
| Truong et al. (2020) *Signal Process.* | PELT (오프라인 변화점 탐지) | 일반 시계열 | 실시간 처리에 제한적 |
| Song et al. (2024) *Geol. J. Earth* | GPR-LSTM | 지열 적외선 배경장 | 복잡한 모델, 고계산 비용 |
| Cheng et al. (2025) *IEEE/CAA* | Adaptive Kalman Filtering (MLE) | 일반 제어 시스템 | MLE 계산 비용 높음 |

### 5.2 ADSV-DPT의 차별화 포인트 정리

```
고정밀 탐지
    ↑
ADSV-DPT ●  ← 이 논문의 위치
    |           (정확도 + 경량성 균형)
OC-SVM  ●
    |
PELT    ●
    |
KNN     ●
    ↓
저정밀 탐지
─────────────────────────────→
저계산비용              고계산비용
              LSTM/GRU ●
                  GPR-LSTM ●
```

### 5.3 ADSV-DPT가 아직 다루지 못하는 최신 연구 흐름

아래는 논문 범위를 벗어나지만 향후 비교가 필요한 연구 방향입니다 (단, 이 부분은 논문 내 직접 인용이 없으므로, 해당 영역의 일반적 연구 동향을 서술하며 **특정 논문을 단정적으로 특정하지 않습니다**):

1. **Transformer 기반 시계열 이상 탐지**: Self-attention 메커니즘을 활용한 장기 의존성 포착 (ADSV-DPT는 로컬 윈도우만 활용)
2. **대조학습(Contrastive Learning) 기반 비지도 이상 탐지**: 레이블 없이 정상 패턴 표현 학습
3. **그래프 신경망(GNN) 기반 다변수 이상 탐지**: 여러 관측소 간 공간적 상관관계 활용

---

## 참고 자료 (출처)

본 답변은 아래의 단일 논문 원문을 직접 분석하여 작성하였습니다:

**주요 출처:**
- **Yang, Z., Wang, J., Huang, Q., Fan, L., Yang, X., Liu, G., Shuai, C. & Yang, F. (2025).** "Anomaly detection algorithm of single variable time-series data based on dynamic parametrization for subsurface fluid data anomaly detection." *Geophysical Journal International*, 243, 1–12. https://doi.org/10.1093/gji/ggaf328

**논문 내 인용 문헌 (비교 분석에 활용):**
- Yan et al. (2021). *Journal of Hydrology*, 599, 126369.
- Liu et al. (2021). *Progress in Geophysics*, 36(03), 901–907.
- Song et al. (2022). *IEEE Transactions on Geoscience and Remote Sensing*, 60, 1–16.
- Truong, C., Oudre, L. & Vayatis, N. (2020). *Signal Processing*, 167, 107299.
- Siffer et al. (2017). *Proceedings of the 23rd ACM SIGKDD*, 1067–1075.
- Yerima, S.Y. & Bashar, A. (2022). *IWSSIP 2022*, 1–4.
- Song et al. (2024). *Geological Journal of Earth*, 46(02), 492–511.
- Cheng et al. (2025). *IEEE/CAA Journal of Automatica Sinica*, 12(1), 228–254.

> ⚠️ **정확도 주의사항**: 2020년 이후 외부 최신 연구(Transformer, GNN 기반 등)와의 정량적 비교는 해당 논문에서 직접 수행되지 않았으므로, 구체적 수치 비교는 제시하지 않았습니다. 위 비교 표는 논문 내 인용된 연구에 한정하여 작성하였습니다.
