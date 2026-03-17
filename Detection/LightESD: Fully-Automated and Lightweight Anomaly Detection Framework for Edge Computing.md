# LightESD: Fully-Automated and Lightweight Anomaly Detection Framework for Edge Computing

## 종합 분석 보고서

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장
LightESD는 엣지 디바이스에서 직접 이상 탐지(anomaly detection)를 수행할 수 있는 **완전 자동화된(fully-automated), 경량(lightweight), 통계학습 기반(statistical learning-based)** 프레임워크이다. 딥러닝 기반 방법들이 중앙 서버와의 데이터 전송에 따른 네트워크 오버헤드, 지연, 에너지 소비 문제를 야기하는 반면, LightESD는 **on-device learning**을 실현하여 이러한 문제를 해소하면서도 경쟁력 있는 탐지 정확도를 달성한다.

### 주요 기여 (3가지)

| 기여 | 설명 |
|------|------|
| **① LightESD 프레임워크 제안** | Weight-free, non-parametric, unsupervised 방식으로, 어떤 단변량 시계열에도 수동 전처리나 하이퍼파라미터 튜닝 없이 자동 적응 |
| **② ADCompScore 메트릭 제안** | 탐지 성능뿐 아니라 연산 자원, 전력 소비까지 통합 평가하는 최초의 종합 메트릭 |
| **③ 종합 실험 평가** | 합성 데이터 및 실제 데이터셋(NAB, Yahoo)에서 SOTA 대비 우수한 종합 성능 입증 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

1. **네트워크 오버헤드 문제**: 딥러닝 모델은 일반적으로 엣지 디바이스에서 수집한 데이터를 중앙 서버로 전송하여 학습하므로, 대규모 네트워크 트래픽, 지연, 에너지 소비가 발생한다.
2. **Concept Drift 문제**: 중앙 서버에서 오프라인 학습 후 엣지에 배포하는 방식은 새로운 데이터 패턴에 적응하지 못하여 concept drift가 발생한다.
3. **수동 하이퍼파라미터 설정의 어려움**: 기존 대부분의 방법은 데이터셋마다 수동으로 모델 파라미터나 하이퍼파라미터를 설정해야 하며, 이는 실용성과 일반화를 저해한다.
4. **정규성 가정의 한계**: 기존 통계적 방법(ARIMA, ESD 등)은 원본 데이터가 가우시안 분포를 따른다는 가정에 의존하나, 실제 데이터는 이를 충족하지 못하는 경우가 많다.

### 2.2 제안하는 방법 (수식 포함)

LightESD는 세 단계로 구성된다: **(1) 주기성 탐지 → (2) 잔차 추출 → (3) 잔차 기반 이상 탐지**

#### 2.2.1 시계열 분해 모델

시계열 $Y_t$는 다음과 같이 가법적(additive)으로 분해된다:

$$Y_t = T_t + \sum_{i=1}^{k} S_t^i + R_t \tag{1}$$

여기서 $T_t$는 **추세(trend)**, $S_t^i$는 $i$번째 **계절성(seasonal)** 성분, $R_t$는 **잔차(residual)** 성분이며, 총 $k \geq 1$개의 계절 성분이 존재한다.

#### 2.2.2 Stage 1: 개선된 주기성 탐지 (Improved Periodicity Detection)

- **Welch's Periodogram Method** [21]을 사용하여 비모수적(non-parametric)으로 Power Spectral Density(PSD)를 추정
- 원본 시계열을 **100번 랜덤 순열(permutation)**하여 각각의 PSD를 계산하고, 99번째 백분위수를 **임계값(threshold)**으로 설정
- 원본 시계열의 PSD 중 이 임계값을 초과하는 **유의미한 피크(peak)**만을 주기로 탐지
- 기존 AutoPeriod [23]의 공간 복잡도 $O(N)$을 $O(1)$로 개선 (유의미한 피크만 저장)

#### 2.2.3 Stage 2: 잔차 추출 (Residuals Extraction)

**비계절적(nonseasonal) 시계열**의 경우, **RobustTrend** [26]를 사용하여 추세를 추출한다:

$$\arg\min_{\mathbf{t}} h_\gamma(\mathbf{y} - \mathbf{t}) + \lambda_1 \|\mathbf{D}_{(1)}\mathbf{t}\|_1 + \lambda_2 \|\mathbf{D}_{(2)}\mathbf{t}\|_1 \tag{2}$$

여기서:
- $h_\gamma(\cdot)$: **Huber Loss** (이상치에 강건한 손실 함수)
- $\mathbf{D}_{(1)}$: 1차 차분 행렬 (abrupt 변화 포착)
- $\mathbf{D}_{(2)}$: 2차 차분 행렬 (slow 변화 포착)
- $\lambda_1, \lambda_2$: 정규화 강도 제어 파라미터

이 최적화 문제는 **ADMM (Alternate Direction Method of Multipliers)** 기반의 Majorization-Minimization [28]으로 풀어 추세 $\mathbf{t}^* = T_t$를 추정하고, 잔차는 $R_t = Y_t - T_t$로 추출된다.

**계절적(seasonal) 시계열**의 경우, **FastRobust-STL** [27]을 사용하여 추세와 다중 계절성을 모두 추출한 후:

$$R_t = Y_t - T_t - \sum_{i=1}^{k} S_t^i$$

#### 2.2.4 Stage 3: 개선된 ESD 기반 이상 탐지

**기존 일반화 ESD 테스트** [33]의 검정 통계량:

$$R = \max_i \left( \frac{|Y_i - \mu|}{\sigma} \right), \quad i = 1, \ldots, n \tag{3}$$

**임계값(critical value)** $\lambda$:

$$\lambda = \frac{t_{n-l-2,\,p} \times (n - l - 1)}{\sqrt{(n - l) \times (t_{n-l-2,\,p}^2 + n - l - 2)}} \tag{4}$$

여기서 $p = 1 - \frac{\alpha}{2 \times (n-l)}$, $l$은 반복 인덱스($0$부터 $a_{max}-1$), $t_{(\cdot,\cdot)}$는 양측 t-분포 값이다.

**LightESD의 개선된 강건 검정 통계량**:

$$R_{\text{robust}} = \max_i \left( \frac{|Y_i - \text{median}(Y)|}{S(Y)} \right), \quad i = 1, \ldots, n \tag{5}$$

$$S(Y) = \text{median}_i \left( \text{median} |Y_i - Y| \right) \tag{6}$$

이 개선의 핵심 효과:
- **유한 표본 붕괴점(finite sample breakdown point)**이 $\frac{1}{n+1}$에서 $\frac{\lfloor n/2 \rfloor}{n}$으로 개선
- 점근적으로 **최대 50%**의 값이 임의로 큰 경우에도 견딜 수 있음
- MAD(Median Absolute Deviation) 대신 $S$ 통계량을 사용하여 **대칭 분포 가정이 불필요**하고, 가우시안 분포에서 더 효율적 [35]

#### 2.2.5 ADCompScore 메트릭

$$ADCS = \frac{1}{\sum w_*} \left[ w_f \cdot f + w_g \cdot (1-g) + w_l \cdot (1-l) + w_c \cdot (1-c) + w_r \cdot (1-r) + w_p \cdot (1-p) \right] \tag{9}$$

여기서:
- $f$: $F_1$-score, $g$: 변동계수(CV), $l$: min-max 정규화된 지연시간
- $c$: CPU 사용률(%), $r$: RAM 사용률(%), $p$: 전력 소비 증가율(%)
- $w_* = \{w_f, w_l, w_c, w_r, w_p, w_g\}$: 각 메트릭의 가중치

### 2.3 모델 구조

LightESD의 전체 파이프라인은 Algorithm 2로 요약된다:

```
1. 주기성 탐지 (Algorithm 1: Welch + 순열 기반 임계값)
2. IF 비계절적 → RobustTrend로 잔차 추출
   ELSE → FastRobust-STL로 잔차 추출  
3. a_max = 0.1 × len(Y)
4. 개선된 ESD 테스트 (α = 0.05 또는 0.001)
5. 경계점 보정 (첫/마지막 점의 고립 이상치 제거)
6. 이상치 인덱스 반환
```

**핵심 구조적 특성:**
- **Weight-free**: 가중치를 저장하지 않으므로 메모리 부담 최소화
- **Non-parametric**: 원본 데이터의 분포/함수 형태에 대한 가정 없음
- **Unsupervised**: 레이블 불필요, 온라인에서 자동 적응
- **시간 복잡도**: $O(N \log N)$ (FFT 기반)
- **공간 복잡도**: $O(N)$ (최악의 경우)

### 2.4 성능 향상

#### 탐지 성능 (Table II 기반)

| 모델 | STD F1 | RW F1 | NAB F1 | Yahoo F1 | 평균 F1 | CV |
|------|--------|-------|--------|----------|--------|-----|
| OC-SVM | 0.59 | 0.59 | 0.56 | 0.59 | 0.58 | 0.03 |
| LOF | 0.63 | 0.73 | 0.60 | 0.68 | 0.66 | 0.10 |
| Iso. Forest | 0.48 | 0.60 | 0.59 | 0.65 | 0.58 | 0.12 |
| ONLAD | 0.71 | 0.67 | 0.76 | 0.73 | 0.72 | 0.06 |
| BRVFL-AE | 0.71 | 0.78 | 0.79 | 0.76 | 0.76 | 0.05 |
| **LightESD-1** | **0.79** | **0.81** | **0.80** | **0.84** | **0.81** | **0.02** |
| **LightESD-2** | **0.83** | **0.96** | **0.84** | **0.86** | **0.87** | 0.06 |

#### 엣지 관련 성능 (Table III 기반)

| 모델 | Latency(s) | CPU(%) | RAM(%) | Power 증가(%) |
|------|-----------|--------|--------|-------------|
| BRVFL-AE | 0.21 | 10.96 | 5.33 | 17.7 |
| ONLAD | 0.19 | 8.87 | 3.11 | 16.3 |
| **LightESD** | **0.24** | **5.47** | **3.29** | **14.3** |

#### ADCompScore 종합 점수
- **LightESD-2**: 0.93 (최고)
- **LightESD-1**: 0.92
- BRVFL-AE: 0.89
- ONLAD: 0.87
- SOTA 대비 **3~6%** 향상, ML 기반 대비 **최대 33%** 향상

### 2.5 한계

1. **배치 학습 의존성**: LightESD는 패턴 학습을 위해 데이터 배치(batch)가 필요하며, 단일 학습 인스턴스만 있는 환경에서는 한계가 있다.
2. **지연시간**: 반복적 ESD 특성으로 인해 ONLAD(0.19s)나 BRVFL-AE(0.21s)보다 약간 높은 지연시간(0.24s)을 보인다.
3. **단변량 시계열 한정**: 현재 프레임워크는 univariate 시계열에만 초점을 맞추고 있으며, 다변량(multivariate) 시계열로의 확장은 논의되지 않았다.
4. **온라인 학습 미지원**: 순수 온라인 학습(streaming single instance) 환경으로의 전환이 향후 과제로 남아 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

LightESD의 일반화 성능은 본 논문의 가장 차별화된 기여 중 하나이며, 다음 메커니즘들에 의해 달성된다:

### 3.1 일반화를 가능하게 하는 설계 원리

#### (1) Non-parametric 특성
- 원본 데이터의 **분포나 함수 형태에 대한 가정을 하지 않음**
- 기존 ARIMA 등의 통계적 방법이 가우시안 가정에 의존하는 것과 대조적
- 다양한 분포를 가진 실제 데이터에 자동 적응 가능

#### (2) Weight-free 모델
- 학습된 가중치를 저장하지 않으므로, **새로운 데이터셋에 대해 재학습 없이 즉시 적응**
- SVM처럼 non-parametric이면서도 가중치(support vector coefficients)를 저장해야 하는 모델과 근본적으로 다름

#### (3) 잔차 기반 이상 탐지의 정당성
- 추세 제거된 시계열(trend-adjusted series)은 **정상(stationary)**이며, 경험적으로 **근사 가우시안 분포**를 따른다는 것이 입증됨 [12]
- 따라서 원본 데이터가 다중 모드(multi-modal)이거나 비가우시안이더라도, 잔차에 대한 ESD 테스트는 통계적으로 **정확(statistically correct)**하고 **신뢰성 있음**

#### (4) 자동 주기성 탐지
- 주기를 사전에 정의할 필요 없이(hourly/daily/weekly 등) **주파수 도메인에서 자동 탐지**
- 어떤 주기 패턴이든 포착 가능하여, 데이터셋 간 전환 시 추가 설정이 불필요

### 3.2 정량적 일반화 성능 평가

**변동계수(CV)** = 표준편차 / 평균 (F1-score 기준)으로 일반화를 측정:

| 모델 | 평균 F1 | CV (낮을수록 좋음) |
|------|--------|-----------------|
| **LightESD-1** | **0.81** | **0.02** (최저) |
| OC-SVM | 0.58 | 0.03 |
| BRVFL-AE | 0.76 | 0.05 |
| ONLAD | 0.72 | 0.06 |
| LightESD-2 | 0.87 | 0.06 |
| LOF | 0.66 | 0.10 |
| Iso. Forest | 0.58 | 0.12 |

LightESD-1은 **CV 0.02**로 가장 낮은 변동성을 보이며, 이는 계절적/비계절적/랜덤 워크/실제 데이터 등 **다양한 유형의 데이터에 걸쳐 가장 일관된 성능**을 유지함을 의미한다.

### 3.3 일반화 성능을 더 향상시킬 수 있는 방향

1. **다변량(multivariate) 시계열 확장**: 현재 단변량에 한정된 프레임워크를 다변량으로 확장하면, IoT 환경에서 다중 센서 데이터에 대한 일반화 가능성이 크게 증가할 수 있다.
2. **온라인 학습으로의 전환**: 배치 학습에서 순수 온라인 학습으로 전환하면 streaming 데이터에 대한 적응력이 향상된다.
3. **적응적 $\alpha$ 조정**: 현재 유의수준 $\alpha$가 고정(0.05 또는 0.001)되어 있으나, 데이터 특성에 따라 자동으로 최적 $\alpha$를 선택하는 메커니즘을 추가하면 일반화가 더 강화될 수 있다.
4. **다중 분해 전략의 자동 선택**: RobustTrend와 FastRobust-STL 외에도 다양한 분해 방법을 자동으로 선택하는 메타 전략이 가능하다.

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 연구에 미치는 영향

#### (1) 엣지 AI 이상 탐지의 패러다임 전환
- 기존 "중앙 서버 학습 → 엣지 배포" 패러다임에서 **"완전 엣지 기반 학습 및 배포"** 패러다임으로의 전환을 제시
- 이는 통신 인프라가 열악한 원격지, 농촌, 재난 현장 등에서의 자율적 이상 탐지 가능성을 열어줌

#### (2) 통계학적 방법의 재평가
- 딥러닝 일변도의 연구 트렌드에서, **잘 설계된 통계적 방법이 자원 제한 환경에서 딥러닝을 능가**할 수 있음을 실증적으로 보여줌
- 특히 "잔차 기반 탐지"라는 전략의 효과성을 입증하여, 향후 통계-딥러닝 하이브리드 연구의 기반을 마련

#### (3) 종합 평가 메트릭(ADCompScore)의 기여
- 엣지 환경에서의 이상 탐지 모델 평가가 단순 F1-score를 넘어 **자원 소비, 전력, 지연 등을 종합적으로 고려**해야 한다는 평가 프레임워크를 확립
- 이는 향후 엣지 AI 연구의 벤치마크 표준으로 활용될 가능성이 있음

#### (4) AutoML 관점에서의 시사점
- 수동 하이퍼파라미터 튜닝 없이 자동 적응하는 모델 설계의 중요성을 강조
- Fully-automated 특성은 대규모 IoT 배포 시나리오에서 운영 비용을 크게 절감

### 4.2 향후 연구 시 고려할 점

1. **온라인/스트리밍 학습으로의 확장**: 논문에서 명시적으로 언급한 한계로, 배치 학습에서 순수 온라인 학습으로의 전환이 필요하다. 이를 위해 sliding window 기반 incremental ESD나 adaptive threshold 메커니즘이 고려될 수 있다.

2. **다변량 시계열 지원**: 실제 IoT 환경에서는 다수의 센서가 동시에 데이터를 생성하므로, 변수 간 상관관계를 고려한 다변량 확장이 필수적이다.

3. **이상 유형의 다양화**: 현재 spike, dip, collective anomaly에 초점을 맞추고 있으나, contextual anomaly나 점진적 drift 등 다양한 이상 유형에 대한 탐지 능력 검증이 필요하다.

4. **대규모 데이터셋 검증**: 5,000 데이터포인트의 합성 데이터와 제한된 실제 데이터셋으로 평가했으므로, 더 대규모이고 다양한 도메인의 벤치마크(예: UCR Archive, KDD Cup 등)에서의 검증이 필요하다.

5. **적대적(adversarial) 환경 고려**: 엣지 디바이스는 보안 공격에 취약할 수 있으므로, adversarial perturbation에 대한 강건성 평가가 필요하다.

6. **분해 방법의 실패 사례 분석**: RobustTrend나 FastRobust-STL이 잘 작동하지 않는 데이터 유형(예: 비정상적으로 복잡한 계절성, 비선형 추세)에 대한 분석이 필요하다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 비교 대상 연구들

| 연구 | 연도 | 핵심 접근법 | 엣지 배포 | 자동화 | 일반화 |
|------|------|-----------|---------|------|------|
| **LightESD** (본 논문) | 2023 | 통계학습 (ESD 기반) | ✅ On-device | ✅ Fully-auto | ✅ Non-parametric, weight-free |
| **TadGAN** [5] (Geiger et al.) | 2020 | GAN 기반 시계열 이상 탐지 | ❌ 서버 필요 | ❌ 하이퍼파라미터 튜닝 필요 | △ 다양한 데이터에 적용 가능하나 학습 비용 높음 |
| **ONLAD** [18] (Tsukada et al.) | 2020 | OS-ELM 기반 단층 FFNN | ✅ On-device | ❌ 뉴런 수 등 설정 필요 | △ 단일 엣지 배포, 가중치 저장 필요 |
| **ANNet** [6] (Sivapalan et al.) | 2022 | 경량 CNN (ECG 이상 탐지) | △ 추론만 엣지 | ❌ 중앙 서버에서 학습 | △ ECG 특화 |
| **LightLog** [4] (Wang et al.) | 2022 | 경량 TCN (로그 이상 탐지) | △ 추론만 엣지 | ❌ 중앙 서버에서 학습 | △ 로그 데이터 특화 |
| **EPBRVFL-AE** [20] (Odiathevar et al.) | 2022 | 베이지안 RVFL AutoEncoder | △ 분산 학습 | ❌ 하이퍼파라미터 설정 필요 | △ 분산 환경 필요 |
| **Fast RobustSTL** [27] (Wen et al.) | 2020 | 시계열 분해 (STL 개선) | N/A (분해 도구) | ✅ 자동 분해 | ✅ 복잡한 패턴 처리 |
| **RobustTrend** [26] (Wen et al.) | 2019 | 추세 필터링 (Huber loss) | N/A (분해 도구) | ✅ 자동 추세 추출 | ✅ 이상치에 강건 |

### 주요 차이점 분석

#### (1) TadGAN [5] (2020) vs LightESD
- **TadGAN**: GAN의 생성자(Generator)와 판별자(Discriminator)를 활용하여 시계열 재구성 오류 기반으로 이상을 탐지. 복잡한 비선형 패턴 포착에 강하나, 대규모 파라미터 학습이 필요하고, 엣지 디바이스에서 직접 학습이 사실상 불가능.
- **LightESD 우위**: 완전 엣지 기반 학습, 가중치 불필요, 동등하거나 우수한 탐지 성능을 극히 적은 자원으로 달성.

#### (2) ONLAD [18] (2020) vs LightESD
- **ONLAD**: OS-ELM 기반으로 엣지에서 직접 학습 가능하나, 히든 레이어 뉴런 수(16/64/128) 등의 하이퍼파라미터를 수동 설정해야 하며, 가중치 행렬과 편향 벡터를 저장해야 함.
- **LightESD 우위**: F1-score 평균 0.81~0.87 vs ONLAD 0.72, CV 0.02~0.06 vs ONLAD 0.06, 전력 소비 14.3% vs 16.3%.

#### (3) EPBRVFL-AE [20] (2022) vs LightESD
- **EPBRVFL-AE**: 분산(distributed) 학습으로 통신 오버헤드 발생, 중앙 서버와 통신 네트워크 필요. 엣지 단독 운영 불가.
- **LightESD 우위**: 통신 네트워크 불필요, ADCompScore 0.92~0.93 vs BRVFL-AE 0.89, CPU 사용률 5.47% vs 10.96%.

#### (4) 최근 Transformer 기반 이상 탐지 연구와의 관계
- **Anomaly Transformer** (Xu et al., ICLR 2022)나 **PatchTST** (Nie et al., ICLR 2023) 등 Transformer 기반 방법은 높은 탐지 정확도를 달성하나, 모델 크기와 연산량이 매우 커서 엣지 배포가 어렵다.
- LightESD는 이러한 고성능 모델들과 **다른 설계 철학(경량성, 자원 효율성)**을 추구하며, 특히 **자원 제약이 심한 엣지 환경**에서의 실용성에 초점을 맞춘다.

#### (5) 최근 AutoML/자동화 시계열 연구와의 비교
- **AutoAI-TS** [25] (Shah et al., 2021): 사전 정의된 주기(hourly/daily/weekly)만 탐지 가능하여, 비표준 주기에 대한 탐지 한계가 있음.
- LightESD는 주파수 도메인에서 **임의의 주기를 자동 탐지**하므로 더 높은 유연성을 가짐.

### 종합 평가

LightESD는 딥러닝 중심의 이상 탐지 연구 흐름에서 **통계학적 접근의 실용적 가치를 재조명**한 의미 있는 연구이다. 특히 엣지 컴퓨팅이라는 명확한 응용 맥락에서, 탐지 정확도와 자원 효율성의 **최적 균형점(trade-off)**을 찾아낸 것이 핵심 기여이다. 향후 온라인 학습 확장, 다변량 지원, 더 대규모 벤치마크 검증이 이루어진다면, 엣지 AI 이상 탐지의 실질적 표준(de facto standard)으로 자리잡을 가능성이 있다.

---

## 참고자료 및 출처

1. **본 논문**: Das, R. & Luo, T. "LightESD: Fully-Automated and Lightweight Anomaly Detection Framework for Edge Computing," arXiv:2305.12266v1, IEEE EDGE 2023.
2. **[5]** Geiger, A. et al., "TadGAN: Time Series Anomaly Detection Using Generative Adversarial Networks," IEEE International Conference on Big Data, 2020.
3. **[18]** Tsukada, M. et al., "A Neural Network-Based On-Device Learning Anomaly Detector for Edge Devices," IEEE Transactions on Computers, vol. 69, no. 7, 2020.
4. **[20]** Odiathevar, M. et al., "A Bayesian Approach to Distributed Anomaly Detection in Edge AI Networks," IEEE Transactions on Parallel and Distributed Systems, vol. 33, no. 12, 2022.
5. **[6]** Sivapalan, G. et al., "ANNet: A Lightweight Neural Network for ECG Anomaly Detection in IoT Edge Sensors," IEEE Transactions on Biomedical Circuits and Systems, vol. 16, no. 1, 2022.
6. **[4]** Wang, Z. et al., "LightLog: A Lightweight Temporal Convolutional Network for Log Anomaly Detection on the Edge," Computer Networks, vol. 203, 2022.
7. **[26]** Wen, Q. et al., "RobustTrend: A Huber Loss with a Combined First and Second Order Difference Regularization for Time Series Trend Filtering," IJCAI, 2019.
8. **[27]** Wen, Q. et al., "Fast RobustSTL: Efficient and Robust Seasonal-Trend Decomposition for Time Series with Complex Patterns," ACM SIGKDD, 2020.
9. **[33]** Rosner, B., "Percentage Points for a Generalized ESD Many-Outlier Procedure," Technometrics, vol. 25, no. 2, 1983.
10. **[35]** Rousseeuw, P.J. & Croux, C., "Alternatives to the Median Absolute Deviation," Journal of the American Statistical Association, vol. 88, no. 424, 1993.
11. **[12]** Hochenbaum, J. et al., "Automatic Anomaly Detection in the Cloud via Statistical Learning," arXiv:1704.07706, 2017.
12. **[23]** Vlachos, M. et al., "On Periodicity Detection and Structural Periodic Similarity," SIAM International Conference on Data Mining, 2005.
13. **[24]** Lavin, A. & Ahmad, S., "Evaluating Real-Time Anomaly Detection Algorithms – The Numenta Anomaly Benchmark," IEEE ICMLA, 2015.

> **정확도 관련 참고사항**: 본 분석은 제공된 논문 원문의 내용에 충실하게 작성되었습니다. Transformer 기반 이상 탐지(Anomaly Transformer, PatchTST 등)와의 직접적인 정량 비교는 본 논문에서 수행되지 않았으므로, 해당 부분은 일반적인 연구 동향 기반의 정성적 비교임을 밝힙니다.
