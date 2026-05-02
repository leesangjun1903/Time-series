# TSB-UAD: An End-to-End Benchmark Suite for Univariate Time-Series Anomaly Detection

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

TSB-UAD의 핵심 주장은 **기존 시계열 이상 탐지(AD) 연구가 편향된 데이터셋, 파라미터 선택, 평가 지표로 인해 '진보의 환상(illusion of progress)'을 만들어 내고 있다**는 것입니다. 이를 해결하기 위해 포괄적이고 재현 가능한 엔드-투-엔드 벤치마크 스위트를 제안합니다.

### 주요 기여

| 기여 항목 | 세부 내용 |
|---|---|
| **대규모 데이터셋 통합** | 18개 공개 데이터셋(1,980개 시계열) 수집 및 통합 포맷 제공 |
| **인공 데이터셋 생성** | UCR Archive 기반 126개 데이터셋에서 958개 시계열 생성 |
| **합성 데이터셋 생성** | 11가지 변환 적용, 92개 데이터셋 / 10,828개 시계열 생성 |
| **총 규모** | **13,766개 시계열**, 다양한 도메인/이상 유형/밀도 포괄 |
| **평가 프레임워크** | 9가지 평가 지표 + 비모수 통계 검정 포함 |
| **난이도 측정 지표** | RC, NC, NA 등 데이터셋 난이도 정량화 지표 제안 |
| **12개 대표 방법 평가** | 기존 및 딥러닝 기반 방법들의 체계적 비교 실험 수행 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

논문은 기존 시계열 AD 평가의 세 가지 핵심 편향을 지적합니다:

**(1) 데이터셋 선택 편향**: 특정 이상 유형(예: point anomaly)만 포함한 데이터로 평가 시, 해당 유형에 특화된 방법이 과대평가됨.

**(2) 모델 파라미터 선택 편향**: 데이터셋 특성에 따라 임계값, 윈도우 크기 등의 파라미터가 일부 방법에 불균형한 이점을 제공.

**(3) 평가 지표 선택 편향**: Precision/Recall은 임계값 의존적이며 집합적 이상(collective anomaly) 평가에 부적절할 수 있음.

**(4) 기존 데이터셋의 결함**: 일부 데이터셋은 trivial한 1줄짜리 baseline으로도 SOTA 성능 달성 가능(Wu & Keogh, 2020).

---

### 2.2 제안하는 방법 (수식 포함)

#### 2.2.1 인공 데이터셋 생성 방법론

UCR Archive의 시계열 분류 데이터셋을 이상 탐지 데이터셋으로 변환하는 원칙적 방법론:

**Step 1: 혼동 인자(Confusion Factor) 계산**

레이블 $j$에서 레이블 $k$로의 혼동 인자 $c_{jk}$는 다음과 같이 정의됩니다:

$$P(\hat{y} = k \mid x) = c_{jk}$$

여기서 $x$는 레이블 $y=j$를 가진 시계열이며, $\hat{y}$는 1-NN 분류기의 예측값입니다.

**Step 2: 친화도(Affinity) 행렬 구성**

레이블 $j$와 $k$ 사이의 정규화된 친화도:

$$\text{Affinity}(j, k) = \frac{c_{jk} + c_{kj}}{2}$$

**Step 3: 최대 신장 트리(MST) 및 이분 착색**

친화도 행렬 기반 MST를 구성하고, 인접 노드를 정상(파란색)과 비정상(빨간색)으로 이분 착색합니다.

**Step 4: 시계열 생성 파라미터**

- 비정상 세그먼트 수: $K$
- 이상 서브시퀀스 비율: $r$
- 정상 세그먼트 수: $N = K/r - K$
- 비정상 집합의 레이블 수: $m$, 각 레이블의 데이터 수: $n$
- 가장 빈번한 비정상 레이블의 빈도: $f_a$
- 각 정상 레이블의 빈도: $f_n = 20 f_a$
- 선택된 정상 레이블 수: $L = N / f_n$

---

#### 2.2.2 데이터 변환 (합성 데이터셋 생성)

원본 시계열 $X = (x_0, x_1, \ldots, x_n)$, 표준편차 $s$로 표기.

**글로벌 변환 (Global Transformations)**

- **랜덤 워크 배경** ($p_{rw}$: 강도):

$$x'_i = x_i + p_{rw} \cdot s \cdot b_i, \quad b_n = \sum_{i=1}^{n} Z_i,\ Z_i \in \{-1, 0, 1\}$$

- **백색 잡음** ( $p_{wn}$: 강도, $b_i \sim \mathcal{N}(0,1)$ ):

$$x'_i = x_i + p_{wn} \cdot s \cdot b_i$$

- **포인트 이상값** ($p_{or}$: 이상 비율):

$$x'_i = x_i + 5s \quad \text{또는} \quad x'_i = \max(X)$$

- **가우시안 스무딩** (필터 $f$, 라인 폭 $p_{sm}$):

$$X' = X * f = \mathcal{F}^{-1}(\mathcal{F}(X)\mathcal{F}(f))$$

- **이중 스무딩 윈도우** ($p_1, p_2$: 두 구간 스무딩 파라미터):

$$X' = [X_{0:n/2} * f_{p_1},\ X_{n/2:n+1} * f_{p_2}]$$

---

#### 2.2.3 평가 지표 (Evaluation Measures)

**기본 지표:**

$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall (TPR)} = \frac{TP}{TP + FN}$$

$$\text{F-score} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

$$\text{FPR} = \frac{FP}{FP + TN}$$

**AUC-ROC**: TPR과 FPR의 관계를 임계값 변화에 따라 기록한 ROC 곡선의 면적. 임계값 설정에 무관한 강건한 지표.

**Precision@k** ($k$: 비정상 포인트 수):

$$\text{Precision@k} = \frac{TP@k}{k}$$

**범위 기반 지표 (Range-Precision, Range-Recall)** [Tatbul et al., 2018]: 연속 이상 구간(subsequence)의 감지 비율, 위치, 파편화 등을 고려한 확장 지표.

---

#### 2.2.4 데이터셋 난이도 측정 지표

**상대 대조도 (Relative Contrast, RC)** [He et al., 2012]:

$$R_c = \frac{\mathbb{E}_{s \in S}[D_{mean}(s)]}{\mathbb{E}_{s \in S}[D_{min}(s)]}$$

$R_c$가 1에 가까울수록 평균 거리와 최근접 이웃 거리가 유사함을 의미 → 데이터가 균일하게 분포 → 클러스터링 난이도 증가.

**정규화 군집성 (Normalized Clusteredness, NC)**:

$$N_c = \frac{\mathbb{E}_{s_i, s_j \in S_{nor}, i < j}[D(s_i, s_j)]}{\mathbb{E}_{s_i, s_j \in S_{ano}, i < j}[D(s_i, s_j)]}$$

$N_c$가 클수록 비정상 포인트들이 서로 가깝게 군집 → 탐지 난이도 증가.

**정규화 인접도 (Normalized Adjacency, NA)** (논문의 신규 제안):

$$N_a = \frac{\min_{c_i \in C_{ano}, c_j \in C_{nor}} D(c_i, c_j)}{\mathbb{E}_{c_i, c_j \in C_{nor}, i < j}[D(c_i, c_j)]}$$

$N_a$가 클수록 비정상 클러스터가 정상 클러스터로부터 더 멀리 위치 → 탐지 용이.

---

#### 2.2.5 통계적 검정

- **쌍별 비교**: Wilcoxon signed-rank test
- **다중 방법 비교**: Friedman test + post-hoc Nemenyi test
- 유의 수준 $\alpha = 0.05$

---

### 2.3 평가된 12가지 방법의 구조 요약

| 방법 | 범주 | 핵심 원리 |
|---|---|---|
| **IForest / IForest1** | 비지도 (근접성) | 이진 트리 공간 분할; 짧은 경로 길이 = 이상 |
| **LOF** | 비지도 (근접성) | 지역 밀도 대비 이웃 밀도 비율 |
| **HBOS** | 비지도 (근접성) | 히스토그램 빈 높이의 역수 = 이상 점수 |
| **MP (Matrix Profile)** | 비지도 (불일치) | 서브시퀀스 간 1-NN 거리 최대값 = 이상 |
| **NORMA** | 비지도 (클러스터링) | 정상 패턴 클러스터 구성 후 각 포인트의 유효 거리 |
| **PCA** | 비지도 (선형) | 저차원 초평면 투영; 거리 큰 포인트 = 이상 |
| **AE (Autoencoder)** | 반지도 (딥러닝) | 잠재 공간 투영 후 재구성 오차 |
| **LSTM-AD** | 반지도 (딥러닝) | 2층 LSTM + Dense; 예측-실제 편차 |
| **CNN (DeepAnT)** | 반지도 (딥러닝) | 3개 Conv Block + MaxPooling; 예측-실제 편차 |
| **POLY** | 반지도 (통계) | 다항식 근사; 예측-실제 편차 |
| **OCSVM** | 반지도 (커널) | 정상 데이터 경계 학습 |

**CNN 세부 구조**:
- 3개 Convolutional Block (filters=8,16,32, kernel\_size=2, strides=1) + MaxPooling(pool\_size=2) + ReLU
- Dense(64) → Dropout(0.2) → Dense(1)
- loss='mse', optimizer='adam', epochs=100, patience=5

**AE 세부 구조** (최적 모델 선택):
- 구조 후보: (32,16,8,16,32), (32,8,32), (32,16,32) MLP 기반
- Activation: ReLU, loss='mse', optimizer='adam', epochs=100

---

### 2.4 성능 향상 결과

**전체 벤치마크 (공개 데이터셋 18개, 세밀 분석)**:
- NORMA와 MP가 나머지 방법 대비 유의미하게 우수 (Friedman+Wilcoxon, $\alpha=0.05$)

**이상 유형별 성능 분화**:

| 방법 | Point-based 이상 | Sequence-based 이상 |
|---|---|---|
| CNN, LSTM-AD | **최상위** | 최하위 수준 |
| NORMA, MP | 3~4위권 | **최상위** |
| POLY, AE, IForest | 중간 | 중간 |

**주요 발견**:
- ECG (단일 반복 패턴, 다수 집합적 이상) → NORMA 우세 (반복 이상에 1-NN 거리가 무력화되는 MP의 한계)
- MGAB (다중 정상 패턴, 소수 이상) → MP 우세 (정상 클러스터 다양성이 NORMA의 가중 거리 계산을 혼란)
- 인공 데이터셋 (126개) → MP 1위 (불균형 샘플링으로 인한 인공적 편향)
- 합성 데이터셋 (92개) → NORMA 1위 (실제적 다양성 반영)

---

### 2.5 한계점

1. **반지도 방법의 훈련 데이터 문제**: LSTM-AD, AE, CNN, OCSVM 등은 이상 없는 훈련 데이터 필요. KDD21, NASA-SMAP/MSL 외에는 초기 10~30%를 훈련에 사용하여 오염 가능성 존재.

2. **파라미터 최적화 미흡**: 공정성을 위해 기본값 사용, 개별 튜닝 시 성능 개선 여지 있음.

3. **단변량 한정**: 다변량 시계열은 각 차원 독립 처리 (AUC > 0.8 차원만 선택). 다변량 상관관계 미반영.

4. **변환 범위의 제한**: 현재 변환은 전체 스펙트럼의 일부. Markov Switching 모델 등 고급 변환은 향후 과제.

5. **평가 방법 수의 제한**: 100개 이상의 기존 방법 중 12개만 평가.

6. **인공 데이터셋의 구조적 편향**: 불균형 샘플링(비정상 1~2개 vs. 정상 20개)이 MP에 인위적 유리함 제공.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 TSB-UAD가 일반화 평가에 기여하는 방식

TSB-UAD는 **세 가지 축**에서 일반화 성능을 체계적으로 평가합니다:

#### (A) 데이터 다양성을 통한 일반화 평가

```
공개 데이터셋 (18개, 1,980 시계열)
    → 도메인 다양성: IoT, 의료(ECG), 우주 텔레메트리, 교통, 환경 센서 등
    → 이상 유형 다양성: Point / Contextual / Collective

인공 데이터셋 (126개, 958 시계열)
    → 실세계 분류 데이터셋 기반 (90% 이상 실제 데이터)
    → 제어된 이상 밀도와 다양한 class affinity 수준

합성 데이터셋 (92개, 10,828 시계열)
    → 11가지 변환으로 점진적 난이도 조절
    → 동일 메서드의 강건성 (robustness) 정량 평가 가능
```

#### (B) 변환 강도와 일반화 성능의 상관관계

논문은 변환 강도($p_{rw}$, $p_{wn}$)와 평균 AUC의 관계를 분석합니다:

- **랜덤 워크 배경, 백색 잡음**: 강도 증가 → AUC 단조 감소 (직관 일치)
- **스무딩**: AUC와 강도 간 명확한 상관 없음
- **IForest vs. MP**: 백색 잡음 $p_{wn} = 0.25$ 부근에서 AUC 역전 → IForest가 MP보다 잡음에 강건

이는 각 방법의 **도메인 이동(domain shift) 하에서의 일반화 강건성** 차이를 보여줍니다.

#### (C) NC, NA, RC와 일반화 성능의 상관관계

$$\text{NC} \uparrow \Rightarrow \text{AUC 감소 (MP: -27\%, NORMA: -25\%, IForest: -31\%)}$$

$$\text{NA} \uparrow \Rightarrow \text{AUC 증가 (IForest: +20\%, NORMA: +45\%)}$$

이는 **데이터 구조의 기하학적 특성**이 일반화 성능에 미치는 영향을 정량화합니다.

#### (D) 일반화 성능 향상을 위한 시사점

**① 방법론 관점**:
- NORMA는 정상 패턴 클러스터 기반 접근으로 **반복적 이상(cardinality 높음)**에 강하며, 다양한 데이터셋에 걸쳐 안정적 순위 유지 → 일반화 성능 우수.
- MP는 단일 패턴 시계열(낮은 cardinality)에서 우세하나 다중 패턴 시계열에서 취약 → **범용성 한계**.
- 딥러닝(CNN, LSTM)은 point anomaly에 특화되어 있어 **이상 유형 분포 변화에 취약** → 일반화 성능 낮음.

**② 데이터 증강 관점**:
- 변환 기반 합성 데이터셋은 **동일 분포 내 데이터 증강**을 가능케 하며, 특정 변환 유형에 대한 방법의 강건성을 체계적으로 평가.
- 이상 밀도, 이상 유형, 정상/비정상 친화도를 제어함으로써 **out-of-distribution 시나리오** 시뮬레이션 가능.

**③ 평가 지표 관점**:
- AUC-ROC는 임계값 무관, 정상/비정상 비율 변화에 강건 → **다양한 데이터 분포**에서도 일관된 평가 가능.
- F-score와 RF는 임계값 의존적이지만 **lag에 덜 민감** → 특정 용도에 적합.

**④ 향후 일반화 향상 방향** (논문 직접 언급):
- Markov Switching 모델, 가변 파라미터 모델 등 **고급 변환** 추가 계획.
- 커뮤니티 입력 기반 **정기 업데이트 리더보드** → 지속적 일반화 평가 기반 구축.

---

## 4. 해당 논문이 앞으로의 연구에 미치는 영향과 연구 시 고려할 점

### 4.1 미래 연구에 미치는 영향

#### (A) 벤치마크 표준화 측면

TSB-UAD는 시계열 AD 분야에서 **UCR Archive (분류) / M-Competition (예측)에 상응하는 표준 벤치마크**로 자리잡을 가능성이 높습니다. 이는:
- 연구 결과 간 직접 비교 가능성 증가
- "진보의 환상" 억제 → 실질적 방법론 발전 촉진
- 재현성 기반 연구 문화 정착

#### (B) 방법론 개발 방향 측면

- **하이브리드 방법**: NORMA(일반성)와 CNN/LSTM(point anomaly 특화)을 결합한 앙상블 방법 연구 촉진.
- **이상 유형 적응 방법**: 데이터 특성(NC, NA, RC)을 자동으로 분석하여 최적 방법을 선택하는 **AutoML 스타일 AD 프레임워크** 연구 가속.
- **사전 학습 기반 전이 학습**: 13,766개 시계열의 대규모 데이터를 활용한 사전 학습 모델 개발 가능.

#### (C) 평가 방법론 측면

- Range-based Precision/Recall의 표준화.
- 단일 지표 대신 **다중 지표 통합 평가** 정착.
- 비모수 통계 검정(Wilcoxon, Friedman, Nemenyi)의 AD 평가 표준 적용.

---

### 4.2 앞으로 연구 시 고려할 점

#### (1) 데이터셋 다양성 확보의 필수성

- 단일 도메인(예: Yahoo, NAB)만으로는 방법의 일반화 성능을 보장할 수 없음.
- 최소한 **point, contextual, collective anomaly** 세 유형을 모두 포함하는 데이터셋에서 평가 필요.

#### (2) 평가 지표의 신중한 선택

- 임계값 의존 지표(F-score, RF) 사용 시 임계값 설정 방법 명확히 보고.
- AUC를 기본 지표로 사용하되, 극심한 클래스 불균형 시 Precision-Recall Curve 보완.
- 집합적 이상이 포함된 경우 반드시 **Range-F score** 병용.

#### (3) 반지도/지도 방법의 공정한 비교

- 훈련 데이터의 오염 여부 및 비율 명시 필수.
- 이상 없는 훈련 데이터 가용 여부에 따른 실험 조건 분리.

#### (4) 난이도 측정 지표의 활용

- NC, NA, RC를 계산하여 데이터셋 특성을 사전 분석.
- 낮은 $R_c$ 값의 데이터셋에서의 성능 저하 원인 분석 필요.

#### (5) 계산 효율성과 정확도 간 트레이드오프 고려

- 딥러닝 모델은 CPU 기준 다른 방법 대비 **1~2 order of magnitude** 높은 연산 시간 (Figure 10 참조).
- 실시간/스트리밍 환경에서는 POLY, PCA 등 경량 방법이 현실적 대안.

#### (6) 다변량 확장 필요성

- 현재 TSB-UAD는 단변량에 한정. 다변량 시계열의 변수 간 상관관계를 고려한 벤치마크 필요.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

> **⚠️ 주의**: 아래 비교는 TSB-UAD 논문(2022) 내 인용 및 제한된 공개 정보에 기반합니다. 논문 외 부분에 대해서는 확실한 내용만 기술하고, 불확실한 사항은 명시합니다.

### 5.1 TSB-UAD 논문 내 직접 비교된 2020년 이후 연구

| 연구 | 연도 | 핵심 내용 | TSB-UAD와의 관계 |
|---|---|---|---|
| **Wu & Keogh (2020)** [98] | 2020 | 기존 벤치마크(Yahoo, NAB 등)의 결함 지적 | TSB-UAD의 동기 제공; 결함 인정하되 맥락화 |
| **Lai et al. (2021)** [52] | 2021 | TODS 벤치마크; 합성 데이터 기반 | TSB-UAD는 실제 데이터 중심으로 보완 |
| **Boniol et al. (2021)** [14,15] | 2021 | NORMA: 비지도 서브시퀀스 AD | TSB-UAD 평가 대상 방법; 전반적 1위권 |
| **Jacob et al. (2020)** [48] | 2020 | Exathlon: 설명 가능 AD 벤치마크 | 단일 애플리케이션/이상 유형 한정 vs. TSB-UAD의 포괄성 |
| **KDD21 Competition [49]** | 2021 | 250개 단일 이상 시계열 경진대회 | top-1 ground truth의 편향 문제 지적 |

### 5.2 TSB-UAD 이후 등장한 관련 연구 동향 (공개된 정보 기반)

> **정확도 한계**: 아래는 2022년 이후 연구에 대한 일반적 동향으로, 구체적 수치 성능은 해당 논문들을 직접 확인하셔야 합니다.

- **Transformer 기반 AD**: Anomaly Transformer(Zhou et al., 2022, ICLR) 등 어텐션 메커니즘 기반 방법들이 등장하였으나, TSB-UAD와 같은 포괄적 벤치마크에서의 집합적 이상 탐지 성능은 별도 검증 필요.

- **Foundation Model 접근**: 대규모 사전 학습 모델의 시계열 AD 적용 연구가 활발하나, TSB-UAD 수준의 다양한 데이터셋에서의 일반화 성능 검증은 아직 미흡한 상태.

- **TSAD-Eval, TimeSeAD 등 후속 벤치마크**: TSB-UAD의 문제의식을 이어받은 평가 프레임워크 연구들이 등장하고 있으나, TSB-UAD 수준의 데이터셋 규모와 포괄성을 갖추고 있는지는 각 논문을 직접 확인하셔야 합니다.

---

## 참고 자료

**1차 참고 자료 (본 분석의 주된 출처)**

- **Paparrizos, J., Kang, Y., Boniol, P., Tsay, R. S., Palpanas, T., & Franklin, M. J. (2022). TSB-UAD: An End-to-End Benchmark Suite for Univariate Time-Series Anomaly Detection. PVLDB, 15(1). LIPADE-TR-No 6, March 24, 2022.**

**논문 내 인용 주요 참고 문헌**

- Wu, R., & Keogh, E. J. (2020). Current Time Series Anomaly Detection Benchmarks are Flawed and are Creating the Illusion of Progress. *arXiv:2009.13807*.
- Tatbul, N., et al. (2018). Precision and Recall for Time Series. *NeurIPS 2018*.
- Emmott, A. F., et al. (2013). Systematic construction of anomaly detection benchmarks from real data. *ACM SIGKDD Workshop ODD*.
- Lai, K.-H., et al. (2021). Revisiting Time Series Outlier Detection: Definitions and Benchmarks. *NeurIPS Track on Datasets and Benchmarks*.
- Jacob, V., et al. (2020). Exathlon: A Benchmark for Explainable Anomaly Detection over Time Series. *arXiv:2010.05073*.
- Boniol, P., et al. (2021). Unsupervised and scalable subsequence anomaly detection in large data series. *The VLDB Journal*.
- Dau, H. A., et al. (2018). The UCR Time Series Classification Archive.
- Blázquez-García, A., et al. (2021). A Review on outlier/Anomaly Detection in Time Series Data. *ACM Computing Surveys (CSUR)*.
- Braei, M., & Wagner, S. (2020). Anomaly detection in univariate time-series: A survey. *arXiv:2004.00433*.
- He, J., Kumar, S., & Chang, S.-F. (2012). On the difficulty of nearest neighbor search. *ICML 2012*.
