# Estimating the Electrical Power Output of Industrial Devices with End-to-End Time-Series Classification in the Presence of Label Noise

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문은 **산업용 열병합발전기(CHP: Combined Heat and Power)의 전력 출력을 시설 전체 전력 소비 데이터만으로 추정**하는 문제를 다룬다. 핵심 주장은 다음과 같다:

> 센서 오류로 인해 자동 생성된 훈련 레이블에 노이즈가 포함될 수밖에 없는 실세계 산업 환경에서, **레이블 노이즈 비율을 사전에 알지 못해도 자기지도(self-supervised) 방식으로 오레이블을 점진적으로 교정**할 수 있는 딥러닝 기반 멀티태스크 프레임워크(SREA)가 효과적이다.

### 주요 기여

| 기여 항목 | 내용 |
|---|---|
| **새로운 알고리즘** | SREA (Self-Re-Labeling with Embedding Analysis) 제안 |
| **실세계 문제 정의** | CHP 전력 추정을 시계열 분류 + 레이블 노이즈 문제로 공식화 |
| **노이즈 비율 독립성** | 사전에 노이즈 비율 $\epsilon$ 을 알 필요가 없음 |
| **다양한 노이즈 유형 평가** | 대칭(Symmetric), 비대칭(Asymmetric), 플립(Flip) 노이즈 모두 실험 |
| **시계열 분야 최초 평가** | 시계열 분류에서 레이블 노이즈 문제를 최초로 체계적으로 평가 |
| **절제 연구(Ablation Study)** | 손실 함수 구성요소 및 하이퍼파라미터 민감도 분석 |

---

## 2. 해결하고자 하는 문제, 제안 방법(수식 포함), 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

**문제 배경:**
- 중규모 기업 시설의 총 전력 소비 신호 $P_{tot}$만으로 CHP의 출력 전력 $P_{CHP}$를 추정
- 훈련 레이블을 CHP 센서로부터 자동 생성하지만, **센서 오류로 인해 레이블 노이즈 발생**
- 딥러닝 모델은 노이즈 레이블에 쉽게 과적합(overfitting)되어 실세계 적용이 어려움

**공식적 문제 정의:**

$k$-클래스 분류 문제로, 훈련 데이터셋은 다음과 같이 정의된다:

$$\mathcal{D} = \{(\mathbf{x}_i, \mathbf{y}_i),\ i = 1, \ldots, n\}, \quad \mathbf{y}_i \in \{0,1\}^k$$

여기서 $\mathbf{y}_i$는 원-핫 인코딩 레이블이며, 일부 레이블은 노이즈에 의해 오염되어 있다.

**레이블 노이즈 모델링:**

레이블 전이 행렬 $T$를 사용하며, $T_{ij}$는 레이블 $i$가 $j$로 뒤바뀔 확률이다.

- **대칭(Symmetric) 노이즈:** $T_{ii} = 1-\epsilon$, $T_{ij} = \dfrac{\epsilon}{k-1}\ (i \neq j)$

- **비대칭(Asymmetric) 노이즈:** $T_{ii} = 1-\epsilon$, $T_{(j+1 \bmod k)j} = \epsilon\ (i \neq j)$

- **플립(Flip) 노이즈 (센서 오류 모사):** $T_{ii} = 1-\epsilon$, $T_{i0} = \epsilon$

---

### 2.2 제안 방법: SREA (Self-Re-Labeling with Embedding Analysis)

#### 아키텍처 구성

SREA는 **세 가지 모듈**이 **공유 임베딩 표현**을 기반으로 작동하는 멀티태스크 딥러닝 구조다:

```
입력 데이터 x
      │
   Encoder e(·)   ← 공유 임베딩 생성 (32차원)
   ┌────┴────┐
Decoder    Classifier   Constrained Clustering
(f_ae)      (f_c)           (f_cc)
   │           │               │
재구성 x̂    클래스 확률 p^c   클러스터 레이블 y^cc
                 └──────────────┘
                  레이블 교정 모듈 → y*
```

#### 손실 함수

**① 오토인코더 재구성 손실 (레이블 노이즈에 영향받지 않음):**

$$\mathcal{L}_{ae} = \frac{1}{n} \sum_{i=1}^{n} (\hat{\mathbf{x}}_i - \mathbf{x}_i)^2 $$

**② 분류기 크로스 엔트로피 손실:**

$$\mathcal{L}_{c} = -\frac{1}{n} \sum_{i=1}^{n} \mathbf{y}_i^T \cdot \log(\mathbf{p}_i^c) $$

여기서 $\mathbf{p}_i^c = \text{softmax}(f_c(\mathbf{x}_i))$

**③ 제약 클러스터링 손실:**

$$\mathcal{L}_{cc} = \frac{1}{n} \sum_{i=1}^{n} \left[ \underbrace{\|e(\mathbf{x}_i) - \mathbf{C}_{\mathbf{y}_i}\|_2^2}_{\text{intra-class}} + \underbrace{\log \sum_{j=1}^{k} \exp\left(-\|e(\mathbf{x}_i) - \mathbf{C}_j\|_2\right)}_{\text{inter-class}} \right] + \ell_{reg} $$

엔트로피 정규화: $\ell_{reg} = -\sum_i^k \min_{i \neq j} \log \|\mathbf{C}_i - \mathbf{C}_j\|_2$

**④ 클래스 붕괴 방지 정규화:**

$$\mathcal{L}_{\rho} = \sum_{j=1}^{k} h_j \cdot \log \frac{h_j}{p_j^{\rho}}$$

여기서 $h_j = 1/k$ (균등 사전분포), $p_j^{\rho}$는 미니배치에서의 평균 소프트맥스 확률

**⑤ 전체 손실 함수:**

$$\mathcal{L} = \mathcal{L}_{ae} + \alpha \left( \mathcal{L}_c + \mathcal{L}_{cc} + \mathcal{L}_{\rho} \right) $$

여기서 $0 \leq \alpha \leq 1$은 훈련 단계에 따라 동적으로 변화

#### 레이블 교정 전략 (Re-Labeling Strategy)

**분류기 예측 레이블 (지수 이동 평균으로 안정화):**

$$\mathbf{y}_i^c = \sum_{\text{last 5 epochs } t} \tau_t [\mathbf{p}_i^c]_t, \quad \tau_t \sim e^{\frac{t-5}{2}} $$

**클러스터링 기반 레이블:**

$$\mathbf{y}_i^{cc} = \text{softmin}_j \left( \|e(\mathbf{x}_i) - \mathbf{C}_j\|_2 \right) $$

**교정된 레이블 (최종 argmax):**

```math
\mathbf{y}_i^* = \underset{}{\text{argmax}} \left[ (1-w)\,\mathbf{y}_i + w\,(\mathbf{y}_i^c + \mathbf{y}_i^{cc}) \right]
```

여기서 $0 \leq w \leq 1$은 훈련 진행도에 따라 동적으로 변화

#### 훈련 다이나믹스 (3단계)

| 단계 | 에포크 범위 | $\alpha$ | $w$ | 목적 |
|---|---|---|---|---|
| **Warm-up** | $0 \to \lambda_{init}$ | $0$ | $0$ | 오토인코더로 임베딩 초기화 |
| **Re-labeling** | $\lambda_{start} \to \lambda_{end}$ | $0 \to 1$ | $0 \to 1$ | 레이블 점진적 교정 |
| **Fine-tuning** | $\lambda_{end} \to 100$ | $1$ | $1$ | 완전 자기지도 학습 |

기본 하이퍼파라미터: $\lambda_{init} = 0$, $\Delta_{start} = 25$, $\Delta_{end} = 30$

---

### 2.3 모델 구조 (네트워크 아키텍처)

**인코더/디코더: CNN 기반 대칭 구조 (4개 ConvBlock)**

각 ConvBlock:
$$y = W * \mathbf{x} + \mathbf{b} \xrightarrow{\text{BatchNorm}} \xrightarrow{\text{ReLU}} \xrightarrow{\text{Dropout}(p=0.2)}$$

| 구성요소 | 상세 |
|---|---|
| Encoder | 4× ConvBlock (128→128→256→256 필터), GlobalAvgPool → 32차원 임베딩 |
| Decoder | 4× TConvBlock (전치 합성곱, 업샘플링) |
| Classifier | Dense(128) → Dense(#classes) |
| 임베딩 차원 | 32 |
| 최적화기 | Adam (lr=0.01, 20에포크마다 절반 감소) |
| 총 훈련 에포크 | 100 |

---

### 2.4 성능 향상

**CHP 전력 추정 결과 ($\mathcal{F}_1$ score):**

| 노이즈 유형 | 노이즈 비율 | CE | SREA | 향상 |
|---|---|---|---|---|
| 없음 | 0% | 0.980 | 0.979 | 유사 |
| Symmetric | 30% | 0.856 | **0.938** | +8.2%p |
| Symmetric | 45% | 0.763 | **0.918** | +15.5%p |
| Flip | 40% | 0.868 | **0.945** | +7.7%p |
| Asymmetric | 30% | 0.895 | 0.919 | +2.4%p |
| Asymmetric | 40% | 0.807 | **0.287** | **크게 저하** |

**UCR 벤치마크 (200 실험 기준):**
- 대칭 노이즈: 62회 유의미하게 우수, 105회 동등, 33회 열세
- 비대칭 노이즈: 86회 유의미하게 우수, 97회 동등, 17회 열세

---

### 2.5 한계점

1. **높은 비대칭 노이즈 취약성:** $\epsilon = 0.40$ 비대칭 노이즈에서 $\mathcal{F}_1 = 0.287$로 급격한 성능 저하 — 워밍업/리레이블링 단계에서 구조적 노이즈가 학습되어 파인튜닝에서 증폭됨

2. **데이터 미공개:** 기업 개인정보 보호 정책으로 CHP 데이터셋을 공개하지 못함

3. **시계열 시간적 의존성 미활용:** 슬라이딩 윈도우 샘플들이 서로 독립으로 처리됨 (인접 윈도우 간 상관관계 무시)

4. **소규모 데이터:** 78일 측정 데이터만 사용 (데이터 부족)

5. **하이퍼파라미터 민감성:** $\Delta_{start}$와 $\Delta_{end}$가 길수록 성능이 향상되지만 최적값은 데이터셋에 따라 달라질 수 있음

---

## 3. 모델의 일반화 성능 향상 가능성 (중점 분석)

### 3.1 일반화를 높이는 설계 요소

#### ① 자기지도 사전학습 (Self-supervised Pretraining)

오토인코더 $\mathcal{L}_{ae}$는 **레이블 정보 없이** 입력 데이터의 구조적 특징을 학습한다. 이는 다음과 같은 이유로 일반화에 기여한다:

$$\mathcal{L}_{ae} = \frac{1}{n} \sum_{i=1}^{n} (\hat{\mathbf{x}}_i - \mathbf{x}_i)^2$$

- 레이블 노이즈에 완전히 독립적인 표현 학습
- 노이즈가 없는 순수한 데이터 분포를 임베딩 공간에서 포착
- 선행 연구[Hendrycks et al., 2019; Huang et al., 2021]에서도 자기지도 학습이 모델 견고성을 향상시킴이 확인됨

#### ② 제약 클러스터링의 앵커 역할

$\mathcal{L}_{cc}$의 핵심 역할은 **자기지도 파인튜닝 단계에서의 붕괴 방지**다. 절제 연구(Table 4)에서 명확히 확인:

| 손실 구성 | 노이즈 없음 | 대칭 30% | 플립 30% |
|---|---|---|---|
| $\mathcal{L}_c$ 만 | 0.472 | 0.355 | 0.405 |
| $\mathcal{L}\_c + \mathcal{L}_{ae}$ | 0.504 | 0.366 | 0.411 |
| $\mathcal{L}\_c + \mathcal{L}_{cc}$ | **0.974** | **0.919** | **0.943** |
| $\mathcal{L}\_c + \mathcal{L}\_{ae} + \mathcal{L}_{cc}$ | **0.980** | **0.930** | **0.957** |

$\mathcal{L}_{cc}$ 없이는 레이블 교정이 **자기 강화적 오류(self-reinforcing error)**에 빠져 일반화 실패

#### ③ 동적 훈련 다이나믹스의 정규화 효과

초기 높은 학습률($lr = 0.01$)과 워밍업 기간의 조합:
- 딥러닝 네트워크는 훈련 초기에 깨끗한 데이터 패턴을 먼저 학습하는 경향이 있음[Zhang et al., 2016; Arpit et al., 2017]
- 이 성질을 활용하여 $w=0$ 기간 동안 노이즈에 오염되기 전에 올바른 임베딩 공간을 형성

$$\alpha(t): 0 \xrightarrow[\text{warm-up}]{} 1, \quad w(t): 0 \xrightarrow[\text{re-labeling}]{} 1$$

#### ④ 모델 불가지론성(Model Agnosticism)

CNN을 기본 블록으로 사용하지만 SREA 프레임워크 자체는 임의의 신경망 구조와 호환:
- RNN, LSTM, Transformer 등으로 교체 가능
- 도메인 특화 아키텍처를 사용하면 해당 도메인에서의 일반화 성능 추가 향상 가능

#### ⑤ 입력 변수 조합의 일반화 영향

입력 변수 절제 연구 결과:

| 입력 | 노이즈 없음 | 비대칭 20% | 비대칭 30% |
|---|---|---|---|
| $T_{amb}$만 | 0.625 | 0.491 | 0.531 |
| $T_{water}$만 | 0.944 | 0.886 | 0.822 |
| $P_{tot}$만 | 0.938 | 0.868 | 0.832 |
| **모두** | **0.978** | **0.941** | **0.921** |

다양한 물리적 신호 통합 → 노이즈에 강건한 다중 관점 학습

### 3.2 일반화 성능의 한계와 개선 가능성

**현재 한계:**
- 슬라이딩 윈도우 샘플 간 **시간적 의존성 무시** → 시계열 특유의 패턴을 활용하지 못함
- 78일이라는 **제한된 훈련 데이터** → 계절적 패턴(자기상관 피크: 140샘플 = 1일) 충분히 포착 어려움
- **검증 세트 없이** 훈련 → 조기 종료(early stopping) 불가 → 최적이 아닐 수 있음

**개선 방향:**
- Transformer 또는 LSTM 인코더 적용으로 시간적 의존성 활용
- 데이터 증강(Data Augmentation) 기법 통합
- 멀티스케일 윈도우 적용

---

## 4. 미래 연구에 미치는 영향 및 고려사항

### 4.1 미래 연구에 미치는 영향

#### ① 산업 AI 분야에 대한 영향

- **실세계 NILM(Non-Intrusive Load Monitoring)** 문제에서 레이블 노이즈를 명시적으로 고려하는 연구 방향을 제시
- 자동 레이블 생성 파이프라인의 신뢰성 문제에 대한 실용적 해결책 제공
- EV 충전 스케줄링, 피크로드 관리 등 에너지 최적화 응용에 직접 활용 가능

#### ② 레이블 노이즈 연구에 대한 영향

- **시계열 분류 + 레이블 노이즈** 문제를 최초로 체계적으로 정의하고 벤치마크화
- 자기지도 + 클러스터링의 결합이 레이블 노이즈에 강건함을 실증
- 노이즈 비율 사전지식 불필요 → 실세계 적용성 향상

#### ③ 멀티태스크 학습 프레임워크에 대한 영향

- 오토인코더를 보조 태스크로 활용하는 패러다임의 유효성 재확인
- 이상 감지(Anomaly Detection), 센서 고장 감지 등 인접 분야로 확장 가능

### 4.2 향후 연구 시 고려할 점

#### 기술적 고려사항

1. **시간적 구조 활용:** Transformer, Temporal Convolutional Network(TCN), 또는 시계열 특화 어텐션 메커니즘 도입으로 슬라이딩 윈도우 간 의존성 포착

2. **더 정교한 레이블 노이즈 모델:** 현재 연구는 $T$ 행렬 기반 단순 노이즈만 고려. 실세계에서는 시간적으로 상관된 노이즈(연속적 센서 오류)를 더 정확히 모델링 필요

3. **비지도 클러스터 수 결정:** 현재 $k$-means 초기화는 클래스 수 $k$를 미리 알아야 함 → 자동 클러스터 수 결정 방법 탐구 필요

4. **불균형 클래스 처리:** 산업 설비에서 CHP 오프 상태가 온 상태보다 훨씬 많을 수 있음 → 클래스 불균형 + 레이블 노이즈의 복합 문제 연구 필요

5. **온라인/증분 학습:** 산업 환경에서는 데이터가 스트리밍으로 계속 유입됨 → 오프라인 배치 학습이 아닌 온라인 레이블 교정 방법 연구

#### 방법론적 고려사항

6. **클린 검증 데이터 확보 전략:** 레이블 교정 품질 평가를 위한 소량의 신뢰 가능한 레이블 데이터 활용 방안 (예: 능동학습과의 결합)

7. **노이즈 비율 추정:** SREA는 $\epsilon$ 사전지식이 불필요하지만, 추정된 $\epsilon$을 활용하면 성능이 더 향상될 수 있음 → 노이즈 비율 자동 추정 방법 통합

8. **재현성 및 데이터 공개:** 산업 데이터의 공개 불가 문제 → 프라이버시 보존 연합학습(Federated Learning)과의 결합 탐구

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의:** 아래 비교는 논문 본문에 직접 인용된 연구와 공개된 정보를 기반으로 하며, 2021년 이후 발표된 일부 연구들은 제가 직접 접근 가능한 범위에서만 서술합니다. 확인되지 않은 내용은 포함하지 않습니다.

### 논문에서 직접 비교한 2020년 이후 연구들

| 방법 | 참고문헌 | 핵심 아이디어 | SREA 대비 차이점 |
|---|---|---|---|
| **DivideMix** | Li et al., arXiv:2002.07394 (2020) | 노이즈 레이블을 반지도학습으로 처리, GMM으로 클린/노이즈 분리 | $\epsilon$ 사전지식 불필요하지만 클린 데이터 검증셋 필요 |
| **SIGUA** | Han et al., ICML 2020 | 노이즈 샘플에 확률적 경사 상승 적용 | **$\epsilon$ 사전지식 필요**, 시계열 미특화 |
| **Self-supervised for semi-supervised TSC** | Jawed et al., PAKDD 2020 | 자기지도 학습으로 반지도 시계열 분류 | 레이블 없는 데이터 가정 (NILM 설정과 다름) |

### 논문 발표(2021) 이후 관련 연구 동향

논문에서 명시적으로 언급된 미래 연구 방향과 일치하는 흐름들이 이후 연구에서 나타났으나, 구체적인 논문 제목과 결과를 확인 없이 서술하는 것은 정확성 측면에서 적절하지 않으므로, **논문 본문에서 확인 가능한 내용만** 기반으로 방향성을 서술합니다:

1. **Transformer 기반 시계열 분류:** 논문은 미래 작업으로 Transformer[Wu et al., 2020] 및 RNN[Karim et al., 2018] 적용을 명시. 이 방향은 시계열의 시간적 의존성 활용 측면에서 SREA를 자연스럽게 확장

2. **이상 감지로의 확장:** 센서 오류 감지 시나리오로의 확장 가능성 명시 (논문 Conclusion 섹션)

3. **Song et al., arXiv:2007.08199 (2020):** "Learning from noisy labels with deep neural networks: A survey" — 노이즈 레이블 학습 전반을 서베이하며 SREA가 속한 연구 흐름을 체계화

---

## 참고 자료

**주요 참고문헌 (논문 본문 직접 인용):**

1. Castellani, A., Schmitt, S., Hammer, B. (2021). *"Estimating the Electrical Power Output of Industrial Devices with End-to-End Time-Series Classification in the Presence of Label Noise."* arXiv:2105.00349v2
2. Arazo et al. (2019). *"Unsupervised label noise modeling and loss correction."* ICML
3. Han et al. (2020). *"SIGUA: Forgetting may make learning with noisy labels more robust."* ICML
4. Li et al. (2020). *"DivideMix: Learning with noisy labels as semi-supervised learning."* arXiv:2002.07394
5. Song et al. (2020). *"Learning from noisy labels with deep neural networks: A survey."* arXiv:2007.08199
6. Fawaz et al. (2019). *"Deep learning for time series classification: a review."* Data Mining and Knowledge Discovery
7. Zhang et al. (2016). *"Understanding deep learning requires rethinking generalization."* arXiv:1611.03530
8. Arpit et al. (2017). *"A closer look at memorization in deep networks."* ICML
9. Jawed et al. (2020). *"Self-supervised learning for semi-supervised time series classification."* PAKDD
10. Hendrycks et al. (2019). *"Using self-supervised learning can improve model robustness and uncertainty."* arXiv:1906.12340
11. Wang et al. (2017). *"Time series classification from scratch with deep neural networks: A strong baseline."* IJCNN
12. Frénay & Verleysen (2013). *"Classification in the presence of label noise: a survey."* IEEE TNNLS
