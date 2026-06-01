
# Self-paced Ensemble for Highly Imbalanced Massive Data Classification (SPE)

> **논문 정보**
> - **저자**: Zhining Liu, Wei Cao, Zhifeng Gao, Jiang Bian, Hechang Chen, Yi Chang, Tie-Yan Liu
> - **소속**: Jilin University / Microsoft Research Asia
> - **발표**: IEEE ICDE 2020 (arXiv: 1909.03500)
> - **출처**: [arXiv](https://arxiv.org/abs/1909.03500) | [IEEE Xplore](https://ieeexplore.ieee.org/document/9101851/) | [저자 PDF](https://zhiningliu.com/files/ICDE_2020_SPE.pdf)

---

## 1. 핵심 주장과 주요 기여 요약

많은 실제 응용 분야에서 불균형 데이터로부터 분류기를 학습하는 것이 어렵고, 빅데이터 시대에 들어서면서 대규모이지만 극도로 불균형하고 저품질의 데이터셋을 다루는 분류 작업이 늘어나고 있으나, 기존의 대부분 학습 방법들은 이러한 시나리오에서 낮은 성능이나 낮은 계산 효율성을 보인다.

이 문제를 해결하기 위해 클래스 불균형의 본질에 대한 깊은 조사를 수행하였으며, 클래스 간의 비율 불균형뿐만 아니라 데이터의 본질에 내재된 다른 어려움, 특히 노이즈와 클래스 겹침(overlap)이 효과적인 분류기 학습을 방해한다는 점을 밝혀냈다. 이를 고려하여, **언더샘플링을 통한 데이터 난이도(hardness)의 자기 조율(self-paced harmonizing)**로 강력한 앙상블을 생성하는 새로운 불균형 분류 프레임워크를 제안한다.

### 주요 기여 요약

| 기여 항목 | 설명 |
|---|---|
| **새로운 문제 정의** | 클래스 불균형 + 노이즈 + 클래스 겹침을 동시에 고려 |
| **분류 난이도(Hardness) 개념 도입** | 샘플별 난이도 분포를 통한 적응적 샘플링 |
| **SPE 프레임워크** | Self-paced Under-sampling + Ensemble 통합 |
| **계산 효율성** | 대규모 데이터에서도 높은 효율 유지 |
| **범용성** | 모든 표준 분류기(base learner)와 호환 가능 |

---

## 2. 해결 문제 · 제안 방법(수식) · 모델 구조 · 성능 · 한계

### 2-1. 해결하고자 하는 문제

전통적인 분류 알고리즘(C4.5, SVM, 신경망 등)은 불균형 데이터셋에서 불만족스러운 성능을 보이며, 데이터셋이 대규모이고 노이즈가 있을 경우 상황이 더욱 악화된다. 이는 양성과 음성 샘플 간의 균형 잡힌 분포에 대한 부적절한 가정 때문이며, 압도적인 다수 클래스 인스턴스로 인해 소수 클래스가 무시되는 경향이 있다.

특히 이 논문은 세 가지 핵심 문제를 복합적으로 다룬다:

1. **클래스 불균형 (Class Imbalance)**: 다수 클래스 대비 소수 클래스 샘플이 극단적으로 적은 문제
2. **클래스 겹침 (Class Overlap)**: 결정 경계 근처에서 두 클래스 샘플이 뒤섞이는 문제
3. **노이즈 (Noise)**: 저품질 레이블 및 이상치 샘플이 학습을 방해하는 문제

노이즈 샘플은 더 높은 hardness 값을 가질 가능성이 높고, 고 hardness 샘플의 비율은 클래스 겹침의 정도를 나타낸다.

---

### 2-2. 제안 방법 (수식 포함)

#### ① 분류 난이도 함수 (Hardness Function) 정의

$H$를 분류 난이도 함수로 정의하며, $H$는 어떤 "분해 가능한(decomposable)" 오차 함수도 될 수 있다. 즉, 전체 오차는 개별 샘플 오차의 합산으로 계산된다. 예시로는 절대 오차(Absolute Error), 제곱 오차(Brier-score), 교차 엔트로피(Cross Entropy) 등이 있다. 훈련된 분류기 $F$에 대해, $F(x)$는 샘플 $x$가 양성 인스턴스일 출력 확률을 나타낸다.

훈련된 분류기 $F$에 대해, 샘플 $(x, y)$의 분류 난이도는 다음과 같이 정의된다:

$$h(x, y) = H(F(x), y)$$

- **양성 샘플 $(y=1)$인 경우 (절대 오차 기준)**:

$$h(x, 1) = 1 - F(x)$$

- **음성 샘플 $(y=0)$인 경우**:

$$h(x, 0) = F(x)$$

여기서 $F(x) \in [0, 1]$은 분류기가 $x$를 양성으로 예측할 확률이다.

---

#### ② Hardness Harmonization (난이도 조화 샘플링)

"Hardness Harmonization"은 다수 클래스 샘플을 hardness 값에 따라 $k$개의 빈(bin)으로 나누는 과정이며, $k$는 하이퍼파라미터이다. 각 빈은 특정 난이도 수준을 나타내며, 대부분의 인스턴스는 모든 빈에 걸쳐 동일한 전체 hardness 기여도를 유지하면서 균형 잡힌 데이터셋으로 언더샘플링된다.

각 빈 $k$에서 추출할 샘플 수를 수식으로 표현하면:

$$n_k = \frac{|P|}{K} \quad (k = 1, 2, \ldots, K)$$

여기서 $|P|$는 소수 클래스 샘플 수, $K$는 전체 빈의 수이다. 각 빈에서 동일한 수의 다수 클래스 샘플을 추출하여 균형 잡힌 hardness 기여를 보장한다.

---

#### ③ Self-paced Factor $\alpha$ 적용

단순히 데이터셋을 균형 맞추거나 직접 클래스 가중치를 할당하는 대신, 분류 난이도의 분포를 고려하여 SPE는 hardness 분포를 기반으로 가장 정보가 많은 다수 데이터셋을 반복적으로 선택한다. 부스팅과 유사한 직렬 훈련이 언더샘플링과 앙상블 전략을 사용하여 수행되며, 최종적으로 가법 모델(additive model)이 얻어진다.

Self-paced factor $\alpha$를 통해 각 반복(iteration) $t$에서 빈 $k$의 샘플링 가중치는 다음과 같이 조정된다:

$$w_k^{(t)} \propto \left(\frac{k}{K}\right)^{\alpha}$$

여기서:
- $\alpha = 0$: 모든 빈에서 균등 샘플링 (hardness 무관)
- $\alpha \to \infty$: 가장 어려운 샘플(고 hardness)에서만 집중 샘플링
- $\alpha$ 가 점차 증가: 학습이 진행될수록 더 어려운 샘플에 집중

서브피겨 (a)는 전체 다수 인스턴스의 분포이고, (b)(c)(d)는 $\alpha = 0$, $\alpha = 0.1$, $\alpha \rightarrow \infty$일 때 샘플링된 서브셋의 분포를 보여준다.

---

#### ④ SPE 앙상블 예측

$T$번의 반복을 통해 $T$개의 약한 분류기 $F_1, F_2, \ldots, F_T$를 학습하고, 최종 예측은 평균 집계를 통해 수행된다:

$$\hat{F}(x) = \frac{1}{T}\sum_{t=1}^{T} F_t(x)$$

최종 레이블 예측:

$$\hat{y} = \mathbb{1}\left[\hat{F}(x) \geq 0.5\right]$$

---

### 2-3. 모델 구조 (파이프라인)

```
┌─────────────────────────────────────────────────────────────┐
│                  SPE Training Pipeline                       │
│                                                             │
│  ┌──────────┐   Step 1: 이전 분류기 F(t-1)로 hardness 계산  │
│  │ 전체 데이터│─────────────────────────────────────────►   │
│  └──────────┘                                               │
│       │                                                     │
│       ▼                                                     │
│  ┌──────────────────────┐                                   │
│  │ Hardness Distribution │  다수 클래스 샘플을 K개 빈으로 분류 │
│  │ (K-bin partitioning)  │                                  │
│  └──────────────────────┘                                   │
│       │                                                     │
│       ▼                                                     │
│  ┌────────────────────────────┐                            │
│  │ Self-paced Under-sampling  │  α에 따라 각 빈에서 샘플 추출  │
│  │ (α 조절 → 점진적 집중)      │                            │
│  └────────────────────────────┘                            │
│       │                                                     │
│       ▼                                                     │
│  ┌──────────────────┐                                       │
│  │ Balanced Subset  │  |P| = |N_sampled|                   │
│  └──────────────────┘                                       │
│       │                                                     │
│       ▼                                                     │
│  ┌──────────────────┐                                       │
│  │ Base Classifier  │  임의의 표준 분류기 (DT, RF, SVM 등)   │
│  │ F_t 훈련         │                                       │
│  └──────────────────┘                                       │
│       │      반복 (t = 1, ..., T)                           │
│       ▼                                                     │
│  ┌──────────────────────┐                                   │
│  │ 앙상블 예측 (평균 집계)│  최종 출력                        │
│  └──────────────────────┘                                   │
└─────────────────────────────────────────────────────────────┘
```

SPE는 대규모 불균형 데이터 분류를 위한 학습 프레임워크로, 어떤 표준 분류기(canonical classifier)의 성능도 향상시키는 데 사용할 수 있다.

---

### 2-4. 성능 향상

광범위한 실험을 통해, 이 프레임워크는 매우 계산 효율적이면서도 클래스가 심하게 겹치고 분포가 극단적으로 편향된 경우에서도 강력한 성능을 발휘할 수 있음이 입증되었다.

광범위한 실험 테스트에서 겹치는 클래스와 편향된 분포를 처리하는 데 있어 계산 효율성을 유지하면서 유망한 결과를 보여준다. 자기 조율 앙상블 방법은 대규모 불균형 분류 문제에서 높은 불균형 비율, 클래스 겹침, 노이즈 존재 등의 도전을 해결한다.

#### 주요 성능 지표 비교 (논문 기준)

| 비교 대상 | 특징 | SPE 대비 |
|---|---|---|
| **SMOTE** | Over-sampling 기반 | 노이즈/겹침에 취약, 계산 비용 높음 |
| **EasyEnsemble** | 랜덤 언더샘플링 앙상블 | Hardness 미고려 |
| **BalanceCascade** | 계단식 언더샘플링 | 과도한 정보 손실 위험 |
| **AdaBoost** | 가중치 부스팅 | 노이즈에 민감 |
| **SPE** | Hardness 기반 적응적 언더샘플링 | **AUROC, F1, G-mean에서 우수** |

---

### 2-5. 한계점

대부분의 기존 언더샘플링 앙상블 방법은 부적절한 샘플링 전략에 취약하여, 다수 클래스의 유용한 정보 손실을 초래하고 모델의 일반화 능력에 영향을 줄 수 있다.

SPE와 EASE는 여전히 다음과 같은 한계를 가지고 있다: 두 알고리즘 모두 언더샘플링으로 인한 정보 손실을 완화하기 위해 빈 전략을 채택하지만, 낮은 예측값을 가진 빈에는 노이즈와 겹침 샘플이 더 많이 포함될 수 있다.

또한:
- **하이퍼파라미터 민감성**: $K$ (빈 수)와 $\alpha$ (self-paced factor), $T$ (앙상블 크기) 설정이 성능에 민감
- **정적 hardness 분포**: 각 iteration에서 hardness 분포가 동적으로 변화하지만, 빈 경계(bin boundary)는 고정적
- **이진 분류 중심**: 다중 클래스 불균형 문제로의 직접 확장이 제한적
- **Over-sampling과의 미통합**: 언더샘플링만 사용하므로 소수 클래스 샘플 수가 매우 적을 경우 취약

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 일반화에 기여하는 메커니즘

언더샘플링 전략은 self-paced 절차에 의해 제어되며, 이는 SPE가 점진적으로 더 어려운 데이터 샘플에 집중하면서도 다수 클래스 샘플의 정보를 유지하여 **과적합(over-fitting)을 방지**할 수 있게 한다.

단순히 양성과 음성 데이터를 균형화하거나 인스턴스 가중치를 사용하는 대신, 분류 난이도의 데이터셋 분포를 고려하고 분포에 따라 가장 정보가 많은 다수 데이터 샘플을 반복적으로 선택한다. 자기 조율 메커니즘이 언더샘플링 과정을 제어하며, 이 자기 조율 절차는 시스템이 단순한 샘플 분포에 대한 인식을 유지하면서 더 복잡한 데이터 샘플에 집중하여 **과적합을 방지**할 수 있게 한다.

추가적인 이점으로, hardness 분포는 다양한 모델에 본질적으로 적응 가능하며, hardness 분포가 성능 향상을 위한 재샘플링 접근 방식을 안내할 수 있다.

### 3-2. 일반화 성능 향상의 핵심 요인 분석

```
일반화 성능 향상 경로
─────────────────────
[Hardness 분포 기반 샘플링]
        │
        ├──► 경계 샘플(borderline) 집중 → 결정 경계 학습 강화
        │
        ├──► 쉬운 샘플 비중 축소 → 과적합 방지
        │
        ├──► 앙상블 다양성(Diversity) 확보 → 분산(Variance) 감소
        │
        └──► 노이즈 비중 자연스러운 조절 → 편향(Bias) 조절
```

노이즈와 평범한 샘플의 영향을 줄이고, 예상하는 경계선(borderline) 샘플의 중요도를 높이는 언더샘플링 메커니즘 설계를 목표로 한다.

### 3-3. 일반화 관련 수식적 해석

$t$ 번째 반복에서 앙상블 분류기의 기대 오차는 **편향-분산 분해(Bias-Variance Decomposition)**로 표현 가능하다:

$$\mathbb{E}[\text{Error}] = \text{Bias}^2 + \text{Variance} + \text{Noise}$$

SPE의 기여:
- **Bias 감소**: Hardness 기반 샘플링으로 결정 경계 근처 샘플 학습 강화
- **Variance 감소**: 앙상블 평균화 + 다양한 서브샘플 사용

$$\text{Var}\left[\hat{F}(x)\right] = \frac{1}{T^2} \sum_{t=1}^{T} \text{Var}[F_t(x)] + \frac{2}{T^2}\sum_{i < j}\text{Cov}[F_i(x), F_j(x)]$$

Self-paced 샘플링으로 매 iteration마다 다른 서브셋을 구성하므로, $F_t$들 간의 공분산(Covariance)을 낮춰 앙상블 분산을 효과적으로 줄인다.

---

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 4-1. 후속 연구에 미친 영향

#### (1) DAPS: Dynamic Self-paced Sampling Ensemble (2022)

DAPS(DynAmic self-Paced sampling enSemble)는 고불균형 겹침 분류를 위한 효과적인 메타 프레임워크로, (1) 정보가 풍부한 인스턴스의 활용을 극대화하고 심각한 정보 손실을 방지하기 위한 합리적이고 효과적인 샘플링, (2) 노이즈 데이터 문제를 해결하기 위한 적절한 인스턴스 가중치 할당을 결합한다.

또한 의사결정 트리, 랜덤 포레스트 등 대부분의 표준 분류기를 DAPS에 통합할 수 있으며, 합성 데이터셋과 실제 데이터셋 모두에서의 포괄적인 실험 결과는 DAPS 모델이 다양한 방법과 비교하여 F1-score에서 상당한 개선을 얻을 수 있음을 보여준다.

#### (2) DSPE: Dynamic Self-paced Ensemble (2026)

DSPE는 메타 학습 기반 앙상블(Self-Paced Ensemble; SPE)과 불균형 데이터 처리를 위한 동적 앙상블 선택 기법을 통합한 새로운 휴리스틱 알고리즘이다. DSPE는 (1) 거리 기반 메트릭(KNN)을 사용하는 오버샘플링/언더샘플링 대신 메타 학습을 통해 다수 클래스의 샘플링 전략을 학습하고, (2) 미지 테스트 샘플의 로컬 영역에서의 분류기 역량을 기반으로 한 맞춤형 앙상블 선택을 통해 불균형 인식 학습을 우수하게 처리한다.

45개의 대표적인 불균형 데이터셋에 대한 실험 분석에 따르면, DSPE는 SPE에 비해 유의미한 성능 향상을 달성하며, 서로 다른 앙상블 크기에서도 알고리즘의 안정성이 증명된다.

#### (3) DSUE: Dynamic Self-paced Undersampling Ensemble (2025)

새로운 동적 자기 조율 언더샘플링 앙상블(DSUE) 방법을 제안하며, 이 방법은 먼저 적응형 자기 조율 언더샘플링을 구현하여 각 기본 분류기에 대한 균형 잡힌 데이터 서브셋을 생성하기 위해 난이도 분포를 기반으로 가장 정보가 풍부한 다수 클래스 샘플을 선택한다. 그런 다음 샘플 가중치 전략을 사용하여 클래스 겹침 영역의 중요 샘플을 식별하여 모델의 예측 성능을 향상시키고 노이즈 데이터의 영향을 줄인다.

#### (4) 의료 · 도메인 특화 응용

hardness 분포의 개념을 확립하고 자기 조율 앙상블을 대규모 불균형 분류의 혁신적인 학습 패러다임으로 제시하며, 기존 학습 알고리즘의 불균형 데이터 성능을 향상시키고 미래 응용 분야에서 더 나은 결과를 제공할 수 있다.

---

### 4-2. 후속 연구에서 고려할 점

#### ① 동적 하이퍼파라미터 자동화
- 현재 $K$, $\alpha$, $T$는 수동 설정이 필요하므로, **AutoML 또는 Bayesian Optimization** 기반 자동 튜닝 연구가 필요하다.
- $\alpha$를 반복마다 자동으로 스케줄링하는 **적응형 학습률 스케줄** 설계가 유망하다.

#### ② 다중 클래스 불균형 확장

$$\text{Multi-class Hardness: } h(x, y) = 1 - F_y(x) \quad \text{where } y \in \{1, 2, \ldots, C\}$$

현재 이진 분류 중심의 프레임워크를 **다중 클래스 불균형** 문제로 확장하는 연구가 요구된다.

#### ③ Over-sampling과의 하이브리드 통합
- Hardness 기반 언더샘플링과 **SMOTE 계열 오버샘플링**을 결합하여, 소수 클래스 정보 손실 최소화와 다수 클래스 정보 보존을 동시에 달성하는 하이브리드 접근이 필요하다.

#### ④ 딥러닝과의 통합
- SPE의 hardness 계산을 **신경망의 손실(loss)** 기반으로 확장하여, **딥러닝 기반 불균형 학습** 프레임워크로 발전시킬 수 있다:

$$h_{\text{DL}}(x, y) = \mathcal{L}_{\text{CE}}(f_\theta(x), y) = -y\log f_\theta(x) - (1-y)\log(1-f_\theta(x))$$

#### ⑤ 데이터 스트림 및 개념 드리프트 대응
언더샘플링 기반 앙상블 방법은 이러한 작업을 해결하는 가장 효과적인 접근 방식 중 하나로 간주되지만, 대부분의 기존 방법은 부적절한 샘플링 전략으로 인해 다수 클래스의 유용한 정보를 손실하여 모델의 일반화 능력에 영향을 줄 수 있다. 이를 해결하기 위한 **온라인 학습 환경**에서의 SPE 변형 연구가 활발히 진행 중이다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구명 | 연도 | 학술지/학회 | SPE 대비 개선점 |
|---|---|---|---|
| **DAPS** (Zhou et al.) | 2022 | *Data Mining and Knowledge Discovery* | 동적 샘플링 + 인스턴스 가중치 |
| **EASE** (Ren et al.) | 2022 | *Knowledge-Based Systems* | 하이퍼파라미터 불확실성 개선, 분류기 가중치 부여 |
| **DSPE** (2026) | 2026 | *Pattern Analysis and Applications* | 메타 학습 기반 샘플링 전략 + 동적 앙상블 선택 |
| **DSUE** (2025) | 2025 | *The Journal of Supercomputing* | 적응형 자기 조율 언더샘플링 + 샘플 가중치 전략 |
| **ASE** (Liang et al.) | 2024 | *Expert Systems with Applications* | 이상치 점수 기반 앙상블 |

SPE 알고리즘의 개선 버전인 Equalization Ensemble(EASE)은 SPE의 불확실한 하이퍼파라미터 문제를 해결하며, 소수 클래스 샘플의 총 수를 빈닝의 하이퍼파라미터로 사용한다. 이는 데이터 균형을 달성하기 위해 각 빈에서 하나의 샘플만 선택됨을 의미하며, 샘플 빈의 균일성을 개선하고 SPE 자체의 무작위성을 줄인다. 또한 EASE는 G-mean 메트릭을 기반으로 각 분류기에 서로 다른 가중치를 할당하여 최종 결과에 대한 높은 위양성률 분류기의 영향을 완화한다.

---

## 📚 참고 자료

1. **Liu, Z., Cao, W., Gao, Z., Bian, J., Chen, H., Chang, Y., & Liu, T.-Y.** (2020). *Self-paced Ensemble for Highly Imbalanced Massive Data Classification*. **IEEE ICDE 2020**, pp. 841–852. → [arXiv:1909.03500](https://arxiv.org/abs/1909.03500) | [IEEE Xplore](https://ieeexplore.ieee.org/document/9101851/) | [저자 PDF](https://zhiningliu.com/files/ICDE_2020_SPE.pdf)

2. **Zhou, F., Gao, S., Ni, L., Pavlovski, M., Dong, Q., Obradovic, Z., & Qian, W.** (2022). *Dynamic Self-paced Sampling Ensemble for Highly Imbalanced and Class-overlapped Data Classification*. **Data Mining and Knowledge Discovery**, 36(5), 1601–1622. → [Springer](https://link.springer.com/article/10.1007/s10618-022-00838-z) | [Semantic Scholar](https://www.semanticscholar.org/paper/Dynamic-self-paced-sampling-ensemble-for-highly-and-Zhou-Gao/209e0da163bff49eb4031722f762b5cb7f89b252)

3. **Ren, J., Wang, Y., Mao, M. et al.** (2022). *Equalization Ensemble for Large Scale Highly Imbalanced Data Classification*. **Knowledge-Based Systems**, 242, 108295.

4. **Liang, X., Gao, Y., & Xu, S.** (2024). *ASE: Anomaly Scoring Based Ensemble Learning for Highly Imbalanced Datasets*. **Expert Systems with Applications**, 238, 122049.

5. **Dynamic self-paced ensemble for imbalance-aware learning**. (2026). *Pattern Analysis and Applications*, 29, 67. → [Springer](https://link.springer.com/article/10.1007/s10044-026-01645-8)

6. **Dynamic self-paced undersampling ensemble for imbalanced classification**. (2025). *The Journal of Supercomputing*. → [Springer](https://link.springer.com/article/10.1007/s11227-025-07633-9)

7. **Self-paced ensemble and big data identification: a classification of substantial imbalance computational analysis**. (2023). *The Journal of Supercomputing*. → [Springer](https://link.springer.com/article/10.1007/s11227-023-05828-6)

8. **Two-step ensemble under-sampling algorithm for massive imbalanced data classification**. (2024). *Information Sciences*. → [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0020025524002640)

9. **Papers With Code - SPE**: [https://paperswithcode.com/paper/training-effective-ensemble-on-imbalanced](https://paperswithcode.com/paper/training-effective-ensemble-on-imbalanced)

10. **공식 GitHub 코드**: [https://github.com/ZhiningLiu1998/self-paced-ensemble](https://github.com/ZhiningLiu1998/self-paced-ensemble)
