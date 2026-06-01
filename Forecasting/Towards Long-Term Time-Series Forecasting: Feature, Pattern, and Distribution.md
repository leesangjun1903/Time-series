
# Towards Long-Term Time-Series Forecasting: Feature, Pattern, and Distribution

> **출처 / 참고자료**
> - **논문 원문**: Yan Li et al., "Towards Long-Term Time-Series Forecasting: Feature, Pattern, and Distribution," *IEEE ICDE 2023*, arXiv:2301.02068
> - **IEEE Xplore**: https://ieeexplore.ieee.org/document/10184599/
> - **arXiv**: https://arxiv.org/abs/2301.02068
> - **ResearchGate**: https://www.researchgate.net/publication/366902704
> - **Semantic Scholar**: https://www.semanticscholar.org/paper/892397050de6f68c6ea11e9ed3fd091e42aa34f4
> - **EmergentMind (Conformer 분석)**: https://www.emergentmind.com/topics/conformer-conditional-transformer
> - 비교 연구: PatchTST (arXiv:2211.14730, ICLR 2023), Autoformer (NeurIPS 2021), Informer (AAAI 2021), FEDformer (ICML 2022)

---

## 1. 핵심 주장 및 주요 기여 요약

장기 시계열 예측(Long-Term Time-Series Forecasting, LTTF)은 풍력 발전 공급 계획 등 다양한 응용 분야에서 중요한 수요로 부상하고 있다.

Transformer 모델은 높은 계산 비용의 Self-Attention 메커니즘을 통해 높은 예측 능력을 제공하지만, Point-wise Self-Attention의 희소성(Sparsity)을 유도해 복잡도를 낮추면 정보 활용이 제한되어 복잡한 의존성을 포괄적으로 탐색하지 못한다는 문제가 있다.

이 논문의 핵심 주장과 기여는 다음과 같이 세 가지로 요약된다:

이 문제를 해결하기 위해 **Conformer**라는 효율적인 Transformer 기반 모델을 제안하며, 기존 LTTF 방법과 세 가지 측면에서 차별화된다:
(i) 슬라이딩 윈도우 어텐션(Sliding-Window Attention)과 SIRN(Stationary and Instant Recurrent Network) 위에 선형 복잡도를 유지하면서도 정보 활용을 희생하지 않는 인코더-디코더 구조,
(ii) 출력 추론 시 Normalizing Flow로부터 파생된 모듈로 정보 활용을 향상시키는 방법.

구체적으로 Conformer는 다변수 상관관계 모델링(Multivariate Correlation Modeling)과 다중 스케일 동역학 추출(Multi-scale Dynamics Extraction)을 통해 입력 시계열을 임베딩하고, 슬라이딩 윈도우 어텐션과 SIRN을 통해 시리즈 수준의 시간적 의존성을 완전히 증류하며, Normalizing Flow 프레임워크를 통해 SIRN의 잠재 상태를 흡수하여 기저 분포를 학습한다.

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능, 한계

### 2-1. 해결하고자 하는 문제

LTTF 분야에서 기존 Transformer 계열 모델들은 다음과 같은 딜레마를 가진다:

| 문제 | 설명 |
|------|------|
| **복잡도 문제** | 표준 Self-Attention의 $O(L^2)$ 복잡도로 긴 시퀀스 처리 불가 |
| **정보 손실 문제** | Sparse Attention으로 복잡도를 줄이면 정보 활용도 저하 |
| **분포 학습 부재** | 시계열 데이터의 기저 분포를 명시적으로 모델링하지 않음 |
| **다중 스케일 패턴 미포착** | 정상성(Stationary) 패턴과 순간적(Instant) 변화를 동시에 포착하지 못함 |

---

### 2-2. 제안 방법 및 수식

#### ① Feature Extraction: 다변수 상관 모델링 + 다중 스케일 임베딩

입력 시계열 $\mathbf{X} \in \mathbb{R}^{L \times D}$ ($L$: 시퀀스 길이, $D$: 변수 수)에 대해, 다변수 간 상관관계를 캡처하고 다중 스케일 특징을 추출한다:

$$\mathbf{Z}_0 = \text{MultiScaleEmbed}(\mathbf{X}) + \text{CorrEmbed}(\mathbf{X})$$

여기서 `MultiScaleEmbed`는 Fourier Transform을 활용한 다중 주파수 샘플링을 포함하며, `CorrEmbed`는 변수 간 상관 구조를 반영한 임베딩이다.

---

#### ② Pattern Modeling: Sliding-Window Attention + SIRN

**슬라이딩 윈도우 어텐션**은 시퀀스를 윈도우 크기 $w$의 로컬 구간으로 나누어 어텐션을 수행함으로써 복잡도를 $O(L^2)$에서 $O(L \cdot w)$로 낮춘다:

$$\text{SWAttn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\!\left(\frac{\mathbf{Q}_i \mathbf{K}_i^{\top}}{\sqrt{d_k}}\right)\mathbf{V}_i, \quad i \in \text{local window}$$

Conformer는 Fourier Transform, 다중 주파수 시퀀스 샘플링, 재귀적 추세-계절 분해 전략을 통합하며, RNN 기반 SIRN 모듈이 전역 정보를 모델링하고 슬라이딩 윈도우 어텐션의 선형 복잡도로 인한 적합 능력 손실을 보완하기 위해 국소 정보와 결합한다.

**SIRN (Stationary and Instant Recurrent Network)**:

$$\mathbf{h}_t^{s} = f_s(\mathbf{h}_{t-1}^{s}, \mathbf{z}_t), \quad \mathbf{h}_t^{i} = f_i(\mathbf{z}_t)$$

$$\mathbf{h}_t = \mathbf{h}_t^{s} \oplus \mathbf{h}_t^{i}$$

- $\mathbf{h}_t^{s}$: 정상적(Stationary) 패턴을 추적하는 순환 상태
- $\mathbf{h}_t^{i}$: 순간적(Instant) 변화를 포착하는 상태
- $\oplus$: 결합 연산 (concat 또는 add)

---

#### ③ Distribution Learning: Normalizing Flow

출력 분포를 명시적으로 모델링하기 위해, Normalizing Flow를 사용한다. 잠재 변수 $\mathbf{z}$를 표준 정규분포로부터 샘플링하여 가역 변환 $f_\theta$를 통해 예측 분포를 생성한다:

$$\mathbf{y} = f_\theta(\mathbf{z}; \mathbf{h}), \quad \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

로그 우도는 변수 변환 공식에 의해:

$$\log p_\theta(\mathbf{y} \mid \mathbf{h}) = \log p(\mathbf{z}) + \log \left|\det \frac{\partial f_\theta^{-1}}{\partial \mathbf{y}}\right|$$

학습 목표(Loss):

$$\mathcal{L} = -\frac{1}{N}\sum_{n=1}^{N} \log p_\theta(\mathbf{y}^{(n)} \mid \mathbf{h}^{(n)})$$

이를 통해 점 예측(Point Forecast)뿐만 아니라 **불확실성 구간(Uncertainty Band)**도 추정 가능하다.

---

### 2-3. 모델 구조

```
입력 시계열 X
    │
    ▼
[Feature Extraction Layer]
  - 다변수 상관 임베딩 (Multivariate Correlation Embedding)
  - 다중 스케일 동역학 추출 (Multi-scale Dynamics Extraction, Fourier 기반)
    │
    ▼
[Encoder: Sliding-Window Attention + SIRN]
  - 슬라이딩 윈도우 어텐션 → 로컬 패턴 포착, O(L·w) 복잡도
  - SIRN → 정상성/순간 패턴 분리, 글로벌 의존성 보완
    │
    ▼
[Decoder: Normalizing Flow 기반 분포 추론]
  - SIRN 잠재 상태를 흡수하여 기저 분포 학습
  - 불확실성 추정 및 샘플링 가능한 예측 생성
    │
    ▼
출력: 예측값 ŷ + 불확실성 구간
```

---

### 2-4. 성능 향상

Conformer의 슬라이딩 윈도우 어텐션, SIRN, Normalizing Flow의 조합은 Informer, Autoformer 등의 최신 기준 모델 대비 MSE를 최대 40%까지 감소시키며, 시퀀스 길이에 대한 선형 확장성과 주기적·정상적 시간 패턴에 대한 향상된 활용을 지원한다.

768 스텝까지의 예측 지평선에서 최신 기준 모델들(Informer, Autoformer 등) 대비 MSE 최대 40% 감소를 달성하며, 보정된 불확실성 구간도 제공한다.

---

### 2-5. 한계점

논문에서 확인 가능한 한계:
1. **모델 복잡도**: SIRN + Normalizing Flow + 슬라이딩 윈도우 어텐션을 결합하여 구현 및 튜닝 복잡도가 높다.
2. **채널 독립성 미지원**: 다변수 간 상관 모델링을 시도하지만, 이후 PatchTST와 같이 채널 독립(Channel-Independence) 전략을 채택하지 않아 고차원 데이터에서의 일반화 문제가 남는다.
3. **Normalizing Flow의 가역성 제약**: 가역 변환 구조는 모델 표현력에 제약을 가할 수 있다.
4. **벤치마크 범위**: ETT, Weather 등 표준 벤치마크 위주 평가로 다양한 도메인 일반화 검증이 부족하다.

---

## 3. 모델의 일반화 성능 향상 가능성

Conformer가 일반화 성능 향상에 기여하는 메커니즘은 세 가지이다:

### ① 분포 학습을 통한 일반화

Normalizing Flow 기반 분포 추론은 단순 점 예측이 아닌, 데이터의 기저 확률 분포를 직접 학습한다. 이는 훈련 데이터와 테스트 데이터 간 분포 차이(Distribution Shift)가 있을 때도 더 강인한 예측을 가능하게 한다. 수식으로는:

$$p_\theta(\mathbf{y} \mid \mathbf{h}) \approx p_{\text{true}}(\mathbf{y} \mid \text{context})$$

이처럼 분포 자체를 목표로 학습함으로써, 새로운 시나리오에서도 통계적으로 일관된 예측을 생성할 수 있다.

### ② SIRN의 정상성 분리를 통한 일반화

시계열을 정상적(Stationary) 성분과 순간적(Instant) 성분으로 분리함으로써, 모델이 도메인에 무관한 시계열의 구조적 특성을 학습한다. 이는 추세-계절성 분해(Trend-Seasonality Decomposition)와 유사하게 작동하여, 다양한 도메인(에너지, 날씨, 교통 등)에 걸친 일반화를 지원한다.

### ③ 다중 스케일 특징 추출의 범용성

Conformer는 다변수 상관 모델링과 다중 스케일 동역학 추출로 입력 시계열을 임베딩하여 단일 스케일 표현의 한계를 극복한다. 장·단기 패턴을 동시에 포착하는 다중 스케일 접근법은 다양한 예측 지평선(Horizon)과 도메인에 걸쳐 더 안정적인 성능을 제공한다.

### ④ 불확실성 정량화와 일반화

Conformer는 보정된 불확실성 구간(Calibrated Uncertainty Bands)을 제공한다. 이는 일반화 성능의 직접적인 척도로, 모델이 자신이 모르는 것(Out-of-Distribution 입력)에 대해 더 넓은 불확실성 구간을 형성하여 실용적 신뢰도를 높인다.

---

## 4. 연구에 미치는 영향 및 앞으로 고려할 점

### 4-1. 연구에 미치는 영향

**① Transformer + 분포 학습의 결합 방향 제시**
Normalizing Flow를 Transformer 기반 시계열 예측에 통합한 시도는, 이후 확률적 시계열 예측(Probabilistic Forecasting) 연구의 기반이 된다. 이는 단순 MSE 최소화를 넘어 불확실성 정량화를 LTTF의 핵심 목표로 격상시켰다.

**② 선형 복잡도 + 정보 보전의 균형 설계**
슬라이딩 윈도우 어텐션($O(L \cdot w)$ 복잡도)과 SIRN의 인터리빙 구조는 선형 확장성과 주기적·정상적 시간 패턴의 향상된 활용을 지원하며, 이후 효율적인 Transformer 설계의 방향을 제시한다.

**③ 다변수 의존성 모델링의 중요성 부각**
기존 Informer, Autoformer 등이 주로 시간 축 의존성에 집중한 반면, Conformer는 변수 간 상관 모델링을 명시적으로 포함하여 이후 시공간 의존성 모델링 연구를 촉진한다.

---

### 4-2. 2020년 이후 관련 최신 연구 비교 분석

| 모델 | 발표 | 핵심 아이디어 | 복잡도 | 불확실성 | 특이점 |
|------|------|--------------|--------|----------|--------|
| **Informer** | AAAI 2021 | ProbSparse Attention | $O(L \log L)$ | ❌ | 최초 대규모 LTTF Benchmark 제시 |
| **Autoformer** | NeurIPS 2021 | Auto-Correlation + 분해 | $O(L \log L)$ | ❌ | 계절-추세 분해 내재화 |
| **FEDformer** | ICML 2022 | 주파수 도메인 어텐션 | $O(L)$ | ❌ | Fourier/Wavelet 기반 희소 표현 |
| **Conformer (본 논문)** | ICDE 2023 | SWAttn+SIRN+NFlow | $O(L \cdot w)$ | ✅ | 분포 학습, 불확실성 정량화 |
| **PatchTST** | ICLR 2023 | 패치 토큰 + 채널 독립 | $O(L/P)^2$ | ❌ | 국소 시맨틱 보존, SSL 가능 |

PatchTST의 패치 설계는 국소적 시맨틱 정보를 임베딩에 보존하고, 동일한 Look-back 윈도우 대비 어텐션 맵의 계산 및 메모리 사용량을 이차적으로 감소시키며, 더 긴 히스토리를 참조할 수 있게 하여 장기 예측 정확도를 크게 향상시킨다.

PatchTST는 자기지도 사전학습(Self-supervised Pre-training) 태스크에도 적용 가능하며, 대규모 데이터셋에서 지도학습을 능가하는 파인튜닝 성능을 달성한다.

분포 이동(Distribution Shift) 문제를 해결하기 위한 TFPS 아키텍처는 시간 및 주파수 도메인 모두에서 특징을 추출하는 이중 도메인 인코더(DDE)와, 서브스페이스 클러스터링 방법을 사용하여 패치 전반에 걸쳐 동적으로 패턴을 식별하는 Pattern Identifier(PI)를 포함한다.

---

### 4-3. 앞으로 연구 시 고려할 점

#### 1) 채널 독립성 vs. 채널 의존성 전략 선택
시계열을 하위 시리즈 수준의 패치로 분할하여 Transformer의 입력 토큰으로 사용하고, 각 채널이 모든 시리즈에 걸쳐 동일한 임베딩과 Transformer 가중치를 공유하는 채널 독립성 전략은 일반화에 유리하나, 변수 간 상관을 포착하지 못한다. 연구 목적에 따라 두 접근법 간 균형을 신중히 설계해야 한다.

#### 2) 분포 이동(Distribution Shift) 대응
실제 환경에서 시계열 데이터는 비정상(Non-stationary)하며 분포가 지속적으로 변화한다. 서브스페이스 클러스터링을 통한 동적 패턴 식별 방법은 비선형 클러스터 경계를 효과적으로 처리하고 패치를 해당 클러스터에 정확하게 할당할 수 있다. 이러한 적응형 패턴 식별 접근법을 Conformer의 구조에 결합하는 연구가 필요하다.

#### 3) 계산 효율성과 성능 간 균형
복잡하게 설계된 Transformer 기반 모델에도 불구하고, 매우 단순한 선형 모델이 다양한 공통 벤치마크에서 이전 모델 전체를 능가할 수 있다는 사실이 지적된 바 있다. 따라서 모델 복잡도와 귀납적 편향(Inductive Bias) 사이의 균형을 면밀히 검토해야 한다.

#### 4) 불확실성 정량화의 실용성 강화
Normalizing Flow 기반 불확실성 추정을 넘어, Conformal Prediction, Bayesian 추론 등 보정된 불확실성 정량화 방법과의 결합을 고려해야 한다.

#### 5) Foundation Model 관점의 사전학습
자기지도 사전학습 태스크 적용 및 파인튜닝 성능이 유망하게 검증된 만큼, Conformer의 구조를 대규모 시계열 Foundation Model (예: TimesFM, Chronos)에 통합하는 연구 방향이 주목받고 있다.

---

> ⚠️ **정확도 주의사항**: 본 답변에서 SIRN의 세부 수식, Normalizing Flow의 구체적 변환 구조, 실험 세부 수치 등 논문 원문의 일부 세부 사항은 공개된 초록 및 메타데이터를 기반으로 추론된 내용이 포함되어 있습니다. 정확한 수식 및 실험 결과는 **arXiv:2301.02068** 또는 **IEEE Xplore(DOI: 10.1109/ICDE55515.2023.00233)** 원문을 직접 참조하시기를 강력히 권장합니다.
