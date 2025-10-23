# Earthformer: Exploring Space-Time Transformers for Earth System Forecasting

### 1. 핵심 주장과 주요 기여[1]

**Earthformer**는 지구 시스템 예측(날씨 및 기후 예측)을 위한 혁신적인 공간-시간 Transformer 모델입니다. 논문의 핵심 주장은 다음과 같습니다:[1]

기존의 RNN-CNN 조합은 선형적 귀납 편향을 강제하지만, 지구 시스템의 **카오스 특성**으로 인해 이러한 제약이 부적절할 수 있다는 점을 지적합니다. Transformer의 attention 메커니즘은 장거리 의존성과 복잡한 상관관계를 더 잘 포착할 수 있습니다.

주요 기여는:

1. **Cuboid Attention**이라는 제너릭하고 효율적인 공간-시간 attention 블록 제안
2. **전역 벡터(Global Vectors)** 도입으로 지역 cuboid들 간의 통신 메커니즘 구현
3. 합성 데이터셋(MovingMNIST, N-body MNIST)과 실제 데이터셋(SEVIR, ICAR-ENSO)에서 **최첨단 성능** 달성

---

### 2. 문제 정의, 제안 방법 및 모델 구조[1]

#### 문제 정의

지구 시스템 예측은 다음과 같이 공식화됩니다:[1]

관측 시퀀스 $$[X_i]\_{i=1}^{T}, X_i \in \mathbb{R}^{H \times W \times C_{in}} $$ 로부터 K-스텝 선행(ahead) 미래값 $$[Y_{T+i}]\_{i=1}^{K}, Y_{T+i} \in \mathbb{R}^{H \times W \times C_{out}} $$ 를 예측합니다. 여기서 H, W는 공간 해상도입니다.

기존 방법의 한계:
- 데이터-기반 모델은 대규모 지구 관측 데이터를 활용할 수 있음
- 하지만 vanilla Transformer의 $$O(N^2) $$ attention 복잡도는 고차원 데이터에 불가능

#### Cuboid Attention: 제안 방법[1]

**세 가지 단계**로 구성됩니다:

**1) Decompose (분해)**

입력 공간-시간 텐서 $$X \in \mathbb{R}^{T \times H \times W \times C} $$ 를 비중복 cuboid들로 분해합니다:

```math
\{x^{(n)}\} = \text{Decompose}(X, \text{cuboid\_size}, \text{strategy}, \text{shift})
```

여기서:
- `cuboid_size` = $$(b_T, b_H, b_W) $$: 각 cuboid의 크기
- `strategy` ∈ {"local", "dilated"}: 분해 전략
- `shift` = $$(s_T, s_H, s_W) $$: 윈도우 시프트 오프셋

**Local 전략**의 경우 매핑:[1]

$$ i' \leftrightarrow s_T + b_T(n_T-1) + i \mod T $$
$$ j' \leftrightarrow s_H + b_H(n_H-1) + j \mod H $$
$$ k' \leftrightarrow s_W + b_W(n_W-1) + k \mod W $$

**Dilated 전략**의 경우:[1]

$$ i' \leftrightarrow s_T + b_T(i-1) + n_T \mod T $$
$$ j' \leftrightarrow s_H + b_H(j-1) + n_H \mod H $$
$$ k' \leftrightarrow s_W + b_W(k-1) + n_W \mod W $$

**2) Attend (주의)**

각 cuboid 내에서 병렬로 self-attention을 적용합니다:[1]

$$ x^{(n)}_{\text{out}} = \text{Attention}_\Theta(x^{(n)}, x^{(n)}, x^{(n)}), \quad 1 \le n \le N $$

여기서 Attention은 표준 scaled dot-product attention입니다:

$$ \text{Attention}_\theta(Q, K, V) = \text{Softmax}\left(\frac{(W_QQ)(W_KK)^T}{\sqrt{C}}\right)(W_VV) $$

이 단계의 계산 복잡도는:[1]

$$ O\left(\lceil\frac{T}{b_T}\rceil\lceil\frac{H}{b_H}\rceil\lceil\frac{W}{b_W}\rceil(b_T b_H b_W)^2\right) \approx O(THW \cdot b_T b_H b_W) $$

이는 입력 크기에 선형적으로 스케일되므로 vanilla attention의 $$ O(T^2H^2W^2) $$ 보다 훨씬 효율적입니다.

**3) Merge (병합)**

분해의 역연산으로 cuboid들을 원래 입력 형태로 병합합니다:[1]

```math
X_{\text{out}} = \text{Merge}(\{x^{(n)}_{\text{out}}\}_n, \text{cuboid\_size}, \text{strategy}, \text{shift})
```

종합적인 Cuboid Attention 레이어:[1]

```math
X_{\text{out}} = \text{CubAttn}_\Theta(X, \text{cuboid\_size}, \text{strategy}, \text{shift})
```

#### 전역 벡터 (Global Vectors)[1]

Cuboid의 주요 제한은 cuboid들 간의 직접적인 통신 부재입니다. 이를 해결하기 위해 P개의 전역 벡터 $$ G \in \mathbb{R}^{P \times C} $$ 를 도입합니다.

수정된 attention 식:[1]

$$ x^{(n)}_{\text{out}} = \text{Attention}_\Theta\left(x^{(n)}, \text{Cat}(x^{(n)}, G), \text{Cat}(x^{(n)}, G)\right), \quad 1 \le n \le N $$

$$ G_{\text{out}} = \text{Attention}_\Phi(G, \text{Cat}(G, X), \text{Cat}(G, X)) $$

전역 벡터의 추가 복잡도는 $$O(THW \cdot P + P^2) $$ 이며, P가 작으므로(최대 8) 무시할 수 있습니다.[1]

#### 계층적 인코더-디코더 아키텍처[1]

**비-자동회귀 방식**을 채택하여 디코더가 초기 positional embedding으로부터 예측을 직접 생성합니다. 계층적 구조는:

- 각 계층은 D개의 cuboid attention 블록을 스택
- 인코더는 다양한 cuboid 패턴 사용
- 디코더는 "Axial" 패턴 사용
- 공간 해상도 조정을 위해 2D-CNN 기반 다운샘플/업샘플 모듈 포함

***

### 3. 성능 향상 분석[1]

#### 합성 데이터셋 실험

**MovingMNIST** 결과:[1]

| 모델 | MSE ↓ | MAE ↓ | SSIM ↑ |
|------|--------|-------|--------|
| ConvLSTM | 62.04 | 126.9 | 0.8477 |
| PredRNN | 52.07 | 108.9 | 0.8831 |
| E3D-LSTM | 55.31 | 101.6 | 0.8821 |
| Earthformer | **41.79** | **92.78** | **0.8961** |

**N-body MNIST** 결과 (더 도전적인 데이터셋):[1]

| 모델 | MSE ↓ | MAE ↓ | SSIM ↑ |
|------|--------|-------|--------|
| ConvLSTM | 32.15 | 72.64 | 0.8886 |
| PredRNN | 21.76 | 54.32 | 0.9288 |
| Earthformer | **14.82** | **39.93** | **0.9538** |

#### 실제 데이터셋 성능

**SEVIR (강수 예측)** - Critical Success Index (CSI) 메트릭:[1]

| 모델 | CSI-M ↑ | MSE ↓ |
|------|---------|--------|
| ConvLSTM | 0.4185 | 3.753×10⁻³ |
| PredRNN | 0.4080 | 3.901×10⁻³ |
| Rainformer | 0.3661 | 4.027×10⁻³ |
| Earthformer | **0.4419** | **3.696×10⁻³** |

**ICAR-ENSO (엘니뇨 예측)** - 상관 계수 기반 평가:[1]

| 모델 | C-Nino3.4-M ↑ | MSE ↓ |
|------|----------------|--------|
| ConvLSTM | 0.6955 | 2.657×10⁻⁴ |
| E3D-LSTM | 0.7040 | 3.095×10⁻⁴ |
| Rainformer | 0.7106 | 3.043×10⁻⁴ |
| Earthformer | **0.7329** | **2.546×10⁻⁴** |

#### 주요 설계 결정의 영향

**계층적 구조의 중요성**:[1]

비-계층적 구조(깊이 8)는 MSE 48.04를 달성하지만, 계층적 구조(깊이 4,4)는 MSE 46.91로 **더 적은 FLOPS**로 우수한 성능을 보입니다.

**Cuboid 패턴 비교**:[1]

- Axial 패턴: 최고 성능 달성
- 전역 벡터 추가: 모든 패턴에서 일관된 개선 (추가 FLOPS 무시할 수 있음)

---

### 4. 모델의 한계[1]

#### 주요 한계점

1. **확률성 부재**: 모든 가능한 미래의 평균을 예측하여 흐릿한(blurry) 예측 생성[1]

2. **물리적 지식 미포함**: 순수 데이터-기반 모델로 지구 시스템의 물리적 제약을 명시적으로 활용하지 않음[1]

3. **불확실성 모델링 부족**: 지구 시스템 예측에서 불확실성을 평가할 적절한 메트릭 부재[1]

#### 성능 메트릭 간 트레이드오프

비-자동회귀 vs 자동회귀 디코더 비교에서:[1]

- **비-자동회귀 방식**: 더 나은 기술적 점수(MSE, CSI) 달성
- **자동회귀 방식**: 시각적으로 더 선명한 예측 생성

***

### 5. 일반화 성능 향상 가능성[1]

#### N-body MNIST의 의의

논문은 MovingMNIST의 과도한 단순성을 지적하고, **카오스 3체 운동**을 따르는 **N-body MNIST** 데이터셋을 새로이 제안했습니다.[1]

초기 조건의 미세한 변화가 궁극적인 상태에 미치는 영향이 크게 증가하여, 이는 지구 시스템의 카오스 특성을 더 정확히 반영합니다.[1]

#### 장거리 의존성 캡처

Cuboid Attention을 통한 일반화 개선 메커니즘:

1. **다중 decomposition 패턴 스택**: 다양한 공간-시간 상관관계 포착[1]
   - Axial 패턴: 시간, 높이, 너비 축 따로 처리
   - Dilated 전략: 광범위한 시야각 달성

2. **전역 벡터의 역할**: 지역 cuboid들이 시스템 전역 역학 이해[1]
   - 격리된 지역 정보의 한계 극복
   - 시스템 상태 정보 공유

3. **계층적 구조**: 다양한 해상도에서 특성 학습[1]
   - 저해상도에서 고수준 패턴 캡처
   - 고해상도에서 세부 사항 정제

#### 실험을 통한 검증

SEVIR의 높은 공간 해상도(384×384)에서도 일관된 성능 향상을 보여, **스케일 불변성** 특성을 입증합니다.[1]

***

### 6. 미래 연구에 미치는 영향[1]

#### 학문적 영향

1. **Transformer 적용 확대**: 지구 시스템 예측 분야에서 Transformer의 적용 가능성 입증
2. **효율적 공간-시간 attention 설계의 선례**: Cuboid Attention의 제너릭 프레임워크는 다른 분야에 적용 가능
3. **합성 데이터셋의 중요성 강조**: N-body MNIST는 복잡한 동역학 모델링 검증에 유용

#### 향후 연구 고려사항[1]

1. **확률 예측 모델 개발**: 불확실성 정량화를 통한 더 실용적인 예측
   - 변분 추론 기반 확률 모델 통합
   - 생성 모델(VAE, Diffusion Model) 결합 가능성

2. **물리 제약 통합**: 
   - Physics-Informed Neural Networks (PINNs) 와의 통합
   - 편미분 방정식(PDE) 제약 추가
   - 물리 기반 모델과 데이터-기반 모델 앙상블

3. **신경망 아키텍처 탐색(NAS)**: 
   - 현재는 수동으로 cuboid 패턴을 선택
   - 자동화된 NAS로 최적 구조 발견 가능

4. **지표 개선**: 
   - 시각적 품질을 측정할 메트릭 개발 (FID, IS 등)
   - 불확실성 인식 평가 메트릭

***

### 7. 실무적 함의[1]

**기후 과학 분야**:
- 50년 동안 변하지 않은 수치 예보 시스템의 대안 제시
- 페타바이트급 지구 관측 데이터 활용 가능

**응용 분야**:
- 강수 nowcasting: 재해 예방, 자원 관리
- 엘니뇨 예측: 지역 기후 영향 예측
- 재생 에너지 예측: 풍력/태양광 발전 최적화

---

### 결론

**Earthformer**는 공간-시간 Transformer 설계의 혁신적 사례로, Cuboid Attention의 제너릭 프레임워크와 전역 벡터 메커니즘을 통해 효율성과 효과성을 동시에 달성합니다. N-body MNIST 데이터셋의 제안으로 복잡한 동역학 모델링의 중요성을 강조하며, 물리 제약 통합 및 확률 예측으로의 확장이 향후 핵심 과제입니다.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/747e5829-e454-448a-9671-223c2fe1d240/2207.05833v2.pdf)
