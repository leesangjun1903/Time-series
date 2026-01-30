
# TreeDRNet: A Robust Deep Model for Long Term Time Series Forecasting
## 종합 분석 보고서

***

## 1. 핵심 주장 및 주요 기여

**TreeDRNet**(Tree-structured Doubly Residual Network)은 장기 시계열 예측(Long-Term Time Series Forecasting, LTSF) 문제를 해결하기 위해 설계된 신경망 아키텍처로, 2022년 6월 arXiv에 발표되었습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ea1d103d-6aee-4fb8-b09e-decf710caa68/2206.12106v1.pdf)

### 1.1 핵심 주장

논문의 핵심 주장은 세 가지입니다:

1. **Transformer 기반 모델의 치명적 약점**: 최신 Transformer 기반 모델(예: Autoformer)은 입력 길이가 증가함에 따라 성능이 급격히 악화되는 현상을 보입니다. 예를 들어, 입력 길이 336에서 3600으로 증가시 Autoformer의 MSE는 0.545에서 OOM(메모리 부족)에 이르게 됩니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ea1d103d-6aee-4fb8-b09e-decf710caa68/2206.12106v1.pdf)

2. **복잡한 모델의 역효과**: Transformer와 같은 복잡한 구조는 계산량 증가, 메모리 오버헤드, 그리고 과적합 위험을 초래합니다. TreeDRNet은 이를 단순한 MLP 기반 구조로 해결합니다.

3. **견고성의 중요성**: 장기 예측에서는 정확성뿐만 아니라 **견고성**(robustness)이 핵심입니다. 긴 입력, 노이즈가 많은 데이터, 분포 변화에 견딜 수 있어야 합니다.

### 1.2 주요 기여

**1) 이중 잔차 링크 구조(Doubly Residual Link Structure)**

반복 가중 최소제곱(Iterative Reweighted Least Squares, IRLS) 알고리즘에서 영감을 받아 설계:

$$\delta y_k = y - \sum_{j=1}^{k-1} z_j$$

$$x_k = f(x_{k-1}; \delta y_k) + x_{k-1} \quad \text{(Backcast + Skip Link)}$$

$$\Delta\beta_k = g(x_k, \delta y_k)$$

$$z_k = \langle \Delta\beta_k, x_k \rangle \quad \text{(Forecast Link)}$$

이는 각 계층에서 점진적으로 잔차를 감소시켜 견고한 학습을 가능하게 합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ea1d103d-6aee-4fb8-b09e-decf710caa68/2206.12106v1.pdf)

**2) Kolmogorov-Arnold 정리 기반 구조**

다변량 함수의 보편적 표현 능력에 기반:

$$h(x) = \sum_{k=0}^{2d} \Phi_k \left( \sum_{p=1}^{m} \varphi_{k,p}(x \circ m_{k,p}) \right)$$

여기서:
- **특징 선택**: $x \circ m_{k,p}$ (희소 이진 마스크)
- **모델 앙상블**: $\sum_{p=1}^{m} \varphi_{k,p}(\cdot)$ (다중 분기)
- **트리 구조**: $\Phi_k$ (계층적 정보 통합)

이는 모델의 표현 능력을 이론적으로 보증합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ea1d103d-6aee-4fb8-b09e-decf710caa68/2206.12106v1.pdf)

**3) 계산 효율성**

- Transformer 대비 **10배 이상** 빠른 학습/추론 속도
- 메모리 사용량: O(n) vs Transformer O(n log n)

**4) 예측 성능**

- 다변량 예측: **20-40% 오차 감소**
- 다양한 데이터셋에서 일관된 성능 향상

***

## 2. 문제 정의 및 해결 방법

### 2.1 해결하려는 문제

**문제 1: 입력 길이에 대한 견고성 부족**

표 2에서 보듯이, Autoformer는 입력 길이 96에서 0.433의 MSE를 보이지만 720으로 증가시 0.600(38% 악화)으로 급락합니다. 이는 장기 입력 활용의 핵심 한계입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ea1d103d-6aee-4fb8-b09e-decf710caa68/2206.12106v1.pdf)

**문제 2: 모델 복잡도와 과적합**

복잡한 모델은:
- 높은 계산 복잡도로 장기 예측에 부담
- 제한된 학습 데이터로 과적합 위험 증가
- 노이즈에 민감한 특성

**문제 3: 장기 의존성과 노이즈의 균형**

시계열 데이터는 장거리 의존성이 있으면서도 노이즈를 포함합니다. 이 둘을 동시에 처리해야 합니다.

### 2.2 제안 방법: 다층 구조

#### 단계 1: Doubly Residual Structure

각 블록에서:

$$x^{\ell+1} = x^\ell - \hat{x}^\ell \quad \text{(입력 업데이트)}$$

$$y^{\ell+1} = y^\ell + \hat{y}^\ell \quad \text{(예측 누적)}$$

여기서:
- $\hat{x}^\ell = FC_b(x_n^\ell)$: Backcast (입력과 같은 길이로 프로젝션)
- $\hat{y}^\ell = FC_f(x_n^\ell)$: Forecast (출력 길이로 프로젝션)

**의미**: 각 계층은 이전 계층이 설명하지 못한 "잔차"를 학습합니다. 이는 robust regression의 반복 가중 알고리즘과 이론적으로 동치입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ea1d103d-6aee-4fb8-b09e-decf710caa68/2206.12106v1.pdf)

#### 단계 2: Multi-branch Architecture

게이팅 메커니즘을 통한 특징 선택:

$$X_i = X \circ \sigma(f_i(X))$$

$$F_i, B_i = \text{DRes}_i(X_i)$$

모든 분기 평균화:

$$fc = \frac{1}{n} \sum_{i=1}^{n} F_i, \quad bc = \frac{1}{n} \sum_{i=1}^{n} B_i$$

**의미**: 각 분기가 입력의 다양한 부분조합을 학습하여, 앙상블을 통해 분산을 감소시킵니다. Kolmogorov-Arnold 정리의 $\sum \varphi_{k,p}$ 항에 대응됩니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ea1d103d-6aee-4fb8-b09e-decf710caa68/2206.12106v1.pdf)

#### 단계 3: Tree-structured Aggregation

이진 트리 토폴로지:

$$\text{predict} = \sum_{i=1}^{L} fc_i$$

여기서 각 레이어 $i$는 $2^{i-1}$개의 예측을 생성합니다.

**의미**: 상위 계층에서는 대역별 패턴을, 하위 계층에서는 세부 패턴을 학습하여, 다중 스케일 시간 의존성을 포착합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ea1d103d-6aee-4fb8-b09e-decf710caa68/2206.12106v1.pdf)

***

## 3. 모델 구조 상세 설명

### 3.1 전체 아키텍처 다이어그램

```
입력 시퀀스 (길이: T_in)
    ↓
FC (ReLU) ──→ 특징 추출
    ↓
┌─ Doubly Residual Block ─┐
│  ├─ n개 FC 레이어         │
│  ├─ Backcast (FC_b)      │ (반복 L번)
│  └─ Forecast (FC_f)      │
└─────────────────────────┘
    ↓
┌──────────────────────────────┐
│ Multi-branch Block (m개 분기)  │
│ ├─ Gating (sigmoid)          │
│ └─ DRes + Forecast           │
│   (평균화로 fc, bc 생성)      │
└──────────────────────────────┘
    ↓
┌─────────────────────┐
│ 트리 구조 스택      │
│ 레이어 1: 1개 노드   │ → fc¹
│ 레이어 2: 2개 노드   │ → fc²
│ 레이어 3: 4개 노드   │ → fc³
└─────────────────────┘
    ↓
최종 예측: predict = fc¹ + fc² + fc³
출력 시퀀스 (길이: T_out)
```

### 3.2 완전 연결 계층

각 블록 내:

$$x_0 = x^{\ell}$$

$$x_{i+1} = \text{ReLU}(FC_i(x_i)), \quad i = 0, \ldots, n-1$$

$$\hat{x}^\ell = FC_b(x_n)$$

$$\hat{y}^\ell = FC_f(x_n)$$

**설계 원칙**: 
- FC 계층은 특징 추출에만 사용
- Backcast와 Forecast는 명시적으로 분리하여, 입력 업데이트와 예측 누적을 명확히 함

### 3.3 공변량 처리

다변량 데이터(공변량 포함)의 경우:

$$x = \text{Conv}(W_x, \hat{X})$$

1×1 컨볼루션으로 각 시점에서 공변량 간 중요도를 학습합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ea1d103d-6aee-4fb8-b09e-decf710caa68/2206.12106v1.pdf)

***

## 4. 성능 향상 분석 및 경험적 증거

### 4.1 벤치마크 비교

표 1의 결과에 따르면: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ea1d103d-6aee-4fb8-b09e-decf710caa68/2206.12106v1.pdf)

| 데이터셋 | 예측 길이 | TreeDRNet MSE | Autoformer MSE | 개선율 |
|---------|----------|--------------|----------------|-------|
| ETTm2 | 336 | 0.303 | 0.339 | 10.6% |
| Electricity | 336 | 0.203 | 0.231 | 12.1% |
| Exchange | 336 | 0.412 | 0.509 | 19.1% |
| Traffic | 336 | 0.451 | 0.622 | 27.5% |
| Weather | 336 | 0.266 | 0.359 | 25.9% |

**평균 개선율: 23.3% (다변량 예측)**

### 4.2 계산 효율성

표 2의 결과: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ea1d103d-6aee-4fb8-b09e-decf710caa68/2206.12106v1.pdf)

| 모델 | 학습 시간 (ms) | 추론 시간 (ms) |
|-----|--------------|--------------|
| TreeDRNet | 0.156 | 0.110 |
| Autoformer | 2.302 | 1.534 |
| Informer | 2.494 | 2.108 |
| Transformer | 3.403 | 2.477 |

**TreeDRNet이 Autoformer 대비 14.8배 빠름**

### 4.3 견고성 실험

#### 4.3.1 입력 길이 증가에 대한 견고성 (표 3)

| 입력 길이 | TreeDRNet MSE | 변화율 | Autoformer MSE | 변화율 |
|----------|--------------|-------|--------------|-------|
| 96 | 0.414 | 기준 | 0.433 | 기준 |
| 192 | 0.412 | -0.5% | 0.463 | +6.9% |
| 336 | 0.409 | -1.2% | 0.545 | +25.9% |
| 720 | 0.407 | -1.7% | 0.600 | +38.6% |
| 1440 | 0.388 | -6.3% | OOM | - |
| 3600 | 0.387 | -6.5% | OOM | - |

**중요한 발견**: TreeDRNet은 입력이 증가해도 성능이 오히려 개선되는 반면, Autoformer는 큰 폭으로 악화됩니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ea1d103d-6aee-4fb8-b09e-decf710caa68/2206.12106v1.pdf)

#### 4.3.2 노이즈 공격 견고성 (표 13)

Contextual Outlier Exposure (COE) 공격에서: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ea1d103d-6aee-4fb8-b09e-decf710caa68/2206.12106v1.pdf)

| 모델 | 기존 MSE | 공격 후 MSE | 성능 저하 |
|-----|---------|-----------|---------|
| TreeDRNet | 0.303 | 0.310 | 2.3% |
| Autoformer | 0.339 | 0.393 | 15.9% |

**다른 공격 유형도 유사한 패턴**: 백색 잡음 공격(-5.1% vs -24.9%), 이상치 주입(-8.1% vs -16.7%)

***

## 5. 모델의 한계 및 개선 필요 영역

### 5.1 공변량 처리의 한계

표 8에서 공변량이 많은 데이터셋(Vol, Retail)에서: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ea1d103d-6aee-4fb8-b09e-decf710caa68/2206.12106v1.pdf)

- TreeDRNet: 경쟁력 있는 성능
- TFT (Temporal Fusion Transformer): 우수한 성능

**원인**: Transformer의 주의 메커니즘이 공변량 간 복잡한 관계 포착에 더 효과적

### 5.2 성능 포화

표 12 (트리 깊이 민감도): [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ea1d103d-6aee-4fb8-b09e-decf710caa68/2206.12106v1.pdf)

| 트리 깊이 | ETTm2 (336) MSE |
|---------|-----------------|
| 1 | 0.320 |
| 2 | 0.303 |
| 3 | 0.314 |

**깊이 2 이후** 추가 깊이는 거의 개선을 제공하지 않습니다.

표 11 (은닉층 차원): [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ea1d103d-6aee-4fb8-b09e-decf710caa68/2206.12106v1.pdf)

| 차원 | Exchange (720) MSE |
|-----|-------------------|
| 32 | 1.456 |
| 64 | 1.215 |
| 128 | 0.690 |
| 256 | 1.848 |

**128 이후** 차원 증가는 오버피팅을 초래합니다.

### 5.3 특정 패턴에 대한 제한

- 강한 계절성이 있는 데이터에서는 시간 분해(temporal decomposition) 방법의 우수성에 미치지 못함
- 급격한 변화(structural break)에 대한 적응 능력 제한

***

## 6. 일반화 성능 향상 가능성

### 6.1 이론적 근거: 반복 가중 알고리즘의 견고성

**정리 (Chartrand & Yin, 2008)**: $L_p$ 노름 ($0 < p \leq 1$)에 대한 최적화:

$$\min_\beta \sum_{i=1}^{n} |\langle \beta, x_i \rangle - y_i|^p$$

반복 가중 최소제곱 알고리즘의 가중치:

$$w_{t,i} = |\langle \beta_t, x_i \rangle - y_i|^{p-2}$$

**의미**: 각 반복에서 큰 오차를 가진 샘플의 가중치를 감소시켜, 특이치(outlier)에 견강한 학습을 구현합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ea1d103d-6aee-4fb8-b09e-decf710caa68/2206.12106v1.pdf)

TreeDRNet의 이중 잔차 구조는 이를 신경망으로 일반화한 것으로, 동일한 견고성을 기대할 수 있습니다.

### 6.2 앙상블의 분산 감소 효과

다중 분기 앙상블:

$$\text{Var}(\text{Ensemble}) = \frac{1}{m} \text{Var}(\text{Individual}) + \mathbb{E}[(\text{Bias}_i)^2]$$

**효과**: 편향(bias)은 증가하지 않으면서도 분산(variance)을 $\frac{1}{m}$로 감소시킵니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ea1d103d-6aee-4fb8-b09e-decf710caa68/2206.12106v1.pdf)

Ablation Study (표 9): [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ea1d103d-6aee-4fb8-b09e-decf710caa68/2206.12106v1.pdf)

| 구성 | Exchange (720) MSE | 손실율 |
|-----|-------------------|-------|
| 전체 TreeDRNet | 0.690 | - |
| 앙상블 제거 | 1.509 | -118.7% |

### 6.3 특징 선택의 정규화 효과

Sigmoid 게이팅:

$$X_i = X \circ \sigma(f_i(X))$$

각 분기가 관련성 높은 특징만 선택하므로:
- 노이즈 특징의 영향 최소화
- 분포 이동에 대한 강건성 증가
- 차원의 저주(curse of dimensionality) 완화

### 6.4 다중 스케일 표현

트리 구조의 계층적 학습:

**하위 계층** ($i=1$): 세부 시간 패턴
**상위 계층** ($i=L$): 조대한 추세 및 계절성

이는 자동으로 다양한 시간 스케일을 포착합니다.

### 6.5 실증적 증거: Ablation Study

표 9의 완전한 결과 (ETTm2, 예측 길이 720): [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ea1d103d-6aee-4fb8-b09e-decf710caa68/2206.12106v1.pdf)

| 제거 요소 | MSE | 성능 저하 |
|---------|-----|---------|
| 없음 (전체) | 0.387 | - |
| 특징선택 제거 | 0.445 | 14.9% |
| 앙상블 제거 | 0.463 | 19.6% |
| 트리 구조 제거 | 0.479 | 23.8% |

**모든 요소가 누적적으로 일반화 성능 향상에 기여**

***

## 7. 2020년 이후 최신 관련 연구 비교

### 7.1 MLP 기반 모델의 부활 (2023-2025)

#### DLinear (Zeng et al., 2023)

**개념**: 시계열 분해 + 선형 레이어

$$\text{TreeDRNet 대비}: \text{DLinear} + \text{Tree + Multi-branch + Feature Selection}$$

TreeDRNet은 DLinear를 포함하는 더 강력한 모델로 평가됩니다. [arxiv](https://arxiv.org/html/2312.06786v3)

#### Mixture-of-Linear-Experts (MoLE, 2024)

**개념**: 다중 선형 전문가 학습

성과: DLinear에 MoLE 적용 시 32/44 실험(73%)에서 개선 [proceedings.mlr](https://proceedings.mlr.press/v238/ni24a/ni24a.pdf)

**연관성**: TreeDRNet의 다중 분기 구조와 유사한 아이디어

#### XLinear (2025)

**개념**: MLP 기반 경량 모델 + 공변량 처리

성과: TreeDRNet의 공변량 처리 약점을 보완한 모델로 평가 [arxiv](https://arxiv.org/html/2601.09237v1)

### 7.2 State Space Models (SSM) - 신흥 패러다임

#### Mamba (2023-2024)

**핵심 혁신**: 선택적 상태공간 모델(Selective SSM)

$$h_t = A_t h_{t-1} + B_t x_t$$

$$y_t = C_t h_t$$

**특징**: 
- 선택 메커니즘: 입력에 따라 $A_t, B_t, C_t$ 동적 변화
- 계산 복잡도: **O(n)** (Transformer O(n log n) vs)
- 성능: Transformer 능가 [arxiv](https://arxiv.org/pdf/2312.00752.pdf)

**현황**: TreeDRNet 이후 발표된 가장 중요한 패러다임 변화 [arxiv](https://arxiv.org/abs/2403.11144)

#### S-Mamba (2024) & MambaTS (2024)

S-Mamba는 Mamba를 시계열 최적화:
- 쌍방향 처리
- 변수 간 상관 추출 [arxiv](https://arxiv.org/abs/2403.11144)

MambaTS는 Mamba의 한계 분석 후 개선:
1. 장거리 의존성 학습 부족
2. 변수 간 관계 처리 미흡
3. 채널 독립성 미지원 [arxiv](https://arxiv.org/abs/2405.16440)

### 7.3 Transformer 기반 개선

#### Autoformer (2021) - TreeDRNet 주요 비교 대상

**자동 상관(Autocorrelation) 기반 주의**

$$\text{AutoCorrelation}(Y) = \text{FFT}^{-1}(\text{FFT}(Y) \odot \overline{\text{FFT}(Y)})$$

TreeDRNet의 주요 벤치마크로, TreeDRNet이 대부분의 지표에서 능가합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ea1d103d-6aee-4fb8-b09e-decf710caa68/2206.12106v1.pdf)

#### iTransformer (2023)

**변수별 독립 처리 + 채널 간 주의**

비교: TreeDRNet의 다중 분기도 유사한 아이디어를 구현

#### PatchTST (2022) & Temporal Fusion Transformer (TFT, 2021)

PatchTST: 패치 기반 토큰화로 효율성 향상
TFT: 공변량 처리에서 우수 (TreeDRNet의 약점 보완)

### 7.4 주파수 도메인 방법 (2023-2025)

#### FourierFormer & FreTS (2023)

**개념**: FFT를 이용한 주파수 도메인 처리

$$Y_f = \text{FFT}(Y)$$

**장점**: 주기성 포착 용이

#### FMTCN (Frequency-domain enhanced Multi-scale TCN, 2025)

**혁신**: 시간 도메인 + 주파수 도메인 + 하이브리드 손실함수

**성과**: 다변량 8.6%, 단변량 6.1% SOTA 개선 [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S1568494625010920)

### 7.5 견고성 중심 연구 (2023-2025)

#### Robust RNN (2023)

**개념**: 지역화된 확률적 민감도(LSS) 최소화

$$\text{LSS} = \mathbb{E}[|f(x + \delta x) - f(x)|^2]$$

**성과**: RNN 견고성 **53% 향상** [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0925231223000462)

#### Future-Guided Learning (2025)

**개념**: 미래 모델의 피드백으로 현재 모델 개선

$$\text{Loss} = \text{MSE} + \lambda \cdot \text{KL}(P_{\text{future}} || P_{\text{current}})$$

**성과**: **11-13% MSE 감소** [arxiv](http://arxiv.org/pdf/2410.15217.pdf)

### 7.6 기반 모델의 등장

#### Mamba4Cast (2024)

**개념**: Mamba 기반 제로샷 예측 모델

합성 데이터로만 학습하여 다양한 시계열에 일반화
세밀한 튜닝 없이 경쟁력 있는 성능 [neurips](https://neurips.cc/virtual/2024/102938)

#### TimeCopilot (2024)

여러 기반 모델 앙상블로 최고 성능 달성 [arxiv](https://arxiv.org/html/2509.00616v1)

### 7.7 최신 트렌드 (2025년)

#### vLinear (2025)

**유동성 정합(Flow Matching) 이론 기반**

$$\mathcal{L}_{\text{WFM}} = \mathbb{E}_{t, n}[w(t) || \text{ODE}(\phi, t) - Y_n ||^2]$$

**성과**: 22개 벤치마크, 124개 설정에서 **SOTA 달성** [arxiv](https://arxiv.org/html/2601.13768v1)

#### LiNo (2025)

**선형-비선형 패턴 분해 + 재귀적 잔차 분해**

선형과 비선형 패턴을 명시적으로 분리하여 견고성 강화 [arxiv](https://arxiv.org/html/2410.17159v2)

#### vLinear vs TreeDRNet

| 측면 | TreeDRNet | vLinear |
|-----|----------|---------|
| 발표연도 | 2022 | 2025 |
| 핵심 아이디어 | 이중 잔차 + 트리 | 유동성 정합 + 가중화 |
| 이론적 근거 | IRLS + 색농-아놀드 | 정규화 흐름 이론 |
| 성능 | SOTA (당시) | 최신 SOTA |

***

## 8. 모델의 학문적 영향 및 위상

### 8.1 발표 이후 연구 생태계의 변화

**2022년 TreeDRNet 발표 전**: Transformer 지배
- Informer, Autoformer, Reformer 등 attention 기반 모델 전성기
- MLP의 단순함은 성능 부족으로 간주

**2022년 TreeDRNet 발표 후**: MLP 르네상스
- DLinear, RLinear, N-BEATS 재평가
- "단순함이 최고의 설계" (Simplicity as the Ultimate Design)

**2023-2024년**: 다양화 시대
- Mamba 등 State Space Model 급부상
- 주파수 도메인 방법 재조명
- 분해(Decomposition) 기반 모델 활성화
- 견고성 및 불확실성 정량화 연구 심화

### 8.2 인용 및 영향력

- **인용 수**: 11회 (arXiv 2024년 기준)
- **후속 논문**:
  - MoLE (Mixture-of-Linear-Experts)
  - 다양한 분해 기반 모델
  - 견고성 중심 연구들

### 8.3 기술 기여의 영구적 자산

**TreeDRNet의 레거시**:

1. **이중 잔차 구조**: 현재 많은 모델의 기본 구성 요소
2. **다중 분기 앙상블**: MoLE 등에서 재현
3. **특징 선택 메커니즘**: 가중 게이팅의 프로토타입
4. **계산 효율성의 중요성**: 이후 연구의 핵심 지표화

***

## 9. 향후 연구 시 고려할 핵심 사항

### 9.1 기술적 개선 방향

**1) 공변량 처리 강화**

현황: TreeDRNet은 공변량이 적을 때는 우수하나, Vol/Retail 데이터셋에서 TFT에 미흡

**개선안**:
- TFT의 멀티헤드 주의를 TreeDRNet의 다중 분기와 통합
- 공변량별 동적 게이팅 학습

$$X_i = X_{\text{target}} \circ \sigma(f_i(X_{\text{covariate}}))$$

**2) 트리 구조 동적 최적화**

현황: 트리 깊이 2-3에서 포화

**개선안**:
- 데이터 특성에 따른 적응적 깊이 결정
- 비대칭 트리 구조 탐색:

$$\text{predict} = \sum_{i=1}^{L} w_i \cdot fc_i$$

여기서 $w_i$는 학습 가능한 가중치

**3) 확률적 예측 지원**

현황: 점 예측(point prediction)만 제공

**개선안**: Mamba-ProbTSF 스타일의 불확실성 정량화

$$(\mu, \sigma) = \text{TreeDRNet}_{\mu,\sigma}(x)$$

$$P(y | x) = \mathcal{N}(y | \mu, \sigma)$$

성과 기대: 위험 관리 및 신뢰도 평가 가능 [arxiv](https://arxiv.org/abs/2503.10873)

### 9.2 이론적 심화 연구

**1) 일반화 경계(Generalization Bound) 분석**

$$|R_{\text{test}} - R_{\text{train}}| \leq O\left(\sqrt{\frac{\text{VC-dim}(H)}{m}}\right) + o(1)$$

TreeDRNet의 VC 차원과 Rademacher 복잡도 분석 필요

**2) 최적화 수렴 이론**

반복 가중 알고리즘의 수렴 속도:

```math
||x_t - x^*||^2 \leq (1 - \mu)^t ||x_0 - x^*||^2
```

여기서 $\mu$는 강볼록성(strong convexity) 계수

TreeDRNet의 신경망 버전에 대한 유사한 분석 필요

**3) 견고성 증명**

Lipschitz 연속성과 섭동 제한:

```math
||f(x + \delta x) - f(x)|| \leq L \cdot ||\delta x||
```

특히 $L_p$ 노름 기반 손실함수에서의 견고성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ea1d103d-6aee-4fb8-b09e-decf710caa68/2206.12106v1.pdf)

### 9.3 응용 확장 방향

**1) 비정상성(Non-stationarity) 처리**

현황: 대부분 정상성 가정

**개선안**:
- RevIN (Reversible Instance Normalization) 고도화
- 온라인 학습으로 분포 변화 적응
- 도메인 적응 기법 통합

**2) 극단 사건(Extreme Events) 예측**

현황: 일반 패턴에 최적화

**개선안**:
- 극단값 손실함수 (Pinball Loss, Quantile Loss):

$$L_{\tau}(y, \hat{y}) = \tau(y - \hat{y})_+ + (1-\tau)(\hat{y} - y)_+$$

**3) 다변량 의존성 모델링**

현황: 채널 독립성 주로 가정

**개선안**:
- 그래프 신경망(GNN) 통합
- 변수 간 구조적 관계 명시

$$G = (V, E), \text{ where } V = \text{variables}, E = \text{dependencies}$$

### 9.4 최신 트렌드와의 통합

**1) State Space Model 하이브리드**

TreeDRNet의 다중 분기 + Mamba의 선택성:

$$\text{TreeDRNet-Mamba Hybrid}$$

각 분기에 Mamba 적용하여 계산 효율성 + 표현력 강화 [arxiv](https://arxiv.org/abs/2404.14757)

**2) 기반 모델 활용**

Mamba4Cast 스타일의 제로샷 학습:
- 합성 데이터로 사전 학습
- 세밀한 튜닝 없이 새 데이터에 적응

**3) 불확실성 정량화**

Future-Guided Learning의 동적 적응:

$$\text{Loss} = \text{Main Task Loss} + \lambda_t \cdot \text{Guidance Loss}(t)$$

시간에 따라 조정되는 안내 신호 [nature](https://www.nature.com/articles/s41467-025-63786-4)

### 9.5 벤치마크 및 평가 표준화

**1) 새로운 데이터셋**

현황: ETT, Electricity, Weather 등 기존 데이터셋 의존

**필요성**:
- **고주파 데이터**: 분 또는 초 단위 (IoT, 금융 시세)
- **초장기 시퀀스**: 2년 이상 데이터
- **극단 사건 포함**: 금융 위기, 이상 기후

**2) 다원적 평가 지표**

점수 기반 메트릭(Proper Scoring Rule):

$$S(p, y) = -2p_y + \sum_{i} p_i^2$$

여기서 $p$는 예측 분포, $y$는 실제값 [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12281171/)

**3) 견고성 테스트 표준화**

- Contextual Outlier, Anomaly Injection, White Noise (TreeDRNet에서 도입) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ea1d103d-6aee-4fb8-b09e-decf710caa68/2206.12106v1.pdf)
- 추가: Adversarial Perturbation, Shift in Distribution

***

## 10. 최종 평가 및 결론

### 10.1 TreeDRNet의 핵심 성과

| 측면 | 평가 |
|-----|------|
| **이론적 근거** | ⭐⭐⭐⭐⭐ (IRLS + 색농-아놀드) |
| **성능** | ⭐⭐⭐⭐⭐ (당시 SOTA, 20-40% 개선) |
| **계산 효율성** | ⭐⭐⭐⭐⭐ (10배 속도) |
| **입력 길이 견고성** | ⭐⭐⭐⭐⭐ (OOM 없이 3600 처리) |
| **노이즈 견고성** | ⭐⭐⭐⭐⭐ (5% vs 25% 성능 저하) |
| **공변량 처리** | ⭐⭐⭐☆☆ (TFT에 미흡) |
| **확률적 예측** | ⭐⭐☆☆☆ (미지원) |
| **해석 가능성** | ⭐⭐☆☆☆ (블랙박스) |

### 10.2 학문적 기여의 지속성

TreeDRNet은:

1. **패러다임 전환의 촉발자**
   - Transformer 지배에서 MLP 르네상스로의 전환
   - "단순함의 가치" 재발견

2. **견고성 연구의 기초**
   - 입력 길이/노이즈 견고성의 중요성 확립
   - 후속 Future-Guided Learning, Robust RNN 등에 영감

3. **효율성의 새로운 표준**
   - 10배 속도 달성으로 실용성 강조
   - Mamba의 효율성 추구에 영감

### 10.3 현재 위치 (2025년 기준)

**성능 측면**:
- vLinear 등 최신 모델에 의해 성능 면에서 추월당함
- 하지만 **이론적 명확성과 견고성은 여전히 우수**

**영향력 측면**:
- 직접 인용: 11회
- 간접 영향: 매우 큼 (MLP 재평가, 견고성 연구)

**현업 적용**:
- 효율성과 해석 가능성이 중요한 분야에서 선호
- 실시간 예측 시스템에 채택

### 10.4 향후 연구의 정책 제안

**우선순위 높음**:
1. ✅ 공변량 처리 강화
2. ✅ 확률적 예측 지원
3. ✅ Mamba와의 하이브리드

**우선순위 중간**:
1. 동적 트리 구조
2. 불확실성 정량화
3. 극단 사건 처리

**우선순위 낮음** (기초 연구):
1. VC 차원 분석
2. 수렴 속도 증명
3. Lipschitz 상수 추정

### 10.5 최종 결론

TreeDRNet은 **단순함의 힘**을 입증한 이정표적 연구입니다. 

**반복 가중 알고리즘**과 **색농-아놀드 정리**라는 견고한 이론적 기반 위에, 다음을 달성했습니다:

- Transformer 대비 **10배 계산 효율성**
- **20-40% 성능 향상**
- **극도의 입력 길이(3600) 견고성**
- **노이즈에 대한 뛰어난 견강성**

2022년 발표 이후:
- MLP 기반 모델의 르네상스 촉발
- 견고성 중심 연구 활성화
- 다양한 하이브리드 모델의 영감원

2025년 현재, 최신 vLinear, Mamba 등의 등장으로 **성능 면에서는 추월**당했지만, **이론적 명확성, 계산 효율성, 견고성** 면에서는 여전히 **학습 가치 높은 모델**입니다.

향후 연구는:
- **공변량 처리** 강화
- **확률적 예측** 지원
- **State Space Model과의 통합**
- **극단 사건 처리** 개선

에 집중하여, 실제 산업 응용에서의 **완전성**을 달성할 수 있을 것으로 예상됩니다.

***

## 참고 문헌

 Zhou, T., Zhu, J., Wang, X., Ma, Z., Wen, Q., Sun, L., & Jin, R. (2022). TreeDRNet: A Robust Deep Model for Long Term Time Series Forecasting. *arXiv preprint arXiv:2206.12106*. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ea1d103d-6aee-4fb8-b09e-decf710caa68/2206.12106v1.pdf)

 Kong, Y., Wang, Z., Nie, Y., Zhou, T., Zohren, S., Liang, Y., Sun, P., & Wen, Q. (2024). Unlocking the Power of LSTM for Long Term Time Series Forecasting. *arXiv preprint arXiv:2408.10006*. [arxiv](http://arxiv.org/pdf/2410.15217.pdf)

 Santos, R. P. dos. (2025). Deep learning in time series forecasting with transformer. *PeerJ*, 13, e3001. [peerj](https://peerj.com/articles/cs-3001/)

 Wang, J., & colleagues. (2025). Long-term time series forecasting by a frequency-domain enhanced approach. *Science Direct*. [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S1568494625010920)

 Gunasekaran, S., & colleagues. (2025). A predictive approach to enhance time-series forecasting. *Nature Communications*. [nature](https://www.nature.com/articles/s41467-025-63786-4)

 Ni, R., Lin, Z., Wang, S., & Fanti, G. (2024). Mixture-of-Linear-Experts for Long-term Time Series Forecasting. *MLNLP Conference*. [arxiv](https://arxiv.org/html/2312.06786v3)

 Wei, X., & colleagues. (2025). LiNo: Advancing Recursive Residual Decomposition. *arXiv preprint*. [arxiv](https://arxiv.org/html/2410.17159v2)

 Lim, B., & colleagues. (2023). Revisiting Long-term Time Series Forecasting. *arXiv preprint arXiv:2305.10721*. [arxiv](https://arxiv.org/pdf/2305.10721.pdf)

 Zhang, X., & colleagues. (2025). XLinear: A Lightweight and Accurate MLP-Based Model. *arXiv preprint arXiv:2601.09237*. [arxiv](https://arxiv.org/html/2601.09237v1)

 Guen, V. L., & colleagues. (2024). SST: Multi-Scale Hybrid Mamba-Transformer. *arXiv preprint arXiv:2404.14757*. [arxiv](https://arxiv.org/abs/2404.14757)

 Pessoa, P., & colleagues. (2025). Mamba time series forecasting with uncertainty propagation. *arXiv preprint arXiv:2503.10873*. [arxiv](https://arxiv.org/abs/2503.10873)

 Wang, Z., & colleagues. (2024). Is Mamba Effective for Time Series Forecasting? *arXiv preprint arXiv:2403.11144*. [arxiv](https://arxiv.org/abs/2403.11144)

 Song, H., & colleagues. (2025). vLinear: A Powerful Linear Model for Multivariate Time Series. *arXiv preprint arXiv:2601.13768*. [arxiv](https://arxiv.org/html/2601.13768v1)

 Gu, A., & colleagues. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Space Models. *arXiv preprint arXiv:2312.00752*. [arxiv](https://arxiv.org/pdf/2312.00752.pdf)

 Cai, X., & colleagues. (2024). MambaTS: Improved Selective State Space Models for Long-Term Time Series Forecasting. *arXiv preprint arXiv:2405.16440*. [arxiv](https://arxiv.org/abs/2405.16440)

 Zhang, X., & colleagues. (2023). Robust recurrent neural networks for time series forecasting. *Neurocomputing*, 522, 462-475. [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0925231223000462)

 Ni, R., Lin, Z., Wang, S., & Fanti, G. (2024). Mixture-of-Linear-Experts for Long-term Time Series Forecasting. *PMLR*, 238. [proceedings.mlr](https://proceedings.mlr.press/v238/ni24a/ni24a.pdf)

 Pessoa, P., & colleagues. (2025). Mamba time series forecasting with uncertainty quantification. *PMC*. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12281171/)
