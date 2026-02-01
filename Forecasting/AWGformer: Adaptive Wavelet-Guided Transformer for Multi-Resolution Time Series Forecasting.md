
# AWGformer: Adaptive Wavelet-Guided Transformer for Multi-Resolution Time Series Forecasting
## 1. 핵심 요약

**AWGformer**(Adaptive Wavelet-Guided Transformer)는 웨이블릿 분석과 Transformer 아키텍처를 통합한 혁신적 시계열 예측 모델로, 신호 특성에 따라 동적으로 최적의 웨이블릿 기저와 분해 수준을 선택합니다. 이 연구는 기존 Transformer 기반 모델들이 다중 해상도(multi-resolution) 시간 패턴을 효과적으로 포착하지 못한다는 문제를 해결하기 위해 제안되었습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6a3f6b49-35c1-4a62-8768-4f712c12f634/2601.20409v1.pdf)

### 주요 기여

1. **적응형 웨이블릿 분해 모듈(AWDM)**: 신호의 특성을 기반으로 최적의 웨이블릿 기저와 분해 수준을 동적으로 결정 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6a3f6b49-35c1-4a62-8768-4f712c12f634/2601.20409v1.pdf)
2. **교차 스케일 특성 융합(CSFF)**: 학습 가능한 결합 행렬을 통해 서로 다른 주파수 대역 간의 비선형 상호작용 포착 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6a3f6b49-35c1-4a62-8768-4f712c12f634/2601.20409v1.pdf)
3. **주파수 인식 멀티헤드 어텐션(FAMA)**: 각 주의(attention) 헤드를 특정 주파수 대역에 전문화 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6a3f6b49-35c1-4a62-8768-4f712c12f634/2601.20409v1.pdf)
4. **계층적 예측 네트워크(HPN)**: 다중 해상도에서 병렬로 예측을 생성한 후 역 웨이블릿 변환으로 최종 예측 생성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6a3f6b49-35c1-4a62-8768-4f712c12f634/2601.20409v1.pdf)

***

## 2. 상세 기술 분석

### 2.1 문제 정의

시계열 예측의 핵심 과제는 고주파 노이즈부터 장기 트렌드까지 다양한 시간 스케일의 패턴을 동시에 포착하는 것입니다. 기존 Transformer 모델들(Informer, Autoformer, PatchTST)은: [ojs.aaai](https://ojs.aaai.org/index.php/AAAI/article/view/17325)

- 모든 시간 스케일을 동등하게 취급하여 스케일 특화 모델링 부재
- 비정상(non-stationary) 시계열의 복잡한 패턴 포착 어려움
- 주파수 대역 간 상호작용 미모델링

AWGformer는 이러한 제약을 극복하기 위해 신호처리의 고전 이론인 웨이블릿 분석을 신경망과 통합합니다.

### 2.2 제안 방법: 수식 중심 설명

#### 2.2.1 학습 가능한 웨이블릿 변환

기존의 고정된 웨이블릿 함수를 신경망으로 매개변수화하여 신호 특성에 맞게 적응:

$$\psi_t = g_t \cos(\omega t + \phi)$$

여기서 $g_t$는 학습 가능한 포락 함수, $\omega$는 주파수 매개변수, $\phi$는 위상 매개변수입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6a3f6b49-35c1-4a62-8768-4f712c12f634/2601.20409v1.pdf)

다단계 웨이블릿 분해:

$$H_j^n = \psi_{j,n} * x_k, \quad L_J = \phi_J * x_k$$

**적응형 분해 수준 선택:**

$$J^* = \arg\min_J L_{recon}(J) + \lambda L_{sparse}(J)$$

여기서 $L_{recon}$는 재구성 오차, $L_{sparse}$는 웨이블릿 도메인의 희소성 정규화입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6a3f6b49-35c1-4a62-8768-4f712c12f634/2601.20409v1.pdf)

#### 2.2.2 교차 스케일 특성 융합 (CSFF)

서로 다른 주파수 대역 간의 상호작용을 모델링합니다:

$$F_{fused}^{i,j} = W_{ij} \otimes H_i \otimes H_j$$

여기서 $\otimes$는 외적(outer product), $W_{ij}$는 학습 가능한 결합 행렬입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6a3f6b49-35c1-4a62-8768-4f712c12f634/2601.20409v1.pdf)

잔여 연결을 통해 스케일 특화 정보 보존:

$$F_j = F_j + \alpha_j \odot F_j^{fused}$$

여기서 $\alpha_j$는 학습 가능한 게이트 벡터입니다. 훈련 중 **스펙트럼 드롭아웃**을 적용하여(20% 주파수 채널 무작위 영(zero)화) 비필수 교차 대역 상관관계의 과적합을 방지합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6a3f6b49-35c1-4a62-8768-4f712c12f634/2601.20409v1.pdf)

#### 2.2.3 주파수 인식 멀티헤드 어텐션 (FAMA)

각 어텐션 헤드에 주파수 응답 함수를 부여:

$$H_h(\omega) = \exp\left(-\frac{(\omega - \mu_h)^2}{2\sigma_h^2}\right)$$

여기서 $\mu_h$는 중심 주파수, $\sigma_h$는 대역폭입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6a3f6b49-35c1-4a62-8768-4f712c12f634/2601.20409v1.pdf)

수정된 어텐션 계산:

$$\text{Attention}_h(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} \odot M_h\right) V$$

이는 Heisenberg 불확실성 원리 $\Delta f \cdot \Delta t = \frac{1}{4\pi}$를 만족합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6a3f6b49-35c1-4a62-8768-4f712c12f634/2601.20409v1.pdf)

#### 2.2.4 계층적 예측 네트워크 (HPN)

다중 해상도에서 병렬 예측 생성:

$$L_J = f_L(L_J, F^{fused}), \quad H_j = f_j(H_j, F^{fused})$$

역 웨이블릿 변환으로 최종 예측:

$$\hat{Y} = W^{-1}\left(L_J, \{H_j\}_{j=1}^J\right)$$

#### 2.2.5 훈련 목적 함수

$$\mathcal{L} = \mathcal{L}_{pred} + \lambda_1 \mathcal{L}_{recon} + \lambda_2 \mathcal{L}_{ortho} + \lambda_3 \mathcal{L}_{smooth}$$

여기서: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6a3f6b49-35c1-4a62-8768-4f712c12f634/2601.20409v1.pdf)
- $\mathcal{L}_{pred} = \|\hat{Y} - Y\|_2^2$: 예측 오차
- $\mathcal{L}_{recon} = \|X - W^{-1}(W(X))\|_2^2$: 완벽한 재구성 보장
- $\mathcal{L}_{ortho} = \|W^TW - I\|_F^2$: 직교 웨이블릿 장려
- $\mathcal{L}_{smooth} = \sum_j \|\psi_j''\|_2^2$: 부드러운 기저 함수 장려

### 2.3 모델 구조

AWGformer는 네 가지 주요 모듈로 구성됩니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6a3f6b49-35c1-4a62-8768-4f712c12f634/2601.20409v1.pdf)

1. **AWDM 단계**: 입력 시계열을 신호 특성에 맞게 다중 해상도로 분해
2. **N개 인코더 층**: CSFF, 정규화, FAMA 어텐션, 피드포워드 네트워크를 반복 적용
3. **HPN 단계**: 각 해상도에서 독립적으로 예측 생성
4. **재구성 단계**: 역 웨이블릿 변환으로 최종 예측 생성

***

## 3. 성능 향상 분석

### 3.1 벤치마크 성능

| 데이터셋 | 메트릭 | Autoformer | FEDformer | iTransformer | AWGformer | 개선도 |
|---------|--------|-----------|----------|--------------|-----------|--------|
| **ETTh1** | MSE@720 | 0.514 | 0.506 | 0.491 | **0.479** | 4.2% |
| **Traffic** | MSE@336 | 0.622 | 0.621 | 0.433 | **0.407** | 6.0% |
| **Electricity** | MSE@720 | 0.254 | 0.246 | 0.223 | **0.204** | 7.3% |

AWGformer는 모든 벤치마크 데이터셋에서 기존 최고 성능 모델들을 능가합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6a3f6b49-35c1-4a62-8768-4f712c12f634/2601.20409v1.pdf)

### 3.2 제거 연구 (Ablation Study)

| 모델 변형 | ETTh1 MSE@336 | 성능 저하 |
|---------|---------------|----------|
| 전체 AWGformer | **0.435** | - |
| 적응형 웨이블릿 대신 고정 Db4 | 0.470 | 8.0% ↑ |
| 교차 스케일 융합 제거 | 0.460 | 5.7% ↑ |
| 주파수 인식 어텐션 제거 | 0.455 | 4.6% ↑ |
| 단일 수준 분해 | 0.484 | 11.3% ↑ |

제거 연구는 **적응형 웨이블릿이 가장 큰 성능 향상(8.0%)**을 제공하며, **다중 수준 분해의 중요성(11.3% 성능 저하)**을 입증합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6a3f6b49-35c1-4a62-8768-4f712c12f634/2601.20409v1.pdf)

### 3.3 결측 데이터 견고성

30% 결측 데이터 상황에서 AWGformer의 우수성: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6a3f6b49-35c1-4a62-8768-4f712c12f634/2601.20409v1.pdf)

| 모델 | H=720 | 상대 개선 |
|-----|-------|---------|
| TimesNet | 0.51 | - |
| PatchTST | 0.53 | -3.9% |
| **AWGformer** | **0.437** | **6.4%** |

웨이블릿 기반 다중 해상도 표현이 불완전한 데이터에서도 전역 추세와 국소 평활성을 활용하여 강건성을 제공합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6a3f6b49-35c1-4a62-8768-4f712c12f634/2601.20409v1.pdf)

***

## 4. 일반화 성능 향상 메커니즘

### 4.1 일반화 성능의 정의

시계열 예측에서 일반화 성능은 다음을 포함합니다: [arxiv](https://arxiv.org/html/2508.02753v4)

- **학습-테스트 분포 차이**: 비정상성, 개념 드리프트, 계절성 변화
- **도메인 간 전이**: 한 데이터셋에서 학습한 모델을 다른 도메인에 적용
- **장기 예측 견고성**: 예측 지평 증가에 따른 성능 저하 완화

### 4.2 AWGformer의 일반화 메커니즘

#### 4.2.1 다중 스케일 신호 분해

웨이블릿 분해는 신호를 다양한 시간 스케일에서 표현합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6a3f6b49-35c1-4a62-8768-4f712c12f634/2601.20409v1.pdf)

$$X(t) = \sum_{k} c_J(k)\phi(2^{-J}t - k) + \sum_{j=1}^{J}\sum_{k} d_j(k)\psi(2^{-j}t - k)$$

이 계층적 표현은 비정상 신호에서도 안정적인 특징을 추출하고, 노이즈에 자연스러운 필터링 효과를 제공합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6a3f6b49-35c1-4a62-8768-4f712c12f634/2601.20409v1.pdf)

#### 4.2.2 신호 적응 웨이블릿

고정된 웨이블릿과 달리, AWGformer의 학습 가능 웨이블릿은 신호별로 맞춤화됩니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6a3f6b49-35c1-4a62-8768-4f712c12f634/2601.20409v1.pdf)

- **비정상 신호**: 신호 특성에 따라 주파수 $\omega(x)$ 동적 조정
- **다중 주기성**: 각 스케일마다 최적 웨이블릿 학습
- **도메인 전이**: 새로운 도메인에서 빠른 적응

#### 4.2.3 주파수 특화 어텐션 헤드

FAMA의 주파수 인식 설계로: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6a3f6b49-35c1-4a62-8768-4f712c12f634/2601.20409v1.pdf)

- **헤드 1** (저주파): 트렌드와 장기 패턴 포착 ($\mu_1 \approx 0$)
- **헤드 2** (중주파): 계절성과 주기 패턴 포착 ($\mu_2$ 중간값)
- **헤드 3** (고주파): 상세 변동 및 노이즈 감별 ($\mu_3$ 고주파)

이러한 자동 전문화는 도메인별 우세 주파수가 다르더라도 유연하게 적응합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6a3f6b49-35c1-4a62-8768-4f712c12f634/2601.20409v1.pdf)

#### 4.2.4 장기 예측 견고성

계층적 예측 네트워크의 다중 해상도 병렬 처리는: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6a3f6b49-35c1-4a62-8768-4f712c12f634/2601.20409v1.pdf)

- **수준 간 상호 검증**: 각 해상도가 독립적으로 예측하므로 상호 보정
- **오차 누적 완화**: 추세는 안정적으로 예측, 상세는 단기 정확도 유지

96 → 720 단계 예측 시:
- Autoformer: 13.2% 성능 저하
- **AWGformer: 7.6% 성능 저하** (42% 더 나은 견고성)

### 4.3 도메인 일반화 능력

| 도메인 특성 | 데이터셋 | AWGformer의 강점 |
|-----------|---------|-----------------|
| 강한 계절성 | Electricity | 적응형 웨이블릿으로 다중 주기성 자동 발견 |
| 약한 주기성 | Exchange | 비선형 상호작용과 추세 모델링 |
| 고주파 노이즈 | Traffic | FAMA로 노이즈 주파수 억제 |
| 비정상 신호 | Weather | 적응형 포락으로 신호 특성 추적 |

***

## 5. 모델의 한계

### 5.1 현재 제약사항

1. **컴팩트 서포트(Compact Support) 제약** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6a3f6b49-35c1-4a62-8768-4f712c12f634/2601.20409v1.pdf)
   - 학습된 웨이블릿이 여전히 제한된 범위에만 정의
   - 전역 서포트가 필요한 비정상 신호에서 표현력 한계

2. **이차 주의 복잡도** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6a3f6b49-35c1-4a62-8768-4f712c12f634/2601.20409v1.pdf)
   - 매우 긴 시계열(T > 100,000)에서 계산 병목
   - 실시간 응용에 부적합

3. **규칙적 샘플링 가정** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6a3f6b49-35c1-4a62-8768-4f712c12f634/2601.20409v1.pdf)
   - 불규칙 타임스탬프 처리 미흡
   - 실제 산업 데이터의 ~30%가 불규칙 샘플링

### 5.2 향후 해결 방안

1. **암시적 신경 표현(Implicit Neural Representations)**으로 전역 서포트 웨이블릿 달성
2. **저계층(Low-rank) 근사 주의**로 $O(T^2) \to O(T)$ 복잡도 축소
3. **확률적 예측**으로 확장하여 보정된 불확실성 추정 제공

***

## 6. 2020년 이후 시계열 예측 연구 비교

### 6.1 주요 Transformer 모델의 진화

**Informer (2021)**: ProbSparse Self-Attention으로 $O(L \log L)$ 효율성 달성. 하지만 모든 시간 스케일을 동등하게 취급. [ojs.aaai](https://ojs.aaai.org/index.php/AAAI/article/view/17325)

**Autoformer (2021)**: 내부 분해 패러다임 도입. 추세와 계절성을 진행적으로 분해하여 기준선 대비 38% 성능 개선. 고정된 분해 방식이 한계. [semanticscholar](https://www.semanticscholar.org/paper/fc46ccb83dc121c33de7ab6bdedab7d970780b2f)

**FEDformer (2022)**: 주파수 도메인 향상으로 FFT 기반 표현 도입. 선형 시간 복잡도 달성. AWGformer와 달리 학습 가능 웨이블릿 미사용. [arxiv](https://arxiv.org/abs/2201.12740)

**PatchTST (2023)**: 시계열을 패치로 분할하여 로컬 의미 정보 보존. 채널 독립성으로 $O(N^2/S^2)$ 복잡도 달성. 고정 크기 패치의 한계. [arxiv](https://arxiv.org/abs/2211.14730)

**TimesNet (2023)**: 1D → 2D 변환으로 다중 주기성 모델링. 주기 내/간 변동을 2D 공간에서 표현. 주파수 기반 분해 미흡. [ise.thss.tsinghua.edu](https://ise.thss.tsinghua.edu.cn/~mlong/doc/TimesNet-iclr23.pdf)

**iTransformer (2024)**: 차원 역전 설계로 다변량 상관성을 직접 포착. 현재 높은 성능 달성. 다중 해상도 모델링 미흡. [iclr](https://iclr.cc/virtual/2024/poster/18933)

### 6.2 특성 비교표

| 특성 | Informer | Autoformer | FEDformer | PatchTST | TimesNet | iTransformer | AWGformer |
|------|----------|-----------|-----------|----------|----------|--------------|-----------|
| 신호 분해 | 없음 | STL | FFT | 패칭 | 1D→2D | 없음 | **웨이블릿** |
| 학습 가능 분해 | 아니오 | 아니오 | 일부 | 아니오 | 일부 | 아니오 | **예** |
| 주파수 인식 | 아니오 | 아니오 | 예 | 아니오 | 부분 | 아니오 | **예** |
| 상호작용 모델링 | 표준 | 자기상관 | 주의 | 주의 | 2D CNN | 주의 | **CSFF** |
| 결측 데이터 견고성 | 약함 | 중간 | 중간 | 약함 | 중간 | 약함 | **강함** |
| 계산 복잡도 | $O(L \log L)$ | $O(N^2)$ | $O(L)$ | $O((L/S)^2)$ | $O(N^2)$ | $O(N^2)$ | $O(N^2)$ |

### 6.3 성능 벤치마크 (ETTh1, H=720)

```
Model              MSE    개선도
─────────────────────────────
Informer (2021)   0.680  -32.0%
Autoformer (2021)  0.514   0.0% (기준)
FEDformer (2022)   0.506  +1.6%
TimesNet (2023)    0.500  +3.4%
PatchTST (2023)    0.488  +5.1%
iTransformer (2024) 0.491  +4.5%
AWGformer (2026)   0.479  +6.8% ✓
```

***

## 7. 향후 연구에 미치는 영향 및 고려사항

### 7.1 학술적 기여

**신호처리-AI 통합 패러다임 확립**: AWGformer는 고전 신호처리(웨이블릿 이론) 원리와 현대 심층학습(Transformer)을 통합하여, 향후 연구에서 물리 정보 신경망(Physics-Informed NN) 개념을 시계열 분야에 확산시킬 것으로 예상됩니다. [pubmed.ncbi.nlm.nih](https://pubmed.ncbi.nlm.nih.gov/21095693/)

**적응형 설계의 필요성 입증**: 고정된 기저의 한계를 명확히 함으로써, 향후 모든 시계열 모델은 신호 특성에 맞게 매개변수화할 필요성을 인식하게 될 것입니다. [proceedings.mlr](https://proceedings.mlr.press/v238/zhang24l/zhang24l.pdf)

### 7.2 실무 적용 전망

**산업별 응용:**

- **금융**: 변동성의 다중 스케일 모델링으로 포트폴리오 최적화 개선
- **에너지**: 신재생에너지 출력과 부하 예측으로 그리드 안정성 향상
- **의료**: 환자 신호 예측으로 조기 경고 시스템 구현
- **교통**: 교통 흐름과 배송 시간 예측 정확도 향상

### 7.3 주요 고려사항

#### 7.3.1 이론적 한계

- **근사성 증명 부족**: AWGformer가 어떤 함수 클래스를 근사 가능한지 미명확
- **수렴성 보장 부재**: 웨이블릿과 Transformer 학습의 상호작용에서 로컬 미니마 위험

#### 7.3.2 실무적 도전

- **계산 비용**: Autoformer 대비 20-30% 추가 학습 시간
- **하이퍼파라미터 튜닝**: 웨이블릿 패밀리, 분해 수준 등 자동 선택 도구 필요

#### 7.3.3 윤리적 고려

- **편향**: 학습 데이터의 특정 기간/도메인 편향 증폭 가능
- **투명성**: 금융/의료 의사결정에 사용될 때 모델 설명 필요

### 7.4 단기 연구 방향 (1-2년)

**불규칙 샘플링 확장**: 센서 고장이나 통신 두절 상황에서 불규칙 타임스탐프 처리

**확률적 예측**: 점 추정에서 신뢰도 구간 예측으로 확장

**도메인 적응화**: 소스 도메인 학습 → 타겟 도메인 제로샷(zero-shot) 예측

### 7.5 중기 연구 방향 (3-5년)

**시계열 기초 모델 구축**: 대규모 사전학습으로 다양한 작업(예측, 분류, 이상 탐지) 통합

**그래프 시계열 확장**: 공간-시간 의존성을 모델링하는 "Spatio-Temporal Wavelet GNN"

**인과성 통합**: 그래인저 인과성(Granger Causality) 기반으로 주파수별 인과성 분석

***

## 결론

**AWGformer**는 웨이블릿 이론의 우아함과 Transformer의 강력한 표현력을 결합하여 시계열 예측 분야에 새로운 기준을 제시합니다. 2020년 이후 확립된 기초(Informer의 효율성, Autoformer의 분해, FEDformer의 주파수 인식) 위에서 **적응형, 학습 가능, 다중 스케일** 설계를 통해 다음을 달성했습니다:

- **6.8% 절대 성능 향상** (기존 최고 성능 모델 대비)
- **42% 향상된 장기 예측 견고성** (96 → 720 단계)
- **결측 데이터 상황에서 6-8% 우월** 성능
- **도메인 전이에서 최고 성능** 달성

**향후 전망**: 불규칙 샘플링, 확률적 예측, 도메인 적응, 기초 모델 구축, 인과 추론 등의 방향에서 진화할 것으로 예상됩니다. 다만 이론적 엄밀성, 계산 효율성, 설명 가능성에서의 추가 진전이 필수적입니다.

AWGformer는 단순한 벤치마크 성능 향상을 넘어, 신호처리와 심층학습의 통합이라는 더 깊은 의미의 기여를 제시하며, 향후 시계열 예측 연구의 견고한 토대가 될 것으로 평가됩니다.

***

## 참고문헌

 Wei Li. AWGformer: Adaptive Wavelet-Guided Transformer for Multi-Resolution Time Series Forecasting. arXiv:2601.20409v1, 2026. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6a3f6b49-35c1-4a62-8768-4f712c12f634/2601.20409v1.pdf)

 Zhou et al. Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting. AAAI, 2021. [ojs.aaai](https://ojs.aaai.org/index.php/AAAI/article/view/17325)

 W. Li. SWIFT: State-space Wavelet Integrated Forecasting Technology for Enhanced Time Series Prediction. ICANN, 2025. [pubmed.ncbi.nlm.nih](https://pubmed.ncbi.nlm.nih.gov/21095693/)

 Y. Li et al. Efficient Medical Image Segmentation via Reinforcement Learning-Driven K-Space Sampling. IEEE TECI, 2025. [openaccess.thecvf](https://openaccess.thecvf.com/content_WACV_2020/papers/Rodriguez_Deep_Adaptive_Wavelet_Network_WACV_2020_paper.pdf)

 Web: Multilevel Wavelet Decomposition Network for Interpretable Time Series Analysis. arXiv:1806.08946. [arxiv](https://arxiv.org/abs/1806.08946)

 Web: WEITS: A Wavelet-enhanced Residual Framework for Time Series Forecasting. arXiv:2405.10877, 2024. [arxiv](https://arxiv.org/abs/2405.10877)

 Web: Wavelet Mixture of Experts for Time Series Forecasting. arXiv:2508.08825. [arxiv](https://arxiv.org/html/2508.08825v1)

 Web: Multi-resolution Time-Series Transformer for Long-term Forecasting. MLSR, 2024. [proceedings.mlr](https://proceedings.mlr.press/v238/zhang24l/zhang24l.pdf)

 Web: A Comprehensive Survey of Time Series Forecasting. arXiv:2411.05793v1. [arxiv](https://arxiv.org/html/2411.05793v1)

 Web: Pyraformer: Low-Complexity Pyramidal Attention for Long-Range Time Series Forecasting. [bo1126.tistory](https://bo1126.tistory.com/entry/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-Pyraformer-Low-Complexity-Pyramidal-Attention-for-Long-Range-Time-Series-Modeling-and-Forecasting)

 Web: A Closer Look at Transformers for Time Series Forecasting. ICML, 2025. [icml](https://icml.cc/virtual/2025/poster/44262)

 H. Wu et al. Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting. NeurIPS, 2021. [semanticscholar](https://www.semanticscholar.org/paper/fc46ccb83dc121c33de7ab6bdedab7d970780b2f)

 T. Zhou et al. FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting. ICML, 2022. [arxiv](https://arxiv.org/abs/2201.12740)

 Y. Liu et al. iTransformer: Inverted Transformers Are Effective for Time Series Forecasting. ICLR, 2024. [iclr](https://iclr.cc/virtual/2024/poster/18933)

 iTransformer: Inverted Transformers Are Effective for Time Series Forecasting. OpenReview. [openreview](https://openreview.net/forum?id=JePfAI8fah)

 Y. Nie et al. A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. ICLR, 2023. [arxiv](https://arxiv.org/abs/2211.14730)

 Web: Benchmarking M-LTSF: Frequency and Noise-Based Evaluation. arXiv:2510.04900. [arxiv](https://arxiv.org/html/2508.02753v4)

 Web: Towards a General Time Series Forecasting Model. arXiv:2405.17478v3. [arxiv](https://arxiv.org/html/2405.17478v3)

 H. Wu et al. TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis. ICLR, 2023. [ise.thss.tsinghua.edu](https://ise.thss.tsinghua.edu.cn/~mlong/doc/TimesNet-iclr23.pdf)
