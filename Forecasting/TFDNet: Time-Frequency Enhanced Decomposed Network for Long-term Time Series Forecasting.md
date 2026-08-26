# TFDNet: Time-Frequency Enhanced Decomposed Network for Long-term Time Series Forecasting

> **⚠️ 투명성 고지**: 본 논문은 arXiv preprint(2308.13386v1, 2023년 8월 25일)로, 동료 심사(peer review)를 아직 통과하지 않은 상태입니다. 일부 수치 및 비교는 이 점을 감안하여 해석해야 합니다.

---

## 1. Executive Summary (10문장 이내)

TFDNet(Time-Frequency Enhanced Decomposed Network)은 장기 시계열 예측(Long-term Time Series Forecasting)을 위해 시간 도메인과 주파수 도메인을 **동시에** 활용하는 딥러닝 모델이다.  
기존 방법들이 시간 또는 주파수 영역 중 하나만 처리한 한계를 극복하기 위해, STFT(Short-Time Fourier Transform)를 활용하여 시계열을 시간-주파수 행렬로 변환한다.  
모델은 트렌드(trend)와 계절(seasonal) 성분을 분리하여 각각 별도의 Time-Frequency Block(TFB)으로 처리하는 이중 분기 구조를 가진다.  
계절 성분의 채널 간 상관관계 패턴에 따라 개별 커널(IK) 또는 다중 공유 커널(MK) 전략을 선택적으로 적용한다.  
다중 스케일 STFT 윈도우($S = \{8, 16, 32\}$)를 통해 다양한 해상도에서 시간-주파수 정보를 포착한다.  
Mixture Loss(L1+L2 결합)를 적용하여 이상치(outlier)에 대한 예측 견고성을 향상시켰다.  
8개 벤치마크 데이터셋 실험에서 최우수 채널 독립(CI) 모델인 PatchTST 대비 MSE 3~3.6% 감소, 최우수 채널 혼합(CM) 모델인 FEDformer 대비 MSE 32% 이상 감소를 달성했다.  
연산 복잡도 $\mathcal{O}(MN^2)$ ($N \ll L, M \ll L$)로 PatchTST 대비 메모리 최대 9.4배, 학습 속도 최대 4.63배 향상되었다.  
모델 선택(IK vs. MK)이 채널 간 상관관계 분석에 의존하여 자동화가 어렵다는 한계가 존재한다. 이론적 수렴 보장 및 모델 선택 자동화는 향후 연구 과제로 남겨졌다.

---

### 1-1. 연구의 목적과 필요성

| 문제 | 설명 |
|------|------|
| **단일 도메인 처리의 한계** | 기존 모델(Autoformer, PatchTST 등)은 시간 도메인 *또는* 주파수 도메인 중 하나만 활용 (p.2) |
| **채널 간 상관관계 미고려** | 채널 독립(CI) 전략에서 채널 간 상관관계를 반영하지 않고 동일한 파라미터를 공유 (p.2) |
| **트랜스포머의 비효율성** | 트랜스포머 기반 모델의 높은 연산 복잡도와 메모리 요구량 (p.1) |

> 📌 **용어 설명**
> - **Long-term Time Series Forecasting**: 수백~수천 스텝 이후의 미래 값을 예측하는 문제. 단기 예측보다 어려운 이유는 오차가 누적되고, 장기 의존성(long-range dependency)을 학습해야 하기 때문
> - **채널(Channel)**: 다변량 시계열에서 하나의 변수(예: 온도, 습도 등 각각이 하나의 채널)

---

## 2. 핵심 주장과 근거 표

| # | 핵심 주장 | 근거/방법 | 위치 |
|---|-----------|-----------|------|
| 1 | 시간-주파수 통합 처리가 단일 도메인보다 우수 | FreqNet, TimeNet 대비 TFDNet 성능 향상 (Table 3) | p.9, Table 3 |
| 2 | 트렌드/계절 분리 처리가 효과적 | TFNet-T(분리 없음) 대비 TFDNet 성능 우위 (Table 3) | p.9, Table 3 |
| 3 | 채널 간 상관관계에 따른 커널 선택이 중요 | IK vs. MK의 데이터셋별 성능 차이 (Table 1) | p.8, Table 1 |
| 4 | 다중 스케일 STFT가 단일 스케일보다 우수 (데이터 의존적) | Table 10: Traffic, Electricity에서 개선, ETTm2는 미비 | p.16, Table 10 |
| 5 | Mixture Loss가 L2 단독보다 견고함 | Table 9: MAE에서 일관된 개선 | p.15, Table 9 |
| 6 | 기존 SOTA 대비 연산 효율성 우위 | 메모리 9.4×↓, 속도 4.63×↑ (Table 4) | p.10, Table 4 |

---

## 2-1. 상세 설명

### 🔴 해결하고자 하는 문제

1. **시간-주파수 정보의 미통합**: 시간 도메인(temporal correlation, periodicity)과 주파수 도메인(global patterns, change patterns)을 동시에 활용하지 못함
2. **채널 간 상관관계 미반영**: CI 전략에서 모든 채널에 동일한 파라미터를 적용하여 채널별 특성 무시
3. **트렌드/계절 성분의 패턴 차이 미고려**: 두 성분의 근본적으로 다른 패턴을 동일한 방식으로 처리

---

### 🟢 제안하는 방법 (수식 포함)

#### Phase 1: 전처리 및 분해 (Eq. 1, p.4)

$$\mathbf{X}_{tr} = \text{AvgPool}(\text{Padding}(\mathbf{X}))$$
$$\mathbf{X}_{se} = \mathbf{X} - \mathbf{X}_{tr}$$

| 기호 | 설명 |
|------|------|
| $\mathbf{X} \in \mathbb{R}^{L \times D}$ | RevIN 정규화 후 입력 시계열 ($L$: 과거 윈도우 길이, $D$: 채널 수) |
| $\mathbf{X}_{tr}$ | 이동 평균으로 추출된 **트렌드 성분** |
| $\mathbf{X}_{se}$ | 원 시계열에서 트렌드를 뺀 **계절 성분** |
| $\text{AvgPool}$ | 이동 평균 연산(주기적 숨겨진 변수를 평활화) |

> 📌 **용어 설명**
> - **RevIN(Reversible Instance Normalization)**: 훈련/테스트 데이터 간 분포 이동(distribution shift)을 완화하기 위해 정규화 후 역정규화하는 기법. 각 채널을 평균 0, 표준편차 1로 정규화
> - **트렌드-계절 분해(Trend-Seasonal Decomposition)**: 시계열을 장기 추세(trend)와 반복 패턴(seasonal)으로 분리하는 신호처리 기법

#### Phase 2: 다중 스케일 시간-주파수 인코더 (Eq. 2, p.5)

$$\mathbf{Z}_{se} = \text{Linear}(\text{SeasonEncoder}(\mathbf{X}_{se}, S_1), \ldots, \text{SeasonEncoder}(\mathbf{X}_{se}, S_s))$$

$$\mathbf{Z}_{tr} = \text{Linear}(\text{TrendEncoder}(\mathbf{X}_{tr}, S_1), \ldots, \text{TrendEncoder}(\mathbf{X}_{tr}, S_s))$$

$$\mathbf{Z} = \mathbf{Z}_{se} + \mathbf{Z}_{tr}$$

| 기호 | 설명 |
|------|------|
| $S_1, \ldots, S_s$ | 다중 STFT 윈도우 크기 집합 (기본값: $\{8, 16, 32\}$) |
| $\mathbf{Z}\_{se}, \mathbf{Z}_{tr} \in \mathbb{R}^{D \times L}$ | 계절/트렌드 융합 표현 |
| $\mathbf{Z}$ | 최종 인코더 출력 (두 표현의 합) |

#### Phase 3: STFT 기반 인코더 구조 (Eq. 3, p.5)

$$\tilde{\mathbf{X}} = \text{STFT}(\mathbf{X})$$
$$\mathbf{Q} = \text{TFB}(\tilde{\mathbf{X}})$$
$$\tilde{\mathbf{Z}} = \mathbf{Q} + \text{Frequency-FFN}(\mathbf{Q})$$
$$\mathbf{Z} = \text{STFT}^{-1}(\tilde{\mathbf{Z}})$$

#### STFT 수식 (Eq. 4, p.5)

$$\tilde{\mathbf{X}} = \sum_{m=0}^{S-1} \text{Window}[m]\mathbf{X}[m + nl]e^{-j\frac{2\pi m\omega}{S}}$$

| 기호 | 설명 |
|------|------|
| $S$ | STFT 슬라이딩 윈도우 크기 |
| $m$ | 윈도우 내 샘플 인덱스 |
| $n$ | 시간 프레임 인덱스 |
| $l$ | 슬라이딩 스트라이드(stride) |
| $\omega$ | 주파수 빈(frequency bin) 인덱스 |
| $\tilde{\mathbf{X}} \in \mathbb{C}^{D \times M \times N}$ | STFT 출력 시간-주파수 행렬 |
| $M = \frac{S}{2} + 1$ | 켤레 대칭에 의한 주파수 성분 수 |
| $N = \frac{L}{l} + 1$ | 시간 프레임 수 |

> 📌 **용어 설명**
> - **STFT(Short-Time Fourier Transform, 단시간 푸리에 변환)**: 시간에 따라 변화하는 신호의 주파수 특성을 분석하기 위해, 신호를 짧은 구간으로 나눠 각 구간에 푸리에 변환을 적용하는 방법. 결과는 "어느 시점에 어떤 주파수가 강한가"를 보여주는 2D 행렬
> - **주파수 빈(Frequency Bin)**: 이산 푸리에 변환 결과에서 특정 주파수 범위에 해당하는 단위

#### TFB 커널 연산 (Eq. 5, p.6)

$$\text{Kernel}(\tilde{\mathbf{X}}^{(i)}, \mathbf{W}) = \tilde{\mathbf{X}}^{(i)}_m \cdot \mathbf{W}_m$$

| 기호 | 설명 |
|------|------|
| $\tilde{\mathbf{X}}^{(i)} \in \mathbb{C}^{M \times N}$ | $i$번째 채널의 STFT 출력 |
| $\mathbf{W} \in \mathbb{C}^{M \times N \times N}$ | 커널 가중치 행렬 |
| $\tilde{\mathbf{X}}^{(i)}_m \in \mathbb{C}^N$ | $m$번째 주파수 빈 벡터 |
| $\mathbf{W}_m \in \mathbb{C}^{N \times N}$ | $m$번째 주파수 빈에 대한 가중치 |

#### Mixture Loss (Eq. 6, p.7)

$$\mathcal{L} = \sum_{i=1}^{D} \alpha|\hat{\mathcal{Y}}_i - \mathcal{Y}_i| + (1-\alpha)\|\hat{\mathcal{Y}}_i - \mathcal{Y}_i\|_2$$
$$\alpha = \text{Tanh}(|\hat{\mathcal{Y}}_i - \mathcal{Y}_i|)$$

| 기호 | 설명 |
|------|------|
| $\hat{\mathcal{Y}}_i$ | $i$번째 채널의 예측값 |
| $\mathcal{Y}_i$ | $i$번째 채널의 실제값 |
| $\alpha \in (0, 1)$ | L1/L2 가중치 조절 계수 (오차가 클수록 L1 비중 증가) |

> 📌 **용어 설명**
> - **L1 Loss(MAE)**: 예측 오차의 절댓값 합. 이상치(outlier)에 덜 민감하여 견고한(robust) 학습에 유리
> - **L2 Loss(MSE)**: 예측 오차의 제곱합. 오차가 작을 때 섬세한 학습에 유리하지만 이상치에 민감

---

### 🔵 모델 구조 (Figure 2 기반)

```
입력 시계열 X
    ↓ RevIN 정규화 + 트렌드-계절 분해
    ├── 계절 성분 Xse → SeasonalEncoder (Seasonal-TFB-IK 또는 MK) × S개 스케일
    └── 트렌드 성분 Xtr → TrendEncoder (Trend-TFB, 단일 공유 커널) × S개 스케일
    ↓ Linear 융합 → Z = Zse + Ztr
    ↓ Linear Projection + RevIN 역정규화
출력 예측 X̂ ∈ R^{T×D}
```

**각 인코더 내부 구조**:
$$\mathbf{X} \xrightarrow{\text{STFT}} \tilde{\mathbf{X}} \xrightarrow{\text{TFB}} \mathbf{Q} \xrightarrow{+\text{Freq-FFN}} \tilde{\mathbf{Z}} \xrightarrow{\text{STFT}^{-1}} \mathbf{Z}$$

> 📌 **용어 설명**
> - **채널 독립(Channel-Independence, CI)**: 각 채널을 독립적으로 처리하는 전략. 다른 채널 정보를 사용하지 않음
> - **채널 혼합(Channel-Mixing, CM)**: 모든 채널을 동시에 처리하여 채널 간 상호작용을 학습하는 전략
> - **저차원 근사(Low-rank Approximation)**: 큰 행렬을 두 개의 작은 행렬의 곱으로 근사하는 기법. $\mathbf{W}_{ind} = \mathbf{W}_1 \cdot \mathbf{W}_2$로 파라미터 수를 줄임

---

### 🟠 성능 향상 및 한계

**성능 향상** (Table 1, p.8):
- vs. FEDformer(최고 CM 모델): MSE 32.3~32.7%↓, MAE 22.0~22.4%↓
- vs. PatchTST(최고 CI 모델): MSE 3.0~3.6%↓, MAE 1.4~4.7%↓
- 연산 효율: 메모리 최대 9.4×↓, 속도 최대 4.63×↑ (Table 4, p.10)

**한계** (p.10, Conclusion):
- IK vs. MK 모델 선택이 수동(채널 간 상관관계 사전 분석 필요)
- 자동 모델 선택 메커니즘 부재
- 이론적 수렴 분석 미비
- 다중 스케일 전략이 모든 데이터셋에서 일관되게 효과적이지 않음 (ETTm2, Table 10)

---

## 3. 각 주장에 페이지/Figure/Table 번호 표시

| 주장 | 위치 |
|------|------|
| 시간-주파수 통합의 필요성 | p.2 (Introduction 2단락) |
| STFT 기반 시간-주파수 변환 | p.5, Eq. 4 |
| 채널 상관관계의 데이터별 차이 | p.2~3, Figure 1 |
| 트렌드/계절 분해 구조 | p.4, Eq. 1, Figure 2 |
| IK vs. MK 커널 전략 | p.6, Figure 3 |
| Mixture Loss 설계 | p.7, Eq. 6 |
| 다변량 예측 SOTA 우위 | p.8~9, Table 1 |
| 단변량 예측 SOTA 우위 | p.9, Table 2 |
| 분해 구조 절제 실험 | p.9, Table 3 |
| 연산 효율성 비교 | p.10, Table 4 |
| 오차 막대(Error bar) | Appendix B, Table 6 |
| 다중 스케일 효과 분석 | Appendix D.2, Table 10, Figure 4 |
| Mixture Loss 효과 | Appendix D.1, Table 9 |

---

## 4. 저자 보고 결과 vs. 내 해석 분리

### 저자가 직접 보고한 결과

**연구 주제** (Abstract, p.1):
> "장기 시계열 예측을 위해 시간-주파수 도메인에서 장기 패턴과 시간적 주기성을 동시에 포착하는 TFDNet을 제안한다"

**방법** (수식):
- Eq. 1~6: 위 2-1절 참조

**결과** (저자 직접 보고, Table 1, p.8):
- TFDNet-IK: FEDformer 대비 MSE 32.3%↓, PatchTST 대비 MSE 3.0%↓
- TFDNet-MK: FEDformer 대비 MSE 32.7%↓, PatchTST 대비 MSE 3.6%↓
- 메모리: PatchTST($L=720$) 13,892 MB → TFDNet-IK 2,046 MB
- 속도: PatchTST($L=720$) 635 s/epoch → TFDNet-IK 222 s/epoch

**한계** (저자 직접 인정, p.10):
> "모델 선택은 주로 채널 간 상관관계 효과에 기반하며, 추가 조사 및 이론적 연구가 필요하다"

---

### 내 해석 (⚠️ 저자 의견 아님)

1. **PatchTST 대비 개선폭(3~3.6%)이 상대적으로 작음**: 실질적 유의미성을 판단하기 위한 통계적 유의성 검정(예: t-test)이 제시되지 않아, 이 차이가 우연에 의한 것인지 단정하기 어려움

2. **모델 선택 문제가 실용성을 제한**: IK/MK 선택을 위해 사전에 채널 상관관계를 분석해야 하는 것은 실제 배포 환경에서 추가 비용을 의미함

3. **단일 GPU 환경 실험**: 분산 학습 환경에서의 성능은 검증되지 않았으며, 실제 산업 환경과 다를 수 있음

4. **Preprint 상태**: 동료 심사 미완료로 결과의 재현성 및 방법론적 타당성이 공식 검증되지 않음

5. **데이터셋 편향 가능성**: 8개 데이터셋 모두 특정 도메인(에너지, 교통, 날씨, 의료)에 집중되어 있어, 금융, 제조 등 다른 도메인에서의 일반화 성능은 미지수

---

## 5. 통계적으로 취약한 부분과 비교 불가능한 수치

| ⚠️ 취약점 | 상세 |
|-----------|------|
| **통계적 유의성 검정 부재** | 3회 반복 실험 후 평균만 보고. PatchTST 대비 3% 개선이 유의한지 t-test 등이 없음 (Table 1) |
| **Error bar 일부만 보고** | Table 6에서 TFDNet과 PatchTST, FiLM만 비교. DLinear, FEDformer, Autoformer의 분산은 미보고 |
| **Autoformer 데이터 로더 버그** | 각주 1: "Autoformer의 초기 데이터 로더 구현이 마지막 배치를 버려 결과를 과대평가할 수 있음. 본 논문은 수정 후 결과 보고" → 타 논문 Autoformer 수치와 직접 비교 불가 ⚠️ |
| **ILI 데이터셋 변동성 큼** | Table 6: TFDNet-MK의 ILI 48시간 예측 MAE STD = ±0.0375로 상대적으로 큰 변동 |
| **효율성 비교의 단일 조건** | Table 4: Electricity 데이터셋, 배치 크기 8, T=720 고정 조건만 비교. 다른 조건에서의 결과 미보고 |
| **모델 선택 기준의 비체계성** | IK vs. MK 선택이 Figure 1의 시각적 상관관계 분석에 의존. 정량적 자동 선택 기준 없음 |
| **Weather 720 예측의 PatchTST 우위** | Table 1: Weather T=720에서 PatchTST(MSE=0.307) < TFDNet-IK(MSE=0.314). 저자는 이를 명시적으로 언급하지 않음 ⚠️ |

> 📌 **용어 설명**
> - **통계적 유의성 검정(Statistical Significance Test)**: 두 결과의 차이가 우연에 의한 것인지 아닌지를 확률적으로 판단하는 방법. p-value < 0.05이면 통계적으로 유의하다고 판단

---

## 6. 문서가 답하지 않는 질문

| # | 미답변 질문 |
|---|------------|
| 1 | IK vs. MK 모델 선택을 **자동화**할 수 있는 기준은 무엇인가? (채널 상관관계 임계값 등) |
| 2 | STFT 외 다른 시간-주파수 변환(예: Wavelet Transform, Hilbert-Huang Transform)과의 비교는? |
| 3 | 모델의 **이론적 수렴 보장**은 존재하는가? |
| 4 | 금융, 제조, IoT 등 **다른 도메인** 데이터에서의 성능은? |
| 5 | 노이즈 수준이 다른 데이터에서 Mixture Loss의 $\alpha$ 조절이 충분한가? |
| 6 | 하이퍼파라미터($S, l$, 배치 크기 등) 선택의 이론적 근거는 무엇인가? |
| 7 | 분산 학습(multi-GPU) 환경에서의 확장성(scalability)은? |
| 8 | 예측 불확실성(Uncertainty Quantification)은 어떻게 처리하는가? |
| 9 | 실시간(online) 학습 또는 지속 학습(continual learning) 시나리오 적용 가능성은? |
| 10 | 계절 성분의 채널 상관관계가 시간에 따라 변화할 경우 모델이 이를 어떻게 처리하는가? |

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p.2): 트렌드-계절 분해 시각화

**내용**: ETTm2(에너지)와 Traffic(교통) 데이터의 2개 채널에 대한 원시 신호 및 분해된 계절/트렌드 성분 비교

**해석**:
- **트렌드 성분**: ETTm2와 Traffic 모두 채널 간 명확한 관계 존재 → 공유 커널(shared kernel) 적용 근거
- **계절 성분**: ETTm2는 채널 간 상관관계 낮음, Traffic은 채널 간 매우 높은 상관관계 → IK(개별 커널)/MK(다중 공유 커널) 분리 설계의 핵심 동기
- **연구적 의의**: 단순히 단일 모델을 적용하는 것보다 데이터 특성에 맞춘 설계가 필요하다는 경험적 근거를 시각적으로 제시

> 📌 **용어 설명**
> - **계절 성분(Seasonal Component)**: 일정한 주기로 반복되는 패턴(예: 매일 출퇴근 시간대 교통량 증가)
> - **트렌드 성분(Trend Component)**: 장기적인 증가/감소 방향성(예: 연간 전력 소비 증가 추세)

---

### Figure 2 (p.4): TFDNet 전체 아키텍처

**내용**: TFDNet의 이중 분기(seasonal + trend) 구조, 인코더 내부의 STFT→TFB→Freq-FFN→ISTFT 흐름, 다중 스케일 윈도우 적용

**해석**:
- 좌측: 시간 도메인 입력/출력, 우측: 각 인코더의 상세 처리 흐름
- 두 분기가 **공통 백본 구조**(STFT+TFB+Freq-FFN+ISTFT)를 공유하되, TFB만 다름
- 다중 스케일 출력을 Linear로 융합하여 단일 표현 $\mathbf{Z}$로 통합
- **설계 철학**: 모듈성(modularity)을 유지하면서 트렌드/계절 특성에 맞게 특화

---

### Figure 3 (p.7): Time-Frequency Block(TFB) 상세 구조

**내용**: Trend-TFB(단일 공유 커널), Seasonal-TFB-IK(개별 커널), Seasonal-TFB-MK(다중 공유 커널 + 게이트 레이어)

**해석**:
- **Trend-TFB**: 하나의 커널 $\mathbf{W}_{tr}$을 모든 채널에 공유 → 단순하고 효율적
- **Seasonal-TFB-IK**: 채널별 독립 커널 $\mathbf{W}_{ind}$, 저차원 근사로 확장성 확보 → 낮은 채널 상관 데이터에 적합
- **Seasonal-TFB-MK**: $k$개의 공유 커널 + 게이트 벡터로 적응적 융합 → 높은 채널 상관 데이터에 적합
- 3가지 설계가 Figure 1의 관찰을 직접적으로 모델 구조로 구현한 핵심 기여

> 📌 **용어 설명**
> - **게이트 레이어(Gate Layer)**: 여러 정보 소스의 중요도를 학습하여 동적으로 가중합하는 메커니즘. LSTM의 게이트와 유사한 개념

---

### Figure 4 (p.16): STFT 시간-주파수 맵

**내용**: ETTm2와 Traffic 데이터의 STFT 출력(윈도우 크기 $\{8, 16, 32\}$별)

**해석**:
- **ETTm2**: 에너지가 최저 주파수 성분에 집중 → 다중 스케일이 추가 정보를 제공하지 못함 (Table 10의 ETTm2 결과와 일치)
- **Traffic**: 에너지가 다양한 주파수에 분산 → 다중 스케일이 서로 다른 해상도의 정보를 포착하여 성능 향상
- **연구적 의의**: 다중 스케일 전략의 효과가 데이터 의존적임을 **설명 가능한 방식**(explainable)으로 제시
- 윈도우가 작을수록($S=8$) 시간 해상도↑, 주파수 해상도↓; 클수록 반대

---

### Table 1 (p.8): 다변량 예측 성능 종합 비교

**내용**: 8개 데이터셋, 4개 예측 지평선에서 8개 모델 MSE/MAE 비교

**해석**:
- TFDNet이 대부분의 조건에서 최고 성능이나, **모든 경우에서 우세하지는 않음**:
  - Weather T=720: PatchTST(0.307) < TFDNet-IK(0.314) ⚠️
  - ETTm2 T=720: TFDNet-IK(0.345)와 PatchTST(0.353) 차이 0.008로 매우 미세
- ILI 데이터셋: 불규칙한 패턴으로 모든 모델의 절대적 오차(MSE>1.7)가 크고 성능 차이도 상대적으로 큼
- Transformer 계열(Informer, Autoformer, FEDformer) 대비 압도적 우위는 선형/단순 모델 계열(DLinear, PatchTST)의 강력함을 재확인

---

## 8. 결론, 후속 연구 계획 및 추가 방향

### 저자가 제시한 시사점 및 후속 연구 계획 (p.10)

| 항목 | 내용 |
|------|------|
| **주요 시사점** | 시간-주파수 통합 처리가 단일 도메인보다 효과적; 채널 간 상관관계 고려가 중요 |
| **후속 연구 계획** | IK/MK 모델 선택의 자동화; 채널 간 상관관계에 대한 이론적 연구 |

---

### 8-1. 모델의 일반화 성능 향상 가능성

#### 현재 일반화의 한계

1. **모델 선택 의존성**: IK/MK 선택이 수동 분석에 의존. 새로운 데이터에 자동 적용 불가
2. **8개 데이터셋, 5개 도메인 한정**: 금융, 제조, 바이오 등 미검증 도메인
3. **고정 하이퍼파라미터**: $S = \{8, 16, 32\}$, $l = \{4, 8, 16\}$이 모든 데이터에 동일 적용
4. **분포 이동(Distribution Shift)**: RevIN으로 완화하지만, 급격한 비정상성(non-stationarity)에 대한 충분한 대응 미검증

#### 일반화 향상을 위한 제안 방향

| 방향 | 구체적 방법 |
|------|------------|
| **자동 채널 상관관계 기반 모델 선택** | 채널 상관계수 행렬을 입력으로 IK/MK를 자동 선택하는 메타 학습(meta-learning) 도입 |
| **적응형 윈도우 학습** | 고정 STFT 윈도우 대신 데이터에서 최적 윈도우를 학습하는 신경 STFT 또는 학습 가능한 웨이블릿 필터 사용 |
| **도메인 일반화(Domain Generalization)** | 여러 도메인의 데이터로 사전 학습 후 파인튜닝(fine-tuning) 전략 적용 |
| **사전 학습 기반 접근** | Time-Series Foundation Model(예: TimesFM, Chronos)의 프레임워크와 통합 |
| **불확실성 정량화** | 베이지안 또는 앙상블 방법으로 예측 신뢰 구간 제공 → 실제 배포 신뢰성 향상 |

> 📌 **용어 설명**
> - **도메인 일반화(Domain Generalization)**: 훈련 데이터와 다른 분포를 가진 새로운 도메인에서도 잘 작동하도록 모델을 훈련하는 방법
> - **메타 학습(Meta-Learning)**: "학습하는 방법을 학습"하는 기법. 새로운 태스크에 빠르게 적응할 수 있는 모델 초기화 또는 선택 알고리즘을 학습

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 아래 비교는 논문 내 인용 및 공개 정보에 기반합니다. 직접 접근하지 않은 논문의 세부 수치는 포함하지 않습니다.

#### 시계열 예측 패러다임 변화 (2020~2023)

| 연도 | 모델 | 핵심 접근 | TFDNet과의 관계 |
|------|------|-----------|----------------|
| 2021 | **Autoformer** (Wu et al.) | 자동 상관 + 트렌드 분해 | TFDNet의 분해 블록 기반, TFDNet이 성능 개선 |
| 2021 | **Informer** (Zhou et al.) | ProbSparse Attention | 트랜스포머 복잡도 감소 시도, TFDNet이 더 효율적 |
| 2022 | **FEDformer** (Zhou et al.) | 푸리에 강화 트랜스포머 | 주파수 도메인 활용, TFDNet 대비 성능·효율 열세 |
| 2022 | **FiLM** (Zhou et al.) | Legendre 다항식 + 푸리에 | 주파수 도메인만 활용, TFDNet이 시간-주파수 통합으로 우위 |
| 2022 | **DLinear** (Zeng et al.) | 단순 선형 모델 | 트랜스포머의 비효율성을 지적, 단순성의 중요성 강조 |
| 2022 | **PatchTST** (Nie et al.) | 패치 기반 CI 트랜스포머 | TFDNet의 가장 강력한 경쟁 모델, 3% 차이 |
| 2022 | **Non-stationary Transformer** (Liu et al.) | 비정상 시계열 처리 | TFDNet에서 RevIN으로 일부 대응 |
| 2022 | **TimesNet** (Wu et al.) | 2D 시간 변동 모델링 | 2D 구조 활용의 유사성, 직접 비교 없음 |
| 2023 | **MICN** (Wang et al.) | 다중 스케일 지역/전역 맥락 | 다중 스케일 개념 공유, TFDNet의 STFT 기반 접근과 상보적 |

#### TFDNet이 앞으로의 연구에 미치는 영향

1. **시간-주파수 통합의 주류화**: STFT를 딥러닝 시계열 모델에 통합하는 방향을 제시. 향후 연구에서 Wavelet, CWT(Continuous Wavelet Transform) 등 다른 시간-주파수 변환 활용 연구 촉진 예상

2. **채널 간 상관관계 설계 원칙 제시**: "데이터 특성에 따라 채널 처리 전략을 다르게 해야 한다"는 원칙을 실증적으로 보여줌. 향후 적응형 채널 처리 연구의 기반

3. **효율적 CI 모델의 가능성 확장**: CI 전략이 트랜스포머보다 효율적이면서도 주파수 정보를 활용할 수 있음을 보여줌

#### 향후 연구 시 고려할 점

| 고려 사항 | 세부 내용 |
|-----------|-----------|
| **Foundation Model과의 통합** | TimesFM(Google), Chronos(Amazon) 등 대형 시계열 기반 모델이 등장하면서, TFDNet의 STFT 기반 특징 추출기를 사전 학습 모델의 인코더로 활용하는 방향 검토 필요 |
| **자동화된 아키텍처 탐색** | Neural Architecture Search(NAS)를 활용하여 IK/MK 선택, 스케일 수, 커널 수 등을 자동 결정하는 연구 |
| **비정상 시계열 강건성** | 급격한 분포 변화(concept drift)에 대한 온라인 적응 메커니즘 추가 필요 |
| **해석 가능성(Explainability)** | STFT 시간-주파수 맵을 활용한 예측 근거 시각화는 의료·금융 분야 적용에 필수 |
| **멀티모달 통합** | 텍스트, 이미지 등 다른 모달리티와 시계열을 결합하는 멀티모달 시계열 예측으로 확장 가능성 |
| **재현성 검증** | Preprint 상태인 만큼, 독립적인 재현 실험 및 다양한 시드(seed)에서의 안정성 검증 필요 |

> 📌 **용어 설명**
> - **Foundation Model(기반 모델)**: 대규모 데이터로 사전 학습된 후 다양한 하위 태스크에 적용 가능한 범용 모델(예: GPT, BERT의 시계열 버전)
> - **Concept Drift**: 시간이 지남에 따라 데이터의 통계적 특성이 변화하는 현상. 예: COVID-19 이후 교통 패턴의 급격한 변화
> - **NAS(Neural Architecture Search)**: 신경망의 최적 구조를 자동으로 탐색하는 방법

---

## 📚 참고자료

**본 논문**:
- Luo, Y., Lyu, Z., & Huang, X. (2023). *TFDNet: Time-Frequency Enhanced Decomposed Network for Long-term Time Series Forecasting*. arXiv:2308.13386v1.

**본 논문 내 인용 문헌 (직접 확인된 것)**:
- Wu et al. (2021). *Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting*. NeurIPS 2021.
- Zhou et al. (2021). *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting*. AAAI 2021.
- Zhou et al. (2022a). *FiLM: Frequency Improved Legendre Memory for Long-Term Time Series Forecasting*. NeurIPS 2022.
- Zhou et al. (2022b). *FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting*. ICML 2022.
- Nie et al. (2022). *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*. arXiv:2211.14730.
- Zeng et al. (2022). *Are Transformers Effective for Time Series Forecasting?* arXiv:2205.13504.
- Kim et al. (2021). *Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift*. ICLR 2021.
- Boashash, B. (2015). *Time-Frequency Signal Analysis and Processing: A Comprehensive Reference*. Academic Press.
- Wang et al. (2023). *MICN: Multi-scale Local and Global Context Modeling for Long-term Series Forecasting*. ICLR 2023.
- Wu et al. (2022). *TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis*. arXiv:2210.02186.
- Han et al. (2023). *The Capacity and Robustness Trade-off: Revisiting the Channel Independent Strategy for Multivariate Time Series Forecasting*. arXiv:2304.05206.
- Kingma & Ba (2014). *Adam: A Method for Stochastic Optimization*. arXiv:1412.6980.
- Paszke et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. NeurIPS 2019.
