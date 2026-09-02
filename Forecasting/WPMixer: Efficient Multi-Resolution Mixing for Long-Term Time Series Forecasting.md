# WPMixer: Efficient Multi-Resolution Mixing for Long-Term Time Series Forecasting

---

## 1. Executive Summary (10문장 이내)

WPMixer(Wavelet Patch Mixer)는 장기 시계열 예측을 위한 MLP 기반 모델로, 웨이블릿 분해·패칭·믹싱의 세 가지 핵심 요소를 결합한다.  
기존 MLP-Mixer 계열 모델(TimeMixer, TSMixer)이 시간 도메인에만 집중하는 한계를 극복하고자, 다중 레벨 이산 웨이블릿 변환(DWT)을 통해 시간·주파수 도메인 정보를 동시에 추출한다.  
각 웨이블릿 계수 시리즈(근사 계수 1개 + 세부 계수 m개)를 독립적인 해상도 브랜치에서 처리함으로써 정보 손실을 최소화한다.  
패칭(Patching)은 로컬 정보를 효율적으로 포착하고, 패치 믹서(Patch Mixer)와 임베딩 믹서(Embedding Mixer)는 글로벌 정보를 포착한다.  
ETTh1, ETTh2, ETTm1, ETTm2, Weather, Electricity, Traffic 등 7개 벤치마크 데이터셋에서 최신 기술 수준의 성능을 달성하였다.  
연산 효율 측면에서 동일 조건 비교 시 TimeMixer 대비 GFLOPs를 10분의 1 이하로 줄였다. SmoothL1 손실 함수와 Optuna 기반 하이퍼파라미터 최적화를 채택하여 훈련 안정성을 높였다.  
다중 무작위 시드 실험에서도 TimeMixer보다 낮은 표준편차를 보여 강건성이 입증되었다.  
이동 평균 기반 분해로는 포착하기 어려운 급격한 스파이크와 딥(dip)도 웨이블릿 분해를 통해 효과적으로 처리한다.  
본 연구는 MLP 기반 시계열 예측 모델의 새로운 기준점을 제시한다.

### 1-1. 연구의 목적과 필요성

| 구분 | 내용 |
|------|------|
| **배경** | 날씨, 전력, 금융 등 다양한 분야에서 장기 시계열 예측의 수요가 급증 |
| **기존 한계 ①** | Transformer 계열 모델(Informer, Autoformer 등)은 계산 비용이 크고, 단순 선형 모델에도 성능이 역전되는 사례 존재 (Zeng et al. 2023) |
| **기존 한계 ②** | TimeMixer의 이동 평균 기반 분해는 복잡한 계절성 패턴과 급격한 스파이크/딥 포착 불가 (Hyndman et al. 2011) |
| **기존 한계 ③** | TSMixer는 긴 look-back window 시 계산 비용이 과도하게 증가 |
| **기존 한계 ④** | SWformer(Sepformer 변형)는 단일 레벨 웨이블릿 변환만 사용하여 잠재력 미달 |
| **연구 목적** | 시간·주파수 도메인 정보를 모두 활용하면서 계산 효율적인 MLP 기반 장기 예측 모델 개발 |

> 💡 **용어 설명**
> - **이동 평균(Moving Average)**: 일정 구간의 데이터 평균값을 순차적으로 계산하여 추세를 파악하는 방법. 급격한 변화(스파이크/딥) 포착에 취약함.
> - **DWT(Discrete Wavelet Transform, 이산 웨이블릿 변환)**: 신호를 고주파(세부) 성분과 저주파(근사) 성분으로 반복 분해하는 수학적 변환. 시간·주파수 정보를 동시에 표현 가능.

---

## 2. 핵심 주장과 근거

### 핵심 주장 표

| # | 핵심 주장 | 근거 / 증거 | 위치 |
|---|-----------|-------------|------|
| ① | 다중 레벨 웨이블릿 분해가 이동 평균 분해보다 우수 | 급격한 스파이크/딥, 복잡한 계절성 포착 능력 이론적 논증 + Table 5 절제 연구 | p.1, Table 5 |
| ② | 독립 해상도 브랜치 처리로 정보 손실 최소화 | Table 5 Case I vs. VII~XII 비교 (믹서 제거 시 MSE 증가) | Table 5 |
| ③ | WPMixer가 SOTA MLP/Transformer 모델 대비 우수한 예측 성능 | 7개 데이터셋, 4개 예측 길이에서 최다 1위 횟수(29/26 MSE/MAE) | Table 2 |
| ④ | 계산 효율성: TimeMixer 대비 GFLOPs 10배 이상 절감 | ETTh1에서 WPMixer 0.210 vs TimeMixer 2.774 GFLOPs (T=96) | Table 3 |
| ⑤ | SmoothL1 손실 함수가 MSE 손실보다 예측 성능 향상 | Table 6 ETTm2/ETTh2 비교 | Table 6 |
| ⑥ | 임베딩 믹서 추가가 성능 향상에 기여 | Table 5 Case I vs. V, II vs. VI 비교 | Table 5 |
| ⑦ | 모델 강건성 (낮은 표준편차) | 3개 랜덤 시드 실험에서 TimeMixer보다 낮은 σ | Table 4 |
| ⑧ | 단변량 예측에서도 PatchTST를 능가 | ETT 4개 데이터셋 단변량 결과 | Table 10 |

---

### 2-1. 상세 설명

#### ① 해결하고자 하는 문제

- **시간 도메인 단독 처리의 한계**: 기존 MLP Mixer 모델들은 시간 도메인 정보만 활용
- **이동 평균 기반 분해의 취약성**: 복잡한 계절성, 스파이크, 딥 포착 불가
- **단일 레벨 웨이블릿의 한계**: SWformer처럼 1레벨만 사용 시 다해상도 특징 추출 불가
- **계산 비효율**: TSMixer의 긴 look-back window 처리 시 과도한 연산량

---

#### ② 제안하는 방법 (수식 포함)

**[Step 1] 다중 레벨 웨이블릿 분해 (Eq. 1)**

$$[\mathbf{X}_{A_m}, \mathbf{X}_{D_m}, \mathbf{X}_{D_{m-1}}, \ldots, \mathbf{X}_{D_1}] = \text{Decomp}(\underline{\mathbf{X}}^T_L, \psi, m) $$

- $\mathbf{X}_{A_m} \in \mathbb{R}^{C \times L_m}$: m번째 레벨의 **근사 계수** 시리즈 (저주파 성분)
- $\mathbf{X}_{D_i} \in \mathbb{R}^{C \times L_i}$: i번째 레벨의 **세부 계수** 시리즈 (고주파 성분)
- $m$: 분해 레벨 수 (하이퍼파라미터, 1~5)
- $\psi$: 웨이블릿 타입 (Daubechies, Symlets, Coiflets, Biorthogonal 계열 중 선택)
- $L_i$: i번째 분해 레벨의 계수 시리즈 길이
- $\underline{\mathbf{X}}^T_L$: RevIN 정규화된 입력 시계열 ($\in \mathbb{R}^{C \times L}$)

> 💡 **용어 설명**
> - **근사 계수(Approximation Coefficients)**: 저역 통과 필터 출력. 신호의 전반적인 형태(저주파 트렌드)를 담음.
> - **세부 계수(Detail Coefficients)**: 고역 통과 필터 출력. 신호의 급격한 변화(고주파 패턴)를 담음.

---

**[Step 2] 패칭 (Eq. 2)**

$$\mathbf{X}_{P_i} = \text{Patch}(\underline{\mathbf{X}}_{W_i}) \in \mathbb{R}^{C \times N_i \times P} $$

- $\mathbf{X}_{W_i}$: i번째 해상도 브랜치의 웨이블릿 계수 시리즈 (근사 또는 세부)
- $N_i = \frac{L_i - P}{S} + 2$: 패치의 수
- $P$: 패치 길이 (예: 16, 32, 48)
- $S$: 스트라이드, 비중첩 구간 길이 (예: 8, 16, 24)

> 💡 **용어 설명**
> - **패칭(Patching)**: 시계열을 겹치는 작은 조각(패치)으로 나누는 기법. 로컬 정보 포착과 연산 효율화에 도움.
> - **스트라이드(Stride)**: 패치 간 이동 간격. 스트라이드가 패치 길이보다 작으면 패치가 겹침(overlapping).

---

**[Step 3] 임베딩 (Eq. 3)**

$$\mathbf{X}_{d_i} = \text{Embedding}(\mathbf{X}_{P_i}) \in \mathbb{R}^{C \times N_i \times d} $$

- $d$: 임베딩 차원 (예: 16, 32, 128, 256)
- 선형 임베딩 레이어가 모든 변수(variate)에 공유됨

---

**[Step 4] 패치 믹서 (Eqs. 4–5)**

$$\mathbf{X}'_{d_i} = \mathcal{P}(BN(\mathbf{X}_{d_i})) \in \mathbb{R}^{d \times C \times N_i} $$

$$\mathbf{X}''_{d_i} = \mathcal{L}_2(\mathcal{G}(\mathcal{L}_1(\mathbf{X}'_{d_i}))) \in \mathbb{R}^{d \times C \times N_i} $$

- $BN(\cdot)$: 2D 배치 정규화
- $\mathcal{P}(\cdot)$: 차원 순열(Permutation) 연산
- $\mathcal{G}(\cdot)$: GELU 활성화 함수
- $\mathcal{L}_1: \mathbb{R}^{d \times C \times N_i} \to \mathbb{R}^{d \times C \times N_i \cdot t_f}$: 첫 번째 선형 레이어 ($t_f$: 확장 인자)
- $\mathcal{L}_2: \mathbb{R}^{d \times C \times N_i \cdot t_f} \to \mathbb{R}^{d \times C \times N_i}$: 두 번째 선형 레이어 (복원)

> 💡 **용어 설명**
> - **GELU(Gaussian Error Linear Unit)**: 딥러닝에서 널리 쓰이는 활성화 함수. ReLU보다 부드럽고 학습 안정성이 높음.
> - **배치 정규화(Batch Normalization)**: 미니배치 단위로 입력을 정규화하여 학습을 안정화하는 기법.

---

**[Step 5] 임베딩 믹서 (Eqs. 6–7)**

$$\underline{\mathbf{X}}''_{d_i} = BN(\mathcal{P}(\mathbf{X}''_{d_i})) \in \mathbb{R}^{C \times N_i \times d} $$

$$\mathbf{X}_{d_{i2}} = \underline{\mathbf{X}}''_{d_i} + \mathcal{L}'_2(\mathcal{G}(\mathcal{L}'_1(\underline{\mathbf{X}}''_{d_i}))) \in \mathbb{R}^{C \times N_i \times d} $$

- $\mathcal{L}'_1: \mathbb{R}^{C \times N_i \times d} \to \mathbb{R}^{C \times N_i \times d \cdot d_f}$: 임베딩 차원 확장 ($d_f$: 확장 인자)
- $\mathcal{L}'_2: \mathbb{R}^{C \times N_i \times d \cdot d_f} \to \mathbb{R}^{C \times N_i \times d}$: 임베딩 차원 복원
- **잔차 연결(Residual Connection)** 포함: 패치 믹서와의 핵심 차이점

---

**[Step 6] 헤드 모듈 (Eqs. 8–9)**

$$\mathbf{Y}_{f_i} = \text{Flatten}(\mathbf{Y}_{d_i}) \in \mathbb{R}^{C \times N_i \cdot d} $$

$$\mathbf{Y}_{h_i} = \text{Linear}(\mathbf{Y}_{f_i}) \in \mathbb{R}^{C \times T_i} $$

- $T_i$: i번째 웨이블릿 계수 시리즈의 예측 길이 (보조 분해를 통해 결정)

---

**[Step 7] 재구성 (Eq. 10)**

$$\mathbf{Y} = \text{Reconstruction}_\psi(\mathbf{Y}_{A_m}, \mathbf{Y}_{D_m}, \mathbf{Y}_{D_{m-1}}, \ldots, \mathbf{Y}_{D_1}) $$

- $\mathbf{Y}_{A_i} \in \mathbb{R}^{C \times T_i}$: 예측된 근사 계수 시리즈
- $\mathbf{Y}_{D_i} \in \mathbb{R}^{C \times T_i}$: 예측된 세부 계수 시리즈
- $\mathbf{Y} \in \mathbb{R}^{C \times T}$: 역웨이블릿 변환으로 재구성된 예측 시계열

---

#### ③ 모델 구조

```
입력 XL ∈ R^{L×C}
    ↓ RevIN 정규화
    ↓ Transposition
    ↓ 다중 레벨 웨이블릿 분해 (m+1개 계수 시리즈 생성)
    ↓
[해상도 브랜치 × (m+1)]
    RevIN 정규화 → Patching → Linear Embedding
    → Mixer-1 (Patch Mixer + Embedding Mixer)
    → Mixer-2 (Patch Mixer + Embedding Mixer + 잔차연결 + BN)
    → Head (Flatten + Linear)
    → RevIN 역정규화
    ↓
다중 레벨 웨이블릿 재구성
    ↓ Transposition
    ↓ RevIN 역정규화
출력 XT ∈ R^{T×C}
```

> 💡 **용어 설명**
> - **RevIN(Reversible Instance Normalization)**: 시계열의 시변 평균·분산 문제를 해결하기 위해 입력 전 정규화, 출력 후 역정규화를 수행하는 기법 (Kim et al. 2021).
> - **잔차 연결(Residual Connection)**: 레이어의 입력을 출력에 직접 더하는 구조. 기울기 소실 문제 완화 및 학습 안정화에 기여.

---

#### ④ 성능 향상 및 한계

**성능 향상 (Table 2, p.6):**

| 데이터셋 | MSE 개선율 (vs TimeMixer) | MAE 개선율 |
|---------|--------------------------|-----------|
| ETTh1 | **7.8%** ↓ | 3.3% ↓ |
| ETTh2 | 2.2% ↓ | **6.4%** ↓ |
| ETTm1 | 3.4% ↓ | 0.5% ↓ |
| ETTm2 | 3.9% ↓ | 2.5% ↓ |
| GFLOPs | **10배 이상** 절감 | — |

**한계:**
- Traffic 데이터셋에서 통합 설정 기준 iTransformer(2024)에 MSE 성능 열세 (Table 9)
- Electricity 데이터셋에서 최적화 설정 기준 TimeMixer가 근소하게 우세 (Table 2)
- 웨이블릿 타입($\psi$), 분해 레벨($m$) 등 많은 하이퍼파라미터 최적화 필요
- 주요 결과가 **단일 랜덤 시드(42)** 기반 (Table 7 Supplementary 참조)

---

## 3. 주장별 페이지/Figure/Table 번호

| 주장 | 근거 위치 |
|------|-----------|
| 웨이블릿 분해 우월성 | p.1~2 (Introduction), Table 5 (p.7) |
| 모델 아키텍처 설명 | **Figure 1** (p.3), pp.3~5 (Proposed Method) |
| 다변량 SOTA 성능 | **Table 2** (p.6) |
| 계산 효율성 | **Table 3** (p.6) |
| 강건성 | **Table 4** (p.7) |
| 절제 연구 (모듈 기여) | **Table 5** (p.7) |
| 분해 레벨 영향 | **Figure 2** (p.7) |
| Look-back window 영향 | **Figure 3** (p.7) |
| SmoothL1 vs MSE | **Table 6** (p.7) |
| 단변량 성능 | **Table 10** (p.12, Supplementary) |
| 통합 설정 다변량 성능 | **Table 9** (p.11, Supplementary) |
| 하이퍼파라미터 상세 | **Table 7, 8** (p.10, Supplementary) |

---

## 4. 저자 보고 결과 vs. 해석 분리

### 4-1. 저자가 직접 보고한 결과

**연구 주제:**
- 장기 다변량/단변량 시계열 예측을 위한 웨이블릿 기반 MLP-Mixer 모델 제안

**방법 (수식):**
- Eq. 1~10: 웨이블릿 분해 → 패칭 → 임베딩 → 패치 믹서 → 임베딩 믹서 → 헤드 → 재구성
- 손실 함수: SmoothL1Loss (기본 임계값)
- 하이퍼파라미터 최적화: Optuna TPE (Tree-structured Parzen Estimator)

**저자 보고 결과:**
- ETTh1 평균 MSE: WPMixer **0.379** vs TimeMixer 0.411 (7.8% 개선) — Table 2
- GFLOPs (T=96, ETTh1): WPMixer **0.210** vs TimeMixer 2.774 — Table 3
- 단변량 ETTh1 평균 MSE: WPMixer **0.068** vs PatchTST/64 0.074 — Table 10
- 통합 설정 ETTh1 MSE: WPMixer 0.422 vs TimeMixer 0.447 (5.6% 개선) — Table 9

---

### 4-2. 검토자(필자)의 해석

- **강점**: 웨이블릿 분해를 통한 주파수 도메인 특징 활용은 이론적으로 타당하며, 특히 비정상(non-stationary) 시계열에서 이동 평균보다 우수한 표현력을 가짐
- **효율성의 원인**: GFLOPs 절감은 주로 패칭을 통한 시퀀스 길이 축소와 MLP의 낮은 연산 복잡도 덕분으로 해석됨
- **통합 설정 Traffic 결과**: Table 9에서 iTransformer(MSE 0.428)가 WPMixer(0.489)보다 우수한데, 이는 Traffic 데이터의 고차원성(862 변수)에서 변수 간 상호작용 모델링이 중요함을 시사
- **단일 시드 주의**: 최적화 실험 결과(Table 2)가 단일 시드(42) 기반이므로, 통계적 유의성 검증에 한계가 있음

---

## 5. 통계적 취약점 및 비교 불가능 수치 ⚠️

| 항목 | 설명 | 위치 |
|------|------|------|
| ⚠️ **단일 시드 주요 결과** | Table 2의 주요 결과가 **seed=42 단일 실행** 기반. 통계적 유의성 검증 미흡 | Table 7 (Supplementary) |
| ⚠️ **불균등 하이퍼파라미터 최적화** | WPMixer는 Optuna로 충분히 최적화되었으나, 일부 비교 모델(*표시)은 선행 논문 결과를 그대로 인용 | Table 2 각주 |
| ⚠️ **GFLOPs 비교 조건** | $d=16$ 임베딩 차원의 통합 설정에서만 GFLOPs 비교. 최적화 설정(Table 2)에서의 실제 GFLOPs 비교 부재 | Table 3 |
| ⚠️ **Traffic/Electricity 혼재 결과** | 최적화 설정(Table 2)에서는 WPMixer가 우세하나, 통합 설정(Table 9)에서는 Traffic에서 iTransformer 열세 | Table 2, Table 9 |
| ⚠️ **Crossformer 결과 인용** | Crossformer 결과가 TimeMixer 논문에서 재인용 (원 논문 결과와 상이할 가능성) | Table 2 각주 |
| ⚠️ **통합 설정 Weather** | Table 9에서 WPMixer(0.243)가 TimeMixer(0.240)보다 MSE 소폭 열세 | Table 9 |
| ⚠️ **단변량 비교 범위 제한** | 단변량 결과(Table 10)에서 TimeMixer와 직접 비교 없음 | Table 10 |

---

## 6. 논문이 답하지 않는 질문

| # | 미답 질문 |
|---|-----------|
| ① | **최적 웨이블릿 타입 선택 기준**: 왜 특정 데이터셋에 db2, db3, bior3.1 등이 최적인지 이론적 설명 부재 |
| ② | **웨이블릿 분해 레벨 m의 자동 결정 방법**: 현재 Optuna 탐색에 의존하며, 데이터 특성 기반 원칙적 선택 방법 미제시 |
| ③ | **단변량 vs 다변량 성능 차이 원인**: 채널 독립 처리의 한계가 다변량 예측에서 어떤 영향을 미치는지 분석 부재 |
| ④ | **실제 추론 시간(inference latency)**: GFLOPs는 하드웨어 독립 지표이지만, 실제 지연 시간 비교 없음 |
| ⑤ | **더 긴 예측 지평선(T > 720)**: 720 이상의 예측 길이에서의 성능 검증 부재 |
| ⑥ | **이상치(outlier) 강건성**: 실제 스파이크/딥 처리 능력을 정량적으로 검증한 실험 부재 |
| ⑦ | **사전 훈련(pre-training)/제로샷(zero-shot) 설정**: 도메인 간 전이 학습 가능성 미탐구 |
| ⑧ | **메모리 사용량**: 다중 해상도 브랜치 구조의 메모리 요구량 비교 부재 |
| ⑨ | **단변량 설정에서 TimeMixer와의 직접 비교**: Table 10에 TimeMixer 결과 없음 |
| ⑩ | **비유클리드(non-Euclidean) 또는 불규칙 간격 시계열**: 균일 간격 시계열만 대상으로 함 |

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1: WPMixer 전체 아키텍처 (p.3)

**해석:**
- **전체 흐름**: 입력 → RevIN 정규화 → 웨이블릿 분해 → (m+1)개 독립 해상도 브랜치 → 웨이블릿 재구성 → RevIN 역정규화 → 출력
- **핵심 설계 원칙**: 각 웨이블릿 계수 시리즈가 독립적인 브랜치에서 처리되어 주파수 대역 간 정보 혼합 방지
- **Mixer 구조**: Patch Mixer (패치 차원 혼합, 글로벌 정보) + Embedding Mixer (임베딩 차원 혼합, 고차원 글로벌 정보) 순차 적용
- **행렬 차원 변화**: $\mathbb{R}^{C \times L_i} \to \mathbb{R}^{C \times N_i \times P} \to \mathbb{R}^{C \times N_i \times d} \to \mathbb{R}^{C \times T_i}$의 명확한 변환 흐름
- **의의**: 웨이블릿 분해와 MLP-Mixer를 결합한 새로운 패러다임 제시

---

### Figure 2: 분해 레벨 m 변화에 따른 성능 (p.7)

**해석:**
- **관찰**: ETTh1/ETTh2에서 최적 m은 데이터셋과 예측 길이에 따라 다름 (m=1~5 범위)
- **ETTh1_336**: m=1~2에서 최저 MSE, 이후 증가 → 고레벨 분해가 항상 유리하지 않음
- **ETTh2_720**: 상대적으로 높은 m에서 개선 효과 지속
- **시사점**: 적절한 m 선택이 중요하며, 이를 하이퍼파라미터로 처리하는 것이 타당함을 실험적으로 입증
- **한계**: m이 클수록 계산 비용 증가, 최적 m의 데이터 특성 기반 예측 원칙 부재

> 💡 **용어 설명**
> - **분해 레벨(Decomposition Level, m)**: 웨이블릿 분해를 반복 적용하는 횟수. m이 클수록 더 저주파 성분까지 분리 가능하나, 계수 시리즈 길이가 줄어듦.

---

### Figure 3: Look-back Window 크기에 따른 성능 (p.7)

**해석:**
- **일반적 경향**: ETTh1/ETTh2 모두 look-back window $L$이 증가할수록 MSE 감소 (긴 역사 정보가 유리)
- **포화 및 역전 현상**: 일정 길이( $L \approx 512$ ~ $1024$ ) 이후 성능 개선 중단 또는 악화 (T=336에서 두드러짐)
- **예측 길이별 차이**: 짧은 예측(96)과 긴 예측(720)의 최적 look-back window가 다를 수 있음
- **실용적 시사점**: 무조건 긴 look-back window가 유리하지 않으며, 과거 정보의 관련성이 일정 시점 이후 감소함
- **모델 설계 의의**: WPMixer가 다양한 window 길이에서 안정적 성능을 보임을 간접 시사

---

### Table 2: 다변량 장기 예측 결과 (p.6) — 핵심 성능 비교표

**해석:**
- **WPMixer의 1위 횟수**: MSE 기준 29회, MAE 기준 26회 (전체 56개 조건 중) — 압도적 우위
- **ETTh1에서의 강점**: 모든 예측 길이에서 WPMixer가 TimeMixer 대비 MSE 우세 (평균 7.8% 개선)
- **Electricity에서의 패턴**: T=192에서 TimeMixer(0.140)가 WPMixer(0.145)보다 소폭 우세 — 데이터 특성에 따른 편차 존재
- **Crossformer의 열세**: 모든 데이터셋에서 WPMixer 대비 MSE/MAE 크게 열세 — 복잡한 attention이 반드시 유리하지 않음을 시사
- **⚠️ 비교 공정성 주의**: *표시 모델들은 TimeMixer 논문에서 결과 인용 (직접 재실험 아님)

---

### Table 5: 모듈 기여도 절제 연구 (p.7)

**해석:**
- **Case I (전체 모델)**: ETTh1 MSE 0.379, ETTh2 0.308 — 모든 케이스 중 최고 성능
- **분해(D) 제거 (Case II)**: ETTh1 0.388 (+0.009) — 웨이블릿 분해가 의미 있는 기여
- **패칭+임베딩 제거 (Case III)**: ETTh1 0.384 (+0.005) — 패칭의 효과 확인
- **패치 믹서+임베딩 믹서 제거 (Case VII)**: ETTh1 0.394 (+0.015) — 믹서 구조의 중요성
- **임베딩 믹서만 제거 (Case V)**: ETTh1 0.378 — 분해+패칭+패치믹서만으로도 합리적 성능, 단 ETTh2에서 차이 존재
- **전체 제거 (Case X)**: ETTh1 0.403 — 기본 헤드만 남은 경우 성능 최저
- **핵심 통찰**: 모든 모듈이 상호 보완적이며, 특히 분해+믹서의 조합이 핵심

---

## 8. 결론 및 후속 연구

### 8-1. 저자 제시 시사점 및 후속 연구 계획

**저자 제시 시사점 (Conclusion, p.8):**
- 다중 레벨 웨이블릿 분해가 실세계 시계열의 복잡한 특성(스파이크, 딥, 복잡 계절성) 포착에 효과적
- 패칭(로컬) + 믹싱(글로벌) 조합이 시계열 예측에서 강력한 귀납 편향(inductive bias)을 제공
- MLP 기반 모델이 Transformer 대비 계산 효율성과 성능을 동시에 달성 가능함을 입증

**저자가 명시한 후속 연구 계획:** 논문에 명시적 future work 섹션 **없음** ⚠️ (논문이 답하지 않는 질문 참조)

---

### 8-1. 모델의 일반화 성능 향상 가능성 (중점 분석)

**현재 일반화 관련 실험:**

| 측면 | 현황 | 한계 |
|------|------|------|
| 데이터셋 다양성 | 7개 벤치마크 (ETT×4, Weather, Electricity, Traffic) | 의료, 금융 고주파, 음성 신호 등 미검증 |
| 도메인 전이 | 미실험 | 제로샷/퓨샷 일반화 미검증 |
| 변수 수 | 7~862개 다양 | 초고차원(>1000) 미검증 |
| 계절성 패턴 | 시간별~10분 단위 | 일별·주별 긴 주기 데이터 미검증 |

**일반화 향상을 위한 제안 방향:**

1. **웨이블릿 타입 자동 선택 (적응형 웨이블릿)**
   - 데이터 특성에 따라 $\psi$를 자동으로 학습하는 학습 가능한 웨이블릿(Learnable Wavelet) 도입
   - 예: 리프팅 스킴(Lifting Scheme) 기반 적응형 웨이블릿 (Cotter 2019)

2. **메타 학습(Meta-Learning) 적용**
   - MAML 등 메타 학습 프레임워크를 통해 소량의 새 도메인 데이터로 빠른 적응 가능하도록 설계

3. **채널 간 의존성 모델링 강화**
   - 현재 채널 독립 처리 → 변수 간 상호작용이 중요한 Traffic 데이터에서의 한계 극복
   - iTransformer의 변수별 토크나이징 방식과 결합 검토

4. **분포 이동(Distribution Shift) 강건성**
   - RevIN 외에 추가적인 비정상성 처리 기법(Non-stationary Transformer의 de-stationary attention 등) 결합

5. **사전 훈련 + 미세 조정(Pre-training + Fine-tuning) 패러다임**
   - 대규모 시계열 데이터셋으로 사전 훈련 후 소규모 도메인 데이터로 미세 조정
   - Time-Series Foundation Model(예: TimesFM, Chronos) 트렌드와 결합

---

### 8-2. 2020년 이후 최신 연구 비교 분석

> **주의**: 아래 비교는 논문에 인용된 모델들을 중심으로 분석하며, 2024년 12월 이후 최신 논문은 포함되지 않음을 명시합니다.

#### 2020년 이후 주요 시계열 예측 연구 계보

```
2021: Informer (Zhou et al.) → prob-sparse attention
2021: Autoformer (Wu et al.) → Auto-correlation + 분해
2022: FEDformer (Zhou et al.) → 푸리에/웨이블릿 주파수 강화
2022: PatchTST (Nie et al.) → 패칭 + Transformer
2022: DLinear (Zeng et al.) → 단순 선형 모델의 반격
2023: TimesNet (Wu et al.) → FFT 기반 2D 시간 변환
2023: TSMixer (Chen et al.) → MLP-Mixer 시계열 적용
2023: iTransformer (Liu et al.) → 변수별 역방향 Transformer
2024: TimeMixer (Wang et al.) → 다중 스케일 MLP 분해+믹싱
2024: WPMixer (Murad et al.) → 웨이블릿 + 패치 MLP 믹서
```

#### 핵심 모델 비교표

| 모델 | 연도 | 분해 방식 | 도메인 | 아키텍처 | 계산 효율 |
|------|------|-----------|--------|----------|-----------|
| Autoformer | 2021 | 이동 평균 | 시간 | Transformer | 중간 |
| FEDformer | 2022 | 푸리에/웨이블릿 | 시간+주파수 | Transformer | 낮음 |
| PatchTST | 2023 | 없음 | 시간 | Transformer+Patch | 중간 |
| TSMixer | 2023 | 없음 | 시간 | MLP-Mixer | 낮음 (긴 윈도우) |
| TimeMixer | 2024 | 이동 평균 다중 스케일 | 시간 | MLP-Mixer | 높음 |
| iTransformer | 2024 | 없음 | 시간 | Inv. Transformer | 중간 |
| **WPMixer** | **2024** | **다중 레벨 DWT** | **시간+주파수** | **MLP-Mixer+Patch** | **매우 높음** |

#### WPMixer가 앞으로의 연구에 미치는 영향

1. **웨이블릿 분해 + MLP Mixer 결합의 유효성 입증**: 향후 유사 구조 연구의 기준점 제시
2. **계산 효율성 기준 상향**: GFLOPs 10배 절감 달성으로 효율성 비교 기준 강화
3. **주파수 도메인 활용의 재부각**: FEDformer 이후 주춤했던 주파수 도메인 연구 재활성화 가능성
4. **독립 해상도 브랜치 패러다임**: 각 주파수 대역을 독립 처리하는 구조적 아이디어가 다른 분야(의료 신호, 오디오 등)로 확장 가능

#### 향후 연구 시 고려할 점

| 고려 사항 | 세부 내용 |
|-----------|-----------|
| **하이퍼파라미터 민감성** | 웨이블릿 타입, 분해 레벨, 패치 크기 등 다수의 하이퍼파라미터 최적화 비용이 큼. 자동화/적응형 방법 필요 |
| **iTransformer와의 통합** | 변수 간 의존성 모델링이 중요한 고차원 데이터에서 WPMixer의 한계를 iTransformer 구조로 보완 가능 |
| **Foundation Model 트렌드** | Chronos, TimesFM 등 대규모 사전 훈련 모델과의 비교 및 결합 방향 탐구 필요 |
| **실시간 적용 가능성** | 낮은 GFLOPs는 에지 디바이스 배포에 유리하나, 메모리 사용량 및 실제 지연 시간 검증 필요 |
| **불규칙 시계열 처리** | 현재 균일 간격 시계열에 한정. 결측값 및 불규칙 간격 데이터 처리 능력 확장 필요 |
| **설명 가능성(XAI)** | 어떤 주파수 대역이 예측에 기여하는지 시각화 및 해석 방법 개발 필요 |
| **재현성** | 단일 시드 결과 의존 문제 → 다중 시드 평균 결과를 기본 보고 방식으로 표준화 권장 |

---

## 참고 자료 (출처)

본 분석은 다음 문서를 기반으로 작성되었습니다:

1. **주 논문**: Murad, M. M. N., Aktukmak, M., & Yilmaz, Y. (2024). *WPMixer: Efficient Multi-Resolution Mixing for Long-Term Time Series Forecasting*. arXiv:2412.17176v1 [cs.LG]. (제공된 PDF 전문)

2. **논문 내 인용 참고문헌** (주요):
   - Wang et al. (2024). *TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting*. ICLR 2024.
   - Chen et al. (2023). *TSMixer: An All-MLP Architecture for Time Series Forecasting*. TMLR.
   - Nie et al. (2023). *A Time Series is Worth 64 Words*. ICLR 2023.
   - Kim et al. (2021). *Reversible Instance Normalization for Accurate Time-Series Forecasting*. ICLR 2021.
   - Zeng et al. (2023). *Are Transformers Effective for Time Series Forecasting?* AAAI 2023.
   - Liu et al. (2024). *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting*. ICLR 2024.
   - Mallat, S. G. (1989). *A theory for multiresolution signal decomposition*. IEEE TPAMI.
   - Cotter, F. (2019). *Uses of Complex Wavelets in Deep Convolutional Neural Networks*. Ph.D. thesis, Cambridge.
   - Tolstikhin et al. (2021). *MLP-Mixer: An All-MLP Architecture for Vision*. NeurIPS 2021.
   - Akiba et al. (2019). *Optuna: A next-generation hyperparameter optimization framework*. KDD 2019.

> **정확도 주의**: 2020년 이후 최신 연구(특히 2024년 말~2025년) 비교 분석에서 WPMixer 논문 발표(2024.12.22) 이후 등장한 연구들(예: TimesFM, Chronos, MOIRAI 등 Foundation Model 계열)과의 비교는 논문 원문에 포함되지 않아 필자의 배경 지식 기반 분석이 포함되었음을 명시합니다. 해당 부분은 일반적 연구 동향 참조로만 활용하시기 바랍니다.
