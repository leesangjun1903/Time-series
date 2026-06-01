# Time Series as Images: Vision Transformer for Irregularly Sampled Time Series

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문은 **불규칙하게 샘플링된 시계열(Irregularly Sampled Time Series)** 데이터를 **선 그래프 이미지(Line Graph Image)** 로 변환한 뒤, 사전 학습된 비전 트랜스포머(Vision Transformer)를 활용하여 분류 문제를 해결하는 **ViTST (Vision Time Series Transformer)** 를 제안합니다.

> "시계열 데이터를 이미지로 변환하면, 자연 이미지에서 사전 학습된 비전 트랜스포머의 강력한 표현 학습 능력을 시계열 분류에 직접 활용할 수 있다."

### 주요 기여 (3가지)

| 기여 | 내용 |
|------|------|
| **① 단순하고 효과적인 방법론** | 불규칙 시계열을 위한 범용 비전 기반 분류 프레임워크 제안 |
| **② 범용성(Versatility)** | 불규칙/정규 시계열 모두에서 경쟁력 있는 성능 달성 |
| **③ 도메인 간 지식 전이** | 자연 이미지 사전 학습 지식 → 합성 시계열 이미지로의 성공적 전이 입증 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 시계열 모델링의 한계:

- **LSTM, TCN, Transformer** 등 기존 모델은 **고정 간격·완전 관측 데이터**를 가정
- 불규칙 샘플링 대응을 위한 **고도로 특수화된 모델**(GRU-D, mTAND, Raindrop 등)은 **도메인 지식·설계 비용이 높음**
- 의료 분야 등에서 흔한 **고결측률(최대 94.9%)** 및 **변수 간 비동기 관측** 문제

```math
\mathcal{S}_i = \left\{ \left[(t_1^d, v_1^d), (t_2^d, v_2^d), \cdots, (t_{n_d}^d, v_{n_d}^d)\right] \mid d = 1, \cdots, D \right\}
```

여기서 관측 시간 간격 $[t_1^d, t_2^d, \cdots, t_{n_d}^d]$ 이 변수 또는 샘플 간에 다르면 불규칙 시계열입니다.

---

### 2.2 제안 방법

#### Step 1: 시계열 → 이미지 변환

**문제 정의:**

$$\mathcal{D} = \{(\mathcal{S}_i, y_i) \mid i = 1, \cdots, N\}, \quad y_i \in \{1, \cdots, C\}$$

각 다변량 시계열 $\mathcal{S}_i$를 이미지 $\mathbf{x}_i$로 변환하는 함수 $f: \mathcal{S}_i \mapsto \mathbf{x}_i$를 정의합니다.

**변환 절차:**

1. 변수 $d$별로 개별 선 그래프 $g_{i,d}$ 생성
   - X축: 타임스탬프, Y축: 관측값
   - 관측 데이터 포인트는 마커 기호 `*`로 표시
   - 관측점 사이는 선형 보간(또는 미보간) 연결
2. $D$개의 선 그래프를 $l \times l$ (또는 $l \times (l+1)$ ) 격자로 배열하여 단일 RGB 이미지 생성
   - $l \times (l-1) < D \leq l \times l$이면 $l \times l$ 격자 사용
3. 결측 비율 순으로 변수 정렬

**이미지 생성 예시 (Matplotlib):**

```python
def TS2Image(t, v, D, colors, image_height, image_width, grid_height, grid_width):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(image_height/100, image_width/100), dpi=100)
    for d in range(D):
        plt.subplot(grid_height, grid_width, d+1)
        plt.plot(t[d], v[d], color=colors[d], linestyle="-", marker="*")
```

#### Step 2: 비전 트랜스포머를 이용한 분류

**Swin Transformer** 를 기본 백본으로 사용하며, 연속적인 Swin Transformer 블록의 수식은 다음과 같습니다:

$$\hat{\mathbf{z}}^l = \text{W-MSA}\left(\text{LN}(\mathbf{z}^{l-1})\right) + \mathbf{z}^{l-1}$$

$$\mathbf{z}^l = \text{MLP}\left(\text{LN}(\hat{\mathbf{z}}^l)\right) + \hat{\mathbf{z}}^l$$

$$\hat{\mathbf{z}}^{l+1} = \text{SW-MSA}\left(\text{LN}(\mathbf{z}^l)\right) + \mathbf{z}^l$$

$$\mathbf{z}^{l+1} = \text{MLP}\left(\text{LN}(\hat{\mathbf{z}}^{l+1})\right) + \hat{\mathbf{z}}^{l+1}$$

- $\hat{\mathbf{z}}^l$: (S)W-MSA 모듈의 출력 피처 (블록 $l$)
- $\mathbf{z}^l$: MLP 모듈의 출력 피처 (블록 $l$)
- LN: Layer Normalization
- **W-MSA**: 비겹침 윈도우 내 자기주의 → 단일 선 그래프의 **지역적(intra-variable)** 패턴 포착
- **SW-MSA**: 이동 윈도우를 통해 서로 다른 선 그래프 간 **전역적(inter-variable)** 상호관계 포착

**정적 피처(Static Features) 통합:**

의료 데이터의 인구통계학적 정보를 자연어 문장으로 변환 후 **RoBERTa-base** 인코더로 인코딩하여 이미지 임베딩과 결합:

$$\hat{y}_i = \text{Head}([\mathbf{e}^{\text{image}}_i; \mathbf{e}^{\text{text}}_i])$$

**자기지도 학습 (Self-supervised Pre-training, 부록):**

마스킹된 이미지 모델링(Masked Image Modeling)을 추가 탐색하였으며, $\ell_1$ 손실을 사용합니다:

$$\mathcal{L} = \frac{1}{\Omega(\mathbf{p}_M)} \|\hat{\mathbf{p}}_M - \mathbf{p}_M\|_1$$

여기서 $\mathbf{p}_M$은 마스킹된 픽셀, $\hat{\mathbf{p}}_M$은 재구성된 픽셀, $\Omega(\cdot)$는 원소 수입니다.

---

### 2.3 모델 구조

```
[다변량 불규칙 시계열 S_i]
        ↓ (Matplotlib 시각화)
[변수별 선 그래프 g_{i,1}, ..., g_{i,D}]
        ↓ (격자 배열)
[단일 RGB 이미지 x_i (예: 384×384)]
        ↓
[Swin Transformer Encoder (ImageNet-21K 사전학습)]
  - W-MSA: 지역 패턴(intra-variable)
  - SW-MSA: 전역 패턴(inter-variable)
        ↓
[Flatten + Linear Head]
        ↓
[예측 레이블 ŷ_i]
```

**구현 세부사항:**
- 백본: Swin Transformer (ImageNet-21K 사전학습 체크포인트)
- 패치 크기: 4, 윈도우 크기: 7
- 각 격자 셀 크기: 64×64 픽셀
- P19·P12: 6×6 격자 (384×384), PAM: 4×5 격자 (256×320)
- 학습률: 2e-5, 배치 크기: 48 (P19·P12), 72 (PAM)

---

### 2.4 성능 향상

#### 주요 데이터셋 결과 (Table 2)

| 데이터셋 | 지표 | 이전 SoTA (Raindrop) | **ViTST** | 향상폭 |
|----------|------|---------------------|-----------|--------|
| P19 | AUROC | 87.0 ± 2.3 | **89.2 ± 2.0** | +2.2%p |
| P19 | AUPRC | 51.8 ± 5.5 | **53.1 ± 3.4** | +1.3%p |
| P12 | AUROC | 82.8 ± 1.7 | **85.1 ± 0.8** | +0.7%p (vs. DGM2-O 84.4) |
| P12 | AUPRC | 48.2 ± 3.4 (mTAND) | **51.1 ± 4.1** | +2.9%p |
| PAM | F1 Score | 89.8 ± 1.0 | **96.5 ± 1.2** | +6.7%p |

#### Leave-sensors-out 결과 (50% 변수 마스킹)

| 지표 | Raindrop (SoTA) | **ViTST** | 향상폭 |
|------|----------------|-----------|--------|
| Accuracy | 46.6 | **79.7** | +33.1%p |
| Precision | 44.5 | **83.4** | +40.9%p |
| Recall | 42.4 | **81.8** | +39.4%p |
| F1 Score | 38.0 | **80.8** | **+42.8%p** |

---

### 2.5 한계점

논문에서 명시된 한계와 추가적으로 파악 가능한 한계는 다음과 같습니다:

| 구분 | 한계 |
|------|------|
| **시각화 방법 고정** | Matplotlib 기반 선 그래프에 국한; 더 효과적인 시각화 방법이 존재할 수 있음 |
| **추론 시간** | P19 기준 44.51초로 기존 방법 대비 현저히 느림 (Transformer: 0.21초) |
| **작동 메커니즘 불명확** | 비전 트랜스포머가 왜 시계열 패턴을 잘 포착하는지 이론적 해석 부재 |
| **예측·이상탐지 미검증** | 분류 태스크에만 집중; 예측(forecasting), 이상 탐지 등 미검증 |
| **데이터셋 제한** | 의료·인간 활동 데이터에 집중; 금융, 기후 등 다른 도메인 검증 부족 |
| **극단값 처리** | 극단값이 y축을 확장하여 대부분의 점이 평탄하게 보이는 문제 발생 가능 |
| **고차원 이미지 비용** | 변수 수 증가 시 이미지 해상도·연산 비용 증가 |

---

## 3. 모델의 일반화 성능 향상 가능성

이 논문에서 일반화 성능과 관련된 핵심 근거와 가능성을 다각도로 분석합니다.

### 3.1 일반화 근거 ①: 불규칙·정규 시계열 모두 처리 가능

정규 시계열 10개 UEA 벤치마크에서도 평균 정확도 2위(0.780)를 달성하여, 전문 모델(TST: 0.791)에 근접한 성능을 보였습니다. 특히 **963개 변수(PS)** 와 **시퀀스 길이 17,984(EW)** 라는 극단적 데이터셋에서 강점을 보였습니다.

$$\text{일반화 가능 도메인} = \{\text{의료, 인간 활동, 금융, 기후, 교통, ...}\}$$

### 3.2 일반화 근거 ②: 결측값에 대한 강력한 견고성

Leave-sensors-out 실험에서 **결측 비율이 10%~50%로 증가해도 성능이 80% 이상 유지**되는 반면, 기존 특화 모델들은 급격한 성능 하락을 보입니다.

이는 선 그래프 이미지 표현이 본질적으로 **결측 구조에 불변(missing-agnostic)** 하기 때문입니다:
- 관측이 없는 변수의 격자 칸은 단순히 빈 칸으로 표현
- 비전 트랜스포머의 어텐션 메커니즘이 정보가 있는 영역에 집중 (Attention Map 분석으로 확인)

### 3.3 일반화 근거 ③: 사전 학습 지식의 성공적 전이

**스크래치 학습 vs. 사전 학습 비교 실험**에서 사전 학습 없이 훈련한 Swin-scratch는 성능이 크게 저하됩니다. 이는 ImageNet-21K에서 학습한 **엣지, 형태, 공간적 관계 인식 능력**이 선 그래프의 기울기 변화, 패턴 형태 인식에 전이됨을 시사합니다.

$$\text{Pre-train: ImageNet-21K} \xrightarrow{\text{Fine-tune}} \text{Time Series Line Graphs}$$

### 3.4 일반화 근거 ④: 플로팅 파라미터에 대한 견고성 (Table 4)

선 스타일(solid/dashed/dotted), 선 두께(0.5/1/2), 마커 스타일(*/∧/○), 마커 크기(1/2/3) 변화에도 AUROC 88.2~89.3으로 안정적 성능을 유지합니다.

### 3.5 일반화 가능성 확장 방향

| 방향 | 근거 |
|------|------|
| **자기지도 학습 적용** | 마스킹된 이미지 모델링(MIM)으로 AUPRC +1.0%p 향상 확인 (부록 B.5) |
| **멀티모달 확장** | 이미지+텍스트 결합 가능; RoBERTa 정적 피처 통합 실험으로 검증 |
| **데이터 증강** | Cutout 등 비전 데이터 증강 기법을 소규모 데이터셋(SCP1, SCP2, JV)에 적용하여 효과 확인 |
| **더 강력한 비전 백본** | Swin V2, DeiT III, EVA 등 최신 비전 모델로 교체 가능 |
| **CLIP 기반 멀티모달** | 시각적 시계열 이미지와 임상 노트를 공유 임베딩 공간에서 학습 가능 |

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

ViTST가 비교 대상으로 삼거나 관련성이 높은 2020년 이후 주요 연구를 분석합니다.

### 4.1 불규칙 시계열 특화 모델

| 모델 | 연도 | 핵심 방법 | P19 AUROC | PAM F1 |
|------|------|-----------|-----------|--------|
| **mTAND** (Shukla & Marlin) | 2021 | 연속 시간 임베딩 + 다중 시간 어텐션 | 84.4 | 76.8 |
| **Raindrop** (Zhang et al.) | 2022 | 그래프 신경망 기반 변수 간 관계 모델링 | 87.0 | 89.8 |
| **UTDE** (Zhang et al.) | 2022 | mTAND + 보간 시계열 + 학습 가능 게이트 | - | - |
| **ViTST** (본 논문) | 2023 | 선 그래프 이미지 + Swin Transformer | **89.2** | **96.5** |

### 4.2 시계열 예측용 트랜스포머 계열 (정규 시계열)

| 모델 | 연도 | 핵심 방법 | 특징 |
|------|------|-----------|------|
| **Informer** (Zhou et al., 2021) | 2021 | ProbSparse Self-Attention | 장기 예측 효율화 |
| **Autoformer** (Wu et al., 2021) | 2021 | 자기상관 기반 분해 | 트렌드-계절성 분리 |
| **PatchTST** (Nie et al., 2022) | 2022 | 시계열을 64단어 서브시리즈로 분할 | "A time series is worth 64 words" |
| **FEDformer** (Zhou et al., 2022) | 2022 | 주파수 도메인 분해 트랜스포머 | 장기 예측 |

**ViTST와의 차이점:** 위 모델들은 수치형 입력을 직접 처리하며 완전 관측을 가정. ViTST는 시각적 모달리티로 변환하여 불규칙 샘플링을 자연스럽게 처리.

### 4.3 시계열 이미지화 관련 연구

| 연구 | 연도 | 방법 | 한계 |
|------|------|------|------|
| Wang & Oates (2015) | 2015 | Gramian Angular Field, Markov Transition Field + CNN | 도메인 지식 필요, CNN 기반 |
| Hatami et al. (2018) | 2018 | Recurrence Plot + 딥 CNN | 특수 설계 필요 |
| Sood et al. (2021) | 2021 | 컨볼루션 오토인코더 기반 이미지 예측 | 예측 전용, 특수 설계 필요 |
| **ViTST (본 논문)** | 2023 | 선 그래프 + 사전학습 비전 트랜스포머 | 단순, 범용, SoTA 초과 |

**핵심 차별성:** 기존 이미지화 방법들이 여전히 수치 모델보다 뒤처진 반면, ViTST는 사전학습 비전 트랜스포머를 활용하여 처음으로 수치 특화 모델을 능가했습니다.

### 4.4 대규모 사전학습 모델의 시계열 적용 트렌드

| 연구 방향 | 대표 연구 | 비고 |
|----------|-----------|------|
| LLM for Time Series | GPT4TS, Time-LLM (2023~) | 언어 모델을 시계열에 직접 적용 |
| Vision Foundation Model 활용 | ViTST (본 논문) | 비전 모델을 시계열 이미지에 적용 |
| 멀티모달 | CLIP 기반 접근 | 이미지+텍스트 공동 학습 |

> **참고:** GPT4TS, Time-LLM 등 2023년 이후 LLM 기반 시계열 연구는 제공된 논문에 포함되지 않아, 해당 모델들과의 정량적 비교는 본 리뷰의 범위를 벗어납니다. 정확한 수치 비교는 원 논문을 직접 확인하시기 바랍니다.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 5.1 연구에 미치는 영향

#### ① 패러다임 전환: 수치형 → 시각적 모달리티

ViTST는 시계열 분석의 패러다임을 "수치 입력 기반 특화 모델 설계"에서 "시각화를 통한 사전학습 모델 활용"으로 전환할 가능성을 열었습니다. 이는 **프롬프트 엔지니어링**의 개념과 유사하게, 도메인 전문가가 시각화 방식을 조정함으로써 모델 성능을 향상시킬 수 있음을 시사합니다.

#### ② 컴퓨터 비전 기술의 시계열 도메인 이전

- **데이터 증강:** CutOut, MixUp, CutMix 등 비전 증강 기법을 시계열에 직접 적용 가능
- **해석 가능성:** Grad-CAM, Attention Rollout 등 비전 해석 기법 활용 가능
- **자기지도 학습:** MAE(Masked Autoencoders) 스타일의 사전학습을 시계열 이미지에 적용 가능
- **모델 아키텍처:** Swin V2, EVA, DINOv2 등 최신 비전 모델 즉시 활용 가능

#### ③ 멀티모달 학습의 기반

이미지 형태로 변환된 시계열은 CLIP, GPT-4V 등 멀티모달 모델과 자연스럽게 결합 가능하며, 임상 노트, 텍스트 보고서 등과의 공동 학습 기반을 마련합니다.

#### ④ 범용 시계열 프레임워크로의 잠재력

불규칙·정규 시계열을 단일 프레임워크로 처리한다는 점에서, 서로 다른 특성의 시계열 데이터를 통합 처리하는 **Foundation Model for Time Series** 구축의 방향성을 제시합니다.

---

### 5.2 앞으로 연구 시 고려할 점

#### 🔴 핵심 미해결 문제

**① 작동 메커니즘의 이론적 해명**

왜 자연 이미지(ImageNet)에서 학습한 표현이 합성 시계열 이미지에 전이되는가에 대한 이론적 설명이 부재합니다. 향후 연구에서는:
- 어텐션 맵 분석을 넘어선 **Mechanistic Interpretability** 연구
- 자연 이미지 특성과 선 그래프 특성의 공통 구조 분석 필요

**② 추론 시간 최적화**

$$\text{ViTST 추론시간 (P19)}: 44.51\text{s} \gg \text{Transformer}: 0.21\text{s}$$

이미지 생성 + 비전 모델 추론의 이중 비용 문제를 해결하기 위해:
- 경량화된 비전 모델(MobileViT, EfficientViT) 적용 탐색
- 이미지 생성 파이프라인 GPU 가속화
- End-to-end 차별화 가능한 시계열→이미지 변환 모듈 설계

**③ 최적 시각화 방법 탐색**

현재 선 그래프 외에도 다양한 시각화가 가능하며, 최적 방법은 데이터 특성에 따라 다를 수 있습니다:
- Gramian Angular Field, Recurrence Plot 등과의 체계적 비교
- **학습 가능한(Learnable) 시계열→이미지 변환 모듈** 설계 (현재는 Matplotlib 고정)
- 시계열 특성에 맞는 맞춤형 시각화 자동 선택 방법

**④ 예측(Forecasting) 및 이상 탐지로의 확장**

현재 분류 태스크에만 검증되었으므로:
- 예측: 미래 시계열 이미지의 생성 또는 수치 예측
- 이상 탐지: 정상 패턴과의 시각적 차이 활용
- 결측값 보간(Imputation): 마스킹된 이미지 재구성 활용

**⑤ 도메인 다양성 검증**

의료, 인간 활동 외에도:
- 금융 시계열(고빈도 거래 데이터)
- 기후 데이터(다중 센서, 장기 시계열)
- 산업 IoT(예지 보전) 등 다양한 도메인에서의 검증 필요

**⑥ 데이터 효율성 (Data Efficiency)**

소규모 레이블 데이터에서의 성능:
- Few-shot, Zero-shot 설정에서의 일반화 능력 검증
- 사전학습 데이터(자연 이미지)와 목표 도메인(시계열)의 도메인 격차(Domain Gap) 정량화

**⑦ 변수 수 확장성 (Scalability)**

변수 수 $D$가 매우 많을 경우($D > 100$):
- 이미지 해상도 급증 → 메모리·연산 비용 폭증
- 각 격자 셀의 크기 축소 → 세부 패턴 손실 가능
- 계층적 이미지 구성 또는 중요 변수 선택 알고리즘 필요

**⑧ 멀티모달 통합의 체계화**

현재 정적 피처를 RoBERTa로 단순 결합하는 수준에서 나아가:
- CLIP 스타일의 **시각-언어 공동 임베딩 공간** 구축
- GPT-4V 등 멀티모달 LLM과의 통합
- 임상 노트, 시계열 이미지, 정형 데이터의 통합 프레임워크

---

### 5.3 연구 방향 우선순위 요약

```
높은 영향 & 단기 달성 가능
├── 더 강력한 비전 백본 적용 (Swin V2, DINOv2)
├── 자기지도 사전학습 확장 (MAE 스타일)
└── 다양한 도메인 벤치마크 검증

중간 영향 & 중기 목표
├── 학습 가능한 시계열→이미지 변환
├── 예측·이상탐지 태스크 확장
└── 추론 시간 최적화

높은 영향 & 장기 연구
├── 작동 메커니즘 이론적 해명
├── 멀티모달 통합 (CLIP, GPT-4V)
└── 시계열 Foundation Model 구축
```

---

## 참고자료

**주 논문:**
- Li, Z., Li, S., & Yan, X. (2023). "Time Series as Images: Vision Transformer for Irregularly Sampled Time Series." *NeurIPS 2023*. arXiv:2303.12799v2.

**논문 내 인용 핵심 참고문헌:**
- Dosovitskiy, A., et al. (2020). "An image is worth 16x16 words: Transformers for image recognition at scale." *ICLR 2021*. [ViT]
- Liu, Z., et al. (2021). "Swin transformer: Hierarchical vision transformer using shifted windows." *ICCV 2021*. [Swin Transformer]
- Zhang, X., et al. (2022). "Graph-guided network for irregularly sampled multivariate time series." *ICLR 2022*. [Raindrop]
- Shukla, S. N., & Marlin, B. (2021). "Multi-time attention networks for irregularly sampled time series." *ICLR 2021*. [mTAND]
- Che, Z., et al. (2018). "Recurrent neural networks for multivariate time series with missing values." *Scientific Reports*. [GRU-D]
- Horn, M., et al. (2020). "Set functions for time series." *ICML 2020*. [SeFT]
- Zerveas, G., et al. (2021). "A transformer-based framework for multivariate time series representation learning." *KDD 2021*. [TST]
- Nie, Y., et al. (2022). "A time series is worth 64 words: Long-term forecasting with transformers." arXiv:2211.14730. [PatchTST]
- He, K., et al. (2022). "Masked autoencoders are scalable vision learners." *CVPR 2022*. [MAE]
- Bagnall, A., et al. (2018). "The UEA multivariate time series classification archive." arXiv:1811.00075.

**GitHub 코드:** https://github.com/Leezekun/ViTST
