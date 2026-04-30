# Lightweight and Fast Time-Series Anomaly Detection via Point-Level and Sequence-Level Reconstruction Discrepancy

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

이 논문은 Industry 4.0 환경에서 **리소스가 제한된 엣지 디바이스(Edge Device)**에 배포 가능한 비지도 학습 기반의 경량 시계열 이상 탐지 모델 **LFTSAD(Lightweight and Fast Time-Series Anomaly Detection)**를 제안합니다. 기존의 딥러닝 기반 TSAD 모델들이 높은 정확도만을 추구하여 깊은 구조와 많은 파라미터를 사용하는 반면, LFTSAD는 **단 4개의 2층 MLP**만으로 구성된 경량 구조를 통해 **정확도·신속성·저자원소비**의 세 가지 목표를 동시에 달성합니다.

### 주요 기여

| 번호 | 기여 내용 |
|------|-----------|
| ① | **최초의 All-MLP 기반 경량 TSAD 모델** 제안 (4개의 병렬화 가능한 2층 MLP 구성) |
| ② | **재구성 불일치(Reconstruction Discrepancy) 기반 이상 점수화 기법** 설계 (포인트 레벨 + 시퀀스 레벨 결합) |
| ③ | 14개 공개 데이터셋 및 2개 엣지 디바이스(Raspberry Pi 4b, Jetson Xavier NX)에서 실험 검증: 딥 SOTA 모델 대비 **3~10배 빠른 속도**, **1/2 수준의 자원 소비**, **동등하거나 우월한 정확도** |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

논문은 두 가지 핵심 도전 과제를 제시합니다.

**도전 1: 신속성·자원 효율성·정확도의 균형**

기존 딥러닝 기반 TSAD 모델(CNN, Transformer, RNN 기반)은 클라우드 서버 환경을 전제로 설계되어 깊은 구조와 대규모 파라미터를 사용합니다. 예를 들어 Raspberry Pi 4b는 2GB RAM, 1.5GHz 클럭만을 보유하여 기존 모델의 안정적 배포가 불가능합니다.

**도전 2: 강력한 이상 점수화 체계 구성**

기존 이상 점수화 방식은 세 가지로 분류됩니다:
- **재구성 오류 기반(Reconstruction Error-based)**: 단순하나 얕은 구조에서 재구성 정확도 한계 존재
- **예측 오류 기반(Prediction Error-based)**: 시계열의 복잡한 특징 처리에 한계
- **연관 불일치 기반(Association Discrepancy-based)** [ATran, DCDet 등]: 세밀한 연관 계산으로 높은 정확도를 달성하나 막대한 연산 비용 발생

### 2.2 제안하는 방법 및 수식

#### 모델 전체 문제 정의

시계열 $\mathbf{X} = \{x_1, \ldots, x_T\} \in \mathbb{R}^{M \times T}$에서 각 타임스탬프에 대한 이상 점수:

$$\text{Score}_t = \text{TSAD}(x_t) $$

이상 여부 판별:

$$\bar{y}_t = \begin{cases} 1, & \text{if } \text{Score}_t \geq \lambda \\ 0, & \text{otherwise} \end{cases} $$

---

#### (A) 포인트 레벨 재구성 불일치 학습 (Point-Level Reconstruction Discrepancy Learning)

**포인트 레벨 샘플링**: 윈도우 $WIN$의 시퀀스를 $P_1 \times P_2$ 크기의 2D 행렬로 변환. 위치 $(i,j)$의 타임스탬프에 대해:
- **로컬 이웃**: 같은 행 $i$의 타임스탬프들
- **글로벌 이웃**: 같은 열 $j$의 타임스탬프들

**로컬 연관 학습 (1번째 MLP)**:

$$h_1(x_t^m) = \text{ReLU}\left(LP(x_t^m) \times \mathbf{W}_1^1\right) $$

$$\tilde{x}_t^m = h_1(x_t^m) \times \mathbf{W}_1^2 $$

여기서 $LP(x_t^m) = \{x_i^m\}_{i=t-P_2}^{t-1} \in \mathbb{R}^{1 \times (P_2-1)}$ 은 로컬 이웃, $\tilde{x}_t^m \in \mathbb{R}$은 로컬 재구성 값, $\mathbf{W}_1^1 \in \mathbb{R}^{(P_2-1) \times d}$, $\mathbf{W}_1^2 \in \mathbb{R}^{d \times 1}$

**글로벌 연관 학습 (2번째 MLP)**:

$$h_2(x_t^m) = \text{ReLU}\left(GP(x_t^m) \times \mathbf{W}_2^1\right) $$

$$\bar{x}_t^m = h_2(x_t^m) \times \mathbf{W}_2^2 $$

여기서 $GP(x_t^m) = \{x_{t-j \times P_2}^m\}_{j=1}^{P_1-1} \in \mathbb{R}^{1 \times (P_1-1)}$ 은 글로벌 이웃, $\bar{x}_t^m \in \mathbb{R}$은 글로벌 재구성 값

**포인트 레벨 손실 함수 (대조 학습)**:

$$\mathcal{L}oss_P = \frac{1}{M} \sum_{m=1}^{M} \frac{1}{WIN} \sum_{t=1}^{WIN} \left(\tilde{x}_t^m - \bar{x}_t^m\right)^2 $$

> **설계 직관**: 정상 타임스탬프는 로컬·글로벌 이웃 모두와 강한 연관을 가지므로 두 재구성 값의 차이(불일치)가 작습니다. 반면 이상 타임스탬프는 글로벌 이웃과의 연관이 약해 불일치가 커집니다.

---

#### (B) 시퀀스 레벨 재구성 불일치 학습 (Sequence-Level Reconstruction Discrepancy Learning)

**시퀀스 레벨 샘플링**: $WIN = S_1 \times S_2 \times S_3$으로 분할. 타임스탬프 $x_t^m$에 대해:
- **대체 시퀀스(Substitute)**: $SEQ(x_t^m) = \{x_i^m\}_{i=t-S_3}^{t}$
- **로컬 이웃**: $LS(x_t^m) = \{x_i^m\}_{i=t-[(S_2-1) \times S_3]}^{t-S_3}$
- **글로벌 이웃**: $GS(x_t^m) = \{\{x_i^m\}_{i=t-j \times (S_2 \times S_3)-S_3}^{t-j \times (S_2 \times S_3)}\}_{j=1}^{S_1-1}$

**시퀀스 레벨 로컬 재구성 (3번째 MLP)**:

$$h_3(x_t^m) = \text{ReLU}\left(LS(x_t^m) \times \mathbf{W}_3^1\right) $$

$$\widehat{SEQ}(x_t^m) = h_3(x_t^m) \times \mathbf{W}_3^2 $$

**시퀀스 레벨 글로벌 재구성 (4번째 MLP)**:

$$h_4(x_t^m) = \text{ReLU}\left(GS(x_t^m) \times \mathbf{W}_4^1\right) $$

$$\overline{SEQ}(x_t^m) = h_4(x_t^m) \times \mathbf{W}_4^2 $$

**시퀀스 레벨 손실 함수**:

$$\mathcal{L}oss_S = \frac{1}{M} \sum_{m=1}^{M} \frac{1}{WIN} \sum_{t=1}^{WIN} \left(\widehat{SEQ}(x_t^m) - \overline{SEQ}(x_t^m)\right)^2 $$

---

#### (C) 재구성 불일치 기반 이상 점수화

$$\text{Score}(x_t) = \alpha \times \text{Score}^P(x_t) + (1 - \alpha) \times \text{Score}^S(x_t) $$

여기서:
- $\text{Score}^P(x_t) = \text{mean}(\{\text{Score}_m^P\}_{m=1}^M)$: M개 변수의 포인트 레벨 불일치 평균
- $\text{Score}^S(x_t) = \text{mean}(\{\text{Score}_m^S\}_{m=1}^M)$: M개 변수의 시퀀스 레벨 불일치 평균
- $\alpha \in [0, 1]$: 두 레벨 점수의 균형 하이퍼파라미터

---

### 2.3 모델 구조

```
입력 시계열 X ∈ R^{M×T}
        │
        ├─── [포인트 레벨 브랜치] ──────────────────────────────┐
        │    ┌─────────────────────────────────────────────┐   │
        │    │  포인트 레벨 샘플링 (P₁×P₂ 2D 행렬 변환)     │   │
        │    │  ┌───────────────┐  ┌────────────────────┐  │   │
        │    │  │ MLP-1 (2층)   │  │ MLP-2 (2층)        │  │   │
        │    │  │ 로컬 이웃→x_t │  │ 글로벌 이웃→x_t    │  │   │
        │    │  │ (P₂-1)→d→1   │  │ (P₁-1)→d→1        │  │   │
        │    │  └───────┬───────┘  └─────────┬──────────┘  │   │
        │    │          │ x̃_t^m              │ x̄_t^m      │   │
        │    │          └────────差(불일치)────┘            │   │
        │    │                ScoreP_m                     │   │
        │    └─────────────────────────────────────────────┘   │
        │                                                       │
        ├─── [시퀀스 레벨 브랜치] ─────────────────────────────┤
        │    ┌─────────────────────────────────────────────┐   │
        │    │  시퀀스 레벨 샘플링 (S₁×S₂×S₃ 구조)         │   │
        │    │  ┌───────────────┐  ┌────────────────────┐  │   │
        │    │  │ MLP-3 (2층)   │  │ MLP-4 (2층)        │  │   │
        │    │  │ 로컬→SEQ(x_t) │  │ 글로벌→SEQ(x_t)   │  │   │
        │    │  │(S₂-1)S₃→d→S₃ │  │(S₁-1)S₃→d→S₃     │  │   │
        │    │  └───────┬───────┘  └─────────┬──────────┘  │   │
        │    │          │ SEQ̃(x_t^m)         │ SEQ̄(x_t^m) │   │
        │    │          └────────差(불일치)────┘            │   │
        │    │                ScoreS_m                     │   │
        │    └─────────────────────────────────────────────┘   │
        │                                                       │
        └─── [이상 점수화] ──────────────────────────────────────┘
             Score(x_t) = α·Score^P + (1-α)·Score^S
             → 임계값 λ 비교 → 이상/정상 판별
```

**파라미터 규모**: 총 약 12.6K 파라미터 (ATran의 4741K 대비 약 1/376)

---

### 2.4 계산 복잡도 분석

| 컴포넌트 | 복잡도 |
|---------|--------|
| 포인트 레벨 재구성 | $\mathcal{O}(T \times (P_1 + P_2) \times (d+1))$ |
| 시퀀스 레벨 재구성 | $\mathcal{O}(T \times (S_1 + S_2) \times S_3 \times d)$ |
| 이상 점수화 | $\mathcal{O}(T \times 2M)$ |
| **전체 (근사)** | $\mathcal{O}(T \times (S_1 + S_2) \times S_3 \times d)$ |

$(S_1 + S_2) \times S_3 \times d \ll T$ 이므로 **T에 대해 거의 선형**에 가깝습니다.

---

### 2.5 성능 향상 및 한계

#### 성능 향상

| 지표 | LFTSAD 결과 |
|------|------------|
| 처리 속도 | 딥 SOTA 모델 대비 **3~10배 빠름** |
| 파라미터 수 | 12.6K (ATran 4741K의 약 1/376) |
| RAM 사용량 | Raspberry Pi 4b에서 185MB (DCDet 402MB 대비 약 1/2) |
| CPU 사용량 | Jetson Xavier NX에서 38% (CSTGL 53.8%, GDN 62.2% 대비 낮음) |
| 정확도 순위 | 14개 데이터셋에서 8개 평가지표 중 절반 이상에서 1~2위 |
| 한 epoch 학습 시간(PC) | 26.5s (DTAAD 148s, DCDet 271.8s 대비 매우 빠름) |

#### 한계점 (논문 명시)

1. **$\alpha$ 파라미터 민감도**: 특정 데이터셋에서 $\alpha$ 값에 따른 성능 변동이 큼. 단변량 데이터셋은 $\alpha \in [0, 0.4]$, 다변량은 $\alpha \in [0.7, 1.0]$이 안정적이나 자동 결정 메커니즘이 부재
2. **변수 간 공간 상관관계 미반영**: 다변량 시계열에서 변수 독립 처리 가정으로 변수 간 관계 학습 불가
3. **온라인 업데이트 및 연합 학습 미지원**: 실시간 업데이트나 분산 배포 기능 부재
4. **전통적 F1 점수의 상대적 약세**: VUS-ROC, VUS-PR 등 고급 지표에서는 강세이나 단순 F1에서는 일부 딥 모델에 뒤처짐

---

## 3. 모델의 일반화 성능 향상 가능성

논문에서 일반화 성능 향상과 직간접으로 관련된 설계 요소를 중점적으로 분석합니다.

### 3.1 포인트 레벨 + 시퀀스 레벨 이중 관점의 상보성

이 설계는 일반화 성능의 핵심입니다. 논문은 이상치를 두 유형으로 분류합니다:

- **명시적 이상치(Explicit Outlier)**: 각 이상 타임스탬프 $x_1$의 포인트 레벨 연관 패턴이 정상 타임스탬프 $y_1$과 현저히 다름 → 포인트 레벨 학습으로 정확히 탐지 가능
- **암묵적 이상치(Implicit Outlier)**: 일부 이상 타임스탬프 $x_2$의 포인트 레벨 패턴이 정상 타임스탬프 $y_2$와 유사하여 오분류 위험 → 시퀀스 레벨에서 $x_2$ 중심 서브시퀀스로 확장하면 연관 패턴 차이가 드러남

이 이중 관점 접근은 **다양한 유형의 이상치에 대한 일반화**를 가능하게 합니다. 절제 실험(Ablation Study)에서:

$$F1_{PA}(\text{포인트만}) = 0.9321 < F1_{PA}(\text{시퀀스만}) = 0.9480 < F1_{PA}(\text{결합}) = 0.9721$$

(MSL 데이터셋 기준)

이는 두 구성 요소의 상보적 관계가 모델 일반화에 기여함을 정량적으로 보여줍니다.

### 3.2 재구성 불일치 기반 점수화의 일반화 효과

기존 재구성 오류 기반 방식은 얕은 MLP 구조에서 재구성 정확도 자체가 낮아 이상 감지 신뢰성이 떨어집니다. LFTSAD는 **두 재구성 값의 상대적 차이(불일치)**를 사용하므로, 절대적 재구성 정확도보다 **상대적 연관 패턴 차이**를 학습합니다. 이는 학습 데이터에 과적합되지 않고 다양한 도메인에서도 일반화 가능한 특성입니다.

절제 실험에서 재구성 불일치 기반 방식이 평균/최대 재구성 오류 기반 방식 대비 **평균 6.51% 향상**을 보였습니다.

### 3.3 변수 독립 병렬 처리의 도메인 일반화

포인트 레벨 학습은 다변량 시계열에서 각 변수를 독립적으로 병렬 처리합니다. 이는:
- 특정 변수 구조에 대한 과적합을 방지
- 변수 수가 다른 다양한 데이터셋에 구조 변경 없이 적용 가능
- 14개 이종 데이터셋(의료, 산업, 환경 등)에서 일관된 성능을 보인 근거

### 3.4 다양한 도메인에서의 실험적 검증

14개 데이터셋은 MSL(화성 탐사), GECCO(IoT 수질), ECG(심전도), UCR(범용), Occupancy(환경) 등 매우 이질적인 도메인을 포함합니다. LFTSAD가 이 모든 도메인에서 상위권 성능을 유지한다는 것은 **도메인 간 일반화 능력**의 실증적 근거입니다.

### 3.5 일반화 성능의 한계와 개선 방향

- **변수 간 상관관계 미반영**: 현재 모델은 변수 독립 가정으로 인해 변수 간 복잡한 상관관계를 가진 데이터셋(SWAT 51차원, WADI 127차원)에서 일부 딥 모델에 뒤처짐. **공간-시간 결합 학습 프레임워크** 도입 시 일반화 성능 추가 향상 가능
- **$\alpha$ 파라미터의 데이터 특성 의존성**: 단변량/다변량에 따라 최적 $\alpha$가 다름. **메타 학습(Meta-Learning)** 또는 **적응적 $\alpha$ 결정 메커니즘** 도입으로 자동화 가능
- **주파수 도메인 특징 미활용**: 논문의 향후 연구 방향으로 시간-주파수 학습 통합을 언급. 주파수 도메인 특징은 다양한 주기 패턴을 가진 시계열에서 일반화를 강화할 수 있음

---

## 4. 최신 연구 비교 분석 (2020년 이후)

### 4.1 주요 비교 모델 정리

| 모델 | 연도 | 핵심 기술 | 구조 깊이 | 파라미터 수 |
|------|------|-----------|-----------|------------|
| **USAD** [20] | 2020 | 이중 오토인코더 + 적대적 학습 | 심층 MLP | 213.6K |
| **OmniAnomaly** [41] | 2019 | VAE + 확률적 RNN | 심층 RNN | 189K |
| **NASA-LSTM** [42] | 2018 | LSTM + 비모수 동적 임계값 | 심층 LSTM | 249K |
| **GDN** [39] | 2021 | 그래프 신경망 기반 | 심층 GNN | 227.2K |
| **MAD-GAN** [40] | 2019 | 생성적 적대 신경망 | 심층 GAN | 214.6K |
| **AnomalyTrans (ATran)** [21] | 2022 | 연관 불일치 + Transformer | 심층 Transformer | 4741K |
| **DCDet** [22] | 2023 | 이중 주의 대조 학습 | 심층 Transformer | 301.5K |
| **PaAD** [19] | 2025 | Patch 기반 MLP-Mixer | 심층 MLP+Transformer | 1477.9K |
| **DTAAD** [34] | 2024 | 이중 TCN-어텐션 | 심층 TCN | 195.6K |
| **ADNEv** [35] | 2024 | 다층 신경 진화 프레임워크 | 얕은 다양한 구조 | 95.3K |
| **LTFAD** [33] | 2025 | 경량 All-MLP 시간-주파수 | 얕은 MLP | 84.2K |
| **FADSD** [32] | 2025 | 주파수 도메인 스펙트럼 불일치 | 얕은 MLP | 86.1K |
| **LFTSAD (Ours)** | 2025 | 포인트+시퀀스 재구성 불일치 | 얕은 MLP (4×2층) | **12.6K** |

### 4.2 기술적 접근 비교

#### 이상 점수화 방식의 진화

```
재구성 오류 기반          연관 불일치 기반              재구성 불일치 기반
(USAD, Omni, LSTM)  →  (ATran 2022, DCDet 2023)  →  (LFTSAD 2025)
     단순하나             고정밀도이나                  두 방식의 장점 융합:
     얕은 구조에서         연산 비용이 매우 높음          단순 구조로도
     한계 존재                                          효과적 이상 감지
```

#### LFTSAD vs. 연관 불일치 기반 모델 (ATran, DCDet)

ATran과 DCDet는 각 타임스탬프가 모든 이웃과의 연관값을 개별 계산합니다. $P_2$개의 이웃이 있으면 $P_2$개의 연관값을 계산해야 합니다. LFTSAD는 이를 **"이웃들로 해당 타임스탬프를 재구성할 수 있는가"**라는 이진적 판단으로 단순화합니다.

$$\text{ATran/DCDet}: \mathcal{O}(T^2) \text{ (self-attention)} \quad \text{vs.} \quad \text{LFTSAD}: \mathcal{O}(T \times (S_1+S_2) \times S_3 \times d)$$

#### LFTSAD vs. 경량 All-MLP 모델 (LTFAD, FADSD)

| 특성 | LTFAD [33] | FADSD [32] | LFTSAD |
|------|------------|------------|--------|
| 파라미터 수 | 84.2K | 86.1K | **12.6K** |
| 도메인 | 시간-주파수 | 주파수 | 시간 |
| 이상치 유형 대응 | - | - | 명시적+암묵적 모두 |
| 점수화 방식 | 재구성 오류 | 스펙트럼 불일치 | 재구성 불일치 |

LFTSAD는 세 모델 중 가장 적은 파라미터를 사용하면서 명시적·암묵적 이상치 모두를 명시적으로 처리합니다.

### 4.3 성능 비교 (MSL 데이터셋 기준)

| 모델 | Acc | Pre | V_ROC | 학습 1 epoch (PC) | RAM (Pi 4b) |
|------|-----|-----|-------|-------------------|-------------|
| LFTSAD | **0.9907** | **0.9509** | **0.9300** | 26.5s | 185MB |
| ATran | 0.9371 | 0.9339 | 0.5121 | 201.1s | 223MB |
| DCDet | 0.9542 | 0.9339 | 0.6520 | 271.8s | 402MB |
| DTAAD | 0.9814 | 0.8807 | 0.5426 | 148.0s | 461MB |
| USAD | 0.7299 | 0.7198 | 0.6102 | 178.1s | 223MB |

---

## 5. 앞으로의 연구에 미치는 영향과 고려할 점

### 5.1 앞으로의 연구에 미치는 영향

#### (1) 엣지-클라우드 협업 이상 탐지 패러다임 전환

LFTSAD는 TSAD 연구의 패러다임이 **"클라우드 중심 고정밀 탐지"**에서 **"엣지 배포 가능한 경량 고효율 탐지"**로 전환되어야 함을 강하게 시사합니다. 향후 연구에서는 엣지의 경량 모델과 클라우드의 정밀 분석을 결합하는 **계층적 탐지 아키텍처** 연구가 활성화될 것으로 예상됩니다.

#### (2) All-MLP 아키텍처의 TSAD 적용 가능성 확립

이 논문은 TSAD 분야에서 All-MLP 기반의 얕은 구조가 실용적으로 적용 가능함을 최초로 체계적으로 증명했습니다. 이는:
- 시계열 예측(Forecasting)에서 FITS[31], FreTS[30] 등이 보여준 MLP 효과성을 탐지(Detection) 분야로 확장
- 향후 더 정교한 All-MLP 기반 TSAD 연구(주파수 영역 통합, 그래프 구조 통합 등)를 촉진

#### (3) 이중 관점(포인트+시퀀스) 이상 탐지 프레임워크

명시적·암묵적 이상치를 각각의 관점으로 처리하는 이중 관점 프레임워크는 새로운 연구 방향을 제시합니다:
- 이상치 유형에 따른 **적응적 관점 선택** 메커니즘 연구
- 포인트·시퀀스·패치 등 **다중 스케일 관점의 통합** 연구

#### (4) 연합 학습(Federated Learning) 기반 경량 TSAD

논문이 한계로 언급한 연합 분산 배포는 Industry 4.0 환경에서 중요한 연구 과제입니다. LFTSAD의 경량 구조는 통신 오버헤드를 최소화하면서 연합 학습을 적용하기에 이상적인 기반을 제공합니다.

### 5.2 향후 연구 시 고려할 점

#### (1) 하이퍼파라미터 자동화

현재 $\alpha$, $P_1$, $P_2$, $S_1$, $S_2$, $S_3$ 등 여러 하이퍼파라미터가 그리드 서치로 결정됩니다. 실제 엣지 배포 환경에서는:
- **자동 하이퍼파라미터 최적화(AutoML)** 기법 통합 필요
- 데이터 특성(단변량/다변량, 이상치 비율 등)을 기반으로 $\alpha$를 자동 결정하는 **메타 러닝** 접근 고려

#### (2) 변수 간 공간 상관관계 모델링

현재 모델은 변수를 독립 처리하므로 고차원 다변량 데이터셋(WADI 127차원 등)에서 한계가 있습니다:
- **그래프 기반 공간 관계 모델링** (GDN 방식의 경량화)과 LFTSAD의 결합
- Attention 없이 MLP만으로 변수 간 관계를 포착하는 **경량 공간-시간 공동 학습** 프레임워크 설계 필요

#### (3) 온라인 학습 및 개념 표류(Concept Drift) 대응

산업 환경에서는 생산 공정 변화로 인한 데이터 분포 변화(개념 표류)가 빈번합니다:
- 배치 재학습 없이 실시간 모델 업데이트를 지원하는 **온라인 학습 메커니즘** 통합
- 연합 학습 환경에서의 안전한 모델 공유 프로토콜 설계

#### (4) 모델 압축 기법 통합

논문이 향후 연구로 언급한 **가지치기(Pruning) 및 양자화(Quantization)** 기법을 통해:
- 이미 경량화된 LFTSAD를 더욱 압축하여 더 저사양 MCU(Microcontroller Unit) 배포 가능
- 정밀도와 경량성의 추가적 균형점 탐색

#### (5) 시간-주파수 통합 학습

논문의 향후 연구 방향 중 하나인 주파수 도메인 학습 통합:
- FITS[31], FreTS[30]의 주파수 영역 처리 방식과 LFTSAD의 재구성 불일치 프레임워크 결합
- 주기성이 강한 산업 시계열에서 일반화 성능 추가 향상 기대

#### (6) 이상치 설명 가능성(Explainability)

엣지 디바이스에서의 이상 탐지는 단순한 탐지를 넘어 **왜 이상인지 설명**할 수 있어야 실제 산업 현장에서 활용 가능합니다:
- 포인트 레벨/시퀀스 레벨 불일치의 시각화를 통한 이상 원인 분석 기능 추가
- 경량 모델에서의 **설명 가능한 AI(XAI)** 기법 통합 연구 필요

#### (7) 표준화된 엣지 배포 벤치마크

현재 엣지 디바이스 실험이 Raspberry Pi 4b와 Jetson Xavier NX 두 플랫폼에 한정됩니다:
- Arduino, STM32 등 초저전력 MCU 환경으로의 확장 실험
- 배터리 소비량, 발열 등 실제 운용 지표를 포함한 **포괄적 엣지 배포 벤치마크** 표준화 기여 필요

---

## 참고 자료

**논문 원문:**
- Chen, L., Tang, J., Zou, Y., Liu, X., Xie, X., & Deng, G. (2025). "Lightweight and Fast Time-Series Anomaly Detection via Point-Level and Sequence-Level Reconstruction Discrepancy." *IEEE Transactions on Neural Networks and Learning Systems*, Vol. 36, No. 9, pp. 17295–17309. DOI: 10.1109/TNNLS.2025.3565807

**논문 내 주요 참고문헌 (직접 분석에 활용):**
- [19] Zhong, Z. et al. "PatchAD: A lightweight patch-based MLP-mixer for time series anomaly detection." arXiv:2401.09793, 2024.
- [20] Audibert, J. et al. "USAD: Unsupervised anomaly detection on multivariate time series." *KDD*, 2020.
- [21] Xu, J. et al. "Anomaly transformer: Time series anomaly detection with association discrepancy." *ICLR*, 2022.
- [22] Yang, Y. et al. "DCdetector: Dual attention contrastive representation learning for time series anomaly detection." *KDD*, 2023.
- [31] Xu, Z., Zeng, A., & Xu, Q. "FITS: Modeling time series with 10k parameters." *ICLR*, 2024.
- [32] Chen, L. et al. "Frequency-domain spectrum discrepancy-based fast anomaly detection for IIoT sensor time-series signals." *IEEE Trans. Instrum. Meas.*, 2025.
- [33] Chen, L. et al. "A lightweight all-MLP time–frequency anomaly detection for IIoT time series." *Neural Networks*, 2025.
- [34] Yu, L.-R. et al. "DTAAD: Dual TCN-attention networks for anomaly detection in multivariate time series data." *Knowl.-Based Syst.*, 2024.
- [35] Pietron, M. et al. "AD-NEv: A scalable multilevel neuroevolution framework for multivariate anomaly detection." *IEEE TNNLS*, 2024.
- [39] Deng, A. & Hooi, B. "Graph neural network-based anomaly detection in multivariate time series." *AAAI*, 2021.
- [5] Zamanzadeh Darban, Z. et al. "Deep learning for time series anomaly detection: A survey." *ACM Computing Surveys*, 2024.

**소스코드:** https://github.com/infogroup502/LFTSAD
