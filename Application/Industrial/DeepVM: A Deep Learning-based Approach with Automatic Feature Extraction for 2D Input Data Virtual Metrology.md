# DeepVM: A Deep Learning-based Approach with Automatic Feature Extraction for 2D Input Data Virtual Metrology

> **참고 자료**: Maggipinto, M., Beghi, A., McLoone, S., & Susto, G. A. (2019). "DeepVM: A Deep Learning-based Approach with Automatic Feature Extraction for 2D Input Data Virtual Metrology." *Journal of Process Control*. https://doi.org/10.1016/j.jprocont.2019.08.006

---

## 1. Executive Summary (10문장 이내)

DeepVM은 반도체 제조 공정에서 2차원 구조의 입력 데이터(OES 데이터)를 활용한 **가상 계측(Virtual Metrology, VM)** 을 위한 딥러닝 기반 자동 특징 추출 프레임워크이다.

> 💡 **Virtual Metrology (가상 계측)**: 물리적 측정 없이 공정 센서 데이터를 이용해 품질 지표(예: 식각률)를 예측하는 소프트웨어 기반 측정 기법

기존 VM 시스템은 수작업(hand-engineered) 특징 추출에 의존하여 확장성이 낮고, 도메인 전문 지식이 필수적이라는 한계가 있었다. DeepVM은 **합성곱 오토인코더(Convolutional Autoencoder)** 를 기반으로 특징 추출을 자동화하여 이 문제를 해결한다.

> 💡 **합성곱 오토인코더(Convolutional Autoencoder)**: 이미지처럼 2차원 데이터를 압축(인코딩)하고 복원(디코딩)하는 딥러닝 구조로, 압축 과정에서 중요한 특징을 자동으로 학습함

입력 데이터인 OES(광학 방출 분광) 데이터는 시간과 파장 두 축으로 진화하는 2차원 구조를 가지며, 이는 이미지와 유사한 특성을 지닌다. DeepVM은 CNN의 각 풀링 레이어에서 추출된 다층 특징을 연결(concatenate)하여 SVR 등의 회귀 모델에 입력하는 **다층 특징 추출 전략**을 채택한다.

N=1,554개 웨이퍼 데이터셋을 활용한 실험에서 DeepVM(표준 AE + SVR)은 $R^2 = 0.52$, $MSE = 2.34 \times 10^{-5}$로 모든 비교 방법론 대비 최고 성능을 달성하였다. 또한 예측값의 약 99%가 실제 식각률 ±10% 이내에 위치하여 실제 산업 적용 가능성을 보였다.

---

## 1-1. 연구의 목적과 필요성

| 구분 | 내용 |
|---|---|
| **산업적 배경** | Industry 4.0 환경에서 반도체 제조 공정 데이터가 폭발적으로 증가하고 있으며, 물리적 계측의 고비용 문제 해결 필요 |
| **기술적 한계** | 기존 VM의 특징 추출은 수작업 기반으로 시간 소모적이고, 도메인 전문 지식 의존적이며, 2D 데이터에는 표준화된 절차가 없음 |
| **연구 목적** | Convolutional Autoencoder를 활용한 자동 특징 추출로 2D OES 데이터 기반 식각률(etch rate) 예측 VM 시스템 구축 |
| **기대 효과** | 도메인 지식 없이도 적용 가능한 범용적·확장 가능한 VM 프레임워크 제시 및 비용 절감 |

> 💡 **OES (Optical Emission Spectroscopy, 광학 방출 분광법)**: 플라즈마 식각 공정 중 방출되는 빛의 파장별 강도를 측정하는 센서로, 공정 상태를 간접적으로 모니터링하는 데 활용됨

---

## 2. 핵심 주장과 근거 표

| 핵심 주장 | 근거 | 위치 |
|---|---|---|
| 2D OES 데이터에 대한 기존 VM 방법의 특징 추출 한계 존재 | 수작업 특징 추출의 확장성·이식성 부재, 자동 통계적 방법의 정보 손실 문제 지적 | Abstract, Section 1 |
| CNN 기반 오토인코더가 2D 데이터 특징 추출에 효과적 | 이미지와 유사한 OES 데이터 구조에 컴퓨터 비전 기법 적용 근거 | Section 1, 3 |
| 다층 특징 연결(multilayer feature concatenation)이 단일 레이어 특징보다 우수 | Table 3, 4, 5에서 All Layers > Last 2 Layers 성능 비교 | Section 4.3, Tables 3-5 |
| DeepVM(AE+SVR)이 모든 비교 방법론 중 최고 성능 달성 | $R^2=0.52$, $MSE=2.34\times10^{-5}$ (Table 1) | Table 1, Figure 13 |
| GPU 병렬화를 통한 실시간 적용 가능성 | CPU 대비 GPU 사용 시 10배 속도 향상 (37ms → 3.5ms) | Section 4.3, Table 6 |
| 도메인 지식 없이 범용 적용 가능 | SME 미포함 상태에서도 경쟁력 있는 성능 | Section 4.3, Section 5 |

---

## 2-1. 상세 설명

### 🔴 해결하고자 하는 문제

1. **2D 입력 데이터 처리 부재**: OES 데이터는 시간(Time)×파장(Wavelength)의 2차원 행렬 구조로, 기존 VM 알고리즘은 이를 직접 처리할 수 없음
2. **수작업 특징 추출의 한계**: 전문가 지식 의존, 시간 소모, 공정별 특화로 이식성 부재
3. **정보 손실**: 단순 통계 기반 자동 특징 추출(평균, 분산 등)은 원데이터의 복잡한 패턴을 포착하지 못함

---

### 🟢 제안하는 방법 및 수식

#### ① Feedforward Neural Network (FNN) 기본 연산

$$y^l = \sigma(W^l y^{l-1} + b^l) $$

| 기호 | 설명 |
|---|---|
| $y^l$ | $l$번째 레이어의 출력 벡터 |
| $W^l$ | $l$번째 레이어의 가중치 행렬 ($q^l \times q^{l-1}$ 크기) |
| $b^l$ | $l$번째 레이어의 편향(bias) 벡터 ($q^l$ 길이) |
| $\sigma(\cdot)$ | 비선형 활성화 함수 (element-wise 적용) |
| $y^{l-1}$ | 이전 레이어의 출력 벡터 ($q^{l-1}$ 차원) |

> 💡 **활성화 함수(Activation Function)**: 뉴럴 네트워크에 비선형성을 추가하는 함수. 이 비선형성이 없으면 레이어를 아무리 쌓아도 단순 선형 변환에 불과함

#### ② ReLU 활성화 함수

$$\sigma(x) = \max(0, x) $$

> 💡 **ReLU (Rectified Linear Unit)**: 입력이 0보다 작으면 0, 크면 그대로 출력하는 함수. 기울기 소실(Vanishing Gradient) 문제를 완화함

#### ③ 합성곱 레이어 연산

$$y^l = \sigma(W^l * y^{l-1} + b^l) $$

| 기호 | 설명 |
|---|---|
| $*$ | 합성곱(Convolution) 연산 |
| $W^l$ | 합성곱 커널(Kernel) |
| $b^l$ | 편향 항 |

> 💡 **합성곱 연산(Convolution)**: 작은 필터(커널)를 슬라이딩하며 지역적 패턴을 추출하는 연산. 이미지에서 엣지, 텍스처 등을 자동으로 감지함

#### ④ VAE의 변분 하한(Variational Lower Bound, ELBO)

$$\mathcal{L}(\theta, \phi) = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}\left[\log p_\theta(\mathbf{x}|\mathbf{z})\right] - D_{KL}\left(q_\phi(\mathbf{z}|\mathbf{x}) \,\|\, p(\mathbf{z})\right) $$

| 기호 | 설명 |
|---|---|
| $\theta$ | 디코더 신경망의 파라미터 |
| $\phi$ | 인코더 신경망의 파라미터 |
| $q_\phi(\mathbf{z} \mid \mathbf{x})$ | 인코더가 근사하는 사후 분포 |
| $p_\theta(\mathbf{x} \mid \mathbf{z})$ | 디코더의 조건부 우도 |
| $p(\mathbf{z})$ | 잠재 변수의 사전 분포 (주로 표준 정규분포) |
| $D_{KL}(\cdot \| \cdot)$ | KL 발산(Kullback-Leibler Divergence): 두 분포 간의 차이 측정 |

> 💡 **KL 발산(KL Divergence)**: 두 확률 분포가 얼마나 다른지를 측정하는 값. VAE에서는 학습된 잠재 분포가 표준 정규분포에서 너무 벗어나지 않도록 제약하는 정규화 역할을 함
>
> 💡 **변분 추론(Variational Inference)**: 계산이 불가능한 사후 분포를 다루기 쉬운 분포로 근사하는 기법

#### ⑤ SVR 최적화 문제 (기본형)

$$\min \frac{1}{2}\|w\|^2 \quad \text{subject to} \begin{cases} y_i - \langle w, x_i \rangle - b \leq \epsilon \\ \langle w, x_i \rangle + b - y_i \leq \epsilon \end{cases} $$

#### ⑥ SVR 최적화 문제 (슬랙 변수 포함)

$$\min \frac{1}{2}\|w\|^2 \quad \text{subject to} \begin{cases} y_i - \langle w, x_i \rangle - b \leq \epsilon + \xi_i \\ \langle w, x_i \rangle + b - y_i \leq \epsilon + \xi_i^* \end{cases} $$

| 기호 | 설명 |
|---|---|
| $w$ | 초평면의 가중치 벡터 |
| $b$ | 편향 항 |
| $\epsilon$ | 허용 오차 범위 (ε-tube 폭) |
| $\xi_i, \xi_i^*$ | 슬랙 변수: ε-tube 밖의 오차를 허용하는 완화 변수 |
| $\langle w, x_i \rangle$ | 내적(Inner Product) |

> 💡 **SVR (Support Vector Regression)**: 예측값이 실제값에서 ε 이내에 들어오도록 하는 초평면을 찾는 회귀 방법. 과적합에 강건하고 소규모 데이터셋에 효과적

---

### 🔵 모델 구조

```
[Raw OES Data (2D: Time × Wavelength)]
         ↓
[Conv Layer 1] → [Avg Pooling 1] → Flatten → ┐
         ↓                                    │
[Conv Layer 2] → [Avg Pooling 2] → Flatten → │→ Concat → X̄ (특징 벡터)
         ↓                                    │
[Conv Layer 3] → [Avg Pooling 3] → Flatten → ┘
                    ↓ (비지도 사전학습 후)
             [Fine-tuning (지도 학습)]
                    ↓
              [SVR / LASSO / Ridge]
                    ↓
            [VM Prediction (Etch Rate)]
```

- **특징 벡터 크기**: 원본 데이터의 1/3 크기로 압축
- **Avg Pooling 사용 이유**: Max-Pooling 대비 부드러운(smooth) 특징 추출 → 회귀 문제에 적합
- **세미-지도 학습 방식**: 오토인코더(비지도) 사전학습 → 회귀 모델(지도) 파인튜닝

> 💡 **세미-지도 학습(Semi-Supervised Learning)**: 레이블 없는 데이터(비지도)와 레이블 있는 데이터(지도)를 모두 활용하는 학습 방식. 레이블 데이터가 부족한 산업 환경에 적합

---

### 🟡 성능 향상 및 한계

**성능 향상**:
- DeepVM(AE+SVR): $R^2=0.52$, $MSE=2.34\times10^{-5}$ (Table 1, p.8)
- Fused LASSO 대비 $R^2$ 약 24% 향상 ($0.42 \to 0.52$)
- GPU 사용 시 실행 시간: 3.52ms (실시간 적용 가능 수준)

**한계**:
- 단일 챔버, 단일 공정 데이터셋만 사용 → 일반화 검증 미흡
- $R^2$ 최고값이 0.52로 절대적 수치는 낮음 (다단계 공정의 조기 예측 문제)
- VAE의 경우 데이터 부족으로 성능 저하
- 학습 시간 약 23시간 (Titan Xp GPU 기준) → 실시간 모델 업데이트 불가

---

## 3. 각 주장의 위치 (페이지/Figure/Table 번호)

| 주장 | 위치 |
|---|---|
| 기존 특징 추출의 한계 (수작업 vs 자동) | p.2, Section 1, Figure 1 |
| DeepVM 전체 아키텍처 | p.3, Figure 2 |
| 다층 특징 추출 절차 | p.4, Figure 3 |
| 개발-생산 파이프라인 | p.4, Figure 4 |
| FNN 구조 및 수식 (1), (2) | p.4, Figure 5, 6 |
| CNN 합성곱 수식 (3) | p.4-5 |
| Max-pooling 시각화 | p.5, Figure 7 |
| 오토인코더 구조 | p.5-6, Figure 8 |
| VAE Bayesian 네트워크 | p.6, Figure 9 |
| SVR 수식 (5), (6) | p.6 |
| OES 데이터 예시 | p.7-8, Figure 11, 12 |
| **최고 성능 (AE+SVR)** | **p.8, Table 1** |
| 오차 비율 (5%, 10%) | p.8, Table 2 |
| 다층 vs 2레이어 특징 비교 | p.9-10, Tables 3-5 |
| Boxplot 성능 분포 | p.9, Figure 13 |
| 산점도 예측 결과 | p.10, Figure 14 |
| 실행 시간 비교 | p.11, Table 6 |

---

## 4. 저자 보고 결과 vs 분석자 해석 분리

### 저자가 직접 보고한 결과

| 항목 | 저자 보고 내용 |
|---|---|
| **최고 성능 모델** | AE+SVR: $R^2=0.52\pm0.08$, $MSE=2.34\pm0.56\times10^{-5}$ (Table 1) |
| **±10% 이내 예측 비율** | 모든 방법론 약 99% (Table 2) |
| **±5% 이내 예측 비율** | DeepVM이 최고 91.01% (VAE+Ridge, Table 2) |
| **실행 시간** | CPU: 37ms, GPU: 3.5ms (Table 6) |
| **학습 시간** | 전체 교차검증 약 23시간 (Titan Xp GPU) |
| **$R^2$ 낮은 이유** | 다단계 공정의 조기 예측 문제 (p.10) |

### 분석자(본 보고서) 해석

| 항목 | 해석 |
|---|---|
| $R^2 = 0.52$ 의 의미 | ⚠️ 절대적 수치는 낮으나, 저자 설명대로 다단계 공정의 1단계만으로 최종 계측값을 예측하는 태스크 특성상 불가피한 한계. 비교 방법론 대비 상대적 우위가 핵심 |
| VAE 성능 저조 | KL 발산 항의 병목 현상 + 1,554개 샘플의 데이터 부족이 복합적으로 작용한 것으로 판단 |
| Ridge > LASSO 성능 | 특징 간 다중공선성(Collinearity)이 높다는 간접 증거 → 저자도 p.11에서 언급 |
| 다층 특징의 우수성 | Tables 3-5 결과에서 일관되게 All Layers > Last 2 Layers → 낮은 레이어의 저수준 특징도 회귀에 유용함을 시사 |

---

## 5. 통계적으로 취약한 부분 및 비교 불가능한 수치

| 구분 | 항목 | 문제점 |
|---|---|---|
| ⚠️ **단일 데이터셋** | N=1,554 웨이퍼, 단일 챔버 | 외부 유효성(External Validity) 검증 불가. 다른 챔버·공정에서의 성능 미확인 |
| ⚠️ **높은 표준편차** | AE+SVR: $R^2=0.52\pm0.08$, DAE+SVR: $R^2=0.48\pm0.15$ | 표준편차가 평균 대비 크게 나타남 → 모델 안정성 불확실 |
| ⚠️ **공개 벤치마크 없음** | 저자 직접 언급: "no publicly available datasets" (p.7) | 타 연구와 절대적 성능 비교 불가 |
| ⚠️ **베이스라인 불균형** | 비교 모델 중 SVR, LASSO, Ridge는 단순 통계 특징만 사용 | ConvNet은 완전 지도 학습 방식으로 설계 철학이 다름 → 공정한 비교 한계 |
| ⚠️ **VAE 하이퍼파라미터 미탐색** | KL 가중치($\beta$) 튜닝 미실시 | VAE 성능이 과소평가되었을 가능성 |
| ⚠️ **통계 검정 부재** | 방법론 간 성능 차이에 대한 통계적 유의성 검정(t-test 등) 없음 | 성능 차이가 우연일 가능성 배제 불가 |

> 💡 **다중공선성(Collinearity)**: 독립변수들 간에 강한 선형 관계가 존재하는 현상. 회귀 모델의 안정성을 저하시킴

---

## 6. 문서가 답하지 않는 질문

| 번호 | 미해결 질문 |
|---|---|
| 1 | 다른 반도체 제조 공정(CVD, CMP 등)이나 다른 장비에서의 DeepVM 성능은? |
| 2 | 최적 오토인코더 구조(레이어 수, 커널 크기, 필터 수)는 어떻게 결정되었는가? (하이퍼파라미터 탐색 과정 미제시) |
| 3 | OES 데이터 외 다른 2D 데이터(예: 웨이퍼 맵 이미지)에도 동일하게 적용 가능한가? |
| 4 | 레이블 부족 시나리오(극소수 레이블)에서 세미-지도 학습의 이점이 얼마나 되는가? |
| 5 | 공정 드리프트(Process Drift) 발생 시 모델 재학습 주기와 성능 저하 패턴은? |
| 6 | VAE의 KL 가중치($\beta$)를 조정하면 성능이 얼마나 개선되는가? |
| 7 | 특징 추출된 잠재 공간(Latent Space)의 물리적 의미는 무엇인가? |
| 8 | 다단계 공정 전체를 고려한 통합 VM 모델 설계 시 성능은? |
| 9 | 소규모 데이터셋(N<500)에서 DeepVM 성능은 얼마나 저하되는가? |
| 10 | 모델 예측의 불확실성(Uncertainty Quantification) 추정이 가능한가? |

---

## 7. 중요 그림 5개 해석

### Figure 1 (p.2): 고전 ML vs 딥러닝의 특징 추출 비교

```
Classic ML: 수작업 특징 → 매핑 → 출력
           or 자동 특징 → 매핑 → 출력

Deep Learning: 단순 특징 → 추상 특징 레이어 → 매핑 → 출력
```

**해석**: 딥러닝은 원시 데이터에서 점진적으로 추상화된 표현을 학습하므로, 수작업 특징 엔지니어링이나 단순 자동 통계 특징보다 정보 보존력이 높음을 시각화. 본 논문의 핵심 동기를 명확히 제시.

---

### Figure 3 (p.4): DeepVM 특징 추출 절차

**해석**: 3개의 합성곱-평균풀링 레이어 블록에서 각각 특징을 추출하여 Flatten 후 Concatenate하는 **다층 특징 벡터 구성** 전략을 도식화. 기존 연구들이 마지막 레이어 특징만 활용한 것과 달리, 저수준(시간·공간적 세부 정보)부터 고수준(전체 패턴) 특징까지 모두 포착하는 핵심 기여를 보여줌.

---

### Figure 11 & 12 (p.7-8): OES 데이터 시각화

**해석**:
- **Figure 11**: OES 데이터의 3D 표면 플롯 — 시간(Time)×파장(Wavelength index)×강도(Intensity)로 구성된 2차원 구조가 이미지와 유사함을 직관적으로 제시
- **Figure 12(좌)**: 특정 시간 단면의 파장별 강도 — 복잡한 스펙트럼 패턴 확인
- **Figure 12(우)**: 특정 파장의 시간별 강도 — 공정 진행에 따른 급격한 변화(식각 시작/종료 구간) 포착

→ 이 데이터 구조가 단순 통계 특징으로는 포착 불가능하며 CNN 기반 접근이 필요함을 실증적으로 보여줌.

---

### Figure 13 (p.9): 20회 MCCV 성능 분포 Boxplot

**해석**:
- **MSE 기준**: DeepVM(AE+SVR)의 중앙값이 가장 낮고 박스 폭(IQR)도 상대적으로 작아 안정적
- **$R^2$ 기준**: ConvNet의 이상치(빨간 +)가 매우 낮게 관찰 → 특정 fold에서 크게 실패하는 불안정성 확인
- **전반적 패턴**: DeepVM 계열이 SVR/LASSO/Ridge/ConvNet 대비 중앙값과 분산 모두에서 우수

⚠️ **주의**: 방법론 간 통계적 유의성 검정이 없어 일부 성능 차이는 우연일 수 있음

> 💡 **MCCV (Monte Carlo Cross Validation)**: 데이터를 무작위로 훈련/테스트로 분할하는 과정을 여러 번 반복하여 모델 성능의 통계적 분포를 추정하는 교차검증 방법

---

### Figure 14 (p.10): 예측값 vs 실제값 산점도

**해석**:
- 이상적인 예측: 모든 점이 $y=x$ 대각선 위에 위치
- **DeepVM 계열** (특히 DAE+Ridge, $R^2=0.57$): 점들이 $y=x$ 선에 비교적 밀집
- **통계 특징 기반** (Stats Ridge/LASSO): 점들이 더 넓게 분산되고, 극단값에서 예측 편향 발생
- 전 방법론에 걸쳐 중간 범위($\approx 0.12 \sim 0.13$)에 데이터가 집중 → **클래스 불균형**과 유사한 데이터 분포 편향 존재

---

## 8. 결론: 시사점, 후속 연구, 추가 연구 방향

### 8-1. 연구자 제시 시사점 및 후속 연구 계획

| 구분 | 내용 |
|---|---|
| **핵심 시사점** | DeepVM은 도메인 지식 없이 2D 산업 데이터에서 자동 특징 추출 가능한 범용 VM 프레임워크 |
| **확장 가능성** | 테이블형 데이터와 병행 사용 시 AE 특징 + 테이블 데이터 연결로 확장 가능 |
| **부가 활용** | 오토인코더를 데이터 압축, 이상 탐지, 품질 모니터링에도 활용 가능 |
| **모델 갱신** | 드리프트 발생 시 일 1회 주기 재학습으로 대응 가능 (전이학습 활용) |
| **향후 연구** | FSCA[53] 등 특징 선택 방법론 통합 탐색 예정 |

> 💡 **FSCA (Forward Selection Component Analysis)**: 예측력이 높은 특징을 순차적으로 선택하는 방법으로, 과적합 방지 및 실행 시간 단축에 유리

---

### 8-1 심화: 모델의 일반화 성능 향상 가능성

논문에서 직접 언급된 일반화 관련 내용과 추가 가능 방향을 다음과 같이 정리한다.

| 전략 | 논문 내 근거 | 기대 효과 |
|---|---|---|
| **전이학습(Transfer Learning)** | "초기 레이어는 일반적 특징 학습" (p.11) 언급 | 유사 공정/챔버 간 사전학습 가중치 재활용 → 데이터 효율성 향상 |
| **데이터 증강(Data Augmentation)** | 미언급 | OES 데이터에 노이즈 추가, 시간축 왜곡 등으로 학습 데이터 다양화 |
| **VAE 정규화 강도 조정** | KL 가중치 미탐색 언급 (p.7) | $\beta$-VAE 방식으로 잠재 공간 disentanglement 개선 → 과적합 억제 |
| **배치 정규화(Batch Normalization)** | 미언급 | 레이어별 분포 안정화 → 다른 공정 데이터에도 안정적 수렴 |
| **도메인 적응(Domain Adaptation)** | 미언급 | 소스 챔버→타겟 챔버 간 분포 불일치 보정 → 챔버 간 이식성 향상 |

> 💡 **전이학습(Transfer Learning)**: 한 도메인에서 학습된 모델의 지식을 다른 도메인 문제에 활용하는 기법. 데이터가 부족한 상황에서 특히 효과적

> 💡 **배치 정규화(Batch Normalization)**: 각 레이어의 입력 분포를 정규화하여 학습 안정성을 높이고 과적합을 억제하는 기법

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **중요 고지**: 아래 내용은 본 논문 제출 시점(2019년) 이후의 연구 동향에 대한 **일반적인 분야 발전 방향**을 제시하는 것입니다. 구체적 논문명·수치는 본 논문 원문에 없으므로, **확인 가능한 범위 내에서만 기술**하며 불확실한 수치는 제시하지 않습니다.

#### 분야별 발전 방향

| 연구 방향 | DeepVM과의 관계 | 발전 내용 |
|---|---|---|
| **Transformer 기반 VM** | DeepVM의 CNN 특징 추출을 대체 가능 | Self-Attention 메커니즘으로 시간-파장 간 전역적 의존관계 포착 가능 |
| **그래프 신경망(GNN) 기반 VM** | 다변수 센서 간 관계 모델링 | 공정 단계 간 인과관계를 그래프로 표현하여 다단계 공정 예측 개선 |
| **불확실성 정량화(UQ)** | DeepVM은 점 예측만 제공 | Bayesian DL, MC Dropout으로 예측 신뢰구간 제공 → 산업 안전성 향상 |
| **연속 학습(Continual Learning)** | 주기적 재학습 언급만 | 공정 드리프트에 대응하는 온라인 학습 체계 구축 |
| **설명 가능한 AI(XAI)** | 블랙박스 문제 미해결 | SHAP, Grad-CAM 등으로 특징 중요도 시각화 → 공정 엔지니어 신뢰 확보 |

> 💡 **Self-Attention (자기 주의 메커니즘)**: 입력 시퀀스의 모든 위치 간의 관계를 동시에 계산하는 기법. CNN이 지역적 패턴만 보는 것과 달리 전역적 패턴 포착 가능
>
> 💡 **MC Dropout**: 추론 시에도 드롭아웃을 활성화하여 여러 번 예측함으로써 불확실성을 추정하는 기법

#### DeepVM이 이후 연구에 미치는 영향

1. **2D 산업 데이터 처리의 선례**: OES 데이터를 이미지로 취급하는 패러다임 제시
2. **다층 특징 연결 전략**: 단일 레이어 특징보다 다층 특징 활용이 유리함을 실증
3. **세미-지도 학습의 VM 적용 가능성 제시**: 레이블 부족 문제 해결의 방향성 제공
4. **코드 공개**: 재현 가능한 연구(Reproducible Research) 문화 기여

#### 향후 연구 시 고려할 점

| 고려 사항 | 구체적 제언 |
|---|---|
| **데이터 규모** | 다양한 챔버·공정 데이터를 통합한 대규모 벤치마크 데이터셋 구축 필요 |
| **평가 지표 다양화** | $R^2$, MSE 외에 MAPE, 예측 구간 커버리지 등 산업 맥락 지표 추가 |
| **계산 효율성** | 경량화(Pruning, Quantization)를 통한 엣지 디바이스 배포 가능성 탐색 |
| **물리 정보 통합** | Physics-Informed Neural Network(PINN)과 결합하여 데이터 효율성 향상 |
| **공정 간 전이** | 챔버 A→챔버 B, 또는 식각→증착 공정 간 전이학습 체계 수립 |
| **불확실성 정량화** | VM 예측의 신뢰 구간 제공 → 공정 제어 의사결정 신뢰성 향상 |

> 💡 **Physics-Informed Neural Network (PINN)**: 물리 법칙(편미분방정식 등)을 손실함수에 통합하여 데이터가 부족해도 물리적으로 타당한 예측을 하는 신경망
>
> 💡 **모델 경량화(Pruning/Quantization)**: 딥러닝 모델의 불필요한 가중치를 제거(Pruning)하거나 정밀도를 낮춰(Quantization) 연산량을 줄이는 기법. 엣지 기기 배포에 필수적

---

## 참고 자료

1. **본 논문**: Maggipinto, M., Beghi, A., McLoone, S., & Susto, G. A. (2019). "DeepVM: A Deep Learning-based Approach with Automatic Feature Extraction for 2D Input Data Virtual Metrology." *Journal of Process Control*. https://doi.org/10.1016/j.jprocont.2019.08.006

2. **딥러닝 기초 이론**: Goodfellow, I., Bengio, Y., Courville, A. (2016). *Deep Learning*. MIT Press. [논문 참고문헌 36]

3. **VAE 원논문**: Kingma, D. P., & Welling, M. "Auto-encoding variational bayes." arXiv:1312.6114 [논문 참고문헌 38]

4. **Denoising AE**: Vincent, P., et al. (2010). "Stacked denoising autoencoders." *Journal of Machine Learning Research*, 11, 3371–3408. [논문 참고문헌 37]

5. **Fused LASSO**: Tibshirani, R., et al. (2005). "Sparsity and smoothness via the fused lasso." *Journal of the Royal Statistical Society: Series B*, 67(1), 91–108. [논문 참고문헌 25]

6. **SVR**: Basak, D., Pal, S., & Patranabis, D. C. (2007). "Support vector regression." *Neural Information Processing-Letters and Reviews*, 11(10), 203–224. [논문 참고문헌 39]

7. **VM 최초 제안**: Chen, P., et al. (2005). "Virtual metrology: A solution for wafer to wafer advanced process control." *ISSM 2005*. [논문 참고문헌 10]
