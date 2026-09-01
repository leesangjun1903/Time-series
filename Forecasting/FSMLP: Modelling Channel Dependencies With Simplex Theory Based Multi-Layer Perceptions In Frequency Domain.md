# FSMLP: Modelling Channel Dependencies With Simplex Theory Based Multi-Layer Perceptions In Frequency Domain

> **⚠️ 정확도 고지**: 본 분석은 제공된 PDF 원문에 기반합니다. 논문에 명시되지 않은 내용은 추정임을 명시하며, 2020년 이후 최신 연구 비교는 제 학습 데이터(~2024년 초) 범위 내에서 제공됩니다.

---

## 1. Executive Summary (10문장 이내)

FSMLP는 시계열 예측(TSF)에서 채널 간 의존성 모델링 시 MLP가 과적합되는 문제를 해결하기 위해 제안된 프레임워크이다.  
저자들은 **Rademacher 복잡도 이론**을 활용하여, 시계열 데이터의 극단값(extreme values)이 MLP의 과적합을 악화시킨다는 것을 이론적으로 규명하였다.  
이를 해결하기 위해 가중치를 **표준 n-심플렉스(Standard n-simplex)** 내로 제한하는 **Simplex-MLP** 레이어를 제안하였다.  
Simplex-MLP의 Rademacher 복잡도 상한은 기존 MLP보다 $B$배 작아 이론적으로 과적합 감소가 보장된다.  
FSMLP는 **SCWM(Simplex Channel-Wise MLP)**과 **FTM(Frequency Temporal MLP)** 두 모듈로 구성되며, 채널 의존성과 시간적 의존성을 주파수 도메인에서 동시에 포착한다.  
7개의 벤치마크 데이터셋에서 12개의 SOTA 모델 대비 우수한 성능을 달성하였다.  
추론 속도, 메모리 효율, 계산 복잡도( $O(NL)$ ) 측면에서도 경쟁력 있는 결과를 보인다.  
또한 Simplex-MLP는 TSMixer, Autoformer 등 기존 모델에 적용 시에도 성능 향상을 가져오는 범용성을 입증하였다.

---

### 1-1. 연구의 목적과 필요성

| 구분 | 내용 |
|------|------|
| **핵심 문제** | 채널 간 의존성 모델링에 MLP를 사용할 때 발생하는 과적합 |
| **원인 분석** | 시계열 데이터의 극단값 → Rademacher 복잡도 증가 → 일반화 성능 저하 |
| **기존 방법의 한계** | TSMixer, Autoformer, TimesNet 등은 훈련 손실은 감소하나 검증 손실이 높음 (Fig. 1) |
| **필요성** | 채널 의존성을 효과적으로 포착하면서도 과적합을 억제하는 MLP 설계 필요 |

> 🔍 **Rademacher 복잡도(Rademacher Complexity)**: 모델이 랜덤 노이즈에 얼마나 잘 맞출 수 있는지를 측정하는 지표. 값이 낮을수록 과적합 경향이 적음을 의미.

---

## 2. 핵심 주장과 근거 표

| # | 핵심 주장 | 근거 유형 | 구체적 근거 | 위치 |
|---|-----------|-----------|-------------|------|
| 1 | MLP의 채널 의존성 모델링 시 극단값이 과적합 유발 | 이론 + 실험 | Rademacher 복잡도 분석 + Table I (극단값 비율) | p.2, Table I |
| 2 | Simplex-MLP가 Rademacher 복잡도 상한을 $B$배 감소 | 수학적 증명 | Theorem 1 증명 | p.5-6 |
| 3 | 주파수 도메인 모델링이 노이즈 감소에 효과적 | 이론 + 실험 | 주기성 분리 원리 + Ablation (Table IV) | p.5, Table IV |
| 4 | FSMLP가 7개 데이터셋에서 SOTA 달성 | 실험 | Table III 전체 결과 | p.7, Table III |
| 5 | Simplex-MLP는 다른 모델에도 적용 가능 | 실험 | Table VI (TSMixer, Autoformer 개선) | p.10, Table VI |
| 6 | FSMLP가 선형 계산 복잡도 $O(NL)$ 달성 | 분석 | Table XIV | p.11, Table XIV |
| 7 | 로그 변환이 최적의 심플렉스 구현 | 실험 | Table VII | p.8-9, Table VII |

---

### 2-1. 상세 설명

#### 🔴 해결하고자 하는 문제

채널 혼합(channel-mix) 방식의 MLP 모델들(TSMixer, Autoformer 등)은 채널 간 의존성을 모델링할 때 다음 문제에 직면한다:

1. **극단값 문제**: 표준 MLP의 비제한 가중치가 시계열 극단값에 과적합됨
2. **Rademacher 복잡도 증가**: 가중치 노름 $B = \|w\|_2$가 커질수록 일반화 성능 저하
3. **시간 도메인 노이즈**: 직접적인 시간 도메인에서의 채널 의존성 모델링 시 노이즈 포함

**Table I** (p.2)에서 주요 데이터셋의 극단값 비율 확인:
- 대부분 값이 $\sigma$ 이내에 분포하지만, $3\sigma$ 초과 이상치도 존재 (ETTh2: 0.99%, Traffic: 1.59%)

> 🔍 **σ (표준편차)**: 데이터 분포의 퍼짐 정도. $3\sigma$를 넘는 값은 통계적으로 이상치(outlier)로 간주됨.

---

#### 🔴 제안하는 방법 (수식 포함)

**① 표준 MLP의 Rademacher 복잡도 상한** (p.5):

$$\mathcal{R}_S(\mathcal{H}) \leq \frac{B}{m} \sqrt{\sum_{i=1}^{m} \|x^{(i)}\|_2^2}$$

- $\mathcal{H}$: MLP의 가설 클래스
- $w \in \mathbb{R}^d$: 모델 가중치 파라미터
- $d$: 입력 차원
- $m$: 훈련 데이터 포인트 수
- $x^{(i)} \in \mathbb{R}^d$: $i$번째 입력 데이터 포인트
- $\|x^{(i)}\|_2$: $i$번째 데이터의 $\ell_2$-노름
- $B = \|w\|_2$: 가중치 벡터 노름의 상한 ($\gg 1$)

**② Simplex-MLP의 Rademacher 복잡도 상한** (Theorem 1, p.6):

$$\mathcal{R}_S(\mathcal{H}_\Delta) \leq \frac{1}{m} \sqrt{\sum_{i=1}^{m} \|x^{(i)}\|_2^2}$$

- $\mathcal{H}_\Delta$: Simplex-MLP의 가설 클래스 ($w \in \Delta^n$으로 제한)
- $B$ 항이 사라짐 → 기존 MLP 대비 $B$배 감소

> **핵심**: $B \gg 1$이므로 Simplex-MLP의 복잡도 상한이 훨씬 낮음

**③ 표준 n-심플렉스 정의** (p.3, 수식 1-3):

```math
\Delta^n = \left\{ w \in \mathbb{R}^{n+1} \;\middle|\; \sum_{i=0}^{n} w_i = 1 \text{ and } w_i \geq 0 \text{ for all } i \right\}
```

- $w_i$: 심플렉스의 $i$번째 좌표값
- 표준 기저 벡터 $\mathbf{e}_i = (0, \ldots, 0, 1, 0, \ldots, 0)$의 볼록 껍질(convex hull)

> 🔍 **표준 n-심플렉스(Standard n-Simplex)**: 모든 좌표가 0 이상이며 합이 1인 점들의 집합. 2D에서는 삼각형, 3D에서는 사면체로 시각화 가능. 확률 단체(probability simplex)와 동일한 구조.

> 🔍 **볼록 껍질(Convex Hull)**: 주어진 점들을 모두 포함하는 가장 작은 볼록 집합.

**④ Simplex-MLP 연산** (p.4, 수식 5):

$$X_{\text{out}} = \text{Matmul}(X_{\text{in}},\; f_{\text{sim}}(W)) + b$$

- $X_{\text{in}}$: 입력 텐서
- $W$: 원본 가중치 행렬
- $f_{\text{sim}}(W)$: 심플렉스로 제약된 가중치
- $b$: 바이어스

**$f_{\text{sim}}$ 세부 구현:**

$$f_{\text{sim}}(W_{i,j}) = f_{\text{norm}}(f_{\text{trans}}(W_{i,j})) = \frac{f_{\text{trans}}(W_{i,j})}{\sum_{j=1}^{N} f_{\text{trans}}(W_{i,j})}$$

세 가지 변환 옵션:

$$f_{\text{trans}}^{\text{abs}}(W_{i,j}) = |W_{i,j}|$$

$$f_{\text{trans}}^{\text{log}}(W_{i,j}) = \log(|W_{i,j}| + 1) \quad \text{(기본값)}$$

$$f_{\text{trans}}^{\text{square}}(W_{i,j}) = W_{i,j}^2$$

> 🔍 **로그 변환이 기본값인 이유**: $\log$ 함수의 도함수가 역함수($1/x$)이므로, 가중치가 클수록 그래디언트가 작아져 가중치의 급격한 성장을 억제함.

**⑤ SCWM 블록 연산** (p.5):

$$Z^l_{\text{Channel}} = \sigma(f_{\text{sim}}(Z^{l-1}_{\text{SCWM}})) + Z^{l-1}_{\text{SCWM}}$$

$$Z^l_{\text{SCWM}} = \sigma(\text{MLP}(Z^l_{\text{Channel}})) + Z^l_{\text{Channel}}$$

- $Z^{l-1}_{\text{SCWM}}$: $l-1$번째 SCWM 블록의 출력
- $\sigma$: 활성화 함수
- 잔차 연결(residual connection) 포함

**⑥ FTM 블록 연산** (p.5):

$$Z^i_{\text{FTM}} = \sigma(\text{Linear}(Z^{i-1}_{\text{FTM}})) + Z^{i-1}_{\text{FTM}}$$

$$\hat{Y} = \text{Linear}(Z^N_{\text{FTM}})$$

- $\hat{Y}$: 최종 예측값

**⑦ 손실 함수** (p.5, 수식 6):

$$\mathcal{L}_{\text{time}} = \frac{\sum_{i=1}^{\tau} \|Y_i - F(X)_i\|^2}{\tau}$$

$$\mathcal{L}_{\text{fre}} = \frac{\sum_{i=1}^{\tau} \|Y_i - F(X)_i\|}{\tau}$$

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{time}} + \mathcal{L}_{\text{fre}}$$

- $\tau$: 예측 기간(horizon)
- $Y_i$: 실제값, $F(X)_i$: 예측값
- 시간 도메인: MSE 손실, 주파수 도메인: MAE 손실 (주파수 성분 간 크기 차이 때문)

---

#### 🔴 모델 구조 (Fig. 2, p.4)

```
입력 X (N×L)
    ↓ Transpose + Frequency Transformation (DCT)
    ↓
[SCWM 블록 × N]
    Simplex-MLP (채널 간 의존성)
    → Dropout
    → Transpose + Linear + Transpose (시간 의존성)
    ↓
[FTM 블록 × N]
    Linear + Dropout + Transpose
    ↓
Linear (예측 헤드)
    ↓ Inverse Frequency Transformation
출력 Ŷ (N×τ)
```

**주요 설계 특징**:
- **DCT(이산 코사인 변환)** 사용: 실수 연산만으로 주파수 변환 가능
- **잔차 연결**: 각 블록에 skip connection 적용
- **인스턴스 정규화**: 입력 정규화로 분포 이동(distribution shift) 대응
- **레이어 수**: 3, 은닉 차원: 128

> 🔍 **DCT(Discrete Cosine Transform, 이산 코사인 변환)**: 신호를 주파수 성분으로 분해하는 변환. FFT와 달리 실수만 다루어 계산이 단순함.

> 🔍 **인스턴스 정규화(Instance Normalization)**: 각 샘플 단위로 평균과 분산을 정규화하여 배치 간 분포 차이를 줄임.

---

#### 🔴 성능 향상

**Table III** (p.7) 기준 주요 결과:

| 데이터셋 | FSMLP MSE | 2위 모델 | 2위 MSE | 개선율 |
|----------|-----------|----------|---------|--------|
| ETTm1 Avg | **0.365** | PatchTST | 0.387 | 5.7%↑ |
| ETTm2 Avg | **0.265** | PatchTST | 0.281 | 5.7%↑ |
| ETTh1 Avg | **0.416** | FEDformer | 0.441 | 5.7%↑ |
| ETTh2 Avg | **0.350** | FITS | 0.377 | 7.2%↑ |
| ECL Avg | **0.159** | Crossformer | 0.182 | 12.6%↑ |
| Traffic Avg | **0.415** | iTransformer | 0.428 | 3.0%↑ |
| Weather Avg | **0.237** | FITS | 0.251 | 5.6%↑ |

**Table V** (p.9) 추론 시간 (256 샘플 기준):
- ETTh1: **0.018s** (최고속)
- Traffic: **0.106s** (FreTS 0.105s와 유사)

---

#### 🔴 한계

1. **고정된 look-back 윈도우**: 모든 데이터셋에 L=96 고정 (유동적 설정 미탐색)
2. **단변량 예측 미평가**: 다변량 채널 믹스에 특화, 단변량 성능 언급 없음
3. **심플렉스 제약의 표현력 한계**: 가중치 합이 1로 고정되어 음의 상관관계 포착 불가 (저자 미언급이나 논리적 한계)
4. **Weather 데이터셋 제한적 우위**: FITS(MSE 0.251)과 차이 미미 (0.237)
5. **이론적 증명의 범위**: 선형 회귀 맥락의 Rademacher 복잡도 분석으로, 비선형 활성화 함수 포함 시 엄밀한 bound 미제시

---

## 3. 각 주장의 위치 표시

| 주장 | 위치 |
|------|------|
| 극단값이 MLP 과적합 악화 | p.2, Table I |
| Rademacher 복잡도 분석 | p.2 (개요), p.5-6 (Theorem 1 증명) |
| 표준 n-심플렉스 정의 | p.3, 수식 (1)-(3) |
| Simplex-MLP 수식 | p.4, 수식 (5) |
| $f_{\text{sim}}$ 상세 구현 | p.4 |
| SCWM/FTM 블록 | p.5 |
| 손실 함수 | p.5, 수식 (6) |
| 전체 아키텍처 | Fig. 2 (p.4) |
| 과적합 비교 시각화 | Fig. 1 (p.1) |
| 전체 성능 결과 | Table III (p.7) |
| Ablation 연구 | Table IV (p.9) |
| 추론 효율성 | Table V (p.9) |
| 훈련 효율성 | Fig. 3 (p.9) |
| Simplex-MLP 구현 비교 | Table VII (p.10) |
| 타 모델 적용 효과 | Table VI (p.10) |
| 다른 정규화 비교 | Table VIII (p.11) |
| 확장성 분석 | Table IX-X, XIII (p.11-12) |
| 계산 복잡도 | Table XIV (p.11) |

---

## 4. 저자 보고 vs. 나의 해석 분리

### 연구 주제

| 구분 | 내용 |
|------|------|
| **저자 직접 보고** | "채널 간 의존성을 심플렉스 이론 기반 MLP로 모델링하는 주파수 도메인 TSF 프레임워크" (Abstract) |
| **나의 해석** | 본 연구는 단순한 성능 개선 논문을 넘어, MLP 과적합 문제에 대한 이론적 원인 분석과 기하학적 제약 기반 해결책을 제시하는 점에서 이론-실험 통합 연구로 평가할 수 있음 |

### 방법

| 구분 | 내용 |
|------|------|
| **저자 직접 보고** | "가중치를 표준 n-심플렉스 내로 제한함으로써 Rademacher 복잡도 상한을 기존 MLP 대비 $B$배 감소" (p.2) |
| **나의 해석** | $f_{\text{sim}}$의 정규화 단계는 본질적으로 소프트맥스(Softmax)와 유사한 구조이나, 비선형 변환($\log$, $\mid \cdot \mid$, $(\cdot)^2$)을 먼저 적용하여 가중치 분포를 다르게 제어한다는 점이 차별적임. 이는 어텐션 메커니즘의 가중합과 개념적으로 연결될 수 있음 |

### 결과

| 구분 | 내용 |
|------|------|
| **저자 직접 보고** | "7개 벤치마크에서 유의미한 성능 향상" (Abstract); Traffic 평균 MSE 0.415, MAE 0.272 (Table III) |
| **나의 해석** | ① Weather 데이터셋에서 FITS(MSE 0.251) 대비 FSMLP(MSE 0.237)의 차이가 상대적으로 작아, 채널 수가 적은(21채널) 단순 데이터셋에서는 심플렉스 제약의 이점이 제한적일 수 있음. ② Traffic(862채널)에서 큰 우위를 보이는 패턴은 고차원 채널 의존성에서 심플렉스 제약이 특히 효과적임을 시사함 |

---

## 5. 통계적 취약점 및 비교 불가능 수치

| ⚠️ 유형 | 항목 | 문제점 |
|---------|------|--------|
| **비교 불가** | Table III의 일부 베이스라인 결과 | 저자들이 TimesNet 저장소 하이퍼파라미터로 재현했다고 명시하나, 원 논문 결과와 상이할 가능성 존재 |
| **통계적 취약** | 10개 랜덤 시드 평균 보고 | 표준편차/신뢰구간 미보고. 성능 차이의 통계적 유의성(p-value 등) 불명확 |
| **통계적 취약** | Table VI의 TSMixer 개선율 143.6% (ETTh2) | 원본 모델(MSE 2.025)이 극단적으로 과적합되었을 때의 상대적 개선으로, 실질적 의미 과장 가능성 |
| **비교 불가** | Fig. 3 메모리 비교 | 배치 크기, 정밀도(FP16/FP32) 등 동일 조건 여부 불명확 |
| **통계적 취약** | Look-back 윈도우 L=96 고정 | L=336, 512 등 다른 설정에서의 성능 불명확 (Table X에서 일부 탐색하나 제한적) |
| **비교 불가** | iTransformer(2024)와의 비교 | 논문 버전(arXiv preprint)이 최종 발표 버전과 다를 수 있음 |
| **통계적 취약** | Theorem 1 적용 범위 | 선형 회귀 기반 증명으로, 실제 모델의 비선형 활성화 함수 적용 시 이론적 보장 약화 |

---

## 6. 논문이 답하지 않는 질문

| # | 미해결 질문 |
|---|------------|
| 1 | **음의 채널 상관관계**: 심플렉스 제약($w_i \geq 0$)으로 인해 채널 간 음의 상관관계를 포착할 수 없는데, 이것이 성능에 미치는 영향은? |
| 2 | **최적 레이어 수**: 왜 3개 레이어가 최적인가? 데이터셋별 최적 레이어 수가 다를 수 있는가? |
| 3 | **단변량 예측 성능**: 채널 수 N=1인 단변량 설정에서 FSMLP의 성능은? |
| 4 | **실시간 적용**: 온라인 학습(online learning) 환경에서의 적용 가능성은? |
| 5 | **심플렉스 제약 완화**: 표준 심플렉스 외에 일반화된 심플렉스($\sum w_i = c$, $c \neq 1$)를 사용했을 때 성능 변화는? |
| 6 | **비정상 시계열(non-stationary)**: 인스턴스 정규화를 적용하더라도, 추세 성분이 강한 시계열에서의 성능은? |
| 7 | **이상치 민감도**: $3\sigma$ 초과 극단값을 사전 제거했을 때 기존 MLP와 FSMLP의 성능 차이가 좁혀지는가? |
| 8 | **교차 데이터셋 전이**: 한 도메인에서 학습한 FSMLP를 다른 도메인에 전이할 때의 성능은? |
| 9 | **심플렉스 수렴 속도**: 심플렉스 제약 하에서 최적화(Adam)의 수렴 속도가 기존 MLP보다 느린가 빠른가? |
| 10 | **해석 가능성**: 심플렉스로 제약된 가중치 행렬이 채널 간 어떤 관계를 학습하는지 시각적 해석 제공 미흡 |

---

## 7. 가장 중요한 그림 5개 해석

### 📊 Figure 1 (p.1): 과적합 비교

```
훈련 손실 vs 검증 손실 (ETTh1, L=96, τ=96)
```

**저자 설명**: TSMixer, TimesNet, Autoformer는 훈련 손실이 빠르게 감소하지만 검증 손실이 높게 유지됨 (과적합). FSMLP는 낮은 검증 손실을 유지.

**나의 해석**: 
- 두 그래프(왼쪽/오른쪽)의 Y축 범위가 다르므로 직접 비교 시 주의 필요
- FSMLP의 검증 손실도 초기에 감소 후 수렴하는 패턴으로, 완전한 과적합 면역은 아님
- 약 10-15 에폭 이후 다른 모델들의 검증 손실이 반등하는 반면, FSMLP는 안정적임
- **통계적 취약점**: 단일 실험 결과로, 시드에 따른 분산 미제시

---

### 📊 Figure 2 (p.4): FSMLP 전체 아키텍처

**저자 설명**: (a) 전체 구조, (b) 기존 MLP, (c) Simplex-MLP 비교

**나의 해석**:
- SCWM이 FTM보다 먼저 적용되어 채널 의존성을 먼저 추출하는 설계 선택이 흥미로움. 반대 순서(시간→채널)가 더 나을 수도 있으나 ablation 미제공
- (b)와 (c)의 비교에서 노란색으로 표시된 "Simplex Constrained Weight"가 핵심 차이임
- DCT를 통한 주파수 변환 후 역변환까지의 파이프라인이 엔드-투-엔드로 학습되는 구조

---

### 📊 Figure 3 (p.9): 효율성 비교 (ETTh1, Weather)

```
X축: 훈련 시간(s/epoch), Y축: MSE, 원 크기: 메모리 사용량
```

**저자 설명**: FSMLP는 ETTh1에서 2.08s/epoch, 608MB로 가장 효율적인 Pareto 최적점 달성.

**나의 해석**:
- ETTh1(a)에서 FSMLP(608MB)가 iTransformer(790MB)보다 작은 메모리로 더 나은 MSE 달성
- Weather(b)에서 iTransformer(6.89s/epoch, 1162MB)보다 FSMLP(5.44s/epoch, 674MB)가 우월
- **주의**: 메모리는 GPU 메모리로 추정되나 측정 조건(배치 크기 등) 미명시 → 직접 비교 주의
- TimesNet은 메모리(13,116MB)와 시간(7.94s) 모두 가장 비효율적

---

### 📊 Table III (p.7): 전체 예측 성능 결과

**저자 설명**: FSMLP가 7개 데이터셋, 4개 예측 길이에서 최고 또는 근접 최고 성능 달성.

**나의 해석**:
- **패턴 분석**: 채널 수가 많을수록(Traffic: 862, ECL: 321) FSMLP의 우위가 뚜렷함
  - Traffic: FSMLP 0.415 vs PatchTST 0.481 (13.7%↑)
  - ETTh1(7채널): FSMLP 0.416 vs FEDformer 0.441 (5.7%↑)
- **주목할 이상**: ETTm1 20% 훈련(Table IX)에서 FSMLP MSE가 40% 훈련보다 낮은 현상(0.481 vs 0.495) → 소량 데이터 불안정성 존재
- Weather에서 FITS(0.251)와 FSMLP(0.237) 차이가 상대적으로 작음 → 채널 수 21개의 소규모 채널에서는 심플렉스 이점 제한적

---

### 📊 Table VI (p.10): 타 모델에 Simplex-MLP 적용 효과

**저자 설명**: TSMixer와 Autoformer에 Simplex-MLP 적용 시 전반적 성능 향상.

**나의 해석**:
- **ETTh2 TSMixer 개선(143.6%)**: 원본 TSMixer MSE 2.025 → 0.589. 이는 TSMixer가 해당 데이터셋에서 극심하게 과적합되었음을 의미하며, 단순히 과적합을 완화한 것만으로도 대폭 향상됨
- **Autoformer 개선 제한**: Traffic에서 37%↑ 개선되었으나 ETTm2에서는 0.7%만 개선 → 데이터셋 특성에 따라 심플렉스 효과가 상이
- 이 결과는 Simplex-MLP가 **플러그인 모듈**로서 다양한 아키텍처에 적용 가능한 범용성을 입증하는 중요한 실험적 증거임

---

## 8. 결론 분석

### 8-1. 연구자 제시 시사점 및 후속 연구 계획

**저자가 제시한 시사점** (p.12, Section VII):
1. 표준 n-심플렉스 제약이 시계열 예측에서 효과적인 정규화 역할
2. 주파수 도메인 변환이 채널 간 주기적 의존성 포착에 효과적
3. 에너지 소비, 웹 데이터 분석, 날씨 예측 등 다양한 도메인 적용 가능성

**후속 연구 계획**: 논문 내 명시적 후속 연구 계획 없음 (저자 미언급)

---

### 🔵 모델의 일반화 성능 향상 가능성 (8-1 중점)

**현재 일반화 성능의 이론적 보장**:

$$\mathcal{R}_S(\mathcal{H}_\Delta) \leq \frac{1}{m} \sqrt{\sum_{i=1}^{m} \|x^{(i)}\|_2^2}$$

이 상한은 $B$ 항을 제거하여 데이터 자체의 복잡도에만 의존하므로, 데이터셋 특성에 따른 일반화 성능이 더 안정적임.

**일반화 성능 향상 가능 방향**:

| 방향 | 설명 | 기대 효과 |
|------|------|-----------|
| **적응형 심플렉스 스케일링** | $\sum w_i = c$ ($c$를 학습 가능한 파라미터로 설정) | 채널 수와 데이터 특성에 따라 최적 제약 강도 자동 조절 |
| **계층별 차별화 심플렉스** | 각 레이어마다 다른 심플렉스 제약 적용 | 저수준-고수준 특징 추출에 맞는 제약 적용 |
| **도메인 적응형 정규화** | 훈련 데이터의 극단값 비율에 따라 심플렉스 강도 동적 조절 | Traffic(극단값 1.59%)에는 강한 제약, ETTh1(0.35%)에는 약한 제약 |
| **앙상블 심플렉스** | 복수의 심플렉스 제약을 앙상블하여 다양한 채널 관계 포착 | 음의 상관관계 표현 가능성 확보 |
| **사전 학습(Pre-training)** | 대규모 시계열 데이터로 사전 학습 후 파인튜닝 | 데이터 부족 상황에서 일반화 성능 향상 |

**Table IX (p.11) 부분 샘플 훈련 결과 분석**:
- FSMLP는 20% 데이터로도 ETTh1 MSE 0.412 달성 (FreTS 0.690, TSMixer 0.890 대비 압도적)
- 데이터 효율성이 높아 소량 데이터 환경에서의 일반화 성능이 우수함
- 단, ETTm1 20%→40%에서 MSE가 오히려 증가(0.481→0.495)하는 불안정성 존재 → **⚠️ 통계적 취약점**

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **정확도 고지**: 아래 비교는 제 학습 데이터 범위(~2024년 초) 내의 논문에 기반합니다. 2024년 중반 이후 발표된 논문은 포함되지 않을 수 있습니다.

#### 시계열 예측 주요 연구 타임라인

| 연도 | 모델 | 핵심 기여 | FSMLP와의 관계 |
|------|------|-----------|----------------|
| 2021 | **Autoformer** [1] | 자기상관(Auto-correlation) 기반 분해 | FSMLP가 과적합 문제 개선 |
| 2022 | **FEDformer** [21] | 주파수 향상 분해 트랜스포머 | FSMLP의 주파수 도메인 접근과 유사한 동기 |
| 2022 | **DLinear** [17] | 단일 선형 레이어로 SOTA 달성 | 채널 독립 한계 존재 |
| 2022 | **PatchTST** [19] | 패치 기반 어텐션으로 시간 의존성 포착 | 채널 간 의존성 모델링 부재 |
| 2023 | **iTransformer** [5] | 역전된 트랜스포머로 채널 의존성 포착 | FSMLP가 더 낮은 복잡도로 유사 성능 달성 |
| 2023 | **TSMixer** [24] | 전체 MLP 아키텍처로 채널 의존성 | 심플렉스 적용 시 추가 개선 가능 |
| 2023 | **FreTS** [28] | 복소수 MLP로 주파수 도메인 모델링 | 채널 의존성 명시적 모델링 부재 |
| 2023 | **FITS** [18] | 10K 파라미터로 효율적 TSF | 채널 독립, FSMLP에 성능 열위 |
| 2024 | **ModernTCN** [12] | 현대적 순수 컨볼루션 구조 | FSMLP 비교 대상 미포함 (잠재적 경쟁자) |
| 2024 | **FilterNet** [14] | 주파수 필터 활용 | FSMLP와 주파수 도메인 접근 공유 |

#### 핵심 연구 트렌드와 FSMLP의 위치

```
2021-2022: 트랜스포머 우위 시대 (Informer, Autoformer, FEDformer)
     ↓
2022-2023: 단순 MLP의 반격 (DLinear, FITS: "Are Transformers Effective?")
     ↓
2023: 채널 혼합 vs 독립 논쟁 (iTransformer, TSMixer vs PatchTST)
     ↓
2023-2024: 주파수 도메인 활용 증가 (FreTS, FilterNet, FSMLP)
     ↓
FSMLP: 채널 혼합 + 주파수 도메인 + 이론적 정규화의 통합
```

#### FSMLP가 앞으로의 연구에 미치는 영향

**긍정적 영향**:
1. **이론-실험 통합 연구 패러다임**: Rademacher 복잡도로 과적합을 이론적으로 분석한 접근은 후속 연구의 방법론적 기준을 높임
2. **플러그인 모듈 아이디어**: Simplex-MLP를 독립 모듈로 제공하여 기존 모델 개선 가능성 제시
3. **기하학적 제약 기반 정규화**: 확률 심플렉스 제약을 딥러닝에 활용하는 새로운 방향 제시

**향후 연구 시 고려할 점**:

| 고려사항 | 세부 내용 |
|----------|-----------|
| **심플렉스 제약 일반화** | 비음수 제약($w_i \geq 0$)으로 인한 음의 상관관계 표현 한계 해결 필요. 부호 없는 심플렉스(signed simplex) 또는 유계 다면체(bounded polytope) 탐색 |
| **Foundation Model과의 결합** | GPT4TS, MOIRAI 등 시계열 파운데이션 모델에 심플렉스 제약 적용 시 파인튜닝 안정성 향상 가능성 |
| **불균형 데이터 처리** | 극단값이 많은 금융 시계열(FSMLP 미평가)에서의 적용성 검증 필요 |
| **해석 가능성 연구** | 심플렉스 가중치 행렬의 시각화를 통한 채널 간 관계 해석 (그래프 네트워크와 연계 가능) |
| **다중 스케일 심플렉스** | 서로 다른 시간 스케일(일/주/월)에 대해 별도의 심플렉스 제약 적용 |
| **비교 공정성 강화** | 동일 look-back 윈도우(L=96) 고정 설정의 편향 가능성 → 다양한 L에서의 실험 필요 |

#### 추가 후속 연구 방향 (제안)

**① 심플렉스-어텐션 하이브리드**:
$$\text{SimplexAttention}(Q, K, V) = f_{\text{sim}}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

어텐션 가중치에 심플렉스 제약을 적용하여 소프트맥스 대체 가능성 탐색

**② 적응형 심플렉스 반경**:

```math
\Delta^n_c = \left\{ w \in \mathbb{R}^n \;\middle|\; w_i \geq 0, \sum_{i=1}^n w_i = c \right\}
```

$c$를 학습 가능한 파라미터로 설정하여 데이터 특성에 맞게 제약 강도 조절

**③ 그래프 기반 심플렉스 채널 모델링**:
- 채널 간 그래프 구조를 사전 정보로 활용하여 심플렉스 내 가중치 초기화
- 물리적으로 연결된 센서(Traffic, ECL)에서 특히 효과적일 것으로 예상

**④ 다중 해상도 심플렉스 혼합**:
- 저주파(추세)와 고주파(계절성) 성분에 서로 다른 심플렉스 제약 적용
- FEDformer의 분해 접근법과 결합

---

## 참고자료 (논문 내 인용 기준)

1. Li, Z. et al., "FSMLP: Modelling Channel Dependencies With Simplex Theory Based Multi-Layer Perceptions In Frequency Domain," arXiv:2412.01654v3, 2026. *(원 논문)*
2. Bartlett, P. L. & Mendelson, S., "Rademacher and Gaussian Complexities: Risk Bounds and Structural Results," *J. Mach. Learn. Res.*, vol. 3, pp. 463–482, 2003. *[25]*
3. Eaves, B. C., *Standard Simplex S and Matrix Operations*, Springer Berlin Heidelberg, 1984. *[26]*
4. Liu, Y. et al., "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting," arXiv:2310.06625, 2023. *[5]*
5. Chen, S.-A. et al., "TSMixer: An All-MLP Architecture for Time Series Forecasting," 2023. *[24]*
6. Yi, K. et al., "Frequency-Domain MLPs Are More Effective Learners in Time Series Forecasting," 2023. *[28]*
7. Wu, H. et al., "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting," NeurIPS, 2021. *[1]*
8. Nie, Y. et al., "A Time Series Is Worth 64 Words: Long-Term Forecasting with Transformers," arXiv:2211.14730, 2022. *[19]*
9. Zeng, A. et al., "Are Transformers Effective for Time Series Forecasting?", 2022. *[17]*
10. Xu, Z. et al., "FITS: Modeling Time Series with 10K Parameters," arXiv:2307.03756, 2023. *[18]*
