# LSTM 오토인코더를 이용한 이상 탐지의 임계치 결정 방법

> **참고 자료 (Primary Source)**
> - 전승현, 박채린, 이규원, 김성종, 구본근, "LSTM 오토인코더를 이용한 이상 탐지의 임계치 결정 방법", *Journal of KIIT*, Vol. 21, No. 4, pp. 21-30, Apr. 30, 2023. DOI: http://dx.doi.org/10.14801/jkiit.2023.21.4.21

> ⚠️ **정확도 고지**: 본 분석은 제공된 PDF 원문에만 근거합니다. 8-2절의 최신 연구 비교는 제 학습 데이터(~2024년 초) 기반이며, 직접 논문을 검색·열람한 결과가 아니므로 개별 수치 인용 시 원문 확인을 권장합니다.

---

## 1. Executive Summary (10문장 이내)

1. 본 논문은 기기 고장의 예지 보전(Predictive Maintenance)을 위해 LSTM 오토인코더(AE) 기반 이상 탐지 시스템에서 **임계치(Threshold)를 체계적으로 결정하는 방법**을 제안한다.
2. 정상 데이터만으로 학습된 LSTM AE는 이상 데이터에 대해 정상 데이터보다 **복원 오차(Reconstruction Error)가 크게 나타나는 특성**을 핵심 전제로 활용한다.
3. 기존 연구들은 임계치를 반복 실험이나 정밀도-재현율(PR) 그래프의 교점에서 경험적으로 결정하였으나, **자동화·체계화된 방법이 부재**하였다.
4. 제안 방법은 **정상 데이터의 복원 오차 최댓값**($ndata_{max}$)과 **이상 데이터의 복원 오차 최솟값**($adata_{min}$) 사이 구간을 분할 비율 $\alpha$로 나누어 임계치를 산출한다.
5. 실험은 Kaggle에 공개된 펌프 센서 데이터(약 22만 행, 50개 센서)를 대상으로 수행되었다.
6. 모델은 인코더·디코더 각 3층의 LSTM 계층으로 구성되며, 잠재 벡터 크기는 20, 손실함수는 MSE를 사용한다.
7. $\alpha = 0.25$일 때 임계치 6.385에서 최적 성능(정확도 99.35%, 재현율 98.26%)이 도출되었다.
8. 이는 정밀도-재현율 그래프(PRC) 방법 대비 **정확도 약 5%, 재현율 약 25% 향상**된 결과이다 (Table 3 기준).
9. 제안 방법의 핵심 이점은 임계치 결정을 위한 **별도 데이터셋(임계치 결정용 데이터)을 활용**함으로써 과적합 없이 일반화 가능한 기준값을 도출한다는 점이다.
10. 향후 연구로 승강기 운행 데이터 기반 모델 고도화 및 **임계치 자동 결정 방법** 개발이 제시되었다.

---

### 1-1. 연구의 목적과 필요성

| 구분 | 내용 |
|------|------|
| **실용적 목적** | 쇼핑몰·빌딩·역사 등 다중이용 건축물 내 승강기, 펌프 등 기기 고장으로 인한 직·간접적 손실 최소화 |
| **기술적 목적** | LSTM AE 기반 이상 탐지에서 성능을 좌우하는 임계치를 체계적으로 결정하는 방법론 제시 |
| **문제 의식** | 기존 연구([10], [11])는 임계치를 반복 실험 또는 PR 그래프의 교점으로 설정했으나, 이는 자동화·재현성이 낮고 도메인 의존적임 |
| **비지도 학습 선택 이유** | 승강기·펌프 등 기기의 이상(고장) 데이터는 정상 데이터에 비해 희귀하여, 레이블이 필요한 지도학습보다 **정상 패턴만 학습하는 비지도 LSTM AE**가 더 현실적 |

*(p. 22, 서론 및 기존 연구 참조)*

---

## 2. 핵심 주장과 근거 (표)

| # | 핵심 주장 | 근거/방법 | 출처 위치 |
|---|-----------|-----------|-----------|
| 1 | 정상 데이터로 학습된 LSTM AE는 이상 데이터에 대해 복원 오차가 더 크다 | LSTM AE의 구조적 특성: 학습 분포 외 입력에 대한 복원 실패 | p. 22, §2.1 |
| 2 | 임계치는 정상 최대 복원 오차와 이상 최소 복원 오차 사이에 존재해야 한다 | 두 값 사이 구간만이 유효한 분리 경계임 | p. 23, §3, Fig. 1 |
| 3 | 분할 비율 $\alpha$를 이용한 수식으로 임계치를 결정할 수 있다 | 식 (1): $th_\alpha = \|ndata_{max} - adata_{min}\| \times \alpha + \min(ndata_{max}, adata_{min})$ | p. 24, 식(1) |
| 4 | $\alpha = 0.25$가 정확도와 재현율의 균형에서 최적 | 임계치 결정용 데이터 실험에서 ACC=99.47%, TPR=99.74% | p. 27, Table 2 |
| 5 | 제안 방법이 PRC 방법보다 테스트 성능이 우수 | 테스트 데이터: 제안법 ACC=99.35%, PRC ACC=94.25% | p. 28, Table 3 |
| 6 | $\alpha$ 값이 작을수록 재현율(TPR) 증가, 거짓 양성(FP) 증가의 트레이드오프 존재 | $\alpha$별 실험 결과 비교 | p. 27, Table 2 |

---

### 2-1. 해결 문제 / 제안 방법 / 모델 구조 / 성능 및 한계 상세 설명

#### ① 해결하고자 하는 문제

- LSTM AE 기반 이상 탐지에서 **임계치 결정이 성능을 좌우하는 핵심 요소**임에도 불구하고, 기존 연구는 이를 체계화하지 못함.
  - [10] 김성종 외: 반복 실험으로 최적 임계치 수동 탐색
  - [11] 이정화 외: PR 그래프의 교점 사용 → 재현율 저하 문제 존재
- 자동화 가능하고, 응용 도메인의 비용 구조(거짓 양성 vs 거짓 음성 비용)를 반영할 수 있는 방법 부재

*(p. 22–23, §1, §2.2, §2.3)*

---

#### ② 제안하는 방법 (수식 포함)

**핵심 아이디어**: 정상 데이터의 복원 오차 최댓값($ndata_{max}$)과 이상 데이터의 복원 오차 최솟값($adata_{min}$) 사이 구간을 비율 $\alpha$로 분할.

$$
th_{\alpha} = |ndata_{\max} - adata_{\min}| \times \alpha + \min(ndata_{\max},\ adata_{\min})
$$

- $th_{\alpha}$: 결정된 임계치
- $ndata_{\max}$: 임계치 결정용 정상 데이터의 복원 오차 최댓값 (실험값: **15.0313**)
- $adata_{\min}$: 임계치 결정용 이상 데이터의 복원 오차 최솟값 (실험값: **3.5025**)
- $\alpha \in (0, 1)$: 분할 비율 (응용 도메인의 비용 구조에 따라 결정)

**복원 오차 계산 (MSE)**:

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (x_i - \hat{x}_i)^2
$$

- $x_i$: 원본 입력 데이터
- $\hat{x}_i$: LSTM AE가 복원한 출력 데이터

**이상 판정 규칙**:

$$
\text{결정} = \begin{cases} \text{이상} & \text{if } \text{MSE}(x, \hat{x}) > th_{\alpha} \\ \text{정상} & \text{if } \text{MSE}(x, \hat{x}) \leq th_{\alpha} \end{cases}
$$

**$\alpha$별 임계치 계산 예시** ($(15.0313 - 3.5025) \times \alpha + 3.5025$):

| $\alpha$ | 임계치 ($th_\alpha$) |
|----------|---------------------|
| 0.5 | 9.267 |
| 0.25 | 6.385 |
| 0.125 | 4.944 |
| 0.0625 | 4.223 |

*(p. 23–24, §3, 식(1), Fig. 2)*

---

#### ③ 모델 구조 (Fig. 3, p. 25)

```
입력 (16, 50)
    ↓
[Encoder]
  LSTM(256) → LSTM(128) → LSTM(64)
    ↓
Latent Vector z (dim=20)
    ↓
[Decoder]
  LSTM(64) → LSTM(128) → LSTM(256)
    ↓
출력 (복원된 데이터, 16, 50)
```

| 항목 | 세부 내용 |
|------|-----------|
| 입력 시퀀스 길이 | 16 (연속된 16개 타임스텝) |
| 센서 수 | 50개 |
| 잠재 벡터 크기 | 20 |
| 옵티마이저 | Adam |
| 손실함수 | MSE (복원 오차) |
| 배치 크기 | 32 |
| 에포크 | 1 |
| 하드웨어 | Intel Xeon, NVIDIA RTX A5000 |
| ML API | Keras 2.10.0 |

*(p. 25, Table 1, Fig. 3)*

---

#### ④ 성능 향상 및 한계

**성능 향상** (Table 3, p. 28 — 테스트 데이터 기준):

| 방법 | ACC | TPR (재현율) |
|------|-----|-------------|
| 제안 방법 ($\alpha=0.25$, $th=6.385$) | **99.35%** | **98.26%** |
| 평균 중간값 (MoA, $th=23.837$) | 84.15% | 38.64% |
| 정밀도-재현율 그래프 (PRC, $th=8.318$) | 94.25% | 78.26% |

- PRC 대비: 정확도 **+5.1%p**, 재현율 **+25.0%p** 향상

**한계**:
1. **단일 데이터셋**: Kaggle 펌프 데이터 1종에만 검증 → 일반화 근거 부족
2. **에포크=1**: 학습이 극히 제한적 → 모델 최적화 여부 불분명
3. **$\alpha$ 결정 기준 미명시**: 어떤 $\alpha$를 선택할지는 여전히 도메인 전문가 판단에 의존
4. **이상 데이터 의존성**: 임계치 결정 시 이상 데이터($adata_{min}$)가 필요 → 완전 비지도 설정에서 적용 어려움

---

## 3. 각 주장에 페이지/Figure/Table 번호 표시

| 주장 | 근거 위치 |
|------|-----------|
| 정상 데이터 복원 오차 < 이상 데이터 복원 오차 | p. 22, §2.1 |
| 임계치는 두 극값 사이에 존재 | p. 23, §3; **Fig. 1** |
| 임계치 결정 수식 제안 | p. 24, **식(1)** |
| $\alpha$에 따른 임계치 변화 시각화 | p. 24, **Fig. 2** |
| LSTM AE 구조 (인코더-잠재벡터-디코더) | p. 25, **Fig. 3** |
| 실험 환경 및 데이터 분할 | p. 25, **Table 1** |
| $\alpha$별 분류 결과 시각화 | p. 26, **Fig. 4–7** |
| 정상/이상 평균 중간값 임계치 결과 | p. 27, **Fig. 8** |
| PRC 임계치 결과 | p. 27, **Fig. 9** |
| 임계치 결정용 데이터 실험 결과 | p. 27, **Table 2** |
| 테스트 데이터 최종 성능 비교 | p. 28, **Table 3** |

---

## 4. 저자 보고 결과 vs. 검토자 해석 분리

### 4-1. 저자가 직접 보고한 결과

- **연구 주제**: LSTM AE 기반 이상 탐지에서 정상 데이터의 최대 복원 오차와 이상 데이터의 최소 복원 오차 사이의 비율을 이용하는 임계치 결정 방법 제안 *(p. 21, Abstract)*
- **방법**: 식 $(1)$: $th_{\alpha} = |ndata_{\max} - adata_{\min}| \times \alpha + \min(ndata_{\max}, adata_{\min})$ *(p. 24)*
- **결과**: $\alpha=0.25$ 기준, 테스트 데이터에서 ACC=99.35%, TPR=98.26%로 PRC 방법 대비 각각 5%, 25% 향상 *(p. 21, Abstract; p. 28, Table 3)*

### 4-2. 검토자(필자)의 해석

> ⚠️ 아래는 저자의 직접 진술이 아닌 분석적 해석입니다.

- **에포크=1 설정의 문제**: 저자는 에포크 수를 1로 설정하였는데(Table 1), 이는 모델이 충분히 학습되지 않았을 가능성을 시사한다. 높은 성능이 데이터의 특성(정상/이상 데이터 간 복원 오차 격차가 크다는 구조적 특성)에 기인할 수 있다.
- **이상 데이터 의존성의 한계**: 제안 방법이 $adata_{min}$ (이상 데이터 최소 복원 오차)을 필요로 한다는 점에서, 이상 데이터가 전혀 없는 완전 비지도 환경에서는 직접 적용이 어렵다. 이는 방법론의 범용성을 제한하는 주요 제약이다.
- **단일 데이터셋 한계**: 1개의 Kaggle 데이터셋만으로 검증하여 다른 도메인·센서 환경에서의 일반화 성능은 미확인 상태이다.
- **MoA(평균 중간값) 방법의 과도한 임계치**: $th_{MoA} = 23.837$로, 이상 데이터 복원 오차 분포 대부분을 커버하지 못해 TPR=38.64%라는 매우 낮은 재현율을 보인다. 이는 비교군으로서 기준선 역할만 한다고 볼 수 있다.

---

## 5. 통계적 취약점 및 비교 불가능한 수치

| 항목 | 취약점/비교 불가 이유 |
|------|----------------------|
| ⚠️ **단일 데이터셋** | Kaggle 펌프 데이터 1종만 사용. 통계적 유의성(p-value 등) 미제시 |
| ⚠️ **에포크=1** | 학습 1회만 진행 → 가중치 초기화에 따른 결과 변동성(분산) 미보고. 반복 실험 없음 |
| ⚠️ **교차 검증 미실시** | K-fold 등 교차 검증 없이 단일 분할(train/threshold/test)로만 평가 |
| ⚠️ **표준편차 미보고** | Table 2, 3의 ACC, TPR 값이 단일 실험 결과이며 오차 범위 미제시 |
| ⚠️ **MoA 비교의 불균형** | MoA 임계치(23.837)는 $ndata_{max}$(15.0313)보다도 높아, 실질적으로 의미 없는 비교군이 될 수 있음 |
| ⚠️ **$\alpha$ 결정 근거** | $\alpha=0.25$를 최적으로 선택한 근거가 임계치 결정용 데이터 성능뿐이며, 일반화 가능한 이론적 기준 없음 |
| ⚠️ **F1-Score 미보고** | 정밀도(Precision)가 보고되지 않아 ACC, TPR만으로는 성능의 전면적 비교 불가 |

---

## 6. 문서가 답하지 않는 질문

1. **다른 도메인에서의 성능**: 승강기, 네트워크, 제조업 등 다른 센서 환경에서도 동일한 $\alpha$ 범위가 유효한가?
2. **에포크 수에 따른 민감도**: 에포크를 늘리면 복원 오차 분포가 달라져 $ndata_{max}$, $adata_{min}$이 변하는가?
3. **이상 데이터가 전혀 없을 때**: $adata_{min}$을 대체할 수 있는 방법이 있는가?
4. **최적 $\alpha$ 결정 자동화 방법**: 응용별로 어떻게 $\alpha$를 자동으로 선택할 수 있는가?
5. **비정상 유형별 성능 차이**: 펌프 이상의 유형(점진적 열화 vs 급격한 고장)에 따라 복원 오차 패턴이 다를 수 있는데, 유형별 성능은 어떠한가?
6. **복원 오차 분포의 통계적 특성**: 정상/이상 데이터 복원 오차가 정규분포를 따르는지, 이상치에 민감한지 여부
7. **실시간 적용 가능성**: 스트리밍 데이터 환경에서 $ndata_{max}$, $adata_{min}$이 실시간으로 갱신되어야 하는가?
8. **모델 학습 수렴 여부**: 에포크=1에서 학습이 충분히 수렴했는가? 학습 손실 곡선이 미제시

---

## 7. 가장 중요한 그림 5개 해석

### Fig. 1 — 복원 오차의 최댓값, 최솟값 (p. 23)

- **내용**: LSTM AE에 정상(×) 및 이상(·) 데이터를 입력했을 때 생성되는 복원 오차의 분포를 개념적으로 표시한 산점도
- **해석**: 이상 데이터(·)의 복원 오차가 정상 데이터(×)보다 전반적으로 상단에 분포. 두 클래스 사이에 **겹치는 영역**이 존재하며, 이 영역이 임계치 결정의 핵심 문제 구간임을 시각화함. 이 그림은 실제 데이터가 아닌 방법론 설명을 위한 모식도임을 저자가 명시
- **방법론적 의의**: 임계치가 $ndata_{max}$와 $adata_{min}$ 사이에 위치해야 함을 직관적으로 보여줌

---

### Fig. 2 — $\alpha$ 값에 따른 임계치의 변화 (p. 24)

- **내용**: $\alpha=0.5$일 때와 $\alpha=0.25$일 때의 임계치($th_{0.5}$, $th_{0.25}$) 위치를 복원 오차 분포 상에 표시
- **해석**: $\alpha$가 작아질수록 임계치가 낮아져 더 많은 데이터를 이상으로 분류. **낮은 $\alpha$**: 거짓 음성(FN)↓, 거짓 양성(FP)↑ / **높은 $\alpha$**: FN↑, FP↓ 의 트레이드오프를 시각화
- **방법론적 의의**: $\alpha$가 도메인 비용 구조(고장 비용 vs 불필요한 점검 비용)에 따라 조정 가능한 유연한 매개변수임을 보여줌

---

### Fig. 3 — LSTM AE 구조 (p. 25)

- **내용**: 인코더(LSTM 256→128→64) + 잠재 벡터($z$, dim=20) + 디코더(LSTM 64→128→256)의 대칭적 구조
- **해석**: 표준적인 LSTM AE 아키텍처. 인코더가 시계열의 고차원 특징을 20차원으로 압축하고 디코더가 원본 차원(16×50)으로 복원. 압축 과정에서 이상 패턴은 손실이 커져 복원 오차가 증가하는 원리를 구조적으로 설명
- **한계**: 잠재 벡터 크기(20), LSTM 계층 수(3), 유닛 수(256/128/64)에 대한 하이퍼파라미터 탐색 과정이 미제시

---

### Fig. 4 vs Fig. 5 — $\alpha=0.5$와 $\alpha=0.25$일 때 분류 결과 (p. 26)

- **내용**: 테스트 데이터 포인트(정상=파랑, 이상=주황)와 결정된 임계치(붉은 선)를 복원 오차 축에 표시
- **해석**:
  - **Fig. 4** ($\alpha=0.5$, $th=9.267$): 붉은 선이 비교적 높게 설정되어 임계치 아래 이상 데이터(거짓 음성, FN=376)가 다수 존재
  - **Fig. 5** ($\alpha=0.25$, $th=6.385$): 붉은 선이 낮아져 대부분의 이상 데이터가 임계치 위로 분류됨(FN=19), 정상 데이터의 일부만 거짓 양성(FP=129) 발생
- **방법론적 의의**: $\alpha$ 조정이 실제로 FN/FP 균형에 미치는 영향을 직접 확인할 수 있는 핵심 근거 그림

---

### Fig. 9 — 정밀도-재현율 그래프의 교점을 임계치로 사용한 결과 (p. 27)

- **내용**: PRC 방법으로 결정된 $th=8.318$을 기준으로 분류한 결과 산점도
- **해석**: 제안 방법($th=6.385$, Fig. 5)에 비해 임계치가 높아 이상 데이터 중 임계치 아래에 위치하는 데이터(거짓 음성, FN=1558)가 훨씬 많음. 테스트 기준 TPR=78.26%로 제안 방법(98.26%)보다 약 20%p 낮음
- **비교 의의**: 기존 PRC 방법의 한계를 명확히 시각화하며, 제안 방법의 우위를 뒷받침하는 대조 그림으로 기능

---

## 8. 결론 요약 및 후속 연구 방향

### 8-1-A. 저자 제시 시사점 및 후속 연구 계획

**시사점** *(p. 28, §V)*:
- 정상 데이터의 최대 복원 오차와 이상 데이터의 최소 복원 오차 간 비율을 활용한 임계치 결정 방법이 예지 보전 분야에서 이상 탐지 성능 향상에 효과적임
- 거짓 음성(FN)과 거짓 양성(FP)의 트레이드오프를 $\alpha$로 조절하여 도메인별 비용 최적화가 가능함

**저자 제시 향후 연구**:
1. 승강기 인버터·센서 데이터를 위한 기계학습 모델 고도화
2. 시험 운행 데이터 기반 **임계치 자동 결정 방법** 개발

---

### 8-1-B. 모델의 일반화 성능 향상 가능성 (중점)

제안 방법의 일반화 성능을 높이기 위해 다음의 방향을 제안합니다.

**① 이상 데이터 독립적 임계치 결정**

현 방법의 가장 큰 일반화 제약은 $adata_{min}$을 위해 이상 데이터가 필요하다는 점입니다. 이를 극복하기 위해:

$$
th_{\text{stat}} = \mu_{normal} + k \cdot \sigma_{normal}, \quad k \in \{2, 3\}
$$

정상 데이터의 복원 오차 분포($\mu$, $\sigma$)만을 이용하는 통계적 방법(3-sigma rule)과의 앙상블이나 대체 가능성을 탐색해야 합니다.

**② 적응형 임계치(Adaptive Threshold)**

단일 고정 임계치 대신, 시간에 따라 변화하는 시스템 상태를 반영하는 슬라이딩 윈도우 기반 동적 임계치를 적용:

$$
th_{\alpha}(t) = |ndata_{\max}^{(t-w:t)} - adata_{\min}^{(t-w:t)}| \times \alpha + \min(\cdot)
$$

**③ 다중 데이터셋 교차 검증**

UCI Machine Learning Repository의 SKAB, SMD(Server Machine Dataset), SMAP, MSL 등 표준 이상 탐지 벤치마크 데이터셋에서 성능 검증 필요.

**④ $\alpha$ 자동 결정: 베이지안 최적화 활용**

$$
\alpha^* = \arg\max_{\alpha \in (0,1)} \left[ \lambda \cdot \text{TPR}(\alpha) + (1-\lambda) \cdot \text{ACC}(\alpha) \right]
$$

응용 도메인의 비용 가중치 $\lambda$를 입력으로 받아 $\alpha$를 자동으로 결정하는 베이지안 최적화 프레임워크 도입.

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **고지**: 아래 연구 목록은 제 학습 데이터에 기반한 일반적 지식이며, 각 논문의 세부 수치는 원문 확인이 필요합니다.

| 연구 | 방법 | 임계치 결정 방식 | 본 논문과의 차이 |
|------|------|-----------------|-----------------|
| Elsayed et al. (2020), "Network Anomaly Detection Using LSTM Based Autoencoder", Q2SWinet '20 [논문 내 ref.8] | LSTM AE + 네트워크 이상 탐지 | 반복 실험으로 수동 설정 | 본 논문의 비교 대상 방법론과 유사한 접근 |
| Nguyen et al. (2021), "Forecasting and Anomaly Detection using LSTM and LSTM AE", *Int. J. Information Management* [논문 내 ref.9] | LSTM AE + 공급망 이상 탐지 | 미명시 (경험적) | 도메인 응용 중심, 임계치 방법론 미체계화 |
| **Audibert et al. (2020)**, "USAD: UnSupervised Anomaly Detection on Multivariate Time Series", *KDD 2020* | 두 개의 AE 기반 적대적 훈련 | 복원 오차 통계 기반 | 이상 데이터 없이 임계치 설정 가능 — 본 논문보다 완전 비지도적 |
| **Su et al. (2019)**, "Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network", *KDD 2019* | LSTM VAE + 정규화된 이상 점수 | POT(Peak Over Threshold) 방법 | 극값 이론 기반 통계적 임계치 — 더 이론적 근거 강함 |
| **Tuli et al. (2022)**, "TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series", *VLDB 2022* | Transformer 기반 AE | 검증 데이터 기반 자동 임계치 | Attention 기반 특징 추출로 LSTM 한계 극복 시도 |
| **Zhao et al. (2020)**, "Multivariate Time-Series Anomaly Detection via Graph Attention Network" (*ICDM 2020*) | GNN + Attention | 동적 임계치 | 변수 간 상관관계를 그래프로 모델링 — LSTM AE의 독립 처리 한계 극복 |

**비교 분석 시사점**:

1. **임계치 결정의 패러다임 변화**: 2020년 이후 연구들은 POT(Peak Over Threshold), 통계적 분포 fitting, 검증셋 기반 자동 최적화 방향으로 발전하고 있음. 본 논문의 방법($\alpha$ 기반 비율법)은 직관적이나, 이론적 근거가 약한 편.
2. **완전 비지도 방향**: USAD, TranAD 등은 이상 데이터 없이 임계치를 결정하는 방향으로 발전 — 본 논문의 $adata_{min}$ 의존성을 극복.
3. **Transformer/GNN 부상**: 2021년 이후 순수 LSTM AE 기반 방법보다 Transformer 기반(TranAD, Anomaly Transformer 등) 또는 그래프 기반 방법이 성능 우위를 보이는 추세.
4. **본 논문의 기여**: 한국 산업 현장(펌프, 승강기) 적용에 특화된 실용적 방법을 제안했다는 점에서 의의가 있으나, 국제 표준 벤치마크 비교 및 최신 방법론과의 성능 비교가 필요.

---

**앞으로의 연구 시 고려할 점**:

1. **표준 벤치마크 활용**: SKAB, SMD, SMAP, MSL, SWaT 등 공개 이상 탐지 벤치마크에서 성능 검증
2. **이상 데이터 불필요한 임계치 설정**: POT(Extreme Value Theory 기반), 3-sigma, 또는 검증셋 기반 최적화 방법 탐색
3. **F1-Score, AUC-ROC, AUC-PR 병행 보고**: ACC와 TPR만으로는 클래스 불균형 상황에서 성능을 정확히 평가하기 어려움
4. **설명 가능한 AI(XAI) 통합**: 어떤 센서/시간대가 이상 탐지에 기여했는지 해석 가능성 확보
5. **온라인/스트리밍 환경 적용**: 실시간 임계치 갱신 메커니즘 연구
6. **멀티변량 이상 유형 분류**: 단순 탐지(detection)를 넘어 이상 원인 진단(diagnosis)으로 확장

---

**참고 자료 전체 목록**

*본문 직접 인용 참고문헌 (논문 내 References 기준)*:
1. Park et al. (2017), "Improved Forecasting Algorithm for Vessel Engine Failure", *Journal of KIIT*, Vol. 15, No. 11
2. Ki and Lee (2017), "A Prediction Scheme for Power Apparatus using Artificial Neural Networks", *Journal of Convergence Information*, Vol. 7, No. 6
3. Jeon et al. (2022), "LSTM Autoencoder Implementation for Anomaly Detection of Equipment", *Fall Conference of KIIT*
4. Lee et al. (2019), "Fault diagnosis of bearings using machine learning algorithm", *JKOSME*, Vol. 43, No. 6
5. Nguyen et al. (2018), "LSTM-based Anomaly Detection on Big Data for Smart Factory Monitoring", *Journal of Digital Contents Society*, Vol. 19, No. 4
6. Lee, Ko, Lee (2022), "Fault Classification Model Based on Deep Learning Using Vibration Data", *Journal of KINGPC*, Vol. 18, No. 2
7. Choi (2020), "Predictive Maintenance of the Robot Trouble Using the Machine Learning Method", *Journal of Semiconductor & Display Technology*, Vol. 19, No. 1
8. Elsayed et al. (2020), "Network Anomaly Detection Using LSTM Based Autoencoder", *Q2SWinet '20*
9. Nguyen et al. (2021), "Forecasting and Anomaly Detection approaches using LSTM and LSTM Autoencoder techniques", *Int. J. Information Management*, Vol. 57
10. Kim and Shon (2022), "LSTM Autoencoder-Based Insider Data Leak Detection", *Journal of Digital Contents Society*, Vol. 23, No. 6
11. Lee and Sohn (2021), "Escalator Anomaly Detection Using LSTM Autoencoder", *2021 Summer Conference of Korea Society of Computer Information*, Vol. 29, No. 2
12. Lee and Kim (2020), "Case Study on Fault Diagnosis of Radiator Using LSTM Autoencoder", *The Journal of KINGComputing*, Vol. 16, No. 6
13. pump_sensor_data, https://www.kaggle.com/datasets/nphantawee/pump-sensor-data

*8-2절 비교 분석 참고 (학습 데이터 기반, 원문 확인 필요)*:
- Audibert et al. (2020), "USAD: UnSupervised Anomaly Detection on Multivariate Time Series", *KDD 2020*
- Su et al. (2019), "Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network", *KDD 2019*
- Tuli et al. (2022), "TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series", *VLDB 2022*
- Zhao et al. (2020), "Multivariate Time-Series Anomaly Detection via Graph Attention Network", *ICDM 2020*
