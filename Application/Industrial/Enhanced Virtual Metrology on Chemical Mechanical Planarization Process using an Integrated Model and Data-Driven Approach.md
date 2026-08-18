# Enhanced Virtual Metrology on Chemical Mechanical Planarization Process using an Integrated Model and Data-Driven Approach

> **참고 문헌**: Di, Y., Jia, X., & Lee, J. (2017). Enhanced Virtual Metrology on Chemical Mechanical Planarization Process using an Integrated Model and Data-Driven Approach. *International Journal of Prognostics and Health Management*, ISSN 2153-2648, 2017 031.
>
> **주의**: 본 논문은 제공된 PDF 전문을 기반으로 분석하였으며, 2020년 이후 비교 연구는 공개된 학술 자료를 바탕으로 하되, 확인 불가능한 세부 수치는 명시적으로 표기합니다.

---

## 1. Executive Summary (10문장 이내)

반도체 제조의 핵심 공정인 CMP(Chemical Mechanical Planarization)에서 **웨이퍼 연마 성능 지표인 MRR(Material Removal Rate)**을 실시간으로 예측하는 가상 계측(Virtual Metrology, VM) 기술의 정확도를 높이는 것이 본 연구의 핵심 목표이다.  
기존의 물리 기반 모델은 미지의 환경 변수로 인해 계수 결정이 어렵고, 단독 데이터 기반 모델은 고차원 입력과 공정 동적 특성 처리에 한계가 있었다.  
저자들은 Preston 방정식 기반의 물리적 특징과 최근접 이웃(KNN)의 소모 부품 사용량 기반 동적 특징을 결합한 통합 피처 추출 전략을 제안하였다.  
Student's t-검정과 OOB(Out-of-Bag) 피처 중요도를 병용하여 고차원 피처 공간을 효과적으로 축소하였다.  
Persistent Model, KNN 회귀, SVR, 선형 회귀, Tree Bagging 등 5개의 회귀 모델을 Monte Carlo 교차검증(CV) 기반 가중 평균으로 앙상블하는 통합 모델을 구성하였다.  
해당 방법론은 PHM Society 2016 Data Challenge의 공개 데이터셋에 적용되어 24개 팀 중 **최저 MSE(7.07)**를 기록하며 1위를 달성하였다.  
통합 모델은 모든 조건(Cond1~3)에서 개별 모델 대비 편향(bias) 감소 효과를 보였다.  
학습 데이터 CV MSE와 테스트 MSE 간의 일관성은 모델의 일반화 성능을 시사한다.  
향후 센서 신호의 심층 패턴 추출, 온라인 모델 갱신, 물리-데이터 융합 모델로의 확장이 필요한 것으로 제시되었다.  
본 연구는 반도체 제조 공정의 지능형 공정 제어(APC) 고도화에 실질적 기여를 한다.

---

### 1-1. 연구의 목적과 필요성

| 구분 | 내용 |
|------|------|
| **산업적 맥락** | CMP는 반도체 다층 배선 공정에서 웨이퍼 표면 평탄화를 위한 필수 공정 |
| **기존 문제** | W2W(Wafer-to-Wafer) 제어는 매 웨이퍼 측정이 필요해 생산 사이클 시간 증가 |
| **지연 문제** | MRR은 공정 후 측정값이므로 APC(Advanced Process Control) 피드백 지연 발생 |
| **VM 필요성** | 공정 중 센서 데이터로 MRR을 사전 예측하는 가상 계측(VM)이 대안 |
| **기존 VM 한계** | 물리 모델: 미지 계수 / 데이터 모델: 고차원·동적 특성 처리 미흡 |
| **연구 목적** | 물리 메커니즘 + 데이터 기반 앙상블을 결합한 향상된 MRR 예측 알고리즘 개발 |

> 💡 **APC (Advanced Process Control)**: 반도체 제조에서 공정 파라미터를 실시간으로 조정해 불량률을 줄이는 고급 공정 제어 기술. 피드백/피드포워드 루프를 활용한다.

> 💡 **Virtual Metrology (VM, 가상 계측)**: 실제 물리적 측정 없이 공정 중 수집된 센서 데이터만으로 품질 지표를 예측하는 기술. 측정 지연과 비용을 동시에 줄일 수 있다.

---

## 2. 핵심 주장과 근거 표

| # | 핵심 주장 | 근거 | 위치 |
|---|-----------|------|------|
| 1 | 물리 기반 피처(프레스턴 방정식)와 데이터 기반 피처(KNN 이웃) 통합이 MRR 예측 정확도를 높인다 | 통합 모델 MSE 7.07로 개별 모델 모두 상회 | Table 5, p.6 |
| 2 | MRR은 시계열 특성을 강하게 가지므로 시간 지연(Time Lag) 피처가 유효하다 | Persistent Model이 KNN Regression보다 모든 조건에서 우수 | Table 4, p.7 |
| 3 | Student's t-검정과 OOB 중요도 병용이 효과적인 피처 선택 방법이다 | Figure 4에서 두 기준의 선택 결과 시각화 | Figure 4, p.6 |
| 4 | Monte Carlo CV 기반 가중 평균 앙상블이 편향 문제를 개선한다 | CV MSE 대비 테스트 MSE 일관성 확인 | Table 4, 5, p.7 |
| 5 | 제안 방법이 Deep Belief Networks보다 우수하다 | MSE: 제안법 7.07 vs. DBN 7.29 | Table 5, p.7 |
| 6 | 3개 레시피(Cond1~3) 분리 예측 전략이 유효하다 | 레시피별 MRR 분포 상이함 확인 | Figure 3, p.5 |

> 💡 **Deep Belief Networks (DBN)**: 여러 겹의 제한된 볼츠만 머신(RBM)을 쌓아 구성한 딥러닝 모델. 비지도 사전학습 후 지도학습으로 미세 조정한다.

---

### 2-1. 상세 설명

#### ① 해결하고자 하는 문제

- **고차원 센서 입력**: CMP 장비에는 수백 개의 센서가 내장되어 있어 입력 차원이 매우 높음 (p.2)
- **공정 동적 특성**: 웨이퍼 연마는 독립 사건이 아니며, 소모 부품(패드, 드레서) 상태에 의해 연속적으로 영향받음 (p.2)
- **모델 단독 한계**: 물리 모델은 미지 변수로 계수 미결정, 단독 데이터 모델은 비선형·동적 특성 포착 어려움

#### ② 제안하는 방법 (수식 포함)

**[Step 1: 피처 추출]**

**(a) Preston 방정식 기반 물리 피처** (Eq. 1, p.3):

$$ARR = K \cdot P \cdot V$$

- $ARR$: 평균 제거율 (Average Removal Rate)
- $K$: 프레스턴 상수 (모든 기타 물리 변수 통합)
- $P$: 웨이퍼-패드 계면 평균 압력
- $V$: 웨이퍼와 패드 테이블 간 상대 속도

> 💡 **Preston 방정식**: 1927년 Frank Preston이 제안한 광학 연마 경험 모델. CMP에서 제거율이 압력과 속도의 곱에 비례한다는 가장 기본적인 물리 모델이다.

**(b) 드레싱 레이트 (Dressing Rate)** (Eq. 2, p.3):

$$\text{Dressing Rate} = K_D \frac{V_D}{RA} \lambda d_0 \left(\frac{P}{H_p}\right)^{1.5}$$

- $K_D$: 드레싱 계수
- $V_D$: 드레싱 속도
- $R, A, \lambda, d_0$: 드레서의 기하학적·재료 관련 파라미터
- $P$: 적용 하중
- $H_P$: 패드 경도(hardness)

**(c) 단순화된 드레싱 레이트** (Eq. 3, p.3):

$$\text{Dressing Rate} = K_D U_D$$

- $U_D$: 드레서 사용량 (usage of dresser)
- 드레싱 속도, 하중, 경도가 런간 크게 변하지 않는다는 가정 하에 단순화

> 💡 **드레서(Dresser)**: CMP 패드 표면을 재생시키는 다이아몬드 소재의 컨디셔닝 도구. 사용할수록 성능이 저하된다.

**(d) 입자 스케일 MRR 모델** (Eq. 4, p.3):

$$MRR = f(F_s, P, V, U_P, U_D)$$

- $F_s$: 슬러리 관련 피처 (slurry flow 관련 통계량)
- $P$: 압력
- $V$: 상대 속도
- $U_P$: 패드 사용량 (usage of pad)
- $U_D$: 드레서 사용량

> 💡 **슬러리(Slurry)**: CMP에서 연마재(abrasive)와 화학물질이 혼합된 액체. 기계적 연마와 화학적 식각을 동시에 수행한다.

**(e) 시간 지연(Time Lag) 피처**:
- $r_{t-i}$: $i$번째 과거 MRR 값 ($i = 1, \ldots, 11$)
- MRR을 시계열로 취급하여 최근 11개 과거값을 피처로 활용

**(f) 사용량 최근접 이웃(Usage Nearest Neighbor) 피처**:
- 유클리드 거리로 소모 부품 사용량이 유사한 K개 이웃 웨이퍼 선택
- 해당 이웃들의 MRR을 입력 피처로 활용 (K=10)

$$d = \sqrt{\sum_{j} (U_j^{(a)} - U_j^{(b)})^2}$$

- $U_j^{(a)}, U_j^{(b)}$: 두 웨이퍼의 $j$번째 사용량 변수 (논문에서 직접 수식으로 명시되지 않았으나 KNN의 일반적 수식 적용)

---

**[Step 2: 피처 선택]**

- **Student's t-검정**: 선형 회귀에서 입력 피처와 MRR 간 유의미한 선형 관계 검증
- **OOB 피처 중요도**: Tree Bagging에서 각 피처 값을 무작위 치환(permutation) 후 예측 오차 증가량으로 중요도 평가
- 두 기준 모두에서 임계값(OOB: 0.15, t-stat: 1.5) 초과한 피처만 선택 (Figure 4, p.6)

> 💡 **OOB (Out-of-Bag)**: 배깅(Bagging)에서 부트스트랩 샘플링 시 선택되지 않은 데이터로, 별도의 검증 세트 없이 모델 성능과 피처 중요도를 평가하는 데 활용된다.

> 💡 **Student's t-검정**: 두 집단 간 평균 차이의 통계적 유의성을 검증하는 기법. 여기서는 회귀 계수가 0과 유의미하게 다른지를 검증한다.

---

**[Step 3: 모델 구성]**

| 모델 | 특징 |
|------|------|
| Persistent Model | $\hat{r}\_t = r_{t-1}$ (시계열 기준선 모델) |
| KNN Regression | 사용량 이웃 K개 MRR 평균 |
| Linear Regression (LR) | 선형 회귀 |
| SVR | 비선형 서포트 벡터 회귀 |
| Tree Bagging | 배깅 앙상블 트리 |

> 💡 **SVR (Support Vector Regression)**: 서포트 벡터 머신을 회귀에 적용한 것으로, 마진 내의 오차는 무시하고 복잡도를 최소화하는 초평면을 학습한다.

> 💡 **Tree Bagging**: 부트스트랩 샘플링으로 여러 개의 결정 트리를 학습한 후 평균을 내는 앙상블 기법. 분산을 줄이는 효과가 있다.

---

**[Step 4: 교차검증 및 가중 평균]**

**개별 모델 예측 오차 상한** (Eq. 5, p.4):

$$e = \text{mean}(\boldsymbol{\epsilon}) + 3 \cdot \text{std}(\boldsymbol{\epsilon})$$

- $\boldsymbol{\epsilon} \in \mathbb{R}^N$: CV 검증 테스트에서 얻은 오차 벡터
- $N$: CV 테스트 반복 횟수 (본 연구에서 $N=20$)
- $e$: 예측 오차의 상한값 (3-sigma 기준)

> 💡 **3-sigma 기준**: 정규분포에서 평균 ± 3표준편차 내에 99.7%의 데이터가 포함된다는 통계 원리. 여기서는 오차의 보수적 상한을 추정하는 데 활용된다.

**모델 가중치 계산** (Eq. 6, p.4):

$$\mathbf{w} = \frac{1/\mathbf{e}^3}{\text{sum}(1/\mathbf{e}^3)}$$

- $\mathbf{e} = [e_{persis}, e_{LR}, e_{SVR}, e_{KNN}, e_{treebagger}]$: 각 모델의 예측 오차 상한 벡터
- $\mathbf{w}$: 각 모델에 부여되는 가중치 벡터
- 오차가 작은 모델에 더 높은 가중치 부여 ($e^3$ 역수 사용으로 차등 강조)

> 💡 **Monte Carlo Cross-Validation**: 학습 데이터를 무작위로 반복 분할하여 검증하는 방법. 단일 분할의 편향을 줄이고 오차 분포의 변동성까지 평가할 수 있다.

---

#### ③ 모델 구조 (요약)

```
[원시 센서 데이터]
       ↓
[Step 1: 피처 추출]
  ├── 물리 피처: 압력/속도/슬러리/사용량의 통계량 (mean, std, range, AUC)
  ├── 시간 지연: r_{t-1} ~ r_{t-11}
  └── 사용량 KNN: 10개 이웃 MRR
       ↓
[Step 2: 피처 선택]
  ├── Student's t-검정 (LR 기반)
  └── OOB 피처 중요도 (Tree Bagging 기반)
       ↓
[Step 3: 5개 모델 병렬 학습]
  Persistent / KNN Reg. / LR / SVR / Tree Bagging
       ↓
[Step 4: Monte Carlo CV → 가중치 계산 → 가중 평균 앙상블]
       ↓
[최종 MRR 예측]
```

---

#### ④ 성능 향상 및 한계

**성능 향상** (Table 4, 5, p.7):

| 기준 | 통합 모델 | 최선 개별 모델(SVR) |
|------|-----------|---------------------|
| CV MSE (Overall) | **6.18** | 6.23 |
| Test MSE | **7.07** | 7.22 (Tree Bagging) |

**한계** (p.7, Section 5):
1. 피처 추출 시 통계 요약량만 사용 → 센서 신호의 복잡한 시간적 패턴 미포착
2. 정적(static) 모델 → 온라인 환경에서 설정 변경 시 모델 재학습 필요
3. 단일 데이터셋(PHM 2016) 검증 → 외부 타당도 미확인
4. 물리-데이터 융합 모델로의 발전 여지

---

## 3. 각 주장의 페이지/Figure/Table 위치

| 주장 | 위치 |
|------|------|
| Preston 방정식 기반 물리 피처 | p.3, Eq.(1)~(4) |
| 시간 지연 및 KNN 피처 | p.3 (Section 2.2) |
| 피처 선택 방법론 | p.3~4 (Section 2.3), Figure 4 (p.6) |
| 모델 통합 전략 (가중 평균) | p.4 (Section 2.4), Eq.(5)~(6) |
| 3 레시피 분리 예측 | p.4~5 (Section 3.2), Figure 3 (p.5), Table 2 (p.5) |
| CV 결과 (통합 모델 우수성) | Table 4 (p.7) |
| 테스트 결과 비교 | Table 5, Table 6 (p.7), Figure 5 (p.6) |
| DBN 비교 | Table 5 (p.7) |
| 한계 및 향후 연구 | p.7 (Section 5 마지막 단락) |

---

## 4. 저자 보고 결과 vs. 분석자 해석 분리

### 저자가 직접 보고한 결과

**[연구 주제]**
- CMP 공정에서의 MRR 가상 계측 향상을 위한 통합 모델 및 데이터 기반 접근법 (p.1, Abstract)

**[방법]**
- 4단계 파이프라인: 피처 추출 → 피처 선택 → 모델 구성 → CV 기반 가중 평균 (Figure 1, p.2)
- 물리 피처: Preston 방정식($ARR = K \cdot P \cdot V$), 입자 스케일 모델( $MRR = f(F_s, P, V, U_P, U_D)$ ) (p.3)
- 125개 피처 추출 후 t-검정/OOB로 선택 (Table 3, p.6)
- 20회 Monte Carlo CV로 가중치 결정 (p.4)

**[결과]**
- PHM 2016 Data Challenge 1위, MSE = 7.07 (Table 6, p.7)
- 통합 모델 CV MSE Overall: 평균 6.18, 표준편차 0.77 (Table 4, p.7)
- DBN(MSE 7.29) 대비 우수 (Table 5, p.7)

---

### 분석자(본 보고서)의 해석

1. **가중치 함수($1/e^3$)의 선택 근거 미제시**: 저자는 $e^3$의 역수를 가중치로 사용했으나, 이 지수(3)의 이론적 근거나 민감도 분석이 논문에 없음. 다른 지수값($e^2$, $e^4$)과의 비교 실험이 없어 최적성 미검증.

2. **시계열 특성 강도 해석**: Persistent Model이 KNN Regression을 모든 레시피에서 능가한 것은 저자도 언급했으나, 이는 사용량 변수의 KNN이 실질적인 상태 유사성을 충분히 포착하지 못했을 가능성을 시사함. 즉, 소모 부품 사용량만으로는 동적 상태 표현이 불충분할 수 있음.

3. **단일 데이터셋의 일반화 위험**: 1위 달성이라는 경쟁 성과는 해당 데이터셋에 대한 최적화 결과일 수 있으며, 다른 CMP 장비나 레시피로의 전이 가능성은 검증되지 않음.

4. **Cond3의 낮은 정확도**: Figure 5에서 Cond3의 예측 분산이 더 크게 나타났으며, Table 4에서도 Cond3 KNN이 높은 표준편차(0.93)를 보임. 이는 Cond3 레시피의 데이터 특성(샘플 수 약 350개로 가장 적음)에 기인할 가능성이 있음.

---

## 5. 통계적으로 취약한 부분 및 비교 불가능한 수치

| 항목 | 내용 | 취약성 유형 |
|------|------|------------|
| ⚠️ **가중치 지수 $e^3$** | 선택 근거 없음, 민감도 분석 없음 | 방법론적 임의성 |
| ⚠️ **CV 20회 선택 근거** | "20회 후 안정적"이라고 서술하나 수렴 기준 미제시 | 통계적 근거 불충분 |
| ⚠️ **DBN 비교** | Wang et al. (2017)의 DBN과 비교하나 동일 데이터셋 여부, 하이퍼파라미터 설정 등 미공개 | 비교 조건 불일치 가능 |
| ⚠️ **Table 6의 공동 2위(MSE 7.4 × 3팀)** | 동점 팀들의 방법론 미기술로 공정 비교 불가 | 비교 불가능 수치 |
| ⚠️ **Cond3 샘플 수** | 약 350개로 Cond1/2(~800개) 대비 현저히 적어 통계적 대표성 약함 | 샘플 불균형 |
| ⚠️ **OOB 임계값(0.15) 및 t-stat 임계값(1.5)** | 선택 근거 미제시 | 방법론적 임의성 |
| ⚠️ **단일 데이터셋 검증** | PHM 2016 데이터만 사용, 외부 검증 없음 | 외부 타당도 미확인 |

---

## 6. 문서가 답하지 않는 질문

1. **가중치 지수 선택 이유**: 왜 $1/e^3$인가? $1/e^2$ 또는 $1/e$와 비교했는가?
2. **KNN의 최적 K값**: 이웃 수 K=10의 선택 근거와 민감도는?
3. **시간 지연 수 11개의 근거**: 왜 11개인가? 더 많은/적은 지연이 성능에 미치는 영향은?
4. **온라인 적용 가능성**: 실시간 스트리밍 환경에서 지연 없이 적용 가능한가?
5. **다른 CMP 장비로의 전이 학습 가능성**: 제조사/모델이 다른 장비에도 적용 가능한가?
6. **이상치(Outlier) 처리 방법**: 전처리 과정에서 이상값 처리 방법이 기술되어 있지 않음
7. **물리 모델 계수 K, $K_D$의 결정 방법**: 실제 데이터에서 어떻게 추정했는가?
8. **레시피 분류 기준**: Stage + Chamber 조합 외 다른 분류 기준 검토 여부
9. **계산 복잡도 및 실시간 처리 시간**: 추론 지연(latency)은 얼마인가?
10. **피처 선택 결과의 재현성**: 매 CV 분할마다 선택 피처가 달라지는 불안정성 해결 방법

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p.2) — 통합 예측 접근법 플로우차트

```
Step 1: 피처 추출 → Step 2: 피처 선택 → Step 3: 모델 구성 → Step 4: CV/가중 평균
```

**해석**: 본 논문의 전체 방법론 구조를 한눈에 보여주는 핵심 다이어그램이다. 4단계가 순차적으로 연결되며, Step 1에서 물리 피처·시간 지연·KNN 이웃이라는 세 가지 이질적 피처 소스가 통합되는 것이 이 연구의 가장 큰 차별점임을 시각적으로 보여준다. Step 4에서 5개 모델의 출력이 가중 평균으로 앙상블되는 구조는 단일 모델의 약점을 상호 보완하는 설계 철학을 반영한다.

---

### Figure 2 (p.3) — CMP 장비 구조도

**해석**: 폴리싱 패드, 웨이퍼 캐리어, 드레서, 슬러리 디스펜서의 기계적 구성을 보여준다. 이 그림은 물리 피처 선택의 근거를 제공한다. 패드와 드레서가 소모성 부품임을 시각적으로 확인할 수 있으며, 사용량(Usage) 변수가 왜 MRR에 영향을 미치는지 직관적으로 이해하게 해준다. 특히 드레서가 패드를 컨디셔닝하는 구조는 Eq.(2)의 Dressing Rate 모델과 직접 연결된다.

---

### Figure 3 (p.5) — 학습 데이터 MRR 분포

**해석**: 세 레시피(Cond1: MRR ≈ 60~100, Cond2: MRR ≈ 50~100, Cond3: MRR ≈ 140~160)의 MRR 범위가 명확히 다름을 보여준다. 이는 레시피별 분리 예측 전략의 정당성을 수치적으로 입증한다. 또한 Cond3의 샘플 수(~350)가 Cond1/2(~800)보다 현저히 적어 데이터 불균형이 존재함을 시각적으로 확인할 수 있다. MRR 시계열에서 불규칙한 변동과 점진적 하락 패턴이 관찰되어 동적 피처의 필요성을 지지한다.

---

### Figure 4 (p.6) — Cond3 피처 선택 결과

**(a) OOB 피처 중요도 (임계값: 0.15), (b) Student's t-검정 (임계값: 1.5)**

**해석**: 125개 피처 중 두 기준 모두에서 임계값을 초과한 피처만 선택됨을 보여준다. (a)에서 소수의 피처만이 임계값(0.15)을 초과하고, (b)에서도 일부만이 t-stat 1.5를 넘어섬을 확인할 수 있다. 두 그래프를 AND 조건으로 결합함으로써 False Positive를 줄이는 보수적 선택 전략을 취하고 있다. 다만, 단일 CV 분할 결과임을 저자 스스로 인정하여 선택 결과의 가변성을 시인한다.

> 💡 **False Positive (위양성)**: 실제로는 중요하지 않은 피처를 중요하다고 잘못 선택하는 오류. 위양성이 많으면 모델 과적합(overfitting)으로 이어진다.

---

### Figure 5 (p.6) — 테스트 데이터 MRR 예측 결과

**해석**: 세 레시피 모두에서 예측값(○)과 실제값(+)이 근접하게 분포함을 보여준다. Cond1과 Cond2는 전반적으로 예측이 잘 이루어지나, Cond3은 일부 샘플에서 예측값과 실제값의 괴리가 더 크게 나타난다. 이는 Table 4에서 Cond3의 MSE가 다른 조건 대비 상대적으로 높은 것과 일치한다. 예측값 분포가 실제값의 변동 범위를 대체로 따라가는 것은 동적 피처(시간 지연, KNN)의 효용을 시각적으로 입증한다.

---

## 8. 결론: 시사점, 후속 연구 계획 및 추가 방향

### 8-1. 모델의 일반화 성능 향상 가능성

#### 저자가 제시한 시사점 (p.7, Section 5)

1. **피처 추출 고도화**: 현재 통계 요약량만 사용 → 더 정교한 시계열 패턴 마이닝 기법 도입 필요
2. **온라인 모델 갱신**: 모델 파라미터와 가중치의 주기적 업데이트로 변화하는 공정 설정에 적응 필요
3. **물리-데이터 융합**: Pillai et al. (2016)의 하이브리드 방식처럼 물리 표현과 데이터 모델의 더 깊은 결합
4. **추가 데이터셋 검증**: 다른 데이터셋으로 제안 방법의 범용성 확인 필요

#### 분석자가 추가 제안하는 일반화 성능 향상 방향

**① 전이 학습(Transfer Learning) 기반 도메인 적응**

다른 CMP 장비나 레시피로 학습된 모델을 새 환경에 적응시키는 방법. 데이터가 적은 Cond3 같은 상황에서 특히 유효하다.

$$\mathcal{L}_{total} = \mathcal{L}_{task} + \lambda \cdot \mathcal{L}_{domain}$$

- $\mathcal{L}_{task}$: MRR 예측 손실
- $\mathcal{L}_{domain}$: 소스-타깃 도메인 분포 차이 손실
- $\lambda$: 균형 파라미터

**② 베이지안 최적화 기반 하이퍼파라미터 자동 튜닝**

현재 임계값(OOB: 0.15, t-stat: 1.5), K값, 지연 수 등이 경험적으로 설정되어 있어 자동 최적화가 필요하다.

**③ 데이터 증강(Data Augmentation) 전략**

소수 레시피(Cond3)의 샘플 부족 문제를 GAN(Generative Adversarial Network)이나 SMOTE 기반으로 보완 가능하다.

> 💡 **SMOTE (Synthetic Minority Over-sampling Technique)**: 소수 클래스 데이터를 기존 샘플 사이를 보간(interpolation)하여 인위적으로 생성하는 데이터 증강 기법.

**④ 불확실성 정량화 (Uncertainty Quantification)**

현재 모델은 점 추정값만 제공하며 예측 신뢰도를 제시하지 않는다. Conformal Prediction 또는 베이지안 신경망을 통해 예측 구간을 제공하면 실제 공정 제어에서의 리스크 관리가 가능하다.

$$\hat{y} \pm z_{\alpha/2} \cdot \hat{\sigma}$$

- $\hat{y}$: MRR 예측값
- $\hat{\sigma}$: 예측 불확실성 추정
- $z_{\alpha/2}$: 신뢰 수준에 대응하는 z값

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **중요 고지**: 아래 연구들은 공개 학술 데이터베이스(IEEE Xplore, arXiv, Google Scholar)에서 확인된 연구 흐름을 기반으로 기술하였으나, 개별 수치나 세부 방법론은 해당 논문 원문에서 직접 확인이 필요합니다. 불확실한 세부 사항은 [미확인]으로 표기합니다.

#### 2020년 이후 주요 연구 동향 비교

| 연구 방향 | 대표 접근법 | Di et al. (2017) 대비 발전 |
|-----------|-------------|---------------------------|
| **딥러닝 기반 VM** | LSTM, Transformer 기반 MRR 예측 | 시계열 특징을 단순 지연(lag) 대신 순환 신경망으로 자동 추출 |
| **Graph Neural Network** | 다중 센서 간 관계를 그래프로 모델링 | 센서 간 상관관계를 구조적으로 학습 |
| **Physics-Informed Neural Networks (PINN)** | 물리 방정식을 손실 함수에 통합 | Di et al.의 물리 피처 수동 설계 vs. PINN의 자동 통합 |
| **Federated Learning** | 다수 공장 데이터를 프라이버시 보호하며 학습 | 단일 데이터셋 한계 극복 가능 |
| **Attention Mechanism** | 중요 시점/센서에 가중치 자동 부여 | 수동 피처 선택(t-검정/OOB) 대체 |
| **Conformal Prediction** | 예측 구간 제공 | 점 추정값만 제공하는 한계 극복 |

> 💡 **PINN (Physics-Informed Neural Networks)**: 물리 법칙을 신경망의 손실 함수에 제약 조건으로 포함시켜, 적은 데이터로도 물리적으로 타당한 예측을 하도록 유도하는 방법이다.

> 💡 **Federated Learning (연합 학습)**: 데이터를 중앙 서버로 모으지 않고 각 기기(공장)에서 로컬 학습 후 모델 파라미터만 공유하는 프라이버시 보존 학습 방법이다.

#### 본 논문이 향후 연구에 미치는 영향

1. **벤치마크 역할**: PHM 2016 데이터셋에 대한 최초 1위 달성 방법론으로 이후 연구의 기준점 제공
2. **하이브리드 접근법 선도**: 물리 모델 + 데이터 기반의 통합이 단독 딥러닝보다 유효함을 실증
3. **동적 피처의 중요성 입증**: 소모 부품 사용량 기반 KNN 피처 아이디어는 이후 상태 기반 VM 연구에 영향
4. **앙상블 가중 평균 전략**: 모델 불확실성을 활용한 가중치 설계 방법론은 후속 앙상블 연구의 참조점

#### 향후 연구 시 고려해야 할 사항

1. **실시간성**: 배포 환경에서의 추론 지연(inference latency) 최소화 - 본 논문은 오프라인 평가에만 초점
2. **설명 가능성(XAI)**: 물리 피처 선택은 해석 가능성을 높이지만, 앙상블 가중치의 의미 해석이 어려움
3. **공정 변화 감지**: 패드 교체, 슬러리 변경 등 급격한 공정 변화 감지 및 대응 메커니즘
4. **다중 품질 지표 동시 예측**: MRR 외에 비균일도(Non-Uniformity) 등 다른 품질 지표와의 동시 예측
5. **에너지 효율**: 배깅 트리 등의 모델은 계산 비용이 높아 엣지 컴퓨팅 환경에서의 경량화 필요

---

## 참고 문헌 목록

**[논문 내 인용 문헌]** (본 분석의 직접 참고 자료)
- Di, Y., Jia, X., & Lee, J. (2017). Enhanced Virtual Metrology on Chemical Mechanical Planarization Process using an Integrated Model and Data-Driven Approach. *International Journal of Prognostics and Health Management*, 2017 031.
- Luo, J., & Dornfeld, D. A. (2001). Material removal mechanism in chemical mechanical polishing. *IEEE Transactions on Semiconductor Manufacturing*, 14(2), 112-133.
- Tso, P. L., & Ho, S. Y. (2007). Factors influencing the dressing rate of CMP pad conditioning. *International Journal of Advanced Manufacturing Technology*, 33(7-8), 720-724.
- Wang, P., Gao, R. X., & Yan, R. (2017). A deep learning-based approach to material removal rate prediction in polishing. *CIRP Annals-Manufacturing Technology*.
- Breiman, L. (1996). Bagging predictors. *Machine Learning*, 24(2), 123-140.
- PHM Society 2016 Data Challenge Competition. http://www.phmsociety.org/events/conference/phm/16/data-challenge

**[2020년 이후 연구 동향 분석 참조 데이터베이스]**
- IEEE Xplore Digital Library: https://ieeexplore.ieee.org
- Google Scholar: https://scholar.google.com
- arXiv Preprint Server: https://arxiv.org

> ⚠️ **최종 고지**: 2020년 이후 비교 분석 섹션(8-2)의 개별 연구 세부 사항은 제공된 PDF 외부 자료에 기반하므로, 원문 확인을 권장합니다. 제공된 PDF 내용에 대한 분석은 100% 원문 기반입니다.
