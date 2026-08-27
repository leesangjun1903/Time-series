# PDMLP: Patch-based Decomposed MLP for Long-Term Time Series Forecasting

---

## 1. Executive Summary (10문장 이내)

PDMLP는 장기 시계열 예측(LTSF) 분야에서 Transformer 기반 모델의 한계를 극복하고자 제안된 순수 MLP 기반 경량 모델이다.  
저자들은 Patch 기반 Transformer의 성능이 Attention 메커니즘이 아닌 Patch 표현 방식에서 기인한다고 주장한다.  
이를 바탕으로, Patch를 활용한 Multi-Scale Patch Embedding(MPE)을 설계하여 다중 스케일 시간 관계를 포착한다.  
핵심 아이디어는 원시 시계열이 아닌 임베딩 벡터 공간에서 이동평균(Moving Average)을 통해 평활 성분(Smooth Component)과 잔차 성분(Residual Component)으로 분해하는 것이다.  
평활 성분에는 채널 믹싱(Channel Mixing)을 적용하여 변수 간 의미론적 정보를 교환하고, 잔차 성분에는 채널 독립(Channel Independence) 처리를 적용한다.  
8개의 실세계 벤치마크 데이터셋에서 90개 지표 중 42개에서 SOTA를 달성하며 2위인 iTransformer를 약 50% 상회한다.  
이 연구는 채널 독립 방식만이 효과적이라는 기존 통념에 반론을 제기하며, 변수 간 상호작용의 중요성을 재조명한다.  
또한 전통적인 원시 시계열 분해보다 임베딩 공간에서의 분해가 더 효과적임을 실험적으로 입증한다.  
구조가 단순하고 하이퍼파라미터에 비교적 둔감하여 실용성이 높다.  
저자들은 이 연구가 효율성·단순성·해석 가능성을 중시하는 새로운 LTSF 연구 방향을 촉발하길 기대한다.

### 1-1. 연구의 목적과 필요성

| 구분 | 내용 |
|------|------|
| **연구 목적** | Transformer 없이도 LTSF에서 SOTA 성능을 달성할 수 있는 단순·효율적 MLP 기반 모델 개발 |
| **필요성 ①** | Transformer 기반 LTSF 모델의 성능이 Attention이 아닌 Patch에서 기인한다는 가설 검증 필요 |
| **필요성 ②** | 채널 독립 방식의 한계: 변수 간 상호작용 정보가 예측 정확도 향상에 중요하나 기존 모델들이 이를 오용 |
| **필요성 ③** | 원시 시계열 분해의 한계: 복잡한 노이즈 혼재 시 트렌드/계절성 분리가 어려워 임베딩 공간 분해가 필요 |
| **필요성 ④** | 단일 스케일 Patch의 한계: 다양한 주기 패턴을 포착하려면 다중 스케일 Patch가 필요 |

---

## 2. 핵심 주장과 근거 표

| # | 핵심 주장 | 근거 | 위치 |
|---|-----------|------|------|
| ① | Transformer의 LTSF 성능은 Attention이 아닌 Patch 덕분이다 | Patch 없는 원래 Transformer는 단순 단층 MLP보다도 성능이 낮음 (Figure 2b) | p.3, Section 2 |
| ② | 채널 믹싱이 채널 독립보다 무조건 열등하지 않다 | 변수 간 강한 상관관계가 실재함 (Figure 6, 7); PDMLP의 Inter-Variable MLP가 성능 향상에 기여 (Table 2) | p.8, p.15-16 |
| ③ | 원시 시계열 분해보다 임베딩 벡터 분해가 효과적이다 | DPMLP(원시 분해)가 PDMLP(임베딩 분해)보다 일관되게 성능 낮음 (Table 2) | p.8, Section 4.2.1 |
| ④ | 다중 스케일 Patch(MPE)가 단일 스케일보다 우수하다 | MPE 제거 시 성능 하락 확인 (PDMLP vs. PDMLP¹, Table 2) | p.8, Section 4.2.1 |
| ⑤ | PDMLP는 SOTA 달성 | 90개 지표 중 42개 1위, 2위 iTransformer 대비 약 50% 더 많은 SOTA (Table 1) | p.7-8, Table 1 |
| ⑥ | 선형 모델은 긴 입력 길이에서 성능이 안정적으로 향상된다 | PDMLP와 DLinear는 입력 길이 증가 시 일관된 성능 향상; 일부 복잡한 모델은 오히려 저하 (Figure 5) | p.9, Section 4.2.2 |

---

## 2-1. 상세 설명

### 해결하고자 하는 문제

1. **Transformer의 허상 문제**: Patch 기반 Transformer가 좋은 성능을 보이지만, 이것이 self-attention의 효과인지 Patch 표현의 효과인지 불분명함.
2. **채널 독립의 과신 문제**: PatchTST 이후 채널 독립이 채널 믹싱보다 우수하다는 인식이 확산되었으나, 변수 간 상호작용 정보가 실질적으로 존재함.
3. **원시 분해의 비효율 문제**: 노이즈가 많은 원시 시계열을 직접 트렌드/계절성으로 분해하면 정확한 분리가 어려움.
4. **단일 스케일 Patch의 한계**: 하나의 Patch 크기로는 다양한 주기 패턴을 동시에 포착 불가.

---

### 제안하는 방법 (수식 포함)

#### 전체 입출력 정의

$$\mathcal{X} = \{x_1, \cdots, x_L \mid x_i \in \mathbb{R}^M\} \rightarrow \hat{\mathcal{X}} = \{x_{L+1}, \cdots, x_{L+T} \mid x_i \in \mathbb{R}^M\}$$

> - $\mathcal{X}$: 입력 시계열 (과거 관측값)
> - $\hat{\mathcal{X}}$: 예측 미래 시계열
> - $L$: 입력 시퀀스 길이 (look-back window)
> - $T$: 예측 시퀀스 길이 (forecast horizon)
> - $M$: 변수(채널) 수

---

#### 3.1 Multi-Scale Patch Embedding (MPE)

다변량 시계열 $\mathcal{X}$를 단변량 $x$로 분리한 후, 패치 집합 $\mathcal{P} = \{p_1, p_2, \ldots\}$ 적용:

$$x_p \in \mathbb{R}^{N \times p}, \quad N = \lfloor L / p \rfloor$$

> - $p \in \mathcal{P}$: 특정 스케일의 패치 크기 (본 논문에서는 $\{48, 24, 12, 6\}$ 사용)
> - $x_p$: 패치 크기 $p$로 분할된 패치 시퀀스
> - $N$: 패치 수

단층 선형 레이어로 각 패치를 임베딩:

$$x_e \in \mathbb{R}^{N \times d}$$

> - $x_e$: 임베딩된 패치 벡터
> - $d$: 해당 스케일의 임베딩 차원 (스케일마다 다를 수 있음)

각 스케일의 임베딩을 펼쳐서 연결(concatenate):

$$X \in \mathbb{R}^{1 \times d_{\text{model}}}$$

> - $X$: 최종 멀티스케일 임베딩 벡터
> - $d_{\text{model}}$: 모델 입력 임베딩 차원 (본 논문에서 1024)

> 💡 **Patch (패치)**: 연속적인 시간 포인트들을 하나의 묶음(토큰)으로 처리하는 방식. NLP의 단어 토큰, 이미지의 이미지 패치와 유사한 개념.

---

#### 3.2 Feature Decomposition (특징 분해)

임베딩 벡터 $X$를 Average Pooling으로 평활화:

$$X_s = \text{AvgPool}(X)$$
$$X_r = X - X_s \tag{1}$$

> - $X_s$: 평활 성분(Smooth Component) — 트렌드, 계절성 등 의미론적 정보
> - $X_r$: 잔차 성분(Residual Component) — 노이즈, 불규칙 변동
> - $\text{AvgPool}$: 평균 풀링 (Average Pooling); 패딩을 적용하여 길이 유지

> 💡 **Average Pooling (평균 풀링)**: 인접한 값들의 평균을 취하는 연산으로, 노이즈를 줄이고 부드러운 추세를 추출하는 데 사용됨.

---

#### 3.3 MLP Layer

**Intra-Variable MLP (변수 내부 MLP)**: 시간 도메인 내 패턴 학습, 변수 간 파라미터 공유

$$\text{FC} \rightarrow \text{GELU} \rightarrow \text{Dropout}$$

> 💡 **GELU (Gaussian Error Linear Unit)**: 비선형 활성화 함수. ReLU보다 부드럽고 Transformer에서 자주 사용됨. $\text{GELU}(x) = x \cdot \Phi(x)$ ($\Phi$: 정규 누적분포함수)

**Inter-Variable MLP (변수 간 MLP)**: 변수 간 상호작용 학습, 도트 곱 메커니즘 활용

$$\text{Output}_{\text{inter}} = \text{MLP}(X) \odot X$$

> - $\odot$: 원소별 곱(element-wise product / 도트 곱)
> - 도트 곱을 통해 단순 덧셈보다 강한 비선형 상호작용 표현 가능

**Residual Connection (잔차 연결)**: 각 MLP 이후 적용

$$X_{\text{out}} = \text{MLP}(X) + X$$

> 💡 **Residual Connection (잔차 연결)**: He et al. (2016)의 ResNet에서 제안. 입력값을 출력에 직접 더해 기울기 소실 문제를 완화하고 깊은 네트워크 학습을 안정화함.

---

#### 3.4 손실 함수 (Loss Function)

$$\mathcal{L} = \frac{1}{M} \sum_{i=1}^{M} \left\| \hat{x}^i_{L+1:L+T} - x^i_{L+1:L+T} \right\|^2_2 \tag{2}$$

> - $\mathcal{L}$: MSE 손실값
> - $\hat{x}^i_{L+1:L+T}$: $i$번째 변수에 대한 예측값 시퀀스
> - $x^i_{L+1:L+T}$: $i$번째 변수에 대한 실제값 시퀀스
> - $M$: 변수 수
> - $\|\cdot\|_2^2$: L2 노름의 제곱 (Mean Squared Error)

---

### 모델 구조 요약

```
입력 X (M × L)
    │
    ▼
[Multi-Scale Patch Embedding]
 Patch 크기: {48, 24, 12, 6}
 각각 Linear 임베딩 → Flatten → Concatenate
    │
    ▼ X ∈ R^{1 × d_model}
[Feature Decomposition]
 AvgPool → X_s (평활 성분)
 X - X_s → X_r (잔차 성분)
    │
  ┌─┴─┐
  ▼   ▼
[MLP Layer for X_s]    [MLP Layer for X_r]
  채널 믹싱(Inter)          채널 독립(Intra only)
  + Intra MLP              
  │
  └───┬───┘
      ▼  (합산/projection)
[Projection: R^{d_model} → R^T]
      │
      ▼
출력 X̂ (M × T)
```

---

### 성능 향상 및 한계

| 구분 | 내용 |
|------|------|
| **성능 향상** | 90개 지표 중 42개 SOTA; iTransformer 대비 평균 MSE 약 3~15% 개선 (데이터셋별 상이) |
| **효율성** | 순수 MLP 기반으로 Transformer 대비 낮은 연산 복잡도 |
| **강건성** | 입력 길이 증가 시 성능이 안정적으로 향상 (Figure 5) |
| **한계 ①** | NeurIPS 체크리스트에서 저자 스스로 한계(Limitations) 섹션 없음을 인정 (p.24, 항목 2) |
| **한계 ②** | 오차 막대(error bar), 신뢰구간 등 통계적 유의성 검증 미제공 (p.26, 항목 7) |
| **한계 ③** | Traffic 데이터셋에서 iTransformer보다 낮은 성능 (Avg MSE: 0.452 vs 0.428) |
| **한계 ④** | Solar-Energy 일부 지표에서도 iTransformer에 미치지 못함 |
| **한계 ⑤** | 단기 시계열 예측(Short-term TSF)에 대한 평가 미실시 |
| **한계 ⑥** | 이론적 수렴 증명 없음; 완전히 실험적 기반 (p.25, 항목 3) |

---

## 3. 각 주장에 페이지/Figure/Table 번호 표시

| 주장 | 위치 |
|------|------|
| Attention보다 Patch가 핵심 | p.2-3, Figure 1, Figure 2(a)(b)(c) |
| 채널 믹싱 유효성 | p.2, p.8, p.15-16, Figure 6, Figure 7 |
| 임베딩 벡터 분해 제안 | p.5, Eq.(1), Figure 3 |
| MPE 제안 | p.4-5, Section 3.1 |
| MLP Layer 구조 | p.5-6, Figure 4, Section 3.3 |
| 손실 함수 | p.6, Eq.(2) |
| SOTA 성능 달성 | p.7-8, Table 1 |
| 절제 연구(Ablation) | p.8, Table 2 |
| 입력 길이 확대 실험 | p.9, Figure 5 |
| 하이퍼파라미터 민감도 | p.21, Figure 13 |
| 정규화 비교 | p.21-22, Table 6 |
| 한계 미기술 | p.24, NeurIPS Checklist 항목 2 |
| 통계 유의성 미제공 | p.26, NeurIPS Checklist 항목 7 |

---

## 4. 저자 직접 보고 결과 vs. 추가 해석

### 4-1. 저자가 직접 보고한 결과

| 항목 | 저자 보고 내용 |
|------|---------------|
| **SOTA 달성** | "90 metrics, 42 SOTA results were attained, surpassing the second-ranked iTransformer by nearly 50%" (p.8) |
| **Patch 효과** | Patch 크기 증가 시 MSE가 감소 후 증가/안정화 패턴 (Figure 2b, p.3) |
| **임베딩 분해 우위** | DPMLP(원시 분해) < PDMLP(임베딩 분해) 일관 확인 (Table 2, p.8) |
| **MPE 유효성** | MPE 제거(PDMLP¹) 시 성능 하락 관찰 (Table 2, p.8) |
| **도트 곱 유효성** | 도트 곱 제거(PDMLP²) 시 성능 저하 관찰 (Table 2, p.8) |
| **하이퍼파라미터 둔감** | LR, 블록 수, d_model 변화에도 성능 변동 제한적 (Figure 13, p.21) |
| **Layer Norm 우위** | 대부분 실험에서 LN이 BN보다 우수 (Table 6, p.22) |

### 4-2. 추가 해석 (내 분석)

| 항목 | 추가 해석 |
|------|----------|
| **"50% 상회"의 실질적 의미** | SOTA 지표 수의 비율(42 vs. ~28)이지, 오차 크기가 50% 개선된 것은 아님. 실제 MSE 개선폭은 데이터셋별로 0~5% 수준으로 미미한 경우도 있음 |
| **Traffic 성능 역전** | PDMLP(Avg MSE 0.452)가 iTransformer(0.428)보다 열세. 고차원 변수(862개)에서 채널 믹싱의 노이즈 민감성이 나타날 수 있음 |
| **도트 곱 메커니즘** | 저자는 "비선형 표현력 강화"로 설명하나, 이것이 Gating 메커니즘과 유사하게 작동할 수 있음. 이론적 분석 부재가 아쉬움 |
| **입력 길이 96의 한계** | 기본 입력 길이가 96으로 매우 짧음. 실제로 더 긴 look-back이 필요한 도메인(금융, 기후 등)에서의 성능은 추가 검증 필요 |
| **임베딩 분해의 해석** | AvgPool 기반 분해는 이동평균과 동일한 원리로, 전통적 HP 필터나 STL 분해보다 단순함. 이 단순성이 오히려 임베딩 공간에서의 부드러운 분해에 적합할 수 있음 |

---

## 5. 통계적으로 취약한 부분과 비교 불가능한 수치

| 구분 | 해당 내용 | 문제점 |
|------|-----------|--------|
| ⚠️ **오차 막대 없음** | Table 1, Table 2의 모든 수치 | 표준편차/신뢰구간 미제공; 저자 스스로 NeurIPS 체크리스트에서 [No] 인정 (p.26) |
| ⚠️ **실험 조건 혼재** | iTransformer 기반 결과 vs. 자체 6회 실행 평균 | "except for results derived from the iTransformer repository, all other results were obtained by conducting six separate runs" (p.20) → 기준 불일치 |
| ⚠️ **Solar-Energy Avg 오류 의심** | Table 1의 Solar-Energy 행, RLinear Avg 수치 (0.369, **0.270**) | 개별 수치(0.339, 0.356, 0.369, 0.397)의 평균이 0.270이 될 수 없음; 표 인쇄 오류로 추정 |
| ⚠️ **비교 불가능 수치: Traffic** | PDMLP Avg MSE 0.452 vs. iTransformer 0.428 | PDMLP가 열세이나 본문에서 이를 충분히 설명하지 않음 |
| ⚠️ **단일 GPU 실험** | NVIDIA V100 16GB 1대 (p.20) | 대규모 데이터셋에서의 확장성(scalability) 미검증 |
| ⚠️ **통계적 유의성 검증 부재** | 전체 실험 | t-검정, Wilcoxon 검정 등 통계적 유의성 검증 없음 |
| ⚠️ **Figure 13 해석 주의** | 하이퍼파라미터 민감도 실험 | 4개 데이터셋만 사용; 모든 데이터셋에서 둔감하다고 일반화하기 어려움 |

---

## 6. 문서가 답하지 않는 질문

| # | 미답 질문 |
|---|----------|
| Q1 | 단기 예측(Short-term TSF)에서도 PDMLP가 효과적인가? |
| Q2 | 변수 수($M$)가 매우 클 때(수천 개 이상) Inter-Variable MLP의 계산 복잡도는 어떻게 되는가? |
| Q3 | AvgPool 커널 크기 선택 기준은 무엇인가? 어떻게 최적값을 결정하는가? |
| Q4 | 도트 곱 메커니즘이 단순 곱셈 이상의 효과를 내는 이론적 근거는 무엇인가? |
| Q5 | 비정상(non-stationary) 시계열 혹은 분포 이동(distribution shift)이 있는 데이터에서 성능이 어떻게 되는가? |
| Q6 | Projection 레이어에서 채널 독립을 선택한 근거(임베딩 분해 이후 왜 다시 독립 처리?)가 무엇인가? |
| Q7 | 훈련 시간 및 추론 지연(inference latency) 비교 데이터가 없다. PDMLP의 실제 속도 이점은 얼마나 되는가? |
| Q8 | Traffic 데이터셋에서 iTransformer 대비 열세인 이유에 대한 분석이 부재함. |
| Q9 | $d_{\text{model}} = 1024$의 선택 근거가 무엇인가? (Figure 13에서 ETTh2 등 소규모 데이터셋에서는 오히려 과적합 가능성 있음) |
| Q10 | 멀티스케일 패치 $\{48, 24, 12, 6\}$의 선택 근거 및 다른 도메인으로의 전이 가능성은? |

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p.2) — Self-Attention Score 시각화

**내용**: 2층 Transformer를 ETTh1에서 훈련 후 Patch 크기 $\{1, 4, 6, 16\}$에서의 Self-Attention Score 히트맵.

**해석**:
- Patch=1 (원래 Transformer): 격자(바둑판) 패턴이 뚜렷 → 유사한 시점끼리 유사한 가중치를 가짐 → 순열 불변(permutation-invariant)한 Attention의 한계 노출
- Patch=16: 블록 패턴이 줄어들고 더 균일한 분포
- **핵심 인사이트**: Patch 크기가 커질수록 Attention의 비효율적 패턴이 완화되지만 완전히 해소되지 않음 → "Patch가 Attention을 구원한다"는 저자의 주장을 시각적으로 지지
- **추가 해석**: 이는 시계열 데이터가 자연적으로 Patch 구조(locality)를 선호함을 의미하며, Attention이 이 구조를 억지로 학습하느라 비효율이 발생함을 보여줌

> 💡 **순열 불변(Permutation-Invariant)**: 입력 순서를 바꿔도 출력이 동일한 성질. Self-Attention은 이 성질을 가져 시간 순서 정보가 손실될 수 있음.

---

### Figure 2 (p.3) — Patch Transformer의 민감도 분석

**내용**: ETTh2에서 (a) 입력 길이 vs. MSE, (b) Patch 크기 vs. MSE, (c) Patch 크기 + d_model vs. MSE.

**해석**:
- **(a)**: 입력 길이가 길어질수록 성능이 단조 감소(↓MSE) → 긴 컨텍스트가 도움
- **(b)**: 동일 입력 길이에서 Patch 크기가 너무 작거나 크면 성능 저하 → 최적 Patch 크기 존재; 입력 길이가 길수록 더 큰 Patch가 필요
- **(c)**: $d_{\text{model}}$이 클수록 성능 향상; 큰 Patch는 더 큰 $d_{\text{model}}$과 함께 사용해야 효과적
- **핵심 인사이트**: Patch 크기와 모델 크기의 균형이 중요. 극단적으로 큰 Patch는 iTransformer(전체 시계열을 1개 토큰)와 동치화됨
- **추가 해석**: 이 결과는 PDMLP의 MPE 설계 동기를 제공 — 단일 Patch로는 최적화 불가능, 다중 스케일이 필요

---

### Figure 3 (p.4) — PDMLP 전체 구조

**내용**: PDMLP의 4대 모듈(MPE, Feature Decomp, MLP Layer, Projection)의 정보 흐름도.

**해석**:
- 좌측 상단: 다중 스케일 Patch로 입력 시퀀스를 임베딩
- 중앙: AvgPool로 임베딩 벡터를 $X_s$와 $X_r$로 분해
- 상단 경로($X_s$): 채널 믹싱 포함 MLP Layer → 변수 간 의미론적 정보 교환
- 하단 경로($X_r$): 채널 독립 MLP Layer → 노이즈 개별 처리
- 우측: 두 경로의 결과를 합산 후 Projection으로 예측값 생성
- **핵심 인사이트**: 분해 기반의 "분리 처리" 전략이 핵심 — 평활 성분(규칙적)과 잔차 성분(불규칙)에 서로 다른 처리 방식 적용
- **추가 해석**: 이 구조는 전통적인 시계열 분해(STL, Autoformer)와 달리 임베딩 공간에서 수행되므로 원시 데이터의 복잡한 노이즈 구조에 덜 민감함

---

### Figure 5 (p.9) — 증가하는 Look-back Window 실험

**내용**: ETTh1, ETTm2, Weather에서 입력 길이 $L \in \{192, 288, 384, 480, 576, 672, 768\}$에 따른 MSE 변화.

**해석**:
- PDMLP와 DLinear: 입력 길이 증가에 따라 MSE가 단조 감소(↓) — 더 많은 과거 정보를 효과적으로 활용
- iTransformer, PatchTST, TimesNet: 일부 구간에서 MSE 증가 또는 불안정 → 긴 입력 처리 능력 한계
- **핵심 인사이트**: 선형 기반 모델의 "장기 의존성 포착 능력"이 Transformer보다 안정적일 수 있음을 시사
- **추가 해석**: 이는 Transformer가 Long-term Dependency를 포착한다는 일반적 믿음에 반하는 결과. 실제로는 긴 입력에서 Attention의 노이즈 민감성이 증가할 수 있음

> 💡 **Look-back Window (과거 관찰 구간)**: 모델이 예측 시 참조하는 과거 시점의 범위. 길수록 더 많은 역사적 패턴을 활용 가능하나 모델의 장기 의존성 학습 능력이 필요.

---

### Figure 6 & 7 (p.15) — 다변량 시계열 변수 간 상관관계 시각화

**내용**: Weather, ECL, Traffic 데이터에 PCA/t-SNE 적용; ETTh2, Exchange 데이터의 쌍별 산점도.

**해석**:
- **Figure 6**: PCA/t-SNE에서 서로 다른 변수들이 겹쳐 분포 → 변수 간 명확한 경계 없음 → 상호작용 정보 존재
- **Figure 7(a) ETTh2**: 대각선 외 서브플롯에서 선형적 정/부 상관 관계 뚜렷
- **Figure 7(b) Exchange**: 대부분 변수가 양의 상관관계
- **핵심 인사이트**: 변수 간 상관관계가 실질적으로 존재하므로, 채널 독립(CI) 방식은 이 정보를 버리는 손실이 발생. 채널 믹싱의 재도입이 정당화됨
- **추가 해석**: 그러나 Traffic처럼 변수가 862개에 달하는 경우, 모든 변수 쌍의 상관관계를 Dense하게 모델링하면 오히려 노이즈 상관이 증폭될 수 있어 채널 독립이 우세할 수 있음. 이것이 Traffic에서 PDMLP가 iTransformer보다 낮은 원인 중 하나일 수 있음.

> 💡 **PCA (Principal Component Analysis, 주성분 분석)**: 고차원 데이터를 저차원으로 압축하는 선형 차원 축소 기법.
> 💡 **t-SNE (t-Distributed Stochastic Neighbor Embedding)**: 고차원 데이터의 비선형 구조를 2D/3D로 시각화하는 기법.

---

## 8. 결론 및 후속 연구

### 8-1. 저자 제시 시사점 및 후속 연구 계획

| 구분 | 내용 |
|------|------|
| **시사점 ①** | Patch 메커니즘이 Transformer 성능의 실질적 원인 → 복잡한 Attention 불필요 |
| **시사점 ②** | 채널 믹싱과 채널 독립의 이분법적 선택이 아닌, 분해 기반 혼합 전략이 효과적 |
| **시사점 ③** | 임베딩 공간에서의 분해가 원시 분해보다 우수 → 분해 위치(공간) 자체가 중요한 설계 변수 |
| **후속 연구 방향 (저자)** | "효율성, 단순성, 해석 가능성을 우선시하는 모델 개발 촉진" (p.9) |
| **후속 연구 방향 (저자)** | "특정 문제 해결에 집중하는 혁신적 예측 방법 창출" (p.9) |

---

### 8-1 (심화). 모델의 일반화 성능 향상 가능성

현재 PDMLP의 일반화 측면에서 다음 개선 방향이 존재한다:

**① 도메인 적응(Domain Adaptation) 강화**

현재 PDMLP는 각 데이터셋에 독립적으로 훈련됨. 사전학습(Pre-training) + 파인튜닝(Fine-tuning) 패러다임을 도입하면 소량 데이터 시나리오에서도 일반화 가능성이 높아짐:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{forecast}} + \lambda \mathcal{L}_{\text{domain}}$$

> 💡 **사전학습(Pre-training)**: 대규모 범용 데이터로 모델을 먼저 훈련하여 일반적 표현을 학습한 후, 특정 과제에 추가 학습하는 전략.

**② 적응형 Patch 크기 선택**

고정된 $\{48, 24, 12, 6\}$ 대신, 데이터의 스펙트럼 분석이나 학습 기반으로 Patch 크기를 동적 선택:

$$p^* = \arg\min_{p \in \mathcal{P}} \mathcal{L}_{\text{val}}(p)$$

**③ 분포 이동(Distribution Shift) 대응**

실세계에서 시계열의 통계적 특성이 시간에 따라 변함 (개념 드리프트). 슬라이딩 윈도우 정규화나 RevIN(Reversible Instance Normalization) 도입:

$$\tilde{x} = \frac{x - \mu_{\text{batch}}}{\sigma_{\text{batch}}}$$

> 💡 **분포 이동(Distribution Shift)**: 학습 데이터와 테스트 데이터의 통계적 분포가 달라지는 현상. 시계열에서는 비정상성(non-stationarity)으로 나타남.

**④ 변수 수 확장성 개선**

Traffic(862개 변수)에서 Inter-Variable MLP의 파라미터 복잡도가 $O(M^2)$에 비례할 수 있음. Sparse Attention이나 Graph-based 변수 선택으로 확장성 개선 가능.

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

| 모델 | 연도 | 핵심 방법 | PDMLP와의 관계 |
|------|------|-----------|----------------|
| **Informer** (Zhou et al.) | 2021 | ProbSparse Attention, 긴 시퀀스 효율화 | PDMLP가 비교 대상으로 간접 참조; Attention 기반 한계를 지적하는 근거 |
| **Autoformer** (Wu et al.) | 2021 | Auto-Correlation + 시계열 분해 (Moving Average) | PDMLP의 분해 아이디어의 기원; 그러나 PDMLP는 임베딩 공간에서 분해하여 차별화 |
| **DLinear** (Zeng et al.) | 2023 | 단순 선형 분해 모델; Transformer보다 우수 | PDMLP의 "단순 모델이 효과적" 가설의 선행 근거; PDMLP가 DLinear를 대부분 능가 |
| **PatchTST** (Nie et al.) | 2022 | Patch + Transformer + 채널 독립 | PDMLP 설계의 핵심 기반; 채널 독립에서 채널 믹싱으로의 전환점 |
| **iTransformer** (Liu et al.) | 2023 | 역전된 Transformer (변수 차원으로 Attention) | PDMLP의 주요 비교 대상; PDMLP가 대부분 우세하나 Traffic에서 열세 |
| **TimeMixer** (Wang et al.) | 2023 | 다중 스케일 분해 + MLP 믹싱 | PDMLP와 가장 유사한 동시기 연구; 다중 스케일 분해를 공유 |
| **TSMixer** (Chen et al.) | 2023 | MLP-Mixer 기반 시계열 모델 | PDMLP와 MLP 기반 접근을 공유; 채널 믹싱 방식에서 유사성 |
| **Pathformer** (Chen et al.) | 2024 | 다중 스케일 Transformer + 적응형 경로 | 다중 스케일 개념 공유; PDMLP보다 복잡하지만 더 정교한 경로 선택 |
| **HDMixer** (Huang et al.) | 2024 | 계층적 의존성 + 확장 가능 Patch | PDMLP와 Patch 기반 MLP 개념 공유; 경계 정보 강화로 차별화 |

---

### PDMLP가 미래 연구에 미치는 영향

**1. Transformer 의존도 탈피 가속화**

DLinear의 충격에 이어 PDMLP는 "복잡성 없이도 SOTA 가능"임을 재확인. 이는 향후 LTSF 연구에서 모델 복잡도 정당화의 기준점을 높임.

**2. 분해 위치의 설계 공간 확장**

"어디서 분해할 것인가"(원시 공간 vs. 임베딩 공간 vs. 주파수 공간)가 새로운 설계 변수로 부각됨. 이 방향에서 후속 연구가 활발해질 전망.

**3. 채널 전략의 재정립**

채널 독립 vs. 채널 믹싱의 이분법을 넘어, 성분별로 다른 채널 전략을 적용하는 "선택적 채널 전략"이 새로운 패러다임으로 확립될 수 있음.

---

### 앞으로 연구 시 고려할 점

| 고려 사항 | 세부 내용 |
|-----------|-----------|
| **📌 통계적 엄밀성** | 오차 막대, 신뢰구간, 통계적 유의성 검증(t-test, Wilcoxon)을 반드시 포함해야 함 |
| **📌 한계 명시** | 논문에 Limitations 섹션을 명시적으로 포함하여 실패 사례와 적용 범위를 기술 |
| **📌 확장성 검증** | 변수 수 $M$이 매우 큰 경우(수천~수만)와 초장기 예측($T > 1000$)에서의 성능 검증 |
| **📌 비정상 시계열** | 금융, 이상 기후 등 비정상(non-stationary) 데이터에서의 견고성 테스트 |
| **📌 적응형 설계** | 고정된 Patch 크기와 커널 크기를 데이터 특성에 따라 동적으로 선택하는 메커니즘 탐색 |
| **📌 계산 효율성** | 파라미터 수, FLOPs, 메모리 사용량의 명시적 비교가 필요 |
| **📌 사전학습 가능성** | PDMLP의 단순 구조는 Foundation Model로 확장하기 용이; Large-scale Pre-training 연구 가능 |

---

## 참고 자료

본 답변은 다음 자료를 참고하였습니다:

1. **Tang, P., & Zhang, W. (2024). "PDMLP: Patch-based Decomposed MLP for Long-Term Time Series Forecasting." arXiv:2405.13575v2** — 분석 대상 논문 (제공된 PDF)

2. **Nie, Y., et al. (2022). "A time series is worth 64 words: Long-term forecasting with transformers." (PatchTST)** arXiv:2211.14730 — 논문 내 참조

3. **Zeng, A., et al. (2023). "Are transformers effective for time series forecasting?" (DLinear)** AAAI 2023 — 논문 내 참조

4. **Liu, Y., et al. (2023). "iTransformer: Inverted Transformers are Effective for Time Series Forecasting."** arXiv:2310.06625 — 논문 내 참조

5. **Wu, H., et al. (2021). "Autoformer: Decomposition transformers with autocorrelation for long-term series forecasting."** NeurIPS 2021 — 논문 내 참조

6. **He, K., et al. (2016). "Deep residual learning for image recognition."** CVPR 2016 — 논문 내 참조

7. **Wang, S., et al. (2023). "TimeMixer: Decomposable multiscale mixing for time series forecasting."** ICLR 2024 — 논문 내 참조

8. **Chen, S.-A., et al. (2023). "TSMixer: An all-MLP architecture for time series forecasting."** arXiv:2303.06053 — 논문 내 참조

> ⚠️ **정확도 고지**: 8-2의 최신 연구 비교 분석 중 일부 모델(TimeMixer, Pathformer, HDMixer)의 세부 수치 비교는 본 논문 내 직접 언급 정보와 논문 참조 목록을 기반으로 작성되었으며, 해당 모델들의 원문을 직접 확인하지 않은 부분은 논문 내 맥락 기반 추론임을 명시합니다.
