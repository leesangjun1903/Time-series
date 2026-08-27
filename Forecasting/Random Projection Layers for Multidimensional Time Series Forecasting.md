# Random Projection Layers for Multidimensional Time Series Forecasting

> **참고 논문**: Yeh et al. (2024). "RPMixer: Shaking Up Time Series Forecasting with Random Projections for Large Spatial-Temporal Data." *Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining.* (arXiv:2402.10487v3)
>
> **⚠️ 주의**: 본 논문은 preprint 버전이며, 2024년 ACM SIGKDD에 게재된 최종 버전([47])의 사전 공개본입니다. 일부 수치는 최종본과 다를 수 있습니다.

---

## 1. Executive Summary (10문장 이내)

1. 본 논문은 고차원 시공간 데이터(Spatial-Temporal Data)에서 기존 all-MLP 계열 모델인 TSMixer가 과적합(Overfitting) 문제로 성능이 저하되는 한계를 극복하기 위해 RPMixer를 제안한다.
2. RPMixer는 잔차 연결(Residual Connection)을 통해 딥러닝 모델이 앙상블(Ensemble)처럼 동작한다는 이론적 해석에 기반하여 설계되었다.
3. 핵심 아이디어는 각 Mixer 블록의 출력 다양성을 높이기 위해 **무작위 투영(Random Projection) 레이어**를 공간 혼합기(Spatial Mixer) 내에 통합하는 것이다.
4. 무작위 투영 레이어는 가중치가 고정된 선형 레이어로, 각 블록이 서로 다른 노드 조합에 집중하도록 유도하여 앙상블 기반 학습자의 다양성을 실현한다.
5. 사전 활성화(Pre-Activation) 설계를 통해 항등 매핑(Identity Mapping) 연결을 강화하여 앙상블 해석을 위한 핵심 구조적 요건을 충족한다.
6. 주기적 시계열 처리를 위해 복소수 선형 레이어(Complex Linear Layer)를 활용하여 주파수 영역에서 시간 혼합을 수행한다.
7. 실험은 캘리포니아 교통 데이터 기반 LargeST 벤치마크(SD, GBA, GLA, CA 4개 서브셋)에서 14개 베이스라인과 비교 수행되었다.
8. RPMixer는 특히 대규모 데이터셋(GLA, CA)에서 모든 지표(MAE, RMSE, MAPE)에 걸쳐 기존 그래프 기반 방법 및 일반 시계열 예측 방법 대비 최고 성능을 달성하였다.
9. 또한 메모리 복잡도가 노드 수에 대해 선형적으로 증가하여, 그래프 기반 방법이 메모리 초과로 실패하는 대규모 데이터셋(CA: 8,600개 센서)에도 적용 가능하다.
10. Appendix에서 수행된 장기 시계열 예측(ETT, Weather 등) 실험에서도 TSMixer 및 PatchTST와 동등한 수준의 성능(t-test, $\alpha=0.05$)을 보여 범용성을 입증하였다.

---

### 1-1. 연구의 목적과 필요성

**[배경 및 문제 상황 — p.1, Introduction]**

현실 세계의 교통 흐름 예측, 수요 예측, 이상 감지 등 다양한 문제에서 **다차원 시공간 시계열 예측**이 핵심적으로 활용된다. 기존에는 GCN(Graph Convolutional Network) 기반의 복잡한 공간-시간 모델들이 주류를 이루었으나, 노드 수가 수천 개에 달하는 대규모 데이터셋에서는 메모리 부족(Out-of-Memory) 문제가 발생한다.

이를 해결하기 위해 TSMixer와 같은 경량 all-MLP 모델이 제안되었으나, 고차원 데이터에 적용할 경우 과적합 문제로 성능이 저하된다는 한계가 있다. 본 연구는 이러한 **"대규모 고차원 시공간 데이터에서의 과적합 방지 및 성능 향상"**을 목적으로 하며, 랜덤 투영을 통해 앙상블 다양성을 극대화하는 새로운 아키텍처를 제안한다.

> 💡 **용어 설명**
> - **GCN (Graph Convolutional Network)**: 노드와 엣지로 구성된 그래프 구조 데이터를 처리하는 신경망. 각 노드의 특성을 이웃 노드 정보를 통해 업데이트한다.
> - **all-MLP (all-Multi Layer Perceptron)**: Transformer나 CNN 없이 오직 완전 연결(Fully Connected) 선형 레이어만으로 구성된 모델 구조.
> - **과적합(Overfitting)**: 모델이 훈련 데이터에 지나치게 맞춰져 새로운 데이터에 대한 예측력이 떨어지는 현상.

---

## 2. 핵심 주장과 근거 표

| # | 핵심 주장 | 근거 | 위치 |
|---|-----------|------|-------|
| 1 | 랜덤 투영 레이어가 블록 간 표현 다양성을 증가시킨다 | 제안 모델과 미적용 모델의 블록 중간 출력 비교 시각화 | Fig. 1, Fig. 8 |
| 2 | 사전 활성화 설계(Pre-Activation)가 항등 매핑을 가능하게 하여 앙상블 해석을 지원한다 | 수식 (3)~(5) 전개, 제거 시 성능 급락 | Section 4.2, Fig. 7 |
| 3 | RPMixer가 대규모 시공간 벤치마크에서 모든 기존 방법보다 우수하다 | 4개 데이터셋, 14개 베이스라인 대비 최고 MAE/RMSE/MAPE | Table 1 |
| 4 | 메모리 복잡도가 노드 수에 선형적으로 스케일된다 | CA(8,600 노드) 적용 가능, 그래프 모델 OOM 발생 비교 | Table 1, Section 5.3 |
| 5 | 랜덤 투영 레이어가 장기 예측에 더 큰 이점을 제공한다 | 시간 단계별 성능 분석 | Fig. 13 (Appendix A.5) |
| 6 | 장기 시계열 예측에서도 경쟁력 있는 성능을 보인다 | ETT 등 7개 데이터셋 t-test 동등 검증 | Table 3, Appendix A.6 |
| 7 | 랜덤 투영 후에도 시계열의 주요 패턴(일간/주간)이 보존된다 | Johnson-Lindenstrauss 보조정리 + 시각적 검증 | Fig. 4, Section 4.1 |

> 💡 **용어 설명**
> - **Johnson-Lindenstrauss 보조정리**: 고차원 데이터를 저차원으로 랜덤 투영해도 데이터 포인트 간의 거리(유사도)가 높은 확률로 보존된다는 수학적 정리.

---

## 2-1. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 🔴 해결하고자 하는 문제

**[p.1, Section 1 / p.3, Section 3]**

- **문제 1**: 고차원 시공간 데이터에서 TSMixer 등 기존 all-MLP 모델의 과적합 문제
- **문제 2**: GCN 기반 모델들의 대규모 데이터 확장성 한계 (메모리 폭발)
- **공식 정의 (Problem 1)**:

$$F(X_{\text{past}}, A) \rightarrow X_{\text{future}}$$

> - $A \in \mathbb{R}^{n \times n}$: 인접 행렬(Adjacency Matrix), $n$개 엔티티 간 공간적 관계 저장
> - $X_{\text{past}} \in \mathbb{R}^{n \times t_{\text{past}}}$: 과거 $t_{\text{past}}$ 타임스텝의 시계열 행렬
> - $X_{\text{future}} \in \mathbb{R}^{n \times t_{\text{future}}}$: 예측할 미래 $t_{\text{future}}$ 타임스텝의 시계열 행렬
> - $F(\cdot)$: 학습할 예측 모델

---

### 🟢 제안하는 방법 (수식 포함)

**[p.4~6, Section 4, 4.1, 4.2]**

#### (1) 시간 혼합기 (Temporal Mixer) 서브블록

$$F_{\text{temp}}(X) := \text{ComplexLinear}(\text{ReLU}(X)) $$

> - $X \in \mathbb{R}^{n \times t_{\text{past}}}$: 입력 시계열 행렬
> - $\text{ReLU}(\cdot)$: 비선형 활성화 함수 (음수 → 0)
> - $\text{ComplexLinear}(\cdot)$: 복소 선형 레이어 (주파수 도메인 처리)

#### (2) 공간 혼합기 (Spatial Mixer) 서브블록

$$F_{\text{sp}}(X) := \text{Linear}(\text{ReLU}(\text{RandProject}(\text{ReLU}(X^T))))^T \tag{2}$$

> - $X^T$: 행렬 전치(Transpose), 노드 차원에 대해 선형 연산 적용 가능하게 함
> - $\text{RandProject}(\cdot)$: 가중치 고정 랜덤 투영 레이어, $\mathbb{R}^n \rightarrow \mathbb{R}^{n_{\text{rand}}}$
> - $n_{\text{rand}}$: 랜덤 투영 뉴런 수 (하이퍼파라미터), 기본값 $= \sqrt{n}$
> - $\text{Linear}(\cdot)$: 학습 가능한 선형 레이어, $\mathbb{R}^{n_{\text{rand}}} \rightarrow \mathbb{R}^n$ 로 복원

#### (3) 믹서 블록 전체 연산

$$\text{Mixer}(X) = F_{\text{sp}}(F_{\text{temp}}(X) + X) + F_{\text{temp}}(X) + X \tag{3}$$

#### (4) 간략화된 표현 (앙상블 해석을 위해)

$G(\cdot)$를 다음과 같이 정의하면:

$$G(X) := F_{\text{sp}}(F_{\text{temp}}(X) + X) + F_{\text{temp}}(X) \tag{4}$$

$$\text{Mixer}(X) = G(X) + X \tag{5}$$

> - 수식 (5)에서 $G(X)$: 가중치 경로(Weighted Path)의 출력
> - $X$: 항등 매핑(Identity Mapping)으로 그대로 더해지는 입력 — 앙상블에서 기본 학습자 역할

#### (5) 앙상블 해석 (3개 블록 예시)

블록이 3개일 때, 최종 출력 $Y$는 출력 선형 레이어 $D(\cdot)$를 통해:

$$Y = D(X + H_1 + H_2 + H_3) = D(X) + D(H_1) + D(H_2) + D(H_3)$$

$$= Y_0 + Y_1 + Y_2 + Y_3$$

> - $H_i = G_i(X + H_1 + \cdots + H_{i-1})$: $i$번째 블록의 가중치 경로 출력
> - $Y_i = D(H_i)$: 각 블록의 개별 예측값 → 이를 합산하여 최종 예측

#### (6) 복소 선형 레이어 연산

$$\text{Output} = (W_{\text{real}}x_{\text{real}} - W_{\text{imag}}x_{\text{imag}}) + i(W_{\text{real}}x_{\text{imag}} + W_{\text{imag}}x_{\text{real}}) \tag{6}$$

> - $x_{\text{real}}, x_{\text{imag}}$: FFT 후 입력의 실수부/허수부
> - $W_{\text{real}}, W_{\text{imag}}$: 복소 가중치의 실수부/허수부

> 💡 **용어 설명**
> - **항등 매핑(Identity Mapping)**: 입력을 변환 없이 그대로 출력에 더하는 연결. ResNet의 Skip Connection과 동일 개념.
> - **사전 활성화(Pre-Activation)**: He et al. (2016)이 제안한 설계로, 가중치 레이어 이전에 활성화 함수를 배치하여 항등 매핑 경로가 완전히 보존되도록 함.
> - **FFT (Fast Fourier Transform)**: 시간 도메인 신호를 주파수 도메인으로 변환하는 알고리즘.
> - **랜덤 투영(Random Projection)**: 고차원 벡터를 무작위로 초기화된 행렬과 곱하여 저차원으로 변환. 가중치는 학습되지 않고 고정된다.

---

### 🔵 모델 구조

**[p.4, Section 4, Fig. 2, Fig. 3]**

```
입력: X_past ∈ ℝ^(n × t_past)
    ↓
[Mixer Block × n_block]
    각 블록 내부:
    ┌─────────────────────────────────────────────────────────┐
    │  Temporal Mixer:                                        │
    │  ReLU → ComplexLinear(t_past→t_past) → (+identity)     │
    │                                                         │
    │  Spatial Mixer:                                         │
    │  Transpose → ReLU → RandProject(n→n_rand)              │
    │  → ReLU → Linear(n_rand→n) → Transpose → (+identity)   │
    └─────────────────────────────────────────────────────────┘
    ↓
[출력 선형 레이어: t_past → t_future]
    ↓
출력: X_future ∈ ℝ^(n × t_future)
```

**주요 하이퍼파라미터**:
- $n_{\text{block}} = 8$ (기본값)
- $n_{\text{rand}} = m_{\text{neuron}} \sqrt{n}$, $m_{\text{neuron}} = 1.0$ (기본값)
- 손실 함수: MAE (Mean Absolute Error)
- 최적화: AdamW

---

### 🟡 성능 향상

**[p.7~9, Table 1, Fig. 7]**

| 데이터셋 | RPMixer Average MAE | 차선 방법 | 개선율 (MAE) |
|---------|---------------------|-----------|------------|
| SD | 16.90 | D²STGNN: 17.38 | ~2.8% |
| GBA | 19.06 | D²STGNN: 20.71 | ~8.0% |
| GLA | 18.46 | GWNET: 21.23 | ~13.1% |
| CA | 17.50 | GWNET: 21.08 | ~17.0% |

> ⚠️ **TSMixer의 GLA/CA RMSE 이상값**: TSMixer의 GLA RMSE = 207.68, CA RMSE = 90.20으로 비정상적으로 높음. 저자들은 TSMixer가 MAE 손실에만 과적합되었기 때문으로 설명하지만, 이는 통계적으로 취약한 주장임 (섹션 5 참조).

---

### 🔴 한계점

1. **도메인 한정성**: 실험이 교통 데이터(LargeST)에만 집중되어 있어 다른 시공간 도메인(기상, 에너지 등)으로의 일반화 검증 부족
2. **비주기적 시계열 처리**: 복소 선형 레이어가 비주기적 데이터에서 최적이 아닐 수 있음을 저자 스스로 인정 (p.4)
3. **랜덤 초기화 의존성**: 랜덤 시드에 따른 성능 분산 분석이 미흡
4. **장기 예측 한계**: ETT 등 장기 예측 일부 데이터셋(ETTh1, ETTm1)에서 RPMixer가 TSMixer/PatchTST보다 성능이 낮음 (Table 3)
5. **그래프 구조 미활용**: 명시적 그래프 정보(인접 행렬 $A$)를 활용하지 않으므로 공간 관계의 정밀한 모델링에 한계

---

## 3. 주장별 페이지/Figure/Table 위치

| 주장 | 위치 |
|------|------|
| 랜덤 투영으로 블록 출력 다양성 증가 | p.2, Fig. 1; p.9, Fig. 8 |
| 앙상블 해석 수식 | p.5~6, 수식 (1)~(5), Fig. 5, Fig. 6 |
| RPMixer 전체 구조 | p.4, Fig. 2, Fig. 3 |
| 복소 선형 레이어 설계 | p.11, Appendix A.1, Fig. 10, Fig. 11, 수식 (6) |
| Johnson-Lindenstrauss 패턴 보존 | p.5, Fig. 4 |
| 벤치마크 성능 비교 | p.8, Table 1 |
| 절제 연구 (Ablation Study) | p.9, Fig. 7, Fig. 13 |
| 하이퍼파라미터 민감도 분석 | p.10, Fig. 9a, 9b |
| 장기 시계열 예측 결과 | p.14, Table 3 |
| 상관-오류 다이어그램 | p.9, Fig. 8; p.13, Fig. 12 |

---

## 4. 저자 직접 보고 결과 vs. 해석 분리

### 📌 저자가 직접 보고한 결과

**연구 주제**:
- 고차원 시공간 시계열 예측에서 랜덤 투영을 통한 앙상블 다양성 강화

**방법**:

$$\text{Mixer}(X) = G(X) + X, \quad G(X) := F_{\text{sp}}(F_{\text{temp}}(X) + X) + F_{\text{temp}}(X)$$

$$n_{\text{rand}} = m_{\text{neuron}}\sqrt{n}$$

**저자 직접 보고 수치** (Table 1):
- GLA 데이터: RPMixer Average MAE = **18.46**, 차선책 GWNET = 21.23
- CA 데이터: RPMixer Average MAE = **17.50**, 차선책 GWNET = 21.08
- 장기 예측(Table 3): t-test ($\alpha=0.05$) 기준 TSMixer, PatchTST와 동등

**절제 연구 결과** (Fig. 7):
- 사전 활성화 제거 시 가장 큰 성능 하락
- 랜덤 투영 제거 시 두 번째 성능 하락
- 주파수 도메인 처리 제거 시 가장 작은 성능 하락

---

### 💬 검토자(본 답변)의 해석

1. **랜덤 투영의 역할**: 저자는 랜덤 투영이 다양성을 증가시킨다고 주장하나, 이것이 **앙상블 이론에서의 다양성**과 동일한 것인지는 엄밀한 이론적 증명 없이 상관-오류 다이어그램으로만 간접 지지된다. 이는 상관관계(Correlation)와 인과관계(Causality)를 혼동할 소지가 있다.

2. **TSMixer의 이상 RMSE**: GLA와 CA에서 TSMixer의 RMSE가 각각 207.68, 90.20으로 비정상적으로 높은 것은 저자 주장처럼 MAE 과적합 때문일 수 있으나, **학습률 불안정 또는 수치 폭발(Gradient Explosion)**의 가능성도 배제할 수 없다. 이에 대한 추가 분석이 없는 것은 한계이다.

3. **스케일 우위의 원인**: RPMixer가 대규모 데이터에서 더 큰 이점을 보이는 것은 랜덤 투영의 정규화 효과($\mathbb{R}^n \rightarrow \mathbb{R}^{\sqrt{n}}$로의 차원 축소)가 대규모 데이터일수록 과적합 방지에 더 효과적으로 작용하기 때문으로 해석된다.

4. **ETT 데이터 혼재 성능**: ETTh2, ETTm2에서는 RPMixer가 TSMixer를 크게 능가하나(예: ETTm2 Horizon 96, MSE: 0.111 vs. 0.163), ETTh1, ETTm1에서는 반대 결과를 보인다. 이는 데이터의 주기성(Periodicity) 특성에 따라 복소 레이어의 효과가 달라지는 것으로 해석 가능하다.

---

## 5. 통계적으로 취약한 부분 및 비교 불가능한 수치

| 항목 | 문제점 | 위치 |
|------|--------|------|
| ⚠️ TSMixer RMSE (GLA: 224.82~229.86) | 비정상적 이상치(Outlier). 원인 분석 미제공. 비교 불공정 소지 | Table 1, GLA 행 |
| ⚠️ TSMixer RMSE (CA: 73.98~106.28) | 동일 문제. RPMixer와의 RMSE 비교가 의미 있는지 불분명 | Table 1, CA 행 |
| ⚠️ 랜덤 시드 의존성 미보고 | 랜덤 투영 가중치 초기화 랜덤성에 대한 분산/표준편차 미제공 | Section 4.1 |
| ⚠️ 장기 예측 t-test 방법론 불명확 | "t-test with α=0.05"라 명시하나, 정확한 검정 방법(paired/unpaired) 및 샘플 수 미기재 | Appendix A.6 |
| ⚠️ 일부 베이스라인의 CA 제외 | ASTGCN, AGCRN, DSTAGNN 등은 OOM으로 CA 미포함. 공정한 전체 비교 불가 | Table 1, CA 섹션 |
| ⚠️ Ablation Study 통계 검증 부재 | Fig. 7의 절제 연구 결과에 대한 통계적 유의성 검증(p-value 등) 없음 | Fig. 7, Section 5.4 |
| ⚠️ ETT 데이터에서 성능 혼재 | ETTh1, ETTm1에서 RPMixer < TSMixer이나, 이를 충분히 설명하지 않음 | Table 3 |

---

## 6. 문서가 답하지 않는 질문

| # | 미답 질문 |
|---|-----------|
| Q1 | 랜덤 투영 가중치의 분포(예: 정규분포, 직교 행렬 등) 선택이 성능에 미치는 영향은? |
| Q2 | 서로 다른 랜덤 시드에 따른 성능 분산(Variance)은 얼마나 되는가? |
| Q3 | 비주기적 시계열 데이터에서 복소 선형 레이어를 제거하면 성능이 어떻게 변하는가? |
| Q4 | TSMixer의 GLA/CA RMSE 이상치의 정확한 원인은 무엇인가? |
| Q5 | 학습 시간(Training Time) 및 추론 시간(Inference Time) 비교가 없다. |
| Q6 | 랜덤 투영 차원 $n_{\text{rand}} = \sqrt{n}$ 공식의 이론적 근거는 무엇인가? (실험적 선택인지?) |
| Q7 | 그래프 인접 행렬 $A$를 명시적으로 활용하면 추가 성능 향상이 가능한가? |
| Q8 | 교통 데이터 외 다른 도메인(기상, 에너지, 의료)에서의 성능은 어떠한가? |
| Q9 | 블록별 랜덤 투영의 다양성을 정량화하는 이론적 지표가 존재하는가? |
| Q10 | 앙상블 해석에서 각 기본 학습자(Base Learner)의 예측 기여도(Contribution)를 어떻게 분석할 수 있는가? |

---

## 7. 가장 중요한 그림 5개 해석

### 📊 Figure 1 (p.2): 블록별 중간 출력 비교

**해석**: 랜덤 투영 적용 모델("proposed")과 미적용 모델("w/o random")의 1번, 3번, 5번, 7번, 최종(8번) 블록 출력을 3개 샘플 노드에서 비교한다. 적용 모델의 블록 출력들은 서로 형태가 뚜렷이 다른 반면(고다양성), 미적용 모델의 출력들은 유사한 패턴을 반복(저다양성)한다. 최종 출력이 ground truth에 더 가깝게 수렴하는 것을 시각적으로 확인할 수 있다. 이 그림은 논문의 핵심 직관을 가장 직접적으로 보여주는 증거이다.

---

### 📊 Figure 3 (p.4): 믹서 블록 상세 구조

**해석**: 하나의 믹서 블록 내부 구조를 보여주며, **빨간 선**으로 표시된 항등 매핑(Identity Mapping) 연결이 핵심이다. 시간 혼합기(Temporal Mixer)와 공간 혼합기(Spatial Mixer)가 순차적으로 배치되며, 공간 혼합기 내부에서 랜덤 투영이 $\mathbb{R}^n \rightarrow \mathbb{R}^{n_{\text{rand}}}$으로 차원을 축소한 후 다시 $\mathbb{R}^{n_{\text{rand}}} \rightarrow \mathbb{R}^n$으로 복원한다. 사전 활성화 설계에 따라 ReLU가 가중치 연산 앞에 위치한다.

---

### 📊 Figure 7 (p.9): 절제 연구 결과

**해석**: 4개 데이터셋에 걸쳐 MAE, RMSE, MAPE 지표로 제안 모델(proposed), Fourier 제거(w/o Fourier), 랜덤 투영 제거(w/o random), 사전 활성화 제거(w/o pre-act) 4개 변형을 비교한다. **w/o pre-act**가 가장 큰 성능 저하를 보이며, **w/o random**이 두 번째로 크다. 이는 사전 활성화 → 항등 매핑 → 앙상블 해석의 연쇄가 성립함을 실증한다. 주파수 도메인(w/o Fourier)은 상대적으로 작은 영향을 보인다.

---

### 📊 Figure 8 (p.9): 상관-오류 다이어그램 (Correlation-Error Diagram)

**해석**: 각 점은 모델 내 두 블록 쌍(Base Learner Pair)을 나타내며, X축은 두 블록 출력 간 피어슨 상관계수(다양성의 역수), Y축은 해당 쌍의 평균 MAE 오류이다. **제안 모델(초록)의 점들은 더 낮은 상관계수(높은 다양성) 쪽에 분포**하고, 미적용 모델(검정)의 점들은 상관계수 1.0 근처에 집중된다. 이는 랜덤 투영이 실제로 블록 간 다양성을 수치적으로 증가시킴을 정량적으로 확인하는 핵심 증거이다. 단, 이 다양성이 성능 향상의 **직접적 원인**임을 증명하지는 않는다는 점에 유의해야 한다.

---

### 📊 Figure 9 (p.10): 하이퍼파라미터 민감도 분석

**해석**: (a) 블록 수: 2개 → 성능 저하 심각, 8개 → 최적, 16개 → 미미한 추가 개선 (계산 비용 2배 증가 대비 불합리). (b) 뉴런 인자($m_{\text{neuron}}$): 1.0이 대부분 최적이며, 2.0은 일부 향상이 있으나 미미함. 검증 데이터와 테스트 데이터의 경향이 일치하므로 검증 세트로 하이퍼파라미터 튜닝이 가능하다. $n_{\text{rand}} = \sqrt{n}$ 설정은 노드 수에 따라 자동 조정되어 크로스-데이터셋 일반화를 지원한다.

---

## 8. 결론: 시사점, 후속 연구 계획 및 추가 연구 방향

### 8-1. 저자 제시 시사점 및 후속 연구 계획

**[p.10, Section 6]**

**저자 시사점**:
- 랜덤 투영이 시공간 예측에서 앙상블 다양성을 효과적으로 증가시키며 성능 향상에 기여함
- 사전 활성화 설계를 통한 항등 매핑 강화가 핵심 설계 요소임
- 메모리 선형 확장성으로 8,600개 이상의 대규모 노드에도 적용 가능함
- 교통 흐름 예측 등 실세계 응용에서 더 정확하고 신뢰성 있는 예측 가능

**저자 후속 연구 계획**:
> "We plan to investigate the potential of applying **time series foundation models** [46] for tackling spatial-temporal forecasting problems." (p.10)

즉, 대규모 사전학습된 시계열 기반 모델(Foundation Model)을 시공간 예측에 적용하는 방향을 제시한다.

---

### 모델의 일반화 성능 향상 가능성

**[Appendix A.6, Table 3; Section 5.3]**

현재 RPMixer의 일반화와 관련된 핵심 관찰:

1. **크로스-도메인 일반화**: 장기 시계열 예측 실험(Appendix A.6)에서 ETTh2, ETTm2에서는 TSMixer/PatchTST 대비 탁월한 성능을 보이나, ETTh1, ETTm1에서는 열등하다. 이는 **데이터의 주기성 여부에 따라 복소 레이어의 기여가 상이함**을 시사한다. 일반화를 위해서는 데이터의 주기성을 자동 감지하는 적응적 메커니즘이 필요하다.

2. **랜덤 투영의 정규화 효과**: $\mathbb{R}^n \rightarrow \mathbb{R}^{\sqrt{n}}$으로의 차원 축소는 Dropout과 유사한 암묵적 정규화(Implicit Regularization) 효과를 제공할 수 있다. 이는 대규모 데이터에서 더 강한 일반화 효과로 나타나는 것으로 해석된다.

3. **하이퍼파라미터 안정성**: Fig. 9에서 검증 세트와 테스트 세트의 경향이 일치하여, 검증 기반 하이퍼파라미터 선택이 새로운 데이터에서도 일반화됨을 보여준다. 특히 $n_{\text{rand}} = \sqrt{n}$ 공식은 데이터셋 크기에 따른 자동 조정을 가능하게 하는 **스케일 불변적 설계**이다.

**일반화 향상을 위한 추가 제안**:

- **적응적 랜덤 투영(Adaptive Random Projection)**: 고정 가중치 대신 소프트맥스 어텐션으로 랜덤 행렬을 가중합하는 방식 도입
- **배치 정규화(Batch Normalization) 또는 레이어 정규화(Layer Normalization) 추가**: 학습 안정성 강화
- **메타러닝(Meta-Learning) 기반 하이퍼파라미터 자동 선택**: $m_{\text{neuron}}$, $n_{\text{block}}$ 등의 자동 최적화

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 아래 내용은 본 논문의 참고문헌 및 공개된 연구 동향을 기반으로 정리한 것이며, 2024년 이후 최신 논문의 정확한 수치는 직접 원문 확인이 필요합니다.

| 모델 | 연도 | 유형 | 핵심 아이디어 | RPMixer와 비교 |
|------|------|------|------------|--------------|
| **Informer** [58] | 2021 | Transformer | ProbSparse Attention, 긴 시퀀스 효율화 | 단순 선형 모델에도 열등함이 입증됨 [56]. RPMixer보다 계산 복잡도 높음 |
| **Autoformer** [42] | 2021 | Transformer | Auto-Correlation + 시계열 분해 | 동일하게 단순 모델 대비 열등 [56] |
| **PatchTST** [28] | 2022 | Transformer | 시계열을 패치(Patch) 단위로 처리 | 장기 예측에서 RPMixer와 동등 (t-test) |
| **TSMixer** [6] | 2023 | MLP-Mixer | All-MLP 시간·특성 혼합 | RPMixer의 직접 기반 모델. 대규모 데이터에서 RPMixer에 열등 |
| **D²STGNN** [34] | 2022 | Graph | 동적 그래프 + 확산/내재 정보 분리 | 소규모(SD, GBA)에서 경쟁력 있으나 대규모(GLA, CA)에 확장 불가 |
| **DGCRN** [18] | 2023 | Graph+RNN | 동적 인접 행렬 생성 | 대규모 데이터 확장성 한계 동일 |
| **TimesNet** (2023, 미인용) | 2023 | CNN | 2D 변환으로 시계열 처리 | 비교 미수행. 추가 비교 필요 |
| **iTransformer** (2024, 미인용) | 2024 | Transformer | 변수 차원에서 어텐션 적용 | 비교 미수행. 특히 다변량 예측에서 강점 |

**이 논문이 앞으로의 연구에 미치는 영향**:

1. **설계 패러다임 변화**: 복잡한 그래프 연산 없이 단순 랜덤 투영만으로도 대규모 시공간 예측이 가능함을 보여줌. 이는 "단순성의 힘(Power of Simplicity)" 트렌드를 강화한다.

2. **앙상블 해석의 실용화**: Veit et al. [40]의 이론적 앙상블 해석을 실제 설계 원칙으로 활용한 최초 시계열 예측 연구로, 후속 연구에서 이 해석을 더 정밀하게 활용하는 방법론 개발이 기대된다.

3. **랜덤 요소의 재평가**: 학습 불가능한 랜덤 고정 가중치가 오히려 다양성과 일반화에 기여한다는 발견은 Lottery Ticket Hypothesis, Random Forest 등의 아이디어와 연계하여 새로운 연구 방향을 제시한다.

**앞으로 연구 시 고려할 점**:

| 고려 사항 | 구체적 방향 |
|-----------|------------|
| **비교 공정성** | 동일 하드웨어/시간 조건에서 그래프 모델과 추론 속도 비교 필수 |
| **도메인 확장** | 에너지, 기상, 의료 시계열 데이터로 검증 확대 |
| **이론적 정당화** | 랜덤 투영 차원 $\sqrt{n}$ 선택의 이론적 근거 제시 |
| **동적 랜덤 투영** | 데이터 분포에 적응하는 학습 가능한 랜덤 투영 탐색 |
| **Foundation Model 통합** | 저자 후속 계획대로 대규모 사전학습 모델과 RPMixer 결합 |
| **해석 가능성** | 각 블록의 예측 기여도(Attribution) 분석 도구 개발 |
| **불확실성 정량화** | 예측 신뢰 구간(Prediction Interval) 제공 기능 추가 |

---

## 참고 자료 (본 답변에서 인용한 주요 자료)

본 답변은 다음 자료를 참고하였습니다:

1. **주 논문**: Yeh, C.-C. M. et al. (2024). *RPMixer: Shaking Up Time Series Forecasting with Random Projections for Large Spatial-Temporal Data.* Proceedings of the 30th ACM SIGKDD. (arXiv:2402.10487v3)

2. **인용 논문 [6]**: Chen, S.-A. et al. (2023). *TSMixer: An All-MLP Architecture for Time Series Forecasting.* arXiv:2303.06053

3. **인용 논문 [12]**: He, K. et al. (2016). *Identity Mappings in Deep Residual Networks.* ECCV 2016.

4. **인용 논문 [15]**: Johnson, W. B. (1984). *Extensions of Lipschitz Mapping into Hilbert Space.* Conference on Modern Analysis and Probability.

5. **인용 논문 [23]**: Liu, X. et al. (2023). *LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting.* arXiv:2306.08259

6. **인용 논문 [28]**: Nie, Y. et al. (2022). *A Time Series is Worth 64 Words: Long-Term Forecasting with Transformers (PatchTST).* arXiv:2211.14730

7. **인용 논문 [40]**: Veit, A. et al. (2016). *Residual Networks Behave Like Ensembles of Relatively Shallow Networks.* NeurIPS 29.

8. **인용 논문 [56]**: Zeng, A. et al. (2023). *Are Transformers Effective for Time Series Forecasting?* AAAI 2023.
