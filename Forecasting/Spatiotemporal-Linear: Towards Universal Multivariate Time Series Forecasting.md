# Spatiotemporal-Linear: Towards Universal Multivariate Time Series Forecasting

> **참고 자료**: Zuo, A., Zhang, H., Li, Z., & Zheng, C. (2023). *Spatiotemporal-Linear: Towards Universal Multivariate Time Series Forecasting*. arXiv:2312.14869v1 [cs.LG]
> 
> **주의**: 본 논문은 2023년 12월 arXiv에 게재된 프리프린트(preprint)이며, 동료 심사(peer review)를 거치지 않은 논문입니다. 이하 분석은 해당 논문 내용에 기반하며, 확인 불가한 주장은 명시적으로 표시합니다.

---

## 1. Executive Summary (10문장 이내)

1. 다변량 시계열 예측(TSF) 분야에서 Transformer 기반 모델들은 반복적 다단계 예측으로 인한 **오차 누적 문제**를 내재하고 있다.
2. 이를 해결하기 위해 제안된 LTSF-Linear 모델은 직접 예측(Direct Forecasting) 방식으로 우수한 성능을 보이지만, 공간적·시간적 정보를 충분히 활용하지 못한다.
3. 특히 관측 데이터가 희소한 상황($T \ll \tau$)에서 Linear 계열 모델은 Transformer 대비 평균 **48% MSE 성능 저하**를 보인다.
4. 이 논문은 이러한 한계를 극복하기 위해 **SpatioTemporal-Linear (STL)** 프레임워크를 제안한다.
5. STL은 세 가지 경로(Core, Temporal, Spatial Route)를 병렬로 결합하는 3-경로 아키텍처를 채택한다.
6. Core Route는 잔차 선형(Res-L) 레이어를, Temporal Route는 위치 임베딩과 동적 인코더-디코더를, Spatial Route는 공간 어텐션(SpatialAttn) 메커니즘을 활용한다.
7. Electricity, ETTh1, ETTm1, Weather, JAAD 등 5개 데이터셋에 걸친 실험에서 STL은 데이터 풍부 및 희소 상황 모두에서 기존 모델들을 능가한다.
8. 정보 희소 환경의 실세계 데이터(JAAD)에서 DLinear 대비 **55% MSE 향상**을 달성했다.
9. Ablation Study를 통해 각 경로가 예측 정확도 향상에 기여함을 실증적으로 검증했다.
10. 저자들은 STL이 데이터 희소 예측 시나리오를 고려한 **범용 TSF 패러다임**으로 자리매김할 수 있음을 주장한다.

### 1-1. 연구의 목적과 필요성

**목적**: 기존 Linear 기반 모델의 두 가지 핵심 결함—(1) 공간·시간 정보 미통합, (2) 데이터 희소 상황에서의 성능 저하—을 동시에 해결하는 범용적 다변량 TSF 모델 개발

**필요성**:

| 문제 상황 | 기존 모델의 한계 | 실세계 예시 |
|-----------|----------------|------------|
| 장기 예측 (데이터 풍부) | Transformer: 오차 누적 | 전력 수요 예측 |
| 단기 예측 (데이터 희소) | LTSF-Linear: 공간·시간 정보 부재 | 자율주행 궤적 예측 |
| 희귀 사건 모델링 | 모든 기존 모델: 일반화 부족 | 희귀 질병 진행 예측 |

> 💡 **오차 누적(Error Accumulation)**: 모델이 미래값을 한 번에 예측하지 않고, 이전 예측값을 다음 입력으로 사용하여 순차적으로 예측할 때, 초기 오차가 점점 증폭되는 현상을 말합니다.

---

## 2. 핵심 주장과 근거 (표)

| # | 핵심 주장 | 근거 | 위치 |
|---|----------|------|------|
| 1 | Linear 모델은 단기 예측·데이터 희소 시나리오에서 성능이 급격히 저하됨 | DLinear가 T=48 고정 시 Transformer 대비 평균 48% MSE 열화 | p.8, Table 2 |
| 2 | 단순 Linear Layer는 시공간 정보를 부분적으로만 포착함 ( $\beta(s', t')$ ) | 선형 레이어가 미래 채널 간 상호의존성을 포착 불가임을 수식으로 증명 | p.13, Appendix C |
| 3 | STL의 3-경로 설계가 더 완전한 시공간 회귀 $\beta(s, t)$를 달성함 | Ablation Study에서 경로 추가 시 단조 성능 향상 확인 | p.9, Fig.4 |
| 4 | 정보 풍부 시나리오에서 STL이 DLinear 대비 최대 14.3% MSE 향상 | Table 1에서 STL이 일관되게 Top 2 달성 | p.8, Table 1 |
| 5 | 정보 희소 시나리오에서 STL이 DLinear 대비 최대 55% MSE 향상 | JAAD 데이터셋에서 STL MSE=0.277 vs DLinear MSE=0.620 (T=10) | p.9, Table 2 |
| 6 | Temporal Route는 T ≤ θT일 때만 활성화함으로써 노이즈 과적합을 방지 | 관측 길이가 충분하면 주기성이 내재되므로 날짜 임베딩이 중복됨 | p.6, §4.2 |

---

## 2-1. 상세 분석

### 해결하고자 하는 문제

```
[문제 1] LTSF-Linear의 시공간 정보 미활용
  → 각 변수를 독립적으로 처리 (채널 간 상호작용 무시)
  → 날짜·시간 정보 미활용

[문제 2] 데이터 희소 시나리오에서의 성능 병목
  → T ≪ τ 상황에서 Linear 모델은 Transformer보다 열등

[문제 3] 오차 누적 문제 (Transformer 계열)
  → 반복적 다단계 예측 방식의 구조적 한계
```

### 제안하는 방법 (수식 포함)

#### [전체 TSF 문제 정의] (p.3)

$$X_{T+1:T+\tau} = X_{1:T}\beta(s,t) + w(s,t) + \epsilon(s,t)$$

| 기호 | 의미 |
|------|------|
| $X_{1:T} \in \mathbb{R}^{T \times C}$ | 관측 시퀀스 행렬 ($T$: 관측 길이, $C$: 변수 수) |
| $X_{T+1:T+\tau} \in \mathbb{R}^{\tau \times C}$ | 예측 목표 행렬 ($\tau$: 예측 길이) |
| $\beta(s,t)$ | 시공간 회귀 계수 ($s$: 공간, $t$: 시간 차원) |
| $w(s,t)$ | 관측 불가한 시공간 변동성 (예: 런던-파리 증시 동조 현상) |
| $\epsilon(s,t)$ | 시공간 패턴이 없는 순수 노이즈 |

> 💡 **회귀 계수(Regression Coefficient, $\beta$)**: 입력 변수가 출력에 미치는 영향의 크기를 나타내는 값입니다. $\beta(s,t)$는 공간($s$)과 시간($t$) 정보를 모두 포함하는 계수입니다.

#### [최종 예측: 세 경로의 합산] (p.5)

$$X_{T+1:T+\tau} = X_{T+1:T+\tau}^{core} + X_{T+1:T+\tau}^{temp} + X_{T+1:T+\tau}^{spat}$$

> 💡 **Skip Connection (잔차 연결)**: 레이어의 입력을 출력에 직접 더해주는 연결 방식으로, 정보 손실 방지와 기울기 소실 문제 완화에 효과적입니다. ResNet에서 유래했습니다.

#### [Core Route: Residual Linear (Res-L)] (p.5)

$$X_{T+1:T+\tau}^{core} = L_1(X_{1:T}) + L_3 \circ g(L_2(X_{1:T}))$$

| 기호 | 의미 |
|------|------|
| $L_1, L_2, L_3$ | 단순 선형 레이어 |
| $g(\cdot)$ | 활성화 함수 (SiLU 또는 LeakyReLU, 데이터셋에 따라 다름) |
| $\circ$ | 함수 합성 연산자 |
| $L_1(X_{1:T})$ | 직접 선형 스킵 연결 ($T \to \tau$) |
| $L_3 \circ g(L_2(X_{1:T}))$ | 비선형 변환 경로 ($T \to h=\tau \to \tau$) |

> 💡 **SiLU (Sigmoid Linear Unit)**: $\text{SiLU}(x) = x \cdot \sigma(x)$로 정의되는 활성화 함수로, ReLU보다 부드러운 비선형성을 제공합니다.

#### [Temporal Route] (p.6)

$$X_{T+1:T+\tau}^{temp} = \text{DD} \circ \text{Res-L} \circ \text{Res-L} \circ \text{DE}(\text{PE}(X_{1:T}))$$

**위치 임베딩(Positional Embedding, PE)**:

$$\text{PE}(p, 2i) = \sin\!\left(p \times \frac{1}{10000^{\frac{2i}{C}}}\right)$$

$$\text{PE}(p, 2i+1) = \cos\!\left(p \times \frac{1}{10000^{\frac{2i}{C}}}\right)$$

| 기호 | 의미 |
|------|------|
| $p$ | 시간 스텝 위치 인덱스 |
| $i$ | 채널 인덱스 ($0 \leq i \leq \lfloor\max(\frac{C}{2}, \frac{C-1}{2})\rfloor$) |
| $C$ | 변수(채널) 수 |
| $\text{PE}$ | 위치 임베딩 레이어 |
| $\text{DE}$ | 동적 인코더 (Dynamic Encoder) |
| $\text{DD}$ | 동적 디코더 (Dynamic Decoder) |

> 💡 **Positional Embedding (위치 임베딩)**: 시퀀스 내 각 위치(시간 스텝)의 순서 정보를 신호로 인코딩하는 기법입니다. 사인/코사인 함수를 사용하면 다양한 주기의 패턴을 포착할 수 있습니다.

**날짜-시간 임베딩(DateTime Embedding)**:

$$\text{datetime feature} \in \mathbb{R}^{T \times 1} = L_{\text{reducer}}([e_{\text{date}}; e_{\text{weekday}}; e_{\text{hour}}])$$

**동적 인코더/디코더(Dynamic Encoder/Decoder)**:

$$X' = X_{1:T}^{pe} + m \times \text{normalize}(\text{datetime features})$$

$$\text{normalize}(x) = \frac{x - \min(x)}{\max(x) - \min(x)}$$

| 기호 | 의미 |
|------|------|
| $m$ | 날짜-시간 임베딩의 영향력을 조절하는 학습 가능한 게이팅 파라미터 |
| $X_{1:T}^{pe}$ | 위치 임베딩이 적용된 입력 시퀀스 |
| $[e_{\text{date}}; e_{\text{weekday}}; e_{\text{hour}}]$ | 날짜, 요일, 시간 임베딩의 연결(concatenation) |
| $\theta_T$ | 시간 경로 활성화 임계값 (관측 길이가 이보다 작을 때만 활성화) |

> 💡 **Gating Mechanism (게이팅 메커니즘)**: 정보의 흐름을 선택적으로 제어하는 구조입니다. 학습 가능한 파라미터 $m$이 날짜-시간 정보가 예측에 얼마나 반영될지를 자동으로 조절합니다.

#### [Spatial Route] (p.7)

$$X_{T+1:T+\tau}^{spat} = \text{Res-L} \circ \text{SpatialAttn}(\text{Res-L} \circ \text{Res-L}(\text{PE}(X_{1:T})))$$

**공간 어텐션(SpatialAttn)**:

$$\text{scores} = \tanh(X_{T+1:T+\tau}^{prel})$$

$$\text{interact scores} \in \mathbb{R}^{C \times C} = \text{scores} \in \mathbb{R}^{C \times \tau} \times \text{scores}^T \in \mathbb{R}^{\tau \times C}$$

$$X_{T+1:T+\tau}^{attn} = X_{T+1:T+\tau}^{prel} + \sum_{i=1}^{C} W_{[i,:]}^T \times X_{T+1:T+\tau}^i$$

| 기호 | 의미 |
|------|------|
| $X_{T+1:T+\tau}^{prel} \in \mathbb{R}^{C \times \tau}$ | Res-L을 통해 생성된 예비 예측값 |
| $\text{interact scores}[i,j]$ | $i$번째와 $j$번째 변수 간의 방향 유사성 및 영향 크기 |
| $W \in \mathbb{R}^{C \times C}$ | Softmax 정규화된 최종 어텐션 가중치 행렬 |
| $W_{[i,:]}^T \in \mathbb{R}^{C \times 1}$ | $i$번째 변수에 대한 어텐션 가중치 벡터 |
| $X_{T+1:T+\tau}^i \in \mathbb{R}^{1 \times \tau}$ | $i$번째 변수의 예비 예측 시퀀스 |

> 💡 **Spatial Attention (공간 어텐션)**: 여러 변수(채널) 간의 상관관계를 학습하는 메커니즘입니다. 예를 들어, 특정 지역의 기온이 다른 지역의 전력 사용량에 미치는 영향을 자동으로 학습합니다.

### 모델 구조 요약

```
입력: X[C, T] (C개 변수, T 시간 스텝)
         │
    ┌────┼────┐
    ▼    ▼    ▼
[Core] [Temporal] [Spatial]
Res-L  PE→DE→     PE→Res-L→
       Res-L→     Res-L→
       Res-L→     SpatialAttn→
       DD         Res-L
    │    │    │
    └────┼────┘
         ▼ (합산, Skip Connection)
출력: X[C, τ] (C개 변수, τ 미래 시간 스텝)
```

### 성능 향상 및 한계

| 구분 | 내용 |
|------|------|
| **성능 향상** | 정보 풍부: DLinear 대비 최대 14.3% MSE 향상 (Table 1) |
| **성능 향상** | 정보 희소(4개 데이터셋): DLinear 대비 최대 34% MSE 향상 (p.9) |
| **성능 향상** | JAAD(실세계 희소): DLinear 대비 55% MSE 향상 (Table 2) |
| **한계** | θT(임계값) 설정에 대한 체계적 방법론 부재 |
| **한계** | 비교 대상이 2023년 이전 모델에 한정 (PatchTST, TimesNet 등 미포함) |
| **한계** | 하이퍼파라미터가 데이터셋별로 크게 달라 범용성 한계 가능성 |
| **한계** | 단일 arXiv 논문으로 동료 심사 미완료 |

---

## 3. 각 주장에 페이지 및 Figure/Table 번호 표시

| 주장 | 근거 위치 |
|------|----------|
| Linear 모델이 데이터 희소 시 Transformer에 뒤처짐 | p.2, Fig.1(b); p.8, §5.3 |
| STL의 3-경로 설계 | p.3, Fig.2; p.5-7, §4.1-4.3 |
| Fragmentary regression coefficient 증명 | p.4, §3.2; p.13, Appendix C |
| 정보 풍부 시나리오 성능 | p.8, Table 1 |
| 정보 희소 시나리오 성능 | p.9, Table 2 |
| DLinear 대비 48% MSE 열화 (Linear 계열) | p.8, §5.3 |
| STL의 34% MSE 향상 (희소 시나리오) | p.9, §5.3 |
| JAAD 55% 향상 | p.9, §5.3; Table 2 |
| Ablation Study 결과 | p.9-10, Fig.4; Fig.5 |
| θT 임계값 기반 Temporal Route 조건부 활성화 | p.6, §4.2 |
| 위치 임베딩 수식 | p.6, §4.2 |
| 공간 어텐션 수식 | p.7-8, §4.3 |
| 다양한 관측 길이 실험 결과 | p.15, Fig.7; p.16-18, Table 5-9 |

---

## 4. 저자 직접 보고 결과 vs. 해석 분리

### 저자가 직접 보고한 결과

| 항목 | 저자 보고 내용 | 위치 |
|------|--------------|------|
| 정보 풍부 시나리오 | STL이 모든 데이터셋·예측 길이에서 Top 2 이상 달성; DLinear 대비 최대 14.3% MSE 향상 | Table 1, p.8 |
| 정보 희소 시나리오 | STL이 DLinear 대비 최대 34% MSE 향상 | Table 2, p.9 |
| JAAD 데이터셋 | DLinear MSE=0.620 (T=10) vs STL MSE=0.277; 55% 향상 | Table 2 |
| DLinear의 희소 시나리오 열화 | 평균 48% MSE 성능 저하 | p.8, §5.3 |
| Ablation Study | 경로를 추가할수록 1/MSE 단조 증가 | Fig.4 |

### 검토자(본 분석)의 해석

| 해석 항목 | 내용 |
|----------|------|
| ⚠️ "범용(Universal)" 주장의 한계 | 5개 데이터셋 기준이며, 금융·바이오·자연어 등 도메인 미포함. 범용성 주장은 과대 해석 가능성 있음 |
| ⚠️ Appendix C의 증명 수준 | 수학적 엄밀성보다는 직관적 논증에 가까움. 형식 증명(Formal Proof)으로 보기 어려움 |
| ⚠️ 14.3% 향상의 조건부 성격 | "최대(up to)" 값으로, 평균 향상률이 아님. 일부 조건에서는 차이가 미미함 (예: ETTh1, τ=192에서 STL=0.404 vs DLinear=0.404) |
| ⚠️ θT 임계값 선택 | 임계값 결정 방법이 명시적으로 제시되지 않아 재현성 문제 가능성 |
| ✅ 희소 시나리오에서의 강점 | JAAD 결과는 실제 자율주행 시나리오를 모사하므로 실용적 의의가 높음 |

---

## 5. 통계적으로 취약한 부분과 비교 불가능한 수치

| 유형 | 항목 | 문제점 |
|------|------|--------|
| ⚠️ **통계적 취약** | 성능 수치가 단일 시드(seed=2021) 기반 | 표준편차·신뢰구간 미보고. 통계적 유의성 검증 없음 |
| ⚠️ **통계적 취약** | "최대(up to) 14.3%, 34%, 55% 향상" 표현 | 최대값만 보고, 평균·중앙값 향상률 미제시 |
| ⚠️ **비교 불가** | JAAD 데이터셋 수치 (단위: $10^{-3}$) | 다른 데이터셋과 MSE 스케일이 다름. 직접 수치 비교 불가 |
| ⚠️ **비교 불가** | Tbest 방식의 정보 풍부 실험 | 모델마다 최적 T를 선택하므로, 완전히 공평한 비교가 아닐 수 있음 |
| ⚠️ **비교 대상 부재** | PatchTST, TimesNet, iTransformer 등 2023년 SOTA 모델 미포함 | 최신 경쟁 모델과의 비교 부재 |
| ⚠️ **소규모 데이터** | JAAD: 2800 trajectories, 4 variables | 극히 제한적인 데이터 크기로 일반화 결론 도출 주의 필요 |
| ⚠️ **하이퍼파라미터 민감도** | 데이터셋별 hidden_size, dropout, LR, 활성화 함수가 모두 다름 | 범용성 주장과 상충. 최적 하이퍼파라미터 탐색 과정 미공개 |

---

## 6. 논문이 답하지 않는 질문

| # | 미답 질문 |
|---|----------|
| 1 | **θT 임계값은 어떻게 결정하는가?** 데이터셋별 최적값 탐색 방법 미제시 |
| 2 | **계산 복잡도(Computational Complexity)는?** FLOPs, 파라미터 수, 추론 속도 비교 없음 |
| 3 | **PatchTST, TimesNet, iTransformer 등 2023년 SOTA 대비 성능은?** |
| 4 | **공간 어텐션이 실제로 의미 있는 변수 간 관계를 포착하는가?** 어텐션 가중치 시각화 없음 |
| 5 | **다른 도메인(금융, 바이오메디컬, 자연어)에서도 범용적으로 작동하는가?** |
| 6 | **학습 데이터 크기에 따른 성능 변화는?** 더 극단적인 희소 상황 실험 없음 |
| 7 | **Temporal Route의 날짜-시간 임베딩이 JAAD처럼 타임스탬프 없는 데이터에는 어떻게 작동하는가?** |
| 8 | **코드와 모델 가중치는 공개될 예정인가?** "Code will be made available"로만 언급 |
| 9 | **하이퍼파라미터 민감도 분석(Sensitivity Analysis)은?** |
| 10 | **멀티-스텝 vs 싱글-스텝 출력 전략의 상세 비교는?** |

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p.2): STL의 우월성 – 예측 길이 및 관측 길이 변화

**설명**: 왼쪽(a)은 관측 길이 T=48 고정 후 예측 길이 τ를 변화시킨 결과, 오른쪽(b)은 예측 길이 τ=336 고정 후 관측 길이 T를 변화시킨 결과 (Weather 데이터셋, y축: 1/MSE, 높을수록 좋음)

**핵심 해석**:
- **(a)**: STL(분홍)이 전 구간에서 가장 높은 1/MSE 유지. Linear(노랑)는 τ가 커질수록 급격히 하락하는 반면, STL은 완만한 하락세를 보임. Transformer 계열은 모든 구간에서 하위권
- **(b)**: T가 24~96 구간(데이터 희소)에서 Linear 계열의 성능 급락이 두드러짐. STL은 T가 줄어들어도 상대적으로 완만하게 감소하여 **데이터 희소 내성**이 검증됨
- **의미**: 이 그림이 논문의 핵심 동기를 가장 직관적으로 표현. 단, 단일 데이터셋(Weather) 기반이라는 점에서 일반화 주의 필요

### Figure 2 (p.3): 아키텍처 비교 – DLinear vs STL

**설명**: 위쪽은 DLinear의 단순 분해-선형 구조, 아래쪽은 STL의 3-경로 설계

**핵심 해석**:
- DLinear는 입력 → 분해 → FC → 합산의 단순 구조
- STL은 동일 입력이 세 경로로 병렬 처리됨: Core(좌), Temporal(중앙, 점선 박스=Res-L 인코더-디코더), Spatial(우)
- **Position Embedding**이 Temporal과 Spatial 두 경로 모두에 공급됨을 확인
- T < θ 조건부로 Temporal Route가 병합되는 구조가 명확히 표현됨
- **의의**: 복잡성 증가 없이 세 경로가 각자 다른 정보를 담당하는 분업 구조의 명확한 시각화

### Figure 3 (p.4): STL 핵심 모듈 – Res-L, SpatialAttn, Dynamic Encoder/Decoder

**설명**: 세 핵심 혁신 모듈의 내부 구조 시각화

**핵심 해석**:
- **Res-L(좌)**: Input→FC→Activation→FC의 비선형 경로 + Input→FC의 직접 경로를 합산. 잔차 구조로 정보 보존과 비선형 표현력을 동시에 확보
- **SpatialAttn(중앙)**: tanh 활성화 → 전치 행렬 곱셈 → Softmax → 원본에 가중합. $C \times C$ 상호작용 행렬이 채널 간 관계를 포착
- **Dynamic Encoder/Decoder(우)**: 날짜-시간 특성을 평탄화(Flatten)하여 학습 파라미터 $m$으로 조절 후 위치 임베딩 시퀀스와 결합. Res-L로 최종 출력 생성
- **의의**: 각 모듈이 서로 다른 유형의 정보(비선형성/공간관계/시간정보)를 담당함을 구조적으로 보여줌

> 💡 **Tanh 활성화**: 출력을 -1~1로 정규화하는 함수. 공간 어텐션에서 수치 안정성(Numerical Stability) 확보를 위해 사용됩니다.

### Figure 4 (p.10): Ablation Study 정량 결과 – Electricity & Weather

**설명**: T=48 고정, τ∈{24,36,...,336}에서 4가지 구성(Full STL / Core+Spatial / Core / Linear)의 1/MSE 비교

**핵심 해석**:
- **Electricity(좌)**: 모든 τ에서 Full STL(보라) > Core+Spatial(주황) > Core(파랑) > Linear(초록) 순서가 일관되게 유지됨. 경로 추가 시 **단조적 성능 향상**이 명확
- **Weather(우)**: 유사한 패턴이지만, 단기 예측(τ=24)에서 Core+Spatial이 Full STL에 매우 근접. Temporal Route의 기여가 단기보다 장기 예측에서 더 두드러짐
- **의의**: Temporal Route와 Spatial Route 각각이 성능에 기여함을 실증. 단, 두 경로의 상대적 기여도 분리 실험이 없어 독립적 효과 정량화의 한계 존재

### Figure 5 (p.10): Ablation Study 시각적 예측 비교 – τ=192

**설명**: T=48, τ=192에서 각 구성의 예측 시퀀스를 Ground Truth(회색)와 시각적으로 비교

**핵심 해석**:
- **Electricity(좌)**: Ground Truth의 진폭과 패턴을 Full STL이 가장 가깝게 추종. Linear(초록)는 진폭이 작고 패턴이 단순화됨. Spatial Route 추가(Core+Spatial)만으로도 상당한 개선 관찰 → **"coarse-refine" 전략의 효과성 확인**
- **Weather(우)**: Core Route 단독으로는 Ground Truth의 음수 영역을 양수로 잘못 예측. Spatial+Temporal Route 추가 후 방향이 교정됨 → **"Rectification(교정)" 효과** 명확히 시각화
- **의의**: 수치 지표만으로는 확인하기 어려운 "예측 방향 교정" 및 "패턴 정제" 기능을 직관적으로 보여주는 핵심 그림

---

## 8. 결론: 시사점, 후속 연구 계획 및 추가 방향

### 저자들이 제시한 시사점

1. **공간·시간 정보의 중요성**: 시계열 예측에서 내재 데이터 포인트만큼이나 시공간 정보 통합이 중요함
2. **데이터 희소 시나리오의 필요성**: 범용 TSF를 위해서는 데이터 희소 상황에 대한 연구가 필수적임

### 저자들의 후속 연구 계획

> ⚠️ **주의**: 논문 내에 명시적인 후속 연구 계획(Future Work)이 별도로 서술되어 있지 않습니다. 이 항목은 논문의 시사점(Impacts, p.11)에서 언급된 내용을 기반으로 도출된 것임을 밝힙니다.

- 더 다양한 실세계 애플리케이션(교통 궤적, 희귀 질병 진행)에 대한 검증 확장
- 데이터 희소 예측 시나리오에 대한 체계적 연구 촉구

### 8-1. 모델의 일반화 성능 향상 가능성

#### 현재 일반화의 한계

```
현재 실험 범위:
- 데이터셋: 5개 (Electricity, ETTh1, ETTm1, Weather, JAAD)
- 도메인: 전력, 날씨, 교통
- θT 임계값: 데이터셋마다 다른 수동 설정
- 하이퍼파라미터: 데이터셋별 개별 최적화
```

#### 일반화 성능 향상을 위한 구체적 방향

**① Adaptive θT 메커니즘**
- 현재 θT는 수동으로 설정되며 결정 기준이 불명확
- **제안**: 관측 데이터의 자기상관함수(ACF) 또는 FFT 기반으로 주기성을 자동 탐지하여 θT를 동적으로 결정하는 메커니즘 개발

**② Meta-Learning 기반 하이퍼파라미터 적응**
- hidden_size, dropout, LR이 데이터셋마다 크게 다름
- **제안**: Model-Agnostic Meta-Learning (MAML)이나 Hypernetwork를 통해 새로운 데이터셋에 빠르게 적응하는 구조 연구

> 💡 **Meta-Learning (메타 학습)**: "학습하는 방법을 학습"하는 접근법입니다. 적은 데이터로도 새로운 태스크에 빠르게 적응할 수 있습니다.

**③ 도메인 확장 실험**
- 금융 시계열(주가, 환율), 의료(EHR), 자연어 기반 시계열 등으로 확장 검증 필요

**④ Spatial Route의 채널 수 확장성**
- 현재 SpatialAttn은 $O(C^2)$ 복잡도를 가짐
- 변수 수 C가 수천~수만인 경우(예: 센서 네트워크) 적용 가능성 검토 필요

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 이하 비교 분석에서 인용된 외부 논문들의 구체적 수치는 본 논문 내에 직접 기재된 것과, 일반적으로 알려진 연구 동향을 기반으로 기술합니다. 각 외부 논문의 정확한 수치는 해당 원문을 직접 확인하시기 바랍니다.

#### 주요 관련 연구 연대표

| 연도 | 연구 | 핵심 기여 | STL과의 관계 |
|------|------|----------|------------|
| 2020 | **Informer** [Zhou et al., AAAI 2021] | ProbSparse Attention으로 긴 시퀀스 효율화 | STL 비교 대상 |
| 2021 | **Autoformer** [Wu et al., NeurIPS 2021] | 자기상관 분해 + 계절성-추세 분리 | STL 비교 대상 |
| 2022 | **FEDformer** [Zhou et al., ICML 2022] | 주파수 도메인 분해 + Transformer | STL 비교 대상 |
| 2023 | **LTSF-Linear** [Zeng et al., AAAI 2023] | 단순 선형 모델이 Transformer 능가 | STL의 직접적 베이스라인 |
| 2023 | **PatchTST** [Nie et al., ICLR 2023] | 시계열 패치(patch) 기반 자기지도 Transformer | **STL 미비교** |
| 2023 | **TimesNet** [Wu et al., ICLR 2023] | 2D 시공간 변환을 통한 패턴 학습 | **STL 미비교** |
| 2023 | **iTransformer** [Liu et al., 2023] | 채널 차원을 토큰으로 취급하는 역방향 Transformer | **STL 미비교** |
| 2024 | **TimesFM** [Google, 2024] | 대규모 사전학습 기반 범용 TSF 파운데이션 모델 | STL의 잠재적 경쟁자 |

> 💡 **PatchTST**: 시계열을 일정 길이의 "패치(patch)"로 분할한 후 각 패치를 Transformer의 토큰으로 처리하는 방법입니다. 로컬 시간 패턴을 효과적으로 포착합니다.

#### STL이 앞으로의 연구에 미치는 영향

1. **데이터 희소 TSF 연구 활성화**: 논문이 최초로 공식화한 "데이터 희소 시나리오" 분류 체계($T \ll \tau$ vs $T \gg \tau$)는 후속 연구의 실험 설계 기준으로 채택될 가능성

2. **Linear-Hybrid 아키텍처 방향 제시**: 순수 Transformer나 순수 Linear가 아닌, Linear를 핵심으로 하되 어텐션과 임베딩을 보조적으로 활용하는 하이브리드 설계 방향을 제시

3. **실용적 응용 영역 확대**: 자율주행(JAAD), 희귀 질병 예측 등 기존 TSF 연구에서 다루지 않던 도메인에 대한 검증을 선도

#### 앞으로 연구 시 고려할 점

| 고려사항 | 이유 |
|---------|------|
| **PatchTST, iTransformer와의 공정 비교** | 2023년 SOTA 대비 성능이 불명확함. 포괄적 비교 필수 |
| **대규모 파운데이션 모델과의 경쟁** | TimesFM 등 사전학습 모델과의 성능/비용 트레이드오프 분석 필요 |
| **θT 자동화** | 수동 임계값 설정은 실용성을 저해. 자동화 메커니즘 개발 우선순위 |
| **계산 효율성 분석** | 3-경로 구조의 추가 연산 비용이 성능 향상을 정당화하는지 FLOPs 비교 필요 |
| **채널 독립 vs 채널 의존 설계의 재검토** | STL은 채널 의존 설계를 채택하지만, 일부 연구에서는 채널 독립이 과적합 방지에 유리함을 보고 |
| **프리트레이닝 결합 가능성** | STL의 구조를 대규모 사전학습과 결합하여 few-shot TSF 성능 향상 가능성 탐색 |
| **불규칙 시계열(Irregular Time Series) 대응** | 현재 STL은 등간격 시계열을 가정. 의료·금융 등 불규칙 간격 데이터 처리 방안 필요 |

---

## 참고 자료 목록

1. **주 논문**: Zuo, A., Zhang, H., Li, Z., & Zheng, C. (2023). *Spatiotemporal-Linear: Towards Universal Multivariate Time Series Forecasting*. arXiv:2312.14869v1
2. Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). *Are Transformers Effective for Time Series Forecasting?* AAAI 2023. [LTSF-Linear]
3. Zhou, H., et al. (2021). *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting*. AAAI 2021.
4. Wu, H., et al. (2021). *Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting*. NeurIPS 2021.
5. Zhou, T., et al. (2022). *FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting*. ICML 2022.
6. Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS 2017.
7. He, K., et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR 2016.
8. Kotseruba, I., Rasouli, A., & Tsotsos, J.K. (2016). *Joint Attention in Autonomous Driving (JAAD)*. arXiv:1609.04741
9. Cressie, N., & Wikle, C.K. (2011). *Statistics for Spatio-Temporal Data*. Wiley.
10. Nie, Y., et al. (2023). *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*. ICLR 2023. [PatchTST] — *STL 논문 내 미인용, 외부 참조*
11. Wu, H., et al. (2023). *TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis*. ICLR 2023. — *STL 논문 내 미인용, 외부 참조*
