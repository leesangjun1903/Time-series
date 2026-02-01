
# Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts

## I. 핵심 주장 및 학술 기여

**Time-MoE**(Billion-Scale Time Series Foundation Models with Mixture of Experts)는 시계열 예측 분야에서 처음으로 **2.4 억만 파라미터 규모**에 도달한 파운데이션 모델이다. 이 논문의 세 가지 핵심 기여는 다음과 같다:[1]

1. **희소 구조를 통한 확장성 달성**: MoE(Mixture-of-Experts) 아키텍처 도입으로 모델 파라미터는 2.4B이지만 활성화 파라미터는 1.1B으로 제한하여, 동일한 계산 예산 내에서 조밀한(Dense) 모델을 크게 능가한다.[1]

2. **대규모 고품질 데이터셋 구축**: 300억 개 이상의 시간 포인트를 포함하는 **Time-300B** 데이터셋을 공개하였으며, 이는 에너지, 금융, 기후, 웹, 운송 등 9개 도메인을 아우른다.[1]

3. **통일된 아키텍처의 유연성**: 고정된 입력/출력 길이 제약 없이 **4096 길이의 맥락**과 **임의의 예측 지평선**을 지원하는 것을 처음으로 달성했다.[1]

***

## II. 문제 정의 및 해결 동기

### II.1 시계열 파운데이션 모델의 구조적 병목

기존 시계열 모델들은 세 가지 근본적 문제에 직면했다:

**계산 효율성 문제**: 기존 조밀한 트랜스포머는 각 입력 토큰이 모든 파라미터와 상호작용하므로, 모델 크기가 증가하면 추론 비용도 선형적으로 증가한다. 반면 자연어처리와 시각 분야의 MoE 기반 대형 모델들(GShard, Switch Transformers)은 이미 이 문제를 해결했으나, 시계열 영역에는 적용되지 않았다.[1]

**데이터 품질 및 불균형**: 공개 시계열 데이터는 높은 비율의 결측값과 이상치를 포함한다. Time-300B 이전 연구들은 이러한 데이터 처리 파이프라인을 충분히 다루지 않았으며, 특히 기후(90.5%)와 에너지(5.17%) 데이터의 극심한 불균형이 존재했다.[1]

**유연성 부족**: 선행 파운데이션 모델들의 제약:
- **Timer**: 임의 출력 길이 지원 부재 → 잘린 출력 문제[1]
- **Moment**: 고정 입력 맥락 길이(512)[1]
- **Moirai**: 입출력 계층의 hardcoded heuristic 의존[1]

### II.2 스케일 법칙의 시계열 영역 미확인

언어 및 시각 모델에서는 스케일 법칙(parameter count, dataset size, compute)이 성능 향상을 예측하는 것으로 잘 알려졌지만, 시계열 영역에서는 다음과 같은 이상 현상이 보고되었다:

- **모델 크기의 비단조성**: 더 복잡한 모델이 항상 더 나은 성능을 보이지 않음
- **지평선의 역설**: 더 긴 입력 맥락이 성능을 해칠 수 있음

따라서 시계열에서 스케일 법칙이 LLM과 동일하게 작동하는지 경험적으로 검증이 필요했다.[1]

***

## III. 제안 방법론: 수식 및 아키텍처 상세

### III.1 문제 정의

주어진 과거 관측값 시계열 $\mathbf{X}_{1:T} = (x_1, x_2, \ldots, x_T) \in \mathbb{R}^T$에서 다음 $H$ 시간 단계를 예측하는 작업:

$$\hat{\mathbf{X}}_{T+1:T+H} = f_\theta(\mathbf{X}_{1:T}) \in \mathbb{R}^H$$

여기서 $T$는 맥락 길이(최대 4096), $H$는 예측 지평선이며, 둘 다 추론 시에 유연하게 변할 수 있다.[1]

### III.2 입력 토큰 임베딩

**포인트-단위 토큰화**를 통해 완전한 시간 정보를 보존:

$$h_t^0 = \text{SwiGLU}(x_t) = \text{Swish}(\mathbf{W}x_t) \otimes (\mathbf{V}x_t)$$

여기서 $\mathbf{W} \in \mathbb{R}^{D \times 1}$, $\mathbf{V} \in \mathbb{R}^{D \times 1}$는 학습 가능한 파라미터이고, $\otimes$는 Hadamard 곱, $D$는 숨겨진 차원이다.[1]

### III.3 MoE 트랜스포머 블록

각 레이어 $l$의 연산 구조:

$$u_t^l = \text{SA}\left(\text{RMSNorm}(h_t^{l-1})\right) + h_t^{l-1}$$

$$\overline{u}_t^l = \text{RMSNorm}(u_t^l)$$

$$h_t^l = \text{Mixture}(\overline{u}_t^l) + u_t^l$$

여기서 SA는 인과 마스크를 가진 자주의(Causal Self-Attention)를 나타낸다.[1]

### III.4 희소 혼합 전문가 레이어

**핵심 수식**:

$$\text{Mixture}(\overline{u}_t^l) = g_{N+1,t} \text{FFN}_{N+1}(\overline{u}_t^l) + \sum_{i=1}^{N} \left[g_{i,t} \text{FFN}_i(\overline{u}_t^l)\right]$$

여기서:
- $N$: 비공유 전문가 수
- $N+1$: 공유 전문가(shared expert)

**라우팅 게이트**:

$$g_{i,t} = \begin{cases}
s_{i,t}, & s_{i,t} \in \text{TopK}(\{s_{j,t}|1 \leq j \leq N\}, K) \\
0, & \text{otherwise}
\end{cases}$$

$$g_{N+1,t} = \text{Sigmoid}(\mathbf{W}_N^l \overline{u}_t^l)$$

$$s_{i,t} = \text{Softmax}_i(\mathbf{W}_i^l \overline{u}_t^l)$$

여기서 $\mathbf{W}_i^l \in \mathbb{R}^{1 \times D}$이고 $K=2$로 설정(즉, 상위-2 라우팅).[1]

이 설계는 다음 이점을 제공한다:
- **희소성**: 토큰당 $K$개의 전문가만 활성화로 계산량 감소
- **전문화**: 각 전문가가 특정 시계열 패턴을 학습
- **강건성**: 공유 전문가가 도메인 간 공통 지식 캡처

### III.5 다해상도 예측(Multi-Resolution Forecasting)

일반적인 단일 출력 헤드 대신, **$P$개의 출력 프로젝션**을 학습:

$$\hat{\mathbf{X}}_{t+1:t+p_j} = \mathbf{W}_{p_j} \mathbf{h}_t^L$$

여기서 $\mathbf{W}_{p_j} \in \mathbb{R}^{p_j \times D}$는 지평선 $p_j$에 대한 학습 가능 가중치이고, $\mathbf{h}_t^L$은 마지막 MoE 블록의 출력이다.[1]

**훈련 중**: 모든 지평선에서의 손실을 결합

**추론 중**: 그리디 스케줄링 알고리즘(Algorithm 1)으로 임의 길이 $H$ 지원:
- 남은 예측 단계 < $p_j$인 가장 큰 지평선부터 선택
- 예측 길이에 도달할 때까지 반복
- 이를 통해 단일 모델이 {96, 192, 336, 720} 뿐만 아니라 모든 지평선 지원

### III.6 손실 함수 및 훈련 안정성

**자회귀 손실(Huber loss 기반)**:

$$L_{\text{ar}}(x_t, \hat{x}_t) = \begin{cases}
\frac{1}{2}(x_t - \hat{x}_t)^2, & |x_t - \hat{x}_t| \leq \delta \\
\delta \left(|x_t - \hat{x}_t| - \frac{1}{2}\delta\right), & \text{otherwise}
\end{cases}$$

Huber loss는 이상치에 대한 강건성이 우수하여 대규모 데이터셋 훈련 안정성을 향상시킨다.[1]

**보조 부하 균형 손실**:

$$L_{\text{aux}} = N \sum_{i=1}^{N} f_i r_i$$

여기서:

$$f_i = \frac{1}{KT} \sum_{t=1}^{T} \mathbb{I}(\text{Time point } t \text{ selects Expert } i)$$

$$r_i = \frac{1}{T} \sum_{t=1}^{T} s_{i,t}$$

이 손실은 전문가 $i$에 할당된 토큰 분수($f_i$)와 라우터 확률의 비율($r_i$)의 곱을 최소화하여, **라우팅 붕괴**(routing collapse: 소수 전문가에 과도한 의존) 방지.[1]

**최종 손실**:

$$L = \frac{1}{P} \sum_{j=1}^{P} L_{\text{ar}}\left(\mathbf{X}_{t+1:t+p_j}, \hat{\mathbf{X}}_{t+1:t+p_j}\right) + \alpha L_{\text{aux}}$$

여기서 $\alpha = 0.02$로 설정.[1]

***

## IV. 모델 구조 및 학습 구성

### IV.1 모델 사양

| 항목 | TIME-MOEbase | TIME-MOElarge | TIME-MOEultra |
|------|--------------|----------------|----------------|
| 레이어 수 | 12 | 12 | 36 |
| 헤드 | 12 | 12 | 16 |
| 전문가 수 | 8 | 8 | 8 |
| K (활성 전문가) | 2 | 2 | 2 |
| 모델 차원($d_{\text{model}}$) | 384 | 768 | 1024 |
| FFN 차원($d_{\text{ff}}$) | 1536 | 3072 | 4096 |
| 전문가 차원($d_{\text{expert}}$) | 192 | 384 | 512 |
| **활성 파라미터** | **50M** | **200M** | **1.1B** |
| **전체 파라미터** | **113M** | **453M** | **2.4B** |

TIME-MOEultra는 8GB 이하 VRAM의 소비자 GPU에서 추론 가능하도록 설계.[1]

### IV.2 Time-300B 데이터셋 구성

| 도메인 | 시퀀스 수 | 관측값 (B) | 비율 |
|--------|----------|----------|------|
| 에너지 | 2,875,335 | 15.981 | 5.17% |
| 금융 | 1,715 | 0.414K | 0.0001% |
| 의료 | 1,752 | 0.471K | 0.0001% |
| 자연 | 31,621,183 | 279.724 | **90.50%** |
| 판매 | 110,210 | 26.382M | 0.008% |
| 합성 | 11,968,625 | 9.222 | 2.98% |
| 운송 | 622,414 | 2.130 | 0.69% |
| 웹 | 972,158 | 1.804 | 0.58% |

**데이터 정제 파이프라인** (Appendix C):
1. **결측값 처리**: NaN/Inf → 시퀀스 분할 (대체 대신)
2. **이상 관측 필터링**: 1차·2차 차분의 영점 비율 > 0.2인 윈도우 제거
3. **다운샘플링**: 날씨, CMIP6, ERA5 같은 대규모 데이터셋 균형화
4. **배치 구성**: 실제 약 117B 시간 포인트로 훈련 (도메인별 고정 비율 샘플링)[1]

### IV.3 훈련 설정

- **훈련 스텝**: 100,000 (총 4 million 시간 포인트/이터레이션)
- **배치 크기**: 1024
- **최대 시퀀스 길이**: 4096
- **멀티해상도 지평선**: {1, 8, 32, 64}
- **옵티마이저**: AdamW (lr=1e-3, weight decay=1e-1, β₁=0.9, β₂=0.95)
- **학습률 스케줄**: 처음 10,000스텝 선형 워밍업 → 코사인 어닐링
- **하드웨어**: 128 × NVIDIA A100-80G GPU
- **정밀도**: BF16 (Flash Attention 통합 시 추론 19% 가속)[1]

***

## V. 성능 향상 및 스케일 법칙 검증

### V.1 Zero-Shot 예측 성능

제로샷 환경에서 6개 벤치마크(ETTh1, ETTh2, ETTm1, ETTm2, Weather, Global Temp)에 대한 평가:

| 모델 | 평균 MSE | 평균 MAE | 랭킹 |
|-----|---------|---------|------|
| **TIME-MOEultra** | **0.322** | **0.372** | **1등 (28회)** |
| Chronos Large | 0.416 | 0.405 | |
| Moment | 0.429 | 0.412 | |
| Moirai Large | 0.359 | 0.373 | |
| TimesFM | 0.396 | 0.413 | |

**핵심 성과**:
- Chronos Large 대비 **23% MSE 감소**
- Moment 대비 **30% MSE 감소**
- Moirai Large 대비 **11% MSE 감소**[1]

### V.2 In-Distribution(전체 예측) 성능

동일 벤치마크에서 한 에포크 파인튜닝 후:

| 모델 | 평균 MSE | 평균 MAE | 랭킹 |
|-----|---------|---------|------|
| **TIME-MOEultra** | **0.301** | **0.358** | **1등 (33회)** |
| iTransformer | 0.349 | 0.382 | |
| TimeMixer | 0.337 | 0.375 | |
| PatchTST | 0.349 | 0.382 | |

**평균 24% MSE 감소** 달성 (기존 최고 성능 대비).[1]

### V.3 스케일 법칙 검증

**핵심 발견**: TIME-MOE는 모델 크기와 데이터 규모 증가에 따른 지속적 성능 개선을 보여, 시계열 영역에서 LLM과 유사한 스케일 법칙이 작동함을 증명:

$$L \propto N^{-\alpha} \quad \text{(파라미터 수에 대한 거듭제곱 법칙)}$$

Figure 3 (Right)에서 보면:
- 더 많은 데이터로 훈련된 모델이 모든 크기에서 일관되게 우월
- 모델 크기(50M → 200M → 1.1B)와 데이터량(~40B → ~120B) 동시 증가 시 최적 성능

이는 다음을 시사한다:
- **비용-효율성**: 동일 계산 예산 내에서 더 크고 희소한 모델이 더 작고 조밀한 모델을 능가
- **선형 확장성**: 추론 비용 동결 하에 모델 크기 증가 가능[1]

***

## VI. 모델의 일반화 성능: 심층 분석

### VI.1 Zero-Shot 일반화 메커니즘

Time-MoE가 미보유 데이터(Zero-shot)에서 우수한 성능을 달성하는 메커니즘:

**다중 도메인 전문가 특화**: 
Figure 4의 게이팅 스코어 분석에서, 서로 다른 벤치마크가 크게 다른 전문가 활성화 패턴을 보인다. 예를 들어:
- 에너지 도메인 데이터 → 전문가 {1,3} 선호
- 기후 도메인 데이터 → 전문가 {2,5} 선호

이는 **사전훈련 중 도메인별 패턴 학습**을 나타내며, 새로운 도메인에 대해서도 이러한 학습된 전문화가 전이(transfer)된다.[1]

**멀티태스크 학습의 일반화 이점**:
표 5(우측)에서 다해상도 헤드 제거 시 성능 저하를 보면:
- 4개 헤드 {1,8,32,64}: MSE 0.262
- 3개 헤드 {1,8,32}: MSE 0.273 (+4.2% 악화)
- 2개 헤드 {1,8}: MSE 0.320 (+22.1% 악화)

다양한 지평선에 대한 멀티태스크 최적화가 특정 지평선뿐만 아니라 **도메인 간 부분 패턴(sub-patterns)의 강건한 표현**을 학습하게 한다.[1]

### VI.2 비교 분석: 아키텍처별 확장성 차이

2024-2025년 최신 연구와의 비교:

**Encoder-only vs. Decoder-only 확장성**:
- Yao et al. (2024) "Towards Neural Scaling Laws for TSFM"에서 인코더-온리 트랜스포머가 OOD 및 ID 설정 모두에서 더 나은 확장 특성을 보임[2]
- Time-MoE의 디코더-온리 설계는 **자회귀 생성(next-token prediction)** 용이성과 **유연한 지평선 길이** 트레이드오프를 반영

**Moirai-MoE (Liu et al., 2024a)와의 비교**:
- Moirai-MoE: 최대 935M 파라미터[1]
- **Time-MoE: 2.4B 파라미터 (2.6배 더 큼)**
- Time-MoE가 다른 expert/routing 설계로 더 큰 규모 달성[1]

### VI.3 일반화 성능의 한계 및 재평가 필요성

**최근 발견 사항 - Context Parroting 문제** (Emami et al., 2025):

혁신적 연구 "Context parroting: A simple but tough-to-beat baseline"에서 지적:
- 많은 시계열 파운데이션 모델이 실제로는 **맥락을 단순히 복사(parroting)하는 전략**에 의존[3]
- Chaos 시스템, 난류, 결합 진동자 등에서 순진한 context parroting이 Time-MoE, Moirai, Chronos 등을 능가[3]

**함의**: 공개 벤치마크(ETT, Weather 등)에서의 높은 성능이 **실제 역학 시스템 학습**보다는 **패턴 외삽(extrapolation)**의 편이성을 반영할 가능성

**REAL-V-TSFM 벤치마크의 추가 증거** (Li et al., 2025):
- 실제 비디오 광학 흐름에서 추출한 시계열에서 Time-MoE 등 SOTA 모델의 성능 대폭 저하[4]
- "제로샷 일반화 간격(generalization gap)" 명시적 증명

### VI.4 일반화 성능 개선 경로

**1단계: 다양한 시계열 특성 학습**
- 시간 스케일: 초 → 년
- 통계적 특성: 비정상성, 계절성, 다중 스케일 구조
- 물리적 제약: 보존 법칙, 에너지 입력

**2단계: 도메인 적응 메커니즘**
- Parameter-efficient fine-tuning (LoRA, adapter)
- Domain-specific prefix tokens
- Contrastive learning으로 도메인 불변 표현 학습[5]

**3단계: 불확실성 정량화**
- 앙상블 방법 (전문가별 분산)
- 베이지안 접근
- 신뢰도 구간 산출으로 위험 관리

***

## VII. 2020년 이후 관련 최신 연구 비교

### VII.1 시계열 파운데이션 모델의 진화

| 연도 | 모델 | 아키텍처 | 파라미터 | 데이터 | 특징 |
|------|------|---------|---------|--------|------|
| 2020 | N-BEATS | 선형 구조 | - | 특정 도메인 | Task-specific, 초기 DL |
| 2023 | DLinear, SparseTSF | 선형 | ~1M-100M | 단일 도메인 | 간단함의 승리 |
| 2024 | TimesFM | Decoder | 200M | 100B | 구글 내부, 제한된 공개 |
| 2024 | Lag-Llama | Decoder | 200M | 360M | 확률적 예측 |
| 2024 | Moment | Encoder | 385M | 1.13B | 마스크 재구성 |
| 2024 | Chronos | Encoder-Decoder | 710M | 84B | Quantized 토큰화 |
| 2024 | Moirai | Encoder | 311M | 27B/231B | SOTA zero-shot |
| **2025** | **Time-MoE** | **Decoder + MoE** | **2.4B** | **300B** | **첫 MoE 파운데이션, 스케일 최대** |

### VII.2 스케일 법칙 연구 진전

**2024 주요 발견들:**

**Dubey et al. (2024) "Scaling-laws for Large Time-series Models"**:[5]
- LLM과 유사한 거듭제곱 법칙 확인: $L \propto N^{-\alpha} D^{-\beta}$
- 파라미터 수, 데이터셋 크기, 컴퓨팅 예산에 걸쳐 5개 자릿수 범위
- **시사점**: 시계열도 단순 스케일 투자로 이득 가능[5]

**Yao et al. (2024) "Towards Neural Scaling Laws for TSFM"**:[2]
- OOD(out-of-distribution) 스케일링 동작이 ID와 다를 수 있음
- Encoder-only 구조 > Decoder-only 구조 in OOD
- 아키텍처 개선(Moirai, TimesFM 수준)이 OOD 확장성 저해 가능[2]

**함의**: Time-MoE의 Decoder-only + MoE 설계는 ID 성능 최적화이며, OOD 확장성은 추가 검증 필요

### VII.3 MoE 기반 시계열 모델의 최신 동향 (2025)

**3가지 신흥 MoE 아키텍처:**

**1. Seg-MoE (2026, Ortigossa & Segal)**:[6]
- **혁신**: Token-wise가 아닌 **segment-wise(세그먼트 기반) 라우팅**
- 연속 시간 세그먼트의 locality 활용
- **성과**: 7개 벤치마크에서 SOTA, 평균 MSE 12% 감소
- **Time-MoE와의 차이**: Token-level routing (Time-MoE) vs. Segment-level routing (Seg-MoE)

**2. MoHETS (2026, Ortigossa & Segal)**:[7]
- **특징**: Heterogeneous experts (depthwise convolution + Fourier-based)
- 다양한 시간 역학 캡처: 전역 추세(depthwise CNN) + 주기성(Fourier)
- **성과**: 평균 MSE 12% 감소
- **Time-MoE와의 차이**: Homogeneous FFN experts (Time-MoE) vs. Heterogeneous (MoHETS)

**3. Multi-Modal MoE (2026, Zhang et al.)**:[8]
- **확장**: 텍스트 조건부 라우팅 (뉴스 기반 금융 예측)
- Expert modulation paradigm
- **Time-MoE와의 차이**: Unimodal (Time-MoE) vs. Multi-modal cross-attention

### VII.4 합성 데이터 기반 스케일링

**CauKer (2025, Ekambaram et al.)**:[9]
- 순수 **합성 시계열만으로 파운데이션 모델 훈련**
- Gaussian Process 커널 + Structural Causal Model 결합
- 10K~10M 샘플 범위에서 명확한 스케일 법칙 관찰
- 실제 데이터는 비규칙적 스케일 행동[9]

**함의**: Time-300B의 2.98% 합성 데이터가 **강화** 되어야 하는 이유 설명

### VII.5 Zero-Shot 일반화의 재평가

**Uncovering Zero-Shot Generalization Gaps (Li et al., 2025)**:[4]
- 표준 벤치마크(ETT, Weather)에서 높은 성능 ≠ 실제 물리 시스템 학습
- **REAL-V-TSFM** 데이터셋: 비디오 기반 실제 광학 흐름
- 모든 SOTA 모델의 성능 저하 → 일반화 간격 존재[4]

***

## VIII. 모델의 한계 및 실무적 고려사항

### VIII.1 아키텍처 관점의 한계

**1. 라우팅 안정성**:
- Auxiliary loss(Equation 10) 필요 → 추가 초하이퍼파라미터 ($\alpha=0.02$)
- 극단적 클래스 불균형(자연: 90.5%, 금융: 0.0001%)에서 라우팅 붕괴 위험[1]
- 실제 배포 시 도메인 구성 변화에 취약

**2. 계산 오버헤드**:
- MoE 레이어의 추가 라우터 계산 (Softmax, Sigmoid)
- **상위-K 연산**: 각 토큰에서 전체 스코어 계산 후 Top-2 선택 → $O(ND)$[1]
- 잠재적 GPU 메모리 단편화(expert별 배치 분할)

**3. 전문가 활용 불균형**:
- Table 7에서 K 값에 민감: Top-1은 정확도 손실, Top-8은 추론 시간 58% 증가[1]
- 최적값(Top-2)이 narrow할 가능성 → 다른 데이터셋에 전이 불확실

### VIII.2 데이터 및 일반화 관점의 한계

**1. 도메인 바이어스**:
- Time-300B 구성의 극심한 불균형: 자연(90.5%) 지배적
- 에너지(5.17%), 웹(0.58%) 등은 대표성 부족
- 금융(0.0001%) 완전히 무시될 수 있음 → 금융 예측에 편향[1]

**2. 합성 데이터 의존성**:
- 2.98%(9.2B) 합성 데이터로 보강 → **실제는 제한적**
- CauKer 연구(2025)에서 순수 합성도 명확 스케일 법칙 보임[9]
- **질문**: Time-300B의 합성 데이터가 실제 역학을 충분히 학습하는가?

**3. Context Parroting 위험**:
- Emami et al. (2025)의 "맥락 복사" 현상[3]
- ETT, Weather 같은 공개 벤치마크의 높은 성능 = 반복 패턴 외삽 가능성
- 비선형 역학(chaos, turbulence)에서 실패 증거

### VIII.3 실무 배포 고려사항

**1. 메모리 효율성**:
- TIME-MOEultra: 훈련 1.77GB, 추론 226.70MB (BF16)[1]
- 비교: TIME-MOEbase (FP32) 453MB → 배 증가
- Mobile/Edge device 배포는 여전히 도전 과제

**2. 파인튜닝 복잡성**:
- 전체 2.4B 파라미터 파인튜닝은 비용 높음
- 논문에서 1-에포크 파인튜닝만 검증[1]
- LoRA/adapter 같은 효율적 방법 미제시

**3. 불확실성 정량화**:
- 논문은 점 예측만 제공 (MSE, MAE)
- 금융/에너지 실무에서 필수인 신뢰 구간 미제시
- 앙상블 방법 제시 필요

***

## IX. 향후 연구 방향 및 제언

### IX.1 모델 확장성 극대화

**1. 더 큰 규모 탐색** (3B, 5B 파라미터):
- Dubey et al. Llama3 (405B)의 스케일 법칙 확인[5]
- 시계열 3B→5B 이행 시 성능 곡선 평탄화 여부 검증
- **비용-효율성 경계** 식별

**2. 이질적 전문가 아키텍처** (MoHETS, Seg-MoE 통합):
- 단순 FFN 전문가 → 특화 구조 (CNN for trends, DFT for seasonality)
- Segment-wise routing 검증으로 locality 활용[7]
- **추예상**: 동일 계산 하에 3-5% 성능 향상

**3. 적응적 라우팅**:
- 고정 K=2 대신 동적 토큰별 K 선택
- 도메인별 최적 전문가 수 학습
- Load balancing loss 개선 (현재 auxiliary loss는 사후 처리)

### IX.2 일반화 성능 개선

**1. 도메인 균형 데이터셋 재구성**:
- Time-300B 리샘플링: log-uniform over domains
- 최소 도메인(금융 0.0001%) → 최소 5% 목표
- **가설**: 균형 데이터셋이 OOD 확장성 향상

**2. 실제 역학 학습 검증**:
- Chaos (Lorenz, Henon) 벤치마크 추가
- Turbulence, PDEs 포함 다이나믹스 스위트[3]
- Context parroting 회피 아키텍처 설계

**3. 멀티모달 확장**:
- 텍스트 조건부 예측 (금융 뉴스, 기상 보고)
- 이미지 + 시계열 (위성 영상 + 기후)[8]
- 교차 모달 일반화 성능 평가

### IX.3 실무 적용성 강화

**1. 불확실성 정량화**:
- 다중 샘플 생성 (diffusion 기반)
- 전문가별 분산 추정
- 신뢰도 구간 제공[7]

**2. 효율적 파인튜닝**:
- LoRA + MoE 라우팅 헤드 파인튜닝
- Prompt tuning for time series
- Task-specific adapter 설계

**3. 온라인 학습 및 연속 개선**:
- Streaming time series 지원
- 드리프트 감지 (concept drift)
- Incremental expert 추가

### IX.4 이론적 이해 심화

**1. 시계열 스케일 법칙의 이론화**:
- Chinchilla 최적성(LLM) → 시계열 적용?[5]
- FLOPs 계산 표준화 (희소 모델의 모호성)
- 시계열 **특수성** 반영 이론 필요

**2. MoE 라우팅의 최적성**:
- Top-K vs. Expert choice routing 비교
- Load balancing의 통계적 보장
- 라우팅 선택의 정보 이론적 하한

**3. Generalization bounds**:
- VC dimension 하한 (시계열 자회귀 특성)
- PAC-Bayes 기반 일반화 경계
- Domain adaptation gap 정량화

***

## X. 결론 및 학술적 의의

### X.1 핵심 학술 기여 요약

Time-MoE는 시계열 예측 분야에서 다음의 획기적 진전을 표현한다:

1. **스케일 달성**: 처음으로 **2.4B 파라미터** 파운데이션 모델 구현 (1.1B activated)
   - 이전 최대: Chronos 710M, Moirai 311M
   - **3-7배 규모 증대**

2. **효율성 입증**: 희소 아키텍처로 **동일 계산 예산 하에 우월 성능**
   - Dense vs. Sparse: 훈련 78% 감소, 추론 39% 감소
   - 비용-성능 프론티어 이동

3. **스케일 법칙 검증**: 시계열 영역에서 **LLM과 유사한 거듭제곱 법칙** 최초 확인
   - $L(N,D) \propto N^{-\alpha} D^{-\beta}$ 적용 가능
   - 이론적 불확실성 제거

4. **통일된 설계**: 고정 길이 제약 없는 **유연한 입출력** 지원
   - 임의 예측 지평선 (그리디 스케줄링)
   - 4096 길이 맥락 (4배 증대)

### X.2 한계와 재평가

그러나 동시대 연구(2024-2025)는 중요한 경고를 제시한다:

- **Context Parroting (2025)**: 높은 벤치마크 성능이 **실제 역학 학습이 아닐 가능성**[3]
- **일반화 간격 (2025)**: 실제 영상 기반 시계열에서 **성능 저하**[4]
- **아키텍처 트레이드오프 (2024)**: Decoder-only + MoE가 **OOD 확장성 상충 가능**[2]

따라서 Time-MoE의 성과는 **공개 벤치마크 내에서의 성공**으로 해석해야 하며, **실제 물리 시스템 일반화**는 추가 검증이 필수이다.

### X.3 미래 연구 생태계의 위치

Time-MoE는 시계열 파운데이션 모델의 발전 궤적에서 **"확장의 시대" 종료와 "효율화 시대" 시작**을 표시한다:

**기대 효과**:
- Seg-MoE, MoHETS 같은 구조적 혁신 가속
- Segment-level 또는 heterogeneous expert routing 주류화
- 멀티모달 시계열 파운데이션 모델 활성화

**미해결 질문들**:
- 3B~5B 규모는 비용 대비 이득이 있는가?
- 도메인 균형 재조정으로 OOD 일반화 향상 가능한가?
- 실제 역학 학습을 강제하는 아키텍처는?

이러한 질문들이 2025-2026년 시계열 AI의 중심이 될 것으로 예상된다.

***

## 참고문헌

 Shi, X., Wang, S., Nie, Y., et al. (2025). "Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts." ICLR 2025. arXiv:2409.16040v4.[1]

 Yao, Q., Liang, H., et al. (2024). "Towards Neural Scaling Laws for Time Series Foundation Models." arXiv:2410.12360.[2]

 Emami, P., Dunn, J., et al. (2025). "Context parroting: A simple but tough-to-beat baseline for foundation models in scientific machine learning." arXiv:2505.11349.[3]

 Li, L., Sleem, L., et al. (2025). "Uncovering Zero-Shot Generalization Gaps in Time-Series Foundation Models." arXiv:2509.26347.[4]

 Dubey, A., et al. (2024). "Scaling-laws for Large Time-series Models." arXiv:2405.13867.[5]

 Ortigossa, E. S., & Segal, E. (2026). "Seg-MoE: Multi-Resolution Segment-wise Mixture-of-Experts for Time Series Forecasting Transformers." arXiv:2601.21641.[6]

 Ortigossa, E. S., & Segal, E. (2026). "MoHETS: Long-term Time Series Forecasting with Mixture-of-Heterogeneous-Experts." arXiv:2601.21866.[7]

 Zhang, L., Maatouk, A., et al. (2026). "Multi-Modal Time Series Prediction via Mixture of Modulated Experts." arXiv:2601.21547.[8]

 Ekambaram, V., et al. (2025). "CauKer: Classification time series foundation models can be pretrained on synthetic data only." arXiv:2508.02879.[9]

 Goswami, M., et al. (2024). "Moment: A family of open time-series foundation models." ICML 2024.[10]

 Woo, G., Liu, C., et al. (2024). "Moirai: A time series foundation model for universal forecasting." arXiv:2402.02592.[11]

출처
[1] 2409.16040v4.pdf https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f241affb-d163-4e72-bcfe-33872f219f90/2409.16040v4.pdf
[2] Towards Neural Scaling Laws for Time Series Foundation Models https://arxiv.org/abs/2410.12360
[3] CauKer: classification time series foundation models can be pretrained on synthetic data only https://arxiv.org/abs/2508.02879
[4] Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts https://arxiv.org/abs/2409.16040
[5] Scaling-laws for Large Time-series Models https://arxiv.org/abs/2405.13867
[6] Time Series Foundation Models and Deep Learning Architectures for Earthquake Temporal and Spatial Nowcasting https://arxiv.org/abs/2408.11990
[7] Only the Curve Shape Matters: Training Foundation Models for Zero-Shot Multivariate Time Series Forecasting through Next Curve Shape Prediction https://arxiv.org/abs/2402.07570
[8] Scaling Law for Large-Scale Pre-Training Using Chaotic Time Series and Predictability in Financial Time Series https://arxiv.org/abs/2509.04921
[9] Time-Series Large Language Models: A Systematic Review of State-of-the-Art https://ieeexplore.ieee.org/document/10856008/
[10] Context parroting: A simple but tough-to-beat baseline for foundation models in scientific machine learning https://arxiv.org/abs/2505.11349
[11] Scaling Law for Time Series Forecasting https://arxiv.org/abs/2405.15124
[12] Towards Neural Scaling Laws for Time Series Foundation Models https://arxiv.org/html/2410.12360
[13] Scaling-laws for Large Time-series Models https://arxiv.org/pdf/2405.13867.pdf
[14] How to Upscale Neural Networks with Scaling Law? A Survey and Practical
  Guidelines https://arxiv.org/pdf/2502.12051.pdf
[15] Scaling Law for Time Series Forecasting http://arxiv.org/pdf/2405.15124.pdf
[16] Sundial: A Family of Highly Capable Time Series Foundation Models https://arxiv.org/html/2502.00816v1
[17] UniCL: A Universal Contrastive Learning Framework for Large Time Series
  Models http://arxiv.org/pdf/2405.10597.pdf
[18] A Scalable and Transferable Time Series Prediction Framework for Demand
  Forecasting http://arxiv.org/pdf/2402.19402.pdf
[19] Can Test-Time Scaling Improve World Foundation Model? https://arxiv.org/html/2503.24320v1
[20] Multi-Modal Time Series Prediction via Mixture of ... https://arxiv.org/abs/2601.21547
[21] Towards Foundation Models for Zero-Shot Time Series ... https://arxiv.org/html/2509.21190v1
[22] MoHETS: Long-term Time Series Forecasting with Mixture- ... https://www.arxiv.org/abs/2601.21866
[23] Large Language Models Are Zero-Shot Time Series ... https://arxiv.org/pdf/2310.07820.pdf
[24] Seg-MoE: Multi-Resolution Segment-wise Mixture-of- ... https://arxiv.org/abs/2601.21641
[25] Empowering Time Series Analysis with Foundation Models https://arxiv.org/html/2405.02358v3
[26] Diversified Scaling Inference in Time Series Foundation ... https://arxiv.org/html/2601.17376v1
[27] Wavelet Mixture of Experts for Time Series Forecasting https://arxiv.org/abs/2508.08825
[28] Uncovering Zero-Shot Generalization Gaps in Time-Series ... https://arxiv.org/html/2509.26347v2
[29] LeMoLE: LLM-Enhanced Mixture of Linear Experts for Time ... https://arxiv.org/abs/2412.00053
[30] Enhancing Zero-Shot Time Series Forecasting in Off-the- ... https://arxiv.org/html/2512.20140v1
[31] Time-MoE: Billion-Scale Time Series Foundation Models ... https://openreview.net/forum?id=e1wDDFmlVu
[32] Mixture-of-Linear-Experts for Long-term Time Series Forecasting https://proceedings.mlr.press/v238/ni24a/ni24a.pdf
[33] Uncovering Zero-Shot Generalization Gaps in Time-Series ... https://arxiv.org/abs/2509.26347
[34] Less is More: Unlocking Specialization of Time Series ... https://neurips.cc/virtual/2025/poster/116420
[35] TIME-MOE: Billion-Scale Time Series Forecasting with Mixture-of-Experts https://www.reddit.com/r/datascience/comments/1h3hxe4/timemoe_billionscale_time_series_forecasting_with/
[36] ZeroTS: Zero-shot Time Series forecasting via https://openreview.net/pdf?id=Lz221VLWrO
[37] Foundation Models for Times Series - Open Data Science https://opendatascience.com/foundation-models-for-times-series/
[38] Mixture-of-Experts-Enhanced Foundation Time Series ... https://arxiv.org/abs/2505.15151
[39] Mixture-of-Linear-Experts for Long-term Time Series Forecasting https://arxiv.org/abs/2312.06786
[40] [ArXiv 2024] Lag-Llama: Towards Foundation Models for ... https://velog.io/@sheoyonj/ArXiv-2024-Lag-Llama-Towards-Foundation-Models-for-Probabilistic-Time-Series-Forecasting
[41] Qingrenn/TSFM-ScalingLaws: [ICLR 2025] Official ... https://github.com/Qingrenn/TSFM-ScalingLaws
[42] [Paper Review] TIME-MOE: BILLION-SCALE TIME SERIES FOUNDATION MODELS WITH MIXTURE OF EXPERTS https://dsba.snu.ac.kr/?kboard_content_redirect=2954
[43] Language Models Still Struggle to Zero-shot Reason about ... https://aclanthology.org/2024.findings-emnlp.201.pdf
