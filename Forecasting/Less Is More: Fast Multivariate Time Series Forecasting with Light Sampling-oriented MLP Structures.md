# Less Is More: Fast Multivariate Time Series Forecasting with Light Sampling-oriented MLP Structures

- 복잡한 RNN·GNN·Transformer 대신, **순수 MLP + 다운샘플링**만으로도 멀티변량 시계열 예측에서 SOTA 또는 준-SOTA 성능과 높은 효율을 달성할 수 있다는 것을 보인다.[^1_1][^1_2]
- 연속(continuous)·간격(interval) 샘플링과 IEBlock(Information Exchange Block)이라는 단순하지만 구조적으로 설계된 MLP 블록을 통해, 장·단기 패턴과 변수 간 상호의존성을 효과적으로 포착한다.[^1_1]
- 8개 벤치마크에서 5개 데이터셋에서 기존 SOTA를 능가하고 나머지에서도 비슷한 성능을 내며, 가장 큰 데이터셋에서는 기존 SOTA 대비 FLOPs를 5% 이하로 줄이고 예측 성능의 분산(랜덤 시드 간)을 크게 감소시킨다.[^1_2][^1_1]

***

## 2. 문제 설정, 방법(수식), 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

- 입력: 시점 $t$에서 길이 $T$의 look-back window

$$
X_t = \{x_{t-T+1}, \dots, x_t \}, \quad x_i \in \mathbb{R}^N
$$

여기서 $N$은 변수(시계열)의 수이다.[^1_1]
- 목표:
    - 다단계 예측(multi-step): $\{x_{t+1}, \dots, x_{t+L}\}$ 예측
    - 단일 시점 예측(single-step): $x_{t+L}$ 예측.[^1_1]
- 난점:

1) 각 변수의 단기(local)·장기(global) 시계열 패턴을 동시에 포착.
2) 변수 간 복잡한 상호의존성 모델링.[^1_1]
- 기존 RNN/GNN/Transformer 기반 모델은 긴 시퀀스·고차원 변수에서 계산 비용, 메모리, 데이터 요구량이 크고, 시드에 따라 성능 변동이 크다는 문제가 있다.[^1_1]

***

### 2.2 제안 방법: 샘플링 전략과 IEBlock

#### 2.2.1 연속 샘플링(continuous sampling)

- 길이 $T$의 단변량 시계열 $X_t \in \mathbb{R}^T$를 서브시퀀스 길이 $C$로 나누면, 총 $\frac{T}{C}$개의 연속 서브시퀀스가 생긴다.[^1_1]
- $j$-번째 열(서브시퀀스)은

$$
X^{\text{con}}_{t,\cdot j} = \{ x_{t-T+(j-1)C+1}, x_{t-T+(j-1)C+2}, \dots, x_{t-T+jC} \} \tag{1}
$$
- 결과적으로 $X^{\text{con}}_t \in \mathbb{R}^{C \times \frac{T}{C}}$이며, 각 열이 서로 겹치지 않는 연속 구간이다.[^1_1]
- 역할: 국소적인 단기 패턴(예: 하루 주기, 짧은 변동)을 집중적으로 학습하도록 유도.[^1_1]


#### 2.2.2 간격 샘플링(interval sampling)

- 동일한 $X_t$에 대해, 고정 간격으로 $C$개를 샘플링하여 서브시퀀스를 만든다.[^1_1]
- $j$-번째 열:

```math
X^{\text{int}}_{t,\cdot j} = \big\{ x_{t-T+j},\; x_{t-T+j+\lfloor \frac{T}{C} \rfloor},\; \dots,\; x_{t-T+j+(C-1)\lfloor \tfrac{T}{C} \rfloor} \big\} \tag{2}
```

- 역시 $X^{\text{int}}_t \in \mathbb{R}^{C \times \frac{T}{C}}$이며, 시퀀스 전체에 걸쳐 고르게 간격을 두고 샘플링한다.[^1_1]
- 역할: 세밀한 로컬 정보를 희생하는 대신 장기적인 패턴(추세·계절성)을 포착.[^1_1]

※ 두 샘플링 모두 “다운샘플링”이지만, 토큰을 버리지 않고 재배열만 하여 정보 손실을 줄이면서 연산량을 줄인다.[^1_1]

***

### 2.2.3 IEBlock: Information Exchange Block

입·출력:

- 입력 $Z \in \mathbb{R}^{H \times W}$:
    - 샘플링 파트에서 $H=C$ (서브시퀀스 길이), $W = \tfrac{T}{C}$ (서브시퀀스 개수).[^1_1]
    - 예측 파트에서 $H$는 feature 차원, $W = N$ (변수 개수).[^1_1]
- 출력: $\tilde{Z} \in \mathbb{R}^{F \times W}$.[^1_1]

구성(“병목” 구조):

1. **Temporal projection** (열 단위 MLP, 공유 가중치):

$$
z^t_{\cdot i} = \text{MLP}_t(z_{\cdot i}),\quad \text{MLP}_t: \mathbb{R}^H \to \mathbb{R}^{F'} \quad (F' \ll F) \tag{3}
$$

모든 열 $i = 1,\dots,W$에 동일한 가중치 사용.[^1_1]
2. **Channel projection** (행 단위 MLP, 공유 가중치):
    - $z^t_{j\cdot} \in \mathbb{R}^W$에 대해

$$
z^c_{j\cdot} = \text{MLP}_c(z^t_{j\cdot}),\quad \text{MLP}_c: \mathbb{R}^W \to \mathbb{R}^W,\quad j=1,\dots,F' \tag{4}
$$

- 변수(채널) 간 상호정보 교환을 수행.[^1_1]
3. **Output projection** (다시 열 단위 MLP):

$$
z^o_{\cdot i} = \text{MLP}_o(z^c_{\cdot i}),\quad \text{MLP}_o: \mathbb{R}^{F'} \to \mathbb{R}^F \tag{5}
$$

- $F' \ll H,F$인 bottleneck 설계로, 긴 시퀀스에서도 channel projection 반복 횟수를 크게 줄여 FLOPs를 절감한다.[^1_1]

***

### 2.3 전체 모델 구조: LightTS

LightTS는 두 파트로 구성된다.[^1_1]

#### Part I: 각 시계열별 장·단기 패턴 추출

- 각 변수(시계열)를 독립적으로 처리(변수 간 상호작용 없음).[^1_1]
- 각 단변량 시계열(길이 $T$)에 대해

1) 연속 샘플링 $\Rightarrow X^{\text{con}}_t\in \mathbb{R}^{C\times T/C}$
2) 간격 샘플링 $\Rightarrow X^{\text{int}}_t\in \mathbb{R}^{C\times T/C}$.[^1_1]
- 각각에 IEBlock-A, IEBlock-B 적용:

$$
H^{\text{con}} \in \mathbb{R}^{F \times T/C},\quad H^{\text{int}} \in \mathbb{R}^{F \times T/C}
$$
- 각 시계열에 대해, 열 방향 선형 사상으로 $\mathbb{R}^{T/C}\to \mathbb{R}$를 적용하여 $F$-차원 feature로 요약:

$$
h^{\text{con}} \in \mathbb{R}^F,\quad h^{\text{int}} \in \mathbb{R}^F
$$
- 최종적으로 각 변수마다 길이 $2F$인 feature 벡터(연속+간격)를 얻는다.[^1_1]


#### Part II: 변수 간 상호의존성과 예측

- 모든 변수의 feature를 concat하면, 입력 행렬

$$
U \in \mathbb{R}^{2F \times N}
$$
- IEBlock-C를 적용하여

$$
\hat{Y} = \text{IEBlock-C}(U) \in \mathbb{R}^{L \times N} \tag{6}
$$

여기서 $L$은 예측 시계열 길이(예: horizon), 각 열이 변수별 예측 시퀀스이다.[^1_1]
- 이 단계에서의 channel projection이 유일하게 변수 간 상호의존성을 명시적으로 모델링하는 모듈이며, 구현상 단일 선형층만 사용된다.[^1_1]

***

### 2.4 성능 향상: 정량적 결과

#### 2.4.1 장기(long sequence) 예측

- 대상: ETTh1/ETTh2/ETTm1/Weather/Electricity 데이터셋, horizon 96–960, MSE/MAE 평가.[^1_1]
- Transformer 계열(LogTrans, Reformer, Informer, Autoformer), RNN 계열(LSTMa, LSTNet), CNN/SCINet 대비:
    - ETTh1, ETTh2, ETTm1, Electricity는 모든 horizon에서 SOTA 또는 두 번째 성능.[^1_1]
    - Weather에서는 약간 열세지만 근접.[^1_1]
- 가장 긴 horizon에서 기존 최고 대비 MSE 상대 개선 (표 2 기준):[^1_1]
    - ETTh1 (720 step): $9.21\%$
    - ETTh2 (720): $33.90\%$
    - ETTm1 (672): $34.18\%$
    - Electricity (960): $13.60\%$.[^1_1]


#### 2.4.2 단기(short sequence) 예측

- 대상: Solar-Energy, Traffic, Electricity, Exchange-Rate (단일-step, 여러 horizon), RSE/CORR 평가.[^1_1]
- Solar-Energy: 모든 horizon에서 RSE 기준 SOTA (3,6,12,24 step에서 각각 4.16%, 4.61%, 3.90%, 2.73% 개선).[^1_1]
- Traffic/Electricity/Exchange-Rate: RSE 기준 MTGNN·SCINet와 비슷한 수준, CORR에서는 데이터셋·horizon에 따라 상이.[^1_1]


#### 2.4.3 효율성과 FLOPs, 시간

- FLOPs: 가장 큰 Traffic 데이터셋에서
    - LightTS: 90M FLOPs vs MTGNN 2370M, SCINet 16348M → MTGNN 대비 약 96.2% 절감, SCINet 대비 약 99.4% 절감.[^1_1]
- 학습 시간(에폭당): Traffic에서 LightTS가 MTGNN보다 44.5배, SCINet보다 13.8배 빠름.[^1_1]
- Electricity (long sequence): Autoformer, SCINet 대비 각각 97.2%, 93.5% FLOPs 절감.[^1_1]


#### 2.4.4 강건성(robustness)

- 5개의 random seed로 MSE/MAE의 평균±표준편차를 비교하면, LightTS는 Autoformer/SCINet 대비 표준편차가 훨씬 작다.[^1_1]
- 예: ETTh1, 가장 긴 horizon에서 MSE 표준편차
    - Autoformer: $0.039$
    - SCINet: $0.013$
    - LightTS: $0.002$.[^1_1]
- 시각화(그림 3)에서도 seed 별 예측 범위(shaded area)가 가장 좁음.[^1_1]

***

### 2.5 한계와 저자 논의

- 일부 데이터셋(Weather, Traffic, Electricity, Exchange-Rate)에서는 항상 SOTA는 아니며, 특히 단기 예측에서 CORR 기준으로는 다른 모델이 더 우수한 경우가 있음을 저자들도 인정.[^1_1]
- 변수 간 상호의존성이 약한 Exchange-Rate에서는 channel projection 제거(w/o CP) 모델이 오히려 더 나은 성능을 보여, LightTS가 모든 도메인에서 자동으로 “좋은” interdependency를 학습한다고 보긴 어렵다.[^1_1]
- 구조가 단순한 만큼, 복잡한 비선형 상호작용이나 불규칙 샘플링, 비정상성에 대한 명시적 처리(예: decomposition, irregular sampling 처리)는 포함되어 있지 않다.[^1_3][^1_1]

***

## 3. 일반화 성능 관점에서의 분석

LightTS가 일반화 측면에서 가지는 장점/한계를 구조적으로 정리하면 다음과 같다.

### 3.1 일반화에 유리한 설계 요소

1. **파라미터 효율성과 병목 구조**
    - IEBlock의 bottleneck ($F'\ll H,F$) 및 단순 MLP 구조는, Transformer나 깊은 CNN에 비해 자유도(파라미터 수)를 줄여 과적합 위험을 낮춘다.[^1_1]
    - FLOPs 및 파라미터 수가 작을수록, 동일 데이터에서 모델 복잡도 대비 “effective data size”가 커져 일반화에 유리하다.[^1_1]
2. **샘플링 기반의 implicit regularization**
    - 연속/간격 샘플링은 원래 시퀀스를 여러 “뷰”로 재구성하여,
        - 국소(view): 노이즈에 덜 민감한 패턴(단기 패턴)
        - 전역(view): 유효한 장기 구조를 강조하게 만든다.[^1_1]
    - 이 과정은 일종의 feature-level augmentation/aggregation으로 작용해, 특정 시점의 노이즈에 과적합되는 것을 완화한다.[^1_1]
3. **채널 독립 처리 후 후단에서만 상호작용**
    - Part I에서 각 변수는 독립적으로 인코딩되고, Part II에서만 간단한 channel projection으로 상호작용을 학습한다.[^1_1]
    - 이는 “variable-wise representation + light cross-variable coupling” 구조로, 복잡한 GNN/attention 기반의 변수 그래프 학습에 비해 overfitting을 줄이고, 특히 데이터가 적을 때 일반화에 유리할 수 있다.[^1_4][^1_1]
4. **경험적 증거: 작은 seed-variance**
    - MSE/MAE의 표준편차가 작고, 시각적으로도 seed별 예측 분산이 작은 것은, 파라미터 landscape가 상대적으로 smooth하며, initialization/optimization 노이즈에 둔감하다는 간접 증거다.[^1_1]
    - 이는 일반적으로 테스트 분포 shift에 대한 안정성·일반화와도 어느 정도 상관이 있다.

### 3.2 일반화 한계 및 개선 여지

1. **데이터 특성에 따른 interdependency modeling 한계**
    - Exchange-Rate처럼 변수 간 상관이 약한 경우, channel projection을 강제로 사용하는 것이 오히려 해가 될 수 있음이 ablation에서 드러난다.[^1_1]
    - 즉, 현재 구조는 “언제 변수 간 상호작용을 강하게/약하게 둘지”를 자동으로 조절하는 메커니즘이 부족하다.
2. **명시적 데이터 분해·불규칙성 처리 부재**
    - Autoformer, DLinear, 최근 PatchMLP/TSMixer류는 추세/계절성 decomposition 또는 patching을 통해 비정상성, multi-scale 패턴을 명시적으로 다룬다.[^1_5][^1_6][^1_4]
    - LightTS는 샘플링으로 암묵적으로 이를 처리하지만, 도메인 shift(새로운 계절성, 다른 타임그리드)에서의 일반화는 decomposition 기반 접근보다 불리할 수 있다.
3. **데이터 크기·도메인 다양성에 대한 체계적 일반화 분석 부재**
    - 논문은 8개 표준 벤치마크에 대해 결과를 제시하지만,
        - cross-dataset transfer,
        - few-shot/low-data regime,
        - 도메인 shift(새로운 지역의 전력·교통 데이터)
등에 대한 체계적인 일반화 실험은 제공하지 않는다.[^1_1]
4. **해석 가능성과 구조적 inductive bias 부족**
    - SHAP 기반 분석으로 MTGNN과의 interdependency 상관을 보이지만, 이 자체가 “정답”은 아니며,
    - 물리적 제약, 계절·업무 주기 등의 도메인 지식을 가진 구조적 bias는 Transformer/graph 기반 모델에 비해 상대적으로 약하다.[^1_7][^1_1]

***

## 4. 2020년 이후 관련 최신 연구와의 비교 (MLP 계열 중심)

여기서는 주로 2020년 이후의 **경량 MLP 기반 다변량 시계열 예측** 연구들을 중심으로 LightTS와 비교한다.


| 연구 | 핵심 아이디어 | LightTS와의 관계 |
| :-- | :-- | :-- |
| N-BEATS (2019) [^1_8] | 순수 MLP 기반 블록 + backward residual 연결로 단변량 예측 SOTA | MLP-only로 시계열을 잘 예측할 수 있다는 전례 제공, LightTS는 이를 multivariate + sampling으로 확장.[^1_1] |
| DLinear (Are Transformers Effective for TS?) 2023 [^1_4][^1_9] | 시계열을 trend/seasonality로 분해 후 각 부분에 1-layer linear 모델 적용 | LightTS와 동일하게 “less is more” 철학, decomposition vs sampling 차이, DLinear는 극단적으로 단순한 선형 구조. |
| LightTS (본 논문) 2022 [^1_1][^1_2] | 연속/간격 샘플링 + IEBlock 기반 MLP, 경량·강건 multivariate 예측 | MLP 기반 multivariate LTSF의 초기 대표. |
| TSMixer 2023 [^1_4][^1_5] | MLP-Mixer를 시계열에 맞게 커스터마이즈, patch 기반 multivariate forecasting | LightTS와 마찬가지로 MLP-only, patching으로 시간 축 구조를 활용, online reconciliation 등 추가 설계. |
| PatchMLP 2024–25 [^1_10] | multi-scale patch + channel mixing, 노이즈 전담 모듈 포함 | LightTS의 sampling과 유사한 multi-scale 아이디어를 더 정교히 구현, 여러 벤치마크에서 Transformer 전반을 능가했다고 주장. |
| Frequency-domain MLPs 2024 [^1_6] | 시계열을 주파수 영역에서 MLP로 처리, global dependency·energy compaction 활용 | LightTS는 시간영역 샘플링 기반, 이들 연구는 일반화 향상을 위해 frequency-domain inductive bias를 추가. |
| FTMLP (Feature-Temporal MLP) 2024 [^1_11] | Feature-Temporal block으로 temporal+feature 의존성 동시 캡처, MLP-only | IEBlock과 유사한 아이디어를 더 일반화한 구조, 여러 real-world 데이터에서 SOTA 보고. |
| MixLinear 2024 [^1_12] | 0.1K 파라미터 수준의 초경량 선형 모델, 시간·주파수 도메인 모두 활용 | LightTS보다 훨씬 가벼운 extreme regime, 임베디드/엣지 장치 타깃. |

요약하면, 2020년 이후 흐름은 다음과 같다.

- 복잡한 Transformer·GNN 대신,
    - **간단한 선형/MLP 구조 + 적절한 도메인 커스터마이즈(샘플링, patch, decomposition, frequency transform)**
로도 충분히 혹은 더 잘 일반화되는 경우가 많다는 “less-is-more” 라인이 형성되었다.[^1_12][^1_6][^1_4]
- LightTS는 multivariate LTSF에서 이 라인을 초기 단계에서 명시적으로 보여준 대표적인 예로, 이후 TSMixer·PatchMLP·FTMLP·MixLinear 등 다양한 변형이 등장해 generalization·효율·도메인 적응을 더 밀어붙이고 있다.[^1_10][^1_11][^1_4][^1_12]

***

## 5. 앞으로의 연구 영향과 향후 고려점

### 5.1 이 논문의 영향

1. **“복잡한 구조가 항상 필요하지 않다”는 실증**
    - LightTS는 Transformer·GNN보다 훨씬 단순한 MLP 구조로도 멀티변량 장기 예측에서 경쟁력 있는 성능을 보이며, 이후 DLinear, TSMixer, PatchMLP, MixLinear, FTMLP 등 일련의 경량 모델 연구를 촉발하는 근거 중 하나가 되었다.[^1_11][^1_4][^1_5][^1_12][^1_1]
2. **샘플링/patch 기반 멀티스케일 설계의 정당화**
    - 연속/간격 샘플링은 이후 multi-scale patch, period-aware tokenization(예: LightGTS)의 설계 철학과도 일맥상통한다.[^1_13][^1_10]
    - “input을 적절히 재구성하면, 복잡한 attention 없이도 장·단기 패턴 학습이 가능하다”는 관점을 확립.
3. **채널(변수) 간 상호작용을 단순 구조로 구현할 수 있음을 실증**
    - 단일 선형 channel projection으로도, GNN 기반 MTGNN이 학습한 interdependency와 높은 상관을 가진 구조를 학습할 수 있음을 SHAP 기반 분석으로 보였다.[^1_1]
    - 이후 FTMLP, TSMixer 등도 MLP 기반 feature-temporal mixing 구조를 강조하게 되는 데 일조했다.[^1_4][^1_11]

### 5.2 앞으로 연구 시 고려할 점 (연구 아이디어 관점)

1. **샘플링 + 다른 inductive bias 결합**
    - LightTS의 연속/간격 샘플링을
        - 추세/계절성 decomposition(DLinear 류),[^1_4]
        - frequency-domain transform(주파수 MLP),[^1_6]
        - multi-scale patching(PatchMLP)[^1_10]
와 결합하면, 일반화 성능을 더 높일 수 있다.
    - 예: 샘플링 후 각 서브시퀀스를 frequency-domain MLP로 처리하는 hybrid 구조.
2. **데이터 적응형 샘플링/채널 프로젝션**
    - 현 구조는 $C$, 샘플링 전략, channel projection 강도가 고정이다.
    - 향후에는
        - 학습 가능한 샘플링 간격,
        - 데이터/변수별로 다른 $C$ 혹은 multi-scale $C$,
        - gating을 통한 channel projection 스킵(예: interdependency가 약한 경우)
등으로, 도메인 및 일반화에 더 유연하게 적응하는 방향이 유망하다.[^1_1]
3. **불규칙 샘플링, missing data, 도메인 shift 처리**
    - LightTS는 정규 시간 격자를 가정한다.
    - 최근 CoFormer 등은 irregular sampling에 특화된 구조를 제안하고 있으며, LightTS의 아이디어를 이러한 setting에 어떻게 이식할지(예: 샘플링을 시간 간격에 따라 비균일하게 조정) 연구할 수 있다.[^1_3]
4. **표준화된 generalization 평가 프로토콜**
    - Cross-dataset transfer, low-resource regime, distribution shift에 대한 공통 벤치마크를 정의하고, LightTS·DLinear·TSMixer·PatchMLP·FTMLP·MixLinear 등을 체계적으로 비교하는 실험이 필요하다.[^1_12][^1_11][^1_10][^1_4]
    - 특히, 경량 모델들이 실제 산업 환경(장기간 운영, 드리프트, 장치 제약)에서 얼마나 잘 일반화되는지에 대한 장기 실증 연구가 중요하다.
5. **설명가능성과 도메인 지식 통합**
    - LightTS의 SHAP 분석은 좋은 출발점이지만, 실제 도메인(에너지, 교통, 금융)의 전문가가 이해·검증할 수 있는 구조적 제약(예: 그래프 구조, 물리 법칙, 계절성 캘린더 변수 등)을 MLP 기반 경량 구조에 어떻게 접목할지에 대한 연구가 필요하다.[^1_7][^1_1]

***

정리하면, 이 논문은 “MLP + 샘플링”이라는 매우 단순한 설계로도 멀티변량 장기 예측에서 높은 성능·효율·강건성을 달성할 수 있음을 보여주었고, 이후의 다양한 경량 MLP 기반 시계열 모델 연구의 중요한 레퍼런스로 자리 잡고 있다. 앞으로는 LightTS의 아이디어를 decomposition, frequency-domain, irregular sampling, 도메인 지식 등과 결합하여 **일반화 성능과 실제 적용 가능성을 동시에 높이는 방향**이 핵심 연구 주제가 될 것이다.[^1_2][^1_5][^1_12][^1_4][^1_1]
<span style="display:none">[^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47]</span>

<div align="center">⁂</div>

[^1_1]: 2207.01186v1.pdf

[^1_2]: https://arxiv.org/abs/2207.01186

[^1_3]: https://arxiv.org/pdf/2310.11022.pdf

[^1_4]: https://arxiv.org/html/2306.09364v1

[^1_5]: https://arxiv.org/pdf/2306.09364.pdf

[^1_6]: https://axi.lims.ac.uk/paper/2311.06184

[^1_7]: https://arxiv.org/abs/2212.02567

[^1_8]: https://arxiv.org/pdf/2504.18878.pdf

[^1_9]: https://www.youtube.com/watch?v=J5Pl5a_mXfE

[^1_10]: https://arxiv.org/html/2405.13575v3

[^1_11]: https://www.sciencedirect.com/science/article/abs/pii/S0925231224011366

[^1_12]: http://arxiv.org/pdf/2410.02081.pdf

[^1_13]: https://arxiv.org/html/2506.06005v1

[^1_14]: https://www.semanticscholar.org/paper/Less-Is-More:-Fast-Multivariate-Time-Series-with-Zhang-Zhang/0ae9ea4846f52f14095e86a2cd384983f074746a

[^1_15]: https://arxiv.org/html/2402.12694v3

[^1_16]: https://arxiv.org/html/2504.00118v1

[^1_17]: https://arxiv.org/html/2601.17815v2

[^1_18]: https://arxiv.org/html/2505.20774v1

[^1_19]: https://arxiv.org/html/2502.15016v3

[^1_20]: https://arxiv.org/html/2601.22515v1

[^1_21]: https://arxiv.org/html/2404.14197v1

[^1_22]: https://arxiv.org/html/2601.09237v1

[^1_23]: https://arxiv.org/html/2601.21112v1

[^1_24]: https://arxiv.org/html/2601.21866v1

[^1_25]: https://journals.unisba.ac.id/index.php/JRS/article/view/1150

[^1_26]: https://link.springer.com/10.1007/978-3-031-08751-6_10

[^1_27]: https://ieeexplore.ieee.org/document/10143895/

[^1_28]: https://www.ssrn.com/abstract=4197041

[^1_29]: https://www.hefjournal.org/index.php/HEF/article/view/137

[^1_30]: https://pubs.aip.org/aip/acp/article/2831222

[^1_31]: https://www.mdpi.com/2071-1050/14/3/1703

[^1_32]: https://ieeexplore.ieee.org/document/10052091/

[^1_33]: https://arxiv.org/html/2409.02869v1

[^1_34]: http://arxiv.org/pdf/2312.04142.pdf

[^1_35]: https://arxiv.org/html/2411.15737v2

[^1_36]: https://arxiv.org/pdf/1901.10738.pdf

[^1_37]: https://arxiv.org/pdf/2110.00578.pdf

[^1_38]: http://arxiv.org/abs/2207.01186

[^1_39]: https://bohrium.dp.tech/paper/arxiv/aa015f7dde82501e948bf

[^1_40]: https://wepub.org/index.php/TCSISR/article/view/4164

[^1_41]: https://www.youtube.com/watch?v=cfjrvpH-DKQ

[^1_42]: https://dblp.org/rec/journals/corr/abs-2207-01186.html

[^1_43]: https://www.reddit.com/r/boyslove/comments/ux019l/light_taiwanese_bl_updates/

[^1_44]: https://blog.csdn.net/qq_33431368/article/details/133937569

[^1_45]: https://www.wunschlicht.com/en-gb/brands/less-n-more.html

[^1_46]: https://blog.csdn.net/qq_45834106/article/details/144584422

[^1_47]: https://arxiv.org/pdf/2601.09237.pdf

