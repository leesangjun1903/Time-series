# UnSupervised Anomaly Detection on Multivariate Time Series

- 이 논문은 다변량 시계열에서 **라벨 없이** 이상 탐지를 수행하기 위해, 두 개의 오토인코더를 **적대적(adversarial)** 으로 학습시키는 USAD(UnSupervised Anomaly Detection) 모델을 제안한다.[^1_1][^1_2]
- USAD는 기존 RNN·GAN 기반 방법보다 **훈련 속도와 안정성**이 높으면서도 SWaT, WADI, SMD, SMAP, MSL 등 5개 공개 데이터셋과 Orange 내부 데이터에서 **동급 또는 우수한 F1 성능**을 보이는 것을 보인다.[^1_3][^1_1]
- 주요 기여는 (1) 오토인코더+GAN 아이디어를 결합한 두 단계 적대적 학습 구조, (2) 대규모 산업 데이터에서의 **스케일·에너지 효율** 검증, (3) 알파–베타 가중치를 통한 **감도 조절 가능 스코어링** 및 노이즈에 대한 강건성 분석이다.[^1_1][^1_3]

***

## 2. 문제 정의, 방법(수식), 모델 구조, 성능과 한계

### 2.1 해결하고자 하는 문제

- 입력: 다변량 시계열

$$
T = \{x_1, \dots, x_T\},\quad x_t \in \mathbb{R}^m
$$

를 **정상 샘플만 포함한다고 가정**하고(unsupervised), 길이 $K$의 슬라이딩 윈도우

$$
W_t = \{x_{t-K+1}, \dots, x_t\}
$$

를 구성해 학습한다.[^1_1]
- 목표: 새로운 윈도우 $\hat W_t$에 대해 이상 점수 $A(\hat W_t)$를 계산하고, 임계값 $\lambda$와 비교해 $y_t \in \{0,1\}$의 이상 라벨을 부여하는 것(이상 구간(point-adjust) 수준으로 평가).[^1_3][^1_1]
- 제약:
    - 차원이 크고 센서 수가 많아 **kNN, K-means, One-Class SVM** 등 전통 기법은 차원의 저주로 성능이 급락.[^1_1]
    - RNN·GAN 기반 SOTA(OmniAnomaly, MSCRED, DAGMM 등)는 **훈련 시간이 길고 에너지 소비가 크며**, 산업 배치에서 재훈련과 안정성 문제가 있다.[^1_4][^1_3][^1_1]

USAD는 “**빠르고 안정적인 비지도 다변량 시계열 이상 탐지**”가 필요하다는 실무 요구(Orange Green AI, 대규모 IT 모니터링)를 직접 겨냥한다.[^1_1]

***

### 2.2 제안 방법: 학습 목적 함수와 수식

#### (1) 기본 오토인코더

- 일반 AE의 재구성 손실:

$$
L_{\text{AE}} = \| X - AE(X) \|_2^2,\quad AE(X) = D(E(X))
$$

로 정상 데이터에 대해 재구성 오차를 최소화한다.[^1_1]
- AE 기반 이상 탐지는 재구성 오차를 이상 점수로 사용하지만, **정상과 근접한 약한 이상(mild anomaly)** 의 경우 오차가 작아 탐지에 실패한다는 한계를 가진다.[^1_5][^1_1]


#### (2) USAD의 핵심: 두 개의 AE와 2단계 적대적 학습

USAD는 하나의 인코더 $E$와 두 개의 디코더 $D_1, D_2$로 구성된 두 오토인코더:

$$
AE_1(W) = D_1(E(W)), \quad AE_2(W) = D_2(E(W))
$$

를 공유 인코더 구조로 결합한다.[^1_1]

**1단계 (순수 AE 학습)**

- 각 윈도우 $W$에 대해 두 AE 모두 정상 재구성을 학습:

$$
L_{AE_1}^{(1)} = \| W - AE_1(W)\|_2^2,\quad
L_{AE_2}^{(1)} = \| W - AE_2(W)\|_2^2
$$

로 초기화한다.[^1_1]

**2단계 (적대적 학습)**

- $AE_1$이 출력한 재구성 $AE_1(W)$를 다시 인코딩–디코딩하여:

$$
AE_2(AE_1(W)) = D_2(E(AE_1(W)))
$$

를 만들고,
    - $AE_1$의 목표: $AE_2(AE_1(W))$가 원본 $W$와 가깝도록 만들어 ** $AE_2$ 를 속이기** (오차 최소).
    - $AE_2$의 목표: $AE_1(W)$에서 온 입력에 대해서는 재구성 오차를 **키워서 구분**하기 (오차 최대).
- 미니맥스 목적:

$$
\min_{AE_1} \max_{AE_2} \, \| W - AE_2(AE_1(W)) \|_2^2
$$

에 해당한다.[^1_1]

**두 손실의 통합(에폭에 따라 비중 변화)**

훈련 에폭을 $n$이라 하면, 에폭이 증가할수록 AE 학습에서 적대적 학습으로 **점진적으로 비중을 이동**한다:[^1_1]

$$
L_{AE_1} =
\frac{1}{n}\|W - AE_1(W)\|_2^2 +
\left(1-\frac{1}{n}\right)\|W - AE_2(AE_1(W))\|_2^2
$$

$$
L_{AE_2} =
\frac{1}{n}\|W - AE_2(W)\|_2^2 -
\left(1-\frac{1}{n}\right)\|W - AE_2(AE_1(W))\|_2^2
$$

- 초기에 $1/n$이 커서 **AE 재구성이 우세** → 안정적인 초기화.
- 후반에는 적대항의 비중이 커지며, $AE_1$는 정상일 때만 잘 속이고, $AE_2$는 $AE_1$가 재구성한 이상 패턴에 **민감한 검출자**로 변한다.[^1_3][^1_1]


#### (3) 이상 점수와 감도 조정

테스트 윈도우 $\hat W$에 대해:

- 1차 재구성: $\hat W'_1 = AE_1(\hat W)$
- 2차 재구성: $\hat W''_2 = AE_2(\hat W'_1)$

이상 점수:

$$
A(\hat W) = \alpha \|\hat W - \hat W'_1\|_2^2 +
\beta \|\hat W - \hat W''_2\|_2^2,\quad \alpha+\beta = 1
$$

으로 정의한다.[^1_1]

- $\alpha$ ↑, $\beta$ ↓: $AE_1$ 재구성에 더 의존 → **보수적(낮은 감도)**, FP 감소, TP 약간 감소.[^1_1]
- $\alpha$ ↓, $\beta$ ↑: $AE_2$ 적대 재구성 오차 비중 ↑ → **공격적(높은 감도)**, TP 증가·FP도 증가.
- 단일 학습 모델로 $(\alpha,\beta)$만 바꿔서 **여러 감도 프로파일**을 구성할 수 있다는 점이 산업 배치에서 중요한 장점으로 강조된다.[^1_1]

***

### 2.3 모델 구조 (네트워크 아키텍처)

- 입력: 윈도우 길이 $K$와 변수 수 $m$에 대해 입력 차원은 $K \times m$ (벡터로 펼쳐서 사용).[^1_1]
- 인코더 $E$: fully-connected 3-layer MLP
    - Linear: input size $\to$ input/2 → ReLU
    - Linear: input/2 $\to$ input/4 → ReLU
    - Linear: input/4 $\to$ latent size $z$ → ReLU.[^1_1]
- 디코더 $D_1, D_2$: 동일 구조의 MLP
    - Linear: $z \to$ input/4 → ReLU
    - Linear: input/4 $\to$ input/2 → ReLU
    - Linear: input/4 $\to$ input → Sigmoid.[^1_1]
- 하이퍼파라미터 예:
    - SWaT: $K=12$, latent dim $m_z=40$, downsampling=5, 70 epochs.[^1_1]
    - SMD, SMAP, MSL에 대해서도 윈도우=5, latent dim을 변수 수와 유사하게 설정.[^1_1]

**특징**:

- 순수 MLP 기반이어서 RNN/LSTM·CNN보다 **경량·고속**이며, GPU에서 초당 수만 윈도우를 처리할 수 있는 수준의 학습 시간(USAD vs OmniAnomaly: 평균 500배 이상 빠름)을 보여준다.[^1_3][^1_1]

***

### 2.4 성능 향상 (비교 실험)과 한계

#### (1) 공개 데이터셋 결과

5개 공개 데이터셋: SWaT, WADI, SMD, SMAP, MSL에 대해 Isolation Forest, 순수 AE, LSTM-VAE, DAGMM, OmniAnomaly와 비교한다.[^1_3][^1_1]

- SWaT/WADI (공격 포함 산업 제어 시스템):
    - point-adjust 기준 F1에서 SWaT 0.846, WADI 0.43 수준으로 OmniAnomaly 및 LSTM-VAE보다 우위.[^1_3][^1_1]
    - non point-adjust에서도 SWaT, WADI 모두 SOTA 혹은 동급.
- SMD/SMAP/MSL (서버, 위성, 로버 데이터):
    - USAD는 SMD에서 OmniAnomaly와 비슷한 F1, SMAP/MSL에서는 약간 더 높은 F1을 기록.[^1_1]
- 전체 평균 (point-adjust 기준):
    - USAD 평균 F1=0.79(±0.18)로 OmniAnomaly(0.78)를 약간 상회하고, IF·DAGMM·기본 AE·LSTM-VAE보다 높다.[^1_1]

**훈련 시간**:

- 동일 GPU(1080Ti)에서 1 epoch 기준 OmniAnomaly 대비 가속:
    - SMD: 87분 vs 0.06분 → 약 1331배
    - SWaT: 13분 vs 0.06분 → 216배
    - 평균 약 547배 가속을 보고한다.[^1_1]


#### (2) 파라미터·일반화 분석

- 다운샘플링 비율: 1, 5, 10, 20, 50 배에서 F1이 거의 변하지 않음 → 상당한 **시간축 축소에도 일반화가 유지**.[^1_1]
- 윈도우 길이 $K$: 5–100에서 $K=10$ 근처에서 최적, 작은 윈도우는 빠른 탐지·짧은 이상에 유리, 큰 윈도우는 긴 이상에 유리.[^1_1]
- 잠재 차원 $m_z$: 너무 작으면 정보 손실로 underfit, 너무 크면 memorization으로 overfit; 중간 영역에서는 F1이 안정적 → **표현 용량이 넓은 범위에서 robust**.[^1_1]
- 학습 데이터 오염(노이즈/이상 비율) 실험:
    - 정상만 있다고 가정하고 학습하지만, 실제로는 최대 10% 수준의 이상이 섞여 있어도 F1이 크게 떨어지지 않음을 보임.[^1_1]
    - 30% 이상 오염 시 FP가 증가하면서 F1 감소 → “현실적으로 그 정도의 미검출 이상 비율은 거의 없다”고 논의.[^1_1]


#### (3) Orange 내부 데이터셋: 실용성

- 33개 지표(기술 27, 비즈니스 6), 32일 학습 / 60일 테스트에서, domain expert가 라벨링한 사건들을 대상으로 F1≈0.69, Precision≈0.74, Recall≈0.64를 달성.[^1_1]
- 특히, 실제 운영팀이 24시간 뒤에야 발견한 광고 설정 오류 사건을 30분 이내에 자동 탐지하는 사례를 제시.[^1_1]


#### (4) 한계 및 논문에서 언급된 이슈

- 완전히 unsupervised라 **임계값 $\lambda$ 선택**은 여전히 어려우며, 논문에서는 “최적 F1을 주는 threshold를 grid search로 사후적으로 선택”하는 실험 프로토콜을 사용한다.[^1_1]
- 정상만 있는 학습 구간을 확보해야 하지만, 실무에서는 “**충분히 깨끗한 기간을 찾는 것 자체가 어렵다**”는 점을 Orange 사례에서 언급하며, 배포 시 데이터 인프라 측면의 과제를 지적한다.[^1_1]
- 모델 구조 자체는 단순 MLP이기 때문에, **복잡한 시계열 구조(긴 시점 의존·비선형 상관 구조)를 충분히 활용하지 못한다**는 잠재적 한계가 있으며, 이후 연구에서 Transformer·GNN·Flow 등 더 강력한 시계열 표현이 도입되고 있다.[^1_6][^1_4]

***

## 3. 모델 일반화 성능 향상 가능성 (논문 내용+이후 연구 관점)

### 3.1 논문 내부에서의 일반화 관련 결과

1. **하이퍼파라미터에 대한 강건성**
    - 다운샘플링 비율, 윈도우 길이, 잠재 차원 등 주요 하이퍼파라미터를 넓은 범위에서 스윕했을 때 F1이 급격히 나빠지지 않고, 특히 중간값 영역에서 plateau 형태를 보인다 → **튜닝에 덜 민감한 모델**로 해석할 수 있다.[^1_1]
2. **노이즈·오염된 정상 데이터에 대한 강건성**
    - 학습 데이터에서 최대 약 10%까지 Gaussian 노이즈/이상 샘플을 주입해도 recall이 크게 떨어지지 않고, precision이 점진적으로 감소하는 형태를 보여, **약간의 라벨 누락·오염 하에서도 정상 패턴을 포착**한다.[^1_1]
    - 이는 AE+adversarial 설계가 “복잡한 정상 패턴을 먼저 모델링한 뒤, 그 주변에서 재구성 오차를 증폭시키는 방향”으로 작동하여, 일부 이상이 섞여 있어도 전체 분포를 정상으로 오인하지 않기 때문이라고 해석할 수 있다.[^1_1]
3. **2단계 학습의 ablation 결과**
    - AE only vs adversarial only vs full(USAD) 비교 시, full이 평균 F1에서 AE-only 대비 약 5.9%p, adversarial-only 대비 약 24%p 개선된다.[^1_1]
    - AE-only는 약한 이상에 둔감(under-sensitive), adversarial-only는 학습이 불안정(일반화 저하)한 반면, 두 단계를 결합하면 **안정성과 민감도를 모두 확보**해 보다 일반화된 이상 점수를 얻는다는 것이 논문의 결론이다.[^1_1]
4. **감도 파라미터 $(\alpha,\beta)$를 통한 활용 측면의 ‘일반화’**
    - 하나의 모델로 여러 threshold와 $(\alpha,\beta)$ 조합을 사용해 **다양한 운영 시나리오(매니저용 vs 운영자용)를 지원**하는 것은, 모델 자체의 통계적 일반화라기보다 “운영 환경에서의 재사용·도메인 일반화”에 가깝다.[^1_1]

### 3.2 2020년 이후 관련 연구와의 비교: 일반화 관점

USAD 이후, 다변량 시계열 이상 탐지(MTSAD) 분야에서는 **더 강력한 표현과 오염된 학습 데이터, threshold-free 검출**을 다루는 방향으로 발전했다.[^1_7][^1_8][^1_4][^1_6]

1. **Transformer·그래프 구조 학습 기반 방법**
    - GTA, GDN, TranAD, MEMTO, GCAD, AMAD 등은 센서 간 구조를 명시적으로 학습(그래프·Granger causality)하거나, Transformer/self-attention으로 **장기 의존성과 전역 상관**을 모델링해 복잡한 도메인에도 잘 일반화하려 한다.[^1_9][^1_10][^1_11][^1_12][^1_13][^1_14]
    - 이들 연구는 USAD가 단순 MLP 구조로 인한 표현력 제한을 갖는다는 점을 보완하지만, **모델 크기·데이터 요구·훈련 시간**은 USAD보다 크다.[^1_4][^1_6]
2. **Flow·밀도 추정 기반 unsupervised 방법**
    - MTGFlow 등은 정상 분포 전체의 밀도를 추정하고 test density로 이상을 판별해, **일부 오염된 데이터에서도 정상/이상 분리**를 시도한다.[^1_15]
    - 이 역시 USAD가 다루는 “정상만 있다고 가정” 한계를 완화하는 방향으로, 오염 환경에서의 일반화 성능을 강화한다.
3. **오픈셋·semi-supervised TSAD**
    - MOSAD, MtsCID 등은 일부 라벨된 이상을 활용해 **보지 못한 유형의 이상(open-set anomaly)** 까지 잘 탐지하려는 방법으로, “새로운 유형의 이상”에 대한 일반화에 초점을 둔다.[^1_16][^1_17][^1_18]
4. **대규모 벤치마크 및 메타 평가**
    - Mejri et al., Darban et al., Li et al. 등의 survey·benchmark에서는 USAD를 포함한 여러 모델을 통일된 프로토콜로 재평가하며,
        - point-adjust의 편향,
        - threshold 선택의 중요성,
        - 모델 안정성·크기·훈련 시간까지 고려한 **실질적 일반화 평가**의 필요를 강조한다.[^1_19][^1_20][^1_7][^1_6][^1_4]

**정리하면**, USAD는 “단순 구조+adversarial 학습으로 약한 이상까지 잡는 빠른 모델”이라는 포지션을 갖고, 이후 연구들은 이를 토대로 (1) 더 강한 표현, (2) 오염·부분 라벨 환경, (3) 대규모 벤치마크에서의 일반화 평가로 확장하는 흐름을 보인다.[^1_7][^1_6][^1_4]

### 3.3 USAD 기반 일반화 성능 향상 아이디어 (연구 관점)

연구자 입장에서 USAD를 출발점으로 일반화 성능을 높이려면:

1. **표현력 향상: 구조만 교체**
    - 인코더·디코더를 현재의 MLP에서
        - CNN-Temporal Conv,
        - LSTM/GRU (짧은 시계열에 한정),
        - Transformer (local attention, AutoMasked attention 등),
        - GNN/graph transformer (센서 관계가 중요할 때)
로 교체하되, **2단계 손실 구조(식 (7)–(9))는 유지**하는 방식으로 발전시킬 수 있다.[^1_10][^1_12][^1_13]
2. **Contaminated training 데이터에 대한 이론·알고리즘 개선**
    - 현재 논문은 경험적으로 “10% 오염까지는 버틴다”고만 보고하므로,
        - robust loss(예: Huber, quantile reconstruction),
        - self-labeling으로 이상 후보를 down-weight,
        - feature decomposition (FDAE)처럼 잠재 공간을 정상/이상 성분으로 분해
등으로 **오염 환경에서 정상 분포를 더 정확히 추정**하는 방향이 유망하다.[^1_21][^1_22][^1_15]
3. **threshold·감도 선택의 자동화**
    - 무라벨 환경에서 $\lambda, \alpha,\beta$ 선택을 위한
        - extreme value theory 기반 adaptive threshold,
        - unsupervised calibration (예: 최대 허용 false alarm rate 제약 하에서 quantile 기반 $\lambda$ 추정),
        - 앙상블 기반 불확실성 추정을 활용한 threshold-free scoring
연구가 필요하다.[^1_23][^1_24][^1_6]
4. **도메인 적응·transfer learning**
    - Orange 사례처럼 시스템 구성·분포가 시간이 지나며 바뀌는 경우,
        - 여러 기간/시스템에서 학습한 USAD 인코더를 공유하고 디코더만 도메인별로 미세조정하는 multi-head 구조,
        - adversarial domain adaptation으로 기간 간 분포 차이를 줄이는 방법
을 통해 **도메인 간 generalization**을 높일 수 있다.[^1_8][^1_4]

***

## 4. 앞으로의 연구에 미치는 영향과 향후 고려 사항

### 4.1 이 논문의 영향

1. **AE+Adversarial 구조의 표준화**
    - USAD는 “오토인코더 기반 이상 탐지에 적대적 아이디어를 단순하게 녹이는” 대표 사례로, 이후 FDAE·DAEMON·HybridAD 등 **두 디코더/멀티헤드 AE·adversarial reconstruction** 구조를 쓰는 연구에 영향을 주었다.[^1_25][^1_22][^1_4]
2. **실용성과 효율성에 대한 기준 제시**
    - OmniAnomaly 같은 복잡 모델과 유사한 F1을 **수백 배 빠른 훈련 시간**으로 달성하며, “실제 산업 환경에서는 성능뿐 아니라 훈련 비용과 재훈련 안정성이 핵심”이라는 메시지를 명확히 했다.[^1_3][^1_1]
    - 이후 survey·benchmark 논문들에서 모델 크기·훈련 시간·안정성을 함께 보고하는 것이 점차 일반화됐다.[^1_19][^1_7][^1_4]
3. **다변량 시계열 이상 탐지 벤치마크의 정착**
    - SWaT, WADI, SMD, SMAP, MSL을 한 세트로 평가하는 프로토콜은 이후 TranAD, MTGFlow, MEMTO 등 많은 후속 연구의 표준 벤치마크 구성에 영향을 미쳤다.[^1_11][^1_12][^1_15][^1_4]

### 4.2 앞으로 연구 시 고려할 점 (연구자 관점)

연구자로서 후속 연구를 설계할 때 고려할 주요 포인트는:

1. **평가 프로토콜의 엄밀성**
    - point-adjust는 segment-level 성능을 과대평가할 수 있으므로,
        - point-level, segment-level, event-level 지표를 모두 보고,
        - threshold 튜닝에 ground truth를 직접 사용하지 않는 **realistic setting**을 설계해야 한다.[^1_6][^1_7][^1_4]
2. **오염된 정상 데이터와 domain shift**
    - 완전한 정상 데이터 구간을 찾기 어렵다는 것은 Orange 사례에서 이미 드러났고, 대규모 실무 데이터에서 흔한 문제다.[^1_19][^1_1]
    - 따라서,
        - contaminated-unsupervised 가정,
        - online/continual learning,
        - concept drift 탐지와 결합
을 전제로 한 방법 설계가 중요하다.[^1_24][^1_15][^1_8]
3. **표현력 vs 효율의 트레이드오프**
    - Transformer·GNN·Flow 모델은 표현력은 높지만 비용이 크다.[^1_4][^1_6]
    - USAD 수준의 간단한 인코더–디코더에
        - 경량 attention,
        - low-rank factorization,
        - pruning/quantization
같은 기법을 조합해 **“industrial-grade” 효율성을 유지하면서도 표현력을 조금씩 확장**하는 방향이 현실적인 타협이다.
4. **설명 가능성과 root cause 분석**
    - USAD는 윈도우 수준의 scalar 이상 점수에 초점을 두기 때문에, “어떤 변수·시간이 문제인지”를 알려주는 해석 가능성은 제한적이다.[^1_1]
    - InterFusion, GNN 기반 기법, memory-based/attention-based 해석 기법과 결합하여,
        - 변수별 또는 서브시스템별 기여도,
        - 이상 유형 분류
를 제공하는 **해석 가능한 USAD 변형**이 앞으로 중요한 연구 방향으로 보인다.[^1_26][^1_27][^1_10][^1_25]
5. **새로운 벤치마크와 실제 배치 실험**
    - 최근 PATH, 다양한 TSAD benchmark들이 공개되면서, 기존 SWaT·WADI 중심 평가에서 **더 다양한 도메인·이상 유형·오염 수준**을 포함한 벤치마크로 확장되고 있다.[^1_20][^1_24][^1_8]
    - 향후 연구는 벤치마크 점수뿐 아니라,
        - real-time 제약,
        - 운영 팀과의 인터랙션(알람 피로, false alarm cost),
        - 장기 유지보수 비용
까지 포괄하는 “end-to-end deployment study”를 동반해야 할 것이다.

요약하면, USAD는 “간단하지만 잘 작동하는” AE+adversarial 프레임워크로서, 비지도 다변량 시계열 이상 탐지 연구의 중요한 출발점이 되었고, 이후 연구들은 이 구조를 더 강한 표현, 더 어려운 환경(오염·open-set), 더 엄밀한 평가로 확장하는 방향으로 진화해 왔다.[^1_7][^1_6][^1_4][^1_1]
<span style="display:none">[^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48]</span>

<div align="center">⁂</div>

[^1_1]: 3394486.3403392.pdf

[^1_2]: https://dl.acm.org/doi/10.1145/3394486.3403392

[^1_3]: https://www.eurecom.fr/publication/6271/download/data-publi-6271_1.pdf

[^1_4]: https://arxiv.org/pdf/2211.05244.pdf

[^1_5]: https://hwk0702.github.io/treatise review/2021/02/15/USAD/

[^1_6]: https://www.sciencedirect.com/science/article/abs/pii/S1566253522001774

[^1_7]: https://arxiv.org/pdf/2212.03637.pdf

[^1_8]: https://pubmed.ncbi.nlm.nih.gov/39796981/

[^1_9]: https://ieeexplore.ieee.org/document/9497343/

[^1_10]: https://ojs.aaai.org/index.php/AAAI/article/view/16523

[^1_11]: http://arxiv.org/pdf/2312.02530.pdf

[^1_12]: https://arxiv.org/pdf/2201.07284.pdf

[^1_13]: https://arxiv.org/html/2504.06643v3

[^1_14]: https://arxiv.org/html/2501.13493v1

[^1_15]: https://arxiv.org/html/2312.11549v2

[^1_16]: https://arxiv.org/html/2310.12294v1

[^1_17]: https://arxiv.org/pdf/2310.12294.pdf

[^1_18]: https://dl.acm.org/doi/10.1145/3696410.3714941

[^1_19]: https://www.sciencedirect.com/science/article/pii/S0957417424017895

[^1_20]: https://github.com/johnpaparrizos/tsadsurvey

[^1_21]: https://arxiv.org/pdf/2108.03585.pdf

[^1_22]: https://dl.acm.org/doi/10.1145/3529836.3529924

[^1_23]: https://ieeexplore.ieee.org/document/9671776/

[^1_24]: https://arxiv.org/pdf/2411.13951.pdf

[^1_25]: https://www.semanticscholar.org/paper/DAEMON:-Unsupervised-Anomaly-Detection-and-for-Time-Chen-Deng/613dcb175b42dbabfd30f7be230c0918c7f1aa7e

[^1_26]: https://dl.acm.org/doi/10.1145/3447548.3467075

[^1_27]: http://arxiv.org/pdf/2410.22735.pdf

[^1_28]: https://dl.acm.org/doi/10.1145/3447548.3467174

[^1_29]: https://www.semanticscholar.org/paper/35be7364025ef567c7be0624e06356f5d0871fd5

[^1_30]: https://ieeexplore.ieee.org/document/9503373/

[^1_31]: https://www.mdpi.com/1099-4300/23/11/1466

[^1_32]: https://linkinghub.elsevier.com/retrieve/pii/S0167404822000517

[^1_33]: https://ieeexplore.ieee.org/document/9574505/

[^1_34]: https://arxiv.org/pdf/2201.04792.pdf

[^1_35]: https://arxiv.org/pdf/2210.09693.pdf

[^1_36]: https://dl.acm.org/doi/pdf/10.1145/3611643.3613896

[^1_37]: https://arxiv.org/html/2503.23060v1

[^1_38]: https://arxiv.org/html/2506.20574v1

[^1_39]: https://arxiv.org/html/2308.12563v5

[^1_40]: https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0303890

[^1_41]: https://arxiv.org/html/2211.05244v3

[^1_42]: https://arxiv.org/abs/2211.05244

[^1_43]: https://arxiv.org/html/2501.15196

[^1_44]: https://github.com/finloop/usad-on-ucr-data

[^1_45]: https://www.sciencedirect.com/science/article/abs/pii/S0925231224005629

[^1_46]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11156414/

[^1_47]: https://velog.io/@d9249/USAD-UnSupervised-Anomaly-Detection-on-Multivariate-Time-Series-ayyvdoi7

[^1_48]: https://www.youtube.com/watch?v=gCleQ9JxibI

