# A Novel Hyperdimensional Computing Framework for Online Time Series Forecasting on the Edge

# 1. 핵심 주장과 주요 기여 (요약)

이 논문은 비선형 시계열의 온라인 예측 문제를 “고차원 선형 회귀 문제”로 재정식화한 TSF‑HD 프레임워크(AR‑HDC, Seq2Seq‑HDC)를 제안한다. 저자들은 학습 가능한 하이퍼디멘셔널 인코더와 선형 회귀기를 온라인 공학습(co‑training)하여 개념 드리프트와 태스크 시프트가 있는 스트림에서도 높은 정확도·낮은 지연·낮은 전력으로 동작하는 엣지 지향 온라인 TSF 모델을 만든다. 8개 실세계 데이터셋과 합성 개념 드리프트 데이터에서 FSNet, OnlineTCN, ER, DER++, Informer, SCINet, NLinear, GBRT 등을 상회하는 단기·장기 예측 성능과 엣지 디바이스 상의 지연·전력 효율을 보이는 것이 핵심 기여이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c6c4d91e-a095-4e82-bd0d-73dbea1e2906/2402.01999v1.pdf)

***

## 2. 문제 정의, 제안 방법(수식), 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

- 기존 **오프라인** 딥 TSF 모델: 개념 드리프트/태스크 시프트가 있는 스트림에서 재학습 없이는 일반화가 어렵다. [arxiv](https://arxiv.org/pdf/2004.10240.pdf)
- 기존 **온라인** 딥 TSF 모델(FSNet, OnlineTCN 등): 고비용(깊은 네트워크, 복잡한 학습 루프)이라 엣지 디바이스에 배치가 어렵다. [arxiv](https://arxiv.org/abs/2202.11672)
- 단순 **선형** 모델(NLinear 등)은 장기 TSF에서 surprisingly strong 하지만, 비선형 구조를 충분히 활용하지 못하는 한계가 있다. [arxiv](https://arxiv.org/pdf/2304.08424.pdf)

이 논문은 “고비용 온라인 딥 모델 ↔ 저비용 선형 모델” 간의 **성능–오버헤드 트레이드오프**를 깨는, 엣지‑친화적 온라인 다변량 TSF 방식을 제안한다. [arxiv](https://arxiv.org/html/2402.01999v1)

***

### 2.2 제안하는 방법: 하이퍼디멘셔널 회귀 및 손실

#### (1) 문제 세팅

입력 시계열 $\(X^* \in \mathbb{R}^{N \times L}\)$ (변수 수 $\(N\)$ , 길이 $\(L\))$ 와 고정된 look‑back $\(T\)$ 가 있을 때, 시점 $\(t\)$ 에서  
과거 구간  

$$
X_{t-T+1:t} = [x_{t-T+1},\dots,x_t] \in \mathbb{R}^{N \times T}
$$  

로부터 향후 구간  

$$
X_{t+1:t+\tau} = [x_{t+1},\dots,x_{t+\tau}] \in \mathbb{R}^{N \times \tau}
$$  

을 온라인으로 예측하는 것이 목표이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c6c4d91e-a095-4e82-bd0d-73dbea1e2906/2402.01999v1.pdf)

데이터는 분포가 시점에 따라 변하는 스트림이며, “언제 태스크가 바뀌는지”에 대한 정보 없는 **task‑free online learning** 시나리오로 설정된다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c6c4d91e-a095-4e82-bd0d-73dbea1e2906/2402.01999v1.pdf)

#### (2) 하이퍼디멘셔널 인코더

입력 $\(X \in \mathbb{R}^{N \times T}\)$ 를 $\(D \gg T\)$ 차원의 하이퍼공간 $\(H\)$ 로 매핑하는 선형+ReLU 인코더를 사용한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c6c4d91e-a095-4e82-bd0d-73dbea1e2906/2402.01999v1.pdf)

$$
H(X) = \mathbf{1}_{X > 0}\,[X W_e + b_e]
$$  

여기서 $\(W_e \in \mathbb{R}^{T \times D}\), \(b_e \in \mathbb{R}^{D}\)$ 이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c6c4d91e-a095-4e82-bd0d-73dbea1e2906/2402.01999v1.pdf)

- $\(\mathbf{1}_{X > 0}\)$ 는 ReLU를 의미한다.  
- Johnson–Lindenstrauss lemma에 기반해, 무작위 투영에서는 거리 보존과 (거의) 직교성이 보장되는데, 저자는 **학습 가능한** $\(W_e\)$ 를 사용하면서도 거리 보존성과 행 간 직교성이 유지됨을 실험적으로 보인다. [arxiv](https://arxiv.org/html/2402.01999v1)

즉, 저차원에서 비선형인 함수도 $\(H(\cdot)\)$ 를 통해 고차원에서는 “가까운 선형 근사”가 가능하며, 이 위에서 선형 회귀를 수행한다는 것이 핵심 아이디어다. [arxiv](https://arxiv.org/html/2402.01999v1)

#### (3) HD 선형 회귀기 및 손실

단일 스텝 $(\(\tau=1\))$ 예측의 경우, 시점 $\(t\)$ 에서 인코딩된 입력 $\(H(X_{t-T+1:t})\)$ 로부터 예측 $\(\tilde{x}_{t+1}\)$ 을 생성한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c6c4d91e-a095-4e82-bd0d-73dbea1e2906/2402.01999v1.pdf)

HD 회귀는 회귀 하이퍼벡터(또는 행렬) $\(W_r\)$ , 바이어스 $\(b_r\)$ 를 사용하며, Huber loss를 최소화하도록 온라인으로 학습된다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c6c4d91e-a095-4e82-bd0d-73dbea1e2906/2402.01999v1.pdf)

Huber loss $(\(\tau=1\))$ 는

$$
\Delta x_{t+1} = x_{t+1} - \tilde{x}_{t+1}
$$  

일 때  

$$
L_H(x_{t+1}, \tilde{x}_{t+1}) =
\begin{cases}
\frac{1}{2}\|\Delta x_{t+1}\|_2^2, & \text{if } |\Delta x_{t+1}| \le 1 \\
\|\Delta x_{t+1}\|_1 - \frac{1}{2}, & \text{otherwise}
\end{cases}
$$  

전체 손실은 L2 정규화를 포함해  

$$
L(x_{t+1}, \tilde{x}_{t+1}) = L_H(x_{t+1}, \tilde{x}_{t+1}) + R(W_e, W_r, b_e, b_r)
$$  

로 정의되며, $\(R(\cdot)\)$ 는 L2 정규화 항이다. 데이터는 온라인 환경을 가정하여 표준화/정규화 없이 원본 스케일로 사용되며, 이때 L2 기반 손실만 사용할 경우 폭주 위험이 있어 Huber loss를 택한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c6c4d91e-a095-4e82-bd0d-73dbea1e2906/2402.01999v1.pdf)

다중 스텝 $(\(\tau > 1\))$ 에서는 예측 시퀀스 $\(\tilde{X}_{t+1:t+\tau}\)$ 에 대해 시퀀스 전체 손실을 정의한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c6c4d91e-a095-4e82-bd0d-73dbea1e2906/2402.01999v1.pdf)

#### (4) HD 추론: 단일 스텝 및 시퀀스

인코딩 이후 단일 스텝 예측은  

$$
\tilde{x}_{t+1} = R(H(X_{t-T+1:t}))
$$  

로 계산되며, 구성 요소별로는  

$$
(\tilde{x}_{t+1})_i = \langle W_r, (H(X_{t-T+1:t}))_i \rangle + b_r
$$  

형태를 가진다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c6c4d91e-a095-4e82-bd0d-73dbea1e2906/2402.01999v1.pdf)

시퀀스‑투‑시퀀스 경우,  

$$
h = H(X_{t-T+1:t})
$$  

로부터  

$$
\tilde{X}_{t+1:t+\tau} = R(h) = h W_r + b_r
$$  

를 사용한다. 여기서 $\(W_r \in \mathbb{R}^{D \times \tau}\), \(b_r \in \mathbb{R}^{\tau}\)$ 이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c6c4d91e-a095-4e82-bd0d-73dbea1e2906/2402.01999v1.pdf)

***

### 2.3 모델 구조: AR‑HDC와 Seq2Seq‑HDC

두 프레임워크 모두 “H(·) 인코더 + 선형 HD 회귀기 R(·)”를 공유하고, 온라인에서 **인코더와 회귀기를 공학습**한다. [arxiv](https://arxiv.org/html/2402.01999v1)

#### (1) AR‑HDC (Autoregressive HDC)

- 기본 아이디어: 고전 AR 모델의 “가중합”을 HD 인코더+회귀로 대체. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c6c4d91e-a095-4e82-bd0d-73dbea1e2906/2402.01999v1.pdf)
- 시점 $\(t\)$ 에서 미래 $\(\tau\)$ 스텝을 하나씩 예측:  
  1. $\(X \leftarrow X_{t-T+1:t}\)$ .  
  2. for $\(i=1,\dots,\tau\)$ :  
     - $\(\tilde{x}_{t+i} \leftarrow R(H(X))\)$ .  
     - look‑back에서 가장 오래된 항 제거 후 $\(\tilde{x}_{t+i}\)$ 삽입.  
  3. 실제 $\(x_{t+1:t+\tau}\)$ 가 관측되면, 모든 $\(i\)$ 에 대해 $\(L(\tilde{x}\_{t+i}, x_{t+i})\)$ 로 OGD+AdamW로 $\(W_e, b_e, W_r, b_r\)$ 업데이트. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c6c4d91e-a095-4e82-bd0d-73dbea1e2906/2402.01999v1.pdf)

- 계산량은 $\(\mathcal{O}(\tau)\)$ 에 선형 비례하므로, 긴 horizon에서는 지연·전력 측면에서 부담이 커진다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c6c4d91e-a095-4e82-bd0d-73dbea1e2906/2402.01999v1.pdf)
- 대신, autoregressive 구조 덕분에 노이즈가 크거나 장기 예측에서 더 좋은 정확도를 보이는 경향이 있다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c6c4d91e-a095-4e82-bd0d-73dbea1e2906/2402.01999v1.pdf)

#### (2) Seq2Seq‑HDC (Sequence‑to‑Sequence HDC)

- 시점 $\(t\)$ 에서 한번에 $\(\tau\)$ 개 미래를 예측하는 **one‑shot** 방식. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c6c4d91e-a095-4e82-bd0d-73dbea1e2906/2402.01999v1.pdf)
- 절차:  
  1. $\(X_{t-T+1:t}\)$ 인코딩: $\(h = H(X_{t-T+1:t})\)$ .  
  2. 선형 회귀: $\(\tilde{X}_{t+1:t+\tau} = h W_r + b_r\)$ .  
  3. 실제 $\(X_{t+1:t+\tau}\)$ 이용해 시퀀스 단위 손실 $\(L(X_{t+1:t+\tau}, \tilde{X}_{t+1:t+\tau})\)$ 로 온라인 업데이트. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c6c4d91e-a095-4e82-bd0d-73dbea1e2906/2402.01999v1.pdf)

- $\(\tau\)$ 에 대해 추론 계산량이 거의 일정(행렬 연산 한 번)이라 장기 예측에서 **지연·전력 효율**이 매우 좋다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c6c4d91e-a095-4e82-bd0d-73dbea1e2906/2402.01999v1.pdf)

***

### 2.4 성능 향상 결과

#### (1) 단기 예측 (Short‑term TSF, $\(\tau \in \{3,6,12\}\)$ )

- 8개 실세계 데이터셋(ETTh1/2, ETTm1/2, WTH, ECL, Exchange, ILI)에서 RSE·CORR 비교 수행. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c6c4d91e-a095-4e82-bd0d-73dbea1e2906/2402.01999v1.pdf)
- AR‑HDC는 24개의 단기 테스트 케이스 전부에서 SOTA 온라인/트랜스포머/선형/GBRT 등보다 더 낮은 RSE 또는 더 높은 CORR을 달성한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c6c4d91e-a095-4e82-bd0d-73dbea1e2906/2402.01999v1.pdf)
- Seq2Seq‑HDC는 24개 중 10개 케이스에서 SOTA보다 우수하며, 2개 케이스에서 최고 성능을 기록한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c6c4d91e-a095-4e82-bd0d-73dbea1e2906/2402.01999v1.pdf)

예: ILI 데이터(질병 발생)에서 $\(\tau=3,6,12\)$ 모두 AR‑HDC는 FSNet, OnlineTCN, Informer 등을 상회하는 RSE/CORR을 보인다. [frontiersin](https://www.frontiersin.org/article/10.3389/fpubh.2020.554542/full)

#### (2) 장기 예측 (Long‑term TSF, $\(\tau\)$ 최대 384)

- AR‑HDC는 18개 장기 테스트 케이스 중 16개에서 baseline보다 더 나은 RSE 또는 CORR을 보여 특히 큰 horizon에서 강점을 보인다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c6c4d91e-a095-4e82-bd0d-73dbea1e2906/2402.01999v1.pdf)
- Seq2Seq‑HDC는 4/18에서 best RSE, 10/18에서 second best RSE로, 장기 예측에서도 경쟁력 있다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c6c4d91e-a095-4e82-bd0d-73dbea1e2906/2402.01999v1.pdf)
- ETTh1, ETTm1, ILI, WTH 등에서 $\(\tau=384\)$ 에서도 AR‑HDC가 안정된 성능을 유지하는 것이 인상적이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c6c4d91e-a095-4e82-bd0d-73dbea1e2906/2402.01999v1.pdf)

#### (3) 수렴 및 개념 드리프트 적응

- 누적 평균 RSE를 살펴보면, Exchange/ETTm1/ETTm2에서 초반 20% 구간의 개념 드리프트 이후 TSF‑HD 모델은 FSNet, ER, DER++, OnlineTCN보다 빠르게 수렴하고 더 낮은 최종 RSE를 유지한다. [arxiv](https://arxiv.org/abs/2202.11672)
- Synthetic Abrupt(S‑A) 데이터에서 여러 AR 프로세스 간 태스크 시프트를 반복할 때, TSF‑HD는 FSNet, ER, DER++보다 예측 변동성과 오차가 작아 “빠른 적응 + 잊지 않음” 특성이 관찰된다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c6c4d91e-a095-4e82-bd0d-73dbea1e2906/2402.01999v1.pdf)

#### (4) 엣지 디바이스 상 지연·전력

- Raspberry Pi 4 및 Jetson Nano에서 **짧은 horizon**( $\(\tau=3\)$ ) 기준:  
  - AR‑HDC가 가장 낮은 추론 지연과 전력, 그다음이 Seq2Seq‑HDC. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c6c4d91e-a095-4e82-bd0d-73dbea1e2906/2402.01999v1.pdf)
- **긴 horizon**( $\(\tau=96\)$ 이상):  
  - AR‑HDC는 루프 구조 때문에 지연이 가장 커지며,  
  - Seq2Seq‑HDC는 한 번의 행렬 연산으로 전체 시퀀스를 예측해, FSNet, OnlineTCN 등보다 낮은 지연 및 전력 소비를 보인다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c6c4d91e-a095-4e82-bd0d-73dbea1e2906/2402.01999v1.pdf)

#### (5) 차원 \(D\)의 영향

- Seq2Seq‑HDC에서 하이퍼벡터 차원 \(D\)를 500→10k로 늘리면 ETTh2, ETTm2에서 RSE가 각각 85%, 79% 감소해 “고차원일수록 좋다”는 HD 컴퓨팅 직관을 뒷받침한다. [arxiv](https://arxiv.org/html/2402.01999v1)
- AR‑HDC는 데이터 효율성이 높아 \(D\) 증가에 따른 이득이 상대적으로 작다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c6c4d91e-a095-4e82-bd0d-73dbea1e2906/2402.01999v1.pdf)

#### (6) 한계

- Exchange 데이터셋에서는 naive/선형(NLinear) 모델이 일부 케이스에서 더 좋게 나와, TSF‑HD가 모든 문제에서 지배적인 것은 아니다. [arxiv](https://arxiv.org/html/2402.01999v1)
- 매우 긴 horizon에서, 특히 Seq2Seq‑HDC의 경우 거리 보존성이 약해지고 인코더 행 간 직교성이 다소 감소해 성능 저하와 연결되는 것으로 보인다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c6c4d91e-a095-4e82-bd0d-73dbea1e2906/2402.01999v1.pdf)
- 원본 스케일 데이터에 대한 완전 온라인 학습 설정 때문에, 대규모 스케일 변화가 심한 데이터에서는 튜닝 없이는 초기 학습이 불안정할 수 있다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c6c4d91e-a095-4e82-bd0d-73dbea1e2906/2402.01999v1.pdf)

***

## 3. 일반화 성능 관점에서의 분석

### 3.1 논문 내 일반화 메커니즘

TSF‑HD가 일반화(특히 온라인·비정상 환경에서의 generalization)를 개선하는 요소는 다음과 같다. [arxiv](https://arxiv.org/html/2402.01999v1)

1. **고차원 선형화에 기반한 표현**  
   - 비선형 시계열을 고차원으로 사상하여, 거리 보존과 직교성을 유지하는 선형 구조를 학습함으로써, 입력 분포가 이동해도 “기하학적 구조”가 크게 변하지 않는 표현을 사용한다. [arxiv](https://arxiv.org/html/2402.01999v1)
   - 이는 새로운 태스크(분포)에서도 같은 인코더/회귀기 구조를 재사용해 **전이 가능**한 representation을 제공한다.  

2. **인코더–회귀기 공학습(co‑training)**  
   - 기존 RegHD류는 고정 인코더+학습 가능한 회귀기 구조라 태스크 시프트 적응력이 약했는데, TSF‑HD는 $\(W_e, W_r\)$ 를 함께 온라인 업데이트한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c6c4d91e-a095-4e82-bd0d-73dbea1e2906/2402.01999v1.pdf)
   - 실험적으로 거리 보존성과 행간 직교성이 유지되는 범위 내에서 $\(W_e\)$ 를 업데이트함으로써, representation을 서서히 이동시키면서도 기존 태스크에 대한 선형성 구조를 유지한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c6c4d91e-a095-4e82-bd0d-73dbea1e2906/2402.01999v1.pdf)

3. **task‑free online learning**  
   - FSNet은 태스크 경계를 가정하고 fast/slow 컴포넌트를 구분하지만, TSF‑HD는 태스크 경계 없이 입력 스트림만 보고 학습한다. [arxiv](https://arxiv.org/abs/2202.11672)
   - Synthetic Abrupt 실험에서 보듯이, 명시적 경계 정보 없이도 새로운 AR 과정에 빠르게 적응하고, 재등장하는 과거 AR 태스크에도 다시 잘 동작한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c6c4d91e-a095-4e82-bd0d-73dbea1e2906/2402.01999v1.pdf)

4. **Huber loss + L2 정규화**  
   - 스케일이 큰 노이즈/이상값에 대한 로버스트한 손실을 사용해, 극단적인 샘플에 과적합되는 것을 방지한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c6c4d91e-a095-4e82-bd0d-73dbea1e2906/2402.01999v1.pdf)

이 모든 요소가 합쳐져, 개념 드리프트 환경에서의 **온라인 generalization**이 개선된다.  

### 3.2 일반화 성능 향상 가능성 (향후 관점)

논문 자체는 explicit generalization bound를 제공하지 않지만, 구조상 다음과 같은 확장 가능성이 있다. [arxiv](https://arxiv.org/html/2402.01999v1)

- **멀티스케일/다중 헤드 인코더**: 현재는 단일 선형+ReLU 인코더지만, 다양한 temporal scale(예: 계절·추세)을 위한 다중 $\(W_e^{(k)}\)$ 를 둔 후, HD 공간 상에서 결합하면 서로 다른 스케일간 패턴을 분리해 generalization을 높일 수 있다. [arxiv](https://arxiv.org/pdf/2307.01616.pdf)
- **모델‑기반 정규화와 representation replay**: HD representation에 대해 experience replay 또는 prototype 기반 정규화를 추가하면, 특정 드리프트에서의 과도한 적응으로 이전 태스크를 잊는 것을 줄이고, continual learning 이론에서 말하는 stability–plasticity trade‑off를 더 잘 제어할 수 있다. [arxiv](https://arxiv.org/abs/2202.11672)
- **멀티모달/텍스트 조건 시계열**: 최근 고차원 TSF에서 텍스트 등 외부 설명 변수의 활용이 generalization에 효과적이라는 결과가 있어, TSF‑HD의 HD representation에 추가 모달리티를 joint‑embedding하면 도메인 전반의 generalization을 높일 여지가 있다. [arxiv](https://arxiv.org/pdf/2501.07048.pdf)

***

## 4. 2020년 이후 관련 최신 연구와의 비교

아래 표는 2020년 이후 주요 온라인/장기 TSF 연구와 TSF‑HD를 비교한 것이다. [arxiv](https://arxiv.org/pdf/2304.08424.pdf)

### 최근 시계열 예측 연구 대비 개요

| 연구(연도) | 핵심 아이디어 | 온라인성 | 모델 복잡도 | 엣지 적합성 | 일반화 논점 |
|-----------|---------------|-----------|-------------|------------|-------------|
| FSNet (Pham+ 2022) [arxiv](https://arxiv.org/abs/2202.11672) | CLS 이론 기반 Fast/Slow 네트워크, adapter + associative memory | 온라인, 태스크 기반 | 깊은 네트워크, 메모리 구조 | 낮음 (비교적 무겁고 학습 복잡) | abrupt/recurrent 패턴 적응 강조, 이론적 bound는 없음 |
| OnlineTCN (Woo+ 2022) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c6c4d91e-a095-4e82-bd0d-73dbea1e2906/2402.01999v1.pdf) | 10‑layer TCN을 온라인 학습 | 온라인 | 중간 (심층 convolution) | 중간 | 계절성·추세 분리, 드리프트 적응은 제한적 |
| NLinear (Zeng+ 2023) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c6c4d91e-a095-4e82-bd0d-73dbea1e2906/2402.01999v1.pdf) | DLinear류 간단 선형 모델로 장기 TSF | 오프라인 | 매우 낮음 | 높음 | 특정 장기 TSF에서 Transformer보다 우수, 그러나 비선형 패턴 제한 |
| TiDE, N‑BEATS 등 MLP/Deep TSF (2020–2024) [mdpi](https://www.mdpi.com/2071-1050/16/18/8227) | 깊은 MLP/Residual/Encoder‑Decoder 구조로 장기 TSF | 주로 오프라인 | 높음 | 낮음~중간 | 데이터 풍부한 정적 분포에서 강력한 generalization |
| 시계열 Foundation 모델 (2024) [arxiv](https://arxiv.org/pdf/2310.10688.pdf) | 거대 decoder‑only Transformer, 대규모 사전학습 후 zero‑shot | 사전학습+zero‑shot | 매우 높음 | 거의 서버 전용 | 크로스‑도메인 generalization, 그러나 엣지/온라인은 아님 |
| TSF‑HD (본 논문, 2024) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c6c4d91e-a095-4e82-bd0d-73dbea1e2906/2402.01999v1.pdf) | HD 인코더 + 선형 회귀기, 온라인 공학습, AR/Seq2Seq 두 모드 | 완전 온라인, task‑free | 낮음 (선형 연산 위주) | 매우 높음 (Raspberry Pi/Jetson 실증) | 드리프트/태스크 시프트 환경에서 높은 generalization 성능을 실증적으로 보임 |

TSF‑HD는 FSNet/OnlineTCN처럼 “온라인 적응”에 초점을 두면서도, 모델 복잡도를 선형 수준으로 낮춰 엣지 디바이스에서 실제로 동작할 수 있게 만든 점이 차별적이다. 또한 NLinear 계열의 “단순 선형 모델도 강하다”는 관찰을 이어 받아, 이를 고차원 선형 회귀로 확장해 보다 복잡한 비선형 패턴도 포착하려고 한다는 점에서 TiDE 등과 다른 방향의 설계이다. [arxiv](https://arxiv.org/pdf/2304.08424.pdf)

***

## 5. 향후 연구에의 영향과 연구 시 고려할 점

### 5.1 향후 연구에의 영향

TSF‑HD는 몇 가지 방향에서 후속 연구의 기반을 제공한다. [arxiv](https://arxiv.org/html/2402.01999v1)

1. **엣지‑우선 온라인 TSF 패러다임**  
   - “고정된 대형 딥 모델 + 서버 인퍼런스” 대신, **경량 선형 HD 모델 + 로컬 온라인 업데이트**라는 설계가 실제 Raspberry Pi, Jetson에서 검증되었다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c6c4d91e-a095-4e82-bd0d-73dbea1e2906/2402.01999v1.pdf)
   - IoT, 스마트그리드, 엣지 센서 네트워크 등에서 “단말 측 학습형 TSF”를 설계할 때 유력한 출발점이 될 수 있다. [arxiv](https://arxiv.org/abs/2402.01999)

2. **HD 컴퓨팅의 TSF/연속 제어로의 확장**  
   - 기존 HD 컴퓨팅은 분류/회귀/FPGA 가속 등에 주로 사용되었는데, 이 논문은 고난이도 연속 TSF 문제에 성공적으로 적용한 사례다. [arxiv](https://arxiv.org/html/2402.01999v1)
   - 이후에는 spatio‑temporal, graph‑based TSF, reinforcement learning에서 HD representation을 사용하는 연구를 촉발할 가능성이 있다. [arxiv](http://arxiv.org/pdf/2212.02567.pdf)

3. **Continual & Online Learning 커뮤니티와의 연결**  
   - “task‑free online HD 회귀”라는 포맷은 기존 CLS/experience replay/regularization 기반 continual learning 방법과 자연스럽게 결합·비교될 수 있다. [arxiv](https://arxiv.org/abs/2202.11672)
   - 이 방향으로의 후속 연구는 TSF‑HD에 EWC, GEM, replay buffer 등을 결합하거나, 반대로 HD representation을 다른 continual learner에 집어넣는 식이 될 수 있다. [arxiv](https://arxiv.org/pdf/2004.10240.pdf)

### 5.2 앞으로 연구 시 고려할 점 (연구자 관점 제언)

1. **데이터 전처리·스케일링 전략**  
   - 논문은 완전 온라인 환경을 가정해 정규화 없이 원 데이터를 사용하지만, 실제 시스템에서는 online standardization, adaptive scaling 등을 넣어 초기 학습 안정성을 높여야 할 수 있다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c6c4d91e-a095-4e82-bd0d-73dbea1e2906/2402.01999v1.pdf)

2. **하이퍼디멘션 설계와 효율성 균형**  
   - 차원 \(D\) 증가가 성능을 크게 올리지만, 메모리·연산량도 함께 증가한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c6c4d91e-a095-4e82-bd0d-73dbea1e2906/2402.01999v1.pdf)
   - 특정 엣지 플랫폼(예: MCU급 vs GPU 엣지)에서 허용 가능한 $\(D, T, \tau\)$ 조합을 체계적으로 탐색하는 연구가 필요하다.  

3. **복합 구조와의 하이브리드화**  
   - HD 인코더를 **상단 선형 모듈**이 아니라, “딥 피처 위의 마지막 선형 레이어”로 결합해 FSNet, N‑BEATS, TiDE 등의 representation을 HD 공간에서 재투영하는 하이브리드 모델도 유망하다. [mdpi](https://www.mdpi.com/2071-1050/16/18/8227)

4. **이론적 일반화/수렴 분석**  
   - 현재는 경험적 결과 중심이므로,  
     - drift‑aware regret bound,  
     - HD 공간에서의 margin 기반 generalization bound,  
     - task recurrence 시 representation stability 분석  
     등이 후속 작업으로 요구된다. [arxiv](https://arxiv.org/pdf/2004.10240.pdf)

5. **도메인별 특화 실증**  
   - 논문에서는 에너지, 날씨, 환율, 질병 등 범용 데이터셋을 사용했으나,  
     - 의료(병상 수, 감염률),  
     - 교통(카‑헤일링, 교통량),  
     - 산업 설비 예지보전  
     같은 high‑stakes 도메인에서 HD 기반 온라인 TSF가 어느 정도 설명력·공정성을 제공하는지 검증하는 것이 중요하다. [tandfonline](https://www.tandfonline.com/doi/full/10.1080/19427867.2024.2313832)

***

정리하면, 이 논문은 “고차원 선형 모델 + 온라인 공학습”이라는 심플한 구조로, 개념 드리프트가 존재하는 환경에서 SOTA 온라인 딥 TSF 모델을 능가하는 성능과 엣지 배포 가능성을 동시에 보여주며, 앞으로의 연구에서 **고차원 표현, 온라인 학습, 엣지 컴퓨팅**을 연결하는 중요한 레퍼런스로 활용될 수 있다. [arxiv](https://arxiv.org/abs/2202.11672)

<span style="display:none">[^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28]</span>

<div align="center">⁂</div>

[^1_1]: 2402.01999v1.pdf

[^1_2]: https://arxiv.org/html/2402.01999v1

[^1_3]: https://arxiv.org/abs/2202.11672

[^1_4]: https://arxiv.org/pdf/2004.10240.pdf

[^1_5]: https://arxiv.org/pdf/2304.08424.pdf

[^1_6]: https://www.frontiersin.org/article/10.3389/fpubh.2020.554542/full

[^1_7]: https://arxiv.org/pdf/2307.01616.pdf

[^1_8]: https://arxiv.org/pdf/2501.07048.pdf

[^1_9]: https://github.com/ddz16/TSFpaper

[^1_10]: https://www.mdpi.com/2071-1050/16/18/8227

[^1_11]: https://arxiv.org/pdf/2310.10688.pdf

[^1_12]: https://arxiv.org/abs/2402.01999

[^1_13]: http://arxiv.org/pdf/2212.02567.pdf

[^1_14]: https://www.tandfonline.com/doi/full/10.1080/19427867.2024.2313832

[^1_15]: https://ieeexplore.ieee.org/document/10676967/

[^1_16]: https://ieeexplore.ieee.org/document/10957222/

[^1_17]: https://www.bio-conferences.org/10.1051/bioconf/20249700113

[^1_18]: https://ieeexplore.ieee.org/document/10834940/

[^1_19]: https://ejournal.raharja.ac.id/index.php/ccit/article/view/3443

[^1_20]: http://www.ajas.uoanbar.edu.iq/Article_Details.php?ID=770

[^1_21]: http://arxiv.org/pdf/2401.09261.pdf

[^1_22]: http://arxiv.org/abs/1905.03806

[^1_23]: https://www.signalpop.com/2023/11/04/understanding-fsnets-for-fast-and-slow-online-time-series-forecasting/

[^1_24]: https://huggingface.co/blog/autoformer

[^1_25]: https://openreview.net/forum?id=ZmkrJy3GlU

[^1_26]: https://iclr.cc/virtual/2023/poster/11198

[^1_27]: https://dl.acm.org/doi/abs/10.1609/aaai.v39i15.33717

[^1_28]: https://www.themoonlight.io/ja/review/a-novel-hyperdimensional-computing-framework-for-online-time-series-forecasting-on-the-edge
