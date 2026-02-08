# FSNet : Learning Fast and Slow for Online Time Series Forecasting

## 1. 핵심 주장과 주요 기여 요약

- 이 논문은 “온라인 시계열 예측을 **과업-경계가 없는 온라인 continual learning 문제**로 재정의하고, 비정상 환경에서 새로운 패턴과 재등장하는 패턴을 동시에 빠르게 처리해야 한다”고 주장합니다. [arxiv](https://arxiv.org/abs/2202.11672)
- 이를 위해 TCN(Temporal Convolutional Network) 위에 **층별(layer-wise) 어댑터(빠른 학습)**와 **연상 메모리(느린 학습)**를 결합한 FSNet(Fast and Slow learning Network)을 제안하여, online 학습 시 빠른 적응과 장기 기억을 동시에 달성합니다. [openreview](https://openreview.net/pdf?id=q-PbpHD3EOk)
- 여러 실제(ETTh2, ETTm1, ECL, Traffic, Weather) 및 합성(S‑Abrupt, S‑Gradual) 데이터셋에서 Online TCN, ER/DER++/MIR/TFCL, Informer보다 **누적 MSE/MAE 및 수렴 속도**에서 일관되게 우수한 성능을 보입니다. [arxiv](https://arxiv.org/pdf/2202.11672.pdf)

***

## 2. 문제, 방법(수식), 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

1) 기본 시계열 예측 설정  
- 길이 $\(T\)$ , 차원 $\(n\)$ 의 시계열  

$$X = (x_1, \dots, x_T) \in \mathbb{R}^{T \times n}$$  
  
  에 대해, 길이 $\(e\)$ 의 look-back 윈도우  

$$X_{i,e} = (x_{i-e+1}, \dots, x_i)$$  
  
  를 보고, 길이 $\(H\)$ 의 미래  

$$f_\omega(X_{i,e}) = (x_{i+1}, \dots, x_{i+H})$$  
  
  를 예측하는 문제를 다룹니다. [arxiv](https://arxiv.org/pdf/2202.11672.pdf)

- 온라인 설정에서는 각 시점 $\(t\)$ 마다 $\(x_t\)$ 에 기반한 윈도우를 입력받아 예측을 수행한 뒤, 실제 $\(y_t\)$ 를 보고 그 한 샘플만으로 파라미터를 업데이트하며, 전체 기간의 누적 손실을 줄이는 것이 목표입니다. [arxiv](https://arxiv.org/abs/2202.11672)

2) 온라인 딥 포캐스팅의 난점  
- batch 학습에서의 mini‑batch·multi‑epoch 이점이 사라져 **수렴이 느리고** [arxiv](https://arxiv.org/pdf/2202.11672.pdf)
- 분포 이동(concept drift) 발생 시 새로운 개념을 학습하려면 많은 샘플이 필요하며,  
- 과거 패턴이 재등장할 때 catastrophic forgetting 때문에 **재학습 비용이 커지는 것**이 핵심 문제입니다. [alphaxiv](https://www.alphaxiv.org/overview/2202.11672v2)

3) 이 논문의 재정의: task‑free online continual learning  
- 시간축을 **locally stationary segment**의 시퀀스로 보고, 각 segment를 하나의 “task”로 간주합니다. [openreview](https://openreview.net/pdf?id=q-PbpHD3EOk)
- task 전환 시점은 주어지지 않고, 모델은 오직 데이터 스트림과 손실/gradient 신호만으로 **stability–plasticity** 균형을 맞춰야 합니다. [arxiv](https://arxiv.org/pdf/2202.11672.pdf)

***

### 2.2 FSNet의 제안 방법 (수식 포함)

FSNet은 TCN 백본 위에 **(i) per-layer adapter, (ii) per-layer associative memory**를 더한 구조입니다. [arxiv](https://arxiv.org/abs/2202.11672)

#### 2.2.1 온라인 손실 정의

각 시점 $\(t\)$ 에서 모델 예측을 $\(\hat{y}\_t = f_\omega(x_t)\)$ , 정답을 $\(y_t\)$ 라 할 때, 다단계 MSE 손실은  

$$
\ell(\hat{y}_t, y_t) 
= \frac{1}{H} \sum_{j=1}^H \lVert \hat{y}_{t,j} - y_{t,j} \rVert^2
$$  

로 정의되며, 온라인 학습의 목표는  

$$
\sum_{t=1}^T \ell(\hat{y}_t, y_t)
$$  

를 최소화하는 것입니다. [arxiv](https://arxiv.org/pdf/2202.11672.pdf)

#### 2.2.2 층별 gradient EMA 기반 fast adapter

각 TCN 층 $\(l\)$ 의 파라미터를 $\(\theta_l\)$ , 그 층의 손실 gradient를 $\(g_l^t = \nabla_{\theta_l}\ell_t\)$ 라 하면, FSNet은 **gradient의 지수이동평균(EMA)**를  

$$
\hat{g}_l \leftarrow \gamma \hat{g}_l + (1 - \gamma) g_l^t
$$  

형태로 유지합니다 $(\(0 < \gamma < 1\))$ . [arxiv](https://arxiv.org/pdf/2202.11672.pdf)

이 $\(\hat{g}_l\)$ 을 adapter의 입력으로 사용하여, 낮은 차원의 적응 계수 $\(u_l\)$ 로 매핑합니다.  
- adapter 파라미터를 $\(\phi_l\)$ 라 할 때,  

$$
  u_l = \Omega(\hat{g}_l; \phi_l)
  $$

- $\(u_l\)$ 는 weight 계수 $\(\alpha_l\)$ 와 feature 계수 $\(\beta_l\)$ 를 포함합니다.  

$$
  u_l = [\alpha_l; \beta_l]
  $$  

적응 과정은  
- weight 변환:  

$$
  \tilde{\theta}_l = \mathrm{tile}(\alpha_l) \odot \theta_l
  $$  

- feature 변환 및 convolution:  

$$
  \tilde{h}_l = \mathrm{tile}(\beta_l) \odot h_l,\quad
  h_l = \tilde{\theta}_l * \tilde{h}_{l-1}
  $$  

입니다. 여기서 $\(*\)$ 는 dilated conv, $\(\odot\)$ 는 element-wise 곱입니다. [arxiv](https://arxiv.org/pdf/2202.11672.pdf)

$\(\Omega\)$ 는 gradient 벡터를 $\(d\)$ 개의 chunk로 나누어 두 층 MLP로 사상하는 **chunking adapter**로 구현됩니다. [arxiv](https://arxiv.org/pdf/2202.11672.pdf)

#### 2.2.3 gradient interference 기반 sparse memory trigger

FSNet은 재등장 패턴을 위해 각 층에 연상 메모리 $\(M_l \in \mathbb{R}^{N \times d}\)$ 를 둡니다. [arxiv](https://arxiv.org/pdf/2202.11672.pdf)

또 하나의 단기 EMA $\(\hat{g}'_l\)$ 를  

$$
\hat{g}'_l \leftarrow \gamma' \hat{g}'_l + (1-\gamma') g_l^t,\quad \gamma' < \gamma
$$  

로 갱신하면서, representation interference를 cosine similarity로 측정합니다. 메모리 인터랙션은 다음 조건에서만 트리거됩니다.  

$$
\cos(\hat{g}_l, \hat{g}'_l) 
= \frac{\hat{g}_l \cdot \hat{g}'_l}{\lVert \hat{g}_l \rVert \lVert \hat{g}'_l \rVert}
< -\tau
$$  

여기서 $\(\tau > 0\)$ 는 threshold로, 보통 0.7 근처의 큰 값을 씁니다. [arxiv](https://arxiv.org/pdf/2202.11672.pdf)

이 조건은 “현재 gradient 방향이 과거와 강하게 반대”일 때 (큰 concept drift 또는 강한 간섭)만 메모리를 사용하도록 하는 sparse gating입니다. [arxiv](https://arxiv.org/pdf/2202.11672.pdf)

#### 2.2.4 연상 메모리 read/write

메모리 read/write에서는 적응 계수의 EMA $\(\hat{u}_l\)$ 를 쿼리로 사용합니다. [arxiv](https://arxiv.org/pdf/2202.11672.pdf)

1) Read (top‑k attention):  
- attention score  

$$
  r_l = \mathrm{softmax}(M_l \hat{u}_l)
  $$  

- 상위 $\(k\)$ 개 선택:  

$$
  r_l^{(k)} = \mathrm{TopK}(r_l)
  $$  

- 과거 적응 계수의 가중합:  

$$
  \tilde{u}_l = \sum_{i=1}^K r_l^{(k)}[i]\, M_l[i]
  $$  

2) 현재–과거 적응 계수 결합:  

$$
u_l \leftarrow \tau u_l + (1-\tau)\tilde{u}_l
$$  

3) Write (outer product 기반 갱신):  

$$
M_l \leftarrow \tau M_l + (1-\tau)\, \hat{u}_l \otimes r_l^{(k)}
$$  

$$
M_l \leftarrow \frac{M_l}{\max(1,\lVert M_l\rVert_2)}
$$  

이를 통해 FSNet은  
- 현재 패턴과 유사한 과거 적응 벡터를 검색하여 **few-shot 재적응**,  
- 동시에 해당 메모리 슬롯에 최신 정보를 누적해 **재등장 패턴을 강화**합니다. [arxiv](https://arxiv.org/pdf/2202.11672.pdf)

***

### 2.3 모델 구조

1) Backbone: TCN  
- $\(L\)$ 개의 dilated causal conv 블록, 각 블록은 residual 구조의 여러 conv 필터를 포함합니다. [arxiv](https://arxiv.org/pdf/2202.11672.pdf)
- 마지막에 linear regressor가 time‑channel representation 전체를 받아 $\(H\)$ -step 예측을 동시에 출력합니다. [arxiv](https://arxiv.org/pdf/2202.11672.pdf)

2) FSNet augmentation  
- 각 conv 필터에 adapter+memory 모듈을 부착해 **필터 단위의 fast–slow 학습**을 구현합니다. [arxiv](https://arxiv.org/pdf/2202.11672.pdf)
- Forward:  
  - 입력 → 각 층에서 gradient EMA 기반 adapter 적용, (trigger 시) memory read → 최종 representation → regressor → 예측  
- Backward:  
  - 각 층 gradient로 $\(\hat{g}_l, \hat{g}'_l, \hat{u}_l\)$ 갱신 → trigger 조건 검사 후 memory read/write → $\(\theta_l,\phi_l\)$ 를 AdamW로 업데이트. [arxiv](https://arxiv.org/pdf/2202.11672.pdf)

3) 복잡도  
- adapter + memory의 파라미터 수는 conv layer 크기에 비례해 $\(\mathcal{O}(N)\)$ 수준이며, 경험 재생을 위한 대형 buffer 없이도 동작합니다. [arxiv](https://arxiv.org/pdf/2202.11672.pdf)

***

### 2.4 실험: 성능 향상

1) 벤치마크  
- Real: ETTh2/ETTm1 (전력 트랜스포머), ECL, Traffic, Weather 등. [arxiv](https://arxiv.org/pdf/2202.11672.pdf)
- Synthetic:  
  - S‑Abrupt:  

$$\mathrm{AR}\_{0.1}(1) \rightarrow \mathrm{AR}_{0.4}(1) \rightarrow \mathrm{AR}_{0.6}(1) \rightarrow \mathrm{AR}_{0.1}(1) \rightarrow \dots$$  
    
형태의 abrupt + recurrent drift. [arxiv](https://arxiv.org/pdf/2202.11672.pdf)
  - S‑Gradual: 세그먼트 마지막 20%에서 두 AR 프로세스를 평균한 형태로 서서히 drift. [arxiv](https://arxiv.org/pdf/2202.11672.pdf)

2) Baselines  
- OnlineTCN, ER, MIR, DER++, TFCL, Informer 등. [arxiv](https://arxiv.org/pdf/2202.11672.pdf)

3) 주요 수치 결과 (예시)  
- ETTh2, \(H=24\):  
  - FSNet: MSE 0.687, MAE 0.467  
  - DER++: MSE 0.828, ER: 0.808, OnlineTCN: 0.830, Informer: 4.629. [arxiv](https://arxiv.org/pdf/2202.11672.pdf)
- S‑Abrupt, \(H=24\):  
  - FSNet: MSE 1.299  
  - DER++: 3.598, ER: 3.375, TFCL: 3.415. [arxiv](https://arxiv.org/pdf/2202.11672.pdf)
- S‑Gradual에서도 FSNet이 drift 직후 큰 spike 이후, baseline보다 빠르게 낮은 오차 수준으로 회복합니다. [alphaxiv](https://www.alphaxiv.org/overview/2202.11672v2)

4) Ablation 결과  
- No‑Memory (adapter만) vs FSNet:  
  - Traffic, S‑Gradual 같이 완만한 drift에서는 성능이 비슷하지만,  
  - ETTh2, S‑Abrupt처럼 abrupt + recurrent drift에서는 메모리가 분명한 이득을 줌. [arxiv](https://arxiv.org/pdf/2202.11672.pdf)
- Naive (gradient EMA 없이 u를 직접 학습) → 성능 크게 저하 → **gradient EMA를 이용한 fast adaptation이 핵심**임을 확인. [arxiv](https://arxiv.org/pdf/2202.11672.pdf)
- 메모리 크기 32→128: 대부분 설정에서 성능 향상 → 예측 일반화와 메모리 용량 간 trade-off를 보여줌. [arxiv](https://arxiv.org/pdf/2202.11672.pdf)

***

### 2.5 한계

- irregular sampling 및 부분 차원 drift: 일부 feature만 드리프트하는 경우 전체 층의 gradient가 trigger를 유도해, 다른 차원의 학습을 방해할 수 있음. [arxiv](https://arxiv.org/pdf/2202.11672.pdf)
- 복잡한 반복 패턴(예: 금융 시계열)에서 메모리 용량보다 많은 distinct pattern이 존재하면, 메모리 내부에서도 catastrophic forgetting이 발생할 수 있음. [arxiv](https://arxiv.org/pdf/2202.11672.pdf)
- z-normalization 고정 통계: drift 이후에도 과거 통계를 쓰는 것은 스케일 mismatch를 일으켜, online/슬라이딩 윈도우 기반 정규화가 필요합니다. [arxiv](https://arxiv.org/pdf/2202.11672.pdf)
- adapter+memory 연산으로 인해 ER/DER++ 대비 throughput(초당 샘플 처리량)은 약간 낮습니다. [arxiv](https://arxiv.org/pdf/2202.11672.pdf)

***

## 3. 모델의 일반화 성능 향상 가능성

이 논문에서 “일반화”는 IID test-set generalization 보다는, **시간축에서의 regime 변화에 대한 적응 및 과거 패턴 재사용 능력**으로 이해할 수 있습니다. [alphaxiv](https://www.alphaxiv.org/overview/2202.11672v2)

### 3.1 Continual 관점의 일반화

- locally stationary segment를 task로 보면서, FSNet이  
  1) 현재 segment의 손실을 빠르게 감소시키고,  
  2) 과거 segment에서 축적한 적응 벡터를 재사용해, 새로운 segment에 **효율적으로 transfer**한다는 점을 보여줍니다. [arxiv](https://arxiv.org/pdf/2202.11672.pdf)

합성 AR 데이터에서, 동일한 $\(\phi\)$ 값을 갖는 AR 프로세스가 재등장할 때 FSNet은  
- 과거에 저장한 $\(u_l\)$ 를 다시 읽어와 적은 샘플로도 빠르게 오차를 줄이는데, 이는 **시간 축 generalization 강화**로 볼 수 있습니다. [arxiv](https://arxiv.org/pdf/2202.11672.pdf)

### 3.2 일반화 향상을 위한 설계상의 장점

1) backbone vs adapter+memory의 역할 분리  
- backbone(TCN)은 비교적 느리게 업데이트되며 일반적인 representation을 형성하고,  
- adapter/메모리는 현재 환경에 특화된 “단기 변형”을 담당하므로, backbone이 과도하게 특정 regime에 overfit되지 않게 합니다. [openreview](https://openreview.net/pdf?id=q-PbpHD3EOk)

2) replay-free 요약 메모리  
- FSNet은 원본 데이터를 저장하지 않고, **gradient 기반 적응 벡터 $\(u_l\)$ **만 저장하므로 privacy/저장 측면에서 유리하며, 다른 도메인/다운스트림 작업에서도 **패턴별 적응 정보를 재활용**할 여지가 있습니다. [alphaxiv](https://www.alphaxiv.org/overview/2202.11672v2)

3) hyper-parameter robustness  
- $\(\gamma, \gamma', \tau\)$ 등을 상당 범위에서 바꾸어도 성능이 크게 흔들리지 않음을 실험적으로 보고하며, 이는 온라인 배포 상황에서 generalization 안정성에 긍정적입니다. [arxiv](https://arxiv.org/pdf/2202.11672.pdf)

***

## 4. 2020년 이후 최신 연구 비교·분석

### 4.1 Online Continual Learning 기반 후속 연구

#### (1) Online Continual Learning for Time Series: a Natural Score‑driven Approach (NatSR, 2026) [arxiv](https://arxiv.org/html/2601.12931v1)

- FSNet을 대표 사례로 언급하면서, 온라인 시계열 예측을 **score-driven 모델과 자연 gradient** 관점에서 재해석합니다. [arxiv](https://arxiv.org/abs/2601.12931)
- 주요 기여:  
  - score-driven 모델(GAS)와 natural gradient descent 사이의 이론적 연결을 증명. [arxiv](https://arxiv.org/html/2601.12931v1)
  - Student’s t likelihood와 결합해 update bound를 제공하여 outlier에 대한 robust 학습을 달성. [openreview](https://openreview.net/forum?id=l4qsixUiir)
  - **Natural Score-driven Replay (NatSR)**: 자연 gradient + t‑likelihood + replay buffer + dynamic scale heuristic으로 drift 시 빠른 적응과 안정성을 동시에 달성. [arxiv](https://arxiv.org/abs/2601.12931)

- FSNet vs NatSR:  
  - FSNet: architectural 방식(adapter+memory), replay‑free, gradient EMA + cosine trigger 중심. [arxiv](https://arxiv.org/abs/2202.11672)
  - NatSR: optimizer/likelihood 수준에서의 접근 (natural gradient + Student’s t + replay), 이론적 optimality와 robust adaptation을 강조. [openreview](https://openreview.net/forum?id=l4qsixUiir)
  - 실험적으로 NatSR는 여러 데이터셋에서 FSNet 등 기존 방법보다 더 강한 forecasting 성능을 보이는 것으로 보고되며, FSNet의 구조적 inductive bias를 “학습 규칙” 측면에서 보완합니다. [arxiv](https://arxiv.org/abs/2601.12931)

#### (2) Buffer‑free ODE 기반 온라인 학습 [arxiv](https://arxiv.org/html/2411.07413v1)

- 일부 연구는 gradient dynamics를 ODE로 모델링해 replay 없이도 안정적인 online 적응을 달성하려 합니다. [arxiv](https://arxiv.org/html/2411.07413v1)
- FSNet과 달리, 구조보다는 **연속시간 동학과 안정성 이론**에 기반해 generalization을 확보하려는 방향입니다.  

### 4.2 Retrieval‑/Memory‑augmented forecasting 계열

#### (1) Retrieval‑Augmented Time Series Forecasting (RAFT, 2024–25) [arxiv](https://arxiv.org/html/2505.04163v1)

- RAFT/RAF는 시계열 입력과 유사한 과거 패턴을 검색해, **그 과거의 “미래 값”까지 함께 입력으로 사용하는 RAG 스타일** 예측 프레임워크를 제안합니다. [arxiv](https://arxiv.org/abs/2411.08249)
- 핵심:  
  - 과거 시계열 패치들을 검색하고, 그 패치들의 미래를 retrieval evidence로 사용해 backbone 예측을 보정. [arxiv](https://arxiv.org/html/2505.04163v1)
  - multi‑period retrieval로 다양한 scale의 패턴을 활용. [arxiv](https://arxiv.org/html/2505.04163v1)

- FSNet vs RAFT:  
  - FSNet: gradient 기반 internal memory로 “어떻게 업데이트했는지”를 저장.  
  - RAFT: raw 시계열 조각과 그 미래를 external memory에서 검색하여 입력 증강.  
  - 온라인 환경에서 RAFT는 과거 데이터를 대규모로 유지해야 하므로 저장/프라이버시 비용이 크고, FSNet은 요약된 적응 정보만 저장하는 장점이 있습니다. [arxiv](https://arxiv.org/html/2505.04163v1)

#### (2) TS‑RAG 및 memory‑augmented TS foundation models [arxiv](https://arxiv.org/html/2503.07649v3)

- TS‑RAG 등은 foundation TS 모델 위에 retrieval‑augmented generation 구조를 더해, zero/low‑shot generalization을 강화합니다. [arxiv](https://arxiv.org/html/2503.07649v3)
- memory‑augmented Transformer 리뷰 논문은 FSNet과 유사한 **base model + memory** 패턴을 일반 sequence modeling으로 확장합니다. [arxiv](https://arxiv.org/html/2508.10824v1)

### 4.3 기타 continual forecasting 연구

- “Continual Learning for Time Series Forecasting” 계열 정리 논문은, FSNet을 architecture‑based OCL forecaster의 대표 사례로 다루며, 이후의 NatSR, RAFT 등을 함께 비교합니다. [semanticscholar](https://www.semanticscholar.org/paper/4d755a5a66a8c46b722b44613788085191524e11)
- Time-FFM, TEMPO 등 시계열 foundation model 연구에서는 prompt/adapter를 활용한 domain adaptation을 다루며, FSNet 스타일의 fast–slow 구조와 자연스럽게 접점을 형성합니다. [arxiv](http://arxiv.org/pdf/2405.14252.pdf)

***

## 5. 앞으로의 연구 영향과 고려할 점

### 5.1 이 논문의 영향

1) 문제 정의 측면  
- 온라인 시계열 예측을 **online, task-free continual learning**으로 보는 관점을 정립하여, 이후 NatSR 등 많은 연구가 같은 framing을 채택합니다. [arxiv](https://arxiv.org/pdf/2601.12931.pdf)

2) 구조적 설계 패턴  
- backbone + fast adapter + memory라는 구분은, 이후 시계열·언어·멀티모달 foundation 모델에서 재사용 가능한 패턴으로 자리잡았습니다. [semanticscholar](https://www.semanticscholar.org/paper/4d755a5a66a8c46b722b44613788085191524e11)

3) 실용적 적용  
- replay-free이면서도 drift 대응력이 좋은 구조를 제시해, 데이터 저장이 어렵거나 프라이버시 제약이 있는 영역에서 **가벼운 online forecaster** 설계의 출발점이 되었습니다. [marktechpost](https://www.marktechpost.com/2022/11/02/salesforce-ai-developed-fsnet-fast-and-slow-learning-network-for-deep-time-series-forecasting-which-can-learn-deep-forecasting-models-on-the-fly-in-a-nonstationary-environment/)

### 5.2 향후 연구 시 고려할 점

1) FSNet × 고급 정규화·전처리  
- RevIN, online mean/variance 추정, regime-aware scaling을 FSNet과 결합하면, ECL·Traffic·Finance처럼 scale shift가 큰 데이터에서 일반화 성능이 크게 향상될 수 있습니다. [arxiv](https://arxiv.org/pdf/2312.17100.pdf)

2) FSNet × foundation/Transformer 기반 TS 모델  
- decoder‑only TS foundation model, TEMPO, Time‑FFM처럼 대규모 사전학습 모델 위에 FSNet식 adapter+memory를 얹는 방향은,  
  - zero/low‑shot forecasting,  
  - 도메인 간 transfer  
에서 강력한 일반화 성능을 기대할 수 있는 연구 방향입니다. [arxiv](http://arxiv.org/pdf/2310.04948.pdf)

3) FSNet × NatSR 스타일 optimizer  
- FSNet 구조를 유지하되, 최적화는 natural gradient + Student’s t loss + dynamic scale(=NatSR)을 사용하면,  
  - 구조적 fast–slow bias +  
  - 이론적으로 정당화된 robust update  
를 동시에 얻을 수 있으며, drift 환경 generalization을 한층 더 강화할 수 있습니다. [arxiv](https://arxiv.org/html/2601.12931v1)

4) Irregular / partial drift 시계열로의 확장  
- feature-wise drift detection, 차원별 trigger 임계, cross-dimensional attention을 도입해, 불규칙 샘플링 및 부분 차원 drift에 대한 FSNet의 한계를 완화할 수 있습니다. [pdfs.semanticscholar](https://pdfs.semanticscholar.org/0678/8ff99baf69e8c8e032bdf2285774299ef1f6.pdf)

5) Retrieval‑based와 gradient‑based 메모리의 결합  
- FSNet의 gradient‑based $\(u_l\)$ memory와 RAFT/TS‑RAG의 raw pattern retrieval을 결합하여,  
  - “어떤 업데이트를 했는지”와  
  - “그때의 실제 future trajectory”  
를 동시에 활용하는 hybrid 메모리 구조를 설계할 수 있습니다. [arxiv](https://arxiv.org/html/2503.07649v3)

6) 이론적 분석  
- FSNet의 gradient EMA, cosine trigger, sparse read/write가  
  - 어떤 조건에서 stability–plasticity trade-off를 보장하는지,  
  - online regret 및 forgetting bound 관점에서 어떤 상한을 가지는지  
는 아직 충분히 분석되지 않았습니다. NatSR에서 사용한 score-driven/NGD 이론을 차용해 FSNet을 분석하면, **일반화와 수렴 속도에 대한 수학적 이해**를 높일 수 있습니다. [openreview](https://openreview.net/forum?id=l4qsixUiir)

***

요약하면, FSNet은 온라인 시계열 예측을 continual learning으로 재정의하고, gradient EMA 기반 fast adapter와 연상 메모리를 통해 drift 환경에서의 일반화 및 적응 속도를 동시에 향상시킨 선도적 연구입니다. 이후 NatSR, RAFT, TS‑RAG, memory‑augmented TSFM 등은 이를 optimizer, retrieval, foundation 모델 방향으로 확장하고 있으며, 이러한 축들을 결합한 하이브리드 아키텍처와 irregular 시계열·대형 모델 적용이 앞으로의 중요한 연구 방향이 됩니다. [arxiv](https://arxiv.org/html/2601.12931v1)

<span style="display:none">[^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37]</span>

<div align="center">⁂</div>

[^1_1]: 2202.11672v2.pdf

[^1_2]: https://arxiv.org/abs/2202.11672

[^1_3]: https://www.semanticscholar.org/paper/Learning-Fast-and-Slow-for-Online-Time-Series-Pham-Liu/4d755a5a66a8c46b722b44613788085191524e11

[^1_4]: https://arxiv.org/abs/2601.12931v1

[^1_5]: https://arxiv.org/abs/2601.12931

[^1_6]: https://www.themoonlight.io/en/review/online-continual-learning-for-time-series-a-natural-score-driven-approach

[^1_7]: https://arxiv.org/html/2411.07413v1

[^1_8]: https://arxiv.org/html/2505.04163v1

[^1_9]: https://arxiv.org/html/2508.10824v1

[^1_10]: https://www.nature.com/articles/s41598-025-31685-9

[^1_11]: https://pdfs.semanticscholar.org/0678/8ff99baf69e8c8e032bdf2285774299ef1f6.pdf

[^1_12]: https://openreview.net/pdf/cdbc017543a68938c60b30486172e644325c2c19.pdf

[^1_13]: https://arxiv.org/html/2402.01999v1

[^1_14]: https://www.marktechpost.com/2022/11/02/salesforce-ai-developed-fsnet-fast-and-slow-learning-network-for-deep-time-series-forecasting-which-can-learn-deep-forecasting-models-on-the-fly-in-a-nonstationary-environment/

[^1_15]: https://arxiv.org/html/2412.08435v3

[^1_16]: https://arxiv.org/pdf/2310.10688.pdf

[^1_17]: http://arxiv.org/pdf/2408.12423.pdf

[^1_18]: https://www.semanticscholar.org/paper/4d755a5a66a8c46b722b44613788085191524e11

[^1_19]: https://arxiv.org/pdf/2202.11672.pdf

[^1_20]: http://arxiv.org/pdf/2411.04669v1.pdf

[^1_21]: http://arxiv.org/pdf/2411.17382.pdf

[^1_22]: https://arxiv.org/pdf/2110.03224.pdf

[^1_23]: http://arxiv.org/pdf/2310.19322.pdf

[^1_24]: http://arxiv.org/pdf/2502.12603.pdf

[^1_25]: https://arxiv.org/html/2509.03810v1

[^1_26]: https://www.arxiv.org/pdf/2510.15404.pdf

[^1_27]: https://arxiv.org/html/2501.04970v1

[^1_28]: https://arxiv.org/html/2601.12931v1

[^1_29]: https://arxiv.org/html/2510.06884v1

[^1_30]: https://arxiv.org/pdf/2601.12931.pdf

[^1_31]: https://arxiv.org/html/2502.12920v1

[^1_32]: https://www.x-mol.com/paper/1496886068456673280

[^1_33]: https://github.com/salesforce/fsnet

[^1_34]: https://smusg.elsevierpure.com/en/publications/learning-fast-and-slow-for-online-time-series-forecasting/

[^1_35]: https://www.signalpop.com/2023/11/04/understanding-fsnets-for-fast-and-slow-online-time-series-forecasting/

[^1_36]: https://www.sciencedirect.com/science/article/abs/pii/S0950705122011157

[^1_37]: https://www.alphaxiv.org/overview/2202.11672v2
