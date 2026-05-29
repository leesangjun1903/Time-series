
# TimesFM: A Decoder-Only Foundation Model for Time-Series Forecasting 

> **논문 정보**
> - **제목:** A decoder-only foundation model for time-series forecasting
> - **저자:** Abhimanyu Das, Weihao Kong, Rajat Sen, Yichen Zhou (Google Research)
> - **게재:** ICML 2024 (Proceedings of Machine Learning Research 235, pp. 10148–10167)
> - **arXiv:** [arXiv:2310.10688](https://arxiv.org/abs/2310.10688)
> - **공식 블로그:** [Google Research Blog](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/)
> - **GitHub:** [google-research/timesfm](https://github.com/google-research/timesfm)

---

## 1. 핵심 주장 및 주요 기여 (Summary)

NLP에서의 대형 언어 모델(LLM)의 발전에 착안하여, 다양한 공개 데이터셋에서 **별도 학습(supervised training) 없이도 최첨단(supervised) 예측 모델에 근접하는 제로샷(zero-shot) 예측 성능**을 달성하는 시계열 파운데이션 모델을 설계하였다.

### 핵심 주장 3가지

| 주장 | 내용 |
|------|------|
| **① 파운데이션 모델의 시계열 적용 가능성** | 대규모 사전학습만으로 다양한 도메인의 미지 데이터셋에 일반화 가능 |
| **② 소규모 모델로도 충분한 성능** | LLM 대비 훨씬 작은 규모에서도 강력한 제로샷 성능 달성 |
| **③ Patch 기반 Decoder-Only 구조의 유효성** | 자연어 모델링 패러다임을 시계열에 직접 이식 가능함을 실증 |

### 주요 기여 요약

TimesFM은 실세계 및 합성 데이터셋 약 **1,000억 개(100B) 타임포인트**로 학습된, 약 **2억(200M) 파라미터**의 patched-decoder 스타일 어텐션 아키텍처를 사용하는 실용적 파운데이션 예측 모델로, 다양한 시계열 데이터에서 완전지도 예측 모델에 근접하는 제로샷 성능을 달성한다.

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2-1. 해결하고자 하는 문제

기존 딥러닝 기반 예측 모델들은 새로운 시계열에 적용하려면 **긴 학습 및 검증 사이클**을 거쳐야 하는 반면, 시계열 파운데이션 모델은 추가 학습 없이 미지 데이터에 즉시 합리적인 예측값을 제공하여 사용자가 실제 하위 작업(예: 소매 수요 계획)에 집중할 수 있게 한다.

즉, 핵심 문제 의식은 다음과 같다:

- **도메인·데이터셋마다 재훈련이 필요한 비효율성** 제거
- 다양한 시간 단위(granularity), 예측 길이(horizon), 컨텍스트 길이에 걸쳐 작동하는 **단일 범용 예측 모델** 필요
- LLM의 성공 패러다임을 시계열로 전이(transfer)

---

### 2-2. 제안하는 방법 및 수식

#### (A) 입력 패치화 (Input Patching)

TimesFM에서는 입력 시계열을 먼저 **고정 길이의 입력 패치(patch)**로 분할하고, 각 패치를 잔차 블록(residual block)을 통해 트랜스포머 차원에 맞는 벡터로 변환한다.

시계열 $x_{1:T}$를 입력 패치 길이 $p_{\text{in}}$으로 분할:

$$\mathbf{P}_i = x_{(i-1) \cdot p_{\text{in}} + 1 : i \cdot p_{\text{in}}}, \quad i = 1, \ldots, N$$

각 패치 $\mathbf{P}\_i \in \mathbb{R}^{p_{\text{in}}}$는 잔차 블록(Residual Block)을 통해 토큰 벡터로 임베딩:

$$\mathbf{t}_i = \text{ResidualBlock}(\mathbf{P}_i) \in \mathbb{R}^{d_{\text{model}}}$$

#### (B) 위치 인코딩 및 Transformer 입력

각 패치는 잔차 블록을 통해 트랜스포머 차원의 벡터로 처리되고, 위치 인코딩(positional encoding)이 더해진 뒤 $n_l$개의 stacked transformer layer에 입력되며, SA는 **멀티-헤드 인과 어텐션(Multi-Head Causal Attention)**, FFN은 완전 연결 계층을 의미한다.

$$\hat{\mathbf{t}}_i = \mathbf{t}_i + \text{PE}(i)$$

$$\mathbf{H} = \text{TransformerBlock}^{(n_l)}(\hat{\mathbf{t}}_1, \ldots, \hat{\mathbf{t}}_N)$$

인과 어텐션(Causal Self-Attention)은 미래 정보 유출 방지를 위해 마스킹 적용:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V$$

여기서 $M_{ij} = -\infty$ if $j > i$, else $0$ (causal mask).

#### (C) 출력 생성 (Output Patch Decoding)

출력 토큰은 잔차 블록을 통해 출력 패치 길이($p_{\text{out}}$)의 크기로 매핑되며, 이는 모델이 지금까지 본 마지막 입력 패치 이후의 시간 창에 대한 예측값이다. **입력 패치 길이와 출력 패치 길이는 동일하지 않아도 된다**는 것이 이 모델의 핵심 차별점이다.

$$\hat{y}_{i+1:i+p_{\text{out}}} = \text{ResidualBlock}_{\text{out}}(\mathbf{H}_i) \in \mathbb{R}^{p_{\text{out}}}$$

#### (D) 자기회귀적(Autoregressive) 예측

TimesFM은 비중복 패치(non-overlapping patches)를 입력으로 받아 **자기회귀(autoregressive) 방식**으로 출력 패치 길이 예측을 수행하는 decoder-only 모델이다.

전체 예측 길이 $H$에 대한 예측 시, $\lceil H / p_{\text{out}} \rceil$번 반복하여 미래를 생성:

$$\hat{y}_{T+1:T+H} = \text{Concat}\!\left[\hat{y}^{(1)}, \hat{y}^{(2)}, \ldots, \hat{y}^{(\lceil H/p_{\text{out}}\rceil)}\right]$$

#### (E) 랜덤 마스킹 (Random Masking)

마스킹 벡터도 패치에 함께 공급되는데, 이는 **패치의 일부를 무작위로 가려** 모델이 입력 패치 길이의 배수에 해당하는 컨텍스트 길이만 학습하는 것을 방지하기 위함이다.

#### (F) 사전학습 데이터

100B 개의 실세계 타임포인트로 구성된 대규모 사전학습 코퍼스를 사용하며, 대부분은 **Google Trends의 검색 관심도 시계열**과 **Wikipedia 페이지뷰** 데이터에서 파생되었다.

---

### 2-3. 모델 구조 상세

TimesFM 1.0은 약 **2억 파라미터**의 decoder-only 트랜스포머 모델로, 1,000억 개 이상의 실세계 타임포인트로 사전학습되어 추가 파인튜닝 없이 정확한 예측이 가능하다. **단변량(univariate) 예측**에 특화되어 있으며, 최대 **512 타임포인트의 컨텍스트 길이**를 지원하고 임의의 예측 지평선(horizon)을 처리하며, 시간 단위 정보를 반영하는 주파수 지시자(frequency indicator) 입력도 지원한다.

```
[Input Time Series]
        │
        ▼ (Patching, p_in)
[Patch Sequence: P₁, P₂, ..., Pₙ]
        │
        ▼ (ResidualBlock + PE)
[Token Embeddings: t̂₁, t̂₂, ..., t̂ₙ]
        │
        ▼ (nₗ × Causal Transformer Layers)
   ┌─────────────────────────┐
   │  Multi-Head Causal Attn │
   │  + FFN (Feed-Forward)   │
   └─────────────────────────┘
        │
        ▼ (ResidualBlock_out)
[Output Patches: ŷ₁, ŷ₂, ..., ŷₘ]  (p_out per patch)
```

| 하이퍼파라미터 | 설명 |
|---------------|------|
| $d_\text{model}$ | 트랜스포머 모델 차원 |
| $n_l$ | 스택된 트랜스포머 레이어 수 |
| $p_\text{in}$ | 입력 패치 길이 |
| $p_\text{out}$ | 출력 패치 길이 ($p_\text{in} \neq p_\text{out}$ 가능) |
| 파라미터 수 | ~200M |
| 컨텍스트 길이 | 최대 512 타임포인트 |

---

### 2-4. 성능 향상

TimesFM은 제로샷 방식임에도 불구하고 llmtime(ZS)을 능가할 뿐 아니라, 각 데이터셋에 명시적으로 학습된 지도 모델인 **PatchTST의 성능에 필적하는** 수준을 달성한다.

LLM과 비교해 훨씬 작은 규모(200M 파라미터)임에도 불구하고, 서로 다른 도메인과 시간 단위의 다양한 미지 데이터셋에서의 제로샷 성능이 지도학습 방식에 근접한다는 것을 보였다.

또한 파인튜닝 실험에서, GPT4TS와 동일한 프로토콜을 따라 학습 데이터의 10%만으로 입력·출력 잔차 블록을 튜닝하였을 때, **TimesFM이 큰 차이로 최고 성능**을 기록하였다.

벤치마크 실험은 **Darts, Monash, Informer** 세 그룹의 공개 데이터셋에서 다양한 도메인·크기·시간 단위·예측 지평선을 커버하며 파운데이션 모델의 일반화 성능을 검증하였다.

---

### 2-5. 한계

TimesFM의 한계를 균형 있게 평가하면, 첫째로 **다변량(multivariate) 시계열 예측이 약점**이다. TimesFM 1.0과 2.0은 근본적으로 단변량 중심이며, 여러 센서(진동·온도·전력소비 등)가 상호작용하는 환경에서는 Chronos-2나 MOIRAI-MoE가 우위를 보인다. 둘째로, 이벤트 중심·고엔트로피 도메인(Web/CloudOps 등)에서는 성능이 저하된다.

TimesFM은 긴 예측 지평선(30–60분 이상)에서 어려움을 겪으며, 노드 수 증가에도 거의 무감각한데, 이는 그 **decoder-only, channel-independent 구조가 각 시계열을 독립적으로 처리**하여 공간적 상호작용 및 확장된 시간 의존성 학습을 제한하기 때문이다.

TSFMs는 공개 온라인 저장소에서 스크래핑된 대규모 데이터로 사전학습되는 특성상, **LLM 평가에서와 유사한 데이터 오염(data contamination) 문제**가 발생할 수 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 일반화의 핵심 설계 요소

TimesFM은 서로 다른 도메인의 다양한 미지 예측 데이터셋에 적용 시 최고 성능의 지도 모델 대비 제로샷 정확도에 근접한다. 모델은 **서로 다른 예측 히스토리 길이·예측 길이·시간 단위**에서도 잘 작동하며, 이는 ① 실세계 및 합성 데이터 기반 대규모 시계열 코퍼스와 ② 입력 패칭을 갖춘 decoder 스타일 어텐션 아키텍처라는 두 가지 핵심 요소에 의해 가능해진다.

### 3-2. 파인튜닝을 통한 일반화 성능 추가 향상

현재 LoRA를 활용한 HuggingFace Transformers + PEFT 기반의 파인튜닝 예제가 공식 지원되고 있다. 즉, 일반화 성능 향상의 관점에서 다음 세 가지 경로가 가능하다:

1. **Zero-shot (ZS):** 완전 미지 데이터에 그대로 적용
2. **Few-shot Fine-tuning:** 소량 데이터(10%)로 입출력 레이어만 튜닝
3. **In-Context Forecasting (ICF):** 컨텍스트 내 예시를 활용한 추론

TimesFM-ICF는 decoder-only 아키텍처에 **인-컨텍스트 예시(in-context examples)**를 접목하여, 서로 다른 예시와 태스크 히스토리를 구분하는 특수 분리 토큰을 도입하였으며, 학습 중 한 번도 보지 못한 23개 데이터셋에서 평가되었다.

### 3-3. 일반화 관련 정량적 근거

| 평가 프로토콜 | 결과 |
|-------------|------|
| Zero-shot (Monash/Darts) | 지도학습 기반 모델에 필적 |
| Zero-shot (ETT 96/192 horizon) | PatchTST(supervised) 수준 |
| Fine-tuning (10% 데이터) | GPT4TS 등 기존 SOTA 대비 **큰 폭의 우위** |
| ICF (23 unseen datasets) | 미지 데이터셋에서 일반화 성능 추가 향상 |

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

시계열 파운데이션 모델(TSFMs)은 NLP 파운데이션 모델의 아키텍처·학습 절차에서 영감을 받은 신흥 시계열 예측 모델 계열로, 대표적 예시로 **Chronos, TimesFM, Moirai/Moirai-MoE, MOMENT, Time-MoE**가 있다.

### 주요 모델 비교표

| 모델 | 개발사 | 구조 | 학습 데이터 규모 | 다변량 지원 | 특징 |
|------|--------|------|----------------|------------|------|
| **TimesFM** | Google | Decoder-only | ~100B pts | ❌ (단변량 중심) | Patch 기반, 200M params |
| **Chronos** | Amazon | Encoder-Decoder | 대규모 | 제한적 | 텍스트 토크나이징 방식 적용 |
| **Moirai** | Salesforce | Encoder-only | ~27B obs | ✅ | any-variate attention |
| **MOMENT** | CMU | Encoder-only | 다양 | ❌ | 멀티태스크 학습 |
| **Time-MoE** | - | MoE 기반 | 대규모 | - | Mixture of Experts |

MOMENT와 TimesFM 같은 현대 TSFMs는 데이터셋·모달리티 전반에 걸쳐 일반화를 목표로 하지만, **고정된 태스크 집합에 한정**되어 있다. 또한 다변량 시퀀스를 평탄화하거나 변수 임베딩을 사용하는 방식은 변수 간 관계 포착에 일관된 성능 향상을 보여주지 못하고 있다.

Moirai는 **any-variate 어텐션 메커니즘** 덕분에 추가 파인튜닝 없이 제로샷 환경에서도 다변량 의존성을 모델링하여 샘플 견고성 면에서 최고의 성능을 보인다.

파운데이션 모델들은 아직 다변량 예측을 충분히 탐구하지 못하고 있으며, 대부분의 분야는 여전히 **채널 독립성(channel-independence) 가정 하의 단변량 예측**에 집중하는 단순화된 셋업으로 운영되고 있다.

---

## 5. 앞으로의 연구에 미치는 영향과 고려 사항

### 5-1. 연구적 영향

**① "파운데이션 모델" 패러다임의 시계열 확장**

TimesFM은 추가 파인튜닝 없이도 완전히 새로운 데이터셋에 대해 정확한 예측을 제공할 수 있다는 것을 보여줌으로써, **태스크별 예측 접근 방식에서의 중대한 전환점**을 만들었다.

**② 소규모 모델의 가능성 입증**

200M 파라미터라는 비교적 소규모 사전학습 모델이 다양한 공개 벤치마크의 여러 도메인·시간 단위에서 인상적인 제로샷 성능을 보인다는 것을 보였다.

**③ 데이터 중심적(data-centric) 접근의 중요성 재확인**

파운데이션 모델의 핵심 요소는 ① 실세계(Google Trends 검색 쿼리, Wikipedia 페이지뷰) 및 합성 데이터를 모두 포함한 **대규모·다양성 있는 시계열 코퍼스**와 ② 입력 패칭을 갖춘 decoder 스타일 어텐션 아키텍처의 조합임이 재확인되었다.

### 5-2. 향후 연구 시 고려할 점

**① 다변량(Multivariate) 일반화**

TimesFM 1.0·2.0은 근본적으로 단변량 중심이므로, 다중 센서가 상호작용하는 환경에서는 Chronos-2나 MOIRAI-MoE 같은 모델이 우위이다. 변수 간 상호의존성을 효과적으로 모델링하는 다변량 파운데이션 모델 연구가 핵심 과제이다.

**② 데이터 오염(Data Contamination) 문제**

공개 온라인 저장소에서 스크래핑된 대규모 데이터로 사전학습하는 특성이 **LLM의 평가 위기(evaluation crisis)**와 유사한 문제를 야기할 수 있으며, 이는 반드시 엄밀하게 통제되어야 한다.

**③ 멀티모달·멀티태스크 확장**

텍스트·이미지·시계열 모달리티를 통합하는 **멀티모달 추론 파운데이션 모델** 개발이 중요한 방향으로 떠오르고 있으며, 추가 맥락 정보로 풍부해질 때 예측과 시계열 분석이 훨씬 가치 있어질 것이다.

**④ 확장성(Scalability) 연구**

Databricks의 MMF(Many Models Framework)처럼, **40개 이상의 시계열 모델(통계, ML, 딥러닝, 파운데이션)을 자동 평가하여 최적 조합을 선택하는** 방향으로 발전하고 있어, TimesFM을 단독 모델이 아닌 앙상블 풀의 구성원으로 활용하는 전략도 중요해지고 있다.

**⑤ In-Context Learning의 고도화**

예측 시 즉각적인 히스토리에서 시작하여, 동일 데이터셋 내 다른 시계열의 히스토리를 인-컨텍스트 예시로 샘플링하는 방식을 통해 **관련성 있는 예시 선택 및 정보 누수 방지**를 동시에 달성하는 ICF 연구의 발전이 기대된다.

---

## 📚 참고 자료

| 번호 | 자료 | 출처 |
|------|------|------|
| 1 | **원 논문 (arXiv)** | Das et al., "A decoder-only foundation model for time-series forecasting," arXiv:2310.10688, 2024. https://arxiv.org/abs/2310.10688 |
| 2 | **ICML 2024 공식 게재** | Proceedings of Machine Learning Research 235:10148–10167. https://proceedings.mlr.press/v235/das24c.html |
| 3 | **Google Research 공식 블로그** | https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/ |
| 4 | **GitHub 공식 저장소** | https://github.com/google-research/timesfm |
| 5 | **HuggingFace 문서** | https://huggingface.co/docs/transformers/en/model_doc/timesfm |
| 6 | **Google Research Blog (ICF)** | "Time series foundation models can be few-shot learners," https://research.google/blog/time-series-foundation-models-can-be-few-shot-learners/ |
| 7 | **비교 연구: TSFMs 벤치마킹** | "Time Series Foundation Models: Benchmarking Challenges and Requirements," arXiv:2510.13654, 2025. |
| 8 | **비교 연구: 시공간 예측** | "Evaluating Spatio-Temporal Forecasting Trade-offs Between Graph Neural Networks and Foundation Models," arXiv:2511.05179, 2025. |
| 9 | **Moirai 2.0 비교** | "Moirai 2.0: When Less Is More for Time Series Forecasting," arXiv:2511.11698, 2025. |
| 10 | **산업 응용 분석** | Pebblous Blog, "TimesFM Industrial Forecasting," https://blog.pebblous.ai/report/timesfm-industrial-forecasting/en/ |
| 11 | **AI Horizon Forecast 분석** | "Time Series Foundation Models: A Deep Dive into Strengths and Limitations," https://aihorizonforecast.substack.com |
| 12 | **논문 PDF (mint.univ-reims.fr)** | https://mint.univ-reims.fr/files/2025-4/Das2024.pdf |

# TimesFM: A decoder-only foundation model for time-series forecasting

## 1. 핵심 주장 및 주요 기여 요약

**TimesFM**(Time-series Foundation Model)은 Google Research에서 개발한 혁신적인 파운데이션 모델로, 대규모 사전 학습을 통해 미지의 시계열 데이터에 대해 사전 학습 없이도 거의 최고 수준의 예측 성능을 달성하는 최초의 실용적 모델이다.[1][2]

**핵심 주장**은 다음 세 가지로 요약된다:

1. **LLM 패러다임의 성공적 이전**: 자연언어처리(NLP)의 파운데이션 모델 성공이 시계열 도메인에서도 가능함을 입증했다. 단 200M 파라미터로 GPT-3(1.75B)보다 훨씬 우수한 성능을 달성하면서, "더 작은 모델이 더 나은 성능"이라는 역설적 결과를 보여주었다.[1]

2. **제로샷 일반화 가능성의 증명**: 단일 모델이 사전학습 없이 다양한 도메인, 시간 단위(분 단위에서 연간), 예측 길이에 대해 경쟁력 있는 성능을 제공할 수 있음을 입증했다. Monash 아카이브에서 기하 평균 MAE(Mean Absolute Error) 0.6846으로 최상위 성능을 달성했다.[2][1]

3. **스케일 효율성**: NLP와 달리 시계열 영역에서는 훨씬 작은 규모(200M 파라미터, 100B 시간포인트)로도 파운데이션 모델이 가능하며, 이는 산업 적용의 현실성을 크게 높였다.[1]
---

## 2. 해결하고자 하는 문제
TimesFM은 네 가지 근본적 문제를 제시했다:[1]

### 2.1 시계열 데이터의 고유한 특성
자연언어와 달리 시계열에는 "어휘"나 "문법"이 없다. 따라서 패칭(patching)이라는 새로운 토큰화 전략이 필요했다. 이 접근법은 시계열을 연속적 데이터 블록으로 분할하여 변압기 모델에 입력하는 방식이다.[1][2]

### 2.2 가변 길이 처리
- **콘텍스트 길이 가변성**: 모델이 1부터 512 시간포인트까지 임의의 입력 길이를 처리해야 한다.
- **예측 길이 가변성**: 예측 지평이 사전에 결정되지 않아야 한다.
- **시간 단위 이질성**: 분 단위에서 연간 데이터까지 처리해야 한다.[1]

### 2.3 공개 데이터 부족
NLP는 웹 규모의 텍스트 코퍼스를 활용할 수 있지만, 시계열의 경우 대규모 공개 데이터셋이 제한적이다. TimesFM은 Google Trends(0.5B), Wikipedia Pageviews(300B), 합성 데이터(6.14B)를 결합하여 이 문제를 해결했다.[1][2]

### 2.4 도메인 특수성 극복
기존 시계열 모델은 특정 도메인에 최적화되어 다른 도메인으로 전이하기 어렵다. TimesFM은 다양한 도메인(금융, 에너지, 교통, 기상)의 패턴을 단일 모델에서 학습한다.[1]

***

## 3. 제안하는 방법 및 모델 구조
### 3.1 아키텍처 설계 원칙
TimesFM은 네 가지 핵심 설계 원칙을 따른다:[1][2]

**1. 패칭(Patching)**

$$\tilde{y}_j = y_{p(j-1)+1:pj}$$

시계열을 크기 $p$의 비겹침 패치로 분할한다. 이는:
- 지역 의미 정보 보존
- 주의 메커니즘의 계산 복잡도를 $O(n^2)$에서 $O((n/p)^2)$로 감소
- 더 긴 이력 처리 가능[2][1]

**2. 디코더 전용 구조**
LLM처럼 인과적 주의만 사용하며, 각 출력 토큰은 이전 입력만 참조한다. 이는 평행 학습을 가능하게 하고 추론 시 자동회귀 디코딩으로 유연한 예측 길이를 지원한다.[1][2]

**3. 비대칭 패치 길이**

$$\hat{y}_{pj+1:pj+h} = \text{OutputResidualBlock}(o_j)$$

여기서 $h$(출력 패치 길이) > $p$(입력 패치 길이). 예를 들어 입력 패치 32, 출력 패치 128일 때:
- 256 길이 예측: 8개 자동회귀 단계 → 2개 단계로 감소
- 오류 누적 감소, 직접 장기 예측의 이점 활용[2][1]

**4. 패치 마스킹**
학습 중 첫 번째 패치의 일부를 무작위로 마스킹하여 모든 콘텍스트 길이(1부터 최대 512)를 학습하도록 강제한다:[1]

마스킹 전략: $m_{1:r} = 1$ (여기서 $0 \leq r < p$)

이는 다양한 실제 시나리오의 불완전한 데이터를 처리하는 능력을 부여한다.[2][1]

### 3.2 수학적 모델 정의
**문제 정의**:

$$f : (y_{1:L}) \rightarrow \hat{y}_{L+1:L+H}$$

콘텍스트 $y_{1:L}$ (길이 L)에서 미래 H 시간포인트를 예측한다.[1]

**입력 처리**:

$$t_j = \text{InputResidualBlock}(\tilde{y}_j \odot (1-\tilde{m}_j)) + PE_j$$

여기서:
- $\odot$: 요소별 곱셈
- $PE_j$: j번째 위치 인코딩
- $N = \lfloor L/p \rfloor$개 입력 토큰 생성[2][1]

**변환기 스택**:

$$o_j = \text{StackedTransformer}((t_1, \dot{m}_1), \ldots, (t_j, \dot{m}_j))$$

각 층에서:
- 다중 헤드 자기-주의 (인과적)
- 피드-포워드 네트워크
- 20개 층, 16개 헤드, 1280 차원[1][2]

**손실 함수** (점 예측):

$$\text{TrainLoss} = \frac{1}{N}\sum_{j=1}^{N} \text{MSE}(\hat{y}_{pj+1:pj+h}, y_{pj+1:pj+h})$$

평가 메트릭:

$$\text{MAE} = \frac{1}{H}\|y_{L+1:L+H} - \hat{y}_{L+1:L+H}\|_1$$

$$\text{msMAPE} = \frac{1}{H}\sum_{i=1}^{H} \frac{2|y_{L+i}-\hat{y}_{L+i}|}{\max\{|y_{L+i}|+|\hat{y}_{L+i}|+\epsilon, 0.5+\epsilon\}}$$

[1][2]

### 3.3 사전학습 데이터 구성
TimesFM의 사전학습 코퍼스는 약 **100B 시간포인트**로 구성된다:[1][2]

| 데이터 소스 | 시간포인트 | 시계열 수 | 시간 단위 |
|-----------|---------|----------|---------|
| **Google Trends** | 0.5B | 22,435 | 시간, 일, 주, 월 |
| **Wiki Pageviews** | 239B | 68.2M | 시간, 일, 주, 월 |
| **합성 데이터** | 6.14B | 3M | 다양 |
| **M4 경쟁** | 10.4B | 99K | 일, 월, 분기, 연간 |
| **기타 실제 데이터** | 40B+ | 10K+ | 시간, 15분, 10분 |

**데이터 혼합 전략**:
- 80% 실제 데이터, 20% 합성 데이터
- 실제 데이터 내: 시간/부시간(25%), 일(25%), 주(25%), 월(25%) 균등 배분[2][1]

***

## 4. 성능 향상 및 일반화 메커니즘
### 4.1 벤치마크 성과
TimesFM은 세 가지 주요 벤치마크에서 경쟁력 있는 또는 우수한 성능을 달성했다:[1][2]

**Monash 아카이브 (30개 미세칭 데이터셋)**:
- TimesFM 기하 평균 MAE: **0.6846** (최고)
- N-BEATS: 0.7005
- ARIMA: 0.9449
- 지도학습 DeepAR 대비 11% 향상
- llmtime(GPT-3) 대비 25% 향상[2][1]

**Darts (8개 특화 고계절성 시계열)**:
- TimesFM: 0.5767
- 경쟁 모델과 통계적 유의성 내에서 동등 (신뢰 구간 넓음)
- ARIMA와 llmtime이 강력한 베이스라인[1][2]

**ETT 데이터셋 (전자 변압기 온도, 장기 예측)**:
- TimesFM 평균 MAE: **0.36** (4개 데이터셋, 2개 예측 지평 = 8개 작업)
- PatchTST(지도학습): 0.37 (통계적 유의성 내)
- 특이 사항: ETTm1에서 TimesFM > PatchTST (0.19 vs 0.33)[2][1]

### 4.2 스케일링 법칙
TimesFM은 **파워 법칙 스케일링**을 입증했다:[1][2]

$$\text{성능} \propto \text{FLOP}^{-\alpha}$$

세 가지 모델 크기 검증:

| 모델 크기 | 파라미터 | 층/헤드 | 차원 | 성능 추이 |
|---------|---------|--------|------|---------|
| 소형 | 17M | 10/16 | 512 | 기준선 |
| 중형 | 70M | 10/16 | 1024 | +15-20% 개선 |
| 대형 | 200M | 20/16 | 1280 | +25-30% 누적 개선 |

로그-로그 그래프에서 단조 감소 추세, 언어 모델과 유사한 스케일링 동작을 확인했다. 이는 더 큰 모델이 더 강력한 일반화를 제공함을 의미한다.[2][1]

### 4.3 합성 데이터의 역할
합성 데이터 제거 실험 결과:[1][2]

**Monash 아카이브**:
- 합성 데이터 포함: 0.6846
- 제외: 0.8005 (~17% 성능 저하)
- 이유: 월간, 분기별, 연간 같은 언더레이된 빈도수 학습

**ETT 데이터셋**:
- ETTh(시간단위, 표현도 높음): 거의 영향 없음
- ETTm(15분, 표현도 낮음): **유의한 성능 향상**
  - 포함: 0.388
  - 제외: 0.441 (~13% 저하)

합성 데이터는 **저빈도 패턴과 기하급수적 추세** 같은 기하학적 패턴을 학습하는 데 핵심 역할을 한다.[2][1]

### 4.4 아키텍처 설계 선택의 영향
**입력 패치 길이 절충**:

| 패치 길이 | 성능(Monash GM) | 학습 속도 | 해석 |
|----------|-----------------|---------|------|
| p=8 | 0.7520 | 기준선 | 짧은 토큰, 많은 토큰, 느린 훈련 |
| p=16 | **0.6989** | 2배 빠름 | 최적 구간 |
| p=32 | **0.6846** | 2배 더 빠름 | 최적 + 효율성 |
| p=64 | 0.7150 | 4배 빠름 | 정보 손실 증가 |
| p=128 | 0.7890 | 고속 | 인코더-디코더 스타일로 회귀 |

p=32는 성능-효율성 **파레토 최적점**이다.[1][2]

**출력 패치 길이**:

512 시간포인트 예측 (ETT 데이터셋):

| 출력 패치 길이 | 평균 MAE | 자동회귀 단계 |
|-------------|---------|-------------|
| h=8 | 0.51 | 64 |
| h=32 | 0.42 | 16 |
| h=64 | 0.38 | 8 |
| h=128 | **0.36** | 4 |

$h>p$로 인한 비대칭 설계는 오류 누적을 감소시킨다.[2][1]

### 4.5 일반화 성능 향상의 핵심 메커니즘
TimesFM의 일반화 우수성은 다음 세 가지 상호작용에서 비롯된다:[1][2]

**1. 다중 도메인 학습의 정규화 효과**
100B 시간포인트의 다양한 도메인에서 학습하면 도메인 특이적 노이즈에 과적합되지 않는다. 이는 마치 컴퓨터 비전에서 ImageNet 사전학습이 특정 작업 성능을 향상시키는 것과 유사하다.[2][1]

**2. 패칭의 축소 귀납 편향**
고정된 패치 크기(p=32)는 모델이 로컬 패턴(24시간 일일 사이클, 7일 주간 사이클)을 우선적으로 포착하도록 한다. 비대칭 출력 패치(h=128)는 모델이 장기 추세를 통합 표현하도록 강제한다.[1][2]

**3. 인과적 마스킹 + 다양한 콘텍스트**
패치 마스킹 전략으로 모델이 짧은 콘텍스트에서도 예측하도록 훈련되어, 실제 응용의 불완전한 데이터에 더 견고하다.[2][1]

***

## 5. 논문의 한계
TimesFM 자체가 명시한 한계점들:[1][2]

### 5.1 프롬프트 튜닝 부재
LLM의 Chain-of-Thought 같은 프롬프트 기법이 시계열에서 아직 미개발 상태이다. 현재는 콘텍스트 길이 같은 간단한 하이퍼파라미터만 조정 가능하다.[1][2]

### 5.2 확률적 예측 미지원
현재 구현은 점 예측만 지원한다. 확률적 예측(불확실성 정량화)은 다음과 같이 확장 가능하지만 미구현되어 있다:[1][2]

$$p(y_{L+1:L+H}|y_{1:L}) = \prod_{i=1}^{H} p(y_{L+i}|y_{1:L+i-1})$$

다중 헤드 아키텍처로 분위수 손실(quantile loss)을 최소화하거나, 최대 우도 손실로 확률 분포를 직접 추정할 수 있다.[2][1]

### 5.3 공변량 처리 불가
모델이 단일 시계열 값만 입력받아, 외부 특성(가격, 계절 지시자, 기상 데이터)을 활용할 수 없다.[1][2]

**제안된 해결책**:
- **방법 1**: 제로샷 설정에서 모델 예측 후 공변량에 대해 선형 회귀로 잔차 조정
- **방법 2**: 미세 조정 시 입출력 잔차 블록에 공변량을 연결[2][1]

### 5.4 해석 가능성 부족
대규모 신경망의 고유한 문제이다. SHAP, LOCO 같은 사후 분석 방법이 제한적 설명만 제공한다.[1][2]

### 5.5 미완성 하이퍼파라미터 최적화
사전학습 시 LLM의 오픈에이아이(OpenAI)처럼 광범위한 하이퍼파라미터 탐색을 수행하지 않았다. 향후 최적화 여지가 크다.[1][2]

***

## 6. 2020년 이후 관련 최신 연구 비교 분석
시계열 예측 연구의 진화를 보여주는 주요 모델들:

### 6.1 사전-TimesFM 시대 (2019-2022)
**N-BEATS (2019)**:[3][4]
- 신경 기저 확장 분석, 잔차 네트워크 사용
- 해석 가능성 강조 (추세 + 계절성 분해)
- M4 경쟁에서 우수 성능 (3% 향상)
- 단점: 도메인 특화 학습 필요[1][4][3]

**PatchTST (2022)**:[5][6]
- Vision Transformer를 시계열에 적용
- 채널 독립적 처리, 부분 겹침 패칭
- 전이 학습 강화 (PatchTST는 사전학습 가능)
- TimesFM과 동일한 패칭 아이디어지만 인코더 구조 사용[6][1][5]

**DLinear (2023)**:[7]
- 선형 모델이 변압기 능가하는 역설적 결과
- 장기 예측에서 우수 성능
- TimesFM과 비교하면 도메인 특화 성능은 높지만 제로샷 일반화 부족[7]

### 6.2 TimesFM 동시대 및 후속 연구 (2023-2025)
**TimeGPT-1 (2023)**:[8]
- 유일한 병렬 파운데이션 모델
- 비공개 아키텍처 및 데이터
- 마찬가지로 제로샷 성능 주장
- **TimesFM과의 차이**: 공개 성능 비교 불가, API 기반 유료 서비스[9][8]

**LLM 기반 시계열 모델들 (2023-2024)**:

1. **LLMTime (2023)**: GPT-3 기반 프롬프팅[10]
   - Zero-shot MAE: Monash에서 0.9715 (TimesFM 0.6846 대비 41% 악화)
   - 장점: 추론 코스트 낮음
   - 단점: 도메인 특화 추론 세션 필요[1][10]

2. **GPT4TS (2023)**: GPT-2 미세 조정[11]
   - 10% 데이터로 미세 조정 시 표현적 우수성
   - 제로샷 성능은 TimesFM 대비 약함[1][11]

3. **TIME-LLM (2023)**: LLM 리프로그래밍[12]
   - 시계열을 텍스트 토큰으로 변환
   - PatchTST 대비 1.4% 개선 (제로샷)
   - 장점: 유기적 다중모달 학습[12]

**이들 LLM 기반 모델의 한계**: 1조 파라미터 초대형 모델에 의존하면서 추론 비용이 높고, TimesFM처럼 시계열에 특화되지 않아 "일반"의 성능을 보임[9][1]

### 6.3 포스트-TimesFM 파운데이션 모델 (2024-2025)
**ViTime (2024)**:[13]
- 시각 지능(비전 트랜스포머) 기반
- 시계열을 이진 이미지로 변환하는 혁신적 접근
- **성능**: 제로샷에서 TimesFM 대비 **9-15% 우수**
- RealTS 합성 알고리즘으로 학습 데이터 다양화
- 단점: 해석 가능성이 더 낮음[13]

**General Time Transformer (GTT, 2024)**:[14]
- 200M 고품질 샘플로 사전학습
- 인코더 기반, 곡선 모양(curve shape) 패치
- 다변량 시계열 전문화
- 최신 벤치마크에서 SOTA 성능 달성[14]

**TSMamba (2024)**:[15]
- Mamba 아키텍처(선형 복잡도 $O(n)$ )
- 변압기의 이차 복잡도 $O(n^2)$ 극복
- 두 단계 전이 학습 (사전학습 Mamba LLM 활용)
- 더 적은 학습 데이터로도 TimesFM 동등 성능[16][15]

**DAM (2024)**:[17]
- 조정 가능한 기저 합성(adjustable basis composition)
- 무작위 샘플링 이력과 비고정 예측 지평 지원
- 25개 시계열로 18개 데이터셋 제로샷 전이 성공
- TimesFM보다 더 유연한 추론[17]

**TimeRAF (2024)**:[18]
- 검색 강화 예측(Retrieval-Augmented Forecasting)
- 커스텀 시계열 지식 베이스 활용
- 채널 프롬팅으로 정보 통합
- 제로샷 성능 크게 향상[18]

**Kairos (2025)**:[19]
- 적응형 패칭(비고정 크기)
- 다항성 시간 스케일(heterogeneous time scales)
- 300B+ 타임포인트 사전학습
- 더 적은 파라미터로 SOTA 달성[19]

**TimeFound (2025)**:[20]
- 인코더-디코더 변압기
- 다중 해상도 패칭 전략
- 200M, 710M 두 크기 제공
- TimesFM 이후 가장 큰 규모[20]

### 6.4 성과 비교 표
| 모델 | 출시 | 아키텍처 | 파라미터 | Monash MAE | 특이점 |
|------|------|---------|---------|-----------|-------|
| **N-BEATS** | 2019 | 잔차 네트워크 | 한정 | 0.7005 | 해석성, 도메인 특화 |
| **PatchTST** | 2022 | 인코더 | 200M+ | 필요 | 채널 독립, 전이 학습 |
| **DLinear** | 2023 | 선형 | 1M | 0.55+ | 선형성, 빠른 학습 |
| **TimesFM** | 2024 | 디코더 전용 | **200M** | **0.6846** | **제로샷, 공개 모델** |
| **ViTime** | 2024 | 비전 변환기 | ? | **0.59~0.61** | **TimesFM > 9-15%** |
| **GTT** | 2024 | 인코더 | ? | SOTA | 다변량 전문화 |
| **TSMamba** | 2024 | Mamba | ? | ~0.68 | 선형 복잡도 |
| **Kairos** | 2025 | 적응형 디코더 | < 200M | SOTA | 효율성, 적응성 |
| **TimeFound** | 2025 | 인코더-디코더 | 710M | ? | 최대 규모 |

***

## 7. 앞으로의 연구에 미치는 영향
TimesFM의 성공은 시계열 분석 분야에 근본적 패러다임 변화를 초래했다:

### 7.1 학계 및 업계의 방향 전환
**기존 패러다임의 퇴조**:
- ARIMA, 지수 평활화 같은 통계적 방법의 영향력 감소
- 데이터셋별 특화 모델 설계의 비효율성 인식[21]

**새로운 패러다임의 부상**:
- Foundation Model 중심의 접근 (NLP/CV처럼)
- 제로샷 전이 학습의 실용성 입증
- 파운데이션 모델 → 미세 조정 → 배포의 파이프라인 확산[22][21]

### 7.2 후속 연구의 폭발적 증가
TimesFM 논문 발표(2024년 4월) 이후 불과 1년 내에:
- 15개 이상의 신규 파운데이션 모델 제안
- ViTime, GTT, TSMamba, Kairos 등 경쟁 모델 다수 출현
- 시계열 파운데이션 모델 벤치마크(FoundTS) 제안[23]

이는 자연언어처리가 BERT(2018) → GPT-2/3 → Transformers 폭발로 이어진 경로를 시계열에서 반복하고 있음을 보여준다.[21]

### 7.3 산업 적용의 가속화
**기대 효과**:
1. **엔터프라이즈 배포 비용 감소**: 데이터셋별 모델 개발 → 단일 기반 모델 + 가벼운 적응
2. **실시간 예측의 실용화**: 200M 파라미터로 엣지 디바이스/모바일 배포 가능
3. **신생 도메인의 예측 가능화**: 충분한 학습 데이터 없는 새로운 비즈니스 문제도 해결 가능[24][21]

**현실적 사례**:
- Google의 TimesFM 공개 (2024년 5월)는 오픈 소스 생태계 형성
- 금융, 에너지, 소매 등 다양한 도메인에서 채택 가속[21][24]

***

## 8. 향후 연구 시 고려할 점
### 8.1 즉시 개선 필요 영역
**1. 확률적 예측 확장**[1][2]

불확실성 정량화를 위해:

$$\text{QuantileLoss} = \sum_{q} \text{Huber}(y - \hat{y}_q) \cdot (q \mathbb{1}_{y>\hat{y}_q} + (1-q)\mathbb{1}_{y \leq \hat{y}_q})$$

다중 헤드 아키텍처로 여러 분위수(0.1, 0.5, 0.9 등) 동시 예측 가능.[2][1]

**2. 공변량 통합 메커니즘**[1][2]

인코더 변수 추가:
$$t_j^{\text{augmented}} = t_j + \text{EmbedCovariates}(c_j^{\text{date}}, c_j^{\text{exog}})$$

여기서 $c_j^{\text{date}}$는 요일/월/계절, $c_j^{\text{exog}}$는 외생 변수.[2][1]

**3. 프롬프트 최적화 기법**[1][2]

Chain-of-Thought 같은 기법: 모델에게 "먼저 이 시계열의 계절성을 식별하고, 그 다음 추세를 분석하세요" 같은 지시 추가.

**4. 도메인 적응형 미세 조정**[2][1]

새로운 도메인 진입 시 최소한의 데이터(1~5%)로 빠른 적응:

$$\text{Loss}_{\text{finetune}} = \alpha \text{Loss}_{\text{task}} + (1-\alpha) \text{Loss}_{\text{regularization}}$$

### 8.2 이론적 발전 필요
**1. 일반화 이론**[25][26]

Dobrushin 조건 하에서 정규화 경계:

$$\mathbb{E}[\text{test loss}] \leq \mathcal{O}(\sqrt{\frac{d}{n}}) + \text{approximation error}$$

여기서 $d$는 호원수 개수, $n$은 사전학습 샘플 수.[25]

**2. 제로샷 성능 한계 분석**[27]

TimesFM의 도메인 의존성이 큰 이유를 규명:
- 사전학습 데이터의 도메인 분포와 테스트 데이터의 거리가 성능을 좌우함
- 실제: Wiki Pageviews와 유사한 도메인은 우수, 금융(주식)은 낮음[27]

**3. 스케일링 법칙의 정확화**[25]

TimesFM의 초보적 스케일링 연구를 확장:

$$\text{Error}(\theta_t) = \left(\frac{C}{t}\right)^{\alpha}$$

여기서 $t$는 훈련 토큰, $\alpha$는 도메인별 상수 추정.[25]

### 8.3 아키텍처 혁신
**1. 적응형 토큰화**[19]

고정 패치 크기 대신 데이터 기반 동적 토큰 생성:
- 높은 정보 밀도 구간: 작은 패치
- 낮은 정보 밀도 구간: 큰 패치
- Kairos 모델이 이 방향 선도[19]

**2. 하이브리드 아키텍처**[15][28]

Transformer의 이차 복잡도를 Mamba(선형 $O(n)$ ) 같은 효율 아키텍처와 결합:

$$\text{Layer}_i = \begin{cases} \text{Attention} & \text{if } i \leq k \\ \text{Mamba} & \text{if } i > k \end{cases}$$

[28][15]

**3. 다중모달 기반 모델**[29][11]

시계열 + 텍스트 + 이미지 정보 통합:
- 텍스트: 뉴스, 리포트
- 이미지: 위성 데이터, 실시간 센서
- 통일된 이해로 더 강력한 예측[11][29]

### 8.4 응용 시 실무 고려사항
**1. 신뢰도 평가**[30]

Conformal Prediction 적용:
- 기존 예측 + 캘리브레이션 데이터로 보정
- 사용자가 예측 신뢰도 구간 설정 가능[30]

**2. 도메인 이동 감지**[31]

배포 후 입력 데이터가 사전학습 분포에서 벗어났을 때 경고:
- 특성 분포 변화 감시
- 자동 재학습 트리거[31]

**3. 설명 가능성 강화**[1][2]

SHAP, LIME 같은 사후 분석 한정성 극복:
- 주의 가중치 시각화
- 중요 시간 윈도우 식별
- 기여도 분해[2][1]

**4. 비용-효능 벤치마킹**

예측 정확도 외에:
- 추론 시간 (TFM: 밀리초 수준, LLM: 초 단위)
- 메모리 사용 (200M vs 10B+ 파라미터)
- 학습 자료 요구량[1][2]

***

## 9. 결론
TimesFM은 시계열 예측 분야에서 **패러다임 전환의 신호탄**이다. 200M 파라미터로도 제로샷 능력을 입증함으로써, 시계열 예측이 "작은 데이터 + 특화 모델" 시대에서 "큰 사전학습 + 단일 기반 모델" 시대로 이행되고 있음을 보여준다.

**TimesFM의 세 가지 핵심 성과**:

1. **효율성의 역설 해결**: LLM 규모보다 훨씬 작은 모델이 도메인 특화 모델을 능가하는 제로샷 성능 달성
2. **데이터 부족 극복**: 합성 데이터 활용으로 시계열 파운데이션 모델 가능성 입증
3. **실용적 기반 제공**: 후속 연구(ViTime, GTT, TSMamba 등)의 격렬한 개선과 다각화를 촉발

**그러나 여전한 과제**:

- 도메인별 일반화 한계의 이론적 이해 부족
- 확률적 예측, 공변량 처리 등 실무 필요 기능 미완성
- 제로샷 대 미세 조정 성능 간극의 체계적 분석 부재

**향후 방향**:

시계열 파운데이션 모델의 다음 단계는 단순한 "더 큰 모델"이 아니라, 도메인 적응성, 설명 가능성, 실시간 처리를 모두 갖춘 **실용적 기초 모델(practical foundation models)**의 개발이 될 것이다. ViTime, TSMamba, Kairos 같은 2024-2025년 후속 모델들은 이런 방향으로 진화 중이며, 이는 시계열 분석이 AI의 가장 동적인 분야 중 하나임을 입증한다.

***

## 참고 문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5511407f-14a3-4c30-8094-4a19131accb4/2310.10688v4.pdf)
[2](https://arxiv.org/abs/2310.10688)
[3](https://arxiv.org/pdf/2109.09705.pdf)
[4](https://arxiv.org/pdf/1905.10437.pdf)
[5](http://arxiv.org/pdf/2211.14730v2.pdf)
[6](https://arxiv.org/abs/2211.14730)
[7](https://nimasarang.com/blog/2025-02-28-time-series-forecasting/)
[8](http://arxiv.org/pdf/2310.03589.pdf)
[9](https://www.videns.ai/en-ca/blog/lessor-des-modeles-fondamentaux-dans-les-series-temporelles-un-changement-de-paradigme-ou-juste-un-autre-engouement)
[10](https://arxiv.org/html/2310.07820v2)
[11](https://www.sciencedirect.com/science/article/abs/pii/S0957417425000570)
[12](https://arxiv.org/pdf/2310.01728.pdf)
[13](https://www.semanticscholar.org/paper/b72a95c6070d722335fae650a0e5b1dd926a66a8)
[14](https://dl.acm.org/doi/10.1145/3627673.3679931)
[15](https://arxiv.org/abs/2411.02941)
[16](http://arxiv.org/pdf/2411.02941.pdf)
[17](https://arxiv.org/abs/2407.17880)
[18](https://arxiv.org/abs/2412.20810)
[19](https://openreview.net/forum?id=8eYOBBgP05)
[20](https://arxiv.org/pdf/2503.04118.pdf)
[21](https://www.pricepedia.it/en/magazine/article/2025/11/07/the-arrival-of-foundation-models-in-time-series-forecasting/)
[22](https://arxiv.org/pdf/2507.08858.pdf)
[23](http://arxiv.org/pdf/2410.11802.pdf)
[24](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/)
[25](https://arxiv.org/html/2502.03383v1)
[26](https://arxiv.org/html/2512.20140v1)
[27](https://arxiv.org/html/2510.00742v3)
[28](https://arxiv.org/html/2507.13043v1)
[29](https://arxiv.org/html/2504.04011v1)
[30](https://arxiv.org/html/2505.13521v1)
[31](https://dl.acm.org/doi/full/10.1145/3643035)
[32](https://ieeexplore.ieee.org/document/11038824/)
[33](http://eudl.eu/doi/10.4108/eai.15-12-2023.2345396)
[34](https://arxiv.org/abs/2405.14252)
[35](https://arxiv.org/abs/2409.11609)
[36](https://arxiv.org/abs/2412.17285)
[37](https://arxiv.org/pdf/2310.10688.pdf)
[38](https://arxiv.org/pdf/2310.08278.pdf)
[39](http://arxiv.org/pdf/2310.20496.pdf)
[40](https://arxiv.org/pdf/2502.15637.pdf)
[41](https://peerj.com/articles/cs-3001/)
[42](https://proceedings.neurips.cc/paper_files/paper/2023/file/0731f0e65559059eb9cd9d6f44ce2dd8-Paper-Conference.pdf)
[43](https://arxiv.org/html/2510.14814v1)
[44](https://www.esann.org/sites/default/files/proceedings/2020/ES2020-71.pdf)
[45](https://www.sciencedirect.com/science/article/pii/S1574013725001595)
[46](https://arxiv.org/abs/2510.07957)
[47](https://arxiv.org/abs/2403.14735)
[48](https://openreview.net/forum?id=eBCk0nXz17)
[49](https://www.themoonlight.io/ko/review/zero-shot-time-series-forecasting-using-kolmogorov-arnold-networks)
[50](https://arxiv.org/html/2503.04118v1)
[51](https://onlinelibrary.wiley.com/doi/10.1002/for.70023?af=R)
[52](https://www.sciencedirect.com/science/article/abs/pii/S0960148125024723)
[53](https://github.com/google-research/timesfm)
[54](https://liner.com/review/samformer-unlocking-potential-transformers-in-time-series-forecasting-with-sharpnessaware)
[55](https://openreview.net/forum?id=v7UqniC9pF)
[56](https://www.lgresearch.ai/blog/view?seq=428)
[57](https://arxiv.org/abs/2310.06625)
[58](https://arxiv.org/html/2510.07957v1)
[59](https://arxiv.org/html/2507.02907v1)
[60](https://arxiv.org/abs/2510.25502)
[61](https://arxiv.org/html/2508.16641v1)
[62](https://arxiv.org/html/2412.17853v1)
[63](https://arxiv.org/html/2509.17845v1)
[64](https://arxiv.org/html/2508.19609v1)
[65](https://arxiv.org/html/2410.08421v1)
[66](https://insoo-hwang.tistory.com/57)
[67](https://icml.cc/virtual/2024/poster/33288)
[68](https://liner.com/ko/review/foundation-models-for-time-series-analysis-tutorial-and-survey)
[69](https://www.jmir.org/2025/1/e74423)
[70](https://arxiv.org/html/2309.15946)
[71](https://arxiv.org/pdf/2104.05522.pdf)
[72](https://arxiv.org/html/2412.17323v3)
[73](https://arxiv.org/pdf/2501.19065.pdf)
[74](https://pmc.ncbi.nlm.nih.gov/articles/PMC9023224/)
[75](https://arxiv.org/abs/2503.11411)
[76](https://www.esann.org/sites/default/files/proceedings/2023/ES2023-171.pdf)
[77](https://research.aimultiple.com/time-series-foundation-models/)
[78](https://proceedings.neurips.cc/paper_files/paper/2024/file/a0b1082fc7823c4c68abcab4fa850e9c-Paper-Conference.pdf)
[79](https://arxiv.org/html/2501.08628v1)
[80](https://www.nature.com/articles/s41598-024-82417-4)
[81](https://www.sciencedirect.com/science/article/pii/S1389128625003627)
[82](https://github.com/yuqinie98/PatchTST)
[83](https://vanha-mathai.tistory.com/4)
[84](https://www.themoonlight.io/ko/review/empowering-time-series-analysis-with-synthetic-data-a-survey-and-outlook-in-the-era-of-foundation-models)
[85](https://secundo.tistory.com/113)
[86](https://aiflower.tistory.com/221)
[87](https://www.techrxiv.org/users/706235/articles/691677/master/file/data/Variational_NBEATS_model_with_hierarchical_timestamp_information_for_Long_Sequence_Time_Series_Forecasting/Variational_NBEATS_model_with_hierarchical_timestamp_information_for_Long_Sequence_Time_Series_Forecasting.pdf)
[88](https://arxiv.org/pdf/2502.13721.pdf)
[89](https://arxiv.org/pdf/2305.12095.pdf)
[90](https://arxiv.org/html/2510.07084v1)
[91](https://arxiv.org/abs/2509.26347)
[92](https://arxiv.org/html/2310.10688v4)
[93](https://arxiv.org/html/2408.04245v1)
[94](https://arxiv.org/abs/2510.00809)
[95](https://arxiv.org/html/2401.13912v1)
[96](https://arxiv.org/html/2506.20167v1)
[97](https://arxiv.org/html/2405.02358v3)
[98](https://arxiv.org/html/2502.14045v1)
[99](https://arxiv.org/html/2408.10483v1)
[100](https://arxiv.org/html/2503.11411v1)
[101](https://arxiv.org/html/2304.08424v5)
[102](https://arxiv.org/html/2401.00230v1)
