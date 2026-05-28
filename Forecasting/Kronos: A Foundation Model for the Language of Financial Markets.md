
# Kronos: A Foundation Model for the Language of Financial Markets

> **📌 논문 정보**
> - **제목**: Kronos: A Foundation Model for the Language of Financial Markets
> - **저자**: Yu Shi, Zongliang Fu, Shuo Chen, Bohan Zhao, Wei Xu, Changshui Zhang, Jian Li (Tsinghua University)
> - **arXiv**: [2508.02739](https://arxiv.org/abs/2508.02739) (2025.08.02)
> - **학회**: AAAI 2026 (Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 40, No. 30, pp. 25366–25373)

---

## 1. 핵심 주장 및 주요 기여 요약

대형 언어 모델(LLM)로 대표되는 대규모 사전학습 패러다임의 성공은 시계열 파운데이션 모델(TSFM)의 발전을 이끌었다. 그러나 금융 캔들스틱(K-라인) 데이터에 대한 적용은 여전히 제한적이며, 비사전학습 아키텍처보다 성능이 떨어지는 경우가 많다. 더욱이 기존 TSFM은 변동성 예측, 합성 데이터 생성 등 중요한 다운스트림 태스크를 간과하는 경향이 있다.

Kronos는 금융 시장의 "언어"—K-라인 시퀀스—에 특화되어 사전학습된 **디코더 전용(decoder-only) 파운데이션 모델 패밀리**이다.

### 🔑 주요 기여 (Main Contributions)

① **계층적 표현을 학습하는 새로운 금융 K-라인 데이터 모델링 프레임워크**를 제안한다. 여기에는 각 다변량 K-라인 레코드를 구조화된 이중 컴포넌트(coarse 및 fine) 토큰으로 양자화하는 특화 토크나이저가 포함되며, 이 서브토큰들을 순차적으로 예측하는 맞춤형 자기회귀 목표 함수와 결합된다. 이 **coarse-to-fine 예측 방식**은 Kronos가 다중 스케일 시장 역학을 명시적으로 모델링할 수 있게 한다.

② **대규모 이종(heterogeneous) 데이터로의 사전학습**: 45개 이상의 글로벌 거래소에서 수집한 120억 건 이상의 K-라인 레코드와 7가지 시간 단위(granularity)로 구성된 방대한 이종 코퍼스에서 자기회귀 사전학습을 수행한다.

③ **zero-shot 및 통합 멀티태스크 수행**: Kronos는 다양한 금융 태스크에서 zero-shot 설정으로 뛰어난 성능을 발휘한다.

④ **오픈소스 공개**: 사전학습된 모델이 공개적으로 제공된다.

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능, 한계

### 2-1. 해결하고자 하는 문제

금융 시장은 그 데이터의 풍부함, 고빈도 관측, 복잡한 비정상성 시간 역학으로 인해 TSFM의 중요하고 도전적인 응용 영역이다. 이 도메인의 핵심은 K-라인 시퀀스로, 캔들스틱 차트에서 파생된 다변량 시계열로서 고정 구간(OHLCVA) 동안의 시가(Open), 고가(High), 저가(Low), 종가(Close), 거래량(Volume), 거래대금(Amount)을 기록한다.

범용 TSFM을 금융 K-라인 데이터에 적용하는 데는 두 가지 주요 요인으로 인해 상당한 어려움이 있다. 첫째, K-라인 시퀀스는 낮은 신호 대 잡음비(low SNR), 강한 비정상성, OHLCVA 속성 간의 복잡하고 고차원적인 의존성 등 독특한 통계적 특성을 보인다.

또한, 금융 시퀀스는 기존 TSFM 사전학습 코퍼스에서 극히 일부를 차지하며, 대부분의 모델이 이 도메인 데이터를 전체의 1% 미만으로만 활용하고 있다.

---

### 2-2. 제안 방법 (Two-Stage Framework)

Kronos는 범용 TSFM과 달리 금융 데이터의 고유한 고잡음 특성을 처리하도록 설계되었다. 이를 위한 **두 단계 프레임워크(two-stage framework)**는: ① 특화된 토크나이저가 연속형 다차원 K-라인 데이터(OHLCV)를 계층적 이산(discrete) 토큰으로 양자화하고, ② 대형 자기회귀 Transformer가 이 토큰들로 사전학습되어 다양한 퀀트 태스크를 위한 통합 모델로 기능한다.

#### 📐 Stage 1: K-라인 토크나이저 (VQ-VAE 기반)

KronosTokenizer는 연속적인 다변량 금융 시계열(OHLCV 캔들스틱 데이터)을 **컴팩트한 이산 토큰 시퀀스로 변환하는 학습된 VQ-VAE 모델**이다. 이 토크나이제이션은 연속적인 가격 데이터와 자기회귀 Transformer 모델이 예측할 수 있는 이산 토큰 시퀀스 간의 간격을 메운다.

토크나이저는 **두 수준 계층적 토큰 시스템**을 사용한다: $s_1$ 토큰(coarse level)은 가격 움직임의 전체적인 구조를 포착하고, $s_2$ 토큰(fine level)은 coarse 토큰에 조건화된 잔차 세부 정보를 포착한다.

수식으로 표현하면, 각 K-라인 레코드 $\mathbf{x}_t \in \mathbb{R}^{d}$ (여기서 $d$는 OHLCVA 차원)에 대해:

$$
\mathbf{x}_t \xrightarrow{\text{Encoder}} \mathbf{z}_t \xrightarrow{\text{VQ}} (s_t^{(1)}, s_t^{(2)})
$$

- $s_t^{(1)} \in \mathcal{V}_1$: **coarse subtoken** — 광범위한 시장 움직임 포착
- $s_t^{(2)} \in \mathcal{V}_2$: **fine subtoken** — 세밀한 가격 변동 포착

각 토큰은 **coarse-grained 서브토큰**과 **fine-grained 서브토큰**으로 구성된다. 이 특성은 계층적 재구성 손실(hierarchical reconstruction loss)을 통해 강제되며, 이는 서브토큰이 서로 다른 정보 수준을 모델링하도록 명시적으로 강제함으로써 **coarse-to-fine 정보 계층**을 생성한다.

계층적 손실 함수는 아래와 같이 구성된다:

$$
\mathcal{L}_{\text{tokenizer}} = \mathcal{L}_{\text{recon}}^{(1)} + \lambda \cdot \mathcal{L}_{\text{recon}}^{(2)} + \mathcal{L}_{\text{VQ}}
$$

- $\mathcal{L}_{\text{recon}}^{(1)}$: coarse 토큰으로부터의 재구성 손실
- $\mathcal{L}_{\text{recon}}^{(2)}$: fine 토큰으로부터의 재구성 손실
- $\mathcal{L}_{\text{VQ}}$: 벡터 양자화(VQ) 커밋 손실 (codebook 학습)

#### 📐 Stage 2: 자기회귀 Transformer 사전학습

구동하는 아키텍처는 **GPT 스타일의 디코더 전용 네트워크**로 계층적 토큰에 특화된 Transformer 기반 자기회귀 모델이다. 훈련 중 모델은 과거 토큰화된 K-라인의 긴 시퀀스를 보고 언어 모델이 다음 단어를 예측하는 것처럼 각 단계에서 다음 토큰(coarse 후 fine)을 예측하는 방법을 학습한다.

자기회귀 목표 함수:

$$
\mathcal{L}_{\text{AR}} = -\sum_{t=1}^{T} \left[ \log P\!\left(s_t^{(1)} \mid s_{<t}, \mathbf{c}\right) + \log P\!\left(s_t^{(2)} \mid s_t^{(1)}, s_{<t}, \mathbf{c}\right) \right]
$$

여기서:
- $s_{ < t}$: 시점 $t$ 이전의 전체 토큰 시퀀스
- $\mathbf{c}$: 시장 메타정보(granularity, exchange ID 등) 컨텍스트

$s_1$ (Pre/Coarse): 첫 번째 $s_1 bits$는 coarse-grained 정보를 표현하고, $s_2$ (Post/Fine): 나머지 $s_2 bits$는 세밀한 세부 정보를 포착한다. 이 계층은 모델이 광범위한 시장 움직임과 특정 가격 변동을 분리할 수 있게 한다.

---

### 2-3. 모델 구조 (Model Architecture)

Kronos는 금융 시장의 "언어"—K-라인 시퀀스에 특화되어 사전학습된 디코더 전용 파운데이션 모델 패밀리이며, 범용 TSFM과 달리 금융 데이터의 고유한 고잡음 특성을 처리하도록 설계되었다.

다양한 계산 및 애플리케이션 요구에 맞게 다양한 용량의 사전학습 모델 패밀리를 공개하고 있다.

| 모델 | 파라미터 수 | 컨텍스트 길이 | 공개 여부 |
|------|-----------|------------|---------|
| Kronos-mini | ~소형 | 2,048 tokens | 공개 |
| Kronos-small | ~소형 | 512 tokens | 공개 |
| Kronos-base | ~기본 | 512 tokens | 공개 |
| Kronos-large | ~499.2M | - | 비공개 (2025.08 기준) |

> 499.2M 파라미터의 large 모델은 2025년 8월 기준 독점으로 유지되고 있으며, 저자들은 기관 파트너십이 확정되면 공개할 가능성을 시사했다. 현재 Kronos-base가 대부분의 전문 애플리케이션에 충분한 성능을 제공한다.

---

### 2-4. 성능 향상

가격 시계열 예측의 핵심 태스크에서 Kronos는 새로운 SOTA를 달성하여, 선도적인 TSFM 대비 RankIC를 **93%** 향상시키고, 최고 성능의 비사전학습 기준선 대비 **87%** 향상시켰다. 또한, 변동성 예측에서 **9% 낮은 MAE**를 달성하고, 합성 K-라인 생성의 생성 충실도에서 **22% 향상**을 이끌었다.

**성능 요약표:**

| 태스크 | 성능 향상 | 비교 대상 |
|--------|---------|---------|
| 가격 시계열 예측 (RankIC) | **+93%** | 선도적 TSFM 대비 |
| 가격 시계열 예측 (RankIC) | **+87%** | 최고 비사전학습 기준선 대비 |
| 변동성 예측 (MAE) | **-9%** | 기준선 대비 |
| 합성 K-라인 생성 (fidelity) | **+22%** | 기준선 대비 |

---

### 2-5. 한계

**① 컨텍스트 길이 제한**: Kronos-small/base의 경우 512 토큰으로 컨텍스트 길이가 제한된다. 이는 약 512개의 캔들 히스토리를 의미한다. 더 긴 룩백이 필요한 경우 Kronos-mini(2048 컨텍스트)를 사용하거나 슬라이딩 윈도우 방식을 구현해야 한다.

**② 블랙스완 예측 불가**: 어떤 시계열 모델과 마찬가지로 Kronos는 블랙스완이나 시장 레짐 전환을 예측할 수 없다. 다중 신호 중 하나로 활용해야 한다.

**③ 가격 동학만 입력**: 현재 OHLCVA 기반 예측에 집중하여, 멀티스텝 가격/수익 예측, 실현 변동성 예측, 합성 K-라인 생성, 거래 시뮬레이션이 주요 지원 태스크이다. 뉴스 감성, 거시경제 지표 등 외부 데이터 통합은 미래 과제로 남아 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. Zero-Shot 일반화의 근거

Kronos는 금융 시장에 파운데이션 모델을 적용하는 데 있어 중요한 이정표를 세웠다. 45개 거래소에서 수집한 120억 건의 K-라인 레코드에 대한 사전학습은 대규모 도메인 특화 사전학습이 시장 역학에 대한 **전이 가능한 지식(transferable knowledge)**을 추출할 수 있음을 보여준다.

### 3-2. 대규모 이종 데이터의 역할

Kronos는 특화된 토크나이저를 활용하여 연속형 다변량 K-라인 입력을 컴팩트한 토큰 시퀀스로 이산화하고, 45개 이상의 글로벌 시장과 7가지 시간 단위에서 수집한 120억 건 이상의 K-라인 레코드를 포함하는 방대하고 이종적인 코퍼스에서 자기회귀 사전학습을 수행한다.

이는 단일 시장 또는 자산 클래스에 국한되지 않는 **크로스-애셋(cross-asset) 표현 학습**을 가능하게 하며, 결과적으로 다음과 같은 일반화 이점을 제공한다:

- **시간적 일반화**: 7가지 시간 단위(granularity)로 단기·중기·장기 패턴 동시 학습
- **지리적 일반화**: 45개 이상 거래소의 글로벌 시장 데이터 학습
- **자산 클래스 간 일반화**: 암호화폐, 주식, 외환 등 다양한 자산 데이터 포함

### 3-3. Fine-Tuning을 통한 도메인 특화 일반화

파인튜닝 스크립트를 통해 특정 자산 클래스(암호화폐 vs. 주식 vs. 외환), 특정 시장 레짐(강세, 약세, 고변동성), 커스텀 예측 구간(스캘핑 vs. 스윙 트레이딩), 멀티모달 입력(가격 + 감성 + 펀더멘털)에 모델을 적응시킬 수 있으며, 파인튜닝된 모델은 도메인 특화 태스크에서 기본 사전학습 모델을 능가하는 것으로 나타났다.

### 3-4. 일반화를 저해하는 요인 (주의점)

최근 연구들은 TSFM이 진정한 "파운데이션" 특성을 보이는지 의문을 제기하고 있다. Zero-shot 능력은 사전학습 도메인에 크게 종속되어 있으며, 파인튜닝된 파운데이션 모델이 증가된 파라미터 수 대비 소형 전용 모델을 일관되게 능가하지는 못한다는 주장이 있다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

시계열 파운데이션 모델(TSFM)은 자연어 처리의 파운데이션 모델, 특히 LLM의 아키텍처와 학습 절차에서 영감을 받은 새로운 시계열 예측 모델 클래스이다. 대표적인 TSFM으로는 Chronos(Ansari et al., 2024), TimesFM(Das et al., 2024), Moirai/Moirai-MoE(Woo et al., 2024; Liu et al., 2024), MOMENT(Goswami et al., 2024), Time-MoE(Shi et al., 2024) 등이 있다.

| 모델 | 기관 | 아키텍처 | 금융 특화 | 주요 특징 |
|------|------|---------|---------|---------|
| **Kronos** (2025) | Tsinghua | Decoder-only Transformer | ✅ 특화 | K-라인 전용, VQ-VAE 토크나이저, 계층적 토큰 |
| **TimesFM** (2024) | Google | Decoder-only Transformer | ❌ 범용 | 100B 타임포인트 사전학습, 주파수 인식 인코딩 |
| **Moirai** (2024) | Salesforce | Encoder-based | ❌ 범용 | 다변량 지원, 패치 기반 입력 |
| **Chronos** (2024) | Amazon | T5 기반 | ❌ 범용 | 확률적 예측, 언어 모델 어휘 활용 |
| **Time-MoE** (2024) | - | MoE | ❌ 범용 | 50M 파라미터, 희소 활성화 |
| **Lag-LLaMA** (2024) | ServiceNow | LLaMA 기반 | ❌ 범용 | 단변량, 래그 특성 활용 |
| **MarketGPT** (2024) | - | GPT | ✅ 금융 관련 | 금융 시계열 모델링 |

기존 범용 TSFM들의 주요 한계는 사전학습 코퍼스 내 금융 데이터 비율의 극히 적은 부분으로, 대부분이 이 도메인에 1% 미만의 데이터만 할당하고 있다는 점이다.

TimesFM은 단변량 모델로, 많은 금융 예측 태스크에서 핵심적인 교차 특성 의존성을 모델링하지 못한다는 한계가 있다.

2024년 말부터 2025년 초까지 Amazon, Google, Salesforce 등 주요 기업들이 TimesFM(Google), Chronos(Amazon), Moirai(Salesforce), TimeGPT, Lag-LLaMA(ServiceNow), Timer-XL(THUML) 등 새로운 시계열 파운데이션 모델을 잇달아 출시하며 시계열 예측 분야에 중요한 발전을 이루었다.

---

## 5. 향후 연구에 미치는 영향 및 고려할 점

### 5-1. 향후 연구에 미치는 영향

파운데이션 모델 패러다임은 시계열 예측 접근 방식에 근본적인 변화를 나타낸다. 제한된 데이터에서 처음부터 학습하는 것이 아닌 **사전학습된 표현(representation)을 활용**하는 능력은 모델 개발의 경제학을 바꾼다. 이는 또한 어떤 문제들이 다룰 수 있는지를 변화시킬 수 있다.

도메인 특화 파운데이션 모델은 다음 프런티어이며, Kronos는 금융 분야에서 그 가능성을 입증하고 있다.

구체적 영향으로는:
1. **도메인 특화 사전학습의 중요성 확인**: 범용 모델보다 금융 특화 사전학습이 더 효과적임을 입증
2. **이산 토크나이제이션의 시계열 확장**: VQ-VAE 기반 계층적 토크나이제이션을 시계열에 성공적으로 적용
3. **통합 멀티태스크 프레임워크**: 예측, 변동성 추정, 생성을 단일 모델로 처리하는 선례 마련
4. **퀀트 파이낸스의 민주화**: 45개 이상 거래소에서 학습된 파운데이션 모델을 오픈소스로 제공함으로써, 개발자와 연구자에게 사내에서 수백만 달러를 투자해야 구축할 수 있는 시스템에 필적하는 도구를 제공한다.

### 5-2. 앞으로 연구 시 고려할 점

| 연구 방향 | 설명 |
|---------|------|
| 🔭 **컨텍스트 길이 확장** | 512 토큰의 한계를 넘어 더 긴 시장 역사를 활용하는 방법 연구 |
| 📰 **멀티모달 통합** | 가격 데이터와 뉴스 감성, 펀더멘털 데이터의 효과적 결합 |
| 🌍 **분포 이동(Distribution Shift) 대응** | 시장 레짐 변화, 블랙스완 이벤트에 대한 강건성 향상 |
| 📏 **스케일링 법칙(Scaling Laws) 연구** | 금융 특화 FM에서의 파라미터-성능 관계 탐구 |
| 🔍 **설명 가능성(XAI)** | 모델 예측의 해석 가능성 강화 (금융 규제 대응) |
| ⚡ **실시간 추론 최적화** | 고빈도 트레이딩 환경에서의 지연 시간 최소화 |

arXiv에 금융 특화 파운데이션 모델, LOB 예측 등 관련 주제의 새 논문들이 정기적으로 등장하고 있으며, 이는 틈새 분야의 호기심을 넘어 주류 연구 방향이 되고 있다. 퀀트 실무자들에게도 이 흐름에 주목할 것을 권한다.

---

## 📚 참고 자료 (References)

| # | 출처 | URL |
|---|------|-----|
| 1 | **[arXiv 원문]** Yu Shi et al., "Kronos: A Foundation Model for the Language of Financial Markets," arXiv:2508.02739, 2025 | https://arxiv.org/abs/2508.02739 |
| 2 | **[AAAI 2026 공식]** Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 40, No. 30, pp. 25366–25373 | https://ojs.aaai.org/index.php/AAAI/article/view/39730 |
| 3 | **[arXiv HTML 전문]** Kronos 논문 HTML 버전 | https://arxiv.org/html/2508.02739v1 |
| 4 | **[GitHub 공식 저장소]** shiyu-coder/Kronos | https://github.com/shiyu-coder/Kronos |
| 5 | **[Hugging Face]** NeoQuasar/Kronos-base | https://huggingface.co/NeoQuasar/Kronos-base |
| 6 | **[Hugging Face Tokenizer]** NeoQuasar/Kronos-Tokenizer-2k | https://huggingface.co/NeoQuasar/Kronos-Tokenizer-2k |
| 7 | **[NeurIPS 2025 Slides]** Kronos 발표 자료 | https://neurips.cc/media/neurips-2025/Slides/130441.pdf |
| 8 | **[IDEAS/RePEC]** Kronos 논문 등재 | https://ideas.repec.org/p/arx/papers/2508.02739.html |
| 9 | **[Arkin's Blog]** Kronos 실전 분석 | https://blog.arkin-dev.com/kronos-foundation-model-financial-markets-2026-04-10/ |
| 10 | **[Jonathan Kinlay Blog]** "Time Series Foundation Models for Financial Markets" | https://jonathankinlay.com/2026/02/time-series-foundation-models-for-financial-markets-kronos-and-the-rise-of-pre-trained-market-models/ |
| 11 | **[PyShine]** Kronos 상세 해설 | https://pyshine.com/Kronos-Foundation-Model-Financial-Markets/ |
| 12 | **[Leeroopedia]** KronosTokenizer 상세 | https://leeroopedia.com/index.php/Principle:Shiyu_coder_Kronos_Tokenizer_Loading |
| 13 | **[arXiv 2507.07296]** "Time Series Foundation Models for Multivariate Financial Time Series Forecasting" | https://arxiv.org/html/2507.07296v1 |
| 14 | **[arXiv 2510.13654]** "Time Series Foundation Models: Benchmarking Challenges and Requirements" | https://arxiv.org/html/2510.13654v1 |
| 15 | **[arXiv 2410.10469]** "Moirai-MoE: Empowering Time Series Foundation Models with Sparse Mixture of Experts" | https://arxiv.org/pdf/2410.10469 |
| 16 | **[ResearchGate]** Kronos 논문 등재 | https://www.researchgate.net/publication/402628844 |
| 17 | **[Medium]** "The First GPT for Financial Markets" | https://medium.com/@sanderink.ursina |
| 18 | **[Open Data Science]** "Foundation Models for Time Series" | https://opendatascience.com/foundation-models-for-times-series/ |

> ⚠️ **정확도 주의**: 본 답변은 공개된 arXiv 논문 전문, 공식 GitHub, AAAI 게재 정보 및 신뢰할 수 있는 블로그 분석을 기반으로 작성되었습니다. 계층적 손실 함수 및 자기회귀 목표 함수의 수식은 논문에서 기술된 개념을 바탕으로 표준 표기법으로 재구성한 것으로, 논문 내 정확한 수식 기호와 일부 차이가 있을 수 있습니다. 정확한 수식은 원문 PDF를 직접 확인하시기를 권장합니다.
