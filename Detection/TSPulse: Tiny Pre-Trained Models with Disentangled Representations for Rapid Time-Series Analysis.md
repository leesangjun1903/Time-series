
# TSPulse: Tiny Pre-Trained Models with Disentangled Representations for Rapid Time-Series Analysis

> **논문 정보**
> - **제목(v1):** TSPulse: Tiny Pre-Trained Models with Disentangled Representations for Rapid Time-Series Analysis
> - **제목(v2):** TSPulse: Dual Space Tiny Pre-Trained Models for Rapid Time-Series Analysis
> - **저자:** Vijay Ekambaram, Subodh Kumar, Arindam Jati, Sumanta Mukherjee, Tomoya Sakai, Pankaj Dayama, Wesley M. Gifford, Jayant Kalagnanam (IBM Research)
> - **arXiv:** [2505.13033](https://arxiv.org/abs/2505.13033) (2025년 5월 제출)
> - **발표:** ICLR 2026 (Conference Paper)
> - **코드/모델:** [HuggingFace ibm-granite/granite-timeseries-tspulse-r1](https://huggingface.co/ibm-granite/granite-timeseries-tspulse-r1)

---

## 1. 핵심 주장 및 주요 기여 요약

### 📌 핵심 주장

시계열 태스크는 시간 대 주파수와 같이 다중 표현 공간, 그리고 로컬 패턴 대 글로벌 의미론과 같이 다양한 추상화 수준에서의 신호로부터 이점을 얻는다. 그러나 기존의 사전 학습된 시계열 모델들은 이러한 이질적인 신호들을 단일한 대형 임베딩으로 얽어 놓아(entangle), 전이 가능성과 직접적인 제로샷 사용성을 제한한다.

이를 해결하기 위해 TSPulse는 **"작지만 강하게(tiny but powerful)"** 라는 철학으로, 다음을 주장합니다:

> **TL;DR:** 분리된(disentangled) 임베딩을 갖는 초경량 시계열 사전 학습 모델(1M 파라미터)로, 이상 탐지, 분류, 결측값 보완, 유사도 검색에서 SOTA 성능을 달성한다.

---

### 📌 5가지 주요 기여

TSPulse는 단 1M 파라미터로 분류, 이상 탐지, 결측값 보완, 검색 등 다양한 태스크에서 강력한 성능을 발휘하도록 특화된 초소형 시계열 사전 학습 모델이다. 아키텍처 수준에서는 시간 및 주파수 도메인 모두에서 학습하는 이중 공간 마스크 재구성(dual-space masked reconstruction) 전략을 채택하며, 세부 분석을 위한 상세 임베딩과 광범위한 태스크 이해를 위한 고수준 의미론적 임베딩을 모두 생성하는 이중 임베딩 분리 접근법으로 강화된다.

| 기여 | 설명 |
|------|------|
| ① 이중 공간 마스크 재구성 | 시간 + 주파수(FFT) 도메인 동시 학습 |
| ② 이중 임베딩 분리(Dual-Embedding Disentanglement) | 세부(Long) 임베딩 + 의미론적(Short) 임베딩 |
| ③ TSLens | 분류를 위한 태스크 인식 피처 추출 모듈 |
| ④ 멀티헤드 삼각측량(Multi-Head Triangulation) | 이상 탐지 강화 |
| ⑤ 하이브리드 마스킹(Hybrid Masking) | 사전 학습 편향 제거, 제로샷 결측값 보완 개선 |

---

## 2. 상세 분석

### 🔴 2-1. 해결하고자 하는 문제

TSPulse는 효율적인 시계열 파운데이션 모델 분야의 중요한 발전을 대표하며, 현재 접근법의 중대한 한계인 대규모 사전 학습 모델의 계산 부담을 해결한다. Chronos, Moment, UniTS와 같은 기존 시계열 파운데이션 모델(TSFM)들은 강력한 성능을 달성하지만, 일반적으로 수억에서 수십억 개의 파라미터를 요구하여 자원이 제한된 환경에서의 배포가 어렵다.

구체적으로, 기존 모델들은 두 가지 근본적인 문제를 가집니다:

1. **표현 공간 혼재 문제:** 기존 사전 학습된 시계열 모델들은 이질적인 신호들을 단일 대형 임베딩으로 얽어 놓아, 전이 가능성과 직접적인 제로샷 사용성을 제한한다.

2. **진단 태스크 지원 부족:** 시계열 사전 학습 모델들은 예측(forecasting) 분야에서 빠른 발전을 이루었지만, 시계열 진단 태스크에 대한 발전은 상대적으로 제한적이다. Moment, UniTS, VQShape, GPT4TS 등 일부 모델들이 진단 태스크의 하위 집합을 지원하지만, 이들의 성능은 여전히 상당한 개선 여지를 남기고 있다.

---

### 🟠 2-2. 제안하는 방법 (수식 포함)

#### (A) 이중 공간 마스크 재구성 (Dual-Space Masked Reconstruction)

TSPulse의 핵심은 이중 공간 마스크 재구성 전략으로, 마스킹된 입력이 시간 및 주파수(FFT) 도메인 모두에서 동시에 재구성된다. 이 공동 학습은 급격한 스파이크 같은 특정 패턴은 시간 도메인에서, 주기성 같은 패턴은 주파수 도메인에서 더 두드러진다는 직관을 활용한다. 양 공간에서 혼합, 어텐션, 재구성을 학습함으로써 TSPulse는 더 풍부하고 견고한 표현을 구축한다.

입력 $\mathbf{X} \in \mathbb{R}^{S \times C}$ (시퀀스 길이 $S$, 채널 수 $C$)에 대해, RevIN 정규화 후 패치 단위로 분할하고 마스킹합니다:

$$\mathbf{X}_{\text{masked}}^{t}, \mathbf{X}_{\text{masked}}^{f} = \text{Mask}(\mathbf{X}^{t}), \quad \text{Mask}(\mathbf{X}^{f})$$

여기서 $\mathbf{X}^{t}$는 시간 도메인, $\mathbf{X}^{f} = \text{FFT}(\mathbf{X}^{t})$는 주파수 도메인 표현입니다.

#### (B) 전체 사전 학습 손실 함수

핵심 혁신은 FFT 처리를 통해 시간 및 주파수 도메인에서 동시에 학습하는 것으로, 모델은 두 공간에서 마스킹된 입력 세그먼트를 재구성한다:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{time}} \mathcal{L}_{\text{time}} + \lambda_{\text{freq}} \mathcal{L}_{\text{freq}} + \lambda_{\text{future}} \mathcal{L}_{\text{future}} + \lambda_{\text{prob}} \mathcal{L}_{\text{prob}}$$

- $\mathcal{L}_{\text{time}}$: 시간 도메인에서의 마스크 재구성 손실
- $\mathcal{L}_{\text{freq}}$: 주파수 도메인에서의 마스크 재구성 손실
- $\mathcal{L}_{\text{future}}$: 단기 미래 예측 손실 (의미론적 임베딩 학습 유도)
- $\mathcal{L}_{\text{prob}}$: 전역 주파수 시그니처(확률적 스펙트럼) 재구성 손실
- $\lambda_*$: 태스크별 가중치 (태스크 특화 시 재조정됨)

TSPulse는 다중 헤드를 제공하므로, 대상 태스크와 가장 관련성이 높은 헤드를 우선시하도록 손실 목표를 재가중화하여 모든 태스크에 대한 사전 학습을 특화한다. 이를 통해 경량 설계를 유지하면서 태스크 특화 표현을 정제하고, 지정된 다운스트림 태스크에 대한 효율적인 전이 학습을 촉진한다.

#### (C) 이중 임베딩 분리 (Dual-Embedding Disentanglement)

TSPulse는 이중 임베딩 분리 개념을 도입하여, 사전 학습 중 두 가지 유형의 임베딩을 생성한다: (i) 세밀한 분석을 위한 상세 임베딩, (ii) 광범위한 태스크 이해를 위한 고수준 의미론적 임베딩. 이를 위해 TSPulse는 두 계층적 수준에서 마스크 재구성을 수행한다: 하나는 상세 구조를 캡처하기 위해 전체 원시 신호를 대상으로 하고, 다른 하나는 고수준 추상화를 인코딩하기 위해 컴팩트한 의미론적 시그니처를 재구성한다. 이 이중 임베딩 설계는 의미론적 고수준 신호(예: 전체 형태)가 필요한 것과 저수준 패턴 충실도(예: 주파수 왜곡)가 필요한 것 등 다양한 다운스트림 태스크에서 효과적이다. 두 수준의 정보를 사전 학습 중 캡처함으로써 TSPulse는 태스크 전반에 걸쳐 더 빠르고 강건한 일반화를 가능하게 한다.

임베딩 구조를 수식으로 표현하면:

$$\mathbf{E} = \underbrace{[\mathbf{e}_{1}^{t}, \ldots, \mathbf{e}_{N}^{t}, \mathbf{e}_{1}^{f}, \ldots, \mathbf{e}_{N}^{f}]}_{\text{Long (Detail) Embeddings: }2N\text{ patches}} \; \oplus \; \underbrace{[\mathbf{r}_{1}, \ldots, \mathbf{r}_{R}]}_{\text{Short (Semantic) Embeddings: }R\text{ registers}}$$

처음 $2N$개의 패치 임베딩은 시간 및 주파수 도메인에 해당하며 세밀한 시간적·스펙트럼 패턴을 캡처한다. 이들은 상세 재구성에 사용된다. 반면 마지막 $R$개의 레지스터 임베딩은 설계상 컴팩트하고 추상적이며, 전역 구조를 요약하고 단기 예측 및 전역 주파수 시그니처 예측과 같은 의미론적 재구성 태스크에 사용된다. 이 이중 임베딩 설계는 저수준 충실도와 고수준 의미론이 공동으로 모델링되도록 보장한다.

#### (D) 하이브리드 마스킹 (Hybrid Masking)

기존 접근법에 만연한 마스크 유발 편향을 개선하고 완화하기 위해, 사전 학습 중 결측 다양성을 향상시키는 간단하지만 효과적인 하이브리드 마스킹 전략을 제안한다.

$$\mathbf{M}_{\text{hybrid}} = \alpha \cdot \mathbf{M}_{\text{block}} + (1-\alpha) \cdot \mathbf{M}_{\text{random}}$$

블록 마스킹과 랜덤 마스킹을 결합하여 다양한 결측 패턴을 학습함으로써 제로샷 결측값 보완 성능을 높입니다.

#### (E) 아이덴티티 초기화 채널 믹서

기존 설계의 핵심 한계는 이러한 믹서들의 무작위 초기화인데, 이는 이미 사전 학습된 레이어 사이에 학습되지 않은 파라미터를 도입한다. 이로 인해 정보 흐름이 방해받고 급격한 활성화 변화가 발생하여, 특히 파인튜닝 초기 단계에서 불안정한 그래디언트 전파로 이어진다. 이를 해결하기 위해 채널 믹서를 아이덴티티 가중치로 초기화하여 사전 학습된 가중치 사이에서 원활한 그래디언트 흐름을 가능하게 한다. 이 레이어들은 이전 지식을 방해하지 않으면서 점진적으로 채널 간 의존성을 학습하여, 실험으로 확인된 바와 같이 훨씬 더 안정적인 파인튜닝 과정을 이끈다.

---

### 🟡 2-3. 모델 구조

TSPulse는 다른 주요 시계열 모델들이 트랜스포머 기반인 것과 달리, IBM의 이전 TSMixer 아키텍처를 백본으로 사용하여 다층 퍼셉트론 블록과 선택적 "게이티드" 어텐션 블록을 교대로 구성한다. 이 초효율적 하이브리드 아키텍처는 TSPulse를 특수 하드웨어 없이 노트북만큼 작은 장치에서도 쉽게 튜닝하고 서비스할 수 있게 한다.

#### 모델 구조 개요

```
Input X ∈ ℝ^{S×C}
     │
     ▼
 [RevIN 정규화]
     │
  ┌──┴──┐
  │     │
  ▼     ▼
시간 도메인  주파수 도메인 (FFT)
패치+마스킹   패치+마스킹
  │             │
  └──────┬──────┘
         ▼
   [TSMixer Backbone]
   (MLP-Mixer + Gated Attention)
         │
   ┌─────┴─────┐
   ▼           ▼
Long Embedding  Short Embedding
(2N patches)    (R register tokens)
시간+주파수 패치   의미론적 요약
         │           │
   [Decoder]   [Semantic Decoder]
         │           │
  시간/주파수    단기예측 + 전역
  재구성 손실    주파수 시그니처
```

#### 태스크별 다운스트림 모듈

TSPulse는 사전 학습 중 두 가지 유형의 임베딩을 생성한다: 상세(Long) 임베딩은 완전한 신호 재구성에 최적화되어 세밀한 시간적 구조를 캡처하며; 의미론적(Short) 임베딩은 학습 가능한 레지스터 토큰에서 파생된 컴팩트 표현으로, 시간 이동, 크기 변화, 노이즈에 강건하게 설계된다. TSPulse의 태스크별 컴포넌트로는 강건한 사전 학습을 위한 하이브리드 마스킹, 의미론적 검색 기능, 분류를 위한 TSLens, 이상 탐지를 위한 멀티헤드 삼각측량이 있다.

| 태스크 | 사용 임베딩 | 특화 모듈 |
|--------|-------------|-----------|
| 결측값 보완 (Imputation) | Long (Detail) | 하이브리드 마스킹 |
| 의미론적 검색 (Retrieval) | Short (Semantic) | Cosine Similarity |
| 분류 (Classification) | Long + Short | TSLens |
| 이상 탐지 (Anomaly Detection) | Long | Multi-Head Triangulation |

#### TSLens 모듈

시계열 분류를 위해 TSPulse는 표준 풀링 연산을 대체하는 신경망 컴포넌트인 TSLens를 도입한다. TSLens는 피처 어텐션 기반으로 동작하며, TSLens를 단순 풀링으로 교체하면 11~16% 성능 저하가 발생하여, 피처 어텐션의 가치를 강조한다.

#### 멀티헤드 삼각측량 (Multi-Head Triangulation)

태스크 수준에서 TSPulse는 태스크별 피처 어텐션을 가능하게 하는 파인튜닝 컴포넌트인 TSLens를 통합한다. 또한 여러 예측 헤드의 편차를 상관시켜 상호 보완적인 모델 출력을 융합함으로써 이상 탐지를 강화하는 멀티헤드 삼각측량 기법을 도입한다.

이상 스코어 산출 예시:

$$\text{AnomalyScore}(t) = \sum_{k \in \{t, f, \text{future}\}} w_k \cdot \left\| \hat{\mathbf{x}}_k(t) - \mathbf{x}(t) \right\|^2$$

#### 사전 학습 배포 모델 변종

세 가지 사전 학습된 변종이 특정 시계열 태스크에 최적화되어 공개된다: `tspulse-hybrid-allhead-512-p8-r1`(이상 탐지 권장), `tspulse-hybrid-dualhead-512-p8-r1`(결측값 보완 및 검색 권장), `tspulse-block-dualhead-512-p16-r1`(분류 권장).

---

### 🟢 2-4. 성능 향상

소형 크기에도 불구하고 TSPulse는 4가지 시계열 진단 태스크에서 일관되게 강력한 성능 향상을 달성한다: TSB-AD 이상 탐지 리더보드에서 +20%, 유사도 검색에서 +25%, 결측값 보완에서 +50%, 다변량 분류에서 +5~16% 향상, 75개 이상의 데이터셋에서 10~100배 큰 모델을 능가한다.

| 태스크 | 벤치마크 | 성능 향상 |
|--------|----------|-----------|
| 이상 탐지 | TSB-AD 리더보드 (단변량+다변량 Rank 1) | **+20%** |
| 유사도 검색 | TS Retrieval Benchmark | **+25%** |
| 결측값 보완 | Zero-Shot Imputation | **+50%** |
| 다변량 분류 | UEA Classification | **+5~16%** |

이상 탐지를 위한 선도적 학술 벤치마크 TSB-AD에서 TSPulse는 두 범주 모두에서 최첨단 성능을 보여, 강력한 통계 모델을 24% 능가하고 다양한 더 큰 파운데이션 모델을 최소 33% 이상 능가했다.

#### 어블레이션 결과

이중 임베딩 설계에서 단기 또는 장기 임베딩 중 하나를 제거하면 평균 정확도가 8~10% 감소하여, 의미론적 및 세밀한 피처 모두를 캡처하는 것의 중요성을 확인한다. 파인튜닝 중 마스킹을 비활성화하면 특히 소규모 데이터셋에서 8% 하락이 발생하며, 이는 마스킹이 정규화 역할을 한다는 것을 강조한다. TSLens를 단순 풀링으로 교체하면 11~16% 하락이 발생하며, 이중 공간 학습을 제거(시간 도메인에서만 재구성)하면 정확도가 7% 낮아진다.

---

### 🔴 2-5. 한계

부록 A.14는 추가 다운스트림 태스크로의 확장, 점진적 학습 활성화, 지도 요구사항 감소 기회를 포함한 한계와 미래 방향을 개략적으로 설명한다.

주요 한계를 정리하면:

1. **예측(Forecasting) 태스크 미지원:** TTM은 최소 1~5M 파라미터로 경쟁력 있는 제로샷/퓨샷 예측을 제공하지만, TSPulse는 주로 진단 태스크에 특화되어 있으며, 이상 탐지, 분류, 결측값 보완 등의 광범위한 다운스트림 태스크로 경량 모델링 패러다임을 확장하는 것이 과제로 남아 있다.
2. **크로스 채널 모델링의 사전 학습 제한:** 이질적인 채널 수 문제로 인해 TSPulse는 유니변량 모드로 사전 학습되어 각 채널을 독립적으로 처리한다. 채널 간 모델링은 파인튜닝으로 미루어지며, 목표 데이터셋에 따라 채널 믹싱이 선택적으로 활성화된다.
3. **증분 학습 부재:** 새로운 도메인 데이터에 대한 지속적/점진적 학습(Continual/Incremental Learning)이 아직 지원되지 않음.
4. **지도 요구사항:** 일부 태스크(예: 분류)에서 레이블 데이터 의존성 존재.

---

## 3. 모델의 일반화 성능 향상 가능성 (중점 분석)

### 3-1. 제로샷 일반화의 핵심 메커니즘

TSPulse의 일반화 성능은 아래 설계 요소들에 의해 구동됩니다:

#### ① 의미론적 임베딩의 불변성

주파수 민감성: 임베딩 공간에서 다른 주파수 신호의 명확한 분리; 노이즈 강건성: 다양한 노이즈 수준에서 의미론적 일관성 유지; 시간 이동 불변성: 시간적 이동에도 일관된 표현; 크기 강건성: 다른 신호 진폭에서도 안정적인 임베딩.

이 이중 임베딩 분리 메커니즘은 세밀한 분석을 위한 세부 임베딩과 광범위한 이해를 위한 의미론적 임베딩을 모두 생성한다. 이 의미론적 임베딩은 시간, 크기, 노이즈의 변화에 본질적으로 강건하여 시계열 검색에 이상적이다.

#### ② 이중 공간 학습에 의한 일반화

이 전역 시그니처는 보조 재구성 대상으로 작동하여, 모델이 고수준 의미론적 패턴을 캡처하고 다운스트림 태스크에 대한 일반화를 개선하는 데 도움을 준다. 로그 변환은 동적 범위를 줄이고 학습을 안정화하며, 소프트맥스는 지배적인 스펙트럼 성분을 강조하여 출력을 확률 유사 분포로 매핑한다.

#### ③ 하이브리드 마스킹에 의한 일반화

하이브리드 마스킹은 사전 학습 중 편향을 줄여 제로샷 결측값 보완을 개선한다. 이는 단순 블록 마스킹만 사용할 경우 모델이 특정 결측 패턴에만 최적화되는 문제를 해결하며, 더 다양한 결측 시나리오에 대한 일반화를 가능하게 합니다.

#### ④ 유니변량 사전 학습으로 인한 도메인 불가지론성

1M 파라미터 제약은 여러 효율성 혁신을 이끈다: 유니변량 사전 학습으로 이질적인 채널 수를 효율적으로 처리; 아이덴티티 초기화된 채널 믹서로 다변량 입력의 파인튜닝을 안정화.

#### ⑤ 분리된 표현의 전이 가능성

광범위한 유용성, 특히 제로샷 전이에서 표현은 시간적, 스펙트럼, 의미론적 신호가 독립적으로 접근될 수 있도록 분리된 형태로 이러한 통찰을 명시적으로 노출해야 한다.

### 3-2. 일반화 성능의 실증적 증거

TSPulse는 분류, 이상 탐지, 결측값 보완, 검색 태스크에서 SOTA 성능을 달성하는 초소형(100만 파라미터) 사전 학습 모델을 도입한다. 이 접근법은 강건한 일반화를 보여주며 더 큰 기존 시계열 파운데이션 모델에 비해 계산 요구사항을 크게 줄인다.

두 수준의 정보를 사전 학습 중 캡처함으로써 TSPulse는 태스크 전반에 걸쳐 더 빠르고 강건한 일반화를 가능하게 한다. 1M 미만의 파라미터로 TSPulse는 10~100배 더 큰 모델을 능가하며, UEA 분류에서 5~16%, TSB-AD 벤치마크/리더보드에서 +20%(단변량 및 다변량 이상 탐지 모두 1위), 제로샷 결측값 보완에서 +50%, TS 유사도 검색에서 +25%를 달성한다.

### 3-3. 일반화 성능 향상의 미래 가능성

1. **더 많은 도메인 데이터로 사전 학습 확장:** 대조적 및 지도 표현 학습 방법은 일반적으로 소형이고 효과적이지만, 종종 각 데이터셋에서 별도로 학습되어 도메인 간 전이 가능성이 부족하다. 반면 TSFM은 대규모 다양한 코퍼스로 사전 학습하여 강력한 일반화를 제공하지만, 상당히 높은 모델 복잡성과 계산 비용을 치른다. TSPulse는 이 균형점을 개선할 여지가 있습니다.

2. **증분 학습:** 새로운 도메인에 대해 전체 재학습 없이 지식을 업데이트하는 메커니즘.

3. **포어캐스팅으로의 확장:** Tiny Time Mixers(TTM)는 소형 모델(1~5M 파라미터)이 시계열 예측에 효과적일 수 있음을 보여준다. TSPulse는 예측 특화 모델에서 분류, 결측값 보완, 이상 탐지와 같은 훨씬 광범위한 태스크로 초소형 사전 학습 모델의 범위를 명시적으로 확장한다.

---

## 4. 최신 관련 연구 비교 분석 (2020년 이후)

Moment(Goswami et al., 2024), UniTS(Gao et al., 2024), VQShape(Wen et al., 2024), GPT4TS(Zhou et al., 2023) 등 일부 사전 학습 모델들이 진단 태스크의 하위 집합을 지원한다.

| 모델 | 연도 | 파라미터 수 | 주요 태스크 | 아키텍처 | 특징 |
|------|------|-------------|-------------|----------|------|
| **GPT4TS** | 2023 | ~수억 | 다목적 | GPT-2 기반 | GPT-2를 백본으로 활용하여 다양한 시계열 분석 태스크를 위한 통합 모델 |
| **Chronos** | 2024 | 46M~710M | 예측 특화 | T5 기반 | 다양한 도메인에서 강력한 제로샷 예측 능력을 보여주는 트랜스포머 기반 아키텍처 |
| **MOMENT** | 2024 | 수억 | 다목적 | Transformer | TimesNet 및 GPT4TS와 같은 성공적인 모델의 연구를 기반으로 한 일반주의자 시계열 파운데이션 모델 |
| **UniTS** | 2024 | 수억 | 다목적 | Transformer | 통합 다중 태스크 시계열 모델 |
| **TTM** | 2024 | 1~5M | **예측 특화** | TSMixer | TSMixer 아키텍처를 예측 목표로 학습; 1~5M 파라미터로 경쟁력 있는 제로샷/퓨샷 예측 제공 |
| **🔴 TSPulse** | 2025 | **1M** | **진단 특화** | TSMixer+FFT | SOTA 제로샷 성능, 효율적인 파인튜닝, GPU 없이 배포 가능 |

### 비교 분석 핵심 포인트

1. **크기 효율성:** 이러한 아키텍처 및 태스크 혁신들은 TSPulse의 성능 향상에 집합적으로 기여한다: UEA 분류 벤치마크에서 5~16%, TSB-AD 이상 탐지 리더보드에서 +20%, 제로샷 결측값 보완에서 +50%, 시계열 검색에서 +25%. 이 결과들은 단 1M 파라미터(기존 SOTA 모델보다 10~100배 작음)로 달성되며 GPU 없는 추론을 허용한다.

2. **아키텍처 패러다임의 전환:** 다른 주요 시계열 모델들이 트랜스포머를 기반으로 하는 반면, TSPulse는 IBM의 이전 TSMixer 아키텍처를 백본으로 사용하여 다층 퍼셉트론 블록과 선택적 "게이티드" 어텐션 블록을 교대로 구성한다.

3. **진단 vs 예측 특화:** 시계열 사전 학습 모델들은 예측 분야에서 빠른 발전을 이루었지만, 시계열 진단 태스크에 대한 발전은 상대적으로 제한적이다. TSPulse는 이 공백을 메우는 최초의 초소형 진단 특화 모델입니다.

---

## 5. 향후 연구에 미치는 영향 및 고려할 점

### 🔵 5-1. 향후 연구에 미치는 영향

#### ① 경량화 패러다임의 확장

TSPulse는 분류, 결측값 보완, 이상 탐지, 유사도 검색 등 모든 분야에서 1M 파라미터 미만으로 SOTA 성능을 달성하는 초소형 시계열 사전 학습 모델의 새 기준을 세운다. 이중 공간 재구성, 분리된 임베딩, TSLens, 하이브리드 마스킹, 멀티헤드 삼각측량과 같은 혁신들을 통해 TSPulse는 강건한 제로샷 및 파인튜닝 성능을 가능하게 한다. 소형임에도 불구하고 10~100배 큰 모델을 능가하며 CPU에서도 효율적으로 실행되어 강력하고 배포 가능한 모델이다.

#### ② 표현 분리(Disentanglement) 연구의 활성화

TSPulse는 단순히 "크게 만들면 잘된다"는 통념을 깨고, **표현의 구조적 분리**가 성능 향상에 핵심임을 보여줍니다. 이는 향후 시계열 분야에서 표현 학습의 방향을 다음으로 이끌 것입니다:
- 태스크별 표현 분리 연구
- 멀티모달 시계열 표현 학습
- 도메인 불변 표현 학습

#### ③ 엣지 AI 및 실용적 배포

그 효율성은 GPU 없는 추론과 빠른 사전 학습을 가능하게 하여, 효율적인 시계열 사전 학습 모델의 새 기준을 세운다. 이로 인해 IoT 기기, 엣지 서버, 의료기기 등 자원 제한 환경에서의 시계열 분석 연구가 활발해질 것입니다.

#### ④ 진단 태스크 통합 벤치마킹

TSPulse가 75개 이상의 데이터셋에서 평가된 것은 향후 진단 특화 시계열 파운데이션 모델 비교를 위한 표준 벤치마크 구성에 영향을 줄 것입니다.

이 연구가 경량 시계열 모델링 분야에서 더 발전된 연구와 혁신에 영감을 줄 것으로 기대된다.

---

### 🔵 5-2. 향후 연구 시 고려할 점

#### ① 예측(Forecasting)과 진단의 통합

현재 TSPulse는 진단 태스크에, TTM은 예측에 특화되어 있습니다. 이 둘을 하나의 초소형 모델로 통합하는 것이 중요한 연구 방향입니다. 이를 위해 다음 손실 함수 확장을 고려할 수 있습니다:

$$\mathcal{L}_{\text{unified}} = \lambda_{\text{time}} \mathcal{L}_{\text{time}} + \lambda_{\text{freq}} \mathcal{L}_{\text{freq}} + \lambda_{\text{forecast}} \mathcal{L}_{\text{forecast}} + \lambda_{\text{future}} \mathcal{L}_{\text{future}} + \lambda_{\text{prob}} \mathcal{L}_{\text{prob}}$$

#### ② 증분/지속 학습 (Incremental/Continual Learning)

부록 A.14는 추가 다운스트림 태스크로의 확장, 증분 학습 활성화, 지도 요구사항 감소 기회를 포함한 한계와 미래 방향을 개략적으로 설명한다. 특히 새로운 도메인 데이터가 지속적으로 추가되는 실제 환경에서의 Catastrophic Forgetting 방지 연구가 필요합니다.

#### ③ 채널 간 의존성의 사전 학습 단계 통합

현재 채널 믹싱이 파인튜닝에서만 활성화되는 구조를 개선하여, 사전 학습 단계에서도 효과적인 채널 간 패턴을 학습할 수 있는 방법을 연구해야 합니다. 예를 들어:

$$\mathbf{E}_{\text{cross-channel}} = \text{CrossChannelMixer}(\mathbf{E}^{(1)}, \ldots, \mathbf{E}^{(C)})$$

#### ④ 자기지도 학습 목표의 다양화

현재 마스크 재구성 중심의 사전 학습을 넘어, 대조 학습(Contrastive Learning)과의 결합을 통해 더 강력한 표현 학습이 가능합니다:

$$\mathcal{L}_{\text{contrastive}} = -\log \frac{\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_j)/\tau)}{\sum_{k=1}^{N} \mathbf{1}_{[k \neq i]} \exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_k)/\tau)}$$

#### ⑤ 불확실성 정량화 (Uncertainty Quantification)

이상 탐지 및 분류에서 예측 불확실성을 함께 출력하는 확률론적 TSPulse 버전 연구가 필요합니다.

#### ⑥ 멀티모달 시계열 (Multi-modal Time Series)

텍스트, 이미지, 수치 데이터를 결합한 멀티모달 시계열 분석으로의 확장은 산업 현장의 실제 요구사항과 부합합니다.

#### ⑦ 설명 가능성 (Explainability)

TSPulse의 특징 중 하나는 더 통계적인 알고리즘과 달리 비정상(non-stationary) 시계열 데이터에서 잘 작동한다는 것이다. 이는 정상성 확인, 시계열 분해, 차분 수행이 필요 없음을 의미한다. 그러나 이러한 블랙박스 특성은 의료, 금융 등 고위험 분야에서의 설명 가능성 연구 필요성을 높입니다.

---

## 📚 참고 자료 (출처)

| # | 출처 | URL |
|---|------|-----|
| 1 | **arXiv 논문 (v1)** — TSPulse: Tiny Pre-Trained Models with Disentangled Representations for Rapid Time-Series Analysis | https://arxiv.org/abs/2505.13033 |
| 2 | **arXiv 논문 (v2)** — TSPulse: Dual Space Tiny Pre-Trained Models for Rapid Time-Series Analysis | https://arxiv.org/abs/2505.13033v2 |
| 3 | **arXiv HTML 전문 (v1)** | https://arxiv.org/html/2505.13033v1 |
| 4 | **OpenReview (ICLR 2026)** — TSPulse: Tiny Pre-Trained Models with Disentangled Representations | https://openreview.net/forum?id=Kw2mvnzCoc |
| 5 | **OpenReview PDF** | https://openreview.net/pdf/f0f91bd47c6d0fb82750a5c821738f9988bdcc2d.pdf |
| 6 | **IBM Research Blog** — An AI model with a finger on the time series pulse | https://research.ibm.com/blog/tspulse-time-series-ai-model |
| 7 | **HuggingFace Model Card** — ibm-granite/granite-timeseries-tspulse-r1 | https://huggingface.co/ibm-granite/granite-timeseries-tspulse-r1 |
| 8 | **IBM Granite Docs** — Granite Time Series | https://www.ibm.com/granite/docs/models/time-series |
| 9 | **alphaXiv Overview** — TSPulse: Dual Space Tiny Pre-Trained Models | https://www.alphaxiv.org/overview/2505.13033v2 |
| 10 | **ResearchGate** — TSPulse: Dual Space Tiny Pre-Trained Models (PDF) | https://www.researchgate.net/publication/391877720 |
| 11 | **ADS Abstract** — TSPulse: Dual Space Tiny Pre-Trained Models | https://ui.adsabs.harvard.edu/abs/2025arXiv250513033E/abstract |
| 12 | **IBM Tutorial** — Time Series Analysis with Granite Time (TSPulse) | https://www.ibm.com/think/tutorials/time-series-analysis-granite-time-tspulse |
| 13 | **(관련 연구) TSMixer** — Ekambaram et al., KDD 2023 | https://arxiv.org/abs/2306.09364 |
| 14 | **(관련 연구) MOMENT** — Goswami et al., ICML 2024 | arXiv:2402.03885 |
| 15 | **(관련 연구) Chronos** — Ansari et al., TMLR 2024 | https://arxiv.org/abs/2403.07815 |
| 16 | **(관련 연구) UniTS** — Gao et al., NeurIPS 2024 | arXiv:2403.00131 |

> ⚠️ **정확도 주의사항:** 본 답변의 수식 중 전체 손실 함수 ($\mathcal{L}_{\text{total}}$)는 alphaXiv의 논문 요약에서 확인된 수식이며, 나머지 수식 (마스킹, 임베딩 구조, 이상 스코어 등)은 논문에서 기술된 내용을 바탕으로 수식 표현으로 변환한 것으로, 논문 원본의 정확한 표기와 일부 다를 수 있습니다. 정확한 수식은 [arXiv 원문](https://arxiv.org/abs/2505.13033) 또는 [OpenReview PDF](https://openreview.net/forum?id=Kw2mvnzCoc) 를 직접 확인하시기 바랍니다.
