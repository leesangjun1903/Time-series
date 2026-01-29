
# LLM-ABBA: Understanding time series via symbolic approximation
## 요약
**LLM-ABBA**는 Adaptive Brownian Bridge-based Symbolic Aggregation (ABBA) 방법을 대규모 언어모델(LLM)과 통합하여 시계열 데이터를 기호 표현으로 변환하는 획기적인 프레임워크이다. 이 접근법은 기존의 패칭 기반 토큰화나 커스텀 토크나이저의 한계를 극복하면서, 시계열의 내재된 의미정보를 LLM의 원래 토큰 공간에서 효과적으로 활용할 수 있다는 점이 핵심이다. 특히 Fixed-Polygonal Chain (FAPCA) 트릭을 도입하여 기호 기반 예측 시 누적 오류를 크게 완화하며, 시계열 회귀(TSER) 벤치마크에서 새로운 SOTA(State-of-The-Art) 성능을 달성한다.

***

## 1. 해결하고자 하는 문제
### 1.1 LLM을 시계열에 적용하는 기존 방법의 한계
LLM 기반 시계열 분석을 위한 기존의 세 가지 주요 접근법은 각각 근본적인 문제를 가지고 있다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ec33e53b-d35c-49f4-8912-7a38fc4a79bc/2411.18506v4.pdf)

**패칭 및 토큰화 방법:**
- 시계열 세그먼트를 토큰화한 후 숫자값을 한 자리씩 생성하므로 생성 속도가 급격히 감소
- LLM의 내재 임베딩 공간과 시계열 특징 공간 간 의미론적 손실 발생

**커스텀 토크나이저 추가:**
- LLM 토크나이저는 수치값을 위해 설계되지 않았으므로 연속값과 시간적 관계를 무시
- 토큰에서 유연한 연속값으로의 변환 필수로 인해 의미 손실 위험 증가

**파운데이션 모델 구축:**
- 매우 높은 개발 비용과 장기간의 학습 필요
- 적용 가능성의 균형을 맞추기 어려움

### 1.2 핵심 난제
시계열을 LLM이 이해할 수 있는 콘텐츠로 변환하면서 동시에 기존 토큰 의미를 보존하고, 생성된 콘텐츠를 다시 시계열 도메인으로 역변환해야 한다는 이중 과제가 존재한다. 특히 예측 작업에서 **누적 오류** 문제가 심각하다.

***

## 2. 제안하는 방법: LLM-ABBA 프레임워크
### 2.1 ABBA 기호 근사의 핵심 원리
ABBA는 두 단계 프로세스로 시계열 $T = [t_1, t_2, \ldots, t_n] \in \mathbb{R}^n$을 기호 표현 $A = a_1a_2 \cdots a_N$ (단, $N \ll n$)으로 변환한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ec33e53b-d35c-49f4-8912-7a38fc4a79bc/2411.18506v4.pdf)

#### 단계 1: 압축 (Compression)

적응형 다각형 체인 근사(APCA)를 통해 인덱스들 $i_0 = 0 < i_1 < \cdots < i_N = n$을 선택한다:

$$\sum_{i=i_{j-1}}^{i_j} \left| t_{i_{j-1}} + (t_{i_j} - t_{i_{j-1}}) \cdot \frac{i - i_{j-1}}{i_j - i_{j-1}} - t_i \right|^2 \leq (i_j - i_{j-1} - 1) \cdot \text{tol}^2 \quad \text{(식 1)}$$

이는 각 세그먼트 $p_j = (\text{len}_j, \text{inc}_j)$가 길이와 증분을 나타낼 때, 원본과의 제곱 유클리드 거리가 tolerance 범위 내에 있음을 보장한다.

#### 단계 2: 디지털화 (Digitization)

길이와 증분을 정규화한 후:

$$p_i^s = \left( \text{scl} \cdot \frac{\text{len}_i}{\sigma_{\text{len}}}, \frac{\text{inc}_i}{\sigma_{\text{inc}}} \right) \quad i = 1, \ldots, N$$

그리디 정렬 기반 집계(Algorithm 1)를 통해 k-평균 클러스터링을 효율적으로 수행한다:

$$\text{SSE} = \sum_{i=1}^{k} \sum_{p^s \in S_i} \|p^s - c_i\|_2^2$$

각 클러스터 중심 $c_i$는 고유한 기호와 연결되며, 실수값 클러스터 중심을 가짐으로써 기호의 자연스러운 word embedding을 제공한다.

#### 단계 3: 역 기호화 (Inverse Symbolization)

예측 작업을 위해 기호를 원래 값으로 재구성한다. 핵심은 **Fixed-Point APCA (FAPCA)**의 도입이다.

### 2.2 오류 분석 및 이론적 보장
**정리 III.1**: 재구성 인덱스 및 값은 다음으로 주어진다:

$$(\tilde{i}_j, \tilde{t}_{i_j}) = \left( \sum_{\ell=1}^{j} \tilde{\text{len}}_\ell, t_0 + \sum_{\ell=1}^{j} \tilde{\text{inc}}_\ell \right) \quad j = 0, \ldots, N \quad \text{(식 3)}$$

여기서 $(\tilde{\text{len}}\_\ell, \tilde{\text{inc}}_\ell)$는 계산된 클러스터 중심이고, 정리 III.2에 의해 시작점과 끝점의 누적 편차가 상쇄된다.

**정리 III.3**: 하이퍼파라미터 $\alpha$에 대해:

$$\max_\ell \{ (d^{\text{inc}}_\ell)^2 + (d^{\text{len}}_\ell)^2 \} \leq \alpha^2 \quad \text{(식 5)}$$

이를 통해 $\alpha$로 오류를 직접 제어할 수 있다.

**정리 III.5** (확률적 경계): Hoeffding 부등식을 이용하면:

$$P(|e^{\text{inc}}_j| \geq h) \leq \exp\left( -\frac{h^2}{2j\alpha^2} \right) \quad \text{and} \quad P(|e^{\text{len}}_j| \geq h) \leq \exp\left( -\frac{h^2}{2j\alpha^2} \right)$$

이는 $\alpha$ 감소 시 재구성 오류가 감소함을 보증한다.

### 2.3 Fixed-Polygonal Chain Trick (FAPCA)
기호 기반 예측의 누적 오류 문제를 해결하기 위해, APCA의 변형으로 $p_j = (\text{len}\_j, t_{i_j})$를 사용한다. 즉, 증분 대신 실제 시계열 값을 직접 사용하여 이전 기호의 오류가 다음 재구성에 미치는 영향을 완화한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ec33e53b-d35c-49f4-8912-7a38fc4a79bc/2411.18506v4.pdf)

**그림 4**는 FAPCA가 기호 오류(perturbed symbols)에서 복원 시 드리프트를 현저히 감소시킴을 시각적으로 보여준다.

### 2.4 다중 시계열 기호화 및 일관성 유지
실무 적용을 위해 여러 시계열을 일관된 기호로 표현하는 절차: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ec33e53b-d35c-49f4-8912-7a38fc4a79bc/2411.18506v4.pdf)

**단계 1**: 각 시계열 $T_i$를 APCA/FAPCA로 압축 → $P_i$

**단계 2**: 정규화된 $P^s_i$들을 연결 → $P^s = [P^s_i]^q_{i=1}$

**단계 3**: 연결된 데이터에 대해 디지털화 수행

**단계 4**: 각 시계열에 기호 할당 (개수 = $|P^s_i|$)

Out-of-sample 데이터의 기호화는 학습 시 얻은 클러스터 중심을 기준으로 가장 가까운 중심에 할당한다.

### 2.5 LLM 피딩 및 프레임워크 통합
$$\begin{align}
(i) \quad & A = \phi(T) \\
(ii) \quad & M_{\text{inp}} = \text{Tokenizer}(\text{Prompt}, A) \\
(iii) \quad & M_{\text{outp}} = f^{\Delta}_{\text{LLM}}(M_{\text{inp}}) \\
(iv) \quad & \hat{Y} = \begin{cases}
\hat{Y} = M_{\text{outp}}, & \text{Classification} \\
\hat{Y} = \phi^{-1}(M_{\text{outp}}), & \text{Regression/Prediction}
\end{cases}
\end{align}$$

분류 작업은 생성된 레이블을 직접 사용하고, 회귀 및 예측 작업은 역 기호화를 통해 수치값을 복원한다.

***

## 3. 모델 구조
### 3.1 전체 아키텍처 (그림 3) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ec33e53b-d35c-49f4-8912-7a38fc4a79bc/2411.18506v4.pdf)
```
시계열 입력 T
    ↓
[압축/기호화] (ABBA ①, FAPCA ①)
    ↓
기호 표현 A
    ↓
[토큰화] LLM 토크나이저 ②
    ↓
[명령어 포함] 작업 지정 프롬프트 ②
    ↓
[LLM 처리] QLoRA 파인튜닝 ③
    ↓
[출력 생성] 분류/예측 ④⑤
    ↓
[역 기호화] ABBA 복원 ⑥⑤
    ↓
최종 예측값 또는 분류 레이블
```

### 3.2 사용된 LLM 모델 및 하이퍼파라미터
**세 가지 LLM 모델**:
- **RoBERTa-Large** (2.65M 파라미터): 양방향 인코더
- **Llama2-7B** (12.7M 파라미터): GPT 스타일 디코더
- **Mistral-7B** (9.56M 파라미터): 최적화된 GPT 스타일

**ABBA 하이퍼파라미터**:
- Tolerance: $\text{tol} \in \{1 \times 10^{-2}, 1 \times 10^{-4}, 1 \times 10^{-6}\}$
- 디지털화 파라미터: $\alpha \in \{1 \times 10^{-2}, 1 \times 10^{-4}, 1 \times 10^{-6}\}$
- 스케일링: $\text{scl} \in \{1, 2, 3\}$

**QLoRA 파인튜닝**:
- 4-bit 양자화
- LoRA rank: $r \in \{16, 64, 256\}$
- 드롭아웃: 0.05
- 억제 임계값(inhibition): 0.3

***

## 4. 성능 향상 및 실험 결과
### 4.1 시계열 분류 (UCR Archive) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ec33e53b-d35c-49f4-8912-7a38fc4a79bc/2411.18506v4.pdf)
**표 V 결과**:

| 데이터셋 | 클래스 | 기호 | RoBERTa J2 | Llama2 J2 | Mistral J2 | SOTA [germain-forestier](https://germain-forestier.info/publis/dmkd2021.pdf) |
|---------|--------|------|-----------|-----------|-----------|---------|
| Coffee | 2 | 701 | 89.3 | 96.5 | 89.3 | 100 |
| Earthquakes | 2 | 940 | 74.8 | 76.3 | 76.3 | 78.4 |
| ECG5000 | 5 | 10,334 | 76.0 | 74.7 | 73.4 | 94.0 |
| Strawberry | 2 | 3,593 | 85.1 | 84.9 | 88.4 | 97.6 |
| Wafer | 2 | 4,805 | 96.8 | 93.5 | 95.2 | 100 |

일부 데이터셋(Coffee, Earthquakes, Wafer)에서는 SOTA와 경쟁 가능한 수준이지만, 대부분 SOTA에는 미치지 못한다.

### 4.2 의료 시계열 분류 (표 VI) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ec33e53b-d35c-49f4-8912-7a38fc4a79bc/2411.18506v4.pdf)
| 데이터셋 | 클래스 | RoBERTa (r=64) | Llama2 (r=16) | Mistral (r=16) | CNN | BiRNN | LSTM |
|---------|--------|--------------|---------------|----------------|-----|-------|------|
| EEG | 2 | 66.0 | 57.4 | 58.0 | 53.1 | 55.3 | 50.7 |
| PTB-DB | 2 | 90.6 | 99.0 | 98.9 | 99.4 | 97.0 | 90.7 |
| MIT-BIH | 5 | 86.4 | 89.6 | 89.3 | 93.4 | 96.5 | 88.1 |

PTB-DB에서 SOTA에 거의 근접한 성능(99.0% vs 99.4%)을 달성한다.

### 4.3 시계열 회귀 (TSER) - **새로운 SOTA** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ec33e53b-d35c-49f4-8912-7a38fc4a79bc/2411.18506v4.pdf)
**표 VII 결과** (19개 데이터셋 중 15개에서 SOTA 달성):

| 데이터셋 | RoBERTa | Llama2 | Mistral | SOTA |
|---------|---------|--------|---------|------|
| BeijingPM10Quality | 66.07 | 93.26 | 65.25 | 93.14 |
| BeijingPM25Quality | 54.16 | 76.75 | 53.50 | 59.50 |
| Benzene | 4.00 | 5.56 | 4.03 | 0.64 |
| AustraliaRainfall | 4.36 | 6.01 | 4.30 | 8.12 |
| PPGDalia | 9.32 | 12.50 | 9.02 | 9.92 |

**주요 성과**: 특히 수건함 적용 분야(에너지, 환경 모니터링)에서 기존 기계학습 SOTA를 상당 부분 초과.

### 4.4 시계열 예측 (표 VIII) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ec33e53b-d35c-49f4-8912-7a38fc4a79bc/2411.18506v4.pdf)
**ETT 데이터셋에서의 성능** (input:168, predict:24):

| 모델 | MSE | MAE |
|------|-----|-----|
| Llama2 (r=16) | 0.689 | 0.653 |
| Mistral (r=16) | 0.631 | 0.681 |
| Informer [semanticscholar](https://www.semanticscholar.org/paper/eb99d858656b9726004d9320966a2701db8d7afc) | 0.626 | 0.677 |
| Time-LLM [ieeexplore.ieee](https://ieeexplore.ieee.org/document/9225319/) | 0.577 | 0.549 |

예측 작업에서는 Informer와 Time-LLM에 비해 다소 뒤처진다. 이는 LLM의 "hallucination" 및 ABBA의 진동/변동 증폭 경향 때문이다.

### 4.5 재구성 및 복원 성능 (표 IV) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ec33e53b-d35c-49f4-8912-7a38fc4a79bc/2411.18506v4.pdf)
ETTh1 데이터 (7 features, input-168-predict-96):

| tol & α | 기호 개수 | MSE | MAE | 상관계수 |
|---------|----------|-----|-----|---------|
| 1×10⁻² | 846 | 2.5×10⁻⁷ | 1×10⁻² | 1.0 |
| 1×10⁻⁴ | 2,713 | 4.2×10⁻⁸ | 1.4×10⁻⁴ | 1.0 |
| 1×10⁻⁶ | 2,789 | 3.2×10⁻⁸ | 1.3×10⁻⁴ | 1.0 |

**해석**: tol과 α를 감소시키면 더 많은 기호를 사용하고 재구성 오류가 감소하지만, 모든 경우 1.0의 완벽한 상관계수를 유지한다.
---

## 5. 모델의 일반화 성능 향상 가능성
### 5.1 Zipf의 법칙과 언어적 특성
**그림 5**에서 ABBA가 생성한 유니그램이 Zipf의 법칙을 대체로 만족함을 보여준다. 이는 ABBA 기호들이 자연 언어 단어와 유사한 분포를 가지며, LLM의 학습된 표현과 더 잘 정렬됨을 의미한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ec33e53b-d35c-49f4-8912-7a38fc4a79bc/2411.18506v4.pdf)

$$\text{Zipf's Law}: P(\text{rank}) \propto \frac{1}{\text{rank}^s}$$

### 5.2 전이 학습(Transfer Learning) 가능성
LLM-ABBA의 강점:

1. **도메인 적응성**: 기호화된 표현이 도메인-불변적이므로, 한 도메인에서 학습한 LLM을 다른 도메인에 적용하기 용이하다.

2. **데이터 효율성**: ABBA 압축이 원본 시계열의 95-99% 정보를 보존하면서 10-100배 차원 감소를 달성하므로, 적은 데이터로도 학습 가능하다.

3. **모달리티 정렬**: 기호가 LLM의 원래 토큰과 동일한 임베딩 공간에서 작동하므로, 추가 매핑 레이어 불필요하다.

### 5.3 한계 및 제약
논문에서 명시된 일반화 제약: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ec33e53b-d35c-49f4-8912-7a38fc4a79bc/2411.18506v4.pdf)

1. **LLM Hallucination**: 더 많은 토큰 생성 시 "환각" 현상이 누적되어 장기 예측에서 성능 저하.

2. **토큰 길이 제약**: RoBERTa 512, Llama2/Mistral 4,096 토큰 제한으로 인해 매우 긴 시계열(>4,000 포인트) 분석 불가.

3. **진동 증폭**: ABBA는 추세와 진폭을 포착하지만, 예측된 기호의 복원 시 진동이 증폭되어 단기 예측(24-96 스텝)에 국한.

4. **FAPCA의 부분적 해결**: FAPCA 도입 후에도 잘못된 길이 $\text{len}_i$는 여전히 경미한 드리프트 야기 가능.

***

## 6. 2020년 이후 관련 최신 연구 비교 분석
### 6.1 연구 진화 단계
#### **단계 1: 패칭 기반 접근 (2020-2022)**

**주요 논문**:
- Transformers for Time Series (TST): 기본 transformer 구조
- PatchTST: 패칭으로 시계열 입력 재구성

**특징**:
- 전문화된 시계열 아키텍처 필요
- 높은 파라미터 수 (수백만 개)
- 태스크별 특화 모델 요구

***

#### **단계 2: LLM 직접 적용 시도 (2023)**

**주요 논문**:
- **GPT4TS** (2023): 두 단계 파인튜닝 (사전학습 + 태스크 특화)
- **Time-LLM** (ICLR 2024): 텍스트 프로토타입 재프로그래밍
- **PromptCast** (2023): 텍스트 쌍으로 시계열 변환

**Time-LLM의 주요 기여**: [arxiv](https://arxiv.org/abs/2310.01728)

$$\text{Input} = \text{Normalize} \rightarrow \text{Patch} \rightarrow \text{TextPrototype} \rightarrow \text{LLM}$$

텍스트 프로토타입 $P_i$는 각 패치를 텍스트로 변환하여 LLM의 고유한 이해 방식에 맞춘다.

**한계**: 여전히 패칭 기반이므로 차원 감소가 제한적이고, 텍스트 변환 과정에서 정보 손실.

***

#### **단계 3: 기호 근사 도입 (2024, **LLM-ABBA**)**

**LLM-ABBA의 차별성**:

| 측면 | Time-LLM | LLM-ABBA |
|------|----------|----------|
| 입력 변환 | 텍스트 프로토타입 | ABBA 기호 (기호 중심) |
| 차원 감소 | 약 10:1 | 100:1 이상 |
| 누적 오류 처리 | 미해결 | FAPCA로 완화 |
| 다중 시계열 일관성 | 제한적 | 완전 일관성 보장 |
| 회귀 성능 | 중상 | SOTA 달성 |

***

#### **단계 4: 파운데이션 모델 시대 (2024-2025)**

**주요 모델**:
- **TimesFM** (Google, 2024): 200M 파라미터, 대규모 사전학습
- **MOIRAI** (Salesforce, 2024): Mixture-of-Experts 구조
- **Chronos** (Amazon, 2024): 토큰 기반 접근
- **Lag-LLama** (ServiceNow, 2023): 지연(lag) 특성 활용

**TimesFM의 아키텍처**: [arxiv](https://arxiv.org/pdf/2310.10688.pdf)

디코더 전용 attention 모델로 대규모 시계열 데이터에서 사전학습 → 제로샷 예측 가능

**성과**: 30,000개 시계열을 통한 벤치마킹에서 TimeGPT-1과 TimesFM이 최고 성능 (TimesFM이 TimeGPT-1의 추론 속도 초과) [reddit](https://www.reddit.com/r/MachineLearning/comments/1d3h5fs/d_benchmarking_foundation_models_for_time_series/)

***

### 6.2 방법론 비교표
| 방법 | 저자 | 연도 | 기술 핵심 | 주요 적용 | 성능 |
|------|------|------|---------|---------|------|
| Time-LLM | Jin et al. | 2024 | 텍스트 프로토타입 | 예측 | SOTA (few-shot) |
| **LLM-ABBA** | **Carson et al.** | **2025** | **ABBA 기호 + FAPCA** | **회귀(SOTA), 분류** | **회귀 SOTA, 예측 중상** |
| TimeCMA | Li et al. | 2024 | 교차 모달리티 정렬 | 예측 | SOTA |
| TimeRAG | Aksu et al. | 2024 | 검색 증강 생성(RAG) | 예측 | 기존 모델 대비 +2.97% |
| LLM-Mixer | Chen et al. | 2024 | 다중 스케일 분해 | 예측 | 중상 |
| TimesFM | Das et al. | 2024 | 파운데이션 모델 | 제로샷 | 최고 (일반적) |
| MOIRAI | Woo et al. | 2024 | Mixture-of-Experts | 예측 | 우수 |
| Chronos | Ansari et al. | 2024 | 토큰화 기반 | 제로샷 | 중상 |

***

### 6.3 일반화 성능 비교
#### **6.3.1 회귀 작업 (TSER)**

LLM-ABBA는 19개 데이터셋 중 **15개에서 SOTA 달성** → **기계학습 기반 방법 우월성 입증**

#### **6.3.2 예측 작업 (시계열 예측)**

- **Time-LLM**: MSE 0.577 (ETTh1 168→24) **← SOTA**
- **TimeCMA**: SOTA 수준
- **LLM-ABBA**: MSE 0.631 (경쟁 가능)
- **TimesFM**: 제로샷에서 우수 (학습 비용 최소)

**해석**: 예측 작업은 여전히 Time-LLM과 파운데이션 모델이 우세. LLM-ABBA의 hallucination 및 진동 증폭이 장기 예측에서 한계.

#### **6.3.3 분류 작업**

- **LLM-ABBA**: UCR 128개 데이터셋 일부에서 경쟁 가능
- **V2S (SOTA 벤치마크)**: 보다 전문화된 방법 우월

***

## 7. 앞으로의 연구에 미치는 영향 및 고려사항
### 7.1 LLM-ABBA의 장기적 영향
#### **7.1.1 기호 근사의 부활**

기존의 SAX, PAA 등 고전적 기호 근사 방법이 LLM 시대에 다시 주목받게 됨. ABBA의 Brownian bridge 모델링이 현대 딥러닝과 결합될 때의 시너지 입증.

#### **7.1.2 모달리티 정렬의 새로운 패러다임**

- 텍스트 프로토타입 (Time-LLM) vs. 기호 표현 (LLM-ABBA)
- 향후: 하이브리드 접근 (시각적 표현, 오디오 표현 등 다중 모달)

#### **7.1.3 효율성과 정확성의 트레이드오프 명확화**

| 모델 | 학습 필요 | 제로샷 | 파라미터 | 추론 속도 |
|------|---------|--------|---------|----------|
| LLM-ABBA | 예 (QLoRA) | 가능 | 7-12.7B | 중상 |
| TimesFM | 대규모 | 우수 | 200M | 빠름 |
| Time-LLM | 예 | 가능 | 7-70B | 중상 |

향후 연구는 "얼마나 큰 모델이 필요한가?"라는 질문에서 "어떤 표현이 최적인가?"로 전환될 가능성.

***

### 7.2 앞으로 연구 시 고려할 점
#### **7.2.1 누적 오류 완전 해결**

**문제**: FAPCA 도입 후에도 길이 반올림 오류가 경미하게 누적.

**해결 방향**:
- 동적 프로그래밍 기반 최적 길이 선택
- 확률적 복원 (베이지안 불확실성 모델링)
- 다중 경로 추론 (앙상블 기호 시퀀스)

$$\tilde{T} = \mathbb{E}[\phi^{-1}(A_{\text{perturb}} | \theta)] \quad \text{(기댓값 기반 복원)}$$

#### **7.2.2 토큰 길이 제약 극복**

**현황**: 4,096 토큰 제한으로 ~4,000 시점 이상 불가

**해결 방향**:
- 계층적 ABBA: 다중 스케일 기호화 (Chronos의 다중 해상도 학습 참고)
- 슬라이딩 윈도우 처리: 긴 시계열을 청크로 분할 후 집계
- 프롬프트 압축: LLM의 토큰 효율성 개선

$$A_{\text{hierarchical}} = [A_{\text{coarse}}, A_{\text{fine}}]$$

#### **7.2.3 LLM Hallucination 정량화 및 제어**

**현황**: "할루시네이션으로 인한 진동 증폭" 명시되지만 정량 분석 부족

**해결 방향**:
- 확률 기반 token sampling (Temperature 최적화)
- 구조화된 생성 (Constrained decoding): 기호만 생성 허용
- 불확실성 추정: 예측 신뢰도 스코어 제공

#### **7.2.4 다중 시계열 간 시간적 동기화**

**현황**: 독립적인 ABBA 적용만 가능

**개선 방향**:
- 교차 시계열 의존성 모델링: 벡터 ABBA (변량 시계열용)
- 동적 시간 정렬(DTW) 기반 기호 정렬
- 다변량 클러스터링: 시간 축과 변량 축 동시 고려

#### **7.2.5 해석성 강화**

**강점**: 기호 표현 자체가 해석 가능

**심화 방향**:
- Chain-of-Thought 프롬프팅: "기호 의미" 설명
- Attention visualization: 어떤 기호가 예측에 중요했는지
- 기호 교집합 분석: 유사한 시계열 패턴 발견

***

### 7.3 제한(Limitations) 재검토
논문이 명시한 5가지 한계: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ec33e53b-d35c-49f4-8912-7a38fc4a79bc/2411.18506v4.pdf)

1. **ABBA의 완전성**: DTW, 2-norm 등 다양한 거리 척도에서 성능 평가 필요
2. **의료 시계열의 특수성**: EEG는 고복잡도로 인해 ~60% 정확도 (개선 여지 있음)
3. **누적 오류 부분 해결**: FAPCA도 장기 예측에서 드리프트 완전 제거 불가
4. **토큰 제약**: 4,096 토큰 상한선이 근본적 제약
5. **Hallucination**: LLM 자체의 한계로 인한 진동 증폭

***

## 8. 결론 및 실무 적용 가능성
### 8.1 LLM-ABBA의 위치
**패러다임 분류**:

- **패칭 + 텍스트 변환** (Time-LLM) → 자연스러운 NLP 활용
- **기호 근사** (LLM-ABBA) ← **수치 정확성과 해석성 최적**
- **파운데이션 모델** (TimesFM) → 최고 일반화 성능 (학습 비용 높음)

### 8.2 추천 적용 분야
| 분야 | 추천 모델 | 이유 |
|------|---------|------|
| **시계열 회귀** | **LLM-ABBA** | SOTA 성능, 해석 가능, 중간 비용 |
| **단기 예측** | Time-LLM | 예측 정확도 최고 |
| **장기 제로샷** | TimesFM | 학습 불필요, 보편적 |
| **의료 진단** | LLM-ABBA (PTB, ECG) | 높은 정확도 + 설명 가능 |
| **이상 탐지** | ABBA 기반 통계 모델 | 기호 패턴의 비정상성 감지 |

### 8.3 향후 10년 전망
**단기 (2025-2026)**:
- 기호 근사 + LLM 하이브리드 추가 연구 활발화
- 토큰 길이 제약 해결 위한 계층적 접근 대두

**중기 (2027-2029)**:
- 멀티모달 기초 모델 출현 (시계열 + 텍스트 + 이미지)
- 도메인 특화 기호화 스킴 개발 (금융, 의료, 제조 산업별)

**장기 (2030+)**:
- 기호 기반 시계열 이해가 표준 패러다임 확립
- "시계열을 읽고 쓸 수 있는" 일반 목적 모델 등장

***

## 참고 논문 및 자료

<span style="display:none">[^1_10][^1_11][^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_8][^1_9]</span>

<div align="center">⁂</div>

[^1_1]: 2411.18506v4.pdf

[^1_2]: https://germain-forestier.info/publis/dmkd2021.pdf

[^1_3]: https://www.semanticscholar.org/paper/eb99d858656b9726004d9320966a2701db8d7afc

[^1_4]: https://ieeexplore.ieee.org/document/9225319/

[^1_5]: https://arxiv.org/abs/2310.01728

[^1_6]: https://arxiv.org/pdf/2310.10688.pdf

[^1_7]: https://www.reddit.com/r/MachineLearning/comments/1d3h5fs/d_benchmarking_foundation_models_for_time_series/

[^1_8]: https://ieeexplore.ieee.org/document/10889933/

[^1_9]: https://arxiv.org/abs/2409.17515

[^1_10]: https://ojs.aaai.org/index.php/AAAI/article/view/34067

[^1_11]: https://arxiv.org/abs/2406.01638

[^1_12]: https://arxiv.org/abs/2403.05798

[^1_13]: https://arxiv.org/abs/2410.11674

[^1_14]: https://arxiv.org/abs/2412.00053

[^1_15]: https://openproceedings.org/2024/conf/edbt/paper-249.pdf

[^1_16]: https://arxiv.org/abs/2306.11025

[^1_17]: https://arxiv.org/pdf/2412.16643.pdf

[^1_18]: https://arxiv.org/html/2504.02119v1

[^1_19]: http://arxiv.org/pdf/2502.11418.pdf

[^1_20]: https://arxiv.org/pdf/2503.09656.pdf

[^1_21]: https://arxiv.org/html/2410.11674

[^1_22]: http://arxiv.org/pdf/2409.14978.pdf

[^1_23]: https://arxiv.org/pdf/2403.05798.pdf

[^1_24]: https://arxiv.org/pdf/2310.01728.pdf

[^1_25]: https://pubmed.ncbi.nlm.nih.gov/35890775/

[^1_26]: https://arxiv.org/html/2512.07624v1

[^1_27]: https://arxiv.org/pdf/1905.00421.pdf

[^1_28]: https://arxiv.org/abs/2512.07624

[^1_29]: https://arxiv.org/html/2510.01111v1

[^1_30]: https://arxiv.org/pdf/1301.5871.pdf

[^1_31]: https://arxiv.org/abs/2510.00742

[^1_32]: https://arxiv.org/html/2508.04231

[^1_33]: https://pubmed.ncbi.nlm.nih.gov/31791613/

[^1_34]: https://arxiv.org/abs/2510.13654

[^1_35]: https://arxiv.org/html/2509.00616v2

[^1_36]: https://arxiv.org/abs/1905.00421

[^1_37]: https://arxiv.org/abs/2512.16022

[^1_38]: https://www.emergentmind.com/topics/symbolic-aggregate-approximation-sax

[^1_39]: https://research.google/blog/time-series-foundation-models-can-be-few-shot-learners/

[^1_40]: https://openreview.net/forum?id=Unb5CVPtae

[^1_41]: https://www.cs.ucr.edu/~eamonn/SAX.pdf

[^1_42]: https://proceedings.iclr.cc/paper_files/paper/2024/file/680b2a8135b9c71278a09cafb605869e-Paper-Conference.pdf

[^1_43]: https://pyts.readthedocs.io/en/stable/auto_examples/approximation/plot_sax.html

[^1_44]: https://arxiv.org/html/2504.04011v1

[^1_45]: https://proceedings.neurips.cc/paper_files/paper/2024/file/6ed5bf446f59e2c6646d23058c86424b-Paper-Conference.pdf

[^1_46]: https://www.reddit.com/r/datascience/comments/1e865bt/the_rise_of_foundation_timeseries_forecasting/

[^1_47]: https://www.linkedin.com/pulse/llms-foundational-models-time-series-forecasting-yet-good-bergmeir-bprwf

[^1_48]: https://justkook.blogspot.com/2017/05/symbolic-aggregate-approximation-sax.html

[^1_49]: https://arxiv.org/abs/2504.04011
