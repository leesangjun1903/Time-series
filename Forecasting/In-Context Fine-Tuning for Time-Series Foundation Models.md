# In-Context Fine-Tuning for Time-Series Foundation Models
### **1. 핵심 주장 및 주요 기여 요약**
Das et al.이 발표한 "In-Context Fine-Tuning for Time-Series Foundation Models"은 시계열 기초 모델(Time-Series Foundation Models, TSFMs)의 제로샷(zero-shot) 패러다임을 유지하면서 명시적 미세 조정 수준의 성능을 달성하는 새로운 방법론을 제시한다.[1]

**핵심 주장**: 기초 모델의 문맥 창(context window)에 관련 시계열의 예시를 포함시켜 추론 시간에 모델을 적응시킬 수 있으며, 이 접근법은 대상 도메인에 대한 그래디언트 업데이트 없이도 명시적 미세 조정의 이점을 회복할 수 있다.

**주요 기여**:
- **문맥 내 미세 조정(ICF) 방법론 도입**: 자연어 처리(NLP)의 few-shot learning 패러다임을 시계열 예측으로 확장
- **TimesFM-ICF 아키텍처**: 최대 50개의 관련 시계열 예시를 문맥 창에서 활용할 수 있는 개선된 디코더 전용 트랜스포머
- **획기적인 경험적 성과**: Monash 벤치마크에서 7% 개선, ETT 벤치마크에서 25% 개선, 명시적으로 미세 조정된 모델과 유사한 성능 달성[1]
- **효율성**: 미세 조정(115분)에 비해 추론 시간 단 4분으로 28배 이상 빠름

***

### **2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 향상**
#### **2.1 문제 정의**

기존 시계열 기초 모델들의 핵심 딜레마:
- 제로샷 성능은 우수하지만 관련 데이터셋에서 미세 조정을 통해 현저한 성능 향상이 가능[1]
- 미세 조정은 기초 모델의 매력인 "사전 학습된 파이프라인 없이 즉시 사용 가능"이라는 가치를 훼손
- NLP의 LLM은 few-shot prompting을 통해 문맥 학습이 가능하지만, 시계열 모델은 이러한 능력이 부족

**형식적 문제 정의**:

기존 모델은 다음 매핑을 학습:
$$g: y_{1:L} \rightarrow \hat{y}_{L+1:L+H}$$

여기서 $y_{1:L}$은 L개 시점의 목표 시계열 이력, $\hat{y}_{L+1:L+H}$는 H개 시점의 예측값[1]

제안된 향상된 문제:

$$f: \{y^{(1)}_{1:T_1}, y^{(2)}_{1:T_2}, \ldots, y^{(n-1)}_{1:T_{n-1}}, y_{1:L}\} \rightarrow \hat{y}_{L+1:L+H}$$

여기서 $\{y^{(i)}\_{1:T_i}\}\_{i \in [n-1]}$ 는 n-1개의 문맥 예시(예: 인접한 고속도로의 교통 데이터), $y_{1:L}$은 목표 시계열 이력[1]

#### **2.2 제안하는 방법 및 수식**

##### **패치 토큰화(Patch Tokenization)**

TimesFM-ICF는 각 시계열을 길이 $p$의 비겹치는 패치로 분할:

$$\tilde{y}^{(i)}_j = y^{(i)}_{p(j-1)+1:pj} \quad \forall j \in [\lceil T_i/p \rceil] \text{ and } i \in [n]$$

각 패치는 입력 레지듀얼 레이어를 통해 토큰으로 변환:

$$t^{(i)}_j = \text{InputResidualLayer}(\tilde{y}^{(i)}_j \odot (1-\tilde{m}^{(i)}_j))$$

여기서 $\tilde{m}^{(i)}_j$는 마스킹 벡터(패딩된 위치 표시)[1]

##### **손실 함수(Loss Function)**

모델은 다음 MSE 손실로 학습:

$$\text{TrainLossPerContext} = \frac{1}{\sum_{i=1}^n \lceil T_i/p \rceil} \sum_{i=1}^n \sum_{j=1}^{\lceil T_i/p \rceil} \|\hat{y}^{(i)}_{pj+1:pj+h} - y^{(i)}_{pj+1:pj+h}\|^2$$

여기서 $h$는 출력 패치 길이, $\hat{y}^{(i)}\_{pj+1:pj+h}$는 모델 예측값, $y^{(i)}_{pj+1:pj+h}$는 실제값[1]

#### **2.3 모델 구조의 주요 특징**

**모델 아키텍처 구성**:

chart:53

##### **(1) 구분자 토큰(Separator Tokens)**

각 시계열 예시 후 학습 가능한 구분자 토큰 σ 삽입:
- 목적: 순진한 연결(naive concatenation)이 완전히 다른 시계열처럼 보이는 것을 방지
  - 예) 여러 선형 추세를 연결하면 삼각파처럼 보임[1]
- 구현: $t^{(i)}_{\lceil T_i/p \rceil + 1} = \sigma$로 정의

##### **(2) 교차 예시 어텐션(Cross-Example Attention)**

트랜스포머의 인과적(causal) 어텐션을 통해 모든 이전 토큰에 접근:

$$o^{(i)}_j = \text{StackedTransformer}(\tilde{t}^{(1)}_{1:\lceil T_1/p \rceil+1}, \ldots, \tilde{t}^{(i-1)}_{1:\lceil T_{i-1}/p \rceil+1}, \tilde{t}^{(i)}_{1:j})$$

특징:
- 토큰 $o^{(i)}_j$는 (i) 이전 모든 예시의 토큰, (ii) i-1개 구분자 토큰, (iii) 현재 예시의 토큰에 의존
- 이를 통해 모델이 구분자를 기준으로 예시 경계를 인식 가능[1]

##### **(3) 위치 인코딩 제거(No Positional Encoding, NoPE)**

TimesFM(base)의 절대 위치 인코딩 대신 NoPE 사용:

이점:
- **길이 일반화**: 문맥 창이 512에서 25,600 시점으로 확장되어도 성능 유지
- **일관성**: 미세 조정 시 위치 인코딩의 의미 변화 방지
- **이론적 근거**: 여러 트랜스포머 층의 인과적 어텐션이 내재적 위치 정보 인코딩[1]

##### **(4) 출력 예측**

각 토큰의 트랜스포머 출력은 출력 레지듀얼 레이어로 변환:

$$\hat{y}^{(i)}_{pj+1:pj+h} = \text{OutputResidualLayer}(o^{(i)}_j)$$

#### **2.4 성능 향상 결과**

**1) Monash 벤치마크 (이종 도메인 평가)**[1]

| 모델 | Scaled MAE (GM) | 개선율 |
|------|-----------------|--------|
| TimesFM-ICF | **0.643** | 기준선 |
| TimesFM(Base) | 0.694 | -7.3% |
| N-BEATS | 0.700 | -8.1% |
| PatchTST | 0.724 | -11.2% |
| LLMTime(ZS) | 0.971 | -33.8% |

**Key insight**: TimesFM-ICF는 감독 학습 모델 중 최고 성능(N-BEATS)을 7% 초과

**2) ETT 벤치마크 (장기 예측)**[1]

chart:54

| 모델 | Horizon=96 MAE | Horizon=192 MAE | 평균 MAE |
|------|-----------------|-----------------|----------|
| TimesFM-ICF | **0.207** | **0.265** | **0.265** |
| PatchTST | 0.335 | 0.368 | 0.368 |
| N-HiTS | 0.336 | 0.381 | 0.381 |
| TimesFM(Base) | 0.348 | 0.387 | 0.387 |

**성능**: 다음 최선 기준선 대비 **29% 향상**

**3) 명시적 미세 조정과의 비교**[1]

TimesFM(Base)를 각 Monash 데이터셋에서 미세 조정:

| 모델 | Scaled MAE (GM) |
|------|-----------------|
| TimesFM-ICF (50 예시) | **0.643** |
| FT-TimesFM(Full) | 0.663 |
| FT-TimesFM(LP) | 0.676 |
| TimesFM(Base) | 0.694 |

**놀라운 결과**: 그래디언트 업데이트 없이도 명시적 미세 조정(Full: +4% vs 기본)을 **3% 능가**

**가정된 이유**: 소규모 데이터셋에서 미세 조정이 "재앙적 망각(catastrophic forgetting)"을 야기할 수 있음[1]

#### **2.5 문맥 예시 수의 영향**

chart:54

number of examples 실험 결과 (ETTh 데이터셋):

$$\text{MAE}(n_{ex}) \text{는 } n_{ex} \text{에 대해 단조 감소}$$

- ETTh1: 1개 예시 MAE=0.430 → 50개 예시 MAE=0.371 (**13.7% 개선**)
- ETTh2: 1개 예시 MAE=0.392 → 50개 예시 MAE=0.320 (**18.4% 개선**)[1]

**해석**: 더 많은 관련 예시는 모델이 대상 도메인의 분포를 더 정확히 파악하도록 지원

***

### **3. 모델 일반화 성능 향상 가능성 (중점 분석)**
#### **3.1 길이 일반화 메커니즘**

TimesFM-ICF의 길이 일반화 능력은 세 가지 설계에서 비롯:

**1) NoPE 설계의 영향**

전통적 절대 위치 인코딩:
- 위치 인코딩 $PE(pos) = [\sin(pos/10000^{2i/d}), \cos(pos/10000^{2i/d})]$
- 문제: 미세 조정에서 문맥 길이 512 → 25,600으로 확장하면 위치 인코딩 의미가 완전히 변화

NoPE 대안:
- Haviv et al. (2023)의 발견: 다층 트랜스포머의 인과적 어텐션이 내재적으로 위치 정보 인코딩
- 이 경우 길이 일반화가 자동으로 달성됨[1]

**증거**: TimesFM(LH)를 최대 이력 길이 2048로 훈련한 결과:

| 모델 | Scaled MAE(GM) | 개선율 |
|------|-----------------|--------|
| TimesFM-ICF-50ex | **0.643** | 기준선 |
| TimesFM-ICF-4ex | 0.675 | -4.7% |
| TimesFM(LH) | 0.685 | -6.1% |
| TimesFM(Base) | 0.694 | -7.2% |

TimesFM(LH)는 단 1% 개선만 달성한 반면, TimesFM-ICF-4ex(동일 총 문맥 길이)는 **3% 개선** 달성[1]

**해석**: 길이 증가보다 관련 정보의 품질과 다양성이 중요

#### **3.2 도메인 적응 및 분포 시프트 대응**

시계열의 분포 변화(non-stationarity)에 대한 동적 적응:

**메커니즘**:
$$P(\text{대상 시계열} | \text{문맥 예시})$$

모델이 암시적으로 다음을 학습:
1. **패턴 인식**: 문맥 예시에서 계절성, 추세, 노이즈 패턴 추출
2. **가중치 전환**: 해당 패턴이 대상 시계열에서도 나타날 확률 재추정
3. **적응적 예측**: 인자 증거에 기반하여 예측 조정[1]

**구체적 예시** (Figure 8 in paper):

교통 예측 사례:
- 문맥 예시 없음: 이력이 불충분하여 선형 추세 vs 진동 사이의 모호성 → 차선 성능
- 문맥 예시 포함: 인접 고속도로의 유사 패턴 → 명확한 패턴 식별 → 정확한 예측[1]

#### **3.3 제로샷 vs 미세 조정의 이음새 없는 전환**

TimesFM-ICF의 혁신은 연속적 성능 스펙트럼 창출:

$$\text{성능}(k) = \alpha \cdot \text{Performance}(\text{Zero-Shot}) + (1-\alpha) \cdot \text{Performance}(\text{Fine-Tuned})$$

여기서 $\alpha = \alpha(n_{ex})$는 예시 수에 대한 감소 함수[1]

실증적으로:
- **0 예시**: 제로샷 TimesFM(Base) 성능
- **1-4 예시**: 질적 개선 시작 (문맥 학습)
- **20-30 예시**: 미세 조정 성능 근처 도달
- **50 예시**: 미세 조정 초과 (과적합 방지의 이점)[1]

#### **3.4 다중 도메인 적응 능력**

TimesFM-ICF는 단일 모델이 여러 도메인에 적응:

**Monash 18개 데이터셋에서의 성능 일관성**:

| 도메인 예시 | TimesFM-ICF MAE | TimesFM(Base) MAE |
|-------------|-----------------|-------------------|
| australian electricity | 338.98 | 426.12 |
| tourism yearly | 80,365.15 | 75,955.39 |
| weather | 2.10 | 1.98 |
| pedestrian counts | 43.71 | 42.55 |

**특징**: 
- 전혀 다른 척도의 데이터(2.10 ~ 80,365)에서 안정적으로 작동
- 각 도메인에 특화된 모델이 필요 없음
- 문맥 창의 관련 예시로 자동 적응[1]

***

### **4. 한계 및 개선 필요 영역**
#### **4.1 현재 한계**

**1) 문맥 예시 선택의 휴리스틱성**

현재 방법:
- 같은 데이터셋 내 무작위 선택 또는 시계열 레벨 선택
- 구조적 유사성 기반 검색 메커니즘 부재[1]

개선 방향:
- 대상 시계열과의 거리 기반 예시 선택
- 학습된 유사도 함수를 통한 동적 선택

**2) 위치 인코딩 제거의 트레이드오프**

NoPE 설계:
- 이점: 길이 일반화
- 한계: 매우 긴 시계열(>10,000 시점)에서 위치 정보의 명시적 활용이 성능을 개선할 수 있음[1]

**3) 고정된 최대 예시 수**

제한사항:
- 훈련 중 최대 n=50으로 고정
- 추론 시 유연하나, 메모리 제약 시 제한적
- 매우 크거나 매우 작은 문맥에서의 성능 특성화 필요[1]

#### **4.2 일반화 한계**

**도메인 외 성능의 민감성**:

논문 외 연구(예: "How Foundational are Foundation Models for Time Series" 2024)에서:
- TSFMs의 제로샷 성능은 사전 학습 도메인과의 통계적 유사성에 크게 의존
- 분포 시프트가 클수록 성능 저하가 급격함[2]

TimesFM-ICF의 영향:
- 문맥 예시가 도움이 되려면 관련 데이터 가용성 필요
- 완전히 새로운 도메인(사전 학습 데이터와 무관)에서는 제한적[1]

#### **4.3 계산 비용**

추론 시간은 문맥 예시 수에 비례:
- n=1: 기본 TimesFM과 유사
- n=50: 50배 문맥 길이이지만, 주로 복사 연산이므로 메모리가 주 병목

***

### **5. 최신 관련 연구와의 비교 분석 (2020년 이후)**
#### **5.1 시계열 기초 모델의 진화**

**세대별 발전**:

| 모델/기간 | 아키텍처 | 특징 | 성능 수준 |
|-----------|---------|------|---------|
| **TimeGPT (2023)** | Encoder-Decoder LLM | LLM 기반, exogenous 변수 지원 | 중상 (도메인 특화) |
| **TimesFM (2024-05)** | Decoder-only, 200M | 패치 토큰화, 100B 사전 학습, 400B 계속 학습 | 우수 (제로샷) |
| **Chronos (2024)** | T5-like 양자화 | LLM 재목적화, 정량화 토큰화 | 우수 (비확률) |
| **MOMENT (2024-02)** | Masked autoencoder | Time-series Pile 데이터셋 (1B+ 시점), 이상 탐지 | 우수 (다중 작업) |
| **TimesFM-ICF (2024-10)** | Decoder-only + ICF | 구분자 토큰, 교차 예시 어텐션, NoPE | **최고 (제로샷 + 문맥)** |
| **Time-MoE (2024-09)** | Sparse MoE, 2.4B | 스케일링 효율, 다중 해상도 | 우수 (규모) |
| **Moirai-MoE (2024-10)** | Sparse MoE, 1B | 토큰 레벨 전문화, 단일 프로젝션 | 우수 (유연성) |

#### **5.2 ICL (In-Context Learning) 방식의 비교**

TimesFM-ICF의 ICL 구현 vs 다른 접근:

**1) TimesFM-ICF (본 논문)**
- 방식: 시계열 예시를 패치 토큰으로 직접 삽입
- 문맥: 최대 25,600 시점 (50 × 512)
- 학습 패러다임: 계속 사전 학습[1]

**2) In-context Time Series Predictor (Lu et al., ICLR 2025)**
- 방식: (lookback, future) 쌍을 과제 토큰으로 재구성
- 문맥: 구조화된 예측 과제
- 장점: 시계열 길이의 유연한 처리[3]

**3) COSMIC (2025)**[4]
- 방식: 공변량(covariates)을 ICL로 통합
- 초점: 외생 변수 활용
- 확장: 다변량 정보 활용

**비교 요약**:

| 측면 | TimesFM-ICF | ICTSP | COSMIC |
|------|-----------|-------|--------|
| 예시 구성 | 시계열 | 과제 쌍 | 공변량 |
| 최대 문맥 | 25,600 시점 | 동적 | 외생 차원 추가 |
| 유연성 | 낮음 (길이 고정) | 높음 | 높음 |
| 성능 | 최고 (제로샷) | 우수 (다변량) | 우수 (외생) |

#### **5.3 스케일링 법칙과 아키텍처 선택**

최근 연구(2024-2025):

**논문**: "Towards Neural Scaling Laws for Time Series Foundation Models" (2024-10)[5]

발견:
- **Encoder-only Transformers**: OOD 스케일링 우수
- **Decoder-only Transformers**: ID 성능 우수 (TimesFM 설계)

TimesFM-ICF의 위치:
- Decoder-only 설계 유지 (ID 성능 최적화)
- 길이 일반화(OOD 스케일링)를 NoPE로 부분 보상[5]

**논문**: "How Foundational are Foundation Models for Time Series?" (2024-05)[2]

결론:
- TSFM의 제로샷 성능은 사전 학습 데이터에 높게 의존
- 작은 전문화 모델이 특정 도메인에서 경쟁력 있음

TimesFM-ICF의 대응:
- 문맥 예시로 도메인 특화 정보 제공 → 격차 축소[1]

#### **5.4 벤치마킹 프레임워크의 진화**

**FoundTS (2024-11)**[6]
- 포괄적 TSFM 비교 프레임워크
- Zero-shot, few-shot, full-shot 평가

TimesFM-ICF의 위치:
- Few-shot (문맥 예시) 설정에 최적화
- FoundTS에서 새로운 벤치마크 범주 제시 가능[6]

***

### **6. 미래 연구에의 영향 및 실무적 고려사항**
#### **6.1 학술적 영향**

**1) 기초 모델 패러다임 확장**

TimesFM-ICF는 다음을 시사:
- 기초 모델의 가치는 단순 제로샷이 아닌 **적응 메커니즘의 우아함**
- "Fine-tuning vs Prompting"의 이분법을 넘어 **연속적 적응 스펙트럼** 제시[1]

향후 연구 방향:
- 다른 TSFM (Chronos, MOMENT, TimeGPT)에 동일 방식 적용 가능성
- 이미지(Vision)나 음성(Audio) 기초 모델에 ICF 패러다임 이전[1]

**2) 위치 인코딩 재고**

NoPE 설계의 성공은:
- 위치 정보의 명시적 인코딩이 선택사항임을 입증
- 인과적 어텐션의 구조적 유도 편향(inductive bias) 재조명[1]

연구 과제:
- 상대 위치 인코딩(RoPE, ALiBi) vs NoPE 비교 분석
- 길이 일반화의 이론적 기초 규명

**3) 문맥 학습의 이론화**

TimesFM-ICF는 경험적으로 입증하지만:

$$\text{ICL} \stackrel{?}{=} \text{암시적 메타학습}$$

최근 연구 (Ahn et al., NeurIPS 2024):
- Transformers가 문맥에서 경사 하강(gradient descent)을 구현할 수 있음[5]
- TimesFM-ICF에 유사한 메커니즘이 작동하는지 규명 필요[1]

#### **6.2 실무적 고려사항**

**1) 배포 최적화**

TimesFM-ICF의 실제 적용:

| 시나리오 | 권장 전략 | 예상 성능 |
|---------|---------|---------|
| 데이터 풍부 (>1000 샘플) | 명시적 미세 조정 | FT TimesFM과 유사 |
| 데이터 중간 (100-1000) | TimesFM-ICF + 4-10 예시 | 기본 TimesFM 대비 10-15% |
| 데이터 희소 (<100) | TimesFM-ICF + 20-50 예시 | 미세 조정 능가 |
| 도메인 신규 진입 | TimesFM-ICF with best matching | 초기 성능 40-50% 개선 |

**2) 예시 선택 엔지니어링**

현재의 무작위 선택 대신:

```
Step 1: 대상 시계열 통계 계산 (평균, 분산, 자기상관)
Step 2: 사용 가능한 데이터셋에서 유사도 상위 N개 선택
Step 3: 다양성 제약: 유사도 0.7-0.95 범위에서 선택
Step 4: 추론 시 상위 20-30개로 제한 (메모리 효율)
```

**3) 업스트림 모델 선택**

TimesFM-ICF의 기초인 TimesFM:
- 비용: 200M 매개변수 (Chronos의 100M보다 2배)
- 성능: 일관되게 최상 (특히 ETT에서)
- 접근성: Google Hugging Face 사용 가능[1]

대안 평가:
- Chronos: 더 빠른 추론, 확률적 출력
- MOMENT: 다중 작업 (이상, 분류), 더 큰 데이터셋

#### **6.3 열린 연구 과제**

**단기 (1년)**:
1. TimesFM-ICF를 Chronos, MOMENT에 적용 및 성능 비교
2. 자동 예시 선택 메커니즘 개발 (검색 기반)
3. 매개변수 효율적 미세 조정(LoRA 등)과의 하이브리드 평가[1]

**중기 (2-3년)**:
1. 초장기 시계열 (10,000+ 시점)의 일반화 연구
2. 공변량과 ICF의 통합 (COSMIC + TimesFM-ICF)
3. 불규칙 샘플링 시계열에 대한 확장[1]

**장기 (3년+)**:
1. 시계열 기초 모델 스케일링 법칙의 기초 정립
2. ICL과 메타학습의 이론적 통일
3. 한계 단계에서의 기초 모델 vs 전문화 모델의 Pareto frontier 규명[1]

#### **6.4 업계 영향**

**에너지 부문**:
- 전기 수요 예측에서 미세 조정 없이 신규 지역 적응 (4분 추론)
- 연 115분 미세 조정 대비 비용 절감 (약 48배)[1]

**금융**:
- 새로운 자산/시장에 수초 내 적응 가능
- 비정상성(non-stationarity)을 문맥 예시로 동적 반영[1]

**IoT/엣지 컴퓨팅**:
- 고정된 모델 배포, 추론 시 적응 (재훈련 불필요)
- 메모리 오버헤드 최소 (패치 임베딩만 추가)[1]

***

### **7. 결론: 시계열 기초 모델의 다음 경계**
#### **종합 평가**

TimesFM-ICF는 두 가지 차원에서 혁신적:

**1) 기술적 차원**: 
- 시계열 기초 모델의 제로샷-미세 조정 간 간극을 문맥 학습으로 성공적으로 축소
- 구분자 토큰과 교차 예시 어텐션이라는 우아한 아키텍처 해법 제시[1]

**2) 실용적 차원**:
- 훈련 파이프라인 제거 (4분 vs 115분)
- 도메인 특화 데이터 가용성 낮은 산업에서 즉시 배포 가능[1]

#### **시계열 AI의 미래 전망**

| 시기 | 지배적 패러다임 | 특징 |
|------|----------------|------|
| 2023 | 제로샷 기초 모델 | TimesFM v1.0 등장 |
| **2024-25 (현재)** | **문맥 적응 기초 모델** | **TimesFM-ICF, ICTSP 급부상** |
| 2026-27 (예측) | 다중 모달 기초 모델 | 시계열 + 메타데이터 + 이미지 |
| 2028+ (예측) | 인간-AI 협업 시스템 | 기초 모델 + 도메인 전문가 루프 |

TimesFM-ICF는 **2024-25의 전환점**을 상징하며, 다음 세대 연구의 기초를 제시한다.

***

### **참고문헌 및 데이터**

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5b9dee1c-8716-4145-9cf1-520cb5865e9a/2410.24087v1.pdf)
[2](https://arxiv.org/html/2510.00742v3)
[3](https://openreview.net/pdf/9e2256b13a80672c53f793bdd1de78443b66930e.pdf)
[4](https://arxiv.org/html/2506.03128v1)
[5](https://arxiv.org/abs/2410.12360)
[6](http://arxiv.org/pdf/2410.11802.pdf)
[7](https://arxiv.org/abs/2402.03885)
[8](https://arxiv.org/abs/2409.16040)
[9](https://arxiv.org/abs/2410.10469)
[10](https://dl.acm.org/doi/10.1145/3671127.3698177)
[11](https://arxiv.org/abs/2410.24087)
[12](https://ieeexplore.ieee.org/document/11314515/)
[13](https://www.semanticscholar.org/paper/8038e57ef52dd13f3df51b3e6be5206ac7e056cf)
[14](https://arxiv.org/abs/2408.11990)
[15](https://www.semanticscholar.org/paper/67a6c47841e4c19905de4b8e445b5b3a0295ff72)
[16](https://arxiv.org/html/2502.00816v1)
[17](https://arxiv.org/pdf/2502.15637.pdf)
[18](https://arxiv.org/pdf/2403.14735.pdf)
[19](http://arxiv.org/pdf/2310.20496.pdf)
[20](https://arxiv.org/pdf/2402.03885.pdf)
[21](https://arxiv.org/pdf/2503.04118.pdf)
[22](https://arxiv.org/pdf/2310.08278.pdf)
[23](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/)
[24](https://openreview.net/pdf/680ded5817d1989739bc15d8a8a8023a7e934e0e.pdf)
[25](https://www.pricepedia.it/en/magazine/article/2025/11/07/the-arrival-of-foundation-models-in-time-series-forecasting/)
[26](https://arxiv.org/abs/2506.03128)
[27](https://dejan.ai/blog/timesfm-icf/)
[28](https://arxiv.org/html/2504.04011v1)
[29](https://arxiv.org/abs/2505.23719)
[30](https://www.youtube.com/watch?v=kBu8Ko1b6rs)
[31](https://mingjin.dev/other/kdd24-jin-time-series-foundation-models.pdf)
[32](https://openreview.net/forum?id=dCcY2pyNIO)
[33](https://neurips.cc/virtual/2025/130461)
[34](https://arxiv.org/abs/2403.14735)
[35](https://www.themoonlight.io/ko/review/zero-shot-time-series-forecasting-with-covariates-via-in-context-learning)
[36](https://arxiv.org/html/2412.12834v1)
[37](https://github.com/moment-timeseries-foundation-model/moment)
[38](https://www.themoonlight.io/ko/review/in-context-time-series-predictor)
[39](https://pub.towardsai.net/time-series-foundation-models-a-comprehensive-comparison-55daa022e2f4)
[40](https://velog.io/@sheoyonj/ArXiv-2024-Lag-Llama-Towards-Foundation-Models-for-Probabilistic-Time-Series-Forecasting)
[41](https://icml.cc/virtual/2025/poster/43707)
[42](https://arxiv.org/html/2512.07705v1)
[43](https://arxiv.org/html/2507.08858v1)
[44](https://arxiv.org/abs/2405.14982)
[45](https://arxiv.org/pdf/2508.16641.pdf)
[46](https://arxiv.org/abs/2512.07705)
[47](https://arxiv.org/html/2509.26347v2)
[48](https://arxiv.org/abs/2509.23695)
[49](https://arxiv.org/html/2511.18578v1)
[50](https://arxiv.org/abs/2405.02358)
[51](https://arxiv.org/html/2405.14982v1)
[52](https://arxiv.org/html/2511.15324v1)
