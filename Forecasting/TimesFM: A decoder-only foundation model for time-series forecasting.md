
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
