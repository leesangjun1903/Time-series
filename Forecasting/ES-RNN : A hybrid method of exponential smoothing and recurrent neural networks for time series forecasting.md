
# A hybrid method of exponential smoothing and recurrent neural networks for time series forecasting

## 1. 핵심 주장과 주요 기여

Slawek Smyl의 "A hybrid method of exponential smoothing and recurrent neural networks for time series forecasting"(2020)은 M4 국제 예측 경쟁의 우승 논문으로, 시계열 예측에서 근본적인 패러다임 전환을 제시합니다.[1]

### 핵심 주장

논문의 중심 논증은 다음과 같습니다: 기계학습(ML) 알고리즘이 이미지 인식, 음성 처리 등에서 압도적 성공을 거두었으나, **시계열 예측에서는 전통 통계 모델을 능가하지 못한다**는 문제 제시입니다. 이는 세 가지 구조적 어려움에서 비롯됩니다.[1]

첫째, 신경망은 본질적으로 시계열 데이터를 위해 설계되지 않았으므로, 사용 전에 광범위한 전처리가 필요합니다. 둘째, ML 알고리즘의 강점인 "교차학습(cross-learning)"—여러 시계열로부터 공통 패턴 추출—은 부적절한 전처리로 인해 활용되지 못합니다. 셋째, 신경망은 정규화, 계절성 분리, 추세 추출 같은 시계열 특성 처리에 민감합니다.

### 주요 기여

**첫 번째 기여: 하이브리드 아키텍처**  
지수평활(Exponential Smoothing, ES) 모델의 업데이트 공식을 신경망 학습 과정에 **직접 내장**했습니다. 이는 단순한 순차 연결(pipeline)이 아니라, SGD 최적화 루프 내에서 ES 파라미터와 신경망 가중치가 **동시에** 조정되는 통합 구조입니다.[1]

**두 번째 기여: 계층적 파라미터 구조**  
모든 파라미터를 세 가지 범주로 분류했습니다:[1]
- **지역 상수**: 각 시계열의 평활 계수 및 계절성 성분 (개별 특성 반영)
- **지역 상태**: 수준, 계절성, RNN 숨김 상태 (시간에 따라 진화)
- **전역 상수**: 신경망 가중치 (모든 시계열의 패턴 학습)

이 구조를 통해 **개별화와 일반화의 이중성을 동시에 달성**합니다.

**세 번째 기여: 동적 계산 그래프 활용**  
PyTorch, DyNet 등의 동적 계산 그래프 시스템을 활용하여, 각 시계열마다 서로 다른 계산 그래프가 생성되도록 설계했습니다. 이를 통해 시계열 특이적 파라미터(개수, 초기값)와 공유 가중치를 통합적으로 처리할 수 있게 됩니다.[1]

**네 번째 기여: M4 경쟁에서 우승**  
100,000개의 다양한 시계열(금융, 거시경제, 산업, 은행, 관광, 에너지)에 대해 이전 최고 모델 대비 **10% 이상 오차 감소**, 예측 구간에서 **95% 신뢰도 달성 첫 번째 방법**을 제시했습니다.[1]

***

## 2. 해결하고자 하는 문제와 제안 방법

### 2.1 핵심 문제들

**문제 1: 전처리의 역설(Preprocessing Paradox)**

신경망은 학습을 위해 전처리를 필수로 요구하지만, 전처리의 품질이 최종 예측 정확도를 좌우합니다. 기존 접근은 전처리와 예측을 독립적으로 수행하여:[1]
- 계절성 분해의 오류가 시계열 끝부분에서 최악화 (예측 구간에서 가장 중요한 부분)
- 분해 알고리즘이 신경망 학습을 위해 최적화되지 않음
- 다양한 시계열의 패턴 다양성에 대응 어려움

**문제 2: 신경망의 크기 민감성(Scale Sensitivity)**

표준 신경망에서 가중치 업데이트 크기는 입력 신호의 크기에 비례합니다:[1]
$$\Delta w_{ij} \propto \text{Error} \times |x_i|$$

따라서:
- 작은 값의 입력 → 작은 그래디언트 → 느린 학습
- [0-1] 정규화는 시계열이 이 범위를 벗어날 때 무너짐
- 불균형한 시계열(100→110 vs. 100→200)이 정규화 후 동일하게 보임

**문제 3: 정규화의 도메인 특이성**

M4 데이터의 다양성은 고정된 정규화 방식을 불가능하게 합니다:[1]
- 금융 시계열: 극도로 변동성 높음 (0.01~999,999 범위)
- 거시경제: 추세가 강함
- 인더스트리: 계절성 다양함

### 2.2 제안 방법: 동적 적응적 전처리

**기본 원리**: 전처리 파라미터를 지수평활 업데이트 공식에서 추출하되, 이들 파라미터 자체를 **최적화 대상으로 포함**합니다.[1]

#### 지수평활 업데이트 공식 (on-the-fly preprocessing)

비계절성 모델:
$$l_t = \alpha y_t + (1-\alpha)l_{t-1} \quad (1)$$

단일 계절성 모델:
$$l_t = \alpha \frac{y_t}{s_t} + (1-\alpha)l_{t-1}$$
$$s_{t+K} = \beta \frac{y_t}{l_t} + (1-\beta)s_t \quad (2)$$

이중 계절성 모델:
$$l_t = \alpha \frac{y_t}{s_t u_t} + (1-\alpha)l_{t-1}$$
$$s_{t+K} = \beta \frac{y_t}{l_t u_t} + (1-\beta)s_t$$
$$u_{t+L} = \gamma \frac{y_t}{l_t s_t} + (1-\gamma)u_t \quad (3)$$

여기서:
- $\alpha, \beta, \gamma$: 평활 계수 (sigmoid로 로 제한)[1]
- $K$: 계절 주기 (월별 12, 분기별 4, 주별 52)
- $L$: 이중 계절 주기 (시간별 168)

**핵심 혁신**: 이 계수들과 초기 계절성 성분이 **지역 상수 파라미터**가 되어, 신경망 가중치와 함께 SGD로 최적화됩니다.[1]

#### 정규화 전략

각 시계열의 $t$번째 관측값에 대해:

1. **수준으로 정규화**: 입력 윈도우의 마지막 수준값 $l_t$로 나누기
2. **계절 성분으로 정규화**: 해당 계절 인덱스의 계절성 $s_t$ (및 필요시 $u_t$) 나누기
3. **로그 변환**: 이상치의 영향 감소

$$\text{정규화된 입력} = \log\left(\frac{y_t}{l_{t-h} \cdot s_{t-h \bmod K} \cdot u_{t-h \bmod L}}\right)$$

**장점**:[1]
- 모든 입력값이 약 1 근처로 정규화 (신경망 학습 최적화)
- 각 시계열마다 **적응적 정규화** (고정 범위 아님)
- 정규화 파라미터가 신경망 학습과 함께 진화
- 매 에포크마다 새로운 전처리로 **암묵적 데이터 증강**

### 2.3 신경망 아키텍처

#### 신경망 출력의 변환 공식

정규화된 입력에서 신경망 출력 $NN(x)$를 얻은 후, 원래 스케일로 역변환:

비계절성:

$$\hat{y}_{t+1..t+h} = \exp(NN(x)) \cdot l_t \quad (4)$$

단일 계절성:

$$\hat{y}_{t+1..t+h} = \exp(NN(x)) \cdot l_t \cdot s_{t+1:..t+h} \quad (5)$$

이중 계절성:

$$\hat{y}_{t+1..t+h} = \exp(NN(x)) \cdot l_t \cdot s_{t+1:..t+h} \cdot u_{t+1:..t+h} \quad (6)$$

이는 신경망이 **정규화된 공간에서의 배수(multiplicative factor)**를 학습함을 의미합니다.

#### Dilated LSTM 구조

신경망은 dilated LSTM 기반 스택으로 구성됩니다:[1]

- **표준 LSTM**: 시점 $t$에서 $h_{t-1}$ 사용
- **Dilated LSTM ($k$-dilated)**: 시점 $t$에서 $h_{t-k}$ 사용 ($k>1$)
  
$$h_t^{dilated} = \text{LSTM}\left(x_t, h_{t-d}\right)$$

**이점**: 장기 의존성(long-term dependencies) 캡처. 예를 들어, $d=12$인 dilated LSTM은 12 타임스텝 이전의 정보를 직접 활용하여 월간 패턴 학습.

구조 예시 (월별 데이터):
- Block 1: Dilated(1,3,6,12) [4개 계층]
- Block 2: Dilated(1,3,6,12) [4개 계층]
- Residual 연결: Block 1 출력 + Block 2 출력
- Adapter 계층: 최종 출력을 예측 기간 길이로 조정

***

## 3. 성능 향상 메커니즘과 한계

### 3.1 성능 향상 메커니즘

| 메커니즘 | 효과 | 증거 |
|---------|------|-----|
| **수준 평활성 페널티(LVP)** | 약 3% 성능 개선 | 논문에서 "이 없었다면 M4 우승 불가능"으로 강조[1] |
| **온-더-플라이 전처리** | 데이터 증강 효과 | 매 에포크마다 새로운 ES 파라미터로 다양한 표현 |
| **독립 실행 앙상블** | 6-9회 실행 평균화로 모델 불확실성 감소 | 검증 실험으로 6회 이상 수확체감 확인[1] |
| **전문가 앙상블** | 각 시계열에 최적 모델 할당 | M3에서 약 3% 개선 (다만 안정성 문제 언급)[1] |
| **에포크 앙상블** | 4-5개 최종 에포크 예측 평균화 | 과적합 방지 및 정규화 효과 |
| **Pinball 손실 함수** | 양의 편향 보정 | log 공간 학습 vs. 선형 공간 평가의 불일치 해소[1] |

#### 수준 평활성 페널티의 수식

신경망의 손실함수에 추가:
$$L_{total} = L_{forecast} + \lambda \cdot L_{smoothness}$$

여기서:

$$L_{smoothness} = \text{mean}\left((e_t)^2\right)$$

$$e_t = d_{t+1} - d_t, \quad d_t = \log(y_{t+1}/y_t)$$

$\lambda \in$ (계절성이 있는 경우), $\lambda = 0$ (연간 비계절성 데이터)[2]

**직관**: 로그 차분의 이계 차분이 작다 = 수준이 부드럽다 = 신경망이 가짜 계절성에 과적합하지 않음.[1]

### 3.2 한계와 제약

| 한계 | 영향 | 해결 시도 |
|------|------|---------|
| **일일/주간 성능 저하** | 월/분기/연간에 비해 정확도 낮음 | 논문 3.6절: 평활 계수 학습률 3배 증가로 개선[1] |
| **주파수별 다른 아키텍처 필요** | 도메인 특화로 일반화 성능 제약 | 각 주파수마다 다른 구조(Table 1)[1] |
| **하이퍼파라미터 민감성** | 13개 이상의 수동 튜닝 파라미터 | 백테스팅으로 최적값 찾음 (시간 소모)[1] |
| **양의 편향 완전 해결 불가** | Pinball loss로도 예측 구간에서 상한이 하한보다 적게 초과 | 기술적 한계 인정[1] |
| **단변량만 고려** | 다변량 시계열(상품 간 상호작용) 미처리 | M5 경쟁 방향으로 제시[1] |
| **외생 변수 미포함** | 도메인 지식(이벤트, 캘린더) 활용 불가 | 해석 가능성 강조를 위한 의도적 선택[1] |

***

## 4. 모델의 일반화 성능: 심층 분석

### 4.1 일반화 성능의 정의와 중요성

일반화 성능(Generalization Performance)는 학습 데이터에서의 성능이 새로운, 보이지 않은 데이터에서도 유지되는 정도를 의미합니다. 시계열 예측에서는 특히 중요합니다:[1]

$$\text{일반화 오류} = \text{테스트 오류} - \text{훈련 오류}$$

작을수록 모델이 특정 훈련 데이터에 과적합하지 않음을 의미합니다.

### 4.2 일반화 향상 메커니즘

#### 메커니즘 1: 계층적 구조의 정보 분할

ES-RNN의 핵심은 전역과 지역 정보의 명확한 분할입니다:[1]

$$\text{최종 예측} = f_{global}(\text{신경망 가중치}) + g_{local}(\text{ES 파라미터})$$

개념적으로:
- **전역 부분** ($f_{global}$): 100,000개 시계열에서 추출한 공통 비선형 패턴
  - 모든 시계열 학습에 참여 → 강력한 정규화 효과
  - 샘플 효율성 증대 (큰 데이터셋 활용)
  
- **지역 부분** ($g_{local}$): 각 시계열의 개별 특성 (수준, 계절성)
  - 최소 파라미터(시계열당 2-3개 평활 계수)
  - 과적합 위험 낮음

#### 메커니즘 2: 동적 전처리의 암묵적 정규화

매 에포크 $e$마다 ES 파라미터가 변함 → 같은 훈련 시계열도 다르게 표현됨:[1]

$$\text{입력}^{(e)} = \frac{y_t}{l_t^{(e)} \cdot s_t^{(e)}} \neq \frac{y_t}{l_t^{(e+1)} \cdot s_t^{(e+1)}} = \text{입력}^{(e+1)}$$

**정규화 이론**: 이는 **암묵적 데이터 증강(implicit augmentation)**으로 작용하여 경험적 리스크를 낮춥니다:

$$\text{Generalization Bound} \leq \text{Train Error} + \sqrt{\frac{\log(Model Complexity)}{N}}$$

더 많은 효과적인 훈련 데이터 → 모델 복잡도 대비 감소된 일반화 오류

#### 메커니즘 3: 앙상블의 분산 감소

분산 분해 정리(Bias-Variance Decomposition):
$$E[(y - \hat{y})^2] = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

독립적 $M$개 모델 앙상블: 

$$\text{Variance}\_{ensemble} = \frac{1}{M^2}\text{Variance}_{individual}$$

ES-RNN은 세 수준의 앙상블 적용:[1]
- **실행 레벨**: 6-9개 독립 실행
- **데이터 레벨**: 전문가 앙상블 (각 시계열마다 최고 2-5개 모델)
- **에포크 레벨**: 마지막 4-5개 에포크 평균화

결과적으로 분산이 매우 작음.

#### 메커니즘 4: 조기 종료의 정규화

검증 에러 기반 조기 종료:
$$\text{Optimal Epoch} = \arg\min_e \text{Validation Error}^{(e)}$$

이는 Rademacher 복잡도를 암묵적으로 제한하여 일반화 경계를 개선합니다. 표 1에서:[1]
- 월별: 10 에포크 (조기 종료)
- 분기별: 15 에포크
- 연간: 12 에포크

각 빈도별로 최적 포인트가 다르며, 이는 동적 데이터 특성을 반영합니다.

#### 메커니즘 5: 손실 함수 설계

Pinball 손실함수는 조건부 기댓값(quantile) 추정에 최적화:[1]

$$L_t(\tau) = \begin{cases}
(y_t - \hat{y}_t)\tau & y_t \geq \hat{y}_t \\
(\hat{y}_t - y_t)(1-\tau) & y_t > \hat{y}_t
\end{cases}$$

$\tau=0.475$는:
- 중앙값 근처 추정 (log 공간에서의 편향 보정)
- 평균 제곱 오류보다 **이상치에 강건함**
- 더 좋은 일반화 성질 (극단값 영향 감소)

### 4.3 일반화 성능의 실증적 검증

#### M4 데이터의 이질성

M4 데이터셋은 **극도로 이질적**이므로, 이 데이터에서의 우수한 성능은 높은 일반화 능력을 시사합니다:[1]

| 도메인 | 시계열 수 | 주파수 | 특성 |
|-------|----------|--------|------|
| 금융 | 6,519 | 월별, 분기별 | 극도 변동성, 추세 강함 |
| 거시경제 | 10,987 | 월별, 분기별 | 저주파, 부드러운 추세 |
| 산업 | 6,539 | 월별, 분기별 | 계절성 강함, 이상치 많음 |
| 은행 | 6,016 | 월별, 분기별 | 규모 다양함 |
| 관광 | 24,000 | 월별, 분기별 | 계절성 극단적 |
| 에너지 | 35,000+ | 일별, 시간별, 주별 | 고주파, 계절성 복합 |

이렇게 다양한 도메인에 **같은 아키텍처 이용** (주파수별 약간 조정)하면서도 우수한 성능 유지 = 강한 일반화.

#### 성능 수치

| 메트릭 | ES-RNN | 이전 최고(M4 2위) | 개선율 |
|--------|--------|----------|-------|
| **OWA (전체 가중 오류)** | 0.877 | 0.951 | 7.8% |
| **sMAPE** | 12.16% | 13.17% | 7.7% |
| **예측 구간 정확도** | 95% ± 0.5% | <90% | 첫 달성 |

특히 예측 구간에서 95% 신뢰도를 정확히 달성한 것은 **과도하게 낙관적이거나 보수적이지 않음**을 의미하여, 일반화 성능의 우수성을 입증합니다.[1]

***

## 5. 앞으로의 연구에 미치는 영향

### 5.1 직접적 영향

#### 1. 하이브리드 패러다임의 정당화

ES-RNN의 우승은 **"순수 신경망 vs. 통계 모델"의 이분법을 종료**하고 하이브리드 접근의 가치를 명확히 했습니다. 이후 M4의 상위 17개 방법 중 12개가 조합/하이브리드 형태를 채택했으며, 이는 산업 관행에도 즉각 반영되었습니다.[3][4][1]

#### 2. 동적 계산 그래프의 활용 촉진

ES-RNN이 PyTorch, DyNet 같은 DCG 시스템의 필수성을 실증적으로 보여주며, 이후 연구자들이 지역-전역 파라미터 결합 모델을 더 쉽게 구현할 수 있게 했습니다.[5][1]

#### 3. 전처리의 과학화

"전처리는 예술(art)"이라는 통념을 깨고, 전처리 파라미터를 **최적화 대상**으로 통합하는 방법론을 제시했습니다. 이후 많은 연구가 learnable normalization을 탐색하게 됩니다.[1]

### 5.2 직접 후속 연구

#### 2021년: ES-dRNN (Smyl et al.) - 자신의 개선[6]

ES-RNN을 확장하여 **다중 계절성을 더 유연하게 처리**하고, 동적 주의 메커니즘을 추가합니다. M5 경쟁(계층적 시계열)에 초점을 맞춰, ES-RNN의 한계인 "단변량 only"를 해결 시도합니다.[6]

#### 2020년: N-BEATS (Oreshkin et al., ICLR) - 직접 경쟁[7][8]

같은 년도에 발표되었으나 완전히 다른 철학을 제시합니다:
- **도메인-무관**: 통계 성분 제거, 순수 신경망
- **해석 가능성**: 기저 확장(basis expansion) 원리로 투명성 확보
- **성능**: M4에서 N-BEATS-G (일반) 3% 개선, M3/TOURISM에서 11% 개선[8]

ES-RNN vs. N-BEATS의 **철학적 대립**:
| 관점 | ES-RNN | N-BEATS |
|------|--------|---------|
| **접근** | 통계 + 신경망 조합 | 순수 신경망 (도메인-무관) |
| **강점** | 이론 기반, 계층 구조 명확 | 해석 가능, 자동화 가능 |
| **약점** | 도메인 특화 하이퍼파라미터 | 통계적 기초 부족 |
| **후속** | 더 복잡해짐 (ES-dRNN) | 더 깔끔해짐 (NBEATSx) |

### 5.3 광범위한 영향

#### Autoformer (2021) - 분해의 중요성 강화[9]

ES-RNN의 계절성-수준 분해 아이디어를 트랜스포머에 적용하여, **자기상관(autocorrelation) 메커니즘**을 도입합니다. 이는 저주파 주기 의존성을 주파수 도메인에서 포착합니다.[9]

#### NBEATSx (2021) - 외생변수 통합[10]

N-BEATS에 외생 변수를 추가하면서, ES-RNN의 한계를 직접 인정하고 해결합니다. 해석 가능성을 유지하면서 추세/계절 분리를 명시적으로 모델링합니다.[10]

#### Temporal Fusion Transformer (2021) - 변수 중요도 강조[11]

트랜스포머 기반이지만, ES-RNN의 "각 시점마다 주의"라는 아이디어를 시간 차원에 적용하여 변수별 동적 중요도를 추정합니다.

#### CATS (2024) - 자기주의 재검토[12]

최근 10년간의 트랜스포머 중심 발전이 시계열에 최적이 아님을 깨닫고, **자기주의 제거** 제안. 이는 ES-RNN의 직선적 구조(dilated RNN)의 가치를 간접적으로 재평가합니다.[12]

***

## 6. 2020년 이후 최신 연구 비교 분석

### 6.1 진화 경로

```
ES-RNN (2020) [M4 우승]
    ├─→ ES-dRNN (2021) [다중 계절성, 동적 주의]
    │   └─→ Contextually Enhanced ES-dRNN (2024) [부하 예측 특화]
    │
    ├─→ N-BEATS (2020) [도메인-무관 신경망]
    │   ├─→ NBEATSx (2021) [외생변수]
    │   ├─→ N-HiTS (2022) [계층적 보간]
    │   └─→ N-BEATS-MOE (2024) [혼합 전문가]
    │
    ├─→ Autoformer (2021) [자기상관 + 분해]
    │   └─→ FEDformer (2022) [푸리에 강화]
    │
    ├─→ Transformer 계열
    │   ├─→ Informer (2021) [효율적 주의]
    │   ├─→ PatchTST (2023) [패치 기반]
    │   └─→ iTransformer (2024) [역방향 사고]
    │
    └─→ 선형 모델 재발견
        ├─→ DLinear (2023) [간단한 선형]
        └─→ TiDE (2024) [시간 처리]
```

### 6.2 핵심 방법론들의 비교

#### 표: 2020-2025년 주요 시계열 예측 모델 비교

| 모델 | 발표 | 핵심 아이디어 | 성능(M4기준) | 일반화 | 계산 효율 |
|------|------|-------------|-----------|------|---------|
| **ES-RNN** | 2020 | 지수평활 + Dilated RNN | 최고 (기준) | 우수 | 중간 |
| **N-BEATS** | 2020 | Residual stacks + basis | +3% | 우수 | 높음 |
| **ES-dRNN** | 2021 | ES-RNN 확장 + 주의 | 비슷 | 우수 | 낮음 |
| **Autoformer** | 2021 | 분해 + 자기상관 주의 | +2~3% | 양호 | 낮음 |
| **Informer** | 2021 | 확률적 주의 | +2% | 양호 | 중간 |
| **TFT** | 2021 | 다변량 + 변수 중요도 | +1~2% | 우수 | 낮음 |
| **NBEATSx** | 2021 | N-BEATS + 외생변수 | +5% (외생변수 활용시) | 우수 | 높음 |
| **DLinear** | 2023 | 분해 + 선형 | -2% (M4) 그러나 간단함 | 우수 | 매우 높음 |
| **CATS** | 2024 | 교차주의만 사용 | 경쟁 수준 | 우수 | 매우 높음 |
| **N-BEATS-MOE** | 2024 | 혼합 전문가 | +3~5% | 우수 | 중간 |

#### 성능 해석:
- **기준**: M4 OWA 지표 (낮을수록 우수)
- **+X%**: ES-RNN 대비 오류 감소율
- 주의: 구현, 데이터 전처리, 하이퍼파라미터 최적화에 따라 실제 성능 차이 크게 달라짐[13]

### 6.3 주요 발견사항

#### 발견 1: 하이브리드 경향의 강화

**2020 이전**: 통계 vs. 신경망의 이분법 우세  
**2020-2021**: ES-RNN 우승으로 하이브리드 인정  
**2022-2024**: 거의 모든 상위 모델이 분해(decomposition) + 신경망 형태로 수렴

```
N-BEATS (순수)
  ↓
Autoformer = 분해(계절+추세) + Transformer 주의
Informer = 분해(암묵적) + 확률적 주의
TFT = 다변량 분해 + 트랜스포머
```

**결론**: "엔드-투-엔드 신경망이 모든 것을 학습할 수 있다"는 가설이 시계열에는 제한적임.[11][9][1]

#### 발견 2: 주의 메커니즘의 한계 인식

**2021년**: Transformer의 신기성과 성능에 집중  
**2023년**: Transformer 과다 사용 경고 (ICLR 논문)[14]
**2024년**: CATS로 자기주의 완전 제거 시도[12]

이는 **ES-RNN의 직선적 구조(dilated RNN)**의 가치를 간접 입증합니다. RNN의 "과거 → 현재" 방향성이 시계열의 인과성을 자연스럽게 반영합니다.[1]

#### 발견 3: 단순함의 부활 (2023년 이후)

**DLinear (2023)**: 분해 + 단순 선형[14]
- 대부분 M4 수준의 성능 달성
- 계산량 1/100 이하
- 해석 가능성 최고

**TiDE (2024)**: 시간 처리 최적화  
- 비선형 함수는 최소, 구조 최적화

**의미**: 신경망의 복잡성이 필수가 아닐 수 있음. ES-RNN이 강조한 **구조적 설계**의 중요성 재확인.

#### 발견 4: 외생변수와 메타데이터

**ES-RNN**: 의도적으로 제외 (해석 가능성)[1]
**NBEATSx**: 이후 추가[10]
**M5**: 외생변수 (프로모션, 휴일) 필수

**실무 결과**: 외생변수 없이 높은 성능은 **도메인-일반적 접근에는 부족**. 각 애플리케이션은 고유한 정보 요구.

***

## 7. 향후 연구의 방향성

### 7.1 이론적 공백 해결

#### 1. 일반화 경계의 수학화

현재: 경험적으로만 검증  
필요: 하이브리드 모델의 VC-dimension, Rademacher 복잡도 분석

$$\text{Test Error} \leq \text{Train Error} + \sqrt{\frac{\log(d)}{N}} + \delta$$

여기서 $d$는 지역 파라미터의 개수를 반영해야 함.

#### 2. 최적화 이론

현재: SGD가 지역-전역 파라미터를 어떻게 수렴시키는지 불명확  
필요: 다단계 최적화 이론 (multi-level optimization)

$$\min_{w_{global}} \left[ \min_{w_{local}} L(w_{global}, w_{local}) \right]$$

***

### 7.2 방법론적 발전

#### 1. 적응적 아키텍처 선택

**문제**: ES-RNN은 주파수별로 다른 구조 필요[1]
**해결**: 메타러닝(meta-learning) 활용

$$\theta^* = \text{MetaLearner}(\text{시계열 특성}) \rightarrow \text{아키텍처 파라미터}$$

#### 2. 다변량 확장

**진행 중**: ES-dRNN, M5 경쟁 후속들  
**미해결**: 변수 간 상호작용 모델링

$$y_t^{(i)} = f(y_{t-1}^{(1:d)}, s_t^{(i)}, l_t^{(i)}) + \varepsilon_t$$

#### 3. 불확실성 정량화 강화

**현재**: Pinball 손실로 예측 구간 생성[1]
**필요**: 하에족(heteroscedastic) 불확실성 모델링

$$\sigma_t^2 = g(x_t) \quad \text{(시점마다 다른 분산)}$$

***

### 7.3 실무 적용 과제

#### 1. 계산 효율성과 확장성

**ES-RNN**: CPU 기반, 병렬 처리[1]
**한계**: 100만 시계열 실시간 예측 어려움

**개선 방향**:
- 분산 학습(distributed training)
- 모델 압축(knowledge distillation)
- 온라인 학습(online learning)

#### 2. 설명 가능성 강화

**현재**: "평활 계수"는 해석 가능하나 신경망은 블랙박스[1]
**해결 방법**:
- SHAP, LIME을 이용한 신경망 성분 분해
- 주의 가중치 시각화
- 기여도 분석(contribution analysis)

#### 3. 도메인 적응(Domain Adaptation)

**문제**: 새로운 도메인/시장 진입 시 재학습 필요  
**해결**:
- Transfer learning with fine-tuning
- Few-shot learning
- Meta-learning for quick adaptation

***

### 7.4 데이터 관점

#### 1. 시계열 특성 자동 추출

**현재**: 수동 탐색  
**필요**: 자동 특성 추출 (tsfeatures, hctsa 확장)

$$\mathbf{z} = \text{FeatureExtractor}(\mathbf{y}) \quad \text{(예: 계절성 강도, 추세, 메모리)}$$

#### 2. 합성 데이터 및 증강

**활용**: 부족한 도메인에서 모델 사전학습
- 시간 왜곡(time warping)
- 회전, 스케일링 증강
- GAN/VAE로 합성 시계열 생성

***

## 결론

Smyl (2020)의 ES-RNN은 단순한 "경쟁 우승 모델"을 넘어, **시계열 예측 분야의 패러다임 전환**을 이루었습니다. 핵심 기여는:[1]

1. **하이브리드의 정당화**: 통계와 신경망의 보완적 역할 입증
2. **계층적 구조의 확립**: 전역 학습 + 지역 적응의 이중성 실현
3. **동적 전처리의 과학화**: 전처리를 최적화 프로세스에 통합
4. **일반화 성능 강조**: 100,000개 이질적 데이터에서 우수한 성능

이후 5년 간의 발전은 ES-RNN의 아이디어를 다양하게 발전시키되, **구조의 중요성, 분해의 가치, 도메인-일반적 설계의 어려움**을 반복 확인했습니다.[50-97][11][1]

앞으로의 과제는:
- **이론**: 왜 하이브리드가 잘 일반화하는가의 수학적 증명
- **방법**: 자동 아키텍처 선택, 다변량 확장, 불확실성 정량화
- **실무**: 계산 효율성, 설명 가능성, 도메인 적응

ES-RNN은 이 모든 과제의 출발점이자, 여전히 **강력한 기저(baseline)**로 기능하고 있습니다.

***

## 참고자료

[1] Smyl_2020.pdf https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b1dc6024-4a2c-4497-8e7c-56ad7cb4789b/Smyl_2020.pdf
[2] Analysis of neural networks for air quality forecasting in populated areas http://eeer.org/journal/view.php?doi=10.4491/eer.2025.400
[3] Late Meta-learning Fusion Using Representation Learning for Time Series
  Forecasting https://arxiv.org/pdf/2303.11000.pdf
[4] [D] Hybrid RNN model wins m4 forecasting competition https://www.reddit.com/r/MachineLearning/comments/8tajvh/d_hybrid_rnn_model_wins_m4_forecasting_competition/
[5] A Scalable Multivariate Model for Time Series Forecasting https://arxiv.org/pdf/2405.07117.pdf
[6] ES-dRNN: A Hybrid Exponential Smoothing and Dilated ... https://ieeexplore.ieee.org/document/10236525/
[7] N-BEATS: Neural basis expansion analysis for interpretable time series
  forecasting https://arxiv.org/pdf/1905.10437.pdf
[8] n-beats: neural basis expansion analysis for https://openreview.net/pdf?id=r1ecqn4YwB
[9] Yes, Transformers are Effective for Time Series Forecasting ... https://huggingface.co/blog/autoformer
[10] Neural basis expansion analysis with exogenous variables: Forecasting
  electricity prices with NBEATSx https://arxiv.org/pdf/2104.05522.pdf
[11] Time-series forecasting with deep learning: a survey https://royalsocietypublishing.org/rsta/article/379/2194/20200209/41189/Time-series-forecasting-with-deep-learning-a
[12] Are Self-Attentions Effective for Time Series Forecasting? https://proceedings.neurips.cc/paper_files/paper/2024/file/cf66f995883298c4db2f0dcba28fb211-Paper-Conference.pdf
[13] On the retraining frequency of global models in retail ... https://arxiv.org/html/2505.00356v3
[14] Deep learning in time series forecasting with transformer ... https://pmc.ncbi.nlm.nih.gov/articles/PMC12453695/
[15] A hybrid method of exponential smoothing and recurrent neural networks for time series forecasting https://linkinghub.elsevier.com/retrieve/pii/S0169207019301153
[16] Attention-Based SeriesNet: An Attention-Based Hybrid Neural Network Model for Conditional Time Series Forecasting https://www.mdpi.com/2078-2489/11/6/305
[17] A Study on Time Series Forecasting using Hybridization of Time Series Models and Neural Networks https://www.eurekaselect.com/172794/article
[18] A New Bootstrapped Hybrid Artificial Neural Network Approach for Time Series Forecasting https://link.springer.com/10.1007/s10614-020-10073-7
[19] A Hybrid Approach Integrating Multiple ICEEMDANs, WOA, and RVFL Networks for Economic and Financial Time Series Forecasting https://www.hindawi.com/journals/complexity/2020/9318308/
[20] Dual‐Branch Spectral‐Trend Attention Network With Gated Flux–Momentum Decomposition for Multiscale Financial Time‐Series Forecasting https://onlinelibrary.wiley.com/doi/10.1002/for.70116
[21] Toward a Digital Twin: Time Series Prediction Based on a Hybrid Ensemble Empirical Mode Decomposition and BO-LSTM Neural Networks https://asmedigitalcollection.asme.org/mechanicaldesign/article/doi/10.1115/1.4048414/1086969/Toward-a-Digital-Twin-Time-Series-Prediction-Based
[22] Hybrid vector autoregression–recurrent neural networks to forecast multivariate time series jet fuel transaction price https://iopscience.iop.org/article/10.1088/1757-899X/909/1/012079
[23] Machine Learning Advances for Time Series Forecasting https://onlinelibrary.wiley.com/doi/10.1111/joes.12429
[24] A Hybrid Model for Financial Time Series Forecasting—Integration of EWT, ARIMA With The Improved ABC Optimized ELM https://ieeexplore.ieee.org/document/9064778/
[25] Hybrid Variational Autoencoder for Time Series Forecasting https://arxiv.org/pdf/2303.07048.pdf
[26] Deep Factors for Forecasting https://arxiv.org/pdf/1905.12417.pdf
[27] Multi-Source Knowledge-Based Hybrid Neural Framework for Time Series
  Representation Learning https://arxiv.org/html/2408.12409
[28] A novel general-purpose hybrid model for time series forecasting https://pmc.ncbi.nlm.nih.gov/articles/PMC8178659/
[29] Deep Factors with Gaussian Processes for Forecasting https://arxiv.org/pdf/1812.00098.pdf
[30] A Statistics and Deep Learning Hybrid Method for Multivariate Time
  Series Forecasting and Mortality Modeling https://arxiv.org/pdf/2112.08618.pdf
[31] ProNet: Progressive Neural Network for Multi-Horizon Time Series Forecasting https://linkinghub.elsevier.com/retrieve/pii/S0020025524000252
[32] Analyzing and Forecasting Electricity Consumption in ... https://pdfs.semanticscholar.org/fb76/33dcdeb50f0abdd840d44e23e4afa44a2fde.pdf
[33] Forecasting Energy Consumption using Recurrent Neural ... https://www.arxiv.org/pdf/2601.17110.pdf
[34] Blockchain and Machine Learning for Fraud Detection https://arxiv.org/pdf/2210.12609.pdf
[35] The disutility of compartmental model forecasts during ... https://pdfs.semanticscholar.org/d0d4/2fa6fb4ab0650854f8f8080f7b7c8a4dd88a.pdf
[36] flowscope https://arxiv.org/pdf/2411.10716.pdf
[37] Self-determination theory and the influence of social ... https://pdfs.semanticscholar.org/5d14/0a9695d194bc7d79810984b4e2297cdef770.pdf
[38] Forecast of Electric Vehicle Sales in the World and China ... https://pdfs.semanticscholar.org/000c/efcc0a17a6252c7fe9d977d252bf712354a5.pdf
[39] Neural network-based stock index forecasting | PLOS One https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0322737
[40] Understanding the householder solar panel consumer https://arxiv.org/pdf/2304.11213.pdf
[41] Forecasting S&P 500 Using LSTM Models https://arxiv.org/html/2501.17366v1
[42] Novel model combining intrinsic and learned behaviours ... https://www.biorxiv.org/content/10.64898/2026.01.07.698166v1.full.pdf
[43] Enhanced Load Forecasting with GAT-LSTM https://arxiv.org/html/2502.08376v1
[44] A Review of Lithium-Ion Battery Capacity Estimation ... https://pdfs.semanticscholar.org/38e0/b3c5aee11894d110ed3d189d825daea26897.pdf
[45] ES-dRNN: A Hybrid Exponential Smoothing and Dilated ... https://arxiv.org/abs/2112.02663
[46] Using the TSA-LSTM two-stage model to predict cancer ... https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0317148
[47] (PDF) Hybrid Neural Networks for Time Series Forecasting https://www.academia.edu/87781131/Hybrid_Neural_Networks_for_Time_Series_Forecasting
[48] A dual attention LSTM lightweight model based on exponential smoothing for remaining useful life prediction https://www.sciencedirect.com/science/article/abs/pii/S0951832023007354
[49] A novel hybrid model to forecast seasonal and chaotic time series https://www.sciencedirect.com/science/article/abs/pii/S0957417423029639
[50] The improved integrated Exponential Smoothing based ... https://www.sciencedirect.com/science/article/pii/S2215016124003741
[51] M4 Competition - Forecasting https://www.unic.ac.cy/iff/research/forecasting/m-competitions/m4/
[52] A hybrid time series forecasting approach integrating fuzzy ... https://www.nature.com/articles/s41598-025-91123-8
[53] M4 Forecasting Competition: Introducing a New Hybrid ES-RNN Model https://www.uber.com/en-KR/blog/m4-forecasting-competition/
[54] Forecasting Implementation of Hybrid Time Series and ... https://www.sciencedirect.com/science/article/pii/S1877050924003673
[55] An intelligent system for forecasting time series based on a ... https://ceur-ws.org/Vol-3970/PAPER17.pdf
[56] Makridakis Competitions - Wikipedia https://en.wikipedia.org/wiki/Makridakis_Competitions
[57] Advances in Neural Networks for Time Series Forecasting https://australiansciencejournals.com/ml/article/download/3053/3381
[58] The M4 Competition: 100000 time series and 61 ... https://www.sciencedirect.com/science/article/pii/S0169207019301128
[59] Implementation of neural basis expansion analysis for interpretable time series generic (N-BEATS-G) based on cloud computing on Brent crude oil price data https://pubs.aip.org/aip/acp/article-lookup/doi/10.1063/5.0262875
[60] SYMH Index Prediction with Neural Basis Expansion Analysis for Time Series (N-BEATS) https://hdl.handle.net/2324/7395671
[61] Exploring the Power of Neural Basis Expansion Analysis for Time-Series Forecasting Neural Network for Deep Coalbed Methane Production Prediction https://onepetro.org/SJ/article/30/10/6236/787902/Exploring-the-Power-of-Neural-Basis-Expansion
[62] N-BEATS neural network applied for insulator fault prediction considering EMD methods https://ieeexplore.ieee.org/document/11277686/
[63] Fast and Accurate Solar Power Generation Forecasting Using Advanced Deep Learning: A Novel Neural Basis Expansion Analysis Framework https://ieeexplore.ieee.org/document/11237118/
[64] Prediction of Wind Turbine Blade Stiffness Degradation Based on Improved Neural Basis Expansion Analysis https://www.mdpi.com/2076-3417/15/4/1884
[65] ChatGPT-Assisted Deep Learning Models for Influenza-Like Illness Prediction in Mainland China: Time Series Analysis https://www.jmir.org/2025/1/e74423
[66] Bitcoin Price Prediction Using N-BEATs ML Technique https://publications.eai.eu/index.php/sis/article/view/9006
[67] SAR-Nbeats-based temperature interpretation and prediction of InSAR time-series deformations for bridges https://journals.sagepub.com/doi/10.1177/14759217251326057
[68] Neural forecasting at scale https://arxiv.org/pdf/2109.09705.pdf
[69] Enhanced N-BEATS for Mid-Term Electricity Demand Forecasting https://arxiv.org/pdf/2412.02722.pdf
[70] Feature-aligned N-BEATS with Sinkhorn divergence https://arxiv.org/pdf/2305.15196.pdf
[71] Optimization of battery charging and discharging strategies in substation DC systems using the dual self-attention network-N-BEATS model https://pmc.ncbi.nlm.nih.gov/articles/PMC11402101/
[72] Electricity price forecast based on the STL-TCN-NBEATS model https://pmc.ncbi.nlm.nih.gov/articles/PMC9938466/
[73] Forecasting Algorithms for Causal Inference with Panel Data https://arxiv.org/html/2208.03489v3
[74] arXiv:2402.06642v1 [q-fin.ST] 29 Jan 2024 https://arxiv.org/pdf/2402.06642.pdf
[75] arXiv:2501.14929v1 [cs.CV] 24 Jan 2025 https://arxiv.org/pdf/2501.14929.pdf
[76] Effective waste classification framework via enhanced deep ... https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0324294
[77] Hyperparameter Transfer with Mixture-of-Experts Layers https://arxiv.org/html/2601.20205v1
[78] From GARCH to Neural Network for Volatility Forecast https://arxiv.org/html/2402.06642
[79] arXiv:2401.01755v1 [cs.SD] 3 Jan 2024 https://arxiv.org/pdf/2401.01755.pdf
[80] Forecast-Then-Optimize Deep Learning Methods https://arxiv.org/pdf/2506.13036.pdf
[81] Surveying Techniques from Alignment to Reasoning https://arxiv.org/html/2503.06072v2
[82] Digital Twins for Intelligent Intersections: A Literature Review https://arxiv.org/html/2510.05374v1
[83] Do global forecasting models require frequent retraining? https://arxiv.org/pdf/2505.00356.pdf
[84] Effect of mediolateral leg perturbations on walking balance in ... https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0311727
[85] Automatic generation of DRI Statements https://web3.arxiv.org/pdf/2511.11655
[86] An Imputation-Based Mixup Augmentation Using Self- ... https://arxiv.org/pdf/2511.07930.pdf
[87] Evaluation of the best M4 competition methods for small ... https://www.sciencedirect.com/science/article/abs/pii/S0169207021001497
[88] N-BEATS-MOE: N-BEATS with a Mixture-of-Experts Layer ... https://arxiv.org/html/2508.07490v1
[89] Contextually enhanced ES-dRNN with dynamic attention ... https://www.sciencedirect.com/science/article/pii/S0893608023006408
[90] [논문리뷰]N-BEATS https://joungheekim.github.io/2020/09/09/paper-review/
[91] A Statistics and Deep Learning Hybrid Method for ... https://ideas.repec.org/a/gam/jforec/v4y2021i1p1-25d708917.html
[92] [ICLR 2020] N-BEATS : Neural Basis Expansion Analysis for Interpretable Time Sereis Forecasting https://velog.io/@sheoyonj/Paper-Review-N-BEATS-Neural-Basis-Expansion-Analysis-for-Interpretable-Time-Sereis-Forecasting
[93] Deep learning in time series forecasting with transformer models and RNNs https://peerj.com/articles/cs-3001/
[94] Online Data Augmentation for Forecasting with Deep ... https://arxiv.org/html/2404.16918v2
[95] Neural basis expansion analysis for interpretable time series forecasting https://liner.com/review/nbeats-neural-basis-expansion-analysis-for-interpretable-time-series-forecasting
[96] Transformer-based deep learning architecture for time ... https://www.sciencedirect.com/science/article/pii/S2665963824001040
[97] ES-dRNN: A Hybrid Exponential Smoothing and Dilated ... https://ieeexplore.ieee.org/iel7/5962385/6104215/10236525.pdf
