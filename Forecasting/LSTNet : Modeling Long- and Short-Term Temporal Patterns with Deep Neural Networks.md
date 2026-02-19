# Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks

### Executive Summary

LSTNet은 2018년 발표된 혁신적인 다변량 시계열 예측 프레임워크로, CNN을 통한 단기 국소 패턴 추출과 RNN을 통한 장기 의존성 포착을 결합한 구조이다. 특히 Recurrent-skip 메커니즘으로 주기적 패턴을 효율적으로 모델링하며, Autoregressive 성분을 병렬로 결합하여 비정상 시계열에서의 스케일 변화에 강건한 예측을 가능하게 한다. 2020년 이후 최신 연구들은 Transformer, MLP, 그리고 혼합 구조를 통해 모델의 효율성과 일반화 성능을 크게 향상시켰다.

***

### 1. 논문의 핵심 주장과 주요 기여

#### 1.1 해결하고자 하는 문제

다변량 시계열 예측은 교통, 에너지, 기상 등 다양한 실제 응용에서 중요하지만, 전형적인 시계열은 다층적인 시간 패턴을 포함한다. 예를 들어, 교통 점유율은 일일 주기(아침-저녁 피크)와 주간 주기(평일-주말)를 동시에 나타낸다. 전통적인 ARIMA 및 VAR(Vector Autoregression) 모델은 이러한 이질적인 패턴을 명시적으로 구분하여 모델링하지 못한다. 또한 Gaussian Process는 높은 계산 복잡도(샘플 수에 대해 입방 복잡도)로 대규모 다변량 문제에 적용하기 어렵다.[1]

**핵심 문제**: 단기 국소 의존성과 장기 주기 패턴을 동시에 포착하는 모델 설계

#### 1.2 제안 방법의 아키텍처

LSTNet은 다음과 같은 주요 컴포넌트로 구성된다:[1]

**① 합성곱 계층 (Convolutional Layer)**

$$h_k = \text{RELU}(W_k * X + b_k)$$

여기서 $W_k$는 크기가 $\omega \times n$ (ω: 필터 너비, n: 변량 수)인 필터이며, 연산을 통해 $d_c \times T$ 크기의 출력을 생성한다. 이 계층은 변량 간 국소 의존성과 시간 방향의 단기 패턴을 추출한다.[1]

**② 반복 계층 (Recurrent Component with GRU)**

$$r_t = \sigma(x_t W_{xr} + h_{t-1} W_{hr} + b_r)$$
$$u_t = \sigma(x_t W_{xu} + h_{t-1} W_{hu} + b_u)$$
$$c_t = \text{RELU}(x_t W_{xc} + r_t \odot (h_{t-1} W_{hc}) + b_c)$$
$$h_t = (1 - u_t) \odot h_{t-1} + u_t \odot c_t$$

GRU(Gated Recurrent Unit)를 사용하며, 은닉 상태 업데이트 활성화함수로 tanh 대신 RELU를 사용한다. 이는 더 안정적인 그래디언트 역전파를 가능하게 한다.[1]

**③ 주기적 스킵-RNN (Recurrent-skip Component)**

기존 GRU/LSTM의 주요 한계는 매우 긴 주기(예: 24시간)의 의존성을 포착하기 어렵다는 점이다. LSTNet은 주기적 성질을 활용한 혁신적인 구조를 제안한다:[1]

$$r_t = \sigma(x_t W_{xr} + h_{t-p} W_{hr} + b_r)$$
$$u_t = \sigma(x_t W_{xu} + h_{t-p} W_{hu} + b_u)$$
$$c_t = \text{RELU}(x_t W_{xc} + r_t \odot (h_{t-p} W_{hc}) + b_c)$$
$$h_t = (1 - u_t) \odot h_{t-p} + u_t \odot c_t$$

여기서 $p$는 주기 길이(예: 24)로, $h_{t-1}$ 대신 $h_{t-p}$를 사용하여 시간 스킵 연결을 형성한다. 이는 gradient vanishing을 완화하면서도 장기 주기 패턴을 효율적으로 모델링한다.[1]

**④ 시간 어텐션 계층 (Temporal Attention Layer)**

주기 길이 $p$가 미리 알려지지 않은 경우를 위해 동적 어텐션 가중치를 계산한다:[1]

$$\alpha_t = \text{AttnScore}(H_t^R, h_{t-1}^R)$$
$$h_t^D = W[c_t; h_{t-1}^R] + b$$

여기서 $c_t = H_t \alpha_t$는 가중 맥락 벡터이고, $H_t^R = [h_{t-q}^R, \ldots, h_{t-1}^R]$는 이전 $q$개 시간 스텝의 은닉 상태 행렬이다.

**⑤ 자동회귀 성분 (Autoregressive Component)**

신경망의 근본적인 한계는 입력 스케일에 대한 출력 불민감성이다. 실제 데이터에서 비주기적 스케일 변화(공휴일, 기상 이변)가 빈번하므로, 선형 AR 모델을 병렬로 추가한다:[1]

$$h_{t,i}^L = \sum_{k=0}^{q_{ar}-1} W_{k}^{ar} y_{t-k,i} + b_{ar}$$

최종 예측은:
$$\hat{Y}_t = h_t^D + h_t^L$$

이러한 분해(선형 + 비선형)는 highway network와 유사한 원리로, 비정상 스케일 변화에 강건한 예측을 제공한다.[1]

#### 1.3 학습 목적함수

기본적으로 제곱 오차를 사용:[1]

$$\min_{\Theta} \sum_{t \in \Omega_{Train}} ||Y_t - \hat{Y}_{t-h}||_F^2$$

Linear SVR의 성능이 우수한 경우를 위해 절대값 손실도 지원한다:[1]

$$\min_{\Theta} \sum_{t \in \Omega_{Train}} \sum_{i=0}^{n-1} |Y_{t,i} - \hat{Y}_{t-h,i}|$$

검증 세트를 통해 손실 함수를 선택하는 적응적 전략을 사용한다.

***

### 2. 모델 구조의 통합 설계

LSTNet의 모델 구조는 각 컴포넌트의 계층적 상호작용으로 설계되어 있다:[1]

| 계층 | 역할 | 입력 | 출력 | 목적 |
|------|------|------|------|------|
| CNN | 단기 국소 패턴 추출 | 원본 시계열 X | $d_c \times T$ | 인접 변량과 짧은 시간 윈도우 내 의존성 |
| GRU | 중기 의존성 | CNN 출력 | $h_t^R$ | 수십 시간의 컨텍스트 학습 |
| Skip-RNN | 장기 주기 패턴 | CNN 출력 | $h_t^S$ | 하루, 주간 주기의 명시적 모델링 |
| Attention | 동적 가중합 | GRU + Skip-RNN | 어텐션 가중치 | 가변적 주기 길이 대응 |
| Dense | 멀티스케일 통합 | 모든 RNN 출력 | $h_t^D$ | 다중 시간 스케일 조화 |
| AR | 스케일 정규화 | 원본 시계열 | $h_t^L$ | 비주기적 변화, 스케일 민감성 |

***

### 3. 성능 향상 및 한계 분석

#### 3.1 실험 결과

논문은 4개의 벤치마크 데이터셋에서 광범위한 평가를 수행했다:[1]

| 데이터셋 | 특성 | LSTNet 성능 | 주기성 |
|---------|------|-----------|--------|
| Traffic | 48개월, 862개 센서 | RSE: 0.4973 (h=24) | 강함(일일+주간) |
| Solar-Energy | 1년, 137 태양광 발전소 | RSE: 0.4643 (h=24) | 중간(일일) |
| Electricity | 3년, 321개 고객 | RSE: 0.1007 (h=24) | 강함(일일) |
| Exchange-Rate | 26년, 8개국 | RSE: 0.0449 (h=24) | 약함 |

**주요 발견**:
- LSTNet은 주기적 패턴이 명확한 데이터셋(Traffic, Electricity)에서 RNN-GRU 대비 9.2%-22.2% RSE 개선[1]
- 주기성이 없는 Exchange-Rate에서는 AR과 LRidge가 더 우수 (LSTNet의 과잉 모델링)
- 장기 예측(h=24)에서 성능 향상이 더 두드러짐

#### 3.2 절제 연구(Ablation Study)

컴포넌트별 기여도 분석:[1]

- **AR 제거 (LSTw/oAR)**: 대부분의 데이터셋에서 가장 큰 성능 저하 (RSE 증가 5-15%)
  - 증명 사례: Electricity 데이터에서 1000시간 시점의 갑작스러운 소비 증가 포착 실패[1]
  
- **Skip-RNN 제거 (LSTw/oskip)**: Solar-Energy와 Traffic에서 성능 저하 (단기 예측에서는 영향 미미)

- **CNN 제거 (LSTw/oCNN)**: 단기 예측(h=3)에서 성능 저하, 장기 예측에서는 상대적으로 영향 적음

**결론**: 세 컴포넌트의 조화가 견고한 성능의 핵심. AR 성분의 중요성이 특히 강조됨.[1]

#### 3.3 주요 한계

1. **주기 길이 하이퍼파라미터**: p값을 수동으로 조정 필요. 비주기적 데이터에서는 Attention 버전 필요[1]

2. **국제화 능력**: 한 데이터셋에 최적화된 모델이 다른 도메인으로의 전이 학습 성능이 제한적

3. **계산 복잡도**: 초장기 예측(h>100)에서는 여전히 계산 비용 증가

4. **비정상성 처리**: AR 성분에 의존하므로, 복잡한 비선형 드리프트는 여전히 챌린지

***

### 4. 모델의 일반화 성능 향상 가능성

#### 4.1 현재 LSTNet의 일반화 특성

**강점**:
- Autoregressive 성분이 서로 다른 스케일의 데이터에 적응
- 메모리 효율적인 구조로 장기 시계열에도 적용 가능
- 공개 코드 및 벤치마크로 재현성 보장[1]

**약점**:
- 고정 주기 가정 (교환율 같은 비주기 데이터에 약함)
- 다변량 간 복잡한 상호작용 모델링 부족
- 분포 시프트(domain shift)에 취약

#### 4.2 2020년 이후 연구의 개선 방향

**방향 1: 적응적 주기성 학습**
- **TimesNet (2023)**: FFT를 통해 데이터에서 주기를 자동 발견, 1D→2D 변환[2]
  - LSTNet 대비 계산 효율 30% 개선
  - 다중 주기 동시 모델링으로 일반화 성능 향상

**방향 2: 효율성 개선**
- **N-HiTS (2022)**: 계층적 보간과 다중 비율 샘플링[3]
  - Transformer 대비 50배 빠른 학습 속도
  - 장기 예측에서 25% 이상 정확도 개선
  - 메모리 선형 증가 (LSTNet은 이차적 증가)

**방향 3: 비정상성 처리 개선**
- **iTransformer (2024)**: 차원 반전으로 계층 정규화 효율성 증대[4]
  - 각 변량별 독립적 정규화로 드리프트 대응
  - 다양한 변량 수에 대한 강건한 성능
  
**방향 4: 교차 변량 모델링**
- **Crossformer, iTransformer**: 변량 간 상관성을 어텐션으로 직접 모델링
- **DAPNet (2024)**: Mixture-of-Experts로 주기성, 상관성, 시간 특성을 분리 학습[5]

**방향 5: 혼합 전문가 구조 (2024-2025)**
- **DeepVARMA (2024)**: LSTM으로 트렌드 제거 후 VARMA로 다변량 상호작용 모델링[6]
- **EffiCANet (2025)**: 분해 컨볼루션 + 그룹 컨볼루션 + 전역 어텐션[7]
  - 계산 오버헤드 감소 + 성능 향상

#### 4.3 LSTNet 기반 개선 전략

LSTNet을 기초로 한 일반화 성능 향상 방안:[1]

1. **주기 길이 자동 추정**
   - FFT 기반 주기 감지 추가
   - 동적 다중 p 값 앙상블
   
2. **강화된 AR 성분**
   - 계절성 ARIMA(SARIMA) 도입
   - 변량별 독립적 AR 파라미터
   
3. **어텐션 메커니즘 고도화**
   - 다중 헤드 셀프 어텐션 (Transformer 스타일)
   - 교차 변량 어텐션 추가
   
4. **정규화 전략 개선**
   - 변량별 정규화 (LayerNorm 적용)
   - 시간 가중 정규화 (비주기적 이상치 대응)

***

### 5. 2020년 이후 관련 최신 연구 비교 분석

#### 5.1 주요 방법론 비교

| 방법 | 발표연도 | 핵심 아이디어 | LSTNet 대비 장점 | 제한사항 |
|------|---------|-----------|-------------|---------|
| **N-BEATS** | 2020 | 스택-블록 MLP | 간결한 구조, 해석성 | 단변량 기반 |
| **Informer** | 2021 | Sparse attention | 계산 복잡도 O(L log L) | 주기성 미명시 |
| **Autoformer** | 2021 | 자동상관 + 분해 | 트렌드-계절 분리 | 분해 가정의 경직성 |
| **N-HiTS** | 2022 | 계층적 보간 | 장기 정확도 25% ↑, 50배 빠름 | 단변량 기반, 상관성 미모델 |
| **TimesNet** | 2023 | 1D→2D 변환 | 다중 주기 적응, 5개 태스크 SOTA | 교차 변량 미모델 |
| **PatchTST** | 2023 | Vision ViT 영감 | 채널 독립 패치 | 어텐션 오버헤드 |
| **iTransformer** | 2024 | 차원 반전 | 다변량 상관, 강건한 정규화 | 계산 비용 여전히 높음 |
| **KAN (2025)** | 2025 | Koopman + MLP | MSE 6.3% ↓, 해석성 향상 | 아직 초기 단계 |

#### 5.2 세부 비교: TimesNet vs LSTNet

**TimesNet의 혁신**:[8]
- 1D 시계열을 주기별로 2D 텐서로 변환
- 각 주기별 내부 변동(intraperiod)과 주기 간 변동(interperiod) 분리
- 2D inception 블록으로 멀티스케일 특징 추출

$$\text{Period} = \text{arg top-k}[\text{FFT}(X)]$$
$$H_{2D}^{(k)} \in \mathbb{R}^{P_k \times (T/P_k)}$$

**비교 결과**:
- LSTNet: 고정 p, Skip-RNN으로 주기 모델링
- TimesNet: 적응적 다중 주기, 2D 커널로 변동 추출
- TimesNet이 주기 발견 유연성에서 우월[2]

#### 5.3 세부 비교: iTransformer vs LSTNet

**iTransformer의 혁신**:[4]
- 전통적 Transformer: 시간 토큰 + 변량 차원
- iTransformer: 변량 토큰 + 시간 차원

$$\text{Traditional: } Q, K, V \in \mathbb{R}^{T \times d}$$
$$\text{iTransformer: } Q, K, V \in \mathbb{R}^{N \times d} \text{ (N = 변량수)}$$

**비교 결과**:
- LSTNet: 국소 CNN + 장기 RNN의 2단계 설계
- iTransformer: 멀티헤드 어텐션으로 모든 변량 상관 동시 포착
- iTransformer가 고차원 다변량에서 우월[9]

#### 5.4 최신 하이브리드 방법 (2024-2025)

**DAPNet (2024)**: Mixture-of-Experts 기반[5]
- 3개 전문가: 주기성 분석, 동적 상관, 하이브리드 시간 특징
- LSTNet의 고정 아키텍처 vs DAPNet의 동적 전문가 라우팅

**EffiCANet (2025)**: 효율적 컨볼루션 어텐션[7]
- 분해 컨볼루션 (depthwise + pointwise)
- 장기 의존성을 위한 대형 커널, 계산 효율성 동시 확보
- LSTNet의 fully-connected dense layer vs EffiCANet의 그룹 컨볼루션

**PeriodNet (2025)**: 기간 어텐션 메커니즘[10]
- 인접 기간 간 시간 유사성 포착
- 반복적 그룹화로 교차 변량 모델링 최적화
- LSTNet의 고정 skip length vs PeriodNet의 적응적 기간 그룹

***

### 6. 앞으로의 연구에 미치는 영향과 고려사항

#### 6.1 LSTNet의 학문적 영향

1. **신경망과 전통 방법의 융합**
   - Autoregressive 선형 성분의 병렬 결합이 이후 하이브리드 모델의 패러다임 제공
   - DeepVARMA, DeepStateSpace 등이 이 아이디어 상속[6]

2. **Recurrent-skip 구조의 확산**
   - LSTNet의 주기적 스킵 연결이 이후 모델의 설계 영감 제공
   - N-HiTS의 계층적 보간, DAPNet의 주기 모듈이 유사 정신 실현

3. **멀티태스크 학습 기초**
   - 단기(CNN) + 장기(RNN) + 주기(Skip-RNN)의 명시적 분리
   - 이후 분해 기반 방법(Autoformer, TimesNet)의 철학 선행

#### 6.2 현재 및 미래 연구 방향

**단기 (2025-2026)**:
1. **파운데이션 모델 확대**
   - TimesFM(Google, 2024), Chronos(Amazon, 2024)와 같은 대규모 사전학습 모델 발전
   - LSTNet 규모의 기초 모델들이 특정 도메인 파인튜닝 기반이 될 것으로 예상

2. **LLM과의 통합**
   - 시계열 데이터를 토큰화하여 LLM에 입력, 텍스트 맥락 활용[11]
   - 예: "일요일 오전은 교통량이 적다"와 같은 도메인 지식 통합

3. **온라인 학습 및 실시간 적응**
   - 스트리밍 데이터에 대한 효율적 갱신
   - LSTNet의 고정 아키텍처 vs 동적 적응 필요

**중기 (2026-2030)**:
1. **물리 정보 신경망 (Physics-Informed Neural Networks, PINN)**
   - 미분방정식 제약을 손실함수에 통합
   - 기후, 에너지 시스템 등 물리 기반 시계열에서 강건성 향상

2. **그래프 신경망 통합**
   - 공간-시간 동시 모델링 (교통망 구조, 전력망 위상)
   - LSTNet의 순수 시간 모델링 → 공간 그래프 통합 진화

3. **인과 추론**
   - 단순 상관성이 아닌 인과 관계 학습
   - 정책 개입의 효과 측정

#### 6.3 연구 시 핵심 고려사항

**1. 데이터 특성에 따른 모델 선택**
```
┌─────────────────────────────────────┐
│ 주기성 명확함 → TimesNet, PeriodNet │
│ 비정상성 강함 → iTransformer        │
│ 계산 제약 강함 → N-HiTS            │
│ 다변량 상관 중요 → iTransformer    │
│ 해석성 우선 → N-BEATS, LSTNet     │
└─────────────────────────────────────┘
```

**2. 하이퍼파라미터 튜닝 전략**
- LSTNet의 p(주기 길이): 자동 감지 (FFT) vs 수동 조정
- CNN 필터 크기: 5-7 권장, 너무 크면 국소성 손상
- skip-RNN 은닉 차원: 50-200, 데이터셋 규모에 따라 조정

**3. 벤치마크 및 평가**
- 공개 벤치마크(ETT, Electricity, Traffic) 사용으로 비교 용이
- 최신 baseline과 비교 필수 (TimesNet, iTransformer 포함)
- 계산 복잡도, 메모리 사용량도 평가 대상

**4. 일반화 성능 검증**
- 훈련-검증-테스트 시간순 분할 (look-ahead bias 방지)
- 도메인 외(out-of-distribution) 성능 평가
- 분포 시프트 시뮬레이션 (스케일 변화, 이상치)

**5. 재현성 및 공개성**
- 코드 공개(GitHub) 필수
- 벤치마크 데이터셋 명시
- 하이퍼파라미터 상세 기록

***

### 7. 종합 평가

#### LSTNet의 역할

LSTNet은 2018년 기준 다변량 시계열 예측의 획기적 진전이었으나, 2024-2025년 기준으로는 다음과 같이 평가된다:

**계속된 가치**:
- 신경망+통계 모델 하이브리드의 패러다임 영향력 (후속 모델들이 상속)
- 해석 가능성 우수 (CNN→RNN→AR의 명확한 역할 분리)
- 공개된 코드와 벤치마크로 재현성 확보

**현재의 제약**:
- 고정 주기 가정 (TimesNet, PeriodNet에 의해 동적 발견으로 개선)
- 교차 변량 모델링 부족 (iTransformer, Crossformer에 의해 개선)
- 계산 효율성 (N-HiTS가 50배 빠름)

#### 추천 활용

1. **레거시 시스템 유지보수**: 기존 LSTNet 배포 시스템은 안정성 있음
2. **특정 도메인 (주기성 강함)**: LSTNet 여전히 경쟁력 (미세 조정으로 충분)
3. **새로운 프로젝트**: TimesNet (유연성) 또는 iTransformer (성능) 권장
4. **리소스 제약**: N-HiTS (효율성)
5. **연구 개발**: 최신 혼합 모델 (DAPNet, EffiCANet) 탐색

***

### 참고문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/52ca85e6-caf5-47c0-9eaf-d9d751b95bf4/1703.07015v3.pdf)
[2](https://wandb.ai/tensorgirl/timesnet/reports/TimesNet-Revolutionizing-Time-Series-Analysis-with-2D-Tensor-Transformation--Vmlldzo2MjQyMTMw)
[3](https://arxiv.org/pdf/2201.12886.pdf)
[4](https://openreview.net/pdf?id=JePfAI8fah)
[5](https://arxiv.org/html/2509.11601v1/)
[6](https://arxiv.org/pdf/2404.17615.pdf)
[7](https://arxiv.org/html/2411.04669v1)
[8](https://arxiv.org/pdf/2210.02186.pdf)
[9](https://openreview.net/forum?id=JePfAI8fah)
[10](https://arxiv.org/pdf/2511.19497.pdf)
[11](https://arxiv.org/pdf/2508.19279.pdf)
[12](https://pmc.ncbi.nlm.nih.gov/articles/PMC12453695/)
[13](https://yanglin1997.github.io/files/TCAN.pdf)
[14](https://arxiv.org/abs/2201.12886)
[15](https://www.sciepublish.com/article/pii/704)
[16](https://ecsenet.com/index.php/2576-6821/article/download/716/284)
[17](https://milvus.io/ai-quick-reference/how-do-attention-mechanisms-enhance-time-series-forecasting-models)
[18](https://pmc.ncbi.nlm.nih.gov/articles/PMC12222760/)
[19](https://towardsdatascience.com/temporal-graph-learning-in-2024-feaa9371b8e2/)
[20](https://www.sciencedirect.com/science/article/abs/pii/S0360835223006915)
[21](https://arxiv.org/html/2502.10721v1)
[22](https://www.sciencedirect.com/science/article/pii/S2590123025002464)
[23](https://pmc.ncbi.nlm.nih.gov/articles/PMC12251672/)
[24](https://www.ijcai.org/proceedings/2025/1187.pdf)
[25](https://arxiv.org/pdf/2406.08627.pdf)
[26](https://www.arxiv.org/pdf/2510.06466.pdf)
[27](https://www.arxiv.org/pdf/2512.03114.pdf)
[28](https://arxiv.org/html/2506.14831v2)
[29](https://www.biorxiv.org/content/10.1101/2025.08.10.669034v1.full.pdf)
[30](https://arxiv.org/pdf/2508.12162.pdf)
[31](https://arxiv.org/html/2502.17495v1)
[32](https://pmc.ncbi.nlm.nih.gov/articles/PMC7336835/)
[33](https://pmc.ncbi.nlm.nih.gov/articles/PMC11059412/)
[34](https://www.nature.com/articles/s41598-025-07654-7)
[35](https://www.sciencedirect.com/science/article/pii/S2665963824001040)
[36](https://www.youtube.com/watch?v=TSGZBXILk14)
[37](https://arxiv.org/abs/2511.09783)
[38](https://arxiv.org/html/2411.05793v1)
[39](https://www.sciencedirect.com/science/article/abs/pii/S1568494622009942)
[40](https://openreview.net/pdf?id=w7vn6ah0Qg)
[41](https://openreview.net/forum?id=kHEVCfES4Q&noteId=mrNbq9EkQa)
[42](https://arxiv.org/html/2509.26468v1)
[43](https://arxiv.org/html/2512.07705v1)
[44](https://web3.arxiv.org/list/cs.LG/2024-03?skip=450&show=2000)
[45](https://arxiv.org/html/2505.19432v2)
[46](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0320368)
[47](https://arxiv.org/html/2412.05579v2)
[48](https://arxiv.org/html/2503.06928v1)
[49](https://arxiv.org/pdf/2507.23276.pdf)
[50](https://arxiv.org/pdf/2509.26468.pdf)
[51](https://www.reddit.com/r/MachineLearning/comments/18ax51t/d_transformers_for_time_series_forecasting/)
[52](https://www.semanticscholar.org/paper/N-HiTS:-Neural-Hierarchical-Interpolation-for-Time-Challu-Olivares/3711a745537f2f2e139ec759d36f31946fa549fd)
[53](https://ise.thss.tsinghua.edu.cn/~mlong/doc/TimesNet-iclr23.pdf)
[54](https://www.youtube.com/watch?v=MU-uAYmd24s)
[55](https://arxiv.org/abs/2310.06625)
[56](https://arxiv.org/abs/2210.02186)
[57](https://www.datasciencewithmarco.com/blog/itransformer-the-latest-breakthrough-in-time-series-forecasting)
[58](https://calmmimiforest.tistory.com/118)
[59](https://arxiv.org/pdf/2507.11439.pdf)
[60](https://arxiv.org/pdf/2409.00480.pdf)
[61](https://arxiv.org/pdf/2311.00214.pdf)
[62](https://arxiv.org/html/2310.06625v4)
[63](https://summarizepaper.com/en/arxiv-id/2201.12886v1/)
[64](https://trivia-starage.tistory.com/96)
[65](https://yetsyl0705.tistory.com/49)
[66](https://ai-onespoon.tistory.com/entry/TimesNet-Temporal-2D-Variation-Modeling-For-General-Time-Series-Analysis-ICLR-2023)
[67](https://www.youtube.com/watch?v=eXiEWzeR8qo)
