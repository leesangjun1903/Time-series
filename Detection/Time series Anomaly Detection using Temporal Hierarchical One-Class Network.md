
# Timeseries Anomaly Detection using Temporal Hierarchical One-Class Network

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 논문의 중심 주장

"Temporal Hierarchical One-Class Network (THOC)"는 실제 세계의 복잡한 시계열 데이터에서 다중 스케일 시간적 역학을 효과적으로 포착하여 이상치를 탐지할 수 있는 심층 학습 기반의 원클래스 분류 모델이다. 이 논문의 핵심 주장은 기존의 단일 초구면(hypersphere)을 사용하는 Deep SVDD의 제한성을 극복하고, 시간적 계층 구조를 통해 다양한 해상도(resolution)에서의 정상 행동을 더욱 정확하게 모델링할 수 있다는 것이다.[1]

### 1.2 주요 기여

THOC의 주요 기여는 다섯 가지로 정리된다:

| 기여 항목 | 내용 | 영향 |
|---------|------|------|
| **다중 스케일 특성 추출** | 확장(dilated) RNN과 스킵 연결을 통해 여러 시간 스케일에서 특성 추출 | 장기 의존성 포착 개선 |
| **계층적 특성 융합** | 모든 중간 레이어의 특성을 확률적 계층 클러스터링으로 통합 | 세밀한 시간 정보 보존 |
| **다중 초구면 설계** | 각 해상도에서 여러 초구면으로 정상 행동 표현 | 복잡한 시간적 역학 포착 |
| **직교성 제약** | 초구면 중심의 다양성을 장려하는 손실함수 추가 | 중복성 감소 및 표현력 증가 |
| **자기 감독 학습** | 시간적 영역의 다단계 예측 손실 추가 | 표현 학습 향상 |

***

## 2. 문제 정의 및 동기

### 2.1 해결하고자 하는 문제

시계열 이상치 탐지는 세 가지 근본적인 어려움에 직면한다:[1]

1. **고도의 비선형 시간 의존성**: 실제 시스템(발전소, 데이터센터, 스마트팩토리)의 센서 데이터는 복잡한 상호작용과 비선형 시간 동학을 포함한다.

2. **이상치 희소성**: 이상 샘플이 드물어서 레이블링이 비용이 크고 시간이 소요된다. 따라서 비지도 학습 설정이 필수적이다.

3. **일반화 성능**: 훈련 데이터에 없던 새로운 유형의 이상치를 탐지해야 한다.

### 2.2 기존 방법의 한계

THOC 이전의 주요 접근법들은 다음과 같은 제한성을 보였다:[1]

- **전통적 원클래스 분류 (OC-SVM, SVDD)**: 고정 차원 입력에만 적용 가능하며, 시간 의존성을 충분히 포착하지 못함
- **LSTM 기반 인코더-디코더**: 긴 수열의 디코딩 시 오차 누적 문제
- **GAN 기반 모델**: 훈련 불안정성 및 생성기-판별기 간 균형 문제

***

## 3. 제안하는 방법론 및 수식

### 3.1 다중 스케일 시간 특성 추출

THOC의 첫 번째 핵심 구성요소는 **확장 RNN(Dilated RNN)**이다. L개 레이어로 구성된 구조에서 시간 t의 각 레이어 l에 대한 숨겨진 상태는 다음과 같이 정의된다:

$$f^l_t = \begin{cases}
F_{RNN}(x_t, f^l_{t-s(l)}) & \text{if } l = 1 \\
F_{RNN}(f^l_{t}, f^l_{t-s(l)}) & \text{otherwise}
\end{cases}$$

여기서 $F_{RNN}$은 LSTM이나 GRU와 같은 RNN 셀, $s(l)$은 지수적으로 증가하는 스킵 길이로 $s(l) = M_0 \prod^{l-1}_{i=1} M$이다 (일반적으로 $M_0=1, M=2$). 이 구조는 하위 레이어에서 단기 정보를, 상위 레이어에서 장기 정보를 포착한다.[1]

### 3.2 계층적 특성 융합 메커니즘

THOC의 가장 혁신적인 부분은 **차별화 가능한 계층적 클러스터링 절차**이다. 각 스케일 $l \in \{1, \ldots, L\}$에서 $K_l$개의 클러스터 중심 $\{c^l_1, \ldots, c^l_{K_l}\}$이 존재한다.

**Step 1 (할당)**: 이전 레이어의 출력 $\bar{f}^{l-1}_{t,i}$를 현재 레이어의 중심으로 확률적으로 할당한다:

$$P^l_{t,i \to j} = P(\bar{f}^{l-1}_{t,i} \to c^l_j) = \frac{\exp(\text{score}(\bar{f}^{l-1}_{t,i}, c^l_j) / \tau)}{\sum^{K_l}_{k=1} \exp(\text{score}(\bar{f}^{l-1}_{t,i}, c^l_k) / \tau)}$$

여기서 $\tau$는 온도 매개변수이고, 유사성 함수는 코사인 유사도를 사용한다:[1]

$$\text{score}(\bar{f}, c) = \frac{\bar{f}^\top c}{\|\bar{f}\| \cdot \|c\|}$$

**Step 2 (업데이트)**: 할당 확률에 따라 특성을 각 클러스터에서 변환한다:

$$\hat{f}^l_{t,j} = \sum^{K_{l-1}}_{i=1} P^l_{t,i \to j} \text{ReLU}(W^l \bar{f}^{l-1}_{t,i} + b^l), \quad j = 1, \ldots, K_l$$

최종 특성은 다음과 같이 정의된다:

$$\bar{f}^l_{t,j} = \begin{cases}
f^1_t & \text{if } l = 0 \\
F_{MLP}([\hat{f}^l_{t,j}; f^{l+1}_t]) & \text{if } 1 \leq l \leq L-1 \\
\hat{f}^L_{t,j} & \text{if } l = L
\end{cases}$$

여기서 $[\cdot; \cdot]$는 연결(concatenation)이고, $F_{MLP}$는 완전연결 레이어이다.[1]

### 3.3 다중 스케일 벡터 데이터 기술(MVDD)

THOC의 훈련 목적 함수는 세 개의 손실 항으로 구성된다. 먼저 최상위 레이어에서의 주요 손실은:

$$L_{THOC} = \frac{1}{NK_L} \sum^N_{s=1} \frac{1}{T_s} \sum^{T_s}_{t=1} \sum^{K_L}_{j=1} R^L_{t,j,s} d(\bar{f}^L_{t,j,s}, c^L_j) + \lambda \Omega(W)$$

여기서 $R^L_{t,j,s}$는 현재 관측이 중심 $c^L_j$와 연관된 정도를 나타내는 재귀적으로 계산된 값이다:[1]

$$R^l_{t,j} = \frac{\exp(\tilde{R}^l_{t,j})}{\sum^{K_l}_{i=1} \exp(\tilde{R}^l_{t,i})}$$

여기서

$$\tilde{R}^l_{t,j} = \begin{cases}
\sum P^1_{t,i \to j} & \text{if } l = 1 \\
\sum^{K_{l-1}}_{i=1} P^l_{t,i \to j} R^{l-1}_{t,i} & \text{if } 1 < l \leq L
\end{cases}$$

### 3.4 직교성 제약 손실

초구면 중심의 다양성을 장려하기 위해 다음 손실을 추가한다:[1]

$$L_{orth} = \frac{1}{L} \sum^L_{l=1} \|(C^l)^\top C^l - I\|^2_F$$

여기서 $C^l = [c^l_1 \cdots c^l_{K_l}]$이고, $\|\cdot\|_F$는 Frobenius 노름이다.

### 3.5 시간적 자기 감독 손실

모든 레이어에서 유용한 특성 학습을 촉진하기 위해 다단계 예측 작업을 추가한다:[1]

$$L_{TSS} = \frac{1}{NL} \sum^N_{s=1} \sum^L_{l=1} \left[ \frac{1}{T_s - s(l)} \sum^{T_s}_{t=s(l)+1} \|W^l_{pred} f^l_{t-s(l),s} - x_{t,s}\|^2 \right]$$

### 3.6 최종 목적 함수

전체 손실은 다음과 같이 정의된다:[1]

$$L_{total} = L_{THOC} + \lambda_{orth} L_{orth} + \lambda_{TSS} L_{TSS}$$

***

## 4. 모델 구조 상세 설명

### 4.1 아키텍처 개요

THOC는 두 개의 주요 구성 부분으로 구성된다:[1]

1. **좌측 (특성 추출)**: L층 확장 RNN으로 다중 스케일 시간 특성 추출
2. **우측 (계층적 융합)**: 모든 중간 레이어의 특성을 계층적으로 처리하여 최상단에서 이상 점수 출력

### 4.2 알고리즘 1: 시간적 계층 원클래스 학습

```
입력: 시계열 X_s = (x_{1,s}, x_{2,s}, ..., x_{T_s,s}); 클러스터 수 {K_l}; 스킵 길이 {s(l)}
반복:
  1. x_{t,s}를 L층 확장 RNN에 입력하여 각 레이어로부터 {f^l_t} 획득
  2. l = 1부터 L까지:
     3. l번째 클러스터링 레이어의 입력 {f̄^{l-1}_{t,i}}를 (8)으로부터 획득 (K_0=1)
     4. 식 (5)에서 확률 {P^l_{t,i→j}} 계산
     5. 식 (10)에서 각 클러스터 중심에 대해 {R^l_{t,j}} 계산
     6. 식 (7)에서 출력 특성 {f̂^l_{t,j}} 업데이트 및 획득
  7. Adam 최적화기로 식 (13)의 MVDD 목적함수 최소화
수렴까지 반복
```

### 4.3 이상 탐지 추론

훈련된 모델을 사용하여 미래 시계열 X의 시간 t에서의 이상 점수는:[1]

$$\text{AnomalyScore}(x_t) = \sum^{K_L}_{j=1} R^L_{t,j} \cdot d(\bar{f}^L_t, c^L_j)$$

미리 정의된 임계값 $\delta$에 대해, $\text{AnomalyScore}(x_t) > \delta$이면 이상으로 분류한다.

***

## 5. 성능 향상 분석

### 5.1 벤치마크 데이터셋 및 실험 설정

THOC는 6개의 실제 시계열 데이터셋에서 평가되었다:[1]

| 데이터셋 | 차원 | 길이 | 특성 |
|---------|------|------|------|
| 2D-gesture | 2 | 80 | 비디오 손짓 |
| power-demand | 1 | 80 | 네덜란드 시설 전력수요 |
| KDD-Cup99 | 34 | 100 | 네트워크 트래픽 |
| SWaT | 51 | 100 | 물 처리 설비 |
| MSL | 55 | 100 | NASA 화성 로버 |
| SMAP | 25 | 100 | NASA 위성 |

### 5.2 성능 비교 결과

THOC는 모든 데이터셋에서 경쟁 방법들을 능가했다:[1]

| 방법 | 2D-gesture | power-demand | KDD-Cup99 | SWaT | 평균 순위 |
|------|-----------|-------------|----------|------|---------|
| LOF | 42.18(8) | 19.81(9) | 97.42(11) | 86.36(7) | 8.75 |
| OC-SVM | 36.78(14) | 20.58(8) | 97.53(10) | 75.98(13) | 11.25 |
| Deep SVDD | 37.32(13) | 19.54(10) | 94.64(14) | 82.82(11) | 12 |
| DAGMM | 38.91(12) | 37.69(4) | 97.86(8) | 85.38(8) | 8.0 |
| EncDec-AD | 39.85(11) | 22.22(6) | 94.37(13) | 75.56(14) | 11 |
| **THOC** | **63.31(1)** | **45.68(1)** | **98.86(1)** | **88.09(1)** | **1.0** |

특히 MSL 데이터셋에서 THOC는 93.67%의 F1 점수를 달성했고, SMAP에서는 95.18%를 기록했다.[1]

### 5.3 절제 연구(Ablation Study)

THOC의 각 구성 요소의 효과를 분석한 결과:[1]

| 모델 변형 | L_orth | L_TSS | Precision | Recall | F1 |
|---------|--------|--------|-----------|--------|-----|
| 기본 (둘 다 제거) | ✗ | ✗ | 52.22 | 24.77 | 33.60 |
| 직교성만 | ✓ | ✗ | 34.00 | 67.29 | 45.17 |
| 시간 감독만 | ✗ | ✓ | 42.08 | 57.71 | 48.67 |
| **전체 THOC** | **✓** | **✓** | **54.78** | **75.00** | **63.31** |

이는 각 손실 항이 중요한 역할을 함을 보여준다. 특히 시간적 자기 감독이 재현율 향상에 결정적이다.

### 5.4 계층 vs 평면 구조 비교

THOC의 계층적 특성 융합이 평면(flat) 구조보다 우월함을 보였다:[1]

| 방법 | Precision | Recall | F1 |
|-----|-----------|--------|-----|
| RNN-top (평면, 1 초구면) | 31.67 | 75.70 | 44.66 |
| RNN-top + 다중 초구면 (평면) | 41.32 | 68.93 | 51.66 |
| THOC-variant (계층, 1 초구면) | 53.27 | 60.98 | 56.86 |
| **THOC** (계층, 다중 초구면) | **54.78** | **75.00** | **63.31** |

이 비교는 다중 해상도 특성과 다중 초구면이 모두 필수적임을 입증한다.

***

## 6. 모델의 일반화 성능 향상 가능성

### 6.1 일반화 성능의 현재 상태

THOC는 여러 측면에서 강력한 일반화 능력을 보인다:[1]

1. **다양한 도메인에 대한 일관된 성능**: 6개의 서로 다른 도메인(제스처, 전력, 네트워크, 물 처리, 우주 센서)에서 우수한 성능을 유지했다.

2. **데이터 특성에 대한 강건성**: 데이터셋 크기, 차원성, 시간적 의존성 구조가 다름에도 불구하고 일관된 성능을 달성했다.

3. **비지도 학습 설정**: 레이블이 없는 훈련 데이터로도 효과적으로 학습할 수 있다.

### 6.2 일반화 개선을 위한 제안 방향

#### 6.2.1 계층적 표현의 이점

THOC의 다중 해상도 설계는 시계열의 내재된 계층적 특성을 활용한다. 이는 다음과 같은 개선 가능성을 시사한다:

- **적응형 해상도 선택**: 데이터의 고유한 시간 스케일에 맞추어 $K_l$과 $s(l)$을 동적으로 조정
- **메타러닝 접근**: 다양한 도메인의 시계열에서 최적의 계층 구조를 학습하는 메타러닝 프레임워크 개발
- **도메인 간 전이 학습**: 사전 훈련된 THOC 모델을 새로운 도메인으로 효율적으로 전이

#### 6.2.2 자기 감독의 강화

현재 THOC는 다단계 예측만을 자기 감독 작업으로 사용한다. 다음 방향들이 탐색될 수 있다:

- **대조 학습 통합**: 정상 패턴들 간의 유사성과 이상 패턴과의 차이를 명시적으로 모델링
- **마스킹 전략**: 입력의 일부를 마스킹하고 재구성하는 방식의 자기 감독 추가
- **보조 작업 다양화**: 노이즈 제거, 회전 예측, 스케일 인식 예측 등 다중 보조 작업

#### 6.2.3 원클래스 분류의 개선

원클래스 설정에서의 일반화 문제 해결:

- **오염 허용(contamination-robust)** 학습: 훈련 데이터에 소수의 이상이 포함되어 있을 수 있는 현실을 고려
- **불확실성 모델링**: Bayesian 접근으로 예측의 신뢰도를 정량화
- **적응형 임계값 설정**: 테스트 분포의 변화에 대응하는 동적 임계값 조정

### 6.3 데이터 특성에 따른 성능 변화

THOC의 성능은 데이터의 특성에 따라 달라진다:[1]

- **훈련 데이터 크기**: 더 큰 훈련 셋 (SMAP: 94,528 샘플)에서 MSL보다 높은 성능 (95.18% vs 93.67%)
- **이상의 다양성**: SWaT 데이터셋(36개의 다양한 공격)에서는 상대적으로 낮은 성능 제시

이는 다음을 시사한다:
- 이상의 유형이 많고 다양할수록 더 강력한 표현 능력이 필요
- 도메인 특화 사전지식의 통합이 도움될 수 있음

***

## 7. 모델의 한계

### 7.1 구조적 한계

1. **고정된 윈도우 크기**: THOC는 고정 길이 윈도우를 가정하며, 가변 길이 시계열에 대한 확장이 자명하지 않다.

2. **클러스터 수 선택**: 각 레이어의 클러스터 수 $K_l$을 사전에 지정해야 하며, 자동 선택 메커니즘이 없다. 이는 하이퍼매개변수 튜닝 부담을 증가시킨다.

3. **계산 복잡도**: 다중 레이어, 다중 클러스터의 구조로 인해 계산 비용이 상당할 수 있으며, 실시간 탐지가 필요한 애플리케이션에서 문제가 될 수 있다.

### 7.2 훈련 관련 한계

1. **하이퍼매개변수 민감도**: 온도 매개변수 $\tau$, 손실 가중치 $\lambda_{orth}$, $\lambda_{TSS}$의 설정이 성능에 중대한 영향을 미친다. 논문에서는 검증 데이터의 F1 점수로 선택하는 방법만 제시했다.

2. **초기화 의존성**: 클러스터 중심의 초기 설정이 수렴 성능에 영향을 미칠 수 있다.

3. **오염된 훈련 데이터 처리**: THOC는 완전히 정상 데이터로만 훈련된다고 가정하지만, 실제 데이터에는 소수의 이상이 포함될 수 있다.

### 7.3 일반화 한계

1. **도메인 간 전이의 어려움**: 한 도메인에서 훈련된 THOC를 다른 도메인으로 직접 전이하기 어렵다. 각 도메인마다 재훈련이 필요할 가능성이 높다.

2. **시간 개념 변화(Concept Drift) 대응 부재**: 정상 행동의 분포가 점진적으로 변할 때 이를 적응적으로 처리하는 메커니즘이 부족하다.

3. **이상 유형의 제한성**: 데이터 오염이 심한 경우나 매우 희귀한 이상의 탐지 성능이 제한적일 수 있다.

### 7.4 평가상의 한계

1. **MSL/SMAP 평가 수정**: 논문은 연속적인 이상 세그먼트에 대해 점조정(point adjustment)을 적용했다. 이는 실제 이상 탐지의 정확도를 과대평가할 수 있다.

2. **실시간 성능 미평가**: 추론 시간과 메모리 사용량에 대한 상세한 분석이 부족하다.

***

## 8. 2020년 이후 관련 최신 연구 비교 분석

### 8.1 주요 기술 진화 방향

#### 8.1.1 Transformer 기반 방법으로의 전환

**TranAD (2022)**: 트랜스포머 기반 이상치 탐지 및 진단 모델[2]
- 특징: 자기 어텐션(self-attention)으로 광범위한 시간적 추세 포착
- 성능: 여러 벤치마크에서 LSTM 기반 모델 능가
- 장점: RNN의 순차 처리 제약 극복, 병렬화 가능
- 한계: 메모리 사용량 증가, 하이퍼매개변수 추가

**VTT (Variable Temporal Transformer, 2024)**: 변수 간 상관관계를 명시적으로 모델링[3]
- 특징: 시간 자기어텐션 + 변수 자기어텐션의 이중 구조
- 성능: THOC보다 해석 가능성 향상
- THOC와의 차이: 초구면 기반이 아닌 재구성 오차 기반

**RTdetector (2025)**: 재구성 추세를 활용한 트랜스포머[4]
- 특징: RT-Attention으로 전역 추세 정보 보존
- 혁신: 재구성 추세 강화를 통한 이상 감지 향상

#### 8.1.2 대조 학습과의 결합

**DCdetector (2023)**: 다중 스케일 이중 어텐션 대조 학습[5]
- 특징: 순열 불변 표현 학습
- 장점: THOC의 다중 스케일 설계와 유사하나 대조 학습으로 강화
- 추세: 비지도 표현 학습의 강화

**Self-Supervised Learning 기반 접근 (2025)**: 시계열 이상치 탐지를 위한 자기 감독 학습[6]
- 최신 동향: 다양한 자기 감독 작업(예측, 재구성, 대조)의 통합
- THOC의 TSS와 비교: THOC는 예측만 사용, 최신 방법들은 다중 작업 활용

#### 8.1.3 그래프 신경망의 부상

**MGUAD (2023)**: 마스킹 전략을 활용한 그래프 신경망[7]
- 특징: 센서 간 숨겨진 인과관계 학습, 다중 마스킹 전략
- 장점: 다변량 시계열의 변수 간 의존성을 명시적으로 모델링
- THOC와의 차이: 센서 관계의 그래프 구조 학습 vs THOC의 계층적 클러스터링
- 성능: THOC와 유사 수준

**GDN (Graph Deviation Network)**: 그래프 구조 학습을 통한 이상 탐지[8]
- 특징: 노드 임베딩, 그래프 구조 학습, 그래프 어텐션 기반 예측
- 최신 성과: 다변량 시계열의 다중 채널 이상 탐지

#### 8.1.4 평가 방법론의 재검토

**QuoVadisTAD (2024)**: 현재 연구의 벤치마킹 관행 비판[9]
- 주요 지적: 
  - 현재 최신 딥러닝 모델들이 사실 선형 매핑만 학습
  - 평가 지표의 결함 (점조정의 과도한 사용)
  - 모델 복잡성 대비 성능 향상 미미
- 시사: THOC의 복잡성에 대한 검토 필요

**Time-IMM (2025)**: 불규칙한 다중 모달 시계열 데이터셋[10]
- 혁신: 실제 세계의 불규칙성을 반영한 데이터셋 구축
- 의미: THOC는 고정 길이 윈도우 가정으로 인한 한계 노출 가능

### 8.2 기술별 성능 비교 표

| 기술 | 주요 논문 | 핵심 특징 | THOC와 비교 | 출현 시기 |
|-----|---------|---------|-----------|---------|
| **Transformer** | TranAD | 자기어텐션 | 긴 범위 의존성, 해석성 개선 | 2022 |
| **Dual Attention Contrastive** | DCdetector | 다중 스케일 + 대조 | THOC과 유사, 대조학습 추가 | 2023 |
| **Graph Neural Network** | MGUAD | 구조 학습 + 마스킹 | 변수 관계 모델링, 복잡성 증가 | 2023 |
| **변수 주의** | VTT | 변수별 어텐션 | 해석성 강화, 계산 효율성 | 2024 |
| **원클래스 + 대조** | OATC | 전이학습 + 대조 | 도메인 간 일반화 개선 | 2025 |
| **One-class + 재구성 추세** | RTdetector | 추세 강화 | THOC의 초구면 설계와 유사 패러다임 | 2025 |

### 8.3 핵심 진화 경향 분석

**1. 표현 학습의 강화**
- THOC: 계층적 클러스터링과 TSS 손실로 표현 학습
- 최신: 대조 학습, 마스킹 전략, 다중 보조 작업으로 더욱 강화
- 영향: 더욱 판별력 있는 표현 학습 가능

**2. 다중 모달 정보 활용**
- THOC: 단일 시계열 데이터 처리
- 최신: 텍스트, 메타데이터 등 다중 모달 정보 통합 (예: Time-IMM)
- 의미: THOC 확장 가능성

**3. 해석 가능성 강화**
- THOC: 계층적 구조로 일부 해석 가능
- 최신: 어텐션 가시화, 인과 관계 학습으로 더욱 향상
- 영향: 산업 애플리케이션 적용성 개선

**4. 적응성과 강건성**
- THOC: 오염 허용성 부족
- 최신: Calibrated OCC, 개념 변화 대응, 동적 임계값 조정
- 필요성: 실제 운영 환경에서의 성능 개선

***

## 9. 연구의 영향 및 기여

### 9.1 학술 임팩트

THOC는 NeurIPS 2020에 게재되어 높은 인용률(511회)을 기록했으며, 다음과 같은 학술적 영향을 미쳤다:[11]

1. **원클래스 분류의 심층 학습 통합**: Deep SVDD의 시계열 확장에 성공적인 사례 제시
2. **다중 스케일 표현 학습**: 시계열의 계층적 특성 활용이 효과적임을 증명
3. **비지도 이상치 탐지의 기준 모델**: 이후 연구들이 비교 대상으로 사용

### 9.2 산업 적용 가능성

THOC의 실제 적용 가치는 다음 영역에서 높다:

- **사이버 물리 시스템 모니터링**: 전력망, 물 처리, 제조 시설 등에서 실시간 이상 탐지
- **네트워크 보안**: KDD-Cup99와 같은 침입 탐지 시나리오
- **우주 시스템 모니터링**: MSL, SMAP 등 우주 임무의 텔레메트리 이상 탐지

***

## 10. 향후 연구 방향 및 개선 방안

### 10.1 단기 개선 과제 (1-2년)

#### 10.1.1 적응형 계층 구조
현재 $K_l$과 $s(l)$이 고정되어 있는 것을 개선:
```
제안: 신경 아키텍처 탐색(NAS) 또는 메타러닝으로 데이터 특성에 맞는 
      최적 계층 구조 자동 선택
구현: 강화학습 또는 베이지안 최적화를 활용한 하이퍼매개변수 자동 선택
기대 효과: 서로 다른 도메인에 대한 일반화 성능 향상
```

#### 10.1.2 오염 허용 학습
훈련 데이터에 소수 이상이 포함된 현실 반영:
```
제안: Calibrated OCC 개념 통합
구현: 불확실성 모델링 및 두 단계 훈련 (이상 탐지 후 정제)
기대 효과: 실제 데이터에서의 강건성 향상
```

#### 10.1.3 시간 개념 변화 대응
점진적인 정상 분포 변화에 대응:
```
제안: 온라인 학습 또는 적응형 임계값
구현: 온라인 클러스터 업데이트, 슬라이딩 윈도우 기반 재훈련
기대 효과: 장기 운영 시스템에서의 성능 유지
```

### 10.2 중기 확장 방향 (2-3년)

#### 10.2.1 다중 모달 THOC
시계열 외에 다른 모달리티 통합:
```
구조:
- 시계열 특성: 확장 RNN (현재)
- 텍스트/메타데이터: Transformer 또는 주의 메커니즘
- 다중 모달 융합: 크로스 모달 어텐션

기대 성능 향상:
- 2D-gesture: 추가 손 위치 정보로 +5-10% 성능 향상 예상
- 산업 센서: 운영 로그와 결합으로 맥락 정보 제공
```

#### 10.2.2 전이 학습 프레임워크
도메인 간 일반화 강화:
```
아키텍처:
1. 사전 훈련: 다양한 도메인의 대규모 정상 시계열로 사전 훈련
2. 적응: 소량의 목표 도메인 데이터로 미세 조정
3. 메타러닝: 새로운 도메인에 빠른 적응

예상 효과: 작은 데이터셋에서도 좋은 성능 달성
```

#### 10.2.3 대조 학습 통합
현재의 TSS 손실을 더욱 강화:
```
개선 방안:
- 긍정 쌍: 동일 정상 패턴의 다른 타임스탬프
- 부정 쌍: 합성 이상 또는 다른 정상 패턴
- 다중 뷰: 다양한 시간 스케일에서의 표현 대조
```

수식 예시:

```math
$$L_{contrastive} = -\log \frac{\exp(sim(\bar{f}^l_{t,i}, \bar{f}^l_{t,j}) / \tau)}
{\sum_k \exp(sim(\bar{f}^l_{t,i}, \bar{f}^l_{t,k}) / \tau)}$$
```

### 10.3 장기 비전 (3-5년)

#### 10.3.1 신경상징적 THOC
기계 학습과 기호적 추론의 결합:
```
설계:
- 신경 계층: THOC의 특성 추출 및 표현 학습
- 기호 계층: 도메인 전문가 규칙이나 본체론적 지식 표현
- 통합: 신경상징적 추론으로 해석 가능한 이상 진단

응용: "센서 A가 높은데 센서 B는 낮음" 같은 규칙 기반 진단
```

#### 10.3.2 자율적 학습 시스템
사람 개입 최소화:
```
특징:
- 자동 재훈련: 온라인 학습으로 지속적 성능 유지
- 자동 임계값 조정: 비용 함수 기반 최적화
- 자동 설정 제안: 새 시스템에 대한 권장 설정 자동 제시

기술: 메타러닝 + 강화학습 + 베이지안 최적화
```

#### 10.3.3 대규모 다변량 시계열
수백 개 이상의 센서 처리:
```
과제: 계산 복잡도 O(n²)로 인한 확장성 문제
해결:
- 계층적 클러스터링: 센서를 그룹으로 계층화
- 스파스 그래프: GNN으로 중요 센서 간 관계만 모델링
- 분산 학습: 여러 디바이스에서 병렬 처리

기대: IoT 및 산업 4.0 대규모 시스템 적용
```

### 10.4 평가 및 벤치마킹 개선

QuoVadisTAD의 비판을 고려한 개선:

1. **더욱 엄격한 평가**
   - 점조정 제거 또는 제한된 사용
   - 실시간 추론 성능 측정
   - 도메인 외 일반화 성능 평가

2. **새로운 벤치마크 구축**
   - 불규칙한 시계열 (Time-IMM 따라)
   - 개념 변화 포함 데이터셋
   - 오염된 훈련 데이터

3. **간소한 베이스라인과의 비교**
   - 선형 모델의 성능 명시
   - 모델 복잡도 대비 성능 향상 정량화

***

## 결론

THOC는 시계열 이상치 탐지 분야에서 **원클래스 분류의 심층 학습 확장**이라는 중요한 이정표를 제시했다. 계층적 특성 융합과 다중 초구면 설계를 통해 복잡한 시간적 역학을 효과적으로 포착했으며, 다양한 실제 데이터셋에서 우수한 성능을 달성했다.

그러나 고정된 구조, 오염 데이터 취약성, 개념 변화 대응 부재 등의 한계가 있으며, 이러한 한계들을 개선하려는 노력이 2021년 이후의 후속 연구들에서 활발하다. Transformer 기반 방법, 대조 학습, 그래프 신경망 등의 최신 기술들이 이러한 문제들을 부분적으로 해결하고 있다.

**향후 연구의 열쇠**는 다음 세 가지에 있다:
1. **적응성**: 데이터 특성과 운영 환경에 자동으로 조응하는 시스템
2. **강건성**: 오염, 개념 변화, 새로운 이상 유형에 강한 모델
3. **해석성**: 산업 애플리케이션을 위한 명확한 이상 진단

THOC의 핵심 아이디어인 계층적 다중 스케일 특성 학습은 여전히 가치 있으며, 이를 최신 기술들과 결합하면 더욱 강력한 시계열 이상치 탐지 시스템을 구축할 수 있을 것으로 기대된다.

***

## 참고 문헌 (논문 내 인용 식별자)

[1] NeurIPS-2020-timeseries-anomaly-detection-using-temporal-hierarchical-one-class-network-Paper.pdf https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/41b028bc-1e68-4ab4-9fdd-1bc052374748/NeurIPS-2020-timeseries-anomaly-detection-using-temporal-hierarchical-one-class-network-Paper.pdf
[2] TranAD: Deep Transformer Networks for Anomaly ... https://arxiv.org/abs/2201.07284
[3] Transformer-based multivariate time series anomaly ... https://pure.korea.ac.kr/en/publications/transformer-based-multivariate-time-series-anomaly-detection-usin/
[4] RTdetector: Deep Transformer Networks for Time Series ... https://www.ijcai.org/proceedings/2025/0644.pdf
[5] DCdetector: Dual Attention Contrastive Representation Learning for Time
  Series Anomaly Detection https://arxiv.org/pdf/2306.10347.pdf
[6] A Review on Self-Supervised Learning for Time Series ... https://arxiv.org/html/2501.15196
[7] Masked Graph Neural Networks for Unsupervised Anomaly ... https://pmc.ncbi.nlm.nih.gov/articles/PMC10490803/
[8] Test Model https://www.mathworks.com/help/deeplearning/ug/multivariate-time-series-anomaly-detection-using-graph-neural-network.html
[9] Position: Quo Vadis, Unsupervised Time Series Anomaly Detection? https://arxiv.org/abs/2405.02678
[10] Time-IMM: A Dataset and Benchmark for Irregular ... https://arxiv.org/html/2506.10412v3
[11] Timeseries Anomaly Detection using Temporal ... https://proceedings.neurips.cc/paper/2020/hash/97e401a02082021fd24957f852e0e475-Abstract.html
[12] An Enhancing Timeseries Anomaly Detection Using LSTM and Bi-LSTM Architectures https://ieeexplore.ieee.org/document/10655101/
[13] Development on Industrial Data Anomaly Detection System Based on Machine Learning https://ieeexplore.ieee.org/document/11344113/
[14] Deep Autoencoders for Unsupervised Anomaly Detection in Wildfire Prediction https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024EA003997
[15] Pattern Recognition and Anomaly Detection in fetal morphology using Deep Learning and Statistical learning (PARADISE): protocol for the development of an intelligent decision support system using fetal morphology ultrasound scan to detect fetal congenital anomaly detection https://bmjopen.bmj.com/lookup/doi/10.1136/bmjopen-2023-077366
[16] Machine learning and deep learning models for preoperative detection of lymph node metastasis in colorectal cancer: a systematic review and meta-analysis https://link.springer.com/10.1007/s00261-024-04668-z
[17] Integrated Deep Learning Model for the Detection, Segmentation, and Morphologic Analysis of Intracranial Aneurysms Using CT Angiography. http://pubs.rsna.org/doi/10.1148/ryai.240017
[18] Deep Guard-IoT: A Systematic Review of AI-Based Anomaly Detection Frameworks for Next-Generation IoT Security (2020-2024) https://wjps.uowasit.edu.iq/index.php/wjps/article/view/598
[19] Sea Surface Temperature and Marine Heat Wave Predictions in the South China Sea: A 3D U-Net Deep Learning Model Integrating Multi-Source Data https://www.mdpi.com/2073-4433/15/1/86
[20] Deep Learning and Process Mining for ERP Anomaly Detection: Toward Predictive and Self-Monitoring Enterprise Platforms https://ijsrcseit.com/CSEIT217554
[21] Ensemble neuroevolution based approach for multivariate time series
  anomaly detection https://arxiv.org/pdf/2108.03585.pdf
[22] TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate
  Time Series Data https://arxiv.org/pdf/2201.07284.pdf
[23] TFAD: A Decomposition Time Series Anomaly Detection Architecture with
  Time-Frequency Analysis https://arxiv.org/pdf/2210.09693.pdf
[24] MadSGM: Multivariate Anomaly Detection with Score-based Generative
  Models https://arxiv.org/pdf/2308.15069.pdf
[25] A Novel Deep Learning Approach for Anomaly Detection of Time Series Data https://downloads.hindawi.com/journals/sp/2021/6636270.pdf
[26] Deep Learning for Time Series Anomaly Detection: A Survey https://arxiv.org/pdf/2211.05244.pdf
[27] MEMTO: Memory-guided Transformer for Multivariate Time Series Anomaly
  Detection http://arxiv.org/pdf/2312.02530.pdf
[28] Rectifying Time Series Anomaly Detection - ar5iv - arXiv https://ar5iv.labs.arxiv.org/html/2203.05167
[29] Temporal Analysis Framework for Intrusion Detection ... https://arxiv.org/html/2511.03799v1
[30] Time Series Data Cleaning with Regular and Irregular ... https://arxiv.org/pdf/2004.08284.pdf
[31] Benchmarking Unsupervised Strategies for Anomaly ... https://arxiv.org/html/2506.20574v1
[32] Temporal Analysis Framework for Intrusion Detection ... https://arxiv.org/pdf/2511.03799.pdf
[33] A Survey on Deep Learning based Time Series Analysis ... https://arxiv.org/html/2302.02173v6
[34] LogLLaMA: Transformer-based log anomaly detection with ... https://arxiv.org/html/2503.14849v1
[35] A Survey of Anomaly Detection in Cyber-Physical Systems https://arxiv.org/html/2502.13256v1
[36] Deep Learning for Satellite Image Time Series Analysis https://arxiv.org/pdf/2404.03936.pdf
[37] [2504.04011] Foundation Models for Time Series: A Survey https://ar5iv.labs.arxiv.org/html/2504.04011
[38] Recent Advances in Multi-Agent Human Trajectory Prediction https://arxiv.org/html/2506.14831v2
[39] BEAT: Balanced Frequency Adaptive Tuning for Long- ... https://arxiv.org/pdf/2501.19065.pdf
[40] [2509.19985] Pi-Transformer: A Physics-informed Attention ... https://arxiv.org/abs/2509.19985
[41] How Visualization is Shaping Malware Detection https://arxiv.org/pdf/2505.07574.pdf
[42] Time series forecasting and anomaly detection using deep learning https://www.sciencedirect.com/science/article/abs/pii/S0098135423004301
[43] An optimized LSTM-based deep learning model for ... https://www.nature.com/articles/s41598-025-85248-z
[44] Deep Learning for Time Series Anomaly Detection: A Survey https://arxiv.org/html/2211.05244v3
[45] Transformer-based multivariate time series anomaly ... https://www.sciencedirect.com/science/article/abs/pii/S0950705124001424
[46] Kolmogorov-Arnold Networks-based GRU and LSTM for ... https://arxiv.org/pdf/2507.13685.pdf
[47] Time Series Anomaly Detection Using Deep Learning https://www.mathworks.com/help/deeplearning/ug/time-series-anomaly-detection-using-deep-learning.html
[48] Time-series anomaly detection with stacked Transformer representations and 1D convolutional network https://www.sciencedirect.com/science/article/abs/pii/S0952197623001483
[49] A comparative study of RNN, LSTM, GRU, and hybrid models https://pmc.ncbi.nlm.nih.gov/articles/PMC12329085/
[50] Deep Learning for Time Series Anomaly Detection: A Survey https://dl.acm.org/doi/10.1145/3691338
[51] LSTM and GRU type recurrent neural networks in model ... https://www.sciencedirect.com/science/article/pii/S0925231225003844
[52] Deep anomaly detection for time series: A survey https://www.sciencedirect.com/science/article/abs/pii/S1574013725000632
[53] Anomalization-based GRU and LSTM classifiers https://pubs.aip.org/aip/acp/article/3342/1/030048/3365288/Anomalization-based-GRU-and-LSTM-classifiers-An
[54] Reconstruction-Based One-Class Classification Anomaly Detection for Tabular Data https://link.springer.com/10.1007/978-981-19-4109-2_29
[55] SLSG: Industrial image anomaly detection with improved feature embeddings and one-class classification https://linkinghub.elsevier.com/retrieve/pii/S0031320324006137
[56] One-Class Classification and Cluster Ensembles for Anomaly Detection and Diagnosis in Multivariate Time Series Data https://drops.dagstuhl.de/entities/document/10.4230/OASIcs.DX.2024.14
[57] Application of Online Anomaly Detection Using One-Class Classification to the Z24 Bridge https://www.mdpi.com/1424-8220/24/23/7866
[58] Anomaly Detection Utilizing One-Class Classification—A Machine Learning Approach for the Analysis of Plant Fast Fluorescence Kinetics https://www.mdpi.com/2673-7140/4/4/51
[59] Understanding Time Series Anomaly State Detection through One-Class Classification https://arxiv.org/abs/2402.02007
[60] Deep one-class classification model assisted by radius constraint for anomaly detection of industrial control systems https://linkinghub.elsevier.com/retrieve/pii/S095219762401515X
[61] Comprehensive Review of One-Class Classification Approaches for Anomaly Detection https://ijcaonline.org/archives/volume186/number45/challa-2024-ijca-924134.pdf
[62] Self-Trained One-class Classification for Unsupervised Anomaly Detection https://www.semanticscholar.org/paper/a7d75aa3a0a9faa310fb524c350fba2093b0ec97
[63] Malware Detection for Internet of Things Using One-Class Classification https://www.mdpi.com/1424-8220/24/13/4122
[64] Exploring the Optimization Objective of One-Class Classification for
  Anomaly Detection https://arxiv.org/pdf/2308.11898.pdf
[65] Self-supervise, Refine, Repeat: Improving Unsupervised Anomaly Detection https://arxiv.org/pdf/2106.06115v1.pdf
[66] Leave-One-Out-, Bootstrap- and Cross-Conformal Anomaly Detectors http://arxiv.org/pdf/2402.16388.pdf
[67] Meta-learning One-class Classifiers with Eigenvalue Solvers for
  Supervised Anomaly Detection https://arxiv.org/pdf/2103.00684.pdf
[68] Calibrated One-class Classification for Unsupervised Time Series Anomaly
  Detection http://arxiv.org/pdf/2207.12201.pdf
[69] Critical Review for One-class Classification: recent advances and the
  reality behind them http://arxiv.org/pdf/2404.17931.pdf
[70] Anomaly Detection using One-Class Neural Networks https://arxiv.org/pdf/1802.06360.pdf
[71] Student-Teacher Feature Pyramid Matching for Anomaly Detection https://arxiv.org/pdf/2103.04257.pdf
[72] autoencoders for anomaly detection are un https://arxiv.org/pdf/2501.13864.pdf
[73] Hierarchical Multivariate Anomaly Detection at Cloud-Scale https://arxiv.org/pdf/2202.06892.pdf
[74] [PDF] Graph Anomaly Detection With Graph Neural Networks https://www.semanticscholar.org/paper/Graph-Anomaly-Detection-With-Graph-Neural-Networks:-Kim-Lee/20309e3990cd612a13e389e1572786e55100f03d
[75] A Hierarchical Framework with Spatio-Temporal Consistency ... https://arxiv.org/html/2401.10300v2
[76] FADE: Adversarial Concept Erasure in Flow Models https://arxiv.org/pdf/2507.12283.pdf
[77] Full-conformal novelty detection: A powerful and non- ... https://arxiv.org/pdf/2501.02703.pdf
[78] A Hierarchical Approach to Conditional Random Fields for ... https://arxiv.org/abs/2210.15030
[79] A Brain-Inspired Framework for Modeling Latent Market ... https://arxiv.org/pdf/2508.02012.pdf
[80] Autoencoders for Anomaly Detection are Unreliable https://arxiv.org/html/2501.13864v1
[81] Dive into Time-Series Anomaly Detection: A Decade Review https://arxiv.org/html/2412.20512v1
[82] A Copula-based variational autoencoder for uncertainty ... https://arxiv.org/pdf/2510.02013.pdf
[83] A Closer Look at TabPFN v2: Strength, Limitation, and ... https://arxiv.org/html/2502.17361v1
[84] Adaptive Anomaly Detection in the Presence of Concept Drift https://arxiv.org/abs/2506.15831
[85] The Financial Connectome: A Brain-Inspired Framework ... https://arxiv.org/html/2508.02012v1
[86] One-classification anomaly detection: Utilizing Contrastive ... https://www.sciencedirect.com/science/article/abs/pii/S026322412402058X
[87] Timeseries Anomaly Detection using Temporal ... https://proceedings.neurips.cc/paper/2020/file/97e401a02082021fd24957f852e0e475-Paper.pdf
[88] A deep one-class classifier for network anomaly detection ... https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2025.1646679/full
[89] Time Series Anomaly Detection using Temporal ... https://emptydb.tistory.com/10
[90] Active anomaly detection based on deep one-class ... https://pure.kaist.ac.kr/en/publications/active-anomaly-detection-based-on-deep-one-class-classification/
[91] [Paper Review] Timeseries Anomaly Detection using ... https://www.youtube.com/watch?v=juBSSe3YYos
[92] Graph Neural Network-Based Anomaly Detection in ... https://arxiv.org/abs/2106.06947
[93] SSL-OCC - Anomaly Detection 논문 리뷰 https://ffighting.net/deep-learning-paper-review/anomaly-detection/ssl-occ/
[94] [논문 리뷰] Timeseries Anomaly Detection using Temporal ... https://velog.io/@plumlee/%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-Timeseries-Anomaly-Detection-using-Temporal-Hierarchical-One-Class-Network
[95] Multivariate Time-Series Anomaly Detection based on ... https://arxiv.org/html/2408.13082v1
[96] [논문 리뷰] 이상치 탐지 | Deep SVDD, Deep One ... - 어쩌다통계 https://slowsteadystat.tistory.com/34
[97] mala-lab/Awesome-Deep-Graph-Anomaly-Detection https://github.com/mala-lab/Awesome-Deep-Graph-Anomaly-Detection
