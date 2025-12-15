
# T2SR: Super‐Resolution in Smart Meter Data Using a Transformer‐Based Framework

## 1. 핵심 주장 및 주요 기여

**T2SR 논문의 핵심 주장**은 **트랜스포머 기반 아키텍처가 저해상도 스마트 미터 데이터로부터 고해상도 데이터를 재구성하는 데 있어 CNN, LSTM, MLP 등 기존 딥러닝 방법들보다 우수하다**는 것입니다.[1]

**주요 기여는 세 가지**입니다:[1]

1. **방법론적 기여**: 스마트 미터 데이터 초해상도 변환을 위한 최초의 트랜스포머 기반 깊은 학습 프레임워크인 T2SR을 제안
2. **응용 기여**: 실제 스마트 미터 데이터(노르웨이 베르겐의 주거 건물 데이터)로 T2SR과 다른 딥러닝 방법들을 검증
3. **비교 분석 기여**: T2SR을 여러 기존 최신 딥러닝 기법과 철저하게 비교하여 현재 연구 환경에서의 성능을 평가

## 2. 문제 정의, 제안 방법 및 모델 구조

### 2.1 해결하고자 하는 문제

스마트 미터는 전력 소비 데이터를 수집하지만, 높은 해상도(예: 초 단위 또는 분 단위)의 데이터 수집은 여러 문제를 야기합니다:[1]

- **데이터 저장 및 전송 비용 증가**: 높은 빈도의 데이터는 방대한 저장 용량과 네트워크 대역폭을 필요로 함
- **전산 자원 부하**: 실시간 처리 능력에 대한 높은 요구
- **정보 손실**: 저해상도 데이터는 단기간의 소비 변동(급격한 피크, 일시적 변동)을 포착하지 못함

초해상도(Super-Resolution, SR) 기술은 저해상도 스마트 미터 데이터로부터 고해상도 데이터를 재구성하여 이러한 문제를 해결합니다.[1]

### 2.2 제안하는 방법 및 수식

#### 2.2.1 위치 인코딩(Positional Encoding)

트랜스포머는 시퀀스의 순서 정보를 본질적으로 이해하지 못하므로, 위치 인코딩을 통해 시간 정보를 제공합니다:[1]

$$PE_{pos,2i} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{pos,2i+1} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

여기서 $pos$는 위치 인덱스, $i$는 임베딩 차원입니다.

#### 2.2.2 스케일 닷-곱 어텐션(Scaled Dot-Product Attention)

인코더와 디코더의 핵심은 다음 수식으로 표현되는 스케일 닷-곱 어텐션입니다:[1]

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

여기서 $Q \in \mathbb{R}^{t \times d_k}$는 쿼리 행렬, $K \in \mathbb{R}^{t \times d_k}$는 키 행렬, $V \in \mathbb{R}^{t \times d_v}$는 값 행렬, $d_k$는 키의 차원입니다.[1]

#### 2.2.3 멀티헤드 어텐션(Multi-Head Attention)

어텐션을 한 번만 계산하는 대신, 다중 어텐션 메커니즘이 병렬로 실행됩니다:[1]

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

여기서 $W_i^Q, W_i^K, W_i^V$는 학습 가능한 가중치 행렬이고, T2SR은 $h=4$개의 어텐션 헤드를 사용합니다.[1]

#### 2.2.4 평가 지표

모델 성능 평가는 MSE와 MAE를 사용합니다:[1]

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

여기서 $y_i$는 실제값, $\hat{y}_i$는 예측값, $n$은 전체 관측값의 개수입니다.[1]

### 2.3 모델 구조

T2SR 프레임워크의 아키텍처는 다음과 같이 구성됩니다:[1]

**입력 임베딩 레이어**: 저해상도 시계열 데이터를 64차원 임베딩 공간으로 투영하여 시간 패턴 감지를 가능하게 합니다.

**위치 인코딩**: 정현파 및 코사인파 함수를 사용하여 시간 순서 정보를 추가합니다.

**트랜스포머 인코더**: 2개의 동일한 레이어로 구성되며 각 레이어는:
- 멀티헤드 자기 어텐션(MHA) 메커니즘
- 피드포워드 신경망(FFN)
- 잔차 연결(Residual Connections)
- 층 정규화(Layer Normalization)

**트랜스포머 디코더**: 2개의 동일한 레이어로 구성되며:
- 마스크된 멀티헤드 자기 어텐션 (미래 시점 참조 방지)
- 크로스 어텐션 레이어 (인코더 출력 참조)
- 피드포워드 신경망
- 잔차 연결 및 층 정규화

**출력 선형 레이어**: 디코더의 고차원 출력을 최종 고해상도 예측값으로 매핑합니다.[1]

## 3. 성능 향상 및 한계

### 3.1 성능 향상

T2SR은 특히 **MSE(Mean Squared Error) 점수에서 뛰어난 성능**을 보입니다:[1]

- **스케일업 팩터 α=5일 때**: CNN, LSTM, MLP과 비교하여 모든 세 가지 훈련 시나리오(1일, 3일, 7일)에서 우수한 MSE 성능을 달성
  - 1일 훈련: T2SR MSE=645.43 vs CNN=920.50
  - 3일 훈련: T2SR MSE=6.25 vs CNN=311.10
  - 7일 훈련: T2SR MSE=46.77 vs CNN=354.25

- **복잡한 패턴 예측 능력**: 단기간 고강도 피크(수 초)와 같은 세밀한 세부사항을 다른 방법들보다 효과적으로 포착합니다.

- **훈련 데이터 증가에 대한 강건성**: 훈련 데이터가 3일에서 7일로 증가할 때 성능 개선이 뚜렷합니다.

### 3.2 주요 한계

**1. MAE 성능의 일관성 부족**

- MAE 점수는 다른 방법들과 비슷하거나 때로 낮습니다
- 이는 모델이 MSE 손실함수로 최적화되었기 때문에, 큰 오류를 줄이는 데 초점을 맞추되 일반적인 정확도(모든 오류를 동등하게 취급)에서는 덜 효과적입니다.[1]

**2. 높은 계산 비용**

- CNN, LSTM, MLP는 수 분 내에 훈련되지만 T2SR은 수 시간이 필요합니다
  - 1일 훈련: 다른 방법들 3.9~6분 vs T2SR 3.9시간
  - 7일 훈련: 다른 방법들 12~46분 vs T2SR 11시간[1]

**3. 높은 스케일업 팩터에서의 성능 저하**

- α가 5에서 150으로 증가하면 T2SR의 성능 우위가 감소합니다
- α=150일 때는 다른 방법들과의 차이가 크게 줄어듭니다
- 특히 고해상도 재구성이 필요할 때 정확도 하락[1]

**4. 패턴 유사성에 대한 민감성**

- 흥미로운 발견: 훈련 데이터가 많다고 해서 반드시 성능이 향상되지 않습니다
- 3일 훈련 데이터가 목표 패턴과 유사할 때 7일 훈련보다 좋은 결과를 얻음
- 이는 데이터의 양보다 **훈련 데이터와 테스트 데이터의 패턴 일치성이 중요**함을 시사합니다.[1]

## 4. 모델의 일반화 성능 향상 가능성

### 4.1 현재 일반화 성능의 한계

**1. 데이터셋 의존성**
- 논문은 단일 건물(노르웨이 베르겐의 5가구 주택)의 데이터만 사용
- 서로 다른 지역, 기후, 건물 특성에 대한 일반화 능력이 검증되지 않음
- 공개 데이터셋 부재로 인한 재현성 한계[1]

**2. 계절성 및 일중 변동성**
- 실제 전력 소비는 계절, 날씨, 요일에 따라 크게 변함
- 매일 다른 패턴이 나타나는 실제 데이터의 복잡성을 충분히 포착하지 못할 수 있음[1]

### 4.2 일반화 성능 향상 전략

**1. 동적 데이터 선택 전략 도입**

논문에서 제시한 대로, 단순히 훈련 데이터의 양을 늘리기보다는 **목표 패턴과 유사한 훈련 데이터를 선택하는 동적 전략**이 필요합니다.[1]

- 시계열 데이터의 유사성 측도(예: DTW, Euclidean Distance)를 사용
- 목표 데이터와 패턴이 유사한 과거 데이터를 우선적으로 선택
- 적응적 훈련 데이터 크기 조정

**2. 전이 학습(Transfer Learning) 적용**

최근 연구에서 전이 학습을 통해 일반화 성능을 향상시킬 수 있음이 입증되었습니다.[2][3]

- 여러 건물/지역의 데이터로 사전 훈련
- 새로운 건물/지역에 대해 미세 조정(Fine-tuning)
- 제한된 데이터로도 좋은 성능 달성 가능

**3. 앙상블 방법 고려**

- T2SR과 다른 모델(CNN, LSTM)의 앙상블
- 각 모델의 강점을 활용: T2SR의 세밀한 패턴 포착 + 다른 모델의 일반적 정확도
- 로버스트성 향상[4][5]

**4. 계층화된 멀티스케일 아키텍처**

최신 연구에서 멀티스케일 기능 추출이 일반화 성능을 크게 향상시킵니다.[6][4]

- 다양한 시간 스케일에서의 패턴 학습
- 단기(시간 단위)와 장기(일/주 단위) 의존성 모두 포착
- Autoformer, PatchTST 등 시간 분해 기반 모델의 개념 통합

**5. 불확실성 추정 메커니즘 추가**

예측의 신뢰도를 정량화하여 일반화 성능을 향상시킬 수 있습니다.[6]

- 확률적 예측 프레임워크 도입
- 예측 불확실성이 높은 경우 추가 관측 요청
- 의사결정 시 리스크 관리

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 주요 초해상도 방법들 비교

| **방법** | **연도** | **기술** | **장점** | **한계** | **성능 메트릭** |
|---------|--------|---------|---------|---------|---------------|
| **Super Resolution Perception (SRP)**[7] | 2020 | CNN 기반 인코더-디코더 | 고정 된 강 계층, 공간 의존성 포착 | 계산 비용 높음, 대규모 훈련 데이터 필요 | MSE: 기준선 대비 우수 |
| **ProfileSR-GAN**[8][9] | 2022 | GAN 기반 생성기-판별기 | 세밀한 고해상도 프로필 생성, 시각적 품질 우수 | 훈련 불안정성, 모드 붕괴 위험, 과도한 계산 비용 | PSNR: 다른 방법 대비 우수 |
| **Constrained GAN with CVXPY**[10][11] | 2024 | GAN + 물리적 제약 | 시간적 일관성 보장, 물리적 제약 준수 | 제약 설정 복잡성, 훈련 어려움 | 최대 변화율: 제약 없는 GAN 대비 50% 개선 |
| **Multivariate Super-Resolution (Wasserstein-CAN)**[12] | 2024 | 조건부 적대적 네트워크 + Wasserstein 손실 | 다변량 데이터 처리, 임의의 시간 스케일 | 높은 계산 복잡도 | 임의의 시간 스케일에서 강건 |
| **T2SR (본 논문)**[1] | 2025 | 트랜스포머 기반 | 병렬 처리 능력, 긴 시퀀스 의존성 포착, 세밀한 패턴 감지 | 높은 훈련 시간, MAE 성능 약함, 높은 스케일업 팩터에서 성능 저하 | MSE: α=5에서 최고 성능 |

### 5.2 에너지 시계열 예측 분야의 트랜스포머 활용

#### 최신 트랜스포머 아키텍처

**1. Temporal Fusion Transformer (TFT)**[13][14][15]

- **특징**: 멀티호라이즌 예측을 위해 특별히 설계, 정적 및 시간 특성 모두 처리
- **성능**: LSTM 비교 대비 RMSE 2.02, MAE 1.50로 우수
- **응용**: 에너지 소비 예측에서 상태 기술(SOTA) 달성[14]
- **장점**: 복잡한 시간 의존성 처리, 자해석성 제공

**2. Autoformer**[16][4]

- **특징**: 시계열 분해 기반 설계, 주기성과 추세 분리
- **성능**: 에너지 소비 데이터에서 가장 효율적인 트랜스포머[16]
- **응용**: 정기성이 강한 에너지 소비 패턴에 최적화

**3. PatchTST (Patch Time Series Transformer)**[17][3]

- **특징**: 패치 기반 입력 처리, 로컬 패턴 캡처
- **성능**: 전이 학습에서 우수한 일반화 능력[3]
- **장점**: 제한된 데이터로도 효과적, 가벼운 계산

**4. EnergyPatchTST**[6]

- **특징**: 멀티스케일 아키텍처, 에너지 데이터 특화
- **혁신**: 미래 알려진 정보 통합, 확률적 예측 프레임워크
- **성능**: 장기 예측에서 멀티스케일 특성 추출로 큰 성능 향상

**5. VARMAformer**[18]

- **특징**: 고전 VARMA 모델과 심층 학습 결합
- **혁신**: 지역 패치 역학 추출 모듈(VFE), 시간 게이트 어텐션
- **성능**: 기존 크로스 어텐션 전용 모델 대비 MSE 개선

#### 하이브리드 접근법의 발전

**CNN-LSTM 조합**[19]

- 특징 추출(CNN)과 시간 의존성(LSTM) 결합
- 수정된 Sea Lion 알고리즘으로 하이퍼파라미터 최적화
- 기존 CNN-LSTM 대비 37% 더 빠른 예측

**Transformer-LSTM-PSO**[5]

- LSTM의 단기 의존성 + Transformer의 장기 의존성
- PSO 알고리즘으로 최적화
- 기존 모델 대비 RMSE 15%, MAE 20% 개선

### 5.3 주요 연구 동향 분석

**1. 아키텍처 혁신**
- RNN/LSTM → 트랜스포머로의 패러다임 전환
- 더 나은 일반화를 위한 하이브리드 모델(CNN-LSTM, Transformer-LSTM)
- 에너지 데이터 특성을 반영한 맞춤형 설계 (Autoformer, PatchTST, EnergyPatchTST)

**2. 초해상도 분야**
- 초기: 보간법, 신호 처리(FFT, 웨이블릿)
- 중기: CNN 기반 SRP, GAN 기반 ProfileSR-GAN
- 최신: 제약 조건 통합 GAN, 트랜스포머 기반 방법

**3. 전이 학습 및 일반화**
- 단일 건물 모델 → 다중 건물 사전 훈련 + 미세 조정
- 제한된 데이터 시나리오에서 우수한 성능[2]
- 영역 간 일반화 능력 향상

**4. 확률적 예측**
- 점 예측 → 확률 분포 예측으로 확장
- 불확실성 정량화의 중요성 증가
- 의사결정 지원 강화

**5. 물리 기반 제약**
- 순수 데이터 주도 방법 → 물리적 현실성 제약 추가
- 예: Constrained GAN, Physics-informed Transformer (Pi-Transformer)
- 생성 데이터의 신뢰성 향상

## 6. 논문이 향후 연구에 미치는 영향

### 6.1 긍정적 영향

**1. 새로운 연구 방향 제시**
- 스마트 미터 초해상도 분야에서 처음으로 트랜스포머 적용
- 기존의 CNN/LSTM 기반 접근법에서 벗어나 셀프 어텐션 메커니즘의 가치 입증
- 에너지 데이터 처리에 트랜스포머 도입의 타당성 확립[1]

**2. 실무 적용 기반 마련**
- 실제 스마트 미터 데이터(노르웨이 베르겐)를 사용한 검증
- 데이터 수집, 전처리, 매핑 함수 등 실제 구현 세부사항 제공
- 공개 코드 제공으로 재현성 및 확장성 지원[1]

**3. 병목 문제 식별**
- 높은 계산 비용(MSE 최적화 중심)과 MAE 성능의 불일치 명확히
- 훈련 데이터의 양보다 패턴 유사성의 중요성 강조
- 높은 스케일업 팩터에서의 성능 한계 제시[1]

### 6.2 한계 및 개선 필요 영역

**1. 일반화 능력의 약점**
- 단일 건물, 단일 기후 지역 데이터만 사용
- 다양한 지역, 건물 유형, 기후 조건에 대한 검증 부재
- 계절성, 날씨 변화, 특이 사건에 대한 강건성 미흡[1]

**2. 계산 효율성 문제**
- 훈련 시간이 수시간에 달함
- 실시간 응용이 필요한 스마트 그리드 환경에서는 실용성 제한
- 엣지 디바이스나 임베디드 시스템 배포 어려움[1]

**3. 평가 메트릭의 편향성**
- MSE 손실함수로 최적화되어 MAE 성능이 떨어짐
- 실제 에너지 관리 문제에서는 일반적 정확도(MAE)도 중요
- 다중 목적 최적화 필요성 대두[1]

## 7. 향후 연구 시 고려사항

### 7.1 기술적 개선 방향

**1. 계산 효율성 향상**
- 모델 경량화: 레이어/헤드 수 감소, 임베딩 차원 조정
- 효율적 어텐션 메커니즘: Flash Attention, Sparse Attention 적용[20]
- 혼합 정밀도 훈련(mixed precision training)
- 지식 증류(knowledge distillation)를 통한 압축

**2. 다중 목적 손실함수**
- MSE와 MAE의 균형: 가중 조합 손실함수
- $$L = \alpha \cdot MSE + (1-\alpha) \cdot MAE$$
- 또는 분위수 손실(quantile loss)로 예측 분포 학습

**3. 동적 데이터 선택 전략**
- 목표 패턴과 유사한 훈련 샘플 자동 선택
- 시계열 유사성 측도 활용 (DTW, Euclidean distance)
- Meta-learning 적용으로 적응적 학습

**4. 불확실성 추정**
- 베이지안 신경망 또는 Monte Carlo Dropout
- 확률적 예측 프레임워크
- 예측 신뢰도 제공으로 의사결정 개선

### 7.2 데이터 및 평가 개선

**1. 공개 데이터셋 구축**
- 다양한 지역, 건물 유형, 기후 조건 포함
- 국제적 협력을 통한 대규모 통합 데이터셋
- 벤치마킹 표준화[1]

**2. 광역 검증**
- 다양한 지역(기후, 건물 특성)에서의 성능 평가
- 계절별, 날씨별, 요일별 세부 분석
- 특이 사건(휴일, 이상 기후) 대응 성능 평가

**3. 실시간 응용 검증**
- 온라인 학습(online learning) 시나리오
- 개념 드리프트(concept drift) 처리
- 점진적 모델 업데이트 메커니즘

### 7.3 방법론 확장

**1. 전이 학습 프레임워크**
- 다중 건물/지역 사전 훈련
- 새로운 건물/지역에 대한 효율적 미세 조정
- 도메인 적응(domain adaptation) 기법 적용[3][2]

**2. 멀티모달 통합**
- 스마트 미터 데이터 + 기상 정보 통합
- 에너지 수요 영향 요인의 다변량 모델링
- 외부 정보 채널 추가로 예측력 향상

**3. 물리 기반 제약 통합**
- 에너지 보존 법칙 임포즈
- 시간적 일관성 보장
- Constrained GAN, Physics-informed Transformer 개념 응용[10][21]

**4. 하이브리드 앙상블**
- T2SR + ProfileSR-GAN: 세밀함 + 현실성
- T2SR + CNN: 계산 효율 + 패턴 포착
- 적응적 앙상블 가중치 학습

### 7.4 응용 분야 확대

**1. 실시간 이상 탐지**
- 고해상도 데이터로 비정상 소비 패턴 조기 감지
- 고장/누수 식별 개선

**2. 비침투적 부하 모니터링(NILM)**
- 고해상도 데이터를 통한 가전제품별 소비 식별 정확도 향상

**3. 수요 대응(Demand Response)**
- 정확한 고해상도 예측으로 효율적 수요 조정
- 실시간 가격 신호 반응 최적화

**4. 마이크로그리드 및 분산 에너지**
- 소규모 태양광, 배터리 시스템과의 통합
- 국소 수급 균형 최적화

## 결론

T2SR은 **스마트 미터 초해상도 변환 분야에서 트랜스포머의 활용 가능성을 최초로 입증한 획기적 연구**입니다. 특히 MSE 성능에서의 우수성과 세밀한 패턴 포착 능력은 향후 에너지 데이터 분석 연구의 새로운 방향을 제시합니다.

그러나 높은 계산 비용, MAE 성능의 한계, 제한된 일반화 능력은 실제 배포에 있어 중요한 과제입니다. 향후 연구는 **전이 학습, 동적 데이터 선택, 다중 목적 최적화, 물리 기반 제약 통합** 등을 통해 이러한 한계를 극복하고, **다양한 환경에서의 강건한 일반화 성능 달성**에 초점을 맞춰야 합니다. 

2024-2025년의 최신 연구들(Autoformer, PatchTST, EnergyPatchTST, VARMAformer)은 트랜스포머의 개선된 변형들이 에너지 시계열 예측에서 지속적으로 발전하고 있음을 보여줍니다. T2SR은 이러한 발전의 중요한 기초석이 될 것으로 예상됩니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1186d130-cfa3-48fc-9303-d43a7bdc32e3/IET-Smart-Grid-2025-Iversen-T2SR-Super-E2-80-90Resolution-in-Smart-Meter-Data-Using-a-Transformer-E2-80-90Based-Framework.pdf)
[2](https://arxiv.org/pdf/2310.19159.pdf)
[3](https://arxiv.org/pdf/2410.14107.pdf)
[4](https://arxiv.org/html/2507.23147v1)
[5](https://arxiv.org/pdf/2408.12129.pdf)
[6](https://arxiv.org/pdf/2508.05454.pdf)
[7](https://www.sciencedirect.com/science/article/abs/pii/S0020025520302681)
[8](https://arxiv.org/ftp/arxiv/papers/2107/2107.09523.pdf)
[9](https://arxiv.org/abs/2107.09523)
[10](http://arxiv.org/pdf/2311.12166.pdf)
[11](https://arxiv.org/pdf/2311.12166.pdf)
[12](https://ieeexplore.ieee.org/document/10629176/)
[13](https://www.academia.edu/108970113/Forecasting_energy_consumption_demand_of_customers_in_smart_grid_using_Temporal_Fusion_Transformer_TFT)
[14](https://pmc.ncbi.nlm.nih.gov/articles/PMC11996805/)
[15](https://pdfs.semanticscholar.org/1139/3c6cb4e05056dd5483a81198d1ebf2b6ae94.pdf)
[16](https://www.jait.us/articles/2025/JAIT-V16N5-623.pdf)
[17](https://peerj.com/articles/cs-3001/)
[18](https://arxiv.org/html/2509.04782v1)
[19](https://www.internationaljournalssrg.org/IJEEE/2023/Volume10-Issue9/IJEEE-V10I9P121.pdf)
[20](https://arxiv.org/pdf/2507.07247.pdf)
[21](https://arxiv.org/html/2509.19985v1)
[22](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/stg2.70010)
[23](http://thesai.org/Publications/ViewPaper?Volume=16&Issue=6&Code=ijacsa&SerialNo=9)
[24](https://www.preprints.org/manuscript/202009.0678/v1)
[25](https://onepetro.org/SPEADIP/proceedings/25ADIP/25ADIP/D041S136R005/793682)
[26](http://pubs.rsna.org/doi/10.1148/radiol.233529)
[27](https://aacrjournals.org/cancerres/article/85/8_Supplement_1/7429/759411/Abstract-7429-Illuminating-the-dark-genome-in)
[28](https://www.semanticscholar.org/paper/137da5f5fffa5422cd6e4d2fd8ba556adc8bd247)
[29](https://wires.onlinelibrary.wiley.com/doi/10.1002/widm.1406)
[30](https://www.tandfonline.com/doi/full/10.1080/0951192X.2023.2235679)
[31](https://www.mdpi.com/2071-1050/16/5/1925/pdf?version=1708963182)
[32](https://linkinghub.elsevier.com/retrieve/pii/S2666546821000550)
[33](https://arxiv.org/pdf/2109.05666.pdf)
[34](https://www.mdpi.com/2076-3417/11/6/2742/pdf)
[35](https://arxiv.org/pdf/1907.11377.pdf)
[36](https://arxiv.org/pdf/1809.06687.pdf)
[37](https://arxiv.org/pdf/2403.01438.pdf)
[38](https://www.mdpi.com/1424-8220/22/21/8543/pdf?version=1667817620)
[39](https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2023.1288683/full)
[40](https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2023.1288683/pdf)
[41](https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2024.1355222/full)
[42](https://arxiv.org/pdf/2107.13653.pdf)
[43](https://www.bohrium.com/paper-details/advancements-in-super-resolution-methods-for-smart-meter-data/954407554941714735-4468)
[44](https://www.sciencedirect.com/science/article/abs/pii/S0957417425021724)
[45](https://par.nsf.gov/servlets/purl/10211261)
[46](https://arxiv.org/pdf/2502.03674.pdf)
[47](https://pubmed.ncbi.nlm.nih.gov/40491967/)
[48](https://arxiv.org/pdf/2309.06793.pdf)
[49](https://pdfs.semanticscholar.org/6ffc/13ac3a37839cb5fa9efe1aa5e4035af7383c.pdf)
[50](https://arxiv.org/pdf/2101.08013.pdf)
[51](https://arxiv.org/html/2510.24990)
[52](https://pmc.ncbi.nlm.nih.gov/articles/PMC9921606/)
[53](https://www.frontiersin.org/articles/10.3389/fenrg.2023.1288683/full)
[54](https://arxiv.org/pdf/2404.06294.pdf)
[55](https://arxiv.org/pdf/2107.12679.pdf)
[56](https://downloads.hindawi.com/journals/cin/2022/1744969.pdf)
[57](https://www.isprs-ann-photogramm-remote-sens-spatial-inf-sci.net/V-3-2022/603/2022/isprs-annals-V-3-2022-603-2022.pdf)
[58](https://www.mdpi.com/1099-4300/24/8/1030/pdf?version=1659429633)
[59](https://www.isprs-ann-photogramm-remote-sens-spatial-inf-sci.net/V-3-2022/591/2022/isprs-annals-V-3-2022-591-2022.pdf)
[60](https://arxiv.org/abs/2107.09523v1)
[61](https://mcbal.github.io/post/an-energy-based-perspective-on-attention-mechanisms-in-transformers/)
[62](https://www.geeksforgeeks.org/nlp/self-attention-in-nlp/)
[63](https://onlinelibrary.wiley.com/doi/full/10.1155/er/3534500)
[64](https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html)
[65](https://arxiv.org/abs/2511.14691)
[66](https://arxiv.org/pdf/2210.01167.pdf)
[67](https://www.semanticscholar.org/paper/Creating-Temporally-Correlated-High-Resolution-GAN-Shah-Azimian/22ca537ed50b529bd4ecaf41f04502ce968dbadc)
[68](https://arxiv.org/html/2405.19464v1)
[69](https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention)
[70](https://www.osti.gov/pages/biblio/2329473)
