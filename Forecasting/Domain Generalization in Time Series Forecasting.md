# Domain Generalization in Time Series Forecasting

### 1. 핵심 주장 및 주요 기여

"Domain Generalization in Time Series Forecasting" 논문은 시계열 데이터에서 **도메인 일반화(domain generalization)**라는 근본적인 문제를 처음 체계적으로 다룬다. 논문의 핵심 주장은 다음과 같다:[1]

첫째, 기존 도메인 일반화 방법들(컴퓨터 비전, NLP에서 발전)이 시계열 예측에 직접 적용될 수 없다는 것이다. 이는 시계열 데이터의 복잡한 시간 의존성과 도메인 간 상이한 데이터 분포 때문이다.

둘째, 저자들은 **도메인 불일치 정규화(domain discrepancy regularization)**라는 단순하면서도 효과적인 방법을 제시한다. 핵심 아이디어는 "상이한 도메인들 간에 예측 성능의 큰 차이가 나면 안 된다"는 제약 조건을 도입하는 것이다.

### 2. 문제 정의 및 제안하는 방법

#### 2.1 문제 설정

시계열 예측 문제는 다음과 같이 정의된다:[1]

$$P(Y|X) = P(y_{1:T} | y_{1:T}, a_{1:T}) = \prod_{t=T+1}^{T+h} P(y_t | y_{1:t-1}, a_{1:t-1}; \theta)$$

여기서 $y_{1:T}$는 역사적 관측값, $a_{1:T}$는 외생 속성, $T$는 과거 시간 단계 수, $h$는 예측 지평(forecasting horizon)이다.

도메인 일반화 문제는 $M$개의 훈련 도메인 $D_{train} = \{D_k\}\_{k=1}^M$에서 학습하여 $K-M$개의 보이지 않는 테스트 도메인 $D_{test} = \{D_k\}_{k=M+1}^K$에서 잘 일반화하는 예측 함수 $F: X \rightarrow Y$를 학습하는 것이다.[1]

#### 2.2 데이터 가정

논문은 두 가지 핵심 가정을 제시한다:[1]

**가정 1 (공통 기저 패턴)**:
$$\epsilon_l \leq ||P_{k_1}(X,Y) - P_{k_2}(X,Y)|| \leq \epsilon_u, \quad 1 \leq k_1 \neq k_2 \leq K, \quad 0 \leq \epsilon_l \leq \epsilon_u$$

이 가정은 도메인들 간에 공통 패턴이 존재하되, 완전히 동일하지는 않음을 의미한다. 소매, 기상, 환경 모니터링 데이터에서 반복되는 계절 패턴이 예시이다.[1]

**가정 2 (급격한 분포 변화 없음)**:
$$|\Delta D_k^t| \leq \theta_s, \quad \forall t \in T, \quad \forall k \in K$$

여기서 $\Delta D_k^t$는 도메인 $k$의 시간 $t$에서의 분포 변화를 나타낸다. 이 가정은 각 도메인 내에서 점진적 변화만 있고 급격한 변화가 없음을 보장한다.[1]

#### 2.3 제안하는 정규화 방법

**기본 도메인 불일치 정규화 (Domain Discrepancy Regularization)**:

$$R_{DD} = \sum_{k_1,k_2} d_H(D_{k_1}, D_{k_2}) \cdot d_{L_{fcst}}(D_{k_1}, D_{k_2})$$

여기서:[1]

- $d_H(D_{k_1}, D_{k_2}) = MMD(H(D_{k_1}), H(D_{k_2}))$ : 최대 평균 불일치(Maximum Mean Discrepancy)를 이용한 분포 발산 측정
  
- $H(D_k)$는 도메인 $k$의 고수준 표현(RNN의 은닉 상태, CNN의 특성 맵)
  
- $d_{L_{fcst}}(D_{k_1}, D_{k_2}) = |Mean(L_{fcst}(D_{k_1})) - Mean(L_{fcst}(D_{k_2}))|^p$ : 도메인 간 예측 손실의 유클리드 거리

**도메인 난이도 인식 확장 버전**:

$$R_{DDD} = \sum_{k_1,k_2} d_H(D_{k_1}, D_{k_2}) \cdot d_{L_{fcst}}(D_{k_1}, D_{k_2}) \cdot \alpha(D_{k_1}, D_{k_2})$$

여기서:[1]

$$\alpha(D_{k_1}, D_{k_2}) = 1 + Std(L_{fcst}(D_{k_1})) + Std(L_{fcst}(D_{k_2}))$$

이 확장은 손실값의 표준편차가 높은(즉, 훈련하기 어려운) 도메인에 더 약한 패널티를 적용한다.[1]

#### 2.4 전체 학습 목적함수

$$L = L_{fcst} + \lambda \cdot R_{DD} \text{ 또는 } L_{fcst} + \lambda \cdot R_{DDD}$$

여기서 $\lambda$는 정규화 항과 기본 예측 손실 사이의 균형을 조절하는 하이퍼파라미터이다.[1]

### 3. 모델 구조 및 구현

#### 3.1 기본 모델

논문은 두 가지 대표적인 시계열 예측 모델을 기본 모델로 사용한다:[1]

**DeepAR** (RNN 기반, 확률적 예측):
- 3개의 은닉 레이어를 가진 LSTM 구조
- 정규화된 입력에 대해 가우시안 분포를 출력
- 장기 의존성 포착에 효과적

**WaveNet** (CNN 기반, 인과적 뉘앙스 있는 시간 컨볼루션):
- 5개의 은닉 레이어, 커널 크기 9
- 병렬화 가능한 빠른 훈련
- 다양한 시간 스케일에서 패턴 포착

#### 3.2 정규화 항 계산

**최대 평균 불일치 (MMD)** 계산:
- 제곱 선형 MMD 사용으로 효율성과 효과성 확보
- 도메인 간 표현 공간에서의 분포 차이 정량화

**손실 기반 성능 차이**:
- 배치 단위의 평균 손실 계산
- 도메인 간 이 평균들의 $L^p$ 거리 (p=1 또는 2)

### 4. 성능 향상 및 일반화 메커니즘

#### 4.1 합성 데이터 성능

4개의 합성 데이터셋(NT-P, PT-N, PN-T, T-PN)에서 Cedar의 성능:[1]

| 데이터셋 | 모델 | Cedar (Q0.5) | MMD (Q0.5) | 기본 (Q0.5) |
|---------|------|-------------|-----------|-----------|
| PT-N | DeepAR | 0.3843 | 0.4298 | 0.4245 |
| PN-T | DeepAR | 0.0389 | 0.0389 | 0.0401 |
| NT-P | WaveNet | 0.6817 | 0.6855 | 0.8671 |

Cedar는 대부분의 경우 MMD(순수 분포 정렬)보다 우수하며, 기본 모델도 개선한다.[1]

#### 4.2 실제 데이터 성능

4개의 실제 데이터셋 결과:[1]

| 데이터셋 | 도메인 수 | Cedar-DeepAR (Q0.5) | Cedar-WaveNet (Q0.5) | 기본 (Q0.5) |
|---------|---------|------------------|------------------|-----------|
| Favorita-cat | 26 | 0.0908 | 0.0884 | 0.1461 |
| US-traffic | 19 | 0.0757 | 0.0735 | 0.0894 |
| Stock-volume | 12 | 0.2098 | 0.2322 | 0.2415 |

Cedar는 일관되게 기본 모델을 능가한다.[1]

#### 4.3 일반화 성능 향상의 원리

**1. 도메인 간 성능 일관성 강화**
- 정규화 항이 도메인 간 예측 손실의 분산을 최소화
- 한 도메인에 과도하게 맞추는 것을 방지
- 보이지 않는 도메인에 대한 로버스트함 증가

**2. 도메인 난이도 인식의 역할**
- 이상치나 변동성이 큰 도메인에 더 약한 패널티 적용
- 모델이 복잡한 패턴을 더 유연하게 학습
- 특히 금융, 의료 데이터처럼 변동성이 큰 영역에서 효과적

**3. 최대 평균 불일치의 역할**
- 고수준 표현 공간에서 도메인 간 분포 차이 포착
- 단순 통계량이 아닌 커널 방법으로 복잡한 관계 모델링

#### 4.4 계산 효율성

- **DeepAR 기반**: 훈련 시간 증가 약 9% 미만
- **WaveNet 기반**: 훈련 시간 증가 약 10% 미만
- 대규모 데이터셋(100K 시간 단위)에서도 선형적 확장성[1]

### 5. 한계 및 제약사항

#### 5.1 가정 위반 시 성능 저하

**가정 1 위반 (도메인 간 극도로 유사한 패턴)**:
- UCI 공기질 데이터셋에서 12개 모니터링 지점이 매우 유사한 패턴 보임
- Cedar의 성능이 기본 모델보다 낮음
- 도메인 일반화 모델 자체가 필요하지 않은 상황[1]

**가정 2 위반 (급격한 분포 변화)**:
- 합성 데이터에 임의의 평균 이동 도입 시 성능 저하
- DeepAR에서 상대적으로 작은 영향, WaveNet에서 심각한 성능 저하
- 도메인 난이도 인식이 훈련 도메인의 특성을 반영하지 못함[1]

#### 5.2 도메인 난이도 정량화의 한계

- 현재는 손실값의 표준편차만 사용
- 도메인 전문가 지식, 외생 변수 분석, 고급 도메인별 지표 필요
- 시계열의 복잡한 특성(계절성, 추세, 노이즈)을 종합적으로 반영하지 못함[1]

### 6. 2020년 이후 관련 최신 연구 비교

#### 6.1 주요 관련 연구

**2022년**: Domain Adaptation for Time Series Forecasting via Attention Sharing (DAF)[2]
- 도메인 불변 특성(쿼리, 키)과 도메인 특정 특성(값)을 구분
- 소스 도메인의 정보를 활용하면서 타겟 도메인 특화

**2023년**: Prompting-based Temporal Domain Generalization[3]
- 매개변수 효율적인 프롬프트 기반 접근
- 시간 드리프트에 대한 적응형 학습

**2024년 (최신)**:

*Domain Generalization for Time-Series Forecasting via Extended Domain-Invariant Representations*[4]
- 입력과 출력 공간 모두에서 도메인 불변 표현 추출
- CEDAR보다 최대 8% 정확도 향상 보고

*Learning Latent Spaces for Domain Generalization in Time Series Forecasting*[5]
- 분해 기반 아키텍처 + Conditional β-VAE
- 추세-순환 및 계절 성분을 독립적으로 모델링
- 시간 의존성을 지배하는 잠재 인수 발견

*Domain Fusion Controllable Generalization (TimeControl)*[6]
- 확산 모델 기반의 도메인 융합 패러다임
- 여러 도메인의 정보를 통합 생성 프로세스로 통합
- 임의 길이의 예측 시퀀스 생성 가능

*Towards a General Time Series Forecasting Model with Unified Representation and Adaptive Transfer*[7]
- 주파수 기반 마스킹으로 다양한 도메인의 통일된 표현 추출
- Time Series Register로 도메인 특정 표현 캡처
- 7개 벤치마크에서 최첨단 성능, 제로샷 능력

#### 6.2 기술적 비교

| 방법 | 주요 특징 | 장점 | 제약사항 |
|-----|---------|------|--------|
| **CEDAR** (2024) | 정규화 기반, 도메인 난이도 인식 | 간단, 저오버헤드, 여러 모델 적용 가능 | 공통 패턴 가정 필요 |
| **DAF** (2022) | 어텐션 기반, 도메인별 값 | 도메인 적응 효과적 | 작은 타겟 데이터 필요 |
| **Domain-Invariant Output** (2024) | 입출력 표현 정렬 | 8% 정확도 향상 | 계산 복잡도 증가 |
| **Latent β-VAE** (2024) | 분해 + 변분 추론 | 잠재 인수 해석성 | 모델 복잡도 높음 |
| **TimeControl** (2024) | 확산 모델, 도메인 융합 | 유연한 길이 출력, 강력한 일반화 | 계산 비용 높음 |
| **Unified Representation** (2024) | 주파수 분해 + 레지스터 | 제로샷 능력, 최첨단 성능 | 사전학습 필요 |

#### 6.3 CEDAR의 위치와 기여

CEDAR는 도메인 일반화 시계열 예측 분야의 **초기 기초 연구(foundation work)**이다. 이후 연구들은 다음 방향으로 진화했다:[1]

1. **표현 학습의 고도화**: 단순 MMD에서 입출력 공간 정렬, 잠재 공간 분해로 진화
2. **생성 모델 도입**: 확산 모델, VAE 등을 통한 보다 정교한 도메인 통합
3. **자기 지도 학습 활용**: 주파수 분해, 프롬프트 기반 접근
4. **확장성 향상**: 기초 모델, 제로샷 학습 능력 추가

### 7. 향후 연구의 중요 고려사항

#### 7.1 이론적 심화

1. **도메인 간 거리 경계**: 일반화 오류에 대한 이론적 경계 제시
2. **최적 정규화 가중치 선택**: $\lambda$ 값의 이론적 정당화
3. **가정의 필요성과 충분성**: 공통 패턴 가정의 형식화

#### 7.2 방법론적 개선

1. **적응형 정규화**: 훈련 과정에서 동적으로 $\lambda$ 조정
2. **고급 도메인 난이도 측정**: 
   - 샘플 복잡도 이론 활용
   - 정보 이론적 지표
   - 도메인 특정 메트릭
3. **다중 스케일 학습**: 다양한 예측 지평에 대한 적응

#### 7.3 응용 확대

1. **이상 탐지**: 도메인 불일치를 이용한 이상 시점 감지
2. **능동 학습**: 어느 도메인의 라벨을 먼저 획득할지 결정
3. **시뮬레이션 투 현실 (Sim-to-Real)**: 물리 시뮬레이션 데이터로 현실 시스템 예측

#### 7.4 실무적 고려사항

1. **비교 불능한 시간 스케일 처리**: 일일/시간/분 단위가 섞인 경우
2. **외생 변수의 역할**: 계절성, 휴일 등의 통합
3. **온라인 학습 시나리오**: 연속 데이터 수신 상황에서의 도메인 적응

### 결론

"Domain Generalization in Time Series Forecasting" 논문은 시계열 예측에서 도메인 일반화라는 근본적이면서도 실용적인 문제를 체계적으로 정의하고, **간단하지만 효과적인 정규화 기반 해결책**을 제시했다.[1]

CEDAR의 핵심 강점은:
- **단순성**: 기존 모델에 쉽게 통합 가능
- **효율성**: 10% 미만의 계산 오버헤드
- **일관성**: 합성 및 실제 데이터에서 체계적인 성능 향상
- **일반성**: 다양한 기본 모델에 적용 가능

이후 2024년 최신 연구들(β-VAE, TimeControl, 통합 표현)은 더 정교한 표현 학습과 생성 모델을 도입하여 성능을 향상시키고 있으나, CEDAR는 여전히 **산업 응용에서 빠른 프로토타이핑과 배포에 가장 실용적인 선택**으로 평가된다.[1]

향후 도메인 일반화 시계열 연구는 **이론적 기초 강화**, **적응형 메커니즘 개발**, **다양한 응용 시나리오 확대**의 세 방향으로 진행될 것으로 예상된다.

***

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2156506b-a771-4558-a38d-6736b8ab34e6/3643035.pdf)
[2](https://arxiv.org/pdf/2102.06828v2.pdf)
[3](https://arxiv.org/html/2310.02473v2)
[4](https://ieeexplore.ieee.org/document/10677579/)
[5](https://arxiv.org/abs/2412.11171)
[6](https://www.semanticscholar.org/paper/dcb7ca9c44181ee9516713160d5f42aa4488a12e)
[7](https://www.semanticscholar.org/paper/39e0e964d6ba714584c6fd58e170fc36370da6a6)
[8](https://dl.acm.org/doi/10.1145/3643035)
[9](https://ieeexplore.ieee.org/document/10352135/)
[10](https://link.springer.com/10.1007/s00521-023-09047-1)
[11](https://arxiv.org/abs/2410.09836)
[12](https://link.springer.com/10.1007/s11227-023-05859-z)
[13](https://www.scitepress.org/DigitalLibrary/Link.aspx?doi=10.5220/0013395500003890)
[14](https://dl.acm.org/doi/pdf/10.1145/3643035)
[15](https://arxiv.org/html/2503.01157v1)
[16](http://arxiv.org/pdf/2412.11171.pdf)
[17](https://arxiv.org/pdf/2503.04118.pdf)
[18](https://arxiv.org/pdf/2310.10688.pdf)
[19](http://arxiv.org/pdf/2410.15217.pdf)
[20](https://s-space.snu.ac.kr/handle/10371/209786)
[21](http://www.wanghao.in/paper/ICML22_DAF.pdf)
[22](https://www.diva-portal.org/smash/get/diva2:1647721/FULLTEXT01.pdf)
[23](https://pure.uva.nl/ws/files/174241098/Domain_Generalization_in_Time_Series_Forecasting.pdf)
[24](https://www.sciencedirect.com/science/article/abs/pii/S0378778825002439)
[25](https://www.sciencedirect.com/science/article/abs/pii/S0020025522001104)
[26](https://arxiv.org/abs/2502.16637)
[27](https://www.sciencedirect.com/science/article/abs/pii/S0031320320304209)
[28](https://dl.acm.org/doi/full/10.1145/3643035)
[29](https://arxiv.org/html/2412.03068v2)
[30](https://arxiv.org/html/2503.03594v1)
[31](https://pubmed.ncbi.nlm.nih.gov/39736527/)
[32](https://arxiv.org/html/2412.11171v1)
[33](https://arxiv.org/html/2508.07195v1)
[34](https://arxiv.org/html/2410.11539v1)
[35](https://arxiv.org/html/2509.26045v1)
[36](https://arxiv.org/html/2509.19465v2)
[37](https://www.semanticscholar.org/paper/5f7e322b845f7d65c7a97b75c2bbe108d58de9ca)
[38](https://deem.berlin/publication/domain-generalisation-for-timeseries-forecasting-tkde/)
