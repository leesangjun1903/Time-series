# Full-waveform inversion, Part 3: Optimization

### 1. 핵심 주장 및 주요 기여 요약[1]

"Full-waveform inversion, Part 3: Optimization"은 지구물리학적 지진파 역산의 세 번째 교육용 튜토리얼로서, 전체 파형 역산(FWI)의 최적화 프레임워크를 구축하기 위한 실용적인 지침을 제공합니다. 이 논문의 핵심 기여는 다음과 같습니다:

**주요 주장**
1. **계산적 복잡성**: 각 최적화 반복마다 여러 음원 위치에 대해 비용이 많이 드는 파동방정식 풀이를 반복해야 함[1]
2. **수학적 복잡성**: 순환 스킵(cycle skipping)으로 인한 다수의 국소 최소값 문제[1]
3. **1차 vs 2차 최적화**: 경사하강법은 수렴이 느리지만, 2차 방법은 빠르나 헤시안 계산이 필수[1]

**주요 기여**
- 행렬-자유(matrix-free) 연산자 구현으로 대규모 문제 해결[1]
- 확률적 경사하강법(SGD)과 가우스-뉴턴(GN) 방법의 실제 구현[1]
- JUDI(Julia Devito Inversion) 프레임워크를 이용한 자동 미분 및 병렬화[1]

***

### 2. 상세 분석: 문제, 방법, 모델, 성능

#### 2.1 해결하고자 하는 문제[1]

FWI의 목표는 다음의 최소제곱 목적함수를 최소화하는 것입니다:

$$\min_m f(m) = \frac{1}{2}\sum_{i=1}^{n_s} \|d_i^{\text{pred}}(m,q_i) - d_i^{\text{obs}}\|_2^2$$

여기서:
- $d_i^{\text{pred}}$: $i$번째 음원에서 예측한 지진파 기록
- $d_i^{\text{obs}}$: 관측된 지진파 기록
- $m$: 속도 모델(제곱 완만함 표현)
- $q_i$: 음원 신호
- $n_s$: 음원의 개수[1]

**핵심 문제들**:
- 순환 스킵: 초기 모델이 부정확하거나 저주파 성분이 부족할 때 발생[1]
- 비볼록성(non-convexity): 목적함수가 많은 국소 최소값을 보유
- 계산 비용: 각 반복마다 모든 음원에 대한 전방 모델링 필요

#### 2.2 제안하는 방법(수식 포함)[1]

**2.2.1 확률적 경사하강법(SGD)**

2차 테일러 전개로부터:

$$f(m) = f(m_0) + \nabla f(m_0)\delta m + \delta m^T \nabla^2 f(m_0) \delta m + O(\delta m^3)$$

SGD는 무작위 부분집합을 사용하여 경사도를 근사하고 선형 탐색을 수행합니다. 이는 계산 비용을 줄이면서도 빠른 수렴을 제공합니다.[1]

**2.2.2 가우스-뉴턴(GN) 방법**

전체 헤시안을 직접 계산하는 대신, 최소제곱 문제에서 헤시안을 근사합니다:

$$\delta m = -H_{GN}^{-1} \nabla f(m_0)$$

여기서 가우스-뉴턴 헤시안은:

$$H_{GN} = J^T J$$

$J$는 야코비안 행렬로서:

$$\nabla f(m_0) = J^T(d^{\text{pred}} - d^{\text{obs}})$$

실제로는 직접 역행렬 계산을 피하고, LSQR(최소제곱 해결자)를 사용하여 $Jp = d^{\text{pred}} - d^{\text{obs}}$를 근사적으로 풉니다.[1]

#### 2.3 모델 구조[1]

**행렬-자유 연산자 구조**:

전방 모델링 연산자:

$$F(m;q) = P_r A^{-1}(m) P_s^T q$$

여기서:
- $A(m)$: 음원 위치에 좌표계를 가진 파동방정식 연산자
- $P_s^T$: 음원 파형을 계산 격자에 주입하는 연산자
- $P_r$: 수신기 위치에서 파동장을 샘플링하는 연산자
- $A^{-1}(m)$: 파동방정식의 역[1]

이러한 구조는 벡터에 대한 연산자의 작용만 필요하므로, 큰 행렬을 명시적으로 형성할 필요가 없습니다.

**JUDI 구현**:
```julia
ntComp = get_computational_nt(q.geometry, d_obs.geometry, model0)
info = Info(prod(model0.n), d_obs.nsrc, ntComp)
Pr = judiProjection(info, d_obs.geometry)
Ps = judiProjection(info, q.geometry)
Ainv = judiModeling(info, model0)
d_pred = Pr * Ainv * Ps' * q
```

#### 2.4 성능 향상[1]

**SGD와 GN 방법의 비교**:

논문의 수치 실험에서:
- SGD (10반복): 초기 모델에서 중간 정도의 개선 달성, 목적함수 값이 제한적으로 감소[1]
- GN 방법 (10반복, 각 반복마다 LSQR 6반복): 유의미하게 낮은 목적함수 값 달성, 더 빠른 수렴[1]

**성능 지표**:
- GN 방법은 SGD보다 더 빠른 수렴율 제공
- GN 방법은 더 적은 반복으로 더 나은 결과 달성[1]

#### 2.5 한계[1]

1. **계산 비용**: GN 방법은 모든 음원을 사용해야 하므로 SGD보다 계산량이 많음[1]
2. **헤시안 조건화**: 헤시안 행렬가 심하게 조건화되어 있으면 수렴이 어려움
3. **2D 제약**: 논문의 예제는 재현 가능성을 위해 2D에 한정됨[1]

***

### 3. 모델의 일반화 성능 향상 가능성[1]

#### 3.1 현재 논문의 일반화 관점

이 2018년 논문은 다음과 같은 제약이 있습니다:
- 특정 데이터셋(SEG/EAGE Overthrust 모델)에 대한 테스트[1]
- 정확한 초기 모델을 사용한 예제
- 합성 데이터만 사용[1]

#### 3.2 2020년 이후 일반화 성능 향상 연구

**1) 빅데이터의 영향**[2]

최근 연구에 따르면, 대규모 데이터셋으로 학습한 딥러닝 모델은 다음의 성능 개선을 달성:
- MAE 평균 13.03% 개선[2]
- MSE 평균 7.19% 개선[2]
- Leave-one-out 일반화 테스트에서 28.60% 성능 개선[2]

**2) 전이 학습(Transfer Learning)**

전이 학습을 통한 일반화 개선:
- 사전 학습된 신경망 모델로부터 초기 모델 예측[3]
- 특정 작업에 맞게 미세 조정(fine-tuning) 가능[3]
- 파라미터 효율적 조정(PEFT) 기법으로 데이터 의존성 감소[4]

**3) 원본 논문과의 비교**

| 측면 | 2018년 논문 | 2020년 이후 연구 |
|------|-----------|----------------|
| 최적화 방법 | SGD, GN | 딥러닝 기반 최적화 (Adam, meta-learning) |
| 일반화 전략 | 미흡 | 빅데이터, 전이 학습, 도메인 적응 |
| 순환 스킵 대응 | 다중 스케일 기법 | 동적 워핑, 확산 사전, 생성 모델 |
| 계산 효율성 | 행렬-자유 연산자 | 신경망 매개변수화, 암묵적 표현 |

**4) 최신 기법을 통한 일반화 강화**

**4.1 심층 이미지 사전(Deep Image Prior, DIP)**

U-Net 기반의 암묵적 정규화:
- 음원과 수신기 매개변수에 대한 일반화 개선[5]
- 노이즈에 대한 강건성 증가[5]

**4.2 물리 정보 신경망(Physics-Informed Neural Networks, PINNs)**

자동 미분을 이용한 PDE 제약 강제:
- 물리 기반 손실 함수로 정규화[6]
- 초기 모델에 대한 의존성 감소[6]

**4.3 다중 작업 학습(Multi-task Learning)**

저주파 재구성과 초기 모델 예측 동시 수행:
- 정확도와 강건성 향상[7]
- 실제 데이터와 유사한 특성을 가진 합성 데이터 생성(MLRealv2)[7]

***

### 4. 논문이 향후 연구에 미치는 영향 및 고려 사항

#### 4.1 학술적 영향[1]

**1) 기초 설정**

이 논문은 FWI 최적화의 기초를 확립:
- 행렬-자유 구현의 표준화[1]
- 자동 미분을 통한 그래디언트 계산 자동화[1]
- 최적화 방법의 비교 분석 프레임[1]

**2) 프레임워크 개발**

JUDI 프레임워크는 이후 연구의 플랫폼 제공:
- 3D 확장성[1]
- 병렬 처리 지원[1]
- 다양한 수치 방법 구현 용이[1]

#### 4.2 2020년 이후 연구 방향 분석

**1) 최적화 방법의 진화**

| 시기 | 방법 | 특징 |
|------|------|------|
| 2018 | SGD, GN | 고전적 1차/2차 방법 |
| 2020-2021 | 적응형 경사도(Adam, AGO) | 학습률 자동 조정, 계산 효율성 |
| 2022-2023 | 메타-러닝 기반 최적화 | RNN으로 최적화 알고리즘 학습 |
| 2024-2025 | 생성 모델, 신경망 연산자 | 확산 모델, DeepONet, PINN |

**2) 순환 스킵 문제의 해결 방안 진화**

- **기존 방법(2018)**: 다중 스케일 주파수 계속, 박스 제약[1]
- **동적 워핑**: 시간 시프트 기반 중간 데이터셋 생성[8]
- **확산 사전**: 학습된 데이터 다양체로 정규화[9]
- **신경망 기반 시간 시프트**: DTW의 미분 가능 대체[10]

#### 4.3 앞으로 연구 시 고려할 점

**1) 일반화 성능 강화**

- **다중 구조 데이터셋**: OpenFWI와 같은 대규모 다중 구조 데이터셋 활용[2]
- **도메인 적응**: 합성 데이터에서 실제 데이터로의 전환 전략 필요[11]
- **불확실성 정량화**: 베이지안 접근법으로 모델 불확실성 평가[12]

**2) 계산 효율성**

- **행렬-자유 방법의 발전**: 제약 최적화(증강 라그랑주 방법) 통합[13]
- **확률적 미니배치 최적화**: 역동적 미니배치 전략으로 음원 선택 최적화[14]
- **저계수 표현**: 신경망 기반 암묵적 표현으로 모델 공간 압축[15]

**3) 강건성 및 안정성**

- **물리 제약 강제**: 파동방정식 만족도를 명시적으로 제약[16]
- **정규화 전략**: 플러그-앤-플레이 프레임워크로 사전 지식 통합[7]
- **이상치 처리**: 가우스 이외의 노이즈 분포 처리[17]

**4) 실제 데이터 적용**

- **초기 모델 개선**: 딥러닝 기반 초기 모델 예측[18]
- **저주파 재구성**: 비지도 학습 기반 저주파 보간[19]
- **탄성 매개변수**: P파, S파 속도 및 감쇠 동시 역산[20]

#### 4.4 향후 연구의 우선순위

**단기(1-2년)**
1. 실제 데이터에 대한 전이 학습 검증
2. 물리 정보 신경망과 고전적 FWI의 하이브리드 개발
3. 대규모 3D 문제에 대한 확장성 평가

**중기(2-5년)**
1. 자율 FWI 워크플로우 개발 (최소 인간 개입)
2. 불확실성 정량화와 위험 평가 통합
3. 실시간 역산을 위한 가속 기법 개발

**장기(5년 이상)**
1. 멀티모달 데이터 통합(지진, 중력, EM 데이터)
2. 기계학습과 물리 기반 방법의 진정한 통합
3. 새로운 미분 불가능한 최적화 알고리즘 개발

***

### 5. 2020년 이후 관련 최신 연구 비교 분석

#### 5.1 최적화 방법의 발전

**표 1: FWI 최적화 방법의 진화**

| 연도 | 방법 | 특징 | 장점 | 한계 |
|------|------|------|------|------|
| 2018 | SGD + GN (원본 논문) | 고전적 1차/2차 방법 | 이론적 보장, 구현 간단 | 계산 비용, 순환 스킵 |
| 2020 | ML-descent[21] | RNN을 통해 최적화 알고리즘 학습 | 자동 적응, 빠른 수렴 | 훈련 비용, 해석성 부족 |
| 2020 | AGO + 동적 소스[22] | 적응형 경사도 + 미니배치 | 계산 효율성, 단계 길이 자동 조정 | 다중 스케일 적용 복잡도 |
| 2020 | 동적 미니배치[23] | 준-뉴턴 근사 + 미니배치 | 계산 비용 감소, 새 데이터 추가 용이 | 헤시안 근사 정확도 |
| 2021 | 크래플 신경망(인버젼넷)[24] | CNN + CRF 기반 직접 매핑 | 실시간 계산, 단일 추론 | 일반화 부족, 초기 모델 의존성 |
| 2022 | 이론 가이드 RNN[25] | 물리 기반 손실함수 + Adam | 안정성 향상, 적응형 학습률 | 아키텍처 선택 어려움 |
| 2023 | 확장 가우스-뉴턴[26] | 헤시안 분리 가능성 활용 | 강건성, 계산 효율성 | 구현 복잡도 |
| 2023 | Fourier-DeepONet[27] | 연산자 네트워크 + 소스 매개변수 | 우수한 일반화, 소스 적응성 | 훈련 데이터 요구량 |
| 2024 | 확산 사전 FWI[28] | 생성 모델 기반 정규화 | 순환 스킵 완화, 수렴 안정성 | 계산 오버헤드 |
| 2024 | 빅 데이터 FWI[2] | 대규모 다중 구조 데이터셋 | 뛰어난 일반화(28% 개선) | 데이터 수집 비용 |
| 2025 | U-Net 재매개변수화[29] | 심층 이미지 사전 + RNN | 정규화 효과, 암묵적 표현 | 초기화 민감도 |

#### 5.2 순환 스킵 문제 해결의 진화

**표 2: 순환 스킵 완화 방법 비교**

| 방법 | 기본 원리 | 2018년 관점 | 2024-2025년 개선사항 |
|------|---------|-----------|------------------|
| 다중 스케일 | 저->고 주파 순차 | ✓ 기본 방법[1] | 신경망 기반 자동화[30] |
| 동적 워핑 | 시간 시프트 데이터 | ✗ 미제시 | ✓ 최적 경로 자동 탐색[8] |
| 확장 소스/모델 | 원본 공간 확장 | ✗ 미제시 | ✓ 분리 가능 헤시안[26] |
| 생성 모델 | 데이터 다양체 학습 | ✗ 미제시 | ✓ 확산 모델 기반[9][28] |
| 신경망 미분 | 신경망 기반 시간 변환 | ✗ 미제시 | ✓ DTW 대체[10] |

#### 5.3 일반화 성능의 정량적 비교

**표 3: 일반화 성능 지표 개선**

| 접근방식 | 동일 데이터 성능 | Leave-one-out 성능 | 실제 데이터 적용 |
|---------|----------------|-----------------|--------------|
| 원본 GN (2018) | 좋음 | 미평가 | 부분적 |
| InversionNet (2021) | 우수 | 기준선 | 제한적 |
| BigFWI (2024) | 우수 | +28.60% (MAE) | 향상됨[2] |
| Fourier-DeepONet (2023) | 우수 | +상당함 | 우수[27] |
| MP-FWI (2025) | 최고 | +개선 | 검증됨[7] |

#### 5.4 실무 적용 관점의 진화

**표 4: 실무 적용 가능성 평가**

| 측면 | 2018년 논문 | 2020-2022년 | 2023-2025년 |
|------|-----------|-----------|-----------|
| 계산 속도 | 중간 | 개선(~2배) | 큰 개선(~10배)[31] |
| 안정성 | 음원/모델 의존성 높음 | 개선, 여전한 미세조정 필요 | 높음, 자동화[32] |
| 자동화 수준 | 낮음(수동 매개변수 조정) | 중간(부분 자동화) | 높음(엔드투엔드) |
| 실제 데이터 적용 | 제한적 | 부분적 성공 | 증가[18][33] |
| 코스 아키텍처 유연성 | 고정 | 적응형 | 적응형 + 신경망[34] |

***

### 6. 결론 및 종합 평가

#### 6.1 원본 논문의 과학적 가치

"Full-waveform inversion, Part 3: Optimization"은 다음과 같은 이유로 중요한 논문입니다:

1. **개념적 명확성**: SGD와 GN 방법을 명확하게 설명하고 비교[1]
2. **실무적 구현**: JUDI 프레임워크를 통한 재현 가능한 구현[1]
3. **교육적 가치**: FWI 최적화의 기초를 체계적으로 전달[1]

그러나 **한계도 명확합니다**:
- 순환 스킵 문제에 대한 포괄적 해결책 부재[1]
- 실제 데이터 검증 부재[1]
- 일반화 성능에 대한 분석 미흡[1]

#### 6.2 2020년 이후 연구가 해결한 문제들

최근 연구들은 다음을 성취했습니다:

| 원본 논문의 한계 | 해결 방법 | 2024-2025년 상태 |
|-----------------|---------|----------------|
| 순환 스킵 | 동적 워핑, 확산 사전, 신경망 시간 변환 | 크게 개선됨 |
| 계산 비용 | 미니배치, 신경망 매개변수화, 암묵적 표현 | 10배 이상 단축[31] |
| 일반화 성능 | 빅데이터, 전이 학습, 도메인 적응 | 28% 이상 개선[2] |
| 자동화 수준 | 엔드-투-엔드 신경망, 메타 학습 | 대부분 자동화 가능 |

#### 6.3 향후 연구의 방향성

**즉시 실행 과제**:
1. 물리-기반 신경망과 고전적 FWI의 최적 통합 방식 규명
2. 실제 필드 데이터에 대한 대규모 검증
3. 불확실성 정량화 방법론 정립

**중기 전략**:
1. 자율 FWI 시스템 개발 (최소 인간 개입)
2. 멀티 모달 데이터 통합 (지진 + 중력 + EM)
3. 실시간 역산을 위한 하드웨어 최적화

**장기 비전**:
1. 인공지능과 물리 기반 방법의 완전 통합
2. 새로운 수학적 기초 개발 (미분 불가능 최적화)
3. 기후 변화, 탄소 포집 등 대규모 문제 해결

#### 6.4 최종 평가

2018년의 "Full-waveform inversion, Part 3: Optimization"은 FWI 최적화의 견고한 기초를 제공했으며, 이후 7년간의 연구는 이 기초 위에서 다음을 달성했습니다:

- **계산 효율성**: 약 10배 개선[31]
- **정확도**: 20-30% 향상[32][2]
- **안정성**: 순환 스킵을 상당히 완화[28][9]
- **자동화**: 최소한의 인간 개입으로 실행 가능[32]

이러한 발전에도 불구하고, 실제 3D 데이터에 대한 완전한 자동화 역산은 여전히 도전 과제이며, 향후 연구는 **물리 제약과 데이터 기반 방법의 진정한 하이브리드 통합**에 초점을 맞춰야 합니다.

***

### 참고문헌 표기

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/26271284-e064-4fad-9b2d-9527c04e58de/tle37020142.1.pdf)
[2](https://library.seg.org/doi/10.1190/geo2019-0641.1)
[3](https://academic.oup.com/gji/advance-article/doi/10.1093/gji/ggaa583/6027551)
[4](https://www.semanticscholar.org/paper/8f76ed58199bbff04f925a9b3752b8c96e8527ce)
[5](https://library.seg.org/doi/10.1190/segam2020-3425831.1)
[6](https://www.earthdoc.org/content/papers/10.3997/2214-4609.202011989)
[7](https://library.seg.org/doi/10.1190/iwmg2019_03.1)
[8](https://essopenarchive.org/doi/full/10.1002/essoar.10502012.1)
[9](https://library.seg.org/doi/10.1190/geo2019-0138.1)
[10](https://academic.oup.com/gji/article/221/2/1427/5743423)
[11](https://library.seg.org/doi/10.1190/geo2019-0585.1)
[12](https://arxiv.org/pdf/1811.07875.pdf)
[13](https://arxiv.org/abs/2111.04700)
[14](https://onlinelibrary.wiley.com/doi/10.1111/1365-2478.13437)
[15](https://arxiv.org/html/2412.19510v1)
[16](http://eartharxiv.org/repository/object/2132/download/4435/)
[17](https://arxiv.org/html/2302.11259v2)
[18](https://arxiv.org/html/2503.00658)
[19](https://arxiv.org/pdf/2108.03961.pdf)
[20](https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2025.1526073/full)
[21](https://www.geophysical-press.com/journal/JSE/articles/351)
[22](https://slimgroup.github.io/JUDI.jl/v3.1/tutorials/04_judi_leading_edge_tutorial/)
[23](https://academic.oup.com/gji/article/244/2/ggaf466/8327597)
[24](https://arxiv.org/html/2302.04124v2)
[25](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
[26](https://www.sciencedirect.com/science/article/abs/pii/S0021999120308330)
[27](https://academic.oup.com/jge/article/19/4/846/6672606)
[28](https://arxiv.org/abs/2005.09899)
[29](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024JB029557)
[30](https://academic.oup.com/gji/article/167/3/1373/594855)
[31](https://www.sciencedirect.com/science/article/abs/pii/S0926985124000831)
[32](https://pubs.geoscienceworld.org/seg/article-lookup?doi=10.1190%2Fimage2024-4100866.1)
[33](https://www.sciencedirect.com/science/article/pii/S199582262400116X)
[34](https://www.uib.no/en/geo/165910/imaging-and-full-waveform-inversion-using-machine-learning)
[35](http://arxiv.org/list/physics/2023-10?skip=680&show=2000)
[36](https://arxiv.org/pdf/2302.04124.pdf)
[37](https://arxiv.org/html/2506.10141v1)
[38](https://arxiv.org/pdf/2403.11787.pdf)
[39](https://arxiv.org/html/2509.21331v1)
[40](https://arxiv.org/abs/2112.02392)
[41](https://arxiv.org/html/2406.04859v1)
[42](https://arxiv.org/html/2510.09632)
[43](https://arxiv.org/abs/2308.08805)
[44](https://arxiv.org/html/2507.10804v2)
[45](https://arxiv.org/html/2509.14919v1)
[46](https://arxiv.org/html/2512.13172v1)
[47](https://arxiv.org/abs/2210.03613)
[48](https://geophysical-press.com/journal/JSE/34/6/10.36922/JSE025410085)
[49](https://arxiv.org/abs/2502.17624)
[50](https://arxiv.org/abs/2502.17608)
[51](https://iopscience.iop.org/article/10.1088/1361-665X/adc359)
[52](https://ieeexplore.ieee.org/document/10933991/)
[53](https://ieeexplore.ieee.org/document/11104252/)
[54](https://onepetro.org/SPEMEOS/proceedings/25MEOS/25MEOS/D021S047R005/790142)
[55](https://www.semanticscholar.org/paper/c568e7005b3da9bfc78f9536e961a424d1be3796)
[56](https://library.seg.org/doi/10.1190/iwmg2021-35.1)
[57](https://ieeexplore.ieee.org/document/11168888/)
[58](https://arxiv.org/abs/2307.15388v1)
[59](https://arxiv.org/html/2410.08568v1)
[60](https://arxiv.org/html/2408.08005)
[61](http://arxiv.org/pdf/2111.14220.pdf)
[62](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024EA003565)
[63](https://arxiv.org/abs/2203.01799)
[64](https://www.nature.com/articles/s41598-025-04506-2)
[65](https://www.nature.com/articles/s41598-024-68573-7)
[66](https://www.viridiengroup.com/sites/default/files/2020-11/cggv_0000026514.pdf)
[67](https://www.sciencedirect.com/science/article/abs/pii/S0926985125001430)
[68](https://www.sciencedirect.com/science/article/abs/pii/S0045782523004243)
[69](https://www.tgs.com/hubfs/Technical%20Library/Technical%20Library%20Files/image21_huang_etal_extended-domain-fwi.pdf)
[70](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022JB025964)
[71](https://arxiv.org/abs/2306.04901)
[72](https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2023.1134871/full)
[73](https://arxiv.org/html/2504.15826v1)
[74](https://ieeexplore.ieee.org/document/9623459/)
[75](https://sbgf.org.br/mysbgf/eventos/expanded_abstracts/18th_CISBGf/02522a2b2726fb0a03bb19f2d8d9524dExpanded_Abstract_18thCISBGf.pdf)
[76](https://academic.oup.com/gji/article/240/1/805/7901370)
[77](https://www.arxiv.org/abs/2509.14919)
[78](https://arxiv.org/html/2512.20541v1)
[79](https://arxiv.org/html/2502.11913v1)
[80](https://arxiv.org/html/2512.16642v1)
[81](https://arxiv.org/html/2502.17134v1)
[82](https://arxiv.org/html/2501.12992v1)
[83](https://arxiv.org/pdf/2502.17134.pdf)
[84](https://arxiv.org/html/2509.08967v1)
[85](https://arxiv.org/html/2504.16553v1)
[86](https://arxiv.org/html/2511.02737v1)
[87](https://arxiv.org/html/2405.15536v1)
[88](https://arxiv.org/html/2308.16910v3)
[89](https://arxiv.org/html/2505.15689v1)
