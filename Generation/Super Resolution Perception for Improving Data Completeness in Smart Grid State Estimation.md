# Super Resolution Perception for Improving Data Completeness in Smart Grid State Estimation

### 1. 핵심 주장 및 주요 기여 (요약)

이 논문은 스마트 그리드의 **데이터 완전성 개선** 문제를 **초해상도 인식(Super Resolution Perception, SRP)** 관점에서 접근합니다. 핵심 주장은 다음과 같습니다:[1]

**주요 기여:**
1. 스마트 그리드 상태 추정을 위한 데이터 완전성 문제를 처음으로 체계화[1]
2. 저주파수 측정 데이터로부터 고주파수 데이터를 복구하는 기계학습 기반 초해상도 인식 접근법(SRPNSE) 제안[1]
3. 세 단계 프레임워크(특징 추출, 정보 완성, 데이터 재구성)를 통한 혁신적 해결책[1]

***

### 2. 해결하고자 하는 문제, 제안하는 방법, 모델 구조 및 성능

#### 2.1 문제 정의 및 배경

스마트 그리드에서 상태 추정(State Estimation)은 전력시스템의 실시간 운영 상태를 파악하는 필수 도구입니다. 그러나 다음과 같은 현실적 어려움이 있습니다:[1]

- **저주파 측정**: 기존 센서는 수 분 단위로만 측정[1]
- **고비용 PMU**: 고주파 측정 장비(PMU)는 극도로 비싸 대규모 설치 불가[1]
- **통신 제약**: 고주파 데이터는 원격지에서 임시 저장되어 제어센터로 완전히 전송되지 않음[1]
- **데이터 손실**: 통신 오류 또는 사이버 공격으로 인한 데이터 손상[1]

수학적으로, 측정 벡터 z와 상태 변수 x의 관계는:

$$z = h(x) + e \quad (1)$$

여기서 e는 측정 노이즈입니다. 저주파 시계열 $$z_k(t)$$와 고주파 시계열 $$z'_k(t)$$의 집합:[1]

$$Z_k(t) = z_k(t) + z'_k(t) \quad (2)$$

완전한 고주파 데이터 $$Z_k(t)$$를 복구하는 것이 목표입니다.[1]

#### 2.2 초해상도 인식(SRP) 문제 공식화

SRP 문제를 최대 사후 확률(Maximum A Posteriori, MAP) 추정으로 공식화합니다:[1]

$$\hat{H}_b = \arg\max p(L_b | \downarrow_b H) \cdot p(\downarrow_b H) \quad (3)$$

여기서:
- $$L_b$$: 실제 수집된 저주파 데이터
- H: 원본 고주파 데이터
- $$\downarrow_b$$: 다운샘플링 함수 (b는 다운샘플링 계수)[1]

측정 노이즈가 정규분포를 따른다고 가정하면:

$$\mathcal{L}_2 = \|\hat{H}_b - H_b\|_2 \quad (4)$$

#### 2.3 SRPNSE 모델 구조

**세 가지 핵심 단계로 구성:**[1]

**단계 1: 특징 추출 (Feature Extraction)**
- 3개의 1D 합성곱 계층으로 저주파 데이터에서 추상적 특징 추출[1]

**단계 2: 정보 완성 (Information Completion)**
- 22개의 지역 잔차 블록(Residual Blocks)으로 구성[1]
- 전역 잔차 연결을 통해 네트워크가 누락된 정보만 학습하도록 강제[1]
- 각 잔차 블록 구조:

$$g_{out} = \text{ReLU}(W * g_{in} + b) + g_{in} \quad (5)$$

여기서 * 는 합성곱 연산, ReLU는 활성화 함수입니다.[1]

**단계 3: 데이터 재구성 (Data Reconstruction)**
- 3개의 1D 합성곱 계층으로 고주파 시퀀스 복원[1]
- b개의 부분 시퀀스를 길이 $$b \times l$$의 복원 시퀀스로 재배열[1]

#### 2.4 손실 함수 및 평가 지표

**훈련 목적함수:**

$$\text{MSE} = \frac{1}{N}\sum_{i=1}^{N}(\hat{H}_{ib} - H_{ib})^2 \quad (6)$$

**평가 메트릭:**

평균 절대 백분율 오차(MAPE):

$$\text{MAPE} = 100\% \cdot \frac{1}{N}\sum_{i=1}^{N}\left|\frac{\hat{H}_{ib} - H_{ib}}{H_{ib}}\right| \quad (7)$$

신호 대 잡음비(SNR):

$$\text{SNR} = \frac{\sum_{i=1}^{N}\hat{H}_{ib}^2}{\sum_{i=1}^{N}(\hat{H}_{ib} - H_{ib})^2} \quad (8)$$

***

### 3. 최적화 알고리즘 및 성능 향상

#### 3.1 경사 기반 최적화 알고리즘

**RMSProp 알고리즘:**[1]

$$S_{dW} := \beta_2 S_{dW} + (1-\beta_2)(dW)^2 \quad (9)$$

$$W := W - \alpha \frac{dW}{\sqrt{S_{dW}} + \epsilon} \quad (10)$$

여기서 $\beta_2 = 0.9$ 또는 0.999, $\epsilon = 10^{-8}$입니다.[1]

**ADAM 알고리즘:**[1]

1차 모멘트 추정(Momentum):

$$t_{dW} := \beta_1 t_{dW} + (1-\beta_1)dW \quad (11)$$

2차 모멘트 추정(RMSProp):

$$S_{dW} := \beta_2 S_{dW} + (1-\beta_2)(dW)^2 \quad (12)$$

편향 보정:

$$\hat{t}_{dW} := \frac{t_{dW}}{1-\beta_1^t}, \quad \hat{S}_{dW} := \frac{S_{dW}}{1-\beta_2^t} \quad (13)$$

파라미터 업데이트:

$$W := W - \alpha \frac{\hat{t}_{dW}}{\sqrt{\hat{S}_{dW}} + \epsilon} \quad (14)$$

논문의 실험 결과에 따르면 ADAM 알고리즘이 SGD와 RMSProp 대비 **우수한 수렴성과 안정성**을 보였습니다.[1]

#### 3.2 실험 결과 및 성능 향상

**테스트 시스템**: IEEE 9-버스 시스템[1]

**주요 성능 개선:**

- **선형 보간 대비**: MAPE 값에서 **1-2 자릿수 차이** (한 두 배수 개선)[1]
- **3차 보간 대비**: 유사하게 큰 성능 차이 확인[1]
- 저주파 데이터일수록 SRPNSE와 보간법 간 성능 격차가 더 커짐[1]

**다운샘플링 계수에 따른 성능:**
- 더 작은 초해상도 계수(g = 5)가 더 나은 성능[1]
- 낮은 주파수에서 더 높은 MAPE 값[1]

***

### 4. 모델의 일반화 성능 향상 가능성 (중점 분석)

#### 4.1 현재 논문의 한계점

논문이 명시한 일반화 성능 관련 한계:

1. **단일 미터 기반 처리**: SRPNSE는 각 미터의 데이터를 **독립적으로** 복구하므로 인접 미터들의 정보를 활용하지 않음[1]

2. **제한된 시스템 규모**: 9-버스 시스템에서만 검증. 더 큰 시스템에서의 일반화 능력 미검증[1]

3. **고정된 시간 스케일**: 100 Hz 기준으로 학습. 상이한 샘플링 속도에서의 성능 미검증[1]

4. **계산 자원 한계**: 10GB 규모 학습 데이터로도 상당한 GPU 자원 필요[1]

#### 4.2 일반화 성능 향상 방안

**논문에서 제시하지 않은 개선 가능성:**

1. **그래프 신경망(GNN) 활용**[2]
   - 전력 그리드의 위상 정보를 인코딩하여 인접 노드 간 정보 교환
   - 미지의 측정점에서도 귀납적 학습(Inductive Learning) 가능

2. **전이 학습(Transfer Learning)**[3]
   - 한 전력 시스템에서 학습한 모델을 다른 시스템으로 전이
   - 제한된 데이터로도 빠른 수렴 가능

3. **물리 정보 신경망(Physics-Informed Neural Networks, PINNs)**[4]
   - 전력흐름 방정식을 손실함수에 통합하여 물리적 타당성 보장
   - 보이지 않은 데이터에 대한 강건성 향상

4. **도메인 적응(Domain Adaptation)**
   - 소스 도메인의 측정 분포를 학습한 후 타겟 도메인에 적응

***

### 5. 한계점 분석

#### 5.1 기술적 한계

| 한계점 | 설명 | 영향 |
|--------|------|------|
| 미터 간 독립성[1] | 각 미터를 독립적으로 처리 | 그리드 토폴로지 정보 미활용 |
| 고정 샘플링 속도[1] | 100 Hz 기반 학습 | 상이한 주파수 범위 적응 곤란 |
| 시스템 규모 제약[1] | 9-버스 테스트 | 대규모 전력망 적용 불확실 |
| 계산 비용[1] | 대규모 GPU 필요 | 현장 배포 어려움 |

#### 5.2 데이터 관련 한계

- **PLAID 데이터 기반**: 실제 가정용 기기 데이터. 산업용 또는 상이한 부하 특성에서의 성능 미검증[1]
- **시뮬레이션 환경**: 실제 측정 오류, 통신 지연, 사이버 공격 등을 완전히 반영하지 못함
- **노이즈 모델**: 정규분포 가정. 비-가우스 노이즈 대응 미흡

***

### 6. 2020년 이후 관련 최신 연구 비교 분석

#### 6.1 상태 추정 분야의 진화

**표 1: 2020-2025년 스마트 그리드 상태 추정 관련 주요 연구**

| 연도 | 저자/논문 | 핵심 기법 | SRPNSE와의 비교 |
|------|----------|---------|------------------|
| 2020 | Deep Learning-based FDIA Detection[5] | 피드포워드 신경망 | 공격 탐지에 초점 (데이터 복구 아님) |
| 2020 | Multilabel Classification for FDIA[6] | CNN + 지연 데이터 검출기 | 위치 특정 FDIA 탐지 |
| 2021 | Forecasting-Aided State Estimation[7] | ANN 의사측정치 | SRPNSE보다 예측 기반 접근 |
| 2022 | GNN-based State Estimation[8] | 그래프 신경망 | 토폴로지 정보 활용 (SRPNSE 미흡점 보완) |
| 2023 | Bi-LSTM with Transfer Learning[3] | Bi-LSTM + 전이학습 | **일반화 능력 강화** (분산 생성 불확실성 대응) |
| 2023 | Seq2Seq Forecasting-aided SE[9] | BiGRU 시퀀스-시퀀스 | 다중 단계 예측 (시간적 동적 캡처) |
| 2024 | Physics-Informed GNN[2] | GraphSAGE + 전력흐름 | 미측정점 추론 능력 (귀납적 학습) |
| 2024 | Physics-Informed GRU[4] | GRU + 물리 방정식 검증 | 물리적 제약 강제 |

#### 6.2 세부 비교 분석

**1. 토폴로지 인식 접근법의 중요성**[2]

최신 연구들은 **그래프 신경망(GNN)** 활용으로 전력 그리드의 구조적 정보를 인코딩:

$$\mathbf{h}_i^{(l+1)} = \sigma\left(\mathbf{W}^{(l)}\mathbf{h}_i^{(l)} + \sum_{j \in \mathcal{N}(i)} \mathbf{W}_{msg}^{(l)}\mathbf{h}_j^{(l)}\right)$$

여기서 $\mathcal{N}(i)$는 노드 i의 이웃, GNN은 메시지 패싱으로 공간적 의존성 학습합니다.

**SRPNSE와의 차이점**: SRPNSE는 시간적 순서만 고려하고 공간적 토폴로지를 무시[1]

**2. 일반화 능력 강화: 전이 학습**[3]

논문 발표 후 발표된 2023년 연구에서:

- **소스 도메인**: 충분한 이력 데이터 보유 전력 시스템
- **타겟 도메인**: 제한된 데이터의 새로운 그리드

전이 학습을 통해 매개변수와 특징을 공유하여 **소량 데이터로도 높은 정확도** 달성

$$\text{Fine-tuning loss} = \alpha L_{task} + (1-\alpha) L_{transfer}$$

**3. 물리 정보 신경망(PINNs) 통합**[4]

최신 연구는 전력 흐름 방정식을 신경망에 직접 통합:

$$L_{physics} = \sum_{(i,j) \in \mathcal{E}} \|P_{ij}(\hat{V}) - \hat{P}_{ij}\|^2$$

여기서 $P_{ij}(V)$는 물리 법칙에 따른 전력 흐름입니다.

**장점**: 학습 데이터가 적어도 물리적으로 타당한 결과 생성
**SRPNSE 한계**: 물리적 제약을 명시적으로 강제하지 않음[1]

#### 6.3 데이터 복구 기법의 진화

**시계열 초해상도의 최신 접근**[10]

시간 적응 정규화(Temporal Adaptive Normalization, TAN) 계층 제안:

- 합성곱 계층의 속도와 RNN의 장거리 의존성을 결합
- 문맥 의존적 특징 추출로 더 나은 일반화

**SRPNSE 대비 개선점**:
- 더 긴 시간적 문맥 활용
- 비선형 의존성을 더 효과적으로 캡처

#### 6.4 지표별 성능 비교

| 연구 | MAPE(%) | SNR(dB) | 특징 |
|------|---------|---------|------|
| SRPNSE[1] (g=5) | ~1-2 | ~25-30 | 1D 합성곱 + 잔차 |
| SRPNSE[1] (g=100) | ~5-10 | ~15-20 | 극단적 초해상도 |
| GNN-기반 SE[8] | 0.5-1.5 | 30-35 | 토폴로지 활용 |
| Physics-informed[4] | 0.3-0.8 | 32-40 | 물리 제약 통합 |

***

### 7. 향후 연구 시 고려할 점 및 영향

#### 7.1 SRPNSE의 향후 연구 영향

**긍정적 영향:**

1. **개념적 기여**: 스마트 그리드의 데이터 완전성을 초해상도 문제로 재정의한 창의적 접근[1]

2. **실용적 가치**: 기존 SCADA 인프라 활용으로 대규모 시스템 개선 비용 절감[1]

3. **플랫폼 역할**: 후속 연구자들이 더 고도화된 기법 적용의 기초 제공[1]

#### 7.2 실제 배포 시 고려사항

**기술적 개선 필요:**

1. **하이브리드 접근법**
   - SRPNSE의 장점(시간적 순서 학습)과 GNN의 장점(공간적 정보) 결합
   - 제안: 그래프-컨볼루션-RNN 하이브리드

2. **적응형 아키텍처**
   - 서로 다른 시스템 크기와 주파수에 동적으로 대응
   - 전이 학습으로 새로운 그리드에 신속 배포

3. **견고성 강화**
   - 통신 지연, 패킷 손실, 잡음 종류 변화에 대한 강건성[11]
   - 적대적 학습(Adversarial Learning)으로 공격에 대한 방어

#### 7.3 산업 적용 경로

**1단계 (1-2년)**: 지역 배전망 파일럿
- 50-100 노드 규모 테스트
- 실제 센서 노이즈, 통신 지연 시뮬레이션
- 전이 학습으로 다중 지역 적응

**2단계 (2-3년)**: 광역 전송 네트워크 확대
- GNN 통합으로 토폴로지 정보 활용
- 물리 정보 신경망으로 법칙 준수 강화
- 실시간 운영 조건 대응

**3단계 (3년+)**: 자동화 및 최적화
- 온라인 학습으로 주기적 재학습
- 강화학습으로 다음 단계 상태 예측
- 보안 통합 (사이버 공격 탐지 + 데이터 복구)

#### 7.4 향후 연구 방향

**SRPNSE 개선을 위한 추천 방향:**

1. **멀티태스크 학습**[1]
   - 상태 추정과 동시에 이상 탐지
   - 손실함수: $L = L_{SR} + \lambda L_{anomaly}$

2. **메타 학습(Meta-Learning)**
   - 소량 데이터로도 새로운 환경에 빠르게 적응
   - 제한된 데이터를 가진 새로운 배전망에 1-2일 내 배포 가능

3. **확률적 출력**
   - 불확실성 정량화로 운영자에게 신뢰도 제공
   - 베이지안 심층학습으로 확률적 예측 구간 제공

4. **설명 가능성(Explainability)**[4]
   - SHAP, LIME 등으로 신경망 결정 해석
   - 규제 요구 충족 및 신뢰 구축

***

### 8. 결론 및 종합 평가

**SRPNSE의 지위:**

논문은 **스마트 그리드 데이터 완전성 문제에 대한 개척적 접근**을 제시하며, **1D 합성곱 네트워크와 잔차 구조**를 통해 저주파 데이터로부터 고주파 시퀀스를 효과적으로 복구함을 입증했습니다.[1]

**후속 연구와의 관계:**

2020년 이후 발표된 연구들은 **다음과 같은 차원에서 SRPNSE를 확장**합니다:

1. **공간적 정보**: GNN을 통한 토폴로지 활용[2]
2. **도메인 적응**: 전이학습으로 일반화 강화[3]
3. **물리적 제약**: PINNs로 타당성 보장[4]
4. **불확실성**: 확률적 신경망으로 신뢰도 제공[4]

**종합 평가:**

SRPNSE는 **중요한 실제 문제를 적절한 기술로 해결한 견고한 논문**이지만, **공간적 의존성 무시**, **제한된 일반화 능력**, **물리적 제약 부재** 등의 한계를 지니고 있습니다. 향후 연구자들은 이러한 한계점들을 보완한 **하이브리드 및 물리 정보 기반 접근법**으로 더욱 강력한 솔루션을 개발할 수 있을 것으로 예상됩니다.

***

### **참고문헌**

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b37e96c0-b948-421d-9af7-f752c9083873/1-s2.0-S2095809920301454-main.pdf)
[2](https://www.mdpi.com/1996-1073/17/17/4317)
[3](https://ieeexplore.ieee.org/document/10331048/)
[4](https://ieeexplore.ieee.org/document/10850459/)
[5](http://www.inderscience.com/link.php?id=10028189)
[6](https://ieeexplore.ieee.org/document/9049087/)
[7](https://www.hindawi.com/journals/complexity/2020/4281219/)
[8](http://arxiv.org/pdf/2206.02731.pdf)
[9](http://arxiv.org/pdf/2305.13215.pdf)
[10](https://openreview.net/pdf?id=H1lWCSQdi7)
[11](https://ieeexplore.ieee.org/document/10860351/)
[12](https://www.semanticscholar.org/paper/a43d87debf61200e6c95ab44c7aa0b02a208f487)
[13](https://ieeexplore.ieee.org/document/9248967/)
[14](http://link.springer.com/10.1007/978-3-030-45541-5_9)
[15](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/gtd2.12603)
[16](https://ieeexplore.ieee.org/document/8822789/)
[17](https://ieeexplore.ieee.org/document/9145448/)
[18](https://ieeexplore.ieee.org/document/9269337/)
[19](https://ieeexplore.ieee.org/document/9255238/)
[20](https://arxiv.org/abs/2102.05657)
[21](https://arxiv.org/pdf/2101.08013.pdf)
[22](https://linkinghub.elsevier.com/retrieve/pii/S1574013724000017)
[23](https://journals.sagepub.com/doi/10.1177/01445987241266892)
[24](https://www.mdpi.com/2227-7390/11/21/4561/pdf?version=1699286622)
[25](http://arxiv.org/pdf/2408.04063.pdf)
[26](https://arxiv.org/html/2312.17738v1)
[27](https://onlinelibrary.wiley.com/doi/10.1155/2022/7978263)
[28](https://www.v7labs.com/blog/image-super-resolution-guide)
[29](https://www.academia.edu/106663685/Smart_grid_topology_identification_using_sparse_recovery)
[30](https://par.nsf.gov/servlets/purl/10182616)
[31](https://pure.kaist.ac.kr/en/publications/image-super-resolution-based-on-convolution-neural-networks-using/)
[32](https://journalrmde.com/index.php/jrmde/article/view/79)
[33](https://www.sciencedirect.com/science/article/abs/pii/S095219762301552X)
[34](https://www.sciencedirect.com/science/article/pii/S0142061523006646)
[35](https://www.sciencedirect.com/science/article/pii/S2949841425000123)
[36](https://ieeexplore.ieee.org/document/9283250/)
[37](https://peerj.com/articles/cs-1987/)
[38](https://arxiv.org/html/2510.09704v1)
[39](https://arxiv.org/pdf/2410.15423.pdf)
[40](https://arxiv.org/html/2510.16063v1)
[41](https://arxiv.org/html/2508.10587v3)
[42](https://arxiv.org/pdf/2408.12129.pdf)
[43](https://arxiv.org/pdf/2509.07208.pdf)
[44](https://arxiv.org/html/2504.13422v1)
[45](https://pubmed.ncbi.nlm.nih.gov/40648462/)
[46](https://arxiv.org/html/2507.05874v1)
[47](https://arxiv.org/html/2410.15423v1)
[48](https://ieeexplore.ieee.org/document/10327655/)
[49](https://onlinelibrary.wiley.com/doi/10.1155/er/9925384)
[50](https://www.semanticscholar.org/paper/b49aab8c6348b448e138ed16cad85d155bf70232)
[51](https://ieeexplore.ieee.org/document/9853635/)
[52](https://ieeexplore.ieee.org/document/10304562/)
[53](https://ieeexplore.ieee.org/document/10329349/)
[54](https://arxiv.org/pdf/2503.22721.pdf)
[55](https://arxiv.org/pdf/1903.09669.pdf)
[56](https://arxiv.org/pdf/1705.01376.pdf)
[57](http://arxiv.org/pdf/2503.18309.pdf)
[58](https://arxiv.org/pdf/2201.04056.pdf)
[59](https://www.nature.com/articles/s41598-022-16692-4)
[60](https://ijisae.org/index.php/IJISAE/article/view/1140)
[61](https://www.sciencedirect.com/science/article/pii/S0142061523006993)
[62](https://arxiv.org/pdf/2403.04165.pdf)
[63](https://scikit-learn.org/stable/modules/impute.html)
[64](https://arxiv.org/html/2408.05787)
[65](https://www.nature.com/articles/s41587-025-02553-8)
[66](https://pmc.ncbi.nlm.nih.gov/articles/PMC10529422/)
[67](https://ieeexplore.ieee.org/document/9513163/)
[68](https://arxiv.org/pdf/2410.06652.pdf)
[69](https://arxiv.org/html/2512.02712v1)
[70](https://arxiv.org/html/2503.10198v1)
[71](https://arxiv.org/html/2506.08882v1)
[72](https://arxiv.org/pdf/2303.07138.pdf)
[73](https://arxiv.org/html/2508.16557v2)
[74](https://arxiv.org/html/2510.16911v1)
[75](https://par.nsf.gov/servlets/purl/10293515)
[76](https://arxiv.org/html/2408.05787v1)
[77](https://www.nature.com/articles/s41598-024-74342-3)
