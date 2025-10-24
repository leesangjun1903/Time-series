# RAINCOAT : Domain Adaptation for Time Series Under Feature and Label Shifts

## 핵심 주장과 주요 기여

**RAINCOAT (fRequency-augmented AlIgN-then-Correct for dOmain Adaptation for Time series)** 는 시계열 데이터에서 feature shift와 label shift를 동시에 해결하는 최초의 domain adaptation 모델입니다.[1]

### 주요 기여점:
1. **시간-주파수 이중 인코딩**: 기존 방법들이 시간 영역 특성만 고려한 반면, RAINCOAT은 DFT를 통해 주파수 영역 특성을 추가로 인코딩합니다[1]
2. **Sinkhorn 발산 기반 정렬**: 주파수 특성의 disjoint support 문제를 해결하기 위해 Sinkhorn divergence를 도입했습니다[1]
3. **Align-then-Correct 전략**: Universal domain adaptation을 위한 새로운 2단계 접근법을 제안했습니다[1]
4. **통합 프레임워크**: Closed-set과 Universal domain adaptation을 모두 지원하는 첫 번째 시계열 DA 방법입니다[1]

## 해결하는 문제와 제안 방법

### 해결하는 문제
시계열 domain adaptation의 핵심 문제는 다음과 같습니다:
- **Feature shift**: $$p_s(x) \neq p_t(x) $$이지만 $$p_s(y|x) = p_t(y|x) $$ [1]
- **Label shift**: $$p_s(y) \neq p_t(y) $$이며, target domain에 source에 없는 private label들이 존재[1]

### 수학적 모델링

#### 1. 시간-주파수 특성 추출
DFT를 통한 주파수 변환:

$$
v[m] = \sum_{t=0}^{T-1} x[t] \cdot e^{-i2\pi \frac{mt}{T}}
$$

진폭과 위상 추출:

$$
a[m] = \frac{|v[m]|}{T} = \frac{\sqrt{\text{Re}(v[m])^2 + \text{Im}(v[m])^2}}{T}
$$

$$
p[m] = \text{atan2}(\text{Im}(v[m]), \text{Re}(v[m]))
$$

최종 표현: $$z_i = [e_{F,i}; e_{T,i}] $$ (주파수 특성과 시간 특성의 결합)[1]

#### 2. Sinkhorn 발산 기반 도메인 정렬

$$
S_\eta(\mu, \nu) = \min_{P \in \Pi(\mu,\nu)} \{\langle C, P \rangle + \eta H(P)\}
$$

여기서 $$H(P) = -\sum_{i,j} P_{ij}\log(P_{ij}) $$ 는 negative entropy이고 $$\eta > 0 $$ 는 정규화 매개변수입니다[1]

#### 3. 통합 손실 함수

$$
L = L_A + L_R + L_C
$$

- $$L_A $$: Sinkhorn 정렬 손실
- $$L_R $$: 재구성 손실
- $$L_C $$: 분류 손실[1]

#### 4. 타겟 샘플 움직임 측정
보정 전후의 prototype과의 거리 차이:

$$
d_{ac}^i = |d(z_{a,i}^t, w_c) - d(z_{c,i}^t, w_c)|
$$

여기서 $$d(\cdot, \cdot) $$는 cosine similarity입니다[1]

## 모델 구조

RAINCOAT은 3단계 처리 과정을 따릅니다:

### Stage 1: 정렬 (Alignment)
- 시간-주파수 인코더 $$G_{TF} $$ 를 통해 특성 추출
- Sinkhorn divergence로 source와 target 도메인 정렬
- 분류기 $$H $$ 와 디코더 $$U_{TF} $$ 동시 학습[1]

### Stage 2: 보정 (Correction)
- Target 도메인에서만 재구성 손실로 인코더와 디코더 재학습
- Target-specific discriminative features 학습
- Common samples는 원래 위치 유지, private samples는 이동[1]

### Stage 3: 추론 (Inference)
- 보정 전후 특성 이동량 계산
- Bimodal test를 통한 private label 탐지
- 2-mean clustering으로 threshold 결정[1]

## 성능 향상과 일반화

### 실험 결과
- **Closed-set DA**: 최대 9.0% 성능 향상 (평균 6.77% 개선)[1]
- **Universal DA**: 최대 16.33% 성능 향상[1]
- 5개 데이터셋에서 13개 SOTA 방법들을 일관되게 능가[1]

### 일반화 성능 향상 메커니즘

#### 1. 주파수 특성의 도메인 불변성
주파수 특성이 시간 특성보다 더 도메인 불변적임을 실험적으로 입증했습니다. Figure 3에서 보듯이, 서로 다른 개인의 보행 활동 시계열에서 시간 영역은 큰 차이를 보이지만 주파수 영역은 상대적으로 안정적입니다.[1]

#### 2. Ben-David 이론적 기반
Ben-David et al.의 이론에 따르면 DA 성능은 source-target 도메인 간 divergence에 의해 제한됩니다. RAINCOAT은 주파수 특성을 추가함으로써 도메인 간 invariant features를 발견하고 transferability를 향상시킵니다.[1]

#### 3. Cluster Assumption 활용
클러스터 가정에 기반하여 같은 클러스터 내 샘플들이 동일한 레이블을 가진다는 전제하에, target discriminative features를 보존하여 unknown samples를 포함한 discriminative clusters 생성을 촉진합니다.[1]

## 한계점

### 1. 계산 복잡도
- 3단계 처리 과정으로 인한 높은 계산 비용
- Sinkhorn divergence 계산의 반복적 특성[1]

### 2. 하이퍼파라미터 민감도
- 정규화 매개변수 $$\eta $$에 대한 tuning 필요
- 각 손실 항목 간 가중치 균형 조절 요구[1]

### 3. 데이터 특성 의존성
- 주파수 특성이 유효하지 않은 시계열에서는 제한적 성능
- Regular time series 가정 (irregular series는 전처리 필요)[1]

### 4. Threshold 의존성
- Private label 탐지를 위한 threshold 설정 필요
- Bimodal test의 p-value 기준 (0.05) 고정[1]

## 향후 연구에 미치는 영향과 고려사항

### 연구에 미치는 영향

#### 1. 시계열 DA 패러다임 전환
기존의 시간 영역 중심 접근에서 **시간-주파수 결합 접근**으로의 패러다임 전환을 이끌 것으로 예상됩니다.[1]

#### 2. Universal DA 확산
시계열 분야에서 처음으로 Universal DA를 실현함으로써, 실제 응용에서 더 현실적인 domain adaptation 연구를 촉진할 것입니다.[1]

#### 3. 의료 AI 응용 확대
특히 healthcare time series에서 다양한 임상 사이트 간 데이터 전이 문제 해결에 직접적인 기여가 예상됩니다.[1]

### 향후 연구 시 고려사항

#### 1. 기술적 확장
- **다른 변환 기법 탐구**: DFT 외에 wavelet, short-time Fourier transform 등 다른 주파수 변환 기법 적용 연구[1]
- **Source-Free Domain Adaptation**: 사전 훈련 단계 수정을 통한 SFDA 확장 가능성 제시[1]
- **Video Domain Adaptation**: 공간-시간 특성을 모두 고려한 비디오 DA로의 확장[1]

#### 2. 이론적 발전
- **수렴성 보장**: Sinkhorn divergence의 수렴성과 안정성에 대한 더 엄밀한 이론적 분석 필요
- **일반화 오차 한계**: 제안된 방법의 일반화 성능에 대한 PAC-learning 기반 이론적 보장 연구

#### 3. 실용성 개선
- **자동 하이퍼파라미터 튜닝**: 도메인별 특성을 고려한 adaptive parameter selection 메커니즘 개발
- **실시간 처리**: 온라인 학습 환경에서의 효율적인 구현 방안 연구
- **불균형 데이터**: 극심한 클래스 불균형 상황에서의 성능 개선 방안

#### 4. 평가 기준 정립
- **새로운 평가 메트릭**: H-score 외에 실제 응용 환경을 반영하는 종합적 평가 기준 개발
- **벤치마크 데이터셋**: 더 다양하고 현실적인 domain shift 시나리오를 포함하는 표준 벤치마크 구축

RAINCOAT은 시계열 domain adaptation 분야에 상당한 기여를 했으며, 특히 **실제 환경의 복잡성**을 반영한 첫 번째 종합적 해결책을 제시했다는 점에서 큰 의의를 가집니다.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1a569637-f14f-490a-9e96-a262386c7b1c/2302.03133v2.pdf)
