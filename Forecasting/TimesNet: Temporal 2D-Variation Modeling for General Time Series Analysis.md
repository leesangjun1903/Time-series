# TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis

## 1. 핵심 주장과 주요 기여

**TimesNet**은 시계열 분석에서 혁신적인 접근법을 제시합니다. 가장 중요한 핵심 주장은 **1차원 시계열을 2차원 공간으로 변환하여 시간적 패턴을 더 효과적으로 모델링할 수 있다는 것**입니다.[1]

주요 기여는 다음 세 가지로 요약됩니다:[1]

1. **Multi-periodicity 기반의 모듈러 아키텍처**: 시계열의 다중 주기성을 이용하여 복잡한 시간적 변화를 intraperiod-variation과 interperiod-variation으로 분해
2. **TimesBlock을 통한 2D 변환**: 1D 시계열을 2D 텐서로 변환하여 parameter-efficient inception block으로 처리
3. **범용 foundation 모델**: 5개 주요 시계열 분석 작업(단기/장기 예측, 보간, 분류, 이상 탐지)에서 일관된 최고 성능 달성

## 2. 해결하고자 하는 문제

### 2.1 기존 방법들의 한계

TimesNet은 기존 방법들이 직면한 근본적인 문제를 해결하고자 합니다:[1]

- **RNN 기반 방법들**: 장기 의존성 포착 실패와 순차 계산으로 인한 효율성 저하
- **CNN 기반 방법들**: 1차원 컨볼루션 커널의 지역성으로 인한 인접 시점 간 변화만 모델링 가능
- **Transformer 기반 방법들**: 복잡한 시간 패턴으로 인해 신뢰할 만한 의존성 발견 어려움

### 2.2 핵심 문제 정의

실제 시계열은 **복잡한 시간적 변화(intricate temporal variations)**를 포함하며, 여러 변화(상승, 하락, 변동 등)가 혼재하고 겹쳐있어 시간적 변화 모델링이 극도로 어렵습니다.[1]

## 3. 제안하는 방법론

### 3.1 Multi-periodicity 분석

TimesNet은 **Fast Fourier Transform(FFT)**를 사용하여 주기성을 발견합니다:[1]

$$A = \text{Avg}(\text{Amp}(\text{FFT}(X_{1D}))), \{f_1, \cdots, f_k\} = \arg \text{Top}_k(A), p_i = \lfloor\frac{T}{f_i}\rfloor$$ (식 1)

여기서:
- $$A \in \mathbb{R}^T$$: 각 주파수의 진폭
- $$\{f_1, \cdots, f_k\}$$: 상위 k개 중요 주파수
- $$\{p_1, \cdots, p_k\}$$: 해당 주기 길이

### 3.2 1D-to-2D 변환

핵심 아이디어는 1D 시계열을 여러 2D 텐서로 변환하는 것입니다:[1]

$$X^i_{2D} = \text{Reshape}\_{p_i,f_i}(\text{Padding}(X_{1D})), i \in \{1, \cdots, k\}$$ (식 3)

변환된 2D 텐서에서:
- **열(columns)**: intraperiod-variation (주기 내 변화)
- **행(rows)**: interperiod-variation (주기 간 변화)

### 3.3 TimesBlock 구조

각 TimesBlock은 다음 과정을 수행합니다:[1]

$$A^{l-1}, \{f_1, \cdots, f_k\}, \{p_1, \cdots, p_k\} = \text{Period}(X^{l-1}_{1D})$$

$$\hat{X}^{l,i}\_{2D} = \text{Inception}(X^{l,i}_{2D}), i \in \{1, \cdots, k\}$$ (식 5)

$$\hat{X}^{l,i}\_{1D} = \text{Trunc}(\text{Reshape}\_{1,(p_i \times f_i)}(\hat{X}^{l,i}_{2D}))$$

### 3.4 적응적 집계(Adaptive Aggregation)

서로 다른 주기의 표현을 진폭 기반으로 가중 합산합니다:[1]

$$\tilde{A}^{l-1}_{f_1}, \cdots, \tilde{A}^{l-1}_{f_k} = \text{Softmax}(A^{l-1}_{f_1}, \cdots, A^{l-1}_{f_k})$$

$$X^l_{1D} = \sum_{i=1}^k \tilde{A}^{l-1}\_{f_i} \times \hat{X}^{l,i}_{1D}$$ (식 6)

## 4. 모델 구조

TimesNet은 **잔차 연결(residual connection)**을 가진 TimesBlock들로 구성됩니다:[1]

$$X^l_{1D} = \text{TimesBlock}(X^{l-1}\_{1D}) + X^{l-1}_{1D}$$ (식 4)

각 TimesBlock 내부에는:
- **Parameter-efficient inception block**: 다중 스케일 2D 커널 사용
- **공유 파라미터 설계**: 하이퍼파라미터 k와 무관하게 모델 크기 일정 유지

## 5. 성능 향상 및 실험 결과

### 5.1 종합 성능 비교

TimesNet은 5개 주요 작업에서 일관된 최고 성능을 달성했습니다:[1]

- **장기 예측**: 9개 데이터셋 중 80% 이상에서 1위
- **단기 예측**: M4 데이터셋에서 SMAPE 11.829, MASE 1.585, OWA 0.851
- **보간**: 6개 데이터셋 모두에서 최고 성능
- **분류**: 평균 정확도 73.6%로 이전 최고 기록 경신
- **이상 탐지**: 평균 F1-score 86.34%

### 5.2 효율성 분석

TimesNet은 성능뿐만 아니라 효율성도 우수합니다:[1]
- **파라미터 수**: 0.067MB (가장 작음)
- **GPU 메모리**: 1245-2491 MiB
- **실행 시간**: 0.024-0.073초/iteration

## 6. 한계점

논문에서 언급된 주요 한계점들:[1]

### 6.1 하이퍼파라미터 민감성
- **k값 선택의 영향**: 저수준 모델링 작업(예측, 이상 탐지)에서는 k값이 성능에 더 큰 영향
- **작업별 최적값 상이**: 보간/분류는 k=3, 예측은 k=5가 최적

### 6.2 주기성 가정의 제약
- **명확한 주기성이 없는 시계열**: 무한 주기 길이로 처리하여 intraperiod-variation이 지배적
- **복잡한 비선형 패턴**: 단순한 FFT 기반 주기 발견의 한계

### 6.3 2D 변환의 한계
- **패딩으로 인한 정보 손실**: 시계열을 2D로 변환할 때 발생하는 zero-padding
- **메모리 오버헤드**: 여러 2D 텐서 저장으로 인한 메모리 사용량 증가

## 7. 일반화 성능 향상 가능성

### 7.1 표현 학습 분석

TimesNet의 뛰어난 일반화 성능은 **CKA(Centered Kernel Alignment) 유사도 분석**을 통해 입증됩니다:[1]

- **적응적 표현 학습**: 작업별로 적절한 표현 학습
  - 예측/이상탐지: 높은 CKA 유사도 (저수준 표현)
  - 보간/분류: 낮은 CKA 유사도 (계층적 표현)

### 7.2 Vision Backbone과의 연결

TimesNet은 **2D 컴퓨터 비전 백본과의 연결**을 통해 일반화 성능을 향상시킵니다:[1]
- ResNet, ResNeXt, ConvNeXt, Swin Transformer 등으로 교체 가능
- 더 강력한 2D 백본 사용 시 성능 추가 향상 가능

### 7.3 Mixed Dataset 실험

TimesNet은 **다양한 주기성을 가진 혼합 데이터셋**에서도 우수한 성능을 보입니다:[1]
- ETTh1, ETTh2 (1시간), ETTm1, ETTm2 (15분) 혼합 학습
- 다른 모델들과 달리 성능 저하 없이 일반화 가능

## 8. 미래 연구에 미치는 영향과 고려사항

### 8.1 연구에 미치는 영향

1. **범용 시계열 분석 패러다임 전환**:
   - 작업별 특화 모델에서 foundation 모델로의 전환 촉진
   - 시계열 분석의 통합적 접근법 제시

2. **2D 변환 방법론의 확산**:
   - 다른 시계열 작업에 2D 변환 기법 적용 가능
   - 컴퓨터 비전 기술의 시계열 분야 활용 촉진

3. **Multi-periodicity 중심 설계**:
   - 주기성 분석의 중요성 재조명
   - 주파수 도메인 분석과 시간 도메인 분석의 융합

### 8.2 향후 연구 시 고려할 점

1. **대규모 사전 훈련(Large-scale Pre-training)**:
   - 저자들이 제안한 향후 연구 방향[1]
   - TimesNet을 백본으로 한 광범위한 downstream 작업 지원

2. **적응적 주기 발견 방법**:
   - FFT 기반 방법의 한계 극복
   - 비선형, 시변 주기성 처리 능력 향상

3. **효율성 최적화**:
   - 더 긴 시계열에 대한 메모리 효율적 처리
   - 실시간 추론을 위한 경량화 기술 개발

4. **도메인별 특화**:
   - 의료, 금융, 산업 등 특정 도메인에 맞춘 모델 변형
   - 도메인 지식과 일반적 시계열 패턴의 결합

5. **설명 가능성 강화**:
   - 2D 변환 과정의 해석 가능성 향상
   - 각 주기성 성분의 기여도 정량화

TimesNet은 시계열 분석 분야에 **패러다임 전환**을 가져온 중요한 연구로, 향후 연구들이 이 접근법을 기반으로 더욱 발전된 모델들을 개발할 것으로 예상됩니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fc9ad902-1521-498b-96d2-2480b3142daf/2210.02186v3.pdf)
