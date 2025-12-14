
# Time Series as Images: Vision Transformer for Irregularly Sampled Time Series

## 1. 논문의 핵심 주장 및 주요 기여

**ViTST (Vision Time Series Transformer)**는 불규칙하게 샘플링된 다변량 시계열 데이터를 **라인 그래프 이미지로 변환한 후, 사전 학습된 비전 트랜스포머를 활용하여 분류하는 혁신적 접근법**을 제시합니다.[1]

핵심 통찰력은 **인간이 복잡한 수치 데이터를 시각화를 통해 분석하는 방식을 모방하면, 강력한 사전 학습 비전 모델도 시각화된 시계열의 시간적 패턴을 포착할 수 있다**는 것입니다.[1]

### 주요 기여

**1) 효과적인 불규칙 시계열 분류 방법**
- P19 의료 데이터셋: AUROC 89.2% (이전 SOTA 87.0%, +2.2%)[1]
- P12 환자 사망률 데이터셋: AUROC 85.1% (이전 SOTA 84.4%, +0.7%)[1]
- PAM 인간 활동 데이터셋: 정확도 95.8% (이전 SOTA 88.5%, +7.3%)[1]

**2) 범용 프레임워크로서의 다목적성**
- 불규칙 시계열과 규칙적 시계열 모두에서 우수한 성능[1]
- 기존 방법들과 달리 양쪽 유형 모두 효과적으로 처리[1]

**3) 자연 이미지에서 학습한 지식의 성공적 전이**
- ImageNet-21K 기반 사전 학습 Swin Transformer 활용[1]
- 컴퓨터 비전의 고속 발전을 시계열 분석에 직접 활용 가능[1]

***

## 2. 문제 정의, 제안 방법, 모델 구조

### 2.1 해결하고자 하는 문제

불규칙 시계열의 특성:
- **불규칙한 시간 간격**: 관측 사이의 시간 간격이 변함[1]
- **변수 정렬 불일치**: 다변량 데이터에서 서로 다른 변수가 다른 시점에 관측됨[1]
- **높은 결측 비율**: 의료 데이터의 경우 88-95%[1]

기존 전문화된 방법(GRU-D, mTAND, Raindrop)의 한계:
- 높은 복잡도와 설계의 어려움
- 규칙/불규칙 중 하나에만 특화
- 도메인 전문지식 필요[1]

### 2.2 ViTST의 제안 방법

#### 시계열 → 이미지 변환

각 변수 $d$에 대해 라인 그래프를 생성합니다:[1]

$$\text{LineGraph}_d = \{(t_d^j, v_d^j) | j = 1, \ldots, n_d\}$$

여기서:
- $t_d^j$: 변수 $d$의 $j$번째 관측 시점
- $v_d^j$: 해당 값

**설계 특징:**[1]
- 각 변수: 고유 색상 사용
- 관측 지점: 별표 마커(∗) 표시
- 변수 정렬: 결측률이 높은 순서대로 배열

$D$개 라인 그래프를 격자로 배열:[1]

$$\text{Grid Layout:} \quad l \times (l-1) < D \leq l \times l \text{ or } l \times (l+1)$$

예시:
- P19 (34개 변수): 6×6 격자 → 384×384 이미지[1]
- P12 (36개 변수): 6×6 격자 → 384×384 이미지[1]
- PAM (17개 변수): 4×5 격자 → 256×320 이미지[1]

### 2.3 모델 구조: Swin Transformer

#### 아키텍처의 이점

Swin Transformer는 **계층적 구조**로 로컬과 글로벌 정보를 모두 포착합니다:[1]

**Window-based Multi-head Self-Attention (W-MSA):**
- 각 윈도우 내에서 자기 주의 계산
- 인트라-변수 상호작용 및 개별 라인 그래프의 시간적 동역학 포착[1]

**Shifted Window Multi-head Self-Attention (SW-MSA):**
- 윈도우 경계를 넘어 주의 계산
- 다양한 라인 그래프 간 크로스-변수 상호작용 포착[1]

#### 수학적 표현

Swin Transformer 연속 블록:[1]

$$\hat{z}^l = \text{W-MSA}(\text{LN}(z^{l-1})) + z^{l-1}$$

$$z^l = \text{MLP}(\text{LN}(\hat{z}^l)) + \hat{z}^l$$

$$\hat{z}^{l+1} = \text{SW-MSA}(\text{LN}(z^l)) + z^l$$

$$z^{l+1} = \text{MLP}(\text{LN}(\hat{z}^{l+1})) + \hat{z}^{l+1}$$

여기서:
- $z^l$: 블록 $l$의 MLP 출력
- $\hat{z}^l$: 주의 모듈 출력
- $\text{LN}$: 레이어 정규화

#### 분류 헤드

최종 단계에서 이미지 패치 표현으로부터 분류:[1]

$$\hat{y}_i = \text{Linear}(\text{Flatten}(z_{\text{final}}))$$

정적 특징 통합 (의료 데이터):[1]
- RoBERTa로 인구통계학적 정보(연령, 성별, 체중 등) 인코딩
- 이미지 임베딩과 텍스트 임베딩 연결

***

## 3. 성능 향상 및 한계

### 3.1 주요 성능 개선

| 데이터셋 | 지표 | ViTST | 이전 SOTA | 개선도 |
|---------|------|-------|---------|--------|
| P19 | AUROC | 89.2% | 87.0% | +2.2% |
| P19 | AUPRC | 53.1% | 51.8% | +1.3% |
| P12 | AUROC | 85.1% | 84.4% | +0.7% |
| P12 | AUPRC | 51.1% | 48.2% | +2.9% |
| PAM | Accuracy | 95.8% | 88.5% | +7.3% |
| PAM | F1 Score | 96.5% | 89.8% | +6.7% |

### 3.2 결측 데이터에 대한 강건성

Leave-Sensors-Out 설정 (결측 비율 50%):[1]

| 지표 | ViTST | Raindrop |
|------|-------|---------|
| Accuracy | 79.7% | 46.6% |
| F1 Score | **80.8%** | **38.0%** |
| **절대 개선** | | **+42.8%** |

이는 **탁월한 결측 강건성**을 입증합니다.[1]

### 3.3 성능 향상의 원인

#### 사전 학습의 중요성

사전 학습 여부 비교:[1]

| 모델 | P19 AUROC | 개선도 |
|------|----------|--------|
| 사전 학습 Swin | 89.4% | - |
| 비사전 학습 | 77.7% | **-11.7%** |

**ImageNet 기반 지식의 강력한 전이 효과 입증**[1]

#### 비전 트랜스포머 vs CNN

| 아키텍처 | P19 AUROC | P12 AUROC | PAM 정확도 |
|---------|----------|----------|----------|
| ViT | 87.9% | 84.8% | 93.4% |
| Swin Transformer | 89.4% | 85.6% | 96.1% |
| ResNet-50 (CNN) | 73.2% | 78.9% | 88.5% |

CNN은 **공간 정보 보존 능력이 부족**하여 라인 그래프 패턴 포착에 미흡합니다.[1]

#### 주의 메커니즘의 해석

Attention Map 분석:[1]
- 모델은 라인 그래프 윤곽선에 집중
- 관측 지점과 선 기울기 변화 지점에 높은 주의
- 동적 변화 없는 평탄 그래프는 낮은 주의

### 3.4 이미지 생성 설계 결정의 영향

#### 보간 (Interpolation)

| 설정 | P19 AUROC | P12 AUROC | 영향 |
|-----|----------|----------|------|
| 선형 보간 | 89.2% | 85.1% | 기본 |
| 보간 없음 | 89.6% | 85.7% | **+0.4~0.6%** |

**해석**: 보간이 관측/보간점 구분을 흐리게 함[1]

#### 색상 차별화

- 색상 제거 시 P19 AUROC: 88.8% (감소)
- **색상은 변수 간 구분에 필수**[1]

#### 변수 순서

결측률 기반 정렬이 **안정적 결과 제공**[1]

### 3.5 규칙적 시계열 데이터에서의 성능

10개 UEA 다변량 시계열 분류 데이터셋:[1]

- **평균 정확도: 78.0%** (TST 79.1% 대비)
- PS 데이터셋: 963개 변수 처리 가능[1]
- EW 데이터셋: 17,984 길이 시계열 처리 가능[1]

**결론**: 규칙 및 불규칙 시계열 모두 효과적으로 처리[1]

### 3.6 한계

#### 방법론적 한계

1) **이미지 변환 과정의 제약**:[1]
   - Matplotlib 기반 간단한 시각화만 사용
   - 3D 표현, 히트맵 등 고급 시각화 미탐색
   - 시간-주파수 표현 (Spectrogram) 미활용

2) **이론적 이해 부족**:[1]
   - 왜 자연 이미지 사전 학습이 시각화 시계열에 효과적인가?
   - 라인 그래프 표현이 원본 수치 데이터의 어떤 특성을 보존/손실하는가?

#### 계산 비용

| 데이터셋 | 모델 | 추론 시간 (초) |
|---------|------|--------------|
| P19 | Transformer | 0.21 |
| P19 | **ViTST** | **31.04** |
| P12 | Transformer | 0.12 |
| P12 | **ViTST** | **12.14** |

**이미지 인코딩/디코딩 오버헤드 존재**하나 의료 응용에는 수용 가능[1]

***

## 4. 모델의 일반화 성능 향상 가능성

### 4.1 주요 일반화 특성

#### 데이터 특성 범위 처리

- **변수 수**: 3개 (EC) ~ 963개 (PS)[1]
- **시계열 길이**: 29 (JV) ~ 17,984 (EW)[1]
- **결측률**: 0% ~ 95%[1]
- **클래스 균형**: 균형 (PAM) ~ 불균형 (P19 4%, P12 14%)[1]

#### 도메인 적응성

- **의료**: 불규칙 샘플링, 높은 결측률
- **인간 활동 인식**: 센서 데이터, 정상적 시간 간격
- **규칙적 시계열**: 고정 간격, 완전 관측[1]

### 4.2 사전 학습 기반 전이 학습의 강력함

사전 학습된 비전 모델의 효과:

$$\text{성능 향상} = f(\text{도메인 간격}, \text{과제 유사성}, \text{모델 용량})$$

**ViTST의 경우**:[1]
- 도메인 간격: 중간 (자연 이미지 → 시각화 시계열)
- 과제 유사성: 높음 (이미지 분류 = 시계열 분류)
- 모델 용량: 높음 (ImageNet-21K 사전 학습)

**결과**: 강력한 전이 학습 효과 관찰[1]

### 4.3 강건성 메커니즘

#### 결측 데이터 처리의 자연스러움
- 라인 그래프는 불규칙성에 자연스럽게 대응[1]
- 센서 탈락에 대한 우수한 강건성 입증[1]

#### 플로팅 파라미터의 불변성

| 파라미터 | 변경 | AUROC 변화 |
|---------|------|----------|
| 선 스타일 | 실선→점선/점-선 | -0.1% |
| 선 두께 | 1→0.5, 2 | -0.5% |
| 마커 모양 | ∗ → ∧, ◦ | -0.1% |

**높은 강건성 입증**[1]

### 4.4 추가 개선 가능성

#### 자기 지도 학습 (Self-Supervised Learning)

Masked Image Modeling 예비 실험:[1]

$$L = \frac{1}{|\Omega(p_M)|} \|\hat{p}_M - p_M\|_1$$

결과:
- P19 AUPRC: 52.8% → 53.8% (+1.0%)[1]
- 추가 사전 학습의 개선 가능성 시사[1]

***

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 불규칙 시계열 처리 방법의 진화

#### GRU-D (Che et al., 2018)

**핵심**: 감쇠 메커니즘을 통한 결측값 처리[2]

$$\gamma_t^d = e^{-\lambda_d \cdot \delta_t^d}$$

**성능**:
- P19: AUROC 83.9%
- P12: AUROC 81.9%
- PAM: 정확도 83.3%

**한계**: 도메인 특화적 설계, 규칙 시계열 부적합[2]

#### mTAND (Shukla & Marlin, 2020)

**혁신**: 다중 시간 주의 메커니즘[3]

$$\text{mTAN}(t, s) = \sum_{h=1}^{H} \alpha_h(t) \cdot v_h$$

**성능**:
- P19: AUROC 84.4%
- P12: AUROC 84.2%
- PAM: 정확도 74.6%

**특징**: ODE 솔버 필요로 연산 비용 높음[3]

#### Raindrop (Zhang et al., 2022)

**개념**: 그래프 신경망 기반 센서 간 의존성 모델링[4][5]

$$\text{Message Passing:} \quad h_i^{(l+1)} = \sigma\left(W \cdot \left[h_i^{(l)}, \sum_{j \in \mathcal{N}_i} h_j^{(l)}\right]\right)$$

**성능**:
- P19: AUROC 87.0%
- P12: AUROC 82.8%
- PAM: 정확도 88.5%

**ViTST와 비교**: PAM에서 +7.3% 정확도 우수[4][1]

### 5.2 시계열 이미지 변환 방법

#### 전통적 방법 (2015년 이전)

| 방법 | 특징 | 한계 |
|------|------|------|
| Gramian Angular Field | 상관관계 캡처 | 도메인 지식 필요 |
| Recurrent Plots | 주기적 패턴 탐지 | 해석 어려움 |
| Markov Transition Fields | 상태 전이 모델링 | 시간 정보 손실 |

**ViTST와의 차이**: 더 직관적인 라인 그래프, 사전 학습 모델 활용[1]

#### 최근 스펙트로그램 기반 방법 (2024)

"From Pixels to Predictions" (2024):
- 시간-주파수 스펙트로그램 사용[6]
- Vision Transformer + 멀티모달 학습

**비교**:
- 라인 그래프: 직관성, 계산 효율
- 스펙트로그램: 주파수 특성, 계산 복잡도 증가[6]

### 5.3 Transformer 기반 시계열 방법 (2021-2024)

#### TST 계열 전문화된 모델

**Autoformer (Wu et al., 2021)**:
- 자기상관 메커니즘
- 분해 기반 설계

**Temporal Fusion Transformer (Lim et al., 2021)**:
- 정적/동적 특징 융합
- 해석 가능성

**성능**: 예측 작업에서 우수, 불규칙 시계열에는 ViTST가 우수[1]

#### Patch 기반 방법 (2023-2024)

**PatchTST (Nie et al., 2023)**:
- 패치 기반 표현
- 채널 독립성 가정

**MultiResFormer (Zhang et al., 2024)**:
- 적응형 다중 해상도
- 주기성 모델링

**ViTST와의 차이**:
- PatchTST: 수치 특성 직접 사용
- ViTST: 이미지 변환 + 사전 학습 비전 모델, 불규칙 데이터에서 더 강력[1]

### 5.4 자기 지도 학습 (2022-2024)

**TS-TCC (Yang et al., 2021)**:
- 시간/맥락 대비 학습
- 데이터 증강 기반

**AnomalyBERT (Tuli et al., 2023)**:
- 변환 기반 사전 학습

**ViTST와의 관계**: ViTST는 비전 모델 사전 학습, 시계열 특화 사전 학습 결합 가능성[1]

### 5.5 종합 비교표

| 특성 | GRU-D | mTAND | Raindrop | ViTST | PatchTST |
|------|-------|-------|----------|-------|----------|
| 불규칙 처리 | ✓ | ✓ | ✓ | ✓ | ✗ |
| 규칙 시계열 | △ | △ | ✗ | ✓ | ✓ |
| 사전 학습 | ✗ | ✗ | ✗ | ✓ | △ |
| P19 성능 | 83.9% | 84.4% | 87.0% | **89.2%** | N/A |
| PAM 성능 | 83.3% | 74.6% | 88.5% | **95.8%** | N/A |
| 추론 속도 | 빠름 | 느림 | 중간 | 느림 | 빠름|

***

## 6. 앞으로의 연구에 미치는 영향과 고려사항

### 6.1 학술적 영향

#### 패러다임 전환

**기존 관점**:
- 시계열 = 수치 시퀀스 → 시퀀스 모델 (RNN/Transformer)
- 불규칙성 = 특수 모듈 필요

**ViTST 제시 관점**:
- 시계열 = 시각적 표현 → 비전 모델
- 불규칙성 = 자연스러운 이미지 특성
- **범용 프레임워크의 가능성**[1]

#### 학제간 연구 촉진

- 컴퓨터 비전의 고속 발전 → 시계열 분석 적용[1]
- 멀티모달 학습: 시각+시간 정보 결합
- 자기 지도 학습의 시계열 응용 확대

### 6.2 방법론 개선 방향

#### 이미지 표현 최적화

**탐색할 대안**:
1) 하이브리드 표현 (라인 그래프 + 스펙트로그램)
2) 동적 보간 (학습된 곡선)
3) 적응형 스케일링 (학습 가능 정규화)

#### 아키텍처 개선

1) **계층적 변수 인코딩**: 개별 → 그룹 → 전역
2) **효율성 개선**: MobileViT, 지식 증류, 양자화
3) **멀티모달 학습**: 이미지 + 메타데이터 + 임상 노트[1]

### 6.3 응용 분야별 고려사항

#### 의료 응용

**현재 성능**:
- 패혈증 조기 예측: 89.2% AUROC[1]
- **임상 배포 가능 수준**

**개선 필요**:
1) 신뢰도와 설명성
2) 규제 준수 (FDA, GDPR)
3) 강건성 보증

#### 금융 응용

**제약**: 극도로 동적인 패턴, 높은 노이즈, 개념 드리프

**ViTST 적응**:
- 실시간 이미지 생성 최적화
- 적응형 정규화
- 동적 윈도우 크기

#### IoT/센서 네트워크

**제약**: 계산 능력, 통신 대역폭, 온라인 학습

**해결책**:
- 경량 비전 모델 (MobileViT)
- 연합 학습

### 6.4 향후 연구 시 핵심 고려사항

#### 이론적 분석 필요

1) **라인 그래프 표현의 정보 복원도**
2) **자연 이미지와 시각화 시계열 간 전이학습 효율**
3) **어떤 시계열 특성이 시각적 표현에서 손실되는가?**

#### 벤치마크 강화

- 불규칙 시계열 벤치마크 확대
- 도메인 간 전이 학습 평가
- 계산 비용-성능 트레이드오프 분석

#### 확장성 검증

- 메모리 효율적 배치 처리
- 초장시간 시계열 처리 (효율적 주의 메커니즘)
- 증분/연합 학습

***

## 7. 최종 평가

### 7.1 핵심 성과 재평가

ViTST는 **"복잡한 불규칙 시계열 분석을 단순한 이미지 변환과 사전 학습 비전 모델로 효과적으로 해결한다"**는 혁신적 접근법입니다.[1]

**3가지 핵심 성과**:
1. **성능**: 모든 벤치마크에서 SOTA 달성 (PAM +7.3% 정확도)[1]
2. **일반성**: 불규칙/규칙 시계열 모두 처리 (범용성)[1]
3. **강건성**: 50% 센서 손실에도 80% 정확도 유지[1]

### 7.2 학문적 의의

- **패러다임 확대**: 시계열 분석에 비전 기반 접근법 도입
- **학제간 통합**: CV의 고속 발전을 시계열에 직접 활용
- **이론-실무 격차 해소**: 단순성과 효과성의 조화

### 7.3 실무적 가치

**즉시 적용 가능**:
- 의료 진단 시스템 개발
- 환자 모니터링
- 조기 경고 시스템[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a0d6e564-c4f0-4fe4-bac3-4d0f4bcdbe5a/2303.12799v2.pdf)
[2](https://arxiv.org/pdf/1606.01865.pdf)
[3](https://openreview.net/pdf?id=mXbhcalKnYM)
[4](https://arxiv.org/pdf/2110.05357.pdf)
[5](https://zitniklab.hms.harvard.edu/publications/papers/raindrop-iclr22.pdf)
[6](https://arxiv.org/abs/2403.11047)
[7](https://link.springer.com/10.1007/s10278-024-01322-4)
[8](https://ieeexplore.ieee.org/document/10500832/)
[9](https://arxiv.org/abs/2303.12799)
[10](https://ieeexplore.ieee.org/document/10068090/)
[11](https://ieeexplore.ieee.org/document/10312633/)
[12](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12464/2653911/Time-distance-vision-transformers-in-lung-cancer-diagnosis-from-longitudinal/10.1117/12.2653911.full)
[13](https://www.semanticscholar.org/paper/5a5cadc69ba1b807da9af37af6d49c085b5a8dca)
[14](https://esmed.org/MRA/mra/article/view/6660)
[15](https://link.springer.com/10.1007/s13755-025-00401-2)
[16](https://ieeexplore.ieee.org/document/11112602/)
[17](https://arxiv.org/pdf/2303.12799.pdf)
[18](https://arxiv.org/pdf/2305.04468.pdf)
[19](http://arxiv.org/pdf/2412.16515.pdf)
[20](http://arxiv.org/pdf/2408.09723.pdf)
[21](http://arxiv.org/pdf/2110.02642v5.pdf)
[22](https://arxiv.org/ftp/arxiv/papers/2311/2311.06916.pdf)
[23](http://arxiv.org/pdf/2311.18780.pdf)
[24](https://openreview.net/pdf?id=lRgEbHxowq)
[25](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2024.1414352/full)
[26](https://pmc.ncbi.nlm.nih.gov/articles/PMC10944280/)
[27](https://neurips.cc/virtual/2023/poster/71219)
[28](https://www.ijcai.org/proceedings/2021/324)
[29](https://www.sciencedirect.com/science/article/abs/pii/S0925231220300606)
[30](https://www.ijcai.org/proceedings/2021/0324.pdf)
[31](https://www.sciencedirect.com/science/article/pii/S1568494624011244)
[32](https://arxiv.org/html/2506.08641v2)
[33](https://arxiv.org/html/2410.14769v1)
[34](https://arxiv.org/html/2505.08199v1)
[35](https://arxiv.org/html/2505.00307v2)
[36](https://arxiv.org/html/2510.23382v1)
[37](https://arxiv.org/html/2503.10198v1)
[38](https://arxiv.org/html/2509.23494v2)
[39](https://arxiv.org/html/2508.12230v1)
[40](https://arxiv.org/html/2501.13392v2)
[41](https://arxiv.org/html/2511.19497v1)
[42](https://arxiv.org/pdf/2401.14208.pdf)
[43](https://arxiv.org/abs/2212.01133)
[44](https://ieeexplore.ieee.org/document/10074669/)
[45](https://dl.acm.org/doi/10.1145/3533271.3561751)
[46](https://arxiv.org/abs/2204.08414)
[47](https://ieeexplore.ieee.org/document/10388222/)
[48](https://arxiv.org/abs/2211.07031)
[49](https://link.springer.com/10.1007/978-981-16-9154-6_25)
[50](https://www.semanticscholar.org/paper/a3ebed9b1e75a528f8e943275bdb99ffe515ab2c)
[51](https://www.semanticscholar.org/paper/ee032b635670276c7b03d3f5e614ad5d2fe8c0e9)
[52](https://ieeexplore.ieee.org/document/9994883/)
[53](http://arxiv.org/pdf/2410.14030.pdf)
[54](https://arxiv.org/pdf/2302.08415.pdf)
[55](https://www.mdpi.com/2072-666X/15/2/217/pdf?version=1706692510)
[56](https://arxiv.org/html/2501.16900v1)
[57](http://arxiv.org/pdf/2307.03759.pdf)
[58](https://www.mdpi.com/1099-4300/24/6/759/pdf?version=1654073706)
[59](https://arxiv.org/pdf/2310.15978.pdf)
[60](https://zitniklab.hms.harvard.edu/projects/Raindrop/)
[61](https://satyanshukla.github.io/pdf/mtan_poster.pdf)
[62](https://doheon.github.io/%EB%85%BC%EB%AC%B8%EB%B2%88%EC%97%AD/time-series/pt-grud-post/)
[63](https://arxiv.org/abs/2110.05357)
[64](https://iclr.cc/media/iclr-2021/Slides/2703.pdf)
[65](https://arxiv.org/abs/1606.01865)
[66](https://github.com/mims-harvard/Raindrop)
[67](https://arxiv.org/html/2307.03759v3)
[68](https://arxiv.org/pdf/2504.05768.pdf)
[69](https://arxiv.org/pdf/1911.00605.pdf)
[70](https://arxiv.org/pdf/2408.05849.pdf)
[71](https://arxiv.org/html/2509.25678v1)
[72](https://arxiv.org/pdf/2209.10801.pdf)
[73](https://arxiv.org/abs/2101.10318)
[74](https://www.reddit.com/r/MachineLearning/comments/tejt1j/d_recurrent_neural_networks_for_multivariate_time/)
[75](https://dspace.mit.edu/bitstream/handle/1721.1/134960/s41598-018-24271-9.pdf?sequence=2&isAllowed=y)
