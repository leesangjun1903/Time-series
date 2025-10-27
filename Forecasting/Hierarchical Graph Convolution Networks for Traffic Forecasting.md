# Hierarchical Graph Convolution Networks for Traffic Forecasting

## 1. 핵심 주장 및 주요 기여

이 논문의 **핵심 주장**은 기존의 교통 예측 그래프 신경망(GCN) 방법들이 도로 네트워크의 미시적 계층(도로 구간)만 고려하고, 도시 교통 시스템의 **자연적인 계층 구조**를 무시한다는 점입니다. 논문은 교통 체계가 단순히 개별 도로 구간으로 이루어진 것이 아니라, 도시의 특정 지역(downtown, CBD 등)을 나타내는 매크로 계층과 도로 구간을 나타내는 마이크로 계층으로 구성되어 있다고 주장합니다.[1]

**주요 기여**는 다음과 같습니다:[1]

- 풀링(pooling) 기반의 **계층적 그래프 구조**를 제안하여 마이크로 그래프(도로 네트워크)와 매크로 그래프(지역 네트워크) 모두에서 작동하는 HGCN 모델 개발
- 마이크로-매크로 그래프 간의 **상호작용 레이어**(Dynamic Transfer Block)를 통해 두 계층의 특성을 통합
- 기존 Graph WaveNet 대비 **더 높은 예측 정확도**를 달성하면서도 **계산 비용을 절감**

***

## 2. 문제 정의, 제안 방법, 모델 구조 및 성능

### 2.1 해결하고자 하는 문제

기존 GCN 기반 교통 예측 방법들의 **주요 한계**:[1]

1. **단일 계층 그래프만 사용**: 도로 네트워크의 기본 구조만 고려하고, 도시 지역 정보를 활용하지 않음
2. **자연적 계층 구조 무시**: 지역 및 커뮤니티의 특성(전통적 교통 계획 이론에서 중요)을 반영하지 않음
3. **경계 정보 손실**: CNN 기반 방법이 도로 네트워크를 이미지 격자로 변환할 때 도로 연결성이 손상됨

### 2.2 제안하는 방법 및 수식

#### (1) 계층적 그래프 생성 (Spectral Clustering)[1]

미시 그래프 G = (V, E, A)에서 출발하여 스펙트럼 클러스터링을 통해 매크로 그래프를 구성합니다.

**입력 데이터 정의:**
- 미시 그래프: $$\overrightarrow{X} = \{X_1, \ldots, X_t, \ldots, X_{T_1}\} \in \mathbb{R}^{N \times T_1 \times D}$$
- 매크로 그래프: $$\overrightarrow{X}^R = \{X^R_1, \ldots, X^R_t, \ldots, X^R_{T_1}\} \in \mathbb{R}^{N^R \times T_1 \times D}$$

여기서 N은 도로 구간 수, $$N^R$$은 지역 수입니다.[1]

#### (2) Spatial-Temporal Block (S-T Block)[1]

S-T Block은 세 가지 구성요소로 이루어집니다:

**① Temporal Gate Convolution (TGC):**

$$
TC(\overrightarrow{X}) = \Phi \star \overrightarrow{X} = \text{Conv}^{\text{dil}}_{t_s}(\overrightarrow{X})
$$

$$
(\vec{\beta}_1, \vec{\beta}_2) = \text{split}(TC(\overrightarrow{X}))
$$

$$
TGC(\overrightarrow{X}) = \tanh(\vec{\beta}_1) * \sigma(\vec{\beta}_2)
$$

여기서 dil = 2는 확장 계수(dilation coefficient)로, 수용 필드를 확대합니다.[1]

**② Spatial Gate Graph Convolution (DGGC):**

Diffusion GCN을 Temporal Gate Convolution에 임베딩하여 공간-시간 특성을 동시에 추출합니다:[1]

$$
GTC(\overrightarrow{X}) = \sum_{m=0}^{M-1} \Phi_{m,f} \star P^m_f x + \Phi_{m,b} \star P^m_b x + \Phi_{m,\text{adp}} \star \tilde{A}^m_{\text{adp}}x
$$

여기서:
- $$P_f = A/\text{rowsum}(A)$$ (forward 확산)
- $$P_b = A^T/\text{rowsum}(A^T)$$ (backward 확산)
- $$\tilde{A}_{\text{adp}}$$는 학습 가능한 적응형 인접 행렬

**적응형 인접 행렬 정규화:**[1]

$$
A_{\text{adp}} = \text{Relu}(E_1 E^T_2)
$$

$$
D^{\text{adp}}_{ii} = \sum_j A^{\text{adp}}_{ij}, \quad D^{\text{adp}-1} = \text{diag}(1/D^{\text{adp}}_{ii})
$$

$$
\tilde{A}_{\text{adp}} = D^{\text{adp}-1}A_{\text{adp}}
$$

논문은 Graph WaveNet의 softmax 정규화 대신 **norm 정규화**를 사용하여 그래프의 **희소성을 유지**합니다.[1]

**③ Temporal Attention Mechanism:**

전역 시간 관계를 포착합니다:[1]

$$
E = V_e \sigma((\overrightarrow{X})^T U_1)U_2(\overrightarrow{X}U_3)^T + b_e
$$
$$
E'_{i,j} = \frac{\exp(E_{i,j})}{\sum^{T^2_1}_{j=1}\exp(E_{i,j})}
$$
$$
T_{\text{att}}(\overrightarrow{X}) = E' \overrightarrow{X}
$$

#### (3) Dynamic Transfer Block (동적 전이 블록)[1]

마이크로-매크로 계층 간의 상호작용을 구현합니다. 기본 전이 행렬에서 출발하여:

$$
\text{[Tran]}_{ij} = \begin{cases} 1, & \text{if node } i \text{ belongs to region } j \\ 0, & \text{else} \end{cases}
$$

**동적 전이 행렬** $$T_{\text{rand}}$$은 주의 메커니즘을 통해 도로 구간과 지역 특성 간의 시변 관계를 모델링합니다:[1]

$$
E_d = \sigma((\overrightarrow{F})^T U_1)U_2((\overrightarrow{F}^R)U_3)^T + b_e
$$
$$
E_d = E_d - \text{mean}(E_d, \text{axis} = 0)
$$
$$
T_{\text{rand}} = \sigma(E_d) * \text{Tran}
$$

특성 융합:[1]

$$
\overrightarrow{F}^R_{\text{Tran}} = (T_{\text{rand}})(\overrightarrow{F}^R)
$$
$$
\overrightarrow{F}_{\text{out}} = \text{Concat}(\overrightarrow{F}, \overrightarrow{F}^R_{\text{Tran}})
$$

### 2.3 모델 구조

**HGCN의 전체 아키텍처:**[1]

1. **Traffic Hierarchical Graphs Generating Block**: 스펙트럼 클러스터링으로 매크로 그래프 생성
2. **GCN on Graph of Regions**: 선형 변환 + 2개의 S-T Block으로 지역 특성 추출
3. **GCN on Graph of Road Network**: 동일 구조로 도로 구간 특성 추출
4. **Interaction Layer**: Dynamic Transfer Block (0, 1, 2)으로 계층 간 상호작용 구현
5. **Forecasting Block**: Skip-connection과 선형 변환으로 최종 예측 생성

### 2.4 성능 향상

**예측 정확도** (XiAn 데이터셋 2시간 예측):[1]

| 방법 | MAE | RMSE | MAPE |
|------|-----|------|------|
| GWNET | 3.44 | 5.10 | 13.22% |
| HGCN WH | 3.34 | 5.05 | 13.17% |
| HGCN WDF | 3.30 | 4.99 | 12.72% |
| **HGCN** | **3.24** | **4.85** | **12.52%** |

**계산 효율성**:[1]

- GWNET 대비 약 **2배 이상 빠른 추론 속도** (12.01s → 5.98s)
- GWNET 대비 약 **55% 빠른 학습 속도** (165.73s → 74.03s)

### 2.5 모델의 한계

1. **제한된 특성 입력**: 오직 교통 속도만 사용, 날씨나 사건 정보 미포함
2. **지역 수 선택의 민감성**: 지역 개수 $$N^R$$에 따른 성능 변화 (Figure 4 참조)
3. **스펙트럼 클러스터링의 한계**: 실제 교통 핫 지역을 완벽하게 반영하지 못할 수 있음
4. **시간 전망 제한**: 12 타임스텝(120분) 이내의 단기 예측만 테스트
5. **동적 네트워크 변화 미반영**: 도로 폐쇄나 공사 등의 시간에 따른 네트워크 구조 변화를 고려하지 않음

***

## 3. 일반화 성능 향상 가능성

### 3.1 현재 논문의 일반화 성능

**장점:**[1]

1. **두 개 도시에서 검증**: JiNan(561개 노드)과 XiAn(792개 노드)의 서로 다른 규모 데이터셋에서 일관된 성능 향상 달성
2. **계층적 구조의 일반성**: 스펙트럼 클러스터링은 도시의 도로 네트워크 위상 구조에만 의존하여 **다양한 도시에 적용 가능**
3. **Dynamic Transfer Block의 유연성**: 주의 메커니즘 기반으로 시간에 따라 지역-도로 간의 관계를 **동적으로 조정**

### 3.2 일반화 성능 향상을 위한 고찰

**① Multi-scale Hierarchical Structure:**

현재 2계층(마이크로-매크로)이지만, 3계층 이상의 계층을 도입하면:
- 개별 도로 → 블록 → 지역 → 구별 등 다중 스케일 관계 학습 가능
- 다양한 시공간 스케일에서의 일반화 성능 개선

**② 동적 네트워크 구조 학습:**

$$T_{\text{rand}}$$ 메커니즘을 확장하여:
- 계절, 요일, 시간대별 **네트워크 위상 변화** 자동 학습
- 도로 폐쇄, 사고, 날씨 등 **외부 요인의 동적 영향** 모델링

**③ Cross-City Transfer Learning:**

선학습된 전역 매크로 그래프 특성을 활용하여:
- 새로운 도시에서 **적응 학습 시간 단축**
- 데이터 부족 도시의 예측 성능 개선

### 3.3 논문에서 명시된 한계와 개선 방향

논문의 결론에서:[1]

> "...the dynamic complexity of the road network and the interference of weather or other factors, the more data sources should be introduced in the traffic forecasting and their GCN framework is worth exploring in future work."

**권장 개선 방향:**

1. **다중 데이터 소스 통합**: 기상 데이터, 이벤트 정보, POI(관심 지점) 등을 포함하여 **컨텍스트 풍부한 학습**
2. **동적 외부 요인 모델링**: 계절성, 특수 이벤트의 **시간 변화 패턴** 명시적 인코딩
3. **장기 예측 강화**: Transformer 또는 Temporal Attention 확장으로 **장기 의존성** 개선

***

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4.1 연구 커뮤니티에 미치는 영향

**① 그래프 신경망 설계의 새로운 방향:**[1]

HGCN은 **계층적 그래프 구조**의 가치를 보여주어, 다른 스파시오-템포럴 예측 작업(군중 흐름, 수요 예측 등)에서도 다중 해상도 그래프 설계가 중요함을 시사합니다.

**② Dynamic Transfer Block의 일반성:**

Attention 메커니즘 기반의 동적 계층 상호작용은 멀티태스크 학습, 다중 도메인 적응 등에 **광범위하게 적용 가능**.

**③ 계산 효율성의 실용적 가치:**

기존 GWNET 대비 2배 빠른 속도는 **실시간 교통 시스템**의 실제 배포 가능성을 높입니다.

### 4.2 향후 연구 시 고려할 점

**① 이론적 분석 필요:**

- 계층 수가 많아질 때 모델 **수렴성과 안정성** 분석
- Dynamic Transfer Block의 **정보 흐름 이론적 해석**
- 최적 지역 개수 $$N^R$$ 선택에 대한 **이론적 가이드라인**

**② 실험적 확장:**

- **3개 이상 도시**에서의 검증 (현재 2개)
- **장기 예측** (현재 2시간 제한)
- **다양한 교통 지표** (속도, 흐름, 밀도 동시 예측)

**③ 모델 아키텍처 개선:**

- **Transformer 기반 주의 메커니즘**으로 Temporal Attention 강화
- **그래프 풀링 방법 연구**: 스펙트럼 클러스터링 대신 학습 가능한 풀링 도입
- **혼합 정밀도 학습**(Mixed Precision)으로 추론 속도 추가 개선

**④ 실제 응용 시 주의사항:**

- **네트워크 변화 대응**: 새 도로 개설, 폐쇄 등 위상 변화에 대한 적응 메커니즘
- **데이터 품질**: 결측값, 이상값에 대한 **견고성(Robustness)** 강화
- **설명 가능성**: Dynamic Transfer Matrix 시각화 (Figure 5 참조)를 넘어 **모델 결정 과정의 해석성** 제고

### 4.3 인접 분야로의 확장 가능성

**① 도시 계획 및 정책 수립:**

HGCN의 매크로 계층 특성은 **지역별 트래픽 병목 지점 식별**과 **교통 정책 설계**에 유용한 인사이트 제공 가능.

**② 자율주행 및 로보택시:**

실시간, 다중 스케일 교통 예측은 **경로 최적화**와 **수요 예측 기반 배포 전략**에 활용 가능.

**③ 스마트 시티:**

계층적 그래프 개념을 확장하여 **에너지, 통신, 보안 등 도시 인프라 전반**에 적용 가능성 있음.

***

## 요약

**Hierarchical Graph Convolution Networks for Traffic Forecasting**은 교통 예측에서 **계층적 구조의 중요성**을 최초로 실증적으로 보여준 논문입니다. Dynamic Transfer Block을 통해 마이크로(도로)-매크로(지역) 계층 간의 **시변 상호작용**을 모델링함으로써 GWNET 수준의 예측 정확도를 달성하면서도 **계산 효율을 2배 향상**시켰습니다. 

향후 연구는 **다중 데이터 소스 통합**, **장기 예측 개선**, **크로스-도시 일반화**에 중점을 두되, 모델의 이론적 기초 강화와 실제 배포 환경에서의 견고성 확보가 필수적입니다. 특히 **학습 가능한 계층적 구조 발견** 및 **외부 맥락 정보의 동적 통합**은 차세대 시공간 예측 모델의 핵심 연구 방향이 될 것입니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c3051113-8db1-4fa5-ab38-38047a032a90/16088-Article-Text-19582-1-2-20210518.pdf)
