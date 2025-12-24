# A Multi-Horizon Quantile Recurrent Forecaster

### 1. 핵심 주장 및 기여 요약

"A Multi-Horizon Quantile Recurrent Forecaster"는 Amazon의 연구자들이 2017년 NIPS 시간계열 워크숍에서 발표한 논문으로, **확률적 시계열 예측을 위한 통합 프레임워크(MQ-RNN)**를 제시합니다.[1]

**논문의 핵심 주장:**
- Sequence-to-Sequence 신경망의 표현력, 정량 회귀의 비매개변수적 특성, 직접 다중수평선 예측의 효율성을 결합하면 대규모 산업 환경에서 우수한 확률적 예측이 가능함
- **Forking-Sequences** 훈련 방식이라는 구조적 혁신을 통해 기존 Recursive 전략의 오류 누적 문제 해결 가능

**주요 기여:**
1. RNN/CNN과 정량 회귀를 결합한 첫 체계적 프레임워크 제시[1]
2. 모든 시간 단계에서 디코더를 배치하는 Forking-Sequences 훈련 방식 개발[1]
3. 미래 계절성 및 이벤트를 처리하는 로컬 MLP 모듈 설계[1]
4. 다양한 인코더(LSTM, NARX, WaveNet) 지원 가능한 유연한 프레임워크 제공

***

### 2. 해결 문제 및 방법론 상세 분석

#### 2.1 해결하고자 하는 문제

기존 신경망 기반 예측은 다음과 같은 한계를 가집니다:[1]

| 문제점 | 설명 | 영향 |
|--------|------|------|
| **오류 누적** | Recursive 전략에서 예측값을 재귀적으로 피드백하면서 오차 누적 | 장기 예측 부정확 |
| **분포 가정** | Gaussian 가정으로 인한 모델 오류 및 편향된 예측 구간 | 실제 분포와 불일치 |
| **효율성 부족** | 각 수평선별로 별도 모델 필요 또는 데이터 증강 필요 | 계산 비용 증가 |
| **미래 정보 활용 부족** | 계절성 변화나 이벤트 스파이크 처리 불충분 | 특수 상황 대응 약함 |

#### 2.2 제안 방법: 정량 손실함수

**정량 손실함수(Quantile Loss):**

$$L_q(y, \hat{y}^q) = (q-1)\max(0, y - \hat{y}^q) + q\max(0, \hat{y}^q - y)$$

여기서:
- $q$: 분위수 (0과 1 사이)
- $y$: 실제값
- $\hat{y}^q$: $q$ 분위수의 예측값

**특성:**
- $q = 0.5$일 때: MAE(Mean Absolute Error)
- 비매개변수적(parametric-free)이므로 분포 가정 불필요
- 다양한 분위수를 동시에 예측하여 신뢰 구간 제공

**총 손실:**

$$\min_{\theta} \sum_t \sum_q L_q(y_{t,k}, \hat{y}^q_{t,k})$$

여기서 합은 모든 예측 생성 시점(FCT)과 분위수에 대해 계산됩니다.

#### 2.3 모델 구조: MQ-RNN 아키텍처

**3단계 설계:**

**Step 1: 인코더 (LSTM)**
$$h_t = \text{LSTM}(y_t, x^h_t, h_{t-1})$$

- $y_t$: $t$ 시점의 관측값
- $x^h_t$: 역사적 공변량
- $h_t$: 은닉 상태

**Step 2: 글로벌 MLP (컨텍스트 생성)**
$$(c_{t_1}, \ldots, c_{t_K}, c_a) = m_G(h_t, x^f_t)$$

- 수평선-특화 컨텍스트: $c_{t_k}$ (각 미래 시점별 정보 캡처)
- 수평선-무관 컨텍스트: $c_a$ (시간 불감 공통 정보)
- 미래 입력: $x^f_t$ (계획된 프로모션, 휴일 등)

**Step 3: 로컬 MLP (분위수 생성)**

$$\hat{y}^{q_1}_{t,k}, \ldots, \hat{y}^{q_Q}_{t,k} = m_L(c_{t_k}, c_a, x^f_{t,k})$$

- 각 시간 단계 $k$별로 $Q$개 분위수 생성
- 매개변수 $m_L$ 모든 수평선에서 공유

#### 2.4 Forking-Sequences: 핵심 훈련 전략

기존 "Cutting-Sequences" 방식의 문제점:
- 시계열을 임의의 FCT에서 절단 후 증강 → 데이터 낭비
- 절단점 간 상관관계 무시 → 최적화 불안정

**Forking-Sequences 해결책:**

매 시간 단계 $t$에서 디코더를 배치하여 다중 손실 계산:

$$\text{Loss} = \sum_{t} \sum_{k=1}^{K} \sum_{q} L_q(y_{t+k}, \hat{y}^q_{t+k|t})$$

**장점:**
1. **효율성**: 한 번의 역전파로 모든 FCT 학습 (데이터 증강 불필요)
2. **안정성**: 상관된 예측 작업의 그래디언트 동시 업데이트
3. **정규화 효과**: 모든 정보를 한 번에 활용하여 과적합 방지

***

### 3. 모델의 일반화 성능 향상 메커니즘

#### 3.1 인코더 확장 기법

**NARX-스타일 인코더 (Skip Connections):**

$$\tilde{h}_t = m_{\text{skip}}(h_t, h_{t-D}, \ldots, h_{t-1})$$

- 장기 의존성 캡처 개선
- 기울기 소실 문제 완화

**지연 시계열 트릭 (Lag-series Input):**

$$\text{Input}_t = [y_t, y_{t-1}, \ldots, y_{t-D}]$$

- 더 효과적인 skip-connection 형성
- 직관적: 과거 시계열 값이 가장 예측력 높음

**WaveNet/CNN 인코더:**

$$h_t^{(l)} = \text{ReLU}(w_l * [h_{t-2^{l-1}}^{(l-1)}, \ldots, h_t^{(l-1)}])$$

- 희석 합성곱(Dilated Convolution)으로 먼 과거 캡처
- 병렬화 가능하여 LSTM보다 빠름

#### 3.2 교차 시계열 학습 (Cross-Series Learning)

**정적 특성 임베딩:**
$$s_i^{\text{emb}} = \text{Embedding}(x^s_i) \text{ (for each series } i)$$

- 서로 다른 시계열의 행동 양식 연결
- 제한된 역사를 가진 신상품의 콜드스타트 문제 해결
- 예: 제품 카테고리, 판매점 위치 등의 정보 활용

#### 3.3 실험적 일반화 증거[1]

Amazon 수요 예측 (60,000 제품):
- **MQ-RNN (모든 기법 적용)**: 모든 수평선에서 최고 성능
- **MQ-RNN_lag**: RNN 모델 중 최고 (지연 입력 효과)
- **MQ-CNN_wave**: 전체 최고 성능 (CNN 우수성 입증)

일반화 능력 지표:
- 짧은 수평선(1-4주)과 긴 수평선(48-52주) 모두에서 안정적 성능
- 콜드스타트 상황에서도 우수한 예측 (Figure 4 참조)
- 계절성 변화와 프로모션 스파이크 처리 가능

***

### 4. 모델의 한계

| 한계 | 설명 | 영향 |
|------|------|------|
| **단변량 의존성** | 다변량 분포 모델링 미흡 | 상관된 시계열 간 의존성 미반영 |
| **정량 교차** | 예측된 분위수가 교차할 수 있음 | 논리적 일관성 위반 가능 |
| **하이퍼파라미터 민감성** | RNN 길이, 은닉층 크기 등 수동 조정 필요 | 대규모 배포 시 최적화 어려움 |
| **계산 복잡도** | 장기 시계열 처리 시 메모리 집약적 | 스트리밍 데이터 적용 제한 |

***

### 5. 2020년 이후 관련 최신 연구 비교 분석

#### 5.1 분위수 회귀 개선 연구

| 연구 | 년도 | 핵심 기여 | MQ-RNN과의 비교 |
|------|------|---------|----------------|
| **Learning Quantile Functions without Crossing** | 2022[2] | 분위수 교차 문제 해결 | MQ-RNN의 주요 한계 극복 |
| **Ensemble Conformalized QR** | 2022[3] | Conformal prediction 적용 | 분포 무관 신뢰도 보증 추가 |
| **Quantile Deep Learning Models** | 2024[4] | 극값 예측에 최적화 | 암호화폐처럼 변동성 높은 데이터 강화 |
| **PatchTST + Quantile Regression** | 2025[5] | Transformer + QR 결합 | Attention 메커니즘으로 해석가능성 증대 |

#### 5.2 확률적 예측 확장 연구

| 방법 | 년도 | 특징 | 장점 |
|------|------|------|------|
| **Deep Ensembles** | 2023-2025 | 여러 모델 앙상블 | 구현 간단, 하이퍼파라미터 적음[6] |
| **Bayesian NNs** | 2023-2025 | 사후 확률 분포 | 이론적 견고성[7] |
| **Functional Neural Processes** | 2024-2025 | 함수 공간 학습 | 비매개변수적 유연성[7] |
| **Conformal Prediction** | 2023-2025 | 분포 무관 접근 | 유한표본 보증[8] |

**예시: Relational Conformal Prediction (2025)**[8]

$$\hat{I}_{t,i} = [\hat{y}_{t,i}^-, \hat{y}_{t,i}^+]$$

여기서 예측 구간은 그래프 신경망과 정량 회귀를 결합하여, 관련된 시계열 간의 의존성을 고려하면서 분포 무관 커버리지 보증을 제공합니다.

#### 5.3 Transformer 기반 모델의 부상

**Temporal Fusion Transformer (TFT, 2023-2025)**:[9]
- 변수 선택 네트워크로 해석가능성 추가
- 다중 헤드 어텐션으로 시간 역학 학습
- MQ-RNN의 로컬 MLP를 어텐션으로 대체
- 더 나은 장기 의존성 캡처

**PatchTST (2025)**:[5]
- 패치 기반 시계열 처리로 계산 효율성 향상
- Quantile regression과 SHAP 해석가능성 통합
- 수평선-특화 컨텍스트를 패치 임베딩으로 구현

#### 5.4 하이브리드 모델의 확장

| 모델 | 결합 | 주요 성과 |
|------|------|----------|
| **DeepVARMA** | LSTM + VARMA | 비정상 다변량 시계열에서 MQ-RNN 능력 향상[10] |
| **LSTM-Transformer** | LSTM + Transformer | 장기 예측의 비선형 관계 + 글로벌 패턴 캡처[11] |
| **RS-LSTM-Transformer** | Random Search + LSTM + Transformer | 파라미터 최적화 자동화로 일반화 성능 강화[12] |

***

### 6. 논문의 영향력 및 향후 연구 고려사항

#### 6.1 학문적 영향

1. **분위수 회귀의 재조명**: 신경망 분야에서 정량 회귀의 실용성 입증 → 이후 많은 연구의 토대
2. **Forking-Sequences의 영향**: 구조적 혁신 → Transformer 기반 모델의 "position encoding" 개념과 유사하게 각 시간 단계의 역할 강조
3. **End-to-End 확률적 예측**: MQ-RNN은 샘플링(DeepAR) 없이 직접 분위수 생성 → 더 효율적인 추론

#### 6.2 산업 응용의 확대

- Amazon의 대규모 수요 예측 시스템에 실제 배포[1]
- GEFCom 2014 전기 예측 경쟁에서 1위 수준 성과[1]
- 전통 통계 모델(ARIMA)을 대체하는 실질적 증거 제시

#### 6.3 향후 연구 시 고려할 점

**1) 정량 교차 문제의 해결**
- MQ-RNN: 이 문제를 인식했으나 미해결
- 개선 방향: Quantile-Rank 표현 또는 Isotonic 제약[2]

**2) 해석가능성 강화**
- 현재: 기여도 분석 미흡
- 개선: Attention mechanism 또는 SHAP 통합[5]

**3) 계산 효율성**
- 현재: LSTM의 순차 처리로 인한 병목
- 개선: Transformer, CNN 기반 구조로 병렬화,[11][9]

**4) 다변량 분포 모델링**
- 현재: 시계열별 독립적 정량 예측
- 개선: Copula 기반 또는 Vine Copula 결합[논문 미언급]

**5) 비정상 데이터 처리**
- 현재: 정적 특성으로 부분적 해결
- 개선: Domain Adaptation 또는 온라인 학습[논문 미언급]

**6) 극단값 예측 최적화**
- 현재: 균등한 모든 분위수 학습
- 개선: 꼬리 위험(tail risk) 중심 손실 가중치[4]

#### 6.4 구체적 연구 로드맵

**단기 (1-2년):**
- ✓ 분위수 교차 제약 조건 추가 (IQF, CQF)
- ✓ Transformer 인코더 교체 (성능 + 해석가능성)
- ✓ 온라인 학습 알고리즘 개발 (개념 드리프트 대응)

**중기 (2-5년):**
- ✓ 다변량 정량 회귀 (협력적 학습)
- ✓ 그래프 신경망 통합 (공간-시간 의존성)
- ✓ 설명가능 AI 프레임워크 (기업 신뢰도 향상)

**장기 (5년+):**
- ✓ 생성 모델과의 결합 (확산 모델 기반 불확실성)
- ✓ 대규모 기초 모델 (TimeGPT 같은 범용 예측 모델)
- ✓ 인과 추론 결합 (개입 효과 예측)

***

### 결론

"A Multi-Horizon Quantile Recurrent Forecaster"는 **정량 회귀, Seq2Seq, 직접 다중수평선 예측**의 세 가지 강점을 효과적으로 결합한 획기적인 작업입니다. Forking-Sequences라는 구조적 혁신은 훈련 효율성과 안정성을 동시에 개선했으며, 대규모 산업 환경에서의 성공적 배포는 신경망 기반 예측의 실용성을 증명했습니다.

2020년 이후의 연구 동향은 **분위수 교차 해결, 해석가능성 강화, 계산 효율성 개선**으로 요약되며, Transformer와 확률적 예측 방법론의 결합이 새로운 방향을 제시합니다. 향후 연구는 다변량 의존성 모델링과 온라인 학습 능력 확보에 집중할 것으로 예상되며, MQ-RNN이 제시한 기초 위에 더욱 견고하고 실용적인 시계열 예측 시스템이 구축될 것입니다.

***

### 참고 문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/811c3507-27b5-4f02-bf44-5da1d72594e2/1711.11053v2.pdf)
[2](https://arxiv.org/pdf/2111.06581.pdf)
[3](https://arxiv.org/pdf/2202.08756.pdf)
[4](https://arxiv.org/html/2411.15674v1)
[5](https://www.mdpi.com/2073-4441/17/11/1661)
[6](https://ieeexplore.ieee.org/document/11011947/)
[7](https://bsj.uowasit.edu.iq/index.php/bsj/article/view/1354)
[8](https://arxiv.org/abs/2502.09443)
[9](https://research.google/blog/interpretable-deep-learning-for-time-series-forecasting/)
[10](https://arxiv.org/pdf/2404.17615.pdf)
[11](https://pmc.ncbi.nlm.nih.gov/articles/PMC11306369/)
[12](https://pmc.ncbi.nlm.nih.gov/articles/PMC11636842/)
[13](https://link.springer.com/10.1007/s42081-025-00297-y)
[14](https://www.frontiersin.org/articles/10.3389/fcomp.2024.1447745/full)
[15](https://www.techscience.com/cmc/v85n2/63839)
[16](https://dl.lib.uom.lk/handle/123/24104)
[17](https://indjst.org/articles/comparative-analysis-of-traditional-time-series-and-machine-learning-models-for-multi-year-temperature-forecasting)
[18](https://link.springer.com/10.1007/s42001-025-00440-5)
[19](https://arxiv.org/pdf/2411.15674.pdf)
[20](http://arxiv.org/pdf/2412.13769.pdf)
[21](https://arxiv.org/pdf/2207.14219.pdf)
[22](http://arxiv.org/pdf/2405.03701.pdf)
[23](http://arxiv.org/pdf/2408.12007.pdf)
[24](https://www.mdpi.com/2078-2489/14/11/598/pdf?version=1699088576)
[25](https://pmc.ncbi.nlm.nih.gov/articles/PMC11696747/)
[26](https://www.sciencedirect.com/science/article/abs/pii/S0360835223006915)
[27](https://www.jatit.org/volumes/Vol103No10/28Vol103No10.pdf)
[28](https://pmc.ncbi.nlm.nih.gov/articles/PMC12329085/)
[29](https://www.sciencedirect.com/science/article/abs/pii/S0020025523014299)
[30](https://pmc.ncbi.nlm.nih.gov/articles/PMC12453695/)
[31](https://decidesoluciones.es/en/multi-horizon-time-series-prediction-with-neural-networks/)
[32](https://www.jisem-journal.com/index.php/journal/article/download/2009/762/3222)
[33](https://arxiv.org/html/2510.08359)
[34](https://arxiv.org/html/2408.02479v1)
[35](https://arxiv.org/pdf/2511.07059.pdf)
[36](https://arxiv.org/pdf/2511.11935.pdf)
[37](https://arxiv.org/html/2512.19970v1)
[38](https://arxiv.org/pdf/2507.08629.pdf)
[39](https://arxiv.org/html/2507.22659v2)
[40](https://arxiv.org/pdf/2508.19279.pdf)
[41](https://arxiv.org/pdf/2505.08288.pdf)
[42](https://pmc.ncbi.nlm.nih.gov/articles/PMC10098670/)
[43](https://www.sciencedirect.com/science/article/pii/S0950705124009134)
