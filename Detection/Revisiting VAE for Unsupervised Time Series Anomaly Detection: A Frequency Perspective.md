# Revisiting VAE for Unsupervised Time Series Anomaly Detection: A Frequency Perspective

### Executive Summary

본 논문은 "Revisiting VAE for Unsupervised Time Series Anomaly Detection: A Frequency Perspective"로, 2024년 2월 WWW 학회에서 발표되었습니다. 이 논문은 기존 VAE 기반 이상 탐지 방법의 근본적인 한계를 분석하고, 주파수 영역 정보를 활용하여 이를 극복하는 FCVAE (Frequency-enhanced Conditional Variational Autoencoder)를 제안합니다. 주요 기여는 (1) VAE의 이중 패턴 포착 실패 원인 규명, (2) 조건부 VAE에 주파수 정보 통합, (3) 타겟 어텐션 메커니즘을 통한 정교한 주파수 선택입니다.[1][2]

***

### 1. 핵심 주장과 주요 기여

#### 1.1 문제 정의

기존 VAE 기반 이상 탐지 방법들의 핵심 한계는 **두 가지 패턴을 동시에 포착하지 못한다**는 점입니다:[2]

- **장주기 이질적 패턴 (long-periodic heterogeneous patterns)**: 시계열 데이터에 나타나는 다양한 주기의 이상 패턴을 포착하지 못함
- **단주기 상세 추세 (detailed short-periodic trends)**: 세부적인 단기 변동을 재구성하는 데 실패

Figure 1의 분석에 따르면, 기존 VAE 방법들은 이상점(blue marker ③)은 잘 무시하지만 정상 지점 중 일부(blue rectangle ①, ellipse ④)의 재구성이 불완전하여 전체 성능을 저하시킵니다.[2]

#### 1.2 근본 원인 분석

논문의 핵심 관찰은 **주파수 영역 정보의 부재**가 이러한 실패의 근본 원인이라는 점입니다. 시간 영역만으로 학습한 VAE는:

- 주기적 변화를 충분히 포착하지 못함
- 정상 데이터의 주파수 특성을 완전히 학습하지 못함
- 이질적인 주기 패턴이 있을 때 특히 취약함

#### 1.3 제안 방법: FCVAE

논문은 이러한 한계를 극복하기 위해 **Frequency-enhanced Conditional Variational Autoencoder (FCVAE)**를 제안합니다:[2]

**핵심 혁신:**
- **전역 주파수 특성**: 전체 윈도우의 주파수 정보를 조건으로 통합
- **국소 주파수 특성**: 부분 윈도우 단위의 상세 주파수 정보
- **타겟 어텐션 메커니즘**: 가장 유용한 부분 주파수를 선택적으로 추출

---

### 2. 방법론 상세 설명

#### 2.1 조건부 VAE (CVAE) 기초

표준 VAE의 Evidence Lower BOund (ELBO) 목적함수:

$$\mathcal{L}_{ELBO} = -\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] + KL(q_\phi(z|x)||p(z))$$

CVAE는 조건 $c$를 추가하여 확장합니다:[2]

$$\mathcal{L}_{CVAE} = -\mathbb{E}_{q_\phi(z|x,c)}[\log p_\theta(x|z,c)] + KL(q_\phi(z|x,c)||p(z|c))$$

여기서:
- $q_\phi(z|x,c)$: 인코더 (조건부 인식)
- $p_\theta(x|z,c)$: 디코더 (조건부 재구성)
- $KL$: Kullback-Leibler 발산

#### 2.2 주파수 특성 추출

**Fast Fourier Transform (FFT)을 통한 주파수 분해:**

$$X_f = FFT(x) = \sum_{n=0}^{N-1} x[n] e^{-2\pi i \frac{kn}{N}}, \quad k=0,1,...,N-1$$

**전역 주파수 특성** $f_{global}$:
- 전체 윈도우에서 추출된 주파수 신호
- Magnitude spectrum의 주요 특성값 (예: peak frequencies, spectral energy)

**국소 주파수 특성 및 타겟 어텐션**:

윈도우를 부분 윈도우로 세분화:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

여기서 query는 재구성 오류 정보에서 유도되며, 가장 유용한 부분 주파수를 선택합니다.[2]

#### 2.3 이상 탐지 스코어

정상 데이터와 이상 데이터의 구분:

$$\text{Anomaly Score} = ||x - \hat{x}||_2^2 + \lambda \cdot KL(q_\phi(z|x,c)||p(z|c))$$

여기서 $\lambda$는 KL 항의 가중치 (재구성 vs. 정규화 간 균형 조절).[2]

**임계값 기반 탐지:**

$$y_t = \begin{cases} 1 & \text{if } AS_t > \tau \\ 0 & \text{otherwise} \end{cases}$$

***

### 3. 모델 구조 및 아키텍처

#### 3.1 FCVAE의 세 가지 핵심 모듈

**1. FFT 전처리 모듈**
- 입력 시계열을 주파수 영역으로 변환
- 진폭 스펙트럼과 위상 정보 추출

**2. 조건 벡터 구성**
- 전역 주파수 특성: Top-K frequencies 및 주파수 에너지
- 국소 주파수 특성: 타겟 어텐션으로 선택된 부분 주파수
- 최종 조건 벡터:

$$\(c=[\text{GlobalFreq},\text{LocalFreq}_{\text{selected}}]\)$$

**3. CVAE 인코더-디코더**
- **인코더**: $q_\phi(z|x,c) \rightarrow (\mu_\phi, \log\sigma^2_\phi)$ (RNN 또는 MLP)
- **샘플링**: $z \sim \mathcal{N}(\mu_\phi, \sigma^2_\phi I)$
- **디코더**: $p_\theta(x|z,c) \rightarrow \hat{x}$ (시계열 재구성)

#### 3.2 훈련 전략

**혼합 이상 정상 데이터 처리:**

논문의 중요한 장점은 VAE 기반 방법들이 **혼합된 훈련 데이터**(약간의 이상 포함)를 다룰 수 있다는 점입니다. 이는 예측 기반 방법과의 핵심 차이입니다.[2]

**누락 데이터 주입 (Missing Data Injection)**:

재구성 기반 이상 탐지 향상을 위해, 훈련 중 일부 시점을 의도적으로 누락시켜 모델의 강건성을 개선합니다.[2]

***

### 4. 성능 향상 및 일반화 성능

#### 4.1 FCVAE의 성능 향상

**실험 결과:**

기존 VAE 기반 방법 대비 FCVAE의 개선:

| 메트릭 | 개선 사항 | 설명 |
|--------|---------|------|
| **이상점 포착** | 향상됨 | 높은 F1-score 달성 |
| **패턴 다양성** | 획기적 | 이질적 장주기 패턴 포착 |
| **재구성 정확도** | 개선됨 | 단주기 상세 추세 복원 |
| **효율성** | 비교 가능 | 예측 기반 방법에 비해 빠름 |

#### 4.2 다른 조건 타입과의 비교

**Figure 9(a) 분석** - CVAE의 조건 선택:[2]

$$\begin{array}{ll}
\text{주파수 정보 조건} & \text{F1 = 0.95 (최우) } \\
\text{타임스탐프 조건} & \text{F1 = 0.62} \\
\text{시간 영역 정보 조건} & \text{F1 = 0.68}
\end{array}$$

**결론**: 주파수 정보가 조건으로 사용될 때 압도적으로 우수한 성능을 보임. 이는 주파수가 시간 영역 정보를 보완하는 **귀중한 사전정보(valuable complementary prior)**이기 때문입니다.[2]

#### 4.3 주파수 기반 VAE (FVAE) vs. FCVAE

**Figure 9(b) 결과**:[2]

$$\text{FCVAE F1 score} > \text{FVAE F1 score}$$

FCVAE의 우월성 이유:
1. **CVAE 구조 자체의 우수성**: CVAE는 일반적으로 다양한 응용에서 VAE를 상회하는 성능을 보임
2. **조건 통합의 효율성**: 주파수를 직접 입력에 추가하는 것(FVAE)보다, 조건으로 통합하는 것(FCVAE)이 더 효과적

#### 4.4 일반화 성능 향상 가능성

**주요 강점:**

1. **새로운 패턴에 대한 학습 능력**
   - 주파수 정보로 인해 데이터셋 내 모든 시계열을 함께 학습
   - 충분한 패턴 다양성 제공
   - 미보유 패턴도 주파수 특성으로 포착 가능

2. **혼합 이상-정상 훈련 데이터 처리**
   - 예측 기반 방법은 **순수 정상 훈련 데이터만 필요**하지만, 실무에서는 거의 불가능
   - FCVAE는 약간의 이상을 포함한 실제 데이터에서도 작동 가능[2]

3. **도메인 적응 가능성**
   - 주파수 특성의 보편성으로 인해 다양한 산업 애플리케이션에 적용 가능

**한계와 제약:**

논문 저자들도 명시적으로 언급하듯이:[2]

> "더욱이, 완전한 일반화는 여전히 도전적이며, 특정 도메인의 특성을 반영하는 추가적인 조건 정보가 필요할 수 있습니다."

**제약사항:**
- 완전히 새로운 도메인으로의 제로샷(zero-shot) 전이는 제한적
- 극단적 이상(extreme anomalies)의 학습은 여전히 필요
- 계절성과 트렌드가 혼재된 복잡한 데이터에서는 추가 분해 필요

***

### 5. 2020년 이후 관련 최신 연구 비교 분석

#### 5.1 연도별 주요 방법론 진화

**2020년 - VAE의 기본 개선 시대:**

- **VELC (Re-Encoder & Latent Constraint)**: 잠재 공간에 제약을 추가하여 이상을 잘 재구성하지 못하도록 강제[3]

**2021년 - Transformer 도입 시대:**

- **Anomaly-Transformer**: 협회 불일치(Association Discrepancy) 메커니즘 도입[4]
  - 1,159회 인용
  - 6개 벤치마크에서 최고 성능
  - 이상 수집의 희소성을 활용한 attention 기반 접근

**2022년 - 시간-주파수 분석 본격화:**

- **TiSAT**: 자체주의 기반 Transformer로 장거리 의존성 포착[5]
- **TFAD (Time-Frequency Anomaly Detection)**: 이중 분기 구조 (시간 + 주파수 도메인)[6]
  - 시계열 분해를 통한 구성 요소별 이상 탐지
  - 데이터 증강 메커니즘 적용
  - Point vs. Pattern 이상 모두 포착

**2023년 - 다중변수 강화 및 하이브리드 접근:**

- **MUTANT (Multivariate Time Series Anomaly Detection)**:[7]
  - GCN + LSTM Attention + VAE 결합
  - 변수 간 동적 상관관계 모델링
  - F1 > 0.96 달성

- **MACE (Multi-normal-pattern Accommodated Complex Event detection)**:[8]
  - 주파수 도메인 스파시티 활용
  - Fourier 기저의 부분 집합 선택으로 계산 효율 향상

**2024년 - 주파수 정보 통합의 정교화:**

- **FCVAE** (본 논문):[2]
  - 전역 + 국소 주파수 특성의 정교한 통합
  - 타겟 어텐션 메커니즘

- **TeVAE (Temporal VAE)**:[9]
  - 이산 온라인 이상 탐지
  - 연속 시계열로의 재매핑
  - 실산업 데이터셋에서 검증

**2025년 - 확률적 융합 및 강화학습 결합:**

- **LPCVAE (Long-term Dependency + Probabilistic time-frequency fusion)**:[10]
  - LSTM으로 장기 의존성 포착
  - Product-of-Experts (PoE) 메커니즘으로 확률적 시간-주파수 융합
  - FCVAE 대비 Yahoo 데이터셋에서 **6.3% F1 개선** (0.920 → 0.976)
  - NAB 데이터셋에서 F1 = 0.995 달성

- **DRSMT (Dynamic Reward Scaling for Multivariate Time Series)**:[11]
  - Deep Q-Network (DQN) + VAE + Active Learning
  - 멀티변수 데이터를 위한 강화학습 기반 접근

***

#### 5.2 방법론 비교 매트릭스



#### 5.3 재구성 기반 vs. 예측 기반 방법

**재구성 기반 (Reconstruction-based):**

장점:
- 혼합 이상-정상 훈련 데이터 처리 가능 (VAE, FCVAE 계열)
- 맥락적 및 집단 이상에 효과적

단점:
- Point 이상 탐지에서 위양성 발생 경향

**예측 기반 (Prediction-based):**

장점:
- Point 이상 탐지에 우수
- 값의 급격한 변화 포착 용이

단점:
- 순수 정상 데이터만으로 훈련 필요
- 주파수 변화 이상에 취약

**FCVAE의 입장:**

FCVAE는 **재구성 기반 패러다임**에 속하지만, 주파수 정보의 통합으로 예측 기반의 약점도 일부 보완합니다. 특히:

$$\text{Frequency domain} \rightarrow \text{Pattern-wise anomalies} \uparrow$$
$$\text{Time domain} \rightarrow \text{Point-wise anomalies}$$

의 상호보완 관계를 활용합니다.[6]

***

### 6. 한계와 미해결 과제

#### 6.1 FCVAE의 명시적 한계

논문에서 인정한 한계:[2]

1. **완전한 일반화 불가능성**
   - 새로운 도메인으로의 제로샷 전이 제한적
   - 도메인 특화 정보 부재 시 성능 저하

2. **극단적 이상 학습 필요**
   - 훈련 데이터에 없는 극단적 이상은 탐지 어려움
   - 예: 처음 보는 종류의 시스템 고장

3. **계절성-추세 분리 미흡**
   - 복잡한 계절성 + 추세 + 불규칙성 데이터에서는 TFAD 같은 분해 기반이 더 우수할 수 있음

#### 6.2 방법론 비교에서의 상대적 위치

| 도전과제 | FCVAE | LPCVAE | TFAD | Anomaly-Transformer |
|---------|-------|--------|------|-------------------|
| **장주기 패턴** | ✓ 개선 | ✓✓ 우수 | ✓✓ 우수 | ✓ 양호 |
| **단주기 추세** | ✓ 개선 | ✓✓ 우수 | ✓✓ 우수 | ✓ 양호 |
| **계산 효율** | ✓✓ 우수 | ✓ 양호 | ✓ 양호 | ✗ 느림 |
| **혼합 훈련 데이터** | ✓✓ 우수 | ✓✓ 우수 | ✓ 양호 | ✗ 취약 |
| **온라인 적응** | ✗ 어려움 | ✗ 어려움 | ✗ 어려움 | ✓ 가능 |

***

### 7. 향후 연구 방향 및 고려사항

#### 7.1 FCVAE 및 관련 연구의 영향

**이론적 기여:**
- VAE 기반 이상 탐지의 주파수 관점 도입
- 조건부 VAE의 효과성 실증적 증명
- 시간-주파수 이원 분석의 중요성 강조

**실무적 영향:**
- 웹 시스템의 실시간 이상 탐지에 적용 가능[2]
- 클라우드 시스템 모니터링 (대규모 검증됨)
- 스마트 센서 네트워크 이상 탐지

#### 7.2 향후 연구 시 고려할 점

**1. 도메인 일반화 (Domain Generalization)**

```
문제: FCVAE는 여전히 도메인 특화성이 높음
해결책:
- 도메인 불변 표현학습 (Domain-Invariant VAE, DIVAD)[34]
- 메타러닝 적용으로 빠른 도메인 적응
- 다중 도메인 사전학습 (Foundation Models)
```

**2. 동적 임계값 (Dynamic Threshold)**

```
기존 방법: 정적 임계값 τ
개선 방향:
- 적응형 동적 임계값[16]
- 베이지안 임계값 추정
- 온라인 학습 기반 임계값 업데이트
```

**3. 설명 가능성 (Interpretability)**

```
FCVAE의 약점: 블랙박스 특성
개선 안:
- 어떤 주파수 대역이 이상 탐지에 기여하는지 시각화
- SHAP/LIME 같은 설명 기법 적용
- 주파수 특성과 이상 유형의 관계 분석
```

**4. 다중변수 확장 (Multivariate Extension)**

```
FCVAE 현 상태: 단변수(Univariate)에 최적화
확장 방향:
- 변수 간 상관관계 명시적 모델링 (GCN, Attention)
- 참고: MUTANT, LPCVAE의 다중변수 구조
- 공간-시간 그래프 신경망 활용
```

**5. 강화학습 통합 (RL Integration)**

```
최신 동향: VAE + 강화학습 결합[3,14,25,39]
이점:
- 동적 보상 조정으로 이상/정상 경계 최적화
- 능동 학습으로 라벨링 비용 감소
- 순차적 의사결정에 적합
```

**6. 시계열 분해 통합 (Decomposition Integration)**

```
FCVAE → FCVAE + Decomposition:
- 계절성(S), 추세(T), 불규칙(R) 분리
- 각 성분별로 다른 주파수 범위 활용
- 예시: TFAD[67]의 분해 + FCVAE의 주파수 통합
```

#### 7.3 실제 적용 시 주의사항

**데이터 전처리:**
- FFT 계산 시 정규화 필수
- 누락 값(Missing Values) 처리 방법 결정
- 이상치 제거 vs. 학습 포함의 트레이드오프

**하이퍼파라미터 튜닝:**
- KL 항 가중치 $\lambda$: 재구성 vs. 정규화 균형
- 부분 윈도우 크기: 주파수 해상도와 시간 정밀성 트레이드오프
- 어텐션 헤드 수: 특성 표현 복잡도

**벤치마크 데이터셋:**
- Yahoo 데이터셋[2]
- NASA SMAP/MSL[2]
- Server Machine Dataset (SMD)
- 실제 클라우드 시스템 로그

***

### 8. 결론

FCVAE는 **주파수 관점의 도입**으로 VAE 기반 이상 탐지의 오랜 한계를 해결하는 획기적 연구입니다. 혼합 훈련 데이터 처리, 이질적 패턴 포착, 계산 효율성의 삼각형에서 좋은 균형을 제시합니다.[2]

그러나 2025년까지의 후속 연구(특히 LPCVAE)들은 **확률적 시간-주파수 융합**, **장기 의존성 모델링**, **강화학습 통합** 등으로 추가 개선의 여지를 보여줍니다. 앞으로의 연구는:

1. **도메인 일반화** 해결이 핵심
2. **설명 가능한 주파수 표현** 개발
3. **온라인 적응 메커니즘** 추가
4. **다중변수 + 공간-시간 모델링** 확대

이들 방향으로 진화할 것으로 예상되며, FCVAE는 이 진화 경로의 **중요한 이정표**로 평가됩니다.

***

"Revisiting VAE for Unsupervised Time Series Anomaly Detection: A Frequency Perspective", WWW 2024, February 4, 2024[1]

 FCVAE - Core paper methodology and contributions[2]

 "Dynamic Reward Scaling for Multivariate Time Series Anomaly Detection: A VAE-Enhanced Reinforcement Learning Approach", 2025[11]

 "LPCVAE: A Conditional VAE with Long-Term Dependency and Probabilistic Time-Frequency Fusion for Time Series Anomaly Detection", 2025[10]

 "VELC: A New Variational AutoEncoder Based Model for Time Series Anomaly Detection", 2020[3]

 "Robust anomaly detection for multivariate time series through temporal GCNs and attention-based VAE" (MUTANT), 2023[7]

 "Unsupervised Anomaly Detection in Multivariate Time Series with Domain Shift" (DIVAD), 2023[12]

 "TeVAE: A Variational Autoencoder Approach for Discrete Online Anomaly Detection in Variable-state Multivariate Time-series Data", 2024[9]

 "Calibrated Unsupervised Anomaly Detection in Multivariate Time-series using Reinforcement Learning", 2025[13]

 "Learning Multi-Pattern Normalities in the Frequency Domain Domain" (MACE), 2023[8]

 "TiSAT: Time Series Anomaly Transformer", 2022[5]

 "Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy", 2021[4]

 "TFAD: A Decomposition Time Series Anomaly Detection Architecture with Time-Frequency Analysis", 2022[6]

출처
[1] 2402.02820v1.pdf https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/704b11f2-9e8b-4bdb-971f-b137f5998a2a/2402.02820v1.pdf
[2] Revisiting VAE for Unsupervised Time Series Anomaly Detection: A Frequency Perspective https://dl.acm.org/doi/10.1145/3589334.3645710
[3] VELC: A New Variational AutoEncoder Based Model for Time Series Anomaly
  Detection https://arxiv.org/abs/1907.01702v1
[4] Time Series Anomaly Detection with Association Discrepancy https://arxiv.org/abs/2110.02642
[5] TiSAT: Time Series Anomaly Transformer https://arxiv.org/abs/2203.05167
[6] TFAD: A Decomposition Time Series Anomaly Detection ... - arXiv https://arxiv.org/abs/2210.09693
[7] Robust anomaly detection for multivariate time series through temporal GCNs and attention-based VAE https://www.sciencedirect.com/science/article/abs/pii/S0950705123004756
[8] Learning Multi-Pattern Normalities in the Frequency ... https://arxiv.org/pdf/2311.16191.pdf
[9] TeVAE: A Variational Autoencoder Approach for Discrete ... https://arxiv.org/abs/2407.06849
[10] LPCVAE: A Conditional VAE with Long-Term Dependency and Probabilistic Time-Frequency Fusion for Time Series Anomaly Detection https://arxiv.org/abs/2510.10915
[11] Dynamic Reward Scaling for Multivariate Time Series Anomaly Detection: A VAE-Enhanced Reinforcement Learning Approach https://arxiv.org/abs/2511.12351
[12] Unsupervised Anomaly Detection in Multivariate Time ... https://arxiv.org/html/2503.23060v1
[13] Calibrated Unsupervised Anomaly Detection in Multivariate ... https://www.arxiv.org/pdf/2502.03245.pdf
[14] Unsupervised Anomaly Detection in Time Series Data via Enhanced VAE-Transformer Framework https://www.techscience.com/cmc/v84n1/61729
[15] Similaritycheck-Time series anomaly detection based on GAN-VAE https://linkinghub.elsevier.com/retrieve/pii/S269461062500013X
[16] Enhancing Industrial Time Series Anomaly Detection Through Graph Modeling and a Hybrid VAE-GAN Approach https://link.springer.com/10.1007/978-3-031-97576-9_7
[17] Anomaly Detection in Multidimensional Time Series Based on a Lightweight GE-GRU-VAE Model https://www.hanspub.org/journal/paperinformation?paperid=126920
[18] Corrigendum to "Online model-based anomaly detection in multivariate time series: Taxonomy, survey, research challenges and future directions" [Eng. Appl. Artif. Intell. 138 (2024) 109323] https://linkinghub.elsevier.com/retrieve/pii/S0952197625001708
[19] Anomaly Detection in Multivariate Time Series with Contaminated Training Data Using VAE https://ieeexplore.ieee.org/document/10565121/
[20] Variational Graph Attention Networks With Self-Supervised Learning for Multivariate Time Series Anomaly Detection https://ieeexplore.ieee.org/document/10758697/
[21] StackVAE-G: An efficient and interpretable model for time series anomaly
  detection https://arxiv.org/pdf/2105.08397.pdf
[22] A Joint Model for IT Operation Series Prediction and Anomaly Detection https://arxiv.org/pdf/1910.03818.pdf
[23] Anomaly Detection in Time Series Data Using Reinforcement Learning,
  Variational Autoencoder, and Active Learning https://arxiv.org/pdf/2504.02999.pdf
[24] Harnessing Feature Clustering For Enhanced Anomaly Detection With
  Variational Autoencoder And Dynamic Threshold https://arxiv.org/pdf/2407.10042.pdf
[25] Fast Particle-based Anomaly Detection Algorithm with Variational
  Autoencoder https://arxiv.org/pdf/2311.17162.pdf
[26] NVAE-GAN Based Approach for Unsupervised Time Series Anomaly Detection https://arxiv.org/pdf/2101.02908.pdf
[27] TeVAE: A Variational Autoencoder Approach for Discrete Online Anomaly
  Detection in Variable-state Multivariate Time-series Data https://arxiv.org/html/2407.06849
[28] Revisiting VAE for Unsupervised Time Series Anomaly ... https://netman.aiops.org/wp-content/uploads/2024/08/Revisiting-VAE-for-Unsupervised-Time-Series-Anomaly-Detection-A-Frequency-Perspective.pdf
[29] Unsupervised Anomaly Detection on Metal Surfaces Based ... https://pubmed.ncbi.nlm.nih.gov/40218761/
[30] [Time-series 논문 리뷰] Revisiting VAE for ... - 딩딩딩2 - 티스토리 https://yoonji-ha.tistory.com/51
[31] Time and Frequency Domain-based Anomaly Detection ... https://arxiv.org/html/2504.18231v1
[32] 2504.02999 https://papers.cool/arxiv/2504.02999
[33] Learning Multi-Pattern Normalities in the Frequency ... https://openreview.net/pdf?id=cAIyA5sCew
[34] Unsupervised Anomaly Detection in Time Series Data via ... https://www.sciencedirect.com/org/science/article/pii/S1546221825005107
[35] Revisiting VAE for Unsupervised Time Series Anomaly Detection: A Frequency Perspective https://arxiv.org/abs/2402.02820
[36] Time series anomaly detection based on time–frequency ... https://www.sciencedirect.com/science/article/abs/pii/S0952197625028064
[37] [논문 리뷰] Revisiting VAE for Unsupervised Time Series ... https://www.themoonlight.io/ko/review/revisiting-vae-for-unsupervised-time-series-anomaly-detection-a-frequency-perspective
[38] [2504.02999] Anomaly Detection in Time Series Data ... https://arxiv.org/abs/2504.02999
[39] Unsupervised Anomaly Detection for Digital Radio ... https://ieeexplore.ieee.org/document/8260738/
[40] LPCVAE: A Conditional VAE with Long-Term Dependency ... https://arxiv.org/pdf/2510.10915.pdf
[41] Unsupervised Time-Series Signal Analysis with ... https://www.arxiv.org/pdf/2504.16972v1.pdf
[42] Unsupervised Incremental Learning with Dual Concept ... https://arxiv.org/pdf/2403.03576.pdf
[43] $K^2$VAE: A Koopman-Kalman Enhanced Variational ... https://arxiv.org/pdf/2505.23017.pdf
[44] Variational Autoencoder for Anomaly Detection https://arxiv.org/html/2408.13561v1
[45] LPCVAE: A Conditional VAE with Long-Term Dependency ... https://arxiv.org/html/2510.10915v1
[46] Time and Frequency Domain-based Anomaly Detection in ... https://arxiv.org/pdf/2504.18231.pdf
[47] VAE-IF: Deep feature extraction with averaging for fully ... https://arxiv.org/abs/2312.05959
[48] LLM-Enhanced Reinforcement Learning for Time Series ... https://arxiv.org/html/2601.02511v1
[49] Towards Foundation Auto-Encoders for Time-Series ... https://arxiv.org/html/2507.01875v1
[50] ech https://file.techscience.com/files/cmc/2025/online/CMC0411/TSP_CMC_63151/TSP_CMC_63151.pdf
[51] LGAT: A novel model for multivariate time series anomaly detection with improved anomaly transformer and learning graph structures https://linkinghub.elsevier.com/retrieve/pii/S0925231224017958
[52] Anomaly Detection in Time Series Data Using Reversible Instance Normalized Anomaly Transformer https://www.mdpi.com/1424-8220/23/22/9272
[53] Decompose Auto-Transformer Time Series Anomaly Detection for Network Management https://www.mdpi.com/2079-9292/12/2/354
[54] Transformer-based multivariate time series anomaly detection using inter-variable attention mechanism https://linkinghub.elsevier.com/retrieve/pii/S0950705124001424
[55] From anomaly detection to classification with graph attention and transformer for multivariate time series https://linkinghub.elsevier.com/retrieve/pii/S1474034624000053
[56] Enhanced Anomaly Detection in IoT: A Transformer Based Approach for Multivariate Time Series data https://ieeexplore.ieee.org/document/10581104/
[57] Research on Time Series Anomaly Detection Algorithm Based on Transformer Coupled with GAN https://ieeexplore.ieee.org/document/10761510/
[58] Multivariate Time Series Anomaly Detection with Adaptive Transformer-CNN Architecture Fusing Adversarial Training https://ieeexplore.ieee.org/document/10606841/
[59] Dynamic deep graph convolution with enhanced transformer networks for time series anomaly detection in IoT https://link.springer.com/10.1007/s10586-024-04707-w
[60] A two-stage adversarial Transformer based approach for multivariate industrial time series anomaly detection https://link.springer.com/10.1007/s10489-024-05395-0
[61] MAAT: Mamba Adaptive Anomaly Transformer with association discrepancy
  for time series https://arxiv.org/pdf/2502.07858.pdf
[62] RESTAD: REconstruction and Similarity based Transformer for time series
  Anomaly Detection http://arxiv.org/pdf/2405.07509.pdf
[63] TCF-Trans: Temporal Context Fusion Transformer for Anomaly Detection in Time Series https://www.mdpi.com/1424-8220/23/20/8508/pdf?version=1697524860
[64] AnomalyBERT: Self-Supervised Transformer for Time Series Anomaly
  Detection using Data Degradation Scheme https://arxiv.org/pdf/2305.04468.pdf
[65] MEMTO: Memory-guided Transformer for Multivariate Time Series Anomaly
  Detection http://arxiv.org/pdf/2312.02530.pdf
[66] TransNAS-TSAD: Harnessing Transformers for Multi-Objective Neural
  Architecture Search in Time Series Anomaly Detection http://arxiv.org/pdf/2311.18061.pdf
[67] EdgeConvFormer: Dynamic Graph CNN and Transformer based Anomaly
  Detection in Multivariate Time Series https://arxiv.org/pdf/2312.01729.pdf
[68] Deep Learning Advancements in Anomaly Detection https://arxiv.org/html/2503.13195v1
[69] RTdetector: Deep Transformer Networks for Time Series ... https://www.ijcai.org/proceedings/2025/0644.pdf
[70] TFAD: A Decomposition Time Series Anomaly Detection ... https://arxiv.org/pdf/2210.09693.pdf
[71] AER: Auto-Encoder with Regression for Time Series Anomaly ... https://dai.lids.mit.edu/wp-content/uploads/2022/11/Lawrence_2022_aer_paper_IEEE.pdf
[72] Time Series Anomaly Detection with Association Discrepancy https://www.semanticscholar.org/paper/Anomaly-Transformer:-Time-Series-Anomaly-Detection-Xu-Wu/a46b06a4b8b4deecf96a4e42cd19b4696f999e66
[73] Time-Will-Tell/anomaly detection/TFAD (CIKM 22).md at main · pseudo-Skye/Time-Will-Tell https://github.com/pseudo-Skye/Time-Will-Tell/blob/main/anomaly%20detection/TFAD%20(CIKM%2022).md
[74] A Novel Unsupervised Video Anomaly Detection ... https://pmc.ncbi.nlm.nih.gov/articles/PMC10221939/
[75] [논문리뷰] Anomaly transformer: Time series ... https://jisoo0-0.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/2024/01/17/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-AnomalyTransformer.html
[76] Papers with Code - TFAD: A Decomposition Time Series Anomaly Detection Architecture with Time-Frequency Analysis https://paperswithcode.com/paper/tfad-a-decomposition-time-series-anomaly
[77] Reconstruction-based anomaly detection for multivariate ... https://www.sciencedirect.com/science/article/abs/pii/S0306457323003060
[78] Time Series Anomaly Detection with Association Discrepancy https://liner.com/review/anomaly-transformer-time-series-anomaly-detection-with-association-discrepancy
[79] TFAD:分解时间序列(TFAD: A Decomposition Time Series ... https://www.zhuanzhiai.com/paper/30eb07808b046297eeb7108bcb155cb6
[80] Making Reconstruction-based Method Great Again for ... https://ui.adsabs.harvard.edu/abs/2023arXiv230112048W/abstract
[81] Foundation Models for Time Series: A Survey https://arxiv.org/html/2504.04011v1
[82] Forecast2Anomaly (F2A): Adapting Multivariate Time ... https://arxiv.org/html/2511.03149v1
[83] Benchmarking Unsupervised Strategies for Anomaly ... https://arxiv.org/html/2506.20574v1
[84] MAAT: Mamba Adaptive Anomaly Transformer with ... https://arxiv.org/html/2502.07858v1
[85] [PDF] TFAD: A Decomposition Time Series Anomaly Detection ... https://www.semanticscholar.org/paper/f7561b056fa5587e048393d03998183318518f37
[86] Single-Step Reconstruction-Free Anomaly Detection and ... https://arxiv.org/html/2508.04818v2
[87] Transformer-based Multivariate Time Series Anomaly ... https://arxiv.org/html/2501.08628v1
[88] Quantifying the Benefit of Supervised Time Series Anomaly ... https://arxiv.org/html/2511.16145v1
[89] Spatio-temporal prediction and reconstruction network for ... https://pubmed.ncbi.nlm.nih.gov/35617331/
[90] Decomposition-based multi-scale transformer framework ... https://arxiv.org/html/2504.14206v1
[91] (PDF) RobustTAD: Robust Time Series Anomaly Detection ... http://www.arxiv.org/pdf/2002.09545v1.pdf
[92] Making Reconstruction-based Method Great Again for ... https://arxiv.org/abs/2301.12048
