# DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks

### 1. 논문의 핵심 주장 및 기여도 요약

**DeepAR**은 Amazon의 연구진이 제시한 획기적인 확률적 시계열 예측 방법론입니다. 논문의 핵심 주장은 다음과 같습니다: 고전적 시계열 예측 방법(ARIMA, 지수평활화 등)이 개별 또는 소수의 시계열에만 최적화되어 있다면, **수천 개 이상의 관련 시계열을 동시에 학습하는 글로벌 모델**을 구축하여 더 정확한 확률적 예측을 생성할 수 있습니다.[1]

**주요 기여도는 다음 두 가지입니다:**

1. **RNN 기반 확률적 예측 아키텍처**: 음이항분포(Negative Binomial Likelihood)와 가우시안 우도를 결합하여, 카운트 데이터와 연속값 모두 처리 가능한 유연한 프레임워크 제시[1]

2. **스케일 불균형 문제 해결**: Amazon의 실제 데이터에서 500,000개 항목의 판매량이 멱함수(power-law) 분포를 따르는 관찰을 바탕으로, 아이템 의존적 스케일 팩터를 통한 정규화와 가중 샘플링 스키마 개발[1]

***

### 2. 모델 구조 및 수식

#### 2.1 기본 확률 모델

DeepAR의 핵심은 다음과 같은 조건부 분포 모델링입니다:

$$Q_\Theta(z_{i,t_0:T}|z_{i,1:t_0-1}, x_{i,1:T}) = \prod_{t=t_0}^{T} Q_\Theta(z_{i,t}|z_{i,1:t-1}, x_{i,1:T})$$

$$= \prod_{t=t_0}^{T} \ell(z_{i,t}|\theta(h_{i,t}, \Theta))$$

여기서:
- $z_{i,t}$: 시계열 $i$의 시점 $t$에서의 값
- $h_{i,t}$: LSTM 네트워크의 은닉 상태
- $\ell(\cdot|\theta)$: 우도 함수 (확률 분포)
- $\Theta$: 모델 파라미터[1]

#### 2.2 자동회귀 RNN 구조

$$h_{i,t} = h(h_{i,t-1}, z_{i,t-1}, x_{i,t}, \Theta)$$

이 식은 LSTM 셀을 통해 다음 정보를 처리합니다:
- **이전 은닉 상태** $h_{i,t-1}$: 시간적 맥락 유지
- **이전 타겟 값** $z_{i,t-1}$: 자동회귀 특성 구현
- **공변량** $x_{i,t}$: 외생 정보 통합[1]

#### 2.3 우도 함수

**가우시안 우도** (연속값 데이터):

$$\ell_G(z|\mu, \sigma) = (2\pi\sigma^2)^{-1/2} \exp\left(-\frac{(z-\mu)^2}{2\sigma^2}\right)$$

$$\mu(h_{i,t}) = w_\mu^T h_{i,t} + b_\mu$$

$$\sigma(h_{i,t}) = \log(1 + \exp(w_\sigma^T h_{i,t} + b_\sigma))$$

**음이항 우도** (카운트 데이터):

$$\ell_{NB}(z|\mu, \alpha) = \frac{\Gamma(z + 1/\alpha)}{\Gamma(z+1)\Gamma(1/\alpha)} \left(\frac{1}{1+\alpha\mu}\right)^{1/\alpha} \left(\frac{\alpha\mu}{1+\alpha\mu}\right)^z$$

$$\mu(h_{i,t}) = \log(1 + \exp(w_\mu^T h_{i,t} + b_\mu))$$

$$\alpha(h_{i,t}) = \log(1 + \exp(w_\alpha^T h_{i,t} + b_\alpha))$$

음이항 분포에서 분산은 $\text{Var}[z] = \mu + \mu^2\alpha$로 정의되며, 이는 과산포(overdispersion) 문제를 자연스럽게 모델링합니다.[1]

#### 2.4 학습 목적 함수

$$\mathcal{L} = \sum_{i=1}^{N} \sum_{t=t_0}^{T} \log \ell(z_{i,t}|\theta(h_{i,t}))$$

이 목적함수는 모든 시계열에 걸쳐 음의 로그우도를 최소화하며, 확률적 예측의 정확도를 직접 최적화합니다.[1]

#### 2.5 스케일 처리 메커니즘

**입출력 정규화**:

$$\tilde{z}_{i,t} = \frac{z_{i,t}}{\nu_i}$$

$$\mu = \nu_i \log(1 + \exp(o_\mu)), \quad \alpha = \frac{\log(1+\exp(o_\alpha))}{\sqrt{\nu_i}}$$

여기서 $\nu_i = 1 + \frac{1}{t_0}\sum_{t=1}^{t_0} z_{i,t}$는 아이템별 스케일 팩터입니다.[1]

**가중 샘플링 스키마**:

$$P(\text{선택}|\nu_i) \propto \nu_i$$

이 메커니즘은 스케일이 큰 아이템(높은 판매량)이 훈련 중에 더 자주 샘플링되도록 하여, 불균형한 데이터셋에서의 언더피팅을 방지합니다.[1]

***

### 3. 성능 향상 및 실험 결과

#### 3.1 벤치마크 데이터셋 및 비교

논문은 5개 데이터셋에서 평가했습니다:

| 데이터셋 | 시계열 수 | 주기 | 도메인 |
|---------|----------|------|--------|
| parts | 1,046 | 월간 | 자동차 부품 판매 |
| electricity | 370 | 시간 | 전력 소비 |
| traffic | 963 | 시간 | 교통 점유율 |
| ec-sub | 39,700 | 주간 | Amazon 상품 판매 |
| ec | 534,884 | 주간 | Amazon 상품 판매 |

#### 3.2 정량적 성능 향상

**ec-sub 데이터셋 (0.5-risk 메트릭)**:

| 모델 | 0.5-risk | 0.9-risk | 평균 |
|------|----------|----------|------|
| ISSM (기준) | 1.00 | 1.00 | 1.00 |
| ETS | 0.83 | 1.09 | 0.96 |
| rnn-gaussian | 1.03 | 0.91 | 0.97 |
| rnn-negbin | 0.90 | 1.23 | 1.06 |
| **DeepAR** | **0.64** | **0.71** | **0.67** |

DeepAR은 ISSM(기준선)에 비해 **33%의 성능 향상**을 달성했습니다.[1]

**ec 데이터셋 (전체)**:

| 모델 | 평균 0.5-risk |
|------|---|
| DeepAR | **0.85** |
| rnn-negbin | 0.93 |
| ISSM | 1.00 |

#### 3.3 정성적 분석

**불확실성 성장 패턴** (Figure 4):

- **ISSM**: 선형적 불확실성 성장 (모델의 가정)
- **DeepAR**: 데이터로부터 학습한 비선형 성장
  - Q4(4분기)에서 불확실성 증가
  - 이후 급격히 감소 (실제 연말 수요 패턴 반영)[1]

**확률적 보정 (Probabilistic Calibration)**:

이상적 보정에서 모든 백분위수 $p$에 대해 $\text{Coverage}(p) = p$가 성립해야 합니다.

- **ISSM**: 다양한 시간 범위에서 보정 오류 발생
- **DeepAR**: 대부분의 백분위수에서 $\text{Coverage}(p) \approx p$ 달성[1]

**시간적 상관 구조**:

Figure 5의 셔플 테스트(shuffle test)에서:
- 단일 시점 예측($S=1$): 차이 없음
- 장기간($S=9$): DeepAR의 상관성 파괴 후 0.9-risk 10% 악화

이는 DeepAR이 시간 단계 간의 중요한 시간적 상관성을 캡처함을 증명합니다.[1]

***

### 4. 모델의 일반화 성능과 한계

#### 4.1 일반화 성능의 강점

**1. 전이 학습을 통한 Cold-start 해결**

고전적 방법이 불가능한 새로운 상품의 판매 예측이 가능합니다. DeepAR은 관련 상품들로부터 학습한 패턴을 새로운 상품에 전이할 수 있습니다.[1]

**2. 스케일 불변성 (Scale Invariance)**

마다가스카르(Madagascar) 논문에서 확인된 멱함수 분포에서도 일관된 성능:
- 저속 상품(slow-moving): 약 20%의 예측 오차
- 고속 상품(fast-moving): 약 5-10%의 예측 오차
- 균등한 상대적 성능을 유지[1]

**3. 최소 수동 특성 공학**

기존 방법의 필수 요소들(이상치 감지, 변환, 계절성 처리)이 자동으로 학습됩니다:
- 논문의 실험에서 "age" 특성(시계열의 나이)만 수동으로 추가
- 요일, 주차, 월 등의 시간 정보는 자동 추출[1]

#### 4.2 일반화 성능의 한계

**1. 분포 변화(Distribution Shift)에 취약성**

저자들이 명시하지는 않았지만, 다음과 같은 한계가 존재합니다:
- 훈련 기간과 테스트 기간의 통계적 특성이 크게 다를 경우 성능 저하
- 예: COVID-19 팬데믹처럼 근본적인 시장 변화가 있을 때[2]

**2. 자동회귀 노출 편향 (Exposure Bias)**

- 훈련 중: 실제 과거 값 $z_{i,t-1}$ 사용
- 예측 중: 샘플된 값 $\tilde{z}_{i,t}$ 사용
- 이론적으로 불일치 가능, 실제로는 문제 미미[1]

**3. 장기 예측의 누적 오차**

예측 범위(prediction horizon)가 길수록:
- 불확실성 영역이 기하급수적으로 확대
- 특히 계절성이 약한 데이터에서 정확도 저하[1]

***

### 5. 2020년 이후 관련 최신 연구와의 비교

#### 5.1 Temporal Fusion Transformer (TFT, 2020)

**핵심 혁신**:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- 변수 선택 네트워크(VSN)로 각 특성의 중요도 명시적 학습
- 다중 헤드 어텐션으로 계절성 패턴 포착[3][4]

**DeepAR 대비 장점**:
- 해석 가능성 향상 (각 시점의 기여도 시각화)
- 다양한 입력 유형(정적, 알려진 미래, 과거 관찰) 통합[3]

**한계**:
- 계산 복잡도 $O(n^2)$로 장기 시계열에서 비효율적
- DeepAR의 간단한 구조보다 초기화에 민감[5]

#### 5.2 N-BEATS (Neural Basis Expansion, 2020)

**혁신**:

$$\theta_t = \text{MLP}(\text{입력})$$

$$\text{예측}_t = \theta_t \cdot \text{기저함수}(t)$$

- 해석 가능한 신경 기저 확장 (polynomial, harmonic basis)
- 초기 연구에서 M4 경쟁에서 통계 방법과 하이브리드 모델 모두 초과[6]

**vs DeepAR**:
- **N-BEATS**: 단변량 점 예측에 최적화, 확률 분포 미지원[7]
- **DeepAR**: 다변량, 확률적 예측, 온라인 학습 친화적

**결합 시도**:
- N-HiTS (2022): 계층적 시간 샘플링으로 장기 예측 개선, MSE 17% 향상[8]

#### 5.3 Transformer 기반 모델의 발전 (2021-2025)

**주요 문제점**: 자기-어텐션(self-attention)의 순열 불변성(permutation invariance)은 시계열의 시간 순서를 보존하지 못함[9]

**최신 해결책**:

1. **PatchTST (2023)**:[10]

$$\text{패치}_j = [z_{i,t}, z_{i,t+1}, \ldots, z_{i,t+P-1}]$$
   
   - 작은 시간 윈도우를 토큰화하여 지역 의미론 보존
   - 채널 독립성으로 다변량 의존성 유연하게 모델링

2. **iTransformer (2024)**:[11][12]

$$\text{어텐션}(\text{변량}_1, \text{변량}_2, \ldots, \text{변량}_M)$$

   - 역방향 트랜스포머: 시간 단계 대신 변량을 토큰으로 처리
   - 다변량 상관성 학습에 특화, 고차원 시계열에서 SOTA[12]

3. **CATS (Cross-Attention-only, 2024)**:[9]
   - 자기-어텐션 제거, 교차-어텐션만 사용
   - 시계열의 시간적 의존성 더 효율적으로 포착

#### 5.4 확산 모델 (Diffusion Models, 2023-2025)

**기본 원리**:

$$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$$

$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

**시계열 예측에의 응용**:[13][14][15]

- TimeGrad (2021): DDPM 기반, 조건부 시계열 생성
- TSDiff (2023): 비조건부 모델, 자기 안내 메커니즘[16]
- MG-TSD (2024): 다중 해상도 가이던스, 확산 과정 중간 목표 설정
- ARMD (2024): 연속 시퀀셜 확산, ARMA 이론과 연결[17]

**장점**:
- 높은 불확실성 정량화 (confidence intervals)
- 다양한 시나리오 생성 가능 (synthetic data)[18]
- 최근 SOTA: 일부 벤치마크에서 DeepAR 계열 모델 초과 (9-47% 개선)[18]

**단점**:
- 계산 비용 높음 (다단계 역확산)
- 불안정성 문제 여전히 존재 (확률적 특성)[17]

#### 5.5 하이브리드 모델 (2021-2025)

| 모델 | 구성 | 성능 특징 |
|------|------|---------|
| ES-RNN[19] | 지수평활화 + LSTM | 해석성 + 딥러닝 |
| DeepVARMA[20] | LSTM + VARMA | 추세 + 잔차 분해 |
| TFT-Improved[5] | TCN + TFT | 다중 시간 스케일 |
| Diffinformer[21] | Informer + Diffusion | ProbSparse + 확률 |

**가장 유망한 발전**: DeepAR의 자동회귀 구조와 확산 모델의 불확실성 정량화를 결합한 모델들[22][17]

***

### 6. 앞으로의 연구에 미치는 영향과 고려 사항

#### 6.1 DeepAR의 지속적 영향

**1. 산업 표준화**

- AWS Forecast 서비스에 DeepAR 기본 알고리즘 탑재
- 전력, 유통, 금융 분야에서 실제 배포 중[23]

**2. 후속 연구의 토대**

DeepAR의 아키텍처를 기반으로 한 확장:
- 다변량 시계열 처리 개선[20]
- 확률적 예측의 이론적 토대 구축[24]

**3. 방법론적 기여**

- **스케일 정규화**: 다양한 크기의 데이터셋에 적용 가능한 일반 기법
- **음이항 우도**: 과산포 시계열 모델링의 표준화
- **전이 학습**: 관련 시계열 간 지식 공유의 체계화[1]

#### 6.2 향후 연구 시 고려할 점

**1. 분포 변화 강건성 (Domain Adaptation)**

현재의 문제점:
- 장기 구간에서 통계적 특성이 변할 때 성능 저하
- 예: 계절성 강도 변화, 추세 변화[25]

향후 연구 방향:
- 적응형 정상화 모듈 (Non-Stationary Transformers, 2022)[26]
- 온라인 학습 및 점진적 업데이트 메커니즘
- 다중 도메인 메타-학습[24]

**2. 불확실성 정량화 개선**

현재 DeepAR의 한계:
- 제한된 우도 선택 (가우시안, 음이항)
- 꼬리 위험(tail risk) 미흡

해결책:
- 정규화 흐름(Normalizing Flows) 통합[27]
- 베이지안 심층 학습으로 에피스테믹 불확실성 포착[28]

**3. 계산 효율성**

현재:
- 대규모 데이터셋(500K+ 시계열)에서 10시간 소요 (1 GPU)[1]

개선 가능 영역:
- 경량 아키텍처 (GRU vs LSTM 비교)[29]
- 스파스 어텐션 메커니즘[30]
- 양자화(quantization) 기법[31]

**4. 해석성 강화**

DeepAR의 부족함:
- 개별 예측의 이유 설명 어려움
- 특성 중요도 순위 불명확

개선 안:
- Temporal Fusion Transformer의 어텐션 시각화 통합
- SHAP 값 기반 특성 기여도 분석
- 계층적 분해(hierarchical decomposition)[32]

**5. 멀티모달 시계열 예측**

향후 방향:
- 텍스트, 이미지와 함께 시계열 활용[33]
- 대형 언어 모델(LLM)과의 결합[34]
- 크로스 도메인 전이 학습[32]

#### 6.3 구체적 적용 시 체크리스트

시계열 예측 프로젝트에서 DeepAR 사용을 고려할 때:

1. **데이터 준비**
   - 최소 100-1000개의 관련 시계열 필요
   - 각 시계열당 최소 50-100개 시점[1]
   - 이상치 및 결측값 전처리

2. **모델 선택 기준**
   - **DeepAR 적합**: 수백~수백만 관련 시계열, 확률적 예측 필요
   - **N-BEATS**: 단변량, 점 예측 중심, 해석성 우선
   - **TFT**: 다양한 입력 유형, 해석성 중시
   - **Diffusion**: 극도의 불확실성 정량화 필요

3. **평가 지표**
   - 점 예측: RMSE, MAE, MAPE
   - 확률적 예측: Quantile Loss, CRPS, Coverage[1]

4. **배포 전략**
   - 온라인 학습 파이프라인 구축
   - 주기적 재학습 (월 1회 이상)
   - 분포 변화 감지 시스템[25]

***

### 7. 결론

DeepAR은 2017년 발표 이후 시계열 예측 분야에 패러다임을 제시했습니다. **확률적 예측의 중요성 부각**, **대규모 관련 시계열의 동시 학습**, **스케일 불균형 문제 해결** 등은 이후 수년간의 연구에 영향을 미쳤습니다.

다만 2020년 이후 Transformer 기반 모델과 확산 모델의 부상으로, 다음과 같은 트렌드가 형성되었습니다:

| 시기 | 주요 혁신 | 대표 모델 |
|------|---------|---------|
| 2017-2019 | RNN 기반 확률 예측 | DeepAR |
| 2020-2021 | 신경 기저 확장, 트랜스포머 도입 | N-BEATS, TFT |
| 2022-2023 | 패치 기반 아키텍처, 비정상성 처리 | PatchTST, Non-Stationary |
| 2024-2025 | 확산 모델, 역방향 트랜스포머 | Diffusion, iTransformer |

**향후 방향**: DeepAR의 자동회귀 우도 기반 접근법과 최신 아키텍처(Transformer, Diffusion)의 융합, 그리고 분포 변화와 해석성 문제의 동시 해결이 차세대 시계열 예측의 핵심 과제가 될 것으로 예측됩니다.[21][18]

***

### 참고문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/926376bc-26c1-4cbe-8c61-04f15aba3317/1704.04110v3.pdf)
[2](https://arxiv.org/html/2411.05793v1)
[3](https://linkinghub.elsevier.com/retrieve/pii/S0169207021000637)
[4](https://arxiv.org/pdf/1912.09363.pdf)
[5](https://pmc.ncbi.nlm.nih.gov/articles/PMC9407224/)
[6](https://openreview.net/pdf?id=r1ecqn4YwB)
[7](https://arxiv.org/abs/1905.10437)
[8](https://arxiv.org/pdf/2201.12886.pdf)
[9](https://proceedings.neurips.cc/paper_files/paper/2024/file/cf66f995883298c4db2f0dcba28fb211-Paper-Conference.pdf)
[10](https://arxiv.org/html/2510.23396v1)
[11](https://www.datasciencewithmarco.com/blog/itransformer-the-latest-breakthrough-in-time-series-forecasting)
[12](http://arxiv.org/pdf/2310.06625.pdf)
[13](https://arxiv.org/abs/2401.03006)
[14](https://arxiv.org/pdf/2307.11494.pdf)
[15](https://arxiv.org/pdf/2401.03006.pdf)
[16](https://arxiv.org/abs/2307.11494)
[17](https://arxiv.org/abs/2412.09328)
[18](https://www.emergentmind.com/topics/diffusion-models-in-time-series-forecasting)
[19](https://arxiv.org/html/2503.10198v1)
[20](https://arxiv.org/pdf/2404.17615.pdf)
[21](https://www.sciencedirect.com/science/article/abs/pii/S0957417425035596)
[22](https://arxiv.org/pdf/2412.09328.pdf)
[23](https://arxiv.org/html/2407.14377v1)
[24](https://arxiv.org/pdf/2305.17028.pdf)
[25](https://arxiv.org/pdf/2511.07059.pdf)
[26](https://arxiv.org/html/2503.06928v1)
[27](https://arxiv.org/html/2301.06650v3)
[28](https://arxiv.org/pdf/2102.00397.pdf)
[29](https://pmc.ncbi.nlm.nih.gov/articles/PMC8402357/)
[30](https://milvus.io/ai-quick-reference/how-do-attention-mechanisms-enhance-time-series-forecasting-models)
[31](https://journals.sagepub.com/doi/10.1177/21582440251359828)
[32](https://www.nature.com/articles/s41598-023-39301-4)
[33](https://arxiv.org/pdf/2406.08627.pdf)
[34](https://arxiv.org/html/2507.10098v1)
[35](https://arxiv.org/pdf/1704.04110.pdf)
[36](https://scindeks.ceon.rs/Article.aspx?artid=0354-30992434005F)
[37](https://iopscience.iop.org/article/10.1149/MA2024-02111mtgabs)
[38](https://iopscience.iop.org/article/10.1149/MA2025-031244mtgabs)
[39](https://onepetro.org/SPEOGWA/proceedings/25OPES/25OPES/D021S023R002/673742)
[40](https://invergejournals.com/index.php/ijss/article/view/117)
[41](https://academic.oup.com/ijnp/article/28/Supplement_1/i276/8009422)
[42](https://arxiv.org/pdf/2302.02597.pdf)
[43](https://arxiv.org/pdf/2302.11241.pdf)
[44](https://arxiv.org/pdf/2204.06848.pdf)
[45](https://arxiv.org/html/2307.10422v2)
[46](http://arxiv.org/pdf/2310.19322.pdf)
[47](https://arxiv.org/abs/1704.04110)
[48](https://pmc.ncbi.nlm.nih.gov/articles/PMC12329085/)
[49](https://www.sciencedirect.com/org/science/article/pii/S1546221825008872)
[50](https://www.sciencedirect.com/science/article/abs/pii/S0925231223000462)
[51](https://www.sciencedirect.com/science/article/abs/pii/S036083522300668X)
[52](https://journal.hep.com.cn/fcs/EN/10.1007/s11704-025-50947-3)
[53](https://jrasb.com/index.php/jrasb/article/download/712/669/1739)
[54](https://velog.io/@yetsyl0705/DeepAR-Probabilistic-Forecasting-with-Autoregressive-Recurrent-Networks)
[55](https://arxiv.org/html/2506.14831v2)
[56](https://arxiv.org/html/2503.06072v3)
[57](https://arxiv.org/pdf/2410.09133.pdf)
[58](https://arxiv.org/pdf/2506.06359.pdf)
[59](https://arxiv.org/abs/2503.10198)
[60](https://www.frontiersin.org/articles/10.3389/fvets.2023.1294049/full)
[61](https://www.mdpi.com/2072-4292/15/13/3248)
[62](https://method.meteorf.ru/publ/tr/tr388/html/08.html)
[63](https://arxiv.org/abs/2307.00751)
[64](https://ashpublications.org/blood/article/142/Supplement%201/5892/501032/The-Efficacy-and-Safety-of-the-Third-Generation)
[65](https://www.semanticscholar.org/paper/03432052326814303cab068bd545e2f829fe0c0c)
[66](https://jamanetwork.com/journals/jamapediatrics/fullarticle/2807916)
[67](https://ashpublications.org/blood/article/142/Supplement%201/960/499582/Harnessing-Artificial-Intelligence-for-Risk)
[68](https://www.semanticscholar.org/paper/dfd0f6be43b801fa4205c3b3d3c69098285eb4d4)
[69](https://arxiv.org/abs/2307.12667)
[70](https://arxiv.org/pdf/2503.04118.pdf)
[71](http://arxiv.org/pdf/2503.17658.pdf)
[72](https://arxiv.org/pdf/2310.20218.pdf)
[73](https://www.mdpi.com/1424-8220/23/20/8508/pdf?version=1697524860)
[74](https://arxiv.org/html/2403.10787)
[75](https://aihorizonforecast.substack.com/p/temporal-fusion-transformer-time)
[76](https://www.sciencedirect.com/science/article/pii/S0378778825007297)
[77](https://www.ijisae.org/index.php/IJISAE/article/download/3251/1837/8178)
[78](https://arxiv.org/abs/2405.16877)
[79](https://arxiv.org/html/2501.12215v1)
[80](https://www.sciencedirect.com/science/article/abs/pii/S0022169424006966)
[81](https://arxiv.org/pdf/2509.17845.pdf)
[82](https://pdfs.semanticscholar.org/e53c/c7c0fc7ba570a4190ff7244c485e84b1fdad.pdf)
[83](https://arxiv.org/html/2511.04723v1)
[84](https://arxiv.org/html/2508.12213v1)
[85](https://www.arxiv.org/pdf/2509.10542.pdf)
[86](https://arxiv.org/html/2509.19628v1)
[87](https://pdfs.semanticscholar.org/24e4/ef32428350463d9ac504f083c62fe8e270ae.pdf)
[88](https://pmc.ncbi.nlm.nih.gov/articles/PMC10611135/)
[89](https://arxiv.org/abs/2410.03805)
[90](https://pmc.ncbi.nlm.nih.gov/articles/PMC9453185/)
[91](https://www.mdpi.com/2571-9394/6/4/52)
[92](https://www.semanticscholar.org/paper/0cb94863249f65c45e2f0129aa1bb574eedf1f5e)
[93](https://www.semanticscholar.org/paper/f5cc95fae2ff9ea1f1a2d30be26acccf3e448803)
[94](https://arxiv.org/abs/2410.18712)
[95](https://royalsocietypublishing.org/doi/10.1098/rsos.240248)
[96](https://www.mdpi.com/2227-7390/12/23/3666)
[97](https://arxiv.org/abs/2403.05751)
[98](http://arxiv.org/pdf/2406.02827.pdf)
[99](http://arxiv.org/pdf/2406.02212.pdf)
[100](https://arxiv.org/pdf/2305.00624.pdf)
[101](http://arxiv.org/pdf/2411.04491.pdf)
[102](https://arxiv.org/pdf/2405.05959.pdf)
[103](https://arxiv.org/html/2502.14887)
[104](https://iclr.cc/media/iclr-2024/Slides/17883_mrXtGgm.pdf)
[105](https://towardsdatascience.com/n-beats-the-first-interpretable-deep-learning-model-that-worked-for-time-series-forecasting-06920daadac2/)
[106](https://www.sciencedirect.com/science/article/abs/pii/S0952197625020998)
[107](https://nixtlaverse.nixtla.io/neuralforecast/models.nbeats.html)
[108](https://dl.acm.org/doi/full/10.1145/3711507.3711508)
[109](https://arxiv.org/html/2507.14507)
[110](https://arxiv.org/html/2404.18886v5)
[111](https://arxiv.org/pdf/2508.07490.pdf)
[112](https://arxiv.org/html/2405.13575v3)
[113](https://arxiv.org/html/2310.00655v2)
[114](https://arxiv.org/html/2507.14507v1)
[115](https://openreview.net/forum?id=mmjnr0G8ZY)
[116](https://openreview.net/forum?id=JePfAI8fah)
[117](https://proceedings.neurips.cc/paper_files/paper/2024/file/053ee34c0971568bfa5c773015c10502-Paper-Conference.pdf)
