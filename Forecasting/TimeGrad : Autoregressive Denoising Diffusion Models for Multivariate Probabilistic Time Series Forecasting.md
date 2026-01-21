# TimeGrad : Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting

### 1. 핵심 주장 및 주요 기여 요약

TimeGrad는 **확산 확률 모델(Diffusion Probabilistic Models, DDPM)**을 시계열 예측에 적용한 선구적 연구로, 다음의 핵심 기여를 제시합니다:[1]

**핵심 주장:**
TimeGrad는 에너지 기반 모델(Energy-Based Models, EBM)과 자기회귀 구조를 결합하여 다변량 확률적 시계열 예측 문제를 해결합니다. 각 시간 단계에서 그래디언트 추정을 통해 데이터 분포를 학습하고, Langevin 샘플링을 이용한 Markov 체인으로 백색 노이즈를 원하는 분포의 샘플로 변환합니다.

**주요 기여:**
- **새로운 방법론:** 자기회귀 EBM을 다변량 확률적 예측에 최초로 적용
- **유연한 함수 형태:** 정규화 흐름(normalizing flows)의 제약이 없는 일반적인 고차원 분포 모델링
- **강력한 성능:** 6개 벤치마크 데이터셋에서 새로운 최첨단 성능 달성
- **실용성:** 복잡한 다변량 의존성 포착으로 업무 의사결정 개선

***

### 2. 문제 정의, 제안 방법 및 모델 구조

#### 2.1 해결하고자 하는 문제

다변량 시계열 예측에서 기존 방법들의 한계:[1]

1. **전통적 방법의 확장성 문제:** 고전적 방법은 각 시계열을 개별적으로 학습하여 확장성 부족
2. **분포 모델링의 어려움:** 일반적인 확률 분포를 모델링하려면 정규분포의 제약(대각 공분산 행렬) 또는 저계수 근사 필요
3. **계산 복잡도:** 완전 공분산 행렬 모델링은 매개변수 $$O(D^2)$$, 손실 계산 $$O(D^3)$$ 복잡도 → 실용적 불가능
4. **다변량 의존성 미포착:** 개별 변수들 간의 통계적 의존성 미반영

#### 2.2 제안하는 TimeGrad 모델

**기본 개념:**

TimeGrad는 조건부 확산 모델 기반의 자기회귀 모델입니다. 다변량 시계열의 조건부 분포를 다음과 같이 모델링합니다:[1]

$$q_X\left(x'_{t_0:T} \mid x'_{1:t_0-1}, c_{1:T}\right) = \prod_{t=t_0}^{T} q_X\left(x'_t \mid x'_{1:t-1}, c_{1:T}\right)$$

여기서 각 항은 조건부 제거잡음 확산 모델을 통해 학습됩니다.

**확산 모델의 기초 이론:**[1]

TimeGrad의 핵심은 Ho et al. (2020)의 DDPM을 확장한 것입니다. 

**정방향 프로세스(Forward Process):**

고정된 Markov 체인으로 노이즈를 단계적으로 추가합니다:

$$q(x_n|x_{n-1}) = \mathcal{N}(x_n; \sqrt{1-\beta_n}x_{n-1}, \beta_n I)$$

여기서 $$\beta_1, \ldots, \beta_N$$은 증가하는 분산 스케줄이며, $$0 < \beta_n < 1$$입니다.

**폐쇄형 샘플링:**

임의의 노이즈 수준 $$n$$에서 샘플을 직접 얻을 수 있습니다:

$$q(x_n|x_0) = \mathcal{N}(x_n; \sqrt{\bar{\alpha}_n}x_0, (1-\bar{\alpha}_n)I)$$

여기서 $$\bar{\alpha}\_n = \prod_{i=1}^{n}(1-\beta_i)$$입니다.

**역방향 프로세스(Reverse Process):**

학습된 Gaussian 전이로 노이즈 제거:

$$p(x_{n-1}|x_n) = \mathcal{N}(x_{n-1}; \mu_\theta(x_n, n), \Sigma_\theta(x_n, n))$$

**손실 함수:**

$$\mathbb{E}_{x_0, \epsilon, n} \left\| \epsilon_\theta\left(\sqrt{\alpha_n}x_0 + \sqrt{1-\alpha_n}\epsilon, h_{t-1}, n\right) - \epsilon \right\|^2$$

여기서 $$\epsilon \sim \mathcal{N}(0, I)$$이고, $$\epsilon_\theta$$는 신경망이 예측해야 할 노이즈입니다.[1]

**KL 발산 형태:**

DDPM의 KL 발산 표현:

$$\log p(x_0|x_1) \leq \mathbb{E}_{q(x_0)} \left[ -\log p(x_N) + \sum_{n=2}^{N} D_{KL}(q(x_{n-1}|x_n, x_0) \| p_\theta(x_{n-1}|x_n)) \right]$$

이는 단순화되어:[1]

$$D_{KL}(q(x_{n-1}|x_n, x_0) \| p_\theta(x_{n-1}|x_n)) \propto \mathbb{E}_q\left[ \left\| \mu(x_n, x_0) - \mu_\theta(x_n, n) \right\|^2 \right]$$

#### 2.3 TimeGrad 모델 구조

**시간 동역학 인코딩:**

RNN(LSTM/GRU)을 사용하여 시간 시퀀스를 인코딩합니다:[1]

$$h_t = \text{RNN}(\text{concat}(x'_{t-1}, c_t), h_{t-2})$$

여기서 $$h_t \in \mathbb{R}^{40}$$는 숨겨진 상태입니다.

**TimeGrad 모델:**

$$\prod_{t=t_0}^{T} p_\theta(x'_t|h_{t-1})$$

이는 자기회귀 구조로, 시간 $$t-1$$의 관측값을 입력으로 시간 $$t$$의 분포를 학습합니다.[1]

**노이즈 네트워크 $$\epsilon_\theta$$ 구조:**

WaveNet과 DiffWave를 기반으로 한 조건부 1차원 확장 합성곱(dilated ConvNets)을 사용합니다:[1]

- 8개의 잔여 블록 ($$b = 0, \ldots, 7$$)
- 각 블록의 팽창: $$2^{b \bmod 2}$$
- 각 채널에 8개의 필터
- Gated Activation Unit ($$\tanh$$)
- 8개의 스킵 연결 합산으로 최종 출력 계산

**알고리즘:**

**알고리즘 1: 훈련 (각 시간 단계 $$t \in [t_0, T]$$)**[1]

```
입력: 데이터 x'_t ~ q_X(x'_t), 상태 h_{t-1}
반복:
  n ~ Uniform(1, ..., N)
  ε ~ N(0, I)
  경사 단계: ∇_θ ||ε_θ(√(α_n) x'_t + √(1-α_n) ε, h_{t-1}, n) - ε||²
반복 종료 (수렴)
```

**알고리즘 2: 샘플링 (Annealed Langevin 동역학)**[1]

```
입력: 노이즈 x_N^t ~ N(0, I), 상태 h_{t-1}
for n = N to 1:
  if n = 1:
    z ~ N(0, I)
  else:
    z = 0
  x_{n-1}^t = (1/√(α_n)) x_n^t - ((β_n/√(1-α_n))) ε_θ(x_n^t, h_{t-1}, n) + √(β_n) z
Return x_0^t
```

#### 2.4 주요 설계 요소

**스케일링 정규화:**[1]

실제 데이터에서 변수들의 스케일 차이가 크므로, 각 변수를 컨텍스트 윈도우 평균으로 정규화:

$$x'_{i,t} = \frac{x_{i,t}}{\text{mean}(x_i[\text{context window}]) \text{ or } 1}$$

이는 모델 학습을 단순화하고 성능을 크게 향상시킵니다.

**공변량 처리:**[1]

- 범주형 특성: 임베딩으로 범주 내 관계 포착
- 시간 의존 특성: 요일, 시간 등
- 지연 특성: 데이터셋 주기에 따라 결정

***

### 3. 성능 향상 및 실험 결과

#### 3.1 평가 메트릭

**연속 순위 확률 점수(Continuous Ranked Probability Score, CRPS)**:[1]

$$\text{CRPS}(F, x) = \int_{\mathbb{R}} (F(z) - I(x \leq z))^2 dz$$

여기서 $$F$$는 누적분포함수(CDF)이고, $$I$$는 지시함수입니다.

**경험적 CDF 근사:**

$$F(z) = \frac{1}{S} \sum_{s=1}^{S} I(x'_{0,s} \leq z)$$

**CRPS 합 지표:**

$$\text{CRPS}_{\text{sum}} = \mathbb{E}_t \left[ \text{CRPS}\left(F^{\text{sum}}_t, \sum_{i=1}^D x'_{i,t}\right) \right]$$

이는 모든 변수의 합에 대한 CRPS로, 다변량 의존성을 평가합니다.

#### 3.2 벤치마크 결과

**테스트 집합 $$\text{CRPS}_{\text{sum}}$$ 비교 (낮을수록 우수)**:[1]

| 방법 | Exchange | Solar | Electricity | Traffic | Taxi | Wikipedia |
|------|----------|-------|------------|---------|------|-----------|
| **VES** | 0.005±0.000 | 0.90±0.003 | 0.88±0.0035 | 0.35±0.0023 | - | - |
| **VAR** | 0.005±0.000 | 0.83±0.006 | 0.039±0.0005 | 0.29±0.005 | 0.292±0.000 | 3.4±0.003 |
| **VAR-Lasso** | 0.012±0.0002 | 0.51±0.006 | 0.025±0.0002 | 0.15±0.002 | - | 3.1±0.004 |
| **KVAE** | 0.014±0.002 | 0.34±0.025 | 0.051±0.019 | 0.10±0.005 | - | 0.095±0.012 |
| **Vec-LSTM (ind-scaling)** | 0.008±0.001 | 0.391±0.017 | 0.025±0.001 | 0.087±0.041 | 0.506±0.005 | 0.133±0.002 |
| **Vec-LSTM (lowrank-Copula)** | 0.007±0.000 | 0.319±0.011 | 0.064±0.008 | 0.103±0.006 | 0.326±0.007 | 0.241±0.033 |
| **GP (Copula)** | 0.007±0.000 | 0.337±0.024 | 0.0245±0.002 | 0.078±0.002 | 0.208±0.183 | 0.086±0.004 |
| **Transformer-MAF** | 0.005±0.003 | 0.301±0.014 | 0.0207±0.000 | 0.056±0.001 | 0.179±0.002 | 0.063±0.003 |
| **TimeGrad** | **0.006±0.001** | **0.287±0.02** | **0.0206±0.001** | **0.044±0.006** | **0.114±0.02** | **0.0485±0.002** |

TimeGrad는 Wikipedia를 제외한 모든 데이터셋에서 최고 성능을 달성합니다.[1]

#### 3.3 확산 단계 수 ablation 연구

**노이즈 단계 N의 영향 (Electricity 데이터셋)**:[1]

- $$N = 2$$: 약간의 성능 저하
- $$N = 10$$: 의미 있는 성능 손실 없음
- $$N = 100$$: **최적값**
- $$N > 100$$: 추가 이점 없음

선택된 설정: $$N = 100$$ (선형 분산 스케줄, $$\beta_1 = 10^{-4}$$에서 $$\beta_N = 0.1$$로)

#### 3.4 비교 방법들

1. **고전 방법:**
   - VAR (Vector AutoRegressive): 주기성에 따른 지연 사용
   - VAR-Lasso: Lasso 정규화
   - GARCH: 다변량 조건부 이분산성 모델
   - VES: 혁신 상태 공간 모델

2. **심층학습 방법:**
   - **KVAE**: 선형 상태 공간 모델 위의 VAE
   - **Vec-LSTM**: RNN 기반 독립 Gaussian 또는 저계수 Gaussian Copula
   - **GP 모델**: LSTM 언롤링 후 저계수 Gaussian Copula
   - **Transformer-MAF**: Transformer + 마스크 자기회귀 흐름[1]

**TimeGrad의 장점 대비 경쟁 모델:**

정규화 흐름(Flow) 기반 모델의 문제:[1]
- 연속 연결된 분포에 연속 변환 적용 필요
- 분리된 모드 모델링 어려움
- 모드 사이에 가짜 밀도 할당 → 부정확성

VAE 기반 모델의 문제:[1]
- 연속 공간에서 분리된 공간으로의 매핑 학습 어려움

**EBM의 장점:**
- 함수 형태에 대한 제약 없음
- 분리된 모드 처리 능력
- 비제약 최적화 가능

***

### 4. 모델의 일반화 성능 향상 가능성

#### 4.1 일반화 성능의 주요 요인

**1. 확산 모델의 본질적 우수성:**[1]

TimeGrad가 높은 일반화 성능을 달성하는 이유:
- **다중 스케일 특성 학습:** 노이즈 스케일이 증가할수록 점진적 특성 학습으로 조악한 및 세밀한 특성 모두 포착
- **가능성 최대화:** 변분 하한을 최적화하여 데이터 분포의 포괄적 표현 학습
- **자기회귀 구조:** 시간 의존성 자동 학습으로 장기 의존성 포착

**2. EBM의 유연성:**[1]

- 정규분포의 제약 없음
- 복잡한 다변량 의존성 직접 모델링
- 에너지 기반 접근으로 비제약 최적화

#### 4.2 일반화 성능 향상을 위한 설계 요소

**스케일링 및 정규화:**

컨텍스트 윈도우 평균으로 정규화하여:[1]
- 변수 간 스케일 차이 극복
- 모델이 상대적 패턴에 집중하도록 유도
- 서로 다른 도메인에서의 적응성 향상

**노이즈 스케줄 설계:**

선형 분산 스케줄 채택:[1]
$$\beta_n: 10^{-4} \to 0.1$$

이는:
- 초기 단계에서는 미세한 변동만 추가
- 후기 단계에서는 점진적으로 더 많은 노이즈 추가
- 역방향 프로세스가 근사 Gaussian 유지 가능

#### 4.3 다변량 의존성 포착 능력

**CRPS_sum 지표의 의미:**[1]

모든 변수의 합에 대한 CRPS 평가는:
- 변수 간 공동 분포 예측 능력 반영
- 단순 주변 분포보다 어려운 과제
- TimeGrad의 우수성 더욱 강조

**예시 (Traffic 데이터):**

963차원의 Traffic 데이터에서 TimeGrad는:[1]
- 이웃 변수들 간의 1-2 자릿수 크기 차이 처리
- 공간적 의존성 학습
- 강력한 일반화 달성

#### 4.4 한계 및 제약 사항

**현재 TimeGrad의 제약:**[1]

1. **샘플링 시간:**
   - 훈련 중에는 루프 필요 없음
   - 샘플링 시 N번 반복 필요 ($$N=100$$) → 샘플링 시간 증가
   - Chen et al. (2021)의 개선된 분산 스케줄로 단계 감소 가능

2. **이산 데이터 처리:**
   - 현재는 실수 데이터에 최적화
   - 정규화 흐름은 데이터에 균일 노이즈 추가 필요
   - EBM은 이산 분포를 명시적으로 모델링 가능하지만 미구현

3. **계산 복잡도:**
   - RNN 기반으로 장시간 시퀀스 처리 제한 가능
   - Transformer로의 교체 제안 (Rasul et al., 2021)

4. **외분포(Out-of-Distribution) 감지:**
   - 향후 연구 방향으로 제시
   - EBM의 이론적 우수성 (OOD 감지)이 실현되지 않음

***

### 5. 2020년 이후 관련 최신 연구 비교 분석

#### 5.1 시간순 진화

**제1세대: 확산 모델의 초도 적용 (2020-2021)**

1. **DDPM (Ho et al., 2020)**[2]
   - 이미지 생성에서의 성공
   - 가우시안 확산 프로세스 확립
   - TimeGrad의 이론적 기초

2. **TimeGrad (Rasul et al., 2021)**[1]
   - 다변량 확률적 시계열 예측 최초 적용
   - 자기회귀 EBM 결합
   - 6개 벤치마크에서 SOTA 달성

3. **CSDI (Tashiro et al., 2021)**[3]
   - 시계열 대체, 추정(imputation) 문제로 확장
   - 조건부 점수 기반 확산 모델
   - 시간 및 특성 의존성 명시적 학습
   - MAE 5-20% 개선

**제2세대: 조건화 메커니즘 개선 (2022-2023)**

4. **ScoreGrad (Yan et al., 2021)**[4]
   - TimeGrad의 직접 경쟁자
   - 연속 에너지 기반 모델 (SDE 기반)
   - 점수 기반 매칭 활용
   - DDPM의 노이즈 스케줄 민감도 개선

5. **TimeDiff (Shen & Kwok, 2023)**[5]
   - **혁신:** 노이즈 예측이 아닌 **데이터 직접 예측**
   - 이미지 생성의 관례를 시계열에서 깸
   - Future Mixup과 자기회귀 초기화 기법
   - 성능 향상

6. **SSSD_S4 (Alcaraz & Strodthoff, 2023)**[5]
   - Structured State Space Models (S4) 활용
   - 장시간 의존성 효율적 처리
   - RNN의 한계 극복

7. **TDSTF (sparse 시계열)**[5]
   - 희소 시계열 데이터 특화
   - 불규칙한 샘플링 처리

**제3세대: 아키텍처 통합 및 기초 모델화 (2023-2025)**

8. **CSDI 확장 - tBN-CSDI (Bishop et al., 2025)**[6]
   - 백색 노이즈 대신 **시변 청색 노이즈** 도입
   - 고주파 변동 및 시간 자기상관 동시 포착
   - 희소성 조건에서 30% 오류 감소

9. **S²DBM (Series-to-Series Diffusion Bridge Model, 2024)**[7]
   - Brownian Bridge 프로세스 활용
   - 역추정에서 랜덤성 감소
   - 정보적 사전확률 통합
   - 불안정성 감소 및 정확도 향상

10. **mr-Diff (Multi-Resolution Diffusion, 2024)**[8]
    - **계층적 확산:** 계절-추세 분해와 확산 인터리빙
    - 추세 성분 순차적 추출
    - 여러 시간 해상도에서 점진적 노이즈 제거
    - 더 신뢰할 수 있는 예측

11. **CCDM (Channel-wise 조건화, 2024)**[9]
    - 채널 간 및 채널 내 상관성 분리 처리
    - Vision Transformer (DiT) 아키텍처
    - 대규모 다변량 데이터 처리

12. **TimeDiT (Foundation Model, 2025)**[10]
    - **기초 모델 패러다임:** 도메인 횡단 사전학습
    - Transformer 기반 확산 모델
    - 영점 학습으로 다른 데이터셋에 적응
    - Solar, Electricity, Taxi에서 SOTA
    - TimeLLM/Timer보다 적은 파라미터로 우수

13. **SimDiff (2025)**[11]
    - **단순화 원칙:** 외부 사전학습 모델 제거
    - Transformer 기반 하나의 통합 프레임워크
    - **데이터 vs 노이즈 예측 재평가:** 시계열 특성상 데이터 예측 우수
    - Skip 연결 제거 (노이즈 증폭 방지)
    - Channel 독립성 처리로 효율성 증대
    - 평균 8.3% MSE 개선 (다른 확산 모델 대비)
    - 추론 속도 대폭 향상

14. **Dynamical Diffusion (2025)**[12]
    - **시간 동역학 명시적 모델링**
    - 각 확산 단계에서 시간적 전이 학습
    - 이전 상태에 대한 의존성 구조화
    - 과학적 시공간 예측, 비디오 생성에 우수

**제4세대: 교차모달 및 거리학습 (2024-2025)**

15. **LDM4TS (Latent Diffusion for Time Series, 2025)**[13]
    - 시각 정보 활용: 시계열을 다중뷰 시각 표현으로 변환
    - 사전학습된 비전 인코더 활용
    - 이미지 재구성 능력의 시계열 활용

16. **RATD (Retrieval-Augmented Time series Diffusion, 2024)**[14]
    - 검색 강화 메커니즘
    - 비주기적 또는 복잡한 시계열에 우수
    - 외적 맥락 정보 활용

17. **StochDiff (Stochastic Latent Spaces, 2024)**[15]
    - **확률적 잠재 공간:** 각 시간 단계에서 데이터 기반 사전 학습
    - 높은 확률성 시계열 모델링
    - 시간 동역학과 고유 불확실성 포착

#### 5.2 기술적 진화 비교

| 측면 | TimeGrad (2021) | 최신 모델들 (2024-2025) | 개선 사항 |
|------|-------------------|----------------------|----------|
| **조건화** | RNN + EBM | Transformer 기반 | 장기 의존성 향상 |
| **예측 대상** | 노이즈 $$\epsilon$$ | 데이터 또는 혼합 | 우수한 수렴성 |
| **구조** | 자기회귀 | 자기회귀 + 비자기회귀 혼합 | 유연성 증가 |
| **시간 해상도** | 단일 | 다중 해상도 | 계층적 특징 |
| **노이즈 프로세스** | 백색 가우시안 | 구조화된 노이즈 | 시간 상관성 보존 |
| **기초 모델화** | 작업별 | 사전학습 후 적응 | 일반화 강화 |
| **샘플링 효율** | N=100 필요 | 빠른 샘플링 기법 | 추론 속도 개선 |

#### 5.3 성능 벤치마크 비교

**주요 데이터셋에서의 경쟁 (CRPS 또는 MSE 지표)**

- **TimeGrad (2021):** Traffic 데이터 CRPS_sum = 0.044
- **mr-Diff (2024):** Traffic에서 SOTA 달성
- **CCDM (2024):** 모든 데이터셋에서 최고 또는 2위
- **SimDiff (2025):** 평균 8.3% MSE 개선
- **TimeDiT (2025):** 영점 학습에서 특화된 모델 능가

#### 5.4 TimeGrad의 위치

**TimeGrad의 역사적 의의:**
- 다변량 확률적 시계열 예측에 확산 모델 최초 적용
- EBM과 자기회귀의 성공적 결합
- 이후 모든 시계열 확산 모델의 기초

**현재 한계:**
- 기초 모델화 미포함
- 단일 RNN 기반 조건화
- 노이즈 기반 예측 (개선된 모델들은 데이터 예측으로 전환)
- 샘플링 효율성 제한

**강점 유지:**
- 명확한 이론적 기초
- 강력한 확률적 예측 능력
- 다변량 의존성 학습 증명

***

### 6. 논문의 영향 및 향후 연구 고려사항

#### 6.1 현재까지의 영향

**학술적 영향:**
- 시계열 확산 모델 연구의 개척
- 2021년 이래 620회 이상 인용[16]
- ICML 2021 채택 및 발표

**실무 활용:**
- 불확실성 정량화 필요 도메인에서 채택 증가
- 다변량 이상 탐지로의 확장
- 에너지, 금융, 의료 시계열 예측에 응용

**후속 연구 촉발:**
- 확산 기반 시계열 학습의 폭발적 성장
- 2023년 이후 연 100+ 논문 게시 ()

#### 6.2 향후 연구 고려사항

**1. 모델 일반화 성능 향상**

TimeGrad에서 직접 제시한 미래 방향:[1]

a) **이산 데이터 처리:**
   - 정규화 흐름은 역양자화(dequantization) 필요
   - EBM은 이산 분포 명시적 모델링 가능
   - **과제:** 이산 변수를 가진 현실 데이터 처리

b) **이상 탐지 및 분포 외(OOD) 감지:**
   - EBM의 이론적 우수성: 데이터 다양체에서 높은 우도, 다른 영역에서 낮은 우도[1]
   - Nalisnick et al. (2019)의 발견: 정규분포/흐름 모델이 OOD에 높은 우도 할당 → TimeGrad는 이 문제 없음
   - **실무:** 이상 탐지 작업에 TimeGrad 활용

c) **장시간 시퀀스 처리:**
   - RNN 기반 조건화의 한계
   - **제안:** Transformer 아키텍처로 교체 (Rasul et al., 2021)
   - **효과:** 더 나은 조건화 및 장기 의존성 학습

d) **그래프 신경망 활용:**
   - 개체 간 관계가 알려진 경우 (예: 센서 네트워크)
   - **제안:** GNN을 noise network $$\epsilon_\theta$$에 통합[1]
   - **이점:** 구조적 귀납 편향 직접 반영

**2. 샘플링 효율성 개선**

TimeGrad의 현저한 한계:[1]

- **현재:** 각 샘플마다 N번 신경망 호출 ($$N=100$$)
- **개선 방안:**
  
  a) **분산 스케줄 최적화 (Chen et al., 2021)**
     - 개선된 분산 스케줄 + L1 손실
     - 샘플링 단계 감소 ($$N' < N$$)
     - 품질 약간 감소로 속도 향상
  
  b) **비-Markovian 프로세스 (Song et al., 2021)**
     - 확산 프로세스 일반화
     - 더 빠른 샘플링 가능
     - 이론적 타당성 유지

  c) **한 단계 모델 (One-step generation)**
     - 최근 추세: 단일 신경망 호출로 샘플 생성
     - **과제:** 시계열에 적응 (이미지보다 어려움)

**3. 아키텍처 개선**

TimeGrad 이후의 발전 교훈:[11][5]

a) **Transformer 기반 노이즈 네트워크**
   - TimeGrad의 CNN 기반 (WaveNet 유형)에서 Transformer로
   - **이점:** 장기 의존성, 병렬화 처리

b) **채널 간 및 채널 내 주의(Attention)**
   - 변수 간 공동 의존성과 변수 내 시간 의존성 분리
   - **효과:** CCDM의 성공사례

c) **데이터 vs 노이즈 예측 재평가**
   - TimeGrad: 노이즈 예측
   - TimeDiff (2023): 데이터 직접 예측가 우수
   - **이유:** 시계열의 범위 제약과 유리한 수치 특성
   - **시사:** 이미지의 관례를 맹목적으로 따르지 말 것

d) **다중 해상도 처리**
   - mr-Diff의 성공: 계절-추세 분해 + 계층적 확산
   - **원리:** 다양한 시간 스케일의 패턴을 계층적으로 학습

**4. 확장성 및 기초 모델화**

최근 트렌드 (2024-2025):[10][11]

a) **사전학습 및 적응**
   - TimeDiT: 대규모 데이터로 사전학습 후 영점 적응
   - **이점:** 도메인 간 전이 가능성
   - **과제:** TimeGrad 수준에서는 미지원

b) **파라미터 효율성**
   - 모델 크기와 성능의 관계 재검토
   - SimDiff: 작은 Transformer로 SOTA 달성
   - **시사:** 필요한 것은 아키텍처 설계, 크기 아님

c) **영점 학습(Zero-shot Learning)**
   - 훈련 없이 새 데이터셋에 적응
   - GPD (Generative Pre-trained Diffusion) 패러다임[17]
   - **과제:** TimeGrad는 작업 특정(task-specific) 훈련 필요

**5. 이론 및 기초**

TimeGrad의 수학적 확장:[18][19]

a) **점수 함수(Score Function)와 SDE**
   - Song & Kingma (2020): 점수 기반 생성 모델의 SDE 해석
   - TimeGrad의 DDPM → SDE 관점 재해석 가능
   - **이점:** 더 일반적 이론 체계

b) **변분 하한의 재검토**
   - TimeGrad의 손실: KL 발산 형태의 변분 하한
   - **향상:** 더 타이트한 하한 개발로 성능 개선 가능

c) **샘플링 안정성 이론**
   - Langevin 동역학의 수렴성 증명
   - TimeGrad는 실증적 성공이지만 수학적 보장 제한
   - **연구:** 고차원 시계열에서의 수렴 조건 도출

**6. 신규 응용 영역**

TimeGrad 아이디어의 확장:

a) **비정상(Non-stationary) 시계열**
   - Non-stationary Transformer의 아이디어와 결합
   - 점진적 분포 변화 모델링

b) **희소 및 불규칙 샘플링**
   - TDSTF의 성공: 시간 불규칙성 처리
   - **응용:** 의료 모니터링, 센서 데이터

c) **다중 스케일 예측**
   - 동일 모델로 다양한 예측 지평 처리
   - MTSF (Multi-Temporal Scale Forecasting)

d) **설명 가능성(Interpretability)**
   - 확산 프로세스의 중간 단계에서 특성 추출
   - 예측 불확실성의 근원 분석

#### 6.3 TimeGrad의 자리매김

**TimeGrad의 지속적 가치:**

1. **이론적 명확성:** 변분 하한과 노이즈 예측의 명확한 연결
2. **확장성 기초:** 이후 모든 시계열 확산 모델의 출발점
3. **다변량 의존성:** CRPS_sum으로 평가되는 공동 분포 예측 능력
4. **실용성:** 불확실성 정량화 필요 응용에 직접 적용 가능

**한계와 초월:**

1. **구식화된 선택:**
   - CNN 기반 노이즈 네트워크 → Transformer 우수
   - 노이즈 예측 → 데이터 예측 우수
   - RNN 조건화 → Transformer 조건화 우수

2. **미해결 문제:**
   - 샘플링 효율성
   - 기초 모델화 가능성
   - OOD 감지 실제 구현

3. **이론적 공백:**
   - 고차원 시계열에서의 수렴성 증명 부재
   - 다변량 의존성 학습의 샘플 복잡도 미분석

***

## 결론

TimeGrad는 **다변량 확률적 시계열 예측에 확산 확률 모델을 최초로 성공적으로 적용한 선구적 연구**입니다. 에너지 기반 모델의 유연성과 자기회귀 구조의 강점을 결합하여, 정규분포의 제약 없이 복잡한 다변량 의존성을 학습할 수 있음을 보였습니다.

**주요 기여:**
- 새로운 방법론: 자기회귀 EBM의 첫 적용
- 강력한 성능: 6개 벤치마크 중 5개에서 SOTA
- 이론적 기초: 변분 하한과 점수 기반 생성의 연결

**한계:**
- 샘플링 효율성(N=100 단계)
- 기초 모델화 미포함
- 장시간 시퀀스에서의 RNN 조건화 제약

**향후 방향:**
최근 2024-2025년의 발전(TimeDiT, SimDiff, Dynamical Diffusion 등)은 TimeGrad의 아이디어를 Transformer 기반 아키텍처, 다중 해상도 처리, 데이터 직접 예측, 기초 모델화로 진화시켰습니다. 이들은 샘플링 효율성을 개선하고 일반화 성능을 향상시켰으나, TimeGrad의 명확한 이론적 기초와 다변량 의존성 학습의 본질적 강점은 유지됩니다.

**연구자를 위한 시사:**
1. 이미지/텍스트의 관례를 시계열에 직접 적용하지 말 것
2. 데이터 도메인의 고유 특성(시간 의존성, 범위 제약)을 아키텍처에 반영할 것
3. 기초 모델화와 적응적 학습은 향후 주요 방향
4. 샘플링 효율성과 이론적 보장의 동시 달성이 과제

***

## 참고 자료

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d648e9de-d639-4e7f-9c8d-b1caa4ccb1a2/rasul21a.pdf)
[2](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf)
[3](https://letter-night.tistory.com/238)
[4](https://arxiv.org/pdf/2106.10121.pdf)
[5](https://www.alphaxiv.org/de/overview/2401.03006v2)
[6](https://academic.oup.com/bioinformaticsadvances/article/5/1/vbaf225/8261367)
[7](http://arxiv.org/pdf/2411.04491.pdf)
[8](https://iclr.cc/media/iclr-2024/Slides/17883_mrXtGgm.pdf)
[9](https://www.emergentmind.com/topics/diffusion-models-in-time-series-forecasting)
[10](https://kdd2025.kdd.org/wp-content/uploads/2025/07/paper_4.pdf)
[11](https://arxiv.org/html/2511.19256v1)
[12](https://arxiv.org/abs/2503.00951)
[13](https://arxiv.org/html/2502.14887)
[14](https://arxiv.org/html/2410.18712v1)
[15](http://arxiv.org/pdf/2406.02827.pdf)
[16](https://arxiv.org/pdf/2101.12072.pdf)
[17](http://arxiv.org/pdf/2406.02212.pdf)
[18](https://openreview.net/pdf/a768d50d1862f891a24a1a17952e029995647ea8.pdf)
[19](https://randomsampling.tistory.com/244)
[20](https://arxiv.org/pdf/2305.00624.pdf)
[21](https://arxiv.org/abs/2011.13456)
[22](https://arxiv.org/html/2507.14507v2)
[23](https://www.jcdr.net/article_fulltext.asp?issn=0973-709x&year=2024&month=October&volume=18&issue=10&page=LC01-LC05&id=20182)
[24](https://www.science-gate.com/IJAAS/2024/V11I8/1021833ijaas202408013.html)
[25](https://ejournal.unisbablitar.ac.id/index.php/antivirus/article/view/3468)
[26](https://www.ewadirect.com/proceedings/aemps/article/view/12649)
[27](https://journalarja.com/index.php/ARJA/article/view/494)
[28](https://www.dovepress.com/comparison-of-arima-and-bayesian-structural-time-series-models-for-pre-peer-reviewed-fulltext-article-IDR)
[29](https://jurnal.uinsyahada.ac.id/index.php/LGR/article/view/8463)
[30](https://www.frontiersin.org/articles/10.3389/fvets.2023.1294049/full)
[31](https://ojs3.unpatti.ac.id/index.php/barekeng/article/view/9090)
[32](https://ieeexplore.ieee.org/document/10394693/)
[33](https://arxiv.org/pdf/2307.11494.pdf)
[34](https://arxiv.org/pdf/2412.09328.pdf)
[35](https://learnopencv.com/denoising-diffusion-probabilistic-models/)
[36](https://www.sciencedirect.com/science/article/abs/pii/S0952197625020998)
[37](https://arxiv.org/abs/2006.11239)
[38](https://openreview.net/forum?id=5Ro7JT5Vaf)
[39](https://aclanthology.org/2025.findings-emnlp.58.pdf)
[40](https://arxiv.org/html/2406.02827v3)
[41](https://arxiv.org/abs/2404.02552)
[42](https://arxiv.org/abs/2507.19003)
[43](https://www.arxiv.org/pdf/2411.04491.pdf)
[44](https://arxiv.org/abs/2402.04384)
[45](https://arxiv.org/abs/2301.08518)
[46](https://arxiv.org/pdf/2401.03006.pdf)
[47](https://milvus.io/ai-quick-reference/what-is-denoising-diffusion-probabilistic-modeling-ddpm)
[48](https://yang-song.net/blog/2021/score/)
[49](https://proceedings.neurips.cc/paper_files/paper/2024/file/053ee34c0971568bfa5c773015c10502-Paper-Conference.pdf)
[50](https://www.youtube.com/watch?v=H45lF4sUgiE)
[51](https://arxiv.org/html/2507.14507)
[52](https://www.scitepress.org/DigitalLibrary/Link.aspx?doi=10.5220/0012428900003654)
[53](https://arxiv.org/abs/2402.12694)
[54](https://link.springer.com/10.1007/s10489-025-06444-y)
[55](https://arxiv.org/abs/2401.06175)
[56](https://arxiv.org/abs/2402.16230)
[57](https://www.tandfonline.com/doi/full/10.1080/03081079.2024.2350542)
[58](https://ieeexplore.ieee.org/document/10615719/)
[59](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13272/3048103/Multivariate-time-series-anomaly-detection-based-on-sparse-autoencoder-and/10.1117/12.3048103.full)
[60](https://jeasiq.uobaghdad.edu.iq/index.php/JEASIQ/article/view/3587)
[61](https://ieeexplore.ieee.org/document/9857639/)
[62](https://arxiv.org/html/2310.06119v2)
[63](https://arxiv.org/pdf/2110.00578.pdf)
[64](https://arxiv.org/pdf/2007.13156.pdf)
[65](http://arxiv.org/pdf/2405.19661.pdf)
[66](https://royalsocietypublishing.org/doi/pdf/10.1098/rsif.2013.0048)
[67](https://arxiv.org/pdf/2109.12218.pdf)
[68](https://arxiv.org/pdf/2310.11022.pdf)
[69](http://arxiv.org/pdf/2312.04142.pdf)
[70](https://openreview.net/pdf?id=VzuIzbRDrum)
[71](https://icml.cc/virtual/2021/poster/8591)
[72](https://axyon.ai/hubfs/Projects%20and%20Thesis/2021%20-%20Edoardo%20Berti%20-%20Multivariate%20Autoregressive%20Denoising%20Diffusion%20Model%20for%20VaR%20Evaluation.pdf?hsLang=en)
[73](https://modulai.io/blog/diffusion-models-for-time-series-forecasting/)
[74](http://arxiv.org/pdf/2107.03502v2.pdf)
[75](https://www.scribd.com/document/775349927/TimeGrad)
[76](https://arxiv.org/html/2411.05793v1)
[77](https://pdfs.semanticscholar.org/f041/ae978cc131e0ac082be4bc25624fbd2be63c.pdf)
[78](https://www.semanticscholar.org/paper/CSDI:-Conditional-Score-based-Diffusion-Models-for-Tashiro-Song/8982bb695dcebdacbfd079c62cd7acca8a8b48dc)
[79](https://arxiv.org/html/2409.02322v1)
[80](https://arxiv.org/abs/2107.03502)
[81](https://papers.neurips.cc/paper_files/paper/2021/file/cfe8504bda37b575c70ee1a8276f3486-Paper.pdf)
[82](https://huggingface.co/blog/autoformer)
