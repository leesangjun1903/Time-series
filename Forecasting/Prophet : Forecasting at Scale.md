
# Forecasting at Scale

## 1. 논문의 핵심 주장과 기여

"Forecasting at Scale"(Taylor & Letham, 2017)의 기본 명제는 단순하지만 강력하다: **시계열 예측의 병목은 컴퓨팅이 아니라 인적 자원과 모델의 해석 가능성**이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2a136195-6899-4065-9889-b7c5378995e7/peerj-preprints-3190.pdf)

### 1.1 문제의 본질

조직이 직면한 현실:
- 시계열 예측 전문가는 극도로 희귀 (특화 교육 필요)
- 필요한 예측은 수백 개 이상 (용량 계획, 목표 설정, 이상 탐지)
- 기존 자동화 방법은 비유연적 (매개변수 튜닝 어려움)

Prophet의 해결책: **Analyst-in-the-Loop** 시스템으로 비전문가도 도메인 지식으로 조정 가능하게 설계 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2a136195-6899-4065-9889-b7c5378995e7/peerj-preprints-3190.pdf)

### 1.2 핵심 기여

| 기여 영역 | 내용 |
|----------|------|
| **모델 설계** | 분해 가능한 시계열 모형: $y(t) = g(t) + s(t) + h(t) + \epsilon_t$ [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2a136195-6899-4065-9889-b7c5378995e7/peerj-preprints-3190.pdf) |
| **추세 모델링** | 포화 성장 & 선형 변점 모형, 자동 변점 감지 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2a136195-6899-4065-9889-b7c5378995e7/peerj-preprints-3190.pdf) |
| **계절성** | Fourier 급수로 다중 주기성 동시 처리 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2a136195-6899-4065-9889-b7c5378995e7/peerj-preprints-3190.pdf) |
| **휴일/이벤트** | 비정기적 이벤트 직관적 모델링 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2a136195-6899-4065-9889-b7c5378995e7/peerj-preprints-3190.pdf) |
| **평가 체계** | Simulated Historical Forecasts (SHF)로 대규모 예측 관리 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2a136195-6899-4065-9889-b7c5378995e7/peerj-preprints-3190.pdf) |
| **해석성** | 각 성분 분리 시각화로 블랙박스 회피 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2a136195-6899-4065-9889-b7c5378995e7/peerj-preprints-3190.pdf) |

***

## 2. 상세 방법론: Prophet 모델 구조

### 2.1 분해 모형의 수학적 기초

Prophet은 **일반화 가산 모형(GAM)** 원칙에 따라 설계되어 있다.

기본 식:
$$y(t) = g(t) + s(t) + h(t) + \epsilon_t \quad (1)$$ [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2a136195-6899-4065-9889-b7c5378995e7/peerj-preprints-3190.pdf)

각 성분의 역할:
- **$g(t)$ (추세)**: 비주기적 변화, 장기 방향성
- **$s(t)$ (주기성)**: 반복적 패턴 (주간, 계절)
- **$h(t)$ (휴일/이벤트)**: 이례적 영향
- **$\epsilon_t$ (오차)**: 정규분포 가정, $\epsilon_t \sim N(0, \sigma^2)$

### 2.2 추세 모델: 두 가지 기본 형태

#### A. 포화 성장 모델 (S자형 성장)

$$g(t) = \frac{C(t)}{1 + \exp(-(k + a(t)^T \delta)(t - (m + a(t)^T \gamma)))} \quad (3)$$ [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2a136195-6899-4065-9889-b7c5378995e7/peerj-preprints-3190.pdf)

**핵심 매개변수**:
- $C(t)$: 시장 수용 능력 (carrying capacity), 시간에 따라 변함
- $k$: 기본 성장률
- $\delta_j$: $j$번째 변점에서의 성장률 조정
- $\gamma_j$: 함수 연속성을 위한 보정값

**변점 연속성 조정식**:
$$\gamma_j = \left(s_j - m - \sum_{l<j}\gamma_l\right)\left(1 - \frac{k + \sum_{l<j}\delta_l}{k + \sum_{l\leq j}\delta_l}\right)$$ [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2a136195-6899-4065-9889-b7c5378995e7/peerj-preprints-3190.pdf)

이 식은 구간 경계에서 함수 및 도함수 연속성을 보장한다.

#### B. 선형 추세 모형 (구간별 선형)

$$g(t) = (k + a(t)^T \delta)t + (m + a(t)^T \gamma) \quad (4)$$ [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2a136195-6899-4065-9889-b7c5378995e7/peerj-preprints-3190.pdf)

**변점 자동 선택 메커니즘**:
라플라스 사전분포 사용: $\delta_j \sim \text{Laplace}(0, \tau)$ [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2a136195-6899-4065-9889-b7c5378995e7/peerj-preprints-3190.pdf)

- $\tau$ 역할: 정규화 강도 제어
  - $\tau$ 작음 → 부드러운 추세 (적은 변점)
  - $\tau$ 큼 → 유연한 추세 (많은 변점, 과적합 위험)

**예측 불확실성 정량화**:
$$\lambda = \frac{1}{S}\sum_{j=1}^{S} |\delta_j|$$ [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2a136195-6899-4065-9889-b7c5378995e7/peerj-preprints-3190.pdf)

미래 변점을 과거의 빈도와 크기로 시뮬레이션하여 신뢰도 구간 생성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2a136195-6899-4065-9889-b7c5378995e7/peerj-preprints-3190.pdf)

### 2.3 계절성 모델: Fourier 급수

$$s(t) = \sum_{n=1}^{N} \left(a_n \cos\left(\frac{2\pi n t}{P}\right) + b_n \sin\left(\frac{2\pi n t}{P}\right)\right) \quad (5-6)$$ [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2a136195-6899-4065-9889-b7c5378995e7/peerj-preprints-3190.pdf)

**실무적 설정**:

| 기간 | P 값 | N 값 | 해석 |
|------|------|------|------|
| 연간 | 365.25 | 10 | ~183일 최단 주파수 |
| 주간 | 7 | 3 | ~3.5일 최단 주파수 |

**정규화**:
$$\beta \sim \text{Normal}(0, \sigma^2)$$ [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2a136195-6899-4065-9889-b7c5378995e7/peerj-preprints-3190.pdf)
- $\sigma$ 작음 → 평활한 계절성
- $\sigma$ 큼 → 가변적 계절성

### 2.4 휴일/이벤트 모델

$$h(t) = Z(t)\kappa$$ [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2a136195-6899-4065-9889-b7c5378995e7/peerj-preprints-3190.pdf)

여기서:
- $Z(t)$: 휴일 지시자 행렬
- $\kappa \sim \text{Normal}(0, \nu^2)$: 휴일별 효과 크기

**실무 기능**:
- 국가별 휴일 구분 가능
- 휴일 전후 창(window) 지원 (예: 추수감사절 주말)
- 각 휴일의 영향을 별도 매개변수로 처리 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2a136195-6899-4065-9889-b7c5378995e7/peerj-preprints-3190.pdf)

***

## 3. 모델 성능: 실증 분석과 한계

### 3.1 성능 우월성 (Facebook Events 데이터)

기존 방법들과 Prophet의 비교: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2a136195-6899-4065-9889-b7c5378995e7/peerj-preprints-3190.pdf)

| 방법 | MAPE | 문제점 |
|------|------|--------|
| auto.arima | 40-50% | 추세 변화 미포착, 계절성 오류 |
| ets | 35-40% | 주간 계절성만 포착 |
| snaive | 45-55% | 모든 추세/휴일 미처리 |
| tbats | 30-40% | 연간 계절성 과반응 |
| **Prophet** | **7-15%** | ✓ 다중 계절성 ✓ 변점 감지 |

### 3.2 설계상 강점

1. **모듈 구조**: 새 성분 추가 용이 (GAM 기반)
2. **비정규 데이터 지원**: 불규칙 시간 간격, 누락값 자동 처리 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2a136195-6899-4065-9889-b7c5378995e7/peerj-preprints-3190.pdf)
3. **빠른 피팅**: L-BFGS 최적화로 매개변수 조정 후 즉시 결과 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2a136195-6899-4065-9889-b7c5378995e7/peerj-preprints-3190.pdf)
4. **해석성**: 각 성분 분리 가능, 시각화 용이 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2a136195-6899-4065-9889-b7c5378995e7/peerj-preprints-3190.pdf)

### 3.3 고유한 한계

#### A. 시간 의존성 부재
- **문제**: AR(자기회귀) 없음 → 단기 동적 미포착
- **결과**: 충격 지속성, 근처 의존성 미모델링
- **해결책**: NeuralProphet의 AR-Net 추가 [arxiv](https://arxiv.org/pdf/2111.15397.pdf)

#### B. 단변량만 지원
- 다변량 자동 확장 곤란
- 변수 간 동시 이동성(co-movement) 미처리

#### C. 불확실성 정량화의 가정
"미래 변점 = 과거의 평균 빈도/크기" 가정이 강함 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2a136195-6899-4065-9889-b7c5378995e7/peerj-preprints-3190.pdf)
→ 구조적 위기 상황에서 신뢰도 구간 정확도 떨어짐

#### D. 데이터 요구사항
- 최소 2-3 주기의 역사 필요 (월간이면 24-36개월)
- 이상치 사전 제거 필수
- 짧은 시계열에서 성능 저하 [arxiv](https://arxiv.org/pdf/2111.15397.pdf)

***

## 4. 대규모 예측 관리 시스템

### 4.1 Simulated Historical Forecasts (SHF)

**핵심 아이디어**: 과거 여러 시점에서 예측을 '역사적으로 재현'하여 성능 평가 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2a136195-6899-4065-9889-b7c5378995e7/peerj-preprints-3190.pdf)

**기대 오차 모델**:
$$\xi(h) = E[\phi(T, h)]$$ [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2a136195-6899-4065-9889-b7c5378995e7/peerj-preprints-3190.pdf)

여기서 $\phi(T, h) = d(\hat{y}(T+h|T), y(T+h))$는 예측 오류

**실무 구현**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2a136195-6899-4065-9889-b7c5378995e7/peerj-preprints-3190.pdf)
- 예측 지평 $H$에 대해 $H/2$ 간격으로 과거 포인트 선택
- LOESS 또는 Isotonic Regression으로 오류 곡선 추정
- 모든 예측 방법에 공통 적용 가능

### 4.2 자동 문제 식별

분석가 개입이 필요한 예측 플래그 기준: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2a136195-6899-4065-9889-b7c5378995e7/peerj-preprints-3190.pdf)
1. 베이스라인 대비 큰 오차 → 모델 오정
2. 모든 방법 동시 실패 → 이상치/구조 변화
3. 오차 급증 추세 → 데이터 생성 프로세스 변화

***

## 5. Analyst-in-the-Loop 철학

### 5.1 분석가 개입의 4가지 방식

Prophet의 설계 철학: **비전문가도 도메인 지식으로 조정 가능** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2a136195-6899-4065-9889-b7c5378995e7/peerj-preprints-3190.pdf)

| 개입 | 매개변수 | 효과 | 예시 |
|------|---------|------|------|
| 용량 설정 | $C(t)$ | 장기 포화 수준 | 시장 규모 상한 |
| 변점 지정 | Known $s_j$ 또는 $\tau$ | 추세 분절 | 제품 출시 날짜 |
| 계절성 기간 | $P$ 추가 | 새 주기 포착 | "분기별 최고점" |
| 평활도 조정 | $\tau, \sigma, \nu$ | 강성/유연성 | "더 유연한 모형" |

### 5.2 통계 vs 판단식 예측의 재정의

**전통적 이분법의 문제**:
- 통계식: 자동이지만 경직됨
- 판단식: 정확하지만 비용 높음

**Prophet의 해답**: 통계 프레임워크 + 선택적 전문가 개입 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2a136195-6899-4065-9889-b7c5378995e7/peerj-preprints-3190.pdf)
→ 도메인 지식을 코딩된 매개변수로 변환

***

## 6. 2020년 이후의 발전: 최신 연구와의 비교

### 6.1 NeuralProphet: 신경망 추가 (2021) [arxiv](https://arxiv.org/pdf/2111.15397.pdf)

**개념**: Prophet + AR-Net (자기회귀 신경망) [arxiv](https://arxiv.org/pdf/2111.15397.pdf)

$$\hat{y}_t = g(t) + s(t) + h(t) + \text{AR-Net}(y_{t-p},...,y_{t-1})$$

**성능 개선**: [arxiv](https://arxiv.org/pdf/2111.15397.pdf)
- 단기 예측(1-7일): Prophet 대비 55-92% 오류 감소
- 로컬 동역학 포착 가능

**한계**: 소규모 데이터에서 Prophet 이상 성과 못함 [arxiv](https://arxiv.org/pdf/2111.15397.pdf)

### 6.2 Transformer 기반 방법 (2020-2025)

#### Temporal Fusion Transformer (TFT)
**아키텍처**:
- Encoder: 과거 정보에서 특성 추출
- Multi-Head Self-Attention: 다중 시간 스케일

**성능**: [gmd.copernicus](https://gmd.copernicus.org/preprints/gmd-2020-270/gmd-2020-270.pdf)
- 장기 예측(180일+): LSTM 대비 15-40% 개선
- 의료기기 판매: MAPE 2.63% (vs 5%+ LSTM)

**한계**: 대량 데이터 필요, 해석성 부족

#### RevIN (Reversible Instance Normalization)
**원리**: 각 시계열을 평균/분산으로 정규화 후 학습 [peerj](https://peerj.com/articles/cs-1001)

$$\tilde{y}_t = \frac{y_t - \mu}{\sigma + \epsilon}$$

**효과**: [peerj](https://peerj.com/articles/cs-1001)
- 분포 이동 강건성
- 로컬 패턴 정확도 향상

### 6.3 하이브리드 방법의 성공 사례

**LSTM + Prophet**: [iieta](https://iieta.org/download/file/fid/106740)
- STL 분해 → Prophet(계절성) + LSTM(추세) 병렬 처리
- 에너지 소비 예측에서 경쟁력 있는 결과

**Prophet + Bayesian Gaussian Process**: [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC9202617/)
- 관광 수요: 97% 정확도
- 다중 방법론의 강점 결합

***

## 7. 일반화 성능 향상: 핵심 과제와 해결책

### 7.1 분포 이동 문제

**현상**: COVID-19처럼 이례적 사건이 패턴을 급변시킴 [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12026713/)
- 학습 데이터(2019년): 정상 상태
- 테스트 데이터(2020년): 급변 상황
- 결과: 과거 과적합 모델이 미래 예측 실패

**실증**: 미세먼지 예측에서 2020년 COVID 영향 시 Prophet 과대추정 [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12026713/)

### 7.2 해결책 1: Instance Normalization

**방법**: [peerj](https://peerj.com/articles/cs-1001)
1. 입력 정규화: $\tilde{y} = (y - \mu) / \sigma$
2. 모델 학습: 정규화된 스케일
3. 출력 역정규화: $\hat{y}\_{\text{original}} = \hat{y}_{\text{norm}} \times \sigma + \mu$

**효과**: 분포 이동 불변성, 로컬 정보 보존 [peerj](https://peerj.com/articles/cs-1001)

### 7.3 해결책 2: 데이터 증강

#### 시간 도메인 증강
- **DTW(Dynamic Time Warping)**: 시간 축 조정 [linkinghub.elsevier](https://linkinghub.elsevier.com/retrieve/pii/S2405844023097529)
- **효과**: 예측 정확도 15% 이상 향상

#### 진폭 도메인 증강
- **Gaussian Noise**: $y' = y + \mathcal{N}(0, \alpha \cdot \sigma)$
- **Amplitude Scaling**: 스케일 확대/축소 [linkinghub.elsevier](https://linkinghub.elsevier.com/retrieve/pii/S2405844023097529)
- **효과**: 모델 강건성 30% 향상, 이상치 처리 능력 개선 [linkinghub.elsevier](https://linkinghub.elsevier.com/retrieve/pii/S2405844023097529)

### 7.4 해결책 3: Foundation Models와 전이학습

**Time-LLM 방식**:
- 사전학습: 수백만 시계열, 다양한 도메인
- 미세조정: 새 도메인 소규모 데이터로 빠른 적응 [mdpi](https://www.mdpi.com/1660-4601/22/4/562)
- **효과**: 데이터 부족 상황에서 성능 극적 개선 [mdpi](https://www.mdpi.com/1660-4601/22/4/562)

***

## 8. 앞으로의 연구 고려사항

### 8.1 데이터 전처리의 중요성

Prophet이든 신경망이든, **입력 품질이 최우선**:

1. **이상치 처리**:
   - 자동 탐지: Isolation Forest, LOF
   - 처리: 제거 vs 보간 vs 모델 강건성

2. **누락값**:
   - Prophet: 선형/평균 보간
   - 심화된 경우: Multiple Imputation

3. **정상성 확보**:
   - 추세 제거, 계절성 제거, Box-Cox 변환 [linkinghub.elsevier](https://linkinghub.elsevier.com/retrieve/pii/S2405844023097529)

### 8.2 도메인 적응 전략

**새 시계열 적용 시**: [linkinghub.elsevier](https://linkinghub.elsevier.com/retrieve/pii/S2405844023097529)
1. 초기 진단: 패턴 식별
2. 모델 선택: Prophet/NeuralProphet/Transformer
3. 매개변수 튜닝: 검증 데이터 활용
4. 지속적 모니터링: 성능 저하 감지

### 8.3 불확실성 정량화 고도화

**개선 방향**:
1. **확률적 예측**: 분위수 손실, Conformal Prediction [journal.kaopg.or](https://journal.kaopg.or.kr/articles/xml/7v7J/)
2. **Calibration**: 예측 불확실성과 실제 오차 일치 [journal.kaopg.or](https://journal.kaopg.or.kr/articles/xml/7v7J/)
3. **구조적 불확실성**: 모델 앙상블, 데이터 불확실성 [journal.kaopg.or](https://journal.kaopg.or.kr/articles/xml/7v7J/)

### 8.4 설명 가능성의 강화

**현대적 발전 결합**:
1. SHAP/LIME: 신경망 변수 중요도 [journal.kaopg.or](https://journal.kaopg.or.kr/articles/xml/7v7J/)
2. Attention 시각화: Transformer 주의 패턴 [journal.kaopg.or](https://journal.kaopg.or.kr/articles/xml/7v7J/)
3. 반사실적 설명: "만약 X가 Y였다면" [journal.kaopg.or](https://journal.kaopg.or.kr/articles/xml/7v7J/)

***

## 9. 종합 결론

### Prophet의 지속적 가치

1. **산업 표준**: 8년 후에도 광범위하게 사용 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2a136195-6899-4065-9889-b7c5378995e7/peerj-preprints-3190.pdf)
2. **우수한 기본 성능**: 많은 상황에서 깊은 신경망 능가
3. **접근성**: 비전문가도 사용 가능

### 2020년 이후의 발전 패턴

| 영역 | 핵심 발전 | 실무 의미 |
|------|---------|---------|
| **로컬 동역학** | NeuralProphet AR-Net [arxiv](https://arxiv.org/pdf/2111.15397.pdf) | 단기 정확도 중요 인식 |
| **장거리 의존성** | Transformer [gmd.copernicus](https://gmd.copernicus.org/preprints/gmd-2020-270/gmd-2020-270.pdf) | 다중 시간 스케일 필수 |
| **분포 강건성** | RevIN, 증강 [peerj](https://peerj.com/articles/cs-1001) | 현실 데이터의 변동성 대응 |
| **확장성** | Foundation Models [mdpi](https://www.mdpi.com/1660-4601/22/4/562) | 데이터 부족 상황 개선 |

### 실무 권고

**기존 Prophet 사용자**:
- 필요시 NeuralProphet 단계적 업그레이드 [arxiv](https://arxiv.org/pdf/2111.15397.pdf)
- 해석성 유지하면서 성능 향상

**새로운 프로젝트**:
- 데이터 충분 → Transformer [gmd.copernicus](https://gmd.copernicus.org/preprints/gmd-2020-270/gmd-2020-270.pdf)
- 데이터 부족 & 해석성 중요 → Prophet/NeuralProphet [arxiv](https://arxiv.org/pdf/2111.15397.pdf)
- 불확실성 중요 → 확률적 방법 [journal.kaopg.or](https://journal.kaopg.or.kr/articles/xml/7v7J/)

**프로덕션 시스템**:
- Analyst-in-the-Loop 원칙 유지 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2a136195-6899-4065-9889-b7c5378995e7/peerj-preprints-3190.pdf)
- 자동 모니터링 및 주기적 재훈련 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2a136195-6899-4065-9889-b7c5378995e7/peerj-preprints-3190.pdf)
- Foundation Models 활용 검토 [mdpi](https://www.mdpi.com/1660-4601/22/4/562)

***

## 참고문헌

<span style="display:none">[^1_11][^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_80]</span>

<div align="center">⁂</div>

[^1_1]: peerj-preprints-3190.pdf

[^1_2]: https://arxiv.org/pdf/2111.15397.pdf

[^1_3]: https://gmd.copernicus.org/preprints/gmd-2020-270/gmd-2020-270.pdf

[^1_4]: https://peerj.com/articles/cs-1001

[^1_5]: https://iieta.org/download/file/fid/106740

[^1_6]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9202617/

[^1_7]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12026713/

[^1_8]: https://linkinghub.elsevier.com/retrieve/pii/S2405844023097529

[^1_9]: https://www.mdpi.com/1660-4601/22/4/562

[^1_10]: https://journal.kaopg.or.kr/articles/xml/7v7J/

[^1_11]: https://discuss.pytorch.kr/t/lstm-vs-gru-vs-transformer/5879

[^1_12]: https://www.impactive-ai.com/tech/how-time-series-data-augmentation-improves-ai-prediction-accuracy

[^1_13]: https://royzero.tistory.com/entry/python-prophet-samsung-stock-forecast

[^1_14]: https://appliedai.skku.edu/appliedailab/journal_pub.do?mode=download\&articleNo=30079\&attachNo=25946

[^1_15]: https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11022689

[^1_16]: https://dining-developer.tistory.com/25

[^1_17]: https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE12025186

[^1_18]: https://www.lgresearch.ai/blog/view?seq=427

[^1_19]: https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE09371907

[^1_20]: https://www.manuscriptlink.com/society/kips/conference/ack2025/file/downloadSoConfManuscript/abs/KIPS_C2025B0253F

[^1_21]: https://koreascience.kr/article/JAKO202410772364958.pdf

[^1_22]: https://wikidocs.net/266272

[^1_23]: https://linda284.tistory.com/7

[^1_24]: https://www.businessresearchinsights.com/ko/market-reports/time-series-forecasting-market-114943

[^1_25]: https://pdfs.semanticscholar.org/cabe/c217481d6938952e9853029f74c2d7bbcf93.pdf

[^1_26]: https://pubmed.ncbi.nlm.nih.gov/33180811/

[^1_27]: https://pubmed.ncbi.nlm.nih.gov/38438528/

[^1_28]: https://pubmed.ncbi.nlm.nih.gov/33588711/

[^1_29]: https://arxiv.org/html/2601.05929v2

[^1_30]: https://arxiv.org/html/2508.15369v1

[^1_31]: https://www.semanticscholar.org/paper/Predicting-Customer-Behavior-Using-Prophet-In-A-Liço-Enesi/710b6caeeab68ce9f62826e3176c7eda9c7ebaee

[^1_32]: https://www.arxiv.org/abs/2410.15217

[^1_33]: https://pubmed.ncbi.nlm.nih.gov/35721410/

[^1_34]: https://www.arxiv.org/abs/2505.12761

[^1_35]: https://www.semanticscholar.org/paper/Forecasting-PV-Panel-Output-Using-Prophet-Time-Shawon-Akter/de724d83866c5cb8487eae1b72159e8226fa1b6b

[^1_36]: https://pubmed.ncbi.nlm.nih.gov/41028729/

[^1_37]: https://arxiv.org/html/2504.01509v1

[^1_38]: https://arxiv.org/abs/2505.12761

[^1_39]: https://www.semanticscholar.org/paper/5bd3cf7782dfc5050690230bf8261908b59a7220

[^1_40]: https://www.semanticscholar.org/paper/ad83ae14dda563ae2f1f84999e166637354f4a8d

[^1_41]: https://www.semanticscholar.org/paper/2440572233f965766a672cdb159aea29fc618b65

[^1_42]: https://www.semanticscholar.org/paper/f6748f0351996368218b960e667112901656e6e3

[^1_43]: https://www.semanticscholar.org/paper/01bea86ce1af7b53d3b804f8fc276676cc5f5b39

[^1_44]: https://www.semanticscholar.org/paper/1b8dfcecb68e0daa87e30f85dacdfe13a6ba75b6

[^1_45]: https://www.semanticscholar.org/paper/d8fa62489c246c554e912a71c2a4d67f50f6a7ef

[^1_46]: https://www.semanticscholar.org/paper/b8fe1760b552e8a3c9131614f3a08726ef7a595a

[^1_47]: https://www.semanticscholar.org/paper/2b618b3d13f1a53fbd9251492eae52b98ad259bf

[^1_48]: http://koreascience.or.kr/journal/view.jsp?kj=HHGHHL\&py=2021\&vnc=v59n2\&sp=191

[^1_49]: https://arxiv.org/pdf/2001.04063.pdf

[^1_50]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10912208/

[^1_51]: http://arxiv.org/pdf/2312.09912.pdf

[^1_52]: https://arxiv.org/pdf/2208.05607.pdf

[^1_53]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9712550/

[^1_54]: https://arxiv.org/html/2410.09206

[^1_55]: https://pypi.org/project/neuralprophet/0.2.6/

[^1_56]: http://daddynkidsmakers.blogspot.com/2024/04/blog-post.html

[^1_57]: https://ai.atsit.in/posts/3494940688/

[^1_58]: https://bongholee.com/python-neural-prophet/

[^1_59]: https://doheon.github.io/코드구현/time-series/ci-4.transformer-post/

[^1_60]: https://mods.go.kr/boardDownload.es?bid=11912\&list_no=381837\&seq=1

[^1_61]: https://neuralprophet.com

[^1_62]: https://datascience0321.tistory.com/46

[^1_63]: https://www.youtube.com/watch?v=pplTpneacAE

[^1_64]: https://aboutnlp.tistory.com/55

[^1_65]: https://haeun161.tistory.com/28

[^1_66]: https://velog.io/@parkchansaem/논문리뷰Are-Transformers-Effective-for-Time-Series-Forecasting

[^1_67]: https://codingalzi.github.io/dlp2/fundamentals_of_ml.html

[^1_68]: https://journals.plos.org/plosone/article/file?type=supplementary\&id=10.1371%2Fjournal.pone.0324000.s003

[^1_69]: https://pdfs.semanticscholar.org/b941/621d469a1835518f6f1199d49b219c2b322e.pdf

[^1_70]: https://pdfs.semanticscholar.org/399b/ff5a0102e2d69538b6b8544bbad201667aee.pdf

[^1_71]: https://pubmed.ncbi.nlm.nih.gov/36177230/

[^1_72]: https://pdfs.semanticscholar.org/2ec6/45ee481499ca8e91744756d09b01117542a6.pdf

[^1_73]: https://www.semanticscholar.org/paper/NeuralProphet:-Explainable-Forecasting-at-Scale-Triebe-Hewamalage/979a165d4638d2c329bd6577700e1d88dbc50b97

[^1_74]: https://pdfs.semanticscholar.org/f56c/aea6a7cbe75be459aa1ba3696870054c42dc.pdf

[^1_75]: https://arxiv.org/abs/2111.15397

[^1_76]: https://pdfs.semanticscholar.org/5da5/a48a36faa3b07f15e927cccdede2203f97ef.pdf

[^1_77]: https://pubmed.ncbi.nlm.nih.gov/35258438/

[^1_78]: https://pdfs.semanticscholar.org/147f/358d40775a73035996558f427c25e86b4a48.pdf

[^1_79]: https://pubmed.ncbi.nlm.nih.gov/35194090/

[^1_80]: https://pdfs.semanticscholar.org/3717/c8c86d547af501f5481bec1df58502b1eec1.pdf
