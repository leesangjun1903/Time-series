
# Stationarity and nonstationarity in time series analysis
## 1. 논문의 핵심 주장과 주요 기여 (간략 요약)

Manuca와 Savit (1996)의 "Stationarity and Nonstationarity in Time Series Analysis"는 **비정상 시계열을 분석하기 위한 혁신적 방법론**을 제시합니다.[1]

### 핵심 주장

1. **정상성 판정의 근본적 문제**: 유한 시계열에서 정상성을 명확히 정의하고 검정할 수 있는 통일된 방법이 없음[1]
2. **메타동역학(Metadynamics) 개념**: 시스템을 내부 동역학 $x$와 외부 구동력 $y$로 분리[1]
3. **동역학적 거리 기반 접근**: 상관 적분을 기반으로 두 시계열 세그먼트의 동역학적 근접성을 직접 측정[1]

### 주요 기여

- **동역학적 거리 측정**: $R_m$ (예측 가능성 비율) 통계량으로 낮은 분산의 거리 측정[1]
- **윈도우 클러스터링**: 동일 메타상태의 시계열 세그먼트 자동 그룹화로 통계 개선[1]
- **메타위상공간 구성**: 비정상성의 구조를 시각화하고 해석 가능한 표현 제공[1]

***

## 2. 해결하는 문제와 제안 방법 (수식 포함)

### 2.1 핵심 문제들

**문제 1**: 전통적 방법의 한계
- 선형 분석: 평균/분산만 검토 → 비선형 특성 놓침[1]
- 작은 윈도우: 통계적 신뢰도 감소[1]

**문제 2**: 비정상성 원인 파악 곤란
- 트렌드? 계절성? 환경 변화? 불명확[1]

### 2.2 제안 방법론

#### 동역학적 시스템 분해

**비자율 시스템의 재해석**:

$$\dot{x} = F(x,t) \rightarrow \begin{cases} \dot{x} = G(x, y) \\ \dot{y} = H(y, t) \end{cases}$$

여기서:
- $x$: 내부 동역학 (에르고딕 가정)
- $y$: 구동력 (시간 변화)
- $\dot{y} = H(y,t)$: 메타동역학[1]

**핵심**: $y$가 반복적이면 재구성 변수로 추가 가능 → 위상공간 재구성 개선[1]

#### 동역학적 거리 측정

**상관 적분** (Grassberger-Procaccia):

$$C_m(x,x) = P(\|x^{(m)}(i) - x^{(m)}(j)\| < \varepsilon)$$

여기서 $x^{(m)}(i) = (x(i), x(i-1), \ldots, x(i-m+1))$[1]

**교차 상관 적분**:

$$C_m(x,y) = P(\|x^{(m)}(i) - y^{(m)}(j)\| < \varepsilon)$$

**예측 가능성**:

$$S_m = \frac{C_{m+1}}{C_m}$$

**예측 가능성 비율** (핵심 지표):

$$R_m = \frac{S_m}{S_{m-1}}$$

**이점**: 분산이 최소 (0.013 vs 0.064 for $C_m$, 표 1에서 Lorenz 시스템)[1]

**대칭화된 거리**:

$$d^{(m)}(x,y) = \max(|R_m(x,x) - R_m(x,y)|, |R_m(y,y) - R_m(x,y)|)$$

**정규화**:

$$d^{(m)}(x,y) = \frac{\sqrt{N_{tot}}}{C_m^{(m)}} \max(|R_m(x,x) - 2R_m(x,y) + R_m(y,y)|)$$[1]

#### 윈도우 클러스터링

**목표**: 동일 메타상태 윈도우 그룹화로 통계 개선

**알고리즘**:

1. 불일치 행렬 계산:
$$M_{ij} = \sum_{k=1}^{n} [\text{if } |d_{ik} - d_{jk}| > \chi \text{ then } 1]$$

2. 각 윈도우의 가까운 이웃 수:
$$N_i = \sum_{j=1}^{n_u} \Theta(\chi/n - M_{ij})$$

3. 최대 이웃을 가진 윈도우부터 클러스터 형성
4. 반복: 조건 만족하는 윈도우 추가
$$\langle M_{i'j'} \rangle_{j'} < \chi \Rightarrow \text{추가}$$[1]

**효과**: 256개 윈도우(390점) → 3개 클러스터로 통계 크기 약 9배 증가[1]

#### 메타위상공간 구성

**기준 윈도우 선택** (정보 최대화):
- 거리 분포의 상관 적분 최소화 → 최대 엔트로피
- 다음: 이전 선택과 선형 상관 최소화

**삼각분할으로 위치 결정**:

$$\vec{r}_i = (d(w_i, w_{ref}^{(1)}), d(w_i, w_{ref}^{(2)}), \ldots)$$

**이점**: 차원 절감 (1500 윈도우 → 3차원)[1]

***

## 3. 모델 구조와 성능 향상

### 3.1 모델 아키텍처

```
시계열 입력
    ↓
[시간 척도 선택: 윈도우 너비 T]
    ↓
[위상공간 재구성]
(임베딩 차원 m, 해상도 ε)
    ↓
[동역학적 거리 계산]
(Rm 기반)
    ↓
[윈도우 클러스터링]
(메타상태 식별)
    ↓
[메타위상공간 구성]
(기준점 선택 + 삼각분할)
    ↓
메타상태 시퀀스 & 메타위상공간
```

### 3.2 성능 향상 메커니즘

**1. 통계적 개선**

클러스터링을 통해 같은 메타상태 윈도우 연결:

$$N_{eff} = \sum_{c=1}^{k} n_c$$

**예시**: 볼 베어링 데이터[1]
- 원래: 1500개 윈도우 × 2400점 = 3.6M 점
- 클러스터링 후: 3 메타상태 × ~500 윈도우
- 통계 신뢰도: $\sqrt{500} \approx 22$배 향상

**2. 차원 최적화**

엔트로피 기준 기반 차원:

$$d^* = \arg\max_d H(d_1, \ldots, d_k)$$

레이저 데이터:[1]
- 원 시계열: 64개 윈도우 × 157점 (1차원)
- 메타위상공간: 3차원
- 정상성: 4차원에서 확인 (이론 예측 일치)

**3. 비선형 특성 포착**

$R_m$ vs 상관 적분 비교 (Lorenz 시스템, 표 1):[1]

| 척도 | 평균 | 표준편차 | 상대분산 |
|------|------|----------|---------|
| $C_2$ | 0.0291 | 0.00082 | 0.064 |
| $S_1$ | 0.232 | 0.012 | 0.051 |
| $R_1$ | 4.184 | 0.053 | **0.013** |

$R_1$의 분산이 최소: 신호 잡음비 우수

***

## 4. 모델의 일반화 성능 향상 가능성

### 4.1 핵심 일반화 기제

**1. 메타동역학 주기성 활용**

구동력 $y$가 반복적이면:

$$P(\text{정확한 예측}) \propto \sqrt{N_{\text{메타상태 반복 횟수}}}$$

**드리프트 감지 (데이터셋 D)**:[1]
- 시스템: 9자유도 포텐셜에서 입자 운동
- 우물 깊이의 느린 드리프트 추가
- 메타위상공간 y좌표: $0.6 \rightarrow 0.2$ (4분)
- 드리프트율: $\Delta y / \Delta t = 0.1$/분

→ 드리프트 알고 있으면 장기 예측 개선 가능

**2. 차원 적응 선택**

시간 척도에 따른 비정상성 다름:

- 낮은 m: 비정상성 명확
- 높은 m: 정상성 회복

"정상성 기준선":[1]
$$T^* = T: \text{ 비정상성 최소}$$

더 큰 T에서 정상적이면 → 추정 신뢰성 증가

**3. 주기적 구동력 이용**

볼 베어링 데이터 (도 16):[1]
- 주기: $P = 84$초
- 메타위상공간 x좌표 진동
- 맥동 구조 발견 (위상 6, 7)
- 재앙 전 3분 전 축 회전 시작

→ 패턴 인식으로 새 시스템 일반화 가능

### 4.2 구체적 성능 개선

**레이저 데이터** (자율 시스템, 예측 가능성 개선):[1]
- 선형 분석: 비정상 신호 ($\Gamma(0)$ 변화)
- 기존 비선형 (4D 상관 적분): 비정상 신호
- **Manuca-Savit (4D $R_m$): 정상** ✓
- 결론: 2-3D는 비정상적, 4D에서 정상 (시스템 이론 일치)

**볼 베어링 조기 탐지** (재앙 감지):[1]
- 전통 방법: 파괴 5-6초 전 탐지
- **메타위상공간 축 회전**: 3분 전 시작
- 주기성 감지: 84초 주기 구조

→ 조기 경고 가능, 일반화 적용 가능

### 4.3 일반화 한계

**1. 데이터 길이 의존성**

$$\text{신뢰도} \propto \sqrt{N}$$

매우 짧은 시계열: 메타상태 반복 불가능

**2. 모수 선택 민감성**[1]

결정 모수:
- 윈도우 너비 $T$: 이완 시간 > T
- 임베딩 차원 $m$: 상관차원 > m
- 해상도 $\varepsilon$: 0.05σ ~ 0.2σ
- 근처 임계값 $\chi$: 0.1n

**3. 적응 시스템 미지원**[1]

> "진화하는 적응 시스템을 이 틀에서 연구할 수 있는지, 어느 정도까지 가능한지는 흥미로운 질문입니다."

메타동역학 자체가 x에 의존하면 실패

**4. 고차원 문제**

$$\text{필요 샘플} \propto \exp(d)$$

차원 저주로 고차원 확장 어려움

***

## 5. 2020년 이후 최신 연구와의 관계

### 5.1 비정상성 처리의 진화

**최근 방법들의 비교**:

| 방법 | 연도 | 비정상 처리 | 메타상태 | 해석성 | 규모성 |
|------|------|-----------|---------|-------|--------|
| Manuca-Savit | 1996 | 메타동역학 | 명시적 | 높음 | 중간 |
| RevIN | 2022 | 정규화 (평균/분산) | 없음 | 낮음 | 높음 |
| DERITS | 2024 | 주파수 미분 | 없음 | 중간 | 높음 |
| UDA | 2025 | HMM 기반 | 잠재 | 중간 | 높음 |

**연결성**:[2][3][4][5][6]
- 메타동역학 ↔ 분포 이동(Distribution Shift)
- 윈도우 클러스터 ↔ 체제 변화 (Regime Change)
- 메타위상공간 ↔ 잠재 표현 (Latent Representation)

### 5.2 최신 구체적 방법들

**1. DynaConF (2024)**[2]
- 조건부 분포 모델링
- 조건부 분포 비정상 처리
- 신경망 기반 (해석성 낮음)

**2. TimeBridge (2024)**[3]
- 비정상성 중요성 강조
- 장기 시계열 거짓 회귀 방지
- 트렌드-계절성 분해

**3. Deep Frequency Derivative Learning (DERITS) (2024)**[6]
- 주파수 영역 미분
$$\text{트렌드 제거: } \frac{d}{df}X(f) \rightarrow \text{더 정상적인 신호}$$
- 여러 차수 미분 병렬 처리

**4. Unknown Distribution Adaptation (UDA) (2025)**[5]
- 분포 이동 시점 미지
- 숨겨진 환경 변수 추론
- 마르코프 모델

### 5.3 Manuca-Savit의 현대적 영향

**직접 인용** (2020+):
- 비정상 수문 시계열 (2024)[7]
- 모델 적응적 위상공간 재구성 (2024)[8]
- 시간-주파수 분석 (TrendLSW, 2024)[4]

**개념적 재해석**:
- 메타동역학 = 분포 시프트의 선구 개념
- Koopman 이론과 연결 (2023)[9]
- 신경 ODE와 통합

***

## 6. 향후 연구 고려사항

### 6.1 방법론적 확장

**1. 고차원 시스템**

**Tensor 접근** (2024):
$$\text{다선형 위상공간 재구성}$$
정보 손실 최소화[10]

**스파스 임베딩**:
$$\phi = \{\tau_i : |\partial d/\partial \tau_i| > \theta\}$$
중요 지연만 선택

**2. 적응 시스템**

**문제**: 메타동역학 $H(y,t)$가 x에 의존

**접근**:
- 메타 학습 (Meta-learning)
- 비정상 RNN
- 연속 적응 (Continual learning)

### 6.2 응용 확대

**1. 다변량 시계열**

VANDA (2024):[11]
$$\text{변수 연관성} = g_{\text{GNN}}(\text{변수들})$$

**메타상태별 다변량 상관**:
$$C_m(x^{(1)}, x^{(2)}, \ldots) = \text{다변량 상관}$$

**2. 금융 시계열**

- 포트폴리오 재조정 신호
- 시장 체제 변화 조기 탐지
- VaR 동적 추정

**3. 생의학 신호** (ECG, EEG, EMG)

- 질병 상태 변화 감지
- 개인맞춤형 기준선
- 임상 개입 타이밍

**장점**: 해석성, 작은 데이터 효율성

### 6.3 이론적 심화

**1. 메트릭 공간 성질**

Wasserstein 거리:[12]
$$d_W(P_1, P_2) = \inf_{\gamma} \mathbb{E}[\|X-Y\|]$$

**리만 기하학**: Gromov-Wasserstein 거리

**2. 통계적 검정**

부트스트랩:
$$p = P(d^* > d_{\text{obs}})$$

베이지안:
$$P(y_1 = y_2 | d) = ?$$

### 6.4 계산 효율성

**근사 최근린 이웃** (2024):
$$O(n^2) \rightarrow O(n \log n)$$
LSH, HNSW 알고리즘[8]

**실시간 스트리밍**:
- 슬라이딩 윈도우
- 증분 삼각분할
- 적응적 기준점

***

## 7. 최종 결론

### 7.1 논문의 과학사적 위치

**혁신성**:
- 1990년대 비정상성의 동역학적 재정의
- 메타동역학 개념의 선구적 제안
- 상관 적분을 넘어 $R_m$의 낮은 분산 활용

**현재적 가치** (2024):
- "분포 시프트" 개념의 선례
- 메타동역학 ↔ 숨겨진 마르코프 사슬
- 해석 가능한 머신러닝의 모범

### 7.2 강점과 약점

**강점 ✓**:
- 높은 해석성 (메타위상공간 시각화)
- 비선형 동역학 직접 포착
- 작은 데이터에서도 작동
- 물리적 직관 제공

**약점 ✗**:
- 모수 선택 민감성 ($T, m, \varepsilon, \chi$)
- 고차원에서 확장성 문제 ( $O(n^2)$ )
- 적응 시스템 미지원
- 엄밀한 메트릭이 아님

### 7.3 향후 방향

**단기** (2024-2025):
- 고차원 확장
- 신경망 통합
- 금융/의료 응용

**중기** (2025-2027):
- 적응 시스템 처리
- 확률적 프레임워크
- 대규모 실시간 시스템

**장기** (2027+):
- 범용 비정상 이론 통합
- 자동 모수 선택
- 도메인 간 전이 학습

### 최종 평가

Manuca-Savit (1996)은 **시계열 정상성 분석의 패러다임 전환**을 제시했습니다. 메타동역학 개념은 오늘날의 "분포 시프트" 개념의 수학적 선구이며, 메타위상공간은 신경망의 "잠재 표현"에 대한 해석 가능한 대안입니다. 비정상 시계열 분석이 중요성을 인정받는 현대 (2020+)에서도 이 프레임워크의 기초적 가치와 실무적 우수성은 지속적으로 인정받고 있습니다.[7][4][11][8]

***

## 참고 논문 (2020년 이후)

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/89e676fe-727d-4f0e-98fd-20f9f973d795/stationary.nonstationary.physicaD.pdf)
[2](https://arxiv.org/html/2209.08411v3)
[3](https://arxiv.org/html/2410.04442v2)
[4](https://arxiv.org/html/2406.05012v1)
[5](https://arxiv.org/html/2402.12767v4)
[6](https://www.ijcai.org/proceedings/2024/0436.pdf)
[7](https://iwaponline.com/jh/article/26/9/2085/103012/Spatio-temporal-nonstationarity-analysis-and)
[8](https://publications.pik-potsdam.de/rest/items/item_30029_2/component/file_30069/content)
[9](https://arxiv.org/pdf/2210.03675.pdf)
[10](https://openreview.net/notes/edits/attachment?id=j7TFD0vrNX&name=pdf)
[11](https://dl.acm.org/doi/10.1145/3663573)
[12](https://arxiv.org/html/2409.19718v1)
[13](https://www.mdpi.com/2077-0383/13/2/534)
[14](https://www.bmj.com/lookup/doi/10.1136/bmj-2024-080944)
[15](https://www.nature.com/articles/s41598-024-53547-6)
[16](https://academic.oup.com/cid/article/79/2/312/7655672)
[17](https://capmh.biomedcentral.com/articles/10.1186/s13034-024-00757-5)
[18](http://www.thejas.com.pk/index.php/pjhs/article/view/1547)
[19](https://ieeexplore.ieee.org/document/10720277/)
[20](https://link.springer.com/10.1007/s10346-024-02249-1)
[21](https://link.aps.org/doi/10.1103/PRXQuantum.5.040325)
[22](https://arxiv.org/pdf/2104.01293.pdf)
[23](https://arxiv.org/abs/2307.01315)
[24](https://pubs.aip.org/aip/cha/article-pdf/doi/10.1063/5.0189402/19862790/043105_1_5.0189402.pdf)
[25](https://pmc.ncbi.nlm.nih.gov/articles/PMC10906746/)
[26](https://encord.com/blog/time-series-predictions-with-recurrent-neural-networks/)
[27](https://ieeexplore.ieee.org/document/9204077/)
[28](https://www.tandfonline.com/doi/full/10.1080/00273171.2024.2436413)
[29](https://www.nature.com/articles/s41598-024-69418-z)
[30](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5215521)
[31](https://proceedings.neurips.cc/paper_files/paper/2024/file/37c6d0bc4d2917dcbea693b18504bd87-Paper-Conference.pdf)
[32](https://www.sciencedirect.com/science/article/abs/pii/S1877750319301838)
[33](https://www.nature.com/articles/s41598-025-13680-2)
[34](https://arxiv.org/html/2505.06917v1)
[35](https://dl.acm.org/doi/10.1145/3649329.3658477)
[36](https://ieeexplore.ieee.org/document/10654246/)
[37](https://www.ijcai.org/proceedings/2024/489)
[38](https://ieeexplore.ieee.org/document/10822753/)
[39](https://ieeexplore.ieee.org/document/10903299/)
[40](https://ieeexplore.ieee.org/document/10605466/)
[41](https://link.springer.com/10.1007/s11042-024-18687-x)
[42](https://www.ijcai.org/proceedings/2020/314)
[43](https://www.mdpi.com/2072-4292/12/19/3196)
[44](http://arxiv.org/pdf/2407.15174.pdf)
[45](https://arxiv.org/pdf/2310.14838v1.pdf)
[46](https://linkinghub.elsevier.com/retrieve/pii/S0031320324005624)
[47](https://arxiv.org/html/2503.01157v1)
[48](https://arxiv.org/pdf/2203.08321.pdf)
[49](http://arxiv.org/pdf/2401.06524.pdf)
[50](https://proceedings.iclr.cc/paper_files/paper/2024/file/440f269a4a6b9d51c51b4997963761ff-Paper-Conference.pdf)
[51](https://www.jatit.org/volumes/Vol102No24/14Vol102No24.pdf)
[52](https://eurasip.org/Proceedings/Eusipco/Eusipco2024/pdfs/0001932.pdf)
[53](https://www.nature.com/articles/s41467-024-54281-3)
[54](https://www.ijcai.org/proceedings/2024/0424.pdf)
[55](https://www.arxiv.org/list/math.DS/2024-04?skip=0&show=1000)
[56](https://arc.aiaa.org/doi/10.2514/6.2022-2440)
[57](https://arxiv.org/html/2407.12128v1)
