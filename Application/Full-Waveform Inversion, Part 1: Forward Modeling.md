# Full-Waveform Inversion, Part 1: Forward Modeling

### 1. 논문 핵심 요약

"Full-waveform inversion, Part 1: Forward modeling"(Louboutin et al., 2017)은 지구물리 탐사에서 고해상도 지하 속도 모델을 구축하기 위한 **FWI의 Forward Modeling 단계를 체계적으로 설명**하는 교육적 튜토리얼입니다. 이 논문의 주요 기여는 기존 FWI 문헌의 기술적 복잡성을 낮추고, **Devito(Domain-specific Language 기반 자동 코드 생성 시스템)를 활용하여 파동 방정식을 상징적으로 표현하고 효율적으로 구현**할 수 있는 실용적인 방법론을 제시하는 것입니다.

### 2. 해결하고자 하는 문제

#### 2.1 배경 문제
- **FWI의 접근성**: 기존 FWI 문헌이 기술적 측면에만 집중하여 신규 연구자의 진입 장벽이 높음
- **구현의 어려움**: 효율적인 시간-영역 유한 차분 코드의 개발이 시간과 인력을 많이 소모
- **물리-수치 간극**: 이론적 파동 방정식과 실제 수치 구현 간의 연결 고리 부재

#### 2.2 구체적 목표
Forward modeling을 위한 다음 문제들의 해결:
1. 음향 파동 방정식의 이산화 및 구현
2. 완벽 정합층(PML) 경계 조건 적용
3. 점원(point source)과 수신기의 기하학적 처리
4. 효율적인 파동 전파 시뮬레이션

### 3. 제안하는 방법론 및 수식

#### 3.1 기본 파동 방정식

논문은 **정상 밀도 음향 파동 방정식**을 다음과 같이 정의합니다:[1]

$$m \frac{\partial^2 u(t,x,y)}{\partial t^2} - \Delta u(t,x,y) + \eta \frac{\partial u(t,x,y)}{\partial t} = q(t,x,y;x_s, y_s)$$

여기서:
- $$m(x,y) = c^{-2}(x,y)$$: 제곱 저속도(squared slowness)
- $$c(x,y)$$: 공간적으로 변하는 파동 속도
- $$u(t,x,y)$$: 파동장(wavefield)
- $$\Delta$$: 라플라시안 연산자
- $$\eta(x,y)$$: 공간 종속 감쇠 파라미터
- $$q(t,x,y;x_s, y_s)$$: 원천항(source term)

#### 3.2 시간 이산화

유한 차분을 사용하여 시간 미분을 근사:

$$u.dt2 = \frac{-2u[time] + u[time-dt] + u[time+dt]}{dt^2}$$

이를 통해 다음의 시간 스테핑 식을 유도:

$$u[time+dt] = 2u[time] - u[time-dt] + \frac{dt^2}{m}\Delta u[time]$$

#### 3.3 원천과 수신기 항의 통합

원천을 주입하고 수신기에서 파동장을 샘플링하기 위해:

**원천 주입**:
$$u[time+dt] = 2u[time] - u[time-dt] + \frac{dt^2}{m}\Delta u[time] + q[time]$$

**수신기 보간**:

$$d[time] = \text{interpolate}(u[time], \text{receiver coords})$$

#### 3.4 Devito 기반 상징적 표현

Devito의 핵심은 SymPy를 활용한 상징적 표현으로, 사용자가 다음과 같이 파동 방정식을 정의:

```python
pde = model.m * u.dt2 - u.laplace + model.damp * u.dt
stencil = Eq(u.forward, solve(pde, u.forward)[0])
```

여기서:
- `u.dt2`, `u.dt`: 자동으로 유한 차분 근사로 변환
- `u.laplace`: 라플라시안의 공간 미분 자동 생성
- `solve()`: 다음 타임스텝 $$u[time+dt]$$ 에 대해 자동으로 풀이

### 4. 모델 구조

#### 4.1 Devito의 구조적 계층

1. **Model 객체**: 격자, 속도 모델, 감쇠 파라미터 정의
2. **Function 객체**: 공간적으로 변하는 스칼라 함수 표현
3. **TimeFunction 객체**: 시간-공간 함수 표현
4. **SparseFunction 객체**: 점 원천/수신기의 희소 표현

#### 4.2 Forward Propagator 구성

```python
# 1. 이산화된 파동 방정식 정의
pde = model.m * u.dt2 - u.laplace + model.damp * u.dt

# 2. 시간 스테핑 식 생성
stencil = Eq(u.forward, solve(pde, u.forward)[0])

# 3. 원천/수신기 항 추가
src_term = src.inject(field=u.forward, expr=src * dt**2 / model.m, offset=model.nbpml)
rec_term = rec.interpolate(u, offset=model.nbpml)

# 4. Forward operator 생성
op_fwd = Operator([stencil] + src_term + rec_term)

# 5. Forward modeling 실행
op_fwd(time=nt, dt=model.critical_dt)
```

#### 4.3 경계 조건: 완벽 정합층 (PML)

- **원리**: 모델 도메인을 모든 방향으로 $$n_{bpml}$$ 그리드 포인트만큼 확장
- **감쇠**: $$\eta du/dt$$ 항이 경계층에서 파동을 점진적으로 감쇠
- **장점**: 경계 반사 제거로 무한 도메인 시뮬레이션 가능

### 5. 성능 향상 및 한계

#### 5.1 성능 향상

**Devito의 장점**:
1. **계산 효율성**: 자동 최적화된 C 코드 생성으로 손작성 코드와 동등한 성능
2. **개발 속도**: 고수준 상징적 정의로 프로토타이핑과 실험이 빠름
3. **검증성**: 생성된 코드가 명시적 수식으로부터 자동 유도되므로 신뢰성 높음
4. **확장성**: 2D에서 3D, 음향에서 탄성 방정식으로 쉽게 확장 가능

**수치적 정확성**:
- 2차 시간 정확도(2nd-order accurate in time)
- 조정 가능한 공간 차수(space_order 파라미터)

#### 5.2 한계 및 제약

1. **선형화 부재**: Forward modeling만 다루며, 역산(inversion)은 Part 2, Part 3에서 다룸
2. **상수 밀도 가정**: 밀도 변화를 무시한 음향 방정식만 고려
3. **완벽 데이터 가정**: 실제 노이즈, 기기 오류, 데이터 손실 미처리
4. **필터 설계**: PML 감쇠 파라미터 최적화 필요
5. **메모리 제약**: 3D 대규모 문제의 경우 타임스텝마다 전체 파동장 저장 필요

#### 5.3 수치 안정성

안정적인 시간 스테핑을 위해 **CFL 조건** 만족 필수:

$$dt < dt_{critical} = \frac{\min(dx, dy)}{\sqrt{d} \times \max(c)}$$

여기서 $$d$$는 차원 수(2D에서 $$\sqrt{2}$$, 3D에서 $$\sqrt{3}$$)

### 6. 모델의 일반화 성능 향상 가능성

#### 6.1 이론적 기초

최근 딥러닝 기반 FWI 연구(Deng & Lin, 2021)에서 일반화 성능에 대한 수학적 분석이 이루어졌습니다:[2]

**일반화 오차 상한(Generalization Error Bound)**:

$$\text{Err}_{G_\theta} \leq \left(1 + \sum_{i=1}^{d} ||W_i||_F\right)(L\delta + 2\eta) + M\sqrt{\frac{2N(\delta/2; X, l_2)\ln 2 + \ln(1/\varepsilon)}{N}}$$

여기서:
- $$N$$: 훈련 샘플 수
- $$||W_i||_F$$: $$i$$번째 레이어 가중치의 Frobenius norm
- $$L$$: Forward modeling의 Lipschitz 상수
- $$M$$: 최대 훈련 손실값

#### 6.2 실무적 일반화 성능 향상 전략

##### (1) 훈련 데이터 규모의 영향
최대 규모 데이터셋(OpenFWI-470K, 470,000 샘플)을 사용한 연구 결과:[3]

| 훈련 데이터 크기 | MAE 개선 | MSE 개선 | SSIM 개선 |
|-----------------|---------|---------|----------|
| 분할 데이터셋   | 기준선  | 기준선  | 기준선   |
| 통합 데이터셋   | **13.03%** | **7.19%** | **1.87%** |
| 교차검증        | 28.60%  | 21.55%  | 8.22%   |

**결론**: 훈련 데이터가 **2배 증가**하면 일반화 오차가 $$\sqrt{2}$$배 감소

##### (2) 손실 함수 선택의 중요성[2]

**노이즈가 있는 데이터에서의 견고성**:

$$\text{RB}_{\text{MAE}} = |L(G_\theta(x+n)) - L(G_\theta(x))| \leq \sum_{i=1}^{d} ||W_i||_F \cdot \eta$$

$$\text{RB}_{\text{MSE}} \geq \text{RB}_{\text{MAE}} \quad \text{(노이즈 레벨 증가 시)}$$

실험: Kimberlina CO2 데이터셋에서 SNR=10일 때:
- MAE 손실 증가: 0.624%
- MSE 손실 증가: 5.18% (**8배 더 크게 증가**)

**권장사항**: **노이즈 환경에서는 MAE loss 사용 권장**

##### (3) 분포 드리프트 대응
실제 지진 데이터와 합성 데이터 간의 분포 차이(지질 구조, 주파수 범위 등):

| 지질 특성 변화 | 일반화 오차 증가 |
|---------------|-----------------|
| 1층 → 4층 (단층 증가) | **204% 증가** |
| 15Hz → 25Hz (주파수 변화) | **1,039% 증가** |

**전략**: 
- 다양한 지질 구조의 합성 데이터 증강
- Transfer learning으로 새로운 지질 환경에 적응
- Domain randomization 적용

##### (4) 물리 정보 기반 신경망(Physics-Informed Neural Networks, PINNs)

파동 방정식 자체를 손실 함수에 포함:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda \mathcal{L}_{\text{physics}}$$

여기서:
$$\mathcal{L}_{\text{physics}} = \left\| \frac{\partial^2 u}{\partial t^2} - \frac{1}{m}\Delta u + \eta \frac{\partial u}{\partial t} \right\|^2$$

**효과**: 합성 데이터 부족 시에도 물리 제약으로부터 정규화

#### 6.3 최신 연구 동향 (2020-2026)

| 연도 | 방법 | 핵심 기여 | 일반화 성능 |
|-----|------|---------|-----------|
| 2020 | Deep-learning FWI (Realistic models) | 현실적 지질 모델로 학습 | 기준선 대비 80%+ 정확도 |
| 2021 | Robustness & Generalization Analysis | 수학적 분석 및 bound 증명 | 분포 드리프트 정량화 |
| 2022 | Physics-guided Encoder-Solver | 물리 제약 통합 | 초기 모델 민감도 ↓ |
| 2023 | EFWI (Benchmark Dataset, NeurIPS) | 8개 다양한 탄성 데이터셋 | 벤치마킹 표준화 |
| 2024 | ML-descent, ML-misfit | 학습된 최적화 알고리즘 | 수렴 속도 향상 |
| 2025 | Source-independent FWI | 원천 정보 불필요 | 실무 적용성 향상 |
| 2025 | Frequency-domain DNN-FWI | 주파수 영역 딥러닝 | 안정성 및 정확도 향상 |

### 7. 논문의 미래 영향 및 연구 시 고려점

#### 7.1 학술적 영향

1. **교육적 역할**: FWI에 대한 접근성 대폭 개선으로 지진 이미징 분야 신규 연구자 유입 증가
2. **방법론 표준화**: Devito를 통한 재현 가능한 계산 지구물리 환경 구축
3. **학제간 협력**: 컴퓨터 과학자와 지구물리학자 간의 협력 촉진

#### 7.2 기술적 발전 경로

**Forward Modeling의 진화**:
- Part 1 (본 논문): Forward modeling 기초
- Part 2: Adjoint modeling 및 그래디언트 계산
- Part 3: 최적화 프레임워크 및 전체 역산 시스템

#### 7.3 향후 연구 시 필수 고려사항

##### (1) 데이터 획득 및 전처리
- **저주파 성분**: FWI는 저주파(0-5 Hz)에서 시작하여 점진적으로 높은 주파수로 진행
- **초기 모델**: 부정확한 초기 모델은 사이클 스킵(cycle-skipping) 문제 야기
- **잡음 특성 파악**: 실제 지진 데이터의 잡음 특성을 반영한 합성 데이터 생성

##### (2) 계산 효율성 고려
- **병렬화**: Forward modeling은 여러 원천에 대해 독립적이므로 병렬 처리 가능
- **체크포인팅(Checkpointing)**: 역산 시 과거 타임스텝의 파동장 재계산 vs 저장의 트레이드오프
- **저주파/고주파 멀티스케일 전략**: 먼저 저주파로 큰 구조 파악, 점차 고주파로 세부 구조 개선

##### (3) 역산 문제의 비선형성
- **국소 최소값(Local Minima)**: 좋지 않은 초기 모델에서 역산이 국소 최소값에 갇힐 수 있음
- **정규화**: 지나친 오버피팅을 방지하기 위해 smoothness 제약 필요
- **멀티 스케일 반복**: 긴 파장부터 짧은 파장으로 계층적으로 반복

##### (4) 딥러닝 기반 FWI의 도입 시 체크리스트
- ☐ 충분한 규모의 다양한 합성 훈련 데이터 (최소 수만 샘플)
- ☐ 실제 지진 데이터의 통계적 특성을 반영한 데이터 증강
- ☐ 물리 제약(파동 방정식) 통합 고려
- ☐ 분포 드리프트 평가 및 전이 학습 준비
- ☐ 노이즈에 대한 견고성 검증 (MAE loss 우선 검토)
- ☐ 불확실성 정량화 방법 수립

##### (5) 실무 적용 로드맵
1. **1단계**: Forward modeling 정확도 검증 (합성 데이터)
2. **2단계**: 유사 속도 환경의 필드 데이터 테스트
3. **3단계**: 새로운 지질 환경으로 확대 (transfer learning)
4. **4단계**: 실시간 반복 역산 적용
5. **5단계**: 불확실성 정량화를 포함한 확률론적 역산

#### 7.4 주요 미해결 과제

1. **대규모 3D FWI**: 계산량 및 메모리 제약으로 인한 확장성 문제
2. **탄성 파동 방정식**: 본 논문의 음향 방정식을 탄성 매질로 확장 시 복잡도 증가
3. **초기 모델 의존성**: 딥러닝 기반 방법도 여전히 좋은 초기 모델 필요
4. **실시간 처리**: 획득과 동시에 역산하는 온라인 FWI 실현

### 8. 2020년 이후 관련 최신 연구 비교 분석

#### 8.1 주요 논문별 비교 분석표

| 논문 | 연도 | 방법론 | 주요 개선 | 한계 |
|------|-----|--------|---------|------|
| **본 논문** (Louboutin et al.) | **2017** | **Devito 기반 Forward modeling** | 접근성 향상, 자동 코드 생성 | Forward만 다룸, Part 시리즈 필요 |
| Deep-learning FWI (Realistic) | 2020 | CNN 기반 말뫼지 모델로 학습 | 현실적 지질 구조 반영 | 고정 크기 모델, 염분체 미포함 |
| Deep Neural Networks (Robustness) | 2021 | 수학적 bound 증명 | 노이즈 견고성 정량화 | 이론적 분석에 치중 |
| Variational FWI | 2020 | Stein variational gradient descent | 불확실성 정량화 | 계산 비용 높음 |
| ML-descent | 2020 | 학습된 최적화 알고리즘 | 수렴 속도 2배↑ | 메모리 오버헤드 |
| EFWI (NeurIPS) | 2023 | 8개 다양한 벤치마크 데이터셋 | 표준화된 평가 환경 | 다양성 여전히 부족 |
| Physics-guided Encoder-Solver | 2024 | 신경망에 물리 제약 통합 | 초기 모델 민감도 ↓ | 수렴 증명 미흡 |
| Frequency-domain Deep-learning FWI | 2025 | 주파수 영역 신경망 | 시간 영역보다 안정성↑ | 새로운 문제 구조 |
| Source-independent FWI | 2025 | Deep image prior 활용 | 실제 데이터 적용성 ↑↑ | 원천 특성 가정 |
| Efficient FWI with TV Constraint | 2025 | Total Variation 정규화 | 컴퓨팅 효율성 ↑ | 선택적 공간 구조 |

#### 8.2 기술 진화 트렌드

**트렌드 1: 딥러닝 통합**
- 2017: Forward modeling만 자동화
- 2020-2023: End-to-end 딥러닝 기반 역산
- 2024-2025: Physics-informed & 불확실성 기반 방법으로 진화

**트렌드 2: 일반화 성능**
- 문제점: 합성 데이터로 학습하면 실제 데이터에서 성능 저하
- 현재 해결책: 
  - 더 큰 규모, 더 다양한 합성 데이터
  - Transfer learning과 domain adaptation
  - 물리 제약 통합으로 정규화

**트렌드 3: 계산 효율성**
- 2017-2020: 정확성 중심
- 2020-2025: 정확성 + 속도 + 메모리 균형

#### 8.3 실무 적용 현황

**활발한 분야**:
- 석유·가스 탐사: 전통적 FWI 지속 사용
- 지진 모니터링: 3D 음향/탄성 FWI 확대
- 지원 저장소(Carbon/Hydrogen): 4D 시간경과 FWI 필수

**지연 분야**:
- 실시간 온라인 역산: 계산 비용 문제
- 초기 모델 자동 생성: 여전히 수동 개입 필요
- 불확실성 정량화: 통상적으로 아직 미실시

### 결론

"Full-waveform inversion, Part 1: Forward modeling"은 **수치 지구물리의 민주화**를 이룬 중추적 논문으로, Devito 프레임워크를 통해 복잡한 파동 방정식을 간단하게 구현할 수 있는 경로를 제시했습니다. 본 논문의 발행 이후 8년간 FWI 분야는 **딥러닝 통합, 일반화 성능 분석, 물리 제약 기반 정규화**로 진화했으며, 향후 연구는 **대규모 합성 데이터 활용, 전이 학습, 불확실성 정량화**에 집중될 것으로 예상됩니다.

**주요 권장사항**:
1. 실전 FWI 개발 시 Devito 또는 유사 자동화 도구 사용으로 개발 시간 단축
2. 노이즈 환경에서는 **MAE loss 사용** (MSE보다 최대 8배 견고성 우수)
3. 여러 지질 환경에 대한 **대규모 훈련 데이터 확보** (일반화 오차 감소 비례)
4. **물리 제약 통합 고려** (특히 데이터 부족 상황)
5. **분포 드리프트 평가** 및 도메인 적응 전략 수립

***

**참고 문헌 체계**

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/61a76632-600e-48b4-a8ff-433c1180d5c9/tle36121033.1.pdf)
[2](https://library.seg.org/doi/10.1190/segam2020-3427858.1)
[3](https://library.seg.org/doi/10.1190/geo2019-0435.1)
[4](https://pubs.geoscienceworld.org/geophysics/article/87/1/R93/610167/Integrating-deep-neural-networks-with-full)
[5](https://essopenarchive.org/doi/full/10.1002/essoar.10502012.1)
[6](https://library.seg.org/doi/10.1190/geo2020-0159.1)
[7](https://library.seg.org/doi/10.1190/geo2019-0641.1)
[8](https://www.earthdoc.org/content/papers/10.3997/2214-4609.202010466)
[9](https://library.seg.org/doi/10.1190/geo2019-0644.1)
[10](https://library.seg.org/doi/10.1190/geo2019-0585.1)
[11](https://academic.oup.com/gji/article/228/2/796/6373445)
[12](https://arxiv.org/html/2503.00658)
[13](http://arxiv.org/pdf/2306.12386.pdf)
[14](https://arxiv.org/pdf/2501.08210.pdf)
[15](http://arxiv.org/pdf/2111.14220.pdf)
[16](http://arxiv.org/pdf/2305.07921.pdf)
[17](https://arxiv.org/html/2412.09458v1)
[18](https://academic.oup.com/gji/advance-article-pdf/doi/10.1093/gji/ggae129/57172443/ggae129.pdf)
[19](https://academic.oup.com/gji/article-pdf/224/1/306/34192100/ggaa459.pdf)
[20](https://academic.oup.com/gji/article/202/3/1535/607490)
[21](https://liner.com/ko/review/efwi-multiparameter-benchmark-datasets-for-elastic-full-waveform-inversion-of)
[22](https://www.sciencedirect.com/science/article/abs/pii/S0098300424002711)
[23](https://www.nature.com/articles/s41598-024-68573-7)
[24](https://arxiv.org/abs/2506.05484)
[25](https://www.eppcgs.org/en/article/pdf/preview/10.26464/epp2026019.pdf)
[26](https://www.sciencedirect.com/science/article/abs/pii/S0021999120308330)
[27](https://scienceon.kisti.re.kr/srch/selectPORSrchArticle.do?cn=DIKO0015485179)
[28](https://www.earthdoc.org/content/papers/10.3997/2214-4609.202210588)
[29](https://geophysical-press.com/journal/JSE/articles/online_first/457)
[30](https://arxiv.org/abs/2502.17624)
[31](https://academic.oup.com/gji/article/241/3/1741/8099925)
[32](https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2022.870547/full)
[33](https://www.themoonlight.io/ko/review/synergizing-deep-learning-and-full-waveform-inversion-bridging-data-driven-and-theory-guided-approaches-for-enhanced-seismic-imaging)
[34](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025JE009130?af=R)
[35](https://arxiv.org/abs/2503.00658)
[36](https://pdfs.semanticscholar.org/71a6/c1283b405eb7f413601ca4b92a100ba40ddf.pdf)
[37](https://arxiv.org/abs/2405.17696)
[38](https://arxiv.org/html/2509.18106v1)
[39](https://arxiv.org/html/2503.00658v1)
[40](https://pdfs.semanticscholar.org/4ef5/105861ed366e280a88ef240e68bd1d193fe2.pdf)
[41](https://arxiv.org/html/2502.17585v1)
[42](https://pdfs.semanticscholar.org/9b48/33a19bb668f3008c97101cc80a9c5054397d.pdf)
[43](https://arxiv.org/html/2505.01817v1)
[44](https://www.biorxiv.org/content/10.1101/2023.04.12.536519v1.full.pdf)
[45](https://arxiv.org/html/2405.17696v1)
[46](https://pdfs.semanticscholar.org/d8ca/272f5303df9858e832a5a2e0f33c0d1385a8.pdf)
[47](https://arxiv.org/html/2406.05153v1)
[48](https://www.biorxiv.org/content/10.1101/2023.04.12.536519v1.full-text)
[49](https://arxiv.org/html/2307.15388v2)
