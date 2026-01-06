# Soft-DTW: a Differentiable Loss Function for Time-Series

### 1. 논문의 핵심 주장과 주요 기여

**핵심 주장:** 시계열 비교를 위한 동적시간와핑(Dynamic Time Warping, DTW)은 우수한 기하학적 성질을 가지지만 미분 불가능하다는 근본적 한계를 가집니다. 본 논문은 이 문제를 해결하기 위해 **Soft-DTW**라는 미분 가능한 손실함수를 제안합니다.

**주요 기여:**

- **미분 가능성 증명**: Soft-DTW가 모든 인자에 대해 미분 가능하며, 역전파를 통해 그래디언트를 효율적으로 계산 가능함을 보임
- **계산 효율성**: O(nm) 시간/공간 복잡도로 원본 DTW와 동일한 수준의 효율성 유지
- **실용적 응용**: 시계열 평균(Averaging), 클러스터링(Clustering), 신경망 기반 예측 학습에서 기존 방법들을 능가

***

### 2. 해결하고자 하는 문제 및 제안 방법

#### 2.1 문제 정의

표준 유클리드 거리는 다음과 같은 시계열 특성을 제대로 처리하지 못합니다:

- **길이 변동성**: 서로 다른 길이의 시계열 비교 불가
- **시간 왜곡**: 속도 변화에 민감 (예: 다른 속도로 발음된 음성)
- **비동기성**: 시간축 정렬 문제

DTW는 이러한 문제들을 동적계획법으로 해결하지만 **미분 불가능**하여 신경망 학습에 사용 불가능합니다.

#### 2.2 제안 방법: Soft-DTW

**기본 정의:**

$$\text{dtwγ}(x, y) := \min_γ\{\langle A, \Delta(x, y) \rangle, A \in \mathcal{A}_{n,m}\}$$

여기서 **일반화된 최소 연산자**는:

$$\min_γ\{a_1, \ldots, a_n\} := \begin{cases} \min_{i \leq n} a_i & \text{if } γ = 0 \\ -γ \log \sum_{i=1}^{n} e^{-a_i/γ} & \text{if } γ > 0 \end{cases}$$

이 수식은 **log-sum-exp** 형태로, Gibbs 분포의 평균 정렬 행렬을 나타냅니다:

$$E_γ[A] := \frac{1}{k_γ^{GA}(x,y)} \sum_{A \in \mathcal{A}_{n,m}} e^{-\langle A, \Delta(x,y)/γ \rangle} A$$

**핵심 통찰**: γ > 0일 때, soft-DTW는 **모든 가능한 정렬의 가중 평균**을 고려하며, 이는 완전히 미분 가능합니다.

#### 2.3 알고리즘: Forward & Backward Pass

**Forward Pass (Algorithm 1):**

$$r_{i,j} = δ(x_i, y_j) + \min_γ\{r_{i-1,j-1}, r_{i-1,j}, r_{i,j-1}\}$$

이는 원본 DTW와 동일한 동적계획 구조로, O(nm) 시간/공간 복잡도를 유지합니다.

**Backward Pass (Algorithm 2) - 핵심 혁신:**

그래디언트를 효율적으로 계산하기 위해 연쇄 법칙을 역방향 재귀에 적용:

$$e_{i,j} := \frac{\partial r_{n,m}}{\partial r_{i,j}}$$

여기서:

$$e_{i,j} = e_{i+1,j} \cdot a + e_{i,j+1} \cdot b + e_{i+1,j+1} \cdot c$$

$$\text{with } \quad a = \exp\left(\frac{1}{γ}(r_{i+1,j} - r_{i,j} - δ_{i+1,j})\right)$$

**최종 그래디언트:**

$$∇_x \text{dtwγ}(x,y) = \left(\frac{\partial \Delta(x,y)}{\partial x}\right)^T E$$

**핵심 성과**: 일반적인 Bellman 재귀를 통한 평균 정렬 행렬 계산은 O(n²m²) 복잡도를 가지지만, 역전파 방식은 O(nm)으로 축소됩니다.

***

### 3. 모델 구조 및 응용 분야

#### 3.1 시계열 평균 (Barycenter Computation)

**최적화 문제:**

$$\min_{x \in ℝ^{p×n}} \sum_{i=1}^{N} \frac{λ_i}{m_i} \text{dtwγ}(x, y_i)$$

여기서 $m_i$는 정규화 인자로, 길이에 따른 편향을 제거합니다.

**특성:**
- 비볼록 최적화 문제 (k-means와 유사)
- γ가 크면 볼록화 효과 (smooth하게 지역 최소값 회피)
- L-BFGS 등 2차 최적화 방법 적용 가능

#### 3.2 클러스터링

Lloyd 알고리즘의 일반화:

$$\min_{x_1,\ldots,x_k} \sum_{i=1}^{N} \frac{1}{m_i} \min_{j \in [k]} \text{dtwγ}(x_j, y_i)$$

#### 3.3 신경망 기반 시계열 예측

$$\min_{θ \in Θ} \sum_{i=1}^{N} \text{dtwγ}\left(f_θ(x^i_{1,t}), x^i_{t+1,n}\right)$$

- MLP, RNN, CNN 등 다양한 네트워크와 호환
- 신경망 파라미터를 직접 학습 가능한 end-to-end 구조

***

### 4. 성능 향상 및 한계

#### 4.1 실험 결과 (UCR 데이터셋 기반)

| 태스크 | 성능 지표 | 결과 |
|--------|---------|------|
| **시계열 평균** | DTW 손실 달성 비율 | γ=0.001일 때 97.47% (DBA 대비) |
| **클러스터링** | DTW 손실 달성 비율 | γ=0.001일 때 96.49% (DBA 대비) |
| **분류 정확도** | 최근접 중심 분류기 | 75% 데이터셋에서 DBA 초과 |
| **예측 성능** | DTW 손실 순위 | MLP 학습 시 평균 순위 1.29 (γ=0.001) |

#### 4.2 일반화 성능 분석

**Smoothing의 효과:**

- **작은 γ (0.001-0.01)**: DTW에 매우 가까우며, 지역 최소값에 빠질 위험 증가
- **중간 γ (0.1)**: 평활과 정확도의 균형 달성
- **큰 γ (1.0)**: 과도하게 평활하여 성능 저하

**그래디언트 성질:**

- Soft-DTW의 그래디언트는 2/γ-Lipschitz 연속이므로, 작은 γ에서 그래디언트가 가파르지 않아 최적화 안정성 향상
- 이는 원본 DTW의 비연속적 그래디언트 (정렬 경로 변화 시 급변)와 대조적

**정규화 효과:**

γ를 통한 implicit regularization:
- 작은 γ: 과소 정규화 (underfitting 가능)
- 적절한 γ: 최적의 일반화
- 큰 γ: 과도 정규화 (underfitting)

#### 4.3 주요 한계

| 한계 | 상세 설명 |
|-----|---------|
| **비볼록성** | dtwγ는 여전히 비볼록이므로 전역 최적해 보장 불가 |
| **파라미터 선택** | γ 값의 적절한 선택이 성능에 큰 영향; 데이터셋별 조정 필요 |
| **메모리 비용** | 전체 R 행렬을 저장해야 하므로 매우 긴 시계열에는 부담 |
| **신경망 구현** | 자동미분 프레임워크에서 직접 구현 어려워 커스텀 backward pass 필요 |
| **이동 길이 고정** | 현재 구현에서는 바이센터의 길이를 고정해야 함 |

***

### 5. 2020년 이후 관련 최신 연구 비교 분석

#### 5.1 Transformer 기반 방법들

| 방법 | 발표 | 핵심 기여 | Soft-DTW와의 비교 |
|-----|------|---------|-----------------|
| **Informer** | 2021 | ProbSparse Attention (O(L log L)) | 긴 시계열에 더 효율적; 하지만 DTW 기하학 미포함 |
| **Temporal Fusion Transformer (TFT)** | 2021 | 변수별 임포턴스 + 다중-호라이즌 예측 | 확률론적 예측 제공; 결정론적 거리 학습은 불가 |
| **TimeFormer** | 2024 | 시간 특성 기반 주의 변조 | 시간 인코딩 명시적 사용; Soft-DTW는 암묵적 |

**평가:** Transformer 계열은 **전역 의존성 포착**에 우수하지만, 시간 왜곡에 대한 강건성이 Soft-DTW보다 낮음

#### 5.2 CNN/TCN 기반 방법들

| 방법 | 발표 | 핵심 기여 | Soft-DTW와의 비교 |
|-----|------|---------|-----------------|
| **WaveNet** | 2016 | Dilated Causal Convolution | 계산 병렬화 가능; 하지만 고정 길이 가정 |
| **ConvTimeNet** | 2024 | 계층적 순수 CNN + 변형 가능 패치 | 다중 스케일 특성 추출; 정렬 고려 없음 |
| **TCN with Attention** | 2024 | Self-Attention + TCN | 효율적 장기 의존성 처리; DTW 거리 미사용 |

**평가:** CNN 방법들은 **계산 효율성**에 우수하지만, 시계열 정렬(alignment)의 명시적 모델링이 없음

#### 5.3 하이브리드 및 거리학습 기반

| 방법 | 핵심 아이디어 | Soft-DTW와의 비교 |
|-----|-------------|-----------------|
| **Distance Metric Learning (DML)** | Mahalanobis 거리 학습 | Soft-DTW는 시간 왜곡을 명시적으로 처리; DML은 선형 변환만 가능 |
| **Earth Mover's Distance (EMD)** | 최적 이동 비용 | Soft-DTW와 유사 원리; 하지만 시계열 순서 구조 미활용 |

#### 5.4 최신 트렌드 (2024-2025)

**주요 발견:**
1. **Transformer 한계 인식**: "Are Transformers Effective for Time Series Forecasting?" 논문들이 transformer의 이차 복잡도와 non-stationarity 문제 지적
2. **CNN 부활**: ConvTimeNet, ModernTCN 등 순수 CNN 모델들이 재조명됨
3. **하이브리드 인기**: TCN + Attention, Transformer + CNN 조합이 주류
4. **생성모델 활용**: 시계열 생성 작업에서 Soft-DTW의 역할 증대

**Soft-DTW의 위치:**
- **강점**: 시간 왜곡에 대한 명시적 강건성
- **약점**: 신경망 학습의 계산 오버헤드 (커스텀 backward pass 필요)
- **기회**: 생성모델(VAE, GAN)과의 결합

***

### 6. 일반화 성능 향상 가능성

#### 6.1 평활화 파라미터 γ의 역할

**이론적 근거:**

1. **Convexification 효과:**
   - γ → 0: DTW로 수렴 (비볼록, 급변하는 그래디언트)
   - γ → ∞: 모든 비용의 합으로 수렴 (δ가 볼록이면 convex)
   - 중간 γ: 최적의 경작성(landscape smoothing)

2. **Regularization Effect:**
   $$\text{dtwγ}(x,y) = -γ \log k^{GA}_γ(x,y)$$
   
   γ는 temperature 파라미터로 작용하여, 작은 γ는 가장 좋은 정렬에 집중하고, 큰 γ는 모든 정렬을 균등 고려

#### 6.2 실험 근거 (논문 Table 1, 2)

**바이센터 계산:**
- Random init: γ=0.001에서 97.47% 성공률
- Euclidean init: γ=0.001에서 89.87% 성공률
- 해석: 적절한 초기화와 γ 선택으로 98% 달성 가능

**클러스터링:**
- γ감소에 따른 성능 개선 추세 명확
- γ=0.001: 96.49% (DBA 초과), 84.48% (서브그래디언트 초과)

**분류 정확도:**
- 75% 데이터셋에서 soft-DTW가 DBA 초과
- γ 선택이 클수록 더 균형잡힌 성능

#### 6.3 일반화 향상의 메커니즘

| 메커니즘 | 효과 | 증거 |
|--------|------|------|
| **지역 최소값 회피** | 더 좋은 솔루션 탐색 | Figure 5: γ=0.01, γ=1.0의 시각적 비교 |
| **Implicit Regularization** | 과적합 방지 | Table 3: soft-DTW로 학습 시 DTW 손실 1.29 순위 |
| **Smooth Gradient Flow** | 안정적 최적화 | Algorithm 2: exp 형태 그래디언트로 수치 안정성 |
| **전체 정렬 고려** | 강건한 특징** | Table 3: Euclidean 초기화로 2.12 순위 달성 |

***

### 7. 앞으로의 연구 방향 및 고려사항

#### 7.1 이론적 확장

1. **비볼록성 분석 심화:**
   - 현재: γ의 smoothing 효과 경험적 확인
   - 미래: Lojasiewicz 부등식 등으로 수렴성 보장

2. **정렬 경로의 통계적 해석:**
   - Gibbs 분포 pγ의 엔트로피 분석
   - 정렬의 불확실성 정량화

3. **다변량 시계열 확장:**
   - 현재: 단변량 또는 독립적 처리
   - 미래: 변수 간 의존성 모델링

#### 7.2 실무적 개선 방향

1. **계산 효율성:**
   - GPU 구현 최적화 (현재 CPU 기준)
   - 병렬 화전파 (현재 순차적)
   - Sparse alignment 행렬 활용

2. **신경망 통합:**
   - 주류 프레임워크(PyTorch, TensorFlow) 공식 지원
   - 자동미분과의 완벽한 호환

3. **적응적 γ 선택:**
   - 동적 γ 스케줄링 (데이터 기반)
   - 데이터셋별 최적 γ 자동 탐색

#### 7.3 최신 흐름과의 통합

| 분야 | 가능한 결합 | 예상 효과 |
|-----|----------|---------|
| **기초 모델** | Soft-DTW + Time Foundation Models | 시간 왜곡 강건성 + 대규모 사전학습 |
| **생성 모델** | VAE/Diffusion + Soft-DTW 손실 | 정렬-불변 생성 모델 |
| **분산 학습** | Federated Learning + Soft-DTW | 프라이버시 보존 시계열 분석 |
| **이상탐지** | Contrastive Learning + Soft-DTW | DTW 거리 기반 이상점 검출 |

#### 7.4 연구 시 주요 고려사항

**긍정적 요소:**
- ✓ 시간 왜곡에 대한 원칙적 접근
- ✓ 이론적 근거가 명확한 미분 가능 손실함수
- ✓ 평균/클러스터링 작업에서 검증된 성능

**제약 조건:**
- ⚠ 현대 딥러닝 프레임워크와의 seamless integration 부족
- ⚠ 매우 긴 시계열(>10,000)에 메모리 문제
- ⚠ 하이퍼파라미터(γ) 선택의 데이터 의존성

**추천 사항:**
1. **단기** (1-2년): 기존 CNN/Transformer와의 하이브리드 모델 개발
2. **중기** (2-4년): 분포 기반 시간 왜곡 모델링 (DMTW, Probabilistic DTW와 결합)
3. **장기** (4+ 년): 고차원 멀티모달 시계열로의 확장

***

### 결론

Soft-DTW는 **시계열의 시간 왜곡 불변성**과 **미분 가능한 손실함수**라는 근본적 요구를 동시에 만족시킨 획기적 논문입니다. 2020년 이후 Transformer와 TCN의 부상에도 불구하고, 시간 정렬 문제의 본질적 해결책으로서의 가치는 여전히 유효합니다. 특히 생성 모델과의 결합, 분산 학습, 그리고 explainability 요구가 높아지는 현대 AI 환경에서 Soft-DTW의 원칙적 접근 방식은 더욱 중요해질 것으로 예상됩니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6705e693-274e-475a-8f63-2a3ea447833d/1703.01541v2.pdf)
[2](https://ieeexplore.ieee.org/document/11039451/)
[3](https://www.mdpi.com/1996-1073/18/10/2434)
[4](https://www.semanticscholar.org/paper/6086a214ecc8401e67977a6bfc98afd7510ae5b5)
[5](https://link.springer.com/10.1007/s41870-024-02327-6)
[6](https://novamindpress.org/index.php/JCIET/article/view/5)
[7](https://bmcpublichealth.biomedcentral.com/articles/10.1186/s12889-025-21982-3)
[8](https://ieeexplore.ieee.org/document/11006217/)
[9](https://link.springer.com/10.1007/s43621-025-01733-5)
[10](https://link.springer.com/10.1007/s13042-025-02585-1)
[11](https://www.jmir.org/2025/1/e74423)
[12](https://arxiv.org/pdf/2312.17100.pdf)
[13](https://arxiv.org/pdf/2412.04532.pdf)
[14](https://www.mdpi.com/1424-8220/23/16/7167/pdf?version=1692015651)
[15](https://arxiv.org/pdf/2310.02280.pdf)
[16](https://arxiv.org/pdf/1703.01541.pdf)
[17](http://arxiv.org/pdf/2310.04948.pdf)
[18](https://www.mdpi.com/2078-2489/14/11/598/pdf?version=1699088576)
[19](http://arxiv.org/pdf/1610.04783.pdf)
[20](https://www.geeksforgeeks.org/machine-learning/dynamic-time-warping-dtw-in-time-series/)
[21](https://aclanthology.org/2021.emnlp-main.709/)
[22](https://jmlr.csail.mit.edu/papers/volume10/weinberger09a/weinberger09a.pdf)
[23](https://www.ijcai.org/proceedings/2025/1187.pdf)
[24](https://www.psypost.org/fascinating-new-neuroscience-model-predicts-intelligence-by-mapping-the-brains-internal-clocks/)
[25](https://www.cs.cmu.edu/~liuy/frame_survey_v2.pdf)
[26](https://arxiv.org/html/2503.10198v1)
[27](https://openaccess.thecvf.com/content/CVPR2022/papers/Han_Temporal_Alignment_Networks_for_Long-Term_Video_CVPR_2022_paper.pdf)
[28](https://www.sciencedirect.com/topics/engineering/distance-metric-learning)
[29](https://www.sciencedirect.com/science/article/abs/pii/S0169023X25000904)
[30](https://openreview.net/forum?id=4dwAZRr9L5)
[31](https://arxiv.org/abs/2209.12727)
[32](https://dl.acm.org/doi/10.1145/3533382)
[33](https://arxiv.org/html/2505.14535v1)
[34](https://openreview.net/forum?id=yDlvteYBbF)
[35](https://royalsocietypublishing.org/rsta/article/379/2194/20200209/41189/Time-series-forecasting-with-deep-learning-a)
[36](https://www.sciencedirect.com/science/article/abs/pii/S0031320325015602)
[37](https://www.sciencedirect.com/science/article/abs/pii/S0893608023002356)
[38](https://github.com/ddz16/TSFpaper)
[39](https://proceedings.iclr.cc/paper_files/paper/2025/file/6ceb6c2150bbf46fd75528a6cd6be793-Paper-Conference.pdf)
[40](https://arxiv.org/html/2502.17495v1)
[41](https://arxiv.org/html/2512.22741v1)
[42](https://pubmed.ncbi.nlm.nih.gov/20709641/)
[43](https://arxiv.org/html/2511.18542v1)
[44](https://ar5iv.labs.arxiv.org/html/1812.05944)
[45](https://arxiv.org/html/2503.04150v3)
[46](https://pubmed.ncbi.nlm.nih.gov/16566500/)
[47](https://arxiv.org/html/2507.14475v1)
[48](https://arxiv.org/html/2505.14158v1)
[49](https://arxiv.org/abs/2003.03960)
[50](https://www.biorxiv.org/content/10.1101/2024.12.05.626975v3.full-text)
[51](https://arxiv.org/abs/2003.06777)
[52](https://arxiv.org/abs/2504.13531)
[53](https://arxiv.org/abs/2210.04369)
[54](https://arxiv.org/html/2401.09736)
[55](https://www.biorxiv.org/content/10.1101/2025.05.19.655003v2)
[56](https://arxiv.org/html/2404.00882v1)
[57](https://arxiv.org/html/2507.01966v1)
[58](https://pmc.ncbi.nlm.nih.gov/articles/PMC7302248/)
[59](https://ieeexplore.ieee.org/document/9930675/)
[60](https://ieeexplore.ieee.org/document/10957909/)
[61](https://ieeexplore.ieee.org/document/10928443/)
[62](https://ieeexplore.ieee.org/document/10652124/)
[63](https://ieeexplore.ieee.org/document/10883100/)
[64](https://dl.acm.org/doi/10.1145/3670105.3670180)
[65](https://ieeexplore.ieee.org/document/10502465/)
[66](https://ieeexplore.ieee.org/document/10894990/)
[67](https://ieeexplore.ieee.org/document/10827306/)
[68](https://ieeexplore.ieee.org/document/10555119/)
[69](https://ieeexplore.ieee.org/document/10796195/)
[70](https://arxiv.org/html/2403.01493v1)
[71](https://arxiv.org/pdf/1904.12546.pdf)
[72](https://arxiv.org/pdf/2404.08472.pdf)
[73](https://www.mdpi.com/1424-8220/20/24/7211/pdf)
[74](https://arxiv.org/pdf/2307.14680.pdf)
[75](http://arxiv.org/pdf/2411.17382.pdf)
[76](http://arxiv.org/pdf/2408.09723.pdf)
[77](https://www.mdpi.com/1424-8220/22/3/841/pdf)
[78](https://www.sciencedirect.com/science/article/abs/pii/S1568494622009942)
[79](https://www.geeksforgeeks.org/deep-learning/transformer-for-time-series-forecasting/)
[80](https://www.cs.cmu.edu/~glai1/papers/swavenet.pdf)
[81](https://www.sciencedirect.com/topics/computer-science/temporal-convolutional-network)
[82](https://arxiv.org/abs/2402.05370)
[83](https://wikidocs.net/228900)
[84](https://jecei.sru.ac.ir/article_1477_6ad60d182e78f7d9211db4608fa63757.pdf)
[85](https://deepmind.google/blog/wavenet-a-generative-model-for-raw-audio/)
[86](https://openreview.net/forum?id=O9nZCwdGcG)
[87](https://arxiv.org/abs/2405.16877)
[88](https://velog.io/@crosstar1228/%EC%83%9D%EC%84%B1%EB%AA%A8%EB%8D%B8Wavenet)
[89](https://arxiv.org/abs/2412.17452)
[90](https://velog.io/@leesungjoon-net/Are-Transformers-Effective-for-Time-Series-Forecasting)
[91](https://deep-generative-models-aim5036.github.io/autoregressive%20models/2022/11/13/wavenet.html)
[92](https://dl.acm.org/doi/abs/10.1145/3723890.3723911)
[93](https://openreview.net/forum?id=kHEVCfES4Q&noteId=mrNbq9EkQa)
[94](https://dmqa.korea.ac.kr/activity/seminar/242)
[95](https://openreview.net/forum?id=vpJMJerXHU)
[96](https://openreview.net/forum?id=iN43sJoib7)
[97](https://arxiv.org/html/2411.04669v1)
[98](https://arxiv.org/pdf/1806.06116.pdf)
[99](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0320368)
[100](https://arxiv.org/abs/2504.00068)
[101](https://arxiv.org/pdf/2305.01638.pdf)
[102](https://arxiv.org/abs/2510.06680)
[103](https://arxiv.org/pdf/1609.03499.pdf)
[104](https://arxiv.org/abs/2408.15737)
[105](https://arxiv.org/abs/2502.06151)
[106](https://arxiv.org/html/2510.15947v1)
[107](https://arxiv.org/html/2410.19722v1)
[108](https://arxiv.org/abs/2410.03805)
[109](https://arxiv.org/pdf/1611.09482.pdf)
[110](https://www.semanticscholar.org/paper/413e0ce1a19253de0550c003822b981068822ad2)
[111](https://arxiv.org/abs/2002.04971)
[112](https://arxiv.org/abs/2405.12038)
[113](https://arxiv.org/abs/2410.23749)
[114](https://arxiv.org/abs/2205.13504)
