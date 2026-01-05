# Using Dynamic Time Warping to Find Patterns in Time Series

### 1. 핵심 주장 및 기여 요약

Berndt와 Clifford의 선구적 논문은 **음성 인식 분야의 동적 타임 워핑(DTW) 기법을 시계열 데이터베이스의 패턴 검출 문제로 일반화**한 최초의 연구입니다.[1]

**핵심 주장**:
- 임의 길이의 시계열에서 템플릿 패턴을 찾기 위해서는 시간축의 비선형 왜곡을 허용하는 "fuzzy matching" 알고리즘이 필수적
- 동적 프로그래밍에 기반한 DTW는 이러한 문제를 효율적으로 해결할 수 있음
- 비즈니스 거래, NASA 원격측정, 환자 모니터링 등 증가하는 시계열 데이터에서 가치 있는 패턴을 자동으로 발견

**주요 기여**:
1. **분야 간 기법 이전**: 음성 인식의 DTW를 일반 시계열 패턴 매칭으로 확장
2. **실제 데이터 검증**: 스노우슈 토끼와 스라소니 개체수 데이터에서 "더블 탑" 피크 패턴 성공적 검출 (점수 0.87, 0.86)
3. **개념적 명확화**: 시계열 패턴 검출이 본질적으로 템플릿-시리즈 정렬 최적화 문제임을 입증

***

### 2. 문제 정의, 방법론, 모델 구조

#### 2.1 해결하고자 하는 문제

시계열 데이터 $S = \{s_1, s_2, \ldots, s_n\}$에서 템플릿 $T = \{t_1, t_2, \ldots, t_m\}$의 인스턴스를 찾기:

$$\text{기본 문제: } \min_{\text{warping paths}} \sum_{k=1}^{K} \delta(w_k)$$

여기서 $\delta(i,j)$는 $s_i$와 $t_j$ 간 거리이며, 다음 중 하나:
- $$d(i,j) = |s_i - t_j|$$ (절댓값 거리)
- $$d(i,j) = (s_i - t_j)^2$$ (제곱 거리)

#### 2.2 제안된 방법론: 동적 프로그래밍 기반 DTW

**알고리즘의 핵심 점화식:**

$$\gamma(i,j) = \delta(i,j) + \min[\gamma(i-1,j), \gamma(i-1,j-1), \gamma(i,j-1)]$$

이는 누적 비용 행렬을 구성하며, 각 셀 $(i,j)$는 다음을 의미합니다:
- $s_i$와 $t_j$를 정렬하는 최소 누적 거리

**시간 복잡도**: $O(n \times m)$ 시간, $O(n \times m)$ 공간

#### 2.3 경로 제약 조건 (탐색 공간 축소)

DTW의 조합론적 폭발을 방지하기 위해 5가지 제약 도입:

| 제약 조건 | 설명 | 효과 |
|---------|-----|------|
| **단조성** (Monotonicity) | $i_{s} \leq i_{s+1}$, $j_{s} \leq j_{s+1}$ | 시간 역행 금지 |
| **연속성** (Continuity) | $\|i_s - i_{s-1}\| \leq 1$, $\|j_s - j_{s-1}\| \leq 1$ | 인접 셀로만 이동 |
| **워핑 윈도우** | $\|i_s - j_s\| \leq w$ | 대각선 근처로 제한 |
| **기울기 제약** | 한 방향 과도한 이동 방지 | 병렬 정렬 강제 |
| **경계 조건** | $i_1=1, j_1=1, i_K=n, j_K=m$ | 시작/종료점 고정 |

#### 2.4 적합도 측정 (Measures of Fit)

템플릿과 시계열 세그먼트의 매칭 품질을 0~1 범위로 정규화:

$$\text{Score} = \max\left(0, 1 - \frac{\text{warping path distance}}{\text{baseline distance}}\right)$$

기준선(baseline)은 시계열 세그먼트 주변의 대역폭으로 설정:
- 예: 데이터 값이 50이면, 데이터폭 100%에서  범위 설정[2][3]

***

### 3. 성능 향상 및 한계

#### 3.1 실험 결과

**산형(Mountain-shaped) 패턴 테스트:**

| 템플릿/시리즈 | flat40 | mnt5  | mnt10 | mnt20 |
|-------------|--------|-------|-------|-------|
| **flat40**  | 1.00   | 0.86  | 0.76  | 0.61  |
| **mnt5**    | 0.84   | 1.00  | 0.91  | 0.73  |
| **mnt10**   | 0.68   | 0.89  | 1.00  | 0.85  |
| **mnt20**   | 0.36   | 0.62  | 0.81  | 1.00  |

→ 더 유사한 패턴일수록 높은 점수 달성 (대각선 근처)

**스노우슈 토끼 데이터 분석 (더블 탑 피크):**
- **최고 점수 위치**: 1849년 (점수 0.87), 1870년 (점수 0.86)
- **낮은 점수**: 대부분 0.6 이하
- **결과 해석**: 패턴 복잡도와 템플릿 단순성이 검출 난이도 증가

#### 3.2 성능 향상 가능성

**원본 논문의 한계:**
1. **템플릿 설계**: 수동 지정 필요 → 자동화 어려움
2. **파라미터 민감도**: 워핑 윈도우, 기울기 제약의 선택이 결과에 큰 영향
3. **확장성**: 장문 시계열에서 $O(n^2)$ 복잡도는 병목

**본질적 한계:**
- **평활성 부족**: 노이즈가 많은 데이터에서 허위 양성 증가
- **메트릭 특성**: 삼각부등식 미충족으로 군집화 어려움
- **해석성**: 고차원 특성에서 패턴 의미 파악 곤란

***

### 4. 모델의 일반화 성능 향상 가능성 (중점)

#### 4.1 모델의 일반화 문제

Berndt & Clifford의 원본 접근법은 다음 측면에서 **일반화 성능 제약**:

1. **고정 템플릿**: 학습되지 않음 → 새로운 데이터 도메인에 이전(transfer) 불가
2. **거리 함수 선택**: $L_1$ vs $L_2$ vs 다른 메트릭 간 민감도
3. **휴리스틱 경로 제약**: 데이터 특성에 따라 최적이 아닐 수 있음

#### 4.2 2020년 이후 일반화 성능 향상 기법들

**① 학습 가능 프로토타입 (DP-DTW, Chang et al. 2021)**[4]

$$\text{min}_{\{p_1,...,p_K\}} \sum_n \text{DTW}(s_n, p_{y_n}) + \lambda \sum_{k \neq y_n} \max(0, \text{DTW}(s_n, p_k) - \text{Margin})$$

**일반화 개선 메커니즘:**
- 클래스 특화 프로토타입 학습 → 도메인 적응성 향상
- 128개 UCR 시계열 데이터셋에서 82.8% 우수 성능
- 평균 순위: 1.6 (vs 1-NN+DTW의 ~2.5)

**② 미분 가능 DTW (DecDTW, Xu et al. 2023)**[5]

엔드-투-엔드 신경망 학습 가능:

$$\text{min}_{\varphi} \hat{f}(x, y, \lambda, \varphi) = \hat{L}(x, y, \varphi) + \lambda\hat{R}(\varphi)$$

제약조건:

$$s^{\min}_i \leq \frac{\varphi_{i+1} - \varphi_i}{\Delta t_i} \leq s^{\max}_i, \quad b^{\min}_i \leq \varphi_i \leq b^{\max}_i$$

**일반화 개선:**
- 음악 정렬: TimeErr 122ms → 16ms (7.6배 개선)
- Visual Place Recognition: @2m 정확도 0.7% → 25.4%
- **핵심**: 훈련-추론 일관성 (Soft-DTW의 불일치 제거)

**③ 주의(Attention) 기반 시간 워핑 (2023)**[6]

이중 주의 모델로 작업별 적응형 워핑:
- 시간 왜곡에 대한 강건성과 판별 능력의 균형 개선
- 신경망의 가중치 매개변수를 통해 학습 가능

**④ 희소 DTW (TimePoint, 2024)**[7]

CPAB 변환을 사용한 자기 지도학습:
- 학습된 핵심점(keypoint)과 설명자(descriptor)로 매칭
- 표준 DTW 대비 15-30배 속도 개선
- 실제 데이터로의 강한 일반화

#### 4.3 일반화 성능 비교 요약

| 방법 | 적응성 | 확장성 | 노이즈 강건성 | 해석성 |
|------|--------|--------|-------------|--------|
| **원본 DTW (1994)** | ✗ | O(n²) | 약함 | 중상 |
| **DP-DTW (2021)** | ✓ 학습 프로토 | O(n²) | 중간 | 우수 |
| **DecDTW (2023)** | ✓ 엔드투엔드 | O(nM²) (느림) | 중간 | 우수 |
| **Deep Attentive (2023)** | ✓ 태스크 적응 | O(n²) | 좋음 | 중간 |
| **TimePoint (2024)** | ✓ 자기 지도 | O(n log n) | 우수 | 낮음 |

***

### 5. 향후 연구 영향 및 고려사항

#### 5.1 원본 논문의 지속적 영향

Berndt & Clifford (1994)는 다음을 제시했습니다:

1. **개념적 기초**: 패턴 검출 = 정렬 최적화
   - 이후 30년간 모든 DTW 연구의 기반
   
2. **알고리즘적 우아함**: $O(n \times m)$ DP 해법
   - 여전히 핵심 계산 엔진 (모든 변형에서 사용)

3. **현실성**: 실제 데이터(스노우슈 토끼)에서 의미 있는 패턴 발견
   - 학계뿐 아니라 산업 응용 가능성 입증

#### 5.2 현대 연구의 주요 과제 및 해결 방향

**① 계산 효율성 문제**

| 해결 방법 | 기법 | 효과 |
|---------|------|------|
| 제약된 윈도우 | Sakoe-Chiba band | 30-50% 속도 개선 |
| 계층적 DTW | Fast DTW | 10-50x 가속 |
| 핵심점 기반 | TimePoint | 15-30x 가속 |
| GPU 병렬화 | 현재 개발 중 | 추정 100x 가속 |

**② 일반화 성능 개선**

현대 방법들의 공통 전략:
```
Data → Feature Extraction → Learnable Prototypes/Attention → DTW → Classification
                    ↑                                                    ↑
                자동 학습됨                              미분 가능하게 학습됨
```

**③ 이론적 기반 강화**

- **DTWNet (2019)**: DTW 손실함수가 구간별 이차식/일차식임을 증명
- **수렴 분석**: 지역 최솟값 탈출의 필요 스텝 크기 명시
- **일반화 경계**: 아직 미개척 (향후 중요 연구 방향)

#### 5.3 현재(2020-2025) 주요 응용 분야

| 분야 | 응용 사례 | 개선 사항 |
|------|---------|---------|
| **비디오 분석** | 액션 분할 (DP-DTW) | 준감독 설정에서 SOTA |
| **음악 정보 검색** | 음악-악보 정렬 (DecDTW) | TimeErr 7배 개선 |
| **로봇 공학** | 시각적 장소 인식 | 위치 정확도 35배 향상 |
| **원격 탐사** | 토지 피복 변화 감지 | 정확도 77.89% |
| **의료** | ECG 기반 질환 진단 | 자동화 및 신뢰성 증대 |
| **재무** | 주가 패턴 분석 | 음성 패턴 인식 강화 |
| **다중모달** | 음성-텍스트 정렬 (2025) | 정렬 정확도 19% 향상, 33배 빠름 |

#### 5.4 향후 연구 시 고려사항

**1. 이론적 공백**
   - DTW 기반 분류기의 일반화 오차에 대한 경계 (표본 복잡도 분석 부재)
   - 트랜스포머 + DTW 결합의 최적성 조건 미확립

**2. 계산-정확도 트레이드오프**
   ```
   빠른 방법 (TimePoint)     → 희소 표현 손실
   정확한 방법 (DecDTW)      → 15-50배 느림
   → 도메인 요구에 맞는 선택 필수
   ```

**3. 하이퍼파라미터 자동화**
   - 워핑 윈도우 너비 선택
   - 정규화 강도 λ 결정
   - 거리 함수 선택 (현재: 수동)

**4. 확장성 한계**
   - 매우 긴 시계열 (>100k 타임스텝): 여전히 문제
   - 고차원 다변량 시계열: 다중 DTW 확장의 효율성 의문

**5. 도메인 특화**
   - 각 응용 분야마다 최적 DTW 변형이 상이
   - 메타러닝 기반 자동 선택 연구 시작 단계

***

### 6. 결론 및 종합 평가

#### 6.1 Berndt & Clifford (1994)의 역사적 중요성

1994년의 논문은:
- **혁신**: 음성 인식 기법의 시계열 데이터 마이닝으로의 성공적 이전
- **우아함**: 단순하면서도 강력한 동적 프로그래밍 공식화
- **영향력**: 이후 30년간 3,000+ 인용, 산업 표준 기술로 정착

#### 6.2 현대적 평가

**강점:**
- 개념적 명확성과 이해의 용이성
- 적응 가능한 기본 알고리즘 구조
- 시간-공간 트레이드오프의 합리적 균형

**약점:**
- 학습 불가능한 고정 템플릿
- 이론적 일반화 경계 부재
- 높은 계산 비용 (O(n²))

#### 6.3 미래 방향

**5년 전망 (2026-2030):**

1. **신경 기호 통합**: DTW와 심층신경망의 완전 통합
   - 물리 기반 제약 + 학습된 특성 추출

2. **다중모달 확장**: 시계열 + 이미지 + 텍스트의 통합 정렬
   - 비전 트랜스포머와의 시너지

3. **이론적 진전**: 일반화 경계, 복잡도 이론의 발전
   - 샘플 효율성 개선

4. **응용 심화**: 자율주행, 생명과학, 금융에서의 산업 배포 가속

**최종 평가**: Berndt & Clifford의 DTW는 **시계열 분석의 "Rosetta Stone"** 역할을 하며, 원본의 단순성 덕분에 30년이 지난 오늘날에도 현대 기법의 핵심을 이루고 있습니다. 향후 연구는 이 기초 위에서 확장성, 학습성, 이론적 깊이를 더하는 방향으로 진행될 것으로 예상됩니다.

***

### 참고문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b16956ab-6fbc-48ef-aff9-e48f2a65ff93/WS94-03-031.pdf)
[2](https://www.geeksforgeeks.org/machine-learning/similarity-search-for-time-series-data/)
[3](https://www.sciencedirect.com/science/article/abs/pii/S0031320324006526)
[4](https://ieeexplore.ieee.org/document/9290106/)
[5](https://linkinghub.elsevier.com/retrieve/pii/S0020025520303054)
[6](https://ieeexplore.ieee.org/document/9182494/)
[7](https://onlinelibrary.wiley.com/doi/10.1002/ldr.3502)
[8](https://link.springer.com/10.1007/s10844-020-00609-6)
[9](https://www.tandfonline.com/doi/full/10.1080/07038992.2020.1740083)
[10](https://www.mdpi.com/2073-4441/12/9/2411)
[11](https://linkinghub.elsevier.com/retrieve/pii/S0031320320305021)
[12](https://ieeexplore.ieee.org/document/9308409/)
[13](http://link.springer.com/10.1007/s10472-019-09682-2)
[14](https://arxiv.org/pdf/2211.00005.pdf)
[15](https://arxiv.org/abs/2109.00978)
[16](https://arxiv.org/pdf/2111.13314.pdf)
[17](http://arxiv.org/pdf/2303.10778.pdf)
[18](https://digital.csic.es/bitstream/10261/240794/1/Herbert_et_al_2020_poster.pdf)
[19](https://arxiv.org/pdf/1505.06531.pdf)
[20](https://arxiv.org/pdf/2310.02280.pdf)
[21](https://arxiv.org/html/2402.08943v1)
[22](https://www.geeksforgeeks.org/machine-learning/dynamic-time-warping-dtw-in-time-series/)
[23](https://onlinelibrary.wiley.com/doi/10.1155/2018/2404089)
[24](https://hrcak.srce.hr/file/351984)
[25](https://pmc.ncbi.nlm.nih.gov/articles/PMC12538969/)
[26](http://papers.neurips.cc/paper/9338-dtwnet-a-dynamic-time-warping-network.pdf)
[27](https://www.frontiersin.org/journals/astronomy-and-space-sciences/articles/10.3389/fspas.2020.00039/full)
[28](https://openaccess.thecvf.com/content/CVPR2021/papers/Chang_Learning_Discriminative_Prototypes_With_Dynamic_Time_Warping_CVPR_2021_paper.pdf)
[29](https://arxiv.org/html/2507.22189v1)
[30](https://www.sciencedirect.com/science/article/abs/pii/S0169023X25000904)
[31](https://arxiv.org/abs/2309.06720)
[32](https://www.sciencedirect.com/science/article/abs/pii/S0950705121010108)
[33](https://ai.meta.com/research/publications/aligning-time-series-on-incomparable-spaces/)
[34](https://www.sciencedirect.com/science/article/abs/pii/S0950705119303995)
[35](https://dl.acm.org/doi/10.1145/1247480.1247544)
[36](https://unit8co.github.io/darts/examples/12-Dynamic-Time-Warping-example.html)
[37](https://ieeexplore.ieee.org/document/7814074/)
[38](https://ieeexplore.ieee.org/document/5477140/)
[39](https://arxiv.org/html/2502.16324v1)
[40](https://koreascience.kr/article/JAKO202514254005926.pdf)
[41](https://arxiv.org/html/2507.16406v1)
[42](https://arxiv.org/html/2310.12399v2)
[43](https://arxiv.org/html/2502.17495v1)
[44](https://arxiv.org/html/2510.21824v1)
[45](https://arxiv.org/html/2501.05750v3)
[46](https://arxiv.org/abs/2507.09826)
[47](https://arxiv.org/html/2411.10418v1)
[48](https://arxiv.org/html/2504.06328v1)
[49](https://arxiv.org/pdf/2507.09826.pdf)
[50](https://arxiv.org/html/2509.20184v1)
[51](https://arxiv.org/pdf/2509.06365.pdf)
[52](https://arxiv.org/abs/2004.08780)
[53](https://arxiv.org/html/2210.00379v7)
[54](https://arxiv.org/html/2511.08134v1)
[55](https://arxiv.org/html/2405.06234v1)
[56](https://arxiv.org/pdf/2511.13936.pdf)
[57](https://www.nature.com/articles/s41598-025-94782-9)
[58](https://ieeexplore.ieee.org/document/9577531/)
[59](https://ojs.aaai.org/index.php/AAAI/article/view/17008)
[60](https://www.temjournal.com/content/144/TEMJournalNovember2025_2895_2910.html)
[61](https://www.semanticscholar.org/paper/1f97f79b6720215bfafdafd83f3cc9074f34c11c)
[62](http://arxiv.org/pdf/2103.09458.pdf)
[63](https://arxiv.org/pdf/1703.01541.pdf)
[64](http://arxiv.org/pdf/2309.06720.pdf)
[65](https://arxiv.org/pdf/2108.06816.pdf)
[66](https://arxiv.org/pdf/2206.02956.pdf)
[67](https://openreview.net/pdf/7bbf2eafca1142aa562090e29a3895f75a0c04ae.pdf)
[68](https://www2.cs.sfu.ca/~mori/research/papers/chang-cvpr21.pdf)
[69](https://aclanthology.org/2025.wmt-1.11.pdf)
[70](https://arxiv.org/pdf/1512.06747.pdf)
[71](https://arxiv.org/abs/2103.09458)
[72](https://dl.acm.org/doi/10.1109/TPAMI.2025.3534202)
[73](https://github.com/BorealisAI/TSC-Disc-Proto)
[74](https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2025.1731410/full)
[75](https://khu-mcl.github.io/research/lab_meeting/8/)
[76](https://www.shadecoder.com/zh/topics/dynamic-time-warping-a-comprehensive-guide-for-2025)
[77](https://dl.acm.org/doi/full/10.1145/3702468.3702479)
[78](https://openreview.net/pdf?id=UClBPxIZqnY)
[79](https://arxiv.org/abs/2508.16366)
[80](https://www.semanticscholar.org/paper/Generalizing-DTW-to-the-multi-dimensional-case-an-Shokoohi-Yekta-Hu/594598cc54267cc34d6eb6445430fd92672138a9)
[81](https://kdd-milets.github.io/milets2022/papers/MILETS_2022_paper_5118.pdf)
[82](https://github.com/cmhungsteve/Awesome-Transformer-Attention)
[83](https://www.arxiv.org/pdf/2505.18799v2.pdf)
[84](https://www.arxiv.org/pdf/2510.02729.pdf)
[85](https://www.arxiv.org/pdf/2509.12024.pdf)
[86](https://arxiv.org/html/2510.09048v1)
[87](https://peerj.com/articles/cs-3097/)
[88](https://arxiv.org/html/2511.01233v1)
[89](https://arxiv.org/html/2501.16656v1)
[90](https://arxiv.org/html/2412.20582v1)
[91](https://arxiv.org/pdf/2510.03232v1.pdf)
[92](https://www.arxiv.org/pdf/2506.02515v2.pdf)
[93](https://arxiv.org/pdf/2407.11480.pdf)
[94](https://arxiv.org/html/2501.13104v1)
[95](https://arxiv.org/html/2506.11169)
[96](https://arxiv.org/html/2412.12829v1)
[97](https://arxiv.org/pdf/2508.18850.pdf)
[98](https://arxiv.org/html/2503.03262v1)
[99](https://arxiv.org/html/2406.04419v2)
[100](https://arxiv.org/html/2508.18850v1)
