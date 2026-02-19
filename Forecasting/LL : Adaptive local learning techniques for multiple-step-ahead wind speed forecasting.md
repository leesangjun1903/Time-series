# Adaptive local learning techniques for multiple-step-ahead wind speed forecasting

이 논문은 물리 기반(white-box), 데이터 기반(black-box), 두 가지를 결합한 grey-box 모델을 체계적으로 비교하며, Lazy Learning 기반의 적응형 국소 학습(Adaptive Local Learning)을 풍속 예측에 적용한 원조 연구 중 하나입니다. 아래에 네 가지 요청 사항을 모두 상세히 다룹니다.

***

## 1. 핵심 주장 및 주요 기여 요약

이 논문(Vaccaro et al., Electric Power Systems Research, 2012)은 풍속의 단기·중기 다단계 예측(Multiple-step-ahead Forecasting)에서 **세 가지 모델 패러다임**을 비교하고, 물리 모델의 한계를 보완하는 **새로운 grey-box 아키텍처**를 제안합니다.[^1_1]

**주요 기여 두 가지:**

- 첫 번째: NWP(수치기상예측, white-box)와 Lazy Learning(black-box) 기반 여러 전략(Iterated, Direct, MIMO)의 정량적 비교 실험 수행[^1_1]
- 두 번째: NWP 예측 오차를 Lazy Learning으로 적응적으로 보정하는 **grey-box 하이브리드 구조** 신규 제안 및 검증[^1_1]

***

## 2. 문제 정의·제안 방법·모델 구조·성능·한계

### 해결하고자 하는 문제

풍력 발전기는 출력이 본질적으로 간헐적이어서, 전력망 운영자(DSO/TSO)는 수 분~24시간 앞의 정확한 풍속 예측을 필요로 합니다.  기존 NWP 모델은 16km×16km 격자 단위의 낮은 공간 해상도와 막대한 계산 자원을 요구하고, ARIMA 기반 모델은 24시간 이상 예측 시 비정상성(non-stationarity)으로 성능이 급격히 저하됩니다.[^1_1]

***

### 제안하는 방법 및 수식

**① 기본 시계열 예측 모델 (NAR)**

시계열 출력 $y(t)$를 과거 입력 벡터 $\phi(t)$로 표현합니다:

$y(t) = f(\phi(t)) + v(t) \tag{1}$

여기서 $v(t)$는 모델링 오차·노이즈이며, 입력 벡터는:

$\phi(t) = [y(t-h),\; y(t-h-1),\; \dots,\; y(t-h-n+1)]^T \tag{2}$

풍속 예측에 특화하면:

$w(t+h) = f(w(t),\; w(t-1),\; \dots,\; w(t-n)) \tag{3}$

***

**② 다단계 예측 전략 세 가지**


| 전략 | 모델 수 | 수식 | 장점 | 단점 |
| :-- | :-- | :-- | :-- | :-- |
| **Iterated** | 1개 (반복) | $w(t+1)=f(w(t),\dots,w(t-n))+v(t+1)$, T회 반복 | 구현 단순 | 오차 누적 |
| **Direct** | T개 (horizon별) | $w(t+h)=f_h(w(t),\dots,w(t-n))+v(t+h),\; h\in\{1,\dots,T\}$ | 오차 비누적 | 예측값 간 의존성 무시 |
| **MIMO** | 1개 (다출력) | $[w(t+T),\dots,w(t+1)]=F(w(t),\dots,w(t-n))+V$ | 예측값 간 의존성 보존 | 구조 복잡 |

[^1_1]

***

**③ Lazy Learning (LL) 알고리즘**

쿼리 포인트 $\phi_q$에 대한 $k$-NN 기반 예측:

$\hat{y}^{(k)}\_q = \frac{1}{k}\sum_{i=1}^{k} y_{[i]} \tag{4}$

최적 이웃 수 $k^*$는 **Leave-One-Out(LOO) 오차**를 최소화하여 결정합니다:

$e_{LOO}(k) = \frac{1}{k}\sum_{j=1}^{k} [e_j(k)]^2, \quad e_j(k) = \frac{k \cdot y_{[j]} - \hat{y}^k}{k-1} \tag{5}$

$k^* = \arg\min_{k \in [2,\dots,K]}\; e_{LOO}(k) \tag{6}$

[^1_1]

***

**④ 성능 평가 지표**

$MAE = \frac{\sum_{i=1}^{N} |w(i) - w_f(i)|}{N}, \quad RMSE = \sqrt{\frac{\sum_{i=1}^{N}(w(i) - w_f(i))^2}{N}} \tag{7}$

[^1_1]

***

### 모델 구조

- **White-box**: ECMWF NWP 모델 (91개 수직 레이어, 25km 수평 해상도, 물리 방정식 기반)[^1_1]
- **Black-box**: LL-Iterated / LL-Direct / LL-MIMO (입력 크기 $n=2$, mRMR 특징 선택 적용)[^1_1]
- **Grey-box**: NWP 예측값 $w_f(t+k)$를 LL 보정기가 오차를 학습하여 수정하는 적응형 구조 (그림 1). 보정 트리거 임계값: 2 m/s[^1_1]

Grey-box 구조의 핵심은 NWP의 잔차(residual) $\varepsilon(t+k) = w(t+k) - w_f(t+k)$를 LL로 학습하고, 신규 측정치가 임계값을 초과하면 데이터셋을 갱신합니다.[^1_1]

***

### 성능 결과

실험은 이탈리아 남부 기상 관측소의 18개월치 시간별 풍속 데이터, ENERCON E-40 500kW 발전기 20기로 수행했습니다.[^1_1]

- **Grey-box 모델**이 단기 및 중기 모든 예측 구간에서 최우수 성능 달성[^1_1]
- **LL-MIMO**는 black-box 중 최고 성능 (특히 중기 예측에서 직접법과 반복법의 장점 결합)[^1_1]
- **NWP(white-box)**은 단기에서 매우 낮은 성능이지만, 중기(12~24h)에서 경쟁력 있음[^1_1]
- LL black-box 모델은 구현 비용이 낮으면서도 매우 단기 예측에서 실용적 수준의 정확도 달성[^1_1]

***

### 한계

- **단일 지역 데이터**: 이탈리아 남부 한 지점만 사용 → 지역 특이성(topographic effect) 일반화 불확실[^1_1]
- **단변량 입력**: 기온·기압·풍향 등 외생 변수(exogenous variable) 미활용, 순수 NAR 모델[^1_1]
- **하이퍼파라미터 경험적 설정**: $n=2$, $g=3$, 임계값 2 m/s 등을 공학적 직관에 의존[^1_1]
- **공간 상관성 미반영**: 인근 관측소 데이터를 활용한 다지점 공간 시계열 분석 미포함[^1_1]
- **18개월 데이터 한계**: 계절적 다양성 포착에 제한적[^1_1]

***

## 3. 모델 일반화 성능 향상 가능성

### LL의 적응적 일반화 메커니즘

Lazy Learning은 **훈련 데이터베이스 자체를 업데이트**하는 것만으로 모델이 갱신되므로, 재학습(retraining) 없이 새로운 운전 조건에 적응할 수 있습니다.  이는 풍속의 비정상성(non-stationarity)과 계절별 변동성에 자연스럽게 대응하는 특성입니다.[^1_1]

### Grey-box 구조의 일반화 기여

Grey-box 아키텍처는 물리 모델이 제공하는 **전역적 추세(global trend)**와 LL이 포착하는 **국소적 잔차 패턴(local residual pattern)**을 결합하여, 학습 데이터 범위 밖의 기상 조건에서도 NWP가 안정적인 사전 추정(prior estimate)으로 기능합니다.  또한 재귀 피드백 구조를 회피하므로 반복 신경망에서 발생하는 오차 폭발(error explosion) 문제가 없어 안정적 일반화가 가능합니다.[^1_1]

### 일반화 향상을 위한 잠재적 방향 (논문 명시)

논문은 다수의 공간 시계열 및 관측 변수 수 증가에 따른 스케일 업 실험을 향후 과제로 제시하며, 이는 일반화 성능의 직접적 검증을 위한 핵심 미해결 과제입니다.[^1_1]

***

## 4. 미래 연구에의 영향 및 고려사항

### 이 논문이 미치는 영향

- **Grey-box 하이브리드 패러다임 정립**: NWP + ML 결합이 단순한 앙상블을 넘어 물리 모델의 오차를 학습하는 구조로 발전하는 원형을 제시함[^1_1]
- **MIMO 전략의 중요성 입증**: 다출력 구조가 중장기 예측에서 오차 누적 없이 예측값 간 의존성을 보존하는 우수성을 실증함[^1_1]
- **Lazy/Local Learning의 풍력 예측 적용 가능성 확립**: 이후 연구들이 kNN·국소 회귀를 DL 구조의 attention 메커니즘으로 발전시키는 방향에 영향을 줌


### 향후 연구 시 고려할 점

| 고려 영역 | 구체적 과제 |
| :-- | :-- |
| **데이터 확장** | 다중 지점·다변량 입력 (풍향, 기압, 습도 등 외생 변수 통합) [^1_1] |
| **공간 모델링** | 인근 관측소 공간 상관성 활용 (GNN 적용 가능) [^1_2] |
| **불확실성 정량화** | 점 예측을 넘어 확률적(probabilistic) 구간 예측 제공 |
| **딥러닝 대체** | LL의 kNN 유사도 비교 개념을 Attention Mechanism으로 대체·확장 [^1_3] |
| **임계값 자동화** | 현재 경험적으로 설정된 보정 트리거 임계값의 적응적 최적화 [^1_1] |
| **전이 학습** | 새로운 지역 데이터 부족 시 기학습 모델의 도메인 적응 [^1_4] |


***

## 2020년 이후 관련 최신 연구 비교 분석

이 논문이 제안한 핵심 아이디어들이 2020년대 연구에서 어떻게 발전·대체되었는지를 아래 표에 정리합니다.


| 연구 (연도) | 핵심 방법 | 본 논문 대비 진화 포인트 |
| :-- | :-- | :-- |
| **VMD-GNN-TCN** (2025) [^1_2] | 변분 모드 분해 + 그래프 신경망 + TCN | LL의 국소 유사도 비교 → GNN 기반 변수 간 쌍별(pairwise) 의존성 그래프 학습 |
| **STC-DPN (ConvLSTM+NWP)** (2023) [^1_5] | ConvLSTM + NWP 격자 데이터 융합 | Grey-box 정신 계승, NWP 공간 격자를 CNN으로 처리 → 공간 해상도 문제 해결 |
| **VMD-CNN-GRU-BiLSTM** (2024) [^1_6] | VMD + MR-IG 특징 선택 + 하이브리드 DL | MIMO 전략 + 신호 분해 + mRMR 대신 MR-IG 특징 선택 고도화 |
| **Physics-LSTM** (2025) [^1_7] | 공기역학 파라미터 + Bayesian 최적화 LSTM | Grey-box 개념에서 물리 파라미터를 딥러닝 입력 특징으로 내재화 |
| **WindDragon (AutoML+NWP)** (2024) [^1_8] | NWP 풍속 맵 + 자동화 딥러닝 | NWP를 보정하는 것이 아닌, NWP 공간 맵을 원시 입력으로 활용 |
| **ELM-MIMO** (2025) [^1_9] | Extreme Learning Machine + MIMO | 본 논문 LL-MIMO의 정신을 ELM으로 계승, 닫힌 형태(closed-form) 해법으로 실시간 적용 |
| **CEEMDAN-Bi-GRU+전이 학습** (2025) [^1_4] | 분해 + 양방향 GRU + 전이 학습 | LL의 적응성을 전이 학습으로 대체, 새 지역 일반화 강화 |

### 종합 흐름

2012년 본 논문이 제안한 핵심 원칙들—①물리 모델과 데이터 기반 모델의 결합(grey-box), ②MIMO 다출력 전략, ③적응형 온라인 갱신—은 2020년대 연구에도 여전히 유효한 설계 원칙으로 계승되고 있습니다. 그러나 **신호 분해(VMD, EEMD)** 전처리, **그래프 신경망** 기반 공간 모델링, **Transformer·Attention** 기반 장기 의존성 포착이 LL의 국소 선형 근사를 대체하는 방향으로 발전했습니다.  특히 NWP 오차 보정이라는 grey-box 개념은 NWP 공간 격자를 CNN/ConvLSTM의 원시 입력으로 직접 활용하는 방향으로 진화하고 있습니다.[^1_5][^1_8][^1_2][^1_4]
<span style="display:none">[^1_10][^1_11][^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48]</span>

<div align="center">⁂</div>

[^1_1]: Adaptive_local_learning_techniques_for_m.pdf

[^1_2]: https://www.bohrium.com/paper-details/a-novel-hybrid-deep-learning-model-for-multi-step-wind-speed-forecasting-considering-pairwise-dependencies-among-multiple-atmospheric-variables/925220073189147353-3814

[^1_3]: https://ieeexplore.ieee.org/document/11240128/

[^1_4]: https://www.sciencedirect.com/science/article/pii/S2352484725005797

[^1_5]: https://orca.cardiff.ac.uk/id/eprint/160517/1/1-s2.0-S0960148123006195-main (1).pdf

[^1_6]: https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/rpg2.70121

[^1_7]: https://ieeexplore.ieee.org/document/11193340/

[^1_8]: https://arxiv.org/html/2402.14385v1

[^1_9]: https://arxiv.org/pdf/2508.12764.pdf

[^1_10]: https://link.springer.com/10.1007/s12145-024-01493-2

[^1_11]: https://ieeexplore.ieee.org/document/10292889/

[^1_12]: https://arxiv.org/abs/2405.11431

[^1_13]: https://link.springer.com/10.1007/s13143-022-00291-4

[^1_14]: https://www.frontiersin.org/articles/10.3389/fbuil.2022.945615/full

[^1_15]: https://onlinelibrary.wiley.com/doi/10.1002/cpe.6772

[^1_16]: https://hdl.handle.net/2117/328183

[^1_17]: https://arxiv.org/pdf/2306.01986.pdf

[^1_18]: http://arxiv.org/pdf/1707.08110.pdf

[^1_19]: https://arxiv.org/pdf/2301.00819.pdf

[^1_20]: https://arxiv.org/pdf/2402.14385.pdf

[^1_21]: https://arxiv.org/pdf/2401.08233.pdf

[^1_22]: https://ijece.iaescore.com/index.php/IJECE/article/download/33168/17009

[^1_23]: https://peerj.com/articles/cs-732

[^1_24]: https://arxiv.org/html/2308.03472v4

[^1_25]: https://peerj.com/articles/cs-3114/

[^1_26]: https://arxiv.org/html/2411.15674v1

[^1_27]: https://peerj.com/articles/cs-2393/

[^1_28]: https://www.biorxiv.org/lookup/external-ref?access_num=10.1142%2FS0218488598000094\&link_type=DOI

[^1_29]: https://arxiv.org/pdf/2408.15554.pdf

[^1_30]: https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0299632

[^1_31]: https://www.semanticscholar.org/paper/e3607d3e3d195e70ef50987be1d7a469f5b233ab

[^1_32]: https://pdfs.semanticscholar.org/bf77/99ae557a0b83ed630f7cd884f63aeb29764e.pdf

[^1_33]: https://arxiv.org/html/2602.13010v1

[^1_34]: https://arxiv.org/pdf/2311.15807.pdf

[^1_35]: https://peerj.com/articles/cs-1949/

[^1_36]: https://www.sciencedirect.com/science/article/abs/pii/S0960148122010126

[^1_37]: https://ideas.repec.org/a/eee/energy/v304y2024ics0360544224017377.html

[^1_38]: https://public.pensoft.net/items/?p=7TVeXpoqfNYT89tyrm3ifrTeG9Wv8P676JSQp%2FH2pj9hhtoybol4GF7LEbj3fxHT5Fo8esHssd8WepBmZBDZahbEH%2F96bYoga45KiCTMEQerWxsfkgx1LbDC%2FzPse%2BUY\&n=qC9NWZh9LY9D7YU3%2Bi%2ByQrzYCtG%2Fs%2BvktsGL5LO1pz9tjw%3D%3D

[^1_39]: https://www.sciencedirect.com/science/article/abs/pii/S0960148122002440

[^1_40]: https://www.nature.com/articles/s41598-025-24640-1

[^1_41]: https://www.sciencedirect.com/science/article/abs/pii/S0360544221022295

[^1_42]: https://www.sciencedirect.com/science/article/abs/pii/S0306261922001404

[^1_43]: https://dl.acm.org/doi/10.1016/j.eswa.2023.121886

[^1_44]: https://www.nature.com/articles/s41598-024-83836-z

[^1_45]: https://www.sciencedirect.com/science/article/abs/pii/S0196890423013912

[^1_46]: https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2023.1298088/full

[^1_47]: https://ideas.repec.org/a/eee/energy/v285y2023ics0360544223028025.html

[^1_48]: https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0289161

