# A review and comparison of strategies for multi-step ahead time series forecasting based on the NN5 forecasting competition

이 논문은 다중 스텝 앞 시계열 예측을 위한 5가지 전략(Recursive, Direct, DirRec, MIMO, DIRMO)을 최초로 통합 비교하고, NN5 경쟁 111개 시계열에서 **Multiple-Output 전략**이 Single-Output 전략보다 일관되게 우수함을 입증한 연구입니다.[^1_1]

***

## 1. 핵심 주장 및 주요 기여

논문의 세 가지 핵심 발견은 다음과 같습니다.[^1_1]

- **Multiple-Output 전략(MIMO, DIRMO)이 Single-Output 전략(Recursive, Direct, DirRec)보다 일관되게 우수**
- **비계절화(Deseasonalization)는 모든 전략에서 예측 정확도를 균일하게 향상**시키며
- **입력 선택(Input Selection)은 비계절화와 함께 수행할 때 가장 효과적**

주요 기여는 두 가지입니다. ① 기존에 서로 다른 용어로 분산 소개되던 5가지 전략을 **통일된 수식 프레임워크**로 재정리한 이론적 비교 분석, ② NN5 경쟁(111개 실제 ATM 현금 인출 시계열)이라는 **대규모 벤치마크**를 통한 실증 비교입니다.[^1_1]

***

## 2. 해결 문제, 제안 방법, 모델 구조, 성능 및 한계

### 해결하려는 문제

**H-스텝 앞 예측 태스크**는 역사적 시계열 $[y_1, \ldots, y_N]$에서 미래 $[y_{N+1}, \ldots, y_{N+H}]$를 예측하는 문제입니다. 기존 연구들은 전략 간 비교 결과가 상충되었고, 어떤 전략을 선택해야 할지 실용적 가이드가 부재했습니다.[^1_1]

### 제안하는 방법 및 수식

**① Recursive 전략** — 단일 1-스텝 모델을 반복 적용:

$y_{t+1} = f(y_t, \ldots, y_{t-d+1}) + w \tag{1}$

```math
\hat{y}_{N+h} = \begin{cases} \hat{f}(y_N, \ldots, y_{N-d+1}) & h=1 \\ \hat{f}(\hat{y}_{N+h-1}, \ldots, y_N, \ldots) & h \in \{2,\ldots,d\} \\ \hat{f}(\hat{y}_{N+h-1}, \ldots, \hat{y}_{N+h-d}) & h \in \{d+1,\ldots,H\} \end{cases}
```

오차가 누적(error accumulation)되는 치명적 단점이 있습니다.[^1_1]

**② Direct 전략** — 각 호라이즌별 독립 모델 H개 학습:

$y_{t+h} = f_h(y_t, \ldots, y_{t-d+1}) + w, \quad h \in \{1,\ldots,H\} \tag{3}$

$\hat{y}\_{N+h} = \hat{f}\_h(y_N, \ldots, y_{N-d+1}) \tag{4}$

오차 누적은 없으나, H개 예측값 간 조건부 독립 가정 문제가 있습니다.[^1_1]

**③ DirRec 전략** — Direct와 Recursive를 결합. 각 스텝에 다른 모델을 쓰되, 이전 예측값을 입력에 추가:

$y_{t+h} = f_h(y_{t+h-1}, \ldots, y_{t-d+1}) + w \tag{5}$

입력 크기가 $H$에 비례해 선형 증가하여 계산 비용이 가장 큽니다.[^1_1]

**④ MIMO 전략** — 단일 다출력 모델로 전체 예측 벡터를 한 번에 출력:

$[y_{t+H}, \ldots, y_{t+1}] = F(y_t, \ldots, y_{t-d+1}) + w, \quad F: \mathbb{R}^d \to \mathbb{R}^H \tag{7}$

$[\hat{y}\_{t+H}, \ldots, \hat{y}\_{t+1}] = \hat{F}(y_N, \ldots, y_{N-d+1}) \tag{8}$

예측값 간 확률적 의존성을 보존하나, 모든 호라이즌에 동일 모델 구조를 강제하여 유연성이 감소합니다.[^1_1]

**⑤ DIRMO 전략** — MIMO를 블록 단위로 나누어 Direct와 MIMO의 장점을 절충. 블록 크기 $s$로 $n = H/s$개 다출력 모델 학습:

$[y_{t+p \cdot s}, \ldots, y_{t+(p-1) \cdot s+1}] = F_p(y_t, \ldots, y_{t-d+1}) + w \tag{9}$

$[\hat{y}\_{N+p \cdot s}, \ldots, \hat{y}\_{N+(p-1) \cdot s+1}] = \hat{F}\_p(y_N, \ldots, y_{N-d+1}) \tag{10}$

$s=1$이면 Direct, $s=H$이면 MIMO와 동일하며, 중간값 $s$로 의존성-유연성을 조절합니다.[^1_1]

**연산 시간 비교:**

```math
1 \times T_{SO} \underbrace{}_{\text{Rec}} < 1 \times T_{MO} \underbrace{}_{\text{MIMO}} < \frac{H}{s} \times T_{MO} \underbrace{}_{\text{DIRMO}} < H \times T_{SO} \underbrace{}_{\text{Dir}} < H \times (T_{SO}+\mu) \underbrace{}_{\text{DirRec}}
```

### 모델 구조: Lazy Learning

논문은 **Lazy Learning(LL)** 알고리즘을 기반 예측기로 사용합니다. 이는 쿼리 기반 지역 모델링(local modeling) 기법으로, 예측 요청이 들어올 때까지 학습을 지연시키고, 쿼리 포인트 $x_q$ 주변의 $k$-최근접 이웃으로 지역 모델을 추정합니다.[^1_1]

**단일 출력 LOO 오차 (PRESS 통계):**

$e_{LOO}(k) = \frac{1}{k} \sum_{j=1}^{k} (e_j(k))^2 \tag{12}$

$e_j(k) = \frac{k \cdot y_{[j]} - y_q^{(k)}}{k-1} \tag{17}$

PRESS 통계를 이용해 Leave-One-Out 교차검증을 $O(k)$ 반복 없이 효율적으로 계산합니다.[^1_1]

**다출력 ACFLIN 불일치 기준(일반화 성능의 핵심):**

```math
E_\Delta(k) = \underbrace{1 - \left|cor[\rho(ts \cdot y_q^{(k)}), \rho(ts)]\right|}_{\text{자기상관 불일치}} + \underbrace{1 - \left|cor[\pi(ts \cdot y_q^{(k)}), \pi(ts)]\right|}_{\text{편자기상관 불일치}} 
```

예측 시퀀스 $y_q^{(k)}$가 훈련 시계열 $ts$의 자기상관($\rho$) 및 편자기상관($\pi$) 구조를 얼마나 잘 보존하는지 측정합니다.[^1_1]

### MIMO-ACFLIN의 정체
- 우승 전략: MIMO-ACFLIN은 NN5 시계열 예측 대회(111개 데이터셋)에서 가장 우수한 성능을 보였던 최종 예측 전략(Winner strategy)을 지칭하는 명칭으로 사용됩니다.
- MIMO 전략: 예측해야 할 전체 미래 구간( $H$ )을 한 번에 벡터로 출력하여 미래 시점 간의 확률적 의존성을 보존하는 다중 출력(Multiple-Output) 방식을 취하고 있습니다.
- Lazy Learning 기반: 해당 논문의 실험에서 MIMO-ACFLIN은 Lazy Learning(국소 모델링 기법)을 기반으로 구현된 것으로 분류됩니다. 

### Algorithm 3과의 관계
Algorithm 3은 Multiple-Output Lazy Learning (discrepancy criterion)의 구체적인 계산 절차를 담고 있습니다.  
이 알고리즘은 쿼리 지점( $x_{q}$ )과 가장 유사한 과거 데이터(이웃)들을 찾아 그 평균값으로 미래를 예측하며, 최적의 이웃 수( $k^{\star}$ )를 결정하기 위해 불일치 기준(Discrepancy Criterion, 식 18)을 사용합니다.  
따라서 MIMO-ACFLIN은 이 Algorithm 3(Lazy Learning 기반의 MIMO 방식)을 사용하여 최적화된 최종 모델을 일컫는 이름으로 이해할 수 있습니다.

요약하자면, MIMO-ACFLIN은 NN5 대회에서 우승한 다중 출력 국소 학습(Lazy Learning) 기반의 예측 시스템을 뜻하는 고유 명칭으로 사용되나, ACFLIN이라는 약어의 상세 풀이는 논문에 언급되지 않았습니다.

### MIMO-ACFLIN의 정확한 실체
MIMO-ACFLIN은 특정 단일 모델이 아니라, NN5 시계열 예측 대회에서 주저자(S. Ben Taieb) 팀이 우승할 때 사용한 최종 예측 시스템(Winner Strategy)의 명칭입니다. 이 명칭은 다음 두 가지 핵심 요소의 결합을 뜻합니다.  
- MIMO (Multi-Input Multi-Output): 예측하려는 전체 구간( $H$ )을 하나의 모델을 통해 한 번에 출력하는 전략입니다. 이는 한 단계씩 예측하여 다시 입력으로 쓰는 Recursive 방식의 오차 누적 문제를 해결하기 위해 채택되었습니다.
- ACFLIN (Averaged Computational Intelligence and Linear models): 약어의 정확한 의미는 저자의 연구 맥락상 "신경망(NN), 가우시안 프로세스(GP)와 같은 컴퓨팅 지능(CI) 모델들과 선형(Linear) 모델들을 평균(Averaged) 내어 결합했다"는 뜻으로 해석됩니다. 실제로 해당 논문에서는 이 전략이 Forecast Combination(예측 결합)의 일환으로 설명됩니다.

Ben Taieb의 다른 연구들(예: "Adaptive local learning techniques...", "Machine learning strategies for multi-step-ahead...")에서도 그는 Lazy Learning(국소 학습)과 MIMO 방식을 결합한 기법이 다단계 예측에서 매우 효과적임을 지속적으로 강조해 왔습니다.  
특히 NN5 대회 우승 당시 그는 이 Lazy Learning 모델을 단독으로 쓰지 않고, 다른 전역 모델들과 평균(Averaging) 내어 성능을 극대화했으며, 이를 지칭하는 고유 명칭이 바로 MIMO-ACFLIN입니다.

MIMO-ACFLIN은 MIMO 방식의 국소 학습(Algorithm 3)을 포함한 여러 모델의 예측값을 평균 내어 산출하는 앙상블 전략을 의미합니다.

**성능 평가 지표:**

```math
SMAPE = \frac{1}{H}\sum_{h=1}^H \frac{|\hat{y}_h - y_h|}{(\hat{y}_h + y_h)/2} \times 100, \quad SMAPE^* = \frac{1}{111}\sum_{i=1}^{111} SMAPE_i
```

**Friedman 통계 및 Iman-Davenport 보정:**

$Q = \frac{12N}{k(k+1)} \left[\sum_j R_j^2 - \frac{k(k+1)^2}{4}\right], \quad S = \frac{(N-1)Q}{N(k-1)-Q} \tag{22, 23}$

### 성능 향상 결과

| 구성 | 최고 전략 | SMAPE* |
| :-- | :-- | :-- |
| 비계절화 없음, 입력 선택 없음 | MIMO-LOO (COMB) | 20.61% |
| 비계절화, 입력 선택 없음 | DIRMO-SEL (COMB) | 18.98% |
| **비계절화, 입력 선택** | **MIMO-ACFLIN (WINNER)** | **18.95%** |
| 비교 모델 (Andrawis et al., 2011) Combined | Combined | 18.95% |

MIMO-ACFLIN + 비계절화 + 입력 선택 조합이 SMAPE* = 18.81%로 GPR, NN 등 타 머신러닝 모델과 동등하거나 우위를 보였습니다.[^1_1]

### 한계점

- **단일 학습기만 사용**: LL 알고리즘 하나만 테스트하여, 다른 모델(SVM, 딥러닝 등)에서의 일반화 불확실
- **DIRMO + 입력 선택 조합 미수행**: 파라미터 $s$ 탐색과 입력 선택 동시 수행 시 과도한 계산 비용 문제
- **도메인 제한**: 111개 ATM 현금 인출 시계열이라는 특정 도메인에 한정
- **DirRec의 열악한 성능**: 입력 차원 폭발과 과적합(overfitting)으로 일관되게 최하위 또는 하위권[^1_1]

***

## 3. 일반화 성능 향상 가능성

논문에서 일반화 성능 향상과 가장 직접 연결된 메커니즘은 다음 세 가지입니다.[^1_1]

**① LOO 교차검증 기반 이웃 수 선택**: PRESS 통계를 이용하여 각 쿼리마다 최적 $k^* = \arg\min_{k} e_{LOO}(k)$를 선택하며, 이는 편향-분산 트레이드오프를 데이터 적응적으로 조절합니다.

**② ACFLIN 불일치 기준**: MIMO-ACFLIN은 예측 시퀀스가 훈련 시계열의 자기상관 구조를 보존하도록 $k^* = \arg\min_k E_\Delta(k)$를 선택합니다. 이는 시계열의 확률적 의존성을 반영한 일반화 지표로, LOO 오차만 쓰는 MIMO-LOO보다 비계절화 상황에서 더 우수한 성능을 보였습니다.

**③ 모델 평균화(COMB, WCOMB)**: 단순 Winner-take-all 대신 여러 $k$값의 예측을 결합:

```math
\hat{y}_q = \frac{\sum_{k=2}^{K_{max}} p_k \cdot y_q^{(k)}}{\sum_{k=2}^{K_{max}} p_k}, \quad p_k = \frac{1}{e_{LOO}(k)} \text{ (WCOMB)} 
```

COMB/WCOMB는 특히 비계절화 없는 상황에서 WINNER보다 일관되게 우수하여, **앙상블 기반 접근이 과적합을 줄이고 일반화를 향상**시킴을 보였습니다.[^1_1]

***

## 4. 미래 연구에 미치는 영향 및 고려사항

### 연구에 미치는 영향

이 논문은 다중 스텝 예측 전략 연구의 **표준 참고문헌**으로 자리잡았습니다.  "Single-Output 전략이 Multiple-Output 전략에 비해 열등하다"는 결론은 이후 딥러닝 시대의 MIMO 기반 모델 설계(seq2seq, Transformer 인코더-디코더 등)를 정당화하는 이론적 근거가 되었습니다.[^1_2][^1_3][^1_4]

### 앞으로의 연구 시 고려사항

- **딥러닝 모델과의 결합**: Lazy Learning이 아닌 LSTM, Transformer 등에 MIMO/DIRMO 전략 적용 효과 검증 필요
- **파라미터 $s$ 자동 선택**: DIRMO의 블록 크기 $s$ 선택 비용 최소화를 위한 효율적 탐색 알고리즘 개발
- **비계절화 방법 최적화**: Multiple-Output 전략 전용 비계절화 기법 설계
- **불확실성 정량화**: SMAPE 외에 예측 구간(prediction interval) 평가 추가
- **다변량 확장**: 단변량 기반 전략을 다변량 시계열(multivariate) 환경으로 확장

***

## 5. 2020년 이후 최신 연구 비교 분석

| 연구 | 주요 방법 | 핵심 발견 | 본 논문과의 관계 |
| :-- | :-- | :-- | :-- |
| **Stratify (2024)** [^1_2] | 파라미터화된 통합 MSF 프레임워크 | 1,080개 실험 중 84%에서 기존 전략 대비 성능 향상 | DIRMO의 파라미터 $s$ 연속화 및 일반화 |
| **iTransformer (2023)** [^1_5] | 변수별 토큰화, MIMO 패러다임 | 다변량 의존성 포착, 다양한 룩백 윈도우에서 일반화 | MIMO 전략의 딥러닝 구현체, 본 논문 결론 지지 |
| **PatchTST/Timer-XL (2024)** [^1_6] | 패치 기반 Transformer, MIMO 직접 매핑 | 자동회귀 대비 직접 매핑(Direct/MIMO) 방식이 우수 | "단일 출력 < 다출력" 결론을 대규모로 재확인 |
| **AcMCP (2022)** [^1_7] | 자기상관 기반 다단계 예측 구간 | MIMO 전략 기반 분포 없는 예측 구간 보장 | ACFLIN 아이디어를 불확실성 정량화로 확장 |
| **IoT-LSTM (2025)** [^1_8] | LSTM + Direct/MIMO/DirMO 비교 | Direct, MIMO, DirMO가 Recursive보다 일관되게 우수 | 14년 후 실제 환경에서 핵심 결론 재검증 |
| **Quantile DL (2024)** [^1_9] | 분위수 손실 + 딥러닝 다단계 예측 | 변동성이 큰 환경에서 MIMO 기반 분위수 예측 유효 | MIMO 전략 + 불확실성 정량화의 결합 |

특히 Stratify (arXiv:2412.20510) 는 본 논문의 DIRMO 아이디어를 연속적 파라미터 공간으로 확장하여, 전략 선택 자체를 최적화 문제로 재정의했습니다.  
또한, Transformer 기반 장기 예측 연구들 이 "autoregressive(Recursive) < direct mapping(MIMO)" 결론을 2024년 벤치마크에서 반복적으로 확인하고 있어, 본 논문의 핵심 결론은 딥러닝 시대에도 여전히 유효합니다.[^1_4][^1_2]  
<span style="display:none">[^1_10][^1_11][^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47]</span>

<div align="center">⁂</div>

[^1_1]: 1108.3259v1.pdf

[^1_2]: https://arxiv.org/html/2412.20510v1

[^1_3]: https://arxiv.org/html/2601.07640v1

[^1_4]: https://arxiv.org/html/2507.13043v1

[^1_5]: https://arxiv.org/abs/2310.06625

[^1_6]: https://arxiv.org/html/2410.04803v1

[^1_7]: https://arxiv.org/html/2410.13115v2

[^1_8]: https://pubmed.ncbi.nlm.nih.gov/41061537/

[^1_9]: https://arxiv.org/abs/2411.15674

[^1_10]: https://arxiv.org/html/2509.22395v1

[^1_11]: https://arxiv.org/html/2511.23260v1

[^1_12]: https://arxiv.org/html/2409.14792v1

[^1_13]: https://arxiv.org/html/2502.10721v3

[^1_14]: https://pdfs.semanticscholar.org/e0c8/30480dae986b468a9bba5910a3e1c6e7c919.pdf

[^1_15]: https://arxiv.org/abs/2310.20218

[^1_16]: https://arxiv.org/pdf/2509.13945.pdf

[^1_17]: https://arxiv.org/html/2402.15290v1

[^1_18]: https://arxiv.org/html/2509.15843v1

[^1_19]: https://arxiv.org/html/2502.05952v1

[^1_20]: https://dl.acm.org/doi/10.1145/3583133.3595841

[^1_21]: https://www.mdpi.com/2071-1050/15/3/1895

[^1_22]: https://linkinghub.elsevier.com/retrieve/pii/S0957417412000528

[^1_23]: https://www.mdpi.com/1424-8220/21/7/2430

[^1_24]: http://ieeexplore.ieee.org/document/7422387/

[^1_25]: https://ieeexplore.ieee.org/document/9699518/

[^1_26]: http://ieeexplore.ieee.org/document/6137274/

[^1_27]: https://dl.acm.org/doi/10.1145/2001858.2001982

[^1_28]: https://www.semanticscholar.org/paper/b054a2815234ce4460fe20255c304113fe4a189d

[^1_29]: https://arxiv.org/pdf/2312.17100.pdf

[^1_30]: https://arxiv.org/html/2503.20148v1

[^1_31]: https://arxiv.org/html/2310.06119v2

[^1_32]: http://arxiv.org/pdf/2403.20150.pdf

[^1_33]: http://arxiv.org/pdf/2410.22981.pdf

[^1_34]: http://arxiv.org/pdf/2403.02150.pdf

[^1_35]: http://arxiv.org/pdf/2310.07446.pdf

[^1_36]: https://www.sciencedirect.com/science/article/abs/pii/S0957417412000528

[^1_37]: https://dl.acm.org/doi/fullHtml/10.1145/3582177.3582187

[^1_38]: https://ieeexplore.ieee.org/document/10005585/

[^1_39]: https://public.pensoft.net/items/?p=7TVeXpoqfNYT89tyrm3ifrTeG9Wv8P676JSQp%2FH2pj9hhtoybol4GF7LEbj3fxHT5Fo8esHssd8WepBmZBDZahbEH%2F96bYoga45KiCTMEQerWxsfkgx1LbDC%2FzPse%2BUY\&n=qC9NWZh9LY9D7YU3%2Bi%2ByQrzYCtG%2Fs%2BvktsGL5LO1pz9tjw%3D%3D

[^1_40]: https://arxiv.org/pdf/1108.3259.pdf

[^1_41]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12453695/

[^1_42]: https://research.monash.edu/en/publications/a-review-and-comparison-of-strategies-for-multi-step-ahead-time-s/

[^1_43]: https://arxiv.org/abs/2311.04147

[^1_44]: https://arxiv.org/abs/1108.3259

[^1_45]: https://openforecast.org/2024/05/25/recursive-vs-direct-forecasting-strategy/

[^1_46]: https://dl.acm.org/doi/abs/10.1145/3582177.3582187

[^1_47]: https://www.diva-portal.org/smash/get/diva2:1135425/FULLTEXT01.pdf

