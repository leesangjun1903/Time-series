
# FreDo: Frequency Domain-based Long-Term Time Series Forecasting
## 요약
**"FreDo: Frequency Domain-based Long-Term Time Series Forecasting"**는 MIT의 Fan-Keng Sun과 Duane S. Boning이 2022년 발표한 논문으로, 장기 시계열 예측 분야에서 근본적인 문제점을 지적하고 혁신적인 솔루션을 제시합니다. 이 논문은 세 가지 핵심 기여를 통해 시계열 예측 연구의 패러다임을 전환했습니다:

1. **오차 누적의 수학적 증명**: 장기 예측에서는 아무리 정교한 모델도 단순 기저선을 능가하기 어렵다는 이론적 근거 제시
2. **AverageTile 기저선 모델**: 파라미터가 0개인 주기성 기반 모델이 최첨단 Transformer 모델과 경쟁 가능함을 입증
3. **FreDo 모델**: 주파수 도메인에서 학습하여 Autoformer를 20~50% 이상 능가하는 성능 달성

## 해결하고자 하는 문제
### 오차 누적 (Error Accumulation) 문제
FreDo의 핵심은 다음의 기본 방정식으로부터 시작합니다:

$$x_t = f(x_{t-1}, x_{t-2}, ...) + e_t$$

여기서 $e_t$는 평균 0, 분산 $\sigma^2$인 오차입니다. p차 자기회귀(AR) 모델의 경우:

$$x_t = c + \sum_{i=1}^{p} \theta_p x_{t-p} + e_t$$

논문은 다음을 엄밀하게 증명합니다: 예측 horizon이 증가할수록 달성 가능한 최소 MSE는 단조증가하며, 약정상 시계열의 경우 시간이 지남에 따라 예측 오차가 $\text{Var}[x_t]$에 접근합니다. 즉, horizon이 충분히 길어지면 어떤 모델도 단순히 평균값을 예측하는 것보다 나을 수 없다는 역설적 결론에 도달합니다.

**수학적 표현**:

$$\text{Var}[x_{p+k}] = \text{Var}\left[\sum_{i=0}^{k} \phi^i e_{p+k-i}\right]$$

이는 $k$가 증가함에 따라 단조증가하므로, 최적 MSE도 다음과 같이 증가합니다:

$$\text{MSE}_{\text{optimal}}(h) \text{ is non-decreasing in forecast horizon } h$$

### 기존 Transformer 모델의 한계
논문은 Autoformer, Informer, LogTrans 같은 복잡한 모델들이 이 오차 누적 문제를 무시하고 설계되었다고 지적합니다. 실제로 이들 모델은 다음과 같은 문제를 야기합니다:

- **과복잡성**: 불필요한 파라미터와 연산 복잡도 증가
- **과적합 경향성**: 제한된 데이터에서 복잡한 패턴 학습의 어려움
- **주기성 무시**: 많은 현실 시계열의 강한 주기성을 충분히 활용하지 못함

## 제안하는 방법
### 1. AverageTile: 비매개변수 기저선 모델
AverageTile은 주기성이 P인 시계열에 대해 다음과 같이 작동합니다:

입력 히스토리 $\{x_{t-I}, ..., x_{t-1}\}$ (단, $I = rP$, $r \in \mathbb{Z}^+$)가 주어질 때:

$$\hat{x}_{t+o} = \frac{1}{r} \sum_{i=1}^{r} x_{t+(o \bmod P)-iP}$$

**핵심 특징**:
- 훈련 가능한 파라미터: **0개**
- 계산 복잡도: $O(1)$
- 주기성을 직접 활용하여 주기적 성분을 모델링

**주기성 결정 과정**:
- ETTm2: 96 (매 15분, 일일 주기)
- Electricity: 24 (시간 간격, 일일 주기)
- Exchange: 1 (주기성 없음)
- Traffic: 24 (시간 간격)
- Weather: 144 (10분 간격, 일일 주기)

### 2. FreDo: 주파수 도메인 신경망 모델
FreDo의 아키텍처는 다음과 같이 구성됩니다:

**Step 1: DFT 변환**

입력 $x \in \mathbb{R}^I$에 대해 이산 푸리에 변환(DFT)을 수행:

$$\zeta_{\text{raw}} = \text{DFT}(x) \in \mathbb{C}^I$$

주파수 빈 중심은:
$$\frac{1}{I}\left[-\frac{I}{2}+1, ..., -1, 0, 1, ..., \frac{I}{2}\right]$$

**Step 2: 실수값 추출 (DFT and Extract)**

복소수 주파수 성분을 실수값으로 변환:

$$\zeta = \left[\Re(\zeta_0), ..., \Re(\zeta_{I/2}), \Im(\zeta_1), ..., \Im(\zeta_{I/2-1})\right] \in \mathbb{R}^I$$

(I가 짝수인 경우)

**Step 3: 비선형 처리**

선형 레이어로 차원 I을 O로 투영:

$$z = W \cdot \zeta + b, \quad W \in \mathbb{R}^{O \times I}$$

Mixer 모듈들을 순차적으로 적용:

$$h_{l+1} = h_l + \text{MLP}_l(h_l), \quad l=1,...,L$$

**Step 4: 역 DFT (Insert and Inverse DFT)**

다시 시간 도메인으로 변환:

$$\hat{x} = \text{IFFT}(\zeta_{\text{processed}})$$
### 3. 모델 구조 비교
### 아키텍처의 핵심 혁신
1. **AverageTile 기반 정제**: 기저선 예측에 학습 가능한 파라미터를 추가하여 점진적으로 개선
2. **실수값 주파수 표현**: 복소수 연산을 피하고 실수 신경망의 표준 역전파 사용
3. **주파수 도메인의 장점**:
   - 주기성을 자연스럽게 인코딩
   - 장기 의존성을 더 효율적으로 모델링
   - 노이즈에 대한 저항성 (낮은 진폭 성분은 자동 필터링)

## 성능 향상 및 실험 결과
### 7개 데이터셋에서의 성능
**Autoformer 설정 (Table 2)**:

| 데이터셋 | Horizon | Autoformer MSE | FreDo MSE | 개선율(%) |
|---------|---------|----------------|-----------|----------|
| ETTm2 | 720 | 0.422 | 0.297 | 29.6 |
| Electricity | 720 | 0.254 | 0.221 | 13.0 |
| Exchange | 720 | 1.447 | 0.534 | 63.1 |
| Traffic | 720 | 0.660 | 0.538 | 18.5 |
| Weather | 720 | 0.419 | 0.331 | 21.0 |

**평균 개선율**: 
- Exchange: 52.9% (최고)
- ETTm2: 30.6%
- Weather: 28.9%
- Electricity: 15.4% (최저)

### TimeDo vs FreDo: 주파수 도메인 검증
주파수 도메인의 우월성을 증명하기 위해 **TimeDo** (시간 도메인 버전)와 비교:
- 동일한 파라미터 수
- 동일한 아키텍처
- 유일한 차이: 시간/주파수 도메인

**결과**: FreDo는 TimeDo에 비해 통계적으로 유의미하게 우수한 성능을 보임 (p < 1%)

### 일반화 성능 향상
**Autoformer 대비 강점**:
1. **모델 크기**: 약 100K 파라미터 vs 1M+ 파라미터
2. **편향-분산 트레이드오프**: 더 작은 모델로 더 낮은 편향과 분산 달성
3. **샤프한 오차 곡선**: Autoformer는 초반부에 역 U자 모양 오차 곡선을 보이며 일반화 실패 징후

## 모델의 한계
논문은 다음과 같은 명시적 한계를 인정합니다:

1. **단기 예측에서 성능 저하**: FreDo는 장기 예측에 최적화되어 있어 단기(horizon < 96) 예측에서는 개선 효과 제한
2. **단변량 설계**: 모델이 본질적으로 단변량이므로 변수 간 상호작용을 직접 학습하지 않음
   - 다변량 데이터셋에서는 각 변수마다 독립적으로 동일 모델 적용
   - 변수 간 의존성 활용 불가
3. **주기성 의존성**: 비주기적 데이터(예: Exchange)에서는 이점 제한
4. **하이퍼파라미터**: 주기 P의 선택이 성능에 중요하며 수동 설정 필요

## 2020년 이후 관련 최신 연구 비교 분석
### 주요 경쟁 모델들
**1. DLinear (AAAI 2023)** [arxiv](https://arxiv.org/abs/2512.10866)
- **아이디어**: 분해 + 단일 선형 레이어
- **성능**: Transformer를 능가하는 단순성과 효율성
- **FreDo와의 차이**: 
  - DLinear: 시간 도메인, 분해 기반 (trend + seasonal)
  - FreDo: 주파수 도메인, 주기성 기반
  - FreDo가 복잡한 주기 구조에서 더 나은 성능

**2. TimesNet (ICLR 2023)** [ar5iv.labs.arxiv](https://ar5iv.labs.arxiv.org/html/2210.02186)
- **혁신**: 1D 시계열을 다중 주기성으로 2D 텐서로 변환
- **방법**: FFT로 주기 탐지 → reshape → 2D CNN
- **강점**: 다중 주기 모델링
- **FreDo와의 비교**: 
  - TimesNet이 더 많은 주기 정보 활용
  - FreDo는 더 간단한 구조로 유사 성능

**3. iTransformer (ICLR 2024)** [arxiv](https://arxiv.org/html/2310.06625v4)
- **패러다임 전환**: 변수를 토큰으로, 시간 포인트를 피처로 처리
- **성능**: Autoformer 대비 평균 38.9% 성능 향상
- **FreDo와의 관계**:
  - iTransformer: 다변량 의존성 학습 가능
  - FreDo: 단변량이지만 주파수 도메인 학습

**4. FEDformer (ICML 2022)** [arxiv](https://arxiv.org/html/2503.06928v1)
- **혁신**: Fourier-Wavelet 하이브리드 + 분해
- **복잡성**: O(n log n)
- **FreDo와의 공통점**: 주파수 도메인 활용
- **차이점**: FEDformer는 attention 메커니즘 내에서 주파수 활용

**5. PatchTST (ICLR 2023)** [arxiv](https://arxiv.org/html/2510.23396v1)
- **아이디어**: 시각 Transformer의 패칭 개념 도입
- **특징**: 채널 독립성 유지, 로컬 의미 정보 보존
- **성능**: DLinear 능가
- **FreDo와의 차이**: 
  - PatchTST: 시간 도메인 패칭
  - FreDo: 주파수 도메인 직접 학습

### 최신 연구 트렌드
**1. 주파수-시간 도메인 하이브리드** [dl.acm](https://dl.acm.org/doi/10.1145/3664647.3681210)
```
최신 관찰: 모든 주파수 성분이 동등하게 중요하지 않음
→ 시나리오별로 동적으로 주파수 기여도 조정
예시: FreDF, Dualformer
```

**2. 선형 모델의 재평가** [arxiv](https://arxiv.org/abs/2512.10866)
- DLinear 이후 선형 모델들이 Transformer 능가
- 시계열 예측의 "역설": 복잡함이 항상 도움이 되지 않음
- 2024-2025년: GLinear, NLinear, RLinear 등 다양한 선형 변형

**3. 일반화 능력 연구** [arxiv](https://arxiv.org/abs/2404.06198)
- **도메인 전이 (Transfer Learning)**:
  - 데이터셋 유사도 중요성 증명
  - 원본 데이터의 다양성과 목표 데이터의 유사성이 모두 중요
  
- **분포 외 일반화 (OOD Generalization)**:
  - 시계열의 비정상성(non-stationarity)으로 인한 분포 변화
  - 불변 표현 학습(Invariant Representation Learning) 대두

**4. 주파수 해석의 심화** [arxiv](https://arxiv.org/html/2510.25800v1)
```
Spectral Bias 현상:
- 신경망은 저주파 성분을 먼저 학습하는 경향
- FreLE (Frequency Loss Enhancement): 주파수별 손실 균형 조정
- FreMixer: 다중 해상도 주파수 분석
```

**5. Foundation Models의 부상** [semanticscholar](https://www.semanticscholar.org/paper/39e0e964d6ba714584c6fd58e170fc36370da6a6)
- TimeGPT, Moirai 등 대규모 사전학습 모델
- Zero-shot/Few-shot 학습 가능
- FreDo의 단변량 한계를 다변량 확장으로 극복

## 모델 일반화 성능 향상 분석
### 왜 FreDo가 일반화에서 우수한가?
**1. 구조적 정규화 효과**
- 주파수 도메인 표현은 자동으로 저주파 성분을 강조
- 고주파 노이즈는 자연스럽게 필터링
- 이는 명시적 정규화 없이도 일반화 성능 향상

**2. 모델 복잡도 관리**
- 파라미터 수: Autoformer 1M+ vs FreDo 100K
- 더 작은 모델로 더 나은 편향-분산 트레이드오프
- 과적합 위험 감소

**3. 주기 구조의 명시적 활용**
- AverageTile이 이미 주기 정보를 효과적으로 캡처
- Mixer 모듈은 오직 잔차(residual) 개선에 집중
- 불필요한 복잡한 패턴 학습 회피

### 새로운 도메인으로의 전이 가능성
**잠재력**:
- 주기성은 많은 실제 도메인의 본질적 특성
- 단변량 모델이므로 새로운 변수에 쉽게 적용 가능

**한계**:
- 비주기적 데이터(시장 데이터, 랜덤 워크)에서는 이점 제한
- 변수 간 강한 의존성이 있는 데이터셋에서 성능 저하

## 앞으로의 연구 방향 및 고려 사항
### 1. 다변량 확장 (Multivariate Extension)
**현황**: FreDo는 각 변수에 독립적으로 적용
```
개선안:
1. Shared frequency components: 모든 변수가 공통 주파수 기저 사용
2. Cross-channel Mixer: 변수 간 상호작용 모델링
3. Hierarchical frequency: 변수 간 주기성 계층 구조
```

### 2. 단기 예측 성능 개선
**문제**: Horizon < 100에서 성능 저하
```
해결책:
1. Multi-scale Mixer: 서로 다른 스케일의 혼합 (timescale-aware)
2. Adaptive horizon weighting: 예측 거리에 따른 동적 가중치
3. Hybrid approach: 단기는 시간 도메인, 장기는 주파수 도메인
```

**참고**: TimeMixer (ICLR 2024)가 이 방향의 성공 사례

### 3. 주기 선택의 자동화
**현황**: 주기 P를 수동으로 설정
```
자동 방법:
1. FFT 기반 자동 주기 탐지 (TimesNet 접근)
2. Multiple candidates: 여러 주기 후보를 동시에 처리
3. Learnable periodicity: P 자체를 학습 가능한 파라미터로
```

### 4. 불변 표현 학습과의 결합
**최신 트렌드** (2024-2025): OOD Generalization
```
아이디어:
1. 주파수 도메인에서 불변 특징 학습
2. 분포 변화에 강건한 sparse frequency components
3. Multi-environment training: 여러 환경에서 공통 주파수 패턴 학습
```

### 5. 대규모 사전학습
**방향**: Foundation Model으로의 진화
```
잠재적 전략:
1. 다양한 도메인의 시계열로 사전학습된 주파수 필터 생성
2. Transfer to new domains: 간단한 linear head 추가로 새 도메인 적응
3. Few-shot learning: 제한된 데이터로 빠른 적응
```

### 6. 이론적 분석 심화
**개선 영역**:
1. 주파수 도메인에서의 오차 누적 이론
2. 샘플 복잡도 분석: 주기성이 있을 때 필요 데이터 양
3. 최적성 보장: FreDo의 주파수 도메인 학습이 최적임을 증명

## 결론
**FreDo**는 시계열 예측에서 근본적인 문제를 날카롭게 지적하고 우아한 해결책을 제시했습니다. 오차 누적의 수학적 증명을 통해 "복잡함이 항상 성능을 보장하지 않는다"는 통찰을 제공했으며, 주파수 도메인 학습의 장점을 체계적으로 검증했습니다.

**핵심 기여의 영향**:
- DLinear의 선형 모델 성공을 선제적으로 설명하는 이론적 근거 제공
- 주파수 도메인 강조 연구(FEDformer, TimesNet, FreDF 등)의 선구자
- 간단한 모델의 가치 재평가로 AI 민주화 기여

**현재 위치 (2024-2025)**:
- 선형 모델이 Transformer와 경쟁하는 시대
- 주파수-시간 도메인 하이브리드 모델의 최적화 추세
- 대규모 사전학습 기반 Foundation Model로 진화 중

**실무 권장사항**:
1. **새로운 도메인**: 주기성 분석 후 FreDo/DLinear 먼저 시도
2. **다변량 데이터**: iTransformer나 PatchTST 더 추천
3. **극도로 장기**: FreDo의 주파수 도메인 접근 탁월
4. **제한된 리소스**: FreDo의 경량성(100K 파라미터) 활용

시간 경과에 따라 이 분야는 계속 진화하고 있으며, FreDo의 핵심 아이디어—주기성과 주파수 구조의 명시적 활용—는 앞으로도 우수한 시계열 모델의 기초가 될 것으로 예상됩니다.

<span style="display:none">[^1_100][^1_101][^1_102][^1_103][^1_104][^1_105][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_80][^1_81][^1_82][^1_83][^1_84][^1_85][^1_86][^1_87][^1_88][^1_89][^1_90][^1_91][^1_92][^1_93][^1_94][^1_95][^1_96][^1_97][^1_98][^1_99]</span>

<div align="center">⁂</div>

[^1_1]: https://arxiv.org/abs/2512.10866

[^1_2]: https://simudyne.com/resources/a-single-linear-layer-is-all-you-need-linear-models-outperform-transformers-for-long-term-time-series-forecasting/

[^1_3]: https://ar5iv.labs.arxiv.org/html/2210.02186

[^1_4]: https://ise.thss.tsinghua.edu.cn/~mlong/doc/foundation-models-for-time-series-analysis-gaitc23.pdf

[^1_5]: https://arxiv.org/html/2310.06625v4

[^1_6]: https://arxiv.org/html/2503.06928v1

[^1_7]: https://arxiv.org/html/2601.15669v1

[^1_8]: https://arxiv.org/html/2510.23396v1

[^1_9]: https://dl.acm.org/doi/10.1145/3664647.3681210

[^1_10]: https://arxiv.org/html/2506.09174v2/

[^1_11]: https://arxiv.org/pdf/2510.10799.pdf

[^1_12]: https://arxiv.org/abs/2404.06198

[^1_13]: https://www.semanticscholar.org/paper/39e0e964d6ba714584c6fd58e170fc36370da6a6

[^1_14]: https://arxiv.org/html/2503.13868v3

[^1_15]: https://arxiv.org/html/2510.25800v1

[^1_16]: https://arxiv.org/pdf/2504.07099.pdf

[^1_17]: https://arxiv.org/html/2504.07099v4

[^1_18]: https://www.nature.com/articles/s41598-025-95529-2

[^1_19]: https://arxiv.org/html/2504.04011v1

[^1_20]: 2205.12301v1.pdf

[^1_21]: https://link.springer.com/10.1007/s11356-024-32228-x

[^1_22]: https://www.ijcit.com/index.php/ijcit/article/view/8

[^1_23]: https://onlinelibrary.wiley.com/doi/10.1002/for.3082

[^1_24]: https://www.mdpi.com/2072-4292/16/11/1915

[^1_25]: https://link.springer.com/10.1007/s42488-024-00122-3

[^1_26]: https://ieeexplore.ieee.org/document/10873159/

[^1_27]: https://pubs2.ascee.org/index.php/IJRCS/article/view/1546

[^1_28]: https://ieeexplore.ieee.org/document/10824647/

[^1_29]: https://www.scitepress.org/DigitalLibrary/Link.aspx?doi=10.5220/0013395500003890

[^1_30]: http://www.proceedings.com/079017-0272.html

[^1_31]: https://arxiv.org/pdf/2404.15772.pdf

[^1_32]: http://arxiv.org/pdf/2410.15217.pdf

[^1_33]: https://arxiv.org/pdf/2308.00709.pdf

[^1_34]: https://arxiv.org/pdf/2312.17100.pdf

[^1_35]: https://arxiv.org/pdf/2411.05793.pdf

[^1_36]: https://arxiv.org/pdf/2310.10688.pdf

[^1_37]: https://arxiv.org/pdf/2304.08424.pdf

[^1_38]: https://arxiv.org/html/2309.15946

[^1_39]: https://arxiv.org/pdf/2308.04791.pdf

[^1_40]: https://arxiv.org/pdf/2509.26468.pdf

[^1_41]: https://www.arxiv.org/pdf/2512.12301.pdf

[^1_42]: https://arxiv.org/html/2301.10874v2

[^1_43]: https://arxiv.org/abs/2511.11817

[^1_44]: https://arxiv.org/pdf/2211.14730.pdf

[^1_45]: https://arxiv.org/html/2509.26468v1

[^1_46]: https://arxiv.org/html/2508.20206v1

[^1_47]: https://arxiv.org/html/2502.13721v1

[^1_48]: https://arxiv.org/html/2511.01468v1

[^1_49]: https://arxiv.org/abs/2004.00574

[^1_50]: https://arxiv.org/pdf/2305.12095.pdf

[^1_51]: https://www.sciencedirect.com/science/article/abs/pii/S1566253523001355

[^1_52]: https://huggingface.co/blog/autoformer

[^1_53]: https://www.nature.com/articles/s41598-024-52240-y

[^1_54]: https://github.com/huggingface/blog/blob/main/autoformer.md

[^1_55]: https://journal.hep.com.cn/fcs/EN/10.1007/s11704-025-50947-3

[^1_56]: https://premierscience.com/pjs-25-1179/

[^1_57]: https://arxiv.org/html/2505.08199v1

[^1_58]: https://www.nature.com/articles/s41598-025-95529-2.pdf

[^1_59]: https://data-newbie.tistory.com/943

[^1_60]: https://royalsocietypublishing.org/rsta/article/379/2194/20200209/41189/Time-series-forecasting-with-deep-learning-a

[^1_61]: https://www.sciencedirect.com/science/article/pii/S0306457325002997

[^1_62]: https://velog.io/@barley_15/%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-Autoformer-Decomposition-Transformers-withAuto-Correlation-for-Long-Term-Series-Forecasting

[^1_63]: https://arxiv.org/abs/2501.01087

[^1_64]: https://www.mdpi.com/2504-2289/8/5/48/pdf?version=1715855564

[^1_65]: http://arxiv.org/pdf/2405.16877.pdf

[^1_66]: http://arxiv.org/pdf/2408.09723.pdf

[^1_67]: http://arxiv.org/pdf/2401.03001.pdf

[^1_68]: http://arxiv.org/pdf/2402.05956v5.pdf

[^1_69]: http://arxiv.org/pdf/2305.18382.pdf

[^1_70]: http://arxiv.org/pdf/2306.05880.pdf

[^1_71]: https://arxiv.org/pdf/2412.13769.pdf

[^1_72]: https://arxiv.org/html/2501.01087v1

[^1_73]: https://arxiv.org/pdf/2501.01087.pdf

[^1_74]: https://arxiv.org/html/2408.04245v1

[^1_75]: https://arxiv.org/pdf/2403.14587.pdf

[^1_76]: https://arxiv.org/html/2406.09009v1

[^1_77]: https://arxiv.org/html/2505.00307v2

[^1_78]: https://arxiv.org/html/2312.06786v2

[^1_79]: https://arxiv.org/pdf/2210.02186v2/1000..pdf

[^1_80]: https://arxiv.org/html/2510.07084v1

[^1_81]: https://openreview.net/pdf?id=m2getD1hpk

[^1_82]: https://www.sciencedirect.com/science/article/abs/pii/S1568494625014036

[^1_83]: https://proceedings.mlr.press/v238/ni24a/ni24a.pdf

[^1_84]: https://towardsdatascience.com/transformers-lose-to-linear-models-902164ca5974/

[^1_85]: https://github.com/lichen0620/timesnet

[^1_86]: https://openreview.net/forum?id=JePfAI8fah

[^1_87]: https://kdd-milets.github.io/milets2025/papers/MILETS_2025_paper_17.pdf

[^1_88]: https://ksp.etri.re.kr/ksp/article/file/69078.pdf

[^1_89]: https://dl.acm.org/doi/10.1145/3637528.3671928

[^1_90]: https://dmqa.korea.ac.kr/uploads/seminar/[241011]Channel Independent Strategy for Time Series Forecasting.pdf

[^1_91]: https://link.springer.com/10.1007/s00521-023-09047-1

[^1_92]: https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-024-02433-w

[^1_93]: https://www.mdpi.com/2227-7390/12/23/3666

[^1_94]: https://arxiv.org/abs/2306.07737

[^1_95]: https://www.journal.esrgroups.org/jes/article/view/2699

[^1_96]: https://link.springer.com/10.1007/s10489-023-04729-8

[^1_97]: https://www.ijournalse.org/index.php/ESJ/article/view/2149

[^1_98]: https://www.mdpi.com/2073-8994/16/2/241/pdf?version=1708075023

[^1_99]: https://arxiv.org/html/2503.01157v1

[^1_100]: https://arxiv.org/pdf/2307.01616.pdf

[^1_101]: http://arxiv.org/pdf/2412.13769.pdf

[^1_102]: https://arxiv.org/pdf/2402.16516.pdf

[^1_103]: http://arxiv.org/pdf/2501.01087.pdf

[^1_104]: https://arxiv.org/pdf/2404.06198.pdf

[^1_105]: https://arxiv.org/pdf/2402.01032.pdf
