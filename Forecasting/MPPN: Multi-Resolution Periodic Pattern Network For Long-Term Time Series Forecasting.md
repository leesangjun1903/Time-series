# MPPN: Multi-Resolution Periodic Pattern Network For Long-Term Time Series Forecasting

## 1. 핵심 주장과 주요 기여

MPPN은 장기 시계열 예측(Long-term Time Series Forecasting)을 위해 시계열 데이터의 고유한 다중 해상도 및 다중 주기 패턴을 명시적으로 포착하는 새로운 딥러닝 아키텍처입니다. 논문의 핵심 기여는 다음과 같습니다:[^1_1]

- **다중 해상도 주기 패턴 마이닝(MPPM)**: 컨텍스트 인지 다중 해상도 시맨틱 유닛을 구성하고 다중 주기 패턴 마이닝을 통해 시계열의 고유 패턴을 명시적으로 추출합니다
- **채널 적응 모듈**: 다변량 시계열에서 각 변수가 서로 다른 패턴에 대해 갖는 개별적 특성과 민감도를 적응적으로 모델링합니다
- **예측 가능성 평가**: 엔트로피 기반 방법으로 시계열의 예측 가능성을 사전 평가하고 예측 정확도의 상한선을 제공합니다

9개의 실제 벤치마크 데이터셋에서 MPPN은 최첨단 Transformer 기반, 분해 기반, 샘플링 기반 방법들을 크게 능가하는 성능을 보였습니다.[^1_1]

## 2. 해결하고자 하는 문제

### 2.1 기존 방법의 한계

**분해 기반 방법의 문제점**: Autoformer, FEDformer 등 기존 분해 기반 방법들은 이동 평균을 기반으로 계절성-추세 분해를 수행하지만, 실제 시계열은 여러 요인의 영향을 받아 단순한 계절성-추세 분해로는 해결할 수 없는 복잡한 패턴을 갖습니다.[^1_2][^1_3][^1_1]

**샘플링 기반 방법의 문제점**: SCINet, MICN 등 다운샘플링 기술을 사용하는 방법들은 시계열의 이상치나 노이즈의 영향을 받기 쉬워 추출된 패턴의 품질이 저하되고 장기 예측 성능에 영향을 미칩니다.[^1_1]

**다변량 예측의 문제점**: 대부분의 다변량 시계열 예측 방법은 변수 간 상관관계 모델링에 집중하고 각 변수의 개별적 특성을 무시하여 예측 정확도에 영향을 줍니다.[^1_1]

## 3. 제안하는 방법론 및 수식

### 3.1 문제 정의

다변량 시계열 $X = [x_1, \ldots, x_t, \ldots, x_T]^T \in \mathbb{R}^{T \times C}$가 주어졌을 때, lookback window $L$과 예측 horizon $H$에 대해 다음과 같은 매핑 함수 $F$를 학습합니다:[^1_1]

$$
[x_{t-L}, \ldots, x_{t-1}]^T \xrightarrow{F} [x_t, \ldots, x_{t+H}]^T
$$

### 3.2 예측 가능성 평가

시계열의 엔트로피율(entropy rate)을 기반으로 예측 가능성을 평가합니다. 이산화된 시계열 $X = \{X_1, X_2, \ldots, X_n\}$의 엔트로피율은:[^1_1]

$$
H_u(X) = \lim_{n \to \infty} \frac{1}{n} \sum_{i=1}^{n} H(X_i | X_{i-1}, \ldots, X_1)
$$

Lempel-Ziv 인코딩을 사용한 추정치는:

$$
S = \left( \frac{1}{n} \sum_{i=1}^{n} \Lambda_i \right)^{-1} \ln(n)
$$

여기서 $\Lambda_i$는 위치 $i$에서 시작하는 부분 문자열 중 위치 1부터 $i-1$까지 나타나지 않은 최소 길이입니다. Fano의 부등식을 통해 예측 가능성의 상한 $\Pi_{max}$를 도출합니다:[^1_1]

$$
S \leq H(\Pi_{max}) + (1 - \Pi_{max}) \log_2(N - 1)
$$

여기서 $H(\Pi_{max}) = -\Pi_{max} \log_2(\Pi_{max}) - (1 - \Pi_{max}) \log_2(1 - \Pi_{max})$는 이진 엔트로피 함수입니다.[^1_1]

### 3.3 다중 해상도 패칭

다중 해상도 패턴을 포착하기 위해 비중첩 다중 스케일 합성곱 커널(inception 메커니즘)을 사용합니다:[^1_1]

$$
X_r = \text{Conv1d}(\text{Padding}(X_{in}))_{\text{kernel}=r}
$$

여기서 $r \in \{r_1, r_2, \ldots, r_n\}$은 사전 정의된 시간 해상도이며, 커널 크기와 stride 모두 $r$로 설정합니다. 논문에서는 $[1, 3, 4, 6]$을 기본 해상도로 사용했습니다.[^1_1]

### 3.4 주기 패턴 마이닝

**주기 계산**: Fast Fourier Transform(FFT)을 사용하여 원본 시계열의 주기를 계산합니다:[^1_1]

$$
A = \text{Avg}(\text{Amp}(\text{FFT}(X))), \quad \{f_1, \ldots, f_k\} = \arg \text{Top}_k(A), \quad \text{Period}_i = \left\lfloor \frac{T}{f_i} \right\rfloor
$$

**확장 합성곱(Dilated Convolution)**: 주기 $\text{Period}_i$와 해상도 $r$에 대해 다음과 같이 패턴을 추출합니다:[^1_1]

$$
X_{\text{Period}_i, r} = \text{Truncate}\left(\text{Conv1d}(X_r)_{\text{kernel}=\lfloor L/\text{Period}_i \rfloor, \text{dilation}=\lfloor \text{Period}_i/r \rfloor}\right)
$$

**패턴 결합**: 동일한 주기에 대해 모든 해상도의 패턴을 연결하고, 최종적으로 다중 주기 패턴을 결합합니다:[^1_1]

$$
X_{\text{Period}_i} = \|_{j=1}^{n} X_{\text{Period}_i, r_j}, \quad X_{\text{Pattern}} = \|_{i=1}^{k} X_{\text{Period}_i}
$$

여기서 $P = \sum_{i=1}^{k} \sum_{j=1}^{n} \lfloor \text{Period}_i / r_j \rfloor$는 총 패턴 모드 수입니다.[^1_1]

### 3.5 채널 적응 모듈

학습 가능한 변수 임베딩 행렬 $E \in \mathbb{R}^{C \times P}$를 정의하고, sigmoid 함수로 활성화한 후 추출된 패턴과 브로드캐스팅 곱셈을 수행합니다:[^1_1]

$$
X_{\text{AdpPattern}} = X_{\text{Pattern}} \cdot \text{sigmoid}(E)
$$

이를 통해 각 채널(변수)이 서로 다른 패턴에 대해 갖는 민감도를 적응적으로 학습합니다.[^1_1]

### 3.6 출력 층

완전 연결 층을 통해 최종 예측을 생성합니다:[^1_1]

$$
X_{out} = \text{Reshape}(X_{\text{AdpPattern}}) \cdot W + b
$$

여기서 $W \in \mathbb{R}^{(PD) \times H}$, $b \in \mathbb{R}^H$는 학습 가능한 파라미터이며, Mean Squared Error(MSE)를 손실 함수로 사용합니다.[^1_1]

## 4. 모델 구조

MPPN 아키텍처는 다음 세 가지 주요 구성 요소로 이루어져 있습니다:[^1_1]

1. **Multi-resolution Periodic Pattern Mining (MPPM)**: 입력 시계열을 다중 해상도 패치로 분할하고 주기적 패턴을 추출
2. **Channel Adaptive Module**: 각 변수의 패턴 인식 가중치를 학습
3. **Output Layer**: 최종 예측 생성

전체 프로세스는 선형 계산 복잡도 $O(L)$를 유지하면서도 시계열의 복잡한 주기적 패턴을 효과적으로 포착합니다.[^1_1]

## 5. 성능 향상

### 5.1 주요 성능 지표

**Weather 데이터셋** (예측 길이 96): MSE를 15.29% 개선 (0.170 → 0.144)[^1_1]

**Traffic 데이터셋** (예측 길이 96): MSE를 6.07% 개선 (0.412 → 0.387)[^1_1]

**Electricity 데이터셋** (예측 길이 96): MSE를 6.43% 개선 (0.140 → 0.131)[^1_1]

**ILI 데이터셋** (예측 길이 24): MSE를 21.23% 개선 (2.280 → 1.796)[^1_1]

**ETTm1 데이터셋** (예측 길이 96): MSE를 4.65% 개선 (0.301 → 0.287)[^1_1]

### 5.2 Ablation Study 결과

- **다중 해상도 제거 시**: Weather 데이터셋에서 MSE가 0.144에서 0.165로 증가 (약 14.6% 성능 저하)[^1_1]
- **주기 샘플링 제거 시**: 성능 저하가 관찰되어 주기 패턴 마이닝의 중요성 확인[^1_1]
- **채널 적응 제거 시**: 특히 Weather 같은 이질적인 변수를 가진 데이터셋에서 눈에 띄는 성능 저하[^1_1]


### 5.3 효율성

MPPN은 선형 계산 복잡도를 유지하면서 DLinear보다 약간 높은 학습 시간을 보이지만, Transformer 기반 모델들(FEDformer, Autoformer, Crossformer)에 비해 훨씬 낮은 계산 비용을 달성했습니다.[^1_1]

## 6. 한계점

### 6.1 예측할 수 없는 변동성

MPPN은 역사적 시계열에서 패턴(주기성, 추세)을 효과적으로 포착하지만, 실제 시계열은 다양한 외부 요인의 영향을 받아 복잡한 패턴이나 이전에 보지 못한 변동을 보일 수 있습니다. 예를 들어:[^1_1]

- 교통 사고로 인한 갑작스러운 교통 체증
- 지진, 허리케인 같은 극단적 기상 재해
- 고객의 행동 패턴 변화(예: 월드컵 시청으로 인한 전력 사용 시간 지연)


### 6.2 외부 요인의 모델링 부족

현재 MPPN은 lookback window의 데이터에서 이러한 변동을 빠르게 포착할 수 있지만, 이를 예측하기는 어렵습니다. 논문은 향후 연구에서 지식 그래프를 도입하여 외부 요인에 대한 지식을 예측 모델에 통합할 계획을 밝혔습니다.[^1_1]

### 6.3 고정된 주기 가정

FFT를 통한 주기 추출 방식은 시계열이 명확한 주기성을 가질 때 효과적이지만, Exchange-Rate나 ETTh2처럼 명확한 주기 패턴이 없는 데이터에서는 상대적으로 제한적입니다.[^1_1]

## 7. 모델의 일반화 성능 향상 가능성

### 7.1 예측 가능성 평가의 의의

MPPN은 엔트로피 기반 방법을 통해 시계열의 예측 가능성을 사전에 평가하고 상한선을 제공합니다. 9개 벤치마크 데이터셋에서 모든 예측 가능성이 0.85 이상으로 나타나 높은 예측 가능성을 보였습니다. 이는:[^1_1]

- 모델 학습 전 데이터의 예측 가능성 판단 가능
- 랜덤 워크 같은 낮은 예측 가능성을 가진 시계열 필터링 가능
- 예측 문제의 유의미성 사전 평가 가능


### 7.2 채널 적응을 통한 일반화

채널 적응 모듈은 각 변수의 고유한 특성을 학습하여 다변량 시계열에서 일반화 성능을 향상시킵니다. ETTh1 데이터셋의 히트맵 분석 결과, 대부분의 변수는 3번째와 4번째 패턴에 크게 영향을 받지만, 'LULF'(Low Useful Load)는 예외적인 패턴을 보였습니다. 이는 모델이 데이터로부터 변수별 특성을 자동으로 학습할 수 있음을 의미합니다.[^1_1]

### 7.3 다중 해상도 접근의 강건성

다중 해상도 패칭은 시계열의 서로 다른 스케일의 패턴을 동시에 포착하여 일반화 성능을 향상시킵니다. 큰 해상도에서는 강한 주기성을, 작은 해상도에서는 세밀한 변동을 포착하여 시계열의 다층적 구조를 효과적으로 모델링합니다.[^1_1]

### 7.4 오류 바 분석 결과

3번의 독립적인 실행에서 표준편차(Std.)는 평균값(Mean)의 3% 이내로 나타나 MPPN이 다양한 초기화 설정에서 강건한 성능을 보임을 확인했습니다.[^1_1]

## 8. 앞으로의 연구에 미치는 영향

### 8.1 명시적 패턴 추출의 중요성

MPPN은 시계열 데이터의 고유한 속성(주기성, 다중 해상도)을 명시적으로 활용하는 것이 중요함을 입증했습니다. 이는 최근 연구 동향과 일치합니다:[^1_1]

- **Linear 모델의 재부상**: DLinear, NLinear이 Transformer를 능가하는 성능을 보인 것은 시계열의 고유한 속성 활용의 중요성을 시사합니다[^1_1]
- **명시적 분해의 필요성**: 단순한 attention 메커니즘보다 시계열 특화된 구조가 더 효과적일 수 있습니다


### 8.2 채널 독립성 vs. 채널 적응

MPPN의 채널 적응 모듈은 최근 논문들이 지적한 "공간적 구별 불가능성(spatial indistinguishability)" 문제에 대한 해답을 제시합니다. 채널 독립(CI) 전략은 성능이 좋지만, 과거 관측값이 유사한 변수들을 동일한 예측으로 붕괴시키는 단점이 있습니다. MPPN은 학습 가능한 변수 식별자를 통해 이 문제를 해결하는 방향을 제시했습니다.[^1_4]

### 8.3 Foundation Model로의 확장 가능성

논문은 향후 연구 방향으로 범용 시계열 분석을 위한 task-general 모델 구축을 제시했습니다. 이는 최근 시계열 Foundation Model 연구와 일치합니다:[^1_5][^1_1]

- 다양한 예측 horizon에 일반화 가능한 단일 모델
- 제로샷 성능 개선
- 사전 학습된 패턴 지식의 활용


## 9. 앞으로 연구 시 고려할 점

### 9.1 외부 지식 통합

MPPN의 한계를 극복하기 위해 다음이 필요합니다:[^1_1]

- **지식 그래프 통합**: 외부 요인(날씨, 이벤트, 정책)을 모델에 통합
- **멀티모달 학습**: 텍스트, 이미지 등 다양한 데이터 소스 활용
- **인과 추론**: 단순 상관관계를 넘어 인과관계 파악


### 9.2 적응적 주기 탐지

고정된 FFT 기반 주기 탐지를 개선하기 위해:

- **시간 변화 주기**: 시간에 따라 변하는 주기 패턴 모델링
- **웨이블릿 변환**: 다중 스케일 시간-주파수 분석
- **학습 가능한 주기 추출**: 데이터 기반으로 주기를 자동 학습


### 9.3 해석 가능성 강화

MPPN은 패턴 시각화를 통해 해석 가능성을 제공하지만, 더 나아가:

- **주의 메커니즘 시각화**: 어떤 과거 시점이 예측에 중요한지 분석
- **패턴 라벨링**: 추출된 패턴에 의미 있는 라벨 부여
- **반사실적 설명**: "만약 ~했다면" 시나리오 분석


### 9.4 비정상성 처리

시계열의 비정상성(non-stationarity)을 더 효과적으로 다루기 위해:

- **정규화 기법**: RevIN, Batch Normalization 등의 통합
- **적응적 정규화**: 각 예측 시점마다 다른 정규화 전략
- **분포 이동 감지**: 훈련-테스트 분포 차이 자동 감지 및 보정


## 10. 2020년 이후 관련 최신 연구 비교 분석

### 10.1 Transformer 기반 모델 (2020-2022)

**Informer (2021)**: ProbSparse attention으로 $O(L \log L)$ 복잡도 달성. 그러나 permutation-invariant attention으로 시간 정보 손실 문제가 있습니다.[^1_2][^1_1]

**Autoformer (2021)**: Auto-correlation 메커니즘과 분해 블록 도입. 계절성-추세 분해를 통해 장기 추세 포착에 강점을 보였지만, 복잡한 시계열에서는 제한적입니다.[^1_3][^1_2][^1_1]

**FEDformer (2022)**: 주파수 도메인에서 Transformer 연산 수행, 희소 Fourier 성분 선택으로 선형 복잡도 달성. MPPN과 유사하게 주파수 분석을 활용하지만, MPPN이 더 명시적인 주기 패턴 추출을 수행합니다.[^1_6][^1_3][^1_1]

### 10.2 Linear 기반 모델의 반격 (2022-2023)

**DLinear/NLinear (2023)**: 단순 선형 모델이 Transformer를 능가함을 입증. 이는 시계열 특화 귀납적 편향의 중요성을 재확인했습니다. MPPN은 이러한 통찰을 바탕으로 주기성이라는 명시적 귀납적 편향을 추가했습니다.[^1_1]

### 10.3 CNN 기반 모델 (2022-2023)

**SCINet (2022)**: 재귀적 다운샘플-합성곱-상호작용 구조. MPPN과 유사하게 다중 해상도를 활용하지만, MPPN은 주기성을 명시적으로 고려합니다.[^1_1]

**MICN (2023)**: 다중 스케일 분해와 다운샘플 합성곱. MPPN보다 복잡한 구조를 가지지만, 주기 패턴 마이닝은 수행하지 않습니다.[^1_1]

### 10.4 최신 발전 (2024-2026)

**PatchTST (2023-2024)**: 채널 독립 전략 채택, 패칭 메커니즘 도입. MPPN의 패칭과 유사하지만, MPPN은 해상도 기반 패칭을 수행합니다.[^1_4]

**SparseTSF (2024)**: Cross-Period Sparse Forecasting으로 1k 파라미터만으로 LTSF 달성. MPPN과 유사하게 주기성을 활용하지만, 극단적인 경량화에 초점을 맞췄습니다.[^1_7]

**Mamba 기반 모델 (2024)**: Bi-Mamba+, ISMRNN 등 State Space Model(SSM)을 시계열에 적용. 선형 복잡도로 장기 의존성을 포착하지만, 시계열의 주기성을 명시적으로 다루지는 않습니다.[^1_8][^1_9]

**KAN 기반 모델 (2026)**: HaKAN은 Kolmogorov-Arnold Networks를 활용하여 해석 가능한 활성화 함수 학습. MPPN보다 더 표현력이 높지만 계산 비용이 증가할 수 있습니다.[^1_10]

**MoHETS (2026)**: Mixture-of-Heterogeneous-Experts로 다양한 시간적 동역학 포착. Fourier 기반 전문가를 사용하여 MPPN과 유사한 주파수 분석 수행하지만, sparse MoE로 확장성을 개선했습니다.[^1_11]

**Evolutionary Forecasting (2026)**: 직접 예측의 그래디언트 충돌 문제 해결. MPPN의 손실 함수 설계에 참고할 수 있는 최적화 기법을 제시합니다.[^1_12]

### 10.5 MPPN의 차별점

MPPN은 2020년 이후 연구들과 비교하여 다음과 같은 차별점을 갖습니다:

1. **명시적 주기 추출**: FFT 기반으로 top-k 주기를 자동 추출하고 dilated convolution으로 패턴 마이닝[^1_1]
2. **다중 해상도 통합**: 단일 해상도가 아닌 여러 해상도에서 패턴을 동시 포착[^1_1]
3. **채널 적응**: 학습 가능한 변수 임베딩으로 각 변수의 패턴 민감도 모델링[^1_1]
4. **예측 가능성 평가**: 엔트로피 기반으로 사전 평가 수행[^1_1]
5. **선형 복잡도**: $O(L)$ 계산 복잡도로 효율성 유지[^1_1]

### 10.6 향후 통합 가능성

최신 연구들의 기법을 MPPN에 통합하면 성능을 더욱 향상시킬 수 있습니다:

- **MoE 구조**: 다양한 주기 패턴에 대한 전문가 네트워크 구성[^1_11]
- **SSM 통합**: Mamba 구조로 장기 의존성 포착 강화[^1_8]
- **학습 목적 함수**: KMB-DF의 커널화된 모멘트 균형 또는 Evolutionary Forecasting의 블록 학습[^1_13][^1_12]
- **Foundation Model**: 사전 학습된 패턴 지식 활용[^1_5]

***

<span style="display:none">[^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39]</span>

<div align="center">⁂</div>

[^1_1]: 2306.06895v1.pdf

[^1_2]: https://arxiv.org/pdf/2106.13008.pdf

[^1_3]: http://arxiv.org/abs/2201.12740

[^1_4]: https://onlinelibrary.wiley.com/doi/10.1002/for.70105

[^1_5]: https://arxiv.org/pdf/2310.10688.pdf

[^1_6]: https://arxiv.org/pdf/2201.12740.pdf

[^1_7]: https://arxiv.org/pdf/2405.00946.pdf

[^1_8]: https://arxiv.org/pdf/2404.15772.pdf

[^1_9]: http://arxiv.org/pdf/2407.10768.pdf

[^1_10]: https://www.semanticscholar.org/paper/c520bcb08c9572e50b92b9463e8b2a5fc73b2045

[^1_11]: https://www.semanticscholar.org/paper/16a6d4826ca05e107b5c435e57a203c5b19b852c

[^1_12]: https://www.emergentmind.com/topics/long-term-time-series-forecasting-ltsf

[^1_13]: https://www.arxiv.org/pdf/2602.00717.pdf

[^1_14]: https://ieeexplore.ieee.org/document/11007288/

[^1_15]: https://ieeexplore.ieee.org/document/11353101/

[^1_16]: https://www.semanticscholar.org/paper/ba60b72a7e94eb07e45983e74860e0b16d97e2c3

[^1_17]: https://www.semanticscholar.org/paper/b39ababa9424797e27f43eb4e3a8b678d8ccffdf

[^1_18]: http://www.emerald.com/ijesm/article/15/2/385-396/121196

[^1_19]: http://link.springer.com/10.1007/s00500-020-04855-2

[^1_20]: https://www.ewadirect.com/proceedings/ace/article/view/31299

[^1_21]: https://arxiv.org/html/2409.17703

[^1_22]: https://arxiv.org/pdf/2411.05793.pdf

[^1_23]: https://arxiv.org/html/2309.15946

[^1_24]: https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0295803

[^1_25]: https://arxiv.org/html/2503.23604v4

[^1_26]: https://pdfs.semanticscholar.org/fb76/33dcdeb50f0abdd840d44e23e4afa44a2fde.pdf

[^1_27]: https://journals.plos.org/plosone/article/file?type=printable\&id=10.1371%2Fjournal.pone.0295803

[^1_28]: https://arxiv.org/pdf/2503.23604.pdf

[^1_29]: https://arxiv.org/html/2503.23604v2

[^1_30]: https://arxiv.org/html/2505.20048v1

[^1_31]: https://arxiv.org/pdf/2506.22552.pdf

[^1_32]: https://www.sciencedirect.com/science/article/abs/pii/S1568494625005198

[^1_33]: https://github.com/thuml/Time-Series-Library

[^1_34]: https://www.nature.com/articles/s41467-025-63786-4

[^1_35]: https://www.sciencedirect.com/science/article/abs/pii/S0016003225000845

[^1_36]: https://www.amazon.science/publications/bridging-self-attention-and-time-series-decomposition-for-periodic-forecasting

[^1_37]: https://arxiv.org/html/2411.05793v1

[^1_38]: https://huggingface.co/blog/autoformer

[^1_39]: https://arxiv.org/html/2503.10198v1

