
# Proactive Model Adaptation Against Concept Drift for Online Time Series Forecasting 

## 1. 핵심 주장과 주요 기여

이 논문은 온라인 시계열 예측에서 기존 방법들이 간과한 중요한 문제를 발견했습니다. 예측 horizon $H$ 만큼의 피드백 지연(feedback delay)으로 인해 학습 샘플과 테스트 샘플 사이에 temporal gap이 발생하며, 이 간격에서 concept drift가 발생하여 모델이 구식 개념에 적응하게 됩니다.[^1_1]

주요 기여는 다음과 같습니다:[^1_1]

- **Proceed** 프레임워크 제안: 테스트 샘플 예측 전에 concept drift를 추정하고 모델 파라미터를 사전적으로(proactively) 조정
- 합성 concept drift 기반 학습을 통한 일반화 능력 향상
- 5개 실제 데이터셋에서 평균 21.9% 성능 향상, 기존 온라인 학습 방법 대비 10.9% 우수한 성능 달성


## 2. 상세 분석

### 해결하고자 하는 문제

기존 온라인 학습 방법(FSNet, OneNet 등)은 시간 $t$에서 사용 가능한 최신 학습 샘플이 $\mathcal{D}\_t^- = \{(\mathbf{X}\_{t'}, \mathbf{Y}_{t'}) \mid t' \leq t-H\}$ 임을 간과합니다. 실증 분석 결과, Practical 전략(실제 사용 가능한 데이터 사용)과 Optimal 전략(미래 정보 유출) 간 평균 107%의 성능 차이가 발생했습니다.[^1_1]

### 제안하는 방법

Proceed는 4단계로 구성됩니다:[^1_1]

**1) 온라인 Fine-tuning**

최근 학습 샘플 $(\mathbf{X}\_{t-H}, \mathbf{Y}_{t-H})$로 모델 파라미터를 gradient descent로 업데이트:

$$
\boldsymbol{\theta}_{t-H} \leftarrow \boldsymbol{\theta}_{t-H-1} - \eta \nabla_{\boldsymbol{\theta}} \|\hat{\mathbf{Y}}_{t-H} - \mathbf{Y}_{t-H}\|_2^2
$$

**2) Concept Drift 추정**

두 개의 concept encoder를 사용하여 잠재 표현을 추출합니다:[^1_1]

$$
\mathbf{c}_{t-H} = \mathcal{E}(\mathcal{D}_t^-) = \text{Average}\left(\{\text{MLP}(\mathbf{X}_{t-H}^{(i)} \| \mathbf{Y}_{t-H}^{(i)})\}_{i=1}^N\right) \in \mathbb{R}^{d_c}
$$

$$
\mathbf{c}_t = \mathcal{E}'(\mathbf{X}_t) = \text{Average}\left(\{\text{MLP}'(\mathbf{X}_t^{(i)})\}_{i=1}^N\right) \in \mathbb{R}^{d_c}
$$

Concept drift는 두 표현의 차이로 계산됩니다:

$$
\boldsymbol{\delta}_{t-H \to t} = \mathbf{c}_t - \mathbf{c}_{t-H}
$$

**3) Proactive Model Adaptation**

Adaptation generator $\mathcal{G}$는 bottleneck 레이어를 통해 적응 계수를 생성합니다:[^1_1]

$$
[\boldsymbol{\alpha}_t^{(\ell)}, \boldsymbol{\beta}_t^{(\ell)}] = \mathbf{W}_2^{(\ell)\top}\left(\sigma\left(\mathbf{W}_1^{(\ell)\top} \boldsymbol{\delta}_{t-H \to t} + \mathbf{b}^{(\ell)}\right)\right) + 1
$$

여기서 $\mathbf{W}\_1^{(\ell)} \in \mathbb{R}^{r \times d_c}$, $\mathbf{W}\_2^{(\ell)} \in \mathbb{R}^{(d_{in} + d_{out}) \times r}$이며, $r$은 작은 bottleneck 차원입니다.[^1_1]

적응된 파라미터는 다음과 같이 계산됩니다:

$$
\hat{\boldsymbol{\theta}}_t^{(\ell)} = \left(\boldsymbol{\alpha}_t^{(\ell)\top} \boldsymbol{\beta}_t^{(\ell)}\right) \odot \boldsymbol{\theta}_{t-H}^{(\ell)}
$$

**4) 온라인 예측**

적응된 모델로 예측 수행:

$$
\hat{\mathbf{Y}}_t = \mathcal{F}(\mathbf{X}_t; \hat{\boldsymbol{\theta}}_t)
$$

### 모델 구조

Proceed의 아키텍처는 세 가지 핵심 컴포넌트로 구성됩니다:[^1_1]

1. **Concept Encoder $\mathcal{E}$**: 학습 샘플의 concept 표현 추출
2. **Concept Encoder $\mathcal{E}'$**: 테스트 샘플의 concept 표현 추출
3. **Adaptation Generator $\mathcal{G}$**: Bottleneck 레이어를 통해 파라미터 조정 계수 생성

파라미터 복잡도는 $O(r(\mathcal{L} + d_c + d_{in} + d_{out}))$로, 완전 연결 레이어의 $O(\mathcal{L}d_c d_{in} d_{out})$에 비해 크게 감소합니다.[^1_1]

### 성능 향상

실험 결과는 다음과 같습니다:[^1_1]


| 모델 | 데이터셋 | 개선율 (vs GD) |
| :-- | :-- | :-- |
| TCN + Proceed | ETTh2 (H=24) | 9.8% |
| TCN + Proceed | Weather (H=96) | 26.9% |
| PatchTST + Proceed | ETTh2 (H=24) | 17.1% |
| iTransformer + Proceed | ECL (H=24) | 3.9% |

Proceed는 평균적으로 다음을 달성했습니다:

- FSNet 대비 12.5% 개선
- OneNet 대비 13.6% 개선
- SOLID++ 대비 6.7% 개선


### 한계점

논문에서 명시적으로 언급된 한계는 다음과 같습니다:[^1_1]

1. **메모리 제약**: 온라인 환경에서 제한된 GPU 메모리로 인해 부분 ground truth를 활용한 학습이 어려움
2. **Out-of-Distribution Concept**: 완전히 새로운 concept에 대한 적응 보장 부족
3. **Horizon 길이 의존성**: Horizon이 길수록 temporal gap이 커져 성능 저하 가능성
4. **계산 비용**: Concept encoder와 adaptation generator의 추가 계산 비용 발생

## 3. 일반화 성능 향상 가능성

Proceed는 여러 메커니즘을 통해 일반화 성능을 향상시킵니다:[^1_1]

### 합성 Concept Drift 학습

역사적 데이터를 무작위로 섞어 다양한 concept drift를 합성합니다. 시각화 분석 결과, 온라인 데이터의 concept drift $\boldsymbol{\delta}\_{4 \to 6}$, $\boldsymbol{\delta}\_{5 \to 7}$는 학습 데이터의 $\boldsymbol{\delta}\_{1 \to 4}$, $\boldsymbol{\delta}_{2 \to 3}$와 유사한 패턴을 보입니다.[^1_1]

### 이론적 근거

Lipschitz 연속성을 가정하면, 충분히 작은 $\Delta > 0$에 대해 다음이 성립합니다:[^1_1]

$$
L_{upper} \|\hat{\boldsymbol{\theta}}_t - \boldsymbol{\theta}_t^*\| < L_{lower} \|\boldsymbol{\theta}_{t-H} - \boldsymbol{\theta}_t^*\|
$$

따라서:

$$
\|\mathcal{F}(\mathbf{X}_t; \hat{\boldsymbol{\theta}}_t) - \mathbf{Y}_t\| < \|\mathcal{F}(\mathbf{X}_t; \boldsymbol{\theta}_{t-H}) - \mathbf{Y}_t\|
$$

### Ablation Study 결과

일반화에 기여하는 주요 요소들:[^1_1]

- Concept drift 기반 적응 (vs 단순 concept 기반): PatchTST에서 6.5% 개선
- 별도의 encoder $\mathcal{E}$와 $\mathcal{E}'$ 사용: 7.8% 개선
- 레이어 간 가중치 공유: 파라미터 효율성 유지하며 0.4-1.9% 성능 유지


## 4. 향후 연구에 미치는 영향과 고려사항

### 연구 영향

**1. 피드백 지연 문제의 재조명**

Proceed는 시계열 예측에서 간과되었던 피드백 지연 문제를 체계적으로 분석하고 해결책을 제시했습니다. 이는 향후 온라인 학습 연구의 새로운 방향을 제시합니다.[^1_1]

**2. Proactive Adaptation 패러다임**

기존의 reactive한 모델 적응 방식에서 proactive한 방식으로의 전환을 제안했습니다. 이는 다른 시계열 응용 분야(금융, 에너지, 의료)로 확장 가능합니다.[^1_1]

**3. 모델 비종속적(Model-Agnostic) 프레임워크**

TCN, PatchTST, iTransformer 등 다양한 모델에 적용 가능한 일반적 프레임워크를 제공합니다.[^1_1]

### 향후 연구 시 고려사항

**1. 다중 샘플 활용**

SOLID의 데이터 샘플링 기법과 Proceed의 proactive adaptation을 결합하는 연구가 필요합니다. 현재 Proceed는 단일 학습 샘플만 사용합니다.[^1_1]

**2. 부분 Ground Truth 활용**

$\{\tilde{\mathbf{Y}}\_{t-i}\}_{i=1}^{H-1}$의 부분 ground truth를 효율적으로 활용하는 방법 연구가 필요합니다. GPU 메모리 제약을 고려한 경량화 기법 개발이 요구됩니다.[^1_1]

**3. 극단적 Concept Drift 처리**

완전히 새로운 concept에 대한 적응 능력 향상이 필요합니다. Meta-learning 기법과의 결합을 고려할 수 있습니다.[^1_1]

**4. Rolling Retraining과의 통합**

장기적인 성능 유지를 위해 주기적 재학습과 일일 온라인 업데이트를 결합하는 전략 개발이 필요합니다.[^1_1]

## 5. 2020년 이후 관련 최신 연구 비교 분석

### Concept Drift 처리 방법론 비교

| 연구 | 연도 | 핵심 접근법 | Proceed와의 차이점 |
| :-- | :-- | :-- | :-- |
| **DDG-DA** [^1_2] | 2022 | 데이터 분포 생성 및 예측 | 사후적 데이터 생성, 피드백 지연 미고려 |
| **OneNet** [^1_3] | 2023 | 온라인 앙상블 | 정보 유출 문제, temporal gap 간과 |
| **FSNet** [^1_1] | 2023 | Gradient 기반 빠른 적응 | 실제로는 $(\mathbf{X}\_{t-1}, \mathbf{Y}_{t-1})$ 사용, 비실용적 |
| **MemDA** [^1_4] | 2023 | 메모리 기반 drift 적응 | 도시 시계열 특화, 일반적 피드백 지연 미해결 |
| **SOLID** [^1_1] | 2024 | Context 유사도 기반 샘플 선택 | Heuristic 기반, 사전적 적응 부재 |
| **Proceed** [^1_1] | 2025 | Proactive concept drift 추정 및 적응 | 피드백 지연 명시적 해결, 합성 drift 학습 |

### 최근 보완 연구 (2024-2025)

**1. LEAF (IJCAI 2025)**[^1_5][^1_6]

LEAF는 concept drift를 macro-drift(안정적 장기 변화)와 micro-drift(급격한 단기 변동)로 분류하고, 2단계 메타학습 프레임워크를 제안합니다. Proceed와의 차이점:[^1_5]

- LEAF: Extrapolation + adjustment 모듈로 macro/micro drift 분리 처리
- Proceed: 단일 통합 프레임워크로 temporal gap의 drift 처리
- 상호 보완 가능성: LEAF의 drift 분류와 Proceed의 proactive adaptation 결합

**2. CEP (Continuous Evolution Pool)**[^1_7]

2025년 5월에 제안된 CEP는 recurring concept drift에 초점을 맞춥니다:[^1_7]

- 특화된 forecaster 풀을 유지하며 통계적 "gene"으로 concept 식별
- 원시 샘플 저장 없이 프라이버시 보존
- Proceed 대비 20% 이상 성능 향상 주장
- 한계: 메모리 제약하의 모델 관리 복잡도

**3. ShifTS (Shift-aware Time Series)**[^1_8]

Temporal shift와 concept drift를 구분하여 처리합니다:[^1_8]

- Soft attention masking으로 불변 패턴 추출
- Exogenous feature의 인과 관계 활용
- Proceed와 직교적: 데이터 정규화 + 모델 적응 결합 가능

**4. Detect-then-Adapt 접근법**[^1_9][^1_10]

2024년 3월 연구는 drift 탐지 후 적응하는 전략을 제안합니다:[^1_9]

- 누적 concept drift로 인한 지속적 성능 저하 문제 지적
- Proceed는 사전적 적응으로 이 문제 완화
- 탐지 기반 방법은 reactive, Proceed는 proactive


### 일반화 및 분포 이동 연구

**1. Out-of-Distribution 일반화**[^1_11]

2025년 3월 서베이는 시계열 OOD 일반화 방법론을 체계적으로 정리합니다:[^1_11]

- Temporal Koopman networks, continuous-time optimal transport
- RevIN  등 비정상성 제거 기법[^1_1]
- Proceed와의 연결: OOD concept에 대한 일반화가 핵심 과제

**2. Temporal Generalization Reality Check**[^1_12]

2025년 연구는 continual learning의 중요성을 강조합니다:[^1_12]

- CL이 forward transfer를 유의미하게 개선
- 파라미터 궤적의 smoothness 유지 중요
- Proceed의 합성 drift 학습과 유사한 철학


### 실무 응용 연구

**1. 건물 부하 예측 (AugPlug)**[^1_13]

Automated data augmentation으로 updating set 개선, 29.37% 성능 향상. Proceed의 합성 drift 생성과 유사한 접근법입니다.[^1_13]

**2. 풍력 발전 예측**[^1_14]

"Dynamic matching and online modeling" 전략으로 1.18-4.32% 개선. 특징 유사도 기반 샘플 선택은 SOLID와 유사합니다.[^1_14]

**3. LSTM-ADWIN**[^1_15]

다변량 시계열에 ADWIN drift 탐지와 LSTM 결합. 탐지 지연 ≤5 단계 달성했으나, proactive 적응은 부재합니다.[^1_15]

### 종합 평가

Proceed의 주요 기여는 **피드백 지연 문제의 체계적 분석**과 **proactive adaptation 패러다임 제시**입니다. 2024-2025년 후속 연구들(LEAF, CEP, ShifTS)은 Proceed의 개념을 확장하거나 보완하고 있으며, 다음 방향으로 발전하고 있습니다:[^1_1]

1. **Drift 세분화**: Macro/micro drift, temporal/concept shift 구분
2. **메모리 효율성**: 경량 표현, 모델 풀 관리
3. **프라이버시 보존**: 원시 데이터 저장 회피
4. **도메인 특화**: 금융, 에너지, 건물 등 특정 응용 최적화

향후 연구는 이러한 접근법들을 통합하여 더욱 강건하고 효율적인 온라인 시계열 예측 시스템을 구축하는 방향으로 진행될 것으로 예상됩니다.
<span style="display:none">[^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39]</span>

<div align="center">⁂</div>

[^1_1]: 2412.08435v5.pdf

[^1_2]: https://arxiv.org/pdf/2201.04038.pdf

[^1_3]: https://arxiv.org/pdf/2309.12659.pdf

[^1_4]: https://arxiv.org/pdf/2309.14216.pdf

[^1_5]: https://www.ijcai.org/proceedings/2025/542

[^1_6]: https://www.ijcai.org/proceedings/2025/0542.pdf

[^1_7]: https://arxiv.org/abs/2506.14790

[^1_8]: https://arxiv.org/html/2510.14814v1

[^1_9]: https://arxiv.org/pdf/2403.14949.pdf

[^1_10]: https://arxiv.org/html/2403.14949v1

[^1_11]: https://arxiv.org/html/2503.13868v1

[^1_12]: https://arxiv.org/html/2509.23487v1

[^1_13]: https://dl.acm.org/doi/10.1145/3737644

[^1_14]: https://ieeexplore.ieee.org/document/10620613/

[^1_15]: https://www.eurekaselect.com/246587/article

[^1_16]: https://arxiv.org/html/2602.03981v1

[^1_17]: https://www.arxiv.org/pdf/2602.03981.pdf

[^1_18]: https://arxiv.org/html/2508.11004

[^1_19]: https://arxiv.org/pdf/2508.11004.pdf

[^1_20]: https://arxiv.org/pdf/2601.20819.pdf

[^1_21]: https://arxiv.org/html/2412.08435v3

[^1_22]: https://arxiv.org/pdf/2601.17933.pdf

[^1_23]: https://arxiv.org/abs/2412.08435

[^1_24]: https://sol.sbc.org.br/index.php/eniac/article/view/38818

[^1_25]: https://link.springer.com/10.1007/s10489-025-07024-w

[^1_26]: https://efa.am/index.php/efa/article/view/127

[^1_27]: https://ieeexplore.ieee.org/document/9377947/

[^1_28]: https://www.clinexprheumatol.org/abstract.asp?a=22525

[^1_29]: https://iieta.org/download/file/fid/112347

[^1_30]: http://arxiv.org/pdf/2412.08435.pdf

[^1_31]: https://www.iieta.org/download/file/fid/116524

[^1_32]: https://arxiv.org/pdf/2409.16843.pdf

[^1_33]: https://arxiv.org/html/2412.08435v5

[^1_34]: https://github.com/SJTU-DMTai/OnlineTSF

[^1_35]: https://dl.acm.org/doi/10.1145/3690624.3709210

[^1_36]: https://www.themoonlight.io/en/review/proactive-model-adaptation-against-concept-drift-for-online-time-series-forecasting

[^1_37]: https://arxiv.org/html/2412.08435v4

[^1_38]: http://arxiv.org/abs/2412.08435

[^1_39]: https://arxiv.org/html/2601.12931v1

