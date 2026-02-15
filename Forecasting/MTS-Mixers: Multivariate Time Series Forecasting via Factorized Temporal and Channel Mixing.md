# MTS-Mixers: Multivariate Time Series Forecasting via Factorized Temporal and Channel Mixing

## 1. 핵심 주장과 주요 기여

MTS-Mixers는 다변량 시계열 예측에서 Transformer 기반 모델의 attention 메커니즘이 필수적이지 않다는 것을 입증하고, **시간(temporal)과 채널(channel) 차원을 분리하여 모델링**하는 새로운 패러다임을 제시합니다. 논문의 핵심 기여는 다음과 같습니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3ceba5f7-151a-486c-be28-49e776864780/2302.04501v1.pdf)

- **Attention의 비필수성 입증**: Self-attention을 Fourier Transform으로 대체하거나 제거해도 예측 성능이 유지되거나 향상됨을 실험적으로 증명
- **저랭크(Low-rank) 특성 활용**: 시계열 데이터의 시간적·채널 간 redundancy를 factorization을 통해 효율적으로 처리
- **SOTA 성능 달성**: 여러 실세계 데이터셋에서 평균 15.4% MSE, 12.9% MAE 감소를 달성하며 높은 효율성 확보 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3ceba5f7-151a-486c-be28-49e776864780/2302.04501v1.pdf)

## 2. 상세 분석

### 해결하고자 하는 문제

**1) Transformer의 attention 메커니즘 재평가**
기존 Transformer 기반 모델들(Informer, Autoformer, FEDformer)은 긴 시계열의 temporal dependency를 포착하는 데 attention이 필수적이라고 가정했으나, MTS-Mixers는 다음 세 가지 문제점을 지적합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3ceba5f7-151a-486c-be28-49e776864780/2302.04501v1.pdf)

- Attention이 temporal dependency 포착에 필수적이지 않음
- Temporal과 channel interaction의 얽힘(entanglement)과 redundancy가 성능 저하를 유발
- 입력과 예측 시퀀스 간 매핑 모델링의 중요성

**2) 데이터의 저랭크 특성 미활용**
시계열 데이터 $X \in \mathbb{R}^{n \times c}$는 일반적으로 $\text{rank}(X) \ll \min(n, c)$의 저랭크 특성을 갖지만, 기존 모델들은 이를 효과적으로 활용하지 못했습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3ceba5f7-151a-486c-be28-49e776864780/2302.04501v1.pdf)

### 제안하는 방법 (수식 포함)

#### Factorized Temporal Mixing

원본 시계열을 $s$개의 하위 시퀀스로 다운샘플링합니다:

$$X_{h,i} = \tilde{X}_h[i-1::s, :], \quad 1 \leq i \leq s$$

각 하위 시퀀스에 temporal feature extractor를 독립적으로 적용한 후 병합합니다:

$$X^T_{h,i} = \text{Temporal}(X_{h,i})$$
$$X^T_h = \text{merge}(X^T_{h,1}, \ldots, X^T_{h,s})$$

#### Factorized Channel Mixing

채널 redundancy를 matrix decomposition으로 처리합니다. $\tilde{X}^C_h = X_h + X^T_h$에 대해:

$$\tilde{X}^C_h = X^C_h + N \approx UV + N$$

여기서 $U \in \mathbb{R}^{n \times m}$, $V \in \mathbb{R}^{m \times c}$ ($m < c$)이며, 실제 구현에서는:

$$X^C_h = \sigma(\tilde{X}^C_h \cdot W_1^{\top} + b_1) \cdot W_2^{\top} + b_2$$

여기서 $W_1 \in \mathbb{R}^{m \times c}$, $W_2 \in \mathbb{R}^{c \times m}$입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3ceba5f7-151a-486c-be28-49e776864780/2302.04501v1.pdf)

#### 전체 프레임워크

MTS-Mixers의 전체 예측 프로세스는:

$$X^T_h = \text{Temporal}(\text{norm}(X_h))$$
$$X^C_h = \text{Channel}(X_h + X^T_h)$$
$$X_f = \text{Linear}(X^T_h + X^C_h)$$

### 모델 구조

MTS-Mixers는 세 가지 구현 변형을 제공합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3ceba5f7-151a-486c-be28-49e776864780/2302.04501v1.pdf)

**1) Attention-based MTS-Mixer**
- Multi-head self-attention으로 temporal dependency 포착
- FFN으로 channel interaction 학습
- Decoder 제거하고 direct projection 사용

**2) Random Matrix MTS-Mixer**
시계열 예측을 행렬 곱셈으로 단순화:

$$X_f = F \cdot \sigma(T) \cdot X_h \cdot \phi(C)$$

여기서 $T \in \mathbb{R}^{n \times n}$ (temporal dependency), $C \in \mathbb{R}^{c \times c}$ (channel dependency), $F \in \mathbb{R}^{m \times n}$ (projection matrix)

**3) Factorized MLP-based MTS-Mixer** (주요 모델)
- Temporal MLP: 다운샘플링된 subsequence에 독립적으로 적용
- Channel MLP: 저차원 factorization으로 효율적 처리
- Reversible Instance Normalization으로 distribution shift 완화 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3ceba5f7-151a-486c-be28-49e776864780/2302.04501v1.pdf)

### 성능 향상

**정량적 개선**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3ceba5f7-151a-486c-be28-49e776864780/2302.04501v1.pdf)
- **ECL**: 18.1% MSE 감소
- **Traffic**: 10.8% MSE 감소  
- **PeMS04**: 17.0% MSE 감소
- **Weather**: 7.2% MSE 감소
- **ILI**: 39.0% MSE 감소

**효율성 개선**:
- Training 시간: FEDformer 대비 약 7배, Autoformer 대비 약 3배 빠름
- Inference 속도: 비슷한 수준 유지하면서 더 나은 성능 달성
- 파라미터 수: Attention 기반 모델 대비 현저히 감소

### 한계점

**1) 하이퍼파라미터 민감성**
- Downsampling factor $s \in \{1,2,3,4,6,8,12\}$의 최적값이 데이터셋마다 상이 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3ceba5f7-151a-486c-be28-49e776864780/2302.04501v1.pdf)
- Channel factorization dimension $m$의 선택이 성능에 영향

**2) 특정 패턴에 대한 제한**
- 고도로 비선형적이고 불규칙한 시계열에는 성능 저하 가능성
- Positional encoding이 temporal information 포착을 방해할 수 있음

**3) 이론적 분석 부족**
- 왜 factorization이 효과적인지에 대한 엄밀한 이론적 설명 부족
- 일반화 성능 보장에 대한 이론적 근거 미흡

## 3. 모델의 일반화 성능 향상 가능성

### Cross-domain 일반화

**Channel-independent 설계**의 장점: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3ceba5f7-151a-486c-be28-49e776864780/2302.04501v1.pdf)
- 서로 다른 물리적 의미를 가진 채널들을 독립적으로 처리
- 새로운 도메인으로의 전이가 상대적으로 용이
- Over-fitting 위험 감소

### Factorization의 정규화 효과

**저차원 표현 학습**:
- Temporal factorization: $s$개의 subsequence로 분할하여 semantic information 강화
- Channel factorization: $m < c$로 설정하여 노이즈 제거 및 핵심 패턴 추출

이는 **implicit regularization** 효과를 제공하여 unseen 데이터에 대한 일반화 성능을 향상시킵니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3ceba5f7-151a-486c-be28-49e776864780/2302.04501v1.pdf)

### 다양한 예측 horizon에 대한 강건성

실험 결과에 따르면 MTS-Mixers는 예측 길이 {96, 192, 336, 720}에서 일관되게 우수한 성능을 보여, **다양한 time horizon에 대한 일반화 능력**이 뛰어남을 입증했습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3ceba5f7-151a-486c-be28-49e776864780/2302.04501v1.pdf)

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 연구 영향

**1) Attention-free 아키텍처의 재조명**
MTS-Mixers는 시계열 예측에서 attention 메커니즘이 필수적이지 않다는 것을 명확히 보여주었으며, 이는 이후 TSMixer, PatchMixer, RPMixer 등 MLP 기반 아키텍처의 발전을 촉진했습니다. [arxiv](https://arxiv.org/pdf/2303.06053.pdf)

**2) Decomposition 패러다임의 확산**
Temporal과 channel을 분리하여 모델링하는 접근법은 다음 연구들에 영향을 미쳤습니다:
- **DisenTS (2024)**: Channel evolving pattern을 disentangle하여 모델링 [arxiv](http://arxiv.org/pdf/2410.22981.pdf)
- **DUET (2025)**: Dual clustering으로 temporal/channel dimension 동시 처리 [arxiv](https://arxiv.org/html/2412.10859v1)

**3) 효율성과 성능의 균형**
복잡한 Transformer 대신 단순한 MLP로도 SOTA 성능을 달성할 수 있다는 것을 증명하여, 실용적 배포 가능성을 높였습니다.

### 앞으로 연구 시 고려할 점

#### 1. 더 정교한 Factorization 전략

**적응적 downsampling**:
- 고정된 $s$ 값 대신 데이터의 특성에 따라 동적으로 조정
- Learnable sampling strategy 개발

**계층적 factorization**:
- Multi-scale temporal patterns 포착을 위한 hierarchical decomposition

#### 2. Channel 간 상호작용 모델링 개선

MTS-Mixers는 channel independence를 가정하지만, **실제 multivariate 시계열에는 복잡한 cross-channel dependency**가 존재합니다. 

**권장 방향**:
- Selective channel interaction: 중요한 채널 쌍만 모델링 (Correlated Attention 참고) [arxiv](https://arxiv.org/abs/2311.11959)
- Graph-based channel modeling: 채널 간 관계를 그래프 구조로 표현

#### 3. 이론적 근거 강화

**필요한 연구**:
- Factorization이 일반화 성능을 향상시키는 이론적 메커니즘 규명
- Low-rank 가정의 타당성 검증을 위한 실증 연구
- PAC-learning framework에서의 generalization bound 도출

#### 4. 비정상성(Non-stationarity) 대응

**Distribution shift 문제**:
- Reversible Instance Normalization 외에 추가적인 normalization 기법 필요
- Adaptive normalization strategies 개발 (KANMTS 참고) [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12222760/)

#### 5. 대규모 사전학습 및 전이학습

**Self-supervised learning**:
- Masked time series modeling
- Contrastive learning for time series representation

최근 PatchTST와 TSMixer는 사전학습의 효과를 입증했으며, MTS-Mixers도 이를 활용할 여지가 큽니다. [github](https://github.com/yuqinie98/PatchTST)

## 5. 2020년 이후 관련 최신 연구 비교 분석

### Linear-based Models (2022-2023)

| 모델 | 연도 | 핵심 아이디어 | MTS-Mixers와의 관계 |
|------|------|---------------|-------------------|
| **DLinear** [arxiv](https://arxiv.org/pdf/2205.13504.pdf) | 2022 | Decomposition + Linear layers | MTS-Mixers가 뛰어넘은 baseline |
| **PatchTST** [github](https://github.com/yuqinie98/PatchTST) | 2023 | Patching + Channel independence | Channel independence는 유사, patching은 다름 |
| **TSMixer** [arxiv](https://arxiv.org/pdf/2303.06053.pdf) | 2023 | Time-mixing + Feature-mixing MLPs | MTS-Mixers의 영향을 받아 발전 |
| **MoLE** [arxiv](https://arxiv.org/html/2312.06786v3) | 2023 | Mixture-of-Linear-Experts | 적응적 predictor 선택으로 확장 |

### Transformer-based Models (2020-2024)

| 모델 | 연도 | 핵심 메커니즘 | 한계 |
|------|------|---------------|------|
| **Informer** [arxiv](https://arxiv.org/pdf/2205.13504.pdf) | 2020 | ProbSparse attention | MTS-Mixers 대비 낮은 성능 |
| **Autoformer** [arxiv](https://arxiv.org/html/2410.13792v1) | 2021 | Auto-correlation + Decomposition | 복잡도 높음, MTS-Mixers가 능가 |
| **FEDformer** [arxiv](https://arxiv.org/pdf/2511.12951.pdf) | 2022 | Frequency-enhanced + Decomposition | MTS-Mixers가 효율성에서 우위 |
| **PatchTST** [github](https://github.com/yuqinie98/PatchTST) | 2023 | Patch-based Transformer | MTS-Mixers와 유사한 성능, 더 복잡 |

### 최신 발전 (2024-2026)

**1) Decomposition 강화**:
- **LiNo (2024)**: Linear/Nonlinear pattern의 recursive residual decomposition [arxiv](https://arxiv.org/html/2410.17159v2)
- **DisenTS (2024)**: Channel evolving pattern disentanglement [arxiv](http://arxiv.org/pdf/2410.22981.pdf)

**2) MLP 아키텍처 발전**:
- **MSD-Mixer (2023)**: Multi-scale decomposition with MLP-Mixer [arxiv](https://arxiv.org/html/2310.11959v2)
- **RPMixer (2024)**: Random projection layers for diversity [arxiv](https://arxiv.org/html/2402.10487v4)
- **KANMTS (2025)**: KAN과 MLP 통합, 해석가능성 강화 [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12222760/)

**3) Attention 메커니즘 개선**:
- **Local Attention Mechanism (2024)**: $O(n \log n)$ 복잡도 [arxiv](https://arxiv.org/abs/2410.03805)
- **Correlated Attention (2023)**: Feature-wise dependencies 포착 [arxiv](https://arxiv.org/abs/2311.11959)
- **ARMA Attention (2024)**: AR/MA 구조로 attention 강화 [openreview](https://openreview.net/forum?id=Z9N3J7j50k)

### 비교 종합

**MTS-Mixers의 독창성**:
1. **명시적 factorization**: Temporal과 channel을 분리하여 처리하는 첫 프레임워크
2. **다양한 구현 제공**: Attention, Random Matrix, MLP 세 가지 변형 제시
3. **효율성-성능 균형**: 단순한 구조로 SOTA 달성

**후속 연구들의 개선**:
- **TSMixer (2023)**: 더 체계적인 mixing 전략과 auxiliary feature 통합 [arxiv](https://arxiv.org/pdf/2303.06053.pdf)
- **DUET (2025)**: Dual clustering으로 더 정교한 temporal/channel modeling [arxiv](https://arxiv.org/html/2412.10859v1)
- **KANMTS (2025)**: 해석가능성 향상과 symbolic regression 통합 [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12222760/)

### 현재 트렌드 (2025-2026)

1. **Foundation Models**: 대규모 사전학습 모델 개발
2. **Hybrid Architectures**: Linear + Attention의 조합
3. **Interpretability**: 예측 결과의 설명 가능성 강화
4. **Efficiency**: 경량화 및 실시간 처리 능력 향상

## 결론

MTS-Mixers는 시계열 예측 분야에서 **"복잡함이 항상 좋은 것은 아니다"**라는 중요한 교훈을 제공했습니다. Factorized temporal and channel mixing이라는 단순하지만 효과적인 접근법으로 SOTA 성능을 달성하며, 이후 연구들에게 새로운 설계 패러다임을 제시했습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3ceba5f7-151a-486c-be28-49e776864780/2302.04501v1.pdf)

향후 연구는 **적응적 factorization, 이론적 근거 강화, 사전학습 활용**에 집중해야 하며, 실용적 배포를 위해 **효율성과 해석가능성의 균형**을 추구해야 할 것입니다. 특히 최근 TSMixer, DUET, KANMTS 등의 발전은 MTS-Mixers가 제시한 방향성이 올바랐음을 입증하고 있습니다. [arxiv](https://arxiv.org/pdf/2303.06053.pdf)

<span style="display:none">[^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76]</span>

<div align="center">⁂</div>

[^1_1]: 2302.04501v1.pdf

[^1_2]: https://arxiv.org/pdf/2303.06053.pdf

[^1_3]: https://ar5iv.labs.arxiv.org/html/2303.06053

[^1_4]: https://arxiv.org/html/2402.10487v4

[^1_5]: https://ar5iv.labs.arxiv.org/html/2310.00655

[^1_6]: http://arxiv.org/pdf/2410.22981.pdf

[^1_7]: https://arxiv.org/html/2412.10859v1

[^1_8]: https://arxiv.org/abs/2311.11959

[^1_9]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12222760/

[^1_10]: https://github.com/yuqinie98/PatchTST

[^1_11]: https://arxiv.org/pdf/2205.13504.pdf

[^1_12]: https://arxiv.org/abs/2205.13504

[^1_13]: https://arxiv.org/html/2312.06786v3

[^1_14]: https://arxiv.org/html/2410.13792v1

[^1_15]: https://huggingface.co/blog/autoformer

[^1_16]: https://arxiv.org/pdf/2511.12951.pdf

[^1_17]: https://arxiv.org/pdf/2201.12740.pdf

[^1_18]: https://arxiv.org/html/2410.17159v2

[^1_19]: https://arxiv.org/html/2310.11959v2

[^1_20]: https://arxiv.org/abs/2410.03805

[^1_21]: https://openreview.net/forum?id=Z9N3J7j50k

[^1_22]: https://www.semanticscholar.org/paper/b26ec89f91a64465b01dfbcb4ed8f0d6fb9cf0e1

[^1_23]: http://www.ije.ir/article_108448.html

[^1_24]: https://link.springer.com/10.1007/s11356-021-14286-7

[^1_25]: https://www.semanticscholar.org/paper/2c286840a127a1e03eea230bbde604cc3ab53613

[^1_26]: https://journal.unpacti.ac.id/index.php/JSCE/article/view/2373

[^1_27]: https://iopscience.iop.org/article/10.1088/1755-1315/514/4/042020

[^1_28]: https://journals.scholarpublishing.org/index.php/AIVP/article/view/19906

[^1_29]: https://www.semanticscholar.org/paper/09774a67df13190db55f9df00ede779d665aa1f5

[^1_30]: https://www.semanticscholar.org/paper/f62d2d628d18a7e615a7a62aa8fed675b31bb506

[^1_31]: https://osf.io/j57pk_v1

[^1_32]: https://arxiv.org/pdf/2307.01616.pdf

[^1_33]: http://arxiv.org/pdf/2501.04339.pdf

[^1_34]: https://arxiv.org/pdf/2109.01657.pdf

[^1_35]: https://arxiv.org/pdf/1703.07015.pdf

[^1_36]: https://arxiv.org/html/2502.10721v1

[^1_37]: https://arxiv.org/pdf/2107.06344.pdf

[^1_38]: http://arxiv.org/list/physics/2023-10?skip=650\&show=2000

[^1_39]: https://arxiv.org/pdf/2209.05684.pdf

[^1_40]: https://arxiv.org/abs/2510.06680

[^1_41]: https://arxiv.org/abs/2509.19985

[^1_42]: https://arxiv.org/abs/2602.10847

[^1_43]: https://openreview.net/pdf/7242052676ec48adcbe9fb56e38282ab9f97c66a.pdf

[^1_44]: https://arxiv.org/abs/2206.12626

[^1_45]: https://dl.acm.org/doi/abs/10.1145/3663976.3664241

[^1_46]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11059412/

[^1_47]: https://arxiv.org/html/2306.09364v4

[^1_48]: https://huggingface.co/papers/2306.09364

[^1_49]: https://www.ijournalse.org/index.php/ESJ/article/view/1225

[^1_50]: https://link.springer.com/10.1007/s10661-022-10274-7

[^1_51]: https://doiserbia.nb.rs/Article.aspx?ID=0354-983622261A

[^1_52]: https://www.scirp.org/journal/doi.aspx?doi=10.4236/ojs.2022.122019

[^1_53]: https://arxiv.org/abs/2212.02567

[^1_54]: https://www.ssrn.com/abstract=4197041

[^1_55]: https://www.mdpi.com/2071-1050/14/15/9081

[^1_56]: https://pubs.aip.org/aip/acp/article/2823504

[^1_57]: https://www.mdpi.com/2072-4292/14/18/4461

[^1_58]: https://ieeexplore.ieee.org/document/9984237/

[^1_59]: https://arxiv.org/html/2410.21448v1

[^1_60]: https://arxiv.org/pdf/2401.13912.pdf

[^1_61]: http://arxiv.org/pdf/2409.10142.pdf

[^1_62]: https://arxiv.org/pdf/2305.10721.pdf

[^1_63]: https://arxiv.org/html/2309.15946

[^1_64]: https://arxiv.org/abs/2305.04800

[^1_65]: https://www.arxiv.org/pdf/2501.01087v2.pdf

[^1_66]: https://arxiv.org/html/2507.15774v1

[^1_67]: https://arxiv.org/html/2512.22702v1

[^1_68]: https://arxiv.org/html/2501.01087v1

[^1_69]: https://arxiv.org/pdf/2405.14982.pdf

[^1_70]: https://velog.io/@jhbale11/DLinear2022논문-리뷰

[^1_71]: https://github.com/cure-lab/LTSF-Linear

[^1_72]: https://nixtlaverse.nixtla.io/neuralforecast/models.dlinear.html

[^1_73]: https://baidukddcup2022.github.io/papers/Baidu_KDD_Cup_2022_Workshop_paper_8925.pdf

[^1_74]: https://simudyne.com/resources/a-single-linear-layer-is-all-you-need-linear-models-outperform-transformers-for-long-term-time-series-forecasting/

[^1_75]: https://sonstory.tistory.com/119

[^1_76]: https://secundo.tistory.com/113
