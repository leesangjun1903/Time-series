# Chronos: Learning the Language of Time Series

### 1. 핵심 주장과 주요 기여

**Chronos** (Ansari et al., 2024)는 시간 시리즈 데이터를 언어 모델의 "언어"로 변환하여 기존 Transformer 아키텍처를 최소 수정으로 재사용할 수 있음을 입증한 선도적 연구입니다. 논문의 기본 가정은 단순하지만 혁신적입니다: **언어 모델의 다음 토큰 예측과 시계열 예측의 근본적 목표는 동일하다**는 것입니다.[1]

Chronos의 핵심 기여는 다음과 같습니다:

- **토큰화를 통한 문제 변환**: 실수값 시계열을 스케일링과 양자화를 통해 이산 토큰으로 변환하여 기존 언어 모델을 직접 활용
- **교차 엔트로피 손실 함수**: 거리를 인식하지 않는 범주형 분포를 사용하면서도 회귀 문제를 분류로 해결
- **강력한 일반화 성능**: 42개 데이터셋 벤치마크에서 학습 데이터셋에 대해 우월한 성능, 미학습 데이터셋에서 영점샷(Zero-shot) 성능이 경쟁 모델과 동등하거나 우월[1]
- **데이터 증강 전략**: TSMixup과 KernelSynth를 통한 합성 데이터 생성으로 일반화 능력 향상

***

### 2. 해결하려는 문제와 제안 방법

#### 2.1 문제 정의

전통적 시계열 예측은 다음과 같은 한계를 가집니다:

- **도메인 특이성**: 각 데이터셋마다 모델을 재학습해야 하는 비효율성
- **데이터 부족**: 자연어 처리와 달리 공개 고품질 시계열 데이터 부족
- **아키텍처 복잡성**: 시간 특화 설계(time features, lag features 등)로 인한 모델 특성화

#### 2.2 제안 방법: Chronos 프레임워크

**기본 절차**:

```math
\text{시계열} \xrightarrow{\text{Scaling}} \tilde{x}_i = \frac{x_i}{s}, \quad s = \frac{1}{C}\sum_{i=1}^{C}|x_i|
```

```math
\tilde{x}_i \xrightarrow{\text{Quantization}} q(x) = \begin{cases} 1 & \text{if } -\infty \leq x < b_1 \\ 2 & \text{if } b_1 \leq x < b_2 \\ \vdots & \vdots \\ B & \text{if } b_{B-1} \leq x < \infty \end{cases}
```

**손실 함수**:

$$\ell(\theta) = -\sum_{h=1}^{H+1}\sum_{i=1}^{|V_{ts}|} \mathbb{1}(z_{C+h+1}=i) \log p_\theta(z_{C+h+1}=i|z_{1:C+h})$$

여기서 $z_{1:C+h}$는 토큰화된 시계열, $p_\theta$는 범주형 분포입니다.[2]

**모델 아키텍처**: T5 family (20M ~ 710M 파라미터)를 기반으로 어휘 크기만 조정하여 직접 활용합니다.

#### 2.3 데이터 증강 전략

**TSMixup**: 서로 다른 훈련 데이터셋의 기본 시계열을 무작위로 표본화하여 볼록 결합으로 새로운 시계열 생성:

$$y = \lambda x_1 + (1-\lambda)x_2, \quad \lambda \in $$[2]

**KernelSynth**: 가우스 프로세스 커널 함수를 무작위로 합성하여 다양한 패턴의 합성 시계열 생성[2]

***

### 3. 모델 구조와 성능

#### 3.1 인코더-디코더 구조

Chronos-T5는 T5의 인코더-디코더 구조를 유지하되:

- **Encoder**: 토큰화된 역사적 맥락 $z_{1:C}$를 밀집 표현으로 변환
- **Decoder**: 자귀적(autoregressive) 토큰 샘플링을 통해 예측 분포 $p_\theta(z_{C+h+1}|z_{1:C+h})$ 생성
- **Dequantization**: 예측된 토큰을 원래 스케일의 값으로 재변환

#### 3.2 성능 비교

| 모델 | 벤치마크 유형 | 성능 | 특징 |
|------|-------------|------|------|
| **Chronos-T5 (Large)** | 영점샷 | SOTA | 자귀적 샘플링, 확률적 예측 |
| **TimesFM** [3][4] | 영점샷 | SOTA | 패치 기반, 디코더-온리, 200M 파라미터 |
| **Moirai 2.0** [5] | 영점샷 | 상위권 | 분위수 예측, 다중 토큰 예측, 30배 작음 |
| **Time-LLM** [6] | 영점샷 | 경쟁 | 텍스트 프로토타입 정렬, 프롬프트-접두사 |

**벤치마크 성능**: 42개 데이터셋에서 Chronos는 학습 데이터셋에 대해 다른 방법을 크게 능가하며, 미학습 데이터셋에서 영점샷 성능은 특화 모델과 동등 또는 우월합니다.[1]

***

### 4. 일반화 성능 향상 가능성

#### 4.1 현재 강점

1. **도메인 간 전이 학습**: 27개 공개 데이터셋 + 합성 데이터로 학습하여 다양한 도메인(금융, 에너지, 의료, 기후)에서 영점샷 예측 가능[1]

2. **확률적 예측**: 범주형 분포는 거리 정보를 직접 사용하지 않지만, 훈련 데이터에서 인접 토큰의 관계를 학습하여 유연한 분포 모양(단봉, 다봉) 생성[2]

3. **적응형 스케일링**: 평균 스케일링은 영점값 보존(예: 야간 태양광 발전 = 0) 및 최적화 안정성 제공[2]

#### 4.2 한계 및 개선 가능성

**현재 한계**:

- **지수 추세 모델링 불충분**: 선형 추세는 정확하나 지수 추세에서 부진 → 로그 스케일링 전처리로 개선 가능[2]
- **짧은 맥락에서 추세 저평가**: 충분한 역사 데이터 부족 시 미래 추세 상대 음과소[2]
- **예측 범위 제한**: 토큰 범위 $[-15, 15] \times s$로 제한되어 강한 추세의 시계열 이론적으로 모델링 불가[2]
- **정밀도 손실**: 매우 작은 스케일의 신호에서 양자화로 인한 정보 손실[2]

**개선 방향**:

1. **거리 인식 손실 함수**: 레이블 스무딩으로 인접 토큰에 확률질량 할당[2]
   
2. **서수(Ordinal) 변수 처리**: 토큰의 순서 구조를 명시적으로 인코딩

3. **적응형 정규화**: 데이터 특성에 따라 표준화, 최소-최대 정규화 등 동적 선택

4. **컨텍스트 길이 유연성**: 장기 시계열을 위한 더 긴 컨텍스트 윈도우

***

### 5. 2020년 이후 관련 최신 연구 비교

| 연구 | 발표 | 아키텍처 | 주요 특징 | 일반화 성능 |
|-----|------|---------|---------|-----------|
| **LLMTime** [7] | 2023 | GPT-3/Llama2 | 숫자 문자열 인코딩, 영점샷 | 경쟁 레벨 |
| **TimesFM** [3][4] | 2024 | 디코더-온리 Transformer | 패치 기반, 100B 시점 학습 | 상위권 SOTA |
| **Chronos** [1] | 2024 | T5 인코더-디코더 | 토큰화 + 자귀적, 합성 데이터 | 상위권 SOTA |
| **Moirai 1.0** [8] | 2024 | 마스크 인코더 Transformer | 다중 패치 크기, 27B 관측 | 경쟁 SOTA |
| **Time-LLM** [6] | 2024 | GPT-2 백본 + 정렬 | 텍스트 프로토타입, 프롬프팅 | 미세조정 강함 |
| **Moirai 2.0** [5] | 2025 | 디코더-온리 | 분위수 예측, 36M 시계열 | 빠름 + 정확함 |
| **TimesFM 2.0** [9] | 2025 | 개선된 디코더 | 더 긴 컨텍스트 (2048), 20% 성능↑ | 리더보드 1위 |
| **ViTime** [10] | 2024 | 비전 기반 | 이진 이미지 메트릭 공간 | 영점샷 TimesFM 9-15% ↑ |
| **Chronos-Bolt** [11] | 2024 | T5 (최적화) | 직접 다단계, 250배 빠름 | 빠름 + 정확함 |

#### 5.1 주요 진화 추세

**1단계 (2023-2024 초기)**: 기초 모델 실행 가능성 증명
- LLMTime, Chronos, TimesFM의 출현으로 영점샷 예측 가능성 입증
- 하지만 미세조정되지 않은 기준 모델과 경쟁 불균형[12]

**2단계 (2024 중기-현재)**: 아키텍처 최적화 및 확장
- 패치 기반 표현 (TimesFM 2.0)
- 분위수 예측 (Moirai 2.0)
- 혼합 전문가 (Mixture of Experts)[13]
- 다중 모드 데이터 처리 (DP-GPT4MTS)[14]

**3단계 (2024 후기-2025)**: 효율성과 성능의 파레토 최적화
- **Moirai 2.0**: 30배 작음 + 2배 빠름 + 더 나은 정확도[5]
- **Chronos-Bolt**: 250배 속도 향상 + 직접 다단계 예측[11]
- **TimesFM 2.0**: GIFT-Eval 리더보드 1위, 더 긴 컨텍스트[9]

#### 5.2 Chronos의 위치

Chronos는 초기 세대 기초 모델의 대표주자로:

**강점**:
- 간단하고 이해하기 쉬운 설계 철학 (최소주의)
- 우수한 영점샷 성능과 일관된 결과
- 합성 데이터 증강으로 데이터 부족 완화
- 확률적 예측으로 불확실성 정량화

**상대적 약점**:
- TimesFM 2.0의 최신 성능에 미치지 못함 (리더보드에서 TimesFM 2.0이 1위)
- 추론 속도에서 Chronos-Bolt 이전 버전은 느림 (최적화 버전 출시로 개선)
- 거리 인식 손실로 보조 구조 미활용

***

### 6. 향후 연구 시 고려사항

#### 6.1 Chronos 자체의 개선 방향

1. **거리-인식 손실 함수 통합**: 레이블 스무딩 또는 서수 회귀 목표 함수 도입으로 토큰 근처성 활용[2]

2. **멀티바리에이트 및 공변량 지원**: 현재는 단변량에 제한되어 있으나 다변량 및 외생 변수 통합 필요[6.1][2]

3. **장기 예측 능력**: 더 긴 컨텍스트 윈도우 지원 및 일관된 길이 예측 개선

4. **미래 개발 방향**:
   - NLP 커뮤니티의 고속 디코딩 기법 (스펙듀러티브 디코딩) 도입[2]
   - CUDA 최적화 및 양자화로 추론 속도 향상
   - 초기 징후 학습으로 이상 탐지 등 다중 작업 확장[2]

#### 6.2 기초 모델 개발의 일반 원칙

**1. 데이터 전략의 우선순위**:
- 모델 아키텍처보다 **고품질 다양한 훈련 데이터**가 일반화 성능을 좌우[15]
- NLP와 달리 공개 시계열 데이터 부족이 핵심 병목 → 합성 데이터 생성 기법 발전 필수
- 데이터 오염 위험: 영점샷 성능 평가 시 학습 데이터와의 중복 검증[12]

**2. 영점샷 vs 미세조정 트레이드오프**:
- Chronos는 영점샷 성능 우수하나, 실무에서는 미세조정 대부분 성능 향상[12]
- 미세조정을 위한 파라미터 효율 방법 (LoRA) 병행 권고[2]
- 컨포멀 예측(Conformal Prediction)으로 미세조정 없이 교정 가능[2]

**3. 모델 복잡성의 함정**:
- 더 큰 모델이 항상 나은 것 아님: Moirai 2.0이 선대 대비 30배 작으면서 더 정확[5]
- 추론 속도와 메모리 효율성을 정확도와 동등하게 고려
- 소규모 특화 모델(예: SAMFormer)이 대규모 기초 모델보다 특정 도메인에서 우월[12]

**4. 분포 이동(Domain Shift) 문제**:
- 기초 모델의 영점샷 성능은 **훈련 데이터와의 도메인 유사성에 크게 의존**[12]
- 미학습 도메인의 실데이터(예: 개인 전력 사용)에서는 성능 급락 관찰[12]
- 해결책: 도메인 적응 기법, 맥락 정보의 중요성[16]

**5. 토큰화와 표현의 중요성**:
- 수치 토큰화 방식이 성능을 크게 좌우 (예: GPT-4의 숫자 토큰화로 GPT-3보다 악화)[7]
- 통계 기반 특성 추출과 LLM의 조합이 효과적[17][18]
- 컨텍스트 정보 제공이 고급 프롬팅 기법보다 더 중요[16]

***

### 7. 한계 및 미해결 문제

#### 7.1 현재 기초 모델의 한계

**패러다임 한계**:
- **불완전한 영점샷**: "영점샷"이라 불리지만 실제는 다른 도메인 데이터에 사전학습된 결과로, 완전히 마주치지 않은 패턴에는 부진[16]
- **노이즈 민감성**: 실제 데이터의 노이즈에 LLM 기반 방법이 특히 취약[19]
- **계산 오류**: 중복된 피연산자로 단순 대수 계산 실패[16]

**기술적 한계**:
- **확장 가능성**: 변량 수가 많은 다변량 시계열 처리 어려움
- **외생 변수**: 미래 공변량을 포함한 복합 시계열 예측의 표준화된 방법 부재
- **비정상성**: 실제 시계열의 비정상적 특성을 일관되게 모델링하지 못함

#### 7.2 미해결 연구 문제

1. **최적 모델 크기**: 추가 파라미터가 항상 성능 향상을 보장하지 않음 → 최적 선택 기준 개발 필요[5]

2. **컨텍스트 길이 vs 정확도**: 장기 의존성 모델링과 추론 효율의 균형[20]

3. **도메인 적응의 체계화**: 새 도메인에 빠르게 적응하는 표준화된 방법 부재

4. **설명가능성**: 기초 모델의 예측 근거 해석의 어려움

***

### 결론

**Chronos**는 최소주의적 설계 철학으로 시계열 예측 문제를 언어 모델링으로 재구성하는 혁신적 접근법을 제시합니다. 토큰화, 교차 엔트로피 손실, 합성 데이터 증강의 조합으로 기존 언어 모델 인프라를 최소 수정으로 재사용 가능함을 입증한 점에서 학술적 기여가 큽니다.

그러나 **TimesFM 2.0**(2025), **Moirai 2.0**(2025)과 같은 후속 모델들이 더 나은 성능, 속도, 효율성을 달성하면서 Chronos는 초기 세대 기초 모델의 대표로 자리매김했습니다. 

향후 연구는 다음을 중점으로 해야 합니다:

1. **데이터 우위**: 모델 아키텍처 혁신보다 고품질 대규모 시계열 데이터 구축 우선[15]
2. **효율성-정확도 파레토 경계**: 리소스 제약 환경에서의 실용적 모델 개발[5]
3. **실제 도메인 이전**: 영점샷 약속을 현실 데이터의 분포 이동 조건에서 실현[12]
4. **다중 작업 일반화**: 시계열 예측을 넘어 분류, 이상 탐지 등 포괄적 기초 모델 구축[2]

Chronos의 단순함과 우수한 성능은 기초 모델 시대에서 최소한의 설계로 최대의 성능을 추구하는 설계 철학을 보여주는 구체적 사례입니다.[21][1]

***

### 참고 문헌

[1](https://arxiv.org/abs/2403.07815)
[2](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1a43bd1d-32f5-4de7-b7d9-aa2474ab824d/2403.07815v3.pdf)
[3](https://arxiv.org/pdf/2310.10688.pdf)
[4](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/)
[5](https://arxiv.org/html/2511.11698v1)
[6](https://arxiv.org/pdf/2310.01728.pdf)
[7](https://arxiv.org/abs/2310.07820)
[8](https://arxiv.org/abs/2402.02592)
[9](https://arxiv.org/html/2505.11163v1)
[10](https://www.semanticscholar.org/paper/b72a95c6070d722335fae650a0e5b1dd926a66a8)
[11](https://aws.amazon.com/blogs/machine-learning/fast-and-accurate-zero-shot-forecasting-with-chronos-bolt-and-autogluon/)
[12](https://arxiv.org/html/2510.00742v3)
[13](https://arxiv.org/abs/2409.16040)
[14](https://arxiv.org/abs/2508.04239)
[15](https://galileo.ai/blog/amazon-chronos-ai-time-series-forecasting-guide)
[16](https://ieeexplore.ieee.org/document/11100503/)
[17](http://arxiv.org/pdf/2503.03594.pdf)
[18](https://arxiv.org/pdf/2503.09656.pdf)
[19](https://aclanthology.org/2025.acl-short.71.pdf)
[20](https://arxiv.org/pdf/2507.08858.pdf)
[21](https://www.scitepress.org/Papers/2025/133638/133638.pdf)
[22](https://ieeexplore.ieee.org/document/11314515/)
[23](https://dl.acm.org/doi/10.1145/3671127.3699536)
[24](https://arxiv.org/html/2504.04011v1)
[25](https://arxiv.org/html/2512.20140v1)
[26](https://zaai.ai/moirai-time-series-foundation-models-for-universal-forecasting/)
[27](https://ieeexplore.ieee.org/document/11137629/)
[28](https://ieeexplore.ieee.org/document/11050326/)
[29](https://www.jmir.org/2025/1/e74423)
[30](https://goodwoodpub.com/index.php/jictl/article/view/3256)
[31](https://ojs.aaai.org/index.php/AAAI/article/view/30383)
[32](https://arxiv.org/abs/2407.20503)
[33](https://dl.acm.org/doi/10.1145/3715073.3715083)
[34](https://arxiv.org/abs/2405.14252)
[35](https://arxiv.org/pdf/2403.05798.pdf)
[36](http://arxiv.org/pdf/2412.11376.pdf)
[37](https://arxiv.org/pdf/2404.15772.pdf)
[38](https://arxiv.org/pdf/2503.04118.pdf)
[39](https://arxiv.org/pdf/2402.02370.pdf)
[40](https://zaai.ai/chronos-the-rise-of-foundation-models-for-time-series-forecasting/)
[41](https://github.com/KimMeen/Time-LLM)
[42](https://github.com/google-research/timesfm/)
[43](https://arxiv.org/html/2403.07815v1)
[44](https://velog.io/@sheoyonj/Are-Language-Models-Actually-Useful-for-Time-Series-Forecasting)
[45](https://research.aimultiple.com/time-series-foundation-models/)
[46](https://www.reddit.com/r/MachineLearning/comments/1behp7t/r_chronos_learning_the_language_of_time_series/)
[47](https://arxiv.org/html/2508.04231v1)
[48](https://github.com/google-research/timesfm)
[49](https://www.youtube.com/watch?v=kBu8Ko1b6rs)
[50](https://liner.com/ko/review/are-language-models-actually-useful-for-time-series-forecasting)
[51](https://velog.io/@sheoyonj/Foundation-model-in-Time-series)
[52](https://kingnamji.tistory.com/70)
[53](https://research.google/blog/time-series-foundation-models-can-be-few-shot-learners/)
[54](https://www.sciencedirect.com/science/article/pii/S0950705125014881)
[55](https://arxiv.org/html/2403.14735v3)
[56](https://arxiv.org/html/2512.20002v1)
[57](https://arxiv.org/html/2506.12953v1)
[58](https://arxiv.org/abs/2402.03885)
[59](https://pubmed.ncbi.nlm.nih.gov/40775972/)
[60](https://arxiv.org/html/2509.00616v1)
[61](https://www.arxiv.org/abs/2509.12080)
[62](https://arxiv.org/abs/2510.15821)
[63](https://arxiv.org/html/2503.08271v1)
[64](https://arxiv.org/abs/2505.13521)
[65](https://arxiv.org/html/2508.07195v1)
[66](https://arxiv.org/abs/2510.00742)
[67](https://arxiv.org/abs/2505.23719)
[68](https://arxiv.org/html/2503.03594v1)
[69](https://arxiv.org/abs/2510.01560)
[70](https://www.semanticscholar.org/paper/13e362a648f73ba7119aa850100632fba2993cbe)
[71](https://arxiv.org/abs/2410.10469)
[72](https://dl.acm.org/doi/10.1145/3671127.3698708)
[73](https://www.semanticscholar.org/paper/3ae2822fcaa5a0054cc450d3209c7493a0ebeac2)
[74](https://ieeexplore.ieee.org/document/11180251/)
[75](https://journals.lww.com/10.4103/ijoem.ijoem_290_23)
[76](https://arxiv.org/abs/2408.11990)
[77](http://arxiv.org/pdf/2405.14252.pdf)
[78](https://www.mdpi.com/2076-3417/15/7/3450)
[79](http://arxiv.org/pdf/2410.11802.pdf)
[80](https://arxiv.org/pdf/2502.15637.pdf)
[81](http://arxiv.org/pdf/2410.11773.pdf)
[82](https://arxiv.org/html/2502.00816v1)
[83](https://arxiv.org/pdf/2412.09880.pdf)
[84](https://www.reddit.com/r/LocalLLaMA/comments/1dnajuy/salesforce_releases_moirai11_time_series/)
[85](https://openaccess.thecvf.com/content/WACV2025/papers/Sui_Just_Shift_It_Test-Time_Prototype_Shifting_for_Zero-Shot_Generalization_with_WACV_2025_paper.pdf)
[86](https://huggingface.co/google/timesfm-1.0-200m)
[87](https://arxiv.org/abs/2506.02389)
[88](https://github.com/OopsCCK/-)
[89](https://www.datasciencewithmarco.com/blog/hands-on-with-moirai-a-foundation-forecasting-model-by-salesforce)
[90](https://liner.com/ko/review/large-language-models-are-zeroshot-time-series-forecasters)
[91](https://huggingface.co/google/timesfm-2.0-500m-pytorch)
[92](https://github.com/SalesforceAIResearch/uni2ts)
[93](https://discuss.pytorch.kr/t/time-llm-llm-time-series-forecasting-by-reprogramming-large-language-models/3436)
[94](https://arxiv.org/html/2410.11773v1)
[95](https://huggingface.co/Salesforce/moirai-1.0-R-large)
[96](https://arxiv.org/pdf/2310.07820.pdf)
[97](https://arxiv.org/abs/2310.10688)
[98](https://arxiv.org/abs/2511.11698)
[99](https://arxiv.org/abs/2512.02833)
[100](https://arxiv.org/html/2506.02389v1)
[101](https://arxiv.org/abs/2410.11773)
[102](https://arxiv.org/pdf/2511.11698.pdf)
[103](https://arxiv.org/html/2310.10688v2)
[104](https://arxiv.org/pdf/2402.02592.pdf)
