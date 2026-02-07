# Online Test-Time Adaptation of Spatial-Temporal Traffic Flow Forecasting

## 1. 핵심 주장과 주요 기여 (간결 요약) 

- **핵심 문제의식**: 과거 데이터 분포 $\(p_h\)$ 로 학습된 시공간 교통 예측 모델은, 시간이 지나며 바뀌는 미래 분포 $\(p_{f_1}, p_{f_2}, \dots\)$ (temporal drift)에서 성능이 급격히 저하된다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/336dcc3f-e645-4bbb-b644-d8d3de8b2b61/2401.04148v1.pdf)
- **핵심 아이디어**: 테스트 시점마다 완전히 온라인으로 작동하는 OTTA(Online Test-Time Adaptation) 설정을 도입하고, 기존 예측 모델 $\(f\)$ 뒤에 경량 모듈 ADCSD(Adaptive Double Correction by Series Decomposition)를 붙여, 출력 시계열을 계절(Seasonal)·추세(Trend-cyclical)로 분해 후 각각 보정하여 분포 변화를 따라가게 한다. [arxiv](https://arxiv.org/html/2401.04148v1)
- **주요 기여**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/336dcc3f-e645-4bbb-b644-d8d3de8b2b61/2401.04148v1.pdf)
  1. 시공간 교통 예측 문제에 대해 **최초로 OTTA 세팅을 공식화**.  
  2. 시계열 분해 + 이중 보정 + 노드별 가중 벡터를 결합한 **ADCSD를 제안**, ASTGCN, AGCRN, PDFormer 등 다양한 교통 모델에 plug-and-play로 부착 가능.  
  3. 4개 실제 데이터셋(PeMS07, BayArea, NYCTaxi, T-Drive)에서 기존 모델 대비 MAE/MAPE/RMSE를 일관되게 향상시키며, CV용 TTA(TENT, TTT-MAE)는 그대로 쓰면 오히려 성능을 악화시킴을 보인다. [ar5iv.labs.arxiv](https://ar5iv.labs.arxiv.org/html/2401.04148)

***

## 2. 해결 문제, 방법(수식), 구조, 성능·한계

### 2.1 해결하고자 하는 문제

1) **Temporal drift / 비정상 시계열**

- 역사 구간에서의 데이터 분포를 $\(p_h\)$ , 미래 테스트 구간에서의 분포를 $\(p_{f_t}\)$ 라 하면,  

$$p_{f_1} \neq p_{f_2} \neq \dots \neq p_h$$  
  
  와 같이 시간이 지남에 따라 분포가 계속 변하는 non‑stationary 상황을 가정한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/336dcc3f-e645-4bbb-b644-d8d3de8b2b61/2401.04148v1.pdf)
- 기존 deep traffic model $\(f\)$ 는 $\(p_h\)$ 에 최적화되어 있어, $\(p_{f_t}\)$ 에서 out-of-distribution이 되어 성능이 떨어진다. [arxiv](https://arxiv.org/html/2401.04148v1)

2) **데이터 정체성의 상대성 (relativity of data identity)**

- 시점 $\(t\)$ 에서 입력 $\(x_t\)$ 에 대한 예측을 한 뒤, 시점 $\(t+1\)$ 이 되면 정답 $\(y_t\)$ 를 관측할 수 있다.  
- 따라서 OTTA 설정에서 **“예측→다음 시점에 정답 관측→즉시 업데이트”** 루프를 돌릴 수 있는데, 기존 CV TTA는 이런 구조를 전혀 활용하지 않는다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/336dcc3f-e645-4bbb-b644-d8d3de8b2b61/2401.04148v1.pdf)

3) **시공간 교통 데이터의 특수성**

- 데이터는  

$$x_t \in \mathbb{R}^{N \times T \times C}, \quad y_t \in \mathbb{R}^{N \times T' \times C}$$  
  로, 여러 노드 $\(N\)$ , 시간 길이 $\(T, T'\)$ , feature 수 $\(C\)$ 로 구성된 멀티변량 시공간 시계열이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/336dcc3f-e645-4bbb-b644-d8d3de8b2b61/2401.04148v1.pdf)
- 노드 간 공간 상관, 각 노드별 상이한 drift 수준 등 때문에, 단순히 CV용 TENT/TTT-MAE를 적용하면 오히려 성능이 심각하게 열화된다. [ar5iv.labs.arxiv](https://ar5iv.labs.arxiv.org/html/2401.04148)

**따라서 목표**는: 이미 학습된 교통 예측 모델 $\(f\)$ 를 고정한 채, 테스트 스트림 $\(\{(x_t, y_t)\}\)$ 가 들어오는 대로, 경량 모듈만 온라인으로 업데이트해 temporal drift를 따라가는 OTTA 프레임워크를 설계하는 것이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/336dcc3f-e645-4bbb-b644-d8d3de8b2b61/2401.04148v1.pdf)

***

### 2.2 제안 방법: ADCSD (수식 포함)

#### 2.2.1 OTTA 설정 공식화

- 미래 테스트 데이터:

$$\mathcal{X} = \{(x_t, y_t)\}_{t=1}^n,$$

$$x_t \in \mathbb{R}^{N \times T \times C}, \quad y_t \in \mathbb{R}^{N \times T' \times C}.$$
- 이미 학습된 history model $\(f\)$ 는 고정(frozen) 상태로 두고, 각 시점 $\(t\)$ 마다:

$$o_t = f(x_t)$$
  를 계산한 후, 시점 $\(t+1\)$ 에서 관측한 $\(y_t\)$ 로 경량 모듈만 한 스텝 업데이트한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/336dcc3f-e645-4bbb-b644-d8d3de8b2b61/2401.04148v1.pdf)

#### 2.2.2 Series decomposition

고정 모델 $\(f\)$ 의 출력 $\(o\)$ 에 대해, STL 스타일의 시계열 분해를 수행한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/336dcc3f-e645-4bbb-b644-d8d3de8b2b61/2401.04148v1.pdf)

1. 추세·순환(trend-cyclical) 성분:

$$o_t = \mathrm{AvgPool}(\mathrm{Padding}(o))$$
   - 시간축 이동평균으로 추세 성분을 뽑는다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/336dcc3f-e645-4bbb-b644-d8d3de8b2b61/2401.04148v1.pdf)

2. 계절(잔차) 성분:

$$o_s = o - o_t$$  

3. 요약:

$$(o_s, o_t) = D(o)$$  
   여기서 $\(D\)$ 는 series decomposition 연산 전체를 나타낸다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/336dcc3f-e645-4bbb-b644-d8d3de8b2b61/2401.04148v1.pdf)

#### 2.2.3 이중 보정 모듈 (Double Correction)

- 계절 성분 보정:

$$\hat{o}_s = g_s(o_s)$$
  여기서 $\(g_s\)$ 는 2‑layer fully-connected + LayerNorm + GELU로 이루어진 MLP이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/336dcc3f-e645-4bbb-b644-d8d3de8b2b61/2401.04148v1.pdf)

- 추세 성분 보정:

$$\hat{o}_t = g_t(o_t)$$
  $\(g_t\)$ 도 동일 구조의 MLP. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/336dcc3f-e645-4bbb-b644-d8d3de8b2b61/2401.04148v1.pdf)

두 모듈 $\(g_s, g_t\)$ 가 논문이 말하는 “lite network”의 핵심이며, 테스트 시점에서 업데이트되는 유일한 비선형 변환이다.  

#### 2.2.4 Adaptive combination (노드별 drift 가중)

- 노드별 drift 강도를 다르게 표현하기 위해,  

$$\lambda_s \in \mathbb{R}^N,\quad \lambda_t \in \mathbb{R}^N$$  
  두 개의 adaptive vector를 도입한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/336dcc3f-e645-4bbb-b644-d8d3de8b2b61/2401.04148v1.pdf)

- 최종 출력:

$$\hat{y} = o + \lambda_s \hat{o}_s + \lambda_t \hat{o}_t$$  
  여기서 $\(\lambda_s, \lambda_t\)$ 는 노드 차원에 따라 broadcast되어 각 노드별 계절/추세 보정량을 스케일링한다. [ar5iv.labs.arxiv](https://ar5iv.labs.arxiv.org/html/2401.04148)

- OTTA에서의 목적함수:

$$\min_{g_s, g_t, \lambda_s, \lambda_t} \; \ell\big(y, \hat{y}\big)$$  
  보통

$$\ell(y, \hat{y}) = \frac{1}{N T' C} \sum_i (y_i - \hat{y}_i)^2$$  
  와 같은 MSE 기반 손실을 사용한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/336dcc3f-e645-4bbb-b644-d8d3de8b2b61/2401.04148v1.pdf)

- 온라인 알고리즘(요약):  
  1. 초기화: $\(g_s, g_t\)$ 는 랜덤, $\(\lambda_s = \lambda_t = 0\)$ . [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/336dcc3f-e645-4bbb-b644-d8d3de8b2b61/2401.04148v1.pdf)
  2. 각 $\(t\)$에 대해:
     - $\(o_t = f(x_t)\)$
     - $\((o_s, o_t) = D(o_t)\)$ 
     - $\(\hat{o}_s = g_s(o_s), \hat{o}_t = g_t(o_t)\)$  
     - $\(\hat{y}_t = o_t + \lambda_s \hat{o}_s + \lambda_t \hat{o}_t\)$  
     - 관측된 $\(y_t\)$ 로 $\(\ell(y_t, \hat{y}_t)\)$ 를 계산, $\(g_s, g_t, \lambda_s, \lambda_t\)$ 만 Adam(lr= $\(10^{-4}\)$ , batch=1, epoch=1)으로 한 스텝 업데이트. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/336dcc3f-e645-4bbb-b644-d8d3de8b2b61/2401.04148v1.pdf)

#### 2.2.5 이론적 분석: lite network 및 residual의 필요성

세 가지 모델을 가정한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/336dcc3f-e645-4bbb-b644-d8d3de8b2b61/2401.04148v1.pdf)

- 모델 1:  

$$\hat{y} = f(x)$$  

- 모델 2 (ADCSD 단순화):  

$$\tilde{y} = f(x) + g(f(x))$$  

- 모델 3 (원 출력 없이 보정만):  

$$\bar{y} = g(f(x))$$  

손실:

$$
\ell(y, \hat{y}) = \frac{1}{N T' C} \sum_{i=1}^{N T' C} (y_i - \hat{y}_i)^2.
$$

- **정리 1**: 적절한 $\(g\)$ 가 존재하여

$$\ell(y, \tilde{y}) < \ell(y, \hat{y})$$
  
  가 성립한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/336dcc3f-e645-4bbb-b644-d8d3de8b2b61/2401.04148v1.pdf)
  → residual 형태로 lite network를 붙인 모델 2가, 모델 1보다 항상 더 낮은 training loss를 구현 가능함을 의미(모델 1은 모델 2의 특수 케이스).  

- **정리 2**: 마찬가지로 적절한 $\(g\)$ 에 대해

$$\ell(y, \tilde{y}) < \ell(y, \bar{y})$$
  
  가 성립한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/336dcc3f-e645-4bbb-b644-d8d3de8b2b61/2401.04148v1.pdf)
  → 원 출력 \(f(x)\)를 제거하고 보정만 사용하는 모델 3은 최적 training loss 관점에서 열등함을 보인다.  

즉, “frozen base model + residual lite network”라는 ADCSD의 구조 선택이 이론적으로도 타당함을 보여준다.  

***

### 2.3 모델 구조 요약

구조 수준에서 ADCSD는 다음 블록으로 구성된다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/336dcc3f-e645-4bbb-b644-d8d3de8b2b61/2401.04148v1.pdf)

1. **고정 history model $\(f\)$ **  
   - ASTGCN, AGCRN, PDFormer 등 어떤 교통 예측 모델도 사용 가능.  

2. **Series Decomposition 블록 $\(D\)$ **  
   - 입력: $\(o = f(x)\)$  
   - 출력: $\((o_s, o_t)\)$  

3. **Seasonal Correction Module $\(g_s\)$ **  
   - 2‑layer FC + LayerNorm + GELU  
   - 출력: $\(\hat{o}_s\)$  

4. **Trend-Cyclical Correction Module $\(g_t\)$ **  
   - 동일 구조, 출력: $\(\hat{o}_t\)$  

5. **Adaptive Combination 레이어**  
   - 파라미터: $\(\lambda_s, \lambda_t \in \mathbb{R}^N\)$  
   - 출력: $\(\hat{y} = o + \lambda_s \hat{o}_s + \lambda_t \hat{o}_t\)$  

테스트 시점에 gradient 업데이트가 일어나는 부분은 $\(g_s, g_t, \lambda_s, \lambda_t\)$ 뿐이고, $\(f\)$ 는 완전히 고정된다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/336dcc3f-e645-4bbb-b644-d8d3de8b2b61/2401.04148v1.pdf)

***

### 2.4 성능 향상 및 한계

#### 2.4.1 성능 향상 (요약 지표)

- **데이터셋**: PeMS07, BayArea(그래프), NYCTaxi, T-Drive(그리드). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/336dcc3f-e645-4bbb-b644-d8d3de8b2b61/2401.04148v1.pdf)
- **베이스라인**: 고정 모델(Test), TENT, TTT-MAE. [arxiv](https://arxiv.org/html/2401.04148v1)

대표적인 결과만 요약하면: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/336dcc3f-e645-4bbb-b644-d8d3de8b2b61/2401.04148v1.pdf)

- PeMS07 / ASTGCN:  
  - Test: MAE = 23.708  
  - ADCSD: MAE = 22.910  

- BayArea / AGCRN:  
  - Test: MAE = 17.169  
  - ADCSD: MAE = 15.507  

- NYCTaxi / PDFormer(inflow):  
  - Test: MAE = 17.150  
  - ADCSD: MAE = 16.987  

- T-Drive / ASTGCN(inflow):  
  - Test: MAE = 41.578  
  - ADCSD: MAE = 41.499  

TENT, TTT‑MAE는 많은 설정에서 Test보다 나쁘고, 특히 TENT는 AGCRN에서 MAE가 2배 이상 악화될 정도로 교통 도메인에서는 부적합함을 보인다. [ar5iv.labs.arxiv](https://ar5iv.labs.arxiv.org/html/2401.04148)

- Horizon별 성능: horizon이 길어질수록(Test 대비) ADCSD의 개선폭이 커져, drift가 누적되는 장기 예측에서 OTTA가 더 큰 효과를 내는 것을 확인한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/336dcc3f-e645-4bbb-b644-d8d3de8b2b61/2401.04148v1.pdf)

- Ablation으로 확인된 점: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/336dcc3f-e645-4bbb-b644-d8d3de8b2b61/2401.04148v1.pdf)
  - 원 출력 $\(o\)$ 제거 → 성능 크게 악화 (residual 필수).  
  - $\(\lambda_s, \lambda_t\)$ 제거 → 노드별 가중이 없어져 항상 성능 하락.  
  - Decomposition 없이 하나의 보정만( $\(y = o + \lambda \hat{o}\)$ ) → 계절/추세 분리보다는 항상 열등.  

- 계산 비용: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/336dcc3f-e645-4bbb-b644-d8d3de8b2b61/2401.04148v1.pdf)
  - NYCTaxi 기준 OTTA 시간: Test < ADCSD < TENT/TTT-MAE.  
  - 학습 파라미터 수는 약 23만 개 정도로 base 모델과 크게 독립; AGCRN/PDFormer에서는 TENT/TTT-MAE보다 적다.  

#### 2.4.2 한계

- $\(t\)$ 에서 예측 후 $\(t+1\)$ 에 $\(y_t\)$ 가 항상 관측된다는 **라벨 지연 가정**이 필요하다. 실제 ITS에서는 센서 결측·지연이 커질 수 있다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/336dcc3f-e645-4bbb-b644-d8d3de8b2b61/2401.04148v1.pdf)
- 출력 시계열 구조를 가진 모델(다단계 예측)을 전제로 하므로, “한 시점만 예측하는 모델”에는 바로 적용하기 어렵다.  
- 테스트 시점에서 역전파가 필요하므로, pure inference에 비해 연산/메모리 비용이 증가한다.  
- 이론 분석은 training loss에 한정되어 있으며, non‑stationary 분포에 대한 일반화 bound는 없다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/336dcc3f-e645-4bbb-b644-d8d3de8b2b61/2401.04148v1.pdf)

***

## 3. 일반화 성능(Generalization) 관점의 논의

### 3.1 Temporal drift에 대한 적응적 일반화

- base 모델 $\(f\)$ 는 $\(p_h\)$ 에서 학습되었지만, ADCSD는 각 시점 $\(t\)$ 의 최신 샘플/라벨을 사용하여 $\(g_s, g_t, \lambda_s, \lambda_t\)$ 를 업데이트하므로, 결과적으로  

$$p_{f_t} \ \text{가 변해도} \ \hat{y}_t = o_t + \lambda_s \hat{o}_s + \lambda_t \hat{o}_t$$  
  
  가 새로운 분포에 맞춰 계속 보정된다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/336dcc3f-e645-4bbb-b644-d8d3de8b2b61/2401.04148v1.pdf)
- horizon이 길수록 개선폭이 커지는 실험 결과는, 시간 경과에 따라 커지는 일반화 실패를 OTTA가 완화한다는 실질적 증거로 볼 수 있다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/336dcc3f-e645-4bbb-b644-d8d3de8b2b61/2401.04148v1.pdf)

### 3.2 아키텍처·도메인 측면의 일반성

- ASTGCN(그래프 CNN + attention), AGCRN(노드‑적응 GCRN), PDFormer(Transformer 계열) 모두에서 ADCSD가 성능을 올린다 → ADCSD는 특정 아키텍처에 특화되지 않고, **출력 시계열 위의 보정 레이어**로 일반적으로 작동한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/336dcc3f-e645-4bbb-b644-d8d3de8b2b61/2401.04148v1.pdf)
- 그래프 기반(도로 센서 네트워크)와 그리드 기반(도시 격자) 모두에서 유사한 개선을 보여, 공간 구조가 달라져도 drift 보정의 일반성이 유지된다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/336dcc3f-e645-4bbb-b644-d8d3de8b2b61/2401.04148v1.pdf)

### 3.3 2020년 이후 관련 연구와의 연결

2020년 이후 분포 변화/OTTA/robust TSF와 관련된 대표 연구들과 비교하면: [arxiv](https://arxiv.org/abs/2501.04970)

- **AdaRNN (2021)**: 학습 시점에서 여러 시간 구간의 분포를 나누고 matching하는 방식으로 drift를 다루지만, 테스트 이후 OTTA는 하지 않는다.  
- **RevIN (2022)**: 입력 정규화/역정규화로 shift에 강건하게 만들지만, 테스트 시점에 파라미터 업데이트는 없음.  
- **Self-adaptive forecasting (2022)**: non‑stationary TSF에서 self-adaptation 전략을 도입하지만 완전한 OTTA라기보다는 training-time 중심.  
- **FlashST (2024)**: 교통 예측용 pre‑trained 모델에 prompt를 붙여 적은 파라미터만 fine-tuning하는 프레임워크.  
- **TAFAS, PETSA, Shift-Aware TTA (2025)**: 일반 TSF를 위한 OTTA 프레임워크로, 부분 라벨, frequency-domain regularization, gating, low‑rank adapter, 벤치마크(TTFBench) 등을 제안.  

ADCSD는 이들보다 **도메인 특화(교통) + 출력 시계열 분해 기반**이라는 강한 inductive bias를 가지며, 교통 도메인에서는 설명력과 안정성이 높고, 향후 PETSA/TAFAS류 일반 TSF‑TTA 프레임워크와 결합할 수 있는 모듈형 헤드로 기능할 잠재력이 크다. [arxiv](https://arxiv.org/html/2506.23424v1)

***

## 4. 앞으로의 연구 영향과 고려할 점 + 2020년 이후 연구 방향

### 4.1 이 논문의 영향

1) **교통 예측에서 OTTA 패러다임을 여는 역할**  
- 교통 예측을 “배포 후 고정된 모델”이 아닌, **운영 중 지속 적응하는 모델**로 보게 만드는 첫 사례 중 하나다. [bonaldli.github](https://bonaldli.github.io/publication/adcsd/)

2) **출력‑단 series decomposition 기반 보정 패턴 제시**  
- 입력 정규화 대신, 예측 결과를  

$$o \xrightarrow{D} (o_s, o_t) \xrightarrow{g_s,g_t} (\hat{o}_s,\hat{o}_t) \xrightarrow{\lambda_s,\lambda_t} \hat{y}$$  
  
  로 보정하는 구조는, 에너지·기상·수요 예측 등 다른 시계열 도메인에도 쉽게 이식 가능하다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/336dcc3f-e645-4bbb-b644-d8d3de8b2b61/2401.04148v1.pdf)

3) **파라미터 효율적 OTTA 설계 사례**  
- base 모델을 전혀 건드리지 않고, 작은 헤드만 업데이트해도 의미 있는 성능 향상이 가능함을 실험·이론으로 보였다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/336dcc3f-e645-4bbb-b644-d8d3de8b2b61/2401.04148v1.pdf)
- 이후 PETSA/TAFAS 등도 같은 철학을 일반 TSF로 확장하고 있어, ADCSD는 도메인 특화된 성공 사례로 참고 가치가 높다. [arxiv](https://arxiv.org/abs/2501.04970)

### 4.2 앞으로 연구 시 고려할 점

1) **라벨 지연·부분 관측 환경에 대한 확장**

- 실제 ITS에서는 일부 시점/노드의 $\(y_t\)$ 가 늦게 오거나 아예 빠질 수 있다.  
- TAFAS/PETSA처럼 **부분 라벨, self-supervised loss, consistency regularization** 등을 접목해, $\(y_t\)$ 가 없을 때도 일부 적응이 가능하도록 확장할 필요가 있다. [arxiv](https://arxiv.org/html/2506.23424v1)

2) **Shift-aware ADCSD**

- Shift-Aware TTA 연구처럼, 분포 변화가 작을 때는 적응 강도를 줄이고, 클 때만 강하게 적응하는 gating을 붙이면 **over‑adaptation과 forgetting**을 완화할 수 있다. [openreview](https://openreview.net/pdf?id=a399SmgWGl)
- 예: 교통량이 안정적인 심야 시간대에는 업데이트 step size를 줄이고, 이벤트/사고 시에는 크게 하는 방식.  

3) **공간 구조를 적극 활용한 OTTA**

- 현재 ADCSD에서 공간 구조 활용은 $\(\lambda_s, \lambda_t \in \mathbb{R}^N\)$ 수준으로 제한적이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/336dcc3f-e645-4bbb-b644-d8d3de8b2b61/2401.04148v1.pdf)
- 향후에는 GNN/attention/meta‑graph를 사용해 **노드 간 상관과 drift를 공동으로 적응**시키는, “graph-aware OTTA 헤드”가 유망하다. [arxiv](https://arxiv.org/html/2212.04475v2)

4) **장기·cross-city 벤치마크에서의 평가**

- Extremely long horizon, cross-city transfer, multi-city pre‑train & fine-tune 등 최신 교통 벤치마크에서 ADCSD류 모듈의 효과를 체계적으로 평가하고, TSF‑TTA 벤치마크(TTFBench)와 결합해 “교통 전용 TTA 벤치마크”를 설계하는 것이 중요하다. [arxiv](https://arxiv.org/html/2406.12693v2)

5) **일반화 이론의 확장**

- 현재 정리 1,2는 training loss 관점의 결과일 뿐, non‑stationary 분포  
  $$\{p_{f_t}\}_{t=1}^T$$  
  에 대한 generalization bound, stability, forgetting trade-off는 제공하지 않는다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/336dcc3f-e645-4bbb-b644-d8d3de8b2b61/2401.04148v1.pdf)
- Kuznetsov & Mohri(2014)류 non‑stationary TSF 일반화 이론을 OTTA 세팅으로 확장해, **적응률/학습률과 일반화 오차의 관계**를 정량화하는 것이 장기적으로 필요한 이론 연구 방향이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/336dcc3f-e645-4bbb-b644-d8d3de8b2b61/2401.04148v1.pdf)

***

### 4.3 2020년 이후 관련 최신 연구(오픈 액세스) 간단 표

| 연도 | 제목 | 링크 | 핵심 아이디어 및 ADCSD와 관계 |
|------|------|------|--------------------------------|
| 2021 | AdaRNN | CIKM 2021 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/336dcc3f-e645-4bbb-b644-d8d3de8b2b61/2401.04148v1.pdf) | temporal distribution matching으로 학습 시 drift를 줄임. OTTA는 아님. ADCSD 위에 결합 가능. |
| 2022 | RevIN | ICLR 2022 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/336dcc3f-e645-4bbb-b644-d8d3de8b2b61/2401.04148v1.pdf) | 입력 정규화/역정규화로 분포 shift 완화. 출력 보정형 ADCSD와 보완적. |
| 2022 | Self-adaptive forecasting | arXiv:2202.02403 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/336dcc3f-e645-4bbb-b644-d8d3de8b2b61/2401.04148v1.pdf) | non‑stationary TSF에서 self-adaptive 전략 제안. 완전한 OTTA는 아님. |
| 2024 | FlashST | arXiv:2405.17898 [arxiv](https://arxiv.org/pdf/2405.17898.pdf) | 교통 예측에 prompt-tuning을 적용, 파라미터 효율적 적응. series decomposition 대신 prompt를 사용한다는 점이 다름. |
| 2024 | Online Test-Time Adaptation of Spatial-Temporal Traffic Flow Forecasting | arXiv:2401.04148 [arxiv](http://arxiv.org/pdf/2401.04148.pdf) | 본 논문. 교통 도메인 첫 OTTA + 시계열 분해 기반 출력 보정. |
| 2025 | TAFAS | arXiv:2501.04970 [arxiv](https://arxiv.org/abs/2501.04970) | 부분 라벨, frequency-domain regularization, gating 기반 일반 TSF‑TTA. ADCSD보다 범용적. |
| 2025 | PETSA | arXiv:2506.23424 [arxiv](https://arxiv.org/html/2506.23424v1) | 입력/출력 adapter + gate로 forecaster를 test-time에 보정. ADCSD와 철학 유사(헤드만 업데이트). |
| 2025 | Shift-Aware TTA & TTFBench | OpenReview [openreview](https://openreview.net/pdf?id=a399SmgWGl) | 다양한 shift 시나리오를 포함한 TSF‑TTA 벤치마크와 shift-aware gating TTA. ADCSD류 교통 OTTA를 일반 벤치마크 위에서 평가할 기반 제공. |

이들 연구를 염두에 두고 ADCSD를 “교통 특화 OTTA 헤드”로 발전시키면서, 일반 TSF‑TTA 프레임워크와 어떻게 통합·비교할지 설계하는 것이 향후 연구에서 중요한 포인트가 될 것이다.

<span style="display:none">[^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39]</span>

<div align="center">⁂</div>

[^1_1]: 2401.04148v1.pdf

[^1_2]: https://arxiv.org/html/2401.04148v1

[^1_3]: https://ar5iv.labs.arxiv.org/html/2401.04148

[^1_4]: https://arxiv.org/html/2506.23424v1

[^1_5]: https://openreview.net/pdf?id=a399SmgWGl

[^1_6]: https://arxiv.org/abs/2501.04970

[^1_7]: https://bonaldli.github.io/publication/adcsd/

[^1_8]: https://arxiv.org/pdf/2405.17898.pdf

[^1_9]: https://arxiv.org/html/2501.10796v1

[^1_10]: https://ieeexplore.ieee.org/document/10400973/

[^1_11]: https://arxiv.org/html/2212.04475v2

[^1_12]: https://arxiv.org/html/2411.11448v1

[^1_13]: https://arxiv.org/html/2406.12693v2

[^1_14]: http://arxiv.org/pdf/2401.04148.pdf

[^1_15]: https://ieeexplore.ieee.org/document/11176852/

[^1_16]: https://www.semanticscholar.org/paper/f4503517aa1cbaff6b5cbef96493d6585206d24f

[^1_17]: https://www.semanticscholar.org/paper/a30daa3f538a6c496f1df18cb6bcb74c32ec5bc8

[^1_18]: https://arxiv.org/pdf/2109.05225.pdf

[^1_19]: http://arxiv.org/pdf/2401.08119.pdf

[^1_20]: http://arxiv.org/pdf/2403.16495.pdf

[^1_21]: https://arxiv.org/abs/2401.04148

[^1_22]: https://arxiv.org/pdf/2401.04148.pdf

[^1_23]: https://www.semanticscholar.org/paper/The-application-of-space-time-ARIMA-model-on-flow-Lin-Huang/ec97803e07a74b67b845337110bb044a35377479

[^1_24]: https://arxiv.org/pdf/2301.09152.pdf

[^1_25]: https://arxiv.org/html/2509.03810v1

[^1_26]: https://arxiv.org/pdf/2506.20762.pdf

[^1_27]: https://arxiv.org/html/2411.03687v1

[^1_28]: https://www.semanticscholar.org/author/Pengxin-Guo/94688673

[^1_29]: https://arxiv.org/list/cs/new

[^1_30]: https://www.arxiv.org/pdf/2601.12893.pdf

[^1_31]: https://arxiv.org/html/2601.12083v1

[^1_32]: https://arxiv.org/html/2507.09095v2

[^1_33]: https://www.emergentmind.com/papers/2401.04148

[^1_34]: https://openreview.net/forum?id=pdyPXHyo6Q

[^1_35]: https://huggingface.co/papers/2401.04148

[^1_36]: https://www.themoonlight.io/ko/review/online-test-time-adaptation-of-spatial-temporal-traffic-flow-forecasting

[^1_37]: https://www.frontiersin.org/journals/sustainable-cities/articles/10.3389/frsc.2025.1631748/full

[^1_38]: https://github.com/Pengxin-Guo/ADCSD

[^1_39]: https://mediatum.ub.tum.de/doc/1788410/qj6nckhx87s9jcsikpvgdrs6z.pdf

