# Frequency-domain MLPs are More Effective Learners in Time Series Forecasting

# 핵심 요약

**주장 및 기여:**  
Frequency-domain MLPs(이하 FreMLP)를 도입한 FreTS 모델은 시계열 예측 시 기존 시계열 도메인 MLP가 갖던 정보 병목과 국소 의존성 한계를 극복하고, 주파수 스펙트럼의 전역 의존성 학습 및 에너지 응집(energy compaction) 특성을 활용해 예측 성능과 일반화 능력을 모두 향상시킨다.[1]

***

## 1. 해결하고자 하는 문제

전통적인 시계열 예측 모델(RNN, CNN, Transformer, GNN 등)은 복잡한 구조로 연산 비용이 높고, 파라미터 수가 많아 데이터가 한정적일 때 과적합 위험이 크다. 최근 MLP 기반 방법(N-BEATS, DLinear 등)은 구조가 단순하고 계산 효율이 뛰어나지만,  
-  시계열을 시점별(point-wise) 매핑으로만 처리해 전역 의존성을 포착하기 어렵고  
-  국소 변동성과 잡음에 취약해 정보 병목이 발생한다.[1]

***

## 2. 제안 방법

FreTS는 시계열을 주파수 영역으로 변환하여 MLP를 적용하는 **두 단계**로 구성된다.[1]

① **도메인 변환(Domain Conversion)**  
– 입력 시계열 $$H\in\mathbb{R}^{N\times L\times d}$$를 이산 푸리에 변환(DFT)으로 복소수 스펙트럼 $$H_f=\Re H_f + j\,\Im H_f$$로 변환  

$$
H_f[v,f] \;=\;\sum_{t=0}^{L-1} H[v,t]\,e^{-2\pi i f t / L}
$$  

② **주파수 학습(Frequency Learning)**  
– **채널 학습(Channel Learner):** 각 시점에 대해 채널 차원별로 복소수 MLP(FreMLP) 적용  

$$
Z_{\text{chan}} = \text{IDFT}\bigl(\text{FreMLP}\bigl(\text{DFT}_{\text{chan}}(H)\bigr)\bigr)
$$  

– **시간 학습(Temporal Learner):** 각 채널에 대해 시간 차원별로 복소수 MLP 적용  

$$
S_{\text{temp}} = \text{IDFT}\bigl(\text{FreMLP}\bigl(\text{DFT}_{\text{temp}}(Z)\bigr)\bigr)
$$  

– 최종 예측을 위해 두 단계를 거친 출력을 FFN에 투입  

$$
Y = \text{FFN}(S)
$$  

FreMLP는 실수부·허수부 분리 연산으로 구현되며, 복소수 매개변수 $$W = W^r + jW^i$$, $$B = B^r + jB^i$$를 갖는다.[1]

***

## 3. 주요 특성

– **전역 시각(Global View):** 주파수 스펙트럼을 활용해 전 시점에 걸친 의존성을 한 번에 학습하는 전역 컨볼루션 효과를 보임$$\bigl(f(\cdot)$$이 DFT 연산$$)$$.[1]
– **에너지 응집(Energy Compaction):** 시계열 에너지가 소수의 주파수 성분에 응집됨을 이용하여, 잡음 성분을 무시하고 핵심 패턴만 학습.[1]

이론적으로 시계열 도메인과 주파수 도메인 에너지가 동일하며(Plancherel 정리), 주파수 도메인 MLP는 시계열 도메인의 전역 컨볼루션과 동일한 연산 효과를 가진다.[1]

***

## 4. 성능 향상 및 한계

### 성능  
-  **단기 예측:** 6개 데이터셋(METR-LA, Traffic, Electricity 등)에서 MAE 9.4%, RMSE 11.6% 개선.[1]
-  **장기 예측:** 6개 데이터셋(Weather, Exchange 등)에서 Transformer 기반 SOTA 대비 MAE·RMSE 20% 이상 감소.[1]

### 한계  
– **채널 학습 오버피팅:** 예측 구간이 매우 길어질수록(channel learner 사용 시) 과적합이 발생, 채널 독립 전략이 필요할 수 있음.[1]
– **복소수 연산 비용:** DFT·IDFT 및 복소수 MLP 연산으로 연산 복잡도가 $$O(N\log N + L\log L)$$로 증가.[1]

***

## 5. 일반화 성능 향상 가능성

주파수 도메인의 전역 뷰와 에너지 응집 특성은  
-  노이즈·변동성이 큰 시계열에서도 핵심 주기성(periodicity) 패턴을 안정적으로 추출하여  
-  데이터가 제한적인 상황에서 과적합을 억제하고 일반화 성능을 향상시킨다.  

실험적 검증에서는 예측 구간 확대에도 안정적 성능 유지를 보였으며, 채널·시간 학습의 조합 및 하이퍼파라미터 조정으로 더욱 견고한 일반화를 달성할 수 있다.[1]

***

## 6. 향후 연구 영향 및 고려 사항

– **다중 해상도 주파수 학습:** FFT 기반이 아닌 웨이브릿 변환을 통한 다중 해상도 스펙트럼 학습으로 국소·전역 패턴 동시 포착 가능성.  
– **채널 독립성 vs. 상호작용:** 예측 길이에 따른 채널 학습 유무 전략 연구.  
– **경량화 및 실시간 적용:** 복소수 연산 최적화, 근사 DFT 알고리즘 적용으로 연산 비용 절감.  
– **비정형 시계열 및 이상 탐지:** 불규칙 샘플링·결측치 보정, 이상치 검출에 주파수 MLP 적용성.  

이 논문은 단순 MLP를 주파수 도메인으로 확장함으로써 시계열 예측 모델링 패러다임에 새로운 방향을 제시하며, 다양한 분야의 시계열 분석 연구에 큰 영향을 미칠 것으로 기대된다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/353ac3c5-6725-4b4d-b9b2-065a861a10b1/2311.06184v1.pdf)
