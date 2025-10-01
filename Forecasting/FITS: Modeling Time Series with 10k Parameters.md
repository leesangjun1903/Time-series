# FITS: Modeling Time Series with 10k Parameters

**핵심 주장 및 주요 기여**  
FITS는 시계열 예측과 이상 탐지에 모두 적용 가능한 **경량(≈10k 매개변수)** 모델로, 복소수 주파수 영역에서 보간(interpolation)을 수행하여 기존 SOTA 모델과 유사한 성능을 보이면서도 매개변수 수를 수십만 배 절감한다. 주요 기여는 다음과 같다:[1]
- 복소수 값 복합 선형 계층(complex-valued linear layer)을 활용해 진폭(amplitude) 및 위상(phase) 정보를 동시에 학습.  
- 저역 통과 필터(LPF) 적용으로 불필요한 고주파 성분 제거, 모델 크기 추가 절감.  
- 예측(forecasting)과 재구성(reconstruction) 모두 동일한 파이프라인으로 처리 가능.

***

## 1. 해결하고자 하는 문제
현대 시계열 모델은 Transformer 기반 구조나 딥 CNN/RNN을 활용해 복잡한 시간 특징을 학습하지만, 수백만~수억 개 매개변수를 요구해 엣지 장치(edge devices)에는 부적합하다.[1]
주파수 영역 변환(FFT)은 시간 도메인보다 더 압축된 표현을 제공하나, 기존 연구는 주로 주파수 에너지 스펙트럼(feature)만 보조적으로 활용했을 뿐, **복소수 주파수 영역에서 직접 보간을 수행**하지 않았다.

***

## 2. 제안 방법 (수식 포함)
1) **rFFT 변환**  
   입력 시계열 $$x \in \mathbb{R}^N$$에 대해 실수 FFT(rFFT)를 적용하여 복소수 주파수 표현 $$\mathbf{X} \in \mathbb{C}^{N/2+1}$$로 변환한다.

2) **정규화 및 LPF**  
   - 복소수 표현의 DC 성분 편중을 완화하기 위해 *reversible instance normalization*(RIN) 적용.  
   - 컷오프 주파수 COF 이하 성분만 남기는 LPF 수행.

3) **복소수 선형 보간**  
   - 출력 길이 비율 $$\alpha = L_o / L_i$$에 따라 입력 길이 $$L$$에서 출력 길이 $$\alpha L$$로 선형 보간:  

$$
       \hat{X}_k = W \, X_{\lfloor k/\alpha \rfloor} + b,\quad W,b \in \mathbb{C}
     $$  
   
   - 진폭 스케일링과 위상 이동을 복소수 곱으로 동시에 학습.

4) **irFFT 복원 및 손실 계산**  
   - 보간된 주파수 $$\hat{\mathbf{X}}$$에 역 rFFT(irFFT) 적용해 시간 도메인 $$\hat{x}$$ 복원.  
   - 예측(forecast) 시 MSE, 재구성(reconstruction) 시 재구성 손실 사용해 학습.

***

## 3. 모델 구조
- **입력**: 길이 $$L$$ 시계열 세그먼트  
- **rFFT → RIN → LPF → Complex Linear Layer → Zero padding → irFFT → iRIN**  
- 파이프라인은 예측과 재구성에 공통 적용되며, 예측 시 backcast(look-back)와 forecast(예측 구간)에 모두 손실을 부여할 수 있다.[1]

***

## 4. 성능 향상 및 비교
- **예측 성능**: ETT, Weather, Traffic, Electricity 등 광범위한 벤치마크에서 SOTA 모델과 비교해 유사하거나 더 우수한 MSE 달성.[1]
- **매개변수 절감**: DLinear(≈140k)의 1/10 이하, PatchTST(≈1.5M)의 1/100 이하, TimesNet(≈300M)의 1/30,000 이하.[1]
- **실행 속도**: CPU 환경에서 sub-millisecond 추론 가능, 엣지 환경 적합.  
- **이상 탐지**: 복원 손실 기반 자가 감독 방식으로 SMD, SWaT 등에서 F1≈99% 이상 기록.[1]

***

## 5. 한계 및 일반화 성능
- **고주파 정보 손실**: LPF로 제거된 고주파 성분은 잡음 억제에 유리하나, 본질적으로 중요한 고주파 패턴을 상실할 수 있다.  
- **분포 이동**: ETTh1 등 분포가 급변하는 데이터셋에서는 긴 look-back이 오히려 성능을 저하시킴.[1]
- **이벤트형 시계열**: SMAP, MSL의 이진(event) 시계열은 주파수 표현 학습에 한계, time-domain 모델이 유리.[1]
- **일반화 향상**:  
  - 주파수 보간 방식은 과적합 우려가 적어, 작은 학습 데이터셋에서도 **강한 일반화 성능**을 보인다.  
  - backcast+forecast 병합 손실은 look-back 구간 재구성 능력을 향상시켜 예측 안정성을 증가시킨다.[1]

***

## 6. 향후 연구 영향 및 고려 사항
- **엣지 AI 확대**: FITS 구조는 매개변수와 연산량이 작아 IoT·웨어러블·모바일 환경에서 시계열 분석과 이상 탐지에 바로 활용 가능하다.  
- **복소수 딥러닝 확장**: 향후 복소수 Transformer나 대규모 complex-valued 네트워크 연구의 토대로 작용할 수 있다.  
- **자동 COF 선정**: 현재 매뉴얼로 선택하는 컷오프 주파수를 **하모닉 분석 기반 자동화**하거나 데이터 적응적 기법으로 개선 필요.  
- **하이브리드 모델**: 이벤트형 시계열 처리 성능 향상을 위해 time-domain 모듈과 복합 주파수 모듈을 결합한 **하이브리드 아키텍처** 탐구 권장.  

이상의 분석은 arXiv:2307.03756v3의 내용을 기반으로 정리되었다.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9f5d3e37-2a3d-4541-94bd-9349a2541ef8/2307.03756v3.pdf)
