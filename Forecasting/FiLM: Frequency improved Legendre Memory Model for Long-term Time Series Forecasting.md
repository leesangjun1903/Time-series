# FiLM: Frequency improved Legendre Memory Model for Long-term Time Series Forecasting

## 주요 주장 및 기여
**FiLM**(Frequency improved Legendre Memory Model)은 장기 시계열 예측에서  
-  역사 정보의 완전한 보존  
-  노이즈 제거  
두 가지를 동시에 달성하며, 기존 최첨단 모델 대비 다변량 및 단변량 예측 정확도를 각각 20.3%, 22.6% 향상시킨다.  
핵심 기여:  
1. **Legendre Projection Unit(LPU)**: 고차원 Legendre 다항식 투영을 통해 긴 시계열을 압축·표현.  
2. **Frequency Enhanced Layer(FEL)**: 저주파 성분 중심의 Fourier 투영과 저차원 근사로 노이즈 제거.  
3. **멀티스케일 전문가 집합**으로 복수 창(길이 T, 2T, … nT)에서 패턴 추출.  
4. **플러그인 모듈성**: LPU는 타 모델에 간단히 결합 가능하며 예측 성능을 개선.  

## 문제 정의
장기 예측(long-term forecasting)은 전통적 ARIMA 등 선형 모델로는 긴 의존성 및 잡음에 취약하여 불가능하다.  
Transformer·RNN 기반 딥러닝도 과도한 복잡도로 잡음을 과적합(overfit)하여 예측 분포가 실제와 크게 벗어나는 문제를 보인다. 따라서  
1) 중요한 역사 신호 보존  
2) 잡음 효과 최소화  
두 가지를 균형 있게 달성할 **견고한 시계열 표현**이 필요하다.[1]

## 제안 방법
### Legendre Projection Unit (LPU)
- 입력 $$x_t$$을 Legendre 다항식 차원 $$N$$으로 투영:  

$$\frac{d}{dt}c(t) = A\,c(t) + B\,f(t)$$  
  
  여기서 $$c(t)\in\mathbb{R}^N$$는 압축 메모리, $$A,B$$는 고정 행렬.
- 큰 $$N$$일수록 정확도는 상승하나 잡음 과적합 우려가 있음.

### Frequency Enhanced Layer (FEL)
- LPU 출력 계수 $$c_n(t)$$에 대해 저주파 성분만 유지:  
  1) $$x\to\mathcal{F}\{c\}$$  
  2) 상위 M 모드를 보존하고 나머지 무작위 샘플링  
  3) 저차원 근사(Low-rank)로 가중치 행렬 $$W$$ 분해  
- 이론적 보장: 고유값 최소 성분 $$\alpha_{\min}$$가 작으면 원공간 근사 손실 $$\mathcal{O}\bigl(\frac{1}{\alpha_{\min}}\bigr)$$.

### 모델 구조
1. 입력 정규화(선택적 RevIN)  
2. LPU → FEL → LPU 복원  
3. 멀티스케일 전문가 예측 결합  
전체는 단일 LPU·FEL 레이어로 구성되어 단순함.

## 성능 향상
- **다변량 예측**: 6개 벤치마크 데이터셋에서 평균 MSE 20.3% 감소  
- **단변량 예측**: 평균 MSE 22.6% 감소  
- **일반화 성능**: Kolmogorov–Smirnov 검정에서 출력 분포와 입력 분포 유사성 유지.  
- **파라미터 효율성**: Transformer 대비 학습 가능 파라미터 80% 절감.  
- **메모리·속도**: 출력 길이 증가에도 메모리 사용 선형, 대용량 배치 지원으로 학습 속도 우수.

## 한계 및 고려사항
- 입력 변수 수가 매우 많은 다변량 예측에서 RevIN 적용 시 학습 속도 2–5배 저하  
- 멀티스케일 전문가 지점 수(n)·LPU 차원(N)·FEL 모드 수(M) 하이퍼파라미터 민감  
- 고주파 성분 일부 필요 시 ‘Low random’ 정책이 유용

## 향후 연구 영향 및 고려사항
FiLM 프레임워크는 **일반적 빌딩 블록**으로 확장 가능하다.  
- LPU를 Chebyshev, Laguerre, Wavelet 등 다른 직교 기저로 교체  
- FEL을 주파수 도메인 필터링 외에 학습 기반 잡음 모델로 발전  
- 멀티스케일 전문가 구조를 동적·적응적 융합 메커니즘과 결합  
- 시간에 따른 분포 변화(비정상성) 대응을 위한 동적 창 크기 및 창 가중치 학습  
이와 같이 FiLM은 다양한 변형·응용을 통해 장기 시계열 예측의 새로운 패러다임을 제시할 것이다.

***

1 2 3 4 15 16 9 4[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/907fa6ce-d735-4d39-9745-a4c3f3df0c95/2205.08897v4.pdf)
