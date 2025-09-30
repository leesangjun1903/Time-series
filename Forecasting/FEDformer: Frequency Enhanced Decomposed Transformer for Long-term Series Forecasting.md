# FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting

**핵심 요약**  
FEDformer는 Transformer 기반의 장기 시계열 예측 모델로, 시계열의 전역 추세를 포착하기 위해 계절-추세 분해(seasonal-trend decomposition)와 주파수 도메인 표현을 결합한다. 이로써 예측 오차를 획기적으로 줄이면서도 계산 복잡도를 시퀀스 길이에 선형 비례하도록 낮춘다.[1]

***

## 1. 해결하려는 문제  
기존 Transformer 기반 시계열 모델은 점별(point-wise) 어텐션으로 인해 시계열 전체 분포의 전역 특성을 유지하지 못하고, 복잡도가 시퀀스 길이 제곱에 비례해 장기 예측에 비효율적이다.[1]

***

## 2. 제안 방법  
1) 계절-추세 분해(Mixture of Experts Decomposition)  
   - 다양한 윈도우 크기($$k\in\{7,12,24,48\}$$)의 평균 풀링 필터를 전문가(expert)로 사용해 여러 추세 $$\{T_i\}$$를 추출  
   - 가중합 $$\displaystyle T_{\text{final}} = \sum_i w_i T_i,\quad w=\mathrm{softmax}(F(x))$$  
2) 주파수 강화 블록(Frequency Enhanced Block, FEB)  
   - 입력 $$\mathbf{x}\in\mathbb{R}^{N\times d}$$를 선형 사상 후 DFT 적용:  

$$
       Q = \mathcal{F}(xW),\quad Q_{\text{sel}} = \mathrm{Select}(Q)\in\mathbb{C}^{M\times d}
     $$  
   
   - 선택된 모드만 남겨 역DFT로 복원하여 전역 주파수 특성 학습:[1]

$$
       \mathrm{FEB}(q)=\mathcal{F}^{-1}\bigl(\mathrm{ZeroPad}(R\cdot Q_{\text{sel}})\bigr)
     $$  

3) 주파수 강화 어텐션(Frequency Enhanced Attention, FEA)  
   - 크로스 어텐션의 쿼리·키·값에도 DFT→Select→역DFT 과정을 적용해 전역 주파수 기반 상관성 학습  
   - 활성화로는 softmax 또는 tanh 사용  
4) 이론적 보증  
   - 임의로 선택한 $$s$$개의 주파수 성분만으로도 원본 DFT 행렬을 근사 보존하는 코헤런스 정리(Theorem 1)[1]
   - 선형 복잡도 $$\mathcal{O}(L)$$ 및 메모리 사용 보장  

***

## 3. 모델 구조  
FEDformer는 다층 인코더–디코더 구조로, 각 레이어마다  
- **MOEDecomp** → **FEB** (또는 **FEA**) → **FeedForward** 블록이 차례로 구성된다.  
- FEB/FEA는 Fourier 기반(FEB-f, FEA-f)과 Wavelet 기반(FEB-w, FEA-w) 두 변형을 지원한다.[1]

***

## 4. 성능 향상  
- 6개 벤치마크 데이터셋(에너지, 교통, 경제, 날씨, 질병)에서 멀티/유니버리트 예측 MSE를 각각 평균 **14.8%**, **22.6%** 절감  
- 기존 Autoformer 대비 장기 예측(96→720) 구간에서도 안정적 성능 유지  
- KS 검정으로 예측 분포와 입력 분포의 일치도 확인, 유일하게 모든 구간에서 귀무가설(동일 분포) 기각되지 않음[1]

***

## 5. 일반화 성능 개선  
- 무작위로 선택된 주파수 모드는 사전 지식 없이 다양한 시계열 특성에 적응 가능  
- Coherence 정리에 따라 저차원 주파수 표현만으로도 원본 정보 손실 최소화  
- Wavelet 버전은 지역 구조 포착에 강점, Fourier 버전은 전역 추세에 효과적  
- 실험에서 Univariate vs. Multivariate, 복잡도 지표(Permutation/SVD entropy)에 따라 모드 수 및 변형 선택으로 일반화 성능 최적화 가능  

***

## 6. 한계 및 고려사항  
- 주파수 성분 수 $$M$$·분해 단계 $$L$$ 하이퍼파라미터에 민감  
- Wavelet 변형은 파라미터·연산량 증가  
- 비주기적(non-periodic) 노이즈나 이상치에 대한 견고성 추가 연구 필요  

***

## 7. 향후 연구 방향 및 고려점  
- **동적 모드 선택**: 시계열 특성에 따라 학습 중 모드 수·주파수 대역을 적응적으로 조정  
- **노이즈 견고성**: 이상치 탐지 또는 노이즈 억제를 위한 분해 블록 확장  
- **멀티모달 융합**: 외부 이벤트, 범주형 변수와 결합한 시계열 예측  
- **트랜스퍼 학습**: 다양한 도메인 간 주파수 표현 공유로 데이터 효율적 학습  
- **실시간 적용**: 선형 복잡도를 활용한 엣지·임베디드 환경으로 확장 가능성 모색  

FEDformer는 주파수 도메인과 분해형 구조를 결합하여 장기 예측의 정확성과 효율성을 동시에 달성한 혁신적인 모델로, 향후 시계열 연구 전반에 걸쳐 주파수 기반 Transformer 설계 패러다임을 확장하는 데 중요한 이정표가 될 것이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/8ec88810-79a6-44f4-890d-65185c51153d/2201.12740v3.pdf)
