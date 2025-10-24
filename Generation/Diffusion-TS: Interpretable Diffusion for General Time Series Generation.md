# Diffusion-TS: Interpretable Diffusion for General Time Series Generation

**핵심 주장 및 주요 기여**  
Diffusion-TS는 시계열 데이터를 위한 비자기회귀(non-autoregressive) 확산 모델로, 추세(trend)와 계절성(seasonality)을 분리하는 해석 가능(interpretable)한 디코더 구조와 푸리에 기반 손실을 결합하여 고품질의 다변량 시계열 샘플을 생성한다. 기존 확산 모델은 시계열의 복합적인 주기성과 장기 의존성을 회복하지 못했으나, Diffusion-TS는 디코더 내에 추세·계절·오차 컴포넌트를 분리 학습함으로써 시맨틱한 시간 해석을 제공하며, 노이즈가 더해진 입력으로부터 직접 원본을 재구성하는 방식을 채택해 생성 정확도를 높였다.[1]

## 1. 문제 정의  
- 목표: 주기·추세·잔차가 복합적으로 얽힌 장·단기 다변량 시계열 $$X = \{x_1, \dots, x_T\}\subset\mathbb R^{d\times T}$$의 분포를 근사해 새로운 샘플을 생성  
- 제어 가능 생성: 결측치 대체(imputation)나 예측(forecasting)과 같은 조건부 생성 $$\,p(x_0\mid y)$$로 확장 가능  

## 2. 제안 방법  
### 2.1 확산 프레임워크  
- 전진 과정(forward):  

$$q(x_t\mid x_{t-1})=\mathcal N(x_t;\sqrt{1-\beta_t}x_{t-1},\beta_tI)$$  

- 역전 과정(reverse): 네트워크가 노이즈가 섞인 $$x_t$$로부터 깨끗한 $$x_0$$를 직접 예측하도록 학습  

$$\mathcal L_{\mathrm{simple}} = \mathbb E_{t,x_0}\bigl[w_t\|x_0 - \hat x_0(x_t,t)\|^2\bigr]$$  

- 푸리에 손실: 시간·주파수 도메인 재구성 오차를 결합  

$$\mathcal L = \mathbb E_{t,x_0}\bigl[w_t\|x_0-\hat x_0\|^2 + \lambda\|\mathrm{FFT}(x_0)-\mathrm{FFT}(\hat x_0)\|^2\bigr]$$ [1]

### 2.2 해석 가능 디코더 구조  
- 인코더: 트랜스포머 블록으로 전체 시퀀스의 글로벌 패턴 포착  
- 디코더: 각 블록에서  
  1) **추세 합성**: 다항 회귀(polynomial regressor)로 느리게 변하는 추세 $$V^{\text{tr}}_t$$ 모델링  
  2) **푸리에 합성**: DFT로부터 상위 $$K$$개 주파수 성분을 선택해 계절성 $$S_t$$ 복원  
  3) **오차**: 잔차 $$R_t$$로 남은 노이즈 처리  
- 최종 재구성: $$\hat x_0 = V^{\text{tr}}_t + S_t + R_t$$[1]

### 2.3 조건부 생성  
- **재구성 기반 샘플링**: 결측치(imputation)나 예측(forecasting) 시, 역전 과정에 gradient guidance를 결합해  
  $$\nabla_{x_t}\log p(y\mid x_{t-1})$$ 를 사용한 Langevin 업데이트로 조건 부합성을 강화.[1]

## 3. 성능 향상 및 한계  
- **무조건부 생성**: 여섯 개 데이터셋(Stocks, ETTh, Energy 등) 전반에서 기존 GAN·VAE·다른 확산 모델 대비 우수한 Discriminative·Predictive·Context-FID·Correlational 스코어 달성.[1]
- **장기 생성**: 시퀀스 길이 24→256 증가에도 성능 저하가 거의 없어 장기 의존성 모델링에 강함.[1]
- **조건부 생성**: 결측 비율 90%에서도 CSDI 대비 낮은 MSE 달성, reconstruction guidance의 효과 입증.[1]
- **한계**: DDPM 기반 특성상 추론(inference) 단계가 GAN 대비 느리고 연산 자원 부담이 큼. 보다 빠른 수렴을 위한 샘플링 최적화 필요.[1]

## 4. 일반화 성능 개선 관점  
- **디스엔탱글먼트(Disentanglement)**: 추세·계절·오차 분리 학습이 잦은 패턴 변화에도 안정적 복원력 제공  
- **트랜스포머 백본**: 장·단기 의존성 모두 효과적으로 캡처, 다양한 주기 구조 일반화 성능 향상  
- **푸리에 손실**: 저주파부터 고주파까지 재구성 품질 고르게 개선, 과적합 방지 및 일반화 강화  

## 5. 향후 영향 및 고려사항  
Diffusion-TS는 시계열 생성, 결측치 대체, 예측 분야에 폭넓게 적용 가능하며, 특히 의료·금융·기후 데이터 등 민감·고차원 시계열 합성에 기여할 전망이다.  
- **고속 샘플링 연구**: DDPM 경량화·지연 시간 단축을 위한 ODE 기반 샘플러나 적응형 스텝 스케줄링 개발  
- **더 복합한 제어 변수**: 다중 제약(예: 외생 변수·이벤트) 통합된 조건부 생성 프레임워크 확장  
- **해석 가능성 심화**: 각 주파수 성분·추세 컴포넌트의 의미 해석 및 전문가 지식 결합 방안 모색

---  
본 요약에서는 Diffusion-TS의 문제 정의, 수식·모델 구조, 성능 개선 및 한계를 중점적으로 기술하였으며, 일반화 성능 강화 요인을 분석하였다. 향후 연구에서는 샘플링 효율화와 복합 제약 통합이 핵심 고려사항이 될 것이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5e5c8208-6686-40f5-8b56-f19aba048aff/2403.01742v3.pdf)
