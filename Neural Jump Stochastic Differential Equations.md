# Neural Jump Stochastic Differential Equations

## 1. 핵심 주장과 주요 기여 (간결 요약)
본 논문은 **연속적 시간 흐름(continuous flow)과 불연속적 사건 점프(discrete jumps)**를 모두 학습 가능한 데이터 기반 하이브리드 동적 모델인 **Neural Jump Stochastic Differential Equations (Neural JSDEs)**를 제안한다.  
- 연속 변화는 Neural ODE로, 불연속 변화는 이벤트 조건부 강도(intensity)와 점프 함수를 뉴럴 네트워크로 학습  
- 두 구성요소를 통합하여 시간에 따라 발생하는 이벤트 시점에 잠재 상태(latent state)가 즉시 불연속적으로 갱신  
- 이 모델로 전통적 Poisson, Hawkes, self-correcting 프로세스 및 마크드 시분할(point) 프로세스를 모두 포괄적이고 효율적으로 예측  

## 2. 상세 분석

### 2.1 해결하고자 하는 문제
- **하이브리드 시스템**: 실제 시계열 데이터는 연속적 변화와 이따금 발생하는 불연속 이벤트에 의해 함께 생성됨  
- 기존 Neural ODE는 연속 흐름은 모델링 가능하나, **이벤트에 의한 즉각적 충격(jumps)**을 반영할 수 없음  
- 반면 점프 확률이나 충격 크기를 미리 정의된 수식으로 제한하면 데이터 특성 반영이 부족  

### 2.2 제안하는 방법
Neural JSDE는 잠재 상태 $$z(t)\in\mathbb{R}^n$$의 연속 흐름과 점프를 동시에 모델링한다.

1) **잠재 상태 동역학**  
   
$$
   dz(t) = f\bigl(z(t),t;\theta\bigr)dt + w\bigl(z(t),k(t),t;\theta\bigr)dN(t)
   $$  
   
   - $$f$$: 연속 변화(Neural ODE)  
   - $$w$$: 이벤트 시점에 잠재 상태에 더해지는 점프 함수  
   - $$N(t)$$: 이벤트 카운팅 프로세스  

2) **이벤트 발생 확률 (조건부 강도)**  
   
$$
   P\{\text{event in }[t, t+dt)\mid H_t\} = \lambda\bigl(z(t)\bigr)dt
   $$  
   
   - $$\lambda(z(t))$$: 잠재 상태를 입력으로 하는 MLP로 학습  
   - 이벤트 유형 $$k(t)$$은 $$\,p\bigl(k\mid z(t)\bigr)$$ (discrete는 softmax, continuous는 Gaussian mixture)

3) **로그우도 손실 함수**  
   
$$
   L = -\sum_j\log\lambda\bigl(z(\tau_j)\bigr)-\sum_j\log p\bigl(k_j\mid z(\tau_j)\bigr)+\int_{t_0}^{t_N}\lambda\bigl(z(t)\bigr)dt
   $$

4) **학습: 불연속을 고려한 역전파**  
   - 연속 구간은 Neural ODE의 **adjoint method**로 처리  
   - 이벤트 시점 $$\tau_j$$에서 잠재 상태와 adjoint 벡터가 모두 불연속 점프  
   - 점프 시 adjoint 보정:  
     
$$
     a(\tau_j)=a(\tau_j^+)+a(\tau_j^+)\frac{\partial w}{\partial z}(z(\tau_j),k_j,\tau_j)
     $$  
   
   - 유사하게 파라미터 그레이디언트 $$a_\theta$$, 시간 그레이디언트 $$a_t$$도 갱신  

### 2.3 성능 향상
- **합성 데이터 실험**: Poisson, Hawkes(지수·거듭제곱), self-correcting 프로세스에서 **평균 절대 백분율 오차**가 RNN 대비 평균 2–4배 개선  
- **이벤트 유형 분류**: Stack Overflow 배지 예측, 의료 방문 이유 분류에서 기존 RNN·LSTM 기반 모델과 유사한 정확도 달성  
- **실수형 특징 예측**: 합성 Hawkes 시퀀스에서 시간 간격 예측 오류 0.35 vs. 단순 평균 예측 3.65  
- **지진 데이터**: 1970–2006 학습 후 2007–2018 발생 위치·시점의 조건부 강도 분포 정확히 포착  

### 2.4 한계 및 향후 과제
- **계산 복잡도**: Neural ODE와 stochastic simulation 병합으로 학습 시 연속 구간 적분 + 이벤트 점프 처리 부담  
- **스케일링**: 대규모 이벤트 시퀀스나 고차원 잠재 상태(n≫50)에서 메모리·시간 비용 증가  
- **가정**: 이벤트는 잠재 상태에만 점프를 일으키고, 내부 상태(c)에는 직접 영향 없음($$\Delta c(t)=0$$)  
- **확률 분포 제약**: 실수형 특징은 Gaussian mixture로만 모델링, 복잡 분포엔 부적합할 수 있음  

***

**결론**: Neural JSDE는 연속 및 불연속 시계열 동작을 통합 학습하는 새로운 프레임워크로, 다양한 시뮬레이션·예측 과제에서 기존 모델을 능가하거나 유사 성능을 보이며 하이브리드 동적 시스템 연구에 중요한 기반을 제시한다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/517cf7a4-4e80-44d3-9f93-c71185b04e66/1905.10403v3.pdf
