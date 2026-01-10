
# Full-waveform Inversion, Part 2: Adjoint Modeling

## 1. 논문의 핵심 주장 및 주요 기여 요약

본 논문은 2018년 The Leading Edge에 발표된 FWI 튜토리얼 시리즈의 제2부로, **Adjoint-state 방법을 통한 효율적인 그래디언트 계산**을 핵심 주장으로 삼고 있습니다.[1]

### 1.1 주요 기여

**첫 번째**, Devito 프레임워크를 활용한 **실제 구현 가능한 이산화된 adjoint 모델링 기법**을 제시했습니다. 저자들은 3단계 그래디언트 계산 프로세스를 명확히 정의했습니다: (1) 순방향 파동방정식 해, (2) 예측-관측 데이터 간 잔차 계산, (3) adjoint 파동방정식을 통한 민감도 커널 도출.[1]

**두 번째**, 최소제곱 FWI 목적함수에 대한 **엄밀한 수식 유도**를 제공했습니다. 특히 adjoint-state 방법이 forward 파동장과 adjoint 파동장의 crosscorrelation을 통해 그래디언트를 계산함을 보였습니다.[1]

**세 번째**, **재현 가능한 코드**를 공개함으로써 학술 커뮤니티의 접근성을 크게 높였습니다.

## 2. 해결하고자 하는 문제, 제안 방법 및 수식

### 2.1 문제 정의

FWI는 다음의 비선형 최소제곱 최적화 문제로 정의됩니다:[1]

$$\min_m f(m) = \frac{1}{2} \sum_{i=1}^{n_s} \left\| \mathbf{d}_i^{\text{pred}}(m, \mathbf{q}_i) - \mathbf{d}_i^{\text{obs}} \right\|_2^2$$

여기서 $m$은 제곱 느린속도(squared slowness), $n_s$는 발진원 개수, $\mathbf{d}^{\text{pred}}$와 $\mathbf{d}^{\text{obs}}$는 각각 예측 및 관측 데이터입니다.[1]

### 2.2 Continuous Adjoint 파동방정식

연속 도메인에서 adjoint 파동방정식은 다음과 같이 정의됩니다:[1]

$$m(x,y) \frac{\partial^2 v}{\partial t^2} - \Delta v - H(t,x,y) = \delta d(t,x,y; x_r, y_r)$$

여기서 $H(t,x,y) = \eta(x,y) \frac{\partial v}{\partial t}$는 감쇠 항입니다. 핵심은 **adjoint 파동방정식이 forward 파동방정식의 진정한 adjoint(transpose)임**을 수학적으로 보장하는 것입니다.[1]

<details>

### 목적 함수 (우리가 최소화하려는 값) 
먼저, 모델로 계산한 값 $(\(u\))$ 과 실제 수신기에서 측정한 값 $(\(d_{obs}\))$ 의 차이를 줄여야 합니다.  
이를 $\(L_{2}\)$ 노름(norm)으로 정의합니다.

$$\(J=\frac{1}{2}\int_{0}^{T}\int_{\Omega }[u(x,t)-d_{obs}(x,t)]^{2}\cdot \delta (x-x_{r})\,dxdt\)$$

- 의미: 모든 시간 $(\(T\))$ 과 모든 공간 $(\(\Omega \))$ 에 대해, 수신기 위치 $(\(x_{r}\))$ 에서 발생한 오차의 제곱합입니다.

### 제약 조건 (물리 법칙) 
하지만 $\(u\)$ 는 마음대로 변하는 게 아니라, 반드시 파동 방정식을 만족해야 합니다.

$$\(Lu=m\frac{\partial ^{2}u}{\partial t^{2}}-\nabla ^{2}u-s=0\)$$

### 라그랑주 함수 구성 (수학적 결합) 
오차 $(\(J\))$ 를 줄이면서 물리 법칙 $(\(Lu=0\))$ 도 지키기 위해, 라그랑주 승수 $\(v\)$ 를 도입하여 두 식을 합칩니다.

$$\(\mathcal{L}(u,v,m)=J(u)-\int_{0}^{T}\int_{\Omega }v(x,t)\left(m\frac{\partial^{2}u}{\partial t^{2}}-\nabla ^{2}u-s\right)dxdt\)$$

여기서 $\(v(x,t)\)$ 가 바로 우리가 구하려는 수반 필드(Adjoint Field)입니다.

여기서 최적의 상태에서는 라그랑주 함수의 변분(Variation)이 0이 되어야 합니다. 즉, $\(\delta \mathcal{L}=0\)$ 입니다. 

##### 라그랑주 승수법
###### 일반적인 원리 (기초 수학) 
목적 함수 $\(f(x,y)\)$ 를 최소화하고 싶은데, $\(g(x,y)=0\)$ 이라는 제약 조건이 붙어 있는 상황을 가정합니다. 
- 라그랑주 함수 구성: $\(\mathcal{L}(x,y,\lambda )=f(x,y)+\lambda \cdot g(x,y)\)$ (여기서 $\(\lambda \$ )는 라그랑주 승수입니다.)

###### 해법: 
$\(\mathcal{L}\)$ 을 각 변수로 편미분하여 $\(0\)$ 이 되는 지점을 찾습니다.
- $\(\frac{\partial \mathcal{L}}{\partial x}=0\implies \frac{\partial f}{\partial x}+\lambda \frac{\partial g}{\partial x}=0\)$
- $\(\frac{\partial \mathcal{L}}{\partial y}=0\implies \frac{\partial f}{\partial y}+\lambda \frac{\partial g}{\partial y}=0\)$
- $\(\frac{\partial \mathcal{L}}{\partial \lambda }=0\implies g(x,y)=0\)$ (원래의 제약 조건 복원)

파동 방정식 문제에 이를 대입하면, 변수가 단순한 숫자 $(\(x,y\))$ 가 아니라 함수 $(\(u,v\))$ 와 모델 파라미터 $(\(m\))$ 로 확장됩니다.  
라그랑주 함수 구성 오차 함수 $\(J(u)\)$ 에 제약 조건(파동 방정식 $\(Lu=0\)$ )을 라그랑주 승수 $\(v(x,t)\)$ 와 곱하여 결합합니다.

### 핵심 단계: 변분(Variation) 수행 
여기서 우리가 궁금한 것은 **"매질 $(\(m\))$ 이 $\(\delta m\)$ 만큼 변하고, 그 결과로 파동 $(\(u\))$ 이 $\(\delta u\)$ 만큼 변했을 때, $\(\mathcal{L}\)$ 은 얼마나 변하는가?"**입니다. 

(수반 변수 $\(v\)$ 는 고정된 상태에서 $\(u\)$ 와 $\(m\)$ 의 변화만 고려합니다.) 

$$\(\delta \mathcal{L}=\delta J(u)-\delta \left[\int _{0}^{T}\int _{\Omega }v\left(m\frac{\partial ^{2}u}{\partial t^{2}}-\nabla ^{2}u-s\right)dxdt\right]\)$$

#### A , B 도출과정

$\(J\)$ 는 $\(u\)$ 에 대한 함수이므로, 체인 룰(Chain Rule)에 의해 다음과 같이 변합니다.

$$\(\delta J=\frac{\partial J}{\partial u}\delta u\)$$

이것이 항 A가 됩니다.

적분 기호 안의 식 $\(f=v(m\frac{\partial^{2}u}{\partial t^{2}}-\nabla ^{2}u-s)\)$ 에 대해 변분을 수행합니다. 

여기서 $\(v\)$ 와 $\(s\)$ 는 고정값(상수 취급)이므로 변화량 $\(\delta v=0,\delta s=0\)$ 입니다. 

- 곱의 미분 법칙을 적용하면: $\(\delta [v(m\frac{\partial ^{2}u}{\partial t^{2}}-\nabla ^{2}u-s)]=v\cdot \delta \left(m\frac{\partial^{2}u}{\partial t^{2}}-\nabla^{2}u-s\right)\)$

- 괄호 안을 하나씩 미분하면: $\(m\frac{\partial^{2}u}{\partial t^{2}}\)$

- 부분: $\(m\)$ 과 $\(u\)$ 가 둘 다 변하므로 곱의 미분을 적용합니다.

$$\(\delta (m\frac{\partial^{2}u}{\partial t^{2}})=(\delta m)\frac{\partial^{2}u}{\partial t^{2}}+m\delta (\frac{\partial ^{2}u}{\partial t^{2}})\)$$

이때 미분 연산과 변분 연산은 순서를 바꿀 수 있으므로 $\(\delta (\frac{\partial^{2}u}{\partial t^{2}})=\frac{\partial^{2}\delta u}{\partial t^{2}}\)$ 가 됩니다.

$$\(\Rightarrow (\delta m)\frac{\partial ^{2}u}{\partial t^{2}}+m\frac{\partial ^{2}\delta u}{\partial t^{2}}\)$$

- $\(\nabla^{2}u\)$ 부분: $\(\delta (\nabla^{2}u)=\nabla^{2}\delta u\)$
- $\(s\)$ 부분: 소스(진원)는 매질이나 파동의 변화와 무관하므로 0입니다. $\(\delta s=0\)$

위에서 구한 조각들을 적분 기호 안에 다시 넣습니다. 

$$\(\delta \mathcal{L}=\frac{\partial J}{\partial u}\delta u-\int_{0}^{T}\int_{\Omega }v\left(\underbrace{\delta m\frac{\partial ^{2}u}{\partial t^{2}}}\_{\text{매질 변화 영향}}+\underbrace{m\frac{\partial^{2}\delta u}{\partial t^{2}}-\nabla^{2}\delta u}_{\text{파동\ 변화\ 영향}}\right)dxdt\)$$

이것이 항 B의 형태입니다. 

이렇게 전개한 이유는 최적화 조건인 ** $\(\delta \mathcal{L}=0\)$ **을 만족시키기 위해서입니다.

매질 $\(m\)$ 이 아주 조금 변할 때 $(\(\delta m\))$ , 오차 $\(\mathcal{L}\)$ 이 어떻게 변하는지 계산합니다 $(\(\delta \mathcal{L}=0\))$ .

$$\(\delta \mathcal{L}=\underbrace{\frac{\partial J}{\partial u}\delta u}\_{\text{A}}-\underbrace{\int v\left(\delta m\frac{\partial ^{2}u}{\partial t^{2}}+m\frac{\partial^{2}\delta u}{\partial t^{2}}-\nabla^{2}\delta u\right)dxdt}_{\text{B}}=0\)$$

우리는 매질의 변화 $(\(\delta m\))$ 에 따른 전체 시스템의 변화를 알고 싶지만, ** $\(\delta u\)$ (파동의 변화)**는 $\(m\)$ 이 바뀔 때마다 매번 파동 방정식을 새로 풀어야 알 수 있는 아주 복잡한 값입니다.  
문제는 $\(\delta u\)$ (파동의 변화)를 계산하기가 너무 어렵다는 것입니다. 그래서 수학적 기교인 부분 적분을 사용하여 미분 기호를 $\(\delta u\)$ 에서 $\(v\)$로 옮겨버립니다.

### 부분 적분 (The Magic of Adjoint) 
#### 시간 항 미분 이동:

$\(\int_{0}^{T}v\left(m\frac{\partial^{2}\delta u}{\partial t^{2}}\right)dt\xrightarrow{\text{ Partial Integral Twice }}\int_{0}^{T}\left(m\frac{\partial^{2}v}{\partial t^{2}}\right)\delta u\,dt+[\text{경계항}]\)$

이때 $\(v(T)=0,v^{\prime }(T)=0\)$ 이라는 최종 조건(Final Condition)을 주어 경계항을 없앱니다. (그래서 수반 방정식은 시간을 거꾸로 풉니다.) 

요약 (2번 부분적분) :

$$\(\int_{0}^{T}v(m\frac{\partial^{2}\delta u}{\partial t^{2}})dt=[vm\frac{\partial \delta u}{\partial t}-\frac{\partial v}{\partial t}m\delta u]\_{0}^{T}+\int_{0}^{T}\frac{\partial^{2}v}{\partial t^{2}}m\delta udt\)$$

#### 공간 항 미분 이동 (라플라시안):

$\(\int _{\Omega }v(\nabla ^{2}\delta u)dx\xrightarrow{\text{그린의\ 제2정리}}\int _{\Omega }(\nabla ^{2}v)\delta u\,dx+[\text{경계항}]\)$

요약 :
공간 항 (라플라시안): 그린의 제2정리에 의해 $\(\int_{\Omega }v\Delta \delta ud\Omega =\int_{\Omega }\delta u\Delta vd\Omega \)$ (경계 조건 가정)

이것이 바로 자기 수반(Self-adjoint) 성질입니다. $\(\nabla^{2}\)$ 은 위치를 바꿔도 형태가 변하지 않습니다.

##### 수반 연산자 
내적(Inner Product)이 정의된 힐베르트 공간 $\(H\)$ 에서 선형 연산자 $\(L\)$ 에 대한 수반 연산자 $\(L^{\*}\)$ 은 다음 관계를 만족하는 유일한 연산자로 정의됩니다. 

$$\(\langle Lu,v\rangle =\langle u,L^{*}v\rangle \)$$

여기서 $\(\langle f,g\rangle =\int_{\Omega }f(x)g(x)\,dx\)$ 는 $\(L_{2}\)$ 내적입니다.

##### 2차 미분 연산자 (Laplacian) 연산자 
$\(L=\nabla ^{2}\)$ (라플라시안)의 경우: 그린의 제2정리에 의해 경계 조건이 적절하다면 다음이 성립합니다.

$$\(\langle \nabla ^{2}u,v\rangle =\langle u,\nabla ^{2}v\rangle \)$$

이처럼 $\(L=L^{*}\)$ 인 경우를 자기 수반(Self-adjoint) 연산자라고 하며, 이는 대칭 행렬과 유사한 성질을 가집니다. 

- 가역성 및 해의 존재성: Fredholm Alternative 정리에 따라, $\(Lu=f\)$ 가 해를 가질 조건은 $\(f\)$ 가 $\(L^{*}\)$ 의 영공간(Null space)에 수직해야 함을 알 수 있습니다.
- 최적 제어 및 민감도 분석: 특정 출력값에 대한 입력 파라미터의 영향을 계산할 때, 원래 방정식을 수천 번 푸는 대신 수반 방정식(Adjoint Equation)을 단 한 번 풀어 민감도를 구할 수 있습니다.
- 데이터 동화 (Data Assimilation): 기상 예보나 수치 모델링에서 관측 데이터를 모델에 역으로 반영하여 초기 상태를 추정할 때 핵심적으로 사용됩니다. 

이제 $\(\delta u\)$ 가 포함된 항들만 모아서 정리합니다.

$$\(\int_{0}^{T}\int_{\Omega }\delta u\cdot \left[\underbrace{(u-d_{obs})\delta (x-x_{r})}\_{\text{오차\ 정보}}-\underbrace{\left(m\frac{\partial ^{2}v}{\partial t^{2}}-\nabla ^{2}v\right)}_{\text{수반\ 연산}}\right]dxdt=0\)$$

$\(\delta u\)$ 가 어떤 값이든 이 식이 0이 되려면, 괄호 안이 0이어야 합니다.

따라서,  
$$\(m\frac{\partial ^{2}v}{\partial t^{2}}-\nabla ^{2}v=(u-d_{obs})\delta (x-x_{r})\)$$

여기서 우변의 $\(u-d_{obs}\)$ 는 데이터 잔차 $\(\delta d\)$ 이며, 수신기 위치 $\(x_{r}\)$ 에서 소스로 작용하게 됩니다. $(\(H\)$ 항은 문제 정의에 따라 추가적인 외력이나 감쇠항을 포함할 수 있습니다.)

- 변화: $\((u-d_{obs})\delta (x-x_{r})\rightarrow \delta d(t,x,y;x_{r},y_{r})\)$
- 의미: 수신기 위치 $(\(x_{r},y_{r}\))$ 에서 계산된 파동 $(\(u\))$ 과 실제 데이터 $(\(d_{obs}\))$ 의 차이를 데이터 잔차(Data Residual)라고 부르며, 이를 통칭하여 $\(\delta d\)$ 라는 기호로 나타낸 것입니다. 수반 방정식에서 이 항은 수신기 위치에서 역으로 퍼져나가는 수반 소스(Adjoint Source) 역할을 합니다.

실제 매질에서는 단순한 파동 전파 외에 감쇠(Attenuation), 점성(Viscosity), 혹은 경계 조건 등의 추가적인 물리 현상이 발생할 수 있습니다.  
$\(H\)$는 단순 모델에서 고려하지 않았던 추가적인 물리적 제약이나 선형 연산자를 일반화하여 포함시킨 것입니다.

참고 : 수학적으로 라플라시안(Laplacian) 연산자는 $\(\nabla ^{2}\)$ 또는 $\(\Delta \)$ 로 혼용하여 표기합니다. 
- 변화: $\(\nabla ^{2}v\rightarrow \Delta v\)$
  
</details>

### 2.3 이산화된 Adjoint Stencil

최적-이산화(optimize-then-discretize) 접근법을 따르면:[1]

$$v[\text{time}-\text{dt}] = 2v[\text{time}] - v[\text{time}+\text{dt}] + \frac{\text{dt}^2}{m} \Delta v[\text{time}]$$

**핵심 차이점**: Forward 전파는 시간 앞방향, adjoint 전파는 시간 역방향으로 진행되며, forward의 초기 시간 조건이 adjoint의 최종 시간 조건이 됩니다.[1]

### 2.4 FWI 그래디언트 계산 공식

Adjoint-state 방법에 의한 그래디언트는:[1]

$$\frac{\partial f}{\partial m} = \sum_{i=1}^{n_s} \sum_{t=1}^{n_t} \ddot{\lambda}_i(t) \odot u_i(t)$$

여기서 $u_i(t)$는 forward 파동장, $\ddot{\lambda}_i(t)$는 adjoint 파동장의 2차 시간 도함수, $\odot$는 점별 곱셈입니다.[1]

### 2.5 계산 효율적 그래디언트 업데이트

메모리 효율을 위해 역시간 루프 내에서 그래디언트를 직접 계산합니다:[1]

$$g = g - u[\text{time}] \odot \frac{v[\text{time}-\text{dt}] - 2v[\text{time}] + v[\text{time}+\text{dt}]}{\text{dt}^2}$$

이를 통해 adjoint 파동장 전체를 저장할 필요가 없습니다.[1]

## 3. 모델 구조

### 3.1 세 단계 그래디언트 계산 프로세스

**단계 1: Forward Modeling**
- 각 발진원 위치에서 파동방정식 풀이
- 수신기 위치에서 파동장 샘플링
- 순방향 파동장 저장 (체크포인팅 가능)

**단계 2: Data Residual Calculation**
- 예측 데이터: $\mathbf{d}^{\text{pred}}(m_0, q_i)$
- 데이터 잔차: $\delta \mathbf{d} = \mathbf{d}^{\text{pred}} - \mathbf{d}^{\text{obs}}$

**단계 3: Adjoint Modeling**
- 데이터 잔차를 adjoint 소스로 활용
- 역시간 방향으로 adjoint 파동방정식 풀이
- Forward/adjoint 파동장의 crosscorrelation으로 그래디언트 도출[1]

### 3.2 Devito 구현 구조

저자들은 Devito를 통해 다음의 핵심 구성요소를 구현했습니다:[1]

**Adjoint 파동장 TimeFunction 정의:**
```python
v = TimeFunction(name="v", grid=model.grid,
                 time_order=2, space_order=4, save=False)
```

**PDE 기호 설정:**
```python
pde = model.m * v.dt2 - v.laplace - model.damp * v.dt
stencil_v = Eq(v.backward, solve(pde, v.backward)[0])
```

**Residual 소스 주입:**
```python
residual = PointSource(name='residual', ntime=nt,
                       grid=model.grid, coordinates=rec_coords)
res_term = residual.inject(field=v.backward,
                           expr=residual * dt**2 / model.m)
```

**Backward 시간 축 연산자:**
```python
op_grad = Operator([stencil_v] + res_term + [grad_update],
                   time_axis=Backward)
```

## 4. 성능 향상 및 수치적 안정성

### 4.1 메모리 최적화

**체크포인팅(Checkpointing)**: Forward 파동장을 전체 저장하지 않고, 필요한 시간 스텝의 파동장만 저장합니다. 역시간 루프에서 필요하면 재계산합니다.[1]

**서브샘플링**: 시간 및 공간 도메인에서 선택적 저장으로 메모리 요구량을 크게 감소시킵니다.[1]

### 4.2 수치적 안정성 보장

**Optimize-then-Discretize 접근법**: 먼저 연속 도메인에서 adjoint 방정식을 유도한 후 이산화합니다. 이를 통해 이산화된 adjoint가 진정한 adjoint임을 보장합니다.[1]

**흡수 경계 조건**: 경계에서의 파동 반사를 제거하여 수치 오류를 최소화합니다.[1]

### 4.3 계산 효율성

역시간 루프 내에서 그래디언트를 직접 계산함으로써 추가 저장 공간 없이 그래디언트를 획득합니다. 2차 시간 도함수는 3개의 시간 스텝만 메모리에 유지하면 계산됩니다.[1]

## 5. 모델의 일반화 성능 향상 가능성

### 5.1 논문의 범위 내 논의

본 튜토리얼은 기본적인 adjoint 모델링에 집중하므로, 일반화 성능에 대한 직접적인 논의는 제한적입니다. 다만, 다음의 개선 경로를 암시합니다:[1]

**멀티스케일 반복(Multiscale Iteration)**: 저주파 성분부터 고주파로 점진적 역전하면 초기 모델 의존성을 감소시킵니다.[1]

**정규화 기법**: Part 3에서 다룰 최적화 알고리즘에 Tikhonov 정규화나 Total Variation 등을 추가하여 일반화 성능을 개선합니다.[1]

### 5.2 2020년 이후의 일반화 성능 개선 전략

#### 5.2.1 신경망 기반 재매개변수화 (2024-2025)

**Deep Reparameterization for FWI**: 신경망을 통해 속도 모델을 저차원 잠재 변수로 재매개변수화합니다. 이는:[2]
- 비선형성을 감소시킴
- 초기 모델 의존성을 완화
- 저주파 편향을 활용하여 수렴 가속화[2]

**실험 결과**: 얕은 CNN(2-3층)이 깊은 네트워크보다 우수한 성능을 보임. 단계적 사전학습-임베딩 방식이 직접 중첩보다 우수함.[2]

#### 5.2.2 Transfer Learning과 사전학습 (2021-2023)

**VelocityGAN**: 생성적 적대 신경망(GAN) 기반으로 다양한 지질 환경에 대한 일반화 능력을 강화합니다.[3]

**Physics-Guided CNN**: 사전학습된 CNN을 초기 속도 모델 구축에 사용하여 수렴 가속화 및 불확실성 감소.[4]

#### 5.2.3 Latent Representation Learning (2025)

**PINNs with Latent Space**: 오토인코더를 통해 속도 모델의 잠재 표현을 학습하고, FWI 업데이트를 잠재 공간에서 수행합니다. 이는:[5]
- 메시 불필요
- 대규모 문제에서 계산 효율성 대폭 향상
- 정확도 및 안정성 개선[5]

#### 5.2.4 Bidirectional Physics-Constrained Framework (2025)

**BP-FWI**: 신경망과 파동 전파 물리를 양방향으로 통합합니다.[4]
- 초기 모델의 부정확성에 강건
- U-Net 기반 동적 잔차 학습
- 저주파 데이터, 노이즈, 불균형 수신기 배치에 모두 강건[4]

## 6. 한계점

### 6.1 방법론적 한계

**국소 최소값 문제**: 비선형 최적화 문제의 본질적 한계입니다. 초기 모델이 진정한 모델에서 많이 벗어나면 잘못된 해로 수렴할 수 있습니다.[1]

**사이클 스킵핑(Cycle Skipping)**: 초기 모델이 부정확하면, 동일한 주기 내의 다른 이벤트를 매칭하여 수렴을 실패합니다.[1]

**선형화 가정**: 일부 접근법에서 사용되는 Born 근사는 약한 산란 가정을 포함합니다.[1]

### 6.2 계산적 한계

**메모리 요구량**: 3D 대규모 문제에서 forward 파동장 저장은 여전히 막대한 메모리를 요구합니다. 체크포인팅으로 완화되지만 재계산 비용이 발생합니다.[1]

**연산 시간**: 매 반복마다 forward 및 adjoint 파동방정식을 모두 풀어야 하므로 계산량이 많습니다.[1]

### 6.3 데이터 관련 한계

**노이즈 민감성**: 데이터 잔차가 노이즈를 포함하면 그래디언트도 노이즈에 오염됩니다.[1]

**불완전한 데이터**: 제한된 오프셋(offset) 또는 방위각(azimuth) 범위는 이미징 해상도를 제약합니다.[1]

**소스 파형 불확실성**: 정확한 소스 시그니처(wavelet)를 모르면 데이터 매칭이 부정확합니다.[1]

## 7. 2020년 이후 관련 최신 연구 비교 분석

### 7.1 향상된 최적화 기법 (2020-2022)

#### Inexact Newton Methods[6]
- **주요 개선**: 2차 도함수(Hessian) 정보 활용
- **기술**: Lanczos 기반 Hessian 근사 및 전처리
- **성과**: 강한 비선형성과 ill-posedness 문제에서 1차 기울기 방법보다 우수
- **구현**: Adjoint-state 방법으로 Hessian-vector 곱 효율적 계산

#### Extended FWI with Augmented Lagrangian[7]
- **핵심 아이디어**: 파동방정식 완화(relaxation)로 선형 체계 확장
- **장점**: 사이클 스킵핑 완화, 초기 모델 의존성 감소
- **구현**: 시간 및 주파수 도메인 모두 가능

### 7.2 물리 제약 딥러닝 (2021-2025)

#### Physics-Informed Neural Networks (PINNs)[8][9]
- **개념**: 파동방정식을 신경망의 손실함수에 직접 포함
- **장점**: 메시 생성 불필요, 임의의 기하학 적용 가능
- **개선**: GaborPINN으로 수렴 속도 **2-3배 향상**[8]
- **응용**: 다양한 주파수 성분을 Gabor 기저 함수로 표현

#### FWIGAN: Physics-Informed GAN[10]
- **핵심**: 레이블 데이터셋 없이 분포 측면의 물리적 일관성 확보
- **장점**: 자동 조정 불필요, 사용자 개입 최소화
- **성과**: 국소 최소값 문제 개선

#### Adjoint-Driven Deep Learning FWI[3]
- **전략**: FCN과 adjoint 그래디언트 결합
- **혁신**: 그래디언트 정보를 공통-소스 도메인으로 변환하여 정보 보존
- **효과**: 초기 모델 부정확성에 더욱 강건

### 7.3 사이클 스킵핑 극복 (2022-2025)

#### Dynamic Time Warping 기반 중간 신호 접근[11]
- **방법**: DTW-MISA (Dynamic Time Warping Multiscale Intermediate Signals)
- **혁신**: 첫 도달파 이상의 정보 활용
- **효과**: 기존 방법보다 더 강건한 매칭

#### 딥러닝 기반 시간 정렬[12][13]
- **기술**: 신경망으로 시간 시프트 학습
- **장점**: 효율적이고 정확한 파형 정렬
- **최신**: 2025년 EAGE 학술대회에서 발표

#### 자기 지도 학습 (Self-Supervised)[14]
- **프레임워크**: 두 단계 접근법
  1. 데이터 매칭: 신경망으로 관측-시뮬레이션 데이터 정렬
  2. FWI: 정렬된 데이터로 자기 지도 학습
- **성과**: 레이블 데이터 없이 초기 모델 좋지 않아도 수렴

### 7.4 일반화 성능 강화 (2022-2025)

#### Deep Reparameterization[15][2]
- **방법**: 신경망으로 속도 모델을 저차원 벡터로 매핑
- **성과**:
  - 비선형성 감소 (저주파 편향 활용)
  - 초기 모델 의존성 감소
  - 얕은 CNN이 깊은 네트워크보다 우수[2]
  - Pretraining vs. Denormalization 비교: 후자가 음의 전이 효과 제거[15]

#### Transfer Learning 전략[3]
- **기법**: VelocityGAN으로 다양한 데이터로 사전학습
- **효과**: 새로운 지질 환경에 대한 일반화 능력 향상

#### Latent Representation Learning[5]
- **혁신**: 오토인코더 잠재 공간에서 FWI 수행
- **이점**:
  - 메시 불필요
  - 대규모 3D 문제에서 효율성 **대폭 향상**
  - 속도 모델의 저주파 편향 완화[5]

### 7.5 최신 물리 제약 프레임워크 (2024-2025)

#### Bidirectional Physics-Constrained FWI (BP-FWI)[4]
- **개념**: 신경망과 파동 물리의 양방향 상호작용
- **구조**: 
  - 초기 속도 모델 기반 사전학습
  - 동적 그래디언트 조정으로 동시 최적화
- **강점**:
  - 초기 모델 부정확성에 강건
  - 저주파 데이터 부재 대응
  - 노이즈 및 불균형 수신기 배치 대응[4]

#### Physics-Informed RNN (PIRNN)[16][17]
- **기술**: RNN으로 파동방정식 구현
- **향상**: 첫 도달파 시간 제약 추가로 초기 모델 오류 극복[17]
- **응용**: 수직 탄성파 탐사(VSP)에서 우수한 성능

#### Fully Automatic Differentiation (FAD)[18]
- **통합**: U-Net + RNN + 자동 미분
- **개념**: 깊은 이미지 사전(deep image prior) 활용
- **효과**: 암시적 정규화로 ill-posed 문제 안정화

### 7.6 Hessian 기반 개선 (2025)

#### Hessian 연산자 전처리[19]
- **기법**: Hessian의 대각 요소로 adjoint 그래디언트 스케일링
- **효과**:
  - 공간 그래디언트 진폭의 불균일성 제거
  - 깊은 영역 이미징 정확도 향상
  - 약한 반사 영역 해상도 개선

### 7.7 계산 효율성 (2020-2025)

#### Lossy Checkpoint Compression[20]
- **기술**: 오차 제어 손실 압축 + 체크포인팅
- **성과**:
  - 메모리 압축율 **최대 100배**
  - 실행 시간 및 데이터 이동 감소
  - 최종 역전 품질에 미미한 영향

#### Deep Compressed Learning (DCL)[21]
- **방법**: 신경망으로 최적 발진원(shot) 자동 선택
- **장점**:
  - 대규모 데이터 입력 차원 축소
  - 계산 비용 현저히 감소
  - 저 샘플링율(10%)에서 PSNR **3dB 향상**

## 8. 향후 연구에 미치는 영향 및 고려사항

### 8.1 이론적 기여

본 논문은 2024년까지도 계속 인용되는 중요한 기초 작업입니다. 특히 **2024년에 Monteiller et al.**이 adjoint-state 방법의 완전한 수학적 유도(탄성 및 점성탄성 경우 포함)를 제공했으나, 본 논문의 acoustic 경우의 명확한 설명이 여전히 학습 자료로 활용되고 있습니다.[22]

### 8.2 방법론적 진화

#### 단기 (1-2년):
1. **Physics-Informed 딥러닝의 주류화**: 본 논문의 adjoint 개념이 신경망 훈련에 직접 통합
2. **Uncertainty Quantification**: Bayesian 접근으로 역전 결과의 신뢰도 평가
3. **멀티 물리 통합**: 탄성파 + 전자기파 + 중력 동시 역전

#### 중기 (2-5년):
1. **전역 최적화 보증**: 국소 최소값 극복을 위한 이론적 기초 확립
2. **실시간 FWI**: 스트리밍 데이터를 이용한 온라인 역전
3. **극단 규모 문제**: 엑사스케일 컴퓨팅에서의 adjoint 최적화

### 8.3 실천적 고려사항

#### 초기 모델 의존성 극복
- **방안**: Transfer learning 기반 사전학습 (VelocityGAN, pretrained PINNs)
- **효과**: 초기 모델이 나쁜 경우에도 수렴 가능
- **구현 난제**: 충분한 학습 데이터 확보 및 다양성 확보

#### 사이클 스킵핑 해결
- **다각적 전략**:
  1. 시간 정렬 기반 (DTW, 신경망)
  2. 자기 지도 학습 (pretrained 신경망 + FWI)
  3. 중간 신호 방식 (매개변수 보간)
- **현재 상태**: 부분적으로 해결되었으나, 극단적 부정확성에서는 여전히 난제

#### 계산 비용 최적화
- **이미 입증된 기법**:
  1. 로시 압축 체크포인팅 (메모리 100배 감소)
  2. Deep compressed learning (계산 10배 감소)
  3. Latent space FWI (3D 대규모 문제 가능화)
- **향후 방향**: 이 기법들의 조합으로 산업 규모 3D 문제 해결

#### 현장 데이터 적용
- **현재 주요 과제**:
  1. 높은 노이즈 강건성
  2. 불완전한 데이터 처리 (제한된 오프셋, 불규칙 수신기)
  3. 소스 파형 불확실성 처리
  4. 다중 출처 데이터 통합

- **해결책 방향**:
  1. Robust statistics (L1 norm, Huber loss)
  2. 정보 이론 기반 정규화
  3. 소스 추정 자동화 (joint inversion)
  4. Semi-supervised learning으로 레이블 없는 데이터 활용

### 8.4 학제 간 통합 방향

#### 기계학습의 최신 기법 도입
1. **강화학습 (Reinforcement Learning)**: DeepWaveRL은 RL로 미분 불가능한 forward 연산자 처리[23]
2. **메타 러닝**: 다양한 지질 환경에서의 빠른 적응
3. **자기지도 학습 (Self-Supervised)**: 레이블 없는 대규모 데이터 활용

#### 신경과학 영감 아키텍처
- Vision Transformer 기반 구조
- Attention 메커니즘으로 중요 파장 자동 선택
- Capsule Networks로 계층 경계 명확히

#### 통계학 기반 접근
- **Bayesian FWI**: 사후 분포 샘플링으로 불확실성 정량화
- **적응형 MCMC**: 자동 미분을 이용한 효율적 샘플링[24]
- **이상치 탐지**: Robust inversion을 위한 이상 데이터 자동 제거

### 8.5 산업 응용 시 중요 고려사항

#### 1. 검증 프로토콜 수립
- Synthetic data로 검증된 방법 → 현장 적용
- Blind test를 통한 객관적 성능 평가
- 다양한 지질 환경에서의 일반화 검증

#### 2. 매개변수 자동 결정
- 정규화 강도, 학습률, 네트워크 깊이 등 자동 결정
- AutoML 기법의 FWI 적용
- Bayesian optimization으로 하이퍼매개변수 최적화

#### 3. 해석 가능성 (Explainability)
- 블랙박스 신경망의 의사결정 과정 투명화
- Gradient-based saliency map으로 중요 영역 파악
- LIME, SHAP 등 설명 기법 적용

#### 4. 배포 및 유지보수
- Edge device에서의 경량 모델 (학습 완료 후 추론만 필요)
- 온라인 학습으로 새로운 데이터 점진적 적응
- Model drift 모니터링 및 재훈련 전략

## 9. 주요 수식 정리

### 목적함수
$$f(m) = \frac{1}{2} \sum_{i=1}^{n_s} \left\| \mathbf{d}_i^{\text{pred}}(m, \mathbf{q}_i) - \mathbf{d}_i^{\text{obs}} \right\|_2^2$$

### Forward 파동방정식
$$m(x,y) \frac{\partial^2 u}{\partial t^2} = \Delta u + f(x,y,t)$$

### Continuous Adjoint 파동방정식
$$m(x,y) \frac{\partial^2 v}{\partial t^2} - \Delta v - \eta(x,y) \frac{\partial v}{\partial t} = \delta d(t,x,y; x_r, y_r)$$

### 이산화된 Forward Stencil
$$u[\text{time}+\text{dt}] = 2u[\text{time}] - u[\text{time}-\text{dt}] + \frac{\text{dt}^2}{m} \Delta u[\text{time}]$$

### 이산화된 Adjoint Stencil
$$v[\text{time}-\text{dt}] = 2v[\text{time}] - v[\text{time}+\text{dt}] + \frac{\text{dt}^2}{m} \Delta v[\text{time}]$$

### 그래디언트 계산 (일반형)
$$\frac{\partial f}{\partial m} = \sum_{i=1}^{n_s} \sum_{t=1}^{n_t} \ddot{\lambda}_i(t) \odot u_i(t)$$

### 역시간 루프 내 그래디언트 업데이트
$$g = g - u[\text{time}] \odot \frac{v[\text{time}-\text{dt}] - 2v[\text{time}] + v[\text{time}+\text{dt}]}{\text{dt}^2}$$

## 결론

2018년 발표된 "Full-waveform inversion, Part 2: Adjoint Modeling"은 FWI의 기초가 되는 adjoint-state 방법을 명확하고 실천적으로 제시한 중요한 튜토리얼입니다. 본 논문이 제공한 이산화된 adjoint 파동방정식과 그래디언트 계산 공식은 이후 2020년 이후의 모든 advanced FWI 연구의 수학적 기초가 되었습니다.

특히, **2020-2025년 사이의 연구 진화**를 보면:

1. **기초 강화 (2020-2022)**: Inexact Newton, 확장 FWI 등 본 논문의 gradient 계산을 활용하되 더 강력한 2차 정보 도입

2. **신경망 통합 (2021-2024)**: PINNs, GAN, 심층 학습이 본 논문의 adjoint 개념을 차용하여 physics-informed 신경망 설계

3. **일반화 성능 획기적 개선 (2024-2025)**: Deep reparameterization, latent space learning, bidirectional physics constraints가 본 논문의 gradient 기반 역전을 신경망과 결합하여 초기 모델 독립성 달성

4. **산업 배포 (2024-2025)**: 로시 압축, shot selection, RNN 기반 자동 미분 등이 대규모 3D 문제를 실현 가능하게 함

본 논문의 진정한 가치는 **명확한 수학적 기초**와 **재현 가능한 구현**을 제공함으로써, 이후 연구자들이 이를 바탕으로 창의적인 개선을 할 수 있는 발판을 마련했다는 점입니다. 향후 연구에서는 이 기초 위에서 (1) 더욱 강력한 최적화 알고리즘, (2) 신경망과의 seamless 통합, (3) 불확실성 정량화, (4) 현장 적용 성숙도 향상 등이 핵심 과제가 될 것으로 예상됩니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a062a1ca-e1ad-4b9a-9ecb-c45ca65e3fca/tle37010069.1.pdf)
[2](https://arxiv.org/html/2504.17375v1)
[3](https://www.semanticscholar.org/paper/Data-Driven-Seismic-Waveform-Inversion:-A-Study-on-Zhang-Lin/40b021ec960cdcd6c7b5af9e9d20fae54404c285)
[4](https://academic.oup.com/gji/article/244/2/ggaf466/8327597)
[5](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024EA004107)
[6](https://iopscience.iop.org/article/10.1088/1361-6420/abb8ea)
[7](https://library.seg.org/doi/10.1190/geo2021-0186.1)
[8](https://ieeexplore.ieee.org/document/10310254/)
[9](https://arxiv.org/pdf/2108.12035.pdf)
[10](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2022JB025493)
[11](https://sbgf.org.br/mysbgf/eventos/expanded_abstracts/18th_CISBGf/02522a2b2726fb0a03bb19f2d8d9524dExpanded_Abstract_18thCISBGf.pdf)
[12](https://arxiv.org/abs/2511.08134)
[13](https://www.shearwatergeo.com/technical-publications/deep-learning-based-time-shift-estimation-for-full-waveform-inversion)
[14](https://imageevent.aapg.org/portals/26/abstracts/2025/4306559.pdf)
[15](https://arxiv.org/html/2506.05484)
[16](https://ieeexplore.ieee.org/document/10649606/)
[17](https://www.mdpi.com/2076-3417/15/10/5757)
[18](https://geophysical-press.com/journal/JSE/articles/452)
[19](https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2025.1526073/full)
[20](https://gmd.copernicus.org/articles/15/3815/2022/gmd-15-3815-2022-discussion.html)
[21](https://arxiv.org/html/2601.01268v1)
[22](https://academic.oup.com/gji/advance-article/doi/10.1093/gji/ggae421/7907869)
[23](https://openreview.net/pdf/4bc8860f22712ad1a37a68d49ae53815835ada99.pdf)
[24](https://arxiv.org/html/2511.02737v1)
[25](http://www.ijgeophysics.ir/article_104782.html)
[26](https://meetingorganizer.copernicus.org/EGU2020/EGU2020-9096.html)
[27](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019JB019129)
[28](https://engrxiv.org/index.php/engrxiv/preprint/view/1324)
[29](https://academic.oup.com/gji/article/223/2/1007/5874260)
[30](https://link.springer.com/10.1007/s00024-020-02593-y)
[31](https://meetingorganizer.copernicus.org/EGU2020/EGU2020-13470.html)
[32](https://arxiv.org/pdf/2212.10141.pdf)
[33](https://arxiv.org/pdf/1811.07875.pdf)
[34](https://academic.oup.com/gji/advance-article-pdf/doi/10.1093/gji/ggac297/45280653/ggac297.pdf)
[35](https://onlinelibrary.wiley.com/doi/10.1111/1365-2478.13437)
[36](http://arxiv.org/pdf/2211.06300.pdf)
[37](https://arxiv.org/pdf/2501.08210.pdf)
[38](https://arxiv.org/html/2503.00658)
[39](https://arxiv.org/html/2407.08181v1)
[40](https://www.academia.edu/58479876/Full_waveform_inversion_Part_2_Adjoint_modeling)
[41](https://academic.oup.com/gji/issue/244/1)
[42](https://academic.oup.com/gji/article/240/2/942/7907869)
[43](https://www.sciencedirect.com/science/article/pii/S0045782523004024)
[44](https://geophysical-press.com/journal/JSE/articles/current_issue)
[45](https://www.geophysik.uni-muenchen.de/~igel/adj_short_fichtner/Wellenforminversion.pdf)
[46](https://www.earthdoc.org/content/papers/10.3997/2214-4609.202510380)
[47](https://eage.eventsair.com/fourth-conference-on-seismic-inversion/)
[48](https://www.uib.no/sites/w3.uib.no/files/attachments/csd_annual_report_2022.pdf)
[49](https://pasta.place/Physik/Master/Full-waveform_Inversion/Slides/WS_21-22/FWI_Lecture_3.pdf)
[50](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6044448)
[51](https://sponse2026.com)
[52](http://arxiv.org/list/physics/2023-10?skip=680&show=2000)
[53](https://arxiv.org/pdf/2104.02750.pdf)
[54](https://arxiv.org/html/2511.08134v1)
[55](https://arxiv.org/html/2507.10804v1)
[56](https://arxiv.org/html/2503.15013v4)
[57](https://arxiv.org/pdf/2401.04393.pdf)
[58](http://arxiv.org/pdf/2106.11892v2.pdf)
[59](https://arxiv.org/abs/2104.02750)
[60](https://arxiv.org/html/2509.14919v1)
[61](https://arxiv.org/html/2312.10568v1)
[62](https://www.semanticscholar.org/paper/a3715b919940fb8b99494ad7ef022040f74c2fec)
[63](https://arxiv.org/abs/2310.08109)
[64](https://www.earthdoc.org/content/papers/10.3997/2214-4609.202410650)
[65](https://ieeexplore.ieee.org/document/10533248/)
[66](https://ieeexplore.ieee.org/document/11121902/)
[67](https://arxiv.org/html/2404.00545v1)
[68](https://www.degruyterbrill.com/document/doi/10.1515/nanoph-2024-0504/html)
[69](https://arxiv.org/pdf/2304.02811.pdf)
[70](https://arxiv.org/pdf/2304.12541.pdf)
[71](http://arxiv.org/pdf/2411.10064.pdf)
[72](https://arxiv.org/pdf/2312.15301.pdf)
[73](https://academic.oup.com/gji/article/167/2/495/559970)
[74](https://www.viridiengroup.com/sites/default/files/2022-05/Mitigating_cycle_skipping_in_full-waveform_inversion_using_partial_matching_filters_Final_Published.pdf)
[75](http://www.theisticscience.com/papers/tree/Adjoint%20solutions/Plessix-167-2-495.pdf)
[76](https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2023.1134871/full)
[77](https://arxiv.org/abs/2303.03260)
[78](https://www.sciencedirect.com/science/article/abs/pii/S009364132500059X)
[79](https://www.arxiv.org/abs/2509.14919)
[80](https://www.semanticscholar.org/topic/Adjoint-state-method/849815)
[81](https://arxiv.org/abs/2509.14919v1)
[82](http://arxiv.org/abs/2303.03260)
[83](https://arxiv.org/html/2512.13172v1)
[84](https://arxiv.org/abs/2405.17696)
[85](https://www.arxiv.org/abs/2408.15060)
[86](https://arxiv.org/html/2509.08967v1)
[87](https://www.arxiv.org/abs/2412.00031)
[88](https://arxiv.org/html/2503.00658v1)
[89](https://www.arxiv.org/abs/2509.08967)
[90](https://arxiv.org/abs/2501.13532)
