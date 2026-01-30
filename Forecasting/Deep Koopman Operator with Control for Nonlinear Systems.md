# Deep Koopman Operator with Control for Nonlinear Systems

### 제1부: 논문 핵심 요약

#### 1.1 기본 정보 및 핵심 주장

본 논문 "Deep Koopman Operator with Control for Nonlinear Systems"(Shi & Meng, 2022)는 IEEE Robotics and Automation Letters에 게재된 연구로, 비선형 동적 시스템의 예측과 제어를 위한 새로운 딥 러닝 프레임워크를 제시합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/849aa21f-eeac-4a10-ad33-9ad6bc827c9c/2202.08004v2.pdf)

**핵심 주장**: 기존 Koopman 기반 제어 방법들이 제어 입력의 비선형성을 적절히 처리하지 못함으로 인해 예측 성능과 제어 성능이 크게 저하되는 문제를 해결하기 위해, embedding function과 Koopman Operator를 end-to-end 방식으로 동시 학습하며, 상태-종속 비선형 제어 항을 인코딩하는 보조 제어 네트워크를 도입했습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/849aa21f-eeac-4a10-ad33-9ad6bc827c9c/2202.08004v2.pdf)

**주요 기여**:
- End-to-end 딥러닝 프레임워크로 embedding function과 Koopman Operator 동시 학습
- 비선형 제어 입력의 상태-종속성을 모델링하는 보조 제어 네트워크 설계
- 원래 상태 공간의 비용 함수 일관성 유지로 제어 성능 향상
- 여러 비선형 동적 시스템에서 기존 방법 대비 수 배 우수한 예측 오차 감소 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/849aa21f-eeac-4a10-ad33-9ad6bc827c9c/2202.08004v2.pdf)

#### 1.2 해결하는 문제와 그 배경

본 논문이 해결하는 문제는 세 가지 차원에서 이해할 수 있습니다.

**문제 1: Embedding Function 선택의 어려움**

Koopman 이론은 비선형 시스템을 고차원 embedding 공간에서 선형으로 표현할 수 있다는 아이디어를 제공합니다. 그러나 이 embedding function을 찾는 것이 근본적 난제입니다. 기존 접근법은 두 가지 방향으로 나뉩니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/849aa21f-eeac-4a10-ad33-9ad6bc827c9c/2202.08004v2.pdf)

- **RBF 기반**: Radial Basis Function을 사용하되, RBF 중심점 선택이 근사 성능을 크게 좌우하며, 수작업 조정이 필수
- **미분 기반**: 시스템의 고차 미분을 lifting function으로 사용하지만, 미분 정보가 필요하거나 불가능한 경우 활용 불가

**문제 2: 제어 입력의 비선형성 미처리**

기존 Koopman 기반 제어 방법의 가장 심각한 제약은 선형 제어 입력만 가정한다는 것입니다. 구체적으로: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/849aa21f-eeac-4a10-ad33-9ad6bc827c9c/2202.08004v2.pdf)

기존 단순화:

$$\text{가정: } \phi(x, u) = \phi_s(x) + B u $$

이는 다음과 같이 진화합니다:

$$\phi_{k+1} = K\phi_k + B u_k \quad \text{(식 5)} $$

그러나 현실의 많은 비선형 시스템, 특히 로봇 시스템에서는 제어가 상태에 의존적으로 비선형입니다:

$$\text{실제: } \phi(x_{k+1}) = K\phi(x_k) + \mathcal{C}(x_k) u_k $$

기존 선형화 가정은 이러한 상태-종속 비선형성을 무시하므로, 예측 품질 및 제어 성능이 심각히 저하됩니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/849aa21f-eeac-4a10-ad33-9ad6bc827c9c/2202.08004v2.pdf)

**문제 3: Embedding Space의 비용 함수 왜곡**

제어 설계에서 비용 함수의 일관성은 매우 중요합니다. 기존 방법에서 원래 상태 공간의 이차 비용 함수 $x^T Q x + u^T R u$가 embedding space로 lifted될 때 왜곡되면, 최적 제어 설계가 원 공간에서 최적이 아닐 가능성이 높습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/849aa21f-eeac-4a10-ad33-9ad6bc827c9c/2202.08004v2.pdf)

### 제2부: 제안 방법론과 수학적 기초

#### 2.1 기본 Koopman 이론과 제어 문제 정식화

**Koopman Operator의 기본 정의**

이산시간 비선형 시스템에서:

$$x_{k+1} = f(x_k, u_k) $$

Koopman Operator $\mathcal{K}$는 관측 함수의 진화를 지배하는 무한차원 선형 연산자입니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/849aa21f-eeac-4a10-ad33-9ad6bc827c9c/2202.08004v2.pdf)

$$\mathcal{K}[\phi(x)] = \phi(f(x, u)) $$

제어가 있는 경우, embedding space에서의 진화는:

$$\phi(x_{k+1}) = K\phi(x_k) + \mathcal{B}(x_k, u_k) \quad \text{(식 3)} $$

여기서 $K$ 는 Koopman operator의 행렬 표현이고, $\mathcal{B}$ 는 제어 영향 항입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/849aa21f-eeac-4a10-ad33-9ad6bc827c9c/2202.08004v2.pdf)

#### 2.2 제안하는 Embedding Function 설계

**핵심 아이디어: 원래 상태 보존**

기존 방법의 문제를 극복하기 위해, 본 논문은 embedding function을 다음과 같이 분해합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/849aa21f-eeac-4a10-ad33-9ad6bc827c9c/2202.08004v2.pdf)

$$\phi(x, u) = [\phi_s(x)^T, \mathcal{C}_\theta(x, u)^T]^T \quad \text{(식 9)} $$

여기서:
- $\phi_s(x) = [x^T, h_\theta(x)^T]^T$: 원 상태를 보존하고 신경망 인코딩 추가
- $\mathcal{C}_\theta(x, u)$: 보조 제어 네트워크로 비선형 제어 항 인코딩
- $h_\theta(x) \in \mathbb{R}^d$: d차원 신경망 인코딩

**복원의 단순화**

원래 상태 공간으로의 복원:

$$x_k = \Psi \phi_k \quad \text{(식 10)} $$

회복 행렬의 단순 형태:

$$\Psi = [I, 0] \in \mathbb{R}^{n \times N} \quad \text{(식 11)} $$

이 설계로 원래 비용 함수가 embedding space에서 일관성을 유지합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/849aa21f-eeac-4a10-ad33-9ad6bc827c9c/2202.08004v2.pdf)

#### 2.3 세 가지 알고리즘 변형

본 논문은 제어 항 비선형성의 수준에 따라 세 가지 변형을 제시합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/849aa21f-eeac-4a10-ad33-9ad6bc827c9c/2202.08004v2.pdf)

**1) DKUC (Deep Koopman U with Control)**

가장 단순한 형태로, 제어를 분리된 선형 항으로 처리:

$$\phi(x_{k+1}) = A\phi(x_k) + B u_k $$

이는 기존의 선형 제어 가정에 대응됩니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/849aa21f-eeac-4a10-ad33-9ad6bc827c9c/2202.08004v2.pdf)

**2) DKAC (Deep Koopman Affine with Control)**

상태-종속 비선형성을 Affine 형태로 모델링:

$$\phi(x_{k+1}) = A\phi(x_k) + \mathcal{C}_\theta(x_k) u_k \quad \text{(식 12)} $$

보조 제어 네트워크:

$$B = g_\theta(x_k) $$

여기서 $g_\theta$는 신경망으로 매개변수화되어, 각 상태에서 제어의 영향을 학습합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/849aa21f-eeac-4a10-ad33-9ad6bc827c9c/2202.08004v2.pdf)

**3) DKN (Deep Koopman Nonlinear)**

완전 비선형 제어 항을 직접 인코딩:

$$\phi(x_{k+1}) = A\phi(x_k) + g_\theta(x_k, u_k) $$

여기서 $g_\theta(x_k, u_k)$ 는 상태와 제어 모두를 입력으로 받는 신경망입니다. 이 방법은 최고의 예측 성능을 제공하지만, 제어 입력 복원의 역함수 문제로 인해 제어 적용에 어려움이 있습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/849aa21f-eeac-4a10-ad33-9ad6bc827c9c/2202.08004v2.pdf)

#### 2.4 K-steps 손실 함수와 학습

**전통적 방법의 문제**

기존 Koopman 학습은 1-step 예측 오차만 최소화합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/849aa21f-eeac-4a10-ad33-9ad6bc827c9c/2202.08004v2.pdf)

$$\mathcal{L}\_{1\text{-step}} = \|A\phi_k + B u_k - \phi_{k+1}\|^2 $$

이는 장기 예측에서 오차가 누적되는 문제를 야기합니다.

**제안: K-steps 손실**

본 논문은 장기 예측을 고려하는 다음과 같은 손실 함수를 제안합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/849aa21f-eeac-4a10-ad33-9ad6bc827c9c/2202.08004v2.pdf)

$$ \mathcal{L} = \sum_{i=1}^{K} \sum_{j=1}^{B} \|\hat{\phi}(x_t^j, i) - \phi(x_{t+i}^j)\|^2 + \lambda \text{RegularizationTerms} \quad \text{(식 14)} $$

여기서:
- K는 미리 설정된 단계 수 (논문에서 K=15)
- B는 배치 크기
- $\hat{\phi}(x_t, i)$는 i단계 선행 예측

이 손실 함수는 단계별로 가중치를 설정하여, 모든 예측 단계를 고려합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/849aa21f-eeac-4a10-ad33-9ad6bc827c9c/2202.08004v2.pdf)

#### 2.5 임베딩 공간에서의 LQR 제어

**원래 공간의 최적 제어 문제**

원래 상태 공간에서의 표준 LQR 문제:

$$\min \sum_{t=0}^{\infty} (x_t^T Q x_t + u_t^T R u_t) \quad \text{s.t. } x_{t+1} = f(x_t, u_t) \quad \text{(식 15)} $$

**Embedding 공간으로의 변환**

제안 방법에서는 embedding 공간에서 이를 다시 정식화합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/849aa21f-eeac-4a10-ad33-9ad6bc827c9c/2202.08004v2.pdf)

$$\min \sum_{t=0}^{\infty} (\tilde{y}_t^T Q \tilde{y}_t + \tilde{u}_t^T R \tilde{u}_t) $$

$$\text{s.t. } \tilde{y}_{t+1} = K\tilde{y}_t + \tilde{u}_t \quad \text{(식 16)} $$

여기서 $\tilde{u}\_t = B^{-1} \mathcal{C}_\theta(x_t) u_t$ (DKAC의 경우). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/849aa21f-eeac-4a10-ad33-9ad6bc827c9c/2202.08004v2.pdf)

**폐쇄형 LQR 해**

이제 선형 시스템이므로 표준 LQR을 적용하여 최적 제어를 얻습니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/849aa21f-eeac-4a10-ad33-9ad6bc827c9c/2202.08004v2.pdf)

$$\tilde{u}_t = -K_{\text{LQR}} \tilde{y}_t \quad \text{(식 17)} $$

마지막으로 원 공간의 제어로 복원:

$$u_t = B^{-1} g_\theta(x_t) \tilde{u}_t \quad \text{(식 18)} $$

### 제3부: 모델 구조 및 성능

#### 3.1 신경망 아키텍처

논문에서 제시하는 신경망 구조는 다음과 같습니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/849aa21f-eeac-4a10-ad33-9ad6bc827c9c/2202.08004v2.pdf)

| 구성요소 | 설명 | 하이퍼파라미터 |
|---------|------|----------------|
| Embedding Network | 입력: 상태 $x$, 출력: $h_\theta(x)$ | 3 hidden layers, 128 units |
| Control Network | 입력: $(x, u)$, 출력: 비선형 제어 항 | 3 hidden layers, 128 units |
| Koopman A Network | Linear transformation | Single layer |
| Koopman B Network | 선형 제어 매트릭스 (DKUC용) | Single layer |
| 학습 설정 | 모든 깊은 학습 기반 방법 동일 | K=15, 학습률 일정 |

전체 프레임워크는 Figure 2에서 보이듯이 K-step 진화를 통해 end-to-end 학습됩니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/849aa21f-eeac-4a10-ad33-9ad6bc827c9c/2202.08004v2.pdf)

#### 3.2 성능 향상 분석

**예측 성능 비교 (15단계)**

| 환경 | 기존 KDerivative | KRBF | KDNN | KRNN | DKUC | DKAC | DKN |
|------|-----------------|------|------|------|------|------|------|
| DampingPendulum | 1.328e-2 | 9.814e-4 | 4.462e-3 | 4.695e-3 | 5.166e-3 | **6.465e-4** | 8.424e-4 |
| Pendulum | 1.936e-2 | 9.173e-2 | 2.020e-2 | 8.550e-2 | 3.138e-2 | 1.809e-2 | **2.677e-3** |
| CartPole | 2.031e-3 | 2.031e-3 | 2.923e-3 | 7.360e-3 | 1.042e-3 | **7.291e-4** | 8.117e-4 |

표에서 볼 수 있듯이, DKAC와 DKN이 대부분 환경에서 최고 성능을 달성합니다. 특히 DampingPendulum에서 DKAC는 기존 KRBF 대비 약 1.5배, KDerivative 대비 20배 이상 오차를 감소시킵니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/849aa21f-eeac-4a10-ad33-9ad6bc827c9c/2202.08004v2.pdf)

**제어 성능 비교 (총 비용)**

| 환경 | KDerivative | KRBF | DKUC | DKAC | iLQR |
|------|------------|------|------|------|------|
| DampingPendulum | 1196.69 | 1265.19 | 1183.10 | **1053.76** | 1072.97 |
| Pendulum | 307.73 | NaN | 305.39 | **309.23** | 301.10 |
| CartPole | 90.83 | NaN | 100.78 | **89.88** | 5265.19 |

이 표는 제어 성능에서 몇 가지 중요한 통찰을 제공합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/849aa21f-eeac-4a10-ad33-9ad6bc827c9c/2202.08004v2.pdf)

1. **안정성**: KRBF는 여러 환경에서 발산(NaN)하는 불안정성 보이며, 이는 RBF 중심점 선택의 의존성 때문
2. **우수성**: DKAC는 대부분 환경에서 iLQR과 비슷하거나 우수한 성능 달성
3. **DampingPendulum의 특이성**: 완전 비선형 제어에서만 DKAC가 실패를 방지할 수 있음. iLQR은 여기서도 1073의 비용으로 전역 최적이 아님. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/849aa21f-eeac-4a10-ad33-9ad6bc827c9c/2202.08004v2.pdf)

**표본 효율성**

논문에서 DampingPendulum 환경에서 표본 수에 따른 성능을 조사했습니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/849aa21f-eeac-4a10-ad33-9ad6bc827c9c/2202.08004v2.pdf)

- 200 샘플: 성능 미흡
- 1,000 샘플: 개선 시작
- 5,000 샘플 (약 25분 데이터): 충분한 성능 달성
- 50,000 샘플: 수렴

DKUC와 DKAC는 KDNN, KRBF와 비교하여 표본 개수 변화에 덜 민감하며, 5,000 샘플에서 이미 좋은 성능을 달성합니다. 이는 Koopman 기반 방법의 표본 효율성 우수성을 보여줍니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/849aa21f-eeac-4a10-ad33-9ad6bc827c9c/2202.08004v2.pdf)

#### 3.3 일반화 성능 분석

**원 공간 상태 보존의 역할**

본 논문의 embedding function 설계 $\phi(x, u) = [x^T, h_\theta(x)^T]^T$는 일반화 성능을 크게 향상시킵니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/849aa21f-eeac-4a10-ad33-9ad6bc827c9c/2202.08004v2.pdf)

1. **비용 함수 일관성**: 원 공간의 $Q, R$ 행렬이 embedding space에서도 동일하게 작용
2. **정보 보존**: 원래 상태를 완전히 보존하므로 정보 손실이 없음
3. **안정적 최적화**: Recovery matrix가 단순하므로 수치 안정성 향상

**K-steps 손실의 일반화 개선**

K-steps 손실 함수는 다음과 같이 일반화 성능을 개선합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/849aa21f-eeac-4a10-ad33-9ad6bc827c9c/2202.08004v2.pdf)

1. **과적합 방지**: 여러 단계의 오차를 동시에 최소화하므로, 1단계 오차만 최소화하는 모델의 과적합을 방지
2. **장기 안정성**: 20-30 단계 예측에서도 안정적 성능 유지
3. **제어 안정성**: 장기 제어 성능이 개선됨

**비선형 제어 항 인코딩의 효과**

DKAC와 DKN이 DKUC보다 우수한 성능을 보이는 이유: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/849aa21f-eeac-4a10-ad33-9ad6bc827c9c/2202.08004v2.pdf)

- DKUC: 완전 비선형 시스템을 일부 선형화하므로 모델링 오차 발생
- DKAC/DKN: 제어의 상태-종속 비선형성을 명시적으로 모델링하여 표현력 증가

특히 DampingPendulum에서 DKAC의 성능이 현저히 우수한 것은, 이 시스템의 제어 입력 비선형성이 강하기 때문입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/849aa21f-eeac-4a10-ad33-9ad6bc827c9c/2202.08004v2.pdf)

### 제4부: 한계 및 미래 방향

#### 4.1 논문에서 명시한 한계

**한계 1: DKN의 역함수 문제**

DKN 알고리즘이 최고의 예측 성능을 제공하지만, 실제 제어 적용에 어려움이 있습니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/849aa21f-eeac-4a10-ad33-9ad6bc827c9c/2202.08004v2.pdf)

- 문제: $u_t = g_\theta^{-1}(x_t, \tilde{u}_t)$의 역함수 복원 불가능
- 이유: $g_\theta$가 상태와 제어 모두의 비선형 함수이므로, 일반적으로 특정 상태에서 역함수가 유일하지 않음
- 해결 방안 (논문 제안):
  - Invertible network (예: coupling flows, autoencoder) 사용
  - 자동 미분을 활용한 역함수 근사

**한계 2: 제약 조건 처리**

입출력 제약 조건이 embedding space에서 왜곡됩니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/849aa21f-eeac-4a10-ad33-9ad6bc827c9c/2202.08004v2.pdf)

- 상태 제약: $x \in \mathcal{X}$는 $\phi$의 첫 번째 부분이 제약 만족 → 쉬움
- 제어 제약: $u \in \mathcal{U}$는 제어 네트워크의 역함수를 통해 표현 → 어려움

**한계 3: 고차원 시스템 확장성**

현재 실험은 7 DOF 로봇 팔까지만 검증되었으며, 훨씬 더 복잡한 시스템(소프트 로봇 등)에서의 성능은 미검증입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/849aa21f-eeac-4a10-ad33-9ad6bc827c9c/2202.08004v2.pdf)

#### 4.2 최근 연구의 개선 사항 (2022-2025)

**2024-2025년 연구의 발전**

본 논문 이후 커뮤니티가 이루어낸 발전을 분석하면 다음과 같습니다:

| 연구 방향 | 주요 논문 | 개선 사항 |
|----------|---------|---------|
| **일반화 이론** | Hashimoto et al. (2024) [arxiv](https://arxiv.org/pdf/2302.05825.pdf), Mohammadigohari et al. (2025) [arxiv](https://arxiv.org/pdf/2512.19199.pdf) | Koopman 기반 일반화 경계 증명, 다중태스크 학습 일반화 |
| **안정성 보장** | Learning Noise-Robust Stable Koopman (2025) [arxiv](https://arxiv.org/html/2408.06607v3), Provably-Stable NN Control (2025) [arxiv](https://arxiv.org/abs/2502.00248) | Lyapunov 기반 안정성 증명, 노이즈 견고성 |
| **고급 구조** | DDKC (2024) [arxiv](http://arxiv.org/pdf/2412.07212.pdf), Bilinear Koopman (2025) [arxiv](https://arxiv.org/html/2507.12578v1), Transformer+Koopman (2025) [openreview](https://openreview.net/forum?id=hbzpEmhMvx) | 분산 학습, 입력-상태 쌍선형성, 주의 메커니즘 |
| **응용 확대** | Deep Koopman + DRO (2025) [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0950705125017903), Satellite Attitude Control (2025) [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0273117725006015) | 불확실성 처리, 실제 로봇 적용 |
| **Transfer Learning** | Hjikakou et al. (2025) [openreview](https://openreview.net/forum?id=hbzpEmhMvx) | Cross-task generalization, fine-tuning 전략 |

**1) 일반화 성능 이론 발전**

Hashimoto et al. (2024)의 연구는 Koopman 연산자를 사용한 신경망 일반화 경계를 제시합니다: [arxiv](https://arxiv.org/pdf/2302.05825.pdf)

$$\text{Generalization Bound} \propto \frac{\|\det(W_j)\|}{n^{1/2}} $$

여기서 $W_j$는 가중치 행렬입니다. 이는 다음을 시사합니다:

- 직교 가중치 행렬 ($\det(W) = 1$)을 사용하면 일반화 경계가 네트워크 폭과 무관
- 본 논문의 "원래 상태 보존" 설계와 직교 가중치 결합 시 이론적 보장 가능

**2) 안정성-성능 트레이드오프**

2024-2025년 연구는 안정성 제약을 추가하여 성능과 안정성을 동시에 보장합니다: [arxiv](https://arxiv.org/html/2408.06607v3)

$$\text{minimize } \mathcal{L}(A, B, g_\theta) $$

$$\text{subject to } \|A\| < 1 \text{ (안정성 제약)} $$

이는 다음의 이점을 제공합니다:
- 오버핏팅 방지 (정규화 효과)
- 실제 환경의 약간의 모델 오류에도 안정성 보증
- 롱호라이즌 제어 성능 향상

**3) 고급 신경망 구조**

최근 연구들은 더 표현력 있는 구조를 제안합니다:

- **쌍선형 Koopman** (2025): 입력-상태 상호작용을 명시적으로 모델링

$$\phi_{k+1} = A\phi_k + \sum_i B_i(x_k) u_i $$
  
- **트랜스포머 기반** (2025): 긴 시계열 의존성 캡처
  - Self-attention으로 Koopman 임베딩 학습
  - 다양한 동역학 동시 처리 가능

**4) Transfer Learning과 일반화**

Hjikakou et al. (2025)의 연구는 Koopman 임베딩의 전이 가능성을 입증했습니다: [openreview](https://openreview.net/forum?id=hbzpEmhMvx)

1. **Pre-training**: Lorenz 시스템에서 Koopman 임베딩 학습
2. **Fine-tuning**: 안전 제어 태스크로 미세 조정
3. **결과**: PCA 기반 기법 대비 우수 성능, 데이터 효율성 향상

이는 Koopman 임베딩이 학습한 동역학 구조가 재사용 가능함을 시사합니다.

**5) 실제 적용 사례**

최근 연구들이 복잡한 실제 시스템에 적용되고 있습니다: [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0950705125017903)

- **위성 자세 제어** (2025): Deep Koopman + 분산로버스트 최적화
  - 불확실성 처리로 실제 성능 개선
  
- **차량 제어** (Frenet 좌표계, 2025): 곡선 경로 추적
  - 실시간 MPC 가능
  
- **탄소 포집** (2025): 산업 프로세스 제어
  - 경제적 MPC로 운영비 절감

### 제5부: 앞으로의 연구 시 고려할 점

#### 5.1 이론적 기초 강화

**필요 연구**:

1. **Koopman 연산자의 유한근사 오류 경계**
   - 현재: 무한차원 연산자를 신경망으로 근사할 때 오류에 대한 엄밀한 경계 부재
   - 필요: $\|\mathcal{K} - \mathcal{K}_\theta\|$에 대한 통계적 학습 이론 기반 경계

2. **일반화와 샘플 복잡도**
   - 현재: K-steps 손실이 일반화를 개선함을 경험적으로만 보임
   - 필요: 다항식 표본 복잡도 증명 (일반화 갭 축소 조건)

3. **다중 도메인 학습 이론**
   - 현재: Transfer learning 성공 사례 증가
   - 필요: Domain gap 하한(lower bound) 도출, 최적 전이 조건 규명

#### 5.2 기술적 개선

**핵심 과제**:

1. **DKN의 역함수 문제 해결**
   - **방향 1**: 정규화된 역신경망

$$g_\theta^{-1}(x, \tilde{u}) = \arg\min_u \|g_\theta(x, u) - \tilde{u}\|^2 + \lambda \|u\| $$
   
   - **방향 2**: 쌍방향 학습
     Invertible network를 사용하여 정확한 역변환 학습

$$g_\theta: (x, u) \rightarrow \tilde{u}, \quad g_\theta^{-1}: (x, \tilde{u}) \rightarrow u $$

2. **제약 조건 처리**
   - **상태 제약**: 첫 n개 성분에 직접 제약
   - **제어 제약**: 다음과 같은 constrained MPC 공식화

$$\text{minimize } \sum_{t} \|\tilde{y}_t\|_Q^2 + \|\tilde{u}_t\|_R^2 $$

$$\text{subject to } \tilde{y}_{t+1} = K\tilde{y}_t + \tilde{u}_t, \quad u_t \in \mathcal{U} $$

여기서 $u_t = g_\theta^{-1}(x_t, \tilde{u}_t)$에 제약 대입

3. **계산 효율성**
   - 현재: 단일 머신 학습만 고려
   - 필요: 분산 Koopman 학습 (논문) [arxiv](http://arxiv.org/pdf/2412.07212.pdf)
     - 여러 에이전트에서 데이터 수집
     - 합의 알고리즘으로 Koopman 연산자 분산 학습
     - 통신 효율성 및 수렴 속도 분석

#### 5.3 응용 및 확장

**확장 가능 영역**:

| 응용 분야 | 현재 진전 | 향후 과제 |
|----------|---------|---------|
| **고차원 로봇** | 7 DOF 팔 | 휴머노이드, 다중 팔 로봇 |
| **소프트 로봇** | 미검증 | 수백 개 자유도, 하이브리드 제어 |
| **부분 관측** | 상태 관측 가정 | Observer 설계, 센서 노이즈 처리 |
| **하이브리드 시스템** | 연속 동역학 | 이산-연속 전환, 접촉 동역학 |
| **불확실성** | 확정적 시스템 | Robust Koopman, 적응 제어 |

#### 5.4 추천되는 미래 연구 로드맵

**단기 (1-2년)**:
1. 안정성 보장 추가 (Lyapunov 함수 기반)
2. 부분 상태 관측 하에서의 학습
3. 실제 로봇 플랫폼에서의 검증

**중기 (2-4년)**:
1. DKN의 역함수 문제 완전 해결 (invertible network 활용)
2. 입출력 제약 포함 최적화
3. Transfer learning 이론과 실제 연구

**장기 (4년 이상)**:
1. 초고차원 시스템으로의 확장 (1000+ DOF)
2. 신경망 구조의 자동 설계 (NAS + Koopman)
3. Physics-informed Koopman 개발

***

### 결론

"Deep Koopman Operator with Control for Nonlinear Systems"는 비선형 동적 시스템의 데이터 기반 제어에서 중대한 기여를 제공합니다. 핵심적으로:

1. **혁신성**: 제어 입력의 상태-종속 비선형성을 명시적으로 모델링하는 첫 번째 체계적 접근
2. **성능**: 기존 Koopman 방법 대비 수 배 우수한 예측 오차, 더 안정적인 제어
3. **실용성**: 5,000 샘플로 충분한 성능 달성, 선형 제어 이론 직접 적용

이후 2022-2025년 연구의 발전은 이 기초 위에서 안정성 보장, 일반화 이론, 고급 구조, 실제 적용까지 다층적으로 진행되고 있습니다. 향후 연구는 역함수 문제 해결, 제약 처리, 고차원 시스템 확장에 집중될 것으로 예상되며, 이들이 해결될 때 Koopman 기반 제어는 현대 로보틱스와 자동화 분야에서 주류 기술이 될 잠재력이 높습니다.

***

### 참고 문헌 (논문에서 인용)

<span style="display:none">[^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_80][^1_81][^1_82][^1_83][^1_84][^1_85][^1_86][^1_87][^1_88][^1_89][^1_90][^1_91]</span>

<div align="center">⁂</div>

[^1_1]: 2202.08004v2.pdf

[^1_2]: https://arxiv.org/pdf/2302.05825.pdf

[^1_3]: https://arxiv.org/pdf/2512.19199.pdf

[^1_4]: https://arxiv.org/html/2408.06607v3

[^1_5]: https://arxiv.org/abs/2502.00248

[^1_6]: http://arxiv.org/pdf/2412.07212.pdf

[^1_7]: https://arxiv.org/html/2507.12578v1

[^1_8]: https://openreview.net/forum?id=hbzpEmhMvx

[^1_9]: https://www.sciencedirect.com/science/article/abs/pii/S0950705125017903

[^1_10]: https://www.sciencedirect.com/science/article/abs/pii/S0273117725006015

[^1_11]: https://arxiv.org/pdf/2508.18954.pdf

[^1_12]: https://epubs.siam.org/doi/pdf/10.1137/16M1062296

[^1_13]: https://www.mdpi.com/1424-8220/25/23/7108

[^1_14]: http://medrxiv.org/lookup/doi/10.1101/2025.10.10.25337626

[^1_15]: https://iopscience.iop.org/article/10.1149/MA2025-02291587mtgabs

[^1_16]: https://al-kindipublisher.com/index.php/ijlps/article/view/10953

[^1_17]: https://invergejournals.com/index.php/ijss/article/view/132

[^1_18]: https://academic.oup.com/rheumap/article/doi/10.1093/rap/rkaf111.011/8313158

[^1_19]: https://academic.oup.com/bib/article/26/Supplement_1/i1/8378014

[^1_20]: https://invergejournals.com/index.php/ijss/article/view/166

[^1_21]: https://invergejournals.com/index.php/ijss/article/view/168

[^1_22]: https://arxiv.org/pdf/2211.08992.pdf

[^1_23]: http://arxiv.org/pdf/2412.01085.pdf

[^1_24]: https://arxiv.org/html/2406.02875

[^1_25]: https://arxiv.org/html/2503.03002v1

[^1_26]: https://arxiv.org/pdf/2212.13828.pdf

[^1_27]: https://arxiv.org/pdf/2307.05884.pdf

[^1_28]: https://arxiv.org/html/2505.08122v2

[^1_29]: https://pubs.aip.org/asa/jasa/article/158/1/154/3351768/Dynamic-neural-network-switching-for-active

[^1_30]: https://arxiv.org/abs/2508.07494

[^1_31]: https://skoge.folk.ntnu.no/prost/proceedings/ifac2002/data/content/00132/132.pdf

[^1_32]: https://www.semanticscholar.org/paper/Deep-learning-for-Koopman-Operator-Optimal-Control.-Al‐Gabalawy/000794cdc9985ec8e3008d8f4e61c56e1dc96820

[^1_33]: https://www.emergentmind.com/topics/koopman-embeddings

[^1_34]: https://www.sciencedirect.com/science/article/abs/pii/S0273117720307821

[^1_35]: https://www.nature.com/articles/s41467-018-07210-0

[^1_36]: https://arxiv.org/html/2505.14828v1

[^1_37]: http://koasas.kaist.ac.kr/bitstream/10203/8173/1/000188464600030.pdf

[^1_38]: https://www.sciencedirect.com/science/article/abs/pii/S0021999124000445

[^1_39]: https://pubmed.ncbi.nlm.nih.gov/31493874/

[^1_40]: https://arxiv.org/pdf/2504.06818.pdf

[^1_41]: https://pubmed.ncbi.nlm.nih.gov/31251205/

[^1_42]: https://arxiv.org/pdf/2110.06509.pdf

[^1_43]: https://www.arxiv.org/pdf/2504.00352v1.pdf

[^1_44]: https://pubmed.ncbi.nlm.nih.gov/18255659/

[^1_45]: https://arxiv.org/pdf/2102.02522.pdf

[^1_46]: https://arxiv.org/pdf/2206.06536.pdf

[^1_47]: https://arxiv.org/html/2504.06818v2

[^1_48]: https://arxiv.org/pdf/2505.16511.pdf

[^1_49]: https://arxiv.org/html/2402.07834v2

[^1_50]: https://onepetro.org/SPEHFTC/proceedings/24HFTC/24HFTC/D021S003R003/540586

[^1_51]: https://sportscience.ldufk.edu.ua/index.php/discourse/article/view/1633

[^1_52]: https://eduvest.greenvest.co.id/index.php/edv/article/view/50021

[^1_53]: https://journal.banjaresepacific.com/index.php/jimr/article/view/707

[^1_54]: https://onepetro.org/JPT/article/74/06/14/494935/E-amp-P-Notes-June-2022

[^1_55]: https://finukr.com.ua/index.php/journal/article/view/29

[^1_56]: https://onepetro.org/ARMAUSRMS/proceedings/ARMA24/ARMA24/D031S034R001/549545

[^1_57]: https://aacrjournals.org/cancerres/article/84/6_Supplement/1056/735828/Abstract-1056-Development-of-a-B-cell-epitope

[^1_58]: https://www.business-inform.net/export_pdf/business-inform-2024-3_0-pages-225_234.pdf

[^1_59]: https://onepetro.org/JPT/article/76/05/8/544711/Comments-Grabbing-the-Brass-Ring-To-Power-the

[^1_60]: http://arxiv.org/pdf/2403.12335.pdf

[^1_61]: http://arxiv.org/pdf/2305.16215.pdf

[^1_62]: https://arxiv.org/pdf/2211.01365.pdf

[^1_63]: http://arxiv.org/pdf/1712.09707v2.pdf

[^1_64]: https://arxiv.org/pdf/2402.07834.pdf

[^1_65]: https://proceedings.iclr.cc/paper_files/paper/2024/file/520fa95508d43d4c5bdfc966c05aff45-Paper-Conference.pdf

[^1_66]: https://www.sciencedirect.com/science/article/abs/pii/S0925231224017223

[^1_67]: http://terrano.ucsd.edu/jorge/publications/data/2024_HaCo-access.pdf

[^1_68]: http://pdclab.seas.ucla.edu/Publications/MAlhajeri/Alhajeri_2024.pdf

[^1_69]: https://arxiv.org/pdf/2403.10623.pdf

[^1_70]: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4900537

[^1_71]: http://arxiv.org/pdf/1710.10256.pdf

[^1_72]: https://openreview.net/forum?id=JN7TcCm9LF

[^1_73]: https://www.themoonlight.io/ko/review/transfer-learning-for-control-systems-via-neural-simulation-relations

[^1_74]: https://pubs.aip.org/aip/cha/article/32/3/033116/2835753/Deep-learning-enhanced-dynamic-mode-decomposition

[^1_75]: https://huggingface.co/papers?q=Koopman+operator

[^1_76]: https://www.nature.com/articles/s41598-025-10021-1

[^1_77]: https://arxiv.org/abs/2405.15945

[^1_78]: https://arxiv.org/html/2302.05825v3

[^1_79]: https://pubmed.ncbi.nlm.nih.gov/40617924/

[^1_80]: https://arxiv.org/pdf/2202.08004.pdf

[^1_81]: https://arxiv.org/html/2510.16016v1

[^1_82]: https://arxiv.org/html/2503.02961v1

[^1_83]: https://arxiv.org/abs/1812.03399

[^1_84]: https://arxiv.org/html/2512.19184v1

[^1_85]: https://ar5iv.labs.arxiv.org/html/1811.09864

[^1_86]: https://arxiv.org/abs/2404.17466

[^1_87]: https://arxiv.org/html/2508.07494v1

[^1_88]: https://arxiv.org/html/2412.01783v1

[^1_89]: https://arxiv.org/html/2506.22304v2

[^1_90]: https://arxiv.org/html/2502.00782v1

[^1_91]: http://arxiv.org/pdf/2501.16489v1.pdf
