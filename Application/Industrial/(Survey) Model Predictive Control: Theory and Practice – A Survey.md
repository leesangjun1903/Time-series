# Model Predictive Control: Theory and Practice – A Survey

# 1. 논문 개요와 핵심 주장

Garcia, Prett, Morari의 “Model Predictive Control: Theory and Practice – A Survey”는 1980년대까지의 **모델 예측 제어(MPC)**를 이론·실무 관점에서 통합 정리한 대표적 서베이이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6b7b8043-574d-4eca-a4a9-4a658d8528c5/ModelPredictiveControl_TheoryandPracticeaSurveyGarca1989.pdf)

이 논문의 핵심 주장은 다음 네 가지로 요약된다.

1. **핵심 주장**
   - 산업 공정(특히 석유·석유화학)에서 요구되는 경제성, 안전, 품질, 장비 제약 등을 만족하려면 **다변수·제약 기반 동적 최적화 문제**를 직접 풀 수 있는 제어 구조가 필요하다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6b7b8043-574d-4eca-a4a9-4a658d8528c5/ModelPredictiveControl_TheoryandPracticeaSurveyGarca1989.pdf)
   - **MPC는 “명시적 모델 + 제한 시간 최적화 + 리시딩 호라이즌(receding horizon)” 구조를 통해, 제약을 체계적으로 다루면서 높은 성능을 내는 거의 유일한 일반적 방법론**이라고 주장한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6b7b8043-574d-4eca-a4a9-4a658d8528c5/ModelPredictiveControl_TheoryandPracticeaSurveyGarca1989.pdf)
   - **Dynamic Matrix Control(DMC), Model Algorithmic Control(MAC), Inferential Control, Internal Model Control(IMC)**, 그리고 고전 LQ 제어 등 다양한 기법을 **IMC/Q-파라미터화 관점에서 통합된 하나의 구조**로 재해석한다. [scribd](https://www.scribd.com/document/649190281/1982-Garcia-C-Internal-Model-Control-IMC-a-Unifying-Review-and-Some-New-Results)
   - **MPC는 고전 피드백보다 본질적으로 더(혹은 덜) 강인한 것이 아니지만, IMC 필터·호라이즌·가중치 조절을 통해 강인성을 훨씬 더 “직접적으로” 튜닝할 수 있다**고 결론짓는다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6b7b8043-574d-4eca-a4a9-4a658d8528c5/ModelPredictiveControl_TheoryandPracticeaSurveyGarca1989.pdf)

2. **주요 기여**
   - 산업 MPC(특히 석유화학 플랜트)의 요구 조건을 체계적으로 정리하고, 이를 기반으로 **“일반적인 제어 문제 정식화”**를 제시.
   - DMC와 MAC를 공통된 수식 구조(스텝/임펄스 응답 모델 + 2-노름 비용 + 제약)로 정식화하고, **선형제약 없는 경우 LQ/LQG와의 등가성**을 상세히 분석. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6b7b8043-574d-4eca-a4a9-4a658d8528c5/ModelPredictiveControl_TheoryandPracticeaSurveyGarca1989.pdf)
   - MPC 스킴이 실제로는 **IMC(Internal Model Control) 구조**라는 것을 명확히 보이고, IMC–고전 피드백 구조의 수학적 동치를 정리함으로써, **Q-파라미터화(Youla parameterization)**를 MPC 관점에서 재조명. [cse.lab.imtlucca](http://cse.lab.imtlucca.it/~bemporad/publications/papers/survey-robust-mpc.pdf)
   - **제약 포함 MPC(QP 기반 QDMC, LP 기반 1-노름/∞-노름 MPC)**, 비선형 MPC, 강인성 논의를 최초의 체계적인 프레임으로 묶어, 이후 robust MPC·NMPC·learning-based MPC의 이론 발전 토대를 제공. [scribd](https://www.scribd.com/document/661205288/Constrained-Model-Predictive-Control-Sta)

***

# 2. 논문이 다루는 문제, 방법, 모델 구조, 성능·한계

## 2.1 문제 정의: 공정 산업에서의 일반 제어 문제

석유·석유화학 공정은 다음과 같은 실무 성능 기준을 동시에 만족해야 한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6b7b8043-574d-4eca-a4a9-4a658d8528c5/ModelPredictiveControl_TheoryandPracticeaSurveyGarca1989.pdf)

- 경제성: 최적화 레이어가 제시하는 목표 근처에서 운전, 동적 비용 최소화.
- 안전·환경: 온도, 압력, 조성 등이 안전·환경 한계를 넘지 않도록 해야 함.
- 장비 제약: 밸브 개도, 펌프 용량, 열교환 용량 등 장비 한계를 넘지 않도록.
- 품질·인간 친화성: 제품 규격 만족, 지나친 조작/출력 출렁임 방지.

이를 수학적으로 요약하면:

> “**시간에 따라 변하는 다수의 목표와 여러 동적 제약 하에서, 조작변수를 온라인으로 갱신하여 성능 기준을 만족하는 문제**” [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6b7b8043-574d-4eca-a4a9-4a658d8528c5/ModelPredictiveControl_TheoryandPracticeaSurveyGarca1989.pdf)

이를 위해, 공정 모형을

$$
x(k+1) = A x(k) + B u(k),\quad y(k) = C x(k)
$$

또는 이에 상응하는 전달함수/임펄스·스텝 응답 모델로 두고, [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6b7b8043-574d-4eca-a4a9-4a658d8528c5/ModelPredictiveControl_TheoryandPracticeaSurveyGarca1989.pdf)

- 비용: **미래 출력·입력의 2-노름 누적**
- 제약: **입력·출력에 대한 불등식 제약(물리 한계·안전 한계)**

으로 정식화한다.

***

## 2.2 기본 MPC 정식화 (DMC를 중심으로)

### 2.2.1 모델: 스텝 응답 기반 예측

안정한 선형 MIMO 시스템에서, 스텝 응답 계수 $\(H_i\)$를 사용해 미래 출력을

$$
y(k) = \sum_{i=1}^{n} H_i \Delta u(k-i) + H_n u(k-n)
$$

으로 근사하고, [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6b7b8043-574d-4eca-a4a9-4a658d8528c5/ModelPredictiveControl_TheoryandPracticeaSurveyGarca1989.pdf)

$$
\Delta u(k) = u(k) - u(k-1)
$$

라 두면, 예측 시점 $\(k\)$ 에서 미래 $\(p\)$ 스텝의 예측 $\( \hat y(k+\ell|k) \)$ 는

```math
\hat y(k+\ell|k) = 
\sum_{i=1}^{\ell} H_i \Delta u(k+\ell-i) 
+ \sum_{i=1}^{n-1} H_i \Delta u(k-i) 
+ H_n u(k+\ell-n) + \hat d(k+\ell|k)
```

형태가 된다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6b7b8043-574d-4eca-a4a9-4a658d8528c5/ModelPredictiveControl_TheoryandPracticeaSurveyGarca1989.pdf)

여기서 $\(\hat d(k+\ell|k)\)$ 는 출력에서 모델 예측값을 뺀 **추정된 외란(모델 불일치)**이다.

### 2.2.2 비용 함수 및 최적화

DMC에서 조작변수 변화 $\(\Delta u(k), \dots, \Delta u(k+m-1)\)$ 를 결정하기 위해 최소화하는 전형적인 목적함수는

$$
\min_{\Delta u(k),\dots,\Delta u(k+m-1)} 
\sum_{\ell=1}^{p} 
\lVert \hat y(k+\ell|k) - r(k+\ell)\rVert_{Q}^2
+
\sum_{\ell=1}^{m} 
\lVert \Delta u(k+\ell-1)\rVert_{R}^2
$$

이며, 여기서 $\(Q \succ 0, R \succeq 0\)$ 는 출력·입력 가중치 행렬이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6b7b8043-574d-4eca-a4a9-4a658d8528c5/ModelPredictiveControl_TheoryandPracticeaSurveyGarca1989.pdf)

제약이 없는 경우, 이는 **표준 선형 최소제곱 문제**이고, 해가 닫힌 형식으로 존재한다. 제약(입력 상·하한, 출력 범위 등)이 포함되면,

- 목적함수: 2-노름 제곱 (quadratic)
- 제약: 선형 부등식

이므로 **Quadratic Programming (QP)** 문제가 되어, 매 샘플마다 QP를 푸는 구조(QDMC)를 갖는다. [research-collection.ethz](https://www.research-collection.ethz.ch/bitstreams/fa7d4c8a-f781-4499-9041-f5ccdc17dbbc/download)

***

## 2.3 MAC, 필터링, IMC 구조

### 2.3.1 MAC: 임펄스 응답 + 외란 필터

MAC는 기본적으로 **임펄스 응답 모델 + 입력 자체 $\(u\)$ 에 대한 가중**을 사용하는 MPC 변종이다. 여기서 중요한 차이는 **외란 추정 필터**:

$$
\hat d(k+1|k) = \alpha \hat d(k|k-1) + (1-\alpha)\big(y_m(k) - \hat y(k)\big),
\quad 0 \le \alpha \le 1
$$

를 도입해, 모델 불일치를 1차 필터로 스무딩하여 **폐루프 대역폭·강인성을 직접 튜닝하는 스칼라 파라미터**로 사용한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6b7b8043-574d-4eca-a4a9-4a658d8528c5/ModelPredictiveControl_TheoryandPracticeaSurveyGarca1989.pdf)

이는 뒤에서 설명하는 IMC의 저역통과 필터와 동일한 역할을 하며, **“폐루프 시간 상수 ↔ $\(\alpha\)$ ”를 거의 일대일로 연결**해준다. [scribd](https://www.scribd.com/document/649190281/1982-Garcia-C-Internal-Model-Control-IMC-a-Unifying-Review-and-Some-New-Results)

### 2.3.2 IMC 구조와 고전 피드백과의 동치

논문은 모든 MPC 스킴이 아래 IMC 구조와 동치임을 보인다. [scribd](https://www.scribd.com/document/649190281/1982-Garcia-C-Internal-Model-Control-IMC-a-Unifying-Review-and-Some-New-Results)

- 플랜트 $\(P\)$
- 플랜트 모델 $\(\tilde P\)$ (이상적으로 $\(P\)$ 와 동일)
- IMC 제어기 $\(Q\)$

도식적으로,

- IMC:

$$
y = P \big( Q (r - \hat y) + d \big), \quad \hat y = \tilde P u
$$

- 고전 피드백:

$$
u = C (r - y)
$$

이 두 구조가 **외부 입·출력에 대해 완전히 동치**가 되려면,

$$
Q = C (I + \tilde P C)^{-1}, \quad
C = Q (I - \tilde P Q)^{-1}
$$

라는 **Youla–Kučera Q-파라미터화 관계**를 만족해야 한다. [cse.lab.imtlucca](http://cse.lab.imtlucca.it/~bemporad/publications/papers/survey-robust-mpc.pdf)

주요 결과:

- $\(P\)$ 가 안정이고 $\(\tilde P = P\)$ 라면, **IMC 폐루프 안정성 조건은 “Q가 안정”인 것뿐**이다.
- 출력·외란 응답은

$$
y = P Q (r - d) + d, \quad 
e = y - r = (I - P Q)(d - r)
$$

이 되어, 성능은 Q에 **선형(affine)**으로 의존한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6b7b8043-574d-4eca-a4a9-4a658d8528c5/ModelPredictiveControl_TheoryandPracticeaSurveyGarca1989.pdf)

이는 “안정한 Q 집합” 상에서 성능을 최적화하는 것이, 직접 C를 최적화하면서 폐루프 안정성을 강제하는 것보다 훨씬 단순함을 의미하며, MPC/IMC의 설계 이점을 이론적으로 정당화한다. [scribd](https://www.scribd.com/document/649190281/1982-Garcia-C-Internal-Model-Control-IMC-a-Unifying-Review-and-Some-New-Results)

***

## 2.4 Unconstrained MPC와 LQ/LQG와의 관계

논문은 제약이 없을 때, 일정한 가중과 무한 호라이즌으로 한계 과정을 취하면, MPC가 **표준 Linear Quadratic Control(LQC)**과 거의 동등해짐을 보인다. [chee.uh](https://www.chee.uh.edu/sites/chbe/files/faculty/212/mpctheoryrevised.pdf)

- 상태·출력 방정식에 Wiener 프로세스 형태의 외란을 두고,
- LQ 비용

$$
J = \mathbb{E}\sum_{k=0}^{\infty} 
\big( y(k)^\top R_3 y(k) + u(k)^\top R_2 u(k) \big)
$$

를 최소화하는 문제를 구성하면,
- 최적 제어는

$$
\Delta u(k) = F_1 \hat x(k) + F_2 (\hat y(k) - r(k))
$$

형태의 선형 피드백이 되며, [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6b7b8043-574d-4eca-a4a9-4a658d8528c5/ModelPredictiveControl_TheoryandPracticeaSurveyGarca1989.pdf)
- 이 해를 MPC 관점에서 보면 “무한 호라이즌, 일정 가중의 리시딩-호라이즌 MPC”와 수렴적으로 동일하다.

따라서:

- **Unconstrained MPC = IMC 구조를 따르는 LQ/LQG 제어**로 해석 가능.
- MPC의 이점은 **“제약이 붙었을 때 자연스럽게 비선형·시변 제어법으로 확장 가능”**하다는 점에 있다. [chee.uh](https://www.chee.uh.edu/sites/chbe/files/faculty/212/mpctheoryrevised.pdf)

***

## 2.5 제약 포함 MPC: 2-노름, 1-노름, ∞-노름

제약이 중요한 상황(실제 산업)에서 논문은 세 가지 대표적 정식화를 비교한다. [research-collection.ethz](https://www.research-collection.ethz.ch/bitstreams/fa7d4c8a-f781-4499-9041-f5ccdc17dbbc/download)

| 정식화 | 목적함수 노름 | 수학적 문제 | 장점 | 단점 |
|--------|---------------|-------------|------|------|
| 2-노름 (QDMC) | $\(\ell_2\)$ (제곱합) | QP | 이론 분석(안정성·성능) 용이, 수학적으로 잘 이해됨 | 변수 수가 많을 때 연산량·메모리 부담 |
| 1-노름 | $\(\ell_1\)$ | LP | QP보다 해석·구현이 단순, 피크 억제 능력 개선 가능 | 닫힌형 제어법 도출 어려움 → 안정성 일반 이론 부족 |
| ∞-노름 | $\(\ell_\infty\)$ | LP | 최대 편차(피크)를 직접 최소화, 안전 여유 극대화에 적합 | 설계 직관은 좋으나 해석·튜닝 경험 부족 |

논문은 **“제약이 활성화되지 않는 영역에서는 선형 시간불변 제어로 동작하고, 제약이 걸릴 때만 비선형·시변 제어가 된다”**는 점을 강조하며, **Unconstrained 영역에서의 분석(특히 IMC 기반 분석)을 유지할 것을 권장**한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6b7b8043-574d-4eca-a4a9-4a658d8528c5/ModelPredictiveControl_TheoryandPracticeaSurveyGarca1989.pdf)

***

## 2.6 비선형 MPC (NMPC)

논문은 비선형 시스템에 대해 세 가지 접근을 소개한다. [d-nb](https://d-nb.info/979741750/34)

1. **비선형 최적제어 직접 해법**
   - 일반 비선형 OCP

$$
     \min_u\; G(x(T)) + \int_0^T F(x(t),u(t)) dt
     $$

     subject to

$$
     \dot x = f(x,u),\quad h(x,u) = 0,\quad g(x,u) \le 0
     $$

   - 변분법/필수 조건을 이용한 직접 해석은 온라인에는 부적합(연산량·복잡도).

2. **선형화 + QDMC**
   - 현재 작동점 주변에서 선형화 → 스텝 응답 모델 추출 → QDMC 적용.
   - 비선형 모델은 예측에만 사용하고, 최적화는 선형 모델로 수행(“모델은 비선형, 제어법은 선형”). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6b7b8043-574d-4eca-a4a9-4a658d8528c5/ModelPredictiveControl_TheoryandPracticeaSurveyGarca1989.pdf)

3. **비선형 연산자 역(inversion)**
   - 목표 $\(r\)$ 에 대해

$$
     P u = r
     $$

  를 만족하는 입력 $\(u\)$ 를 찾는 **비선형 역연산** 문제로 해석하고, 수치적 역연산(뉴턴법, 고정점 반복 등)을 사용. [d-nb](https://d-nb.info/979741750/34)

논문 시점에서는 이들 접근에 대해 “개념은 분명하지만, **일반 이론·계산 효율·제약·강인성까지 모두 만족하는 완성된 프레임워크는 없다**”고 평가한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6b7b8043-574d-4eca-a4a9-4a658d8528c5/ModelPredictiveControl_TheoryandPracticeaSurveyGarca1989.pdf)

***

## 2.7 성능 향상 및 한계

### 성능 향상 요인

- **다변수·제약 포함**에서 고전 제어(분리 루프, 오버라이드 로직 등)에 비해,
  - 플랜트 전체를 하나의 **동적 최적화 문제**로 취급.
  - 제약을 설계·튜닝 단계에서 투명하게 반영.
- 실제 응용( FCCU, 하이드로크래커, 복잡 증류열교환 네트워크 등)에서 **운전 자동화, 품질 개선, 에너지 절감**이 반복적으로 보고됨. [academia](https://www.academia.edu/6160090/Model_predictive_control_Theory_and_practice_A_survey)

### 한계

- **계산 비용**: 당시 CPU 성능으로는 긴 호라이즌·다수 입력·복잡 제약을 다루는 QP/LP를 실시간으로 풀기 어렵고, 산업 적용은 상당한 엔지니어링 노하우에 의존. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6b7b8043-574d-4eca-a4a9-4a658d8528c5/ModelPredictiveControl_TheoryandPracticeaSurveyGarca1989.pdf)
- **모델 불확실성**: MPC는 모델 정확성에 민감하며, 논문은 “MPC는 고전 피드백보다 본질적으로 더 강인하지도, 덜 강인하지도 않다”고 명시한다. [cse.lab.imtlucca](http://cse.lab.imtlucca.it/~bemporad/publications/papers/survey-robust-mpc.pdf)
- **제약 활성 시 안정성 이론 부족**: 제약이 걸린 영역에서는 선형 시스템 이론을 그대로 적용할 수 없고, 당시는 **일반적인 안정성·최적성 이론이 부재**. [scribd](https://www.scribd.com/document/661205288/Constrained-Model-Predictive-Control-Sta)
- **비선형 시스템에 대한 일반 이론 부재**: 개념적 확장은 가능하지만, 계산량·이론 모두 미성숙 상태.

***

# 3. “모델의 일반화 성능” 관점에서 본 논문의 시사점

여기서 “일반화 성능”을 **(1) 새로운 운전 조건·외란·초기 상태로의 확장성, (2) 모델 불확실성/학습 기반 모델의 out-of-distribution 동작에 대한 견고성**으로 해석할 수 있다.

## 3.1 1989 논문의 관점

논문은 명시적으로 “generalization”이라는 용어를 쓰지는 않지만, 다음과 같은 방향에서 일반화·강인성에 대해 논의한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6b7b8043-574d-4eca-a4a9-4a658d8528c5/ModelPredictiveControl_TheoryandPracticeaSurveyGarca1989.pdf)

1. **넓은 운전 영역**  
   - 실제 공정은 **“통합 최적화–제어–로지스틱스”** 구조 아래에서 넓은 운전 영역을 오간다.
   - 이때 고정 선형 설계가 아니라, 리시딩-호라이즌 구조를 가진 MPC가 **변하는 setpoint·제약에 자연스럽게 적응**할 수 있다고 강조. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6b7b8043-574d-4eca-a4a9-4a658d8528c5/ModelPredictiveControl_TheoryandPracticeaSurveyGarca1989.pdf)

2. **모델 불확실성 + 외란**  
   - 모델 불일치·외란을 명시적으로 “출력에 더해진 additive disturbance”로 모델링하고, 이를 피드백 루프를 통해 추정/보상하는 구조를 사용. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6b7b8043-574d-4eca-a4a9-4a658d8528c5/ModelPredictiveControl_TheoryandPracticeaSurveyGarca1989.pdf)
   - IMC 관점에서는 **모델 오차에 대해 민감도를 조절하는 필터**를 도입함으로써, 일반적인 “robustification vs. performance trade-off”를 명확히 해석한다. [scribd](https://www.scribd.com/document/649190281/1982-Garcia-C-Internal-Model-Control-IMC-a-Unifying-Review-and-Some-New-Results)

3. **적응 예측 제어(adaptive predictive control)**  
   - Peterka의 predictive controller, EPSAC, Generalized Predictive Control(GPC) 등, **온라인 파라미터 추정(예: RLS)을 결합한 SISO 적응 MPC**를 또 하나의 가지로 소개한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6b7b8043-574d-4eca-a4a9-4a658d8528c5/ModelPredictiveControl_TheoryandPracticeaSurveyGarca1989.pdf)
   - 다만 이 가지는 주로 **모델 적응**에 초점이 있고, **제약 처리와 강인성 이론은 미비**하다고 지적한다.

결론적으로, 이 논문은 **“구조적으로 일반화에 유리한 프레임(MPC/IMC) + 외란 추정/필터 + 적응적 식별의 결합”**이 향후 발전 방향임을 암시한다. [cse.lab.imtlucca](http://cse.lab.imtlucca.it/~bemporad/publications/papers/survey-robust-mpc.pdf)

***

## 3.2 IMC 필터·Q-파라미터를 통한 일반화·강인성 튜닝

IMC 구조에서 **폐루프 오류 전달함수**는

$$
e = (I - P Q)(d - r)
$$

이므로, 모델 불일치·외란에 대한 민감도는 순전히 \(Q\)의 특성에 의해 결정된다. [scribd](https://www.scribd.com/document/649190281/1982-Garcia-C-Internal-Model-Control-IMC-a-Unifying-Review-and-Some-New-Results)

- **대역폭(시간 상수)**를 가지는 저역통과 필터 \(F(z)\)를

$$
  Q(z) = Q_0(z) F(z)
  $$

  형태로 곱해 넣으면, 모델 불확실성에 대한 민감도는 ** $\(F(z)\)$ 의 컷오프 주파수**에 의해 직접 제어된다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6b7b8043-574d-4eca-a4a9-4a658d8528c5/ModelPredictiveControl_TheoryandPracticeaSurveyGarca1989.pdf)
- 이때 필터 차수·시간 상수는 **“성능–강인성–계산 안정성의 트레이드오프”**를 하나의 스칼라로 매개화한다.

이는 오늘날 **러닝 기반 MPC에서 “보수적인 튜브 폭”, “CVaR 수준”, “GP 분산 기반 constraint tightening”** 등을 조정해 **데이터 분포 밖(out-of-distribution)에서의 일반화**를 제어하는 것과 개념적으로 유사하다. [sciencedirect](https://www.sciencedirect.com/science/article/pii/S2405896322009168)

***

# 4. 2020년 이후 관련 최신 연구 비교 분석

1989년 논문이 제기한 주요 이슈(제약, 비선형성, 강인성, 계산량, 모델 불확실성)는 이후 30+년간 MPC 연구의 주된 축이 되었고, 2020년 이후에는 **데이터 기반·러닝 기반·분포견고(distributionally robust) MPC**로 크게 확장되었다.

아래에서는 “**모델의 일반화 성능**” 관점에서 네 가지 축으로 최근 연구를 정리한다.

## 4.1 데이터 기반·분포견고형 MPC (DR-MPC, Data-driven MPC)

### 4.1.1 Data-driven Distributionally Robust MPC

- Micheli et al., 2022: **Data-driven Distributionally Robust MPC for unknown LTI systems** [research-collection.ethz](https://www.research-collection.ethz.ch/bitstreams/fa7d4c8a-f781-4499-9041-f5ccdc17dbbc/download)
  - 불명확한 선형 시스템에 대해, 제한된 데이터로 추정한 동역학 + 외란 분포에 대해 **Wasserstein ball 기반 모호성 집합(ambiguity set)**을 정의.
  - 비용과 CVaR 제약에 대해 **최악의 분포에 대한 DR OCP**를 구성하고, 이를 tractable convex 문제로 변환. [research-collection.ethz](https://www.research-collection.ethz.ch/bitstreams/fa7d4c8a-f781-4499-9041-f5ccdc17dbbc/download)
  - **유한 샘플에서의 확률 보장** 및 **제약 위반 확률 상계**를 제공 → 통계적 일반화 보장.

- McAllister et al., 2023: **Distributionally Robust MPC** [arxiv](https://arxiv.org/pdf/2309.12758.pdf)
  - 비용 측만 분포견고화한 DR-MPC를 제안, **긴 시간 성능 보장(장기 평균 비용 상계)**를 제공.
  - DR-MPC가 **데이터 부족·분포 불일치 상황에서 SMPC보다 더 안정적인 성능을 낼 수 있음**을 예시. [arxiv](https://arxiv.org/pdf/2309.12758.pdf)

- Zhong et al., 2022: **DR-MPC for Nonlinear Systems** [sciencedirect](https://www.sciencedirect.com/science/article/pii/S2405896322009168)
  - 비선형 시스템에 대해, 데이터로부터 추정한 확률 불확실성에 대해 DR-CVaR 제약을 적용, 안정성·재귀적 타당성(recursive feasibility)을 증명.

이들은 1989 논문에서 제시된 “**모델 불확실성을 명시적으로 고려하는 견고 MPC 필요성**”을, 데이터 기반·통계적 관점에서 구체화하며, **out-of-sample 일반화 보장**을 수학적으로 제공한다. [cse.lab.imtlucca](http://cse.lab.imtlucca.it/~bemporad/publications/papers/survey-robust-mpc.pdf)

### 4.1.2 Data-driven MPC (DeePC 등)와 안정성

- 2024년 리뷰: **“An overview of systems-theoretic guarantees in data-driven MPC”** [arxiv](https://arxiv.org/pdf/2406.04130.pdf)
  - Willems의 Fundamental Lemma를 사용해, **모델을 명시적으로 식별하지 않고 데이터 블록(Hankel 행렬)만으로 예측·제어**하는 DeePC류 알고리즘을 총괄.
  - 다양한 변형에 대해 **안정성, 제약 만족, 강인성 보장 조건**을 비교·정리. [arxiv](https://arxiv.org/pdf/2406.04130.pdf)
- “Stability in data-driven MPC: an inherent robustness perspective” (2022) [arxiv](http://arxiv.org/pdf/2205.11859.pdf)
  - Data-driven MPC의 폐루프 안정성을, **nominal MPC의 “내재적 강인성(inherent robustness)”**와 데이터 노이즈에 대한 연속성을 결합해 증명.
  - 본질적으로 **Garcia–Morari가 IMC 구조에서 보여준 내재적 강인성**을, 데이터 기반 세팅으로 확장하는 관점이다. [scribd](https://www.scribd.com/document/649190281/1982-Garcia-C-Internal-Model-Control-IMC-a-Unifying-Review-and-Some-New-Results)

**요약**: DR-MPC·Data-driven MPC는 **1989년의 “모델 오차를 가진 MPC”**를, **통계적 일반화 이론·최악분포 최적화**로 업그레이드한 것으로 볼 수 있다.

***

## 4.2 러닝 기반·강화학습 결합 MPC (Learning-based MPC, Safe RL with MPC)

### 4.2.1 Learning for MPC with Stability & Safety Guarantees

- Gros & Zanon, “Learning for MPC with Stability & Safety Guarantees” (2020, 2022) [arxiv](http://arxiv.org/abs/2012.07369)
  - **Robust MPC**를 기반으로, **강화학습(RL) 또는 기타 학습 알고리즘이 MPC 파라미터(가중치, 예측 모형 등)를 온라인으로 갱신**하는 프레임워크를 제안.
  - **파라미터 업데이트 중에도 robust MPC의 안정성·제약 만족성이 유지되도록 하는 sufficient condition**을 제시.
  - 결론: **러닝을 사용하더라도, robust MPC 구조 위에서 파라미터 업데이트를 제한하면, 학습 전 과정에서 안전·안정성 보장이 가능**. [arxiv](https://arxiv.org/pdf/2012.07369.pdf)

이는 1989 논문의 “**IMC 구조 위에서 Q를 조정해도, Q가 안정이면 폐루프도 안정**”이라는 관찰을, **러닝으로 진화하는 Q(또는 MPC 파라미터)에 대해 formal하게 확장**한 셈이다. [scribd](https://www.scribd.com/document/649190281/1982-Garcia-C-Internal-Model-Control-IMC-a-Unifying-Review-and-Some-New-Results)

### 4.2.2 Deep / Neural MPC with 안정성 보장

- “Deep Model Predictive Control with Stability Guarantees” (2021) [arxiv](https://arxiv.org/pdf/2104.07171.pdf)
  - 비선형 시스템에 대해 **튜브 기반 MPC + 딥러닝 기반 모델/정책**을 결합.
  - 딥 모델이 학습되는 동안에도 **튜브 기반 robust MPC가 입력-상태 안정성(ISS)과 제약 만족을 보장**하도록 설계. [arxiv](https://arxiv.org/pdf/2104.07171.pdf)
- “Neural Lyapunov Differentiable Predictive Control (NLDPC)” (2022) [arxiv](http://arxiv.org/pdf/2205.10728.pdf)
  - 비용·제약·Lyapunov 조건까지 포함한 **완전 미분가능(differentiable) MPC 컴퓨테이셔널 그래프**를 구축하고, NN 정책을 학습.
  - 학습 과정에서 **Lyapunov 기반 안정성 제약을 직접 강제**해, 학습된 정책에 대한 안정성 보장을 제공. [arxiv](http://arxiv.org/pdf/2205.10728.pdf)

### 4.2.3 Safe RL with MPC-based Shields/Filters

- 2020–2023년 Safe RL 연구에서는, **MPC를 “safety filter” 또는 “shield”**로 사용하는 접근이 활발하다. [semanticscholar](https://www.semanticscholar.org/paper/2b9d75a87f70ab0d024d11d676af555074c4ed73)
  - RL 에이전트가 제안한 행동을 **MPC가 제약 만족/안정성을 보장하는 범위로 투영(project) 또는 수정**하는 구조.
  - 이는 본질적으로 1989 논문에서 “**제약을 직접 MPC 문제에 넣으면, 긴급 상황에서도 자동으로 안전한 조합을 찾아준다**”는 주장을 RL 맥락으로 확장한 것이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6b7b8043-574d-4eca-a4a9-4a658d8528c5/ModelPredictiveControl_TheoryandPracticeaSurveyGarca1989.pdf)

**요약**: 러닝 기반 MPC·Safe RL with MPC는 **IMC/MPC 구조의 “설계 시 제약·강인성 명시”라는 장점을, 데이터 기반 성능 향상과 결합**한 것으로, “일반화 성능”을 **(데이터 분포 변화, 환경 변화, 정책 변화)에 대해 안정적·안전하게 유지**하려는 시도이다.

***

## 4.3 Neural / Explicit MPC: 빠른 최적화와 정책 근사

1989 논문의 큰 한계는 **실시간 최적화 비용**이었다. 2020년 이후에는 이를 해결하기 위해 **정책 근사(explicit MPC) + deep learning**이 크게 발전했다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6b7b8043-574d-4eca-a4a9-4a658d8528c5/ModelPredictiveControl_TheoryandPracticeaSurveyGarca1989.pdf)

- Alessio & Bemporad의 **Explicit MPC Survey** 이후, [cse.lab.imtlucca](http://cse.lab.imtlucca.it/~bemporad/publications/papers/nmpc08-survey-explicit.pdf)
- 2021년 **“Learning explicit predictive controllers”**: [arxiv](https://arxiv.org/pdf/2108.08412.pdf)
  - 노이즈 없는 경우, 데이터 기반으로 학습한 piecewise affine 정책이 **원래 MPC 해와 정확히 일치**함을 증명.
  - 노이즈가 있어도, 샘플 수가 증가하면 **MPC 정책에 점근적으로 수렴**하는 일반화 결과를 제시. [arxiv](https://arxiv.org/pdf/2108.08412.pdf)
- 2024년 **“Neural Networks for Fast Optimisation in MPC: A Review”**: [arxiv](https://arxiv.org/pdf/2309.02668.pdf)
  - 선형/비선형 MPC 최적화 해(또는 value function)를 **딥 네트워크로 근사**하는 다양한 기법을 종합.
  - 주요 이슈: **근사 오차가 제약 위반·안정성 저하로 이어지지 않도록 하는 일반화 보장**을 어떻게 줄 것인가. [arxiv](https://arxiv.org/pdf/2309.02668.pdf)

이 라인업은, 1989년 논문이 강조한 **“Q-파라미터화된 안정한 제어기 집합 위에서의 최적화”** 아이디어를, **신경망 파라미터 공간에서의 함수 근사·정규화·일반화 분석**으로 치환한 것으로 볼 수 있다.

***

## 4.4 NMPC 이론과 안정성 분석의 성숙

- Grüne (2010): **시간 지연이 있는 비선형 MPC의 안정성 분석** [arxiv](https://www.arxiv.org/pdf/1006.2529v1.pdf)
  - 가변 호라이즌, 점근·유한 시간 도달성 가정 하에서, 제약 없는 비선형 MPC의 안정성 조건을 제시.
- Köhler et al. (2019): **Quasi-infinite horizon NMPC with tracking** [arxiv](https://arxiv.org/pdf/1909.12765.pdf)
  - 일반 참조 궤적 추적을 위한 NMPC에서, **적절한 터미널 비용·터미널 피드백 설계로 안정성을 보장**하는 설계 절차 제안.
- Findeisen (2006): **NMPC 모노그래프** [d-nb](https://d-nb.info/979741750/34)
  - 계산 지연, 샘플링, 출력 피드백 등 실무 이슈까지 포함한 NMPC 이론 전체를 체계화.

이들 연구는 1989년 논문이 “표면만 긁었다”고 평가한 비선형/제약 시스템에 대한 이론을 20년 이상에 걸쳐 크게 확장했고, 최근 learning-based NMPC 연구는 이 토대 위에서 진행된다. [arxiv](http://arxiv.org/pdf/2211.13829.pdf)

***

## 4.5 간단 비교 표

| 축 | 1989 Garcia et al. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6b7b8043-574d-4eca-a4a9-4a658d8528c5/ModelPredictiveControl_TheoryandPracticeaSurveyGarca1989.pdf) | 2020년 이후 대표 연구 |
|----|---------------------------|------------------------|
| 제약 처리 | QP/LP 기반 MPC (DMC, MAC, QDMC); 일반 이론은 부분적 | DR-MPC, 튜브 기반 Robust MPC + DR-Learning [research-collection.ethz](https://www.research-collection.ethz.ch/bitstreams/fa7d4c8a-f781-4499-9041-f5ccdc17dbbc/download) |
| 모델 불확실성 | IMC 필터, 경험적 튜닝; robust MPC 초창기 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6b7b8043-574d-4eca-a4a9-4a658d8528c5/ModelPredictiveControl_TheoryandPracticeaSurveyGarca1989.pdf) | Wasserstein DR, GP/ensemble 기반 불확실성 모델링 및 확률 보장 [research-collection.ethz](https://www.research-collection.ethz.ch/bitstreams/fa7d4c8a-f781-4499-9041-f5ccdc17dbbc/download) |
| 비선형 시스템 | 개념적 확장·사례 위주, 일반 이론 미성숙 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6b7b8043-574d-4eca-a4a9-4a658d8528c5/ModelPredictiveControl_TheoryandPracticeaSurveyGarca1989.pdf) | NMPC 안정성 이론·터미널 조건 설계, learning-enhanced NMPC [d-nb](https://d-nb.info/979741750/34) |
| 계산 비용 | 온라인 QP/LP 연산이 병목 | explicit/NN-MPC, differentiable MPC, value function approximation [cse.lab.imtlucca](http://cse.lab.imtlucca.it/~bemporad/publications/papers/nmpc08-survey-explicit.pdf) |
| 데이터/학습 | 적응 MPC(RLS 기반 파라미터 추정) 언급 | RL + MPC, safe learning for MPC, data-driven MPC, DeepMPC [arxiv](http://arxiv.org/abs/2012.07369) |

***

# 5. 이 논문이 이후 연구에 미친 영향과 앞으로의 연구 시 고려점

## 5.1 영향 요약

1989년 서베이는 다음 네 가지 측면에서 **지금도 유효한 기준점**이다.

1. **문제 정식화 표준화**
   - “동적 비용 + 동적 불등식 제약 + 명시적 모델 + 리시딩 호라이즌”이라는 MPC의 전형적 구조를 명확히 정의했다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6b7b8043-574d-4eca-a4a9-4a658d8528c5/ModelPredictiveControl_TheoryandPracticeaSurveyGarca1989.pdf)
   - 이후 robust MPC, NMPC, DR-MPC, learning-based MPC는 모두 이 포맷을 변형·확장한 것이라 할 수 있다. [arxiv](https://arxiv.org/pdf/2012.07369.pdf)

2. **IMC/Q-파라미터화 관점 정립**
   - MPC/IMC 구조와 고전 피드백의 동치를 명시함으로써, **안정한 Q 집합 위에서의 최적화**라는 관점을 제공했다. [scribd](https://www.scribd.com/document/649190281/1982-Garcia-C-Internal-Model-Control-IMC-a-Unifying-Review-and-Some-New-Results)
   - 이는 오늘날 **정책 공간(딥 네트워크) 상의 안전 서브셋에서만 학습**하는 safe RL/MPC 설계와 개념적으로 동일한 철학을 공유한다. [arxiv](http://arxiv.org/abs/2012.07369)

3. **제약 기반 설계 철학 확립**
   - “제약은 설계 단계에서 명시적으로 넣어야 하며, ad-hoc 오버라이드/스플릿 레인지 로직은 유지보수 비용이 크고 성능을 제한한다”는 메시지는, 현대의 **safety filter, shielded RL, safe exploration** 연구와 직접 연결된다. [proceedings.neurips](https://proceedings.neurips.cc/paper/2021/file/73b277c11266681122132d024f53a75b-Paper.pdf)

4. **연구 과제 제시**
   - 논문 말미의 오픈 이슈(대규모 QP/LP, 모델 불확실성 하 제약 만족, 비선형 시스템·제약 포함 안정성 이론 등)는, 이후 30년간 robust MPC, constrained MPC 이론, NMPC 이론, learning-based MPC의 주요 연구 주제가 되었다. [arxiv](http://arxiv.org/pdf/2211.13829.pdf)

***

## 5.2 앞으로 연구 시 고려할 점 (특히 “일반화 성능” 관점)

연구자로서 이 논문을 기반으로 후속 연구를 설계할 때, 다음과 같은 포인트를 고려할 수 있다.

1. **모델-기반 vs 데이터-기반의 통합**
   - MPC의 강점은 여전히 **명시적(또는 암시적) 모델과 제약을 포함한 최적화 구조**에 있다. [sites.engineering.ucsb](https://sites.engineering.ucsb.edu/~jbraw/mpc/MPC-book-2nd-edition-1st-printing.pdf)
   - 데이터 기반 모델(딥 NN, GP, NARX 등)을 도입하더라도,
     - (i) **모델 불확실성(분산, 앙상블 분산, Wasserstein ball 등)**을 정량화하고, [arxiv](https://www.arxiv.org/pdf/2502.05448.pdf)
     - (ii) 이를 **제약 긴축(constraint tightening), 튜브 설계, DR-CVaR 제약**에 반영해,
     - (iii) **out-of-sample 안정성·제약 만족 보장**을 이론적으로 제공하는 것이 “진짜 의미의 일반화 성능 향상”이다.

2. **정책 근사/러닝의 신뢰도 평가**
   - explicit MPC나 NN-MPC를 사용할 경우, **근사된 정책이 원래 MPC 정책과 얼마나 유사한지, 근사 오차의 worst-case가 제약 위반 가능성에 어떤 영향을 주는지**를 분석할 필요가 있다. [arxiv](https://arxiv.org/pdf/2309.02668.pdf)
   - Lyapunov 기반 제약(NLDPC)이나 튜브 기반 안정성 조건(Deep MPC)처럼, **정책 학습 중에도 폐루프 안정성을 유지하는 구조**가 중요하다. [arxiv](https://arxiv.org/pdf/2104.07171.pdf)

3. **데이터 효율과 안전한 탐색**
   - robust/DR-MPC는 이론적으로 강력하지만, **보수성(conservatism)**과 데이터 요구량이 크다는 문제가 있다. [sciencedirect](https://www.sciencedirect.com/science/article/pii/S2405896322009168)
   - Safe RL + MPC shield처럼, **MPC를 안전한 탐색의 커널로 사용하여 데이터 효율과 안전성을 동시에 확보**하는 연구 방향이 유망하다. [sciencedirect](https://www.sciencedirect.com/science/article/pii/S0098135425005241)

4. **복잡 시스템(고차원, 네트워크, 비선형)으로의 일반화**
   - 큰 네트워크 시스템, 강화학습이 필요한 고차원 제어에서, **완전한 모델링이 불가능**한 경우가 많다.
   - 이때 **부분적 모델 + 데이터 기반 잔차 모델 + DR-MPC/튜브 MPC**를 결합하는 **하이브리드 구조(MDR-DeePC 등)**가 유망하며, [arxiv](https://arxiv.org/pdf/2506.19744.pdf)
   - Garcia 등이 강조했던 “모든 계층(측정–제어–최적화–로지스틱스)의 통합”을, **학습·추론·제어의 통합**으로 확장하는 연구가 필요하다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6b7b8043-574d-4eca-a4a9-4a658d8528c5/ModelPredictiveControl_TheoryandPracticeaSurveyGarca1989.pdf)

5. **이론–실무 간 간극을 줄이는 평가 프레임**
   - 2020년대에는 MPC 성능 평가에 대한 별도 서베이도 등장했는데, 이는 1989년 이후 실무–이론 간 간극을 줄이려는 노력의 연장이다. [mdpi](https://www.mdpi.com/1999-4893/13/4/97/pdf)
   - 새로운 learning-based MPC를 제안할 때, **고전 MPC/IMC와 동등한 또는 우월한 성능·견고성·계산 효율을 보여주는 체계적 벤치마크**가 중요하다.

***

## 5.3 정리

요약하면, Garcia–Prett–Morari(1989)는

- **MPC = 명시적 모델 + 제약 포함 동적 최적화 + IMC 구조**라는 표준 프레임을 정립했고, [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6b7b8043-574d-4eca-a4a9-4a658d8528c5/ModelPredictiveControl_TheoryandPracticeaSurveyGarca1989.pdf)
- **일반화 성능(넓은 운전 영역, 제약, 모델 불확실성)에 대응하기 위한 구조적 장점**(IMC, Q-파라미터화, 외란 추정)을 분명히 했으며, [scribd](https://www.scribd.com/document/649190281/1982-Garcia-C-Internal-Model-Control-IMC-a-Unifying-Review-and-Some-New-Results)
- 이 프레임이 오늘날 **robust/DR-MPC, data-driven MPC, learning-based MPC, safe RL with MPC**까지 자연스럽게 확장되도록 방향을 제시했다. [arxiv](http://arxiv.org/pdf/2211.13829.pdf)

향후 연구에서는, 이 프레임을 유지하면서

- 데이터로부터 학습된 모델·정책이 **어디까지 일반화 가능한지(확률적·분포견고 관점)**,
- 그 과정에서 **안정성·제약 만족 보장을 어떻게 유지할 것인지**를
- **IMC/MPC 구조와 결합된 이론적 분석 + 실무 벤치마크**로 동시에 다루는 것이 핵심 과제가 될 것이다.
