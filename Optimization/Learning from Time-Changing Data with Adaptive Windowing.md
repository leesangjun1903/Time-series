# Learning from Time-Changing Data with Adaptive Windowing

1. 핵심 주장과 주요 기여 (간결 요약)
-----------------------------------

- **핵심 주장**: 고정 길이 슬라이딩 윈도우 대신, **데이터 자체가 보여주는 분포 변화 속도에 맞춰 윈도우 길이를 자동 조절하는 ADWIN(ADaptive WINdowing)**을 제안하고, 이 방법이 **거짓 양성/거짓 음성 확률에 대한 이론적 보장**을 가지며 특정 변화 구조에서는 **통계적으로 최적인 윈도우 길이로 자동 수렴**함을 보인다.[^1_1]
- **주요 기여**
    - (1) **통계적 가설검정 기반의 변화 감지 윈도우 알고리즘 ADWIN** 제안 및 **오차율 상한(δ)에 대한 이론적 보장** 제시.
    - (2) 데이터 스트림 알고리즘 기법(지수 히스토그램)을 이용해 **시간·메모리 효율적 버전 ADWIN2** 설계: 길이 $W$ 윈도우를 $O(\log W)$ 메모리와 업데이트 시간으로 유지.
    - (3) ADWIN2를 **외부(change detector)·내부(통계 추정기)** 두 방식으로 나이브 베이즈와 k-means에 통합하여, **고정/휴리스틱 윈도우보다 예측 성능이 일관되게 우수함**을 실험으로 보임.
    - (4) 이 모든 것을 **“윈도우 크기, decay rate 등을 사용자가 미리 정하지 않아도 되는, 파라미터 프리(adaptive) 개념 드리프트 대응 프레임워크”**로 제시.

***

2. 논문이 해결하고자 하는 문제
------------------------------

### 2.1 문제 설정

- **환경**: 무한하거나 매우 긴 데이터 스트림 $x_1, x_2, \dots, x_t, \dots$, 각 시점 $t$에서 $x_t \sim D_t$, 분포 $D_t$는 시간에 따라 바뀔 수 있음. 기대값 $\mu_t$, 분산 $\sigma_t^2$.
- **전제**: 데이터는 $[0, 1]$ 구간 (또는 선형 변환으로 제한 가능한 구간) 안에 있음.
- **과제**

1. **변화(개념/분포 드리프트) 발생 시점 탐지**
2. **과거 어떤 예제를 버리고 어떤 것을 유지할지 결정** (슬라이딩 윈도우 관리, suff. statistics 업데이트)
3. **유의미한 변화가 감지되었을 때 모델을 어떻게 재학습/수정할지**

기존 접근:

- **고정 윈도우**: 윈도우 크기 $W$를 사전에 고정.
    - $W$가 작으면: 최신 분포는 잘 추적하지만, 안정기에도 데이터가 적어 분산(variance)↑ → 과적응 위험↓ / 분산↑.
    - $W$가 크면: 안정기에는 좋지만, 변화가 오면 **오래된 데이터가 모델을 끌고 내려서 반응이 느림**.
- **시간 감쇠(exponential decay)**: 감쇠율 $\lambda$ 역시 사실상 “숨은 윈도우 크기”를 정하는 것과 같아, **알 수 없는 변화 속도에 맞추기 어렵다**.
- **변화 감지 + 재학습 휴리스틱**: 많은 방법이 윈도우 길이/감지 기준을 휴리스틱으로 잡아 이론적 보장이 없다.

이 논문이 노리는 핵심은:

> “**변화 속도의 시간 척도(time scale)에 대한 사전 지식 없이도**, 데이터 스트림의 통계적 특성에 따라 **윈도우 크기를 자동으로 조절하고**, 그 과정에 대해 **이론적 오류 보장**을 제공하는 일반적인 도구를 만들자.”

***

3. 제안 방법: ADWIN 및 ADWIN2 (식 포함)
--------------------------------------

### 3.1 통계적 세팅

현재 시점에서 유지하는 윈도우를 $W$라 하고, 길이를 $n = |W|$라고 하자.
윈도우의 관측 평균과(데이터 기준) 기대값(분포 기준)을 각각

- 관측 평균: $\hat{\mu}_W$
- 기대 평균: $\mu_W$

라고 표기한다. $W$를 어떤 지점에서 두 부분으로 나누어

- $W = W_0 \cdot W_1$
- 길이: $n_0 = |W_0|, n_1 = |W_1|$, $n = n_0 + n_1$
- 각 부분의 관측 평균: $\hat{\mu}\_{W_0}, \hat{\mu}_{W_1}$

이라 할 때, **“두 부분이 같은 분포에서 왔는가?”**라는 귀무가설을 가설검정으로 테스트한다.

이때, 두 윈도우 길이의 **조화평균** $m$을

$$
m = \frac{1}{\frac{1}{n_0} + \frac{1}{n_1}}
$$

로 정의한다. (길이가 비슷할수록 $m$이 커져 검정력이 증가.)

### 3.2 기본 임계값(εcut) – Hoeffding 기반

ADWIN은 전체 윈도우 $W$의 **모든 분할 지점**을 후보 컷포인트로 보고, 각 분할에 대해 두 평균의 차이가 충분히 큰지 검사한다.

- 전체 윈도우 길이가 $n$일 때, 분할 후보는 최대 $n$개이므로, **다중 가설검정** 문제를 피하기 위해
    - 글로벌 신뢰도 파라미터 $\delta$를
    - 각 분할에 대한 유효 파라미터 $\delta' = \delta / n$으로 나눈다.
- Hoeffding 부등식을 이용하면,

$$
\Pr\left( |\hat{\mu}_{W_0} - \hat{\mu}_{W_1}| \ge \epsilon_{\text{cut}} \right) \le \frac{\delta}{n}
$$

를 만족하는 충분조건으로

```math
\epsilon_{\text{cut}}
=
\sqrt{
  \frac{1}{2m} \ln \frac{4}{\delta'}
}
=
\sqrt{
  \frac{1}{2m} \ln \frac{4n}{\delta}
}
```

를 얻는다.[^1_1]

**알고리즘 ADWIN (개념)**

1. 초기 윈도우 $W$를 비워놓고 시작.
2. 새 값 $x_t$가 도착할 때마다 맨 앞에 추가.
3. **반복**:
    - 모든 가능 분할 $W = W_0 \cdot W_1$에 대해
        - $|\hat{\mu}\_{W_0} - \hat{\mu}\_{W_1}| > \epsilon_{\text{cut}}$ 를 만족하는 분할이 존재하면,
            - **분포가 달라졌다고 판단하고, 오래된 부분 $W_0$를 버리고 윈도우를 $W_1$로 줄인다.**
    - 더 이상 조건을 만족하는 분할이 없으면 종료.
4. 현재 윈도우 평균 $\hat{\mu}_W$를 스트림의 “현재 평균” 추정값으로 출력.

요약하면:

> “윈도우 내에서 **‘충분히 큰’ 두 부분의 평균 차이가 통계적으로 유의미해지는 시점에, 과거 부분을 잘라낸다**.”

### 3.3 개선된 임계값 – 분산을 이용한 Bernstein/정규 근사

식 (1)은 분산 상한 $\sigma^2 \le 1/4$를 항상 가정해 **보수적**이다. 실제 분산 $\sigma_W^2$를 추정해 더 날카로운 경계를 얻기 위해, 논문은 다음과 같은 근사식을 사용한다:

```math
\epsilon_{\text{cut}}
=
\sqrt{
  \frac{2}{m} \sigma_W^2 \ln \frac{2}{\delta'}
}
+
\frac{2}{3m} \ln \frac{2}{\delta'}
```

여기서 $\sigma_W^2$는 윈도우 내 관측 분산이다. 첫 항은 사실상 “ $k$ 표준편차 이상” 기준에 해당하고, 둘째 항은 작은 샘플에서도 Hoeffding류 보장을 유지하기 위한 보정항이다.[^1_1]

실제 구현에서는

- 다중검정에 대해 $\delta' = \delta / \ln n$ 정도로 더 완화해도 충분하다고 주장하며,
- ADWIN2에서는 어차피 $O(\log n)$개 분할만 검사하므로 이 선택이 정당화된다고 설명한다.


### 3.4 이론적 보장 (Theorem 3.1 요약)

논문이 제시하는 핵심 정리는 다음 두 가지다.[^1_1]

1. **거짓 양성(변화 없음인데 잘라내는 경우) 상한**
    - 만약 윈도우 $W$ 내에서 $\mu_t$가 **모든 t에 대해 동일**하다면,
        - ADWIN이 이 시점에서 윈도우를 줄일(=변화를 선언할) 확률은 **최대 $\delta$**.
2. **거짓 음성(실제 변화가 있는데 못 보는 경우) 상한**
    - 만약 어떤 분할 $W = W_0 \cdot W_1$에 대해

$$
|\mu_{W_0} - \mu_{W_1}| > 2 \epsilon_{\text{cut}}
$$

를 만족하면,
        - ADWIN은 **확률 최소 $1 - \delta$**로 윈도우를 $W_1$ (또는 더 짧은) 쪽으로 축소한다.

즉, **고정된 신뢰도 파라미터 $\delta$**에 대해 “잘못된 변화 신호”와 “변화 놓침” 확률을 각각 제어할 수 있다.

### 3.5 갑작스러운 변화 vs 점진적 변화에서의 동작

- **갑작스러운 변화 (jump)**: 한동안 $\mu_t = \mu$였다가, 어느 시점에 $\mu_t = \mu + \epsilon$으로 급변.
    - 윈도우 크기는 대략 $O(\mu \log(1/\delta) / \epsilon^2)$ 정도의 과거를 포함한 뒤,
    - 이 시점에서 평균 차이가 임계값을 넘고, 과거 부분을 잘라낸다.
    - 이후 변화가 없다면 윈도우 길이는 다시 선형적으로 증가.
- **점진적 변화 (slope)**: 한동안 $\mu_t = \mu$, 이후 일정 기울기 $\alpha$로 선형 변화.
    - 윈도우 내의 평균 차이가 유의미해지는 시점에서 **일정한 길이 $n_1 \approx O((\mu \log(1/\delta)/\alpha^2)^{1/3})$** 근처로 자동 수렴.
    - 논문은 이 $O(\alpha^{-2/3})$ 스케일이 **분산과 드리프트 오류(과거 데이터 때문에 발생하는 편향)의 합을 최소화하는 최적 윈도우 길이**임을 보인다.

이는 **“윈도우 길이와 반응 속도 간의 최적 trade-off를 자동으로 구현한다”**는 점에서, 모델 일반화 관점에서 매우 중요한 포인트다 (후술).

### 3.6 ADWIN2: 시간·메모리 효율적 구조

기본 ADWIN은 매 스텝마다 $O(n)$ 분할을 검사해야 해 비효율적이다. 이를 해결하기 위해, 논문은 데이터 스트림 알고리즘에서 사용하는 **Exponential Histogram**류 구조를 변형한 **ADWIN2**를 제안한다.[^1_1]

핵심 아이디어:

- 윈도우 전체를 개별 포인트가 아닌, **크기가 1, 2, 4, 8, … (2의 거듭제곱)**인 **버킷(bucket)**들의 리스트로 요약.
- 각 버킷은
    - **capacity**: 포함하는 포인트 수 ($2^i$)
    - **content**: 포인트들의 합(또는 1의 개수)
를 저장.
- 각 크기의 버킷을 **최대 $M$개까지만 유지**하고, $M+1$개가 되면 **가장 오래된 두 개를 merge**하여 상위 크기 버킷으로 올리는 방식.
- 이렇게 하면 윈도우 길이 $W$에 대해

$$
\text{메모리} = O(M \log (W/M))
$$

의 버킷만 유지하면 되며, 각 버킷 경계가 **검사할 분할 후보**가 된다.
- 분할 후보 길이는 대략 $1, (1+\frac{1}{M}), (1+\frac{1}{M})^2, \dots$ 식의 **기하급수적 증가**를 하므로, 전체 길이에 대해 **스케일-프리(scale-free)**하게 다양한 시간 척도를 커버.

정리하면:

- **메모리 사용량**: $O(M \log(W/M))$
- **새 포인트 처리 시간**:
    - 평균 $O(1)$, 최악 $O(\log W)$
- **검사할 분할 수**: $O(\log W)$ (각 버킷 경계)

즉 ADWIN2는

> “사실상 모든 시간 척도에 대해 적당한 분할을 검사하면서도, 전체 복잡도는 $O(\log W)$ 수준에 머무는 adaptive window change detector”

로 볼 수 있다.

***

4. 모델 구조, 실험 결과, 성능 향상 및 한계
---------------------------------------

### 4.1 나이브 베이즈(NB)에서의 통합 구조

논문은 두 가지 방식으로 ADWIN2를 NB에 결합한다.[^1_1]

1. **외부 change detector로 사용**
    - 분류기의 **에러율 시퀀스** (정답=1, 오답=0)를 ADWIN2에 흘려보내고,
    - ADWIN2가 변화(에러율 유의한 증가)를 감지하면,
        - 현재 윈도우 내 예제들(혹은 최근 예제들)을 사용해 NB 모델을 **다시 학습(rebuild)**.
    - 장점: 기존 어떤 모델에도 비교적 쉽게 붙일 수 있는 **모듈형 구조**.
    - 단점: 에러율이라는 **글로벌 스칼라 지표만 본다**는 한계 → 어디서, 어떤 feature에 의해 변화가 생겼는지는 모름.
2. **내부 통계 추정기로 사용 (추천 방식)**
    - NB는 각 조건부확률 $P(x_i = v_j, C = c)$, $P(C=c)$를 **카운트 $N_{i,j,c}, N_c$**를 통해 추정.
    - 각 카운트를 위해 **개별 ADWIN2 인스턴스 $A_{i,j,c}$** 를 하나씩 할당:
        - 새 예제 $(x, c)$가 오면, 해당되는 카운트에 대해
            - 조건 만족 시 1, 아니면 0을 $A_{i,j,c}$에 입력.
        - 예측 시에는 $A_{i,j,c}$가 유지하는 윈도우 내 평균을 카운트/확률 추정에 사용.
    - 각 $A_{i,j,c}$는 **자기 윈도우 길이를 독립적으로 조절**하므로,
        - 어떤 feature-value 조합은 **느리게 변화**해 긴 윈도우를 가져가고,
        - 어떤 조합은 **빠르게 변화**해 짧은 윈도우로 적응.

이 방식은 곧바로 **일반화 성능 개선 가능성**과 연결된다 (5절에서 자세히).

### 4.2 NB 실험 결과 요약

- **합성 데이터 (회전하는 하이퍼플레인)** 실험에서,
    - 완전히 최신 분포만을 학습한 **static NB 베이스라인** 대비,
    - 여러 시간-변화 관리 전략 (고정 윈도우, flushing 윈도우, Gama et al.의 DDM류 방법, ADWIN2 외부 change detector, ADWIN2 내부 통계 추정기)를 비교.
- 결과:[^1_1]
    - **ADWIN2를 카운트에 직접 넣은 incremental NB**가
        - static NB 정확도의 **~99% 이상**을 유지하면서,
        - 고정 윈도우 및 change detector 기반 NB보다 항상 우수 또는 동일.
    - ADWIN2를 **외부 change detector**로 쓴 경우도
        - 기존 change detector (Gama의 방법)보다 더 좋은 성능을 보이는 경우가 많음.
- **실제 데이터 (전기 시장 가격 ELEC2)**에서도
    - “지난 48개 샘플로 static NB를 재학습한 모델”을 베이스라인으로 두고,
    - ADWIN2를 내부 카운트에 쓰는 NB가
        - 대부분의 다른 전략보다 **예측 및 분류 성능이 우월**.


### 4.3 k-means에서의 통합 구조

- 각 클러스터 중심의 각 좌표마다 **하나의 ADWIN2 인스턴스**를 둔다.[^1_1]
    - 새 포인트가 오면 가장 가까운 클러스터에 할당하고,
    - 해당 클러스터 좌표에 대해 ADWIN2를 업데이트 (위치는 평균으로 추정).
- 일정 조건(ADWIN2 윈도우 축소, 평균 거리 변화율 등)이 만족되면,
    - **글로벌 재할당(recomputation)**을 수행해 클러스터 배치를 업데이트.
- 합성 데이터(이동하는 k-가우시안)의 실험 결과:
    - ADWIN2 기반 k-means가
        - 다양한 분산 및 이동속도 조건에서,
        - 고정 윈도우 기반 incremental k-means보다 **클러스터 중심에 더 근접한 추정**을 지속적으로 유지.


### 4.4 성능 향상의 본질

1. **“시간 척도(time scale)” 선택 문제를 제거**
    - 사용자는 더 이상 “윈도우 길이 = 1000” 같은 숫자를 찍어 넣지 않아도 되고,
    - 변화가 느릴 때는 긴 히스토리, 빠를 때는 짧은 히스토리를 자동으로 선택.
2. **에러율·추정 오차의 상한을 이론적으로 제어**
    - $\delta$ 하나로 거짓 양·음성 확률을 제어하며,
    - 어떤 조건에서는 통계적으로 최적인 윈도우 규모로 수렴.
3. **희귀 이벤트(small $p$)에 대한 좋은 추정**
    - 합성 실험에서 **희귀 확률(예: 1/4096 등)을 추정할 때, ADWIN2가 대부분의 고정 윈도우를 능가**하는 것을 보임.[^1_1]
    - 희귀 개념(rare concepts)에서 일반화 성능을 확보하는 데 중요.

### 4.5 한계와 논문에서 인정하는 제약

- **정량적인 drift 설명/위치(localization)는 제공하지 않음**
    - “어디에서 왜 변화가 생겼는지”를 설명하기보다,
    - “윈도우를 줄였다/변화가 있다”는 시그널만 제공.
- **레이블에 대한 의존성 (error-rate 기반 사용 시)**
    - 에러율을 모니터링할 때는 **즉시 레이블 접근이 필요**하며,
    - 이는 많은 실제 시스템(레이블 지연, 부분 라벨링)에선 비현실적.
- **고차원·구조적 데이터에 대한 직접 지원 없음**
    - 스칼라 혹은 간단한 통계를 다루므로,
    - 딥러닝 representation space 등에서는 별도의 embedding/요약을 거쳐야 함.
- **파라미터 $\delta$, 버킷 파라미터 $M$** 선택 문제
    - $\delta$는 이론적으로 의미 있지만, 실제 환경에서 최적값 선택은 여전히 경험적 튜닝이 필요.
    - $M$은 성능 vs 메모리 trade-off를 결정하지만 구체적인 가이드라인은 제한적.

***

5. ADWIN 관점에서 본 “모델 일반화 성능 향상 가능성”
-----------------------------------------------

이 논문의 기여를 **일반화 성능** 측면에서 정리하면 다음과 같다.

### 5.1 시간 축에서의 bias–variance trade-off 최적화

- 고정 윈도우:
    - $W$가 크면 → **분산↓, drift에 대한 bias↑**
    - $W$가 작으면 → **분산↑, drift bias↓**
- ADWIN은 이 trade-off를
    - **역학적으로(adaptively)**, 그리고 어떤 경우에는 **이론적으로 최적으로** 조정:
        - 갑작스러운 변화에서는 작은 창으로 줄여 drift bias를 줄이고,
        - 안정기에는 창을 크게 늘려 분산을 낮춤.
- 이는 “일반화 오차 = 분산 + (drift로 인한) 편향”을 최소화하는 시간-가변 학습 셋을 자동 구성하는 것으로 해석할 수 있다.


### 5.2 feature·클러스터 단위의 비동기적 적응

- NB 내부 통합처럼, 각각의 조건부확률 $P(x_i|C)$마다 ADWIN2를 붙이면,
    - 어떤 feature/value 조합은 **안정적**이라 긴 윈도우를 유지,
    - 어떤 조합은 **빠르게 변함** → 짧은 윈도우로 빠르게 업데이트.
- 이는 “모든 파라미터에 동일한 decay/윈도우를 적용”하는 것보다
    - **국소적인 개념 드리프트(local drift)**에 잘 반응하며,
    - **불필요한 재학습과 과도한 forgetting을 줄여** 전체 일반화 성능을 높이는 효과가 있다.


### 5.3 희귀 이벤트·불균형 데이터에서의 일반화

- 희귀 클래스/이벤트의 확률을 추정할 때,
    - 고정 작은 윈도우는 샘플 수가 너무 적어 추정 분산이 폭발,
    - 고정 큰 윈도우는 오래된 데이터까지 포함해 현재 분포와 mismatch.
- ADWIN은 관측 빈도에 따라 윈도우 길이를 조절하므로,
    - 희귀이면서도 **최근에 자주 등장하기 시작한 이벤트**에 대해서는
        - 윈도우를 적절히 늘려 안정적인 추정을 제공할 수 있고,[^1_1]
    - 반대로 다시 사라져 가는 이벤트에 대해서는 윈도우를 줄여 현재 분포에 맞추는 식으로 적응.


### 5.4 라벨드/언라벨드 정보의 활용

- 에러율 기반 사용 시에는 레이블이 필요하지만,
- 실제로는 **입력 통계나 embedding 통계**에 대해 ADWIN을 적용해
    - **레이블이 없어도 분포 이동을 추적**하는 것이 가능.
- 이 아이디어는 이후 언라벨드 drift detection 연구(예: 드롭아웃 불확실성 기반 UDD, 드리프트렌즈 등)에서 **기본 building block으로 재활용**된다.[^1_2][^1_3][^1_4][^1_5]

***

6. 2020년 이후 관련 최신 연구와 비교 분석
-----------------------------------------

여기서는 **adaptive window / ADWIN 계열 아이디어와 개념 드리프트**를 다룬 최근(2020년 이후) 대표 연구를 중심으로, 어떤 방향으로 확장·변형되었는지 비교한다.

### 6.1 대표 최신 논문 목록 (요약형 정리)

(1) **Uncertainty Drift Detection (UDD)** – Baier et al., 2022[^1_3][^1_4]

- **아이디어**
    - 딥 뉴럴넷 + Monte Carlo Dropout으로 **예측 불확실성(variance)**을 추정.
    - 이 불확실성 시퀀스를 ADWIN에 흘려보내 **레이블 없이** drift를 감지.
- **ADWIN과의 관계**
    - ADWIN을 **변화 감지 엔진**으로 그대로 사용하되,
    - 입력으로 **에러(0/1)** 대신 **모델 불확실성**을 사용.
- **장점/차별점**
    - 실제 운영 환경에서 레이블이 부족한 상황에서도 drift를 감지해 **불필요한 재학습 횟수 감소 + 필요한 시점에만 재학습** → 일반화 유지에 유리.

(2) **ADDM (Autoregressive-based Drift Detection Method)** – Mayaki et al., 2022[^1_6]

- **아이디어**
    - 자기회귀(AR) 기반 threshold 모델로 에러율 시퀀스를 모델링하고,
**자기흥분(threshold) AR 모형의 변화**를 이용해 drift를 검정.
- **ADWIN과 비교**
    - ADWIN, DDM, KSWIN 등 인기 있는 검출기들과 비교했을 때,
        - 여러 합성·실제 데이터에서 **검출 정확도와 적응 성능에서 우수**하다고 보고.
    - 동시에 **이론적 보장 역시 제공**한다는 점에서, ADWIN의 “이론적 rigor”를 잇는 계열.

(3) **Adaptive windowing based recurrent neural network for drift adaption in non-stationary environment** – Suryawanshi et al., 2022[^1_7]

- **아이디어**
    - GRU 기반 RNN에 **adaptive windowing**을 결합하여 비정상 데이터 스트림에서 drift를 처리.
    - 예측이 잘 되면 윈도우를 늘리고, drift가 감지되면 윈도우를 줄이는 방식.
- **ADWIN과의 유사점/차이점**
    - 윈도우를 **성공/실패 신호에 따라 동적으로 조정**한다는 점에서 ADWIN spirit 공유.
    - 그러나 ADWIN처럼 **명시적인 통계 검정과 $\delta$-보장**을 제공하지 않으며, 보다 **휴리스틱/딥러닝 지향**.

(4) **Concept drift detection and adaption framework using optimized deep learning and adaptive sliding window approach (OASW)** – Desale \& Shinde, 2023[^1_8]

- **아이디어**
    - Deep CNN + **optimized adaptive sliding window(OASW)** 조합으로 IoT/스트리밍 데이터의 drift에 대응.
    - 윈도우 크기와 CNN의 하이퍼파라미터를 메타휴리스틱(AHO)로 튜닝.
- **ADWIN과의 비교**
    - ADWIN처럼 **윈도우 길이를 동적으로 조정**하지만,
    - 이론적 bound 대신 **경험적 최적화**와 메타휴리스틱에 의존.
    - 높은 정확도(예: F1 ≈ 98%)를 보고하지만, **이해가능성·이론보장 측면은 ADWIN보다 약함**.

(5) **FLAME: Adaptive and Reactive Concept Drift Mitigation for Federated Learning** – Mavromatis et al., 2024[^1_9]

- **아이디어**
    - 대규모 IoT + Federated Learning 환경에서 drift를 감지·완화하는 시스템.
    - **ADWIN, Page-Hinkley, KSWIN** 등을 조합한 drift detection pipeline을 제안.
- **ADWIN과의 관계**
    - ADWIN을 **federated 환경의 로컬/글로벌 drift detector**로 사용.
    - ADWIN의 **lightweight·온라인 특성**이 대규모 분산 시스템에서 유리함을 실험적으로 확인.
- **의미**
    - ADWIN이 **분산/프라이버시 제약 환경**에서도 현실적인 building block으로 채택되고 있음을 보여줌.

(6) **Adaptive windowing based unsupervised/conditional regression frameworks** – 예: Conditioned Unsupervised Regression Framework (2024)[^1_10]

- **아이디어**
    - 멀티변량 스트림에서 비지도 회귀를 수행하면서,
    - ADWIN + RMSE 기반 drift detection을 결합해 **동적으로 모델을 업데이트**.
- **의미**
    - ADWIN을 **“레이블 대신 모델 성능 지표(RMSE)”**에 적용하는 패턴이 널리 쓰이고 있음을 보여줌.

(7) **개념 드리프트 서베이들** – Suárez-Cetrulo et al., 2022, Hinder et al., 2024 Part A/B[^1_5][^1_11][^1_12]

- **내용**
    - 데이터 스트림, recurring concept drift, unsupervised drift detection/설명 등 폭넓은 최근 연구를 정리.
- **ADWIN 위치**
    - ADWIN은 **“에러율 기반, adaptive window, 이론 보장”**을 가진 고전적·대표적인 베이스라인으로 계속 언급되고,
    - 많은 새로운 알고리즘이 **ADWIN과의 비교 실험**을 필수적으로 수행.

(8) **에너지 효율과 정확도 trade-off 연구** – How to Sustainably Monitor ML-Enabled Systems? (2024)[^1_13]

- **아이디어**
    - 여러 drift detector의 에너지 소비 vs 검출 정확도를 분석.
    - KSWIN vs ADWIN vs HDDM 등.
- **결과**
    - ADWIN은 **정확도-에너지 면에서 “balanced detector”** 카테고리로,
        - 높은 정확도를 유지하면서도 에너지 소비는 중간 수준으로 비교적 효율적임을 보임.


### 6.2 ADWIN 이후 연구의 공통 경향과 ADWIN의 위치

1. **딥러닝 표현 + ADWIN**
    - UDD, DriftLens류 방법은 딥러닝 representation의 분포/불확실성 위에 ADWIN 또는 유사 adaptive window 기법을 얹어,
        - **레이블 부족 + 고차원 데이터**라는 현실 문제를 해결하려 한다.[^1_4][^1_2][^1_3]
    - ADWIN은 여전히 “스칼라 시퀀스에 대한 robust change detector”로 사용.
2. **파라미터 튜닝/메타 학습으로의 확장**
    - ADDM, OASW, RL-Window류 연구는,
        - ADWIN의 철학(동적 윈도우)을 유지하면서,
        - 임계값·윈도우 크기·업데이트 정책을 **AR모형, 강화학습, 메타휴리스틱** 등으로 학습한다.[^1_14][^1_15][^1_6][^1_10]
    - 이는 **“ADWIN 스타일 구조 + 학습 기반 정책”**이라는 새로운 하이브리드 패턴.
3. **이론적 한계에 대한 재검토**
    - 최근 “Why Concept Drift Detection is Ill-posed / The window dilemma”류 연구는,
        - 실제 환경에서 drift가 완전히 관측·검증될 수 없음을 지적하며,
        - 어떤 의미에서든 완벽한 drift detection 문제는 **본질적으로 ill-posed**라고 주장한다.[^1_16][^1_17]
    - ADWIN도 이런 한계 안에 있지만,
        - **명시적인 $\delta$-보장과 윈도우 기반 구조** 덕분에,
        - 이론·실무에서 여전히 **비교 기준(baseline)·구성요소(component)**로 높은 위치를 유지.
4. **설명 가능한 드리프트와 위치 추정**
    - 최근 연구는 단순 **“drift 있음/없음”**에서 나아가,
        - **어떤 feature, 어떤 서브스페이스에서 drift가 일어났는지**,
        - **사람이 이해할 수 있는 방식으로 drift를 설명**하는 데 초점을 둔다.[^1_18][^1_19][^1_5]
    - ADWIN 자체는 이 부분을 다루지 않지만,
        - feature별 ADWIN, cluster별 ADWIN 등으로 확장하면,
            - **“어디에서 윈도우가 자주 줄어드는지”**를 통해 drift 위치 추정과 설명의 단서를 제공할 수 있다.

***

7. 앞으로의 연구에 미치는 영향과 향후 고려 사항
---------------------------------------------

### 7.1 ADWIN/ADWIN2의 영향

- **데이터 스트림/온라인 학습 커뮤니티에서의 표준 구성 요소**
    - River, menelaus 등 현대 스트리밍 라이브러리에서 **기본 제공 drift detector**로 구현되고 있음.[^1_20][^1_21][^1_22]
    - 많은 최신 논문이 drift detector를 설계하거나 벤치마킹할 때 **ADWIN을 필수 비교 대상**으로 삼는다.[^1_11][^1_13]
- **“파라미터 프리/적응형” 설계 패턴의 원형**
    - “윈도우 길이, decay rate를 사용자가 못 맞춘다”는 문제 인식 + 이를 **통계적 테스트로 해결**한 설계 방식은,
        - 이후 **adaptive ensemble, meta-learning 기반 drift adaptation** 등에서 반복적으로 재사용되는 디자인 패턴이다.[^1_23][^1_11]


### 7.2 향후 연구 시 고려할 점 (연구자 관점에서)

1. **고전 ADWIN을 modern deep/continual learning과 접목**
    - 딥러닝 기반 지속 학습(continual learning, EML, lifelong learning)에서,
        - **어디서 어떻게 학습률을 높이고, 어느 지점에서 리플레이/재학습을 할지** 결정하는 신호로,
        - **에러, 불확실성, representation distance 등에 ADWIN을 적용**하는 전략은 이미 성공적으로 시도되고 있다.[^1_23][^1_3][^1_4]
    - 단, 단순히 detector를 붙이는 수준을 넘어서,
        - **드리프트 심각도(severity), 유형(gradual vs sudden), 로컬 vs 글로벌**을 구분할 수 있는 richer signal이 필요.
2. **레이블 비용과 generalization의 균형**
    - 레이블 기반 에러율 모니터링은 레이블 비용이 큰 실제 환경에서 제한적.
    - UDD/DriftLens류처럼,
        - **레이블 없이도 generalization 저하를 감지할 수 있는 지표 (uncertainty, embedding 분포 등)**와
        - ADWIN류 detector를 결합하는 방향이 유망하다.[^1_2][^1_3][^1_4][^1_5]
3. **에너지·지연 제약 하에서의 drift detection**
    - 대규모 IoT·엣지·모바일 환경에서는,
        - drift detection 자체의 **에너지/지연 비용이 generalization과 직결**된다.[^1_9][^1_13]
    - ADWIN은 이미 “balanced detector”로 평가되지만,[^1_13]
        - 향후에는 **탐지 빈도(‘clock’), 버킷 수 $M$** 등을
            - **환경 제약(배터리, bandwidth)**에 맞게 동적으로 trade-off하는 연구가 필요.
4. **drift의 위치·원인 설명 및 causal 관점**
    - 단순 정확도 유지에서 나아가,
        - drift가 **어떤 변수/feature 조합에서**, **어떤 causal 구조 변화**에 의해 발생했는지 이해해야
            - 실제 시스템 운영(예: 의료, 금융)에서 신뢰 가능한 의사결정을 할 수 있다.[^1_24][^1_19][^1_18]
    - ADWIN 기반으로도,
        - feature·클러스터·component별로 detector를 나누고,
        - **어디서 자주 윈도우가 잘리는지**를 분석하면,
            - drift localization/explanation 연구로 자연스럽게 확장될 수 있다.
5. **ill-posedness와 평가 methodology**
    - 최근 “window dilemma”류 연구는,
        - **실제 데이터에서 drift의 “ground truth”를 정의하고 검증하는 것이 매우 어렵다**는 점을 상기시킨다.[^1_17][^1_16]
    - 따라서 향후 연구에서는
        - ADWIN 포함 모든 drift detector에 대해,
            - **synthetic + 반실험적(semisynthetic) 시나리오**,
            - **다양한 drift 유형·속도·label delay 패턴**에서
            - generalization, 비용, detection 안정성 등을 함께 평가하는 프레임워크가 중요하다.[^1_12][^1_5][^1_11]

***

8. 정리
-------

- 이 논문은 **시간에 따라 분포가 변하는 데이터 스트림에서, 슬라이딩 윈도우 크기를 데이터 기반으로 자동 조정하는 ADWIN/ADWIN2**를 제안하고,
    - **거짓 양성/거짓 음성 확률에 대한 이론 보장**,
    - **갑작/점진 드리프트 환경에서의 최적 윈도우 크기 자동 조정**,
    - **시간·메모리 효율적 구현(ADWIN2)**,
    - **NB, k-means에의 적용을 통한 성능 향상 검증**
를 통해, **개념 드리프트 대응의 “표준 도구”**를 제시한다.[^1_1]
- 일반화 관점에서는,
    - **시간 축 bias–variance trade-off를 자동으로 최적화**하고,
    - **feature/클러스터 단위의 비동기적 적응**을 가능하게 함으로써,
    - 여러 drift 패턴 및 희귀 이벤트 환경에서 **보다 안정적인 generalization 성능**을 제공한다.
- 2020년 이후 연구들은
    - **딥러닝 표현, 불확실성, federated learning, RL 기반 정책 학습, 설명 가능한 drift, 에너지 효율 분석** 등 방향으로 이 아이디어를 확장하고 있으며,
    - ADWIN은 여전히 **비교 기준이자 구성요소**로 핵심적인 역할을 하고 있다.[^1_3][^1_4][^1_7][^1_5][^1_11][^1_8][^1_12][^1_9][^1_13]
- 앞으로의 연구에서는,
    - ADWIN류 기법을 **modern continual/deep/federated learning 파이프라인**에 체계적으로 통합하고,
    - **레이블 비용·에너지·설명 가능성·ill-posedness**를 함께 고려하는 평가와 설계를 통해,
    - 실제 대규모, 고차원, 안전-중요 시스템에서도 **일반화 성능을 지속적으로 유지할 수 있는 “self-monitoring ML 시스템”**을 설계하는 것이 핵심 과제가 될 것이다.
<span style="display:none">[^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48]</span>

<div align="center">⁂</div>

[^1_1]: https://www.semanticscholar.org/paper/Learning-from-Time-Changing-Data-with-Adaptive-Bifet-Gavaldà/d7f8d7b89593c5333eb174b2411bf004f5a91f7d

[^1_2]: https://arxiv.org/html/2406.17813v2

[^1_3]: https://ar5iv.labs.arxiv.org/html/2107.01873

[^1_4]: https://arxiv.org/pdf/2107.01873.pdf

[^1_5]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11294200/

[^1_6]: https://arxiv.org/pdf/2203.04769.pdf

[^1_7]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9243804/

[^1_8]: https://www.semanticscholar.org/paper/Concept-drift-detection-and-adaption-framework-deep-Desale-Shinde/74e4fe11d0b84226823930d592313397352c2304

[^1_9]: https://arxiv.org/html/2410.01386v1

[^1_10]: http://arxiv.org/pdf/2312.07682.pdf

[^1_11]: https://reunir.unir.net/bitstream/handle/123456789/14409/a_survey_on_machine_learning.pdf?sequence=1\&isAllowed=y

[^1_12]: https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2024.1330257/full

[^1_13]: http://arxiv.org/pdf/2404.19452.pdf

[^1_14]: https://arxiv.org/html/2507.06901v1

[^1_15]: https://arxiv.org/html/2412.10119v1

[^1_16]: https://arxiv.org/html/2602.06456v1

[^1_17]: https://arxiv.org/pdf/2602.06456.pdf

[^1_18]: https://arxiv.org/abs/2301.08453

[^1_19]: https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2024.1330258/full

[^1_20]: https://riverml.xyz/0.19.0/api/drift/ADWIN/

[^1_21]: https://riverml.xyz/0.11.1/api/drift/ADWIN/

[^1_22]: https://menelaus.readthedocs.io/en/latest/menelaus.concept_drift.html

[^1_23]: https://arxiv.org/pdf/2505.17902.pdf

[^1_24]: https://www.semanticscholar.org/paper/4b769e2495bb4e88e247d6e8f345eb136313b6ff

[^1_25]: Learning_from_Time-Changing_Data_with_Adaptive_Win.pdf

[^1_26]: https://arxiv.org/html/2505.17902v3

[^1_27]: https://arxiv.org/html/2505.07852v1

[^1_28]: https://arxiv.org/abs/2112.02000

[^1_29]: https://arxiv.org/html/2510.15944v1

[^1_30]: https://arxiv.org/html/2511.09953v1

[^1_31]: https://link.springer.com/10.1007/s12243-020-00776-1

[^1_32]: https://ieeexplore.ieee.org/document/9219629/

[^1_33]: https://www.semanticscholar.org/paper/0a638a8f0d09238f3829f04bd69fe96693cded68

[^1_34]: https://ieeexplore.ieee.org/document/9356385/

[^1_35]: https://www.semanticscholar.org/paper/994f9ba1598cb94b314ae11dd487286ae314a10d

[^1_36]: https://www.semanticscholar.org/paper/011b89a091a5d44c0d2eb559a24fe1736d1170d7

[^1_37]: https://www.semanticscholar.org/paper/1692d231104643737f81f8bbc08f5d16e1092722

[^1_38]: https://www.annualreviews.org/doi/10.1146/annurev-bioeng-060418-052203

[^1_39]: https://pubs.geoscienceworld.org/ssa/bssa/article/110/6/2828/588050/Conditional-Ground-Motion-Model-for-Damaging

[^1_40]: http://arxiv.org/pdf/2311.13374.pdf

[^1_41]: http://arxiv.org/pdf/1010.4784.pdf

[^1_42]: https://www.mdpi.com/2076-3417/11/20/9644/pdf?version=1634529436

[^1_43]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9471369/

[^1_44]: https://www.iccs-meeting.org/archive/iccs2025/papers/159030210.pdf

[^1_45]: https://riverml.xyz/dev/api/drift/ADWIN/

[^1_46]: http://www.diva-portal.org/smash/get/diva2:1867280/FULLTEXT01.pdf

[^1_47]: https://www.sciencedirect.com/science/article/abs/pii/S0020025524005620

[^1_48]: https://www.sciencedirect.com/science/article/abs/pii/S0169023X2500120X

