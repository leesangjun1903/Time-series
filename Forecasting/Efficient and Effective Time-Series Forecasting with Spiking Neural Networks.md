# Efficient and Effective Time-Series Forecasting with Spiking Neural Networks

> **참고 자료:**
> - Lv, C., Wang, Y., Han, D., Zheng, X., Huang, X., & Li, D. (2024). "Efficient and Effective Time-Series Forecasting with Spiking Neural Networks." *Proceedings of the 41st International Conference on Machine Learning (ICML 2024)*. PMLR 235. arXiv:2402.01533v2
> - GitHub Repository: https://github.com/microsoft/SeqSNN
> - Horowitz, M. (2014). "1.1 computing's energy problem." ISSCC 2014.
> - Fang, W. et al. (2020b). SpikingjJelly framework.
> - Liu, Y. et al. (2024). "iTransformer." ICLR 2024.

---

## 1. Executive Summary (10문장 이내)

스파이킹 신경망(SNN)은 생물학적 뉴런의 스파이크 발화 메커니즘에서 영감을 받은 **3세대 신경망**으로, 에너지 효율성과 이벤트 기반(event-driven) 패러다임이라는 독보적인 장점을 지닌다.  
본 논문은 SNN을 시계열 예측(time-series forecasting) 태스크에 최초로 체계적으로 적용한 통합 프레임워크(SeqSNN)를 제안한다.  
핵심 과제는 연속형 시계열 데이터와 이산적(discrete) SNN 스파이크 표현 간의 시간적 정렬(temporal alignment) 문제를 해결하는 것이었다.  
이를 위해 **Delta 스파이크 인코더**와 **합성곱 스파이크 인코더** 두 가지를 설계하여 부동소수점(floating-point) 데이터를 스파이크 트레인(spike train)으로 변환한다.  
모델 아키텍처로는 기존 ANN의 CNN, RNN, Transformer를 각각 SNN 버전인 **Spike-TCN, Spike-RNN(Spike-GRU), iSpikformer**로 변환하였다.  
4개의 실제 벤치마크(Metr-la, Pems-bay, Solar, Electricity) 실험 결과, 제안 SNN 모델들은 최신 ANN 기반 모델과 동등하거나 우수한 예측 성능을 달성하였다.  
특히 iSpikformer는 SOTA ANN 모델인 iTransformer와 RSE 기준으로 사실상 동등한 성능(평균 RSE 0.476 vs 0.480)을 보였다.  
에너지 소비 측면에서는 ANN 대비 평균 **70.33% 절감** 효과가 확인되었으며, Spike-GRU는 최대 75.05%의 에너지 절감을 달성하였다.  
합성 주기 신호 실험을 통해 제안된 SNN이 저주파 및 고주파 시계열 모두에서 시간 의존성을 효과적으로 모델링함을 검증하였다.  
본 연구는 SNN 기반 시계열 예측의 실용적 가능성을 최초로 체계적으로 제시하며, 에너지 효율적·생물학적으로 타당한 예측 모델 개발의 새로운 방향을 제시한다.

> **💡 용어 설명**
> - **스파이킹 신경망(SNN, Spiking Neural Network):** 생물 뉴런처럼 막전위(membrane potential)가 임계값에 도달할 때만 '스파이크(0 또는 1의 이진 신호)'를 발화하는 신경망. 연속적 실수값을 사용하는 일반 ANN과 달리, 이진 스파이크 신호로 계산하여 에너지 소비가 극히 낮다.
> - **이벤트 기반(Event-driven) 패러다임:** 입력 데이터에 변화가 발생했을 때만 계산을 수행하는 방식. 뇌의 신경 신호 전달 방식과 유사하다.
> - **스파이크 트레인(Spike Train):** 시간에 따른 스파이크(발화 이벤트)의 연속적 시퀀스. 이진값(0 또는 1)으로 구성된다.

---

### 1-1. 연구의 목적과 필요성

**연구 목적:**
SNN을 시계열 예측 태스크에 효과적으로 적용하기 위한 통합 프레임워크를 개발하고, 기존 ANN 대비 성능은 유지하면서 에너지 효율을 대폭 향상시키는 것.

**연구 필요성 (p.1~2):**

| 필요성 근거 | 상세 설명 |
|---|---|
| **데이터-모델 적합성** | 시계열 데이터는 본질적으로 시간적 구조를 가지며, SNN의 이벤트 기반 시간 처리 방식과 자연스럽게 부합함 |
| **에너지 효율 문제** | 기존 ANN 기반 예측 모델의 막대한 연산(MAC 연산) 비용을 AC 연산 중심의 SNN으로 대체할 필요 |
| **탐구 공백** | 기존 연구들이 시계열에서 SNN의 시간적 특성을 무시하거나 단순 반복(repetition) 방식을 사용해 왔음 (p.1) |
| **실용성** | 뉴로모픽 하드웨어 배포 가능성: 동적 비전 센서(DVS) 데이터셋과 달리, 시계열 데이터는 현실에서 쉽게 획득 가능 |
| **표준 부재** | SNN 시계열 예측을 위한 표준화된 모델 선택 가이드라인이 존재하지 않음 |

> **💡 용어 설명**
> - **MAC(Multiply-Accumulate) 연산:** 일반 ANN에서 사용하는 '곱셈 후 누산' 연산. 에너지 소비가 큰 부동소수점 연산이다.
> - **AC(Accumulate) 연산:** SNN에서 스파이크(0/1)와 가중치의 단순 덧셈 연산. MAC 대비 에너지 소비가 약 5배 낮다.
> - **뉴로모픽 하드웨어(Neuromorphic Hardware):** Intel Loihi, IBM TrueNorth 등 뇌의 신경 구조를 모방하여 스파이크 기반 연산에 특화된 칩. SNN 추론 시 일반 GPU 대비 훨씬 낮은 에너지를 소비한다.
> - **동적 비전 센서(DVS, Dynamic Vision Sensor):** 픽셀 밝기 변화가 발생했을 때만 데이터를 전송하는 이벤트 기반 카메라 센서.

---

## 2. 핵심 주장과 근거 표

| 핵심 주장 | 근거 | 위치 |
|---|---|---|
| SNN은 시계열 예측에서 ANN과 동등하거나 우수한 성능 달성 가능 | iSpikformer: 평균 RSE 0.476 (iTransformer: 0.480), Spike-RNN이 GRU 대비 통계적으로 유의미하게 우수 (p<0.05) | Table 1, p.6~7 |
| 합성곱 스파이크 인코더가 Delta 및 반복 인코더보다 효과적 | Conv 인코더 vs Delta 인코더: 평균 0.09 R² 향상; 반복 인코더는 다수 설정에서 수렴 실패 | Table 2, p.7 |
| SNN은 ANN 대비 평균 70.33% 에너지 절감 | 45nm 뉴로모픽 하드웨어 기준: Spike-GRU 75.05%↓, iSpikformer 66.30%↓ | Table 3, p.8 |
| 시간 동역학 보존이 SNN 성능의 핵심 요인 | Spike-TCN(상태 비유지)은 TCN 대비 성능 하락, Spike-RNN/GRU(상태 유지)는 오히려 GRU 초과 성능 | Table 1, p.6 |
| SNN이 저·고주파 시계열 패턴 모두 포착 가능 | 합성 신호 실험에서 모든 SNN 모델이 두 주파수 대역에서 강력한 예측 성능 시연 | Figure 5, p.8 |
| 시간 스텝 $T_s$ 및 감쇠율 $\beta$ 조정에 대해 모델이 robust함 | $T_s \in \{4,8,12,16\}$, $\beta \in \{0.80,...,0.99\}$ 범위에서 R² 변동 최소 | Figure 4, p.8 |

---

### 2-1. 핵심 내용 상세 설명

#### ① 해결하고자 하는 문제 (p.1~2)

1. **시간적 정렬 문제:** 연속 시계열 $\Delta T$와 SNN 이산 시간 스텝 $\Delta t$ 간의 매핑 불일치
2. **인코딩 복잡성:** 부동소수점 시계열을 정보 손실 없이 이진 스파이크 트레인으로 변환하는 문제
3. **모델 선택 기준 부재:** 시계열 예측에 적합한 SNN 아키텍처 선택 가이드라인 없음

---

#### ② 제안 방법 및 수식

**[Step 1: LIF 뉴런 동역학] (p.3, Figure 1)**

$$U(t) = H(t - \Delta t) + I(t), \quad I(t) = f(\mathbf{x}; \theta) $$

$$H(t) = V_{reset} S(t) + (1 - S(t)) \beta U(t) $$

$$S(t) = \begin{cases} 1, & \text{if } U(t) \geq U_{thr} \\ 0, & \text{if } U(t) < U_{thr} \end{cases} $$

| 기호 | 의미 |
|---|---|
| $U(t)$ | 시간 $t$에서의 막전위(membrane potential) |
| $H(t)$ | 시간 $t$에서의 뉴런 시간적 출력 |
| $I(t)$ | 시간 $t$에서의 공간적 입력 전류 |
| $f(\mathbf{x}; \theta)$ | 입력 $\mathbf{x}$와 학습 파라미터 $\theta$를 사용한 함수 |
| $\Delta t$ | LIF 모델링의 이산화 상수 |
| $V_{reset}$ | 스파이크 발화 후 초기화되는 리셋 전압 |
| $\beta$ | 막전위 감쇠율 (decay rate), $0 < \beta < 1$ |
| $U_{thr}$ | 스파이크 발화 임계값(threshold) |
| $S(t)$ | 시간 $t$에서의 스파이크 출력 (Heaviside step function) |

> **💡 용어 설명**
> - **LIF(Leaky Integrate-and-Fire) 뉴런:** 가장 널리 사용되는 스파이킹 뉴런 모델. 외부 입력 전류를 통합(integrate)하고, 임계값 도달 시 스파이크 발화 후 막전위를 리셋. "Leaky"는 시간이 지남에 따라 감쇠율 $\beta$로 막전위가 자연적으로 새는(leaking) 특성을 의미.
> - **Heaviside Step Function:** 입력이 0 이상이면 1, 미만이면 0을 출력하는 계단 함수. 미분 불가능하여 역전파 시 대리 기울기(surrogate gradient)가 필요.

**[Step 2: 대리 기울기(Surrogate Gradient)] (p.3)**

Heaviside 함수가 미분 불가능하므로, 아크탄젠트 근사를 사용:

$$S(t) \approx \frac{1}{\pi} \arctan\left(\frac{\pi}{2} \alpha U(t)\right) + \frac{1}{2} $$

$$\frac{\partial S(t)}{\partial U(t)} = \frac{\alpha}{2} \cdot \frac{1}{1 + \left(\frac{\pi}{2} \alpha U(t)\right)^2}$$

| 기호 | 의미 |
|---|---|
| $\alpha$ | 아크탄젠트 함수의 주파수를 제어하는 하이퍼파라미터 (실험 설정: $\alpha=2$) |

> **💡 용어 설명**
> - **대리 기울기(Surrogate Gradient):** Heaviside 계단 함수의 기울기가 거의 모든 지점에서 0이므로, 역전파 시 이를 연속 함수(여기서는 arctan)로 근사하여 학습 가능하게 만드는 기법.
> - **BPTT(Backpropagation Through Time):** 시간 축으로 전개된 계산 그래프에 역전파를 적용하는 알고리즘. RNN 계열 모델 학습에 표준적으로 사용.

**[Step 3: 시간 정렬] (p.3~4)**

$$\Delta T = T_s \cdot \Delta t$$

| 기호 | 의미 |
|---|---|
| $\Delta T$ | 시계열 데이터의 시간 스텝 크기 |
| $T_s$ | 하나의 시계열 시간 스텝 내 SNN 시뮬레이션 스텝 수 |
| $\Delta t$ | SNN 시간 스텝 크기 |

**[Step 4: Delta 스파이크 인코더] (p.4, Eq.6)**

$$\mathbf{S} = \mathcal{SN}\left(\text{BN}\left(\text{Linear}(\mathbf{x}_t - \mathbf{x}_{t-1})\right)\right) $$

| 기호 | 의미 |
|---|---|
| $\mathbf{x}\_t - \mathbf{x}_{t-1}$ | 인접 시간 스텝 간의 차분(temporal difference) |
| $\text{Linear}(\cdot)$ | 차분에 다른 민감도를 학습하고 차원을 $T_s \times T \times C$로 확장하는 선형층 |
| $\text{BN}(\cdot)$ | 배치 정규화(Batch Normalization) |
| $\mathcal{SN}(\cdot)$ | 스파이킹 뉴런 레이어 (연속값 → 이진 스파이크 변환) |
| $\mathbf{S}$ | 출력 스파이크 트레인, $\mathbf{S} \in \mathbb{R}^{T_s \times T \times C}$ |

> **💡 용어 설명**
> - **Delta 변조(Delta Modulation):** 신호의 절대값 대신 시간적 변화량(차분)만을 전송하는 신호 처리 기법. 생물학적으로 뉴런이 절대적 자극 강도보다 변화에 민감하다는 사실에 기반.
> - **배치 정규화(Batch Normalization):** 미니배치 단위로 중간 레이어의 출력을 정규화하여 학습을 안정화하는 기법.

**[Step 5: 합성곱 스파이크 인코더] (p.4, Eq.7)**

$$\mathbf{S} = \mathcal{SN}\left(\text{BN}\left(\text{Conv}(\mathbf{X})\right)\right) $$

| 기호 | 의미 |
|---|---|
| $\mathbf{X} \in \mathbb{R}^{T \times C}$ | 원본 시계열 입력 |
| $\text{Conv}(\cdot)$ | 시간 방향 합성곱 레이어 (시퀀스의 형태(shape) 정보를 추출) |

**[Step 6: iSpikformer 임베딩] (p.5, Eq.8)**

$$\mathbf{S}_{emb} = \mathcal{SN}(\text{Linear}(\mathbf{S})) $$

| 기호 | 의미 |
|---|---|
| $\mathbf{S}_{emb} \in \mathbb{R}^{H \times C}$ | $C$개 채널의 차원 $H$ 채널별 스파이크 임베딩 |

**[Step 7: 에너지 소비 계산] (Appendix B, p.12~13)**

SNN 에너지:
$$\text{Energy}(l) = E_{AC} \times \text{SOPs}(l) $$

$$\text{SOPs}(l) = T \times \gamma \times \text{FLOPs}(l) $$

ANN 에너지:
$$\text{Energy}(b) = E_{MAC} \times \text{FLOPs}(b) $$

| 기호 | 의미 |
|---|---|
| $E_{AC} = 0.9 \text{ pJ}$ | 45nm 공정에서 AC(누산) 연산 1회 에너지 |
| $E_{MAC} = 4.6 \text{ pJ}$ | 45nm 공정에서 MAC(곱셈-누산) 연산 1회 에너지 |
| $\text{SOPs}(l)$ | SNN 레이어 $l$의 시냅스 연산(Synaptic Operations) 횟수 |
| $\text{FLOPs}(b)$ | ANN 레이어 $b$의 부동소수점 연산(MAC) 횟수 |
| $T$ | SNN 시뮬레이션 시간 스텝 수 |
| $\gamma$ | 레이어 $l$ 입력 스파이크 트레인의 발화율(firing rate) |

> **💡 용어 설명**
> - **발화율(Firing Rate):** 전체 시간 스텝 중 실제로 스파이크(1)가 발생한 비율. 값이 낮을수록 계산이 희소(sparse)하여 에너지 효율이 높다.
> - **SOPs(Synaptic Operations):** SNN에서 스파이크 신호와 시냅스 가중치 간의 단순 덧셈 연산 횟수.

**[평가 지표] (Appendix A.4, p.12)**

$$\text{RSE} = \sqrt{\frac{\sum_{m=1}^{M} ||\mathbf{Y}^m - \hat{\mathbf{Y}}^m||^2}{\sum_{m=1}^{M} ||\mathbf{Y}^m - \bar{\mathbf{Y}}||^2}} $$

$$R^2 = \frac{1}{MCL} \sum_{m=1}^{M} \sum_{c=1}^{C} \sum_{l=1}^{L} \left[1 - \frac{(Y^m_{c,l} - \hat{Y}^m_{c,l})^2}{(Y^m_{c,l} - \bar{Y}_{c,l})^2}\right] $$

| 기호 | 의미 |
|---|---|
| $M$ | 테스트 샘플 수 |
| $C$ | 변수(채널) 수 |
| $L$ | 예측 길이(horizon) |
| $\mathbf{Y}^m$ | $m$번째 샘플의 실제값 |
| $\hat{\mathbf{Y}}^m$ | $m$번째 샘플의 예측값 |
| $\bar{\mathbf{Y}}$ | 실제값의 평균 |
| $Y^m_{c,l}$ | $m$번째 샘플의 $c$번째 변수 $l$번째 미래 실제값 |
| $\bar{Y}_{c,l}$ | $Y^m_{c,l}$의 전체 샘플 평균 |

---

#### ③ 모델 구조 (p.4~5, Figure 2)

```
입력 시계열 X ∈ ℝ^(T×C)
         ↓
[스파이크 인코더]
  ├── Delta 인코더:   S = SN(BN(Linear(x_t - x_{t-1})))
  └── Conv 인코더:    S = SN(BN(Conv(X)))
  → 출력: S ∈ ℝ^(B × Ts × T × C)
         ↓
[SNN 백본 선택]
  ├── (a) Spike-TCN:      Conv2d → BN → SN → SEW 잔차연결
  ├── (b) Spike-RNN/GRU:  Recurrent Cell (SN 활성화) → BN → Linear
  └── (c) iSpikformer:    SSA → BN → Linear (Block × N)
         ↓
[스파이크 디코딩]
  Linear(S_hidden) → Y ∈ ℝ^(L×C)
```

**세 모델의 핵심 특징 비교:**

| 모델 | 시간 상태 유지 | 병렬 훈련 | 특징 |
|---|---|---|---|
| Spike-TCN | ✗ (매 스텝 $U(t)=0$ 리셋) | ✓ | 로컬 시간 패턴, SEW 잔차 연결 |
| Spike-RNN/GRU | ✓ (스텝 간 상태 전파) | ✗ | 장기 의존성 캡처, 게이팅 메커니즘 |
| iSpikformer | ✓ | ✓ (Transformer) | Spiking Self-Attention(SSA), 채널 간 공간 모델링 |

> **💡 용어 설명**
> - **SEW(Spike-Element-Wise) 잔차 연결:** SNN용 잔차 연결 모듈. 스파이크 값(0/1)으로 항등 사상(identity mapping)을 구현하여 기울기 소실/폭발 문제를 해결.
> - **SSA(Spiking Self-Attention):** Transformer의 Self-Attention을 스파이크 기반으로 재설계한 메커니즘. Q, K, V 행렬 계산 시 부동소수점 곱셈 대신 스파이크 연산을 사용.
> - **iTransformer:** 시계열을 전치(invert)하여 개별 변수를 토큰으로 사용하는 2024년 SOTA Transformer 모델. 채널 간 상관관계를 Self-Attention으로 포착.

---

#### ④ 성능 향상 및 한계

**성능 향상 (Table 1, Table 3):**
- iSpikformer: iTransformer 대비 $R^2$ 차이 단 0.001, RSE는 오히려 더 낮음
- Spike-RNN: GRU 대비 통계적으로 유의미한 성능 향상 (p<0.05, 평균 RSE 0.503*)
- 에너지: 평균 70.33% 절감 (Spike-GRU 75.05%, iSpikformer 66.30%)

**한계 (Appendix D, p.13~14):**
1. **스파이크 디코더 한계:** 출력층에서 부동소수점 선형층을 사용해야 함 → 완전한 SNN 추론 파이프라인 미완성
2. **이론적 분석 부재:** 인코더-SNN-디코더 간 연결에 대한 이론적 근거 부족, 엔지니어링 실험 연구에 가까움
3. **에너지 추정의 단순화:** GPU 환경에서 실제 에너지 측정 불가, 이론적 추정값만 제공
4. **Spike-TCN의 구조적 한계:** 시간 상태를 유지하지 않아 SNN의 이벤트 기반 특성과 충돌

---

## 3. 각 주장에 위치 표시

| 주장 | 페이지/위치 |
|---|---|
| SNN이 제3세대 신경망이며 에너지 효율성을 가짐 | p.1, Section 1 |
| LIF 뉴런 동역학 수식 | p.3, Section 3.1.2, Eq.(1)~(3), Figure 1 |
| 대리 기울기(arctan) 수식 | p.3, Eq.(5) |
| 시간 정렬 핵심 개념 $\Delta T = T_s \Delta t$ | p.3, Section 3.2 |
| Delta 스파이크 인코더 수식 | p.4, Eq.(6) |
| 합성곱 스파이크 인코더 수식 | p.4, Eq.(7) |
| iSpikformer 임베딩 수식 | p.5, Eq.(8) |
| 4개 벤치마크 종합 성능 결과 | p.6~7, Table 1 |
| Critical Difference 다이어그램 | p.6, Figure 3 |
| 인코더 종류별 성능 비교 | p.7, Table 2 |
| 하이퍼파라미터 민감도 분석 | p.8, Figure 4 |
| 에너지 소비 비교 | p.8, Table 3 |
| 합성 신호 예측 시각화 | p.8, Figure 5 |
| 에너지 계산 공식 | p.12~13, Appendix B, Eq.(12)~(14) |
| 한계 및 미래 방향 | p.13~14, Appendix D |

---

## 4. 저자 보고 결과 vs. 해석 분리

### 저자가 직접 보고한 결과

| 항목 | 저자 보고 결과 | 위치 |
|---|---|---|
| iSpikformer $R^2$ vs iTransformer | 0.775 vs 0.776, 차이 0.001 | Table 1, p.6 |
| iSpikformer 평균 RSE | 0.476 (iTransformer 0.480보다 낮음) | Table 1, p.6 |
| Spike-RNN 평균 RSE | 0.503* (p<0.05로 GRU 대비 유의미) | Table 1, p.6 |
| 에너지 절감 | 평균 70.33%; Spike-GRU 75.05%, Spike-TCN 63.60%, iSpikformer 66.30% | Table 3, p.8 |
| 합성곱 vs Delta 인코더 | 평균 R² 0.09 차이로 합성곱 우수 | Table 2, p.7 |
| 반복 인코더 수렴 실패 | 다수 설정에서 R²≈0.02, RSE≈1.05 (수렴 실패) | Table 2, p.7 |
| $T_s$ 하이퍼파라미터 | $T_s=16$에서 약간의 성능 저하 | Figure 4(a)(b), p.8 |
| $\beta$ 하이퍼파라미터 | $\beta$ 높을수록 R² 감소 | Figure 4(c)(d), p.8 |

### 검토자(본 분석)의 해석

| 항목 | 해석 및 평가 |
|---|---|
| **성능 동등성의 의미** | 이진 스파이크(0/1) 연산만으로 부동소수점 모델 수준 달성은 인상적이나, 벤치마크가 4개로 제한적이어서 범용성 주장에는 주의 필요 |
| **Spike-RNN의 성능 우위 원인** | 저자는 시간 상태 유지를 원인으로 제시하나, GRU 자체의 학습 불안정성(Table 1에서 GRU가 일부 단기 예측에서도 부진)이 원인의 일부일 가능성 존재 |
| **에너지 추정의 한계** | 45nm 뉴로모픽 하드웨어 이론값으로, 실제 GPU 실행 시 $T_s$배 느린 추론 속도를 고려하면 실용적 에너지 효율은 다를 수 있음 |
| **$\beta$ 증가 시 성능 저하 해석** | 저자는 장기 기억 보존이 오히려 해롭다고 해석하나, 시계열 예측에서 과거 정보 가중치와 예측 horizon 길이의 상호작용을 추가 분석할 필요 있음 |
| **합성 신호 실험의 한계** | Spike-TCN의 고주파 피크 예측 부진이 단순히 TCN 구조 한계인지, 인코더 민감도 문제인지 구분되지 않음 |

---

## 5. 통계적 취약점 및 비교 불가 수치

### ⚠️ 통계적 취약점

| 문제 | 상세 내용 | 위치 |
|---|---|---|
| **제한된 벤치마크 수** | 4개 데이터셋(Metr-la, Pems-bay, Solar, Electricity)만 사용. ETTh/ETTm 등 시계열 예측에서 광범위하게 사용되는 벤치마크 미포함 | p.6, Table 1 |
| **3개 시드 평균** | "All results are averaged across 3 random seeds"로 통계적 신뢰구간 미보고 | Table 1, p.6 |
| **p<0.05 검정 범위 제한** | 통계적 유의성(p<0.05) 표시가 일부 모델(Spike-RNN, Spike-GRU RSE)에만 적용 | Table 1, p.6 |
| **에너지 추정의 이론적 성격** | 실제 하드웨어 측정값이 아닌 이론적 추정. 저자 스스로 "too simplified"라고 인정 | Appendix B, p.13 |
| **합성 신호 실험의 대표성** | 단 하나의 예측 슬라이스(T=20, L=80)만 시각화 | Figure 5, p.8 |

### ⚠️ 비교 불가능한 수치

| 문제 | 상세 내용 |
|---|---|
| **일부 베이스라인 결과 출처 이질성** | ARIMA, GP, GRU, Autoformer 결과는 Fang et al. (2023) 논문에서 인용. 동일 실험 환경(하드웨어, 시드) 불일치 가능성 |
| **SNN GPU 추론 속도** | GPU에서 SNN은 $T_s$배 느리므로 추론 속도는 직접 비교 불가. 뉴로모픽 하드웨어에서만 속도 이점 실현 가능 |
| **Long-term forecasting 벤치마크 제외** | Weather, Exchange-rate 등 장기 예측 벤치마크 대상 실험 없어 일반화 성능 불확실 |

---

## 6. 논문이 답하지 않는 질문

| # | 미답 질문 |
|---|---|
| 1 | **실제 뉴로모픽 하드웨어(예: Intel Loihi)에서의 측정된 에너지 소비 및 추론 속도는 얼마인가?** |
| 2 | **더 긴 예측 horizon(예: L=336, L=720)에서의 성능은 어떠한가?** (ETTm 계열 장기 예측 미포함) |
| 3 | **스파이크 발화율(firing rate) $\gamma$가 성능에 미치는 영향은?** 발화율이 너무 낮으면 정보 손실이 발생하는가? |
| 4 | **SNN의 이론적 표현 능력(representational capacity)은 ANN과 동일한가?** 이진 연산만으로의 표현력 한계가 어디인가? |
| 5 | **다변량 시계열(multivariate) 내 변수 간 상관관계(cross-variable correlation)를 SNN이 얼마나 효과적으로 포착하는가?** |
| 6 | **스파이크 디코더를 완전히 SNN화(spiking output)하면 회귀 성능이 얼마나 저하되는가?** |
| 7 | **훈련 시간(학습 속도)은 ANN 대비 얼마나 차이가 나는가?** BPTT의 $T_s$배 역전파 비용 분석 미포함 |
| 8 | **데이터 노이즈나 이상치(outlier)에 대한 SNN의 강건성(robustness)은?** |
| 9 | **사전 학습(pre-training) 또는 전이 학습(transfer learning) 시나리오에서 SNN의 적용 가능성은?** |
| 10 | **Spike-TCN의 병렬 훈련을 가능하게 하는 $U(t)$ 리셋 없는 알고리즘은 어떻게 설계할 수 있는가?** (미래 방향으로만 언급) |

---

## 7. 가장 중요한 그림 5개 해석

### **Figure 1 (p.3): LIF 뉴런의 순환 표현**

```
[해석]
이 그림은 LIF 뉴런이 어떻게 시간 단계 t-2 → t-1 → t로 
정보를 전파하는지를 보여준다.

핵심 관계:
- U[t-1]은 이전 H[t-2]와 현재 입력 I[t-1]의 합
- U[t-1] ≥ U_thr: 스파이크 발화(S=1), H 리셋
- U[t-1] < U_thr: 스파이크 없음(S=0), U에 β 감쇠 적용

시사점:
β(감쇠)로 조절되는 막전위의 '기억' 메커니즘이 
SNN의 시간적 정보 통합 능력의 근원임을 시각적으로 설명.
Spike-RNN에서 이 상태가 시계열 시간 스텝 간에도 유지되는 것이
GRU 대비 성능 우위의 핵심 메커니즘.
```

### **Figure 2 (p.4): SeqSNN 프레임워크 전체 개요**

```
[해석]
논문의 핵심 기여를 하나의 그림에 집약.

1. 입력 시계열 → 스파이크 인코더: 
   연속값이 Ts×T×C 형태의 이진 스파이크로 변환되는 과정 시각화
   → 시간 정렬의 핵심: ΔT = Ts·Δt 관계 명시

2. 세 SNN 아키텍처의 구조 차이:
   - Spike-TCN: 2D Conv + BN 블록 (공간-시간 합성곱)
   - Spike-RNN: Recurrent Cell (시간 상태 순환)
   - Spike-Transformer: SSA + BN + Linear 블록 (주의 메커니즘)

3. 출력 프로젝션:
   스파이크 → 부동소수점 변환의 불가피성 (현재 한계)

의의: 기존 SNN 연구가 시간 정렬을 무시했던 것과 달리,
본 프레임워크가 시계열 데이터의 시간 구조를 
SNN 고유 특성과 명시적으로 연결함.
```

### **Figure 3 (p.6): Critical Difference(CD) 다이어그램**

```
[해석]
Wilcoxon-Holm 검정(95% 신뢰수준)으로 
모든 방법의 성능 순위를 통계적으로 비교.

3계층 구조 관찰:
- 1계층(최상위): iTransformer, iSpikformer, TCN, Spike-RNN
- 2계층(중간): Autoformer, Spike-GRU, GRU
- 3계층(하위): Spike-TCN, ARIMA, GP

핵심 발견:
1. iSpikformer가 iTransformer와 통계적으로 구별 불가능한 
   동일 계층에 위치 → SNN이 SOTA ANN과 동등
2. Spike-RNN이 GRU보다 높은 계층 → 시간 상태 유지의 효과 입증
3. Spike-TCN의 하위 계층 배치 → 상태 리셋의 부정적 영향 확인

⚠️ 주의: CD 다이어그램은 순위 기반 비교로,
절대적 성능 차이의 실용적 유의성은 별도 판단 필요.
```

> **💡 용어 설명**
> - **Critical Difference(CD) 다이어그램:** Friedman 검정 후 Nemenyi 사후 검정을 시각화한 그림. 연결된 모델들은 통계적으로 유의미한 차이가 없음을 의미.

### **Figure 4 (p.8): 하이퍼파라미터 민감도 분석**

```
[해석]
(a)(b) Time Step Ts ∈ {4,8,12,16} 영향:
- 전반적으로 R²가 Ts에 안정적 → 모델이 robust
- Ts=16에서 소폭 하락: 저자는 "자기 누적 동역학(self-accumulating
  dynamics)"으로 설명 (대리 기울기 오차 누적 → 기울기 소실/폭발)
- 실용적 선택: Ts=4가 성능과 계산 비용의 최적 균형점

(c)(d) 감쇠율 β ∈ {0.80,...,0.99} 영향:
- β 증가 → R² 감소 (역설적: 높은 β = 더 강한 기억 유지가 오히려 해롭다)
- 해석: 시계열 예측은 최근 패턴이 중요한데, 
  β가 크면 오래된 정보가 과도하게 잔류하여 예측 정확도 저하
- Spike-TCN이 Solar에서 β 변화에 가장 민감 → 
  상태 리셋으로 인해 β의 역할이 제한적으로 작동함을 시사

핵심 메시지: 시계열 SNN에서는 β를 0.99 미만(예: 0.85~0.95)으로 
설정하는 것이 바람직하며, 이는 도메인별로 튜닝 필요.
```

### **Figure 5 (p.8): 합성 시계열 데이터 예측 결과**

```
[해석]
합성 신호: X(t) = A1·sin(ω1·t) + A2·sin(ω2·t+φ) + N(0,σ) [Eq.9]

(a) 저주파 데이터 (ω2=0.04π):
- 세 모델 모두 완만한 곡선 패턴을 정확히 예측
- 시각적으로 거의 완벽한 Ground Truth 추적
- 시사점: SNN이 저주파 트렌드와 계절성 포착에 효과적

(b) 고주파 데이터 (ω2=0.1π):
- Spike-RNN, iSpikformer: 피크값 예측 정확도 높음
- Spike-TCN: 피크 부근에서 약간 낮은 정확도
- 해석: 고주파 신호에서 TCN의 로컬 컨볼루션 한계가 드러남
  RNN의 시간 상태 유지가 고주파 진동 패턴 학습에 유리

의의: 
- SNN이 순수 사인파 합성 신호뿐 아니라 복잡한 
  다주파수 패턴도 포착 가능함을 실증
- 실제 시계열의 추세(trend)+계절성(seasonality)+잔차(residual) 
  분해 구조와 유사한 설계

⚠️ 한계: 단 하나의 예측 슬라이스로 일반화 근거 약함
```

---

## 8. 결론, 시사점, 후속 연구

### 저자 제시 시사점 (p.8~9, Appendix D)

1. **통합 프레임워크 확립:** SNN의 시계열 예측 적용을 위한 표준 가이드라인 제시
2. **에너지 효율 입증:** ANN 대비 평균 70.33% 에너지 절감으로 그린 AI 가능성 제시
3. **시간 동역학 중요성:** SNN에서 시간 상태 보존이 성능의 핵심 결정 요인임을 실험적으로 규명
4. **공간-시간 분리 가능성:** 시간 모델링(SNN)과 공간 모델링(iTransformer)이 독립적으로 개선 가능

### 저자 제시 후속 연구 계획 (Appendix D.2, p.14)

1. **병렬 학습 가능한 TCN형 SNN 알고리즘 개발:** $U(t)$ 리셋 없이 TCN 병렬 학습 지원
2. **스파이킹 그래프 신경망 탐구:** 시계열의 공간 정보 활용을 위한 Spiking GNN 연구

### 추가 제안 후속 연구 방향

| 방향 | 구체적 내용 |
|---|---|
| **스파이크 디코더 완전 SNN화** | 출력층의 부동소수점 선형 레이어를 스파이크 발화율 기반으로 대체하는 회귀-친화적 디코더 설계 |
| **장기 예측(Long-term Forecasting)** | ETTh, ETTm, Weather 등 192~720 스텝 예측에서의 성능 검증 |
| **희소 이벤트 시계열 적용** | 금융 틱 데이터, 지진 센서 등 자연적 이벤트 기반 데이터에서 SNN의 이점 극대화 |
| **연속 학습(Continual Learning)** | 비정상(non-stationary) 시계열에서 SNN의 막전위 상태를 활용한 온라인 적응 학습 |
| **이론적 수렴 분석** | 대리 기울기를 사용한 BPTT의 수렴 보장 조건 및 표현력 이론 정립 |

---

### 8-1. 모델의 일반화 성능 향상 가능성

**현재 일반화 한계:**

1. **도메인 편중:** 4개 벤치마크 모두 교통(Metr-la, Pems-bay) + 에너지(Electricity, Solar) 도메인으로 제한. 의료, 기상, 금융 등 이질적 시계열에서의 성능 불확실
2. **예측 horizon 제한:** 최대 $L=96$ 스텝으로 단·중기 예측에 집중. 장기 예측에서 막전위 누적 오차 문제 예상
3. **단변량 합성 신호 실험:** 실제 다변량 복잡 시계열의 비선형성을 충분히 검증하지 못함

**일반화 향상을 위한 구체적 제언:**

| 제언 | 근거 |
|---|---|
| **사전 학습 + 파인튜닝 패러다임 도입** | 최근 PatchTST, TimesFM 등의 기반 모델(foundation model) 접근법을 SNN에 적용. 스파이크 인코더를 데이터 독립적 특징 추출기로 사전 학습 가능 |
| **적응적 $T_s$ 및 $\beta$ 학습** | 고정 하이퍼파라미터 대신 데이터 분포에 따라 자동 조정되는 메타 학습(meta-learning) 기반 파라미터 적응 |
| **비정상성(non-stationarity) 처리** | 시계열의 분포 변화에 대응하기 위한 적응적 정규화(예: RevIN)를 SNN 인코더에 통합 |
| **희소 어텐션 + 스파이크 결합** | iSpikformer의 전체 어텐션 대신 희소 어텐션을 도입하여 장기 의존성 학습 효율화 |
| **도메인 일반화 실험 확대** | MIMIC-III(의료), M4(경제), SleepEDF(뇌파) 등 다양한 도메인 벤치마크 추가 |

> **💡 용어 설명**
> - **RevIN(Reversible Instance Normalization):** 시계열의 비정상성(non-stationarity) 문제를 해결하기 위해 정규화-역정규화를 쌍으로 적용하는 기법. 분포 변화에 강건한 예측이 가능.
> - **Foundation Model(기반 모델):** 대규모 데이터로 사전 학습되어 다양한 다운스트림 태스크에 파인튜닝으로 적용 가능한 범용 모델. GPT, BERT의 개념을 시계열로 확장.

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> **⚠️ 중요 고지:** 아래 비교 분석은 제공된 논문의 인용 문헌 및 2024년 5월(논문 최종 업로드 기준)까지 공개된 연구를 바탕으로 작성되었습니다. 2024년 이후 발표된 일부 연구에 대해서는 제 학습 데이터 기준(2024년 초)으로 불완전할 수 있으며, 해당 부분은 명시적으로 표시합니다.

#### 시계열 예측 주요 ANN 연구 비교

| 연구 | 연도 | 핵심 기여 | SeqSNN과의 관계 |
|---|---|---|---|
| **Autoformer** (Wu et al.) | 2021 | 자동 상관(Auto-Correlation) 기반 분해 Transformer | SeqSNN 베이스라인으로 사용, SNN이 비슷한 성능 달성 |
| **TimesNet** (Wu et al.) | 2023 | 1D 시계열을 2D 변환하여 시간 변동 모델링 | SeqSNN이 직접 비교하지 않음 → 추가 비교 연구 필요 |
| **iTransformer** (Liu et al.) | 2024 | 변수를 토큰으로 사용, 채널 간 상관 모델링 | SeqSNN의 iSpikformer의 기반 모델, 성능 거의 동등 |
| **PatchTST** (Nie et al.) | 2023 | 패치 기반 Transformer로 지역-전역 정보 통합 | SeqSNN에서 미비교. 패치 개념을 스파이크 인코더에 통합 가능 |

#### SNN 분야 주요 연구 비교

| 연구 | 연도 | 핵심 기여 | SeqSNN과의 관계 |
|---|---|---|---|
| **Spikformer** (Zhou et al.) | 2023 | SNN + Transformer (이미지 분류) | SeqSNN의 SSA 메커니즘 기반 |
| **Spikformer v2** (Zhou et al.) | 2024 | ImageNet SOTA SNN | SeqSNN이 SSA를 iTransformer에 통합 |
| **Spike-driven Transformer** (Yao et al.) | 2023 | 완전 스파이크 기반 Transformer | SeqSNN의 이론적 상한선 제시 |
| **SpikeBERT** (Lv et al.) | 2023 | 지식 증류로 SNN 언어 모델 학습 | 같은 저자 그룹 연구, 지식 증류 기반 SNN 강화 가능성 |

#### 이 논문이 앞으로의 연구에 미치는 영향

1. **SNN 응용 도메인 확장의 선구:** 이미지/NLP에 집중되던 SNN 연구를 시계열이라는 새로운 응용 영역으로 확장하는 기점 역할

2. **통합 프레임워크의 기준점 제시:** 인코더-모델-디코더의 파이프라인이 SNN 시계열 연구의 표준 참조 아키텍처로 기능할 가능성

3. **에너지 효율 연구 자극:** 70.33% 에너지 절감 실증은 엣지 컴퓨팅(edge computing), IoT 센서 시계열 처리에서 SNN 적용을 촉진

#### 앞으로 연구 시 고려할 점

| 고려 사항 | 구체적 방향 |
|---|---|
| **이론적 기반 강화** | 스파이크 연산의 시계열 표현력에 대한 수학적 분석(예: universal approximation for SNNs) 필요 |
| **실제 하드웨어 배포 검증** | Intel Loihi 2, SpiNNaker 등에서의 실측 에너지 및 지연시간(latency) 측정 필수 |
| **공정한 비교 실험** | 최신 SOTA(PatchTST, TimeMixer 등)와의 비교 및 동일 환경에서 베이스라인 재현 |
| **비정상 시계열 처리** | 분포 변화가 많은 실제 시계열(주식, 기상)에서의 SNN 적응 메커니즘 연구 |
| **학습 효율성 개선** | BPTT의 시간 복잡도( $O(T \cdot T_s)$ )를 줄이는 근사 학습 알고리즘 (예: e-prop, OSTL) 탐구 |
| **멀티모달 확장** | 시계열 + 외부 변수(텍스트, 이미지)를 스파이크 기반으로 통합하는 멀티모달 SNN |

> **💡 용어 설명**
> - **엣지 컴퓨팅(Edge Computing):** 클라우드 서버가 아닌 데이터 발생 지점(IoT 기기, 센서) 근처에서 연산을 수행하는 분산 컴퓨팅 패러다임. 저지연, 저전력이 핵심 요구사항이어서 SNN과 높은 시너지를 가짐.
> - **e-prop:** BPTT를 사용하지 않고 온라인(실시간)으로 SNN을 학습하는 생물학적으로 타당한 학습 알고리즘.

---

**[최종 참고 자료 목록]**
- Lv, C. et al. (2024). "Efficient and Effective Time-Series Forecasting with Spiking Neural Networks." ICML 2024. arXiv:2402.01533v2
- Horowitz, M. (2014). ISSCC 2014.
- Maass, W. (1997). "Networks of spiking neurons: the third generation of neural network models." Neural Networks.
- Fang, W. et al. (2020b). SpikingjJelly.
- Liu, Y. et al. (2024). "iTransformer." ICLR 2024.
- Zhou, Z. et al. (2023b). "Spikformer." ICLR 2023.
- Zhou, Z. et al. (2024). "Spikformer v2." arXiv:2401.02020
- Yao, M. et al. (2023a). "Spike-driven Transformer." NeurIPS 2023.
- Wu, H. et al. (2021). "Autoformer." NeurIPS 2021.
- Wu, H. et al. (2023). "TimesNet." ICLR 2023.
- Bai, S. et al. (2018). "TCN." arXiv:1803.01271
- Cho, K. et al. (2014). "GRU." EMNLP 2014.
- Werbos, P.J. (1990). "BPTT." Proc. IEEE.
- Wu, Y. et al. (2019). "Direct training for SNNs." AAAI 2019.
- GitHub: https://github.com/microsoft/SeqSNN
