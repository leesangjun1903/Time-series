# An Enhanced Seasonal-Hybrid ESD Technique for Robust Anomaly Detection on Time Series

**저자:** Rafael G. Vieira, Marcos A. Leone Filho, Robinson Semolini  
**소속:** University of Campinas, Venidera R&D, Elektro S.A (Brazil)

---

## 1. 핵심 주장과 주요 기여 요약

본 논문은 시계열 데이터에서 이상치(anomaly)를 자동으로 탐지하기 위한 **SH-ESD+ (Enhanced Seasonal-Hybrid Extreme Studentized Deviates)** 기법을 제안한다. 핵심 주장과 기여는 다음과 같다:

1. **Twitter의 기존 SH-ESD 기법의 한계를 극복**: 기존 SH-ESD는 계절 주기(seasonal periodicity)와 의심 이상치 수의 상한값($\gamma$)을 사전에 수동 지정해야 하는 한계가 있었으나, SH-ESD+는 이를 **자동으로 식별**하는 절차를 내장하였다.

2. **로버스트 통계 기법의 결합**: Box-Cox 변환, STL 분해(Loess 기반), 수정된 일반화 ESD 검정을 결합하여 트렌드·계절성 스파이크 존재 하에서도 **거짓 양성(false positive)을 최소화**하며 정확한 이상치 탐지를 수행한다.

3. **자동 파라미터 설정 및 갱신(Set and Update)**: 다른 기법들이 파라미터를 "업데이트만" 하거나 "자동화 없음"인 데 비해, SH-ESD+는 **파라미터의 초기 설정과 업데이트를 모두 자동화**한다.

4. **NAB 벤치마크에서 경쟁력 있는 성능**: Numenta HTM에 이어 2위의 NAB Score(59.1)를 달성하면서도, **지연 시간(latency)은 2.9ms로 가장 낮은 수준**을 기록하여 정확도-효율성 간 우수한 트레이드오프를 보여주었다.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

시계열 데이터에서 이상치를 **인간 개입 없이(without human interaction)** 효율적이고 정확하게 탐지하는 것이 핵심 문제이다. 구체적으로:

- **Temporal anomaly (시간적 이상치)**: 개별 데이터 포인트가 전체 패턴에서 이탈하는 경우
- **Spatial/contextual anomaly (공간적/맥락적 이상치)**: 특정 시간 맥락에서만 이상으로 간주되는 경우

기존 기법들의 한계:
- **ARIMA 기반 기법**: 트렌드·계절성은 모델링하지만, 파라미터 결정에 전문 지식 필요
- **Change point detection**: 윈도우 크기·임계값에 민감하여 거짓 양성 빈발
- **Twitter SH-ESD**: 계절 주기($v$)와 의심 이상치 상한($\gamma$)을 사전 지정해야 하며, 평탄 신호(flat signal), 지수 성장(exponential growth), 음의 트렌드(negative trend) 등의 패턴에서 탐지 실패

### 2.2 제안하는 방법 (수식 포함)

SH-ESD+는 세 단계로 구성된다: **데이터 변환 → 시계열 분해 → 잔차 분석**

입력: $\mathcal{X} = [\mathcal{X}_1, \ldots, \mathcal{X}_n] \in \mathbb{R}^n$ (단변량 시계열)  
출력: $\mathcal{A} = [\mathcal{A}_1, \ldots, \mathcal{A}_m] \in \mathbb{R}^m$ ($m < n$개의 이상치)

#### Step 1: Box-Cox 데이터 변환

데이터 정규성을 개선하기 위해 Box-Cox 변환을 적용한다:

```math
\mathcal{Y}_i = 
\begin{cases} 
\dfrac{(\mathcal{X}_i + \lambda_2)^{\lambda_1} - 1}{\lambda_1} & \text{if } \lambda_1 \neq 0 \\[8pt]
\log(\mathcal{X}_i + \lambda_2) & \text{if } \lambda_1 = 0
\end{cases}
```

여기서 $\lambda_1$은 최대우도법(Maximum Likelihood)으로 추정하고, $\lambda_2$는 $\mathcal{X}_i + \lambda_2 > 0$을 보장하도록 선택한다.

#### Step 2: STL 분해 (수정 버전)

변환된 시계열 $\mathcal{Y}$를 트렌드($\mathcal{T}$), 계절($\mathcal{S}$), 잔차($\mathcal{R}$)로 분해한다. 가법-승법 모델 선택 문제를 회피하기 위해 로그 변환을 적용한다:

$$
\log(\mathcal{Y}) = \log(\mathcal{T}) + \log(\mathcal{S}) + \log(\mathcal{R})
$$

**Inner loop** ($k = 1, \ldots, \varphi$회 반복):

1. **Detrending**: $\mathcal{Y}_{det}^{(k)} = \mathcal{Y} - \mathcal{T}^{(k)}$ (초기값 $\mathcal{T}^{(1)} = [0, \ldots, 0]$)
2. **Sub-cycle series smoothing**: Loess 회귀로 $v$개의 하위 주기 시리즈를 평활화 → $\mathcal{C}^{(k)}$
3. **Low-pass filter**: $\mathcal{C}^{(k)}$에 저역 통과 필터 적용 → $\mathcal{L}^{(k)}$
4. **Seasonal component**: $\mathcal{S}^{(k+1)} = \mathcal{C}^{(k)} - \mathcal{L}^{(k)}$
5. **Deseasonalizing**: $\mathcal{Y}_{des}^{(k+1)} = \mathcal{Y} - \mathcal{S}^{(k+1)}$
6. **Trend smoothing**: $\mathcal{Y}_{des}^{(k+1)}$를 평활화하여 $\mathcal{T}^{(k+1)}$ 갱신

**SH-ESD+의 핵심 개선 — 트렌드 추정**:
- 기본: **piecewise median** (에지 보존, 임펄스 왜곡 방지)
- 계절 주기가 존재하거나 평균 이동이 있을 때: **piecewise cubic splines** (평균 이동 처리에 강건)

**계절 주기($v$)의 자동 추정**: Welch의 주기도 평균법(periodogram averaging)으로 후보 주기 범위를 추정한 후, 교차검증 잔차 오차 기반 시간 영역 추정기로 최적 정수 주기 선택

**평균 이동($u$)의 자동 감지**: two-sample Student's t-Test를 사용하여, 각 데이터 포인트 $\mathcal{Y}_i$ 전후 $\frac{n}{10}$개 포인트의 평균을 비교 ($\rho$: t-분포 5% 임계값)

**Inner loop 반복 횟수의 자동 결정**:

$$
\varphi = \max(k) \left| \left( \frac{\|\mathcal{Y}_{des}^{(k)} - \mathcal{Y}_{des}^{(k-1)}\|}{n} \geq 10^{-2} \right) \right| \quad k = 1, 2, \ldots
$$

**Outer loop** ($l = 1, \ldots, \vartheta$회, 본 논문에서 $\vartheta = 2$):

1. 잔차 계산: $\mathcal{R}^{(l)} = \mathcal{Y} - \mathcal{S}^{(k)} - \mathcal{T}^{(k)}$
2. 강건 가중치 할당:

$$
\omega_i^{(l+1)} = B\left(\frac{\mathcal{R}_i^{(l)}}{6 \times \text{median}|\mathcal{R}^{(l)}|}\right), \quad i = 1, \ldots, n
$$

여기서 $B$는 bi-square 가중 함수:

$$
B(z) = 
\begin{cases}
(1 - z^2)^2 & \text{for } |z| \leq 1 \\
0 & \text{for } |z| > 1
\end{cases}
$$

#### Step 3: 수정된 일반화 ESD 검정 (잔차 분석)

잔차 $\mathcal{R}$에 대해 일반화 ESD 검정을 적용한다. 기존 ESD는 의심 이상치 상한 $\gamma$를 사전 지정해야 하지만, SH-ESD+는 이를 자동 결정한다.

**ESD 검정 통계량**:

$$
\tau^{(k)} = \max_{i=1}^{n} \left( \frac{|\mathcal{R}_i^{(k)} - \mu_{\mathcal{R}}^{(k)}|}{\sigma_{\mathcal{R}}^{(k)}} \right)
$$

**임계값**:

$$
\Gamma^{(k)} = \frac{(n-k) \, t_{p, n-k-1}}{\sqrt{(n-k-1+t_{p,n-k-1}^2)(n-k+1)}}
$$

여기서 $p = 1 - \frac{\alpha}{2(n-k+1)}$, $\alpha = 0.05$

**$\gamma$의 자동 결정 (핵심 기여)**: $k$번째 반복에서 남은 잔차 벡터 $\mathcal{R}^{(k)}$와 제거된 관측치 벡터 $\mathcal{Q}^{(k)}$에 대해, 다음 조건이 만족되는 최대 $k$를 $\gamma$로 설정:

$$
\sqrt{2} \, \sigma_{\Upsilon}^{(k)} \geq \sigma_{\Omega}^{(k)}
$$

여기서:
- $\Upsilon_i^{(k)} = |\mathcal{R}\_i^{(k)} - \mu_{\mathcal{R}}^{(k)}|$ (남은 잔차의 평균 이탈)
- $\Omega_i^{(k)} = |\mathcal{Q}\_i^{(k)} - \mu_{\mathcal{Q}}^{(k)}|$ (제거된 관측치의 평균 이탈)

**근거**: 초기에는 이상치가 포함되어 $\sigma_{\Upsilon}^{(k)} > \sigma_{\Omega}^{(k)}$이지만, 이상치가 순차 제거되면서 $\sigma_{\Upsilon}^{(k)}$는 감소하고 $\sigma_{\Omega}^{(k)}$는 증가한다. $\sqrt{2}\sigma_{\Upsilon}^{(k)} < \sigma_{\Omega}^{(k)}$가 되면 모든 이상치가 제거된 것으로 판단한다. 이후 $\tau^{(k)} > \Gamma^{(k)}$를 만족하는 최대 $k$를 최종 이상치 수로 확정하여 거짓 양성을 제어한다.

### 2.3 모델 구조 요약

```
입력 시계열 X
    ↓
[Step 1] Box-Cox 변환 → Y (정규성 개선)
    ↓
[Step 2] 수정 STL 분해 (Loess 기반)
    ├── 자동 계절 주기(v) 추정 (Welch + cross-validation)
    ├── 자동 평균 이동(u) 감지 (t-Test)
    ├── 트렌드 추정: piecewise median / piecewise cubic splines
    ├── Inner loop: 수렴 기반 자동 반복 횟수(φ) 결정
    └── Outer loop: bi-square 가중치로 이상치 영향 제거
    ↓
잔차 R = Y - T - S
    ↓
[Step 3] 수정된 일반화 ESD 검정
    ├── 자동 상한(γ) 결정 (Condition 8)
    └── τ(k) > Γ(k) 기준으로 최종 이상치 수 확정
    ↓
출력: 이상치 집합 A
```

### 2.4 성능 향상

**NAB 벤치마크 결과** (58개 데이터셋, 350,000+ 레코드):

| 기법 | NAB Score | Low FP | Low FN | Latency (ms) |
|------|-----------|--------|--------|--------------|
| Perfect | 100.0 | 100.0 | 100.0 | – |
| **Numenta HTM** | **70.1** | 63.1 | 74.3 | 12.5 |
| **SH-ESD+** | **59.1** | **53.6** | **68.7** | **2.9** |
| KNN-CAD | 58.0 | 43.4 | 64.8 | 13.9 |
| Twitter SH-ESD | 47.1 | 33.6 | 53.5 | 3.4 |

**주요 성능 향상 포인트**:
- Twitter SH-ESD 대비 **NAB Score 25.5% 향상** (47.1 → 59.1)
- **Low FP에서 59.5% 향상** (33.6 → 53.6) — 거짓 양성 대폭 감소
- **Low FN에서 28.4% 향상** (53.5 → 68.7) — 미탐지 감소
- **Latency 14.7% 개선** (3.4ms → 2.9ms)
- KNN-CAD(58.0)와 비슷한 정확도이나 **latency는 약 4.8배 빠름** (13.9ms vs 2.9ms)
- Numenta HTM(70.1)보다 정확도는 낮으나, **latency 약 4.3배 빠름** (12.5ms vs 2.9ms)

### 2.5 한계

1. **오프라인 탐지에 한정**: 본 논문은 오프라인(배치) 이상치 탐지만 다루며, 실시간(스트리밍) 환경에서의 적용은 고려하지 않았다. NAB의 스트리밍 지향 스코어링 체계에서 불리할 수 있다.

2. **Perfect Score(100.0)와의 격차**: 최고 성능인 Numenta HTM도 70.1에 그쳐, SH-ESD+의 59.1은 개선 여지가 상당하다.

3. **단변량 시계열에 한정**: 다변량(multivariate) 시계열에 대한 확장이 논의되지 않았다.

4. **정규성 가정**: ESD 검정은 근사적 정규 분포를 가정하므로, 극단적으로 비정규인 데이터에서는 성능 저하 가능성이 있다.

5. **$\sqrt{2}$ 상수의 경험적 결정**: Condition (8)에서 $\sqrt{2}$라는 상수는 "extensive experiments and careful observation"에 기반한 경험적 값으로, 이론적 근거가 충분히 제시되지 않았다.

6. **개념 변화(concept drift) 처리의 제한**: Table 2에서 concept drift 처리가 가능하다고 표기했으나, 오프라인 배치 처리 특성상 점진적 개념 변화에 대한 적응적 학습은 제한적이다.

---

## 3. 모델의 일반화 성능 향상 가능성

SH-ESD+의 일반화 성능과 관련된 핵심 내용과 향후 향상 가능성을 중점적으로 분석한다.

### 3.1 현재 일반화 성능에 기여하는 요소

**(1) 자동 파라미터 식별 (Parameter Automation: Set and Update)**

Table 2에서 볼 수 있듯이, SH-ESD+는 유일하게 파라미터를 **"Set and update"** 할 수 있는 기법이다. 이는 일반화 성능의 핵심 기반으로:
- 계절 주기 $v$: Welch 주기도 + 교차검증 잔차 오차로 자동 추정
- 이상치 상한 $\gamma$: Condition (8)에 의해 자동 결정
- Inner loop 반복 횟수 $\varphi$: 수렴 기준(Eq. 5)에 의해 자동 결정
- Box-Cox 변환 파라미터 $\lambda_1$: 최대우도법으로 자동 추정

이러한 자동화는 새로운 데이터셋에 대해 인간 개입 없이 적응할 수 있게 하여, 다양한 도메인의 시계열에 대한 일반화를 가능하게 한다.

**(2) 다중 트렌드 추정 전략의 적응적 선택**

- 기본: piecewise median (에지 보존)
- 계절 주기 존재 + 평균 이동 감지 시: piecewise cubic splines (평균 이동에 강건)

이 적응적 전환은 flat signal, exponential growth, negative trend 등 다양한 패턴에 대한 일반화를 가능하게 한다 (Figure 3의 SH-ESD 한계를 극복).

**(3) Box-Cox 변환에 의한 분포 안정화**

승법적(multiplicative) 계절성을 가법적(additive)으로 전환하여, STL 분해의 가법 모델을 일관되게 적용할 수 있게 하고, 이는 다양한 분포 특성을 가진 시계열에 대한 일반화에 기여한다.

**(4) STL의 강건 가중치(Robustness Weights)**

Bi-square 가중 함수(Eq. 3, 4)를 통해 이상치가 트렌드/계절 분해에 미치는 영향을 자동으로 감쇠시켜, 이상치의 빈도나 크기에 관계없이 안정적인 분해를 가능하게 한다.

### 3.2 NAB 벤치마크에서의 일반화 검증

NAB는 58개 데이터셋으로 구성되며 다음을 포함한다:
- 서버 메트릭 (CPU 사용률 등)
- 광고 클릭 데이터
- 인터넷 트래픽 볼륨
- 인공 데이터 (다양한 패턴)

SH-ESD+가 이들 **이질적인 도메인에서 일관되게(consistently)** 높은 성능을 보인 것은 일반화 능력의 증거이다. 특히 Low FP(53.6)와 Low FN(68.7) 모두에서 균형 잡힌 성능을 보였다.

### 3.3 일반화 성능 향상을 위한 개선 방향

**(1) $\sqrt{2}$ 상수의 데이터 적응적 결정**

Condition (8)의 $\sqrt{2}$를 고정 상수 대신 **데이터 특성(예: 분포 꼬리 무거움, 샘플 크기)에 따라 적응적으로 조절**하면, 극단적 분포를 가진 시계열에서의 일반화가 개선될 수 있다.

**(2) 다변량 확장**

현재 단변량($\mathcal{X} \in \mathbb{R}^n$)에 한정되어 있으므로, 다변량 시계열로의 확장(예: 다변량 ESD, 벡터 STL)이 일반화 범위를 크게 넓힐 수 있다.

**(3) 온라인/스트리밍 확장**

오프라인 배치 처리를 증분 학습(incremental learning)이나 슬라이딩 윈도우 기반으로 확장하면, 실시간 환경에서의 일반화가 가능해진다.

**(4) 비정규 분포에 대한 강건성 강화**

ESD 검정의 정규성 가정을 완화하기 위해, 비모수적(nonparametric) 이상치 검정이나 순위 기반 통계량을 결합하면 극단적 비정규 데이터에서의 일반화가 개선될 수 있다.

**(5) 딥러닝 기반 분해와의 하이브리드**

STL의 Loess 기반 분해를 딥러닝 기반 분해(예: N-BEATS, Neural Prophet)와 결합하여 비선형 트렌드·복잡한 계절성 포착 능력을 높이면 일반화 성능이 향상될 수 있다.

---

## 4. 향후 연구에 미치는 영향과 고려할 점

### 4.1 연구에 미치는 영향

**(1) 통계적 기법과 자동화의 결합 패러다임 제시**

SH-ESD+는 전통적 통계 검정(ESD)과 비모수 분해(STL)에 **자동 파라미터 결정 메커니즘**을 결합한 실용적 프레임워크를 제시했다. 이는 이후 연구에서 "통계적 해석 가능성(interpretability)을 유지하면서도 인간 개입을 최소화"하는 방향의 기초가 되었다.

**(2) 정확도-효율성 트레이드오프의 새로운 지평**

Numenta HTM 대비 NAB Score는 약 84% 수준이나 latency는 약 23% 수준(2.9ms vs 12.5ms)으로, **경량 통계 모델도 딥러닝 기반 모델과 경쟁할 수 있음**을 입증했다. 이는 자원 제약 환경(edge computing, IoT)에서의 이상치 탐지 연구에 방향성을 제시한다.

**(3) 벤치마크 기반 표준화된 평가 문화**

NAB를 활용한 체계적 비교 분석은 이상치 탐지 분야에서 **공정하고 재현 가능한 평가**의 중요성을 강조했으며, 이후 많은 연구들이 NAB 또는 유사 벤치마크를 채택하는 데 기여했다.

**(4) 분해 기반 이상치 탐지의 재조명**

STL 분해 → 잔차 분석이라는 파이프라인이 다양한 시계열 패턴에 효과적임을 보여주어, 이후 RobustSTL, MSTL 등 향상된 분해 기반 기법 연구를 촉진했다.

### 4.2 향후 연구 시 고려할 점

1. **실시간 스트리밍 환경**: 오프라인 배치 처리의 한계를 극복하여, 데이터가 도착하는 즉시 이상치를 탐지할 수 있는 온라인 버전 개발이 필요하다.

2. **다변량 시계열 확장**: 실제 시스템은 다수의 상호 연관된 시계열을 생성하므로, 변수 간 상관관계를 활용한 다변량 이상치 탐지로의 확장이 중요하다.

3. **비정상(non-stationary) 환경에서의 적응성**: concept drift가 빈번한 환경에서 파라미터의 점진적 갱신 메커니즘 강화가 필요하다.

4. **딥러닝과의 융합**: Transformer, VAE 등 최신 딥러닝 모델의 특징 추출 능력과 SH-ESD+의 통계적 해석 가능성을 결합하는 하이브리드 접근이 유망하다.

5. **도메인별 평가**: NAB 외에도 의료(ECG), 제조(센서), 금융(거래) 등 도메인별 특수성을 반영한 평가가 필요하다.

6. **이론적 근거 강화**: Condition (8)의 $\sqrt{2}$ 등 경험적으로 결정된 상수에 대한 이론적 최적성 증명이 필요하다.

7. **설명 가능성(Explainability)**: 탐지된 이상치가 **왜** 이상한지에 대한 설명 제공 기능의 추가가 실용적 활용을 위해 중요하다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

SH-ESD+가 제시한 "분해 기반 통계적 이상치 탐지" 패러다임 이후, 2020년 이후 다양한 접근법이 등장하였다.

### 5.1 분해 기반 접근의 발전

| 연구 | 핵심 방법 | SH-ESD+ 대비 차이점 |
|------|----------|------------------|
| **RobustSTL** (Wen et al., 2020) | Robust STL 분해 + bilateral filtering + sparse regularization | LAD 손실 및 bilateral filtering으로 비선형 트렌드에 더 강건. SH-ESD+의 piecewise 접근보다 연속적 |
| **MSTL** (Bandara et al., 2021) | 다중 계절 주기 STL 분해 | 단일 주기($v$)만 추정하는 SH-ESD+와 달리 복수 계절 주기 동시 분해 가능 |

- **Wen, Q., Gao, J., Song, X., Sun, L., Xu, H., & Zhu, S.** (2020). "RobustSTL: A Robust Seasonal-Trend Decomposition Algorithm for Long Time Series." *Proceedings of the AAAI Conference on Artificial Intelligence, 33*, 5409–5416. (원 논문 2019, 후속 활용 2020+)
- **Bandara, K., Hyndman, R. J., & Bergmeir, C.** (2021). "MSTL: A Seasonal-Trend Decomposition Algorithm for Time Series with Multiple Seasonal Patterns." *International Journal of Operational Research* (arXiv:2107.13462).

### 5.2 딥러닝 기반 접근

| 연구 | 핵심 방법 | SH-ESD+ 대비 차이점 |
|------|----------|------------------|
| **USAD** (Audibert et al., 2020) | Adversarially trained autoencoders | 비지도 학습으로 다변량 지원, 복잡한 비선형 패턴 포착. 해석 가능성은 SH-ESD+보다 낮음 |
| **Anomaly Transformer** (Xu et al., 2022) | Transformer + association discrepancy | 시계열 내 포인트 간 연관성의 불일치(association discrepancy)를 활용. SH-ESD+의 통계적 검정 기반 판단보다 표현력이 높으나 계산 비용 큼 |
| **TranAD** (Tuli et al., 2022) | Transformer + adversarial training + focus score | 스트리밍 설정에서 실시간 다변량 이상치 탐지. SH-ESD+의 오프라인 한계 극복 |

- **Audibert, J., Michiardi, P., Guyard, F., Marti, S., & Zuluaga, M. A.** (2020). "USAD: UnSupervised Anomaly Detection on Multivariate Time Series." *Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 3395–3404.
- **Xu, J., Wu, H., Wang, J., & Long, M.** (2022). "Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy." *Proceedings of the 10th International Conference on Learning Representations (ICLR 2022)*.
- **Tuli, S., Casale, G., & Jennings, N. R.** (2022). "TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data." *Proceedings of the VLDB Endowment, 15*(6), 1201–1214.

### 5.3 통계-딥러닝 하이브리드 접근

| 연구 | 핵심 방법 | SH-ESD+ 대비 차이점 |
|------|----------|------------------|
| **SR-CNN** (Ren et al., 2019/2021) | Spectral Residual + CNN | Microsoft에서 개발. 주파수 도메인 잔차와 CNN 판별기 결합. SH-ESD+와 유사한 "분해→잔차 분석" 파이프라인이나 판별을 CNN으로 수행 |
| **N-BEATS for AD** (응용 연구, 2021+) | Neural Basis Expansion + residual analysis | 딥러닝으로 시계열 분해 후 잔차 기반 이상치 판단. SH-ESD+의 STL을 신경망으로 대체 |

- **Ren, H., Xu, B., Wang, Y., Yi, C., Huang, C., Kou, X., Xing, T., Yang, M., Tong, J., & Zhang, Q.** (2019). "Time-Series Anomaly Detection Service at Microsoft." *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 3009–3017.

### 5.4 벤치마크 및 평가 방법론의 발전

| 연구 | 핵심 기여 | SH-ESD+ 평가와의 관계 |
|------|----------|------------------|
| **TimeEval** (Schmidl et al., 2022) | 71개 이상치 탐지 알고리즘의 체계적 벤치마크 | NAB 외에도 다양한 벤치마크 데이터셋 포함. SH-ESD+ 류 통계적 기법이 특정 시나리오에서 딥러닝과 경쟁적임을 확인 |
| **TSB-UAD** (Paparrizos et al., 2022) | 통합 시계열 이상치 탐지 벤치마크 | NAB의 한계(제한된 데이터셋 수, 스트리밍 편향 스코어링)를 보완하는 대규모 벤치마크 제공 |

- **Schmidl, S., Wenig, P., & Papenbrock, T.** (2022). "Anomaly Detection in Time Series: A Comprehensive Evaluation." *Proceedings of the VLDB Endowment, 15*(9), 1779–1797.
- **Paparrizos, J., Kang, Y., Boniol, P., Tsay, R. S., Palpanas, T., & Franklin, M. J.** (2022). "TSB-UAD: An End-to-End Benchmark Suite for Univariate Time-Series Anomaly Detection." *Proceedings of the VLDB Endowment, 15*(8), 1697–1711.

### 5.5 종합 비교

| 차원 | SH-ESD+ (본 논문) | 2020+ 딥러닝 기법 | 2020+ 통계/하이브리드 기법 |
|------|-----------------|---------------|-------------------|
| **해석 가능성** | ✅ 높음 (통계 검정 기반) | ❌ 낮음 (블랙박스) | ⚠️ 중간 |
| **계산 효율** | ✅ 매우 높음 (2.9ms) | ❌ GPU 필요 | ⚠️ 중간 |
| **다변량 지원** | ❌ 없음 | ✅ 대부분 지원 | ⚠️ 일부 지원 |
| **온라인 탐지** | ❌ 오프라인만 | ✅ 대부분 지원 | ✅ 대부분 지원 |
| **파라미터 자동화** | ✅ Set and update | ⚠️ 하이퍼파라미터 튜닝 필요 | ⚠️ 부분적 |
| **비정규 데이터** | ⚠️ Box-Cox로 완화 | ✅ 분포 가정 불필요 | ⚠️ 기법에 따라 다름 |
| **비선형 패턴** | ⚠️ 제한적 | ✅ 우수 | ⚠️ 중간 |

---

## 참고자료

1. **본 논문**: Vieira, R. G., Leone Filho, M. A., & Semolini, R. "An Enhanced Seasonal-Hybrid ESD Technique for Robust Anomaly Detection on Time Series." (논문 원문 PDF)
2. Cleveland, R. B. et al. (1990). "STL: A Seasonal-Trend Decomposition Procedure Based on Loess." *Journal of Official Statistics*, 6(1), 3–73.
3. Rosner, B. (1975). "On the Detection of Many Outliers." *Technometrics*, 17(2), 221–227.
4. Lavin, A. & Ahmad, S. (2015). "Evaluating Real-Time Anomaly Detection Algorithms – The Numenta Anomaly Benchmark." *Proceedings of the 14th IEEE ICMLA*, 38–44.
5. Ahmad, S. & Purdy, S. (2016). "Real-time anomaly detection for streaming analytics." *arXiv:1607.02480*.
6. Ahmad, S. et al. (2017). "Unsupervised real-time anomaly detection for streaming data." *Neurocomputing*, 262(1), 134–147.
7. Audibert, J. et al. (2020). "USAD: UnSupervised Anomaly Detection on Multivariate Time Series." *ACM SIGKDD 2020*, 3395–3404.
8. Xu, J. et al. (2022). "Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy." *ICLR 2022*.
9. Tuli, S. et al. (2022). "TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data." *PVLDB*, 15(6), 1201–1214.
10. Schmidl, S. et al. (2022). "Anomaly Detection in Time Series: A Comprehensive Evaluation." *PVLDB*, 15(9), 1779–1797.
11. Paparrizos, J. et al. (2022). "TSB-UAD: An End-to-End Benchmark Suite for Univariate Time-Series Anomaly Detection." *PVLDB*, 15(8), 1697–1711.
12. Ren, H. et al. (2019). "Time-Series Anomaly Detection Service at Microsoft." *ACM SIGKDD 2019*, 3009–3017.
13. Bandara, K. et al. (2021). "MSTL: A Seasonal-Trend Decomposition Algorithm for Time Series with Multiple Seasonal Patterns." *arXiv:2107.13462*.
14. Wen, Q. et al. (2019). "RobustSTL: A Robust Seasonal-Trend Decomposition Algorithm for Long Time Series." *AAAI 2019*, 33, 5409–5416.
