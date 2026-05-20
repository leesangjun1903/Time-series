# AutoUAD: Hyper-parameter Optimization for Unsupervised Anomaly Detection

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

AutoUAD는 **레이블이 없는 훈련 환경**에서 비지도 이상 탐지(UAD) 모델의 하이퍼파라미터 최적화 및 모델 선택 문제를 해결하기 위한 프레임워크다. 기존 방법들은 역사적 데이터셋 기반 메타학습이나 이상치 비율의 사전 지식을 필요로 하여 비지도 학습의 원칙에 위배되었다. 본 논문은 이를 완전히 비지도 방식으로 해결하는 **대리 평가 지표(surrogate metrics)**를 제안한다.

### 주요 기여

| 기여 | 설명 |
|------|------|
| **RTM (Relative Top-Median)** | 내부 평가 지표: 상위 $\tau$%의 평균 점수와 중앙값의 상대적 차이 측정 |
| **EAG (Expected Anomaly Gap)** | 내부 평가 지표: 이상 점수 분포의 기대 분리도 측정 |
| **NPD (Normalized Pseudo Discrepancy)** | 반내부 평가 지표: 훈련 데이터의 검증셋과 등방성 가우시안에서 생성된 데이터 간의 이상 점수 불일치 측정 |
| **Bayesian Optimization 통합** | 위 세 지표를 BO(TPE 기반)와 결합하여 효율적 하이퍼파라미터 탐색 구현 |
| **이론적 보장** | NPD에 대한 엔트로피 하한, 오류율 상한 등 이론적 분석 제공 |
| **대규모 실험** | 38개 벤치마크 데이터셋, 4개(+5개 추가) UAD 알고리즘에서 검증 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

**문제 정의 (Definition 1 - UAD):**

데이터셋 $\mathcal{X} = \{\boldsymbol{x}_i \in \mathbb{R}^d : i = 1, 2, \ldots, N\}$에서:
- $N_0$개 샘플은 정상 분포 $\mathcal{D}_0$에서, $N_1 = N - N_0$개는 이상 분포 $\mathcal{D}_1$에서 추출
- 각 샘플의 출처 레이블 **미지**
- $N_0 \gg N_1$ (예: $N_0 = 10N_1$)
- $\mathcal{D}\_0$와 $\mathcal{D}\_1$의 overlap $\eta(\mathcal{D}\_0, \mathcal{D}\_1) = \int_{\mathbb{R}^d} \min\{p_{\boldsymbol{x} \sim \mathcal{D}\_0}(\boldsymbol{x}), p_{\boldsymbol{x} \sim \mathcal{D}_1}(\boldsymbol{x})\}d\boldsymbol{x}$이 충분히 작음

**핵심 도전과제:**

**Definition 2 (AutoUAD)**에 따르면, 두 가지 목표를 달성해야 한다:

- **Goal 1** (하이퍼파라미터 최적화): 각 $\mathcal{M}_i \in \mathbb{M}$에 대해

```math
\Theta_i^* = \arg\max_{\Theta_i \in \prod_{j=1}^{H_i} \mathcal{S}_j^{(i)}} \mathbb{E}_{(\tilde{\mathcal{X}}, \tilde{\mathcal{Y}}) \sim \tilde{\mathcal{D}}^{\tilde{N}}} \left[ \mathcal{E}(\tilde{\mathcal{Y}}, \{f_{\mathcal{M}_i}(\boldsymbol{x}|\Theta_i) : \boldsymbol{x} \in \tilde{\mathcal{X}}\}) \right] 
```

- **Goal 2** (최적 모델 선택):

```math
\mathcal{M}^*(\Theta^*) = \arg\max_{\mathcal{M}(\Theta^*) \in \mathbb{M}(\Theta^*)} Q(\mathcal{M}(\Theta^*))
```

레이블이 없으므로, 대리 함수 $\mathcal{V}$를 설계하여:

```math
\mathcal{V}(\mathcal{M}_i, \mathcal{X}) \approx g\left(\mathbb{E}_{(\tilde{\mathcal{X}}, \tilde{\mathcal{Y}}) \sim \tilde{\mathcal{D}}^{\tilde{N}}}\left[\mathcal{E}(\tilde{\mathcal{Y}}, \{f_{\mathcal{M}_i}(\boldsymbol{x}|\Theta_i) : \boldsymbol{x} \in \tilde{\mathcal{X}}\})\right]\right)
```

여기서 $g: \mathbb{R} \rightarrow \mathbb{R}$은 이상적으로 단조증가 함수다.

---

### 2.2 제안하는 방법 (수식 포함)

#### 2.2.1 Relative Top-Median (RTM)

**Assumption 1:** 좋은 UAD 모델은 대다수 데이터에 낮은 이상 점수를, 소수 데이터에 상대적으로 높은 이상 점수를 부여한다.

**Definition 3 (RTM):** 정렬된 이상 점수 $s_{(1)} \leq s_{(2)} \leq \ldots \leq s_{(N)}$에 대해:

```math
\mathcal{V}_{RTM}(\mathcal{M}, \mathcal{X}) = \frac{\text{mean}(\{s_i | s_i \geq s_{(\tau/100 \cdot N)}\}) - \text{median}(\boldsymbol{s})}{\text{median}(\boldsymbol{s}) + \epsilon}
```

- $\tau = 5$ (상위 5%)로 휴리스틱 설정
- $\epsilon = 10^{-6}$: 분모 0 방지용 상수

#### 2.2.2 Expected Anomaly Gap (EAG)

**Definition 4 (AG):** 임계값 $\xi$에서의 이상 간격:

```math
AG(\xi; p(s)) = \frac{w_0(\xi)w_1(\xi)(\mu_0(\xi) - \mu_1(\xi))^2}{w_0(\xi)\sigma_0^2(\xi) + w_1\sigma_1^2(\xi) + \epsilon}
```

여기서 $w_0(\xi) = P(s < \xi)$, $w_1(\xi) = P(s \geq \xi)$, $\mu_0(\xi) = \mathbb{E}[s|s < \xi]$, $\mu_1(\xi) = \mathbb{E}[s|s \geq \xi]$.

**Definition 5 (EAG):** $s_{thr} = G^{-1}(0.8)$ (80번째 백분위수, Assumption 2에 의거)로 설정하여:

```math
\mathcal{V}_{EAG}(\mathcal{M}, \mathcal{X}) = \mathbb{E}[AG(\xi; p(s)) | \xi \geq s_{thr}] = \frac{1}{s_{max} - s_{thr}} \int_{s_{thr}}^{s_{max}} AG(\xi) d\xi
```


**Assumption 2:** 훈련셋의 최대 20%가 실제 이상치와 유사한 데이터.

#### 2.2.3 Normalized Pseudo Discrepancy (NPD) ← 핵심 기여

**Definition 6 (NPD):**

1. 훈련 데이터를 $\mathcal{X}\_{trn}$ (크기 $N-M$)과 $\mathcal{X}_{val}$ (크기 $M$)로 분할
2. $\mathcal{X}_{trn}$으로 UAD 모델 학습
3. $\boldsymbol{\mu}\_{trn}$, $\boldsymbol{\sigma}^2_{trn}$을 계산하여 가우시안 데이터 생성: $\mathcal{X}\_{gen} \sim \mathcal{N}(\boldsymbol{\mu}\_{trn}, \text{diag}(\boldsymbol{\sigma}^2_{trn}))$
4. 이상 점수 계산: $\boldsymbol{s}\_{val} = f_{\mathcal{M}}(\mathcal{X}\_{val}|\Theta)$, $\boldsymbol{s}\_{gen} = f_{\mathcal{M}}(\mathcal{X}_{gen}|\Theta)$
5. NPD 계산:

```math
\mathcal{V}_{NPD}(\mathcal{M}, \mathcal{X}) = \frac{(\text{Mean}(\boldsymbol{s}_{gen}) - \text{Mean}(\boldsymbol{s}_{val}))^2}{2(\text{Var}(\boldsymbol{s}_{gen}) + \text{Var}(\boldsymbol{s}_{val})) + \epsilon}
```

#### 2.2.4 Bayesian Optimization (BO) 통합

Tree-structured Parzen Estimator (TPE)를 사용한 BO로 하이퍼파라미터 순차 탐색:

$$EI_{y^*}(\Theta) \propto \left(\gamma + \frac{\varphi(\Theta)}{\ell(\Theta)}(1-\gamma)\right)^{-1}$$

- $\ell(\Theta)$: 손실값이 $y^*$ 미만인 관측치들의 밀도
- $\varphi(\Theta)$: 나머지 관측치들의 밀도

---

### 2.3 모델 구조

```
훈련 단계:
  ┌─────────────────────────────────────────────────┐
  │  후보 UAD 모델들: {M₁, M₂, ..., Mₒ}              │
  │  각 Mᵢ에 대해:                                   │
  │    Bayesian Optimizer → 하이퍼파라미터 선택        │
  │    → 모델 훈련 on Xtrn                           │
  │    → (RTM/EAG/NPD) 평가 on 훈련셋               │
  │    → BO 피드백 루프                              │
  └─────────────────────────────────────────────────┘
           ↓
  최적 모델 M*(Θ*) 선택 (Goal 2)

테스트 단계:
  M*(Θ*) → 이상 점수 추론 → AUC/F1 평가
```

**NPD 세부 구조:**
```
X → 분할 → Xtrn (훈련) + Xval (검증)
              ↓
           모델 학습
              ↓
Xgen ~ N(μtrn, diag(σ²trn))  ←  가우시안 생성
              ↓
    sval = fM(Xval|Θ), sgen = fM(Xgen|Θ)
              ↓
         NPD = Eq.(7)
```

---

### 2.4 성능 향상

**Table 1 (38개 데이터셋 평균 AUC/F1):**

| 방법 | OCSVM AUC | AE AUC | DeepSVDD AUC | DPAD AUC |
|------|-----------|--------|--------------|---------|
| Max (상한) | 85.75 | 84.86 | 80.18 | 83.38 |
| Default | 78.73 | 81.47 | 73.90 | 73.65 |
| Random | 74.71 | 81.88 | 74.45 | 71.20 |
| EM/MV | 71.41 | 83.08 | 74.43 | 72.59 |
| **NPD (Ours)** | **84.03** | **83.58** | 74.38 | **80.36** |

- NPD는 OCSVM(p<0.0001), AE(p=0.0212), DPAD(p<0.0001)에서 통계적으로 유의한 성능 향상
- UOMS 실험에서도 NPD가 AUC 83.49±16.4, F1 59.30±26.2로 모든 베이스라인 능가

### 2.5 한계점

1. **NPD의 비단조성**: 논문 결론에서 명시: "the highest NPD did not always correspond to the best model performance"
2. **가우시안 가정의 한계**: 실제 데이터가 심하게 비가우시안이거나 다봉분포일 경우 $\mathcal{X}_{gen}$이 실제 이상치를 충분히 커버하지 못할 수 있음
3. **RTM/EAG의 훈련 데이터 과적합**: 복잡한 모델에서 훈련셋에 과적합 발생 가능
4. **계산 비용**: 500회 BO 탐색 수행으로 대규모 데이터셋에서 시간 소요 (50,000개 이상 데이터셋은 실험 제외)
5. **추가 하이퍼파라미터**: RTM의 $\tau$, EAG의 $s_{thr}$ 설정에 민감성 존재 (Figure 10 참조)
6. **일부 UAD 알고리즘 호환성**: DROCC 등 일부 알고리즘에서는 NPD 성능이 제한적 (Table 8)

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 이론적 기반 — FPR+FNR 오류율 상한 (Theorem 4)

**Theorem 4**는 NPD를 사용한 AutoUAD의 일반화 오류에 대한 이론적 상한을 제시한다:

```math
FPR + FNR \leq \kappa - \frac{2\Delta}{c\sqrt{N}}\sqrt{\mathcal{V}_{NPD}(\mathcal{M}, \mathcal{X})} + \frac{\sqrt{2}}{2}\sqrt{D_{KL}(\mathcal{D}_{gen} \| \mathcal{D}'_1)} + \hat{\mathfrak{R}}_{\mathcal{X}_{val}}(\mathcal{F}_{\mathcal{M}}) + \hat{\mathfrak{R}}_{\mathcal{X}_{gen}}(\mathcal{F}_{\mathcal{M}}) + 6\sqrt{\frac{\log\frac{2}{\delta}}{N}} 
```

이 수식이 일반화 성능에 미치는 함의:

- **$\mathcal{V}_{NPD}$ 최대화** → $FPR + FNR$ 감소 → 미지 데이터에 대한 탐지 오류율 감소
- **$D_{KL}(\mathcal{D}_{gen} \| \mathcal{D}'_1)$ 감소** → 생성 분포가 실제 이상 분포에 가까울수록 오류율 감소
- **Rademacher 복잡도 $\hat{\mathfrak{R}}$ 감소** → 모델 복잡도 제어로 일반화 향상
- **$N$ 증가** → $O(1/\sqrt{N})$ 속도로 오류율 상한 감소

### 3.2 NPD가 일반화를 돕는 메커니즘

**Theorem 1 (엔트로피 보장):**

```math
H(\boldsymbol{x}_{gen}) > H(\boldsymbol{x}_{trn}) 
```

$\mathcal{X}\_{gen}$은 동일한 분산을 가지는 모든 분포 중 엔트로피(다양성)가 최대인 가우시안 분포에서 생성된다. 이는 $\mathcal{X}_{gen}$이 훈련 데이터보다 다양한 공간을 커버함을 보장한다.

**기하학적 해석:**

$\mathcal{X}\_{gen}$의 외곽은 $\mathbb{R}^d$에서 $\mathcal{X}_{trn}$의 평균을 중심으로 하는 초구(hypersphere) $\mathcal{S}^{d-1}$을 형성한다. 정상 데이터는 상관관계가 있어 초구의 일부만 점유하고, 나머지 공간은 잠재적 이상치 $\mathcal{D}'_1$로 커버될 수 있다 (Figure 4 시각화).

**Theorem 2 (NPD 상한):**

```math
\mathcal{V}_{NPD}(\mathcal{M}, \mathcal{X}) \leq \frac{\Delta + \Delta'}{(\text{Var}(\boldsymbol{s}_{val}) + \text{Var}(\boldsymbol{s}_{gen})) + \epsilon/2} 
```

여기서 $\Delta$와 $\Delta'$는 정상/이상 데이터 간 평균 점수의 제곱 차이다. NPD를 최대화하면 정상-이상 간 점수 간격이 커지므로 미지 데이터에 대한 분리 성능이 향상된다.

### 3.3 과적합 방지 메커니즘

NPD는 훈련 데이터 $\mathcal{X}\_{trn}$과 독립적인 $\mathcal{X}\_{val}$과 $\mathcal{X}_{gen}$을 사용하므로:
- 훈련셋에 대한 과적합 위험 없음
- 다양한 데이터 관점을 통한 평가로 편향 감소

**Theorem 3**는 $\mathcal{X}\_{trn}$과 $\mathcal{X}_{gen}$의 분포 간 KL 발산 상하한을 제공하여, 생성 데이터가 실제 훈련 분포와 적절히 다름을 보장한다:

$$D_{KL}(p_{gen} \| p_{trn}) \leq \frac{1}{2T}\left(d\log\bar{\lambda} + \underline{\lambda}^{-1}\bar{\phi}^2 + d\underline{\lambda}^{-1}\right) - \frac{d}{2} + \varepsilon$$

---

## 4. 앞으로의 연구에 미치는 영향과 고려 사항

### 4.1 연구에 미치는 영향

**① AutoML의 비지도 학습 영역 확장**

AutoUAD는 기존 AutoML이 지도학습에 집중된 한계를 극복하고, 비지도 이상 탐지라는 중요한 분야에 자동화된 파이프라인을 제시한다. 이는 AutoSC(Fan et al., 2022)의 클러스터링 도메인 확장과 유사한 방향으로, 비지도 학습 전반의 AutoML 연구를 촉진할 것이다.

**② 레이블 없는 모델 평가의 새로운 패러다임**

NPD의 핵심 아이디어—*가우시안 분포로 생성된 프록시 데이터를 활용한 성능 대리 지표*—는 다른 비지도 학습 태스크(클러스터링, 표현 학습, 생성 모델 등)에서의 평가 방법론에도 영향을 줄 수 있다.

**③ 이론적 기여의 파급 효과**

Rademacher 복잡도를 활용한 FPR+FNR 상한 분석은 비지도 이상 탐지 분야에서 이론적으로 엄밀한 분석의 기준을 제시한다.

**④ 실용적 파급 효과**

제조업 품질 관리, 의료 진단, 사이버보안 등 레이블 획득이 어려운 실제 응용 분야에서 UAD의 실용성을 크게 향상시킬 수 있다.

### 4.2 앞으로 연구 시 고려할 점

**① 더 정교한 프록시 분포 탐색**

현재 NPD는 등방성 가우시안을 프록시로 사용하나, 실제 데이터가 심하게 다봉분포(multimodal)이거나 매니폴드 구조를 가질 때 한계가 있다. 향후 연구에서는:
- **가우시안 혼합 모델(GMM)** 기반 프록시 데이터 생성
- **Normalizing Flows** 또는 **VAE**를 활용한 더 정확한 정상 분포 모델링
- **Diffusion 모델** 기반 이상치 프록시 생성 (Zhang et al., 2023의 확산 모델 기반 이상 탐지와 연계)

**② 고차원 데이터 및 비정형 데이터 적용**

현재 실험은 38개의 정형(tabular) 데이터셋에 한정된다. 이미지, 시계열, 그래프 데이터에 대한 확장이 필요하다:
- 이미지 UAD: 패치 기반 NPD 또는 특징 공간에서의 NPD 계산
- 시계열 이상 탐지: 시간적 의존성을 고려한 프록시 데이터 생성

**③ 계산 효율성 개선**

500회 BO 탐색은 대규모 데이터셋과 복잡한 딥러닝 모델에서 계산 비용이 크다. 고려할 방향:
- **조기 종료(Early Stopping) 전략** 통합 (단, UAD 특성상 inlier priority 가정 불가)
- **Hypernetwork** 기반 효율적 탐색 (Ding et al., 2024 참고)
- **Multi-fidelity BO**: 소규모 서브셋으로 먼저 평가 후 유망한 후보만 전체 데이터로 평가

**④ NPD의 비단조성 문제 해결**

NPD가 최고값임에도 최적 모델이 아닌 경우가 존재하는 문제를 해결하기 위해:
- **앙상블 기반 NPD**: 여러 $\mathcal{X}_{gen}$ 샘플의 평균 NPD 사용
- **Confidence-weighted NPD**: 불확실성을 고려한 가중치 적용
- **NPD + 내부 지표 혼합**: RTM/EAG와의 앙상블

**⑤ 도메인 적응 및 분포 이동 대응**

훈련 분포와 테스트 분포 간 차이(distribution shift)가 클 때의 견고성 강화:
- **도메인 일반화** 관점에서의 NPD 재해석
- 점진적 데이터 이동(gradual drift)을 다루는 적응형 AutoUAD

**⑥ 2020년 이후 최신 연구와의 비교**

---

## 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 방법 | UAD 적용 여부 | 레이블 불필요 | 이론 보장 | AutoUAD 대비 |
|------|------|--------------|-------------|----------|------------|
| Zhao et al. (2021) - MetaOD (NeurIPS 2021) | 메타학습 기반 이상 탐지 모델 선택 | △ (transductive) | ✗ (역사적 레이블 필요) | ✗ | 역사적 데이터 의존, 도메인 갭 취약 |
| Zhao et al. (2022) - ICDM 2022 | 메타학습 기반 outlier 모델 선택 | △ (transductive) | ✗ | ✗ | 동일한 메타학습 한계 |
| Ding et al. (2022) - NeurIPS 2022 | Hyper-ensemble (HYPER) | ✗ (outier detection) | △ | ✗ | 앙상블 가중치 추가 하이퍼파라미터 |
| Huang et al. (2024) - KDD 2024 (EntropyStop) | Loss entropy 기반 조기 종료 | ✗ (inlier priority 가정) | ✓ | ✗ | UAD에서 inlier priority 가정 위반 |
| Ding et al. (2024) - KDD 2024 | Hypernetwork 기반 빠른 모델 선택 | △ | ✗ (레이블 기반 메타 훈련) | ✗ | 레이블 의존성 |
| Zhao & Akoglu (2024) - HPOD (AutoML) | Unsupervised outlier detection HPO | △ (transductive) | ✓ | ✗ | transductive, inductive UAD와 구분 |
| **AutoUAD (2025)** | NPD + Bayesian Optimization | ✓ (inductive UAD) | ✓ | ✓ | 완전 비지도, 이론 보장, 범용성 |

**주요 차별점:**

1. **Transductive vs. Inductive**: 기존 연구들은 대부분 훈련셋 내 이상치 탐지(transductive)에 초점. AutoUAD는 정상 데이터만으로 학습하여 미지의 새 이상치를 탐지하는 inductive 설정을 다룸

2. **완전 비지도**: MetaOD, HPOD 등은 역사적 레이블 데이터 필요. AutoUAD는 어떠한 레이블도 불필요

3. **이론적 보장**: 대부분의 경쟁 방법들이 경험적 검증에 의존하는 반면, AutoUAD는 Rademacher 복잡도 기반의 엄밀한 오류율 상한을 제공

4. **알고리즘 독립성**: 특정 UAD 알고리즘에 종속되지 않는 범용 프레임워크

---

## 참고 자료

**주요 참고 논문 (본 PDF에서 직접 인용):**

1. **Wei Dai and Jicong Fan. "AutoUAD: Hyper-parameter Optimization for Unsupervised Anomaly Detection." ICLR 2025.** (본 논문)
2. Yue Zhao, Ryan Rossi, and Leman Akoglu. "Automatic unsupervised outlier model selection." NeurIPS 2021.
3. Yue Zhao, Sean Zhang, and Leman Akoglu. "Toward unsupervised outlier model selection." ICDM 2022.
4. Xueying Ding, Lingxiao Zhao, and Leman Akoglu. "Hyperparameter sensitivity in deep outlier detection." NeurIPS 2022.
5. Yihong Huang et al. "Entropystop: Unsupervised deep outlier detection with loss entropy." KDD 2024.
6. Xueying Ding, Yue Zhao, and Leman Akoglu. "Fast unsupervised deep outlier model selection with hypernetworks." KDD 2024.
7. Yue Zhao and Leman Akoglu. "HPOD: Hyperparameter optimization for unsupervised outlier detection." AutoML 2024.
8. Jasper Snoek, Hugo Larochelle, and Ryan P. Adams. "Practical bayesian optimization of machine learning algorithms." NeurIPS 2012.
9. James Bergstra et al. "Algorithms for hyper-parameter optimization." NeurIPS 2011.
10. Lukas Ruff et al. "Deep one-class classification." ICML 2018. (DeepSVDD)
11. Dazhi Fu, Zhao Zhang, and Jicong Fan. "Dense projection for anomaly detection." AAAI 2024. (DPAD)
12. Jicong Fan et al. "A simple approach to automated spectral clustering." NeurIPS 2022. (AutoSC)
13. Peter L. Bartlett and Shahar Mendelson. "Rademacher and gaussian complexities." JMLR 2002.
14. Songqiao Han et al. "ADBench: Anomaly detection benchmark." NeurIPS 2022.
