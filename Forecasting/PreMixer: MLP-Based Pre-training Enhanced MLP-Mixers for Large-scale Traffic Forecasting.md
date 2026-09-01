# PreMixer: MLP-Based Pre-training Enhanced MLP-Mixers for Large-scale Traffic Forecasting

> **참고 자료:**
> - 원문 논문: Zhang et al., "PreMixer: MLP-Based Pre-training Enhanced MLP-Mixers for Large-scale Traffic Forecasting," arXiv:2412.13607v1, 2024.
> - LargeST 데이터셋: Liu et al., "LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting," NeurIPS 2024.
> - STEP: Shao et al., "Pre-training Enhanced Spatial-Temporal Graph Neural Network for Multivariate Time Series Forecasting," KDD 2022.
> - STD-MAE: Gao et al., "Spatio-Temporal-Decoupled Masked Pre-training for Traffic Forecasting," arXiv:2312.00516, 2023.
> - TSMixer: Chen et al., "TSMixer: An All-MLP Architecture for Time Series Forecasting," arXiv:2303.06053, 2023.
> - MLP-Mixer (Vision): Tolstikhin et al., "MLP-Mixer: An All-MLP Architecture for Vision," NeurIPS 2021.
> - PatchTST: Nie et al., "A Time Series is Worth 64 Words," arXiv:2211.14730, 2022.
> - SimST: Liu et al., "Do We Really Need Graph Neural Networks for Traffic Forecasting?" arXiv:2301.12603, 2023.
> - NexuSQN: Nie et al., "Contextualizing MLP-Mixers Spatiotemporally for Urban Data Forecast at Scale," arXiv:2307.01482, 2023.
> - MAE (Vision): He et al., "Masked Autoencoders Are Scalable Vision Learners," CVPR 2022.

---

## 1. Executive Summary (10문장 이내)

PreMixer는 대규모 교통 네트워크에서 정확하고 효율적인 교통 흐름 예측을 위해 설계된 **완전 MLP 기반** 프레임워크이다.  
기존 STGNNs(시공간 그래프 신경망)과 Transformer 기반 모델들은 노드 수 증가에 따라 계산 복잡도가 폭발적으로 증가하여 대규모 네트워크에 적용이 어렵다는 한계가 있었다.  
PreMixer는 MLP-Mixer 구조를 기반으로 한 예측 모델과, 패치 기반 MLP 사전학습 모델(PIEncoder)을 결합하여 이 문제를 해결한다.  
PIEncoder는 장기 역사 데이터를 패치로 분할하고 마스킹된 오토인코딩(masked autoencoding) 방식으로 학습하며, 각 패치를 독립적으로 처리하여 추론 단계에서 단기 입력만으로도 풍부한 문맥 표현을 생성한다.  
또한 사전 정의된 그래프 없이도 시공간 이질성(spatiotemporal heterogeneity)을 포착하기 위해 시공간 위치 인코딩(STPE)과 학습 가능한 노드 임베딩을 통합한다.  
보완적 대조 학습(complementary contrastive learning)을 통해 인접 시계열 간 관계를 강화한다.  
CA(8,600개 센서) 등 4개의 대규모 실제 데이터셋에서 검증한 결과, PreMixer는 경쟁 모델 대비 훈련 시간이 최대 수백 배 빠르면서도 최고 수준의 예측 정확도를 달성한다.  
전이 학습(transfer learning) 실험에서도 소스-타깃 데이터셋 간 효과적인 특징 전이가 가능함을 보인다.  
이 연구는 단순성과 효율성을 동시에 추구하는 MLP 기반 접근법이 복잡한 딥러닝 모델을 대체할 수 있음을 실증한다.

---

### 1-1. 연구의 목적과 필요성

| 구분 | 내용 |
|------|------|
| **배경** | 도시 간 대규모 교통 예측 수요 증가 (p.1–2) |
| **기존 모델의 한계** | STGNNs: 노드 수 증가 시 계산 복잡도 $O(N^2)$ 수준으로 폭발적 증가; Transformer: 입력 시계열 길이에 비례한 메모리·시간 증가 (p.2) |
| **장기 시간 패턴의 중요성** | 교통 데이터는 주기성·계절성 등 장기 패턴을 포함하지만, 기존 모델들은 단기 입력에만 의존 (Fig.1(b), p.2) |
| **연구 목적** | MLP만으로 대규모 교통 예측에서 정확도와 효율성을 동시에 달성하는 프레임워크 개발 |

> **💡 용어 설명**
> - **STGNNs (Spatio-Temporal Graph Neural Networks):** 시공간 그래프 신경망. 교통 센서 간 공간적 관계를 그래프로 모델링하고, 시간적 변화를 함께 학습하는 모델군.
> - **Transformer:** 셀프 어텐션(Self-attention) 메커니즘을 핵심으로 하는 딥러닝 아키텍처. 시퀀스 길이에 따라 계산량이 제곱으로 증가하는 단점이 있음.

---

## 2. 핵심 주장과 근거 표

| 핵심 주장 | 근거 | 위치 |
|-----------|------|------|
| MLP만으로 STGNNs에 준하는 예측 성능 달성 가능 | CA 데이터셋(8,600 노드)에서 모든 베이스라인 능가 | Table II, p.10 |
| PIEncoder가 계산 효율성을 크게 향상 | PreMixer 훈련시간: CA에서 92초/에폭 vs DCRNN 4,995초/에폭 | Table V, p.12 |
| 패치 독립 임베딩으로 예측 단계에서 장기 입력 불필요 | 패치별 가중치 공유 및 독립 처리 구조 | p.5–7 |
| STPE + 노드 임베딩이 예측 성능을 유의미하게 개선 | w/o STPE 대비 SD MAE 18.96→18.02 향상 | Table IV, p.11 |
| 보완적 대조 학습이 표현 품질 향상 | w/o CL 대비 SD MAE 18.20→18.02 향상 | Table IV, p.11 |
| 전이 학습 가능성 확인 | CA→SD 전이 시 SD 단독 학습과 유사한 성능 | Table III, p.11 |
| 사전 정의 그래프 없이 공간 의존성 포착 | STPE와 노드 임베딩만으로 공간 구조 표현 | p.7 |

---

## 2-1. 상세 설명

### 해결하고자 하는 문제

1. **대규모 확장성 (Scalability):** 기존 STGNNs, Transformer는 수천 개 이상의 노드를 가진 교통망에 적용 불가 (메모리 부족, 훈련 시간 과다)
2. **장기 시간 패턴 학습:** 기존 Transformer 기반 사전학습 모델(STEP, STD-MAE)은 예측 단계에서도 장기 입력 필요 → 대규모 네트워크에 비효율
3. **시공간 이질성(Spatiotemporal Heterogeneity):** 동일 센서라도 시간대·공간 위치에 따라 서로 다른 패턴을 보이는 문제

> **💡 용어 설명**
> - **시공간 이질성 (Spatiotemporal Heterogeneity):** 서로 다른 위치의 센서들이 서로 다른 시간 패턴을 보이는 현상. 예: 고속도로 진입부와 도심 교차로는 혼잡 패턴이 다름.

---

### 제안하는 방법 (수식 포함)

#### ① 문제 정의 (Eq. 1, p.4)

$$F\left(\left[X_{t-(T+1):t}\right]\right) = \left[X_{t:t+T}\right] = \mathcal{Y}$$

- $X_{t-(T+1):t} \in \mathbb{R}^{T \times N \times C}$: $T$-스텝 과거 교통 데이터
- $\mathcal{Y} \in \mathbb{R}^{T \times N \times C}$: 예측 대상 미래 데이터
- $N$: 센서(노드) 수, $C$: 센서당 특징 수

---

#### ② PIEncoder 패치 임베딩 (Eq. 2, p.5)

$$\boldsymbol{z}_p^{(i,n)} = \text{PIEncoder}\left(\boldsymbol{x}_p^{(i,n)}\right)$$

- $\boldsymbol{x}_p^{(i,n)} \in \mathbb{R}^{P}$: $i$번째 패치, $n$번째 노드의 입력 패치 ($P = LC$)
- $\boldsymbol{z}_p^{(i,n)} \in \mathbb{R}^{D}$: 출력 표현 벡터
- $L$: 패치 길이, $T_p = T_{\text{long}} / L$: 총 패치 수
- $i = 1,\ldots,T_p$, $n = 1,\ldots,N$

> **💡 용어 설명**
> - **패치 (Patch):** 시계열을 일정 길이로 잘라낸 조각. 예: 12개 타임스텝(3시간)을 하나의 패치로 처리.
> - **채널(노드) 독립성 (Channel Independence):** 각 센서(노드)를 독립적으로 처리하여 센서 간 상호작용을 임베딩 단계에서 배제하는 방식.

---

#### ③ 재구성 손실 (Eq. 3, p.6)

$$\mathcal{L}_{\text{recon}} = \sum_{i=1}^{T_p} \sum_{n=1}^{N} \left\| \boldsymbol{x}_p^{(i,n)} - \hat{\boldsymbol{x}}_p^{(i,n)} \right\|_2^2$$

- $\hat{\boldsymbol{x}}_p^{(i,n)} = W\boldsymbol{z}^{(i,n)}$: 재구성된 패치 ($W \in \mathbb{R}^{P \times D}$)
- 마스킹된 패치와 비마스킹 패치 모두 재구성 (기존 방법과 차별화)

---

#### ④ 대조 학습 손실 (Eq. 4–5, p.6)

**소프트맥스 유사도 확률:**

$$p\left((i, i'), n\right) = \frac{\exp\left(\boldsymbol{z}_1^{(i,n)} \circ \boldsymbol{z}_1^{(i',n)}\right)}{\sum_{s=1, s\neq i}^{2T_p} \exp\left(\boldsymbol{z}_1^{(i,n)} \circ \boldsymbol{z}_1^{(s,n)}\right)}$$

**대조 손실:**

$$\mathcal{L}_{\text{CL}} = \frac{1}{2T_p N} \sum_{i=1}^{2T_p} \sum_{n=1}^{N} -\log p\left((i, i+T_p), n\right)$$

- $\boldsymbol{z}_1^{(i,n)}$: PIEncoder 첫 번째 레이어의 패치 표현
- $\circ$: 내적(dot product) — 두 벡터 간 유사도 측정
- 보완적 마스킹(complementary masking): 마스크 비율 $m=50\%$로, $m \odot x_p$와 $(1-m) \odot x_p$를 두 뷰(view)로 사용

**최종 손실 (Eq. 6, p.6):**

$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{CL}}$$

> **💡 용어 설명**
> - **대조 학습 (Contrastive Learning):** 유사한 샘플(positive pair)은 가깝게, 다른 샘플(negative pair)은 멀게 표현 공간에서 배치되도록 학습하는 자기지도학습 방식.
> - **보완적 마스킹 (Complementary Masking):** 하나의 시계열을 서로 보완적으로 마스킹하여 두 뷰를 생성. 별도 데이터 증강 없이 positive pair를 만드는 방법.

---

#### ⑤ 시공간 위치 인코딩 STPE (Eq. 7, p.7)

$$\begin{cases} U_{\text{pos}}(t, n, 2i) = \sin\left(t / 10000^{4i/C}\right) \\ U_{\text{pos}}(t, n, 2i+1) = \cos\left(t / 10000^{4i/C}\right) \\ U_{\text{pos}}(t, n, 2j+D/2) = \sin\left(n / 10000^{4j/C}\right) \\ U_{\text{pos}}(t, n, 2j+1+D/2) = \cos\left(n / 10000^{4j/C}\right) \end{cases}$$

- $t$: 시간 인덱스, $n$: 공간(노드) 인덱스
- $i, j$: 정수, $i, j \in [0, C/4]$
- 앞 절반 차원: 시간 위치, 뒤 절반 차원: 공간 위치 인코딩
- 학습 파라미터 추가 없음

---

#### ⑥ 입력 레이어 (Eq. 8–9, p.7)

$$\mathbf{H}_c = \text{Reshape}\left(\text{MLP}\left(X_{t-T+1:t} \| \mathbf{U}_{\text{pos}}\right)\right)$$

$$\mathbf{H}^{(0)} = \text{MLP}(\mathbf{Z}_2) + \mathbf{H}_c$$

- $\|$: 채널 방향 연결(concatenation)
- $\mathbf{Z}_2 \in \mathbb{R}^{N \times D}$: PIEncoder 두 번째 레이어 출력 표현

---

#### ⑦ TemporalMixer (Eq. 10, p.7)

$$\mathbf{H}^{(1)} = \mathbf{H}^{(0)} + \sigma\left(\text{LayerNorm}\left(\mathbf{H}^{(0)}\right)\mathbf{W}_1 + \mathbf{b}_1\right)\mathbf{W}_2 + \mathbf{b}_2$$

- $\sigma$: GELU 활성화 함수
- $\mathbf{W}_1, \mathbf{W}_2, \mathbf{b}_1, \mathbf{b}_2$: 학습 가능한 MLP 파라미터
- 모든 노드(채널)에 공유 적용

> **💡 용어 설명**
> - **GELU (Gaussian Error Linear Unit):** 입력값에 가우시안 누적분포를 곱한 활성화 함수. ReLU보다 부드러운 비선형성을 제공.
> - **LayerNorm:** 레이어 정규화. 각 샘플의 특징 차원에 걸쳐 평균과 분산을 정규화하여 학습 안정성을 향상.

---

#### ⑧ SpatialMixer (Eq. 11–12, p.7–8)

**기본 공간 혼합:**

$$\mathbf{H}^{(l+1)} = \sigma\left(\mathbf{W}_{\text{channel}}^{(l)}\mathbf{H}^{(l)} + \mathbf{b}\right), \quad l \in \{1,\ldots,L\}$$

**구조적 SpatialMixer (NexuSQN 기반, Eq. 12):**

$$\mathbf{m}_{i\leftrightarrow j}^{(l+1)} = \Psi\left(\mathbf{h}_i^{(l)} \| \mathbf{h}_j^{(l)}\right)\left[\mathbf{h}_i^{(l)} \| \mathbf{h}_j^{(l)}\right]$$

$$\mathbf{m}_i^{(l+1)} = \sum_{j=1}^{N} \mathbf{m}_{i\leftrightarrow j}^{(l+1)}$$

$$\mathbf{h}_i^{(l+1)} = \sigma\left(\Theta \mathbf{h}_i^{(l)} + \mathbf{m}_i^{(l+1)}\right)$$

- $\mathbf{W}_{\text{channel}} \in \mathbb{R}^{N' \times N}$: 채널 혼합 파라미터
- $\mathbf{h}_i^{(l)}$: $l$번째 레이어에서 노드 $i$의 표현
- $\Psi$: 시간 문맥화 함수 (temporal contextualization function)
- $\Theta$: 피드포워드 가중치

---

#### ⑨ 회귀 손실 (Eq. 13, p.8)

$$\mathcal{L}_{\text{regression}} = \frac{1}{TNC} \sum_{j=1}^{T} \sum_{i=1}^{N} \sum_{k=1}^{C} \left|\hat{Y}_{ijk} - Y_{ijk}\right|$$

- $\hat{Y}$: 예측값, $Y$: 실제값
- MAE(평균 절대 오차) 기반 손실

---

### 모델 구조

```
[Stage 1: Pre-training]
장기 시계열 (1주일) → 패치 분할 → 마스킹 → PIEncoder (2-layer MLP)
→ 재구성 레이어 + 보완적 대조 학습 → L_recon + L_CL

[Stage 2: Forecasting]
단기 시계열 + STPE → MLP 임베딩 (H_c)
PIEncoder (고정) → Z2 → MLP 투영
H^(0) = MLP(Z2) + H_c
→ TemporalMixer (L2 MLP, 노드 간 공유)
→ SpatialMixer (채널 혼합, 시간 공유, STPE+노드 임베딩 통합)
→ MLP 출력층 → 예측값 Y_hat
```

---

### 성능 향상 및 한계

**성능 향상:**
- CA 데이터셋(8,600 노드)에서 모든 베이스라인 능가 (Table II)
- 훈련 속도: DCRNN 대비 약 54배 빠름 (CA: 92초 vs 4,995초/에폭)
- 메모리 효율: 배치 크기 64로 전체 데이터셋 처리 가능 (다른 모델들은 OOM 발생)

**한계:**
- 소규모 데이터셋(SD)에서 DGCRN, D²STGNN보다 약간 낮은 성능
- PIEncoder가 교통 흐름 피크(첨두 시간) 패턴을 완전히 학습하지 못함 (Fig.6 참조)
- 기상 조건, 특별 이벤트 등 외부 요인 미통합
- 단일 교통 특징(유량)에 집중; 속도·점유율 등 다변수 동시 예측 검증 미흡

---

## 3. 각 주장별 페이지/Figure/Table 번호

| 주장 | 근거 위치 |
|------|-----------|
| 기존 STGNNs의 대규모 확장성 한계 | p.2, Fig.1(a) |
| PreMixer 전체 구조 | p.4–5, Fig.2 |
| PIEncoder 구조 및 사전학습 과정 | p.5–6, Fig.3 |
| STPE 수식 | p.7, Eq.7 |
| TemporalMixer/SpatialMixer 수식 | p.7–8, Eq.10–12 |
| 데이터셋 통계 | p.8, Table I, Fig.4 |
| 성능 비교 (RQ1) | p.10, Table II |
| 전이 학습 결과 (RQ2) | p.11, Table III |
| 어블레이션 스터디 (RQ3) | p.11, Table IV |
| 효율성 비교 (RQ4) | p.12, Table V, Fig.5 |
| 재구성 및 예측 시각화 (RQ5) | p.12–13, Fig.6–7 |

---

## 4. 저자 보고 결과 vs. 해석 분리

### 저자가 직접 보고한 결과

- **Table II:** GLA 데이터셋에서 PreMixer MAE=16.75(Horizon 3), 20.30(Horizon 6), 25.15(Horizon 12). CA 데이터셋에서 MAE=15.65, RMSE=25.97 (Horizon 3)으로 모든 베이스라인 능가.
- **Table V:** PreMixer CA 데이터셋 훈련 시간 92초/에폭, 추론 24초. DCRNN은 4,995초/에폭.
- **Table IV:** w/o Pre-training 시 SD MAE 18.39 → PreMixer 18.02로 개선.
- **Fig.6:** PIEncoder가 패치를 독립적으로 재구성할 수 있으나, 피크 위치 학습에 한계 존재.

### 저자 결과에 대한 해석 (검토자 관점)

- **긍정적:** 계산 효율성 향상은 수치적으로 명확하며, 대규모 데이터에서의 적용 가능성은 유효한 기여임. MLP 기반 사전학습이 Transformer 기반 대비 추론 단계에서 장기 입력이 불필요하다는 설계 철학은 실용적.
- **주의 필요:** 소규모 데이터셋(SD)에서 DGCRN, D²STGNN 대비 성능이 낮음 → 논문의 "comparable SOTA" 주장은 대규모 데이터에 한정된 것임. 어블레이션 개선 폭(예: MAE 18.39→18.02, 약 2%)이 통계적 유의성 검정 없이 보고됨.

---

## 5. 통계적으로 취약한 부분 및 비교 불가능한 수치 ⚠️

| 항목 | 문제점 |
|------|--------|
| **통계적 유의성 검정 부재** | 모든 성능 비교에서 p-value, 신뢰 구간, 표준편차가 보고되지 않음. 개선 폭이 통계적으로 유의한지 불명확. |
| **GLA/CA 데이터 TSMixer RMSE 이상값** | Table II에서 TSMixer GLA RMSE=224.82, 229.86 등 비정상적 수치 기록 → 논문은 "MAE 과적합"으로 설명하나, 수치 신뢰성에 의문 ⚠️ |
| **비교 불가능한 CA 결과** | AGCRN, DSTAGNN, DGCRN, D²STGNN은 CA 데이터셋에서 메모리 부족(OOM)으로 실행 불가 → PreMixer와 직접 비교 불가 ⚠️ |
| **전이 학습 베이스라인 부재** | 전이 학습 실험(Table III)에서 다른 모델의 전이 학습 결과와 비교하지 않아 상대적 우위 판단 불가 ⚠️ |
| **단일 시드 실험 여부** | 랜덤 시드 다중 실행 여부 미명시 → 결과 재현성 불명확 |
| **어블레이션 개선 폭 미미** | w/o CL vs PreMixer: SD MAE 18.20→18.02 (약 1% 향상) — 실용적 유의미성 제한적 |
| **2019년 단일 연도 데이터** | 5년치 데이터(2017–2021) 중 2019년만 사용 → 계절·연도별 일반화 검증 미흡 |

---

## 6. 논문이 답하지 않는 질문

1. **하이퍼파라미터 민감도:** 패치 길이 $L$, 마스킹 비율 $m=50\%$, 잠재 표현 차원 $D=96$ 선택의 근거와 민감도 분석 미제공.
2. **속도·점유율 예측:** 실험이 교통 유량(flow)에만 집중. 속도(speed), 점유율(occupancy) 등 다른 특징에 대한 일반화 검증 없음.
3. **장기 예측(Long-term Forecasting):** 12 스텝(3시간)까지만 검증. 24시간, 1주일 등 장기 예측 성능 미검증.
4. **실시간 적용(Online Learning):** 실시간 데이터 스트림에서의 점진적 학습 가능 여부 미검토.
5. **이상 교통 상황(Anomaly) 처리:** 사고, 공사, 기상 이변 등 비정상 상황에서의 예측 성능 미평가.
6. **타 도시·국가 데이터 일반화:** 미국 캘리포니아 PeMS 데이터만 사용. 다른 지역·나라 교통 데이터에 대한 일반화 검증 없음.
7. **사전학습 데이터 양 민감도:** 1주일(56 패치)을 사용했는데, 더 짧거나 긴 기간 사용 시 성능 변화 미분석.
8. **GNN과의 결합 가능성:** 순수 MLP가 아닌 GNN과 결합 시 성능 향상 가능성 미탐색.
9. **설명 가능성(Explainability):** 모델의 예측 근거(어떤 시간·공간 패턴에 의존하는지) 설명 없음.
10. **에너지 소비:** 훈련/추론 시 전력 소비량 미보고.

---

## 7. 가장 중요한 그림 5개 해석

### Fig.1(a) — 모델 성능 vs. 계산 비용 비교 (p.1)

GLA 데이터셋(3,000개 이상 센서)에서 훈련 시간(x축)과 MAE(y축)를 비교한 산점도. PreMixer는 좌하단(낮은 MAE + 짧은 훈련 시간)에 위치하여 **효율성과 정확도의 최적 균형**을 시각적으로 입증. DSTAGNN, D2STGNN 등은 우상단에 위치하며 높은 비용에도 낮은 정확도를 보임. **해석:** 복잡도 증가가 항상 성능 향상으로 이어지지 않음을 실증.

---

### Fig.2 — PreMixer 전체 구조 개요 (p.5)

두 단계(Stage 1: Pre-training, Stage 2: Prediction)와 PreMixer 내부 구조를 한 그림으로 표현. 왼쪽의 PIEncoder가 마스킹된 패치를 처리하고, 오른쪽의 PreMixer가 단기 입력과 PIEncoder 표현을 융합하는 흐름을 보여줌. **해석:** 사전학습 단계와 예측 단계의 분리가 핵심 설계 철학임을 명확히 보여주며, 두 단계가 어떻게 연결되는지(Z2를 통한 feature fusion) 이해하는 데 필수적.

---

### Fig.3 — PIEncoder 및 보완적 대조 학습 (p.6)

왼쪽: PIEncoder의 마스킹-임베딩-재구성 파이프라인. 오른쪽: 보완적 마스킹으로 anchor/positive/negative를 생성하는 대조 학습 구조. **해석:** 핵심 혁신인 '패치 독립 처리'와 '별도 데이터 증강 없는 대조 학습'이 어떻게 구현되는지를 보여주는 그림. Transformer 기반 방법들이 패치 간 어텐션을 사용하는 반면, PIEncoder는 각 패치를 독립적으로 처리한다는 점이 대규모 적용의 핵심 이유임을 이해할 수 있음.

---

### Fig.5 — CA 데이터셋 파라미터 수·추론 속도·MAE 비교 버블 차트 (p.12)

x축: 추론 속도(초), y축: MAE, 버블 크기: 파라미터 수. PreMixer는 가장 작은 MAE(약 18.4)와 가장 빠른 추론 속도(24초)를 가지며, 버블 크기(3.3M 파라미터)도 합리적. **해석:** PreMixer가 파라미터 효율성, 추론 효율성, 예측 정확도 세 가지를 동시에 달성함을 시각적으로 입증. LSTM은 빠르지만 MAE가 높고, DGCRN 등은 느리고 CA에서 실행 불가.

---

### Fig.7 — 실제 교통 흐름 예측 vs. 실측값 비교 (p.13)

SD, GBA, CA 데이터셋에서 2019년 10월 21일~27일(1주일)의 예측값과 실제값 비교. 주중/주말 패턴 구분, 금요일 아침 첨두 특이 패턴 감지 등 세부 시간 패턴 포착을 시각적으로 보여줌. **해석:** CA 데이터셋(일요일)에서 일부 정확도 저하가 관찰됨 — 저자가 직접 인정한 한계. 주기성 패턴은 잘 포착하지만, 비정형 패턴(이상치, 급격한 변동)에서 한계를 보임.

---

## 8. 결론 및 후속 연구 방향

### 8-1. 저자 제시 시사점 및 후속 연구 계획 (p.13–14)

**저자 제시 시사점:**
- MLP만으로도 대규모 교통 예측에서 SOTA 수준 달성 가능
- 패치 독립 처리 기반 MLP 사전학습이 Transformer 기반 대비 효율적
- STPE + 노드 임베딩이 그래프 없이도 공간 정보 포착 가능

**저자 제시 후속 연구:**
1. **실제 운영 환경 적용:** 기상, 커뮤니티 활동, 인프라 공사 등 외부 요인 통합 및 온라인 학습 적용
2. **외부 다중 모달 입력 통합:** 다중 소스 데이터(날씨, 이벤트, 지도 등) 동적 융합으로 예측 정확도·유연성 향상

---

### 8-1. 모델의 일반화 성능 향상 가능성 (중점)

**현재 일반화 한계:**
- 단일 지역(캘리포니아), 단일 연도(2019), 단일 특징(유량)에 집중
- 소규모 데이터셋에서 GNN 기반 모델 대비 성능 열위

**일반화 향상 방향:**

| 방향 | 설명 |
|------|------|
| **도메인 적응 (Domain Adaptation)** | 다른 국가·도시 교통 데이터에 사전학습 모델 전이. 현재 전이 학습 실험(Table III)이 같은 지역 내 수행으로 제한됨. |
| **데이터 증강 강화** | 계절·기상 조건별 데이터 증강으로 계절 일반화 향상. 보완적 마스킹 이외 추가 증강 전략 탐색. |
| **멀티태스크 학습** | 유량·속도·점유율을 동시 예측하여 공유 표현의 일반화 향상. |
| **메타 학습 (Meta-Learning)** | 소수 샘플만으로 새로운 지역에 빠르게 적응하는 few-shot 전이 학습 통합. |
| **더 긴 사전학습 기간** | 현재 1주일(56 패치). 1개월·1년 데이터로 사전학습 시 계절성 포착 가능성 탐색. |
| **노이즈 강건성** | 센서 결측·오류 상황에서의 성능 평가 및 강건한 마스킹 전략 설계. |

**핵심 통찰:** 패치 독립 처리 구조는 근본적으로 귀납적 편향(inductive bias)이 적어, 새로운 센서나 지역에 대해 동일한 모델 가중치를 재사용할 수 있는 잠재력이 있음. 그러나 SpatialMixer의 $\mathbf{W}_{\text{channel}} \in \mathbb{R}^{N' \times N}$은 노드 수 $N$에 고정되어 있어, 새로운 노드 수의 데이터셋에 직접 적용이 불가능 — **가장 큰 일반화 병목 지점**.

> **💡 용어 설명**
> - **귀납적 편향 (Inductive Bias):** 모델이 학습 데이터를 넘어 일반화할 때 암묵적으로 가정하는 구조적 전제. 예: GNN은 "연결된 노드는 유사하다"는 편향을 가짐.
> - **메타 학습 (Meta-Learning):** "학습하는 법을 학습"하는 기법. 소수의 새 데이터로 빠르게 새 태스크에 적응 가능.

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 방법 | PreMixer와의 관계 |
|------|------|-----------|------------------|
| **GWNET** (Wu et al.) | 2019 | 적응적 그래프 + 확장 인과 합성곱 | 베이스라인. PreMixer가 CA에서 능가 |
| **STEP** (Shao et al., KDD'22) | 2022 | Transformer 기반 마스킹 사전학습 + GNN | PreMixer의 직접적 영감 소스. 단, STEP은 예측 단계에서도 장기 입력 필요 |
| **PatchTST** (Nie et al.) | 2022 | 채널 독립 패치 Transformer | PIEncoder의 패치 개념 참조. PreMixer는 Transformer 대신 MLP 사용 |
| **STD-MAE** (Gao et al.) | 2023 | 시공간 분리 마스킹 사전학습 | 공간·시간 축을 분리 마스킹. PreMixer보다 복잡하나 세밀한 시공간 분리 |
| **SimMTM** (Dong et al., NeurIPS'24) | 2023 | 매니폴드 학습 기반 마스킹 시계열 | 일반 시계열에 적용. 교통 특화 아님 |
| **NexuSQN** (Nie et al.) | 2023 | 컨텍스트화된 MLP-Mixer | PreMixer의 SpatialMixer 설계에 직접 차용 |
| **SimST** (Liu et al.) | 2023 | GNN 없이 공간 근접성 모델링 | "GNN이 반드시 필요하지 않다"는 논제 공유 |
| **STFT** (Wang et al., 2024) | 2024 | Transformer + 시공간 구성요소 for 대규모 | 유사한 대규모 문제 다루나 Transformer 기반 |
| **RPMixer** (Yeh et al.) | 2024 | 랜덤 프로젝션 레이어 기반 MLP-Mixer | MLP-Mixer 교통 예측의 동시대 연구 |

**PreMixer가 앞으로의 연구에 미치는 영향:**

1. **MLP 기반 설계 패러다임 확산:** "단순하지만 효율적인" MLP-Mixer가 Transformer·GNN을 대체할 수 있다는 실증적 근거 제공
2. **사전학습의 경량화 방향:** 패치 독립 처리라는 설계 원칙이 대규모 시공간 데이터의 사전학습 효율화에 새로운 기준을 제시
3. **그래프 없는 교통 예측의 정당화:** SimST에 이어 PreMixer도 GNN 없이 SOTA를 달성함으로써 이 방향의 연구를 더욱 촉진

**앞으로 연구 시 고려할 점:**

1. **노드 수 가변성 문제 해결:** 현재 SpatialMixer는 고정 노드 수를 가정. 새 센서 추가나 다른 크기 데이터셋에 적응하는 구조 설계 필요 (예: 어텐션 기반 동적 그래프 혼합)
2. **실험 범위 확대:** 단일 지역·연도·특징에서 탈피하여 다양한 교통 네트워크(유럽, 아시아 등)와 다변수 예측(속도, 점유율)으로 검증
3. **통계적 엄밀성 강화:** 성능 비교 시 반드시 다중 시드 실험 + 신뢰 구간 보고
4. **Foundation Model 가능성:** 대규모 교통 데이터로 사전학습된 범용 교통 기반 모델(Traffic Foundation Model) 개발에 PIEncoder 아이디어 활용 가능
5. **온라인 학습 통합:** 실시간 교통 변화에 적응하는 점진적 학습(continual learning) 메커니즘 추가 필요
6. **설명 가능성 연구:** MLP 기반 모델의 예측 근거를 시각화하는 기법(예: SHAP, Grad-CAM 적용) 개발

> **💡 용어 설명**
> - **Foundation Model (기반 모델):** 대규모 데이터로 사전학습된 후 다양한 하위 태스크에 미세조정되는 범용 모델. GPT, BERT 등이 대표적.
> - **Continual Learning (점진적 학습):** 새로운 데이터가 들어올 때 기존 지식을 잊지 않으면서 지속적으로 학습하는 기법.

---

*본 분석은 제공된 논문 원문(arXiv:2412.13607v1)을 기반으로 작성되었으며, 논문에 명시되지 않은 내용은 추론임을 명시하였습니다. 불확실한 사항에 대해서는 의도적으로 답변을 유보하였습니다.*
