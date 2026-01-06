# Shape and Time Distortion Loss for Training Deep Time Series Forecasting Models
***

### 1. 핵심 주장 및 주요 기여 요약

*   **핵심 주장:** 기존의 유클리드 거리 기반 손실 함수(MSE, MAE)는 시계열의 **형상(Shape)**과 **시간적 지연(Temporal Delay)**을 구분하지 못해, 급격한 변화가 있는 구간에서 흐릿한(Blurry) 예측을 생성하거나 시점 오차에 과도한 페널티를 부여합니다. 이를 해결하기 위해 형상과 시간 오차를 분리하여 학습해야 합니다.
*   **주요 기여:**
    1.  **DILATE (DIstortion Loss including shApe and TimE):** 형상 왜곡( $\mathcal{L}\_{shape}$ )과 시간 왜곡( $\mathcal{L}_{temporal}$ )을 명시적으로 분리하여 결합한 새로운 목적 함수 제안.
    2.  **미분 가능성:** 딥러닝 모델의 역전파(Backpropagation) 학습이 가능하도록 **Soft-DTW**를 기반으로 한 미분 가능한 손실 함수 설계.
    3.  **성능 향상:** 비정상 시계열 데이터에서 급격한 변동을 더 정확하게 예측하며, 형상 및 시간 정렬 지표에서 MSE 기반 모델보다 우수한 성능 입증.

***

### 2. 상세 분석: 문제, 제안 방법, 모델, 성능 및 한계

#### 2.1 해결하고자 하는 문제 (Problem Statement)
*   **MSE의 한계 (Euclidean Mismatch):** MSE는 동일한 시점의 값끼리만 비교합니다. 예를 들어, 실제 값보다 1스텝 늦게 급격한 상승을 예측한 경우(Shift), 형상은 완벽하더라도 MSE는 매우 커집니다. 반면, 급격한 변화를 예측하지 않고 평균값으로 뭉개버린 예측(Blurry prediction)이 오히려 더 낮은 MSE를 가질 수 있습니다.
*   **결과:** 모델은 안전한 선택(평균값 예측)을 하도록 학습되어, 시계열 예측의 핵심인 '이벤트 발생 시점'과 '변화의 형태'를 놓치게 됩니다.

#### 2.2 제안하는 방법: DILATE (Proposed Method)
저자들은 두 가지 손실 항을 결합한 **DILATE** 손실 함수를 제안합니다.

$$\mathcal{L}\_{DILATE}(\hat{y}, y) = \alpha \mathcal{L}\_{shape}(\hat{y}, y) + (1-\alpha) \mathcal{L}_{temporal}(\hat{y}, y) $$

*   ** $\mathcal{L}\_{shape}$ (형상 손실):** **Soft-DTW**를 사용합니다. DTW(Dynamic Time Warping)는 두 시계열 간의 최적 정렬 경로(Alignment Path)를 찾아 거리를 계산하므로 시간 축의 왜곡(Shift, Stretch)에 강건합니다. 표준 DTW는 미분이 불가능하므로, 평활화된(Smoothed) Soft-DTW를 사용합니다.

$$\mathcal{L}\_{shape} = \text{soft-DTW}\_{\gamma}(\hat{y}, y) = -\gamma \log \left( \sum_{A \in \mathcal{A}} \exp \left( -\frac{\langle A, \Delta(\hat{y}, y) \rangle}{\gamma} \right) \right)$$
    
여기서 $A$는 정렬 행렬(Alignment Matrix), $\Delta$ 는 비용 행렬(Cost Matrix), $\gamma$는 평활화 파라미터입니다.

*   ** $\mathcal{L}\_{temporal}$ (시간 손실):** Soft-DTW가 찾은 최적의 경로( $A^*\_{\gamma}$ )가 대각선(Diagonal)에서 얼마나 벗어났는지를 측정합니다. 이를 통해 모델이 형상은 맞추되, 시간적 지연이 너무 크지 않도록 제어합니다. (TDI: Temporal Distortion Index 개념 차용)

$$\mathcal{L}\_{temporal} = \langle A^*_{\gamma}, \Omega \rangle $$
    
여기서 $\Omega$는 시간 지연에 대한 페널티 행렬(예: $(i-j)^2$)이며, $A^*\_{\gamma}$는 Soft-DTW의 그라디언트($\nabla_\Delta \text{soft-DTW}$)를 통해 계산됩니다.

#### 2.3 모델 구조 (Model Structure)
이 논문은 DILATE가 **모델 구조에 구애받지 않음(Model-Agnostic)**을 강조합니다. 실험에서는 두 가지 구조를 사용했습니다.
*   **Seq2Seq (Encoder-Decoder):** GRU(Gated Recurrent Unit) 기반의 시퀀스 모델.
*   **MLP (Multi-Layer Perceptron):** 단순 완전 연결 신경망.
*   이 손실 함수는 어떤 딥러닝 시계열 모델(예: Transformer, CNN 등)의 출력단에도 적용 가능합니다.

#### 2.4 성능 향상 (Performance Improvement)
*   **정성적 평가:** 합성 데이터(Step function) 및 실제 데이터(ECG, Traffic) 실험에서, MSE로 학습된 모델은 급격한 변화 구간을 부드럽게(Smoothed) 예측하는 반면, DILATE 모델은 **급격한 변화(Sharp Change)를 정확한 형상으로 예측**했습니다.
*   **정량적 평가:** DTW(형상 오차) 및 TDI(시간 오차) 지표에서 MSE 모델 대비 유의미한 성능 향상을 보였습니다.

#### 2.5 한계 (Limitations)
*   **계산 복잡도:** DTW 계산은 시퀀스 길이 $N$에 대해 $O(N^2)$의 시간/공간 복잡도를 가집니다. 긴 시계열 데이터(Long-term forecasting) 학습 시 메모리와 속도 부담이 큽니다.
*   **하이퍼파라미터 민감도:** $\alpha$ (형상 vs 시간 가중치)와 $\gamma$ (Soft-DTW 평활화 정도) 설정에 성능이 민감할 수 있습니다.
*   **노이즈 민감성:** DTW 특성상 노이즈가 많은 데이터에서 최적 경로가 왜곡될 수 있어, 후속 연구(TILDE-Q 등)에서 이 점이 지적되기도 했습니다.

***

### 3. 모델의 일반화 성능 향상 가능성

이 논문의 접근 방식은 **"구조적 일반화(Structural Generalization)"**를 가능하게 합니다.
*   **비정상성(Non-stationarity) 대응:** 학습 데이터와 테스트 데이터의 이벤트 발생 시점이 정확히 일치하지 않더라도(Time Shift), 모델이 "이벤트의 형상" 자체를 학습하도록 유도합니다. 즉, 데이터의 분포가 시간에 따라 변하더라도 파형의 특징을 잡아내는 능력이 향상됩니다.
*   **과적합 방지:** MSE는 학습 데이터의 노이즈나 미세한 타이밍까지 맞추려다 과적합(Overfitting)될 위험이 있지만, Soft-DTW 기반 손실은 약간의 시간적 불일치를 허용하므로 더 강건한(Robust) 특징을 학습하여 일반화 성능을 높입니다.

***

### 4. 향후 연구에 미치는 영향 및 고려할 점

#### 4.1 영향 (Impact)
*   **손실 함수 설계의 패러다임 전환:** 단순히 값의 차이(Point-wise error)만 줄이는 것에서 벗어나, **시계열의 구조적 유사성(Structural Similarity)**을 학습 목표로 삼는 연구 흐름을 주도했습니다.
*   **미분 가능한 동적 프로그래밍:** 딥러닝 내에서 동적 프로그래밍(Dynamic Programming) 알고리즘을 미분 가능한 모듈로 통합하여 학습시키는 방법론을 확산시켰습니다.

#### 4.2 연구 시 고려할 점 (Future Considerations)
*   **효율성 개선:** $O(N^2)$ 복잡도를 줄이기 위해, 최근의 **Linear Attention**이나 **Sparse DTW** 같은 기법을 결합하여 긴 시퀀스에 적용할 수 있는 경량화된 DILATE 변형이 필요합니다.
*   **다변량 확장:** 단변량(Univariate) 시계열뿐만 아니라, 다변량(Multivariate) 데이터 간의 상관관계까지 고려하는 구조적 손실 함수로의 확장이 필요합니다.

***

### 5. 2020년 이후 관련 최신 연구 비교 분석

2020년 이후, DILATE의 아이디어는 더욱 발전되거나 새로운 아키텍처(Transformer)와 결합되는 양상을 보입니다.

| 구분 | DILATE (2019) | 최신 연구 (2020~2025) | 비교 분석 |
| :--- | :--- | :--- | :--- |
| **접근 방식** | **Soft-DTW + TDI**<br>(형상/시간 분리) | **Transformer 기반** (Informer, Autoformer, PatchTST)<br>**주파수/분해 기반 손실** (Coherent Loss, Tre-Loss) | 최신 모델들은 손실 함수보다는 **모델 구조(Attention)**로 장기 의존성을 해결하려 했으나, 최근 다시 **손실 함수의 중요성**이 부각됨. |
| **주요 모델** | Seq2Seq (RNN/GRU) | **PatchTST (2023), iTransformer (2024)**<br>**Time-o1 (NeurIPS 2025)** | 최신 연구(Time-o1 등)는 DILATE가 **자기상관(Autocorrelation)**이 강한 데이터에서 편향될 수 있음을 지적하며, 라벨 변환(Transformed Label)을 통한 개선을 제안함. |
| **손실 함수 트렌드** | 시간/형상 왜곡 보정 | **TILDE-Q (2022):** 변환 불변(Transformation Invariant) 손실 제안.<br>**Patch-wise Structural Loss (2025):** 패치 단위의 구조적 유사성 학습. | DILATE는 노이즈에 민감할 수 있다는 단점이 지적되어, 최신 연구들은 **노이즈 강건성**과 **스케일 불변성**을 추가한 손실 함수(TILDE-Q 등)로 발전함. |
| **일반화** | 시점 왜곡에 강건함 | **Freq-Domain Loss:** 주파수 도메인에서의 정렬을 통해 일반화 성능 극대화 시도. | 시간 도메인(DTW)뿐만 아니라 **주파수 도메인**까지 활용하여 비정상 데이터의 일반화 성능을 높이는 방향으로 진화 중. |

**결론:** DILATE는 시계열 딥러닝에서 "구조적 손실 함수"의 기초를 닦은 중요한 연구입니다. 최신 연구들은 DILATE의 $O(N^2)$ 비용과 노이즈 민감성을 극복하기 위해 패치(Patch) 단위 처리나 주파수 도메인 접근법을 도입하고 있지만, **"형상(Shape)과 시간(Time)을 분리하여 본다"**는 DILATE의 핵심 철학은 여전히 유효하게 계승되고 있습니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/7bf64439-58aa-482b-866d-8a44feaa55bf/1909.09020v4.pdf)
[2](https://journals.sagepub.com/doi/pdf/10.1177/2041669517737561)
[3](https://arxiv.org/html/2404.15809v1)
[4](https://pmc.ncbi.nlm.nih.gov/articles/PMC10721061/)
[5](https://pmc.ncbi.nlm.nih.gov/articles/PMC3485849/)
[6](http://arxiv.org/pdf/2402.17190.pdf)
[7](https://pmc.ncbi.nlm.nih.gov/articles/PMC5697598/)
[8](http://link.aps.org/pdf/10.1103/PhysRevD.109.124026)
[9](http://arxiv.org/abs/2406.07269)
[10](https://thome.isir.upmc.fr/papers/DILATE_neurips19.pdf)
[11](https://ar5iv.labs.arxiv.org/html/1909.09020)
[12](https://thome.isir.upmc.fr/papers/PAMI-TS23.pdf)
[13](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4960118)
[14](https://scholar.google.fr/citations?user=sFkWZ_EAAAAJ&hl=en)
[15](https://zhouchenlin.github.io/Publications/2025-NeurIPS-Time.pdf)
[16](http://proceedings.mlr.press/v70/cuturi17a/cuturi17a.pdf)
[17](https://arxiv.org/html/2505.06917v1)
[18](https://arxiv.org/abs/1909.09020)
[19](https://arxiv.org/html/2503.00877v2)
[20](https://mblondel.org/publications/mcuturi-mblondel-icml2017.pdf)
[21](https://openreview.net/forum?id=p1KkW2kgDp)
[22](https://scholar.google.com/citations?user=3f3Zq-8AAAAJ&hl=en)
[23](https://www.nature.com/articles/s41599-025-05110-5)
[24](https://arxiv.org/abs/1703.01541)
[25](https://www.ijcai.org/proceedings/2024/0436.pdf)
[26](https://www.semanticscholar.org/paper/1883eac797e5d7bfd1135fa1c4280e560c7fabee)
[27](https://www.sciencedirect.com/science/article/pii/S2095756425000777)
[28](https://arxiv.org/abs/2211.00005)
[29](https://proceedings.neurips.cc/paper_files/paper/2024/file/37c6d0bc4d2917dcbea693b18504bd87-Paper-Conference.pdf)
[30](https://arxiv.org/pdf/1909.09020.pdf)
[31](https://arxiv.org/pdf/2503.00877.pdf)
[32](https://arxiv.org/pdf/2210.15050.pdf)
[33](https://arxiv.org/pdf/2510.23672.pdf)
[34](https://arxiv.org/html/2510.24574v1)
[35](https://arxiv.org/pdf/2306.00620.pdf)
[36](https://arxiv.org/html/2511.08229v1)
[37](https://www.biorxiv.org/content/10.1101/2025.04.21.649913v3.full-text)
[38](http://www.arxiv.org/pdf/2002.03848v1.pdf)
[39](https://arxiv.org/html/2410.04442v4)
[40](https://www.biorxiv.org/content/10.1101/2025.04.21.649913v1.full.pdf)
[41](https://ar5iv.labs.arxiv.org/html/2104.04610)
[42](https://arxiv.org/html/2505.11567v2)
[43](https://arxiv.org/html/2507.23253v1)
[44](https://www.semanticscholar.org/paper/AdaRNN:-Adaptive-Learning-and-Forecasting-of-Time-Du-Wang/6ef770d11e3e5918646b2f5a97c0bcc8bc9b9256)
[45](https://arxiv.org/html/2403.15091v1)
[46](https://arxiv.org/html/2503.02609v1)
[47](https://dl.acm.org/doi/fullHtml/10.1145/3469877.3495644)
[48](https://openreview.net/pdf?id=SJxcS6qG6B)
[49](https://openreview.net/forum?id=4A9IdSa1ul)
