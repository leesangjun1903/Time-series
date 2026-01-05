
# GAF-CNN-LSTM for Multivariate Time-Series Images Forecasting

## 1. 핵심 주장 및 주요 기여 요약

이 연구는 다변량 시계열(Multivariate Time-Series) 예측 문제를 **"이미지 처리 문제"**로 재정의하여 해결하고자 합니다.
*   **핵심 주장:** 시계열 데이터를 1차원 신호로 처리하는 기존 방식 대신, **Gramian Angular Field (GAF)**를 통해 2차원 이미지로 변환하면 시계열의 시간적 상관관계를 시각적 패턴(텍스처)으로 보존할 수 있습니다. 이를 **CNN(특징 추출)**과 **LSTM(시계열 예측)**이 결합된 하이브리드 모델에 입력하면, 복잡한 전처리 없이도 높은 예측 성능을 달성할 수 있습니다.
*   **주요 기여:**
    1.  다변량 시계열을 다채널(Multi-channel) 이미지로 인코딩하는 프레임워크 제안.
    2.  수작업 특징 추출(Hand-crafted features) 없이 데이터의 잠재적 특징을 자동으로 학습하는 End-to-End 딥러닝 구조 제시.
    3.  UEA 다변량 시계열 벤치마크에서 기존 모델(LSTM, CRNN 등) 대비 우수한 성능(RMSE, MAPE 감소) 입증.

***

## 2. 상세 분석: 문제, 방법론, 모델, 성능 및 한계

### 2.1 해결하고자 하는 문제
다변량 시계열 예측은 변수 간의 복잡한 상관관계와 시간적 의존성을 동시에 고려해야 하므로 어렵습니다. 기존 연구들은 각 변수에 대해 별도의 특징을 수작업으로 추출하거나, 1차원 신호 처리에 국한되어 있어 변수 간의 비선형적 상호작용을 충분히 포착하지 못하는 한계가 있었습니다.

### 2.2 제안하는 방법 (수식 포함)
이 논문은 시계열 데이터를 **이미지화(Imaging)**하는 과정을 핵심으로 합니다.

**1단계: 데이터 재스케일링 및 극좌표 변환**
주어진 시계열 $X = \{x_1, x_2, ..., x_n\}$을 $[-1, 1]$ 로 정규화한 뒤, 극좌표계로 변환합니다.[1]

$$
\tilde{x}_i \in [-1, 1] \quad \text{(정규화된 값)}
$$

$$
\phi_i = \arccos(\tilde{x}_i), \quad -1 \le \tilde{x}_i \le 1
$$

여기서 $\phi$는 각도, 시간 $t$는 반지름 $r$에 대응됩니다. 이 변환은 시계열의 시간적 순서와 값을 각도와 깊이 정보로 보존합니다.

**2단계: Gramian Angular Field (GAF) 생성**
각 시점 간의 시간적 상관관계를 나타내는 $n \times n$ 매트릭스를 생성합니다. 논문에서는 GAF를 언급하며, 일반적으로 사용되는 **GASF (Summation)** 또는 **GADF (Difference)** 수식은 다음과 같습니다.

*   **GASF (Gramian Angular Summation Field):**

$$
    GASF_{i,j} = \cos(\phi_i + \phi_j) = \tilde{x}_i \tilde{x}_j - \sqrt{1-\tilde{x}_i^2}\sqrt{1-\tilde{x}_j^2}
    $$

*   **GADF (Gramian Angular Difference Field):**

$$
    GADF_{i,j} = \sin(\phi_i - \phi_j) = \sqrt{1-\tilde{x}_i^2}\tilde{x}_j - \tilde{x}_i\sqrt{1-\tilde{x}_j^2}
    $$

이 행렬은 시계열 값을 이미지의 픽셀 값으로 변환하며, 대각 성분은 원래 시계열의 정보를 담고 있습니다. 다변량 시계열의 경우, 각 변수(Variable)마다 하나의 GAF 이미지를 생성하여 **다채널 이미지(예: RGB 채널처럼 변수 1, 2, 3...)**로 구성합니다.

### 2.3 모델 구조 (GAF-CNN-LSTM)
모델은 크게 두 부분으로 구성된 직렬 구조입니다.
1.  **Feature Learning (CNN):**
    *   입력: GAF로 변환된 다변량 시계열 이미지.
    *   구조: `Conv2D` → `ReLU` → `Pooling` → `Flatten`.
    *   역할: 이미지화된 시계열에서 국소적인 패턴(예: 급격한 상승/하강, 주기성 등)을 시각적 특징(Feature Map)으로 추출합니다.
2.  **Sequence Prediction (LSTM):**
    *   구조: `TimeDistributed` 레이어를 통해 연속된 이미지 시퀀스에서 특징을 추출한 뒤, 이를 `LSTM` 셀에 입력합니다.
    *   역할: CNN이 추출한 특징들의 시간적 흐름(Sequence)을 학습하여 미래 시점의 값을 예측합니다.

### 2.4 성능 향상 및 한계
*   **성능:** MotorImagery, EigenWorms 등 UEA 벤치마크 데이터셋에서 순수 LSTM이나 1D-CNN(CRNN, 1D-MTCNN)보다 낮은 RMSE(평균 제곱근 오차)와 MAPE(평균 절대 비율 오차)를 기록했습니다. 이는 이미지 변환이 노이즈를 줄이고 데이터의 구조적 특징을 더 잘 드러냈음을 시사합니다.
*   **한계:**
    *   **데이터 크기 증가:** 시계열 길이 $N$에 대해 $N \times N$ 크기의 이미지를 생성하므로, 시계열이 길어질수록 메모리와 연산량 부담이 제곱( $O(N^2)$ )으로 증가합니다.
    *   **이미지 해상도 의존성:** 긴 시계열을 작은 이미지로 리사이징할 경우 정보 손실이 발생할 수 있습니다.

***

## 3. 모델의 일반화 성능 향상 가능성

이 모델은 다음과 같은 이유로 높은 일반화(Generalization) 가능성을 가집니다.

1.  **도메인 불변성 (Domain Invariance):** 수치 데이터 자체보다는 데이터가 그리는 '형상(Shape)'과 '패턴(Texture)'을 학습합니다. 따라서 주식 차트의 패턴이든, 센서 데이터의 패턴이든 유사한 기하학적 특징이 있다면 도메인 지식 없이도 전이 학습(Transfer Learning)이 용이합니다. 최신 연구(2025년)에서도 GAF 기반 전이 학습이 금융 데이터에서 효과적임이 입증되었습니다.[2]
2.  **강건함 (Robustness):** CNN의 풀링(Pooling) 연산은 이미지의 작은 변동(노이즈)에 둔감하게 만듭니다. 시계열 데이터에 포함된 고주파 노이즈가 이미지 변환 및 축소 과정에서 자연스럽게 필터링되어, 과적합(Overfitting)을 방지하고 일반화 성능을 높입니다.
3.  **자동화된 특징 공학:** 특정 도메인에 특화된 수작업 특징(예: 이동평균, 엔트로피 등)에 의존하지 않으므로, 새로운 유형의 데이터셋에도 구조 변경 없이 즉시 적용 가능합니다.

***

## 4. 향후 연구 영향 및 고려사항

### 4.1 연구에 미친 영향
이 논문은 **"Time-Series as Image"**라는 패러다임을 확산시키는 데 기여했습니다. 단순히 값을 예측하는 수치적 접근에서 벗어나, 컴퓨터 비전(Computer Vision)의 강력한 모델들(ResNet, VGG 등)을 시계열 분석에 도입할 수 있는 교두보를 마련했습니다. 이는 이후 2020년대 초반 비전 트랜스포머(ViT)가 시계열에 적용되는 기초가 되었습니다.

### 4.2 향후 연구 시 고려할 점
*   **희소성(Sparsity) 활용:** GAF 행렬은 크기가 크지만 정보가 중복될 수 있습니다. 전체 이미지를 다 쓰는 대신 중요한 영역만 보거나, 이미지를 압축하여 연산 효율을 높이는 연구가 필요합니다.
*   **설명 가능성 (XAI):** CNN이 이미지의 어떤 부분(즉, 시계열의 어떤 구간)을 보고 예측했는지 `Grad-CAM` 등을 통해 역추적하여, "왜 주가가 오를 것이라 예측했는가?"에 대한 설명력을 확보해야 합니다.

***

## 5. 2020년 이후 관련 최신 연구 비교 분석

2020년 이후, GAF-CNN-LSTM의 아이디어는 더욱 고도화되거나 새로운 아키텍처(Transformer, GNN)와 경쟁하고 있습니다.

| 비교 항목 | GAF-CNN-LSTM (2019) | 최신 연구 트렌드 (2020~2025) | 관련 최신 모델/기술 |
| :--- | :--- | :--- | :--- |
| **데이터 표현** | GAF 이미지 (고정된 2D 변환) | **Graph & Dynamic Graph** | **ForecastGrapher(2024), FourierGNN(2023)**: 변수 간 관계를 고정된 이미지가 아닌, 학습 가능한 그래프 구조로 모델링하여 변수 간 인과관계를 더 명확히 파악함[3][4]. |
| **백본 네트워크** | CNN + LSTM | **Transformer & Attention** | **iTransformer, PatchTST**: 이미지를 거치지 않고 시계열 패치(Patch) 자체에 어텐션을 적용하여 장기 의존성(Long-term dependency) 학습 능력을 획기적으로 개선함. |
| **하이브리드** | CNN(공간) $\to$ LSTM(시간) | **Advanced Hybrids** | **CNN-LSTM-Attention(2024)**: LSTM 출력에 어텐션 메커니즘을 추가하여 중요 시점을 강조하거나, 유전 알고리즘(GA)으로 하이퍼파라미터를 최적화함[5][6]. |
| **이미지 활용** | 단순 패턴 인식 | **Vision Backbone 활용** | **ViT for Time Series(2025)**: CNN 대신 비전 트랜스포머(ViT)를 사용하여 이미지화된 시계열의 전역적(Global) 문맥을 파악하는 연구로 진화함[7]. |
| **성능/효율** | $O(N^2)$ 메모리 소모 | **효율성 중시** | **KANMTS(2025)**: Kolmogorov-Arnold Networks(KAN)를 도입하여 MLP보다 적은 파라미터로 복잡한 함수를 근사, 계산 효율성을 높임[8]. |

**결론적으로:** GAF-CNN-LSTM은 "시각적 변환"의 유효성을 입증한 선구적 연구였으나, 2024-2025년의 연구들은 이미지 변환의 계산 비용을 줄이거나(Graph, KAN), CNN/LSTM의 한계를 넘는(Transformer) 방향으로 발전하고 있습니다. 하지만 노이즈가 심하거나 패턴 인식이 중요한 특정 도메인(금융, 생체 신호)에서는 여전히 GAF 기반 접근이 유효하게 사용되고 있습니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4c6467b3-1917-487e-b00d-da14f1d5e3a2/beamerpostericml_-copia.pdf)
[2](https://arxiv.org/pdf/2504.00378.pdf)
[3](http://arxiv.org/pdf/2405.18036.pdf)
[4](https://arxiv.org/pdf/2311.06190.pdf)
[5](https://www.mdpi.com/1996-1073/17/14/3435)
[6](https://www.ijournalse.org/index.php/ESJ/article/view/679)
[7](https://arxiv.org/html/2502.08869v1)
[8](https://pmc.ncbi.nlm.nih.gov/articles/PMC12222760/)
[9](https://www.mdpi.com/2673-4826/6/2/17)
[10](https://ieeexplore.ieee.org/document/11142246/)
[11](https://ieeexplore.ieee.org/document/11230348/)
[12](https://ecsenet.com/index.php/2576-6821/article/view/716)
[13](https://hbemdata.org/index.php/ojs/article/view/90)
[14](https://www.mdpi.com/2076-3417/16/1/367)
[15](https://zgt.com.ua/en/попередня-оцінка-ефективності-гібри/)
[16](https://inmateh.eu/volumes/volume-61--no-2--2020/article/61-07-zeying-xu-prediction-model-of-ammonia-concentration-in-yellow-feather-broilers-house-durin)
[17](https://www.mdpi.com/2079-9292/13/11/2032)
[18](https://arxiv.org/pdf/2101.06861.pdf)
[19](https://arxiv.org/pdf/2109.06489.pdf)
[20](https://arxiv.org/pdf/2210.06126.pdf)
[21](http://arxiv.org/pdf/2411.17382.pdf)
[22](http://arxiv.org/pdf/2503.04528.pdf)
[23](https://arxiv.org/pdf/2202.08408.pdf)
[24](https://www.emergentmind.com/topics/gramian-angular-difference-field)
[25](https://www.emergentmind.com/topics/cnn-lstm-model-for-time-series-forecasting)
[26](https://www.sciencedirect.com/science/article/pii/S1877050924003648)
[27](https://arxiv.org/html/2504.00378v1)
[28](https://www.sciencedirect.com/science/article/pii/S1877050924003648/pdf)
[29](https://www.semanticscholar.org/paper/LSTM-based-Multivariate-Time-Series-Analysis:-A-of-Saputra-Wibawa/a043e102dba4da827bef4b0e4f414ebd776137df)
[30](https://openaccess.thecvf.com/content/CVPR2022W/PBVS/papers/Paheding_GAF-NAU_Gramian_Angular_Field_Encoded_Neighborhood_Attention_U-Net_for_Pixel-Wise_CVPRW_2022_paper.pdf)
[31](https://research.tue.nl/files/347100807/AlHarazi_A-1.pdf)
[32](https://www.sciencedirect.com/science/article/pii/S1319157824003215)
[33](https://blog.naver.com/gdpresent/223058651021)
[34](https://arxiv.org/html/2405.07117v1)
[35](https://www.pnrjournal.com/index.php/home/article/download/3757/3856/4679)
[36](https://www.sciencedirect.com/science/article/pii/S1059056025008822)
[37](https://www.nature.com/articles/s41598-025-08590-2)
[38](https://www.nature.com/articles/s41598-025-12516-3)
[39](https://min23th.tistory.com/32)
[40](https://www.scribd.com/document/517245953/Gold-Volatility-Prediction-Using-a-CNN-LSTM-Approa)
[41](https://www.studocu.vn/vn/document/truong-dai-hoc-cong-nghe-thong-tin/nhap-mon-thi-giac-may-tinh/time-series-data-processing/101239013)
[42](https://www.semanticscholar.org/paper/Multivariate-CNN-LSTM-Model-for-Multiple-Parallel-Widiputra-Mailangkay/b8d6630ab2e312cbc908e0e4ecd005c8d136ce69)
[43](https://arxiv.org/pdf/1506.00327.pdf)
[44](https://www.semanticscholar.org/paper/Prediction-for-Time-Series-with-CNN-and-LSTM-Jin-Yu/604636c7d3698ea2f24f9cdf2eac8a900e7dc3f8)
[45](https://arxiv.org/pdf/1901.05237.pdf)
[46](https://peerj.com/articles/cs-2719/)
[47](https://arxiv.org/pdf/2502.10721v1.pdf)
[48](https://ar5iv.labs.arxiv.org/html/1506.00327)
[49](https://arxiv.org/abs/2502.16294)
[50](https://arxiv.org/pdf/2112.08060.pdf)
[51](https://peerj.com/articles/cs-1807/)
[52](https://pdfs.semanticscholar.org/b966/ec5528e3ec8a97660841a42cfc561bd93e63.pdf)
[53](https://arxiv.org/html/2502.10721v1)
[54](https://arxiv.org/vc/arxiv/papers/1803/1803.09052v1.pdf)
[55](https://arxiv.org/pdf/2302.02515.pdf)
[56](https://arxiv.org/html/2505.00307v2)
[57](https://arxiv.org/html/2310.07427v3)
[58](https://daeunnniii.tistory.com/210)
