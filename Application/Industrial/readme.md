# APC (Advanced Process Control)

고급 공정 제어(Advanced Process Control, APC)는 산업 현장에서 기본적인 제어 장치(PID 제어 등)만으로 해결하기 어려운 복잡한 공정을 최적화하기 위해 사용하는 고도화된 제어 기술입니다.

- 다변수 모델 예측 제어 (MPC): APC의 핵심 기술로, 여러 개의 입력 변수와 출력 변수 간의 관계를 수학적 모델로 구축합니다. 이를 통해 향후 공정의 움직임을 예측하고 실시간으로 최적의 운전 조건을 결정합니다.
- 동적 최적화: 공정의 제약 조건(온도, 압력 제한 등) 내에서 에너지를 최소화하거나 생산량을 최대화하도록 실시간으로 제어 목표를 조정합니다.

다변수 모델 예측 제어(MPC, Model Predictive Control)는 시스템의 수학적 모델을 사용하여 미래의 거동을 예측하고, 주어진 제약 조건 내에서 최적의 제어 입력을 실시간으로 계산하는 고도화된 제어 기법입니다. 

MPC의 작동 원리는 자동차의 경로를 계산하는 내비게이션과 유사합니다. 
- 모델 (지도): 현재 위치에서 목적지까지 도로 상황과 차량 성능을 아는 상태입니다.
- 예측 (경로 시뮬레이션): "지금 가속하면 10초 뒤에 어디에 있을까?"를 미리 시뮬레이션합니다.
- 최적화 (최적 경로 선택): 연료를 아끼면서도 가장 빨리 가는 최적의 핸들 조작과 가속 정도를 계산합니다. 

기술적 특징 (전문적 관점)
- 다변수 제어 (MIMO): 여러 개의 입력(Input)과 출력(Output)이 서로 복잡하게 얽혀 있는 시스템(Multi-Input Multi-Output)을 동시에 제어할 수 있습니다.
- 제약 조건의 처리: 기계의 물리적 한계(예: 밸브 개폐 범위, 최대 압력)를 수학적 수식 내에 포함하여 사고를 방지하고 안전 범위 내에서 최대 효율을 냅니다.
- 이동 구간 제어 (Receding Horizon): 일정 시간(Horizon) 동안의 미래 시나리오를 모두 계산한 뒤, 당장 필요한 첫 번째 조치만 실행하고 다음 단계에서 다시 계산을 반복하여 변화하는 환경에 실시간으로 대응합니다. 

MPC는 연산 능력이 비약적으로 발전함에 따라 다음과 같은 첨단 분야에서 필수 기술로 자리 잡았습니다.  
자율주행 및 로보틱스: 장애물을 피하며 최적의 주행 경로를 실시간으로 수정할 때 사용됩니다.  
스마트 팩토리: 에너지 소비를 최소화하면서 생산량을 최대화하는 지능형 공정 최적화에 쓰입니다.  
제약 및 바이오: 실시간으로 품질을 예측하고 유지해야 하는 연속 제조 공정(Continuous Manufacturing)에서 활발히 도입되고 있습니다. 

# Evaporation Process Control, OLED Deposition Process Control

# VM (Virtual Metrology)

# Fault Diagnostics & Analysis

# Process Monitoring

# FDC, Defect Detection & Classification

## 가설 1 : Full Wave inverse Transform 

<details>

### 가설 : 비지도 학습(Unsupervised Learning) 기반의 반도체 FDC(설비 고장 탐지 및 분류) 작업
"Full-waveform inversion, Part 1"에서 다루는 Forward Modeling(순방향 모델링) 기술은 물리적 법칙에 기반하여 데이터를 생성하는 시뮬레이션 과정입니다.  
이 기술은 비지도 학습 FDC가 겪는 핵심적인 난제인 '불량 데이터 부족'과 '물리적 해석 불가능성' 문제를 해결하는 데 핵심적인 역할을 할 수 있습니다.

#### 적용 가능성 : 가상 불량 데이터 생성 (Data Augmentation)
비지도 학습 모델(예: Autoencoder)은 주로 '정상' 데이터만 학습하여 이상치를 탐지하지만, 모델이 실제로 불량을 잘 잡아내는지 검증하기 위해서는 불량 데이터가 필요합니다. 그러나 실제 반도체 공정에서 불량 데이터는 극히 희귀합니다.

- 적용: Forward Modeling을 사용하여 반도체 구조 모델에 가상의 결함(Void, Crack, 박리 등)을 의도적으로 삽입하고 시뮬레이션을 돌리면, 물리적으로 타당한 '합성 불량 신호(Synthetic Anomaly Data)'를 무한대로 생성할 수 있습니다. 이를 통해 비지도 학습 모델의 성능을 테스트하거나, 준지도 학습(Semi-supervised Learning)의 학습 데이터로 활용할 수 있습니다. 

#### 예시 모델 : 물리학 기반 비지도 학습 (Physics-Informed Unsupervised Learning)
최근 연구 동향은 데이터만 사용하는 딥러닝의 한계를 극복하기 위해 물리 방정식(PDE)을 학습 과정에 통합하는 것입니다. 
- 적용: 센서 데이터(진동, 초음파 등)를 입력받아 설비 상태를 추론하는 오토인코더(Autoencoder) 구조에서, 디코더(Decoder) 역할을 Forward Modeling(파동 방정식 시뮬레이터)으로 대체하거나 규제항(Regularization)으로 추가할 수 있습니다.
- 효과: 이렇게 하면 AI가 생성한 복원 신호가 반드시 물리 법칙을 따르도록 강제되므로, 노이즈에 의한 오경보(False Alarm)를 획기적으로 줄이고 탐지된 이상의 물리적 원인을 역추적(Inversion)하는 데 유리해집니다. 

### 가설 1-1. Full-waveform Inversion(FWI)의 원리를 반도체 공정의 FDC(설비 고장 탐지 및 분류)에 접목한 고정밀 시계열 데이터 분석 파이프라인 설계 방안

<details>

### 1. [데이터 기반 가상 물리 학습 모델 파이프라인 설계]

#### 1단계: 물리적 디지털 트윈 및 수치 모델 정의 (Foundation)
- 물리 방정식 정의: 분석하고자 하는 공정 센서(압력, OES, RF Power 등)의 거동을 지배하는 방정식을 설정합니다. (예: 가스 흐름은 Navier-Stokes, 열 변화는 Heat Equation)

##### 사용 가능한 방정식
특정 공정(식각, 증착 등)의 내부 물리적 변화를 시뮬레이션할 때 고려됩니다. 
- 이동-확산 방정식 (Drift-Diffusion Equation): 반도체 내 전하 운반자의 거동을 모델링할 때 기본이 됩니다.
- 포아송 방정식 (Poisson's Equation): 정전기 전위와 전하 밀도 간의 관계를 계산하여 공정 중 전기적 특성 변화를 예측합니다.
- 나비에-스토크스 방정식 (Navier-Stokes Equation): 증착이나 식각 장비 내부의 가스 흐름과 열전달을 모델링하는 CFD(전산유체역학)에 사용됩니다. 

반도체 웨이퍼의 결함을 초음파나 광학적 방법으로 감지할 때 활용됩니다. 
- 음향/탄성 파동 방정식 (Acoustic/Elastic Wave Equations): 초음파를 이용한 웨이퍼 내부의 크랙(Crack)이나 보이드(Void) 검사 시 파동의 전파와 반사를 모델링하는 데 특화되어 있습니다.
- 전자기 파동 방정식 (Maxwell's Equations): 리소그래피 공정에서의 빛의 회절이나 박막 두께 측정을 위한 타원계측법(Ellipsometry) 시뮬레이션에 적용할 수 있습니다. 

#### 2단계: 가상 시계열 데이터 생성 (Forward Modeling)
- 공정 레시피 입력: 현재 공정의 타겟 설정값(Source)을 입력합니다.
- 합성 시계열 생성: 정의된 물리 모델을 통해 정상 상태에서 센서들이 기록했어야 할 '이상적인 시계열 데이터(Synthetic Data)'를 생성합니다.
- 가상 결함 데이터(Data Augmentation): 특정 부품 마모나 리크 시나리오를 물리 모델에 주입하여, 실제로는 얻기 힘든 '물리 기반 불량 시계열 패턴'을 확보합니다.

#### 3단계: 실시간 비교 및 잔차 분석 (Data Alignment)
- 실제 데이터 수집: 장비 센서로부터 실시간 시계열 데이터를 수집합니다.
시계열 정렬: DTW(Dynamic Time Warping) 기술을 사용하여 공정 스텝 간의 시간적 미세 편차를 보정하고 시뮬레이션 데이터와 실제 데이터를 1:1로 매칭합니다.
잔차 계산(Residual): 실제 데이터와 시뮬레이션 데이터의 차이를 계산합니다. 이 차이가 클수록 물리 모델(정상)에서 벗어난 '이상 상태'임을 의미합니다.

#### 4단계: 원인 역추적 및 상태 업데이트 (Adjoint Modeling & Inversion)
- 기울기 계산(Adjoint Modeling): 데이터의 차이(잔차)를 수반 방정식(Adjoint Equation)을 통해 역전파시켜, 어떤 물리적 파라미터가 이 오차를 유발했는지의 기울기(Gradient)를 구합니다.
- 파라미터 업데이트: 최적화 알고리즘(L-BFGS 등)을 통해 챔버 내부의 상태값(예: 밸브의 실제 개폐율, 돔 내부 벽면의 증착 두께 등)을 추정합니다.
- 진단(Classification): 추정된 물리적 상태 변화를 바탕으로 "단순 노이즈"인지, "특정 부품의 마모"인지 물리적 근거에 기반하여 분류합니다.

#### 5단계: 비지도 학습 기반 이상 탐지 (Unsupervised Detection)
오토인코더 결합: 위에서 계산된 물리적 잔차와 주요 특성값들을 비지도 학습 모델(예: LSTM-Autoencoder)의 입력으로 사용합니다.
고장 탐지: 모델이 학습한 정상 범위를 벗어나는 시계열 패턴 발생 시 알람을 발생시킵니다. 물리 모델이 1차 필터 역할을 하므로 오경보율이 획기적으로 낮아집니다.

</details>

-> 제약 조건 : 
- 현재 FDC 작업을 하기 위한 시계열 데이터만 존재.
- 이 데이터가 어느 현장에서 온 데이터인지, 정상 데이터인지, 비정상 데이터인지 알 수 없음.
- 물리 방정식을 만들기 위한 공정 도메인 지식 부족

### 가설 1-2. [수정된 FDC 파이프라인: 데이터 기반 가상 물리 학습 모델]

<details>

#### 1단계: 데이터 잠재 공간 임베딩 (Unsupervised Feature Discovery) 
- 문제 해결: 데이터의 정체(정상/비정상)를 모르므로, 우선 데이터의 '본질적 패턴'을 추출합니다.
- 기술: Contrastive Learning (대조 학습) 또는 Masked Time-series Modeling.
- 방법: 시계열의 일부를 가리고 복원하는 학습을 통해 데이터의 규칙성을 학습합니다. 이 과정에서 출처가 다르거나 이상이 있는 데이터는 잠재 공간(Latent Space)에서 서로 다른 군집(Cluster)을 형성하게 됩니다.

#### 2단계: 가상 물리 엔진 구축 (Neural ODE/SDE) 
- 문제 해결: 물리 방정식을 모르므로, 신경망이 방정식 자체를 학습하게 합니다.
- 기술: Neural Ordinary Differential Equations (Neural ODEs).
- 방법: 시계열 데이터의 변화율 $(\(dy/dt\))$ 을 신경망이 근사하도록 학습시킵니다. 이는 FWI Part 1의 Forward Modeling을 '딥러닝 기반 상태 천이 모델'로 대체하는 작업입니다. 물리 지식 없이도 "이 시스템은 이런 흐름으로 변한다"는 동역학을 스스로 학습합니다.

#### 3단계: 가상 Adjoint를 이용한 이상 점수 산출 (Self-Supervised Reconstruction) 
- 문제 해결: 무엇이 정상인지 모르므로, '대부분의 데이터가 따르는 보편적 규칙'을 정상으로 정의합니다.
- 기술: Adjoint State Method 기반의 오차 역전파.
- 방법:2단계에서 만든 모델에 현재 데이터를 넣고 다음 시점을 예측(Forward)합니다.실제 데이터와 예측값의 차이(잔차)를 구합니다. 이 잔차를 역전파(Adjoint)하여 현재 데이터가 시스템의 '보편적 규칙'에서 얼마나 벗어나 있는지 이상 점수(Anomaly Score)를 매깁니다.

#### 4단계: 클러스터링 기반 데이터 라벨링 (Pseudo-Labeling) 
- 문제 해결: 현장 및 정상 여부를 구분합니다.
- 기술: DBSCAN 또는 HDBSCAN (밀도 기반 군집화).
- 방법: 이상 점수와 잠재 변수를 바탕으로 데이터를 그룹화합니다.
- 대규모 군집: 특정 현장의 정상 데이터일 확률 높음.소규모/외곽 군집: 비정상 데이터 또는 특이 공정 데이터로 분류.

#### 5단계: 반복적 모델 정교화 (Iterative Refinement) 
- 방법: 분류된 '가상 정상 데이터'만을 사용하여 2단계의 가상 물리 엔진을 다시 학습시킵니다. 이를 통해 모델은 정상적인 공정 흐름에 극도로 민감해지며, 미세한 장비 이상도 '물리적 예측 실패'로 간주하여 탐지해낼 수 있습니다.

</details>

장점	
- 물리 지식 없이도 시스템의 연속적인 흐름 학습
- 데이터 간의 복잡한 비선형 인과관계 포착 가능

단점	
- 높은 연산량 및 학습 시간 소요
- 'Black-box' 특성으로 인해 초기 수렴이 어려움

현재 프로젝트는 데이터의 출처나 정상 여부를 모르는 상태입니다.  
1-2 는 Neural ODE를 통해 데이터에 숨겨진 '잠재적 물리 방정식'을 스스로 찾아냅니다.  
단순히 현재 값의 통계적 이탈을 보는 1-3 보다, 시간의 흐름에 따른 데이터의 전파 경로(Trajectory)를 예측하므로 미세한 공정 표류(Drift)를 잡아내는 데 훨씬 정밀합니다.

### 가설 1-3. [수정된 FDC 파이프라인: 다변량 물리 이탈 기반 FDC]

<details>

#### [1단계] 데이터 전처리 및 시간적 정렬 (Data Alignment) 
데이터의 출처와 정상을 알 수 없으므로, 모든 시계열 데이터를 동일한 비교 선상에 놓는 과정입니다. 
- 다변량 정규화 (Standardization): 센서별로 단위가 다르므로(압력, 온도, 유량 등), 모든 파라미터를 평균 0, 표준편차 1로 스케일링합니다.
- 시간 축 동기화: DTW(Dynamic Time Warping)를 사용하여 공정의 시작과 끝이 불분명한 시계열들을 패턴 중심으로 정렬합니다.
- FWI 연관성: 지진파 수신 시간(Travel-time)을 맞추는 데이터 처리 과정과 동일합니다.

#### [2단계] 가상 물리 엔진 학습 (Forward Modeling 대체) 
물리 방정식을 대신할 Neural ODE(신경 상미분 방정식)를 통해 시스템의 동역학을 학습합니다. 
- 동역학 학습: 신경망이 시계열 데이터의 변화율 $(\(dy/dt\))$ 을 학습하게 합니다.
- 입력: 현재 시점의 센서 값들 $(\(t_{0}\))$
- 출력: 물리 법칙에 따라 예측된 다음 시점의 센서 값들 $(\(t_{1},t_{2},\dots ,t_{n}\))$
- FWI 연관성: 매질의 속도 모델을 통해 파동 전파를 시뮬레이션하는 Forward Modeling 단계를 데이터 기반 가상 모델로 대체한 것입니다.

#### [3단계] 다변량 상관관계 분석 (Multivariate Interaction) 
단일 파라미터의 튀는 값(Spike)이 아닌, 여러 센서의 복합적 이상을 감지하기 위한 단계입니다. 
- 마할라노비스 거리(Mahalanobis Distance) 계산: 각 파라미터 간의 공분산(Covariance)을 고려하여 거리를 측정합니다.
- 효과: A 센서가 급격히 변할 때 B 센서도 같이 변하는 것이 '정상 물리 규칙'이라면, A만 변하고 B는 가만히 있는 경우를 강력한 이상치로 잡아냅니다.
- FWI 연관성: 관측 데이터와 합성 데이터의 오차(Residual)를 단순 차이값이 아닌, 다변량 벡터 공간에서의 거리로 측정하는 고도화된 목적 함수(Objective Function) 설정 단계입니다.

#### [4단계] 비지도 이상치 기준 설정 (Dynamic Thresholding) 
라벨이 없으므로 통계적 분포를 통해 '이상(Anomaly)'의 경계선을 스스로 정합니다. 
- 동적 임계값 (Adaptive Threshold): 3단계에서 나온 거리 점수들에 대해 KDE(Kernel Density Estimation)를 적용하여 데이터의 밀도 분포를 구합니다.
- 기준: 하위 95~99%의 밀도 영역을 '정상 물리 범위'로 간주하고, 나머지 극단값(Outlier)을 이상치로 자동 분류합니다.
- FWI 연관성: 반전(Inversion) 과정에서 노이즈와 유효 신호를 구분하기 위한 통계적 필터링 과정과 맥을 같이 합니다.

#### [5단계] Adjoint 기반 원인 역추적 (Adjoint Modeling & Sensitivity) 
이상치가 발견되었을 때, 어떤 파라미터가 주범인지 찾아내는 핵심 단계입니다. 
- 기울기 역전파 (Back-propagation of Error): 이상치로 판정된 오차(Residual)를 Adjoint State Method 원리를 이용해 입력 파라미터 쪽으로 역전파합니다.
- 파라미터 기여도 산출: 오차 발생에 가장 큰 영향을 준 센서(기울기 값이 큰 센서)를 식별합니다.
- 결과: "여러 파라미터 중 압력(Pressure)과 RF Power의 상관관계 붕괴가 이번 이상의 주요 원인임"이라는 리포트를 생성합니다.
- FWI 연관성: 데이터 오차로부터 모델 파라미터(지하 속도)를 수정하기 위해 기울기를 계산하는 Adjoint Modeling의 철학을 그대로 구현한 것입니다.

#### [6단계] 모델 자기 정교화 (Iterative Refinement) 
4단계에서 '정상'으로 분류된 깨끗한 데이터만을 사용하여 2단계의 가상 물리 엔진을 재학습(Fine-tuning)합니다.
이 루프를 반복할수록 모델은 정상 상태의 물리적 거동을 더 정밀하게 예측하게 되며, 아주 미세한 이상 징후도 잡아낼 수 있게 됩니다. 

##### [개발 시 참고할 핵심 라이브러리 및 도구] 
- 가상 물리 엔진 구현: Torchdyn (Neural ODE 라이브러리)
- 다변량 거리 및 통계: PyOD (Python Outlier Detection 특화 라이브러리)
- 고성능 연산: Devito (부분적으로 물리 방정식 결합 시 활용) 
  
</details>

장점
- 계산 복잡도가 낮고 결과 해석이 매우 직관적
- 다수 파라미터 간의 상관관계 붕괴 포착에 최적화

단점	
- 선형적 상관관계에 의존적임
- 시계열의 역동적인 변화 자체를 모델링하긴 한계

행렬 연산과 통계적 분포 계산에 기반하므로, GPU 없이 CPU 환경에서도 대규모 시계열 데이터를 빠르게 처리할 수 있습니다.  
반면 1-2 는 ODE Solver를 반복적으로 호출해야 하므로 메모리 점유율이 높고 연산 시간이 길어 실시간 FDC 적용 시 장비 사양을 많이 탑니다.

##### 예시 : 검증 지표
장비 상태의 변동성을 정량화하고 관리 한계를 설정할 때 주로 사용됩니다. 
- 1. 평균 및 표준편차 식: 데이터의 중심 경향성과 산포를 확인합니다.

$\(UCL/LCL=\text{Mean}\pm 3\sigma \)$ (관리 상/하한선)

- 2. 공정 능력 지수 (Cpk): 공정이 규격 한계 내에서 제품을 생산할 수 있는 능력을 평가합니다.

$$\(Cpk=\min (\frac{USL-\text{mean}}{3\sigma },\frac{\text{mean}-LSL}{3\sigma })\)$$

- 격자 및 매질 설정: 반도체 챔버의 기하학적 구조를 격자(Grid)로 모델링하고, 각 부품의 초기 물성치(노후도, 저항, 흡착률 등)를 설정합니다.
- 기술 요소: Devito를 이용한 편미분 방정식(PDE) 코드 생성.

- 결정계수 $(\(R^{2}\))$ : 센서 간의 상관관계를 분석하여 장비 신호의 연관성을 파악합니다.

  
</details>


# Modeling and prediction of OLED performance/characteristics, OLED performance prediction, Prediction of OLED characteristics, OLED property prediction, OLED design prediction

# Accelerating Material Discovery, Improving Experimental Efficiency, Automation in Materials Development : Material Property Prediction, Structure Search / Prediction, Patent Analysis

# Design Optimization, Process Optimization, Engineering Design Automation (EDA)

# Automated Scheduling, Job Shop Scheduling, Production Scheduling
- AI-driven industrial process automation and scheduling

# Industrial Fire Detection, Fingerprint & Biometrics


# Reference
- Awesome Industrial Anomaly Detection : https://github.com/M-3LAB/awesome-industrial-anomaly-detection
