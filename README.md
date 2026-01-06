
# Awesome Series
- Awesome Time Series Forecasting/Prediction Papers : https://github.com/ddz16/TSFpaper

- awesome-TS-anomaly-detection : https://github.com/rob-med/awesome-TS-anomaly-detection

- AI for Time Series (AI4TS) Papers, Tutorials, and Surveys : https://github.com/qingsongedu/awesome-AI-for-time-series-papers

- Awesome Time Series Papers : https://github.com/TSCenter/awesome-time-series-papers

- Awesome Time Series : https://github.com/lmmentel/awesome-time-series

- Awesome Time Series : https://github.com/cure-lab/Awesome-time-series


# Models
- Awesome Large Foundation Models/Task-Specific Models for Weather and Climate : https://github.com/shengchaochen82/Awesome-Foundation-Models-for-Weather-and-Climate

- Diffusion Model for Time Series and Spatio-Temporal Data : https://github.com/yyysjz1997/Awesome-TimeSeries-SpatioTemporal-Diffusion-Model


# Optimization

# Application
## Anomaly Detection
- awesome anomaly detection : https://github.com/hoya012/awesome-anomaly-detection

## Frameworks, Packages, Libraries
### Versatile(Classification, Prediction, Regression...)
- sktime: scikit-learn과 유사한 API를 제공하여 다양한 시계열 작업(분류, 회귀, 예측 등)을 통합적으로 처리할 수 있습니다.
- AIStream의 flow-forecast : 시계열 예측, 분류 및 이상 탐지를 위한 딥러닝 PyTorch 라이브러리입니다(원래는 홍수 예측용으로 개발됨).
- ETNA : 시계열 데이터 구조를 항상 염두에 두고 시계열 예측 및 분석을 수행하는 Python 라이브러리입니다. 통합된 인터페이스를 갖춘 다양한 예측 모델과 함께 탐색적 데이터 분석(EDA) 및 검증 메서드를 제공합니다.
- Tsfresh: 시계열 데이터에서 800개 이상의 유용한 특징(feature)을 자동으로 추출하여 머신러닝 모델의 성능을 향상시키는 데 도움을 줍니다.
- Kats : 시계열 분석을 위한 원스톱 솔루션을 제공하는 것을 목표로 하며, 여기에는 탐지, 예측, 특징 추출/임베딩, 다변량 분석 등이 포함됩니다.
- MatrixProfile : 행렬 프로파일 알고리즘을 활용하여 시계열 데이터 마이닝 작업을 누구나 쉽게 수행할 수 있도록 지원하는 파이썬 라이브러리입니다.
- sktime-dl : sktime용 TensorFlow/Keras 딥러닝 확장 패키지
- tsflex : 유연하고 효율적인 시계열 특징 추출 및 처리 툴킷입니다. 이 패키지는 다변량, 불규칙 샘플링된 시퀀스 데이터에 대한 스트라이드 윈도우 특징 추출을 지원합니다.
- tslearn : 시계열 분석을 위한 머신러닝 도구를 제공하는 파이썬 패키지입니다. 이 패키지는 scikit-learn, numpy, scipy 라이브러리를 기반으로 합니다.

### Causal inferenceg
- Tigramite : 인과 시계열 분석을 위한 파이썬 패키지입니다. 이 패키지를 사용하면 고차원 시계열 데이터 세트에서 인과 그래프를 효율적으로 재구성하고, 얻어진 인과 관계를 모델링하여 인과 매개 및 예측 분석을 수행할 수 있습니다.
- Darts: 고전적인 통계 모델부터 딥러닝 모델까지 다양한 예측 모델을 지원하는 유연한 라이브러리입니다.
- Prophet: Meta(구 Facebook)에서 개발한 것으로, 계절성 및 공휴일 패턴이 있는 비즈니스 데이터 예측에 강점을 보입니다.
- Statsmodels: ARIMA, SARIMA 등 고전적인 통계 모델링 및 분석에 적합합니다.
- GluonTS : MXNet을 기반으로 구축된 확률적 시계열 모델링을 위한 Python 툴킷입니다. GluonTS는 시계열 데이터셋을 불러오고 반복 학습할 수 있는 유틸리티, 바로 학습 가능한 최첨단 모델, 그리고 사용자 정의 모델을 만들 수 있는 구성 요소를 제공합니다.
- Pmdarima: R의 auto.arima와 유사한 기능을 Python에서 제공하여 최적의 ARIMA 모델을 자동으로 찾아줍니다.
- GluonTS: Amazon에서 개발한 확률적 시계열 예측 라이브러리로, 딥러닝 기반 모델에 중점을 둡니다.
- Merlion - 시계열 분석을 위한 머신러닝 프레임워크, ARIMA를 포함한 다양한 모델 지원

### Anomaly Detection
- Chaos Genius(https://github.com/chaos-genius/chaos_genius) : 이상치/이상 징후 탐지 및 근본 원인 분석을 위한 머신러닝 기반 분석 엔진
- Cuebook의 CueObserve : SQL 데이터 웨어하우스 및 데이터베이스의 데이터에 대한 시계열 이상 탐지 및 근본 원인 분석을 제공합니다.
- Hastic : Grafana 기반 UI를 사용하는 시계열 데이터 이상 탐지 도구
- Zillow의 Luminaire : 시계열 데이터에 대한 머신러닝 기반 이상 탐지 및 예측 솔루션을 제공하는 Python 패키지입니다.
- Orion : 비지도 시계열 이상 탐지를 위해 개발된 머신러닝 라이브러리로, 드문 패턴을 식별하고 전문가 검토를 위해 표시하는 여러 "검증된" 머신러닝 파이프라인(Orion 파이프라인)을 제공합니다.
- PyOD : 다변량 데이터에서 이상치를 탐지하기 위한 포괄적이고 확장 가능한 Python 툴킷입니다.
- OutlierDetection.jl : Julia를 이용한 빠르고 확장 가능하며 유연한 이상치 탐지 도구
- ruptures : 오프라인 변화점 감지를 위한 파이썬 라이브러리입니다. 이 패키지는 비정상 신호의 분석 및 분할을 위한 메서드를 제공합니다.
- Skyline : 수십만 개의 지표를 수동적으로 모니터링할 수 있도록 설계된 실시간 이상 탐지 시스템입니다.
- SaxPy 는 SAX의 일반적인 구현체이며, 이상 탐지를 위한 HOTSAX도 제공합니다.

### R
- forecast: ARIMA 및 지수 평활법 등 포괄적인 예측 도구를 제공합니다.
- tseries 및 zoo: 불규칙하거나 규칙적인 시계열 데이터를 처리하고 분석하는 데 유용합니다.
- fable: tidyverse 스타일의 현대적인 시계열 모델링 및 예측 프레임워크를 제공합니다. 
