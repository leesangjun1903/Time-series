# EddyNet: A Deep Neural Network for Pixel-Wise Classification of Oceanic Eddies

## 1. 핵심 주장 및 주요 기여  
EddyNet은 심층 학습 기반의 U-Net 형태 인코더-디코더 아키텍처를 활용하여 해양 표층고(SSH) 맵의 각 픽셀을 비회전성(No-eddy), 반시계(anticyclonic), 시계(cyclonic) 에디로 정확히 분류한다. 주요 기여는 다음과 같다:  
- 전통적 물리·지형학 기법의 전문가 조정 의존성과 노이즈 민감도를 극복하는 **완전 자동화된 픽셀 단위 에디 분류** 시스템 제안  
- U-Net 기반 경량화(3-stage, 32-filter) 구조와 SELU/AlphaDropout 적용을 통해 **학습 속도 및 일반화 성능** 향상  
- Dice 계수를 손실함수로 사용해 주요 클래스(anticyclonic, cyclonic)에 대한 분할 정확도 개선  
- ‘ghost eddies’ 검출에서 종래 PET14 방식보다 누락된 에디 중심을 50% 이상 올바르게 식별  

## 2. 문제 정의 및 제안 방법  

### 문제 정의  
- **대상**: 해양 표층고(SSH) 맵으로부터 mesoscale eddy(반시계·시계)에 대응하는 픽셀을 식별  
- **기존 한계**: Okubo-Weiss 등 물리·지형적 기준은 지역·파라미터 특이적, 노이즈 민감성이 높고 추적 오류(ghost eddies) 발생  

### 제안 방법  
1. **데이터 준비**  
   - CMEMS AVISO-SSH 일일 맵(0.25°)과 PET14 알고리즘 결과(1998–2012)  
   - 128×128 패치 추출, 15년 중 14년은 학습, 마지막 해는 시험  
   - PET14 속성(contour) ⇒ 픽셀별 분할 마스크 생성  

2. **모델 구조**  
   - **인코더**: 3단계, 각 단계 두 개의 3×3 컨볼루션 + SELU(또는 ReLU+BN) + 2×2 맥스풀링  
   - **디코더**: 대칭적 3단계 전치 컨볼루션(deconvolution) + 스킵 커넥션  
   - **정규화**: SELU 모델(EddyNet S)에는 AlphaDropout, BatchNorm 결합; ReLU 모델(EddyNet)에는 Dropout  

3. **손실함수 (Dice Loss)**  
   - 소프트 다이스 계수:  

$$
       \text{softDiceCoef}(P,G) = \frac{2\sum_i p_i g_i}{\sum_i p_i + \sum_i g_i}
     $$  
   
   - 다중 클래스용 평균 one-vs-all softDice 계수 기반  

$$
       \mathcal{L}\_{\text{Dice}} = 1 - \frac{1}{C} \sum_{c=1}^C \text{softDiceCoef}_c
     $$

4. **훈련 및 평가**  
   - Adam 옵티마이저, 배치 크기 16, 조기 중단(Early Stopping)  
   - ReLU+BN vs. SELU 비교, Dice Loss vs. Categorical Cross-Entropy  
   - **결과**:  
     - Dice Loss 사용 시 anticyclonic/cyclonic 클래스 Dice 계수 증가  
     - EddyNet S(SELU) 학습 시간 절반, 전반적 정확도·일관성 우수  

### 성능 향상 및 한계  
- **향상**:  
  - 주요 클래스 Dice 계수 0.68–0.71로 전통 기법 대비 유의미한 개선  
  - Ghost eddy 중심 식별 정확도 45–55%  
- **한계**:  
  - 공간적 단일 프레임 처리, 시간적 연속성·추적 미포함  
  - 데이터 양(5,100 샘플) 대비 네트워크 용량 과한 경우 과적합 위험  
  - 단일 SSH 정보만 활용, 온도·염분 등 추가 변수 미반영  

## 3. 일반화 성능 향상 가능성  
- **경량화 아키텍처**: 32-필터, 3단계 구조와 스킵 커넥션으로 지역별·전역 적용 시 메모리 부담 감소  
- **Self-normalizing 활성화(SELU) + AlphaDropout**: 내부 표현의 분산·평균 제어로 다른 해역·센서 잡음 저항성 강화  
- **Dice 기반 손실**: 클래스 불균형 상황에서도 소수 클래스(시계/반시계)에 집중 학습 유도  
- **Fine-tuning 전략**: 공개된 사전학습 가중치 이용 후, 타 해역 데이터로 미세 조정하면 지역 특이성 적응 가능  

## 4. 향후 연구 영향 및 고려사항  
- **시계열 처리**: 3D U-Net 또는 ConvLSTM 도입해 에디 동역학 모델링  
- **다중 모달 융합**: SST, 염분, 해양 풍·조류 정보 추가로 검출·클래스 분리 정밀도 제고  
- **전지구 적용**: 다양한 해역(북대서양, 태평양 등) 데이터로 일반화 성능 검증  
- **추적 알고리즘 통합**: 후처리 단계에서 Kalman 필터 등 시계열 추적 기법 결합  
- **불확실성 정량화**: Bayesian 딥러닝 기법으로 예측 신뢰도 산출 및 ghost eddy 재검출 강화  

이상의 발전 방향을 통해 EddyNet은 해양 순환 연구, 기후 모델링, 해양 예측 시스템의 핵심 도구로 자리매김할 수 있을 것이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/0f1cf875-ab18-4418-ab28-2a0e29630ba9/1711.03954v1.pdf)
