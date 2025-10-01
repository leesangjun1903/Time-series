# ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis

**주요 주장:** 전통적인 TCN(Temporal Convolutional Network)을 현대화(modern convolution)하여 순수 합성곱 기반 구조만으로도 Transformer·MLP 계열 모델과 동등한 수준의 시계열 분석 성능을 달성하면서도, 합성곱의 효율성을 유지할 수 있음을 입증한다.[1]

**주요 기여:**  
- **모델 일반화:** 시계열 예측·임퓨테이션·분류·이상 탐지 등 5개 주류 과제에서 일관된 최첨단 성능 달성.[1]
- **효율성·성능 균형:** Transformer 대비 메모리·연산 효율성을 유지하면서 성능 경쟁력 확보.[1]
- **확장된 ERF:** 대형 커널 적용 및 차원별 분리 설계를 통해 기존 합성곱 모델보다 넓은 유효 수용장(ERF)을 확보.[1]

***

## 1. 해결하고자 하는 문제  
Transformer·MLP 기반 모델이 시계열 장기 의존성을 잘 포착해 우수한 성능을 보이는 반면, 전통 TCN은 제한된 유효 수용장(ERF)으로 인해 경쟁력이 떨어졌다. 기존 합성곱 시계열 모델들은 복잡한 구조를 추가하나, 커널 자체 최적화에는 소홀하여 여전히 한계를 드러냈다.[1]

***

## 2. 제안 방법  
### 2.1 ModernTCN 블록 구조  
ModernTCN은 1D 합성곱 블록을 다음 세 가지 모듈로 분리·최적화한다(그림 참조).  
1. **DWConv (Depthwise Conv):**  
   - 입력 채널 수 = $$M \times D$$, 그룹 수 = $$M\times D$$로 설정하여 시계열 변수·특징 차원 독립 처리  
   - 대형 커널 $$k$$ 적용으로 ERF 선형 증가($$\propto k$$)  
2. **ConvFFN₁:**  
   - 그룹 수 = $$M$$인 점-와이즈 합성곱으로 변수별 특징 표현 학습  
3. **ConvFFN₂:**  
   - 그룹 수 = $$D$$인 점-와이즈 합성곱으로 특징별 변수 간 의존성 학습  

### 2.2 수식 개요  
- 입력 임베딩:  

$$
    X_\text{emb} = \mathrm{Conv1D}(X_\mathrm{in}),\quad X_\mathrm{in}\in\mathbb R^{M\times L}
  $$

- 블록 내부 순전파 ($$i$$번째 블록):  

$$
    Z^{(i)} = Z^{(i-1)} + \bigl[\mathrm{DWConv}(Z^{(i-1)}) + \mathrm{ConvFFN}_1(\cdot) + \mathrm{ConvFFN}_2(\cdot)\bigr]
  $$

- 출력 표현:  

$$
    Z = \mathrm{Block}_K\bigl(\cdots\mathrm{Block}_1(X_\text{emb})\bigr)
  $$  
  
  여기서 $$M$$은 변수 수, $$D$$는 특징 차원, $$L$$은 시계열 길이이다.[1]

***

## 3. 모델 구조  
ModernTCN은 입력 임베딩 → K개 ModernTCN 블록 스택 → 과제별 헤드(선형·소프트맥스 등)로 구성된다.  
- **임베딩:** 패치화된 변수 독립 1D 합성곱(stem layer)  
- **블록 스택:** $$K$$개 ModernTCN 블록(잔차 연결)  
- **헤드:** 회귀 과제는 선형, 분류 과제는 소프트맥스 헤드  

***

## 4. 성능 향상 및 한계  
### 4.1 성능  
- **5개 과제(state-of-the-art):** 장·단기 예측, 임퓨테이션, 분류, 이상 탐지 전반에서 Transformer·MLP 모델 대비 동등하거나 더 우수.[1]
- **ERF 비교:** 기존 합성곱 모델(MICN, SCINet)보다 넓은 ERF 분포로 장기 의존성 포착 우수.[1]

### 4.2 효율성  
- **메모리·연산:** Transformer 대비 GPU 메모리 사용량 및 학습 속도 우세  
- **단순 구현:** 완전 합성곱 구조로 추가 모듈 불필요  

### 4.3 한계  
- **매개변수 민감도:** 소규모 데이터(ILI 등)에서 모델 크기·FFN 배율 조정 필요  
- **대형 커널 훈련 안정성:** Structural re-parameterization 적용 권장  

***

## 5. 일반화 성능 향상 관련 고찰  
ModernTCN의 **차원별 분리 설계**는 교차 변수·특징·시간 의존성을 각각 전용 모듈로 처리하여  
- **특징 간섭 최소화**  
- **효율적 의존성 학습**  
- **일관된 과제 일반화**  
를 가능하게 한다. 특히 멀티태스크 환경에서 모든 과제에서 성능 우위가 관찰되었다.[1]

***

## 6. 향후 연구 과제 및 고려 사항  
- **Adaptive Kernel:** 데이터 특성에 따라 커널 크기 동적 조정 연구  
- **Self-supervised 학습:** 대규모 미라벨 시계열 사전학습으로 일반화 강화  
- **경량화·하드웨어 최적화:** 임베디드 디바이스 배포를 위한 모델 축소 기술 적용  
- **비정상 분포 처리:** 분포 이동 대응을 위한 RevIN 등 노멀라이제이션 기법 결합  

ModernTCN은 순수 합성곱 기반 접근의 재조명을 촉진하며, 향후 시계열 기반 **범용** 딥러닝 백본으로 확장 가능성을 제시한다.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/36213209-ed31-41dd-b3f1-d9085f75c8d0/5228_ModernTCN_A_Modern_Pure_C.pdf)
