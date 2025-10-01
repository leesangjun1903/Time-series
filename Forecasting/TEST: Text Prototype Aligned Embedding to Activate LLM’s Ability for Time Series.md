# TEST: Text Prototype Aligned Embedding to Activate LLM’s Ability for Time Series  

**핵심 주장 및 주요 기여**  
이 논문은 다변량·단변량 시계열(Time Series, TS)을 별도의 대규모 모델 학습 없이도 기존의 *frozen* 대형 언어 모델(LLM)에 적용 가능하도록 하는 **TS-for-LLM** 패러다임을 제안한다. 핵심 기여는 다음과 같다:  
- TS‐for‐LLM과 LLM‐for‐TS 두 가지 패러다임 정리  
- **TEST**(TimE Series embedding aligned with Text prototype) 방법 제안  
- 인스턴스 대비(instance-wise), 특성 대비(feature-wise), 텍스트 프로토타입 정렬(text‐prototype‐aligned) 대조 학습을 결합한 TS 임베딩  
- 프롬프트 튜닝이 지도 학습 미세조정(supervised fine‐tuning)과 거의 등가함을 이론적으로 증명  
- 분류, 예측, 표현 학습에서 다양한 *frozen* LLM에 적용 시 SOTA 수준 성능 또는 근접 성능 달성  

---  

## 1. 해결 문제  
현재 LLM은 자연어 및 시각 분야에서 탁월하나, 시계열 데이터를 직접 처리할 수 있는 능력은 제한적이다.  
- 멀티변량 TS는 텍스트 토큰 하나하나처럼 일렬로 입력할 경우 종속성 학습이 어렵고 성능이 낮음  
- 모델 중심(LLM‐for‐TS): TS 특화 모델을 새로 훈련하거나 기존 LLM을 대규모 미세조정해야 함  
- **데이터 중심(TS‐for‐LLM)**: TS를 LLM이 이해 가능한 형태로 변환해 *frozen* LLM으로 처리  

## 2. 제안 방법 TEST  

1. **TS 토큰화 & 인코딩**  
   - TS $$x \in \mathbb{R}^{T\times D}$$를 슬라이딩 윈도우로 세그먼트 $$s_k$$로 분할  
   - 인코더 $$f_e$$로 각 세그먼트 임베딩 $$e_k=f_e(s_k)\in\mathbb{R}^M$$ 생성  

2. **인스턴스-와이즈 & 특성-와이즈 대조 학습**  
   - 인스턴스-대조: 동일 세그먼트 증강 쌍(anchor/positive)은 가깝게, 나머지(batch 내) negative는 멀게  

```math
       \mathcal{L}_{\mathrm{ins}}
       =-\sum_i \log\frac{\exp(\text{sim}(f_p(e_i),f_p(e_i^+))/\tau)}
       {\sum_{j}\exp(\text{sim}(f_p(e_i),f_p(e_j))/\tau)}
``` 
   
   - 특성-대조: 배치 임베딩 행(row)은 인스턴스, 열(column)은 특성별 소프트 레이블로 취급해 열 간 대조  

3. **텍스트 프로토타입 정렬 대조**  
   - LLM의 텍스트 토큰 임베딩 공간에서 $$P$$개의 프로토타입 $$t_p$$ 선택  
   - TS 임베딩과 텍스트 프로토타입 간 정렬 및 대조:  

$$
       \mathcal{L}_{\mathrm{text}}
       =-\sum_{p}\sum_{i}\log\frac{\exp(\text{sim}(t_p,e_i)/\tau)}
       {\sum_{q}\exp(\text{sim}(t_q,e_i)/\tau)}
     $$  

4. **소프트 프롬프트 튜닝**  
   - TS 임베딩과 학습된 소프트 프롬프트 $$p_e$$를 LLM 입력층에 삽입  
   - 프롬프트 손실  

$$
       \mathcal{L}_{\mathrm{prompt}}
       =\mathcal{L}_{\mathrm{task}}(\mathrm{LLM}(p_e, \{e_i\}), y)
     $$  
   
   - 이로써 LLM 미세조정 없이 TS 작업 수행 가능함을 이론적으로 증명  

5. **모델 구조**  
   - **인코더**: 10-layer causal TCN (DilatedConv+GELU+BatchNorm) 출력 차원 = LLM 임베딩 차원  
   - **LLM**: GPT-2, BERT, LLaMA2 등 117M–13B 규모 모델  

## 3. 성능 향상 및 한계  

- **분류**: UCR/UAE 데이터셋에서 원본 LLM 랜덤 추측 수준에서 정확도 18–25%p 상승, 300M급 모델부터 TS SOTA 초과[1]
- **예측(단기/장기/제로샷/퓨샷)**: ETT, Weather 등 벤치마크에서 SOTA TS 전용 모델과 근접 또는 상회  
- **표현 학습**: TS2Vec 등 대비 단순 SVM 평가에서도 경쟁력 있는 표현 획득  
- **제약**:  
  - LLM 크기·사전학습 데이터 구성에 민감  
  - 텍스트 프로토타입 수/초기화 방식에 따라 성능 변동(10개 프로토타입 권장)  
  - TS 범위·도메인 특성 완전 반영 어려움  

## 4. 일반화 성능 향상 관점  
- **다중 데이터셋 융합 평가**: 19개 TS 데이터 통합 테스트에서 LLM+TEST 모델이 전용 모델 대비 뛰어난 *일반화* 능력 입증  
- **퓨샷 학습**: 단 10% 학습샘플만으로도 평균 MSE 23.5% 감소, 데이터 부족 환경에 강함  
- **프로토타입 정렬**: 고차원 공간에서 거의 직교하는 프로토타입 활용으로 모델 크기가 변해도 안정적  

## 5. 향후 영향 및 연구 고려사항  
TEST는 **패턴 머신**으로서의 LLM 잠재력을 TS 분야에 확장한다. 향후 연구에서 고려할 점은:  
- **인간 지각 수준 정렬**: 기계 수준에서 TS·텍스트를 정렬했으나, 인간이 이해 가능한 설명성(interpretable) 보장 필요  
- **프로토타입 자동 선정**: 도메인별 최적 프로토타입 클러스터링 및 동적 조정 기법 연구  
- **다중모달 통합**: 이미지·텍스트·TS를 결합한 멀티모달 LLM 응용 확대  
- **자기지도학습 심화**: 더 다양한 CL 손실 및 예측 과제를 결합해 임베딩 품질 향상  

TEST는 TS 처리에 있어 LLM 재사용 가능성 및 효율적 확장성을 제시하며, 향후 TS‐LLM 융합 연구의 토대를 마련한다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a1990491-0c63-4be7-a6d2-656bfa4e414b/2308.08241v2.pdf)
