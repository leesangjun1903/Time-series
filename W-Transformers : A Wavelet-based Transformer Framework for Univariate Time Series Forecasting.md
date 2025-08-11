# W-Transformers : A Wavelet-based Transformer Framework for Univariate Time Series Forecasting

**핵심 주장**  
W-Transformers는 비정상(non-stationary) 및 비선형(un-linear) 특성을 지닌 단변량 시계열의 장·단기 의존성을 동시에 효과적으로 포착하여 예측 성능을 크게 향상시킬 수 있음을 보인다.

**주요 기여**  
1. **MODWT와 Transformer의 결합**  
   - 최대 중첩 이산 웨이브릿 변환(MODWT)을 통해 시계열을 고주파(detail) 및 저주파(smooth) 성분으로 분해  
   - 각 분해 레벨별로 독립적인 Transformer 모델을 학습시킨 뒤, 역 MODWT로 통합 예측  
2. **비정상성 및 장·단기 의존성 처리**  
   - 웨이브릿 분해로 비정상성과 노이즈를 분리  
   - Transformer의 self-attention 메커니즘으로 장기적 비선형 의존성 캡처  
3. **광범위한 실험 검증**  
   - 7개 공개 데이터세트(주가, 유입 트래픽, 일일 태양흑점, 전염병 등)에서 단기·장기 예측 모두 우수한 성능 입증  
   - MCB(multiple comparisons with the best) 검정을 통한 통계적 우수성 확인  

***

# 상세 설명

## 1. 해결하고자 하는 문제  
- 전통 통계 모델(ARIMA, ETS 등)은 비정상·비선형 시계열에서 한계  
- RNN 계열(LSTM, GRU)은 장기 의존성 학습 어려움  
- 기존 Transformer 기반 접근법은 비정상성 처리 미흡  

## 2. 제안 방법  
### 2.1 웨이브릿 분해 (MODWT)  
- MODWT를 적용하여 원시 시계열 $$Y_t$$를 $$J$$개 주파수 대역의 디테일 성분 $$D_{j,t}$$와 스무스 성분 $$S_{J,t}$$으로 분해  

$$
    Y_t = \sum_{j=1}^{J} D_{j,t} + S_{J,t}
  $$

- 각 성분은 원본 길이와 동일하며, 노이즈 제거 및 비정상성 완화  

### 2.2 Transformer 예측  
- 분해된 각 시계열 성분별로 독립적인 Transformer 인코더-디코더 학습  
- Self-attention:  

$$
    \mathrm{Attn}(Q,K,V) = \mathrm{softmax}\Bigl(\frac{QK^\top}{\sqrt{d}}\Bigr)V
  $$

- 다중 헤드:  

$$
    \mathrm{MultiHead}(Q,K,V) = \mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_m)W^O
  $$

- 학습된 예측값 $$\hat{D}\_{j,N+h}, \hat{S}\_{J,N+h}$$을 역 MODWT로 합성하여 최종 $$\hat{Y}_{N+h}$$ 생성  

### 2.3 모델 구조  
- **웨이브릿 분해부**: Haar 필터 기반 MODWT  
- **Parallel Transformer Ensemble**:  
  - 각 분해 레벨별 입력(Embedding1) → 인코더 → 디코더1 → 디코더2 → 선형+Softmax  
  - 학습 후 예측부에서 반복적 다단계 예측  

## 3. 성능 향상 및 한계  
- **성능 향상**  
  - 단기 예측(RMSE, MAE, sMAPE, MASE)에서 주요 데이터세트 대부분에서 1위 달성  
  - 장기 예측에서 특히 우수, 전통·딥러닝 기법 대비 평균 순위 1.12로 통계적 우수성 확보  
- **한계**  
  - 소규모 데이터세트에서 통계 모델(ETS 등)에 비해 성능 열세  
  - 단기 예측 및 매우 짧은 데이터 길이 개선 필요  
  - 다변량 시계열 확장성 미검증  

***

# 일반화 성능 향상 관점

- **웨이브릿 분해 강건성**  
  - 비정상성·노이즈 분리로 다양한 도메인에 적용 가능성  
- **모듈식 Transformer 앙상블**  
  - 각 주파수 대역별 특화 학습 → 도메인 편향 감소  
- **하이퍼파라미터 민감도**  
  - 분해 레벨 $$J$$, 헤드 수 $$m$$, 모델 차원 $$d$$ 조정으로 일반화 균형 조절  

***

# 향후 연구 방향 및 고려 사항

- **소규모 데이터 대응**:  
  - 데이터 증강, 전이 학습(Transfer Learning) 도입  
- **단기 예측 성능 개선**:  
  - 단기용 로컬 Attention 변형 또는 시계열 특화 Positional Encoding 연구  
- **다변량 확장**:  
  - 다변량 MODWT 적용 및 상호 레벨 간 상관성 학습 메커니즘 설계  
- **실시간 예측 시스템화**:  
  - 효율적 모듈 병렬화 및 경량화 모델 최적화 고려  

W-Transformers는 비정상·비선형 시계열 예측 프레임워크로서 Transformer 기반 모델의 범용성을 강화하는 중요한 첫걸음이며, 위 제언을 통해 더욱 폭넓고 견고한 시계열 예측 연구가 가능할 것이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/91714be6-ce28-498b-a954-89b8b0486102/2209.03945v1.pdf
