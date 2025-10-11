# AirFormer: Predicting Nationwide Air Quality in China with Transformers

**핵심 주장 및 주요 기여**  
AirFormer는 중국 전역 1,085개 측정소의 미세먼지(PM2.5) 농도를 72시간까지 예측하는 데 특화된 새로운 Transformer 기반 구조이다.  
1. **이중 단계 학습**: 공간·시간 표현을 효율적으로 학습하는 **결정론적 단계**와 데이터 불확실성을 포착하는 **확률론적 단계**로 분리하여 모델링 효율성과 견고성을 동시에 확보했다.[1]
2. **Dartboard Spatial MSA (DS-MSA)**: 반경별·방향별로 주변 지역을 구획화해 선형 시간 복잡도 $$O(NMC)$$로 공간 의존성을 잡아낸다.[1]
3. **Causal Temporal MSA (CT-MSA)**: 인과적 마스킹과 점진적 수용 영역 확장으로 시계열 의존성을 효율적·정확하게 포착한다.[1]
4. **확률적 잠재변수**: 계층적 잠재변수를 도입해 불확실성을 모델링, 급격한 변화 예측 성능을 개선했다.[1]
5. **성능 개선**: 72시간 예측 시 기존 최첨단 대비 MAE를 58% 감소시켰다.[1]

***

## 1. 문제 정의  
과거 $$T$$시점의 모든 측정소 입력 $$\mathbf{X}_{1:T}\in \mathbb{R}^{T\times N\times D}$$로부터 미래 1시점 $$\mathbf{Y}\in \mathbb{R}^{N\times D}$$을 예측하고, 이를 $$H$$시점 예측까지 확장하는 함수  

$$
\mathbf{Y}_{1:H} = F(\mathbf{X}_{1:T})
$$

를 학습한다.[1]

***

## 2. 제안 방법

### 2.1 결정론적 단계  
1) **입력 임베딩**: 다층 퍼셉트론으로 $$\mathbf{X}_{1:T}$$ → $$\mathbf{H}^0\in\mathbb{R}^{T\times N\times C}$$.  
2) **AirFormer 블록(총 $$L$$개)**  
   - **DS-MSA**: 각 측정소 주변을 반경 $$r_1<r_2<\dots$$ 원과 방위선으로 구획해 $$M$$개 지역 표현 $$\mathbf{R}_i = A_i\mathbf{P}\in\mathbb{R}^{M\times C}$$로 투영.[1]

$$
       \text{MSA}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \text{Softmax}\Bigl(\frac{\mathbf{QK}^\top}{\sqrt{d}}\!+\!B\Bigr)\mathbf{V}
     $$
   
  - **CT-MSA**: 비중첩 윈도우 $$(W)$$ 내 국소적 인과적(attention mask) 자기주의를 수행해 계산량을 $$O(TWC)$$로 절감.[1]

### 2.2 확률론적 단계  
계층적 잠재변수 $$\mathbf{Z}^l_t$$를 VAE 방식으로 도입해 불확실성을 모델링:  
- **Prior**  

$$
  p(\mathbf{Z}_t|\mathbf{X}_{1:t-1})
  = \prod_{l=1}^L\prod_{n=1}^N\mathcal{N}\bigl(\mu^l_{t,n},\,\sigma^l_{t,n}\bigr)
  $$

- **Inference**  

$$
  q(\mathbf{Z}_t|\mathbf{X}_{1:t})
  = \prod_{l=1}^L\prod_{n=1}^N\mathcal{N}\bigl(\tilde\mu^l_{t,n},\,\tilde\sigma^l_{t,n}\bigr)
  $$

- **ELBO 결합 최적화**  

```math
  \mathcal{L} = \underbrace{\|\hat{\mathbf{Y}}-\mathbf{Y}\|_1}_{\mathcal{L}_{\mathrm{pred}}}
  + \underbrace{\sum_t\Bigl(\!-\mathbb{E}_{q}\log p+\mathrm{KL}[q\|p]\Bigr)}_{\mathcal{L}_{\mathrm{ELBO}}}
``` 
  
으로 예측 정확도와 불확실성 학습을 동시에 극대화한다.[1]

***

## 3. 모델 구조  
```
Input  → MLP → [DS-MSA → CT-MSA]ₓₗ → Deterministic H
                                     ↓
                             Latent Z via VAE
                                     ↓
                        MLP → Future PM2.5 Prediction
```
- 블록 수 $$L=4$$, 임베딩 차원 $$C=32$$, 헤드 수 2, 윈도우 크기 $$$$ 사용.[1]

***

## 4. 성능 및 한계

**성능 향상**:  
- 1~24h MAE 16.03 → 두 번째 모델 대비 8.2 감소  
- 25~48h MAE 21.65 → 7.5 감소  
- 49~72h MAE 23.64 → 5.3 감소  
- 급격 변화 예측 MAE 54.92 → 7.3 개선.[1]

**한계**:  
- **계산 자원**: DS-MSA 블록·윈도우 확장 시 연산·메모리 증가  
- **데이터 의존성**: 4년치 중국 전역 측정소 데이터에 맞춘 하이퍼파라미터 최적화 필요  
- **실시간성**: 온라인 학습 및 실시간 예측 시스템 통합 미구현

***

## 5. 일반화 성능 향상 관점  
AirFormer는 도메인 지식(풍향·거리 기반 attention)과 Transformer의 범용 표현력, VAE의 불확실성 모델링을 결합해 다양한 지역·시계열 예측 과제에 적용 가능하다. 특히 잠재변수 계층 구조가 시공간 종속성을 유연히 포착하므로, 데이터 밀도·분포가 다른 다른 국가나 도시로도 적응력이 높다.

***

## 6. 향후 연구 방향 및 고려 사항  
- **온라인·연속 학습**: 실시간 스트리밍 데이터 대응  
- **다중 오염물질 모드**: PM10, NO₂ 등 다변량 예측  
- **경량화**: 엣지 디바이스 배포를 위한 모델 압축·양자화  
- **교차지역 일반화**: 다른 국가·도시별 환경 차이 극복을 위한 도메인 어댑테이션  
- **정책 의사결정 통합**: 예측 결과를 기반으로 한 정부·시민용 리포트 자동 생성 시스템  


[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/03786a7a-f6df-4fd2-93bd-3b84df839aa5/2211.15979v1.pdf)
