# 장기 실종아동 나이 변환(Age Progression) 시스템 구축 가이드

이 문서는 장기 실종아동의 현재 모습을 예측하기 위한 **나이 변환(Age Progression) AI 시스템**의 아키텍처와 구축 방법론을 정리한 가이드입니다.

---

## 1. 로컬 파인튜닝 vs 상용 API 비교

실종아동 예측 시스템은 보안과 정밀한 생체 특징 제어가 필수적이므로, 상용 API보다는 **로컬 환경(온프레미스 또는 폐쇄망)에서 전용 모델을 파인튜닝하는 방식**이 적합합니다.

### 로컬 구축이 필수적인 이유
*   **신원 보존(Identity Preservation):** 상용 API(Midjourney, DALL-E 등)는 프롬프트에 의존하여 아동의 고유한 골격과 생체 비대칭성을 유지하기 어렵습니다. 로컬 전용 파이프라인은 얼굴 일치도($Loss_{ID}$)를 강제할 수 있습니다.
*   **데이터 보안 및 규제:** 실종아동 및 친족의 사진은 민감한 생체 데이터입니다. 상용 클라우드 API로 전송 시 개인정보보호법 및 공공기관 보안 규정에 저촉될 위험이 큽니다.
*   **한국인 특화 데이터 학습:** 한국인 특유의 안면 노화 패턴(K-Face 데이터셋 등)을 온전히 반영하기 위해서는 모델 가중치를 직접 조정(LoRA 또는 Full Fine-tuning)해야 합니다.

---

## 2. 추천 오픈소스 백본(Backbone) 모델

### A. StyleGAN 기반 모델 (추천: 높은 정체성 보존력)
*   **핵심 모델:** `StyleGAN2-ADA`, `StyleGAN3`
*   **관련 프로젝트:** `SAM (Style-based Age Manipulation)`, `InterFaceGAN`
*   **특징:** 잠재 공간(Latent Space)에서 나이 방향 벡터(Age Vector)를 이동시키는 방식으로, 기존 골격과 정체성 유지 성능이 가장 안정적으로 검증되어 있습니다.

### B. Diffusion 기반 모델 (추천: 최신 트렌드 및 유연한 제어)
*   **핵심 모델:** `Stable Diffusion`, `ControlNet`, `IP-Adapter`
*   **관련 프로젝트:** `IP-Adapter-FaceID`, `DiffAE`
*   **특징:** 랜드마크와 얼굴 임베딩을 고정한 상태에서 디노이징을 수행하며, 세부적인 피부 질감이나 환경 제어에 유리합니다.

---

## 3. 기본 파이프라인 아키텍처

로컬에 구축되는 딥러닝 파이프라인은 다음 4단계로 구성됩니다.

1.  **데이터 전처리 (Preprocessing):** `RetinaFace` 등을 활용해 얼굴 랜드마크 기준 정렬(Alignment) 및 크롭 수행. 한국인 종단 데이터셋으로 증강.
2.  **인코딩 및 조건 부여 (Conditioning):** 아동 사진을 인코더에 통과시켜 특징 벡터($w$)를 추출하고, 타겟 연령 벡터를 생성기에 주입.
3.  **이미지 생성 (Generation):** 조건이 결합된 벡터를 기반으로 모델 디코더가 타겟 연령의 얼굴 이미지 합성.
4.  **손실 함수 최적화 (Loss Optimization):** 
    $$ Loss_{Total} = \lambda_1 Loss_{Recon} + \lambda_2 Loss_{ID} + \lambda_3 Loss_{Age} $$
    *   $Loss_{Recon}$: 원본 이미지 복원 오차 방지
    *   $Loss_{ID}$: `ArcFace` 등을 통한 원본-생성본 간 정체성 일치도 평가
    *   $Loss_{Age}$: `MiVOLO` 등을 통한 목표 나이 도달 여부 평가

---

## 4. 부모 유전 정보 반영 (Kinship Guided Generation)

유아기 사진만으로 예측하기 어려운 성인기의 얼굴 특징(하관, 눈매 등)을 보완하기 위해 친부모의 사진을 레퍼런스로 활용하는 고급 기법입니다.

### 구현 로직
1.  **특징 추출 (Feature Extraction):** `ArcFace` 등의 인코더로 실종아동($V_C$)과 친부모($V_F, V_M$)의 얼굴 임베딩을 각각 추출합니다.
2.  **유전적 특징 융합 (Kinship Feature Fusion):** 
    *   **Cross-Attention** 메커니즘을 사용하여 아동의 특징(Query)에 부모의 특징(Key/Value)을 매핑합니다.
    *   아동과 더 닮은 부모의 특정 부위(예: 엄마의 눈, 아빠의 턱) 가중치가 자동으로 학습되어 결합된 **Kinship Vector($V_{Kin}$)**가 생성됩니다.
3.  **조건부 노화 생성:** 아동의 원본 벡터에 '타겟 나이 조건'과 융합된 '$V_{Kin}$'을 함께 주입하여 모델을 통과시킵니다.
4.  **Kinship Loss 최적화:** 
    가족 관계 검증 데이터셋(FIW 등)으로 학습된 샴 네트워크(Siamese Network)를 활용하여, 생성된 얼굴이 부모와 유전적 유사성을 띠도록 강제하는 손실 함수를 추가합니다.
    $$ Loss_{Total} = \lambda_1 Loss_{Recon} + \lambda_2 Loss_{ID} + \lambda_3 Loss_{Age} + \lambda_4 Loss_{Kinship} $$

이러한 Kinship 유도 방식을 적용하면, 무작위적인 노화가 아닌 부모의 유전적 궤적을 따라가는 훨씬 정교하고 현실적인 추정 이미지를 얻을 수 있습니다.