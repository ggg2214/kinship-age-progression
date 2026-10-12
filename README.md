# Kinship-Guided Age Progression System for Long-Term Missing Children
> **장기 실종아동 나이 변환(Age Progression) 및 친부모 유전 정보(Kinship Cross-Attention) 기반 AI 시스템**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ggg2214/kinship-age-progression/blob/main/notebooks/train_colab.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 개요 (Overview)

본 프로젝트는 장기 실종아동의 현재 모습을 예측하기 위해 **친부모 유전 정보(Kinship Guided Generation)**와 **생체 안면 정체성 보존(Identity Preservation)**을 결합한 딥러닝 기반 나이 변환(Age Progression) 시스템입니다.

유아기 사진만으로는 예측하기 어려운 성인기의 하관, 이목구비 골격 변화를 친부모 사진의 Cross-Attention 융합 벡터($V_{Kin}$)를 통해 보완하며, 부모 사진이 결측된 경우에도 자가 폴백(Self-Conditioning) 모드로 안정적으로 동작합니다.

---

## 🏗️ 시스템 아키텍처 (System Architecture)

```
[Child Face Image] -------> [ ArcFace Encoder ] ---------> V_C (512-d)
                                                                 │ (Query)
[Father Face Image] ------> [ ArcFace Encoder ] ──> V_F ──┐      │
                                                          ├──> [ Cross-Attention ] ──> V_Kin (512-d)
[Mother Face Image] ------> [ ArcFace Encoder ] ──> V_M ──┘      │ (Key/Val)           │
                                                                                       │
[Target Age / Delta] -----> [ Age Conditioner ] ─────────> V_Age (512-d) ──────────────┼──> [ Conditioning Fusion ]
                                                                                       │            │
                                                                                       │         W_cond
                                                                                       ▼            ▼
                                                                             [ AdaIN Generator ] ──> [ Aged Face Image ]
```

### 1. 핵심 모듈 구성
1. **ArcFace Encoder (`models/arcface_encoder.py`)**: 아동($V_C$) 및 친부모($V_F, V_M$)의 얼굴에서 고유 안면 골격을 보존하는 512차원 정규화 임베딩 추출.
2. **Kinship Cross-Attention Fusion (`models/kinship_fusion.py`)**: 아동 특징(Query)에 부모 특징(Key/Value)을 매핑하여 특정 부위(하관, 눈매 등) 가중치를 동적으로 결합한 $V_{Kin}$ 산출 (부모 결측 시 Fallback 지원).
3. **Age Conditioner & AdaIN Generator (`models/age_conditioner.py`, `models/generator.py`)**: 목표 나이 Fourier 임베딩과 AdaIN 잔차 블록을 결합하여 자연스러운 노화(주름, 피부 탄력 등) 합성.
4. **4단계 복합 손실 최적화 (`losses/total_loss.py`)**:
   $$ Loss_{Total} = \lambda_1 Loss_{Recon} + \lambda_2 Loss_{ID} + \lambda_3 Loss_{Age} + \lambda_4 Loss_{Kinship} $$

---

## 🖼️ 시각화 및 시연 (Visualizations)

### 1. 연령별 성장 파노라마 (Age Progression Timeline)
`outputs/sample_family_timelapse.png` 예시:
![Age Progression Timeline](outputs/sample_family_timelapse.png)

### 2. 대화형 Web UI (FastAPI Dashboard)
- 아동 및 부모 사진 드래그 앤 드롭 업로드 & 샘플 원클릭 채우기
- 실시간 타겟 나이 슬라이더 조절 (5세 ~ 75세)
- 안면 정체성 보존도 (ArcFace 코사인 유사도) & 친부/친모 유전 기여도 (%) 시각화 차트

---

## 🚀 빠른 시작 (Quick Start)

### 1. 설치 (Installation)
```bash
git clone https://github.com/ggg2214/kinship-age-progression.git
cd kinship-age-progression

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install fastapi uvicorn python-multipart Pillow
```

### 2. Google Colab 무료 T4 GPU 원클릭 학습 (Recommended)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ggg2214/kinship-age-progression/blob/main/notebooks/train_colab.ipynb)
- 상단 뱃지를 클릭하여 Google Colab에서 256x256 해상도 대규모 모델 학습 및 Google Drive 가중치 백업을 바로 실행할 수 있습니다.

### 3. 실제 데이터셋 수집 및 Triplet 매핑 (Dataset Setup)
```bash
python age_progression/data/download_datasets.py --data-dir data_utkface
```

### 4. 저화질 아동 사진 복원 및 선명화 (Face Enhancement)
```bash
python age_progression/data/face_enhancer.py --input blurry_child.jpg --output enhanced_child.png
```

### 5. 대화형 웹 인터페이스 실행 (Web UI)
```bash
python age_progression/web_app.py
```
브라우저에서 `http://127.0.0.1:8000` 접속 (웹에서 AI 안면 복원 토글 지원)

### 6. 연령별 타임랩스 배너 생성 CLI
```bash
python age_progression/visualize_samples.py \
  --child sample_data/family_001/child_age06.jpg \
  --father sample_data/family_001/father_age38.jpg \
  --mother sample_data/family_001/mother_age35.jpg \
  --ages 10 20 30 40 50 60 \
  --output outputs/my_timelapse.png
```

### 7. 소거 연구(Ablation Study) 벤치마크 평가
```bash
python age_progression/evaluate.py --num-samples 16
```

---

## 📁 프로젝트 구조 (Project Directory)

```
.
├── notebooks/
│   └── train_colab.ipynb         # [Colab] Google Colab 무료 T4 GPU 원클릭 학습 노트북
├── age_progression/
│   ├── config.py                 # 하이퍼파라미터 및 손실 가중치 설정
│   ├── demo.py                   # 모델 초기화, Forward, Multi-loss 통합 검증
│   ├── evaluate.py               # 정량 평가 및 소거 연구(Ablation) 벤치마크
│   ├── inference.py              # 나이 변환 엔드투엔드 추론 파이프라인 (복원 지원)
│   ├── train.py                  # 모델 학습 루프 및 체크포인트 저장기
│   ├── visualize_samples.py      # 연령대별 연속 변환 그리드 생성 스크립트
│   ├── web_app.py                # FastAPI 웹 애플리케이션 및 REST API
│   ├── static/                   # 웹 대시보드 프론트엔드 (HTML/CSS, 복원 토글)
│   ├── data/
│   │   ├── dataset.py            # Triplet 데이터셋 로더
│   │   ├── download_datasets.py  # 실제 UTKFace/연령 데이터셋 다운로더 & 매핑기
│   │   ├── face_enhancer.py      # 저화질/블러 아동 사진 복원 및 선명화기
│   │   ├── prepare_dataset.py    # 데이터셋 파싱 및 매니페스트 빌더
│   │   └── transforms.py         # 얼굴 정렬 및 전처리 파이프라인
│   ├── losses/                   # 4단계 복합 손실 함수 (Recon, ID, Age, Kinship)
│   └── models/                   # ArcFace, Kinship Fusion, AdaIN Generator 모델
├── checkpoints/                  # 학습 가중치 저장 디렉토리
├── sample_data/                  # 검증용 합성 가족 Triplet 데이터
└── outputs/                      # 생성된 타임랩스 및 분석 결과 이미지
```

---

## 📄 라이선스 (License)
MIT License

