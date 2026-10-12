---
title: Kinship Age Progression API
emoji: 👶
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Kinship Age Progression Cloud AI Backend
장기 실종아동 나이 변환(Age Progression) 및 친부모 유전 정보(Kinship Cross-Attention) 기반 24시간 무료 클라우드 AI 추론 서버입니다.

## 주요 기능
- `POST /api/predict`: 아동 및 친부모 사진을 입력받아 지정된 나이로 변환된 얼굴 생성
- `POST /api/timelapse`: 10세부터 60세까지 연속 성장 타임랩스 생성
- `GET /api/status`: 서버 상태 및 GPU 가용성 확인
- `GET /docs`: Swagger 대화형 API 문서
