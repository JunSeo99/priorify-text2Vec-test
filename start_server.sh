#!/bin/bash

# 서버 시작 스크립트 for EC2
echo "🚀 Priorify API 서버 시작 중..."

# 환경 변수 설정
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MODEL_PATH="${MODEL_PATH:-models/finetuned_early_stopping}"
export PORT="${PORT:-8000}"
export WORKERS="${WORKERS:-1}"

# GPU 사용 가능 여부 확인
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU 사용 가능"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
else
    echo "ℹ️  CPU 모드로 실행"
fi

# 모델 파일 존재 확인
if [ -d "$MODEL_PATH" ]; then
    echo "✅ 모델 경로 확인: $MODEL_PATH"
else
    echo "⚠️  모델 경로 없음: $MODEL_PATH"
    echo "   기본 모델을 다운로드합니다..."
fi

# 서버 시작
echo "🌟 서버 시작: http://0.0.0.0:$PORT"
echo "   - Workers: $WORKERS"
echo "   - Model Path: $MODEL_PATH"

# 프로덕션 모드로 uvicorn 실행
uvicorn src.api.server:app \
    --host 0.0.0.0 \
    --port $PORT \
    --workers $WORKERS \
    --access-log \
    --loop uvloop \
    --http httptools

# 또는 개발 모드 (reload 포함)
# uvicorn src.api.server:app --host 0.0.0.0 --port $PORT --reload 