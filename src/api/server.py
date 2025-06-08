# app.py
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import torch
import os
import logging

from src.config import CATEGORIES_DEFINITIONS, DEFAULT_OUTPUT_PATH
from src.core.model_utils import ner_generalize_texts

# --- 로깅 및 환경 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- FastAPI 앱 및 모델 로드 ---
app = FastAPI(title="Priorify Category Prediction API", version="4.0")

class TextRequest(BaseModel):
    text: str

# 모델은 애플리케이션 시작 시 한 번만 로드합니다.
# 환경 변수나 설정 파일을 통해 모델 경로를 지정하는 것이 더 유연합니다.
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_OUTPUT_PATH)
model = None
categories = []
category_embeddings = None

@app.on_event("startup")
def load_model():
    """애플리케이션 시작 시 v4 모델 및 카테고리 임베딩 로드"""
    global model, categories, category_embeddings
    
    if not os.path.exists(MODEL_PATH):
        logging.error(f"모델 파일 없음: {MODEL_PATH}")
        logging.error("finetune.py를 먼저 실행하여 모델을 생성하세요!!")
        raise RuntimeError(f"Model not found at {MODEL_PATH}")

    logging.info(f"파인튜닝된 v4 모델 로딩: {MODEL_PATH}")
    model = SentenceTransformer(MODEL_PATH)
    categories = list(CATEGORIES_DEFINITIONS.keys())
    
    logging.info("카테고리 임베딩 계산")
    category_embeddings = model.encode(
        categories, 
        convert_to_tensor=True, 
        normalize_embeddings=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    logging.info("모델 및 카테고리 임베딩 준비 완료.")


@app.post("/predict", summary="텍스트 카테고리 예측")
def predict_category(request: TextRequest):
    """입력된 텍스트의 카테고리 예측"""
    if model is None or category_embeddings is None:
        raise HTTPException(status_code=503, detail="모델 미준비 상태. 잠시 후 재시도 요망.")
        
    try:
        # 1. 입력 텍스트 NER 일반화
        generalized_text = ner_generalize_texts([request.text])[0]
        logging.info(f"원본: '{request.text}' -> 일반화: '{generalized_text}'")

        # 2. 텍스트 임베딩
        text_embedding = model.encode(
            generalized_text,
            convert_to_tensor=True,
            normalize_embeddings=True
        )

        # 3. 코사인 유사도 계산
        similarities = util.cos_sim(text_embedding, category_embeddings)
        best_match_idx = similarities.argmax()
        predicted_category = categories[best_match_idx]
        score = similarities[0][best_match_idx].item()

        return {
            "original_text": request.text,
            "generalized_text": generalized_text,
            "predicted_category": predicted_category,
            "score": score
        }
    except Exception as e:
        logging.error(f"예측 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="내부 서버 오류")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)