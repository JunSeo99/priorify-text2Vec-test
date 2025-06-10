# app.py
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import torch
import os
import logging
import json
from typing import Dict, List
import numpy as np

from src.config import CATEGORIES_DEFINITIONS, DEFAULT_OUTPUT_PATH
from src.core.model_utils import ner_generalize_texts

# --- 로깅 및 환경 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- FastAPI 앱 설정 ---
app = FastAPI(
    title="Priorify Category Prediction API", 
    version="4.0",
    description="한국어 텍스트 카테고리 분류 API (v4 Ensemble + NER)"
)

class TextRequest(BaseModel):
    text: str
    use_ner: bool = True  # NER 사용 여부 옵션

class BatchTextRequest(BaseModel):
    texts: List[str]
    use_ner: bool = True

# 글로벌 변수
model = None
categories = []
ensemble_category_embeddings = None
simple_category_embeddings = None
device = None

def get_model_path():
    """모델 경로 결정 (우선순위: 환경변수 > fine-tuned 모델 > 기본값)"""
    # 1. 환경변수 확인
    if os.getenv("MODEL_PATH"):
        return os.getenv("MODEL_PATH")
    
    # 2. fine-tuned 모델 경로들 확인 (최신 순)
    potential_paths = [
        "models/finetuned_early_stopping",
        "models/finetuned_ensemble_v4", 
        "models/finetuned_model",
        DEFAULT_OUTPUT_PATH
    ]
    
    for path in potential_paths:
        if os.path.exists(path) and os.path.exists(os.path.join(path, "config.json")):
            logger.info(f"Fine-tuned 모델 발견: {path}")
            return path
    
    # 3. 기본 베이스 모델 사용
    logger.warning("Fine-tuned 모델을 찾을 수 없어 기본 모델을 사용합니다.")
    return "jhgan/ko-sroberta-multitask"

def create_ensemble_category_embeddings(model, categories_definitions: Dict[str, List[str]]):
    """V4 방식: 카테고리명 + 키워드들의 앙상블 임베딩 생성"""
    category_embeddings = {}
    
    logger.info("앙상블 카테고리 임베딩 생성 중...")
    for category, keywords in categories_definitions.items():
        # 카테고리명 + 키워드들 조합
        texts_to_embed = [category] + keywords
        
        # 모든 텍스트 임베딩 계산
        embeddings = model.encode(
            texts_to_embed, 
            convert_to_tensor=True, 
            normalize_embeddings=True,
            device=device
        )
        
        # 평균 임베딩 계산 (앙상블)
        avg_embedding = torch.mean(embeddings, dim=0)
        category_embeddings[category] = avg_embedding
        
        logger.debug(f"카테고리 '{category}': {len(texts_to_embed)}개 텍스트로 앙상블 임베딩 생성")
    
    return category_embeddings

@app.on_event("startup")
def load_model():
    """애플리케이션 시작 시 모델 및 카테고리 임베딩 로드"""
    global model, categories, ensemble_category_embeddings, simple_category_embeddings, device
    
    try:
        # 디바이스 설정
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"사용 디바이스: {device}")
        
        # 모델 경로 결정
        model_path = get_model_path()
        logger.info(f"모델 로딩 시작: {model_path}")
        
        # 모델 로드
        model = SentenceTransformer(model_path)
        model.to(device)
        
        # 카테고리 목록
        categories = list(CATEGORIES_DEFINITIONS.keys())
        logger.info(f"카테고리 수: {len(categories)}개")
        
        # 1. 앙상블 카테고리 임베딩 생성 (V4 방식)
        ensemble_category_embeddings = create_ensemble_category_embeddings(
            model, CATEGORIES_DEFINITIONS
        )
        
        # 2. 단순 카테고리 임베딩 생성 (백업용)
        simple_category_embeddings = model.encode(
            categories, 
            convert_to_tensor=True, 
            normalize_embeddings=True,
            device=device
        )
        
        logger.info("✅ 모델 및 임베딩 로딩 완료!")
        
        # 메모리 사용량 로깅
        if device == 'cuda':
            memory_used = torch.cuda.memory_allocated() / 1024**3
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU 메모리 사용량: {memory_used:.2f}GB / {memory_total:.2f}GB")
        
    except Exception as e:
        logger.error(f"모델 로딩 실패: {e}")
        raise RuntimeError(f"Failed to load model: {e}")

def predict_single_text(text: str, use_ner: bool = True, use_ensemble: bool = True) -> Dict:
    """단일 텍스트 예측"""
    try:
        # 1. NER 일반화 (선택적)
        if use_ner:
            generalized_text = ner_generalize_texts([text])[0]
            logger.debug(f"NER 일반화: '{text}' -> '{generalized_text}'")
        else:
            generalized_text = text
        
        # 2. 텍스트 임베딩
        text_embedding = model.encode(
            generalized_text,
            convert_to_tensor=True,
            normalize_embeddings=True,
            device=device
        )
        
        # 3. 유사도 계산 및 예측
        if use_ensemble and ensemble_category_embeddings:
            # V4 앙상블 방식
            similarities = []
            for category in categories:
                category_embedding = ensemble_category_embeddings[category]
                similarity = util.cos_sim(text_embedding, category_embedding.unsqueeze(0))
                similarities.append(similarity.item())
            
            similarities = torch.tensor(similarities)
            best_match_idx = similarities.argmax()
            predicted_category = categories[best_match_idx]
            score = similarities[best_match_idx].item()
            method = "ensemble"
            
        else:
            # 단순 방식 (백업)
            similarities = util.cos_sim(text_embedding, simple_category_embeddings)
            best_match_idx = similarities.argmax()
            predicted_category = categories[best_match_idx]
            score = similarities[0][best_match_idx].item()
            method = "simple"
        
        # 4. 상위 3개 후보도 반환
        top_k = min(3, len(categories))
        if use_ensemble and ensemble_category_embeddings:
            top_indices = similarities.argsort(descending=True)[:top_k]
            top_predictions = [
                {
                    "category": categories[idx], 
                    "score": similarities[idx].item()
                } 
                for idx in top_indices
            ]
        else:
            top_indices = similarities[0].argsort(descending=True)[:top_k]
            top_predictions = [
                {
                    "category": categories[idx], 
                    "score": similarities[0][idx].item()
                } 
                for idx in top_indices
            ]
        
        return {
            "original_text": text,
            "generalized_text": generalized_text,
            "predicted_category": predicted_category,
            "confidence_score": score,
            "top_predictions": top_predictions,
            "method": method,
            "use_ner": use_ner
        }
        
    except Exception as e:
        logger.error(f"예측 중 오류: {e}")
        raise

@app.get("/", summary="API 상태 확인")
def root():
    """API 상태 및 정보 반환"""
    return {
        "status": "running",
        "version": "4.0",
        "model_loaded": model is not None,
        "categories_count": len(categories) if categories else 0,
        "device": device,
        "features": ["NER", "Ensemble", "Batch Processing"]
    }

@app.get("/health", summary="헬스체크")
def health_check():
    """서버 상태 체크"""
    if model is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다.")
    
    return {
        "status": "healthy",
        "model_ready": True,
        "device": device
    }

@app.get("/categories", summary="지원 카테고리 목록")
def get_categories():
    """지원하는 카테고리 목록 반환"""
    return {
        "categories": categories,
        "total_count": len(categories),
        "definitions": CATEGORIES_DEFINITIONS
    }

@app.post("/predict", summary="텍스트 카테고리 예측")
def predict_category(request: TextRequest):
    """단일 텍스트의 카테고리 예측"""
    if model is None:
        raise HTTPException(status_code=503, detail="모델이 준비되지 않았습니다.")
    
    try:
        result = predict_single_text(request.text, use_ner=request.use_ner)
        return result
    except Exception as e:
        logger.error(f"예측 실패: {e}")
        raise HTTPException(status_code=500, detail=f"예측 중 오류 발생: {str(e)}")

@app.post("/predict/batch", summary="배치 텍스트 카테고리 예측")
def predict_batch(request: BatchTextRequest):
    """여러 텍스트의 카테고리 예측 (배치 처리)"""
    if model is None:
        raise HTTPException(status_code=503, detail="모델이 준비되지 않았습니다.")
    
    if len(request.texts) > 100:  # 배치 크기 제한
        raise HTTPException(status_code=400, detail="배치 크기는 100개 이하로 제한됩니다.")
    
    try:
        results = []
        for text in request.texts:
            result = predict_single_text(text, use_ner=request.use_ner)
            results.append(result)
        
        return {
            "results": results,
            "total_count": len(results)
        }
    except Exception as e:
        logger.error(f"배치 예측 실패: {e}")
        raise HTTPException(status_code=500, detail=f"배치 예측 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    # 개발용 서버 실행
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=int(os.getenv("PORT", 8000)),
        reload=False  # 프로덕션에서는 reload=False
    )