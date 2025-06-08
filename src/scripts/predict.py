import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
import logging
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(PROJECT_ROOT)

from src.config import (
    DEFAULT_OUTPUT_PATH, 
    CATEGORIES_DEFINITIONS
)
from src.core.model_utils import ner_generalize_texts

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

class Predictor:
    """
    파인튜닝된 V4 모델을 사용하여 할 일 카테고리를 실시간으로 예측하는 클래스
    """
    def __init__(self, model_path):
        self.model_path = model_path
        if not os.path.exists(self.model_path):
            print(f"오류: 파인튜닝된 모델을 찾을 수 없습니다. 경로: {self.model_path}")
            print("먼저 'src/scripts/finetune.py'를 실행하여 모델을 학습시켜주세요.")
            exit()
            
        print("V4 모델과 관련 리소스를 로딩합니다...")
        self.model = SentenceTransformer(self.model_path)
        self.categories = list(CATEGORIES_DEFINITIONS.keys())
        self.category_embs = self._get_keyword_avg_embs(self.model)
        print("로딩 완료. 예측을 시작할 수 있습니다.")

    def _get_keyword_avg_embs(self, model):
        """V4 알고리즘에 사용될 카테고리별 키워드 평균 임베딩을 계산합니다."""
        category_embs = {}
        for category, keywords in CATEGORIES_DEFINITIONS.items():
            keyword_embs = model.encode(keywords, convert_to_numpy=True, normalize_embeddings=True)
            avg_emb = np.mean(keyword_embs, axis=0)
            if np.linalg.norm(avg_emb) > 0: # 0벡터 방지
                avg_emb = avg_emb / np.linalg.norm(avg_emb)
            category_embs[category] = avg_emb
        return torch.tensor(np.array(list(category_embs.values()))).to(model.device)

    def predict(self, title):
        """입력된 할 일 제목에 대한 카테고리를 예측하고 상위 5개를 반환합니다."""
        
        # 1. NER을 이용해 입력 텍스트 일반화
        generalized_title = ner_generalize_texts([title])[0]
        print(f" > NER 일반화된 텍스트: {generalized_title}")

        # 2. 파인튜닝된 모델로 텍스트 임베딩
        title_emb = self.model.encode(generalized_title, convert_to_tensor=True, normalize_embeddings=True)
        
        # 3. 키워드 임베딩과 코사인 유사도 계산
        similarities = util.cos_sim(title_emb, self.category_embs)[0]
        
        # 4. 유사도 높은 순으로 상위 5개 예측 추출
        top_k_preds = torch.topk(similarities, k=5)
        
        results = []
        for score, idx in zip(top_k_preds.values, top_k_preds.indices):
            results.append({
                "category": self.categories[idx],
                "score": score.item()
            })
        
        return results

def main():
    """
    사용자 입력을 받아 예측을 수행하는 메인 함수
    """
    model_path = os.path.join(PROJECT_ROOT, DEFAULT_OUTPUT_PATH)
    predictor = Predictor(model_path=model_path)
    
    print("\n" + "="*50)
    print("      V4 모델을 사용한 할 일 카테고리 예측")
    print("="*50)
    print("할 일 내용을 입력해주세요 (종료하려면 'exit' 또는 'quit' 입력)")

    while True:
        try:
            user_input = input("\n할 일 입력: ")
            if user_input.lower() in ['exit', 'quit']:
                print("프로그램을 종료합니다.")
                break
            
            if not user_input.strip():
                continue

            predictions = predictor.predict(user_input)
            
            print("\n--- 예측 결과 (Top 5) ---")
            for i, pred in enumerate(predictions):
                print(f"{i+1}. {pred['category']:<15} (유사도: {pred['score']:.4f})")
            print("-" * 35)

        except KeyboardInterrupt:
            print("\n프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"오류가 발생했습니다: {e}")


if __name__ == "__main__":
    main()
