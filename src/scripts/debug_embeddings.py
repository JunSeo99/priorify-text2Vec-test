import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import logging
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(PROJECT_ROOT)

from src.config import BASE_MODEL_NAME, DATA_PATH, CATEGORIES_DEFINITIONS
from src.core.model_utils import ner_generalize_texts

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EmbeddingDebugger:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = SentenceTransformer(BASE_MODEL_NAME)
        self.categories = list(CATEGORIES_DEFINITIONS.keys())
        
        # 데이터 로드
        df = pd.read_csv(data_path)
        df.dropna(subset=['title', 'categories'], inplace=True)
        df['category'] = df['categories'].apply(lambda x: x.split(';')[0].strip() if isinstance(x, str) else (x[0] if isinstance(x, list) and x else None))
        df = df[df['category'].isin(self.categories)]
        
        # NER 적용
        df['generalized_title'] = ner_generalize_texts(df['title'].tolist())
        
        self.test_df = df.copy()
        logging.info(f"테스트 데이터: {len(self.test_df)}개")
        
    def _evaluate_single_embedding(self, name, category_embs):
        """단일 임베딩 방식 평가"""
        text_embs = self.model.encode(
            self.test_df['generalized_title'].tolist(), 
            convert_to_tensor=True, 
            normalize_embeddings=True
        )
        
        similarities = util.cos_sim(text_embs, category_embs)
        top_k_preds = torch.topk(similarities, k=1, dim=1)
        pred_indices = top_k_preds.indices.cpu().numpy()
        
        true_categories = self.test_df['category'].tolist()
        true_indices = [self.categories.index(cat) for cat in true_categories]
        
        correct_at_1 = sum(1 for i, true_idx in enumerate(true_indices) if true_idx == pred_indices[i, 0])
        hit_rate_1 = correct_at_1 / len(self.test_df)
        
        print(f"{name}: Hit@1 = {hit_rate_1:.4f} ({hit_rate_1:.2%})")
        return hit_rate_1
        
    def debug_all_embeddings(self):
        """모든 임베딩 방식 개별 성능 확인"""
        print("="*60)
        print("임베딩 방식별 개별 성능 분석")
        print("="*60)
        
        # 1. 단순 카테고리명 임베딩
        simple_embs = self.model.encode(self.categories, convert_to_tensor=True, normalize_embeddings=True)
        simple_score = self._evaluate_single_embedding("1. 단순 카테고리명", simple_embs)
        
        # 2. 키워드 평균 임베딩
        keyword_avg_embs = self._get_keyword_avg_embs()
        avg_score = self._evaluate_single_embedding("2. 키워드 평균", keyword_avg_embs)
        
        # 3. 키워드 최대값 임베딩
        keyword_max_embs = self._get_keyword_max_embs()
        max_score = self._evaluate_single_embedding("3. 키워드 최대값", keyword_max_embs)
        
        print("\n" + "="*60)
        print("앙상블 조합별 성능 분석")
        print("="*60)
        
        # V2 앙상블: 단순 + 키워드 평균
        v2_ensemble = self._ensemble_two_embeddings(simple_embs, keyword_avg_embs)
        v2_score = self._evaluate_single_embedding("V2 (단순 + 키워드평균)", v2_ensemble)
        
        # V3 앙상블: 키워드 평균 + 키워드 최대값
        v3_ensemble = self._ensemble_two_embeddings(keyword_avg_embs, keyword_max_embs)
        v3_score = self._evaluate_single_embedding("V3 (키워드평균 + 키워드최대)", v3_ensemble)
        
        print("\n" + "="*60)
        print("결론 및 분석")
        print("="*60)
        print(f"키워드 평균이 단순 카테고리명보다 {'좋음' if avg_score > simple_score else '나쁨'}: {avg_score:.4f} vs {simple_score:.4f}")
        print(f"키워드 최대값이 키워드 평균보다 {'좋음' if max_score > avg_score else '나쁨'}: {max_score:.4f} vs {avg_score:.4f}")
        print(f"V3가 V2보다 {'좋음' if v3_score > v2_score else '나쁨'}: {v3_score:.4f} vs {v2_score:.4f}")
        
        # 키워드 수 분석
        self._analyze_keyword_counts()
        
    def _get_keyword_avg_embs(self):
        """키워드 평균 임베딩 계산"""
        category_embs = []
        for category in self.categories:
            keywords = CATEGORIES_DEFINITIONS[category]
            if not keywords:
                category_embs.append(np.zeros(self.model.get_sentence_embedding_dimension()))
                continue
                
            keyword_embs = self.model.encode(keywords, convert_to_numpy=True, normalize_embeddings=True)
            avg_emb = np.mean(keyword_embs, axis=0)
            if np.linalg.norm(avg_emb) > 0:
                avg_emb = avg_emb / np.linalg.norm(avg_emb)
            category_embs.append(avg_emb)
        return torch.tensor(np.array(category_embs)).to(self.model.device)
        
    def _get_keyword_max_embs(self):
        """키워드 최대값 임베딩 계산"""
        category_embs = []
        for category in self.categories:
            keywords = CATEGORIES_DEFINITIONS[category]
            if not keywords:
                category_embs.append(np.zeros(self.model.get_sentence_embedding_dimension()))
                continue
                
            keyword_embs = self.model.encode(keywords, convert_to_numpy=True, normalize_embeddings=True)
            # 각 차원별 최대값
            max_emb = np.max(keyword_embs, axis=0)
            if np.linalg.norm(max_emb) > 0:
                max_emb = max_emb / np.linalg.norm(max_emb)
            category_embs.append(max_emb)
        return torch.tensor(np.array(category_embs)).to(self.model.device)
        
    def _ensemble_two_embeddings(self, emb1, emb2, weight=0.5):
        """두 임베딩을 0.5:0.5로 앙상블"""
        # numpy로 변환하여 계산
        emb1_np = emb1.cpu().numpy() if hasattr(emb1, 'cpu') else emb1
        emb2_np = emb2.cpu().numpy() if hasattr(emb2, 'cpu') else emb2
        
        ensemble = emb1_np * weight + emb2_np * weight
        # 정규화
        norms = np.linalg.norm(ensemble, axis=1, keepdims=True)
        ensemble = ensemble / norms
        
        return torch.tensor(ensemble).to(self.model.device)
        
    def _analyze_keyword_counts(self):
        """카테고리별 키워드 수 분석"""
        print("\n" + "="*60)
        print("카테고리별 키워드 수 분석")
        print("="*60)
        
        for category in self.categories:
            keywords = CATEGORIES_DEFINITIONS[category]
            print(f"{category}: {len(keywords)}개 키워드")
            if len(keywords) <= 3:
                print(f"  -> 키워드: {keywords}")
        
        # 키워드 수가 적은 카테고리들의 성능 확인
        few_keyword_categories = [cat for cat in self.categories if len(CATEGORIES_DEFINITIONS[cat]) <= 5]
        if few_keyword_categories:
            print(f"\n키워드 수가 적은 카테고리 ({len(few_keyword_categories)}개): {few_keyword_categories}")

if __name__ == "__main__":
    import torch
    debugger = EmbeddingDebugger(os.path.join(PROJECT_ROOT, "data/data.csv"))
    debugger.debug_all_embeddings() 