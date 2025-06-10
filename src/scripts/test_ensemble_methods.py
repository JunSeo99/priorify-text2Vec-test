import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(PROJECT_ROOT)

from src.config import BASE_MODEL_NAME, CATEGORIES_DEFINITIONS, NER_ENTITY_PLACEHOLDERS
from src.core.model_utils import ner_generalize_texts
from tqdm import tqdm

def test_ensemble_methods(data_path="data/data_augmented.csv"):
    """다양한 앙상블 방법들의 성능 측정"""
    
    print("데이터 로딩 중...")
    df = pd.read_csv(data_path)
    
    # 데이터 전처리
    if 'categories' not in df.columns and 'category' in df.columns:
        df = df.rename(columns={'category': 'categories'})
    
    df.dropna(subset=['title', 'categories'], inplace=True)
    df['category'] = df['categories'].apply(lambda x: x.split(';')[0].strip() if isinstance(x, str) else (x[0] if isinstance(x, list) and x else None))
    df.dropna(subset=['category'], inplace=True)
    
    categories = list(CATEGORIES_DEFINITIONS.keys())
    df = df[df['category'].isin(categories)]
    
    # 전체 데이터 사용 (샘플링 제거)
    # if len(df) > 500:
    #     df = df.sample(n=500, random_state=42)
    
    print(f"테스트 데이터: {len(df)}개")
    
    # 모델 로드
    print("모델 로딩 중...")
    model = SentenceTransformer(BASE_MODEL_NAME)
    
    # 카테고리 임베딩
    cat_embs = model.encode(categories, convert_to_tensor=True, normalize_embeddings=True)
    
    # 정답 인덱스
    true_indices = [categories.index(cat) for cat in df['category'].tolist()]
    
    # 다양한 접근 방식의 유사도 행렬 계산
    print("각 방법별 유사도 계산 중...")
    
    # 1. 원본
    original_embs = model.encode(df['title'].tolist(), convert_to_tensor=True, normalize_embeddings=True)
    original_similarities = util.cos_sim(original_embs, cat_embs)
    
    # 2. NER 일반화 (의미적)
    generalized_titles = ner_generalize_texts(df['title'].tolist())
    ner_embs = model.encode(generalized_titles, convert_to_tensor=True, normalize_embeddings=True)
    ner_similarities = util.cos_sim(ner_embs, cat_embs)
    
    # 3. 키워드 기반 (카테고리 키워드와의 유사도)
    keyword_similarities = calculate_keyword_similarities(df['title'].tolist(), categories, model)
    
    # 앙상블 방법들 테스트
    ensemble_methods = {
        "원본만": original_similarities,
        "NER만": ner_similarities,
        "키워드만": keyword_similarities,
        "원본+NER (평균)": (original_similarities + ner_similarities) / 2,
        "원본+키워드 (평균)": (original_similarities + keyword_similarities) / 2,
        "NER+키워드 (평균)": (ner_similarities + keyword_similarities) / 2,
        "전체 평균": (original_similarities + ner_similarities + keyword_similarities) / 3,
        "가중평균 (원본0.5+NER0.3+키워드0.2)": 0.5*original_similarities + 0.3*ner_similarities + 0.2*keyword_similarities,
        "가중평균 (원본0.4+NER0.4+키워드0.2)": 0.4*original_similarities + 0.4*ner_similarities + 0.2*keyword_similarities,
        "최대값": torch.max(torch.stack([original_similarities, ner_similarities, keyword_similarities]), dim=0)[0],
    }
    
    results = {}
    
    # 각 앙상블 방법 평가
    for method_name, similarities in ensemble_methods.items():
        preds = torch.topk(similarities, k=1, dim=1).indices.cpu().numpy()
        accuracy = sum(1 for i, true_idx in enumerate(true_indices) if true_idx == preds[i, 0]) / len(true_indices)
        results[method_name] = accuracy
    
    # 결과 출력
    print("\n" + "="*60)
    print("             앙상블 방법별 성능 비교")
    print("="*60)
    baseline = results["원본만"]
    
    for method_name, accuracy in results.items():
        print(f"{method_name:<25}: {accuracy:.2%}")
        if method_name != "원본만":
            change = accuracy - baseline
            print(f"{'':25}  (원본 대비 {change:+.2%})")
    print("="*60)
    
    # 최고 성능 방법 찾기
    best_method = max(results.keys(), key=lambda x: results[x])
    print(f"\n✅ 최고 성능 앙상블: {best_method} ({results[best_method]:.2%})")
    
    # Top-3, Top-5 성능도 확인
    print(f"\n=== {best_method} 방법의 Top-K 성능 ===")
    best_similarities = ensemble_methods[best_method]
    
    for k in [1, 3, 5]:
        topk_preds = torch.topk(best_similarities, k=k, dim=1).indices.cpu().numpy()
        hits = sum(1 for i, true_idx in enumerate(true_indices) 
                  if true_idx in topk_preds[i]) / len(true_indices)
        print(f"Hit Rate @{k}: {hits:.2%}")

def calculate_keyword_similarities(texts, categories, model):
    """카테고리 키워드와의 유사도 계산"""
    from src.config import CATEGORIES_DEFINITIONS
    
    # 각 카테고리의 키워드들을 하나의 문자열로 결합
    category_keywords = []
    for cat in categories:
        if cat in CATEGORIES_DEFINITIONS:
            keywords = " ".join(CATEGORIES_DEFINITIONS[cat][:5])  # 상위 5개 키워드만 사용
            category_keywords.append(keywords)
        else:
            category_keywords.append(cat)  # 키워드가 없으면 카테고리명 사용
    
    # 키워드 임베딩
    keyword_embs = model.encode(category_keywords, convert_to_tensor=True, normalize_embeddings=True)
    
    # 텍스트 임베딩
    text_embs = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    
    # 유사도 계산
    similarities = util.cos_sim(text_embs, keyword_embs)
    
    return similarities

if __name__ == "__main__":
    test_ensemble_methods() 