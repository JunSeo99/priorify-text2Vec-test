import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(PROJECT_ROOT)

from src.config import BASE_MODEL_NAME, CATEGORIES_DEFINITIONS, NER_ENTITY_PLACEHOLDERS, NER_ENTITY_PLACEHOLDERS_SPECIAL
from src.core.model_utils import ner_generalize_texts

def test_placeholder_methods(data_path="data/data_augmented.csv"):
    """다양한 플레이스홀더 방법들의 성능을 비교"""
    
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
    
    # 작은 샘플로 테스트 (200개)
    # if len(df) > 200:
    #     df = df.sample(n=200, random_state=42)
    
    print(f"테스트 데이터: {len(df)}개")
    
    # 모델 로드
    print("모델 로딩 중...")
    model = SentenceTransformer(BASE_MODEL_NAME)
    
    # 카테고리 임베딩
    cat_embs = model.encode(categories, convert_to_tensor=True, normalize_embeddings=True)
    
    # 정답 인덱스
    true_indices = [categories.index(cat) for cat in df['category'].tolist()]
    
    results = {}
    
    # 방법1: 원본 (V1)
    print("V1 (원본) 평가 중...")
    v1_embs = model.encode(df['title'].tolist(), convert_to_tensor=True, normalize_embeddings=True)
    v1_similarities = util.cos_sim(v1_embs, cat_embs)
    v1_preds = torch.topk(v1_similarities, k=1, dim=1).indices.cpu().numpy()
    v1_accuracy = sum(1 for i, true_idx in enumerate(true_indices) if true_idx == v1_preds[i, 0]) / len(true_indices)
    results['V1 (원본)'] = v1_accuracy
    
    # 방법2: 특수 토큰 플레이스홀더 (기존 V2)
    print("V2 (특수토큰) 평가 중...")
    special_titles = ner_generalize_texts_custom(df['title'].tolist(), NER_ENTITY_PLACEHOLDERS_SPECIAL)
    v2_embs = model.encode(special_titles, convert_to_tensor=True, normalize_embeddings=True)
    v2_similarities = util.cos_sim(v2_embs, cat_embs)
    v2_preds = torch.topk(v2_similarities, k=1, dim=1).indices.cpu().numpy()
    v2_accuracy = sum(1 for i, true_idx in enumerate(true_indices) if true_idx == v2_preds[i, 0]) / len(true_indices)
    results['V2 (특수토큰)'] = v2_accuracy
    
    # 방법3: 의미적 한국어 플레이스홀더 (새로운 V2-개선)  
    print("V2-개선 (의미적) 평가 중...")
    semantic_titles = ner_generalize_texts_custom(df['title'].tolist(), NER_ENTITY_PLACEHOLDERS)
    v2_improved_embs = model.encode(semantic_titles, convert_to_tensor=True, normalize_embeddings=True)
    v2_improved_similarities = util.cos_sim(v2_improved_embs, cat_embs)
    v2_improved_preds = torch.topk(v2_improved_similarities, k=1, dim=1).indices.cpu().numpy()
    v2_improved_accuracy = sum(1 for i, true_idx in enumerate(true_indices) if true_idx == v2_improved_preds[i, 0]) / len(true_indices)
    results['V2-개선 (의미적)'] = v2_improved_accuracy
    
    # 방법4: 인물만 일반화 (보수적)
    print("V2-보수적 (인물만) 평가 중...")
    conservative_placeholders = {"PS": "사람"}
    conservative_titles = ner_generalize_texts_custom(df['title'].tolist(), conservative_placeholders, ["PS"])
    v2_conservative_embs = model.encode(conservative_titles, convert_to_tensor=True, normalize_embeddings=True)
    v2_conservative_similarities = util.cos_sim(v2_conservative_embs, cat_embs)
    v2_conservative_preds = torch.topk(v2_conservative_similarities, k=1, dim=1).indices.cpu().numpy()
    v2_conservative_accuracy = sum(1 for i, true_idx in enumerate(true_indices) if true_idx == v2_conservative_preds[i, 0]) / len(true_indices)
    results['V2-보수적 (인물만)'] = v2_conservative_accuracy
    
    # 결과 출력
    print("\n" + "="*60)
    print("           플레이스홀더 방법별 성능 비교")
    print("="*60)
    for method, accuracy in results.items():
        print(f"{method:<20}: {accuracy:.2%}")
        if method != 'V1 (원본)':
            change = accuracy - results['V1 (원본)']
            print(f"{'':20}  (V1 대비 {change:+.2%})")
    print("="*60)
    
    # 변환 예시 보여주기
    print("\n=== 변환 예시 비교 ===")
    for i in range(min(5, len(df))):
        original = df.iloc[i]['title']
        special = special_titles[i] 
        semantic = semantic_titles[i]
        conservative = conservative_titles[i]
        
        if original != special or original != semantic:
            print(f"원본:      {original}")
            if original != special:
                print(f"특수토큰:  {special}")
            if original != semantic:
                print(f"의미적:    {semantic}")
            if original != conservative:
                print(f"보수적:    {conservative}")
            print()

def ner_generalize_texts_custom(texts, placeholders, entities_to_use=None):
    """커스텀 플레이스홀더로 NER 일반화"""
    from src.core.model_utils import get_ner_pipeline
    from tqdm import tqdm
    
    if entities_to_use is None:
        entities_to_use = list(placeholders.keys())
    
    ner_pipeline = get_ner_pipeline()
    if not ner_pipeline:
        return texts

    generalized_texts = []
    
    for text in tqdm(texts, desc="NER 일반화 진행 중"):
        try:
            ner_results = ner_pipeline(text)
            
            entities_to_replace = sorted([
                res for res in ner_results
                if res['score'] > 0.8 and res['entity_group'] in entities_to_use
            ], key=lambda x: x['start'], reverse=True)

            generalized_text = text
            for entity in entities_to_replace:
                entity_type = entity['entity_group']
                
                if entity_type in placeholders:
                    placeholder = placeholders[entity_type]
                    start, end = entity['start'], entity['end']
                    generalized_text = generalized_text[:start] + placeholder + generalized_text[end:]
                
            generalized_texts.append(generalized_text)
        except Exception as e:
            generalized_texts.append(text)
        
    return generalized_texts

if __name__ == "__main__":
    test_placeholder_methods() 