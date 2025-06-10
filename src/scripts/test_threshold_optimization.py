import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(PROJECT_ROOT)

from src.config import BASE_MODEL_NAME, CATEGORIES_DEFINITIONS, NER_ENTITY_PLACEHOLDERS
from src.core.model_utils import get_ner_pipeline
from tqdm import tqdm

def test_threshold_optimization(data_path="data/data_augmented.csv"):
    """NER 임계값을 다양하게 변경하며 성능 측정"""
    
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
    
    print(f"테스트 데이터: {len(df)}개")
    
    # 모델 로드
    print("모델 로딩 중...")
    model = SentenceTransformer(BASE_MODEL_NAME)
    
    # 카테고리 임베딩
    cat_embs = model.encode(categories, convert_to_tensor=True, normalize_embeddings=True)
    
    # 정답 인덱스
    true_indices = [categories.index(cat) for cat in df['category'].tolist()]
    
    # 테스트할 임계값들
    thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    results = {}
    
    # 원본 성능 (기준점)
    print("원본 (일반화 없음) 평가 중...")
    original_embs = model.encode(df['title'].tolist(), convert_to_tensor=True, normalize_embeddings=True)
    original_similarities = util.cos_sim(original_embs, cat_embs)
    original_preds = torch.topk(original_similarities, k=1, dim=1).indices.cpu().numpy()
    original_accuracy = sum(1 for i, true_idx in enumerate(true_indices) if true_idx == original_preds[i, 0]) / len(true_indices)
    results['원본 (임계값 없음)'] = original_accuracy
    
    # 각 임계값별로 테스트
    for threshold in thresholds:
        print(f"임계값 {threshold} 평가 중...")
        
        # NER 일반화
        generalized_titles = ner_generalize_with_threshold(df['title'].tolist(), threshold)
        
        # 임베딩 및 예측
        embs = model.encode(generalized_titles, convert_to_tensor=True, normalize_embeddings=True)
        similarities = util.cos_sim(embs, cat_embs)
        preds = torch.topk(similarities, k=1, dim=1).indices.cpu().numpy()
        accuracy = sum(1 for i, true_idx in enumerate(true_indices) if true_idx == preds[i, 0]) / len(true_indices)
        
        results[f'임계값 {threshold}'] = accuracy
    
    # 결과 출력
    print("\n" + "="*60)
    print("           NER 임계값별 성능 비교")
    print("="*60)
    for method, accuracy in results.items():
        print(f"{method:<20}: {accuracy:.2%}")
        if method != '원본 (임계값 없음)':
            change = accuracy - results['원본 (임계값 없음)']
            print(f"{'':20}  (원본 대비 {change:+.2%})")
    print("="*60)
    
    # 최고 성능 임계값 찾기
    best_threshold = max([k for k in results.keys() if '임계값' in k], key=lambda x: results[x])
    print(f"\n✅ 최고 성능 임계값: {best_threshold} ({results[best_threshold]:.2%})")

def ner_generalize_with_threshold(texts, threshold):
    """특정 임계값으로 NER 일반화"""
    ner_pipeline = get_ner_pipeline()
    if not ner_pipeline:
        return texts

    generalized_texts = []
    
    for text in tqdm(texts, desc=f"NER 일반화 진행 중 (임계값={threshold})"):
        try:
            ner_results = ner_pipeline(text)
            
            entities_to_replace = sorted([
                res for res in ner_results
                if res['score'] > threshold and res['entity_group'] in NER_ENTITY_PLACEHOLDERS
            ], key=lambda x: x['start'], reverse=True)

            generalized_text = text
            for entity in entities_to_replace:
                entity_type = entity['entity_group']
                
                if entity_type in NER_ENTITY_PLACEHOLDERS:
                    placeholder = NER_ENTITY_PLACEHOLDERS[entity_type]
                    start, end = entity['start'], entity['end']
                    generalized_text = generalized_text[:start] + placeholder + generalized_text[end:]
                
            generalized_texts.append(generalized_text)
        except Exception as e:
            generalized_texts.append(text)
        
    return generalized_texts

if __name__ == "__main__":
    test_threshold_optimization() 