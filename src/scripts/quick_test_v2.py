import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
import os
import sys
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(PROJECT_ROOT)

from src.config import BASE_MODEL_NAME, CATEGORIES_DEFINITIONS
from src.core.model_utils import ner_generalize_texts

def quick_compare_v1_v2(data_path):
    """V1과 V2 성능을 빠르게 비교"""
    
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
    
    # 작은 샘플로 테스트 (500개)
    if len(df) > 500:
        df = df.sample(n=500, random_state=42)
    
    print(f"테스트 데이터: {len(df)}개")
    
    # 모델 로드
    print("모델 로딩 중...")
    model = SentenceTransformer(BASE_MODEL_NAME)
    
    # 카테고리 임베딩
    cat_embs = model.encode(categories, convert_to_tensor=True, normalize_embeddings=True)
    
    # V1: 원본 텍스트
    print("V1 평가 중...")
    v1_embs = model.encode(df['title'].tolist(), convert_to_tensor=True, normalize_embeddings=True)
    v1_similarities = util.cos_sim(v1_embs, cat_embs)
    v1_preds = torch.topk(v1_similarities, k=1, dim=1).indices.cpu().numpy()
    
    # V2: NER 일반화된 텍스트  
    print("NER 일반화 중...")
    generalized_titles = ner_generalize_texts(df['title'].tolist())
    
    print("V2 평가 중...")
    v2_embs = model.encode(generalized_titles, convert_to_tensor=True, normalize_embeddings=True)
    v2_similarities = util.cos_sim(v2_embs, cat_embs)
    v2_preds = torch.topk(v2_similarities, k=1, dim=1).indices.cpu().numpy()
    
    # 정답 인덱스
    true_indices = [categories.index(cat) for cat in df['category'].tolist()]
    
    # 정확도 계산
    v1_accuracy = sum(1 for i, true_idx in enumerate(true_indices) if true_idx == v1_preds[i, 0]) / len(true_indices)
    v2_accuracy = sum(1 for i, true_idx in enumerate(true_indices) if true_idx == v2_preds[i, 0]) / len(true_indices)
    
    print("\n" + "="*50)
    print("        V1 vs V2 성능 비교 결과")
    print("="*50)
    print(f"V1 (원본):        {v1_accuracy:.2%}")
    print(f"V2 (NER 적용):    {v2_accuracy:.2%}")
    print(f"성능 변화:        {v2_accuracy - v1_accuracy:+.2%}")
    print("="*50)
    
    # 몇 가지 예시 보여주기
    print("\n=== NER 일반화 예시 ===")
    for i in range(min(10, len(df))):
        original = df.iloc[i]['title']
        generalized = generalized_titles[i]
        if original != generalized:
            print(f"원본: {original}")
            print(f"일반화: {generalized}")
            print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/data_augmented.csv")
    args = parser.parse_args()
    
    quick_compare_v1_v2(args.data_path) 