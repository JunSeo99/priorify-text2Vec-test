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

def test_selective_generalization(data_path="data/data_augmented.csv"):
    """엔티티별 선택적 일반화 성능 측정"""
    
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
    
    # 테스트할 엔티티 조합들
    entity_combinations = {
        "원본": [],  # 일반화 없음
        "인물만": ["PS"],
        "시간만": ["DT", "TI"], 
        "장소만": ["LC"],
        "인물+시간": ["PS", "DT", "TI"],
        "인물+장소": ["PS", "LC"],
        "시간+장소": ["DT", "TI", "LC"],
        "전체 (기존)": ["PS", "LC", "DT", "TI", "QT", "AF"],
        "보수적": ["PS", "DT", "TI"],  # 현재 V2-improved 설정
    }
    
    results = {}
    
    # 각 조합별로 테스트
    for combo_name, entities in entity_combinations.items():
        print(f"{combo_name} 조합 평가 중...")
        
        if entities:  # 일반화할 엔티티가 있는 경우
            generalized_titles = ner_generalize_selective(df['title'].tolist(), entities)
        else:  # 원본
            generalized_titles = df['title'].tolist()
        
        # 임베딩 및 예측
        embs = model.encode(generalized_titles, convert_to_tensor=True, normalize_embeddings=True)
        similarities = util.cos_sim(embs, cat_embs)
        preds = torch.topk(similarities, k=1, dim=1).indices.cpu().numpy()
        accuracy = sum(1 for i, true_idx in enumerate(true_indices) if true_idx == preds[i, 0]) / len(true_indices)
        
        results[combo_name] = accuracy
    
    # 결과 출력
    print("\n" + "="*60)
    print("         엔티티별 선택적 일반화 성능 비교")
    print("="*60)
    baseline = results["원본"]
    
    for combo_name, accuracy in results.items():
        print(f"{combo_name:<15}: {accuracy:.2%}")
        if combo_name != "원본":
            change = accuracy - baseline
            print(f"{'':15}  (원본 대비 {change:+.2%})")
    print("="*60)
    
    # 최고 성능 조합 찾기
    best_combo = max(results.keys(), key=lambda x: results[x])
    print(f"\n✅ 최고 성능 조합: {best_combo} ({results[best_combo]:.2%})")
    
    # 개선된 조합의 변환 예시 출력
    if best_combo != "원본" and results[best_combo] > baseline:
        print(f"\n=== {best_combo} 조합의 변환 예시 ===")
        best_entities = entity_combinations[best_combo]
        sample_generalized = ner_generalize_selective(df['title'].head(5).tolist(), best_entities)
        
        for i, (original, generalized) in enumerate(zip(df['title'].head(5), sample_generalized)):
            if original != generalized:
                print(f"원본:     {original}")
                print(f"변환:     {generalized}")
                print()

def ner_generalize_selective(texts, entities_to_generalize):
    """특정 엔티티만 선택적으로 일반화"""
    if not entities_to_generalize:
        return texts
        
    ner_pipeline = get_ner_pipeline()
    if not ner_pipeline:
        return texts

    generalized_texts = []
    
    for text in tqdm(texts, desc=f"선택적 일반화 진행 중 ({entities_to_generalize})"):
        try:
            ner_results = ner_pipeline(text)
            
            entities_to_replace = sorted([
                res for res in ner_results
                if res['score'] > 0.8 and res['entity_group'] in entities_to_generalize
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
    test_selective_generalization() 