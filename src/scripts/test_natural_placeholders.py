import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(PROJECT_ROOT)

from src.config import BASE_MODEL_NAME, CATEGORIES_DEFINITIONS
from src.core.model_utils import get_ner_pipeline
from src.core.performance_utils import optimize_sentence_transformer, batch_encode_optimized, get_optimal_device
from tqdm import tqdm

def test_natural_placeholders(data_path="data/data_augmented.csv"):
    """더 자연스러운 플레이스홀더들의 성능 측정 (MPS 최적화)"""
    
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
    
    # MPS 최적화된 모델 로드
    print("모델 로딩 중... (MPS 최적화)")
    model, device = optimize_sentence_transformer(BASE_MODEL_NAME)
    print(f"✅ 사용 중인 디바이스: {device}")
    
    # 카테고리 임베딩 (최적화된 배치 처리)
    cat_embs = batch_encode_optimized(model, categories, device)
    
    # 정답 인덱스
    true_indices = [categories.index(cat) for cat in df['category'].tolist()]
    
    # 다양한 플레이스홀더 방법들
    placeholder_methods = {
        "원본 (일반화 없음)": {},
        "기존 (사람, 장소, 날짜, 시간)": {
            "PS": "사람", "LC": "장소", "DT": "날짜", "TI": "시간"
        },
        "자연스러운 v1 (누군가, 어딘가, 언젠가)": {
            "PS": "누군가", "LC": "어딘가", "DT": "언젠가", "TI": "언제"
        },
        "자연스러운 v2 (친구, 곳, 때)": {
            "PS": "친구", "LC": "곳", "DT": "때", "TI": "때"
        },
        "간단한 (타인, 곳, 때)": {
            "PS": "타인", "LC": "곳", "DT": "때", "TI": "때"
        },
        "인물만 일반화 (누군가)": {
            "PS": "누군가"
        },
        "인물만 일반화 (친구)": {
            "PS": "친구"
        },
        "시간만 일반화": {
            "DT": "언젠가", "TI": "언제"
        }
    }
    
    results = {}
    
    # 각 방법별로 테스트
    for method_name, placeholders in placeholder_methods.items():
        print(f"{method_name} 평가 중...")
        
        if placeholders:  # 일반화할 경우
            generalized_titles = ner_generalize_with_placeholders(df['title'].tolist(), placeholders)
        else:  # 원본
            generalized_titles = df['title'].tolist()
        
        # 임베딩 및 예측 (최적화된 배치 처리)
        embs = batch_encode_optimized(model, generalized_titles, device)
        similarities = util.cos_sim(embs, cat_embs)
        preds = torch.topk(similarities, k=1, dim=1).indices.cpu().numpy()
        accuracy = sum(1 for i, true_idx in enumerate(true_indices) if true_idx == preds[i, 0]) / len(true_indices)
        
        results[method_name] = accuracy
    
    # 결과 출력
    print("\n" + "="*70)
    print("              자연스러운 플레이스홀더 성능 비교")
    print("="*70)
    baseline = results["원본 (일반화 없음)"]
    
    for method_name, accuracy in results.items():
        print(f"{method_name:<30}: {accuracy:.2%}")
        if method_name != "원본 (일반화 없음)":
            change = accuracy - baseline
            print(f"{'':30}  (원본 대비 {change:+.2%})")
    print("="*70)
    
    # 최고 성능 방법 찾기
    best_method = max([k for k in results.keys() if k != "원본 (일반화 없음)"], key=lambda x: results[x])
    print(f"\n✅ 최고 성능 방법: {best_method} ({results[best_method]:.2%})")
    
    # 변환 예시 보여주기
    if best_method != "원본 (일반화 없음)":
        print(f"\n=== {best_method} 방법의 변환 예시 ===")
        best_placeholders = placeholder_methods[best_method]
        sample_generalized = ner_generalize_with_placeholders(df['title'].head(10).tolist(), best_placeholders)
        
        for i, (original, generalized) in enumerate(zip(df['title'].head(10), sample_generalized)):
            if original != generalized:
                print(f"원본: {original}")
                print(f"변환: {generalized}")
                print()

def ner_generalize_with_placeholders(texts, placeholders):
    """특정 플레이스홀더로 NER 일반화"""
    if not placeholders:
        return texts
        
    ner_pipeline = get_ner_pipeline()
    if not ner_pipeline:
        return texts

    generalized_texts = []
    entities_to_use = list(placeholders.keys())
    
    for text in tqdm(texts, desc=f"일반화 진행 중 ({list(placeholders.values())})"):
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
    test_natural_placeholders() 