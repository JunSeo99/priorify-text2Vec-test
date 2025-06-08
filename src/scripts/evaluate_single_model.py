import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import numpy as np
import argparse
import logging
import sys
import os
import torch
from sklearn.metrics import precision_recall_fscore_support

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(PROJECT_ROOT)

from src.config import CATEGORIES_DEFINITIONS, DATA_PATH, DEFAULT_OUTPUT_PATH
from src.core.model_utils import ner_generalize_texts

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelEvaluator:
    def __init__(self, model_path: str):
        logging.info(f"'{model_path}' 모델 로드 중...")
        self.model = SentenceTransformer(model_path)
        self.categories = list(CATEGORIES_DEFINITIONS.keys())
        self.category_embeddings = self._get_keyword_avg_embs(self.model)
        logging.info("모델 로드 및 키워드 기반 카테고리 임베딩 완료.")

    def _get_keyword_avg_embs(self, model):
        """카테고리별 키워드 평균 임베딩 계산."""
        category_embs = {}
        for category, keywords in CATEGORIES_DEFINITIONS.items():
            keyword_embs = model.encode(keywords, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            avg_emb = np.mean(keyword_embs, axis=0)
            if np.linalg.norm(avg_emb) > 0:
                avg_emb = avg_emb / np.linalg.norm(avg_emb)
            category_embs[category] = avg_emb
        return torch.tensor(np.array(list(category_embs.values()))).to(self.model.device)

    def evaluate(self, df: pd.DataFrame):
        """데이터프레임에 대해 모델 성능을 평가하고 결과 딕셔너리를 반환합니다."""
        logging.info("평가 시작...")
        
        title_embs = self.model.encode(df['generalized_title'].tolist(), convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True)
        similarities = util.cos_sim(title_embs, self.category_embeddings)
        
        top_k_preds = torch.topk(similarities, k=5, dim=1)
        pred_indices = top_k_preds.indices.cpu().numpy()
        
        true_categories = df['category'].tolist()
        true_indices = [self.categories.index(cat) for cat in true_categories if cat in self.categories]
        
        correct_at_1 = sum(1 for i, true_idx in enumerate(true_indices) if true_idx == pred_indices[i, 0])
        correct_at_3 = sum(1 for i, true_idx in enumerate(true_indices) if true_idx in pred_indices[i, :3])
        correct_at_5 = sum(1 for i, true_idx in enumerate(true_indices) if true_idx in pred_indices[i, :5])

        total_count = len(df)
        hit_rate_1 = correct_at_1 / total_count
        hit_rate_3 = correct_at_3 / total_count
        hit_rate_5 = correct_at_5 / total_count

        top_1_predictions = [self.categories[idx] for idx in pred_indices[:, 0]]
        _, _, f1, _ = precision_recall_fscore_support(
            true_categories, top_1_predictions, average='macro', zero_division=0, labels=self.categories
        )
        
        return {
            "Hit Rate @1": hit_rate_1,
            "Hit Rate @3": hit_rate_3,
            "Hit Rate @5": hit_rate_5,
            "F1 (macro)": f1,
        }

def run_evaluation(model_path: str, data_path: str):
    df = pd.read_csv(data_path)
    if 'categories' not in df.columns and 'category' in df.columns:
        df = df.rename(columns={'category': 'categories'})
    df.dropna(subset=['title', 'categories'], inplace=True)
    df['category'] = df['categories'].apply(lambda x: x.split(';')[0].strip() if isinstance(x, str) else (x[0] if isinstance(x, list) and x else None))
    df.dropna(subset=['category'], inplace=True)
    
    categories_in_config = list(CATEGORIES_DEFINITIONS.keys())
    df = df[df['category'].isin(categories_in_config)]
    logging.info(f"평가 데이터 로드 완료: 총 {len(df)}개")

    df['generalized_title'] = ner_generalize_texts(df['title'].tolist())
    
    evaluator = ModelEvaluator(model_path)
    metrics = evaluator.evaluate(df)

    print("\n" + "="*50)
    print(f"        모델 성능 평가 결과")
    print(f"  Model: {os.path.basename(os.path.normpath(model_path))}")
    print("="*50)
    for name, value in metrics.items():
        display_value = f"{value:.2%}" if "Hit Rate" in name else f"{value:.4f}"
        print(f"{name:<15} | {display_value}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="단일 모델의 성능을 다양한 지표로 평가합니다.")
    parser.add_argument("--model_path", type=str, default=DEFAULT_OUTPUT_PATH, help="성능을 평가할 모델의 경로")
    parser.add_argument("--data_path", type=str, default=DATA_PATH, help="평가에 사용할 데이터 CSV 파일 경로")
    args = parser.parse_args()

    if not os.path.isabs(args.model_path):
        args.model_path = os.path.join(PROJECT_ROOT, args.model_path)
    if not os.path.isabs(args.data_path):
        args.data_path = os.path.join(PROJECT_ROOT, args.data_path)

    run_evaluation(args.model_path, args.data_path)
