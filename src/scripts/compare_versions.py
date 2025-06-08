import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging
import os
import argparse
import torch
from sklearn.metrics import precision_recall_fscore_support
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(PROJECT_ROOT)
from src.config import (
    BASE_MODEL_NAME, DATA_PATH, DEFAULT_OUTPUT_PATH, CATEGORIES_DEFINITIONS
)
from src.core.model_utils import ner_generalize_texts

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AlgorithmEvaluator:
    """
    다양한 버전의 카테고리 분류 알고리즘 성능 비교 평가 클래스
    """
    def __init__(self, data_path, finetuned_model_path):
        self.data_path = data_path
        self.finetuned_model_path = finetuned_model_path
        
        self.base_model = SentenceTransformer(BASE_MODEL_NAME)
        self.finetuned_model = self._load_finetuned_model()
        
        self.categories = list(CATEGORIES_DEFINITIONS.keys())
        self.test_df = self._load_and_prepare_data()
        
        # 공통적으로 사용할 일반화된 제목 미리 계산
        self.test_df['generalized_title'] = ner_generalize_texts(self.test_df['title'].tolist())
        
        # 카테고리 임베딩 미리 계산
        self.simple_cat_embs = self.base_model.encode(self.categories, convert_to_tensor=True, normalize_embeddings=True)
        self.v3_cat_embs = self._get_keyword_avg_embs(self.base_model)
        self.v4_cat_embs = self._get_keyword_avg_embs(self.finetuned_model) if self.finetuned_model else None

    def _load_finetuned_model(self):
        """파인튜닝된 모델 로드. 없으면 None 반환."""
        if os.path.exists(self.finetuned_model_path):
            logging.info(f"파인튜닝 모델 로드: {self.finetuned_model_path}")
            return SentenceTransformer(self.finetuned_model_path)
        else:
            logging.warning(f"파인튜닝된 모델을 찾을 수 없음: {self.finetuned_model_path}")
            logging.warning("V4 평가는 건너뜁니다. 'finetune.py'를 실행하여 모델을 먼저 생성하세요.")
            return None
            
    def _load_and_prepare_data(self):
        """데이터 로드 및 테스트셋 준비."""
        df = pd.read_csv(self.data_path)
        # 'categories' 컬럼이 없는 경우 'category'로 대체 시도
        if 'categories' not in df.columns and 'category' in df.columns:
            df = df.rename(columns={'category': 'categories'})
            
        df.dropna(subset=['title', 'categories'], inplace=True)
        # 여러 카테고리 중 첫 번째 카테고리를 정답으로 사용
        df['category'] = df['categories'].apply(lambda x: x.split(';')[0].strip() if isinstance(x, str) else (x[0] if isinstance(x, list) and x else None))
        df.dropna(subset=['category'], inplace=True)

        # config에 정의된 카테고리만 필터링
        original_len = len(df)
        df = df[df['category'].isin(self.categories)]
        new_len = len(df)
        if original_len > new_len:
            logging.warning(f"{original_len - new_len}개의 데이터가 config에 정의되지 않은 카테고리를 가지고 있어 제외되었습니다.")
            
        logging.info(f"테스트 데이터 로드 완료: 총 {len(df)}개")
        return df.copy()
        
    def _get_keyword_avg_embs(self, model):
        """주어진 모델로 카테고리별 키워드 평균 임베딩 계산."""
        category_embs = {}
        for category, keywords in CATEGORIES_DEFINITIONS.items():
            keyword_embs = model.encode(keywords, convert_to_numpy=True, normalize_embeddings=True)
            avg_emb = np.mean(keyword_embs, axis=0)
            if np.linalg.norm(avg_emb) > 0:
                avg_emb = avg_emb / np.linalg.norm(avg_emb)
            category_embs[category] = avg_emb
        return torch.tensor(np.array(list(category_embs.values()))).to(model.device)

    def _evaluate_version(self, version_name, model, title_column, category_embs):
        """단일 버전 성능 평가 로직. 다양한 지표를 계산하여 반환."""
        logging.info(f"--- {version_name} 평가 시작 ---")
        title_embs = model.encode(self.test_df[title_column].tolist(), convert_to_tensor=True, normalize_embeddings=True)
        
        similarities = util.cos_sim(title_embs, category_embs)
        
        # 상위 K개 예측 추출 (K=5)
        top_k_preds = torch.topk(similarities, k=5, dim=1)
        pred_indices = top_k_preds.indices.cpu().numpy()
        
        true_categories = self.test_df['category'].tolist()
        # 정답 카테고리 이름 -> 인덱스로 변환
        true_indices = [self.categories.index(cat) for cat in true_categories]
        
        # Hit Rate @1, @3, @5 계산
        correct_at_1 = 0
        correct_at_3 = 0
        correct_at_5 = 0
        
        for i, true_idx in enumerate(true_indices):
            # @1: 첫 번째 예측이 정답인 경우
            if true_idx == pred_indices[i, 0]:
                correct_at_1 += 1
            # @3: 상위 3개 예측에 정답이 포함된 경우
            if true_idx in pred_indices[i, :3]:
                correct_at_3 += 1
            # @5: 상위 5개 예측에 정답이 포함된 경우
            if true_idx in pred_indices[i, :5]:
                correct_at_5 += 1

        total_count = len(self.test_df)
        hit_rate_1 = correct_at_1 / total_count
        hit_rate_3 = correct_at_3 / total_count
        hit_rate_5 = correct_at_5 / total_count

        # F1-score (macro) 계산
        top_1_predictions = [self.categories[idx] for idx in pred_indices[:, 0]]
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_categories, 
            top_1_predictions, 
            average='macro', 
            zero_division=0,
            labels=self.categories # 모든 카테고리를 고려하도록 명시
        )
        
        metrics = {
            "Hit Rate @1": hit_rate_1,
            "Hit Rate @3": hit_rate_3,
            "Hit Rate @5": hit_rate_5,
            "F1 (macro)": f1,
        }
        
        logging.info(f"{version_name} 평가 완료: Hit@1={hit_rate_1:.2%}, Hit@3={hit_rate_3:.2%}, Hit@5={hit_rate_5:.2%}, F1={f1:.4f}")
        return metrics

    def run_all_and_compare(self):
        """모든 알고리즘 버전을 실행하고 결과를 비교하여 출력."""
        results = {}
        
        # V1: NER 없음 + 단순 카테고리명
        results['V1 (단순 유사도)'] = self._evaluate_version(
            "V1", self.base_model, 'title', self.simple_cat_embs
        )
        
        # V2: NER 적용 + 단순 카테고리명
        results['V2 (NER 적용)'] = self._evaluate_version(
            "V2", self.base_model, 'generalized_title', self.simple_cat_embs
        )
        
        # V3: NER 적용 + 키워드 평균 임베딩
        results['V3 (NER+키워드 평균)'] = self._evaluate_version(
            "V3", self.base_model, 'generalized_title', self.v3_cat_embs
        )
        
        # V4: NER 적용 + 키워드 평균 임베딩 + 파인튜닝 모델
        if self.finetuned_model and self.v4_cat_embs is not None:
            results['V4 (NER+키워드+파인튜닝)'] = self._evaluate_version(
                "V4", self.finetuned_model, 'generalized_title', self.v4_cat_embs
            )
        else:
            metrics_keys = ["Hit Rate @1", "Hit Rate @3", "Hit Rate @5", "F1 (macro)"]
            results['V4 (NER+키워드+파인튜닝)'] = {key: "N/A" for key in metrics_keys}

        # 최종 결과 출력
        print("\n" + "="*80)
        print(" " * 25 + "알고리즘 버전별 성능 비교 결과")
        print("="*80)
        
        headers = ["Algorithm Version", "Hit Rate @1", "Hit Rate @3", "Hit Rate @5", "F1 (macro)"]
        col_widths = [30, 15, 15, 15, 12]
        header_line = " | ".join([f"{h:<{w}}" for h, w in zip(headers, col_widths)])
        print(header_line)
        print("-" * len(header_line))

        for version, scores in results.items():
            row_data = [version]
            for header in headers[1:]:
                score = scores.get(header)
                if isinstance(score, float):
                    if "Hit Rate" in header:
                        display_score = f"{score:.2%}"
                    else:
                        display_score = f"{score:.4f}"
                else:
                    display_score = str(score)
                row_data.append(display_score)
            
            row_line = " | ".join([f"{d:<{w}}" for d, w in zip(row_data, col_widths)])
            print(row_line)
        print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="다양한 버전의 카테고리 분류 알고리즘 성능을 비교 평가합니다.")
    parser.add_argument("--data_path", type=str, default=DATA_PATH, help="평가에 사용할 데이터 CSV 파일 경로")
    parser.add_argument("--finetuned_model_path", type=str, default=DEFAULT_OUTPUT_PATH, help="V4 평가에 사용할 파인튜닝된 모델의 경로")
    args = parser.parse_args()
    if not os.path.isabs(args.finetuned_model_path):
        args.finetuned_model_path = os.path.join(PROJECT_ROOT, args.finetuned_model_path)
    if not os.path.isabs(args.data_path):
        args.data_path = os.path.join(PROJECT_ROOT, args.data_path)

    evaluator = AlgorithmEvaluator(data_path=args.data_path, finetuned_model_path=args.finetuned_model_path)
    evaluator.run_all_and_compare()
