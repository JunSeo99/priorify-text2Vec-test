import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation, util
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import random
import torch
import os
from tqdm import tqdm
import logging
import argparse
import math
from sentence_transformers.evaluation import SentenceEvaluator
import csv
import sys

# 프로젝트 루트를 경로에 추가
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(PROJECT_ROOT)

from src.config import (
    BASE_MODEL_NAME, NER_MODEL_NAME, DATA_PATH, DEFAULT_OUTPUT_PATH,
    CATEGORIES_DEFINITIONS, NER_SPECIAL_TOKENS
)
from src.core.model_utils import ner_generalize_texts




# 로깅 설정 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 재현성을 위한 시드 고정 
def set_seed(seed):
    """모든 난수 생성기의 시드를 고정하여 재현성을 보장합니다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class CategoryAccuracyEvaluator(SentenceEvaluator):
    """
    주어진 데이터셋에 대해 카테고리 분류 정확도를 평가하는 클래스.
    - Hit Rate @k
    - Mean Cosine Similarity for correct category
    """
    def __init__(self, test_df: pd.DataFrame, categories_definitions: dict, name: str = '', batch_size: int = 32, show_progress_bar: bool = False):
        self.test_df = test_df
        self.name = name
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar

        self.category_names = list(categories_definitions.keys())
        self.categories_definitions = categories_definitions

        self.csv_file = "accuracy_evaluation_results.csv"
        self.csv_headers = ["epoch", "steps", "hit_rate@1", "hit_rate@3", "mean_cosine_similarity"]

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:

        # 카테고리 평균 임베딩 계산
        category_embs = {}
        for name, keywords in self.categories_definitions.items():
            # 키워드가 없는 경우 0벡터로 처리
            if not keywords:
                category_embs[name] = np.zeros(model.get_sentence_embedding_dimension())
                continue

            keyword_embs = model.encode(keywords, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False, batch_size=self.batch_size)
            avg_emb = np.mean(keyword_embs, axis=0)
            if np.linalg.norm(avg_emb) > 0:
                avg_emb = avg_emb / np.linalg.norm(avg_emb)
            category_embs[name] = avg_emb

        category_embs_matrix = np.array([category_embs[name] for name in self.category_names])

        # 테스트 데이터의 제목 임베딩
        test_titles_embs = model.encode(self.test_df['generalized_title'].tolist(), convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=self.show_progress_bar, batch_size=self.batch_size)

        # 유사도 계산
        similarities = util.cos_sim(test_titles_embs, category_embs_matrix)

        hit_rate_at_1 = 0
        hit_rate_at_3 = 0
        correct_category_similarities = []

        for i in range(len(self.test_df)):
            true_categories = set(self.test_df.iloc[i]['categories'])
            if not true_categories: continue

            sim_scores_for_title = similarities[i].cpu().numpy()
            top_k_indices = np.argsort(sim_scores_for_title)[::-1]

            # Hit Rate @1
            predicted_at_1 = self.category_names[top_k_indices[0]]
            if predicted_at_1 in true_categories:
                hit_rate_at_1 += 1

            # Hit Rate @3
            predicted_at_3 = {self.category_names[j] for j in top_k_indices[:3]}
            if not true_categories.isdisjoint(predicted_at_3):
                hit_rate_at_3 += 1

            # 정답 카테고리에 대한 평균 유사도
            correct_indices = [self.category_names.index(cat) for cat in true_categories if cat in self.category_names]
            if correct_indices:
                mean_sim = np.mean([sim_scores_for_title[j] for j in correct_indices])
                correct_category_similarities.append(mean_sim)

        total_count = len(self.test_df)
        final_hit_rate_1 = hit_rate_at_1 / total_count if total_count > 0 else 0
        final_hit_rate_3 = hit_rate_at_3 / total_count if total_count > 0 else 0
        final_mean_sim = np.mean(correct_category_similarities) if correct_category_similarities else 0

        logging.info(f"Evaluation on {self.name} dataset after epoch {epoch} and steps {steps}:")
        logging.info(f"Hit Rate @1: {final_hit_rate_1:.4f}")
        logging.info(f"Hit Rate @3: {final_hit_rate_3:.4f}")
        logging.info(f"Mean Cosine Sim (Correct): {final_mean_sim:.4f}")

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            csv_path = os.path.join(output_path, self.csv_file)
            file_exists = os.path.isfile(csv_path)
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(self.csv_headers)
                writer.writerow([epoch, steps, final_hit_rate_1, final_hit_rate_3, final_mean_sim])

        # 이 평가 점수를 기준으로 best model을 저장하게 됨
        return final_hit_rate_1

class Finetuner:
    def __init__(self, args):
        self.args = args
        set_seed(self.args.seed)

        # config와 model_utils를 사용
        self.categories_definitions = CATEGORIES_DEFINITIONS
        self.ner_special_tokens = NER_SPECIAL_TOKENS

        if not torch.cuda.is_available():
            logging.warning("CUDA를 사용할 수 없습니다. CPU로 학습을 진행합니다. 시간이 오래 걸릴 수 있습니다.")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_and_prepare_data(self):
        """CSV 파일에서 데이터를 로드하고 학습/테스트용으로 분할 및 전처리합니다."""
        logging.info(f"데이터 로딩 및 전처리 시작: {self.args.data_path}")
        try:
            df = pd.read_csv(self.args.data_path, on_bad_lines='skip')
            df.dropna(subset=['title', 'categories'], inplace=True)
            df['categories'] = df['categories'].apply(lambda x: x.split(';') if isinstance(x, str) else [])
            logging.info(f"'{self.args.data_path}'에서 {len(df)}개의 데이터를 로드했습니다.")
        except FileNotFoundError:
            logging.error(f"데이터 파일을 찾을 수 없습니다: '{self.args.data_path}'")
            return None, None

        train_df, test_df = train_test_split(df, test_size=self.args.test_size, random_state=self.args.seed)
        logging.info(f"데이터 분할 완료. 학습 데이터: {len(train_df)}개, 테스트 데이터: {len(test_df)}개")

        train_df = train_df.copy()
        test_df = test_df.copy()

        train_df['generalized_title'] = ner_generalize_texts(train_df['title'].tolist())
        test_df['generalized_title'] = ner_generalize_texts(test_df['title'].tolist())

        return train_df, test_df
    # def ner_generalize_texts(self, texts: list[str]):


    def create_input_examples(self, df: pd.DataFrame):
        """데이터프레임으로부터 InputExample 리스트를 생성합니다."""
        examples = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="학습 예시 생성 중"):
            title = row['generalized_title']
            for category in row['categories']:
                if category in self.categories_definitions and self.categories_definitions[category]:
                    keyword = random.choice(self.categories_definitions[category])
                    examples.append(InputExample(texts=[title, keyword]))
        return examples

    def run_finetuning(self):
        """전체 파인튜닝 파이프라인을 실행합니다."""
        train_df, test_df = self.load_and_prepare_data()
        if train_df is None:
            return

        train_examples = self.create_input_examples(train_df)
        if not train_examples:
            logging.error("학습 예시를 생성할 수 없습니다. 데이터나 카테고리 정의를 확인해주세요.")
            return

        logging.info(f"'{self.args.model_name}' 모델을 로드합니다.")
        model = SentenceTransformer(self.args.model_name, device=self.device)
        model.tokenizer.add_special_tokens({"additional_special_tokens": self.ner_special_tokens})
        model._first_module().auto_model.resize_token_embeddings(len(model.tokenizer))

        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=self.args.batch_size)
        train_loss = losses.MultipleNegativesRankingLoss(model)

        # 평가자(Evaluator) 설정 
        evaluator = CategoryAccuracyEvaluator(test_df, self.categories_definitions, name='category-test')

        warmup_steps = math.ceil(len(train_dataloader) * self.args.num_epochs * self.args.warmup_ratio)
        logging.info(f"학습 파라미터: epochs={self.args.num_epochs}, batch_size={self.args.batch_size}, lr={self.args.lr}, warmup_steps={warmup_steps}")

        logging.info("모델 파인튜닝을 시작합니다...")
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=self.args.num_epochs,
            warmup_steps=warmup_steps,
            optimizer_params={'lr': self.args.lr},
            output_path=self.args.output_path,
            evaluation_steps=self.args.evaluation_steps,
            save_best_model=True,
            checkpoint_path=os.path.join(self.args.output_path, 'checkpoints'),
            checkpoint_save_steps=self.args.evaluation_steps,
            show_progress_bar=True
        )
        logging.info(f"모델 파인튜닝 완료. 최적 모델이 '{self.args.output_path}'에 저장되었습니다.")
        logging.info(f"평가 결과는 '{os.path.join(self.args.output_path, evaluator.csv_file)}' 파일에서 확인할 수 있습니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="주어진 데이터를 사용하여 Sentence Transformer 모델을 파인튜닝합니다.")

    # 경로 및 모델 설정 
    parser.add_argument("--model_name", type=str, default=BASE_MODEL_NAME, help="파인튜닝할 기본 모델 이름")
    parser.add_argument("--ner_model_name", type=str, default=NER_MODEL_NAME, help="NER에 사용할 모델 이름 (현재는 model_utils에서 직접 사용)")
    parser.add_argument("--data_path", type=str, default=DATA_PATH, help="학습 데이터 CSV 파일 경로")
    parser.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT_PATH, help="파인튜닝된 모델과 결과를 저장할 경로")

    # 하이퍼파라미터 
    parser.add_argument("--num_epochs", type=int, default=3, help="총 학습 에포크 수")
    parser.add_argument("--batch_size", type=int, default=32, help="학습 배치 사이즈")
    parser.add_argument("--lr", type=float, default=5e-5, help="학습률 (Learning Rate)")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="전체 스텝 대비 웜업 스텝의 비율")
    parser.add_argument("--evaluation_steps", type=int, default=100, help="모델 평가를 수행할 스텝 간격")

    # 데이터 및 재현성 설정 
    parser.add_argument("--test_size", type=float, default=0.2, help="전체 데이터에서 테스트 데이터가 차지할 비율")
    parser.add_argument("--seed", type=int, default=42, help="실험 재현성을 위한 랜덤 시드")

    args = parser.parse_args()

    if not os.path.isabs(args.data_path):
        args.data_path = os.path.join(PROJECT_ROOT, args.data_path)
    if not os.path.isabs(args.output_path):
        args.output_path = os.path.join(PROJECT_ROOT, args.output_path)

    finetuner = Finetuner(args)
    finetuner.run_finetuning()
