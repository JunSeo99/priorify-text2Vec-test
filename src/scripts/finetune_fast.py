import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation, util
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm
import logging
import argparse
import math
from sentence_transformers.evaluation import SentenceEvaluator
import csv
import sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from torch.utils.data import DataLoader
import json
import warnings
from collections import defaultdict
from sklearn.metrics import f1_score, classification_report
warnings.filterwarnings("ignore", category=UserWarning)

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

def set_seed(seed):
    """모든 난수 생성기의 시드를 고정하여 재현성을 보장합니다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class FastCategoryEvaluator(SentenceEvaluator):
    """빠른 카테고리 분류 정확도 평가"""
    def __init__(self, test_df: pd.DataFrame, categories_definitions: dict, 
                 name: str = '', batch_size: int = 64, show_progress_bar: bool = False):
        self.test_df = test_df
        self.name = name
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.categories_definitions = categories_definitions
        self.category_names = list(categories_definitions.keys())
        
        # 카테고리 임베딩 캐시 (한 번만 계산)
        self.category_embeddings = None
        self.csv_file = "fast_evaluation_results.csv"
        self.csv_headers = ["epoch", "steps", "hit_rate@1", "hit_rate@3", "f1_score"]

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        # 카테고리 임베딩이 없으면 생성 (한 번만)
        if self.category_embeddings is None:
            self._compute_category_embeddings(model)
        
        # 테스트 데이터 임베딩 (배치 처리)
        test_titles = self.test_df['generalized_title'].tolist()
        test_embeddings = model.encode(test_titles, batch_size=self.batch_size, 
                                     convert_to_numpy=True, normalize_embeddings=True)
        
        # 유사도 계산 (벡터화)
        similarities = np.dot(test_embeddings, self.category_embeddings.T)
        
        # Hit Rate 계산
        hit1_count = 0
        hit3_count = 0
        all_true_labels = []
        all_pred_labels = []
        
        for i, (_, row) in enumerate(self.test_df.iterrows()):
            true_categories = set(row['categories'])
            sim_scores = similarities[i]
            
            # Top-1 예측
            top1_idx = np.argmax(sim_scores)
            top1_category = self.category_names[top1_idx]
            
            # Top-3 예측
            top3_indices = np.argsort(sim_scores)[-3:][::-1]
            top3_categories = [self.category_names[idx] for idx in top3_indices]
            
            if top1_category in true_categories:
                hit1_count += 1
            
            if any(cat in true_categories for cat in top3_categories):
                hit3_count += 1
            
            # F1 스코어용 레이블 준비
            all_true_labels.append(list(true_categories)[0] if true_categories else 'unknown')
            all_pred_labels.append(top1_category)
        
        # 메트릭 계산
        hit_rate_1 = hit1_count / len(self.test_df)
        hit_rate_3 = hit3_count / len(self.test_df)
        
        # F1 스코어 계산
        try:
            f1 = f1_score(all_true_labels, all_pred_labels, average='weighted', zero_division=0)
        except:
            f1 = 0.0
        
        # 결과 저장
        if output_path:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
            
            with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, steps, hit_rate_1, hit_rate_3, f1])
        
        logging.info(f"{self.name} - Epoch {epoch}: Hit@1={hit_rate_1:.3f}, Hit@3={hit_rate_3:.3f}, F1={f1:.3f}")
        
        return hit_rate_1  # 주요 메트릭으로 Hit@1 반환

    def _compute_category_embeddings(self, model):
        """카테고리 임베딩 계산 (캐시됨)"""
        # 간단한 앙상블: 카테고리명 + 상위 3개 키워드
        category_texts = []
        for category in self.category_names:
            keywords = self.categories_definitions.get(category, [])
            if keywords:
                # 상위 3개 키워드만 사용
                top_keywords = keywords[:3]
                category_text = f"{category} {' '.join(top_keywords)}"
            else:
                category_text = category
            category_texts.append(category_text)
        
        # 배치로 임베딩 계산
        self.category_embeddings = model.encode(category_texts, batch_size=self.batch_size,
                                              convert_to_numpy=True, normalize_embeddings=True)
        logging.info(f"카테고리 임베딩 계산 완료: {len(self.category_names)}개")

class FastFinetuner:
    """성능 최적화된 빠른 파인튜너"""
    def __init__(self, args):
        self.args = args
        self.categories_definitions = CATEGORIES_DEFINITIONS
        self.category_names = list(self.categories_definitions.keys())
        
        # 로깅 설정
        log_file = os.path.join(args.output_path, 'fast_training.log')
        os.makedirs(args.output_path, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)

    def load_and_prepare_data(self):
        """데이터 로드 및 준비"""
        logging.info(f"데이터 로드 시작: {self.args.data_path}")
        
        df = pd.read_csv(self.args.data_path)
        df['categories'] = df['categories'].apply(lambda x: x.split(';') if isinstance(x, str) else x)
        
        # NER 처리 (배치로 한 번에)
        logging.info("NER 일반화 처리 중...")
        titles = df['title'].tolist()
        generalized_titles = ner_generalize_texts(titles)
        df['generalized_title'] = generalized_titles
        
        # 데이터 분할 (간단하게)
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, 
                                           stratify=df['categories'].apply(lambda x: x[0] if x else 'unknown'))
        
        logging.info(f"학습 데이터: {len(train_df)}, 테스트 데이터: {len(test_df)}")
        
        self.train_df = train_df
        self.test_df = test_df
        
        return train_df, test_df

    def create_simple_examples(self, df: pd.DataFrame):
        """간단하고 빠른 학습 예시 생성"""
        examples = []
        
        # 클래스 가중치 계산 (간단한 버전)
        all_categories = []
        for _, row in df.iterrows():
            all_categories.extend(row['categories'])
        
        category_counts = {cat: all_categories.count(cat) for cat in set(all_categories)}
        total_samples = len(all_categories)
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="학습 예시 생성"):
            title = row['generalized_title']
            
            for category in row['categories']:
                if category in self.categories_definitions:
                    # 클래스 불균형 해결을 위한 반복 횟수 조정
                    weight = total_samples / (len(category_counts) * category_counts[category])
                    repeat_count = min(int(weight), 3)  # 최대 3회
                    
                    # 카테고리 텍스트 (간단한 앙상블)
                    keywords = self.categories_definitions[category][:3]  # 상위 3개만
                    category_text = f"{category} {' '.join(keywords)}" if keywords else category
                    
                    for _ in range(repeat_count):
                        examples.append(InputExample(texts=[title, category_text], label=1.0))
                    
                    # Hard negative (무작위 다른 카테고리 1개)
                    other_categories = [cat for cat in self.category_names if cat != category]
                    if other_categories:
                        neg_category = random.choice(other_categories)
                        neg_keywords = self.categories_definitions[neg_category][:3]
                        neg_text = f"{neg_category} {' '.join(neg_keywords)}" if neg_keywords else neg_category
                        examples.append(InputExample(texts=[title, neg_text], label=0.0))
        
        logging.info(f"총 {len(examples)}개의 학습 예시 생성")
        return examples

    def run_fast_finetuning(self):
        """빠른 파인튜닝 실행"""
        set_seed(42)
        
        # 데이터 준비
        train_df, test_df = self.load_and_prepare_data()
        
        # 모델 로드
        logging.info(f"모델 로드: {BASE_MODEL_NAME}")
        model = SentenceTransformer(BASE_MODEL_NAME)
        
        # GPU 설정
        if torch.cuda.is_available():
            model = model.to('cuda')
            logging.info("CUDA 사용 가능 - GPU로 학습")
        
        # 학습 예시 생성
        train_examples = self.create_simple_examples(train_df)
        
        # 데이터로더 설정 (배치 크기 증가)
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=self.args.batch_size)
        
        # 손실 함수 (단순하게)
        train_loss = losses.CosineSimilarityLoss(model)
        
        # 평가자 설정
        evaluator = FastCategoryEvaluator(test_df, self.categories_definitions, name='fast_eval', batch_size=64)
        
        # 학습 설정
        warmup_steps = math.ceil(len(train_dataloader) * self.args.max_epochs * 0.1)  # 10% 웜업
        
        logging.info("빠른 파인튜닝 시작")
        
        # 학습 실행
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=self.args.max_epochs,
            evaluation_steps=len(train_dataloader) // 2,  # 에포크당 2번 평가
            warmup_steps=warmup_steps,
            output_path=self.args.output_path,
            save_best_model=True,
            optimizer_params={'lr': 2e-5},  # 학습률 고정
            scheduler='constantlr',  # 간단한 스케줄러
            show_progress_bar=True
        )
        
        # 최종 모델 저장
        model.save(os.path.join(self.args.output_path, "final_model"))
        
        # 최종 평가
        final_score = evaluator(model, self.args.output_path, epoch="final")
        logging.info(f"최종 Hit@1 점수: {final_score:.4f}")
        
        return model

def main():
    parser = argparse.ArgumentParser(description='빠른 SentenceTransformer 파인튜닝')
    parser.add_argument('--data_path', type=str, default=DATA_PATH, help='데이터 파일 경로')
    parser.add_argument('--output_path', type=str, default=DEFAULT_OUTPUT_PATH, help='모델 출력 경로')
    parser.add_argument('--max_epochs', type=int, default=3, help='최대 에포크 수')
    parser.add_argument('--batch_size', type=int, default=32, help='배치 크기')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_path, exist_ok=True)
    
    # 파인튜닝 실행
    finetuner = FastFinetuner(args)
    model = finetuner.run_fast_finetuning()
    
    print(f"파인튜닝 완료! 모델이 {args.output_path}에 저장되었습니다.")

if __name__ == "__main__":
    main() 