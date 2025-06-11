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

class SimpleEnsembleWeights(nn.Module):
    """간단한 앙상블 가중치 (성능 유지)"""
    def __init__(self, input_dim=768):
        super().__init__()
        self.weight_net = nn.Sequential(
            nn.Linear(input_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, ner_emb, original_emb):
        combined = torch.cat([ner_emb, original_emb], dim=-1)
        weights = self.weight_net(combined)
        return weights

class BalancedCategoryEvaluator(SentenceEvaluator):
    """성능과 속도 균형을 맞춘 평가자"""
    def __init__(self, test_df: pd.DataFrame, categories_definitions: dict, 
                 name: str = '', batch_size: int = 64, show_progress_bar: bool = False):
        self.test_df = test_df
        self.name = name
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.categories_definitions = categories_definitions
        self.category_names = list(categories_definitions.keys())
        
        # 카테고리 임베딩 캐시
        self.category_embeddings = None
        self.csv_file = "balanced_evaluation_results.csv"
        self.csv_headers = ["epoch", "steps", "hit_rate@1", "hit_rate@3", "f1_score"]
        
        # 성능을 위한 다층 카테고리 텍스트 준비
        self.category_texts = self._prepare_enhanced_category_texts()

    def _prepare_enhanced_category_texts(self):
        """성능 향상을 위한 향상된 카테고리 텍스트"""
        enhanced_texts = {}
        
        for category in self.category_names:
            keywords = self.categories_definitions.get(category, [])
            
            if keywords:
                # 다양한 레이어 조합 (속도와 성능 균형)
                simple_text = category
                keyword_text = f"{category} {' '.join(keywords[:5])}"  # 상위 5개
                ner_keywords = ner_generalize_texts(keywords[:3])  # NER 처리된 상위 3개
                ner_text = f"{category} {' '.join(ner_keywords)}"
                
                # 가중 평균 텍스트 (성능 중요)
                enhanced_texts[category] = {
                    'simple': simple_text,
                    'keywords': keyword_text,
                    'ner': ner_text,
                    'ensemble': f"{simple_text} {keyword_text}"  # 앙상블
                }
            else:
                enhanced_texts[category] = {
                    'simple': category,
                    'keywords': category,
                    'ner': category,
                    'ensemble': category
                }
        
        return enhanced_texts

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        # 카테고리 임베딩이 없으면 생성 (다층 앙상블)
        if self.category_embeddings is None:
            self._compute_enhanced_category_embeddings(model)
        
        # 테스트 데이터 임베딩 (NER + 원본 앙상블)
        test_titles = self.test_df['generalized_title'].tolist()
        original_titles = self.test_df['title'].tolist()
        
        # 두 가지 임베딩 계산
        ner_embeddings = model.encode(test_titles, batch_size=self.batch_size, 
                                    convert_to_numpy=True, normalize_embeddings=True)
        orig_embeddings = model.encode(original_titles, batch_size=self.batch_size,
                                     convert_to_numpy=True, normalize_embeddings=True)
        
        # 앙상블 임베딩 (성능 향상)
        ensemble_embeddings = 0.6 * ner_embeddings + 0.4 * orig_embeddings
        
        # 유사도 계산 (벡터화)
        similarities = np.dot(ensemble_embeddings, self.category_embeddings.T)
        
        # Hit Rate 및 F1 계산
        hit1_count = 0
        hit3_count = 0
        all_true_labels = []
        all_pred_labels = []
        
        for i, (_, row) in enumerate(self.test_df.iterrows()):
            true_categories = set(row['categories'])
            sim_scores = similarities[i]
            
            # Top-1, Top-3 예측
            top1_idx = np.argmax(sim_scores)
            top1_category = self.category_names[top1_idx]
            
            top3_indices = np.argsort(sim_scores)[-3:][::-1]
            top3_categories = [self.category_names[idx] for idx in top3_indices]
            
            if top1_category in true_categories:
                hit1_count += 1
            
            if any(cat in true_categories for cat in top3_categories):
                hit3_count += 1
            
            # F1 스코어용 레이블
            all_true_labels.append(list(true_categories)[0] if true_categories else 'unknown')
            all_pred_labels.append(top1_category)
        
        # 메트릭 계산
        hit_rate_1 = hit1_count / len(self.test_df)
        hit_rate_3 = hit3_count / len(self.test_df)
        
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
        
        return hit_rate_1

    def _compute_enhanced_category_embeddings(self, model):
        """향상된 카테고리 임베딩 계산 (다층 앙상블)"""
        # 각 레이어별 임베딩 계산
        simple_texts = [self.category_texts[cat]['simple'] for cat in self.category_names]
        keyword_texts = [self.category_texts[cat]['keywords'] for cat in self.category_names]
        ner_texts = [self.category_texts[cat]['ner'] for cat in self.category_names]
        
        # 배치로 임베딩 계산
        simple_embs = model.encode(simple_texts, batch_size=self.batch_size, 
                                 convert_to_numpy=True, normalize_embeddings=True)
        keyword_embs = model.encode(keyword_texts, batch_size=self.batch_size,
                                  convert_to_numpy=True, normalize_embeddings=True)
        ner_embs = model.encode(ner_texts, batch_size=self.batch_size,
                              convert_to_numpy=True, normalize_embeddings=True)
        
        # 가중 앙상블 (성능 최적화)
        self.category_embeddings = (0.3 * simple_embs + 0.4 * keyword_embs + 0.3 * ner_embs)
        
        logging.info(f"향상된 카테고리 임베딩 계산 완료: {len(self.category_names)}개")

class BalancedFinetuner:
    """성능과 속도의 균형을 맞춘 파인튜너"""
    def __init__(self, args):
        self.args = args
        self.categories_definitions = CATEGORIES_DEFINITIONS
        self.category_names = list(self.categories_definitions.keys())
        
        # 로깅 설정
        log_file = os.path.join(args.output_path, 'balanced_training.log')
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
        
        # 데이터 분할
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, 
                                           stratify=df['categories'].apply(lambda x: x[0] if x else 'unknown'))
        
        logging.info(f"학습 데이터: {len(train_df)}, 테스트 데이터: {len(test_df)}")
        
        self.train_df = train_df
        self.test_df = test_df
        
        return train_df, test_df

    def create_balanced_examples(self, df: pd.DataFrame):
        """성능과 속도 균형을 맞춘 학습 예시 생성"""
        examples = []
        
        # 향상된 클래스 가중치 계산
        all_categories = []
        for _, row in df.iterrows():
            all_categories.extend(row['categories'])
        
        # 클래스 가중치 (sklearn 기반)
        unique_categories = list(set(all_categories))
        class_weights = compute_class_weight('balanced', classes=np.array(unique_categories), y=all_categories)
        weight_dict = dict(zip(unique_categories, class_weights))
        
        # 카테고리별 텍스트 준비 (성능 향상)
        category_texts = self._prepare_category_texts()
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="균형 학습 예시 생성"):
            title = row['generalized_title']
            original_title = row['title']
            
            for category in row['categories']:
                if category in self.categories_definitions:
                    weight = weight_dict.get(category, 1.0)
                    repeat_count = min(int(weight * 1.5), 4)  # 최대 4회
                    
                    category_info = category_texts[category]
                    
                    for _ in range(repeat_count):
                        # 다양한 positive 조합 (성능 향상)
                        examples.extend([
                            InputExample(texts=[title, category_info['simple']], label=1.0),
                            InputExample(texts=[title, category_info['keywords']], label=1.0),
                            InputExample(texts=[original_title, category_info['ner']], label=1.0),
                            InputExample(texts=[title, category_info['ensemble']], label=1.0)
                        ])
                    
                    # 스마트 Hard Negative (유사도 기반)
                    negative_categories = self._get_similar_categories(category, exclude=row['categories'])
                    for neg_cat in negative_categories[:2]:  # 상위 2개
                        neg_info = category_texts[neg_cat]
                        examples.append(
                            InputExample(texts=[title, neg_info['simple']], label=0.0)
                        )
        
        logging.info(f"총 {len(examples)}개의 균형 학습 예시 생성")
        return examples
    
    def _prepare_category_texts(self):
        """성능 향상을 위한 카테고리 텍스트 준비"""
        category_texts = {}
        
        for category in self.category_names:
            keywords = self.categories_definitions.get(category, [])
            
            if keywords:
                # 성능 최적화된 텍스트 조합
                simple_text = category
                keyword_text = f"{category} {' '.join(keywords[:6])}"  # 상위 6개
                ner_keywords = ner_generalize_texts(keywords[:4])
                ner_text = f"{category} {' '.join(ner_keywords)}"
                ensemble_text = f"{category} {' '.join(keywords[:3])}"
            else:
                simple_text = keyword_text = ner_text = ensemble_text = category
            
            category_texts[category] = {
                'simple': simple_text,
                'keywords': keyword_text,
                'ner': ner_text,
                'ensemble': ensemble_text
            }
        
        return category_texts
    
    def _get_similar_categories(self, target_category, exclude=None):
        """유사한 카테고리 반환 (Hard Negative용)"""
        if exclude is None:
            exclude = []
        
        # 단순 키워드 겹침 기반 유사도 (빠른 계산)
        target_keywords = set(self.categories_definitions.get(target_category, []))
        similarities = []
        
        for category in self.category_names:
            if category != target_category and category not in exclude:
                cat_keywords = set(self.categories_definitions.get(category, []))
                if target_keywords and cat_keywords:
                    overlap = len(target_keywords & cat_keywords)
                    similarity = overlap / len(target_keywords | cat_keywords)
                else:
                    similarity = 0.0
                similarities.append((category, similarity))
        
        # 유사도 순 정렬
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [cat for cat, sim in similarities[:5]]  # 상위 5개

    def run_balanced_finetuning(self):
        """균형 잡힌 파인튜닝 실행"""
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
        train_examples = self.create_balanced_examples(train_df)
        
        # 데이터로더 설정
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=self.args.batch_size)
        
        # 손실 함수 (CosineSimilarity 유지 - 성능 검증됨)
        train_loss = losses.CosineSimilarityLoss(model)
        
        # 평가자 설정
        evaluator = BalancedCategoryEvaluator(test_df, self.categories_definitions, 
                                            name='balanced_eval', batch_size=self.args.batch_size)
        
        # 학습 설정
        warmup_steps = math.ceil(len(train_dataloader) * self.args.max_epochs * 0.1)
        
        logging.info("균형 잡힌 파인튜닝 시작")
        
        # 학습 실행
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=self.args.max_epochs,
            evaluation_steps=len(train_dataloader) // 2,
            warmup_steps=warmup_steps,
            output_path=self.args.output_path,
            save_best_model=True,
            optimizer_params={'lr': 2e-5},
            scheduler='warmupcosine',  # 더 나은 스케줄러
            show_progress_bar=True
        )
        
        # 최종 모델 저장
        model.save(os.path.join(self.args.output_path, "final_model"))
        
        # 최종 평가
        final_score = evaluator(model, self.args.output_path, epoch="final")
        logging.info(f"최종 Hit@1 점수: {final_score:.4f}")
        
        return model

def main():
    parser = argparse.ArgumentParser(description='균형 잡힌 SentenceTransformer 파인튜닝')
    parser.add_argument('--data_path', type=str, default=DATA_PATH, help='데이터 파일 경로')
    parser.add_argument('--output_path', type=str, default=DEFAULT_OUTPUT_PATH, help='모델 출력 경로')
    parser.add_argument('--max_epochs', type=int, default=3, help='최대 에포크 수')
    parser.add_argument('--batch_size', type=int, default=32, help='배치 크기')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_path, exist_ok=True)
    
    # 파인튜닝 실행
    finetuner = BalancedFinetuner(args)
    model = finetuner.run_balanced_finetuning()
    
    print(f"균형 잡힌 파인튜닝 완료! 모델이 {args.output_path}에 저장되었습니다.")

if __name__ == "__main__":
    main() 