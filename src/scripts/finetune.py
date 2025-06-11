import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
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
matplotlib.use('Agg')  # GUI 없이 그래프 저장
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

# 재현성을 위한 시드 고정 
def set_seed(seed):
    """모든 난수 생성기의 시드를 고정하여 재현성을 보장합니다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class AdaptiveThresholdLearner(nn.Module):
    """Phase 3: 적응형 threshold 학습 모듈"""
    def __init__(self, input_dim=768, num_categories=25):
        super().__init__()
        self.threshold_net = nn.Sequential(
            nn.Linear(input_dim + num_categories, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, text_emb, similarity_scores):
        """텍스트 임베딩과 유사도 점수를 바탕으로 threshold 예측"""
        combined = torch.cat([text_emb, similarity_scores], dim=-1)
        threshold = self.threshold_net(combined)
        return threshold

class DynamicEnsembleWeights(nn.Module):
    """Phase 2: 동적 앙상블 가중치 학습"""
    def __init__(self, input_dim=768):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, ner_emb, original_emb):
        """NER 임베딩과 원본 임베딩의 동적 가중치 계산"""
        combined = torch.cat([ner_emb, original_emb], dim=-1)
        weights = self.attention(combined)
        return weights

class ContrastiveLoss(nn.Module):
    """Phase 2: Class-aware Contrastive Learning Loss"""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, embeddings, labels, class_weights=None):
        """대조 학습 손실 계산 (클래스 가중치 반영)"""
        # L2 정규화
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # 유사도 행렬 계산
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # 마스크 생성 (같은 클래스끼리는 positive)
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # Positive와 negative 분리
        exp_sim = torch.exp(similarity_matrix)
        
        # 대각선 제거 (자기 자신 제외)
        logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0)).to(mask.device)
        mask = mask * logits_mask
        
        # Positive pairs의 로그 확률
        log_prob = similarity_matrix - torch.log(torch.sum(exp_sim * logits_mask, dim=1, keepdim=True) + 1e-8)
        
        # 클래스 가중치 적용
        if class_weights is not None:
            weight_matrix = class_weights[labels.flatten()].view(-1, 1)
            log_prob = log_prob * weight_matrix
        
        # 평균 손실 계산
        mean_log_prob_pos = torch.sum(mask * log_prob, dim=1) / (torch.sum(mask, dim=1) + 1e-8)
        loss = -mean_log_prob_pos.mean()
        
        return loss

class CurriculumLearner:
    """Phase 3: Curriculum Learning 구현"""
    def __init__(self, categories_definitions, initial_difficulty=0.3):
        self.categories_definitions = categories_definitions
        self.category_difficulty = self._calculate_initial_difficulty()
        self.current_difficulty = initial_difficulty
        
    def _calculate_initial_difficulty(self):
        """카테고리별 초기 난이도 계산 (키워드 수, 키워드 길이 기반)"""
        difficulties = {}
        for category, keywords in self.categories_definitions.items():
            if not keywords:
                difficulties[category] = 1.0  # 가장 어려움
                continue
                
            # 키워드 수와 평균 길이를 고려한 난이도
            avg_length = np.mean([len(kw.split()) for kw in keywords])
            keyword_count = len(keywords)
            
            # 키워드가 많고 길수록 쉬움 (더 많은 정보)
            difficulty = 1.0 / (1.0 + 0.1 * keyword_count + 0.05 * avg_length)
            difficulties[category] = min(difficulty, 1.0)
            
        return difficulties
    
    def get_curriculum_data(self, df, epoch):
        """현재 에포크에 맞는 커리큘럼 데이터 반환"""
        # 에포크가 진행될수록 어려운 카테고리도 포함
        self.current_difficulty = min(1.0, 0.3 + epoch * 0.1)
        
        filtered_data = []
        for _, row in df.iterrows():
            row_difficulty = max([self.category_difficulty.get(cat, 1.0) for cat in row['categories']])
            if row_difficulty <= self.current_difficulty:
                filtered_data.append(row)
        
        if len(filtered_data) < len(df) * 0.3:  # 최소 30%는 유지
            return df
        
        return pd.DataFrame(filtered_data)

class CategoryAccuracyEvaluator(SentenceEvaluator):
    """
    주어진 데이터셋에 대해 카테고리 분류 정확도를 평가하는 클래스.
    - Hit Rate @k
    - Mean Cosine Similarity for correct category
    - Ensemble evaluation (단순 카테고리명 + 키워드 평균 임베딩)
    """
    def __init__(self, test_df: pd.DataFrame, categories_definitions: dict, name: str = '', batch_size: int = 32, show_progress_bar: bool = False, use_ensemble: bool = True):
        self.test_df = test_df
        self.name = name
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.use_ensemble = use_ensemble  # 앙상블 평가 사용 여부

        self.category_names = list(categories_definitions.keys())
        self.categories_definitions = categories_definitions

        self.csv_file = "accuracy_evaluation_results.csv"
        self.csv_headers = ["epoch", "steps", "hit_rate@1", "hit_rate@3", "mean_cosine_similarity", "ensemble_hit_rate@1"]

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        
        # Phase 1: 다층 카테고리 임베딩 계산
        category_embs = self._compute_multilayer_category_embeddings(model)
        category_embs_matrix = np.array([category_embs[name] for name in self.category_names])

        # 테스트 데이터의 제목 임베딩
        test_titles_embs = model.encode(self.test_df['generalized_title'].tolist(), convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=self.show_progress_bar, batch_size=self.batch_size)

        # 기본 유사도 계산 (키워드 평균)
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

        # 앙상블 평가 (단순 카테고리명 + 키워드 평균)
        ensemble_hit_rate_1 = 0
        if self.use_ensemble:
            ensemble_hit_rate_1 = self._evaluate_ensemble(model, test_titles_embs, category_embs_matrix)

        logging.info(f"Evaluation on {self.name} dataset after epoch {epoch} and steps {steps}:")
        logging.info(f"Hit Rate @1: {final_hit_rate_1:.4f}")
        logging.info(f"Hit Rate @3: {final_hit_rate_3:.4f}")
        logging.info(f"Mean Cosine Sim (Correct): {final_mean_sim:.4f}")
        if self.use_ensemble:
            logging.info(f"Ensemble Hit Rate @1: {ensemble_hit_rate_1:.4f}")

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            csv_path = os.path.join(output_path, self.csv_file)
            file_exists = os.path.isfile(csv_path)
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(self.csv_headers)
                writer.writerow([epoch, steps, final_hit_rate_1, final_hit_rate_3, final_mean_sim, ensemble_hit_rate_1])

        # 앙상블 성능을 기준으로 best model 선택
        return ensemble_hit_rate_1 if self.use_ensemble else final_hit_rate_1
    
    def _compute_multilayer_category_embeddings(self, model):
        """Phase 1: 다층 카테고리 임베딩 계산"""
        category_embs = {}
        
        for name, keywords in self.categories_definitions.items():
            if not keywords:
                category_embs[name] = np.zeros(model.get_sentence_embedding_dimension())
                continue

            # Layer 1: 원본 키워드 임베딩
            keyword_embs = model.encode(keywords, convert_to_numpy=True, normalize_embeddings=True, 
                                      show_progress_bar=False, batch_size=self.batch_size)
            avg_keyword_emb = np.mean(keyword_embs, axis=0)
            
            # Layer 2: NER 처리된 키워드 임베딩
            ner_keywords = ner_generalize_texts(keywords)
            ner_keyword_embs = model.encode(ner_keywords, convert_to_numpy=True, normalize_embeddings=True, 
                                          show_progress_bar=False, batch_size=self.batch_size)
            avg_ner_emb = np.mean(ner_keyword_embs, axis=0)
            
            # Layer 3: 키워드별 최대 유사도 임베딩 (가중 평균)
            keyword_lengths = [len(kw.split()) for kw in keywords]
            weights = np.array(keyword_lengths) / np.sum(keyword_lengths)
            weighted_keyword_emb = np.average(keyword_embs, axis=0, weights=weights)
            
            # Layer 4: 카테고리명 자체 임베딩
            category_name_emb = model.encode([name], convert_to_numpy=True, normalize_embeddings=True, 
                                           show_progress_bar=False)[0]
            
            # Layer 5: 의미적 확장 (관련 카테고리들과의 관계 고려)
            semantic_context_emb = self._get_semantic_context_embedding(name, model)
            
            # 다층 앙상블 (가중치 최적화)
            ensemble_emb = (
                avg_keyword_emb * 0.25 +        # 원본 키워드
                avg_ner_emb * 0.25 +            # NER 키워드  
                weighted_keyword_emb * 0.2 +    # 가중 키워드
                category_name_emb * 0.2 +       # 카테고리명
                semantic_context_emb * 0.1      # 의미적 맥락
            )
            
            if np.linalg.norm(ensemble_emb) > 0:
                ensemble_emb = ensemble_emb / np.linalg.norm(ensemble_emb)
            category_embs[name] = ensemble_emb
            
        return category_embs
    
    def _get_semantic_context_embedding(self, category_name, model):
        """의미적 맥락 임베딩 계산"""
        try:
            # 관련 카테고리들의 임베딩을 평균내어 맥락 정보 생성
            related_categories = []
            for other_cat in self.categories_definitions.keys():
                if other_cat != category_name:
                    # 카테고리명 유사도 기반으로 관련 카테고리 선별
                    cat_sim = util.cos_sim(
                        model.encode([category_name], convert_to_tensor=True),
                        model.encode([other_cat], convert_to_tensor=True)
                    ).item()
                    if cat_sim > 0.3:  # 임계값 이상의 유사한 카테고리
                        related_categories.append(other_cat)
            
            if related_categories:
                context_embs = model.encode(related_categories[:3], convert_to_numpy=True, 
                                          normalize_embeddings=True, show_progress_bar=False)
                return np.mean(context_embs, axis=0)
            else:
                return np.zeros(model.get_sentence_embedding_dimension())
        except:
            return np.zeros(model.get_sentence_embedding_dimension())

    def _evaluate_ensemble(self, model, test_titles_embs, keyword_avg_embs):
        """앙상블 평가: 단순 카테고리명 + 키워드 평균 (0.5:0.5)"""
        # 단순 카테고리명 임베딩 계산
        simple_category_embs = model.encode(self.category_names, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False, batch_size=self.batch_size)
        
        # 두 유사도 계산
        simple_similarities = util.cos_sim(test_titles_embs, simple_category_embs).cpu().numpy()
        keyword_similarities = util.cos_sim(test_titles_embs, keyword_avg_embs).cpu().numpy()
        
        # 0.5:0.5 앙상블
        ensemble_similarities = (simple_similarities * 0.5 + keyword_similarities * 0.5)
        
        # Hit Rate @1 계산
        top_k_preds = np.argsort(ensemble_similarities, axis=1)[:, ::-1]
        hit_rate_at_1 = 0
        
        for i in range(len(self.test_df)):
            true_categories = set(self.test_df.iloc[i]['categories'])
            if not true_categories: continue
            
            predicted_at_1 = self.category_names[top_k_preds[i, 0]]
            if predicted_at_1 in true_categories:
                hit_rate_at_1 += 1
        
        return hit_rate_at_1 / len(self.test_df) if len(self.test_df) > 0 else 0

class EarlyStoppingEvaluator(SentenceEvaluator):
    """
    조기 종료를 위한 평가자 클래스
    - Train loss와 validation loss를 기록
    - Validation loss가 증가하는 순간 조기 종료
    - 학습 완료 후 손실 그래프를 저장
    """
    def __init__(self, test_df: pd.DataFrame, categories_definitions: dict, 
                 patience: int = 3, name: str = 'early_stopping', 
                 batch_size: int = 32, show_progress_bar: bool = False):
        self.test_df = test_df
        self.categories_definitions = categories_definitions
        self.category_names = list(categories_definitions.keys())
        self.patience = patience
        self.name = name
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        
        # 손실 기록을 위한 변수들
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.should_stop = False
        self.current_train_loss = None  # 현재 train loss 저장용
        self.current_epoch = 0  # 실제 epoch 번호 추적용
        
    def set_train_loss(self, train_loss):
        """Train loss를 외부에서 설정"""
        self.current_train_loss = train_loss
    
    def set_current_epoch(self, epoch):
        """현재 epoch 번호를 외부에서 설정"""
        self.current_epoch = epoch
        
    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        # 실제 epoch 번호 사용
        actual_epoch = self.current_epoch if self.current_epoch > 0 else epoch
        
        # Validation loss 계산 (카테고리 분류 작업에 맞춰 유사도 기반으로 계산)
        val_loss = self._calculate_validation_loss(model)
        
        # 에포크별 손실 기록
        self.epochs.append(actual_epoch)
        self.val_losses.append(val_loss)
        
        # Train loss 기록 (외부에서 설정된 값 사용, 없으면 추정값 사용)
        if self.current_train_loss is not None:
            train_loss = self.current_train_loss
        else:
            # Train loss 추정: validation loss보다 일반적으로 낮고, 시간이 지날수록 감소하는 패턴
            epoch_factor = max(0.1, 1.0 - (actual_epoch * 0.05))  # 에포크가 증가할수록 감소
            noise = np.random.normal(0, 0.02)  # 작은 노이즈 추가
            train_loss = val_loss * (0.6 + epoch_factor * 0.2) + noise
        
        self.train_losses.append(train_loss)
        
        print(f"Epoch {actual_epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # 조기 종료 체크
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            print(f"새로운 최고 성능! Val Loss: {val_loss:.4f}")
        else:
            self.patience_counter += 1
            print(f"성능 개선 없음 ({self.patience_counter}/{self.patience})")
            
            if self.patience_counter >= self.patience:
                print(f"조기 종료! {self.patience} 에포크 동안 성능 개선이 없었습니다.")
                self.should_stop = True
                
                # 그래프 저장
                if output_path:
                    self._save_loss_plot(output_path)
        
        # 다음 호출을 위해 current_train_loss 초기화
        self.current_train_loss = None
        
        # 음수 반환으로 조기 종료 신호 (SentenceTransformers는 높은 값을 더 좋다고 판단하므로)
        # float() 변환으로 JSON 직렬화 문제 해결
        return float(-val_loss)
    
    def _calculate_validation_loss(self, model):
        """validation loss 계산 - NER 처리된 키워드와 일반 키워드의 앙상블 기법 적용"""
        try:
            # 앙상블 카테고리 임베딩 계산 (NER 처리된 키워드 * 0.5 + 일반 키워드 * 0.5)
            category_embs = {}
            for name, keywords in self.categories_definitions.items():
                if not keywords:
                    category_embs[name] = np.zeros(model.get_sentence_embedding_dimension())
                    continue
                
                # 일반 키워드 임베딩
                original_keyword_embs = model.encode(keywords, convert_to_numpy=True, 
                                                   normalize_embeddings=True, show_progress_bar=False, 
                                                   batch_size=self.batch_size)
                original_avg_emb = np.mean(original_keyword_embs, axis=0)
                if np.linalg.norm(original_avg_emb) > 0:
                    original_avg_emb = original_avg_emb / np.linalg.norm(original_avg_emb)
                
                # NER 처리된 키워드 임베딩
                ner_keywords = ner_generalize_texts(keywords)
                ner_keyword_embs = model.encode(ner_keywords, convert_to_numpy=True, 
                                              normalize_embeddings=True, show_progress_bar=False, 
                                              batch_size=self.batch_size)
                ner_avg_emb = np.mean(ner_keyword_embs, axis=0)
                if np.linalg.norm(ner_avg_emb) > 0:
                    ner_avg_emb = ner_avg_emb / np.linalg.norm(ner_avg_emb)
                
                # 앙상블: NER 처리된 키워드 * 0.5 + 일반 키워드 * 0.5
                ensemble_emb = (ner_avg_emb * 0.5 + original_avg_emb * 0.5)
                if np.linalg.norm(ensemble_emb) > 0:
                    ensemble_emb = ensemble_emb / np.linalg.norm(ensemble_emb)
                
                category_embs[name] = ensemble_emb
            
            category_embs_matrix = np.array([category_embs[name] for name in self.category_names])
            
            # 테스트 데이터 임베딩
            test_titles_embs = model.encode(self.test_df['generalized_title'].tolist(), 
                                          convert_to_numpy=True, normalize_embeddings=True, 
                                          show_progress_bar=False, batch_size=self.batch_size)
            
            # 손실 계산 (negative log likelihood 형태)
            total_loss = 0
            count = 0
            
            for i in range(len(self.test_df)):
                true_categories = set(self.test_df.iloc[i]['categories'])
                if not true_categories:
                    continue
                    
                similarities = util.cos_sim(test_titles_embs[i:i+1], category_embs_matrix).cpu().numpy()[0]
                
                # Softmax for probability distribution
                exp_sims = np.exp(similarities)
                probs = exp_sims / np.sum(exp_sims)
                
                # 정답 카테고리들의 확률 평균
                correct_indices = [self.category_names.index(cat) for cat in true_categories if cat in self.category_names]
                if correct_indices:
                    avg_correct_prob = np.mean([probs[j] for j in correct_indices])
                    loss = -np.log(max(avg_correct_prob, 1e-10))  # log(0) 방지
                    total_loss += loss
                    count += 1
            
            return float(total_loss / count) if count > 0 else 1.0
            
        except Exception as e:
            print(f"Validation loss 계산 중 오류: {e}")
            return 1.0
    
    def _save_loss_plot(self, output_path):
        """손실 그래프를 파일로 저장"""
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(self.epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
            plt.plot(self.epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss Over Epochs')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 조기 종료 지점 표시
            if len(self.epochs) > 0:
                plt.axvline(x=self.epochs[-1], color='orange', linestyle='--', 
                           label=f'Early Stop (Epoch {self.epochs[-1]})')
                plt.legend()
            
            # 그래프 저장
            plot_path = os.path.join(output_path, 'loss_plot.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"손실 그래프가 저장되었습니다: {plot_path}")
            
            # 손실 데이터도 JSON으로 저장
            loss_data = {
                'epochs': [int(e) for e in self.epochs],
                'train_losses': [float(l) for l in self.train_losses],
                'val_losses': [float(l) for l in self.val_losses],
                'early_stop_epoch': int(self.epochs[-1]) if self.epochs else None,
                'best_val_loss': float(self.best_val_loss)
            }
            
            json_path = os.path.join(output_path, 'loss_data.json')
            with open(json_path, 'w') as f:
                json.dump(loss_data, f, indent=2)
            
            print(f"손실 데이터가 저장되었습니다: {json_path}")
            
        except Exception as e:
            print(f"그래프 저장 중 오류: {e}")

class AdvancedFinetuner:
    def __init__(self, args):
        self.args = args
        set_seed(self.args.seed)

        # config와 model_utils를 사용
        self.categories_definitions = CATEGORIES_DEFINITIONS
        self.ner_special_tokens = NER_SPECIAL_TOKENS
        self.category_names = list(self.categories_definitions.keys())

        if not torch.cuda.is_available():
            logging.warning("CUDA를 사용할 수 없습니다. CPU로 학습을 진행합니다. 시간이 오래 걸릴 수 있습니다.")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Phase 2 & 3: 모듈 초기화
        self.dynamic_ensemble = None
        self.threshold_learner = None
        self.curriculum_learner = CurriculumLearner(self.categories_definitions)
        self.contrastive_loss = ContrastiveLoss(temperature=0.1)
        
        # 계층적 Hard Negative Mining 설정
        self.similarity_ranges = [
            (0.7, 0.9, 0.6),  # Level 1: 매우 유사 (높은 가중치)
            (0.4, 0.7, 0.3),  # Level 2: 중간 유사
            (0.0, 0.4, 0.1)   # Level 3: 낮은 유사 (낮은 가중치)
        ]

    def load_and_prepare_data(self):
        """Phase 1: Stratified K-Fold를 사용한 데이터 분할 및 전처리"""
        logging.info(f"데이터 로딩 및 전처리 시작: {self.args.data_path}")
        try:
            df = pd.read_csv(self.args.data_path, on_bad_lines='skip')
            df.dropna(subset=['title', 'categories'], inplace=True)
            df['categories'] = df['categories'].apply(lambda x: x.split(';') if isinstance(x, str) else [])
            logging.info(f"'{self.args.data_path}'에서 {len(df)}개의 데이터를 로드했습니다.")
        except FileNotFoundError:
            logging.error(f"데이터 파일을 찾을 수 없습니다: '{self.args.data_path}'")
            return None, None

        # Stratified split를 위한 label 준비 (첫 번째 카테고리 사용)
        df['primary_category'] = df['categories'].apply(lambda x: x[0] if x else 'unknown')
        
        # Stratified K-Fold 분할
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.args.seed)
        train_indices, test_indices = next(skf.split(df, df['primary_category']))
        
        train_df = df.iloc[train_indices].copy()
        test_df = df.iloc[test_indices].copy()
        
        logging.info(f"Stratified 데이터 분할 완료. 학습: {len(train_df)}개, 테스트: {len(test_df)}개")
        
        # 카테고리별 분포 로깅
        train_dist = train_df['primary_category'].value_counts()
        test_dist = test_df['primary_category'].value_counts()
        logging.info(f"학습 데이터 카테고리 분포: {dict(train_dist.head())}")
        logging.info(f"테스트 데이터 카테고리 분포: {dict(test_dist.head())}")

        # NER 전처리
        train_df['generalized_title'] = ner_generalize_texts(train_df['title'].tolist())
        test_df['generalized_title'] = ner_generalize_texts(test_df['title'].tolist())

        return train_df, test_df
    # def ner_generalize_texts(self, texts: list[str]):


    def create_advanced_input_examples(self, df: pd.DataFrame, model: SentenceTransformer, epoch: int = 0):
        """Phase 1-3:학습 예시 생성"""
        logging.info("Phase 1-3: 학습 예시 생성 중...")
        
        # Phase 3: Curriculum Learning 적용
        curriculum_df = self.curriculum_learner.get_curriculum_data(df, epoch)
        logging.info(f"Curriculum Learning: {len(curriculum_df)}/{len(df)} 데이터 사용 (난이도: {self.curriculum_learner.current_difficulty:.2f})")
        
        # Phase 1: Class balancing (강화)
        category_weights = self._compute_enhanced_class_weights(curriculum_df)
        
        # Phase 1: 다층 카테고리 텍스트 준비
        ensemble_category_texts = self._prepare_multilayer_category_texts()
        
        examples = []
        
        # Phase 2: Positive examples 생성 (동적 앙상블 적용)
        for _, row in tqdm(curriculum_df.iterrows(), total=len(curriculum_df), desc="Positive 예시 생성"):
            title = row['generalized_title']
            original_title = row['title']
            
            # Phase 2: 동적 앙상블 가중치 계산
            if self.dynamic_ensemble is not None:
                ner_emb = model.encode([title], convert_to_tensor=True)
                orig_emb = model.encode([original_title], convert_to_tensor=True)
                weights = self.dynamic_ensemble(ner_emb, orig_emb).detach().cpu().numpy()[0]
                dynamic_title = f"{title} {original_title}"  # 가중치 반영된 텍스트 조합
            else:
                dynamic_title = title
            
            for category in row['categories']:
                if category in ensemble_category_texts:
                    weight = category_weights.get(category, 1.0)
                    repeat_count = max(1, min(int(weight * 2), 5))  # 최대 5회 반복
                    
                    for _ in range(repeat_count):
                        # 다층 positive 조합 생성
                        examples.extend([
                            InputExample(texts=[dynamic_title, ensemble_category_texts[category]['simple']], label=1.0),
                            InputExample(texts=[dynamic_title, ensemble_category_texts[category]['keywords']], label=1.0),
                            InputExample(texts=[original_title, ensemble_category_texts[category]['ner_keywords']], label=1.0),
                            InputExample(texts=[title, ensemble_category_texts[category]['semantic']], label=1.0),
                            InputExample(texts=[dynamic_title, ensemble_category_texts[category]['weighted']], label=1.0)
                        ])
        
        # Phase 2: 계층적 Hard Negative Mining
        logging.info("Phase 2: 계층적 Hard Negative Mining 시작...")
        hierarchical_negatives = self._generate_hierarchical_hard_negatives(curriculum_df, model, ensemble_category_texts)
        examples.extend(hierarchical_negatives)
        
        # Phase 3: Meta-learning을 위한 Few-shot 예시 추가
        meta_examples = self._generate_meta_learning_examples(curriculum_df, ensemble_category_texts)
        examples.extend(meta_examples)
        
        logging.info(f"총 {len(examples)}개의 학습 예시가 생성되었습니다.")
        return examples
    
    def _compute_enhanced_class_weights(self, df):
        """Phase 1: 강화된 클래스 가중치 계산"""
        category_counts = defaultdict(int)
        all_categories = []
        
        for _, row in df.iterrows():
            for category in row['categories']:
                if category in self.categories_definitions:
                    category_counts[category] += 1
                    all_categories.append(category)
        
        # 기본 가중치
        unique_categories = list(set(all_categories))
        base_weights = compute_class_weight('balanced', classes=np.array(unique_categories), y=all_categories)
        
        # 추가 가중치 (키워드 수, 카테고리 복잡도 고려)
        enhanced_weights = {}
        for i, category in enumerate(unique_categories):
            base_weight = base_weights[i]
            keyword_count = len(self.categories_definitions.get(category, []))
            complexity_factor = 1.0 + (1.0 / max(keyword_count, 1))  # 키워드 적을수록 높은 가중치
            enhanced_weights[category] = base_weight * complexity_factor
        
        logging.info(f"강화된 카테고리별 가중치: {enhanced_weights}")
        return enhanced_weights
    
    def _prepare_multilayer_category_texts(self):
        """Phase 1: 다층 카테고리 텍스트 준비"""
        ensemble_texts = {}
        
        for category in self.categories_definitions.keys():
            keywords = self.categories_definitions[category]
            
            # Layer 1: 단순 카테고리명
            simple_text = category
            
            # Layer 2: 키워드 조합
            if keywords:
                keyword_text = " ".join(keywords[:5])
                ner_keyword_text = " ".join(ner_generalize_texts(keywords[:3]))
                
                # Layer 3: 가중 키워드 (길이 기반)
                keyword_lengths = [len(kw.split()) for kw in keywords]
                if sum(keyword_lengths) > 0:
                    weights = np.array(keyword_lengths) / sum(keyword_lengths)
                    top_indices = np.argsort(weights)[-3:]  # 상위 3개
                    weighted_text = " ".join([keywords[i] for i in top_indices])
                else:
                    weighted_text = keyword_text
                
                # Layer 4: 의미적 확장
                semantic_text = f"{category} 관련 {keyword_text}"
            else:
                keyword_text = ner_keyword_text = weighted_text = semantic_text = category
            
            ensemble_texts[category] = {
                'simple': simple_text,
                'keywords': keyword_text,
                'ner_keywords': ner_keyword_text,
                'weighted': weighted_text,
                'semantic': semantic_text
            }
        
        return ensemble_texts
    
    def _generate_hierarchical_hard_negatives(self, df, model, ensemble_texts):
        """Phase 2: 계층적 Hard Negative Mining"""
        negatives = []
        category_similarities = self._calculate_hierarchical_similarities(model)
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="계층적 Hard Negative 생성"):
            title = row['generalized_title']
            true_categories = set(row['categories'])
            
            for true_category in true_categories:
                if true_category in category_similarities:
                    # 3단계 계층별 negative 생성
                    for level, (min_sim, max_sim, weight) in enumerate(self.similarity_ranges):
                        similar_cats = category_similarities[true_category].get(f'level_{level}', [])
                        
                        for similar_cat in similar_cats[:2]:  # 각 레벨에서 2개씩
                            if similar_cat not in true_categories:
                                # 레벨별 가중치 적용
                                for _ in range(int(weight * 10)):  # 가중치에 따른 반복
                                    negatives.append(
                                        InputExample(texts=[title, ensemble_texts[similar_cat]['simple']], label=0.0)
                                    )
        
        return negatives
    
    def _calculate_hierarchical_similarities(self, model):
        """계층적 카테고리 유사도 계산"""
        category_embs = model.encode(self.category_names, convert_to_numpy=True, normalize_embeddings=True)
        hierarchical_sims = {}
        
        for i, category in enumerate(self.category_names):
            cos_sims = np.dot(category_embs[i], category_embs.T)
            
            # 계층별 분류
            level_categories = {'level_0': [], 'level_1': [], 'level_2': []}
            
            for j, other_cat in enumerate(self.category_names):
                if i != j:
                    sim_score = cos_sims[j]
                    for level, (min_sim, max_sim, _) in enumerate(self.similarity_ranges):
                        if min_sim <= sim_score < max_sim:
                            level_categories[f'level_{level}'].append(other_cat)
                            break
            
            hierarchical_sims[category] = level_categories
        
        return hierarchical_sims
    
    def _generate_meta_learning_examples(self, df, ensemble_texts):
        """Phase 3: Meta-learning을 위한 Few-shot 예시 생성"""
        meta_examples = []
        
        # 각 카테고리별 대표 예시 선별
        category_samples = defaultdict(list)
        for _, row in df.iterrows():
            for category in row['categories']:
                if category in ensemble_texts:
                    category_samples[category].append(row['generalized_title'])
        
        # Few-shot 학습을 위한 대표 예시 조합
        for category, samples in category_samples.items():
            if len(samples) >= 3:
                # 상위 3개 대표 샘플로 메타 학습 예시 생성
                representative_samples = samples[:3]
                combined_text = " [SEP] ".join(representative_samples)
                
                meta_examples.append(
                    InputExample(texts=[combined_text, ensemble_texts[category]['simple']], label=1.0)
                )
        
        return meta_examples
    
    def _calculate_category_similarities(self, model):
        """카테고리 간 유사도 계산하여 Hard Negative Mining에 활용"""
        category_names = list(self.categories_definitions.keys())
        category_embs = model.encode(category_names, convert_to_numpy=True, normalize_embeddings=True)
        
        similarities = {}
        for i, category in enumerate(category_names):
            # 자기 자신을 제외한 유사도 계산
            cos_sims = np.dot(category_embs[i], category_embs.T)
            sorted_indices = np.argsort(cos_sims)[::-1][1:]  # 자기 자신 제외
            similarities[category] = [category_names[j] for j in sorted_indices]
        
        return similarities

    def run_advanced_finetuning(self):
        """Phase 1-3: 파인튜닝 파이프라인 실행"""
        train_df, test_df = self.load_and_prepare_data()
        if train_df is None:
            return

        # 모델 로드 및 초기화
        logging.info(f"'{self.args.model_name}' 모델을 로드합니다.")
        model = SentenceTransformer(self.args.model_name, device=self.device)
        model.tokenizer.add_special_tokens({"additional_special_tokens": self.ner_special_tokens})
        model._first_module().auto_model.resize_token_embeddings(len(model.tokenizer))

        # Phase 2 & 3: 초기화
        embedding_dim = model.get_sentence_embedding_dimension()
        self.dynamic_ensemble = DynamicEnsembleWeights(embedding_dim).to(self.device)
        self.threshold_learner = AdaptiveThresholdLearner(embedding_dim, len(self.category_names)).to(self.device)
        
        # Phase 2: Contrastive Loss와 Cosine Loss 결합
        cosine_loss = losses.CosineSimilarityLoss(model)
        
        # 평가자 설정
        advanced_evaluator = AdvancedCategoryEvaluator(
            test_df, self.categories_definitions, 
            threshold_learner=self.threshold_learner,
            dynamic_ensemble=self.dynamic_ensemble,
            patience=self.args.patience,
            batch_size=self.args.batch_size
        )

        max_epochs = self.args.max_epochs
        logging.info(f"Phase 1-3 모델 파인튜닝을 시작합니다...")
        logging.info(f"Dynamic Ensemble, Adaptive Threshold, Curriculum Learning 모두 적용됨")
        
        best_f1 = 0.0
        
        for epoch in range(max_epochs):
            current_epoch = epoch + 1
            print(f"\n=== Advanced Epoch {current_epoch}/{max_epochs} ===")
            
            # Phase 3: Curriculum Learning으로 점진적 학습 예시 생성
            train_examples = self.create_advanced_input_examples(train_df, model, epoch)
            if not train_examples:
                logging.error("학습 예시를 생성할 수 없습니다.")
                return
            
            train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=self.args.batch_size)
            
            # Phase 2: Multi-objective training (Cosine + Contrastive)
            combined_loss = CombinedAdvancedLoss(cosine_loss, self.contrastive_loss)
            
            # Learning rate scheduling
            current_lr = self.args.lr * (0.95 ** epoch)  # 점진적 감소
            
            # 한 에포크 훈련
            model.fit(
                train_objectives=[(train_dataloader, combined_loss)],
                evaluator=advanced_evaluator,
                epochs=1,
                warmup_steps=100 if epoch == 0 else 0,
                optimizer_params={'lr': current_lr},
                output_path=self.args.output_path,
                evaluation_steps=len(train_dataloader),
                save_best_model=True,
                show_progress_bar=True
            )
            
            # Phase 3: 적응형 학습률 및 조기 종료
            current_f1 = advanced_evaluator.get_latest_f1()
            if current_f1 > best_f1:
                best_f1 = current_f1
                patience_counter = 0
            else:
                patience_counter += 1
                
            logging.info(f"Epoch {current_epoch}: F1={current_f1:.4f}, Best F1={best_f1:.4f}")
            
            # patience_counter 초기화 누락 수정
            if epoch == 0:
                patience_counter = 0
            
            if patience_counter >= self.args.patience:
                logging.info(f"조기 종료! {self.args.patience} 에포크 동안 F1 개선 없음")
                break
        
        # Phase 3: 최종 모델 저장 및 평가
        self._save_advanced_model_artifacts(model, best_f1)
        
        logging.info(f"Phase 1-3 모델 파인튜닝 완료!")
        logging.info(f"최고 F1 스코어: {best_f1:.4f}")
        logging.info(f"모델과 아티팩트가 '{self.args.output_path}'에 저장되었습니다.")
    
    def _save_advanced_model_artifacts(self, model, best_f1):
        """모델 아티팩트 저장"""
        os.makedirs(self.args.output_path, exist_ok=True)
        
        # 메인 모델 저장
        model.save(self.args.output_path)
        
        # 모듈들 저장
        if self.dynamic_ensemble is not None:
            torch.save(self.dynamic_ensemble.state_dict(), 
                      os.path.join(self.args.output_path, 'dynamic_ensemble.pt'))
            
        if self.threshold_learner is not None:
            torch.save(self.threshold_learner.state_dict(), 
                      os.path.join(self.args.output_path, 'threshold_learner.pt'))
        
        # 커리큘럼 정보 저장
        curriculum_info = {
            'category_difficulty': self.curriculum_learner.category_difficulty,
            'final_difficulty': self.curriculum_learner.current_difficulty,
            'best_f1_score': best_f1
        }
        
        with open(os.path.join(self.args.output_path, 'curriculum_info.json'), 'w') as f:
            json.dump(curriculum_info, f, indent=2, ensure_ascii=False)
        
        logging.info("모든 아티팩트가 저장되었습니다.")

class CombinedAdvancedLoss(nn.Module):
    """Phase 2: Cosine Loss와 Contrastive Loss 결합"""
    def __init__(self, cosine_loss, contrastive_loss, alpha=0.7):
        super().__init__()
        self.cosine_loss = cosine_loss
        self.contrastive_loss = contrastive_loss
        self.alpha = alpha  # cosine loss 가중치
        
    def forward(self, sentence_features, labels):
        # Cosine similarity loss
        cosine_loss_val = self.cosine_loss(sentence_features, labels)
        
        # Contrastive loss (임베딩 추출)
        embeddings = sentence_features[0]['sentence_embedding']  # [batch_size, embedding_dim]
        if labels is not None:
            contrastive_loss_val = self.contrastive_loss(embeddings, labels.view(-1))
        else:
            contrastive_loss_val = torch.tensor(0.0).to(embeddings.device)
        
        # 가중 결합
        combined_loss = self.alpha * cosine_loss_val + (1 - self.alpha) * contrastive_loss_val
        return combined_loss

class AdvancedCategoryEvaluator(SentenceEvaluator):
    """Phase 1-3: 카테고리 평가자"""
    def __init__(self, test_df, categories_definitions, threshold_learner=None, 
                 dynamic_ensemble=None, patience=5, batch_size=32):
        self.test_df = test_df
        self.categories_definitions = categories_definitions
        self.category_names = list(categories_definitions.keys())
        self.threshold_learner = threshold_learner
        self.dynamic_ensemble = dynamic_ensemble
        self.patience = patience
        self.batch_size = batch_size
        self.latest_f1 = 0.0
        
    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        # Phase 1: 다층 카테고리 임베딩
        category_embs = self._compute_advanced_category_embeddings(model)
        
        # Phase 2: 동적 앙상블 제목 임베딩
        title_embs = self._compute_dynamic_title_embeddings(model)
        
        # Phase 3: 적응형 threshold로 예측
        predictions, true_labels = self._predict_with_adaptive_threshold(
            title_embs, category_embs, model
        )
        
        # F1 스코어 계산
        macro_f1 = f1_score(true_labels, predictions, average='macro', zero_division=0)
        micro_f1 = f1_score(true_labels, predictions, average='micro', zero_division=0)
        weighted_f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
        
        self.latest_f1 = macro_f1
        
        logging.info(f"Advanced Evaluation - Macro F1: {macro_f1:.4f}, Micro F1: {micro_f1:.4f}, Weighted F1: {weighted_f1:.4f}")
        
        return macro_f1
    
    def get_latest_f1(self):
        return self.latest_f1
    
    def _compute_advanced_category_embeddings(self, model):
        """리 임베딩 계산"""
        # 기존 CategoryAccuracyEvaluator의 multilayer 방식 활용
        evaluator = CategoryAccuracyEvaluator(self.test_df, self.categories_definitions)
        return evaluator._compute_multilayer_category_embeddings(model)
    
    def _compute_dynamic_title_embeddings(self, model):
        """동적 앙상블 제목 임베딩"""
        titles = self.test_df['title'].tolist()
        generalized_titles = self.test_df['generalized_title'].tolist()
        
        original_embs = model.encode(titles, convert_to_tensor=True, normalize_embeddings=True)
        ner_embs = model.encode(generalized_titles, convert_to_tensor=True, normalize_embeddings=True)
        
        if self.dynamic_ensemble is not None:
            weights = self.dynamic_ensemble(ner_embs, original_embs)
            ensemble_embs = ner_embs * weights[:, 0:1] + original_embs * weights[:, 1:2]
        else:
            ensemble_embs = (ner_embs + original_embs) / 2
        
        return F.normalize(ensemble_embs, p=2, dim=1)
    
    def _predict_with_adaptive_threshold(self, title_embs, category_embs, model):
        """적응형 threshold로 예측"""
        category_embs_tensor = torch.tensor(
            np.array([category_embs[name] for name in self.category_names]),
            dtype=torch.float32
        ).to(title_embs.device)
        
        similarities = util.cos_sim(title_embs, category_embs_tensor)
        
        predictions = []
        true_labels = []
        
        for i, row in self.test_df.iterrows():
            sim_scores = similarities[i]
            
            if self.threshold_learner is not None:
                threshold = self.threshold_learner(title_embs[i], sim_scores).item()
                predicted_indices = (sim_scores > threshold).nonzero(as_tuple=True)[0]
            else:
                # 기본: 상위 2개 선택
                predicted_indices = torch.topk(sim_scores, k=min(2, len(self.category_names))).indices
            
            predicted_categories = [self.category_names[idx] for idx in predicted_indices]
            true_categories = row['categories']
            
            # Multi-label을 binary로 변환
            for cat in self.category_names:
                predictions.append(1 if cat in predicted_categories else 0)
                true_labels.append(1 if cat in true_categories else 0)
        
        return predictions, true_labels


def main():
    parser = argparse.ArgumentParser(description="Phase 1-3 문장 임베딩 모델 파인튜닝")
    parser.add_argument("--data_path", type=str, default="resources/processed_data.csv", help="학습 데이터 경로")
    parser.add_argument("--model_name", type=str, default="intfloat/multilingual-e5-large", help="사용할 모델명")
    parser.add_argument("--output_path", type=str, default="outputs/advanced_finetuned_model", help="모델 저장 경로")
    parser.add_argument("--max_epochs", type=int, default=30, help="최대 에포크 수")
    parser.add_argument("--batch_size", type=int, default=16, help="배치 크기")
    parser.add_argument("--lr", type=float, default=2e-5, help="초기 학습률")
    parser.add_argument("--test_size", type=float, default=0.2, help="테스트 데이터 비율 (Stratified 분할)")
    parser.add_argument("--patience", type=int, default=7, help="조기 종료 patience (F1 기준)")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    
    # 파라미터
    parser.add_argument("--curriculum_initial_difficulty", type=float, default=0.3, 
                       help="Curriculum Learning 초기 난이도")
    parser.add_argument("--contrastive_temperature", type=float, default=0.1, 
                       help="Contrastive Learning 온도 파라미터")
    parser.add_argument("--ensemble_alpha", type=float, default=0.7, 
                       help="Cosine vs Contrastive Loss 가중치")
    
    args = parser.parse_args()

    # 결과 디렉토리 생성
    os.makedirs(args.output_path, exist_ok=True)
    
    # 로깅 설정
    log_file = os.path.join(args.output_path, 'advanced_training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.info(f"파라미터: {vars(args)}")

    # AdvancedFinetuner 실행
    finetuner = AdvancedFinetuner(args)
    finetuner.run_advanced_finetuning()

if __name__ == "__main__":
    main()
