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
from itertools import product
import json
from collections import defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(PROJECT_ROOT)
from src.config import (
    BASE_MODEL_NAME, DATA_PATH, DEFAULT_OUTPUT_PATH, CATEGORIES_DEFINITIONS, 
    V2_IMPROVED_ENTITIES, NER_CONFIDENCE_THRESHOLD
)
from src.core.model_utils import ner_generalize_texts

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EnsembleOptimizer:
    """
    앙상블 기법 하이퍼파라미터 최적화 및 성능 평가 클래스
    """
    def __init__(self, data_path, finetuned_model_path, test_size=0.2, seed=42):
        self.data_path = data_path
        self.finetuned_model_path = finetuned_model_path
        self.seed = seed
        
        # 모델 로드
        self.base_model = SentenceTransformer(BASE_MODEL_NAME)
        self.finetuned_model = self._load_finetuned_model()
        
        self.categories = list(CATEGORIES_DEFINITIONS.keys())
        
        # 데이터 준비 (train/test 분할)
        self.train_df, self.test_df = self._load_and_prepare_data(test_size)
        
        # 앙상블용 다양한 임베딩 계산
        self._prepare_embeddings()
        
        logging.info(f"앙상블 최적화 준비 완료: 학습={len(self.train_df)}, 테스트={len(self.test_df)}")

    def _load_finetuned_model(self):
        """파인튜닝된 모델 로드"""
        if os.path.exists(self.finetuned_model_path):
            logging.info(f"파인튜닝 모델 로드: {self.finetuned_model_path}")
            return SentenceTransformer(self.finetuned_model_path)
        else:
            logging.warning(f"파인튜닝된 모델을 찾을 수 없음: {self.finetuned_model_path}")
            return None
            
    def _load_and_prepare_data(self, test_size):
        """데이터 로드 및 train/test 분할"""
        df = pd.read_csv(self.data_path)
        
        # 'categories' 컬럼 처리
        if 'categories' not in df.columns and 'category' in df.columns:
            df = df.rename(columns={'category': 'categories'})
            
        df.dropna(subset=['title', 'categories'], inplace=True)
        df['category'] = df['categories'].apply(
            lambda x: x.split(';')[0].strip() if isinstance(x, str) else (x[0] if isinstance(x, list) and x else None)
        )
        df.dropna(subset=['category'], inplace=True)

        # config에 정의된 카테고리만 필터링
        df = df[df['category'].isin(self.categories)]
        
        # train/test 분할
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=self.seed, stratify=df['category'])
        
        logging.info(f"데이터 분할 완료: 학습 {len(train_df)}개, 테스트 {len(test_df)}개")
        return train_df.copy(), test_df.copy()
        
    def _prepare_embeddings(self):
        """앙상블용 다양한 임베딩 방법 준비"""
        logging.info("앙상블용 임베딩 준비 시작...")
        
        # 1. 텍스트 전처리 변형들
        self.test_df['title_original'] = self.test_df['title']
        self.test_df['title_ner'] = ner_generalize_texts(self.test_df['title'].tolist())
        self.test_df['title_ner_conservative'] = ner_generalize_texts(
            self.test_df['title'].tolist(), 
            entities_to_generalize=["PS"],
            confidence_threshold=0.7
        )
        
        # 2. 카테고리 임베딩 방법들
        self.category_embeddings = {}
        
        # 방법 1: 단순 카테고리명
        self.category_embeddings['simple'] = {
            'base': self.base_model.encode(self.categories, convert_to_numpy=True, normalize_embeddings=True)
        }
        if self.finetuned_model:
            self.category_embeddings['simple']['finetuned'] = self.finetuned_model.encode(
                self.categories, convert_to_numpy=True, normalize_embeddings=True
            )
        
        # 방법 2: 키워드 평균
        self.category_embeddings['keyword_avg'] = {
            'base': self._get_keyword_avg_embs(self.base_model)
        }
        if self.finetuned_model:
            self.category_embeddings['keyword_avg']['finetuned'] = self._get_keyword_avg_embs(self.finetuned_model)
            
        # 방법 3: 키워드 최대 유사도
        self.category_embeddings['keyword_max'] = {
            'base': self._get_keyword_embeddings(self.base_model)
        }
        if self.finetuned_model:
            self.category_embeddings['keyword_max']['finetuned'] = self._get_keyword_embeddings(self.finetuned_model)
        
        logging.info("앙상블용 임베딩 준비 완료")

    def _get_keyword_avg_embs(self, model):
        """키워드 평균 임베딩"""
        category_embs = []
        for category in self.categories:
            keywords = CATEGORIES_DEFINITIONS[category]
            keyword_embs = model.encode(keywords, convert_to_numpy=True, normalize_embeddings=True)
            avg_emb = np.mean(keyword_embs, axis=0)
            if np.linalg.norm(avg_emb) > 0:
                avg_emb = avg_emb / np.linalg.norm(avg_emb)
            category_embs.append(avg_emb)
        return np.array(category_embs)
    
    def _get_keyword_embeddings(self, model):
        """키워드별 개별 임베딩 (최대 유사도용)"""
        category_keyword_embs = {}
        for category in self.categories:
            keywords = CATEGORIES_DEFINITIONS[category]
            keyword_embs = model.encode(keywords, convert_to_numpy=True, normalize_embeddings=True)
            category_keyword_embs[category] = keyword_embs
        return category_keyword_embs

    def _calculate_similarity_scores(self, text_embs, method_name, model_name):
        """주어진 방법으로 유사도 점수 계산"""
        if method_name in ['simple', 'keyword_avg']:
            category_embs = self.category_embeddings[method_name][model_name]
            similarities = util.cos_sim(text_embs, torch.tensor(category_embs))
            return similarities.cpu().numpy()
        
        elif method_name == 'keyword_max':
            # 각 카테고리별로 키워드와의 최대 유사도 계산
            category_keyword_embs = self.category_embeddings[method_name][model_name]
            similarities = []
            
            for text_emb in text_embs:
                text_similarities = []
                for category in self.categories:
                    keyword_embs = category_keyword_embs[category]
                    # 각 키워드와의 유사도 중 최대값
                    max_sim = np.max(util.cos_sim([text_emb], torch.tensor(keyword_embs)).cpu().numpy())
                    text_similarities.append(max_sim)
                similarities.append(text_similarities)
            
            return np.array(similarities)

    def _ensemble_predict(self, text_column, ensemble_config):
        """앙상블 예측 수행"""
        text_embs = {}
        
        # 각 모델별로 텍스트 임베딩 계산
        for model_name in ensemble_config['models']:
            if model_name == 'base':
                model = self.base_model
            elif model_name == 'finetuned' and self.finetuned_model:
                model = self.finetuned_model
            else:
                continue
                
            text_embs[model_name] = model.encode(
                self.test_df[text_column].tolist(), 
                convert_to_numpy=True, 
                normalize_embeddings=True
            )
        
        # 각 방법별 유사도 점수 계산
        all_similarities = []
        weights = []
        
        for method_config in ensemble_config['methods']:
            method_name = method_config['name']
            method_weight = method_config['weight']
            
            for model_name in ensemble_config['models']:
                if model_name not in text_embs:
                    continue
                    
                if method_name not in self.category_embeddings or model_name not in self.category_embeddings[method_name]:
                    continue
                
                similarities = self._calculate_similarity_scores(
                    text_embs[model_name], method_name, model_name
                )
                all_similarities.append(similarities)
                weights.append(method_weight * ensemble_config['model_weights'].get(model_name, 1.0))
        
        if not all_similarities:
            return None
        
        # 가중 평균으로 앙상블
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # 정규화
        
        ensemble_similarities = np.zeros_like(all_similarities[0])
        for i, (sim, weight) in enumerate(zip(all_similarities, weights)):
            ensemble_similarities += sim * weight
        
        return ensemble_similarities

    def _evaluate_ensemble(self, text_column, ensemble_config):
        """앙상블 성능 평가"""
        similarities = self._ensemble_predict(text_column, ensemble_config)
        if similarities is None:
            return None
        
        # Top-k 예측
        top_k_indices = np.argsort(similarities, axis=1)[:, ::-1]
        
        true_categories = self.test_df['category'].tolist()
        true_indices = [self.categories.index(cat) for cat in true_categories]
        
        # Hit Rate 계산
        correct_at_1 = sum(1 for i, true_idx in enumerate(true_indices) if true_idx == top_k_indices[i, 0])
        correct_at_3 = sum(1 for i, true_idx in enumerate(true_indices) if true_idx in top_k_indices[i, :3])
        correct_at_5 = sum(1 for i, true_idx in enumerate(true_indices) if true_idx in top_k_indices[i, :5])

        total_count = len(self.test_df)
        
        return {
            "Hit Rate @1": correct_at_1 / total_count,
            "Hit Rate @3": correct_at_3 / total_count,
            "Hit Rate @5": correct_at_5 / total_count,
        }

    def optimize_hyperparameters(self):
        """하이퍼파라미터 그리드 서치로 최적화"""
        logging.info("앙상블 하이퍼파라미터 최적화 시작...")
        
        # 하이퍼파라미터 후보들
        text_columns = ['title_original', 'title_ner', 'title_ner_conservative']
        
        method_combinations = [
            # 단일 방법
            [{'name': 'simple', 'weight': 1.0}],
            [{'name': 'keyword_avg', 'weight': 1.0}],
            [{'name': 'keyword_max', 'weight': 1.0}],
            
            # 두 방법 조합
            [{'name': 'simple', 'weight': 0.3}, {'name': 'keyword_avg', 'weight': 0.7}],
            [{'name': 'simple', 'weight': 0.5}, {'name': 'keyword_avg', 'weight': 0.5}],
            [{'name': 'simple', 'weight': 0.7}, {'name': 'keyword_avg', 'weight': 0.3}],
            
            [{'name': 'keyword_avg', 'weight': 0.6}, {'name': 'keyword_max', 'weight': 0.4}],
            [{'name': 'keyword_avg', 'weight': 0.8}, {'name': 'keyword_max', 'weight': 0.2}],
            
            # 세 방법 조합
            [
                {'name': 'simple', 'weight': 0.2}, 
                {'name': 'keyword_avg', 'weight': 0.6}, 
                {'name': 'keyword_max', 'weight': 0.2}
            ],
            [
                {'name': 'simple', 'weight': 0.3}, 
                {'name': 'keyword_avg', 'weight': 0.5}, 
                {'name': 'keyword_max', 'weight': 0.2}
            ],
        ]
        
        model_configurations = [
            {'models': ['base'], 'model_weights': {'base': 1.0}},
        ]
        
        if self.finetuned_model:
            model_configurations.extend([
                {'models': ['finetuned'], 'model_weights': {'finetuned': 1.0}},
                {'models': ['base', 'finetuned'], 'model_weights': {'base': 0.3, 'finetuned': 0.7}},
                {'models': ['base', 'finetuned'], 'model_weights': {'base': 0.5, 'finetuned': 0.5}},
                {'models': ['base', 'finetuned'], 'model_weights': {'base': 0.7, 'finetuned': 0.3}},
            ])
        
        best_results = {}
        all_results = []
        
        total_combinations = len(text_columns) * len(method_combinations) * len(model_configurations)
        logging.info(f"총 {total_combinations}개 조합 테스트 시작")
        
        with tqdm(total=total_combinations, desc="하이퍼파라미터 최적화") as pbar:
            for text_col in text_columns:
                for methods in method_combinations:
                    for models_config in model_configurations:
                        ensemble_config = {
                            'methods': methods,
                            **models_config
                        }
                        
                        result = self._evaluate_ensemble(text_col, ensemble_config)
                        if result:
                            config_name = f"{text_col}_{len(methods)}methods_{'_'.join(ensemble_config['models'])}"
                            result['config'] = {
                                'text_column': text_col,
                                'ensemble_config': ensemble_config
                            }
                            result['config_name'] = config_name
                            all_results.append(result)
                        
                        pbar.update(1)
        
        # 각 메트릭별 최고 성능 찾기
        for metric in ["Hit Rate @1", "Hit Rate @3", "Hit Rate @5"]:
            best_result = max(all_results, key=lambda x: x[metric])
            best_results[metric] = best_result
            
        return best_results, all_results

    def apply_best_ensemble_to_versions(self, best_configs):
        """최적 앙상블을 V3, V4에 적용"""
        logging.info("최적 앙상블을 V3, V4에 적용 시작...")
        
        results = {}
        
        # Hit Rate @1 기준 최적 설정 사용
        best_config = best_configs["Hit Rate @1"]
        text_column = best_config['config']['text_column']
        ensemble_config = best_config['config']['ensemble_config']
        
        logging.info(f"최적 설정: {best_config['config_name']}")
        logging.info(f"텍스트 컬럼: {text_column}")
        logging.info(f"앙상블 설정: {ensemble_config}")
        
        # V3 앙상블 (base model + 키워드 기반)
        v3_config = {
            'methods': ensemble_config['methods'],
            'models': ['base'],
            'model_weights': {'base': 1.0}
        }
        v3_result = self._evaluate_ensemble(text_column, v3_config)
        if v3_result:
            results['V3 (NER+키워드+앙상블)'] = v3_result
        
        # V4 앙상블 (finetuned model + 키워드 기반)
        if self.finetuned_model:
            v4_config = {
                'methods': ensemble_config['methods'],
                'models': ['finetuned'],
                'model_weights': {'finetuned': 1.0}
            }
            v4_result = self._evaluate_ensemble(text_column, v4_config)
            if v4_result:
                results['V4 (NER+키워드+파인튜닝+앙상블)'] = v4_result
            
            # V4 혼합 앙상블 (base + finetuned)
            v4_mixed_result = self._evaluate_ensemble(text_column, ensemble_config)
            if v4_mixed_result:
                results['V4 (혼합모델+앙상블)'] = v4_mixed_result
        
        return results

    def run_optimization(self):
        """전체 최적화 프로세스 실행"""
        # 1. 하이퍼파라미터 최적화
        best_configs, all_results = self.optimize_hyperparameters()
        
        # 2. 최적 설정으로 V3, V4 평가
        ensemble_results = self.apply_best_ensemble_to_versions(best_configs)
        
        # 3. 결과 출력
        self._print_optimization_results(best_configs, ensemble_results)
        
        return best_configs, ensemble_results

    def _print_optimization_results(self, best_configs, ensemble_results):
        """최적화 결과 출력"""
        print("\n" + "="*100)
        print(" " * 35 + "앙상블 하이퍼파라미터 최적화 결과")
        print("="*100)
        
        # 최적 설정 출력
        for metric, config in best_configs.items():
            print(f"\n{metric} 최적 설정:")
            print(f"  - 성능: {config[metric]:.4f}")
            print(f"  - 텍스트 컬럼: {config['config']['text_column']}")
            print(f"  - 방법: {[m['name'] for m in config['config']['ensemble_config']['methods']]}")
            print(f"  - 가중치: {[m['weight'] for m in config['config']['ensemble_config']['methods']]}")
            print(f"  - 모델: {config['config']['ensemble_config']['models']}")
        
        # 앙상블 적용 결과
        print(f"\n{'='*50}")
        print("앙상블 기법 적용 결과:")
        print(f"{'='*50}")
        
        headers = ["Version", "Hit Rate @1", "Hit Rate @3", "Hit Rate @5"]
        col_widths = [35, 15, 15, 15]
        header_line = " | ".join([f"{h:<{w}}" for h, w in zip(headers, col_widths)])
        print(header_line)
        print("-" * len(header_line))
        
        for version, scores in ensemble_results.items():
            row_data = [version]
            for header in headers[1:]:
                score = scores.get(header, 0)
                row_data.append(f"{score:.2%}")
            
            row_line = " | ".join([f"{d:<{w}}" for d, w in zip(row_data, col_widths)])
            print(row_line)
        
        print("="*100)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="앙상블 기법 하이퍼파라미터 최적화 및 V3/V4 적용")
    parser.add_argument("--data_path", type=str, default=DATA_PATH, help="데이터 CSV 파일 경로")
    parser.add_argument("--finetuned_model_path", type=str, default=DEFAULT_OUTPUT_PATH, help="파인튜닝된 모델의 경로")
    parser.add_argument("--test_size", type=float, default=0.2, help="테스트 데이터 비율")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    
    args = parser.parse_args()
    
    if not os.path.isabs(args.finetuned_model_path):
        args.finetuned_model_path = os.path.join(PROJECT_ROOT, args.finetuned_model_path)
    if not os.path.isabs(args.data_path):
        args.data_path = os.path.join(PROJECT_ROOT, args.data_path)

    optimizer = EnsembleOptimizer(
        data_path=args.data_path,
        finetuned_model_path=args.finetuned_model_path,
        test_size=args.test_size,
        seed=args.seed
    )
    
    best_configs, ensemble_results = optimizer.run_optimization() 