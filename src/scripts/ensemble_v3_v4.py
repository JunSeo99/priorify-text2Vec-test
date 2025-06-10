import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging
import os
import argparse
import torch
import sys
from itertools import product

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(PROJECT_ROOT)
from src.config import (
    BASE_MODEL_NAME, DATA_PATH, DEFAULT_OUTPUT_PATH, CATEGORIES_DEFINITIONS
)
from src.core.model_utils import ner_generalize_texts

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EnsembleEvaluator:
    """
    V3, V4에 앙상블 기법을 적용하여 성능을 평가하는 클래스
    """
    def __init__(self, data_path, finetuned_model_path):
        self.data_path = data_path
        self.finetuned_model_path = finetuned_model_path
        
        # 모델 로드
        self.base_model = SentenceTransformer(BASE_MODEL_NAME)
        self.finetuned_model = self._load_finetuned_model()
        
        self.categories = list(CATEGORIES_DEFINITIONS.keys())
        self.test_df = self._load_and_prepare_data()
        
        # 다양한 텍스트 전처리 준비
        self._prepare_text_variants()
        
        # 카테고리 임베딩 방법들 준비
        self._prepare_category_embeddings()
        
        logging.info(f"앙상블 평가 준비 완료: 테스트 데이터 {len(self.test_df)}개")

    def _load_finetuned_model(self):
        """파인튜닝된 모델 로드"""
        if os.path.exists(self.finetuned_model_path):
            logging.info(f"파인튜닝 모델 로드: {self.finetuned_model_path}")
            return SentenceTransformer(self.finetuned_model_path)
        else:
            logging.warning(f"파인튜닝된 모델을 찾을 수 없음: {self.finetuned_model_path}")
            return None
            
    def _load_and_prepare_data(self):
        """데이터 로드 및 전처리"""
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
        
        return df.copy()
        
    def _prepare_text_variants(self):
        """다양한 텍스트 전처리 변형 준비"""
        logging.info("텍스트 변형 준비 중...")
        
        # 1. 원본 텍스트
        self.test_df['text_original'] = self.test_df['title']
        
        # 2. NER 적용 텍스트
        self.test_df['text_ner'] = ner_generalize_texts(self.test_df['title'].tolist())
        
        # 3. 보수적 NER (인물만)
        self.test_df['text_ner_conservative'] = ner_generalize_texts(
            self.test_df['title'].tolist(), 
            entities_to_generalize=["PS"],
            confidence_threshold=0.7
        )
        
    def _prepare_category_embeddings(self):
        """다양한 카테고리 임베딩 방법 준비"""
        logging.info("카테고리 임베딩 준비 중...")
        
        self.category_embeddings = {}
        
        # 방법 1: 단순 카테고리명 임베딩
        self.category_embeddings['simple_base'] = self.base_model.encode(
            self.categories, convert_to_numpy=True, normalize_embeddings=True
        )
        if self.finetuned_model:
            self.category_embeddings['simple_finetuned'] = self.finetuned_model.encode(
                self.categories, convert_to_numpy=True, normalize_embeddings=True
            )
        
        # 방법 2: 키워드 평균 임베딩
        self.category_embeddings['keyword_avg_base'] = self._get_keyword_avg_embs(self.base_model)
        if self.finetuned_model:
            self.category_embeddings['keyword_avg_finetuned'] = self._get_keyword_avg_embs(self.finetuned_model)
            
        # 방법 3: 키워드 개별 임베딩 (최대 유사도용)
        self.category_keyword_embs_base = self._get_keyword_individual_embs(self.base_model)
        if self.finetuned_model:
            self.category_keyword_embs_finetuned = self._get_keyword_individual_embs(self.finetuned_model)

    def _get_keyword_avg_embs(self, model):
        """키워드 평균 임베딩 계산"""
        category_embs = []
        for category in self.categories:
            keywords = CATEGORIES_DEFINITIONS[category]
            keyword_embs = model.encode(keywords, convert_to_numpy=True, normalize_embeddings=True)
            avg_emb = np.mean(keyword_embs, axis=0)
            if np.linalg.norm(avg_emb) > 0:
                avg_emb = avg_emb / np.linalg.norm(avg_emb)
            category_embs.append(avg_emb)
        return np.array(category_embs)
    
    def _get_keyword_individual_embs(self, model):
        """키워드별 개별 임베딩 계산"""
        category_keyword_embs = {}
        for category in self.categories:
            keywords = CATEGORIES_DEFINITIONS[category]
            keyword_embs = model.encode(keywords, convert_to_numpy=True, normalize_embeddings=True)
            category_keyword_embs[category] = keyword_embs
        return category_keyword_embs

    def _calculate_similarities(self, text_column, model, embedding_method):
        """유사도 계산"""
        # 텍스트 임베딩
        text_embs = model.encode(
            self.test_df[text_column].tolist(), 
            convert_to_numpy=True, 
            normalize_embeddings=True
        )
        
        if embedding_method.startswith('keyword_max'):
            # 키워드별 최대 유사도 계산
            model_suffix = embedding_method.split('_')[-1]
            if model_suffix == 'base':
                category_keyword_embs = self.category_keyword_embs_base
            else:
                category_keyword_embs = self.category_keyword_embs_finetuned
                
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
        else:
            # 평균 임베딩과 유사도 계산
            category_embs = self.category_embeddings[embedding_method]
            return util.cos_sim(text_embs, torch.tensor(category_embs)).cpu().numpy()

    def _ensemble_predict(self, ensemble_config):
        """앙상블 예측 수행"""
        all_similarities = []
        weights = []
        
        for component in ensemble_config:
            text_column = component['text_column']
            model_name = component['model']
            embedding_method = component['embedding_method']
            weight = component['weight']
            
            # 모델 선택
            if model_name == 'base':
                model = self.base_model
            elif model_name == 'finetuned' and self.finetuned_model:
                model = self.finetuned_model
            else:
                continue
                
            # 유사도 계산
            similarities = self._calculate_similarities(text_column, model, embedding_method)
            all_similarities.append(similarities)
            weights.append(weight)
        
        if not all_similarities:
            return None
        
        # 가중 평균으로 앙상블
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # 정규화
        
        ensemble_similarities = np.zeros_like(all_similarities[0])
        for sim, weight in zip(all_similarities, weights):
            ensemble_similarities += sim * weight
        
        return ensemble_similarities

    def _evaluate_predictions(self, similarities):
        """예측 결과 평가"""
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

    def optimize_ensemble_hyperparameters(self):
        """앙상블 하이퍼파라미터 최적화"""
        logging.info("앙상블 하이퍼파라미터 최적화 시작...")
        
        # 최적화할 하이퍼파라미터들
        text_columns = ['text_original', 'text_ner', 'text_ner_conservative']
        embedding_methods = ['simple_base', 'keyword_avg_base', 'keyword_max_base']
        
        if self.finetuned_model:
            embedding_methods.extend(['simple_finetuned', 'keyword_avg_finetuned', 'keyword_max_finetuned'])
        
        # 가중치 조합들
        weight_combinations = [
            [1.0],  # 단일 방법
            [0.7, 0.3],  # 두 방법 조합
            [0.5, 0.5],
            [0.3, 0.7],
            [0.6, 0.25, 0.15],  # 세 방법 조합
            [0.5, 0.3, 0.2],
            [0.4, 0.4, 0.2],
        ]
        
        best_configs = []
        all_results = []
        
        # 단일 방법 테스트
        for text_col in text_columns:
            for emb_method in embedding_methods:
                config = [{
                    'text_column': text_col,
                    'model': emb_method.split('_')[-1],
                    'embedding_method': emb_method,
                    'weight': 1.0
                }]
                
                similarities = self._ensemble_predict(config)
                result = self._evaluate_predictions(similarities)
                
                if result:
                    result['config'] = config
                    result['config_name'] = f"{text_col}_{emb_method}"
                    all_results.append(result)
        
        # 앙상블 조합 테스트 (상위 성능 방법들로만)
        # Hit Rate @1 기준 상위 5개 방법 선택
        top_single_methods = sorted(all_results, key=lambda x: x["Hit Rate @1"], reverse=True)[:5]
        
        logging.info(f"상위 5개 단일 방법으로 앙상블 조합 테스트...")
        
        for i, method1 in enumerate(top_single_methods):
            for j, method2 in enumerate(top_single_methods[i+1:], i+1):
                for weights in weight_combinations[1:3]:  # 두 방법 조합만
                    config = [
                        {**method1['config'][0], 'weight': weights[0]},
                        {**method2['config'][0], 'weight': weights[1]}
                    ]
                    
                    similarities = self._ensemble_predict(config)
                    result = self._evaluate_predictions(similarities)
                    
                    if result:
                        result['config'] = config
                        result['config_name'] = f"ensemble_{method1['config_name']}_{method2['config_name']}"
                        all_results.append(result)
        
        # 최고 성능 설정들 찾기
        best_hit_1 = max(all_results, key=lambda x: x["Hit Rate @1"])
        best_hit_3 = max(all_results, key=lambda x: x["Hit Rate @3"])
        best_hit_5 = max(all_results, key=lambda x: x["Hit Rate @5"])
        
        return {
            "Hit Rate @1": best_hit_1,
            "Hit Rate @3": best_hit_3,
            "Hit Rate @5": best_hit_5
        }, all_results

    def evaluate_v3_v4_ensemble(self):
        """V3, V4에 최적 앙상블 적용하여 평가"""
        logging.info("V3, V4 앙상블 최적화 및 평가 시작...")
        
        # 하이퍼파라미터 최적화
        best_configs, all_results = self.optimize_ensemble_hyperparameters()
        
        # V3 앙상블 (Base 모델 + 키워드 기반)
        v3_results = {}
        best_config = best_configs["Hit Rate @1"]['config']
        
        # V3: Base 모델로만 구성된 앙상블
        v3_config = []
        for component in best_config:
            if component['model'] == 'base':
                v3_config.append(component)
        
        if v3_config:
            similarities = self._ensemble_predict(v3_config)
            v3_result = self._evaluate_predictions(similarities)
            if v3_result:
                v3_results['V3 (Base+앙상블)'] = v3_result
        
        # V4 앙상블 (Finetuned 모델 포함)
        v4_results = {}
        if self.finetuned_model:
            # V4-1: Finetuned 모델로만 구성된 앙상블
            v4_config = []
            for component in best_config:
                if component['model'] == 'finetuned':
                    v4_config.append(component)
            
            if v4_config:
                similarities = self._ensemble_predict(v4_config)
                v4_result = self._evaluate_predictions(similarities)
                if v4_result:
                    v4_results['V4 (Finetuned+앙상블)'] = v4_result
            
            # V4-2: Base + Finetuned 혼합 앙상블
            mixed_similarities = self._ensemble_predict(best_config)
            mixed_result = self._evaluate_predictions(mixed_similarities)
            if mixed_result:
                v4_results['V4 (혼합+앙상블)'] = mixed_result
        
        return best_configs, v3_results, v4_results, all_results

    def run_evaluation(self):
        """전체 평가 프로세스 실행"""
        best_configs, v3_results, v4_results, all_results = self.evaluate_v3_v4_ensemble()
        
        # 결과 출력
        self._print_results(best_configs, v3_results, v4_results)
        
        return best_configs, v3_results, v4_results

    def _print_results(self, best_configs, v3_results, v4_results):
        """결과 출력"""
        print("\n" + "="*100)
        print(" " * 30 + "V3, V4 앙상블 기법 적용 결과")
        print("="*100)
        
        # 최적 하이퍼파라미터 출력
        print("\n📊 최적 앙상블 하이퍼파라미터:")
        for metric, config_info in best_configs.items():
            print(f"\n{metric} 최적 설정:")
            print(f"  - 성능: {config_info[metric]:.4f}")
            print(f"  - 설정: {config_info['config_name']}")
            
            for i, component in enumerate(config_info['config']):
                print(f"  - 컴포넌트 {i+1}: {component['text_column']} + {component['embedding_method']} (가중치: {component['weight']})")
        
        # V3 결과
        if v3_results:
            print(f"\n🚀 V3 앙상블 결과:")
            headers = ["Version", "Hit Rate @1", "Hit Rate @3", "Hit Rate @5"]
            col_widths = [25, 15, 15, 15]
            header_line = " | ".join([f"{h:<{w}}" for h, w in zip(headers, col_widths)])
            print(header_line)
            print("-" * len(header_line))
            
            for version, scores in v3_results.items():
                row_data = [version]
                for header in headers[1:]:
                    score = scores.get(header, 0)
                    row_data.append(f"{score:.2%}")
                
                row_line = " | ".join([f"{d:<{w}}" for d, w in zip(row_data, col_widths)])
                print(row_line)
        
        # V4 결과
        if v4_results:
            print(f"\n🔥 V4 앙상블 결과:")
            headers = ["Version", "Hit Rate @1", "Hit Rate @3", "Hit Rate @5"]
            col_widths = [25, 15, 15, 15]
            header_line = " | ".join([f"{h:<{w}}" for h, w in zip(headers, col_widths)])
            print(header_line)
            print("-" * len(header_line))
            
            for version, scores in v4_results.items():
                row_data = [version]
                for header in headers[1:]:
                    score = scores.get(header, 0)
                    row_data.append(f"{score:.2%}")
                
                row_line = " | ".join([f"{d:<{w}}" for d, w in zip(row_data, col_widths)])
                print(row_line)
        
        print("="*100)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V3, V4에 앙상블 기법을 적용하여 성능 평가")
    parser.add_argument("--data_path", type=str, default=DATA_PATH, help="데이터 CSV 파일 경로")
    parser.add_argument("--finetuned_model_path", type=str, default=DEFAULT_OUTPUT_PATH, help="파인튜닝된 모델의 경로")
    
    args = parser.parse_args()
    
    if not os.path.isabs(args.finetuned_model_path):
        args.finetuned_model_path = os.path.join(PROJECT_ROOT, args.finetuned_model_path)
    if not os.path.isabs(args.data_path):
        args.data_path = os.path.join(PROJECT_ROOT, args.data_path)

    evaluator = EnsembleEvaluator(
        data_path=args.data_path,
        finetuned_model_path=args.finetuned_model_path
    )
    
    best_configs, v3_results, v4_results = evaluator.run_evaluation() 