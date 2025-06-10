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
    BASE_MODEL_NAME, DATA_PATH, DEFAULT_OUTPUT_PATH, CATEGORIES_DEFINITIONS, 
    V2_IMPROVED_ENTITIES, NER_CONFIDENCE_THRESHOLD
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
        
        # V2 개선 버전들을 위한 일반화된 텍스트들
        self.test_df['v2_improved_title'] = ner_generalize_texts(
            self.test_df['title'].tolist(), 
            entities_to_generalize=V2_IMPROVED_ENTITIES,
            confidence_threshold=NER_CONFIDENCE_THRESHOLD
        )
        self.test_df['v2_conservative_title'] = ner_generalize_texts(
            self.test_df['title'].tolist(), 
            entities_to_generalize=["PS"],  # 인물만 일반화
            confidence_threshold=0.7
        )
        
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
            category_embs[category] = avg_emb.astype(np.float32)
        return torch.tensor(np.array(list(category_embs.values())), dtype=torch.float32).to(model.device)

    def _get_keyword_max_embs(self, model):
        """주어진 모델로 카테고리별 키워드 가중평균 임베딩 계산 (키워드별 TF-IDF 가중치 적용)"""
        category_embs = []
        for category in self.categories:
            keywords = CATEGORIES_DEFINITIONS[category]
            if not keywords:
                # 키워드가 없는 경우 0벡터 사용
                category_embs.append(np.zeros(model.get_sentence_embedding_dimension(), dtype=np.float32))
                continue
                
            keyword_embs = model.encode(keywords, convert_to_numpy=True, normalize_embeddings=True)
            
            # 키워드 길이 기반 가중치 (긴 키워드일수록 더 구체적이므로 높은 가중치)
            weights = np.array([len(kw.split()) for kw in keywords], dtype=np.float32)
            weights = weights / np.sum(weights)  # 정규화
            
            # 가중 평균 계산
            weighted_emb = np.average(keyword_embs, axis=0, weights=weights)
            if np.linalg.norm(weighted_emb) > 0:
                weighted_emb = weighted_emb / np.linalg.norm(weighted_emb)
            category_embs.append(weighted_emb.astype(np.float32))
        return torch.tensor(np.array(category_embs), dtype=torch.float32).to(model.device)

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

    def _evaluate_simple_ensemble(self, version_name, text_column, embedding_configs, weight=0.5):
        """단순 앙상블 평가 (2개 구성요소, 0.5:0.5 가중치)"""
        logging.info(f"--- {version_name} 앙상블 평가 시작 ---")
        
        all_similarities = []
        
        for config in embedding_configs:
            model = config['model']
            category_embs = config['category_embs']
            
            text_embs = model.encode(
                self.test_df[text_column].tolist(), 
                convert_to_tensor=True, 
                normalize_embeddings=True
            )
            similarities = util.cos_sim(text_embs, category_embs).cpu().numpy()
            all_similarities.append(similarities)
        
        # 0.5:0.5 가중 평균 앙상블
        ensemble_similarities = (all_similarities[0] * weight + all_similarities[1] * weight)
        
        # 상위 K개 예측 추출
        top_k_preds = np.argsort(ensemble_similarities, axis=1)[:, ::-1]
        
        true_categories = self.test_df['category'].tolist()
        true_indices = [self.categories.index(cat) for cat in true_categories]
        
        # Hit Rate 계산
        correct_at_1 = sum(1 for i, true_idx in enumerate(true_indices) if true_idx == top_k_preds[i, 0])
        correct_at_3 = sum(1 for i, true_idx in enumerate(true_indices) if true_idx in top_k_preds[i, :3])
        correct_at_5 = sum(1 for i, true_idx in enumerate(true_indices) if true_idx in top_k_preds[i, :5])

        total_count = len(self.test_df)
        
        # F1-score 계산
        top_1_predictions = [self.categories[idx] for idx in top_k_preds[:, 0]]
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_categories, top_1_predictions, average='macro', zero_division=0, labels=self.categories
        )
        
        metrics = {
            "Hit Rate @1": correct_at_1 / total_count,
            "Hit Rate @3": correct_at_3 / total_count,
            "Hit Rate @5": correct_at_5 / total_count,
            "F1 (macro)": f1,
        }
        
        logging.info(f"{version_name} 앙상블 평가 완료: Hit@1={metrics['Hit Rate @1']:.2%}, Hit@3={metrics['Hit Rate @3']:.2%}, Hit@5={metrics['Hit Rate @5']:.2%}, F1={f1:.4f}")
        return metrics

    def _ensemble_predict(self, ensemble_configs, text_column='generalized_title'):
        """앙상블 예측 수행"""
        all_similarities = []
        weights = []
        
        for config in ensemble_configs:
            model_name = config['model']
            embedding_type = config['embedding']
            weight = config['weight']
            
            # 모델 선택
            if model_name == 'base':
                model = self.base_model
            elif model_name == 'finetuned' and self.finetuned_model:
                model = self.finetuned_model
            else:
                continue
                
            # 임베딩 선택
            if embedding_type == 'simple':
                category_embs = self.simple_cat_embs
            elif embedding_type == 'keyword_avg':
                if model_name == 'base':
                    category_embs = self.v3_cat_embs
                else:
                    category_embs = self.v4_cat_embs
            elif embedding_type == 'keyword_max':
                # 키워드별 최대 유사도 계산
                text_embs = model.encode(self.test_df[text_column].tolist(), convert_to_tensor=True, normalize_embeddings=True)
                similarities = self._calculate_keyword_max_similarities(text_embs, model)
                all_similarities.append(similarities)
                weights.append(weight)
                continue
            else:
                continue
                
            # 일반 유사도 계산
            text_embs = model.encode(self.test_df[text_column].tolist(), convert_to_tensor=True, normalize_embeddings=True)
            similarities = util.cos_sim(text_embs, category_embs).cpu().numpy()
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

    def _calculate_keyword_max_similarities(self, text_embs, model):
        """키워드별 최대 유사도 계산"""
        similarities = []
        
        for text_emb in text_embs:
            text_similarities = []
            for category in self.categories:
                keywords = CATEGORIES_DEFINITIONS[category]
                keyword_embs = model.encode(keywords, convert_to_tensor=True, normalize_embeddings=True)
                # 각 키워드와의 유사도 중 최대값
                max_sim = torch.max(util.cos_sim(text_emb.unsqueeze(0), keyword_embs)).item()
                text_similarities.append(max_sim)
            similarities.append(text_similarities)
        
        return np.array(similarities)

    def _evaluate_ensemble(self, ensemble_configs, version_name, text_column='generalized_title'):
        """앙상블 버전 평가"""
        logging.info(f"--- {version_name} 앙상블 평가 시작 ---")
        
        similarities = self._ensemble_predict(ensemble_configs, text_column)
        if similarities is None:
            return {"Hit Rate @1": 0, "Hit Rate @3": 0, "Hit Rate @5": 0, "F1 (macro)": 0}
        
        # 상위 K개 예측 추출
        top_k_preds = np.argsort(similarities, axis=1)[:, ::-1]
        
        true_categories = self.test_df['category'].tolist()
        true_indices = [self.categories.index(cat) for cat in true_categories]
        
        # Hit Rate 계산
        correct_at_1 = sum(1 for i, true_idx in enumerate(true_indices) if true_idx == top_k_preds[i, 0])
        correct_at_3 = sum(1 for i, true_idx in enumerate(true_indices) if true_idx in top_k_preds[i, :3])
        correct_at_5 = sum(1 for i, true_idx in enumerate(true_indices) if true_idx in top_k_preds[i, :5])

        total_count = len(self.test_df)
        
        # F1-score 계산
        top_1_predictions = [self.categories[idx] for idx in top_k_preds[:, 0]]
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_categories, top_1_predictions, average='macro', zero_division=0, labels=self.categories
        )
        
        metrics = {
            "Hit Rate @1": correct_at_1 / total_count,
            "Hit Rate @3": correct_at_3 / total_count,
            "Hit Rate @5": correct_at_5 / total_count,
            "F1 (macro)": f1,
        }
        
        logging.info(f"{version_name} 앙상블 평가 완료: Hit@1={metrics['Hit Rate @1']:.2%}")
        return metrics

    def optimize_ensemble_hyperparameters(self):
        """앙상블 하이퍼파라미터 최적화"""
        logging.info("앙상블 하이퍼파라미터 최적화 시작...")
        
        # 가능한 앙상블 조합들
        ensemble_combinations = [
            # V3 기반 앙상블들
            [
                {'model': 'base', 'embedding': 'simple', 'weight': 0.3},
                {'model': 'base', 'embedding': 'keyword_avg', 'weight': 0.7}
            ],
            [
                {'model': 'base', 'embedding': 'simple', 'weight': 0.4},
                {'model': 'base', 'embedding': 'keyword_avg', 'weight': 0.6}
            ],
            [
                {'model': 'base', 'embedding': 'keyword_avg', 'weight': 0.6},
                {'model': 'base', 'embedding': 'keyword_max', 'weight': 0.4}
            ],
            [
                {'model': 'base', 'embedding': 'simple', 'weight': 0.2},
                {'model': 'base', 'embedding': 'keyword_avg', 'weight': 0.6},
                {'model': 'base', 'embedding': 'keyword_max', 'weight': 0.2}
            ],
        ]
        
        # V4가 있으면 파인튜닝 모델 조합도 추가
        if self.finetuned_model:
            v4_combinations = [
                [
                    {'model': 'finetuned', 'embedding': 'simple', 'weight': 0.3},
                    {'model': 'finetuned', 'embedding': 'keyword_avg', 'weight': 0.7}
                ],
                [
                    {'model': 'base', 'embedding': 'keyword_avg', 'weight': 0.4},
                    {'model': 'finetuned', 'embedding': 'keyword_avg', 'weight': 0.6}
                ],
                [
                    {'model': 'base', 'embedding': 'simple', 'weight': 0.2},
                    {'model': 'base', 'embedding': 'keyword_avg', 'weight': 0.3},
                    {'model': 'finetuned', 'embedding': 'keyword_avg', 'weight': 0.5}
                ],
            ]
            ensemble_combinations.extend(v4_combinations)
        
        # 텍스트 컬럼 후보들
        text_columns = ['title', 'generalized_title', 'v2_improved_title']
        
        best_results = {}
        all_results = []
        
        for text_col in text_columns:
            for i, ensemble_config in enumerate(ensemble_combinations):
                config_name = f"ensemble_{i+1}_{text_col}"
                result = self._evaluate_ensemble(ensemble_config, config_name, text_col)
                result['config'] = ensemble_config
                result['text_column'] = text_col
                result['config_name'] = config_name
                all_results.append(result)
        
        # 각 메트릭별 최고 성능 찾기
        for metric in ["Hit Rate @1", "Hit Rate @3", "Hit Rate @5"]:
            best_result = max(all_results, key=lambda x: x[metric])
            best_results[metric] = best_result
            
        return best_results, all_results

    def run_all_and_compare(self):
        """모든 알고리즘 버전을 실행하고 결과를 비교하여 출력."""
        results = {}
        
        # V1: 단순 임베딩 (NER 없음 + 단순 카테고리명)
        results['V1 (단순 임베딩)'] = self._evaluate_version(
            "V1", self.base_model, 'title', self.simple_cat_embs
        )
        
        # V2: NER + 앙상블 (단순 카테고리명 + 키워드 평균 임베딩을 0.5:0.5로 앙상블)
        v2_configs = [
            {'model': self.base_model, 'category_embs': self.simple_cat_embs},
            {'model': self.base_model, 'category_embs': self.v3_cat_embs}
        ]
        results['V2 (NER + 앙상블)'] = self._evaluate_simple_ensemble(
            "V2", 'generalized_title', v2_configs
        )
        
        # V3: NER + 카테고리 키워드화 + 앙상블 (키워드 평균 + 키워드 최대값을 0.5:0.5로 앙상블)
        v3_configs = [
            {'model': self.base_model, 'category_embs': self.v3_cat_embs},
            {'model': self.base_model, 'category_embs': self._get_keyword_max_embs(self.base_model)}
        ]
        results['V3 (NER + 키워드화 + 앙상블)'] = self._evaluate_simple_ensemble(
            "V3", 'generalized_title', v3_configs
        )
        
        # V4: 파인튜닝 + 앙상블 (파인튜닝된 모델의 단순 카테고리명 + 키워드 평균을 0.5:0.5로 앙상블)
        if os.path.exists(os.path.join(PROJECT_ROOT, 'models/finetuned_ensemble_v4')):
            logging.info("파인튜닝된 앙상블 모델을 로드합니다...")
            try:
                finetuned_model = SentenceTransformer(os.path.join(PROJECT_ROOT, 'models/finetuned_ensemble_v4'))
                
                # 파인튜닝된 모델의 키워드 평균 임베딩
                finetuned_keyword_embs = self._get_keyword_avg_embs(finetuned_model)
                
                # 파인튜닝된 모델의 앙상블 구성 (단순 카테고리명 + 키워드 평균)
                v4_configs = [
                    {'model': finetuned_model, 'category_embs': torch.tensor(
                        finetuned_model.encode(self.categories, convert_to_numpy=True, normalize_embeddings=True)
                    ).to(finetuned_model.device)},
                    {'model': finetuned_model, 'category_embs': finetuned_keyword_embs}
                ]
                results['V4 (파인튜닝 + 앙상블)'] = self._evaluate_simple_ensemble(
                    "V4", 'generalized_title', v4_configs
                )
            except Exception as e:
                logging.warning(f"파인튜닝된 모델 로드 실패: {e}")
                results['V4 (파인튜닝 + 앙상블)'] = {'hit_rate@1': 0, 'hit_rate@3': 0, 'f1_score': 0}
        else:
            logging.warning("파인튜닝된 모델이 존재하지 않습니다. 먼저 finetune.py를 실행해주세요.")
            results['V4 (파인튜닝 + 앙상블)'] = {'hit_rate@1': 0, 'hit_rate@3': 0, 'f1_score': 0}

        # 최종 결과 출력
        print("\n" + "="*80)
        print(" " * 25 + "알고리즘 버전별 성능 비교 결과")
        print("="*80)
        
        headers = ["Algorithm Version", "Hit Rate @1", "Hit Rate @3", "Hit Rate @5", "F1 (macro)"]
        col_widths = [35, 15, 15, 15, 12]
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

    def _print_ensemble_optimization_results(self, best_ensemble_configs):
        """앙상블 최적화 결과 출력"""
        print("\n" + "="*80)
        print(" " * 20 + "🔧 앙상블 하이퍼파라미터 최적화 결과")
        print("="*80)
        
        for metric, config_info in best_ensemble_configs.items():
            print(f"\n📊 {metric} 최적 설정:")
            print(f"   성능: {config_info[metric]:.4f}")
            print(f"   텍스트 컬럼: {config_info['text_column']}")
            print(f"   앙상블 구성:")
            
            for i, component in enumerate(config_info['config']):
                print(f"     {i+1}. 모델: {component['model']} | 임베딩: {component['embedding']} | 가중치: {component['weight']}")
        
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
