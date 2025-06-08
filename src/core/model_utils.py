import logging
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from tqdm import tqdm
from src.config import NER_MODEL_NAME, NER_SPECIAL_TOKENS, GENERALIZED_ENTITIES, NER_ENTITY_PLACEHOLDERS
import os
import sys

# 프로젝트 루트 경로 추가
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(PROJECT_ROOT)

# NER 파이프라인은 무거우므로, 한 번만 로드하고 재사용하기 위한 싱글톤 패턴
_ner_pipeline = None

def get_ner_pipeline():
    """싱글턴 패턴으로 NER 파이프라인을 생성하고 관리합니다."""
    global _ner_pipeline
    if _ner_pipeline is None:
        try:
            print("NER 파이프라인 로딩: " + NER_MODEL_NAME)
            tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_NAME)
            model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_NAME)
            
            # config에 정의된 NER 관련 특수 토큰 추가
            if NER_SPECIAL_TOKENS:
                tokenizer.add_special_tokens({'additional_special_tokens': NER_SPECIAL_TOKENS})
                model.resize_token_embeddings(len(tokenizer))

            _ner_pipeline = pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple"
            )
            print("NER 파이프라인 로딩 완료.")
        except Exception as e:
            print(f"NER 파이프라인 로딩 중 오류 발생: {e}")
            _ner_pipeline = None
            
    return _ner_pipeline

def ner_generalize_texts(texts: list[str]) -> list[str]:
    """
    주어진 텍스트 리스트에 대해 NER을 수행하고, GENERALIZED_ENTITIES에 포함된
    타입의 엔티티를 미리 정의된 플레이스홀더로 변환합니다.
    """
    ner_pipeline = get_ner_pipeline()
    if not ner_pipeline:
        print("NER 파이프라인을 사용할 수 없어 일반화를 건너뜁니다.")
        return texts

    generalized_texts = []
    
    for text in tqdm(texts, desc="NER 일반화 진행 중"):
        try:
            ner_results = ner_pipeline(text)
            
            # 높은 점수의 엔티티만 사용하되, GENERALIZED_ENTITIES에 포함된 것만 필터링
            entities_to_replace = sorted([
                res for res in ner_results
                if res['score'] > 0.9 and res['entity_group'] in GENERALIZED_ENTITIES
            ], key=lambda x: x['start'], reverse=True)

            generalized_text = text
            for entity in entities_to_replace:
                entity_type = entity['entity_group']
                
                # config에 정의된 플레이스홀더 사용
                placeholder = NER_ENTITY_PLACEHOLDERS[entity_type]
                start, end = entity['start'], entity['end']
                generalized_text = generalized_text[:start] + placeholder + generalized_text[end:]
                
            generalized_texts.append(generalized_text)
        except Exception as e:
            logging.error(f"'{text}' 일반화 중 오류 발생: {e}")
            generalized_texts.append(text)  # 오류 발생 시 원본 텍스트 추가
        
    return generalized_texts 