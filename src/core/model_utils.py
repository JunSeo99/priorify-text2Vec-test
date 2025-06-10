import logging
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from tqdm import tqdm
from src.config import NER_MODEL_NAME, NER_SPECIAL_TOKENS, GENERALIZED_ENTITIES, NER_ENTITY_PLACEHOLDERS, NER_CONFIDENCE_THRESHOLD
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
                print(f"Adding {len(NER_SPECIAL_TOKENS)} special tokens: {NER_SPECIAL_TOKENS}")
                tokenizer.add_special_tokens({'additional_special_tokens': NER_SPECIAL_TOKENS})
                
                # 워닝 메시지 설명과 함께 모델 크기 조정
                old_vocab_size = model.config.vocab_size
                model.resize_token_embeddings(len(tokenizer), mean_resizing=True)
                new_vocab_size = len(tokenizer)
                print(f"Vocabulary expanded: {old_vocab_size} → {new_vocab_size} tokens")
                print("ℹ️  새 토큰들은 기존 임베딩의 통계를 바탕으로 초기화됩니다.")

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

def ner_generalize_texts(texts: list[str], entities_to_generalize=None, confidence_threshold=None) -> list[str]:
    """
    주어진 텍스트 리스트에 대해 NER을 수행하고, 지정된 엔티티 타입을
    미리 정의된 플레이스홀더로 변환합니다.
    
    Args:
        texts: 일반화할 텍스트 리스트
        entities_to_generalize: 일반화할 엔티티 타입 리스트 (None이면 기본값 사용)
        confidence_threshold: NER 신뢰도 임계값 (None이면 기본값 사용)
    """
    if entities_to_generalize is None:
        entities_to_generalize = GENERALIZED_ENTITIES
    if confidence_threshold is None:
        confidence_threshold = NER_CONFIDENCE_THRESHOLD
        
    ner_pipeline = get_ner_pipeline()
    if not ner_pipeline:
        print("NER 파이프라인을 사용할 수 없어 일반화를 건너뜁니다.")
        return texts

    generalized_texts = []
    
    for text in tqdm(texts, desc="NER 일반화 진행 중"):
        try:
            ner_results = ner_pipeline(text)
            
            # 설정된 임계값과 엔티티 타입 필터링
            entities_to_replace = sorted([
                res for res in ner_results
                if res['score'] > confidence_threshold and res['entity_group'] in entities_to_generalize
            ], key=lambda x: x['start'], reverse=True)

            generalized_text = text
            for entity in entities_to_replace:
                entity_type = entity['entity_group']
                
                # config에 정의된 플레이스홀더 사용
                if entity_type in NER_ENTITY_PLACEHOLDERS:
                    placeholder = NER_ENTITY_PLACEHOLDERS[entity_type]
                    start, end = entity['start'], entity['end']
                    generalized_text = generalized_text[:start] + placeholder + generalized_text[end:]
                
            generalized_texts.append(generalized_text)
        except Exception as e:
            logging.error(f"'{text}' 일반화 중 오류 발생: {e}")
            generalized_texts.append(text)  # 오류 발생 시 원본 텍스트 추가
        
    return generalized_texts 