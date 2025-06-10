import torch
import os
from sentence_transformers import SentenceTransformer

def get_optimal_device():
    """최적의 디바이스를 자동으로 선택합니다."""
    if torch.backends.mps.is_available():
        return "mps"  # Apple Silicon M1/M2
    elif torch.cuda.is_available():
        return "cuda"  # NVIDIA GPU
    else:
        return "cpu"

def optimize_model_for_device(model, device=None):
    """모델을 최적의 디바이스로 이동시킵니다."""
    if device is None:
        device = get_optimal_device()
    
    print(f"Using device: {device}")
    
    if hasattr(model, 'to'):
        model = model.to(device)
    
    # MPS 최적화 설정
    if device == "mps":
        # Metal Performance Shaders 최적화
        torch.backends.mps.allow_tf32 = True
        # 메모리 효율성을 위한 설정
        torch.mps.empty_cache()
    
    return model, device

def get_optimal_batch_size(device="mps"):
    """디바이스에 따른 최적 배치 크기를 반환합니다."""
    if device == "mps":
        return 64  # M2 칩에 최적화된 크기
    elif device == "cuda":
        return 128  # GPU용
    else:
        return 32   # CPU용

def optimize_sentence_transformer(model_name, device=None):
    """SentenceTransformer를 MPS 최적화하여 로드합니다."""
    if device is None:
        device = get_optimal_device()
    
    print(f"Loading SentenceTransformer with device: {device}")
    
    # 환경 변수 설정 (MPS 최적화)
    if device == "mps":
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    
    model = SentenceTransformer(model_name, device=device)
    
    # 추가 최적화 설정
    if device == "mps":
        # 메모리 효율성 설정
        torch.mps.set_per_process_memory_fraction(0.8)
    
    return model, device

def batch_encode_optimized(model, texts, device="mps", batch_size=None):
    """배치 크기를 최적화하여 텍스트를 인코딩합니다."""
    if batch_size is None:
        batch_size = get_optimal_batch_size(device)
    
    print(f"Encoding {len(texts)} texts with batch_size={batch_size} on {device}")
    
    # 더 큰 배치 크기로 처리
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    
    return embeddings 