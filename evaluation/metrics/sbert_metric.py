"""
evaluation/metrics/sbert_metric.py

Sentence-BERT 코사인 유사도 기반 semantic similarity 계산.
BERTScore 대체 모듈. 외부 API 불필요, 로컬 모델만 사용.

필요 패키지:
    pip install sentence-transformers
"""
from __future__ import annotations

from typing import Dict
import numpy as np


_encoder_cache: Dict[str, any] = {}


def _get_encoder(model_name: str):
    """모델을 캐싱해서 반복 로드를 방지한다."""
    if model_name not in _encoder_cache:
        from sentence_transformers import SentenceTransformer
        _encoder_cache[model_name] = SentenceTransformer(model_name)
    return _encoder_cache[model_name]


def compute_sbert_similarity(
    reference_text: str,
    candidate_text: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Dict[str, float]:
    """
    Sentence-BERT로 reference와 candidate 간 코사인 유사도를 계산한다.

    반환 형식 (bertscore 키와 호환되도록 동일 구조 유지):
    {
        "precision": float,  # 이 구현에서는 cosine_similarity와 동일
        "recall":    float,  # 이 구현에서는 cosine_similarity와 동일
        "f1":        float,  # cosine_similarity
        "cosine_similarity": float,
        "model_name": str,
    }

    기존 bertscore precision/recall/f1 키를 그대로 유지해
    _aggregate_metrics 등 하위 코드를 수정하지 않아도 된다.
    """
    if not reference_text or not candidate_text:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "cosine_similarity": 0.0,
            "model_name": model_name,
        }

    encoder = _get_encoder(model_name)

    ref_emb = encoder.encode(
        reference_text, convert_to_numpy=True, normalize_embeddings=True
    )
    cand_emb = encoder.encode(
        candidate_text, convert_to_numpy=True, normalize_embeddings=True
    )

    ref_emb = np.asarray(ref_emb).flatten()
    cand_emb = np.asarray(cand_emb).flatten()

    # normalize_embeddings=True 이므로 dot product = cosine similarity
    cosine_sim = float(np.dot(ref_emb, cand_emb))
    # 수치 안정성: [-1, 1] 클리핑
    cosine_sim = max(-1.0, min(1.0, cosine_sim))

    return {
        "precision": cosine_sim,
        "recall": cosine_sim,
        "f1": cosine_sim,
        "cosine_similarity": cosine_sim,
        "model_name": model_name,
    }
