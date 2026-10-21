"""
coverage_metric.py — NLI 기반 aspect coverage 평가.

각 aspect query가 answer에 의해 함의(entail)되는지를
DeBERTa-v3-large NLI 모델로 판단한다.

premise  = answer_text  (생성된 답변)
hypothesis = aspect     (각 aspect 문장)
entailment score >= threshold → hit

필요 패키지:
    pip install sentence-transformers
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np


# 전역 모델 캐시 — 반복 로드 방지
_nli_model_cache: Dict[str, Any] = {}

DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"

# 레이블 인덱스: contradiction=0, entailment=1, neutral=2
# MoritzLaurer 모델의 id2label: {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
# → 로드 후 실제 id2label을 확인해 entailment 인덱스를 동적으로 탐색한다


def _get_nli_model(model_name: str) -> Any:
    """NLI CrossEncoder 모델을 캐싱해서 반환한다."""
    if model_name not in _nli_model_cache:
        from sentence_transformers import CrossEncoder

        _nli_model_cache[model_name] = CrossEncoder(model_name)
    return _nli_model_cache[model_name]


def _entailment_index(model: Any) -> int:
    """
    모델의 id2label에서 'entailment' 레이블의 인덱스를 반환한다.
    모델마다 레이블 순서가 다르므로 동적으로 탐색한다.
    찾지 못하면 0을 반환한다 (MoritzLaurer 모델 기본값).
    """
    try:
        id2label = model.model.config.id2label
        for idx, label in id2label.items():
            if label.lower() == "entailment":
                return int(idx)
    except Exception:
        pass
    return 0


def _softmax(x: np.ndarray) -> np.ndarray:
    """Row-wise softmax over the last axis (numerically stable)."""
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def split_aspects(aspect_queries: List[str]) -> List[str]:
    """[SEP]로 이어진 단일 문자열이면 분리, 아니면 그대로 반환."""
    raw = [a.strip() for a in (aspect_queries or []) if a and a.strip()]
    # [SEP]로 이어진 하나의 긴 문자열인 경우 분리
    if len(raw) == 1 and "[SEP]" in raw[0]:
        parts = [p.strip() for p in raw[0].split("[SEP]")]
        return [p for p in parts if p]
    return raw


def compute_coverage(
    aspect_queries: List[str],
    answer_text: str,
    threshold: float = 0.5,
    nli_model_name: str = DEFAULT_NLI_MODEL,
) -> Dict[str, Any]:
    """
    NLI entailment score로 각 aspect가 answer에서 다뤄졌는지 판단한다.

    반환 구조는 기존 lexical baseline과 동일하게 유지한다:
    {
        "mode": "nli_entailment",
        "nli_model": str,
        "per_aspect_scores": List[float],   # entailment score (0~1)
        "per_aspect_hit": List[bool],        # score >= threshold
        "overall_coverage": float,           # hit 비율
    }
    """
    aspects = split_aspects(aspect_queries)

    if not aspects or not (answer_text or "").strip():
        return {
            "mode": "nli_entailment",
            "nli_model": nli_model_name,
            "per_aspect_scores": [],
            "per_aspect_hit": [],
            "overall_coverage": 0.0,
        }

    model = _get_nli_model(nli_model_name)
    ent_idx = _entailment_index(model)

    # 배치 predict: [(answer, aspect), ...]
    pairs = [(answer_text, aspect) for aspect in aspects]
    logits = model.predict(pairs, convert_to_numpy=True)  # shape: (N, num_labels)

    probs = _softmax(logits)  # (N, num_labels)
    ent_scores = probs[:, ent_idx].tolist()  # entailment 확률만 추출

    per_hits = [float(s) >= threshold for s in ent_scores]
    overall = sum(per_hits) / len(per_hits) if per_hits else 0.0

    return {
        "mode": "nli_entailment",
        "nli_model": nli_model_name,
        "per_aspect_scores": [round(s, 6) for s in ent_scores],
        "per_aspect_hit": per_hits,
        "overall_coverage": round(overall, 6),
    }


def compute_retrieval_coverage(
    aspect_queries: List[str],
    chunk_texts: List[str],
    threshold: float = 0.5,
    nli_model_name: str = DEFAULT_NLI_MODEL,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """Aspect coverage over the RETRIEVED CHUNKS (retrieval-level signal).

    For each aspect, take the MAX entailment over the individual chunks (does ANY
    retrieved chunk entail the aspect?); the aspect is 'hit' if that max >=
    threshold. Per-chunk max (not one concatenated premise) because NLI
    cross-encoders truncate long inputs. Returns the same shape as
    ``compute_coverage`` plus the per-aspect best chunk index and a graded
    (mean-entailment) score.

    ``top_k`` limits scoring to the FIRST ``top_k`` chunks (chunks arrive already
    ordered by retrieval rank). ``None`` = use all chunks (current behavior).
    Under a limited budget, diverse (MMR) selections cover more aspects than
    redundant (L2) ones, so the metric becomes discriminative.

    Unlike ``compute_coverage`` (premise = generated answer), this uses each
    retrieved chunk as the premise, so it reflects whether retrieval (e.g. MMR
    diversity) surfaced chunks covering the query aspects — independent of the
    (quantized) LLM generation.
    """
    aspects = split_aspects(aspect_queries)
    chunks = [str(c) for c in (chunk_texts or []) if str(c or "").strip()]
    if top_k is not None:
        chunks = chunks[: int(top_k)]

    if not aspects or not chunks:
        return {
            "mode": "nli_entailment_retrieval",
            "nli_model": nli_model_name,
            "per_aspect_scores": [],
            "per_aspect_hit": [],
            "per_aspect_best_chunk": [],
            "overall_coverage": 0.0,
            "graded_coverage": 0.0,
            "top_k": top_k,
        }

    model = _get_nli_model(nli_model_name)
    ent_idx = _entailment_index(model)

    # 모든 (chunk, aspect) 쌍을 한 번에 배치 predict (aspect-major flatten).
    pairs = [(chunk, aspect) for aspect in aspects for chunk in chunks]
    logits = model.predict(pairs, convert_to_numpy=True)  # (A*C, num_labels)
    probs = _softmax(logits)
    ent = probs[:, ent_idx].reshape(len(aspects), len(chunks))  # (A, C)

    per_aspect_scores = ent.max(axis=1).tolist()          # 아스펙트별 최고 청크 점수
    per_aspect_best_chunk = ent.argmax(axis=1).tolist()   # 그 청크 인덱스
    per_hits = [float(s) >= threshold for s in per_aspect_scores]
    overall = sum(per_hits) / len(per_hits) if per_hits else 0.0  # binary hit-rate
    graded = float(np.mean(per_aspect_scores)) if per_aspect_scores else 0.0

    return {
        "mode": "nli_entailment_retrieval",
        "nli_model": nli_model_name,
        "per_aspect_scores": [round(s, 6) for s in per_aspect_scores],
        "per_aspect_hit": per_hits,
        "per_aspect_best_chunk": [int(i) for i in per_aspect_best_chunk],
        "overall_coverage": round(overall, 6),
        "graded_coverage": round(graded, 6),
        "top_k": top_k,
    }
