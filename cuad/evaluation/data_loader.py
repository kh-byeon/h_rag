from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from evaluation.utils.schema import EvalSample


def _to_aspect_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return [str(value).strip()]


def _to_chunk_list(value: Any) -> List[str]:
    """Coerce a retrieved-chunk field into a list of non-empty strings."""
    if not isinstance(value, list):
        return []
    return [str(x) for x in value if str(x or "").strip()]


def _pick(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def normalize_record(raw: Dict[str, Any], idx: int) -> EvalSample:
    """Normalize key variations into EvalSample."""
    # TODO: add dataset-specific mapping rules if new corpora use other fields.
    sample_id = str(_pick(raw, ["sample_id", "id", "uid"], str(idx)))
    query = str(_pick(raw, ["query", "question", "prompt"], "")).strip()
    ground_truth = str(_pick(raw, ["ground_truth", "reference", "answer_gt"], "")).strip()
    aspect_queries = _to_aspect_list(_pick(raw, ["aspect_queries", "aspects", "query_aspects"], []))

    answer_a = _pick(raw, ["answer_a", "output_a"], None)
    answer_b = _pick(raw, ["answer_b", "output_b"], None)
    answer = _pick(raw, ["answer", "output"], None)

    chunks_a = _to_chunk_list(_pick(raw, ["chunks_a"], None))
    chunks_b = _to_chunk_list(_pick(raw, ["chunks_b"], None))
    chunks = _to_chunk_list(_pick(raw, ["chunks"], None))

    model_name = _pick(raw, ["model_name"], None)
    quantization = _pick(raw, ["quantization"], None)
    retrieval_method = _pick(raw, ["retrieval_method"], None)

    return EvalSample(
        sample_id=sample_id,
        query=query,
        ground_truth=ground_truth,
        aspect_queries=aspect_queries,
        answer_a=str(answer_a) if answer_a is not None else None,
        answer_b=str(answer_b) if answer_b is not None else None,
        answer=str(answer) if answer is not None else None,
        chunks_a=chunks_a,
        chunks_b=chunks_b,
        chunks=chunks,
        model_name=str(model_name) if model_name is not None else None,
        quantization=str(quantization) if quantization is not None else None,
        retrieval_method=str(retrieval_method) if retrieval_method is not None else None,
        metadata=raw,
    )


def load_samples(path: str) -> List[EvalSample]:
    ext = os.path.splitext(path)[1].lower()
    rows: List[Dict[str, Any]] = []

    if ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict):
            # TODO: expand as needed for dataset-specific containers.
            rows = data.get("samples") or data.get("items") or [data]
        else:
            rows = []
    else:
        raise ValueError(f"Unsupported file type: {path}")

    return [normalize_record(r, i) for i, r in enumerate(rows)]
