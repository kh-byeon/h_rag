from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvalSample:
    """Normalized input sample supporting pairwise and single modes."""

    sample_id: str
    query: str
    ground_truth: str
    aspect_queries: List[str]
    answer_a: Optional[str] = None
    answer_b: Optional[str] = None
    answer: Optional[str] = None
    # Retrieved chunk texts per arm (for retrieval-level aspect coverage).
    chunks_a: List[str] = field(default_factory=list)
    chunks_b: List[str] = field(default_factory=list)
    chunks: List[str] = field(default_factory=list)
    model_name: Optional[str] = None
    quantization: Optional[str] = None
    retrieval_method: Optional[str] = None
    # Multi-hop supporting-passage recall (HotpotQA). gold_support = gold corpus
    # paragraph indices; *_doc_ids = ranked parent doc_id per arm (arm A = L2,
    # arm B = MMR). All None/empty for single-answer datasets (e.g. CUAD).
    gold_support: Optional[List[int]] = None
    l2_doc_ids: Optional[List[int]] = None
    mmr_doc_ids: Optional[List[int]] = None
    # Dataset domain ("hotpot"/"cuad"); drives HotpotQA vs CUAD metric branching.
    domain: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JudgeResult:
    """Structured judge result."""

    score: Optional[float] = None
    reasoning: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Per-answer evaluation result container."""

    semantic_similarity: Dict[str, float] = field(default_factory=dict)
    coverage: Dict[str, Any] = field(default_factory=dict)
    faithfulness_judge: Dict[str, Any] = field(default_factory=dict)
    lawyer_suitability_judge: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
