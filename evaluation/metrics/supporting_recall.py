"""
supporting_recall.py — multi-hop supporting-passage recall@k.

The metric where retrieval DIVERSITY pays off: a multi-hop question is answerable
only if the retrieved set covers ALL of its gold supporting paragraphs, which live
in different documents. A relevance-only ranker (L2) tends to pile up near-duplicate
chunks of the single most-similar hop and miss the second hop; MMR spreads the budget
across hops and covers more gold paragraphs at the same k.

Pure functions — no model, no I/O. The eval/sweep layer supplies:
  retrieved_doc_ids : ranked list of the PARENT corpus-paragraph index for each
                      retrieved chunk (chunk -> its source paragraph in hotpot_corpus).
                      Duplicates allowed (several chunks from the same paragraph).
  gold_support      : list/set of gold corpus-paragraph indices for the query.

Definitions (paragraph-level; a gold paragraph counts as retrieved if ANY of its
chunks appears within the top-k chunks):
  recall@k        = |retrieved_gold ∩ gold| / |gold|
  full_support@k  = 1.0 if ALL gold paragraphs retrieved within k else 0.0   (answerable)
  precision@k     = |retrieved_gold| / k                                     (secondary)
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence


def _topk_gold_hits(retrieved_doc_ids: Sequence[int], gold: set, k: int) -> set:
    """Set of gold paragraph ids present among the first k retrieved chunks."""
    if k is None or k <= 0:
        window = list(retrieved_doc_ids)
    else:
        window = list(retrieved_doc_ids)[:k]
    return {d for d in window if d in gold}


def recall_at_k(
    retrieved_doc_ids: Sequence[int],
    gold_support: Iterable[int],
    k: Optional[int] = None,
) -> float:
    gold = {int(g) for g in gold_support}
    if not gold:
        return float("nan")  # undefined; caller should skip in aggregation
    hits = _topk_gold_hits([int(d) for d in retrieved_doc_ids], gold, k if k else 0)
    return len(hits) / len(gold)


def full_support_at_k(
    retrieved_doc_ids: Sequence[int],
    gold_support: Iterable[int],
    k: Optional[int] = None,
) -> float:
    gold = {int(g) for g in gold_support}
    if not gold:
        return float("nan")
    hits = _topk_gold_hits([int(d) for d in retrieved_doc_ids], gold, k if k else 0)
    return 1.0 if hits == gold else 0.0


def precision_at_k(
    retrieved_doc_ids: Sequence[int],
    gold_support: Iterable[int],
    k: int,
) -> float:
    gold = {int(g) for g in gold_support}
    if not gold or not k or k <= 0:
        return float("nan")
    hits = _topk_gold_hits([int(d) for d in retrieved_doc_ids], gold, k)
    return len(hits) / k


def evaluate_supporting_recall(
    retrieved_doc_ids: Sequence[int],
    gold_support: Iterable[int],
    ks: Sequence[Optional[int]] = (2, 5, 10, None),
) -> Dict[str, float]:
    """One query -> {recall@k, full_support@k, precision@k} for each k.

    k=None means "over the whole retrieved list" and is reported as key suffix 'all'.
    """
    out: Dict[str, float] = {}
    for k in ks:
        tag = "all" if (k is None or k <= 0) else str(k)
        out[f"recall@{tag}"] = recall_at_k(retrieved_doc_ids, gold_support, k)
        out[f"full_support@{tag}"] = full_support_at_k(retrieved_doc_ids, gold_support, k)
        if k and k > 0:
            out[f"precision@{tag}"] = precision_at_k(retrieved_doc_ids, gold_support, k)
    return out


def aggregate(rows: List[Dict[str, float]]) -> Dict[str, float]:
    """Mean over queries, skipping NaN (queries with empty gold)."""
    if not rows:
        return {}
    keys = rows[0].keys()
    agg: Dict[str, float] = {}
    for key in keys:
        vals = [r[key] for r in rows if key in r and r[key] == r[key]]  # drop NaN
        agg[key] = sum(vals) / len(vals) if vals else float("nan")
    return agg
