from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from evaluation.config import EvalConfig, load_config
from evaluation.data_loader import load_samples
from evaluation.judges.faithfulness_judge import judge_faithfulness
from evaluation.judges.lawyer_suitability_judge import judge_lawyer_suitability
from evaluation.judges.llm_client import LLMClient
from evaluation.metrics.sbert_metric import compute_sbert_similarity
from evaluation.metrics.coverage_metric import compute_coverage, compute_retrieval_coverage
from evaluation.utils.io_utils import ensure_dir, save_json
from evaluation.utils.schema import EvalSample


# Retrieval-coverage is computed at several budgets. None = all chunks.
RETRIEVAL_COVERAGE_KS: Tuple[Optional[int], ...] = (3, 5, None)


def retrieval_coverage_key(kk: Optional[int]) -> str:
    """Stable key name for a given retrieval-coverage budget (top3/top5/all)."""
    return f"top{kk}" if kk is not None else "all"


def evaluate_one_answer(
    sample: EvalSample,
    answer_text: str,
    cfg: EvalConfig,
    client: Optional[LLMClient],
    dry_run_judges: bool,
    retrieved_chunks: Optional[List[str]] = None,
) -> Dict[str, Any]:
    bert = compute_sbert_similarity(
        reference_text=sample.ground_truth,
        candidate_text=answer_text,
        model_name=cfg.sbert_model_name,
    )
    coverage = compute_coverage(
        aspect_queries=sample.aspect_queries,
        answer_text=answer_text,
        threshold=cfg.coverage_similarity_threshold,
        nli_model_name=cfg.nli_model_name,
    )
    # Retrieval-level aspect coverage at several budgets (top3/top5/all).
    # Under a limited budget MMR's diverse chunks cover more aspects than L2's
    # redundant ones. Uses the local NLI model (runs even under --dry_run_judges).
    retrieval_coverage: Dict[str, Any] = {}
    for kk in RETRIEVAL_COVERAGE_KS:
        rc = compute_retrieval_coverage(
            aspect_queries=sample.aspect_queries,
            chunk_texts=retrieved_chunks or [],
            threshold=cfg.coverage_similarity_threshold,
            nli_model_name=cfg.nli_model_name,
            top_k=kk,
        )
        retrieval_coverage[retrieval_coverage_key(kk)] = rc

    faithfulness = judge_faithfulness(
        client=client if client is not None else LLMClient(provider="openai"),
        query=sample.query,
        ground_truth=sample.ground_truth,
        answer=answer_text,
        dry_run=dry_run_judges,
    )
    lawyer = judge_lawyer_suitability(
        client=client if client is not None else LLMClient(provider="openai"),
        query=sample.query,
        aspect_queries=sample.aspect_queries,
        ground_truth=sample.ground_truth,
        answer=answer_text,
        dry_run=dry_run_judges,
    )
    return {
        "bertscore": bert,
        "coverage": coverage,
        "retrieval_coverage": retrieval_coverage,
        "faithfulness": faithfulness,
        "lawyer_suitability": lawyer,
    }


def _safe_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def retrieval_coverage_metric_keys() -> List[str]:
    """Flat summary keys emitted per arm (binary + graded, one pair per budget)."""
    keys: List[str] = []
    for kk in RETRIEVAL_COVERAGE_KS:
        base = f"retrieval_coverage_{retrieval_coverage_key(kk)}"
        keys.append(base)
        keys.append(base + "_graded")
    return keys


def collect_retrieval_coverage(
    result: Dict[str, Any], buckets: Dict[str, List[float]]
) -> None:
    """Append per-budget binary/graded retrieval-coverage values into buckets."""
    rc = result.get("retrieval_coverage") or {}
    for kk in RETRIEVAL_COVERAGE_KS:
        entry = rc.get(retrieval_coverage_key(kk)) or {}
        base = f"retrieval_coverage_{retrieval_coverage_key(kk)}"
        buckets[base].append(float(entry.get("overall_coverage", 0.0)))
        buckets[base + "_graded"].append(float(entry.get("graded_coverage", 0.0)))


def _aggregate_metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    buckets: Dict[str, List[float]] = {
        "bertscore_precision": [],
        "bertscore_recall": [],
        "bertscore_f1": [],
        "coverage": [],
        "faithfulness_score": [],
        "lawyer_suitability_score": [],
    }
    for k in retrieval_coverage_metric_keys():
        buckets[k] = []
    for row in rows:
        r = row["result"]
        buckets["bertscore_precision"].append(float(r["bertscore"].get("precision", 0.0)))
        buckets["bertscore_recall"].append(float(r["bertscore"].get("recall", 0.0)))
        buckets["bertscore_f1"].append(float(r["bertscore"].get("f1", 0.0)))
        buckets["coverage"].append(float(r["coverage"].get("overall_coverage", 0.0)))
        collect_retrieval_coverage(r, buckets)

        fs = _safe_float(r["faithfulness"].get("score"))
        ls = _safe_float(r["lawyer_suitability"].get("score"))
        if fs is not None:
            buckets["faithfulness_score"].append(fs)
        if ls is not None:
            buckets["lawyer_suitability_score"].append(ls)

    out: Dict[str, float] = {}
    for k, vals in buckets.items():
        out[k] = (sum(vals) / len(vals)) if vals else 0.0
    return out


def _dig(d: Any, path: Tuple[str, ...]) -> Any:
    """Follow a nested-dict key path; return None on any miss."""
    cur = d
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _pairwise_wins(per_sample: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    metrics = {
        "bertscore_f1": ("bertscore", "f1"),
        "coverage": ("coverage", "overall_coverage"),
        "faithfulness_score": ("faithfulness", "score"),
        "lawyer_suitability_score": ("lawyer_suitability", "score"),
    }
    for kk in RETRIEVAL_COVERAGE_KS:
        arm = retrieval_coverage_key(kk)
        metrics[f"retrieval_coverage_{arm}"] = (
            "retrieval_coverage", arm, "overall_coverage",
        )
        metrics[f"retrieval_coverage_{arm}_graded"] = (
            "retrieval_coverage", arm, "graded_coverage",
        )
    wins: Dict[str, Dict[str, int]] = {
        m: {"answer_a": 0, "answer_b": 0, "tie": 0} for m in metrics
    }

    for item in per_sample:
        ra = item["results"].get("answer_a")
        rb = item["results"].get("answer_b")
        if not ra or not rb:
            continue
        for m, path in metrics.items():
            va = _safe_float(_dig(ra, path))
            vb = _safe_float(_dig(rb, path))
            if va is None or vb is None:
                continue
            if abs(va - vb) < 1e-9:
                wins[m]["tie"] += 1
            elif va > vb:
                wins[m]["answer_a"] += 1
            else:
                wins[m]["answer_b"] += 1
    return wins


def run_evaluation(cfg: EvalConfig, dry_run_judges: bool = True) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    samples = load_samples(cfg.input_path)
    if cfg.max_samples is not None:
        samples = samples[: cfg.max_samples]

    client = None if dry_run_judges else LLMClient(
        provider=cfg.llm_judge_provider,
        model=cfg.llm_judge_model,
        temperature=cfg.llm_judge_temperature,
    )

    per_sample: List[Dict[str, Any]] = []
    flat_rows: List[Dict[str, Any]] = []

    for s in samples:
        row: Dict[str, Any] = {
            "sample_id": s.sample_id,
            "query": s.query,
            "results": {},
        }
        if cfg.compare_mode == "pairwise":
            for tag, ans, chunks in (
                ("answer_a", s.answer_a, s.chunks_a),
                ("answer_b", s.answer_b, s.chunks_b),
            ):
                if ans is None:
                    continue
                result = evaluate_one_answer(
                    s, ans, cfg, client, dry_run_judges, retrieved_chunks=chunks
                )
                row["results"][tag] = result
                flat_rows.append({"sample_id": s.sample_id, "answer_tag": tag, "result": result})
        else:
            ans = s.answer or s.answer_a or ""
            chunks = s.chunks or s.chunks_a or s.chunks_b
            result = evaluate_one_answer(
                s, ans, cfg, client, dry_run_judges, retrieved_chunks=chunks
            )
            row["results"]["answer"] = result
            flat_rows.append({"sample_id": s.sample_id, "answer_tag": "answer", "result": result})
        per_sample.append(row)

    summary: Dict[str, Any] = {"averages": _aggregate_metrics(flat_rows)}
    if cfg.compare_mode == "pairwise":
        summary["pairwise_wins"] = _pairwise_wins(per_sample)
    return per_sample, summary


def _demo_samples_path() -> str:
    return os.path.join(os.getcwd(), "evaluation_demo_samples.json")


if __name__ == "__main__":
    # Demo mode: dry-run judges by default to avoid API calls.
    # Set dry_run_judges=False after configuring real API key and model.
    demo_data = [
        {
            "sample_id": "demo-1",
            "query": "Does this contract grant an irrevocable or perpetual license?",
            "ground_truth": "The agreement grants an irrevocable license; perpetual duration is not explicitly stated.",
            "aspect_queries": [
                "scope and subject of the license",
                "rights and obligations",
                "exclusions or limitations",
                "conditions and transfer restrictions",
            ],
            "answer_a": "The contract grants an irrevocable license but does not clearly state perpetual duration.",
            "answer_b": "The contract grants a perpetual, irrevocable license with unlimited transfer rights.",
        }
    ]
    demo_path = _demo_samples_path()
    save_json(demo_path, demo_data)

    cfg = load_config(
        {
            "input_path": demo_path,
            "output_dir": os.path.join(os.getcwd(), "evaluation_outputs"),
            "sbert_model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "coverage_similarity_threshold": 0.3,
            "llm_judge_provider": "openai",
            "llm_judge_model": "gpt-4o-mini",
            "llm_judge_temperature": 0.0,
            "compare_mode": "pairwise",
            "max_samples": 10,
        }
    )

    ensure_dir(cfg.output_dir)
    per_sample, summary = run_evaluation(cfg, dry_run_judges=True)
    save_json(os.path.join(cfg.output_dir, "per_sample_results.json"), per_sample)
    save_json(os.path.join(cfg.output_dir, "summary.json"), summary)
    print(f"[done] samples={len(per_sample)} output_dir={cfg.output_dir}")
