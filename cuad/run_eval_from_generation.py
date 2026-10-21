"""
run_eval_from_generation.py — normalized 생성 결과 평가 실행기.

실행 예시:
    python run_eval_from_generation.py \
        --normalized_json generation_normalized.json \
        --output_dir evaluation_outputs \
        --compare_mode pairwise \
        --dry_run_judges

로드한 canonical JSON 각 레코드에 대해 postprocess.process_record를 in-memory로 적용한 뒤
refined_answer_l2 / refined_answer_mmr를 채워 평가한다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from evaluation.config import EvalConfig
from evaluation.main import (
    collect_retrieval_coverage,
    retrieval_coverage_metric_keys,
    run_evaluation,
)
from evaluation.utils.io_utils import ensure_dir, save_json
from postprocess import process_record


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _as_lut_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def canonical_record_to_postprocess_record(r: Dict[str, Any]) -> Dict[str, Any]:
    """
    postprocess.process_record가 기대하는 flat 입력으로 변환한다.

    지원 형식:
    - flat: answer_l2 / answer_mmr / chunk_lut with l2 and mmr lists
    - canonical: generation.l2.answer, generation.mmr.answer,
      retrieval.l2.chunk_lut, retrieval.mmr.chunk_lut (또는 top-level chunk_lut)
    """
    gen = r.get("generation") or {}
    retrieval = r.get("retrieval") or {}
    top_lut = r.get("chunk_lut") or {}
    l2_lut = _as_lut_list(top_lut.get("l2"))
    mmr_lut = _as_lut_list(top_lut.get("mmr"))
    if not l2_lut and not mmr_lut:
        l2_lut = _as_lut_list((retrieval.get("l2") or {}).get("chunk_lut"))
        mmr_lut = _as_lut_list((retrieval.get("mmr") or {}).get("chunk_lut"))
    chunk_lut = {"l2": l2_lut, "mmr": mmr_lut}

    ans_l2 = r.get("answer_l2") or ((gen.get("l2") or {}).get("answer")) or ""
    ans_mmr = r.get("answer_mmr") or ((gen.get("mmr") or {}).get("answer")) or ""
    if not isinstance(ans_l2, str):
        ans_l2 = str(ans_l2) if ans_l2 is not None else ""
    if not isinstance(ans_mmr, str):
        ans_mmr = str(ans_mmr) if ans_mmr is not None else ""

    return {
        "query": r.get("query", ""),
        "query_idx": r.get("query_idx", r.get("idx")),
        "ground_truth": r.get("ground_truth", ""),
        "aspect_queries": r.get("aspect_queries"),
        "aspect_queries_raw": r.get("aspect_queries_raw"),
        "metadata": r.get("metadata"),
        "chunk_lut": chunk_lut,
        "answer_l2": ans_l2,
        "answer_mmr": ans_mmr,
    }


def apply_postprocess_to_normalized_records(
    rows: List[Dict[str, Any]],
    *,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Each row: flat input → process_record → merge refined_* onto a shallow copy of the row."""
    enriched: List[Dict[str, Any]] = []
    refined_l2 = 0
    refined_mmr = 0
    for i, r in enumerate(rows):
        flat = canonical_record_to_postprocess_record(r)
        merged = dict(r)
        try:
            processed = process_record(flat.copy())
        except Exception as e:
            print(f"[경고] postprocess 실패 (레코드 {i}): {e}", file=sys.stderr)
            enriched.append(merged)
            continue
        ra = processed.get("refined_answer_l2", "")
        rb = processed.get("refined_answer_mmr", "")
        merged["refined_answer_l2"] = ra
        merged["refined_answer_mmr"] = rb
        if isinstance(ra, str) and ra.strip():
            refined_l2 += 1
        if isinstance(rb, str) and rb.strip():
            refined_mmr += 1
        enriched.append(merged)

    n = len(rows)
    if verbose and n:
        print(f"[postprocess] applied to {n} records")
        print(
            f"[postprocess] refined answers found for "
            f"L2 {refined_l2}/{n}, MMR {refined_mmr}/{n} (non-empty strings)"
        )
    return enriched


def _split_sep_aspects(aspect_queries: List[str]) -> List[str]:
    """
    [SEP]로 이어진 단일 문자열이면 분리해 리스트로 반환한다.
    이미 분리된 리스트면 그대로 반환한다.
    """
    raw = [a.strip() for a in (aspect_queries or []) if a and a.strip()]
    if len(raw) == 1 and "[SEP]" in raw[0]:
        parts = [p.strip() for p in raw[0].split("[SEP]")]
        return [p for p in parts if p]
    return raw


def convert_generation_normalized_to_eval_input(
    rows: List[Dict[str, Any]],
    compare_mode: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, r in enumerate(rows):
        sample_id = str(r.get("query_idx", r.get("idx", i)))
        aspect_queries = _split_sep_aspects(
            r.get("aspect_queries_raw") or r.get("aspect_queries", [])
        )

        base: Dict[str, Any] = {
            "sample_id": sample_id,
            "query": r.get("query", ""),
            "ground_truth": r.get("ground_truth", ""),
            "aspect_queries": aspect_queries,
            "model_name": ((r.get("metadata") or {}).get("model_name")),
            "quantization": ((r.get("metadata") or {}).get("quantization")),
            "retrieval_method": ((r.get("metadata") or {}).get("retrieval_mode")),
        }

        # Retrieved chunks per arm (retrieval-level aspect coverage).
        # Prefer canonical retrieval.<arm>.contexts; fall back to chunk_lut[i].chunk_text.
        ret = r.get("retrieval") or {}

        def _chunks(arm: str) -> List[str]:
            a = (ret.get(arm) or {})
            ctx = a.get("contexts")
            if isinstance(ctx, list) and ctx:
                return [str(x) for x in ctx]
            lut = a.get("chunk_lut") or []
            return [
                str(e.get("chunk_text", ""))
                for e in lut
                if isinstance(e, dict)
            ]

        chunks_l2 = _chunks("l2")
        chunks_mmr = _chunks("mmr")

        # normalized JSON 우선 사용, legacy generation 구조는 fallback으로만 유지.
        gen = r.get("generation") or {}
        ans_l2 = (
            r.get("refined_answer_l2")
            or r.get("answer_l2")
            or ((gen.get("l2") or {}).get("answer"))
            or ""
        )
        ans_mmr = (
            r.get("refined_answer_mmr")
            or r.get("answer_mmr")
            or ((gen.get("mmr") or {}).get("answer"))
            or ""
        )

        if compare_mode == "pairwise":
            base["answer_a"] = ans_l2
            base["answer_b"] = ans_mmr
            base["chunks_a"] = chunks_l2
            base["chunks_b"] = chunks_mmr
        else:
            base["answer"] = ans_l2 or ans_mmr
            base["chunks"] = chunks_l2 or chunks_mmr
        out.append(base)
    return out


def compute_pairwise_answer_averages(
    per_sample: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Mean metrics over samples, split by pairwise arm (answer_a vs answer_b).

    Keys under each arm match summary["averages"] (bertscore_*, coverage,
    faithfulness_score, lawyer_suitability_score).
    """

    def _safe_float(v: Any) -> Optional[float]:
        try:
            return float(v)
        except Exception:
            return None

    def _aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, float]:
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
        for r in results:
            buckets["bertscore_precision"].append(
                float(r["bertscore"].get("precision", 0.0))
            )
            buckets["bertscore_recall"].append(
                float(r["bertscore"].get("recall", 0.0))
            )
            buckets["bertscore_f1"].append(float(r["bertscore"].get("f1", 0.0)))
            buckets["coverage"].append(
                float(r["coverage"].get("overall_coverage", 0.0))
            )
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

    def _collect(tag: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for item in per_sample:
            res = (item.get("results") or {}).get(tag)
            if res:
                rows.append(res)
        return rows

    return {
        "answer_a": _aggregate_results(_collect("answer_a")),
        "answer_b": _aggregate_results(_collect("answer_b")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run evaluation from normalized generation JSON."
    )
    parser.add_argument(
        "--normalized_json",
        required=True,
        help="정규화 생성 결과 JSON(list) 경로",
    )
    parser.add_argument(
        "--output_dir",
        default="evaluation_outputs",
        help="Directory to save evaluation outputs.",
    )
    parser.add_argument(
        "--compare_mode",
        choices=["pairwise", "single"],
        default="pairwise",
        help="Evaluation mode.",
    )
    parser.add_argument(
        "--sbert_model_name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-BERT model name for semantic similarity.",
    )
    parser.add_argument(
        "--nli_model_name",
        default="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        help="NLI CrossEncoder model name (aspect coverage).",
    )
    parser.add_argument(
        "--coverage_similarity_threshold",
        type=float,
        default=0.5,
        help="Coverage threshold.",
    )
    parser.add_argument(
        "--llm_judge_provider",
        default="openai",
        help="Judge provider.",
    )
    parser.add_argument(
        "--llm_judge_model",
        default="gpt-4o-mini",
        help="Judge model.",
    )
    parser.add_argument(
        "--llm_judge_temperature",
        type=float,
        default=0.0,
        help="Judge temperature.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional max samples limit.",
    )
    parser.add_argument(
        "--dry_run_judges",
        action="store_true",
        help="Skip actual LLM judge API calls.",
    )
    parser.add_argument(
        "--save_converted_input",
        action="store_true",
        help="Also save converted evaluation input JSON.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress postprocess summary prints.",
    )
    args = parser.parse_args()

    raw = _load_json(args.normalized_json)
    if not isinstance(raw, list):
        raise ValueError("normalized_json must be a JSON list of records.")

    enriched = apply_postprocess_to_normalized_records(
        raw, verbose=not args.quiet
    )
    converted = convert_generation_normalized_to_eval_input(
        enriched, compare_mode=args.compare_mode
    )

    ensure_dir(args.output_dir)
    if args.save_converted_input:
        converted_path = os.path.join(args.output_dir, "converted_eval_input.json")
        save_json(converted_path, converted)
        print(f"[saved] {converted_path}")

    temp_input_path = os.path.join(args.output_dir, "_temp_eval_input.json")
    save_json(temp_input_path, converted)

    cfg = EvalConfig(
        input_path=temp_input_path,
        output_dir=args.output_dir,
        sbert_model_name=args.sbert_model_name,
        nli_model_name=args.nli_model_name,
        coverage_similarity_threshold=args.coverage_similarity_threshold,
        llm_judge_provider=args.llm_judge_provider,
        llm_judge_model=args.llm_judge_model,
        llm_judge_temperature=args.llm_judge_temperature,
        compare_mode=args.compare_mode,  # type: ignore[arg-type]
        max_samples=args.max_samples,
    )

    per_sample, summary = run_evaluation(
        cfg=cfg,
        dry_run_judges=args.dry_run_judges,
    )

    if args.compare_mode == "pairwise":
        # Drop merged "averages"; emit pairwise-only summary with stable key order.
        summary = {
            "pairwise_answer_averages": compute_pairwise_answer_averages(per_sample),
            "pairwise_wins": summary.get("pairwise_wins", {}),
        }

    save_json(os.path.join(args.output_dir, "per_sample_results.json"), per_sample)
    save_json(os.path.join(args.output_dir, "summary.json"), summary)
    print(f"[done] samples={len(per_sample)} output_dir={os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
