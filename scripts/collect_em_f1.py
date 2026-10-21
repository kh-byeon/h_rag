#!/usr/bin/env python3
"""
Per-combo EM / F1 / supporting-recall collector for a sweep generation run.

Why this exists
---------------
``run_eval_from_generation.py`` has no ``sweep_idx`` grouping: hand it a
108-combo normalized-generation JSON and it averages every combo together,
which is meaningless for a hyperparameter sweep. This script splits by
``metadata.sweep_idx`` and scores each combo separately.

It also skips the heavy path on purpose. For ``domain=hotpot`` the only live
metrics are answer EM/F1 and supporting-passage recall -- both are pure
functions (``evaluation.metrics.answer_em_f1`` / ``.supporting_recall``) with no
model and no I/O. BERTScore / aspect-coverage / LLM judges are CUAD-only and are
skipped upstream anyway, so no SBERT or NLI checkpoint needs to load. This runs
on CPU in seconds.

Input
-----
One or more normalized-generation JSON files (the ``--output_normalized_json``
product of ``llm_generator.py``). Records must carry ``metadata.sweep_idx``;
``gold_support`` and ``retrieval.retrieved_doc_ids`` enable supporting recall.

Output
------
A tidy CSV: one row per (source, sweep_idx), with the sweep parameters joined in
from metadata, plus EM/F1 per arm and their deltas.

Usage
-----
    python scripts/collect_em_f1.py \
        --normalized_json llm_out_fp16/generation_normalized.json \
        --label fp16 \
        --out figures/em_f1_by_combo.csv

    # several quantizations at once
    python scripts/collect_em_f1.py \
        --normalized_json generation_normalized_fp16.json:fp16 \
                          generation_normalized_int8.json:int8 \
                          generation_normalized_int4.json:int4 \
        --out figures/em_f1_by_combo.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evaluation.metrics.answer_em_f1 import (  # noqa: E402
    aggregate as aggregate_em_f1,
)
from evaluation.metrics.answer_em_f1 import (  # noqa: E402
    extract_final_answer,
    score_answer,
)
from evaluation.metrics.supporting_recall import (  # noqa: E402
    aggregate as aggregate_support,
)
from evaluation.metrics.supporting_recall import evaluate_supporting_recall  # noqa: E402

SUPPORT_KS: Tuple[Optional[int], ...] = (2, 5, 10)
PARAM_KEYS = ("top_k", "mmr_lambda", "tau_dup", "gamma", "window_size",
              "model_name", "quantization", "run_id")


def _golds(rec: Dict[str, Any]) -> List[str]:
    """Gold answers, tolerating str | list shapes."""
    gt = rec.get("ground_truth")
    if gt is None:
        return []
    if isinstance(gt, (list, tuple)):
        return [str(x) for x in gt if str(x).strip()]
    return [str(gt)] if str(gt).strip() else []


def _answers(rec: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """(l2, mmr) raw generations.

    Handles all three layouts seen in this repo:
      canonical : generation = {"l2": {"answer": ...}, "mmr": {"answer": ...}}
      detail    : generation = {"answer_l2": ..., "answer_mmr": ...}
      flat      : record     = {"answer_l2": ..., "answer_mmr": ...}
    """
    gen = rec.get("generation")
    if isinstance(gen, dict):
        arm_l2, arm_mmr = gen.get("l2"), gen.get("mmr")
        if isinstance(arm_l2, dict) or isinstance(arm_mmr, dict):
            return (
                (arm_l2 or {}).get("answer") if isinstance(arm_l2, dict) else None,
                (arm_mmr or {}).get("answer") if isinstance(arm_mmr, dict) else None,
            )
        if "answer_l2" in gen or "answer_mmr" in gen:
            return gen.get("answer_l2"), gen.get("answer_mmr")
    return rec.get("answer_l2"), rec.get("answer_mmr")


def _as_id_list(value: Any) -> Optional[List[int]]:
    """Coerce a retrieved-id field to List[int]; tolerate a str-ified list."""
    if isinstance(value, str):
        try:
            value = json.loads(value.replace("'", '"'))
        except (ValueError, TypeError):
            return None
    if isinstance(value, (list, tuple)):
        try:
            return [int(x) for x in value]
        except (ValueError, TypeError):
            return None
    return None


def _retrieved_ids(rec: Dict[str, Any]) -> Dict[str, List[int]]:
    """Ranked parent doc_ids per arm.

    canonical : retrieval = {"l2": {"retrieved_doc_ids": [...]}, "mmr": {...}}
    sweep     : retrieved_doc_ids = {"l2": [...], "mmr": [...]}
    """
    out: Dict[str, List[int]] = {}
    r = rec.get("retrieval")
    if isinstance(r, dict):
        for arm in ("l2", "mmr"):
            sub = r.get(arm)
            if isinstance(sub, dict):
                ids = _as_id_list(sub.get("retrieved_doc_ids"))
                if ids is not None:
                    out[arm] = ids
        if out:
            return out
    flat = rec.get("retrieved_doc_ids")
    if isinstance(flat, dict):
        for arm in ("l2", "mmr"):
            ids = _as_id_list(flat.get(arm))
            if ids is not None:
                out[arm] = ids
    return out


def score_group(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """EM/F1 (both arms) + supporting recall (both arms) for one combo."""
    rows_l2: List[Dict[str, float]] = []
    rows_mmr: List[Dict[str, float]] = []
    sup_l2: List[Dict[str, float]] = []
    sup_mmr: List[Dict[str, float]] = []
    n_missing_answer = 0

    for rec in records:
        golds = _golds(rec)
        a_l2, a_mmr = _answers(rec)
        if not golds:
            continue
        if a_l2 is None and a_mmr is None:
            n_missing_answer += 1
            continue
        rows_l2.append(score_answer(extract_final_answer(a_l2 or ""), golds))
        rows_mmr.append(score_answer(extract_final_answer(a_mmr or ""), golds))

        gold_support = rec.get("gold_support")
        ids = _retrieved_ids(rec)
        if gold_support and ids:
            if "l2" in ids:
                sup_l2.append(evaluate_supporting_recall(ids["l2"], gold_support, ks=SUPPORT_KS))
            if "mmr" in ids:
                sup_mmr.append(evaluate_supporting_recall(ids["mmr"], gold_support, ks=SUPPORT_KS))

    out: Dict[str, Any] = {
        "n_queries": len(rows_l2),
        "n_missing_answer": n_missing_answer,
        "em_f1": {"l2": aggregate_em_f1(rows_l2), "mmr": aggregate_em_f1(rows_mmr)},
    }
    if sup_l2 or sup_mmr:
        out["supporting_recall"] = {
            "l2": aggregate_support(sup_l2),
            "mmr": aggregate_support(sup_mmr),
        }
    return out


def load_and_group(path: str) -> Dict[int, List[Dict[str, Any]]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(f"{path}: expected a JSON list of records")
    groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for rec in data:
        meta = rec.get("metadata") or {}
        groups[int(meta.get("sweep_idx") or 0)].append(rec)
    return dict(groups)


def load_combos_dir(combos_dir: str) -> Dict[int, List[Dict[str, Any]]]:
    """Group from ``combos/llm_combo_*.json`` instead of the normalized JSON.

    The normalized JSON is written only after every combo finishes, so a run that
    dies mid-sweep leaves none. The per-combo files are flushed as each combo
    completes, which makes them the crash-safe source of truth. They carry the
    answers and ground truth but no ``gold_support`` -- supporting recall is
    already in the sweep CSV, so join on ``sweep_idx`` there instead.
    """
    import glob

    groups: Dict[int, List[Dict[str, Any]]] = {}
    for p in sorted(glob.glob(os.path.join(combos_dir, "llm_combo_*.json"))):
        with open(p, encoding="utf-8") as f:
            payload = json.load(f)
        idx = int(payload.get("sweep_idx") or 0)
        params = payload.get("parameters") or {}
        recs: List[Dict[str, Any]] = []
        for r in payload.get("per_query_results") or []:
            rec = dict(r)
            rec["metadata"] = {"sweep_idx": idx, "run_id": payload.get("run_id"), **params}
            recs.append(rec)
        groups[idx] = recs
    if not groups:
        raise SystemExit(f"no llm_combo_*.json under {combos_dir}")
    return groups


def _params(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    meta = (records[0].get("metadata") or {}) if records else {}
    return {k: meta.get(k) for k in PARAM_KEYS}


def _parse_inputs(items: Iterable[str]) -> List[Tuple[str, str]]:
    """Accept ``path`` or ``path:label``; default label = filename stem."""
    out: List[Tuple[str, str]] = []
    for it in items:
        # rsplit so Windows drive letters (C:\...) survive
        if ":" in it and not os.path.exists(it):
            path, _, label = it.rpartition(":")
        else:
            path, label = it, ""
        label = label or os.path.splitext(os.path.basename(path))[0]
        out.append((path, label))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--normalized_json", nargs="+",
                     help="one or more paths, optionally 'path:label'")
    src.add_argument("--combos_dir", nargs="+",
                     help="one or more llm_out*/combos dirs, optionally 'dir:label' "
                          "(crash-safe: works before the normalized JSON is written)")
    ap.add_argument("--out", default="figures/em_f1_by_combo.csv")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    header = ["source", "sweep_idx", *PARAM_KEYS, "n_queries",
              "l2_em", "mmr_em", "em_delta", "l2_f1", "mmr_f1", "f1_delta"]
    for k in SUPPORT_KS:
        header += [f"l2_recall@{k}", f"mmr_recall@{k}", f"l2_full@{k}", f"mmr_full@{k}"]

    from_combos = bool(args.combos_dir)
    inputs = _parse_inputs(args.combos_dir if from_combos else args.normalized_json)

    rows: List[Dict[str, Any]] = []
    for path, label in inputs:
        if from_combos:
            if not os.path.isdir(path):
                raise SystemExit(f"not a directory: {path}")
            groups = load_combos_dir(path)
        else:
            if not os.path.isfile(path):
                raise SystemExit(f"not found: {path}")
            groups = load_and_group(path)
        if not args.quiet:
            print(f"[{label}] {path}: {len(groups)} combo(s), "
                  f"{sum(len(v) for v in groups.values())} records")
        for idx in sorted(groups):
            recs = groups[idx]
            s = score_group(recs)
            em = s["em_f1"]
            row: Dict[str, Any] = {"source": label, "sweep_idx": idx,
                                   **_params(recs), "n_queries": s["n_queries"]}
            l2e, mme = em["l2"].get("em"), em["mmr"].get("em")
            l2f, mmf = em["l2"].get("f1"), em["mmr"].get("f1")
            row.update({
                "l2_em": l2e, "mmr_em": mme,
                "em_delta": (mme - l2e) if (l2e is not None and mme is not None) else None,
                "l2_f1": l2f, "mmr_f1": mmf,
                "f1_delta": (mmf - l2f) if (l2f is not None and mmf is not None) else None,
            })
            sr = s.get("supporting_recall") or {}
            for k in SUPPORT_KS:
                row[f"l2_recall@{k}"] = (sr.get("l2") or {}).get(f"recall@{k}")
                row[f"mmr_recall@{k}"] = (sr.get("mmr") or {}).get(f"recall@{k}")
                row[f"l2_full@{k}"] = (sr.get("l2") or {}).get(f"full_support@{k}")
                row[f"mmr_full@{k}"] = (sr.get("mmr") or {}).get(f"full_support@{k}")
            rows.append(row)

            if not args.quiet:
                print(f"  [{label}] combo {idx:>4} (top_k={row.get('top_k')}, "
                      f"λ={row.get('mmr_lambda')}, τ={row.get('tau_dup')}, γ={row.get('gamma')}) "
                      f"n={s['n_queries']:>3} | "
                      f"EM {l2e:.4f}→{mme:.4f} ({row['em_delta']:+.4f}) | "
                      f"F1 {l2f:.4f}→{mmf:.4f} ({row['f1_delta']:+.4f})")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in header})
    print(f"\n[collect_em_f1] {len(rows)} row(s) -> {args.out}")


if __name__ == "__main__":
    main()
