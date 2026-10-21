#!/usr/bin/env python3
"""
Reference-free generation scoring for a sweep: faithfulness + answer relevance.

Why these two
-------------
EM / F1 / ROUGE measure overlap with the gold answer, so they inherit a bias
against a retriever that trades relevance for diversity. Faithfulness and answer
relevance never look at the gold answer:

  faithfulness      — are the answer's claims entailed by the RETRIEVED passages?
                      (NLI, max entailment over passages per claim)
  answer_relevance  — does the answer address the QUESTION? (SBERT cosine)

Cost control
------------
The same (answer, context-set) pair recurs across many hyperparameter combos —
measured ~89% duplication on this repo's sweeps. Work is deduplicated on
``work_key`` before any model call, so the NLI budget is a few thousand pairs
rather than a few hundred thousand.

Models are loaded lazily and only once:
  NLI   : MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli  (GPU strongly advised)
  SBERT : sentence-transformers/all-MiniLM-L6-v2

Usage
-----
    python scripts/score_faithfulness_relevance.py \
        --combos_dir llm_out_fp16/combos:fp16 \
                     llm_out_int8/combos:int8 \
                     llm_out_int4/combos:int4 \
        --top_k 5 \
        --out figures/faith_rel_by_query.csv

    # smoke test on two combos before committing the GPU
    python scripts/score_faithfulness_relevance.py --combos_dir llm_out_fp16/combos:fp16 \
        --top_k 5 --limit_combos 2 --out /tmp/smoke.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evaluation.metrics.faithfulness_relevance import (  # noqa: E402
    compute_answer_relevance,
    compute_faithfulness,
    work_key,
)


def _contexts_of(rec: Dict[str, Any], arm: str) -> List[str]:
    """Retrieved passage texts for one arm, from the combo record's chunk_lut."""
    lut = rec.get("chunk_lut")
    if isinstance(lut, dict):
        entries = lut.get(arm) or []
    elif isinstance(lut, list) and arm == "l2":
        entries = lut
    else:
        entries = []
    out: List[str] = []
    for e in entries:
        if isinstance(e, dict):
            t = e.get("chunk_text") or e.get("text") or ""
        else:
            t = str(e)
        t = str(t).strip()
        if t:
            out.append(t)
    return out


def _parse_inputs(items: Sequence[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for it in items:
        if ":" in it and not os.path.exists(it):
            path, _, label = it.rpartition(":")
        else:
            path, label = it, ""
        out.append((path, label or os.path.basename(os.path.dirname(path.rstrip("/\\")))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--combos_dir", nargs="+", required=True, help="dir or 'dir:label'")
    ap.add_argument("--top_k", type=int, default=None, help="only score this top_k")
    ap.add_argument("--out", default="figures/faith_rel_by_query.csv")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="entailment probability above which a claim counts as supported")
    ap.add_argument("--max_claims", type=int, default=12)
    ap.add_argument("--limit_combos", type=int, default=None, help="smoke-test cap")
    ap.add_argument("--nli_model", default=None)
    ap.add_argument("--sbert_model", default=None)
    args = ap.parse_args()

    kw: Dict[str, Any] = {"threshold": args.threshold, "max_claims": args.max_claims}
    if args.nli_model:
        kw["nli_model_name"] = args.nli_model
    rel_kw: Dict[str, Any] = {}
    if args.sbert_model:
        rel_kw["sbert_model_name"] = args.sbert_model

    # ---- pass 1: collect unique work units -------------------------------
    units: Dict[str, Dict[str, Any]] = {}          # work_key -> payload
    rows: List[Dict[str, Any]] = []                # one row per (label, combo, query, arm)
    for path, label in _parse_inputs(args.combos_dir):
        files = sorted(glob.glob(os.path.join(path, "llm_combo_*.json")))
        if not files:
            raise SystemExit(f"no llm_combo_*.json under {path}")
        # Filter by top_k BEFORE applying --limit_combos, otherwise the cap can
        # select only combos that the filter then discards (silently yielding 0 rows).
        loaded = [(fp, json.load(open(fp, encoding="utf-8"))) for fp in files]
        if args.top_k is not None:
            kept = [(fp, pl) for fp, pl in loaded
                    if int((pl.get("parameters") or {}).get("top_k") or 0) == args.top_k]
            if not kept:
                avail = sorted({int((pl.get("parameters") or {}).get("top_k") or 0)
                                for _, pl in loaded})
                raise SystemExit(
                    f"{path}: no combo with top_k={args.top_k}. Available top_k: {avail}")
            loaded = kept
        if args.limit_combos:
            loaded = loaded[: args.limit_combos]
        print(f"[scorer] {label}: {len(loaded)} combo(s)"
              + (f" at top_k={args.top_k}" if args.top_k is not None else ""))
        for fp, pl in loaded:
            prm = pl.get("parameters") or {}
            for r in pl.get("per_query_results") or []:
                q = r.get("rewritten_query") or r.get("query") or ""
                for arm in ("l2", "mmr"):
                    ans = r.get(f"answer_{arm}") or ""
                    ctx = _contexts_of(r, arm)
                    k = work_key(ans, ctx)
                    units.setdefault(k, {"answer": ans, "contexts": ctx, "question": q})
                    rows.append({
                        "source": label, "sweep_idx": pl.get("sweep_idx"),
                        "query_idx": r.get("query_idx"), "arm": arm,
                        "top_k": prm.get("top_k"), "mmr_lambda": prm.get("mmr_lambda"),
                        "tau_dup": prm.get("tau_dup"), "gamma": prm.get("gamma"),
                        "key": k,
                    })
    if not rows:
        raise SystemExit("[scorer] no rows collected — check --combos_dir and --top_k")
    print(f"[scorer] rows={len(rows):,}  unique work units={len(units):,} "
          f"({100*(1-len(units)/len(rows)):.1f}% deduplicated)")

    # ---- pass 2: score each unique unit once ------------------------------
    scored: Dict[str, Dict[str, float]] = {}
    for i, (k, u) in enumerate(units.items(), 1):
        f = compute_faithfulness(u["answer"], u["contexts"], **kw)
        rel = compute_answer_relevance(u["question"], u["answer"], **rel_kw)
        scored[k] = {
            "faithfulness": f["faithfulness"],
            "faithfulness_mean_entail": f["faithfulness_mean_entail"],
            "n_claims": f["n_claims"],
            "n_no_info": f.get("n_no_info", 0),
            "answer_relevance": rel["answer_relevance"],
        }
        if i % 200 == 0 or i == len(units):
            print(f"[scorer] {i:,}/{len(units):,}", flush=True)

    # ---- emit -------------------------------------------------------------
    cols = ["source", "sweep_idx", "query_idx", "arm", "top_k",
            "mmr_lambda", "tau_dup", "gamma",
            "faithfulness", "faithfulness_mean_entail", "answer_relevance",
            "n_claims", "n_no_info"]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            r.update(scored[r["key"]])
            w.writerow({c: r.get(c) for c in cols})
    print(f"[scorer] -> {args.out}")

    # ---- quick arm summary (query-clustered, the only valid unit) ---------
    import statistics as st
    per = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for m in ("faithfulness", "faithfulness_mean_entail", "answer_relevance"):
            v = r.get(m)
            if v is not None and v == v:
                per[(r["source"], r["arm"], m)][r["query_idx"]].append(float(v))
    srcs = sorted({r["source"] for r in rows})
    print(f"\n{'source':<8}{'metric':<26}{'L2':>9}{'MMR':>9}{'Δ':>10}")
    for s in srcs:
        for m in ("faithfulness", "faithfulness_mean_entail", "answer_relevance"):
            a = per[(s, "l2", m)]
            b = per[(s, "mmr", m)]
            qs = sorted(set(a) & set(b))
            if not qs:
                continue
            av = st.mean(st.mean(a[q]) for q in qs)
            bv = st.mean(st.mean(b[q]) for q in qs)
            print(f"{s:<8}{m:<26}{av:>9.4f}{bv:>9.4f}{bv-av:>+10.4f}")
    print("\n(n_query-clustered means. Use the CSV for CIs — bootstrap over query_idx,\n"
          " never over rows: combos repeat the same queries.)")


if __name__ == "__main__":
    main()
