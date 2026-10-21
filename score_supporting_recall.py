#!/usr/bin/env python3
"""score_supporting_recall.py — multi-hop supporting-passage recall from a sweep jsonl.

The sweep CSV only carries CUAD-style retrieval quality (recall_at_k_50, ILD, RR).
The multi-hop headline numbers (recall@k, full_support@k per arm) are NOT in that
CSV; they are computed here from the retrieval jsonl the sweep already exported,
which carries per-query ``gold_support`` and ``retrieved_doc_ids={l2,mmr}``.

No model, no torch — pure Python, runs instantly. Groups rows by
(mmr_lambda, tau_dup, gamma, top_k) so each combo gets its own L2 vs MMR row.

Usage:
  python score_supporting_recall.py \
      sweep_exports/<run>/retrieval_<run>.jsonl [--ks 2,5,10] [--out recall.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from typing import Any, Dict, List, Optional

from evaluation.metrics.supporting_recall import evaluate_supporting_recall, aggregate


def _arm_ids(rec: Dict[str, Any], arm: str) -> List[int]:
    rdi = rec.get("retrieved_doc_ids")
    if isinstance(rdi, dict) and arm in rdi and rdi[arm] is not None:
        return [int(x) for x in rdi[arm]]
    return []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl", help="retrieval jsonl exported by sweep_eval")
    ap.add_argument("--ks", default="2,5,10", help="comma k list, e.g. 2,5,10")
    ap.add_argument("--out", default=None, help="optional CSV output path")
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    ks_full = tuple(ks) + (None,)  # add 'all'

    rows = [json.loads(l) for l in open(args.jsonl, encoding="utf-8") if l.strip()]
    # group by combo
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = (r.get("mmr_lambda"), r.get("tau_dup"), r.get("gamma"), r.get("top_k"))
        groups[key].append(r)

    out_rows: List[Dict[str, Any]] = []
    for key, recs in sorted(groups.items(), key=lambda kv: [str(x) for x in kv[0]]):
        lam, tau, gam, tk = key
        n_gold = sum(1 for r in recs if r.get("gold_support"))
        l2_rows, mmr_rows = [], []
        for r in recs:
            gs = r.get("gold_support")
            if not gs:
                continue
            l2_rows.append(evaluate_supporting_recall(_arm_ids(r, "l2"), gs, ks=ks_full))
            mmr_rows.append(evaluate_supporting_recall(_arm_ids(r, "mmr"), gs, ks=ks_full))
        l2_agg, mmr_agg = aggregate(l2_rows), aggregate(mmr_rows)
        row = {"mmr_lambda": lam, "tau_dup": tau, "gamma": gam, "top_k": tk,
               "n_queries": len(recs), "n_with_gold": n_gold}
        for k in ks:
            row[f"l2_recall@{k}"] = round(l2_agg.get(f"recall@{k}", float("nan")), 4)
            row[f"mmr_recall@{k}"] = round(mmr_agg.get(f"recall@{k}", float("nan")), 4)
            row[f"l2_full@{k}"] = round(l2_agg.get(f"full_support@{k}", float("nan")), 4)
            row[f"mmr_full@{k}"] = round(mmr_agg.get(f"full_support@{k}", float("nan")), 4)
        out_rows.append(row)

    # print
    for row in out_rows:
        print(f"\ncombo λ={row['mmr_lambda']} τ={row['tau_dup']} γ={row['gamma']} "
              f"top_k={row['top_k']}  (n={row['n_queries']}, gold={row['n_with_gold']})")
        print(f"  {'k':>4} | {'L2 recall':>10} {'MMR recall':>10} | "
              f"{'L2 full':>8} {'MMR full':>8}   Δfull")
        for k in ks:
            lf, mf = row[f"l2_full@{k}"], row[f"mmr_full@{k}"]
            print(f"  {k:>4} | {row[f'l2_recall@{k}']:>10} {row[f'mmr_recall@{k}']:>10} | "
                  f"{lf:>8} {mf:>8}   {mf - lf:+.4f}")

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
        print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()
