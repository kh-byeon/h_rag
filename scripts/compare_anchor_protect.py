#!/usr/bin/env python3
"""Compare retrieval-only runs that differ in anchor_protect_n.

    python scripts/compare_anchor_protect.py \
        sweep_exports/<run_a>:ap0 sweep_exports/<run_b>:ap1 ...

For each run: full support@5 and gold recall@5 for both arms, the MMR-vs-L2
discordance (McNemar), and the diversity aggregates from the sweep CSV. The L2 arm
is invariant to anchor_protect_n by construction -- it is printed as a correctness
check and must be identical across runs.

No LLM, no GPU. Retrieval metrics only.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

rng = np.random.default_rng(20260827)


def boot(d, B=20000):
    d = np.asarray(d, float)
    n = len(d)
    m = d[rng.integers(0, n, (B, n))].mean(1)
    return d.mean(), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def load(run_dir):
    hits = glob.glob(os.path.join(run_dir, "retrieval_*.jsonl"))
    if not hits:
        raise SystemExit(f"no retrieval_*.jsonl in {run_dir}")
    rows = []
    with open(hits[0], encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            gs = [int(g) for g in (d.get("gold_support") or [])]
            ids = d.get("retrieved_doc_ids") or {}
            k = int(d.get("top_k") or 5)
            row = {"query_idx": int(d["query_idx"])}
            for arm in ("l2", "mmr"):
                got = [int(x) for x in (ids.get(arm) or [])][:k]
                row[f"{arm}_full"] = int(bool(gs) and all(g in got for g in gs))
                row[f"{arm}_rec"] = (sum(g in got for g in gs) / len(gs)) if gs else np.nan
            rows.append(row)
    df = pd.DataFrame(rows)
    csv = os.path.join(run_dir, "sweep_results_full_grid.csv")
    agg = pd.read_csv(csv).iloc[0].to_dict() if os.path.isfile(csv) else {}
    return df, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run_dir or 'run_dir:label'")
    args = ap.parse_args()

    items = []
    for it in args.runs:
        if ":" in it and not os.path.exists(it):
            path, _, label = it.rpartition(":")
        else:
            path, label = it, os.path.basename(it.rstrip("/\\"))
        items.append((path, label))

    print(f"{'label':>10} {'n':>5} {'L2 full@5':>10} {'MMR full@5':>11} {'delta':>9} "
          f"{'95% CI':>20} {'MMR-only':>9} {'L2-only':>8} {'McNemar p':>10}")
    base_l2 = None
    keep = {}
    for path, label in items:
        df, agg = load(path)
        keep[label] = (df, agg)
        b = int(((df.l2_full == 1) & (df.mmr_full == 0)).sum())
        c = int(((df.l2_full == 0) & (df.mmr_full == 1)).sum())
        p = stats.binomtest(c, b + c, 0.5).pvalue if (b + c) else float("nan")
        mu, lo, hi = boot((df.mmr_full - df.l2_full).values)
        print(f"{label:>10} {len(df):>5} {df.l2_full.mean():>10.4f} {df.mmr_full.mean():>11.4f} "
              f"{mu:>+9.4f} [{lo:+7.4f},{hi:+7.4f}] {c:>9} {b:>8} {p:>10.3g}")
        if base_l2 is None:
            base_l2 = df.l2_full.mean()
        elif abs(df.l2_full.mean() - base_l2) > 1e-12:
            print(f"{'':>10}  WARNING: the L2 arm moved ({df.l2_full.mean():.4f} vs "
                  f"{base_l2:.4f}). It must be invariant to anchor_protect_n -- "
                  f"something other than the protection differs between these runs.")

    print(f"\n{'label':>10} {'MMR recall@5':>13} {'MMR ILD':>9} {'MMR RR':>9} {'MMR breadth':>12}"
          f"   (diversity must not collapse -- that is the cost side)")
    for _, label in items:
        df, agg = keep[label]
        print(f"{label:>10} {df.mmr_rec.mean():>13.4f} {agg.get('mmr_ild', float('nan')):>9.4f} "
              f"{agg.get('mmr_rr', float('nan')):>9.4f} {agg.get('mmr_breadth', float('nan')):>12.4f}")

    if len(items) >= 2:
        (pa, la), (pb, lb) = items[0], items[1]
        a, b_ = keep[la][0], keep[lb][0]
        m = a[["query_idx", "mmr_full"]].merge(
            b_[["query_idx", "mmr_full"]], on="query_idx", suffixes=("_a", "_b"))
        gained = int(((m.mmr_full_a == 0) & (m.mmr_full_b == 1)).sum())
        lost = int(((m.mmr_full_a == 1) & (m.mmr_full_b == 0)).sum())
        p = stats.binomtest(gained, gained + lost, 0.5).pvalue if (gained + lost) else float("nan")
        print(f"\nPaired change {la} -> {lb} on the MMR arm: "
              f"{gained} queries gained full support, {lost} lost it, McNemar p = {p:.3g}")
        if gained + lost == 0:
            print("  Zero queries changed. The tau_dup gate is not firing at this tau, so "
                  "anchor protection is a no-op here -- the residual losses come from the "
                  "MMR score's diversity term, not from the dedup gate.")


if __name__ == "__main__":
    main()
