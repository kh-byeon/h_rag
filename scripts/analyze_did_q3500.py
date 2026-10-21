#!/usr/bin/env python3
"""The pre-registered test: does MMR degrade less from QUANTS[0] to QUANTS[-1]?

    PAPER_RUN=q1000 python scripts/analyze_did_q3500.py
    PAPER_RUN=q3500 python scripts/analyze_did_q3500.py --run_id <run_id>

Reads combos, per-query CSVs and the official EM/F1 CSV from RUN (see
scripts/make_paper_tables.py). The DiD contrast is always first→last of
RUN["quants"] — fp16→int4 on the n=1000 run, int8→int4 on n=3500.

Query-clustered throughout: one value per query_idx, then a paired bootstrap over
the query means (20,000 resamples, fixed seed). n = queries, never rows.

PRE-REGISTERED DECISION RULE (fixed before the n=3500 run; see run_q3500_did.sh):
  * EM or F1 DiD 95% CI excludes 0 -> the quantization-robustness claim is
    supported and becomes the paper's main result.
  * neither -> stop. Fall back to the operating-point framing (RESULTS §4). Do not
    escalate n again and do not try further reformulations; five have failed.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.make_paper_tables import RUN, QUANTS, check_quants

B, SEED = 20000, 20260828
rng = np.random.default_rng(SEED)


def boot(d):
    d = np.asarray(d, float); d = d[~np.isnan(d)]
    m = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return d.mean(), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def perquery(root):
    from scripts.collect_em_f1 import _answers, _golds
    from evaluation.metrics.answer_em_f1 import extract_final_answer, score_answer
    frames = []
    for q in QUANTS:
        hits = sorted(glob.glob(os.path.join(root, RUN["combos"].format(q=q), "llm_combo_*.json")))
        if not hits:
            raise SystemExit(f"missing {RUN['combos'].format(q=q)} -- run the sweep first")
        with open(hits[0], encoding="utf-8") as fh:
            payload = json.load(fh)
        rows = []
        for rec in payload.get("per_query_results") or []:
            g = _golds(rec)
            if not g:
                continue
            a, b = _answers(rec)
            s1 = score_answer(extract_final_answer(a or ""), g)
            s2 = score_answer(extract_final_answer(b or ""), g)
            rows.append(dict(query_idx=rec.get("query_idx"), source=q,
                             l2_em=s1["em"], mmr_em=s2["em"],
                             l2_f1=s1["f1"], mmr_f1=s2["f1"]))
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    check_quants()

    pq = perquery(ROOT)
    out_path = args.out or RUN["perq"]
    pq.to_csv(os.path.join(ROOT, out_path), index=False)
    off = pd.read_csv(os.path.join(ROOT, RUN["official"]))
    for q in QUANTS:                       # same identity check as analyze_q1000.py
        s, o = pq[pq.source == q], off[off.source == q].iloc[0]
        for arm in ("l2", "mmr"):
            assert abs(s[f"{arm}_f1"].mean() - float(o[f"{arm}_f1"])) < 1e-9, \
                f"{q}/{arm} per-query F1 does not reproduce {RUN['official']}"
    n = pq.query_idx.nunique()
    if n != RUN["n"]:
        raise SystemExit(f"n = {n} queries, expected RUN n={RUN['n']}")
    print(f"n = {n} queries   (per-query EM/F1 reproduces {RUN['official']} to 1e-9)\n")

    w = pq.pivot_table(index="query_idx", columns="source",
                       values=["l2_em", "mmr_em", "l2_f1", "mmr_f1"])
    faith = pd.read_csv(os.path.join(ROOT, RUN["faith"]))
    wf = faith.pivot_table(index="query_idx", columns=["source", "arm"],
                           values="faithfulness_mean_entail").dropna()

    print("=" * 76)
    print("WITHIN-PRECISION: MMR vs L2   (established at n=1000; expected to hold)")
    print("=" * 76)
    print(f"{'quant':>6} {'metric':>6} {'L2':>8} {'MMR':>8} {'delta':>9} {'95% CI':>21} {'P':>9}")
    for q in QUANTS:
        s = pq[pq.source == q]
        for nm, c in (("F1", "f1"), ("EM", "em")):
            mu, lo, hi = boot((s[f"mmr_{c}"] - s[f"l2_{c}"]).values)
            p = stats.ttest_rel(s[f"mmr_{c}"], s[f"l2_{c}"]).pvalue
            print(f"{q:>6} {nm:>6} {s[f'l2_{c}'].mean():8.4f} {s[f'mmr_{c}'].mean():8.4f} "
                  f"{mu:+9.4f} [{lo:+7.4f},{hi:+7.4f}] {p:9.2e}")

    q_hi, q_lo = QUANTS[0], QUANTS[-1]
    print()
    print("=" * 76)
    print(f"THE PRE-REGISTERED TEST: does MMR degrade LESS from {q_hi} to {q_lo}?")
    print("=" * 76)
    verdict = {}
    rows = [
        ("EM", w[("l2_em", q_hi)], w[("l2_em", q_lo)], w[("mmr_em", q_hi)], w[("mmr_em", q_lo)]),
        ("F1", w[("l2_f1", q_hi)], w[("l2_f1", q_lo)], w[("mmr_f1", q_hi)], w[("mmr_f1", q_lo)]),
        ("faithfulness", wf[(q_hi, "l2")], wf[(q_lo, "l2")], wf[(q_hi, "mmr")], wf[(q_lo, "mmr")]),
    ]
    for nm, l8, l4, m8, m4 in rows:
        dL = (np.asarray(l4, float) - np.asarray(l8, float))
        dM = (np.asarray(m4, float) - np.asarray(m8, float))
        did = dM - dL
        mu, lo, hi = boot(did)
        t = stats.ttest_1samp(did[~np.isnan(did)], 0)
        sig = lo > 0 or hi < 0
        verdict[nm] = sig
        ratio = dL.mean() / dM.mean() if dM.mean() else float("nan")
        print(f"\n  {nm}")
        print(f"    L2 degrades  {dL.mean():+.4f}      MMR degrades {dM.mean():+.4f}   "
              f"(L2 falls {ratio:.2f}x further)")
        print(f"    DiD {mu:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]   "
              f"t({int(np.sum(~np.isnan(did)))-1}) = {t.statistic:.2f}   P = {t.pvalue:.4f}"
              f"   {'*** SUPPORTED' if sig else 'not resolved'}")

    print("\n" + "=" * 76)
    if verdict.get("EM") or verdict.get("F1"):
        which = " and ".join(k for k in ("EM", "F1") if verdict.get(k))
        print(f"VERDICT: SUPPORTED on {which}.")
        print("  The quantization-robustness claim now has a defensible form. Write it as a")
        print("  difference-in-differences with the CI and P value above, state the")
        print("  pre-registration, and report the metrics where it did NOT reach")
        print("  significance in the same table -- partial support reported in full is")
        print("  much stronger than a selected subset.")
    else:
        print("VERDICT: NOT RESOLVED. The pre-registered rule says stop here.")
        print("  Do not escalate n again -- if the true effect is near the lower half of")
        print("  the n=1000 interval, even the full 7,405-question set would not settle it")
        print("  (see the power table in docs/RESULTS_q1000.md §5).")
        print("  Fall back to the operating-point framing in docs/RESULTS_q1000.md §4:")
        print("  at int4 MMR beats L2-only, and recovers the accuracy cost of int4.")
        print("  Report this DiD with its CI as a stated limitation, not as a claim.")
    print("=" * 76)


if __name__ == "__main__":
    main()
