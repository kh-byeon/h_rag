#!/usr/bin/env python3
"""Paper tables for the n=3500 run: retrieval quality, and degradation under quantization.

    python scripts/make_tables_q3500.py            # markdown to stdout + LaTeX to figures/
    python scripts/make_tables_q3500.py --no_tex

Table 1  retrieval: L2 vs MMR, absolute and relative. Measured before generation,
         so it does not depend on the LLM or its precision.
Table 2  generation: how far F1 / EM / faithfulness fall from int8 to int4, per arm.
         The difference-in-differences row is reported WITH its interval because it
         is not significant -- see docs/RESULTS_q3500.md section 4.
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

from scripts.make_paper_tables import PAPER_RUN, RUN, QUANTS, check_quants

RUN_ID = RUN["run_id"]
rng = np.random.default_rng(20260829)


def boot(d, B=40000):
    d = np.asarray(d, float); d = d[~np.isnan(d)]
    m = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return d.mean(), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def retrieval_rows():
    sw = pd.read_csv(os.path.join(ROOT, "sweep_exports", RUN_ID,
                                  "sweep_results_full_grid.csv")).iloc[0]
    # per-query, for the CI and McNemar on the two testable rows
    path = glob.glob(os.path.join(ROOT, "sweep_exports", RUN_ID, "retrieval_*.jsonl"))[0]
    full_l2, full_mmr = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            gs = [int(g) for g in (d.get("gold_support") or [])]
            ids = d.get("retrieved_doc_ids") or {}
            k = int(d.get("top_k") or 5)
            for arm, acc in (("l2", full_l2), ("mmr", full_mmr)):
                got = [int(x) for x in (ids.get(arm) or [])][:k]
                acc.append(int(bool(gs) and all(g in got for g in gs)))
    a, b = np.array(full_l2), np.array(full_mmr)
    mu, lo, hi = boot(b - a)
    only_l2 = int(((a == 1) & (b == 0)).sum()); only_mmr = int(((a == 0) & (b == 1)).sum())
    p = stats.binomtest(only_mmr, only_l2 + only_mmr, 0.5).pvalue

    rows = [("Full support @5 (both gold paragraphs)", float(sw["l2_full@5"]), float(sw["mmr_full@5"]),
             f"[{lo:+.4f}, {hi:+.4f}]", f"McNemar {only_mmr} vs {only_l2}, $P={p:.1e}$"),
            ("Gold recall @5", float(sw["l2_recall@5"]), float(sw["mmr_recall@5"]), "", ""),
            ("Passage recall", float(sw["l2_recall"]), float(sw["mmr_recall"]), "", ""),
            ("ROUGE-L of retrieved text", float(sw["l2_rouge_l"]), float(sw["mmr_rouge_l"]), "", ""),
            ("ILD (intra-list diversity) $\\uparrow$", float(sw["l2_ild"]), float(sw["mmr_ild"]), "", "aggregate only"),
            ("RR (near-duplicate rate) $\\downarrow$", float(sw["l2_rr"]), float(sw["mmr_rr"]), "", "aggregate only"),
            ("Centroid breadth $\\uparrow$", float(sw["l2_breadth"]), float(sw["mmr_breadth"]), "", "aggregate only")]
    return rows


def generation_rows():
    pq = pd.read_csv(os.path.join(ROOT, RUN["perq"]))
    fa = pd.read_csv(os.path.join(ROOT, RUN["faith"]))
    out = []
    for nm in ("F1", "EM", "faithfulness"):
        per_arm, did_src = {}, {}
        for q in QUANTS:
            if nm == "faithfulness":
                w = fa[fa.source == q].pivot_table(index="query_idx", columns="arm",
                                                   values="faithfulness_mean_entail").dropna()
                per_arm[q] = (w["l2"], w["mmr"])
            else:
                s = pq[pq.source == q].set_index("query_idx")
                per_arm[q] = (s[f"l2_{nm.lower()}"], s[f"mmr_{nm.lower()}"])
            did_src[q] = per_arm[q]
        stats_row = {}
        ref = QUANTS[0]
        for i, arm in enumerate(("L2", "MMR")):
            absv = {q: float(np.mean(per_arm[q][i])) for q in QUANTS}
            relv = {q: 100 * (absv[q] - absv[ref]) / absv[ref] for q in QUANTS}
            stats_row[arm] = {"abs": absv, "rel": relv}
        lo_q, hi_q = QUANTS[0], QUANTS[-1]
        frame = pd.concat(
            {
                "l_ref": did_src[lo_q][0], "m_ref": did_src[lo_q][1],
                "l_last": did_src[hi_q][0], "m_last": did_src[hi_q][1],
            },
            axis=1,
        ).dropna()
        dL = frame["l_last"].to_numpy(float) - frame["l_ref"].to_numpy(float)
        dM = frame["m_last"].to_numpy(float) - frame["m_ref"].to_numpy(float)
        did = dM - dL
        mu, lo, hi = boot(did)
        p = float(stats.ttest_1samp(did[~np.isnan(did)], 0).pvalue)
        out.append((nm, stats_row, mu, lo, hi, p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_tex", action="store_true")
    args = ap.parse_args()
    check_quants()
    ret, gen = retrieval_rows(), generation_rows()

    # ---------------------------------------------------------------- table 1 --
    print(f"\n### Table 1 - Retrieval quality (n={RUN['n']}, LLM-independent)\n")
    print("| Metric | L2 | MMR | Delta | Delta % |")
    print("|---|---|---|---|---|")
    for nm, l2, mmr, _ci, _note in ret:
        d = mmr - l2
        star = " *" if "aggregate" in _note else ""
        print(f"| {nm}{star} | {l2:.4f} | {mmr:.4f} | {d:+.4f} | {100*d/l2:+.1f}% |")
    print("\n\\* corpus aggregate; no per-query interval available.")

    # ---------------------------------------------------------------- table 2 --
    steps = list(QUANTS)
    head = " | ".join(f"{q}" for q in steps)
    drops = " | ".join(f"Delta % ({steps[0]}->{q})" for q in steps[1:])
    print(f"\n### Table 2 - Degradation under quantization (n={RUN['n']})\n")
    print(f"| Metric | Arm | {head} | {drops} |")
    print("|---|---|" + "---|" * (len(steps) + len(steps) - 1))
    for nm, sr, _mu, _lo, _hi, _p in gen:
        for arm in ("L2", "MMR"):
            vals = " | ".join(f"{sr[arm]['abs'][q]:.4f}" for q in steps)
            rel = " | ".join(f"**{sr[arm]['rel'][q]:+.2f}%**" for q in steps[1:])
            print(f"| {nm if arm=='L2' else ''} | {arm} | {vals} | {rel} |")

    # ------------------------------------- statistics, for the BODY TEXT only --
    print("\n" + "=" * 74)
    print("FOR THE BODY TEXT (deliberately kept out of the tables)")
    print("=" * 74)
    for nm, sr, mu, lo, hi, p in gen:
        last = steps[-1]
        rl, rm = sr["L2"]["rel"][last], sr["MMR"]["rel"][last]
        print(f"\n  {nm}: the L2 baseline falls {rl/rm:.2f}x further than MMR "
              f"({rl:+.2f}% vs {rm:+.2f}%), but the difference-in-differences is not")
        print(f"    resolved: {mu:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}], P = {p:.3f}.")
    print("\n  Retrieval: full support@5 rises from "
          f"{ret[0][1]:.4f} to {ret[0][2]:.4f} "
          f"({100*(ret[0][2]-ret[0][1])/ret[0][1]:+.1f}%); {ret[0][4]}.")
    print("\n  Report the DiD intervals in the text. Omitting them and quoting only the")
    print("  ratio invites the reviewer to compute the interaction test themselves.")

    if args.no_tex:
        return
    tex = [r"% Generated by scripts/make_tables_q3500.py -- run " + RUN["run_id"],
           r"% Statistics live in the body text, not here: see the script's stdout.",
           r"\begin{table}[t]", r"\centering",
           r"\caption{Retrieval quality, $n=" + str(RUN["n"]) + r"$ HotpotQA distractor queries, $k=5$. "
           r"Measured before generation and therefore independent of the language model "
           r"and its precision. Rows marked $\ast$ are stored as corpus aggregates and "
           r"have no per-query interval.}",
           r"\begin{tabular}{lrrrr}", r"\toprule",
           r"Metric & L2 & MMR & $\Delta$ & $\Delta$\,\% \\", r"\midrule"]
    for nm, l2, mmr, _ci, note in ret:
        d = mmr - l2
        star = r"$^{\ast}$" if "aggregate" in note else ""
        tex.append(f"{nm}{star} & {l2:.4f} & {mmr:.4f} & {d:+.4f} & {100*d/l2:+.1f} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
            r"\begin{table}[t]", r"\centering",
            r"\caption{Answer quality under quantization, $n=" + str(RUN["n"]) + r"$. "
            r"The L2 baseline degrades further than MMR on every metric; the "
            r"difference-in-differences is reported in the text.}",
            r"\begin{tabular}{ll" + "r" * (len(steps) * 2 - 1) + r"}", r"\toprule",
            "Metric & Arm & " + " & ".join(steps) + " & "
            + " & ".join(fr"$\Delta$\,\% ({steps[0]}$\to${q})" for q in steps[1:]) + r" \\",
            r"\midrule"]
    for nm, sr, *_ in gen:
        for arm in ("L2", "MMR"):
            vals = " & ".join(f"{sr[arm]['abs'][q]:.4f}" for q in steps)
            rel = " & ".join(f"{sr[arm]['rel'][q]:+.2f}" for q in steps[1:])
            tex.append(f"{nm if arm=='L2' else ''} & {arm} & {vals} & {rel} \\\\")
        if nm != gen[-1][0]:
            tex.append(r"\addlinespace")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out = os.path.join(ROOT, "figures", f"tables_{PAPER_RUN}.tex")
    open(out, "w", encoding="utf-8").write("\n".join(tex) + "\n")
    print(f"\nLaTeX (booktabs) -> figures/tables_{PAPER_RUN}.tex")


if __name__ == "__main__":
    main()
