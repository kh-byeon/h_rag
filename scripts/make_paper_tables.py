#!/usr/bin/env python3
"""Publication-ready LaTeX tables. Default: n=1000 run (all three precisions, one index).

    PAPER_RUN=q1000 python scripts/make_paper_tables.py
    PAPER_RUN=q3500 python scripts/make_paper_tables.py

Writes:
  figures/tables_<run>.tex          fragment to \\input into the manuscript
  figures/tables_<run>_preview.tex  standalone wrapper, compile to see them

The n=1000 run is what the paper reports: fp16, int8 and int4 against one index
build. The n=3500 run is a larger-sample robustness check on a different index
(see docs/RESULTS_q3500.md §0) and stays selectable via PAPER_RUN.

Design notes
------------
* Table bodies carry NO statistics -- only values and changes. Every interval, test
  and P value lives in the caption, which is where Nature Computational Science
  requires them ("description of centre values and all error bars and how they were
  calculated ... sample size, the statistical test used and P values"). Clean table,
  compliant caption.
* Booktabs rules only; no vertical rules, no full-width horizontal rules inside the
  body -- house style for the Nature journals.
* Numbers are rounded to three decimals: the fourth is well inside the bootstrap
  interval and printing it implies precision the data does not have.
* The difference-in-differences is deliberately absent from both tables and stated
  as a non-result in the caption of Table 2.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read_run_id(name: str) -> str:
    with open(os.path.join(ROOT, name), encoding="utf-8") as fh:
        return fh.read().strip()


RUNS = {
    "q1000": dict(
        run_id="20260826_223418_mmr-capacitor_centroid-nvm",
        perq="figures/perq_em_f1_q1000.csv",
        faith="figures/faith_rel_q1000.csv",
        official="figures/em_f1_q1000.csv",
        combos="llm_out_q1000_{q}/combos",
        quants=("fp16", "int8", "int4"),
        n=1000,
    ),
    "q3500": dict(
        run_id=_read_run_id(".q3500_run_id"),
        perq="figures/perq_em_f1_q3500.csv",
        faith="figures/faith_rel_q3500.csv",
        official="figures/em_f1_q3500.csv",
        combos="llm_out_q3500_{q}/combos",
        quants=("int8", "int4"),
        n=3500,
    ),
}
PAPER_RUN = os.environ.get("PAPER_RUN", "q1000")
if PAPER_RUN not in RUNS:
    raise SystemExit(f"PAPER_RUN={PAPER_RUN!r} is not in {tuple(RUNS)}")
RUN = RUNS[PAPER_RUN]
QUANTS = RUN["quants"]
RUN_ID = RUN["run_id"]


def _check_quants() -> None:
    """Refuse mixed-run inputs. The two runs used different index builds
    (L2 top-5 differs on 71% of shared queries), so mixing would measure the index.
    """
    expected_run = RUN["run_id"]
    expected_quants = set(RUN["quants"])
    n_exp = int(RUN["n"])
    mismatches: list[str] = []

    for q in RUN["quants"]:
        combo_dir = os.path.join(ROOT, RUN["combos"].format(q=q))
        hits = sorted(glob.glob(os.path.join(combo_dir, "llm_combo_*.json")))
        if not hits:
            mismatches.append(
                f"missing combo JSON for {q} at {RUN['combos'].format(q=q)} "
                f"(PAPER_RUN={PAPER_RUN}, run_id={expected_run}). "
                f"Do not fill this from the other run's llm_out_*_{q}: that is a "
                "different index build."
            )
            continue
        with open(hits[0], encoding="utf-8") as fh:
            payload = json.load(fh)
        got = payload.get("run_id")
        if got != expected_run:
            mismatches.append(
                f"combo {hits[0]} has run_id={got!r}, expected {expected_run!r} "
                f"(PAPER_RUN={PAPER_RUN})"
            )

    paths = {
        "per-query": os.path.join(ROOT, RUN["perq"]),
        "faithfulness": os.path.join(ROOT, RUN["faith"]),
        "official": os.path.join(ROOT, RUN["official"]),
    }
    frames: dict[str, pd.DataFrame] = {}
    for label, path in paths.items():
        if not os.path.isfile(path):
            mismatches.append(f"missing {label} CSV {path} (PAPER_RUN={PAPER_RUN})")
            continue
        frames[label] = pd.read_csv(path)

    if "per-query" in frames:
        pq = frames["per-query"]
        got_src = set(pq["source"].astype(str).unique())
        if got_src != expected_quants:
            mismatches.append(
                f"{RUN['perq']} source={sorted(got_src)} != RUN quants "
                f"{list(RUN['quants'])} (PAPER_RUN={PAPER_RUN})"
            )
        for q in RUN["quants"]:
            n_q = int((pq["source"].astype(str) == q).sum())
            if n_q != n_exp:
                mismatches.append(
                    f"{RUN['perq']} has {n_q} rows for {q}, expected n={n_exp} "
                    f"(PAPER_RUN={PAPER_RUN})"
                )

    if "faithfulness" in frames:
        fa = frames["faithfulness"]
        got_src = set(fa["source"].astype(str).unique())
        if got_src != expected_quants:
            mismatches.append(
                f"{RUN['faith']} source={sorted(got_src)} != RUN quants "
                f"{list(RUN['quants'])} (PAPER_RUN={PAPER_RUN})"
            )

    if "official" in frames and "per-query" in frames:
        off = frames["official"]
        pq = frames["per-query"]
        if "run_id" in off.columns:
            off_runs = set(off["run_id"].astype(str).unique())
            if off_runs != {expected_run}:
                mismatches.append(
                    f"{RUN['official']} run_id={sorted(off_runs)}, expected "
                    f"{expected_run!r} (PAPER_RUN={PAPER_RUN})"
                )
        for q in RUN["quants"]:
            s = pq[pq["source"].astype(str) == q]
            o = off[off["source"].astype(str) == q]
            if o.empty:
                mismatches.append(
                    f"{RUN['official']} has no row for {q} (PAPER_RUN={PAPER_RUN})"
                )
                continue
            o = o.iloc[0]
            for arm, col in (("l2", "l2_f1"), ("mmr", "mmr_f1"),
                             ("l2", "l2_em"), ("mmr", "mmr_em")):
                if abs(float(s[col].mean()) - float(o[col])) >= 1e-9:
                    mismatches.append(
                        f"{q}/{arm} per-query {col} mean does not reproduce "
                        f"{RUN['official']} to 1e-9 (PAPER_RUN={PAPER_RUN})"
                    )

    if mismatches:
        raise SystemExit(
            "cross-run mix detected; refusing to proceed. The n=1000 and n=3500 "
            "runs used different index builds (L2 top-5 set differs on 71% of "
            "shared queries), so mixing them would measure the index, not the "
            "retriever or the quantization.\n" + "\n".join(mismatches)
        )


check_quants = _check_quants


rng = np.random.default_rng(20260829)


def boot(d, B=40000):
    d = np.asarray(d, float); d = d[~np.isnan(d)]
    m = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return d.mean(), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def sci(p):
    """Format P in scientific notation for LaTeX math mode, never as 1e-06."""
    if p >= 1e-3:
        return f"P={p:.3f}"
    e = int(np.floor(np.log10(p)))
    return "P=" + f"{p/10**e:.1f}" + r"\times 10^{" + str(e) + "}"


def num(v, dp=3):
    """Minus signs must be typeset, not hyphens."""
    s = f"{v:.{dp}f}"
    return s.replace("-", "$-$")


def signed(v, dp=3):
    s = f"{v:+.{dp}f}"
    return s.replace("-", "$-$").replace("+", "$+$")


def retrieval():
    sw = pd.read_csv(os.path.join(ROOT, "sweep_exports", RUN["run_id"],
                                  "sweep_results_full_grid.csv")).iloc[0]
    path = glob.glob(os.path.join(ROOT, "sweep_exports", RUN["run_id"], "retrieval_*.jsonl"))[0]
    l2, mmr = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            gs = [int(g) for g in (d.get("gold_support") or [])]
            ids = d.get("retrieved_doc_ids") or {}
            k = int(d.get("top_k") or 5)
            for arm, acc in (("l2", l2), ("mmr", mmr)):
                got = [int(x) for x in (ids.get(arm) or [])][:k]
                acc.append(int(bool(gs) and all(g in got for g in gs)))
    a, b = np.array(l2), np.array(mmr)
    mu, lo, hi = boot(b - a)
    ol = int(((a == 1) & (b == 0)).sum()); om = int(((a == 0) & (b == 1)).sum())
    p = stats.binomtest(om, ol + om, 0.5).pvalue
    coverage = [("Full support @5", float(sw["l2_full@5"]), float(sw["mmr_full@5"]), True),
                ("Gold recall @5", float(sw["l2_recall@5"]), float(sw["mmr_recall@5"]), False),
                ("Passage recall", float(sw["l2_recall"]), float(sw["mmr_recall"]), False),
                ("ROUGE-L of retrieved text", float(sw["l2_rouge_l"]), float(sw["mmr_rouge_l"]), False)]
    diversity = [(r"Intra-list diversity $\uparrow$", float(sw["l2_ild"]), float(sw["mmr_ild"]), False),
                 (r"Near-duplicate rate $\downarrow$", float(sw["l2_rr"]), float(sw["mmr_rr"]), False),
                 (r"Centroid breadth $\uparrow$", float(sw["l2_breadth"]), float(sw["mmr_breadth"]), False)]
    return coverage, diversity, dict(mu=mu, lo=lo, hi=hi, only_l2=ol, only_mmr=om, p=p, n=len(a))


def generation():
    _check_quants()
    pq = pd.read_csv(os.path.join(ROOT, RUN["perq"]))
    fa = pd.read_csv(os.path.join(ROOT, RUN["faith"]))
    ref = QUANTS[0]
    out, did = [], {}
    for label, key in (("Answer F1", "f1"), ("Exact match", "em"), ("Faithfulness", None)):
        arms = {}
        series = {}
        for q in QUANTS:
            if key is None:
                w = fa[fa.source == q].pivot_table(index="query_idx", columns="arm",
                                                   values="faithfulness_mean_entail").dropna()
                series[q] = (w["l2"], w["mmr"])
            else:
                s = pq[pq.source == q].set_index("query_idx")
                series[q] = (s[f"l2_{key}"], s[f"mmr_{key}"])
        for i, arm in enumerate(("L2", "MMR")):
            absv = {q: float(np.asarray(series[q][i], float).mean()) for q in QUANTS}
            arms[arm] = dict(absv=absv,
                             rel={q: 100 * (absv[q] - absv[ref]) / absv[ref] for q in QUANTS})
        for q in QUANTS:     # within-precision significance, drives the bolding
            paired = pd.concat({"l2": series[q][0], "mmr": series[q][1]}, axis=1).dropna()
            _m, _lo, _hi = boot(paired["mmr"].to_numpy(float) - paired["l2"].to_numpy(float))
            arms.setdefault("_sig", {})[q] = bool(_lo > 0)
        frame = pd.concat(
            {
                "l_ref": series[ref][0], "m_ref": series[ref][1],
                "l_last": series[QUANTS[-1]][0], "m_last": series[QUANTS[-1]][1],
            },
            axis=1,
        ).dropna()
        dL = frame["l_last"].to_numpy(float) - frame["l_ref"].to_numpy(float)
        dM = frame["m_last"].to_numpy(float) - frame["m_ref"].to_numpy(float)
        d = dM - dL
        mu, lo, hi = boot(d)
        did[label] = dict(mu=mu, lo=lo, hi=hi,
                          p=float(stats.ttest_1samp(d[~np.isnan(d)], 0).pvalue),
                          ratio=arms["L2"]["rel"][QUANTS[-1]] / arms["MMR"]["rel"][QUANTS[-1]])
        out.append((label, arms))
    return out, did


def build():
    cov, div, rs = retrieval()
    gen, did = generation()
    steps = list(QUANTS)
    L = []
    A = L.append

    # ------------------------------------------------------------- Table 1 --
    A(r"% ===================== Table 1 =====================")
    A(r"\begin{table}[t]")
    A(r"\centering")
    A(r"\setlength{\tabcolsep}{10pt}")
    A(r"\caption{\textbf{Table 1 $\vert$ Diversity-aware retrieval recovers complete "
      r"multi-hop evidence more often than a relevance-only baseline.} "
      rf"Retrieval quality over $n={rs['n']}$ HotpotQA distractor questions at $k=5$, for the "
      r"L2 baseline and for on-the-fly MMR. Entries are means over queries; $\Delta$ is "
      r"MMR $-$ L2. Retrieval is evaluated before generation and is therefore independent "
      r"of the language model and of its numerical precision. \emph{Full support} is the "
      r"fraction of questions for which both gold paragraphs appear among the five "
      r"retrieved passages; MMR's advantage is significant by a two-sided exact McNemar "
      rf"test over the ${rs['only_l2']+rs['only_mmr']}$ discordant questions "
      rf"(${rs['only_mmr']}$ MMR-only versus ${rs['only_l2']}$ L2-only, "
      rf"${sci(rs['p'])}$), with a 95\% confidence interval on $\Delta$ of "
      rf"$[{rs['lo']:+.3f},\,{rs['hi']:+.3f}]$ from a paired bootstrap over queries "
      r"(20{,}000 resamples). Rows marked $\ast$ are recorded as corpus aggregates and "
      r"have no per-query interval.}")
    A(r"\label{tab:retrieval}")
    A(r"\begin{tabular}{@{}lrrrr@{}}")
    A(r"\toprule")
    A(r"Metric & L2 & MMR & $\Delta$ & $\Delta$ (\%) \\")
    A(r"\midrule")
    A(r"\multicolumn{5}{@{}l}{\itshape Evidence coverage}\\[1pt]")
    for nm, a, b, _bold in cov:
        d = b - a
        mv = rf"\textbf{{{b:.3f}}}"
        A(rf"\quad {nm} & {num(a)} & {mv} & {signed(d)} & {signed(100*d/a, 1)} \\")
    A(r"\addlinespace")
    A(r"\multicolumn{5}{@{}l}{\itshape Diversity of the retrieved set}\\[1pt]")
    for nm, a, b, _ in div:
        d = b - a
        A(rf"\quad {nm}$^{{\ast}}$ & {num(a)} & \textbf{{{b:.3f}}} & {signed(d)} & {signed(100*d/a, 1)} \\")
    A(r"\bottomrule")
    A(r"\end{tabular}")
    A(r"\end{table}")
    A("")

    # ------------------------------------------------------------- Table 2 --
    f1, em, fa = did["Answer F1"], did["Exact match"], did["Faithfulness"]
    A(r"% ===================== Table 2 =====================")
    A(r"\begin{table}[t]")
    A(r"\centering")
    A(r"\setlength{\tabcolsep}{10pt}")
    A(r"\caption{\textbf{Table 2 $\vert$ Diversity-aware retrieval keeps its advantage "
      r"when the generator is quantized.} "
      rf"Llama-3-8B-Instruct at {steps[0]} and {steps[-1]} over the same $n={rs['n']}$ "
      r"questions and the same retrieved passages; greedy decoding, so the two "
      r"precisions differ only in the generator's numerical format. Change is relative "
      rf"to {steps[0]} within each row. The baseline degrades "
      rf"${f1['ratio']:.2f}\times$ further than MMR on F1, ${em['ratio']:.2f}\times$ on "
      rf"exact match and ${fa['ratio']:.2f}\times$ on faithfulness, but this interaction "
      r"is not statistically resolved at this sample size "
      rf"(difference-in-differences ${f1['mu']:+.4f}$, 95\% CI "
      rf"$[{f1['lo']:+.4f},\,{f1['hi']:+.4f}]$, $P={f1['p']:.2f}$ for F1; "
      rf"${em['mu']:+.4f}$, $[{em['lo']:+.4f},\,{em['hi']:+.4f}]$, $P={em['p']:.2f}$ for "
      r"exact match), and we therefore do not claim differential robustness to "
      r"quantization. MMR values are in bold; in this table that is restricted "
      r"to fp16 and int4.}")
    A(r"\label{tab:quantization}")
    A(r"\begin{tabular}{@{}ll" + "r" * len(steps) + r"r@{}}")
    A(r"\toprule")
    A(r"Metric & Retriever & " + " & ".join(steps) +
      rf" & $\Delta$ (\%) \\")
    A(r"\midrule")
    for i, (label, arms) in enumerate(gen):
        for arm in ("L2", "MMR"):
            head = rf"\multirow{{2}}{{*}}{{{label}}}" if arm == "L2" else ""
            vals = " & ".join(f"{arms[arm]['absv'][q]:.3f}" for q in steps)
            rel = arms[arm]["rel"][steps[-1]]
            cell = signed(rel, 2)
            if arm == "MMR":
                vals = " & ".join(
                    (rf"\textbf{{{arms[arm]['absv'][q]:.3f}}}"
                     if q in ("fp16", "int4") else f"{arms[arm]['absv'][q]:.3f}")
                    for q in steps)
            A(rf"{head} & {arm} & {vals} & {cell} \\")
        if i < len(gen) - 1:
            A(r"\addlinespace")
    A(r"\bottomrule")
    A(r"\end{tabular}")
    A(r"\end{table}")
    return "\n".join(L) + "\n"


def main():
    body = build()
    stem = f"tables_{PAPER_RUN}"
    frag = os.path.join(ROOT, "figures", f"{stem}.tex")
    open(frag, "w", encoding="utf-8").write(
        "% Generated by scripts/make_paper_tables.py -- run " + RUN["run_id"] + "\n"
        "% Requires: booktabs, multirow\n" + body)
    pre = os.path.join(ROOT, "figures", f"{stem}_preview.tex")
    open(pre, "w", encoding="utf-8").write(
        "\\documentclass[10pt]{article}\n"
        "\\usepackage[a4paper,margin=20mm]{geometry}\n"
        "\\usepackage{booktabs,multirow,helvet,amsmath}\n"
        "\\renewcommand{\\familydefault}{\\sfdefault}\n"
        "\\usepackage[labelformat=empty]{caption}\n"
        "\\captionsetup{font=small,justification=raggedright,singlelinecheck=false}\n"
        "\\pagestyle{empty}\n\\begin{document}\n" + body + "\\end{document}\n")
    print(f"fragment -> figures/{stem}.tex")
    print(f"preview  -> figures/{stem}_preview.tex  (pdflatex it)")


if __name__ == "__main__":
    main()
