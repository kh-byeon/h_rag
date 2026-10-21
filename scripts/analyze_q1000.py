#!/usr/bin/env python3
"""
Reproduce every published number for the n=1000 distractor run, and write them
to one workbook.

    python scripts/analyze_q1000.py                  # report + figures/q1000_metrics.xlsx
    python scripts/analyze_q1000.py --rebuild        # re-derive per-query EM/F1 from the combo JSONs
    python scripts/analyze_q1000.py --no_xlsx        # console report only

Everything is QUERY-CLUSTERED: each metric is reduced to one value per
``query_idx`` first, then bootstrapped (paired, over queries). n = 1000 queries,
never 6,000 rows -- see HANDOFF_mmr_evaluation.md section 6 for why that
distinction is load-bearing.

Inputs (all products of scripts/run_q1000_distractor.sh):
  figures/faith_rel_q1000.csv                       faithfulness / relevance, per query x arm x quant
  figures/em_f1_q1000.csv                           official per-combo EM/F1 (used as a cross-check)
  figures/perq_em_f1_q1000.csv                      per-query EM/F1 (rebuilt from combos if absent)
  llm_out_q1000_{fp16,int8,int4}/combos/*.json      generations (only read with --rebuild)
  sweep_exports/<RUN_ID>/retrieval_<RUN_ID>.jsonl   retrieval, for gold-support per query
  processed_hotpot/hotpot_queries.json              qtype / level metadata

Requires numpy, pandas, scipy (+ openpyxl for the workbook).
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

RUN_ID = "20260826_223418_mmr-capacitor_centroid-nvm"
QUANTS = ("fp16", "int8", "int4")
B = 20000          # bootstrap resamples
SEED = 20260827    # fixed so the CIs are reproducible to the digit
N_TESTS = 20       # family size used for the Bonferroni column

rng = np.random.default_rng(SEED)


# ----------------------------------------------------------------- helpers --
def boot(d):
    """Paired bootstrap over queries. Returns (mean, lo95, hi95)."""
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 2:
        return (d.mean() if n else np.nan), np.nan, np.nan
    m = d[rng.integers(0, n, (B, n))].mean(1)
    return d.mean(), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired(a, b):
    """One row of the standard comparison: b (MMR) against a (L2)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    d = b - a
    mu, lo, hi = boot(d)
    p_t = float(stats.ttest_rel(b, a).pvalue)
    try:
        p_w = float(stats.wilcoxon(d).pvalue)
    except ValueError:
        p_w = float("nan")
    return dict(n=len(a), l2=a.mean(), mmr=b.mean(), delta=mu, lo=lo, hi=hi,
                p_t=p_t, p_wilcoxon=p_w,
                sig95="yes" if (lo > 0 or hi < 0) else "no",
                bonferroni=("pass" if max(p_t, p_w) < 0.05 / N_TESTS else "fail"))


def need_n(effect, sd, power=0.8):
    """Queries needed for `power` at the observed effect (paired, two-sided)."""
    if not effect or np.isnan(effect):
        return np.nan
    z = 1.959964 + (0.8416212 if power == 0.8 else 1.2815516)
    return int(np.ceil((z * sd / abs(effect)) ** 2))


# -------------------------------------------------------------- data loads --
def rebuild_perquery(root):
    """Per-query EM/F1 straight from the combo JSONs.

    Uses the same scorers as scripts/collect_em_f1.py, so the six arm x quant
    means reproduce figures/em_f1_q1000.csv exactly -- that identity is what
    licenses per-query confidence intervals, and it is asserted below.
    """
    from scripts.collect_em_f1 import _answers, _golds
    from evaluation.metrics.answer_em_f1 import extract_final_answer, score_answer

    frames = []
    for q in QUANTS:
        hits = sorted(glob.glob(os.path.join(root, f"llm_out_q1000_{q}", "combos", "llm_combo_*.json")))
        if not hits:
            raise SystemExit(f"no combo JSON under llm_out_q1000_{q}/combos -- run the sweep first")
        with open(hits[0], encoding="utf-8") as fh:
            payload = json.load(fh)
        rows = []
        for rec in payload.get("per_query_results") or []:
            golds = _golds(rec)
            if not golds:
                continue
            a_l2, a_mmr = _answers(rec)
            s1 = score_answer(extract_final_answer(a_l2 or ""), golds)
            s2 = score_answer(extract_final_answer(a_mmr or ""), golds)
            rows.append(dict(query_idx=rec.get("query_idx"), source=q,
                             l2_em=s1["em"], mmr_em=s2["em"],
                             l2_f1=s1["f1"], mmr_f1=s2["f1"], n_gold=len(golds)))
        frames.append(pd.DataFrame(rows))
        print(f"  rebuilt {q}: {len(rows)} queries")
    return pd.concat(frames, ignore_index=True)


def load_retrieval(root):
    """Per-query gold-support outcome for both arms, from the retrieval JSONL."""
    path = os.path.join(root, "sweep_exports", RUN_ID, f"retrieval_{RUN_ID}.jsonl")
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            gs = [int(g) for g in (d.get("gold_support") or [])]
            ids = d.get("retrieved_doc_ids") or {}
            k = int(d.get("top_k") or 5)
            row = dict(query_idx=int(d["query_idx"]), query=d.get("query", ""), n_gold=len(gs))
            for arm in ("l2", "mmr"):
                got = [int(x) for x in (ids.get(arm) or [])][:k]
                row[f"{arm}_full"] = int(bool(gs) and all(g in got for g in gs))
                row[f"{arm}_rec"] = (sum(g in got for g in gs) / len(gs)) if gs else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def load_all(root, rebuild):
    perq_path = os.path.join(root, "figures", "perq_em_f1_q1000.csv")
    retr = load_retrieval(root)
    if rebuild or not os.path.isfile(perq_path):
        print("[1/3] rebuilding per-query EM/F1 from the combo JSONs ...")
        perq = rebuild_perquery(root)
        perq = perq.merge(retr[["query_idx", "query", "l2_full", "mmr_full", "l2_rec", "mmr_rec"]],
                          on="query_idx", how="left")
        os.makedirs(os.path.join(root, "figures"), exist_ok=True)
        perq.to_csv(perq_path, index=False)
        print(f"      wrote {os.path.relpath(perq_path, root)}")
    else:
        perq = pd.read_csv(perq_path)
    faith = pd.read_csv(os.path.join(root, "figures", "faith_rel_q1000.csv"))
    official = pd.read_csv(os.path.join(root, "figures", "em_f1_q1000.csv"))

    # Identity check: the per-query table must reproduce the official aggregate.
    for q in QUANTS:
        s, o = perq[perq.source == q], official[official.source == q].iloc[0]
        for arm in ("l2", "mmr"):
            got, want = s[f"{arm}_f1"].mean(), float(o[f"{arm}_f1"])
            assert abs(got - want) < 1e-9, f"{q}/{arm} F1 {got} != official {want}"
    print("[2/3] per-query EM/F1 reproduces figures/em_f1_q1000.csv to 1e-9  [OK]")
    return perq, faith, retr, official


def wide(faith, col, quant):
    return (faith[faith.source == quant]
            .pivot_table(index="query_idx", columns="arm", values=col)
            .dropna())


# ------------------------------------------------------------- the analysis --
def analyse(root, perq, faith, retr):
    out = {}

    # A. headline: MMR vs L2 at each precision -------------------------------
    rows = []
    for q in QUANTS:
        s = perq[perq.source == q]
        for nm, c in (("F1", "f1"), ("EM", "em")):
            rows.append(dict(quantization=q, metric=nm, **paired(s[f"l2_{c}"], s[f"mmr_{c}"])))
        for nm, c in (("faithfulness", "faithfulness_mean_entail"),
                      ("faithfulness@0.5", "faithfulness"),
                      ("answer_relevance", "answer_relevance"),
                      ("n_no_info (lower better)", "n_no_info")):
            w = wide(faith, c, q)
            rows.append(dict(quantization=q, metric=nm, **paired(w["l2"], w["mmr"])))
    out["headline"] = pd.DataFrame(rows)

    # B. retrieval ------------------------------------------------------------
    b = int(((retr.l2_full == 1) & (retr.mmr_full == 0)).sum())
    c_ = int(((retr.l2_full == 0) & (retr.mmr_full == 1)).sum())
    rows = [dict(metric="full support @5 (both gold paragraphs)",
                 **paired(retr.l2_full, retr.mmr_full),
                 mcnemar_l2_only=b, mcnemar_mmr_only=c_,
                 mcnemar_p=float(stats.binomtest(c_, b + c_, 0.5).pvalue)),
            dict(metric="gold recall @5", **paired(retr.l2_rec, retr.mmr_rec),
                 mcnemar_l2_only="", mcnemar_mmr_only="", mcnemar_p="")]
    sweep = pd.read_csv(os.path.join(root, "sweep_exports", RUN_ID, "sweep_results_full_grid.csv")).iloc[0]
    for nm, kl, km, direction in (("ILD (diversity)", "l2_ild", "mmr_ild", "higher"),
                                  ("RR (near-duplicate rate)", "l2_rr", "mmr_rr", "lower"),
                                  ("centroid breadth", "l2_breadth", "mmr_breadth", "higher"),
                                  ("ROUGE-L of retrieved text", "l2_rouge_l", "mmr_rouge_l", "higher")):
        rows.append(dict(metric=f"{nm} [{direction} is better]", n=1000,
                         l2=float(sweep[kl]), mmr=float(sweep[km]),
                         delta=float(sweep[km]) - float(sweep[kl]),
                         lo=np.nan, hi=np.nan, p_t=np.nan, p_wilcoxon=np.nan,
                         sig95="not testable (sweep CSV is aggregate-only)", bonferroni="",
                         mcnemar_l2_only="", mcnemar_mmr_only="", mcnemar_p=""))
    out["retrieval"] = pd.DataFrame(rows)

    # C. key contrasts --------------------------------------------------------
    w = perq.pivot_table(index="query_idx", columns="source", values=["l2_f1", "mmr_f1", "l2_em", "mmr_em"])
    rows = []
    for nm, c in (("F1", "f1"), ("EM", "em")):
        r = paired(w[(f"l2_{c}", "fp16")], w[(f"mmr_{c}", "int4")])
        tost = max(stats.ttest_rel(w[(f"mmr_{c}", "int4")], w[(f"l2_{c}", "fp16")] + 0.02).pvalue / 2,
                   stats.ttest_rel(w[(f"mmr_{c}", "int4")], w[(f"l2_{c}", "fp16")] - 0.02).pvalue / 2)
        rows.append(dict(contrast=f"MMR@int4  vs  L2@fp16   ({nm})", **r, tost_pm0_02=float(tost)))
    wf = faith.pivot_table(index="query_idx", columns=["source", "arm"],
                           values="faithfulness_mean_entail").dropna()
    rows.append(dict(contrast="MMR@int4  vs  L2@fp16   (faithfulness)",
                     **paired(wf[("fp16", "l2")], wf[("int4", "mmr")]), tost_pm0_02=np.nan))
    for arm in ("l2", "mmr"):
        for nm, c in (("F1", "f1"),):
            rows.append(dict(contrast=f"{arm.upper()} degradation fp16->int4 ({nm})",
                             **paired(w[(f"{arm}_{c}", "fp16")], w[(f"{arm}_{c}", "int4")]),
                             tost_pm0_02=np.nan))
        rows.append(dict(contrast=f"{arm.upper()} degradation fp16->int4 (faithfulness)",
                         **paired(wf[("fp16", arm)], wf[("int4", arm)]), tost_pm0_02=np.nan))
    did_f1 = ((w[("mmr_f1", "int4")] - w[("mmr_f1", "fp16")])
              - (w[("l2_f1", "int4")] - w[("l2_f1", "fp16")])).values
    did_fa = ((wf[("int4", "mmr")] - wf[("fp16", "mmr")])
              - (wf[("int4", "l2")] - wf[("fp16", "l2")])).values
    for nm, d in (("F1", did_f1), ("faithfulness", did_fa)):
        mu, lo, hi = boot(d)
        rows.append(dict(contrast=f"DiD: does MMR degrade LESS fp16->int4?  ({nm})",
                         n=len(d), l2=np.nan, mmr=np.nan, delta=mu, lo=lo, hi=hi,
                         p_t=float(stats.ttest_1samp(d, 0).pvalue), p_wilcoxon=np.nan,
                         sig95="yes" if (lo > 0 or hi < 0) else "no",
                         bonferroni="", tost_pm0_02=np.nan))
    out["contrasts"] = pd.DataFrame(rows)

    # D. mechanism: split the F1 delta by retrieval outcome -------------------
    m = perq.drop(columns=["l2_full", "mmr_full"], errors="ignore").merge(
        retr[["query_idx", "l2_full", "mmr_full"]], on="query_idx")
    m["group"] = np.select(
        [(m.l2_full == 1) & (m.mmr_full == 1), (m.l2_full == 0) & (m.mmr_full == 0),
         (m.l2_full == 0) & (m.mmr_full == 1), (m.l2_full == 1) & (m.mmr_full == 0)],
        ["both arms full", "neither arm full", "MMR-only full", "L2-only full"], default="?")
    rows = []
    for q in QUANTS:
        s = m[m.source == q]
        for g, sub in s.groupby("group"):
            rows.append(dict(quantization=q, group=g, **paired(sub.l2_f1, sub.mmr_f1)))
        sub = s[s.group.isin(["both arms full", "neither arm full"])]
        rows.append(dict(quantization=q, group="RETRIEVAL-MATCHED (both+neither)",
                         **paired(sub.l2_f1, sub.mmr_f1)))
    out["mechanism"] = pd.DataFrame(rows)

    # E. subgroups + faithfulness cutoff sweep --------------------------------
    qs = json.load(open(os.path.join(root, "processed_hotpot", "hotpot_queries.json"), encoding="utf-8"))
    meta = pd.DataFrame({"query_idx": range(len(qs)),
                         "qtype": [x.get("qtype") for x in qs],
                         "level": [x.get("level") for x in qs]})
    rows = []
    for q in QUANTS:
        s = perq[perq.source == q].merge(meta, on="query_idx")
        for key in ("qtype", "level"):
            for g, sub in s.groupby(key):
                rows.append(dict(quantization=q, split=key, group=str(g),
                                 **paired(sub.l2_f1, sub.mmr_f1)))
    out["subgroups"] = pd.DataFrame(rows)

    rows = []
    for t in (0.5, 0.7, 0.9, 0.95, 0.99):
        for q in QUANTS:
            s = faith[faith.source == q].copy()
            s["bin"] = (s.faithfulness_mean_entail >= t).astype(float)
            wv = s.pivot_table(index="query_idx", columns="arm", values="bin").dropna()
            rows.append(dict(cutoff=t, quantization=q, **paired(wv["l2"], wv["mmr"]),
                             pct_at_ceiling=100 * float((s["bin"] == 1).mean())))
    out["faith_cutoff"] = pd.DataFrame(rows)

    # F. power ----------------------------------------------------------------
    rows = []
    for q in QUANTS:
        s = perq[perq.source == q]
        for nm, d in ((f"dF1 {q}", (s.mmr_f1 - s.l2_f1).values),
                      (f"dEM {q}", (s.mmr_em - s.l2_em).values)):
            sd = float(np.nanstd(d, ddof=1))
            rows.append(dict(quantity=nm, effect=float(np.nanmean(d)), sd_per_query=sd,
                             ci_halfwidth_at_n1000=1.959964 * sd / np.sqrt(1000),
                             n_for_80pct_power=need_n(float(np.nanmean(d)), sd)))
        wv = wide(faith, "faithfulness_mean_entail", q)
        d = (wv["mmr"] - wv["l2"]).values
        sd = float(np.nanstd(d, ddof=1))
        rows.append(dict(quantity=f"dFaithfulness {q}", effect=float(np.nanmean(d)), sd_per_query=sd,
                         ci_halfwidth_at_n1000=1.959964 * sd / np.sqrt(1000),
                         n_for_80pct_power=need_n(float(np.nanmean(d)), sd)))
    for nm, d in (("DiD F1 (fp16->int4)", did_f1), ("DiD faithfulness (fp16->int4)", did_fa)):
        sd = float(np.nanstd(d, ddof=1))
        rows.append(dict(quantity=nm, effect=float(np.nanmean(d)), sd_per_query=sd,
                         ci_halfwidth_at_n1000=1.959964 * sd / np.sqrt(1000),
                         n_for_80pct_power=need_n(float(np.nanmean(d)), sd)))
    out["power"] = pd.DataFrame(rows)
    out["perquery"] = m.drop(columns=["group"]).sort_values(["source", "query_idx"])
    return out


# ------------------------------------------------------------------ report --
def report(out):
    pd.set_option("display.width", 200, "display.max_columns", 40,
                  "display.float_format", lambda v: f"{v:.4f}")
    for name in ("headline", "retrieval", "contrasts", "mechanism", "subgroups",
                 "faith_cutoff", "power"):
        print(f"\n{'=' * 78}\n{name.upper()}\n{'=' * 78}")
        print(out[name].to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true",
                    help="re-derive per-query EM/F1 from llm_out_q1000_*/combos (slow, authoritative)")
    ap.add_argument("--no_xlsx", action="store_true")
    ap.add_argument("--out", default="figures/q1000_metrics.xlsx")
    args = ap.parse_args()

    perq, faith, retr, _ = load_all(ROOT, args.rebuild)
    out = analyse(ROOT, perq, faith, retr)
    report(out)
    if not args.no_xlsx:
        from scripts.q1000_workbook import write_workbook
        path = os.path.join(ROOT, args.out)
        write_workbook(out, path, run_id=RUN_ID, seed=SEED, boots=B, n_tests=N_TESTS)
        print(f"\n[3/3] workbook -> {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
