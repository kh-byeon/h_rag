#!/usr/bin/env python3
"""Tables for the simulation ablation (scripts/run_ablation_q1000.sh).

    python scripts/analyze_ablation.py ablation/manifest.tsv --out ablation/ablation_q1000 \
        [--runs_dir ablation/runs] [--reference sweep_exports/20260826_223418_mmr-capacitor_centroid-nvm]

Inputs: the manifest (arm, scenario, seed, run_id) and, per run,
<runs_dir>/<run_id>/retrieval_<run_id>.jsonl and sweep_results_full_grid.csv.

Per run (MMR arm): full support@5, its paired difference from the L2 arm with a 95% bootstrap
interval over questions, gold-paragraph recall@5, and the corpus-aggregate diversity measures the
sweep records (intra-list diversity, near-duplicate rate, centroid breadth). Against reference
runs: the fraction of questions whose five retrieved paragraphs are the same set as in digital
identical MMR and in the average-device run, and a two-sided exact McNemar test on full support@5
against the average device (only discordant questions carry information). The L2 arm is taken
from the digital_mmr_std run; it does not depend on the MMR backend (checked across all runs).
With --reference, the average-device run is compared question by question with the published run.

Writes <out>_runs.csv (one row per run), <out>_summary.csv (mean, s.d., min and max over dies per
scenario) and <out>.md (the table for the Supplementary Information).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics as st
import sys
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.metrics.supporting_recall import evaluate_supporting_recall  # noqa: E402

K = 5
B = 20000


def load_run(d: str) -> Tuple[Dict[str, dict], dict]:
    rid = os.path.basename(os.path.normpath(d))
    recs = {}
    with open(os.path.join(d, f"retrieval_{rid}.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                recs[r["query"]] = r
    grid_p = os.path.join(d, "sweep_results_full_grid.csv")
    grid: dict = {}
    if os.path.isfile(grid_p):
        with open(grid_p, encoding="utf-8") as f:
            grid = next(csv.DictReader(f))
    return recs, grid


def find_run(rid: str, runs_dir: str, root: str) -> str:
    for d in (os.path.join(runs_dir, rid), os.path.join(root, "sweep_exports", rid)):
        if os.path.isdir(d):
            return d
    raise FileNotFoundError(rid)


def ids(rec: dict, arm: str) -> List[int]:
    return [int(x) for x in (rec.get("retrieved_doc_ids") or {}).get(arm) or []][:K]


def retrieval(recs: Dict[str, dict], arm: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    full, rec5 = {}, {}
    for q, r in recs.items():
        gs = r.get("gold_support")
        if not gs:
            continue
        m = evaluate_supporting_recall(ids(r, arm), gs, ks=(K,))
        full[q], rec5[q] = m[f"full_support@{K}"], m[f"recall@{K}"]
    return full, rec5


def paired_ci(a: Dict[str, float], b: Dict[str, float], seed: int = 0) -> Tuple[float, float, float]:
    """Mean paired difference a - b over questions and its 95% percentile bootstrap interval."""
    qs = sorted(a.keys() & b.keys())
    d = np.asarray([a[q] - b[q] for q in qs], dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(B)
    for i in range(0, B, 1000):
        means[i:i + 1000] = d[rng.integers(0, d.size, (1000, d.size))].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def mcnemar_exact(a: Dict[str, float], b: Dict[str, float]) -> Tuple[int, int, float]:
    qs = a.keys() & b.keys()
    n10 = sum(1 for q in qs if a[q] == 1 and b[q] == 0)
    n01 = sum(1 for q in qs if a[q] == 0 and b[q] == 1)
    n = n10 + n01
    if n == 0:
        return n10, n01, 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(n10, n01) + 1)) / 2 ** n
    return n10, n01, min(1.0, 2 * tail)


def agreement(x: Dict[str, dict], y: Dict[str, dict], arm_x: str = "mmr", arm_y: str = "mmr") -> float:
    qs = x.keys() & y.keys()
    return sum(set(ids(x[q], arm_x)) == set(ids(y[q], arm_y)) for q in qs) / len(qs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest")
    ap.add_argument("--root", default=".")
    ap.add_argument("--runs_dir", default="ablation/runs")
    ap.add_argument("--reference", default=None, help="published run directory for the identity check")
    ap.add_argument("--out", default="ablation/ablation_q1000")
    ap.add_argument("--export_perq", default=None,
                    help="optional .csv.gz with one row per run and question (retrieved ids and full support@5), "
                         "a compact record of all runs that replaces the per-run JSONL files")
    a = ap.parse_args()

    with open(a.manifest, encoding="utf-8") as f:
        man = list(csv.DictReader(f, delimiter="\t"))
    runs = {(m["arm"], m["scenario"], m["seed"]): load_run(find_run(m["run_id"], a.runs_dir, a.root)) for m in man}
    dig_key = ("digital_mmr_std", "none", "0") if ("digital_mmr_std", "none", "0") in runs else ("digital_mmr", "none", "0")
    ref_dig, ref_dev = runs[dig_key], runs[("device_avg", "none", "0")]
    n_q = len(ref_dev[0])
    dev_full, _ = retrieval(ref_dev[0], "mmr")
    l2_full, l2_rec = retrieval(ref_dig[0], "l2")

    # the L2 arm must not depend on the MMR backend
    l2_same = min(agreement(r[0], ref_dig[0], "l2", "l2") for r in runs.values())
    notes = [f"questions per run: {n_q}", f"L2 arm identical across all runs: {l2_same:.4f}"]
    if a.reference:
        ref = load_run(a.reference)[0]
        qs = ref_dev[0].keys() & ref.keys()
        same = sum(ids(ref_dev[0][q], "mmr") == ids(ref[q], "mmr") and ids(ref_dev[0][q], "l2") == ids(ref[q], "l2")
                   for q in qs)
        notes.append(f"average-device run vs published run {os.path.basename(os.path.normpath(a.reference))}: "
                     f"{same}/{len(qs)} questions with identical L2 and MMR lists")

    # paired comparisons between the reference arms (full support@5)
    comparisons: Dict[str, dict] = {"n_questions": n_q, "l2_identical": l2_same}
    for other in (("digital_mmr_std", "none", "0"), ("digital_mmr_plain", "none", "0")):
        if other in runs:
            of, _ = retrieval(runs[other][0], "mmr")
            d, lo, hi = paired_ci(dev_full, of)
            n10, n01, p = mcnemar_exact(dev_full, of)
            notes.append(f"device_avg − {other[0]}: full support@5 {d:+.3f} [{lo:+.3f}, {hi:+.3f}], "
                         f"discordant {n10} vs {n01}, exact McNemar P = {p:.3g}")
            comparisons[f"device_avg_minus_{other[0]}"] = dict(diff=d, lo=lo, hi=hi, n10=n10, n01=n01, p=p)

    g = ref_dig[1]
    dig_rid = next(m["run_id"] for m in man if (m["arm"], m["scenario"], m["seed"]) == dig_key)
    rows = [dict(arm="digital_l2", scenario="none", seed="0", run_id=dig_rid,
                 full5=st.mean(l2_full.values()), d_full5_vs_l2=0.0, d_lo=float("nan"), d_hi=float("nan"),
                 recall5=st.mean(l2_rec.values()),
                 ild=float(g["l2_ild"]), near_dup=float(g["l2_rr"]), breadth=float(g["l2_breadth"]),
                 same_set_vs_digital=agreement(ref_dig[0], ref_dig[0], "l2", "mmr"),
                 same_set_vs_device=agreement(ref_dig[0], ref_dev[0], "l2", "mmr"),
                 mcnemar_vs_device_p=mcnemar_exact(l2_full, dev_full)[2])]
    for m in man:
        key = (m["arm"], m["scenario"], m["seed"])
        recs, g = runs[key]
        full, rec5 = retrieval(recs, "mmr")
        d, lo, hi = paired_ci(full, l2_full)
        rows.append(dict(arm=m["arm"], scenario=m["scenario"], seed=m["seed"], run_id=m["run_id"],
                         full5=st.mean(full.values()), d_full5_vs_l2=d, d_lo=lo, d_hi=hi,
                         recall5=st.mean(rec5.values()),
                         ild=float(g["mmr_ild"]), near_dup=float(g["mmr_rr"]), breadth=float(g["mmr_breadth"]),
                         same_set_vs_digital=agreement(recs, ref_dig[0]),
                         same_set_vs_device=agreement(recs, ref_dev[0]),
                         mcnemar_vs_device_p=mcnemar_exact(full, dev_full)[2]))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.export_perq:
        import gzip
        with gzip.open(a.export_perq, "wt", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["arm", "scenario", "seed", "run_id", "query_idx", "query", "gold_support", "l2_ids", "mmr_ids",
                        "full5_l2", "full5_mmr"])
            for m in man:
                recs = runs[(m["arm"], m["scenario"], m["seed"])][0]
                fl2, _ = retrieval(recs, "l2")
                fm, _ = retrieval(recs, "mmr")
                for q, r in sorted(recs.items(), key=lambda kv: int(kv[1].get("query_idx", 0))):
                    w.writerow([m["arm"], m["scenario"], m["seed"], m["run_id"], r.get("query_idx"), q,
                                " ".join(map(str, r.get("gold_support") or [])),
                                " ".join(map(str, ids(r, "l2"))), " ".join(map(str, ids(r, "mmr"))),
                                fl2.get(q, ""), fm.get(q, "")])
    with open(a.out + "_runs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    groups: Dict[Tuple[str, str], List[dict]] = {}
    for r in rows:
        groups.setdefault((r["arm"], r["scenario"]), []).append(r)
    metrics = ["full5", "d_full5_vs_l2", "recall5", "ild", "near_dup", "breadth",
               "same_set_vs_digital", "same_set_vs_device", "mcnemar_vs_device_p"]
    summ = []
    for (arm, scen), rs in groups.items():
        row = {"arm": arm, "scenario": scen, "n_dies": len(rs)}
        for mt in metrics:
            v = [r[mt] for r in rs if r[mt] == r[mt]]
            row[f"{mt}_mean"] = st.mean(v) if v else float("nan")
            row[f"{mt}_sd"] = st.stdev(v) if len(v) > 1 else float("nan")
            row[f"{mt}_min"] = min(v) if v else float("nan")
            row[f"{mt}_max"] = max(v) if v else float("nan")
        if len(rs) == 1:
            row["d_lo"], row["d_hi"] = rs[0]["d_lo"], rs[0]["d_hi"]
        summ.append(row)
    with open(a.out + "_summary.csv", "w", newline="", encoding="utf-8") as f:
        keys: List[str] = []
        for r in summ:
            keys += [k for k in r if k not in keys]
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(summ)

    def fmt(r, mt, d=3, sign=False):
        m, s = r[f"{mt}_mean"], r[f"{mt}_sd"]
        if m != m:
            return "—"
        t = f"{m:+.{d}f}" if sign else f"{m:.{d}f}"
        if s == s:
            return t + f" ± {s:.{d}f}"
        if mt == "d_full5_vs_l2" and r.get("d_lo") == r.get("d_lo") and "d_lo" in r:
            return t + f" [{r['d_lo']:+.{d}f}, {r['d_hi']:+.{d}f}]"
        return t
    lines = ["| Arm | Variation | Dies | Full support@5 | MMR − L2 | Gold recall@5 | Near-dup rate | "
             "Same set as digital MMR | Same set as average device |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in summ:
        dd = "—" if r["arm"] == "digital_l2" else fmt(r, "d_full5_vs_l2", sign=True)
        lines.append(f"| {r['arm']} | {r['scenario']} | {r['n_dies']} | {fmt(r, 'full5')} | {dd} | {fmt(r, 'recall5')} | "
                     f"{fmt(r, 'near_dup')} | {fmt(r, 'same_set_vs_digital')} | {fmt(r, 'same_set_vs_device')} |")
    text = "\n".join(lines) + "\n\n" + "\n".join(notes) + "\n"
    if a.reference:
        comparisons["identical_to_reference"] = same
    with open(a.out + "_comparisons.json", "w", encoding="utf-8") as f:
        json.dump(comparisons, f, indent=1)
    with open(a.out + ".md", "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"wrote {a.out}_runs.csv, {a.out}_summary.csv, {a.out}.md")


if __name__ == "__main__":
    main()
