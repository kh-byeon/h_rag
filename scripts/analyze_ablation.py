#!/usr/bin/env python3
"""Tables for the four-arm ablation (scripts/run_ablation_q1000.sh).

    python scripts/analyze_ablation.py ablation/manifest.tsv --out ablation/ablation_q1000

Inputs: the manifest (arm, scenario, seed, run_id) and, per run,
sweep_exports/<run_id>/retrieval_<run_id>.jsonl and sweep_results_full_grid.csv.

Per run (MMR arm): full support@5, gold-paragraph recall@5, and the corpus-aggregate diversity
measures the sweep records (intra-list diversity, near-duplicate rate, centroid breadth).
Against reference runs: the fraction of questions whose five retrieved paragraphs are the same
set as in digital MMR and in the average-device run, the mean Jaccard overlap, and a two-sided
exact McNemar test on full support@5 (only discordant questions carry information).
The L2 arm is taken from the digital-MMR run; it does not depend on the MMR backend.

Writes <out>_runs.csv (one row per run), <out>_summary.csv (mean, s.d., min, max over seeds per
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.metrics.supporting_recall import evaluate_supporting_recall  # noqa: E402

K = 5


def load_run(rid: str, root: str) -> Tuple[Dict[str, dict], dict]:
    d = os.path.join(root, "sweep_exports", rid)
    recs = {}
    with open(os.path.join(d, f"retrieval_{rid}.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                recs[r["query"]] = r
    with open(os.path.join(d, "sweep_results_full_grid.csv"), encoding="utf-8") as f:
        grid = next(csv.DictReader(f))
    return recs, grid


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


def mcnemar_exact(a: Dict[str, float], b: Dict[str, float]) -> Tuple[int, int, float]:
    qs = a.keys() & b.keys()
    n10 = sum(1 for q in qs if a[q] == 1 and b[q] == 0)
    n01 = sum(1 for q in qs if a[q] == 0 and b[q] == 1)
    n = n10 + n01
    if n == 0:
        return n10, n01, 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(n10, n01) + 1)) / 2 ** n
    return n10, n01, min(1.0, 2 * tail)


def agreement(x: Dict[str, dict], y: Dict[str, dict]) -> Tuple[float, float]:
    qs = x.keys() & y.keys()
    same, jac = 0, []
    for q in qs:
        a, b = set(ids(x[q], "mmr")), set(ids(y[q], "mmr"))
        same += a == b
        jac.append(len(a & b) / len(a | b) if a | b else 1.0)
    return same / len(qs), sum(jac) / len(jac)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest")
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="ablation/ablation_q1000")
    a = ap.parse_args()

    with open(a.manifest, encoding="utf-8") as f:
        man = list(csv.DictReader(f, delimiter="\t"))
    runs = {(m["arm"], m["scenario"], m["seed"]): load_run(m["run_id"], a.root) for m in man}
    ref_dig = runs[("digital_mmr", "none", "0")]
    ref_dev = runs[("device_avg", "none", "0")]
    dig_full, _ = retrieval(ref_dig[0], "mmr")
    dev_full, _ = retrieval(ref_dev[0], "mmr")

    rows = []
    # arm 1 from the digital run's L2 arm
    l2_full, l2_rec = retrieval(ref_dig[0], "l2")
    g = ref_dig[1]
    dig_rid = next(m["run_id"] for m in man if m["arm"] == "digital_mmr")
    rows.append(dict(arm="digital_l2", scenario="none", seed="0", run_id=dig_rid,
                     full5=st.mean(l2_full.values()), recall5=st.mean(l2_rec.values()),
                     ild=float(g["l2_ild"]), near_dup=float(g["l2_rr"]), breadth=float(g["l2_breadth"]),
                     same_set_vs_digital=float("nan"), jaccard_vs_digital=float("nan"),
                     same_set_vs_device=float("nan"), jaccard_vs_device=float("nan"),
                     mcnemar_vs_device_p=mcnemar_exact(l2_full, dev_full)[2]))
    for m in man:
        key = (m["arm"], m["scenario"], m["seed"])
        recs, g = runs[key]
        full, rec5 = retrieval(recs, "mmr")
        sd, jd = agreement(recs, ref_dig[0])
        sv, jv = agreement(recs, ref_dev[0])
        rows.append(dict(arm=m["arm"], scenario=m["scenario"], seed=m["seed"], run_id=m["run_id"],
                         full5=st.mean(full.values()), recall5=st.mean(rec5.values()),
                         ild=float(g["mmr_ild"]), near_dup=float(g["mmr_rr"]), breadth=float(g["mmr_breadth"]),
                         same_set_vs_digital=sd, jaccard_vs_digital=jd,
                         same_set_vs_device=sv, jaccard_vs_device=jv,
                         mcnemar_vs_device_p=mcnemar_exact(full, dev_full)[2]))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    cols = list(rows[0].keys())
    with open(a.out + "_runs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

    groups: Dict[Tuple[str, str], List[dict]] = {}
    for r in rows:
        groups.setdefault((r["arm"], r["scenario"]), []).append(r)
    metrics = ["full5", "recall5", "ild", "near_dup", "breadth", "same_set_vs_digital",
               "jaccard_vs_digital", "same_set_vs_device", "mcnemar_vs_device_p"]
    summ = []
    for (arm, scen), rs in groups.items():
        row = {"arm": arm, "scenario": scen, "n_seeds": len(rs)}
        for mt in metrics:
            v = [r[mt] for r in rs if r[mt] == r[mt]]
            row[f"{mt}_mean"] = st.mean(v) if v else float("nan")
            row[f"{mt}_sd"] = st.stdev(v) if len(v) > 1 else float("nan")
            row[f"{mt}_min"] = min(v) if v else float("nan")
            row[f"{mt}_max"] = max(v) if v else float("nan")
        summ.append(row)
    with open(a.out + "_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys())); w.writeheader(); w.writerows(summ)

    def fmt(r, mt, d=3):
        m, s = r[f"{mt}_mean"], r[f"{mt}_sd"]
        if m != m:
            return "—"
        return f"{m:.{d}f}" + (f" ± {s:.{d}f}" if s == s else "")
    lines = ["| Arm | Variation | Seeds | Full support@5 | Gold recall@5 | Near-dup rate | Same set as digital MMR | Same set as average device |",
             "|---|---|---|---|---|---|---|---|"]
    for r in summ:
        lines.append(f"| {r['arm']} | {r['scenario']} | {r['n_seeds']} | {fmt(r, 'full5')} | {fmt(r, 'recall5')} | "
                     f"{fmt(r, 'near_dup')} | {fmt(r, 'same_set_vs_digital')} | {fmt(r, 'same_set_vs_device')} |")
    with open(a.out + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {a.out}_runs.csv, {a.out}_summary.csv, {a.out}.md")


if __name__ == "__main__":
    main()
