#!/usr/bin/env python3
"""
Paper figure: on-the-fly MMR (FeFET/IGZO device path) vs L2-only retrieval.

Reads a sweep_eval CSV (sweep_results_full_grid.csv) and emits a 3-panel
grouped-bar figure with per-metric delta annotations:

  (a) Diversity   -- ILD (up), RR (down)          MMR wins
  (b) Exploration -- centroid breadth (up)        MMR wins
  (c) Accuracy    -- recall / ROUGE-L / recall@5 / full_support@5   MMR cost

Bars are means over the lambda x tau x gamma grid at a fixed top_k; error bars
are the std over that grid. The L2 arm is invariant to lambda/tau/gamma by
construction, so its std is exactly 0 -- that is a correctness check, not a
missing error bar.

Usage
-----
    python scripts/plot_mmr_vs_l2_figure.py \
        --csv sweep_exports/<run_id>/sweep_results_full_grid.csv \
        --top_k 20 --out_dir figures
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics as st
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- Style: neutral, print-safe, no chartjunk -------------------------------
C_L2 = "#8C9196"   # muted grey  - baseline
C_MMR = "#2F6F9F"  # steel blue  - proposed
C_GOOD = "#1B7F4B"
C_BAD = "#B3391C"

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon": False,
    "savefig.bbox": "tight",
})

# label, l2_col, mmr_col, higher_is_better
MetricSpec = Tuple[str, str, str, bool]

PANEL_DIVERSITY: List[MetricSpec] = [
    ("ILD $\\uparrow$", "l2_ild", "mmr_ild", True),
    ("RR $\\downarrow$", "l2_rr", "mmr_rr", False),
]
PANEL_BREADTH: List[MetricSpec] = [
    ("Centroid\nbreadth $\\uparrow$", "l2_breadth", "mmr_breadth", True),
]
PANEL_ACCURACY: List[MetricSpec] = [
    ("Recall", "l2_recall", "mmr_recall", True),
    ("ROUGE-L", "l2_rouge_l", "mmr_rouge_l", True),
    ("Recall@5", "l2_recall@5", "mmr_recall@5", True),
    ("Full-supp@5", "l2_full@5", "mmr_full@5", True),
]


def load_rows(csv_path: str, top_k: int) -> List[Dict[str, str]]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if int(float(r["top_k"])) == top_k]
    if not rows:
        raise SystemExit(f"no rows with top_k={top_k} in {csv_path}")
    return rows


def summarize(rows: Sequence[Dict[str, str]], specs: Sequence[MetricSpec]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for label, a, b, higher in specs:
        A = [float(r[a]) for r in rows]
        B = [float(r[b]) for r in rows]
        ma, mb = st.mean(A), st.mean(B)
        delta = mb - ma
        rel = (delta / ma * 100.0) if ma else float("nan")
        # "improvement" is signed by the metric's own direction
        improved = delta > 0 if higher else delta < 0
        out.append({
            "label": label, "l2": ma, "mmr": mb,
            "l2_sd": st.pstdev(A), "mmr_sd": st.pstdev(B),
            "delta": delta, "rel": rel,
            "higher_is_better": higher,
            "improved": improved if abs(delta) > 1e-12 else None,
        })
    return out


def draw_panel(ax, stats: List[Dict[str, Any]], title: str, ylabel: str,
               annotate_rel: bool = True) -> None:
    n = len(stats)
    x = np.arange(n, dtype=float)
    w = 0.34

    l2 = [s["l2"] for s in stats]
    mmr = [s["mmr"] for s in stats]
    l2_sd = [s["l2_sd"] for s in stats]
    mmr_sd = [s["mmr_sd"] for s in stats]

    ax.bar(x - w / 2, l2, w, yerr=l2_sd, capsize=2.5, label="L2-only",
           color=C_L2, edgecolor="black", linewidth=0.6,
           error_kw={"elinewidth": 0.8, "capthick": 0.8})
    ax.bar(x + w / 2, mmr, w, yerr=mmr_sd, capsize=2.5, label="On-the-fly MMR",
           color=C_MMR, edgecolor="black", linewidth=0.6, hatch="///",
           error_kw={"elinewidth": 0.8, "capthick": 0.8})

    top = max(max(v + e for v, e in zip(l2, l2_sd)),
              max(v + e for v, e in zip(mmr, mmr_sd)))
    ax.set_ylim(0, top * 1.34)

    for xi, s in zip(x, stats):
        y = max(s["l2"] + s["l2_sd"], s["mmr"] + s["mmr_sd"])
        if s["improved"] is None:
            txt, col = "no change", "#555555"
        else:
            col = C_GOOD if s["improved"] else C_BAD
            txt = (f"{s['delta']:+.3f}\n({s['rel']:+.1f}%)"
                   if annotate_rel else f"{s['delta']:+.2f}")
        ax.text(xi, y + top * 0.045, txt, ha="center", va="bottom",
                fontsize=7.4, color=col, linespacing=1.15)

    ax.set_xticks(x)
    ax.set_xticklabels([s["label"] for s in stats], fontsize=8.2)
    ax.set_ylabel(ylabel, fontsize=8.6)
    ax.set_title(title, fontsize=9.2, pad=6)
    ax.tick_params(axis="y", labelsize=7.8)
    ax.grid(axis="y", linewidth=0.4, alpha=0.35)
    ax.set_axisbelow(True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--out_dir", default="figures")
    ap.add_argument("--tag", default=None, help="filename stem (default: mmr_vs_l2_topk<k>)")
    args = ap.parse_args()

    rows = load_rows(args.csv, args.top_k)
    n_combo = len(rows)
    n_q = int(float(rows[0].get("support_n") or 0))

    div = summarize(rows, PANEL_DIVERSITY)
    bre = summarize(rows, PANEL_BREADTH)
    acc = summarize(rows, PANEL_ACCURACY)

    fig = plt.figure(figsize=(11.0, 3.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 0.85, 2.9], wspace=0.34)
    ax0, ax1, ax2 = (fig.add_subplot(gs[0, i]) for i in range(3))

    draw_panel(ax0, div, "(a) Retrieval diversity", "score")
    draw_panel(ax1, bre, "(b) Exploration", "unique centroids", annotate_rel=False)
    draw_panel(ax2, acc, "(c) Answer / supporting-passage accuracy", "score")

    ax0.legend(loc="upper left", fontsize=7.8, ncol=1, handlelength=1.5)

    fig.suptitle(
        f"On-the-fly MMR vs L2-only retrieval  "
        f"(HotpotQA, cross-document scope, top-$k$={args.top_k}, "
        f"{n_q} queries, mean $\\pm$ s.d. over {n_combo} $\\lambda\\times\\tau\\times\\gamma$ configs)",
        fontsize=9.4, y=1.06,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    stem = args.tag or f"mmr_vs_l2_topk{args.top_k}"
    png = os.path.join(args.out_dir, f"{stem}.png")
    fig.savefig(png, dpi=300)
    plt.close(fig)

    # Companion table (the numbers behind the bars).
    tsv = os.path.join(args.out_dir, f"{stem}_table.csv")
    with open(tsv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["panel", "metric", "higher_is_better", "L2_mean", "L2_sd",
                    "MMR_mean", "MMR_sd", "delta", "rel_pct", "mmr_better"])
        for panel, stats in (("diversity", div), ("exploration", bre), ("accuracy", acc)):
            for s in stats:
                w.writerow([
                    panel, s["label"].split(" $")[0].replace("\n", " "),
                    s["higher_is_better"],
                    f"{s['l2']:.4f}", f"{s['l2_sd']:.4f}",
                    f"{s['mmr']:.4f}", f"{s['mmr_sd']:.4f}",
                    f"{s['delta']:+.4f}", f"{s['rel']:+.2f}", s["improved"],
                ])

    print(f"[figure] {png}\n[table ] {tsv}")
    for panel, stats in (("diversity", div), ("exploration", bre), ("accuracy", acc)):
        for s in stats:
            print(f"  {panel:12s} {s['label'].split(' $')[0]:14s} "
                  f"L2={s['l2']:.4f}  MMR={s['mmr']:.4f}  Δ={s['delta']:+.4f} ({s['rel']:+.1f}%)")


if __name__ == "__main__":
    main()