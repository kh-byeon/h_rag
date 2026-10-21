#!/usr/bin/env python3
"""Summarise IGZO-tier (MMR buffer) write / readout statistics from a retrieval run.

Reads the per-query ``write_stats`` block that ``step5_on_the_fly_mmr`` now records
(``sweep_exports/<run_id>/retrieval_<run_id>.jsonl``) and produces the numbers the
introduction needs:

  * pre-write filter: fraction of streamed candidates rejected before any array write
    (tau_dup gate + sticky-buffer score gate), and the write reduction it buys;
  * writes per re-ranking episode (anchors + MMR inserts + replacements);
  * readouts per episode and, from the IGZO circuit timing, the time the selected set
    must be held in the array (t_episode = readouts x T_END).

Endurance and retention requirements are printed as PLACEHOLDERS (they need device
data that is not in the repo yet).

Usage
  python scripts/analyze_write_stats.py sweep_exports/<run_id> [more run dirs...] \
      [--check_against sweep_exports/20260826_223418_mmr-capacitor_centroid-nvm]

--check_against pairs queries by text with a reference run and asserts the retrieved
doc-id lists of BOTH arms are identical, i.e. the instrumentation did not change
retrieval. Output: <run_dir>/write_stats_summary.json + a console table.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

FIELDS = [
    "n_pool", "n_anchor", "mmr_slots", "n_stream_avail", "buffer_rows_final",
    "n_streamed", "n_stream_main", "n_stream_fallback",
    "n_reject_tau", "n_reject_score", "n_write_append", "n_write_replace",
    "writes_mmr", "writes_total", "n_readout_main", "n_readout_calib", "n_readout_total",
]


def _find_jsonl(run_dir: str) -> str:
    if run_dir.endswith(".jsonl"):
        return run_dir
    hits = sorted(glob.glob(os.path.join(run_dir, "retrieval_*.jsonl")))
    if not hits:
        sys.exit(f"no retrieval_*.jsonl under {run_dir}")
    return hits[0]


def _load(jsonl: str) -> List[Dict[str, Any]]:
    rows = []
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _t_end_s(fit_json: str) -> float:
    with open(fit_json, encoding="utf-8") as f:
        return float(json.load(f)["circuit_params"]["T_END"])


def _stat(v: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(v)), "median": float(np.median(v)),
        "p95": float(np.percentile(v, 95)), "min": float(np.min(v)),
        "max": float(np.max(v)), "sum": float(np.sum(v)),
    }


def _check_against(rows: List[Dict[str, Any]], ref_dir: str) -> Dict[str, Any]:
    ref = {r["query"]: r for r in _load(_find_jsonl(ref_dir))}
    paired = same_l2 = same_mmr = 0
    for r in rows:
        q = ref.get(r["query"])
        if q is None:
            continue
        paired += 1
        a, b = r.get("retrieved_doc_ids") or {}, q.get("retrieved_doc_ids") or {}
        same_l2 += int(list(a.get("l2") or []) == list(b.get("l2") or []))
        same_mmr += int(list(a.get("mmr") or []) == list(b.get("mmr") or []))
    return {"reference": ref_dir, "paired": paired, "identical_l2": same_l2,
            "identical_mmr": same_mmr, "identical": paired > 0 and same_l2 == paired and same_mmr == paired}


def analyze(run_dir: str, t_end_s: float, check_against: Optional[str]) -> Dict[str, Any]:
    jsonl = _find_jsonl(run_dir)
    rows = _load(jsonl)
    ws = [r.get("write_stats") for r in rows]
    ws = [w for w in ws if isinstance(w, dict)]
    if not ws:
        sys.exit(f"{jsonl}: no write_stats blocks (run produced by the pre-instrumentation code?)")

    per: Dict[str, np.ndarray] = {k: np.asarray([w.get(k, 0) for w in ws], dtype=np.float64) for k in FIELDS}
    n_streamed = per["n_streamed"]
    rejected = per["n_reject_tau"] + per["n_reject_score"]
    with np.errstate(divide="ignore", invalid="ignore"):
        reject_frac_q = np.where(n_streamed > 0, rejected / n_streamed, np.nan)
        write_red_q = np.where(per["writes_mmr"] > 0, n_streamed / per["writes_mmr"], np.nan)
    t_episode = per["n_readout_total"] * t_end_s

    # conservation check: every streamed candidate is in exactly one bin
    binned = per["n_reject_tau"] + per["n_reject_score"] + per["n_write_append"] + per["n_write_replace"]
    conserved = bool(np.all(binned == n_streamed))

    tot_stream = float(n_streamed.sum())
    tot_rej = float(rejected.sum())
    tot_w_mmr = float(per["writes_mmr"].sum())
    derived = {
        # pooled over all queries (what the text should quote)
        "reject_frac_pooled": tot_rej / tot_stream if tot_stream else float("nan"),
        "reject_frac_tau_pooled": float(per["n_reject_tau"].sum()) / tot_stream if tot_stream else float("nan"),
        "reject_frac_score_pooled": float(per["n_reject_score"].sum()) / tot_stream if tot_stream else float("nan"),
        "write_reduction_pooled": tot_stream / tot_w_mmr if tot_w_mmr else float("nan"),
        "write_reduction_incl_anchors_pooled": (float(per["n_anchor"].sum()) + tot_stream)
            / float(per["writes_total"].sum()) if per["writes_total"].sum() else float("nan"),
        # per-query distributions
        "reject_frac_per_query": _stat(reject_frac_q[np.isfinite(reject_frac_q)]) if np.isfinite(reject_frac_q).any() else None,
        "write_reduction_per_query": _stat(write_red_q[np.isfinite(write_red_q)]) if np.isfinite(write_red_q).any() else None,
        "writes_total_per_query": _stat(per["writes_total"]),
        "writes_mmr_per_query": _stat(per["writes_mmr"]),
        "readouts_per_query": _stat(per["n_readout_total"]),
        "t_episode_s": _stat(t_episode),
        "t_end_s": t_end_s,
        "conservation_ok": conserved,
    }
    out = {
        "run_dir": run_dir, "jsonl": os.path.basename(jsonl), "n_queries": len(ws),
        "config": {k: rows[0].get(k) for k in ("domain", "mmr_lambda", "tau_dup", "gamma", "top_k", "mmr_mode", "centroid_mode")},
        "fields": {k: _stat(v) for k, v in per.items()},
        "derived": derived,
        "placeholders": {
            "retention_required_s": "TBD  (>= t_episode_s p95 = %.3e s under the 'hold until the MMR stream ends' definition)" % derived["t_episode_s"]["p95"],
            "endurance_required_cycles": "TBD  (= writes_total/query %.2f x queries per array lifetime, TBD)" % derived["writes_total_per_query"]["mean"],
        },
    }
    if check_against:
        out["identity_check"] = _check_against(rows, check_against)
    return out


def _print(o: Dict[str, Any]) -> None:
    d, f = o["derived"], o["fields"]
    print(f"\n=== {o['run_dir']}  (n = {o['n_queries']} queries, {o['config']})")
    print(f"  pool size            mean {f['n_pool']['mean']:.1f}   p95 {f['n_pool']['p95']:.0f}")
    print(f"  anchors / MMR slots  {f['n_anchor']['mean']:.1f} / {f['mmr_slots']['mean']:.1f}")
    print(f"  streamed candidates  mean {f['n_streamed']['mean']:.1f}   p95 {f['n_streamed']['p95']:.0f}   (fallback steps mean {f['n_stream_fallback']['mean']:.2f})")
    print(f"  rejected before write  {100*d['reject_frac_pooled']:.1f} %   (tau gate {100*d['reject_frac_tau_pooled']:.1f} % + score gate {100*d['reject_frac_score_pooled']:.1f} %)")
    print(f"  writes/episode: MMR  mean {d['writes_mmr_per_query']['mean']:.2f}  p95 {d['writes_mmr_per_query']['p95']:.0f}"
          f"   | total incl. anchors  mean {d['writes_total_per_query']['mean']:.2f}  p95 {d['writes_total_per_query']['p95']:.0f}")
    print(f"  write reduction      {d['write_reduction_pooled']:.2f}x (streamed / MMR writes)   {d['write_reduction_incl_anchors_pooled']:.2f}x incl. anchors")
    print(f"  readouts/episode     mean {d['readouts_per_query']['mean']:.1f}  p95 {d['readouts_per_query']['p95']:.0f}"
          f"   (main {f['n_readout_main']['mean']:.1f} + calib {f['n_readout_calib']['mean']:.1f})")
    print(f"  t_episode = readouts x T_END({d['t_end_s']:.1e} s):  mean {d['t_episode_s']['mean']:.3e} s   p95 {d['t_episode_s']['p95']:.3e} s")
    print(f"  conservation check   {'OK' if d['conservation_ok'] else 'FAILED'}")
    for k, v in o["placeholders"].items():
        print(f"  PLACEHOLDER {k}: {v}")
    if "identity_check" in o:
        ic = o["identity_check"]
        print(f"  identity vs {ic['reference']}: paired {ic['paired']}, L2 identical {ic['identical_l2']}, "
              f"MMR identical {ic['identical_mmr']}  -> {'IDENTICAL' if ic['identical'] else 'DIFFERS'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+", help="sweep_exports/<run_id> directories (or JSONL paths)")
    ap.add_argument("--igzo_fit", default="h_aimdc_sim/igzo_fit.json", help="IGZO fit JSON (T_END is read from circuit_params)")
    ap.add_argument("--check_against", default=None, help="reference run dir; assert retrieved doc ids are identical (paired by query text)")
    ap.add_argument("--no_write", action="store_true", help="do not write write_stats_summary.json")
    args = ap.parse_args()
    t_end = _t_end_s(args.igzo_fit)
    for rd in args.run_dirs:
        out = analyze(rd, t_end, args.check_against)
        _print(out)
        if not args.no_write:
            dest = os.path.join(rd if os.path.isdir(rd) else os.path.dirname(rd), "write_stats_summary.json")
            with open(dest, "w", encoding="utf-8") as fh:
                json.dump(out, fh, indent=2, ensure_ascii=False)
            print(f"  -> {dest}")


if __name__ == "__main__":
    main()
