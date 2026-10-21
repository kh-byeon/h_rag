#!/usr/bin/env python3
"""Figure 2 analytical counts (problem-definition figure).

Device-free. Every count is derived by counting operations in the algorithm,
with N, |S| and the write counts taken from the MEASURED write_stats k-sweep
(sweep_exports/write_stats_k/<scope>/k<K>/<run_id>/write_stats_summary.json,
n=1000, produced by scripts/run_write_stats_ksweep.sh).

No device energies, latencies or FeFET/IGZO parameters appear in any output.

Usage (repo root):
    python scripts/make_fig2_counts.py [--out figures/fig2]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

# --------------------------------------------------------------- parameters --
D = 768           # embedding dimension           (offline_data_*/config.pkl)
N_LIST = 128      # IVF centroids                 (config.pkl)
N_PROBE = 32      # clusters probed per query     (online_accelerator_device.py:3767)
M_PQ = 8          # PQ code bytes per vector      (8 subquantizers x uint8)
B_E = 4           # bytes per embedding element as stored digitally (float32)
B_C = 4           # bytes per centroid element (float32)
R_ROUNDS = 1      # MMR rounds per query (measured: n_stream_fallback ~ 0.005/query)

# Q_rebuild scenarios. ASSUMED -- no value for this exists in the repository.
# query rates are stated per day; the corpus is rebuilt at the stated cadence.
Q_REBUILD = {
    "S1_edge_nightly_1e2":       1.0e2,   # personal/edge deployment, nightly rebuild
    "S2_team_nightly_1e3":       1.0e3,   # team deployment, nightly rebuild
    "S3_enterprise_nightly_1e4": 1.0e4,   # enterprise, nightly rebuild
    "S4_enterprise_weekly_1e4":  7.0e4,   # enterprise, weekly rebuild at 1e4/day
}


def load_runs(root: str):
    """Measured per-query means from the n=1000 k-sweep."""
    out = []
    pat = os.path.join(root, "sweep_exports", "write_stats_k", "*", "k*", "*",
                       "write_stats_summary.json")
    for f in sorted(glob.glob(pat)):
        d = json.load(open(f, encoding="utf-8"))
        if d["n_queries"] != 1000:
            continue                       # skip the NQ=20 smoke runs
        F = d["fields"]
        mean = lambda n: F[n]["mean"]
        rec = dict(
            run_id=os.path.basename(os.path.dirname(f)),
            scope=f.split(os.sep + "write_stats_k" + os.sep)[1].split(os.sep)[0],
            k=int(d["config"]["top_k"]),
            n_queries=d["n_queries"],
            n_pool=mean("n_pool"),
            N=mean("n_streamed"),
            S=mean("buffer_rows_final"),
            n_anchor=mean("n_anchor"),
            w_append=mean("n_write_append"),
            w_replace=mean("n_write_replace"),
            w_total=mean("writes_total"),
            n_calib=mean("n_readout_calib"),
            rd_total=mean("n_readout_total"),
        )
        # attach the L2-arm redundancy aggregates from the same run
        csvp = os.path.join(os.path.dirname(f), "sweep_results_full_grid.csv")
        if os.path.exists(csvp):
            row = list(csv.DictReader(open(csvp, encoding="utf-8")))[0]
            for c in ("l2_rr", "mmr_rr", "l2_ild", "mmr_ild", "l2_breadth",
                      "mmr_breadth", "l2_recall", "mmr_recall"):
                rec[c] = float(row[c]) if row.get(c) not in (None, "") else None
        rr_l2, rr_mmr = rec.get("l2_rr"), rec.get("mmr_rr")
        rec["mmr_active"] = bool(
            rec["N"] > 0 and rr_l2 is not None and rr_mmr is not None
            and abs(rr_l2 - rr_mmr) > 1e-9)
        out.append(rec)
    out.sort(key=lambda r: (r["scope"], r["k"]))
    return out


# ------------------------------------------------------------ 3.1 counts ----
def counts_rows(runs):
    """One row per (parameter set, data object, implementation)."""
    rows = []
    for r in runs:
        N, S, W, nc = r["N"], r["S"], r["w_total"], r["n_calib"]
        base = dict(scope=r["scope"], run_id=r["run_id"], N=round(N, 3), k=r["k"],
                    d=D, n_list=N_LIST, m=M_PQ, r=R_ROUNDS,
                    S_measured=round(S, 3), writes_rows=round(W, 3),
                    calib_calls=round(nc, 3), mmr_active=r["mmr_active"])

        def add(obj, impl, formula, re_, we_, rb=None, wb=None):
            rows.append(dict(base, object=obj, impl=impl, formula=formula,
                             reads_elem=re_, writes_elem=we_,
                             reads_bytes=(re_ * B_E if rb is None else rb),
                             writes_bytes=(we_ * B_E if wb is None else wb),
                             reads_vec=re_ / D, writes_vec=we_ / D))

        for impl in ("software", "in-array"):
            add("centroids", impl, "n_list * d", N_LIST * D, 0.0)
            add("candidates_pq", impl, "n_pool * m (bytes)", 0.0, 0.0,
                rb=r["n_pool"] * M_PQ, wb=0.0)
            add("candidate_broadcast", impl, "N * d", N * D, 0.0)

        add("selected_set_S", "software", "N * |S| * d  reads / writes_rows * d",
            N * S * D, W * D)
        add("selected_set_S", "in-array", "N scalars out / writes_rows * d",
            N * 1.0, W * D)
        add("standardization_pass", "software", "n_c * (n_c - 1) * d",
            nc * max(nc - 1.0, 0.0) * D, 0.0)
        add("standardization_pass", "in-array", "n_c scalars out",
            nc * 1.0, 0.0)
    return rows


# ------------------------------------------------------------ 3.2 totals ----
def totals_rows(runs):
    rows = []
    for r in runs:
        N, S, nc = r["N"], r["S"], r["n_calib"]
        # diversity stage only, elements per query
        bc = N * D                                    # candidate broadcast (both)
        sw_div = N * S * D                            # re-read S per candidate
        ar_div = N * 1.0                              # one scalar per candidate
        sw_cal = nc * max(nc - 1.0, 0.0) * D
        ar_cal = nc * 1.0
        cal_bc = nc * D

        T_sw_excl = sw_div + sw_cal                   # broadcast NOT counted
        T_ar_excl = ar_div + ar_cal
        T_sw_incl = bc + sw_div + cal_bc + sw_cal     # broadcast counted (handoff 3.2)
        T_ar_incl = bc + ar_div + cal_bc + ar_cal

        rows.append(dict(
            scope=r["scope"], k=r["k"], N=round(N, 3), d=D, S_measured=round(S, 3),
            calib_calls=round(nc, 3),
            mmr_active=r["mmr_active"],
            T_sw_diversity_stage=T_sw_incl, T_arr_diversity_stage=T_ar_incl,
            ratio_diversity_stage=(T_sw_incl / T_ar_incl) if T_ar_incl else float("nan"),
            expected_1_plus_k=1 + r["k"],
            T_sw_S_array_only=T_sw_excl, T_arr_S_array_only=T_ar_excl,
            ratio_S_array_only=(T_sw_excl / T_ar_excl) if T_ar_excl else float("nan"),
        ))
    return rows


# ------------------------------------------------------------- 3.3 tiers ----
def tier_rows(runs):
    """Fig 2b coordinates. reads_per_write is defined PER CELL:
    how many times a cell is read between two writes to that cell.
      static  (centroids): read once per query, written once per rebuild -> Q_rebuild
      dynamic (S): every readout touches every row, so a cell is read rd_total
        times per query; a given row is rewritten w_total/|S| times per query
        -> rd_total * |S| / w_total
    lifetime is in queries: |S| / w_total for the dynamic tier (< 1 query),
    Q_rebuild for the static tier.
    """
    rows = []
    for name, q in Q_REBUILD.items():
        rows.append(dict(tier="static tier (centroids)", scope="-", k="-",
                         scenario=name, reads_per_write=q, lifetime_queries=q,
                         basis="reads once per query; written once per rebuild"))
    for r in runs:
        if not r["mmr_active"] or r["w_total"] <= 0:
            continue      # MMR inert: L2 and MMR metrics are bit-identical
        rpw = r["rd_total"] * r["S"] / r["w_total"]
        life = r["S"] / r["w_total"]
        rows.append(dict(tier="dynamic tier (selected set S)", scope=r["scope"],
                         k=r["k"], scenario="measured n=1000",
                         reads_per_write=rpw, lifetime_queries=life,
                         basis="rd_total * |S| / writes_rows ; |S| / writes_rows"))
    return rows


def gap_rows(tiers):
    import math
    dyn = [t for t in tiers if t["tier"].startswith("dynamic")]
    sta = [t for t in tiers if t["tier"].startswith("static")]
    if not dyn or not sta:
        return []
    life_dyn = [t["lifetime_queries"] for t in dyn]
    out = []
    for s in sta:
        gaps = [math.log10(s["lifetime_queries"] / ld) for ld in life_dyn]
        out.append(dict(scenario=s["scenario"], Q_rebuild=s["lifetime_queries"],
                        gap_decades_min=min(gaps), gap_decades_max=max(gaps),
                        four_orders_holds=(min(gaps) >= 4.0)))
    return out


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", path, f"({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="figures/fig2")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    runs = load_runs(a.root)
    if not runs:
        raise SystemExit("no n=1000 write_stats_k runs found -- run scripts/run_write_stats_ksweep.sh")
    print(f"loaded {len(runs)} measured runs: "
          + ", ".join(f"{r['scope']}/k{r['k']}" for r in runs))

    write_csv(os.path.join(a.out, "fig2_counts.csv"), counts_rows(runs))
    write_csv(os.path.join(a.out, "fig2_totals.csv"), totals_rows(runs))
    tiers = tier_rows(runs)
    write_csv(os.path.join(a.out, "fig2_tiers.csv"), tiers)
    write_csv(os.path.join(a.out, "fig2_lifetime_gap.csv"), gap_rows(tiers))
    write_csv(os.path.join(a.out, "fig2_redundancy.csv"), [
        dict(scope=r["scope"], k=r["k"], n_queries=r["n_queries"],
             mmr_active=r["mmr_active"],
             l2_near_dup_rate=r.get("l2_rr"), mmr_near_dup_rate=r.get("mmr_rr"),
             l2_intra_list_diversity=r.get("l2_ild"), mmr_intra_list_diversity=r.get("mmr_ild"),
             l2_centroid_breadth=r.get("l2_breadth"), mmr_centroid_breadth=r.get("mmr_breadth"),
             note="corpus aggregate over 1000 queries; cosine>0.7 pair fraction; "
                  "no per-query distribution is persisted by the pipeline")
        for r in runs])
    write_csv(os.path.join(a.out, "fig2_measured_inputs.csv"), runs)


if __name__ == "__main__":
    main()
