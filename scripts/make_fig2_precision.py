#!/usr/bin/env python3
"""Figure 2 — stored-value precision requirement per tier (b_c and b_e).

How many bits does each tier actually need per stored vector element?
Sweeps a per-cell / per-value Gaussian write-resolution sigma on the STORED
side of each tier and measures where end-to-end retrieval first degrades.

  static tier  (centroids)   : sigma on V_W, full scale 1.3106 V (measured
                               FeFET notch range; scaler fs1p31)
  dynamic tier (selected set): sigma on the stored embedding, full scale 3 V
                               (the +-1.5 V IGZO stored-value window; the +-3 V
                               figure is the SL drive swing, not the range a
                               stored element can occupy)

Bit convention (single, repo-wide):   bit = log2(FS / 6*sigma)
i.e. adjacent levels must be 6 sigma apart to be distinguishable. The other
convention seen in the repo, log2(FS/sigma), is 2.58 bit more optimistic and is
emitted as a second column for reference only.

Both tiers use the same retrieval proxy as sweep_exports/tol_run/tol_probe.py
(the one the published tolerance numbers were produced with): IVF probe of
nprobe centroids, cosine re-rank inside the pooled candidates, and Carbonell
MMR over the top-MMR_WINDOW. This is NOT the full on-the-fly two-phase MMR of
online_accelerator_device.py -- see fig2_notes.md.

    python scripts/make_fig2_precision.py --out figures/fig2
    python scripts/make_fig2_precision.py --smoke        # 2 sigmas, 1 seed
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sweep_exports", "tol_run"))
import tol_probe as tp                                    # noqa: E402

FIT = "h_aimdc_sim/fefet_ref_fit_smu.json"
SCALER_FS = "h_aimdc_sim/fefet_ref_fit_smu.fs1p31.scaler.json"
QUERY_EMB = "sweep_exports/query_emb_q1000_raw.npy"
V_ML = 1.4

FS_STATIC = 1.3106          # V, measured FeFET notch range (chip3, 2026-08-29)
# V, the +-1.5 V IGZO stored-value window. NOT the +-3 V SL drive swing:
# full scale here must be the span a stored element can actually occupy, and a
# 4T2C node holds +-1.5 V. Using 6.0 overstated every dynamic-tier bit by
# log2(6/3) = exactly 1.000 bit.
FS_DYNAMIC = 3.0
IGZO_SCALER = "h_aimdc_sim/igzo_fit.scaler.json"

SIG_STATIC = [0, 1, 2, 3.5, 5, 7.5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500]
# Halved together with FS_DYNAMIC (6 V -> 3 V) so that sig/1000 * x_span/FS,
# the quantity the sweep actually injects, is point-for-point what the old
# 6 V grid injected. Keeping the grid in step with FS is what let the existing
# fig2_precision.csv be relabelled rather than re-run, and it is what lets
# load_done() still recognise those rows as complete.
SIG_DYNAMIC = [0, 1, 2.5, 5, 10, 25, 50, 100, 200, 350, 500, 700, 1000]
SEEDS = [0, 1, 2, 3, 4]

# measured device resolutions, for reference lines only (not used in any bound)
MEAS_STATIC_MV = {"R1C1 (MBL -1.25)": 5.6, "R0C0 (MBL -1.75)": 18.4}

TOPK, MMR_WINDOW, MMR_LAMBDA = tp.TOPK, tp.MMR_WINDOW, tp.MMR_LAMBDA


# ------------------------------------------------------------------ metrics --
def _dcg(rel):
    return float(np.sum(rel / np.log2(np.arange(2, len(rel) + 2))))


def rank_metrics(ranked_docs, gold, n=10):
    """nDCG@n, MRR and MAP@n for a ranked list of doc ids against a gold set."""
    rel = np.isin(ranked_docs[:n], gold).astype(float)
    ideal = np.zeros(n); ideal[:min(len(gold), n)] = 1.0
    ndcg = _dcg(rel) / _dcg(ideal) if ideal.any() else 0.0
    hits = np.flatnonzero(rel)
    mrr = 1.0 / (hits[0] + 1) if hits.size else 0.0
    if hits.size:
        prec = np.cumsum(rel)[hits] / (hits + 1.0)
        ap = float(prec.sum() / min(len(gold), n))
    else:
        ap = 0.0
    return ndcg, mrr, ap


TAU_EVERY = 5      # every 5th of 1000 queries -> the 200-query subsample the
                   # static tier's Kendall tau already uses


def evaluate(ix, Q, sets, stored_noise=None, rng=None, idxs=None, base=None,
             tau_every=None, tau_ref=None):
    """Per-query metrics. `stored_noise` (volts) perturbs the STORED embeddings
    that form the MMR diversity term (the dynamic tier), mapped through the
    stored-value window; the query side and the L2 arm are untouched.

    With `tau_every` set, also returns `_rho` and `_selpos` for every
    `tau_every`-th query: `_rho[qi]` is the redundancy of each of the
    MMR_WINDOW candidates against the selected set, min_j d(c, S) written as
    max_j cos(c, S_j), and `_selpos` is the set S as window positions.

    `tau_ref` pins S to the *ideal* selection. Both orderings then have to be
    scored against one fixed reference set, which is what makes them
    comparable: letting each run use its own S would mix a change in the
    ordering with a change in what it is measured against. The window itself is
    already stored-noise-free -- it comes from the clean query-side scores --
    so the candidate list is identical run to run.
    """
    X, doc, members, gold = ix["X"], ix["doc_of_chunk"], ix["members"], ix["gold"]
    nq = Q.shape[0]
    keys = ("l2_recall", "l2_full", "mmr_recall", "mmr_full",
            "mmr_ndcg", "mmr_mrr", "mmr_map", "l2_ndcg", "top5_jaccard")
    res = {k: (np.zeros(nq) if base is None else base[k].copy()) for k in keys}
    res["_sel"] = {}
    res["_rho"] = {}
    res["_selpos"] = {}
    rows = range(nq) if idxs is None else idxs
    for qi in rows:
        pool = np.concatenate([members[int(c)] for c in sets[qi]])
        g = gold[qi]
        s = X[pool] @ Q[qi]
        order = np.argsort(-s)
        top_rows = pool[order[:TOPK]]
        top_docs = np.unique(doc[top_rows])
        found = np.isin(g, top_docs)
        res["l2_recall"][qi] = found.mean(); res["l2_full"][qi] = found.all()
        res["l2_ndcg"][qi] = rank_metrics(doc[pool[order[:10]]], g)[0]

        win = pool[order[:MMR_WINDOW]]
        sw = s[order[:MMR_WINDOW]]
        Xw = X[win]
        if stored_noise is not None:
            Xw = Xw + rng.normal(0.0, stored_noise, size=Xw.shape)
        cc = Xw @ Xw.T
        sel = tp.mmr_select(sw, cc, TOPK, MMR_LAMBDA)
        sel_rows = win[sel]
        if tau_every and qi % tau_every == 0:
            ref = tau_ref[qi] if tau_ref is not None else list(sel)
            res["_selpos"][qi] = list(sel)
            res["_rho"][qi] = cc[:, ref].max(axis=1)
        found = np.isin(g, np.unique(doc[sel_rows]))
        res["mmr_recall"][qi] = found.mean(); res["mmr_full"][qi] = found.all()
        nd, mr, ap = rank_metrics(doc[sel_rows], g, n=TOPK)
        res["mmr_ndcg"][qi] = nd; res["mmr_mrr"][qi] = mr; res["mmr_map"][qi] = ap
        res["top5_jaccard"][qi] = np.nan       # filled by the caller against ideal
        res["_sel"][qi] = set(int(v) for v in sel_rows)
    return res


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a or b) else 1.0


# ------------------------------------------------------------------- sweeps --
ROW_KEYS = ["tier", "regime", "sigma_mv", "seed", "n_changed", "l2_recall", "l2_full",
            "mmr_recall", "mmr_full", "mmr_ndcg", "mmr_mrr", "mmr_map", "l2_ndcg",
            "top5_jaccard", "centroid_kendall_tau"]


def load_done(path):
    if not os.path.exists(path):
        return {}, []
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    done = {(r["tier"], r["regime"], float(r["sigma_mv"]), int(r["seed"])) for r in rows}
    return done, rows


def append_row(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ROW_KEYS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in ROW_KEYS})


def sweep_static(ix, Q, C, tier, pq_tag, sigmas, seeds, done=frozenset(), path=None,
                 deadline=None):
    from scipy.stats import kendalltau
    d0 = tier.distances(Q, C)
    sets0 = tp.probe_sets(d0)
    base = evaluate(ix, Q, sets0)
    sel0 = base.pop("_sel")
    base.pop("_rho"); base.pop("_selpos")
    rows = []
    for sig in sigmas:
        for sd in (seeds if sig > 0 else [0]):
            if ("static", pq_tag, float(sig), sd) in done:
                continue
            if deadline and time.time() > deadline:
                print("  [deadline] stopping static sweep early", flush=True)
                return rows, base, sel0
            rng = np.random.default_rng(2000 + sd)
            off = rng.normal(0.0, sig / 1000.0, size=C.shape) if sig > 0 else None
            d = tier.distances(Q, C, cell_offset=off)
            sets = tp.probe_sets(d)
            changed = np.array([not np.array_equal(a, b) for a, b in zip(sets, sets0)])
            idx = np.where(changed)[0]
            ev = evaluate(ix, Q, sets, idxs=idx, base=base)
            sel = ev.pop("_sel")
            ev.pop("_rho"); ev.pop("_selpos")
            jac = np.ones(Q.shape[0])
            for qi in idx:
                jac[qi] = jaccard(sel0[qi], sel.get(qi, sel0[qi]))
            kt = np.array([kendalltau(d0[i], d[i]).statistic for i in range(0, Q.shape[0], 5)])
            rr = _row("static", pq_tag, sig, sd, ev, jac, kt, int(changed.sum()))
            rows.append(rr)
            if path:
                append_row(path, rr)
            print(f"  static/{pq_tag} sigma={sig:6.1f} mV seed={sd} done", flush=True)
    return rows, base, sel0


def sweep_dynamic(ix, Q, C, tier, sigmas, seeds, x_span, done=frozenset(), path=None,
                  deadline=None):
    from scipy.stats import kendalltau
    d0 = tier.distances(Q, C)
    sets0 = tp.probe_sets(d0)
    base = evaluate(ix, Q, sets0, tau_every=TAU_EVERY)
    sel0 = base.pop("_sel")
    # the ideal selection, and the ideal redundancy ordering measured against it
    selpos0 = base.pop("_selpos")
    rho0 = base.pop("_rho")
    rows = []
    for sig in sigmas:
        for sd in (seeds if sig > 0 else [0]):
            if ("dynamic", "exact", float(sig), sd) in done:
                continue
            if deadline and time.time() > deadline:
                print("  [deadline] stopping dynamic sweep early", flush=True)
                return rows, base, sel0
            rng = np.random.default_rng(3000 + sd)
            # sigma is a write resolution in mV on the stored node; divide by
            # the volts-per-embedding-unit of that same node to get embedding
            # units. FS_DYNAMIC here and FS_DYNAMIC in crossings() are the one
            # physical number, so the bit count log2(x_span / 6*noise_emb) it
            # ends up reporting does not depend on FS at all -- see fig2_notes.
            noise = (sig / 1000.0) * x_span / FS_DYNAMIC if sig > 0 else None
            ev = evaluate(ix, Q, sets0, stored_noise=noise, rng=rng, base=None,
                          tau_every=TAU_EVERY, tau_ref=selpos0)
            sel = ev.pop("_sel")
            rho = ev.pop("_rho"); ev.pop("_selpos")
            jac = np.array([jaccard(sel0[qi], sel.get(qi, sel0[qi])) for qi in range(Q.shape[0])])
            # The dynamic-tier counterpart of the static tier's centroid rank
            # tau: how far the stored-value error reorders the 50 candidates by
            # redundancy against the ideal selected set. Same column name and
            # same 200-query subsample, so the two tiers' tau rows are read the
            # same way.
            kt = np.array([kendalltau(rho0[qi], rho[qi]).statistic
                           for qi in sorted(rho0)])
            rr = _row("dynamic", "exact", sig, sd, ev, jac, kt, int((jac < 1.0).sum()))
            rows.append(rr)
            if path:
                append_row(path, rr)
            print(f"  dynamic sigma={sig:6.1f} mV seed={sd} done", flush=True)
    return rows, base, sel0


def _row(tier, regime, sig, seed, ev, jac, kt, n_changed):
    r = dict(tier=tier, regime=regime, sigma_mv=sig, seed=seed, n_changed=n_changed)
    for k, v in ev.items():
        r[k] = float(np.mean(v))
    r["top5_jaccard"] = float(np.mean(jac))
    r["centroid_kendall_tau"] = float(np.mean(kt)) if kt is not None else ""
    return r


# ------------------------------------------------------------- bit crossing --
METRICS = ["mmr_recall", "mmr_full", "mmr_ndcg", "mmr_mrr", "mmr_map",
           "l2_recall", "l2_ndcg", "top5_jaccard", "centroid_kendall_tau"]


def crossings(rows, tier, regime, fs_v, drop=0.01):
    """First sigma at which the seed-mean of each metric falls `drop` (relative)
    below its sigma=0 value, by linear interpolation in log(sigma)."""
    sel = [r for r in rows if r["tier"] == tier and r["regime"] == regime]
    if not sel:
        return []
    xs = sorted({r["sigma_mv"] for r in sel})
    out = []
    for m in METRICS:
        vals = []
        for x in xs:
            v = [r[m] for r in sel if r["sigma_mv"] == x and r[m] != ""]
            vals.append(np.mean([float(t) for t in v]) if v else np.nan)
        vals = np.array(vals, dtype=float)
        if np.isnan(vals).all():
            continue
        v0 = vals[0]
        if not np.isfinite(v0) or v0 == 0:
            continue
        thr = v0 * (1 - drop)
        sig_star = None
        for i in range(1, len(xs)):
            if np.isfinite(vals[i]) and vals[i] < thr:
                x1, x2 = max(xs[i - 1], 1e-3), xs[i]
                y1, y2 = vals[i - 1], vals[i]
                t = (thr - y1) / (y2 - y1) if y2 != y1 else 0.0
                sig_star = float(np.exp(np.log(x1) + t * (np.log(x2) - np.log(x1))))
                break
        rec = dict(tier=tier, regime=regime, metric=m, baseline=float(v0),
                   drop_pct=100 * drop, full_scale_v=fs_v,
                   sigma_star_mv=(sig_star if sig_star is not None else ""),
                   largest_tested_mv=xs[-1])
        if sig_star:
            rec["bits_6sigma"] = float(np.log2(fs_v / (6 * sig_star / 1000.0)))
            rec["bits_1sigma_ref"] = float(np.log2(fs_v / (sig_star / 1000.0)))
        else:
            rec["bits_6sigma"] = ""; rec["bits_1sigma_ref"] = ""
        out.append(rec)
    return out


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print("wrote", path, f"({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/fig2")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max_seconds", type=float, default=0,
                    help="stop cleanly after this many seconds; rerun to resume")
    a = ap.parse_args()
    os.makedirs(os.path.join(ROOT, a.out), exist_ok=True)
    t0 = time.time()

    sig_s = [0, 30, 300] if a.smoke else SIG_STATIC
    sig_d = [0, 100, 1400] if a.smoke else SIG_DYNAMIC
    seeds = [0] if a.smoke else SEEDS

    ix = tp.load_index(ROOT)
    Q = np.load(os.path.join(ROOT, QUERY_EMB)).astype(np.float64) @ ix["R"]
    C = ix["C"]
    tier = tp.FefetCentroidTier(os.path.join(ROOT, FIT), os.path.join(ROOT, SCALER_FS), v_ml=V_ML)
    sc = json.load(open(os.path.join(ROOT, IGZO_SCALER), encoding="utf-8"))
    x_span = float(sc["x_hi"]) - float(sc["x_lo"])
    print(f"IGZO scaler span x_hi-x_lo = {x_span:.6f} over a {FS_DYNAMIC} V window")

    csv_path = os.path.join(ROOT, a.out, "fig2_precision.csv")
    done, _ = load_done(csv_path)
    if done:
        print(f"resuming: {len(done)} points already in {csv_path}")
    deadline = (time.time() + a.max_seconds) if a.max_seconds else None

    want = ([("static", "exact", float(s_), sd) for s_ in sig_s for sd in (seeds if s_ > 0 else [0])]
            + ([] if a.smoke else
               [("static", "pq", float(s_), sd) for s_ in sig_s for sd in (seeds if s_ > 0 else [0])])
            + [("dynamic", "exact", float(s_), sd) for s_ in sig_d for sd in (seeds if s_ > 0 else [0])])
    todo = [w for w in want if w not in done]
    print(f"{len(want) - len(todo)}/{len(want)} points done, {len(todo)} remaining")

    if any(w[0] == "static" and w[1] == "exact" for w in todo):
        print("[static / exact]")
        sweep_static(ix, Q, C, tier, "exact", sig_s, seeds, done, csv_path, deadline)
    if (not a.smoke) and any(w[0] == "static" and w[1] == "pq" for w in todo) \
            and not (deadline and time.time() > deadline):
        print("[static / PQ]")
        ix_pq = dict(ix); ix_pq["X"] = np.load(os.path.join(ROOT, "X_pq.npy")).astype(np.float64)
        sweep_static(ix_pq, Q, C, tier, "pq", sig_s, seeds, done, csv_path, deadline)
    if any(w[0] == "dynamic" for w in todo) and not (deadline and time.time() > deadline):
        print("[dynamic / exact]")
        sweep_dynamic(ix, Q, C, tier, sig_d, seeds, x_span, done, csv_path, deadline)

    done2, rows = load_done(csv_path)
    for r in rows:
        for k in r:
            if k not in ("tier", "regime") and r[k] != "":
                r[k] = float(r[k])
    remaining = [w for w in want if w not in done2]
    print(f"{len(want) - len(remaining)}/{len(want)} points complete"
          + (f"; {len(remaining)} remaining -- rerun to resume" if remaining else ""))
    cr = (crossings(rows, "static", "exact", FS_STATIC)
          + crossings(rows, "static", "pq", FS_STATIC)
          + crossings(rows, "dynamic", "exact", FS_DYNAMIC))
    write_csv(os.path.join(ROOT, a.out, "fig2_precision_bits.csv"), cr)
    json.dump(dict(full_scale_static_v=FS_STATIC, full_scale_dynamic_v=FS_DYNAMIC,
                   full_scale_dynamic_note=(
                       "stored-value window of a 4T2C node, +-1.5 V. NOT the "
                       "+-3 V SL drive swing. This one number both converts "
                       "sigma into embedding units and divides the bit "
                       "conversion, so bits_6sigma does not depend on it; what "
                       "depends on it is the volt label on sigma."),
                   noise_unit="volts on the stored node (sigma_mv column is mV)",
                   bit_convention="log2(FS / 6*sigma)",
                   bits_below_1_reporting="report as '< 1', never as a negative number",
                   measured_static_sigma_mv=MEAS_STATIC_MV,
                   igzo_x_span=x_span, n_queries=int(Q.shape[0]),
                   nprobe=tp.NPROBE, topk=TOPK, mmr_window=MMR_WINDOW,
                   mmr_lambda=MMR_LAMBDA,
                   proxy="tol_probe Carbonell MMR, not the pipeline two-phase MMR",
                   wall_clock_s=round(time.time() - t0, 1)),
              open(os.path.join(ROOT, a.out, "fig2_precision_meta.json"), "w"), indent=2)
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
