"""Retrieval performance vs storage/read resolution (per-cell V_W noise σ).

Sweeps per-cell Gaussian σ on the FeFET centroid V_W from ~0 up to 500 mV,
several seeds each, for both in-pool regimes (exact embeddings and PQ), and
plots ABSOLUTE recall@5 (MMR + L2) vs σ. Measured device resolutions are
marked: R1C1 σ=5.6 mV (best read cond.), R0C0 σ=18.4 mV. A secondary axis
shows the equivalent storage resolution in bits over the 1.3106 V full scale.
"""
import os, sys, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("HAIMDC_ROOT", os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import tol_probe as tp

FS = 1.3106                       # full-scale V_W window (measured notch range)
SIGMAS = [0.0, 1, 2, 3.5, 5, 7.5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500]
SEEDS = [0, 1, 2, 3, 4]
MEAS = {"R1C1 (MBL −1.25)": 5.6, "R0C0 (MBL −1.75)": 18.4}   # measured 1σ read noise (mV)


def run(query_emb, pq=None, v_ml=1.4, scaler="h_aimdc_sim/fefet_ref_fit_smu.fs1p31.scaler.json",
        fit="h_aimdc_sim/fefet_ref_fit_smu.json"):
    ix = tp.load_index(ROOT)
    if pq:
        ix["X"] = np.load(os.path.join(ROOT, pq)).astype(np.float64)
    Q = np.load(os.path.join(ROOT, query_emb)).astype(np.float64) @ ix["R"]
    C = ix["C"]
    tier = tp.FefetCentroidTier(os.path.join(ROOT, fit), os.path.join(ROOT, scaler), v_ml=v_ml)
    d0 = tier.distances(Q, C); sets0 = tp.probe_sets(d0)
    base = tp.evaluate(ix, Q, sets0)
    rows = []
    for sig in SIGMAS:
        for sd in SEEDS:
            if sig == 0.0 and sd != 0:
                continue
            rng = np.random.default_rng(2000 + sd)
            off = rng.normal(0.0, sig / 1000.0, size=C.shape) if sig > 0 else None
            d = tier.distances(Q, C, cell_offset=off)
            sets = tp.probe_sets(d)
            changed = np.array([not np.array_equal(a, b) for a, b in zip(sets, sets0)])
            idx = np.where(changed)[0]
            ev = tp.evaluate(ix, Q, sets, idxs=idx)
            mmr = base["mmr_hit"].copy(); mmr[idx] = ev["mmr_hit"][idx]
            l2 = base["l2_hit"].copy(); l2[idx] = ev["l2_hit"][idx]
            mrec = base["mmr_recall"].copy(); mrec[idx] = ev["mmr_recall"][idx]
            lrec = base["l2_recall"].copy(); lrec[idx] = ev["l2_recall"][idx]
            gc = base["gold_cov"].copy(); gc[idx] = ev["gold_cov"][idx]
            rows.append(dict(sigma_mv=sig, seed=sd,
                             mmr_recall=float(mrec.mean()), l2_recall=float(lrec.mean()),
                             mmr_hit=float(mmr.mean()), l2_hit=float(l2.mean()),
                             gold_cov=float(gc.mean()), n_changed=int(changed.sum())))
        print(f"  sigma={sig:6.1f} mV done")
    return rows, dict(mmr_recall=float(base["mmr_recall"].mean()), l2_recall=float(base["l2_recall"].mean()))


def agg(rows, key):
    import collections
    d = collections.defaultdict(list)
    for r in rows:
        d[r["sigma_mv"]].append(r[key])
    xs = sorted(d)
    return np.array(xs), np.array([np.mean(d[x]) for x in xs]), np.array([np.std(d[x]) for x in xs])


def main():
    t0 = time.time()
    out = os.path.join(HERE, "out")
    os.makedirs(out, exist_ok=True)
    query_emb = "sweep_exports/query_emb_q1000_raw.npy"
    print("[exact]"); rows_e, base_e = run(query_emb, pq=None)
    print("[pq]");    rows_p, base_p = run(query_emb, pq="X_pq.npy")
    json.dump({"exact": rows_e, "pq": rows_p, "base_exact": base_e, "base_pq": base_p,
               "full_scale_v": FS, "measured_sigma_mv": MEAS},
              open(os.path.join(out, "res_curve.json"), "w"), indent=2)

    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6})
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
    for ax, (rows, base, title) in zip(axes, [(rows_e, base_e, "exact re-rank (upper bound)"),
                                              (rows_p, base_p, "PQ re-rank (pipeline)")]):
        for key, col, lab in [("mmr_recall", "#1f4e79", "MMR recall@5"),
                              ("l2_recall", "#c0504d", "L2 recall@5")]:
            x, m, s = agg(rows, key)
            xm = np.where(x == 0, 0.4, x)   # place σ=0 at left tick on log axis
            ax.plot(xm, m, "-o", color=col, ms=3, lw=1.1, label=lab)
            ax.fill_between(xm, m - s, m + s, color=col, alpha=0.15, lw=0)
        ax.set_xscale("log")
        for name, sig in MEAS.items():
            ax.axvline(sig, color="0.35", lw=0.8, ls="--")
            ax.text(sig, ax.get_ylim()[0], "  " + name.split(" ")[0], rotation=90,
                    va="bottom", ha="left", fontsize=6, color="0.3")
        ax.set_xlabel("per-cell V$_W$ resolution σ (mV)")
        ax.set_title(title, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, which="both", axis="y", lw=0.3, alpha=0.4)
        # secondary axis: storage resolution in bits = log2(FS / (6σ))
        sec = ax.secondary_xaxis("top", functions=(lambda s: np.log2(FS / (6 * np.clip(s, 1e-6, None) / 1000)),
                                                    lambda b: 1000 * FS / (6 * 2 ** b)))
        sec.set_xlabel("storage resolution (bit, 6σ over 1.31 V)", fontsize=7)
    axes[0].set_ylabel("recall@5")
    axes[1].legend(fontsize=6.5, frameon=False, loc="lower left")
    axes[0].legend(fontsize=6.5, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_recall_vs_resolution.png"), dpi=300)
    fig.savefig(os.path.join(out, "fig_recall_vs_resolution.svg"))
    print(f"[done] {time.time()-t0:.0f}s -> {out}/fig_recall_vs_resolution.png")

    # console summary: recall drop vs baseline at each sigma (MMR, both regimes)
    for tag, rows, base in [("exact", rows_e, base_e), ("pq", rows_p, base_p)]:
        x, m, s = agg(rows, "mmr_recall")
        print(f"\n{tag}: baseline MMR recall@5 = {base['mmr_recall']:.4f}")
        for xi, mi in zip(x, m):
            print(f"  σ={xi:6.1f} mV  MMR recall@5={mi:.4f}  Δ={100*(mi-base['mmr_recall']):+.2f}%p")


if __name__ == "__main__":
    main()
