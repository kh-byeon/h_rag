"""Crossings + figure for tol_probe outputs."""
import json, sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THRESH = (-0.005, -0.010)


def crossing(x, y, thr):
    """First |x| (log axis) where y (monotone-ish) drops below thr; log-linear interpolation. x sorted ascending, x>0."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    for i in range(1, len(x)):
        if y[i] <= thr and y[i - 1] > thr:
            lx = np.log10(x[i - 1]) + (thr - y[i - 1]) * (np.log10(x[i]) - np.log10(x[i - 1])) / (y[i] - y[i - 1])
            return float(10 ** lx)
    if len(y) and y[0] <= thr:
        return float(x[0])
    return float("nan")


def analyze(tag, outdir="out"):
    df = pd.read_csv(os.path.join(outdir, f"tol_sweep_{tag}.csv"))
    summ = json.load(open(os.path.join(outdir, f"tol_summary_{tag}.json")))
    fs = summ["full_scale_v"]
    res = {}
    lines = []
    for arm in ("mmr", "l2"):
        col = f"d_{arm}_recall"
        # offset: per sign
        for sgn, name in ((1, "offset_pos"), (-1, "offset_neg")):
            d = df[(df.axis == "offset") & (np.sign(df.value) == sgn)].copy()
            d["a"] = d.value.abs(); d = d.sort_values("a")
            r = {f"delta_{int(-t*1000)}pct".replace("pct", "mpct"): crossing(d.a, d[col], t) for t in THRESH}
            r = {("m0p5pct" if k.startswith("delta_5") else "m1pct"): v for k, v in r.items()}
            r["largest_ns_mv"] = float(d[d[f"{arm}_mcnemar_p"] > 0.05].a.max()) if (d[f"{arm}_mcnemar_p"] > 0.05).any() else float("nan")
            res[f"{arm}_{name}"] = r
        # gain: x = 1-g
        d = df[df.axis == "gain"].copy(); d["a"] = 1 - d.value; d = d.sort_values("a")
        res[f"{arm}_gain"] = {"m0p5pct_1mg": crossing(d.a, d[col], THRESH[0]), "m1pct_1mg": crossing(d.a, d[col], THRESH[1]),
                              "largest_ns_1mg": float(d[d[f"{arm}_mcnemar_p"] > 0.05].a.max()) if (d[f"{arm}_mcnemar_p"] > 0.05).any() else float("nan")}
        # cell sigma / cell gain sigma: seed-averaged
        for ax in ("cell_sigma", "cell_gain_sigma"):
            d = df[df.axis == ax].groupby("value").agg(y=(col, "mean"), ysd=(col, "std"), p=(f"{arm}_mcnemar_p", "max")).reset_index().sort_values("value")
            res[f"{arm}_{ax}"] = {"m0p5pct": crossing(d.value, d.y, THRESH[0]), "m1pct": crossing(d.value, d.y, THRESH[1]),
                                  "largest_ns": float(d[d.p > 0.05].value.max()) if (d.p > 0.05).any() else float("nan")}
    # console table
    def f(v, u=""):
        return "n/a" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.3g}{u}"
    lines.append(f"== {tag}  (FS {fs:.4f} V, in-pool {summ.get('in_pool_vectors','exact')}, baseline mmr_recall@5={summ['baseline_device']['mmr_recall']:.4f}, l2={summ['baseline_device']['l2_recall']:.4f})")
    lines.append(f"{'axis':>22s} | {'arm':>3s} | {'-0.5%':>10s} | {'-1%':>10s} | {'largest p>0.05':>15s}")
    for arm in ("mmr", "l2"):
        for k, u in (("offset_pos", " mV"), ("offset_neg", " mV")):
            r = res[f"{arm}_{k}"]; lines.append(f"{k:>22s} | {arm:>3s} | {f(r['m0p5pct'],u):>10s} | {f(r['m1pct'],u):>10s} | {f(r['largest_ns_mv'],u):>15s}")
        r = res[f"{arm}_gain"]; lines.append(f"{'gain (1-g)':>22s} | {arm:>3s} | {f(r['m0p5pct_1mg']):>10s} | {f(r['m1pct_1mg']):>10s} | {f(r['largest_ns_1mg']):>15s}")
        for k, u in (("cell_sigma", " mV"), ("cell_gain_sigma", "")):
            r = res[f"{arm}_{k}"]; lines.append(f"{k:>22s} | {arm:>3s} | {f(r['m0p5pct'],u):>10s} | {f(r['m1pct'],u):>10s} | {f(r['largest_ns'],u):>15s}")
    txt = "\n".join(lines); print(txt)
    json.dump({"tag": tag, "full_scale_v": fs, "crossings": res, "baseline": summ["baseline_device"], "fidelity": summ["fidelity"]},
              open(os.path.join(outdir, f"tol_crossings_{tag}.json"), "w"), indent=2)
    return df, res, summ


def figure(tags, outdir="out", fname="fig_tolerance_sweep.png"):
    plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
    colors = {tags[0]: "#1f4e79"}
    if len(tags) > 1:
        colors[tags[1]] = "#c0504d"
    for tag in tags:
        df = pd.read_csv(os.path.join(outdir, f"tol_sweep_{tag}.csv")); c = colors[tag]
        ax = axes[0]
        for sgn, ls, lab in ((1, "-", "+δ"), (-1, "--", "−δ")):
            d = df[(df.axis == "offset") & (np.sign(df.value) == sgn)].copy(); d["a"] = d.value.abs(); d = d.sort_values("a")
            ax.plot(d.a, 100 * d.d_mmr_recall, ls, color=c, marker="o", ms=2.5, lw=0.9, label=f"{tag} {lab} MMR")
            ax.plot(d.a, 100 * d.d_l2_recall, ls, color=c, marker="o", ms=2.5, lw=0.5, alpha=0.4, mfc="none")
        ax.set_xscale("log"); ax.set_xlabel("systematic V$_W$ offset |δ| (mV)")
        ax = axes[1]
        d = df[df.axis == "gain"].copy(); d["a"] = 1 - d.value; d = d.sort_values("a")
        ax.plot(d.a, 100 * d.d_mmr_recall, "-", color=c, marker="s", ms=2.5, lw=0.9, label=f"{tag} MMR")
        ax.plot(d.a, 100 * d.d_l2_recall, "-", color=c, marker="s", ms=2.5, lw=0.5, alpha=0.4, mfc="none", label=f"{tag} L2")
        ax.set_xscale("log"); ax.set_xlabel("range compression 1−g")
        ax = axes[2]
        for axn, mk, lab in (("cell_sigma", "^", "σ$_{cell}$ (mV)"), ("cell_gain_sigma", "v", "curvature σ (×100 → %)")):
            g = df[df.axis == axn].groupby("value").agg(y=("d_mmr_recall", "mean"), sd=("d_mmr_recall", "std"), y2=("d_l2_recall", "mean")).reset_index().sort_values("value")
            x = g.value if axn == "cell_sigma" else 100 * g.value
            ax.errorbar(x, 100 * g.y, yerr=100 * g.sd.fillna(0), fmt="-" + mk, color=c, ms=2.5, lw=0.9, capsize=1.5, label=f"{tag} {lab} MMR")
            ax.plot(x, 100 * g.y2, "-" + mk, color=c, ms=2.5, lw=0.5, alpha=0.4, mfc="none")
        ax.set_xscale("log"); ax.set_xlabel("per-cell σ (mV) / curvature σ (%)")
    for ax in axes:
        for thr in THRESH:
            ax.axhline(100 * thr, color="0.4", lw=0.6, ls=":")
        ax.axhline(0, color="0.7", lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=5, frameon=False, loc="lower left")
    axes[0].set_ylabel("Δ recall@5 (percentage points)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, fname), dpi=300); fig.savefig(os.path.join(outdir, fname.replace(".png", ".svg")))
    print("saved", os.path.join(outdir, fname))


if __name__ == "__main__":
    tags = sys.argv[1:]
    for t in tags:
        analyze(t)
    figure(tags)
