from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aimdc_device_distance import DeviceDistance, NoiseModel, noise_model_from_json_path


def _cmd_quicklook(args: argparse.Namespace) -> None:
    scaler = args.scaler
    if not scaler and args.fit:
        stem, _ = os.path.splitext(os.path.abspath(args.fit))
        cand = f"{stem}.scaler.json"
        if os.path.isfile(cand):
            scaler = cand
    noise = NoiseModel()
    dd = DeviceDistance(
        args.fit,
        scaler_path=scaler,
        reduce="sum",
        use_log=True,
        noise=noise,
    )
    v_sl = np.linspace(-1.0, 1.0, 21, dtype=np.float64)
    v_prg = np.zeros_like(v_sl)
    i_an = np.asarray(dd.raw_bit_current(v_sl, v_prg), dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogy(v_sl, i_an, "bo-", lw=2, ms=6, label="analytical I_bit")
    if args.ref_npz and os.path.isfile(args.ref_npz):
        z = np.load(args.ref_npz)
        vx = np.asarray(z["v_sl"], dtype=float)
        iy = np.asarray(z["i_xyce"], dtype=float)
        ax.semilogy(vx, iy, "r.--", lw=1, ms=4, label="reference")
    ax.set_xlabel("V_SL (V)")
    ax.set_ylabel("|I| (A)")
    ax.set_title("AIMDC analytical per-bit current (V_PRG=0)")
    ax.grid(True, which="both", ls="--")
    ax.legend()
    fig.tight_layout()
    out = args.out or "quicklook_vcurve.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


def _cmd_noise_study(args: argparse.Namespace) -> None:
    stem, _ = os.path.splitext(os.path.abspath(args.fit))
    scaler = args.scaler or (f"{stem}.scaler.json" if os.path.isfile(f"{stem}.scaler.json") else None)
    noise = (
        noise_model_from_json_path(args.noise)
        if args.noise
        else NoiseModel(sigma_vth=0.05)
    )
    if args.seed is not None:
        from dataclasses import replace

        noise = replace(noise, rng_seed=int(args.seed))
    dd = DeviceDistance(
        args.fit,
        scaler_path=scaler,
        reduce="sum",
        use_log=True,
        noise=noise,
    )
    rng = np.random.default_rng(0)
    if dd._x_lo is None:
        emb = rng.standard_normal((512, 64))
        dd.fit_scaler(emb)
    d = int(dd.embedding_dim) if dd.embedding_dim > 0 else 64
    q = rng.standard_normal(d).astype(np.float64)
    c = rng.standard_normal(d).astype(np.float64)
    dists: list[float] = []
    n_trials = int(args.n_trials)
    for t in range(n_trials):
        dd.reset_rng((noise.rng_seed or 0) + 1009 * t)
        dists.append(dd.distance(q, c))
    arr = np.asarray(dists, dtype=np.float64)
    summary: dict[str, Any] = {
        "n_trials": n_trials,
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "p5": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }
    jpath = args.summary_json or "noise_study_summary.json"
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {jpath}", summary)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(arr, bins=min(48, max(8, n_trials // 16)))
    ax.set_title("Device distance (Monte Carlo)")
    ax.set_xlabel("distance")
    ax.set_ylabel("count")
    fig.tight_layout()
    hpath = args.hist_png or "noise_study_hist.png"
    fig.savefig(hpath, dpi=200)
    plt.close(fig)
    print(f"Saved {hpath}")


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m aimdc_device_distance")
    sub = p.add_subparsers(dest="cmd", required=True)

    pq = sub.add_parser("quicklook", help="Plot analytical V-curve (optional NPZ overlay)")
    pq.add_argument("--fit", required=True, help="igzo_fit.json")
    pq.add_argument("--scaler", default=None, help="Scaler JSON (default: <fit_stem>.scaler.json)")
    pq.add_argument("--ref-npz", dest="ref_npz", default=None)
    pq.add_argument("--out", default=None)
    pq.set_defaults(func=_cmd_quicklook)

    pn = sub.add_parser("noise-study", help="Monte Carlo histogram of distance(q,c)")
    pn.add_argument("--fit", required=True)
    pn.add_argument("--noise", default=None, help="JSON for NoiseModel fields")
    pn.add_argument("--scaler", default=None)
    pn.add_argument("--n-trials", dest="n_trials", type=int, default=512)
    pn.add_argument("--seed", type=int, default=None)
    pn.add_argument("--summary-json", dest="summary_json", default=None)
    pn.add_argument("--hist-png", dest="hist_png", default=None)
    pn.set_defaults(func=_cmd_noise_study)

    ns = p.parse_args()
    func = getattr(ns, "func", None)
    if func is None:
        p.print_help()
        sys.exit(1)
    func(ns)


if __name__ == "__main__":
    main()
