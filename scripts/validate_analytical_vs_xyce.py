"""
Manual validation: 5x5 grid in [-1,1]^2, Xyce vs analytical per-bit I (noise off).
Requires Docker + Xyce image (see aimdc_distance_sim.py).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
_SIM = _REPO / "h_aimdc_sim"
# Repo root: needed for aimdc_device_distance
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
# h_aimdc_sim/: needed for aimdc_sim_core
if str(_SIM) not in sys.path:
    sys.path.insert(0, str(_SIM))

import aimdc_sim_core as core  # noqa: E402
from aimdc_device_distance import DeviceDistance, NoiseModel  # noqa: E402


def main() -> None:
    fit_json = _REPO / "h_aimdc_sim" / "igzo_fit.json"
    if not fit_json.is_file():
        raise SystemExit(f"Missing {fit_json}; run aimdc_sim_core.py fit first.")
    os.chdir(_SIM)
    popt, _meta = core.load_fit_json(str(fit_json))
    expr = core.xyce_b_source_expression(popt)
    dd = DeviceDistance(str(fit_json), scaler_path=None, noise=NoiseModel())
    dd._x_lo, dd._x_hi = -1.0, 1.0
    dd.v_min, dd.v_max = -1.0, 1.0
    dd.embedding_dim = 1

    grid = np.linspace(-1.0, 1.0, 5, dtype=np.float64)
    rows: list[dict[str, float]] = []
    core.assert_docker_available()
    t_sample = float(core.DEFAULT_CIRCUIT_PARAMS["T_SAMPLE"])
    for v_prg in grid:
        for v_sl in grid:
            params = core.merge_circuit_params(None, float(v_prg))
            core.generate_spice_deck(expr, float(v_sl), params, workdir=".")
            core.run_docker_sim(workdir=".")
            i_xy = float(core.extract_i_ml("sim_distance.csv", t_sample))
            i_an = float(dd.raw_bit_current(np.array([v_sl]), np.array([v_prg]))[0])
            rows.append(
                {
                    "v_sl": float(v_sl),
                    "v_prg": float(v_prg),
                    "i_xyce": i_xy,
                    "i_analytical": i_an,
                }
            )

    df = pd.DataFrame(rows)
    df["log10_i_xyce"] = np.log10(np.maximum(df["i_xyce"], 1e-30))
    df["log10_i_an"] = np.log10(np.maximum(df["i_analytical"], 1e-30))
    rmse = float(
        np.sqrt(np.mean((df["log10_i_xyce"] - df["log10_i_an"]) ** 2))
    )
    out_csv = _REPO / "analytical_vs_xyce.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} RMSE_log10={rmse:.4f}")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(df["log10_i_xyce"], df["log10_i_an"], s=36)
    lim = [df["log10_i_xyce"].min(), df["log10_i_xyce"].max()]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("log10 I Xyce")
    ax.set_ylabel("log10 I analytical")
    ax.set_title(f"AIMDC grid (RMSE={rmse:.3f})")
    fig.tight_layout()
    png = _REPO / "analytical_vs_xyce_scatter.png"
    fig.savefig(png, dpi=200)
    plt.close(fig)
    print(f"Wrote {png}")
    if rmse >= 0.3:
        raise SystemExit(f"RMSE log10 I too large: {rmse} >= 0.3")


if __name__ == "__main__":
    main()
