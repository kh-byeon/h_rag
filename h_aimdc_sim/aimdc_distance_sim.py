"""
AIMDC capacitive-boosting distance macro: CLI wrapper around aimdc_sim_core.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

import aimdc_sim_core as _core
from aimdc_sim_core import (
    DEFAULT_CIRCUIT_PARAMS,
    assert_docker_available,
    extract_i_ml,
    fit_igzo_model,
    generate_spice_deck,
    merge_circuit_params,
    run_docker_sim_with_retries,
)

# Sweep / plot defaults (overridable via CLI)
DEFAULT_V_PRG = 0.0
V_SL_MIN = -1.0
V_SL_MAX = 1.0
V_SL_STEP = 0.1


def plot_v_curve(v_sl_array: np.ndarray, i_list: list[float], *, v_prg: float) -> None:
    """Semilog V-curve, V_PRG marker, boosting note (PNG only)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    y = ma.masked_invalid(np.asarray(i_list, dtype=float))
    ax.semilogy(v_sl_array, y, "bo-", lw=2, ms=8, label="|I_ML|")
    ax.axvline(
        v_prg,
        color="r",
        ls="--",
        label=f"Stored V_prg = {v_prg} V",
    )
    ax.set_xlabel("Input Voltage V_SL (V)")
    ax.set_ylabel("|I_ML| (A)")
    ax.set_title("AIMDC Capacitive-Boosting Distance Macro (V-curve, L2-like)")
    ax.text(
        0.02,
        0.98,
        "Ideal boosting: alpha = 1 (SN fully floats with SL_bar)",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax.grid(True, which="both", ls="--")
    ax.legend()
    fig.tight_layout()
    out = "distance_v_curve.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved figure: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AIMDC distance macro Xyce sweep")
    parser.add_argument(
        "--stop-on-error",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Abort sweep on first Docker/Xyce or extract failure (default: true)",
    )
    parser.add_argument(
        "--csv",
        default="igzo_measure.csv",
        help="IGZO measurement CSV path (default: igzo_measure.csv)",
    )
    parser.add_argument(
        "--v-prg",
        type=float,
        default=DEFAULT_V_PRG,
        dest="v_prg",
        help=f"Stored program voltage V_PRG (default: {DEFAULT_V_PRG})",
    )
    args = parser.parse_args()
    stop_on_error: bool = args.stop_on_error
    v_prg = float(args.v_prg)
    csv_path = args.csv if os.path.isabs(args.csv) else os.path.abspath(args.csv)

    dbin = shutil.which("docker")
    if dbin is None:
        raise RuntimeError(
            "docker binary not found in PATH. Is Docker Desktop WSL integration enabled?"
        )
    _core.AIMDC_DOCKER_BIN = dbin  # type: ignore[attr-defined]

    assert_docker_available()

    params: dict[str, Any] = merge_circuit_params(None, v_prg)
    t_sample = float(DEFAULT_CIRCUIT_PARAMS["T_SAMPLE"])

    _popt, expr = fit_igzo_model(csv_path)
    v_sl_array = np.arange(V_SL_MIN, V_SL_MAX + V_SL_STEP / 2, V_SL_STEP)
    n = len(v_sl_array)
    i_list: list[float] = []

    for k, v in enumerate(v_sl_array, start=1):
        generate_spice_deck(expr, float(v), params, workdir=None)
        try:
            run_docker_sim_with_retries(
                workdir=None,
                k=k,
                n=n,
                v_sl=float(v),
            )
        except RuntimeError as e:
            print(f"[{k}/{n}] V_SL = {v:.2f} V  -> Docker/Xyce failed after retries: {e}")
            if stop_on_error:
                raise
            i_list.append(float("nan"))
            continue
        try:
            i_ml = extract_i_ml("sim_distance.csv", t_sample)
        except Exception as e:
            print(f"[{k}/{n}] extract failed: {e}")
            if stop_on_error:
                raise
            i_list.append(float("nan"))
            continue
        i_list.append(float(i_ml))
        print(f"[{k}/{n}] V_SL = {v:.2f} V  |I_ML| = {i_ml:g} A")

    plot_v_curve(v_sl_array, i_list, v_prg=v_prg)


if __name__ == "__main__":
    main()
