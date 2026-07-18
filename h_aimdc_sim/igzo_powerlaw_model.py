from __future__ import annotations

import argparse
import os
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numpy.typing import NDArray
from scipy.optimize import curve_fit  # type: ignore[import-untyped]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Shared with Xyce B-source emission (aimdc_sim_core_v3.XYCE_POWER_EPS).
POWERLAW_EPS = 1e-15


def igzo_model(
    V_gs: NDArray[np.floating[Any]] | NDArray[np.float64] | np.ndarray,
    I_off: float,
    V_th: float,
    ss_v_per_dec: float,
    I0_sub: float,
    I_sat_max: float,
    V_sat_scale: float,
    gamma: float,
    delta: float,
) -> NDArray[np.floating[Any]]:
    v = np.asarray(V_gs, dtype=float)
    v_gst = v - V_th
    v_gst_smooth = 0.5 * (v_gst + np.sqrt(v_gst * v_gst + delta * delta))

    # Smooth subthreshold clamp (matches SPICE: cap * tanh(x/cap)).
    SUB_CAP = 5.0
    exp_arg = SUB_CAP * np.tanh((v - V_th) / ss_v_per_dec / SUB_CAP)
    i_sub = I0_sub * np.exp(exp_arg)
    tanh_arg = v_gst_smooth / V_sat_scale
    tanh_part = np.tanh(tanh_arg)
    i_above = I_sat_max * np.power(np.maximum(tanh_part, 0.0) + POWERLAW_EPS, gamma)
    ids = I_off + i_sub + i_above
    y = np.log10(np.maximum(np.abs(ids), 1e-30))
    return cast(NDArray[np.floating[Any]], y)


def fit_igzo_powerlaw(
    csv_path: str,
    *,
    vds: float | None = None,
) -> tuple[NDArray[np.float64], dict[str, float | list[str]]]:
    df = pd.read_csv(csv_path)
    df.columns = [str(c).replace("#", "").strip().lower() for c in df.columns]
    missing = {"vgs", "vds", "ids"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns after normalization: {sorted(missing)}")

    seen: list[float] = []
    for v in df["vds"].astype(float):
        fv = float(v)
        if not any(np.isclose(fv, s) for s in seen):
            seen.append(fv)
    if not seen:
        raise ValueError("No Vds values found in CSV")
    if vds is not None:
        vds0 = float(vds)
        if not any(np.isclose(vds0, s) for s in seen):
            raise ValueError(
                f"Requested vds={vds0} not in CSV unique Vds={seen}"
            )
    else:
        # Deterministic: prefer closest to 0.5 V, else smallest |Vds|.
        vds0 = float(min(seen, key=lambda x: (abs(abs(x) - 0.5), abs(x), x)))


    meas = df[np.isclose(df["vds"].astype(float), vds0)].sort_values("vgs")
    x = meas["vgs"].values.astype(float)
    ids = np.abs(meas["ids"].values.astype(float))
    y_log = np.log10(ids + 1e-15)

    i_off0 = float(10.0 ** np.min(y_log))
    if x.size >= 3:
        dy = np.gradient(y_log, x)
        idx = int(np.argmax(dy))
        v_th0 = float(x[idx])
    else:
        y_med = float(np.median(y_log))
        idx = int(np.argmin(np.abs(y_log - y_med)))
        v_th0 = float(x[idx]) if x.size else 0.0

    vss0 = 0.1

    sub_mask = x < (v_th0 - 0.5)
    if np.count_nonzero(sub_mask) >= 3:
        i0_sub0 = float(10.0 ** np.median(y_log[sub_mask]))
    else:
        i0_sub0 = float(i_off0 * 100.0)

    n = x.size
    k_top = max(1, int(np.ceil(0.1 * n)))
    order = np.argsort(x)
    top_idx = order[-k_top:]
    ids_top = ids[top_idx]
    i_sat_med = float(np.median(ids_top))
    i_sat_max0 = float(1.2 * i_sat_med)

    v_sat_scale0 = 0.5
    gamma0 = 1.5
    delta0 = 0.05

    p0 = np.array(
        [i_off0, v_th0, vss0, i0_sub0, i_sat_max0, v_sat_scale0, gamma0, delta0],
        dtype=float,
    )
    lo = [1e-15, -3.0, 0.01, 1e-15, 1e-9, 0.05, 0.3, 0.001]
    hi = [1e-9, 3.0, 1.0, 1e-6, 1e-2, 5.0, 4.0, 0.5]
    lo_arr = np.asarray(lo, dtype=float)
    hi_arr = np.asarray(hi, dtype=float)

    print(
        "Initial guess (primary seed): "
        f"I_off={i_off0:.6g} V_th={v_th0:.6g} Vss={vss0:.6g} "
        f"I0_sub={i0_sub0:.6g} I_sat_max={i_sat_max0:.6g} "
        f"V_sat_scale={v_sat_scale0:.6g} gamma={gamma0:.6g} delta={delta0:.6g}"
    )

    seeds: list[np.ndarray] = [p0]
    seeds.append(
        np.array(
            [i_off0, v_th0, 0.05, i_off0, max(i_sat_max0 * 0.8, 1e-9), 0.3, 2.0, 0.03],
            dtype=float,
        )
    )
    seeds.append(
        np.array(
            [i_off0, v_th0, 0.2, i_off0, max(i_sat_max0 * 1.2, 1e-9), 1.0, 2.5, 0.08],
            dtype=float,
        )
    )
    seeds.append(
        np.array([i_off0, 0.0, 0.1, i_off0 * 10.0, max(i_sat_med, 1e-9), 0.8, 2.0, 0.05], dtype=float)
    )
    seeds.append(
        np.array([1e-13, v_th0, 0.1, 1e-14, max(i_sat_med * 1.5, 1e-9), 1.5, 2.5, 0.05], dtype=float)
    )
    ids_max = float(np.max(ids))
    ids_p95 = float(np.percentile(ids, 95))
    seeds.append(
        np.array(
            [
                i_off0,
                v_th0,
                0.12,
                i0_sub0,
                float(np.clip(1.15 * ids_p95, 1e-9, 1e-2)),
                0.6,
                2.2,
                0.06,
            ],
            dtype=float,
        )
    )
    seeds.append(
        np.array(
            [
                i_off0,
                v_th0,
                0.08,
                i_off0 * 50.0,
                float(np.clip(1.2 * ids_max, 1e-9, 1e-2)),
                0.4,
                1.8,
                0.04,
            ],
            dtype=float,
        )
    )

    rng = np.random.default_rng(0)
    for _ in range(32):
        seeds.append(
            np.array(
                [
                    10.0 ** rng.uniform(-15, -9),
                    rng.uniform(-3.0, 3.0),
                    rng.uniform(0.01, 1.0),
                    10.0 ** rng.uniform(-15, -6),
                    10.0 ** rng.uniform(-9, -2),
                    rng.uniform(0.05, 5.0),
                    rng.uniform(0.3, 4.0),
                    10.0 ** rng.uniform(-3.0, np.log10(0.5)),
                ],
                dtype=float,
            )
        )

    best_rmse = float("inf")
    best_popt: NDArray[np.float64] | None = None
    for seed in seeds:
        p0_try = np.clip(seed, lo_arr, hi_arr)
        try:
            popt_try, _ = curve_fit(
                igzo_model,
                x,
                y_log,
                p0=p0_try,
                bounds=(lo, hi),
                maxfev=20000,
            )
        except Exception:
            continue
        popt_try64 = np.asarray(popt_try, dtype=np.float64)
        y_hat_try = igzo_model(x, *popt_try64)
        rmse_try = float(np.sqrt(np.mean((y_log - y_hat_try) ** 2)))
        if rmse_try < best_rmse:
            best_rmse = rmse_try
            best_popt = popt_try64

    if best_popt is None:
        raise RuntimeError("curve_fit failed for all initialization seeds")
    popt64 = best_popt
    y_hat = igzo_model(x, *popt64)
    rmse_log10 = float(np.sqrt(np.mean((y_log - y_hat) ** 2)))
    print(
        "Fitted saturation diagnostics: "
        f"I_sat_max={float(popt64[4]):.6g} A, "
        f"V_sat_scale={float(popt64[5]):.6g} V, "
        f"gamma={float(popt64[6]):.6g}"
    )

    info: dict[str, float | list[str]] = {
        "param_names": [
            "I_off",
            "V_th",
            "Vss",
            "I0_sub",
            "I_sat_max",
            "V_sat_scale",
            "gamma",
            "delta",
        ],
        "vds_fit": float(vds0),
        "rmse_log10": rmse_log10,
        "n_points": float(x.size),
    }
    return popt64, info


def evaluate_model(
    V_gs: NDArray[np.floating[Any]],
    popt: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    return igzo_model(
        V_gs,
        float(popt[0]),
        float(popt[1]),
        float(popt[2]),
        float(popt[3]),
        float(popt[4]),
        float(popt[5]),
        float(popt[6]),
        float(popt[7]),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit IGZO with subthreshold + tanh saturation (composite) model"
    )
    parser.add_argument(
        "--csv",
        default="igzo_measure.csv",
        help="IGZO CSV path (relative paths resolved against script directory)",
    )
    args = parser.parse_args()
    csv_path = args.csv if os.path.isabs(args.csv) else os.path.join(SCRIPT_DIR, args.csv)
    csv_path = os.path.abspath(csv_path)

    popt, info = fit_igzo_powerlaw(csv_path)
    names = list(info["param_names"]) if isinstance(info["param_names"], list) else []
    print(f"csv: {csv_path}")
    print(f"vds_fit: {float(info['vds_fit']):.6g}")
    for i, name in enumerate(names):
        print(f"{name} = {float(popt[i]):.6g}")
    print(f"rmse_log10 = {float(info['rmse_log10']):.6g}")
    print(f"n_points = {int(float(info['n_points']))}")
