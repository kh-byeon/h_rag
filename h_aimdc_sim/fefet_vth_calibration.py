"""Rigid Vth-shift calibration of FeFET DAC-sweep measurements."""
from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar  # type: ignore[import-untyped]

from h_aimdc_sim.fefet_measure_io import FefetSweep, load_sweep_dir


def _r_squared(y: np.ndarray, y_hat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def _rmse_shift(
    dv: float,
    *,
    v_g: np.ndarray,
    y: np.ndarray,
    v_g_ref: np.ndarray,
    y_ref: np.ndarray,
    fit_window: tuple[float, float],
) -> float:
    """RMSE of rigid horizontal shift on the reference V_G grid.

    Compares y_ref(v) to y_state(v - dv) over fit_window. Positive dv means the
    state curve is left-shifted relative to the reference (turns on earlier).
    """
    vlo, vhi = fit_window
    mask = (v_g_ref >= vlo) & (v_g_ref <= vhi)
    if not np.any(mask):
        return 1e6
    v = v_g_ref[mask]
    y_r = y_ref[mask]
    x = v - float(dv)
    y_pred = np.interp(x, v_g, y)
    inside = (x >= float(v_g[0])) & (x <= float(v_g[-1]))
    if int(np.count_nonzero(inside)) < 3:
        return 1e6
    err = y_pred[inside] - y_r[inside]
    return float(np.sqrt(np.mean(err * err)))


def fit_vth_calibration(
    sweeps: list[FefetSweep],
    *,
    ref_code: int,
    fit_window: tuple[float, float] = (-3.0, 2.0),
) -> dict[str, Any]:
    if not sweeps:
        raise ValueError("fit_vth_calibration: empty sweeps list")
    by_code = {int(s.code): s for s in sweeps}
    if ref_code not in by_code:
        raise ValueError(
            f"ref_code={ref_code} not found in sweeps; have {sorted(by_code)}"
        )
    ref = by_code[ref_code]
    v_g_ref = np.asarray(ref.v_g, dtype=float)
    y_ref = np.log10(np.maximum(np.asarray(ref.i_d, dtype=float), 1e-30))

    states: list[dict[str, Any]] = []
    codes: list[int] = []
    dvs: list[float] = []
    residual_rmses: list[float] = []

    for code in sorted(by_code):
        sw = by_code[code]
        v_g = np.asarray(sw.v_g, dtype=float)
        y = np.log10(np.maximum(np.asarray(sw.i_d, dtype=float), 1e-30))

        if code == ref_code:
            dv = 0.0
            residual = 0.0
        else:
            def objective(dv_cand: float) -> float:
                return _rmse_shift(
                    float(dv_cand),
                    v_g=v_g,
                    y=y,
                    v_g_ref=v_g_ref,
                    y_ref=y_ref,
                    fit_window=fit_window,
                )

            res = minimize_scalar(
                objective, bounds=(-6.0, 6.0), method="bounded"
            )
            dv = float(res.x)
            residual = float(objective(dv))

        # Convert relative ADC noise (~fraction) to decades for noise_ratio.
        # rel_noise ≈ σ_I / I  ⇒  σ_log10(I) ≈ rel_noise / ln(10).
        rel = float(sw.rel_noise)
        noise_dec = rel / math.log(10.0) if rel > 0.0 else float("nan")
        noise_ratio = (
            float(residual / noise_dec)
            if noise_dec > 0.0 and np.isfinite(noise_dec)
            else float("nan")
        )
        if np.isfinite(noise_ratio) and noise_ratio > 4.0:
            warnings.warn(
                f"code={code}: noise_ratio={noise_ratio:.3g} > 4.0 "
                "(rigid Vth shift may be breaking down)",
                stacklevel=2,
            )

        states.append(
            {
                "code": int(code),
                "dv": float(dv),
                "residual_rmse_dec": float(residual),
                "noise_ratio": float(noise_ratio) if np.isfinite(noise_ratio) else None,
                "source_sha256": sw.source_sha256,
                "measured_at": sw.measured_at,
            }
        )
        codes.append(int(code))
        dvs.append(float(dv))
        residual_rmses.append(float(residual))

    codes_a = np.asarray(codes, dtype=float)
    dvs_a = np.asarray(dvs, dtype=float)

    # Linear: dv = slope * code + intercept
    slope, intercept = np.polyfit(codes_a, dvs_a, 1)
    y_lin = slope * codes_a + intercept
    r2_lin = _r_squared(dvs_a, y_lin)

    # Log: dv = a + b * ln(code)
    ln_c = np.log(codes_a)
    b_log, a_log = np.polyfit(ln_c, dvs_a, 1)  # polyfit returns [b, a] for a + b x
    # np.polyfit(ln_c, dvs, 1) -> [b, a] where y = b*ln_c + a
    y_log = a_log + b_log * ln_c
    r2_log = _r_squared(dvs_a, y_log)

    preferred = "log" if (np.isfinite(r2_log) and r2_log >= r2_lin) else "linear"
    vth_span = float(np.max(dvs_a) - np.min(dvs_a))

    i_floors = [float(s.i_floor) for s in sweeps]
    rel_noises = [float(s.rel_noise) for s in sweeps]
    v_g_mins = [float(np.min(s.v_g)) for s in sweeps]
    v_g_maxs = [float(np.max(s.v_g)) for s in sweeps]
    v_ds_vals = [float(s.v_ds) for s in sweeps]

    expected = set(range(300, 1300, 100)) | {1500, 1600, 1700}
    present = set(codes)
    missing = sorted(expected - present)

    return {
        "schema_version": 1,
        "ref_code": int(ref_code),
        "fit_window_v": [float(fit_window[0]), float(fit_window[1])],
        "states": states,
        "model_linear": {
            "slope_v_per_code": float(slope),
            "intercept_v": float(intercept),
            "r_squared": float(r2_lin),
        },
        "model_log": {
            "a": float(a_log),
            "b": float(b_log),
            "r_squared": float(r2_log),
        },
        "preferred_model": preferred,
        "vth_span_v": vth_span,
        "measurement": {
            "v_g_min": float(min(v_g_mins)),
            "v_g_max": float(max(v_g_maxs)),
            "v_ds": float(np.median(v_ds_vals)),
            "i_floor_a": float(np.median(i_floors)),
            "rel_noise": float(np.median(rel_noises)),
            "n_states": len(states),
            "missing_codes": missing,
        },
    }


def save_vth_calibration(result: dict[str, Any], out_path: str) -> None:
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="fefet_vth_calibration",
        description="Fit rigid Vth shifts of FeFET DAC-sweep measurements.",
    )
    parser.add_argument(
        "--dir",
        required=True,
        help="Directory of <code>_dac_sweep_*.xlsx files",
    )
    parser.add_argument("--ref-code", type=int, default=300)
    parser.add_argument(
        "--out",
        default="fefet_vth_calibration.json",
        help="Output JSON path",
    )
    parser.add_argument("--fit-vlo", type=float, default=-3.0)
    parser.add_argument("--fit-vhi", type=float, default=2.0)
    args = parser.parse_args(argv)

    sweeps = load_sweep_dir(args.dir)
    result = fit_vth_calibration(
        sweeps,
        ref_code=int(args.ref_code),
        fit_window=(float(args.fit_vlo), float(args.fit_vhi)),
    )
    out_abs = os.path.abspath(args.out)
    save_vth_calibration(result, out_abs)

    by_code = {int(s["code"]): s for s in result["states"]}
    dv_1700 = by_code.get(1700, {}).get("dv")
    print(f"Wrote {out_abs}")
    print(f"  n_states = {result['measurement']['n_states']}")
    print(f"  dv(1700) = {dv_1700}")
    print(
        f"  model_linear: slope={result['model_linear']['slope_v_per_code']:.6g} "
        f"intercept={result['model_linear']['intercept_v']:.6g} "
        f"R^2={result['model_linear']['r_squared']:.4f}"
    )
    print(
        f"  model_log: a={result['model_log']['a']:.6g} "
        f"b={result['model_log']['b']:.6g} "
        f"R^2={result['model_log']['r_squared']:.4f}"
    )
    print(f"  preferred_model = {result['preferred_model']}")
    print(f"  vth_span_v = {result['vth_span_v']:.4f}")


if __name__ == "__main__":
    main()
