"""Reference-curve FeFET device model (LUT or 3-param power-law).

Replaces igzo_powerlaw_model.py for the measured 2-FeFET path.
Legacy IGZO imports remain untouched.
"""
from __future__ import annotations

import hashlib
import json
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit  # type: ignore[import-untyped]
from scipy.signal import savgol_filter  # type: ignore[import-untyped]

from h_aimdc_sim.fefet_measure_io import FefetSweep

POWERLAW_EPS = 1e-15  # shared Python / Xyce clamp for power-law base


@dataclass(frozen=True)
class FefetRefModel:
    backend: Literal["lut", "powerlaw3"]
    v_grid: Optional[np.ndarray]  # lut backend
    log_i_grid: Optional[np.ndarray]  # lut backend
    params: Optional[dict[str, float]]  # powerlaw3: A, Vth, n
    i_floor: float
    v_valid: tuple[float, float]  # measured gate-voltage window
    meta: dict = field(default_factory=dict)

    def current(
        self,
        v_g: np.ndarray | float,
        *,
        dv: float = 0.0,
        allow_extrapolation: bool = False,
        v_ds: float | None = None,
        v_ds_factor: bool = False,
    ) -> np.ndarray:
        """I(V_G - dv). Raises if any sample falls outside v_valid by >0.1 V."""
        v = np.asarray(v_g, dtype=np.float64) - float(dv)
        vlo, vhi = self.v_valid
        if not allow_extrapolation:
            below = v < (vlo - 0.1)
            above = v > (vhi + 0.1)
            if np.any(below) or np.any(above):
                bad = v[below | above]
                raise ValueError(
                    f"FefetRefModel.current: {bad.size} sample(s) outside "
                    f"v_valid=[{vlo}, {vhi}] by >0.1 V "
                    f"(min={float(np.min(bad)):.4g}, max={float(np.max(bad)):.4g}). "
                    "Pass allow_extrapolation=True to override."
                )
        if self.backend == "lut":
            assert self.v_grid is not None and self.log_i_grid is not None
            log_i = np.interp(v, self.v_grid, self.log_i_grid)
            i = np.power(10.0, log_i)
        else:
            assert self.params is not None
            i = _powerlaw3_current(v, self.params, self.i_floor)

        if v_ds_factor:
            # Unvalidated heuristic (single measured V_DS point); see module meta.
            vds_ref = float(self.meta.get("v_ds_ref_heuristic", abs(self.meta.get("v_ds_measured", 0.5))))
            k_step = float(self.meta.get("k_step", 50.0))
            if v_ds is None:
                v_ds = float(self.meta.get("v_ds_measured", -0.5))
            vds = np.asarray(v_ds, dtype=np.float64)
            step_pos = 0.5 * (1.0 + np.tanh(k_step * vds))
            sat = vds / np.sqrt(vds * vds + vds_ref * vds_ref)
            i = i * step_pos * sat

        return np.asarray(i, dtype=np.float64)

    def xyce_expression(self, *, v_ds_factor: bool = False) -> str:
        if self.backend == "lut":
            assert self.v_grid is not None and self.log_i_grid is not None
            # TABLE of (V, I) pairs; Xyce interpolates linearly between points.
            pairs: list[str] = []
            for vv, li in zip(self.v_grid, self.log_i_grid):
                ii = float(10.0 ** float(li))
                pairs.append(f"{float(vv):.8g},{ii:.8e}")
            table = ",".join(pairs)
            ids_vgs = f"TABLE(V(g,s),{table})"
        else:
            assert self.params is not None
            A = float(self.params["A"])
            Vth = float(self.params["Vth"])
            n = float(self.params["n"])
            i_floor = float(self.i_floor)
            eps = POWERLAW_EPS
            # I = A * max(V_G - Vth, 0)^n + I_floor
            # Use EXP(n*LOG(base+eps)) — never ** with fractional n.
            od = f"MAX(V(g,s) - ({Vth:e}), 0.0)"
            base = f"(({od}) + ({eps:e}))"
            ids_vgs = f"(({A:e}) * EXP(({n:e}) * LOG({base})) + ({i_floor:e}))"

        if not v_ds_factor:
            return ids_vgs

        vds_ref = float(self.meta.get("v_ds_ref_heuristic", abs(self.meta.get("v_ds_measured", 0.5))))
        k_step = float(self.meta.get("k_step", 50.0))
        # UNVALIDATED HEURISTIC: V_DS factor never fitted (single measured V_DS).
        step_pos = f"0.5 * (1.0 + TANH({k_step:e} * V(d,s)))"
        sat = f"V(d,s) / SQRT(V(d,s)*V(d,s) + ({vds_ref:e})*({vds_ref:e}))"
        return f"({ids_vgs}) * (({step_pos}) * ({sat}))"


def _powerlaw3_current(
    v_g: np.ndarray,
    params: Mapping[str, float],
    i_floor: float,
) -> np.ndarray:
    A = float(params["A"])
    Vth = float(params["Vth"])
    n = float(params["n"])
    od = np.maximum(np.asarray(v_g, dtype=np.float64) - Vth, 0.0)
    return A * np.power(od + POWERLAW_EPS, n) + float(i_floor)


def _build_lut(
    sweep: FefetSweep,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    v = np.asarray(sweep.v_g, dtype=np.float64)
    i = np.maximum(np.asarray(sweep.i_d, dtype=np.float64), 1e-30)
    i_floor = float(sweep.i_floor)
    log_i = np.log10(i)

    # Savitzky–Golay smooth in log domain.
    wl = 7
    if v.size < wl:
        wl = v.size if v.size % 2 == 1 else max(v.size - 1, 3)
    if wl >= 3 and wl % 2 == 1:
        log_s = savgol_filter(log_i, window_length=wl, polyorder=min(2, wl - 1))
    else:
        log_s = log_i.copy()

    # v_knee: lowest V_G where I_ref exceeds 3 * i_floor.
    above = np.where(i > 3.0 * i_floor)[0]
    if above.size == 0:
        v_knee = float(v[-1])
    else:
        v_knee = float(v[int(above[0])])

    # Below knee: clamp at instrument floor (max with floor, not flatten).
    # Above knee: enforce monotonicity via accumulate.
    log_out = log_s.copy()
    floor_log = float(np.log10(max(i_floor, 1e-30)))
    log_out = np.maximum(log_out, floor_log)
    above_mask = v >= v_knee
    if np.any(above_mask):
        seg = log_out[above_mask].copy()
        seg = np.maximum.accumulate(seg)
        log_out[above_mask] = seg

    return v, log_out.astype(np.float64, copy=False), i_floor, v_knee


def _fit_powerlaw3(
    sweep: FefetSweep,
    *,
    v_knee: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    v = np.asarray(sweep.v_g, dtype=np.float64)
    i = np.asarray(sweep.i_d, dtype=np.float64)
    i_floor = float(sweep.i_floor)
    mask = v >= float(v_knee)
    if int(np.count_nonzero(mask)) < 4:
        raise ValueError(
            f"powerlaw3: fewer than 4 points with V_G >= v_knee={v_knee}"
        )
    vv = v[mask]
    ii = np.maximum(i[mask], 1e-30)
    y_log = np.log10(ii)

    def log_model(vg: np.ndarray, A: float, Vth: float, n: float) -> np.ndarray:
        od = np.maximum(vg - Vth, 0.0)
        return np.log10(A * np.power(od + POWERLAW_EPS, n) + i_floor)

    vth_lo = float(np.min(vv) - 2.0)
    vth_hi = float(np.min(vv))  # Vth should sit at or below the fit window start
    bounds_lo = np.array([1e-12, vth_lo, 0.5], dtype=float)
    bounds_hi = np.array([1e-4, vth_hi, 2.5], dtype=float)
    p0 = np.array([5e-8, float(v_knee) - 0.5, 1.2], dtype=float)
    p0 = np.clip(p0, bounds_lo + 1e-12, bounds_hi - 1e-12)

    popt, pcov = curve_fit(
        log_model,
        vv,
        y_log,
        p0=p0,
        bounds=(bounds_lo, bounds_hi),
        maxfev=50000,
    )
    names = ("A", "Vth", "n")
    params = {names[k]: float(popt[k]) for k in range(3)}

    # Relative std-error diagnostics.
    diag = np.sqrt(np.maximum(np.diag(np.asarray(pcov, dtype=float)), 0.0))
    for k, name in enumerate(names):
        p = abs(float(popt[k]))
        se = float(diag[k])
        if p > 0.0 and (se / p) > 0.5:
            warnings.warn(
                f"powerlaw3: parameter {name} unconstrained by data "
                f"(rel. std err={se / p:.2%})",
                stacklevel=2,
            )
    # Railed-parameter check. Scale param A uses log-space fraction.
    for k, name in enumerate(names):
        lo_b = float(bounds_lo[k])
        hi_b = float(bounds_hi[k])
        p = float(popt[k])
        if name == "A" and lo_b > 0.0 and hi_b > 0.0 and p > 0.0:
            span_log = np.log(hi_b) - np.log(lo_b)
            dist = min(abs(np.log(p) - np.log(lo_b)), abs(np.log(hi_b) - np.log(p)))
            frac = float(dist / span_log) if span_log > 0 else 0.0
        else:
            span = hi_b - lo_b
            frac = (
                min(abs(p - lo_b), abs(hi_b - p)) / span if span > 0.0 else 0.0
            )
        if frac < 1e-3:
            raise ValueError(
                f"powerlaw3: parameter {name}={p:.6g} is within "
                f"1e-3 of its bound [{lo_b:.6g}, {hi_b:.6g}]"
            )

    info = {
        "pcov_diag": [float(x) for x in diag],
        "v_knee": float(v_knee),
        "n_fit_points": int(np.count_nonzero(mask)),
        "fit_rmse_log10": float(
            np.sqrt(np.mean((log_model(vv, *popt) - y_log) ** 2))
        ),
    }
    return params, info


def fit_fefet_ref_model(
    ref_sweep: FefetSweep,
    *,
    backend: Literal["lut", "powerlaw3"] = "lut",
    calibration: Mapping[str, Any] | None = None,
    source_files: list[str] | None = None,
) -> FefetRefModel:
    v_grid, log_i_grid, i_floor, v_knee = _build_lut(ref_sweep)
    v_valid = (float(np.min(ref_sweep.v_g)), float(np.max(ref_sweep.v_g)))

    meta: dict[str, Any] = {
        "schema_version": 2,
        "model": "fefet_ref_lut_v1" if backend == "lut" else "fefet_ref_powerlaw3_v1",
        "backend": backend,
        "source_sha256": ref_sweep.source_sha256,
        "source_files": source_files
        or [os.path.basename(ref_sweep.source_path)],
        "v_valid": [v_valid[0], v_valid[1]],
        "v_ds_measured": float(ref_sweep.v_ds),
        "i_floor_a": float(i_floor),
        "rel_noise": float(ref_sweep.rel_noise),
        "v_knee": float(v_knee),
        "ref_code": int(ref_sweep.code),
        # V_DS factor is an unvalidated heuristic (single measured V_DS point).
        "v_ds_factor_note": (
            "UNVALIDATED HEURISTIC: step_pos * vds/sqrt(vds^2+vds_ref^2) was never "
            "fitted; only one V_DS point was measured."
        ),
        "v_ds_ref_heuristic": abs(float(ref_sweep.v_ds)),
        "sigma_vth_note": "assumed/unmeasured (SWEEP_REPEAT=1; no cycle-to-cycle data)",
        "power_law_p": None,
        "p_note": (
            "linear/triode region; |V_DS|=0.5V; saturation not reachable due to "
            "read disturb limit on gate sweep"
        ),
        "reduction_mode": "linear_sum",
    }
    if calibration is not None:
        meta["calibration_ref"] = calibration.get(
            "path", "fefet_vth_calibration.json"
        )
        meta["calibration_sha256"] = calibration.get("sha256")

    if backend == "lut":
        # Estimate p from a quick powerlaw3 for metadata only (non-fatal).
        try:
            pl_params, _ = _fit_powerlaw3(
                ref_sweep, v_knee=max(float(v_knee), -1.0)
            )
            meta["power_law_p"] = float(pl_params["n"])
        except Exception as exc:  # noqa: BLE001
            meta["power_law_p"] = 1.15
            meta["power_law_p_fit_error"] = str(exc)
        return FefetRefModel(
            backend="lut",
            v_grid=v_grid,
            log_i_grid=log_i_grid,
            params=None,
            i_floor=i_floor,
            v_valid=v_valid,
            meta=meta,
        )

    # Power-law is only meaningful above the triode turn-on; facts use V_G >= -1 V.
    pl_lo = max(float(v_knee), -1.0)
    params, info = _fit_powerlaw3(ref_sweep, v_knee=pl_lo)
    meta["power_law_p"] = float(params["n"])
    meta["powerlaw3_fit"] = info
    meta["model"] = "fefet_ref_powerlaw3_v1"
    return FefetRefModel(
        backend="powerlaw3",
        v_grid=None,
        log_i_grid=None,
        params=params,
        i_floor=i_floor,
        v_valid=v_valid,
        meta=meta,
    )


def save_fefet_ref_fit(model: FefetRefModel, out_path: str) -> None:
    payload: dict[str, Any] = dict(model.meta)
    payload["i_floor"] = float(model.i_floor)
    payload["v_valid"] = [float(model.v_valid[0]), float(model.v_valid[1])]
    payload["backend"] = model.backend
    if model.backend == "lut":
        assert model.v_grid is not None and model.log_i_grid is not None
        payload["v_grid"] = [float(x) for x in model.v_grid]
        payload["log_i_grid"] = [float(x) for x in model.log_i_grid]
        # popt placeholder for loaders that expect an array; empty for LUT.
        payload["popt"] = []
    else:
        assert model.params is not None
        payload["params"] = {k: float(v) for k, v in model.params.items()}
        payload["popt"] = [
            float(model.params["A"]),
            float(model.params["Vth"]),
            float(model.params["n"]),
        ]
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_fefet_ref_fit(path: str) -> FefetRefModel:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    backend = str(d.get("backend", "lut"))
    v_valid_raw = d.get("v_valid", [-5.0, 2.0])
    v_valid = (float(v_valid_raw[0]), float(v_valid_raw[1]))
    i_floor = float(d.get("i_floor", d.get("i_floor_a", 1e-8)))
    meta = {k: v for k, v in d.items() if k not in ("v_grid", "log_i_grid", "params", "popt")}
    if backend == "lut":
        v_grid = np.asarray(d["v_grid"], dtype=np.float64)
        log_i_grid = np.asarray(d["log_i_grid"], dtype=np.float64)
        return FefetRefModel(
            backend="lut",
            v_grid=v_grid,
            log_i_grid=log_i_grid,
            params=None,
            i_floor=i_floor,
            v_valid=v_valid,
            meta=meta,
        )
    params = d.get("params")
    if not isinstance(params, dict):
        popt = d.get("popt") or []
        if len(popt) != 3:
            raise ValueError(f"powerlaw3 fit JSON missing params: {path}")
        params = {"A": float(popt[0]), "Vth": float(popt[1]), "n": float(popt[2])}
    return FefetRefModel(
        backend="powerlaw3",
        v_grid=None,
        log_i_grid=None,
        params={k: float(v) for k, v in params.items()},
        i_floor=i_floor,
        v_valid=v_valid,
        meta=meta,
    )


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


__all__ = [
    "FefetRefModel",
    "POWERLAW_EPS",
    "fit_fefet_ref_model",
    "load_fefet_ref_fit",
    "save_fefet_ref_fit",
    "file_sha256",
]
