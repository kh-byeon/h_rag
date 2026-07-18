"""Load FeFET DAC-sweep measurement Excel files into FefetSweep records."""
from __future__ import annotations

import hashlib
import glob
import os
import re
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

INTEG_CAP_F_FALLBACK = 300e-12  # F; config sheet reports 0.0 incorrectly

_KNOWN_NUMERIC_CONFIG_KEYS = frozenset(
    {
        "READ_V",
        "INTEG_CAP_F",
        "INTEG_DISCHARGE",
        "READ_SETTLE_PRE_S",
        "READ_SETTLE_POST_S",
        "INTEG_ADC_VREF",
        "INTEG_ADC_BITS",
        "INTEG_ADC_GAIN",
        "INTEG_CODE_BASELINE",
        "SWEEP_V_START",
        "SWEEP_V_STOP",
        "SWEEP_V_STEP",
        "SWEEP_BIDIRECTIONAL",
        "SWEEP_REPEAT",
        "SWEEP_POINT_DELAY_S",
        "DAC_VMAX",
        "BAUD",
        "READ_CH",
        "READ_DAC_CH",
        "PULSE_DAC_CH",
        "PROGRAM_CODE",
    }
)


@dataclass(frozen=True)
class FefetSweep:
    code: int  # program code from filename
    v_g: np.ndarray  # (N,) gate voltage, ascending
    i_d: np.ndarray  # (N,) drain current, A, positive magnitude
    v_ds: float  # single value, from config READ_V
    i_floor: float  # estimated instrument floor, A
    rel_noise: float  # median within-point relative std from code_raw
    source_path: str
    source_sha256: str
    measured_at: str  # ISO timestamp of first row


def _strip_config_key(raw: str) -> str:
    """'INTEG_CAP_F (F)' -> 'INTEG_CAP_F'; keep bare names unchanged."""
    s = str(raw).strip()
    m = re.match(r"^([A-Za-z0-9_]+)", s)
    return m.group(1) if m else s


def _parse_config_sheet(df: pd.DataFrame) -> dict[str, str]:
    if df.shape[1] < 2:
        raise ValueError("config sheet must have at least two columns (parameter, value)")
    raw: dict[str, str] = {}
    for _, row in df.iterrows():
        key = _strip_config_key(row.iloc[0])
        if not key or key.lower() == "parameter":
            continue
        val = row.iloc[1]
        raw[key] = "" if pd.isna(val) else str(val).strip()
    return raw


def _coerce_config(raw: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = dict(raw)
    for k in _KNOWN_NUMERIC_CONFIG_KEYS:
        if k not in raw:
            continue
        s = raw[k]
        try:
            if re.fullmatch(r"-?\d+", s):
                out[k] = int(s)
            else:
                out[k] = float(s)
        except ValueError:
            pass
    return out


def _code_from_filename(path: str) -> int:
    base = os.path.basename(path)
    m = re.match(r"^(\d+)_dac_sweep_", base)
    if not m:
        raise ValueError(f"Cannot parse program code from filename: {base!r}")
    return int(m.group(1))


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_code_raw(cell: Any) -> np.ndarray:
    if isinstance(cell, (list, tuple, np.ndarray)):
        return np.asarray(cell, dtype=float).ravel()
    s = str(cell).strip()
    if not s:
        return np.asarray([], dtype=float)
    parts = re.split(r"[,;\s]+", s)
    vals = [float(p) for p in parts if p]
    return np.asarray(vals, dtype=float)


def _resolve_integ_cap_f(config: dict[str, Any], path: str) -> float:
    raw = config.get("INTEG_CAP_F")
    if raw is not None:
        try:
            c = float(raw)
        except (TypeError, ValueError):
            c = 0.0
        if c > 0.0:
            return c
    warnings.warn(
        f"INTEG_CAP_F missing or non-positive in {os.path.basename(path)}; "
        f"using INTEG_CAP_F_FALLBACK={INTEG_CAP_F_FALLBACK:g} F",
        stacklevel=2,
    )
    return float(INTEG_CAP_F_FALLBACK)


def load_sweep(path: str) -> FefetSweep:
    path_abs = os.path.abspath(path)
    if not os.path.isfile(path_abs):
        raise FileNotFoundError(path_abs)

    fname = os.path.basename(path_abs)
    code_fn = _code_from_filename(path_abs)

    cfg_df = pd.read_excel(path_abs, sheet_name="config")
    config_raw = _parse_config_sheet(cfg_df)
    config = _coerce_config(config_raw)

    code = code_fn
    if "PROGRAM_CODE" in config:
        try:
            code_cfg = int(config["PROGRAM_CODE"])
        except (TypeError, ValueError) as e:
            raise ValueError(f"{fname}: invalid PROGRAM_CODE in config") from e
        if code_cfg != code_fn:
            warnings.warn(
                f"{fname}: PROGRAM_CODE={code_cfg} mismatches filename code={code_fn}; "
                "preferring config PROGRAM_CODE",
                stacklevel=2,
            )
        code = code_cfg

    reads = pd.read_excel(path_abs, sheet_name="reads")
    required = {
        "v_dac1",
        "v_dac0",
        "phase",
        "t_integ_ms",
        "V_cap_V",
        "I_A",
        "code_raw",
        "timestamp",
    }
    missing = required - set(reads.columns)
    if missing:
        raise ValueError(f"{fname}: missing columns {sorted(missing)}")

    n = len(reads)
    if n != 71:
        raise ValueError(f"{fname}: expected 71 points, got {n}")

    v_g = reads["v_dac1"].astype(float).to_numpy(dtype=float)
    if not np.all(np.diff(v_g) > 0):
        raise ValueError(f"{fname}: v_dac1 is not monotonically ascending")

    v_dac0 = reads["v_dac0"].astype(float).to_numpy(dtype=float)
    if len(np.unique(np.round(v_dac0, decimals=9))) != 1:
        raise ValueError(f"{fname}: v_dac0 is not a single unique value")

    phases = reads["phase"].astype(str).to_numpy()
    if not np.all(phases == phases[0]):
        raise ValueError(f"{fname}: phase is not uniform across rows")

    if "READ_V" in config:
        v_ds = float(config["READ_V"])
    else:
        v_ds = float(v_dac0[0])

    c_f = _resolve_integ_cap_f(config, path_abs)
    t_integ_s = reads["t_integ_ms"].astype(float).to_numpy(dtype=float) * 1e-3
    v_cap = reads["V_cap_V"].astype(float).to_numpy(dtype=float)
    if np.any(t_integ_s <= 0):
        raise ValueError(f"{fname}: non-positive t_integ_ms")
    i_recomputed = c_f * v_cap / t_integ_s
    i_stored = np.abs(reads["I_A"].astype(float).to_numpy(dtype=float))
    # Match stored I_A: Excel V_cap rounding makes pure rtol=1e-9 fail (~1e-6),
    # so require |ΔI| / max(|I|, 1) <= 1e-9 (absolute floor of 1 A keeps small-I
    # points on an absolute 1e-9 A budget, which is << instrument floor).
    abs_err = np.abs(i_recomputed - i_stored)
    scale = np.maximum(np.abs(i_stored), 1.0)
    rel_err = abs_err / scale
    if float(np.max(rel_err)) > 1e-9:
        bad = int(np.argmax(rel_err))
        raise ValueError(
            f"{fname}: recomputed I_A mismatches stored column "
            f"(max scaled err={float(np.max(rel_err)):.3e} at row {bad}: "
            f"stored={i_stored[bad]:.6e}, recomputed={i_recomputed[bad]:.6e})"
        )
    i_d = np.abs(i_recomputed.astype(np.float64, copy=False))

    floor_mask = v_g < -3.5
    if not np.any(floor_mask):
        raise ValueError(f"{fname}: no points with v_g < -3.5 for i_floor estimate")
    i_floor = float(np.median(i_d[floor_mask]))

    # Skip non-positive means (spec) and near-baseline ADC codes: at the
    # integrator floor, code_raw ~ O(10) and relative std is instrument-dominated,
    # which would push the median above the on-curve ~8% noise floor.
    _CODE_MEAN_MIN = 30.0
    rel_noise_rows: list[float] = []
    for cell in reads["code_raw"]:
        samples = _parse_code_raw(cell)
        if samples.size < 2:
            continue
        mu = float(np.mean(samples))
        if mu < _CODE_MEAN_MIN:
            continue
        rel_noise_rows.append(float(np.std(samples, ddof=1) / mu))
    if not rel_noise_rows:
        raise ValueError(f"{fname}: could not estimate rel_noise from code_raw")
    rel_noise = float(np.median(np.asarray(rel_noise_rows, dtype=float)))

    measured_at = str(reads["timestamp"].iloc[0])
    sha = _file_sha256(path_abs)

    return FefetSweep(
        code=int(code),
        v_g=v_g.astype(np.float64, copy=False),
        i_d=i_d,
        v_ds=float(v_ds),
        i_floor=i_floor,
        rel_noise=rel_noise,
        source_path=path_abs,
        source_sha256=sha,
        measured_at=measured_at,
    )


def load_sweep_dir(
    dirpath: str,
    pattern: str = "*_dac_sweep_*.xlsx",
) -> list[FefetSweep]:
    d = os.path.abspath(dirpath)
    paths = sorted(glob.glob(os.path.join(d, pattern)))
    if not paths:
        raise FileNotFoundError(f"No files matching {pattern!r} under {d}")
    sweeps = [load_sweep(p) for p in paths]
    sweeps.sort(key=lambda s: s.code)
    return sweeps


__all__ = [
    "FefetSweep",
    "INTEG_CAP_F_FALLBACK",
    "load_sweep",
    "load_sweep_dir",
]
