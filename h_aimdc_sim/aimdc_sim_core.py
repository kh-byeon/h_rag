"""
Importable AIMDC single-bit Xyce driver (no CLI side effects).
Docker binary/image are configured via module-level attributes.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import time
from typing import Any, Mapping, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numpy.typing import NDArray
from scipy.optimize import curve_fit  # type: ignore[import-untyped]

# ---- Module-level Docker / Xyce configuration (set before run_docker_sim) ----
AIMDC_DOCKER_BIN: str | None = None
AIMDC_DOCKER_IMAGE: str = "h-aimdc-sim:xyce78"

DEFAULT_CIRCUIT_PARAMS: dict[str, float] = {
    "C_BOOST": 100e-15,
    "V_MBL": 0.1,
    "V_ML": 0.0,
    "V_RWL": 3.0,
    "V_WL": -3.0,
    "V_BL": 0.0,
    "V_BL_BAR": 0.0,
    "T_RAMP": 9.9e-6,
    "T_APPLY": 10e-6,
    "T_END": 20e-6,
    "T_SAMPLE": 15e-6,
    "v_min": -1.0,
    "v_max": 1.0,
}


def _workdir(wd: str | None) -> str:
    return os.path.abspath(wd or os.getcwd())


def resolve_docker_bin() -> str:
    if AIMDC_DOCKER_BIN:
        return AIMDC_DOCKER_BIN
    p = shutil.which("docker")
    if p is None:
        raise RuntimeError(
            "docker binary not found in PATH. Is Docker Desktop WSL integration enabled?"
        )
    return p


def composite_model(
    v: NDArray[np.floating[Any]],
    c0: float,
    k: float,
    vth: float,
    p0: float,
    p1: float,
    p2: float,
    p3: float,
) -> NDArray[np.floating[Any]]:
    s = 0.5 * (1.0 + np.tanh(k * (v - vth)))
    on_poly = p0 + p1 * v + p2 * v**2 + p3 * v**3
    return cast(NDArray[np.floating[Any]], c0 * (1 - s) + on_poly * s)


def xyce_b_source_expression(popt: NDArray[np.floating[Any]]) -> str:
    """Public wrapper: Xyce B-source RHS from fitted θ (local nodes g,s)."""
    return _build_xyce_b_source_expression(popt)


def _build_xyce_b_source_expression(popt: NDArray[np.floating[Any]]) -> str:
    c0, k, vth, p0, p1, p2, p3 = popt
    s_expr = f"0.5 * (1.0 + TANH({k:e} * (V(g,s) - ({vth:e}))))"
    on_poly_expr = (
        f"({p0:e} + {p1:e}*V(g,s) + {p2:e}*V(g,s)*V(g,s) + {p3:e}*V(g,s)*V(g,s)*V(g,s))"
    )
    poly_expr = f"({c0:e}) * (1.0 - ({s_expr})) + ({on_poly_expr}) * ({s_expr})"
    return f"EXP(2.302585093 * ({poly_expr}))"


def fit_igzo_model(
    csv_path: str,
    *,
    vds: float | None = None,
) -> tuple[NDArray[np.float64], str]:
    """
    Read measurement CSV, select Vds (explicit or deterministic), fit composite_model
    to log10(|Ids|+1e-15) vs Vgs.
    Returns (popt, xyce_expr) where popt = [c0, k, vth, p0, p1, p2, p3].
    """
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
            raise ValueError(f"Requested vds={vds0} not in CSV unique Vds={seen}")
    else:
        vds0 = float(min(seen, key=lambda x: (abs(abs(x) - 0.5), abs(x), x)))
    meas = df[np.isclose(df["vds"].astype(float), vds0)].sort_values("vgs")
    x = meas["vgs"].values.astype(float)
    y_log = np.log10(np.abs(meas["ids"].values.astype(float)) + 1e-15)

    initial_guess = [np.min(y_log), 5.0, 0.0, np.max(y_log), 1.0, 0.0, 0.0]
    bounds = (
        [-20.0, 0.1, -5.0, -20.0, -10.0, -10.0, -10.0],
        [-5.0, 50.0, 5.0, 5.0, 10.0, 10.0, 10.0],
    )
    popt, _ = curve_fit(
        composite_model, x, y_log, p0=initial_guess, bounds=bounds, maxfev=10000
    )
    popt64 = np.asarray(popt, dtype=np.float64)
    return popt64, _build_xyce_b_source_expression(popt64)


def compute_fit_residual_rmse(
    csv_path: str, popt: NDArray[np.floating[Any]]
) -> tuple[float, float]:
    """RMSE of composite_model in log10 domain vs measurement; returns (rmse, vds_fit)."""
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
    vds0 = float(seen[0])
    meas = df[np.isclose(df["vds"].astype(float), vds0)].sort_values("vgs")
    x = meas["vgs"].values.astype(float)
    y_log = np.log10(np.abs(meas["ids"].values.astype(float)) + 1e-15)
    pred = composite_model(x, *popt)
    rmse = float(np.sqrt(np.mean((pred - y_log) ** 2)))
    return rmse, vds0


def save_fit_json(
    popt: NDArray[np.floating[Any]],
    out_path: str,
    *,
    source_csv: str,
    vds_fit: float,
    fit_rmse_log10: float,
    circuit_params: Mapping[str, Any] | None = None,
) -> None:
    cp = dict(DEFAULT_CIRCUIT_PARAMS)
    if circuit_params:
        cp.update({k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in dict(circuit_params).items()})
    payload = {
        "schema_version": 1,
        "popt": [float(x) for x in np.asarray(popt, dtype=np.float64).ravel()],
        "source_csv": str(source_csv),
        "vds_fit": float(vds_fit),
        "fit_rmse_log10": float(fit_rmse_log10),
        "circuit_params": {k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v for k, v in cp.items()},
    }
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_fit_json(path: str) -> tuple[NDArray[np.float64], dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    model = d.get("model")
    # Dispatch on explicit model string when present (schema_version >= 2 / FeFET).
    if isinstance(model, str) and model.startswith("fefet_ref_"):
        popt = np.asarray(d.get("popt", []), dtype=np.float64)
        meta = {k: v for k, v in d.items() if k != "popt"}
        return popt, meta
    if isinstance(model, str) and model in ("igzo_composite_v1", "igzo_powerlaw_v3"):
        popt = np.asarray(d["popt"], dtype=np.float64)
        meta = {k: v for k, v in d.items() if k != "popt"}
        return popt, meta
    # Legacy: infer from popt length (7 = v1 composite, 8 = v3 powerlaw).
    popt = np.asarray(d["popt"], dtype=np.float64)
    meta = {k: v for k, v in d.items() if k != "popt"}
    if "model" not in meta:
        if popt.size == 7:
            meta["model"] = "igzo_composite_v1"
        elif popt.size == 8:
            meta["model"] = "igzo_powerlaw_v3"
    return popt, meta


def generate_spice_deck(
    expression: str,
    v_sl_val: float,
    params: Mapping[str, Any],
    *,
    workdir: str | None = None,
) -> None:
    """Write sim_distance.sp under workdir (default: cwd)."""
    wd = _workdir(workdir)
    os.makedirs(wd, exist_ok=True)
    p = dict(params)
    spice_path = os.path.join(wd, "sim_distance.sp")
    spice = f"""* AIMDC Capacitive Boosting Distance Macro
.OPTIONS NONLIN CONTINUE
.TEMP 25

.SUBCKT igzo_tft d g s
BIDS d s I={{{expression}}}
.ENDS igzo_tft

* ---- Left cell ----
XW1   BL    WL    SN     igzo_tft
Ccpl1 SL_bar SN     {p["C_BOOST"]}
XR1   MBL   RWL   N_mid1 igzo_tft
XS1   N_mid1 SN   ML     igzo_tft

* ---- Right cell (differential) ----
XW2   BL_bar WL   SN_bar igzo_tft
Ccpl2 SL     SN_bar {p["C_BOOST"]}
XR2   MBL   RWL   N_mid2 igzo_tft
XS2   N_mid2 SN_bar ML   igzo_tft

* ---- Numerical DC path (ideal float; 1e15 Ohm ~ open) ----
R_leak1 SN     0 1e15
R_leak2 SN_bar 0 1e15

* ---- Biases ----
V_MBL  MBL   0 DC {p["V_MBL"]}
V_ML   ML    0 DC {p["V_ML"]}
V_RWL  RWL   0 DC {p["V_RWL"]}
V_WL   WL    0 DC {p["V_WL"]}
V_BL   BL    0 DC {p["V_BL"]}
V_BLb  BL_bar 0 DC {p["V_BL_BAR"]}

* ---- Stored weights (floating SN / SN_bar) ----
.IC V(SN)={p["V_PRG"]}  V(SN_bar)={-p["V_PRG"]}

* ---- Differential input PWL ----
V_SL    SL     0 PWL(0 0  {p["T_RAMP"]} 0  {p["T_APPLY"]} {v_sl_val}   {p["T_END"]} {v_sl_val})
V_SLb   SL_bar 0 PWL(0 0  {p["T_RAMP"]} 0  {p["T_APPLY"]} {-v_sl_val}  {p["T_END"]} {-v_sl_val})

.TRAN 0.1u {p["T_END"]}
.PRINT TRAN FORMAT=CSV file=sim_distance.csv V(SL) V(SN) V(SN_bar) I(V_MBL)
.END
"""
    with open(spice_path, "w", encoding="ascii", newline="\n") as f:
        f.write(spice)


def _cleanup_stale_sim_outputs(workdir: str | None = None) -> None:
    wd = _workdir(workdir)
    for name in ("sim_distance.csv",):
        path = os.path.join(wd, name)
        if os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                pass
    pattern = os.path.join(wd, "sim_distance.csv.*")
    for path in glob.glob(pattern):
        try:
            os.remove(path)
        except OSError:
            pass


def _tail(s: str | bytes | None, n: int = 2000) -> str:
    if s is None:
        return ""
    if isinstance(s, bytes):
        s = s.decode(errors="replace")
    s = str(s).strip()
    return s[-n:] if len(s) > n else s


def run_docker_sim(*, workdir: str | None = None) -> None:
    """Run Xyce in Docker; working directory must contain sim_distance.sp."""
    docker_bin = resolve_docker_bin()
    wd = _workdir(workdir)
    _cleanup_stale_sim_outputs(wd)

    inner = "cd /work && Xyce sim_distance.sp"
    cmd = [
        docker_bin,
        "run",
        "--rm",
        "-e",
        "OMPI_ALLOW_RUN_AS_ROOT=1",
        "-e",
        "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1",
        "-v",
        f"{wd}:/work",
        AIMDC_DOCKER_IMAGE,
        "bash",
        "-lc",
        inner,
    ]
    cmd_str = subprocess.list2cmdline(cmd)

    try:
        r = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired as te:
        out = _tail(te.stdout)
        err = _tail(te.stderr)
        if out:
            print("--- Xyce stdout (tail) ---\n", out, sep="")
        if err:
            print("--- Xyce stderr (tail) ---\n", err, sep="")
        print("Docker command was:\n ", cmd_str)
        raise RuntimeError("docker/Xyce timed out after 120s") from te

    if r.returncode != 0:
        out = _tail(r.stdout)
        err = _tail(r.stderr)
        if out:
            print("--- Xyce stdout (tail) ---\n", out, sep="")
        if err:
            print("--- Xyce stderr (tail) ---\n", err, sep="")
        print("Docker command was:\n ", cmd_str)
        raise RuntimeError(f"docker/Xyce failed with exit code {r.returncode}")

    if r.stdout:
        lines = [ln for ln in r.stdout.strip().splitlines() if ln.strip()]
        if lines:
            last = lines[-1]
            if "Total Simulation Time" in last:
                print(last)


def extract_i_ml(csv_path: str, t_sample: float) -> float:
    """Sample |I(V_MBL)| at time closest to t_sample. Returns NaN on failure."""
    try:
        sim_df = pd.read_csv(csv_path)
        sim_df.columns = [str(c).strip().upper() for c in sim_df.columns]
        time_col = "TIME" if "TIME" in sim_df.columns else None
        if time_col is None:
            for c in sim_df.columns:
                if c.replace(" ", "") == "TIME":
                    time_col = c
                    break
        if time_col is None:
            return float("nan")

        i_col = None
        for c in sim_df.columns:
            cu = c.replace(" ", "").upper()
            if cu == "I(V_MBL)" or ("I(" in cu and "V_MBL" in cu):
                i_col = c
                break
        if i_col is None:
            return float("nan")

        t_idx = int(np.abs(sim_df[time_col].astype(float) - t_sample).argmin())
        i_mbl = float(sim_df[i_col].iloc[t_idx])
        return abs(i_mbl)
    except Exception:
        return float("nan")


def run_docker_sim_with_retries(
    *,
    workdir: str | None = None,
    k: int | None = None,
    n: int | None = None,
    v_sl: float | None = None,
) -> None:
    """Up to 3 attempts with 1s delay; raises RuntimeError if all fail."""
    last_err: RuntimeError | None = None
    for attempt in range(3):
        if attempt > 0:
            if k is not None and n is not None and v_sl is not None:
                print(f"[retry {attempt}/3] V_SL step [{k}/{n}] V_SL = {v_sl:.4f} V")
            else:
                print(f"[retry {attempt}/3] Docker/Xyce")
            time.sleep(1.0)
        try:
            run_docker_sim(workdir=workdir)
            return
        except RuntimeError as e:
            last_err = e
    assert last_err is not None
    raise last_err


def assert_docker_available() -> str:
    """Verify daemon and image; return resolved docker binary path."""
    dbin = resolve_docker_bin()
    try:
        subprocess.run(
            [dbin, "ps"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "Docker daemon not reachable — is Docker Desktop running with WSL2 integration?"
        ) from e
    try:
        subprocess.run(
            [dbin, "image", "inspect", AIMDC_DOCKER_IMAGE],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Docker image {AIMDC_DOCKER_IMAGE!r} not found. Build/pull the image before LUT generation."
        ) from e
    return dbin


def circuit_params_json(params: Mapping[str, Any]) -> str:
    """Serialize circuit params for NPZ sidecar (JSON string)."""
    out: dict[str, float] = {}
    for k, v in params.items():
        if isinstance(v, (float, int)):
            out[str(k)] = float(v)
        elif isinstance(v, (np.floating, np.integer)):
            out[str(k)] = float(v)
    return json.dumps(out, sort_keys=True)


def merge_circuit_params(
    base: Mapping[str, Any] | None,
    v_prg: float,
) -> dict[str, Any]:
    p = dict(DEFAULT_CIRCUIT_PARAMS)
    if base is not None:
        p.update(dict(base))
    p["V_PRG"] = float(v_prg)
    return p


def _cli_main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="aimdc_sim_core")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_fit = sub.add_parser("fit", help="Fit IGZO CSV and write JSON with popt + circuit_params.")
    p_fit.add_argument("--csv", required=True, help="Path to igzo_measure.csv")
    p_fit.add_argument("--out", required=True, help="Output JSON path (e.g. igzo_fit.json)")
    args = parser.parse_args(argv)
    if args.cmd == "fit":
        csv_abs = os.path.abspath(args.csv)
        popt, _expr = fit_igzo_model(csv_abs)
        rmse, vds0 = compute_fit_residual_rmse(csv_abs, popt)
        save_fit_json(
            popt,
            os.path.abspath(args.out),
            source_csv=os.path.basename(csv_abs),
            vds_fit=vds0,
            fit_rmse_log10=rmse,
        )
        print(f"Wrote {args.out!r} fit_rmse_log10={rmse:.6f} vds_fit={vds0}")


if __name__ == "__main__":
    import sys

    _cli_main(sys.argv[1:])
