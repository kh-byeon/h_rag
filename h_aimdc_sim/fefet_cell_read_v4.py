"""
FeFET 2-cell read-mode DC sweep (measured reference-curve B-source).
Inverter removed: differential SL_F / SL_F_bar with programmed Vth offsets.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import shutil
import subprocess
import sys
from typing import Any, TypedDict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

try:
    import aimdc_sim_core
    from aimdc_sim_core import assert_docker_available
except ImportError:
    from h_aimdc_sim import aimdc_sim_core
    from h_aimdc_sim.aimdc_sim_core import assert_docker_available

from h_aimdc_sim.fefet_measure_io import load_sweep_dir
from h_aimdc_sim.fefet_ref_model import (
    FefetRefModel,
    fit_fefet_ref_model,
    load_fefet_ref_fit,
    save_fefet_ref_fit,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs_fefet_v4")
PLOT_WINDOW_XY = (80, 80)


def place_figure_window(fig: matplotlib.figure.Figure, xy: tuple[int, int] | None = None) -> None:
    x, y = xy if xy is not None else PLOT_WINDOW_XY
    try:
        backend = matplotlib.get_backend().lower()
        mgr = fig.canvas.manager
        win = getattr(mgr, "window", None)
        if win is None:
            return
        if backend == "tkagg":
            win.wm_geometry(f"+{int(x)}+{int(y)}")
        elif "qt" in backend:
            win.move(int(x), int(y))
    except Exception:
        pass


def _norm_header(name: str) -> str:
    return str(name).strip().upper().replace(" ", "")


def assert_gate_window_valid(
    *,
    v_fold: float,
    w: float,
    vmin: float,
    vmax: float,
    v_valid: tuple[float, float],
    label: str = "",
) -> None:
    """Fail fast if SL lines or effective gates leave the measured window."""
    vlo, vhi = float(v_valid[0]), float(v_valid[1])
    w = float(w)
    v_fold = float(v_fold)
    vmin = float(vmin)
    vmax = float(vmax)
    checks: list[tuple[str, float]] = [
        ("w", w),
        ("V_FOLD - w", v_fold - w),
        ("V_SL_min", vmin),
        ("V_SL_max", vmax),
        ("V_FOLD - V_SL_min", v_fold - vmin),
        ("V_FOLD - V_SL_max", v_fold - vmax),
        # Effective gates after series offsets (SPICE G1/G2):
        ("G1_at_vmin (V_SL-w)", vmin - w),
        ("G1_at_vmax (V_SL-w)", vmax - w),
        ("G2_at_vmin (w-V_SL)", w - vmin),
        ("G2_at_vmax (w-V_SL)", w - vmax),
    ]
    for name, val in checks:
        if val < vlo - 1e-9 or val > vhi + 1e-9:
            over = val - vhi if val > vhi else vlo - val
            raise ValueError(
                f"Validity guard{(' (' + label + ')') if label else ''}: "
                f"{name}={val:.6g} V is outside v_valid=[{vlo}, {vhi}] "
                f"by {over:.6g} V"
            )


def generate_fefet_cell_deck_v4(
    expression: str,
    *,
    v_fold: float,
    w: float,
    v_mbl: float,
    v_ml: float,
    vsweep: tuple[float, float, float],
    deck_name: str,
    csv_name: str,
    workdir: str,
    allow_second_dac: bool = False,
) -> str:
    wd = os.path.abspath(workdir)
    os.makedirs(wd, exist_ok=True)
    spice_path = os.path.join(wd, deck_name)
    vmin, vmax, vstep = vsweep
    if allow_second_dac:
        # Externally driven complement: second independent DC source.
        # Linked by sweeping SL_F; SL_F_B set to V_FOLD - midpoint as a static
        # placeholder — for true dual-DAC sweeps the caller should emit custom decks.
        sl_b_block = f"V_SLB SL_F_B 0 DC {float(v_fold):.8g}"
    else:
        sl_b_block = f"BSLB  SL_F_B 0 V={{ {float(v_fold):.8g} - V(SL_F) }}"

    spice = f"""* FeFET 2-cell read mode (measured reference-curve B-source)
.OPTIONS NONLIN CONTINUE
.TEMP 25

.SUBCKT fefet d g s
BIDS d s I={{{expression}}}
.ENDS fefet

* Weight = programmed Vth, realised as an ideal series gate offset.
* Validated by measurement: rigid Vth shift, residual at the noise floor.
V_W1  SL_F    G1      DC {float(w):.8g}
V_W2  SL_F_B  G2      DC {float(v_fold) - float(w):.8g}

XMF1  MBL_F  G1  ML_F  fefet
XMF2  MBL_F  G2  ML_F  fefet

V_SL  SL_F   0 DC 0
{sl_b_block}

V_MBL MBL_F 0 DC {float(v_mbl):.8g}
V_ML  ML_F  0 DC {float(v_ml):.8g}

.DC V_SL {float(vmin):.8g} {float(vmax):.8g} {float(vstep):.8g}
.PRINT DC FORMAT=CSV file={csv_name}
+ V(SL_F) V(SL_F_B) V(G1) V(G2) I(V_ML) I(V_MBL)
.END
"""
    with open(spice_path, "w", encoding="ascii", newline="\n") as f:
        f.write(spice)
    return spice_path


def _tail(s: str | bytes | None, n: int = 2000) -> str:
    if s is None:
        return ""
    if isinstance(s, bytes):
        s = s.decode(errors="replace")
    s = str(s).strip()
    return s[-n:] if len(s) > n else s


def _cleanup_stale_outputs(*, csv_name: str, workdir: str) -> None:
    wd = os.path.abspath(workdir)
    path = os.path.join(wd, csv_name)
    if os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            pass
    for p in glob.glob(os.path.join(wd, f"{csv_name}.*")):
        try:
            os.remove(p)
        except OSError:
            pass


def run_docker_sim(
    *,
    workdir: str,
    deck_name: str,
    csv_name: str = "fefet_cell_read.csv",
) -> None:
    docker_bin = aimdc_sim_core.resolve_docker_bin()
    wd = os.path.abspath(workdir)
    os.makedirs(wd, exist_ok=True)
    _cleanup_stale_outputs(csv_name=csv_name, workdir=wd)
    inner = f"cd /work && Xyce {deck_name}"
    cmd = [
        docker_bin, "run", "--rm",
        "-e", "OMPI_ALLOW_RUN_AS_ROOT=1",
        "-e", "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1",
        "-v", f"{wd}:/work",
        aimdc_sim_core.AIMDC_DOCKER_IMAGE,
        "bash", "-lc", inner,
    ]
    cmd_str = subprocess.list2cmdline(cmd)
    try:
        r = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired as te:
        print("Docker command was:\n ", cmd_str)
        raise RuntimeError("docker/Xyce timed out after 120s") from te
    if r.returncode != 0:
        out, err = _tail(r.stdout), _tail(r.stderr)
        if out:
            print("--- Xyce stdout (tail) ---\n", out, sep="")
        if err:
            print("--- Xyce stderr (tail) ---\n", err, sep="")
        print("Docker command was:\n ", cmd_str)
        raise RuntimeError(f"docker/Xyce failed with exit code {r.returncode}")


def format_w(w: float) -> str:
    x = float(w)
    if abs(x) < 1e-15:
        return "0"
    neg = x < 0
    ax = abs(x)
    if abs(ax - round(ax)) < 1e-9:
        body = str(int(round(ax)))
    else:
        body = f"{ax:.12f}".rstrip("0").rstrip(".").replace(".", "_")
    return ("m" if neg else "") + body


def parse_w_sweep_list(text: str) -> list[float]:
    out: list[float] = []
    for token in text.split(","):
        s = token.strip()
        if not s:
            continue
        out.append(float(s))
    if not out:
        raise ValueError("--w-sweep was provided but no valid values were parsed")
    return out


class WSweepResult(TypedDict):
    w: float
    v_sl_match: float
    i_min: float
    i_max: float
    on_off: float
    v_sl: np.ndarray
    i_cell: np.ndarray


def refine_match_point_vsl(
    v_sl: np.ndarray,
    i_cell: np.ndarray,
) -> tuple[float, int]:
    """Argmin |I_cell| with optional parabolic refinement (3-point, interior)."""
    if v_sl.size == 0 or i_cell.size == 0:
        return float("nan"), -1
    order = np.argsort(v_sl)
    vs = np.asarray(v_sl[order], dtype=float)
    ic = np.abs(np.asarray(i_cell[order], dtype=float))
    idx_min = int(np.argmin(ic))
    n = ic.size
    if idx_min <= 0 or idx_min >= n - 1:
        return float(vs[idx_min]), idx_min
    x = vs[idx_min - 1 : idx_min + 2]
    y = ic[idx_min - 1 : idx_min + 2]
    coef = np.polyfit(x, y, 2)
    a_poly = float(coef[0])
    b_poly = float(coef[1])
    if abs(a_poly) < 1e-300 or a_poly <= 0.0:
        return float(vs[idx_min]), idx_min
    xv = -b_poly / (2.0 * a_poly)
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if xv < xmin - 1e-9 or xv > xmax + 1e-9:
        return float(vs[idx_min]), idx_min
    return float(xv), idx_min


def find_match_point_continuous(
    model: FefetRefModel,
    *,
    w: float,
    v_fold: float,
    v_mbl: float,
    v_ml: float,
    bracket: tuple[float, float],
) -> float:
    """Bounded 1-D minimizer of |I_ML|(V_SL); more accurate than grid argmin for flat V-curves."""
    from scipy.optimize import minimize_scalar

    lo, hi = bracket

    def obj(v: float) -> float:
        i = analytical_i_ml(
            model,
            v_sl=np.asarray([v], dtype=float),
            w=w,
            v_fold=v_fold,
            v_mbl=v_mbl,
            v_ml=v_ml,
        )
        return float(np.abs(i[0]))

    res = minimize_scalar(obj, bounds=(lo, hi), method="bounded")
    return float(res.x)


def extract_cell_iv_v4(
    csv_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return V(SL_F), V(SL_F_B), V(G1), V(G2), I(V_ML), I(V_MBL)."""
    df = pd.read_csv(csv_path)
    colmap: dict[str, str] = {_norm_header(c): c for c in df.columns}

    def pick(key: str) -> str:
        k = _norm_header(key)
        if k not in colmap:
            raise ValueError(f"Missing required column matching {key!r} in {csv_path!r}")
        return colmap[k]

    v_sl = df[pick("V(SL_F)")].astype(float).to_numpy(dtype=float)
    v_slb = df[pick("V(SL_F_B)")].astype(float).to_numpy(dtype=float)
    v_g1 = df[pick("V(G1)")].astype(float).to_numpy(dtype=float)
    v_g2 = df[pick("V(G2)")].astype(float).to_numpy(dtype=float)
    i_ml = df[pick("I(V_ML)")].astype(float).to_numpy(dtype=float)
    i_mbl = df[pick("I(V_MBL)")].astype(float).to_numpy(dtype=float)
    return v_sl, v_slb, v_g1, v_g2, i_ml, i_mbl


def analytical_i_ml(
    model: FefetRefModel,
    *,
    v_sl: np.ndarray,
    w: float,
    v_fold: float,
    v_mbl: float,
    v_ml: float,
) -> np.ndarray:
    """Python twin of the 2-FeFET cell (no V_DS factor by default)."""
    v_sl_a = np.asarray(v_sl, dtype=float)
    # Effective gates after series Vth offsets (same as SPICE V_W1/V_W2).
    # Xyce: G1 = SL_F - w  when V_W1 is from SL_F to G1 with DC=w
    # Actually V_W1 SL_F G1 DC w  means V(SL_F) - V(G1) = w  ⇒ V(G1) = V(SL_F) - w
    g1 = v_sl_a - float(w)
    g2 = (float(v_fold) - v_sl_a) - (float(v_fold) - float(w))  # = w - v_sl
    # Source at ML_F, drain at MBL_F ⇒ V_gs ≈ V(G) - V(ML)
    v_gs1 = g1 - float(v_ml)
    v_gs2 = g2 - float(v_ml)
    i1 = model.current(v_gs1, dv=0.0, allow_extrapolation=False)
    i2 = model.current(v_gs2, dv=0.0, allow_extrapolation=False)
    return np.asarray(i1 + i2, dtype=float)


def save_operating_point_json_v2(
    out_path: str,
    *,
    circuit_params: dict[str, float],
    w_sweep: list[float],
    sweep_results: list[WSweepResult],
    fit_path: str,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    w_arr = np.asarray(w_sweep, dtype=float)
    vsl_list: list[float] = []
    i_min_list: list[float] = []
    i_max_list: list[float] = []
    for res in sweep_results:
        v_sl = np.asarray(res["v_sl"], dtype=float)
        i_cell = np.abs(np.asarray(res["i_cell"], dtype=float))
        vsm, _ = refine_match_point_vsl(v_sl, i_cell)
        vsl_list.append(float(vsm))
        i_min_list.append(float(res["i_min"]))
        i_max_list.append(float(res["i_max"]))

    vsl_arr = np.asarray(vsl_list, dtype=float)
    valid = np.isfinite(vsl_arr) & np.isfinite(w_arr)
    if int(np.sum(valid)) >= 2:
        x = w_arr[valid]
        y = vsl_arr[valid]
        slope, intercept = np.polyfit(x, y, 1)
        y_hat = slope * x + intercept
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        slope_val, intercept_val = float(slope), float(intercept)
    else:
        slope_val = intercept_val = r2 = float("nan")

    payload: dict[str, Any] = {
        "schema_version": 2,
        "validated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "circuit_params": {k: float(v) for k, v in circuit_params.items()},
        "validation": {
            "w_sweep": [float(v) for v in w_arr.tolist()],
            "v_sl_match_observed": vsl_list,
            "i_min_per_w": i_min_list,
            "i_max_per_w": i_max_list,
            "match_slope": slope_val,
            "match_intercept": intercept_val,
            "match_r_squared": r2,
        },
        "source": {
            "script": "fefet_cell_read_v4.py",
            "fit_path": str(fit_path),
        },
    }
    if extra_metadata:
        payload["source"].update({k: str(v) for k, v in extra_metadata.items()})
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved NVM operating point: {out_path}")
    print(f"  V_SL_match vs w slope = {slope_val:g}  (ideal: 1.0)")
    print(f"  V_SL_match vs w R^2   = {r2:g}")


def run_analytical_w_sweep(
    model: FefetRefModel,
    *,
    w_list: list[float],
    v_fold: float,
    v_mbl: float,
    v_ml: float,
    vsweep: tuple[float, float, float],
) -> list[WSweepResult]:
    vmin, vmax, vstep = vsweep
    v_sl = np.arange(vmin, vmax + vstep * 0.5, vstep, dtype=float)
    results: list[WSweepResult] = []
    for w in w_list:
        assert_gate_window_valid(
            v_fold=v_fold, w=w, vmin=vmin, vmax=vmax, v_valid=model.v_valid, label=f"w={w}"
        )
        i_ml = analytical_i_ml(
            model, v_sl=v_sl, w=w, v_fold=v_fold, v_mbl=v_mbl, v_ml=v_ml
        )
        i_cell = np.abs(i_ml)
        # Continuous refine around the grid argmin (LUT V-curves can be very flat).
        v_sl_match = find_match_point_continuous(
            model,
            w=w,
            v_fold=v_fold,
            v_mbl=v_mbl,
            v_ml=v_ml,
            bracket=(float(vmin), float(vmax)),
        )
        i_min = float(np.min(i_cell))
        i_max = float(np.max(i_cell))
        results.append(
            {
                "w": float(w),
                "v_sl_match": float(v_sl_match),
                "i_min": i_min,
                "i_max": i_max,
                "on_off": float(i_max / max(i_min, 1e-300)),
                "v_sl": v_sl.copy(),
                "i_cell": i_cell,
            }
        )
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="FeFET 2-cell read-mode DC sweep (v4, no inverter)"
    )
    parser.add_argument("--v-fold", type=float, default=0.0, dest="v_fold")
    parser.add_argument("--v-mbl", type=float, default=0.5)
    parser.add_argument("--v-ml", type=float, default=0.0)
    parser.add_argument("--w", type=float, default=None, help="Single programmed weight (V)")
    parser.add_argument(
        "--w-sweep",
        default=None,
        metavar="LIST",
        help="Comma-separated w list (replaces --vp-sweep)",
    )
    parser.add_argument("--vmin", type=float, default=-2.0)
    parser.add_argument("--vmax", type=float, default=2.0)
    parser.add_argument("--vstep", type=float, default=0.05)
    parser.add_argument(
        "--allow-second-dac",
        action="store_true",
        help="Drive SL_F_B with an independent DC source instead of BSLB.",
    )
    parser.add_argument(
        "--fit-json",
        default=None,
        help="Existing FefetRefModel fit JSON (default: fit from --measure-dir)",
    )
    parser.add_argument(
        "--measure-dir",
        default=None,
        help="Directory of *_dac_sweep_*.xlsx (used when --fit-json omitted)",
    )
    parser.add_argument(
        "--analytical-only",
        action="store_true",
        help="Skip Docker/Xyce; evaluate the Python twin of the cell.",
    )
    parser.add_argument(
        "--save-operating-point",
        dest="save_operating_point",
        type=str,
        default=None,
    )
    _argv = list(sys.argv[1:] if argv is None else argv)
    _flags = {"--w-sweep"}
    _fixed: list[str] = []
    i = 0
    while i < len(_argv):
        tok = _argv[i]
        if tok in _flags and i + 1 < len(_argv):
            nxt = _argv[i + 1]
            if nxt.startswith("-") and not nxt.startswith("--"):
                _fixed.append(f"{tok}={nxt}")
                i += 2
                continue
        _fixed.append(tok)
        i += 1
    args = parser.parse_args(_fixed)

    if args.w_sweep is not None and args.w is not None:
        parser.error("--w-sweep cannot be combined with --w")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.fit_json:
        model = load_fefet_ref_fit(
            args.fit_json if os.path.isabs(args.fit_json)
            else os.path.join(SCRIPT_DIR, args.fit_json)
        )
        fit_path = args.fit_json
    else:
        mdir = args.measure_dir or os.environ.get("FEFET_MEASURE_DIR")
        if not mdir:
            raise SystemExit("Provide --fit-json or --measure-dir")
        sweeps = load_sweep_dir(mdir)
        ref = next(s for s in sweeps if s.code == 300)
        model = fit_fefet_ref_model(
            ref,
            backend="lut",
            source_files=[os.path.basename(s.source_path) for s in sweeps],
        )
        fit_path = os.path.join(OUTPUT_DIR, "fefet_ref_fit.json")
        save_fefet_ref_fit(model, fit_path)

    expr = model.xyce_expression(v_ds_factor=False)
    vsweep = (args.vmin, args.vmax, args.vstep)
    v_fold = float(args.v_fold)

    if args.w_sweep is not None:
        w_list = parse_w_sweep_list(args.w_sweep)
    elif args.w is not None:
        w_list = [float(args.w)]
    else:
        w_list = [0.0]

    if args.analytical_only or len(w_list) > 1:
        results = run_analytical_w_sweep(
            model,
            w_list=w_list,
            v_fold=v_fold,
            v_mbl=float(args.v_mbl),
            v_ml=float(args.v_ml),
            vsweep=vsweep,
        )
        # Interior minimum check + linearity
        w_arr = np.array([r["w"] for r in results], dtype=float)
        vsm = np.array([r["v_sl_match"] for r in results], dtype=float)
        for r in results:
            # Single interior minimum: match not at endpoints
            vs = r["v_sl"]
            if not (float(np.min(vs)) < r["v_sl_match"] < float(np.max(vs))):
                print(f"WARNING: match for w={r['w']} not interior: {r['v_sl_match']}")
        if w_arr.size >= 2:
            slope, intercept = np.polyfit(w_arr, vsm, 1)
            y_hat = slope * w_arr + intercept
            ss_res = float(np.sum((vsm - y_hat) ** 2))
            ss_tot = float(np.sum((vsm - np.mean(vsm)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            print(f"V_SL_match vs w: slope={slope:.4f} R^2={r2:.6f}")
            if not (0.97 <= slope <= 1.03 and r2 > 0.99):
                raise SystemExit(
                    f"Acceptance failed: slope={slope}, R^2={r2} "
                    "(want slope in [0.97,1.03], R^2>0.99)"
                )
            print("ACCEPTANCE OK (analytical)")
        if args.save_operating_point:
            save_path = (
                args.save_operating_point
                if os.path.isabs(args.save_operating_point)
                else os.path.join(SCRIPT_DIR, args.save_operating_point)
            )
            save_operating_point_json_v2(
                save_path,
                circuit_params={
                    "V_FOLD": v_fold,
                    "V_MBL": float(args.v_mbl),
                    "V_ML": float(args.v_ml),
                    "v_min": float(args.vmin),
                    "v_max": float(args.vmax),
                },
                w_sweep=w_list,
                sweep_results=results,
                fit_path=str(fit_path),
            )
        if args.analytical_only:
            return

    # Docker path for single-w (or continue after analytical multi-w)
    dbin = shutil.which("docker")
    if dbin is None:
        print("docker not found; analytical-only results above (if any).")
        return
    aimdc_sim_core.AIMDC_DOCKER_BIN = dbin
    assert_docker_available()

    for w in w_list:
        assert_gate_window_valid(
            v_fold=v_fold,
            w=w,
            vmin=args.vmin,
            vmax=args.vmax,
            v_valid=model.v_valid,
            label=f"w={w}",
        )
        tag = format_w(w)
        deck_name = f"fefet_cell_read_w{tag}.sp"
        csv_name = f"fefet_cell_read_w{tag}.csv"
        generate_fefet_cell_deck_v4(
            expr,
            v_fold=v_fold,
            w=w,
            v_mbl=float(args.v_mbl),
            v_ml=float(args.v_ml),
            vsweep=vsweep,
            deck_name=deck_name,
            csv_name=csv_name,
            workdir=OUTPUT_DIR,
            allow_second_dac=bool(args.allow_second_dac),
        )
        run_docker_sim(workdir=OUTPUT_DIR, deck_name=deck_name, csv_name=csv_name)
        v_sl, _v_slb, _g1, _g2, i_ml, _i_mbl = extract_cell_iv_v4(
            os.path.join(OUTPUT_DIR, csv_name)
        )
        v_match, _ = refine_match_point_vsl(v_sl, i_ml)
        print(f"w={w:g}: V_SL_match={v_match:g}, |I|_min={np.min(np.abs(i_ml)):g}")


if __name__ == "__main__":
    main()
