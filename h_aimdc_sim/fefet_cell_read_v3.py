"""
FeFET single bit cell read-mode DC sweep with v3 IGZO TFT B-source model.
Integrated inverter + parallel FeFET pair; outputs under outputs_fefet_v3/.
"""
from __future__ import annotations

import argparse
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
from numpy.typing import NDArray

import aimdc_sim_core
from aimdc_sim_core import assert_docker_available
from aimdc_sim_core_v3 import DEFAULT_K_STEP, DEFAULT_VDS_REF, fit_igzo_model_v3

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs_fefet_v3")

PLOT_WINDOW_XY = (80, 80)


def place_figure_window(fig: matplotlib.figure.Figure, xy: tuple[int, int] | None = None) -> None:
    """Move figure window to (x, y). Ignore failures."""
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
            win.raise_()
            try:
                win.activateWindow()
            except Exception:
                pass
        elif "wx" in backend:
            win.SetPosition((int(x), int(y)))
    except Exception:
        pass


def _norm_header(name: str) -> str:
    return str(name).strip().upper().replace(" ", "")


def evaluate_v3_model(
    popt: NDArray[np.floating[Any]] | np.ndarray,
    v_gs: NDArray[np.floating[Any]] | float,
    v_ds: NDArray[np.floating[Any]] | float,
    *,
    vds_ref: float = DEFAULT_VDS_REF,
    k_step: float = DEFAULT_K_STEP,
) -> NDArray[np.floating[Any]] | np.floating[Any]:
    """Evaluate the v3 IGZO TFT model in Python (mirrors the SPICE B-source)."""
    popt_a = np.asarray(popt, dtype=float).ravel()
    if popt_a.size != 8:
        raise ValueError(f"popt must have 8 parameters; got {popt_a.size}.")
    I_off, V_th, Vss, I0_sub, I_sat_max, V_sat_scale, gamma_, delta = [float(x) for x in popt_a]
    SUB_CAP = 5.0
    v_gs_a = np.asarray(v_gs, dtype=float)
    v_ds_a = np.asarray(v_ds, dtype=float)
    vgst = v_gs_a - V_th
    vgst_smooth = 0.5 * (vgst + np.sqrt(vgst * vgst + delta * delta))
    sub_arg = np.minimum(vgst / Vss, SUB_CAP)
    i_sub = I0_sub * np.exp(sub_arg)
    tanh_core = np.tanh(vgst_smooth / V_sat_scale)
    tanh_core = np.maximum(tanh_core, 1e-300)
    i_above = I_sat_max * np.power(tanh_core, gamma_)
    ids_vgs = I_off + i_sub + i_above
    step_pos = 0.5 * (1.0 + np.tanh(k_step * v_ds_a))
    sat = v_ds_a / np.sqrt(v_ds_a * v_ds_a + vds_ref * vds_ref)
    factor = step_pos * sat
    out = ids_vgs * factor
    return out


def generate_fefet_cell_deck(
    expression: str,
    *,
    vdd: float,
    vgg: float,
    vss: float,
    v_mbl: float,
    v_ml: float,
    v_p_a: float,
    v_p_b: float,
    vsweep: tuple[float, float, float],
    deck_name: str,
    csv_name: str,
    workdir: str,
) -> str:
    wd = os.path.abspath(workdir)
    os.makedirs(wd, exist_ok=True)
    spice_path = os.path.join(wd, deck_name)
    vmin, vmax, vstep = vsweep
    spice = f"""* FeFET cell read mode (IGZO TFT v3 power-law B-source)
.OPTIONS NONLIN CONTINUE
.TEMP 25

.SUBCKT igzo_tft d g s
BIDS d s I={{{expression}}}
.ENDS igzo_tft

* Inverter feeding FeFET gates (series program offsets)
XM1_INV VDD VGG B igzo_tft
XM2_INV B A VSS igzo_tft

V_PA A A_eff_a DC {v_p_a}
V_PB B A_eff_b DC {v_p_b}

XFA MBL A_eff_a ML igzo_tft
XFB MBL A_eff_b ML igzo_tft

V_VDD VDD 0 DC {vdd}
V_VGG VGG 0 DC {vgg}
V_VSS VSS 0 DC {vss}
V_MBL MBL 0 DC {v_mbl}
V_ML  ML  0 DC {v_ml}
V_A   A   0 DC 0

.DC V_A {vmin} {vmax} {vstep}
.PRINT DC FORMAT=CSV file={csv_name}
+ V(A) V(B) V(A_eff_a) V(A_eff_b) I(V_ML) I(V_MBL) I(V_VDD) I(V_VSS)
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
    """Run Xyce in Docker with workdir mounted at /work (same pattern as v3 inverter)."""
    docker_bin = aimdc_sim_core.resolve_docker_bin()
    wd = os.path.abspath(workdir)
    os.makedirs(wd, exist_ok=True)
    _cleanup_stale_outputs(csv_name=csv_name, workdir=wd)

    inner = f"cd /work && Xyce {deck_name}"
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
        aimdc_sim_core.AIMDC_DOCKER_IMAGE,
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
        for line in r.stdout.strip().splitlines():
            if "Total Simulation Time" in line:
                print(line)
                break


def format_vp(vp: float) -> str:
    """Filename-safe token: 0 -> '0', 0.5 -> '0_5', 2 -> '2'; minus -> 'm' prefix."""
    x = float(vp)
    if abs(x) < 1e-15:
        return "0"
    neg = x < 0
    ax = abs(x)
    if abs(ax - round(ax)) < 1e-9:
        body = str(int(round(ax)))
    else:
        body = f"{ax:.12f}".rstrip("0").rstrip(".")
        body = body.replace(".", "_")
    return ("m" if neg else "") + body


def parse_vp_sweep_list(text: str) -> list[float]:
    """Parse comma-separated V_p magnitudes for symmetric (+vp, -vp) sweep."""
    out: list[float] = []
    for token in text.split(","):
        s = token.strip()
        if not s:
            continue
        out.append(float(s))
    if not out:
        raise ValueError("--vp-sweep was provided but no valid values were parsed")
    return out


class VpSweepResult(TypedDict):
    vp: float
    v_a_match: float
    i_min: float
    i_max: float
    on_off: float
    v_a: np.ndarray
    i_cell: np.ndarray


def refine_match_point_va(
    v_a: np.ndarray,
    i_cell: np.ndarray,
) -> tuple[float, int]:
    """Argmin |I_cell| with optional parabolic refinement (3-point window, interior only)."""
    if v_a.size == 0 or i_cell.size == 0:
        return float("nan"), -1
    order = np.argsort(v_a)
    va = np.asarray(v_a[order], dtype=float)
    ic = np.asarray(i_cell[order], dtype=float)
    idx_min = int(np.argmin(ic))
    n = ic.size
    if idx_min <= 0 or idx_min >= n - 1:
        return float(va[idx_min]), idx_min
    x = va[idx_min - 1 : idx_min + 2]
    y = ic[idx_min - 1 : idx_min + 2]
    coef = np.polyfit(x, y, 2)
    a_poly = float(coef[0])
    b_poly = float(coef[1])
    if abs(a_poly) < 1e-300:
        return float(va[idx_min]), idx_min
    # Minimum of upward-opening parabola
    if a_poly <= 0.0:
        return float(va[idx_min]), idx_min
    xv = -b_poly / (2.0 * a_poly)
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if xv < xmin - 1e-9 or xv > xmax + 1e-9:
        return float(va[idx_min]), idx_min
    return float(xv), idx_min


def extract_cell_iv(csv_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Xyce DC CSV; return V(A), V(B), I(V_ML), I(V_MBL), I(V_VDD)."""
    df = pd.read_csv(csv_path)
    targets = {
        "V(A)",
        "V(B)",
        "I(V_ML)",
        "I(V_MBL)",
        "I(V_VDD)",
    }
    colmap: dict[str, str] = {}
    for c in df.columns:
        colmap[_norm_header(c)] = c

    def pick(key: str) -> str:
        k = _norm_header(key)
        if k not in colmap:
            raise ValueError(f"Missing required column matching {key!r} in {csv_path!r}")
        return colmap[k]

    v_a = df[pick("V(A)")].astype(float).to_numpy(dtype=float)
    v_b = df[pick("V(B)")].astype(float).to_numpy(dtype=float)
    i_ml = df[pick("I(V_ML)")].astype(float).to_numpy(dtype=float)
    i_mbl = df[pick("I(V_MBL)")].astype(float).to_numpy(dtype=float)
    i_vdd = df[pick("I(V_VDD)")].astype(float).to_numpy(dtype=float)
    return v_a, v_b, i_ml, i_mbl, i_vdd


def _extract_eff_gates(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    colmap: dict[str, str] = {}
    for c in df.columns:
        colmap[_norm_header(c)] = c

    def pick(key: str) -> str:
        k = _norm_header(key)
        if k not in colmap:
            raise ValueError(f"Missing required column matching {key!r} in {csv_path!r}")
        return colmap[k]

    v_ea = df[pick("V(A_eff_a)")].astype(float).to_numpy(dtype=float)
    v_eb = df[pick("V(A_eff_b)")].astype(float).to_numpy(dtype=float)
    return v_ea, v_eb


def plot_cell_iv(
    v_a: np.ndarray,
    v_b: np.ndarray,
    i_ml: np.ndarray,
    _i_mbl: np.ndarray,
    _i_vdd: np.ndarray,
    v_a_eff_a: np.ndarray,
    v_a_eff_b: np.ndarray,
    popt: NDArray[np.floating[Any]],
    *,
    v_mbl: float,
    v_ml: float,
    v_p_a: float,
    v_p_b: float,
    out_path: str,
) -> None:
    """2x2 summary figure: cell current, inverter VTC, effective gates, per-device currents."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    i_cell = np.abs(i_ml)
    eps = 1e-300
    pos = np.where(np.isfinite(i_cell) & (i_cell > 0), i_cell, np.nan)
    hi = np.nanmax(pos)
    lo = np.nanmin(pos)
    if not np.isfinite(hi) or not np.isfinite(lo) or lo <= 0:
        ratio = 1.0
    else:
        ratio = float((hi + eps) / (lo + eps))
    if not np.isfinite(ratio):
        ratio = 1.0
    ax00 = axes[0, 0]
    if ratio > 100.0:
        plot_y = np.maximum(i_cell, np.nanmax(i_cell) * 1e-15 + 1e-300)
        ax00.semilogy(v_a, plot_y, "C0-", lw=2)
        ax00.set_ylabel("|I_cell| (A)")
    else:
        ax00.plot(v_a, i_cell, "C0-", lw=2)
        ymin = float(np.nanmin([0.0, np.nanmin(i_cell)]))
        ymax = float(np.nanmax(i_cell))
        pad = (ymax - ymin) * 0.05 + 1e-30
        ax00.set_ylim(ymin - pad, ymax + pad)
        ax00.set_ylabel("|I_cell| (A)")
    ax00.axvline(0.0, color="0.65", ls="--", lw=1.0)
    ax00.set_xlabel("V_A (V)")
    ax00.set_title("Cell Read Current vs V_A")
    ax00.grid(True, ls="--", alpha=0.6)

    axes[0, 1].plot(v_a, v_b, "b-", lw=2)
    axes[0, 1].axhline(0.0, color="0.75", ls="--", lw=1.0)
    axes[0, 1].grid(True, ls="--")
    axes[0, 1].set_xlabel("V_A (V)")
    axes[0, 1].set_ylabel("V_B (V)")
    axes[0, 1].set_title("Inverter VTC (sanity check)")

    axes[1, 0].plot(v_a, v_a_eff_a, "C1-", lw=2, label="V(A_eff_a)")
    axes[1, 0].plot(v_a, v_a_eff_b, "C2-", lw=2, label="V(A_eff_b)")
    axes[1, 0].axhline(0.0, color="0.75", ls=":", lw=1.0)
    axes[1, 0].set_xlabel("V_A (V)")
    axes[1, 0].set_ylabel("Voltage (V)")
    axes[1, 0].set_title("Effective FeFET Gate Voltages")
    axes[1, 0].grid(True, ls="--", alpha=0.6)
    axes[1, 0].legend(loc="best")
    txt = f"V_p_a={v_p_a:+.4g} V, V_p_b={v_p_b:+.4g} V"
    axes[1, 0].annotate(
        txt,
        xy=(0.02, 0.98),
        xycoords="axes fraction",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.35),
    )

    v_ds_cell = float(v_mbl - v_ml)
    i_a = evaluate_v3_model(popt, v_a_eff_a, v_ds_cell)
    i_b = evaluate_v3_model(popt, v_a_eff_b, v_ds_cell)
    axes[1, 1].plot(v_a, np.asarray(i_a, dtype=float), "C1-", lw=2, label="I_FeFET_a")
    axes[1, 1].plot(v_a, np.asarray(i_b, dtype=float), "C2-", lw=2, label="I_FeFET_b")
    axes[1, 1].set_xlabel("V_A (V)")
    axes[1, 1].set_ylabel("I (A)")
    axes[1, 1].set_title("Per-FeFET current contributions")
    axes[1, 1].grid(True, ls="--", alpha=0.6)
    axes[1, 1].legend(loc="best")

    fig.tight_layout()
    fig.canvas.draw()
    place_figure_window(fig)
    fig.savefig(out_path, dpi=300)
    print(f"Saved figure: {out_path}")
    place_figure_window(fig)
    plt.show()


def _interp_at_va(v_a: np.ndarray, y: np.ndarray, va0: float) -> float:
    if v_a.size == 0:
        return float("nan")
    order = np.argsort(v_a)
    xs = v_a[order]
    ys = y[order]
    return float(np.interp(va0, xs, ys))


def plot_vp_sweep(
    results: list[VpSweepResult],
    *,
    out_path: str,
) -> None:
    """2x2 figure: I_cell overlays, match tracking, I_min log, on/off ratio log."""
    n = len(results)
    if n == 0:
        return
    colors = plt.cm.viridis(np.linspace(0.0, 1.0, n))
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax00 = axes[0, 0]
    for i, res in enumerate(results):
        vp = float(res["vp"])
        v_a = res["v_a"]
        ic = np.asarray(res["i_cell"], dtype=float)
        plot_y = np.maximum(ic, np.nanmax(ic) * 1e-15 + 1e-300)
        label = f"V_p = {vp:g} V"
        ax00.semilogy(v_a, plot_y, "-", lw=2, color=colors[i], label=label)
        vam = float(res["v_a_match"])
        ymin = max(float(res["i_min"]), 1e-300)
        ax00.semilogy(vam, ymin, "+", ms=10, mew=1.5, color=colors[i])
        ax00.annotate(
            f"V_A_match={vam:.3g} V",
            xy=(vam, ymin),
            xytext=(6, (8 + (i % 4) * 6) * (1 if i % 2 == 0 else -1)),
            textcoords="offset points",
            fontsize=8,
            color=colors[i],
        )
    ax00.set_xlabel("V_A (V)")
    ax00.set_ylabel("|I_cell| (A)")
    ax00.set_title("Cell Read Current vs V_A (V_program sweep)")
    ax00.grid(True, ls="--", alpha=0.6)
    ax00.legend(loc="best", fontsize=8)

    vp_arr = np.array([float(r["vp"]) for r in results], dtype=float)
    va_m = np.array([float(r["v_a_match"]) for r in results], dtype=float)
    ax01 = axes[0, 1]
    ax01.plot(vp_arr, va_m, "o-", color="C0", lw=2, ms=7, label="Measured")
    vlo = float(np.min(vp_arr))
    vhi = float(np.max(vp_arr))
    ax01.plot([vlo, vhi], [vlo, vhi], "--", color="0.45", lw=1.5, label="Ideal (V_A_match = V_p)")
    ax01.set_xlabel("V_p (V)")
    ax01.set_ylabel("V_A at min |I_cell| (V)")
    ax01.set_title("Match Point Tracking")
    ax01.grid(True, ls="--", alpha=0.6)
    ax01.legend(loc="best")

    i_mins = np.array([float(r["i_min"]) for r in results], dtype=float)
    ax10 = axes[1, 0]
    ax10.semilogy(vp_arr, np.maximum(i_mins, 1e-300), "o-", color="C1", lw=2, ms=7)
    ax10.set_xlabel("V_p (V)")
    ax10.set_ylabel("min |I_cell| (A)")
    ax10.set_title("Minimum Cell Current at Match")
    ax10.grid(True, ls="--", alpha=0.6)

    ratios = np.array([float(r["on_off"]) for r in results], dtype=float)
    ax11 = axes[1, 1]
    ax11.semilogy(vp_arr, np.maximum(ratios, 1e-300), "o-", color="C2", lw=2, ms=7)
    ax11.set_xlabel("V_p (V)")
    ax11.set_ylabel("max |I_cell| / min |I_cell|")
    ax11.set_title("On/Off Ratio (max I / min I)")
    ax11.grid(True, ls="--", alpha=0.6)

    fig.tight_layout()
    fig.canvas.draw()
    place_figure_window(fig)
    fig.savefig(out_path, dpi=300)
    print(f"Saved figure: {out_path}")
    place_figure_window(fig)
    plt.show()


def print_vp_sweep_summary(
    results: list[VpSweepResult],
    *,
    vdd: float,
    vgg: float,
    vss: float,
    v_mbl: float,
    v_ml: float,
) -> None:
    vp_arr = np.array([float(r["vp"]) for r in results], dtype=float)
    va_match_arr = np.array([float(r["v_a_match"]) for r in results], dtype=float)
    if vp_arr.size >= 2:
        coef = np.polyfit(vp_arr, va_match_arr, 1)
        slope = float(coef[0])
        intercept = float(coef[1])
        y_hat = slope * vp_arr + intercept
        ss_res = float(np.sum((va_match_arr - y_hat) ** 2))
        mean_y = float(np.mean(va_match_arr))
        ss_tot = float(np.sum((va_match_arr - mean_y) ** 2))
        r2 = float("nan") if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)
    elif vp_arr.size == 1:
        slope = float("nan")
        r2 = float("nan")
    else:
        slope = float("nan")
        r2 = float("nan")

    print()
    print("=== V_program Sweep Results ===")
    print(
        f"Bias: VDD={vdd:g} V, VGG={vgg:g} V, VSS={vss:g} V, "
        f"MBL={v_mbl:g} V, ML={v_ml:g} V"
    )
    print()
    hdr = "V_p (V) | V_A_match (V) | I_min (A)      | I_max (A)      | On/Off"
    sep = "--------|-----------------|---------------|---------------|--------"
    print(hdr)
    print(sep)
    for r in results:
        vp = float(r["vp"])
        print(
            f" {vp:5.3g}  | {float(r['v_a_match']):15.6g} | {float(r['i_min']):13.6g} | "
            f"{float(r['i_max']):13.6g} | {float(r['on_off']):7.4g}"
        )
    print()
    print(f"Match point linearity: V_A_match vs V_p slope = {slope:g} (ideal: 1.0)")
    print(f"Match point linearity: V_A_match vs V_p R^2 = {r2:g}")


def _peak_abs_di_dva(v_a: np.ndarray, i_ml: np.ndarray) -> tuple[float, float]:
    if v_a.size < 2:
        return float("nan"), float("nan")
    order = np.argsort(v_a)
    xs = v_a[order]
    ys = i_ml[order]
    di = np.gradient(ys, xs)
    adi = np.abs(di)
    idx = int(np.nanargmax(adi))
    return float(adi[idx]), float(xs[idx])


def save_operating_point_json(
    out_path: str,
    *,
    circuit_params: dict[str, float],
    vp_sweep: list[float],
    sweep_results: list[VpSweepResult],
    fit_csv: str,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    """Save validated NVM cell operating point with V_A_match measurements.

    The JSON contains:
      - circuit_params: VDD, VGG, VSS, V_MBL, V_ML, v_min, v_max — the bias
        configuration under which the V-shape was validated.
      - validation: per-V_p V_A_match readings, slope/R^2 of V_A_match vs V_p,
        current floor/ceiling, dynamic range. Empirical evidence that the cell
        produces clean V-shape distance behavior at this operating point.
      - source: provenance (script name, fit CSV, ISO timestamp).

    Downstream consumption:
      - aimdc_sim_core_v3.py::fit_v3 CLI takes this JSON via --operating-point,
        embedding circuit_params into igzo_fit_v3.json so the online pipeline
        uses identical voltages.
      - online_accelerator_device.py prints the operating point at startup so
        the user can confirm the run is using the validated configuration.
    """
    import datetime

    vp_arr = np.asarray(vp_sweep, dtype=float)
    vam_list: list[float] = []
    i_min_list: list[float] = []
    i_max_list: list[float] = []
    for res in sweep_results:
        v_a = np.asarray(res["v_a"], dtype=float)
        i_cell = np.abs(np.asarray(res["i_cell"], dtype=float))
        vam, _ = refine_match_point_va(v_a, i_cell)
        vam_list.append(float(vam))
        if i_cell.size:
            i_min_list.append(
                float(np.min(i_cell[i_cell > 0])) if np.any(i_cell > 0) else float("nan")
            )
            i_max_list.append(float(np.max(i_cell)))
        else:
            i_min_list.append(float("nan"))
            i_max_list.append(float("nan"))

    vam_arr = np.asarray(vam_list, dtype=float)
    valid_mask = np.isfinite(vam_arr) & np.isfinite(vp_arr)
    if int(np.sum(valid_mask)) >= 2:
        x = vp_arr[valid_mask]
        y = vam_arr[valid_mask]
        slope, intercept = np.polyfit(x, y, 1)
        y_hat = slope * x + intercept
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        slope_val = float(slope)
        intercept_val = float(intercept)
    else:
        slope_val = float("nan")
        intercept_val = float("nan")
        r2 = float("nan")

    finite_min = [v for v in i_min_list if np.isfinite(v) and v > 0]
    finite_max = [v for v in i_max_list if np.isfinite(v) and v > 0]
    if finite_min and finite_max:
        floor = float(min(finite_min))
        ceiling = float(max(finite_max))
        dynamic_range = ceiling / floor if floor > 0 else float("nan")
    else:
        floor = float("nan")
        ceiling = float("nan")
        dynamic_range = float("nan")

    payload: dict[str, Any] = {
        "schema_version": 1,
        "validated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "circuit_params": {k: float(v) for k, v in circuit_params.items()},
        "validation": {
            "vp_sweep": [float(v) for v in vp_arr.tolist()],
            "v_a_match_observed": vam_list,
            "i_min_per_vp": i_min_list,
            "i_max_per_vp": i_max_list,
            "match_slope": slope_val,
            "match_intercept": intercept_val,
            "match_r_squared": r2,
            "current_floor": floor,
            "current_ceiling": ceiling,
            "dynamic_range": dynamic_range,
        },
        "source": {
            "script": "fefet_cell_read_v3.py",
            "fit_csv": str(fit_csv),
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
    print(f"  V_A_match slope = {slope_val:g}  (ideal: 1.0)")
    print(f"  V_A_match R^2   = {r2:g}")
    if np.isfinite(dynamic_range):
        print(f"  Dynamic range   = {dynamic_range:.3g} ({floor:.2e} A -> {ceiling:.2e} A)")


def main() -> None:
    parser = argparse.ArgumentParser(description="FeFET cell read-mode DC sweep (v3 IGZO TFT)")
    parser.add_argument("--vdd", type=float, default=3.0)
    parser.add_argument("--vgg", type=float, default=3.0)
    parser.add_argument("--vss", type=float, default=-3.0)
    parser.add_argument("--v-mbl", type=float, default=0.5)
    parser.add_argument("--v-ml", type=float, default=0.0)
    parser.add_argument(
        "--vp-a",
        type=float,
        default=None,
        dest="vp_a",
        metavar="V",
        help="V_p_a program offset (V); single-point mode only (default 1.0 if omitted)",
    )
    parser.add_argument(
        "--vp-b",
        type=float,
        default=None,
        dest="vp_b",
        metavar="V",
        help="V_p_b program offset (V); single-point mode only (default -1.0 if omitted)",
    )
    parser.add_argument(
        "--vp-sweep",
        default=None,
        metavar="LIST",
        help="Comma-separated V_p list (e.g. '0,0.5,1'). For negative-leading "
             "values, use either `--vp-sweep=-1,-0.5,0,0.5,1` (with '=') or rely "
             "on the built-in argv fixup that handles `--vp-sweep -1,...` too. "
             "Mutually exclusive with --vp-a/--vp-b.",
    )
    parser.add_argument("--vmin", type=float, default=-2.0)
    parser.add_argument("--vmax", type=float, default=2.0)
    parser.add_argument("--vstep", type=float, default=0.05)
    parser.add_argument(
        "--save-operating-point",
        dest="save_operating_point",
        type=str,
        default=None,
        help="If set with --vp-sweep, save the validated NVM operating point and "
             "V_A_match measurements as a JSON file. The file will be consumed by "
             "fit_v3 CLI (--operating-point) and printed by online pipeline at startup.",
    )
    parser.add_argument(
        "--csv",
        default="igzo_measure.csv",
        help="IGZO measurement CSV (relative paths resolved against script dir)",
    )
    # Workaround for argparse's option-prefix heuristic: if the user wrote
    # `--flag <neg-leading-value>`, rewrite it as `--flag=<value>` so the
    # leading `-` is not mistaken for a new option. Targets only options
    # that take comma-separated lists or arbitrary strings.
    _argv = list(sys.argv[1:])
    _flags_needing_fixup = {"--vp-sweep"}
    _fixed: list[str] = []
    i = 0
    while i < len(_argv):
        tok = _argv[i]
        if tok in _flags_needing_fixup and i + 1 < len(_argv):
            nxt = _argv[i + 1]
            # Heuristic: if next token starts with '-' but is NOT another known
            # option, treat it as the value and join with '='.
            if (
                nxt.startswith("-")
                and not nxt.startswith("--")
                and not nxt.lstrip("-").replace(".", "", 1).isalpha()
            ):
                _fixed.append(f"{tok}={nxt}")
                i += 2
                continue
        _fixed.append(tok)
        i += 1
    args = parser.parse_args(_fixed)

    if args.vp_sweep is not None and (args.vp_a is not None or args.vp_b is not None):
        parser.error("--vp-sweep cannot be combined with --vp-a or --vp-b")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    csv_fit = args.csv if os.path.isabs(args.csv) else os.path.join(SCRIPT_DIR, args.csv)
    csv_fit = os.path.abspath(csv_fit)
    if not os.path.isfile(csv_fit):
        raise FileNotFoundError(f"IGZO CSV not found: {csv_fit!r}")

    dbin = shutil.which("docker")
    if dbin is None:
        raise RuntimeError("docker binary not found in PATH.")
    aimdc_sim_core.AIMDC_DOCKER_BIN = dbin
    assert_docker_available()

    popt, expr = fit_igzo_model_v3(csv_fit)
    vsweep = (args.vmin, args.vmax, args.vstep)

    if args.vp_sweep is not None:
        vp_list = parse_vp_sweep_list(args.vp_sweep)
        results: list[VpSweepResult] = []
        for vp in vp_list:
            tag = format_vp(vp)
            deck_name = f"fefet_cell_read_vp{tag}.sp"
            csv_name = f"fefet_cell_read_vp{tag}.csv"
            v_p_a = float(vp)
            v_p_b = float(-vp)
            generate_fefet_cell_deck(
                expr,
                vdd=args.vdd,
                vgg=args.vgg,
                vss=args.vss,
                v_mbl=args.v_mbl,
                v_ml=args.v_ml,
                v_p_a=v_p_a,
                v_p_b=v_p_b,
                vsweep=vsweep,
                deck_name=deck_name,
                csv_name=csv_name,
                workdir=OUTPUT_DIR,
            )
            run_docker_sim(workdir=OUTPUT_DIR, deck_name=deck_name, csv_name=csv_name)
            out_csv = os.path.join(OUTPUT_DIR, csv_name)
            v_a, _v_b, i_ml, _i_mbl, _i_vdd = extract_cell_iv(out_csv)
            i_cell = np.abs(np.asarray(i_ml, dtype=float))
            i_min = float(np.min(i_cell))
            i_max = float(np.max(i_cell))
            i_floor = max(i_min, 1e-300)
            on_off = float(i_max / i_floor)
            v_a_match, _idx_ref = refine_match_point_va(v_a, i_cell)
            row: VpSweepResult = {
                "vp": float(vp),
                "v_a_match": float(v_a_match),
                "i_min": i_min,
                "i_max": i_max,
                "on_off": on_off,
                "v_a": np.asarray(v_a, dtype=float),
                "i_cell": i_cell,
            }
            results.append(row)

        out_png = os.path.join(OUTPUT_DIR, "fefet_cell_read_vp_sweep.png")
        plot_vp_sweep(results, out_path=out_png)
        print_vp_sweep_summary(
            results,
            vdd=args.vdd,
            vgg=args.vgg,
            vss=args.vss,
            v_mbl=args.v_mbl,
            v_ml=args.v_ml,
        )
        if args.save_operating_point:
            save_path = (
                args.save_operating_point
                if os.path.isabs(args.save_operating_point)
                else os.path.join(SCRIPT_DIR, args.save_operating_point)
            )
            circuit_params_dict = {
                "VDD": float(args.vdd),
                "VGG": float(args.vgg),
                "VSS": float(args.vss),
                "V_MBL": float(args.v_mbl),
                "V_ML": float(args.v_ml),
                "v_min": float(args.vmin),
                "v_max": float(args.vmax),
            }
            save_operating_point_json(
                save_path,
                circuit_params=circuit_params_dict,
                vp_sweep=vp_list,
                sweep_results=results,
                fit_csv=str(args.csv),
            )
        return

    vp_a_single = 1.0 if args.vp_a is None else float(args.vp_a)
    vp_b_single = -1.0 if args.vp_b is None else float(args.vp_b)

    deck_name = "fefet_cell_read.sp"
    csv_name = "fefet_cell_read.csv"
    png_name = "fefet_cell_read.png"

    generate_fefet_cell_deck(
        expr,
        vdd=args.vdd,
        vgg=args.vgg,
        vss=args.vss,
        v_mbl=args.v_mbl,
        v_ml=args.v_ml,
        v_p_a=vp_a_single,
        v_p_b=vp_b_single,
        vsweep=vsweep,
        deck_name=deck_name,
        csv_name=csv_name,
        workdir=OUTPUT_DIR,
    )
    run_docker_sim(workdir=OUTPUT_DIR, deck_name=deck_name)

    out_csv = os.path.join(OUTPUT_DIR, csv_name)
    out_png = os.path.join(OUTPUT_DIR, png_name)

    v_a, v_b, i_ml, i_mbl, i_vdd = extract_cell_iv(out_csv)
    v_ea, v_eb = _extract_eff_gates(out_csv)

    i_cell = np.abs(i_ml)
    ic_min = float(np.nanmin(i_cell))
    ic_max = float(np.nanmax(i_cell))
    ic_at0 = _interp_at_va(v_a, i_ml, 0.0)
    peak_di, va_peak = _peak_abs_di_dva(v_a, i_ml)

    print()
    print("=== FeFET Cell Read Simulation ===")
    print(
        f"Bias: VDD={args.vdd:g} V, VGG={args.vgg:g} V, VSS={args.vss:g} V, "
        f"MBL={args.v_mbl:g} V, ML={args.v_ml:g} V"
    )
    print(f"Stored state: V_p_a={vp_a_single:+.4g} V, V_p_b={vp_b_single:+.4g} V")
    print()
    print(f"I_cell range: [{ic_min:g}, {ic_max:g}] A")
    print(f"I_cell at V_A=0 V: {ic_at0:g} A")
    print(f"Peak |dI_cell/dV_A|: {peak_di:g} A/V at V_A={va_peak:g} V")

    plot_cell_iv(
        v_a,
        v_b,
        i_ml,
        i_mbl,
        i_vdd,
        v_ea,
        v_eb,
        popt,
        v_mbl=args.v_mbl,
        v_ml=args.v_ml,
        v_p_a=vp_a_single,
        v_p_b=vp_b_single,
        out_path=out_png,
    )


if __name__ == "__main__":
    main()
