from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

import aimdc_sim_core
from aimdc_sim_core import assert_docker_available
from aimdc_sim_core_v2 import fit_igzo_model_v2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 플롯 창 왼쪽 위 모서리를 둘 화면 좌표(픽셀). 듀얼 모니터면 값을 바꿔 보세요.
PLOT_WINDOW_XY = (80, 80)


def place_figure_window(fig, xy: tuple[int, int] | None = None) -> None:
    """Tk/Qt/Wx 백엔드에서 figure 창 위치를 (x, y)로 옮김. 실패 시 무시."""
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


def generate_inverter_deck(
    expression: str,
    vdd: float,
    vgg: float,
    vsweep: tuple[float, float, float],
    *,
    workdir: str | None = None,
    deck_name: str = "inverter_vtc.sp",
    csv_name: str = "inverter_vtc.csv",
) -> str:
    wd = os.path.abspath(workdir or os.getcwd())
    os.makedirs(wd, exist_ok=True)
    spice_path = os.path.join(wd, deck_name)
    vstart, vstop, vstep = vsweep
    spice = f"""* NMOS-only inverter VTC (IGZO TFT, v2)
.OPTIONS NONLIN CONTINUE
.TEMP 25

.SUBCKT igzo_tft d g s
BIDS d s I={{{expression}}}
.ENDS igzo_tft

* M1 (load): D=VDD, G=VGG, S=B
XM1 VDD VGG B igzo_tft
* M2 (driver): D=B, G=A, S=0
XM2 B A 0 igzo_tft

V_VDD VDD 0 DC {vdd}
V_VGG VGG 0 DC {vgg}
V_A   A   0 DC 0

.DC V_A {vstart} {vstop} {vstep}
.PRINT DC FORMAT=CSV file={csv_name} V(A) V(B)
.END
"""
    with open(spice_path, "w", encoding="ascii", newline="\n") as f:
        f.write(spice)
    return spice_path


def extract_vtc(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    normalized_cols = {str(c): str(c).strip().upper().replace(" ", "") for c in df.columns}

    va_col: str | None = None
    vb_col: str | None = None
    for original, norm in normalized_cols.items():
        if norm == "V(A)":
            va_col = original
        elif norm == "V(B)":
            vb_col = original

    if va_col is None or vb_col is None:
        missing: list[str] = []
        if va_col is None:
            missing.append("V(A)")
        if vb_col is None:
            missing.append("V(B)")
        raise ValueError(f"Missing required column(s): {missing}")

    v_a = df[va_col].astype(float).to_numpy(dtype=float)
    v_b = df[vb_col].astype(float).to_numpy(dtype=float)
    return v_a, v_b


def plot_vtc(
    v_a: np.ndarray, v_b: np.ndarray, vdd: float, vgg: float, out_path: str = "inverter_vtc.png"
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(v_a, v_b, "b-", lw=2)
    ax.axhline(0.0, color="0.75", ls="--", lw=1.0)
    ax.axhline(vdd, color="0.75", ls="--", lw=1.0)
    ax.set_title(f"NMOS-only Inverter VTC (VDD={vdd} V, VGG={vgg} V)")
    ax.set_xlabel("Input V_A (V)")
    ax.set_ylabel("Output V_B (V)")
    ax.grid(True, ls="--")
    fig.tight_layout()
    fig.canvas.draw()
    place_figure_window(fig)
    fig.savefig(out_path, dpi=300)
    print(f"Saved figure: {out_path}")
    place_figure_window(fig)
    plt.show()


def plot_vtc_sweep(
    curves: list[tuple[float, np.ndarray, np.ndarray]],
    vdd: float,
    out_path: str = "inverter_vtc_vgg_sweep.png",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0.0, 1.0, len(curves)))
    for i, (vgg, v_a, v_b) in enumerate(curves):
        ax.plot(v_a, v_b, lw=2, color=colors[i], label=f"VGG = {vgg:g} V")
    ax.axhline(0.0, color="0.75", ls="--", lw=1.0)
    ax.axhline(vdd, color="0.75", ls="--", lw=1.0)
    ax.set_title(f"NMOS-only Inverter VTC (VDD={vdd} V, VGG sweep)")
    ax.set_xlabel("Input V_A (V)")
    ax.set_ylabel("Output V_B (V)")
    ax.grid(True, ls="--")
    ax.legend()
    fig.tight_layout()
    fig.canvas.draw()
    place_figure_window(fig)
    fig.savefig(out_path, dpi=300)
    print(f"Saved figure: {out_path}")
    place_figure_window(fig)
    plt.show()


def _tail(s: str | bytes | None, n: int = 2000) -> str:
    if s is None:
        return ""
    if isinstance(s, bytes):
        s = s.decode(errors="replace")
    s = str(s).strip()
    return s[-n:] if len(s) > n else s


def _cleanup_stale_outputs(
    csv_name: str = "inverter_vtc.csv", workdir: str | None = None
) -> None:
    wd = os.path.abspath(workdir or os.getcwd())
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


def _run_inverter_docker_sim(
    *,
    workdir: str | None = None,
    deck_name: str = "inverter_vtc.sp",
    csv_name: str = "inverter_vtc.csv",
) -> None:
    docker_bin = aimdc_sim_core.resolve_docker_bin()
    wd = os.path.abspath(workdir or os.getcwd())
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


def _find_crossing_x(x: np.ndarray, y: np.ndarray, target: float) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    diff = y - target
    for i in range(len(diff) - 1):
        y0 = float(diff[i])
        y1 = float(diff[i + 1])
        x0 = float(x[i])
        x1 = float(x[i + 1])
        if y0 == 0.0:
            return x0
        if y0 * y1 <= 0.0:
            if y1 == y0:
                return x0
            t = (0.0 - y0) / (y1 - y0)
            return x0 + t * (x1 - x0)
    if float(diff[-1]) == 0.0:
        return float(x[-1])
    return float("nan")


def _parse_vgg_sweep(text: str) -> list[float]:
    values: list[float] = []
    for token in text.split(","):
        s = token.strip()
        if not s:
            continue
        values.append(float(s))
    if not values:
        raise ValueError("--vgg-sweep was provided but no valid values were parsed")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="NMOS-only inverter VTC Xyce DC sweep (v2)")
    parser.add_argument("--vdd", type=float, default=3.0, help="Supply voltage VDD (V)")
    parser.add_argument("--vgg", type=float, default=1.0, help="Load gate bias VGG (V)")
    parser.add_argument(
        "--vgg-sweep",
        default=None,
        help="Comma-separated VGG list (e.g. 1,2,3,5,7). If set, --vgg is ignored.",
    )
    parser.add_argument("--vmin", type=float, default=-3.0, help="Input sweep start V_A (V)")
    parser.add_argument("--vmax", type=float, default=3.0, help="Input sweep stop V_A (V)")
    parser.add_argument("--vstep", type=float, default=0.1, help="Input sweep step V_A (V)")
    parser.add_argument(
        "--csv",
        default="igzo_measure.csv",
        help="IGZO measurement CSV path (relative paths are resolved against the script directory)",
    )
    args = parser.parse_args()
    csv_path = args.csv if os.path.isabs(args.csv) else os.path.join(SCRIPT_DIR, args.csv)
    csv_path = os.path.abspath(csv_path)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"IGZO CSV not found: --csv {args.csv!r}, resolved path: {csv_path!r}"
        )

    dbin = shutil.which("docker")
    if dbin is None:
        raise RuntimeError(
            "docker binary not found in PATH. Is Docker Desktop WSL integration enabled?"
        )
    aimdc_sim_core.AIMDC_DOCKER_BIN = dbin

    assert_docker_available()
    _popt, expr = fit_igzo_model_v2(csv_path)
    vsweep = (args.vmin, args.vmax, args.vstep)

    if args.vgg_sweep:
        vgg_values = _parse_vgg_sweep(args.vgg_sweep)
        curves: list[tuple[float, np.ndarray, np.ndarray]] = []
        vm = args.vdd / 2.0
        for vgg in vgg_values:
            suffix = f"{vgg:g}".replace("-", "m")
            deck_name = f"inverter_vtc_vgg{suffix}.sp"
            csv_name = f"inverter_vtc_vgg{suffix}.csv"
            generate_inverter_deck(
                expr,
                args.vdd,
                vgg,
                vsweep,
                workdir=SCRIPT_DIR,
                deck_name=deck_name,
                csv_name=csv_name,
            )
            _run_inverter_docker_sim(workdir=SCRIPT_DIR, deck_name=deck_name, csv_name=csv_name)
            out_csv = os.path.join(SCRIPT_DIR, csv_name)
            v_a, v_b = extract_vtc(out_csv)
            curves.append((vgg, v_a, v_b))
            va_at_vm = _find_crossing_x(v_a, v_b, vm)
            print(f"VGG={vgg:g} V -> V_A at V_B=VDD/2 ({vm:.6g} V): {va_at_vm:.6g} V")

        out_png = os.path.join(SCRIPT_DIR, "inverter_vtc_vgg_sweep.png")
        plot_vtc_sweep(curves, args.vdd, out_path=out_png)
        return

    generate_inverter_deck(
        expr,
        args.vdd,
        args.vgg,
        vsweep,
        workdir=SCRIPT_DIR,
    )
    _run_inverter_docker_sim(workdir=SCRIPT_DIR)

    out_csv = os.path.join(SCRIPT_DIR, "inverter_vtc.csv")
    out_png = os.path.join(SCRIPT_DIR, "inverter_vtc.png")
    v_a, v_b = extract_vtc(out_csv)

    vm = args.vdd / 2.0
    va_at_vm = _find_crossing_x(v_a, v_b, vm)
    vb_min = float(np.nanmin(v_b)) if v_b.size else float("nan")
    vb_max = float(np.nanmax(v_b)) if v_b.size else float("nan")
    print(f"Points: {v_a.size}")
    print(f"V_B range: [{vb_min:.6g}, {vb_max:.6g}] V")
    print(f"V_A at V_B=VDD/2 ({vm:.6g} V): {va_at_vm:.6g} V")
    plot_vtc(v_a, v_b, args.vdd, args.vgg, out_path=out_png)


if __name__ == "__main__":
    main()
