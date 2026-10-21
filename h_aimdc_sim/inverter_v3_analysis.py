from __future__ import annotations

import argparse
import os
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs_v3")
PLOT_WINDOW_XY = (80, 80)
VDD_REF = 3.0


def place_figure_window(fig, xy: tuple[int, int] | None = None) -> None:
    """Move figure window to (x, y) for Tk/Qt/Wx backends; ignore failures."""
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


def _parse_vgg_list(text: str) -> list[float]:
    values: list[float] = []
    for token in text.split(","):
        s = token.strip()
        if not s:
            continue
        values.append(float(s))
    if not values:
        raise ValueError("No valid values parsed from --vgg-list")
    return values


def _parse_vss_list(text: str) -> list[float]:
    values: list[float] = []
    for token in text.split(","):
        s = token.strip()
        if not s:
            continue
        values.append(float(s))
    if not values:
        raise ValueError("No valid values parsed from --vss-list")
    return values


def _parse_range(text: str) -> tuple[float, float]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        raise ValueError("--vth-range must be in format A,B")
    a = float(parts[0])
    b = float(parts[1])
    if a > b:
        a, b = b, a
    return a, b


def _extract_vtc(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
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


def _find_crossing_x(x: np.ndarray, y: np.ndarray, target: float = 0.0) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    d = y - target
    for i in range(len(d) - 1):
        y0 = float(d[i])
        y1 = float(d[i + 1])
        x0 = float(x[i])
        x1 = float(x[i + 1])
        if y0 == 0.0:
            return x0
        if y0 * y1 <= 0.0:
            if y1 == y0:
                return x0
            t = (0.0 - y0) / (y1 - y0)
            return x0 + t * (x1 - x0)
    if float(d[-1]) == 0.0:
        return float(x[-1])
    return float("nan")


def _interpolate_b_at_x(v_a: np.ndarray, v_b: np.ndarray, x: float) -> float:
    if x <= float(v_a[0]):
        return float(v_b[0])
    if x >= float(v_a[-1]):
        return float(v_b[-1])
    i = int(np.searchsorted(v_a, x) - 1)
    i = max(0, min(i, len(v_a) - 2))
    x0 = float(v_a[i])
    x1 = float(v_a[i + 1])
    y0 = float(v_b[i])
    y1 = float(v_b[i + 1])
    if x1 == x0:
        return y0
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def _usable_intervals(
    v_a: np.ndarray, v_b: np.ndarray, vth_min: float, vth_max: float
) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    if v_a.size < 2:
        return intervals
    for i in range(len(v_a) - 1):
        x0 = float(v_a[i])
        x1 = float(v_a[i + 1])
        y0 = float(v_b[i])
        y1 = float(v_b[i + 1])
        if x1 == x0:
            continue

        # Candidate breakpoints where any inequality boundary can change sign.
        pts = [x0, x1]
        for xb in (vth_min, vth_max):
            if min(x0, x1) <= xb <= max(x0, x1):
                pts.append(float(xb))
        for yb in (vth_min, vth_max):
            dy = y1 - y0
            if dy != 0.0:
                t = (yb - y0) / dy
                if 0.0 <= t <= 1.0:
                    pts.append(x0 + t * (x1 - x0))
        pts_sorted = sorted(set(pts))

        for j in range(len(pts_sorted) - 1):
            a = float(pts_sorted[j])
            b = float(pts_sorted[j + 1])
            if b <= a:
                continue
            xm = 0.5 * (a + b)
            ym = _interpolate_b_at_x(v_a, v_b, xm)
            ok = (vth_min <= xm <= vth_max) and (vth_min <= ym <= vth_max)
            if ok:
                intervals.append((a, b))

    if not intervals:
        return []
    intervals.sort(key=lambda t: t[0])
    merged: list[tuple[float, float]] = [intervals[0]]
    for a, b in intervals[1:]:
        la, lb = merged[-1]
        if a <= lb + 1e-12:
            merged[-1] = (la, max(lb, b))
        else:
            merged.append((a, b))
    return merged


def _segment_xy(v_a: np.ndarray, v_b: np.ndarray, a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = [a]
    ys: list[float] = [_interpolate_b_at_x(v_a, v_b, a)]
    for x, y in zip(v_a, v_b):
        xf = float(x)
        if a < xf < b:
            xs.append(xf)
            ys.append(float(y))
    xs.append(b)
    ys.append(_interpolate_b_at_x(v_a, v_b, b))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def find_linear_region(
    v_a: np.ndarray, v_b: np.ndarray, gain_tolerance: float = 0.10
) -> dict[str, Any]:
    gain = -np.gradient(v_b, v_a)
    if gain.size < 2:
        return {
            "found": False,
            "v_a_start": None,
            "v_a_end": None,
            "width": 0.0,
            "avg_gain": None,
            "gain_std_ratio": None,
            "v_b_start": None,
            "v_b_end": None,
            "linear_fit_slope": None,
            "linear_fit_intercept": None,
            "linear_fit_rmse": None,
        }

    gmax = float(np.max(gain))
    valid = gain > (0.05 * gmax)
    if not np.any(valid):
        return {
            "found": False,
            "v_a_start": None,
            "v_a_end": None,
            "width": 0.0,
            "avg_gain": None,
            "gain_std_ratio": None,
            "v_b_start": None,
            "v_b_end": None,
            "linear_fit_slope": None,
            "linear_fit_intercept": None,
            "linear_fit_rmse": None,
        }

    valid_idx = np.where(valid)[0]
    best_i = -1
    best_j = -1
    best_width = -1.0

    for i_idx in range(len(valid_idx)):
        i = int(valid_idx[i_idx])
        for j_idx in range(i_idx + 1, len(valid_idx)):
            j = int(valid_idx[j_idx])
            if j <= i:
                continue
            # contiguous interval only
            if not np.all(valid[i : j + 1]):
                continue
            gseg = gain[i : j + 1]
            gmean = float(np.mean(gseg))
            if gmean <= 0.0:
                continue
            flatness = (float(np.max(gseg)) - float(np.min(gseg))) / gmean
            if flatness < gain_tolerance:
                width = float(v_a[j] - v_a[i])
                if width > best_width:
                    best_width = width
                    best_i = i
                    best_j = j

    if best_i < 0 or best_j <= best_i:
        return {
            "found": False,
            "v_a_start": None,
            "v_a_end": None,
            "width": 0.0,
            "avg_gain": None,
            "gain_std_ratio": None,
            "v_b_start": None,
            "v_b_end": None,
            "linear_fit_slope": None,
            "linear_fit_intercept": None,
            "linear_fit_rmse": None,
        }

    va_seg = v_a[best_i : best_j + 1]
    vb_seg = v_b[best_i : best_j + 1]
    gseg = gain[best_i : best_j + 1]
    gmean = float(np.mean(gseg))
    gstd_ratio = float(np.std(gseg) / gmean) if gmean > 0 else float("nan")
    m, b = np.polyfit(va_seg, vb_seg, 1)
    vb_fit = m * va_seg + b
    rmse = float(np.sqrt(np.mean((vb_seg - vb_fit) ** 2)))

    return {
        "found": True,
        "v_a_start": float(va_seg[0]),
        "v_a_end": float(va_seg[-1]),
        "width": float(va_seg[-1] - va_seg[0]),
        "avg_gain": gmean,
        "gain_std_ratio": gstd_ratio,
        "v_b_start": float(vb_seg[0]),
        "v_b_end": float(vb_seg[-1]),
        "linear_fit_slope": float(m),
        "linear_fit_intercept": float(b),
        "linear_fit_rmse": rmse,
    }


def _fmt_bias_for_name(value: float) -> str:
    s = f"{value:g}"
    if s.startswith("-"):
        s = "m" + s[1:]
    s = s.replace(".", "_")
    return s


def _load_curves(vgg_values: list[float], vss_values: list[float]) -> dict[float, list[tuple[float, np.ndarray, np.ndarray]]]:
    curves_by_vss: dict[float, list[tuple[float, np.ndarray, np.ndarray]]] = {}
    for vss in vss_values:
        vss_tag = _fmt_bias_for_name(vss)
        curves: list[tuple[float, np.ndarray, np.ndarray]] = []
        for vgg in vgg_values:
            vgg_tag = _fmt_bias_for_name(vgg)
            primary = os.path.join(OUTPUT_DIR, f"inverter_vtc_v3_vgg{vgg_tag}_vss{vss_tag}.csv")
            fallback = os.path.join(OUTPUT_DIR, f"inverter_vtc_v3_vgg{vgg_tag}.csv")
            legacy = os.path.join(SCRIPT_DIR, f"inverter_vtc_v3_vgg{vgg_tag}.csv")
            csv_path = primary
            if not os.path.isfile(csv_path) and vss == 0.0:
                csv_path = fallback
            if not os.path.isfile(csv_path) and vss == 0.0 and os.path.isfile(legacy):
                csv_path = legacy
                print(
                    f"[notice] Using legacy CSV from script directory: {legacy}. "
                    "Consider moving old files to outputs_v3."
                )
            if not os.path.isfile(csv_path):
                print(f"[warning] Missing CSV for VGG={vgg:g} V, VSS={vss:g} V: {primary}")
                continue
            try:
                v_a, v_b = _extract_vtc(csv_path)
                curves.append((vgg, v_a, v_b))
            except Exception as e:
                print(f"[warning] Failed to parse VGG={vgg:g} V, VSS={vss:g} V ({csv_path}): {e}")
                continue
        curves_by_vss[vss] = curves
    return curves_by_vss


def _analyze_one_axes(
    ax1: plt.Axes,
    ax2: plt.Axes,
    ax3: plt.Axes,
    ax4: plt.Axes,
    ax5: plt.Axes,
    curves: list[tuple[float, np.ndarray, np.ndarray]],
    vss: float,
    vth_min: float,
    vth_max: float,
    gain_tolerance: float,
    colors: np.ndarray,
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    if not curves:
        ax1.set_title(f"Gain (VSS={vss:g} V) - no data")
        ax2.set_title(f"Trajectory (VSS={vss:g} V) - no data")
        ax3.set_title(f"Differential (VSS={vss:g} V) - no data")
        ax4.set_title(f"Read Window (VSS={vss:g} V) - no data")
        ax5.set_title(f"Linear Region (VSS={vss:g} V) - no data")
        return summary

    all_va = np.concatenate([v_a for _, v_a, _ in curves])
    line_x = np.linspace(float(np.min(all_va)), float(np.max(all_va)), 300)
    y_min = float(min(np.min(v_b) for _, _, v_b in curves))
    y_max = float(max(np.max(v_b) for _, _, v_b in curves))

    for i, (vgg, v_a, v_b) in enumerate(curves):
        color = colors[i]
        dvb_dva = -np.gradient(v_b, v_a)
        ax1.plot(v_a, dvb_dva, color=color, lw=2, label=f"VGG = {vgg:g} V")
        peak_idx = int(np.argmax(dvb_dva))
        peak_gain = float(dvb_dva[peak_idx])
        peak_va = float(v_a[peak_idx])
        ax1.plot([peak_va], [peak_gain], marker="+", color=color, markersize=8)
        ax1.annotate(
            f"peak={peak_gain:.2g} @ V_A={peak_va:.2g}",
            (peak_va, peak_gain),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
            color=color,
        )

        ax2.plot(v_a, v_b, color=color, lw=2, label=f"VGG = {vgg:g} V")
        self_cross_va = _find_crossing_x(v_a, v_b - v_a, 0.0)
        if not np.isnan(self_cross_va):
            ax2.annotate(
                f"eq @ {self_cross_va:.2g}V",
                (self_cross_va, self_cross_va),
                textcoords="offset points",
                xytext=(4, -10),
                fontsize=8,
                color=color,
            )

        delta = v_a - v_b
        ax3.plot(v_a, delta, color=color, lw=2, label=f"VGG = {vgg:g} V")

        intervals = _usable_intervals(v_a, v_b, vth_min, vth_max)
        if intervals:
            best = max(intervals, key=lambda t: t[1] - t[0])
            a0, b0 = best
            width = float(b0 - a0)
            label4 = f"VGG = {vgg:g}V (usable range: V_A in [{a0:.2g}, {b0:.2g}])"
            ax4.plot(v_a, v_b, color=color, lw=1.2, alpha=0.4)
            for a_seg, b_seg in intervals:
                xs, ys = _segment_xy(v_a, v_b, a_seg, b_seg)
                ax4.plot(xs, ys, color=color, lw=3.0, alpha=0.95, label=label4)
                label4 = "_nolegend_"
        else:
            a0 = float("nan")
            b0 = float("nan")
            width = 0.0
            ax4.plot(
                v_a,
                v_b,
                color=color,
                lw=1.2,
                alpha=0.4,
                label=f"VGG = {vgg:g}V (no usable range)",
            )
        linear = find_linear_region(v_a, v_b, gain_tolerance=gain_tolerance)
        ax5.plot(v_a, v_b, color="0.6", lw=1.0, alpha=0.3)
        if bool(linear["found"]):
            la = float(linear["v_a_start"])
            lb = float(linear["v_a_end"])
            xs, ys = _segment_xy(v_a, v_b, la, lb)
            ax5.plot(xs, ys, color=color, lw=3.0, label=f"VGG={vgg:g}V")
            lf_m = float(linear["linear_fit_slope"])
            lf_b = float(linear["linear_fit_intercept"])
            ax5.plot(xs, lf_m * xs + lf_b, color=color, ls="--", lw=1.6, label="_nolegend_")
            xm = 0.5 * (la + lb)
            ym = _interpolate_b_at_x(v_a, v_b, xm)
            ax5.annotate(
                f"V_A:[{la:.2g},{lb:.2g}]\nwidth:{float(linear['width']):.2g}\n"
                f"gain:{float(linear['avg_gain']):.2g}\nRMSE:{float(linear['linear_fit_rmse']):.2g}",
                (xm, ym),
                textcoords="offset points",
                xytext=(4, 8),
                fontsize=8,
                color=color,
            )
        else:
            ax5.plot(v_a, v_b, color=color, lw=1.2, alpha=0.5, label=f"VGG={vgg:g}V (no linear region)")

        summary.append(
            {
                "vgg": float(vgg),
                "peak_gain": peak_gain,
                "peak_va": peak_va,
                "cross_va": self_cross_va,
                "usable_a": a0,
                "usable_b": b0,
                "usable_width": width,
                "linear": linear,
            }
        )

    ax1.set_xlabel("Input V_A (V)")
    ax1.set_ylabel("Gain  -dV_B/dV_A")
    ax1.set_title(f"Inverter Small-Signal Gain (v3), VSS={vss:g}V")
    ax1.grid(True, ls="--")
    ax1.legend(fontsize=8)

    ax2.plot(line_x, line_x, color="0.5", ls="--", lw=1.0)
    ax2.plot(line_x, VDD_REF - line_x, color="0.5", ls=":", lw=1.0)
    ax2.set_xlabel("V_A (V)")
    ax2.set_ylabel("V_B (V)")
    ax2.set_title(f"Inverter Output Trajectory (v3), VSS={vss:g}V")
    ax2.grid(True, ls="--")
    ax2.legend(fontsize=8)

    ax3.axhline(0.0, color="0.5", ls="--", lw=1.0)
    ax3.set_xlabel("V_A (V)")
    ax3.set_ylabel("V_A - V_B (V)")
    ax3.set_title(f"Differential Gate Drive (v3), VSS={vss:g}V")
    ax3.grid(True, ls="--")
    ax3.legend(fontsize=8)

    ax4.axvspan(vth_min, vth_max, color="#7db4ff", alpha=0.15)
    ax4.axhspan(vth_min, vth_max, color="#ffbb7d", alpha=0.15)
    ax4.fill(
        [vth_min, vth_max, vth_max, vth_min],
        [vth_min, vth_min, vth_max, vth_max],
        color="#8e7dff",
        alpha=0.25,
        zorder=0,
    )
    ax4.axvline(vth_min, color="0.5", ls="--", lw=1.0)
    ax4.axvline(vth_max, color="0.5", ls="--", lw=1.0)
    ax4.axhline(vth_min, color="0.5", ls="--", lw=1.0)
    ax4.axhline(vth_max, color="0.5", ls="--", lw=1.0)
    ax4.set_xlim(float(np.min(all_va)), float(np.max(all_va)))
    ax4.set_ylim(min(y_min, vth_min) - 0.1, max(y_max, vth_max) + 0.1)
    ax4.set_xlabel("V_A (V)")
    ax4.set_ylabel("V_B (V)")
    ax4.set_title(f"Read Window Analysis (FeFET V_th_eff in [{vth_min:g}, {vth_max:g}] V), VSS={vss:g}V")
    ax4.grid(True, ls="--")
    ax4.legend(fontsize=8)

    ax5.set_xlabel("V_A (V)")
    ax5.set_ylabel("V_B (V)")
    ax5.set_title(f"Linear Region of VTC (gain tolerance +/-{100.0 * gain_tolerance:.0f}%)")
    ax5.grid(True, ls="--")
    ax5.legend(fontsize=8)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze existing inverter_v3 sweep CSV files")
    parser.add_argument(
        "--vgg-list",
        default="1,2,3,5,7",
        help="Comma-separated VGG values (files: inverter_vtc_v3_vgg{val}.csv)",
    )
    parser.add_argument(
        "--vss-list",
        default="0",
        help="Comma-separated VSS values (files: inverter_vtc_v3_vgg{vgg}_vss{vss}.csv)",
    )
    parser.add_argument(
        "--vth-range",
        default="-2,2",
        help="FeFET V_th_effective range as A,B (default: -2,2)",
    )
    parser.add_argument(
        "--gain-tolerance",
        type=float,
        default=0.10,
        help="Relative gain flatness tolerance for linear-region detection (default: 0.10)",
    )
    args = parser.parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Reading CSVs from: {OUTPUT_DIR}")
    print(f"Saving analysis to: {OUTPUT_DIR}")

    vgg_values = _parse_vgg_list(args.vgg_list)
    vss_values = _parse_vss_list(args.vss_list)
    vth_min, vth_max = _parse_range(args.vth_range)
    curves_by_vss = _load_curves(vgg_values, vss_values)
    if not any(curves_by_vss.get(vss, []) for vss in vss_values):
        print("No valid VTC CSV files found. Nothing to plot.")
        return

    all_summary: list[dict[str, Any]] = []
    if len(vss_values) > 1:
        fig, axes = plt.subplots(5, len(vss_values), figsize=(5 * len(vss_values), 20), squeeze=False)
        for col, vss in enumerate(vss_values):
            curves = curves_by_vss.get(vss, [])
            colors = plt.cm.viridis(np.linspace(0.0, 1.0, max(len(curves), 1)))
            summary = _analyze_one_axes(
                axes[0, col],
                axes[1, col],
                axes[2, col],
                axes[3, col],
                axes[4, col],
                curves,
                vss,
                vth_min,
                vth_max,
                args.gain_tolerance,
                colors,
            )
            for row in summary:
                row["vss"] = float(vss)
                all_summary.append(row)
    else:
        vss = vss_values[0]
        curves = curves_by_vss.get(vss, [])
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        colors = plt.cm.viridis(np.linspace(0.0, 1.0, max(len(curves), 1)))
        summary = _analyze_one_axes(
            axes[0, 0],
            axes[0, 1],
            axes[0, 2],
            axes[1, 0],
            axes[1, 1],
            curves,
            vss,
            vth_min,
            vth_max,
            args.gain_tolerance,
            colors,
        )
        axes[1, 2].axis("off")
        for row in summary:
            row["vss"] = float(vss)
            all_summary.append(row)

    fig.tight_layout()
    fig.canvas.draw()
    place_figure_window(fig)

    out_path = os.path.join(OUTPUT_DIR, "inverter_v3_analysis.png")
    fig.savefig(out_path, dpi=300)
    print(f"Saved figure: {out_path}")
    place_figure_window(fig)

    for row in all_summary:
        vss = float(row["vss"])
        vgg = float(row["vgg"])
        peak_gain = float(row["peak_gain"])
        peak_va = float(row["peak_va"])
        cross_va = float(row["cross_va"])
        a0 = float(row["usable_a"])
        b0 = float(row["usable_b"])
        width = float(row["usable_width"])
        linear = row["linear"]
        cross_str = f"{cross_va: .3g}" if not np.isnan(cross_va) else "nan"
        print(
            f"  VGG={vgg:.1f}V VSS={vss:.1f}V: peak gain = {peak_gain:.3g} @ V_A = {peak_va: .3g} V, "
            f"V_B=V_A at V_A = {cross_str} V"
        )
        if np.isfinite(a0) and np.isfinite(b0) and width > 0.0:
            center = 0.5 * (a0 + b0)
            print(
                f"  VGG={vgg:.1f}V VSS={vss:.1f}V: usable V_A = [{a0:+.3g}, {b0:+.3g}] V, "
                f"width = {width:.3g} V, center = {center:+.3g} V"
            )
        else:
            print(f"  VGG={vgg:.1f}V VSS={vss:.1f}V: usable V_A = [nan, nan] V, width = 0 V, center = nan V")

    linear_ranked = sorted(
        all_summary,
        key=lambda r: float(r["linear"]["width"]) if bool(r["linear"]["found"]) else 0.0,
        reverse=True,
    )
    print("Linear region by (VGG, VSS), sorted by width:")
    for idx, row in enumerate(linear_ranked, start=1):
        vgg = float(row["vgg"])
        vss = float(row["vss"])
        linear = row["linear"]
        tag = "  [RECOMMENDED]" if idx == 1 else ""
        if bool(linear["found"]):
            print(
                f"  {idx}. VGG={vgg:.1f}V VSS={vss:.1f}V: V_A in [{float(linear['v_a_start']):+.3g}, "
                f"{float(linear['v_a_end']):+.3g}], width={float(linear['width']):.3g}V, "
                f"gain={float(linear['avg_gain']):.3g}, RMSE={float(linear['linear_fit_rmse']):.3g}V{tag}"
            )
        else:
            print(f"  {idx}. VGG={vgg:.1f}V VSS={vss:.1f}V: no linear region{tag}")

    for row in all_summary:
        linear = row["linear"]
        if not bool(linear["found"]):
            row["combined_a"] = float("nan")
            row["combined_b"] = float("nan")
            row["combined_width"] = 0.0
            continue
        la = float(linear["v_a_start"])
        lb = float(linear["v_a_end"])
        ua = float(row["usable_a"])
        ub = float(row["usable_b"])
        if np.isfinite(ua) and np.isfinite(ub):
            ca = max(la, ua)
            cb = min(lb, ub)
            cw = max(0.0, cb - ca)
        else:
            ca = float("nan")
            cb = float("nan")
            cw = 0.0
        row["combined_a"] = ca
        row["combined_b"] = cb
        row["combined_width"] = cw

    ranked = sorted(all_summary, key=lambda r: float(r["combined_width"]), reverse=True)
    print("Best operating points (linear AND in V_th_eff window):")
    for rank, row in enumerate(ranked[:5], start=1):
        vss = float(row["vss"])
        vgg = float(row["vgg"])
        a0 = float(row["combined_a"])
        b0 = float(row["combined_b"])
        width = float(row["combined_width"])
        tag = "  [BEST]" if rank == 1 else ""
        if np.isfinite(a0) and np.isfinite(b0) and width > 0.0:
            print(
                f"  {rank}. VGG={vgg:.1f}V VSS={vss:.1f}V: combined V_A range = "
                f"[{a0:+.3g}, {b0:+.3g}], width={width:.3g}V{tag}"
            )
        else:
            print(f"  {rank}. VGG={vgg:.1f}V VSS={vss:.1f}V: no combined range{tag}")

    plt.show()


if __name__ == "__main__":
    main()
