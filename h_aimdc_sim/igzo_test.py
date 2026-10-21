"""
IGZO TFT transfer IV: load igzo_measure.csv, fit log10(|Ids|) vs Vgs, plot vs measurement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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


def composite_model(v, c0, k, vth, p0, p1, p2, p3):
    s = 0.5 * (1.0 + np.tanh(k * (v - vth)))
    on_poly = p0 + p1 * v + p2 * v**2 + p3 * v**3
    return c0 * (1 - s) + on_poly * s


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).replace("#", "").strip().lower() for c in df.columns]
    required = {"vgs", "vds", "ids"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns after normalization: {sorted(missing)}")
    return df


def fit_iv(vgs: np.ndarray, ids: np.ndarray):
    """
    Fit log10(|Ids| + 1e-15) vs Vgs. Returns (popt, rmse_log) or None on failure.
    popt order: c0, k, vth, p0, p1, p2, p3
    """
    x = np.asarray(vgs, dtype=float)
    y_log = np.log10(np.abs(np.asarray(ids, dtype=float)) + 1e-15)

    initial_guess = [np.min(y_log), 5.0, 0.0, np.max(y_log), 1.0, 0.0, 0.0]
    bounds = (
        [-20.0, 0.1, -5.0, -20.0, -10.0, -10.0, -10.0],
        [-5.0, 50.0, 5.0, 5.0, 10.0, 10.0, 10.0],
    )

    try:
        popt, _ = curve_fit(
            composite_model, x, y_log, p0=initial_guess, bounds=bounds, maxfev=10000
        )
    except Exception as e:
        print(f"[fit_iv] curve_fit failed: {e}")
        return None

    y_hat = composite_model(x, *popt)
    rmse_log = float(np.sqrt(np.mean((y_log - y_hat) ** 2)))
    return popt, rmse_log


def plot_iv(df: pd.DataFrame, fits: dict[float, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(fits), 1)))

    for i, vds in enumerate(sorted(fits.keys())):
        color = cmap[i % len(cmap)]
        sub = df[np.isclose(df["vds"].astype(float), float(vds))].sort_values("vgs")
        vgs_m = sub["vgs"].values
        ids_m = sub["ids"].values
        ax.semilogy(
            vgs_m,
            np.abs(ids_m),
            "o",
            color=color,
            markersize=5,
            markerfacecolor=color,
            label=f"measured (Vds = {vds:g} V)",
        )

        popt = fits[vds]
        v_line = np.linspace(vgs_m.min(), vgs_m.max(), 500)
        y_line = composite_model(v_line, *popt)
        ids_line = np.maximum(10.0**y_line - 1e-15, 1e-30)
        ax.semilogy(
            v_line,
            ids_line,
            "-",
            color="red",
            linewidth=2,
            label=f"fitting (Vds = {vds:g} V)",
        )

    ax.set_xlabel("Vgs (V)")
    ax.set_ylabel("|Ids| (A)")
    ax.set_title("IGZO TFT Transfer Characteristics - Measured vs Fitted")
    ax.grid(True, which="both", ls="--")
    ax.legend()
    fig.tight_layout()
    fig.canvas.draw()
    place_figure_window(fig)
    fig.savefig("iv_fit.png", dpi=300)
    place_figure_window(fig)
    plt.show()


def _print_fit_report(vds: float, popt: np.ndarray, rmse_log: float) -> None:
    c0, k, vth, p0, p1, p2, p3 = popt
    print(f"\n--- Vds = {vds:g} V ---")
    print(f"  c0  = {c0:.6g}")
    print(f"  k   = {k:.6g}")
    print(f"  vth = {vth:.6g}")
    print(f"  p0  = {p0:.6g}")
    print(f"  p1  = {p1:.6g}")
    print(f"  p2  = {p2:.6g}")
    print(f"  p3  = {p3:.6g}")
    print(f"  RMSE (log10 |Ids| domain) = {rmse_log:.6g}")


def main() -> None:
    path = "igzo_measure.csv"
    df = load_data(path)

    fits: dict[float, np.ndarray] = {}
    for vds, group in df.groupby("vds", sort=True):
        vds_f = float(vds)
        g = group.sort_values("vgs")
        result = fit_iv(g["vgs"].values, g["ids"].values)
        if result is None:
            print(f"[main] Skipping Vds = {vds_f:g} V (fit failed).")
            continue
        popt, rmse_log = result
        fits[vds_f] = popt
        _print_fit_report(vds_f, popt, rmse_log)

    if not fits:
        print("[main] No successful fits; plot skipped.")
        return

    plot_iv(df, fits)


if __name__ == "__main__":
    main()
