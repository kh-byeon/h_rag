from __future__ import annotations

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from igzo_powerlaw_model import evaluate_model, fit_igzo_powerlaw

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


def _load_measurement(csv_path: str) -> tuple[np.ndarray, np.ndarray, float]:
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
    vds0 = float(seen[0])
    meas = df[np.isclose(df["vds"].astype(float), vds0)].sort_values("vgs")
    x = meas["vgs"].values.astype(float)
    y = np.abs(meas["ids"].values.astype(float))
    return x, y, vds0


def main() -> None:
    csv_path = os.path.join(SCRIPT_DIR, "igzo_measure.csv")
    csv_path = os.path.abspath(csv_path)
    popt, info = fit_igzo_powerlaw(csv_path)
    names = list(info["param_names"]) if isinstance(info["param_names"], list) else []

    print(f"csv: {csv_path}")
    print(f"vds_fit: {float(info['vds_fit']):.6g}")
    for i, name in enumerate(names):
        print(f"{name} = {float(popt[i]):.6g}")
    print(f"rmse_log10 = {float(info['rmse_log10']):.6g}")
    print(f"n_points = {int(float(info['n_points']))}")

    v_meas, i_meas, _vds = _load_measurement(csv_path)

    v_dense_meas = np.linspace(float(np.min(v_meas)), float(np.max(v_meas)), 500)
    y_dense_meas = evaluate_model(v_dense_meas, popt)
    i_dense_meas = np.maximum(10.0**y_dense_meas, 1e-30)

    v_dense_ext = np.linspace(-4.0, 6.0, 500)
    y_dense_ext = evaluate_model(v_dense_ext, popt)
    i_dense_ext = np.maximum(10.0**y_dense_ext, 1e-30)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.semilogy(v_meas, i_meas, "o", color="tab:blue", markersize=5, label="Measured")
    ax1.semilogy(v_dense_meas, i_dense_meas, "-", color="red", linewidth=2, label="Fitted")
    ax1.set_title("IGZO TFT Fit (Measurement Range)")
    ax1.set_xlabel("V_gs (V)")
    ax1.set_ylabel("|Ids| (A)")
    ax1.grid(True, which="both", ls="--")
    ax1.legend()

    ax2.semilogy(v_meas, i_meas, "o", color="tab:blue", markersize=5, label="Measured")
    ax2.semilogy(v_dense_ext, i_dense_ext, "-", color="red", linewidth=2, label="Fitted")
    ax2.axvline(-3.0, color="0.5", ls="--")
    ax2.axvline(3.0, color="0.5", ls="--")
    y_anno = float(np.sqrt(np.min(i_dense_ext) * np.max(i_dense_ext)))
    ax2.text(0.0, y_anno, "Measured range", ha="center", va="center")
    ax2.set_title("Fit Extrapolation Check")
    ax2.set_xlabel("V_gs (V)")
    ax2.set_ylabel("|Ids| (A)")
    ax2.grid(True, which="both", ls="--")
    ax2.legend()

    fig.tight_layout()
    fig.canvas.draw()
    place_figure_window(fig)
    out_path = os.path.join(SCRIPT_DIR, "igzo_powerlaw_fit.png")
    fig.savefig(out_path, dpi=300)
    print(f"Saved figure: {out_path}")
    place_figure_window(fig)
    plt.show()


if __name__ == "__main__":
    main()
