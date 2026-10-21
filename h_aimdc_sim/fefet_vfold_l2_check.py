"""Quick check: does the 2-FeFET cell (V_FOLD set) produce an L2/L1-shaped
match-line current response, using the measured code=300 reference curve?

Runs the parity-verified Python twin (identical to the Xyce deck to 1e-6) and
ALSO emits per-weight .sp decks so the same config can be run in Docker/Xyce.

Config: V_FOLD = 0 (ideal complement is exact here), V_MBL = 0.5, V_ML tunes
the match-point bias into subthreshold so the V-curve has real depth.
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from h_aimdc_sim import fefet_measure_io as io
from h_aimdc_sim import fefet_ref_model as rm
from h_aimdc_sim.fefet_cell_read_v4 import generate_fefet_cell_deck_v4

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "outputs_fefet_v4")
os.makedirs(OUT_DIR, exist_ok=True)


def main() -> None:
    meas = os.environ["FEFET_MEASURE_DIR"]
    sweeps = io.load_sweep_dir(meas)
    ref = next(s for s in sweeps if s.code == 300)
    model = rm.fit_fefet_ref_model(ref, backend="lut", source_files=[
        os.path.basename(s.source_path) for s in sweeps])

    cal = json.load(open(os.path.join(SCRIPT_DIR, "fefet_vth_calibration.json")))
    dv = {s["code"]: s["dv"] for s in cal["states"]}

    V_FOLD = 0.0
    V_MBL = 0.5
    V_ML = 2.5           # biases the match point into subthreshold
    codes = [300, 500, 700, 900, 1100]
    expr = model.xyce_expression(v_ds_factor=False)

    def i_ml(v_sl: np.ndarray, w: float) -> np.ndarray:
        u = v_sl - w
        return (model.current(u - V_ML, dv=0.0, allow_extrapolation=True)
                + model.current(-u - V_ML, dv=0.0, allow_extrapolation=True))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))
    summary = []
    for code in codes:
        w = dv[code]
        vsl = np.arange(w - 2.0, w + 2.0 + 1e-9, 0.02)
        y = i_ml(vsl, w)
        imin, imax = float(y.min()), float(y.max())
        # emit an .sp deck for local Xyce (V_SL sweep centred on this weight)
        generate_fefet_cell_deck_v4(
            expr, v_fold=V_FOLD, w=w, v_mbl=V_MBL, v_ml=V_ML,
            vsweep=(float(w - 2.0), float(w + 2.0), 0.02),
            deck_name=f"vfoldcheck_code{code}.sp",
            csv_name=f"vfoldcheck_code{code}.csv",
            workdir=OUT_DIR,
        )
        axL.plot(vsl, y * 1e6, label=f"code {code} (w={w:.2f})")
        axL.axvline(w, color="gray", ls=":", lw=0.6)
        # exponent of rising flank about the minimum
        u = vsl - w
        sel = (u > 0.3) & (u <= 2.0)
        exc = y[sel] - imin
        pos = exc > 0
        p = float(np.polyfit(np.log(np.abs(u[sel][pos])), np.log(exc[pos]), 1)[0])
        summary.append((code, w, imin, imax, imax / imin, p))
        # shape panel: normalised excess current vs |V_SL - w|
        uu = np.abs(u[pos & sel]) if False else np.abs(u)
        exc_all = y - imin
        m = exc_all > 0
        axR.plot(np.abs(u[m]), exc_all[m] / (imax - imin), lw=1.2,
                 label=f"code {code}, p≈{p:.2f}")

    # L1 / L2 reference shapes on the shape panel
    uref = np.linspace(0.05, 2.0, 100)
    axR.plot(uref, (uref / 2.0) ** 1.0, "k--", lw=1.0, label="ideal L1 (p=1)")
    axR.plot(uref, (uref / 2.0) ** 2.0, "k:", lw=1.2, label="ideal L2 (p=2)")

    axL.set_xlabel("V_SL  (V)")
    axL.set_ylabel("I_ML  (µA)")
    axL.set_title("Match-line current vs input  (V_FOLD=0, V_MBL=0.5, V_ML=2.5)")
    axL.legend(fontsize=8)
    axL.grid(alpha=0.3)

    axR.set_xlabel("|V_SL - w|  (V)")
    axR.set_ylabel("(I_ML - I_min) / (I_max - I_min)")
    axR.set_title("Response shape vs L1 / L2")
    axR.legend(fontsize=8)
    axR.grid(alpha=0.3)

    fig.tight_layout()
    png = os.path.join(OUT_DIR, "vfold_l2_check.png")
    fig.savefig(png, dpi=130)
    print("saved plot:", png)
    print()
    print("%5s %6s %11s %11s %7s %6s" % ("code", "w", "I_min", "I_max", "DR", "p"))
    for code, w, imin, imax, dr, p in summary:
        print("%5d %6.3f %11.3e %11.3e %7.2f %6.2f" % (code, w, imin, imax, dr, p))


if __name__ == "__main__":
    main()
