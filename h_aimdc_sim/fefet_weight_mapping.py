"""Weight -> (two-FeFET program-state) mapping for the 2-FeFET distance cell.

Combines the two measurement sources:
  * SMU (B1500) reference curve  -> accurate current levels + true subthreshold
    shape  (sets notch DEPTH; this is I_ref).
  * DAC sweep Vth calibration     -> threshold-shift vs program-code, dv-vs-code
    (sets how a requested shift is realised as a program code; range ~2.255 V).

Cell (V_FOLD = 0 fold, source = ML_F, gate window from SMU):
    I_ML(V_SL) = I_ref(V_SL - V_ML - s1) + I_ref(-V_SL - V_ML - s2)

with s1, s2 the programmed threshold shifts of MF1 / MF2 (s>0 = erased / higher
Vth / curve shifted right).  From d/dV_SL = 0:

    notch POSITION  V_SL* = (s1 - s2) / 2                 <- differential  (= weight w)
    notch DEPTH     I_ML(V_SL*) = 2 * I_ref(-(s1+s2)/2 - V_ML)   <- common  (= erase C)

So a target weight w at fixed common erase C maps to
    s1 = C + w ,   s2 = C - w .
C must be large enough that the centre overdrive -(C) - V_ML sits in the device
off-region (=> deep notch); with this device and V_ML = 0.5, C >~ 0.8 V.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

import numpy as np

from h_aimdc_sim import fefet_smu_io as smu

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass(frozen=True)
class WeightMap:
    v_g: np.ndarray          # reference gate grid (SMU)
    log_i: np.ndarray        # log10 |Id| of the reference state
    v_valid: tuple[float, float]
    i_floor: float
    v_ml: float
    common_erase: float      # C
    dac_slope: float         # dv = slope*code + intercept  (V per code)
    dac_intercept: float

    # ---- device current with a threshold shift s (rigid-shift, DAC-validated) ----
    def _idev(self, vgs: np.ndarray, shift: float) -> np.ndarray:
        return 10.0 ** np.interp(vgs - shift, self.v_g, self.log_i,
                                 left=self.log_i[0], right=self.log_i[-1])

    def shifts_for_weight(self, w: float) -> tuple[float, float]:
        """Target notch position w -> (MF1 shift, MF2 shift)."""
        return self.common_erase + w, self.common_erase - w

    def shift_to_code(self, shift: float) -> float:
        """Program code that realises a threshold shift (via DAC dv-vs-code)."""
        return (shift - self.dac_intercept) / self.dac_slope

    def cell_current(self, v_sl: np.ndarray, w: float) -> np.ndarray:
        s1, s2 = self.shifts_for_weight(w)
        return (self._idev(v_sl - self.v_ml, s1)
                + self._idev(-v_sl - self.v_ml, s2))

    def notch(self, w: float, span: float = 3.0, step: float = 0.01):
        v_sl = np.arange(w - span, w + span + 1e-9, step)
        i = self.cell_current(v_sl, w)
        k = int(np.argmin(i))
        return v_sl, i, v_sl[k], float(i[k])

    def max_weight(self, min_shift: float = 0.1) -> float:
        """Largest |w| keeping the lesser-erased device at/above min_shift."""
        return self.common_erase - min_shift


def load_weight_map(
    smu_ref_csv: str | None = None,
    *,
    dac_calibration_json: str | None = None,
    v_ml: float = 0.5,
    common_erase: float = 0.9,
    smu_block: int = 0,
) -> WeightMap:
    if smu_ref_csv is None:
        cands = glob.glob(os.path.join(SCRIPT_DIR, "**", "*vth_1.0*.csv"),
                          recursive=True)
        if not cands:
            raise FileNotFoundError("Provide smu_ref_csv (B1500 vth_1.0 file)")
        smu_ref_csv = cands[0]
    s = smu.load_smu_idvg(smu_ref_csv, block=smu_block)
    log_i = np.log10(s.i_d)
    i_floor = float(np.median(s.i_d[s.v_g < -3.5]))

    dac_slope, dac_intercept = 1.478e-3, -0.170  # documented fallback
    if dac_calibration_json is None:
        cj = os.path.join(SCRIPT_DIR, "fefet_vth_calibration.json")
        dac_calibration_json = cj if os.path.isfile(cj) else None
    if dac_calibration_json and os.path.isfile(dac_calibration_json):
        d = json.load(open(dac_calibration_json))
        dac_slope = float(d["model_linear"]["slope_v_per_code"])
        dac_intercept = float(d["model_linear"]["intercept_v"])

    return WeightMap(
        v_g=s.v_g, log_i=log_i,
        v_valid=(float(s.v_g.min()), float(s.v_g.max())),
        i_floor=i_floor, v_ml=float(v_ml), common_erase=float(common_erase),
        dac_slope=dac_slope, dac_intercept=dac_intercept,
    )


def main(argv: list[str] | None = None) -> None:
    import argparse

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(
        description="Plot weight->2-FeFET deep notches at multiple positions.")
    ap.add_argument("--smu-ref", default=None,
                    help="B1500 SMU CSV (vth_1.0); auto-detected if omitted")
    ap.add_argument("--weights", default="-0.7,-0.35,0,0.35,0.7",
                    help="comma-separated target notch positions (use --weights=..)")
    ap.add_argument("--v-ml", type=float, default=0.5)
    ap.add_argument("--common-erase", type=float, default=0.9, help="depth knob C")
    ap.add_argument("--dac-json", default=None, help="fefet_vth_calibration.json")
    ap.add_argument("--smu-block", type=int, default=0)
    ap.add_argument("--out", default="weight_mapping_notches.png")
    args = ap.parse_args(argv)

    wm = load_weight_map(args.smu_ref, dac_calibration_json=args.dac_json,
                         v_ml=args.v_ml, common_erase=args.common_erase,
                         smu_block=args.smu_block)
    weights = [float(x) for x in args.weights.replace(" ", "").split(",") if x]
    print(f"V_ML={wm.v_ml}  C(common erase)={wm.common_erase}  "
          f"v_valid={wm.v_valid}  i_floor={wm.i_floor:.2e}  max|w|~{wm.max_weight():.2f}")
    print(f"{'w':>6} | {'s1':>5} {'s2':>5} | {'code1':>6} {'code2':>6} | "
          f"{'notch':>9} {'pos':>6}")

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    for w in weights:
        v_sl, i, pos, mn = wm.notch(w)
        s1, s2 = wm.shifts_for_weight(w)
        line, = ax.semilogy(v_sl, i, lw=1.8,
                            label=f"w={w:+.2f} (notch {mn:.0e} @ {pos:.2f})")
        ax.plot(pos, mn, "v", color=line.get_color(), ms=8)
        print(f"{w:>6.2f} | {s1:>5.2f} {s2:>5.2f} | "
              f"{wm.shift_to_code(s1):>6.0f} {wm.shift_to_code(s2):>6.0f} | "
              f"{mn:>9.2e} {pos:>6.2f}")

    ax.axhline(2 * wm.i_floor, color="gray", ls=":", lw=1, label="2*i_floor (floor)")
    ax.set_title(f"Weight->2-FeFET mapping: deep notch at multiple positions "
                 f"(V_ML={wm.v_ml}, C={wm.common_erase})")
    ax.set_xlabel("V_SL (V)  [= embedding-dim input]")
    ax.set_ylabel("|I_ML| (A)")
    ax.set_ylim(1e-13, 1e-6)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper center")
    fig.tight_layout()
    out = args.out if os.path.isabs(args.out) else os.path.join(
        SCRIPT_DIR, "outputs_fefet_v4", args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print("saved:", out)


__all__ = ["WeightMap", "load_weight_map"]


if __name__ == "__main__":
    main()
