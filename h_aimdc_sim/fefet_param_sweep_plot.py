"""Sweep one cell parameter (V_FOLD / V_ML / V_MBL), run each config in
Docker/Xyce, and overlay the match-line current I_ML(V_SL) on one plot.

Falls back to the parity-verified Python twin (identical to Xyce to 1e-6) when
Docker is unavailable, so the same command runs here and on your machine.

IMPORTANT — sweeping V_FOLD: in the present ideal-source cell the gate voltages
are  V(G1)=V_SL-w  and  V(G2)=w-V_SL,  from which V_FOLD cancels exactly. So a
V_FOLD sweep produces overlapping curves (verified to ~1e-22 A). V_FOLD is a
common-mode knob that only keeps V_SL and V_FOLD-V_SL inside the measured gate
window; it does not change the differential distance response. The parameter
that actually reshapes the V-curve is V_ML (--sweep v_ml).

Examples
--------
  python -m h_aimdc_sim.fefet_param_sweep_plot --sweep v_fold \
      --values " -6,-3,0,2" --w 1.125 --out vfold_sweep.png
  python -m h_aimdc_sim.fefet_param_sweep_plot --sweep v_ml \
      --values "0,1.5,2.0,2.5,3.0" --w 1.125 --out vml_sweep.png
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from h_aimdc_sim import aimdc_sim_core
from h_aimdc_sim.fefet_measure_io import load_sweep_dir
from h_aimdc_sim.fefet_ref_model import fit_fefet_ref_model, load_fefet_ref_fit
from h_aimdc_sim.fefet_cell_read_v4 import (
    analytical_i_ml,
    extract_cell_iv_v4,
    generate_fefet_cell_deck_v4,
    run_docker_sim,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "outputs_fefet_v4")


def _parse_values(s: str) -> list[float]:
    return [float(x) for x in s.replace(" ", "").split(",") if x]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", choices=["v_fold", "v_ml", "v_mbl", "w"], default="v_fold")
    ap.add_argument("--values", required=True,
                    help="comma-separated sweep values (for --sweep w: the stored weights)")
    ap.add_argument("--w", type=float, default=-1,
                    help="programmed weight dv (V); ignored when --sweep w")
    ap.add_argument("--v-fold", type=float, default=-1.0)
    ap.add_argument("--v-ml", type=float, default=0.5)
    ap.add_argument("--v-mbl", type=float, default=0.0)
    ap.add_argument("--vmin", type=float, default=None, help="default w-3.0")
    ap.add_argument("--vmax", type=float, default=None, help="default w+3.0")
    ap.add_argument("--vstep", type=float, default=0.02)
    ap.add_argument("--fit-json", default=None)
    ap.add_argument("--smu-ref", default=None,
                    help="B1500 SMU CSV; build I_ref (accurate current levels) from it")
    ap.add_argument("--smu-block", type=int, default=0,
                    help="SMU stress-time block index (0 = fresh)")
    ap.add_argument("--measure-dir", default=os.environ.get("FEFET_MEASURE_DIR"))
    ap.add_argument("--analytical", action="store_true", help="force Python twin")
    ap.add_argument("--ymin", type=float, default=None,
                    help="y-axis lower limit in A (default: model i_floor)")
    ap.add_argument("--ymax", type=float, default=None,
                    help="y-axis upper limit in A (default: from data)")
    ap.add_argument("--out", default="param_sweep.png")
    args = ap.parse_args(argv)

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.smu_ref:
        from h_aimdc_sim.fefet_smu_io import load_smu_idvg, smu_to_sweep
        ref = smu_to_sweep(load_smu_idvg(args.smu_ref, block=args.smu_block), code=300)
        model = fit_fefet_ref_model(ref, backend="lut")
        print(f"reference: SMU {os.path.basename(args.smu_ref)} block={args.smu_block} "
              f"(i_floor={model.i_floor:.2e} A, v_valid={model.v_valid})")
    elif args.fit_json:
        model = load_fefet_ref_fit(args.fit_json)
    elif args.measure_dir:
        ref = next(s for s in load_sweep_dir(args.measure_dir) if s.code == 300)
        model = fit_fefet_ref_model(ref, backend="lut")
        print(f"reference: DAC integrator sweep code=300 "
              f"(i_floor={model.i_floor:.2e} A) — NOTE: current levels are "
              f"instrument-limited; use --smu-ref for accurate current.")
    else:
        raise SystemExit("Provide --smu-ref, --fit-json, or --measure-dir")

    expr = model.xyce_expression(v_ds_factor=False)
    values = _parse_values(args.values)
    sweep_w = args.sweep == "w"
    # Fixed global V_SL window so the moving notch is visible on a shared axis.
    if sweep_w:
        lo_default, hi_default = min(values) - 2.0, max(values) + 2.0
    else:
        lo_default, hi_default = args.w - 2.0, args.w + 2.0
    vmin = args.vmin if args.vmin is not None else lo_default
    vmax = args.vmax if args.vmax is not None else hi_default
    vsweep = (float(vmin), float(vmax), float(args.vstep))

    use_docker = False
    if not args.analytical:
        try:
            use_docker = aimdc_sim_core.resolve_docker_bin() is not None
        except Exception:
            use_docker = False
    engine = "Xyce (Docker)" if use_docker else "Python twin (Xyce-parity 1e-6)"
    print(f"engine: {engine}")

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    i_all: list[np.ndarray] = []
    for val in values:
        params = {"v_fold": args.v_fold, "v_ml": args.v_ml, "v_mbl": args.v_mbl}
        w_i = float(val) if sweep_w else float(args.w)
        if not sweep_w:
            params[args.sweep] = val
        tag = f"{args.sweep}{val:+.2f}".replace(".", "p")
        deck = f"sweep_{tag}.sp"
        csv = f"sweep_{tag}.csv"
        generate_fefet_cell_deck_v4(
            expr, v_fold=params["v_fold"], w=w_i, v_mbl=params["v_mbl"],
            v_ml=params["v_ml"], vsweep=vsweep, deck_name=deck, csv_name=csv,
            workdir=OUT_DIR,
        )
        if use_docker:
            run_docker_sim(workdir=OUT_DIR, deck_name=deck, csv_name=csv)
            v_sl, _b, _g1, _g2, i_ml, _i = extract_cell_iv_v4(os.path.join(OUT_DIR, csv))
            i_ml = np.abs(i_ml)
        else:
            v_sl = np.arange(vsweep[0], vsweep[1] + vsweep[2] * 0.5, vsweep[2])
            # Mirror Xyce TABLE(): clamp (extrapolate) at the measured-window ends
            # instead of raising, so a wide shared V_SL axis is allowed.
            u = v_sl - w_i
            vgs1 = u - params["v_ml"]
            vgs2 = -u - params["v_ml"]
            i_ml = (model.current(vgs1, allow_extrapolation=True)
                    + model.current(vgs2, allow_extrapolation=True))
        i_all.append(np.asarray(i_ml, dtype=float))
        lbl = f"w={val:g}" if sweep_w else f"{args.sweep}={val:g}"
        line, = ax.plot(v_sl, i_ml, lw=1.4, label=lbl)
        # mark the notch (minimum) so the shift is explicit
        j = int(np.argmin(i_ml))
        ax.plot(v_sl[j], i_ml[j], "v", color=line.get_color(), ms=7)
        if sweep_w:
            ax.axvline(float(val), color=line.get_color(), ls=":", lw=0.6)

    ax.set_yscale("log")
    ax.set_xlabel("V_SL  (V)")
    ax.set_ylabel("|I_ML|  (A)")
    stacked = np.concatenate(i_all) if i_all else np.array([model.i_floor])
    ymin = args.ymin if args.ymin is not None else float(max(model.i_floor, 1e-15))
    ymax = args.ymax if args.ymax is not None else float(np.nanmax(stacked) * 1.5)
    if ymax <= ymin:
        ymax = ymin * 10.0
    ax.set_ylim(ymin, ymax)
    fixed = {k: v for k, v in
             (("V_FOLD", args.v_fold), ("V_ML", args.v_ml), ("V_MBL", args.v_mbl))
             if k.lower() != args.sweep}
    if not sweep_w:
        fixed_str = ", ".join(f"{k}={v:g}" for k, v in fixed.items()) + f", w={args.w:g}"
    else:
        fixed_str = ", ".join(f"{k}={v:g}" for k, v in fixed.items())
    ax.set_title(f"I_ML vs V_SL — sweep {args.sweep}  [{engine}]\n"
                 f"fixed: " + fixed_str)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = args.out if os.path.isabs(args.out) else os.path.join(OUT_DIR, args.out)
    fig.savefig(out, dpi=130)
    print("saved:", out)


if __name__ == "__main__":
    main(sys.argv[1:])
