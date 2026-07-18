"""
v2 extensions for AIMDC single-bit Xyce driver.
Keeps v1 API available via aimdc_sim_core and adds Vds-aware B-source helpers.
v3 adds power-law-with-saturation IGZO fit from igzo_powerlaw_model.

NOTE (V_DS factor): step_pos * vds / sqrt(vds^2 + vds_ref^2) is an
UNVALIDATED HEURISTIC — never fitted to multi-V_DS data; with a single measured
V_DS point it still cannot be. Treat vds_ref as a scale constant only.
"""
from __future__ import annotations

import os
import json
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from aimdc_sim_core import *  # noqa: F401,F403
from aimdc_sim_core import fit_igzo_model, save_fit_json
from igzo_powerlaw_model import POWERLAW_EPS, fit_igzo_powerlaw

# v2 model: Vds dependence parameters
DEFAULT_VDS_REF: float = 0.5  # V; heuristic scale (see module note)
DEFAULT_K_STEP: float = 50.0  # 1/V; sharpness of V_ds<0 cutoff
XYCE_POWER_EPS: float = POWERLAW_EPS


def xyce_b_source_expression_v2(
    popt: NDArray[np.floating[Any]],
    *,
    vds_ref: float = DEFAULT_VDS_REF,
    k_step: float = DEFAULT_K_STEP,
) -> str:
    """Public wrapper: Xyce B-source RHS from fitted θ, including V_ds dependence.
    Local nodes for the subcircuit must be (d, g, s) so that V(d,s) is well-defined."""
    return _build_xyce_b_source_expression_v2(popt, vds_ref=vds_ref, k_step=k_step)


def _build_xyce_b_source_expression_v2(
    popt: NDArray[np.floating[Any]],
    *,
    vds_ref: float,
    k_step: float,
) -> str:
    c0, k, vth, p0, p1, p2, p3 = popt
    # Vgs-only part (same as v1)
    s_expr = f"0.5 * (1.0 + TANH({k:e} * (V(g,s) - ({vth:e}))))"
    on_poly_expr = (
        f"({p0:e} + {p1:e}*V(g,s) + {p2:e}*V(g,s)*V(g,s) + {p3:e}*V(g,s)*V(g,s)*V(g,s))"
    )
    poly_expr = f"({c0:e}) * (1.0 - ({s_expr})) + ({on_poly_expr}) * ({s_expr})"
    ids_vgs = f"EXP(2.302585093 * ({poly_expr}))"
    # V_ds factor (new in v2)
    step_pos = f"0.5 * (1.0 + TANH({k_step:e} * V(d,s)))"
    sat = f"V(d,s) / SQRT(V(d,s)*V(d,s) + ({vds_ref:e})*({vds_ref:e}))"
    factor = f"({step_pos}) * ({sat})"
    return f"({ids_vgs}) * ({factor})"


def fit_igzo_model_v2(
    csv_path: str,
    *,
    vds_ref: float = DEFAULT_VDS_REF,
    k_step: float = DEFAULT_K_STEP,
) -> tuple[NDArray[np.float64], str]:
    """Same fit as v1 (Vgs-only on log10|Ids|), but returns the v2 expression."""
    popt, _v1_expr = fit_igzo_model(csv_path)
    expr_v2 = _build_xyce_b_source_expression_v2(popt, vds_ref=vds_ref, k_step=k_step)
    return popt, expr_v2


def generate_spice_deck_v2(
    expression: str,
    v_sl_val: float,
    params: Mapping[str, Any],
    *,
    workdir: str | None = None,
) -> None:
    """Write sim_distance.sp under workdir (default: cwd), using v2 expression."""
    wd = os.path.abspath(workdir or os.getcwd())
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


def _build_xyce_b_source_expression_v3(
    popt: NDArray[np.floating[Any]],
    *,
    vds_ref: float,
    k_step: float,
    sub_cap: float = 5.0,
) -> str:
    # popt[2] = ss_v_per_dec (subthreshold swing); NOT the VSS supply rail.
    I_off, V_th, ss_v_per_dec, I0_sub, I_sat_max, V_sat_scale, gamma, delta = popt

    vgs = "V(g,s)"
    vds = "V(d,s)"

    vgst_diff = f"({vgs} - ({V_th:e}))"
    vgst_smooth = (
        f"0.5 * ({vgst_diff} + SQRT({vgst_diff}*{vgst_diff} + ({delta:e})*({delta:e})))"
    )

    # Smooth subthreshold clamp (differentiable; replaces MIN)
    sub_arg = f"({sub_cap:e}) * TANH(({vgst_diff} / ({ss_v_per_dec:e})) / ({sub_cap:e}))"
    i_sub = f"({I0_sub:e}) * EXP({sub_arg})"

    tanh_arg = f"TANH(({vgst_smooth}) / ({V_sat_scale:e}))"
    i_above = (
        f"({I_sat_max:e}) * EXP(({gamma:e}) * LOG(({tanh_arg}) + ({XYCE_POWER_EPS:e})))"
    )

    i_off = f"({I_off:e})"
    ids_vgs = f"({i_off} + {i_sub} + {i_above})"

    # UNVALIDATED HEURISTIC V_DS factor (see module docstring).
    step_pos = f"0.5 * (1.0 + TANH({k_step:e} * {vds}))"
    sat = f"{vds} / SQRT({vds}*{vds} + ({vds_ref:e})*({vds_ref:e}))"
    factor = f"({step_pos}) * ({sat})"

    return f"({ids_vgs}) * ({factor})"


def xyce_b_source_expression_v3(
    popt: NDArray[np.floating[Any]],
    *,
    vds_ref: float = DEFAULT_VDS_REF,
    k_step: float = DEFAULT_K_STEP,
) -> str:
    return _build_xyce_b_source_expression_v3(popt, vds_ref=vds_ref, k_step=k_step)


def fit_igzo_model_v3(
    csv_path: str,
    *,
    vds_ref: float = DEFAULT_VDS_REF,
    k_step: float = DEFAULT_K_STEP,
    vds: float | None = None,
) -> tuple[NDArray[np.float64], str]:
    """Fit the new power-law-with-saturation model and return (popt, xyce_expr)."""
    popt, info = fit_igzo_powerlaw(csv_path, vds=vds)
    popt_f = np.asarray(popt, dtype=np.float64).ravel()
    if popt_f.size != 8:
        raise ValueError(
            f"fit_igzo_powerlaw returned {popt_f.size} parameters; expected 8 for v3."
        )
    vds_fit = float(info["vds_fit"])
    if not np.isclose(abs(vds_fit), abs(vds_ref), rtol=0.0, atol=1e-6):
        raise ValueError(
            f"vds_ref={vds_ref} disagrees with measured vds_fit={vds_fit}. "
            "Pass vds_ref=abs(vds_fit) or an explicit vds=... to fit_igzo_powerlaw. "
            "Note: the V_DS factor itself remains an unvalidated heuristic."
        )
    expr = _build_xyce_b_source_expression_v3(popt_f, vds_ref=vds_ref, k_step=k_step)
    return popt_f, expr


def generate_spice_deck_v3(
    expression: str,
    v_sl_val: float,
    params: Mapping[str, Any],
    *,
    workdir: str | None = None,
) -> None:
    """Write sim_distance.sp under workdir (default: cwd), using v3 expression."""
    wd = os.path.abspath(workdir or os.getcwd())
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


def save_fit_json_v3(
    csv_path: str,
    out_path: str,
    *,
    circuit_params: Mapping[str, Any] | None = None,
) -> tuple[NDArray[np.float64], dict]:
    """v3 (8-param power-law) fit and save to JSON.

    Same JSON schema as v1: popt length is self-describing (8 vs 7), so existing
    load_fit_json works unchanged. Caller passes circuit_params to embed NVM
    operating point (VDD, VGG, VSS, V_MBL, V_ML, v_min, v_max) in the JSON.
    """
    popt, info = fit_igzo_powerlaw(csv_path)
    save_fit_json(
        popt,
        out_path,
        source_csv=os.path.basename(csv_path),
        vds_fit=float(info["vds_fit"]),
        fit_rmse_log10=float(info["rmse_log10"]),
        circuit_params=circuit_params,
    )
    return popt, info


def _cli_main_v3(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="aimdc_sim_core_v3")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # fit_v3
    p3 = sub.add_parser("fit_v3", help="Fit 8-param IGZO power-law and save JSON.")
    p3.add_argument("--csv", required=True, help="Path to igzo_measure.csv")
    p3.add_argument(
        "--out", required=True, help="Output JSON path (e.g. igzo_fit_v3.json)"
    )
    p3.add_argument(
        "--vmin", type=float, default=-1.0, help="Embedding scaler v_min (default -1.0)"
    )
    p3.add_argument(
        "--vmax", type=float, default=1.0, help="Embedding scaler v_max (default +1.0)"
    )
    p3.add_argument("--vdd", type=float, default=3.0)
    p3.add_argument("--vgg", type=float, default=3.0)
    p3.add_argument("--vss", type=float, default=-3.0)
    p3.add_argument("--v-mbl", dest="v_mbl", type=float, default=0.5)
    p3.add_argument("--v-ml", dest="v_ml", type=float, default=0.0)
    p3.add_argument(
        "--operating-point",
        dest="operating_point",
        type=str,
        default=None,
        help="Path to NVM operating-point JSON (from fefet_cell_read_v3.py "
             "--save-operating-point). When set, circuit_params are loaded from "
             "the JSON. Explicit CLI flags (--vdd, --vgg, ...) override the JSON.",
    )

    # fit_v1 (re-export of existing CLI for convenience)
    p1 = sub.add_parser("fit_v1", help="Fit 7-param composite model and save JSON.")
    p1.add_argument("--csv", required=True)
    p1.add_argument("--out", required=True)

    args = parser.parse_args(argv)

    if args.cmd == "fit_v3":
        import sys as _sys

        op_circuit: dict[str, float] = {}
        op_validation: dict[str, Any] | None = None
        op_data: dict[str, Any] | None = None
        op_path: str | None = None
        if args.operating_point:
            op_path = os.path.abspath(args.operating_point)
            if not os.path.isfile(op_path):
                raise FileNotFoundError(f"Operating-point JSON not found: {op_path}")
            with open(op_path, encoding="utf-8") as f:
                loaded = json.load(f)
            if not isinstance(loaded, dict):
                raise ValueError(f"Operating-point JSON must be an object: {op_path}")
            op_data = loaded
            if int(op_data.get("schema_version", 0)) != 1:
                raise ValueError(
                    "Unsupported operating-point schema_version: "
                    f"{op_data.get('schema_version')!r} (expected 1)"
                )
            cp_in = op_data.get("circuit_params") or {}
            if not isinstance(cp_in, dict):
                raise ValueError("operating-point circuit_params must be an object")
            op_circuit = {str(k): float(v) for k, v in cp_in.items()}
            raw_validation = op_data.get("validation")
            if isinstance(raw_validation, dict):
                op_validation = raw_validation
            print(f"[fit_v3] Loaded operating-point JSON: {op_path}")
            if op_validation:
                slope = op_validation.get("match_slope")
                r2 = op_validation.get("match_r_squared")
                if slope is not None and r2 is not None:
                    print(f"[fit_v3]   validation slope = {float(slope):g}, R^2 = {float(r2):g}")

        attr_to_json = {
            "vdd": "VDD",
            "vgg": "VGG",
            "vss": "VSS",
            "v_mbl": "V_MBL",
            "v_ml": "V_ML",
            "vmin": "v_min",
            "vmax": "v_max",
        }
        json_to_attr = {v: k for k, v in attr_to_json.items()}
        flag_to_attr = {
            "--vdd": "vdd",
            "--vgg": "vgg",
            "--vss": "vss",
            "--v-mbl": "v_mbl",
            "--v-ml": "v_ml",
            "--vmin": "vmin",
            "--vmax": "vmax",
        }
        cli_tokens = list(argv) if argv is not None else list(_sys.argv[1:])
        explicit_flags: set[str] = set()
        for flag, attr in flag_to_attr.items():
            if flag in cli_tokens or any(tok.startswith(f"{flag}=") for tok in cli_tokens):
                explicit_flags.add(attr)

        def _resolve(name_cli: str, cli_value: float, json_key: str) -> float:
            if op_circuit and (name_cli not in explicit_flags) and (json_key in op_circuit):
                return float(op_circuit[json_key])
            return float(cli_value)

        circuit_params: dict[str, Any] = {
            "VDD": _resolve("vdd", args.vdd, "VDD"),
            "VGG": _resolve("vgg", args.vgg, "VGG"),
            "VSS": _resolve("vss", args.vss, "VSS"),
            "V_MBL": _resolve("v_mbl", args.v_mbl, "V_MBL"),
            "V_ML": _resolve("v_ml", args.v_ml, "V_ML"),
            "v_min": _resolve("vmin", args.vmin, "v_min"),
            "v_max": _resolve("vmax", args.vmax, "v_max"),
        }

        if op_data and op_validation and "validated_at" in op_data:
            circuit_params["_validated_at"] = str(op_data.get("validated_at", ""))
            if op_path is not None:
                circuit_params["_op_source"] = os.path.basename(op_path)

        print("[fit_v3] Embedding circuit_params into fit JSON:")
        for k, v in circuit_params.items():
            attr = json_to_attr.get(k)
            if attr is None:
                src = "JSON"
            elif op_circuit and k in op_circuit and (attr not in explicit_flags):
                src = "JSON"
            else:
                src = "CLI" if (attr in explicit_flags or not op_circuit) else "default"
            if isinstance(v, (int, float, np.floating, np.integer)):
                print(f"  {k:12s} = {float(v):7.3f}  [{src}]")
            else:
                print(f"  {k:12s} = {v!s}  [{src}]")

        csv_abs = os.path.abspath(args.csv)
        popt, info = save_fit_json_v3(
            csv_abs,
            os.path.abspath(args.out),
            circuit_params=circuit_params,
        )
        print(f"Wrote {args.out!r}")
        print(f"  rmse_log10 = {float(info['rmse_log10']):.6f}")
        print(f"  vds_fit    = {float(info['vds_fit'])}")
        print(f"  popt (8)   = {[f'{float(x):.6g}' for x in popt]}")
    elif args.cmd == "fit_v1":
        from aimdc_sim_core import compute_fit_residual_rmse

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

    _cli_main_v3(sys.argv[1:])
