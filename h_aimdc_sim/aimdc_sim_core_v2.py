"""
v2 extensions for AIMDC single-bit Xyce driver.
Keeps v1 API available via aimdc_sim_core and adds Vds-aware B-source helpers.
"""
from __future__ import annotations

import os
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from aimdc_sim_core import *  # noqa: F401,F403
from aimdc_sim_core import fit_igzo_model

# v2 model: Vds dependence parameters
DEFAULT_VDS_REF: float = 0.5  # V; matches measurement CSV Vds
DEFAULT_K_STEP: float = 50.0  # 1/V; sharpness of V_ds<0 cutoff


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
