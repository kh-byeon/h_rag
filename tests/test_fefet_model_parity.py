"""Parity / regression tests for the measured 2-FeFET AIMDC path."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MEASURE_DIR = os.environ.get(
    "FEFET_MEASURE_DIR",
    r"D:\OneDrive\FeFET_meas_res\sweep_",
)


def _have_measures() -> bool:
    return os.path.isdir(MEASURE_DIR) and any(
        n.endswith(".xlsx") for n in os.listdir(MEASURE_DIR)
    )


class TestFefetModelParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not _have_measures():
            raise unittest.SkipTest(f"FeFET measure dir not found: {MEASURE_DIR}")
        from h_aimdc_sim.fefet_measure_io import load_sweep_dir
        from h_aimdc_sim.fefet_ref_model import fit_fefet_ref_model
        from h_aimdc_sim.fefet_vth_calibration import fit_vth_calibration

        cls.sweeps = load_sweep_dir(MEASURE_DIR)
        cls.ref = next(s for s in cls.sweeps if s.code == 300)
        cls.lut = fit_fefet_ref_model(cls.ref, backend="lut")
        cls.pl = fit_fefet_ref_model(cls.ref, backend="powerlaw3")
        cls.cal = fit_vth_calibration(cls.sweeps, ref_code=300)

    def test_python_spice_parity_powerlaw3(self) -> None:
        """FefetRefModel.current vs evaluated Xyce-like EXP/LOG expression."""
        from h_aimdc_sim.fefet_ref_model import POWERLAW_EPS

        params = self.pl.params
        assert params is not None
        A, Vth, n = params["A"], params["Vth"], params["n"]
        i_floor = self.pl.i_floor
        v = np.linspace(self.pl.v_valid[0] + 0.2, self.pl.v_valid[1] - 0.2, 61)
        i_py = self.pl.current(v, dv=0.0)
        # Mirror xyce_expression powerlaw3 (no V_DS factor)
        od = np.maximum(v - Vth, 0.0)
        i_xyce = A * np.exp(n * np.log(od + POWERLAW_EPS)) + i_floor
        rel = np.max(np.abs(i_py - i_xyce) / np.maximum(np.abs(i_py), 1e-30))
        self.assertLess(rel, 1e-6)

    def test_lut_reproduces_ref_above_knee(self) -> None:
        vk = float(self.lut.meta["v_knee"])
        m = self.ref.v_g >= vk
        i_pred = self.lut.current(self.ref.v_g, dv=0.0)
        y_m = np.log10(np.maximum(self.ref.i_d, 1e-30))
        y_p = np.log10(np.maximum(i_pred, 1e-30))
        rmse = float(np.sqrt(np.mean((y_p[m] - y_m[m]) ** 2)))
        self.assertLessEqual(rmse, 0.02)

    def test_powerlaw3_n_range(self) -> None:
        n = float(self.pl.params["n"])  # type: ignore[index]
        self.assertGreaterEqual(n, 1.2)
        self.assertLessEqual(n, 1.6)

    def test_reduction_minkowski_p(self) -> None:
        from aimdc_nvm_device_distance import NvmDeviceDistance

        p = 1.15
        tmp = tempfile.mkdtemp()
        fit = os.path.join(tmp, "fit.json")
        with open(fit, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "schema_version": 2,
                    "model": "fefet_ref_lut_v1",
                    "backend": "lut",
                    "popt": [],
                    "power_law_p": p,
                    "reduction_mode": "linear_sum",
                    "circuit_params": {"v_min": -1.0, "v_max": 1.0},
                },
                f,
            )
        dd = NvmDeviceDistance(fit, reduce="linear_sum", p_root=p, use_log=False)
        dd._x_lo, dd._x_hi = -1.0, 1.0
        dd.v_min, dd.v_max = -1.0, 1.0

        def synth(v_sl, v_prg):
            return np.abs(np.asarray(v_sl, dtype=np.float64) - np.asarray(v_prg, dtype=np.float64)) ** p

        dd._bit_current = synth  # type: ignore[method-assign]
        rng = np.random.default_rng(0)
        q = rng.normal(0, 0.3, size=(8,))
        c = rng.normal(0, 0.3, size=(5, 8))
        dd._x_lo = float(min(q.min(), c.min()) - 0.1)
        dd._x_hi = float(max(q.max(), c.max()) + 0.1)

        def scale(x):
            t = (x - dd._x_lo) / (dd._x_hi - dd._x_lo)
            return dd.v_min + (dd.v_max - dd.v_min) * t

        true = np.sum(np.abs(scale(q)[None, :] - scale(c)) ** p, axis=-1) ** (1.0 / p)
        got = dd.pairwise(q, c)
        rel = float(np.max(np.abs(got - true) / np.maximum(true, 1e-30)))
        self.assertLess(rel, 1e-6)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            dd2 = NvmDeviceDistance(fit, reduce="legacy_sum", use_log=True)
        dd2._x_lo, dd2._x_hi = dd._x_lo, dd._x_hi
        dd2.v_min, dd2.v_max = dd.v_min, dd.v_max
        dd2._bit_current = synth  # type: ignore[method-assign]
        got_leg = dd2.pairwise(q, c)
        rel_leg = float(np.max(np.abs(got_leg - true) / np.maximum(true, 1e-30)))
        self.assertGreater(rel_leg, 1e-3)

    def test_calibration_regression(self) -> None:
        by = {int(s["code"]): s for s in self.cal["states"]}
        self.assertAlmostEqual(by[1700]["dv"], 2.25, delta=0.05)
        self.assertAlmostEqual(self.cal["model_linear"]["r_squared"], 0.907, delta=0.01)

    def test_rigid_shift_residual_bound(self) -> None:
        for s in self.cal["states"]:
            self.assertLessEqual(float(s["residual_rmse_dec"]), 0.15)

    def test_validity_guard(self) -> None:
        from h_aimdc_sim.fefet_cell_read_v4 import assert_gate_window_valid

        with self.assertRaises(ValueError):
            assert_gate_window_valid(
                v_fold=10.0,
                w=0.0,
                vmin=-1.0,
                vmax=1.0,
                v_valid=self.lut.v_valid,
            )

    def test_negative_distance_section_penalty(self) -> None:
        """Same-heading penalty must move negative distances toward -inf (closer)."""
        weight = 1.5
        for d in (3.0, -3.0):
            delta = abs(d) * (1.0 - 1.0 / weight)
            d2 = d - delta
            self.assertLess(d2, d)
        # Explicit negative case: old divide would make -3/1.5=-2 (farther).
        d = -3.0
        old_wrong = d / weight
        delta = abs(d) * (1.0 - 1.0 / weight)
        new = d - delta
        self.assertLess(new, d)
        self.assertLess(new, old_wrong)

    def test_v_sl_match_vs_w_powerlaw3(self) -> None:
        from h_aimdc_sim.fefet_cell_read_v4 import run_analytical_w_sweep

        w_list = [0.0, 0.25, 0.5, 0.75, 1.0]
        results = run_analytical_w_sweep(
            self.pl,
            w_list=w_list,
            v_fold=0.0,
            v_mbl=-0.5,
            v_ml=0.0,
            vsweep=(-1.0, 1.0, 0.02),
        )
        w_arr = np.array([r["w"] for r in results])
        vsm = np.array([r["v_sl_match"] for r in results])
        slope, intercept = np.polyfit(w_arr, vsm, 1)
        yhat = slope * w_arr + intercept
        r2 = 1.0 - float(np.sum((vsm - yhat) ** 2) / np.sum((vsm - vsm.mean()) ** 2))
        self.assertGreaterEqual(slope, 0.97)
        self.assertLessEqual(slope, 1.03)
        self.assertGreater(r2, 0.99)


if __name__ == "__main__":
    unittest.main()
