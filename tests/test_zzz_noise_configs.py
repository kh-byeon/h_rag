"""Sanity: the committed noise config JSONs parse and enable only the expected fields.

Named ``test_zzz_*`` so this module runs after ``test_zz_aimdc_scope`` (which
installs ``aimdc_device_distance`` stubs). ``setUp`` restores the real package
for assertions that need the full ``NoiseModel`` / ``DeviceDistance``.
"""
from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG_DIR = ROOT / "noise_configs"


def _ensure_real_aimdc_device_distance() -> None:
    """Undo ``test_zz_aimdc_scope`` stub so imports resolve to the real package."""
    mod = sys.modules.get("aimdc_device_distance")
    need_reload = (
        mod is None
        or not hasattr(mod, "DeviceDistance")
        or not hasattr(mod.DeviceDistance, "pairwise")
    )
    if need_reload:
        sys.modules.pop("aimdc_device_distance", None)
        importlib.import_module("aimdc_device_distance")


class TestNoiseConfigs(unittest.TestCase):
    def setUp(self) -> None:
        _ensure_real_aimdc_device_distance()

    def test_all_configs_parse(self) -> None:
        from aimdc_device_distance import noise_model_from_json_path

        for p in sorted(CONFIG_DIR.glob("*.json")):
            n = noise_model_from_json_path(str(p))
            self.assertIsNotNone(n, msg=f"parse failed: {p.name}")

    def test_clean_disables_everything(self) -> None:
        from aimdc_device_distance import noise_model_from_json_path

        n = noise_model_from_json_path(str(CONFIG_DIR / "clean.json"))
        self.assertFalse(n.enabled())
        self.assertEqual(n.ktc_C, 0.0)
        self.assertEqual(n.sigma_alpha, 0.0)
        self.assertEqual(n.sigma_vth, 0.0)
        self.assertEqual(n.sigma_log_mu, 0.0)
        self.assertIsNone(n.retention_tau)
        self.assertEqual(n.flicker_sigma, 0.0)
        self.assertFalse(n.shot_enable)
        self.assertEqual(n.rtn_sigma, 0.0)
        self.assertIsNone(n.adc_bits)

    def test_d2d_enables_only_d2d(self) -> None:
        from aimdc_device_distance import noise_model_from_json_path

        n = noise_model_from_json_path(str(CONFIG_DIR / "d2d.json"))
        self.assertTrue(n.enabled())
        self.assertGreater(n.sigma_vth, 0.0)
        self.assertGreater(n.sigma_log_mu, 0.0)
        self.assertEqual(n.ktc_C, 0.0)
        self.assertEqual(n.sigma_alpha, 0.0)
        self.assertIsNone(n.retention_tau)
        self.assertEqual(n.flicker_sigma, 0.0)
        self.assertFalse(n.shot_enable)
        self.assertEqual(n.rtn_sigma, 0.0)
        self.assertIsNone(n.adc_bits)

    def test_d2d_retention_enables_two(self) -> None:
        from aimdc_device_distance import noise_model_from_json_path

        n = noise_model_from_json_path(str(CONFIG_DIR / "d2d_retention.json"))
        self.assertTrue(n.enabled())
        self.assertGreater(n.sigma_vth, 0.0)
        self.assertGreater(n.sigma_log_mu, 0.0)
        self.assertIsNotNone(n.retention_tau)
        self.assertGreater(float(n.retention_tau or 0.0), 0.0)
        self.assertGreater(n.t_hold, 0.0)
        self.assertIsNone(n.adc_bits)
        self.assertEqual(n.ktc_C, 0.0)
        self.assertEqual(n.sigma_alpha, 0.0)
        self.assertFalse(n.shot_enable)

    def test_full_enables_d2d_retention_adc(self) -> None:
        from aimdc_device_distance import noise_model_from_json_path

        n = noise_model_from_json_path(str(CONFIG_DIR / "full.json"))
        self.assertTrue(n.enabled())
        self.assertGreater(n.sigma_vth, 0.0)
        self.assertGreater(n.sigma_log_mu, 0.0)
        self.assertIsNotNone(n.retention_tau)
        self.assertIsNotNone(n.adc_bits)
        self.assertGreater(int(n.adc_bits or 0), 0)
        self.assertGreater(n.adc_i_fullscale, 0.0)
        self.assertEqual(n.ktc_C, 0.0)
        self.assertEqual(n.sigma_alpha, 0.0)
        self.assertFalse(n.shot_enable)
        self.assertEqual(n.rtn_sigma, 0.0)
        self.assertEqual(n.flicker_sigma, 0.0)

    def test_determinism_per_config(self) -> None:
        """Same config + same seed → identical distance."""
        from aimdc_device_distance import DeviceDistance, noise_model_from_json_path

        fit = ROOT / "h_aimdc_sim" / "igzo_fit.json"
        if not fit.is_file():
            raise unittest.SkipTest("igzo_fit.json missing")
        rng = np.random.default_rng(0)
        emb = rng.standard_normal((50, 16)).astype(np.float64)
        q = rng.standard_normal(16)
        c = rng.standard_normal(16)
        for cfg in ["d2d.json", "d2d_retention.json", "full.json"]:
            n = noise_model_from_json_path(str(CONFIG_DIR / cfg))
            vals: list[float] = []
            for _ in range(3):
                dd = DeviceDistance(str(fit), scaler_path=None, noise=n)
                dd.fit_scaler(emb)
                vals.append(dd.distance(q, c))
            self.assertAlmostEqual(vals[0], vals[1], places=12, msg=f"{cfg}: run1 vs run2 diverged")
            self.assertAlmostEqual(vals[1], vals[2], places=12, msg=f"{cfg}: run2 vs run3 diverged")


if __name__ == "__main__":
    unittest.main()
