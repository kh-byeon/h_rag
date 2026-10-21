"""Regression: calibrate must not crash on pre-existing empty tmp scaler path.

Loads after ``test_zz_aimdc_scope`` so ``setUp`` can restore the real
``aimdc_device_distance`` without breaking stub-based tests.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ensure_real_aimdc_device_distance() -> None:
    """Undo ``test_zz_aimdc_scope`` stub so we exercise the real ``DeviceDistance``."""
    mod = sys.modules.get("aimdc_device_distance")
    need_reload = (
        mod is None
        or not hasattr(mod, "DeviceDistance")
        or not hasattr(mod.DeviceDistance, "pairwise")
    )
    if need_reload:
        sys.modules.pop("aimdc_device_distance", None)
        importlib.import_module("aimdc_device_distance")


def _load_calibrate_module():
    path = ROOT / "scripts" / "calibrate_scale_k.py"
    spec = importlib.util.spec_from_file_location("calibrate_scale_k", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestCalibrateTmpfileFix(unittest.TestCase):
    def setUp(self) -> None:
        _ensure_real_aimdc_device_distance()

    def test_empty_scaler_file_does_not_crash_device_init(self) -> None:
        """DeviceDistance must tolerate an existing zero-byte scaler file."""
        from aimdc_device_distance import DeviceDistance, NoiseModel

        fit = ROOT / "h_aimdc_sim" / "igzo_fit.json"
        if not fit.is_file():
            raise unittest.SkipTest("igzo_fit.json missing")
        with tempfile.TemporaryDirectory() as td:
            empty_scaler = os.path.join(td, "empty.scaler.json")
            with open(empty_scaler, "w", encoding="utf-8") as f:
                pass
            self.assertEqual(os.path.getsize(empty_scaler), 0)
            dd = DeviceDistance(
                str(fit),
                scaler_path=empty_scaler,
                noise=NoiseModel(),
            )
            self.assertIsNone(dd._x_lo)
            self.assertIsNone(dd._x_hi)

    def test_calibrate_end_to_end_synthetic(self) -> None:
        """Full calibrate_scale_k.run_calibration path must succeed end-to-end."""
        cal = _load_calibrate_module()
        rng = np.random.default_rng(0)
        D = 4
        m, ksz, sub = 2, 8, 2
        assert m * sub == D
        with tempfile.TemporaryDirectory() as td:
            raw = rng.standard_normal((40, D)).astype(np.float32)
            np.save(os.path.join(td, "raw_embeddings.npy"), raw)
            lut = rng.standard_normal((m, ksz, sub)).astype(np.float32)
            np.save(os.path.join(td, "pq_codebooks.npy"), lut)
            posting = {
                0: [{"pq_code": np.array([1, 2], dtype=np.int64)} for _ in range(50)]
            }
            with open(os.path.join(td, "posting_lists.pkl"), "wb") as f:
                pickle.dump(posting, f)
            with open(os.path.join(td, "metadata_db.pkl"), "wb") as f:
                pickle.dump({}, f)
            np.save(
                os.path.join(td, "centroids.npy"),
                np.zeros((1, D), dtype=np.float32),
            )

            fit = ROOT / "h_aimdc_sim" / "igzo_fit.json"
            if not fit.is_file():
                raise unittest.SkipTest("igzo_fit.json missing")

            payload = cal.run_calibration(
                td,
                str(fit),
                n_queries=8,
                n_candidates=16,
                seed=0,
            )
        self.assertIn("k", payload)
        self.assertTrue(np.isfinite(float(payload["k"])))
        self.assertGreater(float(payload["k"]), 0.0)


if __name__ == "__main__":
    unittest.main()
