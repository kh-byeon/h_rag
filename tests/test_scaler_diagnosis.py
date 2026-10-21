"""Tests for AIMDC scaler diagnosis (offline vs PQ-recon)."""
from __future__ import annotations

import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aimdc_scaler_diagnosis import (  # noqa: E402
    build_summary_payload,
    load_offline_scaler_inputs,
    run_diagnosis,
)


def _write_min_offline(
    d: Path,
    *,
    with_opq: bool,
    n_raw: int = 400,
    n_post: int = 400,
    m: int = 4,
    k: int = 16,
    sub: int = 8,
) -> None:
    rng = np.random.default_rng(42)
    dim = m * sub
    raw = rng.standard_normal((n_raw, dim)).astype(np.float32)
    np.save(d / "raw_embeddings.npy", raw)
    if with_opq:
        q, _r = np.linalg.qr(rng.standard_normal((dim, dim)).astype(np.float64))
        np.save(d / "opq_rotation.npy", q.astype(np.float32))
    lut = rng.standard_normal((m, k, sub)).astype(np.float32)
    np.save(d / "pq_codebooks.npy", lut)
    codes = rng.integers(0, k, size=(n_post, m), dtype=np.uint8)
    posting: dict[int, list] = {0: []}
    for i in range(n_post):
        posting[0].append({"pq_code": codes[i].copy(), "id": i})
    with open(d / "posting_lists.pkl", "wb") as f:
        pickle.dump(posting, f)


class TestScalerDiagnosis(unittest.TestCase):
    def test_synthetic_identical_single_scaler(self) -> None:
        rng = np.random.default_rng(0)
        q = rng.standard_normal((800, 12)).astype(np.float32)
        s = build_summary_payload(
            q_pre=q.copy(),
            q_post=q.copy(),
            pq_recon=q.copy(),
            has_opq_rotation=True,
        )
        self.assertEqual(s["recommendation"], "single_scaler")

    def test_synthetic_low_std_dual_scaler(self) -> None:
        rng = np.random.default_rng(1)
        q = rng.standard_normal((5000, 24)).astype(np.float32)
        p = rng.standard_normal((5000, 24)).astype(np.float32) * 0.3
        s = build_summary_payload(
            q_pre=q.copy(),
            q_post=q.copy(),
            pq_recon=p,
            has_opq_rotation=True,
        )
        self.assertEqual(s["recommendation"], "dual_scaler")

    def test_no_opq_rotation_flag(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _write_min_offline(d, with_opq=False)
            data = load_offline_scaler_inputs(str(d))
            self.assertIsNone(data["opq_rotation"])
            out = d / "diag_out"
            s = run_diagnosis(str(d), str(out), sample_size=200, rng_seed=0)
            self.assertFalse(s["has_opq_rotation"])
            self.assertTrue((out / "summary.json").is_file())

    def test_missing_lut_no_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            rng = np.random.default_rng(0)
            np.save(d / "raw_embeddings.npy", rng.standard_normal((50, 8)).astype(np.float32))
            with open(d / "posting_lists.pkl", "wb") as f:
                pickle.dump({0: [{"pq_code": np.zeros(4, dtype=np.uint8)}]}, f)
            out = d / "should_stay_empty"
            with self.assertRaises(FileNotFoundError):
                run_diagnosis(str(d), str(out), sample_size=10, rng_seed=0)
            self.assertFalse(out.exists(), "diagnosis must not create out_dir when inputs are invalid")

    def test_summary_json_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _write_min_offline(d, with_opq=True)
            o1 = Path(td) / "o1"
            o2 = Path(td) / "o2"
            run_diagnosis(str(d), str(o1), sample_size=150, rng_seed=7)
            run_diagnosis(str(d), str(o2), sample_size=150, rng_seed=7)
            b1 = (o1 / "summary.json").read_bytes()
            b2 = (o2 / "summary.json").read_bytes()
            self.assertEqual(b1, b2)
            obj = json.loads(b1.decode("utf-8"))
            self.assertIn(obj["recommendation"], ("single_scaler", "dual_scaler", "single_scaler_with_union_pool"))


if __name__ == "__main__":
    unittest.main()
