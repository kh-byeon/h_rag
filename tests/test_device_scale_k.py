"""B1 scale_k: load, MMR score, τ, dumps, calibrate sanity."""
from __future__ import annotations

import importlib.util
import io
import json
import os
import pickle
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def _load_calibrate_module():
    path = ROOT / "scripts" / "calibrate_scale_k.py"
    spec = importlib.util.spec_from_file_location("calibrate_scale_k", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _minimal_hw(D: int) -> dict:
    return {
        "l2_accelerator_1_centroids": np.zeros((1, D), dtype=np.float32),
        "l2_accelerator_2_pq_codes": {},
        "lut_memory": np.zeros((1, 4, D), dtype=np.float32),
        "metadata_sram": {},
        "df_dict": {},
        "total_docs": 0,
    }


def _make_cands(n: int, D: int) -> list:
    out = []
    for i in range(n):
        emb = np.zeros(D, dtype=np.float32)
        emb[0] = float(i) * 0.1 + 0.5
        emb[1] = 1.0
        out.append(
            {
                "chunk_id": str(i),
                "embedding": emb,
                "metadata": {"section_heading": "H", "doc_id": "0"},
            }
        )
    return out


class _FlatNegDevice:
    noise = None

    def __init__(self) -> None:
        from aimdc_device_distance import NoiseModel

        self.noise = NoiseModel()

    def pair_matrix(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        del a
        return np.full((1, max(b.shape[0], 1)), -6500.0, dtype=np.float64)


class TestDeviceScaleK(unittest.TestCase):
    def test_k_load_roundtrip(self) -> None:
        pytest.importorskip("aimdc_device_distance")
        from online_accelerator_device import L2RAGOnlinePipeline, _configure_device_distance

        fit = ROOT / "h_aimdc_sim" / "igzo_fit.json"
        if not fit.is_file():
            raise unittest.SkipTest("igzo_fit.json missing")
        with tempfile.TemporaryDirectory() as td:
            kpath = os.path.join(td, "k.json")
            with open(kpath, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "k": 1.234e-3,
                        "n_queries": 10,
                        "n_candidates": 64,
                        "ratio_stats": {"p5": 1e-4, "p50": 1.2e-3, "p95": 2e-3},
                        "schema_version": 1,
                    },
                    f,
                )
            scaler = os.path.join(td, "s.json")
            lut = np.zeros((2, 4, 4), dtype=np.float32)
            posting = {0: [{"pq_code": np.array([0, 0], dtype=np.int64)}]}
            hw = {"lut_memory": lut, "l2_accelerator_2_pq_codes": posting}
            dd, k = _configure_device_distance(
                hw,
                device_fit_path=str(fit),
                device_scaler_path=scaler,
                device_reduce="sum",
                device_use_log=True,
                device_v_ml=0.0,
                noise_config=None,
                noise_seed=None,
                device_scale_k_path=kpath,
            )
        self.assertIsNotNone(dd)
        self.assertAlmostEqual(float(k or 0.0), 1.234e-3, places=12)
        pl = L2RAGOnlinePipeline(
            _minimal_hw(8),
            top_k=3,
            device_distance=dd,
            device_scale_k=k,
            verbose=False,
        )
        self.assertAlmostEqual(float(pl.device_scale_k or 0.0), 1.234e-3, places=12)

    def test_score_formula_with_k_dump(self) -> None:
        pytest.importorskip("aimdc_device_distance")
        from online_accelerator_device import L2RAGOnlinePipeline

        D = 8
        hw = _minimal_hw(D)
        q = np.zeros(D, dtype=np.float32)
        q[0] = 1.0
        cands = _make_cands(4, D)
        lam = 0.5
        k = 1e-4
        pl = L2RAGOnlinePipeline(
            hw,
            top_k=4,
            gamma=0.25,
            mmr_lambda=lam,
            tau_dup=-1e9,
            verbose=False,
            dump_embeddings=True,
            device_distance=_FlatNegDevice(),
            device_scale_k=k,
        )
        pl._dump_reset()
        pl.step5_on_the_fly_mmr(q, cands, s_buffer=[], buffer_top_k=4, gamma=0.25)
        for rec in pl._dump_records["mmr_steps"]:
            if not rec.get("selected") or rec.get("mmr_score") is None:
                continue
            dqc = float(rec["dist_q_c_raw"])
            mdu = float(rec["min_dist_c_s_used"])
            exp = -lam * dqc + (1.0 - lam) * (k * mdu)
            self.assertAlmostEqual(float(rec["mmr_score"]), exp, places=9)
            self.assertAlmostEqual(float(rec["device_scale_k"]), k, places=12)
            self.assertAlmostEqual(float(rec["min_dist_c_s_scaled"]), k * mdu, places=9)

    def test_tau_with_k(self) -> None:
        pytest.importorskip("aimdc_device_distance")
        from online_accelerator_device import L2RAGOnlinePipeline

        D = 4
        hw = _minimal_hw(D)
        q = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        cands = _make_cands(5, D)
        k = 1e-4

        pl_pass = L2RAGOnlinePipeline(
            hw,
            top_k=3,
            gamma=0.34,
            mmr_lambda=0.85,
            tau_dup=-1.0,
            verbose=False,
            device_distance=_FlatNegDevice(),
            device_scale_k=k,
        )
        _, st_pass = pl_pass.step5_on_the_fly_mmr(
            q, cands, s_buffer=[], buffer_top_k=3, gamma=0.34
        )
        self.assertEqual(st_pass["early_dedup_count"], 0)

        pl_skip = L2RAGOnlinePipeline(
            hw,
            top_k=3,
            gamma=0.34,
            mmr_lambda=0.85,
            tau_dup=0.0,
            verbose=False,
            device_distance=_FlatNegDevice(),
            device_scale_k=k,
        )
        _, st_skip = pl_skip.step5_on_the_fly_mmr(
            q, cands, s_buffer=[], buffer_top_k=3, gamma=0.34
        )
        self.assertGreater(st_skip["early_dedup_count"], 0)

    def test_k_none_raw_scale_matches(self) -> None:
        pytest.importorskip("aimdc_device_distance")
        from online_accelerator_device import L2RAGOnlinePipeline

        D = 8
        hw = _minimal_hw(D)
        q = np.zeros(D, dtype=np.float32)
        q[0] = 1.0
        cands = _make_cands(4, D)
        lam = 0.6
        dd = _FlatNegDevice()
        p1 = L2RAGOnlinePipeline(
            hw,
            top_k=4,
            gamma=0.25,
            mmr_lambda=lam,
            tau_dup=-1e9,
            verbose=False,
            dump_embeddings=True,
            device_distance=dd,
            device_scale_k=None,
        )
        p2 = L2RAGOnlinePipeline(
            hw,
            top_k=4,
            gamma=0.25,
            mmr_lambda=lam,
            tau_dup=-1e9,
            verbose=False,
            dump_embeddings=True,
            device_distance=dd,
        )
        p1._dump_reset()
        p2._dump_reset()
        p1.step5_on_the_fly_mmr(q, cands, s_buffer=[], buffer_top_k=4, gamma=0.25)
        p2.step5_on_the_fly_mmr(q, cands, s_buffer=[], buffer_top_k=4, gamma=0.25)
        s1 = [r for r in p1._dump_records["mmr_steps"] if r.get("selected") and r.get("mmr_score") is not None]
        s2 = [r for r in p2._dump_records["mmr_steps"] if r.get("selected") and r.get("mmr_score") is not None]
        self.assertEqual(len(s1), len(s2))
        for a, b in zip(s1, s2):
            self.assertAlmostEqual(float(a["mmr_score"]), float(b["mmr_score"]), places=9)
            self.assertIsNone(a.get("device_scale_k"))
            self.assertIsNone(a.get("min_dist_c_s_scaled"))

    def test_override_beats_path(self) -> None:
        pytest.importorskip("aimdc_device_distance")
        from online_accelerator_device import _configure_device_distance

        fit = ROOT / "h_aimdc_sim" / "igzo_fit.json"
        if not fit.is_file():
            raise unittest.SkipTest("igzo_fit.json missing")
        with tempfile.TemporaryDirectory() as td:
            kpath = os.path.join(td, "k.json")
            with open(kpath, "w", encoding="utf-8") as f:
                json.dump({"k": 0.01, "ratio_stats": {}, "schema_version": 1}, f)
            scaler = os.path.join(td, "s.json")
            lut = np.zeros((2, 4, 4), dtype=np.float32)
            posting = {0: [{"pq_code": np.array([0, 0], dtype=np.int64)}]}
            hw = {"lut_memory": lut, "l2_accelerator_2_pq_codes": posting}
            buf = io.StringIO()
            with redirect_stdout(buf):
                _dd, k = _configure_device_distance(
                    hw,
                    device_fit_path=str(fit),
                    device_scaler_path=scaler,
                    device_reduce="sum",
                    device_use_log=True,
                    device_v_ml=0.0,
                    noise_config=None,
                    noise_seed=None,
                    device_scale_k_path=kpath,
                    device_scale_k_override=2.5e-4,
                )
        self.assertAlmostEqual(float(k or 0.0), 2.5e-4, places=12)
        self.assertNotIn("loaded scale_k from", buf.getvalue())

    def test_dump_schema_keys(self) -> None:
        pytest.importorskip("aimdc_device_distance")
        from online_accelerator_device import L2RAGOnlinePipeline

        D = 4
        hw = _minimal_hw(D)
        q = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        cands = _make_cands(4, D)
        pl = L2RAGOnlinePipeline(
            hw,
            top_k=3,
            gamma=0.34,
            mmr_lambda=0.85,
            tau_dup=-1e9,
            verbose=False,
            dump_embeddings=True,
            device_distance=None,
        )
        pl._dump_reset()
        pl.step5_on_the_fly_mmr(q, cands, s_buffer=[], buffer_top_k=3, gamma=0.34)
        for rec in pl._dump_records["mmr_steps"]:
            self.assertIn("device_scale_k", rec)
            self.assertIn("min_dist_c_s_scaled", rec)
            self.assertIsNone(rec["device_scale_k"])
            self.assertIsNone(rec["min_dist_c_s_scaled"])

    def test_calibrate_synthetic_k(self) -> None:
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
            np.save(os.path.join(td, "centroids.npy"), np.zeros((1, D), dtype=np.float32))

            class PropDev:
                factor = 4000.0

                def pairwise(self, q: np.ndarray, c: np.ndarray) -> np.ndarray:
                    sw = np.linalg.norm(c - q.reshape(1, -1), axis=1)
                    return (sw / self.factor).astype(np.float64)

            payload = cal.run_calibration(
                td,
                os.path.join(td, "dummy_fit.json"),
                n_queries=24,
                n_candidates=32,
                seed=1,
                device=PropDev(),
            )
        k_meas = float(payload["k"])
        self.assertGreater(k_meas, 0.0)
        self.assertLess(abs(k_meas - 4000.0) / 4000.0, 0.05)
