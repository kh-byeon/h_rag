"""AIMDC scope: diversity-only device path, no query-on-AIMDC, PQ-recon scaler.

Loads after ``test_device_distance`` / ``test_mmr_scoring_units`` so the
``aimdc_device_distance`` stub here does not break those imports.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _install_aimdc_device_stubs() -> None:
    """Minimal ``aimdc_device_distance`` so ``online_accelerator_device`` imports."""

    @dataclass
    class NoiseModel:
        rng_seed: Optional[int] = None
        ktc_C: float = 0.0
        sigma_alpha: float = 0.0
        sigma_vth: float = 0.0
        retention_tau: float = 0.0
        adc_bits: int = 0

        def enabled(self) -> bool:
            return False

    def noise_model_from_json_path(_path: str) -> NoiseModel:
        return NoiseModel()

    class DeviceDistance:
        v_min: float = -1.0
        v_max: float = 1.0

        def __init__(
            self,
            _fit_abs: str,
            *,
            scaler_path: str,
            reduce: str,
            use_log: bool,
            v_ml: float,
            noise: NoiseModel,
        ) -> None:
            self._scaler_path = scaler_path
            self.reduce = reduce
            self.use_log = use_log
            self.v_ml = v_ml
            self.noise = noise
            self._x_lo = 0.0
            self._x_hi = 1.0
            self.embedding_dim = 0
            if os.path.isfile(scaler_path):
                with open(scaler_path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                self._x_lo = float(d.get("x_lo", 0.0))
                self._x_hi = float(d.get("x_hi", 1.0))
                self.embedding_dim = int(d.get("embedding_dim", 0))

        def fit_scaler(self, x: np.ndarray) -> None:
            a = np.asarray(x, dtype=np.float64)
            if a.ndim != 2 or a.size == 0:
                return
            self.embedding_dim = int(a.shape[1])
            self._x_lo = float(np.percentile(a, 1.0))
            self._x_hi = float(np.percentile(a, 99.0))

        def save_scaler(self, path: str) -> None:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "x_lo": self._x_lo,
                        "x_hi": self._x_hi,
                        "embedding_dim": self.embedding_dim,
                    },
                    f,
                )

        def load_scaler(self, path: str) -> None:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            self._x_lo = float(d["x_lo"])
            self._x_hi = float(d["x_hi"])
            self.embedding_dim = int(d.get("embedding_dim", 0))

    mod = ModuleType("aimdc_device_distance")
    mod.DeviceDistance = DeviceDistance
    mod.NoiseModel = NoiseModel
    mod.noise_model_from_json_path = noise_model_from_json_path
    sys.modules["aimdc_device_distance"] = mod


_install_aimdc_device_stubs()

from online_accelerator_device import (  # noqa: E402
    L2RAGOnlinePipeline,
    _configure_device_distance,
    _reconstruct_pq_all,
)


def _cand(emb: List[float], cid: str) -> dict[str, Any]:
    return {
        "embedding": np.asarray(emb, dtype=np.float32),
        "chunk_id": cid,
        "metadata": {"section_heading": "A", "doc_id": "0"},
    }


def _serialize_step5(
    final_out: List[dict[str, Any]], stats: dict[str, Any]
) -> str:
    rows: List[dict[str, Any]] = []
    for e in final_out:
        md = float(e.get("_mmr_min_dist_s", float("nan")))
        rows.append(
            {
                "chunk_id": e.get("chunk_id"),
                "score": float(e.get("score", float("nan"))),
                "method": e.get("_retrieval_method"),
                "_mmr_min_dist_s": None if not np.isfinite(md) else md,
                "emb": np.asarray(e.get("embedding"), dtype=np.float64).tolist(),
            }
        )
    return json.dumps({"stats": stats, "rows": rows}, sort_keys=True)


class _CountingDevice:
    pairwise_calls = 0
    pair_shapes: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []

    def __init__(self) -> None:
        from aimdc_device_distance import NoiseModel

        self.noise = NoiseModel()

    def pairwise(self, _q: np.ndarray, _c: np.ndarray) -> np.ndarray:
        _CountingDevice.pairwise_calls += 1
        return np.zeros(len(_c), dtype=np.float64)

    def pair_matrix(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        _CountingDevice.pair_shapes.append((tuple(a.shape), tuple(b.shape)))
        d = np.linalg.norm(b - a, axis=1)
        return d.reshape(1, -1)


class TestAimdcScope(unittest.TestCase):
    def setUp(self) -> None:
        _CountingDevice.pairwise_calls = 0
        _CountingDevice.pair_shapes.clear()

    def test_no_pairwise_in_step5(self) -> None:
        hw: dict[str, Any] = {
            "l2_accelerator_1_centroids": np.zeros((1, 3), dtype=np.float32),
            "l2_accelerator_2_pq_codes": {},
            "lut_memory": np.zeros((1, 4, 3), dtype=np.float32),
            "metadata_sram": {},
        }
        dd = _CountingDevice()
        pl = L2RAGOnlinePipeline(
            hw,
            domain_mode="cuad",
            top_k=3,
            tau_dup=-1e9,
            mmr_lambda=0.85,
            gamma=0.34,
            verbose=False,
            dump_embeddings=False,
            device_distance=dd,
        )
        q = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        cands = [
            _cand([0.9, 0.1, 0.0], "a"),
            _cand([0.5, 0.5, 0.0], "b"),
            _cand([0.0, 1.0, 0.0], "c"),
            _cand([0.2, 0.2, 0.9], "d"),
        ]
        pl.step5_on_the_fly_mmr(q, cands, s_buffer=[], buffer_top_k=3)
        self.assertEqual(_CountingDevice.pairwise_calls, 0)

    def test_pair_matrix_calls_one_per_step(self) -> None:
        """Exactly one pair_matrix call per MMR iteration (no redundant readouts)."""
        hw: dict[str, Any] = {
            "l2_accelerator_1_centroids": np.zeros((1, 3), dtype=np.float32),
            "l2_accelerator_2_pq_codes": {},
            "lut_memory": np.zeros((1, 4, 3), dtype=np.float32),
            "metadata_sram": {},
        }
        dd = _CountingDevice()
        pl = L2RAGOnlinePipeline(
            hw,
            domain_mode="cuad",
            top_k=3,
            tau_dup=-1e9,
            mmr_lambda=0.5,
            gamma=0.34,
            verbose=False,
            device_distance=dd,
            # Isolate the streaming-loop readout count: the per-query diversity
            # spread-matching (div_scale) intentionally issues extra sampling
            # readouts, which are unrelated to this per-step invariant.
            standardize_mmr_terms=False,
        )
        q = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        cands = [
            _cand([0.9, 0.1, 0.0], "a"),
            _cand([0.5, 0.5, 0.0], "b"),
            _cand([0.0, 1.0, 0.0], "c"),
            _cand([0.2, 0.2, 0.9], "d"),
        ]
        pl.step5_on_the_fly_mmr(q, cands, s_buffer=[], buffer_top_k=3)

        # top_k=3, gamma=0.34 -> 1 anchor, 2 mmr_slots, 3 candidates in MMR stream.
        # Each stream candidate triggers exactly one pair_matrix call.
        # (Fallback adds calls only when skipped_for_dedup is non-empty and buffer not full.)
        self.assertEqual(
            len(_CountingDevice.pair_shapes),
            3,
            f"expected 3 calls, got {len(_CountingDevice.pair_shapes)}",
        )

        pool_sizes = [bsh[0] for _, bsh in _CountingDevice.pair_shapes]
        self.assertEqual(
            pool_sizes,
            sorted(pool_sizes),
            f"pool size must be monotone: {pool_sizes}",
        )
        self.assertEqual(
            pool_sizes[0],
            1,
            f"first call should see anchor-only pool: {pool_sizes[0]}",
        )

    def test_pair_matrix_shapes(self) -> None:
        hw: dict[str, Any] = {
            "l2_accelerator_1_centroids": np.zeros((1, 3), dtype=np.float32),
            "l2_accelerator_2_pq_codes": {},
            "lut_memory": np.zeros((1, 4, 3), dtype=np.float32),
            "metadata_sram": {},
        }
        dd = _CountingDevice()
        pl = L2RAGOnlinePipeline(
            hw,
            domain_mode="cuad",
            top_k=3,
            tau_dup=-1e9,
            mmr_lambda=0.85,
            gamma=0.34,
            verbose=False,
            device_distance=dd,
        )
        q = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        cands = [
            _cand([0.9, 0.1, 0.0], "a"),
            _cand([0.5, 0.5, 0.0], "b"),
            _cand([0.0, 1.0, 0.0], "c"),
            _cand([0.2, 0.2, 0.9], "d"),
        ]
        pl.step5_on_the_fly_mmr(q, cands, s_buffer=[], buffer_top_k=3)
        self.assertGreater(len(_CountingDevice.pair_shapes), 0)
        for ash, bsh in _CountingDevice.pair_shapes:
            self.assertEqual(ash, (1, 3))
            self.assertGreaterEqual(bsh[0], 1)
            self.assertEqual(bsh[1], 3)

    def test_scaler_fit_matches_pq_recon_percentiles(self) -> None:
        rng = np.random.default_rng(0)
        m, k, sub = 2, 8, 4
        lut = rng.standard_normal((m, k, sub)).astype(np.float32)
        posting: dict[int, list] = {
            0: [{"pq_code": np.array([1, 3], dtype=np.int64)} for _ in range(120)]
        }
        pq = _reconstruct_pq_all(lut, posting, max_rows=50000)
        x_lo = float(np.percentile(pq, 1.0))
        x_hi = float(np.percentile(pq, 99.0))
        with tempfile.TemporaryDirectory() as td:
            fitp = os.path.join(td, "fit.json")
            with open(fitp, "w", encoding="utf-8") as f:
                json.dump({}, f)
            scaler_p = os.path.join(td, "s.json")
            hw = {"lut_memory": lut, "l2_accelerator_2_pq_codes": posting}
            dd, _k_unused = _configure_device_distance(
                hw,
                device_fit_path=fitp,
                device_scaler_path=scaler_p,
                device_reduce="mean",
                device_use_log=True,
                device_v_ml=0.0,
                noise_config=None,
                noise_seed=None,
            )
        self.assertAlmostEqual(dd._x_lo, x_lo, places=6)
        self.assertAlmostEqual(dd._x_hi, x_hi, places=6)

    def test_no_device_step5_byte_stable(self) -> None:
        hw: dict[str, Any] = {
            "l2_accelerator_1_centroids": np.zeros((1, 3), dtype=np.float32),
            "l2_accelerator_2_pq_codes": {},
            "lut_memory": np.zeros((1, 4, 3), dtype=np.float32),
            "metadata_sram": {},
        }
        pl = L2RAGOnlinePipeline(
            hw,
            domain_mode="cuad",
            top_k=3,
            tau_dup=-1e9,
            mmr_lambda=0.85,
            gamma=0.34,
            verbose=False,
            device_distance=None,
        )
        q = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        cands = [
            _cand([0.9, 0.1, 0.0], "a"),
            _cand([0.5, 0.5, 0.0], "b"),
            _cand([0.0, 1.0, 0.0], "c"),
            _cand([0.2, 0.2, 0.9], "d"),
        ]
        o1, s1 = pl.step5_on_the_fly_mmr(q, cands, s_buffer=[], buffer_top_k=3)
        o2, s2 = pl.step5_on_the_fly_mmr(q, cands, s_buffer=[], buffer_top_k=3)
        self.assertEqual(_serialize_step5(o1, s1), _serialize_step5(o2, s2))

    def test_mmr_score_formula_with_device_dump(self) -> None:
        class _FixedDevice:
            def __init__(self) -> None:
                from aimdc_device_distance import NoiseModel

                self.noise = NoiseModel()

            def pair_matrix(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
                _ = a
                n = b.shape[0]
                return np.full((1, n), 50.0, dtype=np.float64)

        hw: dict[str, Any] = {
            "l2_accelerator_1_centroids": np.zeros((1, 3), dtype=np.float32),
            "l2_accelerator_2_pq_codes": {},
            "lut_memory": np.zeros((1, 4, 3), dtype=np.float32),
            "metadata_sram": {},
        }
        lam = 0.85
        pl = L2RAGOnlinePipeline(
            hw,
            domain_mode="cuad",
            top_k=4,
            tau_dup=-1e9,
            mmr_lambda=lam,
            gamma=0.25,
            verbose=False,
            dump_embeddings=True,
            device_distance=_FixedDevice(),
        )
        q = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        cands = [
            _cand([0.9, 0.1, 0.0], "a"),
            _cand([0.5, 0.5, 0.0], "b"),
            _cand([0.0, 1.0, 0.0], "c"),
            _cand([0.2, 0.2, 0.9], "d"),
        ]
        pl._dump_reset()
        pl.step5_on_the_fly_mmr(q, cands, s_buffer=[], buffer_top_k=4)
        for rec in pl._dump_records["mmr_steps"]:
            if not rec.get("selected"):
                continue
            dqc = float(rec["dist_q_c_raw"])
            mds = float(rec["min_dist_c_s_used"])
            score = float(rec["mmr_score"])
            exp = -lam * dqc + (1.0 - lam) * mds
            self.assertLess(abs(score - exp), 1e-5)

    def test_dump_no_beta_keys(self) -> None:
        hw: dict[str, Any] = {
            "l2_accelerator_1_centroids": np.zeros((1, 3), dtype=np.float32),
            "l2_accelerator_2_pq_codes": {},
            "lut_memory": np.zeros((1, 4, 3), dtype=np.float32),
            "metadata_sram": {},
        }
        pl = L2RAGOnlinePipeline(
            hw,
            domain_mode="cuad",
            top_k=4,
            tau_dup=-1e9,
            mmr_lambda=0.85,
            gamma=0.25,
            verbose=False,
            dump_embeddings=True,
            device_distance=None,
        )
        q = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        cands = [
            _cand([0.9, 0.1, 0.0], "a"),
            _cand([0.5, 0.5, 0.0], "b"),
            _cand([0.0, 1.0, 0.0], "c"),
            _cand([0.2, 0.2, 0.9], "d"),
        ]
        pl._dump_reset()
        pl.step5_on_the_fly_mmr(q, cands, s_buffer=[], buffer_top_k=4)

        def walk(x: Any) -> None:
            if isinstance(x, dict):
                self.assertNotIn("dist_q_c_aimdc", x)
                self.assertNotIn("distance_backend_step1", x)
                for v in x.values():
                    walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)

        walk(pl._dump_records)


if __name__ == "__main__":
    unittest.main()
