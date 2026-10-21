"""Regression: one pair_matrix call per MMR step; noise samples consistent within a step."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _minimal_hw(D: int) -> dict[str, Any]:
    return {
        "l2_accelerator_1_centroids": np.zeros((1, D), dtype=np.float32),
        "l2_accelerator_2_pq_codes": {},
        "lut_memory": np.zeros((1, 4, D), dtype=np.float32),
        "metadata_sram": {},
    }


def _cand(emb: List[float], cid: str) -> dict[str, Any]:
    return {
        "embedding": np.asarray(emb, dtype=np.float32),
        "chunk_id": cid,
        "metadata": {"section_heading": "A", "doc_id": "0"},
    }


class _NoisyRecordingDevice:
    """Injects per-call Gaussian noise to expose cross-call inconsistency bugs."""

    def __init__(self, seed: int = 0) -> None:
        from aimdc_device_distance import NoiseModel

        self.noise = NoiseModel()
        self.rng = np.random.default_rng(seed)
        self.returned_arrays: List[np.ndarray] = []
        self.calls = 0

    def pair_matrix(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        self.calls += 1
        base = np.linalg.norm(b - a, axis=1).reshape(1, -1)
        # Large multiplicative noise so two independent calls would diverge visibly.
        noise = 1.0 + 0.3 * self.rng.standard_normal(base.shape)
        out = (base * noise).astype(np.float64)
        self.returned_arrays.append(out.copy())
        return out


def test_min_dist_c_s_used_matches_raw_array_min_under_noise() -> None:
    """With noise on, min_dist_c_s_used must equal the min of the dumped raw array
    (modulo section-heading penalty). This can only hold if both come from the
    same pair_matrix call.
    """
    pytest.importorskip("aimdc_device_distance")
    from online_accelerator_device import L2RAGOnlinePipeline

    D = 3
    hw = _minimal_hw(D)
    dd = _NoisyRecordingDevice(seed=42)
    pl = L2RAGOnlinePipeline(
        hw,
        domain_mode="cuad",
        top_k=3,
        tau_dup=-1e9,
        mmr_lambda=0.5,
        gamma=0.34,
        verbose=False,
        dump_embeddings=True,
        device_distance=dd,
    )
    q = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    cands = [
        _cand([0.9, 0.1, 0.0], "a"),
        _cand([0.5, 0.5, 0.0], "b"),
        _cand([0.0, 1.0, 0.0], "c"),
        _cand([0.2, 0.2, 0.9], "d"),
    ]
    pl._dump_reset()
    pl.step5_on_the_fly_mmr(q, cands, s_buffer=[], buffer_top_k=3)

    mismatches: List[Tuple[float, float]] = []
    for rec in pl._dump_records["mmr_steps"]:
        if rec.get("method") != "MMR" or rec.get("skipped_dedup"):
            continue
        pd = np.asarray(rec["per_pair_dists"], dtype=np.float64)
        if pd.size == 0:
            continue
        used = float(rec["min_dist_c_s_used"])
        raw_min = float(np.min(pd))
        diff = abs(used - raw_min)
        diff_penalized = abs(used * pl.section_penalty_weight - raw_min)
        if min(diff, diff_penalized) > 1e-6 * max(abs(raw_min), 1.0):
            mismatches.append((used, raw_min))
    assert not mismatches, (
        f"inconsistencies between min_dist_c_s_used and per_pair_dists: {mismatches[:3]}"
    )


def test_call_count_equals_mmr_stream_length() -> None:
    """With top_k=3, gamma=0.34, 4 candidates -> 1 anchor + 3 stream candidates.
    Exactly 3 pair_matrix calls expected (no fallback invocations here)."""
    pytest.importorskip("aimdc_device_distance")
    from online_accelerator_device import L2RAGOnlinePipeline

    D = 3
    hw = _minimal_hw(D)
    dd = _NoisyRecordingDevice(seed=0)
    pl = L2RAGOnlinePipeline(
        hw,
        domain_mode="cuad",
        top_k=3,
        tau_dup=-1e9,
        mmr_lambda=0.5,
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
    assert dd.calls == 3, f"expected 3 calls, got {dd.calls}"
