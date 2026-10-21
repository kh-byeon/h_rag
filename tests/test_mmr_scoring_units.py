from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np

from aimdc_device_distance import NoiseModel
from online_accelerator_device import L2RAGOnlinePipeline


class _MockNoiseDev:
    """Minimal device_distance stub with deterministic AIMDC-like outputs."""

    noise = NoiseModel()

    def pairwise(self, qq: np.ndarray, cm: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def pair_matrix(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def _minimal_hw(D: int = 768) -> Dict[str, Any]:
    return {
        "l2_accelerator_1_centroids": np.zeros((1, D), dtype=np.float32),
        "l2_accelerator_2_pq_codes": {},
        "lut_memory": None,
        "metadata_sram": {},
        "df_dict": {},
        "total_docs": 0,
    }


def _make_cands(n: int, D: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i in range(n):
        emb = np.zeros(D, dtype=np.float32)
        emb[0] = float(i) * 0.15
        emb[1] = 1.0
        out.append({"chunk_id": str(i), "embedding": emb, "metadata": {}})
    return out


# Frozen snapshot: step5, device_distance=None, top_k=4, gamma=0.25, tau_dup=-100, D=768, six candidates above.
_GOLDEN_NO_DEV: Dict[str, Any] = {
    "ids": ["5", "4", "3", "2"],
    "scores": [
        -1.0307763814926147,
        -0.4635164737701416,
        -0.49563559889793396,
        -0.5353277921676636,
    ],
    "early": 0,
    "cov": None,
    "min_dists_tail": [
        0.1499999761581421,
        0.15000003576278687,
        0.1499999761581421,
    ],
}


def _assert_no_dev_golden(out: List[Dict[str, Any]], stats: Dict[str, Any]) -> None:
    g = _GOLDEN_NO_DEV
    assert [e.get("chunk_id") for e in out] == g["ids"]
    np.testing.assert_allclose(
        [float(e["score"]) for e in out], g["scores"], rtol=1e-5, atol=1e-5
    )
    assert stats["early_dedup_count"] == g["early"]
    assert stats["coverage_met_at_step"] is g["cov"]
    md = stats["min_dists_accepted"]
    assert math.isinf(float(md[0]))
    np.testing.assert_allclose(
        [float(x) for x in md[1:4]], g["min_dists_tail"], rtol=1e-5, atol=1e-5
    )


def test_step1_sort_order_invariant_with_device() -> None:
    D = 768
    hw = _minimal_hw(D)
    q = np.zeros(D, dtype=np.float32)
    q[0] = 1.0
    cands = _make_cands(8, D)
    ranked = sorted(
        cands,
        key=lambda c: float(
            np.linalg.norm(q - np.asarray(c["embedding"], dtype=np.float32))
        ),
    )
    ranked_ids = [str(c["chunk_id"]) for c in ranked]

    class _MockDev(_MockNoiseDev):
        def pairwise(self, qq: np.ndarray, cm: np.ndarray) -> np.ndarray:
            out = np.empty((cm.shape[0],), dtype=np.float64)
            for i in range(cm.shape[0]):
                out[i] = float(np.sum((cm[i] - qq.reshape(-1)) ** 3))
            return out

        def pair_matrix(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return np.ones((a.shape[0], b.shape[0]), dtype=np.float64) * 2.5

    p0 = L2RAGOnlinePipeline(
        hw,
        top_k=8,
        gamma=0.5,
        mmr_lambda=0.5,
        tau_dup=-100.0,
        verbose=False,
        device_distance=None,
    )
    p1 = L2RAGOnlinePipeline(
        hw,
        top_k=8,
        gamma=0.5,
        mmr_lambda=0.5,
        tau_dup=-100.0,
        verbose=False,
        device_distance=_MockDev(),  # type: ignore[arg-type]
    )
    kw = {"buffer_top_k": 8, "gamma": 0.5}
    out0, _ = p0.step5_on_the_fly_mmr(q, list(cands), s_buffer=[], **kw)
    out1, _ = p1.step5_on_the_fly_mmr(q, list(cands), s_buffer=[], **kw)
    ac = max(1, min(8, int(8 * 0.5)))
    for j in range(ac):
        assert str(out0[j]["chunk_id"]) == ranked_ids[j]
        assert str(out1[j]["chunk_id"]) == ranked_ids[j]


def test_mmr_score_aimdc_magnitude() -> None:
    D = 768
    hw = _minimal_hw(D)
    q = np.zeros(D, dtype=np.float32)
    q[0] = 1.0
    cands = _make_cands(6, D)

    class _MockDev(_MockNoiseDev):
        def pairwise(self, qq: np.ndarray, cm: np.ndarray) -> np.ndarray:
            return np.full((cm.shape[0],), -120.0, dtype=np.float64)

        def pair_matrix(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return np.full((a.shape[0], b.shape[0]), 50.0, dtype=np.float64)

    p = L2RAGOnlinePipeline(
        hw,
        top_k=4,
        gamma=0.25,
        mmr_lambda=0.5,
        tau_dup=-100.0,
        verbose=False,
        device_distance=_MockDev(),  # type: ignore[arg-type]
    )
    out, _ = p.step5_on_the_fly_mmr(
        q, cands, s_buffer=[], buffer_top_k=p.top_k, gamma=p.gamma
    )
    for e in out:
        if str(e.get("_retrieval_method")) == "MMR":
            assert abs(float(e["score"])) > 10.0


def test_anchor_score_units() -> None:
    D = 32
    hw = _minimal_hw(D)
    q = np.zeros(D, dtype=np.float32)
    q[0] = 1.0
    cands = _make_cands(5, D)

    class _MockDev(_MockNoiseDev):
        def pairwise(self, qq: np.ndarray, cm: np.ndarray) -> np.ndarray:
            return np.full((cm.shape[0],), -80.0, dtype=np.float64)

        def pair_matrix(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return np.ones((a.shape[0], b.shape[0]), dtype=np.float64)

    p_dev = L2RAGOnlinePipeline(
        hw,
        top_k=3,
        gamma=0.34,
        mmr_lambda=0.5,
        tau_dup=-100.0,
        verbose=False,
        device_distance=_MockDev(),  # type: ignore[arg-type]
    )
    out_dev, _ = p_dev.step5_on_the_fly_mmr(
        q, cands, s_buffer=[], buffer_top_k=p_dev.top_k, gamma=p_dev.gamma
    )
    for e in out_dev:
        if str(e.get("_retrieval_method")) == "L2_Anchor":
            s = abs(float(e["score"]))
            assert 0.0 <= s <= 2.0

    p_sw = L2RAGOnlinePipeline(
        hw,
        top_k=3,
        gamma=0.34,
        mmr_lambda=0.5,
        tau_dup=-100.0,
        verbose=False,
        device_distance=None,
    )
    out_sw, _ = p_sw.step5_on_the_fly_mmr(
        q, cands, s_buffer=[], buffer_top_k=p_sw.top_k, gamma=p_sw.gamma
    )
    for e in out_sw:
        if str(e.get("_retrieval_method")) == "L2_Anchor":
            s = abs(float(e["score"]))
            assert 0.0 <= s <= 2.0


def test_fallback_exact_match_golden_no_device() -> None:
    D = 768
    hw = _minimal_hw(D)
    q = np.zeros(D, dtype=np.float32)
    q[0] = 1.0
    cands = _make_cands(6, D)
    p = L2RAGOnlinePipeline(
        hw,
        top_k=4,
        gamma=0.25,
        mmr_lambda=0.5,
        tau_dup=-100.0,
        verbose=False,
        device_distance=None,
    )
    kw = {"buffer_top_k": p.top_k, "gamma": p.gamma}
    out1, s1 = p.step5_on_the_fly_mmr(q, cands, s_buffer=[], **kw)
    out2, s2 = p.step5_on_the_fly_mmr(q, cands, s_buffer=[], **kw)
    _assert_no_dev_golden(out1, s1)
    _assert_no_dev_golden(out2, s2)


def test_dump_schema_and_single_pairwise() -> None:
    D = 16
    hw = _minimal_hw(D)
    q = np.zeros(D, dtype=np.float32)
    q[0] = 1.0
    cands = _make_cands(4, D)

    class _MockDev(_MockNoiseDev):
        def __init__(self) -> None:
            self.pairwise_calls = 0
            self.pair_matrix_calls = 0

        def pairwise(self, qq: np.ndarray, cm: np.ndarray) -> np.ndarray:
            self.pairwise_calls += 1
            return np.arange(cm.shape[0], dtype=np.float64) - 50.0

        def pair_matrix(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
            self.pair_matrix_calls += 1
            return np.ones((a.shape[0], b.shape[0]), dtype=np.float64) * 3.0

    dd = _MockDev()
    p = L2RAGOnlinePipeline(
        hw,
        top_k=3,
        gamma=0.34,
        mmr_lambda=0.6,
        tau_dup=-50.0,
        verbose=False,
        dump_embeddings=True,
        device_distance=dd,  # type: ignore[arg-type]
    )
    p._dump_reset()
    p.step5_on_the_fly_mmr(
        q, cands, s_buffer=[], buffer_top_k=p.top_k, gamma=p.gamma
    )
    assert dd.pairwise_calls == 0
    assert dd.pair_matrix_calls >= 1
    for rec in p._dump_records["candidates"]:
        assert isinstance(rec["dist_q_c"], float)
        assert rec["distance_backend"] == "analytical"
    for rec in p._dump_records["mmr_steps"]:
        assert isinstance(rec["dist_q_c_raw"], float)
        assert "dist_q_c_aimdc" not in rec


def test_mmr_score_formula_software_qterm() -> None:
    D = 8
    hw = _minimal_hw(D)
    q = np.zeros(D, dtype=np.float32)
    q[0] = 1.0
    cands = _make_cands(4, D)
    lam = 0.6

    class _MockDev(_MockNoiseDev):
        def pairwise(self, qq: np.ndarray, cm: np.ndarray) -> np.ndarray:
            del qq
            return np.array([-10.0, -20.0, -30.0, -40.0], dtype=np.float64)

        def pair_matrix(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
            del a
            return np.full((1, b.shape[0]), 5.0, dtype=np.float64)

    p = L2RAGOnlinePipeline(
        hw,
        top_k=3,
        gamma=0.34,
        mmr_lambda=lam,
        tau_dup=-100.0,
        verbose=False,
        dump_embeddings=True,
        device_distance=_MockDev(),  # type: ignore[arg-type]
    )
    p._dump_reset()
    p.step5_on_the_fly_mmr(
        q, cands, s_buffer=[], buffer_top_k=p.top_k, gamma=p.gamma
    )
    for rec in p._dump_records["mmr_steps"]:
        if rec.get("method") != "MMR":
            continue
        if rec.get("skipped_dedup"):
            continue
        if rec.get("mmr_score") is None:
            continue
        d_sw = float(rec["dist_q_c_raw"])
        min_u = float(rec["min_dist_c_s_used"])
        exp = -lam * d_sw + (1.0 - lam) * min_u
        assert abs(float(rec["mmr_score"]) - exp) < 1e-5
