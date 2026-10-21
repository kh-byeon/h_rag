"""Unit tests for the per-query MMR diversity-term spread-matching (div_scale).

Covers:
  (i)   div_scale == 1.0 when device_distance is None (software-L2 path).
  (ii)  with a synthetic compressed diversity distance (std ~9x smaller than the
        relevance term), the standardized diversity-term std ~= relevance std.
  (iii) the candidate rank order by the diversity term is unchanged by div_scale.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np

from online_accelerator_device import L2RAGOnlinePipeline


class _CompressedDevice:
    """Fake DeviceDistance: pairwise distance = software L2 / compression.

    Only needs pair_matrix(a, b) -> (len(a), len(b)) matrix, which is what
    L2RAGOnlinePipeline._mmr_diversity_scale calls.
    """

    def __init__(self, compression: float = 9.0) -> None:
        self.compression = float(compression)

    def pair_matrix(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
        return d / self.compression


def _make_pipeline(device_distance=None) -> L2RAGOnlinePipeline:
    return L2RAGOnlinePipeline(
        {},
        domain_mode="cuad",
        verbose=False,
        device_distance=device_distance,
    )


def _make_scored(n: int = 24, dim: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)
    query = rng.standard_normal(dim).astype(np.float32)
    embs = rng.standard_normal((n, dim)).astype(np.float32)
    scored = []
    for i in range(n):
        dist_q_c = float(np.linalg.norm(query - embs[i]))
        scored.append(({"embedding": embs[i], "chunk_id": f"c{i}"}, dist_q_c))
    scored.sort(key=lambda x: x[1])
    return scored


def _diversity_mins(scored, device, compression_scaled=True):
    embs = np.stack(
        [np.asarray(c["embedding"], dtype=np.float32) for c, _ in scored[:32]],
        axis=0,
    )
    mins = []
    for i in range(len(embs)):
        others = np.delete(embs, i, axis=0)
        dd = device.pair_matrix(
            embs[i : i + 1].astype(np.float64), others.astype(np.float64)
        )[0]
        mins.append(float(np.min(dd)))
    return np.asarray(mins, dtype=np.float64)


def test_div_scale_is_one_when_software_path():
    """(i) No device distance -> div_scale is exactly 1.0 (byte-for-byte SW path)."""
    pipe = _make_pipeline(device_distance=None)
    scored = _make_scored()
    div_scale = pipe._mmr_diversity_scale(scored, apply_k=False, k_val=1.0)
    assert div_scale == 1.0


def test_div_scale_disabled_flag_is_one():
    """Explicitly disabling standardization keeps div_scale == 1.0 on analog path."""
    pipe = _make_pipeline(device_distance=_CompressedDevice(9.0))
    pipe.mmr_standardize_terms = False
    scored = _make_scored()
    assert pipe._mmr_diversity_scale(scored, apply_k=False, k_val=1.0) == 1.0


def test_standardized_diversity_std_matches_relevance_std():
    """(ii) Compressed diversity term rescaled to match relevance-term spread."""
    pipe = _make_pipeline(device_distance=_CompressedDevice(9.0))
    scored = _make_scored()

    div_scale = pipe._mmr_diversity_scale(scored, apply_k=False, k_val=1.0)

    dqc_std = float(np.std(np.asarray([d for _, d in scored], dtype=np.float64)))
    mins = _diversity_mins(scored, pipe.device_distance)
    div_std_raw = float(np.std(mins))

    # The raw diversity spread is much smaller than the relevance spread...
    assert div_std_raw < dqc_std
    # ...and after applying div_scale the two spreads match.
    standardized_std = float(np.std(div_scale * mins))
    assert np.isclose(standardized_std, dqc_std, rtol=1e-9, atol=1e-9)


def test_div_scale_preserves_rank_order():
    """(iii) A positive scalar multiply leaves the diversity-term ranking intact."""
    pipe = _make_pipeline(device_distance=_CompressedDevice(9.0))
    scored = _make_scored(seed=7)

    div_scale = pipe._mmr_diversity_scale(scored, apply_k=False, k_val=1.0)
    assert div_scale > 0.0

    mins = _diversity_mins(scored, pipe.device_distance)
    order_raw = np.argsort(mins)
    order_scaled = np.argsort(div_scale * mins)
    assert np.array_equal(order_raw, order_scaled)


def test_diversity_stats_disabled_on_software_path():
    """(gate) Software path / disabled -> (md_mean, md_std) sentinel (0.0, 0.0)."""
    pipe = _make_pipeline(device_distance=None)
    scored = _make_scored()
    assert pipe._mmr_diversity_stats(scored, apply_k=False, k_val=1.0) == (0.0, 0.0)

    pipe2 = _make_pipeline(device_distance=_CompressedDevice(9.0))
    pipe2.mmr_standardize_terms = False
    assert pipe2._mmr_diversity_stats(scored, apply_k=False, k_val=1.0) == (0.0, 0.0)


def test_zscored_tau_has_unit_std():
    """(acceptance) The z-standardized gate value has unit std across candidates."""
    pipe = _make_pipeline(device_distance=_CompressedDevice(9.0))
    scored = _make_scored(n=32, seed=1)

    md_mean, md_std = pipe._mmr_diversity_stats(scored, apply_k=False, k_val=1.0)
    assert md_std > 1e-12

    mins = _diversity_mins(scored, pipe.device_distance)
    z = (mins - md_mean) / md_std
    assert np.isclose(float(np.std(z)), 1.0, rtol=1e-9, atol=1e-9)


def test_zscore_gate_gives_finegrained_control_vs_raw():
    """(acceptance) Sweeping tau over [-2, 0]:
    - z-standardized gate: admit/reject fraction moves smoothly (many distinct
      values, monotonic, no single all-or-nothing flip);
    - raw gate: the compressed distances bunch away from [-2, 0], so the reject
      fraction is constant (all-or-nothing) over the same grid.
    """
    pipe = _make_pipeline(device_distance=_CompressedDevice(9.0))
    scored = _make_scored(n=32, seed=2)

    md_mean, md_std = pipe._mmr_diversity_stats(scored, apply_k=False, k_val=1.0)
    mins = _diversity_mins(scored, pipe.device_distance)

    # Raw diversity distances are tightly bunched (std ~9x smaller than SW L2).
    dqc_std = float(np.std(np.asarray([d for _, d in scored], dtype=np.float64)))
    assert float(np.std(mins)) < dqc_std

    taus = [-2.0, -1.5, -1.0, -0.5, 0.0]
    z = (mins - md_mean) / md_std
    reject_frac_z = [float(np.mean(z < t)) for t in taus]
    reject_frac_raw = [float(np.mean(mins < t)) for t in taus]

    # z gate: monotonic non-decreasing and finely graded.
    assert all(b >= a - 1e-12 for a, b in zip(reject_frac_z, reject_frac_z[1:]))
    assert len(set(np.round(reject_frac_z, 3))) >= 3
    # no single threshold step rejects a huge chunk at once (smooth control).
    max_jump_z = max(b - a for a, b in zip(reject_frac_z, reject_frac_z[1:]))
    assert max_jump_z <= 0.5

    # raw gate: all-or-nothing over this (unmatched) grid — constant fraction.
    assert len(set(np.round(reject_frac_raw, 6))) == 1


def test_apply_k_scales_both_consistently():
    """k_val cancels in the ratio direction: div_scale tracks 1/k for the term.

    With apply_k, the raw diversity distances are multiplied by k_val, so div_std
    grows by k_val and div_scale shrinks by 1/k_val. The product div_scale*(k*mins)
    still matches the relevance spread.
    """
    pipe = _make_pipeline(device_distance=_CompressedDevice(9.0))
    scored = _make_scored(seed=3)
    k_val = 5.0

    div_scale = pipe._mmr_diversity_scale(scored, apply_k=True, k_val=k_val)
    dqc_std = float(np.std(np.asarray([d for _, d in scored], dtype=np.float64)))
    mins = _diversity_mins(scored, pipe.device_distance)
    standardized_std = float(np.std(div_scale * (k_val * mins)))
    assert np.isclose(standardized_std, dqc_std, rtol=1e-9, atol=1e-9)
