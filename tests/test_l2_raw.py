"""L2_RAW path: raw-embedding distance baseline."""
from __future__ import annotations

import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _write_min_offline(d: Path, *, n: int = 20, dim: int = 8) -> None:
    rng = np.random.default_rng(0)
    raw = rng.standard_normal((n, dim)).astype(np.float32)
    np.save(d / "raw_embeddings.npy", raw)
    # OPQ identity
    np.save(d / "opq_rotation.npy", np.eye(dim, dtype=np.float32))
    # PQ: M=2, K=4, subdim=4
    lut = rng.standard_normal((2, 4, 4)).astype(np.float32)
    np.save(d / "pq_codebooks.npy", lut)
    codes = rng.integers(0, 4, size=(n, 2), dtype=np.uint8)
    posting: dict[int, list] = {0: []}
    metadata_sram: dict[str, Any] = {}
    for i in range(n):
        cid = f"c_{i}"
        posting[0].append({"pq_code": codes[i].copy(), "id": i, "chunk_id": cid, "vector_id": i})
        metadata_sram[cid] = {"raw_row": i, "section_heading": "H", "doc_id": "0"}
    with open(d / "posting_lists.pkl", "wb") as f:
        pickle.dump(posting, f)
    with open(d / "metadata_db.pkl", "wb") as f:
        pickle.dump(metadata_sram, f)
    np.save(d / "centroids.npy", np.zeros((1, dim), dtype=np.float32))


class TestL2Raw(unittest.TestCase):
    def test_raw_row_index_resolves(self) -> None:
        from online_accelerator_device import L2RAGOnlinePipeline, load_hw_state_from_dir

        with tempfile.TemporaryDirectory() as td:
            _write_min_offline(Path(td))
            hw = load_hw_state_from_dir(td)
            pl = L2RAGOnlinePipeline(hw, top_k=5, verbose=False)
            idx = pl._build_raw_row_index()
            self.assertGreater(len(idx), 0)
            for cid, row in idx.items():
                self.assertIsInstance(cid, str)
                self.assertIsInstance(row, int)

    def test_l2_raw_differs_from_pq_l2(self) -> None:
        """Raw-L2 ranking must differ from PQ-recon L2 when PQ is lossy."""
        from online_accelerator_device import L2RAGOnlinePipeline, load_hw_state_from_dir

        with tempfile.TemporaryDirectory() as td:
            _write_min_offline(Path(td))
            hw = load_hw_state_from_dir(td)
            pl = L2RAGOnlinePipeline(hw, top_k=5, verbose=False, use_centroid_filter=False)
            rng = np.random.default_rng(123)
            q = rng.standard_normal(8).astype(np.float32)
            pl._build_query_embedding_streams = (  # type: ignore[method-assign]
                lambda _t: (q, q, [], None)
            )
            l2_out, _s1, _l1 = pl._run_l2_only("dummy", target_doc_id="0")
            l2raw_out, _s2, _l2 = pl._run_l2_only_raw("dummy", target_doc_id="0")
            l2_ids: List[str] = [str(c["chunk_id"]) for c in l2_out]
            l2raw_ids: List[str] = [str(c["chunk_id"]) for c in l2raw_out]
            self.assertNotEqual(
                l2_ids[:3],
                l2raw_ids[:3],
                msg=(
                    f"L2 and L2_RAW returned identical top-3 — PQ may not be lossy enough; "
                    f"or raw path fell through to PQ. L2={l2_ids[:3]}, raw={l2raw_ids[:3]}"
                ),
            )

    def test_l2_raw_dist_matches_manual(self) -> None:
        """Top-K order equals ascending ||q_raw - c_raw|| (within float tolerance)."""
        from online_accelerator_device import L2RAGOnlinePipeline, load_hw_state_from_dir

        with tempfile.TemporaryDirectory() as td:
            _write_min_offline(Path(td))
            hw = load_hw_state_from_dir(td)
            pl = L2RAGOnlinePipeline(hw, top_k=3, verbose=False, use_centroid_filter=False)
            rng = np.random.default_rng(999)
            q = rng.standard_normal(8).astype(np.float32)
            pl._build_query_embedding_streams = (  # type: ignore[method-assign]
                lambda _t: (q, q, [], None)
            )
            l2raw_out, _, _ = pl._run_l2_only_raw("dummy", target_doc_id="0")
            raw_table = np.asarray(hw["raw_embeddings"], dtype=np.float32)
            idx = pl._build_raw_row_index()
            dists = [
                float(np.linalg.norm(q - raw_table[int(idx[str(c["chunk_id"])])]))
                for c in l2raw_out
            ]
            self.assertEqual(dists, sorted(dists))
            if l2raw_out:
                c0 = l2raw_out[0]
                r0 = int(idx[str(c0["chunk_id"])])
                d0 = float(np.linalg.norm(q - raw_table[r0]))
                self.assertLess(abs(d0 - dists[0]), 1e-6)

    def test_l2_raw_missing_raw_embeddings_raises(self) -> None:
        """Fail loudly when raw_embeddings is unavailable."""
        from online_accelerator_device import L2RAGOnlinePipeline

        hw: dict[str, Any] = {
            "l2_accelerator_1_centroids": np.zeros((1, 8), dtype=np.float32),
            "l2_accelerator_2_pq_codes": {},
            "lut_memory": np.zeros((1, 4, 4), dtype=np.float32),
            "metadata_sram": {},
        }
        pl = L2RAGOnlinePipeline(hw, top_k=5, verbose=False)
        pl._build_query_embedding_streams = (  # type: ignore[method-assign]
            lambda _t: (
                np.zeros(8, dtype=np.float32),
                np.zeros(8, dtype=np.float32),
                [],
                None,
            )
        )
        with self.assertRaises(RuntimeError):
            pl._run_l2_only_raw("dummy", target_doc_id="0")

    def test_compare_all_mode_returns_three_result_sets(self) -> None:
        """run_pipeline with COMPARE_ALL must return all three rankings."""
        from online_accelerator_device import L2RAGOnlinePipeline, load_hw_state_from_dir

        with tempfile.TemporaryDirectory() as td:
            _write_min_offline(Path(td))
            hw = load_hw_state_from_dir(td)
            pl = L2RAGOnlinePipeline(hw, top_k=5, verbose=False, use_centroid_filter=False)
            rng = np.random.default_rng(7)
            q = rng.standard_normal(8).astype(np.float32)
            pl._build_query_embedding_streams = (  # type: ignore[method-assign]
                lambda _t: (q, q, [], None)
            )
            out = pl.run_pipeline("q", target_doc_id="0", log_summary=False, mode="COMPARE_ALL")
            self.assertIsInstance(out, tuple)
            assert isinstance(out, tuple)
            self.assertEqual(len(out), 6)
            l2_out, l2raw_out, mmr_out, _s1, _s2, _s3 = out
            self.assertEqual(len(l2_out), len(l2raw_out), "top-K count must match")
            self.assertTrue(all(c.get("_retrieval_method") == "L2_Raw" for c in l2raw_out))


if __name__ == "__main__":
    unittest.main()
