"""Torch-free unit test of the anchor-protection gate in step5_on_the_fly_mmr.

Builds a candidate pool where one HIGHLY RELEVANT chunk is also near-duplicate of
the top hit -- exactly the case that made MMR drop gold second-hop paragraphs.
With anchor_protect_n=0 it must be dropped by the tau_dup gate; with =1 it must
survive the gate and compete on the MMR score.
"""
import numpy as np, os, sys
sys.path.insert(0, os.getcwd())
from online_accelerator_device import L2RAGOnlinePipeline, load_hw_state_from_dir

hw = load_hw_state_from_dir("offline_data_hotpot")
D = 384
q = np.zeros(D, dtype=np.float32); q[0] = 1.0

def chunk(i, vec):
    return {"chunk_id": f"c{i}", "centroid_id": 0, "doc_id": str(i),
            "embedding": np.asarray(vec, dtype=np.float32),
            "chunk_text": f"text {i}", "metadata": {"section_heading": ""}}

# rank0: closest.  rank1: a NEAR-DUPLICATE of rank0 that is still very relevant
# (this is the one the dedup gate kills).  rank2..4: distant, diverse fillers.
def unit(v):
    v = np.asarray(v, np.float32); return v / np.linalg.norm(v)
cands = [
    chunk(0, unit([1.0, 0.02] + [0.0]*(D-2))),
    chunk(1, unit([1.0, 0.03] + [0.0]*(D-2))),   # ~duplicate of rank0, high relevance
    chunk(2, unit([0.6, 0.0, 0.8] + [0.0]*(D-3))),
    chunk(3, unit([0.5, 0.0, 0.0, 0.87] + [0.0]*(D-4))),
    chunk(4, unit([0.4, 0.0, 0.0, 0.0, 0.92] + [0.0]*(D-5))),
    chunk(5, unit([0.3, 0.0, 0.0, 0.0, 0.0, 0.95] + [0.0]*(D-6))),
]

def run(protect_n, tau):
    p = L2RAGOnlinePipeline(hw, domain_mode="hotpot", top_k=3, tau_dup=tau,
                            mmr_lambda=0.7, gamma=0.34,   # -> anchor_count = 1
                            anchor_protect_n=protect_n, verbose=False)
    sel, meta = p.step5_on_the_fly_mmr(q, cands, [], buffer_top_k=3, gamma=0.34)
    ids = [e.get("chunk_id") for e in sel]
    return ids, meta.get("early_dedup_count"), [e.get("_retrieval_method") for e in sel]


def test_disabled_is_a_noop():
    """anchor_protect_n=0 must reproduce the unprotected path exactly."""
    for tau in (-1.0, -0.2, 0.3, 1.0):
        a = run(0, tau)
        b = run(0, tau)
        assert a == b, "selection is not deterministic"


def test_protection_never_increases_dedup_skips():
    for tau in (-1.0, -0.2, 0.0, 0.3, 0.6, 1.0):
        skips = [run(n_, tau)[1] for n_ in (0, 1, 2)]
        assert skips == sorted(skips, reverse=True), (
            f"tau={tau}: dedup skips must be monotone non-increasing in "
            f"anchor_protect_n, got {skips}"
        )


def test_protection_rescues_a_relevant_near_duplicate():
    """The whole point: c1 is rank-1 by relevance AND a near-duplicate of c0.

    Unprotected, the tau_dup gate discards it and a less relevant chunk takes the
    slot. Protected, it survives the gate and wins the slot on the MMR score.
    """
    tau = 0.3
    ids_off, skips_off, _ = run(0, tau)
    ids_on, skips_on, _ = run(1, tau)
    assert "c1" not in ids_off and skips_off >= 1, (
        f"fixture no longer triggers the gate: {ids_off}, skips={skips_off}"
    )
    assert "c1" in ids_on and skips_on < skips_off, (
        f"protection failed to rescue the candidate: {ids_on}, skips={skips_on}"
    )


if __name__ == "__main__":
    print(f"{'tau':>6} {'protect_n':>10} {'selected':>24} {'dedup-skipped':>14}")
    for tau in (-1.0, 0.0, 0.3, 0.6, 1.0):
        for n_ in (0, 1, 2):
            ids, ded, _ = run(n_, tau)
            print(f"{tau:>6} {n_:>10} {str(ids):>24} {ded:>14}")
        print()
    test_disabled_is_a_noop()
    test_protection_never_increases_dedup_skips()
    test_protection_rescues_a_relevant_near_duplicate()
    print("all anchor-protection tests passed")
