"""Torch-free unit test of the IGZO-tier write/readout counters in step5.

Synthetic 12-candidate pool, top_k=5, gamma=0.7 -> 3 anchors + 2 MMR slots.
Checks (software-L2 path, no device model needed):
  * conservation: every streamed candidate lands in exactly one bin
    (tau reject / score reject / append / replace);
  * writes_total = n_anchor + writes_mmr;
  * the two MMR slots are filled by appends, everything else that enters the
    buffer is a replacement;
  * one match-line readout per streamed candidate in the main loop;
  * the counters do not change the selection (bit-identical to a second call).
"""
import numpy as np, os, sys
sys.path.insert(0, os.getcwd())
from online_accelerator_device import L2RAGOnlinePipeline, load_hw_state_from_dir

HW_DIR = "offline_data_hotpot_q1000" if os.path.isdir("offline_data_hotpot_q1000") else "offline_data_hotpot"
hw = load_hw_state_from_dir(HW_DIR)
D = 64
rng = np.random.default_rng(0)


def unit(v):
    v = np.asarray(v, np.float64)
    return (v / np.linalg.norm(v)).astype(np.float32)


def chunk(i, vec):
    return {"chunk_id": f"c{i}", "centroid_id": 0, "doc_id": str(i),
            "embedding": vec, "chunk_text": f"text {i}", "metadata": {"section_heading": ""}}


q = unit([1.0] + [0.0] * (D - 1))
# 12 candidates with decreasing relevance and random diversity
cands = []
for i in range(12):
    v = np.zeros(D); v[0] = 1.0 - 0.05 * i; v[1 + i] = 0.4; v += 0.05 * rng.standard_normal(D)
    cands.append(chunk(i, unit(v)))


def run(tau=-1.0):
    p = L2RAGOnlinePipeline(hw, domain_mode="hotpot", top_k=5, tau_dup=tau,
                            mmr_lambda=0.7, gamma=0.7, verbose=False)
    sel, meta = p.step5_on_the_fly_mmr(q, cands, [], buffer_top_k=5, gamma=0.7)
    return [e["chunk_id"] for e in sel], meta


def test_geometry_and_conservation():
    ids, meta = run()
    ws = meta["write_stats"]
    assert ws["n_pool"] == 12
    assert ws["n_anchor"] == 3 and ws["mmr_slots"] == 2
    assert ws["n_stream_avail"] == 9
    assert len(ids) == 5 and ws["buffer_rows_final"] == 5
    binned = ws["n_reject_tau"] + ws["n_reject_score"] + ws["n_write_append"] + ws["n_write_replace"]
    assert binned == ws["n_streamed"], ws
    assert ws["writes_mmr"] == ws["n_write_append"] + ws["n_write_replace"]
    assert ws["writes_total"] == ws["n_anchor"] + ws["writes_mmr"]


def test_slots_filled_by_appends_then_replacements():
    ids, meta = run(tau=-1e9)  # gate never fires (software path: raw distances >= 0)
    ws = meta["write_stats"]
    assert ws["n_reject_tau"] == 0
    assert ws["n_write_append"] == 2, ws
    assert ws["n_streamed"] == 9
    assert ws["n_reject_score"] + ws["n_write_replace"] == 7


def test_one_readout_per_streamed_candidate_main_loop():
    _, meta = run(tau=-1e9)
    ws = meta["write_stats"]
    assert ws["n_stream_fallback"] == 0
    assert ws["n_readout_main"] == ws["n_stream_main"]
    assert ws["n_readout_calib"] == 0  # software path: no standardization pass


def test_counters_are_pure_instrumentation():
    a = run(); b = run()
    assert a[0] == b[0]
    assert a[1]["write_stats"] == b[1]["write_stats"]
    assert a[1]["early_dedup_count"] == a[1]["write_stats"]["n_reject_tau"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
