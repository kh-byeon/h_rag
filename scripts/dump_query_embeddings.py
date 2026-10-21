"""Dump the 1000 hotpot query embeddings (pre-OPQ, L2-normalised) exactly as
online_accelerator_device.step1_query_in_and_encode would produce them, so that
the device-tolerance probe can run without the encoder.

Run from the repo root inside WSL with the project venv active:
    python scripts/dump_query_embeddings.py
Output: sweep_exports/query_emb_q1000_raw.npy  (1000, 768) float32
"""
import json, numpy as np
from sentence_transformers import SentenceTransformer

q = json.load(open("processed_hotpot_q1000/hotpot_queries.json", encoding="utf-8"))
texts = [x["query_concat"] or x["query"] for x in q]
m = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")
E = m.encode(texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
E = np.asarray(E, dtype=np.float32)
np.save("sweep_exports/query_emb_q1000_raw.npy", E)
print("saved sweep_exports/query_emb_q1000_raw.npy", E.shape, "norm[0]=%.4f" % np.linalg.norm(E[0]))
