#!/usr/bin/env python3
"""Dump the query embeddings of the n = 1,000 HotpotQA build exactly as the pipeline encodes them.

The pipeline does not encode the raw question: _prepare_cuad_search_text() first prepends the
first double-quoted phrase as a keyword (126 of the 1,000 questions contain one), and
step1_query_in_and_encode() then encodes that search text one query at a time. This script
reproduces both steps, so a run can later replay retrieval without the encoder
(tools: --query_emb_map). Output: sweep_exports/query_emb_q1000_searchtext.npz with
'texts' (the exact encoder inputs), 'questions' and 'emb' (1000, 768) float32.

Run inside WSL with the project venv active (any working directory):
    python scripts/dump_search_text_embeddings.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)  # paths below are relative to the repo root, wherever the script is started
sys.path.insert(0, ROOT)
from online_accelerator_device import L2RAGOnlinePipeline as P, DOMAIN_EMBEDDING_MODEL_MAP  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

src = "processed_hotpot_q1000/hotpot_queries.json"
if not os.path.isfile(src):
    src = "processed_hotpot/hotpot_queries.json"
q = json.load(open(src, encoding="utf-8"))
assert len(q) == 1000, f"{src} holds {len(q)} questions, expected 1000"
questions = [x["query"] for x in q]
texts = [P._prepare_cuad_search_text(P, t)[0] for t in questions]
m = SentenceTransformer(DOMAIN_EMBEDDING_MODEL_MAP["cuad"])
emb = np.stack([
    np.asarray(m.encode(t, convert_to_numpy=True, normalize_embeddings=True), dtype=np.float32).flatten()
    for t in texts
])
os.makedirs("sweep_exports", exist_ok=True)
out = "sweep_exports/query_emb_q1000_searchtext.npz"
np.savez(out, texts=np.array(texts, dtype=object), questions=np.array(questions, dtype=object), emb=emb)
print(f"saved {out}: {emb.shape}, {sum(a != b for a, b in zip(texts, questions))} search texts differ from the question")
