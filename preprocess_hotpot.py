#!/usr/bin/env python3
"""preprocess_hotpot.py — HotpotQA (multi-hop) → local corpus/queries JSON.

Mirrors preprocess_cuad.py so the rest of the pipeline consumes the same shape.
Difference from CUAD: a question's answer is spread over MULTIPLE gold paragraphs
that come from DIFFERENT documents. So instead of a single ``ref_doc_id`` we emit
``gold_support`` = the set of corpus indices of the paragraphs marked as
supporting facts. This is what makes the retrieval genuinely cross-document and
lets us score supporting-passage recall@k (the metric where retrieval diversity
helps).

Corpus construction (tractable, standard for RAG-retrieval eval):
  Pool every paragraph appearing in any question's ``context`` (gold + distractors),
  dedupe by paragraph title, and index paragraph-level. Each paragraph's text is the
  title followed by its concatenated sentences.

Input: the standard HotpotQA json (a list of records), e.g. the dev distractor or
fullwiki file from https://hotpotqa.github.io/ :
  {
    "_id", "question", "answer", "type", "level",
    "context": [ [title, [sent0, sent1, ...]], ... ],
    "supporting_facts": [ [title, sent_idx], ... ]
  }

Output (into ./processed_hotpot/):
  hotpot_corpus.json  : List[str]                 # one string per paragraph
  hotpot_queries.json : List[dict] with keys
      query, query_concat, answers, gold_support (List[int]), candidate_pool (List[int]
      = this question's own ~10 context paragraphs, for the distractor retrieval scope),
      gold_titles (List[str]), qtype, level, ref_doc_id (= gold_support[0], compat only)

Usage:
  python preprocess_hotpot.py --input hotpot_dev_distractor_v1.json \
      --out_dir processed_hotpot [--max_questions N]
"""
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Tuple


def _norm_title(t: str) -> str:
    """Stable key for de-duplicating paragraphs across questions."""
    return re.sub(r"\s+", " ", str(t or "").strip()).lower()


def _paragraph_text(title: str, sentences: List[str]) -> str:
    body = " ".join(s.strip() for s in (sentences or []) if str(s).strip())
    title = str(title or "").strip()
    return f"{title}. {body}".strip() if title else body


def build_hotpot(records: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Return (corpus, queries). Pure transform — no model, deterministic."""
    corpus: List[str] = []
    title_to_idx: Dict[str, int] = {}

    def _add_paragraph(title: str, sentences: List[str]) -> int:
        key = _norm_title(title)
        if key in title_to_idx:
            return title_to_idx[key]
        idx = len(corpus)
        corpus.append(_paragraph_text(title, sentences))
        title_to_idx[key] = idx
        return idx

    queries: List[Dict[str, Any]] = []
    for rec in records:
        context = rec.get("context") or []
        # register every paragraph of this question into the shared corpus.
        # candidate_pool preserves THIS question's own context paragraphs (the
        # 2 gold + ~8 distractors) as corpus indices — the distractor-setting
        # retrieval scope restricts candidates to exactly this set.
        local_title_idx: Dict[str, int] = {}
        candidate_pool: List[int] = []
        for para in context:
            if not isinstance(para, (list, tuple)) or len(para) < 2:
                continue
            title, sents = para[0], para[1]
            idx = _add_paragraph(title, sents)
            local_title_idx[_norm_title(title)] = idx
            if idx not in candidate_pool:
                candidate_pool.append(idx)

        # gold supporting paragraphs = distinct titles cited in supporting_facts
        gold_titles_raw = []
        for sf in (rec.get("supporting_facts") or []):
            if isinstance(sf, (list, tuple)) and sf:
                gold_titles_raw.append(sf[0])
        gold_titles = list(dict.fromkeys(str(t).strip() for t in gold_titles_raw if str(t).strip()))
        gold_support = sorted({
            local_title_idx[_norm_title(t)]
            for t in gold_titles
            if _norm_title(t) in local_title_idx
        })

        q = str(rec.get("question", "")).strip()
        queries.append({
            "query": q,
            "query_concat": q,                 # no aspect decomposition for hotpot
            "answers": str(rec.get("answer", "")).strip(),
            "gold_support": gold_support,       # corpus indices of gold paragraphs
            "candidate_pool": sorted(candidate_pool),  # this question's own 10 paras (distractor scope)
            "gold_titles": gold_titles,
            "qtype": rec.get("type"),
            "level": rec.get("level"),
            "ref_doc_id": gold_support[0] if gold_support else 0,  # compat only
        })
    return corpus, queries


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="HotpotQA json (list of records)")
    ap.add_argument("--out_dir", default="processed_hotpot")
    ap.add_argument("--max_questions", type=int, default=None)
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError("HotpotQA input must be a JSON list of records")
    if args.max_questions:
        records = records[: args.max_questions]

    corpus, queries = build_hotpot(records)

    # sanity: every query should have >=2 gold paragraphs (multi-hop); warn otherwise
    n_multi = sum(1 for q in queries if len(q["gold_support"]) >= 2)
    n_empty = sum(1 for q in queries if not q["gold_support"])
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "hotpot_corpus.json"), "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False)
    with open(os.path.join(args.out_dir, "hotpot_queries.json"), "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False)

    print(f"[hotpot] questions={len(queries)}  corpus_paragraphs={len(corpus)}")
    print(f"[hotpot] multi-hop (>=2 gold paras)={n_multi}/{len(queries)} "
          f"| empty-gold={n_empty} (should be 0)")
    print(f"[hotpot] wrote {args.out_dir}/hotpot_corpus.json, hotpot_queries.json")


if __name__ == "__main__":
    main()
