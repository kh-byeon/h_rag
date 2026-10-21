#!/usr/bin/env python3
"""download_hotpot.py — fetch HotpotQA from the HuggingFace mirror and dump it to
the raw list-of-records JSON that preprocess_hotpot.py expects.

Why this exists: the official CMU host (curtis.ml.cmu.edu) is frequently down. The
HF dataset ``hotpotqa/hotpot_qa`` carries the same data but in a COLUMNAR layout
(context = {title:[...], sentences:[[...]]}, supporting_facts = {title:[...],
sent_id:[...]}). This script reshapes each record into the canonical
  context:          [[title, [sent0, sent1, ...]], ...]
  supporting_facts: [[title, sent_id], ...]
so the downstream loader stays single-format.

Usage:
  pip install datasets
  python download_hotpot.py --config distractor --split validation \
      --out hotpot_dev_distractor_v1.json
  python preprocess_hotpot.py --input hotpot_dev_distractor_v1.json --out_dir processed_hotpot
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List


def _reshape_context(ctx: Any) -> List[list]:
    """Columnar {title:[...], sentences:[[...]]} OR already-paired -> [[title,[sents]]]."""
    if isinstance(ctx, dict):
        titles = ctx.get("title") or []
        sentences = ctx.get("sentences") or []
        return [[titles[i], list(sentences[i])] for i in range(min(len(titles), len(sentences)))]
    # already list of [title, [sents]]
    out = []
    for para in ctx or []:
        if isinstance(para, (list, tuple)) and len(para) >= 2:
            out.append([para[0], list(para[1])])
    return out


def _reshape_support(sf: Any) -> List[list]:
    """Columnar {title:[...], sent_id:[...]} OR already-paired -> [[title, sent_id]]."""
    if isinstance(sf, dict):
        titles = sf.get("title") or []
        sids = sf.get("sent_id", sf.get("sent_ids")) or []
        return [[titles[i], int(sids[i])] for i in range(min(len(titles), len(sids)))]
    out = []
    for pair in sf or []:
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            out.append([pair[0], int(pair[1])])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="distractor",
                    choices=["distractor", "fullwiki"], help="HotpotQA HF config")
    ap.add_argument("--split", default="validation",
                    help="validation (=dev) or train")
    ap.add_argument("--out", default="hotpot_dev_distractor_v1.json")
    ap.add_argument("--max_questions", type=int, default=None)
    args = ap.parse_args()

    from datasets import load_dataset  # imported here so --help works without the dep

    print(f"[download] load_dataset('hotpotqa/hotpot_qa', '{args.config}', split='{args.split}')")
    try:
        ds = load_dataset("hotpotqa/hotpot_qa", args.config, split=args.split)
    except Exception:
        # older datasets versions need trust_remote_code for this dataset script
        ds = load_dataset("hotpotqa/hotpot_qa", args.config, split=args.split,
                          trust_remote_code=True)

    if args.max_questions:
        ds = ds.select(range(min(args.max_questions, len(ds))))

    records: List[Dict[str, Any]] = []
    for ex in ds:
        records.append({
            "_id": ex.get("id") or ex.get("_id"),
            "question": ex.get("question", ""),
            "answer": ex.get("answer", ""),
            "type": ex.get("type"),
            "level": ex.get("level"),
            "context": _reshape_context(ex.get("context")),
            "supporting_facts": _reshape_support(ex.get("supporting_facts")),
        })

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

    n_multi = sum(1 for r in records
                  if len({t for t, _ in r["supporting_facts"]}) >= 2)
    print(f"[download] wrote {args.out}  records={len(records)}  "
          f"multi-hop(>=2 gold titles)={n_multi}")
    print("[download] next: python preprocess_hotpot.py --input "
          f"{args.out} --out_dir processed_hotpot")


if __name__ == "__main__":
    main()
