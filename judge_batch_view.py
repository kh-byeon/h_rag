#!/usr/bin/env python3
"""Print a compact judge-ready view of a slice of a normalized generation JSON.

Used to feed the LLM-as-a-judge (Claude) one batch of queries at a time:
    python judge_batch_view.py --gen gen_top5_int8.json --start 0 --count 5

Shows, per query: question, ground_truth, L2 answer, MMR answer (truncated).
"""
from __future__ import annotations

import argparse
import json


def _ans(rec, arm):
    g = rec.get("generation") or {}
    a = (g.get(arm) or {}).get("answer")
    if a is None:
        a = rec.get(f"refined_answer_{arm}") or rec.get(f"answer_{arm}")
    return str(a or "").replace("\n", " ")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True, help="normalized generation JSON")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--chars", type=int, default=1100, help="truncate answers to N chars")
    args = ap.parse_args()

    d = json.load(open(args.gen, encoding="utf-8"))
    n = len(d)
    end = min(args.start + args.count, n)
    print(f"[judge_batch] file={args.gen}  total={n}  showing [{args.start}:{end}]\n")
    for i in range(args.start, end):
        r = d[i]
        q = r.get("rewritten_query") or r.get("query")
        gt = str(r.get("ground_truth", "")).replace("\n", " ")
        print("#" * 88)
        print(f"Q{i} || {q}")
        print(f"GT  || {gt[:500]}")
        print(f"L2  || {_ans(r, 'l2')[:args.chars]}")
        print(f"MMR || {_ans(r, 'mmr')[:args.chars]}")
        print()


if __name__ == "__main__":
    main()
