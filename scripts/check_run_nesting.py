#!/usr/bin/env python3
"""Is the larger run a strict superset of the smaller one?

    python scripts/check_run_nesting.py <small_run_dir> <large_run_dir>

`preprocess_hotpot.py --max_questions N` takes ``records[:N]``, so the first 1000
questions of a 3500-question build are the same questions in the same order. But
the index is rebuilt over a ~3.5x larger corpus -- PQ codebooks and the OPQ
rotation are refitted -- so retrieval can move even though the questions do not.

Checks, in order of severity:
  1. query TEXT matches position-by-position over the overlap
  2. gold_support matches
  3. per-query CHURN: is the top-5 SET the same query by query?

Check 3 is the decisive one. An earlier version of this script gated on the
AGGREGATE full@5 drift and wrongly reported "NESTED" for a pair of runs whose L2
top-5 set differed on 71% of queries -- the gains and losses cancelled, leaving
the marginal rate almost unchanged. Never judge nesting from a marginal rate.

Exit 0 = nested, 1 = not nested.
"""
from __future__ import annotations

import glob
import json
import os
import sys

CHURN_THRESHOLD = 0.95   # L2 top-5 sets must agree on at least this fraction


def load(run_dir):
    hits = glob.glob(os.path.join(run_dir, "retrieval_*.jsonl"))
    if not hits:
        raise SystemExit(f"no retrieval_*.jsonl in {run_dir}")
    out = {}
    with open(hits[0], encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            gs = sorted(int(g) for g in (d.get("gold_support") or []))
            ids = d.get("retrieved_doc_ids") or {}
            k = int(d.get("top_k") or 5)
            rec = {"query": d.get("query", ""), "gold": gs}
            for arm in ("l2", "mmr"):
                got = [int(x) for x in (ids.get(arm) or [])][:k]
                rec[f"{arm}_top"] = got
                rec[f"{arm}_full"] = int(bool(gs) and all(g in got for g in gs))
            out[int(d["query_idx"])] = rec
    return out


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    small, large = load(sys.argv[1]), load(sys.argv[2])
    overlap = sorted(set(small) & set(large))
    print(f"small run: {len(small)} queries    large run: {len(large)} queries    "
          f"overlapping query_idx: {len(overlap)}")
    if not overlap:
        print("FAIL: no overlapping query_idx at all")
        return 1

    bad_text = [i for i in overlap if small[i]["query"] != large[i]["query"]]
    bad_gold = [i for i in overlap if small[i]["gold"] != large[i]["gold"]]
    print(f"\n1. query text identical   : {len(overlap)-len(bad_text)}/{len(overlap)}")
    print(f"2. gold_support identical : {len(overlap)-len(bad_gold)}/{len(overlap)}")
    if bad_text:
        print(f"   FAIL -- first mismatches at query_idx {bad_text[:5]}")
        print("   The runs index different questions at the same positions.")
        return 1
    if bad_gold:
        print("   FAIL -- gold_support drifted; preprocessing is not positional.")
        return 1

    print()
    churn = {}
    for arm in ("l2", "mmr"):
        same_set = sum(set(small[i][f"{arm}_top"]) == set(large[i][f"{arm}_top"]) for i in overlap)
        same_full = sum(small[i][f"{arm}_full"] == large[i][f"{arm}_full"] for i in overlap)
        churn[arm] = same_set / len(overlap)
        print(f"3. {arm.upper():>3} top-5 SET identical       : {same_set:>5}/{len(overlap)} "
              f"({100*same_set/len(overlap):5.1f}%)")
        print(f"   {arm.upper():>3} full-support outcome same : {same_full:>5}/{len(overlap)} "
              f"({100*same_full/len(overlap):5.1f}%)")

    l2s = sum(small[i]["l2_full"] for i in overlap) / len(overlap)
    l2l = sum(large[i]["l2_full"] for i in overlap) / len(overlap)
    ms = sum(small[i]["mmr_full"] for i in overlap) / len(overlap)
    ml = sum(large[i]["mmr_full"] for i in overlap) / len(overlap)
    print(f"\n   full@5 over the overlap: L2 {l2s:.4f} -> {l2l:.4f}   MMR {ms:.4f} -> {ml:.4f}")
    print("   (a stable aggregate does NOT imply stable per-query results -- see above)")

    if churn["l2"] < CHURN_THRESHOLD:
        print(f"\nNOT NESTED: the L2 top-5 set is identical on only "
              f"{100*churn['l2']:.1f}% of the overlapping queries.")
        print("The L2 arm is a pure function of (index, query), so this is the rebuilt")
        print("index retrieving differently -- not the algorithm.")
        print("\nBoth runs remain internally valid. But they measure different retrieval")
        print("systems, so:")
        print("  * report the larger run STANDALONE; do not call the smaller one a subset")
        print("  * never join the two on query_idx")
        print("  * do not attribute the change in effect size between them to sample size")
        print("    alone -- part of it is the index")
        return 1

    print(f"\nNESTED: same questions, same gold, L2 top-5 set stable on "
          f"{100*churn['l2']:.1f}% of the overlap.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
