#!/usr/bin/env python3
"""
Back-fill the prompt cache from generations that already exist on disk.

Why
---
``cache/gen_llama3-8b_{fp16,int8,int4}.sqlite`` hold 2,996 rows each -- all of
them from the *cross* sweep (run 20260813_162948). The 135-combo *distractor*
run (20260816_210722) was generated without ``--cache``, so its 27,000
generations per quantization are on disk in ``llm_out_*/combos/*.json`` but
invisible to the cache.

Every combo record stores the exact prompt it was generated from
(``prompt.<arm>.system_prompt_main`` / ``user_prompt_main``) alongside the
answer. Decoding is greedy, so re-inserting (prompt -> answer) into the cache is
the same lossless memoization ``gen_cache`` would have written at generation
time -- not an approximation.

Effect on the n=1000 distractor run: the n=100 query set is
``random.sample(all_1000, 100, seed=42)``, i.e. a subset of the 1000, so those
~100 queries at lambda=0.7 tau=-1.0 gamma=0.7 top_k=5 come back as exact cache
hits instead of being regenerated.

Usage
-----
    python scripts/prime_gen_cache.py \
        --combos_dir llm_out_int4/combos --cache cache/gen_llama3-8b_int4.sqlite

    # check first, write nothing
    python scripts/prime_gen_cache.py ... --dry_run

One cache DB per model+quantization -- the key does not encode either.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sqlite3
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts import gen_cache  # noqa: E402

_SCHEMA = """
CREATE TABLE IF NOT EXISTS generations (
    key        TEXT PRIMARY KEY,
    answer     TEXT NOT NULL,
    created_at REAL NOT NULL DEFAULT (strftime('%s','now'))
);
"""


def iter_pairs(combos_dir: str):
    """Yield (key, answer) for every (query, arm) in every combo file."""
    files = sorted(glob.glob(os.path.join(combos_dir, "llm_combo_*.json")))
    if not files:
        raise SystemExit(f"no llm_combo_*.json under {combos_dir}")
    for path in files:
        try:
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  [skip] {os.path.basename(path)}: {exc}")
            continue
        for rec in payload.get("per_query_results") or []:
            prompts = rec.get("prompt") or {}
            for arm in ("l2", "mmr"):
                meta = prompts.get(arm) or {}
                system = meta.get("system_prompt_main")
                user = meta.get("user_prompt_main")
                answer = rec.get(f"answer_{arm}")
                # Only exact, complete records are cacheable. A run that wrote
                # prompt paths instead of prompt text is not usable here.
                if not isinstance(system, str) or not isinstance(user, str):
                    continue
                if not isinstance(answer, str) or not answer:
                    continue
                yield gen_cache.make_key(system, user, ARGS.max_new_tokens), answer
        yield None, os.path.basename(path)  # progress marker


def main() -> None:
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--combos_dir", nargs="+", required=True)
    ap.add_argument("--cache", required=True, help="sqlite path, one per model+quantization")
    ap.add_argument("--max_new_tokens", type=int, default=512,
                    help="must match the value the answers were generated with "
                         "(512 = llm_generator / run_generation_sweep default)")
    ap.add_argument("--dry_run", action="store_true")
    ARGS = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(ARGS.cache)) or ".", exist_ok=True)
    conn = sqlite3.connect(ARGS.cache, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.executescript(_SCHEMA)
    conn.commit()
    before = conn.execute("SELECT COUNT(*) FROM generations").fetchone()[0]

    seen, new, dup, files = set(), 0, 0, 0
    batch = []
    for key, answer in (p for d in ARGS.combos_dir for p in iter_pairs(d)):
        if key is None:
            files += 1
            print(f"  [{files:>4}] {answer}", end="\r", flush=True)
            continue
        if key in seen:
            dup += 1
            continue
        seen.add(key)
        row = conn.execute(
            "SELECT 1 FROM generations WHERE key = ?", (key,)
        ).fetchone()
        if row:
            dup += 1
            continue
        batch.append((key, answer))
        new += 1
        if len(batch) >= 500 and not ARGS.dry_run:
            conn.executemany(
                "INSERT OR IGNORE INTO generations (key, answer) VALUES (?, ?)", batch
            )
            conn.commit()
            batch.clear()
    if batch and not ARGS.dry_run:
        conn.executemany(
            "INSERT OR IGNORE INTO generations (key, answer) VALUES (?, ?)", batch
        )
        conn.commit()

    after = conn.execute("SELECT COUNT(*) FROM generations").fetchone()[0]
    conn.close()
    print(" " * 60, end="\r")
    print(f"cache      : {ARGS.cache}")
    print(f"combo files: {files}")
    print(f"rows before: {before}")
    print(f"new        : {new}{'  (DRY RUN -- not written)' if ARGS.dry_run else ''}")
    print(f"already in : {dup}")
    print(f"rows after : {after}")


if __name__ == "__main__":
    main()
