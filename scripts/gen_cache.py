#!/usr/bin/env python3
"""
Disk-backed generation cache for the sweep (lossless, opt-in).

Why this is safe
----------------
``llm_generator.generate_answer`` decodes with ``do_sample=False`` (greedy),
fixed ``max_new_tokens`` and a fixed model/quantization. Greedy decoding is
deterministic, so an identical prompt always yields an identical string. Caching
on the prompt is therefore an exact memoization, not an approximation.

Why it matters
--------------
In a lambda x tau x gamma x top_k sweep the L2 arm does not depend on
lambda/tau/gamma at all, and many MMR configurations select the same chunks.
Measured on the 108-combo HotpotQA run:

    naive     : 21,600 generations
    deduped   :  4,259 generations   (L2 400, MMR 3,859)  -> 80.3% saved

Enable by setting ``LLM_GEN_CACHE`` to a sqlite path before running
``llm_generator.py``. Unset => the cache is bypassed entirely and behaviour is
byte-identical to before.

The DB path should encode model + quantization (e.g. ``cache_llama3-8b_fp16.sqlite``);
the key does not include them, so never share one DB across models.

CLI
---
    python scripts/gen_cache.py stats  cache.sqlite
    python scripts/gen_cache.py clear  cache.sqlite
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import sys
import threading
from typing import Optional

_LOCK = threading.Lock()
_CONN: Optional[sqlite3.Connection] = None
_CONN_PATH: Optional[str] = None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS generations (
    key        TEXT PRIMARY KEY,
    answer     TEXT NOT NULL,
    created_at REAL NOT NULL DEFAULT (strftime('%s','now'))
);
"""


def cache_path() -> Optional[str]:
    """Path from ``LLM_GEN_CACHE``; None (disabled) when unset/empty."""
    p = os.environ.get("LLM_GEN_CACHE", "").strip()
    return p or None


def _connect(path: str) -> sqlite3.Connection:
    global _CONN, _CONN_PATH
    if _CONN is not None and _CONN_PATH == path:
        return _CONN
    if _CONN is not None:
        try:
            _CONN.close()
        except sqlite3.Error:
            pass
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0, check_same_thread=False)
    # WAL: a crashed run keeps everything already committed.
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.executescript(_SCHEMA)
    conn.commit()
    _CONN, _CONN_PATH = conn, path
    return conn


def make_key(system_prompt: str, user_prompt: str, max_new_tokens: int) -> str:
    h = hashlib.sha256()
    for part in (str(system_prompt or ""), str(user_prompt or ""), str(int(max_new_tokens))):
        h.update(part.encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()


def get(key: str) -> Optional[str]:
    path = cache_path()
    if not path:
        return None
    with _LOCK:
        try:
            cur = _connect(path).execute(
                "SELECT answer FROM generations WHERE key = ?", (key,)
            )
            row = cur.fetchone()
        except sqlite3.Error:
            return None
    return row[0] if row else None


def put(key: str, answer: str) -> None:
    path = cache_path()
    if not path:
        return
    with _LOCK:
        try:
            conn = _connect(path)
            conn.execute(
                "INSERT OR REPLACE INTO generations (key, answer) VALUES (?, ?)",
                (key, str(answer)),
            )
            conn.commit()
        except sqlite3.Error:
            pass  # cache must never break a run


def stats(path: str) -> dict:
    conn = sqlite3.connect(path)
    try:
        n = conn.execute("SELECT COUNT(*) FROM generations").fetchone()[0]
        b = conn.execute("SELECT COALESCE(SUM(LENGTH(answer)), 0) FROM generations").fetchone()[0]
    finally:
        conn.close()
    return {"entries": n, "answer_bytes": b,
            "db_bytes": os.path.getsize(path) if os.path.exists(path) else 0}


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    cmd, path = sys.argv[1], sys.argv[2]
    if cmd == "stats":
        for k, v in stats(path).items():
            print(f"{k:14s} {v}")
    elif cmd == "clear":
        conn = sqlite3.connect(path)
        conn.execute("DELETE FROM generations")
        conn.commit()
        conn.close()
        print(f"cleared {path}")
    else:
        raise SystemExit(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()
