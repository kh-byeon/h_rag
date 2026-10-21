#!/usr/bin/env python3
"""
Resumable driver for LLM generation over a full sweep_eval run.

Problems this works around
--------------------------
1. ``llm_generator.py`` opens the detail JSONL with mode ``"w"``, so a restart
   truncates everything written so far. This driver moves any existing detail
   file aside (``*.part<N>.jsonl``) before each attempt, so no attempt clobbers
   an earlier one.
2. The normalized JSON is written only after the *last* combo, so a run that
   dies mid-sweep produces none. The per-combo ``combos/llm_combo_*.json`` files
   are flushed as each combo finishes -- this driver uses their presence as the
   resume marker, and ``scripts/collect_em_f1.py --combos_dir`` can score them
   directly.
3. A naive sweep regenerates identical prompts thousands of times. With
   ``--cache`` the driver points ``LLM_GEN_CACHE`` at a sqlite memo (greedy
   decoding => exact), which on the 108-combo HotpotQA run cuts 21,600
   generations to ~4,259.

The model is loaded once per attempt and every still-missing combo is passed in
a single ``--target_indices`` call, so resuming does not pay repeated 8B load
costs.

Usage
-----
    python scripts/run_generation_sweep.py \
        --retrieval_jsonl sweep_exports/<run_id>/retrieval_<run_id>.jsonl \
        --llm_output_root llm_out_fp16 \
        --quantization fp16 \
        --cache cache/gen_llama3-8b_fp16.sqlite

    # dry run: show what would execute, touch nothing
    python scripts/run_generation_sweep.py ... --dry_run
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from typing import Dict, List, Optional, Set

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Rough chars-per-token for English Wikipedia prose under the Llama-3 BPE.
# Only used for the pre-flight truncation warning, never for anything scored.
_CHARS_PER_TOKEN = 3.6


def scan_retrieval(retrieval_jsonl: str) -> Dict[int, Dict[str, object]]:
    """Per-sweep_idx: top_k plus the longest rendered prompt seen for that combo.

    The prompt length matters because ``llm_generator`` silently truncates the
    input at ``GENERATION_MAX_INPUT_LENGTH``. A combo whose prompts exceed that
    is not really running at its nominal top_k, and nothing in the pipeline says
    so -- hence the pre-flight report below.
    """
    sys.path.insert(0, ROOT)
    import types

    for m in ("torch", "transformers"):
        sys.modules.setdefault(m, types.ModuleType(m))
    import llm_generator as L  # noqa: N812

    info: Dict[int, Dict[str, object]] = {}
    with open(retrieval_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            idx = int(d["sweep_idx"])
            rec = info.setdefault(idx, {"top_k": int(d.get("top_k") or 0), "max_chars": 0})
            for arm in ("l2_contexts", "mmr_contexts"):
                ctx = d.get(arm) or []
                n = len(L.build_generation_user_content_hotpot(d.get("query", ""), ctx))
                if n > int(rec["max_chars"]):
                    rec["max_chars"] = n
    return info


def sweep_indices(retrieval_jsonl: str) -> List[int]:
    seen: Set[int] = set()
    with open(retrieval_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                seen.add(int(json.loads(line)["sweep_idx"]))
    return sorted(seen)


def preflight(info: Dict[int, Dict[str, object]], max_input_tokens: int) -> Set[int]:
    """Print a per-top_k prompt-length table; return the truncating sweep_idx set."""
    by_k: Dict[int, List[int]] = {}
    for idx, rec in info.items():
        by_k.setdefault(int(rec["top_k"]), []).append(int(rec["max_chars"]))
    truncating: Set[int] = set()
    print(f"\n[preflight] GENERATION_MAX_INPUT_LENGTH = {max_input_tokens} tokens")
    print(f"[preflight] {'top_k':>6} {'combos':>7} {'max chars':>10} {'~max tok':>9}  status")
    for k in sorted(by_k):
        mx = max(by_k[k])
        tok = mx / _CHARS_PER_TOKEN
        bad = tok > max_input_tokens
        print(f"[preflight] {k:>6} {len(by_k[k]):>7} {mx:>10} {tok:>9.0f}  "
              f"{'TRUNCATED -- see warning' if bad else 'ok'}")
        if bad:
            truncating |= {i for i, r in info.items() if int(r["top_k"]) == k}
    if truncating:
        ks = sorted({int(info[i]['top_k']) for i in truncating})
        print(f"\n[preflight] WARNING: top_k={ks} exceeds the {max_input_tokens}-token input cap.")
        print("[preflight] Those combos silently drop trailing passages, so they do NOT")
        print("[preflight] measure their nominal top_k. Exclude them from the paper table,")
        print("[preflight] or re-run them with --skip_truncating to leave them out entirely.")
    return truncating


def done_indices(out_root: str) -> Set[int]:
    import glob
    import re

    out: Set[int] = set()
    for p in glob.glob(os.path.join(out_root, "combos", "llm_combo_*.json")):
        m = re.search(r"llm_combo_(\d+)\.json$", os.path.basename(p))
        if not m:
            continue
        try:
            with open(p, encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue  # truncated by a kill -9; regenerate it
        if payload.get("per_query_results"):
            out.add(int(m.group(1)))
    return out


def preserve_detail(out_root: str, run_id: str) -> Optional[str]:
    """Move an existing detail JSONL aside so the next attempt cannot truncate it."""
    detail = os.path.join(out_root, f"llm_generation_detail_{run_id}.jsonl")
    if not os.path.isfile(detail) or os.path.getsize(detail) == 0:
        return None
    n = 1
    while os.path.exists(os.path.join(out_root, f"llm_generation_detail_{run_id}.part{n}.jsonl")):
        n += 1
    dest = os.path.join(out_root, f"llm_generation_detail_{run_id}.part{n}.jsonl")
    os.replace(detail, dest)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval_jsonl", required=True)
    ap.add_argument("--llm_output_root", required=True)
    ap.add_argument("--quantization", default="fp16", choices=["fp16", "int8", "int4"])
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--domain", default="hotpot", choices=["cuad", "hotpot"])
    ap.add_argument("--cache", default=None,
                    help="sqlite path for the lossless prompt cache (recommended). "
                         "Must be per model+quantization.")
    ap.add_argument("--chunk", type=int, default=0,
                    help="max combos per attempt, one process each (0 = all remaining). "
                         "This bounds host RAM (llm_generator accumulates every combo's "
                         "normalized records in memory) and the blast radius of a crash. "
                         "It does NOT lower peak GPU memory -- that is set by the single "
                         "longest prompt and is already hard-capped by "
                         "GENERATION_MAX_INPUT_LENGTH.")
    ap.add_argument("--order_by_top_k", action="store_true", default=True,
                    help="run cheap (small top_k) combos first (default: on). Fails fast "
                         "and warms the prompt cache before the expensive combos.")
    ap.add_argument("--no_order_by_top_k", dest="order_by_top_k", action="store_false")
    ap.add_argument("--skip_truncating", action="store_true",
                    help="skip combos whose prompts exceed the input cap (see preflight)")
    ap.add_argument("--max_attempts", type=int, default=20)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(args.retrieval_jsonl):
        raise SystemExit(f"not found: {args.retrieval_jsonl}")
    run_id = os.path.basename(args.retrieval_jsonl)
    run_id = run_id[len("retrieval_"):-len(".jsonl")] if run_id.startswith("retrieval_") else run_id

    all_idx = sweep_indices(args.retrieval_jsonl)
    out_root = os.path.abspath(args.llm_output_root)
    if not args.dry_run:
        os.makedirs(out_root, exist_ok=True)
    print(f"[driver] run_id      : {run_id}")
    print(f"[driver] combos      : {len(all_idx)}  ({all_idx[0]}..{all_idx[-1]})")
    print(f"[driver] output root : {out_root}")

    env = dict(os.environ)
    if args.cache:
        cache_abs = os.path.abspath(args.cache)
        if not args.dry_run:
            os.makedirs(os.path.dirname(cache_abs) or ".", exist_ok=True)
        env["LLM_GEN_CACHE"] = cache_abs
        print(f"[driver] cache       : {cache_abs}")
    else:
        env.pop("LLM_GEN_CACHE", None)
        print("[driver] cache       : DISABLED (pass --cache to enable)")

    info = scan_retrieval(args.retrieval_jsonl)
    sys.path.insert(0, ROOT)
    import llm_generator as _L  # torch/transformers already stubbed by scan_retrieval

    truncating = preflight(info, int(_L.GENERATION_MAX_INPUT_LENGTH))
    if args.skip_truncating and truncating:
        all_idx = [i for i in all_idx if i not in truncating]
        print(f"[driver] --skip_truncating: {len(truncating)} combo(s) excluded, "
              f"{len(all_idx)} remain")

    def order(idx_list: List[int]) -> List[int]:
        if not args.order_by_top_k:
            return idx_list
        return sorted(idx_list, key=lambda i: (int(info[i]["top_k"]), i))

    for attempt in range(1, args.max_attempts + 1):
        done = done_indices(out_root)
        todo = order([i for i in all_idx if i not in done])
        print(f"\n[driver] attempt {attempt}: {len(done)} done, {len(todo)} remaining")
        if not todo:
            print("[driver] all combos complete.")
            break
        batch = todo[: args.chunk] if args.chunk > 0 else todo

        cmd = [
            args.python, os.path.join(ROOT, "llm_generator.py"),
            "--retrieval_jsonl", os.path.abspath(args.retrieval_jsonl),
            "--llm_output_root", out_root,
            "--quantization", args.quantization,
            "--model", args.model,
            "--max_new_tokens", str(args.max_new_tokens),
            "--domain", args.domain,
            "--output_normalized_json",
            os.path.join(out_root, f"generation_normalized_{args.quantization}.json"),
            "--target_indices", *[str(i) for i in batch],
        ]
        print(f"[driver] running {len(batch)} combo(s): {batch[:8]}{' ...' if len(batch) > 8 else ''}")
        if args.dry_run:
            print("[driver] DRY RUN:", " ".join(shlex.quote(c) for c in cmd))
            break

        moved = preserve_detail(out_root, run_id)
        if moved:
            print(f"[driver] preserved previous detail -> {os.path.basename(moved)}")

        rc = subprocess.call(cmd, cwd=ROOT, env=env)
        if rc != 0:
            print(f"[driver] llm_generator exited {rc}; completed combos are kept. "
                  f"Re-run this command to resume.", file=sys.stderr)
            if len(done_indices(out_root)) == len(done):
                print("[driver] no progress this attempt -- stopping to avoid a loop.",
                      file=sys.stderr)
                sys.exit(rc)
    else:
        print(f"[driver] hit --max_attempts={args.max_attempts} with combos still missing.",
              file=sys.stderr)

    if args.cache and not args.dry_run:
        sys.path.insert(0, ROOT)
        try:
            from scripts import gen_cache
            print(f"[driver] cache stats: {gen_cache.stats(env['LLM_GEN_CACHE'])}")
        except (ImportError, OSError):
            pass

    print("\n[driver] next: score the combos (CPU only, no model needed)")
    print(f"  python scripts/collect_em_f1.py --combos_dir "
          f"{os.path.relpath(out_root, ROOT)}/combos:{args.quantization} "
          f"--out figures/em_f1_{args.quantization}.csv")


if __name__ == "__main__":
    main()
