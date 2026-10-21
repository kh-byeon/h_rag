#!/usr/bin/env python3
"""CLI: AIMDC scaler distribution (query post-OPQ vs PQ-reconstruction)."""
from __future__ import annotations

import argparse
import os
import sys


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    root = _repo_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    from aimdc_scaler_diagnosis import print_diagnosis_stdout, run_diagnosis

    p = argparse.ArgumentParser(description="Diagnose AIMDC percentile scaler fit vs inference.")
    p.add_argument("--offline_dir", required=True, help="Offline bundle (raw_embeddings, PQ book, postings).")
    p.add_argument("--out_dir", required=True, help="Output directory for summary.json + PNGs.")
    p.add_argument(
        "--sample_size",
        type=int,
        default=20_000,
        help="Max rows per stream for stats/plots (0 = use all; bounded by memory budget).",
    )
    p.add_argument("--rng_seed", type=int, default=0, help="Seed for subsampling (query vs PQ use seed and seed+1).")
    args = p.parse_args()
    sample = None if int(args.sample_size) == 0 else int(args.sample_size)
    summary = run_diagnosis(
        args.offline_dir,
        args.out_dir,
        sample_size=sample,
        rng_seed=int(args.rng_seed),
    )
    print_diagnosis_stdout(summary, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
