#!/usr/bin/env python3
"""End-to-end MMR parameter search driver (run on the LOCAL GPU machine).

For each candidate retrieval combo (the top winners from the retrieval sweep),
this drives the existing CLIs via subprocess:

  1) retrieval : sweep_eval.py  (single combo, --num_queries N, FeFET centroid +
                 IGZO-cap MMR dedup)          -> retrieval_<run>.jsonl + manifest
  2) generation: llm_generator.py per quant   -> gen_<tag>_<q>.json
  3) evaluation: run_eval_from_generation.py  -> summary.json  (dry-run judges:
                 bertscore + coverage only, fast)

It then (a) aggregates the automated dry-run metrics into a CSV, and (b) dumps a
compact JUDGE PACKET (question, ground_truth, L2 vs MMR answers) for the first
--judge_queries queries so Claude can score them as an LLM-as-a-judge.

Requires torch + transformers + bitsandbytes + the Llama model (generation) and
sentence-transformers (bertscore). This does NOT run in the Cowork sandbox.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from datetime import datetime

# --- Candidate combos: top winners from the retrieval sweep (edit as needed) ---
# (mmr_lambda, tau_dup, gamma, top_k, tag)
DEFAULT_COMBOS = [
    (0.7, -0.97, 0.2, 5,  "top5_best"),     # top5: recall+diversity both up
    (0.5, -0.97, 0.2, 10, "top10_best"),    # top10: max diversity, ~0 rouge loss
    (0.5, -1.00, 0.2, 20, "top20_best"),    # top20: recall+rouge+ild all up
]

HW_FLAGS = [
    "--centroid_mode", "nvm",
    "--centroid_fit_path", "h_aimdc_sim/fefet_ref_fit_smu.json",
    "--centroid_reduce", "linear_sum",
    "--centroid_v_ml", "1.4",
    "--mmr_mode", "capacitor",
    "--device_fit_path", "h_aimdc_sim/igzo_fit.json",
    "--device_scale_k_path", "h_aimdc_sim/igzo_fit.k.json",
    "--device_reduce", "sum",
]


def run(cmd: list[str]) -> None:
    print("＄ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def newest_manifest(sweep_root: str) -> str:
    dirs = sorted(glob.glob(os.path.join(sweep_root, "sweep_exports", "*")),
                  key=os.path.getmtime)
    if not dirs:
        raise FileNotFoundError("no sweep_exports/<run> produced")
    return os.path.join(dirs[-1], "run_manifest.json")


def dump_judge_packet(gen_json: str, n: int, out_path: str, tag: str, quant: str) -> list:
    recs = json.load(open(gen_json, encoding="utf-8"))
    rows = []
    for r in recs[:n]:
        g = r.get("generation") or {}
        rows.append({
            "combo": tag, "quantization": quant,
            "query": r.get("rewritten_query") or r.get("query"),
            "ground_truth": r.get("ground_truth"),
            "answer_l2": (g.get("l2") or {}).get("answer"),
            "answer_mmr": (g.get("mmr") or {}).get("answer"),
        })
    existing = json.load(open(out_path)) if os.path.isfile(out_path) else []
    existing.extend(rows)
    json.dump(existing, open(out_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    return rows


def parse_summary(summary_json: str) -> dict:
    s = json.load(open(summary_json, encoding="utf-8"))
    avg = s.get("pairwise_answer_averages") or s.get("averages") or {}
    return avg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num_queries", type=int, default=10,
                    help="queries per combo for retrieval/generation")
    ap.add_argument("--judge_queries", type=int, default=2,
                    help="how many queries to dump into the Claude judge packet")
    ap.add_argument("--quants", type=str, default="int8",
                    help="comma list of quantizations, e.g. 'int8' or 'fp16,int8,int4'")
    ap.add_argument("--model", type=str,
                    default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--full_judges", action="store_true",
                    help="run local LLM judges instead of --dry_run_judges")
    ap.add_argument("--out_dir", type=str, default="param_search_out")
    ap.add_argument("--root", type=str, default=".")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    quants = [q for q in args.quants.replace(" ", "").split(",") if q]
    judge_packet = os.path.join(args.out_dir, "judge_packet.json")
    if os.path.isfile(judge_packet):
        os.remove(judge_packet)
    agg_rows = []

    for (lam, tau, gam, tk, tag) in DEFAULT_COMBOS:
        print("\n" + "=" * 78 + f"\n[combo] {tag}: lambda={lam} tau={tau} gamma={gam} top_k={tk}\n" + "=" * 78, flush=True)
        # 1) retrieval (single combo)
        run([sys.executable, "sweep_eval.py", *HW_FLAGS,
             "--num_queries", str(args.num_queries),
             "--lambdas", str(lam), "--tau_dups", str(tau),
             "--gammas", str(gam), "--top_ks", str(tk),
             "--output_dir", args.root, "--domain", "cuad", "--verbose"])
        manifest = newest_manifest(args.root)

        for q in quants:
            gen_json = os.path.join(args.out_dir, f"gen_{tag}_{q}.json")
            eval_dir = os.path.join(args.out_dir, f"eval_{tag}_{q}")
            # 2) generation
            run([sys.executable, "llm_generator.py",
                 "--sweep_manifest", manifest, "--model", args.model,
                 "--quantization", q, "--output_normalized_json", gen_json])
            # dump judge packet (first N queries)
            dump_judge_packet(gen_json, args.judge_queries, judge_packet, tag, q)
            # 3) evaluation (dry-run by default)
            eval_cmd = [sys.executable, "run_eval_from_generation.py",
                        "--normalized_json", gen_json, "--output_dir", eval_dir,
                        "--compare_mode", "pairwise"]
            if not args.full_judges:
                eval_cmd.append("--dry_run_judges")
            run(eval_cmd)
            avg = parse_summary(os.path.join(eval_dir, "summary.json"))
            agg_rows.append({"combo": tag, "lambda": lam, "tau": tau,
                             "gamma": gam, "top_k": tk, "quant": q, "metrics": avg})

    out_json = os.path.join(args.out_dir, "param_search_summary.json")
    json.dump(agg_rows, open(out_json, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n[done] automated metrics -> {out_json}")
    print(f"[done] Claude judge packet ({args.judge_queries} queries/combo) -> {judge_packet}")
    print("Next: give judge_packet.json back to Claude for LLM-as-a-judge scoring.")


if __name__ == "__main__":
    main()
