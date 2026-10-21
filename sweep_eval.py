#!/usr/bin/env python3
"""
Grid search over mmr_lambda × tau_dup × gamma × top_k
for run_mmr_evaluation with retrieval_mode=COMPARE (intra-document-only L2 vs MMR).

After clause-level offline chunking, adjacent chunk stitching is disabled,
so window_size is fixed to 1 (no stitching).

Pass arguments directly to online_accelerator.run_mmr_evaluation
(equivalent to CLI: --top-k, --window-size 1, --gamma, --retrieval-mode COMPARE, ...).

Each combo's retrieved_context is saved as JSON/JSONL under
`output_dir/sweep_exports/<timestamp>/`, and CSV stores only combo JSON relative path
(combo_artifact).
"""
import argparse
import csv
import itertools
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

from online_accelerator_device import run_mmr_evaluation
from evaluation.metrics.supporting_recall import evaluate_supporting_recall, aggregate

LAMBDAS = [0.3 ]
# z-standardized tau gate (when --standardize_mmr_terms is on, the default for the
# analog/capacitor path): tau_dup is in per-query std units, NOT a raw device
# distance. 0 = mean candidate distance; more negative = only skip near-duplicates;
# less negative / toward 0 = dedup more aggressively.
# (Old raw-distance grids near -0.97 [-1,1] or -1.3 [-3,3] no longer apply.)
TAU_DUPS = [ -1.0]
# Intra-document MMR behavior sweep (gamma is clamped to [0, 1] in pipeline)
GAMMAS = [0.3]
# Final Top-K after retrieval/reranking (same meaning as online --top-k)
TOP_K_LIST = [10,20]
# No stitching after clause-level chunking: always 1 for compatibility column
FIXED_WINDOW_SIZE = 1
# Multi-hop supporting-passage recall budgets (HotpotQA). Blank for CUAD sweeps.
SUPPORT_KS = (2, 5, 10)


def _support_columns() -> List[str]:
    """Per-k supporting-recall CSV columns, built from SUPPORT_KS so they stay
    in sync. Order per k: l2_recall@k, mmr_recall@k, l2_full@k, mmr_full@k,
    full_delta@k. A single support_n column follows all of them.
    """
    cols: List[str] = []
    for k in SUPPORT_KS:
        cols.extend([
            f"l2_recall@{k}",
            f"mmr_recall@{k}",
            f"l2_full@{k}",
            f"mmr_full@{k}",
            f"full_delta@{k}",
        ])
    cols.append("support_n")
    return cols


CSV_HEADER = [
    "mmr_mode",
    "centroid_mode",
    "mmr_lambda",
    "tau_dup",
    "gamma",
    "top_k",
    "window_size",
    "l2_recall",
    "mmr_recall",
    "l2_rouge_l",
    "mmr_rouge_l",
    "l2_ild",
    "mmr_ild",
    "l2_rr",
    "mmr_rr",
    "l2_breadth",
    "mmr_breadth",
    *_support_columns(),
    "combo_artifact",
]


def _metric_float(d: Dict[str, Any], key: str) -> float:
    v = d.get(key, 0.0)
    if v is None:
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    global LAMBDAS, TAU_DUPS, GAMMAS, TOP_K_LIST
    parser = argparse.ArgumentParser(
        description=(
            "Grid search: mmr_lambda × tau_dup × gamma × top_k "
            "→ run_mmr_evaluation (intra-document COMPARE: baseline L2 vs proposed MMR); "
            "window_size=1 fixed (no stitching)"
        )
    )
    parser.add_argument("--domain", type=str, default="cuad")
    parser.add_argument(
        "--retrieval_scope",
        type=str,
        default="intra",
        choices=["intra", "cross", "distractor"],
        help=(
            "Candidate scope: 'intra' = single target-doc (cuad, default); "
            "'cross' = global cross-document from coarse centroids (hotpot default); "
            "'distractor' = this query's own ~10 context paragraphs (HotpotQA "
            "distractor setting). hotpot auto-forces 'cross' unless 'distractor'."
        ),
    )
    parser.add_argument("--num_queries", type=int, default=50)
    parser.add_argument("--nprobe", type=int, default=32,
                        help="Coarse-search centroids probed. Raise (e.g. 128 = all) "
                        "for hotpot so gold paragraphs enter the candidate pool.")
    parser.add_argument("--no_centroid_filter", action="store_true",
                        help="Disable IVF centroid filter (scan all posting lists). "
                        "Guarantees every gold paragraph is a candidate.")
    # Optional grid overrides (comma-separated). Omit to use module defaults.
    parser.add_argument("--lambdas", type=str, default=None,
                        help="Override mmr_lambda grid, e.g. '0.5' or '0.3,0.5,0.7'")
    parser.add_argument("--tau_dups", type=str, default=None,
                        help="Override tau_dup grid, e.g. '-0.97' or '-1.0,-0.97'")
    parser.add_argument("--gammas", type=str, default=None,
                        help="Override gamma grid, e.g. '0.2'")
    parser.add_argument("--top_ks", type=str, default=None,
                        help="Override top_k grid, e.g. '10' or '5,10,20'")
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help=(
            "결과 CSV 경로 (미지정 시 output_dir/sweep_exports/<실행시각>/sweep_results_full_grid.csv)"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Base dir for offline_data_{domain} (same as online_accelerator CLI)",
    )

    # MMR (capacitor) hardware path — Phase 1 capacitor-based AIMDC
    parser.add_argument(
        "--mmr_mode",
        type=str,
        default="software",
        choices=["software", "capacitor"],
        help="MMR pair-distance backend. 'software' (default) uses numpy L2; "
        "'capacitor' uses analytical capacitor DeviceDistance.",
    )
    parser.add_argument(
        "--device_fit_path",
        type=str,
        default="h_aimdc_sim/igzo_fit.json",
        help="Path to capacitor v1 fit JSON (used when --mmr_mode capacitor).",
    )
    parser.add_argument(
        "--device_scaler_path",
        type=str,
        default=None,
        help="Optional capacitor scaler JSON (default: <fit_stem>.scaler.json).",
    )
    parser.add_argument(
        "--device_scale_k_path",
        type=str,
        default="h_aimdc_sim/igzo_fit.k.json",
        help="Path to capacitor scale_k JSON (calibrates device distance to software-L2 scale).",
    )
    parser.add_argument(
        "--device_reduce",
        type=str,
        default="sum",
        choices=["sum", "mean"],
    )
    parser.add_argument(
        "--device_use_log",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--device_v_ml",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--noise_config",
        type=str,
        default=None,
        help="Optional capacitor NoiseModel JSON.",
    )
    parser.add_argument(
        "--noise_seed",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--device_scale_k",
        type=float,
        default=None,
        help="Override capacitor scale_k (skips loading from --device_scale_k_path).",
    )
    parser.add_argument(
        "--device_v_min",
        type=float,
        default=-3.0,
        help="Capacitor MMR-path operating voltage window lower bound "
        "(default -3.0; measured range, grows analog distance spread ~2.5x vs [-1,1]).",
    )
    parser.add_argument(
        "--device_v_max",
        type=float,
        default=3.0,
        help="Capacitor MMR-path operating voltage window upper bound (default 3.0).",
    )
    parser.add_argument(
        "--anchor_protect_n",
        type=int,
        default=0,
        help="ANCHOR PROTECTION: number of candidates just below the gamma anchors, "
        "by query-relevance rank, that bypass the tau_dup dedup gate (they still "
        "compete on the full MMR score). Uses only the relevance ranking that gamma "
        "already uses -- no labels. 0 = off = previous behaviour.",
    )
    parser.add_argument(
        "--standardize_mmr_terms",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Per-query spread-match the analog MMR diversity term to the "
        "relevance term (default on; analog path only, rank/shape preserving).",
    )

    # Centroid (NVM/FeFET) hardware path — Phase 1 NVM analytical
    parser.add_argument(
        "--centroid_mode",
        type=str,
        default="software",
        choices=["software", "nvm"],
        help="Cluster-centroid L2 backend. 'software' (default) uses numpy L2; "
        "'nvm' uses analytical NvmDeviceDistance (FeFET cell).",
    )
    parser.add_argument(
        "--centroid_fit_path",
        type=str,
        default="h_aimdc_sim/igzo_fit_v3.json",
        help="Path to NVM v3 fit JSON (used when --centroid_mode nvm).",
    )
    parser.add_argument(
        "--centroid_scaler_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--centroid_reduce",
        type=str,
        default="sum",
        choices=["sum", "mean", "linear_sum"],
        help="Use 'linear_sum' for the FeFET centroid fit (hardware match-line sum).",
    )
    parser.add_argument(
        "--centroid_use_log",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--centroid_v_ml",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--centroid_noise_config",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--centroid_noise_seed",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--centroid_inverter_vtc_path",
        type=str,
        default=None,
        help="Optional inverter VTC npz (Phase 1.5). When set, NvmDeviceDistance "
        "uses LUT-based V_A->V_B; otherwise ideal -V_A.",
    )

    args = parser.parse_args()

    # Apply optional grid overrides (comma-separated CLI values).
    def _flist(s):
        return [float(x) for x in str(s).replace(" ", "").split(",") if x]
    def _ilist(s):
        return [int(float(x)) for x in str(s).replace(" ", "").split(",") if x]
    if args.lambdas is not None:
        LAMBDAS = _flist(args.lambdas)
    if args.tau_dups is not None:
        TAU_DUPS = _flist(args.tau_dups)
    if args.gammas is not None:
        GAMMAS = _flist(args.gammas)
    if args.top_ks is not None:
        TOP_K_LIST = _ilist(args.top_ks)

    # Build hardware-mode keyword arguments for run_mmr_evaluation.
    # These are fixed for the entire sweep (only RAG hyperparameters vary).
    hw_kwargs: Dict[str, Any] = {}

    # Retrieval-side ablation; not a hardware flag, but it rides the same kwargs dict.
    if int(args.anchor_protect_n) > 0:
        hw_kwargs["anchor_protect_n"] = int(args.anchor_protect_n)
        print(f"[sweep] ANCHOR PROTECTION: anchor_protect_n={args.anchor_protect_n}", flush=True)

    # Capacitor (MMR) path
    if args.mmr_mode == "capacitor":
        hw_kwargs.update(
            {
                "use_device_distance": True,
                "device_fit_path": args.device_fit_path,
                "device_scaler_path": args.device_scaler_path,
                "device_reduce": args.device_reduce,
                "device_use_log": bool(args.device_use_log),
                "device_v_ml": float(args.device_v_ml),
                "noise_config": args.noise_config,
                "noise_seed": args.noise_seed,
                "device_scale_k_path": args.device_scale_k_path,
                "device_scale_k": args.device_scale_k,
                "device_v_min": float(args.device_v_min),
                "device_v_max": float(args.device_v_max),
                "standardize_mmr_terms": bool(args.standardize_mmr_terms),
            }
        )

    # NVM (centroid) path
    if args.centroid_mode == "nvm":
        hw_kwargs.update(
            {
                "use_centroid_device_distance": True,
                "centroid_fit_path": args.centroid_fit_path,
                "centroid_scaler_path": args.centroid_scaler_path,
                "centroid_reduce": args.centroid_reduce,
                "centroid_use_log": bool(args.centroid_use_log),
                "centroid_v_ml": float(args.centroid_v_ml),
                "centroid_noise_config": args.centroid_noise_config,
                "centroid_noise_seed": args.centroid_noise_seed,
                "centroid_inverter_vtc_path": args.centroid_inverter_vtc_path,
            }
        )

    mode_tag = f"mmr-{args.mmr_mode}_centroid-{args.centroid_mode}"
    print(f"[sweep] hardware mode: {mode_tag}", flush=True)
    if hw_kwargs:
        print(f"[sweep] hardware kwargs: {sorted(hw_kwargs.keys())}", flush=True)

    offline_data_dir = os.path.join(args.output_dir, f"offline_data_{args.domain}")
    if not os.path.isdir(offline_data_dir):
        print(
            f"[오류] 오프라인 인덱스가 없습니다: {offline_data_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{mode_tag}"
    out_root = os.path.join(args.output_dir, "sweep_exports", run_id)
    combos_dir = os.path.join(out_root, "combos")
    os.makedirs(combos_dir, exist_ok=True)
    jsonl_path = os.path.join(out_root, f"retrieval_{run_id}.jsonl")

    if args.output_csv:
        out_path = os.path.abspath(args.output_csv)
    else:
        out_path = os.path.abspath(
            os.path.join(out_root, "sweep_results_full_grid.csv")
        )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    rows: List[Dict[str, Any]] = []

    total = (
        len(LAMBDAS)
        * len(TAU_DUPS)
        * len(GAMMAS)
        * len(TOP_K_LIST)
    )
    print(
        f"[sweep] 그리드: |Λ|={len(LAMBDAS)} |τ|={len(TAU_DUPS)} |γ|={len(GAMMAS)} "
        f"|top_k|={len(TOP_K_LIST)} → 총 {total}회 (window_size={FIXED_WINDOW_SIZE} 고정)",
        flush=True,
    )
    print(f"[sweep] TOP_K_LIST = {TOP_K_LIST}", flush=True)
    print(f"[sweep] GAMMAS = {GAMMAS}", flush=True)
    print(f"[sweep] 아티팩트 디렉터리: {out_root}", flush=True)
    print(f"[sweep] JSONL: {jsonl_path}", flush=True)

    idx = 0
    with open(jsonl_path, "w", encoding="utf-8") as jsonl_f:
        for lam, tau, gam, tk in itertools.product(
            LAMBDAS, TAU_DUPS, GAMMAS, TOP_K_LIST
        ):
            idx += 1
            print(
                f"[{idx}/{total}] lambda={lam} tau={tau} gamma={gam} "
                f"top_k={tk} ...",
                flush=True,
            )
            report = run_mmr_evaluation(
                args.domain,
                offline_data_dir=offline_data_dir,
                num_queries=args.num_queries,
                max_contexts=None,
                top_k=int(tk),
                nprobe=int(args.nprobe),
                use_centroid_filter=not args.no_centroid_filter,
                top_r=128,
                tau_dup=tau,
                mmr_lambda=lam,
                gamma=gam,
                window_size=FIXED_WINDOW_SIZE,
                verbose=False,
                show_docs=False,
                export_normalized_json=None,
                target_doc_id=None,
                retrieval_mode="COMPARE",
                retrieval_scope=args.retrieval_scope,
                **hw_kwargs,
            )
            l2_metrics = report.get("avg_metrics_l2", {}) or {}
            mmr_metrics = report.get("avg_metrics_mmr", {}) or {}
            retrieved = report.get("retrieved_context") or []

            l2_recall = _metric_float(l2_metrics, "recall_at_k_50")
            mmr_recall = _metric_float(mmr_metrics, "recall_at_k_50")
            l2_rouge_l = _metric_float(l2_metrics, "rouge_l_max_recall")
            mmr_rouge_l = _metric_float(mmr_metrics, "rouge_l_max_recall")
            l2_ild = _metric_float(l2_metrics, "intra_list_distance")
            mmr_ild = _metric_float(mmr_metrics, "intra_list_distance")
            l2_rr = _metric_float(l2_metrics, "redundancy_rate")
            mmr_rr = _metric_float(mmr_metrics, "redundancy_rate")
            l2_breadth = _metric_float(l2_metrics, "unique_centroids_selected")
            mmr_breadth = _metric_float(mmr_metrics, "unique_centroids_selected")

            combo_relpath = f"combos/combo_{idx:05d}.json"
            combo_path = os.path.join(out_root, combo_relpath)
            combo_payload = {
                "run_id": run_id,
                "sweep_idx": idx,
                "domain": args.domain,
                "num_queries": args.num_queries,
                "parameters": {
                    "mmr_mode": args.mmr_mode,
                    "centroid_mode": args.centroid_mode,
                    "mmr_lambda": lam,
                    "tau_dup": tau,
                    "gamma": gam,
                    "top_k": tk,
                    "window_size": FIXED_WINDOW_SIZE,
                },
                "metrics": {
                    "l2_recall": l2_recall,
                    "mmr_recall": mmr_recall,
                    "l2_rouge_l": l2_rouge_l,
                    "mmr_rouge_l": mmr_rouge_l,
                    "l2_ild": l2_ild,
                    "mmr_ild": mmr_ild,
                    "l2_rr": l2_rr,
                    "mmr_rr": mmr_rr,
                    "l2_breadth": l2_breadth,
                    "mmr_breadth": mmr_breadth,
                },
                "retrieved_context": retrieved,
            }
            with open(combo_path, "w", encoding="utf-8") as jf:
                json.dump(combo_payload, jf, ensure_ascii=False, indent=2)

            # Multi-hop supporting-passage recall, accumulated over this combo's
            # queries. Only queries with a non-empty gold_support contribute
            # (empty for CUAD → aggregate() returns {} and columns stay blank).
            l2_rows: List[Dict[str, float]] = []
            mmr_rows: List[Dict[str, float]] = []
            for qi, sample in enumerate(retrieved):
                gold = sample.get("gold_support") or []
                if gold:
                    rdi = sample.get("retrieved_doc_ids") or {}
                    l2_rows.append(
                        evaluate_supporting_recall(rdi.get("l2") or [], gold, ks=SUPPORT_KS)
                    )
                    mmr_rows.append(
                        evaluate_supporting_recall(rdi.get("mmr") or [], gold, ks=SUPPORT_KS)
                    )
                line = {
                    "run_id": run_id,
                    "sweep_idx": idx,
                    "query_idx": qi,
                    "domain": args.domain,
                    "mmr_mode": args.mmr_mode,
                    "centroid_mode": args.centroid_mode,
                    "mmr_lambda": lam,
                    "tau_dup": tau,
                    "gamma": gam,
                    "top_k": tk,
                    "window_size": FIXED_WINDOW_SIZE,
                    "query": sample.get("query"),
                    "ground_truth": sample.get("ground_truth"),
                    # aspect_queries drives the coverage metric downstream; without
                    # it llm_generator emits empty aspects and eval coverage -> 0.
                    "aspect_queries": sample.get("aspect_queries", []),
                    "chunk_lut": sample.get("chunk_lut"),
                    "l2_contexts": sample.get("l2_contexts", sample.get("b1_contexts")),
                    "mmr_contexts": sample.get("mmr_contexts"),
                    # Multi-hop supporting-passage recall (hotpot). gold_support =
                    # gold corpus-paragraph indices; retrieved_doc_ids = ranked
                    # parent doc_id per arm. None for cuad → eval skips recall.
                    "gold_support": sample.get("gold_support"),
                    "retrieved_doc_ids": sample.get("retrieved_doc_ids"),
                    # IGZO-tier write/readout counters for the MMR arm (see
                    # scripts/analyze_write_stats.py). Additive; None for old runs.
                    "write_stats": sample.get("write_stats"),
                    "retrieved_context": {
                        "l2_contexts": sample.get("l2_contexts", sample.get("b1_contexts")),
                        "mmr_contexts": sample.get("mmr_contexts"),
                    },
                }
                jsonl_f.write(json.dumps(line, ensure_ascii=False) + "\n")

            l2_agg = aggregate(l2_rows)
            mmr_agg = aggregate(mmr_rows)

            row = {
                "mmr_mode": args.mmr_mode,
                "centroid_mode": args.centroid_mode,
                "mmr_lambda": lam,
                "tau_dup": tau,
                "gamma": gam,
                "top_k": tk,
                "window_size": FIXED_WINDOW_SIZE,
                "l2_recall": l2_recall,
                "mmr_recall": mmr_recall,
                "l2_rouge_l": l2_rouge_l,
                "mmr_rouge_l": mmr_rouge_l,
                "l2_ild": l2_ild,
                "mmr_ild": mmr_ild,
                "l2_rr": l2_rr,
                "mmr_rr": mmr_rr,
                "l2_breadth": l2_breadth,
                "mmr_breadth": mmr_breadth,
                "combo_artifact": combo_relpath,
            }
            # Supporting-passage recall columns (blank for CUAD: empty aggregates).
            row["support_n"] = len(l2_rows)
            for k in SUPPORT_KS:
                l2_recall_k = l2_agg.get(f"recall@{k}")
                mmr_recall_k = mmr_agg.get(f"recall@{k}")
                l2_full_k = l2_agg.get(f"full_support@{k}")
                mmr_full_k = mmr_agg.get(f"full_support@{k}")
                row[f"l2_recall@{k}"] = "" if l2_recall_k is None else l2_recall_k
                row[f"mmr_recall@{k}"] = "" if mmr_recall_k is None else mmr_recall_k
                row[f"l2_full@{k}"] = "" if l2_full_k is None else l2_full_k
                row[f"mmr_full@{k}"] = "" if mmr_full_k is None else mmr_full_k
                row[f"full_delta@{k}"] = (
                    (mmr_full_k - l2_full_k)
                    if (l2_full_k is not None and mmr_full_k is not None)
                    else ""
                )
            rows.append(row)

            print(
                f"Lambda={lam} | Tau={tau} | Gamma={gam} | top_k={tk} || "
                f"Recall(L2:{l2_recall:.2f}, MMR:{mmr_recall:.2f}) | "
                f"ROUGE-L(L2:{l2_rouge_l:.2f}, MMR:{mmr_rouge_l:.2f}) | "
                f"ILD(L2:{l2_ild:.2f}, MMR:{mmr_ild:.2f}) | "
                f"RR(L2:{l2_rr:.2f}, MMR:{mmr_rr:.2f})",
                flush=True,
            )

    manifest = {
        "run_id": run_id,
        "mode_tag": mode_tag,
        "mmr_mode": args.mmr_mode,
        "centroid_mode": args.centroid_mode,
        "artifact_root": out_root,
        "domain": args.domain,
        "num_queries": args.num_queries,
        "total_combinations": total,
        "csv_path": out_path,
        "jsonl_path": jsonl_path,
        "combos_subdir": "combos",
    }
    with open(
        os.path.join(out_root, "run_manifest.json"), "w", encoding="utf-8"
    ) as mf:
        json.dump(manifest, mf, ensure_ascii=False, indent=2)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in CSV_HEADER})

    print(
        f"\n[완료] {len(rows)} rows (lambda×tau×gamma×top_k; window_size={FIXED_WINDOW_SIZE}) → {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
