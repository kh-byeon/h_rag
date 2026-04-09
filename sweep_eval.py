#!/usr/bin/env python3
"""
Grid search over mmr_lambda × tau_dup × gamma × top_k
for run_mmr_evaluation with retrieval_mode=COMPARE (B1 vs B2 vs MMR metrics).

조항 단위 오프라인 청킹 이후에는 인접 청크 스티칭이 불필요하므로 window_size는 항상 1(스티칭 없음)로 고정.

online_accelerator.run_mmr_evaluation에 직접 인자를 넘깁니다
(equivalent to CLI: --top-k, --window-size 1, --gamma, --retrieval-mode COMPARE, ...).

각 조합의 검색 컨텍스트(retrieved_context)는 `output_dir/sweep_exports/<타임스탬프>/`
아래에 JSON·JSONL로 저장되며, CSV에는 해당 콤보 JSON 상대 경로(combo_artifact)만 기록합니다.
"""
import argparse
import csv
import itertools
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

from online_accelerator import run_mmr_evaluation

LAMBDAS = [0.3, 0.5, 0.7]
TAU_DUPS = [0.3, 0.5, 0.7 ]
# inter/intra-doc 비중 탐색 (파이프라인에서 gamma는 [0, 1]로 클램프됨)
GAMMAS = [0.3, 0.5, 0.7]
# 검색·재순위화 후 최종 Top-K (online --top-k 와 동일 의미)
TOP_K_LIST = [5, 10, 20]
# 조항 단위 청킹 후 스티칭 비사용: run_mmr_evaluation 및 산출물에 항상 1로 기록 (하위 호환용 컬럼)
FIXED_WINDOW_SIZE = 1

CSV_HEADER = [
    "mmr_lambda",
    "tau_dup",
    "gamma",
    "top_k",
    "window_size",
    "b1_recall",
    "b2_recall",
    "mmr_recall",
    "b1_ild",
    "b2_ild",
    "mmr_ild",
    "b1_rr",
    "b2_rr",
    "mmr_rr",
    "b1_breadth",
    "b2_breadth",
    "mmr_breadth",
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
    parser = argparse.ArgumentParser(
        description=(
            "Grid search: mmr_lambda × tau_dup × gamma × top_k "
            "→ run_mmr_evaluation (COMPARE: B1, B2, MMR); window_size=1 고정(스티칭 없음)"
        )
    )
    parser.add_argument("--domain", type=str, default="cuad")
    parser.add_argument("--num_queries", type=int, default=50)
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
    args = parser.parse_args()

    offline_data_dir = os.path.join(args.output_dir, f"offline_data_{args.domain}")
    if not os.path.isdir(offline_data_dir):
        print(
            f"[오류] 오프라인 인덱스가 없습니다: {offline_data_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
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
                nprobe=32,
                top_r=128,
                tau_dup=tau,
                mmr_lambda=lam,
                gamma=gam,
                window_size=FIXED_WINDOW_SIZE,
                verbose=False,
                show_docs=False,
                export_json=None,
                target_doc_id=None,
                retrieval_mode="COMPARE",
                anchor_lambda=1.0,
                anchor_tau_dup=0.0,
            )
            b1_metrics = report.get("avg_metrics_l2", {}) or {}
            b2_metrics = report.get("avg_metrics_b2", {}) or {}
            mmr_metrics = report.get("avg_metrics_mmr", {}) or {}
            retrieved = report.get("retrieved_context") or []

            b1_recall = _metric_float(b1_metrics, "recall_at_k_50")
            b2_recall = _metric_float(b2_metrics, "recall_at_k_50")
            mmr_recall = _metric_float(mmr_metrics, "recall_at_k_50")
            b1_ild = _metric_float(b1_metrics, "intra_list_distance")
            b2_ild = _metric_float(b2_metrics, "intra_list_distance")
            mmr_ild = _metric_float(mmr_metrics, "intra_list_distance")
            b1_rr = _metric_float(b1_metrics, "redundancy_rate")
            b2_rr = _metric_float(b2_metrics, "redundancy_rate")
            mmr_rr = _metric_float(mmr_metrics, "redundancy_rate")
            b1_breadth = _metric_float(b1_metrics, "unique_centroids_selected")
            b2_breadth = _metric_float(b2_metrics, "unique_centroids_selected")
            mmr_breadth = _metric_float(mmr_metrics, "unique_centroids_selected")

            combo_relpath = f"combos/combo_{idx:05d}.json"
            combo_path = os.path.join(out_root, combo_relpath)
            combo_payload = {
                "run_id": run_id,
                "sweep_idx": idx,
                "domain": args.domain,
                "num_queries": args.num_queries,
                "parameters": {
                    "mmr_lambda": lam,
                    "tau_dup": tau,
                    "gamma": gam,
                    "top_k": tk,
                    "window_size": FIXED_WINDOW_SIZE,
                },
                "metrics": {
                    "b1_recall": b1_recall,
                    "b2_recall": b2_recall,
                    "mmr_recall": mmr_recall,
                    "b1_ild": b1_ild,
                    "b2_ild": b2_ild,
                    "mmr_ild": mmr_ild,
                    "b1_rr": b1_rr,
                    "b2_rr": b2_rr,
                    "mmr_rr": mmr_rr,
                    "b1_breadth": b1_breadth,
                    "b2_breadth": b2_breadth,
                    "mmr_breadth": mmr_breadth,
                },
                "retrieved_context": retrieved,
            }
            with open(combo_path, "w", encoding="utf-8") as jf:
                json.dump(combo_payload, jf, ensure_ascii=False, indent=2)

            for qi, sample in enumerate(retrieved):
                line = {
                    "run_id": run_id,
                    "sweep_idx": idx,
                    "query_idx": qi,
                    "domain": args.domain,
                    "mmr_lambda": lam,
                    "tau_dup": tau,
                    "gamma": gam,
                    "top_k": tk,
                    "window_size": FIXED_WINDOW_SIZE,
                    "query": sample.get("query"),
                    "ground_truth": sample.get("ground_truth"),
                    "b1_contexts": sample.get("b1_contexts"),
                    "b2_contexts": sample.get("b2_contexts"),
                    "mmr_contexts": sample.get("mmr_contexts"),
                    "retrieved_context": {
                        "b1_contexts": sample.get("b1_contexts"),
                        "b2_contexts": sample.get("b2_contexts"),
                        "mmr_contexts": sample.get("mmr_contexts"),
                    },
                }
                jsonl_f.write(json.dumps(line, ensure_ascii=False) + "\n")

            row = {
                "mmr_lambda": lam,
                "tau_dup": tau,
                "gamma": gam,
                "top_k": tk,
                "window_size": FIXED_WINDOW_SIZE,
                "b1_recall": b1_recall,
                "b2_recall": b2_recall,
                "mmr_recall": mmr_recall,
                "b1_ild": b1_ild,
                "b2_ild": b2_ild,
                "mmr_ild": mmr_ild,
                "b1_rr": b1_rr,
                "b2_rr": b2_rr,
                "mmr_rr": mmr_rr,
                "b1_breadth": b1_breadth,
                "b2_breadth": b2_breadth,
                "mmr_breadth": mmr_breadth,
                "combo_artifact": combo_relpath,
            }
            rows.append(row)

            print(
                f"Lambda={lam} | Tau={tau} | Gamma={gam} | top_k={tk} || "
                f"Recall(B1:{b1_recall:.2f}, B2:{b2_recall:.2f}, MMR:{mmr_recall:.2f}) | "
                f"ILD(B1:{b1_ild:.2f}, B2:{b2_ild:.2f}, MMR:{mmr_ild:.2f}) | "
                f"RR(B1:{b1_rr:.2f}, B2:{b2_rr:.2f}, MMR:{mmr_rr:.2f})",
                flush=True,
            )

    manifest = {
        "run_id": run_id,
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
