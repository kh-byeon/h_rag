"""
검색 결과 JSON(예: cuad_eval_data.json)을 읽어 LLM 생성 없이 순수 검색 성능(Recall@K, MRR@K)만 평가.
"""
from __future__ import annotations

import argparse
import json
import re
from typing import List, Dict, Any


def _normalize(s: str) -> str:
    """대소문자 통일 및 공백 정규화."""
    if not s or not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def is_chunk_in_ground_truth(chunk_text: str, ground_truth: str) -> bool:
    """
    검색된 chunk 텍스트가 ground_truth(계약서 원본) 안에 부분 문자열로 포함되는지 확인.
    대소문자 무시, 공백 정규화 적용.
    """
    chunk = _normalize(chunk_text)
    gt = _normalize(ground_truth)
    if not chunk:
        return False
    return chunk in gt


def recall_at_k(contexts: List[str], ground_truth: str, k: int) -> float:
    """Top-K 중 정답(ground_truth에 포함된 텍스트)이 1개라도 있으면 1, 없으면 0."""
    for i in range(min(k, len(contexts))):
        if is_chunk_in_ground_truth(contexts[i], ground_truth):
            return 1.0
    return 0.0


def mrr_at_k(contexts: List[str], ground_truth: str, k: int) -> float:
    """정답이 포함된 문서가 발견된 최초 순위(1-based)의 역수. 없으면 0."""
    for rank in range(1, min(k, len(contexts)) + 1):
        if is_chunk_in_ground_truth(contexts[rank - 1], ground_truth):
            return 1.0 / rank
    return 0.0


def run_eval(
    data: List[Dict[str, Any]],
    k: int,
) -> Dict[str, Any]:
    """전체 쿼리에 대해 L2/MMR별 Recall@K, MRR@K 평균 계산."""
    if not data:
        return {
            "k": k,
            "num_queries": 0,
            "l2_recall_avg": 0.0,
            "l2_mrr_avg": 0.0,
            "mmr_recall_avg": 0.0,
            "mmr_mrr_avg": 0.0,
        }

    l2_recall: List[float] = []
    l2_mrr: List[float] = []
    mmr_recall: List[float] = []
    mmr_mrr: List[float] = []

    for item in data:
        gt = item.get("ground_truth") or ""
        l2_ctx = item.get("l2_contexts") or []
        mmr_ctx = item.get("mmr_contexts") or []

        l2_recall.append(recall_at_k(l2_ctx, gt, k))
        l2_mrr.append(mrr_at_k(l2_ctx, gt, k))
        mmr_recall.append(recall_at_k(mmr_ctx, gt, k))
        mmr_mrr.append(mrr_at_k(mmr_ctx, gt, k))

    n = len(data)
    return {
        "k": k,
        "num_queries": n,
        "l2_recall_avg": sum(l2_recall) / n,
        "l2_mrr_avg": sum(l2_mrr) / n,
        "mmr_recall_avg": sum(mmr_recall) / n,
        "mmr_mrr_avg": sum(mmr_mrr) / n,
    }


def print_report(report: Dict[str, Any]) -> None:
    """Recall@K, MRR@K 리포트 출력 (소수점 4자리)."""
    k = report.get("k", 0)
    n = report.get("num_queries", 0)
    print()
    print("=" * 56)
    print(f"  Retriever 평가 리포트  (Top-K = {k}, 쿼리 수 = {n})")
    print("=" * 56)
    print("  Recall@{} (평균):".format(k))
    print("    L2  Top-K:  {:.4f}".format(report.get("l2_recall_avg", 0)))
    print("    MMR Top-K: {:.4f}".format(report.get("mmr_recall_avg", 0)))
    print("  MRR@{} (평균):".format(k))
    print("    L2  Top-K:  {:.4f}".format(report.get("l2_mrr_avg", 0)))
    print("    MMR Top-K: {:.4f}".format(report.get("mmr_mrr_avg", 0)))
    print("=" * 56)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="검색 결과 JSON으로 순수 검색 성능(Recall@K, MRR@K) 평가"
    )
    parser.add_argument(
        "json_path",
        type=str,
        nargs="?",
        default=None,
        help="검색 결과 JSON 파일 경로 (예: cuad_eval_data.json)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="평가할 Top-K 개수 (기본값: 5)",
    )
    args = parser.parse_args()

    if not args.json_path:
        parser.error("검색 결과 JSON 파일 경로를 지정하세요. 예: python retriever_eval.py cuad_eval_data.json --k 5")
        return

    with open(args.json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        data = [data]

    report = run_eval(data, k=args.k)
    print_report(report)


if __name__ == "__main__":
    main()
