#!/usr/bin/env python3
"""
Collect every fact the supplementary note needs into one JSON bundle.

Reads the primary artifacts only — nothing is hand-transcribed:
  sweep_exports/20260811_181301_*/retrieval_*.jsonl   (query, GT spans, top-10 x 2 arms)
  capOnLiability2_{L2,MMR}_generation_{fp16,int8,int4}.txt
  sweep_exports/20260811_181301_*/sweep_results_full_grid.csv  (n=50 aggregate)
  eval_{fp16,int8,int4}/summary.json                            (per-precision generation metrics)

Out: note_bundle.json  (consumed by build_note_docx.js)
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DOC_ID = "337"
TAG_RE = re.compile(r"\[Source:\s*([^|\]]+)")

CLAUSE_KR = {
    "CE45B7": "§13.1 연속 — 결과적손해 면책",
    "C42340": "§13.1 LIMIT OF LIABILITY (조항 서두)",
    "C8C2D2": "§9.1 INDEMNIFICATION OF LICENSOR",
    "CA3EA4": "§9.2 INDEMNIFICATION OF LICENSEE",
    "C21F8E": "§13.1 LIMIT OF LIABILITY",
    "C7A673": "§9.1 INDEMNIFICATION (Art.9 경계)",
    "CFBACB": "§9.2 INDEMNIFICATION OF LICENSEE (겹침 윈도)",
    "C7BBE4": "§12.4 EXCLUSIVE REMEDIES / 보증 부인",
    "C56C5E": "§9.3 INDEMNIFICATION PROCEDURE",
    "C8492E": "§13.1 연속 → Art.14 (Term) 경계",
    "C5CE22": "도산·파산 해지 트리거",
    "C1E48B": "반환·파기·환불 구제",
    "C6B071": "§14.3 DUTIES UPON TERMINATION",
    "C1BE9E": "형평법상 구제 / 특정이행",
}
SPAN_KR = [
    "A — 30일 치유기간",
    "B — §13.1 총액 상한",
    "C — 반환·파기·환불 단독 구제",
    "D — 결과적손해 면책",
    "E — 배타적 구제",
]


def cid(t: str) -> str:
    return "C" + hashlib.md5(t[:120].encode()).hexdigest()[:5].upper()


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def load_record():
    path = sorted(glob.glob(os.path.join(HERE, "sweep_exports/20260811_181301*/retrieval_*.jsonl")))[0]
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if "Cap On Liability" not in d.get("query", ""):
                continue
            lut = (d.get("chunk_lut") or {}).get("l2") or []
            if DOC_ID in {str(x.get("doc_id")) for x in lut if isinstance(x, dict)}:
                d["_src"] = os.path.relpath(path, HERE)
                return d
    raise SystemExit("record not found")


def arm_rows(rec, arm):
    out = []
    for ctx, x in zip(rec[f"{arm}_contexts"], (rec.get("chunk_lut") or {}).get(arm) or []):
        m = TAG_RE.findall(ctx)
        body = (x.get("chunk_text") or "").strip()
        out.append({"tag": m[0].strip() if m else "?", "cid": cid(body),
                    "clause": CLAUSE_KR.get(cid(body), "(미분류)"),
                    "text": re.sub(r"\s+", " ", body).strip()})
    return out


def span_hits(rows, span):
    w = span.split()
    frags = [" ".join(w[i:i + 8]) for i in range(0, max(1, len(w) - 8), 4)]
    hit = set()
    for fr in frags:
        for i, r in enumerate(rows, 1):
            if norm(fr) in norm(r["text"]):
                hit.add(i)
    return sorted(hit)


def gen_text(arm, q):
    p = os.path.join(HERE, f"capOnLiability2_{arm}_generation_{q}.txt")
    return open(p, encoding="utf-8", errors="replace").read().split("=" * 40)[-1].strip()


def main():
    rec = load_record()
    l2, mmr = arm_rows(rec, "l2"), arm_rows(rec, "mmr")
    spans = [re.sub(r"\s+", " ", s).strip()
             for s in (rec.get("ground_truth") or "").split("[SEP]") if s.strip()]
    l2s, ms = {r["cid"] for r in l2}, {r["cid"] for r in mmr}
    shared = l2s & ms
    pl = {r["cid"]: i for i, r in enumerate(l2, 1)}
    pm = {r["cid"]: i for i, r in enumerate(mmr, 1)}

    for rows, other in ((l2, ms), (mmr, l2s)):
        for r in rows:
            r["shared"] = r["cid"] in other

    csvp = sorted(glob.glob(os.path.join(HERE, "sweep_exports/20260811_181301*/sweep_results_full_grid.csv")))[0]
    agg = list(csv.DictReader(open(csvp, encoding="utf-8")))[0]

    evals = {}
    for q in ("fp16", "int8", "int4"):
        p = os.path.join(HERE, f"eval_{q}", "summary.json")
        if os.path.exists(p):
            s = json.load(open(p, encoding="utf-8"))["pairwise_answer_averages"]
            evals[q] = {"l2_cov": s["answer_a"]["coverage"], "mmr_cov": s["answer_b"]["coverage"],
                        "l2_bs": s["answer_a"]["bertscore_f1"], "mmr_bs": s["answer_b"]["bertscore_f1"],
                        "l2_rc3g": s["answer_a"]["retrieval_coverage_top3_graded"],
                        "mmr_rc3g": s["answer_b"]["retrieval_coverage_top3_graded"]}

    bundle = {
        "src_jsonl": rec["_src"], "query_idx": rec["query_idx"], "doc_id": DOC_ID,
        "params": {k: rec[k] for k in ("top_k", "gamma", "mmr_lambda", "tau_dup", "window_size")},
        "query": rec["query"].strip(),
        "spans": [{"name": n, "text": t} for n, t in zip(SPAN_KR, spans)],
        "l2": l2, "mmr": mmr,
        "n_anchor": sum(1 for r in mmr if r["tag"] == "L2_Anchor"),
        "n_shared": len(shared),
        "n_l2_excl": len(l2s - shared), "n_mmr_excl": len(ms - shared),
        "n9_l2": sum(1 for r in l2 if "§9" in r["clause"]),
        "n9_mmr": sum(1 for r in mmr if "§9" in r["clause"]),
        "l2_excl": [{"cid": c, "clause": CLAUSE_KR[c]} for c in sorted(l2s - shared, key=lambda c: pl[c])],
        "mmr_excl": [{"cid": c, "clause": CLAUSE_KR[c]} for c in sorted(ms - shared, key=lambda c: pm[c])],
        "span_cov": [{"name": n, "l2": span_hits(l2, t), "mmr": span_hits(mmr, t)}
                     for n, t in zip(SPAN_KR, spans)],
        "rank": [{"cid": c, "clause": CLAUSE_KR[c], "l2": pl[c], "mmr": pm[c]}
                 for c in sorted(shared, key=lambda c: pl[c])],
        "gen": {q: {a: gen_text(a, q) for a in ("L2", "MMR")} for q in ("fp16", "int8", "int4")},
        "agg": {k: agg[k] for k in ("l2_ild", "mmr_ild", "l2_rr", "mmr_rr", "l2_breadth",
                                    "mmr_breadth", "l2_recall", "mmr_recall",
                                    "l2_rouge_l", "mmr_rouge_l")},
        "agg_n": 50,
        "evals": evals,
    }
    out = os.path.join(HERE, "note_bundle.json")
    json.dump(bundle, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote", out)
    print(f"  anchors={bundle['n_anchor']} shared={bundle['n_shared']} "
          f"L2excl={bundle['n_l2_excl']} MMRexcl={bundle['n_mmr_excl']} "
          f"§9: L2={bundle['n9_l2']}/10 MMR={bundle['n9_mmr']}/10")


if __name__ == "__main__":
    main()
