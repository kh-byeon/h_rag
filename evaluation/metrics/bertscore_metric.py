from __future__ import annotations

from typing import Dict


def compute_bertscore(
    reference_text: str,
    candidate_text: str,
    model_type: str,
    lang: str,
) -> Dict[str, float]:
    """Compute BERTScore for one reference-candidate pair."""
    from bert_score import score as bertscore_score

    p, r, f1 = bertscore_score(
        [candidate_text or ""],
        [reference_text or ""],
        model_type=model_type,
        lang=lang,
        verbose=False,
    )
    return {
        "precision": float(p[0].item()),
        "recall": float(r[0].item()),
        "f1": float(f1[0].item()),
    }
