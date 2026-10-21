"""
answer_em_f1.py — HotpotQA / SQuAD-style short-answer EM & token-F1.

The standard generative-QA answer metrics: after normalization (lowercase, strip
articles, drop punctuation, collapse whitespace), EM is exact string match and F1
is the token-overlap F1 between prediction and gold. No model, no I/O — pure funcs.

Also provides ``extract_final_answer`` to pull the short answer out of a generated
reasoning string. The hotpot generation prompt asks the model to end with a line
``Answer: <short answer>``; we parse that, with sensible fallbacks.
"""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Dict, List, Optional, Sequence


# ---------------------------------------------------------------- normalization
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(s: str) -> str:
    """SQuAD/HotpotQA normalization: lowercase, remove punctuation, articles, extra ws."""
    s = str(s or "").lower()
    s = s.translate(_PUNCT_TABLE)
    s = _ARTICLES_RE.sub(" ", s)
    s = " ".join(s.split())
    return s


# ------------------------------------------------------------------ answer pull
_ANSWER_LINE_RE = re.compile(r"(?is)answer\s*[:\-]\s*(.+?)\s*$")


def extract_final_answer(generated: str) -> str:
    """Best-effort short answer from a generated reasoning string.

    Priority:
      1. the text after the LAST ``Answer:`` marker (the hotpot prompt format),
      2. else the last non-empty line,
      3. else the whole string, trimmed.
    Strips surrounding markdown/quotes and trailing punctuation.
    """
    text = str(generated or "").strip()
    if not text:
        return ""
    # 1) last "Answer:" occurrence (there may be one; take the last to be safe)
    matches = list(re.finditer(r"(?is)answer\s*[:\-]\s*", text))
    if matches:
        tail = text[matches[-1].end():].strip()
        # keep only the first line of that tail
        tail = tail.splitlines()[0] if tail else tail
        cand = tail
    else:
        # 2) last non-empty line
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        cand = lines[-1] if lines else text
    # cleanup: strip markdown bold/italics, surrounding quotes, trailing period
    cand = cand.strip().strip("*_`").strip()
    cand = cand.strip('"\'' + "“”‘’").strip()
    cand = re.sub(r"\s+", " ", cand).strip()
    return cand


# ----------------------------------------------------------------------- scores
def _f1(pred: str, gold: str) -> float:
    p_toks = normalize_answer(pred).split()
    g_toks = normalize_answer(gold).split()
    if not p_toks and not g_toks:
        return 1.0  # both empty → trivially equal
    if not p_toks or not g_toks:
        return 0.0
    common = Counter(p_toks) & Counter(g_toks)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(p_toks)
    recall = n_same / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def _em(pred: str, gold: str) -> float:
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0


def score_answer(prediction: str, golds) -> Dict[str, float]:
    """EM & F1 of one prediction against one or more gold answers (max over golds).

    ``golds`` may be a string or a list of acceptable answers (HotpotQA is single
    gold, but we support a list for robustness).
    """
    if isinstance(golds, str):
        gold_list: List[str] = [golds]
    else:
        gold_list = [g for g in (golds or []) if str(g).strip()]
    if not gold_list:
        return {"em": float("nan"), "f1": float("nan")}
    em = max(_em(prediction, g) for g in gold_list)
    f1 = max(_f1(prediction, g) for g in gold_list)
    return {"em": em, "f1": f1}


def aggregate(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Mean EM/F1 over rows, skipping NaN (queries without a gold answer)."""
    out: Dict[str, float] = {}
    for key in ("em", "f1"):
        vals = [r[key] for r in rows if key in r and r[key] == r[key]]  # drop NaN
        out[key] = sum(vals) / len(vals) if vals else float("nan")
    return out
