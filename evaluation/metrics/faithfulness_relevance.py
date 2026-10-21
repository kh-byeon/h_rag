"""
faithfulness_relevance.py — reference-free generation metrics (RAGAS-style).

Two scores that never look at the gold answer, so they carry none of the
"ground-truth overlap penalises diversity" bias that EM/F1 do:

**Faithfulness** — is every claim in the answer backed by the retrieved context?
    Split the generation into claims (sentences), then for each claim ask an NLI
    model whether ANY retrieved passage entails it (premise = passage,
    hypothesis = claim, score = max entailment over passages). This is the
    standard "a backing chunk exists" formulation. Reported as both the mean
    entailment probability and the fraction of claims above a threshold.

**Answer relevance** — does the answer actually address the question?
    Cosine similarity between the SBERT embedding of the question and of the
    answer. (RAGAS generates synthetic questions from the answer and compares
    them to the original; that needs a generator LLM. The direct embedding
    cosine is the standard model-free proxy.)

Both are pure functions over (question, answer, contexts). No file I/O.

Cost note: a naive sweep does |claims| x |passages| NLI forward passes per
(answer, arm, combo, query). Callers should deduplicate on
``work_key(answer, contexts)`` first — in this repo's sweeps the same
(answer, context-set) pair recurs across many hyperparameter combos.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
DEFAULT_SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Reuse the coverage metric's model cache/entailment-index logic so both metrics
# agree on which logit column means "entailment".
from evaluation.metrics.coverage_metric import (  # noqa: E402
    _entailment_index,
    _get_nli_model,
    _softmax,
)

_sbert_cache: Dict[str, Any] = {}

# Sentence split that keeps abbreviations and decimals intact well enough for
# claim extraction. Deliberately simple — no extra dependency.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\(])")
# Bullet/enumeration prefixes the hotpot prompt tends to produce.
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
# The final "Answer: ..." line is a restatement, not an independent claim.
_ANSWER_LINE = re.compile(r"(?is)^\s*answer\s*[:\-]\s*")
# "From Passage 2: <fact>" / "Passage 2 mentions ..." — strip the citation frame
# so the hypothesis handed to the NLI model is the bare factual assertion.
# Without this the hypothesis is a statement *about the prompt*, which the
# passages cannot entail, depressing faithfulness for both arms.
_CITE_PREFIX = re.compile(
    r"(?i)^\s*(?:from\s+|in\s+|according to\s+)?passages?\s*\d+\s*"
    r"(?:[:,\-]\s*|(?:mentions?|states?|says?|notes?|indicates?|shows?|"
    r"describes?|reports?|confirms?)\s+(?:that\s+)?)"
)
# Meta/preamble lines that assert nothing ("Here are the relevant facts:").
_PREAMBLE = re.compile(
    r"(?i)^\s*(?:here (?:are|is)\b|the relevant facts\b|relevant facts\b|"
    r"to answer\b|let'?s\b|based on the passages\b|step \d+\b)"
)
# Explicit "this passage says nothing" statements. They are *about* the
# retrieval, not about the world, and no passage can entail them. Counting them
# as unfaithful would penalise whichever arm reports gaps more honestly, so they
# are excluded from the denominator and reported separately.
_NO_INFO = re.compile(
    r"(?i)\b(?:no relevant information|does not mention|doesn'?t mention|"
    r"not mentioned|no mention of|does not (?:provide|contain)|no information)\b"
)


def _get_sbert(model_name: str) -> Any:
    if model_name not in _sbert_cache:
        from sentence_transformers import SentenceTransformer

        _sbert_cache[model_name] = SentenceTransformer(model_name)
    return _sbert_cache[model_name]


def work_key(answer: str, contexts: Sequence[str]) -> str:
    """Stable hash of the (answer, context-set) pair, for caller-side dedup."""
    h = hashlib.sha256()
    h.update(str(answer or "").encode("utf-8", "replace"))
    h.update(b"\x00")
    for c in contexts or []:
        h.update(str(c or "").encode("utf-8", "replace"))
        h.update(b"\x01")
    return h.hexdigest()


def extract_claims(generated: str, *, min_words: int = 3,
                   include_answer_line: bool = False,
                   return_no_info: bool = False):
    """Split a generation into atomic, entailment-checkable claims.

    Removes, in order: bullet markers, the trailing ``Answer: X`` restatement,
    ``From Passage N:`` citation frames, meta/preamble lines, and fragments
    shorter than ``min_words``. "No relevant information" statements are pulled
    out separately — they assert something about the retrieval rather than about
    the world, so no passage can entail them.

    Returns ``claims`` (or ``(claims, no_info)`` when ``return_no_info``).
    """
    text = str(generated or "").strip()
    if not text:
        return ([], []) if return_no_info else []
    claims: List[str] = []
    no_info: List[str] = []
    for raw_line in text.splitlines():
        line = _BULLET.sub("", raw_line).strip()
        if not line:
            continue
        if _ANSWER_LINE.match(line):
            if not include_answer_line:
                continue
            line = _ANSWER_LINE.sub("", line).strip()
        for sent in _SENT_SPLIT.split(line):
            s = _CITE_PREFIX.sub("", sent.strip()).strip()
            if len(s.split()) < min_words:
                continue
            if _PREAMBLE.match(s):
                continue
            if s.endswith(":"):          # header like "Relevant facts:"
                continue
            if _NO_INFO.search(s):
                no_info.append(s)
                continue
            claims.append(s)
    return (claims, no_info) if return_no_info else claims


def compute_faithfulness(
    answer: str,
    contexts: Sequence[str],
    *,
    threshold: float = 0.5,
    nli_model_name: str = DEFAULT_NLI_MODEL,
    max_claims: Optional[int] = 12,
) -> Dict[str, Any]:
    """Fraction / mean strength of answer claims entailed by some passage.

    Returns ``n_claims = 0`` with NaN scores when the answer has no scorable
    claim; callers should skip those rows rather than treat them as zeros.
    """
    claims, no_info = extract_claims(answer, return_no_info=True)
    if max_claims is not None:
        claims = claims[:max_claims]
    ctx = [str(c or "").strip() for c in (contexts or []) if str(c or "").strip()]
    if not claims or not ctx:
        return {
            "faithfulness": float("nan"),
            "faithfulness_mean_entail": float("nan"),
            "n_claims": len(claims),
            "n_no_info": len(no_info),
            "n_contexts": len(ctx),
            "per_claim_max_entail": [],
        }

    model = _get_nli_model(nli_model_name)
    ent_idx = _entailment_index(model)
    # premise = passage, hypothesis = claim
    pairs = [(c, cl) for cl in claims for c in ctx]
    logits = model.predict(pairs, convert_to_numpy=True)
    probs = _softmax(np.asarray(logits, dtype=np.float64))
    ent = probs[:, ent_idx].reshape(len(claims), len(ctx))
    per_claim = ent.max(axis=1)  # "does ANY passage back this claim?"

    return {
        "faithfulness": float(np.mean(per_claim >= threshold)),
        "faithfulness_mean_entail": float(np.mean(per_claim)),
        "n_claims": len(claims),
        "n_no_info": len(no_info),
        "n_contexts": len(ctx),
        "per_claim_max_entail": [round(float(x), 6) for x in per_claim],
    }


def compute_answer_relevance(
    question: str,
    answer: str,
    *,
    sbert_model_name: str = DEFAULT_SBERT_MODEL,
    strip_answer_prefix: bool = True,
) -> Dict[str, Any]:
    """Cosine(question, answer) in SBERT space — does the answer address the ask?"""
    q = str(question or "").strip()
    a = str(answer or "").strip()
    if strip_answer_prefix:
        m = list(re.finditer(r"(?is)answer\s*[:\-]\s*", a))
        if m:
            a = a[m[-1].end():].strip() or a
    if not q or not a:
        return {"answer_relevance": float("nan")}
    enc = _get_sbert(sbert_model_name)
    v = enc.encode([q, a], convert_to_numpy=True, normalize_embeddings=True)
    return {"answer_relevance": float(np.dot(v[0], v[1]))}


def aggregate(rows: Iterable[Dict[str, Any]], keys: Sequence[str]) -> Dict[str, float]:
    """Mean over rows, skipping NaN (answers with no scorable claim)."""
    out: Dict[str, float] = {}
    rows = list(rows)
    for k in keys:
        vals = [float(r[k]) for r in rows if k in r and r[k] == r[k]]
        out[k] = sum(vals) / len(vals) if vals else float("nan")
        out[f"{k}_n"] = len(vals)
    return out


__all__ = [
    "DEFAULT_NLI_MODEL",
    "DEFAULT_SBERT_MODEL",
    "aggregate",
    "compute_answer_relevance",
    "compute_faithfulness",
    "extract_claims",
    "work_key",
]
