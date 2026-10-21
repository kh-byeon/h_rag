from __future__ import annotations

import re
from typing import List


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def lowercase(text: str) -> str:
    return (text or "").lower()


def simple_tokenize(text: str) -> List[str]:
    """Lightweight tokenizer for lexical baselines."""
    norm = lowercase(normalize_whitespace(text))
    return re.findall(r"[a-z0-9_]+", norm)
