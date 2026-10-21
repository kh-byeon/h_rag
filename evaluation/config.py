from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional


CompareMode = Literal["pairwise", "single"]


@dataclass
class EvalConfig:
    """Runtime configuration for legal QA evaluation."""

    input_path: str
    output_dir: str
    sbert_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    nli_model_name: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    coverage_similarity_threshold: float = 0.5
    llm_judge_provider: str = "openai"
    llm_judge_model: str = "gpt-4o-mini"
    llm_judge_temperature: float = 0.0
    compare_mode: CompareMode = "pairwise"
    max_samples: Optional[int] = None


def load_config(config_or_path: Dict[str, Any] | str) -> EvalConfig:
    """Load config from a dictionary or JSON file path."""
    if isinstance(config_or_path, str):
        with open(config_or_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raw = dict(config_or_path)
    return EvalConfig(**raw)
