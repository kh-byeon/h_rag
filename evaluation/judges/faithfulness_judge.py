from __future__ import annotations

from typing import Any, Dict

from evaluation.judges.llm_client import LLMClient


def build_faithfulness_prompt(query: str, ground_truth: str, answer: str) -> str:
    return f"""
You are a strict legal faithfulness judge.
Evaluate the answer only against the provided ground truth.

Rules:
- Do not reward fluent paraphrase if it introduces unsupported legal meaning.
- Penalize unsupported additions, contradictions, and legally meaningful distortions.
- Penalize claims like "perpetual" unless explicitly supported.
- Treat omission as a faithfulness issue only if the answer adds a misleading implication.

Return JSON only:
{{
  "score": 1-5,
  "supported_claims": ["..."],
  "unsupported_claims": ["..."],
  "contradictions": ["..."],
  "reasoning": "short string"
}}

[Query]
{query}

[Ground Truth]
{ground_truth}

[Answer]
{answer}
""".strip()


def judge_faithfulness(
    client: LLMClient,
    query: str,
    ground_truth: str,
    answer: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    if dry_run:
        return {
            "score": None,
            "supported_claims": [],
            "unsupported_claims": [],
            "contradictions": [],
            "reasoning": "dry_run: judge skipped",
        }
    prompt = build_faithfulness_prompt(query, ground_truth, answer)
    return client.generate_json(prompt)
