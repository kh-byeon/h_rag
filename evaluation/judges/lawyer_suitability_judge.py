from __future__ import annotations

from typing import Any, Dict, List

from evaluation.judges.llm_client import LLMClient


def build_lawyer_suitability_prompt(
    query: str,
    aspect_queries: List[str],
    ground_truth: str,
    answer: str,
) -> str:
    aspect_block = "\n".join(f"- {a}" for a in aspect_queries)
    return f"""
You are evaluating whether an answer is suitable for lawyer review.

A legally useful answer should not only identify the main conclusion.
It should also preserve legally material qualifiers and review-worthy limitations.
Omission of carve-outs, sublicensing restrictions, transfer limits, or party-direction errors should be penalized.

Focus:
- scope of the clause
- rights and obligations
- exclusions / carve-outs / limitations
- conditions / procedures / assignment-related constraints

Return JSON only:
{{
  "score": 1-5,
  "strengths": ["..."],
  "missing_legal_points": ["..."],
  "risky_omissions": ["..."],
  "reasoning": "short string"
}}

[Query]
{query}

[Aspect Queries]
{aspect_block}

[Ground Truth]
{ground_truth}

[Answer]
{answer}
""".strip()


def judge_lawyer_suitability(
    client: LLMClient,
    query: str,
    aspect_queries: List[str],
    ground_truth: str,
    answer: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    if dry_run:
        return {
            "score": None,
            "strengths": [],
            "missing_legal_points": [],
            "risky_omissions": [],
            "reasoning": "dry_run: judge skipped",
        }
    prompt = build_lawyer_suitability_prompt(query, aspect_queries, ground_truth, answer)
    return client.generate_json(prompt)
