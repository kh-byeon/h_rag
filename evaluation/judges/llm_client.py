from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional


class LLMClient:
    """Minimal OpenAI-compatible JSON generation client."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        api_key: Optional[str] = None,
    ) -> None:
        self.provider = provider.lower().strip()
        self.model = model
        self.temperature = temperature
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    def _extract_json(self, text: str) -> Dict[str, Any]:
        text = (text or "").strip()
        if not text:
            return {"_error": "empty_response"}

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return {"_error": "json_not_found", "raw": text[:2000]}
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return {"_error": "json_parse_failed", "raw": text[:2000]}

    def generate_json(self, prompt: str) -> Dict[str, Any]:
        if self.provider != "openai":
            return {"_error": f"unsupported_provider:{self.provider}"}
        if not self.api_key:
            return {"_error": "missing_api_key"}

        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key)
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            text = (resp.choices[0].message.content or "").strip()
            return self._extract_json(text)
        except Exception as e:
            return {"_error": f"request_failed:{e}"}
