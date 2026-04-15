from __future__ import annotations

import json
import re
from typing import Any

from anthropic import Anthropic

PLAUSIBLE_SYSTEM = """
You analyze raw Plausible analytics JSON.

Return valid JSON only with this shape:
{
  "summary": "string",
  "key_metrics": [
    {"name": "string", "value": "string", "insight": "string"}
  ],
  "top_pages": [
    {"page": "string", "insight": "string"}
  ],
  "referrals": [
    {"source": "string", "insight": "string"}
  ],
  "trends": [
    {"title": "string", "insight": "string"}
  ]
}
Do not include markdown fences.
Do not include commentary outside JSON.
""".strip()

KAPA_SYSTEM = """
You analyze raw Kapa JSON.

Return valid JSON only with this shape:
{
  "summary": "string",
  "themes": [
    {
      "name": "string",
      "evidence_count": 0,
      "examples": ["string"],
      "insight": "string",
      "recommended_action": "string"
    }
  ],
  "notable_threads": [
    {
      "title": "string",
      "insight": "string",
      "recommended_action": "string"
    }
  ]
}
Do not include markdown fences.
Do not include commentary outside JSON.
""".strip()

SYNTHESIS_SYSTEM = """
You synthesize Plausible analysis and Kapa analysis into a weekly docs report.

Return valid JSON only with this shape:
{
  "executive_summary": {
    "summary": "string",
    "top_priorities": ["string"]
  },
  "notable_takeaways": [
    {
      "title": "string",
      "evidence": "string",
      "interpretation": "string",
      "recommended_action": "string",
      "priority": "high|medium|low"
    }
  ],
  "themes": [
    {
      "name": "string",
      "evidence_count": 0,
      "representative_examples": ["string"],
      "why_it_matters": "string",
      "recommended_doc_action": "string",
      "priority": "high|medium|low"
    }
  ],
  "sprint_recommendations": [
    {
      "title": "string",
      "priority": "high|medium|low",
      "scope": "string",
      "why_now": "string",
      "expected_impact": "string"
    }
  ]
}
Do not include markdown fences.
Do not include commentary outside JSON.
""".strip()


class ClaudePipeline:
    def __init__(self, api_key: str, model: str, max_input_tokens: int = 120000) -> None:
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.max_input_tokens = max_input_tokens

    def _extract_text(self, response: Any) -> str:
        parts: list[str] = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        return "".join(parts).strip()

    def _strip_code_fences(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        return text.strip()

    def _extract_json_object(self, text: str) -> str:
        text = self._strip_code_fences(text)
        if not text:
            raise ValueError("Claude returned empty text")

        # happy path
        if text.startswith("{") or text.startswith("["):
            return text

        # try to find first JSON object/array in surrounding prose
        obj_start = text.find("{")
        arr_start = text.find("[")
        starts = [i for i in [obj_start, arr_start] if i != -1]
        if not starts:
            raise ValueError(f"Claude returned no JSON:\n{text[:1000]}")

        start = min(starts)
        candidate = text[start:].strip()

        # try full tail first
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

        # fallback: greedy brace extraction for object
        if candidate.startswith("{"):
            depth = 0
            for i, ch in enumerate(candidate):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return candidate[: i + 1]

        # fallback: greedy bracket extraction for array
        if candidate.startswith("["):
            depth = 0
            for i, ch in enumerate(candidate):
                if ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        return candidate[: i + 1]

        return candidate

    def _repair_json(self, system_prompt: str, bad_text: str) -> dict[str, Any]:
        repair_prompt = f"""
Your previous response was not valid JSON.

Return the same information again as valid JSON only.
Do not use markdown fences.
Do not add explanation.

Previous invalid output:
{bad_text[:12000]}
""".strip()

        response = self.client.messages.create(
            model=self.model,
            max_tokens=3000,
            system=system_prompt,
            messages=[{"role": "user", "content": repair_prompt}],
        )
        repaired_text = self._extract_text(response)
        repaired_json = self._extract_json_object(repaired_text)
        return json.loads(repaired_json)

    def message_json(self, system_prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
        user_content = json.dumps(payload, ensure_ascii=False)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )

        text = self._extract_text(response)
        try:
            json_text = self._extract_json_object(text)
            return json.loads(json_text)
        except Exception:
            return self._repair_json(system_prompt, text)

    def analyze_plausible_raw(self, plausible_raw: dict[str, Any]) -> dict[str, Any]:
        payload = {"source": "plausible", "raw": plausible_raw}
        return self.message_json(PLAUSIBLE_SYSTEM, payload)

    def analyze_kapa_raw(self, kapa_raw: dict[str, Any]) -> dict[str, Any]:
        payload = {"source": "kapa", "raw": kapa_raw}
        return self.message_json(KAPA_SYSTEM, payload)

    def synthesize(
        self,
        metadata: dict[str, Any],
        plausible_analysis: dict[str, Any],
        kapa_analysis: dict[str, Any],
    ) -> dict[str, Any]:
        payload = {
            "metadata": metadata,
            "plausible_analysis": plausible_analysis,
            "kapa_analysis": kapa_analysis,
        }
        return self.message_json(SYNTHESIS_SYSTEM, payload)
