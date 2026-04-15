from __future__ import annotations

import json
from typing import Any

from anthropic import Anthropic

from app.utils import compact_json


PLAUSIBLE_SYSTEM = """
You are analyzing raw Plausible Analytics JSON for a weekly docs report.
Use only the provided raw JSON. Do not invent fields or metrics.
Return valid JSON with these keys:
- key_metrics
- traffic_patterns
- engagement_findings
- page_findings
- referral_findings
- caveats
""".strip()

KAPA_SYSTEM = """
You are analyzing raw Kapa conversation and analytics JSON for a weekly docs/support report.
Use only the provided raw JSON. Do not invent fields or counts.
Prefer recurring support patterns over one-off issues.
Return valid JSON with these keys:
- chunk_summary
- themes
- repeated_questions
- notable_conversations
- feedback_findings
- caveats
""".strip()

SYNTHESIS_SYSTEM = """
You are combining raw-analysis outputs from Plausible and Kapa into a weekly report.
Use only the supplied analyses.
Return valid JSON with exactly these keys:
- executive_summary
- notable_takeaways
- themes
- sprint_recommendations
""".strip()


class ClaudePipeline:
    def __init__(self, api_key: str, model: str, max_input_tokens: int) -> None:
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.max_input_tokens = max_input_tokens

    def count_tokens(self, system: str, user_payload: Any) -> int:
        resp = self.client.messages.count_tokens(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": compact_json(user_payload)}],
        )
        return int(resp.input_tokens)

    def message_json(self, system: str, user_payload: Any, max_tokens: int = 4000) -> dict[str, Any]:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": compact_json(user_payload)}],
        )
        chunks: list[str] = []
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                chunks.append(block.text)
        text = "".join(chunks).strip()
        return json.loads(text)

    def chunk_list_payload(self, items: list[Any], wrapper_key: str) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        current: list[Any] = []
        for item in items:
            candidate = {wrapper_key: current + [item]}
            tokens = self.count_tokens(KAPA_SYSTEM, candidate)
            if current and tokens > self.max_input_tokens:
                chunks.append({wrapper_key: current})
                current = [item]
            else:
                current.append(item)
        if current:
            chunks.append({wrapper_key: current})
        return chunks

    def analyze_plausible_raw(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.message_json(PLAUSIBLE_SYSTEM, payload)

    def analyze_kapa_raw(self, payload: dict[str, Any]) -> dict[str, Any]:
        conversations = payload.get("conversations", {})
        items = conversations.get("data") or conversations.get("conversations") or []
        if not isinstance(items, list) or not items:
            return self.message_json(KAPA_SYSTEM, payload)

        shared = {k: v for k, v in payload.items() if k != "conversations"}
        chunk_payloads = self.chunk_list_payload(items, "conversations")
        chunk_analyses: list[dict[str, Any]] = []
        for index, chunk in enumerate(chunk_payloads, start=1):
            chunk_input = {**shared, **chunk, "chunk_index": index, "chunk_count": len(chunk_payloads)}
            chunk_analyses.append(self.message_json(KAPA_SYSTEM, chunk_input))
        return {
            "chunk_count": len(chunk_payloads),
            "chunk_analyses": chunk_analyses,
        }

    def synthesize(
        self,
        report_context: dict[str, Any],
        plausible_analysis: dict[str, Any],
        kapa_analysis: dict[str, Any],
    ) -> dict[str, Any]:
        return self.message_json(
            SYNTHESIS_SYSTEM,
            {
                "report_context": report_context,
                "plausible_analysis": plausible_analysis,
                "kapa_analysis": kapa_analysis,
            },
        )
