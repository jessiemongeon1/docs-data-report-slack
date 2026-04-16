from __future__ import annotations

import json
import math
import time
from typing import Any

from anthropic import Anthropic, RateLimitError

PLAUSIBLE_SYSTEM = """
You analyze raw Plausible analytics JSON.

Be concise, specific, and grounded only in the provided data.
Focus on key metrics, top pages, referrals, and notable trends.
""".strip()

KAPA_CHUNK_SYSTEM = """
You analyze one raw chunk of Kapa question/answer data.

This chunk is only part of the full dataset.
Do not assume it represents the whole reporting period.

For each theme you identify:
- Count the exact number of conversations that support it (evidence_count must reflect actual Q&A items in the chunk).
- Prefer recurring issues over one-off issues.

Keep the output compact.
""".strip()

KAPA_SYNTHESIS_SYSTEM = """
You synthesize multiple chunk-level analyses of raw Kapa question/answer data into one weekly Kapa analysis.

Rules:
- Sum evidence_count values across chunks when merging the same theme.
- Report the exact total conversation count (sum of all evidence_counts across all themes).
- Merge repeated themes; do not duplicate them.
- Prefer recurring issues over one-off issues.

Keep the output compact.
""".strip()

SYNTHESIS_SYSTEM = """
You synthesize Plausible analytics and Kapa Q&A analyses into a weekly docs report.

Rules:
- State the exact total number of Kapa conversations analyzed in the executive summary.
- State the top 20 viewed pages from Plausible.
- State the correlation between top viewed pages on Plausible and top themes from Kapa.
- Include details about Plausible referral sources that are chatbots/agents.
- State the exact number of distinct themes identified.
- For notable_takeaways, every item must include concrete evidence (exact counts or metric values, not vague phrases like "several" or "many").
- For sprint_recommendations, categorize each item under exactly one of:
    - documentation_action (missing, unclear, or outdated docs)
    - tooling_action (SDK, CLI, API, or integration issues surfaced by users)
    - developer_experience_action (onboarding friction, confusing UX, workflow gaps)
  Include this category as a field in each sprint recommendation.
- Be specific, concise, and action-oriented.
- Do not repeat the same point across sections.
""".strip()


PLAUSIBLE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "key_metrics": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "value": {"type": "string"},
                    "insight": {"type": "string"},
                },
                "required": ["name", "value", "insight"],
                "additionalProperties": False,
            },
        },
        "top_pages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "page": {"type": "string"},
                    "insight": {"type": "string"},
                },
                "required": ["page", "insight"],
                "additionalProperties": False,
            },
        },
        "referrals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "insight": {"type": "string"},
                },
                "required": ["source", "insight"],
                "additionalProperties": False,
            },
        },
        "trends": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "insight": {"type": "string"},
                },
                "required": ["title", "insight"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "key_metrics", "top_pages", "referrals", "trends"],
    "additionalProperties": False,
}

KAPA_CHUNK_SCHEMA = {
    "type": "object",
    "properties": {
        "chunk_summary": {"type": "string"},
        "themes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "evidence_count": {"type": "integer"},
                    "insight": {"type": "string"},
                    "recommended_action": {"type": "string"},
                },
                "required": ["name", "evidence_count", "insight", "recommended_action"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["chunk_summary", "themes"],
    "additionalProperties": False,
}

KAPA_SYNTHESIS_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "total_conversations": {"type": "integer"},
        "total_themes": {"type": "integer"},
        "themes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "evidence_count": {"type": "integer"},
                    "insight": {"type": "string"},
                    "recommended_action": {"type": "string"},
                },
                "required": ["name", "evidence_count", "insight", "recommended_action"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "total_conversations", "total_themes", "themes"],
    "additionalProperties": False,
}

FINAL_SCHEMA = {
    "type": "object",
    "properties": {
        "executive_summary": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "total_kapa_conversations": {"type": "integer"},
                "total_themes_identified": {"type": "integer"},
                "top_priorities": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "summary",
                "total_kapa_conversations",
                "total_themes_identified",
                "top_priorities",
            ],
            "additionalProperties": False,
        },
        "notable_takeaways": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "evidence": {"type": "string"},
                    "interpretation": {"type": "string"},
                    "recommended_action": {"type": "string"},
                    "priority": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                },
                "required": [
                    "title",
                    "evidence",
                    "interpretation",
                    "recommended_action",
                    "priority",
                ],
                "additionalProperties": False,
            },
        },
        "themes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "evidence_count": {"type": "integer"},
                    "why_it_matters": {"type": "string"},
                    "recommended_doc_action": {"type": "string"},
                    "priority": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                },
                "required": [
                    "name",
                    "evidence_count",
                    "why_it_matters",
                    "recommended_doc_action",
                    "priority",
                ],
                "additionalProperties": False,
            },
        },
        "sprint_recommendations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "category": {
                        "type": "string",
                        "enum": [
                            "documentation_action",
                            "tooling_action",
                            "developer_experience_action",
                        ],
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                    "scope": {"type": "string"},
                    "why_now": {"type": "string"},
                    "expected_impact": {"type": "string"},
                },
                "required": [
                    "title",
                    "category",
                    "priority",
                    "scope",
                    "why_now",
                    "expected_impact",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "executive_summary",
        "notable_takeaways",
        "themes",
        "sprint_recommendations",
    ],
    "additionalProperties": False,
}


class ClaudePipeline:
    def __init__(self, api_key: str, model: str, max_input_tokens: int = 8000) -> None:
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.max_input_tokens = max_input_tokens

    def _sleep_from_rate_limit(self, exc: RateLimitError) -> None:
        retry_after = None
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers:
            retry_after = headers.get("retry-after")

        if retry_after:
            try:
                time.sleep(float(retry_after) + 1.0)
                return
            except ValueError:
                pass

        time.sleep(20)

    def _messages_create_with_retry(self, **kwargs: Any) -> Any:
        attempts = 0
        while True:
            attempts += 1
            try:
                return self.client.messages.create(**kwargs)
            except RateLimitError as exc:
                if attempts >= 6:
                    raise
                self._sleep_from_rate_limit(exc)

    def _structured_json(
        self,
        *,
        system_prompt: str,
        payload: dict[str, Any],
        schema: dict[str, Any],
        max_tokens: int,
    ) -> dict[str, Any]:
        response = self._messages_create_with_retry(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": schema,
                }
            },
        )

        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason == "max_tokens":
            raise ValueError(
                f"Claude output was truncated at max_tokens={max_tokens}. Increase max_tokens or shrink the schema/output."
            )

        text_parts: list[str] = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)

        text = "".join(text_parts).strip()
        if not text:
            raise ValueError("Claude returned no structured output text")

        return json.loads(text)

    def _estimate_tokens(self, obj: Any) -> int:
        text = json.dumps(obj, ensure_ascii=False)
        return max(1, math.ceil(len(text) / 4))

    def _normalize_qa_items(self, question_answers: Any) -> list[Any]:
        if isinstance(question_answers, list):
            return question_answers
        if isinstance(question_answers, dict):
            if isinstance(question_answers.get("results"), list):
                return question_answers["results"]
            if isinstance(question_answers.get("data"), list):
                return question_answers["data"]
            return [question_answers]
        return [question_answers]

    def _chunk_by_local_size(
        self,
        items: list[Any],
        field_name: str,
        base_payload: dict[str, Any],
        target_tokens: int,
    ) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        current_items: list[Any] = []
        current_tokens = self._estimate_tokens(base_payload)

        for item in items:
            item_tokens = self._estimate_tokens(item)
            if current_items and current_tokens + item_tokens > target_tokens:
                chunks.append({**base_payload, "raw": {field_name: current_items}})
                current_items = [item]
                current_tokens = self._estimate_tokens(base_payload) + item_tokens
            else:
                current_items.append(item)
                current_tokens += item_tokens

        if current_items:
            chunks.append({**base_payload, "raw": {field_name: current_items}})

        return chunks

    def analyze_plausible_raw(self, plausible_raw: dict[str, Any]) -> dict[str, Any]:
        return self._structured_json(
            system_prompt=PLAUSIBLE_SYSTEM,
            payload={"source": "plausible", "raw": plausible_raw},
            schema=PLAUSIBLE_SCHEMA,
            max_tokens=2500,
        )

    def analyze_kapa_raw(self, kapa_raw: dict[str, Any]) -> dict[str, Any]:
        initial_payload = {"source": "kapa", "raw": kapa_raw}
        initial_tokens = self._estimate_tokens(initial_payload)

        if initial_tokens <= self.max_input_tokens:
            return self._structured_json(
                system_prompt=KAPA_SYNTHESIS_SYSTEM,
                payload=initial_payload,
                schema=KAPA_SYNTHESIS_SCHEMA,
                max_tokens=8000,
            )

        qa_items = self._normalize_qa_items(kapa_raw.get("question_answers", []))

        target_tokens = min(self.max_input_tokens, 8000)
        chunks = self._chunk_by_local_size(
            qa_items,
            "question_answers",
            {
                "source": "kapa",
                "chunk_type": "question_answers",
                "project_id": kapa_raw.get("project_id"),
            },
            target_tokens,
        )

        print(f"Kapa estimated tokens: {initial_tokens}")
        print(f"Kapa chunk count: {len(chunks)}")

        chunk_analyses: list[dict[str, Any]] = []
        for i, chunk in enumerate(chunks, start=1):
            print(f"Analyzing Kapa chunk {i}/{len(chunks)}")
            chunk_analyses.append(
                self._structured_json(
                    system_prompt=KAPA_CHUNK_SYSTEM,
                    payload=chunk,
                    schema=KAPA_CHUNK_SCHEMA,
                    max_tokens=3500,
                )
            )

        return self._structured_json(
            system_prompt=KAPA_SYNTHESIS_SYSTEM,
            payload={
                "source": "kapa",
                "project_id": kapa_raw.get("project_id"),
                "chunk_analyses": chunk_analyses,
            },
            schema=KAPA_SYNTHESIS_SCHEMA,
            max_tokens=8000,
        )

    def synthesize(
        self,
        metadata: dict[str, Any],
        plausible_analysis: dict[str, Any],
        kapa_analysis: dict[str, Any],
    ) -> dict[str, Any]:
        return self._structured_json(
            system_prompt=SYNTHESIS_SYSTEM,
            payload={
                "metadata": metadata,
                "plausible_analysis": plausible_analysis,
                "kapa_analysis": kapa_analysis,
            },
            schema=FINAL_SCHEMA,
            max_tokens=8000,
        )
