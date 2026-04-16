from __future__ import annotations

from typing import Any
import json
import math
import time

from anthropic import Anthropic, RateLimitError

PLAUSIBLE_SYSTEM = """
You analyze raw Plausible analytics JSON.
Be concise, specific, and grounded only in the provided data.
""".strip()

KAPA_CHUNK_SYSTEM = """
You analyze one raw chunk of Kapa question/answer data.

This chunk is only part of the full dataset.
Do not assume it represents the whole reporting period.
Prefer recurring issues over one-off issues.
""".strip()

KAPA_SYNTHESIS_SYSTEM = """
You synthesize multiple chunk-level analyses of raw Kapa question/answer data into one weekly Kapa analysis.

Merge repeated themes across chunks.
Prefer recurring issues over one-off issues.
""".strip()

SYNTHESIS_SYSTEM = """
You synthesize Plausible analysis and Kapa analysis into a weekly docs report.

Use only the provided analyses.
Be specific, concise, and action-oriented.
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
                    "examples": {"type": "array", "items": {"type": "string"}},
                    "insight": {"type": "string"},
                    "recommended_action": {"type": "string"},
                },
                "required": ["name", "evidence_count", "examples", "insight", "recommended_action"],
                "additionalProperties": False,
            },
        },
        "notable_threads": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "insight": {"type": "string"},
                    "recommended_action": {"type": "string"},
                },
                "required": ["title", "insight", "recommended_action"],
                "additionalProperties": False,
            },
        },
        "open_questions": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["chunk_summary", "themes", "notable_threads", "open_questions"],
    "additionalProperties": False,
}

KAPA_SYNTHESIS_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "themes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "evidence_count": {"type": "integer"},
                    "examples": {"type": "array", "items": {"type": "string"}},
                    "insight": {"type": "string"},
                    "recommended_action": {"type": "string"},
                },
                "required": ["name", "evidence_count", "examples", "insight", "recommended_action"],
                "additionalProperties": False,
            },
        },
        "notable_threads": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "insight": {"type": "string"},
                    "recommended_action": {"type": "string"},
                },
                "required": ["title", "insight", "recommended_action"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "themes", "notable_threads"],
    "additionalProperties": False,
}

FINAL_SCHEMA = {
    "type": "object",
    "properties": {
        "executive_summary": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "top_priorities": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["summary", "top_priorities"],
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
                    "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                },
                "required": ["title", "evidence", "interpretation", "recommended_action", "priority"],
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
                    "representative_examples": {"type": "array", "items": {"type": "string"}},
                    "why_it_matters": {"type": "string"},
                    "recommended_doc_action": {"type": "string"},
                    "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                },
                "required": [
                    "name",
                    "evidence_count",
                    "representative_examples",
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
                    "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                    "scope": {"type": "string"},
                    "why_now": {"type": "string"},
                    "expected_impact": {"type": "string"},
                },
                "required": ["title", "priority", "scope", "why_now", "expected_impact"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["executive_summary", "notable_takeaways", "themes", "sprint_recommendations"],
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
        schema_name: str,
        schema: dict[str, Any],
        max_tokens: int = 2200,
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

        structured = getattr(response, "structured_output", None)
        if structured is not None:
            return structured

        # fallback for SDK versions that surface the JSON in content text
        text_parts = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)
        text = "".join(text_parts).strip()
        if not text:
            raise ValueError("Claude returned no structured output")
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
            schema_name="plausible_analysis",
            schema=PLAUSIBLE_SCHEMA,
        )

    def analyze_kapa_raw(self, kapa_raw: dict[str, Any]) -> dict[str, Any]:
        initial_payload = {"source": "kapa", "raw": kapa_raw}
        initial_tokens = self._estimate_tokens(initial_payload)

        if initial_tokens <= self.max_input_tokens:
            return self._structured_json(
                system_prompt=KAPA_SYNTHESIS_SYSTEM,
                payload=initial_payload,
                schema_name="kapa_analysis",
                schema=KAPA_SYNTHESIS_SCHEMA,
            )

        qa_items = self._normalize_qa_items(kapa_raw.get("question_answers", []))
        target_tokens = min(self.max_input_tokens, 8000)
        chunks = self._chunk_by_local_size(
            qa_items,
            "question_answers",
            {"source": "kapa", "chunk_type": "question_answers", "project_id": kapa_raw.get("project_id")},
            target_tokens,
        )

        chunk_analyses: list[dict[str, Any]] = []
        for i, chunk in enumerate(chunks, start=1):
            print(f"Analyzing Kapa chunk {i}/{len(chunks)}")
            chunk_analyses.append(
                self._structured_json(
                    system_prompt=KAPA_CHUNK_SYSTEM,
                    payload=chunk,
                    schema_name="kapa_chunk_analysis",
                    schema=KAPA_CHUNK_SCHEMA,
                )
            )

        return self._structured_json(
            system_prompt=KAPA_SYNTHESIS_SYSTEM,
            payload={
                "source": "kapa",
                "project_id": kapa_raw.get("project_id"),
                "chunk_analyses": chunk_analyses,
            },
            schema_name="kapa_synthesis",
            schema=KAPA_SYNTHESIS_SCHEMA,
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
            schema_name="weekly_report_analysis",
            schema=FINAL_SCHEMA,
            max_tokens=2600,
        )
