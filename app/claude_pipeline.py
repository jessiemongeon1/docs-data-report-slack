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
Return exactly 20 pages in top_pages (or all pages if fewer than 20 exist).
""".strip()

KAPA_CHUNK_SYSTEM = """
You analyze one raw chunk of Kapa question/answer data.

This chunk is only part of the full dataset.
Do not assume it represents the whole reporting period.

For each theme you identify:
- Count the exact number of conversations that support it (evidence_count must reflect actual Q&A items in the chunk).
- Prefer recurring issues over one-off issues.
- Return at most 5 themes. Merge minor ones into the closest major theme rather than listing them separately.
- chunk_summary: 1 sentence only.
- insight and recommended_action: 1 sentence each, under 20 words.

You MUST also return a classified_conversations array that maps every conversation in the chunk to one of your identified theme names. For each conversation include:
- theme: the exact name of one of your themes (must match a name in the themes array)
- index: the zero-based index of the conversation in the input question_answers array
""".strip()

KAPA_SYNTHESIS_SYSTEM = """
You synthesize multiple chunk-level analyses of raw Kapa question/answer data into one weekly Kapa analysis.

Rules:
- Sum evidence_count values across chunks when merging the same theme.
- Report the exact total conversation count (sum of all evidence_counts across all themes).
- Merge repeated themes; do not duplicate them.
- Prefer recurring issues over one-off issues.
- Return at most 7 themes total.
- summary: 2 sentences max.
- insight and recommended_action: 1 sentence each, under 20 words.
""".strip()

SYNTHESIS_SYSTEM = """
You synthesize Plausible analytics and Kapa Q&A analyses into a weekly docs report.

SECTION RESPONSIBILITIES — each piece of information appears in exactly one place:
- executive_summary: high-level numbers and a 2-sentence summary only. No action lists.
- notable_takeaways: the 5 most surprising or high-impact cross-signal observations (Plausible + Kapa together). Each takeaway must contain a specific metric or count not already stated elsewhere. Do NOT restate theme names or sprint titles here.
- themes: Kapa pain points only — what developers are struggling with and why it matters. No recommended actions here; those go in sprint_recommendations.
- page_theme_correlations: traffic/behavior signal from Plausible mapped to a Kapa theme. One insight per row, nowhere else.
- sprint_recommendations: the concrete work to do. Title must differ from theme names and takeaway titles. Each item is self-contained; do not reference or repeat evidence already in takeaways or themes.
- chatbot_referrals: AI referral sources only; no overlap with other sections.

Output limits — strictly enforce these:
- executive_summary.summary: 2 sentences max.
- page_theme_correlations: at most 8 items. insight: 1 sentence, under 15 words.
- notable_takeaways: at most 4 items. evidence: exact metric/count only (no prose). interpretation: 1 sentence. recommended_action: omit — actions go in sprint_recommendations only.
- themes: at most 6 items. why_it_matters: 1 sentence, under 20 words. recommended_doc_action: omit — actions go in sprint_recommendations only.
- sprint_recommendations: at most 7 items total. scope, why_now, expected_impact: 1 sentence each, under 20 words.

Content rules:
- State the exact total number of Kapa conversations and distinct themes in the executive summary.
- In chatbot_referrals, include every referral source identifiable as a chatbot or AI agent.
- For notable_takeaways, evidence must be a raw number or metric — never "several", "many", or vague phrases.
- For sprint_recommendations, category must be exactly one of:
    - documentation_action: changes to existing documentation pages (new guides, clarifications, restructuring, missing reference content, broken examples).
    - tooling_action: improvements to existing developer tooling (better CLI error messages, clearer Move compiler diagnostics, SDK error handling, IDE plugin fixes, framework documentation gaps that should be solved by tooling output rather than docs). Tooling actions improve tools that already exist.
    - developer_experience_action: net-new capabilities that do not yet exist (new tools, new SDK features, new CLI commands, new dashboards, new APIs, new observability surfaces, new templates). DX actions propose new things to build, not fixes to existing things.
- Each sprint recommendation must belong to exactly one category. Do not duplicate the same recommendation across categories. If a finding could fit multiple categories, place it in the most specific one (tooling_action over documentation_action when the underlying issue is tool output; developer_experience_action only when the proposed solution does not yet exist in any form).
- If you find yourself writing the same finding in two sections, delete it from the lower-priority section. Priority order: sprint_recommendations > notable_takeaways > themes > executive_summary.
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
                    "is_chatbot_or_agent": {"type": "boolean"},
                    "insight": {"type": "string"},
                },
                "required": ["source", "is_chatbot_or_agent", "insight"],
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
        "classified_conversations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "theme": {"type": "string"},
                    "index": {"type": "integer"},
                },
                "required": ["theme", "index"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["chunk_summary", "themes", "classified_conversations"],
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
            },
            "required": [
                "summary",
                "total_kapa_conversations",
                "total_themes_identified",
            ],
            "additionalProperties": False,
        },
        "page_theme_correlations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "page": {"type": "string"},
                    "related_kapa_theme": {"type": "string"},
                    "insight": {"type": "string"},
                },
                "required": ["page", "related_kapa_theme", "insight"],
                "additionalProperties": False,
            },
        },
        "chatbot_referrals": {
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
        "notable_takeaways": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "evidence": {"type": "string"},
                    "interpretation": {"type": "string"},
                    "priority": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                },
                "required": [
                    "title",
                    "evidence",
                    "interpretation",
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
                    "priority": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                },
                "required": [
                    "name",
                    "evidence_count",
                    "why_it_matters",
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
        "page_theme_correlations",
        "chatbot_referrals",
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

        # Use most of the model's input budget per chunk so we don't make
        # hundreds of tiny API calls. Leave headroom for the system prompt and
        # response budget.
        target_tokens = max(20000, self.max_input_tokens - 20000)
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

        # Hard ceiling on chunk count to prevent a runaway billing event if
        # something upstream (e.g. Kapa returning duplicates) inflates input
        # size. Increase if you legitimately need to process more.
        max_chunks = 50
        if len(chunks) > max_chunks:
            raise RuntimeError(
                f"Kapa produced {len(chunks)} chunks (cap is {max_chunks}). "
                f"This usually means the input data is unexpectedly large or "
                f"contains duplicates. Aborting before incurring large API costs. "
                f"Inspect the Kapa raw payload or raise max_chunks if intentional."
            )

        print(f"Kapa estimated tokens: {initial_tokens}")
        print(f"Kapa target tokens per chunk: {target_tokens}")
        print(f"Kapa chunk count: {len(chunks)}")

        chunk_analyses: list[dict[str, Any]] = []
        all_classified: list[dict[str, Any]] = []
        for i, chunk in enumerate(chunks, start=1):
            print(f"Analyzing Kapa chunk {i}/{len(chunks)}")
            result = self._structured_json(
                system_prompt=KAPA_CHUNK_SYSTEM,
                payload=chunk,
                schema=KAPA_CHUNK_SCHEMA,
                max_tokens=4000,
            )
            chunk_analyses.append(result)

            # Resolve indices back to the original QA items in this chunk.
            chunk_items = chunk.get("raw", {}).get("question_answers", [])
            for cc in result.get("classified_conversations", []):
                idx = cc.get("index", -1)
                if 0 <= idx < len(chunk_items):
                    item = chunk_items[idx]
                    all_classified.append({
                        "theme": cc["theme"],
                        "question": (item.get("question") or "")[:300],
                        "answer_snippet": (item.get("answer") or "")[:200],
                        "thread_id": item.get("thread_id"),
                    })

        synthesis = self._structured_json(
            system_prompt=KAPA_SYNTHESIS_SYSTEM,
            payload={
                "source": "kapa",
                "project_id": kapa_raw.get("project_id"),
                "chunk_analyses": chunk_analyses,
            },
            schema=KAPA_SYNTHESIS_SCHEMA,
            max_tokens=8000,
        )

        # Attach the classified conversations so downstream consumers
        # (e.g. the HTML report) can show full conversation detail per theme.
        synthesis["classified_conversations"] = all_classified
        return synthesis

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
