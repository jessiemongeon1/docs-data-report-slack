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

DOCS_CATEGORIES = [
    "Accessing Data",
    "Cryptography",
    "Manage Packages",
    "Objects",
    "Publish & Upgrade Packages",
    "Security",
    "Sui Architecture",
    "Testing & Debugging",
    "Transaction Payment",
    "Transactions",
    "Write Move",
    "Other",
]

KAPA_CHUNK_SYSTEM = """
You analyze one raw chunk of Kapa question/answer data.

This chunk is only part of the full dataset.
Do not assume it represents the whole reporting period.

For each theme you identify:
- Count the exact number of questions that support it (evidence_count must reflect actual Q&A items in the chunk).
- Prefer recurring issues over one-off issues.
- Return at most 8 themes. Merge minor ones into the closest major theme rather than listing them separately.
- chunk_summary: 1 sentence only.
- insight and recommended_action: 1 sentence each, under 20 words.

You MUST also return a classified_questions array that maps every question in the chunk. For each question include:

- category: assign each question to EXACTLY ONE of these fixed categories (matching docs.sui.io/develop):
  "Accessing Data" — GraphQL, gRPC, JSON-RPC, indexers, data queries, reading chain state, event subscriptions
  "Cryptography" — key schemes, signatures, hashing, zkLogin, multisig, key management
  "Manage Packages" — dependency management, Move.toml, package configuration
  "Objects" — object model, ownership, shared objects, dynamic fields, display, wrapping, object IDs, transfers between addresses
  "Publish & Upgrade Packages" — publishing, upgrading, versioning, sui client publish, upgrade policies
  "Security" — audit, access control, capability patterns, Seal encryption
  "Sui Architecture" — consensus, validators, epochs, networks (devnet/testnet/mainnet), full nodes, state sync, protocol
  "Testing & Debugging" — unit tests, debugging, local development, simulators, error diagnosis
  "Transaction Payment" — gas, gas estimation, sponsored transactions, gas coins, gas budgets
  "Transactions" — PTBs, transaction building, signing, executing, Move calls, splitCoins, mergeCoins, transaction effects
  "Write Move" — Move language, structs, generics, abilities, modules, functions, standard library, Coin/Balance, events
  "Other" — wallet setup, ecosystem questions, non-technical questions, off-topic, greetings, questions not fitting above categories

- topic: a specific label within the category for what this question is about.
  RULES:
  - Be specific but REUSABLE — multiple questions about the same feature MUST get the same topic label.
  - GOOD: "GraphQL Pagination", "gRPC getCheckpoint", "PTB splitCoins", "Move Generics", "Object Ownership", "Gas Estimation", "zkLogin Setup"
  - BAD (too unique): "splitCoins type error on line 42", "how to query checkpoint 12345"
  - BAD (too broad): "Data", "Errors", "Transactions", "General"
  - Think of it as a docs page title — specific enough to be useful, general enough that 2-10 questions share it.
  - Use consistent naming: if one question gets "Move Generics", another about generic constraints should also get "Move Generics".

- theme: the exact name of one of your themes (must match a name in the themes array) — used only for the high-level summary
- index: the zero-based index of the question in the input question_answers array
""".strip()

KAPA_SYNTHESIS_SYSTEM = """
You synthesize multiple chunk-level analyses of raw Kapa question/answer data into one weekly Kapa analysis.

You produce a single unified list of topics. Each topic has both a certain_count and an uncertain_count.

Each chunk analysis contains a classified_questions array where each question has a "confidence" field ("certain" or "uncertain") and a "theme" field. Use these to:
1. Merge chunk-level topics that are about the same specific feature or task across chunks.
2. For each merged topic, count how many questions were "certain" and how many were "uncertain".
3. Set evidence_count = certain_count + uncertain_count.

TOPIC QUALITY RULES:
- Keep topics specific. Only merge chunk topics if they genuinely refer to the same feature/concept.
- Do NOT merge topics just because they are loosely related (e.g. "GraphQL Pagination" and "GraphQL Type Errors" should stay separate).
- Prefer more topics with precise names over fewer topics with broad names.
- Return at most 12 topics. Drop one-off topics (evidence_count=1) if you need room.

Rules:
- Report the exact total question count (sum of all evidence_counts).
- Merge repeated topics; do not duplicate.
- summary: 2 sentences max.
- insight and recommended_action: 1 sentence each, under 20 words.
""".strip()

SYNTHESIS_SYSTEM = """
You synthesize Plausible analytics and Kapa Q&A analyses into a weekly docs report.

SECTION RESPONSIBILITIES — each piece of information appears in exactly one place:
- executive_summary: high-level numbers and a 2-sentence summary only. No action lists.
- notable_takeaways: the 5 most surprising or high-impact cross-signal observations (Plausible + Kapa together). Each takeaway must contain a specific metric or count not already stated elsewhere. Do NOT restate theme names or sprint titles here.
- themes: Topics developers are asking about, each with certain_count (bot answered well) and uncertain_count (bot struggled). No recommended actions here; those go in sprint_recommendations.
- page_theme_correlations: traffic/behavior signal from Plausible mapped to a Kapa theme. One insight per row, nowhere else.
- sprint_recommendations: the concrete work to do. Title must differ from theme names and takeaway titles. Each item is self-contained; do not reference or repeat evidence already in takeaways or themes.
- chatbot_referrals: AI referral sources only; no overlap with other sections.

Output limits — strictly enforce these:
- executive_summary.summary: 2 sentences max.
- page_theme_correlations: at most 8 items. insight: 1 sentence, under 15 words.
- notable_takeaways: at most 4 items. evidence: exact metric/count only (no prose). interpretation: 1 sentence. recommended_action: omit — actions go in sprint_recommendations only.
- themes: at most 12 items. Keep topic names specific. why_it_matters: 1 sentence, under 20 words.
- sprint_recommendations: exactly 7 items total, distributed as follows:
    - At least 3 documentation_action items
    - At least 2 tooling_action items
    - At least 2 developer_experience_action items
  scope, why_now, expected_impact: 1 sentence each, under 20 words.

Content rules:
- Do NOT include raw counts of questions or themes in the executive_summary.summary text — those are shown separately as structured fields. The summary should focus on the key findings and actionable insights.
- In chatbot_referrals, include every referral source identifiable as a chatbot or AI agent.
- For notable_takeaways, evidence must be a raw number or metric — never "several", "many", or vague phrases.
- For sprint_recommendations, category must be exactly one of:
    - documentation_action: changes to existing documentation pages (new guides, clarifications, restructuring, missing reference content, broken examples).
    - tooling_action: improvements to existing engineering-owned developer tools. Derive these from patterns in what developers ask about — if many questions involve CLI errors, recommend better error messages; if questions show confusion about SDK behavior, recommend improved SDK output or validation; if questions reveal common pitfalls with Move compilation, recommend compiler warnings. Examples: better `sui client` error messages, SDK method validation improvements, Move compiler diagnostic improvements, IDE plugin enhancements, framework API ergonomics, CLI help text improvements. These are code changes an engineering team ships, NOT documentation and NOT third-party services like Kapa.ai.
    - developer_experience_action: net-new engineering capabilities that do not yet exist. Derive these from gaps developers encounter — if developers repeatedly need to do something manually, recommend a tool for it; if there is no way to inspect something, recommend a new command or API. Examples: new CLI commands, new SDK utilities, new APIs, new developer dashboards, new debugging tools, new project templates, new observability surfaces. These are things for engineering to build from scratch, NOT docs and NOT third-party services like Kapa.ai.
- CRITICAL: You MUST include at least 2 tooling_action and at least 2 developer_experience_action items. If you cannot find strong signals, look harder at the Kapa questions — every developer pain point implies either a tool that could work better (tooling_action) or a tool that should exist but does not (developer_experience_action). Think about what an SDK, CLI, or framework engineer would want to know from this data.
- tooling_action and developer_experience_action must ONLY contain recommendations that require engineering work (code changes to tools, SDKs, CLIs, frameworks, or APIs). They must NEVER contain documentation changes, content rewrites, or improvements to third-party services like Kapa.ai.
- Each sprint recommendation must belong to exactly one category. Do not duplicate the same recommendation across categories.
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
        "classified_questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": DOCS_CATEGORIES,
                    },
                    "topic": {"type": "string"},
                    "theme": {"type": "string"},
                    "index": {"type": "integer"},
                },
                "required": ["category", "topic", "theme", "index"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["chunk_summary", "themes", "classified_questions"],
    "additionalProperties": False,
}

KAPA_SYNTHESIS_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "total_questions": {"type": "integer"},
        "total_themes": {"type": "integer"},
        "themes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "evidence_count": {"type": "integer"},
                    "certain_count": {"type": "integer"},
                    "uncertain_count": {"type": "integer"},
                    "insight": {"type": "string"},
                    "recommended_action": {"type": "string"},
                },
                "required": ["name", "evidence_count", "certain_count", "uncertain_count", "insight", "recommended_action"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "total_questions", "total_themes", "themes"],
    "additionalProperties": False,
}

FINAL_SCHEMA = {
    "type": "object",
    "properties": {
        "executive_summary": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "total_kapa_questions": {"type": "integer"},
                "total_themes_identified": {"type": "integer"},
            },
            "required": [
                "summary",
                "total_kapa_questions",
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
                    "certain_count": {"type": "integer"},
                    "uncertain_count": {"type": "integer"},
                    "why_it_matters": {"type": "string"},
                    "priority": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                },
                "required": [
                    "name",
                    "evidence_count",
                    "certain_count",
                    "uncertain_count",
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
            max_tokens=5000,
        )

    def analyze_kapa_raw(self, kapa_raw: dict[str, Any]) -> dict[str, Any]:
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

        estimated_tokens = self._estimate_tokens({"source": "kapa", "raw": kapa_raw})
        print(f"Kapa estimated tokens: {estimated_tokens}")
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
                max_tokens=16000,
            )
            chunk_analyses.append(result)

            # Resolve indices back to the original QA items in this chunk.
            chunk_items = chunk.get("raw", {}).get("question_answers", [])
            for cc in result.get("classified_questions", []):
                idx = cc.get("index", -1)
                if 0 <= idx < len(chunk_items):
                    item = chunk_items[idx]
                    all_classified.append({
                        "category": cc.get("category", "Other"),
                        "topic": cc.get("topic") or cc["theme"],
                        "theme": cc["theme"],
                        "question": (item.get("question") or "")[:300],
                        "answer_snippet": (item.get("answer") or "")[:200],
                        "thread_id": item.get("thread_id"),
                        "confidence": "uncertain" if item.get("is_uncertain") else "certain",
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

        # Attach the classified questions so downstream consumers
        # (e.g. the HTML report) can show full question detail per theme.
        synthesis["classified_questions"] = all_classified
        return synthesis

    @staticmethod
    def normalize_topics(raw_topics: list[str]) -> dict[str, str]:
        """Map raw per-question topic labels to normalized canonical names.

        Uses fast string normalization — no LLM call. Merges labels that
        are identical after lowercasing, stripping, and collapsing whitespace.
        Picks the most common casing as canonical.
        """
        if not raw_topics:
            return {}

        import re

        def _normalize_key(s: str) -> str:
            s = s.strip().lower()
            s = re.sub(r"\s+", " ", s)
            return s

        # Group raw labels by normalized key, track frequency of each casing
        groups: dict[str, dict[str, int]] = {}
        for raw in raw_topics:
            key = _normalize_key(raw)
            if key not in groups:
                groups[key] = {}
            groups[key][raw] = groups[key].get(raw, 0) + 1

        # For each group, pick the most frequent casing as canonical
        canonical_map: dict[str, str] = {}
        for key, variants in groups.items():
            canonical = max(variants, key=lambda v: variants[v])
            canonical_map[key] = canonical

        # Build the final mapping: raw → canonical
        return {raw: canonical_map[_normalize_key(raw)] for raw in dict.fromkeys(raw_topics)}

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

    def fact_check_recommendations(
        self,
        recommendations: list[dict[str, Any]],
        site_pages: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Validate recommendations against live docs content.

        Args:
            recommendations: sprint_recommendations from the final synthesis.
            site_pages: mapping of URL → page text fetched from the live site.

        Returns:
            The same recommendations list with added ``fact_check_status``
            and ``fact_check_note`` fields.
        """
        if not recommendations or not site_pages:
            return recommendations

        schema = {
            "type": "object",
            "properties": {
                "checked": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {"type": "integer"},
                            "status": {
                                "type": "string",
                                "enum": [
                                    "confirmed",
                                    "already_addressed",
                                    "partially_addressed",
                                ],
                            },
                            "note": {"type": "string"},
                        },
                        "required": ["index", "status", "note"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["checked"],
            "additionalProperties": False,
        }

        system = (
            "You fact-check documentation improvement recommendations against "
            "live documentation pages.\n\n"
            "For each recommendation (identified by index), determine whether:\n"
            "- confirmed: the issue still exists and the recommendation is relevant.\n"
            "- already_addressed: the docs already cover this topic adequately.\n"
            "- partially_addressed: some content exists but the recommendation "
            "  is still partially relevant.\n\n"
            "Provide a concise note (1 sentence) explaining your verdict, "
            "citing a specific page URL when possible."
        )

        result = self._structured_json(
            system_prompt=system,
            payload={
                "recommendations": [
                    {"index": i, "title": r["title"], "scope": r.get("scope", "")}
                    for i, r in enumerate(recommendations)
                ],
                "site_pages": site_pages,
            },
            schema=schema,
            max_tokens=2000,
        )

        status_map: dict[int, dict[str, str]] = {}
        for item in result.get("checked", []):
            status_map[item["index"]] = {
                "status": item["status"],
                "note": item["note"],
            }

        for i, rec in enumerate(recommendations):
            check = status_map.get(i, {"status": "confirmed", "note": ""})
            rec["fact_check_status"] = check["status"]
            rec["fact_check_note"] = check["note"]

        return recommendations
