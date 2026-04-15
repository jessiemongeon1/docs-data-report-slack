from __future__ import annotations

import json
import math
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

KAPA_CHUNK_SYSTEM = """
You analyze one raw chunk of Kapa JSON.

This chunk is only part of the full dataset.
Do not assume it represents the whole reporting period.

Return valid JSON only with this shape:
{
  "chunk_summary": "string",
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
  ],
  "open_questions": ["string"]
}
Do not include markdown fences.
Do not include commentary outside JSON.
""".strip()

KAPA_SYNTHESIS_SYSTEM = """
You synthesize multiple chunk-level analyses of raw Kapa JSON into one weekly Kapa analysis.

Each chunk analysis came from a separate subset of the raw Kapa dataset.
Merge repeated themes across chunks.
Prefer recurring issues over one-off issues.

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

        if text.startswith("{") or text.startswith("["):
            return text

        obj_start = text.find("{")
        arr_start = text.find("[")
        starts = [i for i in [obj_start, arr_start] if i != -1]
        if not starts:
            raise ValueError(f"Claude returned no JSON:\n{text[:1000]}")

        start = min(starts)
        candidate = text[start:].strip()

        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

        if candidate.startswith("{"):
            depth = 0
            for i, ch in enumerate(candidate):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return candidate[: i + 1]

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

    def _estimate_tokens(self, payload: dict[str, Any]) -> int:
        # Rough estimate: 1 token ~= 4 chars.
        text = json.dumps(payload, ensure_ascii=False)
        return math.ceil(len(text) / 4)

    def _chunk_list(self, items: list[Any], target_tokens: int) -> list[list[Any]]:
        chunks: list[list[Any]] = []
        current: list[Any] = []
        current_size = 0

        for item in items:
            item_text = json.dumps(item, ensure_ascii=False)
            item_tokens = max(1, math.ceil(len(item_text) / 4))

            if current and current_size + item_tokens > target_tokens:
                chunks.append(current)
                current = [item]
                current_size = item_tokens
            else:
                current.append(item)
                current_size += item_tokens

        if current:
            chunks.append(current)

        return chunks

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

    def _build_kapa_chunks(self, kapa_raw: dict[str, Any]) -> list[dict[str, Any]]:
        # Keep raw structure, but split the largest collections.
        question_answers = kapa_raw.get("question_answers", [])
        threads = kapa_raw.get("threads", {})
        end_users = kapa_raw.get("end_users", {})

        # Normalize QA payload to a list if possible.
        qa_items: list[Any]
        if isinstance(question_answers, list):
            qa_items = question_answers
        elif isinstance(question_answers, dict):
            # common API patterns
            if isinstance(question_answers.get("results"), list):
                qa_items = question_answers["results"]
            elif isinstance(question_answers.get("data"), list):
                qa_items = question_answers["data"]
            else:
                qa_items = [question_answers]
        else:
            qa_items = [question_answers]

        thread_items: list[Any]
        if isinstance(threads, dict):
            # thread map -> list of objects
            if all(isinstance(v, dict) for v in threads.values()):
                thread_items = [{"thread_id": k, **v} for k, v in threads.items()]
            else:
                thread_items = [threads]
        elif isinstance(threads, list):
            thread_items = threads
        else:
            thread_items = [threads]

        # Most of the size is usually in question_answers or threads.
        target_tokens = min(120000, max(30000, self.max_input_tokens - 10000))

        qa_chunks = self._chunk_list(qa_items, target_tokens)
        thread_chunks = self._chunk_list(thread_items, target_tokens)

        chunks: list[dict[str, Any]] = []

        if qa_items:
            for i, chunk in enumerate(qa_chunks, start=1):
                chunks.append(
                    {
                        "source": "kapa",
                        "chunk_type": "question_answers",
                        "chunk_index": i,
                        "chunk_count": len(qa_chunks),
                        "project_id": kapa_raw.get("project_id"),
                        "raw": {
                            "question_answers": chunk,
                        },
                    }
                )

        if thread_items and thread_items != [{}]:
            for i, chunk in enumerate(thread_chunks, start=1):
                chunks.append(
                    {
                        "source": "kapa",
                        "chunk_type": "threads",
                        "chunk_index": i,
                        "chunk_count": len(thread_chunks),
                        "project_id": kapa_raw.get("project_id"),
                        "raw": {
                            "threads": chunk,
                        },
                    }
                )

        # Keep lighter metadata together in one extra chunk.
        meta_chunk = {
            "source": "kapa",
            "chunk_type": "metadata",
            "chunk_index": 1,
            "chunk_count": 1,
            "project_id": kapa_raw.get("project_id"),
            "raw": {
                "project_id": kapa_raw.get("project_id"),
                "thread_ids_discovered": kapa_raw.get("thread_ids_discovered", []),
                "end_users": end_users,
            },
        }
        chunks.append(meta_chunk)

        return chunks

    def analyze_kapa_raw(self, kapa_raw: dict[str, Any]) -> dict[str, Any]:
        estimated = self._estimate_tokens({"source": "kapa", "raw": kapa_raw})

        if estimated <= self.max_input_tokens:
            return self.message_json(KAPA_SYNTHESIS_SYSTEM, {"source": "kapa", "raw": kapa_raw})

        chunk_payloads = self._build_kapa_chunks(kapa_raw)
        chunk_analyses: list[dict[str, Any]] = []

        for chunk in chunk_payloads:
            chunk_analysis = self.message_json(KAPA_CHUNK_SYSTEM, chunk)
            chunk_analyses.append(
                {
                    "chunk_type": chunk["chunk_type"],
                    "chunk_index": chunk["chunk_index"],
                    "chunk_count": chunk["chunk_count"],
                    "analysis": chunk_analysis,
                }
            )

        synthesis_payload = {
            "source": "kapa",
            "project_id": kapa_raw.get("project_id"),
            "chunk_analyses": chunk_analyses,
        }
        return self.message_json(KAPA_SYNTHESIS_SYSTEM, synthesis_payload)

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
