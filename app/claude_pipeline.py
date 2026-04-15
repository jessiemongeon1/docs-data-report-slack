from __future__ import annotations

import json
import re
import time
from typing import Any

from anthropic import Anthropic, RateLimitError

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
    def __init__(self, api_key: str, model: str, max_input_tokens: int = 12000) -> None:
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

    def _count_tokens(self, system_prompt: str, payload: dict[str, Any]) -> int:
        user_content = json.dumps(payload, ensure_ascii=False)
        resp = self.client.messages.count_tokens(
            model=self.model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )
        return resp.input_tokens

    def _repair_json(self, system_prompt: str, bad_text: str) -> dict[str, Any]:
        repair_prompt = f"""
Your previous response was not valid JSON.

Return the same information again as valid JSON only.
Do not use markdown fences.
Do not add explanation.

Previous invalid output:
{bad_text[:12000]}
""".strip()

        response = self._messages_create_with_retry(
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

        response = self._messages_create_with_retry(
            model=self.model,
            max_tokens=2500,
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

    def _normalize_thread_items(self, threads: Any) -> list[Any]:
        if isinstance(threads, dict):
            if all(isinstance(v, dict) for v in threads.values()):
                return [{"thread_id": k, **v} for k, v in threads.items()]
            return [threads]
        if isinstance(threads, list):
            return threads
        return [threads]

    def _chunk_items_by_token_count(
        self,
        items: list[Any],
        system_prompt: str,
        base_payload: dict[str, Any],
        field_name: str,
        target_tokens: int,
    ) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        current_items: list[Any] = []

        for item in items:
            tentative = {**base_payload, "raw": {field_name: current_items + [item]}}
            tokens = self._count_tokens(system_prompt, tentative)

            if current_items and tokens > target_tokens:
                chunks.append({**base_payload, "raw": {field_name: current_items}})
                current_items = [item]
            else:
                current_items.append(item)

        if current_items:
            chunks.append({**base_payload, "raw": {field_name: current_items}})

        return chunks

    def _build_kapa_chunks(self, kapa_raw: dict[str, Any]) -> list[dict[str, Any]]:
        qa_items = self._normalize_qa_items(kapa_raw.get("question_answers", []))
        thread_items = self._normalize_thread_items(kapa_raw.get("threads", {}))
        end_users = kapa_raw.get("end_users", {})
        project_id = kapa_raw.get("project_id")

        # Stay well under the org's 30k ITPM cap.
        target_tokens = min(self.max_input_tokens, 10000)

        chunks: list[dict[str, Any]] = []

        qa_base = {
            "source": "kapa",
            "chunk_type": "question_answers",
            "project_id": project_id,
        }
        qa_chunks = self._chunk_items_by_token_count(
            qa_items,
            KAPA_CHUNK_SYSTEM,
            qa_base,
            "question_answers",
            target_tokens,
        )
        for i, chunk in enumerate(qa_chunks, start=1):
            chunk["chunk_index"] = i
            chunk["chunk_count"] = len(qa_chunks)
            chunks.append(chunk)

        if thread_items and thread_items != [{}]:
            thread_base = {
                "source": "kapa",
                "chunk_type": "threads",
                "project_id": project_id,
            }
            thread_chunks = self._chunk_items_by_token_count(
                thread_items,
                KAPA_CHUNK_SYSTEM,
                thread_base,
                "threads",
                target_tokens,
            )
            for i, chunk in enumerate(thread_chunks, start=1):
                chunk["chunk_index"] = i
                chunk["chunk_count"] = len(thread_chunks)
                chunks.append(chunk)

        meta_chunk = {
            "source": "kapa",
            "chunk_type": "metadata",
            "chunk_index": 1,
            "chunk_count": 1,
            "project_id": project_id,
            "raw": {
                "project_id": project_id,
                "thread_ids_discovered": kapa_raw.get("thread_ids_discovered", []),
                "end_users": end_users,
            },
        }
        if self._count_tokens(KAPA_CHUNK_SYSTEM, meta_chunk) <= target_tokens:
            chunks.append(meta_chunk)

        return chunks

    def analyze_kapa_raw(self, kapa_raw: dict[str, Any]) -> dict[str, Any]:
        initial_payload = {"source": "kapa", "raw": kapa_raw}
        initial_tokens = self._count_tokens(KAPA_SYNTHESIS_SYSTEM, initial_payload)

        if initial_tokens <= self.max_input_tokens:
            return self.message_json(KAPA_SYNTHESIS_SYSTEM, initial_payload)

        chunk_payloads = self._build_kapa_chunks(kapa_raw)
        print(f"Kapa input tokens: {initial_tokens}")
        print(f"Kapa chunk count: {len(chunk_payloads)}")

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
