from __future__ import annotations

from typing import Any

import requests


class KapaClient:
    base_url = "https://api.kapa.ai"

    def __init__(self, api_key: str, project_id: str) -> None:
        self.api_key = api_key
        self.project_id = project_id

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        response = requests.get(
            f"{self.base_url}{path}",
            headers={
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json",
            },
            params=params,
            timeout=90,
        )

        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            raise requests.HTTPError(
                f"Kapa API error {response.status_code}: {response.text}"
            ) from e

        return response.json()

    def _extract_thread_ids(self, question_answers_payload: Any) -> list[str]:
        """
        Best-effort extraction.
        The public docs clearly show a list question-answers endpoint and a retrieve-thread endpoint,
        but they do not clearly document the exact list response shape in the snippets we have.
        So this method defensively checks several likely key names.
        """
        thread_ids: set[str] = set()

        def walk(obj: Any) -> None:
            if isinstance(obj, dict):
                # likely shapes
                for key in ("thread_id", "thread", "threadId"):
                    value = obj.get(key)
                    if isinstance(value, str) and value:
                        thread_ids.add(value)

                for value in obj.values():
                    walk(value)

            elif isinstance(obj, list):
                for item in obj:
                    walk(item)

        walk(question_answers_payload)
        return sorted(thread_ids)

    def fetch_weekly_bundle(self, start_date: str, end_date: str) -> dict[str, Any]:
        # Documented public endpoint:
        # GET /query/v1/projects/:project_id/question-answers/
        # The docs page confirms the route, but the snippet does not fully show all query params.
        # So we pass the common window params and keep the raw response.
        question_answers = self.get(
            f"/query/v1/projects/{self.project_id}/question-answers/",
            params={
                "start_date": start_date,
                "end_date": end_date,
            },
        )

        # Documented public endpoint:
        # GET /query/v1/projects/:project_id/end-users/
        # This can be useful for additional audience context.
        try:
            end_users = self.get(
                f"/query/v1/projects/{self.project_id}/end-users/",
                params={},
            )
        except requests.HTTPError as e:
            # Non-fatal; keep pipeline running if this endpoint is unavailable for your project
            end_users = {
                "error": str(e),
            }

        # Documented public endpoint:
        # GET /query/v1/threads/:id/
        # Only fetch these if we can discover thread ids in the question_answers payload.
        thread_ids = self._extract_thread_ids(question_answers)
        threads: dict[str, Any] = {}

        for thread_id in thread_ids[:50]:
            # cap to avoid runaway calls in the first pass
            try:
                threads[thread_id] = self.get(f"/query/v1/threads/{thread_id}/")
            except requests.HTTPError as e:
                threads[thread_id] = {"error": str(e)}

        return {
            "project_id": self.project_id,
            "question_answers": question_answers,
            "end_users": end_users,
            "thread_ids_discovered": thread_ids,
            "threads": threads,
        }
