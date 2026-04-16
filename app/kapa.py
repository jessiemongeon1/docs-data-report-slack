from __future__ import annotations

import time
from typing import Any
import requests


class KapaClient:
    base_url = "https://api.kapa.ai"
    max_attempts = 5
    initial_backoff_seconds = 2.0

    def __init__(self, api_key: str, project_id: str) -> None:
        self.api_key = api_key
        self.project_id = project_id

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = requests.get(
                    f"{self.base_url}{path}",
                    headers={
                        "X-API-KEY": self.api_key,
                        "Content-Type": "application/json",
                    },
                    params=params,
                    timeout=90,
                )
            except (requests.ConnectionError, requests.Timeout) as e:
                last_exc = e
                if attempt == self.max_attempts:
                    raise
                backoff = self.initial_backoff_seconds * (2 ** (attempt - 1))
                print(
                    f"Kapa request failed ({type(e).__name__}); "
                    f"retrying in {backoff:.0f}s (attempt {attempt}/{self.max_attempts})"
                )
                time.sleep(backoff)
                continue

            if response.status_code >= 500 and attempt < self.max_attempts:
                backoff = self.initial_backoff_seconds * (2 ** (attempt - 1))
                print(
                    f"Kapa returned {response.status_code}; "
                    f"retrying in {backoff:.0f}s (attempt {attempt}/{self.max_attempts})"
                )
                time.sleep(backoff)
                continue

            try:
                response.raise_for_status()
            except requests.HTTPError as e:
                raise requests.HTTPError(
                    f"Kapa API error {response.status_code}: {response.text}"
                ) from e

            return response.json()

        raise RuntimeError("Kapa query exhausted retries") from last_exc

    def _extract_qa_items(self, payload: Any) -> list[dict[str, Any]]:
        """
        Normalize Kapa response into clean QA list.
        Keeps ONLY useful fields for analysis.
        """

        if isinstance(payload, dict):
            if "results" in payload and isinstance(payload["results"], list):
                items = payload["results"]
            elif "data" in payload and isinstance(payload["data"], list):
                items = payload["data"]
            else:
                items = [payload]
        elif isinstance(payload, list):
            items = payload
        else:
            items = [payload]

        cleaned: list[dict[str, Any]] = []

        for item in items:
            if not isinstance(item, dict):
                continue

            question = (
                item.get("question")
                or item.get("query")
                or item.get("user_message")
                or ""
            )

            answer = (
                item.get("answer")
                or item.get("response")
                or item.get("assistant_message")
                or ""
            )

            # skip empty garbage rows
            if not question and not answer:
                continue

            cleaned.append(
                {
                    "question": question,
                    "answer": answer,
                    "timestamp": item.get("created_at") or item.get("timestamp"),
                    "thread_id": item.get("thread_id"),
                }
            )

        return cleaned

    def fetch_weekly_bundle(self, start_date: str, end_date: str) -> dict[str, Any]:
        path = f"/query/v1/projects/{self.project_id}/question-answers/"
        page_size = 100
        page = 1
        max_pages = 100  # safety cap: 10,000 items
        all_items: list[dict[str, Any]] = []
        # Track the raw payload from the first page so we can keep top-level
        # metadata (e.g. count) for downstream debugging.
        first_payload: dict[str, Any] | None = None

        while page <= max_pages:
            raw = self.get(
                path,
                params={
                    "start_date": start_date,
                    "end_date": end_date,
                    "page": page,
                    "page_size": page_size,
                },
            )
            if first_payload is None:
                first_payload = raw

            page_items = self._extract_qa_items(raw)
            if not page_items:
                # Empty page means we've consumed all available results.
                break
            all_items.extend(page_items)

            # Stop if the API signals there are no more pages, or if the page
            # came back smaller than requested (last page).
            if isinstance(raw, dict):
                if raw.get("next") in (None, "", False):
                    # Some Kapa endpoints expose a `next` cursor/url; if it's
                    # explicitly null treat it as the final page. If the field
                    # is missing entirely, fall through to the size check.
                    if "next" in raw:
                        break
            if len(page_items) < page_size:
                break

            page += 1

        print(f"Kapa fetched {len(all_items)} QA items across {page} page(s)")

        return {
            "project_id": self.project_id,
            "question_answers": all_items,
            "count": len(all_items),
        }
