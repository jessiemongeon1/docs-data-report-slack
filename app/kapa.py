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

        # Log available keys from first item (only once)
        if items and isinstance(items[0], dict) and not hasattr(self, '_logged_keys'):
            print(f"Kapa item keys: {sorted(items[0].keys())}")
            self._logged_keys = True

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
                    "conversation_id": (
                        item.get("conversation_id")
                        or item.get("thread_id")
                        or item.get("session_id")
                    ),
                    "user_id": (
                        item.get("end_user_id")
                        or item.get("user_id")
                        or item.get("user")
                        or item.get("anonymous_user_id")
                        or item.get("user_identifier")
                        or item.get("external_user_id")
                        or item.get("fingerprint")
                    ),
                }
            )

        return cleaned

    def fetch_weekly_bundle(self, start_date: str, end_date: str) -> dict[str, Any]:
        path = f"/query/v1/projects/{self.project_id}/question-answers/"
        page_size = 100
        max_pages = 20  # hard cap: 2,000 items max
        max_items = 2000
        all_items: list[dict[str, Any]] = []
        # Dedup against repeated payloads in case Kapa ignores pagination params
        # (we'd otherwise loop returning the same first page over and over).
        seen_keys: set[str] = set()

        for page in range(1, max_pages + 1):
            raw = self.get(
                path,
                params={
                    "start_date": start_date,
                    "end_date": end_date,
                    "page": page,
                    "page_size": page_size,
                },
            )

            page_items = self._extract_qa_items(raw)
            if not page_items:
                break

            new_items: list[dict[str, Any]] = []
            for item in page_items:
                key = (
                    str(item.get("thread_id"))
                    if item.get("thread_id")
                    else f"{item.get('question', '')[:200]}|{item.get('timestamp')}"
                )
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                new_items.append(item)

            # If everything on this page was already seen, the API isn't
            # paginating - stop instead of looping forever.
            if not new_items:
                print(
                    f"Kapa page {page} returned only duplicate items; "
                    "stopping pagination."
                )
                break

            all_items.extend(new_items)

            if len(all_items) >= max_items:
                print(f"Kapa hit max_items cap of {max_items}; stopping.")
                all_items = all_items[:max_items]
                break

            # Stop if the API explicitly signals no more pages.
            if isinstance(raw, dict) and "next" in raw and not raw.get("next"):
                break
            if len(page_items) < page_size:
                break

        print(f"Kapa fetched {len(all_items)} unique QA items")

        return {
            "project_id": self.project_id,
            "question_answers": all_items,
            "count": len(all_items),
        }
