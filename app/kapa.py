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

    def fetch_weekly_bundle(self, start_date: str, end_date: str) -> dict[str, Any]:
        project_id = self.project_id
        params = {
            "project_id": project_id,
            "start_date": start_date,
            "end_date": end_date,
        }

        return {
            "project_id": project_id,
            "conversations": self.get("/api/v1/conversations", params=params),
            "question_answers": self.get("/api/v1/question-answers", params=params),
            "usage": self.get("/api/v1/usage", params=params),
        }
