from __future__ import annotations

from typing import Any

import requests


class KapaClient:
    base_url = "https://api.kapa.ai"

    def __init__(self, api_key: str, project_id: str) -> None:
        self.api_key = api_key
        self.project_id = project_id

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        merged_params = {"project_id": self.project_id}
        if params:
            merged_params.update(params)

        response = requests.get(
            f"{self.base_url}{path}",
            headers={"X-API-KEY": self.api_key},
            params=merged_params,
            timeout=90,
        )
        response.raise_for_status()
        return response.json()

    def fetch_weekly_bundle(self, start_date: str, end_date: str) -> dict[str, Any]:
        # Adjust these endpoints to match the exact Kapa API objects enabled for your project.
        return {
            "conversations": self.get(
                "/api/v1/conversations",
                {"start_date": start_date, "end_date": end_date, "limit": 200},
            ),
            "usage": self.get(
                "/api/v1/analytics/usage",
                {"start_date": start_date, "end_date": end_date},
            ),
            "feedback": self.get(
                "/api/v1/analytics/feedback",
                {"start_date": start_date, "end_date": end_date},
            ),
        }
