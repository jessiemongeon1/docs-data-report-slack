from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import requests


class PlausibleClient:
    base_url = "https://plausible.io/api/v2/query"

    def __init__(self, api_key: str, site_id: str) -> None:
        self.api_key = api_key
        self.site_id = site_id

    def query(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = requests.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"site_id": self.site_id, **payload},
            timeout=90,
        )

        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            raise requests.HTTPError(
                f"Plausible API error {response.status_code}: {response.text}"
            ) from e

        return response.json()

    @staticmethod
    def _parse_iso_date(value: str) -> date:
        return datetime.strptime(value, "%Y-%m-%d").date()

    def fetch_weekly_bundle(self, start_date: str, end_date: str) -> dict[str, Any]:
        start = self._parse_iso_date(start_date)
        end = self._parse_iso_date(end_date)

        start_14d = end - timedelta(days=13)
        start_30d = end - timedelta(days=29)

        summary_metrics = [
            "visitors",
            "pageviews",
            "bounce_rate",
            "views_per_visit",
            "visit_duration",
        ]

        return {
            "summary_7d": self.query(
                {
                    "date_range": [start.isoformat(), end.isoformat()],
                    "metrics": summary_metrics,
                }
            ),
            "summary_14d": self.query(
                {
                    "date_range": [start_14d.isoformat(), end.isoformat()],
                    "metrics": summary_metrics,
                }
            ),
            "summary_30d": self.query(
                {
                    "date_range": [start_30d.isoformat(), end.isoformat()],
                    "metrics": summary_metrics,
                }
            ),
            "top_pages": self.query(
                {
                    "date_range": [start.isoformat(), end.isoformat()],
                    "dimensions": ["event:page"],
                    "metrics": [
                        "visitors",
                        "pageviews",
                        "time_on_page",
                        "scroll_depth",
                    ],
                    "order_by": [["pageviews", "desc"]],
                    "pagination": {"limit": 50, "offset": 0},
                }
            ),
            "referrals": self.query(
                {
                    "date_range": [start.isoformat(), end.isoformat()],
                    "dimensions": ["visit:source"],
                    "metrics": ["visitors", "bounce_rate", "visit_duration"],
                    "order_by": [["visitors", "desc"]],
                    "pagination": {"limit": 25, "offset": 0},
                }
            ),
            "timeseries": self.query(
                {
                    "date_range": [start.isoformat(), end.isoformat()],
                    "dimensions": ["time:day"],
                    "metrics": ["visitors", "pageviews", "bounce_rate"],
                    "order_by": [["time:day", "asc"]],
                    "pagination": {"limit": 100, "offset": 0},
                }
            ),
        }
