from __future__ import annotations

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
        response.raise_for_status()
        return response.json()

    def fetch_weekly_bundle(self, start_date: str, end_date: str) -> dict[str, Any]:
        range_payload = {"date_range": [start_date, end_date]}
        return {
            "summary_7d": self.query(
                {
                    **range_payload,
                    "metrics": [
                        "visitors",
                        "pageviews",
                        "bounce_rate",
                        "views_per_visit",
                        "visit_duration",
                    ],
                }
            ),
            "summary_14d": self.query(
                {
                    "date_range": [start_date, end_date],
                    "comparison": {
                        "mode": "previous_period"
                    },
                    "metrics": [
                        "visitors",
                        "pageviews",
                        "bounce_rate",
                        "views_per_visit",
                        "visit_duration",
                    ],
                }
            ),
            "top_pages": self.query(
                {
                    **range_payload,
                    "dimensions": ["event:page"],
                    "metrics": [
                        "visitors",
                        "pageviews",
                        "bounce_rate",
                        "time_on_page",
                        "scroll_depth",
                    ],
                    "order_by": [["pageviews", "desc"]],
                    "limit": 50,
                }
            ),
            "referrals": self.query(
                {
                    **range_payload,
                    "dimensions": ["visit:source"],
                    "metrics": ["visitors", "bounce_rate", "visit_duration"],
                    "order_by": [["visitors", "desc"]],
                    "limit": 25,
                }
            ),
            "timeseries": self.query(
                {
                    **range_payload,
                    "dimensions": ["time:day"],
                    "metrics": ["visitors", "pageviews", "bounce_rate"],
                    "order_by": [["time:day", "asc"]],
                    "limit": 100,
                }
            ),
        }
