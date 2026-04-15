from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List


@dataclass
class SiteConfig:
    name: str
    plausible_site_id: str
    kapa_project_id: str
    kapa_api_key_env: str


@dataclass
class AppConfig:
    plausible_api_key: str
    anthropic_api_key: str
    slack_webhook_url: str
    report_base_url: str
    sites: List[SiteConfig]


def load_sites() -> List[SiteConfig]:
    raw = os.environ["SITES_JSON"]
    items = json.loads(raw)

    sites: List[SiteConfig] = []
    for item in items:
        sites.append(
            SiteConfig(
                name=item["name"],
                plausible_site_id=item["plausible_site_id"],
                kapa_project_id=item["kapa_project_id"],
                kapa_api_key_env=item["kapa_api_key_env"],
            )
        )

    return sites


def load_config() -> AppConfig:
    return AppConfig(
        plausible_api_key=os.environ["PLAUSIBLE_API_KEY"],
        anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
        slack_webhook_url=os.environ["SLACK_WEBHOOK_URL"],
        report_base_url=os.environ["REPORT_BASE_URL"].rstrip("/"),
        sites=load_sites(),
    )
