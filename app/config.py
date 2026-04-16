from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class SiteConfig:
    name: str
    plausible_site_id: str
    kapa_project_id: str
    kapa_api_key_env: str


@dataclass(frozen=True)
class Settings:
    plausible_api_key: str
    anthropic_api_key: str
    slack_webhook_url: str
    sites: list[SiteConfig]
    report_base_url: str
    raw_output_dir: Path
    report_output_dir: Path
    site_output_dir: Path
    claude_model: str
    claude_max_input_tokens: int
    report_days: int

    @staticmethod
    def from_env() -> "Settings":
        sites_json = os.environ["SITES_JSON"]
        parsed_sites = json.loads(sites_json)

        sites = [
            SiteConfig(
                name=item["name"],
                plausible_site_id=item["plausible_site_id"],
                kapa_project_id=item["kapa_project_id"],
                kapa_api_key_env=item["kapa_api_key_env"],
            )
            for item in parsed_sites
        ]

        report_base_url = os.getenv("REPORT_BASE_URL", "").rstrip("/")

        return Settings(
            plausible_api_key=os.environ["PLAUSIBLE_API_KEY"],
            anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
            slack_webhook_url=os.environ["SLACK_WEBHOOK_URL"],
            sites=sites,
            report_base_url=report_base_url,
            raw_output_dir=Path(os.getenv("RAW_OUTPUT_DIR", "./artifacts/raw")),
            report_output_dir=Path(os.getenv("REPORT_OUTPUT_DIR", "./artifacts/reports")),
            site_output_dir=Path(os.getenv("SITE_OUTPUT_DIR", "./reports"))
            claude_model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"),
            claude_max_input_tokens=int(os.getenv("CLAUDE_MAX_INPUT_TOKENS", "120000")),
            report_days=int(os.getenv("REPORT_DAYS", "7")),
        )
