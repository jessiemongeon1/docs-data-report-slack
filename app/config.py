from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    if raw == "":
        return default
    return int(raw)


def _get_env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    if raw == "":
        return default
    return raw


@dataclass(frozen=True)
class SiteConfig:
    name: str
    plausible_site_id: str
    kapa_project_id: str
    kapa_api_key_env: str
    slack_webhook_urls: tuple[str, ...]


@dataclass(frozen=True)
class Settings:
    plausible_api_key: str
    anthropic_api_key: str
    slack_webhook_url: str
    sites: list[SiteConfig]
    raw_output_dir: Path
    report_output_dir: Path
    site_output_dir: Path
    claude_model: str
    claude_max_input_tokens: int
    report_days: int

    @staticmethod
    def from_env() -> "Settings":
        global_webhook = os.environ.get("SLACK_WEBHOOK_URL", "")

        sites_json = os.environ["SITES_JSON"]
        parsed_sites = json.loads(sites_json)
        sites = []
        for item in parsed_sites:
            # Accept either a single URL (slack_webhook_url) or a list
            # (slack_webhook_urls). Falls back to the global webhook if neither set.
            urls_value = item.get("slack_webhook_urls")
            if urls_value is None:
                single = item.get("slack_webhook_url") or global_webhook
                webhook_urls = (single,)
            elif isinstance(urls_value, str):
                webhook_urls = (urls_value,)
            else:
                webhook_urls = tuple(urls_value)
            sites.append(
                SiteConfig(
                    name=item["name"],
                    plausible_site_id=item["plausible_site_id"],
                    kapa_project_id=item["kapa_project_id"],
                    kapa_api_key_env=item["kapa_api_key_env"],
                    slack_webhook_urls=webhook_urls,
                )
            )
        return Settings(
            plausible_api_key=os.environ["PLAUSIBLE_API_KEY"],
            anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
            slack_webhook_url=global_webhook,
            sites=sites,
            raw_output_dir=Path(_get_env_str("RAW_OUTPUT_DIR", "./artifacts/raw")),
            report_output_dir=Path(_get_env_str("REPORT_OUTPUT_DIR", "./artifacts/reports")),
            site_output_dir=Path(_get_env_str("SITE_OUTPUT_DIR", "./reports")),
            claude_model=_get_env_str("CLAUDE_MODEL", "claude-sonnet-4-6"),
            claude_max_input_tokens=_get_env_int("CLAUDE_MAX_INPUT_TOKENS", 120000),
            report_days=_get_env_int("REPORT_DAYS", 7),
        )
