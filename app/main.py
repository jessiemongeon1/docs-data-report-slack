from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path

from app.claude_pipeline import analyze_site_raw_data
from app.config import AppConfig, SiteConfig, load_config
from app.kapa import KapaClient
from app.plausible import PlausibleClient
from app.reporting import render_site_report, render_index_page
from app.slack_notifier import post_summary_to_slack


def compute_default_window() -> tuple[str, str]:
    today = date.today()
    end = today - timedelta(days=1)
    start = end - timedelta(days=6)
    return start.isoformat(), end.isoformat()


def slugify_site_name(name: str) -> str:
    return name.replace(".", "-").replace("/", "-").lower()


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def process_site(
    app_config: AppConfig,
    site: SiteConfig,
    run_id: str,
    start_date: str,
    end_date: str,
) -> dict:
    plausible = PlausibleClient(
        api_key=app_config.plausible_api_key,
        site_id=site.plausible_site_id,
    )

    kapa = KapaClient(
        api_key=os.environ[site.kapa_api_key_env],
        project_id=site.kapa_project_id,
    )

    plausible_raw = plausible.fetch_weekly_bundle(start_date, end_date)
    kapa_raw = kapa.fetch_weekly_bundle(start_date, end_date)

    raw_dir = Path("runs") / run_id / "raw" / slugify_site_name(site.name)
    save_json(raw_dir / "plausible.json", plausible_raw)
    save_json(raw_dir / "kapa.json", kapa_raw)

    claude_output = analyze_site_raw_data(
        anthropic_api_key=app_config.anthropic_api_key,
        site_name=site.name,
        start_date=start_date,
        end_date=end_date,
        plausible_raw=plausible_raw,
        kapa_raw=kapa_raw,
    )

    analysis_dir = Path("runs") / run_id / "analysis"
    save_json(analysis_dir / f"{slugify_site_name(site.name)}.json", claude_output)

    report_url = (
        f"{app_config.report_base_url}/{run_id}/{slugify_site_name(site.name)}.html"
    )

    report_context = {
        "site_name": site.name,
        "start_date": start_date,
        "end_date": end_date,
        "plausible_raw": plausible_raw,
        "kapa_raw": kapa_raw,
        "analysis": claude_output,
        "report_url": report_url,
    }

    output_path = Path("site") / run_id / f"{slugify_site_name(site.name)}.html"
    render_site_report(report_context, output_path)

    return {
        "site_name": site.name,
        "report_url": report_url,
        "analysis": claude_output,
    }


def main() -> None:
    config = load_config()
    start_date, end_date = compute_default_window()
    run_id = end_date

    site_summaries = []
    for site in config.sites:
        site_summaries.append(
            process_site(
                app_config=config,
                site=site,
                run_id=run_id,
                start_date=start_date,
                end_date=end_date,
            )
        )

    render_index_page(
        {
            "run_id": run_id,
            "site_summaries": site_summaries,
            "start_date": start_date,
            "end_date": end_date,
        },
        Path("site") / run_id / "index.html",
    )

    post_summary_to_slack(
        webhook_url=config.slack_webhook_url,
        run_id=run_id,
        start_date=start_date,
        end_date=end_date,
        site_summaries=site_summaries,
    )


if __name__ == "__main__":
    main()
