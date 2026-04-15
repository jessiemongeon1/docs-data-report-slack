from __future__ import annotations

import json
from pathlib import Path

from app.claude_pipeline import ClaudePipeline
from app.config import Settings
from app.kapa import KapaClient
from app.plausible import PlausibleClient
from app.render_report import render_html
from app.slack_notifier import SlackNotifier
from app.utils import dump_json, ensure_dir, report_window, utc_now


def build_report_url(base_url: str, run_id: str, filename: str) -> str:
    if not base_url:
        return filename
    return f"{base_url}/{run_id}/{filename}"


def run() -> None:
    settings = Settings.from_env()
    start_date, end_date = report_window(settings.report_days)
    run_id = utc_now().strftime("%Y%m%dT%H%M%SZ")

    raw_dir = settings.raw_output_dir / run_id
    report_dir = settings.report_output_dir / run_id
    site_dir = settings.site_output_dir / run_id
    ensure_dir(raw_dir)
    ensure_dir(report_dir)
    ensure_dir(site_dir)

    claude = ClaudePipeline(
        settings.anthropic_api_key,
        settings.claude_model,
        settings.claude_max_input_tokens,
    )

    site_reports: list[dict] = []
    for site in settings.sites:
        plausible = PlausibleClient(settings.plausible_api_key, site.plausible_site_id)
        kapa = KapaClient(settings.kapa_api_key, site.kapa_project_id)

        plausible_raw = plausible.fetch_weekly_bundle(start_date, end_date)
        kapa_raw = kapa.fetch_weekly_bundle(start_date, end_date)

        site_slug = site.name.replace('.', '-').replace('/', '-')
        dump_json(raw_dir / f"{site_slug}_plausible_raw.json", plausible_raw)
        dump_json(raw_dir / f"{site_slug}_kapa_raw.json", kapa_raw)

        plausible_analysis = claude.analyze_plausible_raw(plausible_raw)
        kapa_analysis = claude.analyze_kapa_raw(kapa_raw)

        dump_json(report_dir / f"{site_slug}_plausible_analysis.json", plausible_analysis)
        dump_json(report_dir / f"{site_slug}_kapa_analysis.json", kapa_analysis)

        final_analysis = claude.synthesize(
            {
                "site_name": site.name,
                "run_id": run_id,
                "start_date": start_date,
                "end_date": end_date,
            },
            plausible_analysis,
            kapa_analysis,
        )
        dump_json(report_dir / f"{site_slug}_final_analysis.json", final_analysis)

        report_filename = f"{site_slug}.html"
        html = render_html(
            Path("templates"),
            "weekly_report.html.j2",
            {
                "site_name": site.name,
                "run_id": run_id,
                "start_date": start_date,
                "end_date": end_date,
                "plausible_raw": plausible_raw,
                "kapa_raw": kapa_raw,
                "plausible_analysis": plausible_analysis,
                "kapa_analysis": kapa_analysis,
                "final_analysis": final_analysis,
            },
        )
        report_path = site_dir / report_filename
        report_path.write_text(html, encoding="utf-8")

        site_reports.append(
            {
                "site_name": site.name,
                "report_filename": report_filename,
                "report_url": build_report_url(settings.report_base_url, run_id, report_filename),
                "final_analysis": final_analysis,
            }
        )

    index_html = render_html(
        Path("templates"),
        "index.html.j2",
        {
            "run_id": run_id,
            "start_date": start_date,
            "end_date": end_date,
            "site_reports": site_reports,
        },
    )
    (site_dir / "index.html").write_text(index_html, encoding="utf-8")

    slack_blocks_raw = render_html(
        Path("templates"),
        "slack_summary.json.j2",
        {
            "start_date": start_date,
            "end_date": end_date,
            "site_reports": site_reports,
        },
    )
    slack_blocks = json.loads(slack_blocks_raw)

    notifier = SlackNotifier(settings.slack_webhook_url)
    notifier.send(
        text=f"Weekly docs analytics summary ({start_date} to {end_date})",
        blocks=slack_blocks,
    )

    print(f"Saved reports under {site_dir}")


if __name__ == "__main__":
    run()
