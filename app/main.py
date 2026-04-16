from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path

from app.claude_pipeline import ClaudePipeline
from app.config import Settings, SiteConfig
from app.kapa import KapaClient
from app.plausible import PlausibleClient
from app.render_report import render_html
from app.slack_notifier import SlackNotifier
from app.utils import dump_json, ensure_dir, utc_now


def compute_default_window(report_days: int) -> tuple[str, str]:
    today = date.today()
    end = today - timedelta(days=1)
    start = end - timedelta(days=report_days - 1)
    return start.isoformat(), end.isoformat()


def slugify_site_name(name: str) -> str:
    return name.replace(".", "-").replace("/", "-").lower()


def build_report_url(repo: str, branch: str, path: str) -> str:
    return f"https://github.com/{repo}/blob/{branch}/{path}"


def process_site(
    settings: Settings,
    site: SiteConfig,
    run_id: str,
    start_date: str,
    end_date: str,
    claude: ClaudePipeline,
    raw_dir: Path,
    report_dir: Path,
    reports_root: Path,
) -> dict:
    plausible = PlausibleClient(
        settings.plausible_api_key,
        site.plausible_site_id,
    )

    kapa = KapaClient(
        os.environ[site.kapa_api_key_env],
        site.kapa_project_id,
    )

    plausible_raw = plausible.fetch_weekly_bundle(start_date, end_date)
    kapa_raw = kapa.fetch_weekly_bundle(start_date, end_date)

    site_slug = slugify_site_name(site.name)

    dump_json(raw_dir / f"{site_slug}_plausible_raw.json", plausible_raw)
    dump_json(raw_dir / f"{site_slug}_kapa_raw.json", kapa_raw)

    plausible_analysis = claude.analyze_plausible_raw(plausible_raw)
    kapa_analysis = claude.analyze_kapa_raw(kapa_raw)
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

    dump_json(report_dir / f"{site_slug}_plausible_analysis.json", plausible_analysis)
    dump_json(report_dir / f"{site_slug}_kapa_analysis.json", kapa_analysis)
    dump_json(report_dir / f"{site_slug}_final_analysis.json", final_analysis)

    report_filename = f"{site_slug}.html"
    report_path = reports_root / report_filename

    repo = os.environ.get("GITHUB_REPOSITORY", "your-org/your-repo")
    branch = os.environ.get("GITHUB_REF_NAME", "main")

    report_url = build_report_url(
        repo,
        branch,
        f"reports/{run_id}/{report_filename}",
    )

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
            "report_url": report_url,
        },
    )
    report_path.write_text(html, encoding="utf-8")

    return {
        "site_name": site.name,
        "site_slug": site_slug,
        "report_filename": report_filename,
        "report_url": report_url,
        "slack_webhook_url": site.slack_webhook_url,
        "final_analysis": final_analysis,
    }


def write_latest_reports(run_reports_dir: Path, latest_dir: Path) -> None:
    ensure_dir(latest_dir)

    for html_file in run_reports_dir.glob("*.html"):
        target = latest_dir / html_file.name
        target.write_text(html_file.read_text(encoding="utf-8"), encoding="utf-8")


def run() -> None:
    settings = Settings.from_env()
    start_date, end_date = compute_default_window(settings.report_days)
    run_id = utc_now().strftime("%Y%m%dT%H%M%SZ")

    raw_dir = settings.raw_output_dir / run_id
    analysis_dir = settings.report_output_dir / run_id
    reports_dir = settings.site_output_dir / run_id
    latest_dir = settings.site_output_dir / "latest"

    ensure_dir(raw_dir)
    ensure_dir(analysis_dir)
    ensure_dir(reports_dir)
    ensure_dir(latest_dir)

    claude = ClaudePipeline(
        settings.anthropic_api_key,
        settings.claude_model,
        settings.claude_max_input_tokens,
    )

    site_reports: list[dict] = []

    for site in settings.sites:
        site_report = process_site(
            settings=settings,
            site=site,
            run_id=run_id,
            start_date=start_date,
            end_date=end_date,
            claude=claude,
            raw_dir=raw_dir,
            report_dir=analysis_dir,
            reports_root=reports_dir,
        )
        site_reports.append(site_report)

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

    (reports_dir / "index.html").write_text(index_html, encoding="utf-8")
    (settings.site_output_dir / "index.html").write_text(index_html, encoding="utf-8")

    write_latest_reports(reports_dir, latest_dir)
    (latest_dir / "index.html").write_text(index_html, encoding="utf-8")

    # Send a separate Slack notification to each site's webhook.
    for site_report in site_reports:
        slack_blocks_raw = render_html(
            Path("templates"),
            "slack_summary.json.j2",
            {
                "start_date": start_date,
                "end_date": end_date,
                "run_id": run_id,
                # Pass only this site's report so the template renders one site.
                "site_reports": [site_report],
            },
        )
        slack_blocks = json.loads(slack_blocks_raw)

        notifier = SlackNotifier(site_report["slack_webhook_url"])
        notifier.send(
            text=f"Weekly docs analytics: {site_report['site_name']} ({start_date} to {end_date})",
            blocks=slack_blocks,
        )

    print(f"Saved raw data under {raw_dir}")
    print(f"Saved analyses under {analysis_dir}")
    print(f"Saved reports under {reports_dir}")
    print(f"Updated latest reports under {latest_dir}")


if __name__ == "__main__":
    run()
