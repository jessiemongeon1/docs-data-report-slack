from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from collections import Counter

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


def build_report_url(repo: str, run_id: str, report_filename: str) -> str:
    # Use GitHub Pages URL if GITHUB_PAGES_URL is set or derivable from the repo name.
    pages_base = os.environ.get("GITHUB_PAGES_URL", "").rstrip("/")
    if not pages_base:
        # Default convention: https://<owner>.github.io/<repo>/
        parts = repo.split("/", 1)
        if len(parts) == 2:
            owner, repo_name = parts
            pages_base = f"https://{owner}.github.io/{repo_name}"
    return f"{pages_base}/{run_id}/{report_filename}"


def compute_kapa_user_stats(kapa_raw: dict[str, Any]) -> dict[str, Any]:
    """Compute returning-user and conversation stats from Kapa QA items."""
    qa_items = kapa_raw.get("question_answers", [])

    # Map user_id → set of conversation_ids
    user_conversations: dict[str, set[str]] = {}
    # Also count conversations with no user_id
    anonymous_conversations: set[str] = set()

    for item in qa_items:
        user_id = item.get("user_id")
        conv_id = item.get("conversation_id") or item.get("thread_id") or "unknown"

        if user_id:
            user_conversations.setdefault(str(user_id), set()).add(str(conv_id))
        else:
            anonymous_conversations.add(str(conv_id))

    total_users = len(user_conversations)
    returning_users = {
        uid: convs for uid, convs in user_conversations.items() if len(convs) > 1
    }
    total_conversations = len(
        set().union(*user_conversations.values()) | anonymous_conversations
        if user_conversations
        else anonymous_conversations
    )

    # Build a sorted list of returning users by conversation count (desc)
    returning_user_list = sorted(
        [
            {"user_id": uid, "conversations": len(convs)}
            for uid, convs in returning_users.items()
        ],
        key=lambda x: x["conversations"],
        reverse=True,
    )

    # Conversation-count distribution for all identified users
    convo_counts = Counter(len(convs) for convs in user_conversations.values())
    distribution = [
        {"conversations": k, "users": v}
        for k, v in sorted(convo_counts.items())
    ]

    return {
        "total_identified_users": total_users,
        "total_conversations": total_conversations,
        "anonymous_conversations": len(anonymous_conversations),
        "returning_user_count": len(returning_users),
        "returning_users": returning_user_list[:20],  # top 20 by volume
        "conversation_distribution": distribution,
    }


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
    kapa_user_stats = compute_kapa_user_stats(kapa_raw)

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

    report_url = build_report_url(repo, run_id, report_filename)

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
            "kapa_user_stats": kapa_user_stats,
            "plausible_analysis": plausible_analysis,
            "kapa_analysis": kapa_analysis,
            "final_analysis": final_analysis,
            "report_url": report_url,
        },
    )
    report_path.write_text(html, encoding="utf-8")

    # Surface referrer URLs to the Slack template (sorted by visitors desc
    # by Plausible). Direct/no-referrer entries are filtered out. If
    # REFERRER_ALLOWLIST is set, only referrers whose hostname matches one of
    # the listed domains (or any of their subdomains) are included.
    top_referrers_raw = (
        plausible_raw.get("top_referrers", {}).get("results", [])
        if isinstance(plausible_raw, dict)
        else []
    )

    allowlist_raw = os.getenv("REFERRER_ALLOWLIST", "").strip()
    allowlist = [
        d.strip().lower().lstrip(".")
        for d in allowlist_raw.split(",")
        if d.strip()
    ]

    denylist_raw = os.getenv("REFERRER_DENYLIST", "").strip()
    denylist = [
        d.strip().lower().lstrip(".")
        for d in denylist_raw.split(",")
        if d.strip()
    ]

    def _hostname(referrer: str) -> str:
        # Strip scheme and path; keep just the host.
        host = referrer.lower()
        if "://" in host:
            host = host.split("://", 1)[1]
        host = host.split("/", 1)[0]
        return host

    def _matches_allowlist(host: str) -> bool:
        if not allowlist:
            return True
        # Match the exact domain or any subdomain of an allowed domain.
        return any(host == d or host.endswith("." + d) for d in allowlist)

    def _matches_denylist(host: str) -> bool:
        # Denylist is exact-match only so a deny entry like "docs.sui.io"
        # filters that subdomain without removing the rest of "sui.io".
        return host in denylist

    top_referrers: list[dict[str, Any]] = []
    for row in top_referrers_raw:
        dimensions = row.get("dimensions") or []
        metrics = row.get("metrics") or []
        if not dimensions or not metrics:
            continue
        referrer = dimensions[0]
        if not referrer or referrer == "Direct / None":
            continue
        host = _hostname(referrer)
        if _matches_denylist(host):
            continue
        if not _matches_allowlist(host):
            continue
        top_referrers.append({
            "referrer": referrer,
            "visitors": metrics[0] if len(metrics) > 0 else 0,
            "pageviews": metrics[1] if len(metrics) > 1 else 0,
        })

    return {
        "site_name": site.name,
        "site_slug": site_slug,
        "report_filename": report_filename,
        "report_url": report_url,
        "slack_webhook_urls": list(site.slack_webhook_urls),
        "final_analysis": final_analysis,
        "top_referrers": top_referrers,
        "kapa_user_stats": kapa_user_stats,
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

        for webhook_url in site_report["slack_webhook_urls"]:
            notifier = SlackNotifier(webhook_url)
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
