from __future__ import annotations

import json
import os
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import requests
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


def fetch_site_search_pages(
    site_domain: str,
    recommendations: list[dict[str, Any]],
    max_pages: int = 15,
) -> dict[str, str]:
    """Fetch pages from the live docs site relevant to the recommendations.

    Uses the site's /llms.txt if available (a markdown summary of all pages),
    then falls back to fetching individual pages based on keyword search.
    """
    pages: dict[str, str] = {}

    # Try to fetch llms.txt first — it's a concise summary of all site content
    # and is much cheaper than fetching individual pages.
    llms_url = f"https://{site_domain}/llms.txt"
    try:
        resp = requests.get(llms_url, timeout=15, headers={"User-Agent": "DocsReportBot/1.0"})
        if resp.status_code == 200 and len(resp.text) > 100:
            # Truncate to ~80K chars to stay within token limits
            pages[llms_url] = resp.text[:80000]
            print(f"Fetched {llms_url} ({len(resp.text)} chars)")
            return pages
    except Exception as e:
        print(f"Could not fetch {llms_url}: {e}")

    # Fallback: extract keywords from recommendation titles and fetch
    # likely relevant pages via sitemap or direct URL patterns.
    keywords: list[str] = []
    for rec in recommendations:
        title = rec.get("title", "")
        # Extract meaningful words (3+ chars, lowercase)
        words = re.findall(r"[a-zA-Z]{3,}", title.lower())
        keywords.extend(words)

    # Dedupe and take top terms
    seen: set[str] = set()
    unique_keywords: list[str] = []
    stop_words = {"the", "and", "for", "with", "from", "that", "this", "into", "about"}
    for w in keywords:
        if w not in seen and w not in stop_words:
            seen.add(w)
            unique_keywords.append(w)

    # Try fetching key pages based on common docs site URL patterns
    candidate_paths = set()
    for kw in unique_keywords[:20]:
        candidate_paths.add(f"/{kw}")
    # Also add common top-level pages
    for p in ["/", "/getting-started", "/develop", "/references"]:
        candidate_paths.add(p)

    fetched = 0
    for path in list(candidate_paths):
        if fetched >= max_pages:
            break
        url = f"https://{site_domain}{path}"
        try:
            resp = requests.get(
                url, timeout=10,
                headers={"User-Agent": "DocsReportBot/1.0"},
                allow_redirects=True,
            )
            if resp.status_code == 200 and "text/html" in resp.headers.get("content-type", ""):
                # Strip HTML tags to get text content
                text = re.sub(r"<script[^>]*>.*?</script>", "", resp.text, flags=re.DOTALL)
                text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
                text = re.sub(r"<[^>]+>", " ", text)
                text = re.sub(r"\s+", " ", text).strip()
                if len(text) > 200:
                    pages[url] = text[:5000]
                    fetched += 1
        except Exception:
            pass

    print(f"Fetched {len(pages)} pages from {site_domain} for fact-checking")
    return pages


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
    """Compute user engagement stats from Kapa QA items.

    Each Kapa QA item is one question asked by one user (thread_id is per
    question, not per session). So total_questions == number of QA items.
    We group by end_user_id to find returning users.
    """
    qa_items = kapa_raw.get("question_answers", [])

    # Map user_id → list of question thread_ids
    user_questions: dict[str, list[str]] = {}

    for item in qa_items:
        user_id = item.get("user_id")
        thread_id = item.get("thread_id") or item.get("conversation_id") or "unknown"
        if user_id:
            user_questions.setdefault(str(user_id), []).append(str(thread_id))

    total_questions = len(qa_items)
    total_users = len(user_questions)
    returning_users = {
        uid: questions for uid, questions in user_questions.items()
        if len(questions) > 1
    }

    # Build a sorted list of returning users by question count (desc)
    returning_user_list = sorted(
        [
            {"user_id": uid, "questions": len(questions)}
            for uid, questions in returning_users.items()
        ],
        key=lambda x: x["questions"],
        reverse=True,
    )

    # Questions-per-user distribution
    question_counts = Counter(len(qs) for qs in user_questions.values())
    distribution = [
        {"questions": k, "users": v}
        for k, v in sorted(question_counts.items())
    ]

    return {
        "total_questions": total_questions,
        "total_identified_users": total_users,
        "returning_user_count": len(returning_users),
        "returning_users": returning_user_list[:20],
        "question_distribution": distribution,
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

    # Fact-check recommendations against the live docs site.
    recommendations = final_analysis.get("sprint_recommendations", [])
    if recommendations:
        print(f"Fact-checking {len(recommendations)} recommendations against {site.name}")
        site_pages = fetch_site_search_pages(site.name, recommendations)
        if site_pages:
            final_analysis["sprint_recommendations"] = claude.fact_check_recommendations(
                recommendations, site_pages,
            )
            checked_count = sum(
                1 for r in final_analysis["sprint_recommendations"]
                if r.get("fact_check_status")
            )
            print(f"Fact-checked {checked_count}/{len(recommendations)} recommendations")

    # Group classified questions into a two-level hierarchy:
    #   category (matching docs.sui.io/develop) → topic (specific feature)
    classified = kapa_analysis.get("classified_questions", [])

    # Normalize topic labels: merge only true duplicates (casing variants).
    raw_topics = [c.get("topic") or c.get("theme", "Other") for c in classified]
    print(f"Normalizing {len(set(raw_topics))} unique topic labels")
    topic_mapping = ClaudePipeline.normalize_topics(raw_topics)

    for c in classified:
        raw = c.get("topic") or c.get("theme", "Other")
        c["topic"] = topic_mapping.get(raw, raw)

    # Group by thread_id to detect multi-question conversations
    thread_questions: dict[str, list[dict[str, Any]]] = {}
    for c in classified:
        tid = c.get("thread_id") or ""
        if tid:
            thread_questions.setdefault(tid, []).append(c)

    # Tag each conversation with how many questions are in its thread
    for c in classified:
        tid = c.get("thread_id") or ""
        c["thread_question_count"] = len(thread_questions.get(tid, [])) if tid else 1

    # Build category → topic → questions hierarchy
    from app.claude_pipeline import DOCS_CATEGORIES

    category_data: list[dict[str, Any]] = []
    # Group questions by category
    cat_groups: dict[str, list[dict[str, Any]]] = {}
    for c in classified:
        cat = c.get("category", "Other")
        cat_groups.setdefault(cat, []).append(c)

    # Process categories in the canonical order, then append any extras
    ordered_cats = [c for c in DOCS_CATEGORIES if c in cat_groups]
    for cat_name in ordered_cats:
        cat_qs = cat_groups[cat_name]
        # Group by topic within this category
        topic_groups: dict[str, list[dict[str, Any]]] = {}
        for q in cat_qs:
            topic_groups.setdefault(q["topic"], []).append(q)
        # Sort topics by count descending
        topics_sorted = sorted(topic_groups.items(), key=lambda kv: len(kv[1]), reverse=True)

        topic_list = []
        for topic_name, tqs in topics_sorted:
            certain = sum(1 for q in tqs if q.get("confidence") == "certain")
            uncertain = len(tqs) - certain
            topic_list.append({
                "name": topic_name,
                "evidence_count": len(tqs),
                "certain_count": certain,
                "uncertain_count": uncertain,
                "questions": tqs,
            })

        cat_certain = sum(t["certain_count"] for t in topic_list)
        cat_uncertain = sum(t["uncertain_count"] for t in topic_list)
        category_data.append({
            "name": cat_name,
            "evidence_count": len(cat_qs),
            "certain_count": cat_certain,
            "uncertain_count": cat_uncertain,
            "topics": topic_list,
        })

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
            "category_data": category_data,
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

    # Collect previous runs for the archive index.
    past_runs: list[dict[str, Any]] = []
    for run_dir in sorted(settings.site_output_dir.iterdir(), reverse=True):
        if not run_dir.is_dir() or run_dir.name in ("latest",):
            continue
        site_htmls = sorted(f.name for f in run_dir.glob("*.html") if f.name != "index.html")
        if not site_htmls:
            continue
        past_runs.append({
            "run_id": run_dir.name,
            "reports": [{"filename": f, "slug": f.removesuffix(".html")} for f in site_htmls],
        })

    index_context = {
        "run_id": run_id,
        "start_date": start_date,
        "end_date": end_date,
        "site_reports": site_reports,
        "past_runs": past_runs,
    }

    index_html = render_html(Path("templates"), "index.html.j2", index_context)

    (reports_dir / "index.html").write_text(index_html, encoding="utf-8")
    (settings.site_output_dir / "index.html").write_text(index_html, encoding="utf-8")

    write_latest_reports(reports_dir, latest_dir)
    (latest_dir / "index.html").write_text(index_html, encoding="utf-8")

    # Send a separate Slack notification to each site's webhook.
    if os.getenv("SKIP_SLACK", "").lower() in ("1", "true", "yes"):
        print("SKIP_SLACK is set — skipping Slack notifications")
    else:
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
