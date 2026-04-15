# Weekly Plausible + Kapa + Claude Report to Slack

This repo runs a fully automated weekly pipeline for one or more docs sites:

1. Fetch raw JSON from Plausible
2. Fetch raw JSON from Kapa
3. Save raw snapshots locally (or in artifact storage)
4. Count tokens and chunk oversized payloads for Claude
5. Analyze Plausible raw payloads and Kapa raw payloads with Claude
6. Run a final Claude synthesis pass per site
7. Render a full HTML report per site plus an index page
8. Post a Slack summary with a link to each full report

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m app.main
```

## Required environment variables

- `PLAUSIBLE_API_KEY`
- `KAPA_API_KEY`
- `ANTHROPIC_API_KEY`
- `SLACK_WEBHOOK_URL`
- `SITES_JSON` JSON array of site configs

Optional:

- `REPORT_BASE_URL` public base URL for hosted reports, for example a GitHub Pages or S3 static site URL
- `RAW_OUTPUT_DIR` default: `./artifacts/raw`
- `REPORT_OUTPUT_DIR` default: `./artifacts/reports`
- `SITE_OUTPUT_DIR` default: `./site`
- `CLAUDE_MODEL` default: `claude-sonnet-4-6`
- `CLAUDE_MAX_INPUT_TOKENS` default: `120000`
- `REPORT_DAYS` default: `7`

## Example SITES_JSON

```json
[
  {
    "name": "docs.sui.io",
    "plausible_site_id": "docs.sui.io",
    "kapa_project_id": "docs"
  },
  {
    "name": "docs.wal.app",
    "plausible_site_id": "docs.wal.app",
    "kapa_project_id": "docs"
  }
]
```

If `docs.sui.io` and `docs.wal.app` use different Kapa projects, set a different `kapa_project_id` for each site.

## Hosting the full reports

The Slack message links to the full HTML reports using `REPORT_BASE_URL`.

Typical options:
- GitHub Pages
- S3 static website hosting
- Cloudflare R2 + public bucket
- any internal static host

The pipeline writes site HTML to `./site/<run_id>/`.
Host that folder and set `REPORT_BASE_URL` to the published base URL.

## Notes

- Plausible raw data is fetched via the Stats API `/api/v2/query`.
- Kapa raw data is fetched via the public API with `X-API-KEY` auth and `project_id`.
- Claude raw analysis is split into: Plausible pass, Kapa chunk passes, final synthesis pass.
- Slack delivery uses an incoming webhook.

Adjust Kapa endpoint paths in `app/kapa.py` if your project uses different response shapes or endpoint names.
# docs-data-report-slack
