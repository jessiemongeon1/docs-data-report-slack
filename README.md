# Weekly Plausible + Kapa + Claude Report to Slack

<img src="icon.jpeg" alt="Weekly Analytics Report" />

This repo runs a fully automated weekly data analysis report for [docs.sui.io](https://docs.sui.io) and [docs.wal.app](https://docs.wal.app).

## Pipeline

1. Fetch raw JSON from Plausible
2. Fetch raw JSON from Kapa
3. Save raw snapshots locally (or in artifact storage)
4. Count tokens and chunk oversized payloads for Claude
5. Analyze Plausible and Kapa raw payloads with Claude
6. Run a final Claude synthesis pass per site
7. Render a full HTML report per site plus an index page
8. Post a Slack summary with a link to each full report
