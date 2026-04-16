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

