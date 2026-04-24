# fred-macro

Personal system for daily macro education. Ingests Federal Reserve economic data (FRED), builds a structured knowledge base, runs a research phase to detect empirically interesting findings, and sends a short daily lesson by email.

Private project.

## Architecture

```
FRED API (8600+ series)
    └── src/ingest/          # discovery, incremental updates, release calendar
         └── data/           # parquet, partitioned by frequency (daily/weekly/monthly/quarterly/annual)

data/ + knowledge/
    └── src/analytics/       # shared helpers: data loading, stats, charts, episodes, FOMC events
    └── src/research/        # finding detection, relationship monitoring, web enrichment
         └── knowledge/findings.md   # findings backlog

findings.md + data/ + knowledge/concepts.md
    └── src/select/          # lesson selection (upcoming releases, surprising findings, continuity)
    └── src/compose/         # email drafting, fact-check pass, chart generation
    └── src/deliver/         # SES delivery
```

**Data layer**: long-format parquet files partitioned by update frequency. Incremental updates with a revision buffer (90 days daily/weekly, 2 years monthly+). Cross-series analysis within a frequency is a single file read.

**Research phase**: rolling correlations, z-scores, structural break detection, regime identification, FOMC event studies, recession logit, jump detection, bond predictability. Findings written to `knowledge/findings.md` with full audit trail (series IDs, windows, computed statistics).

**Composition**: one organizing idea per email, 400-600 words. All numbers queried from parquet — the LLM writes prose around data, not the other way around. Separate fact-check pass before send.

**Analytics toolkit** (`src/analytics/`): `data.py`, `stats.py`, `charts.py`, `episodes.py`, `fomc.py`, `bonds.py`, `recession.py`, `indicators.py`, `format.py`. Shared across research, composition, and ad-hoc analysis.

## Data sources

- **FRED** (primary): ~8600 series tracked. Universe defined by policy (top-N per category + major Fed releases), not a hand-picked list. Full history, no lookback restriction.
- **Fed releases tracked**: H.4.1, H.6, H.8, H.15, Employment Situation, CPI, PCE, GDP, Industrial Production, Retail Sales, and others via FRED release calendar.
- **Knowledge sources**: FEDS Notes, Liberty Street Economics, SF Fed Economic Letter, BLS Handbook of Methods, BEA NIPA Methods, BIS Quarterly Review, NBER, FOMC minutes, Jackson Hole proceedings. See `knowledge/sources.md`.
- **Paper library**: 92 curated papers across inflation, yield curve, credit, FOMC event studies, nowcasting, risk appetite, FX, commodities, macro×equity, SOFR plumbing.

## Cron schedule (EC2, UTC)

| Time | Job |
|------|-----|
| 05:45 weekdays | `send_daily.py` |
| 06:00 daily | `refresh_data.py` |
| 14:30 weekdays | `refresh_data.py --discover` |
| 22:00 daily | `run_research.py` |
| 23:00 weekdays | `run_harvest.py --max 5` |

## Setup

```bash
conda activate fred-macro
cp .env.example .env  # fill in email config
```

Credentials expected at:
- `~/keys/fred/key.txt` — FRED API key
- `~/keys/aws/ses-credentials.json` — SES SMTP credentials

## Entry points

- `scripts/refresh_data.py` — pull/refresh series data
- `scripts/run_research.py` — recompute findings (`--fomc`, `--fomc-only`, `--relationships-only`)
- `scripts/run_harvest.py` — web-enrich findings backlog
- `scripts/send_daily.py` — compose and send (`--dry-run`, `--force-finding <slug>`)
- `scripts/status.py` — system health dashboard
