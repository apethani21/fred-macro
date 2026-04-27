# fred-macro

Personal system for daily macro education. Ingests Federal Reserve and ECB economic data, builds a structured knowledge base, runs a nightly research phase to detect empirically interesting findings, and sends a short daily lesson by email.

Private project.

## Architecture

```
FRED API (8600+ series) + ECB SDW API
    └── src/ingest/          # discovery, incremental updates, release calendar
         └── data/           # parquet, partitioned by frequency (daily/weekly/monthly/quarterly/annual)

data/ + knowledge/
    └── src/analytics/       # shared helpers: data loading, stats, charts, episodes, FOMC events, ECB
    └── src/research/        # finding detection, relationship monitoring, web enrichment
         └── knowledge/findings.md   # findings backlog
         └── state/topic_seeds.jsonl # nightly-written topic seeds (statistical backbone, 7-day TTL)

topic_seeds.jsonl + findings.md + data/ + knowledge/concepts.md
    └── src/select/          # lesson selection: seeds preferred, findings fallback
    └── src/compose/         # email drafting, fact-check pass, chart generation
    └── src/deliver/         # SES delivery
```

**Data layer**: long-format parquet files partitioned by update frequency. Incremental updates with a revision buffer (90 days daily/weekly, 2 years monthly+). ECB series stored alongside FRED in the same partition files, prefixed `ECB.*`.

**Research phase**: rolling correlations, z-scores, structural break detection (Bai-Perron), regime identification, FOMC event studies, recession logit, jump detection (BNS bipower), bond predictability (CP factor), BTP-Bund fragmentation monitoring, named relationship monitoring. Findings written to `knowledge/findings.md` with full audit trail.

**Topic-seed composition**: nightly scan writes lightweight `TopicSeed` objects (raw stats, no prose) to `state/topic_seeds.jsonl`. Morning send picks the best seed and Claude decides the organizing idea and hook at compose time using fresh parquet data — separating detection (nightly) from interpretation (morning).

**Composition**: one organizing idea per email, 400–600 words. All numbers queried from parquet — the LLM writes prose around data, not the other way around. Three-pass pipeline before send: (1) fast fact-check for number accuracy and style, (2) agentic web-search citation pass that identifies mechanistic claims, finds primary academic sources, and adds inline `[N]` superscripts with a References section, (3) revision incorporating citations. BM25 concept retrieval from `knowledge/concepts.md` for institutional context.

**Analytics toolkit** (`src/analytics/`): `data.py`, `stats.py`, `charts.py`, `episodes.py`, `fomc.py`, `bonds.py`, `recession.py`, `indicators.py`, `ecb.py`, `format.py`. Shared across research, composition, and ad-hoc analysis.

## Data sources

- **FRED** (primary): ~8600 series tracked. Universe defined by policy (top-N per category + major Fed releases), not a hand-picked list. Full history, no lookback restriction.
- **ECB SDW**: 12 series — DFR, €STR, EA HICP (total + core), AAA yield curve (2Y/10Y), Germany 10Y, Italy 10Y (BTP), EA M3, negotiated wages. Two derived: BTP-Bund spread, Bund slope.
- **Fed releases tracked**: H.4.1, H.6, H.8, H.15, Employment Situation, CPI, PCE, GDP, Industrial Production, Retail Sales. ECB Governing Council meetings and Eurostat HICP flash dates also in release calendar.
- **Knowledge sources**: FEDS Notes, Liberty Street Economics, SF Fed Economic Letter, BLS Handbook of Methods, BEA NIPA Methods, BIS Quarterly Review, NBER, FOMC minutes, Jackson Hole proceedings. See `knowledge/sources.md`.
- **Paper library**: 132 curated papers across inflation, yield curve, credit/FI, FOMC event studies, nowcasting, risk appetite, FX, commodities, macro×equity, SOFR plumbing, European macro/ECB, systematic fixed income, cross-asset factor pricing (Groups A–N).

## Cron schedule (EC2, UTC)

| Time | Job |
|------|-----|
| 05:45 daily | `send_daily.py` |
| 06:00 daily | `refresh_data.py --ecb` |
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
- `~/keys/anthropic/key.txt` — Anthropic API key
- `~/keys/aws/ses-credentials.json` — SES SMTP credentials
- `~/keys/telegram/bot_token.txt` — Telegram bot token (from @BotFather)

## Entry points

- `scripts/refresh_data.py` — pull/refresh series data (`--ecb` includes ECB SDW series)
- `scripts/run_research.py` — recompute findings (`--fomc`, `--fomc-only`, `--relationships-only`, `--questions-only`)
- `scripts/run_harvest.py` — web-enrich findings backlog
- `scripts/send_daily.py` — compose and send (`--dry-run`, `--force-finding <slug>`, `--force-seed <id>`)
- `scripts/status.py` — system health dashboard
- `scripts/smoke_analytics.py` — analytics smoke test
- `scripts/telegram_bot.py` — Telegram Q&A bot (`--get-chat-id` for first-time setup); runs as a persistent daemon on EC2
